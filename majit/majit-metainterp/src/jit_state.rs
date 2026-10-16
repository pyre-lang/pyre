//! No direct RPython equivalent — trait abstraction for interpreter state
//! (RPython uses concrete MetaInterp class in pyjitpl.py).

use majit_ir::{GcRef, OpRef, Type, Value};

use crate::blackhole::ExceptionState;
use crate::jitdriver::JitDriverStaticData;
use crate::resume::{
    MaterializedValue, MaterializedVirtual, ReconstructedState, ResolvedPendingFieldWrite,
    ResumeFrameLayoutSummary,
};
use crate::virtualizable::VirtualizableInfo;

/// Layout description for replaying a pending field or array write during deopt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendingFieldWriteLayout {
    Field {
        offset: usize,
        value_type: Type,
    },
    ArrayItem {
        base_offset: usize,
        item_size: usize,
        item_type: Type,
    },
}

/// Session-scoped materialization cache that persists across guard failure
/// resumption within the same deopt session.
///
/// When GUARD_NOT_FORCED triggers resumption after CALL_MAY_FORCE, the same
/// virtuals may need to be materialized again. This cache ensures each virtual
/// is materialized at most once per deopt session and reused throughout.
#[derive(Debug, Clone, Default)]
pub struct DeoptMaterializationCache {
    /// Maps virtual indices to materialized GcRefs. `None` means not yet
    /// materialized; `Some(gcref)` means already materialized in this session.
    pub refs: Vec<Option<GcRef>>,
}

impl DeoptMaterializationCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self { refs: Vec::new() }
    }

    /// Create a cache pre-sized for the given number of virtuals.
    pub fn with_capacity(num_virtuals: usize) -> Self {
        Self {
            refs: vec![None; num_virtuals],
        }
    }

    /// Return the number of cached entries (including None slots).
    pub fn len(&self) -> usize {
        self.refs.len()
    }

    /// Return true if no slots have been allocated.
    pub fn is_empty(&self) -> bool {
        self.refs.is_empty()
    }

    /// Return the number of successfully materialized entries.
    pub fn materialized_count(&self) -> usize {
        self.refs.iter().filter(|r| r.is_some()).count()
    }

    /// Clear all cached entries, ending the deopt session.
    pub fn clear(&mut self) {
        self.refs.clear();
    }

    /// Look up a cached materialization by virtual index.
    pub fn get(&self, virtual_index: usize) -> Option<GcRef> {
        self.refs.get(virtual_index).copied().flatten()
    }

    /// Ensure the cache has room for at least `count` virtual indices.
    pub fn ensure_capacity(&mut self, count: usize) {
        if self.refs.len() < count {
            self.refs.resize(count, None);
        }
    }

    /// Insert a materialized ref at the given virtual index.
    pub fn insert(&mut self, virtual_index: usize, gc_ref: GcRef) {
        self.ensure_capacity(virtual_index + 1);
        self.refs[virtual_index] = Some(gc_ref);
    }
}

/// resume.py rebuild_from_resumedata return value.
///
/// RPython returns `(liveboxes, virtualizable_boxes, virtualref_boxes)`.
/// `liveboxes[i]` = InputArgBox for TAGBOX(i), None otherwise.
/// Frame registers are filled with InputArgBox/ConstBox/VirtualBox mix.
///
/// Rust equivalent: `RebuiltFrame.values` carries per-slot RebuiltValue
/// (Box/Const/Int/Virtual/Unassigned). `constants` holds constant OpRef
/// entries for the optimizer constant pool (no RPython equivalent —
/// RPython uses ConstBox objects natively in the trace recording).
///
/// resume.py `decode_box(num, kind)` parity: each
/// `RebuiltValue::Box(idx, kind)` carries its own kind, sourced from the
/// parent guard's `fail_arg_types` at `decode_tagged` time. Consumers
/// read the kind directly from the variant — no parallel
/// `fail_arg_types` field on this struct is needed.
#[derive(Debug, Clone)]
pub struct ResumeDataResult {
    /// resume.py:1057: per-frame decoded values from rd_numb.
    /// Each RebuiltValue::Box(i, kind) → `liveboxes[i]` in RPython.
    pub frames: Vec<majit_ir::resumedata::RebuiltFrame>,
    /// `rebuild_from_resumedata` virtualizable boxes (decoded from the vable
    /// section). Each `RebuiltValue` is a box (`Box` / `Const` / `Virtual`).
    pub virtualizable_boxes: Vec<majit_ir::resumedata::RebuiltValue>,
    /// resume.py:1045: virtualref box pairs (decoded from vref section).
    pub virtualref_values: Vec<majit_ir::resumedata::RebuiltValue>,
    /// compile.py `ResumeGuardDescr` storage handle — the shared
    /// `Arc<ResumeStorage>` owned by the guard whose rd_numb we just
    /// decoded. Virtual materialization (resume.py:1251
    /// `decode_box` → `self.consts[num-TAG_CONST_OFFSET]`) reads
    /// `storage.rd_consts()` directly; no owned copy lives on this
    /// struct.
    pub storage: Option<std::sync::Arc<crate::resume::ResumeStorage>>,
    /// resume.py rebuild_from_resumedata num_failargs read from rd_numb header.
    /// Needed by `materialize_bridge_virtual` to translate negative TAGBOX
    /// numbers in `rd_virtuals` fieldnums (resume.py decode_box:
    /// Python-style negative indexing into the parent guard's liveboxes
    /// array; the decoder normalizes via `val + num_failargs`).
    pub num_failargs: i32,
    /// compile.py:797 ResumeGuardDescr.fail_arg_types — the type tag of
    /// each parent-guard failarg, in order.
    ///
    /// resume.py decode_box reads `self.liveboxes[num]` and
    /// returns the actual Box object whose `.type` is intrinsic
    /// (history.py:220). For the bridge tracer, those liveboxes ARE the
    /// bridge's `InputArg{Int,Ref,Float}` slots — pyre mints them via
    /// `OpRef::input_arg_typed(idx, fail_arg_types[idx])`.
    /// `materialize_bridge_virtual::decode_fieldnum` indexes into this
    /// vector so the recovered TAGBOX OpRef matches the typed variant
    /// the optimizer / heap-cache later see.
    pub fail_arg_types: Vec<Type>,
}

/// resume.py `decode_box(num, kind)` parity for a generic
/// (non-virtualizable) JitDriver sym. Returns the bridge `InputArg` OpRef
/// for failarg `n` paired with its concrete deadframe shadow
/// `fail_values[n]`. Both the OpRef and the shadow key off the Box's own
/// failarg index `n` (authoritative), so they stay correct regardless of
/// how `n` maps to a sym slot. Greens / `Const` / `Virtual` reds are
/// caller-skipped.
pub fn bridge_decode_red(
    n: usize,
    kind: Type,
    fail_values: &[i64],
    fail_types: &[Type],
) -> (OpRef, i64) {
    debug_assert_eq!(
        fail_types.get(n).copied().unwrap_or(kind),
        kind,
        "bridge_decode_red: Box({n}) failarg type != decoded kind"
    );
    (OpRef::input_arg_typed(n as u32, kind), fail_values[n])
}

/// One live jitcode register as a guard's resume data described it.
///
/// `pyjitpl.py rebuild_state_after_failure` rebuilds a `MIFrame` per encoded
/// section and `resume.py ResumeDataBoxReader.consume_boxes` fills every live
/// register of each from the guard's numbering.  This is one such register:
/// `bank` says which of the three banks `index` addresses, `opref` is the
/// symbolic value the trace continues with — a bridge `InputArg` for a
/// failarg, a pool constant for one the guard already carried folded — and
/// `value` is its concrete shadow, which the walk needs because the walk is
/// what executes.
#[derive(Debug, Clone, Copy)]
pub struct GuardResumeReg {
    pub bank: Type,
    pub index: u32,
    pub opref: OpRef,
    pub value: i64,
}

/// One rebuilt interpreter frame of a guard's resume stack.
///
/// `resume.py rebuild_from_resumedata` reads one `(jitcode_pos, pc)` section at
/// a time and pushes a real `MIFrame` for each — `metainterp.newframe(jitcode)`,
/// then `f.setup_resume_at_op(pc)`, then `consume_boxes` into that frame's own
/// registers.  The sections arrive caller-first, because the writer put them
/// that way (`opencoder.py SnapshotIterator.__init__`), so appending them in
/// stream order leaves the framestack outermost-at-the-bottom.  This is one
/// such section, and a slice of them is in that order: outermost caller first,
/// the frame the guard failed in last.
///
/// `result_slot` is the piece the resume stream does not carry.  A frame that
/// is not the innermost is suspended in an inline call, and the register its
/// callee's result lands in was deliberately left out of its liveness
/// (`opencoder.py _ensure_parent_resumedata` blanks it before reading).
/// Upstream never restores it: `pyjitpl.py MIFrame.make_result_of_lastop`
/// reads it off the bytecode at the moment the callee returns, so a rebuilt
/// frame needs nothing.  majit's return path takes it from the frame instead
/// (`MIFrame::return_i` / `_r` / `_f`, wired by `BC_INLINE_CALL`), so the same
/// read has to happen here, at rebuild time — that is
/// `JitCode::inline_call_ending_at`.  `None` on any frame whose callee returns
/// nothing, and on the innermost frame, which is suspended in a guard rather
/// than in a call.
#[derive(Debug, Clone)]
pub struct GuardResumeFrame {
    pub jitcode: std::sync::Arc<crate::jitcode::JitCode>,
    pub pc: usize,
    pub regs: Vec<GuardResumeReg>,
    pub result_slot: Option<(crate::jitcode::JitArgKind, usize)>,
    /// The caller `descrs` slot naming this frame's jitcode, as the
    /// `BC_INLINE_CALL` that pushed it named it.  `None` for the root frame,
    /// which no inline call pushed.
    pub sub_idx: Option<usize>,
}

/// Interpreter-specific JIT state contract.
/// `rebuild_from_resumedata`'s `virtualizable_boxes, virtualref_boxes`:
/// what `consume_vref_and_vable_boxes` decoded ahead of the frames, each box
/// with the concrete the reader stamped on it.
///
/// The `(OpRef, Value)` copies are not roots. A later `bridge_decode_box`
/// `NEW` / Trace-pool append, or `rebuild_portal_framestack_from_resumedata`,
/// can collect before `setup_bridge_sym` stores them. `vable_ref_roots` holds
/// each Ref as a translated livevar (`OwnerRootGuard`) so the collector
/// forwards it in place, the way `consume_boxes` keeps the decoded GCREF in
/// a traced local across the setattr.
#[derive(Default)]
pub struct VrefVableBoxes {
    /// `consume_virtualizable_boxes`: `[vable, static fields..., array items...]`.
    pub virtualizable_boxes: Vec<(OpRef, Value)>,
    /// `consume_virtualref_boxes`: `(virtual, vref)` pairs, flattened.
    pub virtualref_boxes: Vec<(OpRef, Value)>,
    /// One slot per `virtualizable_boxes` entry; `None` for a non-Ref.
    vable_ref_roots: Vec<Option<majit_gc::shadow_stack::OwnerRootGuard>>,
}

impl std::fmt::Debug for VrefVableBoxes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VrefVableBoxes")
            .field("virtualizable_boxes", &self.virtualizable_boxes)
            .field("virtualref_boxes", &self.virtualref_boxes)
            .field("vable_ref_roots", &self.vable_ref_roots.len())
            .finish()
    }
}

impl VrefVableBoxes {
    fn pin_ref_value(val: Value) -> Option<majit_gc::shadow_stack::OwnerRootGuard> {
        match val {
            Value::Ref(gc) if !gc.is_null() && gc != GcRef::NO_CONCRETE => {
                // `pin_root` / `normalize_published_slot`: a decode that
                // collected may hand back a word that already carries a
                // forwarding stub. Publish the to-space address; the
                // guard then tracks later moves.
                let live = majit_gc::gc_current_object_address(gc.0);
                if live == 0 {
                    return None;
                }
                Some(majit_gc::shadow_stack::OwnerRootGuard::new(
                    majit_ir::GcRef(live),
                ))
            }
            _ => None,
        }
    }

    /// Pin every `Box` GCREF from the deadframe before the first decode can
    /// collect. Index-aligned with `virtualizable_boxes`.
    pub fn pre_pin_vable_boxes(
        &mut self,
        vvals: &[majit_ir::resumedata::RebuiltValue],
        fail_values: &[i64],
    ) {
        self.vable_ref_roots.clear();
        self.vable_ref_roots.reserve(vvals.len());
        for v in vvals {
            let guard = match v {
                majit_ir::resumedata::RebuiltValue::Box(n, Type::Ref) => fail_values
                    .get(*n)
                    .copied()
                    .map(|bits| Value::Ref(GcRef(bits as usize)))
                    .and_then(Self::pin_ref_value),
                _ => None,
            };
            self.vable_ref_roots.push(guard);
        }
    }

    /// Reload a pre-pinned Box, or pin a Virtual/Const just decoded.
    pub fn reload_or_pin_vable(&mut self, idx: usize, val: Value) -> Value {
        if let Some(g) = self.vable_ref_roots.get(idx).and_then(|slot| slot.as_ref()) {
            return Value::Ref(g.get());
        }
        if let Some(g) = Self::pin_ref_value(val) {
            let live = Value::Ref(g.get());
            if idx < self.vable_ref_roots.len() {
                self.vable_ref_roots[idx] = Some(g);
            }
            return live;
        }
        val
    }

    /// Live GCREF at store time (`setup_bridge_sym`). The guard is a root
    /// across `rebuild_portal_framestack_from_resumedata`.
    pub fn live_vable(&self, idx: usize, fallback: Value) -> Value {
        let live = self
            .vable_ref_roots
            .get(idx)
            .and_then(|slot| slot.as_ref())
            .map(|g| Value::Ref(g.get()))
            .unwrap_or(fallback);
        live
    }
}

pub trait JitState: Sized {
    type Meta: Clone;
    type Sym;
    type Env: ?Sized;

    /// Portal return kind recorded by `#[jit_interp]` from `finish_return_for`.
    ///
    /// `None` means the portal declared no `FinishReturn`. `JitDriver::new`
    /// leaves `result_type` at `Type::Ref` (`warmspot.py` `getkind` default
    /// before a portal signature is known). `Some` is applied once there.
    const PORTAL_RESULT_TYPE: Option<Type> = None;

    /// Whether this frontend's own guard-failure recovery already filled the
    /// virtualizable from the resume stream.
    ///
    /// `pyjitpl.py rebuild_state_after_failure` closes by assigning
    /// `self.virtualizable_boxes` and then calling
    /// `self.synchronize_virtualizable()` — the boxes are the trace's copy and
    /// the object still holds whatever compiled code last spilled, so the
    /// second half is what makes a walk resumed at the guard read the right
    /// fields. Bridge setup here performs it for a frontend that answers
    /// `false`.
    ///
    /// A frontend answers `true` when its own recovery already wrote those
    /// fields, which it may have to: the write this crate can make converts a
    /// shadow value with `value_to_raw_bits`, and a frontend whose array slots
    /// hold boxed objects needs a field-aware conversion that only it can
    /// spell. Writing twice is not the hazard — writing raw bits into a slot
    /// that wanted a box is.
    const SYNCHRONIZES_VIRTUALIZABLE_AFTER_GUARD_FAILURE: bool = false;

    fn can_trace(&self) -> bool {
        true
    }

    fn build_meta(&self, header_pc: usize, env: &Self::Env) -> Self::Meta;

    fn extract_live(&self, meta: &Self::Meta) -> Vec<i64>;

    fn extract_live_values(&self, meta: &Self::Meta) -> Vec<Value> {
        let raw = self.extract_live(meta);
        let types = self.live_value_types(meta);
        raw.into_iter()
            .zip(types)
            .map(|(v, t)| match t {
                Type::Float => Value::Float(f64::from_bits(v as u64)),
                Type::Ref => Value::Ref(GcRef(v as usize)),
                _ => Value::Int(v),
            })
            .collect()
    }

    fn live_value_types(&self, meta: &Self::Meta) -> Vec<Type> {
        self.extract_live(meta).iter().map(|_| Type::Int).collect()
    }

    /// Append this state's live values to `out` instead of returning a fresh
    /// `Vec`.
    ///
    /// `warmstate.py maybe_compile_and_run` builds the assembler's
    /// arguments by unspecializing reds that are already unboxed locals, so
    /// the upstream entry path allocates nothing to describe them. The
    /// `Vec`-returning form above cannot express that: the caller entering
    /// compiled code has a buffer whose shape the compiled artifact fixed, and
    /// a return-by-value forces a new allocation per entry regardless.
    ///
    /// `raw` and `types` are the caller's scratch for the two halves the pair
    /// is built from — cleared on arrival, unread on return, and reused across
    /// entries. An implementation that needs neither may ignore them.
    ///
    /// The default defers to [`Self::extract_live_values`], so a state that
    /// overrides neither — or that overrides only the `Vec`-returning form —
    /// keeps exactly today's values and today's allocations. Overriding this
    /// method is what removes them.
    fn extract_live_values_into(
        &self,
        meta: &Self::Meta,
        out: &mut Vec<Value>,
        raw: &mut Vec<i64>,
        types: &mut Vec<Type>,
    ) {
        let _ = (raw, types);
        out.extend(self.extract_live_values(meta));
    }

    /// Fill compiled-entry reds without `compiled_loops`.
    ///
    /// `execute_assembler` receives reds the caller already unboxed. It does
    /// not look up a side table of compiled metadata. Returns `true` only
    /// when `is_compatible` does not read meta and the reds do not either,
    /// so the entry can skip `IndexMap::get` of `CompiledEntry`. The default
    /// is the meta path.
    fn fill_entry_reds_without_meta(&self, out: &mut Vec<Value>) -> bool {
        let _ = out;
        false
    }

    /// `warmstate.py maybe_compile_and_run`: the reds as `unspecialize_value`
    /// words, in inputarg order. `false` when the state cannot produce them
    /// without the compiled meta (same condition as
    /// `fill_entry_reds_without_meta`).
    fn fill_entry_raw_reds(&self, out: &mut Vec<i64>) -> bool {
        let _ = out;
        false
    }

    /// Append-to-buffer form of [`Self::extract_live`], for
    /// [`Self::extract_live_values_into`] implementations.
    fn extract_live_into(&self, meta: &Self::Meta, out: &mut Vec<i64>) {
        out.extend(self.extract_live(meta));
    }

    /// Append-to-buffer form of [`Self::live_value_types`], for
    /// [`Self::extract_live_values_into`] implementations.
    fn live_value_types_into(&self, meta: &Self::Meta, out: &mut Vec<Type>) {
        out.extend(self.live_value_types(meta));
    }

    fn create_sym(meta: &Self::Meta, header_pc: usize) -> Self::Sym;

    /// Seed tracing-time concrete mirrors in the symbolic state.
    ///
    /// Interpreters that lower state reads through jitcode need this so
    /// branch decisions during tracing use the real runtime values.
    fn initialize_sym(&self, _sym: &mut Self::Sym, _meta: &Self::Meta) {}

    /// Drop every symbolic inputarg binding `create_sym` minted, leaving the
    /// concrete value mirrors `initialize_sym` seeded intact.
    ///
    /// `create_sym` numbers each state field `InputArg{Ty}(offset)` by walking
    /// the declaration order, and `record_input_arg` numbers the trace's own
    /// inputargs from its `op_count`. Both start at 0 and both issue into the
    /// one flat position space `OpRef::raw()` addresses, so the two agree only
    /// while they enumerate the same sequence. On a loop trace they do:
    /// `extract_live` / `live_value_types` emit the same partition in the same
    /// order that `create_sym` advances its offset, so field `k` and inputarg
    /// `k` name the same value.
    ///
    /// A bridge breaks that. `Trace::with_input_types(fail_descr.fail_arg_types())`
    /// numbers the bridge's inputargs in the guard's failarg order, which has no
    /// relation to declaration order, while `create_sym` runs unchanged and
    /// mints the same 0-based positions it would for a loop. Each surviving mint
    /// is then a well-formed position in a space it was not issued from:
    /// `TraceIterator`'s `_cache` is indexed by `raw()` alone, so the lookup
    /// resolves against whatever the bridge put at that position and hands back
    /// a box for an unrelated value instead of missing.
    ///
    /// `setup_bridge_sym` is the bridge's binding authority — it decodes each
    /// field from the resume data and rebinds it into the failarg space. It
    /// binds scalars only, and skips any field the resume frame does not carry,
    /// so calling it does not by itself retire the stale mints. Clearing them
    /// first makes an unbound field read as absent rather than as a live
    /// reference to someone else's slot; `resume.py` has no counterpart to
    /// preserve, because upstream a field the resume data did not resurrect
    /// simply has no box.
    ///
    /// The default is a no-op: a `Sym` that holds no `OpRef` has nothing to
    /// clear, and the unit-`Sym` test drivers are in that class.
    fn clear_sym_inputarg_bindings(_sym: &mut Self::Sym) {}

    /// How many of `sym`'s `OpRef` fields are still bound, or `None` if this
    /// state cannot say.
    ///
    /// This exists to grade the *call* to `clear_sym_inputarg_bindings`, not
    /// the function. Nothing downstream can: `setup_bridge_sym` rebinds a
    /// field to the same value whether the previous one was `OpRef::NONE` or
    /// a stale loop-shaped mint, and an overwrite that ignores its target
    /// erases the difference. The only place the two states are
    /// distinguishable is the window between the two calls, so the check has
    /// to be sited there.
    ///
    /// `None` rather than `0` for the default deliberately. A state that
    /// cannot report has to be skipped, not read as clean — zero bound and
    /// no way to count spell the same number, and a default of `0` would
    /// make every state that never overrides this pass the assertion in
    /// silence.
    fn count_bound_sym_inputargs(_sym: &Self::Sym) -> Option<usize> {
        None
    }

    /// pyjitpl.py `_unpack_boxes` parity: read concrete values
    /// from the live boxes at a successful close-loop back-edge before the
    /// trace history is cleared.
    fn close_loop_live_values(
        ctx: &crate::TraceCtx,
        _sym: &Self::Sym,
        _meta: &Self::Meta,
        live_arg_boxes: &[OpRef],
    ) -> Option<Vec<Value>> {
        live_arg_boxes
            .iter()
            .copied()
            .map(|box_| ctx.box_value(box_))
            .collect()
    }

    fn driver_descriptor(&self, _meta: &Self::Meta) -> Option<JitDriverStaticData> {
        None
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool;

    /// pyjitpl.py get_procedure_token(greenboxes): compute the
    /// green key for a given loop header PC. Used by bridge trace close
    /// to find compiled targets at the destination loop header.
    fn green_key_for_pc(&self, _pc: usize) -> Option<u64> {
        None
    }

    /// The code object pointer for green key computation.
    /// RPython: `jitdriver_sd.jitcodes[jitcode_pos]`
    fn code_ptr(&self) -> usize {
        0
    }

    /// compile.py:269-270 parity: update meta when the trace is cut at
    /// a cross-loop merge point. `header_pc` is the bytecode PC of the
    /// new loop header; `original_box_types` are the types of the live
    /// boxes at the cut point (the new inputargs).
    fn update_meta_for_cut(
        _meta: &mut Self::Meta,
        _header_pc: usize,
        _original_box_types: &[Type],
    ) {
    }

    /// `resume.py` `rebuild_from_resumedata`: `metainterp_sd.jitcodes[jitcode_pos]`
    /// names every frame's jitcode through one table. A frontend that keeps
    /// runtime bodies, or lazily materialized frozen bodies, outside
    /// `staticdata.jitcodes` resolves the absolute index here; `None` leaves
    /// the metainterp table as the only source.
    fn resolve_resume_jitcode(_index: usize) -> Option<std::sync::Arc<crate::jitcode::JitCode>> {
        None
    }

    /// resume.py rebuild_from_resumedata parity: set up bridge-specific symbolic local mapping.
    /// Called after rebuild_from_resumedata to map frame locals to bridge
    /// InputArg OpRefs. In RPython, MIFrame.registers are populated with
    /// InputArg/Const boxes from rebuild; this is the Rust equivalent.
    ///
    /// `ctx` is the bridge's active trace context. When `resume_data` carries
    /// `RebuiltValue::Virtual` entries, the implementation must emit
    /// `NEW_WITH_VTABLE`/`SETFIELD_GC` ops via `ctx` to materialize each
    /// virtual at trace start, mirroring RPython's
    /// `ResumeDataBoxReader.consume_boxes` → `rd_virtuals[i].allocate(...)`
    /// → `metainterp.execute_new_with_vtable` (resume.py getvirtual_ptr).
    /// `rd_virtuals` is the parent guard's virtual descriptor table from
    /// the source trace's exit layout (None when the parent had no virtuals).
    ///
    /// This default body is an intentional no-op — correct for primary
    /// traces and single-frame bridges, which carry no inlined-callee
    /// resume frames to reconstruct. `JitState` implementations that
    /// support multi-frame (inlined/recursive) bridges override it to
    /// decode the resume stream into per-callee recipes and stash a
    /// `BridgeInlineCarrier` on `ctx` (drained by `trace_bytecode` right
    /// before `interpret()`).
    ///
    /// `reader` is the one `ResumeDataBoxReader` state of this bridge entry,
    /// built in `start_bridge_tracing` and shared by `_prepare`,
    /// `consume_boxes` and `setup_bridge_sym`; its allocator, when it has
    /// one, is the applying half (`execute_and_record`) for an entry no
    /// direct reader preceded.
    /// The array-descr factory the bridge reader mints `VRawBuffer` virtuals
    /// with (`BridgeVirtualCache::mint_raw_array_descr`). Descr identity is
    /// the consumer's gccache concern, so a state that virtualizes raw
    /// buffers supplies its own keyed factory; the default mints a fresh
    /// descr.
    fn mint_bridge_array_descr(
        base_size: usize,
        item_size: usize,
        len_offset: Option<usize>,
        item_type: Type,
        signed: bool,
    ) -> majit_ir::DescrRef {
        crate::default_bridge_array_descr(base_size, item_size, len_offset, item_type, signed)
    }

    /// resume.py `AbstractResumeDataReader._prepare` (`_prepare_virtuals`
    /// then `_prepare_pendingfields`). Runs before `newframe` /
    /// `consume_boxes`, once per reader; `setup_bridge_sym` never applies
    /// the pending fields again. The default is a no-op like
    /// `setup_bridge_sym`'s; production fills virtuals so `getvirtual_ptr`
    /// during consume stores the same box.
    fn prepare_bridge_resume(
        _sym: &mut Self::Sym,
        _ctx: &mut crate::trace_ctx::TraceCtx,
        _resume_data: &ResumeDataResult,
        _rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        _fail_values: &[i64],
        _fail_types: &[Type],
        _reader: &mut crate::BridgeVirtualCache<'_>,
    ) {
    }

    /// resume.py `ResumeDataBoxReader.consume_vref_and_vable_boxes`:
    /// decode the virtualizable and virtualref boxes, in that order, before
    /// any frame's `consume_boxes` — so a virtual they name is allocated
    /// ahead of one a frame names. The boxes come back with the concrete
    /// each decode stamped, for the fill `setup_bridge_sym` does with them.
    fn consume_vref_and_vable_boxes(
        _sym: &mut Self::Sym,
        _ctx: &mut crate::trace_ctx::TraceCtx,
        _resume_data: &ResumeDataResult,
        _rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        _fail_values: &[i64],
        _fail_types: &[Type],
        _reader: &mut crate::BridgeVirtualCache<'_>,
    ) -> VrefVableBoxes {
        VrefVableBoxes::default()
    }

    /// pyjitpl.py `rebuild_state_after_failure` after
    /// `rebuild_from_resumedata` returned: the frontend's fill from the
    /// rebuilt frames and from `boxes`.
    #[allow(clippy::too_many_arguments)]
    fn setup_bridge_sym(
        _sym: &mut Self::Sym,
        _ctx: &mut crate::trace_ctx::TraceCtx,
        _resume_data: &ResumeDataResult,
        _rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        _fail_values: &[i64],
        _fail_types: &[Type],
        _reader: &mut crate::BridgeVirtualCache<'_>,
        _boxes: &VrefVableBoxes,
    ) {
    }

    /// The int-register range this state's symbol reserves for identity
    /// slots, when it reserves one.
    ///
    /// A non-root frame's snapshot has that range blanked to `Const(0)`
    /// placeholders (`get_list_of_active_snapshot_boxes`), and nothing
    /// re-derives it: `resume.py write_an_int` writes each section's decoded
    /// values, zeros included, into that frame's own registers, and the only
    /// root-to-callee propagation deopt performs is on the virtualizable
    /// FIELDS. A resume that spans more than one frame therefore cannot
    /// reconstruct those registers, and the root's are not a substitute —
    /// they hold a different quantity at the same index.
    ///
    /// So a bridge across inline frames is refused when this is `Some`. It
    /// exists as a `JitState` hook rather than a `JitCodeSym` call because
    /// `Self::Sym` carries no `JitCodeSym` bound, and the split-unaware
    /// `StateFieldLayout` reports a non-empty range for symbols that reserve
    /// nothing — gating on that would refuse every multi-frame bridge.
    ///
    /// `None` — the default, and every symbol that has not opted in via
    /// `split_dispatch` — means no register in this state is identity-only,
    /// so nothing was blanked and the resume is complete.
    fn reserved_int_identity_range() -> Option<(usize, usize)> {
        None
    }

    /// Enter the walk at the jitcode position a guard failed on.
    ///
    /// `pyjitpl.py handle_guard_failure` reaches
    /// `rebuild_state_after_failure`, whose per-frame
    /// `setup_resume_at_op(pc)` puts the rebuilt frame back at the guard, and
    /// then keeps interpreting from there.  The rest of the opcode the guard
    /// sits inside is therefore recorded, which is the whole difference
    /// between it and a resume that hands the interpreter a later position:
    /// the tail runs once either way, but only here does it also land in the
    /// trace.
    ///
    /// `frames` is the rebuilt framestack — outermost first, each carrying its
    /// own register file, because the walk cannot re-derive either: a guard is
    /// not a position the interpreter is re-enterable at, and the callers it
    /// sits under are suspended mid-instruction.
    ///
    /// The default declines.  A state with no dispatch jitcode has no position
    /// to enter, and declining is not a loss: `compile.py handle_fail` treats
    /// bridging and the blackhole as alternatives, so the caller takes the
    /// other one.
    fn trace_from_guard_resume_position<R: crate::pyjitpl::JitCodeRuntime>(
        _ctx: &mut crate::trace_ctx::TraceCtx,
        _sym: &mut Self::Sym,
        _frames: &[GuardResumeFrame],
        _outer_program_pc: usize,
        _runtime: &R,
    ) -> Option<crate::TraceAction> {
        None
    }

    /// Rebind portal reds on a guard-resume framestack from the live
    /// interpreter state.
    ///
    /// Generated `setup_bridge_sym` binds a state-field identity register
    /// the resume frame does not carry as a `Const` from the live mirrors
    /// `initialize_sym` left behind. A frontend whose reds *are* those
    /// identities — Grain's live frame and live Vm, republished on every
    /// merge-point consult — has the same gap on the opposite side: the
    /// register is present, but `bridge_decode_red` keys it by failarg
    /// index and the concrete bits can still be a walk-local address the
    /// parent loop walk snapshotted. Residual calls then write through
    /// that address instead of the interpreter object still on the stack.
    ///
    /// Only the concrete shadow is rewritten. The `OpRef` stays the
    /// failarg or folded `Const` the resume stream named, so a compiled
    /// bridge keeps taking a per-eval red rather than baking this eval's
    /// stack pointer in. The default is a no-op: generated frontends
    /// already rebound their identity registers in `setup_bridge_sym`,
    /// and a state whose reds are not heap/stack identities has nothing
    /// to replace.
    fn rebind_bridge_reds(&self, _frames: &mut [GuardResumeFrame]) {}

    /// Same as [`Self::rebind_bridge_reds`], with the compiled guard's
    /// failarg dump so a red that is a livebox but not a portal-register
    /// occupant can still name an `InputArg`.
    fn rebind_bridge_reds_from_fail(&self, frames: &mut [GuardResumeFrame], fail_values: &[i64]) {
        let _ = fail_values;
        self.rebind_bridge_reds(frames);
    }

    /// resume.py rebuild_from_resumedata: decode rd_numb to
    /// reconstruct the complete frame state for bridge tracing.
    ///
    /// RPython creates ResumeDataBoxReader, consumes vable/vref/frame
    /// sections, fills MIFrame registers with InputArgBox/ConstBox mix,
    /// returns (liveboxes, virtualizable_boxes, virtualref_boxes).
    ///
    /// Returns None when storage is not available (legacy path).
    /// `storage` is the guard-owned shared `Arc<ResumeStorage>` handle
    /// (compile.py `ResumeGuardDescr`); readers borrow
    /// `storage.rd_numb` / `storage.rd_consts()` instead of owned
    /// copies.
    fn rebuild_from_resumedata(
        _meta: &mut Self::Meta,
        _fail_arg_types: &[Type],
        _storage: Option<&std::sync::Arc<crate::resume::ResumeStorage>>,
    ) -> Option<ResumeDataResult> {
        None
    }

    /// pyjitpl.py compile_loop parity: build final meta from
    /// the MergePoint that matched at close time, not from the trace start.
    ///
    /// RPython's compile_loop extracts greenkey and inputargs from
    /// `original_boxes` (which comes from `current_merge_points[j]`).
    /// This method rebuilds the meta entirely from MergePoint data,
    /// replacing the post-patch `update_meta_for_cut` band-aid.
    fn build_meta_from_merge_point(
        provisional: &Self::Meta,
        header_pc: usize,
        original_boxes: &[crate::trace_ctx::GreenBox],
    ) -> Self::Meta {
        // Default: clone + patch (backward compat for non-pyre consumers)
        let mut meta = provisional.clone();
        let original_box_types: Vec<Type> = original_boxes.iter().map(|gb| gb.ty).collect();
        Self::update_meta_for_cut(&mut meta, header_pc, &original_box_types);
        meta
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]);

    /// Restore from the typed register banks separately. The forward-resume
    /// path (jitdriver back-edge) holds int reds in `registers_i` and ref
    /// reds in `registers_r`; a ref-typed state field must be read from the
    /// ref bank, not the int bank. The default ignores `ref_values` and
    /// restores int-only, so interpreters with no ref state fields keep
    /// their existing behavior; the `#[jit_interp]` macro overrides this
    /// when a `ref(...)` state field is declared.
    fn restore_banked(&mut self, meta: &Self::Meta, int_values: &[i64], _ref_values: &[i64]) {
        self.restore(meta, int_values);
    }

    /// Restore from all three typed register banks separately. The default
    /// preserves existing int/ref interpreters by forwarding to
    /// `restore_banked` and ignoring the float slice.
    fn restore_banked3(
        &mut self,
        meta: &Self::Meta,
        int_values: &[i64],
        ref_values: &[i64],
        _float_values: &[i64],
    ) {
        self.restore_banked(meta, int_values, ref_values);
    }

    fn restore_values(&mut self, meta: &Self::Meta, values: &[Value]) {
        // Split the canonical-order typed values into the int and ref
        // register banks (preserving per-type order) so ref-typed state
        // fields route to `restore_banked3`'s ref slice and float-typed state
        // fields route to its float slice. Invariant: no non-state-field
        // `Value::Float` rides this live stream today; adding a general float
        // red here must extend the positional restore contract explicitly.
        let mut ints: Vec<i64> = Vec::with_capacity(values.len());
        let mut refs: Vec<i64> = Vec::new();
        let mut floats: Vec<i64> = Vec::new();
        for value in values {
            match value {
                Value::Int(v) => ints.push(*v),
                Value::Float(v) => floats.push(v.to_bits() as i64),
                Value::Ref(r) => refs.push(r.as_usize() as i64),
                Value::Void => ints.push(0),
            }
        }
        self.restore_banked3(meta, &ints, &refs, &floats);
    }

    fn restore_guard_failure_values(
        &mut self,
        meta: &Self::Meta,
        values: &[Value],
        _exception: &ExceptionState,
    ) -> bool {
        self.restore_values(meta, values);
        true
    }

    fn recover_after_compiled_run(&mut self) {}

    fn debug_state_fields(&self, _meta: &Self::Meta) -> Option<String> {
        None
    }

    fn debug_state_live_labels(&self, _meta: &Self::Meta) -> Option<Vec<String>> {
        None
    }

    /// Whole-circuit single-pass: read the walk's scalar state-field concrete
    /// values off the still-live persistent sym, in scalar state-field index
    /// order (idx `0..num_scalars`). Called from the CloseLoop arm of
    /// `merge_point` while the sym is `Some`, before the arm clears it; the
    /// captured values are stashed on the `MetaInterp` and applied to native
    /// `state` after the walk closes (`writeback_scalar_state_fields_from_values`
    /// via the `jit_merge_point!` hook). Empty when the state has no scalar
    /// fields. Default empty; the `#[jit_interp]` macro overrides it for
    /// state-field consumers.
    fn collect_scalar_state_field_values(_sym: &Self::Sym) -> Vec<i64> {
        Vec::new()
    }

    /// Ref-bank sibling of `collect_scalar_state_field_values`, in ref-scalar
    /// state-field index order (idx `0..num_ref_scalars`). Raw pointer bits, the
    /// encoding `registers_r` holds. Read off the same still-live sym, for the
    /// same reason: the walk keeps ref state fields on the sym and native
    /// `state` is frozen at trace-start.
    fn collect_ref_scalar_state_field_values(_sym: &Self::Sym) -> Vec<i64> {
        Vec::new()
    }

    /// Whole-circuit single-pass: write the walk's scalar state-field concrete
    /// values (captured at close time by `collect_scalar_state_field_values`)
    /// into native state. `values[idx]` is the scalar at state-field index
    /// `idx`, written in the same order the sym read them. A scalar the walk
    /// mutates but `recover` cannot re-derive from shared storage (a
    /// `selected` storage index set from a program operand, say) would
    /// otherwise stay frozen at its trace-start value. Applied before
    /// `recover_after_compiled_run` so recover then refines the storage-derived
    /// caches (stacksize / list refs) from the now-current scalars. Default
    /// no-op; the `#[jit_interp]` macro overrides it for state-field consumers.
    fn writeback_scalar_state_fields_from_values(&mut self, _values: &[i64]) {}

    /// Blackhole CRN path for virtualizable-array states: write only the
    /// scalar state fields whose fixed identity registers are live in the
    /// terminal blackhole frame. RPython resume writes through the liveness
    /// enumeration (`resume.py:_prepare_next_section` callbacks), so an
    /// unlived register is not a dense state-field image and must not clobber
    /// native state.
    fn writeback_live_scalar_state_field(&mut self, _field_idx: usize, _value: i64) {}

    /// Ref-bank sibling of `writeback_live_scalar_state_field`.
    fn writeback_live_ref_scalar_state_field(&mut self, _field_idx: usize, _value: i64) {}

    /// Float-bank sibling of `writeback_live_scalar_state_field`. `value`
    /// is the raw f64 bit carrier from `registers_f`; generated impls convert
    /// with `f64::from_bits`.
    fn writeback_live_float_scalar_state_field(&mut self, _field_idx: usize, _value: i64) {}

    /// Whole-circuit single-pass: write the walk's loop-carried virtualizable
    /// array element values (captured at close time by the trace ctx's
    /// `collect_virtualizable_element_values`) into native state, the array
    /// analog of `writeback_scalar_state_fields_from_values`. `values` is the
    /// flat `[arr0_elem0.., arr1_elem0.., ..]` layout in state-field declaration
    /// order (the same order `extract_live` / `collect_jump_args_with_boxes`
    /// use). The walk mutates the array on the trace-ctx shadow only, so without
    /// this native state's array stays frozen at trace-start and the compiled
    /// loop re-executes the peeled iteration. Applied before
    /// `recover_after_compiled_run`. Default no-op; the `#[jit_interp]` macro
    /// overrides it only for consumers declaring a `[.. ; virt]` array.
    fn writeback_virt_array_state_fields_from_values(&mut self, _values: &[i64]) {}

    /// blackhole.py `_exit_frame_with_exception` → warmspot.py:998-1005.
    ///
    /// The resumed blackhole frame chain raised an exception (`exc`, a GC ref
    /// to the exception object) that escaped every reconstructed frame. The
    /// blackhole resume has *finished* — the chain is fully consumed — so this
    /// hands the exception to the interpreter's own exception machinery instead
    /// of re-executing from the guard pc, mirroring how the Python-bytecode JIT
    /// re-raises the stored Ref into the outer interpreter (`eval.rs`
    /// `is_exit_frame_with_exception` → `PyError::from_exc_object`). Returns the
    /// pc to resume at (the interpreter's unwind/handler entry).
    ///
    /// A compiled run that ends in the exit-with-exception FINISH
    /// (`ExitFrameWithExceptionDescrRef.handle_fail`) is delivered here too.
    ///
    /// The default returns `None`: an interpreter with no exception machinery
    /// never raises and so can never produce `ExitFrameWithExceptionRef` from a
    /// blackhole resume, leaving this unreachable for it; the caller then falls
    /// back to crude state recovery. An interpreter that *can* raise overrides
    /// this to deliver `exc` and return its handler pc.
    fn deliver_blackhole_exception(&mut self, _exc: GcRef) -> Option<usize> {
        None
    }

    /// State-field JIT register layout: the per-instance scalar count,
    /// flattened fixed `[int]` array lengths, and virt-array count, mirroring
    /// `live_slots_for_state_field_jit`.  The blackhole resume path threads
    /// this onto the root `BlackholeInterpreter` so its `state_field` handlers
    /// map a logical scalar/array index to the flat register slot the resume
    /// reader seeded (`blackhole.py setarg_i`).  Default empty for
    /// interpreters with no state fields (pyre).
    fn state_field_layout(&self) -> crate::blackhole::StateFieldLayout {
        crate::blackhole::StateFieldLayout::default()
    }

    fn sync_virtualizable_before_jit(
        &mut self,
        meta: &Self::Meta,
        virtualizable: &str,
        info: &VirtualizableInfo,
    ) -> bool {
        let Some(obj_ptr) = self.virtualizable_heap_ptr(meta, virtualizable, info) else {
            return true;
        };
        let lengths = if info.can_read_all_array_lengths_from_heap() {
            unsafe { info.read_array_lengths_from_heap(obj_ptr.cast_const()) }
        } else {
            let Some(lengths) = self.virtualizable_array_lengths(meta, virtualizable, info) else {
                return true;
            };
            lengths
        };
        let boxes = unsafe { info.read_boxes(obj_ptr.cast_const(), &lengths) };
        self.import_virtualizable_boxes(meta, virtualizable, info, &boxes, &lengths)
    }

    fn sync_virtualizable_after_jit(
        &mut self,
        meta: &Self::Meta,
        virtualizable: &str,
        info: &VirtualizableInfo,
    ) {
        let Some(obj_ptr) = self.virtualizable_heap_ptr(meta, virtualizable, info) else {
            return;
        };
        let Some((boxes, array_lengths)) =
            self.export_virtualizable_boxes(meta, virtualizable, info)
        else {
            return;
        };
        unsafe {
            info.write_boxes_with_lengths(obj_ptr, &boxes, &array_lengths);
            info.reset_vable_token(obj_ptr);
        }
    }

    fn sync_named_virtualizable_before_jit(
        &mut self,
        meta: &Self::Meta,
        virtualizable: &str,
        info: Option<&VirtualizableInfo>,
    ) -> bool {
        match info {
            Some(info) => self.sync_virtualizable_before_jit(meta, virtualizable, info),
            None => true,
        }
    }

    fn sync_named_virtualizable_after_jit(
        &mut self,
        meta: &Self::Meta,
        virtualizable: &str,
        info: Option<&VirtualizableInfo>,
    ) {
        if let Some(info) = info {
            self.sync_virtualizable_after_jit(meta, virtualizable, info);
        }
    }

    fn virtualizable_heap_ptr(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<*mut u8> {
        None
    }

    fn virtualizable_array_lengths(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<Vec<usize>> {
        None
    }

    /// Take the virtualizable's fields as [`VirtualizableInfo::read_boxes`]
    /// lists them — every static, then every array item — with the length of
    /// each array.
    fn import_virtualizable_boxes(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
        _boxes: &[i64],
        _array_lengths: &[usize],
    ) -> bool {
        true
    }

    /// The virtualizable's fields in [`VirtualizableInfo::read_boxes`] order,
    /// with the length of each array: the pair
    /// [`VirtualizableInfo::write_boxes_with_lengths`] takes.
    fn export_virtualizable_boxes(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<(Vec<i64>, Vec<usize>)> {
        None
    }

    /// Buffer-filling form of [`Self::export_virtualizable_boxes`], for the
    /// compiled-entry path that re-reads the virtualizable on every call.
    ///
    /// `warmstate.py execute_assembler` passes the virtualizable to
    /// the assembler as a single pointer argument and only clears its token;
    /// it never copies the fields out, so upstream has no per-entry cost here
    /// to mirror. majit's entry flattens the fields into scalar inputargs
    /// instead, which is a real per-entry read — but the *storage* for that
    /// read is fixed by the artifact's inputarg shape and so need not be
    /// reallocated each time.
    ///
    /// `boxes` and `array_lengths` arrive cleared with whatever capacity
    /// previous entries left them, so an implementation that extends them in
    /// place reuses that storage. Returns whether the export succeeded,
    /// matching the `Option` of the owning form; on `false` the buffers'
    /// contents are unspecified and the caller discards them.
    ///
    /// The default defers to [`Self::export_virtualizable_boxes`] so existing
    /// implementations keep working unchanged.
    fn export_virtualizable_boxes_into(
        &self,
        meta: &Self::Meta,
        virtualizable: &str,
        info: &VirtualizableInfo,
        boxes: &mut Vec<i64>,
        array_lengths: &mut Vec<usize>,
    ) -> bool {
        let Some((b, l)) = self.export_virtualizable_boxes(meta, virtualizable, info) else {
            return false;
        };
        *boxes = b;
        *array_lengths = l;
        true
    }

    /// `reached_loop_header` builds `live_arg_boxes` from the merge-point
    /// operands and stashes it on `TraceCtx::close_jump_boxes`. Generated
    /// `#[jit_interp]` states do not keep a second copy on the sym.
    fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
        Vec::new()
    }

    /// Loop-close jump-arg collection with access to the live
    /// `virtualizable_boxes` shadow (the trace ctx's standard-virtualizable
    /// box list, identity LAST).
    ///
    /// pyjitpl.py `reached_loop_header`:
    /// `live_arg_boxes += self.virtualizable_boxes; live_arg_boxes.pop()`
    /// — the loop-carried args include every virtualizable element box
    /// (dropping the trailing identity). The macro state-field JIT carries
    /// `[int; virt]` array elements through the vable shadow
    /// (`set_virtualizable_entry_at`, no IR), so the elements only exist in
    /// `boxes`, not in `Sym`; this hook lets the carrier splice them into the
    /// JUMP in place of the `<arr>_ptr`/`<arr>_len` placeholders that the
    /// no-ctx `collect_jump_args` emits.
    ///
    /// `boxes` is `TraceCtx::collect_virtualizable_typed_boxes()` extracted by
    /// the caller (owned clone, so the trace-ctx borrow is released before the
    /// subsequent `compile_loop`). Default delegates to `collect_jump_args`,
    /// so PyFrame (which closes via `CloseLoopWithArgs`, never this path) and
    /// non-virtualizable state-field JITs (e.g. `regs: [int]`) are unaffected.
    ///
    /// The boxes are typed because the same construction also builds the
    /// merge-point registration's `original_boxes`
    /// (`JitCodeSym::loop_carried_boxes`), and those become a cut trace's
    /// typed LABEL inputargs. RPython gets that for free — one
    /// `live_arg_boxes` list of `Box`es, each already carrying its type.
    fn collect_jump_args_with_boxes(sym: &Self::Sym, _boxes: &[(OpRef, Type)]) -> Vec<OpRef> {
        Self::collect_jump_args(sym)
    }

    /// Closing JUMP boxes from the live portal frame
    /// (`pyjitpl.py` `reached_loop_header` `live_arg_boxes` =
    /// `greenboxes + redboxes` then `+= virtualizable_boxes; pop`).
    /// Default is the vable-only list. Header-revisit CloseLoop uses
    /// `reached_loop_header_live_arg_boxes` instead of this hook.
    fn collect_jump_args_from_portal(
        sym: &Self::Sym,
        _frame: &crate::pyjitpl::MIFrame,
        boxes: &[(OpRef, Type)],
    ) -> Option<Vec<OpRef>> {
        Some(Self::collect_jump_args_with_boxes(sym, boxes))
    }

    /// `pyjitpl.py MetaInterp.capture_resumedata` for jitdriver-level
    /// guard sites (`force_finish_trace`'s GuardAlwaysFails). Walks the
    /// live framestack; identity slots already hold the reds. Default
    /// no-op returns `None`. Reachable from `merge_point` /
    /// `force_finish_trace` where `S::Sym` is not constrained to
    /// `JitCodeSym`.
    fn populate_frame_for_guard(
        _sym: &Self::Sym,
        _frames: &mut crate::pyjitpl::MIFrameStack,
        _op_live: u8,
        _all_liveness: &[u8],
        _virtualizable_boxes: &[OpRef],
        _virtualref_boxes: &[(OpRef, usize)],
    ) -> Option<crate::recorder::Snapshot> {
        None
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        Self::collect_jump_args(sym)
            .into_iter()
            .map(|opref| (opref, Type::Int))
            .collect()
    }

    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool;

    fn validate_close_with_jump_args(
        sym: &Self::Sym,
        meta: &Self::Meta,
        _jump_args: &[OpRef],
    ) -> bool {
        Self::validate_close(sym, meta)
    }

    /// resume.py:1049 parity: push a reconstructed caller frame.
    /// Called during multi-frame deopt: innermost frame is current,
    /// outer frames are pushed in reverse order (outermost first).
    ///
    /// `frame_index`: 0 = outermost, N-1 = innermost (current)
    /// `values`: typed values for this frame's slots
    /// `pc`: the program counter / bytecode position for this frame
    /// `jitcode_index`: JitCode index for CodeObject lookup
    ///
    /// Returns true if the frame was successfully pushed.
    fn push_caller_frame(
        &mut self,
        _meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _values: &[Value],
        _pc: u64,
        _jitcode_index: i32,
    ) -> bool {
        false // Default: no multi-frame support
    }

    /// Pop the current frame and resume the caller frame.
    /// Called when blackhole execution of the innermost frame completes.
    fn pop_to_caller_frame(&mut self, _meta: &Self::Meta) -> bool {
        false // Default: no multi-frame support
    }

    /// RPython virtualizable.py: build VirtualizableInfo.
    /// Override when the interpreter has a virtualizable object.
    ///
    /// Returns `Arc<VirtualizableInfo>` so the field descriptors built
    /// by `finalize_arc` can carry the vinfo backref that
    /// `FieldDescr::get_vinfo()` returns at `emit_force_virtualizable`
    /// time (pyjitpl.py).
    #[allow(non_snake_case)]
    fn __build_virtualizable_info()
    -> Option<std::sync::Arc<crate::virtualizable::VirtualizableInfo>> {
        None
    }

    /// Check if multi-frame caller-stack restore is supported.
    fn supports_multi_frame_restore(&self) -> bool {
        false
    }

    fn materialize_virtual_ref(
        &mut self,
        _meta: &Self::Meta,
        _virtual_index: usize,
        _materialized: &MaterializedVirtual,
    ) -> Option<GcRef> {
        None
    }

    fn materialize_virtual_ref_with_refs(
        &mut self,
        meta: &Self::Meta,
        virtual_index: usize,
        materialized: &MaterializedVirtual,
        _materialized_refs: &[Option<GcRef>],
    ) -> Option<GcRef> {
        self.materialize_virtual_ref(meta, virtual_index, materialized)
    }

    /// `resume.py setarrayitem` / `resume.py setfield`
    /// parity — RPython dispatches via `descr.is_pointer_field()` /
    /// `arraydescr.is_array_of_pointers()`.  pyre passes the live
    /// `descr` Arc so the driver can call the same `FieldDescr` /
    /// `ArrayDescr` trait methods directly.
    fn pending_field_write_layout(
        &self,
        _meta: &Self::Meta,
        _descr: Option<&majit_ir::DescrRef>,
        _is_array_item: bool,
    ) -> Option<PendingFieldWriteLayout> {
        None
    }

    fn restore_reconstructed_frames(
        &mut self,
        meta: &Self::Meta,
        reconstructed_state: &ReconstructedState,
        materialized_virtuals: &[MaterializedVirtual],
        exception: &ExceptionState,
    ) -> bool {
        let cache = self.materialize_virtual_refs(meta, materialized_virtuals);
        self.try_restore_reconstructed_frames_with_cache(
            meta,
            reconstructed_state,
            exception,
            &cache,
        )
    }

    fn restore_reconstructed_frames_with_resume_layout(
        &mut self,
        meta: &Self::Meta,
        reconstructed_state: &ReconstructedState,
        frame_layouts: Option<&[ResumeFrameLayoutSummary]>,
        materialized_virtuals: &[MaterializedVirtual],
        exception: &ExceptionState,
    ) -> bool {
        let total_frames = reconstructed_state.frames.len();
        let can_use_generic_layout = total_frames == 1
            || reconstructed_state
                .frames
                .iter()
                .enumerate()
                .any(|(frame_index, frame)| {
                    frame.slot_types.as_ref().is_some()
                        || self
                            .reconstructed_frame_value_types_with_metadata(
                                meta,
                                frame_index,
                                total_frames,
                                frame,
                            )
                            .is_some()
                        || frame_layouts
                            .and_then(|layouts| layouts.get(frame_index))
                            .filter(|layout| {
                                layout.pc == frame.pc
                                    && layout.slot_layouts.len() == frame.values.len()
                            })
                            .and_then(|layout| layout.slot_types.as_ref())
                            .is_some()
                });

        if can_use_generic_layout {
            let cache = self.materialize_virtual_refs(meta, materialized_virtuals);
            if self.try_restore_reconstructed_frames_with_cache_and_layout(
                meta,
                reconstructed_state,
                exception,
                &cache,
                frame_layouts,
            ) {
                return true;
            }
        }

        self.restore_reconstructed_frames(
            meta,
            reconstructed_state,
            materialized_virtuals,
            exception,
        )
    }

    fn reconstructed_frame_value_types(
        &self,
        _meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _frame_pc: u64,
    ) -> Option<Vec<Type>> {
        None
    }

    fn reconstructed_frame_value_types_with_metadata(
        &self,
        meta: &Self::Meta,
        frame_index: usize,
        total_frames: usize,
        frame: &crate::resume::ReconstructedFrame,
    ) -> Option<Vec<Type>> {
        self.reconstructed_frame_value_types(meta, frame_index, total_frames, frame.pc)
    }

    fn restore_reconstructed_frame_values(
        &mut self,
        _meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _frame_pc: u64,
        _values: &[Value],
        _exception: &ExceptionState,
    ) -> bool {
        false
    }

    fn restore_reconstructed_frame_values_with_metadata(
        &mut self,
        meta: &Self::Meta,
        frame_index: usize,
        total_frames: usize,
        frame: &crate::resume::ReconstructedFrame,
        values: &[Value],
        exception: &ExceptionState,
    ) -> bool {
        self.restore_reconstructed_frame_values(
            meta,
            frame_index,
            total_frames,
            frame.pc,
            values,
            exception,
        )
    }

    fn try_restore_reconstructed_frames_with_cache(
        &mut self,
        meta: &Self::Meta,
        reconstructed_state: &ReconstructedState,
        exception: &ExceptionState,
        materialized_refs: &[Option<GcRef>],
    ) -> bool {
        self.try_restore_reconstructed_frames_with_cache_and_layout(
            meta,
            reconstructed_state,
            exception,
            materialized_refs,
            None,
        )
    }

    fn try_restore_reconstructed_frames_with_cache_and_layout(
        &mut self,
        meta: &Self::Meta,
        reconstructed_state: &ReconstructedState,
        exception: &ExceptionState,
        materialized_refs: &[Option<GcRef>],
        frame_layouts: Option<&[ResumeFrameLayoutSummary]>,
    ) -> bool {
        if reconstructed_state.frames.is_empty() {
            return false;
        }

        let total_frames = reconstructed_state.frames.len();
        let mut used_generic = false;
        for (frame_index, frame) in reconstructed_state.frames.iter().enumerate() {
            let layout_types = frame.slot_types.clone().or_else(|| {
                frame_layouts
                    .and_then(|layouts| layouts.get(frame_index))
                    .filter(|layout| {
                        layout.pc == frame.pc && layout.slot_layouts.len() == frame.values.len()
                    })
                    .and_then(|layout| layout.slot_types.clone())
            });
            if let Some(types) = self
                .reconstructed_frame_value_types_with_metadata(
                    meta,
                    frame_index,
                    total_frames,
                    frame,
                )
                .or(layout_types)
            {
                let Some(values) =
                    frame_values_from_reconstructed(&frame.values, &types, materialized_refs)
                else {
                    return false;
                };
                if self.restore_reconstructed_frame_values_with_metadata(
                    meta,
                    frame_index,
                    total_frames,
                    frame,
                    &values,
                    exception,
                ) {
                    used_generic = true;
                    continue;
                }
                if total_frames == 1 && frame_index + 1 == total_frames {
                    self.restore_values(meta, &values);
                    return true;
                }
                return false;
            }
        }

        if used_generic {
            return true;
        }

        let frame = reconstructed_state.frames.last().expect("non-empty");
        let types = frame
            .slot_types
            .clone()
            .or_else(|| {
                frame_layouts
                    .and_then(|layouts| layouts.last())
                    .filter(|layout| {
                        layout.pc == frame.pc && layout.slot_layouts.len() == frame.values.len()
                    })
                    .and_then(|layout| layout.slot_types.clone())
            })
            .unwrap_or_else(|| self.live_value_types(meta));
        if types.len() != frame.values.len() {
            return false;
        }
        let Some(values) =
            frame_values_from_reconstructed(&frame.values, &types, materialized_refs)
        else {
            return false;
        };
        self.restore_values(meta, &values);
        true
    }

    fn restore_guard_failure(
        &mut self,
        meta: &Self::Meta,
        fail_values: &[i64],
        reconstructed_state: Option<&ReconstructedState>,
        materialized_virtuals: &[MaterializedVirtual],
        pending_field_writes: &[ResolvedPendingFieldWrite],
        exception: &ExceptionState,
    ) -> bool {
        self.restore_guard_failure_with_resume_layout(
            meta,
            fail_values,
            reconstructed_state,
            None,
            materialized_virtuals,
            pending_field_writes,
            exception,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    fn restore_guard_failure_with_resume_layout(
        &mut self,
        meta: &Self::Meta,
        fail_values: &[i64],
        reconstructed_state: Option<&ReconstructedState>,
        frame_layouts: Option<&[ResumeFrameLayoutSummary]>,
        materialized_virtuals: &[MaterializedVirtual],
        pending_field_writes: &[ResolvedPendingFieldWrite],
        exception: &ExceptionState,
    ) -> bool {
        self.restore_guard_failure_with_session_cache(
            meta,
            fail_values,
            reconstructed_state,
            frame_layouts,
            materialized_virtuals,
            pending_field_writes,
            exception,
            None,
        )
    }

    fn materialize_virtual_refs(
        &mut self,
        meta: &Self::Meta,
        materialized_virtuals: &[MaterializedVirtual],
    ) -> Vec<Option<GcRef>> {
        let mut refs = vec![None; materialized_virtuals.len()];
        let mut progress = true;
        while progress {
            progress = false;
            for (virtual_index, virtual_value) in materialized_virtuals.iter().enumerate() {
                if refs[virtual_index].is_some() {
                    continue;
                }
                if let Some(materialized) = self.materialize_virtual_ref_with_refs(
                    meta,
                    virtual_index,
                    virtual_value,
                    &refs,
                ) {
                    refs[virtual_index] = Some(materialized);
                    progress = true;
                }
            }
        }
        refs
    }

    /// Materialize virtual refs into an existing session cache.
    ///
    /// Virtuals that are already present in the cache (from a previous deopt
    /// in the same session) are skipped. Only unmaterialized entries trigger
    /// a call to `materialize_virtual_ref_with_refs`.
    fn materialize_virtual_refs_into_cache(
        &mut self,
        meta: &Self::Meta,
        materialized_virtuals: &[MaterializedVirtual],
        cache: &mut DeoptMaterializationCache,
    ) {
        cache.ensure_capacity(materialized_virtuals.len());
        let mut progress = true;
        while progress {
            progress = false;
            for (idx, virt) in materialized_virtuals.iter().enumerate() {
                if cache.refs[idx].is_some() {
                    continue; // Already materialized (possibly from previous session)
                }
                if let Some(materialized) =
                    self.materialize_virtual_ref_with_refs(meta, idx, virt, &cache.refs)
                {
                    cache.refs[idx] = Some(materialized);
                    progress = true;
                }
            }
        }
    }

    /// Force a single virtual using the session cache. Returns the cached
    /// result if already materialized, otherwise materializes and inserts
    /// into the cache.
    fn force_virtual_with_cache(
        &mut self,
        meta: &Self::Meta,
        cache: &mut DeoptMaterializationCache,
        virtual_index: usize,
        materialized: &MaterializedVirtual,
    ) -> Option<GcRef> {
        // Check cache first
        if let Some(cached) = cache.get(virtual_index) {
            return Some(cached);
        }
        // Materialize and cache
        cache.ensure_capacity(virtual_index + 1);
        let result =
            self.materialize_virtual_ref_with_refs(meta, virtual_index, materialized, &cache.refs)?;
        cache.refs[virtual_index] = Some(result);
        Some(result)
    }

    /// Restore guard failure state using a persistent session cache.
    ///
    /// This is the session-aware variant of `restore_guard_failure_with_resume_layout`.
    /// When `session_cache` is provided, materialized virtuals are stored in
    /// it and reused across subsequent deopts within the same session (e.g.,
    /// when GUARD_NOT_FORCED triggers resumption after CALL_MAY_FORCE).
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    fn restore_guard_failure_with_session_cache(
        &mut self,
        meta: &Self::Meta,
        fail_values: &[i64],
        reconstructed_state: Option<&ReconstructedState>,
        frame_layouts: Option<&[ResumeFrameLayoutSummary]>,
        materialized_virtuals: &[MaterializedVirtual],
        pending_field_writes: &[ResolvedPendingFieldWrite],
        exception: &ExceptionState,
        session_cache: Option<&mut DeoptMaterializationCache>,
    ) -> bool {
        if let Some(reconstructed_state) = reconstructed_state {
            let materialized_refs = if let Some(cache) = session_cache {
                self.materialize_virtual_refs_into_cache(meta, materialized_virtuals, cache);
                cache.refs.clone()
            } else {
                self.materialize_virtual_refs(meta, materialized_virtuals)
            };
            let total_frames = reconstructed_state.frames.len();

            // Multi-frame caller-stack reconstruction: push outer frames onto
            // the interpreter's call stack and restore only the innermost as
            // current state. This mirrors RPython's resume.py which rebuilds
            // the full framestack so blackhole can unwind through callers.
            if total_frames > 1 && self.supports_multi_frame_restore() {
                let restored = self.restore_multi_frame_with_caller_stack(
                    meta,
                    reconstructed_state,
                    frame_layouts,
                    exception,
                    &materialized_refs,
                );
                if restored {
                    self.replay_pending_field_writes(
                        meta,
                        pending_field_writes,
                        &materialized_refs,
                    );
                    return true;
                }
                // Fall through to existing paths if multi-frame push failed
            }

            let can_use_generic_cache = total_frames == 1
                || reconstructed_state
                    .frames
                    .iter()
                    .enumerate()
                    .any(|(frame_index, frame)| {
                        frame.slot_types.as_ref().is_some()
                            || self
                                .reconstructed_frame_value_types_with_metadata(
                                    meta,
                                    frame_index,
                                    total_frames,
                                    frame,
                                )
                                .is_some()
                            || frame_layouts
                                .and_then(|layouts| layouts.get(frame_index))
                                .filter(|layout| {
                                    layout.pc == frame.pc
                                        && layout.slot_layouts.len() == frame.values.len()
                                })
                                .and_then(|layout| layout.slot_types.as_ref())
                                .is_some()
                    });

            let restored = if can_use_generic_cache {
                let restored = self.try_restore_reconstructed_frames_with_cache_and_layout(
                    meta,
                    reconstructed_state,
                    exception,
                    &materialized_refs,
                    frame_layouts,
                );
                if restored {
                    true
                } else {
                    self.restore_reconstructed_frames_with_resume_layout(
                        meta,
                        reconstructed_state,
                        frame_layouts,
                        materialized_virtuals,
                        exception,
                    )
                }
            } else {
                self.restore_reconstructed_frames_with_resume_layout(
                    meta,
                    reconstructed_state,
                    frame_layouts,
                    materialized_virtuals,
                    exception,
                )
            };

            if !restored {
                return false;
            }
            self.replay_pending_field_writes(meta, pending_field_writes, &materialized_refs);
            return true;
        }

        self.restore(meta, fail_values);
        true
    }

    /// Multi-frame caller-stack reconstruction.
    ///
    /// Pushes outer frames (outermost first) onto the interpreter's call
    /// stack via `push_caller_frame`, then restores the innermost frame as
    /// the current interpreter state. This allows blackhole execution to
    /// unwind through the reconstructed caller frames.
    fn restore_multi_frame_with_caller_stack(
        &mut self,
        meta: &Self::Meta,
        reconstructed_state: &ReconstructedState,
        frame_layouts: Option<&[ResumeFrameLayoutSummary]>,
        exception: &ExceptionState,
        materialized_refs: &[Option<GcRef>],
    ) -> bool {
        let total_frames = reconstructed_state.frames.len();
        if total_frames < 2 {
            return false;
        }

        // Push outer frames (indices 0..total_frames-1) onto the call stack.
        // Frame 0 is the outermost caller, frame total_frames-2 is the
        // direct caller of the innermost frame.
        for frame_index in 0..total_frames - 1 {
            let frame = &reconstructed_state.frames[frame_index];
            let types =
                self.resolve_frame_types(meta, frame_index, total_frames, frame, frame_layouts);
            let Some(types) = types else {
                return false;
            };
            let Some(values) =
                frame_values_from_reconstructed(&frame.values, &types, materialized_refs)
            else {
                return false;
            };
            if !self.push_caller_frame(
                meta,
                frame_index,
                total_frames,
                &values,
                frame.pc,
                frame.jitcode_index,
            ) {
                return false;
            }
        }

        // Restore the innermost frame as current state.
        let innermost_index = total_frames - 1;
        let innermost = &reconstructed_state.frames[innermost_index];
        let types = self.resolve_frame_types(
            meta,
            innermost_index,
            total_frames,
            innermost,
            frame_layouts,
        );
        let Some(types) = types else {
            return false;
        };
        let Some(values) =
            frame_values_from_reconstructed(&innermost.values, &types, materialized_refs)
        else {
            return false;
        };
        self.restore_reconstructed_frame_values_with_metadata(
            meta,
            innermost_index,
            total_frames,
            innermost,
            &values,
            exception,
        )
    }

    /// Resolve the type layout for a single reconstructed frame, trying
    /// (in order): per-frame slot_types, interpreter-provided types,
    /// and resume layout types.
    fn resolve_frame_types(
        &self,
        meta: &Self::Meta,
        frame_index: usize,
        total_frames: usize,
        frame: &crate::resume::ReconstructedFrame,
        frame_layouts: Option<&[ResumeFrameLayoutSummary]>,
    ) -> Option<Vec<Type>> {
        frame
            .slot_types
            .clone()
            .or_else(|| {
                self.reconstructed_frame_value_types_with_metadata(
                    meta,
                    frame_index,
                    total_frames,
                    frame,
                )
            })
            .or_else(|| {
                frame_layouts
                    .and_then(|layouts| layouts.get(frame_index))
                    .filter(|layout| {
                        layout.pc == frame.pc && layout.slot_layouts.len() == frame.values.len()
                    })
                    .and_then(|layout| layout.slot_types.clone())
            })
    }

    fn replay_pending_field_writes(
        &mut self,
        meta: &Self::Meta,
        pending_field_writes: &[ResolvedPendingFieldWrite],
        materialized_refs: &[Option<GcRef>],
    ) {
        for pending in pending_field_writes {
            // `resume.py PENDINGFIELDSTRUCT.lldescr` is always
            // present in RPython; a `None` here would mean the
            // optimizer/recorder failed to attach the descr and replaying
            // the write would corrupt memory.  Fail loud rather than
            // silently skipping (parity with the `expect` at
            // `compile.rs`'s `build_guard_metadata`).
            let descr = pending.descr.as_ref().expect(
                "resume.py `_prepare_pendingfields`: PENDINGFIELDSTRUCT.lldescr must be set",
            );
            let Some(layout) =
                self.pending_field_write_layout(meta, Some(descr), pending.item_index.is_some())
            else {
                continue;
            };
            let Some(target) = materialized_value_to_i64(&pending.target, materialized_refs) else {
                continue;
            };
            let Some(value) = materialized_value_to_i64(&pending.value, materialized_refs) else {
                continue;
            };
            unsafe {
                apply_pending_field_write(layout, target as usize, pending.item_index, value)
            };
        }
    }
}

fn frame_values_from_reconstructed(
    frame_values: &[crate::resume::ReconstructedValue],
    types: &[Type],
    materialized_refs: &[Option<GcRef>],
) -> Option<Vec<Value>> {
    if frame_values.len() != types.len() {
        return None;
    }
    let mut result = Vec::with_capacity(frame_values.len());
    for (value, expected) in frame_values.iter().zip(types.iter().copied()) {
        let decoded = match (value, expected) {
            (crate::resume::ReconstructedValue::Value(raw), Type::Int) => Value::Int(*raw),
            (crate::resume::ReconstructedValue::Value(raw), Type::Ref) => {
                Value::Ref(GcRef(*raw as usize))
            }
            (crate::resume::ReconstructedValue::Value(raw), Type::Float) => {
                Value::Float(f64::from_bits(*raw as u64))
            }
            (crate::resume::ReconstructedValue::Value(_), Type::Void) => Value::Void,
            (crate::resume::ReconstructedValue::Virtual(index), _) => {
                // Virtual objects always materialize as Ref, regardless of the
                // expected type. The interpreter's restore_values() knows to
                // treat these as object references.
                Value::Ref(materialized_refs.get(*index).copied().flatten()?)
            }
            (crate::resume::ReconstructedValue::Uninitialized, _) => Value::Void,
            (crate::resume::ReconstructedValue::Unavailable, _) => Value::Void,
        };
        result.push(decoded);
    }
    Some(result)
}

fn materialized_value_to_i64(
    value: &MaterializedValue,
    materialized_refs: &[Option<GcRef>],
) -> Option<i64> {
    match value {
        MaterializedValue::Value(value) => Some(*value),
        MaterializedValue::VirtualRef(index) => materialized_refs
            .get(*index)
            .copied()
            .flatten()
            .map(|gc_ref| gc_ref.as_usize() as i64),
    }
}

unsafe fn apply_pending_field_write(
    layout: PendingFieldWriteLayout,
    target: usize,
    item_index: Option<usize>,
    value: i64,
) {
    unsafe {
        match layout {
            PendingFieldWriteLayout::Field { offset, value_type } => {
                let ptr = (target as *mut u8).add(offset);
                write_typed_value(ptr, value, value_type);
            }
            PendingFieldWriteLayout::ArrayItem {
                base_offset,
                item_size,
                item_type,
            } => {
                let index = item_index.unwrap_or(0);
                let ptr = (target as *mut u8).add(base_offset + index * item_size);
                write_typed_value(ptr, value, item_type);
            }
        }
    }
}

unsafe fn write_typed_value(ptr: *mut u8, value: i64, value_type: Type) {
    unsafe {
        match value_type {
            Type::Int => (ptr as *mut i64).write(value),
            Type::Ref => (ptr as *mut usize).write(value as usize),
            Type::Float => (ptr as *mut f64).write(f64::from_bits(value as u64)),
            Type::Void => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resume::{MaterializedValue, MaterializedVirtual};
    use std::cell::Cell;

    /// Test state that tracks how many times `materialize_virtual_ref` was
    /// called, so we can verify the session cache prevents re-materialization.
    #[derive(Default)]
    struct CachingTestState {
        materialize_call_count: Cell<usize>,
        restored_values: Vec<Value>,
        restored_frames: Vec<(usize, u64, Vec<Value>)>,
    }

    impl JitState for CachingTestState {
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

        fn restore_values(&mut self, _: &(), values: &[Value]) {
            self.restored_values = values.to_vec();
        }

        fn restore_reconstructed_frame_values_with_metadata(
            &mut self,
            _meta: &(),
            frame_index: usize,
            _total_frames: usize,
            frame: &crate::resume::ReconstructedFrame,
            values: &[Value],
            _exception: &ExceptionState,
        ) -> bool {
            self.restored_frames
                .push((frame_index, frame.pc, values.to_vec()));
            true
        }

        fn materialize_virtual_ref(
            &mut self,
            _meta: &(),
            virtual_index: usize,
            _materialized: &MaterializedVirtual,
        ) -> Option<GcRef> {
            self.materialize_call_count
                .set(self.materialize_call_count.get() + 1);
            Some(GcRef(0xB000 + virtual_index))
        }

        fn collect_jump_args(_: &()) -> Vec<OpRef> {
            Vec::new()
        }
        fn validate_close(_: &(), _: &()) -> bool {
            true
        }
    }

    #[test]
    fn test_deopt_materialization_cache_basic() {
        let mut cache = DeoptMaterializationCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.materialized_count(), 0);

        cache.insert(0, GcRef(0x100));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(0), Some(GcRef(0x100)));
        assert_eq!(cache.get(1), None);
        assert_eq!(cache.materialized_count(), 1);

        cache.insert(2, GcRef(0x200));
        assert_eq!(cache.len(), 3); // 0, 1(None), 2
        assert_eq!(cache.get(2), Some(GcRef(0x200)));
        assert_eq!(cache.materialized_count(), 2);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.materialized_count(), 0);
    }

    #[test]
    fn test_materialization_cache_persists_across_guard_not_forced() {
        let mut state = CachingTestState::default();

        let virtuals_session1 = vec![
            MaterializedVirtual::Obj {
                descr: None,
                type_id: 1,
                fields: vec![(0, MaterializedValue::Value(100))],
            },
            MaterializedVirtual::Obj {
                descr: None,
                type_id: 2,
                fields: vec![(0, MaterializedValue::Value(200))],
            },
        ];

        // First deopt: materialize A and B into session cache
        let mut session_cache = DeoptMaterializationCache::new();
        state.materialize_virtual_refs_into_cache(&(), &virtuals_session1, &mut session_cache);

        let calls_after_first = state.materialize_call_count.get();
        assert_eq!(calls_after_first, 2);
        assert_eq!(session_cache.get(0), Some(GcRef(0xB000)));
        assert_eq!(session_cache.get(1), Some(GcRef(0xB001)));

        // Second deopt (GUARD_NOT_FORCED resumption) in same session:
        // same virtuals should NOT be re-materialized
        state.materialize_virtual_refs_into_cache(&(), &virtuals_session1, &mut session_cache);

        let calls_after_second = state.materialize_call_count.get();
        assert_eq!(
            calls_after_second, calls_after_first,
            "virtuals should not be re-materialized from cache"
        );
        assert_eq!(session_cache.get(0), Some(GcRef(0xB000)));
        assert_eq!(session_cache.get(1), Some(GcRef(0xB001)));
    }

    #[test]
    fn test_materialization_cache_extends_for_new_virtuals() {
        let mut state = CachingTestState::default();

        let virtuals_first = vec![
            MaterializedVirtual::Obj {
                descr: None,
                type_id: 1,
                fields: vec![(0, MaterializedValue::Value(100))],
            },
            MaterializedVirtual::Obj {
                descr: None,
                type_id: 2,
                fields: vec![(0, MaterializedValue::Value(200))],
            },
        ];

        // First deopt: materialize A and B
        let mut session_cache = DeoptMaterializationCache::new();
        state.materialize_virtual_refs_into_cache(&(), &virtuals_first, &mut session_cache);
        assert_eq!(state.materialize_call_count.get(), 2);

        // Second deopt adds virtual C
        let virtuals_second = vec![
            MaterializedVirtual::Obj {
                descr: None,
                type_id: 1,
                fields: vec![(0, MaterializedValue::Value(100))],
            },
            MaterializedVirtual::Obj {
                descr: None,
                type_id: 2,
                fields: vec![(0, MaterializedValue::Value(200))],
            },
            MaterializedVirtual::Obj {
                descr: None,
                type_id: 3,
                fields: vec![(0, MaterializedValue::Value(300))],
            },
        ];

        state.materialize_virtual_refs_into_cache(&(), &virtuals_second, &mut session_cache);

        // Only 1 new call for virtual C (A and B were cached)
        assert_eq!(state.materialize_call_count.get(), 3);
        assert_eq!(session_cache.get(0), Some(GcRef(0xB000)));
        assert_eq!(session_cache.get(1), Some(GcRef(0xB001)));
        assert_eq!(session_cache.get(2), Some(GcRef(0xB002)));
    }

    #[test]
    fn test_virtual_ref_forcing_uses_session_cache() {
        let mut state = CachingTestState::default();
        let mut cache = DeoptMaterializationCache::new();

        let virt = MaterializedVirtual::Obj {
            descr: None,
            type_id: 1,
            fields: vec![(0, MaterializedValue::Value(42))],
        };

        // First call: materializes and caches
        let result1 = state.force_virtual_with_cache(&(), &mut cache, 0, &virt);
        assert_eq!(result1, Some(GcRef(0xB000)));
        assert_eq!(state.materialize_call_count.get(), 1);

        // Second call: returns cached value without materializing
        let result2 = state.force_virtual_with_cache(&(), &mut cache, 0, &virt);
        assert_eq!(result2, Some(GcRef(0xB000)));
        assert_eq!(
            state.materialize_call_count.get(),
            1,
            "should not re-materialize cached virtual"
        );
    }
}
