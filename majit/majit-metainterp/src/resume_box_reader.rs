//! Bridge-entry virtual materialization — the trace-emit (box-reader) flavour
//! of the resume-data reader.
//!
//! resume.py runs virtual rematerialization through a single
//! `AbstractVirtualInfo.allocate(decoder, index)` (resume.py) that is
//! polymorphic over the `decoder`: `ResumeDataDirectReader` (blackhole — real
//! `cpu.bh_new`, resume.py) writes memory, while
//! `ResumeDataBoxReader` (tracing/bridge — `metainterp.execute_new_with_vtable`,
//! resume.py) records NEW/SETFIELD ops into the trace. The direct
//! flavour lives in `resume.rs` (`VirtualInfoBlackholeExt::allocate`); this
//! module is the box-reader flavour, shared by every `JitState` consumer whose
//! `setup_bridge_sym` must re-materialize the guard's virtuals and replay its
//! deferred heap writes at bridge entry.

use majit_ir::OpRef;

/// Default `VRawBuffer` array-descr factory for consumers that never
/// materialize raw-buffer virtuals (e.g. a `#[jit_interp]` interpreter with no
/// ctypes/array virtuals). Mints a fresh `SimpleArrayDescr` via majit-ir,
/// ignoring `len_offset` (raw stores read only item size, and no length field
/// is consulted). A consumer that DOES virtualize raw buffers must inject its
/// own gccache-keyed factory instead.
pub fn default_bridge_array_descr(
    base_size: usize,
    item_size: usize,
    _len_offset: Option<usize>,
    item_type: majit_ir::Type,
    signed: bool,
) -> majit_ir::DescrRef {
    majit_ir::descr::make_array_descr_signed(base_size, item_size, item_type, signed)
}

/// resume.py VirtualCache — per-virtual-number `OpRef` banks the box
/// reader probes before allocating, plus the concrete `GcRef`/int shadows a
/// consumer may seed for branch-fold parity. `mint_raw_array_descr` is the
/// consumer-provided array-descr factory for `VRawBuffer` materialization (the
/// only virtual kind whose descr is minted rather than looked up in a parent
/// `SizeDescr`); descr identity is the consumer's gccache concern, so it is
/// injected rather than fixed in core.
pub struct BridgeVirtualCache<'a> {
    virtuals_ptr_cache: Vec<Option<OpRef>>,
    virtuals_int_cache: Vec<Option<OpRef>>,
    concrete_ptr_cache: Vec<Option<majit_ir::GcRef>>,
    concrete_int_cache: Vec<Option<i64>>,
    mint_raw_array_descr:
        fn(usize, usize, Option<usize>, majit_ir::Type, bool) -> majit_ir::DescrRef,
    /// The applying half of resume.py's reader pair, present only for the
    /// entry `ResumeDataBoxReader` serves upstream: one that no direct reader
    /// preceded.  `allocate_with_vtable` there is
    /// `metainterp.execute_new_with_vtable` and `setfield` is
    /// `metainterp.execute_setfield_gc` — `execute_and_record`, which applies
    /// to the heap AND records; `ResumeDataDirectReader` is the same walk with
    /// `cpu.bh_*` and no recording.
    ///
    /// `None` is the recording-only reader, which upstream has no counterpart
    /// for and which is sound only because the direct reader has already run
    /// for this guard: applying again would allocate a SECOND set of virtuals
    /// and record an identity the interpreter does not hold.
    ///
    /// The reader is built once in `start_bridge_tracing` and lives through
    /// `_prepare`, `consume_boxes` and `setup_bridge_sym`, so the allocator
    /// borrow spans the whole `rebuild_from_resumedata`.
    executing: Option<&'a dyn crate::resume::BlackholeAllocator>,
    /// `_prepare_pendingfields` writes this reader applied to a ref array:
    /// the target box, the item index and the value box. Upstream needs no
    /// record because `consume_boxes` hands frames their boxes; a consumer
    /// that rediscovers a frame's locals through that same array reuses the
    /// box the write just produced instead of a fresh load.
    pending_ref_array_writes: Vec<PendingRefArrayWrite>,
    /// The guard's fail values, which `resume.py decode_ref` reads a TAGBOX
    /// operand out of (`cpu.get_ref_value(self.deadframe, num)`). Empty for
    /// the recording-only reader, which never needs a concrete.
    ///
    /// Int and float failargs only — a ref failarg is read out of
    /// [`Self::fail_ref_roots`] instead.
    fail_values: Vec<i64>,
    /// The ref-typed fail values, indexed as `fail_values` is and registered
    /// as resume-construction GC roots.
    ///
    /// `decode_ref` reads through `self.deadframe`, an object the collector
    /// traces and updates, so upstream re-reads a moved address for free. A
    /// slice copied out of the deadframe does not move with it, and this walk
    /// allocates: a `bh_new` between the copy and the read collects, and the
    /// store then targets — or writes — an address that has moved. Rooting
    /// the copy restores what reading through the deadframe gave.
    ///
    /// Only ref-typed slots are carried; every other index holds 0, because
    /// the root walker hands each slot to the collector as a `GcRef` and an
    /// int that happens to land in the nursery range would be rewritten as if
    /// it were a pointer.
    fail_ref_roots: Vec<i64>,
    /// Addresses of the virtuals the applying reader has allocated, indexed by
    /// virtual number and registered as resume-construction GC roots.
    ///
    /// resume.py keeps each freshly allocated virtual in `virtuals_cache`,
    /// an ordinary RPython object and therefore a root, and its own comment
    /// says why: `allocate()` must fill the cache "as soon as they have the
    /// object, before they fill its fields", because a later allocation in the
    /// same walk can collect. Between one `bh_new` and the store that
    /// publishes its result nothing else refers to that object, so without
    /// these roots the second virtual of a two-virtual guard collects the
    /// first. The length is fixed at construction, so the registered slice
    /// never moves, and the collector writes forwarded addresses back through
    /// it — always read the address back from here, never from a copy.
    concrete_roots: Vec<i64>,
    /// Root-stack depth to unwind to, captured before this cache registered
    /// anything. Equal to the entry depth for the recording-only reader, whose
    /// unwind is then a no-op.
    roots_depth: usize,
}

impl Drop for BridgeVirtualCache<'_> {
    fn drop(&mut self) {
        majit_gc::shadow_stack::pop_resume_ref_roots_to(self.roots_depth);
    }
}

/// One `SETARRAYITEM_GC` of a ref item that `_prepare_pendingfields`
/// applied, kept on the reader (see `BridgeVirtualCache`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingRefArrayWrite {
    pub target: OpRef,
    pub item_index: usize,
    pub value: OpRef,
}

impl<'a> BridgeVirtualCache<'a> {
    /// The recording-only reader, for a bridge entered after a direct reader
    /// has already applied this guard's writes.
    pub fn new(
        size: usize,
        mint_raw_array_descr: fn(
            usize,
            usize,
            Option<usize>,
            majit_ir::Type,
            bool,
        ) -> majit_ir::DescrRef,
    ) -> Self {
        Self::recording(size, mint_raw_array_descr, &[])
    }

    /// The recording-only reader seeded with the objects the direct reader
    /// that preceded it allocated, indexed by virtual number — the
    /// `all_virtuals` a `ResumeDataDirectReader` is constructed with
    /// (`compile.py` `ResumeGuardForcedDescr.handle_fail`), carried here
    /// because this reader records what that one applied. A
    /// `getvirtual_ptr` hit then returns that object's box rather than
    /// allocating a second one.
    pub fn recording(
        size: usize,
        mint_raw_array_descr: fn(
            usize,
            usize,
            Option<usize>,
            majit_ir::Type,
            bool,
        ) -> majit_ir::DescrRef,
        all_virtuals: &[Option<majit_ir::GcRef>],
    ) -> Self {
        // Recording-only still allocates concrete shadows
        // (`materialize_concrete_virtual_ptr`). resume.py keeps each virtual
        // in `virtuals_cache` before filling its fields, because the next
        // allocation can collect. Register the same fixed slice the applying
        // reader uses; `Drop` unregisters it.
        let mut cache = Self {
            virtuals_ptr_cache: vec![None; size],
            virtuals_int_cache: vec![None; size],
            concrete_ptr_cache: vec![None; size],
            concrete_int_cache: vec![None; size],
            mint_raw_array_descr,
            executing: None,
            pending_ref_array_writes: Vec::new(),
            fail_values: Vec::new(),
            fail_ref_roots: Vec::new(),
            concrete_roots: vec![0i64; size],
            roots_depth: majit_gc::shadow_stack::resume_ref_roots_depth(),
        };
        if size > 0 {
            unsafe {
                majit_gc::shadow_stack::push_resume_ref_roots(&mut cache.concrete_roots);
            }
        }
        for (vidx, slot) in all_virtuals.iter().enumerate() {
            if let Some(gcref) = *slot {
                cache.set_concrete_ptr(vidx, gcref);
            }
        }
        cache
    }

    /// `ResumeDataBoxReader`: the same walk, applying each write through
    /// `allocator` as it records it.
    pub fn executing(
        size: usize,
        mint_raw_array_descr: fn(
            usize,
            usize,
            Option<usize>,
            majit_ir::Type,
            bool,
        ) -> majit_ir::DescrRef,
        allocator: &'a dyn crate::resume::BlackholeAllocator,
        fail_values: &[i64],
        fail_types: &[majit_ir::Type],
    ) -> Self {
        // Built field by field rather than from `Self::new`: the cache
        // unregisters its roots in `Drop`, and a type that implements `Drop`
        // cannot be moved out of by struct-update syntax.
        let mut cache = Self {
            virtuals_ptr_cache: vec![None; size],
            virtuals_int_cache: vec![None; size],
            concrete_ptr_cache: vec![None; size],
            concrete_int_cache: vec![None; size],
            mint_raw_array_descr,
            executing: Some(allocator),
            pending_ref_array_writes: Vec::new(),
            fail_values: fail_values.to_vec(),
            fail_ref_roots: fail_values
                .iter()
                .enumerate()
                .map(|(i, &bits)| match fail_types.get(i) {
                    Some(majit_ir::Type::Ref) => bits,
                    _ => 0,
                })
                .collect(),
            concrete_roots: vec![0i64; size],
            roots_depth: majit_gc::shadow_stack::resume_ref_roots_depth(),
        };
        // SAFETY, both registrations: each buffer is heap-allocated at a fixed
        // address, never resized after this point, and unregistered by `Drop`
        // before the cache dies.
        if size > 0 {
            unsafe {
                majit_gc::shadow_stack::push_resume_ref_roots(&mut cache.concrete_roots);
            }
        }
        if !cache.fail_ref_roots.is_empty() {
            unsafe {
                majit_gc::shadow_stack::push_resume_ref_roots(&mut cache.fail_ref_roots);
            }
        }
        cache
    }

    /// Publish a freshly allocated virtual as a root and remember which
    /// `OpRef` names it, so later operands resolve to the address the
    /// collector is maintaining rather than to a copy taken before it moved.
    fn set_concrete_root(&mut self, vidx: usize, address: i64) {
        if let Some(slot) = self.concrete_roots.get_mut(vidx) {
            *slot = address;
        }
    }

    /// The address a materialized virtual currently lives at, by the `OpRef`
    /// the recording half minted for it.
    pub(crate) fn concrete_root_of(&self, opref: OpRef) -> Option<i64> {
        let vidx = self
            .virtuals_ptr_cache
            .iter()
            .position(|slot| *slot == Some(opref))?;
        match self.concrete_roots.get(vidx) {
            Some(0) | None => None,
            Some(address) => Some(*address),
        }
    }

    /// The allocator the applying half stores through, or `None` when this is
    /// the recording-only reader.
    pub fn allocator(&self) -> Option<&'a dyn crate::resume::BlackholeAllocator> {
        self.executing
    }

    /// `_prepare_pendingfields` applied this ref-array write.
    pub fn push_pending_ref_array_write(&mut self, write: PendingRefArrayWrite) {
        self.pending_ref_array_writes.push(write);
    }

    /// The ref-array writes `_prepare_pendingfields` applied, in order.
    pub fn pending_ref_array_writes(&self) -> &[PendingRefArrayWrite] {
        &self.pending_ref_array_writes
    }

    /// The guard's fail values, for resolving a TAGBOX operand's concrete.
    fn fail_values(&self) -> &[i64] {
        &self.fail_values
    }

    /// The address a ref failarg currently lives at, read back through the
    /// rooted copy so a collection since the guard failed is accounted for.
    fn fail_ref_root(&self, index: usize) -> Option<i64> {
        self.fail_ref_roots.get(index).copied()
    }

    pub fn get_any(&self, i: usize) -> Option<OpRef> {
        self.virtuals_ptr_cache
            .get(i)
            .copied()
            .flatten()
            .or_else(|| self.virtuals_int_cache.get(i).copied().flatten())
    }

    pub fn set_ptr(&mut self, i: usize, v: OpRef) {
        self.virtuals_ptr_cache[i] = Some(v);
    }

    pub fn set_int(&mut self, i: usize, v: OpRef) {
        self.virtuals_int_cache[i] = Some(v);
    }

    /// Seed both banks from a box an earlier reader already allocated.
    /// Out-of-range is a no-op: the caller's cache may be shorter than
    /// the shared `TraceCtx` list when it was built without `rd_virtuals`.
    pub fn recall_op(&mut self, i: usize, v: OpRef) {
        if let Some(slot) = self.virtuals_ptr_cache.get_mut(i) {
            *slot = Some(v);
        }
        if let Some(slot) = self.virtuals_int_cache.get_mut(i) {
            *slot = Some(v);
        }
    }

    /// The applying reader's object wins: it is the one the `NEW` carries,
    /// read through the root the collector maintains. Only a recording-only
    /// reader answers from the address a later decode parked.
    pub fn get_concrete_ptr(&self, i: usize) -> Option<majit_ir::GcRef> {
        // The collector writes the forwarded address into `concrete_roots`.
        // A zero slot is unpublished; read the allocation-time cache.
        match self.concrete_roots.get(i) {
            Some(0) | None => self.concrete_ptr_cache.get(i).copied().flatten(),
            Some(address) => Some(majit_ir::GcRef(*address as usize)),
        }
    }

    pub fn set_concrete_ptr(&mut self, i: usize, v: majit_ir::GcRef) {
        if i < self.concrete_ptr_cache.len() {
            self.concrete_ptr_cache[i] = Some(v);
        }
        // `resume.py virtuals_cache` is the object the collector traces.
        // Seed `concrete_roots` so `get_concrete_ptr` re-reads the
        // forwarded address after an allocating `_prepare_pendingfields`
        // / `consume_boxes` step, rather than the unrooted copy.
        self.set_concrete_root(i, v.0 as i64);
    }

    pub fn get_concrete_int(&self, i: usize) -> Option<i64> {
        self.concrete_int_cache.get(i).copied().flatten()
    }

    pub fn set_concrete_int(&mut self, i: usize, v: i64) {
        if i < self.concrete_int_cache.len() {
            self.concrete_int_cache[i] = Some(v);
        }
    }
}

fn emit_stroruni_oopspec_call(
    ctx: &mut crate::TraceCtx,
    oopspec: majit_ir::effectinfo::OopSpecIndex,
    args: &[OpRef],
) -> OpRef {
    let cic = ctx
        .callinfocollection
        .as_ref()
        .expect(
            "TraceCtx.callinfocollection missing — bridge-virtual VStr/VUni \
             Concat/Slice materialization requires pyjitpl to populate it \
             (resume.py:1143-1188)",
        )
        .clone();
    let (calldescr, func) = cic.callinfo_for_oopspec(oopspec);
    let calldescr = calldescr.expect("callinfo_for_oopspec missing entry for VStr/VUni oopspec");
    let func_const = ctx.const_int(func as i64);
    let mut call_args = Vec::with_capacity(1 + args.len());
    call_args.push(func_const);
    call_args.extend_from_slice(args);
    // resume.py concat_strings `execute_and_record_varargs(CALL_R, ...)`.
    ctx.profiler()
        .count_ops(majit_ir::OpCode::CallR, crate::counters::OPS);
    ctx.profiler()
        .count_ops(majit_ir::OpCode::CallR, crate::counters::RECORDED_OPS);
    ctx.record_op_with_descr(majit_ir::OpCode::CallR, &call_args, calldescr.clone())
}

/// resume.py decode_box parity for fieldnums (i16 tagged): decode one
/// tagged array/field value into its bridge `OpRef` (typed InputArg for TAGBOX,
/// const for TAGINT/TAGCONST, recursively materialized virtual for TAGVIRTUAL).
/// The concrete `executor.execute(INT_ADD, a, b)` would have produced for the
/// two `INT_ADD`s a resume decode synthesizes.
///
/// Neither has a live execution behind it, so the sum exists exactly when both
/// operands decoded to constants — which is also the only case
/// `TraceCtx::execute_and_record` folds in. It is evaluated through the same
/// executor row the funnel folds with, so the two cannot disagree.
fn int_add_concrete(a: OpRef, b: OpRef) -> Option<majit_ir::Value> {
    let (majit_ir::Value::Int(x), majit_ir::Value::Int(y)) =
        (a.inline_const_to_value()?, b.inline_const_to_value()?)
    else {
        return None;
    };
    crate::executor::execute_binary_int_const(majit_ir::OpCode::IntAdd, x, y)
        .map(majit_ir::Value::Int)
}

pub fn decode_fieldnum(
    ctx: &mut crate::TraceCtx,
    tagged: i16,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    resume_data: &crate::jit_state::ResumeDataResult,
    cache: &mut BridgeVirtualCache<'_>,
) -> OpRef {
    use majit_ir::resumedata::{TAG_CONST_OFFSET, TAGBOX, TAGCONST, TAGINT, TAGVIRTUAL, untag};
    // resume.py `decode_box` dispatches purely on the tag bits;
    // it has no UNINITIALIZED case. The UNINITIALIZED skip lives in
    // the callers (e.g. VArrayStructInfo.allocate, resume.py),
    // so this decoder mirrors `decode_box` exactly — an UNINITIALIZED
    // tag reaching here falls into the TAGCONST arm and fails loud on
    // the out-of-range const index, matching upstream's IndexError.
    let (val, tagbits) = untag(tagged);
    match tagbits {
        TAGBOX => {
            // resume.py decode_box parity:
            //   if num < 0: num += len(liveboxes)
            //   return self.liveboxes[num]
            // The returned Box object carries `box.type` intrinsically
            // (history.py:220). For the bridge tracer, those liveboxes
            // are the bridge's `InputArg{Int,Ref,Float}` slots, so we
            // mint the typed `OpRef::input_arg_typed` variant matching
            // `fail_arg_types[idx]` rather than a bare untyped raw
            // OpRef — variant-aware Eq (`OpRef`'s `PartialEq` in
            // `majit-ir/src/resoperation.rs`) requires
            // the optimizer/heap-cache key to be the same typed variant
            // the bridge inputarg list produces.
            let idx = if val < 0 {
                val + resume_data.num_failargs
            } else {
                val
            };
            // resume.py `box = self.liveboxes[num]` — direct
            // indexing, IndexError on out-of-range. Encoder /
            // decoder asymmetry is a bug, not a silent fallback;
            // mirror the upstream fail-loud contract.
            let tp = *resume_data
                .fail_arg_types
                .get(idx as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "decode_fieldnum TAGBOX out-of-range: idx={} num_failargs={} \
                         fail_arg_types.len()={} (encoder/decoder mismatch — see \
                         resume.py:1245-1264 decode_box)",
                        idx,
                        resume_data.num_failargs,
                        resume_data.fail_arg_types.len()
                    )
                });
            OpRef::input_arg_typed(idx as u32, tp)
        }
        TAGINT => ctx.const_int(val as i64),
        TAGCONST => {
            // resume.py decode_box parity:
            //   if tag == TAGCONST:
            //       if tagged_eq(tagged, NULLREF):
            //           box = CONST_NULL
            //       else:
            //           box = self.consts[num - TAG_CONST_OFFSET]
            if tagged == majit_ir::resumedata::NULLREF {
                return ctx.const_null();
            }
            let ci = (val - TAG_CONST_OFFSET) as usize;
            // resume.py `box = self.consts[num - TAG_CONST_OFFSET]`
            // — direct indexing, fail-fast on out-of-range (mirrors
            // Python IndexError; never silently substitutes).
            // compile.py `ResumeGuardDescr` storage — read off
            // the shared Arc so the bridge tracer observes the
            // same pool the GC walker updates.
            let storage = resume_data
                .storage
                .as_ref()
                .expect("resume_data.storage missing");
            let c = storage.rd_consts().unwrap_or(&[])[ci];
            match c.get_type() {
                majit_ir::Type::Ref => ctx.const_ref(c.getref_base().as_usize() as i64),
                majit_ir::Type::Float => ctx.const_float(c.getfloatstorage()),
                _ => ctx.const_int(c.getint()),
            }
        }
        TAGVIRTUAL => {
            // resume.py assign_number_to_virtual nested virtuals are numbered negatively;
            // getvirtual resolves them via Python negative list indexing
            // into rd_virtuals (resume.py:951-954).
            let vidx = if val < 0 {
                (rd_virtuals.map_or(0, |v| v.len()) as i32 + val) as usize
            } else {
                val as usize
            };
            materialize_bridge_virtual(ctx, vidx, rd_virtuals, resume_data, cache)
        }
        _ => OpRef::NONE,
    }
}

/// resume.py `decode_ref` / `decode_int` / `decode_float`, concrete half.
///
/// The box reader reads a TAGBOX operand straight out of the deadframe
/// (`cpu.get_ref_value(self.deadframe, num)`), so an input-arg `OpRef`
/// resolves against the guard's fail values. Everything else — a pool
/// constant, or a virtual this walk has just allocated — carries its concrete
/// on the shadow the recording half already stamped.
fn operand_concrete(
    ctx: &crate::TraceCtx,
    cache: &BridgeVirtualCache<'_>,
    opref: OpRef,
) -> Option<majit_ir::Value> {
    // `decode_ref` reads a TAGBOX operand out of the deadframe and nothing
    // else, so the guard's fail values win for an input arg — a shadow carried
    // on the OpRef describes some earlier run of the same slot.
    if let Some(address) = cache.concrete_root_of(opref) {
        return Some(majit_ir::Value::Ref(majit_ir::GcRef(address as usize)));
    }
    if opref.is_input_arg() {
        let index = opref.raw() as usize;
        return Some(match opref {
            // Through the roots, not through the copy: this walk allocates,
            // and the address a ref failarg held when the guard failed is not
            // the address it holds after a `bh_new` collected.
            OpRef::InputArgRef(_) => {
                majit_ir::Value::Ref(majit_ir::GcRef(cache.fail_ref_root(index)? as usize))
            }
            OpRef::InputArgFloat(_) => {
                majit_ir::Value::Float(f64::from_bits(*cache.fail_values().get(index)? as u64))
            }
            _ => majit_ir::Value::Int(*cache.fail_values().get(index)?),
        });
    }
    ctx.box_value(opref)
}

/// resume.py `ResumeDataBoxReader.setfield` applying half.
///
/// `metainterp.execute_setfield_gc` executes the store as well as recording
/// it, dispatching on `descr.is_pointer_field()` / `is_float_field()` — the
/// same three-way split `ResumeDataDirectReader.setfield` makes over
/// `cpu.bh_setfield_gc_{r,f,i}`.
///
/// Returns `false` when the store cannot be applied: a target or value whose
/// concrete this walk never learned, or a value whose bank disagrees with the
/// field's. A recorded write whose heap half did not happen would leave the
/// bridge reading state the interpreter does not hold, so the caller must fail
/// the whole entry rather than carry on.
fn apply_setfield(
    ctx: &crate::TraceCtx,
    cache: &BridgeVirtualCache<'_>,
    allocator: &dyn crate::resume::BlackholeAllocator,
    struct_op: OpRef,
    value_op: OpRef,
    info: &majit_ir::FieldDescrInfo,
) -> bool {
    use majit_ir::Value;
    let Some(Value::Ref(target)) = operand_concrete(ctx, cache, struct_op) else {
        return false;
    };
    let Some(value) = operand_concrete(ctx, cache, value_op) else {
        return false;
    };
    let target = target.as_usize() as i64;
    match (info.field_type, value) {
        (majit_ir::Type::Ref, Value::Ref(r)) => {
            allocator.bh_setfield_gc_r(target, r.as_usize() as i64, info)
        }
        (majit_ir::Type::Float, Value::Float(f)) => {
            allocator.bh_setfield_gc_f(target, f.to_bits() as i64, info)
        }
        (majit_ir::Type::Int, Value::Int(i)) => allocator.bh_setfield_gc_i(target, i, info),
        _ => return false,
    }
    true
}

/// resume.py `ResumeDataBoxReader.setarrayitem` applying half.
///
/// `_prepare_pendingfields` dispatches `itemindex >= 0` to
/// `execute_setarrayitem_gc` — the same three-way split
/// `ResumeDataDirectReader.setarrayitem` makes over
/// `cpu.bh_setarrayitem_gc_{r,f,i}`. Without this twin the applying
/// reader declined every array pending field, so a guard that deferred
/// a `SETARRAYITEM_GC` could only blackhole.
fn apply_setarrayitem(
    ctx: &crate::TraceCtx,
    cache: &BridgeVirtualCache<'_>,
    allocator: &dyn crate::resume::BlackholeAllocator,
    array_op: OpRef,
    index: i32,
    value_op: OpRef,
    descr: &majit_ir::DescrRef,
) -> bool {
    use majit_ir::Value;
    let Some(ad) = descr.as_array_descr() else {
        return false;
    };
    let Some(Value::Ref(array)) = operand_concrete(ctx, cache, array_op) else {
        return false;
    };
    let Some(value) = operand_concrete(ctx, cache, value_op) else {
        return false;
    };
    let array = array.as_usize() as i64;
    let index = index as usize;
    if ad.is_array_of_pointers() {
        let Value::Ref(r) = value else {
            return false;
        };
        allocator.bh_setarrayitem_gc_r(array, index, r.as_usize() as i64, descr);
    } else if ad.is_array_of_floats() {
        let Value::Float(f) = value else {
            return false;
        };
        allocator.bh_setarrayitem_gc_f(array, index, f.to_bits() as i64, descr);
    } else {
        let Value::Int(i) = value else {
            return false;
        };
        allocator.bh_setarrayitem_gc_i(array, index, i, descr);
    }
    true
}

pub fn materialize_bridge_virtual(
    ctx: &mut crate::TraceCtx,
    vidx: usize,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    resume_data: &crate::jit_state::ResumeDataResult,
    cache: &mut BridgeVirtualCache<'_>,
) -> OpRef {
    use majit_ir::OpCode;
    use majit_ir::resumedata::{TAG_CONST_OFFSET, TAGBOX, TAGCONST, TAGINT, TAGVIRTUAL, untag};

    // resume.py VirtualCache: list caches indexed by virtual number
    // (ptr and int banks). This bridge helper is still OpRef-typed, so it
    // probes both banks before allocating.
    if let Some(cached) = cache.get_any(vidx) {
        // resume.py `getvirtual_ptr`: cache hit returns the box. Publish
        // it so a later reader (`consume_boxes` after `_prepare`) stores
        // the same OpRef. Do not allocate or setfields again.
        ctx.remember_bridge_virtual_op(vidx, cached);
        return cached;
    }
    if let Some(cached) = ctx.bridge_virtual_op(vidx) {
        // Same `virtuals_cache` the first reader filled. `allocate()`
        // already ran setfields; a second apply would SETFIELD a second
        // `wrappeditems` array against the live object.
        cache.recall_op(vidx, cached);
        return cached;
    }

    // resume.py assert self.virtuals_cache is not None — a TAGVIRTUAL in
    // the stream guarantees rd_virtuals is present; None is an encoder bug.
    let virtuals = rd_virtuals.expect("materialize_bridge_virtual: rd_virtuals is None");
    // resume.py:951 self.rd_virtuals[index] — direct indexing, IndexError on
    // an out-of-range virtual number is a bug, not a silent NONE fallback.
    let entry = &virtuals[vidx];

    // resume.py VirtualInfo dispatch by virtual kind.
    // RPython: rd_virtuals[index].allocate(self, index) — polymorphic on
    // the AbstractVirtualInfo subclass. Rust equivalent: match on
    // RdVirtualInfo enum variant.

    /// resume.py AbstractVirtualStructInfo.setfields helper.
    /// Walks fielddescrs in lock-step with fieldnums, decoding each
    /// fieldnum and emitting SETFIELD_GC.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    fn setfields(
        ctx: &mut crate::TraceCtx,
        struct_op: OpRef,
        fielddescrs: &[majit_ir::FieldDescrInfo],
        fieldnums: &[i16],
        parent_descr: majit_ir::DescrRef,
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        resume_data: &crate::jit_state::ResumeDataResult,
        cache: &mut BridgeVirtualCache<'_>,
    ) -> bool {
        // resume.py setfields — range(len(fielddescrs)), index
        // fieldnums[i]. The len-equality assert (resume.py) is in
        // debug_prints, not this allocate path: a short fieldnums raises
        // IndexError here, a longer one is ignored.
        for i in 0..fielddescrs.len() {
            let fd_info = &fielddescrs[i];
            let fnum = fieldnums[i];
            if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                continue;
            }
            let value = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
            if value.is_none() {
                // The recording reader can leave a field it could not decode to
                // whatever the allocation already holds; the applying one
                // cannot, because that field is the heap the bridge resumes
                // against.
                if cache.allocator().is_some() {
                    return false;
                }
                continue;
            }
            // resume.py self.setfields → decoder.setfield(struct,
            // fieldnum, self.fielddescrs[i]): reuse the parent SizeDescr's
            // live FieldDescr (canonical immutable / quasi-immutable /
            // ei_index) rather than reconstructing a partial copy. The row
            // is `fielddescrs[i]`; an offset alone does not name it when two
            // fields overlay one word.
            let field_descr =
                crate::resume::live_field_descr_for_resume(Some(&parent_descr), i, fd_info)
                    .unwrap_or_else(|| {
                        majit_ir::descr::field_descr_from_parent_by_offset(
                            &parent_descr,
                            fd_info.offset,
                        )
                    });
            // resume.py allocate_with_vtable materializer operations use
            // `execute_and_record`.
            ctx.profiler()
                .count_ops(OpCode::SetfieldGc, crate::counters::OPS);
            ctx.profiler()
                .count_ops(OpCode::SetfieldGc, crate::counters::RECORDED_OPS);
            // `execute_and_record` reaches `_record_helper`, whose
            // `invalidate_caches` is `mark_escaped` alone for a store
            // (`clear_caches_not_necessary` answers true for SETFIELD_GC).
            // Recording the store without it leaves a struct that
            // `allocate_struct` had just flagged unescaped still flagged so
            // after it was written into one that is not, and the walk that
            // follows keeps folding its fields across calls that can reach it.
            ctx.heap_cache_mut()
                .mark_escaped(OpCode::SetfieldGc, None, &[struct_op, value]);
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[struct_op, value], field_descr.clone());
            if let Some(allocator) = cache.allocator()
                && !apply_setfield(ctx, cache, allocator, struct_op, value, fd_info)
            {
                return false;
            }
            // Bridge virtual rematerialisation — `upd.setfield(valuebox)`
            // parity: cache stores the Box identity (`value` OpRef).
            // Cache-hit readers resolve the intrinsic value via
            // `box_value(cached)` at hit time (covering const pool,
            // standard-virtualizable shadow, and the frontend object's
            // `value` field) — non-Const operands whose runtime concrete
            // was stamped at the original record site (or threaded from
            // the parent guard's fail_args via `set_opref_concrete`)
            // surface through that `value` field; unstamped operands
            // return `None` so the downstream sanity check skips.
            ctx.heapcache_setfield_cached(struct_op, fd_info.index, value);
        }
        true
    }

    match entry.as_ref() {
        // resume.py VirtualInfo.allocate
        majit_ir::RdVirtualInfo::VirtualInfo {
            descr,
            fielddescrs,
            fieldnums,
            ..
        } => {
            let Some(size_descr) = descr.clone() else {
                return OpRef::NONE;
            };
            // resume.py decoder.allocate_with_vtable(descr=self.descr)
            // is `metainterp.execute_new_with_vtable` → `execute_and_record`,
            // which allocates then `history.record(..., resvalue)`.
            ctx.profiler()
                .count_ops(OpCode::NewWithVtable, crate::counters::OPS);
            ctx.profiler()
                .count_ops(OpCode::NewWithVtable, crate::counters::RECORDED_OPS);
            let allocated = match cache.allocator() {
                Some(allocator) => {
                    let vtable = size_descr.as_size_descr().map_or(0, |sd| sd.vtable());
                    Some(allocator.allocate_with_vtable(&size_descr, vtable))
                }
                None => None,
            };
            let value = allocated
                .filter(|&ptr| ptr != 0)
                .map(|ptr| majit_ir::Value::Ref(majit_ir::GcRef(ptr as usize)));
            let new_op = ctx.record_op_with_descr_value(
                OpCode::NewWithVtable,
                &[],
                size_descr.clone(),
                value,
            );
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py decoder.virtuals_cache.set_ptr(index, struct)
            cache.set_ptr(vidx, new_op);
            ctx.remember_bridge_virtual_op(vidx, new_op);
            if let Some(ptr) = allocated {
                if ptr == 0 {
                    return OpRef::NONE;
                }
                cache.set_concrete_root(vidx, ptr);
            }
            // resume.py self.setfields(decoder, struct)
            if !setfields(
                ctx,
                new_op,
                fielddescrs,
                fieldnums,
                size_descr,
                rd_virtuals,
                resume_data,
                cache,
            ) {
                return OpRef::NONE;
            }
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VirtualInfo → OpRef::from_raw({})",
                    vidx,
                    new_op.raw(),
                );
            }
            new_op
        }
        // resume.py VStructInfo.allocate
        majit_ir::RdVirtualInfo::VStructInfo {
            typedescr,
            fielddescrs,
            fieldnums,
            ..
        } => {
            let Some(struct_descr) = typedescr.clone() else {
                return OpRef::NONE;
            };
            // resume.py decoder.allocate_struct(self.typedescr)
            // is `metainterp.execute_new` → `execute_and_record`, which
            // allocates then `history.record(..., resvalue)`. The concrete
            // lives on the recorded OpRef (`History._make_op`) so the object
            // the trace names and the object the interpreter resumes against
            // are the same one, and `walk_active_trace_refs` forwards it.
            ctx.profiler().count_ops(OpCode::New, crate::counters::OPS);
            ctx.profiler()
                .count_ops(OpCode::New, crate::counters::RECORDED_OPS);
            let allocated = match cache.allocator() {
                Some(allocator) => Some(allocator.bh_new(&struct_descr)),
                None => None,
            };
            let value = allocated
                .filter(|&ptr| ptr != 0)
                .map(|ptr| majit_ir::Value::Ref(majit_ir::GcRef(ptr as usize)));
            let new_op =
                ctx.record_op_with_descr_value(OpCode::New, &[], struct_descr.clone(), value);
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py decoder.virtuals_cache.set_ptr(index, struct), which
            // its own comment requires BEFORE the fields are filled: the cache
            // is what keeps the object reachable across the allocations
            // `setfields` may itself perform.
            cache.set_ptr(vidx, new_op);
            ctx.remember_bridge_virtual_op(vidx, new_op);
            if let Some(ptr) = allocated {
                if ptr == 0 {
                    return OpRef::NONE;
                }
                cache.set_concrete_root(vidx, ptr);
            }
            // resume.py self.setfields(decoder, struct)
            if !setfields(
                ctx,
                new_op,
                fielddescrs,
                fieldnums,
                struct_descr,
                rd_virtuals,
                resume_data,
                cache,
            ) {
                return OpRef::NONE;
            }
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VStructInfo → OpRef::from_raw({})",
                    vidx,
                    new_op.raw(),
                );
            }
            new_op
        }
        // resume.py AbstractVArrayInfo.allocate (clear=True or False)
        majit_ir::RdVirtualInfo::VArrayInfoClear {
            fieldnums,
            kind,
            arraydescr,
            ..
        }
        | majit_ir::RdVirtualInfo::VArrayInfoNotClear {
            fieldnums,
            kind,
            arraydescr,
            ..
        } => {
            let clear = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VArrayInfoClear { .. }
            );
            let kind = *kind;
            let length = fieldnums.len();
            let len_ref = ctx.const_int(length as i64);
            // resume.py decoder.allocate_array(length, arraydescr, self.clear)
            let alloc_opcode = if clear {
                OpCode::NewArrayClear
            } else {
                OpCode::NewArray
            };
            // resume.py AbstractVArrayInfo.__init__ asserts arraydescr is
            // not None; resume.py allocate reads self.arraydescr directly.
            let array_descr = arraydescr.clone().expect("VArrayInfo: arraydescr is None");
            ctx.profiler().count_ops(alloc_opcode, crate::counters::OPS);
            ctx.profiler()
                .count_ops(alloc_opcode, crate::counters::RECORDED_OPS);
            // resume.py decoder.allocate_array → execute_new_array[_clear]
            // → `execute_and_record` allocates then records with resvalue.
            let allocated = match cache.allocator() {
                Some(allocator) => Some(if clear {
                    allocator.bh_new_array_clear(length, &array_descr)
                } else {
                    allocator.bh_new_array(length, &array_descr)
                }),
                None => None,
            };
            let value = allocated
                .filter(|&ptr| ptr != 0)
                .map(|ptr| majit_ir::Value::Ref(majit_ir::GcRef(ptr as usize)));
            let new_op = ctx.record_op_with_descr_value(
                alloc_opcode,
                &[len_ref],
                array_descr.clone(),
                value,
            );
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py decoder.virtuals_cache.set_ptr(index, array)
            cache.set_ptr(vidx, new_op);
            ctx.remember_bridge_virtual_op(vidx, new_op);
            if let Some(ptr) = allocated {
                if ptr == 0 {
                    return OpRef::NONE;
                }
                cache.set_concrete_root(vidx, ptr);
            }
            // resume.py:656-670 element loop: dispatch by arraydescr kind
            // NB. the check for the kind of array elements is moved out of the loop
            let set_opcode = match kind {
                0 => OpCode::SetarrayitemGc, // arraydescr.is_array_of_pointers()
                2 => OpCode::SetarrayitemGc, // arraydescr.is_array_of_floats() — TODO: SetarrayitemRaw/Float
                _ => OpCode::SetarrayitemGc, // int
            };
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                let value = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
                if value.is_none() {
                    if cache.allocator().is_some() {
                        return OpRef::NONE;
                    }
                    continue;
                }
                let idx_ref = ctx.const_int(i as i64);
                // resume.py:660/665/670 setarrayitem_{ref,float,int}
                ctx.profiler().count_ops(set_opcode, crate::counters::OPS);
                ctx.profiler()
                    .count_ops(set_opcode, crate::counters::RECORDED_OPS);
                ctx.record_op_with_descr(
                    set_opcode,
                    &[new_op, idx_ref, value],
                    array_descr.clone(),
                );
                if let Some(allocator) = cache.allocator()
                    && !apply_setarrayitem(
                        ctx,
                        cache,
                        allocator,
                        new_op,
                        i as i32,
                        value,
                        &array_descr,
                    )
                {
                    return OpRef::NONE;
                }
            }
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VArrayInfo(clear={}) → OpRef::from_raw({})",
                    vidx,
                    clear,
                    new_op.raw(),
                );
            }
            new_op
        }
        // resume.py VArrayStructInfo.allocate
        majit_ir::RdVirtualInfo::VArrayStructInfo {
            arraydescr,
            fielddescrs,
            size,
            fieldnums,
            ..
        } => {
            let len_ref = ctx.const_int(*size as i64);
            // resume.py: array = decoder.allocate_array(self.size,
            // self.arraydescr, clear=True) — uses the live `self.arraydescr`
            // directly.
            let array_descr = arraydescr
                .as_ref()
                .expect("VArrayStructInfo: arraydescr is None");
            ctx.profiler()
                .count_ops(OpCode::NewArrayClear, crate::counters::OPS);
            ctx.profiler()
                .count_ops(OpCode::NewArrayClear, crate::counters::RECORDED_OPS);
            let new_op =
                ctx.record_op_with_descr(OpCode::NewArrayClear, &[len_ref], array_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py: decoder.virtuals_cache.set_ptr(index, array)
            cache.set_ptr(vidx, new_op);
            ctx.remember_bridge_virtual_op(vidx, new_op);
            // resume.py:752-759:
            //   p = 0
            //   for i in range(self.size):
            //       for j in range(len(self.fielddescrs)):
            //           num = self.fieldnums[p]
            //           if not tagged_eq(num, UNINITIALIZED):
            //               decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
            //           p += 1
            let num_fields = fielddescrs.len();
            // resume.py:752-759 reads exactly size × len(fielddescrs) entries
            // via self.fieldnums[p] with no length-equality check: a short
            // fieldnums is an out-of-bounds error here (IndexError parity), a
            // longer one leaves its tail unread.
            let mut p = 0;
            for i in 0..*size {
                for fielddescr in fielddescrs.iter().take(num_fields) {
                    let fnum = fieldnums[p];
                    p += 1;
                    if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                        continue;
                    }
                    let value = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
                    if value.is_none() {
                        continue;
                    }
                    let idx_ref = ctx.const_int(i as i64);
                    // resume.py: decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
                    ctx.record_op_with_descr(
                        OpCode::SetinteriorfieldGc,
                        &[new_op, idx_ref, value],
                        fielddescr.clone(),
                    );
                }
            }
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VArrayStructInfo → OpRef::from_raw({})",
                    vidx,
                    new_op.raw(),
                );
            }
            new_op
        }
        // resume.py VRawBufferInfo.allocate_int
        majit_ir::RdVirtualInfo::VRawBufferInfo {
            func,
            size,
            offsets,
            descrs,
            fieldnums,
        } => {
            // resume.py: buffer = decoder.allocate_raw_buffer(self.func, self.size)
            // resume.py: ResumeDataBoxReader.allocate_raw_buffer →
            //   execute_and_record_varargs(rop.CALL_I, [ConstInt(func), ConstInt(size)], calldescr)
            let func_ref = ctx.const_int(*func);
            let size_ref = ctx.const_int(*size as i64);
            // resume.py allocate_raw_buffer: calldescr comes from the shared
            // callinfocollection, not a freshly minted synthetic descr. The
            // func is NOT taken from callinfo_for_oopspec (resume.py:1127-1130:
            // several malloc variants share the oopspec), so only the calldescr
            // is read from the CIC and *func from the VRawBufferInfo is kept.
            let cic = ctx
                .callinfocollection
                .as_ref()
                .expect(
                    "TraceCtx.callinfocollection missing — bridge-virtual \
                     VRawBufferInfo materialization requires pyjitpl to populate \
                     it (resume.py:1124-1126)",
                )
                .clone();
            // resume.py:1126: calldescr, _ = cic.callinfo_for_oopspec(
            //   OS_RAW_MALLOC_VARSIZE_CHAR). callinfo_for_oopspec returns
            // (None, 0) on a missing entry (effectinfo.py) — no
            // lookup-time check; the calldescr is used directly.
            let (calldescr, _) =
                cic.callinfo_for_oopspec(majit_ir::descr::OopSpecIndex::RawMallocVarsizeChar);
            // resume.py:1131-1132: execute_and_record_varargs(CALL_I,
            //   [func, size], calldescr). A missing entry surfaces here, as the
            // calldescr is consumed by the CALL_I op, not as a separate lookup
            // assertion.
            ctx.profiler()
                .count_ops(OpCode::CallI, crate::counters::OPS);
            ctx.profiler()
                .count_ops(OpCode::CallI, crate::counters::RECORDED_OPS);
            let buffer = ctx.record_op_with_descr(
                OpCode::CallI,
                &[func_ref, size_ref],
                calldescr
                    .cloned()
                    .expect("OS_RAW_MALLOC_VARSIZE_CHAR calldescr (callinfocollection)"),
            );
            // resume.py: decoder.virtuals_cache.set_int(index, buffer)
            cache.set_int(vidx, buffer);
            ctx.remember_bridge_virtual_op(vidx, buffer);
            // resume.py:705-708 iterate by len(self.offsets), indexing
            // self.descrs[i] and self.fieldnums[i] by the same i — a short
            // descrs/fieldnums raises IndexError here (encoder bug), a longer
            // one is ignored. No len-equality assert (VRawBufferInfo has none).
            for i in 0..offsets.len() {
                let off = offsets[i];
                let fnum = fieldnums[i];
                // resume.py VRawBufferStateInfo.allocate_int passes
                // fieldnums[i] straight to setrawbuffer_item with no
                // UNINITIALIZED skip (unlike VArrayStructInfo) — a raw buffer
                // is fully written by the encoder.
                // resume.py: itembox = self.decode_box(fieldnum, kind).
                // `decode_box` always returns a box (no UNINITIALIZED case),
                // so the store is unconditional, matching setrawbuffer_item.
                let item = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
                // resume.py: setrawbuffer_item (direct reader).
                // Dispatches pointer/float/int via arraydescr — all types allowed.
                let di = &descrs[i];
                let tp = match di.item_type {
                    0 => majit_ir::Type::Ref,
                    2 => majit_ir::Type::Float,
                    _ => majit_ir::Type::Int,
                };
                let store_descr = (cache.mint_raw_array_descr)(
                    di.base_size,
                    di.item_size,
                    di.len_offset,
                    tp,
                    di.is_signed,
                );
                let offset_ref = ctx.const_int(off);
                ctx.profiler()
                    .count_ops(OpCode::RawStore, crate::counters::OPS);
                ctx.profiler()
                    .count_ops(OpCode::RawStore, crate::counters::RECORDED_OPS);
                ctx.record_op_with_descr(
                    OpCode::RawStore,
                    &[buffer, offset_ref, item],
                    store_descr,
                );
            }
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VRawBufferInfo(func={:#x}, size={}) → OpRef::from_raw({})",
                    vidx,
                    func,
                    size,
                    buffer.raw(),
                );
            }
            buffer
        }
        // resume.py VRawSliceInfo.allocate_int
        majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
            // resume.py:724: assert len(self.fieldnums) == 1
            assert!(
                fieldnums.len() == 1,
                "VRawSliceInfo must have exactly 1 fieldnum"
            );
            // resume.py: base_buffer = decoder.decode_int(self.fieldnums[0])
            let base_buffer = decode_fieldnum(ctx, fieldnums[0], rd_virtuals, resume_data, cache);
            // resume.py: buffer = decoder.int_add_const(base_buffer, self.offset)
            let offset_ref = ctx.const_int(*offset);
            // `INT_ADD` is always-pure, so the funnel neither reads
            // `last_exc_value` nor consults the cpu — the integer row folds
            // from the operands alone.
            let cpu = crate::cpu::default_cpu();
            let buffer = ctx.execute_and_record(
                Some(cpu.as_ref()),
                OpCode::IntAdd,
                None,
                &[base_buffer, offset_ref],
                int_add_concrete(base_buffer, offset_ref),
                0,
            );
            // resume.py: decoder.virtuals_cache.set_int(index, buffer)
            cache.set_int(vidx, buffer);
            ctx.remember_bridge_virtual_op(vidx, buffer);
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={vidx} VRawSliceInfo(offset={offset}) → {buffer:?}",
                );
            }
            buffer
        }
        // resume.py VStrPlainInfo.allocate / resume.py
        // VUniPlainInfo.allocate — `ResumeDataBoxReader.allocate_string /
        // allocate_unicode` followed by `string_setitem` / `unicode_setitem`
        // per character.
        //
        //     length = len(self.fieldnums)
        //     string = decoder.allocate_string(length)        # NEWSTR
        //     decoder.virtuals_cache.set_ptr(index, string)
        //     for i in range(length):
        //         charnum = self.fieldnums[i]
        //         if not tagged_eq(charnum, UNINITIALIZED):
        //             decoder.string_setitem(string, i, charnum)  # STRSETITEM
        //     return string
        majit_ir::RdVirtualInfo::VStrPlainInfo { fieldnums }
        | majit_ir::RdVirtualInfo::VUniPlainInfo { fieldnums } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniPlainInfo { .. }
            );
            let length = fieldnums.len();
            let length_ref = ctx.const_int(length as i64);
            let (alloc_opcode, set_opcode) = if is_unicode {
                (OpCode::Newunicode, OpCode::Unicodesetitem)
            } else {
                (OpCode::Newstr, OpCode::Strsetitem)
            };
            // resume.py: string = decoder.allocate_string(length)
            ctx.profiler().count_ops(alloc_opcode, crate::counters::OPS);
            ctx.profiler()
                .count_ops(alloc_opcode, crate::counters::RECORDED_OPS);
            let string = ctx.record_op(alloc_opcode, &[length_ref]);
            // resume.py: decoder.virtuals_cache.set_ptr(index, string)
            cache.set_ptr(vidx, string);
            ctx.remember_bridge_virtual_op(vidx, string);
            // resume.py: string_setitem for each filled char.
            for (i, &charnum) in fieldnums.iter().enumerate() {
                if charnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                // resume.py ResumeDataBoxReader.string_setitem:
                //   charbox = self.decode_box(charnum, INT)
                //   execute_and_record(rop.STRSETITEM, string, ConstInt(index), charbox)
                let charbox = decode_fieldnum(ctx, charnum, rd_virtuals, resume_data, cache);
                if charbox.is_none() {
                    continue;
                }
                let idx_ref = ctx.const_int(i as i64);
                ctx.profiler().count_ops(set_opcode, crate::counters::OPS);
                ctx.profiler()
                    .count_ops(set_opcode, crate::counters::RECORDED_OPS);
                ctx.record_op(set_opcode, &[string, idx_ref, charbox]);
            }
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} V{}PlainInfo(length={}) → OpRef::from_raw({})",
                    vidx,
                    if is_unicode { "Uni" } else { "Str" },
                    length,
                    string.raw(),
                );
            }
            string
        }
        // resume.py VStrConcatInfo.allocate / resume.py
        // VUniConcatInfo.allocate:
        //
        //     left, right = self.fieldnums
        //     string = decoder.concat_strings(left, right)   # CALL_R(OS_STR_CONCAT)
        //     decoder.virtuals_cache.set_ptr(index, string)
        //
        // `ResumeDataBoxReader.concat_strings` at resume.py:
        //
        //     cic = self.metainterp.staticdata.callinfocollection
        //     calldescr, func = cic.callinfo_for_oopspec(OS_STR_CONCAT)
        //     str1box = self.decode_box(str1num, REF)
        //     str2box = self.decode_box(str2num, REF)
        //     execute_and_record_varargs(CALL_R, [ConstInt(func), str1box, str2box], calldescr)
        majit_ir::RdVirtualInfo::VStrConcatInfo { fieldnums, .. }
        | majit_ir::RdVirtualInfo::VUniConcatInfo { fieldnums, .. } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniConcatInfo { .. }
            );
            assert_eq!(
                fieldnums.len(),
                2,
                "VStr/VUniConcatInfo must have exactly 2 fieldnums (left, right)"
            );
            let left = decode_fieldnum(ctx, fieldnums[0], rd_virtuals, resume_data, cache);
            let right = decode_fieldnum(ctx, fieldnums[1], rd_virtuals, resume_data, cache);
            let oopspec = if is_unicode {
                majit_ir::effectinfo::OopSpecIndex::UniConcat
            } else {
                majit_ir::effectinfo::OopSpecIndex::StrConcat
            };
            let string = emit_stroruni_oopspec_call(ctx, oopspec, &[left, right]);
            cache.set_ptr(vidx, string);
            ctx.remember_bridge_virtual_op(vidx, string);
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} V{}ConcatInfo → OpRef::from_raw({})",
                    vidx,
                    if is_unicode { "Uni" } else { "Str" },
                    string.raw(),
                );
            }
            string
        }
        // resume.py VStrSliceInfo.allocate / resume.py
        // VUniSliceInfo.allocate:
        //
        //     largerstr, start, length = self.fieldnums
        //     string = decoder.slice_string(largerstr, start, length)
        //     decoder.virtuals_cache.set_ptr(index, string)
        //
        // `ResumeDataBoxReader.slice_string` at resume.py /
        // `slice_unicode` at resume.py:
        //
        //     cic = self.metainterp.staticdata.callinfocollection
        //     calldescr, func = cic.callinfo_for_oopspec(OS_STR_SLICE)
        //     strbox = self.decode_box(strnum, REF)
        //     startbox = self.decode_box(startnum, INT)
        //     lengthbox = self.decode_box(lengthnum, INT)
        //     stopbox = execute_and_record(INT_ADD, startbox, lengthbox)
        //     execute_and_record_varargs(CALL_R,
        //         [ConstInt(func), strbox, startbox, stopbox], calldescr)
        majit_ir::RdVirtualInfo::VStrSliceInfo { fieldnums, .. }
        | majit_ir::RdVirtualInfo::VUniSliceInfo { fieldnums, .. } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniSliceInfo { .. }
            );
            assert_eq!(
                fieldnums.len(),
                3,
                "VStr/VUniSliceInfo must have exactly 3 fieldnums (largerstr, start, length)"
            );
            let largerstr = decode_fieldnum(ctx, fieldnums[0], rd_virtuals, resume_data, cache);
            let start = decode_fieldnum(ctx, fieldnums[1], rd_virtuals, resume_data, cache);
            let length = decode_fieldnum(ctx, fieldnums[2], rd_virtuals, resume_data, cache);
            // resume.py:1157-1158 / :1185-1186: stopbox = INT_ADD(startbox, lengthbox)
            // See the `VRawSliceInfo` arm on the cpu and `last_exc_value`
            // arguments an always-pure opcode does not reach.
            let cpu = crate::cpu::default_cpu();
            let stop = ctx.execute_and_record(
                Some(cpu.as_ref()),
                OpCode::IntAdd,
                None,
                &[start, length],
                int_add_concrete(start, length),
                0,
            );
            let oopspec = if is_unicode {
                majit_ir::effectinfo::OopSpecIndex::UniSlice
            } else {
                majit_ir::effectinfo::OopSpecIndex::StrSlice
            };
            let string = emit_stroruni_oopspec_call(ctx, oopspec, &[largerstr, start, stop]);
            cache.set_ptr(vidx, string);
            ctx.remember_bridge_virtual_op(vidx, string);
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} V{}SliceInfo → OpRef::from_raw({})",
                    vidx,
                    if is_unicode { "Uni" } else { "Str" },
                    string.raw(),
                );
            }
            string
        }
        // resume.py/954 getvirtual_ptr direct-indexes rd_virtuals[index];
        // the function preamble already documents this as fail-loud ("not a
        // silent NONE fallback"). An Empty hole here means the resume stream
        // tagged a virtual index that was never assigned a real virtual
        // (encoder/decoder asymmetry) — surface it rather than poison the
        // operand chain with OpRef::NONE.
        majit_ir::RdVirtualInfo::Empty => panic!(
            "materialize_bridge_virtual: null rd_virtuals[{vidx}] (resume.py: null rd_virtuals[index])"
        ),
    }
}

/// resume.py `decode_box` symbolic parity for an already-decoded
/// [`majit_ir::resumedata::RebuiltValue`]: mint the bridge `OpRef` (typed `InputArg` for a live box,
/// const for a pooled const, recursively materialized virtual for a virtual).
pub fn rebuilt_value_to_opref(
    ctx: &mut crate::TraceCtx,
    v: &majit_ir::resumedata::RebuiltValue,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    resume_data: &crate::jit_state::ResumeDataResult,
    cache: &mut BridgeVirtualCache<'_>,
) -> OpRef {
    use majit_ir::resumedata::RebuiltValue;
    match v {
        RebuiltValue::Box(n, tp) => OpRef::input_arg_typed(*n as u32, *tp),
        RebuiltValue::Const(c) => match c.get_type() {
            majit_ir::Type::Ref => ctx.const_ref(c.getref_base().as_usize() as i64),
            majit_ir::Type::Float => ctx.const_float(c.getfloatstorage()),
            _ => ctx.const_int(c.getint()),
        },
        RebuiltValue::Virtual(vidx) => {
            materialize_bridge_virtual(ctx, *vidx, rd_virtuals, resume_data, cache)
        }
        RebuiltValue::Unassigned => OpRef::NONE,
    }
}

/// resume.py `_prepare_pendingfields` op-emission: replay one deferred
/// heap write as a bridge-entry `SETFIELD_GC` / `SETARRAYITEM_GC` and seed the
/// heapcache so a later same-slot get folds against it. `item_index < 0` marks
/// a struct field; `>= 0` an array element.
pub fn emit_pending_field_op(
    ctx: &mut crate::TraceCtx,
    target_op: OpRef,
    value_op: OpRef,
    item_index: i32,
    descr: &majit_ir::DescrRef,
    cache: &BridgeVirtualCache<'_>,
) -> bool {
    use majit_ir::OpCode;
    if item_index < 0 {
        // resume.py `_prepare_pendingfields` replays through
        // `execute_and_record`.
        ctx.profiler()
            .count_ops(OpCode::SetfieldGc, crate::counters::OPS);
        ctx.profiler()
            .count_ops(OpCode::SetfieldGc, crate::counters::RECORDED_OPS);
        // `_record_helper`'s `invalidate_caches`, which for a store is
        // `mark_escaped` alone — see the same call in `setfields`.
        ctx.heap_cache_mut()
            .mark_escaped(OpCode::SetfieldGc, None, &[target_op, value_op]);
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[target_op, value_op], descr.clone());
        if let Some(allocator) = cache.allocator() {
            let Some(fd) = descr.as_field_descr() else {
                return false;
            };
            let info = majit_ir::FieldDescrInfo {
                index: descr.index(),
                offset: fd.offset(),
                field_type: fd.field_type(),
                field_size: fd.field_size(),
            };
            if !apply_setfield(ctx, cache, allocator, target_op, value_op, &info) {
                return false;
            }
        }
        ctx.heapcache_setfield_cached(target_op, descr.index(), value_op);
    } else {
        // resume.py `_prepare_pendingfields` → `setarrayitem` →
        // `execute_setarrayitem_gc`. Record the store and, when this
        // is the applying reader, write it through the same
        // `bh_setarrayitem_gc_*` split the direct reader uses.
        let index_op = ctx.const_int(item_index as i64);
        ctx.profiler()
            .count_ops(OpCode::SetarrayitemGc, crate::counters::OPS);
        ctx.profiler()
            .count_ops(OpCode::SetarrayitemGc, crate::counters::RECORDED_OPS);
        ctx.record_op_with_descr(
            OpCode::SetarrayitemGc,
            &[target_op, index_op, value_op],
            descr.clone(),
        );
        if let Some(allocator) = cache.allocator() {
            if !apply_setarrayitem(
                ctx, cache, allocator, target_op, item_index, value_op, descr,
            ) {
                return false;
            }
        }
        ctx.heapcache_setarrayitem(target_op, index_op, descr.index(), value_op);
    }
    true
}

/// resume.py `_prepare_pendingfields` (box-reader flavour): replay the
/// guard's deferred heap writes as bridge-entry `SETFIELD_GC` /
/// `SETARRAYITEM_GC` ops so the compiled bridge observes the same heap state the
/// blackhole would rebuild. Symbolic-only — a consumer that also seeds concrete
/// shadows or routes a magic descr (e.g. an exception channel) wraps this or
/// re-implements the loop with those extensions.
pub fn replay_pending_fields(
    ctx: &mut crate::TraceCtx,
    resume_data: &crate::jit_state::ResumeDataResult,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    cache: &mut BridgeVirtualCache<'_>,
) -> bool {
    let __diag = crate::bridge_diag_enabled();
    let Some(storage) = resume_data.storage.as_ref() else {
        if __diag {
            eprintln!("[replay] storage=None (no pendingfields replayed)");
        }
        return true;
    };
    if __diag {
        eprintln!(
            "[replay] storage=Some pendingfields={} rd_virtuals={} num_failargs={}",
            storage.rd_pendingfields().map_or(0, <[_]>::len),
            storage.rd_virtuals().map_or(0, <[_]>::len),
            resume_data.num_failargs,
        );
    }
    let num_virtuals = storage.rd_virtuals().map_or(0, <[_]>::len);
    for pending in storage.rd_pendingfields().unwrap_or(&[]) {
        let Some(descr) = pending.descr.as_ref() else {
            if __diag {
                eprintln!(
                    "[replay]   pending item_index={} descr=None SKIP",
                    pending.item_index
                );
            }
            if cache.allocator().is_some() {
                return false;
            }
            continue;
        };
        // `resume.py` `_prepare_pendingfields` decodes both operands the same way as frame
        // boxes.
        let rd_consts = storage.rd_consts().unwrap_or(&[]);
        let target = majit_ir::resumedata::decode_tagged_value(
            pending.target_tagged,
            resume_data.num_failargs,
            rd_consts,
            &resume_data.fail_arg_types,
            num_virtuals,
        );
        let value = majit_ir::resumedata::decode_tagged_value(
            pending.value_tagged,
            resume_data.num_failargs,
            rd_consts,
            &resume_data.fail_arg_types,
            num_virtuals,
        );
        if __diag {
            eprintln!(
                "[replay]   pending item_index={} target_tag={} value_tag={} target={:?} value={:?}",
                pending.item_index,
                pending.target_tagged,
                pending.value_tagged,
                std::mem::discriminant(&target),
                std::mem::discriminant(&value),
            );
        }
        let target_op = rebuilt_value_to_opref(ctx, &target, rd_virtuals, resume_data, cache);
        let value_op = rebuilt_value_to_opref(ctx, &value, rd_virtuals, resume_data, cache);
        if target_op.is_none() || value_op.is_none() {
            if __diag {
                eprintln!(
                    "[replay]   -> SKIP (target_op none={} value_op none={})",
                    target_op.is_none(),
                    value_op.is_none()
                );
            }
            if cache.allocator().is_some() {
                return false;
            }
            continue;
        }
        if __diag {
            eprintln!(
                "[replay]   -> emit_pending_field_op target_op={} value_op={}",
                target_op.raw(),
                value_op.raw()
            );
        }
        if !emit_pending_field_op(ctx, target_op, value_op, pending.item_index, descr, cache) {
            return false;
        }
    }
    true
}

/// `pyjitpl.py rebuild_state_after_failure`:
///
/// ```python
/// if vinfo is not None:
///     self.virtualizable_boxes = virtualizable_boxes
/// ```
///
/// `virtualizable_boxes` is what `resume.py consume_virtualizable_boxes`
/// built out of the guard's vable section: the virtualizable itself comes
/// first (`resume.py virtualizable = self.next_ref()`), then
/// `virtualizable.py load_list_of_boxes` reads one box per static field
/// and one per array element — off the LIVE object, which is where the array
/// lengths come from — and returns the list with the virtualizable appended,
/// so `boxes[-1]` is the identity every consumer indexes.
///
/// Without this a bridge traces with no virtualizable bound at all, and the
/// first vable-shaped op it reaches has nothing to resolve against.
///
/// Returns `false` (leaving the ctx untouched) when the stream cannot be
/// turned into a complete shadow: no vable section, a null identity, a
/// length that disagrees with the live object's layout, or an entry whose
/// concrete value the guard did not carry.
pub fn seed_bridge_virtualizable_boxes(
    ctx: &mut crate::TraceCtx,
    info: &std::sync::Arc<crate::virtualizable::VirtualizableInfo>,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    resume_data: &crate::jit_state::ResumeDataResult,
    cache: &mut BridgeVirtualCache<'_>,
    fail_values: &[i64],
) -> bool {
    use majit_ir::resumedata::RebuiltValue;
    use majit_ir::{Const, GcRef, Type, Value};

    fn concrete(v: &RebuiltValue, fail_values: &[i64]) -> Option<Value> {
        let typed = |ty: Type, bits: i64| match ty {
            Type::Int => Value::Int(bits),
            Type::Float => Value::Float(f64::from_bits(bits as u64)),
            Type::Ref => Value::Ref(GcRef(bits as usize)),
            Type::Void => Value::Void,
        };
        match v {
            RebuiltValue::Box(n, ty) => fail_values.get(*n).map(|&bits| typed(*ty, bits)),
            RebuiltValue::Const(Const::Int(i)) => Some(Value::Int(*i)),
            RebuiltValue::Const(Const::Float(f)) => Some(Value::Float(*f)),
            RebuiltValue::Const(Const::Ref(r)) => Some(Value::Ref(*r)),
            // Unassigned has nothing to write. A virtual is materialized
            // below (`ResumeDataBoxReader.allocate` / `getvirtual_ptr`)
            // and its concrete is the allocated object, not a deadframe slot.
            RebuiltValue::Virtual(_) | RebuiltValue::Unassigned => None,
        }
    }

    let Some((identity, slots)) = resume_data.virtualizable_boxes.split_first() else {
        return false;
    };
    // `resume.py virtualizable = self.next_ref()` — the first entry is a
    // ref box holding the virtualizable itself.  Anything else means this
    // guard's vable section does not name the virtualizable (the box was
    // replaced, folded, or never made it into the guard's fail args), and
    // dereferencing whatever value sits there would be a wild read; decline
    // the seed instead, which leaves the guard deopting through the blackhole
    // exactly as it did before — `compile.py compile.giveup()`.
    let Some(identity_value @ Value::Ref(vable_ref)) = concrete(identity, fail_values) else {
        return false;
    };
    let vable_ptr = vable_ref.as_usize() as *const u8;
    if vable_ptr.is_null() {
        return false;
    }
    // virtualizable.py:150-153 `lst = getattr(virtualizable, fieldname); for j
    // in range(len(lst))` — the live object is the authority on how many array
    // boxes the stream carries.
    if !info.can_read_all_array_lengths_from_heap() {
        return false;
    }
    // Upstream never validates the decoded identity here: `rebuild_state_after
    // _failure` takes `virtualizable_boxes[-1]` and goes straight on to
    // `reset_token_gcref` + `synchronize_virtualizable()` (pyjitpl.py),
    // because the box came out of ITS OWN numbering and cannot name anything
    // else.  Pyre's does not carry that guarantee — the state-field vable
    // section is only as good as the numbering that produced it — and the
    // pointer is dereferenced below, so an identity that is not the live object
    // would be a wild read.  Check it and decline instead (`compile.giveup()`,
    // compile.py:27), which is strictly more conservative than upstream: the
    // guard keeps deopting through the blackhole exactly as it did before.
    match ctx.virtualizable_heap_ptr() {
        Some(live) if live == vable_ptr => {}
        Some(_) | None => return false,
    }
    let array_lengths = unsafe { info.read_array_lengths_from_heap(vable_ptr) };
    let expected = info.num_static_extra_boxes + array_lengths.iter().sum::<usize>();
    if slots.len() != expected {
        return false;
    }
    let mut values = Vec::with_capacity(expected + 1);
    let mut boxes: Vec<OpRef> = Vec::with_capacity(expected + 1);
    // `OpRef::NONE` is how the applying reader says it met a virtual kind it
    // has no allocating twin for. Under that reader the box would name an
    // object the heap does not hold, and every later vable op reads through
    // this list, so the seed fails whole rather than binding one slot to
    // nothing — the same rule `replay_pending_fields` applies. The recording
    // reader is unaffected: a direct reader allocated for it, and an
    // unresolved slot there is the pre-existing tolerated case.
    //
    // resume.py `getvirtual_ptr` allocates a virtual vable slot before
    // `write_from_resume_data_partial` stores it. The deadframe has no
    // concrete for that slot; the allocated object is the concrete.
    for slot in slots {
        if matches!(slot, RebuiltValue::Virtual(_)) {
            let op = rebuilt_value_to_opref(ctx, slot, rd_virtuals, resume_data, cache);
            if op.is_none() {
                return false;
            }
            let Some(value) = ctx.concrete_of_opref(op) else {
                return false;
            };
            values.push(value);
            boxes.push(op);
            continue;
        }
        match concrete(slot, fail_values) {
            Some(value) => values.push(value),
            None => return false,
        }
        let op = rebuilt_value_to_opref(ctx, slot, rd_virtuals, resume_data, cache);
        if op.is_none() && cache.allocator().is_some() {
            return false;
        }
        boxes.push(op);
    }
    // virtualizable.py:143-144 "the returned list is in the format expected of
    // virtualizable_boxes, so it ends in the virtualizable itself".
    let identity_op = rebuilt_value_to_opref(ctx, identity, rd_virtuals, resume_data, cache);
    if identity_op.is_none() && cache.allocator().is_some() {
        return false;
    }
    boxes.push(identity_op);
    values.push(identity_value);
    ctx.install_virtualizable_info(info.clone());
    ctx.set_virtualizable_boxes_with_info(boxes, values, info.as_ref(), &array_lengths);
    // resume.py `ResumeDataDirectReader.consume_vable_info` and
    // pyjitpl.py `MetaInterp.rebuild_state_after_failure` both
    // reset_token_gcref here: assembler (virtualizable.py case 4)
    // into tracing (case 2). CALL_MAY_FORCE stored the jitframe;
    // a later non-GUARD_NOT_FORCED exit does not force.
    // `tracing_before_residual_call` asserts TOKEN_NONE. Reset
    // without forcing: resume data already holds the boxes.
    // The other site is jitdriver.rs after setup_bridge_sym.
    if info.has_vable_token() {
        unsafe {
            if info.is_token_nonnull_gcref(vable_ptr) {
                info.reset_token_gcref(vable_ptr as *mut u8);
            }
        }
    }
    // `rebuild_state_after_failure`'s trailing `self.synchronize_virtualizable()`
    // (pyjitpl.py) — the object and the shadow have to agree before the bridge
    // replays a single vable op.
    //
    // A token-less vinfo is a `#[jit_interp]` `state` struct, and it is the
    // family with no other writer for this: the compiled loop carries its banks
    // in machine registers, so the guard leaves the struct holding whatever the
    // run was entered with, and the macro-generated mainloop that otherwise
    // keeps it current ran no opcode of that entry. A token-bearing one is a
    // host object whose slots have a boxing protocol the generic
    // `value_to_raw_bits` cannot serve, and whose own field-aware guard-failure
    // writer has already synchronized it.
    if !info.has_vable_token() {
        ctx.synchronize_virtualizable();
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{OpCode, Type};

    fn empty_resume_data(fail_arg_types: Vec<Type>) -> crate::jit_state::ResumeDataResult {
        let num_failargs = fail_arg_types.len() as i32;
        crate::jit_state::ResumeDataResult {
            frames: Vec::new(),
            virtualizable_boxes: Vec::new(),
            virtualref_values: Vec::new(),
            storage: None,
            num_failargs,
            fail_arg_types,
        }
    }

    fn tag_int(value: i32) -> i16 {
        ((value << 2) | majit_ir::resumedata::TAGINT as i32) as i16
    }

    fn tag_box(index: i32) -> i16 {
        ((index << 2) | majit_ir::resumedata::TAGBOX as i32) as i16
    }

    /// `VRawSliceInfo` is `base_buffer + offset`; the base decodes to a
    /// constant whenever the guard captured a TAGINT, which is the common case
    /// for a resume decode.
    fn materialize_raw_slice(base: i16, fail_arg_types: Vec<Type>) -> (OpRef, Vec<OpCode>) {
        let mut ctx = crate::TraceCtx::for_test_types(&fail_arg_types);
        let resume_data = empty_resume_data(fail_arg_types);
        let mut cache = BridgeVirtualCache::new(1, default_bridge_array_descr);
        let virtuals = vec![std::rc::Rc::new(majit_ir::RdVirtualInfo::VRawSliceInfo {
            offset: 7,
            fieldnums: vec![base],
        })];
        let buffer =
            materialize_bridge_virtual(&mut ctx, 0, Some(&virtuals), &resume_data, &mut cache);
        let ops = ctx
            .into_recorder()
            .ops()
            .iter()
            .map(|op| op.opcode)
            .collect();
        (buffer, ops)
    }

    #[test]
    fn a_raw_slice_off_a_constant_base_folds_its_int_add() {
        let (buffer, ops) = materialize_raw_slice(tag_int(5), Vec::new());
        assert_eq!(buffer, OpRef::const_int(12));
        assert!(!ops.contains(&OpCode::IntAdd));

        let (buffer, ops) = materialize_raw_slice(tag_box(0), vec![Type::Int]);
        assert!(!buffer.is_constant());
        assert!(ops.contains(&OpCode::IntAdd));
    }

    /// resume.py fills `virtuals_cache` before the next `allocate`, and the
    /// collector updates that cache. A recording-only cache has the same
    /// window: `materialize_concrete_virtual_ptr` allocates, then a later
    /// virtual's allocation collects.
    #[test]
    fn recording_cache_forwards_a_concrete_virtual_across_collection() {
        let mut cache = BridgeVirtualCache::new(2, default_bridge_array_descr);
        cache.set_concrete_ptr(0, majit_ir::GcRef(0x1000));
        cache.set_concrete_ptr(1, majit_ir::GcRef(0x2000));
        majit_gc::shadow_stack::walk_resume_ref_roots(|root| {
            if root.0 == 0x1000 || root.0 == 0x2000 {
                root.0 += 0x80;
            }
        });
        assert_eq!(cache.get_concrete_ptr(0), Some(majit_ir::GcRef(0x1080)));
        assert_eq!(cache.get_concrete_ptr(1), Some(majit_ir::GcRef(0x2080)));
        assert_eq!(cache.get_concrete_int(0), None);
    }

    /// `resume.py virtuals_cache` is a field on the reader. A DirectReader
    /// object copied into the recording cache is re-read from `concrete_roots`
    /// after a collection, not from the unrooted `concrete_ptr_cache` copy.
    #[test]
    fn seed_direct_virtuals_rereads_concrete_roots_after_collect() {
        let mut cache = BridgeVirtualCache::recording(
            1,
            default_bridge_array_descr,
            &[Some(majit_ir::GcRef(0x1000))],
        );
        assert_eq!(cache.get_concrete_ptr(0), Some(majit_ir::GcRef(0x1000)));
        majit_gc::shadow_stack::walk_resume_ref_roots(|root| {
            if root.0 == 0x1000 {
                root.0 = 0x2000;
            }
        });
        assert_eq!(cache.get_concrete_ptr(0), Some(majit_ir::GcRef(0x2000)));
    }

    /// The funnel folds through the executor row, so the concrete this helper
    /// supplies has to come from the same one.
    #[test]
    fn int_add_concrete_answers_only_for_two_constant_operands() {
        assert_eq!(
            int_add_concrete(OpRef::const_int(5), OpRef::const_int(7)),
            Some(majit_ir::Value::Int(12))
        );
        assert_eq!(
            int_add_concrete(OpRef::input_arg_int(0), OpRef::const_int(7)),
            None
        );
        assert_eq!(
            int_add_concrete(OpRef::const_int(5), OpRef::const_ptr(majit_ir::GcRef(8))),
            None
        );
    }

    /// resume.py decode_box asserts `box.type == kind`. kind comes from
    /// decoder.setfield (`is_pointer_field` → REF). The stored box is the
    /// SETFIELD value optimize_SETFIELD_GC kept as-is, already REF for a
    /// FLAG_POINTER field. apply_setfield therefore sees
    /// `(Type::Ref, Value::Ref)` and writes through bh_setfield_gc_r.
    #[test]
    fn materialize_pointer_field_ref_operand_matches_snapshot_kind() {
        use majit_ir::descr::SimpleSizeDescr;
        use majit_ir::{
            ArrayFlag, Const, Descr, FieldDescr, FieldDescrInfo, GcRef, SimpleFieldDescr,
        };
        use std::cell::RefCell;
        use std::sync::Arc;

        #[derive(Default)]
        struct RecAllocator {
            refs: RefCell<Vec<i64>>,
            ints: RefCell<Vec<i64>>,
        }
        impl crate::resume::BlackholeAllocator for RecAllocator {
            fn allocate_with_vtable(&self, _descr: &majit_ir::DescrRef, _vtable: usize) -> i64 {
                0xBEEF
            }
            fn bh_setfield_gc_r(&self, _struct_ptr: i64, value: i64, _descr_info: &FieldDescrInfo) {
                self.refs.borrow_mut().push(value);
            }
            fn bh_setfield_gc_i(&self, _struct_ptr: i64, value: i64, _descr_info: &FieldDescrInfo) {
                self.ints.borrow_mut().push(value);
            }
        }

        let ptr_fd = Arc::new(
            SimpleFieldDescr::new(0x1000_0abc, 16, 8, Type::Ref, false)
                .with_flag(ArrayFlag::Pointer)
                .with_index_in_parent(0),
        );
        let parent: majit_ir::DescrRef = Arc::new(
            SimpleSizeDescr::new(0x3000_0001, 32, 1)
                .with_all_fielddescrs(vec![ptr_fd.clone() as Arc<dyn FieldDescr>]),
        );
        ptr_fd.set_parent_descr(&parent);
        assert!(ptr_fd.is_pointer_field());

        let tagged = ((majit_ir::resumedata::TAG_CONST_OFFSET << 2)
            | majit_ir::resumedata::TAGCONST as i32) as i16;
        let storage = crate::resume::new_resume_storage(
            Vec::new(),
            vec![Const::Ref(GcRef(0x1111))],
            Vec::new(),
            Vec::new(),
        );
        let resume_data = crate::jit_state::ResumeDataResult {
            frames: Vec::new(),
            virtualizable_boxes: Vec::new(),
            virtualref_values: Vec::new(),
            storage: Some(storage),
            num_failargs: 0,
            fail_arg_types: Vec::new(),
        };
        let virtuals = vec![std::rc::Rc::new(majit_ir::RdVirtualInfo::VirtualInfo {
            descr: Some(parent),
            type_id: 1,
            known_class: None,
            fielddescrs: vec![FieldDescrInfo {
                index: ptr_fd.index(),
                offset: ptr_fd.offset(),
                field_type: Type::Ref,
                field_size: ptr_fd.field_size(),
            }],
            fieldnums: vec![tagged],
            descr_size: 32,
        })];
        let alloc = RecAllocator::default();
        let mut ctx = crate::TraceCtx::for_test_types(&[]);
        let mut cache =
            BridgeVirtualCache::executing(1, default_bridge_array_descr, &alloc, &[], &[]);
        let result =
            materialize_bridge_virtual(&mut ctx, 0, Some(&virtuals), &resume_data, &mut cache);
        assert!(!result.is_none(), "pointer-field virtual must materialize");
        assert_eq!(*alloc.refs.borrow(), vec![0x1111]);
        assert!(alloc.ints.borrow().is_empty());

        // setfields passes `self.fielddescrs[i]`. An Int row overlaying the
        // pointer's word sits earlier in the list; the row is the one at the
        // snapshot's position, not the first at that offset.
        let word_fd = Arc::new(
            SimpleFieldDescr::new(0x1000_0abd, 16, 8, Type::Int, false).with_index_in_parent(0),
        );
        let overlay_ptr_fd = Arc::new(
            SimpleFieldDescr::new(0x1000_0abc, 16, 8, Type::Ref, false)
                .with_flag(ArrayFlag::Pointer)
                .with_index_in_parent(1),
        );
        let overlay_parent: majit_ir::DescrRef = Arc::new(
            SimpleSizeDescr::new(0x3000_0002, 32, 1).with_all_fielddescrs(vec![
                word_fd.clone() as Arc<dyn FieldDescr>,
                overlay_ptr_fd.clone() as Arc<dyn FieldDescr>,
            ]),
        );
        let live = crate::resume::live_field_descr_for_resume(
            Some(&overlay_parent),
            1,
            &FieldDescrInfo {
                index: overlay_ptr_fd.index(),
                offset: 16,
                field_type: Type::Ref,
                field_size: 8,
            },
        )
        .expect("the snapshot row is listed");
        assert!(live.as_field_descr().unwrap().is_pointer_field());
        assert_eq!(live.index(), overlay_ptr_fd.index());
    }

    /// pyjitpl.py `MetaInterp.rebuild_state_after_failure` /
    /// resume.py `ResumeDataDirectReader.consume_vable_info`: leaving
    /// compiled code with `vable_token` still the jitframe (CALL_MAY_FORCE
    /// stored it; a later non-GUARD_NOT_FORCED exit does not force) must
    /// reset it to TOKEN_NONE before tracing. `tracing_before_residual_call`
    /// asserts that.
    #[test]
    fn seed_bridge_virtualizable_boxes_resets_assembler_token() {
        let info = std::sync::Arc::new(crate::virtualizable::VirtualizableInfo::new(0));
        let mut obj = Box::new([0u64; 2]);
        let obj_ptr = obj.as_mut_ptr().cast::<u8>();
        unsafe {
            *(obj_ptr as *mut u64) = 0xDEAD_BEEF;
            assert!(info.is_token_nonnull_gcref(obj_ptr as *const u8));
        }

        let mut ctx = crate::TraceCtx::for_test_types(&[Type::Ref]);
        ctx.set_virtualizable_heap_ptr(obj_ptr as *const u8);

        let fail_values = [obj_ptr as usize as i64];
        let resume_data = crate::jit_state::ResumeDataResult {
            frames: Vec::new(),
            virtualizable_boxes: vec![majit_ir::resumedata::RebuiltValue::Box(0, Type::Ref)],
            virtualref_values: Vec::new(),
            storage: None,
            num_failargs: 1,
            fail_arg_types: vec![Type::Ref],
        };
        let mut cache = BridgeVirtualCache::new(1, default_bridge_array_descr);
        let seeded = seed_bridge_virtualizable_boxes(
            &mut ctx,
            &info,
            None,
            &resume_data,
            &mut cache,
            &fail_values,
        );
        assert!(seeded, "identity matches the live heap pointer");
        unsafe {
            assert!(
                !info.is_token_nonnull_gcref(obj_ptr as *const u8),
                "reset_token_gcref must leave TOKEN_NONE"
            );
        }
    }
}
