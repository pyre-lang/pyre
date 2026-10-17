#![cfg_attr(test, allow(unused_variables))]

use indexmap::{IndexMap, IndexSet};
/// GC rewriter — transforms high-level allocation and store operations
/// into GC-aware lower-level IR before code generation.
///
/// Converts:
/// - NEW / NEW_WITH_VTABLE -> CALL_MALLOC_NURSERY + tid initialization
///   (consecutive fixed-size allocations are batched into a single
///   CALL_MALLOC_NURSERY with NURSERY_PTR_INCREMENT for subsequent objects)
/// - NEW_ARRAY / NEW_ARRAY_CLEAR -> CALL_MALLOC_NURSERY_VARSIZE
/// - SETFIELD_GC with a Ref-typed value -> COND_CALL_GC_WB + SETFIELD_GC
///
/// Reference: rpython/jit/backend/llsupport/rewrite.py GcRewriterAssembler.
use majit_ir::Type;
use majit_ir::descr::{DescrRef, FieldDescr, LoopTokenDescr, SizeDescr};
use majit_ir::forwarding::{Forwarded, ForwardingHost};
use majit_ir::operand::Operand;
use majit_ir::resoperation::{Op, OpCode, OpRc, OpRef};
use majit_ir::{Const, ConstMap, GcRef, Value};
use std::borrow::Cow;

use crate::{GcRewriter, WriteBarrierDescr};
use rustc_hash::FxBuildHasher;

type FxIndexMap<K, V> = IndexMap<K, V, FxBuildHasher>;
type FxIndexSet<K> = IndexSet<K, FxBuildHasher>;
type FxHashSet<T> = std::collections::HashSet<T, FxBuildHasher>;

fn boehm_malloc_fn() -> Option<usize> {
    let addr = crate::malloc_fixedsize_addr();
    if addr == 0 { None } else { Some(addr) }
}

fn boehm_malloc_fixedsize_descr() -> DescrRef {
    use std::sync::{Arc, OnceLock};
    static DESCR: OnceLock<DescrRef> = OnceLock::new();
    DESCR
        .get_or_init(|| {
            Arc::new(majit_ir::descr::SimpleCallDescr::new(
                0x5000_0011,
                vec![Type::Int],
                Type::Ref,
                false,
                std::mem::size_of::<usize>(),
                majit_ir::EffectInfo::MOST_GENERAL,
            ))
        })
        .clone()
}

fn boehm_malloc_array_descr() -> DescrRef {
    use std::sync::{Arc, OnceLock};
    static DESCR: OnceLock<DescrRef> = OnceLock::new();
    DESCR
        .get_or_init(|| {
            Arc::new(majit_ir::descr::SimpleCallDescr::new(
                0x5000_0012,
                vec![Type::Int, Type::Int, Type::Int, Type::Int],
                Type::Ref,
                false,
                std::mem::size_of::<usize>(),
                majit_ir::EffectInfo::MOST_GENERAL,
            ))
        })
        .clone()
}

fn mk_op(opcode: OpCode, args: &[Operand]) -> Op {
    Op::new(opcode, args)
}
#[allow(dead_code)]
fn mk_op_descr(opcode: OpCode, args: &[Operand], descr: DescrRef) -> Op {
    Op::with_descr(opcode, args, descr)
}

/// rewrite.py `emit_op`'s reference-constant loop, run on its own.
///
/// The full rewrite ([`GcRewriterImpl`]) also lowers mallocs, GC
/// loads/stores and write barriers. A backend that runs only this half
/// still gets the constant-pointer contract: a raw `GcRef`
/// baked as a code immediate is a pointer the moving collector can
/// neither find nor update, so the first minor collection that promotes
/// the referenced object out of the nursery leaves the immediate
/// dangling at an address the nursery later reuses or zeroes. Running
/// just this pass gives such a backend the same `LoadFromGcTable` +
/// [`GcTable`](crate::GcTable) contract the native backends get from the
/// full rewrite.
///
/// `next_pos` is the first free value position; the emitted loads take
/// positions from there upward, so the caller's existing operand
/// numbering is untouched. Returns the rewritten ops and the
/// `gcrefs_output_list` (rewrite.py:352) the caller turns into the
/// per-loop table.
/// rewrite.py `emit_op` operand rewrite: follow `_forwarded`, then
/// `remove_constptr` on a non-null `ConstPtr`. A folded InputArg/Op
/// carries its constant on `_forwarded`; `const_value()` on the operand
/// itself is then `None`, so follow the replacement first.
fn rewrite_operand(
    operand: Operand,
    defined: &FxHashSet<u32>,
    gcrefs: &mut Vec<GcRef>,
    gcrefs_map: &mut FxIndexMap<usize, u32>,
    recently_loaded: &mut FxIndexMap<u32, Operand>,
    next_pos: &mut u32,
    out: &mut Vec<Op>,
) -> Operand {
    // rewrite.py `get_box_replacement(arg)` — `not_const=False`.
    let replaced = operand.get_box_replacement(false);
    if let Some(load) = intern_constptr_operand(
        replaced.clone(),
        gcrefs,
        gcrefs_map,
        recently_loaded,
        next_pos,
        out,
    ) {
        return load;
    }
    // rewrite.py keeps a null ConstPtr inline. A live InputArg/Op forwarded
    // to null still has a runtime pointer (the virtualizable identity);
    // baking ConstPtr(0) over it publishes null at deopt.
    if let Some(Value::Ref(gcref)) = replaced.const_value()
        && gcref.is_null()
    {
        let orig = operand.to_opref();
        if !orig.is_none() && !orig.is_constant() && defined.contains(&orig.raw()) {
            return operand;
        }
    }
    replaced
}

/// rewrite.py `_gcref_index` — put a non-null ConstPtr in the output
/// table without emitting a load. Failargs stay constants
/// (`emit_op` `get_box_replacement` only); the backend rematerializes
/// them from this table on the deopt path.
fn register_constptr(
    operand: &Operand,
    gcrefs: &mut Vec<GcRef>,
    gcrefs_map: &mut FxIndexMap<usize, u32>,
) {
    let Some(Value::Ref(gcref)) = operand.const_value() else {
        return;
    };
    if gcref.is_null() {
        return;
    }
    gcrefs_map.entry(gcref.0).or_insert_with(|| {
        let index = gcrefs.len() as u32;
        gcrefs.push(gcref);
        index
    });
}

fn intern_constptr_operand(
    operand: Operand,
    gcrefs: &mut Vec<GcRef>,
    gcrefs_map: &mut FxIndexMap<usize, u32>,
    recently_loaded: &mut FxIndexMap<u32, Operand>,
    next_pos: &mut u32,
    out: &mut Vec<Op>,
) -> Option<Operand> {
    // rewrite.py `bool(arg.value)` — null stays inline.
    let Value::Ref(gcref) = operand.const_value()? else {
        return None;
    };
    if gcref.is_null() {
        return None;
    }
    // rewrite.py `_gcref_index`.
    let index = *gcrefs_map.entry(gcref.0).or_insert_with(|| {
        let index = gcrefs.len() as u32;
        gcrefs.push(gcref);
        index
    });
    // rewrite.py `remove_constptr`.
    Some(match recently_loaded.get(&index) {
        Some(load) => load.clone(),
        None => {
            let load_op = OpRc::new(mk_op(
                OpCode::LoadFromGcTable,
                &[Operand::const_from_value(Value::Int(index as i64))],
            ));
            load_op.pos().set(OpRef::ref_op(*next_pos));
            *next_pos += 1;
            out.push((*load_op).clone());
            let load = Operand::from_bound_op(&load_op);
            recently_loaded.insert(index, load.clone());
            load
        }
    })
}

pub fn remove_ref_constants(ops: &[Op], next_pos: u32) -> (Vec<Op>, Vec<GcRef>) {
    remove_ref_constants_for_inputs(ops, next_pos, &[])
}

/// Same as [`remove_ref_constants`], but token input indices stay live-ins.
/// A live InputArg forwarded to null must not be rewritten to `ConstPtr(0)`:
/// that publishes a null virtualizable identity and panics in
/// `consume_vable_info`. A dead leftover InputArg (not in this set) is still
/// interned or inlined as rewrite.py does.
pub fn remove_ref_constants_for_inputs(
    ops: &[Op],
    mut next_pos: u32,
    input_indices: &[u32],
) -> (Vec<Op>, Vec<GcRef>) {
    // rewrite.py:352-354 `gcrefs_output_list` / `gcrefs_map` /
    // `gcrefs_recently_loaded`.
    let mut gcrefs: Vec<GcRef> = Vec::new();
    let mut gcrefs_map: FxIndexMap<usize, u32> = IndexMap::default();
    let mut recently_loaded: FxIndexMap<u32, Operand> = IndexMap::default();
    let mut out: Vec<Op> = Vec::with_capacity(ops.len());
    let mut defined: FxHashSet<u32> = ops
        .iter()
        .filter_map(|op| {
            let pos = op.pos().get();
            (pos != OpRef::NONE && !pos.is_constant()).then_some(pos.raw())
        })
        .collect();
    defined.extend(input_indices.iter().copied());

    for op in ops {
        // rewrite.py:1005 — the per-basic-block CSE cache is dropped at
        // every Label. A load emitted in the preamble must not be reused
        // after the header: a resume entry jumps straight to the label
        // and would read an unwritten slot.
        if op.opcode == OpCode::Label {
            recently_loaded.clear();
        }
        let op = op.clone();
        // rewrite.py:105 `keep` — JIT_DEBUG keeps its constants inline.
        if op.opcode != OpCode::JitDebug {
            for i in 0..op.num_args() {
                op.setarg(
                    i,
                    rewrite_operand(
                        op.arg(i),
                        &defined,
                        &mut gcrefs,
                        &mut gcrefs_map,
                        &mut recently_loaded,
                        &mut next_pos,
                        &mut out,
                    ),
                );
            }
            // rewrite.py `emit_op`:
            //   op.setfailargs([self.get_box_replacement(a, True) ...])
            // Follow `_forwarded` only. `remove_constptr` is the arg
            // loop, not failargs: a ConstPtr failarg is materialized at
            // deopt from the table, not loaded on every succeeding
            // iteration. Still `_gcref_index` it so the moving collector
            // can forward the slot the backend rematerializes.
            if let Some(fail_args) = op.getfailargs() {
                let rewritten: Vec<Operand> = fail_args
                    .into_iter()
                    .map(|arg| {
                        let replaced = arg.get_box_replacement(false);
                        // Same keep as `rewrite_operand`: a live InputArg
                        // forwarded to null is still the virtualizable
                        // identity. Baking ConstPtr(0) here is what
                        // `consume_vable_info` then reads as identity 0.
                        if let Some(Value::Ref(gcref)) = replaced.const_value()
                            && gcref.is_null()
                        {
                            let orig = arg.to_opref();
                            if !orig.is_none()
                                && !orig.is_constant()
                                && defined.contains(&orig.raw())
                            {
                                return arg;
                            }
                        }
                        register_constptr(&replaced, &mut gcrefs, &mut gcrefs_map);
                        replaced
                    })
                    .collect();
                // `setfailargs`, not `store_final_boxes`: two failargs can
                // forward onto one live producer, and the optimizer-stage
                // uniqueness check would fire on the rewritten vector.
                op.setfailargs(rewritten.into());
            }
        }
        out.push(op);
    }
    (out, gcrefs)
}

/// Alignment for nursery allocations (8 bytes).
const NURSERY_ALIGN: usize = 8;

/// `GcRewriterAssembler.round_up_for_allocation` — raise to the nursery's
/// minimum object size, then align.
///
/// A nursery object that moves has its header replaced by the forwarded marker
/// and the following word by the new address, so a block shorter than
/// [`GcHeader::MIN_NURSERY_OBJ_SIZE`] would let that second word land in the
/// next object.  `Nursery::alloc` already raises the runtime bump; the JIT
/// bumps by whatever this hands `CALL_MALLOC_NURSERY` /
/// `NURSERY_PTR_INCREMENT`, so it has to raise it too.  Upstream spells the
/// same floor as `max(size, 2 * WORD)`.
fn round_up(size: usize) -> usize {
    let size = size.max(crate::header::GcHeader::MIN_NURSERY_OBJ_SIZE);
    (size + NURSERY_ALIGN - 1) & !(NURSERY_ALIGN - 1)
}

/// `get_array_token(rstr.STR, ...)` / `get_array_token(rstr.UNICODE, ...)`
/// as consumed by rewrite.py:295-318.  Reads `(itemsize, basesize)` from
/// the injected ArrayDescr and applies the `basesize -= 1` correction
/// for STR (the extra_item_after_alloc null).
fn strgetsetitem_token(op: &Op, is_str: bool) -> (i64, i64) {
    let descr = op
        .getdescr()
        .expect("STR/UNICODE getitem/setitem op must carry an ArrayDescr");
    let ad = descr
        .as_array_descr()
        .expect("STR/UNICODE getitem/setitem descr must be an ArrayDescr");
    let itemsize = ad.item_size() as i64;
    let mut basesize = ad.base_size() as i64;
    if is_str {
        // rewrite.py:298 assert itemsize == 1 for STR.
        assert_eq!(
            itemsize, 1,
            "rewrite.py:298 STR getitem/setitem itemsize must be 1"
        );
        basesize -= 1; // rewrite.py:299 — skip extra null character
    }
    (itemsize, basesize)
}

/// resoperation.py `OpHelpers.get_gc_load`.
/// Select GC_LOAD_I / GC_LOAD_R / GC_LOAD_F by result type.
fn get_gc_load(tp: Type) -> OpCode {
    match tp {
        Type::Int => OpCode::GcLoadI,
        Type::Ref => OpCode::GcLoadR,
        Type::Float => OpCode::GcLoadF,
        Type::Void => panic!("get_gc_load: cannot lower a Void-typed load"),
    }
}

/// resoperation.py `OpHelpers.get_gc_load_indexed`.
/// Select GC_LOAD_INDEXED_I / GC_LOAD_INDEXED_R / GC_LOAD_INDEXED_F by
/// result type.
fn get_gc_load_indexed(tp: Type) -> OpCode {
    match tp {
        Type::Int => OpCode::GcLoadIndexedI,
        Type::Ref => OpCode::GcLoadIndexedR,
        Type::Float => OpCode::GcLoadIndexedF,
        Type::Void => panic!("get_gc_load_indexed: cannot lower a Void-typed load"),
    }
}

/// regalloc.py:871 — compiled_loop_token._ll_initial_locs + frame_info.
///
/// Per-callee metadata needed by handle_call_assembler to allocate and
/// fill the callee jitframe.
#[derive(Clone, Debug)]
pub struct CallAssemblerCalleeLocs {
    /// regalloc.py:869 — byte offsets of each inputarg within jf_frame,
    /// relative to BASEITEMOFS (i.e. index_list[i] = loc.value - base_ofs).
    pub _ll_initial_locs: Vec<i32>,
    /// Total jitframe depth (in slots).
    /// regalloc.py:861 — needed for gen_malloc_frame allocation size.
    pub frame_depth: usize,
    /// rewrite.py:669 — ptr2int(loop_token.compiled_loop_token.frame_info).
    /// Raw address of the callee's JitFrameInfo struct.
    pub frame_info_ptr: usize,
    /// pyjitpl.py direct_call_may_force — jd.index_of_virtualizable.
    /// Index into the original arglist of the virtualizable box.
    /// -1 if no virtualizable.
    pub index_of_virtualizable: i32,
}

/// GC rewriter implementation.
///
/// Walks a list of IR operations and rewrites allocation / store ops
/// into backend-friendly forms (CALL_MALLOC_NURSERY, COND_CALL_GC_WB, etc.).
pub struct GcRewriterImpl {
    /// Nursery free pointer address (for inline allocation).
    pub nursery_free_addr: usize,
    /// Nursery top pointer address.
    pub nursery_top_addr: usize,
    /// Maximum object size for nursery allocation.
    pub max_nursery_size: usize,
    /// `gc.py self.write_barrier_descr = WriteBarrierDescr(self)` — `None`
    /// for a collector that needs no write barrier, exactly as
    /// `gc.py GcLLDescr_boehm.write_barrier_descr` is.
    /// `rewrite.py:393` gates the whole barrier section on this being set,
    /// so a `None` here means no `COND_CALL_GC_WB*` is ever emitted; the
    /// backend asserts the same agreement when it assembles one.
    pub wb_descr: Option<WriteBarrierDescr>,
    /// JitFrame info for call_assembler rewriting.
    /// rewrite.py — handle_call_assembler needs frame layout info.
    pub jitframe_info: Option<JitFrameDescrs>,
    /// rewrite.py `handle_call_assembler` — `loop_token.compiled_loop_token._ll_initial_locs`.
    /// Upstream `op.getdescr()` IS the `JitCellToken`; the backend recovers
    /// that object from `LoopTokenDescr::token_handle_any` and reads the CLT.
    pub call_assembler_callee_locs:
        Option<Box<dyn Fn(&dyn LoopTokenDescr) -> Option<CallAssemblerCalleeLocs> + Send>>,
    /// llmodel.py `load_supported_factors = (1,)` — the default for
    /// CPUs whose addressing mode only scales by one. x86 overrides this
    /// at `rpython/jit/backend/x86/runner.py:31` with `(1, 2, 4, 8)`.
    /// Consumed by `cpu_simplify_scale` (rewrite.py) to decide
    /// whether a non-constant index's factor can be folded into the
    /// backend's addressing mode or must be pre-scaled in IR.
    pub load_supported_factors: &'static [i64],
    /// model.py `supports_load_effective_address = False` (base) /
    /// x86/runner.py:22 + arm/runner.py:26 override to `True`.
    /// Consumed by `emit_load_effective_address` (rewrite.py): when
    /// `True` the helper emits a single `LOAD_EFFECTIVE_ADDRESS`; when
    /// `False` it expands to `(INT_LSHIFT?) + INT_ADD + INT_ADD`.
    pub supports_load_effective_address: bool,
    /// llsupport/gc.py __init__ `malloc_zero_filled` parity.
    ///
    /// `true` when the allocation path itself guarantees zero-filled
    /// payload bytes. Production backends set this to `false` whenever a
    /// real collector is installed and to `true` for the Boehm/raw-calloc
    /// fallback (both backends' `gc_rewriter`). Gates
    /// `clear_gc_fields` per rewrite.py; the non-zero-fill path
    /// emits explicit NULL-pointer stores at flush time
    /// (rewrite.py:761-766).
    pub malloc_zero_filled: bool,
    /// [`crate::GcAllocator::headerless_fixedsize`]: bump the payload
    /// itself. `gen_malloc_nursery` emits `CallMallocNurseryHeaderless`
    /// and does not stamp a tid.
    pub headerless_fixedsize: bool,
    /// llsupport/gc.py `self.memcpy_fn = memcpy_fn` cast to a Signed
    /// integer via `cast_ptr_to_adr` + `cast_adr_to_int`
    /// (rewrite.py:1046-1047). Embedded as a ConstInt into the lowered
    /// `CALL_N(memcpy_fn, dst, src, n)` emitted by
    /// `rewrite_copy_str_content` for COPYSTRCONTENT / COPYUNICODECONTENT.
    pub memcpy_fn: i64,
    /// llsupport/gc.py:40-43 `self.memcpy_descr = get_call_descr(...)`.
    /// CallDescr stamped onto the lowered `CALL_N(memcpy_fn, ...)` —
    /// `[Signed, Signed, Signed] -> Void`, `EF_CANNOT_RAISE`,
    /// `can_collect=False`. Single instance shared across rewrites
    /// (`make_memcpy_calldescr` singleton, descr.rs).
    pub memcpy_descr: DescrRef,
    /// llsupport/gc.py:46 `self.str_descr = get_array_descr(self, rstr.STR)`.
    /// Provides `basesize` / `itemsize=1` for `rewrite_copy_str_content`
    /// COPYSTRCONTENT lowering — basesize is offset by `-1` at use to
    /// skip the `extra_item_after_alloc=True` null terminator
    /// (rstr.py:1226; rewrite.py:1051-1053).
    pub str_descr: DescrRef,
    /// llsupport/gc.py `self.unicode_descr = get_array_descr(self, rstr.UNICODE)`.
    /// Provides `basesize` / `itemsize` (2 or 4 bytes per UCS) for
    /// `rewrite_copy_str_content` COPYUNICODECONTENT lowering;
    /// `itemscale = log2(itemsize)` (rewrite.py:1057-1063).
    pub unicode_descr: DescrRef,
    /// llsupport/gc.py:48 `self.str_hash_descr = get_field_descr(self, rstr.STR, 'hash')`.
    /// FieldDescr for the `hash` field of `rstr.STR`. Consumed by
    /// `clear_varsize_gc_fields` at NEWSTR allocation when
    /// `malloc_zero_filled=false` (rewrite.py:529-530), where upstream
    /// emits `emit_setfield(result, ConstInt(0), descr=hash_descr)`.
    pub str_hash_descr: DescrRef,
    /// llsupport/gc.py `self.unicode_hash_descr = get_field_descr(self, rstr.UNICODE, 'hash')`.
    /// Same role as `str_hash_descr` for NEWUNICODE
    /// (rewrite.py:531-535).
    pub unicode_hash_descr: DescrRef,
    /// llsupport/gc.py:33-37 `self.fielddescr_vtable = get_field_descr(
    /// self, rclass.OBJECT, 'typeptr')` or `None` under
    /// `gcremovetypeptr=True`.  Consumed by `handle_new`'s
    /// NEW_WITH_VTABLE branch (rewrite.py:482-484) to stamp the vtable
    /// onto the freshly-allocated object's typeptr slot.  `None`
    /// disables the vtable store entirely (matching upstream's
    /// `gcremovetypeptr` build configuration).
    pub fielddescr_vtable: Option<DescrRef>,
    /// llsupport/gc.py:157 `fielddescr_tid = None` (Boehm) /
    /// llsupport/gc.py:394 `self.fielddescr_tid = get_field_descr(self,
    /// self.GCClass.HDR, 'tid')` (framework GC).  Consumed by
    /// `rewrite.py` `gen_initialize_tid` to stamp the type id
    /// onto the freshly-allocated object's header word.  `None` makes
    /// `gen_initialize_tid` a no-op (Boehm path).
    ///
    /// pyre's HDR sits *before* the object pointer (vs RPython's HDR at
    /// the object pointer); `gen_initialize_tid` translates the descr's
    /// offset by `-HDR_SIZE` so the GC_STORE addresses the header word.
    pub fielddescr_tid: Option<DescrRef>,
    /// gc.py `generate_function('malloc_array', ...)` function addr.
    pub malloc_array_fn: i64,
    /// gc.py `generate_function('malloc_array_nonstandard', ...)`
    /// function addr.
    pub malloc_array_nonstandard_fn: i64,
    /// Old-generation twin of [`Self::malloc_array_fn`], selected by
    /// `gen_malloc_array` when the array descr is
    /// [`majit_ir::descr::ArrayDescr::non_moving`].  Same call signature,
    /// so it shares `malloc_array_descr`; only the allocator differs.
    pub malloc_array_oldgen_fn: i64,
    /// Old-generation twin of [`Self::malloc_array_nonstandard_fn`], and
    /// shares `malloc_array_nonstandard_descr` for the same reason.
    pub malloc_array_nonstandard_oldgen_fn: i64,
    /// gc.py `generate_function('malloc_str', ...)` function addr.
    pub malloc_str_fn: i64,
    /// gc.py `generate_function('malloc_unicode', ...)` function addr.
    pub malloc_unicode_fn: i64,
    /// gc.py `generate_function('malloc_big_fixedsize', ...)` function addr.
    /// Consumed by `rewrite.py gen_malloc_fixedsize` framework-GC arm
    /// when a fixed-size NEW exceeds the nursery threshold.
    pub malloc_big_fixedsize_fn: i64,
    /// Old-generation twin of [`Self::malloc_big_fixedsize_fn`], selected by
    /// `gen_malloc_fixedsize` when the size descr is
    /// [`majit_ir::descr::SizeDescr::non_moving`].  Same call signature, so it
    /// shares `malloc_big_fixedsize_descr`; only the allocator differs.
    ///
    /// Upstream has no such twin: `gen_malloc_fixedsize` is reached only by an
    /// oversized NEW there, and an oversized object is non-moving whichever arm
    /// it takes.  pyre reuses the same generator for a descr that demands a
    /// non-moving address at *any* size — `PYFRAME_DESCR_GROUP.size_descr` is
    /// the live one — and that demand is not something
    /// `malloc_big_fixedsize`'s own allocator can answer any more, now that it
    /// dispatches on size and its small arm is the nursery.
    pub malloc_big_fixedsize_oldgen_fn: i64,
    /// gc.py:45 `self.malloc_array_descr = get_call_descr(...)`.
    pub malloc_array_descr: DescrRef,
    /// gc.py:45 `self.malloc_array_nonstandard_descr = get_call_descr(...)`.
    pub malloc_array_nonstandard_descr: DescrRef,
    /// gc.py:45 `self.malloc_str_descr = get_call_descr(...)`.
    pub malloc_str_descr: DescrRef,
    /// gc.py:45 `self.malloc_unicode_descr = get_call_descr(...)`.
    pub malloc_unicode_descr: DescrRef,
    /// gc.py:45 `self.malloc_big_fixedsize_descr = get_call_descr(...)`.
    pub malloc_big_fixedsize_descr: DescrRef,
    /// gc.py:396 `self.standard_array_basesize`.
    pub standard_array_basesize: usize,
    /// gc.py:397 `self.standard_array_length_ofs`.
    pub standard_array_length_ofs: usize,
}

/// JitFrame field descriptors for handle_call_assembler.
///
/// rewrite.py:666 — `descrs = self.gc_ll_descr.getframedescrs(self.cpu)`
#[derive(Clone)]
pub struct JitFrameDescrs {
    /// GC type id for JitFrame (from gc.register_type).
    pub jitframe_tid: u32,
    /// JitFrame fixed header size (bytes).
    pub jitframe_fixed_size: usize,
    /// Byte offsets of JitFrame fields (from majit_backend::jitframe).
    pub jf_frame_info_ofs: i32,
    pub jf_descr_ofs: i32,
    pub jf_force_descr_ofs: i32,
    pub jf_savedata_ofs: i32,
    pub jf_guard_exc_ofs: i32,
    pub jf_forward_ofs: i32,
    /// Offset from JitFrame start to jf_frame length field.
    pub jf_frame_ofs: usize,
    /// unpack_arraydescr(arraydescr): basesize, measured from JitFrame start.
    pub jf_frame_baseitemofs: usize,
    /// descrs.arraydescr.lendescr.offset, measured from JitFrame start.
    pub jf_frame_lengthofs: usize,
    /// SIGN_SIZE: width of a Signed header field (`jf_frame_info`,
    /// `jfi_frame_size`, the `jf_frame` length).
    pub sign_size: usize,
    /// unpack_arraydescr_size(descrs.arraydescr): width of one `jf_frame`
    /// item.  Upstream's `JITFRAME.jf_frame` is `Array(Signed)`, so this is
    /// `sign_size` on the native backends.  The wasm backend keeps every
    /// value in an i64 slot, so its items are 8 bytes wide on wasm32 too.
    pub jf_frame_itemsize: usize,
}

impl JitFrameDescrs {
    /// llmodel.py + llmodel.py getarraydescr_for_frame — itemsize of the per-arg-type
    /// frame arraydescr (signedarraydescr / refarraydescr / floatarraydescr),
    /// read via getarraydescr_for_frame + unpack_arraydescr_size.
    ///
    /// Upstream builds:
    ///   signedarraydescr = ad                        (itemsize = WORD)
    ///   refarraydescr    = ArrayDescr(.., ad.itemsize, ..)
    ///   floatarraydescr  = ArrayDescr(..,
    ///         ad.itemsize * 2 if WORD == 4 else ad.itemsize, ..)
    /// so a FLOAT spans two items when an item is 4 bytes wide.  `ad` is the
    /// `jf_frame` arraydescr, so its itemsize is `jf_frame_itemsize`.
    fn frame_itemsize(&self, ty: Type) -> i64 {
        let itemsize = self.jf_frame_itemsize as i64;
        match ty {
            // Value slots are i64 on every backend. wasm32's Signed is 4,
            // but `publish_ca_initial_locs` and the entry loader address
            // `FRAME_SLOT_BASE + k*8` and the loader does an 8-byte load.
            // A 4-byte store leaves the high half stale, so a pointer
            // compare against a zero-extended field load fails.
            Type::Int | Type::Ref => 8,
            Type::Float => {
                if itemsize == 4 {
                    itemsize * 2
                } else {
                    itemsize
                }
            }
            Type::Void => panic!("CALL_ASSEMBLER arg must have a concrete type"),
        }
    }
}

/// rewrite.py `cpu_simplify_scale` return value.
///
/// Encodes the three semantic outcomes that upstream signals via the
/// `(index_box, emit)` tuple:
/// - `Const`        — index folded into offset; caller must emit the
///   non-indexed GC_LOAD / GC_STORE form (upstream's
///   `index_box is None` check at rewrite.py:148, :199).
/// - `Passthrough`  — original index re-used unchanged; the CPU's
///   addressing mode supports the requested factor.
/// - `PreScale(op)` — the helper constructed an unemitted INT_LSHIFT /
///   INT_MUL; `_emit_mul_if_factor_offset_not_supported`
///   (or vector_ext callers) emits it before reading the
///   OpRef back as the final indexed-op `index` arg.
#[allow(clippy::large_enum_variant)]
enum ScaledIndex {
    Const,
    Passthrough(Operand),
    PreScale(Op),
}

/// rewrite.py:719-758 last_zero_arrays parity.
///
/// A ZERO_ARRAY op already emitted into `out` whose start/length will
/// be tightened (or zeroed out) at flush time based on which indices
/// were covered by subsequent SETARRAYITEM ops.  Mirrors RPython's
/// in-place setarg pattern — one ZERO_ARRAY per array, trimmed from
/// both ends; never split into multiple runs.
struct PendingZero {
    /// Index in `out` of the already-emitted ZERO_ARRAY op.
    out_index: usize,
    /// The OpRef of the array being zeroed.
    array_ref: OpRef,
    /// Initial length (number of items) before any trimming.
    length: usize,
    /// Per-item byte size from the array descriptor (`scale` in
    /// rewrite.py:727).
    scale: i64,
}

/// Per-rewrite mutable state (not stored on the struct so that
/// `rewrite_for_gc` can take `&self`).
struct RewriteState<'a> {
    /// Output operation list. `Rc`-carried so `emit` can hand back a
    /// BOUND result box — downstream references to a newly emitted op
    /// live-track its producer instead of fabricating position-only
    /// boxes (#9 operand-union grind).
    out: Vec<majit_ir::OpRc>,
    /// Next position index for emitted result ops that do not have an
    /// explicit source position to preserve.
    next_pos: u32,
    /// Constant pool (from optimizer) — maps OpRef key → typed `Const`
    /// value. Each box variant carries its own type (`Const::get_type`).
    /// Borrowed: `resolve_constant` only reads it.
    constants: &'a ConstMap<Const>,

    // ── Nursery batching ──
    /// The index in `out` of the current CALL_MALLOC_NURSERY op, if any.
    /// We try to merge consecutive small allocations into one bump.
    pending_malloc_idx: Option<usize>,
    /// Total size accumulated in the current CALL_MALLOC_NURSERY.
    pending_malloc_total: usize,
    /// Size of the *previous* individual allocation (used for
    /// NURSERY_PTR_INCREMENT offset).
    previous_size: usize,
    /// The last object produced by the current nursery batch.
    last_malloced_ref: Operand,

    // ── Write barrier tracking ──
    /// rewrite.py:41-45: _write_barrier_applied — set of OpRef indices
    /// whose write barrier has already been emitted (freshly allocated
    /// objects, or objects we already issued a WB for). Cleared whenever
    /// we emit an operation that can trigger a collection or on LABEL.
    wb_applied: FxIndexSet<OpRef>,

    // ── Array length tracking (rewrite.py:59 _known_lengths) ──
    /// Maps array OpRef → known length. Populated when NEW_ARRAY has a
    /// constant length operand (rewrite.py:551). Cleared on LABEL
    /// (rewrite.py) and emitting_an_operation_that_can_collect.
    known_lengths: FxIndexMap<OpRef, usize>,

    // ── Pending zero tracking ──
    /// Deferred ZERO_ARRAY ops that may be optimized away if subsequent
    /// SETARRAYITEM writes cover the entire range.
    pending_zeros: Vec<PendingZero>,
    /// Tracks which array indices have been explicitly SET since the
    /// pending zero was recorded. Keyed by array OpRef index.
    initialized_indices: FxIndexMap<OpRef, FxIndexSet<usize>>,
    /// rewrite.py `_delayed_zero_setfields = {}`.
    ///
    /// Map from base OpRef → set of byte-offsets of zero-init SETFIELD_GC
    /// stores deferred by `clear_gc_fields`.  An explicit SETFIELD_GC
    /// that overwrites the same offset removes the entry via
    /// `consider_setfield_gc` (rewrite.py); anything still
    /// pending at the next can-collect / flush point is emitted as
    /// `GC_STORE(ptr, ofs, 0, WORD)` by `emit_pending_zeros`
    /// (rewrite.py:761-766).
    /// The dict key is the base box; the map keys it by position and keeps
    /// the box beside its offsets so the flush stores through it.
    _delayed_zero_setfields: FxIndexMap<OpRef, (Operand, FxIndexSet<i64>)>,

    // ── INT_ADD/INT_SUB constant-fold tracking (rewrite.py:64) ──
    /// `_constant_additions[box]` = `(older_box, constant_add)` for an
    /// int_add/int_sub whose constant operand can be folded into a
    /// downstream GC_STORE_INDEXED / GC_LOAD_INDEXED offset.  See
    /// rewrite.py record_int_add_or_sub and rewrite.py
    /// _try_use_older_box.
    ///
    /// pyre's emit_setarrayitem path does not currently lower to
    /// GC_STORE_INDEXED, so this map is populated but its
    /// _try_use_older_box consumer is parked until that lowering is
    /// ported.  The parity skeleton is kept here so the structural
    /// presence matches upstream and the consumer can be wired without
    /// re-introducing the field.
    _constant_additions: FxIndexMap<OpRef, (Operand, i64)>,
    /// Reserved next constant index in the passed-in constant namespace.
    /// Current GC-rewrite parity emits fresh `ConstInt` values inline
    /// (`history.py:227`) instead of allocating pool entries, so this is
    /// initialized in `with_constants` but has no writer until a backend
    /// egress path needs pooled constants again.
    #[allow(dead_code)]
    // rewrite.py constant-pool cursor; history.py ConstInt is emitted inline
    next_const_idx: u32,

    /// rewrite.py `_changed_op` / `_changed_op_to`.
    ///
    /// When `remove_tested_failarg` rewrites an upcoming guard's failargs
    /// to substitute the tested box with a fresh `SAME_AS_I`, the guard is
    /// remembered here with its copy; the main loop swaps the copy in when
    /// it reaches that guard.
    _changed_op: Option<OpRc>,
    _changed_op_to: Option<OpRc>,

    /// rewrite.py:352 `gcrefs_output_list` — the per-loop list of
    /// reference constants pulled out of operations by `remove_constptr`.
    /// The backend builds a `GcTable` from this and emits
    /// `LoadFromGcTable(index)` against its base.
    gcrefs_output_list: Vec<GcRef>,
    /// rewrite.py:353 `gcrefs_map` — dedup map from a reference constant
    /// (keyed by `GcRef.0`) to its index in `gcrefs_output_list`.
    gcrefs_map: FxIndexMap<usize, u32>,
    /// rewrite.py:354 `gcrefs_recently_loaded` — CSE cache from a gc_table
    /// index to the `LoadFromGcTable` result box already emitted in the
    /// current basic block. Reset at every Label (rewrite.py:1005). Reuse
    /// across a can-collect point is sound: the load result is a Ref box,
    /// so the register allocator keeps it in the GC map and the collector
    /// forwards the held copy.
    gcrefs_recently_loaded: FxIndexMap<u32, Operand>,
}

impl<'a> RewriteState<'a> {
    fn with_constants(
        hint: usize,
        next_pos: u32,
        constants: &'a ConstMap<Const>,
    ) -> RewriteState<'a> {
        // P3 category E — `constants` is an index-keyed constant pool
        // (raw u32 key), not a Box-identity dict.  Bit-helpers replace
        // the `OpRef::from_raw(k).is_constant()` round-trip that would
        // land on the to-be-retired `OpRef::Untyped` variant.
        let next_const_idx = constants
            .keys()
            .filter(|&&k| OpRef::raw_is_constant(k))
            .map(|&k| OpRef::raw_const_index(k))
            .max()
            .map_or(0, |m| m + 1);
        RewriteState {
            out: Vec::with_capacity(hint + hint / 4),
            next_pos,
            constants,
            pending_malloc_idx: None,
            pending_malloc_total: 0,
            previous_size: 0,
            last_malloced_ref: Operand::none(),
            wb_applied: IndexSet::default(),
            known_lengths: IndexMap::default(),
            pending_zeros: Vec::new(),
            initialized_indices: IndexMap::default(),
            _delayed_zero_setfields: IndexMap::default(),
            _constant_additions: IndexMap::default(),
            next_const_idx,
            _changed_op: None,
            _changed_op_to: None,
            gcrefs_output_list: Vec::new(),
            gcrefs_map: IndexMap::default(),
            gcrefs_recently_loaded: IndexMap::default(),
        }
    }

    /// Resolve a constant value from the box (RPython parity: caller
    /// passes the Box; `isinstance(box, ConstInt) and box.getint()`).
    /// history.py/268/314 — inline-Const variants carry their value
    /// directly; legacy pool-indexed variants look up via the snapshot.
    fn resolve_constant(&self, b: &Operand) -> Option<i64> {
        // rewrite.py:548/590/1013 gate these reads on `isinstance(arg,
        // ConstInt)` → `getint()`. ConstFloat/ConstPtr have
        // no `getint` (history.py/314), so a Float/Ptr constant is
        // rejected outright. ConstInt carries its value inline
        // (history.py). A non-Const box that the backend proved
        // constant resolves through the index-keyed `constants` snapshot
        // by raw OpRef key; production pools are keyed in the constant
        // namespace (CONST_BIT set) so a genuine box position never
        // collides and stays unresolved.
        match b.const_value() {
            Some(Value::Int(v)) => Some(v),
            Some(_) => None,
            None => self
                .constants
                .get(&b.to_opref().raw())
                .map(|c| c.as_raw_i64()),
        }
    }

    /// Emit a fresh constant OpRef for `value`.
    ///
    /// rewrite.py:149/671/682 parity: RPython constructs a new
    /// `ConstInt(value)` at each call site. history.py
    /// `ConstInt.value` is inline on the Box; pyre mints
    /// `OpRef::ConstInt(value)` with the value carried inline
    /// per history.py `ConstInt.type = 'i'`.
    fn const_int(&mut self, value: i64) -> Operand {
        Operand::const_from_value(Value::Int(value))
    }

    /// rewrite.py `_gcref_index` — dedup a reference constant
    /// into `gcrefs_output_list`, returning its stable index.
    fn gcref_index(&mut self, gcref: GcRef) -> u32 {
        if let Some(&index) = self.gcrefs_map.get(&gcref.0) {
            return index;
        }
        let index = self.gcrefs_output_list.len() as u32;
        self.gcrefs_map.insert(gcref.0, index);
        self.gcrefs_output_list.push(gcref);
        index
    }

    /// rewrite.py `remove_constptr` — replace a reference
    /// constant with the result of a `LoadFromGcTable(index)` load,
    /// reusing one already emitted in this basic block (CSE) when
    /// possible.
    fn remove_constptr(&mut self, gcref: GcRef) -> Operand {
        let index = self.gcref_index(gcref);
        // rewrite.py `remove_constptr`: reuse `gcrefs_recently_loaded[index]`.
        if let Some(load) = self.gcrefs_recently_loaded.get(&index) {
            return load.clone();
        }
        // rewrite.py `remove_constptr` appends the `LOAD_FROM_GC_TABLE`
        // straight onto `_newops`. The only arg is `ConstInt`, so this
        // must not re-enter `emit` / `remove_constptrs_in`.
        let index_box = Operand::const_from_value(Value::Int(index as i64));
        let load_op = OpRc::new(mk_op(OpCode::LoadFromGcTable, &[index_box]));
        let pos = OpRef::ref_op(self.next_pos);
        self.next_pos += 1;
        load_op.pos().set(pos);
        let load = self.push_emitted(load_op, pos);
        self.gcrefs_recently_loaded.insert(index, load.clone());
        load
    }

    /// rewrite.py `emit_op` arg loop — substitute each non-null
    /// reference-constant argument with a `LoadFromGcTable` result so no
    /// raw `GcRef` is baked into the backend. JIT_DEBUG keeps its
    /// constants inline (rewrite.py:105 `keep`). Null pointers stay inline
    /// (rewrite.py `bool(arg.value)`). Failargs are NOT processed —
    /// they are resolved as plain constants in `rewrite_op` (rewrite.py:121
    /// `get_box_replacement`), matching upstream.
    ///
    /// rewrite.py:123-126 additionally appends the op's *descr* gcref to
    /// the table when the descr itself is a GC object. pyre descrs are
    /// `Arc<FailDescrCell>` pinned via the CLT keepalive
    /// (`asmmemmgr_gcreftracers`), not moving-GC objects, so there is no
    /// descr gcref to forward and that branch is intentionally not ported.
    fn remove_constptrs_in(&mut self, op: &Op) {
        if op.opcode == OpCode::JitDebug {
            return;
        }
        for i in 0..op.num_args() {
            if let Some(Value::Ref(gcref)) = op.arg(i).const_value()
                && !gcref.is_null()
            {
                let load = self.remove_constptr(gcref);
                op.setarg(i, load);
            }
        }
    }

    /// Emit an op. Void ops do not consume a result id.
    ///
    /// The result OpRef carries its own type via the variant tag
    /// (`int_op`/`float_op`/`ref_op`) — rewrite.py:930 `v.type` parity —
    /// so no separate type table is maintained.
    fn emit(&mut self, op: Op) -> Operand {
        self.emit_rc(OpRc::new(op))
    }

    fn emit_rc(&mut self, op: OpRc) -> Operand {
        self.remove_constptrs_in(&op);
        let rt = op.result_type();
        let pos = if rt == Type::Void {
            OpRef::NONE
        } else {
            let pos = match rt {
                Type::Int => OpRef::int_op(self.next_pos),
                Type::Float => OpRef::float_op(self.next_pos),
                Type::Ref => OpRef::ref_op(self.next_pos),
                Type::Void => unreachable!("Void filtered above"),
            };
            self.next_pos += 1;
            pos
        };
        op.pos().set(pos);
        self.push_emitted(op, pos)
    }

    /// Emit a result-producing op, preserving the provided position when the
    /// source trace already assigned one.
    ///
    /// The result OpRef carries its own type via the variant tag, so no
    /// separate type table is maintained — see `emit()` doc.
    fn emit_result(&mut self, op: Op, preferred_pos: OpRef) -> Operand {
        self.emit_result_rc(OpRc::new(op), preferred_pos)
    }

    fn emit_result_rc(&mut self, op: OpRc, preferred_pos: OpRef) -> Operand {
        self.remove_constptrs_in(&op);
        let rt = op.result_type();
        let pos = if preferred_pos.is_none() {
            let pos = OpRef::op_typed(self.next_pos, rt);
            self.next_pos += 1;
            pos
        } else {
            preferred_pos
        };
        op.pos().set(pos);
        self.push_emitted(op, pos)
    }

    fn push_emitted(&mut self, rc: OpRc, pos: OpRef) -> Operand {
        let result = if pos.is_none() || rc.result_type() == Type::Void {
            Operand::none()
        } else {
            Operand::from_bound_op(&rc)
        };
        self.out.push(rc);
        result
    }

    /// rewrite.py `emit_op` — append `op` itself unless an arg (or a
    /// guard failarg) was replaced, in which case `copy_and_change`.
    fn emit_op(&mut self, op: &OpRc) -> Operand {
        let keep = op.opcode == OpCode::JitDebug;
        let mut replaced = false;
        let out = op.clone();
        for i in 0..out.num_args() {
            let orig = out.arg(i);
            let mut arg = self.get_box_replacement(orig.clone());
            if !keep
                && let Some(Value::Ref(gcref)) = arg.const_value()
                && !gcref.is_null()
            {
                arg = self.remove_constptr(gcref);
            }
            if arg != orig {
                // The op IS the box (resoperation.py). setarg writes the
                // replacement on this ResOperation; a second cls() would
                // only mint an identical identity for get_box_replacement
                // to find via _forwarded.
                out.setarg(i, arg);
                replaced = true;
            }
        }
        if out.opcode.is_guard() {
            let failargs_changed = out
                .guard_fail_args()
                .is_some_and(|fa| fa.iter().any(|a| self.get_box_replacement(a.clone()) != *a));
            if failargs_changed {
                out.map_failargs_in_place(|a| {
                    *a = self.get_box_replacement(a.clone());
                });
                replaced = true;
            }
        }
        let rt = out.result_type();
        let pos = if replaced {
            if rt == Type::Void {
                OpRef::NONE
            } else if op.pos().get().is_none() {
                let p = OpRef::op_typed(self.next_pos, rt);
                self.next_pos += 1;
                p
            } else {
                op.pos().get()
            }
        } else {
            op.pos().get()
        };
        out.pos().set(pos);
        self.push_emitted(out, pos)
    }

    /// rewrite.py emitting_an_operation_that_can_collect
    fn emitting_an_operation_that_can_collect(&mut self) {
        self.pending_malloc_idx = None;
        self.wb_applied.clear();
        self.emit_pending_zeros();
        // rewrite.py:708-711: clear _constant_additions here, not only
        // in emit_label, to avoid keeping the older boxes alive across
        // a potentially-collecting op.
        self._constant_additions.clear();
    }

    /// rewrite.py record_int_add_or_sub.
    ///
    /// When `op` is `INT_ADD/INT_ADD_OVF/INT_SUB/INT_SUB_OVF` whose
    /// non-result operand is a `ConstInt`, remember the (older box,
    /// constant) pair so a downstream GC_STORE_INDEXED / GC_LOAD_INDEXED
    /// emit can fold the constant into its `offset` argument via
    /// `_try_use_older_box`.
    ///
    /// `is_subtraction` distinguishes INT_SUB (negate the constant
    /// before storing) from INT_ADD.  Mirrors rewrite.py:1015.
    fn record_int_add_or_sub(&mut self, op: &Op, is_subtraction: bool) {
        let v_arg0 = op.arg(0);
        let v_arg1 = op.arg(1);
        let (box_arg, mut constant) = if let Some(c) = self.resolve_constant(&v_arg1) {
            let signed = if is_subtraction { -c } else { c };
            (v_arg0, signed)
        } else if !is_subtraction {
            // rewrite.py:1019-1024: int_add only — try arg0 as constant.
            let Some(c) = self.resolve_constant(&v_arg0) else {
                return;
            };
            (v_arg1, c)
        } else {
            return;
        };
        // rewrite.py:1026-1030 invariant: if box itself is a recorded
        // sum, fold its constant in and chain to the older origin.
        let box_arg =
            if let Some((older, extra)) = self._constant_additions.get(&box_arg.to_opref()) {
                constant += *extra;
                older.clone()
            } else {
                box_arg
            };
        self._constant_additions
            .insert(op.pos().get(), (box_arg, constant));
    }

    /// rewrite.py _try_use_older_box.
    ///
    /// If `index_box` is a recorded `_constant_additions` entry, replace
    /// it with the older box and add `factor * extra_offset` to
    /// `offset`.
    fn _try_use_older_box(&self, index_box: &Operand, factor: i64, offset: i64) -> (Operand, i64) {
        if let Some((older, extra)) = self._constant_additions.get(&index_box.to_opref()) {
            return (older.clone(), offset + factor * *extra);
        }
        (index_box.clone(), offset)
    }

    fn remember_wb(&mut self, r: &Operand) {
        self.wb_applied.insert(r.to_opref());
    }

    /// rewrite.py: remember_known_length
    fn remember_known_length(&mut self, op: &Operand, length: usize) {
        self.known_lengths.insert(op.to_opref(), length);
    }

    /// rewrite.py: known_length(op, default)
    fn known_length(&self, op: &Operand, default: usize) -> usize {
        self.known_lengths
            .get(&op.to_opref())
            .copied()
            .unwrap_or(default)
    }

    /// rewrite.py: write_barrier_applied(op)
    fn wb_already_applied(&self, r: &Operand) -> bool {
        self.wb_applied.contains(&r.to_opref())
    }

    /// rewrite.py:930 parity: `v.type` — the Box carries its type
    /// intrinsically (history.py:220/261/307).
    fn result_type_of(&self, b: &Operand) -> Option<Type> {
        if b.is_none() {
            return None;
        }
        Some(b.type_())
    }

    /// rewrite.py:930 parity: `isinstance(v, ConstPtr) and not needs_write_barrier(v.value)`.
    /// A null ConstPtr never needs a write barrier.
    fn is_null_constant(&self, b: &Operand) -> bool {
        // history.py ConstPtr.value inline — null sentinel directly readable.
        match b.const_value() {
            Some(Value::Ref(r)) => r.0 == 0,
            Some(Value::Int(v)) => v == 0,
            _ => false,
        }
    }

    /// rewrite.py `get_box_replacement(op, allow_none)`: follow the
    /// `_forwarded` chain `replace_op_with` set.
    fn get_box_replacement(&self, r: Operand) -> Operand {
        r.get_box_replacement(false)
    }

    #[allow(dead_code)]
    fn rewrite_op(&self, op: &Op) -> Op {
        let rewritten = op.clone();
        // optimizer.py force_box loop parity:
        //   for i in range(op.numargs()): op.setarg(i, ...)
        for i in 0..rewritten.num_args() {
            rewritten.setarg(i, self.get_box_replacement(rewritten.arg(i)));
        }
        if let Some(fail_args) = rewritten.fail_args_mut() {
            for arg in fail_args.iter_mut() {
                // Same shed as `setarg` above: a forwarding target bound to
                // its producer stays a live-tracking operand.
                *arg = self.get_box_replacement(arg.clone());
            }
        }
        rewritten.pos().set(OpRef::NONE);
        rewritten
    }

    /// rewrite.py `replace_op_with(op, newop)`: `op.set_forwarded(newop)`.
    fn replace_op_with(&self, op: &OpRc, newop: &Operand) {
        let target = newop
            .bound_op()
            .expect("replace_op_with: the replacement is an emitted op");
        op.set_forwarded_op(&target);
    }

    fn emit_rewritten_from(&mut self, original: &OpRc, rewritten: Op) -> Operand {
        if original.result_type() == Type::Void {
            return self.emit(rewritten);
        }
        let result = self.emit_result(rewritten, original.pos().get());
        self.replace_op_with(original, &result);
        result
    }

    /// rewrite.py `replace_op_with(op, newload)` for a lowered memory
    /// accessor. The main loop emits `lowered` where it reaches `original`.
    fn set_forwarded(&self, original: &OpRc, lowered: Op) {
        original.set_forwarded_op(&OpRc::new(lowered));
    }

    /// rewrite.py `emit_op`, whose `op = self.get_box_replacement(op)`
    /// picks up the lowered form `transform_to_gc_load` forwarded `op` to.
    fn emit_maybe_forwarded(&mut self, original: &OpRc) -> Operand {
        match original.get_forwarded() {
            Forwarded::Op(lowered) => self.emit_forwarded(original, lowered),
            _ => self.emit_op(original),
        }
    }

    /// Emit the lowered op `original` is forwarded to, at `original`'s
    /// position.
    fn emit_forwarded(&mut self, original: &OpRc, lowered: OpRc) -> Operand {
        if original.result_type() == Type::Void {
            return self.emit_rc(lowered);
        }
        self.emit_result_rc(lowered, original.pos().get())
    }

    /// rewrite.py `delayed_zero_setfields(op)` — get-or-create the
    /// per-base byte-offset set, resolving `r` through the forwarding
    /// map first (RPython calls `get_box_replacement(op)` here).
    fn delayed_zero_setfields(&mut self, r: &Operand) -> &mut FxIndexSet<i64> {
        let base = self.get_box_replacement(r.clone());
        &mut self
            ._delayed_zero_setfields
            .entry(base.to_opref())
            .or_insert_with(|| (base, FxIndexSet::default()))
            .1
    }

    /// Record that a SETARRAYITEM wrote to `array_ref[index]`,
    /// so the pending zero for that slot can be skipped.
    fn record_setarrayitem_index(&mut self, array_ref: &Operand, index: usize) {
        let key = array_ref.to_opref();
        if self.pending_zeros.iter().any(|pz| pz.array_ref == key) {
            self.initialized_indices
                .entry(key)
                .or_default()
                .insert(index);
        }
    }

    /// rewrite.py emit_pending_zeros.
    ///
    /// Mutates each previously-emitted ZERO_ARRAY in place: trim from
    /// both ends past any indices that subsequent SETARRAYITEM writes
    /// covered, then rewrite arg(1)/arg(2) as byte offset / byte
    /// length (multiplied by the array's `scale`) and arg(3)/arg(4)
    /// to ConstInt(1) so the backend treats arg(1)/arg(2) as raw
    /// bytes.  Length 0 leaves the op as a no-op for the backend
    /// (matches rewrite.py:754 "may be ConstInt(0)").
    fn emit_pending_zeros(&mut self) {
        let pending = std::mem::take(&mut self.pending_zeros);
        let inited = std::mem::take(&mut self.initialized_indices);

        for pz in pending {
            let written = inited.get(&pz.array_ref);

            // rewrite.py:744-753 trim-from-front / trim-from-back.
            let mut start: usize = 0;
            while start < pz.length && written.is_some_and(|s| s.contains(&start)) {
                start += 1;
            }
            let mut stop: usize = pz.length;
            while stop > start && written.is_some_and(|s| s.contains(&(stop - 1))) {
                stop -= 1;
            }
            let scaled_start = self.const_int(start as i64 * pz.scale);
            let scaled_len = self.const_int((stop - start) as i64 * pz.scale);
            let one = self.const_int(1);

            let op = &self.out[pz.out_index];
            // resoperation.py AbstractResOp.setarg parity.
            op.setarg(1, scaled_start);
            op.setarg(2, scaled_len);
            op.setarg(3, one.clone());
            op.setarg(4, one);
        }

        // rewrite.py:760-766 — NULL-pointer writes still pending for
        // any zero-init fields not covered by a subsequent explicit
        // SETFIELD_GC.  The store width is `WORD`, the target's pointer
        // size (`symbolic.py WORD = sizeof(lltype.Signed)`) — 4 on the
        // wasm32 target, where an 8-byte zero would run past the field
        // and null the neighbouring one.
        //
        // The constant path inside `emit_gc_store_or_indexed`
        // (rewrite.py:148-150) collapses (ConstInt(ofs), factor=1,
        // offset=0) to a plain `GC_STORE(ptr, ConstInt(ofs),
        // ConstInt(0), ConstInt(WORD))`, which is what we emit here
        // directly.
        let pending_zsf = std::mem::take(&mut self._delayed_zero_setfields);
        for (_, (v, entries)) in pending_zsf {
            // rewrite.py `v = self.get_box_replacement(v)`.
            let ptr_box = self.get_box_replacement(v);
            for ofs in entries.iter().copied() {
                let ofs_ref = self.const_int(ofs);
                let zero_ref = self.const_int(0);
                let word_ref = self.const_int(std::mem::size_of::<usize>() as i64);
                let store = mk_op(
                    OpCode::GcStore,
                    &[ptr_box.clone(), ofs_ref, zero_ref, word_ref],
                );
                self.emit(store);
            }
        }
    }
}

impl GcRewriterImpl {
    /// Can we use the nursery for this allocation size?
    fn can_use_nursery(&self, size: usize) -> bool {
        // gc.py `can_use_nursery_malloc`: `max_size_of_young_obj` is
        // `large_object` (`nonlarge_max + 1`), hence an exclusive boundary.
        size < self.max_nursery_size
    }

    /// rewrite.py `could_merge_with_next_guard` parity.
    ///
    /// Returns true when `op` should be kept adjacent to the next guard,
    /// triggering a `emit_pending_zeros` flush at the top of the iteration
    /// (rewrite.py:376-377). Two cases:
    ///   * `op` is an overflow-producing arithmetic op (INT_*_OVF),
    ///     which pairs with a following GUARD_NO_OVERFLOW / GUARD_OVERFLOW.
    ///   * `op` is a comparison whose boolean result is tested by the
    ///     immediately following GUARD_TRUE/GUARD_FALSE/COND_CALL. In that
    ///     case the tested value appearing in the guard's failargs is
    ///     hoisted out into a dedicated `SAME_AS_I(0/1)` via
    ///     `remove_tested_failarg`.
    fn could_merge_with_next_guard(
        &self,
        op: &Op,
        i: usize,
        ops: &[OpRc],
        st: &mut RewriteState<'_>,
    ) -> bool {
        if !op.opcode.is_comparison() {
            // rewrite.py:436 fallback: int_xxx_ovf + guard_{,no_}overflow
            return op.opcode.is_ovf();
        }
        if i + 1 >= ops.len() {
            return false;
        }
        let next_op = &ops[i + 1];
        // rewrite.py:441-443 — merge only with a directly-consuming guard/cond_call.
        // RPython's `rop.COND_CALL` is the void-result variant, matching
        // pyre's `CondCallN`.
        if !matches!(
            next_op.opcode,
            OpCode::GuardTrue | OpCode::GuardFalse | OpCode::CondCallN
        ) {
            return false;
        }
        // rewrite.py `next_op.getarg(0) is not op` — in pyre OpRef
        // carries the same identity role as RPython's box object.
        if next_op.arg(0).to_opref() != op.pos().get() {
            return false;
        }
        self.remove_tested_failarg(next_op, st);
        true
    }

    /// rewrite.py `remove_tested_failarg` parity.
    ///
    /// When a GUARD_TRUE/GUARD_FALSE's tested value is also present in the
    /// guard's failargs, emit a `SAME_AS_I(value)` (where `value = 0` for
    /// GUARD_TRUE / `1` for GUARD_FALSE — the constant the tested box would
    /// hold on the failure path) and rewrite the failargs list so the
    /// guard points at that SAME_AS_I instead of the boolean. The rewritten
    /// guard is remembered in `st._changed_op` / `st._changed_op_to` so the
    /// main loop substitutes it on the next iteration.
    fn remove_tested_failarg(&self, op: &OpRc, st: &mut RewriteState<'_>) {
        // rewrite.py:452-453: no-op for non-GUARD_{TRUE,FALSE} (e.g. COND_CALL
        // is merge-eligible via could_merge_with_next_guard but does not
        // carry failargs in the RPython sense).
        if !matches!(op.opcode, OpCode::GuardTrue | OpCode::GuardFalse) {
            return;
        }
        let fail_args = match op.guard_fail_args() {
            Some(fa) if !fa.is_empty() => fa,
            _ => return,
        };
        let target = op.arg(0).to_opref();
        // rewrite.py:456-459: guard's failargs contain the tested box?
        let Some(idx) = fail_args.iter().position(|a| a.to_opref() == target) else {
            return;
        };
        // rewrite.py `value = int(opnum == rop.GUARD_FALSE)`
        let value: i64 = i64::from(op.opcode == OpCode::GuardFalse);
        let const_ref = st.const_int(value);
        let same = mk_op(OpCode::SameAsI, &[const_ref]);
        let same_pos = st.emit_result(same, OpRef::NONE);

        // rewrite.py — rewrite failargs + stash the copy-and-changed
        // guard for the next iteration to pick up.
        let new_guard = (**op).clone();
        if let Some(fa) = new_guard.fail_args_mut() {
            // `same_pos` is bound to the freshly-emitted SAME_AS producer
            // (emit_result returns `from_bound_op`), so lower it to the
            // live-tracking `Operand::Op` rather than a frozen position-only
            // `Operand::Box` (#9): the failarg then auto-tracks the producer's
            // `op.pos` if it is renumbered, matching RPython's box-identity
            // failarg.
            fa[idx] = same_pos;
        }
        // pos is reassigned when emit/emit_result runs on the substituted op.
        new_guard.pos().set(OpRef::NONE);
        st._changed_op = Some(op.clone());
        st._changed_op_to = Some(OpRc::new(new_guard));
    }

    // ────────────────────────────────────────────────────────
    // NEW / NEW_WITH_VTABLE  → CALL_MALLOC_NURSERY + tid init
    // ────────────────────────────────────────────────────────

    fn handle_new(&self, op: &OpRc, st: &mut RewriteState<'_>) {
        let descr_arc = op.getdescr().expect("NEW must have a SizeDescr");
        let descr = descr_arc
            .as_size_descr()
            .expect("NEW descr must be SizeDescr");

        if self.headerless_fixedsize {
            // `gen_malloc_nursery` batches consecutive bumps. The size is
            // the payload: no `GcHeader`, no tid store. A size the nursery
            // declines goes to `malloc_fixedsize` at the same raw size.
            let size = round_up(descr.size());
            let result_pos = op.pos().get();
            let obj_ref = if descr.non_moving() {
                self.gen_malloc_fixedsize(size, descr.type_id(), true, result_pos, st)
            } else if let Some(r) = self.gen_malloc_nursery(size, result_pos, st) {
                r
            } else {
                self.gen_malloc_fixedsize(size, descr.type_id(), false, result_pos, st)
            };
            st.replace_op_with(op, &obj_ref);
            if op.opcode == OpCode::NewWithVtable
                && let Some(vtable_fd_ref) = self.fielddescr_vtable.as_ref()
            {
                let vtable = descr.vtable();
                if vtable != 0 {
                    self.gen_initialize_vtable(obj_ref.clone(), vtable, vtable_fd_ref, st);
                }
            }
            self.clear_gc_fields(descr, obj_ref, st);
            return;
        }

        if descr.headerless() {
            let size = round_up(descr.size());
            let result_pos = op.pos().get();
            let obj_ref = if self.can_use_nursery(size) {
                // No tid store: the payload is a raw nursery bump.  GC pointer
                // fields still go through `clear_gc_fields` below
                // (`handle_new_fixedsize`), so a leftover slot is NULLed at the
                // next can-collect op unless `consider_setfield_gc` drops it.
                st.emitting_an_operation_that_can_collect();
                let size_ref = st.const_int(size as i64);
                let malloc_op = mk_op(OpCode::CallMallocNurseryHeaderless, &[size_ref]);
                let obj_ref = st.emit_result(malloc_op, result_pos);
                st.last_malloced_ref = obj_ref.clone();
                st.remember_wb(&obj_ref);
                obj_ref
            } else {
                // rewrite.py:884-886 parity: if nursery malloc is disallowed,
                // decline to the same fixed-size non-nursery helper used by
                // the headered branch below.  That helper takes the headered
                // total size and stamps the type id itself.
                let total_size = round_up(descr.size() + crate::header::GcHeader::SIZE);
                self.gen_malloc_fixedsize(
                    total_size,
                    descr.type_id(),
                    descr.non_moving(),
                    result_pos,
                    st,
                )
            };
            st.replace_op_with(op, &obj_ref);
            if op.opcode == OpCode::NewWithVtable
                && let Some(vtable_fd_ref) = self.fielddescr_vtable.as_ref()
            {
                let vtable = descr.vtable();
                if vtable != 0 {
                    self.gen_initialize_vtable(obj_ref.clone(), vtable, vtable_fd_ref, st);
                }
            }
            self.clear_gc_fields(descr, obj_ref, st);
            return;
        }

        // rewrite.py handle_malloc_operation parity:
        // descr.size in RPython already includes the GC header (the
        // OBJECT type is built with `size = sizeof(header) + sizeof(fields)`).
        // pyre's PyreSizeDescr reports `obj_size` as the bare struct size
        // (e.g. `size_of::<W_IntObject>() == 16`) WITHOUT the GC header,
        // so we add it here so that CallMallocNursery sees the same
        // "object-with-header" total that the cranelift backend expects
        // (it strips the header back off before passing to the alloc
        // shim's payload size).
        let size = round_up(descr.size() + crate::header::GcHeader::SIZE);
        let type_id = descr.type_id();

        // rewrite.py — `if gen_malloc_nursery(size, op):
        //                          gen_initialize_tid(op, descr.tid)
        //                       else:
        //                          gen_malloc_fixedsize(size, descr.tid, op)`.
        // The fast path stamps the tid header inline; the slow path's
        // CALL_R helper (`malloc_big_fixedsize`) does it inside the
        // helper so `gen_malloc_fixedsize` does NOT call
        // `gen_initialize_tid` after the fact.
        // A descr marked `non_moving` declines the nursery the same way an
        // oversized one does, and lands on the same generator; there it picks
        // `malloc_big_fixedsize_oldgen_fn`, because the upstream helper answers
        // by size and would put a small non-moving struct in the nursery.
        let nursery_ref = if descr.non_moving() {
            None
        } else {
            self.gen_malloc_nursery(size, op.pos().get(), st)
        };
        let obj_ref = match nursery_ref {
            Some(r) => {
                self.gen_initialize_tid(r.clone(), type_id, st);
                r
            }
            None => {
                self.gen_malloc_fixedsize(size, type_id, descr.non_moving(), op.pos().get(), st)
            }
        };
        st.replace_op_with(op, &obj_ref);

        // rewrite.py handle_malloc_operation parity:
        //   elif opnum == rop.NEW_WITH_VTABLE:
        //       ...
        //       if self.gc_ll_descr.fielddescr_vtable is not None:
        //           self.emit_setfield(op, ConstInt(descr.get_vtable()),
        //                              descr=self.gc_ll_descr.fielddescr_vtable)
        //
        // Emit the vtable setfield SYNCHRONOUSLY (matching RPython). A
        // previous pyre-only deferral stored (obj, vtable) pairs in
        // `pending_vtable_inits` and flushed them on the next non-setfield
        // op; that left the object's ob_type slot uninitialized across
        // intermediate setfields and, when a guard fired in that window,
        // fail_args captured a partially-initialized nursery pointer whose
        // ob_type=NULL eventually crashed the blackhole's binary_op_fn
        // path (memory: phase5_super_lift_bisect_2026_04_17.md).
        if op.opcode == OpCode::NewWithVtable {
            // rewrite.py `if self.gc_ll_descr.fielddescr_vtable is not None`.
            if let Some(vtable_fd_ref) = self.fielddescr_vtable.as_ref() {
                let vtable = descr.vtable();
                // Defensive — pyre's NEW_WITH_VTABLE descrs in production
                // always carry a non-zero vtable; some test fixtures
                // synthesize 0, in which case skip the store rather than
                // emit a NULL typeptr.
                if vtable != 0 {
                    self.gen_initialize_vtable(obj_ref.clone(), vtable, vtable_fd_ref, st);
                }
            }
        }
        // rewrite.py `self.clear_gc_fields(descr, op)` — record every
        // GC-pointer field's byte offset so a pending NULL store is
        // emitted at the next flush point, unless cleared first by an
        // explicit SETFIELD_GC (`consider_setfield_gc`).
        self.clear_gc_fields(descr, obj_ref.clone(), st);

        // Pyre's layout has a `w_class` GCREF beside the vtable. Stamp it
        // after `clear_gc_fields` so `emit_setfield` can drop the delayed
        // NULL the same way a later SETFIELD_GC would
        // (`rewrite.py` `consider_setfield_gc`).
        if let Some(w_class) = descr.w_class_obj()
            && w_class != 0
            && let Some(w_class_fd) = descr.class_word_field()
        {
            self.gen_initialize_w_class(obj_ref, w_class, w_class_fd.as_ref(), st);
        }
    }

    /// rewrite.py `clear_gc_fields`.
    ///
    /// For every GC-pointer field on the fresh allocation, remember
    /// that a NULL-pointer store is needed unless a subsequent
    /// SETFIELD_GC overwrites it first.  Early-returns when the
    /// allocator already zero-fills payload bytes
    /// (`self.malloc_zero_filled`, rewrite.py:499-500).
    fn clear_gc_fields(&self, descr: &dyn SizeDescr, result: Operand, st: &mut RewriteState<'_>) {
        if self.malloc_zero_filled {
            return;
        }
        // rewrite.py — populate `delayed_zero_setfields[result][ofs] = None`
        // per GC-pointer field (`descr.gc_fielddescrs` / unpack_fielddescr).
        let entries = st.delayed_zero_setfields(&result);
        for fd in descr.gc_fielddescrs() {
            entries.insert(fd.offset() as i64);
        }
    }

    // ────────────────────────────────────────────────────────
    // NEW_ARRAY / NEW_ARRAY_CLEAR  → CALL_MALLOC_NURSERY_VARSIZE / CALL_R
    // ────────────────────────────────────────────────────────

    /// rewrite.py handle_new_array parity.
    ///
    /// kind: FLAG_ARRAY=0, FLAG_STR=1, FLAG_UNICODE=2.
    ///
    /// `descr_ref` is the ArrayDescr to use for size / length-field /
    /// per-item layout queries.  Upstream rewrite.py:489-494 passes
    /// `self.gc_ll_descr.{str,unicode}_descr` for NEWSTR/NEWUNICODE and
    /// `op.getdescr()` for NEW_ARRAY; the dispatcher in
    /// `rewrite_for_gc_with_constants` is what threads the right
    /// instance through this signature.
    ///
    /// TODO: pyre still lacks the Boehm branch
    /// (`gen_boehm_malloc_array`).  Framework-GC path #4
    /// (`gen_malloc_array` / `gen_malloc_str` / `gen_malloc_unicode`)
    /// is ported below and emits CALL_R + CHECK_MEMORY_ERROR like
    /// rewrite.py:768-846.
    fn handle_new_array(
        &self,
        descr_ref: DescrRef,
        op: &OpRc,
        st: &mut RewriteState<'_>,
        kind: i64,
    ) {
        let descr = descr_ref
            .as_array_descr()
            .expect("handle_new_array descr must be ArrayDescr");

        let item_size = descr.item_size();
        // A `non_moving` descr declines both nursery routes and lands on the
        // typed malloc helpers, whose old-generation twin never moves the
        // block.  Same opt-in as `SizeDescr::non_moving` on the fixed-size
        // path.
        let non_moving = descr.non_moving();
        let v_length = st.get_box_replacement(op.arg(0)); // the length operand
        let length_const = st.resolve_constant(&v_length);

        // rewrite.py:548-558 — total_size for the constant-size /
        // zero-itemsize fast path.  Stays at -1 when v_length is a
        // ConstInt that overflows `basesize + itemsize * num_elem`,
        // matching upstream's `OverflowError: pass`.
        let mut total_size: i64 = -1;
        if let Some(num_elem) = length_const {
            if num_elem >= 0
                && let Some(var_size) = (item_size as i64).checked_mul(num_elem)
                && let Some(t) = (descr.base_size() as i64).checked_add(var_size)
            {
                total_size = t;
            }
        } else if item_size == 0 {
            // rewrite.py:557-558 — non-const length but zero itemsize
            // means no variable payload; fold to fixed-size basesize.
            total_size = descr.base_size() as i64;
        } else if !non_moving && self.can_use_nursery(1) {
            // rewrite.py:559-568 path #1 — varsize nursery fast path.
            // Returns None when the descr is non-standard FLAG_ARRAY,
            // in which case we fall through to path #4
            // (`gen_malloc_array_nonstandard`) per rewrite.py:853-856.
            if let Some(r) = self.gen_malloc_nursery_varsize(
                descr_ref.clone(),
                kind,
                v_length.clone(),
                op.pos().get(),
                st,
            ) {
                st.replace_op_with(op, &r);
                if let Some(len_descr) = descr.len_descr() {
                    self.gen_initialize_len(r.clone(), v_length.clone(), len_descr, st);
                }
                self.clear_varsize_gc_fields(
                    kind,
                    descr_ref.clone(),
                    item_size as i64,
                    r,
                    v_length,
                    op.opcode,
                    st,
                );
                return;
            }
        }

        // rewrite.py:569-584 paths #2 / #4.
        let result = if total_size >= 0 {
            // pyre layout note: gen_malloc_nursery expects HDR + payload
            // bytes (handle_new_fixedsize line 836); upstream's basesize
            // already includes the header offset.  Add HDR_SIZE here so
            // the bump-pointer alloc covers the same span.
            let s = if self.headerless_fixedsize {
                total_size as usize
            } else {
                crate::header::GcHeader::SIZE + total_size as usize
            };
            let nursery_ref = if non_moving {
                None
            } else {
                self.gen_malloc_nursery(s, op.pos().get(), st)
            };
            if let Some(r) = nursery_ref {
                // rewrite.py:569-572 path #2 — constant-size nursery.
                st.replace_op_with(op, &r);
                if !self.headerless_fixedsize {
                    self.gen_initialize_tid(r.clone(), descr.type_id(), st);
                }
                if let Some(len_descr) = descr.len_descr() {
                    self.gen_initialize_len(r.clone(), v_length.clone(), len_descr, st);
                }
                r
            } else {
                // rewrite.py:573-584 path #4 — typed slow malloc helpers.
                let r = match op.opcode {
                    OpCode::NewArray | OpCode::NewArrayClear => self.gen_malloc_array(
                        descr_ref.clone(),
                        v_length.clone(),
                        op.pos().get(),
                        st,
                    ),
                    OpCode::Newstr => self.gen_malloc_str(v_length.clone(), op.pos().get(), st),
                    OpCode::Newunicode => {
                        self.gen_malloc_unicode(v_length.clone(), op.pos().get(), st)
                    }
                    _ => panic!("unexpected varsize alloc opcode: {:?}", op.opcode),
                };
                st.replace_op_with(op, &r);
                r
            }
        } else {
            let r = match op.opcode {
                OpCode::NewArray | OpCode::NewArrayClear => {
                    self.gen_malloc_array(descr_ref.clone(), v_length.clone(), op.pos().get(), st)
                }
                OpCode::Newstr => self.gen_malloc_str(v_length.clone(), op.pos().get(), st),
                OpCode::Newunicode => self.gen_malloc_unicode(v_length.clone(), op.pos().get(), st),
                _ => panic!("unexpected varsize alloc opcode: {:?}", op.opcode),
            };
            st.replace_op_with(op, &r);
            r
        };

        // rewrite.py (path #1 inline) / rewrite.py
        // (paths #2/#3/#4 tail) clear_varsize_gc_fields.  Emits
        // ZERO_ARRAY for NEW_ARRAY_CLEAR and a hash-field zeroing
        // store for NEWSTR / NEWUNICODE, gated on !malloc_zero_filled.
        self.clear_varsize_gc_fields(
            kind,
            descr_ref.clone(),
            item_size as i64,
            result.clone(),
            v_length,
            op.opcode,
            st,
        );

        // rewrite.py:551: if isinstance(v_length, ConstInt):
        //     self.remember_known_length(op, v_length.getint())
        // Upstream calls this BEFORE total_size computation, but the key
        // is the OpRef of the alloc result, so the call is safely
        // hoisted to here without changing the semantic outcome.
        if let Some(num_elem) = length_const {
            st.remember_known_length(&result, num_elem as usize);
        }
    }

    /// rewrite.py `clear_varsize_gc_fields`.
    ///
    /// Short-circuits on `malloc_zero_filled=true` — pyre's production
    /// nursery zero-fills payload bytes, so callers already observe a
    /// zeroed array / hash field.  Under `malloc_zero_filled=false`
    /// this fans out per `kind`:
    ///   * FLAG_ARRAY + NEW_ARRAY_CLEAR → `handle_clear_array_contents`
    ///   * FLAG_STR / FLAG_UNICODE → zero the `hash` field at offset 0
    ///
    /// The upstream hash-field store comes from
    /// `gc_ll_descr.{str,unicode}_hash_descr`; rstr.STR and
    /// rstr.UNICODE both keep `hash` as the first Signed field
    /// (rstr.py:1226-1238), so a `GC_STORE(result, 0, 0, WORD)`
    /// matches the upstream `emit_setfield(result, c_zero,
    /// descr=hash_descr)` contract.
    #[allow(clippy::too_many_arguments)]
    fn clear_varsize_gc_fields(
        &self,
        kind: i64,
        arraydescr: DescrRef,
        ad_itemsize: i64,
        result: Operand,
        v_length: Operand,
        opnum: OpCode,
        st: &mut RewriteState<'_>,
    ) {
        if self.malloc_zero_filled {
            return;
        }
        // rewrite.py FLAG_ARRAY path.
        if kind == 0 {
            if opnum == OpCode::NewArrayClear {
                self.handle_clear_array_contents(arraydescr, ad_itemsize, result, v_length, st);
            }
            return;
        }
        // rewrite.py FLAG_STR / FLAG_UNICODE: zero the hash
        // field via emit_setfield(result, ConstInt(0), descr=hash_descr).
        // Offset / size come from gc_ll_descr.{str,unicode}_hash_descr
        // (gc.py:48-49) — both rstr.STR and rstr.UNICODE keep `hash` at
        // offset 0 with `Signed` size, but reading it through the
        // descr keeps the layout assumption explicit.
        if kind == 1 || kind == 2 {
            let hash_descr_ref = if kind == 1 {
                &self.str_hash_descr
            } else {
                &self.unicode_hash_descr
            };
            let hash_fd = hash_descr_ref
                .as_field_descr()
                .expect("gc_ll_descr.{str,unicode}_hash_descr must be a FieldDescr");
            let zero = st.const_int(0);
            self.emit_setfield(result, zero, hash_fd, st);
        }
    }

    /// rewrite.py `handle_clear_array_contents`.
    ///
    /// Emits a `ZERO_ARRAY` covering the entire array, registering the
    /// op in `pending_zeros` when `v_length` is a constant so
    /// `emit_pending_zeros` can trim the range against subsequent
    /// SETARRAYITEM_GC writes (rewrite.py:610-611).
    fn handle_clear_array_contents(
        &self,
        arraydescr: DescrRef,
        ad_itemsize: i64,
        v_arr: Operand,
        v_length: Operand,
        st: &mut RewriteState<'_>,
    ) {
        // rewrite.py:589 assert v_length is not None.
        if v_length.is_none() {
            return;
        }
        // rewrite.py:590-591 constant zero-length short-circuit.
        let length_const = st.resolve_constant(&v_length);
        if matches!(length_const, Some(0)) {
            return;
        }
        // rewrite.py:600-602 — only pre-scale for non-ConstInt v_length.
        // Upstream gates the helper call on `not isinstance(v_length,
        // ConstInt)`; for ConstInt the scale stays in args[3]/args[4]
        // and `emit_pending_zeros` handles it later.
        let mut scale = ad_itemsize;
        let mut v_length_scaled = v_length;
        if length_const.is_none() {
            let (new_scale, _new_offset, scaled_opt) = self
                ._emit_mul_if_factor_offset_not_supported(v_length_scaled.clone(), scale, 0, st);
            if let Some(s) = scaled_opt {
                v_length_scaled = s;
                scale = new_scale;
            }
        }
        // rewrite.py:603-609 emit ZERO_ARRAY with scale doubled into
        // args[3] and args[4] (upstream puts both to `ConstInt(scale)`;
        // emit_pending_zeros later rewrites both to 1 after byte-level
        // trim for ConstInt lengths).
        let c_zero = st.const_int(0);
        let c_scale_a = st.const_int(scale);
        let c_scale_b = st.const_int(scale);
        let zero_op = mk_op(
            OpCode::ZeroArray,
            &[v_arr.clone(), c_zero, v_length_scaled, c_scale_a, c_scale_b],
        );
        zero_op.setdescr(arraydescr);
        let out_index = st.out.len();
        st.emit(zero_op);
        // rewrite.py:610-611 — register in last_zero_arrays only for
        // ConstInt lengths so emit_pending_zeros can optimize the range.
        if let Some(n) = length_const {
            st.pending_zeros.push(PendingZero {
                out_index,
                array_ref: v_arr.to_opref(),
                length: n as usize,
                scale,
            });
        }
    }

    // ────────────────────────────────────────────────────────
    // CALL_MALLOC_NURSERY_VARSIZE / slow malloc helpers
    // ────────────────────────────────────────────────────────

    /// rewrite.py `gen_malloc_nursery_varsize`.
    ///
    /// Returns `Some(opref)` when the nursery fast path emits the
    /// CALL_MALLOC_NURSERY_VARSIZE op, `None` when the upstream
    /// `kind == FLAG_ARRAY` non-standard-shape guard
    /// (rewrite.py:853-856) rejects the descr — in that case the
    /// caller falls through to `gen_malloc_array` /
    /// `gen_boehm_malloc_array` like upstream's path #4.
    fn gen_malloc_nursery_varsize(
        &self,
        arraydescr: DescrRef,
        kind: i64,
        v_length: Operand,
        result_pos: OpRef,
        st: &mut RewriteState<'_>,
    ) -> Option<Operand> {
        let ad = arraydescr
            .as_array_descr()
            .expect("gen_malloc_nursery_varsize descr must be ArrayDescr");
        // rewrite.py — standard-shape gate: only `FLAG_ARRAY`
        // (kind == 0) is constrained; `FLAG_STR` (1) / `FLAG_UNICODE`
        // (2) always proceed because their descrs are by definition
        // non-standard-shaped but the nursery layout still accepts
        // them.
        const FLAG_ARRAY: i64 = 0;
        if kind == FLAG_ARRAY {
            let length_ofs = ad
                .len_descr()
                .map_or(self.standard_array_length_ofs, |fd| fd.offset());
            if ad.base_size() != self.standard_array_basesize
                || length_ofs != self.standard_array_length_ofs
            {
                return None;
            }
        }
        st.emitting_an_operation_that_can_collect();
        let kind_ref = st.const_int(kind);
        let itemsize_ref = st.const_int(ad.item_size() as i64);
        // Headerless: same args and descr, no `GcHeader` in the bump.
        // `CallMallocNurseryVarsize` still stamps a tid before the payload.
        let opcode = if self.headerless_fixedsize {
            OpCode::CallMallocNurseryVarsizeHeaderless
        } else {
            OpCode::CallMallocNurseryVarsize
        };
        let varsize_op = mk_op(opcode, &[kind_ref, itemsize_ref, v_length]);
        varsize_op.setdescr(arraydescr);
        // rewrite.py:863-866, said out loud because the fixed-size sibling
        // does the opposite and the difference is one absent call:
        //
        //     # don't record v_result into self.write_barrier_applied:
        //     # it can be a large, young array with card marking, and then
        //     # the GC relies on the write barrier being called
        //
        // No `remember_write_barrier` here. The length is not known at
        // rewrite time, so this array may be the one case a fresh allocation
        // is not exempt: `try_alloc_young_nonmoving_with_cards` gives a large
        // young array GCFLAG_HAS_CARDS, and a card is set by the barrier or
        // not at all.
        Some(st.emit_result(varsize_op, result_pos))
    }

    /// rewrite.py `_gen_call_malloc_gc`.
    fn gen_call_malloc_gc(
        &self,
        args: &[Operand],
        result_pos: OpRef,
        calldescr: DescrRef,
        st: &mut RewriteState<'_>,
    ) -> Operand {
        st.emitting_an_operation_that_can_collect();
        let call_op = mk_op(OpCode::CallR, args);
        call_op.setdescr(calldescr);
        let result = st.emit_result(call_op, result_pos);
        st.emit(mk_op(
            OpCode::CheckMemoryError,
            std::slice::from_ref(&result),
        ));
        result
    }

    /// rewrite.py `gen_malloc_array`.
    fn gen_malloc_array(
        &self,
        arraydescr: DescrRef,
        v_num_elem: Operand,
        result_pos: OpRef,
        st: &mut RewriteState<'_>,
    ) -> Operand {
        // rewrite.py `gen_boehm_malloc_array` when `malloc_fn_ptr` is published.
        if boehm_malloc_fn().is_some() {
            let ad = arraydescr
                .as_array_descr()
                .expect("gen_malloc_array descr must be ArrayDescr");
            let len_descr = ad.len_descr();
            let length_ofs = len_descr.map_or(self.standard_array_length_ofs, |fd| fd.offset());
            let fn_ref = st.const_int(crate::boehm_malloc_array as *const () as usize as i64);
            return self.gen_call_malloc_gc(
                &[
                    fn_ref,
                    st.const_int(ad.base_size() as i64),
                    v_num_elem,
                    st.const_int(ad.item_size() as i64),
                    st.const_int(length_ofs as i64),
                ],
                result_pos,
                boehm_malloc_array_descr(),
                st,
            );
        }
        let ad = arraydescr
            .as_array_descr()
            .expect("gen_malloc_array descr must be ArrayDescr");
        let len_descr = ad.len_descr();
        let length_ofs = len_descr.map_or(self.standard_array_length_ofs, |fd| fd.offset());
        let is_standard = ad.base_size() == self.standard_array_basesize
            && len_descr.is_some_and(|fd| fd.offset() == self.standard_array_length_ofs);
        // The old-generation twins take the same arguments as the nursery
        // helpers, so only the callee address changes.
        let non_moving = ad.non_moving();
        if is_standard {
            let fn_ref = st.const_int(if non_moving {
                self.malloc_array_oldgen_fn
            } else {
                self.malloc_array_fn
            });
            let itemsize_ref = st.const_int(ad.item_size() as i64);
            let tid = ad.type_id();
            majit_ir::descr::assert_array_tid_for_malloc(tid, "gen_malloc_array");
            let typeid_ref = st.const_int(tid as i64);
            self.gen_call_malloc_gc(
                &[fn_ref, itemsize_ref, typeid_ref, v_num_elem],
                result_pos,
                self.malloc_array_descr.clone(),
                st,
            )
        } else {
            let fn_ref = st.const_int(if non_moving {
                self.malloc_array_nonstandard_oldgen_fn
            } else {
                self.malloc_array_nonstandard_fn
            });
            let basesize_ref = st.const_int(ad.base_size() as i64);
            let itemsize_ref = st.const_int(ad.item_size() as i64);
            let lengthofs_ref = st.const_int(length_ofs as i64);
            let tid = ad.type_id();
            majit_ir::descr::assert_array_tid_for_malloc(tid, "gen_malloc_array");
            let typeid_ref = st.const_int(tid as i64);
            self.gen_call_malloc_gc(
                &[
                    fn_ref,
                    basesize_ref,
                    itemsize_ref,
                    lengthofs_ref,
                    typeid_ref,
                    v_num_elem,
                ],
                result_pos,
                self.malloc_array_nonstandard_descr.clone(),
                st,
            )
        }
    }

    /// rewrite.py `gen_malloc_fixedsize` (framework GC arm).
    ///
    /// Emits `CALL_R(malloc_big_fixedsize_fn, size, typeid)` followed
    /// by `CHECK_MEMORY_ERROR` (via `gen_call_malloc_gc`).
    ///
    /// rewrite.py stamps `remember_write_barrier` on the result
    /// there, sound because `malloc_big_fixedsize` hands back a *young*
    /// raw-malloced object: it calls `do_malloc_fixedsize_clear`, i.e. the
    /// ordinary `malloc_fixedsize`, whose large arm is
    /// `external_malloc(typeid, 0, alloc_young=True)`.  A young object needs no
    /// barrier for a young pointer stored into it.  Both backend helpers now
    /// call that entry point (`dynasm_malloc_big_fixedsize` /
    /// `gc_malloc_big_fixedsize_helper` → `alloc_nursery_typed`, which is
    /// `MiniMarkGC::alloc_with_type`), so the stamp is applied.
    ///
    /// "This is always true because it's a fixed-size object" is the load-bearing
    /// half of upstream's comment, and it is about cards, not about size.
    /// Fixed-size means `external_malloc(typeid, 0, ...)`: length zero, so no
    /// `GCFLAG_HAS_CARDS` and `extra_flags = 0`, so no
    /// `GCFLAG_TRACK_YOUNG_PTRS`.  A young rawmalloced object without that flag
    /// is appended to `old_objects_pointing_to_young` the moment
    /// `_visit_young_rawmalloced_object` promotes it, so its slots are traced
    /// without any barrier ever having run; unreached, it is freed and its slots
    /// never mattered.  `MiniMarkGC::finish_alloc_young_nonmoving` births the
    /// header the same way.  This is also why `gen_call_malloc_gc` must *not*
    /// stamp in general — "v_result might be a large young array", and a
    /// card-marked array is born carrying `GCFLAG_TRACK_YOUNG_PTRS`, which puts
    /// its writes back under the barrier.
    ///
    /// It is applied *conditionally*, because upstream's large arm has one
    /// outcome and pyre's has a rollback gate: under
    /// `MAJIT_GC_YOUNG_RAWMALLOC=0` the same helper births the object in the old
    /// generation, and an old result under an elided barrier loses the first
    /// young pointer stored into it.  The gate is process-wide and read once, so
    /// asking it here answers for every trace this process compiles.
    ///
    /// Two outcomes stay young-or-nothing and so do not need a condition.  A
    /// failed allocation returns NULL, which the `CHECK_MEMORY_ERROR` emitted by
    /// [`gen_call_malloc_gc`](Self::gen_call_malloc_gc) turns into a
    /// MemoryError before any store can reach the object.  A weakref type is
    /// the one kind `MiniMarkGC::type_alloc_may_be_young` still sends to the
    /// born-old birth, and it is excluded by the same contract upstream relies
    /// on — `malloc_fixedsize`'s large arm opens with
    /// `ll_assert(not contains_weakptr, "'contains_weakptr' specified for a
    /// large object")` — a weakref payload being one `weakptr` slot, never
    /// anything near the large-object threshold.
    ///
    /// pyre is framework-GC only; the Boehm `else` arm
    /// (`malloc_fixedsize_fn`) is intentionally not ported.  The
    /// helper itself stamps the tid into the GC header (matching
    /// upstream `malloc_big_fixedsize` at gc.py which goes
    /// through `do_malloc_fixedsize_clear` with type_id), so callers
    /// MUST NOT emit a separate `gen_initialize_tid`.
    fn gen_malloc_fixedsize(
        &self,
        size: usize,
        type_id: u32,
        non_moving: bool,
        result_pos: OpRef,
        st: &mut RewriteState<'_>,
    ) -> Operand {
        // rewrite.py `gen_malloc_fixedsize` Boehm arm: `CALL_R(malloc_fn_ptr, size)`.
        if let Some(addr) = boehm_malloc_fn() {
            let fn_ref = st.const_int(addr as i64);
            let size_ref = st.const_int(size as i64);
            let result = self.gen_call_malloc_gc(
                &[fn_ref, size_ref],
                result_pos,
                boehm_malloc_fixedsize_descr(),
                st,
            );
            st.remember_wb(&result);
            return result;
        }
        debug_assert_eq!(
            size & (std::mem::size_of::<usize>() - 1),
            0,
            "rewrite.py:785 `assert (size & (WORD-1)) == 0` — size must be word-aligned"
        );
        let fn_ref = st.const_int(if non_moving {
            self.malloc_big_fixedsize_oldgen_fn
        } else {
            self.malloc_big_fixedsize_fn
        });
        let size_ref = st.const_int(size as i64);
        let typeid_ref = st.const_int(type_id as i64);
        let result = self.gen_call_malloc_gc(
            &[fn_ref, size_ref, typeid_ref],
            result_pos,
            self.malloc_big_fixedsize_descr.clone(),
            st,
        );
        // rewrite.py: "mark 'v_result' as freshly malloced, so not needing a
        // write barrier (this is always true because it's a fixed-size
        // object)".
        if !non_moving && crate::collector::young_rawmalloc_enabled() {
            st.remember_wb(&result);
        }
        result
    }

    /// rewrite.py `gen_malloc_str`.
    ///
    /// Upstream's `malloc_str` helper closure
    /// captures `str_type_id = self.str_descr.tid` (gc.py:451) at
    /// generate-time.  Rust `extern "C" fn` can't lexically capture, so
    /// the type id is threaded as an explicit CALL arg sourced from
    /// `gc_ll_descr.str_descr.type_id()`.  Matches the
    /// `malloc_str_calldescr` Signed/Signed signature documented there.
    fn gen_malloc_str(
        &self,
        v_num_elem: Operand,
        result_pos: OpRef,
        st: &mut RewriteState<'_>,
    ) -> Operand {
        let fn_ref = st.const_int(self.malloc_str_fn);
        let type_id = self
            .str_descr
            .as_array_descr()
            .expect("gc_ll_descr.str_descr must be an ArrayDescr")
            .type_id();
        let type_id_ref = st.const_int(type_id as i64);
        self.gen_call_malloc_gc(
            &[fn_ref, type_id_ref, v_num_elem],
            result_pos,
            self.malloc_str_descr.clone(),
            st,
        )
    }

    /// rewrite.py `gen_malloc_unicode`.
    ///
    /// See `gen_malloc_str`.  Type id sourced
    /// from `gc_ll_descr.unicode_descr.type_id()` (gc.py:455
    /// `unicode_type_id = self.unicode_descr.tid`).
    fn gen_malloc_unicode(
        &self,
        v_num_elem: Operand,
        result_pos: OpRef,
        st: &mut RewriteState<'_>,
    ) -> Operand {
        let fn_ref = st.const_int(self.malloc_unicode_fn);
        let type_id = self
            .unicode_descr
            .as_array_descr()
            .expect("gc_ll_descr.unicode_descr must be an ArrayDescr")
            .type_id();
        let type_id_ref = st.const_int(type_id as i64);
        self.gen_call_malloc_gc(
            &[fn_ref, type_id_ref, v_num_elem],
            result_pos,
            self.malloc_unicode_descr.clone(),
            st,
        )
    }

    // ────────────────────────────────────────────────────────
    // COPYSTRCONTENT / COPYUNICODECONTENT → memcpy CALL_N
    // ────────────────────────────────────────────────────────

    /// rewrite.py `rewrite_copy_str_content`.
    ///
    /// Lowers `COPYSTRCONTENT(src, dst, src_start, dst_start, length)` (and
    /// the UNICODE variant) to:
    ///
    /// ```text
    /// i1 = LOAD_EFFECTIVE_ADDRESS(src_gcptr, src_start, basesize, shift)
    /// i2 = LOAD_EFFECTIVE_ADDRESS(dst_gcptr, dst_start, basesize, shift)
    /// CALL_N(memcpy_fn, i2, i1, count, descr=memcpy_descr)
    /// ```
    ///
    /// For UNICODE, `count` is `length << shift` (byte count); for STR the
    /// basesize is additionally offset by `-1` to skip the STR
    /// `extra_item_after_alloc` null terminator (rewrite.py:1051-1053;
    /// rstr.py:1226 `extra_item_after_alloc=True`).
    fn rewrite_copy_str_content(&self, op: &Op, st: &mut RewriteState<'_>) {
        // rewrite.py:1046-1048 — pull memcpy_fn / memcpy_descr off the
        // gc_ll_descr instance fields (gc.py:39-43).
        let memcpy_fn = self.memcpy_fn;
        let memcpy_descr = self.memcpy_descr.clone();

        // rewrite.py:1049-1064 — basesize/itemscale come from the
        // canonical str_descr / unicode_descr held on gc_ll_descr
        // (gc.py:46-47), NOT from the op itself; upstream's
        // COPY{STR,UNICODE}CONTENT carries no arraydescr.
        let (mut basesize, itemscale) = if op.opcode == OpCode::Copystrcontent {
            let ad = self
                .str_descr
                .as_array_descr()
                .expect("gc_ll_descr.str_descr must be an ArrayDescr");
            // rewrite.py `assert self.gc_ll_descr.str_descr.itemsize == 1`.
            assert_eq!(
                ad.item_size(),
                1,
                "rewrite.py:1054 str_descr.itemsize must be 1"
            );
            (ad.base_size() as i64, 0i64)
        } else {
            let ad = self
                .unicode_descr
                .as_array_descr()
                .expect("gc_ll_descr.unicode_descr must be an ArrayDescr");
            let itemsize = ad.item_size() as i64;
            // rewrite.py:1059-1063 — itemscale = log2(itemsize) for 2/4.
            let itemscale = match itemsize {
                2 => 1,
                4 => 2,
                _ => {
                    panic!("rewrite.py:1064 unknown unicode itemsize {itemsize} — expected 2 or 4")
                }
            };
            (ad.base_size() as i64, itemscale)
        };
        if op.opcode == OpCode::Copystrcontent {
            // rewrite.py:1051-1053 — one extra item after the string buffer
            // (rstr.py:1226 `extra_item_after_alloc=True`), so the `chars`
            // array starts at `str_descr.basesize - 1`.
            basesize -= 1;
        }

        // rewrite.py:1065-1068 — effective source / destination addresses.
        let src_gcptr = st.get_box_replacement(op.arg(0));
        let dst_gcptr = st.get_box_replacement(op.arg(1));
        let src_index = st.get_box_replacement(op.arg(2));
        let dst_index = st.get_box_replacement(op.arg(3));

        let i1 = self.emit_load_effective_address(src_gcptr, src_index, basesize, itemscale, st);
        let i2 = self.emit_load_effective_address(dst_gcptr, dst_index, basesize, itemscale, st);

        // rewrite.py:1069-1078 — byte count.
        //   STR:     arg = op.getarg(4)                         (itemscale=0)
        //   UNICODE: arg = ConstInt(op.getarg(4).getint() << itemscale)
        //            or INT_LSHIFT(op.getarg(4), ConstInt(itemscale))
        let arg = if op.opcode == OpCode::Copystrcontent {
            st.get_box_replacement(op.arg(4))
        } else {
            let v_length = st.get_box_replacement(op.arg(4));
            if let Some(c) = st.resolve_constant(&v_length) {
                // rewrite.py:1073-1074 — constant-fold the shift.
                st.const_int(c << itemscale)
            } else {
                // rewrite.py:1075-1078 — emit INT_LSHIFT.
                let shift_ref = st.const_int(itemscale);
                let lshift = mk_op(OpCode::IntLshift, &[v_length, shift_ref]);
                st.emit_result(lshift, OpRef::NONE)
            }
        };

        // rewrite.py:1079-1080 — CALL_N(memcpy_fn, i2, i1, arg, descr=memcpy_descr).
        let memcpy_fn_const = st.const_int(memcpy_fn);
        let call_op = mk_op(OpCode::CallN, &[memcpy_fn_const, i2, i1, arg]);
        call_op.setdescr(memcpy_descr);
        st.emit(call_op);
    }

    /// rewrite.py `emit_load_effective_address`.
    ///
    /// CPUs with `supports_load_effective_address = True` (x86, aarch64)
    /// emit a single LEA op; CPUs without it (model.py:22 base default)
    /// expand to `(INT_LSHIFT? + INT_ADD + INT_ADD)`.  The LEA arg order
    /// is `[v_gcptr, v_index, c_baseofs, c_shift]` per
    /// resoperation.py:1052-1054.
    fn emit_load_effective_address(
        &self,
        v_gcptr: Operand,
        v_index: Operand,
        base: i64,
        itemscale: i64,
        st: &mut RewriteState<'_>,
    ) -> Operand {
        if self.supports_load_effective_address {
            // rewrite.py:1083-1088 — single LEA op.
            let base_ref = st.const_int(base);
            let shift_ref = st.const_int(itemscale);
            let lea = mk_op(
                OpCode::LoadEffectiveAddress,
                &[v_gcptr, v_index, base_ref, shift_ref],
            );
            st.emit_result(lea, OpRef::NONE)
        } else {
            // rewrite.py:1089-1098 — fallback expansion.
            //   if itemscale > 0:
            //       v_index = INT_LSHIFT(v_index, ConstInt(itemscale))
            //   i1b = INT_ADD(v_gcptr, v_index)
            //   i1  = INT_ADD(i1b, ConstInt(base))
            let scaled = if itemscale > 0 {
                let shift_ref = st.const_int(itemscale);
                let lshift = mk_op(OpCode::IntLshift, &[v_index, shift_ref]);
                st.emit_result(lshift, OpRef::NONE)
            } else {
                v_index
            };
            let add1 = mk_op(OpCode::IntAdd, &[v_gcptr, scaled]);
            let i1b = st.emit_result(add1, OpRef::NONE);
            let base_ref = st.const_int(base);
            let add2 = mk_op(OpCode::IntAdd, &[i1b, base_ref]);
            st.emit_result(add2, OpRef::NONE)
        }
    }

    // ────────────────────────────────────────────────────────
    // SETFIELD_GC  → maybe COND_CALL_GC_WB + SETFIELD_GC
    // ────────────────────────────────────────────────────────

    /// rewrite.py `handle_write_barrier_setfield`.
    /// Emits a write barrier before the store when the stored value is a
    /// non-null reference into a pointer-bearing field AND the base has
    /// not already been WB'd.  Does *not* emit the store itself — the
    /// caller is expected to follow up with `emit_maybe_forwarded` so
    /// the lowered GC_STORE (forwarded by `transform_to_gc_load`) lands
    /// after the WB.
    fn handle_write_barrier_setfield(&self, op: &Op, st: &mut RewriteState<'_>) {
        let obj = st.get_box_replacement(op.arg(0));
        if st.wb_already_applied(&obj) {
            return;
        }
        // rewrite.py:930-931: check the stored VALUE's type.
        //   v = op.getarg(1)
        //   if (v.type == 'r' and (not isinstance(v, ConstPtr) or
        //       rgc.needs_write_barrier(v.value))):
        //
        // Gate on field descriptor: if the field is not a pointer field,
        // the GC won't trace it, so no WB is needed regardless of value
        // type.  In RPython val.type=='r' implies the field is GCREF;
        // here ForceToken (Ref) stores to an Int-typed field (offset 128),
        // a pyre-specific divergence.
        let field_is_ptr = op
            .getdescr()
            .and_then(|d| d.as_field_descr().map(|fd| fd.is_pointer_field()))
            .unwrap_or(false);
        let val = st.get_box_replacement(op.arg(1));
        let val_is_ref = if field_is_ptr {
            match st.result_type_of(&val) {
                Some(tp) => tp == Type::Ref,
                // `result_type_of` is None only for the OpRef::None /
                // virtual-object marker (history.py Box.type has no analog
                // for it). Stored into a pointer field it is a GCREF slot,
                // so it needs a barrier — assume Ref.
                None => true,
            }
        } else {
            false
        };
        if !val_is_ref || st.is_null_constant(&val) {
            return;
        }
        self.gen_write_barrier(obj, st);
    }

    /// rewrite.py `gen_write_barrier`.
    fn gen_write_barrier(&self, v_base: Operand, st: &mut RewriteState<'_>) {
        let wb_op = mk_op(OpCode::CondCallGcWb, std::slice::from_ref(&v_base));
        st.emit(wb_op);
        st.remember_wb(&v_base);
    }

    /// rewrite.py `consider_setfield_gc`.
    ///
    /// Drops the `(base, offset)` entry from `_delayed_zero_setfields`
    /// so the pending-zero flush at `emit_pending_zeros`
    /// (rewrite.py:761-766) does not re-zero a slot that this explicit
    /// SETFIELD_GC is about to overwrite.
    ///
    /// Under the Boehm/raw-calloc fallback (`malloc_zero_filled = true`),
    /// `clear_gc_fields` skips its insertion path, so this is effectively a
    /// no-op. With a real collector, production backends set the flag to
    /// false (in their `gc_rewriter`), activating the delayed-zero
    /// tracking.
    fn consider_setfield_gc(&self, op: &Op, st: &mut RewriteState<'_>) {
        let Some(descr) = op.getdescr() else { return };
        let Some(fd) = descr.as_field_descr() else {
            return;
        };
        let offset = fd.offset() as i64;
        let base = st.get_box_replacement(op.arg(0));
        if let Some((_, entries)) = st._delayed_zero_setfields.get_mut(&base.to_opref()) {
            entries.swap_remove(&offset);
        }
    }

    // ────────────────────────────────────────────────────────
    // SETARRAYITEM_GC  → maybe COND_CALL_GC_WB{_ARRAY} + SETARRAYITEM_GC
    // rewrite.py handle_write_barrier_setarrayitem
    // ────────────────────────────────────────────────────────

    /// rewrite.py consider_setarrayitem_gc: record the constant
    /// index so emit_pending_zeros can skip this slot.
    ///
    /// ```text
    /// if not isinstance(array_box, ConstPtr) and index_box.is_constant():
    ///     self.remember_setarrayitem_occurred(array_box, index_box.getint())
    /// ```
    fn consider_setarrayitem_gc(&self, op: &Op, st: &mut RewriteState<'_>) {
        let array_ref = st.get_box_replacement(op.arg(0));
        let index_ref = op.arg(1);
        if st.resolve_constant(&array_ref).is_some() {
            return;
        }
        let Some(idx_val) = st.resolve_constant(&index_ref) else {
            return;
        };
        st.record_setarrayitem_index(&array_ref, idx_val as usize);
    }

    /// rewrite.py: handle_write_barrier_setarrayitem.
    /// Emits CondCallGcWb / CondCallGcWbArray as needed; the SETARRAYITEM
    /// op itself is NOT emitted here — RPython forwards the op to
    /// GC_STORE_INDEXED inside transform_to_gc_load (rewrite.py)
    /// and then `self.emit_op(op)` follows the forwarding. We do the
    /// equivalent in the caller by invoking handle_setarrayitem after WB.
    fn handle_write_barrier_setarrayitem(&self, op: &Op, st: &mut RewriteState<'_>) {
        let val = st.get_box_replacement(op.arg(0));
        // rewrite.py:938-942
        if !st.wb_already_applied(&val) {
            let v = st.get_box_replacement(op.arg(2));
            let val_is_ref = match st.result_type_of(&v) {
                Some(tp) => tp == Type::Ref,
                // None only for the OpRef::None / virtual marker (no type
                // tag); route the barrier decision through the array
                // descriptor — a pointer array is a GCREF slot.
                None => op
                    .getdescr()
                    .and_then(|d| d.as_array_descr().map(|ad| ad.is_array_of_pointers()))
                    .unwrap_or(false),
            };
            if val_is_ref && !st.is_null_constant(&v) {
                self.gen_write_barrier_array(val, st.get_box_replacement(op.arg(1)), st);
            }
        }
    }

    /// rewrite.py handle_setarrayitem.
    /// Lowers SETARRAYITEM_GC / SETARRAYITEM_RAW into GC_STORE /
    /// GC_STORE_INDEXED via `emit_gc_store_or_indexed`, which forwards
    /// the original op to the lowered form (the emission happens later
    /// in the main loop via `emit_maybe_forwarded` for RAW and via the
    /// SETARRAYITEM_GC write-barrier arm for GC).
    fn handle_setarrayitem(&self, op: &OpRc, st: &mut RewriteState<'_>) {
        let descr = op.getdescr().expect("SETARRAYITEM needs ArrayDescr");
        let ad = descr
            .as_array_descr()
            .expect("SETARRAYITEM descr must be ArrayDescr");
        let itemsize = ad.item_size() as i64;
        let basesize = ad.base_size() as i64;
        let ptr = st.get_box_replacement(op.arg(0));
        let index = st.get_box_replacement(op.arg(1));
        let value = st.get_box_replacement(op.arg(2));
        self.emit_gc_store_or_indexed(
            Some(op),
            ptr,
            index,
            value,
            itemsize,
            itemsize,
            basesize,
            st,
        );
    }

    /// rewrite.py emit_gc_store_or_indexed (with cpu_simplify_scale
    /// inlined). `load_supported_factors` drives the non-constant branch:
    /// factors outside that set are pre-scaled in IR, factors inside it pass
    /// through to the backend's native addressing mode.
    ///
    /// When `original` is `Some`, the lowered GC_STORE / GC_STORE_INDEXED is
    /// *forwarded* as the replacement for the original op (upstream's
    /// `replace_op_with`); the main loop emits the forwarded op at the
    /// appropriate point in the output stream (after any write barrier).
    /// When `None`, the lowered op is emitted directly — used for
    /// internal stores synthesised by the rewriter that do not replace an
    /// input op (e.g. tid initialisation for fresh allocations).
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameters mirror rewrite.py's GC store descriptor, address calculation, value, and replacement channels; grouping them would obscure opcode parity"
    )]
    fn emit_gc_store_or_indexed(
        &self,
        original: Option<&OpRc>,
        ptr: Operand,
        index: Operand,
        value: Operand,
        itemsize: i64,
        factor: i64,
        offset: i64,
        st: &mut RewriteState<'_>,
    ) {
        // rewrite.py:142-143
        let (index, offset) = st._try_use_older_box(&index, factor, offset);

        // rewrite.py:144-146
        let (factor, offset, index_opt) =
            self._emit_mul_if_factor_offset_not_supported(index, factor, offset, st);

        // rewrite.py:148-154 — GC_STORE vs GC_STORE_INDEXED.
        let newload = match index_opt {
            None => {
                let offset_ref = st.const_int(offset);
                let itemsize_ref = st.const_int(itemsize);
                mk_op(OpCode::GcStore, &[ptr, offset_ref, value, itemsize_ref])
            }
            Some(idx) => {
                let factor_ref = st.const_int(factor);
                let offset_ref = st.const_int(offset);
                let itemsize_ref = st.const_int(itemsize);
                mk_op(
                    OpCode::GcStoreIndexed,
                    &[ptr, idx, value, factor_ref, offset_ref, itemsize_ref],
                )
            }
        };

        // rewrite.py:155-158
        if let Some(original) = original {
            st.set_forwarded(original, newload);
        } else {
            st.emit(newload);
        }
    }

    /// rewrite.py `_emit_mul_if_factor_offset_not_supported`.
    ///
    /// Wrapper around `cpu_simplify_scale` that emits the constructed
    /// pre-scale op (when one was needed) and folds the result into the
    /// `(factor, offset, index)` triple consumed by the indexed GC
    /// load/store builders.  `Option<OpRef>` mirrors upstream's
    /// `index_box is None` ConstInt sentinel — `None` means the index
    /// folded into `offset` and the caller should emit the non-indexed
    /// GC_LOAD / GC_STORE form.
    fn _emit_mul_if_factor_offset_not_supported(
        &self,
        index_box: Operand,
        factor: i64,
        offset: i64,
        st: &mut RewriteState<'_>,
    ) -> (i64, i64, Option<Operand>) {
        let (factor, offset, scaled) = self.cpu_simplify_scale(&index_box, factor, offset, st);
        let index_opt = match scaled {
            ScaledIndex::Const => None,
            ScaledIndex::Passthrough(idx) => Some(idx),
            // rewrite.py:169-170 — the pre-scale op is emitted here.
            ScaledIndex::PreScale(op) => Some(st.emit_result(op, OpRef::NONE)),
        };
        (factor, offset, index_opt)
    }

    /// rewrite.py `cpu_simplify_scale`.
    ///
    /// Pure decision function: given the raw `(index, factor, offset)`
    /// triple supplied by an indexed GC load/store builder, decide
    /// whether to (a) fold a ConstInt index into `offset`, (b) pass the
    /// triple through unchanged when the CPU's addressing mode supports
    /// `factor`, or (c) construct (but not emit) an `INT_LSHIFT` /
    /// `INT_MUL` pre-scale op.  The returned `ScaledIndex` carries the
    /// constructed pre-scale op so `_emit_mul_if_factor_offset_not_
    /// supported` can emit it; vector_ext.py callers (rpython/jit/
    /// backend/llsupport/vector_ext.py:127, :157) emit it themselves.
    fn cpu_simplify_scale(
        &self,
        index_box: &Operand,
        factor: i64,
        offset: i64,
        st: &mut RewriteState<'_>,
    ) -> (i64, i64, ScaledIndex) {
        // rewrite.py:1118-1122 — ConstInt path.
        if let Some(index_val) = st.resolve_constant(index_box) {
            return (1, index_val * factor + offset, ScaledIndex::Const);
        }
        // rewrite.py:1124-1133 — non-constant path.
        if factor != 1 && !self.load_supported_factors.contains(&factor) {
            assert!(factor > 0, "cpu_simplify_scale: factor must be positive");
            // rewrite.py:1127-1131 — power-of-two factor → INT_LSHIFT,
            // else INT_MUL.
            let mul_op = if (factor & (factor - 1)) == 0 {
                let shift = (factor as u64).trailing_zeros() as i64;
                let shift_ref = st.const_int(shift);
                mk_op(OpCode::IntLshift, &[index_box.clone(), shift_ref])
            } else {
                let factor_ref = st.const_int(factor);
                mk_op(OpCode::IntMul, &[index_box.clone(), factor_ref])
            };
            return (1, offset, ScaledIndex::PreScale(mul_op));
        }
        // rewrite.py:1132 — pass-through.
        (factor, offset, ScaledIndex::Passthrough(index_box.clone()))
    }

    /// rewrite.py `handle_getarrayitem`.
    /// Lowers GETARRAYITEM_{GC,RAW}_{I,R,F}, including the PURE variants,
    /// into GC_LOAD / GC_LOAD_INDEXED through `emit_gc_load_or_indexed`.
    fn handle_getarrayitem(&self, op: &OpRc, st: &mut RewriteState<'_>) {
        let descr = op.getdescr().expect("GETARRAYITEM needs ArrayDescr");
        let ad = descr
            .as_array_descr()
            .expect("GETARRAYITEM descr must be ArrayDescr");
        let itemsize = ad.item_size() as i64;
        let ofs = ad.base_size() as i64;
        let sign = ad.is_item_signed();
        let ptr = st.get_box_replacement(op.arg(0));
        let index = st.get_box_replacement(op.arg(1));
        self.emit_gc_load_or_indexed(op, ptr, index, itemsize, itemsize, ofs, sign, st);
    }

    /// rewrite.py emit_gc_load_or_indexed (with cpu_simplify_scale
    /// inlined). Forwards `original` to either GC_LOAD_{I,R,F} (when the
    /// index resolves to a constant) or GC_LOAD_INDEXED_{I,R,F}.
    ///
    /// The caller is expected to supply the already-resolved
    /// `ptr` / `index` args and the raw (itemsize, factor, offset, sign)
    /// tuple from `unpack_arraydescr` / `unpack_fielddescr` /
    /// `unpack_interiorfielddescr` or from `get_array_token` /
    /// `get_field_token` for the string and unicode helpers.
    ///
    /// `sign` is encoded into the emitted `itemsize` arg by negating it
    /// (rewrite.py:192-194) — the backend decodes the sign back out of
    /// the sign bit on the nsize operand.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameters mirror rewrite.py's GC load descriptor, address calculation, sign, and replacement channels; grouping them would obscure opcode parity"
    )]
    fn emit_gc_load_or_indexed(
        &self,
        original: &OpRc,
        ptr: Operand,
        index: Operand,
        itemsize: i64,
        factor: i64,
        offset: i64,
        sign: bool,
        st: &mut RewriteState<'_>,
    ) {
        // rewrite.py:186-187
        let (index, offset) = st._try_use_older_box(&index, factor, offset);

        // rewrite.py:188-190
        let (factor, offset, index_opt) =
            self._emit_mul_if_factor_offset_not_supported(index, factor, offset, st);

        // rewrite.py:192-194 — encode signed-ness into the itemsize value.
        let itemsize_enc = if sign { -itemsize } else { itemsize };

        // rewrite.py:196-198 — optype from op.type (result-kind of the
        // original load op determines the GC_LOAD_I / R / F variant).
        let optype = original.opcode.result_type();

        // rewrite.py:199-205 — GC_LOAD vs GC_LOAD_INDEXED.
        let newload = match index_opt {
            None => {
                let offset_ref = st.const_int(offset);
                let itemsize_ref = st.const_int(itemsize_enc);
                mk_op(get_gc_load(optype), &[ptr, offset_ref, itemsize_ref])
            }
            Some(idx) => {
                let factor_ref = st.const_int(factor);
                let offset_ref = st.const_int(offset);
                let itemsize_ref = st.const_int(itemsize_enc);
                mk_op(
                    get_gc_load_indexed(optype),
                    &[ptr, idx, factor_ref, offset_ref, itemsize_ref],
                )
            }
        };

        // rewrite.py:206-209 — pyre callers always pass `op`, so we
        // always replace_op_with (set_forwarded).
        st.set_forwarded(original, newload);
    }

    /// rewrite.py `emit_setfield`.
    ///
    /// Synthetic field store helper: emits `GC_STORE(ptr, ConstInt(0),
    /// value, ConstInt(size))` via `emit_gc_store_or_indexed` (which
    /// folds the ConstInt(0)*1 + ofs into a non-indexed GC_STORE per
    /// `cpu_simplify_scale`'s ConstInt branch).  `op=None` so the
    /// lowered op is emitted directly rather than forwarded.
    fn emit_setfield(
        &self,
        ptr: Operand,
        value: Operand,
        fd: &dyn FieldDescr,
        st: &mut RewriteState<'_>,
    ) {
        self.emit_setfield_raw(ptr, value, fd.offset() as i64, fd.field_size() as i64, st);
    }

    /// Raw-offset variant of `emit_setfield` (Rust adaptation).
    ///
    /// pyre's `JitFrameDescrs` carries raw `i32` offsets and
    /// `sign_size: usize` rather than upstream's per-field `FieldDescr`
    /// objects (see rewrite.py `emit_setfield(frame, c_null,
    /// descr=descrs.jf_*)`).  `handle_call_assembler` consumes those
    /// raw offsets directly; this helper deduplicates the
    /// `Op::new(GcStore, ...) + st.emit` boilerplate so the lowering
    /// still passes through `emit_gc_store_or_indexed` and
    /// `cpu_simplify_scale` like every other field store.  Convergence
    /// path: replace `JitFrameDescrs::jf_*_ofs` with `SimpleFieldDescr`
    /// instances, then have these callers use `emit_setfield(fd)`
    /// directly.
    fn emit_setfield_raw(
        &self,
        ptr: Operand,
        value: Operand,
        ofs: i64,
        size: i64,
        st: &mut RewriteState<'_>,
    ) {
        // rewrite.py `consider_setfield_gc`: a store at this offset
        // cancels the delayed NULL `clear_gc_fields` recorded.
        let base = st.get_box_replacement(ptr.clone());
        if let Some((_, entries)) = st._delayed_zero_setfields.get_mut(&base.to_opref()) {
            entries.swap_remove(&ofs);
        }
        let zero = st.const_int(0);
        self.emit_gc_store_or_indexed(None, ptr, zero, value, size, 1, ofs, st);
    }

    /// rewrite.py `transform_to_gc_load`.
    ///
    /// Central dispatcher that lowers high-level memory accessors to
    /// GC_LOAD / GC_LOAD_INDEXED / GC_STORE / GC_STORE_INDEXED. Each arm
    /// matches its upstream counterpart line-by-line; the emission uses
    /// `emit_gc_load_or_indexed` / `emit_gc_store_or_indexed`, which
    /// either forward the op (when `original` is `Some`) or emit directly.
    ///
    /// Returns `true` only for the `GETFIELD_GC_*` fast-path at
    /// rewrite.py:259-260, which flushes pending zeros, forwards, and
    /// emits the forwarded op itself — the caller (`rewrite`) then
    /// skips the rest of the main-loop body.  All other arms forward
    /// the op and return `false`, delegating emission to the main loop
    /// via `emit_maybe_forwarded` (or the write-barrier arms).
    fn transform_to_gc_load(&self, op: &OpRc, st: &mut RewriteState<'_>) -> bool {
        const NOT_SIGNED: bool = false;
        let opnum = op.opcode;

        // rewrite.py:216-218 `rop.is_getarrayitem(opnum) or opnum in
        // (GETARRAYITEM_RAW_I, GETARRAYITEM_RAW_F)`.  Upstream omits
        // GETARRAYITEM_RAW_R because codewriter rejects raw ref array
        // reads at `rpython/jit/codewriter/jtransform.py:775`
        // (`getarrayitem_raw_r not supported`); mirror that omission so
        // a rogue GETARRAYITEM_RAW_R does not get silently lowered into
        // GC_LOAD_INDEXED_R here.
        if opnum.is_getarrayitem()
            || matches!(opnum, OpCode::GetarrayitemRawI | OpCode::GetarrayitemRawF)
        {
            self.handle_getarrayitem(op, st);
            return false;
        }
        // rewrite.py:220-221
        if matches!(opnum, OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw) {
            self.handle_setarrayitem(op, st);
            return false;
        }
        // rewrite.py:222-227 RAW_STORE
        if matches!(opnum, OpCode::RawStore) {
            let descr = op.getdescr().expect("RAW_STORE needs ArrayDescr");
            let ad = descr
                .as_array_descr()
                .expect("RAW_STORE descr must be ArrayDescr");
            let itemsize = ad.item_size() as i64;
            let ofs = ad.base_size() as i64;
            let ptr = st.get_box_replacement(op.arg(0));
            let index = st.get_box_replacement(op.arg(1));
            let value = st.get_box_replacement(op.arg(2));
            self.emit_gc_store_or_indexed(Some(op), ptr, index, value, itemsize, 1, ofs, st);
            return false;
        }
        // rewrite.py:228-232 RAW_LOAD_{I,F}
        if matches!(opnum, OpCode::RawLoadI | OpCode::RawLoadF) {
            let descr = op.getdescr().expect("RAW_LOAD needs ArrayDescr");
            let ad = descr
                .as_array_descr()
                .expect("RAW_LOAD descr must be ArrayDescr");
            let itemsize = ad.item_size() as i64;
            let ofs = ad.base_size() as i64;
            let sign = ad.is_item_signed();
            let ptr = st.get_box_replacement(op.arg(0));
            let index = st.get_box_replacement(op.arg(1));
            self.emit_gc_load_or_indexed(op, ptr, index, itemsize, 1, ofs, sign, st);
            return false;
        }
        // rewrite.py:233-238 GETINTERIORFIELD_GC_{I,R,F}
        if matches!(
            opnum,
            OpCode::GetinteriorfieldGcI | OpCode::GetinteriorfieldGcR | OpCode::GetinteriorfieldGcF
        ) {
            let descr = op
                .getdescr()
                .expect("GETINTERIORFIELD needs InteriorFieldDescr");
            let ifd = descr
                .as_interior_field_descr()
                .expect("GETINTERIORFIELD descr must be InteriorFieldDescr");
            let ad = ifd.array_descr();
            let fd = ifd.field_descr();
            let ofs = (ad.base_size() + fd.offset()) as i64;
            let itemsize = ad.item_size() as i64;
            let fieldsize = fd.field_size() as i64;
            let sign = fd.is_field_signed();
            let ptr = st.get_box_replacement(op.arg(0));
            let index = st.get_box_replacement(op.arg(1));
            self.emit_gc_load_or_indexed(op, ptr, index, fieldsize, itemsize, ofs, sign, st);
            return false;
        }
        // rewrite.py:239-245 SETINTERIORFIELD_{RAW,GC}
        if matches!(
            opnum,
            OpCode::SetinteriorfieldRaw | OpCode::SetinteriorfieldGc
        ) {
            let descr = op
                .getdescr()
                .expect("SETINTERIORFIELD needs InteriorFieldDescr");
            let ifd = descr
                .as_interior_field_descr()
                .expect("SETINTERIORFIELD descr must be InteriorFieldDescr");
            let ad = ifd.array_descr();
            let fd = ifd.field_descr();
            let ofs = (ad.base_size() + fd.offset()) as i64;
            let itemsize = ad.item_size() as i64;
            let fieldsize = fd.field_size() as i64;
            let ptr = st.get_box_replacement(op.arg(0));
            let index = st.get_box_replacement(op.arg(1));
            let value = st.get_box_replacement(op.arg(2));
            self.emit_gc_store_or_indexed(
                Some(op),
                ptr,
                index,
                value,
                fieldsize,
                itemsize,
                ofs,
                st,
            );
            return false;
        }
        // rewrite.py:246-247 GETFIELD_{GC,RAW}_{I,R,F}.
        if matches!(
            opnum,
            OpCode::GetfieldGcI
                | OpCode::GetfieldGcR
                | OpCode::GetfieldGcF
                | OpCode::GetfieldRawI
                | OpCode::GetfieldRawR
                | OpCode::GetfieldRawF
        ) {
            let descr = op.getdescr().expect("GETFIELD needs FieldDescr");
            let fd = descr
                .as_field_descr()
                .expect("GETFIELD descr must be FieldDescr");
            let ofs = fd.offset() as i64;
            let itemsize = fd.field_size() as i64;
            let sign = fd.is_field_signed();
            let ptr = st.get_box_replacement(op.arg(0));
            let cint_zero = st.const_int(0);
            let is_gc = matches!(
                opnum,
                OpCode::GetfieldGcI | OpCode::GetfieldGcR | OpCode::GetfieldGcF,
            );
            if is_gc {
                // rewrite.py:250-260 — flush pending zeros, forward, and
                // emit the forwarded op *here* so that the main loop
                // short-circuits (return True).
                st.emit_pending_zeros();
                self.emit_gc_load_or_indexed(op, ptr, cint_zero, itemsize, 1, ofs, sign, st);
                if let Forwarded::Op(lowered) = op.get_forwarded() {
                    st.emit_forwarded(op, lowered);
                }
                return true;
            }
            self.emit_gc_load_or_indexed(op, ptr, cint_zero, itemsize, 1, ofs, sign, st);
            return false;
        }
        // rewrite.py:262-266 SETFIELD_{GC,RAW}
        if matches!(opnum, OpCode::SetfieldGc | OpCode::SetfieldRaw) {
            let descr = op.getdescr().expect("SETFIELD needs FieldDescr");
            let fd = descr
                .as_field_descr()
                .expect("SETFIELD descr must be FieldDescr");
            let ofs = fd.offset() as i64;
            let itemsize = fd.field_size() as i64;
            let ptr = st.get_box_replacement(op.arg(0));
            let value = st.get_box_replacement(op.arg(1));
            let cint_zero = st.const_int(0);
            self.emit_gc_store_or_indexed(Some(op), ptr, cint_zero, value, itemsize, 1, ofs, st);
            return false;
        }
        // rewrite.py:267-272 ARRAYLEN_GC
        if matches!(opnum, OpCode::ArraylenGc) {
            let descr = op.getdescr().expect("ARRAYLEN_GC needs ArrayDescr");
            let ad = descr
                .as_array_descr()
                .expect("ARRAYLEN_GC descr must be ArrayDescr");
            let ofs = ad
                .len_descr()
                .expect("ARRAYLEN_GC descr must have lendescr")
                .offset() as i64;
            // rewrite.py:272 WORD itemsize, unsigned.
            let word = std::mem::size_of::<usize>() as i64;
            let ptr = st.get_box_replacement(op.arg(0));
            let cint_zero = st.const_int(0);
            self.emit_gc_load_or_indexed(op, ptr, cint_zero, word, 1, ofs, NOT_SIGNED, st);
            return false;
        }
        // rewrite.py:273-282 STRLEN / UNICODELEN — load length field
        // via `get_array_token(...).ofs_length`, which lives on the
        // ArrayDescr as `lendescr.offset`.  Upstream reads a WORD,
        // unsigned.
        if matches!(opnum, OpCode::Strlen | OpCode::Unicodelen) {
            let word = std::mem::size_of::<usize>() as i64;
            let descr = op
                .getdescr()
                .expect("STRLEN/UNICODELEN op must carry an ArrayDescr");
            let ad = descr
                .as_array_descr()
                .expect("STRLEN/UNICODELEN descr must be an ArrayDescr");
            let ld = ad
                .len_descr()
                .expect("STR/UNICODE ArrayDescr must carry lendescr");
            let ofs = ld.offset() as i64;
            let ptr = st.get_box_replacement(op.arg(0));
            let cint_zero = st.const_int(0);
            self.emit_gc_load_or_indexed(op, ptr, cint_zero, word, 1, ofs, NOT_SIGNED, st);
            return false;
        }
        // rewrite.py:283-294 STRHASH / UNICODEHASH — `get_field_token(
        // rstr.STR/UNICODE, 'hash', ...)` with `sign=True` and
        // `assert size == WORD`.  The upstream call returns
        // (offset, size); pyre injects a FieldDescr that carries both.
        if matches!(opnum, OpCode::Strhash | OpCode::Unicodehash) {
            let word = std::mem::size_of::<usize>() as i64;
            let descr = op
                .getdescr()
                .expect("STRHASH/UNICODEHASH op must carry a FieldDescr");
            let fd = descr
                .as_field_descr()
                .expect("STRHASH/UNICODEHASH descr must be a FieldDescr");
            assert_eq!(fd.field_size() as i64, word, "rewrite.py:286/292 assert");
            let ofs = fd.offset() as i64;
            let ptr = st.get_box_replacement(op.arg(0));
            let cint_zero = st.const_int(0);
            self.emit_gc_load_or_indexed(op, ptr, cint_zero, word, 1, ofs, true, st);
            return false;
        }
        // rewrite.py:295-301 STRGETITEM — `basesize -= 1` skips the
        // `extra_item_after_alloc` null terminator carried by
        // `rstr.STR.chars` (`rstr.py`).  `itemsize == 1` is
        // asserted upstream at rewrite.py:298.
        if matches!(opnum, OpCode::Strgetitem) {
            let (itemsize, basesize) = strgetsetitem_token(op, /*is_str=*/ true);
            let ptr = st.get_box_replacement(op.arg(0));
            let index = st.get_box_replacement(op.arg(1));
            self.emit_gc_load_or_indexed(
                op, ptr, index, itemsize, itemsize, basesize, NOT_SIGNED, st,
            );
            return false;
        }
        // rewrite.py:302-306 UNICODEGETITEM — UNICODE has no
        // extra_item_after_alloc, so basesize is used as-is.
        if matches!(opnum, OpCode::Unicodegetitem) {
            let (itemsize, basesize) = strgetsetitem_token(op, /*is_str=*/ false);
            let ptr = st.get_box_replacement(op.arg(0));
            let index = st.get_box_replacement(op.arg(1));
            self.emit_gc_load_or_indexed(
                op, ptr, index, itemsize, itemsize, basesize, NOT_SIGNED, st,
            );
            return false;
        }
        // rewrite.py:307-313 STRSETITEM.
        if matches!(opnum, OpCode::Strsetitem) {
            let (itemsize, basesize) = strgetsetitem_token(op, /*is_str=*/ true);
            let ptr = st.get_box_replacement(op.arg(0));
            let index = st.get_box_replacement(op.arg(1));
            let value = st.get_box_replacement(op.arg(2));
            self.emit_gc_store_or_indexed(
                Some(op),
                ptr,
                index,
                value,
                itemsize,
                itemsize,
                basesize,
                st,
            );
            return false;
        }
        // rewrite.py:314-318 UNICODESETITEM.
        if matches!(opnum, OpCode::Unicodesetitem) {
            let (itemsize, basesize) = strgetsetitem_token(op, /*is_str=*/ false);
            let ptr = st.get_box_replacement(op.arg(0));
            let index = st.get_box_replacement(op.arg(1));
            let value = st.get_box_replacement(op.arg(2));
            self.emit_gc_store_or_indexed(
                Some(op),
                ptr,
                index,
                value,
                itemsize,
                itemsize,
                basesize,
                st,
            );
            return false;
        }
        // rewrite.py:319-330 GC_LOAD_INDEXED_{I,R,F} normalisation.
        if matches!(
            opnum,
            OpCode::GcLoadIndexedI | OpCode::GcLoadIndexedR | OpCode::GcLoadIndexedF
        ) {
            let scale = st
                .resolve_constant(&op.arg(2))
                .expect("GC_LOAD_INDEXED scale must be ConstInt");
            let offset = st
                .resolve_constant(&op.arg(3))
                .expect("GC_LOAD_INDEXED offset must be ConstInt");
            let size = st
                .resolve_constant(&op.arg(4))
                .expect("GC_LOAD_INDEXED size must be ConstInt");
            let ptr = st.get_box_replacement(op.arg(0));
            let index = st.get_box_replacement(op.arg(1));
            self.emit_gc_load_or_indexed(op, ptr, index, size.abs(), scale, offset, size < 0, st);
            return false;
        }
        if matches!(opnum, OpCode::GcStoreIndexed) {
            let scale = st
                .resolve_constant(&op.arg(3))
                .expect("GC_STORE_INDEXED scale must be ConstInt");
            let offset = st
                .resolve_constant(&op.arg(4))
                .expect("GC_STORE_INDEXED offset must be ConstInt");
            let size = st
                .resolve_constant(&op.arg(5))
                .expect("GC_STORE_INDEXED size must be ConstInt");
            let ptr = st.get_box_replacement(op.arg(0));
            let index = st.get_box_replacement(op.arg(1));
            let value = st.get_box_replacement(op.arg(2));
            // rewrite.py:338: use abs(size) for safety even though store
            // size is expected to be positive.
            self.emit_gc_store_or_indexed(
                Some(op),
                ptr,
                index,
                value,
                size.abs(),
                scale,
                offset,
                st,
            );
            return false;
        }
        // rewrite.py:342
        false
    }

    // ────────────────────────────────────────────────────────
    // rewrite.py gen_write_barrier_array
    // ────────────────────────────────────────────────────────

    fn gen_write_barrier_array(
        &self,
        v_base: Operand,
        v_index: Operand,
        st: &mut RewriteState<'_>,
    ) {
        // rewrite.py:956-957 reads `self.gc_ll_descr.write_barrier_descr`
        // unguarded: this is only reached from the barrier section, which
        // `rewrite.py:393` already gated on the descriptor being set.
        let wb_descr = self
            .wb_descr
            .as_ref()
            .expect("gen_write_barrier_array reached with no write barrier descriptor");
        if wb_descr.has_write_barrier_from_array() {
            // rewrite.py `gen_write_barrier_array`. If we know
            // statically the length of 'v_base', and it is not too big,
            // then produce a regular write_barrier. If it's unknown or
            // too big, produce a write_barrier_from_array.
            const LARGE: usize = 130;
            let length = st.known_length(&v_base, LARGE);
            if length >= LARGE {
                // Unknown or too big: produce COND_CALL_GC_WB_ARRAY.
                let wb_op = mk_op(OpCode::CondCallGcWbArray, &[v_base, v_index]);
                st.emit(wb_op);
                // rewrite.py:970: a WB_ARRAY is not enough to prevent
                // any future write barriers, so don't remember_wb!
                return;
            }
        }
        // Fall-back: produce a regular write_barrier.
        let wb_op = mk_op(OpCode::CondCallGcWb, std::slice::from_ref(&v_base));
        st.emit(wb_op);
        st.remember_wb(&v_base);
    }

    // ────────────────────────────────────────────────────────
    // gen_malloc_nursery: batched bump-pointer allocation
    // ────────────────────────────────────────────────────────

    /// rewrite.py `gen_malloc_nursery` parity.
    ///
    /// Try to emit (or extend) a CALL_MALLOC_NURSERY for `size` bytes.
    /// Returns `Some(result)` on success; you still need to write the
    /// tid (rewrite.py:881-882).  Returns `None` when the requested
    /// size exceeds `can_use_nursery_malloc` — upstream's caller then
    /// falls back to `gen_malloc_fixedsize` /
    /// `gen_malloc_array` / `gen_malloc_str` / `gen_malloc_unicode`,
    /// whose `_gen_call_malloc_gc` helper does NOT mark the result as
    /// wb_applied (rewrite.py:775-776) because the slow malloc path
    /// may return an oldgen object.
    fn gen_malloc_nursery(
        &self,
        size: usize,
        result_pos: OpRef,
        st: &mut RewriteState<'_>,
    ) -> Option<Operand> {
        let size = round_up(size);

        // rewrite.py:884-886 — caller picks a slow path when nursery
        // can't accommodate the size.
        if !self.can_use_nursery(size) {
            return None;
        }

        // rewrite.py:893-898 merge with previous CALL_MALLOC_NURSERY
        if let Some(prev_idx) = st.pending_malloc_idx {
            let new_total = st.pending_malloc_total + size;
            if self.can_use_nursery(new_total) {
                let new_total_ref = st.const_int(new_total as i64);
                st.out[prev_idx].setarg(0, new_total_ref);
                st.pending_malloc_total = new_total;

                // rewrite.py:896: NURSERY_PTR_INCREMENT(last, ConstInt(previous_size))
                let prev_size_ref = st.const_int(st.previous_size as i64);
                let incr_op = mk_op(
                    OpCode::NurseryPtrIncrement,
                    &[st.last_malloced_ref.clone(), prev_size_ref],
                );
                let r = st.emit_result(incr_op, result_pos);
                st.previous_size = size;
                st.last_malloced_ref = r.clone();
                st.remember_wb(&r);
                return Some(r);
            }
        }

        // rewrite.py:903: CALL_MALLOC_NURSERY(ConstInt(size)). A
        // headerless descr emits the raw-bump form of the same op.
        st.emitting_an_operation_that_can_collect();
        let size_ref = st.const_int(size as i64);
        let malloc_opcode = if self.headerless_fixedsize {
            OpCode::CallMallocNurseryHeaderless
        } else {
            OpCode::CallMallocNursery
        };
        let op = mk_op(malloc_opcode, &[size_ref]);
        let r = st.emit_result(op, result_pos);
        st.pending_malloc_idx = Some(st.out.len() - 1);
        st.pending_malloc_total = size;
        st.previous_size = size;
        st.last_malloced_ref = r.clone();
        st.remember_wb(&r);
        Some(r)
    }

    // ────────────────────────────────────────────────────────
    // Helpers for header initialisation
    // ────────────────────────────────────────────────────────

    /// rewrite.py gen_initialize_tid parity.
    ///
    /// RPython:
    /// ```python
    /// def gen_initialize_tid(self, v_newgcobj, tid):
    ///     if self.gc_ll_descr.fielddescr_tid is not None:
    ///         self.emit_setfield(v_newgcobj, ConstInt(tid),
    ///                            descr=self.gc_ll_descr.fielddescr_tid)
    /// ```
    /// `emit_setfield` lowers to `GC_STORE(ptr, ConstInt(offset),
    /// ConstInt(tid), ConstInt(size))` via `emit_gc_store_or_indexed`.
    ///
    /// pyre layout note: HDR sits at `obj_ptr - HDR_SIZE` (vs RPython's
    /// HDR at `obj_ptr + 0`).  `fielddescr_tid.offset` is the offset of
    /// `tid` within the header struct (0 for pyre's single-word HDR);
    /// the actual store address is `obj_ptr + (-HDR_SIZE +
    /// descr.offset())`.  None disables the store (Boehm parity).
    ///
    /// `fielddescr_tid.field_size` is `WORD` (descr.rs
    /// `make_tid_field_descr`), the whole Signed `HDR.tid`: the store
    /// writes the type id and clears the flags half, so neither the
    /// backend's bump nor a NURSERY_PTR_INCREMENT interior object needs a
    /// separate header clear. That is sound for the same reason it is
    /// upstream: `gc.py malloc_nursery_slowpath` answers a size that
    /// passed `can_use_nursery` with a young object whose flags are clear.
    fn gen_initialize_tid(&self, obj: Operand, tid: u32, st: &mut RewriteState<'_>) {
        let Some(tid_fd_ref) = self.fielddescr_tid.as_ref() else {
            return;
        };
        let tid_fd = tid_fd_ref
            .as_field_descr()
            .expect("gc_ll_descr.fielddescr_tid must be a FieldDescr");
        let ofs = st.const_int(-(crate::header::GcHeader::SIZE as i64) + tid_fd.offset() as i64);
        let tid_val = st.const_int(tid as i64);
        let size = st.const_int(tid_fd.field_size() as i64);
        let store = mk_op(OpCode::GcStore, &[obj, ofs, tid_val, size]);
        st.emit(store);
    }

    /// [`Self::gen_initialize_tid`] storing only the type-id half of
    /// `HDR.tid`, so flags the slow-path allocator set survive.
    fn gen_initialize_tid_keep_flags(&self, obj: Operand, tid: u32, st: &mut RewriteState<'_>) {
        let Some(tid_fd_ref) = self.fielddescr_tid.as_ref() else {
            return;
        };
        let tid_fd = tid_fd_ref
            .as_field_descr()
            .expect("gc_ll_descr.fielddescr_tid must be a FieldDescr");
        let ofs = st.const_int(-(crate::header::GcHeader::SIZE as i64) + tid_fd.offset() as i64);
        let tid_val = st.const_int(tid as i64);
        let size = st.const_int((crate::header::TYPE_ID_BITS / 8) as i64);
        let store = mk_op(OpCode::GcStore, &[obj, ofs, tid_val, size]);
        st.emit(store);
    }

    /// rewrite.py:479-484 gen_initialize_vtable parity.
    ///
    /// RPython: emit_setfield(obj, ConstInt(vtable), descr=fielddescr_vtable)
    /// — the typeptr field of `rclass.OBJECT`. Offset / size come from
    /// the supplied `fielddescr_vtable` (gc.py:36 `get_field_descr(self,
    /// rclass.OBJECT, 'typeptr')`); for the canonical layout the
    /// vtable pointer sits at offset 0 with `Signed` size.
    fn gen_initialize_vtable(
        &self,
        obj: Operand,
        vtable: usize,
        vtable_fd_ref: &DescrRef,
        st: &mut RewriteState<'_>,
    ) {
        let vtable_fd = vtable_fd_ref
            .as_field_descr()
            .expect("gc_ll_descr.fielddescr_vtable must be a FieldDescr");
        let vtable_ref = st.const_int(vtable as i64);
        self.emit_setfield(obj, vtable_ref, vtable_fd, st);
    }

    fn gen_initialize_w_class(
        &self,
        obj: Operand,
        w_class: i64,
        w_class_fd: &dyn FieldDescr,
        st: &mut RewriteState<'_>,
    ) {
        let w_class_ref = Operand::const_from_value(Value::Ref(GcRef(w_class as usize)));
        self.emit_setfield(obj, w_class_ref, w_class_fd, st);
    }

    /// rewrite.py gen_initialize_len parity.
    ///
    /// RPython: `emit_setfield(v_newgcobj, v_length, descr=arraylen_descr)`.
    /// Routes through the local `emit_setfield` helper so the lowered
    /// store passes through `cpu_simplify_scale` (rewrite.py)
    /// for the ConstInt(0) index fold, matching upstream's emission
    /// path.
    fn gen_initialize_len(
        &self,
        obj: Operand,
        length: Operand,
        len_descr: &dyn FieldDescr,
        st: &mut RewriteState<'_>,
    ) {
        self.emit_setfield(obj, length, len_descr, st);
    }

    /// rewrite.py handle_call_assembler:
    ///   1. gen_malloc_frame — allocate callee jitframe from nursery
    ///   2. gen_initialize_tid + zero GC fields
    ///   3. store each arg at _ll_initial_locs[i] offset
    ///   4. replace multi-arg CALL_ASSEMBLER with single-arg [frame]
    ///
    /// Dispatched from the `CallAssembler{I,R,F,N}` arm of `rewrite_loop`
    /// (rewrite.py:413-416 parity).
    fn handle_call_assembler(&self, op: &OpRc, st: &mut RewriteState<'_>) {
        let descrs = self.jitframe_info.as_ref().unwrap();
        let lookup = self.call_assembler_callee_locs.as_ref().unwrap();

        // rewrite.py:667-668 — loop_token = op.getdescr(); JitCellToken
        let descr_arc = op
            .getdescr()
            .expect("CallAssembler op must carry a loop-token descriptor");
        let loop_token_descr = descr_arc
            .as_loop_token_descr()
            .expect("CallAssembler op must carry a loop-token descriptor");

        // rewrite.py:673 — index_list = loop_token.compiled_loop_token._ll_initial_locs
        // RPython: compiled_loop_token is pre-allocated with the token;
        // frame_info pointer is stable. Recursive calls emit against either a
        // compiled token or a `compile_tmp_callback` token, so metadata must
        // already exist before the rewriter sees CALL_ASSEMBLER.
        let callee_locs = lookup(loop_token_descr)
            .expect("CALL_ASSEMBLER target metadata must be registered before rewriter runs");
        if crate::majit_log_enabled() {
            eprintln!(
                "[gc-rewrite][call-assembler] token={} frame_info_ptr=0x{:x} ll_initial_locs={:?} frame_depth={} index_of_virtualizable={}",
                loop_token_descr.loop_token_number(),
                callee_locs.frame_info_ptr,
                callee_locs._ll_initial_locs,
                callee_locs.frame_depth,
                callee_locs.index_of_virtualizable,
            );
        }

        // rewrite.py — gen_malloc_frame(llfi)
        // RPython reads jfi_frame_size from frame_info AT RUNTIME so
        // the allocation size is correct even for self-recursive calls
        // (where frame_info is pre-allocated with [0,0] and updated
        // after compilation).
        let llfi = st.const_int(callee_locs.frame_info_ptr as i64);
        // jitframe.py — JITFRAMEINFO.jfi_frame_depth and
        // jfi_frame_size are both lltype.Signed, so the unpack_fielddescr
        // size read by emit_getfield is sign_size (the Signed word width).
        let signed_size = st.const_int(descrs.sign_size as i64);

        // rewrite.py:628-632 — GC_LOAD_I(frame_info, jfi_frame_size_ofs,
        // sign_size) where (ofs, sign_size, sign) = unpack_fielddescr(
        // descrs.jfi_frame_size).
        let jfi_frame_size_ofs = st.const_int(std::mem::size_of::<isize>() as i64);
        let size = st.emit(mk_op(
            OpCode::GcLoadI,
            &[llfi.clone(), jfi_frame_size_ofs, signed_size.clone()],
        ));
        // rewrite.py — gen_malloc_nursery_varsize_frame(size)
        st.emitting_an_operation_that_can_collect();
        let malloc_op = mk_op(OpCode::CallMallocNurseryVarsizeFrame, &[size]);
        let frame = st.emit_result(malloc_op, OpRef::NONE);
        // A frame past the large-object threshold is born young and
        // non-moving, as `gen_malloc_fixedsize` states; under
        // `MAJIT_GC_YOUNG_RAWMALLOC=0` it is born old and keeps its barriers.
        if crate::collector::young_rawmalloc_enabled() {
            st.remember_wb(&frame);
        }

        // rewrite.py — gen_initialize_tid(frame, descrs.arraydescr.tid),
        // narrowed to the type-id half of `HDR.tid`. A frame of
        // `large_object` bytes or more comes back from the slow path as a
        // young raw-malloced object, and pyre records that generation's
        // membership as the `YOUNG_RAWMALLOC` header flag where incminimark
        // keeps the `young_rawmalloced_objects` dict; a whole-word store would
        // clear it and the minor collection would free a live frame. The fast
        // path has already zeroed the header word.
        self.gen_initialize_tid_keep_flags(frame.clone(), descrs.jitframe_tid, st);

        // rewrite.py — emit_setfield(frame, c_null, descr=jf_*)
        // with (_, size, _) = unpack_fielddescr(descr). jitframe.py:63-81
        // every zeroed field (jf_descr / jf_force_descr / jf_savedata /
        // jf_guard_exc / jf_forward) is a GCREF or Ptr, i.e. pointer-
        // sized. majit's homogeneous JitFrame layout keeps all six at
        // sign_size; route through sign_size to mirror the per-descr
        // read.
        let zero = st.const_int(0);
        let signed_size_val = descrs.sign_size as i64;
        for &ofs in &[
            descrs.jf_descr_ofs,
            descrs.jf_force_descr_ofs,
            descrs.jf_savedata_ofs,
            descrs.jf_guard_exc_ofs,
            descrs.jf_forward_ofs,
        ] {
            self.emit_setfield_raw(frame.clone(), zero.clone(), ofs as i64, signed_size_val, st);
        }

        // rewrite.py — emit_getfield(frame_info, descrs.jfi_frame_depth),
        // rewrite.py — gen_initialize_len(frame, length, ...).
        // Both read/write lltype.Signed values (jfi_frame_depth and the
        // jf_frame length field).
        let jfi_frame_depth_ofs = st.const_int(0);
        let length = st.emit(mk_op(
            OpCode::GcLoadI,
            &[llfi.clone(), jfi_frame_depth_ofs, signed_size],
        ));
        self.emit_setfield_raw(
            frame.clone(),
            length,
            descrs.jf_frame_lengthofs as i64,
            signed_size_val,
            st,
        );

        // rewrite.py — emit_setfield(frame, ConstInt(llfi),
        // descr=descrs.jf_frame_info). jf_frame_info is Ptr(JITFRAMEINFO)
        // (jitframe.py) so the field size is the pointer width, which
        // in majit's layout coincides with sign_size.
        self.emit_setfield_raw(
            frame.clone(),
            llfi,
            descrs.jf_frame_info_ofs as i64,
            signed_size_val,
            st,
        );

        // rewrite.py:672-683 — store each arg at _ll_initial_locs[i] with
        // per-arg itemsize from getarraydescr_for_frame(arg.type).
        let arglist: Vec<Operand> = op
            .getarglist()
            .iter()
            .map(|a| st.get_box_replacement(a.clone()))
            .collect();
        let index_list = &callee_locs._ll_initial_locs;
        for (i, arg) in arglist.iter().enumerate() {
            // rewrite.py:675-677 — descr = cpu.getarraydescr_for_frame(arg.type);
            //                      _, itemsize, _ = unpack_arraydescr_size(descr)
            let arg_ty = st
                .result_type_of(arg)
                .expect("CALL_ASSEMBLER arg lacks a typed producer");
            let itemsize = descrs.frame_itemsize(arg_ty);
            let itemsize_ref = st.const_int(itemsize);
            // rewrite.py:678-681 — array_offset = index_list[i] (bytes);
            //                      _, basesize, _ = unpack_arraydescr(descr);
            //                      offset = basesize + array_offset.
            let offset = descrs.jf_frame_baseitemofs as i32 + index_list[i];
            let ofs_ref = st.const_int(offset as i64);
            st.emit(mk_op(
                OpCode::GcStore,
                &[frame.clone(), ofs_ref, arg.clone(), itemsize_ref],
            ));
        }

        // rewrite.py:685-695 — replace multi-arg with [frame] or
        // [frame, arglist[index_of_virtualizable]]
        let new_args = if callee_locs.index_of_virtualizable >= 0 {
            let vable_idx = callee_locs.index_of_virtualizable as usize;
            vec![frame, arglist[vable_idx].clone()]
        } else {
            vec![frame]
        };
        let call_asm = mk_op(op.opcode, &new_args);
        if let Some(d) = op.getdescr() {
            call_asm.setdescr(d);
        }
        if let Some(fa) = op.guard_fail_args() {
            call_asm.setfailargs(fa.iter().cloned().collect());
            if let Some(slot) = call_asm.fail_args_mut() {
                for a in slot.iter_mut() {
                    *a = st.get_box_replacement(a.clone());
                }
            }
        }
        st.emit_rewritten_from(op, call_asm);
    }
}

impl GcRewriterImpl {
    /// rewrite.py remove_bridge_exception: check a common
    /// case where SaveExcClass + SaveException + RestoreException
    /// appear at the start of a bridge and are unused. Strip them.
    ///
    /// rewrite.py:991 leaves an `XXX should check if the boxes are used
    /// later; but we just assume they aren't for now`.  A routed
    /// exception-guard handler bridge breaks that assumption: it records the
    /// same prefix but keeps the SaveException result as `last_exc_value`
    /// for handler code (`except E as e`), so stripping the prefix would
    /// leave a dangling reference.  Do the deferred use-check — the
    /// RESTORE_EXCEPTION carries the canonical class/value operands, so only
    /// strip when no op after the prefix reuses either.
    fn remove_bridge_exception(ops: &[OpRc]) -> Cow<'_, [OpRc]> {
        let mut start = 0;
        if ops
            .first()
            .is_some_and(|op| op.opcode == OpCode::IncrementDebugCounter)
        {
            start = 1;
        }
        if ops.len() >= start + 3
            && ops[start].opcode == OpCode::SaveExcClass
            && ops[start + 1].opcode == OpCode::SaveException
            && ops[start + 2].opcode == OpCode::RestoreException
        {
            let restore = &ops[start + 2];
            let class_ref = restore.arg(0);
            let value_ref = restore.arg(1);
            let used_later = ops[start + 3..].iter().any(|op| {
                (0..op.num_args()).any(|k| {
                    let a = op.arg(k);
                    a.same_box(&class_ref) || a.same_box(&value_ref)
                }) || op.guard_fail_args().is_some_and(|fa| {
                    fa.iter()
                        .any(|a| a.same_box(&class_ref) || a.same_box(&value_ref))
                })
            });
            if !used_later {
                let mut result = Vec::with_capacity(ops.len() - 3);
                result.extend_from_slice(&ops[..start]);
                result.extend_from_slice(&ops[start + 3..]);
                return Cow::Owned(result);
            }
        }
        // `GcRewriterAssembler.remove_bridge_exception` returns the same
        // list when the prefix is absent.
        Cow::Borrowed(ops)
    }
}

/// First free value position over every segment.
///
/// Result positions are consumed only by result-producing ops; a Void-result
/// op never occupies a position slot. Skip sentinels and constants for the
/// same reason `emit` assigns no position id when `rt == Type::Void`.
///
/// `rewrite.py remove_constptr` replaces a ConstPtr argument with a fresh
/// LOAD_FROM_GC_TABLE box. RPython boxes retain distinct identity when their
/// producer was optimized away, so pyre's numeric namespace must reserve
/// argument and guard-fail positions too. Forced virtuals can carry
/// producerless constant boxes as allocation lengths; reusing such a raw
/// position would alias an Int box with the new Ref box in backend SSA.
///
/// The `VoidOp(u32::MAX)` sentinel is the one raw payload that must not enter
/// the high-water mark: it would pin `next_pos` at `u32::MAX`, which the first
/// `+= 1` in `emit` overflows — a panic where overflow checks are on, and a
/// silent wrap to 0 in release, handing every rewritten op a position that
/// aliases a live operand. Every other non-constant payload still counts,
/// `TempVar` included: its sentinel range sits above the body positions, and
/// lowering the mark past it lets a new box collide with a position that
/// reaches the backend through resume data rather than through
/// `pos`/args/failargs. A trace whose ops all carry void results (`Label` /
/// `SETARRAYITEM_GC` / `FINISH`) still gets its mark from the argument scan.
fn result_pos_high_water(segments: &[&[OpRc]]) -> u32 {
    let counts = |pos: OpRef| !pos.is_none() && !pos.is_constant() && pos.ty() != Some(Type::Void);
    let mut max_raw_pos: Option<u32> = None;
    let mut reserve = |pos: OpRef| {
        if counts(pos) {
            max_raw_pos = Some(max_raw_pos.map_or(pos.raw(), |old| old.max(pos.raw())));
        }
    };
    for ops in segments {
        for op in *ops {
            reserve(op.pos().get());
            for arg in op.getarglist() {
                reserve(arg.to_opref());
            }
            if let Some(fail_args) = op.guard_fail_args() {
                for arg in fail_args.iter() {
                    reserve(arg.to_opref());
                }
            }
        }
    }
    max_raw_pos.map_or(0, |max_pos| max_pos.saturating_add(1))
}

impl GcRewriterImpl {
    /// One pass of `GcRewriterAssembler.rewrite` over a single segment.
    /// The caller supplies a fresh `RewriteState`; only `next_pos` and
    /// `gcrefs_output_list` are carried across segments.
    fn rewrite_ops_with_state(&self, ops: &[OpRc], st: &mut RewriteState<'_>) {
        for (i, orig_op) in ops.iter().enumerate() {
            // rewrite.py `if op is self._changed_op: op = self._changed_op_to`.
            let op_rc: OpRc = match (&st._changed_op, &st._changed_op_to) {
                (Some(changed), Some(to)) if OpRc::ptr_eq(changed, orig_op) => to.clone(),
                _ => orig_op.clone(),
            };
            let op: &OpRc = &op_rc;

            // rewrite.py — is_guard OR could_merge_with_next_guard
            // triggers emit_pending_zeros at the top of the iteration.
            // could_merge_with_next_guard may also emit a SAME_AS_I and
            // stash a rewritten guard via remove_tested_failarg, so it
            // must be called regardless of whether the flush path is
            // taken — the flush only fires when one of the two branches
            // returns true.
            let merges = self.could_merge_with_next_guard(op, i, ops, st);
            if op.opcode.is_guard() || merges {
                st.emit_pending_zeros();
            }

            // rewrite.py — transform_to_gc_load forwards memory
            // accessors to GC_LOAD / GC_STORE forms.  Returns true only
            // for the GETFIELD_GC fast-path, which also emits the
            // forwarded op itself.
            if self.transform_to_gc_load(op, st) {
                continue;
            }

            match op.opcode {
                // Skip debug merge points (they carry no semantics).
                OpCode::DebugMergePoint => continue,

                // rewrite.py emit_label
                OpCode::Label => {
                    st.emitting_an_operation_that_can_collect();
                    st.known_lengths.clear();
                    // rewrite.py emit_label resets the per-block load CSE,
                    // then the main loop falls through to emit_op.
                    st.gcrefs_recently_loaded.clear();
                    st.emit_maybe_forwarded(&op_rc);
                }

                // ── Allocation ──
                OpCode::New | OpCode::NewWithVtable => {
                    self.handle_new(op, st);
                }
                // rewrite.py:485-494 — descr source per opcode mirrors
                // upstream: NEW_ARRAY threads op.getdescr(); NEWSTR /
                // NEWUNICODE thread self.gc_ll_descr.{str,unicode}_descr.
                OpCode::NewArray | OpCode::NewArrayClear => {
                    let descr_ref = op.getdescr().expect("NEW_ARRAY must carry an ArrayDescr");
                    self.handle_new_array(descr_ref, op, st, 0); // FLAG_ARRAY
                }
                OpCode::Newstr => {
                    // rewrite.py `handle_new_array(self.gc_ll_descr.str_descr, op, FLAG_STR)`.
                    self.handle_new_array(self.str_descr.clone(), op, st, 1);
                }
                OpCode::Newunicode => {
                    // rewrite.py `handle_new_array(self.gc_ll_descr.unicode_descr, op, FLAG_UNICODE)`.
                    self.handle_new_array(self.unicode_descr.clone(), op, st, 2);
                }

                // ── COPYSTRCONTENT / COPYUNICODECONTENT → memcpy CALL_N ──
                // rewrite.py `rewrite_copy_str_content` replaces
                // the copy op with LOAD_EFFECTIVE_ADDRESS × 2 + CALL_N.
                OpCode::Copystrcontent | OpCode::Copyunicodecontent => {
                    self.rewrite_copy_str_content(op, st);
                }

                // ── Stores that may need a write barrier ──
                //
                // rewrite.py:392-404 — the write-barrier section runs AFTER
                // `transform_to_gc_load` has forwarded the store op to
                // GC_STORE / GC_STORE_INDEXED.  `emit_maybe_forwarded`
                // follows the forward and emits the lowered op.
                // rewrite.py:393 `if self.gc_ll_descr.write_barrier_descr is
                // not None:` — a collector that needs no write barrier
                // (`gc.py GcLLDescr_boehm.write_barrier_descr = None`)
                // must not have `COND_CALL_GC_WB*` emitted for it at all.
                OpCode::SetfieldGc if self.wb_descr.is_some() => {
                    // rewrite.py — consider_setfield_gc clears the
                    // pending zero-init entry before WB emission.
                    self.consider_setfield_gc(op, st);
                    self.handle_write_barrier_setfield(op, st);
                    st.emit_maybe_forwarded(&op_rc);
                    continue;
                }
                OpCode::SetinteriorfieldGc if self.wb_descr.is_some() => {
                    // rewrite.py:946 `handle_write_barrier_setinteriorfield
                    // = handle_write_barrier_setarrayitem`.
                    self.handle_write_barrier_setarrayitem(op, st);
                    st.emit_maybe_forwarded(&op_rc);
                    continue;
                }
                OpCode::SetarrayitemGc if self.wb_descr.is_some() => {
                    // rewrite.py:401-404
                    self.consider_setarrayitem_gc(op, st);
                    self.handle_write_barrier_setarrayitem(op, st);
                    st.emit_maybe_forwarded(&op_rc);
                    continue;
                }
                // rewrite.py:405-412 — the `else` arm of that gate: the
                // zero-init bookkeeping still runs and no barrier is emitted.
                // Upstream reaches the ordinary lowering by *not* `continue`
                // -ing; pyre's loop body ends at this match, so the same
                // lowering is spelled out as the `emit_maybe_forwarded` the
                // barrier arms and the catch-all both use.
                // `SETINTERIORFIELD_GC` gets no `consider_*` there, so it
                // falls to the catch-all unchanged.
                OpCode::SetfieldGc => {
                    self.consider_setfield_gc(op, st);
                    st.emit_maybe_forwarded(&op_rc);
                    continue;
                }
                OpCode::SetarrayitemGc => {
                    self.consider_setarrayitem_gc(op, st);
                    st.emit_maybe_forwarded(&op_rc);
                    continue;
                }

                // ── call_assembler: rewrite.py handle_call_assembler ──
                OpCode::CallAssemblerI
                | OpCode::CallAssemblerR
                | OpCode::CallAssemblerF
                | OpCode::CallAssemblerN => {
                    // rewrite.py always `handle_call_assembler` →
                    // `gen_malloc_frame` → `gen_malloc_nursery_varsize_frame`.
                    self.handle_call_assembler(op, st);
                    continue;
                }

                // ── Operations that can trigger GC ──
                _ if op.opcode.can_malloc() => {
                    // rewrite.py — emitting_an_operation_that_can_collect
                    // already flushes pending zeros, then emit_op.
                    st.emitting_an_operation_that_can_collect();
                    st.emit_maybe_forwarded(&op_rc);
                }

                // ── GUARD_ALWAYS_FAILS lowering (rewrite.py:419-426) ──
                // Upstream turns an always-failing guard into
                //   SAME_AS_I(0)
                //   GUARD_VALUE(same_as, ConstInt(1))
                // so the backend can share its GUARD_VALUE emission path.
                // failargs are carried over via copy_and_change.
                OpCode::GuardAlwaysFails => {
                    let zero = st.const_int(0);
                    let one = st.const_int(1);
                    let same = mk_op(OpCode::SameAsI, &[zero]);
                    let same_pos = st.emit_result(same, OpRef::NONE);
                    let newop =
                        op.copy_and_change(OpCode::GuardValue, Some(&[same_pos, one]), None);
                    st.emit_op(&OpRc::new(newop));
                }

                // ── Guards: emit_pending_zeros was already called at the
                // top of the iteration per rewrite.py:376-378; here we only
                // need to emit the (forwarded) guard op itself. Guards do
                // not clear wb_applied — only emitting_an_operation_that_
                // can_collect does that (rewrite.py:699-711).
                _ if op.opcode.is_guard() => {
                    // rewrite.py falls through to emit_op after
                    // emit_pending_zeros. emit_op copy_and_changes guards
                    // (failargs) and emit_maybe_forwarded records a
                    // non-void result (GUARD_EXCEPTION) under the original
                    // position.
                    st.emit_maybe_forwarded(&op_rc);
                }

                // ── Everything else: pass through unchanged. ──
                OpCode::CondCallGcWb => {
                    let obj = st.get_box_replacement(op.arg(0));
                    st.emit_maybe_forwarded(&op_rc);
                    st.remember_wb(&obj);
                }
                OpCode::CondCallGcWbArray => {
                    // rewrite.py `handle_write_barrier_setarrayitem`: WB_ARRAY
                    // does not mark the base as barrier-applied; future
                    // setarrayitems still need their own barrier (no
                    // remember_wb call). Then emit_op.
                    st.emit_maybe_forwarded(&op_rc);
                }
                // ── Final ops (Jump, Finish) flush pending zeros before emit. ──
                _ if op.opcode.is_final() => {
                    st.emit_pending_zeros();
                    st.emit_maybe_forwarded(&op_rc);
                }

                // rewrite.py:383-387 — record INT_ADD/INT_SUB whose
                // constant operand is later folded into a GC_STORE_INDEXED
                // / GC_LOAD_INDEXED offset via `_try_use_older_box`.
                OpCode::IntAdd | OpCode::IntAddOvf => {
                    // rewrite.py records on the original op, then emit_op.
                    st.record_int_add_or_sub(op, false);
                    st.emit_maybe_forwarded(&op_rc);
                }
                OpCode::IntSub | OpCode::IntSubOvf => {
                    st.record_int_add_or_sub(op, true);
                    st.emit_maybe_forwarded(&op_rc);
                }

                // ── Everything else: follow forwarding if `transform_to_gc_load`
                // has forwarded this op (GETARRAYITEM, GETFIELD_RAW,
                // SETFIELD_RAW, SETARRAYITEM_RAW, SETINTERIORFIELD_RAW,
                // ARRAYLEN_GC, RAW_LOAD, RAW_STORE, GC_LOAD_INDEXED,
                // GC_STORE_INDEXED …); otherwise pass through unchanged.
                _ => {
                    st.emit_maybe_forwarded(&op_rc);
                }
            }
        }

        // Flush any remaining pending zeros at end of trace.
        st.emit_pending_zeros();

        // rewrite.py `self.newops` is the list emit just appended;
        // hand those `OpRc` identities to the backend instead of
        // cloning each `Op` out of the Rc.
        // rewrite.py post-condition: `remove_constptr` replaced
        // every non-null reference-constant *operand* with a
        // `LoadFromGcTable` result, so no raw non-null `GcRef` is left for
        // a backend to bake as an immortal immediate. Failargs are NOT
        // operands — they keep their constants and are forwarded by the
        // resume path (`rd_consts`), so this scans operands only.
        // JIT_DEBUG keeps its constants inline (rewrite.py:105). This is
        // the "ConstPtr transient-only" invariant: any survivor is a
        // missed emit point.
        #[cfg(debug_assertions)]
        for op in &st.out {
            if op.opcode == OpCode::JitDebug {
                continue;
            }
            for i in 0..op.num_args() {
                debug_assert!(
                    !matches!(op.arg(i).const_value(), Some(Value::Ref(g)) if !g.is_null()),
                    "rewrite output {:?} still carries a non-null ConstPtr operand at \
                     arg {i}; remove_constptr must route it through the gc_table",
                    op.opcode
                );
            }
        }
    }
}

impl GcRewriter for GcRewriterImpl {
    fn rewrite_for_gc(&self, ops: &[OpRc]) -> Vec<OpRc> {
        let (rewritten, gcrefs) = self.rewrite_for_gc_with_constants(ops, &ConstMap::default());
        // This wrapper drops the gc_table output list. A non-null ConstPtr
        // operand is rewritten to LoadFromGcTable, which needs that list to
        // build the table; callers carrying one must use the
        // constants-returning form. Fail fast rather than emit IR whose
        // table is silently discarded.
        assert!(
            gcrefs.is_empty(),
            "rewrite_for_gc discards gc_table refs; use rewrite_for_gc_with_constants"
        );
        rewritten
    }

    fn rewrite_for_gc_with_constants(
        &self,
        ops: &[OpRc],
        constants: &ConstMap<Const>,
    ) -> (Vec<OpRc>, Vec<GcRef>) {
        let (mut parts, gcrefs) = self
            .rewrite_for_gc_segments(std::slice::from_ref(&ops), constants)
            .expect("GcRewriterImpl segmented rewrite");
        (parts.pop().unwrap_or_default(), gcrefs)
    }

    fn rewrite_for_gc_segments(
        &self,
        segments: &[&[OpRc]],
        constants: &ConstMap<Const>,
    ) -> Option<(Vec<Vec<OpRc>>, Vec<GcRef>)> {
        // `GcRewriterAssembler.rewrite` runs once per assemble_bridge, so
        // each segment starts from a fresh RewriteState. The result-position
        // high-water mark is taken over every segment first, and minted
        // positions keep climbing, so no position is issued twice. One
        // gcrefs list is shared so LoadFromGcTable indices stay unique.
        let stripped: Vec<std::borrow::Cow<'_, [OpRc]>> = segments
            .iter()
            .copied()
            .map(Self::remove_bridge_exception)
            .collect();
        let views: Vec<&[OpRc]> = stripped.iter().map(|ops| ops.as_ref()).collect();
        let mut next_pos = result_pos_high_water(&views);
        let mut gcrefs = Vec::new();
        let mut outs = Vec::with_capacity(views.len());
        for ops in views {
            let mut st = RewriteState::with_constants(ops.len(), next_pos, constants);
            st.gcrefs_output_list = std::mem::take(&mut gcrefs);
            self.rewrite_ops_with_state(ops, &mut st);
            next_pos = st.next_pos;
            gcrefs = std::mem::take(&mut st.gcrefs_output_list);
            outs.push(st.out);
        }
        Some((outs, gcrefs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::collector::const_ptr_table_test_guard;
    use majit_ir::descr::{ArrayDescr, Descr, DescrRef, SizeDescr};
    use majit_ir::value::{InputArg, Type};

    #[test]
    fn remove_ref_constants_interns_forwarded_inputarg_on_label() {
        let _const_ptr_table = const_ptr_table_test_guard();
        // Compact remap leaves a virtual InputArg at a dead index whose
        // `_forwarded` is Const(Ref). `to_opref` still spells InputArgRef(99);
        // rewrite.py `remove_constptr` sees the Const after get_box_replacement.
        let ia = InputArg::new_ref_rc(99);
        let operand = Operand::from_bound_inputarg(&ia);
        operand.set_forwarded_const(Const::from_gcref(GcRef(0x9C_0000_1100)));
        let label = Op::new(OpCode::Label, &[operand]);
        let (out, gcrefs) = remove_ref_constants(&[label], 0);
        assert_eq!(gcrefs, vec![GcRef(0x9C_0000_1100)]);
        assert_eq!(out[0].opcode, OpCode::LoadFromGcTable);
        assert_eq!(out[1].opcode, OpCode::Label);
        assert_eq!(out[1].arg(0).to_opref(), OpRef::ref_op(0));
    }

    #[test]
    fn remove_ref_constants_does_not_intern_inputarg_observation() {
        // `_resref` on an InputArg is the object seen while tracing, not a
        // folded constant. Interning it would replace every later input
        // with that first object.
        let ia = InputArg::new_ref_rc(99);
        ia.set_value(Value::Ref(GcRef(0x1000)));
        let operand = Operand::from_bound_inputarg(&ia);
        let label = Op::new(OpCode::Label, &[operand]);
        let (out, gcrefs) = remove_ref_constants(&[label], 0);
        assert!(gcrefs.is_empty(), "observation must not enter the gc table");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].opcode, OpCode::Label);
        assert_eq!(out[0].arg(0).to_opref(), OpRef::input_arg_ref(99));
    }

    #[test]
    fn remove_ref_constants_does_not_intern_offstream_resref_observation() {
        let producer = OpRc::new(Op::new(OpCode::SameAsR, &[]));
        producer.pos().set(OpRef::ref_op(99));
        producer.set_value(Value::Ref(GcRef(0x1000)));
        let operand = Operand::from_bound_op(&producer);
        let label = Op::new(OpCode::Label, &[operand]);
        let (out, gcrefs) = remove_ref_constants(&[label], 0);
        assert!(gcrefs.is_empty(), "observation must not enter the gc table");
        assert_eq!(out[0].opcode, OpCode::Label);
        assert_eq!(out[0].arg(0).to_opref(), OpRef::ref_op(99));
    }

    #[test]
    fn remove_ref_constants_keeps_live_inputarg_forwarded_to_null() {
        // Token input 0 is the virtualizable identity. Optimizer forwarding
        // it to ConstPtr(0) must not bake that null into the guard snapshot:
        // deopt would then call consume_vable_info with identity 0.
        let ia = InputArg::new_ref_rc(0);
        let operand = Operand::from_bound_inputarg(&ia);
        operand.set_forwarded_const(Const::from_gcref(GcRef(0)));
        let guard = Op::new(OpCode::GuardTrue, &[operand.clone()]);
        guard.setfailargs(vec![operand].into());
        let (out, gcrefs) = remove_ref_constants_for_inputs(&[guard], 1, &[0]);
        assert!(gcrefs.is_empty(), "null must stay un-interned");
        let fa = out[0].getfailargs().expect("guard failargs");
        assert!(
            fa[0].to_opref().is_input_arg(),
            "live InputArg forwarded to null must stay an InputArg, got {:?}",
            fa[0].to_opref()
        );
    }

    impl GcRewriterImpl {
        fn rewrite_ops(&self, ops: &[Op]) -> Vec<OpRc> {
            let boxed: Vec<OpRc> = ops.iter().cloned().map(OpRc::new).collect();
            GcRewriter::rewrite_for_gc(self, &boxed)
        }

        fn rewrite_ops_with_constants(
            &self,
            ops: &[Op],
            constants: &ConstMap<Const>,
        ) -> (Vec<OpRc>, ConstMap<Const>, Vec<GcRef>) {
            let boxed: Vec<OpRc> = ops.iter().cloned().map(OpRc::new).collect();
            let (out, gcrefs) = self.rewrite_for_gc_with_constants(&boxed, constants);
            // The rewrite only reads the pool. Hand the caller's map back
            // so existing assertions keep their tuple shape.
            (out, constants.clone(), gcrefs)
        }
    }

    const TEST_STANDARD_ARRAY_BASESIZE: usize = std::mem::size_of::<usize>();
    const TEST_STANDARD_ARRAY_LENGTH_OFS: usize = 0;
    const TEST_MALLOC_ARRAY_FN: i64 = 0x1111;
    const TEST_MALLOC_ARRAY_NONSTANDARD_FN: i64 = 0x2222;
    const TEST_MALLOC_ARRAY_OLDGEN_FN: i64 = 0x1113;
    const TEST_MALLOC_ARRAY_NONSTANDARD_OLDGEN_FN: i64 = 0x2223;
    const TEST_MALLOC_STR_FN: i64 = 0x3333;
    const TEST_MALLOC_UNICODE_FN: i64 = 0x4444;
    const TEST_MALLOC_BIG_FIXEDSIZE_FN: i64 = 0x5555;
    const TEST_MALLOC_BIG_FIXEDSIZE_OLDGEN_FN: i64 = 0x5556;

    // ── Minimal concrete descriptor implementations for testing ──

    #[derive(Debug)]
    struct TestSizeDescr {
        size: usize,
        type_id: u32,
        vtable: usize,
        w_class: Option<i64>,
        headerless: bool,
        non_moving: bool,
        gc_fields: Vec<Arc<dyn FieldDescr>>,
    }

    impl Descr for TestSizeDescr {
        fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
            Some(self)
        }
    }

    impl SizeDescr for TestSizeDescr {
        fn size(&self) -> usize {
            self.size
        }
        fn type_id(&self) -> u32 {
            self.type_id
        }
        fn is_immutable(&self) -> bool {
            false
        }
        fn is_object(&self) -> bool {
            self.vtable != 0
        }
        fn vtable(&self) -> usize {
            self.vtable
        }
        fn w_class_obj(&self) -> Option<i64> {
            self.w_class
        }
        fn headerless(&self) -> bool {
            self.headerless
        }
        fn non_moving(&self) -> bool {
            self.non_moving
        }
        fn gc_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
            &self.gc_fields
        }
    }

    #[derive(Debug)]
    struct TestFieldDescr {
        offset: usize,
        field_size: usize,
        field_type: Type,
    }

    impl Descr for TestFieldDescr {
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for TestFieldDescr {
        fn offset(&self) -> usize {
            self.offset
        }
        fn field_size(&self) -> usize {
            self.field_size
        }
        fn field_type(&self) -> Type {
            self.field_type
        }
    }

    #[derive(Debug)]
    struct TestWClassFieldDescr {
        offset: usize,
    }

    impl Descr for TestWClassFieldDescr {
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for TestWClassFieldDescr {
        // This fixture exists to be the class word; it says so rather than
        // relying on a consumer to recognise its name.
        fn is_w_class(&self) -> bool {
            true
        }
        fn offset(&self) -> usize {
            self.offset
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> Type {
            Type::Ref
        }
        // Deliberately *not* spelled `w_class`, and deliberately spelled like
        // an ordinary payload field. `clear_gc_fields` must find this slot
        // through `is_w_class()` above; a consumer that recognised the class
        // word by name instead would see an ordinary field here and re-arm the
        // delayed NULL store that overwrites the class object, which is what
        // the tests below assert against.
        fn field_name(&self) -> &str {
            "cls"
        }
    }

    #[derive(Debug)]
    struct TestArrayDescr {
        base_size: usize,
        item_size: usize,
        type_id: u32,
        item_type: Type,
        len_descr: Option<Arc<TestFieldDescr>>,
        non_moving: bool,
    }

    impl Descr for TestArrayDescr {
        fn as_array_descr(&self) -> Option<&dyn ArrayDescr> {
            Some(self)
        }
    }

    impl ArrayDescr for TestArrayDescr {
        fn base_size(&self) -> usize {
            self.base_size
        }
        fn item_size(&self) -> usize {
            self.item_size
        }
        fn type_id(&self) -> u32 {
            self.type_id
        }
        fn item_type(&self) -> Type {
            self.item_type
        }
        fn len_descr(&self) -> Option<&dyn FieldDescr> {
            self.len_descr
                .as_ref()
                .map(|fd| fd.as_ref() as &dyn FieldDescr)
        }
        fn non_moving(&self) -> bool {
            self.non_moving
        }
    }

    fn make_rewriter() -> GcRewriterImpl {
        GcRewriterImpl {
            nursery_free_addr: 0x1000,
            nursery_top_addr: 0x2000,
            max_nursery_size: 4096,
            wb_descr: Some(WriteBarrierDescr {
                jit_wb_if_flag: 1,
                jit_wb_if_flag_byteofs: 0,
                jit_wb_if_flag_singlebyte: 1,
                jit_wb_cards_set: 0,
                jit_wb_card_page_shift: 0,
                jit_wb_cards_set_byteofs: 0,
                jit_wb_cards_set_singlebyte: 0,
            }),
            jitframe_info: None,
            call_assembler_callee_locs: None,
            // llmodel.py:39 default keeps existing pre-scale-everything behavior
            // in tests written against it; per-backend overrides have dedicated
            // tests below.
            load_supported_factors: &[1],
            // x86/runner.py:22 + arm/runner.py:26 — production parity;
            // dedicated `_without_load_effective_address` tests flip this
            // to `false` to exercise the fallback (rewrite.py:1089-1098).
            supports_load_effective_address: true,
            // Match the production nursery's zero-fill behavior; the
            // `clear_gc_fields` / `_delayed_zero_setfields` tests flip
            // this to `false` per-test.
            malloc_zero_filled: true,
            // gc.py:39-43, gc.py:46-49 fields.
            memcpy_fn: majit_ir::memcpy_fn_addr(),
            memcpy_descr: majit_ir::make_memcpy_calldescr(),
            str_descr: str_array_descr(),
            unicode_descr: unicode_array_descr(),
            str_hash_descr: hash_field_descr(),
            unicode_hash_descr: hash_field_descr(),
            // gc.py:33-37 `fielddescr_vtable`. Test fixtures always
            // install a Some so the existing test_new_with_vtable
            // continues to exercise the typeptr stamping path.
            fielddescr_vtable: Some(majit_ir::make_vtable_field_descr()),
            // gc.py:394 `fielddescr_tid`. Test fixtures always install a
            // Some so existing handle_new tests continue to exercise the
            // tid header stamping path (matches framework-GC mode).
            fielddescr_tid: Some(majit_ir::make_tid_field_descr()),
            malloc_array_fn: TEST_MALLOC_ARRAY_FN,
            malloc_array_nonstandard_fn: TEST_MALLOC_ARRAY_NONSTANDARD_FN,
            malloc_array_oldgen_fn: TEST_MALLOC_ARRAY_OLDGEN_FN,
            malloc_array_nonstandard_oldgen_fn: TEST_MALLOC_ARRAY_NONSTANDARD_OLDGEN_FN,
            malloc_str_fn: TEST_MALLOC_STR_FN,
            malloc_unicode_fn: TEST_MALLOC_UNICODE_FN,
            malloc_big_fixedsize_fn: TEST_MALLOC_BIG_FIXEDSIZE_FN,
            malloc_big_fixedsize_oldgen_fn: TEST_MALLOC_BIG_FIXEDSIZE_OLDGEN_FN,
            malloc_array_descr: majit_ir::make_malloc_array_calldescr(),
            malloc_array_nonstandard_descr: majit_ir::make_malloc_array_nonstandard_calldescr(),
            malloc_str_descr: majit_ir::make_malloc_str_calldescr(),
            malloc_unicode_descr: majit_ir::make_malloc_unicode_calldescr(),
            malloc_big_fixedsize_descr: majit_ir::make_malloc_big_fixedsize_calldescr(),
            standard_array_basesize: TEST_STANDARD_ARRAY_BASESIZE,
            standard_array_length_ofs: TEST_STANDARD_ARRAY_LENGTH_OFS,
            headerless_fixedsize: false,
        }
    }

    fn mk_op(opcode: OpCode, args: &[OpRef], pos: u32) -> Op {
        let args: Vec<Operand> = args.iter().map(|a| ro(*a)).collect();
        let op = Op::new(opcode, &args);
        op.pos().set(OpRef::op_typed(pos, opcode.result_type()));
        op
    }

    use majit_ir::forwarding::bound_operand_from_opref as ro;

    fn mk_op_with_descr(opcode: OpCode, args: &[OpRef], pos: u32, descr: DescrRef) -> Op {
        let args: Vec<Operand> = args.iter().map(|a| ro(*a)).collect();
        let op = Op::with_descr(opcode, &args, descr);
        op.pos().set(OpRef::op_typed(pos, opcode.result_type()));
        op
    }

    fn size_descr(size: usize, type_id: u32) -> DescrRef {
        Arc::new(TestSizeDescr {
            size,
            type_id,
            vtable: 0,
            w_class: None,
            headerless: false,
            non_moving: false,
            gc_fields: Vec::new(),
        })
    }

    fn headerless_size_descr(size: usize, type_id: u32) -> DescrRef {
        headerless_size_descr_with_gc_fields(size, type_id, Vec::new())
    }

    fn headerless_size_descr_with_gc_fields(
        size: usize,
        type_id: u32,
        gc_fields: Vec<Arc<dyn FieldDescr>>,
    ) -> DescrRef {
        Arc::new(TestSizeDescr {
            size,
            type_id,
            vtable: 0,
            w_class: None,
            headerless: true,
            non_moving: false,
            gc_fields,
        })
    }

    fn non_moving_size_descr(size: usize, type_id: u32) -> DescrRef {
        Arc::new(TestSizeDescr {
            size,
            type_id,
            vtable: 0,
            w_class: None,
            headerless: false,
            non_moving: true,
            gc_fields: Vec::new(),
        })
    }

    fn size_descr_with_gc_fields(
        size: usize,
        type_id: u32,
        gc_fields: Vec<Arc<dyn FieldDescr>>,
    ) -> DescrRef {
        Arc::new(TestSizeDescr {
            size,
            type_id,
            vtable: 0,
            w_class: None,
            headerless: false,
            non_moving: false,
            gc_fields,
        })
    }

    fn size_descr_with_w_class(
        size: usize,
        type_id: u32,
        vtable: usize,
        w_class: Option<i64>,
        gc_fields: Vec<Arc<dyn FieldDescr>>,
    ) -> DescrRef {
        Arc::new(TestSizeDescr {
            size,
            type_id,
            vtable,
            w_class,
            headerless: false,
            non_moving: false,
            gc_fields,
        })
    }

    fn vtable_descr(size: usize, type_id: u32, vtable: usize) -> DescrRef {
        Arc::new(TestSizeDescr {
            size,
            type_id,
            vtable,
            w_class: None,
            headerless: false,
            non_moving: false,
            gc_fields: Vec::new(),
        })
    }

    fn ref_field_descr() -> DescrRef {
        Arc::new(TestFieldDescr {
            offset: 0,
            field_size: 8,
            field_type: Type::Ref,
        })
    }

    fn int_field_descr() -> DescrRef {
        Arc::new(TestFieldDescr {
            offset: 8,
            field_size: 8,
            field_type: Type::Int,
        })
    }

    fn array_len_field_descr() -> Arc<TestFieldDescr> {
        Arc::new(TestFieldDescr {
            offset: TEST_STANDARD_ARRAY_LENGTH_OFS,
            field_size: std::mem::size_of::<usize>(),
            field_type: Type::Int,
        })
    }

    fn array_descr_ref() -> DescrRef {
        Arc::new(TestArrayDescr {
            base_size: TEST_STANDARD_ARRAY_BASESIZE,
            item_size: 8,
            type_id: 5,
            item_type: Type::Ref,
            len_descr: Some(array_len_field_descr()),
            non_moving: false,
        })
    }

    fn non_moving_array_descr() -> DescrRef {
        Arc::new(TestArrayDescr {
            base_size: TEST_STANDARD_ARRAY_BASESIZE,
            item_size: 8,
            type_id: 55,
            item_type: Type::Ref,
            len_descr: Some(array_len_field_descr()),
            non_moving: true,
        })
    }

    fn array_descr_int() -> DescrRef {
        Arc::new(TestArrayDescr {
            base_size: TEST_STANDARD_ARRAY_BASESIZE,
            item_size: 4,
            type_id: 6,
            item_type: Type::Int,
            len_descr: Some(array_len_field_descr()),
            non_moving: false,
        })
    }

    // ── Test 1: NEW → CALL_MALLOC_NURSERY + tid init ──

    #[test]
    fn test_new_rewrite() {
        let rw = make_rewriter();
        let ops = vec![Op::with_descr(OpCode::New, &[], size_descr(32, 7))];

        let (result, constants, _gcrefs) =
            rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        // Expect: CallMallocNursery, GcStore (tid)
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::CallMallocNursery);
        // rewrite.py:474-484 parity: size arg is ConstInt(descr.size + GcHeader::SIZE).
        let size_val = result[0]
            .arg(0)
            .to_opref()
            .inline_const_bits()
            .expect("inline ConstInt");
        assert_eq!(size_val, (32 + crate::header::GcHeader::SIZE) as i64);
        assert_eq!(result[1].opcode, OpCode::GcStore); // tid init
        let tid_val = result[1]
            .arg(2)
            .to_opref()
            .inline_const_bits()
            .expect("inline ConstInt");
        assert_eq!(tid_val, 7); // type_id = 7
        // gen_initialize_tid stores the whole Signed `HDR.tid`, clearing the
        // flags half along with stamping the type id.
        let store_size = result[1]
            .arg(3)
            .to_opref()
            .inline_const_bits()
            .expect("inline ConstInt");
        assert_eq!(store_size, std::mem::size_of::<usize>() as i64);

        for (_key, c) in &constants {
            assert_eq!(
                c.get_type(),
                Type::Int,
                "GC rewriter must return the RPython ConstInt.type equivalent \
                 for each constant it mints"
            );
        }
    }

    /// A `non_moving` descr takes the `malloc_big_fixedsize` helper even at a
    /// size the nursery would happily accept, so the object is born in the
    /// non-moving old generation. The size the helper is handed is still the
    /// headered total, and no separate `gen_initialize_tid` follows — the
    /// helper stamps the tid itself.
    #[test]
    fn test_non_moving_new_declines_nursery_at_small_size() {
        let rw = make_rewriter();
        let payload_size = 32;
        let type_id = 53;
        let ops = vec![Op::with_descr(
            OpCode::New,
            &[],
            non_moving_size_descr(payload_size, type_id),
        )];

        let (result, _constants, _gcrefs) =
            rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        assert!(
            result
                .iter()
                .all(|op| op.opcode != OpCode::CallMallocNursery),
            "a non_moving descr must never reach the nursery bump path"
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::CallR);
        assert_eq!(result[1].opcode, OpCode::CheckMemoryError);
        assert_eq!(
            result[0]
                .arg(0)
                .to_opref()
                .inline_const_bits()
                .expect("malloc_big_fixedsize fn const"),
            TEST_MALLOC_BIG_FIXEDSIZE_OLDGEN_FN,
            "a non_moving descr takes the old-generation twin: the plain helper \
             dispatches on size and would nursery-allocate this one"
        );
        assert_eq!(
            result[0]
                .arg(1)
                .to_opref()
                .inline_const_bits()
                .expect("headered total size const"),
            round_up(payload_size + crate::header::GcHeader::SIZE) as i64
        );
    }

    /// The varsize twin: a `non_moving` array descr declines the nursery at a
    /// constant length the nursery would accept, and lands on the old-generation
    /// `malloc_array` helper rather than the nursery one.  The length still
    /// reaches the helper as its last argument, so the block's GcArray length
    /// header is stamped by the helper itself.
    #[test]
    fn test_non_moving_new_array_declines_nursery_at_small_length() {
        let rw = make_rewriter();
        let num_elem = 3;
        let len_ref = OpRef::int_op(10_000);
        let mut constants: ConstMap<Const> = ConstMap::default();
        constants.insert(10_000, Const::Int(num_elem));
        let new_array = Op::with_descr(
            OpCode::NewArrayClear,
            &[ro(len_ref)],
            non_moving_array_descr(),
        );
        new_array.pos().set(OpRef::ref_op(0));
        let ops = vec![new_array, Op::new(OpCode::Finish, &[])];

        let (result, _constants, _gcrefs) = rw.rewrite_ops_with_constants(&ops, &constants);

        assert!(
            result
                .iter()
                .all(|op| op.opcode != OpCode::CallMallocNursery
                    && op.opcode != OpCode::CallMallocNurseryVarsize),
            "a non_moving array descr must never reach either nursery path"
        );
        let call = result
            .iter()
            .find(|op| op.opcode == OpCode::CallR)
            .expect("non_moving NEW_ARRAY must emit the typed malloc CALL_R");
        assert_eq!(
            call.arg(0)
                .to_opref()
                .inline_const_bits()
                .expect("malloc_array fn const"),
            TEST_MALLOC_ARRAY_OLDGEN_FN
        );
        assert_eq!(
            call.arg(3).to_opref(),
            len_ref,
            "the length operand is threaded through as the helper's num_elem"
        );
    }

    /// `GcRewriterAssembler.round_up_for_allocation` raises to
    /// `minimal_size_in_nursery` before aligning. Without the floor the JIT's inline bump
    /// and `Nursery::alloc`'s bump disagree for a sub-minimum object, and the
    /// next allocation lands inside the forwarding word of the previous one.
    #[test]
    fn a_headerless_new_below_the_nursery_minimum_bumps_by_the_minimum() {
        let rw = make_rewriter();
        let payload_size = std::mem::size_of::<usize>();
        assert!(payload_size < crate::header::GcHeader::MIN_NURSERY_OBJ_SIZE);

        let ops = vec![Op::with_descr(
            OpCode::New,
            &[],
            headerless_size_descr(payload_size, 78),
        )];

        let (result, _constants, _gcrefs) =
            rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        let malloc = result
            .iter()
            .find(|o| o.opcode == OpCode::CallMallocNurseryHeaderless)
            .expect("a one-word headerless struct fits the nursery");
        assert_eq!(
            malloc
                .arg(0)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            crate::header::GcHeader::MIN_NURSERY_OBJ_SIZE as i64
        );
    }

    #[test]
    fn test_headerless_new_declines_large_nursery_alloc() {
        let rw = make_rewriter();
        let payload_size = 8192;
        let type_id = 77;
        let ops = vec![Op::with_descr(
            OpCode::New,
            &[],
            headerless_size_descr(payload_size, type_id),
        )];

        let (result, _constants, _gcrefs) =
            rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::CallR);
        assert_eq!(result[1].opcode, OpCode::CheckMemoryError);
        assert_eq!(
            result[0]
                .arg(0)
                .to_opref()
                .inline_const_bits()
                .expect("malloc_big_fixedsize fn const"),
            TEST_MALLOC_BIG_FIXEDSIZE_FN
        );
        assert_eq!(
            result[0]
                .arg(1)
                .to_opref()
                .inline_const_bits()
                .expect("headered total size const"),
            round_up(payload_size + crate::header::GcHeader::SIZE) as i64
        );
        assert_eq!(
            result[0]
                .arg(2)
                .to_opref()
                .inline_const_bits()
                .expect("type id const"),
            type_id as i64
        );
    }

    // ── Test 2: NEW_ARRAY → CALL_MALLOC_NURSERY_VARSIZE ──

    #[test]
    fn test_new_array_rewrite() {
        let rw = make_rewriter();
        let length_ref = OpRef::int_op(100); // some prior op producing the length
        let ops = vec![Op::with_descr(
            OpCode::NewArray,
            &[ro(length_ref)],
            array_descr_int(),
        )];

        let result = rw.rewrite_ops(&ops);

        // Expect: CallMallocNurseryVarsize
        assert!(
            result
                .iter()
                .any(|o| o.opcode == OpCode::CallMallocNurseryVarsize)
        );
        let varsize = result
            .iter()
            .find(|o| o.opcode == OpCode::CallMallocNurseryVarsize)
            .unwrap();
        // rewrite.py:858: [ConstInt(kind), ConstInt(itemsize), v_length]
        assert_eq!(varsize.arg(2).to_opref(), length_ref);
    }

    /// Headerless nurseries use `CallMallocNurseryVarsizeHeaderless`:
    /// the same length/itemsize args, and no `GcHeader` in the bump.
    #[test]
    fn test_headerless_new_array_uses_headerless_varsize() {
        let mut rw = make_rewriter();
        rw.headerless_fixedsize = true;
        let length_ref = OpRef::int_op(100);
        let ops = vec![Op::with_descr(
            OpCode::NewArray,
            &[ro(length_ref)],
            array_descr_int(),
        )];
        let result = rw.rewrite_ops(&ops);
        assert!(
            result
                .iter()
                .any(|o| o.opcode == OpCode::CallMallocNurseryVarsizeHeaderless),
            "headerless NEW_ARRAY must emit CallMallocNurseryVarsizeHeaderless, got {result:?}"
        );
        assert!(
            !result
                .iter()
                .any(|o| o.opcode == OpCode::CallMallocNurseryVarsize)
        );
    }

    #[test]
    fn nursery_size_boundary_is_exclusive() {
        let rw = make_rewriter();
        assert!(rw.can_use_nursery(rw.max_nursery_size - 1));
        assert!(!rw.can_use_nursery(rw.max_nursery_size));
    }

    /// Constant-length oversized arrays: rewrite.py:573-584 routes these
    /// through `gen_malloc_array`, not CALL_MALLOC_NURSERY_VARSIZE.
    /// Verify pyre now emits CALL_R(malloc_array_fn, ...) plus
    /// CHECK_MEMORY_ERROR, with the typed slow helper receiving the
    /// descriptor's type id directly.
    #[test]
    fn test_new_array_const_oversize_uses_malloc_array_helper() {
        let rw = make_rewriter(); // max_nursery_size = 4096
        let len_ref = OpRef::int_op(10_000);
        // array_descr_ref: base_size=8, item_size=8 →
        //   total = 8 + 8*512 = 4104; gen_malloc_nursery sees
        //   round_up(GcHeader::SIZE + 4104) = 4112 > 4096 → returns None.
        let mut constants: ConstMap<Const> = ConstMap::default();
        constants.insert(10_000, Const::Int(512));
        let new_array = Op::with_descr(OpCode::NewArray, &[ro(len_ref)], array_descr_ref());
        new_array.pos().set(OpRef::ref_op(0));
        let ops = vec![new_array, Op::new(OpCode::Finish, &[])];

        let (result, consts, _gcrefs) = rw.rewrite_ops_with_constants(&ops, &constants);

        assert!(
            !result
                .iter()
                .any(|o| o.opcode == OpCode::CallMallocNurseryVarsize),
            "constant-length oversize must not fall back to \
             CALL_MALLOC_NURSERY_VARSIZE anymore"
        );
        let call_idx = result
            .iter()
            .position(|o| o.opcode == OpCode::CallR)
            .expect("constant-length oversize must emit CALL_R slow helper");
        let call = &result[call_idx];
        assert_eq!(
            call.arg(0)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            TEST_MALLOC_ARRAY_FN
        );
        assert_eq!(
            call.arg(1)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            8
        );
        assert_eq!(
            call.arg(2)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            5
        );
        assert_eq!(call.arg(3).to_opref(), len_ref);
        assert!(
            result
                .get(call_idx + 1)
                .is_some_and(|o| o.opcode == OpCode::CheckMemoryError),
            "CALL_R slow helper must be followed by CHECK_MEMORY_ERROR"
        );
    }

    // ── Test 3: SETFIELD_GC with Ref value → write barrier inserted ──

    #[test]
    fn test_setfield_gc_ref_needs_wb() {
        // rewrite.py + 401-404: transform_to_gc_load forwards
        // SETFIELD_GC to GC_STORE; the write-barrier arm emits WB then
        // emit_maybe_forwarded follows the forward.
        let rw = make_rewriter();
        let obj = OpRef::ref_op(0);
        let val = OpRef::ref_op(1);
        let ops = vec![Op::with_descr(
            OpCode::SetfieldGc,
            &[ro(obj), ro(val)],
            ref_field_descr(),
        )];

        let result = rw.rewrite_ops(&ops);

        // Expect: CondCallGcWb(obj), GcStore(obj, 0, val, itemsize)
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::CondCallGcWb);
        assert_eq!(result[0].arg(0).to_opref(), obj);
        assert_eq!(result[1].opcode, OpCode::GcStore);
    }

    #[test]
    fn test_no_write_barrier_descr_emits_no_barrier_but_keeps_the_store() {
        // rewrite.py:393 `if self.gc_ll_descr.write_barrier_descr is not
        // None:` — a collector that needs no write barrier reports `None`
        // (`gc.py GcLLDescr_boehm.write_barrier_descr`), and the whole
        // barrier section is skipped for it.  rewrite.py:405-412's `else`
        // still lowers the store; only the barrier goes away.
        //
        // The backend asserts the same agreement from its side
        // (`emit_write_barrier_fastpath_for_base` expects a descriptor
        // because `COND_CALL_GC_WB` only exists if the rewriter emitted
        // one), so emitting a barrier here would panic at assembly time.
        let mut rw = make_rewriter();
        rw.wb_descr = None;
        let obj = OpRef::ref_op(0);
        let val = OpRef::ref_op(1);
        let ops = vec![Op::with_descr(
            OpCode::SetfieldGc,
            &[ro(obj), ro(val)],
            ref_field_descr(),
        )];

        let result = rw.rewrite_ops(&ops);

        assert!(
            !result
                .iter()
                .any(|o| matches!(o.opcode, OpCode::CondCallGcWb | OpCode::CondCallGcWbArray)),
            "no barrier may be emitted without a write barrier descriptor"
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GcStore);
    }

    #[test]
    fn test_no_write_barrier_descr_still_lowers_setarrayitem() {
        // rewrite.py — the `else` arm calls `consider_setarrayitem_gc`
        // and leaves the store to the ordinary lowering.
        let mut rw = make_rewriter();
        rw.wb_descr = None;
        let arr = OpRef::ref_op(0);
        let idx = OpRef::int_op(1);
        let val = OpRef::ref_op(2);
        let ops = vec![Op::with_descr(
            OpCode::SetarrayitemGc,
            &[ro(arr), ro(idx), ro(val)],
            array_descr_ref(),
        )];

        let result = rw.rewrite_ops(&ops);

        assert!(
            !result
                .iter()
                .any(|o| matches!(o.opcode, OpCode::CondCallGcWb | OpCode::CondCallGcWbArray)),
            "no barrier may be emitted without a write barrier descriptor"
        );
        assert!(
            result
                .iter()
                .any(|o| matches!(o.opcode, OpCode::GcStore | OpCode::GcStoreIndexed)),
            "the store itself must still be lowered: {result:?}"
        );
    }

    #[test]
    fn test_getarrayitem_raw_r_not_lowered() {
        // rewrite.py:216-218 only pulls GETARRAYITEM_RAW_I and
        // GETARRAYITEM_RAW_F into the lowering arm; GETARRAYITEM_RAW_R
        // is intentionally missing because `jtransform.py:775`
        // (`getarrayitem_raw_r not supported`) rejects raw ref array
        // reads earlier at codewriter time.  If one somehow reaches
        // the rewriter here it must pass through — otherwise we would
        // be enabling a code path upstream explicitly disallows.
        let rw = make_rewriter();
        let obj = OpRef::ref_op(0);
        let idx = OpRef::int_op(1);
        let ops = vec![Op::with_descr(
            OpCode::GetarrayitemRawR,
            &[ro(obj), ro(idx)],
            array_descr_ref(),
        )];

        let result = rw.rewrite_ops(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GetarrayitemRawR);
        assert!(
            !result
                .iter()
                .any(|op| matches!(op.opcode, OpCode::GcLoadIndexedR | OpCode::GcLoadR)),
            "GETARRAYITEM_RAW_R must not be lowered to GC_LOAD_INDEXED_R / GC_LOAD_R"
        );
    }

    // ── Test 4: SETFIELD_GC with Int value → no write barrier ──

    #[test]
    fn test_setfield_gc_int_no_wb() {
        let rw = make_rewriter();
        let obj = OpRef::ref_op(0);
        let val = OpRef::ref_op(1);
        let ops = vec![Op::with_descr(
            OpCode::SetfieldGc,
            &[ro(obj), ro(val)],
            int_field_descr(),
        )];

        let result = rw.rewrite_ops(&ops);

        // Only the lowered GC_STORE — no WB for non-ref fields.
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GcStore);
    }

    // ── Tests 4a-c: delayed_zero_setfields (rewrite.py, 761-766) ──

    fn ref_field_descr_at(offset: usize) -> Arc<dyn FieldDescr> {
        Arc::new(TestFieldDescr {
            offset,
            field_size: 8,
            field_type: Type::Ref,
        })
    }

    fn ref_field_descr_ref_at(offset: usize) -> DescrRef {
        Arc::new(TestFieldDescr {
            offset,
            field_size: 8,
            field_type: Type::Ref,
        })
    }

    fn w_class_field_descr_at(offset: usize) -> Arc<dyn FieldDescr> {
        Arc::new(TestWClassFieldDescr { offset })
    }

    /// rewrite.py + rewrite.py — malloc_zero_filled=true
    /// short-circuits `clear_gc_fields`, so NEW emits no pending NULL
    /// stores at the next flush point.  Mirrors pyre's production
    /// nursery (which `alloc_zeroed`s).
    #[test]
    fn test_clear_gc_fields_zero_filled_skips() {
        let rw = make_rewriter(); // malloc_zero_filled = true
        let gc_fields = vec![ref_field_descr_at(24), ref_field_descr_at(32)];
        let descr = size_descr_with_gc_fields(48, 42, gc_fields);
        let ops = vec![
            Op::with_descr(OpCode::New, &[], descr),
            Op::new(OpCode::Jump, &[]),
        ];

        let (result, _consts, _gcrefs) = rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        // Allocation header stores only (CallMallocNursery + tid GcStore) + Jump.
        // No delayed-zero NULL-pointer stores must be emitted because
        // the allocator already zero-fills.
        let gc_stores: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GcStore)
            .collect();
        assert_eq!(
            gc_stores.len(),
            1,
            "malloc_zero_filled=true must emit only the tid init store, got {:?}",
            result
        );
    }

    /// rewrite.py + rewrite.py — when the allocator does
    /// not zero-fill, every GC field's byte offset is remembered and
    /// flushed as `GC_STORE(ptr, ofs, 0, 8)` at the next can-collect /
    /// flush point.
    #[test]
    fn test_emit_pending_zeros_flushes_delayed_setfields() {
        let mut rw = make_rewriter();
        rw.malloc_zero_filled = false;
        let gc_fields = vec![ref_field_descr_at(24), ref_field_descr_at(32)];
        let descr = size_descr_with_gc_fields(48, 42, gc_fields);
        let ops = vec![
            Op::with_descr(OpCode::New, &[], descr),
            Op::new(OpCode::Jump, &[]),
        ];

        let (result, consts, _gcrefs) = rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        // Collect the NULL-pointer stores emitted by the pending-zero flush.
        let mut seen_offsets: Vec<i64> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GcStore)
            // skip the tid header store (ofs=0, value=type_id, itemsize=4).
            .filter(|o| o.arg(2).to_opref().inline_const_bits() == Some(0))
            .map(|o| {
                o.arg(1)
                    .to_opref()
                    .inline_const_bits()
                    .expect("inline ConstInt")
            })
            .collect();
        seen_offsets.sort();
        assert_eq!(
            seen_offsets,
            vec![24, 32],
            "pending-zero flush must emit one NULL store per zero-init GC field"
        );
    }

    /// rewrite.py:506-512 — an explicit SETFIELD_GC at offset `ofs`
    /// removes `ofs` from `_delayed_zero_setfields`, so the flush does
    /// not re-zero the slot.
    #[test]
    fn test_consider_setfield_gc_drops_overwritten_offset() {
        let _const_ptr_table = const_ptr_table_test_guard();
        let mut rw = make_rewriter();
        rw.malloc_zero_filled = false;
        let gc_fields = vec![ref_field_descr_at(24), ref_field_descr_at(32)];
        let descr = size_descr_with_gc_fields(48, 42, gc_fields);
        let val = OpRef::const_ptr(majit_ir::GcRef(0x1234));
        // The SETFIELD_GC names the allocation by position, so the allocation
        // has to carry that position rather than rely on the rewriter minting
        // it there.
        let new_op = Op::with_descr(OpCode::New, &[], descr);
        new_op.pos().set(OpRef::ref_op(0));
        let ops = vec![
            new_op,
            Op::with_descr(
                OpCode::SetfieldGc,
                &[ro(OpRef::ref_op(0)), ro(val)],
                ref_field_descr_ref_at(24),
            ),
            Op::new(OpCode::Jump, &[]),
        ];

        let (result, _consts, _gcrefs) = rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        let null_offsets: Vec<i64> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GcStore)
            .filter(|o| o.arg(2).to_opref().inline_const_bits() == Some(0))
            .map(|o| {
                o.arg(1)
                    .to_opref()
                    .inline_const_bits()
                    .expect("inline ConstInt")
            })
            .collect();
        assert_eq!(
            null_offsets,
            vec![32],
            "SETFIELD_GC at ofs=24 must drop the pending-zero at ofs=24; only ofs=32 remains"
        );
    }

    /// `handle_new_fixedsize` records delayed zeros for headerless NEW too:
    /// leftover `gc_fielddescrs` flush as NULL stores at the next can-collect
    /// / end-of-trace point when the allocator does not zero-fill.
    #[test]
    fn test_headerless_new_flushes_delayed_setfields() {
        let mut rw = make_rewriter();
        rw.malloc_zero_filled = false;
        let gc_fields = vec![ref_field_descr_at(0), ref_field_descr_at(8)];
        let descr = headerless_size_descr_with_gc_fields(16, 78, gc_fields);
        let ops = vec![
            Op::with_descr(OpCode::New, &[], descr),
            Op::new(OpCode::Jump, &[]),
        ];

        let (result, _consts, _gcrefs) = rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        assert!(
            result
                .iter()
                .any(|o| o.opcode == OpCode::CallMallocNurseryHeaderless),
            "headerless NEW must stay on CallMallocNurseryHeaderless, got {result:?}"
        );
        let mut seen_offsets: Vec<i64> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GcStore)
            .filter(|o| o.arg(2).to_opref().inline_const_bits() == Some(0))
            .map(|o| {
                o.arg(1)
                    .to_opref()
                    .inline_const_bits()
                    .expect("inline ConstInt")
            })
            .collect();
        seen_offsets.sort();
        assert_eq!(
            seen_offsets,
            vec![0, 8],
            "headerless pending-zero flush must emit one NULL store per GC field, got {result:?}"
        );
    }

    /// An explicit SETFIELD_GC still cancels the matching delayed zero on a
    /// headerless NEW, so the flush does not re-zero a field the caller wrote.
    #[test]
    fn test_headerless_consider_setfield_gc_drops_overwritten_offset() {
        let _const_ptr_table = const_ptr_table_test_guard();
        let mut rw = make_rewriter();
        rw.malloc_zero_filled = false;
        let gc_fields = vec![ref_field_descr_at(0), ref_field_descr_at(8)];
        let descr = headerless_size_descr_with_gc_fields(16, 78, gc_fields);
        let val = OpRef::const_ptr(majit_ir::GcRef(0x1234));
        let new_op = Op::with_descr(OpCode::New, &[], descr);
        new_op.pos().set(OpRef::ref_op(0));
        let ops = vec![
            new_op,
            Op::with_descr(
                OpCode::SetfieldGc,
                &[ro(OpRef::ref_op(0)), ro(val)],
                ref_field_descr_ref_at(0),
            ),
            Op::new(OpCode::Jump, &[]),
        ];

        let (result, _consts, _gcrefs) = rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        let null_offsets: Vec<i64> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GcStore)
            .filter(|o| o.arg(2).to_opref().inline_const_bits() == Some(0))
            .map(|o| {
                o.arg(1)
                    .to_opref()
                    .inline_const_bits()
                    .expect("inline ConstInt")
            })
            .collect();
        assert_eq!(
            null_offsets,
            vec![8],
            "headerless SETFIELD_GC at ofs=0 must drop that pending-zero; only ofs=8 remains"
        );
    }

    // ── Test 5: Non-GC ops pass through unchanged ──

    #[test]
    fn test_passthrough() {
        let rw = make_rewriter();
        let ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[ro(OpRef::int_op(0)), ro(OpRef::int_op(1))],
            ),
            Op::new(OpCode::GuardTrue, &[ro(OpRef::int_op(2))]),
            Op::new(OpCode::Jump, &[]),
        ];

        let result = rw.rewrite_ops(&ops);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::GuardTrue);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    #[test]
    fn test_guard_reuses_recorded_rc_when_failargs_are_already_resolved() {
        let rw = make_rewriter();
        let guard = Op::new(OpCode::GuardTrue, &[ro(OpRef::int_op(2))]);
        guard.store_final_boxes(vec![ro(OpRef::int_op(0))]);
        let boxed: Vec<OpRc> = vec![
            OpRc::new(Op::new(
                OpCode::IntAdd,
                &[ro(OpRef::int_op(0)), ro(OpRef::int_op(1))],
            )),
            OpRc::new(guard),
            OpRc::new(Op::new(OpCode::Jump, &[])),
        ];
        let recorded = boxed[1].clone();
        let result = GcRewriter::rewrite_for_gc(&rw, &boxed);
        let out = result
            .iter()
            .find(|o| o.opcode == OpCode::GuardTrue)
            .expect("guard survives rewrite");
        assert!(
            OpRc::ptr_eq(out, &recorded),
            "emit_op must not mint a second ResOperation when failargs are unchanged"
        );
    }

    // ── Test 6: Multiple allocations are batched ──

    #[test]
    fn test_batched_allocations() {
        let rw = make_rewriter();
        let ops = vec![
            Op::with_descr(OpCode::New, &[], size_descr(24, 1)),
            Op::with_descr(OpCode::New, &[], size_descr(32, 2)),
        ];

        let (result, constants, _gcrefs) =
            rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        assert!(result.iter().any(|o| o.opcode == OpCode::CallMallocNursery));
        assert!(
            result
                .iter()
                .any(|o| o.opcode == OpCode::NurseryPtrIncrement)
        );

        let malloc = result
            .iter()
            .find(|o| o.opcode == OpCode::CallMallocNursery)
            .unwrap();
        // rewrite.py:893-895: combined size = round_up(24+8) + round_up(32+8) = 32 + 40 = 72
        let header = crate::header::GcHeader::SIZE;
        let expected_size = round_up(24 + header) as i64 + round_up(32 + header) as i64;
        assert_eq!(
            malloc
                .arg(0)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            expected_size
        );

        let incr = result
            .iter()
            .find(|o| o.opcode == OpCode::NurseryPtrIncrement)
            .unwrap();
        // rewrite.py:898: ConstInt(previous_size) = round_up(24 + GcHeader::SIZE) = 32
        assert_eq!(
            incr.arg(1)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            round_up(24 + header) as i64
        );

        // Both have tid initialisation, one Signed `HDR.tid` store each: the
        // store clears the flags half too, so the interior allocation needs
        // no separate header clear.
        let tid_stores: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GcStore)
            .collect();
        assert_eq!(tid_stores.len(), 2);
        let tid_ofs = -(crate::header::GcHeader::SIZE as i64);
        for (store, type_id) in tid_stores.iter().zip([1, 2]) {
            let arg = |i: usize| {
                store
                    .arg(i)
                    .to_opref()
                    .inline_const_bits()
                    .expect("inline ConstInt")
            };
            assert_eq!(arg(1), tid_ofs);
            assert_eq!(arg(2), type_id);
            assert_eq!(arg(3), std::mem::size_of::<usize>() as i64);
        }
    }

    // ── Test 7: A collecting operation between two NEWs prevents batching ──

    #[test]
    fn test_call_breaks_batch() {
        let rw = make_rewriter();
        let ops = vec![
            Op::with_descr(OpCode::New, &[], size_descr(24, 1)),
            Op::new(OpCode::CallN, &[ro(OpRef::ref_op(99))]),
            Op::with_descr(OpCode::New, &[], size_descr(24, 2)),
        ];

        let result = rw.rewrite_ops(&ops);

        // There should be two separate CallMallocNursery ops
        // (the CallN in between flushes the batch).
        let malloc_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::CallMallocNursery)
            .count();
        assert_eq!(malloc_count, 2);
    }

    /// `handle_call_assembler` runs `gen_malloc_nursery_varsize_frame`, which
    /// calls `emitting_an_operation_that_can_collect` before the call: the
    /// first NEW's pending NULL store flushes, and the second NEW is a fresh
    /// `CallMallocNursery`. The call itself becomes one frame argument.
    #[test]
    fn test_call_assembler_flushes_pending_and_breaks_batch() {
        #[derive(Debug)]
        struct TestLoopToken(u64);
        impl majit_ir::Descr for TestLoopToken {
            fn as_loop_token_descr(&self) -> Option<&dyn majit_ir::LoopTokenDescr> {
                Some(self)
            }
        }
        impl majit_ir::LoopTokenDescr for TestLoopToken {
            fn loop_token_number(&self) -> i64 {
                self.0 as i64
            }

            fn token_handle_any(&self) -> Option<&dyn std::any::Any> {
                Some(self)
            }
        }

        let token = Arc::new(TestLoopToken(9));
        let mut rw = make_rewriter();
        rw.malloc_zero_filled = false;
        rw.jitframe_info = Some(JitFrameDescrs {
            jitframe_tid: 7,
            jitframe_fixed_size: 56,
            jf_frame_info_ofs: 0,
            jf_descr_ofs: 8,
            jf_force_descr_ofs: 16,
            jf_savedata_ofs: 24,
            jf_guard_exc_ofs: 32,
            jf_forward_ofs: 40,
            jf_frame_ofs: 56,
            jf_frame_baseitemofs: 64,
            jf_frame_lengthofs: 56,
            sign_size: 8,
            jf_frame_itemsize: 8,
        });
        rw.call_assembler_callee_locs = Some({
            let expected = Arc::clone(&token);
            Box::new(move |ltd| {
                ltd.token_handle_any()
                    .and_then(|any| any.downcast_ref::<TestLoopToken>())
                    .filter(|got| std::ptr::eq(*got, Arc::as_ptr(&expected)))
                    .map(|_| CallAssemblerCalleeLocs {
                        _ll_initial_locs: Vec::new(),
                        frame_depth: 4,
                        frame_info_ptr: 0x1000,
                        index_of_virtualizable: -1,
                    })
            })
        });
        let gc_fields = vec![ref_field_descr_at(24)];
        let descr = size_descr_with_gc_fields(48, 42, gc_fields);
        let call = Op::with_descr(OpCode::CallAssemblerN, &[], Arc::clone(&token) as DescrRef);
        let ops = vec![
            Op::with_descr(OpCode::New, &[], descr),
            call,
            Op::with_descr(OpCode::New, &[], size_descr(24, 2)),
        ];

        let result = rw.rewrite_ops(&ops);

        let frame_idx = result
            .iter()
            .position(|o| o.opcode == OpCode::CallMallocNurseryVarsizeFrame)
            .expect("handle_call_assembler allocates the callee frame");
        let call_idx = result
            .iter()
            .position(|o| o.opcode == OpCode::CallAssemblerN)
            .expect("rewritten CALL_ASSEMBLER");
        assert!(frame_idx < call_idx);
        assert_eq!(result[call_idx].num_args(), 1);
        // A frame at or above `large_object` is a young raw-malloced object
        // whose `YOUNG_RAWMALLOC` flag lives in the upper half of `HDR.tid`,
        // so the frame's tid stamp covers the type-id half only.
        let tid_store = &result[frame_idx + 1];
        assert_eq!(tid_store.opcode, OpCode::GcStore);
        assert_eq!(tid_store.arg(2).to_opref().inline_const_bits(), Some(7));
        assert_eq!(
            tid_store.arg(3).to_opref().inline_const_bits(),
            Some((crate::header::TYPE_ID_BITS / 8) as i64)
        );
        let null_before: Vec<i64> = result[..frame_idx]
            .iter()
            .filter(|o| o.opcode == OpCode::GcStore)
            .filter(|o| o.arg(2).to_opref().inline_const_bits() == Some(0))
            .map(|o| {
                o.arg(1)
                    .to_opref()
                    .inline_const_bits()
                    .expect("inline ConstInt")
            })
            .collect();
        assert!(
            null_before.contains(&24),
            "pending NULL GcStore for the first New must flush before the frame malloc, got {null_before:?} in {result:?}"
        );
        let malloc_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::CallMallocNursery)
            .count();
        assert_eq!(
            malloc_count, 2,
            "second New must be a fresh CallMallocNursery, got {result:?}"
        );
    }

    /// `getarraydescr_for_frame` answers the `jf_frame` item width, not the
    /// Signed width.  On wasm32 a Signed is 4 bytes but the callee reads
    /// every input as an i64 slot, so a 4-byte store left the upper half of
    /// a recycled nursery slot in the callee's frame argument.
    #[test]
    fn test_call_assembler_stores_args_at_frame_item_width() {
        #[derive(Debug)]
        struct TestLoopToken(u64);
        impl majit_ir::Descr for TestLoopToken {
            fn as_loop_token_descr(&self) -> Option<&dyn majit_ir::LoopTokenDescr> {
                Some(self)
            }
        }
        impl majit_ir::LoopTokenDescr for TestLoopToken {
            fn loop_token_number(&self) -> i64 {
                self.0 as i64
            }

            fn token_handle_any(&self) -> Option<&dyn std::any::Any> {
                Some(self)
            }
        }

        let token = Arc::new(TestLoopToken(9));
        let mut rw = make_rewriter();
        rw.jitframe_info = Some(JitFrameDescrs {
            jitframe_tid: 7,
            jitframe_fixed_size: 28,
            jf_frame_info_ofs: 0,
            jf_descr_ofs: 4,
            jf_force_descr_ofs: 8,
            jf_savedata_ofs: 12,
            jf_guard_exc_ofs: 16,
            jf_forward_ofs: 20,
            jf_frame_ofs: 28,
            jf_frame_baseitemofs: 32,
            jf_frame_lengthofs: 28,
            sign_size: 4,
            jf_frame_itemsize: 8,
        });
        rw.call_assembler_callee_locs = Some({
            let expected = Arc::clone(&token);
            Box::new(move |ltd| {
                ltd.token_handle_any()
                    .and_then(|any| any.downcast_ref::<TestLoopToken>())
                    .filter(|got| std::ptr::eq(*got, Arc::as_ptr(&expected)))
                    .map(|_| CallAssemblerCalleeLocs {
                        _ll_initial_locs: vec![8, 16, 24],
                        frame_depth: 4,
                        frame_info_ptr: 0x1000,
                        index_of_virtualizable: -1,
                    })
            })
        });
        let call = Op::with_descr(
            OpCode::CallAssemblerR,
            &[
                ro(OpRef::ref_op(0)),
                ro(OpRef::int_op(1)),
                ro(OpRef::float_op(2)),
            ],
            Arc::clone(&token) as DescrRef,
        );

        let result = rw.rewrite_ops(&[call]);

        let frame_idx = result
            .iter()
            .position(|o| o.opcode == OpCode::CallMallocNurseryVarsizeFrame)
            .expect("handle_call_assembler allocates the callee frame");
        let arg_stores: Vec<(i64, i64)> = result[frame_idx..]
            .iter()
            .filter(|o| o.opcode == OpCode::GcStore)
            .filter_map(|o| {
                let ofs = o.arg(1).to_opref().inline_const_bits()?;
                let size = o.arg(3).to_opref().inline_const_bits()?;
                (ofs >= 40).then_some((ofs, size))
            })
            .collect();
        assert_eq!(
            arg_stores,
            vec![(40, 8), (48, 8), (56, 8)],
            "each CALL_ASSEMBLER argument fills a whole jf_frame item, got {result:?}"
        );
    }

    // ── Test 8: WB not duplicated for same object ──

    #[test]
    fn test_wb_not_duplicated() {
        let rw = make_rewriter();
        let obj = OpRef::ref_op(0);
        let val1 = OpRef::ref_op(1);
        let val2 = OpRef::ref_op(2);
        let ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[ro(obj), ro(val1)], ref_field_descr()),
            Op::with_descr(OpCode::SetfieldGc, &[ro(obj), ro(val2)], ref_field_descr()),
        ];

        let result = rw.rewrite_ops(&ops);

        // Only one CondCallGcWb, then two lowered GC_STORE.
        let wb_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWb)
            .count();
        assert_eq!(wb_count, 1);
        assert_eq!(
            result
                .iter()
                .filter(|o| o.opcode == OpCode::GcStore)
                .count(),
            2
        );
    }

    // ── Test 9: Freshly allocated object skips WB ──

    #[test]
    fn test_fresh_alloc_skips_wb() {
        let rw = make_rewriter();
        let ops = vec![
            Op::with_descr(OpCode::New, &[], size_descr(32, 1)),
            // The freshly allocated object (pos 0) is used as the target of a store.
            // We build the SetfieldGc referencing pos=0 from the CallMallocNursery result.
        ];

        rw.rewrite_ops(&ops);

        // Now rewrite a SetfieldGc that stores a ref into the new object.
        // The allocation carries the position the store names it by.
        let new_op = Op::with_descr(OpCode::New, &[], size_descr(32, 1));
        new_op.pos().set(OpRef::ref_op(0));
        let ops2 = vec![
            new_op,
            Op::with_descr(
                OpCode::SetfieldGc,
                &[ro(OpRef::ref_op(0)), ro(OpRef::ref_op(99))], // arg(0) = pos of the alloc
                ref_field_descr(),
            ),
        ];

        let result2 = rw.rewrite_ops(&ops2);

        // The CallMallocNursery result at pos=0 is in wb_applied,
        // so the SetfieldGc at arg(0)=OpRef::ref_op(0) should NOT get a WB.
        // Expected: CallMallocNursery, GcStore(tid), SetfieldGc
        // No CondCallGcWb because OpRef::ref_op(0) was remembered.
        let wb_count = result2
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWb)
            .count();
        assert_eq!(wb_count, 0);
    }

    #[test]
    fn segmented_rewrite_does_not_carry_write_barrier_state() {
        let _const_ptr_table = const_ptr_table_test_guard();
        // Segment 1 allocates. Segment 2 stores through that result the way
        // a spliced fail arg names it. `GcRewriterAssembler.rewrite` runs
        // once per bridge, so the store still gets a barrier. One trace
        // would see the allocation's `wb_applied` and elide it.
        let rw = make_rewriter();
        let new_op = Op::with_descr(OpCode::New, &[], size_descr(32, 1));
        new_op.pos().set(OpRef::ref_op(5));
        let store = Op::with_descr(
            OpCode::SetfieldGc,
            &[
                ro(OpRef::ref_op(5)),
                Operand::const_from_value(Value::Ref(GcRef(0xABC))),
            ],
            ref_field_descr(),
        );
        let seg1: Vec<OpRc> = vec![OpRc::new(new_op.clone())];
        let seg2: Vec<OpRc> = vec![OpRc::new(store.clone())];
        let (parts, _gcrefs) = rw
            .rewrite_for_gc_segments(&[&seg1, &seg2], &ConstMap::default())
            .expect("segmented rewrite");
        assert!(
            parts[1].iter().any(|op| op.opcode == OpCode::CondCallGcWb),
            "segment 2 must barrier a store into an object allocated in segment 1"
        );
        let combined = vec![OpRc::new(new_op), OpRc::new(store)];
        let (flat, _) = rw.rewrite_for_gc_with_constants(&combined, &ConstMap::default());
        assert!(
            !flat.iter().any(|op| op.opcode == OpCode::CondCallGcWb),
            "one trace elides the barrier because wb_applied survives the allocation"
        );
        let minted = |ops: &[OpRc]| -> Vec<u32> {
            ops.iter()
                .filter_map(|op| {
                    let pos = op.pos().get();
                    (!pos.is_none() && !pos.is_constant() && pos.ty() != Some(Type::Void))
                        .then(|| pos.raw())
                })
                .collect()
        };
        let a = minted(&parts[0]);
        let b = minted(&parts[1]);
        assert!(
            b.iter().any(|pos| !a.contains(pos)),
            "segment 2 must mint a position of its own, got {a:?} and {b:?}"
        );
        assert!(
            a.iter().all(|pos| !b.contains(pos)),
            "minted positions must be disjoint, got {a:?} and {b:?}"
        );
    }

    // ── Test 10: NEW_WITH_VTABLE also writes vtable ──

    #[test]
    fn test_new_with_vtable() {
        let rw = make_rewriter();
        let ops = vec![Op::with_descr(
            OpCode::NewWithVtable,
            &[],
            vtable_descr(48, 3, 0xDEAD),
        )];

        let (result, constants, _gcrefs) =
            rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        // CallMallocNursery + GcStore(tid) + GcStore(vtable)
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::CallMallocNursery);
        assert_eq!(result[1].opcode, OpCode::GcStore);
        let tid_ref = result[1].arg(2);
        assert_eq!(
            tid_ref
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            3
        );
        assert_eq!(result[2].opcode, OpCode::GcStore);
        let vtable_ref = result[2].arg(2);
        assert_eq!(
            vtable_ref
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            0xDEAD_i64
        );
    }

    #[test]
    fn test_new_with_vtable_initializes_w_class_without_vtable_store() {
        let _const_ptr_table = const_ptr_table_test_guard();
        let mut rw = make_rewriter();
        rw.fielddescr_vtable = None;
        rw.malloc_zero_filled = false;
        let w_class = 0xCAFE;
        let ops = vec![
            Op::with_descr(
                OpCode::NewWithVtable,
                &[],
                size_descr_with_w_class(48, 3, 0, Some(w_class), vec![w_class_field_descr_at(16)]),
            ),
            Op::new(OpCode::Jump, &[]),
        ];

        let (result, _constants, gcrefs) =
            rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        assert_eq!(gcrefs, vec![GcRef(w_class as usize)]);
        assert!(
            result.iter().any(|o| o.opcode == OpCode::LoadFromGcTable),
            "non-null w_class constant must be routed through the GC table"
        );
        let w_class_stores: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GcStore)
            .filter(|o| o.arg(1).to_opref().inline_const_bits() == Some(16))
            .collect();
        assert_eq!(
            w_class_stores.len(),
            1,
            "w_class must be initialized once and not followed by a delayed zero store: {result:?}"
        );
        assert_ne!(
            w_class_stores[0].arg(2).to_opref().inline_const_bits(),
            Some(0),
            "w_class initialization must not be a NULL delayed-zero store"
        );
    }

    #[test]
    fn test_new_with_vtable_eagerly_initializes_w_class_without_trace_store() {
        let _const_ptr_table = const_ptr_table_test_guard();
        let mut rw = make_rewriter();
        rw.fielddescr_vtable = None;
        let w_class = 0xD00D;
        let ops = vec![Op::with_descr(
            OpCode::NewWithVtable,
            &[],
            size_descr_with_w_class(48, 3, 0, Some(w_class), vec![w_class_field_descr_at(8)]),
        )];

        let (result, _constants, gcrefs) =
            rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        assert_eq!(gcrefs, vec![GcRef(w_class as usize)]);
        assert_eq!(
            result
                .iter()
                .filter(|op| {
                    op.opcode == OpCode::GcStore
                        && op.arg(1).to_opref().inline_const_bits() == Some(8)
                        && op.arg(2).to_opref().inline_const_bits() != Some(0)
                })
                .count(),
            1,
            "allocation lowering must remain the sole w_class writer: {result:?}"
        );
    }

    #[test]
    fn test_new_initializes_w_class_after_clear_gc_fields() {
        let _const_ptr_table = const_ptr_table_test_guard();
        let mut rw = make_rewriter();
        rw.fielddescr_vtable = None;
        rw.malloc_zero_filled = false;
        let w_class = 0xD00D;
        let ops = vec![
            Op::with_descr(
                OpCode::New,
                &[],
                size_descr_with_w_class(48, 3, 0, Some(w_class), vec![w_class_field_descr_at(8)]),
            ),
            Op::new(OpCode::Jump, &[]),
        ];

        let (result, _constants, gcrefs) =
            rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        assert_eq!(gcrefs, vec![GcRef(w_class as usize)]);
        let w_class_stores: Vec<_> = result
            .iter()
            .filter(|op| {
                op.opcode == OpCode::GcStore && op.arg(1).to_opref().inline_const_bits() == Some(8)
            })
            .collect();
        assert_eq!(w_class_stores.len(), 1, "{result:?}");
        assert_ne!(
            w_class_stores[0].arg(2).to_opref().inline_const_bits(),
            Some(0),
            "plain NEW must receive the eager class value, not delayed NULL"
        );
    }

    #[test]
    fn test_clear_gc_fields_zeros_w_class_without_init_value() {
        let mut rw = make_rewriter();
        rw.fielddescr_vtable = None;
        rw.malloc_zero_filled = false;
        let ops = vec![
            Op::with_descr(
                OpCode::NewWithVtable,
                &[],
                size_descr_with_w_class(48, 3, 0, None, vec![w_class_field_descr_at(16)]),
            ),
            Op::new(OpCode::Jump, &[]),
        ];

        let (result, _constants, _gcrefs) =
            rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        assert!(
            result.iter().any(|o| {
                o.opcode == OpCode::GcStore
                    && o.arg(1).to_opref().inline_const_bits() == Some(16)
                    && o.arg(2).to_opref().inline_const_bits() == Some(0)
                    && o.arg(3).to_opref().inline_const_bits() == Some(8)
            }),
            "w_class field without an init value must be included in delayed-zero stores: {result:?}"
        );
    }

    // ── Test 11: SETARRAYITEM_GC with Ref — no card marking → regular WB ──

    #[test]
    fn test_setarrayitem_gc_ref_wb() {
        // make_rewriter() has jit_wb_cards_set = 0 (card marking disabled).
        // rewrite.py: without card marking, gen_write_barrier_array
        // falls back to gen_write_barrier → COND_CALL_GC_WB.
        // rewrite.py:132 + 1124-1130: non-constant index, itemsize=8 is
        // power-of-2 and not in load_supported_factors=[1] → pre-scale
        // via INT_LSHIFT before GC_STORE_INDEXED.
        //
        // Emission order is [pre-scale, WB, lowered store] because
        // `transform_to_gc_load` runs *before* the write-barrier arm
        // (rewrite.py:368-370, 401-404) — the pre-scale INT_LSHIFT is
        // emitted inline inside emit_gc_store_or_indexed while the
        // GC_STORE_INDEXED itself is forwarded and emitted only after
        // the WB via `emit_maybe_forwarded`.
        let rw = make_rewriter();
        let obj = OpRef::ref_op(0);
        let idx = OpRef::int_op(1);
        let val = OpRef::ref_op(2);
        let ops = vec![Op::with_descr(
            OpCode::SetarrayitemGc,
            &[ro(obj), ro(idx), ro(val)],
            array_descr_ref(),
        )];

        let result = rw.rewrite_ops(&ops);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::IntLshift);
        assert_eq!(result[1].opcode, OpCode::CondCallGcWb);
        assert_eq!(result[1].arg(0).to_opref(), obj);
        assert_eq!(result[2].opcode, OpCode::GcStoreIndexed);
    }

    // ── Test 12: Collecting op clears WB memoisation ──

    #[test]
    fn test_collecting_op_clears_wb() {
        let rw = make_rewriter();
        let obj = OpRef::ref_op(0);
        let val = OpRef::ref_op(1);
        let ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[ro(obj), ro(val)], ref_field_descr()),
            // This call can collect, clearing the WB set.
            Op::new(OpCode::CallN, &[ro(OpRef::ref_op(99))]),
            Op::with_descr(OpCode::SetfieldGc, &[ro(obj), ro(val)], ref_field_descr()),
        ];

        let result = rw.rewrite_ops(&ops);

        // Two CondCallGcWb — the second one is needed because the CallN cleared the set.
        let wb_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWb)
            .count();
        assert_eq!(wb_count, 2);
    }

    #[test]
    fn test_explicit_result_positions_are_preserved_through_rewrite() {
        let rw = make_rewriter();
        // `OpCode::New` produces a Ref result (resoperation.py:469
        // `RefOp` mixin), so the test mints typed `RefOp` pos rather
        // than the default `Untyped` minted by `mk_op_with_descr`.
        let new_a = Op::with_descr(OpCode::New, &[], size_descr(24, 1));
        new_a.pos().set(OpRef::ref_op(2));
        let new_b = Op::with_descr(OpCode::New, &[], size_descr(16, 2));
        new_b.pos().set(OpRef::ref_op(3));
        let ops = vec![new_a, new_b, mk_op(OpCode::Finish, &[OpRef::ref_op(3)], 4)];

        let result = rw.rewrite_ops(&ops);

        let first_alloc = result
            .iter()
            .find(|op| op.opcode == OpCode::CallMallocNursery)
            .unwrap();
        let second_alloc = result
            .iter()
            .find(|op| op.opcode == OpCode::NurseryPtrIncrement)
            .unwrap();
        let finish = result.last().unwrap();

        assert_eq!(first_alloc.pos().get(), OpRef::ref_op(2));
        assert_eq!(second_alloc.pos().get(), OpRef::ref_op(3));
        assert_eq!(finish.opcode, OpCode::Finish);
        assert_eq!(finish.arg(0).to_opref(), OpRef::ref_op(3));
        assert!(
            result
                .iter()
                .filter(|op| op.opcode == OpCode::GcStore)
                .all(|op| op.pos().get().is_none())
        );
    }

    /// A trace whose ops all carry void results has no result position to
    /// seed the fresh-box high-water mark, and then the argument scan is the
    /// only thing keeping a minted position off a live input arg.
    /// `OpRef::raw()` folds `InputArgRef(0)` and `IntOp(0)` onto one payload
    /// and the backends key their SSA values by it, so a mark left at 0 hands
    /// the pre-scale `INT_LSHIFT` (rewrite.py) the array base
    /// pointer's slot and the store loses its base.
    #[test]
    fn test_result_less_trace_reserves_input_arg_positions() {
        let rw = make_rewriter();
        let none = OpRef::NONE.raw();
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[
                    OpRef::input_arg_ref(0),
                    OpRef::input_arg_int(1),
                    OpRef::input_arg_int(2),
                ],
                none,
            ),
            // 4-byte items, and `make_rewriter` takes the llmodel.py:39
            // `load_supported_factors = (1,)` default, so the factor forces a
            // pre-scale rather than an addressing-mode scale.
            mk_op_with_descr(
                OpCode::SetarrayitemGc,
                &[
                    OpRef::input_arg_ref(0),
                    OpRef::input_arg_int(1),
                    OpRef::input_arg_int(2),
                ],
                none,
                array_descr_int(),
            ),
            mk_op(OpCode::Finish, &[], none),
        ];

        let result = rw.rewrite_ops(&ops);

        assert!(
            result.iter().any(|op| op.opcode == OpCode::IntLshift),
            "expected a pre-scale op; without one the test proves nothing"
        );
        for pos in result
            .iter()
            .map(|op| op.pos().get())
            .filter(|p| !p.is_none())
        {
            assert!(
                pos.raw() > 2,
                "minted position {} aliases input arg {}",
                pos.raw(),
                pos.raw()
            );
        }
    }

    #[test]
    fn test_rewrite_preserves_incoming_wb_and_adds_its_own() {
        // rewrite.py gen_write_barrier_array does NOT call
        // remember_wb(), so consecutive SETARRAYITEM_GC with Ref values
        // each emit their own WB. Running the rewriter over a trace that
        // already contains a COND_CALL_GC_WB_ARRAY + SETARRAYITEM pair
        // preserves the incoming WB and emits a new one for the
        // SETARRAYITEM — matching upstream rather than an idempotence
        // shortcut.
        let rw = make_rewriter();
        let once = vec![
            mk_op(
                OpCode::CondCallGcWbArray,
                &[OpRef::ref_op(5), OpRef::int_op(1)],
                7,
            ),
            mk_op_with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef::ref_op(5), OpRef::int_op(1), OpRef::ref_op(6)],
                8,
                array_descr_ref(),
            ),
        ];

        let twice = rw.rewrite_ops(&once);

        let wb_count = twice
            .iter()
            .filter(|op| op.opcode == OpCode::CondCallGcWbArray)
            .count();
        assert!(
            wb_count >= 1,
            "at least the pre-existing WB_ARRAY is preserved"
        );
        // rewrite.py:132 + 220-221: SETARRAYITEM_GC is lowered to
        // GC_STORE_INDEXED via handle_setarrayitem → emit_gc_store_or_indexed.
        assert_eq!(
            twice
                .iter()
                .filter(|op| op.opcode == OpCode::SetarrayitemGc)
                .count(),
            0
        );
        assert_eq!(
            twice
                .iter()
                .filter(|op| op.opcode == OpCode::GcStoreIndexed)
                .count(),
            1
        );
    }

    // ── could_merge_with_next_guard / remove_tested_failarg tests ──

    #[test]
    fn test_comparison_guard_true_hoists_tested_failarg() {
        // rewrite.py:431-471 parity: INT_LT followed by GUARD_TRUE(INT_LT)
        // with the comparison's result appearing in the guard's failargs.
        // The rewriter must emit SAME_AS_I(0) before the comparison and
        // rewrite the guard's failargs to reference the SAME_AS_I output.
        let rw = make_rewriter();

        let int_lt = Op::new(OpCode::IntLt, &[ro(OpRef::int_op(0)), ro(OpRef::int_op(1))]);
        int_lt.pos().set(OpRef::int_op(2));
        let guard = Op::new(OpCode::GuardTrue, &[ro(OpRef::int_op(2))]);
        guard.store_final_boxes(vec![
            ro(OpRef::int_op(0)),
            ro(OpRef::int_op(2)),
            ro(OpRef::int_op(1)),
        ]);
        let ops = vec![int_lt, guard, Op::new(OpCode::Finish, &[])];

        let result = rw.rewrite_ops(&ops);

        // Expect the rewriter to have emitted SAME_AS_I BEFORE the IntLt.
        let same_idx = result
            .iter()
            .position(|o| o.opcode == OpCode::SameAsI)
            .expect("SAME_AS_I must be emitted");
        let lt_idx = result
            .iter()
            .position(|o| o.opcode == OpCode::IntLt)
            .expect("IntLt survives");
        let guard_idx = result
            .iter()
            .position(|o| o.opcode == OpCode::GuardTrue)
            .expect("GuardTrue survives");
        assert!(
            same_idx < lt_idx,
            "SAME_AS_I must be emitted before the comparison"
        );
        assert!(lt_idx < guard_idx, "GuardTrue must follow the comparison");

        // The guard's failargs must now reference the SAME_AS_I output
        // at the position where OpRef::ref_op(2) (the IntLt result) used to appear.
        let same_pos = result[same_idx].pos().get();
        let guard_fa = result[guard_idx]
            .guard_fail_args()
            .expect("guard keeps failargs");
        assert_eq!(
            guard_fa.iter().map(|b| b.to_opref()).collect::<Vec<_>>(),
            vec![OpRef::int_op(0), same_pos, OpRef::int_op(1)],
            "OpRef::ref_op(2) → SAME_AS_I substitution"
        );
    }

    #[test]
    fn test_comparison_guard_false_hoists_with_one_constant() {
        // GUARD_FALSE: rewrite.py:463 `value = int(opnum == GUARD_FALSE)` ⇒ 1.
        let rw = make_rewriter();
        let int_eq = Op::new(OpCode::IntEq, &[ro(OpRef::int_op(0)), ro(OpRef::int_op(1))]);
        int_eq.pos().set(OpRef::int_op(2));
        let guard = Op::new(OpCode::GuardFalse, &[ro(OpRef::int_op(2))]);
        guard.store_final_boxes(vec![ro(OpRef::int_op(2))]);
        let ops = vec![int_eq, guard, Op::new(OpCode::Finish, &[])];

        let (result, consts, _gcrefs) = rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        let same = result
            .iter()
            .find(|o| o.opcode == OpCode::SameAsI)
            .expect("SAME_AS_I must be emitted for GUARD_FALSE merge");
        let const_ref = same.arg(0);
        assert_eq!(
            const_ref
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            1,
            "GUARD_FALSE hoists SAME_AS_I(1) per rewrite.py:463",
        );
    }

    #[test]
    fn test_guard_always_fails_lowers_to_same_as_guard_value() {
        // rewrite.py:419-425: GUARD_ALWAYS_FAILS ⇒ SAME_AS_I(0) +
        // GUARD_VALUE(same_as, 1). Failargs are propagated via
        // copy_and_change.
        let rw = make_rewriter();
        let guard = Op::new(OpCode::GuardAlwaysFails, &[]);
        guard.store_final_boxes(vec![ro(OpRef::int_op(10)), ro(OpRef::int_op(11))]);
        let ops = vec![guard, Op::new(OpCode::Finish, &[])];

        let (result, consts, _gcrefs) = rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        assert!(
            result.iter().all(|o| o.opcode != OpCode::GuardAlwaysFails),
            "GUARD_ALWAYS_FAILS is lowered"
        );
        let same = result
            .iter()
            .find(|o| o.opcode == OpCode::SameAsI)
            .expect("SAME_AS_I is emitted");
        let gv = result
            .iter()
            .find(|o| o.opcode == OpCode::GuardValue)
            .expect("GuardValue replaces GuardAlwaysFails");
        assert_eq!(gv.arg(0).to_opref(), same.pos().get());
        assert_eq!(
            gv.arg(1)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            1,
            "GuardValue checks against ConstInt(1)",
        );
        assert_eq!(
            same.arg(0)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            0,
            "SAME_AS_I uses ConstInt(0) per rewrite.py:421",
        );
        let gv_fa = gv.guard_fail_args().expect("GuardValue inherits failargs");
        assert_eq!(
            gv_fa.iter().map(|b| b.to_opref()).collect::<Vec<_>>(),
            vec![OpRef::int_op(10), OpRef::int_op(11)]
        );
    }

    #[test]
    fn test_comparison_guard_mismatch_is_passthrough() {
        // Guard that does NOT test the previous op's result: merge does
        // not fire, no SAME_AS_I is emitted.
        let rw = make_rewriter();
        let int_lt = Op::new(OpCode::IntLt, &[ro(OpRef::int_op(0)), ro(OpRef::int_op(1))]);
        int_lt.pos().set(OpRef::int_op(2));
        // GuardTrue reads some unrelated OpRef::ref_op(5), not OpRef::ref_op(2).
        let guard = Op::new(OpCode::GuardTrue, &[ro(OpRef::int_op(5))]);
        guard.store_final_boxes(vec![ro(OpRef::int_op(0)), ro(OpRef::int_op(1))]);
        let ops = vec![int_lt, guard, Op::new(OpCode::Finish, &[])];

        let result = rw.rewrite_ops(&ops);
        assert!(
            result.iter().all(|o| o.opcode != OpCode::SameAsI),
            "no merge → no hoisted SAME_AS_I"
        );
    }

    // ── Pending zero flush tests ──

    /// Helper: build a constants map mapping `key` → `value` for tests
    /// that need the rewriter's resolve_constant to find a length.
    fn const_pool(entries: &[(u32, i64)]) -> ConstMap<Const> {
        entries.iter().map(|&(k, v)| (k, Const::Int(v))).collect()
    }

    #[test]
    fn test_pending_zero_fully_initialized() {
        // NEW_ARRAY_CLEAR(3) + SET[0] + SET[1] + SET[2] → ZERO_ARRAY emitted
        // with length=0 (RPython rewrite.py:754 "may be ConstInt(0)").
        // rewrite.py consider_setarrayitem_gc requires the index
        // to be `ConstInt` (`index_box.is_constant()` / `getint()`); the
        // pyre equivalent is an entry in the constant pool. OpRefs 10/11/12
        // hold the literal indices 0/1/2 so `resolve_constant` returns the
        // item number.
        //
        // malloc_zero_filled=false exercises the `clear_varsize_gc_fields`
        // path (rewrite.py:521) that actually emits ZERO_ARRAY.
        let mut rw = make_rewriter();
        rw.malloc_zero_filled = false;
        let new_array = Op::with_descr(
            OpCode::NewArrayClear,
            &[ro(OpRef::int_op(3))],
            array_descr_int(),
        );
        new_array.pos().set(OpRef::ref_op(0));
        let constants = const_pool(&[(3, 3), (10, 0), (11, 1), (12, 2)]);

        let ops = vec![
            new_array,
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[
                    ro(OpRef::ref_op(0)),
                    ro(OpRef::int_op(10)),
                    ro(OpRef::int_op(100)),
                ],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[
                    ro(OpRef::ref_op(0)),
                    ro(OpRef::int_op(11)),
                    ro(OpRef::int_op(100)),
                ],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[
                    ro(OpRef::ref_op(0)),
                    ro(OpRef::int_op(12)),
                    ro(OpRef::int_op(100)),
                ],
                array_descr_int(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];

        let (result, _out_consts, _gcrefs) = rw.rewrite_ops_with_constants(&ops, &constants);

        // All indices were SET, so the in-place ZERO_ARRAY is rewritten
        // to byte_length 0 — backend treats it as a no-op.
        let zeros: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::ZeroArray)
            .collect();
        assert_eq!(zeros.len(), 1, "ZERO_ARRAY stays in place per parity");
        assert_eq!(
            zeros[0]
                .arg(2)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            0,
            "byte length must be 0"
        );
    }

    #[test]
    fn test_pending_zero_partially_initialized() {
        // NEW_ARRAY_CLEAR(4) + SET[0] + SET[1] → ZERO_ARRAY trimmed to
        // start=2 items, length=2 items → byte_start=8, byte_len=8.
        // Index OpRefs 10/11 are ConstInt 0/1 per rewrite.py:514-518.
        let mut rw = make_rewriter();
        rw.malloc_zero_filled = false;
        let new_array = Op::with_descr(
            OpCode::NewArrayClear,
            &[ro(OpRef::int_op(4))],
            array_descr_int(),
        );
        new_array.pos().set(OpRef::ref_op(0));
        let constants = const_pool(&[(4, 4), (10, 0), (11, 1)]);

        let ops = vec![
            new_array,
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[
                    ro(OpRef::ref_op(0)),
                    ro(OpRef::int_op(10)),
                    ro(OpRef::int_op(100)),
                ],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[
                    ro(OpRef::ref_op(0)),
                    ro(OpRef::int_op(11)),
                    ro(OpRef::int_op(100)),
                ],
                array_descr_int(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];

        let (result, _out_consts, _gcrefs) = rw.rewrite_ops_with_constants(&ops, &constants);

        let zeros: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::ZeroArray)
            .collect();
        assert_eq!(zeros.len(), 1, "should emit exactly one ZERO_ARRAY");
        // item_size = 4 (array_descr_int), so start_items=2 → 8 bytes,
        // length_items=2 → 8 bytes.
        assert_eq!(
            zeros[0]
                .arg(1)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            8,
            "byte start"
        );
        assert_eq!(
            zeros[0]
                .arg(2)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            8,
            "byte length"
        );
        assert_eq!(
            zeros[0]
                .arg(3)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            1,
            "scale arg(3) is 1"
        );
        assert_eq!(
            zeros[0]
                .arg(4)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            1,
            "scale arg(4) is 1"
        );
    }

    #[test]
    fn test_pending_zero_flushed_at_guard() {
        // Guard forces pending zero flush even if no indices were SET.
        let mut rw = make_rewriter();
        rw.malloc_zero_filled = false;
        let new_array = Op::with_descr(
            OpCode::NewArrayClear,
            &[ro(OpRef::int_op(3))],
            array_descr_int(),
        );
        new_array.pos().set(OpRef::ref_op(0));
        let constants = const_pool(&[(3, 3)]);

        let ops = vec![
            new_array,
            Op::new(OpCode::GuardTrue, &[ro(OpRef::int_op(50))]),
            Op::new(OpCode::Finish, &[]),
        ];

        let (result, _out_consts, _gcrefs) = rw.rewrite_ops_with_constants(&ops, &constants);

        // The ZERO_ARRAY should appear before the guard.
        let zero_idx = result.iter().position(|o| o.opcode == OpCode::ZeroArray);
        let guard_idx = result.iter().position(|o| o.opcode == OpCode::GuardTrue);
        assert!(zero_idx.is_some(), "ZERO_ARRAY should be emitted");
        assert!(guard_idx.is_some(), "GuardTrue should be present");
        assert!(
            zero_idx.unwrap() < guard_idx.unwrap(),
            "ZERO_ARRAY should come before GuardTrue"
        );

        let zero = result
            .iter()
            .find(|o| o.opcode == OpCode::ZeroArray)
            .unwrap();
        // No SETs, length=3 items × 4 bytes/item = 12 bytes.
        assert_eq!(
            zero.arg(1)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            0,
            "byte start"
        );
        assert_eq!(
            zero.arg(2)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            12,
            "byte length"
        );
    }

    #[test]
    fn test_pending_zero_gap_in_middle() {
        // NEW_ARRAY_CLEAR(5) + SET[0] + SET[2] + SET[4] — RPython does
        // trim-from-both-ends only (no middle splitting): start=1 (skip
        // index 0), stop=4 (skip index 4), length=3.  Index 2 falls
        // inside the zero range and is re-zeroed before the SET.
        // Index OpRefs 10/12/14 hold ConstInt 0/2/4 per rewrite.py:514-518.
        let mut rw = make_rewriter();
        rw.malloc_zero_filled = false;
        let new_array = Op::with_descr(
            OpCode::NewArrayClear,
            &[ro(OpRef::int_op(5))],
            array_descr_int(),
        );
        new_array.pos().set(OpRef::ref_op(0));
        let constants = const_pool(&[(5, 5), (10, 0), (12, 2), (14, 4)]);

        let ops = vec![
            new_array,
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[
                    ro(OpRef::ref_op(0)),
                    ro(OpRef::int_op(10)),
                    ro(OpRef::int_op(100)),
                ],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[
                    ro(OpRef::ref_op(0)),
                    ro(OpRef::int_op(12)),
                    ro(OpRef::int_op(100)),
                ],
                array_descr_int(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[
                    ro(OpRef::ref_op(0)),
                    ro(OpRef::int_op(14)),
                    ro(OpRef::int_op(100)),
                ],
                array_descr_int(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];

        let (result, _out_consts, _gcrefs) = rw.rewrite_ops_with_constants(&ops, &constants);

        let zeros: Vec<_> = result
            .iter()
            .filter(|o| o.opcode == OpCode::ZeroArray)
            .collect();
        assert_eq!(zeros.len(), 1, "rewrite.py:719 emits one ZERO_ARRAY");
        // start_items=1 → 4 bytes, length_items=3 → 12 bytes.
        assert_eq!(
            zeros[0]
                .arg(1)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            4,
            "byte start"
        );
        assert_eq!(
            zeros[0]
                .arg(2)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            12,
            "byte length"
        );
    }

    #[test]
    fn test_pending_zero_no_clear() {
        // Plain NEW_ARRAY (not CLEAR) should NOT produce any ZERO_ARRAY.
        let rw = make_rewriter();
        let new_array =
            Op::with_descr(OpCode::NewArray, &[ro(OpRef::int_op(3))], array_descr_int());
        new_array.pos().set(OpRef::ref_op(0));

        let ops = vec![new_array, Op::new(OpCode::Finish, &[])];

        let result = rw.rewrite_ops(&ops);

        let zero_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::ZeroArray)
            .count();
        assert_eq!(
            zero_count, 0,
            "plain NEW_ARRAY should not produce ZERO_ARRAY"
        );
    }

    // ── Test: gen_write_barrier_array LARGE threshold ──

    fn make_rewriter_with_cards() -> GcRewriterImpl {
        GcRewriterImpl {
            nursery_free_addr: 0x1000,
            nursery_top_addr: 0x2000,
            max_nursery_size: 4096,
            wb_descr: Some(WriteBarrierDescr {
                jit_wb_if_flag: 1,
                jit_wb_if_flag_byteofs: 0,
                jit_wb_if_flag_singlebyte: 1,
                jit_wb_cards_set: 0x40,
                jit_wb_card_page_shift: 7,
                jit_wb_cards_set_byteofs: 0,
                jit_wb_cards_set_singlebyte: 0x40,
            }),
            jitframe_info: None,
            call_assembler_callee_locs: None,
            load_supported_factors: &[1],
            supports_load_effective_address: true,
            malloc_zero_filled: true,
            memcpy_fn: majit_ir::memcpy_fn_addr(),
            memcpy_descr: majit_ir::make_memcpy_calldescr(),
            str_descr: str_array_descr(),
            unicode_descr: unicode_array_descr(),
            str_hash_descr: hash_field_descr(),
            unicode_hash_descr: hash_field_descr(),
            // gc.py:33-37 `fielddescr_vtable`. Test fixtures always
            // install a Some so the existing test_new_with_vtable
            // continues to exercise the typeptr stamping path.
            fielddescr_vtable: Some(majit_ir::make_vtable_field_descr()),
            // gc.py:394 `fielddescr_tid`. Test fixtures install a Some
            // (framework-GC mode) so handle_new tests exercise the tid
            // header stamping path.
            fielddescr_tid: Some(majit_ir::make_tid_field_descr()),
            malloc_array_fn: TEST_MALLOC_ARRAY_FN,
            malloc_array_nonstandard_fn: TEST_MALLOC_ARRAY_NONSTANDARD_FN,
            malloc_array_oldgen_fn: TEST_MALLOC_ARRAY_OLDGEN_FN,
            malloc_array_nonstandard_oldgen_fn: TEST_MALLOC_ARRAY_NONSTANDARD_OLDGEN_FN,
            malloc_str_fn: TEST_MALLOC_STR_FN,
            malloc_unicode_fn: TEST_MALLOC_UNICODE_FN,
            malloc_big_fixedsize_fn: TEST_MALLOC_BIG_FIXEDSIZE_FN,
            malloc_big_fixedsize_oldgen_fn: TEST_MALLOC_BIG_FIXEDSIZE_OLDGEN_FN,
            malloc_array_descr: majit_ir::make_malloc_array_calldescr(),
            malloc_array_nonstandard_descr: majit_ir::make_malloc_array_nonstandard_calldescr(),
            malloc_str_descr: majit_ir::make_malloc_str_calldescr(),
            malloc_unicode_descr: majit_ir::make_malloc_unicode_calldescr(),
            malloc_big_fixedsize_descr: majit_ir::make_malloc_big_fixedsize_calldescr(),
            standard_array_basesize: TEST_STANDARD_ARRAY_BASESIZE,
            standard_array_length_ofs: TEST_STANDARD_ARRAY_LENGTH_OFS,
            headerless_fixedsize: false,
        }
    }

    /// The `malloc_big_fixedsize` arm of rewrite.py `gen_malloc_fixedsize`
    /// ends with `remember_write_barrier(v_result)`, sound because the helper
    /// hands back a young object: it is `do_malloc_fixedsize_clear`, whose
    /// large arm is `external_malloc(typeid, 0, alloc_young=True)`.
    #[test]
    fn test_setfield_gc_after_big_fixedsize_alloc_no_wb() {
        let rw = make_rewriter();
        let new_op = Op::with_descr(OpCode::New, &[], size_descr(8192, 71));
        new_op.pos().set(OpRef::ref_op(0));
        let ops = vec![
            new_op,
            Op::with_descr(
                OpCode::SetfieldGc,
                &[ro(OpRef::ref_op(0)), ro(OpRef::ref_op(1))],
                ref_field_descr(),
            ),
        ];

        let result = rw.rewrite_ops(&ops);

        assert_eq!(
            result
                .iter()
                .filter(|o| o.opcode == OpCode::CallR)
                .filter_map(|o| o.arg(0).to_opref().inline_const_bits())
                .collect::<Vec<_>>(),
            vec![TEST_MALLOC_BIG_FIXEDSIZE_FN],
            "an oversized NEW takes the size-dispatching helper"
        );
        assert_eq!(
            result
                .iter()
                .filter(|o| o.opcode == OpCode::CondCallGcWb)
                .count(),
            0,
            "the fresh young object is wb_applied, so the store needs no barrier"
        );
    }

    /// The `non_moving` twin gets no stamp: it births in the old generation,
    /// where a young pointer stored under an elided barrier would be lost at
    /// the next minor.
    #[test]
    fn test_setfield_gc_after_non_moving_alloc_keeps_wb() {
        let rw = make_rewriter();
        let new_op = Op::with_descr(OpCode::New, &[], non_moving_size_descr(32, 72));
        new_op.pos().set(OpRef::ref_op(0));
        let ops = vec![
            new_op,
            Op::with_descr(
                OpCode::SetfieldGc,
                &[ro(OpRef::ref_op(0)), ro(OpRef::ref_op(1))],
                ref_field_descr(),
            ),
        ];

        let result = rw.rewrite_ops(&ops);

        assert_eq!(
            result
                .iter()
                .filter(|o| o.opcode == OpCode::CallR)
                .filter_map(|o| o.arg(0).to_opref().inline_const_bits())
                .collect::<Vec<_>>(),
            vec![TEST_MALLOC_BIG_FIXEDSIZE_OLDGEN_FN]
        );
        assert_eq!(
            result
                .iter()
                .filter(|o| o.opcode == OpCode::CondCallGcWb)
                .count(),
            1,
            "an old-generation birth keeps its barrier"
        );
    }

    #[test]
    fn test_setarrayitem_gc_after_const_alloc_no_wb() {
        // rewrite.py — gen_malloc_nursery's tail
        // `remember_write_barrier(op)` records the fresh nursery alloc
        // in wb_applied; rewrite.py `if not write_barrier_applied
        // (val): ...` then short-circuits the WB on the immediate
        // SETARRAYITEM_GC.  Both length=10 (< LARGE) and length=200
        // (>= LARGE) take handle_new_array path #2 (constant-size
        // nursery), so neither gen_write_barrier_array branch fires
        // (LARGE threshold logic stays gated behind path #4 fallback,
        // not yet ported in pyre).
        for &num_elem in &[10_i64, 200_i64] {
            let rw = make_rewriter_with_cards();
            let len_ref = OpRef::int_op(10_000);
            let mut constants: ConstMap<Const> = ConstMap::default();
            constants.insert(10_000, Const::Int(num_elem));
            let new_array =
                Op::with_descr(OpCode::NewArrayClear, &[ro(len_ref)], array_descr_ref());
            new_array.pos().set(OpRef::ref_op(0));
            let ops = vec![
                new_array,
                Op::with_descr(
                    OpCode::SetarrayitemGc,
                    &[
                        ro(OpRef::ref_op(0)),
                        ro(OpRef::int_op(1)),
                        ro(OpRef::ref_op(2)),
                    ],
                    array_descr_ref(),
                ),
                Op::new(OpCode::Finish, &[]),
            ];
            let (result, _, _gcrefs) = rw.rewrite_ops_with_constants(&ops, &constants);
            let wb = result
                .iter()
                .filter(|o| o.opcode == OpCode::CondCallGcWb)
                .count();
            let wb_arr = result
                .iter()
                .filter(|o| o.opcode == OpCode::CondCallGcWbArray)
                .count();
            assert_eq!(
                wb, 0,
                "fresh alloc is wb_applied → no regular WB (num_elem={num_elem})"
            );
            assert_eq!(
                wb_arr, 0,
                "fresh alloc is wb_applied → no WB_ARRAY (num_elem={num_elem})"
            );
        }
    }

    /// rewrite.py:863-866: a variable-length nursery array is NOT recorded in
    /// `write_barrier_applied`, unlike every other fresh allocation.
    ///
    /// The exemption the fixed-size path takes is sound because a small fresh
    /// object cannot have cards. This one can: its length is unknown at
    /// rewrite time, so the allocation may land on the large young arm and
    /// come back carded, and then a card is set by the barrier or not at all.
    /// The invariant lives as an *absent* `remember_write_barrier` call, which
    /// is exactly the shape a later optimizer removes by accident.
    #[test]
    fn a_varsize_nursery_array_is_not_exempt_from_the_write_barrier() {
        let rw = make_rewriter_with_cards();
        // A non-constant length: no entry in `constants`, so `resolve_constant`
        // answers None and `handle_new_array` takes the varsize nursery path.
        let len_ref = OpRef::int_op(10_000);
        let new_array = Op::with_descr(OpCode::NewArray, &[ro(len_ref)], array_descr_ref());
        new_array.pos().set(OpRef::ref_op(0));
        let ops = vec![
            new_array,
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[
                    ro(OpRef::ref_op(0)),
                    ro(OpRef::int_op(1)),
                    ro(OpRef::ref_op(2)),
                ],
                array_descr_ref(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];

        let result = rw.rewrite_ops(&ops);

        assert_eq!(
            result
                .iter()
                .filter(|o| o.opcode == OpCode::CallMallocNurseryVarsize)
                .count(),
            1,
            "the fixture must reach the varsize nursery path, or it asserts nothing"
        );
        let wb_arr = result
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWbArray)
            .count();
        assert_eq!(
            wb_arr, 1,
            "a varsize nursery array keeps its card barrier: it may be large, \
             young and carded"
        );
    }

    #[test]
    fn test_setarrayitem_gc_unknown_length_uses_wb_array() {
        // rewrite.py:962: unknown length defaults to LARGE → WB_ARRAY.
        let rw = make_rewriter_with_cards();
        // No NEW_ARRAY, so v_base has unknown length.
        let ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[
                    ro(OpRef::ref_op(0)),
                    ro(OpRef::int_op(1)),
                    ro(OpRef::ref_op(2)),
                ],
                array_descr_ref(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];
        let result = rw.rewrite_ops(&ops);
        let wb_arr = result
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWbArray)
            .count();
        assert_eq!(wb_arr, 1, "unknown length should get WB_ARRAY");
    }

    #[test]
    fn test_setarrayitem_gc_int_value_no_wb() {
        // rewrite.py:940: v.type != 'r' → no write barrier at all.
        // The stored value is an int-typed box; resoperation.py:567
        // pins `IntOp.type = 'i'`, so the write-barrier check sees
        // `val_is_ref = false`.
        let rw = make_rewriter_with_cards();
        let ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[
                    ro(OpRef::ref_op(0)),
                    ro(OpRef::int_op(1)),
                    ro(OpRef::int_op(2)),
                ],
                array_descr_int(),
            ),
            Op::new(OpCode::Finish, &[]),
        ];
        let result = rw.rewrite_ops(&ops);
        let any_wb = result
            .iter()
            .filter(|o| o.opcode == OpCode::CondCallGcWb || o.opcode == OpCode::CondCallGcWbArray)
            .count();
        assert_eq!(any_wb, 0, "int store should not produce any WB");
    }

    fn str_array_descr() -> DescrRef {
        // rstr.py:1226 `extra_item_after_alloc=True` → base_size = 17
        // (16-byte GC header surrogate + 1 trailing null), itemsize = 1.
        Arc::new(TestArrayDescr {
            base_size: 17,
            item_size: 1,
            type_id: 7,
            item_type: Type::Int,
            len_descr: Some(Arc::new(TestFieldDescr {
                offset: std::mem::size_of::<usize>(),
                field_size: std::mem::size_of::<usize>(),
                field_type: Type::Int,
            })),
            non_moving: false,
        })
    }

    fn unicode_array_descr() -> DescrRef {
        // rstr.py UNICODE has no extra_item_after_alloc; itemsize = 4.
        Arc::new(TestArrayDescr {
            base_size: 16,
            item_size: 4,
            type_id: 8,
            item_type: Type::Int,
            len_descr: Some(Arc::new(TestFieldDescr {
                offset: std::mem::size_of::<usize>(),
                field_size: std::mem::size_of::<usize>(),
                field_type: Type::Int,
            })),
            non_moving: false,
        })
    }

    /// Test stand-in for `gc_ll_descr.{str,unicode}_hash_descr`
    /// (gc.py:48-49): the `hash` field of rstr.STR / rstr.UNICODE
    /// lives at offset 0 with `Signed` size.
    fn hash_field_descr() -> DescrRef {
        Arc::new(TestFieldDescr {
            offset: 0,
            field_size: std::mem::size_of::<usize>(),
            field_type: Type::Int,
        })
    }

    // ── COPYSTRCONTENT → LEA × 2 + CALL_N(memcpy) ──
    //
    // rpython/jit/backend/llsupport/test/test_rewrite.py test_rewrite_copystrcontents
    // `test_rewrite_copystrcontents`.
    #[test]
    fn test_rewrite_copystrcontents() {
        let rw = make_rewriter();
        // [p0, p1, i0, i1, i_len]
        let p0 = OpRef::ref_op(0);
        let p1 = OpRef::ref_op(1);
        let i0 = OpRef::int_op(2);
        let i1 = OpRef::int_op(3);
        let i_len = OpRef::int_op(4);
        let ops = vec![Op::with_descr(
            OpCode::Copystrcontent,
            &[ro(p0), ro(p1), ro(i0), ro(i1), ro(i_len)],
            str_array_descr(),
        )];

        let (result, constants, _gcrefs) =
            rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        assert_eq!(
            result.len(),
            3,
            "expected LEA + LEA + CALL_N, got {:?}",
            result.iter().map(|o| o.opcode).collect::<Vec<_>>()
        );
        // i_src = load_effective_address(p0, i0, basesize-1=16, shift=0)
        assert_eq!(result[0].opcode, OpCode::LoadEffectiveAddress);
        assert_eq!(result[0].arg(0).to_opref(), p0);
        assert_eq!(result[0].arg(1).to_opref(), i0);
        assert_eq!(
            result[0]
                .arg(2)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            16
        );
        assert_eq!(
            result[0]
                .arg(3)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            0
        );
        // i_dst = load_effective_address(p1, i1, 16, 0)
        assert_eq!(result[1].opcode, OpCode::LoadEffectiveAddress);
        assert_eq!(result[1].arg(0).to_opref(), p1);
        assert_eq!(result[1].arg(1).to_opref(), i1);
        assert_eq!(
            result[1]
                .arg(2)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            16
        );
        assert_eq!(
            result[1]
                .arg(3)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            0
        );
        // call_n(memcpy_fn, i_dst, i_src, i_len)
        assert_eq!(result[2].opcode, OpCode::CallN);
        assert_eq!(result[2].arg(1).to_opref(), result[1].pos().get()); // dst
        assert_eq!(result[2].arg(2).to_opref(), result[0].pos().get()); // src
        assert_eq!(result[2].arg(3).to_opref(), i_len);
        assert!(result[2].has_descr(), "CALL_N must carry memcpy_descr");
    }

    // ── COPYUNICODECONTENT with non-constant length → LEA × 2 + LSHIFT + CALL_N ──
    #[test]
    fn test_rewrite_copyunicodecontents_dynamic_length() {
        let rw = make_rewriter();
        let p0 = OpRef::ref_op(0);
        let p1 = OpRef::ref_op(1);
        let i0 = OpRef::int_op(2);
        let i1 = OpRef::int_op(3);
        let i_len = OpRef::int_op(4);
        let ops = vec![Op::with_descr(
            OpCode::Copyunicodecontent,
            &[ro(p0), ro(p1), ro(i0), ro(i1), ro(i_len)],
            unicode_array_descr(),
        )];

        let (result, constants, _gcrefs) =
            rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        // Expect: LEA, LEA, INT_LSHIFT(i_len, 2), CALL_N
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::LoadEffectiveAddress);
        // basesize=16, shift=2 (itemsize=4 → itemscale=2)
        assert_eq!(
            result[0]
                .arg(2)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            16
        );
        assert_eq!(
            result[0]
                .arg(3)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            2
        );
        assert_eq!(result[1].opcode, OpCode::LoadEffectiveAddress);
        assert_eq!(result[2].opcode, OpCode::IntLshift);
        assert_eq!(result[2].arg(0).to_opref(), i_len);
        assert_eq!(
            result[2]
                .arg(1)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            2
        );
        assert_eq!(result[3].opcode, OpCode::CallN);
        assert_eq!(result[3].arg(3).to_opref(), result[2].pos().get());
    }

    // ── COPYSTRCONTENT without LEA → INT_ADD × 4 + CALL_N ──
    //
    // rpython/jit/backend/llsupport/test/test_rewrite.py test_rewrite_copystrcontents_without_load_effective_address
    // `test_rewrite_copystrcontents_without_load_effective_address`.
    #[test]
    fn test_rewrite_copystrcontents_without_load_effective_address() {
        let mut rw = make_rewriter();
        rw.supports_load_effective_address = false;
        let p0 = OpRef::ref_op(0);
        let p1 = OpRef::ref_op(1);
        let i0 = OpRef::int_op(2);
        let i1 = OpRef::int_op(3);
        let i_len = OpRef::int_op(4);
        let ops = vec![Op::with_descr(
            OpCode::Copystrcontent,
            &[ro(p0), ro(p1), ro(i0), ro(i1), ro(i_len)],
            str_array_descr(),
        )];

        let (result, constants, _gcrefs) =
            rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        // Expect (itemscale=0 so no INT_LSHIFT):
        //   i2b = int_add(p0, i0)
        //   i2  = int_add(i2b, basesize-1)
        //   i3b = int_add(p1, i1)
        //   i3  = int_add(i3b, basesize-1)
        //   call_n(memcpy_fn, i3, i2, i_len, descr=memcpy_descr)
        assert_eq!(result.len(), 5);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[0].arg(0).to_opref(), p0);
        assert_eq!(result[0].arg(1).to_opref(), i0);
        assert_eq!(result[1].opcode, OpCode::IntAdd);
        assert_eq!(result[1].arg(0).to_opref(), result[0].pos().get());
        assert_eq!(
            result[1]
                .arg(1)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            16
        ); // str_basesize - 1
        assert_eq!(result[2].opcode, OpCode::IntAdd);
        assert_eq!(result[2].arg(0).to_opref(), p1);
        assert_eq!(result[2].arg(1).to_opref(), i1);
        assert_eq!(result[3].opcode, OpCode::IntAdd);
        assert_eq!(result[3].arg(0).to_opref(), result[2].pos().get());
        assert_eq!(
            result[3]
                .arg(1)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            16
        );
        assert_eq!(result[4].opcode, OpCode::CallN);
        assert_eq!(result[4].arg(1).to_opref(), result[3].pos().get()); // dst
        assert_eq!(result[4].arg(2).to_opref(), result[1].pos().get()); // src
        assert_eq!(result[4].arg(3).to_opref(), i_len);
        assert!(result[4].has_descr(), "CALL_N must carry memcpy_descr");
    }

    // ── COPYUNICODECONTENT without LEA → LSHIFT + INT_ADD + INT_ADD per side + LSHIFT(len) + CALL_N ──
    //
    // rpython/jit/backend/llsupport/test/test_rewrite.py test_rewrite_copyunicodecontents_without_load_effective_address
    // `test_rewrite_copyunicodecontents_without_load_effective_address`.
    #[test]
    fn test_rewrite_copyunicodecontents_without_load_effective_address() {
        let mut rw = make_rewriter();
        rw.supports_load_effective_address = false;
        let p0 = OpRef::ref_op(0);
        let p1 = OpRef::ref_op(1);
        let i0 = OpRef::int_op(2);
        let i1 = OpRef::int_op(3);
        let i_len = OpRef::int_op(4);
        let ops = vec![Op::with_descr(
            OpCode::Copyunicodecontent,
            &[ro(p0), ro(p1), ro(i0), ro(i1), ro(i_len)],
            unicode_array_descr(),
        )];

        let (result, constants, _gcrefs) =
            rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        // Expect (itemscale=2):
        //   i0s = int_lshift(i0, 2)
        //   i2b = int_add(p0, i0s)
        //   i2  = int_add(i2b, basesize)
        //   i1s = int_lshift(i1, 2)
        //   i3b = int_add(p1, i1s)
        //   i3  = int_add(i3b, basesize)
        //   i4  = int_lshift(i_len, 2)
        //   call_n(memcpy_fn, i3, i2, i4, descr=memcpy_descr)
        assert_eq!(result.len(), 8);
        assert_eq!(result[0].opcode, OpCode::IntLshift);
        assert_eq!(result[0].arg(0).to_opref(), i0);
        assert_eq!(
            result[0]
                .arg(1)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            2
        );
        assert_eq!(result[1].opcode, OpCode::IntAdd);
        assert_eq!(result[1].arg(0).to_opref(), p0);
        assert_eq!(result[1].arg(1).to_opref(), result[0].pos().get());
        assert_eq!(result[2].opcode, OpCode::IntAdd);
        assert_eq!(result[2].arg(0).to_opref(), result[1].pos().get());
        assert_eq!(
            result[2]
                .arg(1)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            16
        );
        assert_eq!(result[3].opcode, OpCode::IntLshift);
        assert_eq!(result[3].arg(0).to_opref(), i1);
        assert_eq!(
            result[3]
                .arg(1)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            2
        );
        assert_eq!(result[4].opcode, OpCode::IntAdd);
        assert_eq!(result[4].arg(0).to_opref(), p1);
        assert_eq!(result[4].arg(1).to_opref(), result[3].pos().get());
        assert_eq!(result[5].opcode, OpCode::IntAdd);
        assert_eq!(result[5].arg(0).to_opref(), result[4].pos().get());
        assert_eq!(
            result[5]
                .arg(1)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            16
        );
        assert_eq!(result[6].opcode, OpCode::IntLshift);
        assert_eq!(result[6].arg(0).to_opref(), i_len);
        assert_eq!(
            result[6]
                .arg(1)
                .to_opref()
                .inline_const_bits()
                .expect("inline ConstInt"),
            2
        );
        assert_eq!(result[7].opcode, OpCode::CallN);
        assert_eq!(result[7].arg(1).to_opref(), result[5].pos().get()); // dst
        assert_eq!(result[7].arg(2).to_opref(), result[2].pos().get()); // src
        assert_eq!(result[7].arg(3).to_opref(), result[6].pos().get());
    }

    /// GUARD_EXCEPTION carries a Ref result (the caught exception).  The
    /// guard arm emits through `emit_rewritten_from`, which must keep the
    /// guard's original position and register a forwarding entry; a plain
    /// `emit` renumbers the result after a preceding NEW expands into
    /// multiple ops, dangling any downstream consumer of the caught
    /// exception.
    #[test]
    fn test_guard_exception_result_keeps_pos_and_forwards_to_consumer() {
        let rw = make_rewriter();
        let exc_box = OpRef::ref_op(1);
        let ops = vec![
            // Expands to CallMallocNursery + GcStore(tid), shifting the
            // rewriter's position counter past the guard's original pos.
            mk_op_with_descr(OpCode::New, &[], 0, size_descr(16, 7)),
            mk_op(OpCode::GuardException, &[OpRef::int_op(99)], 1),
            // Downstream consumer of the caught exception.
            mk_op_with_descr(
                OpCode::SetfieldGc,
                &[OpRef::ref_op(0), exc_box],
                2,
                ref_field_descr(),
            ),
        ];

        let result = rw.rewrite_ops(&ops);

        let guard = result
            .iter()
            .find(|o| o.opcode == OpCode::GuardException)
            .expect("GUARD_EXCEPTION must survive the rewrite");
        assert_eq!(
            guard.pos().get(),
            OpRef::op_typed(1, Type::Ref),
            "non-void guard result must keep its original position"
        );
        // SETFIELD_GC lowers to GcStore(ptr, ofs, value, size); the value
        // must still reference the guard's result, not a renumbered slot.
        let store = result
            .iter()
            .filter(|o| o.opcode == OpCode::GcStore)
            .find(|o| o.arg(2).to_opref() == guard.pos().get())
            .expect("consumer GcStore must reference the guard's result");
        assert_eq!(store.arg(0).to_opref(), result[0].pos().get());
    }

    // ── gc_table: remove_constptr / gcref dedup / load CSE ──
    // rewrite.py emit_op ConstPtr branch, rewrite.py
    // remove_constptr, rewrite.py _gcref_index, rewrite.py
    // emit_label CSE reset.

    #[test]
    fn test_remove_constptr_emits_load_and_collects_gcref() {
        let _const_ptr_table = const_ptr_table_test_guard();
        let rw = make_rewriter();
        let r = majit_ir::GcRef(0x4000);
        let ops = vec![Op::new(
            OpCode::GuardNonnull,
            &[Operand::const_from_value(Value::Ref(r))],
        )];

        let (result, _consts, gcrefs) = rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        // The reference constant is collected at index 0.
        assert_eq!(gcrefs, vec![r]);
        // A LoadFromGcTable(0) is emitted before the consuming op.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::LoadFromGcTable);
        assert_eq!(result[0].arg(0).const_value(), Some(Value::Int(0)));
        // The consumer's ConstPtr arg is replaced by the load result.
        assert_eq!(result[1].opcode, OpCode::GuardNonnull);
        assert_eq!(result[1].arg(0).to_opref(), result[0].pos().get());
    }

    #[test]
    fn test_remove_constptr_dedup_and_block_cse() {
        let _const_ptr_table = const_ptr_table_test_guard();
        let rw = make_rewriter();
        let r = majit_ir::GcRef(0x4000);
        let ops = vec![
            Op::new(
                OpCode::GuardNonnull,
                &[Operand::const_from_value(Value::Ref(r))],
            ),
            Op::new(
                OpCode::GuardNonnull,
                &[Operand::const_from_value(Value::Ref(r))],
            ),
        ];

        let (result, _consts, gcrefs) = rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        // Same ref ⇒ one gcref entry (dedup) and one LoadFromGcTable (block CSE).
        assert_eq!(gcrefs, vec![r]);
        let loads = result
            .iter()
            .filter(|o| o.opcode == OpCode::LoadFromGcTable)
            .count();
        assert_eq!(loads, 1);
        // Both consumers reference the single load result.
        let load_pos = result
            .iter()
            .find(|o| o.opcode == OpCode::LoadFromGcTable)
            .unwrap()
            .pos()
            .get();
        for g in result.iter().filter(|o| o.opcode == OpCode::GuardNonnull) {
            assert_eq!(g.arg(0).to_opref(), load_pos);
        }
    }

    #[test]
    fn test_remove_constptr_distinct_refs() {
        let _const_ptr_table = const_ptr_table_test_guard();
        let rw = make_rewriter();
        let r0 = majit_ir::GcRef(0x4000);
        let r1 = majit_ir::GcRef(0x5000);
        let ops = vec![
            Op::new(
                OpCode::GuardNonnull,
                &[Operand::const_from_value(Value::Ref(r0))],
            ),
            Op::new(
                OpCode::GuardNonnull,
                &[Operand::const_from_value(Value::Ref(r1))],
            ),
        ];

        let (result, _consts, gcrefs) = rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        // Distinct refs occupy distinct, stable indices.
        assert_eq!(gcrefs, vec![r0, r1]);
        let load_indices: Vec<i64> = result
            .iter()
            .filter(|o| o.opcode == OpCode::LoadFromGcTable)
            .map(|o| match o.arg(0).const_value() {
                Some(Value::Int(i)) => i,
                other => panic!("LoadFromGcTable index must be ConstInt, got {other:?}"),
            })
            .collect();
        assert_eq!(load_indices, vec![0, 1]);
    }

    #[test]
    fn test_remove_constptr_null_stays_inline() {
        let _const_ptr_table = const_ptr_table_test_guard();
        let rw = make_rewriter();
        let null = majit_ir::GcRef(0);
        let ops = vec![Op::new(
            OpCode::GuardNonnull,
            &[Operand::const_from_value(Value::Ref(null))],
        )];

        let (result, _consts, gcrefs) = rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        // rewrite.py `bool(arg.value)` — a null ConstPtr stays inline.
        assert!(
            gcrefs.is_empty(),
            "null ConstPtr must not enter the gc_table"
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::GuardNonnull);
        assert_eq!(result[0].arg(0).const_value(), Some(Value::Ref(null)));
    }

    #[test]
    fn test_remove_constptr_cse_resets_at_label() {
        let _const_ptr_table = const_ptr_table_test_guard();
        let rw = make_rewriter();
        let r = majit_ir::GcRef(0x4000);
        let ops = vec![
            Op::new(
                OpCode::GuardNonnull,
                &[Operand::const_from_value(Value::Ref(r))],
            ),
            Op::new(OpCode::Label, &[]),
            Op::new(
                OpCode::GuardNonnull,
                &[Operand::const_from_value(Value::Ref(r))],
            ),
        ];

        let (result, _consts, gcrefs) = rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        // Dedup persists across the block boundary (one gcref) ...
        assert_eq!(gcrefs, vec![r]);
        // ... but the load CSE resets at the Label, so the ref is reloaded.
        let loads = result
            .iter()
            .filter(|o| o.opcode == OpCode::LoadFromGcTable)
            .count();
        assert_eq!(loads, 2);
    }

    #[test]
    fn test_jit_debug_keeps_constptr_inline() {
        let _const_ptr_table = const_ptr_table_test_guard();
        let rw = make_rewriter();
        let r = majit_ir::GcRef(0x4000);
        let ops = vec![Op::new(
            OpCode::JitDebug,
            &[Operand::const_from_value(Value::Ref(r))],
        )];

        let (result, _consts, gcrefs) = rw.rewrite_ops_with_constants(&ops, &ConstMap::default());

        // rewrite.py:105 `keep` — JIT_DEBUG keeps its constants inline.
        assert!(gcrefs.is_empty(), "JIT_DEBUG keeps its constants inline");
        assert!(
            !result.iter().any(|o| o.opcode == OpCode::LoadFromGcTable),
            "JIT_DEBUG must not route its ConstPtr through the gc_table"
        );
    }

    #[test]
    fn remove_bridge_exception_strips_unused_prefix() {
        let class = OpRc::new(Op::new(OpCode::SaveExcClass, &[]));
        let value = OpRc::new(Op::new(OpCode::SaveException, &[]));
        let restore = Op::new(
            OpCode::RestoreException,
            &[
                Operand::from_bound_op(&class),
                Operand::from_bound_op(&value),
            ],
        );
        let ops = vec![
            Op::new(OpCode::SaveExcClass, &[]),
            Op::new(OpCode::SaveException, &[]),
            restore,
            Op::new(OpCode::Finish, &[]),
        ];
        let boxed: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
        let out = GcRewriterImpl::remove_bridge_exception(&boxed);
        // No op reuses the saved class/value, so the prefix is stripped.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].opcode, OpCode::Finish);
    }

    #[test]
    fn remove_bridge_exception_keeps_prefix_when_value_reused() {
        let class = OpRc::new(Op::new(OpCode::SaveExcClass, &[]));
        let value = OpRc::new(Op::new(OpCode::SaveException, &[]));
        let restore = Op::new(
            OpCode::RestoreException,
            &[
                Operand::from_bound_op(&class),
                Operand::from_bound_op(&value),
            ],
        );
        // A routed handler bridge keeps the saved value as `last_exc_value`
        // for `except E as e`, so a later op still references it.
        let consumer = Op::new(OpCode::SameAsR, &[Operand::from_bound_op(&value)]);
        let ops = vec![
            Op::new(OpCode::SaveExcClass, &[]),
            Op::new(OpCode::SaveException, &[]),
            restore,
            consumer,
        ];
        let boxed: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
        let out = GcRewriterImpl::remove_bridge_exception(&boxed);
        // The reused value pins the whole prefix in place.
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].opcode, OpCode::SaveExcClass);
    }

    #[test]
    fn remove_ref_constants_follows_offstream_failarg_forwarding() {
        use majit_ir::forwarding::ForwardingHost;

        // Optimizer forwarded an off-stream producer (pos 10) onto a live
        // loop value (pos 3). `forget_optimization_info` never sees the
        // off-stream box, so `_forwarded` survives into the backend rewrite.
        let live = Op::new(
            OpCode::IntAdd,
            &[ro(OpRef::int_op(1)), ro(OpRef::int_op(2))],
        );
        live.pos().set(OpRef::int_op(3));
        let live_rc = OpRc::new(live.clone());

        let ghost = Op::new(OpCode::SameAsI, &[ro(OpRef::int_op(3))]);
        ghost.pos().set(OpRef::int_op(10));
        ghost.set_forwarded_op(&live_rc);

        let cond = Op::new(OpCode::IntLt, &[ro(OpRef::int_op(3)), ro(OpRef::int_op(4))]);
        cond.pos().set(OpRef::int_op(5));
        let guard = Op::new(OpCode::GuardTrue, &[ro(OpRef::int_op(5))]);
        guard.store_final_boxes(vec![Operand::from_bound_op(&OpRc::new(ghost))]);

        let (out, _gcrefs) = remove_ref_constants(&[live, cond, guard], 11);
        let rewritten_guard = out
            .iter()
            .rev()
            .find(|op| op.opcode == OpCode::GuardTrue)
            .expect("guard survives");
        let fail_args = rewritten_guard.getfailargs().expect("guard failargs");
        assert_eq!(fail_args.len(), 1);
        assert_eq!(
            fail_args[0].to_opref(),
            OpRef::int_op(3),
            "failarg must follow _forwarded onto the live producer"
        );
    }

    #[test]
    fn remove_ref_constants_does_not_load_failarg_constptr_on_the_hot_path() {
        let _const_ptr_table = const_ptr_table_test_guard();
        // rewrite.py emit_op intern's only op args. A ConstPtr failarg
        // stays a constant; the table still records it so deopt can
        // rematerialize a forwarded address.
        let cond = Op::new(OpCode::IntLt, &[ro(OpRef::int_op(1)), ro(OpRef::int_op(2))]);
        cond.pos().set(OpRef::int_op(3));
        let guard = Op::new(OpCode::GuardTrue, &[ro(OpRef::int_op(3))]);
        guard
            .setfailargs(vec![Operand::const_from_value(Value::Ref(GcRef(0x9C_0000_1000)))].into());
        let (out, gcrefs) = remove_ref_constants(&[cond, guard], 4);
        assert_eq!(gcrefs, vec![GcRef(0x9C_0000_1000)]);
        assert!(
            out.iter().all(|op| op.opcode != OpCode::LoadFromGcTable),
            "failarg ConstPtr must not emit a LoadFromGcTable in the stream"
        );
        let rewritten_guard = out
            .iter()
            .rev()
            .find(|op| op.opcode == OpCode::GuardTrue)
            .expect("guard survives");
        let fail_args = rewritten_guard.getfailargs().expect("guard failargs");
        assert_eq!(fail_args.len(), 1);
        assert_eq!(
            fail_args[0].const_value(),
            Some(Value::Ref(GcRef(0x9C_0000_1000))),
            "failarg stays a ConstPtr for deopt rematerialize"
        );
    }
}
