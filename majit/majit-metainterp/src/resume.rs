//! Resume data: encodes the mapping from guard fail_args to full interpreter state.
//!
//! When a guard fails, the JIT needs to reconstruct the interpreter's full state
//! (program counter, local variables, stack contents) from the values stored in
//! the DeadFrame. Resume data provides this mapping.
//!
//! This is the RPython equivalent of `rpython/jit/metainterp/resume.py`.

use indexmap::{IndexMap, IndexSet};
use rustc_hash::FxBuildHasher;
use smallvec::SmallVec;
use std::cell::UnsafeCell;
use std::sync::Arc;

use majit_backend::{
    ExitFrameLayout, ExitPendingFieldLayout, ExitRecoveryLayout, ExitValueSourceLayout,
    ExitVirtualLayout, FailArgSource,
};
use majit_ir::operand::Operand;
use majit_ir::{Const, GcRef, OpRef, Type};

/// resume.py:656-670: element kind from arraydescr.
/// 0=ref (is_array_of_pointers), 1=int, 2=float (is_array_of_floats).
fn array_kind_from_descr(arraydescr: Option<&majit_ir::DescrRef>) -> u8 {
    arraydescr
        .and_then(|d| d.as_array_descr())
        .map(|ad| {
            if ad.is_array_of_pointers() {
                0u8
            } else if ad.is_array_of_floats() {
                2u8
            } else {
                1u8
            }
        })
        .unwrap_or(0)
}

// RPython resume.py TagOverflow — structural port (i16 tags).

// resume.py TagOverflow
#[derive(Debug)]
pub struct TagOverflow;

// resume.py tag
pub fn tag(value: i32, tagbits: u8) -> Result<i16, TagOverflow> {
    debug_assert!(tagbits <= 3);
    let sx = value >> 13;
    if sx != 0 && sx != -1 {
        return Err(TagOverflow);
    }
    Ok(((value << 2) | tagbits as i32) as i16)
}

// resume.py untag
pub fn untag(value: i16) -> (i32, u8) {
    let widened = value as i32;
    let tagbits = (widened & TAGMASK as i32) as u8;
    (widened >> 2, tagbits)
}

// resume.py tagged_eq
#[inline]
pub fn tagged_eq(x: i16, y: i16) -> bool {
    (x as i32) == (y as i32)
}

// resume.py tagged_list_eq
pub fn tagged_list_eq(tl1: &[i16], tl2: &[i16]) -> bool {
    if tl1.len() != tl2.len() {
        return false;
    }
    tl1.iter().zip(tl2.iter()).all(|(&a, &b)| tagged_eq(a, b))
}

// resume.py:123-132
pub const TAGCONST: u8 = 0;
pub const TAGINT: u8 = 1;
pub const TAGBOX: u8 = 2;
pub const TAGVIRTUAL: u8 = 3;
const TAGMASK: u8 = 3;

pub const UNASSIGNED: i16 = ((-1i32 << 13) << 2 | TAGBOX as i32) as i16;
pub const UNASSIGNEDVIRTUAL: i16 = ((-1i32 << 13) << 2 | TAGVIRTUAL as i32) as i16;
pub const NULLREF: i16 = ((-1i32 << 2) | TAGCONST as i32) as i16;
pub const UNINITIALIZED_TAG: i16 = ((-2i32 << 2) | TAGCONST as i32) as i16;
pub const TAG_CONST_OFFSET: i32 = 0;

/// Ordered livebox map: canonical box (`Rc::ptr_eq`) → i16 tag.
///
/// `ResumeDataLoopMemo._number_boxes` / `_number_virtuals`: RPython uses
/// `dict` keyed by the actual Box object (object `is` identity). In
/// Python 3 that dict is insertion-ordered, and `_number_virtuals`
/// iterates it directly. #160/S11 keys this map by the
/// canonical [`Operand`](majit_ir::operand::Operand) (`Rc::ptr_eq` on the
/// producer = PyPy `box is box`), the faithful port of the dict-by-`is` —
/// `Operand` IS the box object `resume.py liveboxes` stores. The backing is an
/// insertion-ordered sequence of `(box, tag)` pairs — resume numbering walks it
/// in that order — looked up by a linear identity scan while the map is small,
/// which is where IndexMap's separate hash-index allocation was the cost, and
/// through an identity index above that, which is where the scan would be.
/// Neither changes dict-assignment semantics.
/// Two reaches of one logical box
/// resolve to one producer Rc (via `from_bound_op`/`from_bound_inputarg`) and
/// collapse to one key; distinct boxes — e.g. an `InputArg` vs a `ResOp`
/// result — stay distinct, where a raw-position key could have aliased them.
///
/// Invariant: keys are never Const boxes. Per
/// `ResumeDataLoopMemo._number_boxes`, `isinstance(box, Const)`
/// short-circuits to
/// `self.getconst(box)` and the result is never written into
/// `numb_state.liveboxes`. Only the `else` branch (line 207-223,
/// non-Const Box) reaches `liveboxes[box] = tagged`. `insert` enforces
/// this via `debug_assert!` so a const-keyed insertion fails loudly in
/// debug builds rather than silently producing an out-of-RPython-shape
/// numbering state.
pub struct LiveboxMap {
    /// Insertion-order live set. Eight pairs stay inline so the common
    /// four-or-fewer liveboxes never heap-grow (24 B/pair, Vec 1→2→4 = 96 B).
    entries: SmallVec<[(majit_ir::operand::Operand, i16); 8]>,
    /// Identity index over `entries`, built once the map outgrows
    /// [`Self::LINEAR_MAX`] and maintained from then on.
    ///
    /// `dict` is O(1) at every size, and a guard's live set is not bounded:
    /// `_number_boxes` looks up (and usually inserts) once per live box, so a
    /// scan-only map is quadratic per guard and multiplies bridge-compilation
    /// time on a trace with hundreds of them. The scan is kept for the small
    /// maps the JIT workloads actually produce — where a few pointer compares
    /// beat hashing and the index's allocation is the cost being avoided —
    /// and abandoned as soon as that stops being true.
    index: Option<std::collections::HashMap<majit_ir::operand::Operand, usize, FxBuildHasher>>,
}

impl LiveboxMap {
    /// Entries held before the identity index is built. Below this a linear
    /// identity scan is a handful of pointer compares against one cache line
    /// or two; above it the scan is what the index exists to replace.
    const LINEAR_MAX: usize = 16;

    pub fn new() -> Self {
        Self {
            entries: SmallVec::new(),
            index: None,
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.index = None;
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Slot of `b` in `entries`, through the index once there is one.
    #[inline(always)]
    fn position(&self, b: &majit_ir::operand::Operand) -> Option<usize> {
        match self.index.as_ref() {
            Some(index) => index.get(b).copied(),
            None => self.entries.iter().position(|(key, _)| key == b),
        }
    }

    #[inline(always)]
    pub fn get(&self, b: &majit_ir::operand::Operand) -> Option<i16> {
        self.position(b).map(|at| self.entries[at].1)
    }

    #[inline(always)]
    pub fn insert(&mut self, b: majit_ir::operand::Operand, value: i16) {
        debug_assert!(
            b.const_value().is_none(),
            "LiveboxMap::insert: Const box {b:?} violates resume.py:204-223 \
             `_number_boxes` invariant — `isinstance(box, Const)` is encoded \
             via `getconst(box)` and never enters numb_state.liveboxes",
        );
        if let Some(at) = self.position(&b) {
            self.entries[at].1 = value;
            return;
        }
        let at = self.entries.len();
        self.entries.push((b.clone(), value));
        match self.index.as_mut() {
            Some(index) => {
                index.insert(b, at);
            }
            None if self.entries.len() > Self::LINEAR_MAX => {
                self.index = Some(
                    self.entries
                        .iter()
                        .enumerate()
                        .map(|(at, (key, _))| (key.clone(), at))
                        .collect(),
                );
            }
            None => {}
        }
    }

    #[inline(always)]
    pub fn contains_key(&self, b: &majit_ir::operand::Operand) -> bool {
        self.position(b).is_some()
    }

    /// Iterate over all (canonical box, tag) pairs in RPython dict insertion
    /// order (Rc::ptr_eq identity = PyPy `box is box`).
    pub fn iter(&self) -> impl Iterator<Item = (majit_ir::operand::Operand, i16)> + '_ {
        self.entries.iter().map(|(op, value)| (op.clone(), *value))
    }
}

impl Default for LiveboxMap {
    fn default() -> Self {
        Self::new()
    }
}

// resume.py NumberingState
pub struct NumberingState {
    pub writer: crate::resumecode::Writer,
    pub liveboxes: LiveboxMap,
    pub num_boxes: i32,
    pub num_virtuals: i32,
}

impl NumberingState {
    pub fn new(size: usize) -> Self {
        NumberingState {
            writer: crate::resumecode::Writer::new(size),
            liveboxes: LiveboxMap::new(),
            num_boxes: 0,
            num_virtuals: 0,
        }
    }
    pub fn append_short(&mut self, item: i16) {
        self.writer.append_short(item as i32);
    }
    pub fn append_int(&mut self, item: i64) {
        self.writer.append_int(item);
    }
    pub fn patch_current_size(&mut self, index: usize) {
        self.writer.patch_current_size(index);
    }
    pub fn create_numbering(&self) -> Vec<u8> {
        self.writer.create_numbering()
    }
    pub fn create_numbering_arc(&self) -> majit_ir::NumberingRef {
        self.writer.create_numbering_arc()
    }
}

/// RPython snapshot: the state captured at a guard point.
/// Corresponds to RPython's SnapshotIterator output:
/// snapshot_iter.vable_array, snapshot_iter.vref_array, snapshot_iter.framestack.
///
/// NOTE: RPython does not have this struct. It uses `trace.get_snapshot_iter(position)`
/// which returns a lazy iterator over the trace's snapshot data (opencoder.py).
/// We use an eager struct because pyre's tracing records fail_args directly
/// on guard ops rather than using RPython's snapshot log format.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Virtualizable field boxes (resume.py:234-241).
    pub vable_array: Vec<SnapshotBox>,
    /// Virtualref pairs (resume.py:243-247). Length must be even.
    pub vref_array: Vec<SnapshotBox>,
    /// Frame chain (resume.py:249-253). Multiple frames for inlined calls.
    pub framestack: Vec<SnapshotFrame>,
}

/// A snapshot entry corresponding to one RPython Box.
///
/// RPython carries `box.type` on the Box object itself. Pyre's typed
/// `OpRef` enum (resoperation.py InputArgInt/727/739 InputArg{Int,Float,Ref},
/// resoperation.py IntOp *Op mixin variants) carries the same type
/// tag intrinsically. There is no parallel type word: a second field
/// made a two-box snapshot 48 B.
#[derive(Debug, Clone, Copy)]
pub struct SnapshotBox {
    /// The trace-position ref this snapshot slot references. A
    /// `Const{Ptr}` slot carries its gcref inline (history.py
    /// `ConstPtr.value`); during compilation the snapshot root walker
    /// (`walk_compile_snapshot_refs`) forwards it in place through a
    /// collected `*mut OpRef` slot address.
    pub opref: majit_ir::OpRef,
}

impl SnapshotBox {
    pub fn untyped(opref: majit_ir::OpRef) -> Self {
        SnapshotBox { opref }
    }

    pub fn typed(opref: majit_ir::OpRef, _tp: majit_ir::Type) -> Self {
        SnapshotBox { opref }
    }

    /// The trace-position `OpRef` view of this slot.
    pub fn opref(&self) -> majit_ir::OpRef {
        self.opref
    }

    /// history.py `Box.type` — the OpRef variant tag.
    pub fn tp(&self) -> Option<majit_ir::Type> {
        self.opref.ty()
    }

    pub fn map_opref(&self, f: impl FnOnce(majit_ir::OpRef) -> majit_ir::OpRef) -> Self {
        SnapshotBox {
            opref: f(self.opref),
        }
    }
}

impl From<majit_ir::OpRef> for SnapshotBox {
    fn from(opref: majit_ir::OpRef) -> Self {
        SnapshotBox::untyped(opref)
    }
}

#[cfg(test)]
mod snapshot_box_size {
    #[test]
    fn snapshot_box_is_one_opref() {
        assert_eq!(
            std::mem::size_of::<super::SnapshotBox>(),
            std::mem::size_of::<majit_ir::OpRef>(),
            "SnapshotBox must stay one OpRef; box.type lives on the variant"
        );
    }
}

/// One frame in a snapshot's frame chain.
#[derive(Debug, Clone)]
pub struct SnapshotFrame {
    /// Index into the jitcode table (resume.py:250 jitcode_index).
    pub jitcode_index: i32,
    /// JitCode byte offset (`resume.py` `ResumeDataLoopMemo.number`).
    pub pc: i32,
    /// Python instruction coordinate captured with the snapshot. Not a
    /// numbering word: `ResumeDataLoopMemo.number` writes `jitcode_index`
    /// and `pc` only. Resume recovers the Python pc from those two via
    /// `resume_py_pc_for_jitcode_word`.
    pub py_pc: i32,
    /// Live boxes for this frame's registers (resume.py:253).
    pub boxes: Vec<SnapshotBox>,
}

impl Snapshot {
    /// Create a simple single-frame snapshot (pyre common case).
    ///
    /// `jitcode_index` identifies the code this frame is running — the
    /// index into `METAINTERP_SD.jitcodes`. Required so the decoder's
    /// `frame_value_count_at(jitcode_index, pc)` query resolves the
    /// frame's liveness in the correct jitcode instead of silently
    /// falling through the pc-out-of-range LiveVars path on `jitcodes[0]`.
    pub fn single_frame(jitcode_index: i32, pc: i32, boxes: Vec<majit_ir::OpRef>) -> Self {
        Self::single_frame_boxes(
            jitcode_index,
            pc,
            boxes.into_iter().map(SnapshotBox::from).collect(),
        )
    }

    pub fn single_frame_boxes(jitcode_index: i32, pc: i32, boxes: Vec<SnapshotBox>) -> Self {
        Snapshot {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            framestack: vec![SnapshotFrame {
                jitcode_index,
                pc,
                py_pc: pc,
                boxes,
            }],
        }
    }

    /// Create a multi-frame snapshot from per-frame (jitcode_index, pc,
    /// boxes) tuples. Read-side ordering matches upstream: after
    /// `SnapshotIterator.__init__` calls `self.framestack.reverse()`
    /// (`opencoder.py`), `framestack[0]` is the outermost/caller
    /// frame and the last element is the innermost/callee — as asserted
    /// by `test_opencoder.py` (jc_index=2 at `framestack[0]`,
    /// jc_index=4 at `framestack[1]`). Input tuples for this factory
    /// follow the same caller-first order.
    pub fn multi_frame(frames: Vec<(i32, i32, Vec<majit_ir::OpRef>)>) -> Self {
        Self::multi_frame_boxes(
            frames
                .into_iter()
                .map(|(jitcode_index, pc, boxes)| {
                    (
                        jitcode_index,
                        pc,
                        boxes.into_iter().map(SnapshotBox::from).collect(),
                    )
                })
                .collect(),
        )
    }

    pub fn multi_frame_boxes(frames: Vec<(i32, i32, Vec<SnapshotBox>)>) -> Self {
        Snapshot {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            framestack: frames
                .into_iter()
                .map(|(jitcode_index, pc, boxes)| SnapshotFrame {
                    jitcode_index,
                    pc,
                    py_pc: pc,
                    boxes,
                })
                .collect(),
        }
    }

    /// Estimated encoded size for NumberingState capacity hint.
    pub fn estimated_size(&self) -> usize {
        let frame_size: usize = self.framestack.iter().map(|f| f.boxes.len() + 2).sum();
        self.vable_array.len() + self.vref_array.len() + frame_size + 4
    }
}

/// Re-export BoxEnv from majit-ir.
pub use majit_ir::BoxEnv;

/// Simple BoxEnv implementation backed by constant/type associative containers.
/// Used in tests and for simple snapshot numbering.
pub struct SimpleBoxEnv {
    pub constants: indexmap::IndexMap<u32, (i64, majit_ir::Type)>,
    pub replacements: indexmap::IndexMap<u32, majit_ir::OpRef>,
    pub types: indexmap::IndexMap<u32, majit_ir::Type>,
    pub virtuals: indexmap::IndexSet<u32>,
    pub virtual_fields: indexmap::IndexMap<u32, majit_ir::VirtualFieldsInfo>,
    /// #160/S11: one canonical box `Operand` per replacement-walked OpRef. This
    /// env holds no producer Ops, so it memoizes here to give `Rc::ptr_eq` dedup
    /// parity with production (where `from_bound_op` memoizes on the Op).
    box_cache: std::cell::RefCell<indexmap::IndexMap<majit_ir::OpRef, majit_ir::operand::Operand>>,
}

impl Default for SimpleBoxEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleBoxEnv {
    pub fn new() -> Self {
        SimpleBoxEnv {
            constants: indexmap::IndexMap::new(),
            replacements: indexmap::IndexMap::new(),
            types: indexmap::IndexMap::new(),
            virtuals: indexmap::IndexSet::new(),
            virtual_fields: indexmap::IndexMap::new(),
            box_cache: std::cell::RefCell::new(indexmap::IndexMap::new()),
        }
    }
}

impl BoxEnv for SimpleBoxEnv {
    // resoperation.py get_box_replacement walks the chain
    // op -> op._forwarded -> ... until None / AbstractInfo, returning the
    // last item before that. Iterate the replacement map the same way.
    fn get_box_replacement(&self, opref: majit_ir::OpRef) -> majit_ir::OpRef {
        // history.py/268/314 inline-Const is its own forwarding root
        // (Const objects never participate in `_forwarded` per
        // resoperation.py default `_forwarded = None`).
        if opref.inline_const_bits().is_some() {
            return opref;
        }
        let mut opref = opref;
        while !opref.is_constant() {
            let Some(next) = self.replacements.get(&opref.raw()).copied() else {
                break;
            };
            if next == opref {
                return opref;
            }
            opref = next;
        }
        opref
    }

    fn get_box_replacement_operand(&self, opref: majit_ir::OpRef) -> majit_ir::operand::Operand {
        // #160/S11: no producer Ops here, so memoize one Operand per
        // replacement-walked OpRef to mirror production's from_bound_op
        // memoization — two reaches of one logical box share an Rc (ptr_eq).
        let root = self.get_box_replacement(opref);
        if let Some(b) = self.box_cache.borrow().get(&root) {
            return b.clone();
        }
        // This env holds no producer Ops; in tests, synthesize a rooted bound
        // producer so the box sheds to `Operand::Op`/`InputArg` (the Operand-
        // keyed liveboxes/cached maps reject a position-only box). The method
        // is never reached in non-test builds, where `from_opref` is retained.
        #[cfg(test)]
        let b = crate::history::test_support::rooted_operand_from_opref(root);
        #[cfg(not(test))]
        let b = majit_ir::operand::Operand::from_opref(root);
        self.box_cache.borrow_mut().insert(root, b.clone());
        b
    }

    // resoperation.py:64-65 not_const arm: stop one hop before reaching
    // a Const target, returning the predecessor.
    fn get_box_replacement_not_const(&self, opref: majit_ir::OpRef) -> majit_ir::OpRef {
        if opref.inline_const_bits().is_some() {
            return opref;
        }
        let mut opref = opref;
        while let Some(next) = self.replacements.get(&opref.raw()).copied() {
            if next == opref {
                return opref;
            }
            if next.is_constant() {
                return opref;
            }
            // Legacy idx Const sentinel in side table.
            if next.inline_const_bits().is_none() && self.constants.contains_key(&next.raw()) {
                return opref;
            }
            opref = next;
        }
        opref
    }

    fn is_const(&self, box_: &majit_ir::operand::Operand) -> bool {
        let opref = box_.to_opref();
        // history.py/268/314 inline-Const variants are constants by tag.
        if opref.is_constant() {
            return true;
        }
        if opref.is_input_arg() {
            return false;
        }
        self.constants.contains_key(&opref.raw())
    }
    fn get_const(&self, box_: &majit_ir::operand::Operand) -> (i64, majit_ir::Type) {
        let opref = box_.to_opref();
        // history.py ConstInt.value / :268 ConstFloat.value / :314 ConstPtr.value
        // inline on the Box; read directly without side-table.
        if let (Some(bits), Some(tp)) = (opref.inline_const_bits(), opref.ty()) {
            return (bits, tp);
        }
        self.constants
            .get(&opref.raw())
            .copied()
            .unwrap_or((0, majit_ir::Type::Int))
    }
    fn get_type(&self, opref: majit_ir::OpRef) -> majit_ir::Type {
        // resoperation.py opclasses[opnum].type — every typed OpRef
        // variant pins `.type` (history.py:220/261/307 + resoperation.py/589/615).  allow-line-citation
        if let Some(tp) = opref.ty() {
            return tp;
        }
        self.types
            .get(&opref.raw())
            .copied()
            .unwrap_or(majit_ir::Type::Int)
    }
    fn is_virtual_ref(&self, box_: &majit_ir::operand::Operand) -> bool {
        let opref = box_.to_opref();
        if opref.inline_const_bits().is_some() {
            return false;
        }
        self.virtuals.contains(&opref.raw())
    }
    fn is_virtual_raw(&self, box_: &majit_ir::operand::Operand) -> bool {
        let opref = box_.to_opref();
        if opref.inline_const_bits().is_some() {
            return false;
        }
        self.virtuals.contains(&opref.raw())
    }
    fn get_virtual_fields(&self, opref: majit_ir::OpRef) -> Option<majit_ir::VirtualFieldsInfo> {
        if opref.inline_const_bits().is_some() {
            return None;
        }
        self.virtual_fields.get(&opref.raw()).cloned()
    }
}

// resume.py — tag constants (i64 widened for rd_numb encoding).
// Same values as the i16 TAGCONST/TAGINT/TAGBOX/TAGVIRTUAL above.
const TAGMASK_I64: i64 = TAGMASK as i64;
// resume.py `NULLREF = tag(-1, TAGCONST)` — pre-shift num for
// `Const::Ref(NULL)`. Shared by the i16 `NULLREF` constant and the
// i64 `getconst_i64`/`decode_box` pair.
const ENCODED_NULLREF: i64 = -1;
const ENCODED_UNINITIALIZED: i64 = -2;
const ENCODED_UNAVAILABLE: i64 = -3;

// Two low bits are reserved for the tag.
const INLINE_TAGGED_MIN: i64 = -(1_i64 << 61);
const INLINE_TAGGED_MAX: i64 = (1_i64 << 61) - 1;

/// `compile.py AbstractResumeGuardDescr.get_resumestorage(): return self`.
/// The resume storage is the guard descr itself: `rd_numb` / `rd_consts` /
/// `rd_virtuals` / `rd_pendingfields` are its own attributes, so every
/// reader — bridge retrace, blackhole resume, the GC root walker — holds
/// the one descr and nothing is copied per failure.
pub type ResumeStorage = dyn majit_ir::FailDescr;

/// `get_resumestorage()` on a descr handle. `None` for a descr that carries
/// no resume payload (the `_DoneWithThisFrameDescr` family /
/// `ExitFrameWithExceptionDescrRef`), matching the `_attrs_`-only-on-
/// `AbstractResumeGuardDescr` contract.
pub fn get_resumestorage(descr: &majit_ir::DescrRef) -> Option<Arc<ResumeStorage>> {
    descr
        .clone()
        .as_fail_descr_arc()
        .filter(|fd| fd.rd_numb().is_some())
}

/// A resume guard descr carrying the given payload, the way
/// `ResumeDataVirtualAdder.finish` leaves one (`storage.rd_numb = ...`).
/// Fixtures that need a storage without a compiled trace build it here.
pub fn new_resume_storage(
    rd_numb: Vec<u8>,
    rd_consts: Vec<Const>,
    rd_virtuals: Vec<std::rc::Rc<majit_ir::RdVirtualInfo>>,
    rd_pendingfields: Vec<majit_ir::GuardPendingFieldEntry>,
) -> Arc<ResumeStorage> {
    let descr = crate::compile::make_resume_guard_descr_typed(vec![]);
    let fd = descr.as_fail_descr().expect("resume guard descr");
    fd.set_rd_numb(Some(rd_numb));
    fd.set_rd_consts(Some(rd_consts));
    fd.set_rd_virtuals(Some(rd_virtuals));
    fd.set_rd_pendingfields(Some(rd_pendingfields));
    descr.as_fail_descr_arc().expect("resume guard descr")
}

/// `resume.py _prepare_pendingfields` read off `rd_pendingfields`: this
/// guard's deferred writes as exit-slot sources, ready to replay against
/// the deadframe.  Derived on demand from the guard-owned payload, so
/// nothing is kept per guard for it.
pub fn exit_pending_field_layouts(
    storage: &ResumeStorage,
    num_failargs: i32,
) -> Vec<ExitPendingFieldLayout> {
    let rd_consts = storage.rd_consts().unwrap_or(&[]);
    let num_virtuals = storage.rd_virtuals().map_or(0, <[_]>::len) as i32;
    storage
        .rd_pendingfields()
        .unwrap_or(&[])
        .iter()
        .map(|pf| exit_pending_field_layout(pf, num_failargs, rd_consts, num_virtuals))
        .collect()
}

/// One `resume.py` PENDINGFIELDSTRUCT entry (lldescr / num / fieldnum /
/// itemindex) as the layout the replayers consume: `itemindex >= 0` names a
/// setarrayitem, a field descr a setfield; target and value are the tagged
/// resume slots decoded by [`exit_source_from_tagged`].
pub fn exit_pending_field_layout(
    pf: &majit_ir::GuardPendingFieldEntry,
    num_failargs: i32,
    rd_consts: &[Const],
    num_virtuals: i32,
) -> ExitPendingFieldLayout {
    // The lldescr is always present: it is captured off the SETFIELD_GC /
    // SETARRAYITEM_GC that produced the pending field (heap.py
    // force_lazy_sets_for_guard); pyre's producer in `optimizer.rs`'s
    // `emit_guard_operation` mirrors this with `pf.descr = pf_op.descr`.
    let descr = pf
        .descr
        .clone()
        .expect("resume.py `_prepare_pendingfields`: PENDINGFIELDSTRUCT.lldescr must be set");
    let item_index = if descr.as_array_descr().is_some() {
        Some(
            usize::try_from(pf.item_index)
                .expect("resume.py `setarrayitem` pending field requires non-negative item_index"),
        )
    } else if descr.as_field_descr().is_some() {
        None
    } else {
        panic!(
            "pending field descr must be FieldDescr or ArrayDescr (descr={:?})",
            descr,
        );
    };
    ExitPendingFieldLayout {
        descr: pf.descr.clone(),
        is_array_item: item_index.is_some(),
        item_index,
        target: exit_source_from_tagged(pf.target_tagged, num_failargs, rd_consts, num_virtuals),
        value: exit_source_from_tagged(pf.value_tagged, num_failargs, rd_consts, num_virtuals),
    }
}

/// `resume.py` tag decoding of one `fieldnums` / `rd_pendingfields` entry
/// into the exit-slot source the recovery readers consume: TAGBOX names a
/// deadframe slot (negative counts from the end), TAGVIRTUAL an
/// `rd_virtuals` entry (negative for the nested virtuals
/// `resume.py assign_number_to_virtual` numbers that way, resolved as
/// `resume.py AbstractResumeDataReader.getvirtual_ptr` does), TAGINT an inline
/// int and TAGCONST an `rd_consts` entry (`resume.py` `self.consts[num -
/// TAG_CONST_OFFSET]`).
pub fn exit_source_from_tagged(
    tagged: i16,
    num_failargs: i32,
    rd_consts: &[Const],
    num_virtuals: i32,
) -> ExitValueSourceLayout {
    let (val, tagbits) = majit_ir::resumedata::untag(tagged);
    match tagbits {
        majit_ir::resumedata::TAGBOX => {
            let idx = if val >= 0 {
                val as usize
            } else {
                (num_failargs + val) as usize
            };
            ExitValueSourceLayout::ExitValue(idx)
        }
        majit_ir::resumedata::TAGVIRTUAL => {
            let idx = if val >= 0 {
                val as usize
            } else {
                (num_virtuals + val) as usize
            };
            ExitValueSourceLayout::Virtual(idx)
        }
        majit_ir::resumedata::TAGINT => ExitValueSourceLayout::Constant(val as i64, Type::Int),
        majit_ir::resumedata::TAGCONST => {
            let idx = (val - majit_ir::resumedata::TAG_CONST_OFFSET) as usize;
            let c = rd_consts.get(idx).copied().unwrap_or(Const::Int(0));
            ExitValueSourceLayout::Constant(c.as_raw_i64(), c.get_type())
        }
        _ => ExitValueSourceLayout::Constant(0, Type::Int),
    }
}

/// The `pc` of the first frame section of `rd_numb`, the frame
/// `rebuild_from_resumedata` builds first. The numbering is laid out as
/// `resume.py ResumeDataVirtualAdder.number` writes it and
/// `rebuild_from_numbering` reads it: `total_size, num_failargs, vable_len,
/// vable..., vref_len, vref pairs..., (jitcode_index, pc, values...)...`.
pub fn outermost_frame_pc(rd_numb: &[u8]) -> Option<i32> {
    let mut reader = crate::resumecode::Reader::new(rd_numb);
    if !reader.has_more() {
        return None;
    }
    let total_size = reader.next_item();
    let _num_failargs = reader.next_item();
    let vable_len = reader.next_item();
    reader.jump(vable_len.max(0) as usize);
    let vref_len = reader.next_item();
    reader.jump((vref_len.max(0) * 2) as usize);
    if reader.items_read >= total_size as usize || !reader.has_more() {
        return None;
    }
    let _jitcode_index = reader.next_item();
    if !reader.has_more() {
        return None;
    }
    Some(reader.next_item())
}

// Pyre is single-threaded; UnsafeCell prevents auto-Send/Sync so
// provide them explicitly (matches RPython's non-thread-safe
// ResumeGuardDescr).
/// resume.py: ResumeGuardDescr storage fields.
///
/// `rd_numb` is a flat encoded numbering section (resume.py:466):
/// 1. items_resume_section (total rd_numb length)
/// 2. count (number of liveboxes, resume.py)
/// 3. number of frames
/// 4. per-frame `(pc, slot_count, slot_sources...)`
///
/// Fields match RPython's `ResumeGuardDescr`:
/// - `rd_numb`: encoded numbering (resume.py)
/// - `rd_consts`: shared constant pool (resume.py)
/// - `rd_virtuals`: live VirtualInfo objects (compile.py:858)
/// - `rd_pendingfields`: pending field writes (resume.py)
#[derive(Debug, Clone)]
pub struct EncodedResumeData {
    /// resume.py:466 storage.rd_numb — flat encoded numbering section.
    pub rd_numb: Vec<i64>,
    /// resume.py:467 storage.rd_consts — shared constant pool.
    ///
    /// RPython stores `list[Const]` (history.py/261/307). We keep the
    /// same shape so Ref entries stay visible to the minor-collection root
    /// walker (framework.py `root_walker.walk_roots` parity).
    pub rd_consts: Vec<Const>,
    /// resume.py:468 storage.rd_pendingfields — pending field writes.
    pub rd_pendingfields: Vec<EncodedPendingFieldWrite>,
    /// compile.py:858 storage.rd_virtuals — live VirtualInfo objects.
    pub rd_virtuals: Vec<VirtualInfo>,
    /// resume.py liveboxes — compact TAGBOX(n) → original FailArg index.
    /// In RPython, `liveboxes[n]` is the Box object that was assigned TAGBOX(n).
    /// Here, `liveboxes[n]` is the original deadframe slot index.
    pub liveboxes: Vec<usize>,
    /// Per-frame slot count — equivalent to jitcode liveness info.
    /// RPython uses jitcode.get_live_vars_info(pc) at decode time;
    /// we store the counts at encode time since this path lacks jitcodes.
    pub frame_sizes: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedResumeLayout {
    pub vable_array: Vec<ResumeValueSource>,
    pub vref_array: Vec<ResumeValueSource>,
    pub frames: Vec<FrameInfo>,
    pub virtuals: Vec<VirtualInfo>,
    pub pending_fields: Vec<PendingFieldInfo>,
}

// ResumeValueKind / ResumeValueLayoutSummary moved to
// majit-ir::resumedata (Phase C-1 cascade) so backend codepaths can
// reference the resume-value tag discriminator + summary without a
// metainterp dependency.
pub use majit_ir::resumedata::{ResumeValueKind, ResumeValueLayoutSummary};

// ResumeFrameLayoutSummary moved to majit-ir::resumedata (Phase C-1
// cascade); re-exported for caller compatibility.
pub use majit_ir::resumedata::ResumeFrameLayoutSummary;

// ResumeVirtualKind moved to majit-ir::resumedata (Phase C-1
// cascade); re-exported for caller compatibility.
pub use majit_ir::resumedata::ResumeVirtualKind;

// ResumeVirtualLayoutSummary moved to majit-ir::resumedata (Phase C-1
// cascade); re-exported for caller compatibility.
pub use majit_ir::resumedata::ResumeVirtualLayoutSummary;

// PendingFieldLayoutSummary moved to majit-ir::resumedata along with
// opt_descr_arc_ptr_eq helper (Phase C-1 cascade); re-exported for
// caller compatibility.
pub use majit_ir::resumedata::{PendingFieldLayoutSummary, opt_descr_arc_ptr_eq};

#[derive(Debug, Clone)]
pub struct ResumeLayoutSummary {
    pub num_frames: usize,
    pub frame_pcs: Vec<u64>,
    pub frame_slot_counts: Vec<usize>,
    pub frame_layouts: Vec<ResumeFrameLayoutSummary>,
    pub num_virtuals: usize,
    pub virtual_kinds: Vec<ResumeVirtualKind>,
    pub virtual_layouts: Vec<ResumeVirtualLayoutSummary>,

    pub pending_field_count: usize,
    pub pending_field_layouts: Vec<PendingFieldLayoutSummary>,
    pub const_pool_size: usize,
}

// ResumeValueLayoutSummary inherent impls moved to
// majit-backend::resume_value (Phase C-1 cascade).  The conversion
// methods (raw_fail_arg_position, to_resume_source, to_exit_source)
// are available through the ResumeValueLayoutSummaryExt trait
// re-exported above.

// ResumeFrameLayoutSummary inherent impls extracted as free functions
// (cross-crate orphan rule after the type moved to majit-ir::resumedata).
fn resume_frame_layout_to_frame_info(layout: &ResumeFrameLayoutSummary) -> FrameInfo {
    FrameInfo {
        jitcode_index: layout.jitcode_index,
        pc: layout.pc,
        slot_map: layout
            .slot_layouts
            .iter()
            .map(|slot| slot.to_resume_source())
            .collect(),
    }
}

fn resume_frame_layout_to_exit_frame_layout(
    layout: &ResumeFrameLayoutSummary,
    virtual_offset: usize,
) -> ExitFrameLayout {
    ExitFrameLayout {
        trace_id: layout.trace_id,
        header_pc: layout.header_pc,
        source_guard: layout.source_guard,
        pc: layout.pc,
        jitcode_index: layout.jitcode_index,
        slots: layout
            .slot_layouts
            .iter()
            .map(|slot| slot.to_exit_source(virtual_offset))
            .collect(),
        slot_types: layout.slot_types.clone(),
    }
}

/// Build a `ResumeFrameLayoutSummary` from a backend-origin `ExitFrameLayout`.
///
/// Each `ExitValueSourceLayout` slot is converted to the corresponding
/// `ResumeValueLayoutSummary`, preserving slot types when present.
/// Free function — `ResumeFrameLayoutSummary` lives in `majit-ir`
/// (orphan rule prevents inherent impl outside of that crate).
pub fn resume_frame_layout_from_exit_frame_layout(
    exit_frame: &ExitFrameLayout,
) -> ResumeFrameLayoutSummary {
    let slot_layouts: Vec<ResumeValueLayoutSummary> = exit_frame
        .slots
        .iter()
        .map(majit_backend::resume_value_layout_summary_from_exit_value_source)
        .collect();

    ResumeFrameLayoutSummary {
        trace_id: exit_frame.trace_id,
        header_pc: exit_frame.header_pc,
        source_guard: exit_frame.source_guard,
        jitcode_index: exit_frame.jitcode_index,
        pc: exit_frame.pc,
        slot_sources: Vec::new(),
        slot_layouts,
        slot_types: exit_frame.slot_types.clone(),
    }
}

// `from_exit_value_source` constructor moved to
// majit-backend::resume_value as
// `resume_value_layout_summary_from_exit_value_source` (cross-crate
// orphan rule prevents inherent impl on the foreign type).
// `to_virtual_info` / `to_exit_virtual_layout` extracted as free
// functions after `ResumeVirtualLayoutSummary` moved to
// `majit-ir::resumedata` (cross-crate orphan rule).
fn resume_virtual_layout_to_virtual_info(layout: &ResumeVirtualLayoutSummary) -> VirtualInfo {
    let s = layout;
    {
        match s {
            ResumeVirtualLayoutSummary::Object {
                descr,
                type_id,
                known_class,
                fields,
                fielddescrs,
                descr_size,
            } => VirtualInfo::VirtualObj {
                descr: descr.clone(),
                type_id: *type_id,
                known_class: *known_class,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.to_resume_source()))
                    .collect(),
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            ResumeVirtualLayoutSummary::Struct {
                typedescr,
                type_id,
                fields,
                fielddescrs,
                descr_size,
            } => VirtualInfo::VStruct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.to_resume_source()))
                    .collect(),
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            ResumeVirtualLayoutSummary::Array {
                arraydescr,
                clear,
                items,
            } => VirtualInfo::VArray {
                arraydescr: arraydescr.clone(),
                clear: *clear,
                items: items
                    .iter()
                    .map(|source| source.to_resume_source())
                    .collect(),
            },
            ResumeVirtualLayoutSummary::ArrayStruct {
                arraydescr,
                fielddescrs,
                element_fields,
                ..
            } => VirtualInfo::VArrayStruct {
                arraydescr: arraydescr.clone(),
                fielddescrs: fielddescrs.clone(),
                element_fields: element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_descr, source)| (*field_descr, source.to_resume_source()))
                            .collect()
                    })
                    .collect(),
            },
            ResumeVirtualLayoutSummary::RawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => VirtualInfo::VRawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values: values
                    .iter()
                    .map(|source| source.to_resume_source())
                    .collect(),
            },
            ResumeVirtualLayoutSummary::RawSlice { offset, parent } => VirtualInfo::VRawSlice {
                offset: *offset,
                parent: parent.to_resume_source(),
            },
            ResumeVirtualLayoutSummary::StrPlain { chars } => VirtualInfo::VStrPlain {
                chars: chars
                    .iter()
                    .map(|source| source.to_resume_source())
                    .collect(),
            },
            ResumeVirtualLayoutSummary::StrConcat { left, right } => VirtualInfo::VStrConcat {
                left: Box::new(left.to_resume_source()),
                right: Box::new(right.to_resume_source()),
            },
            ResumeVirtualLayoutSummary::StrSlice {
                source,
                start,
                length,
            } => VirtualInfo::VStrSlice {
                source: Box::new(source.to_resume_source()),
                start: Box::new(start.to_resume_source()),
                length: Box::new(length.to_resume_source()),
            },
            ResumeVirtualLayoutSummary::UniPlain { chars } => VirtualInfo::VUniPlain {
                chars: chars
                    .iter()
                    .map(|source| source.to_resume_source())
                    .collect(),
            },
            ResumeVirtualLayoutSummary::UniConcat { left, right } => VirtualInfo::VUniConcat {
                left: Box::new(left.to_resume_source()),
                right: Box::new(right.to_resume_source()),
            },
            ResumeVirtualLayoutSummary::UniSlice {
                source,
                start,
                length,
            } => VirtualInfo::VUniSlice {
                source: Box::new(source.to_resume_source()),
                start: Box::new(start.to_resume_source()),
                length: Box::new(length.to_resume_source()),
            },
            ResumeVirtualLayoutSummary::Hole => VirtualInfo::Empty,
        }
    }
}

fn resume_virtual_layout_to_exit_virtual_layout(
    layout: &ResumeVirtualLayoutSummary,
    virtual_offset: usize,
) -> ExitVirtualLayout {
    let s = layout;
    {
        match s {
            ResumeVirtualLayoutSummary::Object {
                descr,
                type_id,
                known_class,
                fields,
                fielddescrs,
                descr_size,
            } => ExitVirtualLayout::Object {
                descr: descr.clone(),
                type_id: *type_id,
                known_class: *known_class,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.to_exit_source(virtual_offset)))
                    .collect(),
                target_slot: None,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            ResumeVirtualLayoutSummary::Struct {
                typedescr,
                type_id,
                fields,
                fielddescrs,
                descr_size,
            } => ExitVirtualLayout::Struct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                fields: fields
                    .iter()
                    .map(|(field_descr, source)| {
                        (*field_descr, source.to_exit_source(virtual_offset))
                    })
                    .collect(),
                target_slot: None,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            ResumeVirtualLayoutSummary::Array {
                arraydescr,
                clear,
                items,
            } => ExitVirtualLayout::Array {
                arraydescr: arraydescr.clone(),
                clear: *clear,
                // resume.py:656-670: element type from arraydescr
                kind: array_kind_from_descr(arraydescr.as_ref()),
                items: items
                    .iter()
                    .map(|source| source.to_exit_source(virtual_offset))
                    .collect(),
            },
            ResumeVirtualLayoutSummary::ArrayStruct {
                arraydescr,
                fielddescrs,
                element_fields,
            } => ExitVirtualLayout::ArrayStruct {
                arraydescr: arraydescr.clone(),
                fielddescrs: fielddescrs.clone(),
                element_fields: element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_descr, source)| {
                                (*field_descr, source.to_exit_source(virtual_offset))
                            })
                            .collect()
                    })
                    .collect(),
            },
            ResumeVirtualLayoutSummary::RawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => ExitVirtualLayout::RawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values: values
                    .iter()
                    .map(|source| source.to_exit_source(virtual_offset))
                    .collect(),
            },
            ResumeVirtualLayoutSummary::RawSlice { offset, parent } => {
                ExitVirtualLayout::RawSlice {
                    offset: *offset,
                    base: parent.to_exit_source(virtual_offset),
                }
            }
            ResumeVirtualLayoutSummary::StrPlain { chars } => ExitVirtualLayout::StrPlain {
                is_unicode: false,
                chars: chars
                    .iter()
                    .map(|source| source.to_exit_source(virtual_offset))
                    .collect(),
            },
            ResumeVirtualLayoutSummary::UniPlain { chars } => ExitVirtualLayout::StrPlain {
                is_unicode: true,
                chars: chars
                    .iter()
                    .map(|source| source.to_exit_source(virtual_offset))
                    .collect(),
            },
            // resume.py VStrConcatInfo / resume.py VUniConcatInfo
            // — funcptr/calldescr resolved at materialization via
            // `callinfocollection.funcptr_for_oopspec(OS_STR_CONCAT /
            // OS_UNI_CONCAT)` (resume.py / 1494-1495), so the
            // exit layout carries no funcptr.
            ResumeVirtualLayoutSummary::StrConcat { left, right } => ExitVirtualLayout::StrConcat {
                is_unicode: false,
                left: left.to_exit_source(virtual_offset),
                right: right.to_exit_source(virtual_offset),
            },
            ResumeVirtualLayoutSummary::UniConcat { left, right } => ExitVirtualLayout::StrConcat {
                is_unicode: true,
                left: left.to_exit_source(virtual_offset),
                right: right.to_exit_source(virtual_offset),
            },
            // resume.py VStrSliceInfo / resume.py VUniSliceInfo
            // — funcptr/calldescr resolved via callinfocollection at
            // materialization (resume.py / 1504-1505).
            ResumeVirtualLayoutSummary::StrSlice {
                source,
                start,
                length,
            } => ExitVirtualLayout::StrSlice {
                is_unicode: false,
                str_src: source.to_exit_source(virtual_offset),
                start: start.to_exit_source(virtual_offset),
                length: length.to_exit_source(virtual_offset),
            },
            ResumeVirtualLayoutSummary::UniSlice {
                source,
                start,
                length,
            } => ExitVirtualLayout::StrSlice {
                is_unicode: true,
                str_src: source.to_exit_source(virtual_offset),
                start: start.to_exit_source(virtual_offset),
                length: length.to_exit_source(virtual_offset),
            },
            ResumeVirtualLayoutSummary::Hole => ExitVirtualLayout::Hole,
        }
    }
}

// PendingFieldLayoutSummary inherent impls extracted as free
// functions (cross-crate orphan rule after the type moved to
// majit-ir::resumedata).
fn pending_field_layout_to_pending_field_info(
    layout: &PendingFieldLayoutSummary,
) -> PendingFieldInfo {
    PendingFieldInfo {
        descr: layout.descr.clone(),
        target: layout.target.to_resume_source(),
        value: layout.value.to_resume_source(),
        item_index: layout.item_index,
    }
}

fn pending_field_layout_to_exit_pending_field_layout(
    layout: &PendingFieldLayoutSummary,
    virtual_offset: usize,
) -> ExitPendingFieldLayout {
    ExitPendingFieldLayout {
        descr: layout.descr.clone(),
        item_index: layout.item_index,
        is_array_item: layout.is_array_item,
        target: layout.target.to_exit_source(virtual_offset),
        value: layout.value.to_exit_source(virtual_offset),
    }
}

impl ResumeLayoutSummary {
    pub fn to_resume_data(&self) -> ResumeData {
        ResumeData {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames: self
                .frame_layouts
                .iter()
                .map(resume_frame_layout_to_frame_info)
                .collect(),
            virtuals: self
                .virtual_layouts
                .iter()
                .map(resume_virtual_layout_to_virtual_info)
                .collect(),
            pending_fields: self
                .pending_field_layouts
                .iter()
                .map(pending_field_layout_to_pending_field_info)
                .collect(),
        }
    }

    pub fn to_exit_recovery_layout(&self) -> ExitRecoveryLayout {
        self.to_exit_recovery_layout_with_caller_prefix(None)
    }

    pub fn to_exit_recovery_layout_with_caller_prefix(
        &self,
        caller_prefix: Option<&ExitRecoveryLayout>,
    ) -> ExitRecoveryLayout {
        if self.frame_layouts.is_empty() {
            return caller_prefix.cloned().unwrap_or(ExitRecoveryLayout {
                vable_array: Vec::new(),
                vref_array: Vec::new(),
                frames: Vec::new(),
                virtual_layouts: Vec::new(),
                pending_field_layouts: Vec::new(),
            });
        }

        let prefix_frame_count = caller_prefix
            .map(|layout| layout.frames.len().saturating_sub(self.frame_layouts.len()))
            .unwrap_or(0);
        let preserve_prefix = prefix_frame_count > 0;

        let mut frames = caller_prefix
            .map(|layout| layout.frames[..prefix_frame_count].to_vec())
            .unwrap_or_default();
        // RPython parity: rd_virtuals is stored once on the guard descriptor
        // and never replaced (compile.py:866, resume.py:492). Always preserve
        // caller_prefix's virtual_layouts — they originate from
        // build_guard_metadata and must not be overwritten.
        let mut virtual_layouts = caller_prefix
            .map(|layout| layout.virtual_layouts.clone())
            .unwrap_or_default();
        let mut pending_field_layouts = if preserve_prefix {
            caller_prefix
                .map(|layout| layout.pending_field_layouts.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let virtual_offset = virtual_layouts.len();

        frames.extend(
            self.frame_layouts
                .iter()
                .map(|frame| resume_frame_layout_to_exit_frame_layout(frame, virtual_offset)),
        );
        virtual_layouts.extend(
            self.virtual_layouts
                .iter()
                .map(|virt| resume_virtual_layout_to_exit_virtual_layout(virt, virtual_offset)),
        );
        pending_field_layouts.extend(self.pending_field_layouts.iter().map(|pending| {
            pending_field_layout_to_exit_pending_field_layout(pending, virtual_offset)
        }));

        ExitRecoveryLayout {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames,
            virtual_layouts,
            pending_field_layouts,
        }
    }

    pub fn reconstruct_state(&self, fail_values: &[i64]) -> ReconstructedState {
        let resume_data = self.to_resume_data();
        let virtuals = resume_data.materialize_virtuals(fail_values);
        let pending_fields =
            ResumeData::resolve_pending_field_writes(&resume_data.pending_fields, fail_values);
        let frames = self
            .frame_layouts
            .iter()
            .map(|frame| ReconstructedFrame {
                trace_id: frame.trace_id,
                header_pc: frame.header_pc,
                source_guard: frame.source_guard,
                pc: frame.pc,
                jitcode_index: frame.jitcode_index,
                slot_types: frame.slot_types.clone(),
                values: frame
                    .slot_layouts
                    .iter()
                    .map(|slot| {
                        ResumeData::resolve_frame_slot_source(&slot.to_resume_source(), fail_values)
                    })
                    .collect(),
            })
            .collect();
        ReconstructedState {
            frames,
            virtuals,
            pending_fields,
        }
    }

    pub fn reconstruct(&self, fail_values: &[i64]) -> Vec<ReconstructedFrame> {
        self.reconstruct_state(fail_values).frames
    }

    pub fn reconstruct_frame(
        &self,
        frame_index: usize,
        fail_values: &[i64],
    ) -> Option<ReconstructedFrame> {
        let frame = self.frame_layouts.get(frame_index)?;
        Some(ReconstructedFrame {
            trace_id: frame.trace_id,
            header_pc: frame.header_pc,
            source_guard: frame.source_guard,
            pc: frame.pc,
            jitcode_index: frame.jitcode_index,
            slot_types: frame.slot_types.clone(),
            values: frame
                .slot_layouts
                .iter()
                .map(|slot| {
                    ResumeData::resolve_frame_slot_source(&slot.to_resume_source(), fail_values)
                })
                .collect(),
        })
    }

    pub fn materialize_virtuals(&self, fail_values: &[i64]) -> Vec<MaterializedVirtual> {
        self.to_resume_data().materialize_virtuals(fail_values)
    }

    pub fn resolve_pending_field_writes(
        &self,
        fail_values: &[i64],
    ) -> Vec<ResolvedPendingFieldWrite> {
        let resume_data = self.to_resume_data();
        ResumeData::resolve_pending_field_writes(&resume_data.pending_fields, fail_values)
    }
}

fn can_inline_tagged(value: i64) -> bool {
    (INLINE_TAGGED_MIN..=INLINE_TAGGED_MAX).contains(&value)
}

// `encode_tagged_source` has been promoted to a method on
// `ResumeDataLoopMemo` (see `ResumeDataLoopMemo::encode_tagged_source`)
// so it can share `self.consts` with `getconst`/`newconst` — matching
// RPython's single `self.consts: list[Const]` pool (resume.py).

/// resume.py tag() — i64 widened variant for rd_numb encoding.
fn tag_i64(value: i64, tagbits: u8) -> i64 {
    debug_assert!(tagbits <= 3);
    debug_assert!(
        can_inline_tagged(value),
        "tagged resume value {value} exceeds inline range"
    );
    (value << 2) | tagbits as i64
}

/// resume.py untag() — i64 widened variant for rd_numb decoding.
fn untag_i64(encoded: i64) -> (i64, u8) {
    ((encoded >> 2), (encoded & TAGMASK_I64) as u8)
}

fn encode_len(value: usize) -> i64 {
    i64::try_from(value).expect("resume length exceeds i64")
}

fn decode_len(value: i64) -> usize {
    usize::try_from(value).expect("negative or oversized resume length")
}

fn encode_u64(value: u64) -> i64 {
    value as i64
}

fn decode_u64(value: i64) -> u64 {
    value as u64
}

// FrameInfo moved to majit-backend::resume_value (Phase C-1 cascade).
// Re-exported so existing crate::resume::FrameInfo references stay
// resolvable.
pub use majit_backend::FrameInfo;

// ResumeData moved to majit-backend::resume_value (Phase C-1 cascade);
// re-exported for caller compatibility.  Inherent impl methods are
// provided by ResumeDataExt trait declared below.
pub use majit_backend::ResumeData;

// ResumeValueSource moved to majit-backend::resume_value (Phase C-1
// cascade) so the resume-data type chain can live in a
// backend-accessible crate.  Re-export so existing
// crate::resume::ResumeValueSource references stay resolvable.
pub use majit_backend::ResumeValueLayoutSummaryExt;
pub use majit_backend::ResumeValueSource;

// FrameSlotSource type alias re-exported from majit-backend
// (Phase C-1 cascade) alongside FrameInfo.
pub use majit_backend::FrameSlotSource;

// VirtualInfo moved to majit-backend::resume_value (Phase C-1
// cascade) along with its PartialEq + first impl block
// (field_sources / kind / layout_summary).  The second impl block
// (allocate / is_about_raw, line ~5577) stays in metainterp since it
// depends on ResumeDataDirectReader + BlackholeAllocator which are
// metainterp-specific.  Re-exported here for caller compatibility.
pub use majit_backend::VirtualInfo;

// VirtualFieldSource type alias re-exported from majit-backend
// (Phase C-1 cascade), same as FrameSlotSource.
pub use majit_backend::VirtualFieldSource;

/// Convert a tagged fieldnum (i16, resume.py encoding) to a VirtualFieldSource.
///
/// resume.py decode_int/decode_ref: tagged values encode where
/// each field value comes from at resume time.
///
/// `consts` is the rd_consts array. `count` is the number of fail_args
/// (used for negative TAGBOX indices). `num_virtuals` is the length of
/// rd_virtuals (used for negative TAGVIRTUAL indices — nested virtuals
/// are numbered negatively by `assign_number_to_virtual`,
/// resume.py, and resolved via Python negative list indexing).
/// All come from the containing ResumeGuardDescr / EncodedResumeData.
pub fn tagged_to_source(
    tagged: i16,
    consts: &[majit_ir::Const],
    count: i32,
    num_virtuals: usize,
) -> VirtualFieldSource {
    if tagged_eq(tagged, UNASSIGNED) {
        return ResumeValueSource::Unavailable;
    }
    if tagged_eq(tagged, UNINITIALIZED_TAG) {
        return ResumeValueSource::Uninitialized;
    }
    if tagged_eq(tagged, NULLREF) {
        // history.py CONST_NULL = ConstPtr(null). resume.py:1589 parity.
        return ResumeValueSource::Constant(majit_ir::Const::Ref(majit_ir::GcRef::NULL));
    }
    let (num, tag_bits) = untag(tagged);
    match tag_bits {
        TAGCONST => {
            let idx = (num - TAG_CONST_OFFSET) as usize;
            if idx < consts.len() {
                // resume.py self.consts[num - TAG_CONST_OFFSET] — the
                // Const object carries its type (ConstInt/ConstFloat/ConstPtr).
                ResumeValueSource::Constant(consts[idx])
            } else {
                ResumeValueSource::Constant(majit_ir::Const::Int(0))
            }
        }
        // resume.py ConstInt(num) — always Int type for TAGINT.
        TAGINT => ResumeValueSource::Constant(majit_ir::Const::Int(num as i64)),
        TAGBOX => {
            let mut idx = num;
            if idx < 0 {
                idx += count;
            }
            ResumeValueSource::FailArg(idx as usize)
        }
        TAGVIRTUAL => {
            let mut idx = num;
            if idx < 0 {
                idx += num_virtuals as i32;
            }
            ResumeValueSource::Virtual(idx as usize)
        }
        _ => ResumeValueSource::Unavailable,
    }
}

/// Convert an `RdVirtualInfo` (IR-level, from compile.rs/pyjitpl.rs)
/// to a `VirtualInfo` (resume-level, used by ResumeDataDirectReader).
///
/// Keep `fieldnums` tagged until `ResumeDataDirectReader` consumes them.
/// RPython's `AbstractVirtualInfo.fieldnums` has exactly this shape: a
/// TAGCONST is resolved through the live `rd_consts` list only after any
/// allocation-triggered collection has forwarded that list.
///
/// The numbering sentinels are not values and must remain semantic markers:
/// RPython tests `UNINITIALIZED` before decoding string fields, while pyre's
/// `VirtualInfo` also uses `Unavailable` for an unassigned field.
fn rd_virtual_field_source(tagged: i16) -> VirtualFieldSource {
    if tagged_eq(tagged, UNINITIALIZED_TAG) {
        ResumeValueSource::Uninitialized
    } else if tagged_eq(tagged, UNASSIGNED) || tagged_eq(tagged, UNASSIGNEDVIRTUAL) {
        ResumeValueSource::Unavailable
    } else {
        ResumeValueSource::Tagged(tagged)
    }
}

pub fn rd_virtual_to_virtual_info(
    rd: &majit_ir::RdVirtualInfo,
    _consts: &[majit_ir::Const],
    _count: i32,
    _num_virtuals: usize,
) -> VirtualInfo {
    virtual_info_from_rd(rd)
}

/// `resume.py` `ResumeDataVirtualAdder._number_virtuals` stores
/// `storage.rd_virtuals` once. Resume reads that list
/// (`AbstractResumeDataReader._prepare_virtuals`).
pub(crate) enum GuardVirtualInfos<'a> {
    Cached(&'a [VirtualInfo]),
    Fresh(Vec<VirtualInfo>),
}

impl GuardVirtualInfos<'_> {
    pub(crate) fn as_slice(&self) -> &[VirtualInfo] {
        match self {
            Self::Cached(slice) => slice,
            Self::Fresh(infos) => infos.as_slice(),
        }
    }
}

fn shaped_virtual_infos<'a>(fd: &'a dyn majit_ir::FailDescr) -> Option<&'a [VirtualInfo]> {
    let infos = fd
        .shaped_virtuals_any()?
        .downcast_ref::<Vec<VirtualInfo>>()?;
    if infos.is_empty() {
        None
    } else {
        Some(infos.as_slice())
    }
}

/// The guard's decoded virtual infos. The first resume that sees a
/// non-empty `rd_virtuals` stores the decoding on the descr; later
/// resumes borrow it. A descr that cannot store the box returns
/// [`GuardVirtualInfos::Fresh`].
pub(crate) fn guard_virtual_infos<'a>(
    fd: &'a dyn majit_ir::FailDescr,
) -> Option<GuardVirtualInfos<'a>> {
    if let Some(infos) = shaped_virtual_infos(fd) {
        return Some(GuardVirtualInfos::Cached(infos));
    }
    let built = {
        match fd.rd_virtuals() {
            Some(rds) if !rds.is_empty() => rds
                .iter()
                .map(|rd| virtual_info_from_rd(rd))
                .collect::<Vec<_>>(),
            _ => return None,
        }
    };
    match fd.cache_shaped_virtuals(Box::new(built)) {
        Ok(()) => {
            let infos = shaped_virtual_infos(fd).expect("shaped virtuals were just stored");
            Some(GuardVirtualInfos::Cached(infos))
        }
        Err(boxed) => match boxed.downcast::<Vec<VirtualInfo>>() {
            Ok(infos) => {
                if infos.is_empty() {
                    None
                } else {
                    Some(GuardVirtualInfos::Fresh(*infos))
                }
            }
            Err(_) => None,
        },
    }
}

/// The conversion itself.
///
/// It reads nothing but `rd`: the tagged `fieldnums` stay tagged for
/// `ResumeDataDirectReader` to resolve against the live pool, so the constant
/// pool, the deadframe width and the virtual count that a reader would need to
/// resolve them play no part here. Callers holding those may go through
/// `rd_virtual_to_virtual_info`, which names them for the reader-shaped call
/// sites; the purity is what lets `guard_virtual_infos` build the
/// result once.
pub fn virtual_info_from_rd(rd: &majit_ir::RdVirtualInfo) -> VirtualInfo {
    match rd {
        majit_ir::RdVirtualInfo::VirtualInfo {
            descr,
            type_id,
            known_class,
            fielddescrs,
            fieldnums,
            descr_size,
        } => {
            // AbstractVirtualStructInfo.setfields is 1:1 fielddescrs[i] /
            // fieldnums[i]. Two flattened leaves can share
            // FieldDescrInfo.index (stable_field_index of the same
            // offset/size/field_type); keying the vec by that hash makes
            // a later find return the first fieldnum for every duplicate.
            let fields = fielddescrs
                .iter()
                .zip(fieldnums.iter())
                .enumerate()
                .map(|(i, (_, &tagged))| (i as u32, rd_virtual_field_source(tagged)))
                .collect();
            VirtualInfo::VirtualObj {
                descr: descr.clone(),
                type_id: *type_id,
                known_class: *known_class,
                fields,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            }
        }
        majit_ir::RdVirtualInfo::VStructInfo {
            typedescr,
            type_id,
            fielddescrs,
            fieldnums,
            descr_size,
        } => {
            let fields = fielddescrs
                .iter()
                .zip(fieldnums.iter())
                .enumerate()
                .map(|(i, (_, &tagged))| (i as u32, rd_virtual_field_source(tagged)))
                .collect();
            VirtualInfo::VStruct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                fields,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            }
        }
        majit_ir::RdVirtualInfo::VArrayInfoClear {
            arraydescr,
            fieldnums,
            ..
        } => {
            let items = fieldnums
                .iter()
                .map(|&tagged| rd_virtual_field_source(tagged))
                .collect();
            VirtualInfo::VArray {
                arraydescr: arraydescr.clone(),
                clear: true,
                items,
            }
        }
        majit_ir::RdVirtualInfo::VArrayInfoNotClear {
            arraydescr,
            fieldnums,
            ..
        } => {
            let items = fieldnums
                .iter()
                .map(|&tagged| rd_virtual_field_source(tagged))
                .collect();
            VirtualInfo::VArray {
                arraydescr: arraydescr.clone(),
                clear: false,
                items,
            }
        }
        majit_ir::RdVirtualInfo::VArrayStructInfo {
            arraydescr,
            size,
            fielddescrs: rd_fielddescrs,
            fieldnums,
            ..
        } => {
            // resume.py: VArrayStructInfo(arraydescr, size, fielddescrs)
            // fieldnums is flat: size * len(fielddescrs) entries
            let num_fields = rd_fielddescrs.len().max(1);
            let mut element_fields = Vec::with_capacity(*size);
            for chunk in fieldnums.chunks(num_fields) {
                // resume.py:754: for j in range(len(self.fielddescrs)):
                let elem: Vec<(u32, VirtualFieldSource)> = chunk
                    .iter()
                    .enumerate()
                    .map(|(j, &tagged)| (j as u32, rd_virtual_field_source(tagged)))
                    .collect();
                element_fields.push(elem);
            }
            while element_fields.len() < *size {
                element_fields.push(vec![]);
            }
            VirtualInfo::VArrayStruct {
                arraydescr: arraydescr.clone(),
                fielddescrs: rd_fielddescrs.clone(),
                element_fields,
            }
        }
        majit_ir::RdVirtualInfo::VRawBufferInfo {
            func,
            size,
            offsets,
            descrs,
            fieldnums,
        } => {
            assert_eq!(offsets.len(), descrs.len());
            assert_eq!(offsets.len(), fieldnums.len());
            let values = fieldnums
                .iter()
                .map(|&tagged| rd_virtual_field_source(tagged))
                .collect();
            VirtualInfo::VRawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values,
            }
        }
        majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
            let parent = fieldnums
                .first()
                .map(|&tagged| rd_virtual_field_source(tagged))
                .unwrap_or(ResumeValueSource::Unavailable);
            VirtualInfo::VRawSlice {
                offset: *offset,
                parent,
            }
        }
        majit_ir::RdVirtualInfo::VStrPlainInfo { fieldnums } => {
            let chars = fieldnums
                .iter()
                .map(|&tagged| rd_virtual_field_source(tagged))
                .collect();
            VirtualInfo::VStrPlain { chars }
        }
        majit_ir::RdVirtualInfo::VStrConcatInfo { fieldnums } => {
            let left = Box::new(rd_virtual_field_source(fieldnums[0]));
            let right = Box::new(rd_virtual_field_source(fieldnums[1]));
            VirtualInfo::VStrConcat { left, right }
        }
        majit_ir::RdVirtualInfo::VStrSliceInfo { fieldnums } => {
            let source = Box::new(rd_virtual_field_source(fieldnums[0]));
            let start = Box::new(rd_virtual_field_source(fieldnums[1]));
            let length = Box::new(rd_virtual_field_source(fieldnums[2]));
            VirtualInfo::VStrSlice {
                source,
                start,
                length,
            }
        }
        majit_ir::RdVirtualInfo::VUniPlainInfo { fieldnums } => {
            let chars = fieldnums
                .iter()
                .map(|&tagged| rd_virtual_field_source(tagged))
                .collect();
            VirtualInfo::VUniPlain { chars }
        }
        majit_ir::RdVirtualInfo::VUniConcatInfo { fieldnums } => {
            let left = Box::new(rd_virtual_field_source(fieldnums[0]));
            let right = Box::new(rd_virtual_field_source(fieldnums[1]));
            VirtualInfo::VUniConcat { left, right }
        }
        majit_ir::RdVirtualInfo::VUniSliceInfo { fieldnums } => {
            let source = Box::new(rd_virtual_field_source(fieldnums[0]));
            let start = Box::new(rd_virtual_field_source(fieldnums[1]));
            let length = Box::new(rd_virtual_field_source(fieldnums[2]));
            VirtualInfo::VUniSlice {
                source,
                start,
                length,
            }
        }
        majit_ir::RdVirtualInfo::Empty => VirtualInfo::Empty,
    }
}

// PendingFieldInfo moved to majit-backend::resume_value (Phase C-1
// cascade) along with its PartialEq and layout_summary impl.
// Re-exported for caller compatibility.
pub use majit_backend::PendingFieldInfo;

/// Concrete pending heap write reconstructed from resume data.
///
/// `resume.py _prepare_pendingfields` parity — RPython
/// hands the live `descr` Arc into `setfield` / `setarrayitem` and
/// they dispatch via `descr.is_pointer_field()` /
/// `descr.is_array_of_pointers()` etc.
#[derive(Debug, Clone)]
pub struct ResolvedPendingFieldWrite {
    /// `resume.py:88 lldescr` — the field/array descriptor itself.
    pub descr: Option<majit_ir::DescrRef>,
    /// Concrete object/array pointer.
    pub target: MaterializedValue,
    /// Concrete value to write.
    pub value: MaterializedValue,
    /// Array item index. `None` means a plain field write.
    pub item_index: Option<usize>,
}

impl PartialEq for ResolvedPendingFieldWrite {
    fn eq(&self, other: &Self) -> bool {
        // `history.py:125 id(descr)` parity — descr identity via Arc::ptr_eq.
        majit_ir::resumedata::opt_descr_arc_ptr_eq(&self.descr, &other.descr)
            && self.target == other.target
            && self.value == other.value
            && self.item_index == other.item_index
    }
}
impl Eq for ResolvedPendingFieldWrite {}

/// Encoded pending field write stored alongside an encoded resume snapshot.
///
/// `resume.py PENDINGFIELDSTRUCT` parity — the encoded form
/// carries `lldescr` (the descriptor object itself) so decoding can
/// hand back a live `Arc<dyn Descr>` via `descr.clone()` rather than
/// rebuilding it through an index lookup
/// (`resume.py cast_base_ptr_to_instance`).
#[derive(Debug, Clone)]
pub struct EncodedPendingFieldWrite {
    /// `resume.py:88 lldescr` — the field/array descriptor itself.
    pub descr: Option<majit_ir::DescrRef>,
    pub target: i64,
    pub value: i64,
    pub item_index: Option<usize>,
}

impl PartialEq for EncodedPendingFieldWrite {
    fn eq(&self, other: &Self) -> bool {
        // `history.py:125 id(descr)` parity — descr identity via Arc::ptr_eq.
        majit_ir::resumedata::opt_descr_arc_ptr_eq(&self.descr, &other.descr)
            && self.target == other.target
            && self.value == other.value
            && self.item_index == other.item_index
    }
}
impl Eq for EncodedPendingFieldWrite {}

impl EncodedResumeData {
    pub fn encode(rd: &ResumeData) -> Self {
        Self::from_semantic(
            &rd.vable_array,
            &rd.vref_array,
            &rd.frames,
            &rd.virtuals,
            &rd.pending_fields,
        )
    }

    /// resume.py number + resume.py finish
    ///
    /// Walks all frames via _number_boxes, assigning compact sequential
    /// TAGBOX numbers to unique liveboxes (resume.py).
    ///
    /// Unlike `ResumeDataLoopMemo::encode_shared`, this is a single-shot
    /// encoder (no cross-guard dedup). It builds a local memo that shares
    /// the same encoding logic so the tagged output is bit-compatible with
    /// the shared path. Used by tests and standalone embedders that don't
    /// carry a memo instance.
    fn from_semantic(
        vable_array: &[ResumeValueSource],
        vref_array: &[ResumeValueSource],
        frames: &[FrameInfo],
        virtuals: &[VirtualInfo],
        pending_fields: &[PendingFieldInfo],
    ) -> Self {
        let mut memo = ResumeDataLoopMemo::new();
        let mut rd_numb = Vec::new();
        // resume.py numb_state.liveboxes — compact TAGBOX numbering state.
        let mut liveboxes: Vec<usize> = Vec::new();
        let mut box_map: indexmap::IndexMap<usize, usize> = indexmap::IndexMap::new();

        // resume.py:234-235: reserve slots for items_resume_section and count.
        rd_numb.push(0); // [0] = items_resume_section (patched later)
        rd_numb.push(0); // [1] = count (patched later)
        rd_numb.push(encode_len(vable_array.len()));
        for source in vable_array {
            let tagged = memo.encode_tagged_source(source, &mut liveboxes, &mut box_map);
            rd_numb.push(tagged);
        }
        // resume.py:243-247: vref_array (pairs).
        assert!(
            vref_array.len().is_multiple_of(2),
            "vref_array must have even length (pairs)"
        );
        rd_numb.push(encode_len(vref_array.len() / 2));
        for source in vref_array {
            let tagged = memo.encode_tagged_source(source, &mut liveboxes, &mut box_map);
            rd_numb.push(tagged);
        }

        // resume.py: per-frame encoding via _number_boxes.
        let mut frame_sizes = Vec::with_capacity(frames.len());
        for frame in frames {
            rd_numb.push(frame.jitcode_index as i64);
            rd_numb.push(encode_u64(frame.pc));
            // resume.py _number_boxes(snapshot_iter, iter_array(snapshot), numb_state)
            for source in &frame.slot_map {
                let tagged = memo.encode_tagged_source(source, &mut liveboxes, &mut box_map);
                rd_numb.push(tagged);
            }
            frame_sizes.push(frame.slot_map.len());
        }

        // compile.py:858 rd_virtuals — stored as live objects, not serialized.
        let rd_virtuals = virtuals.to_vec();

        // resume.py:412-418: visitor_walk_recursive — register virtual field boxes.
        for vinfo in &rd_virtuals {
            for source in vinfo.field_sources() {
                if let ResumeValueSource::FailArg(index) = source {
                    box_map.entry(*index).or_insert_with(|| {
                        let n = liveboxes.len();
                        liveboxes.push(*index);
                        n
                    });
                }
            }
        }

        // resume.py:420-430: walk pending fields — register + encode.
        let rd_pendingfields: Vec<_> = pending_fields
            .iter()
            .map(|pending| EncodedPendingFieldWrite {
                // resume.py:547 lldescr = cast_instance_to_base_ptr(descr) —
                // the encoded form carries the descr itself, not a handle.
                descr: pending.descr.clone(),
                target: memo.encode_tagged_source(&pending.target, &mut liveboxes, &mut box_map),
                value: memo.encode_tagged_source(&pending.value, &mut liveboxes, &mut box_map),
                item_index: pending.item_index,
            })
            .collect();
        let rd_consts = memo.take_consts();

        // resume.py:260: numb_state.patch_current_size(0) → items_resume_section
        rd_numb[0] = encode_len(rd_numb.len());
        // resume.py:464: numb_state.patch(1, len(liveboxes)) → count
        rd_numb[1] = encode_len(liveboxes.len());

        EncodedResumeData {
            rd_numb,
            rd_consts,
            rd_pendingfields,
            rd_virtuals,
            liveboxes,
            frame_sizes,
        }
    }

    /// resume.py AbstractResumeDataReader._init — decode rd_numb.
    fn decode_layout(&self) -> DecodedResumeLayout {
        let mut cursor = 0usize;
        // resume.py:919 items_resume_section
        let items_resume_section = self.next_word(&mut cursor);
        assert_eq!(
            decode_len(items_resume_section),
            self.rd_numb.len(),
            "resume item count mismatch"
        );
        // resume.py self.count — number of liveboxes in the deadframe.
        let _count = decode_len(self.next_word(&mut cursor));

        let vable_count = decode_len(self.next_word(&mut cursor));
        let mut vable_array = Vec::with_capacity(vable_count);
        for _ in 0..vable_count {
            vable_array.push(self.decode_box(self.next_word(&mut cursor)));
        }
        let vref_count = decode_len(self.next_word(&mut cursor));
        let mut vref_array = Vec::with_capacity(vref_count * 2);
        for _ in 0..(vref_count * 2) {
            vref_array.push(self.decode_box(self.next_word(&mut cursor)));
        }
        // resume.py number: per-frame jitcode_index, pc, [tagged_values...].
        // jitcode.get_live_vars_info(pc) is the frame boundary;
        // self.frame_sizes[] stored at encode time is that count.
        let items_resume_len = decode_len(items_resume_section);
        let mut frames = Vec::new();
        let mut frame_idx = 0usize;
        while cursor < items_resume_len {
            let jitcode_index = self.next_word(&mut cursor) as i32;
            let pc = decode_u64(self.next_word(&mut cursor));
            let slot_count = if frame_idx < self.frame_sizes.len() {
                self.frame_sizes[frame_idx]
            } else {
                // Single-frame fallback: consume all remaining items.
                items_resume_len - cursor
            };
            let mut slot_map = Vec::with_capacity(slot_count);
            for _ in 0..slot_count {
                slot_map.push(self.decode_box(self.next_word(&mut cursor)));
            }
            frames.push(FrameInfo {
                jitcode_index,
                pc,
                slot_map,
            });
            frame_idx += 1;
        }

        // compile.py:858 rd_virtuals — live objects, not deserialized from rd_numb.
        let virtuals = self.rd_virtuals.clone();

        assert_eq!(
            cursor,
            self.rd_numb.len(),
            "resume decoder left trailing data"
        );
        // resume.py _prepare_pendingfields — lldescr is restored
        // directly from `PENDINGFIELDSTRUCT.lldescr` via
        // `cast_base_ptr_to_instance(AbstractDescr, lldescr)`; pyre keeps
        // the live `Arc<dyn Descr>` on the encoded record, so decoding is
        // a clone.
        let pending_fields = self
            .rd_pendingfields
            .iter()
            .map(|pending| PendingFieldInfo {
                descr: pending.descr.clone(),
                target: self.decode_box(pending.target),
                value: self.decode_box(pending.value),
                item_index: pending.item_index,
            })
            .collect();
        DecodedResumeLayout {
            vable_array,
            vref_array,
            frames,
            virtuals,
            pending_fields,
        }
    }

    /// resume.py:919 resumecodereader.next_item()
    fn next_word(&self, cursor: &mut usize) -> i64 {
        let word = self
            .rd_numb
            .get(*cursor)
            .copied()
            .expect("truncated encoded resume data");
        *cursor += 1;
        word
    }

    /// resume.py decode_box — decode a tagged value from rd_numb.
    fn decode_box(&self, encoded: i64) -> ResumeValueSource {
        let (value, tag) = untag_i64(encoded);
        match tag {
            // resume.py:1257 ConstInt(num).
            TAGINT => ResumeValueSource::Constant(majit_ir::Const::Int(value)),
            // resume.py self.liveboxes[num] — compact TAGBOX → original FailArg.
            TAGBOX => {
                let compact_idx = decode_len(value);
                let original_idx = self.liveboxes[compact_idx];
                ResumeValueSource::FailArg(original_idx)
            }
            TAGVIRTUAL => ResumeValueSource::Virtual(decode_len(value)),
            TAGCONST => match value {
                // resume.py decode_ref: `if tagged_eq(tagged,
                // NULLREF): return CONST_NULL`. The i64 decoder mirrors
                // the free `decode_box(tagged: i16, ..)`'s NULLREF
                // fast-path so encoder/decoder stay symmetric.
                ENCODED_NULLREF => {
                    ResumeValueSource::Constant(majit_ir::Const::Ref(majit_ir::GcRef::NULL))
                }
                ENCODED_UNINITIALIZED => ResumeValueSource::Uninitialized,
                ENCODED_UNAVAILABLE => ResumeValueSource::Unavailable,
                index if index >= 0 => {
                    // resume.py/1571/1583 self.consts[num - TAG_CONST_OFFSET]
                    // — the Const carries its own type.
                    let c = *self
                        .rd_consts
                        .get(decode_len(index))
                        .expect("resume const pool index out of bounds");
                    ResumeValueSource::Constant(c)
                }
                other => panic!("unknown CONST-tagged resume sentinel {other}"),
            },
            other => panic!("unknown resume tag {other}"),
        }
    }

    /// Decode this encoded snapshot back into a `ResumeData`.
    pub fn decode(&self) -> ResumeData {
        let layout = self.decode_layout();
        ResumeData {
            vable_array: layout.vable_array,
            vref_array: layout.vref_array,
            frames: layout.frames,
            virtuals: layout.virtuals,
            pending_fields: layout.pending_fields,
        }
    }

    /// Return a compact summary of this snapshot's frame/jitframe layout.
    pub fn layout_summary(&self) -> ResumeLayoutSummary {
        let layout = self.decode_layout();
        ResumeLayoutSummary {
            num_frames: layout.frames.len(),
            frame_pcs: Vec::new(),
            frame_slot_counts: Vec::new(),
            frame_layouts: layout
                .frames
                .iter()
                .map(|frame| ResumeFrameLayoutSummary {
                    trace_id: None,
                    header_pc: None,
                    source_guard: None,
                    jitcode_index: frame.jitcode_index,
                    pc: frame.pc,
                    slot_sources: Vec::new(),
                    slot_layouts: frame
                        .slot_map
                        .iter()
                        .map(|source| source.layout_summary())
                        .collect(),
                    slot_types: None,
                })
                .collect(),
            num_virtuals: layout.virtuals.len(),
            virtual_kinds: layout.virtuals.iter().map(VirtualInfo::kind).collect(),
            virtual_layouts: layout
                .virtuals
                .iter()
                .map(|virt| virt.layout_summary())
                .collect(),

            pending_field_count: layout.pending_fields.len(),
            pending_field_layouts: layout
                .pending_fields
                .iter()
                .map(|pending| pending.layout_summary())
                .collect(),
            const_pool_size: self.rd_consts.len(),
        }
    }

    /// Reconstruct the full interpreter state directly from the encoded snapshot.
    pub fn reconstruct_state(&self, fail_values: &[i64]) -> ReconstructedState {
        let layout = self.decode_layout();
        let virtuals = ResumeData::materialize_virtuals_from_infos(&layout.virtuals, fail_values);
        let pending_fields =
            ResumeData::resolve_pending_field_writes(&layout.pending_fields, fail_values);
        let frames = layout
            .frames
            .iter()
            .map(|frame| ReconstructedFrame {
                trace_id: None,
                header_pc: None,
                source_guard: None,
                pc: frame.pc,
                jitcode_index: frame.jitcode_index,
                slot_types: None,
                values: frame
                    .slot_map
                    .iter()
                    .map(|slot| ResumeData::resolve_frame_slot_source(slot, fail_values))
                    .collect(),
            })
            .collect();
        ReconstructedState {
            frames,
            virtuals,
            pending_fields,
        }
    }

    /// Reconstruct only the interpreter frames from the encoded snapshot.
    pub fn reconstruct(&self, fail_values: &[i64]) -> Vec<ReconstructedFrame> {
        self.reconstruct_state(fail_values).frames
    }

    /// Materialize virtual objects referenced by this encoded snapshot.
    pub fn materialize_virtuals(&self, fail_values: &[i64]) -> Vec<MaterializedVirtual> {
        let layout = self.decode_layout();
        ResumeData::materialize_virtuals_from_infos(&layout.virtuals, fail_values)
    }

    /// Resolve pending heap writes referenced by this encoded snapshot.
    pub fn resolve_pending_field_writes(
        &self,
        fail_values: &[i64],
    ) -> Vec<ResolvedPendingFieldWrite> {
        let layout = self.decode_layout();
        ResumeData::resolve_pending_field_writes(&layout.pending_fields, fail_values)
    }
}

/// Metainterp-side extension trait for `ResumeData` (which lives in
/// `majit-backend::resume_value` after the Phase C-1 cascade).
///
/// All inherent methods on `ResumeData` move here as trait methods
/// (with default impls) — the trait sits in metainterp because the
/// methods reference metainterp-specific types (`EncodedResumeData`,
/// `DecodedResumeLayout`, `ReconstructedState`, `ReconstructedFrame`,
/// `MaterializedVirtual`, `MaterializedValue`, `ReconstructedValue`,
/// `ResolvedPendingFieldWrite`) and the orphan rule forbids inherent
/// impls on a foreign type.
///
/// Callers reach the methods through `use ResumeDataExt;` plus the
/// usual receiver syntax (`data.encode()`, `data.reconstruct_state(...)`,
/// etc.) or the static-method syntax (`ResumeData::simple(...)`,
/// `ResumeData::resolve_field_source(...)`) — both work for trait
/// associated items as long as the trait is in scope.
pub trait ResumeDataExt {
    /// Create a simple ResumeData for a single-frame trace.
    fn simple(pc: u64, num_slots: usize) -> Self
    where
        Self: Sized;

    /// Encode this resume snapshot into a compact RPython-style numbering.
    fn encode(&self) -> EncodedResumeData;

    /// Decode the encoded resume snapshot back into a layout summary.
    fn decode_layout(&self) -> DecodedResumeLayout;

    /// Reconstruct the full resume state from fail_args data.
    fn reconstruct_state(&self, fail_values: &[i64]) -> ReconstructedState;

    /// Reconstruct frame slots from fail_args data.
    fn reconstruct(&self, fail_values: &[i64]) -> Vec<ReconstructedFrame>;

    /// Materialize virtual objects from resume data.
    fn materialize_virtuals(&self, fail_values: &[i64]) -> Vec<MaterializedVirtual>;

    fn materialize_virtuals_from_infos(
        virtuals: &[VirtualInfo],
        fail_values: &[i64],
    ) -> Vec<MaterializedVirtual>;

    /// Resolve pending heap writes into concrete values.
    fn resolve_pending_field_writes(
        pending_fields: &[PendingFieldInfo],
        fail_values: &[i64],
    ) -> Vec<ResolvedPendingFieldWrite>;

    /// Resolve a single VirtualFieldSource to a concrete i64 value.
    fn resolve_field_source(source: &VirtualFieldSource, fail_values: &[i64]) -> i64;

    fn resolve_materialized_source(
        source: &VirtualFieldSource,
        fail_values: &[i64],
    ) -> MaterializedValue;

    /// Resolve a single frame-slot source into a reconstructed value.
    fn resolve_frame_slot_source(
        source: &FrameSlotSource,
        fail_values: &[i64],
    ) -> ReconstructedValue;
}

impl ResumeDataExt for ResumeData {
    fn simple(pc: u64, num_slots: usize) -> Self {
        let slot_map: Vec<FrameSlotSource> = (0..num_slots).map(FrameSlotSource::FailArg).collect();
        ResumeData {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames: vec![FrameInfo {
                jitcode_index: 0,
                pc,
                slot_map,
            }],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        }
    }

    fn encode(&self) -> EncodedResumeData {
        EncodedResumeData::from_semantic(
            &self.vable_array,
            &self.vref_array,
            &self.frames,
            &self.virtuals,
            &self.pending_fields,
        )
    }

    fn decode_layout(&self) -> DecodedResumeLayout {
        self.encode().decode_layout()
    }

    fn reconstruct_state(&self, fail_values: &[i64]) -> ReconstructedState {
        let decoded = self.decode_layout();
        let materialized_virtuals = <ResumeData as ResumeDataExt>::materialize_virtuals_from_infos(
            &decoded.virtuals,
            fail_values,
        );
        let frames = decoded
            .frames
            .iter()
            .map(|frame| {
                let values = frame
                    .slot_map
                    .iter()
                    .map(|slot| {
                        <ResumeData as ResumeDataExt>::resolve_frame_slot_source(slot, fail_values)
                    })
                    .collect();
                ReconstructedFrame {
                    trace_id: None,
                    header_pc: None,
                    source_guard: None,
                    pc: frame.pc,
                    jitcode_index: frame.jitcode_index,
                    slot_types: None,
                    values,
                }
            })
            .collect();
        ReconstructedState {
            frames,
            virtuals: materialized_virtuals,
            pending_fields: <ResumeData as ResumeDataExt>::resolve_pending_field_writes(
                &decoded.pending_fields,
                fail_values,
            ),
        }
    }

    fn reconstruct(&self, fail_values: &[i64]) -> Vec<ReconstructedFrame> {
        self.reconstruct_state(fail_values).frames
    }

    fn materialize_virtuals(&self, fail_values: &[i64]) -> Vec<MaterializedVirtual> {
        let decoded = self.decode_layout();
        <ResumeData as ResumeDataExt>::materialize_virtuals_from_infos(
            &decoded.virtuals,
            fail_values,
        )
    }

    fn materialize_virtuals_from_infos(
        virtuals: &[VirtualInfo],
        fail_values: &[i64],
    ) -> Vec<MaterializedVirtual> {
        let mut result = Vec::with_capacity(virtuals.len());
        for vinfo in virtuals {
            result.push(MaterializedVirtual::from_info(vinfo));
        }
        for (i, vinfo) in virtuals.iter().enumerate() {
            result[i].resolve_fields(vinfo, fail_values);
        }
        result
    }

    fn resolve_pending_field_writes(
        pending_fields: &[PendingFieldInfo],
        fail_values: &[i64],
    ) -> Vec<ResolvedPendingFieldWrite> {
        pending_fields
            .iter()
            .map(|pending| ResolvedPendingFieldWrite {
                descr: pending.descr.clone(),
                target: <ResumeData as ResumeDataExt>::resolve_materialized_source(
                    &pending.target,
                    fail_values,
                ),
                value: <ResumeData as ResumeDataExt>::resolve_materialized_source(
                    &pending.value,
                    fail_values,
                ),
                item_index: pending.item_index,
            })
            .collect()
    }

    fn resolve_field_source(source: &VirtualFieldSource, fail_values: &[i64]) -> i64 {
        match <ResumeData as ResumeDataExt>::resolve_materialized_source(source, fail_values) {
            MaterializedValue::Value(value) => value,
            MaterializedValue::VirtualRef(_) => 0,
        }
    }

    fn resolve_materialized_source(
        source: &VirtualFieldSource,
        fail_values: &[i64],
    ) -> MaterializedValue {
        match source {
            ResumeValueSource::FailArg(idx) => {
                MaterializedValue::Value(fail_values.get(*idx).copied().unwrap_or(0))
            }
            ResumeValueSource::Constant(c) => MaterializedValue::Value(c.as_raw_i64()),
            ResumeValueSource::Virtual(idx) => MaterializedValue::VirtualRef(*idx),
            ResumeValueSource::Tagged(_) => {
                panic!("runtime tagged source requires ResumeDataDirectReader")
            }
            ResumeValueSource::Uninitialized | ResumeValueSource::Unavailable => {
                MaterializedValue::Value(0)
            }
        }
    }

    fn resolve_frame_slot_source(
        source: &FrameSlotSource,
        fail_values: &[i64],
    ) -> ReconstructedValue {
        match source {
            ResumeValueSource::FailArg(idx) => {
                ReconstructedValue::Value(fail_values.get(*idx).copied().unwrap_or(0))
            }
            ResumeValueSource::Constant(c) => ReconstructedValue::Value(c.as_raw_i64()),
            ResumeValueSource::Virtual(idx) => ReconstructedValue::Virtual(*idx),
            ResumeValueSource::Tagged(_) => {
                panic!("runtime tagged source requires ResumeDataDirectReader")
            }
            ResumeValueSource::Uninitialized => ReconstructedValue::Uninitialized,
            ResumeValueSource::Unavailable => ReconstructedValue::Unavailable,
        }
    }
}

/// A reconstructed interpreter frame from resume data.
#[derive(Debug, Clone)]
pub struct ReconstructedFrame {
    /// Compiled trace identifier for this frame, when known.
    pub trace_id: Option<u64>,
    /// Trace header pc associated with this frame, when known.
    pub header_pc: Option<u64>,
    /// Source guard this frame's trace is attached to, when known.
    pub source_guard: Option<(u64, u32)>,
    /// Program counter for this frame.
    pub pc: u64,
    /// resume.py:1051: jitcode index for CodeObject lookup.
    pub jitcode_index: i32,
    /// Typed layout of the reconstructed slots, when known.
    pub slot_types: Option<Vec<Type>>,
    /// Reconstructed values for each slot.
    pub values: Vec<ReconstructedValue>,
}

impl ReconstructedFrame {
    /// Lossy conversion: extract integer values, dropping virtual/unavailable info.
    pub fn lossy_values(&self) -> Vec<i64> {
        self.values
            .iter()
            .map(ReconstructedValue::lossy_i64)
            .collect()
    }
}

/// Full reconstructed state for a guard recovery.
#[derive(Debug, Clone)]
pub struct ReconstructedState {
    /// Reconstructed interpreter frames, outermost first.
    pub frames: Vec<ReconstructedFrame>,
    /// Materialized virtual objects referenced by frame slots.
    pub virtuals: Vec<MaterializedVirtual>,
    /// Deferred heap writes that the interpreter must replay after reconstruction.
    pub pending_fields: Vec<ResolvedPendingFieldWrite>,
}

/// Reconstructed slot value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconstructedValue {
    /// Concrete raw value, including ints, refs, and float bits.
    Value(i64),
    /// Reference to a materialized virtual in `ReconstructedState.virtuals`.
    Virtual(usize),
    /// Slot exists but remains uninitialized.
    Uninitialized,
    /// Slot is dead/unavailable at this guard.
    Unavailable,
}

impl ReconstructedValue {
    /// Lossy conversion used by the current integer-only compatibility layer.
    pub fn lossy_i64(&self) -> i64 {
        match self {
            ReconstructedValue::Value(value) => *value,
            ReconstructedValue::Virtual(_)
            | ReconstructedValue::Uninitialized
            | ReconstructedValue::Unavailable => 0,
        }
    }
}

/// A materialized virtual object, ready for the interpreter to allocate.
///
/// After a guard failure, virtual objects must be allocated on the heap
/// and their fields populated from the DeadFrame values. This struct
/// holds the resolved field values for a single virtual object.
///
/// Mirrors RPython's `_materialize_virtual()` in resume.py.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaterializedValue {
    Value(i64),
    VirtualRef(usize),
}

impl MaterializedValue {
    pub fn resolve_with_refs(&self, materialized_refs: &[Option<GcRef>]) -> Option<i64> {
        match self {
            MaterializedValue::Value(value) => Some(*value),
            MaterializedValue::VirtualRef(index) => materialized_refs
                .get(*index)
                .copied()
                .flatten()
                .map(|gc_ref| gc_ref.as_usize() as i64),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MaterializedVirtual {
    /// Object with vtable — resume.py VirtualInfo.
    /// Carries `descr` (resume.py self.descr) so the deopt path can
    /// `allocate_with_vtable(descr=self.descr)` and replay fields generically,
    /// without special-casing the vtable at the JIT-state layer.
    Obj {
        /// SizeDescr for allocation (exposes vtable + obj_size).
        descr: Option<majit_ir::DescrRef>,
        type_id: u32,
        /// (field_descr_index, concrete_value).
        fields: Vec<(u32, MaterializedValue)>,
    },
    /// Plain struct — resume.py VStructInfo.
    Struct {
        /// resume.py:631 self.typedescr.
        descr: Option<majit_ir::DescrRef>,
        type_id: u32,
        fields: Vec<(u32, MaterializedValue)>,
    },
    /// Array — resume.py:646 VArrayInfo*.
    Array {
        /// resume.py:646 self.arraydescr.
        descr: Option<majit_ir::DescrRef>,
        items: Vec<MaterializedValue>,
    },
    /// Array of structs — resume.py VArrayStructInfo.
    ArrayStruct {
        /// resume.py:739 self.arraydescr.
        descr: Option<majit_ir::DescrRef>,
        /// Per-element: Vec<(field_index, value)>.
        elements: Vec<Vec<(u32, MaterializedValue)>>,
    },
    /// Raw buffer.
    RawBuffer {
        func: i64,
        size: usize,
        /// rawbuffer.py __init__ stores offsets as RPython unbounded ints.
        offsets: Vec<i64>,
        descrs: Vec<majit_ir::ArrayDescrInfo>,
        values: Vec<MaterializedValue>,
    },
}

impl MaterializedVirtual {
    /// Create an empty shell from a VirtualInfo (forward-reference safe).
    fn from_info(info: &VirtualInfo) -> Self {
        match info {
            VirtualInfo::VirtualObj { descr, type_id, .. } => MaterializedVirtual::Obj {
                descr: descr.clone(),
                type_id: *type_id,
                fields: Vec::new(),
            },
            VirtualInfo::VStruct {
                typedescr, type_id, ..
            } => MaterializedVirtual::Struct {
                descr: typedescr.clone(),
                type_id: *type_id,
                fields: Vec::new(),
            },
            VirtualInfo::VArray {
                arraydescr,
                clear: _,
                items,
                ..
            } => MaterializedVirtual::Array {
                descr: arraydescr.clone(),
                items: vec![MaterializedValue::Value(0); items.len()],
            },
            VirtualInfo::VArrayStruct {
                arraydescr,
                fielddescrs: _,
                element_fields,
                ..
            } => MaterializedVirtual::ArrayStruct {
                descr: arraydescr.clone(),
                elements: vec![Vec::new(); element_fields.len()],
            },
            VirtualInfo::VRawBuffer {
                func,
                size,
                offsets,
                descrs,
                ..
            } => MaterializedVirtual::RawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values: vec![MaterializedValue::Value(0); offsets.len()],
            },
            VirtualInfo::VRawSlice { .. } => MaterializedVirtual::Struct {
                descr: None,
                type_id: 0,
                fields: Vec::new(),
            },
            // resume.py VStrPlainInfo VStr/VUni*Info — virtual string shells
            // reserved for future vstring.py port. Represented as struct
            // shells for now (zero fields) so the materializer doesn't
            // walk into them; actual allocate_string / string_setitem /
            // concat_strings / slice_string still live on the roadmap.
            VirtualInfo::VStrPlain { .. }
            | VirtualInfo::VStrConcat { .. }
            | VirtualInfo::VStrSlice { .. }
            | VirtualInfo::VUniPlain { .. }
            | VirtualInfo::VUniConcat { .. }
            | VirtualInfo::VUniSlice { .. } => MaterializedVirtual::Struct {
                descr: None,
                type_id: 0,
                fields: Vec::new(),
            },
            VirtualInfo::Empty => panic!("resume.py: null rd_virtuals[index]"),
        }
    }

    /// Resolve fields from fail_values.
    fn resolve_fields(&mut self, info: &VirtualInfo, fail_values: &[i64]) {
        match (self, info) {
            (
                MaterializedVirtual::Obj { fields, .. },
                VirtualInfo::VirtualObj {
                    fields: src_fields, ..
                },
            )
            | (
                MaterializedVirtual::Struct { fields, .. },
                VirtualInfo::VStruct {
                    fields: src_fields, ..
                },
            ) => {
                *fields = src_fields
                    .iter()
                    .map(|(idx, src)| {
                        (
                            *idx,
                            ResumeData::resolve_materialized_source(src, fail_values),
                        )
                    })
                    .collect();
            }
            (
                MaterializedVirtual::Array { items, .. },
                VirtualInfo::VArray {
                    items: src_items, ..
                },
            ) => {
                *items = src_items
                    .iter()
                    .map(|src| ResumeData::resolve_materialized_source(src, fail_values))
                    .collect();
            }
            (
                MaterializedVirtual::ArrayStruct { elements, .. },
                VirtualInfo::VArrayStruct {
                    element_fields: src_elems,
                    ..
                },
            ) => {
                *elements = src_elems
                    .iter()
                    .map(|elem_fields| {
                        elem_fields
                            .iter()
                            .map(|(idx, src)| {
                                (
                                    *idx,
                                    ResumeData::resolve_materialized_source(src, fail_values),
                                )
                            })
                            .collect()
                    })
                    .collect();
            }
            (
                MaterializedVirtual::RawBuffer { values, .. },
                VirtualInfo::VRawBuffer {
                    values: src_values, ..
                },
            ) => {
                *values = src_values
                    .iter()
                    .map(|src| ResumeData::resolve_materialized_source(src, fail_values))
                    .collect();
            }
            _ => {} // type mismatch — should not happen
        }
    }

    pub fn resolve_with_refs(
        &self,
        materialized_refs: &[Option<GcRef>],
    ) -> Option<MaterializedVirtual> {
        match self {
            MaterializedVirtual::Obj {
                descr,
                type_id,
                fields,
            } => Some(MaterializedVirtual::Obj {
                descr: descr.clone(),
                type_id: *type_id,
                fields: fields
                    .iter()
                    .map(|(idx, value)| {
                        Some((
                            *idx,
                            MaterializedValue::Value(value.resolve_with_refs(materialized_refs)?),
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
            MaterializedVirtual::Struct {
                descr,
                type_id,
                fields,
            } => Some(MaterializedVirtual::Struct {
                descr: descr.clone(),
                type_id: *type_id,
                fields: fields
                    .iter()
                    .map(|(idx, value)| {
                        Some((
                            *idx,
                            MaterializedValue::Value(value.resolve_with_refs(materialized_refs)?),
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
            MaterializedVirtual::Array { descr, items } => Some(MaterializedVirtual::Array {
                descr: descr.clone(),
                items: items
                    .iter()
                    .map(|value| {
                        Some(MaterializedValue::Value(
                            value.resolve_with_refs(materialized_refs)?,
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
            MaterializedVirtual::ArrayStruct { descr, elements } => {
                Some(MaterializedVirtual::ArrayStruct {
                    descr: descr.clone(),
                    elements: elements
                        .iter()
                        .map(|fields| {
                            fields
                                .iter()
                                .map(|(idx, value)| {
                                    Some((
                                        *idx,
                                        MaterializedValue::Value(
                                            value.resolve_with_refs(materialized_refs)?,
                                        ),
                                    ))
                                })
                                .collect::<Option<Vec<_>>>()
                        })
                        .collect::<Option<Vec<_>>>()?,
                })
            }
            MaterializedVirtual::RawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => Some(MaterializedVirtual::RawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values: values
                    .iter()
                    .map(|value| {
                        Some(MaterializedValue::Value(
                            value.resolve_with_refs(materialized_refs)?,
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
        }
    }
}

/// Builder for constructing ResumeData during trace compilation.
///
/// resume.py ResumeDataVirtualAdder.finish() is implemented
/// across two functions in majit:
/// - `store_final_boxes_in_guard` (mod.rs) — numbering + rd_numb/rd_consts
/// - `store_final_boxes_in_guard` (optimizer.rs) — virtual expansion + rd_virtuals
pub struct ResumeDataVirtualAdder {
    vable_array: Vec<ResumeValueSource>,
    vref_array: Vec<ResumeValueSource>,
    frames: Vec<FrameInfoBuilder>,
    virtuals: Vec<VirtualInfo>,
    pending_fields: Vec<PendingFieldInfo>,
}

struct FrameInfoBuilder {
    jitcode_index: i32,
    pc: u64,
    slot_map: Vec<FrameSlotSource>,
}

impl ResumeDataVirtualAdder {
    /// Create a new builder.
    pub fn new() -> Self {
        ResumeDataVirtualAdder {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames: Vec::new(),
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        }
    }

    pub fn set_vable_array(&mut self, values: Vec<ResumeValueSource>) {
        self.vable_array = values;
    }

    /// Push a new frame onto the stack.
    /// resume.py `ResumeDataLoopMemo.number`: jitcode_index, pc per frame.
    pub fn push_frame(&mut self, jitcode_index: i32, pc: u64) {
        self.frames.push(FrameInfoBuilder {
            jitcode_index,
            pc,
            slot_map: Vec::new(),
        });
    }

    /// Map a slot in the current frame to a fail_arg index.
    pub fn map_slot(&mut self, slot_idx: usize, fail_arg_idx: usize) {
        self.set_slot_source(slot_idx, FrameSlotSource::FailArg(fail_arg_idx));
    }

    /// Set a slot in the current frame to a tagged source.
    pub fn set_slot_source(&mut self, slot_idx: usize, source: FrameSlotSource) {
        let frame = self.frames.last_mut().expect("no frame pushed");
        while frame.slot_map.len() <= slot_idx {
            frame.slot_map.push(FrameSlotSource::Unavailable);
        }
        frame.slot_map[slot_idx] = source;
    }

    /// Set a frame slot to a compile-time constant.
    pub fn set_slot_constant(&mut self, slot_idx: usize, constant: majit_ir::Const) {
        self.set_slot_source(slot_idx, FrameSlotSource::Constant(constant));
    }

    /// Set a frame slot to reference a virtual object.
    pub fn set_slot_virtual(&mut self, slot_idx: usize, virtual_idx: usize) {
        self.set_slot_source(slot_idx, FrameSlotSource::Virtual(virtual_idx));
    }

    /// Mark a frame slot as present but uninitialized.
    pub fn set_slot_uninitialized(&mut self, slot_idx: usize) {
        self.set_slot_source(slot_idx, FrameSlotSource::Uninitialized);
    }

    /// Mark a frame slot as dead/unavailable.
    pub fn set_slot_unavailable(&mut self, slot_idx: usize) {
        self.set_slot_source(slot_idx, FrameSlotSource::Unavailable);
    }

    /// Add a virtual object description. Returns the index in the virtuals array.
    pub fn add_virtual(&mut self, info: VirtualInfo) -> usize {
        let idx = self.virtuals.len();
        self.virtuals.push(info);
        idx
    }

    /// Convenience: add a virtual object (NEW_WITH_VTABLE).
    pub fn add_virtual_obj(
        &mut self,
        descr: Option<majit_ir::DescrRef>,
        type_id: u32,
        known_class: Option<i64>,
        fields: Vec<(u32, VirtualFieldSource)>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    ) -> usize {
        self.add_virtual(VirtualInfo::VirtualObj {
            descr,
            type_id,
            known_class,
            fields,
            fielddescrs,
            descr_size,
        })
    }

    /// Convenience: add a virtual struct (NEW).
    pub fn add_virtual_struct(
        &mut self,
        typedescr: Option<majit_ir::DescrRef>,
        type_id: u32,
        fields: Vec<(u32, VirtualFieldSource)>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    ) -> usize {
        self.add_virtual(VirtualInfo::VStruct {
            typedescr,
            type_id,
            fields,
            fielddescrs,
            descr_size,
        })
    }

    /// Convenience: add a virtual array (NEW_ARRAY).
    pub fn add_virtual_array(
        &mut self,
        arraydescr: Option<majit_ir::DescrRef>,
        clear: bool,
        items: Vec<VirtualFieldSource>,
    ) -> usize {
        self.add_virtual(VirtualInfo::VArray {
            arraydescr,
            clear,
            items,
        })
    }

    /// resume.py: visit_varraystruct(arraydescr, size, fielddescrs)
    ///                 → VArrayStructInfo(arraydescr, size, fielddescrs)
    pub fn add_virtual_array_struct(
        &mut self,
        arraydescr: Option<majit_ir::DescrRef>,
        fielddescrs: Vec<majit_ir::DescrRef>,
        element_fields: Vec<Vec<(u32, VirtualFieldSource)>>,
    ) -> usize {
        self.add_virtual(VirtualInfo::VArrayStruct {
            arraydescr,
            fielddescrs,
            element_fields,
        })
    }

    /// Convenience: add a virtual raw buffer.
    pub fn add_virtual_raw_buffer(
        &mut self,
        func: i64,
        size: usize,
        offsets: Vec<i64>,
        descrs: Vec<majit_ir::ArrayDescrInfo>,
        values: Vec<VirtualFieldSource>,
    ) -> usize {
        self.add_virtual(VirtualInfo::VRawBuffer {
            func,
            size,
            offsets,
            descrs,
            values,
        })
    }

    /// Add a deferred field write to replay on resume.
    ///
    /// `resume.py PENDINGFIELDSTRUCT.lldescr` — RPython always
    /// captures a live descr off the originating SetfieldGc op.
    pub fn add_pending_field_write(
        &mut self,
        descr: Option<majit_ir::DescrRef>,
        target: ResumeValueSource,
        value: ResumeValueSource,
    ) {
        self.pending_fields.push(PendingFieldInfo {
            descr,
            target,
            value,
            item_index: None,
        });
    }

    /// Add a deferred array item write to replay on resume.
    pub fn add_pending_arrayitem_write(
        &mut self,
        descr: Option<majit_ir::DescrRef>,
        target: ResumeValueSource,
        item_index: usize,
        value: ResumeValueSource,
    ) {
        self.pending_fields.push(PendingFieldInfo {
            descr,
            target,
            value,
            item_index: Some(item_index),
        });
    }

    /// resume.py: visit_vrawslice(offset) — add virtual raw slice.
    pub fn add_virtual_raw_slice(&mut self, offset: i64, parent: VirtualFieldSource) -> usize {
        self.add_virtual(VirtualInfo::VRawSlice { offset, parent })
    }

    /// Build the final ResumeData.
    pub fn build(self) -> ResumeData {
        ResumeData {
            vable_array: self.vable_array,
            vref_array: self.vref_array,
            frames: self
                .frames
                .into_iter()
                .map(|f| FrameInfo {
                    jitcode_index: f.jitcode_index,
                    pc: f.pc,
                    slot_map: f.slot_map,
                })
                .collect(),
            virtuals: self.virtuals,
            pending_fields: self.pending_fields,
        }
    }
}

impl Default for ResumeDataVirtualAdder {
    fn default() -> Self {
        Self::new()
    }
}

/// A handle to a `ResumeDataLoopMemo` some in-flight `Optimizer` still owns.
///
/// `Weak`, so a dropped `Optimizer` needs no unregister hook; the root walk
/// drops the handles that no longer upgrade.
pub type LiveResumeMemo = std::rc::Weak<std::cell::RefCell<ResumeDataLoopMemo>>;

/// Shared resume data storage that deduplicates common snapshot sections
/// across multiple guards in the same trace.
///
/// RPython's `ResumeDataLoopMemo` shares constant pools and frame sections
/// across guards. We use a shared `ResumeEncoder` state so that the same
/// large constant only appears once in the pool.
/// RPython resume.py ResumeDataLoopMemo.
/// Shared constant pool + box numbering cache across all guards in a loop.
///
/// NOTE: RPython's ResumeDataLoopMemo also stores `metainterp_sd` and `cpu`
/// (for box allocation during rebuild). pyre doesn't need these: the BoxEnv
/// trait provides box access. The canonical decoder lives in
/// `majit_ir::resumedata::rebuild_from_numbering`.
/// RPython's nvirtuals/nvholes/nvreused stats are kept for monitoring.
pub struct ResumeDataLoopMemo {
    /// resume.py:147 — shared constant pool.
    /// RPython stores Const objects (with type INT/REF/FLOAT).
    /// We store (value, type) pairs to preserve type information.
    consts: Arc<majit_ir::SharedConstPool>,
    /// resume.py — large integers (outside TAGINT range) → tagged const.
    large_ints: indexmap::IndexMap<i64, i16>,
    /// resume.py:149 — ref pointers → tagged const.
    refs: indexmap::IndexMap<i64, i16>,
    /// resume.py:147 self.consts — constant pool for encode_shared.
    /// Becomes storage.rd_consts (resume.py:467).
    ///
    /// resume.py:150-151 — cached box/virtual numbering.
    pub cached_boxes: indexmap::IndexMap<majit_ir::operand::Operand, i32>,
    pub cached_virtuals: indexmap::IndexMap<majit_ir::operand::Operand, i32>,
    /// resume.py:153-155 — statistics.
    pub nvirtuals: usize,
    pub nvholes: usize,
    pub nvreused: usize,
    /// Reused by `number_slices` / `finish` so timed-section guards do not
    /// mint a fresh writer `Vec` and livebox hole list on every bridge.
    writer_scratch: Vec<i32>,
    livebox_opt_scratch: Vec<Option<majit_ir::operand::Operand>>,
    ordered_livebox_scratch: Vec<majit_ir::operand::Operand>,
    livebox_map_scratch: LiveboxMap,
    new_livebox_map_scratch: LiveboxMap,
    virtual_fields_scratch: indexmap::IndexMap<majit_ir::OpRef, majit_ir::VirtualFieldsInfo>,
    virtual_worklist_scratch: Vec<majit_ir::OpRef>,
}

impl ResumeDataLoopMemo {
    pub fn new() -> Self {
        ResumeDataLoopMemo {
            consts: majit_ir::SharedConstPool::new(Vec::new()),
            large_ints: indexmap::IndexMap::new(),
            refs: indexmap::IndexMap::new(),
            cached_boxes: indexmap::IndexMap::new(),
            cached_virtuals: indexmap::IndexMap::new(),
            nvirtuals: 0,
            nvholes: 0,
            nvreused: 0,
            writer_scratch: Vec::new(),
            livebox_opt_scratch: Vec::new(),
            ordered_livebox_scratch: Vec::new(),
            livebox_map_scratch: LiveboxMap::new(),
            new_livebox_map_scratch: LiveboxMap::new(),
            virtual_fields_scratch: indexmap::IndexMap::new(),
            virtual_worklist_scratch: Vec::new(),
        }
    }

    /// Drop per-compile caches (`resume.py ResumeDataLoopMemo.__init__`)
    /// but keep the writer / livebox scratch capacities.
    pub fn recycle_for_next_compile(&mut self) {
        self.consts = majit_ir::SharedConstPool::new(Vec::new());
        self.large_ints.clear();
        self.refs.clear();
        self.cached_boxes.clear();
        self.cached_virtuals.clear();
        self.nvirtuals = 0;
        self.nvholes = 0;
        self.nvreused = 0;
        self.livebox_map_scratch.clear();
        self.new_livebox_map_scratch.clear();
        self.virtual_fields_scratch.clear();
        self.virtual_worklist_scratch.clear();
    }

    fn take_livebox_map(&mut self) -> LiveboxMap {
        let mut map = std::mem::take(&mut self.livebox_map_scratch);
        map.clear();
        map
    }

    fn return_livebox_map(&mut self, mut map: LiveboxMap) {
        map.clear();
        if map.entries.capacity() > self.livebox_map_scratch.entries.capacity() {
            self.livebox_map_scratch = map;
        }
    }

    fn take_new_livebox_map(&mut self) -> LiveboxMap {
        let mut map = std::mem::take(&mut self.new_livebox_map_scratch);
        map.clear();
        map
    }

    fn return_new_livebox_map(&mut self, mut map: LiveboxMap) {
        map.clear();
        if map.entries.capacity() > self.new_livebox_map_scratch.entries.capacity() {
            self.new_livebox_map_scratch = map;
        }
    }

    fn take_virtual_fields(
        &mut self,
    ) -> indexmap::IndexMap<majit_ir::OpRef, majit_ir::VirtualFieldsInfo> {
        let mut fields = std::mem::take(&mut self.virtual_fields_scratch);
        fields.clear();
        fields
    }

    fn return_virtual_fields(
        &mut self,
        mut fields: indexmap::IndexMap<majit_ir::OpRef, majit_ir::VirtualFieldsInfo>,
    ) {
        fields.clear();
        if fields.capacity() > self.virtual_fields_scratch.capacity() {
            self.virtual_fields_scratch = fields;
        }
    }

    fn take_virtual_worklist(&mut self) -> Vec<majit_ir::OpRef> {
        let mut worklist = std::mem::take(&mut self.virtual_worklist_scratch);
        worklist.clear();
        worklist
    }

    fn return_virtual_worklist(&mut self, mut worklist: Vec<majit_ir::OpRef>) {
        worklist.clear();
        if worklist.capacity() > self.virtual_worklist_scratch.capacity() {
            self.virtual_worklist_scratch = worklist;
        }
    }

    /// Return the `finish` livebox list so the next guard reuses its
    /// allocation. The caller must be done reading the boxes.
    pub fn recycle_ordered_liveboxes(&mut self, mut liveboxes: Vec<majit_ir::operand::Operand>) {
        liveboxes.clear();
        if liveboxes.capacity() > self.ordered_livebox_scratch.capacity() {
            self.ordered_livebox_scratch = liveboxes;
        }
    }

    /// resume.py `_number_boxes` + resume.py `getconst` parity.
    ///
    /// Encode one `ResumeValueSource` into the i64 tagged form written to
    /// `rd_numb`. `liveboxes` / `box_map` track compact TAGBOX numbering
    /// identical to RPython's `numb_state.liveboxes` dict.
    ///
    /// Pushes Ref/Float/large-Int constants through `getconst_i64`, which
    /// shares `self.consts` with the i16 `getconst` path — so there is
    /// exactly one pool per memo (RPython parity: `self.consts: list[Const]`).
    pub fn encode_tagged_source(
        &mut self,
        source: &ResumeValueSource,
        liveboxes: &mut Vec<usize>,
        box_map: &mut indexmap::IndexMap<usize, usize>,
    ) -> i64 {
        match source {
            // resume.py: new box → liveboxes[box] = tag(num_boxes, TAGBOX)
            ResumeValueSource::FailArg(index) => {
                let compact = *box_map.entry(*index).or_insert_with(|| {
                    let n = liveboxes.len();
                    liveboxes.push(*index);
                    n
                });
                tag_i64(encode_len(compact), TAGBOX)
            }
            // resume.py: isinstance(box, Const) → self.getconst(box).
            ResumeValueSource::Constant(c) => self.getconst_i64(c),
            // resume.py: virtual → tag(num_virtuals, TAGVIRTUAL)
            ResumeValueSource::Virtual(index) => tag_i64(encode_len(*index), TAGVIRTUAL),
            ResumeValueSource::Tagged(_) => {
                panic!("runtime tagged source cannot be encoded into a new resume stream")
            }
            ResumeValueSource::Uninitialized => tag_i64(ENCODED_UNINITIALIZED, TAGCONST),
            ResumeValueSource::Unavailable => tag_i64(ENCODED_UNAVAILABLE, TAGCONST),
        }
    }

    /// resume.py getconst(const) — tag a constant value.
    /// Unified entry point matching RPython's getconst(const) which
    /// dispatches on const.type (INT, REF, FLOAT).
    ///
    /// `Err(TagOverflow)` when the constant pool has outgrown the tag's 14-bit
    /// signed field; RPython lets `tag`'s exception out of `getconst` the same
    /// way, up to `optimizer.py`'s `except resume.TagOverflow: raise
    /// compile.giveup()`.
    /// Forward every `Const::Ref` the pool holds, then re-key the `refs`
    /// cache from the forwarded pool.
    ///
    /// `resume.py ResumeDataLoopMemo.__init__` keys `self.refs` with
    /// `new_ref_dict()`, whose hash is the object's identity hash and therefore
    /// survives a move. Pyre keys the same cache by the raw address, so a
    /// collection between two guards of one optimizer leaves every key naming
    /// the pre-move address: `getconst_ref` would answer a later guard with a
    /// tag whose pool slot now holds a different object. Rebuilding the keys
    /// here is what makes the raw-address encoding equivalent to the identity
    /// dict.
    ///
    /// `large_ints` needs no such pass — an integer key does not move.
    pub fn walk_const_ptr_refs_mut(&mut self, visitor: &mut dyn FnMut(&mut GcRef)) {
        // SAFETY: the stop-the-world root walker is the only code running and
        // holds exclusive access to this optimizer memo.
        let consts = unsafe { self.consts.as_mut_vec_for_gc() };
        for c in consts.iter_mut() {
            if let majit_ir::Const::Ref(slot) = c {
                visitor(slot);
            }
        }
        if self.refs.is_empty() {
            return;
        }
        // `_newconst` mints `tag(len(consts) + TAG_CONST_OFFSET, TAGCONST)`
        // before pushing, so the tag names the pool slot that holds this
        // entry's forwarded address. Rebuild in iteration order: `refs` is a
        // dict upstream and the numbering reads it as one.
        let mut rebuilt: IndexMap<i64, i16> = IndexMap::with_capacity(self.refs.len());
        for (_, &tagged) in self.refs.iter() {
            let (num, _) = untag(tagged);
            let idx = (num - TAG_CONST_OFFSET) as usize;
            if let Some(majit_ir::Const::Ref(gcref)) = self.consts.get(idx) {
                rebuilt.insert(gcref.0 as i64, tagged);
            }
        }
        self.refs = rebuilt;
    }

    pub fn getconst(&mut self, val: i64, tp: majit_ir::Type) -> Result<i16, TagOverflow> {
        match tp {
            majit_ir::Type::Int => self.getconst_int(val),
            majit_ir::Type::Ref => self.getconst_ref(val),
            majit_ir::Type::Float => self.getconst_float(val),
            majit_ir::Type::Void => self.newconst(val, tp),
        }
    }

    /// resume.py getconst for INT type.
    pub fn getconst_int(&mut self, val: i64) -> Result<i16, TagOverflow> {
        // Try inline TAGINT (-8191..8190 in RPython's i16 range).
        let shifted = val >> 13;
        if shifted == 0 || shifted == -1 {
            return Ok(((val << 2) | TAGINT as i64) as i16);
        }
        // Large int: check cache.
        if let Some(&tagged) = self.large_ints.get(&val) {
            return Ok(tagged);
        }
        let tagged = self.newconst(val, majit_ir::Type::Int)?;
        self.large_ints.insert(val, tagged);
        Ok(tagged)
    }

    /// resume.py getconst for REF type.
    pub fn getconst_ref(&mut self, val: i64) -> Result<i16, TagOverflow> {
        if val == 0 {
            return Ok(NULLREF);
        }
        if let Some(&tagged) = self.refs.get(&val) {
            return Ok(tagged);
        }
        let tagged = self.newconst(val, majit_ir::Type::Ref)?;
        self.refs.insert(val, tagged);
        Ok(tagged)
    }

    /// resume.py getconst fallback for FLOAT type.
    pub fn getconst_float(&mut self, val: i64) -> Result<i16, TagOverflow> {
        // FLOAT constants always go to the pool (no inline encoding).
        // RPython: return self._newconst(const)
        self.newconst(val, majit_ir::Type::Float)
    }

    /// resume.py `_newconst` — add to consts pool, return TAGCONST-tagged.
    ///
    /// ```python
    /// result = tag(len(self.consts) + TAG_CONST_OFFSET, TAGCONST)
    /// self.consts.append(const)
    /// return result
    /// ```
    ///
    /// The tag is minted through [`tag`] rather than by an inline shift-and-
    /// cast, so a pool larger than the 14-bit signed field answers
    /// `Err(TagOverflow)` instead of wrapping. A wrapped tag names a different
    /// pool slot, and the deopt then restores a different constant than the
    /// one the guard captured — with no diagnostic anywhere.
    ///
    /// The pool entry is pushed only after the tag is minted, so a rejected
    /// tag leaves the pool exactly as it was.
    fn newconst(&mut self, val: i64, tp: majit_ir::Type) -> Result<i16, TagOverflow> {
        let tagged = tag(self.consts.len() as i32 + TAG_CONST_OFFSET, TAGCONST)?;
        // SAFETY: numbering is the sole mutator and no resume reader exists
        // until the optimized trace is published.
        unsafe {
            self.consts
                .push_during_optimization(majit_ir::Const::from_raw_i64(val, tp));
        }
        Ok(tagged)
    }

    /// resume.py getconst — i64-sized variant used by the rd_numb
    /// encoder (`encode_shared`). Shares the pool (`self.consts`) with the
    /// i16 variant (`getconst`) so there is exactly one `rd_consts` per
    /// memo, matching RPython's single `self.consts: list[Const]`.
    fn getconst_i64(&mut self, c: &majit_ir::Const) -> i64 {
        match c {
            // resume.py: try tag(val, TAGINT).
            majit_ir::Const::Int(value) if can_inline_tagged(*value) => tag_i64(*value, TAGINT),
            majit_ir::Const::Int(value) => {
                // resume.py:168-172 large int.
                if let Some(&tagged_i16) = self.large_ints.get(value) {
                    let (num, _) = untag(tagged_i16);
                    return tag_i64(encode_len((num - TAG_CONST_OFFSET) as usize), TAGCONST);
                }
                let index = self.consts.len();
                unsafe {
                    self.consts
                        .push_during_optimization(majit_ir::Const::Int(*value));
                }
                // Also publish through the i16 cache so that a later
                // `getconst_int(value)` returns the same pool slot
                // (resume.py:171 self.large_ints[val] = tagged). The cache
                // records only slots an i16 tag can name: this encoder's own
                // result is i64 and never truncates, so a slot past the tag's
                // range is still encodable here, and skipping the entry leaves
                // a later `getconst_int` to mint a fresh one and answer
                // `Err(TagOverflow)` itself rather than be handed a wrapped
                // tag naming somebody else's constant.
                if let Ok(tagged_i16) = tag((index as i32) + TAG_CONST_OFFSET, TAGCONST) {
                    self.large_ints.insert(*value, tagged_i16);
                }
                tag_i64(encode_len(index), TAGCONST)
            }
            majit_ir::Const::Ref(gcref) => {
                // resume.py val = 0 → NULLREF sentinel (no pool
                // entry allocated). `NULLREF = tag(-1, TAGCONST)` —
                // encoder emits `tag_i64(-1, TAGCONST)` and the
                // matching decoder in `decode_box` recognizes
                // `ENCODED_NULLREF` before the positive-index branch.
                let raw = gcref.as_usize() as i64;
                if raw == 0 {
                    return tag_i64(ENCODED_NULLREF, TAGCONST);
                }
                if let Some(&tagged_i16) = self.refs.get(&raw) {
                    let (num, _) = untag(tagged_i16);
                    return tag_i64(encode_len((num - TAG_CONST_OFFSET) as usize), TAGCONST);
                }
                let index = self.consts.len();
                unsafe {
                    self.consts
                        .push_during_optimization(majit_ir::Const::Ref(*gcref));
                }
                // See the Int arm on why an unrepresentable tag skips the cache.
                if let Ok(tagged_i16) = tag((index as i32) + TAG_CONST_OFFSET, TAGCONST) {
                    self.refs.insert(raw, tagged_i16);
                }
                tag_i64(encode_len(index), TAGCONST)
            }
            majit_ir::Const::Float(v) => {
                // resume.py _newconst (no dedup for floats in RPython).
                let index = self.consts.len();
                unsafe {
                    self.consts
                        .push_during_optimization(majit_ir::Const::Float(*v));
                }
                tag_i64(encode_len(index), TAGCONST)
            }
        }
    }

    /// resume.py num_cached_boxes — length of the box dedup cache.
    pub fn num_cached_boxes(&self) -> usize {
        self.cached_boxes.len()
    }

    /// resume.py num_cached_virtuals — length of the virtual dedup cache.
    pub fn num_cached_virtuals(&self) -> usize {
        self.cached_virtuals.len()
    }

    /// resume.py assign_number_to_box — returns a negative number.
    /// resume.py assign_number_to_box(box, boxes).
    ///
    /// RPython version mutates `boxes` list:
    /// - cached: `boxes[-num - 1] = box`
    /// - new: `boxes.append(box); num = -len(boxes)`
    pub fn assign_number_to_box(
        &mut self,
        b: &majit_ir::operand::Operand,
        boxes: &mut Vec<OpRef>,
    ) -> i32 {
        if let Some(&num) = self.cached_boxes.get(b) {
            // resume.py:268: boxes[-num - 1] = box
            let idx = (-num - 1) as usize;
            if idx < boxes.len() {
                boxes[idx] = b.to_opref();
            }
            return num;
        }
        // resume.py:270-271: boxes.append(box); num = -len(boxes)
        boxes.push(b.to_opref());
        let num = -(boxes.len() as i32);
        self.cached_boxes.insert(b.clone(), num);
        num
    }

    /// `ResumeDataLoopMemo.assign_number_to_box` retains the box object.
    /// RPython's `new_liveboxes = [None] * memo.num_cached_boxes()`.
    pub fn assign_number_to_box_opt(
        &mut self,
        b: &majit_ir::operand::Operand,
        boxes: &mut Vec<Option<majit_ir::operand::Operand>>,
    ) -> i32 {
        if let Some(&num) = self.cached_boxes.get(b) {
            let idx = (-num - 1) as usize;
            if idx < boxes.len() {
                boxes[idx] = Some(b.clone());
            }
            return num;
        }
        boxes.push(Some(b.clone()));
        let num = -(boxes.len() as i32);
        self.cached_boxes.insert(b.clone(), num);
        num
    }

    /// resume.py assign_number_to_virtual — returns a negative number.
    pub fn assign_number_to_virtual(&mut self, b: &majit_ir::operand::Operand) -> i32 {
        if let Some(&num) = self.cached_virtuals.get(b) {
            return num;
        }
        // resume.py:283: num = self.cached_virtuals[box] = -len(self.cached_virtuals) - 1
        let num = -(self.num_cached_virtuals() as i32) - 1;
        self.cached_virtuals.insert(b.clone(), num);
        num
    }

    /// resume.py update_counters(profiler).
    ///
    /// Roll the memo's cumulative NVIRTUALS / NVHOLES / NVREUSED into the
    /// caller-supplied profiler. Called from optimizeopt/optimizer.py:557
    /// once per trace compilation. The caller owns the profiler state;
    /// the memo only exposes its counters.
    pub fn update_counters(&self, profiler: &crate::jitprof::JitProfiler) {
        profiler.count(crate::pyjitpl::counters::NVIRTUALS, self.nvirtuals);
        profiler.count(crate::pyjitpl::counters::NVHOLES, self.nvholes);
        profiler.count(crate::pyjitpl::counters::NVREUSED, self.nvreused);
    }

    /// resume.py clear_box_virtual_numbers.
    pub fn clear_box_virtual_numbers(&mut self) {
        self.cached_boxes.clear();
        self.cached_virtuals.clear();
    }

    /// Access the shared constant pool (value, type) pairs. Parity with
    /// RPython `memo.consts` list access (resume.py).
    pub fn consts(&self) -> &[majit_ir::Const] {
        self.consts.as_slice()
    }

    /// Take ownership of the shared constant pool — used by single-shot
    /// encoders that discard the memo after encoding.
    pub fn take_consts(&mut self) -> Vec<majit_ir::Const> {
        self.consts.snapshot()
    }

    /// resume.py register_box — add a non-const, non-seen box to
    /// new_liveboxes with `UNASSIGNED`. The virtual classification is
    /// applied separately by `register_virtual_fields` (resume.py),
    /// which overwrites the entry with `UNASSIGNEDVIRTUAL` (or a
    /// pre-numbered tag from `liveboxes_from_env`). RPython's
    /// `register_box` does not consult `env.is_virtual` — see
    /// resume.py:
    ///     if (box is not None and not isinstance(box, Const)
    ///         and box not in self.liveboxes_from_env
    ///         and box not in self.liveboxes):
    ///         self.liveboxes[box] = UNASSIGNED
    fn register_box(
        &self,
        opref: majit_ir::OpRef,
        env: &dyn majit_ir::BoxEnv,
        liveboxes_from_env: &LiveboxMap,
        new_liveboxes: &mut LiveboxMap,
    ) {
        if opref.is_none() || opref.is_constant() {
            return;
        }
        // resume.py — constants are handled by _gettagged
        // (TAGCONST/TAGINT) and don't need livebox slots.
        let b = env.get_box_replacement_operand(opref);
        if env.is_const(&b) {
            return;
        }
        if liveboxes_from_env.contains_key(&b) || new_liveboxes.contains_key(&b) {
            return;
        }
        new_liveboxes.insert(b, UNASSIGNED);
    }

    /// resume.py register_virtual_fields — stamp a virtual
    /// box's livebox tag and queue it for visitor_walk_recursive.
    ///
    /// `tagged = liveboxes_from_env.get(virtualbox, UNASSIGNEDVIRTUAL)`
    /// then `self.liveboxes[virtualbox] = tagged` unconditionally
    /// (overwriting any UNASSIGNED a prior `register_box` may have
    /// installed). The pre-numbered branch lets a virtual that was
    /// already TAGVIRTUAL'd in numbering keep its negative index.
    fn register_virtual_box(
        &self,
        virtualbox: majit_ir::OpRef,
        env: &dyn majit_ir::BoxEnv,
        liveboxes_from_env: &LiveboxMap,
        new_liveboxes: &mut LiveboxMap,
    ) {
        // #160/S11: key by the canonical box (Rc::ptr_eq). virtualbox is
        // replacement-walked by the caller; re-walking is idempotent.
        let b = env.get_box_replacement_operand(virtualbox);
        let tagged = liveboxes_from_env.get(&b).unwrap_or(UNASSIGNEDVIRTUAL);
        new_liveboxes.insert(b, tagged);
    }

    /// resume.py `_number_virtuals(liveboxes, num_env_virtuals)`.
    ///
    /// Walks `new_liveboxes` in insertion order, converts UNASSIGNED /
    /// UNASSIGNEDVIRTUAL tags into real negative numbers via
    /// `assign_number_to_box` / `assign_number_to_virtual`, then
    /// materializes each virtual's fieldnums through
    /// `env.make_virtual_info` and stores them into the returned
    /// `rd_virtuals` Vec.
    ///
    /// `liveboxes` is extended in place with the freshly numbered boxes
    /// (resume.py `liveboxes.extend(new_liveboxes)`). Returns
    /// `(rd_virtuals, nholes)` where nholes is used for the
    /// `_invalidation_needed` heuristic check by the caller.
    #[allow(clippy::too_many_arguments)]
    fn _number_virtuals(
        &mut self,
        liveboxes: &mut Vec<Option<majit_ir::operand::Operand>>,
        new_liveboxes: &mut LiveboxMap,
        virtual_fields: &indexmap::IndexMap<majit_ir::OpRef, majit_ir::VirtualFieldsInfo>,
        num_env_virtuals: usize,
        numb_state: &NumberingState,
        env: &dyn majit_ir::BoxEnv,
    ) -> Result<(Vec<std::rc::Rc<majit_ir::RdVirtualInfo>>, usize), TagOverflow> {
        // resume.py: new_liveboxes = [None] * memo.num_cached_boxes()
        // then reverse + extend onto `liveboxes`. When this finish() saw no
        // field-walk boxes and no virtuals, that list is all holes — extend
        // them in place and skip the intermediate Vec the memo's cached-box
        // count would otherwise allocate on every timed-section guard.
        if new_liveboxes.is_empty() && virtual_fields.is_empty() {
            let n = self.num_cached_boxes();
            liveboxes.extend(std::iter::repeat_n(None, n));
            return Ok((Vec::new(), n));
        }
        let mut new_boxes_list = vec![None; self.num_cached_boxes()];
        let mut count = 0;
        // Iterate in insertion order (RPython dict iteration = insertion order).
        // resoperation.py same_box parity: keys carry the typed OpRef
        // each entry was inserted with so virtual numbering preserves
        // `box.type` (history.py:220).
        // #160/S11: new_liveboxes.iter() yields the canonical box directly.
        let keys: Vec<(majit_ir::operand::Operand, i16)> = new_liveboxes.iter().collect();
        for (box_id, tagged) in keys {
            let (_, tagbits) = untag(tagged);
            if tagbits == TAGBOX {
                // resume.py: index = assign_number_to_box; liveboxes[box] = tag(index, TAGBOX)
                // `tag` raises `TagOverflow`; `compile.py giveup` abandons the compile.
                let index = self.assign_number_to_box_opt(&box_id, &mut new_boxes_list);
                new_liveboxes.insert(box_id, tag(index, TAGBOX)?);
                count += 1;
            } else {
                debug_assert_eq!(tagbits, TAGVIRTUAL);
                if tagged_eq(tagged, UNASSIGNEDVIRTUAL) {
                    // resume.py: index = assign_number_to_virtual; liveboxes[box] = tag(index, TAGVIRTUAL)
                    let index = self.assign_number_to_virtual(&box_id);
                    new_liveboxes.insert(box_id, tag(index, TAGVIRTUAL)?);
                }
            }
        }
        // resume.py:483-484: new_liveboxes.reverse(); liveboxes.extend(new_liveboxes)
        new_boxes_list.reverse();
        for box_id in &new_boxes_list {
            liveboxes.push(box_id.clone());
        }
        let nholes = new_boxes_list.len() - count;

        // resume.py:488-506: create rd_virtuals
        // resume.py: make_virtual_info(info, fieldnums) via BoxEnv dispatch
        let mut rd_virtuals: Vec<std::rc::Rc<majit_ir::RdVirtualInfo>> = Vec::new();
        if !virtual_fields.is_empty() {
            // resume.py: length = num_env_virtuals + memo.num_cached_virtuals()
            let length = num_env_virtuals + self.num_cached_virtuals();
            // resume.py `virtuals = [None] * length`. The inner
            // element is not Option because descr-side `rd_virtuals`
            // is `Arc<[Rc<RdVirtualInfo>]>`; `RdVirtualInfo::Empty` /
            // `VirtualInfo::Empty` is that None.
            rd_virtuals.resize(length, std::rc::Rc::new(majit_ir::RdVirtualInfo::Empty));
            // resume.py:493-494: memo.nvirtuals += length; memo.nvholes += length - len(vfieldboxes)
            self.nvirtuals += length;
            self.nvholes += length - virtual_fields.len();

            for (&opref_id, vf) in virtual_fields {
                // resume.py: num, _ = untag(self.liveboxes[virtualbox])
                // Check both numb_state.liveboxes (env virtuals) and
                // new_liveboxes (nested virtuals discovered via worklist).
                let opref = opref_id;
                // #160/S11: liveboxes is box-keyed; form the canonical box.
                let b = env.get_box_replacement_operand(opref);
                let tagged = numb_state
                    .liveboxes
                    .get(&b)
                    .or_else(|| new_liveboxes.get(&b))
                    .unwrap_or(UNASSIGNEDVIRTUAL);
                let (num, _) = untag(tagged);
                // RPython uses Python negative indexing: virtuals[-1] = virtuals[len-1].
                // Negative nums come from assign_number_to_virtual for nested virtuals.
                let num_idx = if num >= 0 {
                    num as usize
                } else {
                    (rd_virtuals.len() as i32 + num) as usize
                };
                if num_idx < rd_virtuals.len() {
                    // resume.py: fieldnums = [self._gettagged(box) for box in fieldboxes]
                    let mut fieldnums: Vec<i16> = Vec::with_capacity(vf.field_oprefs.len());
                    for &opref in &vf.field_oprefs {
                        fieldnums.push(self._gettagged(
                            opref,
                            env,
                            &numb_state.liveboxes,
                            new_liveboxes,
                        )?);
                    }
                    let reused = env.virtual_info_would_be_reused(opref_id, &fieldnums);
                    // resume.py: vinfo = self.make_virtual_info(info, fieldnums)
                    if let Some(rd_virt) = env.make_virtual_info(opref_id, fieldnums) {
                        if reused {
                            // resume.py:504-505: cached `_cached_vinfo` reused.
                            self.nvreused += 1;
                        }
                        rd_virtuals[num_idx] = rd_virt;
                    }
                }
            }
        }
        Ok((rd_virtuals, nholes))
    }

    /// resume.py `_add_pending_fields(pending_setfields)`.
    ///
    /// Tags the target/value boxes of each pending SETFIELD_GC/SETARRAYITEM_GC
    /// operation so the resume path can replay them against rehydrated
    /// struct instances. RPython decodes descr/opnum/itemindex from a
    /// `ResOperation` inline; pyre has already split the op into a
    /// `GuardPendingFieldEntry` by the time finish() is called, so this
    /// method only tags the target and value OpRefs in place.
    fn _add_pending_fields(
        &mut self,
        pending_setfields: &mut [majit_ir::GuardPendingFieldEntry],
        env: &dyn majit_ir::BoxEnv,
        liveboxes_from_env: &LiveboxMap,
        new_liveboxes: &LiveboxMap,
    ) -> Result<(), TagOverflow> {
        for pf in pending_setfields.iter_mut() {
            let target = env.get_box_replacement(pf.target);
            let value = env.get_box_replacement(pf.value);
            // resume.py num = self._gettagged(box); fieldnum = self._gettagged(fieldbox)
            pf.target_tagged = self._gettagged(target, env, liveboxes_from_env, new_liveboxes)?;
            pf.value_tagged = self._gettagged(value, env, liveboxes_from_env, new_liveboxes)?;
        }
        Ok(())
    }

    /// resume.py `_add_optimizer_sections(numb_state, liveboxes, liveboxes_from_env)`.
    ///
    /// Delegates to bridgeopt.py `serialize_optimizer_knowledge(optimizer,
    /// numb_state, liveboxes, liveboxes_from_env, memo)`. Emits three
    /// serialized sections on every guard (RPython emits zeros when the
    /// optheap/optrewrite caches are empty; the deserializer relies on the
    /// sections always being present):
    ///
    /// 1. known-class bitfield per Ref livebox (bridgeopt.py)
    /// 2. heap field + array item triples (bridgeopt.py)
    /// 3. loopinvariant call results (bridgeopt.py)
    ///
    /// RPython's `memo` is `self`; `numb_state.liveboxes` plays the role of
    /// the caller's `liveboxes_from_env` (the dict-like live-set). Pyre
    /// additionally carries an explicit `new_liveboxes` map so the per-
    /// guard tagged numbers assigned during `_number_virtuals` line up with
    /// the optimizer_knowledge lookup.
    fn _add_optimizer_sections(
        &mut self,
        numb_state: &mut NumberingState,
        liveboxes: &[Option<majit_ir::operand::Operand>],
        new_liveboxes: &LiveboxMap,
        env: &dyn majit_ir::BoxEnv,
        optimizer_knowledge: Option<&OptimizerKnowledgeForResume>,
    ) -> Result<(), TagOverflow> {
        // resume.py:572-574: serialize_optimizer_knowledge(
        //     self.optimizer, numb_state, liveboxes, liveboxes_from_env, self.memo)
        crate::optimizeopt::bridgeopt::serialize_optimizer_knowledge(
            self,
            numb_state,
            liveboxes,
            new_liveboxes,
            env,
            optimizer_knowledge,
        )
    }

    /// resume.py `_invalidation_needed(nliveboxes, nholes)`.
    ///
    /// Heuristic for when the shared memo's cached-box dedup should be
    /// flushed after a successful resume encoding:
    ///
    /// ```python
    /// def _invalidation_needed(self, nliveboxes, nholes):
    ///     failargs_limit = memo.metainterp_sd.options.failargs_limit
    ///     if nliveboxes > (failargs_limit // 2):
    ///         if nholes > nliveboxes // 3:
    ///             return True
    ///     return False
    /// ```
    ///
    /// pyre uses the IR's compile-time `FAILARGS_LIMIT` (majit_ir `value.rs`)
    /// as the metainterp option isn't wired yet. Matches RPython semantics
    /// exactly: "lots of live boxes, many of them holes" → invalidate.
    fn _invalidation_needed(&self, nliveboxes: usize, nholes: usize) -> bool {
        // resume.py:514-517
        let failargs_limit = majit_ir::FAILARGS_LIMIT;
        if nliveboxes > failargs_limit / 2 && nholes > nliveboxes / 3 {
            return true;
        }
        false
    }

    /// resume.py _gettagged — resolve an OpRef to its tagged number.
    /// Looks up in liveboxes_from_env first, then new_liveboxes, then constant.
    ///
    /// Only the constant arm can fail, and it fails the way `getconst` does.
    pub(crate) fn _gettagged(
        &mut self,
        opref: majit_ir::OpRef,
        env: &dyn majit_ir::BoxEnv,
        liveboxes_from_env: &LiveboxMap,
        new_liveboxes: &LiveboxMap,
    ) -> Result<i16, TagOverflow> {
        if opref.is_none() {
            return Ok(UNINITIALIZED_TAG);
        }
        // resume.py: isinstance(box, Const) → getconst
        if let Some(bits) = opref.inline_const_bits() {
            return self.getconst(bits, opref.ty().unwrap());
        }
        let b = env.get_box_replacement_operand(opref);
        if env.is_const(&b) {
            let (val, tp) = env.get_const(&b);
            return self.getconst(val, tp);
        }
        // resume.py: liveboxes_from_env → existing tag
        if let Some(tagged) = liveboxes_from_env.get(&b) {
            return Ok(tagged);
        }
        if let Some(tagged) = new_liveboxes.get(&b) {
            // Resolve UNASSIGNED to real cached number
            if tagged_eq(tagged, UNASSIGNED)
                && let Some(&num) = self.cached_boxes.get(&b)
            {
                // `resume.py tag` raises `TagOverflow`; `compile.py giveup`
                // abandons the compile rather than writing `UNASSIGNED`.
                return tag(num, TAGBOX);
            }
            if tagged_eq(tagged, UNASSIGNEDVIRTUAL)
                && let Some(&num) = self.cached_virtuals.get(&b)
            {
                return tag(num, TAGVIRTUAL);
            }
            return Ok(tagged);
        }
        Ok(UNASSIGNED)
    }

    /// `ResumeDataLoopMemo._number_boxes` tags each box in a snapshot section.
    ///
    /// `env` supplies the operations used by the upstream method: replacement
    /// lookup, constant lookup, box type lookup, and virtual Ref/raw-pointer
    /// classification.
    pub fn _number_boxes(
        &mut self,
        boxes: &[SnapshotBox],
        numb_state: &mut NumberingState,
        env: &dyn BoxEnv,
    ) -> Result<(), TagOverflow> {
        for snapshot_box in boxes {
            self._number_one(*snapshot_box, None, numb_state, env)?;
        }
        Ok(())
    }

    /// One iteration of `ResumeDataLoopMemo._number_boxes`.
    ///
    /// `ResumeDataLoopMemo.number` pulls each box from `SnapshotIterator`
    /// (`iter_vable_array` / `iter_vref_array` / `iter_array`) and numbers it
    /// before the next box is decoded. Callers that already hold a slice use
    /// `_number_boxes`; the byte-bridge walk calls this directly.
    ///
    /// `cached` is the box object `TraceIterator._cache` holds for the slot
    /// when the caller has it (`SnapshotIterator._untag` hands out that
    /// object), so the replacement walk starts from the box itself.
    fn _number_one(
        &mut self,
        snapshot_box: SnapshotBox,
        cached: Option<&Operand>,
        numb_state: &mut NumberingState,
        env: &dyn BoxEnv,
    ) -> Result<(), TagOverflow> {
        let raw_opref = snapshot_box.opref();
        if raw_opref.is_none() {
            numb_state.append_short(NULLREF);
            return Ok(());
        }
        // resume.py ResumeDataLoopMemo._number_boxes retains the resolved
        // box for both classification and the liveboxes identity lookup.
        // Inline constants need no producer object or identity-map entry.
        if let Some(bits) = raw_opref.inline_const_bits() {
            numb_state.append_short(self.getconst(bits, raw_opref.ty().unwrap())?);
            return Ok(());
        }
        let b = match cached {
            Some(cached) => {
                let b = cached.get_box_replacement(false);
                // An unregistered position mints a fresh box on every
                // positional lookup, so only a registered one has an answer
                // to compare against.
                #[cfg(debug_assertions)]
                {
                    let positional = env.get_box_replacement_operand(raw_opref);
                    if positional.same_box(&env.get_box_replacement_operand(raw_opref)) {
                        debug_assert!(
                            b.same_box(&positional),
                            "_number_one {raw_opref:?}: cached box resolves to {:?}, \
                             position resolves to {:?}",
                            b.to_opref(),
                            positional.to_opref()
                        );
                    }
                }
                b
            }
            None => env.get_box_replacement_operand(raw_opref),
        };
        let opref = b.to_opref();
        if opref.is_none() {
            numb_state.append_short(NULLREF);
            return Ok(());
        }
        // resume.py: isinstance(box, Const) → getconst
        if env.is_const(&b) {
            let (val, tp) = env.get_const(&b);
            let tagged = self.getconst(val, tp)?;
            numb_state.append_short(tagged);
            return Ok(());
        }
        // ResumeDataLoopMemo._number_boxes: liveboxes
        if let Some(tagged) = numb_state.liveboxes.get(&b) {
            numb_state.append_short(tagged);
            return Ok(());
        }
        // ResumeDataLoopMemo._number_boxes:
        //
        //     box = iter.get(...)
        //     box = box.get_box_replacement()
        //     ...
        //     if box.type == 'r':
        //
        // The type used for virtual classification is the replacement
        // box's type, not the original snapshot slot's fallback type.  A
        // snapshot slot can carry an Int fallback from tracing but forward
        // to a Ref virtual after optimization; keeping the stale fallback
        // would number that virtual as a TAGBOX and the subsequent
        // fail-arg force would materialize it.
        let box_type = opref.ty().unwrap_or_else(|| env.get_type(opref));
        let is_virtual = match box_type {
            majit_ir::Type::Ref => env.is_virtual_ref(&b),
            majit_ir::Type::Int => env.is_virtual_raw(&b),
            _ => false,
        };
        let tagged = if is_virtual {
            let t = tag(numb_state.num_virtuals, TAGVIRTUAL)?;
            numb_state.num_virtuals += 1;
            t
        } else {
            // resume.py `_number_boxes` reads `box.type` off the box.
            // `box_type` above is `opref.ty()` (Operand::type_ / OpRef::ty),
            // with `env.get_type` only when the variant carries no tag.
            let t = tag(numb_state.num_boxes, TAGBOX)?;
            numb_state.num_boxes += 1;
            t
        };
        numb_state.liveboxes.insert(b, tagged);
        numb_state.append_short(tagged);
        Ok(())
    }

    /// `ResumeDataLoopMemo.number` section order, fed one box at a time.
    ///
    /// `vable_len` / `vref_len` are the array lengths (`vref_len` is the
    /// pair count times two, matching `vref_array` before the `>> 1`
    /// stored in the resume bytes). `frames` is
    /// `(jitcode_index, pc, box_count)`. `next_vable`, `next_vref`,
    /// and `next_frame_box` are pulled that many times, in that order —
    /// the same order `SnapshotIterator` yields. Each pull may also hand
    /// out the slot's `TraceIterator._cache` box (see [`Self::_number_one`]).
    pub(crate) fn number_sections<'c>(
        &mut self,
        vable_len: i64,
        mut next_vable: impl FnMut() -> (SnapshotBox, Option<&'c Operand>),
        vref_len: i64,
        mut next_vref: impl FnMut() -> (SnapshotBox, Option<&'c Operand>),
        frames: &[(i32, i32, usize)],
        mut next_frame_box: impl FnMut() -> (SnapshotBox, Option<&'c Operand>),
        env: &dyn BoxEnv,
        minimum_virtualizable_size: i64,
    ) -> Result<NumberingState, TagOverflow> {
        let size_hint = (vable_len.max(0) as usize)
            + (vref_len.max(0) as usize)
            + frames.iter().map(|(_, _, boxes)| boxes + 2).sum::<usize>()
            + 4;
        let mut writer = majit_ir::resumecode::Writer {
            current: std::mem::take(&mut self.writer_scratch),
        };
        writer.current.clear();
        writer.current.reserve(size_hint);
        let mut numb_state = NumberingState {
            writer,
            liveboxes: self.take_livebox_map(),
            num_boxes: 0,
            num_virtuals: 0,
        };

        // resume.py number: patch later
        numb_state.append_int(0); // slot 0: size of resume section
        numb_state.append_int(0); // slot 1: number of failargs (patched by finish())

        // resume.py number: if minimum_virtualizable_size != -1, the
        // virtualizable itself is one entry in the array too, so use '>'.
        if minimum_virtualizable_size != -1 {
            assert!(
                vable_len > minimum_virtualizable_size,
                "vable_array length {} not > minimum_virtualizable_size {}",
                vable_len,
                minimum_virtualizable_size
            );
        }

        // resume.py number: virtualizable array, identity-first, unchanged.
        numb_state.append_int(vable_len);
        for _ in 0..vable_len.max(0) as usize {
            let (snapshot_box, cached) = next_vable();
            self._number_one(snapshot_box, cached, &mut numb_state, env)?;
        }

        // resume.py number: virtualref array. The stored count is pairs.
        debug_assert!(vref_len & 1 == 0, "vref_array length must be even");
        numb_state.append_int(vref_len >> 1);
        for _ in 0..vref_len.max(0) as usize {
            let (snapshot_box, cached) = next_vref();
            self._number_one(snapshot_box, cached, &mut numb_state, env)?;
        }

        // resume.py number: frame chain.
        // Per-frame: jitcode_index, pc, [tagged_values...].
        for &(jitcode_index, pc, nboxes) in frames {
            numb_state.append_int(jitcode_index as i64);
            numb_state.append_int(pc as i64);
            for _ in 0..nboxes {
                let (snapshot_box, cached) = next_frame_box();
                self._number_one(snapshot_box, cached, &mut numb_state, env)?;
            }
        }

        // resume.py number: patch total size
        numb_state.patch_current_size(0);
        Ok(numb_state)
    }

    /// resume.py number() — serialize a guard's full snapshot.
    ///
    /// Output format (in NumberingState):
    /// ```text
    /// [0]  size (patched later)
    /// [1]  number of failargs (patched later)
    /// [2]  vable_array_length  (0 if no virtualizable)
    ///      [tagged boxes for vable_array]
    /// [n]  vref_array_length   (0 if no virtualrefs)
    ///      [tagged boxes for vref_array]
    /// [m]  frame0_jitcode_index frame0_pc frame0_slots...
    /// [m+] frame1_jitcode_index frame1_pc frame1_slots...
    /// ...
    /// ```
    ///
    /// `frames` carries `(jitcode_index, pc, fail_args_slice)` for each frame.
    /// In pyre (single frame), this is typically one frame.
    /// resume.py number() — serialize a guard's full snapshot.
    ///
    /// Exact port of RPython's `number(self, position, trace, ...)`.
    ///
    /// `snapshot` describes the guard's state:
    /// - `vable_array`: virtualizable field boxes
    /// - `vref_array`: virtualref pairs
    /// - `framestack`: list of (jitcode_index, pc, boxes) per frame
    ///
    /// `env` implements BoxEnv for box operations.
    ///
    /// Returns `Err(TagOverflow)` if any box index exceeds the tag range.
    /// RPython: raises TagOverflow → caller does compile.giveup().
    ///
    /// NOTE: Slot 1 (number of failargs) is left as 0 here.
    /// RPython patches it later in ResumeDataVirtualAdder.finish()
    /// (resume.py). Callers must call
    /// `numb_state.writer.patch(1, num_liveboxes)` after finish().
    fn number_slices(
        &mut self,
        vable_array: &[SnapshotBox],
        vref_array: &[SnapshotBox],
        frames: &[(i32, i32, &[SnapshotBox])],
        env: &dyn BoxEnv,
        minimum_virtualizable_size: i64,
    ) -> Result<NumberingState, TagOverflow> {
        let frame_meta: SmallVec<[(i32, i32, usize); 16]> = frames
            .iter()
            .map(|(jc, pc, boxes)| (*jc, *pc, boxes.len()))
            .collect();
        let mut vable_i = 0usize;
        let mut vref_i = 0usize;
        let mut frame_i = 0usize;
        let mut box_i = 0usize;
        self.number_sections(
            vable_array.len() as i64,
            || {
                let snap_box = vable_array[vable_i];
                vable_i += 1;
                (snap_box, None)
            },
            vref_array.len() as i64,
            || {
                let snap_box = vref_array[vref_i];
                vref_i += 1;
                (snap_box, None)
            },
            &frame_meta,
            || {
                while frames[frame_i].2.is_empty() || box_i == frames[frame_i].2.len() {
                    frame_i += 1;
                    box_i = 0;
                }
                let snap_box = frames[frame_i].2[box_i];
                box_i += 1;
                (snap_box, None)
            },
            env,
            minimum_virtualizable_size,
        )
    }

    pub fn number(
        &mut self,
        snapshot: &Snapshot,
        env: &dyn BoxEnv,
        minimum_virtualizable_size: i64,
    ) -> Result<NumberingState, TagOverflow> {
        let frames: SmallVec<[(i32, i32, &[SnapshotBox]); 16]> = snapshot
            .framestack
            .iter()
            .map(|frame| (frame.jitcode_index, frame.pc, frame.boxes.as_slice()))
            .collect();
        self.number_slices(
            &snapshot.vable_array,
            &snapshot.vref_array,
            &frames,
            env,
            minimum_virtualizable_size,
        )
    }

    /// `resume.py ResumeDataLoopMemo.number` over the trace's existing
    /// snapshot arrays.
    ///
    /// RPython's `SnapshotIterator` reads the opencoder buffer lazily; it does
    /// not copy every live-box array once per guard. The optimizer stores the
    /// equivalent arrays separately, so this entry point borrows those slices
    /// and only builds the frame descriptor inline for the common
    /// sixteen-or-fewer-frame case (regex `and`/`or` inlines past eight).
    pub fn number_from_parts(
        &mut self,
        snapshot_boxes: &[SnapshotBox],
        frame_sizes: Option<&[usize]>,
        frame_pcs: &[(i32, i32)],
        vable_array: &[SnapshotBox],
        vref_array: &[SnapshotBox],
        env: &dyn BoxEnv,
        minimum_virtualizable_size: i64,
    ) -> Result<NumberingState, TagOverflow> {
        let mut frames: SmallVec<[(i32, i32, &[SnapshotBox]); 16]> = SmallVec::new();
        // An explicitly empty framestack is the terminal force snapshot
        // (opencoder.py create_empty_top_snapshot), not an absent layout.
        // A one-frame sizes vector is the default single-frame case
        // (`number` already emits that layout); keep it on the fallback
        // so a size that does not cover every snapshot box cannot drop
        // the remainder. Multi-frame and empty stay explicit.
        if let Some(sizes) = frame_sizes.filter(|sizes| sizes.len() != 1) {
            let mut offset = 0;
            for (i, &size) in sizes.iter().enumerate() {
                let end = (offset + size).min(snapshot_boxes.len());
                let (jitcode_index, pc) = frame_pcs.get(i).copied().unwrap_or((0, 0));
                frames.push((jitcode_index, pc, &snapshot_boxes[offset..end]));
                offset = end;
            }
        } else {
            let (jitcode_index, pc) = frame_pcs.first().copied().unwrap_or((0, 0));
            frames.push((jitcode_index, pc, snapshot_boxes));
        }
        self.number_slices(
            vable_array,
            vref_array,
            &frames,
            env,
            minimum_virtualizable_size,
        )
    }

    /// resume.py ResumeDataVirtualAdder.finish() — exact port.
    ///
    /// `numb_state`: output of `number()`
    /// `env`: BoxEnv for resolving box properties (constants, types).
    ///   Virtual fields are discovered via `env.get_virtual_fields()`,
    ///   matching RPython's `visitor_walk_recursive` callback pattern.
    /// `pending_setfields`: resume.py register_box + visitor_walk_recursive,
    ///   resume.py _add_pending_fields tagging.
    ///   target_tagged/value_tagged are filled in-place.
    /// `optimizer_knowledge`: bridgeopt.py serialize_optimizer_knowledge.
    ///   Heap field triples and known-class info for bridge compilation.
    ///
    /// Returns `(rd_numb, rd_consts, rd_virtuals, liveboxes)`.
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    pub fn finish(
        &mut self,
        mut numb_state: NumberingState,
        env: &dyn majit_ir::BoxEnv,
        pending_setfields: &mut [majit_ir::GuardPendingFieldEntry],
        optimizer_knowledge: Option<&OptimizerKnowledgeForResume>,
    ) -> Result<
        (
            majit_ir::NumberingRef,
            Arc<majit_ir::SharedConstPool>,
            Vec<std::rc::Rc<majit_ir::RdVirtualInfo>>,
            Vec<majit_ir::operand::Operand>,
        ),
        TagOverflow,
    > {
        let num_env_virtuals = numb_state.num_virtuals;

        // resume.py: split liveboxes_from_env into TAGBOX/TAGVIRTUAL
        let mut liveboxes = std::mem::take(&mut self.livebox_opt_scratch);
        liveboxes.clear();
        liveboxes.resize(numb_state.num_boxes as usize, None);

        // resume.py:413: self.vfieldboxes collected by virtual walk
        // resume.py:408: self.liveboxes — newly discovered boxes from field walk
        let mut new_liveboxes = self.take_new_livebox_map();

        // resume.py:414-426: iterate liveboxes_from_env, discover virtual
        // fields. RPython walks the dict in insertion order; pyre's
        // `LiveboxMap` keeps that order in `entries` so the
        // `.iter()` sequence already matches, which the
        // virtual worklist drain below relies on for byte-identical
        // visitor_walk_recursive sequencing. Sorting by tag would
        // observably re-order virtuals across builds.
        //
        // TAGBOX placement at `liveboxes[i] = box` uses the
        // tag-derived index (resume.py), so it is iteration-order-
        // invariant; only the TAGVIRTUAL worklist push order matters.
        //
        // Keep each dictionary key's box identity in the livebox list.

        // Collect virtual fields discovered via env.get_virtual_fields()
        // (resume.py:419-426 visitor_walk_recursive pattern). Keyed by
        // typed OpRef so the same_box (resoperation.py) identity is
        // preserved end-to-end through the worklist drain.
        let mut virtual_fields = self.take_virtual_fields();

        // resume.py:419-426: visitor_walk_recursive — worklist for nested virtuals.
        let mut virtual_worklist = self.take_virtual_worklist();

        for (b, tagged) in numb_state.liveboxes.iter() {
            // Only the virtual worklist still uses positional references.
            let opref = b.to_opref();
            let (i, tagbits) = untag(tagged);
            if tagbits == TAGBOX {
                if (i as usize) < liveboxes.len() {
                    liveboxes[i as usize] = Some(b);
                }
            } else {
                debug_assert_eq!(tagbits, TAGVIRTUAL);
                virtual_worklist.push(opref);
            }
        }

        // Worklist-based recursive virtual discovery (RPython visitor_walk_recursive).
        // Process each virtual: register its field boxes, and if any field is
        // itself a virtual, add it to the worklist for later processing.
        let mut worklist_idx = 0;
        while worklist_idx < virtual_worklist.len() {
            let opref_id = virtual_worklist[worklist_idx];
            worklist_idx += 1;

            if virtual_fields.contains_key(&opref_id) {
                continue; // already_seen_virtual
            }
            let vf_result = env.get_virtual_fields(opref_id);
            if let Some(vf) = vf_result {
                // resume.py: register_virtual_fields
                for &field_opref in &vf.field_oprefs {
                    // resume.py: register_box (UNASSIGNED for
                    // non-virtual fields).
                    self.register_box(field_opref, env, &numb_state.liveboxes, &mut new_liveboxes);
                    // resume.py:419-426 visitor_walk_recursive: if the
                    // field is a virtual, register_virtual_fields
                    // overwrites the UNASSIGNED stamp with the env-
                    // pre-numbered tag (or UNASSIGNEDVIRTUAL).
                    let resolved_box = env.get_box_replacement_operand(field_opref);
                    let resolved = resolved_box.to_opref();
                    if !resolved.is_none()
                        && !virtual_fields.contains_key(&resolved)
                        && (env.is_virtual_ref(&resolved_box) || env.is_virtual_raw(&resolved_box))
                    {
                        self.register_virtual_box(
                            resolved,
                            env,
                            &numb_state.liveboxes,
                            &mut new_liveboxes,
                        );
                        virtual_worklist.push(resolved);
                    }
                }
                virtual_fields.insert(opref_id, vf);
            }
        }

        // resume.py: process pending_setfields — register_box on
        // target and value, then visitor_walk_recursive on virtual fieldbox.
        for pf in pending_setfields.iter() {
            let box_opref = env.get_box_replacement(pf.target);
            let fieldbox = env.get_box_replacement(pf.value);
            // resume.py: self.register_box(box); self.register_box(fieldbox)
            self.register_box(box_opref, env, &numb_state.liveboxes, &mut new_liveboxes);
            self.register_box(fieldbox, env, &numb_state.liveboxes, &mut new_liveboxes);
            // resume.py:440-442 — info.visitor_walk_recursive requires
            // the fieldbox to be a virtual:
            //     info = getptrinfo(fieldbox)
            //     assert info is not None and info.is_virtual()
            // A non-virtual fieldbox in pending_setfields is an
            // invariant violation in the optheap producer side; the
            // previous silent skip masked it.
            let vf = env.get_virtual_fields(fieldbox).unwrap_or_else(|| {
                panic!(
                    "pending_setfields fieldbox {:?} (target={:?}) is not virtual \
                     (resume.py:441 assert info is not None and info.is_virtual())",
                    fieldbox, box_opref
                )
            });
            // resume.py register_virtual_fields: stamp the virtual fieldbox
            // UNASSIGNEDVIRTUAL (overwriting the UNASSIGNED that register_box
            // installed above) so _number_virtuals numbers it as a virtual
            // rather than a livebox — otherwise it would be force-boxed.
            self.register_virtual_box(fieldbox, env, &numb_state.liveboxes, &mut new_liveboxes);
            for &field_opref in &vf.field_oprefs {
                self.register_box(field_opref, env, &numb_state.liveboxes, &mut new_liveboxes);
                let resolved_box = env.get_box_replacement_operand(field_opref);
                let resolved = resolved_box.to_opref();
                if !resolved.is_none()
                    && !virtual_fields.contains_key(&resolved)
                    && (env.is_virtual_ref(&resolved_box) || env.is_virtual_raw(&resolved_box))
                {
                    self.register_virtual_box(
                        resolved,
                        env,
                        &numb_state.liveboxes,
                        &mut new_liveboxes,
                    );
                    virtual_worklist.push(resolved);
                }
            }
            virtual_fields.insert(fieldbox, vf);
        }

        // resume.py:440-442 parity: drain worklist for nested virtuals
        // discovered from pending_setfields. RPython's visitor_walk_recursive
        // recursively processes all levels; our worklist pattern resumes here.
        while worklist_idx < virtual_worklist.len() {
            let opref_id = virtual_worklist[worklist_idx];
            worklist_idx += 1;
            if virtual_fields.contains_key(&opref_id) {
                continue;
            }
            if let Some(vf) = env.get_virtual_fields(opref_id) {
                for &field_opref in &vf.field_oprefs {
                    self.register_box(field_opref, env, &numb_state.liveboxes, &mut new_liveboxes);
                    let resolved_box = env.get_box_replacement_operand(field_opref);
                    let resolved = resolved_box.to_opref();
                    if !resolved.is_none()
                        && !virtual_fields.contains_key(&resolved)
                        && (env.is_virtual_ref(&resolved_box) || env.is_virtual_raw(&resolved_box))
                    {
                        self.register_virtual_box(
                            resolved,
                            env,
                            &numb_state.liveboxes,
                            &mut new_liveboxes,
                        );
                        virtual_worklist.push(resolved);
                    }
                }
                virtual_fields.insert(opref_id, vf);
            }
        }

        // resume.py self._number_virtuals(liveboxes, num_env_virtuals)
        let (rd_virtuals, nholes) = self._number_virtuals(
            &mut liveboxes,
            &mut new_liveboxes,
            &virtual_fields,
            num_env_virtuals as usize,
            &numb_state,
            env,
        )?;

        // resume.py: if self._invalidation_needed(...): memo.clear_box_virtual_numbers()
        if self._invalidation_needed(liveboxes.len(), nholes) {
            self.clear_box_virtual_numbers();
        }

        // resume.py self._add_pending_fields(pending_setfields)
        self._add_pending_fields(
            pending_setfields,
            env,
            &numb_state.liveboxes,
            &new_liveboxes,
        )?;

        // resume.py:447: numb_state.patch(1, len(liveboxes))
        numb_state.writer.patch(1, liveboxes.len() as i32);

        // resume.py: self._add_optimizer_sections(numb_state, liveboxes, liveboxes_from_env)
        self._add_optimizer_sections(
            &mut numb_state,
            &liveboxes,
            &new_liveboxes,
            env,
            optimizer_knowledge,
        )?;

        // resume.py:450-451: storage.rd_numb, storage.rd_consts
        let rd_numb = numb_state.create_numbering_arc();
        self.writer_scratch = std::mem::take(&mut numb_state.writer.current);
        let rd_consts = self.consts.clone();

        // resume.py ResumeDataVirtualAdder.finish returns liveboxes[:]. Keep
        // those exact numbered objects: looking up their positions again can
        // bind a different producer and collapse two TAGBOX slots onto one box.
        let mut ordered_liveboxes = std::mem::take(&mut self.ordered_livebox_scratch);
        ordered_liveboxes.clear();
        ordered_liveboxes.extend(
            liveboxes
                .iter()
                .map(|b| b.clone().unwrap_or_else(majit_ir::operand::Operand::none)),
        );
        liveboxes.clear();
        self.livebox_opt_scratch = liveboxes;
        self.return_livebox_map(std::mem::take(&mut numb_state.liveboxes));
        self.return_new_livebox_map(new_liveboxes);
        self.return_virtual_fields(virtual_fields);
        self.return_virtual_worklist(virtual_worklist);

        Ok((rd_numb, rd_consts, rd_virtuals, ordered_liveboxes))
    }

    /// resume.py finish (on ResumeDataVirtualAdder) — encode with shared pool.
    pub fn encode_shared(&mut self, rd: &ResumeData) -> EncodedResumeData {
        let mut rd_numb = Vec::new();
        // resume.py compact TAGBOX numbering state.
        let mut liveboxes: Vec<usize> = Vec::new();
        let mut box_map: indexmap::IndexMap<usize, usize> = indexmap::IndexMap::new();

        // resume.py:234-235: reserve slots
        rd_numb.push(0); // [0] = items_resume_section
        rd_numb.push(0); // [1] = count
        rd_numb.push(encode_len(rd.vable_array.len()));
        for source in &rd.vable_array {
            let tagged = self.encode_tagged_source(source, &mut liveboxes, &mut box_map);
            rd_numb.push(tagged);
        }
        // resume.py:243-247: vref_array (pairs).
        assert!(
            rd.vref_array.len().is_multiple_of(2),
            "vref_array must have even length (pairs)"
        );
        rd_numb.push(encode_len(rd.vref_array.len() / 2));
        for source in &rd.vref_array {
            let tagged = self.encode_tagged_source(source, &mut liveboxes, &mut box_map);
            rd_numb.push(tagged);
        }

        // resume.py number: per-frame jitcode_index, pc, [tagged_values...].
        let mut frame_sizes = Vec::with_capacity(rd.frames.len());
        for frame in &rd.frames {
            rd_numb.push(frame.jitcode_index as i64);
            rd_numb.push(encode_u64(frame.pc));
            for source in &frame.slot_map {
                let tagged = self.encode_tagged_source(source, &mut liveboxes, &mut box_map);
                rd_numb.push(tagged);
            }
            frame_sizes.push(frame.slot_map.len());
        }

        let rd_virtuals = rd.virtuals.clone();

        // resume.py:412-418: register virtual field boxes.
        for vinfo in &rd_virtuals {
            for source in vinfo.field_sources() {
                if let ResumeValueSource::FailArg(index) = source {
                    box_map.entry(*index).or_insert_with(|| {
                        let n = liveboxes.len();
                        liveboxes.push(*index);
                        n
                    });
                }
            }
        }

        // resume.py:420-430: walk pending fields — register + encode.
        // Collect first, then encode — can't hold iter borrow and call
        // `&mut self` method in the same expression.
        let pending_fields_snapshot: Vec<_> = rd.pending_fields.to_vec();
        let rd_pendingfields: Vec<_> = pending_fields_snapshot
            .into_iter()
            .map(|pending| EncodedPendingFieldWrite {
                // resume.py:547 lldescr = cast_instance_to_base_ptr(descr) —
                // the encoded form carries the descr itself, not a handle.
                descr: pending.descr.clone(),
                target: self.encode_tagged_source(&pending.target, &mut liveboxes, &mut box_map),
                value: self.encode_tagged_source(&pending.value, &mut liveboxes, &mut box_map),
                item_index: pending.item_index,
            })
            .collect();

        // resume.py:260 patch_current_size, resume.py:464 patch count
        rd_numb[0] = encode_len(rd_numb.len());
        rd_numb[1] = encode_len(liveboxes.len());

        EncodedResumeData {
            rd_numb,
            // resume.py:451 storage.rd_consts = self.memo.consts — single pool.
            rd_consts: self.consts.snapshot(),
            rd_pendingfields,
            rd_virtuals,
            liveboxes,
            frame_sizes,
        }
    }

    /// Number of entries in the shared constant pool.
    pub fn num_shared_consts(&self) -> usize {
        self.consts.len()
    }
}

impl Default for ResumeDataLoopMemo {
    fn default() -> Self {
        Self::new()
    }
}

/// resume.py: AbstractResumeDataReader — reads resume data to
/// reconstruct interpreter state after a guard failure.
///
/// Two concrete implementations in RPython:
/// - ResumeDataBoxReader: creates boxes (for blackhole interpreter)
/// - ResumeDataDirectReader: reads values directly (for fast path)
pub struct ResumeDataReader<'a> {
    /// The resume data to read from.
    resume_data: &'a ResumeData,
    /// Fail argument values from the guard failure.
    fail_values: &'a [i64],
    /// Materialized virtuals (lazily populated).
    virtuals: Vec<Option<i64>>,
}

impl<'a> ResumeDataReader<'a> {
    /// resume.py: AbstractResumeDataReader.__init__
    pub fn new(resume_data: &'a ResumeData, fail_values: &'a [i64]) -> Self {
        let num_virtuals = resume_data.virtuals.len();
        ResumeDataReader {
            resume_data,
            fail_values,
            virtuals: vec![None; num_virtuals],
        }
    }

    /// resume.py: _decode_box — decode a tagged value reference.
    pub fn decode_frame_slot(&self, source: &FrameSlotSource) -> i64 {
        self.decode_value(source)
    }

    /// Decode a ResumeValueSource to a concrete value.
    pub fn decode_value(&self, source: &ResumeValueSource) -> i64 {
        match source {
            ResumeValueSource::FailArg(idx) => self.fail_values.get(*idx).copied().unwrap_or(0),
            ResumeValueSource::Constant(c) => c.as_raw_i64(),
            ResumeValueSource::Virtual(vidx) => {
                self.virtuals.get(*vidx).copied().flatten().unwrap_or(0)
            }
            ResumeValueSource::Tagged(_) => {
                panic!("runtime tagged source requires ResumeDataDirectReader")
            }
            ResumeValueSource::Uninitialized | ResumeValueSource::Unavailable => 0,
        }
    }

    /// resume.py: consume_boxes — read all frame slots for one frame.
    pub fn read_frame_slots(&self, frame_idx: usize) -> Vec<i64> {
        if frame_idx >= self.resume_data.frames.len() {
            return vec![];
        }
        let frame = &self.resume_data.frames[frame_idx];
        frame
            .slot_map
            .iter()
            .map(|source| self.decode_frame_slot(source))
            .collect()
    }

    /// Number of frames in the resume data.
    pub fn num_frames(&self) -> usize {
        self.resume_data.frames.len()
    }

    /// PC for a given frame.
    pub fn frame_pc(&self, frame_idx: usize) -> u64 {
        self.resume_data
            .frames
            .get(frame_idx)
            .map(|f| f.pc)
            .unwrap_or(0)
    }
}

/// resume.py VirtualInfo parity.
/// Describes a virtual object's fields for materialization.
/// RPython uses a class hierarchy (VirtualInfo, VStructInfo, VArrayInfoClear, etc.).
/// We use a single struct with tagged field values.
#[derive(Debug, Clone, Default)]
pub struct VirtualFieldValues {
    /// Descriptor (type/class) for the virtual object.
    pub descr: Option<majit_ir::DescrRef>,
    /// Known class pointer (ob_type for NewWithVtable).
    pub known_class: Option<i64>,
    /// Tagged field values (i16 tags referencing consts/boxes/other virtuals).
    pub fieldnums: Vec<i16>,
}

/// resume.py:554-557 — tagged pending field entry.
/// RPython stores (lldescr, num, fieldnum, itemindex) where num and fieldnum
/// are tagged references into the numbering system.
#[derive(Debug, Clone)]
pub struct TaggedPendingField {
    pub descr_index: u32,
    pub item_index: i32,
    /// Tagged reference to target box (from _gettagged).
    pub num: i16,
    /// Tagged reference to value box (from _gettagged).
    pub fieldnum: i16,
}

/// bridgeopt.py serialize_optimizer_knowledge — optimizer knowledge for resume data encoding.
/// Passed into finish() for _add_optimizer_sections.
pub struct OptimizerKnowledgeForResume {
    /// (obj_opref, descr_index, val_opref) heap field triples.
    /// bridgeopt.py:96-101
    pub heap_fields: Vec<(majit_ir::OpRef, i32, majit_ir::OpRef)>,
    /// (array_opref, index, descr_index, val_opref) heap array item quads.
    /// bridgeopt.py:102-108
    pub heap_arrayitems: Vec<(majit_ir::OpRef, i64, i32, majit_ir::OpRef)>,
    /// (const_func_ptr, result_opref) loop-invariant call results.
    pub loopinvariant_results: Vec<(i64, majit_ir::OpRef)>,
}

impl OptimizerKnowledgeForResume {
    pub fn is_empty(&self) -> bool {
        self.heap_fields.is_empty()
            && self.heap_arrayitems.is_empty()
            && self.loopinvariant_results.is_empty()
    }
}

/// bridgeopt.py decode_box return type.
///
/// RPython's decode_box returns actual Const/Box objects. Two Rust
/// variants mirror that: `LiveBox` for `Box` references (TAGBOX) and
/// `Const` for all Const subtypes (TAGINT/TAGCONST/NULLREF), collapsing
/// into a single `majit_ir::Const` that carries its own type.
#[derive(Clone, Debug, PartialEq)]
pub enum DecodedBox {
    /// TAGBOX → `liveboxes[num]` (bridge inputarg / optimizer box).
    LiveBox(majit_ir::OpRef),
    /// TAGINT / TAGCONST / NULLREF — all Const subtypes.
    Const(majit_ir::Const),
}

/// bridgeopt.py decode_box: untag a tagged value from rd_numb.
///
/// Line-by-line port of PyPy's decode_box(). Returns DecodedBox to
/// preserve the Const vs Box distinction that RPython encodes via
/// Python class hierarchy.
pub fn decode_box(
    tagged: i16,
    rd_consts: &[majit_ir::Const],
    liveboxes: &[majit_ir::OpRef],
) -> DecodedBox {
    let (num, tag_type) = untag(tagged);
    // NB: the TAGVIRTUAL case can't happen here, because this code runs after
    // virtuals are already forced again.
    match tag_type {
        TAGCONST => {
            if tagged_eq(tagged, NULLREF) {
                // bridgeopt.py:51: box = CONST_NULL (history.py).
                DecodedBox::Const(majit_ir::Const::Ref(majit_ir::GcRef::NULL))
            } else {
                // bridgeopt.py:54: box = resumestorage.rd_consts[num - TAG_CONST_OFFSET]
                // — direct list index, IndexError on out-of-range. A
                // bridgeopt-knowledge stream that names a const slot
                // outside `rd_consts` is corrupt, and silently swapping
                // in CONST_NULL hides the corruption from the optimizer.
                let idx = (num - TAG_CONST_OFFSET) as usize;
                let c = rd_consts.get(idx).copied().unwrap_or_else(|| {
                    panic!(
                        "bridgeopt decode_box TAGCONST out-of-range: idx={} \
                         rd_consts.len()={} (corrupt knowledge stream — see \
                         bridgeopt.py:54)",
                        idx,
                        rd_consts.len()
                    )
                });
                DecodedBox::Const(c)
            }
        }
        // bridgeopt.py:56: box = ConstInt(num)
        TAGINT => DecodedBox::Const(majit_ir::Const::Int(num as i64)),
        TAGBOX => {
            // bridgeopt.py:58: box = liveboxes[num] — direct list index,
            // IndexError on out-of-range. See TAGCONST comment above.
            let idx = num as usize;
            let lb = *liveboxes.get(idx).unwrap_or_else(|| {
                panic!(
                    "bridgeopt decode_box TAGBOX out-of-range: idx={} \
                     liveboxes.len()={} (corrupt knowledge stream — see \
                     bridgeopt.py:58)",
                    idx,
                    liveboxes.len()
                )
            });
            DecodedBox::LiveBox(lb)
        }
        _ => {
            // bridgeopt.py:60: raise AssertionError("unreachable")
            unreachable!("bridgeopt decode_box: unexpected tag type {}", tag_type);
        }
    }
}

// VirtualFieldInfo removed: replaced by majit_ir::VirtualFieldsInfo.
// finish() now discovers virtual fields via env.get_virtual_fields().

#[cfg(test)]
mod tests {
    /// A collection between two guards of one optimizer moves a `ConstPtr`
    /// the memo already pooled. The walk must forward the pool entry and
    /// re-key `refs`, so the next guard's `getconst_ref` answers the moved
    /// address with the tag that already names it — and answers the vacated
    /// address with a fresh one.
    #[test]
    fn walking_the_memo_forwards_its_ref_consts_and_rekeys_the_cache() {
        let mut memo = super::ResumeDataLoopMemo::new();
        let before = 0x1000i64;
        let after = 0x2000i64;
        let tagged = memo.getconst_ref(before).expect("pool the ref const");
        assert_eq!(
            memo.getconst_ref(before).unwrap(),
            tagged,
            "cached before the move"
        );

        memo.walk_const_ptr_refs_mut(&mut |slot: &mut majit_ir::GcRef| {
            if slot.0 as i64 == before {
                *slot = majit_ir::GcRef(after as usize);
            }
        });

        assert_eq!(
            memo.getconst_ref(after).unwrap(),
            tagged,
            "the moved object keeps the tag its pool slot already holds"
        );
        let vacated = memo.getconst_ref(before).unwrap();
        assert_ne!(
            vacated, tagged,
            "the pre-move address names nothing; a later object there must not \
             inherit the moved object's pool slot"
        );
    }

    use super::*;
    use majit_ir::resumedata::{RebuiltValue, rebuild_from_numbering};

    #[test]
    fn orthodox_resume_shaped_virtuals_are_built_once() {
        let descr = majit_backend::make_resume_guard_descr_typed(vec![]);
        let fd = descr.as_fail_descr().expect("resume guard");
        fd.set_rd_virtuals(Some(vec![std::rc::Rc::new(
            majit_ir::RdVirtualInfo::VRawSliceInfo {
                offset: 3,
                fieldnums: vec![],
            },
        )]));
        let first = super::guard_virtual_infos(fd).expect("virtuals");
        let second = super::guard_virtual_infos(fd).expect("virtuals");
        assert_eq!(first.as_slice().as_ptr(), second.as_slice().as_ptr());
        assert!(matches!(
            first.as_slice()[0],
            VirtualInfo::VRawSlice { offset: 3, .. }
        ));
        drop(first);
        drop(second);
        fd.set_rd_virtuals(Some(vec![std::rc::Rc::new(
            majit_ir::RdVirtualInfo::VRawSliceInfo {
                offset: 4,
                fieldnums: vec![],
            },
        )]));
        let third = super::guard_virtual_infos(fd).expect("rebuilt");
        assert!(matches!(
            third.as_slice()[0],
            VirtualInfo::VRawSlice { offset: 4, .. }
        ));
        let fourth = super::guard_virtual_infos(fd).expect("cached rebuild");
        assert_eq!(third.as_slice().as_ptr(), fourth.as_slice().as_ptr());
    }

    #[test]
    fn resume_register_box_pairs_failargs_and_consts_and_skips_holes() {
        let fail_values = [11i64, 22];
        let (op, bits) =
            super::resume_register_box(&RebuiltValue::Box(1, majit_ir::Type::Int), &fail_values)
                .expect("failarg");
        assert_eq!(op, majit_ir::OpRef::input_arg_typed(1, majit_ir::Type::Int));
        assert_eq!(bits, 22);
        assert!(
            super::resume_register_box(&RebuiltValue::Box(4, majit_ir::Type::Int), &fail_values)
                .is_none()
        );
        let (op, bits) =
            super::resume_register_box(&RebuiltValue::Const(majit_ir::Const::Int(7)), &fail_values)
                .expect("const");
        assert_eq!(op, majit_ir::OpRef::const_int(7));
        assert_eq!(bits, 7);
        assert!(super::resume_register_box(&RebuiltValue::Unassigned, &fail_values).is_none());
        assert!(super::resume_register_box(&RebuiltValue::Virtual(0), &fail_values).is_none());
    }

    #[test]
    fn numbering_keeps_the_resolved_box_for_identity_lookup() {
        struct Env {
            inner: SimpleBoxEnv,
            resolutions: std::cell::Cell<usize>,
        }
        impl BoxEnv for Env {
            fn get_box_replacement(&self, _: majit_ir::OpRef) -> majit_ir::OpRef {
                panic!("numbering must retain its resolved box, not resolve its position again")
            }
            fn get_box_replacement_operand(
                &self,
                opref: majit_ir::OpRef,
            ) -> majit_ir::operand::Operand {
                self.resolutions.set(self.resolutions.get() + 1);
                self.inner.get_box_replacement_operand(opref)
            }
            fn is_const(&self, box_: &majit_ir::operand::Operand) -> bool {
                self.inner.is_const(box_)
            }
            fn get_const(&self, box_: &majit_ir::operand::Operand) -> (i64, majit_ir::Type) {
                self.inner.get_const(box_)
            }
            fn get_type(&self, opref: majit_ir::OpRef) -> majit_ir::Type {
                self.inner.get_type(opref)
            }
            fn is_virtual_ref(&self, box_: &majit_ir::operand::Operand) -> bool {
                self.inner.is_virtual_ref(box_)
            }
            fn is_virtual_raw(&self, box_: &majit_ir::operand::Operand) -> bool {
                self.inner.is_virtual_raw(box_)
            }
        }
        let mut env = Env {
            inner: SimpleBoxEnv::new(),
            resolutions: std::cell::Cell::new(0),
        };
        env.inner.replacements.insert(1, majit_ir::OpRef::int_op(0));
        env.inner
            .replacements
            .insert(2, majit_ir::OpRef::const_int(7));
        let boxes = [
            majit_ir::OpRef::int_op(1),
            majit_ir::OpRef::int_op(1),
            majit_ir::OpRef::int_op(2),
            majit_ir::OpRef::const_int(9),
        ]
        .map(SnapshotBox::untyped);
        let mut state = NumberingState::new(4);
        ResumeDataLoopMemo::new()
            ._number_boxes(&boxes, &mut state, &env)
            .unwrap();
        assert_eq!(state.num_boxes, 1);
        assert_eq!(state.liveboxes.iter().count(), 1);
        assert_eq!(env.resolutions.get(), 3);
    }

    #[test]
    fn livebox_map_preserves_box_identity_and_insertion_order() {
        let mut liveboxes = LiveboxMap::new();
        // Two distinct logical boxes — an InputArg and a ResOp result, each
        // bound to a rooted producer so they shed to Operand::InputArg / Op.
        // Under ptr_eq keying they stay distinct keys (PyPy `box is box`),
        // never collapsed by a shared raw slot index.
        let input = crate::history::test_support::rooted_operand_from_opref(
            majit_ir::OpRef::input_arg_int(0),
        );
        let op =
            crate::history::test_support::rooted_operand_from_opref(majit_ir::OpRef::int_op(0));

        liveboxes.insert(input.clone(), UNASSIGNED);
        liveboxes.insert(op.clone(), UNASSIGNEDVIRTUAL);

        assert_eq!(liveboxes.get(&input), Some(UNASSIGNED));
        assert_eq!(liveboxes.get(&op), Some(UNASSIGNEDVIRTUAL));
        // iter() yields each key's Operand directly, so compare by the stable
        // (type, position) OpRef identity + order.
        assert_eq!(
            liveboxes
                .iter()
                .map(|(b, t)| (b.to_opref(), t))
                .collect::<Vec<_>>(),
            vec![
                (input.to_opref(), UNASSIGNED),
                (op.to_opref(), UNASSIGNEDVIRTUAL)
            ]
        );
    }

    #[test]
    fn test_simple_resume_data() {
        let rd = ResumeData::simple(42, 3);
        let fail_values = vec![10, 20, 30];
        let frames = rd.reconstruct(&fail_values);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].pc, 42);
        assert_eq!(
            frames[0].values,
            vec![
                ReconstructedValue::Value(10),
                ReconstructedValue::Value(20),
                ReconstructedValue::Value(30),
            ]
        );
    }

    #[test]
    fn test_resume_data_with_gaps() {
        let rd = ResumeData {
            frames: vec![FrameInfo {
                jitcode_index: 0,
                pc: 100,
                slot_map: vec![
                    FrameSlotSource::FailArg(2),
                    FrameSlotSource::Unavailable,
                    FrameSlotSource::FailArg(0),
                ],
            }],
            virtuals: Vec::new(),
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            pending_fields: Vec::new(),
        };
        let fail_values = vec![10, 20, 30];
        let frames = rd.reconstruct(&fail_values);
        assert_eq!(
            frames[0].values,
            vec![
                ReconstructedValue::Value(30),
                ReconstructedValue::Unavailable,
                ReconstructedValue::Value(10),
            ]
        );
        assert_eq!(frames[0].lossy_values(), vec![30, 0, 10]);
    }

    #[test]
    fn test_multi_frame_resume() {
        let rd = ResumeData {
            frames: vec![
                FrameInfo {
                    jitcode_index: 0,
                    pc: 10,
                    slot_map: vec![FrameSlotSource::FailArg(0), FrameSlotSource::FailArg(1)],
                },
                FrameInfo {
                    jitcode_index: 1,
                    pc: 20,
                    slot_map: vec![FrameSlotSource::FailArg(2), FrameSlotSource::FailArg(3)],
                },
            ],
            virtuals: Vec::new(),
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            pending_fields: Vec::new(),
        };
        let fail_values = vec![1, 2, 3, 4];
        let frames = rd.reconstruct(&fail_values);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].pc, 10);
        assert_eq!(
            frames[0].values,
            vec![ReconstructedValue::Value(1), ReconstructedValue::Value(2)]
        );
        assert_eq!(frames[1].pc, 20);
        assert_eq!(
            frames[1].values,
            vec![ReconstructedValue::Value(3), ReconstructedValue::Value(4)]
        );
    }

    #[test]
    fn test_builder() {
        let mut builder = ResumeDataVirtualAdder::new();
        builder.push_frame(0, 42);
        builder.map_slot(0, 0);
        builder.map_slot(2, 1); // gap at slot 1
        let rd = builder.build();

        assert_eq!(rd.frames.len(), 1);
        assert_eq!(rd.frames[0].pc, 42);
        assert_eq!(
            rd.frames[0].slot_map,
            vec![
                FrameSlotSource::FailArg(0),
                FrameSlotSource::Unavailable,
                FrameSlotSource::FailArg(1),
            ]
        );
    }

    #[test]
    fn test_memo_number_simple() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        let snapshot = Snapshot::single_frame(
            0,
            8,
            vec![OpRef::const_int(42), OpRef::int_op(1), OpRef::int_op(2)],
        );
        let numb_state = memo.number(&snapshot, &env, -1).unwrap();
        // Should have: [size, num_failargs, 0(vable), 0(vref), 0(jitcode), 8(pc), tagged...]
        let items = crate::resumecode::unpack_numbering(&numb_state.create_numbering());
        // items[0] = total size
        assert!(items[0] > 0);
        // items[1] = num_failargs: 0 (not patched yet — RPython patches in finish())
        // After finish: patch(1, numb_state.liveboxes.len()) would set to 2.
        assert_eq!(items[1], 0);
        // items[2] = vable_array_length = 0
        assert_eq!(items[2], 0);
        // items[3] = vref_array_length = 0
        assert_eq!(items[3], 0);
        // items[4] = jitcode_index = 0
        assert_eq!(items[4], 0);
        // items[5] = pc = 8
        assert_eq!(items[5], 8);
        // items[6] = inline-Const(42) tagged as TAGINT(42) since 42 fits in 13 bits
        let (val, tagbits) = untag(items[6] as i16);
        assert_eq!(tagbits, TAGINT);
        assert_eq!(val, 42);
        // items[7] = OpRef::int_op(1) tagged as TAGBOX(0) — first live box
        let (val, tagbits) = untag(items[7] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 0);
        // items[8] = OpRef::int_op(2) tagged as TAGBOX(1) — second live box
        let (val, tagbits) = untag(items[8] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 1);
    }

    #[test]
    fn number_from_parts_preserves_an_empty_framestack_with_vable_state() {
        // opencoder.py create_empty_top_snapshot, reached after
        // pyjitpl.py finishframe has popped the last MIFrame.
        let snapshot = Snapshot {
            framestack: Vec::new(),
            vable_array: vec![
                SnapshotBox::from(OpRef::ref_op(0)),
                SnapshotBox::from(OpRef::const_int(7)),
            ],
            vref_array: Vec::new(),
        };
        let env = SimpleBoxEnv::new();
        let expected = ResumeDataLoopMemo::new()
            .number(&snapshot, &env, -1)
            .unwrap()
            .create_numbering();
        let actual = ResumeDataLoopMemo::new()
            .number_from_parts(&[], Some(&[]), &[], &snapshot.vable_array, &[], &env, -1)
            .unwrap()
            .create_numbering();
        assert_eq!(actual, expected, "no synthetic (0, 0, 0) frame at FINISH");
    }

    #[test]
    fn number_from_parts_keeps_a_one_frame_sizes_vector_on_the_default_path() {
        // A single-frame sizes entry is the implicit layout `number` already
        // emits. Using it as an explicit split would drop snapshot boxes
        // past that count.
        let snapshot = Snapshot {
            framestack: vec![SnapshotFrame {
                jitcode_index: 3,
                pc: 4,
                py_pc: 5,
                boxes: vec![
                    SnapshotBox::from(OpRef::ref_op(0)),
                    SnapshotBox::from(OpRef::const_int(9)),
                    SnapshotBox::from(OpRef::ref_op(1)),
                ],
            }],
            vable_array: Vec::new(),
            vref_array: Vec::new(),
        };
        let env = SimpleBoxEnv::new();
        let expected = ResumeDataLoopMemo::new()
            .number(&snapshot, &env, -1)
            .unwrap()
            .create_numbering();
        let boxes = snapshot.framestack[0].boxes.clone();
        let short_size = Some(&[1usize][..]);
        let actual = ResumeDataLoopMemo::new()
            .number_from_parts(&boxes, short_size, &[(3, 4)], &[], &[], &env, -1)
            .unwrap()
            .create_numbering();
        assert_eq!(
            actual, expected,
            "a one-frame sizes vector must not truncate snapshot boxes"
        );
    }

    #[test]
    fn test_number_rebuild_roundtrip() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        let snapshot = Snapshot::single_frame(
            0,
            8,
            vec![OpRef::const_int(42), OpRef::int_op(1), OpRef::int_op(2)],
        );
        let mut numb_state = memo.number(&snapshot, &env, -1).unwrap();
        // RPython: ResumeDataVirtualAdder.finish() patches slot 1 with num_boxes.
        numb_state.writer.patch(1, numb_state.num_boxes);
        let rd_numb = numb_state.create_numbering();

        let fail_arg_types = vec![majit_ir::Type::Int, majit_ir::Type::Int];
        let (num_failargs, _vable_values, _vref_values, rebuilt_frames) =
            rebuild_from_numbering(&rd_numb, memo.consts(), &fail_arg_types, None, 0);
        assert_eq!(num_failargs, 2);
        assert_eq!(rebuilt_frames.len(), 1);
        assert_eq!(rebuilt_frames[0].pc, 8);
        assert_eq!(rebuilt_frames[0].values.len(), 3);
        assert_eq!(
            rebuilt_frames[0].values[0],
            RebuiltValue::Const(majit_ir::Const::Int(42))
        );
        assert_eq!(
            rebuilt_frames[0].values[1],
            RebuiltValue::Box(0, majit_ir::Type::Int)
        );
        assert_eq!(
            rebuilt_frames[0].values[2],
            RebuiltValue::Box(1, majit_ir::Type::Int)
        );
    }

    #[test]
    fn test_number_rebuild_with_virtual() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.virtuals.insert(2); // OpRef::ref_op(2) is virtual (Ref type)
        env.types.insert(2, majit_ir::Type::Ref);
        let snapshot = Snapshot::single_frame(
            0,
            10,
            vec![OpRef::int_op(1), OpRef::ref_op(2), OpRef::int_op(3)],
        );
        let mut numb_state = memo.number(&snapshot, &env, -1).unwrap();
        // RPython: finish() patches with len(newboxes) which is num_boxes
        // (not liveboxes which includes virtuals).
        numb_state.writer.patch(1, numb_state.num_boxes);
        let rd_numb = numb_state.create_numbering();

        let fail_arg_types = vec![majit_ir::Type::Int, majit_ir::Type::Int];
        let (num_failargs, _vable_values, _vref_values, rebuilt_frames) =
            rebuild_from_numbering(&rd_numb, memo.consts(), &fail_arg_types, None, 0);
        assert_eq!(num_failargs, 2); // OpRef::int_op(1) and OpRef::int_op(3) are boxes
        assert_eq!(rebuilt_frames[0].values.len(), 3);
        assert_eq!(
            rebuilt_frames[0].values[0],
            RebuiltValue::Box(0, majit_ir::Type::Int)
        );
        assert_eq!(rebuilt_frames[0].values[1], RebuiltValue::Virtual(0));
        assert_eq!(
            rebuilt_frames[0].values[2],
            RebuiltValue::Box(1, majit_ir::Type::Int)
        );
    }

    #[test]
    fn test_memo_number_with_virtual() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.virtuals.insert(2);
        env.types.insert(2, majit_ir::Type::Ref);
        let snapshot = Snapshot::single_frame(
            0,
            10,
            vec![OpRef::int_op(1), OpRef::ref_op(2), OpRef::int_op(3)],
        );
        let numb_state = memo.number(&snapshot, &env, -1).unwrap();
        let items = crate::resumecode::unpack_numbering(&numb_state.create_numbering());
        // items[1] = num_failargs: 0 (not patched — RPython patches in finish())
        assert_eq!(items[1], 0);
        // items[6] = OpRef::int_op(1) → TAGBOX(0)
        let (val, tagbits) = untag(items[6] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 0);
        // items[7] = OpRef::ref_op(2) → TAGVIRTUAL(0)
        let (val, tagbits) = untag(items[7] as i16);
        assert_eq!(tagbits, TAGVIRTUAL);
        assert_eq!(val, 0);
        // items[8] = OpRef::int_op(3) → TAGBOX(1)
        let (val, tagbits) = untag(items[8] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 1);
    }

    #[test]
    fn test_number_boxes_uses_replacement_type_for_virtual_classification() {
        use majit_ir::OpRef;
        struct RefOnlyVirtualEnv {
            constants: indexmap::IndexMap<u32, (i64, majit_ir::Type)>,
            replacements: indexmap::IndexMap<u32, majit_ir::OpRef>,
            types: indexmap::IndexMap<u32, majit_ir::Type>,
            virtuals: indexmap::IndexSet<u32>,
            virtual_fields: indexmap::IndexMap<u32, majit_ir::VirtualFieldsInfo>,
            box_cache:
                std::cell::RefCell<indexmap::IndexMap<majit_ir::OpRef, majit_ir::operand::Operand>>,
        }

        impl RefOnlyVirtualEnv {
            fn new() -> Self {
                Self {
                    constants: indexmap::IndexMap::new(),
                    replacements: indexmap::IndexMap::new(),
                    types: indexmap::IndexMap::new(),
                    virtuals: indexmap::IndexSet::new(),
                    virtual_fields: indexmap::IndexMap::new(),
                    box_cache: std::cell::RefCell::new(indexmap::IndexMap::new()),
                }
            }
        }

        impl BoxEnv for RefOnlyVirtualEnv {
            fn get_box_replacement(&self, opref: majit_ir::OpRef) -> majit_ir::OpRef {
                self.replacements
                    .get(&opref.raw())
                    .copied()
                    .unwrap_or(opref)
            }

            fn get_box_replacement_operand(
                &self,
                opref: majit_ir::OpRef,
            ) -> majit_ir::operand::Operand {
                // #160/S11: memoize one Operand per replacement-walked OpRef.
                let root = self.get_box_replacement(opref);
                if let Some(b) = self.box_cache.borrow().get(&root) {
                    return b.clone();
                }
                // Synthesize a rooted bound producer so the box sheds to
                // `Operand::Op`/`InputArg` for the Operand-keyed liveboxes map.
                let b = crate::history::test_support::rooted_operand_from_opref(root);
                self.box_cache.borrow_mut().insert(root, b.clone());
                b
            }

            fn get_box_replacement_not_const(&self, opref: majit_ir::OpRef) -> majit_ir::OpRef {
                self.get_box_replacement(opref)
            }

            fn is_const(&self, box_: &majit_ir::operand::Operand) -> bool {
                let opref = box_.to_opref();
                self.constants.contains_key(&opref.raw())
            }

            fn get_const(&self, box_: &majit_ir::operand::Operand) -> (i64, majit_ir::Type) {
                let opref = box_.to_opref();
                self.constants
                    .get(&opref.raw())
                    .copied()
                    .unwrap_or((0, majit_ir::Type::Int))
            }

            fn get_type(&self, opref: majit_ir::OpRef) -> majit_ir::Type {
                self.types
                    .get(&opref.raw())
                    .copied()
                    .unwrap_or(majit_ir::Type::Int)
            }

            fn is_virtual_ref(&self, box_: &majit_ir::operand::Operand) -> bool {
                let opref = box_.to_opref();
                self.virtuals.contains(&opref.raw())
            }

            fn is_virtual_raw(&self, _box: &majit_ir::operand::Operand) -> bool {
                false
            }

            fn get_virtual_fields(
                &self,
                opref: majit_ir::OpRef,
            ) -> Option<majit_ir::VirtualFieldsInfo> {
                self.virtual_fields.get(&opref.raw()).cloned()
            }
        }

        let mut memo = ResumeDataLoopMemo::new();
        let mut env = RefOnlyVirtualEnv::new();

        // RPython resume.py reads box.type after get_box_replacement().
        // Model a stale Int-typed snapshot slot that now forwards to a Ref
        // virtual, the shape produced by optimized boxed-int locals.
        let source = OpRef::int_op(1);
        let target = OpRef::ref_op(2);
        env.replacements.insert(source.raw(), target);
        env.virtuals.insert(target.raw());
        env.types.insert(target.raw(), majit_ir::Type::Ref);

        let snapshot = Snapshot::single_frame_boxes(
            0,
            10,
            vec![SnapshotBox::typed(source, majit_ir::Type::Int)],
        );
        let numb_state = memo.number(&snapshot, &env, -1).unwrap();
        let items = crate::resumecode::unpack_numbering(&numb_state.create_numbering());

        // items[6] = the frame's box.
        let (val, tagbits) = untag(items[6] as i16);
        assert_eq!(tagbits, TAGVIRTUAL);
        assert_eq!(val, 0);
        assert_eq!(numb_state.num_boxes, 0);
        assert_eq!(numb_state.num_virtuals, 1);
    }

    #[test]
    fn test_multi_frame_snapshot() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let env = SimpleBoxEnv::new();

        let snapshot = Snapshot {
            vable_array: vec![],
            vref_array: vec![],
            framestack: vec![
                SnapshotFrame {
                    jitcode_index: 0,
                    pc: 10,
                    py_pc: 10,
                    boxes: vec![OpRef::int_op(1).into(), OpRef::const_int(99).into()],
                },
                SnapshotFrame {
                    jitcode_index: 1,
                    pc: 20,
                    py_pc: 20,
                    boxes: vec![OpRef::int_op(2).into(), OpRef::int_op(3).into()],
                },
            ],
        };

        let mut numb_state = memo.number(&snapshot, &env, -1).unwrap();
        numb_state.writer.patch(1, numb_state.num_boxes);
        let rd_numb = numb_state.create_numbering();

        // Multi-frame encoding: no box_count, RPython parity.
        let items = crate::resumecode::unpack_numbering(&rd_numb);
        assert_eq!(items[1], 3); // num_failargs: 3 boxes patched
        // Frame 0: items[4]=jitcode(0), items[5]=pc(10), items[6..7]=tagged.
        assert_eq!(items[4], 0);
        assert_eq!(items[5], 10);
        // Frame 1: items[8]=jitcode(1), items[9]=pc(20).
        assert_eq!(items[8], 1);
        assert_eq!(items[9], 20);

        // Roundtrip with liveness-based closure.
        let rd_consts: Vec<majit_ir::Const> = memo.consts().to_vec();
        let frame_count = |jitcode_index: i32, _pc: i32| -> usize {
            match jitcode_index {
                0 => 2, // Frame 0 has 2 boxes
                1 => 2, // Frame 1 has 2 boxes
                _ => 0,
            }
        };
        let fail_arg_types = vec![
            majit_ir::Type::Int,
            majit_ir::Type::Int,
            majit_ir::Type::Int,
        ];
        let (num_failargs, _vable_values, _vref_values, rebuilt_frames) =
            rebuild_from_numbering(&rd_numb, &rd_consts, &fail_arg_types, Some(&frame_count), 0);
        assert_eq!(num_failargs, 3);
        assert_eq!(rebuilt_frames.len(), 2);
        assert_eq!(rebuilt_frames[0].jitcode_index, 0);
        assert_eq!(rebuilt_frames[0].pc, 10);
        assert_eq!(rebuilt_frames[0].values.len(), 2);
        assert_eq!(rebuilt_frames[1].jitcode_index, 1);
        assert_eq!(rebuilt_frames[1].pc, 20);
        assert_eq!(rebuilt_frames[1].values.len(), 2);
    }

    #[test]
    fn test_finish_produces_rd_numb_and_liveboxes() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.virtuals.insert(2);
        env.types.insert(2, majit_ir::Type::Ref);

        let snapshot = Snapshot::single_frame(
            0,
            8,
            vec![
                OpRef::const_int(42),
                OpRef::int_op(1),
                OpRef::ref_op(2),
                OpRef::int_op(3),
            ],
        );
        let numb_state = memo.number(&snapshot, &env, -1).unwrap();
        let (rd_numb, rd_consts, _rd_virtuals, liveboxes) =
            memo.finish(numb_state, &env, &mut [], None).unwrap();

        // liveboxes should contain only TAGBOX entries: OpRef::int_op(1) and OpRef::int_op(3)
        assert_eq!(liveboxes.len(), 2);
        assert_eq!(liveboxes[0].to_opref(), OpRef::int_op(1)); // box #0
        assert_eq!(liveboxes[1].to_opref(), OpRef::int_op(3)); // box #1

        // rd_numb should be valid
        let fail_arg_types = vec![majit_ir::Type::Int, majit_ir::Type::Int];
        let (num_failargs, _vable_values, _vref_values, rebuilt_frames) =
            rebuild_from_numbering(&rd_numb, &rd_consts, &fail_arg_types, None, 0);
        assert_eq!(num_failargs, 2);
        assert_eq!(rebuilt_frames.len(), 1);
        assert_eq!(
            rebuilt_frames[0].values[0],
            RebuiltValue::Const(majit_ir::Const::Int(42))
        );
        assert_eq!(
            rebuilt_frames[0].values[1],
            RebuiltValue::Box(0, majit_ir::Type::Int)
        );
        assert_eq!(rebuilt_frames[0].values[2], RebuiltValue::Virtual(0));
        assert_eq!(
            rebuilt_frames[0].values[3],
            RebuiltValue::Box(1, majit_ir::Type::Int)
        );
    }

    #[test]
    fn finish_preserves_canonical_numbered_box_identity() {
        // Every spelling of one replacement must return its canonical
        // producer so ResumeDataVirtualAdder.finish retains one box identity.
        use majit_ir::OpRef;

        struct KeepNumberedInputArgEnv {
            inner: SimpleBoxEnv,
            numbered: OpRef,
            collapsed: OpRef,
            collapsed_operand: majit_ir::operand::Operand,
        }

        impl BoxEnv for KeepNumberedInputArgEnv {
            fn get_box_replacement(&self, opref: OpRef) -> OpRef {
                if opref == self.numbered {
                    return self.numbered;
                }
                self.inner.get_box_replacement(opref)
            }

            fn get_box_replacement_operand(&self, opref: OpRef) -> majit_ir::operand::Operand {
                if opref == self.numbered || opref == self.collapsed {
                    return self.collapsed_operand.clone();
                }
                self.inner.get_box_replacement_operand(opref)
            }

            fn get_box_replacement_not_const(&self, _: OpRef) -> OpRef {
                panic!("finish must retain the numbered box, not resolve its position again")
            }

            fn is_const(&self, box_: &majit_ir::operand::Operand) -> bool {
                self.inner.is_const(box_)
            }
            fn get_const(&self, box_: &majit_ir::operand::Operand) -> (i64, majit_ir::Type) {
                self.inner.get_const(box_)
            }
            fn get_type(&self, opref: OpRef) -> majit_ir::Type {
                self.inner.get_type(opref)
            }
            fn is_virtual_ref(&self, box_: &majit_ir::operand::Operand) -> bool {
                self.inner.is_virtual_ref(box_)
            }
            fn is_virtual_raw(&self, box_: &majit_ir::operand::Operand) -> bool {
                self.inner.is_virtual_raw(box_)
            }
            fn get_virtual_fields(&self, opref: OpRef) -> Option<majit_ir::VirtualFieldsInfo> {
                self.inner.get_virtual_fields(opref)
            }
        }

        let numbered = OpRef::input_arg_ref(231);
        let collapsed = OpRef::input_arg_ref(1);
        let env = KeepNumberedInputArgEnv {
            inner: SimpleBoxEnv::new(),
            numbered,
            collapsed,
            collapsed_operand: crate::history::test_support::rooted_operand_from_opref(collapsed),
        };
        assert_eq!(env.get_box_replacement(numbered), numbered);
        assert_eq!(
            env.get_box_replacement_operand(numbered).to_opref(),
            collapsed
        );

        let mut memo = ResumeDataLoopMemo::new();
        let first = env.get_box_replacement_operand(numbered);
        let second = env.get_box_replacement_operand(collapsed);
        assert!(first.same_box(&second));
        assert_eq!(first.to_opref(), second.to_opref());
        let snapshot = Snapshot::single_frame(0, 8, vec![numbered, collapsed, numbered]);
        let numb_state = memo.number(&snapshot, &env, -1).unwrap();
        // `_number_boxes` inserts the canonical replacement producer.
        assert_eq!(
            numb_state
                .liveboxes
                .iter()
                .next()
                .map(|(op, _)| op.to_opref()),
            Some(collapsed),
        );
        let (rd_numb, rd_consts, _rd_virtuals, liveboxes) =
            memo.finish(numb_state, &env, &mut [], None).unwrap();

        // resume.py `_number_boxes` keys the canonical replacement object.
        assert_eq!(liveboxes.len(), 1);
        assert!(liveboxes[0].to_opref() == collapsed);
        let guard = majit_ir::Op::new(majit_ir::OpCode::GuardTrue, &[first.clone()]);
        guard.store_final_boxes(liveboxes);
        let (_, _, _, frames) =
            rebuild_from_numbering(&rd_numb, &rd_consts, &[majit_ir::Type::Ref; 1], None, 0);
        assert_eq!(
            frames[0].values,
            vec![
                RebuiltValue::Box(0, majit_ir::Type::Ref),
                RebuiltValue::Box(0, majit_ir::Type::Ref),
                RebuiltValue::Box(0, majit_ir::Type::Ref),
            ]
        );
        // initialize_state_from_guard_failure creates one recorder InputArg
        // per TAGBOX number. Repeated frame references must share that object
        // before bridge knowledge and optimization, not be merged by location
        // later in the backend.
        let mut recorder = crate::recorder::Trace::with_input_types(&[majit_ir::Type::Ref; 1]);
        let restored: Vec<_> = frames[0]
            .values
            .iter()
            .map(|value| match value {
                RebuiltValue::Box(index, tp) => {
                    recorder.box_for_operand(OpRef::input_arg_typed(*index as u32, *tp))
                }
                _ => unreachable!(),
            })
            .collect();
        assert!(restored[0].same_box(&restored[1]));
        assert!(restored[0].same_box(&restored[2]));
    }

    #[test]
    fn guard_storages_share_the_memos_growing_const_pool() {
        let mut memo = ResumeDataLoopMemo::new();
        let env = SimpleBoxEnv::new();

        let first = Snapshot::single_frame(0, 1, vec![OpRef::const_int(1)]);
        let first_state = memo.number(&first, &env, -1).unwrap();
        let (_, first_pool, _, _) = memo.finish(first_state, &env, &mut [], None).unwrap();
        assert!(first_pool.is_empty());

        let second = Snapshot::single_frame(0, 2, vec![OpRef::const_int(100_000)]);
        let second_state = memo.number(&second, &env, -1).unwrap();
        let (_, second_pool, _, _) = memo.finish(second_state, &env, &mut [], None).unwrap();

        assert!(std::sync::Arc::ptr_eq(&first_pool, &second_pool));
        assert_eq!(first_pool.as_slice(), &[majit_ir::Const::Int(100_000)]);
    }

    #[test]
    fn test_number_virtualizable_array_preserves_payload_then_identity_order() {
        use majit_ir::OpRef;

        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.types.insert(7, majit_ir::Type::Ref);
        env.types.insert(1, majit_ir::Type::Int);

        let snapshot = Snapshot {
            // pyjitpl.py:3326-3330 parity: payload slots first,
            // virtualizable identity (`virtualizable_boxes[-1]`) last.
            vable_array: vec![OpRef::int_op(1).into(), OpRef::ref_op(7).into()],
            vref_array: vec![],
            framestack: vec![SnapshotFrame {
                jitcode_index: 0,
                pc: 8,
                py_pc: 8,
                boxes: vec![OpRef::int_op(1).into()],
            }],
        };

        let numb_state = memo.number(&snapshot, &env, 0).unwrap();
        let items = crate::resumecode::unpack_numbering(&numb_state.create_numbering());

        assert_eq!(items[2], 2);
        let (val, tagbits) = untag(items[3] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 0);

        let (val, tagbits) = untag(items[4] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 1);

        assert_eq!(items[5], 0); // vref_array_length
        assert_eq!(items[6], 0); // jitcode_index
        assert_eq!(items[7], 8); // pc

        // The frame slot reuses the payload tag because numbering follows
        // Box identity exactly: upstream dedups only when the same Box object
        // appears twice, and in this test we passed the same OpRef twice.
        let (val, tagbits) = untag(items[8] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 0);
    }

    #[test]
    fn blackhole_from_resumedata_accepts_runtime_jitcode_without_canonical_pair() {
        use crate::blackhole::BlackholeInterpBuilder;
        use crate::jitcode::JitCodeBuilder;
        use crate::jitcode::insns::{BC_ABORT, BC_CATCH_EXCEPTION, BC_LIVE, BC_RVMPROF_CODE};

        let mut writer = crate::resumecode::Writer::new(7);
        writer.append_int(0); // items_resume_section (patched below)
        writer.append_int(0); // count: no failargs
        writer.append_int(0); // vable_array length
        writer.append_int(0); // vref_array length
        writer.append_int(0); // jitcode_pos
        writer.append_int(0); // pc
        writer.patch_current_size(0);
        let rd_numb = writer.create_numbering();

        let mut runtime = JitCodeBuilder::default().finish();
        runtime.body_mut().code = vec![BC_LIVE, 0, 0, BC_ABORT];
        runtime.body_mut().c_num_regs_i = 1;
        runtime.body_mut().constants_i = vec![321];
        // Hand-crafted body bypasses the builder's `start_instr` path,
        // so populate `startpoints` explicitly: BC_LIVE at 0,
        // BC_ABORT at 3.  RPython `jitcode.py:85-90` asserts
        // `pc in self._startpoints`.
        runtime.body_mut().startpoints = Some([0_usize, 3].into_iter().collect());
        let runtime = std::sync::Arc::new(runtime);

        let mut builder = BlackholeInterpBuilder::new();
        builder.setup_cached_control_opcodes(
            BC_LIVE as i32,
            BC_CATCH_EXCEPTION as i32,
            BC_RVMPROF_CODE as i32,
        );
        let resolve_jitcode = |_jitcode_pos: i32, _pc: i32| -> Option<ResolvedJitCode> {
            Some(ResolvedJitCode::new(runtime.clone(), 0))
        };

        let all_liveness: Vec<u8> = vec![0, 0, 0];
        let (bh, virtualizable_ptr) = blackhole_from_resumedata(
            &mut builder,
            &resolve_jitcode,
            &rd_numb,
            &[],
            &all_liveness,
            FailArgSource::Slice(&[]),
            None,
            None,
            None,
            None,
            None,
            None,
            None, // all_virtuals
            &NullAllocator,
        );

        assert_eq!(virtualizable_ptr, 0);
        assert!(std::sync::Arc::ptr_eq(&bh.jitcode, &runtime));
        assert_eq!(bh.position, 0);
        assert_eq!(bh.registers_i, vec![0, 321]);
    }

    struct TestVirtualizableInfo;

    impl VirtualizableInfo for TestVirtualizableInfo {
        fn get_total_size(&self, _virtualizable: i64) -> usize {
            0
        }

        fn reset_token_gcref(&self, _virtualizable: i64) {}

        fn write_from_resume_data_partial(
            &self,
            _virtualizable: i64,
            _reader: &mut ResumeDataDirectReader,
        ) {
        }
    }

    #[test]
    #[should_panic(expected = "vable_size > 0")]
    fn test_consume_vref_and_vable_asserts_zero_vable_size_when_vinfo_present() {
        let mut writer = crate::resumecode::Writer::new(3);
        writer.append_int(0); // items_resume_section (patched below)
        writer.append_int(0); // count
        writer.append_int(0); // vable_size
        writer.patch_current_size(0);
        let rd_numb = writer.create_numbering();

        let mut reader = ResumeDataDirectReader::new(
            &rd_numb,
            &[],
            &[],
            FailArgSource::Slice(&[]),
            None,
            None,
            &NullAllocator,
        );
        reader.consume_vref_and_vable(None, Some(&TestVirtualizableInfo), None);
    }

    /// resume.py:1368-1375 / compile.py:956-963 — the GUARD_NOT_FORCED resume
    /// path.  `handle_async_forcing` already materialized the virtuals and
    /// already rewrote the virtualizable, so this resume must reuse that cache
    /// and leave the virtualizable alone: `consume_vref_and_vable` jumps the
    /// vable and vref sections (resume.py) instead of consuming them.
    #[test]
    fn blackhole_from_resumedata_with_all_virtuals_skips_the_vable_section() {
        use crate::blackhole::BlackholeInterpBuilder;
        use crate::jitcode::JitCodeBuilder;
        use crate::jitcode::insns::{BC_ABORT, BC_CATCH_EXCEPTION, BC_LIVE, BC_RVMPROF_CODE};

        let mut writer = crate::resumecode::Writer::new(8);
        writer.append_int(0); // items_resume_section (patched below)
        writer.append_int(1); // count
        writer.append_int(1); // vable_size: the identity only (get_total_size == 0)
        writer.append_int(tag(0, TAGBOX).unwrap() as i64); // virtualizable identity
        writer.append_int(0); // vref_array length
        writer.append_int(0); // jitcode_pos
        writer.append_int(0); // pc
        writer.patch_current_size(0);
        let rd_numb = writer.create_numbering();

        let mut runtime = JitCodeBuilder::default().finish();
        runtime.body_mut().code = vec![BC_LIVE, 0, 0, BC_ABORT];
        runtime.body_mut().c_num_regs_i = 1;
        runtime.body_mut().constants_i = vec![321];
        runtime.body_mut().startpoints = Some([0_usize, 3].into_iter().collect());
        let runtime = std::sync::Arc::new(runtime);
        let all_liveness: Vec<u8> = vec![0, 0, 0];
        let deadframe = [0x4000_i64];
        let deadframe_types = [majit_ir::Type::Ref];

        let resume = |all_virtuals: Option<(Vec<i64>, Vec<i64>)>| {
            let mut builder = BlackholeInterpBuilder::new();
            builder.setup_cached_control_opcodes(
                BC_LIVE as i32,
                BC_CATCH_EXCEPTION as i32,
                BC_RVMPROF_CODE as i32,
            );
            let resolve_jitcode = |_jitcode_pos: i32, _pc: i32| -> Option<ResolvedJitCode> {
                Some(ResolvedJitCode::new(runtime.clone(), 0))
            };
            blackhole_from_resumedata(
                &mut builder,
                &resolve_jitcode,
                &rd_numb,
                &[],
                &all_liveness,
                FailArgSource::Slice(&deadframe),
                Some(&deadframe_types),
                None, // rd_virtuals
                None, // rd_guard_pendingfields
                None,
                Some(&TestVirtualizableInfo),
                None,
                all_virtuals,
                &NullAllocator,
            )
        };

        // resume.py:1427-1428: the ordinary path consumes the vable section, so
        // the reader surfaces the virtualizable the identity item named.
        let (bh, virtualizable_ptr) = resume(None);
        assert_eq!(virtualizable_ptr, 0x4000);
        assert_eq!(bh.position, 0);

        // resume.py:1433-1435: with a GUARD_NOT_FORCED cache the same items are
        // jumped — no virtualizable is surfaced, and the frame section behind
        // them still decodes, which is what proves the jump lengths line up.
        let (bh, virtualizable_ptr) = resume(Some((vec![0x1234], vec![7])));
        assert_eq!(virtualizable_ptr, 0);
        assert_eq!(bh.position, 0);
    }

    /// `resume.py` `decode_ref` leaves a `NULLREF` register null.
    /// `consume_vable_info` reads the virtualizable with `next_ref` and does
    /// not reuse that word for any other resume slot.
    fn leaf3_resume_null_identity() -> (i64, i64) {
        use crate::blackhole::BlackholeInterpBuilder;
        use crate::jitcode::JitCodeBuilder;
        use crate::jitcode::insns::{BC_ABORT, BC_CATCH_EXCEPTION, BC_LIVE, BC_RVMPROF_CODE};

        let mut writer = crate::resumecode::Writer::new(9);
        writer.append_int(0); // items_resume_section (patched below)
        writer.append_int(1); // count
        writer.append_int(1); // vable_size: the identity only (get_total_size == 0)
        writer.append_int(NULLREF as i64); // identity: folded out, so no box
        writer.append_int(0); // vref_array length
        writer.append_int(0); // jitcode_pos
        writer.append_int(0); // pc
        writer.append_int(NULLREF as i64); // frame ref register r0: null
        writer.patch_current_size(0);
        let rd_numb = writer.create_numbering();

        let mut runtime = JitCodeBuilder::default().finish();
        runtime.body_mut().code = vec![BC_LIVE, 0, 0, BC_ABORT];
        runtime.body_mut().c_num_regs_r = 1;
        runtime.body_mut().startpoints = Some([0_usize, 3].into_iter().collect());
        let runtime = std::sync::Arc::new(runtime);
        // length_i=0, length_r=1, length_f=0, then the live-register BITMASK
        // (`LivenessIterator` scans set bits, it does not read indices): bit 0
        // set names r0 as the one live ref register.
        let all_liveness: Vec<u8> = vec![0, 1, 0, 0b0000_0001];
        let deadframe = [0x4000_i64];
        let deadframe_types = [majit_ir::Type::Ref];

        let mut builder = BlackholeInterpBuilder::new();
        builder.setup_cached_control_opcodes(
            BC_LIVE as i32,
            BC_CATCH_EXCEPTION as i32,
            BC_RVMPROF_CODE as i32,
        );
        let resolve_jitcode = |_jitcode_pos: i32, _pc: i32| -> Option<ResolvedJitCode> {
            Some(ResolvedJitCode::new(runtime.clone(), 0))
        };
        let (bh, virtualizable_ptr) = blackhole_from_resumedata(
            &mut builder,
            &resolve_jitcode,
            &rd_numb,
            &[],
            &all_liveness,
            FailArgSource::Slice(&deadframe),
            Some(&deadframe_types),
            None, // rd_virtuals
            None, // rd_guard_pendingfields
            None, // vrefinfo
            Some(&TestVirtualizableInfo),
            None, // ginfo
            None, // all_virtuals
            &NullAllocator,
        );
        (virtualizable_ptr, bh.registers_r[0])
    }

    /// CONTROL for [`leaf3_resume_null_identity`]. With no override the
    /// substitution cannot happen, so this pins two things the subject test
    /// cannot: that the fixture REACHES the register (NULLREF survives the
    /// writer, the liveness names r0, and the seeding path runs), and that the
    /// refusal added to `consume_vable_info` does NOT fire without an override.
    ///
    /// Without it, a subject test that panics and a register that was never
    /// written are the same observation.
    #[test]
    fn a_null_ref_register_is_seeded_when_no_identity_override_is_supplied() {
        let (virtualizable_ptr, r0) = leaf3_resume_null_identity();
        assert_eq!(virtualizable_ptr, 0, "NULLREF identity decodes to null");
        assert_eq!(r0, 0, "control: r0 is seeded, and null");
    }

    /// resume.py `_prepare_virtuals` resets `virtuals_cache` to zeros.
    /// That is why `blackhole_from_resumedata` must not run `_prepare` on the
    /// GUARD_NOT_FORCED path (resume.py): there the preloaded cache is
    /// the resume's only source of virtuals, and `rd_virtuals` stays None.
    #[test]
    fn prepare_virtuals_resets_a_preloaded_guard_not_forced_cache() {
        let rd = majit_ir::RdVirtualInfo::VRawSliceInfo {
            offset: 0,
            fieldnums: vec![],
        };
        let virtuals = [virtual_info_from_rd(&rd)];
        let mut reader = ResumeDataDirectReader::new(
            &[0, 0],
            &[],
            &[],
            FailArgSource::Slice(&[]),
            None,
            Some((vec![0x1234], vec![7])),
            &NullAllocator,
        );
        assert_eq!(reader.resume_after_guard_not_forced, 2);
        assert_eq!(reader.virtuals_cache.get_ptr(0), 0x1234);
        assert_eq!(reader.virtuals_cache.get_int(0), 7);

        reader.prepare(Some(&virtuals), None);
        assert_eq!(reader.virtuals_cache.get_ptr(0), 0);
        assert_eq!(reader.virtuals_cache.get_int(0), 0);
    }

    #[test]
    #[should_panic(expected = "load_next_value_of_type: unexpected type Void")]
    fn test_next_value_of_type_rejects_void() {
        let mut reader = ResumeDataDirectReader::new(
            &[0, 0],
            &[],
            &[],
            FailArgSource::Slice(&[]),
            None,
            None,
            &NullAllocator,
        );
        let _ = reader.next_value_of_type(majit_ir::Type::Void);
    }

    #[test]
    fn rd_virtual_keeps_const_tag_until_direct_reader_decode() {
        // resume.py keeps AbstractVirtualInfo.fieldnums tagged and
        // calls decoder.setfield(..., num, ...). In particular it does not
        // copy ConstPtr.value out of rd_consts before allocate(), because that
        // allocation may collect and forward rd_consts.
        let tagged = tag(TAG_CONST_OFFSET, TAGCONST).unwrap();
        let rd = majit_ir::RdVirtualInfo::VRawSliceInfo {
            offset: 0,
            fieldnums: vec![tagged],
        };
        let mut consts = vec![majit_ir::Const::Ref(majit_ir::GcRef(0x1000))];
        let virtual_info = virtual_info_from_rd(&rd);
        let VirtualInfo::VRawSlice { parent, .. } = virtual_info else {
            panic!("expected VRawSlice");
        };
        assert_eq!(parent, ResumeValueSource::Tagged(tagged));

        // Model the root walker forwarding rd_consts during the allocation
        // window. The later field decode must observe the new pointer.
        consts[0] = majit_ir::Const::Ref(majit_ir::GcRef(0x2000));
        let mut reader = ResumeDataDirectReader::new(
            &[0, 0],
            &consts,
            &[],
            FailArgSource::Slice(&[]),
            None,
            None,
            &NullAllocator,
        );
        assert_eq!(reader.decode_field_source(&parent), 0x2000);
    }

    #[test]
    fn rd_virtual_normalizes_numbering_sentinels_before_decode() {
        assert_eq!(
            rd_virtual_field_source(UNINITIALIZED_TAG),
            ResumeValueSource::Uninitialized
        );
        assert_eq!(
            rd_virtual_field_source(UNASSIGNED),
            ResumeValueSource::Unavailable
        );
        assert_eq!(
            rd_virtual_field_source(UNASSIGNEDVIRTUAL),
            ResumeValueSource::Unavailable
        );
    }

    /// The TAGCONST field is 14 signed bits, so pool index 8192 is the first
    /// one it cannot name. Minting through `tag` makes that index an error;
    /// the shift-and-cast it replaced produced `-8192`, a tag that reads back
    /// as a *different* pool slot with no diagnostic anywhere.
    #[test]
    fn a_const_pool_past_the_tag_field_gives_up_instead_of_wrapping() {
        let mut memo = ResumeDataLoopMemo::new();
        // Floats never take the inline encoding, so each one is a pool entry.
        for i in 0..8192 {
            let tagged = memo
                .getconst_float(i as i64)
                .expect("index 0..8191 fits the 14-bit field");
            assert_eq!(untag(tagged), (i, TAGCONST));
        }
        assert_eq!(memo.consts().len(), 8192);

        // The whole getconst family gives up once the pool is full, and none
        // of them leaves a half-written entry behind: `newconst` mints the
        // tag before it pushes.
        assert!(memo.getconst_float(0).is_err());
        assert!(memo.getconst_ref(0x1234).is_err());
        assert!(memo.getconst_int(1 << 40).is_err());
        assert!(
            memo.getconst(2.5f64.to_bits() as i64, majit_ir::Type::Float)
                .is_err()
        );
        assert_eq!(memo.consts().len(), 8192);

        // A wrapped tag would have been silently accepted: it stays in i16
        // range and untags to a live pool slot that holds someone else's
        // constant.
        let wrapped = ((8192i32 << 2) | TAGCONST as i32) as i16;
        assert_eq!(untag(wrapped), (-8192, TAGCONST));
    }
}

// resume.py AbstractResumeDataReader
// resume.py ResumeDataDirectReader
//
// Direct reader that decodes resume data and fills blackhole
// interpreter registers with concrete values from the deadframe.

use crate::blackhole::BlackholeInterpreter;
use crate::resumecode::Reader;

/// RPython virtualref_info interface for resume data consumption.
///
/// Corresponds to `metainterp_sd.virtualref_info` (VirtualRefInfo).
pub trait VRefInfo {
    /// resume.py:1397 vrefinfo.continue_tracing(vref, virtual)
    fn continue_tracing(&self, vref: i64, virtual_ref: i64);
}

/// RPython virtualizable_info interface for resume data consumption.
///
/// Corresponds to `jitdriver_sd.virtualizable_info` (VirtualizableInfo).
pub trait VirtualizableInfo {
    /// resume.py:1406 vinfo.get_total_size(virtualizable)
    fn get_total_size(&self, virtualizable: i64) -> usize;

    /// resume.py:1407 vinfo.reset_token_gcref(virtualizable)
    fn reset_token_gcref(&self, virtualizable: i64);

    /// pyre: register virtualizable-owned Ref buffers for the resume
    /// construction window. RPython keeps the virtualizable object and its
    /// array fields GC-traced while `blackhole_from_resumedata` fills frames.
    fn push_resume_ref_roots(&self, _virtualizable: i64) {}

    /// pyre: register virtualizable-owned Ref buffers for any virtualizable
    /// object already present in a blackhole Ref register bank.
    fn push_resume_ref_roots_for_value(&self, _value: i64) {}

    /// False for a state-field machine with no heap `vable_token`.
    /// `consume_one_section` asks once so it can skip the per-ref dyn call.
    fn has_vable_token(&self) -> bool {
        false
    }

    /// pyre: register virtualizable-owned Ref buffers for any virtualizable
    /// object already present in a blackhole Ref register bank.
    fn push_resume_ref_roots_for_registers(&self, registers_r: &[i64]) {
        for &value in registers_r {
            self.push_resume_ref_roots_for_value(value);
        }
    }

    /// resume.py:1408 vinfo.write_from_resume_data_partial(virtualizable, self)
    ///
    /// Read fields from the resume reader and write them into the virtualizable.
    fn write_from_resume_data_partial(
        &self,
        virtualizable: i64,
        reader: &mut ResumeDataDirectReader,
    );
}

/// RPython greenfield_info interface for resume data consumption.
///
/// Corresponds to `jitdriver_sd.greenfield_info`.
pub trait GreenfieldInfo {}

/// resume.py ResumeDataDirectReader
///
/// Reads encoded resume data (rd_numb) and fills blackhole interpreter
/// resume.py AbstractVirtualCache / get_VirtualCache_class
///
/// ```text
/// class AbstractVirtualCache(object):
///     pass
///
/// def get_VirtualCache_class(suffix):
///     class VirtualCache(AbstractVirtualCache):
///         def __init__(self, virtuals_ptr_cache, virtuals_int_cache):
///             self.virtuals_ptr_cache = virtuals_ptr_cache
///             self.virtuals_int_cache = virtuals_int_cache
///
///         def get_ptr(self, i):  return self.virtuals_ptr_cache[i]
///         def get_int(self, i):  return self.virtuals_int_cache[i]
///         def set_ptr(self, i, v): self.virtuals_ptr_cache[i] = v
///         def set_int(self, i, v): self.virtuals_int_cache[i] = v
/// ```
///
/// RPython generates two flavours of this class — one for
/// `ResumeDataDirectReader` (raw `i64` slots) and one for the future
/// `ResumeDataBoxReader` (boxed). majit only emits the direct flavour at
/// runtime, so a single struct backs both.
#[derive(Default)]
pub struct VirtualCache {
    pub virtuals_ptr_cache: Vec<i64>,
    pub virtuals_int_cache: Vec<i64>,
}

impl VirtualCache {
    pub fn new() -> Self {
        VirtualCache::default()
    }

    /// resume.py __init__
    pub fn from_caches(virtuals_ptr_cache: Vec<i64>, virtuals_int_cache: Vec<i64>) -> Self {
        VirtualCache {
            virtuals_ptr_cache,
            virtuals_int_cache,
        }
    }

    /// resume.py get_ptr
    #[inline]
    pub fn get_ptr(&self, i: usize) -> i64 {
        self.virtuals_ptr_cache[i]
    }

    /// resume.py get_int
    #[inline]
    pub fn get_int(&self, i: usize) -> i64 {
        self.virtuals_int_cache[i]
    }

    /// resume.py set_ptr
    #[inline]
    pub fn set_ptr(&mut self, i: usize, v: i64) {
        self.virtuals_ptr_cache[i] = v;
    }

    /// resume.py set_int
    #[inline]
    pub fn set_int(&mut self, i: usize, v: i64) {
        self.virtuals_int_cache[i] = v;
    }

    /// `len(self.virtuals_ptr_cache)` — both halves stay the same length.
    pub fn len(&self) -> usize {
        self.virtuals_ptr_cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.virtuals_ptr_cache.is_empty()
    }
}

/// registers directly from the deadframe's fail_args values.
///
/// Combines AbstractResumeDataReader (resume.py) mixin with
/// ResumeDataDirectReader (resume.py) concrete class.
pub struct ResumeDataDirectReader<'a> {
    // AbstractResumeDataReader fields (resume.py)
    /// resume.py resumecodereader
    pub resumecodereader: Reader<'a>,
    /// resume.py:919 items_resume_section — total items in resume section
    pub items_resume_section: i32,
    /// resume.py count — number of failargs
    pub count: i32,
    /// resume.py:922 consts — constant pool from rd_consts.
    ///
    /// RPython stores `list[Const]` where each `Const` carries its own type
    /// (history.py ConstInt / :261 ConstFloat / :307 ConstPtr). In the
    /// Rust port we represent each entry as `(raw_i64, Type)` so that
    /// `ConstPtr.getref_base()` parity (returning a GC-tracked pointer) can
    /// be surfaced to the minor-collection root walker — see
    /// `walk_rd_consts_refs` on `MetaInterp`. Raw `Vec<i64>` hid the
    /// Ref-typed entries from the GC and caused nursery use-after-free in
    /// TAGCONST decode paths (resume.py decode_int / :1566 decode_ref /
    /// :1578 decode_float).
    pub consts: &'a [majit_ir::Const],

    // ResumeDataDirectReader fields (resume.py)
    /// resume.py `ResumeDataDirectReader.deadframe` — `cpu.get_int_value`
    pub deadframe: FailArgSource<'a>,
    /// pyre flat-deadframe adaptation: original type of each deadframe slot.
    /// RPython's CPU exposes typed getters (get_ref_value/get_int_value/...);
    /// pyre passes a flat raw slice and needs slot kinds to emulate
    /// load_box_from_cpu(kind) for TAGBOX decode.
    pub deadframe_types: Option<&'a [majit_ir::Type]>,

    // resume.py:1358 resume_after_guard_not_forced
    //   0: not a GUARD_NOT_FORCED
    //   1: in handle_async_forcing
    //   2: resuming from the GUARD_NOT_FORCED
    pub resume_after_guard_not_forced: u8,

    // resume.py:909 rd_virtuals
    rd_virtuals: Option<&'a [VirtualInfo]>,

    /// resume.py:910 virtuals_cache — lazy-allocated virtual objects.
    /// Wraps both the ptr and int half so callers go through the RPython
    /// `VirtualCache` API (`get_ptr`/`set_ptr`/`get_int`/`set_int`).
    pub virtuals_cache: VirtualCache,

    virtualizable_root_base_depth: Option<usize>,

    /// resume.py:1367 — CPU allocation backend.
    /// RPython uses self.cpu (from metainterp_sd.cpu) for allocate_with_vtable etc.
    allocator: &'a dyn BlackholeAllocator,

    /// resume.py `self.metainterp_sd.liveness_info` — shared
    /// packed `all_liveness` buffer used by `_prepare_next_section` /
    /// `enumerate_vars`. RPython reaches it through `self.metainterp_sd`;
    /// pyre holds the slice directly because `ResumeDataDirectReader`
    /// lives outside the `MetaInterpStaticData` ownership graph.
    pub all_liveness: &'a [u8],

    /// resume.py: virtualizable pointer read by consume_vable_info.
    /// Stored so the caller (blackhole_from_resumedata) can access it
    /// after consume_vref_and_vable completes.
    pub virtualizable_ptr: i64,
}

/// resume.py:1433-1456 CPU allocation interface for virtual materialization.
///
/// ResumeDataDirectReader calls these methods when TAGVIRTUAL values
/// need to be lazily allocated during decode_ref/decode_int.
pub trait BlackholeAllocator {
    /// resume.py allocate_with_vtable(known_class, descr) →
    ///   exec_new_with_vtable(self.cpu, descr)
    fn allocate_with_vtable(&self, descr: &majit_ir::DescrRef, vtable: usize) -> i64 {
        let _ = (descr, vtable);
        0
    }
    /// resume.py:1442 cpu.bh_new(typedescr)
    fn bh_new(&self, typedescr: &majit_ir::DescrRef) -> i64 {
        let _ = typedescr;
        0
    }
    /// resume.py:1446 cpu.bh_new_array_clear(length, arraydescr)
    fn bh_new_array_clear(&self, length: usize, arraydescr: &majit_ir::DescrRef) -> i64 {
        let _ = (length, arraydescr);
        0
    }
    /// resume.py:1447 cpu.bh_new_array(length, arraydescr)
    fn bh_new_array(&self, length: usize, arraydescr: &majit_ir::DescrRef) -> i64 {
        let _ = (length, arraydescr);
        0
    }
    /// resume.py:1533 cpu.bh_setarrayitem_gc_i(array, i, value, arraydescr)
    fn bh_setarrayitem_gc_i(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let _ = (array, index, value, descr);
    }
    /// resume.py:1537 cpu.bh_setarrayitem_gc_r(array, i, value, arraydescr)
    fn bh_setarrayitem_gc_r(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let _ = (array, index, value, descr);
    }
    /// resume.py:1541 cpu.bh_setarrayitem_gc_f(array, i, value, arraydescr)
    fn bh_setarrayitem_gc_f(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let _ = (array, index, value, descr);
    }
    /// resume.py setinteriorfield(index, array, fieldnum, descr)
    /// RPython passes the live descr object; backend reads offset/size/type from it.
    fn bh_setinteriorfield_gc_i(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let _ = (array, index, value, descr);
    }
    fn bh_setinteriorfield_gc_r(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let _ = (array, index, value, descr);
    }
    fn bh_setinteriorfield_gc_f(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let _ = (array, index, value, descr);
    }
    /// resume.py allocate_string(length) → cpu.bh_newstr(length)
    fn bh_newstr(&self, length: usize) -> i64 {
        let _ = length;
        0
    }
    /// resume.py string_setitem(str, index, charnum) →
    /// cpu.bh_strsetitem(str, index, char) — `char` is the decoded
    /// integer from the tagged `charnum`.
    fn bh_strsetitem(&self, string: i64, index: usize, char: i64) {
        let _ = (string, index, char);
    }
    /// resume.py concat_strings(str1, str2) — implementations
    /// look up `OS_STR_CONCAT` via `callinfocollection.funcptr_for_oopspec`
    /// (resume.py) and call it directly.  The variant carries
    /// no funcptr.
    fn os_str_concat(&self, str1: i64, str2: i64) -> i64 {
        let _ = (str1, str2);
        0
    }
    /// resume.py slice_string(str, start, length) →
    /// `funcptr_for_oopspec(OS_STR_SLICE)(str, start, stop)` where the
    /// caller pre-computes `stop = start + length` (the OS_STR_SLICE
    /// oopspec signature).  Implementations resolve the funcptr via
    /// `callinfocollection`.
    fn os_str_slice(&self, str: i64, start: i64, stop: i64) -> i64 {
        let _ = (str, start, stop);
        0
    }
    /// resume.py allocate_unicode(length) →
    /// cpu.bh_newunicode(length)
    fn bh_newunicode(&self, length: usize) -> i64 {
        let _ = length;
        0
    }
    /// resume.py unicode_setitem(str, index, charnum) →
    /// cpu.bh_unicodesetitem(str, index, char)
    fn bh_unicodesetitem(&self, string: i64, index: usize, char: i64) {
        let _ = (string, index, char);
    }
    /// resume.py concat_unicodes(str1, str2) →
    /// `funcptr_for_oopspec(OS_UNI_CONCAT)(str1, str2)`. Implementations
    /// resolve the funcptr via `callinfocollection`.
    fn os_uni_concat(&self, str1: i64, str2: i64) -> i64 {
        let _ = (str1, str2);
        0
    }
    /// resume.py slice_unicode(str, start, length) →
    /// `funcptr_for_oopspec(OS_UNI_SLICE)(str, start, stop)` where the
    /// caller pre-computes `stop = start + length`.  Implementations
    /// resolve the funcptr via `callinfocollection`.
    fn os_uni_slice(&self, str: i64, start: i64, stop: i64) -> i64 {
        let _ = (str, start, stop);
        0
    }
    /// resume.py allocate_raw_buffer(func, size)
    fn allocate_raw_buffer(&self, func: i64, size: usize) -> i64 {
        let _ = (func, size);
        0
    }
    /// resume.py:1547 cpu.bh_raw_store_f(buffer, offset, value, descr) —
    /// float raw store dispatched from setrawbuffer_item when
    /// `descr.is_array_of_floats()`.  `offset` mirrors
    /// `RawBuffer.offsets[i]` and is signed (rawbuffer.py).
    fn bh_raw_store_f(
        &self,
        buffer: i64,
        offset: i64,
        value: i64,
        descr: &majit_ir::ArrayDescrInfo,
    ) {
        let _ = (buffer, offset, value, descr);
    }
    /// resume.py:1550 cpu.bh_raw_store_i(buffer, offset, value, descr) —
    /// integer raw store dispatched from setrawbuffer_item (default
    /// branch — descr is not an array of pointers / floats).
    fn bh_raw_store_i(
        &self,
        buffer: i64,
        offset: i64,
        value: i64,
        descr: &majit_ir::ArrayDescrInfo,
    ) {
        let _ = (buffer, offset, value, descr);
    }
    /// `resume.py:1517 cpu.bh_setfield_gc_i(struct, value, descr)` —
    /// integer setfield dispatched from `resume.py setfield`.
    fn bh_setfield_gc_i(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        let _ = (struct_ptr, value, descr_info);
    }
    /// `resume.py:1512 cpu.bh_setfield_gc_r(struct, value, descr)` —
    /// pointer setfield dispatched when `descr.is_pointer_field()`.
    fn bh_setfield_gc_r(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        let _ = (struct_ptr, value, descr_info);
    }
    /// `resume.py:1515 cpu.bh_setfield_gc_f(struct, value, descr)` —
    /// float setfield dispatched when `descr.is_float_field()`.
    fn bh_setfield_gc_f(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        let _ = (struct_ptr, value, descr_info);
    }
    /// Pyre-specific: box a raw int to a PyObject ref.
    ///
    /// RPython equivalent: cpu.get_ref_value always returns GCREF because
    /// the jitframe stores typed values. Pyre's deadframe is untyped i64;
    /// when a slot typed as Int is read through decode_ref, this method
    /// wraps it into a valid GCREF (W_IntObject).
    fn box_int(&self, value: i64) -> i64 {
        value // default: return raw value (override in pyre allocator)
    }
    /// Pyre-specific: box raw float bits to a PyObject ref.
    fn box_float(&self, value: i64) -> i64 {
        value
    }
}

/// Default no-op allocator.
pub struct NullAllocator;
impl BlackholeAllocator for NullAllocator {}

/// `cpu.bh_setfield_gc_*` / `cpu.bh_setarrayitem_gc_*` (`llmodel.py`).
///
/// The applying reader records a deferred store *and* writes it. A
/// frontend that has no object-model override still owes those writes
/// — otherwise `start_bridge_tracing` marks the replay incomplete and
/// every pending field declines to the blackhole.
pub struct LlmodelBlackholeAllocator;

/// `cpu.bh_new` / `bh_new_array` (`llmodel.py`): old-gen when a GC owns
/// the heap, otherwise a zeroed raw block. A GC that is installed but
/// cannot satisfy the request returns 0 rather than a raw pointer the
/// collector will not trace.
fn llmodel_alloc(type_id: u32, size: usize) -> i64 {
    let size = size.max(1);
    // Same order as `BC_NEW_ARRAY`: typed allocator, then the published
    // hook (`GcLLDescr_boehm.malloc_fixedsize`), then the raw allocator.
    // `GC_malloc` is zero-filled; do not clear the hook's block again.
    let gc = majit_gc::alloc_oldgen_typed(type_id, size);
    if gc.0 != 0 {
        return gc.0 as i64;
    }
    let malloc_addr = majit_gc::malloc_fixedsize_addr();
    if malloc_addr != 0 {
        let malloc: extern "C" fn(usize) -> *mut u8 = unsafe { std::mem::transmute(malloc_addr) };
        let ptr = malloc(size);
        if ptr.is_null() {
            return 0;
        }
        return ptr as i64;
    }
    if majit_gc::gc_allocator_installed() {
        return 0;
    }
    let mut block = vec![0u8; size].into_boxed_slice();
    let ptr = block.as_mut_ptr();
    std::mem::forget(block);
    ptr as i64
}

impl BlackholeAllocator for LlmodelBlackholeAllocator {
    fn allocate_with_vtable(&self, descr: &majit_ir::DescrRef, vtable: usize) -> i64 {
        let Some(sd) = descr.as_size_descr() else {
            return 0;
        };
        let ptr = llmodel_alloc(sd.type_id(), sd.size());
        if ptr != 0 && vtable != 0 {
            unsafe {
                majit_backend::llmodel::write_ref_at_mem(ptr as usize, 0, vtable);
            }
        }
        ptr
    }

    fn bh_new(&self, typedescr: &majit_ir::DescrRef) -> i64 {
        let Some(sd) = typedescr.as_size_descr() else {
            return 0;
        };
        llmodel_alloc(sd.type_id(), sd.size())
    }

    fn bh_new_array(&self, length: usize, arraydescr: &majit_ir::DescrRef) -> i64 {
        let Some(ad) = arraydescr.as_array_descr() else {
            return 0;
        };
        let size = ad
            .base_size()
            .saturating_add(length.saturating_mul(ad.item_size()));
        let ptr = llmodel_alloc(ad.type_id(), size);
        if ptr != 0 {
            if let Some(ld) = ad.len_descr() {
                // `llmodel.py bh_new_array` → `gc_malloc_array` writes
                // `length` at `arraydescr.lendescr`. Dynasm does the same
                // via `dynasm_alloc_oldgen_varsize_typed_and_set_len`.
                unsafe {
                    majit_backend::llmodel::write_int_at_mem(
                        ptr as usize,
                        ld.offset(),
                        ld.field_size(),
                        length as i64,
                    );
                }
            }
        }
        ptr
    }

    fn bh_new_array_clear(&self, length: usize, arraydescr: &majit_ir::DescrRef) -> i64 {
        self.bh_new_array(length, arraydescr)
    }
    fn bh_setfield_gc_i(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        if struct_ptr == 0 {
            return;
        }
        unsafe {
            majit_backend::llmodel::write_int_at_mem(
                struct_ptr as usize,
                descr_info.offset,
                descr_info.field_size,
                value,
            );
        }
    }

    fn bh_setfield_gc_r(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        if struct_ptr == 0 {
            return;
        }
        // Barrier first: `gc_write_barrier` must run before the store so
        // a concurrent minor cannot miss the young referent.
        majit_gc::gc_write_barrier(majit_ir::GcRef(struct_ptr as usize));
        unsafe {
            majit_backend::llmodel::write_ref_at_mem(
                struct_ptr as usize,
                descr_info.offset,
                value as usize,
            );
        }
    }

    fn bh_setfield_gc_f(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        if struct_ptr == 0 {
            return;
        }
        unsafe {
            majit_backend::llmodel::write_float_at_mem(
                struct_ptr as usize,
                descr_info.offset,
                f64::from_bits(value as u64),
            );
        }
    }

    fn bh_setarrayitem_gc_i(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let Some(ad) = descr.as_array_descr() else {
            return;
        };
        let ofs = ad
            .base_size()
            .wrapping_add(index.wrapping_mul(ad.item_size()));
        unsafe {
            majit_backend::llmodel::write_int_at_mem(array as usize, ofs, ad.item_size(), value);
        }
    }

    fn bh_setarrayitem_gc_r(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let Some(ad) = descr.as_array_descr() else {
            return;
        };
        let ofs = ad
            .base_size()
            .wrapping_add(index.wrapping_mul(ad.item_size()));
        majit_gc::gc_write_barrier(majit_ir::GcRef(array as usize));
        unsafe {
            majit_backend::llmodel::write_ref_at_mem(array as usize, ofs, value as usize);
        }
    }

    fn bh_setarrayitem_gc_f(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let Some(ad) = descr.as_array_descr() else {
            return;
        };
        let ofs = ad
            .base_size()
            .wrapping_add(index.wrapping_mul(ad.item_size()));
        unsafe {
            majit_backend::llmodel::write_float_at_mem(
                array as usize,
                ofs,
                f64::from_bits(value as u64),
            );
        }
    }
}

/// Metainterp-side extension methods on `VirtualInfo` (which lives in
/// majit-backend since the Phase C-1 cascade).  These methods depend
/// on `ResumeDataDirectReader` + `BlackholeAllocator` — both
/// metainterp-specific — so they stay here as a trait extension.
pub trait VirtualInfoBlackholeExt {
    fn is_about_raw(&self) -> bool;
    /// `resume.py if rd_virtual is not None`: detect the
    /// `RdVirtualInfo::Empty` placeholder propagated as a zero-shaped
    /// `VirtualObj` (the `rd_virtual_to_virtual_info` conversion). Used by
    /// `force_all_virtuals` to skip slots that PyPy keeps as `None`.
    fn is_empty_placeholder(&self) -> bool;
    fn allocate(
        &self,
        decoder: &mut ResumeDataDirectReader,
        index: usize,
        allocator: &dyn BlackholeAllocator,
    ) -> i64;
    fn allocate_int(
        &self,
        decoder: &mut ResumeDataDirectReader,
        index: usize,
        allocator: &dyn BlackholeAllocator,
    ) -> i64;
}

#[inline]
fn virtual_source_is_uninitialized(source: &VirtualFieldSource) -> bool {
    match source {
        ResumeValueSource::Uninitialized => true,
        ResumeValueSource::Tagged(tagged) => tagged_eq(*tagged, UNINITIALIZED_TAG),
        _ => false,
    }
}

/// `resume.py VStrPlainInfo.allocate` and `resume.py
/// VUniPlainInfo.allocate` share the same loop body — the only
/// difference is `decoder.allocate_string` vs `allocate_unicode` and
/// `string_setitem` vs `unicode_setitem`.  The pyre helper takes an
/// `is_unicode` flag to dispatch.  `chars` carries one
/// `VirtualFieldSource` per character (the pyre equivalent of RPython's
/// tagged `fieldnums`); `VirtualFieldSource::Uninitialized` matches
/// resume.py's `tagged_eq(charnum, UNINITIALIZED)` skip.
fn vstr_plain_info_allocate(
    decoder: &mut ResumeDataDirectReader,
    index: usize,
    chars: &[VirtualFieldSource],
    is_unicode: bool,
) -> i64 {
    let length = chars.len();
    // resume.py string = decoder.allocate_string(length)
    // resume.py string = decoder.allocate_unicode(length)
    let string = if is_unicode {
        decoder.allocate_unicode(length)
    } else {
        decoder.allocate_string(length)
    };
    // resume.py / 825 decoder.virtuals_cache.set_ptr(index, string)
    decoder.virtuals_cache.set_ptr(index, string);
    for (i, char_source) in chars.iter().enumerate() {
        // resume.py / 828 if not tagged_eq(charnum, UNINITIALIZED)
        if virtual_source_is_uninitialized(char_source) {
            continue;
        }
        // resume.py:774 / 829 decoder.{string,unicode}_setitem(string, i, charnum)
        if is_unicode {
            decoder.unicode_setitem(string, i, char_source);
        } else {
            decoder.string_setitem(string, i, char_source);
        }
    }
    string
}

/// Recover the live `FieldDescr` `AbstractResumeDataReader.setfield`
/// would receive. Snapshot `FieldDescrInfo` is the spec form; the parent
/// SizeDescr still owns the canonical descr (`is_pointer_field` /
/// `is_float_field`).
///
/// `index`+offset is unique for a FLAG_POINTER Int whose snapshot
/// `field_type` was stored as Int. Two flattened leaves can share that
/// pair (`stable_field_index`); then keep the snapshot bank
/// (`snapshot_field_descr_info`).
pub(crate) fn live_field_descr_for_resume(
    parent: Option<&majit_ir::DescrRef>,
    i: usize,
    info: &majit_ir::FieldDescrInfo,
) -> Option<majit_ir::DescrRef> {
    let sd = parent?.as_size_descr()?;
    let fields = sd.all_fielddescrs();
    // `AbstractVirtualStructInfo.setfields` reads `self.fielddescrs[i]`,
    // and the snapshot was written from `descr.get_all_fielddescrs()` in
    // order. Two fields that overlay one word share the offset, so only
    // the position names the row.
    if let Some(fd) = fields
        .get(i)
        .filter(|fd| fd.index() == info.index && fd.offset() == info.offset)
    {
        return Some(std::sync::Arc::clone(fd) as majit_ir::DescrRef);
    }
    let want_ptr = info.field_type == majit_ir::Type::Ref;
    let want_float = info.field_type == majit_ir::Type::Float;
    let same_index_offset: Vec<&std::sync::Arc<dyn majit_ir::FieldDescr>> = fields
        .iter()
        .filter(|fd| fd.index() == info.index && fd.offset() == info.offset)
        .collect();
    if same_index_offset.len() == 1 {
        return Some(std::sync::Arc::clone(same_index_offset[0]) as majit_ir::DescrRef);
    }
    if let Some(fd) = same_index_offset
        .iter()
        .find(|fd| fd.is_pointer_field() == want_ptr && fd.is_float_field() == want_float)
    {
        return Some(std::sync::Arc::clone(*fd) as majit_ir::DescrRef);
    }
    fields
        .iter()
        .find(|fd| {
            fd.offset() == info.offset
                && fd.is_pointer_field() == want_ptr
                && fd.is_float_field() == want_float
        })
        .map(|fd| std::sync::Arc::clone(fd) as majit_ir::DescrRef)
}

/// `resume.py AbstractResumeDataReader.setfield` kind:
/// `is_pointer_field` → REF, `is_float_field` → FLOAT, else INT.
/// Snapshot (`snapshot_field_descr_info`) writes the same split, so a
/// Ref/Float snapshot is the bank `fieldnums[i]` was tagged with — do
/// not decode_int it because a colliding live Int sibling won the
/// parent-list lookup. Live flags only *upgrade* an Int snapshot
/// (FLAG_POINTER, or `field_type` Ref without FLAG_POINTER).
fn resume_setfield_kind(
    live: Option<&majit_ir::DescrRef>,
    info: &majit_ir::FieldDescrInfo,
) -> majit_ir::Type {
    if info.field_type == majit_ir::Type::Ref || info.field_type == majit_ir::Type::Float {
        return info.field_type;
    }
    if let Some(fd) = live.and_then(|d| d.as_field_descr()) {
        if fd.is_pointer_field() {
            majit_ir::Type::Ref
        } else if fd.is_float_field() {
            majit_ir::Type::Float
        } else {
            fd.field_type()
        }
    } else {
        info.field_type
    }
}

/// `AbstractVirtualStructInfo.setfields` is
/// `self.fieldnums[i]` with `self.fielddescrs[i]`. Encode
/// (`get_virtual_fields` / `parent_list_slot_in`) already lined those
/// two snapshot lists up; rematching by `FieldDescrInfo.index` or the
/// live parent list would re-cross a 1:1 zip when two leaves share a
/// hash. `VArrayStructInfo.allocate` already walks `fielddescrs[j]`
/// with `element_fields[j]` the same way.
fn virtual_struct_field_source(
    fields: &[(u32, VirtualFieldSource)],
    enumerate_i: usize,
) -> Option<&VirtualFieldSource> {
    fields.get(enumerate_i).map(|(_, src)| src)
}

/// `resume.py AbstractVirtualStructInfo.setfields(decoder, struct)`
/// — iterate fielddescrs/fieldnums and call decoder.setfield per
/// non-UNINITIALIZED entry.  pyre threads the spec-form `FieldDescrInfo`
/// through `bh_setfield_gc_{i,r,f}` because the descr Arc is not interned
/// alongside the spec on `VirtualInfo::{VirtualObj, VStruct}.fielddescrs`.
/// Decode kind follows `AbstractResumeDataReader.setfield` on the live
/// FieldDescr recovered from the parent SizeDescr, not the snapshot
/// `field_type` alone (a FLAG_POINTER Int field, or a snapshot fallback
/// that stored `Type::Int`, still decodes as Ref).
fn abstract_virtual_struct_info_setfields(
    decoder: &mut ResumeDataDirectReader,
    allocator: &dyn BlackholeAllocator,
    index: usize,
    parent: Option<&majit_ir::DescrRef>,
    fielddescrs: &[majit_ir::FieldDescrInfo],
    fields: &[(u32, VirtualFieldSource)],
) {
    // resume.py AbstractVirtualStructInfo.setfields:
    // for i in range(len(self.fielddescrs)):
    //     decoder.setfield(struct, self.fieldnums[i], self.fielddescrs[i])
    for (i, descr_info) in fielddescrs.iter().enumerate() {
        let live = live_field_descr_for_resume(parent, i, descr_info);
        let Some(source) = virtual_struct_field_source(fields, i) else {
            continue;
        };
        // resume.py if not tagged_eq(num, UNINITIALIZED)
        if virtual_source_is_uninitialized(source) {
            continue;
        }
        // resume.py decoder.setfield(struct, num, descr)
        // AbstractResumeDataReader.setfield: is_pointer_field / is_float_field / else.
        //
        // decode_field_source* may materialize a nested virtual, whose
        // allocation can trigger a minor collection that relocates this
        // struct.  RPython's `struct` local is GC-traced and forwarded in
        // place; pyre's is a raw i64, so re-read the forwarded pointer from
        // the rooted virtuals_ptr_cache[index] slot after each decode and
        // before the write (see getvirtual_ptr at the `bad cache` comment).
        match resume_setfield_kind(live.as_ref(), descr_info) {
            majit_ir::Type::Ref => {
                let value = decoder.decode_field_source(source);
                let struct_ptr = decoder.virtuals_cache.get_ptr(index);
                allocator.bh_setfield_gc_r(struct_ptr, value, descr_info);
            }
            majit_ir::Type::Float => {
                let value = decoder.decode_field_source_float(source);
                let struct_ptr = decoder.virtuals_cache.get_ptr(index);
                allocator.bh_setfield_gc_f(struct_ptr, value, descr_info);
            }
            _ => {
                let value = decoder.decode_field_source_int(source);
                let struct_ptr = decoder.virtuals_cache.get_ptr(index);
                allocator.bh_setfield_gc_i(struct_ptr, value, descr_info);
            }
        }
    }
}

impl VirtualInfoBlackholeExt for VirtualInfo {
    /// resume.py AbstractVirtualInfo kind attribute — REF for object/struct/array/string,
    /// INT for raw buffers.
    fn is_about_raw(&self) -> bool {
        matches!(
            self,
            VirtualInfo::VRawBuffer { .. } | VirtualInfo::VRawSlice { .. }
        )
    }

    fn is_empty_placeholder(&self) -> bool {
        // resume.py `if rd_virtual is not None`
        matches!(self, VirtualInfo::Empty)
    }

    /// resume.py/634/650 allocate(decoder, index)
    ///
    /// Allocate a virtual object and fill in its fields from the decoder.
    /// Sets `virtuals_cache_ptr[index]` before filling fields (for recursive refs).
    fn allocate(
        &self,
        decoder: &mut ResumeDataDirectReader,
        index: usize,
        allocator: &dyn BlackholeAllocator,
    ) -> i64 {
        match self {
            VirtualInfo::VirtualObj {
                fields,
                fielddescrs,
                descr,
                known_class,
                ..
            } => {
                // resume.py struct = decoder.allocate_with_vtable(descr=self.descr)
                let vtable = known_class.unwrap_or(0) as usize;
                let obj = descr
                    .as_ref()
                    .map(|d| decoder.allocate_with_vtable(d, vtable))
                    .unwrap_or(0);
                decoder.virtuals_cache.set_ptr(index, obj);
                // resume.py return self.setfields(decoder, struct)
                abstract_virtual_struct_info_setfields(
                    decoder,
                    allocator,
                    index,
                    descr.as_ref(),
                    fielddescrs,
                    fields,
                );
                // re-read the forwarded pointer (setfields may have triggered
                // a relocating minor collection); return the live cache slot.
                decoder.virtuals_cache.get_ptr(index)
            }
            VirtualInfo::VStruct {
                typedescr,
                fields,
                fielddescrs,
                ..
            } => {
                // resume.py struct = decoder.allocate_struct(self.typedescr)
                let obj = typedescr
                    .as_ref()
                    .map(|d| decoder.allocate_struct(d))
                    .unwrap_or(0);
                decoder.virtuals_cache.set_ptr(index, obj);
                // resume.py return self.setfields(decoder, struct)
                abstract_virtual_struct_info_setfields(
                    decoder,
                    allocator,
                    index,
                    typedescr.as_ref(),
                    fielddescrs,
                    fields,
                );
                // re-read the forwarded pointer (setfields may have triggered
                // a relocating minor collection); return the live cache slot.
                decoder.virtuals_cache.get_ptr(index)
            }
            VirtualInfo::VArray {
                arraydescr,
                clear,
                items,
                ..
            } => {
                let length = items.len();
                // resume.py: array = decoder.allocate_array(length, arraydescr, self.clear)
                let array = arraydescr
                    .as_ref()
                    .map(|d| decoder.allocate_array(length, d, *clear))
                    .unwrap_or(0);
                decoder.virtuals_cache.set_ptr(index, array);
                // resume.py:656-670: dispatch by arraydescr element type
                let is_pointers = arraydescr
                    .as_ref()
                    .and_then(|d| d.as_array_descr())
                    .is_some_and(|ad| ad.is_array_of_pointers());
                let is_floats = arraydescr
                    .as_ref()
                    .and_then(|d| d.as_array_descr())
                    .is_some_and(|ad| ad.is_array_of_floats());
                if let Some(ad) = arraydescr.as_ref() {
                    for (i, source) in items.iter().enumerate() {
                        // decode_field_source may materialize a nested virtual
                        // and relocate this array; re-read the forwarded
                        // pointer from the rooted cache slot before the write
                        // (same hazard as abstract_virtual_struct_info_setfields).
                        if is_pointers {
                            // resume.py:659: decoder.bh_setarrayitem_gc_r(array, i, num, arraydescr)
                            let value = decoder.decode_field_source(source);
                            let array = decoder.virtuals_cache.get_ptr(index);
                            allocator.bh_setarrayitem_gc_r(array, i, value, ad);
                        } else if is_floats {
                            // resume.py:664: decoder.bh_setarrayitem_gc_f(array, i, num, arraydescr)
                            let value = decoder.decode_field_source_float(source);
                            let array = decoder.virtuals_cache.get_ptr(index);
                            allocator.bh_setarrayitem_gc_f(array, i, value, ad);
                        } else {
                            // resume.py:669: decoder.bh_setarrayitem_gc_i(array, i, num, arraydescr)
                            let value = decoder.decode_field_source_int(source);
                            let array = decoder.virtuals_cache.get_ptr(index);
                            allocator.bh_setarrayitem_gc_i(array, i, value, ad);
                        }
                    }
                }
                decoder.virtuals_cache.get_ptr(index)
            }
            // resume.py: VArrayStructInfo.allocate
            VirtualInfo::VArrayStruct {
                arraydescr,
                fielddescrs,
                element_fields,
                ..
            } => {
                let size = element_fields.len();
                // resume.py: array = decoder.allocate_array(self.size, self.arraydescr, clear=True)
                let array = arraydescr
                    .as_ref()
                    .map(|d| decoder.allocate_array(size, d, /* clear */ true))
                    .unwrap_or(0);
                decoder.virtuals_cache.set_ptr(index, array);
                // resume.py:752-759:
                //   for i in range(self.size):
                //       for j in range(len(self.fielddescrs)):
                //           num = self.fieldnums[p]
                //           if not tagged_eq(num, UNINITIALIZED):
                //               decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
                //           p += 1
                for (i, fields) in element_fields.iter().enumerate() {
                    debug_assert_eq!(
                        fields.len(),
                        fielddescrs.len(),
                        "VArrayStruct element_fields[{i}] has {} fields but {} fielddescrs",
                        fields.len(),
                        fielddescrs.len()
                    );
                    for (j, (_, source)) in fields.iter().enumerate() {
                        if virtual_source_is_uninitialized(source) {
                            continue;
                        }
                        // resume.py: decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
                        decoder.setinteriorfield(i, index, source, &fielddescrs[j], allocator);
                    }
                }
                decoder.virtuals_cache.get_ptr(index)
            }
            // resume.py VStrPlainInfo.allocate
            VirtualInfo::VStrPlain { chars } => {
                vstr_plain_info_allocate(decoder, index, chars, /* is_unicode */ false)
            }
            // resume.py VUniPlainInfo.allocate
            VirtualInfo::VUniPlain { chars } => {
                vstr_plain_info_allocate(decoder, index, chars, /* is_unicode */ true)
            }
            // resume.py VStrConcatInfo.allocate
            VirtualInfo::VStrConcat { left, right } => {
                let string = decoder.concat_strings(left, right);
                decoder.virtuals_cache.set_ptr(index, string);
                string
            }
            // resume.py VStrSliceInfo.allocate
            VirtualInfo::VStrSlice {
                source,
                start,
                length,
            } => {
                let string = decoder.slice_string(source, start, length);
                decoder.virtuals_cache.set_ptr(index, string);
                string
            }
            // resume.py VUniConcatInfo.allocate
            VirtualInfo::VUniConcat { left, right } => {
                let string = decoder.concat_unicodes(left, right);
                decoder.virtuals_cache.set_ptr(index, string);
                string
            }
            // resume.py VUniSliceInfo.allocate
            VirtualInfo::VUniSlice {
                source,
                start,
                length,
            } => {
                let string = decoder.slice_unicode(source, start, length);
                decoder.virtuals_cache.set_ptr(index, string);
                string
            }
            VirtualInfo::Empty => panic!("resume.py: null rd_virtuals[index]"),
            _ => {
                decoder.virtuals_cache.set_ptr(index, 0);
                0
            }
        }
    }

    /// resume.py VRawBufferInfo.allocate_int / VRawSliceInfo.allocate_int
    fn allocate_int(
        &self,
        decoder: &mut ResumeDataDirectReader,
        index: usize,
        allocator: &dyn BlackholeAllocator,
    ) -> i64 {
        match self {
            VirtualInfo::VRawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => {
                assert_eq!(offsets.len(), descrs.len());
                assert_eq!(offsets.len(), values.len());
                // resume.py: buffer = decoder.allocate_raw_buffer(self.func, self.size)
                let buffer = decoder.allocate_raw_buffer(*func, *size);
                // resume.py:704
                decoder.virtuals_cache.set_int(index, buffer);
                // resume.py:705-708: for i in range(len(self.offsets)):
                //     offset = self.offsets[i]; descr = self.descrs[i]
                //     decoder.setrawbuffer_item(buffer, fieldnums[i], offset, descr)
                //
                // Pyre stores fieldnums[i] as a tagged
                // VirtualFieldSource (the value lives on the virtual
                // layout rather than in the resume tape), so we encode
                // it back into the i16 charnum the dispatcher accepts
                // via decode_field_source_{float,int} → tag.
                for i in 0..offsets.len() {
                    let descr = &descrs[i];
                    let source = &values[i];
                    if virtual_source_is_uninitialized(source) {
                        continue;
                    }
                    // pyre extracts the per-entry value from the virtual
                    // layout instead of the tagged fieldnum, then writes
                    // through bh_raw_store_{i,f} per descr kind — same
                    // dispatch as resume.py setrawbuffer_item.
                    assert!(
                        descr.item_type != 0,
                        "raw buffer entry must not be pointer type"
                    );
                    if descr.item_type == 2 {
                        let value = decoder.decode_field_source_float(source);
                        allocator.bh_raw_store_f(buffer, offsets[i], value, descr);
                    } else {
                        let value = decoder.decode_field_source_int(source);
                        allocator.bh_raw_store_i(buffer, offsets[i], value, descr);
                    }
                }
                buffer
            }
            VirtualInfo::VRawSlice { offset, parent } => {
                // resume.py allocate_int — parent is an INT virtual (raw buffer)
                let parent_val = decoder.decode_field_source_int(parent);
                let result = parent_val + *offset;
                decoder.virtuals_cache.set_int(index, result);
                result
            }
            _ => panic!("allocate_int called on non-raw virtual"),
        }
    }
}

impl<'a> ResumeDataDirectReader<'a> {
    /// resume.py __init__
    pub fn new(
        rd_numb: &'a [u8],
        rd_consts: &'a [majit_ir::Const],
        all_liveness: &'a [u8],
        deadframe: FailArgSource<'a>,
        deadframe_types: Option<&'a [majit_ir::Type]>,
        all_virtuals: Option<(Vec<i64>, Vec<i64>)>,
        allocator: &'a dyn BlackholeAllocator,
    ) -> Self {
        // resume.py _init
        let mut resumecodereader = Reader::new(rd_numb);
        let items_resume_section = resumecodereader.next_item();
        let count = resumecodereader.next_item();

        // resume.py:1368-1376
        let (resume_after_guard_not_forced, virtuals_cache) =
            if let Some((ptrs, ints)) = all_virtuals {
                // resume.py:1373-1374: special case for GUARD_NOT_FORCED
                (2, VirtualCache::from_caches(ptrs, ints))
            } else {
                (0, VirtualCache::new())
            };

        ResumeDataDirectReader {
            resumecodereader,
            items_resume_section,
            count,
            consts: rd_consts,
            deadframe,
            deadframe_types,
            resume_after_guard_not_forced,
            rd_virtuals: None,
            virtuals_cache,
            virtualizable_root_base_depth: None,
            allocator,
            all_liveness,
            virtualizable_ptr: 0,
        }
    }

    /// resume.py _prepare — init virtuals and pending fields.
    pub fn prepare(
        &mut self,
        rd_virtuals: Option<&'a [VirtualInfo]>,
        rd_guard_pendingfields: Option<&[majit_ir::GuardPendingFieldEntry]>,
    ) {
        // resume.py:925
        self.prepare_virtuals(rd_virtuals);
        // resume.py:926
        if let Some(guard_pf) = rd_guard_pendingfields {
            self.prepare_guard_pendingfields(guard_pf);
        }
    }

    /// resume.py _prepare_pendingfields — variant for GuardPendingFieldEntry.
    ///
    /// RPython encodes pendingfield target/value as tagged values (TAGBOX/
    /// TAGCONST/TAGVIRTUAL) via _gettagged in _add_pending_fields
    /// (resume.py). At restore time, decode_ref(num) resolves
    /// the tagged value to a concrete pointer.
    ///
    /// `target_tagged` / `value_tagged` are always populated by the
    /// time pendingfields reach this method:
    ///
    ///   * The single production constructor at
    ///     `optimizeopt/optimizer.rs`'s `emit_guard_operation` initializes both to
    ///     `UNASSIGNED`, but the entries are immediately fed through
    ///     `memo.finish()` (`optimizeopt/mod.rs`'s `store_final_boxes_in_guard`) →
    ///     `_add_pending_fields` (this file), which writes
    ///     the tags from `_gettagged`.
    ///   * The sharing path (`_copy_resume_data_from`) routes resume
    ///     reads through `ResumeGuardCopiedDescr.prev` (compile.py
    ///     `get_resumestorage(): return prev`); readers reach the
    ///     donor's already-tagged entries via the descr's `prev`
    ///     pointer rather than touching the shared op directly.
    ///
    /// Hence the only branch is the RPython tagged path; an
    /// `UNASSIGNED` entry escaping into restore-time would be a
    /// soundness bug, so this method panics matching RPython's
    /// implicit invariant (`resume.py` indexes the tagged
    /// number with `decode_ref`).
    fn prepare_guard_pendingfields(&mut self, pendingfields: &[majit_ir::GuardPendingFieldEntry]) {
        for pf in pendingfields {
            // resume.py PENDINGFIELDSTRUCT.lldescr parity:
            // derive (offset, size, type) from the descr (FieldDescr or
            // ArrayDescr) at consume time. RPython always carries
            // lldescr — pyre's producer in `optimizer.rs`'s
            // `emit_guard_operation` mirrors
            // this by setting `pf.descr = pf_op.descr.clone()` for
            // every pending field (pf_op is always a Setfield_gc /
            // Setarrayitem_gc op with a descr).
            let descr = pf.descr.as_ref().expect(
                "resume.py `_prepare_pendingfields`: PENDINGFIELDSTRUCT.lldescr must be set",
            );
            let field_info = if let Some(fd) = descr.as_field_descr() {
                Some((fd.offset(), fd.field_size(), fd.field_type()))
            } else if descr.as_array_descr().is_some() {
                None
            } else {
                panic!(
                    "pending field descr must be FieldDescr or ArrayDescr (descr={:?})",
                    descr,
                );
            };
            // resume.py tagged path. UNASSIGNED tags must
            // never reach this method: `_add_pending_fields` writes the
            // `_gettagged` tags before restore time.
            assert!(
                pf.target_tagged != UNASSIGNED && pf.value_tagged != UNASSIGNED,
                "GuardPendingFieldEntry reached prepare_guard_pendingfields with \
                 UNASSIGNED tag (target_tagged={}, value_tagged={}, descr={:?}); \
                 _add_pending_fields must have run before restore time",
                pf.target_tagged,
                pf.value_tagged,
                descr,
            );
            // resume.py: struct = self.decode_ref(num)
            let struct_ptr = self.decode_ref(pf.target_tagged);

            if pf.item_index < 0 {
                let _ = field_info; // setfield dispatcher reads descr directly.
                // resume.py: self.setfield(struct, fieldnum, descr)
                self.setfield(struct_ptr, pf.value_tagged, descr);
            } else {
                // resume.py: self.setarrayitem(struct, itemindex,
                //                  fieldnum, descr).
                let index = pf.item_index as usize;
                self.setarrayitem(struct_ptr, index, pf.value_tagged, descr);
            }
        }
    }

    /// `resume.py setfield(struct, fieldnum, descr)` dispatcher:
    /// forwards to `bh_setfield_gc_r` / `bh_setfield_gc_f` /
    /// `bh_setfield_gc_i` based on the field's value kind.  `fieldnum` is
    /// the resume.py-tagged value to decode (decode_ref / decode_float /
    /// decode_int per kind).
    ///
    /// Routes on `field_type()` (the value kind the optimizer numbered the
    /// stored box as) rather than `is_pointer_field()`.  In RPython these
    /// coincide — `FLAG_POINTER` is set iff the field is a GC `Ptr`, so
    /// `is_pointer_field()` ⟺ `field_type == Ref`.  Pyre overloads the
    /// flag for the write barrier: `ExecutionContext.sys_exc_value`
    /// (`pyre-jit-trace/src/descr.rs`) carries `field_type = Ref` (so the
    /// optimizer tracks and numbers the value as a GC ref) but a
    /// non-pointer flag (so `rewrite.rs handle_write_barrier_setfield`
    /// emits no barrier into the non-GC EC root).  The resume decode must
    /// match how the value was numbered — by `field_type` — or a Ref value
    /// (here a virtual exception object) is mis-decoded via `decode_int`
    /// → `getvirtual_int` on a non-raw virtual.
    pub fn setfield(&mut self, struct_ptr: i64, fieldnum: i16, descr: &majit_ir::DescrRef) {
        let fd = descr
            .as_field_descr()
            .expect("resume.py:1509 setfield requires FieldDescr");
        let descr_info = majit_ir::FieldDescrInfo {
            index: descr.index(),
            offset: fd.offset(),
            field_type: fd.field_type(),
            field_size: fd.field_size(),
        };
        if fd.field_type() == majit_ir::Type::Ref {
            // resume.py newvalue = self.decode_ref(fieldnum)
            // resume.py self.cpu.bh_setfield_gc_r(struct, newvalue, descr)
            let value = self.decode_ref(fieldnum);
            self.allocator
                .bh_setfield_gc_r(struct_ptr, value, &descr_info);
        } else if fd.is_float_field() {
            // resume.py newvalue = self.decode_float(fieldnum)
            // resume.py self.cpu.bh_setfield_gc_f(struct, newvalue, descr)
            let value = self.decode_float(fieldnum);
            self.allocator
                .bh_setfield_gc_f(struct_ptr, value, &descr_info);
        } else {
            // resume.py newvalue = self.decode_int(fieldnum)
            // resume.py self.cpu.bh_setfield_gc_i(struct, newvalue, descr)
            let value = self.decode_int(fieldnum);
            self.allocator
                .bh_setfield_gc_i(struct_ptr, value, &descr_info);
        }
    }

    /// resume.py allocate_with_vtable(descr) →
    /// `executor.exec_new_with_vtable(self.cpu, descr)` — pyre's
    /// `vtable` argument is the resolved class pointer carried on the
    /// virtual layout (info.py:318 _known_class).
    pub fn allocate_with_vtable(&self, descr: &majit_ir::DescrRef, vtable: usize) -> i64 {
        self.allocator.allocate_with_vtable(descr, vtable)
    }

    /// resume.py allocate_struct(typedescr) → cpu.bh_new(typedescr).
    pub fn allocate_struct(&self, typedescr: &majit_ir::DescrRef) -> i64 {
        self.allocator.bh_new(typedescr)
    }

    /// resume.py allocate_array(length, arraydescr, clear) →
    /// `cpu.bh_new_array_clear` (clear=True) or `cpu.bh_new_array`.
    pub fn allocate_array(
        &self,
        length: usize,
        arraydescr: &majit_ir::DescrRef,
        clear: bool,
    ) -> i64 {
        if clear {
            self.allocator.bh_new_array_clear(length, arraydescr)
        } else {
            self.allocator.bh_new_array(length, arraydescr)
        }
    }

    /// resume.py allocate_raw_buffer(func, size) → calldescr =
    /// `callinfo_for_oopspec(OS_RAW_MALLOC_VARSIZE_CHAR)`,
    /// `cpu.bh_call_i(func, [size], None, None, calldescr)` — pyre's
    /// `BlackholeAllocator::allocate_raw_buffer` keeps the wrapped form
    /// (the calldescr / cic resolution lives inside the allocator impl
    /// because pyre lacks a callinfocollection on the reader side).
    pub fn allocate_raw_buffer(&self, func: i64, size: usize) -> i64 {
        self.allocator.allocate_raw_buffer(func, size)
    }

    /// resume.py allocate_string(length) — forward to allocator.
    pub fn allocate_string(&self, length: usize) -> i64 {
        self.allocator.bh_newstr(length)
    }

    /// resume.py string_setitem(str, index, charnum) — decode
    /// the per-character source and forward to the allocator.  Pyre
    /// threads a `VirtualFieldSource` where resume.py threads a tagged
    /// i16 charnum; the structural shape matches because both decoders
    /// resolve to an integer character value before calling
    /// bh_strsetitem.
    pub fn string_setitem(&mut self, string: i64, index: usize, source: &VirtualFieldSource) {
        let char = self.decode_field_source_int(source);
        self.allocator.bh_strsetitem(string, index, char);
    }

    /// resume.py concat_strings(str1num, str2num) — decode
    /// the two ref sources and dispatch to OS_STR_CONCAT.  The funcptr
    /// is resolved by the allocator via
    /// `callinfocollection.funcptr_for_oopspec(OS_STR_CONCAT)`
    /// (resume.py); the variant carries no funcptr.
    pub fn concat_strings(
        &mut self,
        str1_source: &VirtualFieldSource,
        str2_source: &VirtualFieldSource,
    ) -> i64 {
        let str1 = self.decode_field_source(str1_source);
        let str2 = self.decode_field_source(str2_source);
        self.allocator.os_str_concat(str1, str2)
    }

    /// resume.py slice_string(strnum, startnum, lengthnum) →
    /// OS_STR_SLICE funcptr(str, start, start + length).  Funcptr is
    /// resolved by the allocator via `callinfocollection`.
    pub fn slice_string(
        &mut self,
        str_source: &VirtualFieldSource,
        start_source: &VirtualFieldSource,
        length_source: &VirtualFieldSource,
    ) -> i64 {
        let str = self.decode_field_source(str_source);
        let start = self.decode_field_source_int(start_source);
        let length = self.decode_field_source_int(length_source);
        let stop = start.wrapping_add(length);
        self.allocator.os_str_slice(str, start, stop)
    }

    /// resume.py allocate_unicode(length) → cpu.bh_newunicode.
    pub fn allocate_unicode(&self, length: usize) -> i64 {
        self.allocator.bh_newunicode(length)
    }

    /// resume.py unicode_setitem(str, index, charnum) — same
    /// shape as string_setitem.
    pub fn unicode_setitem(&mut self, string: i64, index: usize, source: &VirtualFieldSource) {
        let char = self.decode_field_source_int(source);
        self.allocator.bh_unicodesetitem(string, index, char);
    }

    /// resume.py concat_unicodes(str1num, str2num).  Funcptr
    /// is resolved by the allocator via `callinfocollection`.
    pub fn concat_unicodes(
        &mut self,
        str1_source: &VirtualFieldSource,
        str2_source: &VirtualFieldSource,
    ) -> i64 {
        let str1 = self.decode_field_source(str1_source);
        let str2 = self.decode_field_source(str2_source);
        self.allocator.os_uni_concat(str1, str2)
    }

    /// resume.py setrawbuffer_item(buffer, fieldnum, offset,
    /// descr) dispatcher: `assert not descr.is_array_of_pointers()`,
    /// then dispatch to `bh_raw_store_f` (float) or `bh_raw_store_i`
    /// (default).
    pub fn setrawbuffer_item(
        &mut self,
        buffer: i64,
        fieldnum: i16,
        offset: i64,
        descr: &majit_ir::ArrayDescrInfo,
    ) {
        // resume.py:1544 assert not descr.is_array_of_pointers()
        assert!(
            descr.item_type != 0,
            "setrawbuffer_item: descr must not be array_of_pointers"
        );
        if descr.item_type == 2 {
            // resume.py newvalue = self.decode_float(fieldnum)
            //                    self.cpu.bh_raw_store_f(...)
            let value = self.decode_float(fieldnum);
            self.allocator.bh_raw_store_f(buffer, offset, value, descr);
        } else {
            // resume.py newvalue = self.decode_int(fieldnum)
            //                    self.cpu.bh_raw_store_i(...)
            let value = self.decode_int(fieldnum);
            self.allocator.bh_raw_store_i(buffer, offset, value, descr);
        }
    }

    /// resume.py slice_unicode(strnum, startnum, lengthnum).
    /// Funcptr resolved by the allocator via `callinfocollection`.
    pub fn slice_unicode(
        &mut self,
        str_source: &VirtualFieldSource,
        start_source: &VirtualFieldSource,
        length_source: &VirtualFieldSource,
    ) -> i64 {
        let str = self.decode_field_source(str_source);
        let start = self.decode_field_source_int(start_source);
        let length = self.decode_field_source_int(length_source);
        let stop = start.wrapping_add(length);
        self.allocator.os_uni_slice(str, start, stop)
    }

    /// `resume.py setarrayitem(array, index, fieldnum, descr)`
    /// dispatcher: forwards to `setarrayitem_ref` /
    /// `setarrayitem_float` / `setarrayitem_int` based on the live
    /// `arraydescr.is_array_of_pointers()` / `is_array_of_floats()`
    /// methods.  `fieldnum` is the resume.py-tagged value to decode.
    pub fn setarrayitem(
        &mut self,
        array: i64,
        index: usize,
        fieldnum: i16,
        arraydescr: &majit_ir::DescrRef,
    ) {
        let ad = arraydescr
            .as_array_descr()
            .expect("resume.py `setarrayitem` requires ArrayDescr");
        if ad.is_array_of_pointers() {
            // resume.py:1011 self.bh_setarrayitem_gc_r(array, index, fieldnum, arraydescr)
            let value = self.decode_ref(fieldnum);
            self.allocator
                .bh_setarrayitem_gc_r(array, index, value, arraydescr);
        } else if ad.is_array_of_floats() {
            // resume.py:1013 self.bh_setarrayitem_gc_f(array, index, fieldnum, arraydescr)
            let value = self.decode_float(fieldnum);
            self.allocator
                .bh_setarrayitem_gc_f(array, index, value, arraydescr);
        } else {
            // resume.py:1015 self.bh_setarrayitem_gc_i(array, index, fieldnum, arraydescr)
            let value = self.decode_int(fieldnum);
            self.allocator
                .bh_setarrayitem_gc_i(array, index, value, arraydescr);
        }
    }

    /// resume.py handling_async_forcing
    pub fn handling_async_forcing(&mut self) {
        self.resume_after_guard_not_forced = 1;
    }

    // AbstractResumeDataReader methods (resume.py)

    /// resume.py `read_jitcode_pos_pc`.
    pub fn read_jitcode_pos_pc(&mut self) -> (i32, i32) {
        let jitcode_pos = self.resumecodereader.next_item();
        let pc = self.resumecodereader.next_item();
        (jitcode_pos, pc)
    }

    /// resume.py next_int
    #[inline]
    pub fn next_int(&mut self) -> i64 {
        let tagged = self.resumecodereader.next_item() as i16;
        // resume.py decode_int `TAGINT`: the payload is the signed value.
        if (tagged as u16) & TAGMASK as u16 == TAGINT as u16 {
            return (tagged >> 2) as i64;
        }
        self.decode_int(tagged)
    }

    /// resume.py next_ref
    pub fn next_ref(&mut self) -> i64 {
        let tagged = self.resumecodereader.next_item() as i16;
        self.decode_ref(tagged)
    }

    fn next_ref_for_resume_slot(&mut self) -> i64 {
        let tagged = self.resumecodereader.next_item() as i16;
        // resume.py decode_ref `TAGBOX`: the payload is a deadframe index.
        if (tagged as u16) & TAGMASK as u16 == TAGBOX as u16 {
            let mut idx = (tagged >> 2) as i32;
            if idx < 0 {
                idx += self.count;
            }
            return self.deadframe.get(idx as usize);
        }
        self.decode_ref(tagged)
    }

    /// resume.py next_float
    pub fn next_float(&mut self) -> i64 {
        let tagged = self.resumecodereader.next_item() as i16;
        self.decode_float(tagged)
    }

    /// resume.py load_next_value_of_type
    pub fn next_value_of_type(&mut self, tp: majit_ir::Type) -> i64 {
        match tp {
            majit_ir::Type::Int => self.next_int(),
            majit_ir::Type::Ref => self.next_ref(),
            majit_ir::Type::Float => self.next_float(),
            other => panic!("load_next_value_of_type: unexpected type {other:?}"),
        }
    }

    /// resume.py done_reading
    pub fn done_reading(&self) -> bool {
        self.resumecodereader.items_read >= self.items_resume_section as usize
    }

    /// resume.py getvirtual_ptr
    ///
    /// Returns the index'th virtual, building it lazily if needed.
    /// Note that this may be called recursively; that's why the
    /// allocate() methods must fill in the cache as soon as they
    /// have the object, before they fill its fields.
    pub fn getvirtual_ptr(&mut self, index: usize) -> i64 {
        // resume.py:950: assert self.virtuals_cache is not None
        assert!(
            !self.virtuals_cache.is_empty(),
            "getvirtual_ptr: virtuals_cache is empty (rd_virtuals not prepared)"
        );
        // resume.py:951-952
        let v = self.virtuals_cache.get_ptr(index);
        if v != 0 {
            return v;
        }
        // resume.py:953-955: lazy allocation
        assert!(self.rd_virtuals.is_some(), "rd_virtuals is None");
        // Safety: rd_virtuals is an immutable slice reference that we need to
        // read while mutating virtuals_cache through self. The slice data is
        // never modified by allocate(), only the cache vectors are written.
        let rd_virtuals_ptr = self.rd_virtuals.unwrap().as_ptr();
        let rd_virtuals_len = self.rd_virtuals.unwrap().len();
        let vinfo = unsafe { &*rd_virtuals_ptr.add(index) };
        debug_assert!(index < rd_virtuals_len);
        let allocator = self.allocator as *const dyn BlackholeAllocator;
        // resume.py `v = self.rd_virtuals[index].allocate(self, index)`.
        // RPython returns `allocate`'s result and asserts `v == cache`,
        // relying on `v` being a GC-traced stack local that a minor
        // collection triggered while `allocate` fills the virtual's fields
        // keeps equal to the (also-rooted) cache slot.  Pyre's `v` is a raw
        // i64 — not a GC root — so such a collection forwards the rooted
        // `virtuals_ptr_cache` slot in place (rooted by
        // `blackhole_from_resumedata`) but leaves `v` pointing into
        // from-space.  Return the live cache slot so callers (e.g. the
        // pending-field `setfield` that publishes the exception into
        // `EC.sys_exc_value`) store the forwarded pointer, not a dangling
        // from-space one.
        vinfo.allocate(self, index, unsafe { &*allocator });
        self.virtuals_cache.get_ptr(index)
    }

    /// resume.py getvirtual_int
    pub fn getvirtual_int(&mut self, index: usize) -> i64 {
        // resume.py:959: assert self.virtuals_cache is not None
        assert!(
            !self.virtuals_cache.is_empty(),
            "getvirtual_int: virtuals_cache is empty (rd_virtuals not prepared)"
        );
        // resume.py:960-961
        let v = self.virtuals_cache.get_int(index);
        if v != 0 {
            return v;
        }
        // resume.py:962-966
        assert!(self.rd_virtuals.is_some(), "rd_virtuals is None");
        let rd_virtuals_ptr = self.rd_virtuals.unwrap().as_ptr();
        let vinfo = unsafe { &*rd_virtuals_ptr.add(index) };
        assert!(vinfo.is_about_raw(), "getvirtual_int: not a raw virtual");
        let allocator = self.allocator as *const dyn BlackholeAllocator;
        let v = vinfo.allocate_int(self, index, unsafe { &*allocator });
        debug_assert_eq!(
            v,
            self.virtuals_cache.get_int(index),
            "resume.py: bad cache"
        );
        v
    }

    /// resume.py force_all_virtuals
    pub fn force_all_virtuals(&mut self) -> (&[i64], &[i64]) {
        // Materializing one virtual allocates, so it can trigger a collection
        // that relocates the virtuals materialized before it. `getvirtual_ptr`
        // answers with the cache slot rather than `allocate`'s raw result on
        // the premise that the slot is a registered root the collection
        // forwards in place; check the premise here, once per force.
        debug_assert!(
            self.virtuals_cache.virtuals_ptr_cache.is_empty()
                || majit_gc::shadow_stack::resume_ref_slice_registered(
                    self.virtuals_cache.virtuals_ptr_cache.as_ptr()
                ),
            "force_all_virtuals: virtuals_ptr_cache is not a registered resume root"
        );
        if let Some(rd_virtuals) = self.rd_virtuals {
            for (i, rd_virtual) in rd_virtuals.iter().enumerate() {
                // resume.py `if rd_virtual is not None`: skip empty
                // slots (Pyre carries them as the `Empty`-derived
                // placeholder shape).
                if rd_virtual.is_empty_placeholder() {
                    continue;
                }
                if rd_virtual.is_about_raw() {
                    // resume.py:977: kind == INT
                    self.getvirtual_int(i);
                } else {
                    // resume.py:976: kind == REF
                    self.getvirtual_ptr(i);
                }
            }
        }
        (
            &self.virtuals_cache.virtuals_ptr_cache,
            &self.virtuals_cache.virtuals_int_cache,
        )
    }

    /// resume.py _prepare_virtuals
    fn prepare_virtuals(&mut self, virtuals: Option<&'a [VirtualInfo]>) {
        if let Some(v) = virtuals {
            if v.is_empty() {
                return;
            }
            self.rd_virtuals = Some(v);
            // resume.py:990-991
            self.virtuals_cache = VirtualCache::from_caches(vec![0; v.len()], vec![0; v.len()]);
        }
    }

    // ResumeDataDirectReader methods (resume.py)

    /// A `-live-` entry names one bank. `TAGINT` (or a const of another
    /// type) in the ref or float section is the same bug as an int box in
    /// `registers_r`: `getref_base` is not defined on that box.
    fn expect_liveness_item_bank(
        &self,
        jitcode_name: &str,
        pc: usize,
        bank: majit_ir::Type,
        reg_idx: u32,
        num_regs: usize,
    ) {
        crate::blackhole::expect_liveness_bank(jitcode_name, pc, bank, reg_idx, num_regs);
        let tagged = self.resumecodereader.peek() as i16;
        let (num, tag) = untag(tagged);
        let got = match tag {
            TAGINT => Some(majit_ir::Type::Int),
            TAGCONST => {
                if bank == majit_ir::Type::Ref && tagged_eq(tagged, NULLREF) {
                    Some(majit_ir::Type::Ref)
                } else {
                    let idx = (num - TAG_CONST_OFFSET) as usize;
                    self.consts.get(idx).map(|c| c.get_type())
                }
            }
            _ => None,
        };
        if let Some(got) = got
            && got != bank
        {
            panic!(
                "liveness: jitcode {jitcode_name} pc {pc} bank {bank:?} register {reg_idx} holds {got:?}"
            );
        }
    }

    /// resume.py `consume_one_section(self, blackholeinterp)`.
    ///
    /// ```python
    /// def consume_one_section(self, blackholeinterp):
    ///     self.blackholeinterp = blackholeinterp
    ///     info = blackholeinterp.get_current_position_info()
    ///     self._prepare_next_section(info)
    /// ```
    pub fn consume_one_section(
        &mut self,
        bh: &mut BlackholeInterpreter,
        vinfo: Option<&dyn VirtualizableInfo>,
    ) {
        // resume.py:1383
        let info = bh.get_current_position_info();
        // resume.py:1384
        self._prepare_next_section(info, bh, vinfo);
    }

    /// resume.py `_prepare_next_section(self, info)`.
    ///
    /// ```python
    /// def _prepare_next_section(self, info):
    ///     from rpython.jit.codewriter.jitcode import enumerate_vars
    ///     enumerate_vars(info,
    ///             self.metainterp_sd.liveness_info,
    ///             self._callback_i,
    ///             self._callback_r,
    ///             self._callback_f,
    ///             self.unique_id)
    /// ```
    ///
    /// `self.all_liveness` shadows `self.metainterp_sd.liveness_info` —
    /// the shared packed buffer that `enumerate_vars` indexes with
    /// `info`. The three callbacks still call `next_int`/`next_ref`/
    /// `next_float` on this reader (resume.py), matching
    /// `_callback_i/_callback_r/_callback_f` plus `write_an_int/write_a_ref/
    /// write_a_float` (resume.py).
    fn _prepare_next_section(
        &mut self,
        info: usize,
        bh: &mut BlackholeInterpreter,
        vinfo: Option<&dyn VirtualizableInfo>,
    ) {
        use majit_jitcode::liveness::LivenessIterator;

        let all_liveness: &[u8] = self.all_liveness;

        // jitcode.py:149-151 — three length bytes.
        let length_i = all_liveness[info] as u32;
        let length_r = all_liveness[info + 1] as u32;
        let length_f = all_liveness[info + 2] as u32;
        // jitcode.py:152
        let mut offset = info + 3;

        // resume.py `write_an_int` / `write_a_ref` / `write_a_float`
        // assign the register file directly. `setarg_*` is the same
        // store plus a `jit_strict_mode` bound check; the regex leaf
        // pays that call once per live slot per frame per character.
        let vinfo_heap = vinfo.filter(|v| v.has_vable_token());
        // resume.py `_callback_i` / jitcode.py:153-157.
        if length_i != 0 {
            let mut it = LivenessIterator::new(offset, length_i, all_liveness);
            for reg_idx in it.by_ref() {
                self.expect_liveness_item_bank(
                    &bh.jitcode.name,
                    bh.position,
                    majit_ir::Type::Int,
                    reg_idx,
                    bh.jitcode.num_regs_i(),
                );
                bh.registers_i[reg_idx as usize] = self.next_int();
            }
            offset = it.offset;
        }
        // resume.py `_callback_r` / jitcode.py:158-162.
        if length_r != 0 {
            let mut it = LivenessIterator::new(offset, length_r, all_liveness);
            for reg_idx in it.by_ref() {
                self.expect_liveness_item_bank(
                    &bh.jitcode.name,
                    bh.position,
                    majit_ir::Type::Ref,
                    reg_idx,
                    bh.jitcode.num_regs_r(),
                );
                let value = self.next_ref_for_resume_slot();
                bh.registers_r[reg_idx as usize] = value;
                if let Some(vinfo) = vinfo_heap {
                    vinfo.push_resume_ref_roots_for_value(value);
                }
            }
            offset = it.offset;
        }
        // resume.py `_callback_f` / jitcode.py:163-166.
        if length_f != 0 {
            let mut it = LivenessIterator::new(offset, length_f, all_liveness);
            for reg_idx in it {
                self.expect_liveness_item_bank(
                    &bh.jitcode.name,
                    bh.position,
                    majit_ir::Type::Float,
                    reg_idx,
                    bh.jitcode.num_regs_f(),
                );
                bh.registers_f[reg_idx as usize] = self.next_float();
            }
            // `offset` is the end of the float section; no further use.
            let _ = offset;
        }
    }

    /// Callback-driven sibling of `_prepare_next_section` — drives the
    /// same `enumerate_vars(info, all_liveness, _callback_i/r/f)`
    /// walk (`resume.py`) but lets the caller decide what to
    /// do with each `(kind, reg_idx, value)` triple.  Three Rust
    /// FnMut closures cannot share `&mut bh` simultaneously (E0524),
    /// so the kind dispatch happens INSIDE the single closure rather
    /// than across three separate ones.  The on-demand cranelift
    /// deopt callback uses a closure that appends each
    /// value to a flat `Vec<i64>` mirroring the recovery_layout
    /// walker's `rebuilt` output.
    pub fn _prepare_next_section_with(
        &mut self,
        info: usize,
        jitcode_name: &str,
        pc: usize,
        num_regs: (usize, usize, usize),
        mut cb: impl FnMut(majit_ir::Type, u32, i64),
    ) {
        use majit_jitcode::liveness::LivenessIterator;

        // `self.all_liveness` is `&'a [u8]` — copying the reference does
        // not borrow `self`, so the inner `self.next_*` calls below are
        // free to take `&mut self`.
        let all_liveness: &[u8] = self.all_liveness;

        // jitcode.py:149-151 — three length bytes.
        let length_i = all_liveness[info] as u32;
        let length_r = all_liveness[info + 1] as u32;
        let length_f = all_liveness[info + 2] as u32;
        // jitcode.py:152
        let mut offset = info + 3;

        // resume.py `_callback_i` / jitcode.py:153-157.
        if length_i != 0 {
            let mut it = LivenessIterator::new(offset, length_i, all_liveness);
            for reg_idx in it.by_ref() {
                self.expect_liveness_item_bank(
                    jitcode_name,
                    pc,
                    majit_ir::Type::Int,
                    reg_idx,
                    num_regs.0,
                );
                let value = self.next_int();
                cb(majit_ir::Type::Int, reg_idx, value);
            }
            offset = it.offset;
        }
        // resume.py `_callback_r` / jitcode.py:158-162.
        if length_r != 0 {
            let mut it = LivenessIterator::new(offset, length_r, all_liveness);
            for reg_idx in it.by_ref() {
                self.expect_liveness_item_bank(
                    jitcode_name,
                    pc,
                    majit_ir::Type::Ref,
                    reg_idx,
                    num_regs.1,
                );
                let value = self.next_ref_for_resume_slot();
                cb(majit_ir::Type::Ref, reg_idx, value);
            }
            offset = it.offset;
        }
        // resume.py `_callback_f` / jitcode.py:163-166.
        if length_f != 0 {
            let mut it = LivenessIterator::new(offset, length_f, all_liveness);
            for reg_idx in it {
                self.expect_liveness_item_bank(
                    jitcode_name,
                    pc,
                    majit_ir::Type::Float,
                    reg_idx,
                    num_regs.2,
                );
                let value = self.next_float();
                cb(majit_ir::Type::Float, reg_idx, value);
            }
            // `offset` is the end of the float section; no further use.
            let _ = offset;
        }
    }

    /// On-demand variant for cranelift's deopt path: walk the resume
    /// tape section-by-section, append each decoded `(int|ref|float)`
    /// value into a flat `Vec<i64>` (innermost-first concatenation,
    /// matching the existing recovery_layout walker's `rebuilt`
    /// output).  `resolve_jitcode` mirrors `resume.py:1339
    /// jitcode = jitcodes[jitcode_pos]` and returns the per-PC
    /// `op_live` byte that `BlackholeInterpreter::get_current_position_info`
    /// uses to index `all_liveness`.
    ///
    /// Caller is expected to drive `prepare(rd_virtuals,
    /// rd_guard_pendingfields)` + `consume_vref_and_vable` first per
    /// `resume.py blackhole_from_resumedata`.
    pub fn consume_all_sections_into_vec(
        &mut self,
        resolve_jitcode: &dyn Fn(
            i32,
            i32,
        )
            -> Option<(std::sync::Arc<crate::jitcode::JitCode>, usize, u8)>,
        outputs: &mut Vec<i64>,
    ) -> bool {
        while !self.done_reading() {
            // resume.py read_jitcode_pos_pc.
            let (jitcode_pos, pc) = self.read_jitcode_pos_pc();
            let Some((jitcode, resolved_pc, op_live)) = resolve_jitcode(jitcode_pos, pc) else {
                return false;
            };
            // `blackhole.rs`'s `get_current_position_info` parity —
            // `jitcode.get_live_vars_info(position, op_live)` is the
            // section info offset for the current PC.
            let info = jitcode.get_live_vars_info(resolved_pc, op_live);
            self._prepare_next_section_with(
                info,
                &jitcode.name,
                resolved_pc,
                (
                    jitcode.num_regs_i(),
                    jitcode.num_regs_r(),
                    jitcode.num_regs_f(),
                ),
                |_kind, _reg_idx, value| {
                    outputs.push(value);
                },
            );
        }
        true
    }

    /// resume.py consume_virtualref_info
    pub fn consume_virtualref_info(&mut self, vrefinfo: Option<&dyn VRefInfo>) {
        // resume.py:1389
        let size = self.resumecodereader.next_item();
        // resume.py:1390-1391
        if vrefinfo.is_none() || size == 0 {
            // resume.py:1391: assert size == 0
            assert!(
                size == 0,
                "consume_virtualref_info: vrefinfo is None but size={size} != 0"
            );
            return;
        }
        let vrefinfo = vrefinfo.unwrap();
        // resume.py:1393-1397
        for _i in 0..size {
            let virtual_val = self.next_ref();
            let vref = self.next_ref();
            // resume.py:1397
            vrefinfo.continue_tracing(vref, virtual_val);
        }
    }

    /// resume.py `consume_vable_info`
    pub fn consume_vable_info(&mut self, vinfo: &dyn VirtualizableInfo, vable_size: i32) {
        // resume.py consume_vable_info: virtualizable = self.next_ref()
        assert!(vable_size > 0);
        let virtualizable = self.next_ref();
        self.virtualizable_ptr = if virtualizable == 0 {
            0
        } else {
            majit_gc::gc_current_object_address(virtualizable as usize) as i64
        };
        if self.virtualizable_root_base_depth.is_none() {
            self.virtualizable_root_base_depth =
                Some(majit_gc::shadow_stack::resume_ref_roots_depth());
            // RPython's ResumeDataDirectReader holds the virtualizable as a
            // GC-traced reference while consume_vable_info restores its
            // fields.  Root that owner before exposing any of its fields or
            // running write_from_resume_data_partial: both operations may
            // allocate, and a nursery PyFrame can move during either one.
            // Registering a field inside the frame is not a substitute — its
            // address moves with the owner.
            unsafe {
                majit_gc::shadow_stack::push_resume_ref_roots(std::slice::from_mut(
                    &mut self.virtualizable_ptr,
                ));
            }
        }
        vinfo.push_resume_ref_roots(self.virtualizable_ptr);
        // resume.py: assert vinfo.get_total_size(virtualizable) == vable_size - 1
        let expected = vinfo.get_total_size(self.virtualizable_ptr) as i32;
        assert!(
            expected == vable_size - 1,
            "consume_vable_info: vinfo.get_total_size(0x{:x}) = {} != vable_size - 1 = {}",
            self.virtualizable_ptr,
            expected,
            vable_size - 1
        );
        vinfo.reset_token_gcref(self.virtualizable_ptr);
        vinfo.write_from_resume_data_partial(self.virtualizable_ptr, self);
    }

    /// resume.py consume_vref_and_vable
    pub fn consume_vref_and_vable(
        &mut self,
        vrefinfo: Option<&dyn VRefInfo>,
        vinfo: Option<&dyn VirtualizableInfo>,
        ginfo: Option<&dyn GreenfieldInfo>,
    ) {
        // resume.py:1425
        let vable_size = self.resumecodereader.next_item();

        if self.resume_after_guard_not_forced != 2 {
            // resume.py:1427-1428
            if let Some(vi) = vinfo {
                self.consume_vable_info(vi, vable_size);
            }
            // resume.py:1429-1430
            if ginfo.is_some() {
                let _ginfo_item = self.resumecodereader.next_item();
            }
            // resume.py:1431
            self.consume_virtualref_info(vrefinfo);
        } else {
            // resume.py:1433-1435
            self.resumecodereader.jump(vable_size as usize);
            let vref_size = self.resumecodereader.next_item();
            self.resumecodereader.jump(vref_size as usize * 2);
        }
    }

    /// resume.py: TAGVIRTUAL num → rd_virtuals/virtuals_cache index.
    ///
    /// RPython indexes `self.rd_virtuals[num]` / `virtuals_cache.get_*(num)`
    /// directly with a possibly-negative `num` and relies on Python list
    /// negative indexing (cached/nested virtuals get negative nums from
    /// `assign_number_to_virtual`). Rust's getvirtual_* take a `usize`, so
    /// remap here, mirroring `_number_virtuals` (`rd_virtuals.len() + num`).
    fn virtual_index(&self, num: i32) -> usize {
        if num >= 0 {
            num as usize
        } else {
            // On the GUARD_NOT_FORCED path (resume.py:1373-1374) the
            // virtuals arrive preloaded in `virtuals_cache` while
            // `rd_virtuals` stays None; the cache has one slot per
            // virtual, so its length is the same wrap base.
            let len =
                self.rd_virtuals
                    .map_or_else(|| self.virtuals_cache.len(), |v| v.len()) as i32;
            (len + num) as usize
        }
    }

    /// resume.py decode_int
    pub fn decode_int(&mut self, tagged: i16) -> i64 {
        let (num, tag) = untag(tagged);
        match tag {
            TAGCONST => {
                // resume.py:1555 — ConstInt.getint(): return the i64 value.
                let idx = (num - TAG_CONST_OFFSET) as usize;
                self.consts[idx].getint()
            }
            TAGINT => {
                // resume.py:1557
                num as i64
            }
            TAGVIRTUAL => {
                // resume.py:1559
                let idx = self.virtual_index(num);
                self.getvirtual_int(idx)
            }
            TAGBOX => {
                // resume.py:1561-1564
                let mut idx = num;
                if idx < 0 {
                    idx += self.count;
                }
                self.deadframe.get(idx as usize)
            }
            _ => unreachable!("bad tag: {tag}"),
        }
    }

    /// resume.py decode_ref
    pub fn decode_ref(&mut self, tagged: i16) -> i64 {
        let (num, tag) = untag(tagged);
        match tag {
            TAGCONST => {
                // resume.py:1569-1571
                if tagged_eq(tagged, NULLREF) {
                    return 0; // ConstPtr.value (null pointer)
                }
                // resume.py:1571
                let idx = (num - TAG_CONST_OFFSET) as usize;
                // history.py ConstPtr.getref_base() returns the GCREF value.
                self.consts[idx].getref_base().as_usize() as i64
            }
            TAGVIRTUAL => {
                // resume.py:1573
                let idx = self.virtual_index(num);
                self.getvirtual_ptr(idx)
            }
            TAGBOX => {
                // resume.py:1575-1578
                let mut idx = num;
                if idx < 0 {
                    idx += self.count;
                }
                let value = self.deadframe.get(idx as usize);
                // `resume.py decode_ref`: `cpu.get_ref_value(deadframe, num)`.
                // The slot is a ref. An Int/Float here means `_number_boxes`
                // put a non-ref into a ref position.
                if let Some(tys) = self.deadframe_types {
                    let slot_type = *tys.get(idx as usize).unwrap_or_else(|| {
                        panic!(
                            "decode_ref: deadframe_types has {} entries, slot {idx} requested",
                            tys.len()
                        )
                    });
                    debug_assert!(
                        slot_type == majit_ir::Type::Ref,
                        "decode_ref TAGBOX saw {slot_type:?}; fix the encoder in `_number_boxes`"
                    );
                }
                value
            }
            _ => {
                // resume.py `assert tag == TAGBOX`: in a ref slot
                // only TAGCONST / TAGVIRTUAL / TAGBOX are valid.  TAGINT
                // here means the numbering stage produced an int-tagged
                // entry in a ref position — a producer bug.
                panic!("decode_ref: unexpected tag {tag}")
            }
        }
    }

    /// resume.py decode_float
    pub fn decode_float(&mut self, tagged: i16) -> i64 {
        let (num, tag) = untag(tagged);
        match tag {
            TAGCONST => {
                // resume.py:1583 — ConstFloat.getfloatstorage(): i64 bits.
                let idx = (num - TAG_CONST_OFFSET) as usize;
                self.consts[idx].getfloatstorage()
            }
            TAGBOX => {
                // resume.py:1585-1588
                let mut idx = num;
                if idx < 0 {
                    idx += self.count;
                }
                self.deadframe.get(idx as usize)
            }
            _ => {
                // resume.py — only TAGCONST and TAGBOX valid for floats
                panic!("decode_float: unexpected tag {tag}")
            }
        }
    }

    /// Decode a VirtualFieldSource as a REF value (resume.py decode_ref).
    ///
    /// Virtual sources go through getvirtual_ptr (REF virtuals).
    pub fn decode_field_source(&mut self, source: &VirtualFieldSource) -> i64 {
        match source {
            ResumeValueSource::FailArg(index) => self.deadframe.get(*index),
            // resume.py:1568 ConstPtr.getref_base() — the Const carries its type.
            ResumeValueSource::Constant(c) => c.getref_base().as_usize() as i64,
            ResumeValueSource::Virtual(index) => self.getvirtual_ptr(*index),
            ResumeValueSource::Tagged(tagged) => self.decode_ref(*tagged),
            ResumeValueSource::Uninitialized => 0,
            ResumeValueSource::Unavailable => 0,
        }
    }

    /// Decode a VirtualFieldSource as an INT value (resume.py decode_int).
    ///
    /// Virtual sources go through getvirtual_int (INT/raw virtuals).
    pub fn decode_field_source_int(&mut self, source: &VirtualFieldSource) -> i64 {
        match source {
            ResumeValueSource::FailArg(index) => self.deadframe.get(*index),
            // resume.py:1555 ConstInt.getint().
            ResumeValueSource::Constant(c) => c.getint(),
            ResumeValueSource::Virtual(index) => self.getvirtual_int(*index),
            ResumeValueSource::Tagged(tagged) => self.decode_int(*tagged),
            ResumeValueSource::Uninitialized => 0,
            ResumeValueSource::Unavailable => 0,
        }
    }

    /// Decode a VirtualFieldSource as a FLOAT value (resume.py decode_float).
    ///
    /// Floats are stored as raw i64 bits. TAGVIRTUAL is invalid for
    /// float fields — virtual floats would route through a different
    /// VirtualInfo variant.
    pub fn decode_field_source_float(&mut self, source: &VirtualFieldSource) -> i64 {
        match source {
            ResumeValueSource::FailArg(index) => self.deadframe.get(*index),
            // resume.py:1583 ConstFloat.getfloatstorage().
            ResumeValueSource::Constant(c) => c.getfloatstorage(),
            ResumeValueSource::Virtual(_) => {
                panic!("decode_field_source_float: TAGVIRTUAL not valid for float field")
            }
            ResumeValueSource::Tagged(tagged) => self.decode_float(*tagged),
            ResumeValueSource::Uninitialized => 0,
            ResumeValueSource::Unavailable => 0,
        }
    }

    /// resume.py setinteriorfield(index, array, fieldnum, descr)
    ///
    /// Dispatches by descr.is_pointer_field() / is_float_field() / else.
    pub fn setinteriorfield(
        &mut self,
        index: usize,
        virtual_index: usize,
        source: &VirtualFieldSource,
        descr: &majit_ir::DescrRef,
        allocator: &dyn BlackholeAllocator,
    ) {
        let is_pointer = descr
            .as_interior_field_descr()
            .is_some_and(|ifd| ifd.field_descr().is_pointer_field());
        let is_float = descr
            .as_interior_field_descr()
            .is_some_and(|ifd| ifd.field_descr().is_float_field());
        // decode_field_source* may materialize a nested virtual and relocate
        // the array; re-read the forwarded pointer from the rooted cache slot
        // before the write (same hazard as abstract_virtual_struct_info_setfields).
        if is_pointer {
            let value = self.decode_field_source(source);
            let array = self.virtuals_cache.get_ptr(virtual_index);
            allocator.bh_setinteriorfield_gc_r(array, index, value, descr);
        } else if is_float {
            let value = self.decode_field_source_float(source);
            let array = self.virtuals_cache.get_ptr(virtual_index);
            allocator.bh_setinteriorfield_gc_f(array, index, value, descr);
        } else {
            let value = self.decode_field_source_int(source);
            let array = self.virtuals_cache.get_ptr(virtual_index);
            allocator.bh_setinteriorfield_gc_i(array, index, value, descr);
        }
    }

    /// resume.py int_add_const
    pub fn int_add_const(&self, base: i64, offset: i64) -> i64 {
        base + offset
    }
}

impl Drop for ResumeDataDirectReader<'_> {
    fn drop(&mut self) {
        if let Some(depth) = self.virtualizable_root_base_depth {
            majit_gc::shadow_stack::pop_resume_ref_roots_to(depth);
        }
    }
}

#[cfg(test)]
mod setfields_tests {
    use super::*;
    use majit_backend::FailArgSource;
    use majit_ir::descr::SimpleSizeDescr;
    use majit_ir::{
        ArrayFlag, Const, Descr, FieldDescr, FieldDescrInfo, GcRef, SimpleFieldDescr, Type,
    };
    use std::cell::RefCell;
    use std::sync::Arc;

    #[derive(Default)]
    struct RecAllocator {
        refs: RefCell<Vec<i64>>,
        ints: RefCell<Vec<i64>>,
    }

    impl BlackholeAllocator for RecAllocator {
        fn allocate_with_vtable(&self, _descr: &majit_ir::DescrRef, _vtable: usize) -> i64 {
            0xBEEF
        }
        fn bh_new(&self, _typedescr: &majit_ir::DescrRef) -> i64 {
            0xBEEF
        }
        fn bh_setfield_gc_r(&self, _struct_ptr: i64, value: i64, _descr_info: &FieldDescrInfo) {
            self.refs.borrow_mut().push(value);
        }
        fn bh_setfield_gc_i(&self, _struct_ptr: i64, value: i64, _descr_info: &FieldDescrInfo) {
            self.ints.borrow_mut().push(value);
        }
    }

    fn pointer_as_int_parent() -> (
        majit_ir::DescrRef,
        Arc<SimpleFieldDescr>,
        Arc<SimpleFieldDescr>,
    ) {
        let ptr_fd = Arc::new(
            SimpleFieldDescr::new(0x1000_0abc, 16, 8, Type::Int, false)
                .with_flag(ArrayFlag::Pointer)
                .with_index_in_parent(0),
        );
        let int_fd = Arc::new(
            SimpleFieldDescr::new(0x1000_0def, 24, 8, Type::Int, false).with_index_in_parent(1),
        );
        let parent: majit_ir::DescrRef = Arc::new(
            SimpleSizeDescr::new(0x3000_0001, 32, 1).with_all_fielddescrs(vec![
                ptr_fd.clone() as Arc<dyn FieldDescr>,
                int_fd.clone() as Arc<dyn FieldDescr>,
            ]),
        );
        ptr_fd.set_parent_descr(&parent);
        int_fd.set_parent_descr(&parent);
        (parent, ptr_fd, int_fd)
    }

    fn info_for(fd: &SimpleFieldDescr, field_type: Type) -> FieldDescrInfo {
        FieldDescrInfo {
            index: fd.index(),
            offset: fd.offset(),
            field_type,
            field_size: fd.field_size(),
        }
    }

    fn allocate_virtual_obj(
        parent: majit_ir::DescrRef,
        fielddescrs: Vec<FieldDescrInfo>,
        fields: Vec<(u32, VirtualFieldSource)>,
        consts: &[Const],
        alloc: &RecAllocator,
    ) {
        let vinfo = VirtualInfo::VirtualObj {
            descr: Some(parent),
            type_id: 1,
            known_class: None,
            fields,
            fielddescrs,
            descr_size: 32,
        };
        let mut reader = ResumeDataDirectReader::new(
            &[0, 0],
            consts,
            &[],
            FailArgSource::Slice(&[]),
            None,
            None,
            alloc,
        );
        reader.virtuals_cache = VirtualCache::from_caches(vec![0], vec![0]);
        vinfo.allocate(&mut reader, 0, alloc);
    }

    #[test]
    fn setfields_decodes_pointer_field_by_is_pointer_field_not_field_type() {
        // AbstractResumeDataReader.setfield uses is_pointer_field, not
        // field_type(). A FLAG_POINTER field whose snapshot FieldDescrInfo
        // still says Int must decode_ref.
        let (parent, ptr_fd, _int_fd) = pointer_as_int_parent();
        assert!(ptr_fd.is_pointer_field());
        assert_eq!(ptr_fd.field_type(), Type::Int);
        let tagged = tag(TAG_CONST_OFFSET, TAGCONST).unwrap();
        let consts = [Const::Ref(GcRef(0x1111))];
        let alloc = RecAllocator::default();
        allocate_virtual_obj(
            parent,
            vec![info_for(&ptr_fd, Type::Int)],
            vec![(ptr_fd.index(), ResumeValueSource::Tagged(tagged))],
            &consts,
            &alloc,
        );
        assert_eq!(*alloc.refs.borrow(), vec![0x1111]);
        assert!(alloc.ints.borrow().is_empty());
    }

    #[test]
    fn setfields_pairs_fieldnums_by_parent_list_slot_not_enumerate() {
        // Parent list [ref, int]; visitor/snapshot fielddescrs [int, ref].
        // get_virtual_fields already lined fieldnums up with that visitor
        // list (parent_list_slot_in). AbstractVirtualStructInfo.setfields
        // is fielddescrs[i] / fieldnums[i] — rematching through the live
        // parent list would feed the Ref TAGCONST to the Int descr.
        let (parent, ptr_fd, int_fd) = pointer_as_int_parent();
        let tagged_ref = tag(TAG_CONST_OFFSET, TAGCONST).unwrap();
        let tagged_int = tag(7, TAGINT).unwrap();
        let consts = [Const::Ref(GcRef(0x2222))];
        let alloc = RecAllocator::default();
        allocate_virtual_obj(
            parent,
            vec![info_for(&int_fd, Type::Int), info_for(&ptr_fd, Type::Int)],
            vec![
                (0, ResumeValueSource::Tagged(tagged_int)),
                (1, ResumeValueSource::Tagged(tagged_ref)),
            ],
            &consts,
            &alloc,
        );
        assert_eq!(*alloc.refs.borrow(), vec![0x2222]);
        assert_eq!(*alloc.ints.borrow(), vec![7]);
    }

    #[test]
    fn setfields_decodes_ref_field_type_without_pointer_flag() {
        // field_type=Ref, is_pointer_field=false. Snapshot stores else
        // field_type(); setfields must still decode_ref.
        let fd = Arc::new(
            SimpleFieldDescr::new(0x1000_0eee, 8, 8, Type::Ref, false)
                .with_flag(ArrayFlag::Signed)
                .with_index_in_parent(0),
        );
        assert!(!fd.is_pointer_field());
        assert_eq!(fd.field_type(), Type::Ref);
        let parent: majit_ir::DescrRef = Arc::new(
            SimpleSizeDescr::new(0x3000_0002, 16, 2)
                .with_all_fielddescrs(vec![fd.clone() as Arc<dyn FieldDescr>]),
        );
        fd.set_parent_descr(&parent);
        let tagged = tag(TAG_CONST_OFFSET, TAGCONST).unwrap();
        let consts = [Const::Ref(GcRef(0x3333))];
        let alloc = RecAllocator::default();
        allocate_virtual_obj(
            parent,
            vec![info_for(&fd, Type::Int)],
            vec![(fd.index(), ResumeValueSource::Tagged(tagged))],
            &consts,
            &alloc,
        );
        assert_eq!(*alloc.refs.borrow(), vec![0x3333]);
        assert!(alloc.ints.borrow().is_empty());
    }

    #[test]
    fn virtual_info_from_rd_setfields_uses_live_pointer_flag() {
        let (parent, ptr_fd, _int_fd) = pointer_as_int_parent();
        let tagged = tag(TAG_CONST_OFFSET, TAGCONST).unwrap();
        let rd = majit_ir::RdVirtualInfo::VirtualInfo {
            descr: Some(parent),
            type_id: 1,
            known_class: None,
            fielddescrs: vec![info_for(&ptr_fd, Type::Int)],
            fieldnums: vec![tagged],
            descr_size: 32,
        };
        let vinfo = virtual_info_from_rd(&rd);
        let consts = [Const::Ref(GcRef(0x4444))];
        let alloc = RecAllocator::default();
        let mut reader = ResumeDataDirectReader::new(
            &[0, 0],
            &consts,
            &[],
            FailArgSource::Slice(&[]),
            None,
            None,
            &alloc,
        );
        reader.virtuals_cache = VirtualCache::from_caches(vec![0], vec![0]);
        vinfo.allocate(&mut reader, 0, &alloc);
        assert_eq!(*alloc.refs.borrow(), vec![0x4444]);
        assert!(alloc.ints.borrow().is_empty());
    }

    #[test]
    fn setfields_same_offset_uses_snapshot_kind_not_first_sibling() {
        // Flattened leaves share offset 16. Snapshot Int must pair with
        // slot 1 (AbstractResumeDataReader.setfield kind), not the first
        // same-offset Ref sibling at slot 0.
        let ptr_fd = Arc::new(
            SimpleFieldDescr::new(0x1000_3002, 16, 8, Type::Ref, false).with_index_in_parent(0),
        );
        let int_fd = Arc::new(
            SimpleFieldDescr::new(0x1000_3001, 16, 8, Type::Int, false).with_index_in_parent(1),
        );
        let parent: majit_ir::DescrRef = Arc::new(
            SimpleSizeDescr::new(0x3000_0003, 32, 1).with_all_fielddescrs(vec![
                ptr_fd.clone() as Arc<dyn FieldDescr>,
                int_fd.clone() as Arc<dyn FieldDescr>,
            ]),
        );
        ptr_fd.set_parent_descr(&parent);
        int_fd.set_parent_descr(&parent);
        let int_info = FieldDescrInfo {
            index: 0xDEAD_0001,
            offset: 16,
            field_type: Type::Int,
            field_size: 8,
        };
        let ref_info = FieldDescrInfo {
            index: 0xDEAD_0002,
            offset: 16,
            field_type: Type::Ref,
            field_size: 8,
        };
        let tagged_ref = tag(TAG_CONST_OFFSET, TAGCONST).unwrap();
        let tagged_int = tag(9, TAGINT).unwrap();
        let consts = [Const::Ref(GcRef(0x5555))];
        let alloc = RecAllocator::default();
        allocate_virtual_obj(
            parent,
            vec![int_info, ref_info],
            vec![
                (0, ResumeValueSource::Tagged(tagged_int)),
                (1, ResumeValueSource::Tagged(tagged_ref)),
            ],
            &consts,
            &alloc,
        );
        assert_eq!(*alloc.refs.borrow(), vec![0x5555]);
        assert_eq!(*alloc.ints.borrow(), vec![9]);
    }

    #[test]
    fn setfields_snapshot_ref_does_not_decode_int_colliding_live_int() {
        // stable_field_index can make the live Int sibling share
        // index+offset with the snapshot Ref leaf. Live-first
        // AbstractResumeDataReader.setfield dispatch would then
        // decode_int the TAGCONST. Snapshot field_type is the bank
        // snapshot_field_descr_info wrote.
        let collide = 0x1000_00aa;
        let int_fd = Arc::new(
            SimpleFieldDescr::new(collide, 16, 8, Type::Int, false).with_index_in_parent(0),
        );
        let parent: majit_ir::DescrRef = Arc::new(
            SimpleSizeDescr::new(0x3000_0006, 24, 1)
                .with_all_fielddescrs(vec![int_fd.clone() as Arc<dyn FieldDescr>]),
        );
        int_fd.set_parent_descr(&parent);
        let ref_info = FieldDescrInfo {
            index: collide,
            offset: 16,
            field_type: Type::Ref,
            field_size: 8,
        };
        let tagged_ref = tag(TAG_CONST_OFFSET, TAGCONST).unwrap();
        let consts = [Const::Ref(GcRef(0x8888))];
        let alloc = RecAllocator::default();
        allocate_virtual_obj(
            parent,
            vec![ref_info],
            vec![(0, ResumeValueSource::Tagged(tagged_ref))],
            &consts,
            &alloc,
        );
        assert_eq!(*alloc.refs.borrow(), vec![0x8888]);
        assert!(alloc.ints.borrow().is_empty());
    }

    #[test]
    fn setfields_duplicate_index_uses_positional_fieldnums() {
        // stable_field_index can give two snapshot leaves the same
        // FieldDescrInfo.index. virtual_info_from_rd used to key fields
        // by that hash, so find fed the Ref TAGCONST to the Int descr
        // (Const::getint). AbstractVirtualStructInfo.setfields is
        // positional.
        let ptr_fd = Arc::new(
            SimpleFieldDescr::new(0x1000_4002, 8, 8, Type::Ref, false).with_index_in_parent(0),
        );
        let int_fd = Arc::new(
            SimpleFieldDescr::new(0x1000_4001, 16, 8, Type::Int, false).with_index_in_parent(1),
        );
        let parent: majit_ir::DescrRef = Arc::new(
            SimpleSizeDescr::new(0x3000_0004, 24, 1).with_all_fielddescrs(vec![
                ptr_fd.clone() as Arc<dyn FieldDescr>,
                int_fd.clone() as Arc<dyn FieldDescr>,
            ]),
        );
        ptr_fd.set_parent_descr(&parent);
        int_fd.set_parent_descr(&parent);
        let collide = 0x1000_00ff;
        let ref_info = FieldDescrInfo {
            index: collide,
            offset: 8,
            field_type: Type::Ref,
            field_size: 8,
        };
        let int_info = FieldDescrInfo {
            index: collide,
            offset: 16,
            field_type: Type::Int,
            field_size: 8,
        };
        let tagged_ref = tag(TAG_CONST_OFFSET, TAGCONST).unwrap();
        let tagged_int = tag(11, TAGINT).unwrap();
        let rd = majit_ir::RdVirtualInfo::VirtualInfo {
            descr: Some(parent),
            type_id: 1,
            known_class: None,
            fielddescrs: vec![ref_info, int_info],
            fieldnums: vec![tagged_ref, tagged_int],
            descr_size: 24,
        };
        let vinfo = virtual_info_from_rd(&rd);
        match &vinfo {
            VirtualInfo::VirtualObj { fields, .. } => {
                assert_eq!(fields[0].0, 0);
                assert_eq!(fields[1].0, 1);
            }
            other => panic!("expected VirtualObj, got {other:?}"),
        }
        let consts = [Const::Ref(GcRef(0x6666))];
        let alloc = RecAllocator::default();
        let mut reader = ResumeDataDirectReader::new(
            &[0, 0],
            &consts,
            &[],
            FailArgSource::Slice(&[]),
            None,
            None,
            &alloc,
        );
        reader.virtuals_cache = VirtualCache::from_caches(vec![0], vec![0]);
        vinfo.allocate(&mut reader, 0, &alloc);
        assert_eq!(*alloc.refs.borrow(), vec![0x6666]);
        assert_eq!(*alloc.ints.borrow(), vec![11]);
    }

    #[test]
    fn setfields_duplicate_index_vstruct_uses_positional_fieldnums() {
        let ptr_fd = Arc::new(
            SimpleFieldDescr::new(0x1000_4102, 8, 8, Type::Ref, false).with_index_in_parent(0),
        );
        let int_fd = Arc::new(
            SimpleFieldDescr::new(0x1000_4101, 16, 8, Type::Int, false).with_index_in_parent(1),
        );
        let parent: majit_ir::DescrRef = Arc::new(
            SimpleSizeDescr::new(0x3000_0005, 24, 2).with_all_fielddescrs(vec![
                ptr_fd.clone() as Arc<dyn FieldDescr>,
                int_fd.clone() as Arc<dyn FieldDescr>,
            ]),
        );
        ptr_fd.set_parent_descr(&parent);
        int_fd.set_parent_descr(&parent);
        let collide = 0x1000_00fe;
        let ref_info = FieldDescrInfo {
            index: collide,
            offset: 8,
            field_type: Type::Ref,
            field_size: 8,
        };
        let int_info = FieldDescrInfo {
            index: collide,
            offset: 16,
            field_type: Type::Int,
            field_size: 8,
        };
        let tagged_ref = tag(TAG_CONST_OFFSET, TAGCONST).unwrap();
        let tagged_int = tag(13, TAGINT).unwrap();
        let rd = majit_ir::RdVirtualInfo::VStructInfo {
            typedescr: Some(parent),
            type_id: 2,
            fielddescrs: vec![ref_info, int_info],
            fieldnums: vec![tagged_ref, tagged_int],
            descr_size: 24,
        };
        let vinfo = virtual_info_from_rd(&rd);
        let consts = [Const::Ref(GcRef(0x7777))];
        let alloc = RecAllocator::default();
        let mut reader = ResumeDataDirectReader::new(
            &[0, 0],
            &consts,
            &[],
            FailArgSource::Slice(&[]),
            None,
            None,
            &alloc,
        );
        reader.virtuals_cache = VirtualCache::from_caches(vec![0], vec![0]);
        vinfo.allocate(&mut reader, 0, &alloc);
        assert_eq!(*alloc.refs.borrow(), vec![0x7777]);
        assert_eq!(*alloc.ints.borrow(), vec![13]);
    }
}

/// resume.py blackhole_from_resumedata
///
/// Build a chain of BlackholeInterpreters from encoded resume data.
/// Returns the topmost (innermost) interpreter.
///
/// `resolve_jitcode` corresponds to RPython's `jitcodes[jitcode_pos]` lookup
/// (`resume.py`). Matches upstream's `(jitcode, pc)` tuple result.
pub struct ResolvedJitCode {
    pub jitcode: std::sync::Arc<crate::jitcode::JitCode>,
    pub pc: usize,
    pub virtualizable_stack_base: Option<usize>,
}

impl ResolvedJitCode {
    pub fn new(jitcode: std::sync::Arc<crate::jitcode::JitCode>, pc: usize) -> Self {
        Self {
            jitcode,
            pc,
            virtualizable_stack_base: None,
        }
    }

    pub fn with_virtualizable_stack_base(mut self, stack_base: usize) -> Self {
        self.virtualizable_stack_base = Some(stack_base);
        self
    }
}

/// resume.py `consume_boxes` liveness split for a generic JitDriver
/// `setup_bridge_sym`: the live register indices of a guard's resume frame,
/// split by the three liveness banks (int / ref / float).  A frame decoded
/// by `rebuild_from_numbering` lays its `values` out in this same bank order
/// (all int-bank values, then all ref-bank, then float), so the i-th `int`
/// index here pairs with the i-th int-bank `RebuiltValue`.
#[derive(Default, Clone, Debug)]
pub struct FrameLivenessRegIndices {
    pub int: Vec<u32>,
    pub ref_: Vec<u32>,
    pub float: Vec<u32>,
}

impl FrameLivenessRegIndices {
    pub fn total_len(&self) -> usize {
        self.int.len() + self.ref_.len() + self.float.len()
    }
}

/// Materialized `enumerate_vars` kept for `JitDriver::start_bridge_tracing`
/// → `setup_bridge_sym`.
///
/// `setup_bridge_sym` is static and metainterp-blind: it runs in a different
/// function and has no jitcode, so it cannot walk `enumerate_vars` itself.
/// `start_bridge_tracing` stores these three banks on the trace ctx and the
/// macro reads them back. `bridge_from_guard_resume_position` does not use
/// this; it drives `enumerate_vars` once and pairs each index with the next
/// rebuilt value in the callback.
///
/// Returns empty banks when `pc` is not a decodable liveness startpoint
/// (`JitCode::can_decode_live_vars` is false, or the three length bytes do
/// not fit in `all_liveness`). `JitCode::get_live_vars_info` asserts on a
/// missing startpoint (`MissingLiveness`); a frame resuming through the
/// Python pc legitimately has no JitCode liveness at this coordinate, so
/// this declines instead of panicking. An empty return can mask a genuinely
/// bad resume coordinate (the caller then seeds nothing), so each decline
/// is logged under `MAJIT_BRIDGE_DEBUG`.
pub fn read_frame_liveness_reg_indices(
    jitcode: &crate::jitcode::JitCode,
    pc: usize,
    op_live: u8,
    all_liveness: &[u8],
) -> FrameLivenessRegIndices {
    if !jitcode.can_decode_live_vars(pc, op_live) {
        if crate::bridge_debug_enabled() {
            eprintln!(
                "[bridgeB] read_frame_liveness_reg_indices: no liveness startpoint at pc={pc} op_live={op_live} — declining (empty banks)"
            );
        }
        return FrameLivenessRegIndices::default();
    }
    let info = jitcode.get_live_vars_info(pc, op_live);
    if info + 2 >= all_liveness.len() {
        if crate::bridge_debug_enabled() {
            eprintln!(
                "[bridgeB] read_frame_liveness_reg_indices: liveness info {info} out of range (len={}) at pc={pc} — declining (empty banks)",
                all_liveness.len()
            );
        }
        return FrameLivenessRegIndices::default();
    }
    // `enumerate_vars` yields exactly these many indices per bank.
    let mut int = Vec::with_capacity(all_liveness[info] as usize);
    let mut ref_ = Vec::with_capacity(all_liveness[info + 1] as usize);
    let mut float = Vec::with_capacity(all_liveness[info + 2] as usize);
    majit_jitcode::codewriter::jitcode::enumerate_vars(
        info,
        all_liveness,
        |index| int.push(index),
        |index| ref_.push(index),
        |index| float.push(index),
    );
    FrameLivenessRegIndices { int, ref_, float }
}

/// One `resume.py` `ResumeDataBoxReader` callback result: the box to store
/// in the live register, and the concrete bits `_copy_data_from_miframe`
/// reads back. `Virtual` has no box yet: `consume_boxes` allocates it
/// through `getvirtual_ptr` (`materialize_bridge_virtual`) and stores
/// that box before any later `get_list_of_active_boxes`. `Unassigned`
/// stays empty (`box is None`).
pub fn resume_register_box(
    value: &majit_ir::resumedata::RebuiltValue,
    fail_values: &[i64],
) -> Option<(majit_ir::OpRef, i64)> {
    use majit_ir::resumedata::RebuiltValue;
    match value {
        RebuiltValue::Box(n, kind) => {
            let bits = *fail_values.get(*n)?;
            Some((majit_ir::OpRef::input_arg_typed(*n as u32, *kind), bits))
        }
        RebuiltValue::Const(majit_ir::Const::Int(v)) => Some((majit_ir::OpRef::const_int(*v), *v)),
        RebuiltValue::Const(majit_ir::Const::Ref(g)) => {
            Some((majit_ir::OpRef::const_ptr(*g), g.0 as i64))
        }
        RebuiltValue::Const(majit_ir::Const::Float(f)) => {
            Some((majit_ir::OpRef::const_float(*f), f.to_bits() as i64))
        }
        RebuiltValue::Virtual(_) | RebuiltValue::Unassigned => None,
    }
}

/// RAII guard that pops every resume-construction ref-slice root pushed
/// `warmstate.py:416-422 handle_fail(deadframe, ...)` reads a failing guard's
/// exit values out of the JITFRAME, and the JITFRAME stays a GC root — the
/// collector forwards its `jf_gcmap` slots in place — for the whole deopt.
/// That matters because both halves of the deopt allocate: bridge tracing
/// (`compile.py:701-717`) and blackhole reconstruction (`resume.py`), and  allow-line-citation
/// a minor collection in either moves exactly the objects those exit values
/// name. `decode_ref`'s `TAGBOX` arm then reads the moved-from address.
///
/// pyre has no host-visible JITFRAME: the backend copies the exit values into
/// a host `Vec<i64>` when `execute_token` returns and hands that buffer down
/// through `handle_fail`. This scope gives the copy the rooting the JITFRAME
/// provided — registering the `Ref`-typed slots only, matching the gcmap's
/// precision rather than scanning the whole array (an `Int` slot whose value
/// happened to equal a nursery object start would otherwise be "forwarded"
/// into a different integer).
pub struct DeadFrameRefRoots {
    base_depth: usize,
}

impl DeadFrameRefRoots {
    /// Root every `Ref`-typed slot of `values` until the returned scope drops.
    /// `is_ref` selects the slots, and each is pushed as its own one-element
    /// slice so the non-`Ref` slots in between stay untouched.
    ///
    /// # Safety
    /// `values` must stay alive and at a fixed address for the scope's whole
    /// lifetime; the collector writes forwarded addresses back through the
    /// registered pointers.
    pub unsafe fn enter(values: &mut [i64], is_ref: impl Fn(usize) -> bool) -> Self {
        let base_depth = majit_gc::shadow_stack::resume_ref_roots_depth();
        let base = values.as_mut_ptr();
        for index in 0..values.len() {
            if is_ref(index) {
                // SAFETY: `index < values.len()`, and the caller pins the
                // buffer for the scope. The collector is the only writer.
                unsafe {
                    majit_gc::shadow_stack::push_resume_ref_roots(std::slice::from_raw_parts_mut(
                        base.add(index),
                        1,
                    ));
                }
            }
        }
        DeadFrameRefRoots { base_depth }
    }
}

impl Drop for DeadFrameRefRoots {
    fn drop(&mut self) {
        majit_gc::shadow_stack::pop_resume_ref_roots_to(self.base_depth);
    }
}

/// during `blackhole_from_resumedata` back to the depth captured at entry.
/// Drop runs on ordinary return, `?` propagation, and panic unwind, so the
/// `virtuals_cache` / `registers_r` slices never outlive the construction
/// window in the GC root set.
pub struct ResumeRefRootsScope {
    base_depth: usize,
}

impl ResumeRefRootsScope {
    fn enter() -> Self {
        ResumeRefRootsScope {
            base_depth: majit_gc::shadow_stack::resume_ref_roots_depth(),
        }
    }
}

impl Drop for ResumeRefRootsScope {
    fn drop(&mut self) {
        majit_gc::shadow_stack::pop_resume_ref_roots_to(self.base_depth);
    }
}

pub fn prepare_resume_heap_with_roots<'a>(
    resumereader: &mut ResumeDataDirectReader<'a>,
    rd_virtuals: Option<&'a [VirtualInfo]>,
    rd_guard_pendingfields: Option<&[majit_ir::GuardPendingFieldEntry]>,
) -> ResumeRefRootsScope {
    // resume.py _prepare = _prepare_virtuals + _prepare_pendingfields.
    //
    // Root the lazily-filled `virtuals_cache` for the whole construction
    // window: decoding a virtual target (`getvirtual_ptr` → allocator)
    // materializes a fresh boxed object, and a minor collection triggered by a
    // later materialization relocates the already-built ones. The cache must be
    // rooted BEFORE the first materialization, including materialization that
    // happens while applying pending fields.
    let resume_roots = ResumeRefRootsScope::enter();
    // resume.py _prepare_virtuals
    resumereader.prepare_virtuals(rd_virtuals);
    unsafe {
        majit_gc::shadow_stack::push_resume_ref_roots(
            &mut resumereader.virtuals_cache.virtuals_ptr_cache,
        );
    }
    // resume.py _prepare_pendingfields
    if let Some(guard_pf) = rd_guard_pendingfields {
        resumereader.prepare_guard_pendingfields(guard_pf);
    }
    resume_roots
}

/// Prepare resume heap side effects without entering a blackhole.
///
/// This is the bridge / legacy-deopt counterpart to the heap-prepare block in
/// `blackhole_from_resumedata`: it sizes `rd_virtuals`, roots the virtual cache,
/// materializes virtual pending-field targets/values on demand, and applies
/// `rd_pendingfields`.
#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
)]
pub fn prepare_resume_heap<'a>(
    rd_numb: &'a [u8],
    rd_consts: &'a [majit_ir::Const],
    all_liveness: &'a [u8],
    deadframe: &'a [i64],
    deadframe_types: Option<&'a [majit_ir::Type]>,
    rd_virtuals: Option<&'a [VirtualInfo]>,
    rd_guard_pendingfields: Option<&[majit_ir::GuardPendingFieldEntry]>,
    allocator: &'a dyn BlackholeAllocator,
) {
    let Some(guard_pf) = rd_guard_pendingfields else {
        return;
    };
    if guard_pf.is_empty() {
        return;
    }
    let mut resumereader = ResumeDataDirectReader::new(
        rd_numb,
        rd_consts,
        all_liveness,
        deadframe.into(),
        deadframe_types,
        None,
        allocator,
    );
    let _resume_roots =
        prepare_resume_heap_with_roots(&mut resumereader, rd_virtuals, Some(guard_pf));
}

#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
)]
pub fn blackhole_from_resumedata<'a>(
    builder: &mut crate::blackhole::BlackholeInterpBuilder,
    resolve_jitcode: &dyn Fn(i32, i32) -> Option<ResolvedJitCode>,
    rd_numb: &'a [u8],
    rd_consts: &'a [majit_ir::Const],
    all_liveness: &'a [u8],
    deadframe: FailArgSource<'a>,
    deadframe_types: Option<&'a [majit_ir::Type]>,
    rd_virtuals: Option<&'a [VirtualInfo]>,
    rd_guard_pendingfields: Option<&[majit_ir::GuardPendingFieldEntry]>,
    vrefinfo: Option<&dyn VRefInfo>,
    vinfo: Option<&dyn VirtualizableInfo>,
    ginfo: Option<&dyn GreenfieldInfo>,
    // resume.py `blackhole_from_resumedata(..., all_virtuals=None)`.
    // `Some` only when resuming from a GUARD_NOT_FORCED whose
    // `handle_async_forcing` already materialized the virtuals
    // (compile.py:956-958): the reader then runs with
    // `resume_after_guard_not_forced == 2`, reuses this cache and leaves
    // the already-forced virtualizable alone.
    all_virtuals: Option<(Vec<i64>, Vec<i64>)>,
    allocator: &'a dyn BlackholeAllocator,
) -> (Box<BlackholeInterpreter>, i64) {
    // resume.py:1315-1327 The initialization is stack-critical code: it
    // must not be interrupted by StackOverflow, otherwise the
    // jit_virtual_refs are left in a dangling state.
    //
    // RPython wraps the body in try/finally so _stop() runs on every
    // exit path. The RAII CriticalCodeGuard gives us Drop-based
    // guarantee — ordinary returns, `?` propagation, AND panic unwind
    // all re-enable the report_error flag.
    let _cc_guard = crate::CriticalCodeGuard::enter();
    // resume.py:1317-1321
    let resuming_after_guard_not_forced = all_virtuals.is_some();
    let mut resumereader = ResumeDataDirectReader::new(
        rd_numb,
        rd_consts,
        all_liveness,
        deadframe,
        deadframe_types,
        all_virtuals,
        allocator,
    );

    // resume.py: `_prepare` — and so both `_prepare_virtuals` and
    // `_prepare_pendingfields` — runs only in the `all_virtuals is None`
    // case. Resuming after a GUARD_NOT_FORCED, the force's own reader
    // already built the virtuals and applied the pending fields; redoing
    // either would rebuild the objects it handed to the interpreter and
    // replay its heap writes.
    let empty_pending = rd_guard_pendingfields.is_none_or(|p| p.is_empty());
    let empty_virtuals = rd_virtuals.is_none_or(|v| v.is_empty());
    let _resume_roots = if resuming_after_guard_not_forced {
        Some(prepare_resume_heap_with_roots(
            &mut resumereader,
            None,
            None,
        ))
    } else if empty_virtuals && empty_pending {
        // regex leaf: nvirtuals=0 and no pending fields. Skip the
        // empty `VirtualCache` root and the pending-field walk.
        None
    } else {
        Some(prepare_resume_heap_with_roots(
            &mut resumereader,
            rd_virtuals,
            rd_guard_pendingfields,
        ))
    };

    // resume.py:1325
    resumereader.consume_vref_and_vable(vrefinfo, vinfo, ginfo);
    drop(_cc_guard);

    // consume_vable_info rooted `virtualizable_ptr` before restoring any
    // fields.  The reader owns that root through this chain build and drops it
    // with its `virtualizable_root_base_depth` scope.

    // resume.py:1332-1343
    // Build chain bottom-up: first frame acquired is the outermost.
    let mut curbh: Option<Box<BlackholeInterpreter>> = None;

    while !resumereader.done_reading() {
        // resume.py:1334-1336
        let mut nextbh = builder.acquire_interp();
        nextbh.nextblackholeinterp = curbh;

        // resume.py:1338-1340
        let (jitcode_pos, pc) = resumereader.read_jitcode_pos_pc();
        // resume.py:1339-1340: jitcode = jitcodes[jitcode_pos]; curbh.setposition(jitcode, pc).
        let resolved = resolve_jitcode(jitcode_pos, pc).unwrap_or_else(|| {
            panic!("blackhole_from_resumedata: invalid jitcode index {jitcode_pos} at pc {pc}")
        });
        if crate::bh_debug_enabled() {
            eprintln!(
                "[bh-frame] jitcode_pos={jitcode_pos} encoded_pc={pc} resolved_pc={}",
                resolved.pc
            );
        }
        nextbh.setposition_ref(&resolved.jitcode, resolved.pc);
        if let Some(stack_base) = resolved.virtualizable_stack_base {
            nextbh.virtualizable_stack_base = stack_base;
        }

        // `setposition` sized this frame's register files; root the ref
        // bank before filling it so a materialization collection during
        // `consume_one_section` forwards the refs already written here (the
        // Vec buffer is stable — `setarg_r` only indexes — and the frame
        // itself is already boxed, so chaining it moves a pointer). A pooled
        // interpreter is rooted for its whole life already.
        if !nextbh.is_rooted() {
            unsafe {
                majit_gc::shadow_stack::push_resume_ref_roots(&mut nextbh.registers_r);
            }
        }

        // resume.py:1341
        resumereader.consume_one_section(&mut nextbh, vinfo);

        // resume.py:1342
        if nextbh.op_rvmprof_code != majit_jitcode::insns::BC_ABSENT {
            nextbh.handle_rvmprof_enter();
        }

        curbh = Some(nextbh);
    }

    // Read the (possibly forwarded) virtualizable pointer back from the rooted
    // reader slot: a minor collection during the loop above rewrites it in
    // place to the to-space address.
    let virtualizable_ptr = resumereader.virtualizable_ptr;

    let curbh = curbh.expect("blackhole_from_resumedata: resume data contains no frame");
    (curbh, virtualizable_ptr)
}

/// resume.py force_from_resumedata
///
/// Force all virtuals from resume data without running a blackhole.
/// Used for GUARD_NOT_FORCED handling.
///
/// Returns (virtuals_cache_ptr, virtuals_cache_int) — RPython VirtualCache
/// parity — plus the virtualizable the vable section named, which the caller
/// writes onto the deadframe via `cpu.set_savedata_ref`.
#[allow(clippy::needless_lifetimes)]
#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
)]
pub fn force_from_resumedata<'a>(
    profiler: &crate::jitprof::JitProfiler,
    rd_numb: &'a [u8],
    rd_consts: &'a [majit_ir::Const],
    all_liveness: &'a [u8],
    deadframe: FailArgSource<'a>,
    deadframe_types: Option<&'a [majit_ir::Type]>,
    rd_virtuals: Option<&'a [VirtualInfo]>,
    rd_guard_pendingfields: Option<&[majit_ir::GuardPendingFieldEntry]>,
    vrefinfo: Option<&dyn VRefInfo>,
    vinfo: Option<&dyn VirtualizableInfo>,
    ginfo: Option<&dyn GreenfieldInfo>,
    allocator: &'a dyn BlackholeAllocator,
) -> (Vec<i64>, Vec<i64>, i64) {
    let _bh_phase = majit_gc::BhProbePhase::enter("resume");
    // resume.py:1346
    profiler.count(crate::pyjitpl::counters::FORCE_VIRTUALIZABLES, 1);
    // resume.py:1347-1348
    let mut resumereader = ResumeDataDirectReader::new(
        rd_numb,
        rd_consts,
        all_liveness,
        deadframe,
        deadframe_types,
        None,
        allocator,
    );
    // resume.py common-case __init__ calls self._prepare(storage)
    // before handling_async_forcing() flips the GUARD_NOT_FORCED state.
    //
    // The scope must outlive `force_all_virtuals` below. `getvirtual_ptr`
    // returns the cache slot rather than `allocate`'s result precisely because
    // a materialization collection is expected to forward that slot in place;
    // with a bare `prepare` the slice is registered nowhere, so a minor
    // triggered while materializing virtual N leaves every virtual already
    // written for N-1 pointing into from-space, and a major sweeps them.
    let _resume_roots =
        prepare_resume_heap_with_roots(&mut resumereader, rd_virtuals, rd_guard_pendingfields);
    resumereader.handling_async_forcing();
    // resume.py:1350
    resumereader.consume_vref_and_vable(vrefinfo, vinfo, ginfo);
    // resume.py:1404 the virtualizable the vable section just named. The scope
    // above roots `virtuals_ptr_cache` only, not this field, so nothing would
    // forward the reader's slot in place — read it before `force_all_virtuals`
    // allocates. The value stays valid across that window because a
    // virtualizable is an interpreter-created frame from `FrameBox::new` →
    // `try_gc_alloc_stable_raw`, which no collection moves.
    let virtualizable_ptr = resumereader.virtualizable_ptr;
    // resume.py: return resumereader.force_all_virtuals()
    let (ptrs, ints) = resumereader.force_all_virtuals();
    (ptrs.to_vec(), ints.to_vec(), virtualizable_ptr)
}
