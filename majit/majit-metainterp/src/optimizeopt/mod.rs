/// JIT optimization pipeline.
///
/// Translated from rpython/jit/metainterp/optimizeopt/.
///
/// The optimizer chains multiple passes, each implementing the Optimization trait.
/// Operations flow through the chain: IntBounds → Rewrite → Virtualize → String →
/// Pure → Guard → Simplify → Heap (configurable).
#[expect(
    clippy::collapsible_if,
    clippy::cloned_ref_to_slice_refs,
    clippy::manual_range_contains,
    reason = "this module is generated verbatim from real.rules; local rewrites would fail the freshness check, so lint cleanup belongs in the generator before the artifact can change"
)]
pub mod autogenintrules;
pub mod bridgeopt;
pub mod dependency;
pub mod earlyforce;
pub mod guard;
pub mod heap;
pub mod info;
pub mod intbounds;
pub mod intdiv;
pub mod intutils;
// optimize module is at crate::optimize (RPython: metainterp/optimize.py)
pub mod optimizer;
pub use optimizer::{Optimization, OptimizationResult};
pub mod pure;
pub mod rawbuffer;
pub mod renamer;
pub mod rewrite;
pub mod schedule;
pub mod shortpreamble;
pub mod simplify;
pub mod unroll;
pub mod util;
pub mod vector;
pub mod version;
pub mod virtualize;
pub mod virtualstate;
pub mod vstring;
// walkvirtual moved to crate::walkvirtual (RPython: metainterp/walkvirtual.py)

use crate::optimizeopt::intutils::{IntBound, IntBoundMakeGuards};
use crate::resume::SnapshotBox;
use indexmap::{IndexMap, IndexSet};
use info::{EnsuredPtrInfo, PtrInfo};
use majit_ir::operand::Operand;
use majit_ir::{DescrRef, GcRef, Op, OpCode, OpRc, OpRef, Type, Value};
use rustc_hash::{FxBuildHasher, FxHashMap};

/// `optimizer.py` extra-operations list. Almost always 0–2 ops (one
/// lazy SETFIELD / force). Inline those so the first `VecDeque` grow
/// (64 B) is gone.
#[derive(Default)]
pub(crate) struct ExtraQueue(smallvec::SmallVec<[(usize, majit_ir::OpRc); 2]>);

impl ExtraQueue {
    pub(crate) fn new() -> Self {
        Self(smallvec::SmallVec::new())
    }

    pub(crate) fn push_back(&mut self, item: (usize, majit_ir::OpRc)) {
        self.0.push(item);
    }

    pub(crate) fn push_front(&mut self, item: (usize, majit_ir::OpRc)) {
        self.0.insert(0, item);
    }

    pub(crate) fn pop_front(&mut self) -> Option<(usize, majit_ir::OpRc)> {
        if self.0.is_empty() {
            None
        } else {
            Some(self.0.remove(0))
        }
    }

    #[allow(dead_code)] // optimizer.py extra-op queue front
    pub(crate) fn front(&self) -> Option<&(usize, majit_ir::OpRc)> {
        self.0.first()
    }

    #[allow(dead_code)] // optimizer.py extra-op queue back
    pub(crate) fn back(&self) -> Option<&(usize, majit_ir::OpRc)> {
        self.0.last()
    }

    pub(crate) fn remove(&mut self, at: usize) -> Option<(usize, majit_ir::OpRc)> {
        (at < self.0.len()).then(|| self.0.remove(at))
    }

    pub(crate) fn iter(&self) -> std::slice::Iter<'_, (usize, majit_ir::OpRc)> {
        self.0.iter()
    }

    #[allow(dead_code)] // optimizer.py extra-op queue len
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Extend<(usize, majit_ir::OpRc)> for ExtraQueue {
    fn extend<T: IntoIterator<Item = (usize, majit_ir::OpRc)>>(&mut self, iter: T) {
        self.0.extend(iter);
    }
}

impl IntoIterator for ExtraQueue {
    type Item = (usize, majit_ir::OpRc);
    type IntoIter = smallvec::IntoIter<[(usize, majit_ir::OpRc); 2]>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// ConstPtr indexes created while an optimizer pass runs.
///
/// `history.py` `ConstPtr.value` lives on the op the optimizer is rewriting.
/// A bridge seeds `input_ops` empty, and the recorder pool does not yet name
/// an index minted only on those ops, on `Forwarded`, or in `OptContext`.
/// The compile walker traces the published holders. Drop pops by pointer
/// identity and does not dereference the holder: the `Vec` may already have
/// moved.
#[derive(Default)]
pub(crate) struct CompileLiveOpRoots {
    contexts: Vec<usize>,
    spans: Vec<(usize, usize)>,
    vecs: Vec<usize>,
    /// `TreeLoop` still owned by the compile after `compile_tracing` is taken.
    loops: Vec<usize>,
}

/// Pops one published holder. Empty when the optimizer has no metainterp slot.
#[must_use = "dropping the guard unpublishes the holder"]
pub(crate) struct LiveOpPublication {
    roots: *mut CompileLiveOpRoots,
    kind: u8,
    key: usize,
}

impl CompileLiveOpRoots {
    pub(crate) fn clear(&mut self) {
        self.contexts.clear();
        self.spans.clear();
        self.vecs.clear();
        self.loops.clear();
    }

    /// True when no compile holder is published. The snapshot-walk test
    /// checks the guard released every vec, including a `TreeLoop`.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.contexts.is_empty()
            && self.spans.is_empty()
            && self.vecs.is_empty()
            && self.loops.is_empty()
    }

    pub(crate) fn publish_context(&mut self, ctx: *const OptContext) -> LiveOpPublication {
        let key = ctx as usize;
        if key == 0 {
            return LiveOpPublication::noop();
        }
        self.contexts.push(key);
        LiveOpPublication {
            roots: self as *mut _,
            kind: 1,
            key,
        }
    }

    pub(crate) fn publish_span(&mut self, ops: &[OpRc]) -> LiveOpPublication {
        if ops.is_empty() {
            return LiveOpPublication::noop();
        }
        let key = ops.as_ptr() as usize;
        self.spans.push((key, ops.len()));
        LiveOpPublication {
            roots: self as *mut _,
            kind: 2,
            key,
        }
    }

    // The stored address is the `Vec` header. A slice would not name that
    // header once the caller reallocates the buffer.
    #[allow(clippy::ptr_arg)]
    pub(crate) fn publish_vec(&mut self, ops: &Vec<OpRc>) -> LiveOpPublication {
        let key = ops as *const Vec<OpRc> as usize;
        self.vecs.push(key);
        LiveOpPublication {
            roots: self as *mut _,
            kind: 3,
            key,
        }
    }

    /// `trace` must stay put until the guard drops. A moved `TreeLoop`
    /// leaves this address pointing at the old header.
    pub(crate) fn publish_tree_loop(
        &mut self,
        trace: &mut crate::history::TreeLoop,
    ) -> LiveOpPublication {
        let key = trace as *mut crate::history::TreeLoop as usize;
        self.loops.push(key);
        LiveOpPublication {
            roots: self as *mut _,
            kind: 4,
            key,
        }
    }

    pub(crate) fn walk(&self, visitor: &mut dyn FnMut(&mut GcRef)) {
        for &addr in &self.contexts {
            if addr == 0 {
                continue;
            }
            // SAFETY: the publisher drops this entry before moving the
            // context. The walker runs only while that compile is paused
            // for a collection.
            unsafe { (*(addr as *mut OptContext)).walk_live_const_ptrs(visitor) };
        }
        for &(ptr, len) in &self.spans {
            if ptr == 0 || len == 0 {
                continue;
            }
            let ops = unsafe { std::slice::from_raw_parts(ptr as *const OpRc, len) };
            for op in ops {
                walk_live_op(op, visitor);
            }
        }
        for &addr in &self.vecs {
            if addr == 0 {
                continue;
            }
            let ops = unsafe { &*(addr as *const Vec<OpRc>) };
            for op in ops {
                walk_live_op(op, visitor);
            }
        }
        for &addr in &self.loops {
            if addr == 0 {
                continue;
            }
            // SAFETY: the publisher drops this entry before moving the
            // `TreeLoop`. The walker runs only while that compile is paused.
            let trace = unsafe { &mut *(addr as *mut crate::history::TreeLoop) };
            for op in &trace.ops {
                walk_live_op(op, visitor);
            }
            for ia in &trace.inputargs {
                if let Some(majit_ir::Value::Ref(mut r)) = ia.get_value() {
                    visitor(&mut r);
                    ia.set_value(majit_ir::Value::Ref(r));
                }
            }
            for snapshot in &mut trace.snapshots {
                snapshot.walk_const_ptr_refs(visitor);
            }
        }
    }
}

impl LiveOpPublication {
    pub(crate) fn noop() -> Self {
        Self {
            roots: std::ptr::null_mut(),
            kind: 0,
            key: 0,
        }
    }
}

impl Drop for LiveOpPublication {
    fn drop(&mut self) {
        if self.roots.is_null() {
            return;
        }
        // SAFETY: `roots` is the `CompileLiveOpRoots` that pushed `key`.
        // Popping does not read the published holder.
        unsafe {
            let roots = &mut *self.roots;
            match self.kind {
                1 => {
                    if let Some(i) = roots.contexts.iter().rposition(|p| *p == self.key) {
                        roots.contexts.swap_remove(i);
                    }
                }
                2 => {
                    if let Some(i) = roots.spans.iter().rposition(|(p, _)| *p == self.key) {
                        roots.spans.swap_remove(i);
                    }
                }
                3 => {
                    if let Some(i) = roots.vecs.iter().rposition(|p| *p == self.key) {
                        roots.vecs.swap_remove(i);
                    }
                }
                4 => {
                    if let Some(i) = roots.loops.iter().rposition(|p| *p == self.key) {
                        roots.loops.swap_remove(i);
                    }
                }
                _ => {}
            }
        }
    }
}

fn walk_live_op(op: &Op, visitor: &mut dyn FnMut(&mut GcRef)) {
    op.walk_const_ptr_refs_mut(visitor);
    // A folded const can sit on `_forwarded` before emit copies it into an arg.
    // `ForwardedSlot::borrow` copies the packed word; `walk_const_ptr_refs`
    // still traces the `SmallWide` index.
    op.forwarded().borrow().walk_const_ptr_refs(visitor);
    if let Some(Value::Ref(mut gcref)) = op.get_value() {
        visitor(&mut gcref);
        op.set_value(Value::Ref(gcref));
    }
    // recorder.rs `walk_const_ptr_refs` walks each slot's QuasiImmutDescr
    // (`quasiimmut.py self.struct` / `constantfieldbox`). After
    // `compile_tracing` is taken, this published-op walk is that holder.
    // `heap.py is_still_valid_for` compares those words to the optimizer's
    // ConstPtr; a compile-time minor that forwarded the ConstPtr but not
    // the descr raises `InvalidLoop('quasi immutable field changed during
    // tracing')` on a field that did not change.
    if let Some(descr) = op.getdescr()
        && let Some(qd) = descr.as_quasi_immut_descr()
    {
        qd.walk_const_ptr_refs(visitor);
    }
}

#[cfg(test)]
mod walk_live_op_quasiimmut_tests {
    use super::*;
    use majit_ir::QuasiImmutHandle;

    #[derive(Debug)]
    struct Handle;
    impl QuasiImmutHandle for Handle {
        fn is_current(&self) -> bool {
            true
        }
        fn register_loop_token(&self, _token: &std::sync::Arc<dyn majit_ir::QuasiImmutLoopToken>) {}
        fn instance_identity(&self) -> usize {
            1
        }
    }

    #[test]
    fn walk_live_op_forwards_quasiimmut_struct_ptr() {
        let field = std::sync::Arc::new(majit_ir::SimpleFieldDescr::new(0, 8, 8, Type::Ref, false))
            as DescrRef;
        let descr = std::sync::Arc::new(majit_ir::QuasiImmutDescr::new(
            field,
            0x4100,
            std::sync::Arc::new(Handle),
            None,
        )) as DescrRef;
        let obj = Operand::bound_from_opref(OpRef::input_arg_typed(0, Type::Ref));
        let op = OpRc::new(Op::with_descr(
            OpCode::QuasiimmutField,
            &[obj],
            descr.clone(),
        ));
        let ops = vec![op];
        let mut roots = CompileLiveOpRoots::default();
        let _published = roots.publish_vec(&ops);
        roots.walk(&mut |slot| {
            if slot.0 == 0x4100 {
                slot.0 = 0x5100;
            }
        });
        assert_eq!(descr.as_quasi_immut_descr().unwrap().struct_ptr(), 0x5100);
    }
}

/// Compile-time snapshot box list. RPython keeps the live boxes themselves;
/// this adapter stores `SnapshotBox` (OpRef plus the iterator-cache
/// Operand) in a side table. Twelve boxes mint from the chunked slab.
/// `create_numbering_arc` mints a `NumberingRef` slab cell
/// (`resumecode.create_numbering` / `lltype.malloc(NUMBERING)`).
const SNAP_LIST_SLAB: usize = 12;
const SNAP_LIST_SLAB_BYTES: usize = SNAP_LIST_SLAB * std::mem::size_of::<SnapshotBox>();
const SNAP_LIST_SLAB_BIT: u32 = 1 << 31;
const SNAP_LIST_CHUNK: usize = 2048;

const _: () = assert!(
    std::mem::size_of::<SnapshotBox>() > 0
        && std::mem::size_of::<SnapshotBox>() <= SNAP_LIST_SLAB_BYTES
);

struct SnapListHeap {
    chunks: Vec<(*mut SnapshotBox, usize)>,
    free: Vec<*mut SnapshotBox>,
}

unsafe impl Send for SnapListHeap {}
unsafe impl Sync for SnapListHeap {}

static SNAP_LIST_HEAP: std::sync::Mutex<SnapListHeap> = std::sync::Mutex::new(SnapListHeap {
    chunks: Vec::new(),
    free: Vec::new(),
});

fn alloc_snap_list_slab() -> *mut SnapshotBox {
    let mut heap = SNAP_LIST_HEAP.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(p) = heap.free.pop() {
        return p;
    }
    if let Some((base, used)) = heap.chunks.last_mut()
        && *used < SNAP_LIST_CHUNK
    {
        let p = unsafe { (*base).add(*used * SNAP_LIST_SLAB) };
        *used += 1;
        return p;
    }
    let layout = std::alloc::Layout::from_size_align(
        SNAP_LIST_CHUNK * SNAP_LIST_SLAB_BYTES,
        std::mem::align_of::<SnapshotBox>(),
    )
    .expect("snapshot-list chunk");
    let base = unsafe { std::alloc::alloc(layout) as *mut SnapshotBox };
    assert!(!base.is_null(), "snapshot-list chunk alloc failed");
    heap.chunks.push((base, 1));
    base
}

fn free_snap_list_slab(p: *mut SnapshotBox) {
    SNAP_LIST_HEAP
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .free
        .push(p);
}

pub struct SnapshotBoxList {
    ptr: std::ptr::NonNull<SnapshotBox>,
    len: u32,
    cap: u32,
}

unsafe impl Send for SnapshotBoxList {}
unsafe impl Sync for SnapshotBoxList {}

impl SnapshotBoxList {
    pub fn new() -> Self {
        Self {
            ptr: std::ptr::NonNull::dangling(),
            len: 0,
            cap: 0,
        }
    }

    pub fn with_capacity(n: usize) -> Self {
        if n == 0 {
            return Self::new();
        }
        if n <= SNAP_LIST_SLAB {
            return Self {
                ptr: unsafe { std::ptr::NonNull::new_unchecked(alloc_snap_list_slab()) },
                len: 0,
                cap: SNAP_LIST_SLAB as u32 | SNAP_LIST_SLAB_BIT,
            };
        }
        let layout = std::alloc::Layout::array::<SnapshotBox>(n).expect("snapshot box list");
        let ptr = unsafe { std::alloc::alloc(layout) as *mut SnapshotBox };
        assert!(!ptr.is_null(), "snapshot box list alloc failed");
        Self {
            ptr: unsafe { std::ptr::NonNull::new_unchecked(ptr) },
            len: 0,
            cap: n as u32,
        }
    }

    fn capacity(&self) -> usize {
        (self.cap & !SNAP_LIST_SLAB_BIT) as usize
    }

    fn is_slab(&self) -> bool {
        self.cap & SNAP_LIST_SLAB_BIT != 0
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[SnapshotBox] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len as usize) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [SnapshotBox] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len as usize) }
    }

    pub fn push(&mut self, item: SnapshotBox) {
        let len = self.len as usize;
        if len == self.capacity() {
            self.grow();
        }
        unsafe {
            self.ptr.as_ptr().add(len).write(item);
        }
        self.len += 1;
    }

    fn grow(&mut self) {
        let old_len = self.len as usize;
        let old_cap = self.capacity();
        let new_cap = (old_cap.max(1) * 2).max(old_len + 1);
        let layout = std::alloc::Layout::array::<SnapshotBox>(new_cap).expect("snapshot box grow");
        let new_ptr = unsafe { std::alloc::alloc(layout) as *mut SnapshotBox };
        assert!(!new_ptr.is_null(), "snapshot box grow failed");
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr, old_len);
        }
        self.dealloc_storage(old_len);
        self.ptr = unsafe { std::ptr::NonNull::new_unchecked(new_ptr) };
        self.cap = new_cap as u32;
    }

    fn dealloc_storage(&mut self, len: usize) {
        if self.capacity() == 0 {
            return;
        }
        if self.is_slab() {
            free_snap_list_slab(self.ptr.as_ptr());
        } else {
            let layout = std::alloc::Layout::array::<SnapshotBox>(self.capacity())
                .expect("snapshot box dealloc");
            unsafe {
                std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
        let _ = len;
    }
}

impl Default for SnapshotBoxList {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SnapshotBoxList {
    fn clone(&self) -> Self {
        let mut out = Self::with_capacity(self.len as usize);
        for item in self.as_slice() {
            out.push(item.clone());
        }
        out
    }
}

impl Drop for SnapshotBoxList {
    fn drop(&mut self) {
        unsafe {
            for i in 0..self.len as usize {
                self.ptr.as_ptr().add(i).drop_in_place();
            }
        }
        self.dealloc_storage(self.len as usize);
    }
}

impl std::ops::Deref for SnapshotBoxList {
    type Target = [SnapshotBox];

    fn deref(&self) -> &[SnapshotBox] {
        self.as_slice()
    }
}

impl std::ops::DerefMut for SnapshotBoxList {
    fn deref_mut(&mut self) -> &mut [SnapshotBox] {
        self.as_mut_slice()
    }
}

impl Extend<SnapshotBox> for SnapshotBoxList {
    fn extend<T: IntoIterator<Item = SnapshotBox>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        if let Some(extra) = iter
            .size_hint()
            .0
            .checked_sub(self.capacity() - self.len as usize)
            && extra > 0
            && !self.is_slab()
        {
            let want = self.len as usize + extra;
            if want > self.capacity() {
                let layout =
                    std::alloc::Layout::array::<SnapshotBox>(want).expect("snapshot box extend");
                let new_ptr = unsafe { std::alloc::alloc(layout) as *mut SnapshotBox };
                assert!(!new_ptr.is_null(), "snapshot box extend failed");
                unsafe {
                    std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr, self.len as usize);
                }
                self.dealloc_storage(self.len as usize);
                self.ptr = unsafe { std::ptr::NonNull::new_unchecked(new_ptr) };
                self.cap = want as u32;
            }
        }
        for item in iter {
            self.push(item);
        }
    }
}

impl FromIterator<SnapshotBox> for SnapshotBoxList {
    fn from_iter<T: IntoIterator<Item = SnapshotBox>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut list = Self::with_capacity(iter.size_hint().0);
        list.extend(iter);
        list
    }
}

impl From<Vec<SnapshotBox>> for SnapshotBoxList {
    fn from(v: Vec<SnapshotBox>) -> Self {
        v.into_iter().collect()
    }
}

impl<'a> IntoIterator for &'a SnapshotBoxList {
    type Item = &'a SnapshotBox;
    type IntoIter = std::slice::Iter<'a, SnapshotBox>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a> IntoIterator for &'a mut SnapshotBoxList {
    type Item = &'a mut SnapshotBox;
    type IntoIter = std::slice::IterMut<'a, SnapshotBox>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

pub type SnapshotBoxes = Vec<Option<SnapshotBoxList>>;

/// Concatenated per-snapshot runs, matching `opencoder.py`'s flat snapshot
/// payloads. Regex `and`/`or` inlines past eight frames
/// (`ResumeDataLoopMemo::number_from_parts`), so a `Vec<T>` per snapshot is a
/// 72 B malloc (9 × usize) on every compiled guard.
#[derive(Clone, Debug)]
pub struct SnapshotRunTable<T> {
    data: Vec<T>,
    /// `(start, len)` for snapshot id `i`. `len == u32::MAX` means absent.
    spans: Vec<(u32, u32)>,
}

const SNAP_RUN_ABSENT: u32 = u32::MAX;

impl<T> SnapshotRunTable<T> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            spans: Vec::new(),
        }
    }

    /// `snapshots` is the snapshot count; `frames_hint` is an optional
    /// per-snapshot depth. Prefer [`Self::reserve_frames`] with the captured
    /// total when it is known (regex `and`/`or` inlines past eight).
    pub fn with_capacity(snapshots: usize, frames_hint: usize) -> Self {
        Self {
            data: Vec::with_capacity(snapshots.saturating_mul(frames_hint)),
            spans: Vec::with_capacity(snapshots),
        }
    }

    pub fn reserve_frames(&mut self, frames: usize) {
        if frames > self.data.capacity() {
            self.data.reserve(frames - self.data.len());
        }
    }

    pub fn get(&self, pos: i32) -> Option<&[T]> {
        if pos < 0 {
            return None;
        }
        let &(start, len) = self.spans.get(pos as usize)?;
        if len == SNAP_RUN_ABSENT {
            return None;
        }
        let start = start as usize;
        Some(&self.data[start..start + len as usize])
    }

    pub fn push_run(&mut self, items: impl IntoIterator<Item = T>) {
        let start = self.data.len() as u32;
        self.data.extend(items);
        let len = self.data.len() as u32 - start;
        self.spans.push((start, len));
    }

    pub fn insert_run(&mut self, pos: i32, items: impl IntoIterator<Item = T>) {
        assert!(pos >= 0, "snapshot position must be non-negative");
        let start = self.data.len() as u32;
        self.data.extend(items);
        let len = self.data.len() as u32 - start;
        let idx = pos as usize;
        if self.spans.len() <= idx {
            self.spans.resize(idx + 1, (0, SNAP_RUN_ABSENT));
        }
        self.spans[idx] = (start, len);
    }

    /// Remap a snapshot id onto an existing run. Sizes and pcs are immutable
    /// after capture, so the two ids share one slice (unroll.py copies the
    /// list identity, not a second payload).
    pub fn copy_run(&mut self, from: i32, to: i32) {
        if from < 0 || to < 0 {
            return;
        }
        let Some(&span) = self.spans.get(from as usize) else {
            return;
        };
        if span.1 == SNAP_RUN_ABSENT {
            return;
        }
        let idx = to as usize;
        if self.spans.len() <= idx {
            self.spans.resize(idx + 1, (0, SNAP_RUN_ABSENT));
        }
        self.spans[idx] = span;
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.spans.clear();
    }
}

impl<T> Default for SnapshotRunTable<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PartialEq> PartialEq for SnapshotRunTable<T> {
    fn eq(&self, other: &Self) -> bool {
        let n = self.spans.len().max(other.spans.len());
        (0..n).all(|i| self.get(i as i32) == other.get(i as i32))
    }
}

impl<T: Eq> Eq for SnapshotRunTable<T> {}

pub type SnapshotFrameSizes = SnapshotRunTable<usize>;
pub type SnapshotFramePcs = SnapshotRunTable<(i32, i32)>;

#[cfg(test)]
mod snapshot_box_list_size {
    #[test]
    fn option_snapshot_box_list_niches() {
        assert_eq!(
            std::mem::size_of::<Option<super::SnapshotBoxList>>(),
            16,
            "Option<SnapshotBoxList> must niche so a 2-slot map leaves 48 B"
        );
    }

    #[test]
    fn twelve_snapshot_boxes_stay_on_the_slab() {
        let list = super::SnapshotBoxList::with_capacity(12);
        assert!(
            list.is_slab(),
            "a 12-box resume frame must use the chunked slab, not a 96 B malloc"
        );
        assert!(list.capacity() >= 12);
    }

    #[test]
    fn nine_frame_sizes_share_one_run_table() {
        let mut sizes = super::SnapshotFrameSizes::with_capacity(2, 16);
        sizes.push_run(0..9);
        sizes.push_run(9..18);
        assert_eq!(sizes.get(0), Some(&[0, 1, 2, 3, 4, 5, 6, 7, 8][..]));
        assert_eq!(sizes.get(1), Some(&[9, 10, 11, 12, 13, 14, 15, 16, 17][..]));
        sizes.copy_run(0, 3);
        assert_eq!(sizes.get(3), sizes.get(0));
        assert!(sizes.get(2).is_none());
    }
}
type OpRefFxIndexMap<V> = indexmap::IndexMap<OpRef, V, FxBuildHasher>;

pub(crate) fn snapshot_get<T>(store: &[Option<T>], pos: i32) -> Option<&T> {
    if pos < 0 {
        return None;
    }
    store.get(pos as usize).and_then(Option::as_ref)
}

pub(crate) fn snapshot_contains<T>(store: &[Option<T>], pos: i32) -> bool {
    snapshot_get(store, pos).is_some()
}

/// Bind each snapshot slot's iterator-cache box (`TraceIterator._cache`).
/// Call while the OpRef is still in the cache's key space, before a
/// later remap rewrites it.
pub(crate) fn attach_snapshot_map_iter_cache(map: &mut SnapshotBoxes, cache: &[Option<Operand>]) {
    for list in map.iter_mut().flatten() {
        for boxref in list.iter_mut() {
            boxref.attach_iter_cache(cache);
        }
    }
}

pub(crate) fn attach_resume_snapshot_caches(
    boxes: &mut SnapshotBoxes,
    vable: &mut SnapshotBoxes,
    vref: &mut SnapshotBoxes,
    cache: &[Option<Operand>],
) {
    attach_snapshot_map_iter_cache(boxes, cache);
    attach_snapshot_map_iter_cache(vable, cache);
    attach_snapshot_map_iter_cache(vref, cache);
}

pub(crate) fn snapshot_insert<T>(store: &mut Vec<Option<T>>, pos: i32, value: T) {
    assert!(pos >= 0, "snapshot position must be non-negative");
    let idx = pos as usize;
    if store.len() <= idx {
        store.resize_with(idx + 1, || None);
    }
    store[idx] = Some(value);
}

pub(crate) fn next_snapshot_pos<T>(store: &[Option<T>]) -> i32 {
    store.len() as i32
}

pub(crate) use crate::majit_log_enabled;

/// info.py `getrawptrinfo` / `getptrinfo` return shape, with
/// RPython `_forwarded` object identity preserved.
///
/// Two variants mirror upstream's two return paths:
///   - `Const(PtrInfo)` — fresh `ConstPtrInfo(op)` synthesis
///     (info.py / 888-889).  Upstream allocates a brand-new
///     `ConstPtrInfo` per call; pyre carries the freshly built
///     `PtrInfo::Constant(_)` inline.
///   - `Live(Rc<RefCell<PtrInfo>>)` — the `return fw` arm
///     (info.py / 890-893).  Carries the live `Rc` handle
///     into the chain terminal's `_forwarded` cell so RPython object
///     identity is preserved: two `Live` handles cloned from the
///     same cell observe each other's in-place mutations
///     (`Rc::ptr_eq` ≡ Python `is`).  Holding a handle keeps the cell
///     alive even if the terminal operand later swaps its
///     `_forwarded` slot to a different info — mirroring Python local
///     variables that keep a previously read `fw` alive.
///
/// Read access:
///   - `handle.borrow()` → `PtrInfoHandleRef<'_>` which `Deref`s into
///     `&PtrInfo` for ergonomic method calls.  The `Live` arm holds
///     a `Ref` on the underlying `RefCell`; caller must drop the
///     guard before any sibling `borrow_mut` on the same cell.
///   - `handle.borrow_mut()` → `Option<RefMut<'_, PtrInfo>>` for the
///     `Live` arm; `None` for `Const` (mutating a freshly minted
///     `ConstPtrInfo` snapshot would not propagate).
///   - `handle.same_info(&other)` → RPython `same_info` parity:
///     non-constant live infos compare by object identity, while
///     ConstPtrInfo compares the wrapped constant value.
pub enum PtrInfoHandle {
    Const(PtrInfo),
    Live(std::rc::Rc<std::cell::RefCell<PtrInfo>>),
}

impl PtrInfoHandle {
    /// Wrap a freshly synthesized `ConstPtrInfo` (info.py /
    /// 888-889 return path).
    pub fn const_(info: PtrInfo) -> Self {
        PtrInfoHandle::Const(info)
    }

    /// Wrap a live `_forwarded` cell handle (info.py:875-877 /
    /// 890-893 return path).
    pub fn live(rc: std::rc::Rc<std::cell::RefCell<PtrInfo>>) -> Self {
        PtrInfoHandle::Live(rc)
    }

    /// RPython `PtrInfo.same_info(other)` parity.
    ///
    /// Base `PtrInfo.same_info` is object identity (`self is other`,
    /// info.py), so non-constant live infos must share the same
    /// `_forwarded` cell. `ConstPtrInfo` overrides this and compares
    /// the wrapped constant value (`_const.same_constant`, info.py),
    /// so two independently synthesized ConstPtrInfo handles for the
    /// same pointer are `same_info`.
    pub fn same_info(&self, other: &PtrInfoHandle) -> bool {
        if let (PtrInfoHandle::Live(a), PtrInfoHandle::Live(b)) = (self, other)
            && std::rc::Rc::ptr_eq(a, b)
        {
            return true;
        }

        fn constptr_same_info(a: &PtrInfo, b: &PtrInfo) -> bool {
            match (a, b) {
                (PtrInfo::Constant(left), PtrInfo::Constant(right)) => left == right,
                _ => false,
            }
        }

        match (self, other) {
            (PtrInfoHandle::Const(a), PtrInfoHandle::Const(b)) => constptr_same_info(a, b),
            (PtrInfoHandle::Const(a), PtrInfoHandle::Live(b)) => {
                let b = b.borrow();
                constptr_same_info(a, &b)
            }
            (PtrInfoHandle::Live(a), PtrInfoHandle::Const(b)) => {
                let a = a.borrow();
                constptr_same_info(&a, b)
            }
            (PtrInfoHandle::Live(a), PtrInfoHandle::Live(b)) => {
                let a = a.borrow();
                let b = b.borrow();
                constptr_same_info(&a, &b)
            }
        }
    }

    /// Read access — yields a guard that `Deref`s into `&PtrInfo`.
    pub fn borrow(&self) -> PtrInfoHandleRef<'_> {
        match self {
            PtrInfoHandle::Const(info) => PtrInfoHandleRef::Const(info),
            PtrInfoHandle::Live(rc) => PtrInfoHandleRef::Live(rc.borrow()),
        }
    }

    /// Mutable access — `Some(RefMut<'_, PtrInfo>)` for `Live`,
    /// `None` for `Const`.
    pub fn borrow_mut(&self) -> Option<std::cell::RefMut<'_, PtrInfo>> {
        match self {
            PtrInfoHandle::Const(_) => None,
            PtrInfoHandle::Live(rc) => Some(rc.borrow_mut()),
        }
    }

    /// Convert to an owned `PtrInfo` snapshot.  Clones for `Live`;
    /// destructures for `Const`.
    pub fn into_ptr_info(self) -> PtrInfo {
        match self {
            PtrInfoHandle::Const(info) => info,
            PtrInfoHandle::Live(rc) => rc.borrow().clone(),
        }
    }

    /// Cheap clone-as-snapshot for read-only callsites that only need
    /// an owned `PtrInfo` and don't care about identity.
    pub fn snapshot(&self) -> PtrInfo {
        match self {
            PtrInfoHandle::Const(info) => info.clone(),
            PtrInfoHandle::Live(rc) => rc.borrow().clone(),
        }
    }

    /// `info.py` `ConstPtrInfo._const.getref_base()` read through the
    /// handle, without copying the info.
    pub fn constant_ref(&self) -> Option<GcRef> {
        match self {
            PtrInfoHandle::Const(PtrInfo::Constant(g)) => Some(*g),
            PtrInfoHandle::Live(rc) => match &*rc.borrow() {
                PtrInfo::Constant(g) => Some(*g),
                _ => None,
            },
            _ => None,
        }
    }
}

impl std::fmt::Debug for PtrInfoHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PtrInfoHandle::Const(info) => {
                f.debug_tuple("PtrInfoHandle::Const").field(info).finish()
            }
            PtrInfoHandle::Live(rc) => f
                .debug_tuple("PtrInfoHandle::Live")
                .field(&*rc.borrow())
                .finish(),
        }
    }
}

/// Borrow guard returned by `PtrInfoHandle::borrow()`.
/// `Deref<Target = PtrInfo>` lets callers call any `PtrInfo` method
/// uniformly without first matching on the variant.
pub enum PtrInfoHandleRef<'a> {
    Const(&'a PtrInfo),
    Live(std::cell::Ref<'a, PtrInfo>),
}

impl std::ops::Deref for PtrInfoHandleRef<'_> {
    type Target = PtrInfo;
    fn deref(&self) -> &PtrInfo {
        match self {
            PtrInfoHandleRef::Const(info) => info,
            PtrInfoHandleRef::Live(r) => r,
        }
    }
}

/// IntBound counterpart to [`PtrInfoHandle`].
///
/// `optimizer.py getintbound(op)` returns the live `IntBound`
/// object stored on `box._forwarded`; downstream code calling
/// `getintbound(box).intersect(b)` mutates that same object so any
/// other holder observes the change.  In pyre, `OpInfo::IntBound`
/// carries `Rc<RefCell<IntBound>>` (Phase 1A), so sharing the cell
/// between two operands reproduces the RPython object-identity
/// behaviour.  This handle is the public API for that identity:
///
///   - `Const(IntBound)` — a freshly synthesized `IntBound` from a
///     `ConstInt` (`optimizer.py from_constant`).  Two
///     `Const` handles never compare equal under `ptr_eq` even when
///     they wrap the same value.
///   - `Live(IntBoundRc)` — the actual `_forwarded` cell.
///     `IntBoundRc::ptr_eq` ≡ Python `is`.  In-place mutation via
///     `handle.borrow_mut()` propagates to every other live handle
///     cloned from the same cell.
pub enum IntBoundHandle {
    /// Freshly synthesized `IntBound::from_constant(_)` object
    /// (`optimizer.py` `from_constant` return path). Each call mints
    /// a distinct slab cell (two `from_constant(7)` calls return two
    /// distinct objects), so mutations do not propagate across calls.
    Const(majit_ir::intbound::IntBoundRc),
    /// Live `_forwarded` cell — mutations propagate to every handle
    /// cloned from the same slab slot and through `box._forwarded`.
    Live(majit_ir::intbound::IntBoundRc),
}

impl IntBoundHandle {
    /// Wrap a freshly synthesized `IntBound::from_constant(_)`
    /// (`optimizer.py` `from_constant` return path). Mints a fresh
    /// slab cell so each call produces a distinct object.
    pub fn const_(b: crate::optimizeopt::intutils::IntBound) -> Self {
        IntBoundHandle::Const(majit_ir::intbound::IntBoundRc::new(b))
    }

    /// Wrap a live `_forwarded` cell handle (`optimizer.py`
    /// `getintbound` return path).
    pub fn live(rc: majit_ir::intbound::IntBoundRc) -> Self {
        IntBoundHandle::Live(rc)
    }

    /// Identity comparison. Two handles are `ptr_eq` iff they hold
    /// the same slab slot — Python `is` parity. Const/Live cross-arm
    /// pairs are never equal because they live in disjoint namespaces.
    pub fn ptr_eq(&self, other: &IntBoundHandle) -> bool {
        match (self, other) {
            (IntBoundHandle::Const(a), IntBoundHandle::Const(b))
            | (IntBoundHandle::Live(a), IntBoundHandle::Live(b)) => {
                majit_ir::intbound::IntBoundRc::ptr_eq(a, b)
            }
            _ => false,
        }
    }

    /// Read access — yields a guard that `Deref`s into `&IntBound`.
    pub fn borrow(&self) -> IntBoundHandleRef<'_> {
        match self {
            IntBoundHandle::Const(rc) => IntBoundHandleRef::Const(rc.borrow()),
            IntBoundHandle::Live(rc) => IntBoundHandleRef::Live(rc.borrow()),
        }
    }

    /// Mutable access for both arms. PyPy `getintbound(ConstInt)`
    /// returns a mutable `IntBound.from_constant(...)` object whose
    /// mutations are private to that object (no propagation back to
    /// the box). The `Const` arm mirrors that: borrow_mut yields a
    /// RefMut into the fresh per-call cell.
    pub fn borrow_mut(
        &self,
    ) -> Option<std::cell::RefMut<'_, crate::optimizeopt::intutils::IntBound>> {
        match self {
            IntBoundHandle::Const(rc) | IntBoundHandle::Live(rc) => Some(rc.borrow_mut()),
        }
    }

    /// Convert to an owned `IntBound` snapshot.
    pub fn into_int_bound(self) -> crate::optimizeopt::intutils::IntBound {
        match self {
            IntBoundHandle::Const(rc) | IntBoundHandle::Live(rc) => rc.borrow().clone(),
        }
    }

    /// Cheap clone-as-snapshot for read-only callsites that only need
    /// an owned `IntBound` and don't care about identity.
    pub fn snapshot(&self) -> crate::optimizeopt::intutils::IntBound {
        match self {
            IntBoundHandle::Const(rc) | IntBoundHandle::Live(rc) => rc.borrow().clone(),
        }
    }
}

impl std::fmt::Debug for IntBoundHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntBoundHandle::Const(rc) => f
                .debug_tuple("IntBoundHandle::Const")
                .field(&*rc.borrow())
                .finish(),
            IntBoundHandle::Live(rc) => f
                .debug_tuple("IntBoundHandle::Live")
                .field(&*rc.borrow())
                .finish(),
        }
    }
}

/// Borrow guard returned by `IntBoundHandle::borrow()`.
/// `Deref<Target = IntBound>` lets callers call any `IntBound`
/// method uniformly without first matching on the variant.
pub enum IntBoundHandleRef<'a> {
    Const(std::cell::Ref<'a, crate::optimizeopt::intutils::IntBound>),
    Live(std::cell::Ref<'a, crate::optimizeopt::intutils::IntBound>),
}

impl std::ops::Deref for IntBoundHandleRef<'_> {
    type Target = crate::optimizeopt::intutils::IntBound;
    fn deref(&self) -> &crate::optimizeopt::intutils::IntBound {
        match self {
            IntBoundHandleRef::Const(r) | IntBoundHandleRef::Live(r) => r,
        }
    }
}

/// info.py INFO_NULL / INFO_NONNULL / INFO_UNKNOWN constants.
///
/// Re-exported from `majit_ir::optimize` so the bound / info types
/// that reference them can be hosted there without a circular dep.
pub use majit_ir::optimize::{INFO_NONNULL, INFO_NULL, INFO_UNKNOWN};

/// Create a ResumeAtPositionDescr for optimizer-generated guards.
///
/// Delegates to compile::make_resume_at_position_descr which wraps a
/// real ResumeGuardDescr — clone_descr() preserves resume data (RPython
/// ResumeAtPositionDescr is a plain subclass of ResumeGuardDescr).
pub fn make_resume_at_position_descr() -> DescrRef {
    crate::compile::make_resume_at_position_descr()
}

/// optimizer.py OptimizationResult: deferred postprocess for GUARD_CLASS/GUARD_NONNULL_CLASS.
/// RPython's postprocess_GUARD_CLASS runs after the guard is emitted to
/// _newoperations. In majit, recorded here by rewrite and executed by
/// emit_operation.
#[derive(Debug)]
pub struct PendingGuardClassPostprocess {
    pub obj: majit_ir::OpRef,
    pub class_val: i64,
}

#[derive(Clone, Debug)]
pub struct ImportedShortPureOp {
    pub opcode: OpCode,
    pub descr: Option<DescrRef>,
    /// Exporting-phase boxes of the pure op (shortpreamble.py
    /// `PureOp.produce_op`: `self.res`, or `self.orig_op` when invented).
    pub args: Vec<majit_ir::operand::Operand>,
    pub result: OpRef,
    /// RPython: PreambleOp stored in pure cache. Used by force_op_from_preamble.
    pub pop: crate::optimizeopt::info::PreambleOp,
    /// `pure.py` `RecentPureOps.add` stores the op, and `lookup2` calls
    /// `get_box_replacement` on its args at lookup time. These are those
    /// argument boxes. Rematerializing them from an `OpRef` in the peeled
    /// context drops the forwarding a later `guard_value` wrote.
    pub cache_args: Vec<majit_ir::operand::Operand>,
}

impl ImportedShortPureOp {
    /// Construct with auto-generated PreambleOp from fields.
    ///
    /// `args` are the exporting phase's boxes. shortpreamble.py
    /// `PureOp.produce_op` hands that op to the pure cache; the replay op
    /// is built from those Operands directly.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub fn new(
        ctx: &mut OptContext,
        opcode: OpCode,
        descr: Option<DescrRef>,
        args: Vec<majit_ir::operand::Operand>,
        result: OpRef,
        source: OpRef,
        invented_name: bool,
        same_as_source: Option<majit_ir::operand::Operand>,
    ) -> Self {
        let mut replay = majit_ir::Op::new(opcode, &args);
        // shortpreamble.py PureOp.produce_op constructs TWO distinct
        // RPython Op objects:
        //
        //   * `op` — the alt identifier. For invented_name it is
        //     `self.orig_op.copy_and_change(...).set_forwarded(self.res)`,
        //     a freshly-allocated Op with its own `_forwarded` slot.
        //   * `preamble_op` — the replay Op passed in from `add_op_to_short`.
        //     Its `_forwarded` slot is seeded with the exported info by
        //     `ShortPreambleBuilder.__init__` (shortpreamble.py).
        //
        // The two `_forwarded` slots live on two different RPython Op
        // objects, so `op.set_forwarded(self.res)` and
        // `preamble_op.set_forwarded(info)` never collide.
        //
        // The body-visible `op` and the replay `preamble_op` are distinct
        // RPython Box objects for every PureOp, not only invented aliases.
        // Keep their OpRef identities distinct as well: `source` names
        // self.res, while `result` names the replay operation — including
        // the non-invented arm, where `add_op_to_short` (shortpreamble.py
        // `op.copy_and_change(opnum, args=arglist)`) still hands
        // `ProducedShortOp` a freshly allocated replay op rather than
        // `self.res` itself.
        replay.pos().set(result);
        if let Some(d) = descr.clone() {
            replay.setdescr(d);
        }
        // shortpreamble.py:116-120: pop.op = self.orig_op.copy_and_change(...)
        // for invented (the alt identifier) or self.res for non-invented.
        // pyre's `source` IS the alt identifier for invented (the synthetic
        // alias allocated by the compound-dedup pass at
        // `ShortBoxes` in `shortpreamble.rs`) and IS self.res for non-invented.
        let pop_op = ctx.materialize_operand_at(source);
        ImportedShortPureOp {
            opcode,
            descr,
            args,
            result,
            cache_args: Vec::new(),
            pop: crate::optimizeopt::info::PreambleOp {
                op: pop_op,
                invented_name,
                preamble_op: OpRc::new(replay),
                same_as_source,
            },
        }
    }
}

impl PartialEq for ImportedShortPureOp {
    fn eq(&self, other: &Self) -> bool {
        self.opcode == other.opcode
            && self.descr.as_ref().map(|d| d.index()) == other.descr.as_ref().map(|d| d.index())
            && self.args == other.args
            && self.result == other.result
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedShortAlias {
    pub result: OpRef,
    /// The canonical operand of the SameAs source, carried directly off
    /// `extra_same_as` (already an `Operand` op-arg) instead of round-tripping
    /// through a positional box.
    pub same_as_source: majit_ir::operand::Operand,
    pub same_as_opcode: OpCode,
}

/// optimizer.py: constant_fold — allocate an immutable object at
/// compile time when all fields are constants. The callback receives the
/// SizeDescr size_bytes, and returns a raw pointer (GcRef) to freshly
/// allocated memory. The optimizer writes field values directly.
pub type ConstantFoldAllocFn = Box<dyn Fn(usize) -> majit_ir::GcRef>;

/// Re-export of `info::StringLengthResolver` for callers that import
/// `optimizeopt::StringLengthResolver`. The runtime hook signature is
/// `Arc<dyn Fn(GcRef, u8) -> Option<i64> + Send + Sync>`. See
/// `info::EnsuredPtrInfo::getlenbound` for the consumer side.
pub use crate::optimizeopt::info::StringLengthResolver;
pub use crate::optimizeopt::info::{StringConstantAllocator, StringContentResolver};

use crate::optimizeopt::info::PtrInfoExt;

/// One line naming an operand for an assertion message: its producing opcode
/// and, for a field access, the descr's field name. The operand's own `Debug`
/// walks the whole abstract-value graph, which is unusable in a panic message.
fn describe_operand_for_assert(b: &Operand) -> String {
    let Some(op) = b.bound_op() else {
        return format!("{:?}", b.to_opref());
    };
    let name = op
        .getdescr()
        .and_then(|d| d.as_field_descr().map(|fd| fd.field_name().to_string()))
        .unwrap_or_default();
    if name.is_empty() {
        format!("{:?} {:?}", op.opcode, op.pos().get())
    } else {
        format!("{:?} {:?} `{name}`", op.opcode, op.pos().get())
    }
}

/// The one short-preamble evaluation the preview pass performs, carried from
/// `optimizer.rs` (where the `ShortBoxes` object lives) to
/// `unroll.rs::export_state_with_bounds`.
///
/// unroll.py:466-480 computes `virtual_state`, `label_args`, `virtuals`,
/// `short_boxes` and `short_inputargs` in one straight line. majit splits that
/// line across the preview pass and the export site, so the preview's single
/// evaluation travels here: producing the potential short-preamble ops mutates
/// Box forwarding, and a second, Rust-only recomputation at the export site is
/// observably different.
///
/// The whole struct is published at one site, so the vectors below are always
/// built together from one evaluation of the preview.  That makes them mutually
/// consistent — it does NOT make all three index-aligned: only
/// `short_inputargs` and `short_inputarg_refs` share an index space (one entry
/// per `add_short_input_arg`).  `short_boxes` is a different population
/// entirely — the surviving *produced* short ops, after the export site drops
/// every one whose result folded to a `Const` — so its length is unrelated and
/// `short_boxes[i]` pairs with nothing.
pub struct PreviewShortState {
    /// unroll.py:480 `short_inputargs = sb.create_short_inputargs(label_args
    /// + virtuals)` — the ShortBoxes-stored renamed inputarg positions
    /// themselves (measured identical to the export-site `label_args +
    /// virtuals` recompute across the corpus, 2026-06-11).
    pub short_inputargs: Vec<OpRef>,
    /// Rooted `InputArgRc` carriers for `short_inputargs`, index-aligned with
    /// that vector. Keeping the strong Rc alive here preserves the renamed
    /// inputarg producer across the export boundary; consumers that need an
    /// operand bind the stored position directly.
    pub short_inputarg_refs: Vec<majit_ir::InputArgRc>,
    /// RPython shortpreamble.py: pass-collected preamble producers aligned to
    /// the exported loop-header inputargs.
    pub short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp>,
    /// The virtual-state / inputarg expansion the preview built ShortBoxes
    /// from. Absent on the bridge arm that breaks out of the export before the
    /// VS capture; present whenever the preview ran to completion.
    pub args_state: Option<(
        crate::optimizeopt::virtualstate::VirtualState,
        Vec<OpRef>,
        Vec<OpRef>,
        Vec<usize>,
        Vec<majit_ir::operand::Operand>,
    )>,
}

/// Context provided to optimization passes.
///
/// Holds the shared state that passes read from and write to.
pub struct OptContext {
    /// Backend cost capability for RPython's constant-division rewrite.
    pub supports_efficient_uint_mul_high: bool,
    /// The output operation list being built.
    pub new_operations: Vec<majit_ir::OpRc>,
    /// Once the output buffer is drained, PtrInfo guard positions stamped
    /// against the old buffer no longer name operations in this context.
    new_operations_drained: bool,
    /// O(1) `OpRef → OpRc` index mirroring `new_operations`, keyed by each
    /// op's result position. `find_producer_op`'s highest-priority lookup
    /// otherwise scans `new_operations` with `iter().rfind(...)`, which is
    /// O(n) per call and O(n²) over a whole trace — the dominant cost when
    /// optimizing a large loop — a whole-program trace can reach tens of
    /// thousands of ops.
    /// Maintained incrementally by `push_new_operation` (insert overwrites,
    /// so the last push at a position wins, matching the `rfind`
    /// last-occurrence semantics); rebuilt by `rebuild_new_operations_index`
    /// after the rare structural mutations, and cleared with the context.
    pub(crate) new_operations_index: FxHashMap<OpRef, majit_ir::OpRc>,
    /// optimizer.py `self._emittedoperations = {}` — the result boxes
    /// of ops emitted by THIS optimizer run, keyed by box identity
    /// (`Rc::ptr_eq`) to mirror the upstream dict keyed by the `op` object.
    /// `as_operation` (optimizer.py) only treats a box as a producer
    /// when `op in self._emittedoperations`, so pattern-matching rewrites
    /// (e.g. autogenintrules.py int_add chain reassociation) never reach
    /// across an optimization-run boundary into a previous phase's ops.
    /// A Phase-2 lookup that followed a label arg to its Phase-1 producer
    /// would re-express a per-iteration value in terms of the PREAMBLE's
    /// entry value, which the loop header does not carry per-iteration.
    pub emitted_operations:
        indexmap::IndexSet<majit_ir::operand::Operand, rustc_hash::FxBuildHasher>,
    /// Number of input arguments, used to offset emitted op positions
    /// so that variable indices don't collide with input arg indices.
    num_inputs: u32,
    /// opencoder.py:259-267 inputarg base in the OpRef namespace.
    ///
    /// RPython lets each TraceIterator allocate fresh inputarg boxes whose
    /// Python identity (`is`) distinguishes them from any other phase's
    /// boxes; majit needs to encode that as a numeric offset because OpRef
    /// IS the identity. Phase 1 uses `inputarg_base = 0` (legacy positional
    /// layout); Phase 2/bridges shift inputarg OpRefs above the parent
    /// trace's high water mark by setting `inputarg_base = parent_high_water`.
    /// `reserve_pos` floors `next_pos` at `inputarg_base + num_inputs +
    /// new_operations.len()` so freshly emitted ops never collide with
    /// inputargs or imported high-water marks.
    pub(crate) inputarg_base: u32,
    /// Next unique op position for newly emitted or queued extra operations.
    pub(crate) next_pos: u32,
    /// RPython emit_extra(op, emit=False) parity: ops queued to be
    /// processed starting from a specific pass index (skipping earlier passes).
    /// Used by heap's force_lazy_set to route ops through remaining passes
    /// without re-entering the heap pass itself.
    /// Held as `OpRc` (resoperation.py: emit_extra appends a ResOperation
    /// object) so the queued op carries object identity into the drain.
    pub(crate) extra_operations_after: ExtraQueue,
    /// The queues `Optimizer::drain_extra_operations_from` is working through,
    /// innermost last. The drain moves `extra_operations_after` aside before it
    /// starts so that a nested drain only sees what was queued after it began —
    /// that is what keeps `emit_extra`'d ops ahead of the op being propagated
    /// (`optimizer.py`) instead of ahead of everything still parked.
    ///
    /// Parked here rather than in a local so `flush_queued_producer` can still
    /// reach one entry: `info.py force_box` clears the virtual flag and
    /// emits the allocation as one step, while pyre's `emit_op` only queues the
    /// allocation when the force runs from a pass, so a store can be emitted
    /// while the `NEW_WITH_VTABLE` that defines it is still parked.
    pub(crate) extra_pending: Vec<ExtraQueue>,
    /// optimizer.py OptimizationResult: deferred postprocess for GUARD_CLASS.
    /// Set by rewrite pass, executed by emit_operation after the guard
    /// is added to new_operations (matching RPython's callback pattern).
    pub(crate) pending_guard_class_postprocess: Option<PendingGuardClassPostprocess>,
    /// virtualize.py postprocess_FINISH queues the stashed
    /// GUARD_NOT_FORCED_2 here so the outer optimizer can insert it at
    /// `new_operations.len() - 1` with full `store_final_boxes_in_guard`
    /// semantics — the pass that stashes it holds no Optimizer, and both the
    /// finalization and the knowledge collection are Optimizer-side.
    pub(crate) pending_finish_guard_postprocess: Option<Op>,
    /// rewrite.py: postprocess_GUARD_NONNULL → mark_last_guard.
    /// Deferred until emit adds the guard to new_operations.
    pub(crate) pending_mark_last_guard: Option<OpRef>,
    // ptr_info merged into forwarded (Forwarded::Info variant)
    //
    // RPython parity: per-OpRef IntBound storage lives ENTIRELY on
    // `box._forwarded` (Forwarded::IntBound), accessed via getintbound /
    // setintbound. The previous `int_lower_bounds` (heap.py array length
    // hint), `int_bounds` (per-pass snapshot), and `imported_int_bounds`
    // (preamble import) maps were a majit-only divergence from RPython's
    // single source of truth. They've been merged into Forwarded::IntBound
    // at write time so reads naturally see all sources via getintbound.
    /// RPython shortpreamble.py / pure.py: imported pure-operation results from
    /// the preamble. Phase 2 uses these as cross-iteration CSE facts.
    pub imported_short_pure_ops: Vec<ImportedShortPureOp>,
    /// (base_len, short_args): virtual field values start at base_len
    /// within short_args. Used by install_imported_virtuals.
    pub imported_virtual_args: Option<(usize, Vec<OpRef>)>,
    /// `rewrite.py` `self.loop_invariant_results = {}` — keyed by
    /// constant function pointer. PyPy uses a dict; pyre replaces it
    /// with a Vec of `(func_ptr, source_opref)` pairs and linear-scan
    /// dedup. CALL_LOOPINVARIANT is rare and the live set per trace is
    /// tiny, so O(n) lookup is acceptable.
    pub imported_loop_invariant_results: Vec<(i64, OpRef)>,
    /// Phase 2 imported virtuals (from Phase 1 export). Used by
    /// store_final_boxes_in_guard to resolve NONE positions
    /// inherited from Phase 1 virtualization.
    pub(crate) imported_virtuals: Vec<crate::optimizeopt::optimizer::ImportedVirtual>,
    /// Phase 2 imported label args (OpRefs in Phase 2 namespace).
    pub imported_label_args: Option<Vec<OpRef>>,
    /// RPython shortpreamble.py: active phase-2 short preamble builder.
    /// Tracks which imported short facts are actually consumed by the body.
    pub imported_short_preamble_builder:
        Option<crate::optimizeopt::shortpreamble::ShortPreambleBuilder>,
    /// `optimizer.py` `self.quasi_immutable_deps = None` (initialized
    /// lazily as a dict in `heap.py:821-823`). Each entry is one `QuasiImmut`
    /// instance the trace folded a field of, exactly as upstream keys the dict
    /// (`heap.py OptHeap.optimize_QUASIIMMUT_FIELD quasi_immutable_deps[qmutdescr.qmut] = None`); PyPy
    /// uses `dict[k] = None` for set semantics, but the HashMap house rule
    /// forbids that — pyre uses a Vec with linear-scan dedup on instance
    /// identity. Typical size is small (< a few dozen entries per trace), so
    /// O(n) inserts are acceptable.
    pub quasi_immutable_deps: Vec<std::sync::Arc<dyn majit_ir::QuasiImmutHandle>>,
    /// `info.py:722` `optheap.const_infos.get(ref, None)` /
    /// `info.py:725` `optheap.const_infos[ref] = info`. Stores
    /// `StructPtrInfo` / `ArrayPtrInfo` for constant GC objects keyed
    /// by pointer address. PyPy uses `new_ref_dict()`; the house rule
    /// forbids hash containers, so pyre uses a Vec-backed associative
    /// container with linear-scan lookup.
    pub const_infos: crate::FxIndexMap<usize, crate::optimizeopt::info::PtrInfo>,
    /// `unroll.py` `self.optunroll.potential_extra_ops[op] = preamble_op` /
    /// `optimizer.py` `preamble_op = self.optunroll.potential_extra_ops.pop(op)`.
    /// A keyed map, like the dict upstream keeps: insert, pop and `in` are
    /// each one lookup, where the `Vec<(OpRef, PreambleOp)>` this replaced
    /// scanned the whole pool for all three.
    ///
    /// `swap_remove` is the removal the `Vec` already performed, and nothing
    /// iterates this map — every use is a keyed insert, pop or membership
    /// test — so the order it leaves behind is unobservable. `IndexMap`
    /// rather than `HashMap` because iteration order stays deterministic
    /// even if a later reader does walk it.
    ///
    /// The key stays the box's `OpRef` position rather than PyPy's box
    /// identity (`unroll.py:37` keys by the box object). This is a
    /// PRE-EXISTING-ADAPTATION beyond the Vec-vs-dict shape: the insert key
    /// is the Phase-1 preamble source box (`force_op_from_preamble_op`,
    /// `preamble_op.op`) while the `force_box` pop key is the Phase-2 body
    /// box resolved to a position (`get_replacement_opref`) — two distinct
    /// `Rc`s sharing one position across the peel boundary. An operand
    /// `Rc::ptr_eq` key would silent-miss the pop. Re-keying to box identity
    /// is gated on the same short-preamble / InputArg identity unification
    /// that defers `resolve_operand_operand`'s InputArg arm (#9).
    pub(crate) potential_extra_ops:
        indexmap::IndexMap<Operand, crate::optimizeopt::info::PreambleOp>,
    /// RPython unroll.py: live ExtendedShortPreambleBuilder while replaying an
    /// existing target token's short preamble.
    active_short_preamble_producer:
        Option<crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder>,
    /// The preview pass's short-preamble publication, produced in one group
    /// at the end of `optimize_with_constants_and_inputs_at` and read once by
    /// `export_state_with_bounds`. `None` means no preview ran, so the export
    /// site recomputes everything locally.
    pub preview_short_state: Option<PreviewShortState>,
    /// shortpreamble.py:251-253: a way to produce const boxes, e.g.
    /// `setfield_gc(p0, Const)`. We need to remember those, but they don't
    /// produce any new boxes. This list is consumed only by import-side
    /// `produce_op` replay, never by `used_boxes`: unroll.py never
    /// registers a const PreambleOp in `potential_extra_ops`. It carries no
    /// index alignment to the `PreviewShortState` vectors, so it stays on its
    /// own channel.
    pub exported_const_short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp>,
    /// optimizer.py: `can_replace_guards` — disable guard replacement during
    /// bridge compilation. Defaults to true for preamble.
    can_replace_guards: bool,
    /// RPython optimizer.py: `patchguardop` — the last GUARD_FUTURE_CONDITION op.
    /// Used by unroll to attach resume data to extra guards from short preamble.
    pub patchguardop: Option<Op>,
    /// RPython optimizer.py: end_args after force_at_the_end_of_preamble().
    /// export_state() prefers this over a raw get_replacement() snapshot.
    pub preamble_end_args: Option<Vec<OpRef>>,
    /// Phase-2 loop-body mode from optimizer.skip_flush.
    /// RPython unroll.py relies on this distinction so virtualize can keep
    /// body-side allocations concrete when guard recovery cannot rebuild them.
    pub skip_flush_mode: bool,
    /// Bridge mode from `optimizer.building_bridge` (unroll.py:183-236
    /// `optimize_bridge`).
    ///
    /// A bridge's inputargs are rebuilt from the deadframe frame-first, so the
    /// virtualizable identity sits at slot 0 whatever the front end's own
    /// layout says. `inputarg_base` cannot carry that distinction: Phase 2
    /// shifts the base as well (`unroll.rs` `phase2_inputarg_base`), so a
    /// consumer keying on `inputarg_base != 0` treats an unrolled loop body as
    /// a bridge. This flag is the discriminator; the base is the namespace.
    pub building_bridge: bool,
    /// Index of the pass currently executing propagate_forward.
    /// Used by passes to call send_extra_operation_after(self_idx, ..)
    /// matching RPython's emit_extra(op, emit=False) which routes to
    /// self.next_optimization.
    pub current_pass_idx: usize,
    /// earlyforce.py:32: self.optimizer.optearlyforce = self
    /// Index of the OptEarlyForce pass in the pass chain.
    /// Used by force_at_the_end_of_preamble and force_box to route
    /// forced operations starting from earlyforce.next (= heap).
    pub optearlyforce_idx: usize,
    /// optimizer.py: pendingfields — deferred SetfieldGc/SetarrayitemGc ops
    /// where the stored value is virtual. Set by OptHeap.emitting_operation()
    /// before a guard, consumed by emit_operation() to encode into
    /// the guard's rd_pendingfields.
    pub pending_for_guard: Vec<Op>,
    /// optimizer.py: pure_from_args1 parity — reverse-pure relationships
    /// registered by rewrite pass (CAST_*, CONVERT_*) and consumed by pure pass.
    /// Each entry: (opcode, key_arg, forwarded, descr). The register helpers
    /// push `(opcode, key, forwarded, descr)` so OptPure's drain can call
    /// `pure_from_args1(opcode, key, forwarded)`. `descr` is `None` for the
    /// common case; `Some(DescrRef)` matches
    /// `pure_from_args(opnum, [arg], result, descr=op.getdescr())`.
    pub pending_pure_from_args: Vec<(OpCode, Operand, Operand, Option<majit_ir::DescrRef>)>,
    /// optimizer.py: pure_from_args2 parity — binary reverse-pure relationships
    /// registered by rewrite pass (INSTANCE_PTR_EQ/NE swapped-args). Consumed
    /// by OptPure. Each entry: (opcode, arg0, arg1, forwarded) meaning
    /// pure(opcode, arg0, arg1) forwards to `forwarded`.
    pub pending_pure_from_args2: Vec<(OpCode, Operand, Operand, Operand)>,
    /// Live recent-ops ring. `OptPure` owns the allocation and publishes
    /// the handle; `OptRewrite.find_rewritable_bool` reads it the way
    /// `Optimization.get_pure_result` reads `optimizer.optpure`. Cleared
    /// by `OptPure.setup` (each optimize run), not by a label op.
    pub(crate) pure_ops: Option<pure::SharedPureOps>,
    /// optimizer.py: constant_fold allocator callback.
    /// When set, the optimizer can fold immutable virtuals filled with
    /// constants into compile-time constant pointers (info.py:140-145).
    pub constant_fold_alloc: Option<ConstantFoldAllocFn>,
    /// info.py `ConstPtrInfo.getstrlen1(mode)` — runtime hook for
    /// constant byte-string / unicode-string length lookup. Set by the
    /// host runtime (pyre etc.) at OptContext construction time. When
    /// `None`, `EnsuredPtrInfo::getlenbound(Some(_))` falls back to
    /// `IntBound::nonnegative()`.
    pub string_length_resolver: Option<StringLengthResolver>,
    /// info.py `ConstPtrInfo._unpack_str(mode)` — runtime hook for
    /// extracting character data from a constant string GcRef.
    pub string_content_resolver: Option<StringContentResolver>,
    /// history.py `get_const_ptr_for_string(s)` — runtime hook for
    /// creating a constant string GcRef from char values (used by
    /// force_box constant-folding path, vstring.py).
    pub string_constant_alloc: Option<StringConstantAllocator>,
    /// True while optimizer.py:_emit_operation equivalent is forcing args
    /// just before final emission. In this phase, virtual forcing must emit
    /// directly into new_operations instead of re-entering the pass chain.
    pub in_final_emission: bool,
    /// `info.py` `_force_elements` calls `optforce.optimizer.force_box(fld)`.
    /// `Optimizer.force_box` defaults `optforce` to the `Optimizer`, so the
    /// child allocation and its stores use `Optimizer.emit_extra` (direct
    /// append) even when the parent used `Optimization.emit_extra`. Set around
    /// that `force_box` and restored after. `emit_for_force` reads it for
    /// STRLEN and the string stores. `_int_add` does not: it always
    /// `optimizer.send_extra_operation` from `first_optimization`.
    pub force_direct: bool,
    /// effectinfo.py: CallInfoCollection — maps oopspec indices to
    /// (calldescr, func_ptr) pairs. Used by generate_modified_call
    /// (vstring.py) to emit specialized string comparison calls.
    pub callinfocollection: Option<std::sync::Arc<majit_ir::CallInfoCollection>>,
    /// optimizer.py is_call_pure_pure_canraise `self.resumedata_memo`.
    ///
    /// RPython allocates one ResumeDataLoopMemo in `Optimizer.__init__` and
    /// every guard emitted by that optimizer numbers through the same object.
    /// `Rc<RefCell<_>>` preserves that shared object identity while allowing
    /// `OptContext` (the Rust borrow-splitting adaptation of Optimizer) to own
    /// the guard-emission path.
    pub resumedata_memo: std::rc::Rc<std::cell::RefCell<crate::resume::ResumeDataLoopMemo>>,
    /// resume.py parity: per-guard snapshot boxes from tracing time.
    /// Used by emit() to call store_final_boxes_in_guard inline (RPython
    /// calls this during optimization, not post-assembly).
    pub snapshot_boxes: SnapshotBoxes,
    /// Per-frame box counts for multi-frame snapshots.
    /// opencoder.py capture_resumedata encodes multiple frames;
    /// this tracks the boundary between callee and caller sections.
    pub snapshot_frame_sizes: SnapshotFrameSizes,
    /// Per-guard virtualizable boxes from tracing-time snapshots.
    pub snapshot_vable_boxes: SnapshotBoxes,
    /// resume.py:399-402 `minimum_virtualizable_size`, handed to
    /// `memo.number()` in `store_final_boxes_in_guard`. `-1` disables the
    /// `resume.py` length check; a non-negative value arms it.
    /// Copied from `Optimizer::minimum_virtualizable_size`, which
    /// `default_pipeline_with_virtualizable` derives from the vable config.
    pub minimum_virtualizable_size: i64,
    /// Per-guard virtualref boxes from tracing-time snapshots.
    /// resume.py _number_boxes consumes vref_array as a section
    /// after vable_array. opencoder.py create_top_snapshot records vref_boxes here.
    pub snapshot_vref_boxes: SnapshotBoxes,
    /// Per-guard per-frame (jitcode_index, pc) from tracing-time snapshots.
    pub snapshot_frame_pcs: SnapshotFramePcs,
    /// Byte-mode bridge resume. See `Optimizer::byte_bridge_resume`.
    pub(crate) byte_bridge_resume: Option<crate::recorder::ByteBridgeResume>,
    /// optimizer.py `self.inputargs = inputargs` parity.
    /// Typed InputArg OpRefs; slot `i` is `OpRef::input_arg_typed(i, tp)`.
    pub inputargs: Vec<majit_ir::OpRef>,
    /// Strong `InputArgRc` ownership for the inputargs seeded by
    /// `with_inputarg_types`. Production traces own their `InputArgRc`s
    /// via `TreeLoop.inputargs`; the test-and-fallback helper
    /// `with_inputarg_types` has no upstream `TreeLoop`, so it stashes
    /// fresh `InputArgRc`s here so the canonical host each InputArg OpRef
    /// resolves to is the same object across the run. `make_equal_to`
    /// then routes the chain step through `Forwarded::InputArg(_)`
    /// (`optimizer.py op.set_forwarded(newop)`) instead of the
    /// retired orphan-box forwarding fallback.
    /// Keyed by position rather than indexed by it. A bridge's fresh label
    /// inputargs are allocated at `inputarg_base = parent_high_water`, so the
    /// positions in use are two small clusters — `[0, num_inputs)` and
    /// `[base, base + num_inputs)` — with the whole parent trace between them.
    /// A `Vec` had to be grown across that gap every time one was named, which
    /// made naming an inputarg cost the parent trace's length, on every bridge
    /// the optimizer ran including the ones it declined.
    pub(crate) inputarg_refs: FxHashMap<u32, majit_ir::InputArgRc>,
    /// Synthetic `OpRc` stand-ins for ResOp operand placeholders whose
    /// real producer has not been (and may never be) emitted, indexed
    /// sparsely by `OpRef::raw()`. `materialize_operand_at` falls back to
    /// synthesising a `SameAsI/F/R` (or `Jump`) Op with the requested
    /// type and binding the operand to it so `make_equal_to` routes a
    /// chain step that targets such a placeholder through
    /// `Forwarded::Op(_)`. When a real producer Op is later emitted at
    /// the same OpRef position, `emit()` re-binds the operand to that
    /// Op (carrying forwarded state across) and the synthetic stand-in
    /// becomes unreferenced from the operand but is still retained here
    /// for the OptContext's lifetime, so a position that is looked up again
    /// resolves to the same `_forwarded` host rather than a second stand-in.
    ///
    /// Keyed by the full type-tagged `OpRef`, so a typed and an untyped
    /// (or differently-typed) position sharing a raw `u32` are distinct
    /// entries instead of evicting each other in a raw-indexed slot.
    // Insertion-ordered map (`IndexMap`) rather than the Vec-backed
    // `IndexMap`: `find_producer_op` does `resop_refs.get(&opref)` once per
    // live box of every guard inside `store_final_boxes_in_guard`, and a
    // Vec-backed `get` is O(n), making the box-numbering O(n^2) on very
    // large traces, where it can dominate the whole compile time.
    // `IndexMap` keeps the same insertion-ordered `values()` semantics the
    // earlier container guaranteed but resolves `get` in O(1).  No PyPy
    // counterpart (upstream keys producers on `box._forwarded`, not a
    // positional map).
    pub(crate) resop_refs: OpRefFxIndexMap<majit_ir::resoperation::OpRc>,
    /// Live `mint_synthetic_resop` / `install_canonical_producer` stand-ins
    /// that have NOT been superseded by an `emit` at their position. Recorder
    /// ops from `bind_input_resops` are the iterator box itself
    /// (`opencoder.py TraceIterator.next` `_cache`); they stay in `resop_refs`
    /// and are not listed here. The Phase-1 drain walks remaining entries
    /// into `phase1_emit_ops` so retrace sees them in
    /// `partial_trace_operations`. At most one live entry per position
    /// (emit invariant). A Vec scan is enough: this set is only synthetics
    /// and extra producers, not the whole trace.
    pub(crate) live_synthetics: Vec<majit_ir::resoperation::OpRc>,
    /// Phase 1 emit ops carried into Phase 2's lookup surface.
    ///
    /// In RPython, a Box referenced cross-phase keeps its `.type` attribute
    /// because the Box object survives both phases (history.py:220 parity).
    /// Pyre's flat `OpRef(u32)` requires explicit carry: Phase 1's emit ops
    /// — at positions in `[num_inputs, phase2_inputarg_base)` — appear as
    /// `imported_label_args`, fail-arg sources, and `record_same_as` source
    /// arguments inside Phase 2, but never inside Phase 2's own `new_operations`.
    /// `op_at` falls back to this slice so `op.type_` stays the single source
    /// of truth for Phase 1 emit OpRef types.
    pub phase1_emit_ops: Vec<majit_ir::OpRc>,
    /// `pos -> producer` index over `phase1_emit_ops`, mirroring the `rfind`
    /// (last-occurrence-wins) lookup in `find_producer_op`. This OptContext
    /// field is written exactly once — the Phase 1→2 handoff
    /// (`ctx.phase1_emit_ops = mem::take(...)` in `optimizer.rs`) — and never
    /// mutated afterwards, so the index is rebuilt in lockstep there via
    /// `rebuild_phase1_emit_ops_index`. Without it, `find_producer_op` scans
    /// the full cross-phase carry (tens of thousands of ops on a
    /// whole-program loop) per live
    /// box of every Phase 2 guard, the dominant O(n^2) compile cost. No PyPy
    /// counterpart: upstream keys producers on `box._forwarded`, not a
    /// positional map.
    pub(crate) phase1_emit_ops_index: FxHashMap<OpRef, majit_ir::OpRc>,
    /// Recorder trace ops that carry the input operands' producer `Op`
    /// (e.g. the `IntLt`/immutable `GetfieldGcI` operands of a recorded loop).
    /// Seeded at optimizer setup from the recorder's `Rc<Op>` slice
    /// (`TreeLoop.ops` at the loop-finish / simple-loop sites, or the Phase-2
    /// threaded `explicit_input_ops_seed`). `ensure_inputarg_bindings` reads
    /// the InputArg objects their operands carry.
    pub(crate) input_ops: Vec<majit_ir::OpRc>,
    /// optimizer.py:644,679 _last_guard_op — index of the last guard in
    /// new_operations that had full resume data built. Consecutive guards
    /// share resume data via _copy_resume_data_from (ResumeGuardCopiedDescr).
    ///
    /// Production runs through `Optimizer::emit_operation`, which owns the
    /// guard chain via its own `last_guard_op_idx` (`optimizer.rs`).
    /// This field tracks the chain for the standalone OptContext entry
    /// (unit tests that drive `OptContext::emit` without an `Optimizer`);
    /// `OptContext::emit` gates its guard handling on `!in_final_emission`
    /// to avoid duplicating Optimizer's bookkeeping.
    last_guard_idx: Option<usize>,
    /// `_emit_operation`'s `self._last_guard_op = None`, raised for the
    /// `Optimizer`-owned chain.
    ///
    /// RPython has a single emit funnel: `emit_extra` -> `emit` ->
    /// `_emit_operation`, so an op emitted while the Optimizer is mid-flight
    /// (a forced virtual's NEW/SETFIELD_GC, a flushed lazy set) still runs the
    /// chain reset.  Pyre's `in_final_emission` shortcut emits those straight
    /// into `new_operations`, bypassing `Optimizer::emit_operation`.  Record
    /// the break here instead; `Optimizer::emit_guard_operation` takes it
    /// before deciding whether the next guard may share.
    guard_chain_broken: bool,
    /// Last rd_resume_position with a valid snapshot. Used as fallback
    /// for optimizer-created guards that can't share from a previous guard.
    /// resume.py parity: RPython guards always get a snapshot via
    /// capture_resumedata; pyre tracks the nearest valid position.
    pub last_seen_snapshot_pos: Option<i32>,
    /// `optimizer.cpu` (`model.py AbstractCPU`) backref — the
    /// shared backend services entry point Optimization sub-classes
    /// reach via `self.optimizer.cpu.<method>()` in RPython.  Pyre's
    /// OptContext (the shared state holder Optimization sub-classes
    /// route through) carries an `Arc<dyn Cpu>` clone of `Optimizer.cpu`
    /// because it has no direct backref to the surrounding Optimizer.
    /// Used by `cls_of_box(&Operand)` (mod.rs body) and reachable by
    /// future `bh_*` ports.
    pub cpu: std::sync::Arc<dyn crate::cpu::Cpu>,
    /// llmodel.py:55 `self.remove_gctypeptr =
    /// translator.config.translation.gcremovetypeptr`. model.py:26
    /// default is `False`; PyPy x86 enables `--gcremovetypeptr` which
    /// flips this to `True` so the GC header carries the typeid and
    /// `obj[0]` no longer holds the rclass typeptr.
    ///
    /// Pyre's PyObject layout keeps `ob_type` at offset 0 BUT many
    /// static singletons (INSTANCE_TYPE, INT_TYPE, …) carry no GC
    /// header — `GUARD_IS_OBJECT` reads `obj - GcHeader::SIZE` and
    /// SIGBUSes on them. The pyre default matches the `True` branch
    /// (skip GUARD_IS_OBJECT, emit GUARD_NONNULL_CLASS as a single
    /// op). Consumed by `info.py/:348 InstancePtrInfo.make_guards`.
    pub remove_gctypeptr: bool,
    /// optimizer.py `Optimization.last_emitted_operation` — set
    /// to the just-emitted op (or `REMOVED` sentinel) by the base
    /// class's `_emit_operation`, read by callers like
    /// `optimize_GUARD_NO_EXCEPTION` (rewrite.py) to check
    /// whether the preceding op was dropped. The slot is updated on
    /// every emit across all passes, so a remove in pass N is visible
    /// to pass N+1.
    ///
    /// Pyre folds the REMOVED sentinel into a `bool` and lifts the
    /// slot from per-pass storage (where it was OptRewrite-local) to
    /// `OptContext` so the cross-pass scope matches the upstream
    /// base-class contract. Set by `propagate_from_pass_range` on
    /// `OptimizationResult::Remove`, reset on every successful
    /// `emit_operation` (see `Optimizer::emit_operation`).
    pub last_op_removed: bool,
    /// `history.py Const*` object identity. `get_box_replacement` on a
    /// Const box returns that same box; minting a fresh `Operand::Const`
    /// on every `resolve_to_operand` lookup is a lookup-time allocation
    /// RPython never pays.
    const_operands: std::cell::RefCell<crate::FxIndexMap<OpRef, Operand>>,
    /// Deferred `InvalidLoop` signal.  RPython `raise InvalidLoop`
    /// abandons the trace; pyre carries the same signal as a
    /// `Result<_, InvalidLoop>` threaded through the optimizer driver so
    /// it works under `panic=abort` (wasm32) as well as `panic=unwind`.
    /// Leaf sites that cannot return `Result` directly (e.g.
    /// `get_const_info_mut_box`, deep inside a pass) record the signal
    /// here and return a benign value; the driver checks it at the next
    /// barrier (`propagate_from_pass_range`) and converts it to `Err`.
    /// First write wins so the innermost reason is preserved.  `Cell` so
    /// `&self` leaf methods (e.g. `protect_speculative_operation`) can
    /// record without threading `&mut`.
    pub pending_invalid_loop: std::cell::Cell<Option<crate::optimize::InvalidLoop>>,
}

/// `heaptracker.all_fielddescrs`: `if name == 'typeptr': continue`
/// Uses FieldDescr.is_typeptr() which checks `field_name() == "typeptr"`,
/// matching RPython's name-based filtering.
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn is_typeptr_field(
    field_idx: u32,
    field_descrs: &[majit_ir::DescrRef],
    _descr: &majit_ir::DescrRef,
) -> bool {
    field_descrs
        .get(field_idx as usize)
        .and_then(|d| d.as_field_descr())
        .map(|fd| fd.is_typeptr())
        .unwrap_or(false)
}

/// resume.py _number_boxes parity — BoxEnv for optimizer context.
///
/// Wraps an immutable reference to OptContext, implementing the BoxEnv
/// trait so that ResumeDataLoopMemo.number() can tag boxes during
/// store_final_boxes_in_guard.
pub struct OptBoxEnv<'a> {
    pub ctx: &'a OptContext,
}

/// resume.py `getconst`: the word a `Const` carries, and the type tag naming
/// which register file it belongs in. `Void` has no register file of its own
/// and numbers as a zero int, the same as an unresolvable operand.
fn const_value_as_word(value: Value) -> (i64, majit_ir::Type) {
    match value {
        Value::Int(v) => (v, majit_ir::Type::Int),
        Value::Float(f) => (f.to_bits() as i64, majit_ir::Type::Float),
        Value::Ref(r) => (r.0 as i64, majit_ir::Type::Ref),
        Value::Void => (0, majit_ir::Type::Int),
    }
}

impl<'a> majit_ir::BoxEnv for OptBoxEnv<'a> {
    fn get_box_replacement(&self, opref: OpRef) -> OpRef {
        let walked = self.ctx.get_replacement_opref(opref);
        // `_number_boxes` does `box = get_box_replacement(box)` then
        // `isinstance(box, Const)`. A stack-resident InputArg folded
        // to ConstPtr would TAGCONST and drop out of the dump. Stay
        // on the InputArg so the next `Vm::new` still has a livebox.
        if opref.is_input_arg()
            && matches!(walked, OpRef::ConstPtr(_))
            && walked
                .inline_const_bits()
                .is_some_and(|bits| OptContext::ref_addr_is_stack_resident(bits as usize))
        {
            return opref;
        }
        walked
    }

    fn get_box_replacement_operand(&self, opref: OpRef) -> Operand {
        // resume.py:202 box.get_box_replacement() as a box OBJECT. The canonical
        // host is the producer Op / InputArg, so two reaches of one logical box
        // return the same producer Rc (ptr_eq) — the #160 livebox dedup key.
        if opref.is_none() {
            return Operand::None;
        }
        self.ctx.get_box_replacement_operand(opref)
    }

    fn get_box_replacement_not_const(&self, opref: OpRef) -> OpRef {
        // resoperation.py:64-65 not_const arm. The resume liveboxes boundary
        // is legitimately OpRef-keyed (rd_numb wire format), so resolve
        // through the OpRef glue directly.
        self.ctx.get_box_replacement_impl(opref, true)
    }

    fn is_const(&self, box_: &Operand) -> bool {
        // resume.py _number_boxes classifies the already resolved object
        // (`isinstance(box, Const)`). A stack-resident red forwarded to
        // ConstPtr must stay a TAGBOX; heap ConstPtrInfo stays Const.
        debug_assert!(box_.same_box(&box_.get_box_replacement(false)));
        if let Some(value) = box_.const_value() {
            return match value {
                majit_ir::Value::Ref(gcref) => !OptContext::ref_addr_is_stack_resident(gcref.0),
                _ => true,
            };
        }
        match box_.ptr_info().as_deref() {
            Some(crate::optimizeopt::info::PtrInfo::Constant(gcref)) => {
                !OptContext::ref_addr_is_stack_resident(gcref.0)
            }
            _ => false,
        }
    }

    fn get_const(&self, box_: &Operand) -> (i64, majit_ir::Type) {
        debug_assert!(box_.same_box(&box_.get_box_replacement(false)));
        if let Some(value) = box_.const_value() {
            return const_value_as_word(value);
        }
        match box_.ptr_info().as_deref() {
            Some(crate::optimizeopt::info::PtrInfo::Constant(gcref)) => {
                (gcref.0 as i64, majit_ir::Type::Ref)
            }
            // resume.py getconst is only called after is_const. Keep the
            // existing unrecognized-box contract instead of panicking.
            _ => (0, majit_ir::Type::Int),
        }
    }

    fn get_type(&self, opref: OpRef) -> majit_ir::Type {
        // `BoxEnv::get_type` is the resume-serdes `box.type` reader
        // (resume.py:201 `box = box.get_box_replacement()` before any
        // type judgement). It is the non-`Option` adapter over
        // `OptContext::opref_type`, the single 5-layer `box.type`
        // resolver that mirrors upstream `AbstractValue.type`
        // (resoperation.py). A VoidOp (`Some(Void)`) or a type-less
        // OpRef (`None`) maps to the `Int` default the resume encoder
        // expects for non-typed slots.
        match self.ctx.opref_type(opref) {
            Some(tp) if tp != majit_ir::Type::Void => tp,
            _ => majit_ir::Type::Int,
        }
    }

    fn is_virtual_ref(&self, box_: &Operand) -> bool {
        // resume.py _number_boxes reads getptrinfo on its resolved box.
        debug_assert!(box_.same_box(&box_.get_box_replacement(false)));
        box_.ptr_info().is_some_and(|info| info.is_virtual())
    }

    fn is_virtual_raw(&self, box_: &Operand) -> bool {
        debug_assert!(box_.same_box(&box_.get_box_replacement(false)));
        box_.ptr_info().is_some_and(|info| {
            matches!(
                &*info,
                crate::optimizeopt::info::PtrInfo::VirtualRawBuffer(_)
                    | crate::optimizeopt::info::PtrInfo::VirtualRawSlice(_)
            )
        })
    }

    fn has_known_class(&self, box_: &Operand) -> bool {
        // bridgeopt.py:79-80, and the comment at line 72: the class is
        // known only after the guard. A constant ref's class is the
        // object. An instance class with `last_guard_pos == -1` was not
        // established by a guard in this trace, so the bridge must not
        // read one failure's runtime class and treat it as known.
        let Some(info) = box_.ptr_info() else {
            return false;
        };
        if info.get_known_class(self.ctx.cpu.as_ref()).is_none() {
            return false;
        }
        if box_.const_value().is_some() || matches!(&*info, PtrInfo::Constant(_)) {
            return true;
        }
        info.get_last_guard_pos().is_some()
    }

    fn get_virtual_fields(&self, opref: OpRef) -> Option<majit_ir::VirtualFieldsInfo> {
        let resolved_box = self.ctx.get_box_replacement_operand_opt(opref);
        let info = resolved_box.as_ref().and_then(Operand::ptr_info)?;
        let fielddescrs = info.all_fielddescrs_from_descr();
        match &*info {
            PtrInfo::Virtual(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: Some(vi.descr.clone()),
                known_class: vi.known_class,
                // info.py `_visitor_walk_recursive` registers `_fields` in
                // `descr.get_all_fielddescrs()` order. Storage keys by
                // `parent_list_slot` (identity in the field's parent list),
                // which is not always the enumerate index of this list.
                field_oprefs: virtual_field_oprefs_for_descrs(
                    &vi.fields,
                    &fielddescrs,
                    self.ctx,
                    Some(&vi.descr),
                ),
            }),
            PtrInfo::VirtualStruct(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: Some(vi.descr.clone()),
                known_class: None,
                field_oprefs: virtual_field_oprefs_for_descrs(
                    &vi.fields,
                    &fielddescrs,
                    self.ctx,
                    Some(&vi.descr),
                ),
            }),
            PtrInfo::VirtualArray(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: Some(vi.descr.clone()),
                known_class: None,
                field_oprefs: vi
                    .items
                    .iter()
                    .map(|vref| self.ctx.get_replacement_opref(vref.to_opref()))
                    .collect(),
            }),
            PtrInfo::VirtualArrayStruct(vi) => {
                // Same list `visit_varraystruct` snapshots: ArrayDescr
                // `get_all_fielddescrs` is `all_interiorfielddescrs`
                // (`info.py ArrayStructInfo.visitor_dispatch_virtual_type`).
                let fielddescrs = varraystruct_visitor_fielddescrs(&vi.descr, &vi.fielddescrs);
                Some(majit_ir::VirtualFieldsInfo {
                    descr: Some(vi.descr.clone()),
                    known_class: None,
                    field_oprefs: vi
                        .element_fields
                        .iter()
                        .flat_map(|ef| {
                            virtual_field_oprefs_for_descrs(ef, fielddescrs, self.ctx, None)
                        })
                        .collect(),
                })
            }
            PtrInfo::VirtualRawBuffer(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: None,
                known_class: None,
                field_oprefs: vi
                    .buffer
                    .values()
                    .iter()
                    .map(|vref| self.ctx.get_replacement_opref(vref.to_opref()))
                    .collect(),
            }),
            // `info.py` `RawSlicePtrInfo._visitor_walk_recursive`:
            //
            // ```python
            // def _visitor_walk_recursive(self, op, visitor):
            //     source_op = get_box_replacement(op.getarg(0))
            //     visitor.register_virtual_fields(op, [source_op])
            //     if self.parent.is_virtual():
            //         self.parent.visitor_walk_recursive(source_op, visitor)
            // ```
            //
            // pyre's consumer (the virtual worklist in
            // `ResumeDataLoopMemo::finish`) drives the recursion off
            // `get_virtual_fields`
            // — registering the parent OpRef here lets the worklist enqueue
            // the parent and re-enter `get_virtual_fields` on it, which
            // matches RPython's `parent.visitor_walk_recursive(source_op,
            // visitor)` follow-up.  Only fires while the slice is still
            // virtual (`slice.parent` non-NONE); after `force_box_impl`
            // materializes the slice the gate in `is_virtual` flips False
            // and the caller drops the entry.
            PtrInfo::VirtualRawSlice(vi) if !vi.parent.is_none() => {
                Some(majit_ir::VirtualFieldsInfo {
                    descr: None,
                    known_class: None,
                    field_oprefs: vec![self.ctx.get_replacement_opref(vi.parent.to_opref())],
                })
            }
            // vstring.py VStringPlainInfo._visitor_walk_recursive:
            //   `visitor.register_virtual_fields(instbox, self._chars)`
            //
            // vstring.py VStringSliceInfo._visitor_walk_recursive:
            //   `visitor.register_virtual_fields(instbox, [self.s, self.start, self.lgtop])`
            //   (then recurses into the parent string if it is itself virtual).
            //
            // vstring.py VStringConcatInfo._visitor_walk_recursive:
            //   `visitor.register_virtual_fields(instbox, [self.vleft, self.vright])`
            //
            // Only virtual StrPtrInfo variants register fields; a non-virtual
            // `VStringVariant::Ptr` skips this arm (falls through to None).
            PtrInfo::Str(sinfo) if sinfo.is_virtual() => {
                use crate::optimizeopt::info::VStringVariant;
                let field_oprefs: Vec<OpRef> = match &sinfo.variant {
                    // vstring.py _visitor_walk_recursive: self._chars. `None` slots represent
                    // unfilled positions — the resume encoder later tags those
                    // with UNINITIALIZED (matching how RPython treats missing
                    // char boxes, since STRSETITEM may not have run for every
                    // index yet at guard time).
                    VStringVariant::Plain(p) => p
                        ._chars
                        .iter()
                        .map(|slot| {
                            slot.as_ref()
                                .map(|r| self.ctx.get_replacement_opref(r.to_opref()))
                                .unwrap_or(OpRef::NONE)
                        })
                        .collect(),
                    // vstring.py _visitor_walk_recursive: [self.s, self.start, self.lgtop].
                    VStringVariant::Slice(s) => vec![
                        self.ctx.get_replacement_opref(s.s.to_opref()),
                        self.ctx.get_replacement_opref(s.start.to_opref()),
                        self.ctx.get_replacement_opref(s.lgtop.to_opref()),
                    ],
                    // vstring.py _visitor_walk_recursive: [self.vleft, self.vright].
                    VStringVariant::Concat(c) => vec![
                        self.ctx.get_replacement_opref(c.vleft.to_opref()),
                        self.ctx.get_replacement_opref(c.vright.to_opref()),
                    ],
                    // Non-virtual `VStringVariant::Ptr` would not reach
                    // here because of the `is_virtual()` guard above.
                    VStringVariant::Ptr => Vec::new(),
                };
                Some(majit_ir::VirtualFieldsInfo {
                    descr: None,
                    known_class: None,
                    field_oprefs,
                })
            }
            _ => None,
        }
    }

    fn make_virtual_info(
        &self,
        opref: OpRef,
        fieldnums: Vec<i16>,
    ) -> Option<std::rc::Rc<majit_ir::RdVirtualInfo>> {
        let resolved_box = self.ctx.get_box_replacement_operand_opt(opref);
        let info = resolved_box.as_ref().and_then(Operand::ptr_info)?;
        // resume.py `ResumeDataVirtualAdder.make_virtual_info`:
        //
        //     vinfo = info._cached_vinfo
        //     if vinfo is not None and vinfo.equals(fieldnums):
        //         return vinfo
        //     vinfo = info.visitor_dispatch_virtual_type(self)
        //     vinfo.set_content(fieldnums)
        //     info._cached_vinfo = vinfo
        //     return vinfo
        //
        // The cache stores an `Rc<RdVirtualInfo>` so that cache hits return
        // the same shared handle (matching RPython's Python object identity
        // on cache hit in ResumeDataVirtualAdder.make_virtual_info). Downstream storage in
        // `storage.rd_virtuals` keeps the shared handle so two guards that
        // reference the same virtual with the same fieldnums end up pointing
        // at the same `RdVirtualInfo` object.
        if let Some(cache) = info.cached_vinfo()
            && let Some(vinfo) = cache.borrow().as_ref()
            && vinfo.equals(&fieldnums)
        {
            return Some(std::rc::Rc::clone(vinfo));
        }
        let mut builder = RdVirtualInfoBuilder;
        let mut vinfo = info.visitor_dispatch_virtual_type(&mut builder)??;
        // resume.py: vinfo.set_content(fieldnums)
        vinfo.set_content(fieldnums);
        let shared = std::rc::Rc::new(vinfo);
        // ResumeDataVirtualAdder.make_virtual_info stores the shared handle
        // so a later equals-hit returns the SAME object.
        if let Some(cache) = info.cached_vinfo() {
            *cache.borrow_mut() = Some(std::rc::Rc::clone(&shared));
        }
        Some(shared)
    }

    fn virtual_info_would_be_reused(&self, opref: OpRef, fieldnums: &[i16]) -> bool {
        // resume.py reads the same info._cached_vinfo written above.
        let resolved_box = self.ctx.get_box_replacement_operand_opt(opref);
        let Some(info) = resolved_box.as_ref().and_then(Operand::ptr_info) else {
            return false;
        };
        let Some(cache) = info.cached_vinfo() else {
            return false;
        };
        cache
            .borrow()
            .as_ref()
            .is_some_and(|vinfo| vinfo.equals(fieldnums))
    }
}

/// resume.py `ResumeDataVirtualAdder` in its role as a
/// `VirtualVisitor` — each `visit_*` builds a fresh `RdVirtualInfo`
/// subclass without fieldnums (the caller attaches those via
/// `set_content`). In pyre, `make_virtual_info` lives on the
/// `BoxEnv` impl (not on `ResumeDataVirtualAdder`) because PtrInfo
/// lookup is the optimizer's responsibility, so the visitor adapter
/// is this zero-sized helper instead of `ResumeDataVirtualAdder`
/// itself.
struct RdVirtualInfoBuilder;

/// `info.py ArrayStructInfo.visitor_dispatch_virtual_type` hands
/// `descr.get_all_fielddescrs()` (ArrayDescr: `all_interiorfielddescrs`)
/// to `visit_varraystruct`. Fall back to the cached list when the array
/// descr has not published interiors yet.
fn varraystruct_visitor_fielddescrs<'a>(
    arraydescr: &'a majit_ir::DescrRef,
    fallback: &'a [majit_ir::DescrRef],
) -> &'a [majit_ir::DescrRef] {
    arraydescr
        .as_array_descr()
        .and_then(|ad| ad.get_all_interiorfielddescrs())
        .unwrap_or(fallback)
}

/// Storage slot for a visitor fielddescr. `visit_varraystruct` unwraps
/// `InteriorFieldDescr`; SETFIELD / SETINTERIORFIELD key `_fields` by
/// `parent_list_slot` of that inner `FieldDescr`.
///
/// RPython `ArrayStructInfo._compute_index` uses
/// `fielddescr.get_field_descr().get_index()` against
/// `arraydescr.get_all_fielddescrs()`. pyre's `parent_list_slot` is the
/// identity lookup in the parent SizeDescr list that replaces a stale
/// `index_in_parent`. Those two lists can differ in order or length, so
/// `fieldnums[i]` must be looked up by that slot, not by enumerate.
fn visitor_fielddescr_storage_slot(
    descr: &majit_ir::DescrRef,
    owner: Option<&majit_ir::DescrRef>,
) -> Option<u32> {
    let fd = descr
        .as_field_descr()
        .or_else(|| descr.as_interior_field_descr().map(|ifd| ifd.field_descr()))?;
    Some(crate::optimizeopt::virtualize::parent_list_slot_in(
        owner, fd,
    ))
}

/// `field_oprefs[i]` is the value stored for `fielddescrs[i]`, so
/// `set_content` fieldnums pair 1:1 with the snapshot `visit_*` wrote.
fn virtual_field_oprefs_for_descrs(
    fields: &[(u32, Operand)],
    fielddescrs: &[majit_ir::DescrRef],
    ctx: &OptContext,
    owner: Option<&majit_ir::DescrRef>,
) -> Vec<OpRef> {
    fielddescrs
        .iter()
        .enumerate()
        .map(|(fi, descr)| {
            let slot = visitor_fielddescr_storage_slot(descr, owner).unwrap_or(fi as u32);
            fields
                .iter()
                .find(|(field_idx, _)| *field_idx == slot)
                .map(|(_, vref)| ctx.get_replacement_opref(vref.to_opref()))
                .unwrap_or(OpRef::NONE)
        })
        .collect()
}

/// Snapshot a live field descr the way `setfield` classifies it:
/// `is_pointer_field` → Ref, `is_float_field` → Float, else `field_type()`.
/// A FLAG_POINTER field whose IR type is Int (raw `*mut`) still decodes as
/// Ref. Unwraps `as_interior_field_descr().field_descr()` when
/// `as_field_descr()` is None (`visit_varraystruct` does the same unwrap).
fn snapshot_field_descr_info(descr: &majit_ir::DescrRef) -> majit_ir::FieldDescrInfo {
    let fd = descr
        .as_field_descr()
        .or_else(|| descr.as_interior_field_descr().map(|ifd| ifd.field_descr()));
    match fd {
        Some(fd) => {
            let field_type = if fd.is_pointer_field() {
                majit_ir::Type::Ref
            } else if fd.is_float_field() {
                majit_ir::Type::Float
            } else {
                fd.field_type()
            };
            majit_ir::FieldDescrInfo {
                index: descr.index(),
                offset: fd.offset(),
                field_type,
                field_size: fd.field_size(),
            }
        }
        None => majit_ir::FieldDescrInfo {
            index: descr.index(),
            offset: 0,
            field_type: majit_ir::Type::Int,
            // The fallback pairs with `Type::Int` above, whose storage is
            // `i64` on every target — only a `Ref` field follows the target
            // word.
            field_size: crate::jitcode::scalar_size(majit_ir::Type::Int),
        },
    }
}

impl crate::walkvirtual::VirtualVisitor for RdVirtualInfoBuilder {
    type VInfo = Option<majit_ir::RdVirtualInfo>;

    fn visit_not_virtual(&mut self, _value: OpRef) -> Self::VInfo {
        // resume.py `visit_not_virtual` asserts unreachable.
        debug_assert!(false, "visit_not_virtual reached via virtual dispatch");
        None
    }

    // resume.py visit_virtual → VirtualInfo(descr, fielddescrs)
    fn visit_virtual(
        &mut self,
        descr: &majit_ir::DescrRef,
        _fielddescr_indices: &[u32],
        fielddescrs: &[majit_ir::DescrRef],
    ) -> Self::VInfo {
        // `FieldDescrInfo.index` must carry the stable descriptor index
        // (tagged offset) so resume-data readers can identify the field by
        // byte offset. Previously stored `fi as u32` (iteration counter),
        // which made `extract_runtime_field_offset` always fail for virtuals
        // being materialized on bridge entry.
        let built_fielddescrs: Vec<majit_ir::FieldDescrInfo> =
            fielddescrs.iter().map(snapshot_field_descr_info).collect();
        let sd = descr.as_size_descr();
        Some(majit_ir::RdVirtualInfo::VirtualInfo {
            descr: Some(descr.clone()),
            type_id: sd.map(|s| s.type_id()).unwrap_or(0),
            // resume.py allocate_with_vtable(descr=self.descr) — the
            // vtable is derived from descr; majit mirrors by reading it
            // off the SizeDescr when the descr carries class info.
            known_class: sd.map(|s| s.vtable() as i64).filter(|&v| v != 0),
            fielddescrs: built_fielddescrs,
            fieldnums: Vec::new(),
            descr_size: sd.map(|s| s.size()).unwrap_or(0),
        })
    }

    // resume.py visit_vstruct → VStructInfo(typedescr, fielddescrs)
    fn visit_vstruct(
        &mut self,
        typedescr: &majit_ir::DescrRef,
        _fielddescr_indices: &[u32],
        fielddescrs: &[majit_ir::DescrRef],
    ) -> Self::VInfo {
        // See `visit_virtual` — index must be the stable descriptor index,
        // not the iteration counter.
        let built_fielddescrs: Vec<majit_ir::FieldDescrInfo> =
            fielddescrs.iter().map(snapshot_field_descr_info).collect();
        let sd = typedescr.as_size_descr();
        Some(majit_ir::RdVirtualInfo::VStructInfo {
            typedescr: Some(typedescr.clone()),
            type_id: sd.map(|s| s.type_id()).unwrap_or(0),
            fielddescrs: built_fielddescrs,
            fieldnums: Vec::new(),
            descr_size: sd.map(|s| s.size()).unwrap_or(0),
        })
    }

    // resume.py visit_varray → VArrayInfoClear / VArrayInfoNotClear
    fn visit_varray(&mut self, arraydescr: &majit_ir::DescrRef, clear: bool) -> Self::VInfo {
        let kind = arraydescr
            .as_array_descr()
            .map(|ad| match ad.item_type() {
                majit_ir::Type::Float => 2u8,
                majit_ir::Type::Int => 1u8,
                _ => 0u8,
            })
            .unwrap_or(0);
        let ad = Some(arraydescr.clone());
        Some(if clear {
            majit_ir::RdVirtualInfo::VArrayInfoClear {
                arraydescr: ad,
                kind,
                fieldnums: Vec::new(),
            }
        } else {
            majit_ir::RdVirtualInfo::VArrayInfoNotClear {
                arraydescr: ad,
                kind,
                fieldnums: Vec::new(),
            }
        })
    }

    // resume.py visit_varraystruct → VArrayStructInfo
    fn visit_varraystruct(
        &mut self,
        arraydescr: &majit_ir::DescrRef,
        length: usize,
        _fielddescr_indices: &[u32],
        fielddescrs: &[majit_ir::DescrRef],
    ) -> Self::VInfo {
        // info.py ArrayStructInfo.visitor_dispatch_virtual_type hands
        // descr.get_all_fielddescrs() (all_interiorfielddescrs); fall
        // back to the variant's cached fielddescrs when descr lacks it.
        let canonical_fielddescrs: Vec<majit_ir::DescrRef> =
            varraystruct_visitor_fielddescrs(arraydescr, fielddescrs).to_vec();
        let mut fo = Vec::new();
        let mut fs = Vec::new();
        let mut ft = Vec::new();
        for fd in &canonical_fielddescrs {
            let info = snapshot_field_descr_info(fd);
            fo.push(info.offset);
            fs.push(info.field_size);
            // resume.py setinteriorfield: is_pointer_field / is_float_field.
            ft.push(match info.field_type {
                majit_ir::Type::Float => 2u8,
                majit_ir::Type::Int => 1u8,
                _ => 0u8,
            });
        }
        if ft.is_empty() {
            ft = vec![0u8; canonical_fielddescrs.len()];
        }
        let ad = arraydescr.as_array_descr();
        let fielddescr_count = canonical_fielddescrs.len();
        Some(majit_ir::RdVirtualInfo::VArrayStructInfo {
            arraydescr: Some(arraydescr.clone()),
            size: length,
            fielddescrs: canonical_fielddescrs,
            fielddescr_indices: (0..fielddescr_count).map(|i| i as u32).collect(),
            field_types: ft,
            base_size: ad.map(|a| a.base_size()).unwrap_or(0),
            item_size: ad.map(|a| a.item_size()).unwrap_or(0),
            field_offsets: fo,
            field_sizes: fs,
            fieldnums: Vec::new(),
        })
    }

    // resume.py visit_vrawbuffer → VRawBufferInfo
    fn visit_vrawbuffer(
        &mut self,
        func: i64,
        size: usize,
        offsets: &[i64],
        descrs: &[majit_ir::DescrRef],
    ) -> Self::VInfo {
        let descr_infos: Vec<majit_ir::ArrayDescrInfo> = descrs
            .iter()
            .map(|d| {
                let ad = d.as_array_descr();
                majit_ir::ArrayDescrInfo {
                    index: d.index(),
                    base_size: ad.map_or(0, |a| a.base_size()),
                    item_size: ad.map_or(8, |a| a.item_size()),
                    // descr.py ArrayDescr.lendescr.offset — preserved
                    // across the resume summary so the materialize path
                    // sees the same length word offset the producer used.
                    len_offset: ad.and_then(|a| a.len_descr().map(|fd| fd.offset())),
                    item_type: ad.map_or(1, |a| {
                        if a.is_array_of_pointers() {
                            0
                        } else if a.is_array_of_floats() {
                            2
                        } else {
                            1
                        }
                    }),
                    is_signed: ad.is_none_or(|a| a.is_item_signed()),
                }
            })
            .collect();
        Some(majit_ir::RdVirtualInfo::VRawBufferInfo {
            func,
            size,
            offsets: offsets.to_vec(),
            descrs: descr_infos,
            fieldnums: Vec::new(),
        })
    }

    // resume.py visit_vrawslice → VRawSliceInfo
    fn visit_vrawslice(&mut self, offset: i64) -> Self::VInfo {
        Some(majit_ir::RdVirtualInfo::VRawSliceInfo {
            offset,
            fieldnums: Vec::new(),
        })
    }

    // resume.py visit_vstrplain → VStrPlainInfo / VUniPlainInfo
    fn visit_vstrplain(&mut self, is_unicode: bool) -> Self::VInfo {
        Some(if is_unicode {
            majit_ir::RdVirtualInfo::VUniPlainInfo {
                fieldnums: Vec::new(),
            }
        } else {
            majit_ir::RdVirtualInfo::VStrPlainInfo {
                fieldnums: Vec::new(),
            }
        })
    }

    // resume.py visit_vstrconcat → VStrConcatInfo / VUniConcatInfo
    fn visit_vstrconcat(&mut self, is_unicode: bool) -> Self::VInfo {
        // resume.py visit_vstrconcat — visitor constructs the shell variant with
        // no funcptr; the decoder looks up OS_STR_CONCAT / OS_UNI_CONCAT
        // via `callinfocollection.funcptr_for_oopspec(...)` at
        // materialization time (resume.py:1467-1468 / 1494-1495).
        Some(if is_unicode {
            majit_ir::RdVirtualInfo::VUniConcatInfo {
                fieldnums: Vec::new(),
            }
        } else {
            majit_ir::RdVirtualInfo::VStrConcatInfo {
                fieldnums: Vec::new(),
            }
        })
    }

    // resume.py visit_vstrslice → VStrSliceInfo / VUniSliceInfo
    fn visit_vstrslice(&mut self, is_unicode: bool) -> Self::VInfo {
        Some(if is_unicode {
            majit_ir::RdVirtualInfo::VUniSliceInfo {
                fieldnums: Vec::new(),
            }
        } else {
            majit_ir::RdVirtualInfo::VStrSliceInfo {
                fieldnums: Vec::new(),
            }
        })
    }

    fn register_virtual_fields(&mut self, _virtualbox: OpRef, _fieldboxes: &[OpRef]) {
        // resume.py register_virtual_fields — field registration happens elsewhere in pyre
        // (via resume.rs worklist + get_virtual_fields), not through the
        // visitor. This adapter is only a builder for RdVirtualInfo.
    }

    fn already_seen_virtual(&mut self, _virtualbox: OpRef) -> bool {
        // resume.py — same split as register_virtual_fields; the
        // pyre worklist tracks visited boxes directly.
        false
    }
}

impl OptContext {
    /// `optimizer.py:243` quasi-immutable dep registration with
    /// dict-as-set semantics (`heap.py:821-823`
    /// `self.optimizer.quasi_immutable_deps[qmutdescr.qmut] = None`).
    /// Vec-backed set keyed on instance identity, the way a Python dict keys
    /// on the object.
    pub fn add_quasi_immutable_dep(&mut self, dep: std::sync::Arc<dyn majit_ir::QuasiImmutHandle>) {
        if !self
            .quasi_immutable_deps
            .iter()
            .any(|seen| seen.instance_identity() == dep.instance_identity())
        {
            self.quasi_immutable_deps.push(dep);
        }
    }

    /// Record an `InvalidLoop` abandon signal from a leaf site that cannot
    /// return `Result` directly.  First write wins so the innermost reason
    /// survives.  The driver converts it to `Err` at the next barrier.
    pub fn signal_invalid_loop(&self, reason: &'static str) {
        let cur = self.pending_invalid_loop.take();
        self.pending_invalid_loop
            .set(Some(cur.unwrap_or(crate::optimize::InvalidLoop(reason))));
    }

    /// Consume any pending `InvalidLoop` signal recorded via
    /// [`signal_invalid_loop`](Self::signal_invalid_loop).
    pub fn take_invalid_loop(&self) -> Option<crate::optimize::InvalidLoop> {
        self.pending_invalid_loop.take()
    }

    /// Non-consuming check for a pending `InvalidLoop` signal.  Used by
    /// leaf sites (e.g. `constant_fold`) that must bail immediately after a
    /// failed `protect_speculative_operation` without consuming the signal,
    /// so the driver still aborts at its barrier.
    pub fn has_pending_invalid_loop(&self) -> bool {
        let v = self.pending_invalid_loop.take();
        let present = v.is_some();
        self.pending_invalid_loop.set(v);
        present
    }

    pub fn new(estimated_ops: usize) -> Self {
        OptContext {
            supports_efficient_uint_mul_high: true,
            new_operations: Vec::with_capacity(estimated_ops),
            new_operations_drained: false,
            new_operations_index: FxHashMap::with_capacity_and_hasher(
                estimated_ops,
                Default::default(),
            ),
            emitted_operations: indexmap::IndexSet::with_capacity_and_hasher(
                estimated_ops,
                rustc_hash::FxBuildHasher,
            ),
            num_inputs: 0,
            inputarg_base: 0,
            next_pos: 0,
            extra_operations_after: ExtraQueue::new(),
            extra_pending: Vec::new(),
            pending_guard_class_postprocess: None,
            pending_mark_last_guard: None,
            pending_finish_guard_postprocess: None,
            imported_short_pure_ops: Vec::new(),
            imported_virtual_args: None,
            imported_loop_invariant_results: Vec::new(),
            imported_short_preamble_builder: None,
            const_infos: crate::FxIndexMap::default(),

            potential_extra_ops: indexmap::IndexMap::new(),
            active_short_preamble_producer: None,
            preview_short_state: None,
            exported_const_short_boxes: Vec::new(),

            imported_virtuals: Vec::new(),
            imported_label_args: None,
            can_replace_guards: true,
            patchguardop: None,
            preamble_end_args: None,
            skip_flush_mode: false,
            building_bridge: false,
            current_pass_idx: 0,
            optearlyforce_idx: 0,

            in_final_emission: false,
            force_direct: false,
            callinfocollection: None,
            resumedata_memo: std::rc::Rc::new(std::cell::RefCell::new(
                crate::resume::ResumeDataLoopMemo::new(),
            )),
            pending_for_guard: Vec::new(),
            pending_pure_from_args: Vec::new(),
            pending_pure_from_args2: Vec::new(),
            pure_ops: None,
            constant_fold_alloc: None,
            string_length_resolver: None,
            string_content_resolver: None,
            string_constant_alloc: None,
            quasi_immutable_deps: Vec::new(),
            snapshot_boxes: Vec::new(),
            snapshot_frame_sizes: SnapshotFrameSizes::new(),
            snapshot_vable_boxes: Vec::new(),
            // resume.py:401-402: `-1` until an Optimizer built with a vable
            // config overwrites it in `optimize_with_constants_and_inputs_at`.
            minimum_virtualizable_size: -1,
            snapshot_vref_boxes: Vec::new(),
            snapshot_frame_pcs: SnapshotFramePcs::new(),
            byte_bridge_resume: None,

            inputargs: Vec::new(),
            inputarg_refs: FxHashMap::with_capacity_and_hasher(
                estimated_ops.min(32),
                Default::default(),
            ),
            resop_refs: indexmap::IndexMap::with_capacity_and_hasher(
                estimated_ops,
                Default::default(),
            ),
            live_synthetics: Vec::new(),
            phase1_emit_ops: Vec::new(),
            phase1_emit_ops_index: FxHashMap::default(),
            input_ops: Vec::with_capacity(estimated_ops),
            last_guard_idx: None,
            guard_chain_broken: false,
            last_seen_snapshot_pos: None,
            cpu: crate::cpu::default_cpu(),
            remove_gctypeptr: true,
            last_op_removed: false,
            pending_invalid_loop: std::cell::Cell::new(None),
            const_operands: std::cell::RefCell::new(crate::FxIndexMap::default()),
        }
    }

    /// Test-only inputarg-free constructor. Production paths must always go
    /// through [`Self::with_inputarg_types`] so every inputarg slot lands a
    /// typed operand matching `opencoder.py:259 inputarg_from_tp(arg.type)`;
    /// passing `num_inputs > 0` here would silently drop the type tag and
    /// produce `Type::Void` reads. Sealed under `#[cfg(test)]` to make the
    /// rule structural rather than discipline-only.
    #[cfg(test)]
    pub fn with_num_inputs(estimated_ops: usize, num_inputs: usize) -> Self {
        debug_assert_eq!(
            num_inputs, 0,
            "with_num_inputs(_, {num_inputs}) — non-zero M requires typed seeding; \
             use `with_inputarg_types(estimated_ops, &[Type; M])` instead \
             (opencoder.py:259 inputarg_from_tp parity)",
        );
        Self::with_inputarg_types(estimated_ops, &[])
    }

    /// Construct an `OptContext` and seed `inputarg_refs` with one canonical
    /// `InputArg::from_type(tp, i)` per entry of `inputarg_types`.
    ///
    /// Mirrors `TraceIterator::new` (`opencoder.rs`, parity with
    /// `opencoder.py` `inputarg_from_tp(arg.type)`). Test fixtures
    /// that construct via this helper exercise the optimizer's operand-direct
    /// routing — the production path.
    ///
    /// `inputarg_types` carries the type tags needed to round-trip
    /// `OpRef::input_arg_typed(i, tp)` on read; `with_num_inputs` is the
    /// untyped delegate that defaults every slot to `Type::Void`.
    pub fn with_inputarg_types(estimated_ops: usize, inputarg_types: &[majit_ir::Type]) -> Self {
        let num_inputs = inputarg_types.len();
        let mut ctx =
            Self::with_num_inputs_and_start_pos(estimated_ops, num_inputs, 0, num_inputs as u32);
        // Seed a fresh canonical `InputArgRc` per slot so the optimizer's
        // `make_equal_to` routes an InputArg-targeted chain step through
        // `Forwarded::InputArg(_)` (`optimizer.py op.set_forwarded(newop)`).
        // The `InputArgRc`s are stashed in `ctx.inputarg_refs` so every
        // reader of a slot resolves to the same host for the OptContext's
        // lifetime. Production traces own these via
        // `TreeLoop.inputargs`; this test-and-fallback helper has no upstream
        // `TreeLoop`.
        ctx.inputarg_refs = inputarg_types
            .iter()
            .enumerate()
            .map(|(i, &tp)| {
                (
                    i as u32,
                    majit_ir::InputArgRc::new(majit_ir::InputArg::from_type(tp, i as u32)),
                )
            })
            .collect();
        // Seed `ctx.inputargs` so strict accessors like
        // `inputarg_type_at_strict` return `Some(tp)` matching slot i. Each
        // entry is `OpRef::input_arg_typed(i, tp)` so the variant tag IS the
        // type (resoperation.py InputArgInt/727/739).
        ctx.inputargs = majit_ir::OpRef::inputarg_refs(inputarg_types);
        ctx
    }

    /// Mirror every InputArg position into `inputarg_refs[pos]`: an
    /// already-bound slot keeps the `InputArgRc` it binds; an unbound slot
    /// gets a fresh one (bound here) carrying the position's type. Stashing
    /// the strong ref in `inputarg_refs` gives the canonical-host readers
    /// (`resolve_to_operand` / `read_forwarded` / `clear_forwarded`) a live
    /// `InputArg.forwarded` host to resolve to.
    ///
    /// Phase 2 enters with a fresh per-iteration inputarg set that the
    /// previous OptContext's `inputarg_refs` no longer names. Re-binding here restores
    /// `Forwarded::InputArg(_)` reachability for every InputArg operand the
    /// optimizer will hand to `make_equal_to` (`optimizer.py
    /// op.set_forwarded(newop)`, `OptUnroll.import_state`). Idempotent — re-running
    /// re-mirrors each slot to the same `InputArgRc`.
    pub(crate) fn ensure_inputarg_bindings(&mut self) {
        // Reuse a reminted / recorder InputArg already in `input_ops`
        // (`inputarg_from_tp` / `history.inputargs`) so the mint below
        // does not create a second object for the same index.
        for op in &self.input_ops {
            for arg in op.args_slice() {
                if let Some(ia) = arg.bound_inputarg() {
                    self.inputarg_refs.entry(ia.index).or_insert(ia);
                }
            }
            if let Some(failargs) = op.guard_fail_args() {
                for arg in failargs {
                    if let Some(ia) = arg.bound_inputarg() {
                        self.inputarg_refs.entry(ia.index).or_insert(ia);
                    }
                }
            }
        }
        // Derive the materialized InputArg positions from `ctx` state.
        // The InputArg positions are exactly the
        // canonical/inherited set (`self.inputargs` = `optimizer.py
        // self.inputargs`, positions `[0, num_inputs)` carried across Phase 1 →
        // Phase 2 by `opt_p2.trace_inputargs = self.trace_inputargs`) UNION the
        // fresh per-iteration label set at `[inputarg_base, inputarg_base +
        // num_inputs)` (`TraceIterator` allocates these; their types match
        // `self.inputargs` because Phase 2 walks the body half with the same
        // per-arg types). Pre-populating `inputarg_refs` for both subsets makes
        // every InputArg OpRef resolve through `inputarg_refs` (read path:
        // `resolve_to_operand` / `read_forwarded`; write path: `materialize_operand_at`'s
        // InputArg branch). `materialize_operand_at` type-repairs any position this derive
        // misses. Both loops no-op when `self.inputargs` is empty
        // (`seed_boxes_canonical` fixtures populate `inputarg_refs` directly).
        // Void slots are skipped: `InputArg{Int,Ref,Float}` has no Void
        // encoding (resoperation.py InputArgInt/727/739), so a Void sentinel in
        // `inputargs` is not a real input-arg host and carries no binding.
        for op in self.inputargs.clone() {
            match op.ty() {
                Some(tp) if tp != majit_ir::Type::Void => {
                    self.bind_canonical_inputarg(op.raw() as usize, tp);
                }
                _ => {}
            }
        }
        let base = self.inputarg_base as usize;
        for i in 0..self.num_inputs as usize {
            match self.inputargs.get(i).and_then(|op| op.ty()) {
                Some(tp) if tp != majit_ir::Type::Void => {
                    self.bind_canonical_inputarg(base + i, tp);
                }
                _ => {}
            }
        }
    }

    /// Ensure `inputarg_refs[pos]` holds a canonical `InputArgRc` of type
    /// `tp` (the `_forwarded` host that `resolve_to_operand` / `read_forwarded`
    /// / `clear_forwarded` / `materialize_operand_at` route the matching InputArg OpRef
    /// through). Idempotent: keeps an existing same-shape host (preserving any
    /// `_forwarded` chain and the chain targets naming it) and only
    /// (re)allocates when the slot is absent or its type/index mismatch (mirrors
    /// the `materialize_operand_at` InputArg arm).
    fn bind_canonical_inputarg(&mut self, pos: usize, tp: majit_ir::Type) {
        let pos = pos as u32;
        match self.inputarg_refs.get(&pos) {
            Some(ia) if ia.tp.get() == tp && ia.index == pos => {}
            _ => {
                self.inputarg_refs.insert(
                    pos,
                    majit_ir::InputArgRc::new(majit_ir::InputArg::from_type(tp, pos)),
                );
            }
        }
    }

    /// Find the canonical producer `OpRc`
    /// for `opref` by scanning the current phase's `new_operations`
    /// first, then cross-phase `phase1_emit_ops` (`history.py:220`
    /// box.type parity for Phase 1 emit OpRefs visible from Phase 2),
    /// then synthetic stand-ins / pre-bound input ops in
    /// `resop_refs`. Reverse scan on the first two lists mirrors
    /// `op_at`'s ordering so a later replacement at the same `pos`
    /// wins. Returns `None` for inputargs, constants, OpRefs without
    /// a producer in any of the three stores, and sentinel
    /// `OpRef::none()`.
    pub(crate) fn find_producer_op(&self, opref: OpRef) -> Option<majit_ir::OpRc> {
        if opref.is_none() || opref.is_constant() {
            return None;
        }
        // resoperation.py AbstractInputArg is not an AbstractResOp.
        // Its canonical owner is inputarg_refs, not any result-op store.
        if matches!(
            opref,
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_)
        ) {
            return None;
        }
        if let Some(op) = self.new_operations_index.get(&opref).cloned() {
            return Some(op);
        }
        if let Some(op) = self.phase1_emit_ops_index.get(&opref).cloned() {
            return Some(op);
        }
        self.resop_refs.get(&opref).cloned()
    }

    /// Append an emitted op and mirror it into `new_operations_index` so
    /// `find_producer_op`'s highest-priority lookup stays O(1). Insert
    /// overwrites, so the last push at a given position wins — matching the
    /// `iter().rfind(...)` (last-occurrence) scan the index replaces.
    pub(crate) fn push_new_operation(&mut self, op: majit_ir::OpRc) {
        self.new_operations_index.insert(op.pos().get(), op.clone());
        self.new_operations.push(op);
    }

    /// Overwrite an already-emitted op in place (the guard-strengthening
    /// replacements of `optimizer.py:716-718` / `rewrite.py:343`) and mirror
    /// the new op into `new_operations_index`, so the index keeps agreeing
    /// with `new_operations` for the rest of the forward phase.
    pub(crate) fn replace_new_operation(&mut self, idx: usize, op: majit_ir::OpRc) {
        self.new_operations_index.insert(op.pos().get(), op.clone());
        self.new_operations[idx] = op;
    }

    /// Rebuild `new_operations_index` from the current `new_operations`.
    /// Called after the rare structural mutations (tail truncate, jump
    /// reorder) that the incremental `push_new_operation` maintenance cannot
    /// track by itself.
    pub(crate) fn rebuild_new_operations_index(&mut self) {
        self.new_operations_index.clear();
        self.new_operations_index.reserve(self.new_operations.len());
        for op in &self.new_operations {
            self.new_operations_index.insert(op.pos().get(), op.clone());
        }
    }

    /// Rebuild `phase1_emit_ops_index` from `phase1_emit_ops`. Called once
    /// at the Phase 1→2 handoff right after `phase1_emit_ops` is assigned;
    /// the field is never mutated afterwards. Forward iteration with
    /// `insert` makes the last occurrence at each position win, matching the
    /// `rfind` the index replaces.
    pub(crate) fn rebuild_phase1_emit_ops_index(&mut self) {
        self.phase1_emit_ops_index.clear();
        self.phase1_emit_ops_index
            .reserve(self.phase1_emit_ops.len());
        for op in &self.phase1_emit_ops {
            self.phase1_emit_ops_index
                .insert(op.pos().get(), op.clone());
        }
    }

    /// Append `op` to `live_synthetics`. Recorder ops are not stored here
    /// (`bind_input_resops` keeps them in `resop_refs` only); this set is
    /// `mint_synthetic_resop` / extra-producer stand-ins.
    pub(crate) fn live_synthetics_push(&mut self, op: majit_ir::OpRc) {
        self.live_synthetics.push(op);
    }

    /// Remove the live synthetic/extra at position `pos` (if any).
    pub(crate) fn live_synthetics_swap_remove_pos(&mut self, pos: OpRef) -> Option<majit_ir::OpRc> {
        let i = self
            .live_synthetics
            .iter()
            .position(|op| op.pos().get() == pos)?;
        Some(self.live_synthetics.swap_remove(i))
    }

    /// Mint a `SameAsI/F/R` (or `Jump` for `Void`) synthetic
    /// stand-in `OpRc` for `opref` with the correct result type and
    /// stash it in `resop_refs[opref]` so `emit()`'s
    /// `bound_is_synthetic` rebind path later upgrades the binding to
    /// the real producer via `bind_op`'s carry-over. The synthetic
    /// stays referenced from `resop_refs` for the OptContext's
    /// lifetime so a repeat lookup of the position resolves to the same
    /// `_forwarded` host until rebind.
    pub(crate) fn mint_synthetic_resop(
        &mut self,
        opref: OpRef,
        ty: majit_ir::Type,
    ) -> majit_ir::OpRc {
        use majit_ir::resoperation::{Op, OpCode};
        let opcode = match ty {
            majit_ir::Type::Int => OpCode::SameAsI,
            majit_ir::Type::Float => OpCode::SameAsF,
            majit_ir::Type::Ref => OpCode::SameAsR,
            majit_ir::Type::Void => OpCode::Jump,
        };
        let synthetic = OpRc::new(Op::new(opcode, &[]));
        synthetic.pos().set(opref);
        self.resop_refs.insert(opref, synthetic.clone());
        self.live_synthetics_push(synthetic.clone());
        synthetic
    }

    /// `replace_op_with` (optimizer.py:replace_op_with) parity for the
    /// dispatcher's `Restart` arm: the rewritten op supersedes the original
    /// as the producer at its (preserved) position. RPython performs
    /// `original.set_forwarded(newop)`, after which every read of that
    /// position resolves to `newop` and `setintbound`/`setptrinfo` writes
    /// decorate `newop` directly. In pyre's producer-registry model the
    /// equivalent is to make `restart_op` the SOLE registered producer at
    /// its position: `find_producer_op(pos)` then returns `restart_op`, so
    /// `materialize_operand_at(pos)` / `from_bound_op(restart_op)` agree on one canonical
    /// `_forwarded` host, and the bound the re-dispatch accumulates survives
    /// `emit`'s `live_synthetics` catch-up onto the emitted op.
    ///
    /// Without this the re-dispatch runs against a fresh, unregistered
    /// `Rc<Op>` (`Rc::new(restart_op)`): writes such as `setintbound` land on
    /// a host absent from `live_synthetics`, the superseded original keeps its
    /// stale stand-in slot, and the carry-over at emit migrates the wrong
    /// (empty) `_forwarded` — dropping the rewrite's bound.
    pub(crate) fn supersede_restart_producer(&mut self, restart_op: &majit_ir::OpRc) {
        self.install_canonical_producer(restart_op);
    }

    /// Make `op` the sole registered producer at its result position,
    /// inheriting any superseded stand-in's accumulated `_forwarded`.
    ///
    /// optimizer.py `replace_op_with`: after `newop =
    /// op.copy_and_change(..)` it carries the prior host's info forward —
    /// `opinfo = get_box_replacement(op).get_forwarded(); if opinfo is not
    /// None: newop.set_forwarded(opinfo)` — and only then `op.set_forwarded(
    /// newop)` makes `newop` the single box at that position. In pyre's
    /// producer registry the equivalent is to drop the stand-in
    /// `live_synthetics` entry, migrate its `_forwarded` onto `op` (so the
    /// `IntBound`/`PtrInfo`/const facts accumulated before the rewrite survive),
    /// then register `op` as the lone producer. The blind `_forwarded` clone
    /// mirrors `emit`'s live_synthetics catch-up: a producer-less stand-in only
    /// ever holds `None`/`Info`, never a redirect.
    fn install_canonical_producer(&mut self, op: &majit_ir::OpRc) {
        let pos = op.pos().get();
        if pos.is_none() || pos.is_constant() {
            return;
        }
        // InputArg positions have no producing op (handled by
        // `ensure_inputarg_bindings`); a rewrite never targets one.
        if matches!(
            pos,
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_)
        ) {
            return;
        }
        // Drop the superseded stand-in at this position and carry its
        // accumulated `_forwarded` onto `op` (replace_op_with's `opinfo`
        // hand-off). The current box is a live synthetic if one was minted,
        // otherwise the recorder `OpRc` in `resop_refs` (`optimizer.py`
        // `replace_op_with` / `op.set_forwarded(newop)`). The `ptr_eq`
        // guard skips the self-clone — and its `RefCell` double-borrow —
        // when `op` is already the registered stand-in.
        if let Some(superseded) = self
            .live_synthetics_swap_remove_pos(pos)
            .or_else(|| self.resop_refs.get(&pos).cloned())
            && !OpRc::ptr_eq(&superseded, op)
        {
            let carried = superseded.forwarded().borrow().clone();
            // An Op redirect is not info to carry: when it names `op` (a
            // stand-in already forwarded to the op now taking its position),
            // copying it closes a one-node cycle.
            if !matches!(carried, majit_ir::forwarding::Forwarded::Op(_)) {
                *op.forwarded().borrow_mut() = carried;
            }
            // replace_op_with parity (optimizer.py): forward the
            // superseded stand-in to `op`. A consumer dispatched before
            // this supersession bound its operand to `superseded` (the
            // producer registered at its dispatch-entry rebind); without
            // this link its box-native walk freezes on the orphaned
            // stand-in while `find_producer_op` resolves `op`, so a later
            // operand fold lands on a different host than the OpRef path
            // observes. Mirrors `emit`'s live-synthetic catch-up so a
            // supersession chain (stand-in → extra-producer → emitted op)
            // stays transitively resolvable to the final producer.
            use majit_ir::forwarding::ForwardingHost;
            superseded.set_forwarded_op(op);
        }
        self.resop_refs.insert(pos, op.clone());
        self.live_synthetics_push(op.clone());
    }

    /// Read `_forwarded` for `opref` directly off the canonical
    /// host (`op.forwarded` / `inputarg.forwarded`). Mirrors
    /// `resoperation.py get_forwarded` semantics
    /// but bypasses the wrapper allocation. Returns `Forwarded::None`
    /// for constants (`resoperation.py` `Const._forwarded` is
    /// permanently `None`), `None` for sentinel `OpRef::none()` and
    /// for ResOp positions whose producer is not in any canonical
    /// store (`new_operations` / `phase1_emit_ops` / `resop_refs`).
    pub(crate) fn read_forwarded(&self, opref: OpRef) -> Option<majit_ir::forwarding::Forwarded> {
        if opref.is_none() {
            return None;
        }
        if opref.is_constant() {
            return Some(majit_ir::forwarding::Forwarded::None);
        }
        match opref {
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) => self
                .inputarg_refs
                .get(&opref.raw())
                .map(|ia| ia.forwarded.borrow().clone()),
            _ => self
                .find_producer_op(opref)
                .map(|op| op.forwarded().borrow().clone()),
        }
    }

    /// Resolve a position to its canonical producer as an [`Operand`]
    /// (`Op` / `InputArg` `_forwarded` host, or inline-`Const`). The bound
    /// handle ensures every `set_forwarded_*` / `get_forwarded` routes
    /// through the same `Op.forwarded` / `InputArg.forwarded` slot, so two
    /// calls for the same `opref` observe each other's writes via the
    /// canonical host. Const variants return `Operand::const_from_value`;
    /// returns `None` for sentinel `OpRef::none()` and for ResOp positions
    /// without a producer in any canonical store. Production paths populate
    /// `inputarg_refs` via `bind_input_resops` plus emit-time `bind_op`, so
    /// every chain-walker-reachable position resolves to its bound producer.
    pub(crate) fn resolve_to_operand(&self, opref: OpRef) -> Option<Operand> {
        if opref.is_none() {
            return None;
        }
        if opref.is_constant() {
            if let Some(cached) = self.const_operands.borrow().get(&opref).cloned() {
                return Some(cached);
            }
            let minted = match opref {
                OpRef::ConstInt(v) => Some(Operand::const_from_value(Value::Int(v))),
                OpRef::ConstFloat(v) => Some(Operand::const_from_value(Value::Float(v))),
                OpRef::ConstPtr(v) => Some(Operand::const_from_value(Value::Ref(
                    majit_ir::const_ptr_table::resolve(v),
                ))),
                _ => None,
            };
            if let Some(ref op) = minted {
                self.const_operands.borrow_mut().insert(opref, op.clone());
            }
            return minted;
        }
        // `resoperation.py AbstractInputArg` is not an `AbstractResOp`;
        // `inputarg_refs` is the owner. Classify before `find_producer_op`
        // so InputArg positions never pay the producer-map lookup
        // (`optimizer.py Optimizer.get_box_replacement` walks the box).
        match opref {
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) => {
                let idx = opref.raw() as usize;
                // An unbound position reports "no binding" rather than a
                // box, which makes `get_replacement_opref` return the
                // position itself — `resoperation.py:57-68
                // get_box_replacement`, where a box with no `_forwarded`
                // resolves to itself. The index test is the invariant the
                // write paths maintain, not a filter: a key never holds a box
                // for a different position.
                if let Some(ia) = self.inputarg_refs.get(&(idx as u32))
                    && ia.index as usize == idx
                {
                    return Some(Operand::from_bound_inputarg(ia));
                }
                return None;
            }
            _ => {}
        }
        if let Some(op) = self.find_producer_op(opref) {
            return Some(Operand::from_bound_op(&op));
        }
        None
    }

    /// Write `Forwarded::None` to the canonical host for
    /// `opref` (`resoperation.py` `set_forwarded(None)` /
    /// `:50` clear semantics). No-op for sentinel `OpRef::none()`,
    /// constants (whose `_forwarded` is permanently `None`), and
    /// positions without a canonical Op/InputArg.
    pub(crate) fn clear_forwarded(&self, opref: OpRef) {
        if opref.is_none() || opref.is_constant() {
            return;
        }
        match opref {
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) => {
                if let Some(ia) = self.inputarg_refs.get(&opref.raw()) {
                    *ia.forwarded.borrow_mut() = majit_ir::forwarding::Forwarded::None;
                }
            }
            _ => {
                if let Some(op) = self.find_producer_op(opref) {
                    *op.forwarded().borrow_mut() = majit_ir::forwarding::Forwarded::None;
                }
            }
        }
    }

    /// Test-only: seed the canonical producer stores (`resop_refs` /
    /// `inputarg_refs`) from a list of already-bound operands, mirroring
    /// what the production recorder→optimizer handoff populates. Each box
    /// is distributed by its bound identity: InputArg boxes land in
    /// `inputarg_refs[index]`, ResOp boxes in `resop_refs[pos]`, so that
    /// `resolve_to_operand` / `materialize_operand_at` / `find_producer_op` resolve each
    /// OpRef through the same canonical hosts production uses, returning a
    /// fresh operand bound to the seeded `Op` / `InputArg`.
    #[cfg(test)]
    pub(crate) fn seed_boxes_canonical(&mut self, operands: &[Operand]) {
        for o in operands {
            if let Some(ia) = o.bound_inputarg() {
                self.inputarg_refs.insert(ia.index, ia);
            } else if let Some(op) = o.bound_op() {
                self.resop_refs.insert(op.pos().get(), op);
            }
        }
    }

    /// Record every input op's resop producer `OpRc` into
    /// `resop_refs` so any `Forwarded::Op(_)` chain step targeting the
    /// slot resolves to a bound host from the start of the optimization
    /// run. Absent this pre-pass, `getintbound_box` →
    /// `get_box_replacement_operand_opt` (a `&self` reader) can land on an
    /// unbound terminal and a subsequent `set_forwarded_info` write
    /// trips `write_forwarded`'s bound-precondition assert.
    ///
    /// The producer `OpRc` is stashed in `resop_refs[pos]` so `emit()`
    /// reuses that same box (`optimizer.py Optimizer._emit_operation`
    /// appends `op`).
    ///
    /// InputArg slots are skipped (handled by `ensure_inputarg_bindings`);
    /// only resop positions land here. Each phase's input `ops` carry that
    /// phase's own positions (Phase 2 body ops sit above the Phase 1 emit
    /// namespace, so they never collide with an inherited bound slot), and
    /// the `resop_refs[opref]` dedup covers intra-`ops` repeats.
    pub(crate) fn bind_input_resops(&mut self, ops: &[majit_ir::OpRc]) {
        // The loop over the caller-threaded `ops` self-guards (no-op on an
        // empty slice) and records each resop producer into `resop_refs` —
        // the iterator object is the box (`opencoder.py TraceIterator.next`
        // `_cache`; `optimizer.py Optimizer._emit_operation` appends it).
        for op in ops {
            // Bridge / Phase-2 reminted InputArgs live on the op
            // (`inputarg_from_tp`). `inputarg_base != 0` is that reminted
            // namespace; reuse those Rcs instead of the stand-in
            // `ensure_inputarg_bindings` just minted. `inputarg_base == 0`
            // keeps test fixtures that seed `inputarg_refs` separately
            // from per-call `bound_arg` mints.
            if self.inputarg_base != 0 {
                for arg in op.args_slice() {
                    if let Some(ia) = arg.bound_inputarg() {
                        self.inputarg_refs.insert(ia.index, ia);
                    }
                }
                if let Some(failargs) = op.guard_fail_args() {
                    for arg in failargs {
                        if let Some(ia) = arg.bound_inputarg() {
                            self.inputarg_refs.insert(ia.index, ia);
                        }
                    }
                }
            }
            let pos = op.pos().get();
            if pos.is_none() || pos.is_constant() {
                continue;
            }
            // InputArg slots are handled by `ensure_inputarg_bindings`; only
            // resop positions land in `resop_refs`.
            if matches!(
                pos,
                OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_)
            ) {
                continue;
            }
            // Dedup: a producer for this exact pos is already recorded.
            if self.resop_refs.contains_key(&pos) {
                continue;
            }
            // Record the caller-threaded `OpRc` so the iterated input op
            // and `resop_refs` share one `Op` identity (no second private
            // copy). Not pushed to `live_synthetics`: that set is only
            // synthetics / extra producers.
            self.resop_refs.insert(pos, op.clone());
        }
    }

    /// `replace_op_with`: carry the terminal's `Info` onto `new_op`, then
    /// `old.set_forwarded(new_op)` except for a void result. `ptr_eq` skips a
    /// one-node cycle, including when the terminal walk already landed on
    /// `new_op`.
    pub(crate) fn link_replaced_producer(old: &majit_ir::OpRc, new_op: &majit_ir::OpRc) {
        use majit_ir::forwarding::ForwardingHost;
        if OpRc::ptr_eq(old, new_op) {
            return;
        }
        // optimizer.py `replace_op_with`: a void result is not forwarded.
        if new_op.result_type() == Type::Void {
            return;
        }
        let terminal = Operand::from_bound_op(old).get_box_replacement(false);
        let Some(term_op) = terminal.bound_op() else {
            return;
        };
        if OpRc::ptr_eq(&term_op, new_op) {
            return;
        }
        let opinfo = term_op.get_forwarded();
        if matches!(opinfo, majit_ir::forwarding::Forwarded::Info(_))
            && matches!(
                new_op.get_forwarded(),
                majit_ir::forwarding::Forwarded::None
            )
        {
            new_op.store_forwarded(opinfo);
        }
        term_op.set_forwarded_op(new_op);
    }

    /// Construct an `OptContext` whose inputarg / fresh-OpRef numbering is
    /// shifted to start above a parent trace's high water mark.
    ///
    /// `inputarg_base` corresponds to RPython's `start = trace._start` for
    /// `TraceIterator`: it is the smallest OpRef this iteration may use, and
    /// inputargs occupy `[inputarg_base, inputarg_base + num_inputs)`.
    /// `start_next_pos` is the value of `_index` after the inputargs were
    /// pre-allocated, i.e. the first fresh OpRef the optimizer will assign
    /// to a non-void op result. Phase 1 / standalone passes use
    /// `inputarg_base = 0`, `start_next_pos = num_inputs`; Phase 2 / bridges
    /// pass `inputarg_base = parent_high_water`,
    /// `start_next_pos = parent_high_water + num_inputs`.
    pub fn with_num_inputs_and_start_pos(
        estimated_ops: usize,
        num_inputs: usize,
        inputarg_base: u32,
        start_next_pos: u32,
    ) -> Self {
        OptContext {
            supports_efficient_uint_mul_high: true,
            new_operations: Vec::with_capacity(estimated_ops),
            new_operations_drained: false,
            new_operations_index: FxHashMap::with_capacity_and_hasher(
                estimated_ops,
                Default::default(),
            ),
            emitted_operations: indexmap::IndexSet::with_capacity_and_hasher(
                estimated_ops,
                rustc_hash::FxBuildHasher,
            ),
            num_inputs: num_inputs as u32,
            inputarg_base,
            next_pos: start_next_pos,
            extra_operations_after: ExtraQueue::new(),
            extra_pending: Vec::new(),
            pending_guard_class_postprocess: None,
            pending_mark_last_guard: None,
            pending_finish_guard_postprocess: None,
            imported_short_pure_ops: Vec::new(),
            imported_virtual_args: None,
            imported_loop_invariant_results: Vec::new(),
            imported_short_preamble_builder: None,
            const_infos: crate::FxIndexMap::default(),

            potential_extra_ops: indexmap::IndexMap::new(),
            active_short_preamble_producer: None,
            preview_short_state: None,
            exported_const_short_boxes: Vec::new(),

            imported_virtuals: Vec::new(),
            imported_label_args: None,
            can_replace_guards: true,
            patchguardop: None,
            preamble_end_args: None,
            skip_flush_mode: false,
            building_bridge: false,
            current_pass_idx: 0,
            optearlyforce_idx: 0,

            in_final_emission: false,
            force_direct: false,
            callinfocollection: None,
            resumedata_memo: std::rc::Rc::new(std::cell::RefCell::new(
                crate::resume::ResumeDataLoopMemo::new(),
            )),
            pending_for_guard: Vec::new(),
            pending_pure_from_args: Vec::new(),
            pending_pure_from_args2: Vec::new(),
            pure_ops: None,
            constant_fold_alloc: None,
            string_length_resolver: None,
            string_content_resolver: None,
            string_constant_alloc: None,
            quasi_immutable_deps: Vec::new(),
            snapshot_boxes: Vec::new(),
            snapshot_frame_sizes: SnapshotFrameSizes::new(),
            snapshot_vable_boxes: Vec::new(),
            // resume.py:401-402: `-1` until an Optimizer built with a vable
            // config overwrites it in `optimize_with_constants_and_inputs_at`.
            minimum_virtualizable_size: -1,
            snapshot_vref_boxes: Vec::new(),
            snapshot_frame_pcs: SnapshotFramePcs::new(),
            byte_bridge_resume: None,

            inputargs: Vec::new(),
            inputarg_refs: FxHashMap::with_capacity_and_hasher(
                estimated_ops.min(32),
                Default::default(),
            ),
            resop_refs: indexmap::IndexMap::with_capacity_and_hasher(
                estimated_ops,
                Default::default(),
            ),
            live_synthetics: Vec::new(),
            phase1_emit_ops: Vec::new(),
            phase1_emit_ops_index: FxHashMap::default(),
            input_ops: Vec::with_capacity(estimated_ops),
            last_guard_idx: None,
            guard_chain_broken: false,
            last_seen_snapshot_pos: None,
            cpu: crate::cpu::default_cpu(),
            remove_gctypeptr: true,
            last_op_removed: false,
            pending_invalid_loop: std::cell::Cell::new(None),
            const_operands: std::cell::RefCell::new(crate::FxIndexMap::default()),
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.num_inputs as usize
    }

    /// Allocate a fresh OpRef position with the producer's result type
    /// stamped on the variant tag. Callers always know `box.type` —
    /// the resulting OpRef is recognized at priority 0 by `opref_type` /
    /// `OptBoxEnv::get_type`, and there is no type side-table to grow
    /// for these positions.
    pub fn alloc_op_position_typed(&mut self, tp: majit_ir::Type) -> OpRef {
        self.reserve_pos_typed(tp)
    }

    /// Allocate a fresh OpRef position and eagerly mint its canonical
    /// `_forwarded` host — a `SameAs*`/`Jump` synthetic in `resop_refs` —
    /// returning both the position and an operand bound to that host.
    ///
    /// This is the explicit creation primitive for producer-less
    /// synthetics: importers that allocate a position purely to carry a
    /// forwarded write (PtrInfo / IntBound / Const for an imported virtual
    /// state leaf) get a bound write target in one step. The minted
    /// synthetic is the `ResOperation(...)` analog (`mint_synthetic_resop`);
    /// a later `emit()` for the same position supersedes it through the
    /// same `live_synthetics` catch-up. Reserve a bare position via
    /// `alloc_op_position_typed` instead when no forwarded write or read
    /// follows (e.g. an `Unknown` leaf), to avoid an eager synthetic for a
    /// position that is never written.
    /// Explicit create: mint a `SameAs*` synthetic at `opref` and return an
    /// operand bound to it, so a subsequent `set_forwarded_*` lands on the
    /// canonical `Op._forwarded` host. `opref` must be a non-const,
    /// non-sentinel resop position whose producer is not yet emitted (a
    /// virgin alias). Callers reach this only on the `None` arm of
    /// `get_box_replacement_operand_opt`; an already-minted or
    /// producer-backed opref resolves there.
    pub(crate) fn mint_box_at(&mut self, opref: OpRef) -> Operand {
        let tp = opref.ty().unwrap_or(majit_ir::Type::Void);
        let synthetic = self.mint_synthetic_resop(opref, tp);
        Operand::from_bound_op(&synthetic)
    }

    pub(crate) fn reserve_virtual_box(&mut self, tp: majit_ir::Type) -> (OpRef, Operand) {
        let opref = self.reserve_pos_typed(tp);
        let b = self.mint_box_at(opref);
        (opref, b)
    }

    /// Dispatch on a `Value`'s type tag and produce a typed `*Op` OpRef
    /// at the given position (resoperation.py IntOp
    /// IntOp/FloatOp/RefOp/VoidOp mixins).
    pub(crate) fn op_ref_for_value(pos: u32, value: &Value) -> OpRef {
        OpRef::op_typed(pos, value.get_type())
    }

    /// info.py emit during force_box: routes through emit_extra
    /// normally, or direct emit when in_final_emission is true.
    pub fn emit_for_force(&mut self, op: Op) -> OpRef {
        self.emit_for_force_with(&mut None, op)
    }

    /// `emit_for_force` with the owning `Optimizer`, so a force-time STRLEN
    /// or string store follows `Optimization.emit_extra` immediately
    /// (`emit_extra_immediate`) instead of parking on `extra_operations_after`.
    /// `force_direct` or `in_final_emission` appends via `Optimizer.emit_extra`
    /// (`emit` / `emit_rc`) and does not force arguments here — arg forcing
    /// stays in `force_box`'s `emit_op_rc`.
    pub(crate) fn emit_for_force_with(
        &mut self,
        optimizer: &mut Option<&mut optimizer::Optimizer>,
        op: Op,
    ) -> OpRef {
        self.emit_for_force_rc_with(optimizer, majit_ir::OpRc::new(op))
    }

    /// `emit_for_force` when the caller keeps the op as its box.
    pub fn emit_for_force_rc(&mut self, op: majit_ir::OpRc) -> OpRef {
        self.emit_for_force_rc_with(&mut None, op)
    }

    /// `emit_for_force_rc` with the owning `Optimizer`. See `emit_for_force_with`.
    pub(crate) fn emit_for_force_rc_with(
        &mut self,
        optimizer: &mut Option<&mut optimizer::Optimizer>,
        op: majit_ir::OpRc,
    ) -> OpRef {
        let direct = self.in_final_emission || self.force_direct;
        if direct {
            self.emit_rc(op)
        } else if let Some(opt) = optimizer.as_deref_mut() {
            // `vstring.py` `optstring.emit_extra` / `Optimization.emit_extra`:
            // same entry index as `emit_extra_rc` (`current_pass_idx + 1`).
            if let Err(crate::optimize::InvalidLoop(reason)) =
                opt.emit_extra_immediate(self.current_pass_idx, &op, self)
            {
                self.signal_invalid_loop(reason);
            }
            op.pos().get()
        } else {
            self.emit_extra_rc(self.current_pass_idx, op)
        }
    }

    /// optimizer.py new_const_item(arraydescr) — default value for
    /// the given item type.
    pub fn new_const_item(&mut self, item_type: Type) -> OpRef {
        match item_type {
            Type::Int | Type::Void => self.make_constant_int(0),
            Type::Ref => self.make_constant_ref(GcRef::NULL),
            Type::Float => self.make_constant_float(0.0),
        }
    }

    /// optimizer.py new_const(fielddescr) — default value for an
    /// unset field of a freshly allocated virtual struct.  GC allocations are
    /// zero-filled, so optimizeopt can answer the read without materialising
    /// the virtual.
    pub fn new_const(&mut self, field_type: Type) -> OpRef {
        match field_type {
            Type::Int | Type::Void => self.make_constant_int(0),
            Type::Ref => self.make_constant_ref(GcRef::NULL),
            Type::Float => self.make_constant_float(0.0),
        }
    }

    /// vstring.py getstrlen / 171-175 / 251-253 / 281-295
    /// Per-subclass getstrlen() dispatch — returns a cached lgtop box if
    /// available, or computes/emits the length and caches in StrPtrInfo.lgtop.
    /// Always returns a box, never an i64 summary.
    ///
    /// Delegates to `getstrlen_for(opref, opref, mode)`.
    pub fn getstrlen_opref(&mut self, opref: OpRef, mode: u8) -> Operand {
        self.getstrlen_opref_with(&mut None, opref, mode)
    }

    /// `getstrlen_opref` with the owning `Optimizer` (`vstring.py`
    /// `optstring.emit_extra` for the STRLEN, via `emit_for_force_rc_with`).
    pub(crate) fn getstrlen_opref_with(
        &mut self,
        optimizer: &mut Option<&mut optimizer::Optimizer>,
        opref: OpRef,
        mode: u8,
    ) -> Operand {
        self.getstrlen_for_with(optimizer, opref, opref, mode)
    }

    /// vstring.py StrPtrInfo.getstrlen(op, optstring, mode)
    ///
    /// Matches RPython's method dispatch where `self` (info) and `op` may
    /// differ: info lookup comes from `info_opref`, but the fallback STRLEN
    /// emission uses `op_opref`.  Cached lgtop is stored on `info_opref`'s
    /// PtrInfo.
    ///
    /// When both are the same, use `getstrlen_opref(opref, mode)` instead.
    pub fn getstrlen_for(&mut self, info_opref: OpRef, op_opref: OpRef, mode: u8) -> Operand {
        self.getstrlen_for_with(&mut None, info_opref, op_opref, mode)
    }

    /// `getstrlen_for` with the owning `Optimizer`. Entry index stays
    /// `current_pass_idx + 1` (`emit_for_force`); it is not retargeted at
    /// `optstring`.
    pub(crate) fn getstrlen_for_with(
        &mut self,
        optimizer: &mut Option<&mut optimizer::Optimizer>,
        info_opref: OpRef,
        op_opref: OpRef,
        mode: u8,
    ) -> Operand {
        let resolved_box = self.get_box_replacement_operand_opt(info_opref);
        // vstring.py:112/283: if self.lgtop is not None: return self.lgtop
        if let Some(info) = resolved_box.as_ref().and_then(|b| self.getptrinfo(b))
            && let Some(lgtop) = info.get_cached_lgtop()
        {
            return lgtop;
        }
        // vstring.py:174/253: constant or structurally-known length
        let known_len = resolved_box
            .as_ref()
            .and_then(|b| self.getptrinfo(b))
            .and_then(|info| info.get_known_str_length(self, mode));
        if let Some(len) = known_len {
            let len_box = Operand::const_(majit_ir::Const::Int(len));
            if let Some(b) = self.get_box_replacement_operand_opt(info_opref) {
                self.set_str_lgtop(&b, len_box.clone());
            }
            return len_box;
        }
        // vstring.py: VStringConcatInfo.getstrlen — recursive
        // dispatch: getstrlen on each child, then _int_add.
        // Borrow-checker adaptation: extract vleft/vright before &mut self calls.
        let concat_children = resolved_box
            .as_ref()
            .and_then(|b| self.getptrinfo(b))
            .and_then(|info| {
                use crate::optimizeopt::info::VStringVariant;
                if let PtrInfo::Str(sinfo) = info
                    && let VStringVariant::Concat(c) = sinfo.variant
                {
                    return Some((c.vleft.to_opref(), c.vright.to_opref()));
                }
                None
            });
        if let Some((vleft, vright)) = concat_children {
            // vstring.py:286-293
            let left_len = self.getstrlen_for_with(optimizer, vleft, vleft, mode);
            let right_len = self.getstrlen_for_with(optimizer, vright, vright, mode);
            let result =
                crate::optimizeopt::vstring::_int_add_with(optimizer, &left_len, &right_len, self);
            // vstring.py: self.lgtop = _int_add(optstring, len1box, len2box)
            if let Some(b) = self.get_box_replacement_operand_opt(info_opref) {
                self.set_str_lgtop(&b, result.clone());
            }
            return result;
        }
        // vstring.py: base StrPtrInfo — emit STRLEN/UNICODELEN
        // RPython: lengthop = ResOperation(mode.STRLEN, [op])
        // `op` comes from op_opref (the first arg to getstrlen in RPython).
        let op_resolved = self.get_replacement_opref(op_opref);
        let strlen_opcode = if mode != 0 {
            majit_ir::OpCode::Unicodelen
        } else {
            majit_ir::OpCode::Strlen
        };
        let arg1 = self.materialize_operand_at(op_resolved);
        let lengthop = majit_ir::OpRc::new(majit_ir::Op::new(strlen_opcode, &[arg1]));
        // vstring.py StrPtrInfo.getstrlen: `optstring.emit_extra(lengthop)`.
        // Optimization.emit_extra → send_extra_operation down the rest of
        // the chain; during force (`in_final_emission`) that lands in
        // Optimizer.emit immediately, so the STRLEN precedes NEWSTR.
        // `emit_for_force_rc` is that routing. `emit_extra_rc` queues the
        // STRLEN after the NEWSTR/COPYSTRCONTENT that consume it.
        self.emit_for_force_rc_with(optimizer, lengthop.clone());
        let result = Operand::from_bound_op(&lengthop);
        // vstring.py: lengthop.set_forwarded(self.getlenbound(mode)). The
        // lenbound is lazily installed on the StrPtrInfo, a PtrInfo-internal
        // mutation.
        let lenbound = self
            .get_box_replacement_operand_opt(info_opref)
            .as_ref()
            .and_then(|b| self.get_str_lenbound(b));
        if let Some(bound) = lenbound {
            self.setintbound(&result, &bound);
        }
        // vstring.py:117: self.lgtop = lengthop
        if let Some(b) = self.get_box_replacement_operand_opt(info_opref) {
            self.set_str_lgtop(&b, result.clone());
        }
        result
    }

    /// `vstring.py:117/174/293 self.lgtop = lengthop` — cache the length
    /// box in `StrPtrInfo.lgtop`. Direct PtrInfo field write,
    /// unconditional per `info.py`.
    ///
    /// `op: &Operand` is the StrPtrInfo-bearing box; `lgtop` is the length
    /// box.
    pub(crate) fn set_str_lgtop(&mut self, op: &Operand, lgtop: Operand) {
        // optimizer.py `get_box_replacement` chain walk before mutation.
        let resolved = op.get_box_replacement(false);
        if resolved.is_constant() {
            return;
        }
        self.with_ptr_info_mut(&resolved, |info| {
            if let PtrInfo::Str(si) = info {
                si.lgtop = Some(lgtop.clone());
            }
        });
    }

    /// `vstring.py StrPtrInfo.getlenbound(mode)` — get lenbound
    /// from StrPtrInfo, lazily initializing it from self.length:
    /// ```python
    /// def getlenbound(self, mode):
    ///     if self.lenbound is None:
    ///         if self.length == -1:
    ///             self.lenbound = IntBound.nonnegative()
    ///         else:
    ///             self.lenbound = IntBound.from_constant(self.length)
    ///     return self.lenbound
    /// ```
    /// Const inputs short-circuit (RPython getlenbound is called on a
    /// StrPtrInfo instance, never on a Const).
    fn get_str_lenbound(&self, op: &Operand) -> Option<crate::optimizeopt::intutils::IntBound> {
        // optimizer.py-style chain walk; mirror PyPy `getptrinfo(op)` shape
        // by reading the chain terminal via Operand::get_box_replacement.
        let resolved = op.get_box_replacement(false);
        if resolved.is_constant() {
            return None;
        }
        self.with_ptr_info_mut(&resolved, |info| {
            if let PtrInfo::Str(si) = info {
                // vstring.py:65-70
                if si.lenbound.is_none() {
                    si.lenbound = Some(if si.length == -1 {
                        crate::optimizeopt::intutils::IntBound::nonnegative()
                    } else {
                        crate::optimizeopt::intutils::IntBound::from_constant(si.length as i64)
                    });
                }
                si.lenbound.clone()
            } else {
                None
            }
        })
        .flatten()
    }

    /// Typed `reserve_pos`. Tags the resulting OpRef with the producer's
    /// result type so `AbstractValue.type` parity (resoperation.py)
    /// is set at allocation time. Readers consult
    /// `opref.ty()` at priority 0 in `opref_type` / `OptBoxEnv::get_type`,
    /// so typed positions never grow `value_types`.
    pub(crate) fn reserve_pos_typed(&mut self, tp: majit_ir::Type) -> OpRef {
        let raw = self.allocate_next_pos_raw();
        // Bare position: no producer is minted. Callers that later write
        // `_forwarded` or read the box must use `reserve_virtual_box` /
        // `mint_box_at` so the `ResOperation(...)` analog exists from
        // allocation (`resoperation.py`). An eager synthetic for a position
        // that is reserved but never emitted (label / jump on an empty
        // trace) would leak into `phase1_emit_ops` via `live_synthetics`.
        // PyPy/RPython has no Box for positions that no `ResOperation()` /
        // `InputArg()` call produced (`resoperation.py`).
        OpRef::op_typed(raw, tp)
    }

    /// opencoder.py:271 `_index` parity: floor at the iteration's inputarg
    /// base + num_inputs + emitted-op count, so a freshly allocated raw
    /// position never lands inside the inputarg slice or below the parent
    /// trace's high water mark when called from a Phase 2 / bridge
    /// OptContext.
    fn allocate_next_pos_raw(&mut self) -> u32 {
        self.next_pos = self
            .next_pos
            .max(self.inputarg_base + self.num_inputs + self.new_operations.len() as u32);
        // Skip positions already claimed by a constant forwarding
        // (`make_constant`/`seed_constant`'s `set_forwarded_const` write) —
        // those positions' canonical host is already a constant identity and
        // cannot be reused for a fresh op. Reads the canonical `_forwarded`
        // host for the position (`resop_refs[pos]` / `inputarg_refs[pos]`):
        // `make_constant` writes `Forwarded::Const` to that host
        // (resoperation.py AbstractResOpOrInputArg).
        while self.position_is_const_forwarded(self.next_pos) {
            self.next_pos += 1;
        }
        debug_assert!(
            !OpRef::raw_is_constant(self.next_pos),
            "reserve_pos overflowed into constant namespace: {}",
            self.next_pos
        );
        let raw = self.next_pos;
        self.next_pos += 1;
        raw
    }

    /// Whether the canonical `_forwarded` host for raw position `raw`
    /// (any `resop_refs` entry whose `OpRef` shares this raw, or
    /// `inputarg_refs[raw]` for an InputArg slot) carries `Forwarded::Const`.
    /// The position-keyed const probe used by `allocate_next_pos_raw`.
    fn position_is_const_forwarded(&self, raw: u32) -> bool {
        use majit_ir::forwarding::Forwarded;
        // `resop_refs` is keyed by the full type-tagged `OpRef`; a raw `u32`
        // can host more than one entry (typed vs untyped). Any host at this
        // raw carrying `Forwarded::Const` claims the position.
        //
        // Every key is the value's own `op.pos` (`install_canonical_producer`,
        // `materialize_operand_at`), so the hosts at `raw` are exactly the
        // OpRef variants that carry it. Probing those keys keeps this O(1):
        // scanning `resop_refs` made `allocate_next_pos_raw` linear in the
        // trace, and it runs once per `reserve_pos_typed`, so the whole
        // optimizer emit path was quadratic in trace length.
        let resop_const = [
            OpRef::IntOp(raw),
            OpRef::FloatOp(raw),
            OpRef::RefOp(raw),
            OpRef::VoidOp(raw),
            OpRef::InputArgInt(raw),
            OpRef::InputArgFloat(raw),
            OpRef::InputArgRef(raw),
        ]
        .iter()
        .filter_map(|key| self.resop_refs.get(key))
        .any(|op| {
            matches!(
                op.forwarded().borrow(),
                Forwarded::Const(_) | Forwarded::SmallConst(_) | Forwarded::SmallWide(_)
            )
        });
        let inputarg_const = self.inputarg_refs.get(&raw).is_some_and(|ia| {
            matches!(
                ia.forwarded.borrow(),
                Forwarded::Const(_) | Forwarded::SmallConst(_) | Forwarded::SmallWide(_)
            )
        });
        resop_const || inputarg_const
    }

    /// RPython `box.type` invariant (history.py:220
    /// `InputArg{Int,Ref,Float}.type`, resoperation.py:1693
    /// `opclasses[opnum].type`): every emitted op's intrinsic
    /// `Op.type_` must agree with both the producer's
    /// `OpCode::result_type()` and the `OpRef.pos` variant tag.
    /// Checked at the emit/emit_extra producer surfaces, where both the
    /// `Op` result type and the typed `OpRef` are available.
    fn debug_assert_box_type_invariant(op: &Op) {
        let pos = op.pos().get();
        debug_assert_eq!(
            op.type_,
            op.opcode.result_type(),
            "Op.type_ ({:?}) disagrees with opcode.result_type() ({:?}) at \
             {:?} (opcode={:?}) — dual-source contract violation",
            op.type_,
            op.opcode.result_type(),
            pos,
            op.opcode,
        );
        if let Some(variant_tp) = pos.ty() {
            debug_assert_eq!(
                variant_tp, op.type_,
                "OpRef variant tag ({:?}) disagrees with Op.type_ ({:?}) at \
                 {:?} (opcode={:?}) — typed-factory mismatch \
                 (history.py:220 / resoperation.py:1693 Box.type parity)",
                variant_tp, op.type_, pos, op.opcode,
            );
        }
    }

    /// Emit an operation to the output.
    ///
    /// If the op has no pos assigned (NONE), sets it to `num_inputs + idx`
    /// so the backend's variable numbering stays consistent.
    ///
    /// `optimizer.py Optimizer._emit_operation` appends the same `op`
    /// object it was handed. A recorder / extra / synthetic `OpRc` already
    /// registered at this position is reused; a genuinely new
    /// `ResOperation` (`emit_extra`, `copy_and_change`) is cloned.
    pub fn emit(&mut self, op: Op) -> OpRef {
        self.emit_impl(op)
    }

    /// Append an already-allocated `OpRc` without a second `ResOperation()`.
    /// Used when the optimizer pass-through is the recorder/iterator op
    /// itself (`optimizer.py _newoperations.append(op)`).
    pub(crate) fn emit_rc(&mut self, op: majit_ir::OpRc) -> OpRef {
        if op.pos().get().is_none() || op.pos().get().is_constant() {
            op.pos().set(self.reserve_pos_typed(op.result_type()));
        } else {
            self.next_pos = self.next_pos.max(op.pos().get().raw().saturating_add(1));
        }
        let pos_ref = op.pos().get();
        if op.opcode.is_guard() {
            if !self.in_final_emission {
                let _ = self.emit_guard_operation(&op);
            }
        } else {
            let dominated_by_side_effect = !((op.opcode.has_no_side_effect()
                || op.opcode.is_ovf()
                || op.opcode.is_jit_debug())
                && !Self::is_call_pure_pure_canraise(&op));
            if dominated_by_side_effect {
                self.last_guard_idx = None;
                self.guard_chain_broken = true;
            }
        }
        Self::debug_assert_box_type_invariant(&op);
        // Same live-synthetic hand-off as `emit`'s clone path. A pass-through
        // `OpRc` is often the stand-in itself; the helper's `ptr_eq` guard
        // leaves that producer unlinked to itself.
        self.adopt_live_synthetic(&op);
        self.emitted_operations
            .insert(majit_ir::operand::Operand::from_bound_op(&op));
        self.push_new_operation(op);
        pos_ref
    }

    /// Move the `live_synthetics` stand-in at `op_rc`'s position onto the
    /// producer about to be appended. Copies the stand-in's `_forwarded`
    /// onto `op_rc`, then `synth.set_forwarded(op_rc)` when the two are
    /// distinct objects (`replace_op_with`). A carried `Forwarded::Op` that
    /// already names `op_rc` is left in place so the copy cannot close a
    /// one-node cycle.
    fn adopt_live_synthetic(&mut self, op_rc: &majit_ir::OpRc) {
        let op_pos = op_rc.pos().get();
        // A live SameAs/extra stand-in, else the recorder box in
        // `resop_refs`. `optimizer.py replace_op_with` forwards the old
        // box (`op.set_forwarded(newop)`) when emit clones a new
        // ResOperation (`copy_and_change`). `ptr_eq` with the recorder
        // itself is a no-op carry.
        let Some(synth) = self.live_synthetics_swap_remove_pos(op_pos).or_else(|| {
            self.resop_refs
                .get(&op_pos)
                .cloned()
                .filter(|old| !OpRc::ptr_eq(old, op_rc))
        }) else {
            return;
        };
        let carried = synth.forwarded().borrow().clone();
        // An Op redirect is not info to carry. Copying it onto the producer
        // closes a cycle when the redirect already names that producer, or
        // names a box that forwards back to it.
        let carried_is_op = matches!(carried, majit_ir::forwarding::Forwarded::Op(_));
        if !carried_is_op {
            *op_rc.forwarded().borrow_mut() = carried;
        }
        if !OpRc::ptr_eq(&synth, op_rc) {
            use majit_ir::forwarding::ForwardingHost;
            synth.set_forwarded_op(op_rc);
        }
    }

    pub(crate) fn stamp_emitted_op(src: &Op, dst: &Op) {
        for i in 0..src.num_args() {
            dst.setarg(i, src.arg(i));
        }
        match src.getdescr() {
            Some(d) => dst.setdescr(d),
            None => dst.cleardescr(),
        }
        if src.opcode.is_guard() {
            dst.copy_failargs_shared(src);
            match src.get_fail_arg_types() {
                Some(ts) => dst.set_fail_arg_types(ts),
                None => dst.clear_fail_arg_types(),
            }
            dst.set_rd_resume_position(src.rd_resume_position());
        }
        dst.pos().set(src.pos().get());
    }

    /// Same as [`emit`]: recorder ops reuse the registered `OpRc`.
    #[allow(dead_code)] // resoperation.py AbstractResOp emit reuse
    pub(crate) fn emit_reusing(&mut self, op: Op) -> OpRef {
        self.emit_impl(op)
    }

    fn emit_impl(&mut self, mut op: Op) -> OpRef {
        if op.pos().get().is_none() || op.pos().get().is_constant() {
            // Tag the freshly allocated position with the producer op's
            // result type so the variant-tag readers
            // (`opref_type`/`OptBoxEnv::get_type`) resolve at priority 0
            // (resoperation.py `opclasses[opnum].type` parity).
            op.pos().set(self.reserve_pos_typed(op.result_type()));
        } else {
            // Position invariants for an op that already carries a position:
            //
            // (a) body optimization runs through a fresh TraceIterator whose
            //     `_index` starts at `next_global_opref`, so Phase 2 op
            //     results live in a disjoint `[next_global_opref..)`
            //     range that no prior `emit` has touched.
            //
            // (b) preamble / standalone runs start `next_pos` at
            //     `max(num_inputs, max_raw_pos + 1)`, and `reserve_pos`
            //     is monotonic, so fresh positions are always above any
            //     raw trace op.pos the trace carries.
            //
            // (c) `import_state` only creates op/inputarg forwarding chains
            //     on inputarg slots (in `[inputarg_base..inputarg_base +
            //     num_inputs)`) — never on op-result positions that a
            //     later `emit` would try to use.
            //
            // Together these guarantee that:
            //   - `new_operations` never contains two ops at the same pos
            //   - an op being emitted whose pos is a non-void result does
            //     not already have an op-forwarding redirect set
            //
            // Earlier majit revisions compensated for violations of this invariant
            // with two reactive branches in `emit()` (a collision reassign
            // that called `reserve_pos` again, and a forwarding-redirect
            // that called `reserve_pos` + `make_equal_to(old, new)` to route
            // downstream readers to the fresh position). Both branches
            // are not part of this model. Hard-assert the invariants here so
            // any regression is caught at the emit site rather than at a
            // downstream symptom.
            debug_assert!(
                !self
                    .new_operations
                    .iter()
                    .any(|e| e.pos().get() == op.pos().get()),
                "emit: OpRef collision at {:?} — new_operations already contains this position. \
                 Body optimization should use a fresh TraceIterator and reserve_pos() should be \
                 monotonic above all raw trace positions.",
                op.pos().get(),
            );
            debug_assert!(
                !(self
                    .get_box_replacement_operand_opt(op.pos().get())
                    .is_some_and(|b| self.has_op_forwarding(&b))
                    && op.result_type() != majit_ir::Type::Void),
                "emit: op-forwarding redirect set on non-void result position {:?} — \
                 import_state should only forward inputarg slots in \
                 [inputarg_base..inputarg_base + num_inputs), and body op results \
                 live in a disjoint range above the parent trace high-water mark.",
                op.pos().get(),
            );
            self.next_pos = self.next_pos.max(op.pos().get().raw().saturating_add(1));
        }
        let pos_ref = op.pos().get();
        // RPython parity: emit() does NOT clear forwarding.
        // In RPython, Box._forwarded is never cleared by emit — each Box
        // has unique identity. The forwarding set by import_box must
        // survive body op emission for consumer switchover to work.

        // RPython optimizer.py emit_guard_operation — guard resume
        // data sharing via _copy_resume_data_from / ResumeGuardCopiedDescr.
        //
        // RPython has exactly one emit path (`Optimizer._emit_operation`,
        // optimizer.py).  Pyre's `Optimizer::emit_operation`
        // (`optimizer.rs`) handles guard dispatch (force_box on args,
        // emit_guard_operation, force_box on fail_args, _maybe_replace_guard_value)
        // and then calls `ctx.emit(op.clone())` for the
        // `_newoperations.append(op)` step (optimizer.py).  The
        // OptContext-side emit_guard_operation is the standalone path
        // used by unit tests that drive `OptContext::emit` directly
        // without going through `Optimizer::emit_operation`.
        //
        // Skip the OptContext guard handling when the Optimizer is
        // mid-flight (`in_final_emission == true`) so production runs
        // through one emit_guard_operation (RPython parity).  The
        // OptContext path remains the sole guard handler for the
        // standalone test entry point.
        //
        // optimizer.py: side-effectful non-guard ops clear the
        // sharing chain.  Only relevant for the OptContext-managed
        // `last_guard_idx`; in_final_emission runs use
        // `Optimizer::last_guard_op_idx` instead.
        if op.opcode.is_guard() {
            if !self.in_final_emission
                && let Some(newop) = self.emit_guard_operation(&op)
            {
                op = newop;
            }
        } else {
            // optimizer.py `_emit_operation` — the test runs for every emitted non-guard
            // op, `in_final_emission` included: RPython reaches this code
            // through `_emit_operation` no matter which layer emitted the op.
            // is_call_pure_pure_canraise: a CallPure that
            // can_raise(ignore_memoryerror=True) counts as side-effectful even
            // though has_no_side_effect is true for call_pure opcodes.
            let dominated_by_side_effect = !((op.opcode.has_no_side_effect()
                || op.opcode.is_ovf()
                || op.opcode.is_jit_debug())
                && !Self::is_call_pure_pure_canraise(&op));
            if dominated_by_side_effect {
                self.last_guard_idx = None;
                self.guard_chain_broken = true;
            }
        }

        // The op is about to be pushed into `new_operations`,
        // after which `op_at` resolves its intrinsic `op.type_`
        // (resoperation.py:1693 parity) and `opref_type` returns it via
        // the primary fast path. The Box.type invariant is checked by
        // `debug_assert_box_type_invariant` below.
        Self::debug_assert_box_type_invariant(&op);
        let op_pos = op.pos().get();
        // optimizer.py Optimizer._emit_operation appends the same `op`
        // object (`TraceIterator.next` `_cache`). Reuse the registered
        // `OpRc` when opcode and arity still match — the recorder box, an
        // extra producer, or a SameAs synthetic being emitted as itself.
        // Guard emit mutates failargs/descr on the owned `op` first
        // (`emit_guard_operation`); stamp that state onto the registered
        // box. Opcode/arity mismatch is `copy_and_change`: clone, and
        // `adopt_live_synthetic` / `link_replaced_producer` forward the
        // old box.
        if let Some(reused) = self.resop_refs.get(&op_pos).cloned()
            && reused.opcode == op.opcode
            && reused.num_args() == op.num_args()
        {
            Self::stamp_emitted_op(&op, &reused);
            self.adopt_live_synthetic(&reused);
            self.emitted_operations
                .insert(majit_ir::operand::Operand::from_bound_op(&reused));
            self.push_new_operation(reused);
            return op_pos;
        }
        let op_rc = OpRc::new(op);
        // Catch up any operand placeholder that `materialize_operand_at` created for
        // `op_pos` ahead of this emit (forward-reference path).
        // `resoperation.py AbstractResOpOrInputArg _forwarded` lives on the operation
        // object; late binding establishes that connection so
        // subsequent `box.set_forwarded` reaches `op.forwarded`.
        //
        // The synthetic stand-in registered for `op_pos` by `materialize_operand_at`
        // is the `live_synthetics` entry at this position.
        // `adopt_live_synthetic` migrates its `_forwarded` onto the real producer
        // and drops it from `live_synthetics` so the superseded stand-in is not
        // drained into `phase1_emit_ops`. Each `op_pos` has at most one live
        // stand-in, so the position match is unambiguous. This is the sole
        // carry-over path: the synthetic is the `_forwarded` host every
        // `find_producer_op` reaches before this emit supersedes it.
        // A consumer dispatched BEFORE this producer emitted (forward reference)
        // bound its operand to `synth`; without this link its box-native walk
        // freezes on the orphaned stand-in while the OpRef path resolves `op_rc`.
        self.adopt_live_synthetic(&op_rc);
        // optimizer.py `self._emittedoperations[op] = None`.
        self.emitted_operations
            .insert(majit_ir::operand::Operand::from_bound_op(&op_rc));
        self.push_new_operation(op_rc);
        pos_ref
    }

    /// Register an op dispatched through the passes (`emit_extra` /
    /// `send_extra_operation`) as the producer for its result position,
    /// mirroring `bind_input_resops`. `find_producer_op(pos)` then resolves
    /// box lookups for this position — `materialize_operand_at(pos)` and operand
    /// resolution alike — to this `OpRc`'s `_forwarded` host
    /// (resoperation.py AbstractResOpOrInputArg) instead of a freshly-minted stand-in, keeping a
    /// single box identity per position. Without this, a pass that folds the
    /// dispatched op via `make_equal_to(from_bound_op(op_rc), ..)` writes the
    /// forwarding onto a private `Rc<Op>` that `find_producer_op` cannot
    /// reach, so the replacement is silently dropped (RPython needs no analog:
    /// the op IS its box, resoperation.py AbstractResOpOrInputArg).
    ///
    /// `emit`'s `live_synthetics` catch-up upgrades the binding to the real
    /// producer once the op is emitted; a folded op stays as the chain host.
    /// When `materialize_operand_at` already minted a stand-in for this position,
    /// `op_rc` supersedes it (carrying its `_forwarded`) rather than skipping
    /// registration — otherwise queued-pass reads resolve to the stale stand-in
    /// while writes land on the unregistered `op_rc`, and emit's blind catch-up
    /// then clobbers those writes with the stand-in's slot. (InputArg / const /
    /// sentinel positions are excluded.)
    pub(crate) fn register_extra_producer(&mut self, op_rc: &majit_ir::OpRc) {
        self.install_canonical_producer(op_rc);
    }

    /// Make `o` reachable as the canonical host for its own position.
    ///
    /// `OptUnroll.import_state` forwards a Phase-2 source to the literal carried Box,
    /// and RPython needs no analog because the op IS its box
    /// (`resoperation.py AbstractResOpOrInputArg`). pyre resolves an `OpRef` through a producer
    /// registry instead, so a Box that reaches a rebuilt context only as a
    /// private `Rc` is invisible to `find_producer_op`; a forwarding chain
    /// ending on it leaves guard resume numbering with no producer to bind.
    ///
    /// For Op hosts, only a genuinely unbound position is filled. An
    /// already-registered Op is the one every other chain resolves
    /// through, and overwriting it with a foreign Phase-1 `Rc` would
    /// split one position across two boxes. The InputArg branch below
    /// is the exception: it overwrites `inputarg_refs` so the carried
    /// box replaces the stand-in.
    pub(crate) fn register_carried_host(&mut self, o: &Operand) {
        let pos = o.to_opref();
        if pos.is_none() || pos.is_constant() {
            return;
        }
        // A carried InputArg is the first object for that index
        // (`inputarg_from_tp` / `history.inputargs`). Replace the
        // `ensure_inputarg_bindings` stand-in so `_forwarded` and
        // `resolve_to_operand` name the same Rc.
        if let Some(ia) = o.bound_inputarg() {
            self.inputarg_refs.insert(ia.index, ia);
            return;
        }
        if self.resolve_to_operand(pos).is_some() {
            return;
        }
        if let Some(op) = o.bound_op() {
            self.install_canonical_producer(&op);
        }
    }

    /// RPython emit_extra(op, emit=False) parity: queue an operation to
    /// be processed through passes AFTER the calling pass. Skips earlier
    /// passes (including the caller) to avoid re-absorption loops.
    /// `after_pass_idx`: index of the calling pass (op starts from idx+1).
    pub fn emit_extra(&mut self, after_pass_idx: usize, op: Op) -> OpRef {
        // emit_extra enters the chain at the pass AFTER the caller; the first
        // re-processing pass is `after_pass_idx + 1`.
        self.emit_extra_at(after_pass_idx + 1, OpRc::new(op))
    }

    /// `emit_extra` when the caller already holds the ResOperation object
    /// (`heap.py _lazy_set = op` then `emit_extra(op)`). Queue that `OpRc`
    /// instead of a second `cls()`.
    pub fn emit_extra_rc(&mut self, after_pass_idx: usize, op: majit_ir::OpRc) -> OpRef {
        self.emit_extra_at(after_pass_idx + 1, op)
    }

    /// Queue `op` to be (re)processed by the pass chain starting at
    /// `start_pass`. `emit_extra` enters downstream of the caller; head
    /// re-dispatch (`send_extra_operation`, optimizer.py with
    /// `opt=first_optimization`) enters at pass 0.
    /// Assign a result position to an extra op and check the box-type
    /// invariant. Shared by the queued `emit_extra` path and
    /// `Optimizer::emit_extra_immediate`.
    pub(crate) fn assign_extra_pos(&mut self, op: &majit_ir::OpRc) {
        if op.pos().get().is_none() {
            // Typed allocation, same rationale as `emit`.
            op.pos().set(self.reserve_pos_typed(op.result_type()));
        } else {
            self.next_pos = self.next_pos.max(op.pos().get().raw().saturating_add(1));
        }
        // Queued ops carry their intrinsic `op.type_`
        // (resoperation.py:1693 parity). Once the queued op flushes
        // through `propagate_one` into `new_operations`, `op_at` resolves
        // its type without the side-table detour.
        Self::debug_assert_box_type_invariant(op);
    }

    fn emit_extra_at(&mut self, start_pass: usize, op: majit_ir::OpRc) -> OpRef {
        self.assign_extra_pos(&op);
        let pos_ref = op.pos().get();
        // Register the queued op as the producer for its position so a fold
        // through `make_equal_to(from_bound_op(op_rc), ..)` reaches a host
        // `find_producer_op` resolves to.
        self.register_extra_producer(&op);
        self.extra_operations_after.push_back((start_pass, op));
        pos_ref
    }

    /// optimizer.py send_extra_operation(op) with `opt=None`:
    /// re-dispatch an operation from `first_optimization` (the head of the
    /// pass chain, pass 0), so every pass — including those BEFORE the caller,
    /// e.g. OptIntBounds — reprocesses it. `emit_for_force` only routes through
    /// downstream passes, so a bound computed by an earlier pass is skipped.
    pub fn send_extra_operation(&mut self, op: Op) -> OpRef {
        if self.in_final_emission {
            self.emit(op)
        } else {
            self.emit_extra_at(0, OpRc::new(op))
        }
    }

    /// `send_extra_operation` when the caller already holds the ResOperation
    /// object (`rewrite.py` sending each operation `intdiv` built). Queue
    /// that `OpRc` so a box built on an earlier one stays bound to it.
    pub fn send_extra_operation_rc(&mut self, op: majit_ir::OpRc) -> OpRef {
        if self.in_final_emission {
            self.emit_rc(op)
        } else {
            self.emit_extra_at(0, op)
        }
    }

    pub fn initialize_imported_short_preamble_builder(
        &mut self,
        label_args: &[OpRef],
        short_inputargs: &[OpRef],
        exported_short_boxes: &[crate::optimizeopt::shortpreamble::PreambleOp],
    ) {
        let produced: Vec<(
            majit_ir::operand::Operand,
            crate::optimizeopt::shortpreamble::ProducedShortOp,
        )> = exported_short_boxes
            .iter()
            .map(|entry| {
                // `materialize_operand_at`, not `from_bound_op`: a
                // const-folded entry carries an inline-Const pos,
                // which resolves to its Const operand. The map keys by this
                // res operand; the single-op re-export lookup
                // (pure.rs) reproduces it via `materialize_operand_at(source)`.
                // The operand's identity is the canonical `_forwarded` host
                // Rc (a synthetic producer registered on first
                // materialization), stable across calls, so no warm-up is
                // needed for the lookup keys to be ptr_eq.
                let pos = entry.op.pos().get();
                let res = self.materialize_operand_at(pos);
                (
                    res.clone(),
                    crate::optimizeopt::shortpreamble::ProducedShortOp {
                        kind: entry.kind.clone(),
                        res,
                        preamble_op: OpRc::new((*entry.op).clone()),
                        source_op: entry.source_op.clone().unwrap_or_else(|| entry.op.clone()),
                        invented_name: entry.invented_name,
                        same_as_source: entry.same_as_source.clone(),
                        label_arg_idx: entry.label_arg_idx,
                    },
                )
            })
            .collect();
        let builder = crate::optimizeopt::shortpreamble::ShortPreambleBuilder::new(
            label_args,
            &produced,
            short_inputargs,
        );
        self.imported_short_preamble_builder = Some(builder);
    }

    /// shortpreamble.py `ShortPreambleBuilder.__init__` parity.
    ///
    /// Reads `exported_state.short_boxes` (`ExportedState.short_boxes`)
    /// directly and rebuilds the `ShortPreambleBuilder` keyed by Phase 2
    /// OpRefs. `short_args` is the builder's `label_args`.
    ///
    /// The Phase-2 result OpRef for each entry is read from `result_map`,
    /// computed before this constructor just as the target `Box` identities
    /// exist before `ProducedShortOp.produce_op` runs. Replay args stay the
    /// exported `preamble_op` args (`rebind_arg` rewrites a dependency to the
    /// replay op minted for it). A loop-invariant func that is not a constant
    /// int declines the short preamble (`LoopInvariantOp.produce_op` reads
    /// `self.res.getarg(0).getint()`).
    pub fn initialize_imported_short_preamble_builder_from_short_boxes(
        &mut self,
        short_args: &[OpRef],
        short_inputargs: &[OpRef],
        short_boxes: &[(OpRef, crate::optimizeopt::shortpreamble::ProducedShortOp)],
        result_map: &indexmap::IndexMap<OpRef, OpRef, rustc_hash::FxBuildHasher>,
        exported_infos: &indexmap::IndexMap<
            majit_ir::operand::Operand,
            crate::optimizeopt::info::OpInfo,
        >,
    ) -> bool {
        use crate::optimizeopt::shortpreamble::{
            PreambleOpKind, ProducedShortOp, ShortPreambleBuilder,
        };

        // shortpreamble.py `preamble_op.set_forwarded(info)` writes the
        // ResOperation object `add_op_to_short` created. Phase 2's
        // OptContext does not inherit Phase 1 `resop_refs`, so bind those
        // boxes here (`bind_input_resops` / TraceIterator `_cache`) before
        // `set_preamble_forwarded_info` resolves them by position.
        let preamble_ops: Vec<majit_ir::OpRc> = short_boxes
            .iter()
            .map(|(_, produced_op)| produced_op.preamble_op.clone())
            .collect();
        self.bind_input_resops(&preamble_ops);

        // shortpreamble.py ShortPreambleBuilder.__init__ parity:
        //
        //   for produced_op in short_boxes:
        //       op = produced_op.short_op.res
        //       preamble_op = produced_op.preamble_op
        //       if isinstance(op, Const):
        //           info = optimizer.getinfo(op)
        //       else:
        //           info = exported_infos.get(op, None)
        //           if info is None:
        //               info = empty_info
        //       preamble_op.set_forwarded(info)
        //
        // RPython sets `_forwarded = info` on each `preamble_op` so a
        // later `use_box` (≡ majit `force_op_from_preamble_op`) reads
        // the info via `get_forwarded()` and routes it through
        // `setinfo_from_preamble(box, info, None)` + `info.make_guards`.
        //
        // majit's `force_op_from_preamble_op` reads the equivalent via
        // `self.get_ptr_info(preamble_source)`; without this pre-seed,
        // info from `exported_infos` for sources outside
        // `next_iteration_args` (Pure / Heap / LoopInvariant short-op
        // sources) never reaches the use_box path.
        //
        // Guard against clobbering existing forwarding (set by an
        // earlier `make_equal_to` from `import_state` for sources that
        // happen to coincide with `next_iteration_args`): only seed
        // when no forwarding is recorded yet.
        //
        // Every ProducedShortOp carries two Boxes: `short_op.res` (body)
        // and `preamble_op` (replay, minted by `add_op_to_short`). Guards
        // and short-op args name `preamble_op`, not `res`.
        for (source, produced_op) in short_boxes {
            // shortpreamble.py:417-421: op = produced_op.short_op.res;
            //     if isinstance(op, Const): info = optimizer.getinfo(op)
            //     else: info = exported_infos.get(op, None)
            // Look up by the result BOX (`produced_op.res`) — the exporter keyed
            // `_expand_info(produced_op.res)` by the same Rc (cloned from the
            // shared `exported_short_boxes` entry), so the box-identity lookup
            // hits. unroll.py gates only `_expand_info`; const heap short
            // boxes take the optimizer.getinfo(op) arm at
            // shortpreamble.py:419-420. They travel on the separate const
            // channel, so a Const arriving through this non-const list is
            // still declined here.
            if produced_op.res.to_opref().is_constant() {
                continue;
            }
            if let Some(info) = exported_infos.get(&produced_op.res) {
                // shortpreamble.py `preamble_op.set_forwarded(info)` — the
                // info lives on the replay op, so `make_guards(preamble_op)`
                // names that result rather than the export-time box.
                self.set_preamble_forwarded_info(&produced_op.preamble_op, info);
            }
        }

        // `produced` pairs each exported entry's position with the replay op
        // minted for it, so `rebind_arg` below can rebind a dependency arg
        // (which carries the exported entry's position) to that replay op.
        // `builder_entries` is the
        // builder map: ONE entry per short box keyed by the Phase-1
        // carried res box (`produced_op.res`, the same Rc the produce loop
        // reads as `self.res`). The carried box is invariant to the
        // invented-name replay-position aliasing the dual key compensates for,
        // so the builder map collapses to a single box-identity key.
        let mut produced: Vec<(OpRef, ProducedShortOp)> = Vec::with_capacity(short_boxes.len());
        let mut builder_entries: Vec<(majit_ir::operand::Operand, ProducedShortOp)> =
            Vec::with_capacity(short_boxes.len());
        // shortpreamble.py:PreambleOp.add_op_to_short — Pure ops whose
        // opcode is a Call get rewritten to the CallPure* equivalent so
        // the short preamble can replay the cached call without
        // re-executing arbitrary side effects.
        let pure_call_opcode = |op: OpCode| -> OpCode {
            match op {
                OpCode::CallI => OpCode::CallPureI,
                OpCode::CallR => OpCode::CallPureR,
                OpCode::CallF => OpCode::CallPureF,
                OpCode::CallN => OpCode::CallPureN,
                other => other,
            }
        };
        // shortpreamble.py:PreambleOp.add_op_to_short — LoopInvariant ops
        // become CallLoopinvariant* so the short preamble re-executes the
        // call exactly once per loop iteration.
        let loop_invariant_opcode = |result_type: majit_ir::Type| -> OpCode {
            match result_type {
                majit_ir::Type::Int => OpCode::CallLoopinvariantI,
                majit_ir::Type::Ref => OpCode::CallLoopinvariantR,
                majit_ir::Type::Float => OpCode::CallLoopinvariantF,
                majit_ir::Type::Void => OpCode::CallLoopinvariantN,
            }
        };
        // shortpreamble.py `produce_arg`: an exported replay-op arg is a
        // Const, a renamed short inputarg, or a dependency's replay op. The
        // first two are carried unchanged; a dependency is rebound to the
        // replay op this builder minted for it, so `use_box` appends that op
        // rather than the exporting phase's producer.
        let rebind_arg = |produced: &[(OpRef, ProducedShortOp)], arg: Operand| -> Operand {
            let r = arg.to_opref();
            if r.is_constant() {
                return arg;
            }
            produced
                .iter()
                .rev()
                .find(|(k, _)| *k == r)
                .map(|(_, dep)| Operand::from_bound_op(&dep.preamble_op))
                .unwrap_or(arg)
        };

        for (source, produced_op) in short_boxes {
            // Some ProducedShortOps (PreambleOpKind::Heap with non-getfield /
            // non-getarrayitem opcodes, or other non-emitting entries) have
            // no Phase-2 result. They are no-ops for the imported builder.
            if result_map.get(source).is_none() {
                continue;
            }
            match produced_op.kind {
                PreambleOpKind::Pure => {
                    let resolved_arg_boxes: Vec<Operand> = produced_op
                        .preamble_op
                        .args_slice()
                        .iter()
                        .map(|a| rebind_arg(&produced, a.clone()))
                        .collect();
                    let mut op = Op::new(
                        pure_call_opcode(produced_op.preamble_op.opcode),
                        &resolved_arg_boxes,
                    );
                    op.pos().set(produced_op.preamble_op.pos().get());
                    if let Some(d) = produced_op.preamble_op.getdescr() {
                        op.setdescr(d);
                    }
                    let res = self.materialize_operand_at(op.pos().get());
                    let new_pop = ProducedShortOp {
                        kind: PreambleOpKind::Pure,
                        res,
                        preamble_op: OpRc::new(op),
                        source_op: produced_op.source_op.clone(),
                        invented_name: produced_op.invented_name,
                        same_as_source: produced_op.same_as_source.clone(),
                        label_arg_idx: produced_op.label_arg_idx,
                    };
                    builder_entries.push((produced_op.res.clone(), new_pop.clone()));
                    // Exported args name either the entry or its replay op.
                    let replay = produced_op.preamble_op.pos().get();
                    produced.push((*source, new_pop.clone()));
                    if *source != replay {
                        produced.push((replay, new_pop));
                    }
                }
                PreambleOpKind::Heap => {
                    let result_type = produced_op.preamble_op.result_type();
                    let descr = match produced_op.preamble_op.getdescr() {
                        Some(d) => d,
                        None => continue,
                    };
                    let new_pop = match produced_op.preamble_op.opcode {
                        OpCode::GetfieldGcI | OpCode::GetfieldGcR | OpCode::GetfieldGcF => {
                            let opcode = match result_type {
                                majit_ir::Type::Int => OpCode::GetfieldGcI,
                                majit_ir::Type::Ref => OpCode::GetfieldGcR,
                                majit_ir::Type::Float => OpCode::GetfieldGcF,
                                majit_ir::Type::Void => return false,
                            };
                            // shortpreamble.py:416-426 keeps the exported
                            // `preamble_op` object on the builder. Its receiver
                            // is the renamed short-inputarg (or replay result),
                            // while `source_op` separately carries the original
                            // receiver used by HeapOp.produce_op for PtrInfo.
                            // Resolving this arg through `short_args` collapses
                            // those identities and emits guards on an exporting-
                            // phase box that a retrace cannot bind.
                            let obj_b = rebind_arg(&produced, produced_op.preamble_op.arg(0));
                            let mut op = Op::new(opcode, &[obj_b]);
                            op.pos().set(produced_op.preamble_op.pos().get());
                            op.setdescr(descr);
                            let res = self.materialize_operand_at(op.pos().get());
                            ProducedShortOp {
                                kind: PreambleOpKind::Heap,
                                res,
                                preamble_op: OpRc::new(op),
                                source_op: produced_op.source_op.clone(),
                                invented_name: produced_op.invented_name,
                                same_as_source: produced_op.same_as_source.clone(),
                                label_arg_idx: produced_op.label_arg_idx,
                            }
                        }
                        OpCode::GetarrayitemGcI
                        | OpCode::GetarrayitemGcR
                        | OpCode::GetarrayitemGcF => {
                            let opcode = match result_type {
                                majit_ir::Type::Int => OpCode::GetarrayitemGcI,
                                majit_ir::Type::Ref => OpCode::GetarrayitemGcR,
                                majit_ir::Type::Float => OpCode::GetarrayitemGcF,
                                majit_ir::Type::Void => return false,
                            };
                            // shortpreamble.py `HeapOp.produce_op` reads
                            // `g.getarg(1).getint()` from the original heap op.
                            // The replay op keeps `preamble_op`'s own index arg.
                            let obj_b = rebind_arg(&produced, produced_op.preamble_op.arg(0));
                            let index_b = rebind_arg(&produced, produced_op.preamble_op.arg(1));
                            let mut op = Op::new(opcode, &[obj_b, index_b]);
                            op.pos().set(produced_op.preamble_op.pos().get());
                            op.setdescr(descr);
                            let res = self.materialize_operand_at(op.pos().get());
                            ProducedShortOp {
                                kind: PreambleOpKind::Heap,
                                res,
                                preamble_op: OpRc::new(op),
                                source_op: produced_op.source_op.clone(),
                                invented_name: produced_op.invented_name,
                                same_as_source: produced_op.same_as_source.clone(),
                                label_arg_idx: produced_op.label_arg_idx,
                            }
                        }
                        _ => continue,
                    };
                    builder_entries.push((produced_op.res.clone(), new_pop.clone()));
                    // Exported args name either the entry or its replay op.
                    let replay = produced_op.preamble_op.pos().get();
                    produced.push((*source, new_pop.clone()));
                    if *source != replay {
                        produced.push((replay, new_pop));
                    }
                }
                PreambleOpKind::LoopInvariant => {
                    let result_type = produced_op.preamble_op.result_type();
                    // shortpreamble.py `LoopInvariantOp.produce_op` requires
                    // `getarg(0).getint()`. Read it off the replay op's arg 0.
                    if produced_op.preamble_op.arg(0).const_int().is_none() {
                        return false;
                    }
                    let func_b = rebind_arg(&produced, produced_op.preamble_op.arg(0));
                    let mut op = Op::new(loop_invariant_opcode(result_type), &[func_b]);
                    op.pos().set(produced_op.preamble_op.pos().get());
                    let res = self.materialize_operand_at(op.pos().get());
                    let new_pop = ProducedShortOp {
                        kind: PreambleOpKind::LoopInvariant,
                        res,
                        preamble_op: OpRc::new(op),
                        source_op: produced_op.source_op.clone(),
                        invented_name: produced_op.invented_name,
                        same_as_source: produced_op.same_as_source.clone(),
                        label_arg_idx: produced_op.label_arg_idx,
                    };
                    builder_entries.push((produced_op.res.clone(), new_pop.clone()));
                    // Exported args name either the entry or its replay op.
                    let replay = produced_op.preamble_op.pos().get();
                    produced.push((*source, new_pop.clone()));
                    if *source != replay {
                        produced.push((replay, new_pop));
                    }
                }
                PreambleOpKind::InputArg | PreambleOpKind::Guard => {}
            }
        }

        let builder = ShortPreambleBuilder::new(short_args, &builder_entries, short_inputargs);
        self.imported_short_preamble_builder = Some(builder);
        true
    }

    /// unroll.py: force_op_from_preamble(preamble_op)
    ///
    /// RPython receives a PreambleOp with invented_name already set.
    /// Calls use_box then registers in potential_extra_ops.
    pub fn force_op_from_preamble_op(
        &mut self,
        preamble_op: &crate::optimizeopt::info::PreambleOp,
    ) -> Operand {
        let preamble_source = preamble_op.op.to_opref();
        // RPython `return preamble_op.op` returns the carried Box. In majit,
        // shortpreamble.py:434 `op = preamble_op.op.get_box_replacement()` —
        // walk the box's own `_forwarded` chain (total; identity on a miss),
        // object-native rather than positional.
        let resolved = preamble_op.op.get_box_replacement(false);
        let result = resolved.to_opref();
        let result_type = preamble_op.preamble_op.result_type();
        let is_constant = resolved.const_value().is_some();
        // unroll.py:32: use_box(op, preamble_op.preamble_op, self).
        // RPython passes the preamble_op directly — no lookup miss possible.
        // majit prefers the produced_short_boxes lookup (Phase-2 remapped pos)
        // with fallback to info::PreambleOp.preamble_op.
        let Some((arg_guards, result_guards)) =
            self.collect_use_box_guards(&preamble_op.preamble_op)
        else {
            self.signal_invalid_loop("short preamble GC layout tid is unresolved");
            return preamble_op.op.clone();
        };
        // unroll.py:28: assert self.short_preamble_producer is not None
        if let Some(mut builder) = self.active_short_preamble_producer.take() {
            builder.use_box(
                preamble_source,
                &preamble_op.preamble_op,
                &arg_guards,
                &result_guards,
            );
            self.active_short_preamble_producer = Some(builder);
        } else if let Some(mut builder) = self.imported_short_preamble_builder.take() {
            builder.use_box(
                preamble_source,
                &preamble_op.preamble_op,
                &arg_guards,
                &result_guards,
            );
            self.imported_short_preamble_builder = Some(builder);
        } else {
            unreachable!("force_op_from_preamble_op: no short_preamble_producer");
        }
        // shortpreamble.py:401-405: info = preamble_op.get_forwarded();
        // preamble_op.set_forwarded(None);
        // optimizer.setinfo_from_preamble(box, info, None)
        //
        // RPython reads `_forwarded` from the replay Op object
        // (`preamble_op`), NOT from `preamble_op.op` (= self.res or
        // the alt for invented). pyre's flat-OpRef equivalent is
        // `pop.preamble_op.pos` — the OpRef the replay Op was
        // constructed at by `ImportedShortPureOp::new`.
        // For invented Pure that OpRef differs from `pop.op` (the
        // alt) so the alt's `make_equal_to(...)` chain at
        // `forwarded[pop.op]` does not collide with the replay's
        // info at `forwarded[pop.preamble_op.pos]`.
        if let Some(info) = self.take_preamble_forwarded_opinfo(preamble_op.preamble_op.pos().get())
        {
            self.setinfo_from_preamble_item_option(&resolved, &info, None);
        }
        // `unroll.py setinfo_from_preamble` rebuilds an imported string as
        // `StrPtrInfo(preamble_info.mode)` with no `lgtop`, so the peeled
        // body re-derives the length from the box it actually holds.
        // Upstream gets that for free: the body's box is a fresh `Box` with
        // no `_forwarded`. pyre shares the Phase-1 res across the peel
        // boundary (`produced_short_op`), so this import lands on a box that
        // already carries the preamble's `StrPtrInfo`, and
        // `setinfo_from_preamble`'s `get_forwarded() is not None` early
        // return then skips the rebuild. Drop the stale length box here:
        // kept, the body's `string_copy_parts` reuses the preamble's STRLEN
        // result — a value no short-preamble replay produces — and
        // `assemble_peeled_trace_with_jump_args` carries it on the loop
        // LABEL as a tail arg that no bridge close can reproduce, which the
        // backend rejects (`x86/regalloc.py consider_jump`
        // `assert len(arglocs) == jump_op.numargs()`). A constant length is
        // phase-independent, so it stays.
        if let Some(resolved_box) = self.get_box_replacement_operand_opt(result)
            && self
                .getptrinfo(&resolved_box)
                .and_then(|info| info.get_cached_lgtop())
                .is_some_and(|lgtop| !lgtop.is_constant())
        {
            self.with_ptr_info_mut(&resolved_box, |info| {
                if let PtrInfo::Str(si) = info {
                    si.lgtop = None;
                }
            });
        }
        // RPython PreambleOp carries Box.type intrinsically.
        // the replay `result` OpRef is typed via the upstream factory
        // (`op_typed`); priority 0 of `opref_type`
        // resolves it from the variant tag without a side-table seed.
        let _ = result_type;
        // unroll.py:34-37: potential_extra_ops[op] = preamble_op
        if !is_constant {
            // unroll.py:29/37 keys by `preamble_op.op`, which is
            // short_op.res: the exact Box returned to and held by the
            // body. The replay result is deliberately a different Box.
            let key = resolved.clone();
            if crate::optimizeopt::majit_log_enabled() {
                eprintln!(
                    "[jit] potential_extra_ops.insert key={key:?} source={preamble_source:?} result={result:?} invented={}",
                    preamble_op.invented_name
                );
            }
            // `unroll.py:37` dict-assign semantics — overwrite if the
            // key already exists, otherwise append.
            self.potential_extra_ops.insert(key, preamble_op.clone());
        }
        // unroll.py `return preamble_op.op`. RPython's `preamble_op.op`
        // equals `self.res` (shortpreamble.py:120 `op = self.res`); pyre's
        // Phase 1 source IS `self.res` for the imported short box. Return
        // it directly so non-invented Pure body references resolve through
        // the Phase 1 OpRef. The use-before-def pass at LABEL emission
        // (unroll.rs `assemble_peeled_trace_with_jump_args`) extends
        // `LABEL.arglist` with that Phase 1 OpRef when the body actually
        // uses it; the orthodox `force_box → potential_extra_ops.pop →
        // add_preamble_op` path (shortpreamble.py) handles
        // `used_boxes` / `short_preamble_jump` / `extra_same_as` for the
        // imported short box.
        let _ = result;
        preamble_op.op.clone()
    }

    /// shortpreamble.py:383-396,401-406: collect guards from the forwarded
    /// info of preamble_op's args and result. RPython's `info = arg.get_forwarded()`
    /// returns whatever is stored — PtrInfo *or* IntBound — and calls
    /// `info.make_guards(...)` uniformly. None declines reconstruction when a
    /// required pointer-layout guard cannot name its runtime tid.
    fn collect_use_box_guards(
        &mut self,
        preamble_op: &Op,
    ) -> Option<(Vec<majit_ir::OpRc>, Vec<majit_ir::OpRc>)> {
        // shortpreamble.py:383-401 line-by-line:
        //
        //   for arg in preamble_op.getarglist():
        //       if isinstance(arg, Const):
        //           continue
        //       if isinstance(arg, AbstractInputArg):
        //           info = arg.get_forwarded()
        //           if info is not None and info is not empty_info:
        //               info.make_guards(arg, self.short, optimizer)
        //       elif arg.get_forwarded() is None:
        //           pass
        //       else:
        //           self.short.append(arg)
        //           info = arg.get_forwarded()
        //           if info is not empty_info:
        //               info.make_guards(arg, self.short, optimizer)
        //           arg.set_forwarded(None)
        //
        // RPython has three branches per arg:
        //   * Const → skip (pyre: `OpRef::is_constant()`).
        //   * AbstractInputArg with forwarded info → emit guards, do NOT
        //     clear (info lives across iterations on input args).
        //   * non-input non-Const with forwarded info → also append the
        //     arg op to `self.short` (handled by the builder's `use_box`
        //     dependency walk in `shortpreamble.rs`), emit
        //     guards, AND clear the replay op's marker to prevent double-emission.
        //
        // RPython's non-input `arg` is the replay ResOperation object, while
        // its live body Box keeps separate optimizer info. In pyre's flat-OpRef
        // model the IntBound/PtrInfo lives in the canonical operand forwarding
        // slot, so clearing that slot here would drop live body facts before
        // loop-close virtualstate.py:491-492 can see them. The builder consumes
        // the replay op's own marker (`Op.forwarded`) in `shortpreamble.rs`'s
        // `use_box`
        // to mirror `arg.set_forwarded(None)` without clearing optimizer info.
        enum ForwardedInfo {
            // shortpreamble.py EmptyInfo / empty_info sentinel.
            // Its presence is meaningful for short-preamble dedup, but it
            // intentionally emits no guards.
            Empty,
            // info.py PtrInfo + ConstPtrInfo (info.py). PtrInfo
            // dispatches further to ConstPtrInfo::make_guards when the
            // PtrInfo is a Constant variant.
            Ptr(PtrInfo),
            // intutils.py IntBound::make_guards. Constant ints come
            // through this arm via IntBound::is_constant().
            Int(crate::optimizeopt::intutils::IntBound),
            // info.py FloatConstInfo carries a single ConstFloat;
            // make_guards (info.py) emits a GUARD_VALUE pinning `op`
            // to the ConstFloat. `set_preamble_forwarded_info` plants this
            // shape per shortpreamble.py:416
            // `preamble_op.set_forwarded(info)`.
            FloatConst(f64),
        }
        let snapshot_forwarded = |ctx: &Self, arg: OpRef| -> Option<ForwardedInfo> {
            // shortpreamble.py:387 `info = arg.get_forwarded()` — PyPy
            // returns the AbstractInfo subtype stored in `_forwarded`.
            // Pyre's canonical `_forwarded` host carries:
            //   `Forwarded::Info(OpInfo::Ptr(_))` — info.py PtrInfo
            //   `Forwarded::Info(OpInfo::IntBound(_))` — intutils.py
            //   `Forwarded::Info(OpInfo::FloatConstInfo(_))` — info.py
            //       FloatConstInfo planted via set_preamble_forwarded_info.
            //   `Forwarded::Info(OpInfo::EmptyInfo(_))` —
            //       shortpreamble.py empty_info.
            let forwarded = ctx.read_forwarded(arg)?;
            use crate::optimizeopt::info::OpInfo;
            match &forwarded {
                majit_ir::forwarding::Forwarded::Info(OpInfo::EmptyInfo(_)) => {
                    Some(ForwardedInfo::Empty)
                }
                majit_ir::forwarding::Forwarded::Info(OpInfo::Ptr(info)) => {
                    Some(ForwardedInfo::Ptr(info.borrow().clone()))
                }
                majit_ir::forwarding::Forwarded::Info(OpInfo::IntBound(b)) => {
                    Some(ForwardedInfo::Int(b.borrow().clone()))
                }
                majit_ir::forwarding::Forwarded::Info(OpInfo::FloatConstInfo(f)) => {
                    Some(ForwardedInfo::FloatConst(f.getconst()))
                }
                _ => None,
            }
        };
        // Phase 1 (read-only): classify each arg per the PyPy three-branch
        // shape and snapshot the info-bearing slot.
        struct ArgEntry {
            arg: OpRef,
            info: ForwardedInfo,
            #[allow(dead_code)]
            is_input: bool,
        }
        let mut arg_entries: Vec<ArgEntry> = Vec::new();
        for arg_operand in preamble_op.args_slice().iter() {
            let arg = arg_operand.to_opref();
            // Branch 1: shortpreamble.py:384 `isinstance(arg, Const): continue`.
            if arg.is_constant() || arg.is_none() {
                continue;
            }
            // shortpreamble.py `isinstance(arg, AbstractInputArg)` — the
            // classification is the arg's TYPE, not membership in
            // `short_inputargs`. A renamed short inputarg and an original
            // label arg are both AbstractInputArg; either way an input arg's
            // forwarded info must NOT be cleared (it lives across iterations).
            let is_input = matches!(
                arg,
                OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_)
            );
            let has_replay_marker = arg_operand.bound_op().is_some_and(|op| {
                !matches!(
                    &op.forwarded().borrow(),
                    majit_ir::forwarding::Forwarded::None
                )
            });
            if (is_input || has_replay_marker)
                && let Some(info) = snapshot_forwarded(self, arg)
            {
                arg_entries.push(ArgEntry {
                    arg,
                    info,
                    is_input,
                });
            }
            // shortpreamble.py:391 `elif arg.get_forwarded() is None: pass`
            // is the no-info branch; falling out of `snapshot_forwarded`
            // returning None is the equivalent.
        }
        let result_info: Option<(OpRef, ForwardedInfo)> =
            snapshot_forwarded(self, preamble_op.pos().get())
                .map(|info| (preamble_op.pos().get(), info));

        // Phase 2: generate guards — `make_guards` takes `&mut self`
        // directly. Constants seed via reserve_const_ref + seed_constant
        // (mirroring `ConstInt` / `ConstPtr` inline construction); producer
        // OpRefs come from `alloc_op_position_typed`.
        let mut arg_guards = Vec::new();
        // info.py FloatConstInfo.make_guards / ConstPtrInfo path —
        // single-value info classes emit a GUARD_VALUE that pins `op` to
        // the recorded constant.
        let emit_const_guard = |arg: &Operand, value: &Value, guards: &mut Vec<majit_ir::OpRc>| {
            // history.py Const{Int,Float,Ptr}.value inline — GUARD_VALUE
            // second operand is the inline-Const directly.
            assert!(
                !matches!(value, Value::Void),
                "emit_const_guard: ConstVoid not allowed"
            );
            let c = Operand::const_from_value(*value);
            guards.push(majit_ir::OpRc::new(Op::new(
                OpCode::GuardValue,
                &[arg.clone(), c],
            )));
        };
        for entry in &arg_entries {
            let arg = self.materialize_operand_at(entry.arg);
            match &entry.info {
                ForwardedInfo::Empty => {}
                ForwardedInfo::Ptr(p) => {
                    if !p.make_guards(&arg, &mut arg_guards, self) {
                        return None;
                    }
                }
                ForwardedInfo::Int(b) => b.make_guards(&arg, &mut arg_guards, self),
                ForwardedInfo::FloatConst(f) => {
                    emit_const_guard(&arg, &Value::Float(*f), &mut arg_guards)
                }
            }
        }
        let mut result_guards = Vec::new();
        if let Some((result_ref, info)) = &result_info {
            let result = self.materialize_operand_at(*result_ref);
            match info {
                ForwardedInfo::Empty => {}
                ForwardedInfo::Ptr(p) => {
                    if !p.make_guards(&result, &mut result_guards, self) {
                        return None;
                    }
                }
                ForwardedInfo::Int(b) => b.make_guards(&result, &mut result_guards, self),
                ForwardedInfo::FloatConst(f) => {
                    emit_const_guard(&result, &Value::Float(*f), &mut result_guards)
                }
            }
        }
        Some((arg_guards, result_guards))
    }

    /// shortpreamble.py `preamble_op.set_forwarded(info)` for imported
    /// short preamble ops. Store the same family of info values that RPython
    /// stores in `_forwarded`, without transforming them through
    /// `setinfo_from_preamble` yet. The `preamble_op` object is the box
    /// (`resoperation.py AbstractResOp`); register it so a later position
    /// reader shares that host.
    pub(crate) fn set_preamble_forwarded_info(
        &mut self,
        preamble_op: &majit_ir::OpRc,
        info: &crate::optimizeopt::info::OpInfo,
    ) {
        use crate::optimizeopt::info::OpInfo;
        let source = preamble_op.pos().get();
        if source.is_constant() || source.is_none() {
            return;
        }
        if !matches!(
            source,
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_)
        ) {
            self.bind_input_resops(std::slice::from_ref(preamble_op));
        }
        let b = Operand::from_bound_op(preamble_op);
        if self.has_forwarding(&b) {
            return;
        }
        match info {
            OpInfo::Unknown => b.clear_forwarded(),
            OpInfo::EmptyInfo(_) => b.set_forwarded_info(info.clone()),
            other => b.set_forwarded_info(other.clone()),
        }
    }

    /// shortpreamble.py:401-405 line-by-line:
    ///
    /// ```python
    /// info = preamble_op.get_forwarded()
    /// preamble_op.set_forwarded(None)
    /// if optimizer is not None:
    ///     optimizer.setinfo_from_preamble(box, info, None)
    /// ```
    ///
    /// RPython reads `_forwarded` from the `preamble_op` Op object directly
    /// — `get_box_replacement` is NOT applied to the slot. Box replacement
    /// only matters for the `box` argument that subsequently receives the
    /// info via `setinfo_from_preamble(box, info, None)`. Walking the
    /// replacement chain on the source side would point at the body-visible
    /// OpRef whose slot is empty (the seed at `forwarded[source]` was
    /// installed by `set_preamble_forwarded_info`).
    ///
    /// `set_forwarded(None)` clears the slot so a second `use_box` for the
    /// same preamble op never re-fires `info.make_guards`. In majit's flat
    /// OpRef model the slot is shared with the box replacement chain
    /// (`Forwarded::Op` / `Forwarded::InputArg`), which other code follows
    /// via `get_box_replacement`; clearing that variant would silently break
    /// downstream replacement, so only the info-bearing variants
    /// (Info / IntBound / Const) take + clear, matching PyPy's clear
    /// semantics on the info-bearing branches.
    fn take_preamble_forwarded_opinfo(
        &mut self,
        source: OpRef,
    ) -> Option<crate::optimizeopt::info::OpInfo> {
        use crate::optimizeopt::info::OpInfo;
        // Forwarding-slot-authoritative read. PyPy stores the replay op's forwarded
        // info directly on `preamble_op._forwarded`; pyre stores the same
        // state in the operand's forwarding slot keyed by `source`. Non-constant
        // `Forwarded::Op`/`InputArg` is a replacement chain and is excluded.
        // Const targets can still appear from legacy bridge/fixture replay
        // paths; normalize them to the OpInfo shape consumed by
        // `setinfo_from_preamble_item_option`.
        let result = {
            let fwd = self.read_forwarded(source)?;
            match &fwd {
                majit_ir::forwarding::Forwarded::Info(OpInfo::EmptyInfo(e)) => {
                    Some(OpInfo::EmptyInfo(*e))
                }
                majit_ir::forwarding::Forwarded::Info(OpInfo::Ptr(p)) => {
                    Some(OpInfo::Ptr(p.clone()))
                }
                majit_ir::forwarding::Forwarded::Info(OpInfo::IntBound(ib)) => {
                    Some(OpInfo::IntBound(ib.clone()))
                }
                // info.py FloatConstInfo planted via
                // `set_preamble_forwarded_info` (shortpreamble.py:416
                // `preamble_op.set_forwarded(info)`).
                majit_ir::forwarding::Forwarded::Info(OpInfo::FloatConstInfo(f)) => {
                    Some(OpInfo::FloatConstInfo(*f))
                }
                majit_ir::forwarding::Forwarded::Const(_)
                | majit_ir::forwarding::Forwarded::SmallConst(_)
                | majit_ir::forwarding::Forwarded::SmallWide(_) => {
                    // optimizer.py `getinfo` parity for the Const
                    // terminal — Refs surface as `ConstPtrInfo`, Floats as
                    // `FloatConstInfo`, Ints as `IntBound::from_constant`.
                    match fwd.const_value() {
                        Some(majit_ir::Value::Ref(gcref)) => Some(OpInfo::ptr(
                            crate::optimizeopt::info::PtrInfo::Constant(gcref),
                        )),
                        Some(majit_ir::Value::Float(f)) => Some(OpInfo::FloatConstInfo(
                            crate::optimizeopt::info::FloatConstInfo::new(f),
                        )),
                        Some(majit_ir::Value::Int(i)) => Some(OpInfo::int_bound(
                            crate::optimizeopt::intutils::IntBound::from_constant(i),
                        )),
                        Some(majit_ir::Value::Void) | None => None,
                    }
                }
                _ => None,
            }
        };
        if result.is_some() {
            // shortpreamble.py:401 preamble_op.set_forwarded(None) —
            // write directly to the canonical host so we don't
            // re-fetch the operand wrapper.
            self.clear_forwarded(source);
        }
        result
    }

    /// unroll.py: setinfo_from_preamble(op, preamble_info, exported_infos)
    /// RPython uses sequential `if` (not elif) so multiple properties accumulate.
    /// `exported_infos`: None from use_box path (shortpreamble.py),
    /// Some from import_state path. When None, virtual branch does NOT recurse.
    ///
    /// `preamble_info_handle` is the live `Rc<RefCell<PtrInfo>>` cell from
    /// the exporter's `_forwarded` slot (or the shortpreamble entry's
    /// `OpInfo::Ptr(rc)`). The virtual branch shares the SAME `Rc`
    /// (`unroll.py` `op.set_forwarded(preamble_info)`) so future
    /// mutations to virtual fields propagate through both export and
    /// import sides — RPython object identity. Non-virtual branches
    /// snapshot the inner `PtrInfo` once because they intentionally
    /// mint fresh info objects per upstream (`unroll.py:71` etc.).
    fn setinfo_from_preamble(
        &mut self,
        op: &majit_ir::operand::Operand,
        preamble_info_handle: &std::rc::Rc<std::cell::RefCell<PtrInfo>>,
        exported_infos: Option<
            &indexmap::IndexMap<majit_ir::operand::Operand, crate::optimizeopt::info::OpInfo>,
        >,
    ) {
        let op_box = op.get_box_replacement(false);
        // unroll.py:55: if op.get_forwarded() is not None: return
        // (covers Op redirect + Info + IntBound + Const states uniformly,
        // matching the sibling setinfo_from_preamble_item pattern below.)
        if self.has_forwarding(&op_box) {
            return;
        }
        // unroll.py:57: if op.is_constant(): return
        if op_box.const_value().is_some() {
            return;
        }
        // operand shim for `set_ptr_info` / `make_nonnull` calls below.
        // RPython `unroll.py:54` `op = get_box_replacement(op)` followed
        // by `op.set_forwarded(...)` writes unconditionally; `op` was
        // chain-resolved and checked non-forwarded / non-constant above, so
        // `materialize_operand_at` returns its canonical `_forwarded` host
        // (minting one only for an unbound preamble/test slot).  The mint is
        // load-bearing: a chain-resolved operand is not necessarily registered
        // in `resop_refs`, and an unregistered position later reaches guard
        // resume numbering, where `get_box_replacement_operand` has no producer
        // to bind.
        let op_box = self.materialize_write_host(op_box);
        // unroll.py:60-64: virtual — set_forwarded + recurse, then return.
        // Identity-preserving install: clone the `Rc` (not the inner
        // `PtrInfo`) so that the exporter, the importer, and the recursive
        // virtual fields all observe the same `RefCell<PtrInfo>` cell,
        // matching PyPy `op.set_forwarded(preamble_info)` object sharing.
        let is_virtual = preamble_info_handle.borrow().is_virtual();
        if is_virtual {
            // unroll.py `setinfo_from_preamble` virtual arm: always
            // `op.set_forwarded(preamble_info)` and recurse. A
            // constant-filled wrapint, range, or range-iterator must
            // stay virtual so Phase 2 still sees the field boxes
            // (`for _ in range(4)` otherwise becomes
            // `'int' object is not an iterator`).
            let resolved = op_box.get_box_replacement(false);
            if !resolved.is_constant() {
                resolved.set_forwarded_info(crate::optimizeopt::info::OpInfo::Ptr(
                    std::rc::Rc::clone(preamble_info_handle),
                ));
            }
            if let Some(infos) = exported_infos {
                // unroll.py: setinfo_from_preamble_list(
                //     preamble_info.all_items(), exported_infos).
                // Read field BOXES via `all_items()` — the SAME accessor the
                // exporter (`_expand_infos_from_virtual`) walks on this shared
                // `PtrInfo` cell, so the box-identity (`Rc::ptr_eq`) lookup hits.
                // Using `all_items()` (not a per-variant arm) keeps export/import
                // symmetric: `VirtualRawBuffer`/`VirtualArrayStruct` return `[]`
                // here exactly as on the export side (RPython
                // `RawBufferPtrInfo.all_items()` is also empty), so the import
                // never iterates raw-buffer slots and never calls
                // `clear_forwarded` on a `from_opref`-minted unbound box.
                let items: Vec<majit_ir::operand::Operand> = preamble_info_handle
                    .borrow()
                    .all_items()
                    .iter()
                    .map(|(_, e)| e.as_seen_operand())
                    .collect();
                self.setinfo_from_preamble_list(&items, infos);
            }
            return;
        }

        // Snapshot the non-virtual PtrInfo once. Non-virtual paths
        // mint fresh info objects on each install (`unroll.py:71` etc.),
        // so identity sharing is not required (and matches PyPy by
        // intentionally not sharing).
        let preamble_info_owned = preamble_info_handle.borrow().clone();
        let preamble_info: &PtrInfo = &preamble_info_owned;

        // unroll.py:65-68: constant — return early
        if let PtrInfo::Constant(gcref) = preamble_info {
            self.make_constant_box(&op_box, Value::Ref(*gcref));
            return;
        }

        // Sequential checks (RPython: NOT elif, all accumulate)

        // unroll.py:69-74: Struct/Instance with descr → set_forwarded
        if preamble_info.get_descr().is_some() {
            if let PtrInfo::Struct(sinfo) = preamble_info {
                self.set_ptr_info(&op_box, PtrInfo::struct_ptr(sinfo.descr.clone()));
            }
            if let PtrInfo::Instance(iinfo) = preamble_info {
                self.set_ptr_info(&op_box, PtrInfo::instance(iinfo.descr.clone(), None));
            }
        }

        // unroll.py:75-77: known_class → make_constant_class(op, class, False)
        if let Some(cls) = preamble_info.get_known_class(self.cpu.as_ref()) {
            crate::optimizeopt::optimizer::Optimizer::make_constant_class(
                self, &op_box, cls, false, // update_last_guard=False (unroll.py:77)
            );
        }

        // unroll.py:79-84: ArrayPtrInfo → set_forwarded(ArrayPtrInfo(descr, lenbound))
        if let PtrInfo::Array(ainfo) = preamble_info {
            self.set_ptr_info(
                &op_box,
                PtrInfo::array(ainfo.descr.clone(), ainfo.lenbound.clone()),
            );
        }

        // unroll.py:85-89: StrPtrInfo — clone lenbound
        if let PtrInfo::Str(sinfo) = preamble_info {
            let mut new_info = crate::optimizeopt::info::StrPtrInfo {
                lenbound: sinfo.lenbound.clone(),
                lgtop: None,
                mode: sinfo.mode,
                length: -1,
                // unroll.py:86: StrPtrInfo(preamble_info.mode) — always
                // rebuild a plain non-virtual StrPtrInfo; never carry
                // the previous iteration's virtual variant across.
                variant: crate::optimizeopt::info::VStringVariant::Ptr,
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            };
            if new_info.lenbound.is_none() {
                new_info.lenbound = Some(crate::optimizeopt::intutils::IntBound::nonnegative());
            }
            self.set_ptr_info(&op_box, PtrInfo::Str(new_info));
        }

        // unroll.py:91-92: is_nonnull → make_nonnull
        if preamble_info.is_nonnull() {
            self.make_nonnull(&op_box);
        }
    }

    /// unroll.py setinfo_from_preamble_list(lst, infos):
    ///
    /// ```python
    /// def setinfo_from_preamble_list(self, lst, infos):
    ///     for item in lst:
    ///         if item is None:
    ///             continue
    ///         i = infos.get(item, None)
    ///         if i is not None:
    ///             self.setinfo_from_preamble(item, i, infos)
    ///         else:
    ///             item.set_forwarded(None)
    ///             # let's not inherit stuff we don't
    ///             # know anything about
    /// ```
    ///
    /// Every `infos.get(item) is not None` branch funnels through
    /// `setinfo_from_preamble`, which starts with the early-return checks
    /// at unroll.py:54-58 (`get_box_replacement` + `get_forwarded` +
    /// `is_constant`). A shortcut that applies IntBound / FloatConst /
    /// Constant without those checks overwrites already-forwarded boxes.
    /// `setinfo_from_preamble_item` below is the shared dispatcher: it
    /// does the checks once and then routes to the variant-specific
    /// logic, so this method becomes the literal unroll.py loop body.
    fn setinfo_from_preamble_list(
        &mut self,
        items: &[majit_ir::operand::Operand],
        exported_infos: &indexmap::IndexMap<
            majit_ir::operand::Operand,
            crate::optimizeopt::info::OpInfo,
        >,
    ) {
        for item in items {
            // unroll.py:42-43: if item is None: continue
            if item.to_opref().is_none() {
                continue;
            }
            // unroll.py:45-46: i = infos.get(item, None) — keyed by the field
            // box. `item` is read from the SAME shared virtual `PtrInfo` that
            // the exporter walked (`_expand_infos_from_virtual`), so the
            // box-identity (`Rc::ptr_eq`) lookup hits exactly when RPython's
            // box-keyed dict does.
            match exported_infos.get(item).cloned() {
                Some(info) => {
                    // unroll.py: self.setinfo_from_preamble(item, i, infos)
                    self.setinfo_from_preamble_item(item, &info, exported_infos);
                }
                None => {
                    // unroll.py:49: item.set_forwarded(None)
                    // "let's not inherit stuff we don't know anything about".
                    // Clears `item`'s OWN forwarded slot directly — `item` is
                    // the raw field box (RPython passes the unresolved
                    // `all_items()` box here). `clear_forwarded` is a no-op for
                    // a Const box (Const has no `_forwarded`), matching RPython
                    // where `Const.set_forwarded` raises.
                    item.clear_forwarded();
                }
            }
        }
    }

    /// unroll.py `setinfo_from_preamble(op, preamble_info, exported_infos)`.
    ///
    /// Shared dispatcher covering the `isinstance(preamble_info, ...)` chain
    /// at unroll.py, 93, 97 — used by both `setinfo_from_preamble_list`
    /// (mod.rs recursive virtual field walker) and `OptUnroll::import_state`
    /// (unroll.rs top-level import). Centralising the dispatch avoids
    /// diverging shortcuts that skip the early-return checks.
    fn setinfo_from_preamble_item(
        &mut self,
        op: &majit_ir::operand::Operand,
        preamble_info: &crate::optimizeopt::info::OpInfo,
        exported_infos: &indexmap::IndexMap<
            majit_ir::operand::Operand,
            crate::optimizeopt::info::OpInfo,
        >,
    ) {
        use crate::optimizeopt::info::OpInfo;
        // unroll.py:53-54 `op = get_box_replacement(op)`
        let target = op.get_box_replacement(false);
        // unroll.py:55-56 `if op.get_forwarded() is not None: return`
        if self.has_forwarding(&target) {
            return;
        }
        // unroll.py:57-58 `if op.is_constant(): return`
        if target.const_value().is_some() {
            return;
        }
        // Every arm below writes through `target`, so it has to be the
        // canonical `_forwarded` host rather than a merely chain-resolved
        // operand — see the sibling `setinfo_from_preamble` on why the mint is
        // load-bearing for guard resume numbering.
        let target = self.materialize_write_host(target);
        match preamble_info {
            // unroll.py:65-68 ConstPtrInfo: set_forwarded(preamble_info.getconst())
            // unroll.py:59-92 general PtrInfo dispatch. The `Ptr` arm now
            // carries the `Rc<RefCell<PtrInfo>>` handle; borrow once to
            // dispatch on the inner variant.
            OpInfo::Ptr(rc) => {
                let const_gcref = match &*rc.borrow() {
                    crate::optimizeopt::info::PtrInfo::Constant(gcref) => Some(*gcref),
                    _ => None,
                };
                if let Some(gcref) = const_gcref {
                    self.make_constant_box(&target, Value::Ref(gcref));
                } else {
                    // Pass the Rc handle so the virtual branch can
                    // preserve `_forwarded` object identity per
                    // unroll.py:61.
                    self.setinfo_from_preamble(&target, rc, Some(exported_infos));
                }
            }
            // unroll.py:93-96 IntBound with widen(): intersect unconditionally.
            OpInfo::IntBound(bound) => {
                let widened = bound.borrow().widen();
                self.with_intbound_mut(&target, |bm| {
                    let _ = bm.intersect(&widened);
                });
            }
            // unroll.py:97-98 FloatConstInfo: op.set_forwarded(preamble_info._const)
            OpInfo::FloatConstInfo(f) => {
                self.make_constant_box(&target, Value::Float(f.getconst()));
            }
            // unroll.py:53-98 has no dispatch arm for "no info" — the
            // caller never stores an `Unknown` entry in `exported_infos`
            // (see `collect_exported_info`'s `None` return
            // in `unroll.rs`, mirroring unroll.py `if info:`).
            OpInfo::Unknown | OpInfo::EmptyInfo(_) => unreachable!(
                "exported_infos must never contain OpInfo::Unknown/EmptyInfo; \
                 the absent-entry branch (clear_forwarded) handles that case"
            ),
        }
    }

    fn setinfo_from_preamble_item_option(
        &mut self,
        op: &Operand,
        preamble_info: &crate::optimizeopt::info::OpInfo,
        exported_infos: Option<
            &indexmap::IndexMap<majit_ir::operand::Operand, crate::optimizeopt::info::OpInfo>,
        >,
    ) {
        use crate::optimizeopt::info::OpInfo;
        // optimizer.py `setinfo_from_preamble`: `op = get_box_replacement(op)`;
        // `if op.get_forwarded() is not None: return`; `if isinstance(op,
        // Const): return`.
        let target = self.resolve_operand_operand(op);
        if target.const_value().is_some() || self.has_forwarding(&target) {
            return;
        }
        match preamble_info {
            OpInfo::Ptr(rc) => {
                // Pass the Rc handle (unroll.py:61 identity preservation).
                self.setinfo_from_preamble(&target, rc, exported_infos);
            }
            OpInfo::IntBound(bound) => {
                let widened = bound.borrow().widen();
                self.with_intbound_mut(&target, |bm| {
                    let _ = bm.intersect(&widened);
                });
            }
            OpInfo::FloatConstInfo(f) => {
                self.make_constant_box(&target, Value::Float(f.getconst()));
            }
            OpInfo::Unknown | OpInfo::EmptyInfo(_) => {}
        }
    }

    /// `optimizer.py` `preamble_op = self.optunroll.potential_extra_ops.pop(op)`.
    pub fn take_potential_extra_op(
        &mut self,
        result: &Operand,
    ) -> Option<crate::optimizeopt::info::PreambleOp> {
        self.potential_extra_ops.swap_remove(result)
    }

    /// `unroll.py` `self.optunroll.potential_extra_ops[op] = preamble_op` —
    /// dict-assign semantics: overwrite if `key` exists, else append.
    pub fn set_potential_extra_op(
        &mut self,
        key: Operand,
        preamble_op: crate::optimizeopt::info::PreambleOp,
    ) {
        self.potential_extra_ops.insert(key, preamble_op);
    }

    /// Dict-`in` parity for `potential_extra_ops`.
    pub fn has_potential_extra_op(&self, key: &Operand) -> bool {
        self.potential_extra_ops.contains_key(key)
    }

    pub fn activate_short_preamble_producer(
        &mut self,
        builder: crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder,
    ) {
        self.active_short_preamble_producer = Some(builder);
    }

    pub fn active_short_preamble_producer_mut(
        &mut self,
    ) -> Option<&mut crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder> {
        self.active_short_preamble_producer.as_mut()
    }

    /// Address of the active producer's storage for the MetaInterp root
    /// walker. This field has the same
    /// `Option<ExtendedShortPreambleBuilder>` type as
    /// `Optimizer.short_preamble_producer`, so either address is valid for
    /// the walker's cast while the builder is moved between them.
    pub(crate) fn active_short_preamble_producer_slot_addr(&mut self) -> usize {
        (&mut self.active_short_preamble_producer
            as *mut Option<crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder>)
            as usize
    }

    pub fn build_active_short_preamble(
        &self,
    ) -> Option<crate::optimizeopt::shortpreamble::ShortPreamble> {
        self.active_short_preamble_producer.as_ref().map(|builder| {
            // history.py/268/314 — `Const{Int,Float,Ptr}.value` rides
            // inline on the OpRef; producer-side `make_constant` writes
            // inline-Const variants into `op.args` directly, so no
            // `loop_constants` snapshot is needed at this boundary.
            builder.build_short_preamble_struct()
        })
    }

    pub fn take_active_short_preamble_producer(
        &mut self,
    ) -> Option<crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder> {
        self.active_short_preamble_producer.take()
    }

    pub fn build_imported_short_preamble(
        &self,
    ) -> Option<crate::optimizeopt::shortpreamble::ShortPreamble> {
        self.imported_short_preamble_builder
            .as_ref()
            .map(|builder| {
                // history.py/268/314 — `Const{Int,Float,Ptr}.value`
                // rides inline on the OpRef. `make_constant` mints
                // `Const*` variants directly into `op.args`, so the
                // cross-compile `loop_constants` snapshot is empty
                // along every production path and the builder's
                // `known_constants` set picks up Const operands intrinsically
                // via `OpRef::is_constant()`.
                builder.build_short_preamble_struct()
            })
    }

    pub fn used_imported_short_aliases(&self) -> Vec<ImportedShortAlias> {
        self.imported_short_preamble_builder
            .as_ref()
            .map(|builder| {
                builder
                    .extra_same_as()
                    .iter()
                    .map(|op| ImportedShortAlias {
                        result: op.pos().get(),
                        same_as_source: op.arg(0),
                        same_as_opcode: op.opcode,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// `pure_from_args1(opnum, arg0, op)`: `key` is the pure op's arg,
    /// `forwarded` is the box the synthetic op forwards to.
    /// Consumed by OptPure before the next propagate.
    pub fn register_pure_from_args1(&mut self, opcode: OpCode, key: Operand, forwarded: Operand) {
        self.pending_pure_from_args
            .push((opcode, key, forwarded, None));
    }

    /// `pure_from_args(opnum, [arg], result, descr=...)`.
    /// `descr` discriminates the pure cache slot so cross-descr
    /// collisions (ARRAYLEN_GC across distinct array descrs) don't
    /// collapse onto the same key.
    pub fn register_pure_from_args1_with_descr(
        &mut self,
        opcode: OpCode,
        key: Operand,
        forwarded: Operand,
        descr: majit_ir::DescrRef,
    ) {
        self.pending_pure_from_args
            .push((opcode, key, forwarded, Some(descr)));
    }

    /// `pure_from_args(ARRAYLEN_GC, [op], ConstInt(len))`.
    pub fn pure_from_args_arraylen(&mut self, array: Operand, length: i64) {
        let len = Operand::const_from_value(Value::Int(length));
        self.register_pure_from_args1(OpCode::ArraylenGc, array, len);
    }

    /// Probe the pure-op ring for `op`. Empty until `OptPure` publishes.
    pub fn get_pure_result(&mut self, op: &Op) -> Option<Operand> {
        let table = self.pure_ops.clone()?;
        pure::shared_get_pure_result(&table, op, self)
    }

    /// `pure_from_args2(opnum, arg0, arg1, op)`: `forwarded` is the box
    /// the synthetic op forwards to. Consumed by OptPure before the next
    /// propagate.
    pub fn register_pure_from_args2(
        &mut self,
        opcode: OpCode,
        forwarded: Operand,
        arg0: Operand,
        arg1: Operand,
    ) {
        self.pending_pure_from_args2
            .push((opcode, arg0, arg1, forwarded));
    }

    /// Grain's reds (`Vm`, frame, `Scope`) live on the thread stack and
    /// change address every `Vm::new`. Folding such an InputArg to
    /// `ConstPtr` makes resume number it TAGCONST, so the next evaluation
    /// has no live box to rebind. PyPy heap frames have stable GC identity
    /// and stay foldable: only a Ref whose bits sit in this thread's stack
    /// window is refused. Null / low sentinels (`addr <= 0x1000`) are not
    /// stack objects.
    fn operand_is_stack_resident_inputarg(op: &Operand) -> bool {
        if !op.is_inputarg() {
            return false;
        }
        match op.get_value() {
            Some(Value::Ref(gcref)) => Self::ref_addr_is_stack_resident(gcref.0),
            _ => false,
        }
    }

    /// True when `addr` lives in this thread's stack window.
    ///
    /// Native: 16 MiB around a stack probe. wasm32: `__stack_low` ..
    /// `__stack_high`, because one linear memory also holds the nursery.
    pub(crate) fn ref_addr_is_stack_resident(addr: usize) -> bool {
        if addr <= 0x1000 {
            return false;
        }
        #[cfg(target_arch = "wasm32")]
        {
            // One linear memory holds the shadow stack, the data segment and
            // the nursery within its first few tens of megabytes, so distance
            // from a stack probe says nothing about residency. wasm-ld
            // synthesizes the shadow stack's bounds when they are referenced.
            unsafe extern "C" {
                static __stack_low: u8;
                static __stack_high: u8;
            }
            let low = std::ptr::addr_of!(__stack_low) as usize;
            let high = std::ptr::addr_of!(__stack_high) as usize;
            (low..high).contains(&addr)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let probe = 0usize;
            (std::ptr::addr_of!(probe) as usize).abs_diff(addr) < 16 * 1024 * 1024
        }
    }

    /// `optimizer.py make_equal_to(op, newop)` (line-by-line port):
    ///
    /// ```python
    /// def make_equal_to(self, op, newop):
    ///     op = get_box_replacement(op)
    ///     if op is newop: return
    ///     opinfo = op.get_forwarded()
    ///     op.set_forwarded(newop)
    ///     if opinfo is not None and not newop.is_constant():
    ///         newop.set_forwarded(opinfo)
    /// ```
    pub fn make_equal_to(&mut self, op: &Operand, newop: &Operand) {
        // optimizer.py:381 Const.set_forwarded asserts; pyre no-ops the
        // chain head when `op` is itself a Const so callers can fold const
        // sources without an explicit guard.
        if op.is_constant() {
            return;
        }
        // A stack-resident red (Grain's `Vm` / frame / `Scope`) folded
        // to `ConstPtr` becomes TAGCONST in resume numbering
        // (`isinstance(box, Const)` after `get_box_replacement`). The
        // next evaluation has a different stack address, so the live
        // InputArg must stay a TAGBOX.
        if newop.is_constant()
            && (Self::operand_is_stack_resident_inputarg(op)
                || (op.is_inputarg()
                    && newop.get_value().is_some_and(
                        |v| matches!(v, Value::Ref(g) if Self::ref_addr_is_stack_resident(g.0)),
                    )))
        {
            return;
        }
        // optimizer.py op = get_box_replacement(op)
        let op = op.get_box_replacement(false);
        // optimizer.py:387 Box.type invariant: cross-type forwards would
        // silently retype the chain head. Always-on (not `debug_assert_eq!`)
        // for parity with the Const-invariant `assert!`s in `set_forwarded_*`;
        // asserted on the already-chain-walked `op` so it costs no extra walk.
        // The message names both sides: which two operations disagree is the
        // whole content of this failure, and the operand `Debug` walks the
        // abstract-value graph, so it cannot be printed here.
        assert_eq!(
            op.type_(),
            newop.type_(),
            "make_equal_to: cross-type forward (Box.type invariant): {} <- {}",
            describe_operand_for_assert(&op),
            describe_operand_for_assert(newop),
        );
        // optimizer.py:392 if op is newop: return
        if &op == newop {
            return;
        }
        if op.is_constant() {
            return;
        }
        // optimizer.py:393 opinfo = op.get_forwarded()
        use crate::optimizeopt::info::OpInfo;
        let info_to_transfer: Option<OpInfo> = match &op.get_forwarded() {
            majit_ir::forwarding::Forwarded::Info(
                opinfo @ (OpInfo::Ptr(_) | OpInfo::IntBound(_) | OpInfo::FloatConstInfo(_)),
            ) => Some(opinfo.clone()),
            _ => None,
        };
        // optimizer.py:394 op.set_forwarded(newop). RPython's `newop` is
        // always a real box object. A position-only unbound operand — the
        // flat-OpRef artifact where a Phase-1 producer was not carried into
        // this rebuilt Phase-2 OptContext (`find_producer_op` miss, the
        // cross-phase resolution gap; e.g. `import_state`'s next-iteration
        // target, a heap/virtual cache entry) — is materialized to its
        // canonical `SameAs*` stand-in (`resoperation.py` "the box
        // always exists"). The forward then targets a bound `Op`/`InputArg`
        // and never a position-only unbound-box redirect. An unbound
        // `newop` has no producer at its position, so it cannot alias the
        // bound chain head `op` (no self-cycle).
        let materialized_newop;
        let newop: &Operand = if newop.is_constant()
            || newop.bound_op().is_some()
            || newop.bound_inputarg().is_some()
        {
            newop
        } else {
            materialized_newop = self.materialize_operand_at(newop.to_opref());
            &materialized_newop
        };
        if newop.is_constant() {
            let value = newop
                .const_value()
                .expect("is_constant() implies const_value() Some");
            op.set_forwarded_const(majit_ir::Const::from_value(value));
        } else if let Some(target_op) = newop.bound_op() {
            // Op-target chain step: route through Forwarded::Op(Rc<Op>)
            // so the chain refers to the canonical Rc<Op> (PyPy
            // resoperation.py set_forwarded(forwarded_to) where
            // forwarded_to is an AbstractResOp), i.e. the
            // `Forwarded::Op` forwarding step.
            //
            // `optimizer.py:392 if op is newop: return` — PyPy's
            // identity check uses Python `is`; after `bind_op`, two
            // separate `Rc<Box>` wrappers can share the same canonical
            // `OpRc`, so `&op == newop` (which compares the `Rc<Box>`)
            // misses that case and falls through to `set_forwarded_op`,
            // tripping `set_forwarded_op`'s self-cycle assert. Honour
            // the upstream `is` semantics by comparing the bound `Op`
            // identities first.
            if op.bound_op().is_some_and(|o| OpRc::ptr_eq(&o, &target_op)) {
                return;
            }
            op.set_forwarded_op(&target_op);
        } else if let Some(target_ia) = newop.bound_inputarg() {
            // InputArg-target chain step (compile.py:478, unroll.py).
            // Same `optimizer.py:392` idempotent gate as the
            // `bound_op` arm above, against the bound `InputArg`
            // identities.
            if op
                .bound_inputarg()
                .is_some_and(|i| majit_ir::InputArgRc::ptr_eq(&i, &target_ia))
            {
                return;
            }
            op.set_forwarded_inputarg(&target_ia);
        } else {
            // Unreachable: `newop` was normalized above to a constant or a
            // bound `Op`/`InputArg`, so the dispatch always resolves through
            // `set_forwarded_const` / `set_forwarded_op` / `set_forwarded_inputarg`.
            // The orphan unbound-box redirect that the deleted `Forwarded::Box`
            // variant once encoded no longer exists.
            unreachable!(
                "make_equal_to: newop must be constant or bound after \
                 materialize_operand_at normalization (orphan unbound-box \
                 redirect retired)"
            );
        }
        // optimizer.py:395-396
        //   if opinfo is not None and not newop.is_constant():
        //       newop.set_forwarded(opinfo)
        if let Some(opinfo) = info_to_transfer
            && !newop.is_constant()
        {
            newop.set_forwarded_info(opinfo);
        }
    }

    /// info.py `mark_last_guard(optimizer)` parity (line-by-line port).
    ///
    /// ```python
    /// def mark_last_guard(self, optimizer):
    ///     if (optimizer.getlastop() is None or
    ///             not optimizer.getlastop().is_guard()):
    ///         return
    ///     self.last_guard_pos = len(optimizer._newoperations) - 1
    ///     assert self.get_last_guard(optimizer).is_guard()
    /// ```
    ///
    /// Upstream defines this as a method ON `PtrInfo`
    /// (`opinfo.mark_last_guard(self.optimizer)` per optimizer.py);
    /// pyre keeps it at the `OptContext` level so the chain walk and
    /// `ptr_info_mut` interior-mutability stay together. Semantics match
    /// the upstream method: no-op when the last emitted op is not a guard,
    /// otherwise stamps `last_guard_pos = len(_newoperations) - 1` on the
    /// terminal box's PtrInfo.
    pub fn mark_last_guard(&self, op: &Operand) {
        // info.py:112-116: optimizer.getlastop().is_guard() check
        let pos = match self.new_operations.last() {
            Some(o) if o.opcode.is_guard() => (self.new_operations.len() - 1) as i32,
            _ => return,
        };
        // info.py:117: self.last_guard_pos = pos
        // `_forwarded` PtrInfo lives on the terminal of the chain.
        let resolved = op.get_box_replacement(false);
        if let Some(mut info) = resolved.ptr_info_mut() {
            info.set_last_guard_pos(pos);
        }
    }

    /// info.py `get_last_guard(optimizer)` parity (line-by-line port).
    ///
    /// ```python
    /// def get_last_guard(self, optimizer):
    ///     if self.last_guard_pos == -1:
    ///         return None
    ///     return optimizer._newoperations[self.last_guard_pos]
    /// ```
    ///
    /// Upstream is a `PtrInfo` method; pyre keeps it at the `OptContext`
    /// level so the chain walk and `ptr_info()` read stay together with
    /// the `_newoperations` index. Returns the guard `Op` at the PtrInfo's
    /// stored `last_guard_pos`, or `None` when the slot is `-1` (no guard
    /// recorded) or the operand has no PtrInfo.
    pub fn get_last_guard(&self, op: &Operand) -> Option<&Op> {
        if self.new_operations_drained {
            return None;
        }
        // info.py: read last_guard_pos from terminal PtrInfo.
        let resolved = op.get_box_replacement(false);
        let pos = resolved.ptr_info().and_then(|p| p.get_last_guard_pos())?;
        let op = self.new_operations.get(pos).map(|rc| rc.as_ref())?;
        debug_assert!(
            op.opcode.is_guard(),
            "info.py:118 last_guard_pos must denote a guard"
        );
        Some(op)
    }

    /// resoperation.py get_box_replacement: follow the forwarding
    /// chain (op._forwarded) until we reach a terminal. RPython: walks
    /// op → op._forwarded → ... until None or Info instance.
    ///
    /// RPython invariant: get_box_replacement NEVER returns None.
    /// `_forwarded = None` means "no forwarding" (terminal), NOT
    /// "forwarded to None".
    ///
    /// NEVER consults mapping dicts — RPython's get_box_replacement only
    /// follows the _forwarded chain on the box itself.
    ///
    /// `_forwarded` is a single slot per operand (matching RPython's
    /// single Python slot per box). The walker advances through
    /// `Forwarded::Op`/`Forwarded::InputArg` and terminates at `None` /
    /// `Forwarded::Info(_)` / a Const target's reconstructed
    /// `OpRef::const_int/float/ptr`.
    fn get_box_replacement_impl(&self, opref: OpRef, not_const: bool) -> OpRef {
        if opref.is_constant() || opref.is_none() {
            return opref;
        }
        let Some(start) = self.resolve_to_operand(opref) else {
            return opref;
        };
        // resoperation.py:57-68: walk the operand's `_forwarded` chain.
        let terminal = start.get_box_replacement(not_const);
        // When the walker did not advance — chain root has Forwarded::None,
        // Forwarded::Info(_), or (not_const=true and the immediate target
        // is Const) — return the source OpRef variant unchanged. The walker
        // terminated before reading position()/type_(), so callers expect
        // the OpRef shape they passed in.
        if start == terminal {
            return opref;
        }
        // Const targets reconstruct their inline source OpRef from the value
        // (history.py:227/268/314), so `operand_to_opref` reconstruction is
        // direct — `Operand::Const::to_opref` rebuilds the inline-Const OpRef.
        self.operand_to_opref(&terminal, opref)
    }

    /// Convert a chain-walk terminal [`Operand`] back into an `OpRef` — the
    /// OpRef-side glue around [`Operand::get_box_replacement`]; PyPy callers
    /// hold the operand directly and skip this step.
    ///
    /// A `Const` terminal reconstructs its inline source OpRef from the value
    /// (`history.py:227/268/314`); a position-bearing terminal rebuilds
    /// `input_arg_typed` / `op_typed` from its `position()` + `type_()`. A
    /// `Type::Void` phantom is a lazy-allocated placeholder (the
    /// `bound_from_opref` fallback for OpRef variants with no `ty()`) carrying
    /// no type information, so it preserves the `source` variant via `with_raw`
    /// instead of promoting to `void_op` / `input_arg_typed(_, Void)`.
    fn operand_to_opref(&self, terminal: &Operand, source: OpRef) -> OpRef {
        if terminal.const_value().is_some() {
            return terminal.to_opref();
        }
        if let Some(pos) = terminal.position() {
            let tp = terminal.type_();
            if matches!(tp, majit_ir::Type::Void) {
                return source.with_raw(pos);
            }
            if terminal.is_inputarg() {
                return OpRef::input_arg_typed(pos, tp);
            }
            if terminal.is_resop() {
                return OpRef::op_typed(pos, tp);
            }
        }
        source
    }

    /// [`Operand`]-yielding total mirror of [`get_box_replacement`](Self::get_box_replacement):
    /// resolve a position's `_forwarded` terminal directly as an `Operand`
    /// (canonical `Op` / `InputArg` host, or inline-`Const`) without minting a
    /// wrapper. Walks the chain via [`resolve_to_operand`](Self::resolve_to_operand)
    /// + [`Operand::get_box_replacement`].
    ///
    /// Total, like the operand sibling [`Operand::get_box_replacement`]
    /// (returns the position-only operand on a miss) and
    /// `resoperation.py get_box_replacement` (returns the last
    /// `AbstractResOpOrInputArg` before `None` / Info / a rejected Const).
    /// A position that resolves to neither a producer `Op`, an
    /// `inputarg_refs` slot, nor a Const is a bind-at-alloc gap: production
    /// write-sites mint via `reserve_virtual_box` / `mint_box_at`. The
    /// `bound_from_opref` arm is a debug-asserted release fallback (two
    /// calls for the same unbound OpRef are not `ptr_eq`).
    pub(crate) fn get_box_replacement_operand(&self, opref: OpRef) -> Operand {
        if opref.is_none() {
            return Operand::None;
        }
        if let Some(start) = self.resolve_to_operand(opref) {
            return start.get_box_replacement(false);
        }
        debug_assert!(
            cfg!(test),
            "get_box_replacement_operand S9 bound_from_opref fallback: {opref:?} \
             has no producer; mint at allocation (mint_box_at / reserve_virtual_box)"
        );
        self.s9_probe_fire(opref);
        Operand::bound_from_opref(opref)
    }

    /// Write receiver for `op.set_forwarded(info)` on an already
    /// chain-resolved operand. A bound ResOp is its own host; its position is
    /// made to resolve to it, since guard resume numbering reads by position.
    /// Other operands go through `materialize_operand_at`. The `None`
    /// sentinel passes through untouched.
    pub(crate) fn materialize_write_host(&mut self, o: Operand) -> Operand {
        let pos = o.to_opref();
        if pos.is_none() {
            return o;
        }
        let Some(op) = o.bound_op() else {
            return self.materialize_operand_at(pos);
        };
        // `op.set_forwarded(info)` writes the chain terminal itself. A
        // terminal its position does not resolve to becomes the position's
        // producer, and an unforwarded stand-in registered there is
        // forwarded to it, so position readers reach the same box.
        match self.find_producer_op(pos) {
            Some(p) if OpRc::ptr_eq(&p, &op) => {}
            Some(p) => {
                use majit_ir::forwarding::ForwardingHost;
                if matches!(p.get_forwarded(), majit_ir::forwarding::Forwarded::None) {
                    p.set_forwarded_op(&op);
                }
            }
            None => {
                self.resop_refs.insert(pos, op);
            }
        }
        o
    }

    /// "Box always exists" materializer (`resoperation.py AbstractResOpOrInputArg._forwarded`).
    /// Returns the canonical bound `_forwarded` host for `opref` as an
    /// [`Operand`] (`Op` / `InputArg`). For a const-namespace OpRef
    /// returns a fresh `Operand::Const` (`history.py` no-dedup; Const
    /// boxes have no `_forwarded`, so any write the caller attempts is a
    /// no-op). The InputArg arm binds a missing `inputarg_refs` slot.
    /// A producer-less resop is a bind-at-alloc gap: production write-sites
    /// mint via `reserve_virtual_box` / `mint_box_at`. The resop mint arm
    /// is a debug-asserted release fallback. Unlike `resolve_to_operand`
    /// this never returns `None` for a value-bearing OpRef. The sentinel
    /// `OpRef::none()` has no operand (debug-asserted); resolve it with
    /// `resolve_to_operand` / `get_box_replacement_operand` instead.
    pub(crate) fn materialize_operand_at(&mut self, opref: OpRef) -> Operand {
        debug_assert!(
            !opref.is_none(),
            "materialize_operand_at: sentinel OpRef::none() has no operand"
        );
        if opref.is_constant() {
            // history.py/261/307: a Const carries its Value on the OpRef.
            let value = self.get_constant(opref).unwrap_or_else(|| {
                panic!(
                    "materialize_operand_at: const OpRef {opref:?} carries no Value — \
                     a Const carries its Value (history.py:220/261/307)"
                )
            });
            return Operand::const_from_value(value);
        }
        // Align the write-path host with `resolve_to_operand` (the read
        // path). For ResOp variants, resolve to the producing `Op`'s
        // canonical `_forwarded` host first. `find_producer_op`
        // distinguishes the ResOp namespace from the InputArg namespace by
        // full `OpRef` (`op.pos == opref`), so a raw-slot position-collapse
        // — where one slot index served both a ResOp and an InputArg — does
        // not route a ResOp write to `inputarg_refs[idx].forwarded` while
        // the matching read routes to `op.forwarded`. Returns `None` for
        // InputArg / input positions (no producing op), falling through to
        // the arms below.
        if let Some(op_rc) = self.find_producer_op(opref) {
            return Operand::from_bound_op(&op_rc);
        }
        match opref {
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) => {
                // InputArg write path: route through the canonical
                // `inputarg_refs` host (symmetric with
                // `resolve_to_operand`'s InputArg branch and the
                // `read_forwarded` / `clear_forwarded` writers), so a
                // `set_forwarded_*` write lands the same
                // `InputArg.forwarded` slot a later `resolve_to_operand`
                // read observes. Materialize / repair the slot's canonical
                // `InputArgRc` by the canonical `inputargs` slot type — the
                // OpRef variant tag can disagree across a phase boundary (a
                // Phase-2 OpRef referencing a Phase-1 low slot), so it is a
                // fallback only when no canonical type is recorded
                // (`resoperation.py InputArgInt/727/739` + `:233` the `_forwarded`
                // host).
                let idx = opref.raw();
                let tp = self
                    .inputarg_type(opref)
                    .unwrap_or_else(|| opref.ty().unwrap_or(majit_ir::Type::Void));
                self.bind_canonical_inputarg(idx as usize, tp);
                Operand::from_bound_inputarg(&self.inputarg_refs[&idx])
            }
            _ => {
                // Producer-less resop: production write-sites mint at
                // allocation (`reserve_virtual_box` / `mint_box_at`). This
                // arm is a bind-at-alloc gap (`resoperation.py
                // ResOperation(...)` creates the box immediately).
                debug_assert!(
                    cfg!(test),
                    "materialize_operand_at resop arm: {opref:?} has no producer; \
                     mint at allocation (mint_box_at / reserve_virtual_box)"
                );
                self.mint_box_at(opref)
            }
        }
    }

    /// S9 probe classifier for a `from_opref` fallback fire (see
    /// [`OptContext::get_box_replacement`]). `&self`-only, no mutation.
    ///
    /// Precondition: `find_producer_op` already missed (full type-tagged
    /// `OpRef` match across `new_operations` / `phase1_emit_ops` / `resop_refs`
    /// / `input_ops`) and `resolve_to_operand` returned `None`; `opref` is
    /// non-`none`, non-`Const`.
    ///
    /// - `b-inputarg`: an InputArg position whose `inputarg_refs` slot is unbound.
    /// - `b-typetag`: a producer Op exists at the same RAW position under a
    ///   different type tag (the #97 type-tag divergence signal).
    /// - `b-rekey-*`: the OpRef is named by a carried Phase-1 registry
    ///   (`inputargs` / `imported_label_args` / `imported_virtual_args`), so a
    ///   producer exists in Phase 1 but is not in a `find_producer_op` store.
    /// - `b-shortpure`: the OpRef is the `result` of a carried
    ///   `imported_short_pure_ops` entry — a short-preamble producer survives
    ///   but `find_producer_op` does not consult that table.
    /// - `a-producerless`: no producer Op for this position is reachable in any
    ///   OptContext-local registry — correct-as-is (an unforwarded box returns
    ///   itself, `resoperation.py`).
    fn classify_s9_fallback(&self, opref: OpRef) -> &'static str {
        match opref {
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) => {
                return "b-inputarg";
            }
            _ => {}
        }
        let raw = opref.raw();
        let raw_hit = self
            .new_operations
            .iter()
            .chain(self.phase1_emit_ops.iter())
            .chain(self.input_ops.iter())
            .any(|op| op.pos().get().raw() == raw);
        if raw_hit {
            return "b-typetag";
        }
        if self.inputargs.contains(&opref) {
            return "b-rekey-inputargs";
        }
        if let Some(label_args) = self.imported_label_args.as_ref()
            && label_args.contains(&opref)
        {
            return "b-rekey-label";
        }
        if let Some((_, vargs)) = self.imported_virtual_args.as_ref()
            && vargs.contains(&opref)
        {
            return "b-rekey-vargs";
        }
        if self
            .imported_short_pure_ops
            .iter()
            .any(|sp| sp.result == opref)
        {
            return "b-shortpure";
        }
        "a-producerless"
    }

    /// S9 probe sink (env-gated). No-op unless `MAJIT_S9_PROBE` is set; appends
    /// one classified line per `from_opref` fallback fire to
    /// `/tmp/majit_s9_probe.log`. Process-global file sink because the caller is
    /// `&self` (no context-local counter). When the flag is unset, only a
    /// cached bool is read and the sink is never opened.
    fn s9_probe_fire(&self, opref: OpRef) {
        use parking_lot::Mutex;
        use std::sync::LazyLock;
        static ENABLED: LazyLock<bool> =
            LazyLock::new(|| std::env::var_os("MAJIT_S9_PROBE").is_some());
        if !*ENABLED {
            return;
        }
        static SINK: LazyLock<Mutex<std::fs::File>> = LazyLock::new(|| {
            Mutex::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("/tmp/majit_s9_probe.log")
                    .expect("open /tmp/majit_s9_probe.log"),
            )
        });
        let cat = self.classify_s9_fallback(opref);
        // `MAJIT_S9_PROBE=bt` additionally captures the caller backtrace per
        // fire, to attribute fallback fires to their `get_box_replacement`
        // call sites.
        static BT: LazyLock<bool> = LazyLock::new(|| {
            std::env::var("MAJIT_S9_PROBE")
                .map(|v| v == "bt")
                .unwrap_or(false)
        });
        {
            let mut f = SINK.lock();
            use std::io::Write;
            let _ = writeln!(f, "cat={} opref={:?} raw={}", cat, opref, opref.raw());
            if *BT {
                let _ = writeln!(f, "{}", std::backtrace::Backtrace::force_capture());
            }
        }
    }

    /// Native `Operand`-in / `Operand`-out resolver. Resolves an operand to its
    /// `_forwarded` terminal WITHOUT minting a wrapper — the box-native walk
    /// (`arg.get_box_replacement`) and the `OpRef`-store fallback
    /// (`get_box_replacement_operand`) both
    /// stay on the `Operand` carrier. Two arms: a bound / const operand walks
    /// its own chain and defers to the store only when it self-resolves; an
    /// unbound operand resolves positionally.
    ///
    /// `Operand` is the only carrier — the Box-carrier resolver family this
    /// used to be described against was removed with `BoxRef` itself.
    ///
    /// The heal keys on the bound `Op` host directly off the `Operand`; the
    /// native walk (`arg.get_box_replacement`) is byte-identical to the legacy
    /// resolve-then-rewrap, asserted below.
    /// The loop-carried operand for a trace input arg that a peeled body
    /// names by its recorder slot.
    ///
    /// Phase 2 optimizes the recorded operations a second time, so a body
    /// operation still carries `InputArg(i)` as the recorder emitted it,
    /// while `import_state` installs that iteration's value on the phase 2
    /// host at `inputarg_base + i`. Reading slot `i` alone finds the
    /// unforwarded recorder host and loses the import, so a slot the
    /// preamble proved constant would reach the body as an unknown.
    /// `OptUnroll.import_state`'s `source.set_forwarded(target)` writes onto the very
    /// box the body operations hold; the flat `OpRef` namespace splits that
    /// one box in two, and this rejoins them.
    ///
    /// `None` when there is nothing to rejoin: phase 1 (`inputarg_base` 0),
    /// a position outside the input args, or a host that resolves back to
    /// `arg` itself — the identity import a pass-through slot records.
    fn imported_inputarg_operand(&self, arg: &Operand) -> Option<Operand> {
        if self.inputarg_base == 0 || !arg.is_inputarg() {
            return None;
        }
        let slot = arg.to_opref().raw();
        if slot >= self.num_inputs {
            return None;
        }
        let host = OpRef::input_arg_typed(self.inputarg_base + slot, arg.type_());
        let imported = self.get_box_replacement_operand_opt(host)?;
        if imported.same_box(arg) {
            return None;
        }
        // Only a value that means the same thing on every entry to the body
        // may stand for a body operand's label arg.  A constant does, and so
        // does another input arg: both are fixed for the whole loop, and the
        // preamble-proved constants this rejoin exists for are exactly the
        // first case.
        //
        // A result produced inside the trace does not.  The loop redefines it
        // every iteration, so substituting it turns "the value this iteration
        // started with" into "the value it ends with".  That is not a
        // theoretical distinction: a virtualized field store is re-resolved
        // late, when `_force_elements` emits it at the loop-closing jump, so
        // `head = Node(i, head)` would store the node the body just allocated
        // into that node's own `next`.
        //
        // `OptUnroll.import_state`'s `source.set_forwarded(target)` unifies the label arg
        // with the value the preamble jumps with, and those name one runtime
        // value only at loop entry.
        //
        // Another input arg does not qualify either, though it is fixed for
        // the whole loop: substituting one label arg for another says the two
        // slots hold the same value on EVERY entry, which is a claim about the
        // loop rather than about this iteration. A slot a residual writes
        // behind the trace's back — a frame local reached through the running
        // frame's `f_locals` — then reads as loop-invariant, and the label arg
        // and the reload after the call both fall away, so the loop keeps
        // answering the value it was compiled with. Only a constant is fixed
        // in the sense this rejoin needs, and a preamble-proved constant is
        // the case it exists for.
        imported.is_constant().then_some(imported)
    }

    /// Debug witness: `got` is the box `get_box_replacement` reached from a
    /// caller-held operand, `probed` is that operand's position. When the
    /// position has a producer, the two terminals are the same box.
    #[cfg(debug_assertions)]
    pub(crate) fn debug_assert_operand_matches_positional(
        &self,
        got: &Operand,
        probed: OpRef,
        site: &str,
    ) {
        if probed.is_none() {
            return;
        }
        if let Some(old) = self.get_box_replacement_operand_opt(probed) {
            debug_assert!(
                got.same_box(&old),
                "{site} {probed:?}: new {:?} old {:?}",
                got.to_opref(),
                old.to_opref()
            );
        }
    }

    /// `get_box_replacement(op)`: walk the operand's own `_forwarded`
    /// chain. A self-resolved InputArg in the peeled body keeps the
    /// constant rejoin of `imported_inputarg_operand`. Debug builds check
    /// the result against the positional resolver, which still consults
    /// the `OpRef` store, so a second box for one position fails there.
    pub fn resolve_operand_operand(&self, arg: &Operand) -> Operand {
        if arg.is_constant() {
            return arg.clone();
        }
        if !arg.is_resop() && !arg.is_inputarg() {
            return self.get_box_replacement_operand(arg.to_opref());
        }
        let resolved = arg.get_box_replacement(false);
        // `resoperation.py AbstractValue.get_box_replacement` is the walk
        // alone. A self-resolved recorder InputArg of the peeled body still
        // takes the preamble-constant rejoin: `import_state` forwards the
        // phase 2 host, not the box the body operations hold.
        let resolved = if resolved.same_box(arg) {
            self.imported_inputarg_operand(arg).unwrap_or(resolved)
        } else {
            resolved
        };
        // An unregistered position has no positional answer: the old
        // resolver fabricated a fresh box for it on every call.
        #[cfg(debug_assertions)]
        if self.resolve_to_operand(arg.to_opref()).is_some() {
            let old = self.resolve_operand_operand_positional(arg);
            debug_assert!(
                resolved.same_box(&old),
                "resolve_operand_operand {:?}: new {:?} old {:?}",
                arg.to_opref(),
                resolved.to_opref(),
                old.to_opref()
            );
        }
        resolved
    }

    #[cfg(debug_assertions)]
    fn resolve_operand_operand_positional(&self, arg: &Operand) -> Operand {
        // The canonical `heal_arg_to_canonical` would link to, read without
        // writing `_forwarded`, so the witness leaves no trace in debug runs.
        if arg.bound_op().is_some()
            && matches!(arg.get_forwarded(), majit_ir::forwarding::Forwarded::None)
            && let Some(canon) = self.get_box_replacement_operand_opt(arg.to_opref())
        {
            return canon;
        }

        // `Const.get_box_replacement()` is identity in resoperation.py: a
        // Const has no `_forwarded` slot and cannot have a different
        // canonical host in the positional stores. Returning the carried
        // object also preserves Const identity; the old OpRef round-trip
        // allocated a replacement `Rc<Cell<Value>>` on every pass visit.
        if arg.is_constant() {
            return arg.clone();
        }

        // info.py `getptrinfo` starts with `get_box_replacement(op)`.
        // Both result ops and input args carry that native `_forwarded`
        // chain; consult it before the positional store fallback.
        if arg.is_resop() || arg.is_inputarg() {
            let resolved = arg.get_box_replacement(false);
            // Self-resolved box-native: the canonical forwarding for this
            // position lives in the `OpRef` store (see `get_box_replacement_operand`).
            if resolved.same_box(arg) {
                if let Some(imported) = self.imported_inputarg_operand(arg) {
                    return imported;
                }
                self.get_box_replacement_operand(arg.to_opref())
            } else {
                resolved
            }
        } else {
            self.get_box_replacement_operand(arg.to_opref())
        }
    }

    /// `Option`-returning sibling of
    /// [`resolve_operand_operand`](Self::resolve_operand_operand): `None`
    /// only for the NONE sentinel or a position-only operand whose position
    /// resolves to nothing, so callers can supply their own fallback.
    pub fn resolve_operand_operand_opt(&self, arg: &Operand) -> Option<Operand> {
        if arg.is_constant() || arg.is_resop() || arg.is_inputarg() {
            return Some(self.resolve_operand_operand(arg));
        }
        self.get_box_replacement_operand_opt(arg.to_opref())
    }

    /// Link an unforwarded ResOp `arg` to the producer registered at its
    /// position (`arg.set_forwarded(canonical)`) and return that producer's
    /// terminal; `None` when `arg` is unbound, already forwarded, or its
    /// position has no producer.
    ///
    /// The one caller is `export_state`'s short-preamble conversion: a
    /// replay op's args are the dependencies' replay handles, which share a
    /// position with the body producer and carry no forwarding of their own.
    ///
    /// The cycle guard compares the bound `Op`s: a store wrapper around the
    /// same `Rc<Op>` as `arg` is returned without linking, since linking would
    /// be a one-node self-cycle. The canonical is a terminal, so no chain
    /// cycle forms once a distinct op is linked.
    pub(crate) fn heal_arg_to_canonical(&self, arg: &Operand) -> Option<Operand> {
        arg.bound_op()?;
        if !matches!(arg.get_forwarded(), majit_ir::forwarding::Forwarded::None) {
            return None;
        }
        let canon = self.get_box_replacement_operand_opt(arg.to_opref())?;
        if arg.same_box(&canon) {
            return Some(canon);
        }
        // Skip when `canon` wraps the same bound `Op` as `arg` under a distinct
        // host — linking would be a one-node self-cycle.
        if let (Some(ao), Some(co)) = (arg.bound_op(), canon.bound_op())
            && OpRc::ptr_eq(&ao, &co)
        {
            return Some(canon);
        }
        if let Some(value) = canon.const_value() {
            arg.set_forwarded_const(majit_ir::Const::from_value(value));
        } else if let Some(canon_op) = canon.bound_op() {
            arg.set_forwarded_op(&canon_op);
        } else if let Some(canon_ia) = canon.bound_inputarg() {
            arg.set_forwarded_inputarg(&canon_ia);
        }
        // A constant link mints its own Const box; return that carried object,
        // just as the caller's subsequent native walk did before this reuse.
        Some(arg.get_box_replacement(false))
    }

    /// `not_const=True` operand resolver (resoperation.py:64-65): walk the
    /// `_forwarded` chain on the `Operand` carrier but stop before stepping
    /// into a `Const` target, so a guard fail_arg keeps the runtime identity
    /// while resume numbering carries constants as TAGCONST. Returns `None`
    /// for a const / NONE / unresolved operand. Lets a fail-arg consumer
    /// resolve `op.arg(i)` directly off the `Operand`.
    pub fn get_box_replacement_not_const_operand(&self, op: &Operand) -> Option<Operand> {
        if op.is_constant() || op.is_none() {
            return None;
        }
        if op.bound_op().is_some() {
            let resolved = op.get_box_replacement(true);
            #[cfg(debug_assertions)]
            {
                if let Some(start) = self.resolve_to_operand(op.to_opref()) {
                    let legacy = start.get_box_replacement(true);
                    debug_assert!(
                        resolved.same_box(&legacy) || resolved.to_opref() == legacy.to_opref(),
                        "get_box_replacement_not_const_operand: native walk diverged \
                         from OpRef path for {:?}",
                        op.to_opref()
                    );
                }
            }
            Some(resolved)
        } else {
            let start = self.resolve_to_operand(op.to_opref())?;
            Some(start.get_box_replacement(true))
        }
    }

    /// `OpRef` round-trip view of [`OptContext::get_box_replacement`] for
    /// the remaining flat-`OpRef` bridge callers (the
    /// `get_box_replacement(o).to_opref()` idiom — RPython callers hold the
    /// box and skip this step; the bridge retires with the operand union,
    /// #9). Identical to that idiom observationally — an unresolvable root
    /// returns `opref` unchanged (`from_opref(o).to_opref() == o`) — but
    /// without fabricating the intermediate position-only box.
    pub(crate) fn get_replacement_opref(&self, opref: OpRef) -> OpRef {
        // `resoperation.py get_box_replacement(Const)` returns that same
        // Const immediately: Const has no `_forwarded` slot.  The flat OpRef
        // bridge already carries its complete value, so do not manufacture an
        // `Rc<Cell<Value>>` object merely to convert it straight back to the
        // identical OpRef.
        if opref.is_constant() || opref.is_none() {
            return opref;
        }
        match self.get_box_replacement_operand_opt(opref) {
            Some(o) => {
                let resolved = o.to_opref();
                // A bound ResOp whose `pos` was never stamped has
                // `to_opref() == None`. Returning that sentinel loses the
                // box: `export_state` can no longer resolve it.
                // `get_box_replacement` returns the box, not None.
                if resolved.is_none() && !o.is_none() {
                    opref
                } else {
                    resolved
                }
            }
            None => opref,
        }
    }

    /// `Option`-exposing `Operand` resolver (`resoperation.py:57-68
    /// get_box_replacement(self, op)`): walk the `_forwarded` chain rooted at
    /// `opref` and return the terminal as an `Operand`, or `None` when the root
    /// does not resolve to a producer / inputarg / const. The `None` lets
    /// callers that must distinguish "no bound box" — a sentinel, or a test /
    /// retrace baseline with no upstream binding — branch on it rather than act
    /// on a position-only placeholder. Native form of
    /// `get_box_replacement_operand`'s body without the position-only panic
    /// fallback.
    pub fn get_box_replacement_operand_opt(&self, opref: OpRef) -> Option<Operand> {
        self.resolve_to_operand(opref)
            .map(|start| start.get_box_replacement(false))
    }

    /// `optimizer.py getptrinfo + info.is_virtual()` operand-path
    /// helper. Returns whether the box at `opref` (after chain walk)
    /// carries a `PtrInfo` whose `is_virtual()` is true. Reads via
    /// `Operand::ptr_info()` on the chain-walked terminal box; an
    /// unresolvable opref (synthetic test paths) returns `false`.
    /// `optimizer.py is_virtual(op)`:
    /// ```python
    /// def is_virtual(self, op):
    ///     opinfo = getptrinfo(op)
    ///     return opinfo is not None and opinfo.is_virtual()
    /// ```
    /// operand-direct read — chain walks via
    /// `Operand::get_box_replacement` then queries `ptr_info().is_virtual()`.
    pub fn is_virtual(&self, op: &Operand) -> bool {
        op.with_box_replacement(|b| b.ptr_info().is_some_and(|p| p.is_virtual()))
    }

    /// `info.py PtrInfo.is_nonnull` (base False) + subclass
    /// overrides — true when the box at `op` carries a non-null
    /// `PtrInfo` in its `_forwarded` Info slot. Chain walks via
    /// `Operand::get_box_replacement` then reads `ptr_info()`.
    pub fn is_nonnull(&self, op: &Operand) -> bool {
        op.with_box_replacement(|b| b.ptr_info().is_some_and(|p| p.is_nonnull()))
    }

    /// `optimizer.py getintbound(op)` read variant — returns an
    /// owned `IntBound` snapshot from the chain terminal's `_forwarded`
    /// slot, plus the ConstInt arm:
    ///
    /// ```python
    /// op = get_box_replacement(op)
    /// if isinstance(op, ConstInt):
    ///     return IntBound.from_constant(op.getint())
    /// fw = op.get_forwarded()
    /// if isinstance(fw, IntBound): return fw
    /// return None     # upstream returns IntBound.unbounded(); the
    ///                 # peek variant signals "no specific bound" instead.
    /// ```
    ///
    /// The full lazy-install path (missing-info → `IntBound.unbounded()`)
    /// lives in [`Self::getintbound_handle`]; this snapshot is the side-
    /// effect-free reader used by gates and read-only intersect comparisons.
    pub fn peek_intbound_box(
        &self,
        op: &Operand,
    ) -> Option<crate::optimizeopt::intutils::IntBound> {
        op.with_box_replacement(|resolved| {
            if let Some(Value::Int(v)) = resolved.const_value() {
                return Some(crate::optimizeopt::intutils::IntBound::from_constant(v));
            }
            resolved.int_bound().map(|ib| ib.clone())
        })
    }

    /// `info.py op.get_forwarded()` + `isinstance(fw, PtrInfo)` —
    /// snapshot read of the chain terminal's `_forwarded` PtrInfo.
    /// Clones the inner `PtrInfo` out of its `Rc<RefCell<>>` cell, so
    /// the result is independent of subsequent mutations.  For RPython
    /// object identity (`same_info`, in-place mutation propagation),
    /// use [`Operand::ptr_info_handle`], which returns the live `Rc`.
    pub fn peek_ptr_info(&self, op: &Operand) -> Option<crate::optimizeopt::info::PtrInfo> {
        op.get_box_replacement(false).ptr_info().map(|p| p.clone())
    }

    /// info.py: getptrinfo(op) — mutable variant. Walks the chain on `op`
    /// and runs the closure against the terminal operand's `_forwarded`
    /// PtrInfo via `ptr_info_mut()`. The forwarding slot is the authoritative
    /// storage; no separate mirror step is needed.
    ///
    /// Closure semantics: returns `Some(f(info))` when a `PtrInfo` exists
    /// at the terminal box, `None` otherwise (no closure invocation).
    pub fn with_ptr_info_mut<R>(
        &self,
        op: &Operand,
        f: impl FnOnce(&mut PtrInfo) -> R,
    ) -> Option<R> {
        let resolved = op.get_box_replacement(false);
        let mut pi = resolved.ptr_info_mut()?;
        let result = f(&mut pi);
        Some(result)
    }

    /// Closure-style wrapper around [`Self::ensure_ptr_info_arg0`].
    ///
    /// Closure mutations through `EnsuredPtrInfo::as_mut()` land on the
    /// operand's `RefCell<Forwarded>` directly — single-slot RPython parity
    /// with `optimizer.py ensure_ptr_info_arg0`'s mutate-in-place
    /// behavior.
    pub fn with_ensured_ptr_info_arg0<R>(
        &mut self,
        op: &Op,
        f: impl FnOnce(crate::optimizeopt::info::EnsuredPtrInfo) -> R,
    ) -> R {
        f(self.ensure_ptr_info_arg0(op))
    }

    /// `info.py PtrInfo.get_last_guard_pos` operand-direct reader.
    /// Walks chain to terminal and reads its `_forwarded` PtrInfo slot.
    pub fn last_guard_pos(&self, op: &Operand) -> Option<usize> {
        if self.new_operations_drained {
            return None;
        }
        op.get_box_replacement(false)
            .ptr_info()
            .and_then(|p| p.get_last_guard_pos())
    }

    /// `info.py getptrinfo(op) is not None` parity — true when
    /// the box carries any `PtrInfo` in its chain-terminal `_forwarded`
    /// Info slot. Walks via `Operand::get_box_replacement(false)` then
    /// queries `ptr_info().is_some()`.
    pub fn has_ptr_info(&self, op: &Operand) -> bool {
        // Mirror `getptrinfo(op).is_some()` so the gate behaves
        // identically. info.py dispatches by `op.type`: only
        // Int and Ref boxes can carry PtrInfo (raw-ptr Int via
        // `getrawptrinfo`, regular Ref via `getptrinfo`). Float and
        // Void return None / are rejected upstream — short-circuit
        // here so callers of `has_ptr_info` can pass any typed operand
        // without first guarding on the type.
        match op.type_() {
            majit_ir::Type::Int | majit_ir::Type::Ref => self.getptrinfo_handle(op).is_some(),
            majit_ir::Type::Float | majit_ir::Type::Void => false,
        }
    }

    /// TODO: RPython's virtualizable handling lives
    /// tracing-side (`pyjitpl.py _nonstandard_virtualizable`),
    /// not in optimizeopt — there is no direct `is_virtualizable` helper
    /// in `optimizer.py`. The pyre dedicated `PtrInfo::Virtualizable`
    /// variant + this helper exist because pyre routes virtualizable
    /// field tracking through the optimizer's `_forwarded` PtrInfo slot.
    /// Returns true when the chain-terminal carries `PtrInfo::Virtualizable`.
    pub fn is_virtualizable(&self, op: &Operand) -> bool {
        use crate::optimizeopt::info::PtrInfo;
        op.get_box_replacement(false)
            .ptr_info()
            .is_some_and(|p| matches!(*p, PtrInfo::Virtualizable(_)))
    }

    /// resoperation.py: op.get_forwarded() is not None — check if OpRef
    /// has any forwarding entry (Op, Info, IntBound, Const).
    ///
    /// `Const.get_forwarded()` returns `None` in RPython
    /// (`resoperation.py`); short-circuit on the const-namespace
    /// `OpRef` so the caller doesn't index a raw-keyed store with a
    /// CONST_BIT `raw()` — which would either miss (large-index) or
    /// alias an unrelated slot.
    pub fn has_forwarding(&self, op: &Operand) -> bool {
        // `resoperation.py Const.get_forwarded()` returns None;
        // Const boxes carry no `_forwarded` slot upstream.
        if op.is_constant() {
            return false;
        }
        // `resoperation.py _forwarded = None` — slot is None until
        // `set_forwarded` writes. `op.get_forwarded() is not None`.
        !matches!(op.get_forwarded(), majit_ir::forwarding::Forwarded::None)
    }

    /// True only when opref has a non-const forwarding redirect.
    ///
    /// `make_equal_to(_, non_const)` writes either `Forwarded::Op(_)` or
    /// `Forwarded::InputArg(_)`; the const branches go through
    /// `Forwarded::Const`. Splitting on the variant excludes the
    /// const-target shape so this returns true only for the AbstractValue
    /// redirect case used by `import_state`.
    ///
    /// `Const.get_forwarded()` returns `None` upstream
    /// (`resoperation.py`); short-circuit on the const-namespace
    /// `OpRef` so the caller doesn't index a raw-keyed store with a
    /// CONST_BIT `raw()`.
    pub fn has_op_forwarding(&self, op: &Operand) -> bool {
        if op.is_constant() {
            return false;
        }
        matches!(
            &op.get_forwarded(),
            majit_ir::forwarding::Forwarded::Op(_) | majit_ir::forwarding::Forwarded::InputArg(_)
        )
    }

    /// Bulk-seed entry for the recorder / backend constant import. NOT a
    /// substitute for the RPython `make_constant(box, constbox)`
    /// (`optimizer.py`); production optimizer-time const promotions
    /// must go through `OptContext::make_constant_box`, which overwrites any
    /// existing forwarding per upstream.
    ///
    /// `box_` is the canonical `_forwarded` host — the caller resolves it
    /// via `materialize_operand_at` (for a Const-namespace OpRef that is a
    /// fresh `new_const`; for a body OpRef the producing `Op` / `InputArg`
    /// host). A Const box carries its value intrinsically
    /// (history.py:220/261/307), so the const arm only asserts the box and
    /// value types agree. A non-Const box is forwarded to a fresh Const
    /// target, but only when the slot is `Forwarded::None`, preserving any
    /// PtrInfo / IntBound / Box(Const) forwarding installed by an earlier
    /// pass (PyPy's `make_constant` short-circuits on `box.is_constant()`
    /// before reaching `set_forwarded`; this is the recorder/bulk-seed
    /// entry where the forwarding slot is authoritative when present). The
    /// recorder calls this before optimization passes run, so the
    /// no-clobber rule never collides with a real `make_constant` caller.
    ///
    /// RPython parity: `ConstInt`, `ConstPtr`, `ConstFloat` are distinct
    /// Box subclasses (history.py:220/261/307); a typed seed over a slot
    /// already holding a different-typed value is a bug (typical source:
    /// `Value::Ref(0)` reseeded where `Value::Int(0)` lives, flipping
    /// `opref_type` Int→Ref). Assert the invariant rather than overwrite.
    pub fn seed_constant(&mut self, box_: &Operand, value: Value) {
        if box_.is_constant() {
            debug_assert!(
                box_.type_() == value.get_type(),
                "seed_constant: Const box type {:?} mismatches value {:?}",
                box_.type_(),
                value,
            );
        } else {
            // optimizer.py `box.set_forwarded(constbox)`, gated on
            // `Forwarded::None` so an earlier pass's PtrInfo / IntBound /
            // Box(Const) forwarding is never clobbered.
            if matches!(box_.get_forwarded(), majit_ir::forwarding::Forwarded::None) {
                box_.set_forwarded_const(majit_ir::Const::from_value(value));
            }
        }
    }

    /// Read-only variant of `getintbound` — returns the IntBound stored on
    /// `box._forwarded` without materializing an unbounded one on first
    /// access. Returns `None` for boxes that have no IntBound forwarding.
    /// Used by exporters that take `&OptContext` and cannot mutate.
    #[allow(dead_code)]
    pub(crate) fn peek_intbound(
        &self,
        opref: OpRef,
    ) -> Option<crate::optimizeopt::intutils::IntBound> {
        // optimizer.py getintbound: assert op.type == 'i'
        // None is allowed for test fixtures that don't seed value_types.
        assert!(
            matches!(self.opref_type(opref), Some(majit_ir::Type::Int) | None),
            "peek_intbound: expected 'i'-typed OpRef, got {:?}",
            self.opref_type(opref)
        );
        let replaced = self.get_replacement_opref(opref);
        if let Some(Value::Int(v)) = self
            .get_box_replacement_operand_opt(replaced)
            .and_then(|cb| cb.const_value())
        {
            return Some(crate::optimizeopt::intutils::IntBound::from_constant(v));
        }
        // optimizer.py:107 second `assert op.type == 'i'` — Box.type is
        // immutable in RPython, so the replaced op must still be int-typed.
        assert!(
            matches!(self.opref_type(replaced), Some(majit_ir::Type::Int) | None),
            "peek_intbound: replaced OpRef must be int-typed, got {:?}",
            self.opref_type(replaced)
        );
        if replaced.is_constant() {
            return None;
        }
        // Forwarding-slot-authoritative reader. IntBound writers populate the
        // operand via `materialize_operand_at`.
        let b = self.get_box_replacement_operand_opt(replaced)?;
        b.int_bound().map(|ib| ib.clone())
    }

    /// `optimizer.py getintbound(op)` orthodox identity variant.
    ///
    /// Walks `op.get_box_replacement(false)` and returns:
    ///   - `IntBoundHandle::Const(from_constant(v))` for a `ConstInt`
    ///     terminal (line 102-103).
    ///   - `IntBoundHandle::Const(unbounded)` when the terminal carries
    ///     a non-IntBound `_forwarded` slot (raw-pointer Int with
    ///     PtrInfo etc.; `optimizer.py` non-IntBound `fw` branch).
    ///   - `IntBoundHandle::Live(rc)` for an existing
    ///     `OpInfo::IntBound(rc)` slot, **or** a freshly installed
    ///     `unbounded` cell when the slot was `Forwarded::None` —
    ///     mirroring RPython's lazy `op.set_forwarded(IntBound())`
    ///     side-effect at line 111.
    pub fn getintbound_handle(&self, op: &Operand) -> IntBoundHandle {
        use crate::optimizeopt::info::OpInfo;
        // optimizer.py:100 `assert op.type == 'i'`. Void admitted as the
        // pyre placeholder-box tolerance shared with `setintbound`.
        assert!(
            matches!(op.type_(), majit_ir::Type::Int | majit_ir::Type::Void),
            "getintbound_handle: expected 'i'-typed operand, got {:?}",
            op.type_()
        );
        let resolved = op.get_box_replacement(false);
        // optimizer.py:107 `assert op.type == 'i'` lifted ahead of every
        // post-walker branch.  PyPy's structural-error catch fires when a
        // non-`ConstInt` terminal reaches the `fw is None` arm and the
        // assert sees a non-`'i'` `op.type` — pyre matches that position
        // up-front so a cross-type-forwarded Int box (Int → ConstPtr /
        // ConstFloat) panics here instead of being silently absorbed by
        // the unbounded branch below.  Mirrors `peek_intbound_handle`'s
        // assert shape.
        assert!(
            matches!(resolved.type_(), majit_ir::Type::Int | majit_ir::Type::Void),
            "getintbound_handle: chain terminal lost 'i' type, got {:?}",
            resolved.type_()
        );
        if let Some(Value::Int(v)) = resolved.const_value() {
            return IntBoundHandle::const_(crate::optimizeopt::intutils::IntBound::from_constant(
                v,
            ));
        }
        match &resolved.get_forwarded() {
            majit_ir::forwarding::Forwarded::Info(OpInfo::IntBound(rc)) => {
                return IntBoundHandle::live(rc.clone());
            }
            majit_ir::forwarding::Forwarded::None => {}
            _ => {
                return IntBoundHandle::const_(crate::optimizeopt::intutils::IntBound::unbounded());
            }
        }
        // optimizer.py:110-112 lazy install — the new cell is the live
        // identity that downstream `intersect`/`make_*` mutations
        // propagate through.
        let intbound = crate::optimizeopt::intutils::IntBound::unbounded();
        resolved.set_forwarded_info(OpInfo::int_bound(intbound));
        let rc = resolved
            .int_bound_handle()
            .expect("just installed OpInfo::IntBound");
        IntBoundHandle::live(rc)
    }

    /// optimizer.py: setintbound(op, bound) line-by-line port.
    ///
    /// ```python
    /// def setintbound(self, op, bound):
    ///     assert op.type == 'i'
    ///     op = get_box_replacement(op)
    ///     if op.is_constant():
    ///         return
    ///     cur = op.get_forwarded()
    ///     if cur is not None:
    ///         if isinstance(cur, IntBound):
    ///             cur.intersect(bound)
    ///     else:
    ///         op.set_forwarded(bound)
    /// ```
    pub fn setintbound(&self, op: &Operand, bound: &crate::optimizeopt::intutils::IntBound) {
        use crate::optimizeopt::info::OpInfo;
        // optimizer.py:116: assert op.type == 'i' — structural assert,
        // matches RPython's release-build invariant. Type::Void boxes are
        // pyre-only phantom placeholders surfaced by `materialize_operand_at` when the
        // recorder has not yet typed the position; accept them as the pyre
        // equivalent of RPython's "the trace typing hasn't reached this
        // OpRef yet" tolerance.
        assert!(
            matches!(op.type_(), majit_ir::Type::Int | majit_ir::Type::Void),
            "setintbound: expected 'i'-typed operand, got {:?}",
            op.type_()
        );
        // optimizer.py: op = get_box_replacement(op)
        let op = op.get_box_replacement(false);
        // optimizer.py:118-119: if op.is_constant(): return
        if op.is_constant() {
            return;
        }
        // optimizer.py:120-122: cur = op.get_forwarded()
        //                       if cur is not None and isinstance(cur, IntBound):
        //                           cur.intersect(bound)
        if let Some(mut cur) = op.int_bound_mut() {
            let _ = cur.intersect(bound);
            return;
        }
        // optimizer.py:123-125: else (cur is None): op.set_forwarded(bound)
        // When cur is a non-None non-IntBound (e.g. RawBufferPtrInfo on a
        // raw-pointer Int), upstream's outer `if cur is not None` already
        // consumed control; the else branch only runs when cur is None.
        use majit_ir::forwarding::Forwarded as BoxFwd;
        if matches!(op.get_forwarded(), BoxFwd::None) {
            op.set_forwarded_info(OpInfo::int_bound(bound.clone()));
        }
    }

    /// In-place mutation helper for the IntBound stored on `box._forwarded`.
    ///
    /// RPython pattern equivalence: where RPython writes
    /// `self.getintbound(box).<method>(...)` and the method mutates the
    /// `IntBound` returned from `box.get_forwarded()` directly, the Rust
    /// borrow checker forces us to materialize the bound, mutate it, and
    /// store it back. This helper performs that read-modify-write atomically
    /// and threads through any return value from the closure (e.g. the
    /// `Result<bool, InvalidLoop>` flag from `intersect`/`make_*`).
    ///
    /// For Constant boxes the bound is "fixed" — RPython's `getintbound`
    /// returns `IntBound.from_constant(...)` and any `intersect` is a
    /// no-op (the constant value is already in range or InvalidLoop). This
    /// helper mirrors that by running the closure on a temporary that is
    /// discarded after — the constant cannot be widened.
    ///
    /// For non-IntBound forwarded info (RawBufferPtrInfo etc.), RPython's
    /// `getintbound` falls through to "return IntBound.unbounded()" without
    /// overwriting forwarding. We mirror by running the closure on a
    /// temporary unbounded that is discarded.
    pub fn with_intbound_mut<F, R>(&self, op: &Operand, f: F) -> R
    where
        F: FnOnce(&mut crate::optimizeopt::intutils::IntBound) -> R,
    {
        use crate::optimizeopt::info::OpInfo;
        use majit_ir::forwarding::Forwarded;
        // optimizer.py getintbound: assert op.type == 'i'. Active in release
        // builds per upstream. Void-typed phantoms (`materialize_operand_at` lazy-alloc)
        // are accepted because they are placeholder boxes pending recorder
        // typing — their chain walk may still terminate at an int-typed
        // Const/InputArg.
        assert!(
            matches!(op.type_(), majit_ir::Type::Int | majit_ir::Type::Void),
            "with_intbound_mut: expected 'i'-typed operand, got {:?}",
            op.type_()
        );
        // optimizer.py: op = get_box_replacement(op)
        let resolved = op.get_box_replacement(false);
        // optimizer.py:102-103: ConstInt → IntBound.from_constant(...).
        // RPython's getintbound returns a fresh bound; intersect on it is
        // a no-op (already at the constant value), and the bound is
        // discarded after the closure — the Const box stays canonical.
        if let Some(Value::Int(v)) = resolved.const_value() {
            let mut tmp = crate::optimizeopt::intutils::IntBound::from_constant(v);
            return f(&mut tmp);
        }
        if resolved.is_constant() {
            // Non-Int constant (Float / Ref) — getintbound's "assert op.type
            // == 'i'" would fail upstream; majit returns unbounded for
            // raw-pointer Int constants and Type::Void phantoms, both of
            // which surface here when the typed-namespace OpRef is forced
            // through `with_intbound_mut`.
            let mut tmp = crate::optimizeopt::intutils::IntBound::unbounded();
            return f(&mut tmp);
        }
        // optimizer.py:104-109: branch on forwarded slot.
        let needs_init = matches!(resolved.get_forwarded(), Forwarded::None);
        if needs_init {
            // optimizer.py:110-112 first-access: materialize unbounded,
            // mutate via closure, install on `_forwarded`.
            let mut new_bound = crate::optimizeopt::intutils::IntBound::unbounded();
            let result = f(&mut new_bound);
            resolved.set_forwarded_info(OpInfo::int_bound(new_bound));
            return result;
        }
        if let Some(mut bound) = resolved.int_bound_mut() {
            // optimizer.py:106-107: existing IntBound — mutate in place.
            return f(&mut bound);
        }
        // optimizer.py:108-109 rare case: forwarded is AbstractInfo other
        // than IntBound (RawBufferPtrInfo, FloatConst, etc.) — return a
        // temporary unbounded that gets discarded after the closure.
        let mut tmp = crate::optimizeopt::intutils::IntBound::unbounded();
        f(&mut tmp)
    }

    /// `Operand` form of [`OptContext::make_constant_box`]. optimizer.py make_constant
    /// `make_constant(box, constbox)` does `box = get_box_replacement(box)` then
    /// forwards the constant; this takes that first resolve operand-native via
    /// `resolve_operand_operand_opt` instead of collapsing the operand to an `OpRef`.
    pub fn make_constant_arg(&mut self, arg: &Operand, value: Value) {
        let b = self.resolve_operand_operand_opt(arg).or_else(|| {
            let opref = arg.to_opref();
            (!opref.is_none() && !opref.is_constant()).then(|| self.materialize_operand_at(opref))
        });
        if let Some(b) = b {
            // `resolve_operand_operand_opt` / `materialize_operand_at` yield a
            // bound-or-const operand, so the lowering is panic-free.
            self.make_constant_box(&b, value);
        }
    }

    /// optimizer.py make_constant(box, constbox)
    pub fn make_constant_box(&mut self, op: &Operand, value: Value) {
        if Self::operand_is_stack_resident_inputarg(op)
            || (op.is_inputarg()
                && matches!(value, Value::Ref(g) if Self::ref_addr_is_stack_resident(g.0)))
        {
            return;
        }
        // optimizer.py: box = get_box_replacement(box)
        let op = op.get_box_replacement(false);
        // optimizer.py:418-429: IntBound safety check
        if let Value::Int(intval) = value
            && let Some(mut bound) = op.int_bound_mut()
        {
            if !bound.contains(intval) {
                drop(bound);
                self.signal_invalid_loop("constant int is outside the range allowed for that box");
                return;
            }
            let _ = bound.make_eq_const(intval);
        }
        // optimizer.py:430: if box.is_constant(): return
        if op.is_constant() || op.const_value().is_some() {
            return;
        }
        // optimizer.py:432-434: copy_fields_to_const for Ref
        if let Value::Ref(gcref) = value
            && let Some(pos) = op.position()
        {
            let opref = majit_ir::OpRef::ref_op(pos);
            self.copy_fields_to_const(opref, gcref);
        }
        // optimizer.py:432: box.set_forwarded(constbox). Terminate the
        // chain in an inline value-typed Const payload (history.py/
        // 268/314) — no separate constant box carrier and no pool index.
        // `get_box_replacement` rematerializes the const and `operand_to_opref`
        // recovers the inline-Const OpRef via `Operand::Const::to_opref`'s
        // value-derived branch.
        if matches!(value, Value::Void) {
            panic!("make_constant: Value::Void has no ConstVoid upstream (history.py:220/261/307)");
        }
        op.set_forwarded_const(value.to_const());
    }

    /// info.py (AbstractStructPtrInfo) + info.py (ArrayPtrInfo)
    /// `copy_fields_to_const(constinfo, optheap)`.
    ///
    /// ```text
    /// # AbstractStructPtrInfo
    /// def copy_fields_to_const(self, constinfo, optheap):
    ///     if self._fields is not None:
    ///         info = constinfo._get_info(self.descr, optheap)
    ///         assert isinstance(info, AbstractStructPtrInfo)
    ///         info._fields = self._fields[:]
    ///
    /// # ArrayPtrInfo
    /// def copy_fields_to_const(self, constinfo, optheap):
    ///     descr = self.descr
    ///     if self._items is not None:
    ///         info = constinfo._get_array_info(descr, optheap)
    ///         assert isinstance(info, ArrayPtrInfo)
    ///         info._items = self._items[:]
    /// ```
    ///
    /// majit folds both per-type entries into a single helper because the
    /// per-source dispatch happens via the PtrInfo enum match. The
    /// `_get_info`/`_get_array_info` half is `const_infos.entry(...)`
    /// (RPython: `optheap.const_infos[ref]`).
    fn copy_fields_to_const(&mut self, source: OpRef, gcref: majit_ir::GcRef) {
        use crate::optimizeopt::info::{
            ArrayPtrInfo, CachedFieldList, FieldEntry, PtrInfo, StructPtrInfo,
        };
        // `source` is always chain-walked by the caller (`make_constant`),
        // so peek's chain
        // walk is a no-op — owned PtrInfo clone here matches the prior
        // `Forwarded::Info(info)` immediate-slot read.
        let source_box = self.get_box_replacement_operand_opt(source);
        let Some(info) = source_box.as_ref().and_then(|b| self.peek_ptr_info(b)) else {
            return;
        };
        let key = gcref.as_usize();
        match info {
            // info.py AbstractStructPtrInfo.copy_fields_to_const →
            // constinfo._get_info(self.descr, optheap) → StructPtrInfo(descr).
            PtrInfo::Instance(v) if !v.fields.is_empty() => {
                let Some(descr) = v.descr.clone() else {
                    return;
                };
                let fields = v.fields.clone();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Struct(StructPtrInfo {
                        descr,
                        fields: CachedFieldList::new(),

                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Struct(s) = ci {
                    s.fields = fields;
                }
            }
            PtrInfo::Struct(v) if !v.fields.is_empty() => {
                let descr = v.descr.clone();
                let fields = v.fields.clone();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Struct(StructPtrInfo {
                        descr,
                        fields: CachedFieldList::new(),

                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Struct(s) = ci {
                    s.fields = fields;
                }
            }
            PtrInfo::Virtual(v) if !v.fields.is_empty() => {
                let descr = v.descr.clone();
                let fields: CachedFieldList = v
                    .fields
                    .iter()
                    .map(|(k, r)| (*k, FieldEntry::Value(r.clone())))
                    .collect();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Struct(StructPtrInfo {
                        descr,
                        fields: CachedFieldList::new(),

                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Struct(s) = ci {
                    s.fields = fields;
                }
            }
            PtrInfo::VirtualStruct(v) if !v.fields.is_empty() => {
                let descr = v.descr.clone();
                let fields: CachedFieldList = v
                    .fields
                    .iter()
                    .map(|(k, r)| (*k, FieldEntry::Value(r.clone())))
                    .collect();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Struct(StructPtrInfo {
                        descr,
                        fields: CachedFieldList::new(),

                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Struct(s) = ci {
                    s.fields = fields;
                }
            }
            // info.py ArrayPtrInfo.copy_fields_to_const →
            // constinfo._get_array_info(descr, optheap) → ArrayPtrInfo(descr).
            PtrInfo::Array(v) if !v.items.is_empty() => {
                let descr = v.descr.clone();
                let lenbound = v.lenbound.clone();
                let items = v.items.clone();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Array(ArrayPtrInfo {
                        descr,
                        lenbound,
                        items: Vec::new(),
                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Array(a) = ci {
                    a.items = items;
                }
            }
            PtrInfo::VirtualArray(v) if !v.items.is_empty() => {
                let descr = v.descr.clone();
                let len = v.items.len() as i64;
                let items: Vec<FieldEntry> = v
                    .items
                    .iter()
                    .map(|r| FieldEntry::Value(r.clone()))
                    .collect();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Array(ArrayPtrInfo {
                        descr,
                        lenbound: IntBound::from_constant(len),
                        items: Vec::new(),
                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Array(a) = ci {
                    a.items = items;
                }
            }
            _ => {}
        }
    }

    /// resume.py getconst parity for synthetic rd_numb encoding.
    /// Returns the (raw bits, type) of a constant operand, or None if it
    /// is not a constant. Type comes from the `Value` variant directly;
    /// raw-pointer Int constants live as inline `Const` operands with
    /// `Value::Ref` (Ref-typed) per the typed-pointer model, so
    /// `Value::Int` is always a real integer here.
    pub fn getconst(&self, op: &Operand) -> Option<(i64, majit_ir::Type)> {
        // Walk the chain and read the terminal's const_value (Const Box).
        let resolved = op.get_box_replacement(false);
        if let Some(val) = resolved.const_value() {
            let (raw, tp) = match val {
                Value::Int(v) => (v, majit_ir::Type::Int),
                Value::Float(f) => (f.to_bits() as i64, majit_ir::Type::Float),
                Value::Ref(r) => (r.0 as i64, majit_ir::Type::Ref),
                _ => return None,
            };
            return Some((raw, tp));
        }
        // info.py: ConstPtrInfo — GcRef constant stored in PtrInfo.
        if let Some(crate::optimizeopt::info::PtrInfo::Constant(gcref)) =
            resolved.ptr_info().as_deref()
        {
            return Some((gcref.0 as i64, majit_ir::Type::Ref));
        }
        None
    }

    /// Actual-Const reader: `box = box.get_box_replacement(); isinstance(box, Const)`.
    ///
    /// This intentionally does not use `get_constant_box`, because PyPy's
    /// `optimizer.get_constant_box` also synthesizes `ConstInt` from constant
    /// `IntBound`. Call this only for source sites that literally test
    /// `isinstance(..., Const)` / `box.is_constant()` / direct `ConstInt`.
    pub fn get_constant(&self, opref: OpRef) -> Option<Value> {
        match opref {
            OpRef::ConstInt(v) => return Some(Value::Int(v)),
            OpRef::ConstFloat(v) => return Some(Value::Float(v)),
            OpRef::ConstPtr(v) => {
                return Some(Value::Ref(majit_ir::const_ptr_table::resolve(v)));
            }
            // Non-constant OpRefs walk the forwarding chain below to find a
            // value forwarded onto them by `make_constant`.
            _ => {}
        }
        self.get_box_replacement_operand_opt(opref)
            .and_then(|b| b.const_value())
    }

    /// resoperation.py `InputArg*.getint/getref_base/getfloatstorage`
    /// — extract the concrete runtime value carried by an OpRef's OWN box.
    ///
    /// virtualstate.py `runtime_box.constbox()`, :494 `.getint()`, :579
    /// `.nonnull()`, :601/:608 `cpu.cls_of_box(runtime_box)` read the runtime
    /// box object itself: `getint`/`getref_base` (resoperation.py) return
    /// `_resint`/`_resref` — the box's own value slot, set when the box was
    /// created — and never walk `_forwarded`. This resolves the box at
    /// `opref`'s own position (`resolve_to_operand`, the canonical host WITHOUT
    /// the `get_box_replacement` chain walk) and reads its value directly; an
    /// optimizer forwarding (`make_equal_to` / `make_constant`) never takes
    /// precedence over the box's own observed value.
    ///
    /// `None` for an own slot that carries no value: pyre's value slots are
    /// `Option<Value>` (resoperation.rs / forwarding.rs) where RPython's are
    /// `_resint=0` / `_resref=NULL` defaults, so an unobserved box reads as
    /// `None` here. Callers (`runtime_nonnull`, IntBounded, `runtime_cls_of`,
    /// `get_runtime_field`) treat `None` as "no runtime guidance" and refuse
    /// the guard — the conservative direction. The runtime boxes threaded into
    /// generate_guards are the recorded JUMP args (unroll.py:105), whose
    /// producing ops carry their own observed values, so this reads the real
    /// runtime value without consulting `_forwarded`.
    pub fn runtime_value_of(&self, opref: OpRef) -> Option<Value> {
        let own = self.resolve_to_operand(opref)?;
        own.const_value().or_else(|| own.get_value())
    }

    /// `runtime_box.nonnull()` — resoperation.py `IntOp.nonnull`
    /// (`self._resint != 0`), :609 `FloatOp.nonnull`
    /// (`bool(extract_bits(self._resfloat))`), `RefOp.nonnull`
    /// (`bool(self.getref_base())`). Reads the runtime box's carried value
    /// (`runtime_value_of`) and applies the per-type rule. Returns `false`
    /// when no runtime value is plumbed: a box with no observed value must
    /// not be claimed nonnull (virtualstate.py:579 gates GUARD_NONNULL on
    /// `runtime_box.nonnull()`, so a null/absent value refuses the guard).
    pub fn runtime_nonnull(&self, opref: OpRef) -> bool {
        match self.runtime_value_of(opref) {
            Some(Value::Int(i)) => i != 0,
            Some(Value::Float(f)) => f.to_bits() != 0,
            Some(Value::Ref(r)) => !r.is_null(),
            Some(Value::Void) | None => false,
        }
    }

    /// `cpu.bh_classof(runtime_box.getref_base())` — virtualstate.py
    /// KnownClass arms, model.py `cls_of_box`. Reads the runtime box's
    /// OWN ref value (`getref_base`, resoperation.py — the box's own
    /// `_resref` slot, never the `_forwarded` chain) via
    /// `runtime_value_of`, then returns `ptr2int(typeptr)`, the immortal
    /// vtable address as a plain integer. Typeptr is read with
    /// `cpu.bh_classof`; `cls_of_box` delegates to that hook, so a
    /// backend override still applies. No synthetic ConstPtr is minted.
    ///
    /// Unlike `cls_of_box(&Operand)` (which walks `get_box_replacement` to
    /// reach a Const terminal), this resolves through the no-forward
    /// `runtime_value_of`: virtualstate's KnownClass arms read the runtime
    /// box itself, with no optimizer-tracked / forwarded precedence. Returns
    /// `None` for non-Ref / null / unobserved values, so a KnownClass guard
    /// refuses rather than reading a forwarded class.
    pub fn runtime_cls_of(&self, opref: OpRef) -> Option<i64> {
        match self.runtime_value_of(opref)? {
            Value::Ref(gcref) if !gcref.is_null() => {
                let typeptr = self.cpu.bh_classof(gcref);
                if typeptr == 0 { None } else { Some(typeptr) }
            }
            _ => None,
        }
    }

    /// resoperation.py `AbstractResOpOrInputArg.same_box`: `self is other`
    /// — Python object identity, NOT the value-aware `Const.same_box`.
    ///
    /// Walks both operands through `get_box_replacement` (resoperation.py)
    /// then compares the resolved `OpRef`s.
    ///
    /// IDENTITY CAVEAT: `OpRef::Const*` carries the constant VALUE inline
    /// (history.py:227), not a pool-index slot, so two independently-minted
    /// `ConstInt(5)` resolve to the *same* `OpRef` and ARE `box_is`-equal.
    /// For constants this therefore matches PyPy's value-based
    /// `Const.same_box` (history.py), NOT PyPy's object-identity `is`
    /// (two distinct `ConstInt(5)` objects are `is`-False). For non-constant
    /// boxes (InputArg*/`*-Op` positions) the variant tag still encodes a
    /// unique position, so `box_is` remains a faithful 1:1 encoding of `is`.
    ///
    /// USAGE / HAZARD: use this where RPython writes `arg0 is arg1`; use
    /// `same_box` where RPython writes `arg0.same_box(arg1)`. Because
    /// constants collapse by value here, only call `box_is` at an `is`-site
    /// where treating two equal-valued constants as identical is correct (or
    /// conservatively safe). The current `is`-on-constant call sites are
    /// value-safe:
    ///   - rewrite.rs `_optimize_oois_ooisnot` `elif arg0 is arg1`
    ///     (rewrite.py:542): folding equal `ConstPtr`/`ConstInt` to "equal"
    ///     is the correct result.
    ///   - heap.rs `lookup_cached` `cached_index is indexbox` (heap.py):
    ///     a hit on an equal constant index is valid (the var-index cache is
    ///     write-invalidated, so no stale hit can survive).
    ///     A future `is`-site that must treat equal-valued *distinct* constants
    ///     as DISTINCT cannot use `box_is` as-is.
    pub fn box_is(&self, a: OpRef, b: OpRef) -> bool {
        self.get_replacement_opref(a) == self.get_replacement_opref(b)
    }

    /// resoperation.py `same_box` (non-Const: `self is other`) +
    /// history.py `Const.same_box` (value comparison via
    /// `same_constant`). Resolves both operands through
    /// `get_box_replacement_operand_opt` then delegates to `Operand::same_box`. Falls
    /// back to resolved-`OpRef` identity plus constant-value comparison
    /// when either box is absent: two references to the same unresolved
    /// variable are still the same box (`self is other`), and a
    /// producer-less position has no canonical operand to compare by `Rc`
    /// identity, so resolved-`OpRef` equality stands in for object identity.
    pub fn same_box(&self, query: OpRef, stored: OpRef) -> bool {
        match (
            self.get_box_replacement_operand_opt(query),
            self.get_box_replacement_operand_opt(stored),
        ) {
            (Some(ref a), Some(ref b)) => a.same_box(b),
            _ => {
                let query = self.get_replacement_opref(query);
                let stored = self.get_replacement_opref(stored);
                if query == stored {
                    return true;
                }
                match (
                    self.get_box_replacement_operand_opt(query)
                        .and_then(|cb| cb.const_value()),
                    self.get_box_replacement_operand_opt(stored)
                        .and_then(|cb| cb.const_value()),
                ) {
                    (Some(a), Some(b)) => a == b,
                    _ => false,
                }
            }
        }
    }

    /// vstring.py:237 `optstring.getintbound(box).is_constant()` pattern.
    /// Returns the constant value if known either from the constant pool
    /// or from IntBound analysis.
    pub fn get_constant_int_or_bound_box(&self, b: &Operand) -> Option<i64> {
        if let Some(Value::Int(i)) = self.get_constant_box(b) {
            return Some(i);
        }
        self.peek_intbound_box(b)
            .filter(|ib| ib.is_constant())
            .map(|ib| ib.get_constant_int())
    }

    /// history.py CONST_NULL = ConstPtr(ConstPtr.value).
    /// `CONST_NULL.same_constant(op)` parity (history.py `CONST_NULL =
    /// ConstPtr(ConstPtr.value)`). True iff `op` resolves to a Ref-typed
    /// null constant. Walks the chain and reads the terminal's
    /// `const_value()` directly — Const-namespace OpRefs whose
    /// forwarding chain terminates at a `Forwarded::Const`
    /// with `Value::Ref(GcRef(0))` are detected here.
    pub fn is_const_null(&self, op: &Operand) -> bool {
        matches!(
            op.get_box_replacement(false).const_value(),
            Some(Value::Ref(r)) if r.0 == 0
        )
    }

    /// Take the pending `_last_guard_op = None` raised by a side-effecting op
    /// that was emitted straight into `new_operations` while the Optimizer was
    /// mid-flight.  Clears the flag.
    pub(crate) fn take_guard_chain_broken(&mut self) -> bool {
        std::mem::take(&mut self.guard_chain_broken)
    }

    /// optimizer.py is_call_pure_pure_canraise — a CallPure op whose
    /// effectinfo says check_can_raise(ignore_memoryerror=True). These ops are
    /// formally side-effect-free (has_no_side_effect), but their potential to
    /// raise means they break guard resume-data sharing.  Both guard paths
    /// answer this from the same place so they cannot drift.
    fn is_call_pure_pure_canraise(op: &Op) -> bool {
        optimizer::Optimizer::is_call_pure_pure_canraise(op)
    }

    /// optimizer.py emit_guard_operation — decide whether to share
    /// resume data from the previous guard (_copy_resume_data_from) or build
    /// new resume data (store_final_boxes_in_guard).
    fn emit_guard_operation(&mut self, op: &Op) -> Option<Op> {
        let opnum = op.opcode;

        // optimizer.py:655-664: GUARD_(NO_)EXCEPTION following a guard that
        // is NOT GUARD_NOT_FORCED — give up sharing.  GUARD_NOT_FORCED_2
        // is excluded for the same reason as in the Optimizer path:
        // pyjitpl.py:3236 emits it at finish() only, so no exception
        // guard can follow.
        if (opnum == OpCode::GuardNoException || opnum == OpCode::GuardException)
            && let Some(idx) = self.last_guard_idx
            && self.new_operations[idx].opcode != OpCode::GuardNotForced
        {
            self.last_guard_idx = None;
        }

        // optimizer.py:665-670: GUARD_ALWAYS_FAILS must never share.
        if opnum == OpCode::GuardAlwaysFails {
            self.last_guard_idx = None;
        }

        // optimizer.py `emit_guard_operation`: `self._last_guard_op and guard_op.getdescr() is
        // None` — the whole test.  A recorded guard also has no descr until
        // `invent_fail_descr_for_op` mints one, so it shares too: the shared
        // guard resumes at the donor's position and the operations between
        // them re-execute.  `_emit_operation` only keeps the chain across
        // side-effect-free / guard / jit_debug / ovf operations, which is
        // what makes that sound.  A cloned `rd_resume_position` of its own is
        // then simply unused and must not gate the test.
        // compile.py:925-926: GUARD_NOT_FORCED* must never share —
        // invent_fail_descr_for_op asserts copied_from_descr is None.
        // `op.rd_resume_position() < 0`: see the same conjunct in
        // `Optimizer::emit_guard_operation` — a guard that owns a recorded
        // snapshot cannot inherit another's without replaying the recorded
        // bytecode range, and its committed side effects with it.
        let can_share = self.last_guard_idx.is_some()
            && !op.has_descr()
            && op.rd_resume_position() < 0
            && opnum != OpCode::GuardNotForced
            && opnum != OpCode::GuardNotForced2;

        if can_share {
            let idx = self.last_guard_idx.unwrap();
            // compile.py ResumeGuardCopiedDescr(prev) parity: stamp
            // a `ResumeGuardCopiedDescr` whose `prev` references the
            // donor's descr.  Readers go through
            // `FailDescr::rd_*()` which chases `prev` automatically
            // (compile.py `get_resumestorage(): return prev`).
            // GUARD_EXCEPTION / GUARD_NO_EXCEPTION mint the exc variant.
            //
            // optimizer.py:691 `assert isinstance(last_descr,
            // compile.ResumeGuardDescr)` — the donor must be a finalized
            // ResumeGuardDescr (or subclass).  RPython enforces this on
            // every sharing emit; pyre's standalone OptContext path
            // matched the production Optimizer in name only and used to
            // silently leave `op.descr = None` when the donor lacked a
            // descr.  Tighten to RPython parity.
            let donor_descr = self.new_operations[idx].getdescr().expect(
                "optimizer.py:691 assert isinstance(last_descr, \
                     ResumeGuardDescr): donor guard has no descr",
            );
            assert!(
                donor_descr.is_resume_guard(),
                "optimizer.py:691 assert isinstance(last_descr, \
                 ResumeGuardDescr): donor descr_index={} is not a \
                 ResumeGuardDescr subclass",
                donor_descr.index()
            );
            op.setdescr(match opnum {
                OpCode::GuardException | OpCode::GuardNoException => {
                    crate::compile::make_resume_guard_copied_exc_descr(donor_descr)
                }
                _ => crate::compile::make_resume_guard_copied_descr(donor_descr),
            });
            // optimizer.py _copy_resume_data_from: guard_op.setfailargs(last_guard_op.getfailargs())
            match self.new_operations[idx].guard_fail_args() {
                Some(fa) => op.setfailargs(fa.iter().cloned().collect()),
                None => op.clearfailargs(),
            }
            // The sharer resumes at the donor's snapshot, so it must carry the
            // donor's position too — its own recorded one is unused.  Mirrors
            // the Optimizer path's `_copy_resume_data_from`.
            op.set_rd_resume_position(self.new_operations[idx].rd_resume_position());
            // bridgeopt.py parity: fail_arg_types carry the types the
            // serializer used when writing the class-knowledge bitfield in
            // rd_numb. A
            // shared guard's rd_numb encodes the donor's livebox type
            // layout, so the sharer must inherit fail_arg_types too —
            // otherwise `deserialize_optimizer_knowledge` (`bridgeopt.rs`)
            // reconstructs a different Ref-set and reads past the buffer.
            match self.new_operations[idx].get_fail_arg_types() {
                Some(types) => op.set_fail_arg_types(types.to_vec()),
                None => op.clear_fail_arg_types(),
            }
            // optimizer.py: _maybe_replace_guard_value after copy.
            let replaced = if op.opcode == OpCode::GuardValue {
                self.maybe_replace_guard_value(op)
            } else {
                None
            };
            // Don't update last_guard_idx — copied guards don't become sources.
            if opnum == OpCode::GuardException {
                self.last_guard_idx = None;
            }
            replaced
        } else {
            // optimizer.py: store_final_boxes_in_guard.  This is
            // the standalone OptContext path (used by tests and the
            // direct ctx.emit_guard hook); it has no `pending_for_guard`
            // staging, so pass an empty Vec for the descr-side
            // set_rd_pendingfields write.
            self.store_final_boxes_in_guard(op, None, Vec::new());
            self.last_guard_idx = Some(self.new_operations.len());
            // optimizer.py: force_box on fail_args for unrolling.
            // Mirrors Optimizer.force_box contract: resolve replacement,
            // handle tracked preamble ops, force virtuals.
            if let Some(fa) = op.guard_fail_args() {
                let fargs: smallvec::SmallVec<[Operand; 8]> = fa.iter().cloned().collect();
                for farg in &fargs {
                    if !farg.is_none() {
                        // regalloc.py:1206: Const objects skip forcing.
                        // Constant OpRefs may collide with virtual positions;
                        // forcing would corrupt the virtual's PtrInfo.
                        if farg.get_box_replacement(false).const_value().is_none() {
                            self.force_box_inline(farg);
                        }
                    }
                }
            }
            // optimizer.py: _maybe_replace_guard_value after store.
            let replaced = if op.opcode == OpCode::GuardValue {
                self.maybe_replace_guard_value(op)
            } else {
                None
            };
            // optimizer.py: GUARD_EXCEPTION clears sharing.
            if opnum == OpCode::GuardException {
                self.last_guard_idx = None;
            }
            replaced
        }
    }

    /// optimizer.py _maybe_replace_guard_value — turn
    /// guard_value(bool) into guard_true/guard_false.
    fn maybe_replace_guard_value(&self, op: &Op) -> Option<Op> {
        let arg0 = op.arg(0);
        // optimizer.py:755: if op.getarg(0).type == 'i'
        let arg0_resolved = self.resolve_operand_operand(&arg0).to_opref();
        if self.opref_type(arg0_resolved) != Some(majit_ir::Type::Int) {
            return None;
        }
        // optimizer.py: b = self.getintbound(op.getarg(0))
        let Some(bound) = self
            .get_box_replacement_operand_opt(arg0_resolved)
            .and_then(|b| self.peek_intbound_box(&b))
        else {
            return None;
        };
        if !bound.is_bool() {
            return None;
        }
        let arg1 = op.arg(1);
        let constvalue = self
            .resolve_operand_operand_opt(&arg1)
            .and_then(|cb| cb.const_int())?;
        let new_opcode = match constvalue {
            0 => OpCode::GuardFalse,
            1 => OpCode::GuardTrue,
            _ => return None, // optimizer.py: strange code, just disable
        };
        // optimizer.py newop = self.replace_op_with(op, opnum,
        //                                  [op.getarg(0)], descr)
        // — produce a fresh op with new opcode and trimmed args, descr
        // unchanged.  copy_and_change preserves fail_args / rd_resume_position
        // / fail_arg_types for guard ops (resoperation.py).
        Some(op.copy_and_change(new_opcode, Some(&[arg0]), None))
    }

    /// optimizer.py force_box — inline equivalent for
    /// emit_guard_operation's fail_arg forcing (optimizer.py).
    /// Mirrors Optimizer.force_box contract: handle tracked preamble ops,
    /// then force virtuals to concrete. Body refs route through the preamble
    /// source directly, so the prior reverse-lookup 3rd key is no longer
    /// needed.
    pub(crate) fn force_box_inline(&mut self, op: &Operand) -> Operand {
        // optimizer.py: op = get_box_replacement(op)
        if op.is_constant() {
            return op.clone();
        }
        let resolved_box = self.resolve_operand_operand(op);
        // optimizer.py:351-359: a result that folded to an inline Const can
        // never be a `potential_extra_ops` key (the pool is keyed by the pure
        // op's result Box; the Const inlines at use sites instead of being
        // produced by a short box), so skip the short-preamble recording for
        // const-resolved results — otherwise the Const reaches `used_boxes`
        // and the carried label slot trips `OpRef::raw()` in unroll.rs.
        if resolved_box.is_constant() {
            return resolved_box;
        }
        let tracked = self
            .take_potential_extra_op(&resolved_box)
            .or_else(|| self.take_potential_extra_op(op));
        if let Some(preamble_op) = tracked {
            // shortpreamble.py `op = preamble_op.op.get_box_replacement()` —
            // walk the box's own `_forwarded` chain, the resolved Box itself
            // is handed to the builder.
            let resolved_for_pop = preamble_op.op.get_box_replacement(false);
            if let Some(builder) = self.active_short_preamble_producer_mut() {
                builder.add_preamble_op_from_pop(&preamble_op, resolved_for_pop);
            } else if let Some(builder) = self.imported_short_preamble_builder.as_mut() {
                builder.add_preamble_op_from_pop(&preamble_op, resolved_for_pop);
            }
        }
        // optimizer.py:361-362: if op.type == 'i' and info.is_constant():
        //     return ConstInt(info.get_constant_int())
        // Mirrors Optimizer::force_box — a forced operand with an already-constant
        // IntBound materializes as ConstInt; peek the bound without installing.
        if resolved_box.type_() == Type::Int
            && let Some(bound) = self.peek_intbound_box(&resolved_box)
            && bound.is_constant()
        {
            return Operand::const_(majit_ir::Const::Int(bound.get_constant_int()));
        }
        // optimizer.py force_box reads the live info. Only the recursive
        // materialization path needs an owned snapshot across &mut self.
        let virtual_info = resolved_box
            .ptr_info()
            .filter(|info| info.is_virtual())
            .map(|info| info.clone());
        if let Some(mut info) = virtual_info {
            let forced = info.force_box(&resolved_box, self);
            return self.resolve_operand_operand(&forced);
        }
        resolved_box
    }

    /// RPython optimizer.py store_final_boxes_in_guard inline.
    /// Called from emit() for every guard during optimization. Produces
    /// rd_numb via memo.number() using the CURRENT optimizer state
    /// (replacement chain, constants, virtual info).
    /// resume.py ResumeDataVirtualAdder.finish() parity:
    /// Generate rd_numb + rd_consts + rd_virtuals for a guard.
    /// Called from store_final_boxes_in_guard in optimizer.rs.
    /// Uses snapshot data (vable_boxes, frame_pcs, multi-frame) when available.
    pub fn finalize_guard_resume_data(
        &mut self,
        op: &Op,
        knowledge: Option<crate::resume::OptimizerKnowledgeForResume>,
        pending_setfields: Vec<majit_ir::GuardPendingFieldEntry>,
    ) {
        self.store_final_boxes_in_guard(op, knowledge, pending_setfields);
    }

    fn resume_snapshot_ready(&self, pos: i32) -> bool {
        snapshot_contains(&self.snapshot_boxes, pos) || self.byte_resume_contains(pos)
    }

    fn byte_resume_contains(&self, pos: i32) -> bool {
        self.byte_bridge_resume
            .as_ref()
            .is_some_and(|feed| feed.contains(pos))
    }

    pub(crate) fn store_final_boxes_in_guard(
        &mut self,
        op: &Op,
        knowledge: Option<crate::resume::OptimizerKnowledgeForResume>,
        mut pending_setfields: Vec<majit_ir::GuardPendingFieldEntry>,
    ) {
        use crate::resume::{ResumeDataLoopMemo, Snapshot};

        // optimizer.py store_final_boxes_in_guard parity:
        //   if op.getdescr() is not None:
        //       descr = op.getdescr()
        //       assert isinstance(descr, compile.ResumeGuardDescr)
        //   else:
        //       descr = compile.invent_fail_descr_for_op(op.getopnum(), self)
        //       op.setdescr(descr)
        //
        // RPython has exactly one emit path, so this function never
        // sees a `ResumeGuardCopiedDescr` (sibling, not subclass —
        // compile.py) nor an already-finalized `ResumeGuardDescr`
        // (resume.py `assert not storage.rd_numb` ensures finish()
        // runs at most once per descr).
        //
        // OptContext::emit gates its emit_guard_operation on
        // `!in_final_emission` so production runs through
        // `Optimizer::emit_guard_operation` once; the OptContext path is
        // limited to the standalone test entry.  Either way only fresh
        // descrs reach this function.
        assert!(
            op.getdescr().is_none_or(|d| d.is_resume_guard()),
            "optimizer.py `store_final_boxes_in_guard` expects \
             ResumeGuardDescr, got non-resume descr (kind={:?}, copied={})",
            op.getdescr().map(|d| d.index()),
            op.getdescr().is_some_and(|d| d.is_resume_guard_copied())
        );

        // resume.py `assert not storage.rd_numb` — finish() runs at
        // most once per ResumeGuardDescr.  RPython makes this invariant
        // load-bearing: a second call would clobber an already-numbered
        // livebox set and break bridge attachment.  Promoted from
        // debug_assert! so release builds catch double-finish too.
        assert!(
            op.resolved_rd_numb().is_none(),
            "resume.py:397 finish() invoked twice on the same ResumeGuardDescr"
        );

        // resume.py:396-397:
        //   resume_position = self.guard_op.rd_resume_position
        //   assert resume_position >= 0
        // RPython: every guard has a valid rd_resume_position set by either
        // capture_resumedata (tracer guards) or patchguardop copy
        // (unroll.py:336/409). No fallback — the position is always set
        // before store_final_boxes_in_guard runs.
        let resume_pos = op.rd_resume_position();
        let has_snapshot = self.resume_snapshot_ready(resume_pos);
        // resume.py: `assert resume_position >= 0` —
        // RPython asserts the position is set before calling
        // store_final_boxes_in_guard. Every guard from the production
        // pyre tracer captures its own snapshot via generate_guard /
        // capture_resumedata, and `gen_store_back_in_vable` inherits
        // the previous guard's snapshot id so its GUARD_NOT_FORCED_2
        // also has valid resume data. Hard-assert the invariant when
        // production snapshot data is wired through; the empty
        // `snapshot_boxes` case marks an isolated optimizer unit test
        // that constructs synthetic guards without going through the
        // pyre snapshot path, where the silent drop is acceptable.
        if !has_snapshot {
            // unroll.py:336/409 parity: when unroll creates a new guard from
            // a short preamble / virtual state import, it copies
            // rd_resume_position from patchguardop. If the new guard arrives
            // here without a snapshot, it must come from a patchguardop
            // context — inherit the patchguardop's resume_position.
            // resume.py:396-397: RPython asserts resume_position >= 0.
            let fallback_pos = self
                .patchguardop
                .as_ref()
                .map(|p| p.rd_resume_position())
                .filter(|&p| self.resume_snapshot_ready(p));
            if let Some(fb_pos) = fallback_pos {
                op.set_rd_resume_position(fb_pos);
                // resume.py _add_optimizer_sections: forward knowledge
                // to the patchguardop snapshot so heap/class/loopinvariant
                // sections are serialized into rd_numb. RPython's finish()
                // always serializes current optimizer knowledge regardless
                // of which snapshot provides the frame boxes.
                self.finalize_guard_resume_data(op, knowledge, pending_setfields);
                return;
            }
            // resume.py:396-397 parity:
            //   `resume_position = self.guard_op.rd_resume_position`
            //   `assert resume_position >= 0`
            // RPython asserts unconditionally here.  Tests that construct
            // guards directly must seed snapshot_boxes explicitly instead of
            // inventing a fail_args-derived fallback in this path.
            panic!(
                "store_final_boxes_in_guard: guard {:?} (pos={:?}, \
                 resume_pos={}) has no snapshot and no patchguardop \
                 ancestor — RPython resume.py:397 \
                 `assert resume_position >= 0` parity",
                op.opcode,
                op.pos().get(),
                op.rd_resume_position()
            );
        }

        // Byte-mode bridges have no prebuilt box list. Number from
        // `SnapshotIterator` before any `snapshot_boxes` borrow: the feed
        // has to move out of `self` for the `BoxEnv` borrow.
        let use_byte = self.byte_resume_contains(resume_pos)
            && !snapshot_contains(&self.snapshot_boxes, resume_pos);
        let byte_numbered = if use_byte {
            let min_vable = self.minimum_virtualizable_size;
            let feed = self.byte_bridge_resume.take().expect("byte resume");
            let numbered = {
                let env = OptBoxEnv { ctx: self };
                let mut memo = self.resumedata_memo.borrow_mut();
                feed.number_guard(&mut memo, &env, op.rd_resume_position(), min_vable)
            };
            self.byte_bridge_resume = Some(feed);
            Some(numbered)
        } else {
            None
        };

        // RPython parity: snapshot path handles ALL guards with snapshots,
        // including guards with rd_virtuals. The snapshot uses original boxes
        // and PtrInfo to correctly assign TAGVIRTUAL via _number_boxes.
        // _number_virtuals then builds rd_virtuals from PtrInfo.
        let snapshot_boxes: &[crate::resume::SnapshotBox] =
            snapshot_get(&self.snapshot_boxes, resume_pos)
                .map(|list| list.as_slice())
                .unwrap_or(&[]);
        let vable_oprefs = snapshot_get(&self.snapshot_vable_boxes, resume_pos)
            .map(|v| v.as_slice())
            .unwrap_or_default();
        let vref_oprefs = snapshot_get(&self.snapshot_vref_boxes, op.rd_resume_position())
            .map(|v| v.as_slice())
            .unwrap_or_default();
        let frame_pcs = self
            .snapshot_frame_pcs
            .get(op.rd_resume_position())
            .unwrap_or_default();

        // resume.py:201-202 get_box_replacement parity:
        // Pass ORIGINAL (unresolved) snapshot boxes. _number_boxes calls
        // env.get_box_replacement per-box, which resolves through the
        // replacement chain while preserving virtual identity.
        let frame_sizes = self.snapshot_frame_sizes.get(op.rd_resume_position());

        // Compare every snapshot cell's carried type with the type encoded by
        // its `OpRef` variant. `None` is an unclassified producer rather than
        // agreement. This complements the variant audit because `OpRef::ty()`
        // deliberately collapses variants within one type bank.
        #[cfg(feature = "jit-audits")]
        if !use_byte && majit_ir::opref_audit::enabled() {
            let mut agree = 0usize;
            let mut disagree = 0usize;
            let mut untyped = 0usize;
            let cells = snapshot_boxes
                .iter()
                .chain(vable_oprefs.iter())
                .chain(vref_oprefs.iter());
            for cell in cells {
                let encoded = cell.opref.ty();
                match cell.tp() {
                    None => untyped += 1,
                    Some(tp) if Some(tp) == encoded => agree += 1,
                    Some(tp) => {
                        disagree += 1;
                        eprintln!(
                            "[snapbox-tp] DISAGREE opref={:?} tp={:?} ty={:?}",
                            cell.opref, tp, encoded
                        );
                    }
                }
            }
            eprintln!(
                "[snapbox-tp] cells={} agree={} disagree={} untyped={}",
                agree + disagree + untyped,
                agree,
                disagree,
                untyped
            );
        }

        if !use_byte && crate::callee_rca_enabled() {
            let env = OptBoxEnv { ctx: self };
            let vable_debug: Vec<(OpRef, OpRef, bool, Type)> = vable_oprefs
                .iter()
                .map(|boxref| {
                    let boxref = boxref.opref();
                    let resolved = self.get_replacement_opref(boxref);
                    let is_virtual = self
                        .get_box_replacement_operand_opt(boxref)
                        .as_ref()
                        .is_some_and(|b| self.is_virtual(b));
                    let tp = majit_ir::BoxEnv::get_type(&env, boxref);
                    (boxref, resolved, is_virtual, tp)
                })
                .collect();
            eprintln!(
                "[callee-rca][store-final-vable] op={:?} pos={:?} resume_pos={} vable={:?}",
                op.opcode,
                op.pos().get(),
                op.rd_resume_position(),
                vable_debug,
            );
        }

        if !use_byte && majit_log_enabled() && op.opcode == OpCode::GuardNotForced2 {
            let env = OptBoxEnv { ctx: self };
            let snapshot_debug: Vec<(OpRef, OpRef, bool, Type)> = snapshot_boxes
                .iter()
                .map(|boxref| {
                    let boxref = boxref.opref();
                    let resolved = self.get_replacement_opref(boxref);
                    let is_virtual = self
                        .get_box_replacement_operand_opt(boxref)
                        .as_ref()
                        .is_some_and(|b| self.is_virtual(b));
                    let tp = majit_ir::BoxEnv::get_type(&env, boxref);
                    (boxref, resolved, is_virtual, tp)
                })
                .collect();
            let vable_debug: Vec<(OpRef, OpRef, bool, Type)> = vable_oprefs
                .iter()
                .map(|boxref| {
                    let boxref = boxref.opref();
                    let resolved = self.get_replacement_opref(boxref);
                    let is_virtual = self
                        .get_box_replacement_operand_opt(boxref)
                        .as_ref()
                        .is_some_and(|b| self.is_virtual(b));
                    let tp = majit_ir::BoxEnv::get_type(&env, boxref);
                    (boxref, resolved, is_virtual, tp)
                })
                .collect();
            eprintln!(
                "[jit][guard-resume] pos={:?} snapshot={:?} vable={:?}",
                op.pos().get(),
                snapshot_debug,
                vable_debug
            );
        }

        // resume.py: delegate to ResumeDataVirtualAdder.finish()
        // resume.py passes `minimum_virtualizable_size` here, which
        // arms the `resume.py` length check inside `number()`. This
        // call site used to hardcode `-1`, so the check — ported faithfully in
        // `resume.rs::number` — was disabled for every guard ever numbered,
        // and a 0-length vable section reached the backend unremarked. It then
        // surfaced only if the guard was actually deopted, as
        // `assert!(vable_size > 0)` in `resume.rs::consume_vable_info`.
        // optimizer.py:761-766 answers a failed numbering by abandoning the
        // whole compilation:
        //
        //     try:
        //         newboxes = modifier.finish(pendingfields)
        //         ...
        //     except resume.TagOverflow:
        //         raise compile.giveup()
        //
        // Returning without that signal leaves this guard with no resume data
        // and lets the trace compile regardless. Nothing reports it: the guard
        // is only consulted if it later fails, and the deopt then rebuilds
        // interpreter state out of an empty numbering, so the interpreter
        // resumes holding values that are not the ones it had. The witness is
        // the interpreted program's own answer, not any JIT statistic.
        //
        // `signal_invalid_loop` rather than a literal `giveup()`: pyre has no
        // `SwitchToBlackhole`, and this is the established way for a leaf that
        // cannot raise to abandon the trace (`protect_speculative_operation`
        // uses the same one). Both discard the compilation and leave the
        // interpreter to carry on from state the JIT never took over.
        let min_vable = self.minimum_virtualizable_size;
        let numbered = if let Some(numbered) = byte_numbered {
            numbered
        } else {
            let env = OptBoxEnv { ctx: self };
            let mut memo = self.resumedata_memo.borrow_mut();
            memo.number_from_parts(
                snapshot_boxes,
                frame_sizes,
                frame_pcs,
                vable_oprefs,
                vref_oprefs,
                &env,
                min_vable,
            )
        };
        let Ok(numb_state) = numbered else {
            self.signal_invalid_loop("resume numbering: TagOverflow");
            return;
        };
        let env = OptBoxEnv { ctx: self };
        let mut memo = self.resumedata_memo.borrow_mut();

        // resume.py, 520-558: pending_setfields are passed to finish()
        // which handles register_box, visitor_walk_recursive, and tagging.
        let Ok((rd_numb, rd_consts, rd_virtuals, liveboxes)) =
            memo.finish(numb_state, &env, &mut pending_setfields, knowledge.as_ref())
        else {
            self.signal_invalid_loop("resume numbering: TagOverflow");
            return;
        };
        drop(env);

        if crate::callee_rca_enabled() {
            let vable_items = if rd_numb.len() >= 3 {
                let len = rd_numb[2] as usize;
                rd_numb
                    .iter()
                    .skip(3)
                    .take(len)
                    .copied()
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            eprintln!(
                "[callee-rca][store-final-numbered] op={:?} pos={:?} resume_pos={} \
                 vable_items={:?} liveboxes={:?} livebox_types={:?} rd_consts={:?}",
                op.opcode,
                op.pos().get(),
                op.rd_resume_position(),
                vable_items,
                liveboxes,
                liveboxes.iter().map(|b| b.type_()).collect::<Vec<_>>(),
                rd_consts,
            );
        }

        if majit_log_enabled() && op.opcode == OpCode::GuardNotForced2 {
            eprintln!(
                "[jit][guard-resume] pos={:?} liveboxes={:?} rd_virtuals={} livebox_types={:?}",
                op.pos().get(),
                liveboxes,
                rd_virtuals.len(),
                liveboxes.iter().map(|b| b.type_()).collect::<Vec<_>>(),
            );
        }

        // resume.py reads `box.type` off each live box (`Operand::type_`).
        let new_types: Vec<majit_ir::Type> = liveboxes
            .iter()
            .map(|b| {
                if b.is_none() {
                    return majit_ir::Type::Ref;
                }
                b.type_()
            })
            .collect();

        // optimizer.py Optimizer.store_final_boxes_in_guard passes finish()'s
        // exact box objects to ResumeGuardDescr.store_final_boxes. Rebinding
        // their positions here loses the identity used by resume numbering.
        op.store_final_boxes(liveboxes.iter().cloned());
        memo.recycle_ordered_liveboxes(liveboxes);
        let final_operands = op.guard_fail_args().unwrap_or(&[]);
        let logical_rd_locs: majit_ir::RdLocs = final_operands
            .iter()
            .enumerate()
            .map(|(index, operand)| {
                if operand.is_none() {
                    0xFFFF
                } else {
                    u16::try_from(index).expect("resume failarg position must fit in rd_locs")
                }
            })
            .collect();
        if crate::callee_rca_enabled() {
            let final_oprefs: Vec<_> = final_operands
                .iter()
                .map(|operand| operand.to_opref())
                .collect();
            eprintln!(
                "[callee-rca][store-final-operands] op={:?} pos={:?} resume_pos={} final_oprefs={:?}",
                op.opcode,
                op.pos().get(),
                op.rd_resume_position(),
                final_oprefs,
            );
        }
        op.set_fail_arg_types(new_types.clone());
        // optimizer.py `store_final_boxes_in_guard` parity:
        //   if op.getdescr() is not None:
        //       descr = op.getdescr()
        //       assert isinstance(descr, compile.ResumeGuardDescr)
        //   else:
        //       descr = compile.invent_fail_descr_for_op(op.getopnum(), self)
        //       op.setdescr(descr)
        // RPython preserves the existing descr object (and its
        // `fail_index`, subtype, vector_info) and only mutates its
        // `fail_arg_types`. Pyre's `ResumeGuardDescr` /
        // ResumeAtPositionDescr / CompileLoopVersionDescr keep `types`
        // in `UnsafeCell<Vec<Type>>`, exposed via
        // `FailDescr::set_fail_arg_types`, so we mutate in place — the
        // load-bearing contract that subtype markers
        // (`is_resume_at_position()`, `loop_version()`) survive
        // `store_final_boxes_in_guard` (compile.py:1035-1043, mirrored
        // in `pyjitpl.rs`'s `compile_bridge` `is_resume_at_position()`).
        match op.getdescr() {
            Some(existing) => {
                if let Some(fd) = existing.as_fail_descr() {
                    fd.set_fail_arg_types(new_types);
                }
            }
            None => {
                // RPython compile.py `invent_fail_descr_for_op`
                // dispatches on opcode:
                //   GUARD_NOT_FORCED / GUARD_NOT_FORCED_2 → ResumeGuardForcedDescr
                //   GUARD_EXCEPTION  / GUARD_NO_EXCEPTION → ResumeGuardExcDescr
                //   else                                  → ResumeGuardDescr
                // The exception-flow / async-forcing special cases at
                // `pyjitpl.rs` opcode-check sites (e.g. the
                // GUARD_EXCEPTION → `is_exception_guard` and
                // GUARD_NOT_FORCED chains) can migrate to descr-keyed
                // dispatch via `is_guard_exc()` / `is_guard_forced()`
                // without reshaping this match arm.
                use majit_ir::OpCode;
                op.setdescr(match op.opcode {
                    OpCode::GuardNotForced | OpCode::GuardNotForced2 => {
                        crate::compile::make_resume_guard_forced_descr_typed(new_types)
                    }
                    OpCode::GuardException | OpCode::GuardNoException => {
                        crate::compile::make_resume_guard_exc_descr_typed(new_types)
                    }
                    _ => crate::compile::make_resume_guard_descr_typed(new_types),
                });
            }
        }
        // compile.py ResumeGuardDescr `_attrs_` parity: write the
        // post-numbering resume payload onto the descr that
        // store_final_boxes_in_guard just minted (or onto the existing
        // ResumeGuardDescr / ResumeAtPositionDescr / ResumeGuardForcedDescr
        // / ResumeGuardExcDescr that capture_resumedata stamped earlier).
        // The descr is the single source of truth — readers go through
        // FailDescr::rd_*().
        let descr_rd_virtuals = if rd_virtuals.is_empty() {
            None
        } else {
            Some(rd_virtuals)
        };
        let descr_pending = if pending_setfields.is_empty() {
            None
        } else {
            Some(pending_setfields)
        };
        let __descr_arc = op.getdescr();
        if let Some(fd) = __descr_arc.as_ref().and_then(|d| d.as_fail_descr()) {
            // pyjitpl.py `initialize_state_from_guard_failure` receives
            // `inputargs_and_holes` and returns only its non-None boxes.
            // Keep that distinction on the ResumeGuardDescr before handing
            // it to a backend: dynasm replaces these identity locations with
            // its compact frame locations, while cranelift and wasm keep
            // failargs in their logical slots and therefore need this
            // identity-with-holes layout as-is.  Deriving the mask later from
            // backend-populated rd_locs made bridgeopt deserialize dead Ref
            // holes on the latter backends, shifting the known-class
            // bitfield into the heap-knowledge stream.
            fd.set_rd_locs(logical_rd_locs);
            fd.set_rd_numb_arc(Some(rd_numb));
            fd.set_rd_consts_arc(Some(rd_consts));
            fd.set_rd_virtuals(descr_rd_virtuals);
            fd.set_rd_pendingfields(descr_pending);
            // The back-edge poll of the eval-breaker word reaches this
            // function like any other guard, and this is the last point where
            // its condition chain and its descr are both in hand — the
            // backends see only the emitted guard.  Stamping here also covers
            // every re-emission (unroll's preamble and body copies, and the
            // poll a bridge records at its own back edge), because each mints
            // a fresh descr through this same call.
            if majit_ir::eval_breaker_word::is_back_edge_poll_guard(op) {
                fd.set_back_edge_poll();
            }
        }
        // resume.py: RPython does NOT carry frame sizes out-of-band.
        // The decoder reads jitcode liveness (jitcode.position_info) at
        // each frame's resume pc. majit routes this through the global
        // `frame_value_count_at` callback registered by pyre-jit-trace.
        let _ = frame_sizes;
    }

    /// Construct a constant OpRef whose value rides inline on the variant
    /// tag.
    ///
    /// RPython equivalent: `ConstInt(value)` — constants in RPython are
    /// first-class Const objects, not boxes. With inline Consts the value
    /// is carried on the `OpRef::const_*` variant itself: no pool
    /// reservation and no `seed_constant` step (its const arm is a
    /// tautological no-op, asserting only that the variant tag the caller
    /// already minted from `value` matches `value`'s type).
    ///
    /// NOTE: do NOT route through `make_constant`. That helper is the
    /// `optimizer.py:make_constant(box, constbox)` analogue and is meant
    /// to forward an existing **box** OpRef to a constant value. It bails
    /// out early when the input is already a constant OpRef
    /// (`is_constant()` true), which would silently drop the new entry.
    pub fn make_constant_int(&mut self, value: i64) -> OpRef {
        // history.py ConstInt.value inline.
        OpRef::const_int(value)
    }

    pub fn make_constant_ref(&mut self, value: GcRef) -> OpRef {
        // history.py ConstPtr.value inline. The GC walker forwards the
        // inline GcRef across minor collection.
        OpRef::const_ptr(value)
    }

    pub fn make_constant_float(&mut self, value: f64) -> OpRef {
        // history.py ConstFloat.value inline.
        OpRef::const_float(value)
    }

    /// Look up the operation that produces a given box.
    /// Used for pattern matching nested operations (e.g., int_add(int_add(x, C1), C2)).
    /// Returns a clone to avoid borrow conflicts with mutable ctx methods.
    ///
    /// optimizer.py `as_operation` ("You should never check
    /// isinstance(op, AbstractResOp) directly. Instead, use this
    /// helper.") admits a producer only when `op in
    /// self._emittedoperations` — ops emitted by THIS optimizer run.
    /// Without the gate, a Phase-2 rule can follow a label arg to its
    /// Phase-1 producer and re-express a per-iteration value in terms
    /// of the preamble's entry box (e.g. int_add reassociation turning
    /// `loop_arg - 1` into `preamble_entry - 2`), a box the loop header
    /// does not carry per-iteration.
    pub fn get_producing_op(&self, op: &Operand) -> Option<majit_ir::OpRc> {
        // resoperation.py AbstractResOpOrInputArg `_forwarded` host: a box's producing op is its
        // bound op (set at emit, mod.rs bind_op before new_operations.push).
        // Walk the forwarding chain first (resoperation.py get_box_replacement) so the
        // replacement box's producer is read.
        let producer = op.get_box_replacement(false).bound_op()?;
        // optimizer.py `op in self._emittedoperations` — keyed by the
        // producer's box identity (`from_bound_op` memoizes one box per op Rc,
        // so this is the same box recorded at emit).
        if !self
            .emitted_operations
            .contains(&majit_ir::operand::Operand::from_bound_op(&producer))
        {
            return None;
        }
        Some(producer)
    }

    /// Number of emitted operations so far.
    pub fn num_emitted(&self) -> usize {
        self.new_operations.len()
    }

    /// Get the last emitted operation, if any.
    pub fn last_emitted_operation(&self) -> Option<&Op> {
        self.new_operations.last().map(|rc| rc.as_ref())
    }

    /// `optimizer.py get_constant_box`:
    /// ```python
    /// def get_constant_box(self, box):
    ///     box = get_box_replacement(box)
    ///     if isinstance(box, Const):
    ///         return box
    ///     if box.type == 'i':
    ///         info = box.get_forwarded()
    ///         if isinstance(info, IntBound) and info.is_constant():
    ///             return ConstInt(info.get_constant_int())
    ///     return None
    /// ```
    pub fn get_constant_box(&self, op: &Operand) -> Option<Value> {
        // optimizer.py: box = get_box_replacement(box)
        op.with_box_replacement(|resolved| {
            // optimizer.py:381-382: isinstance(box, Const) → return box
            if let Some(v) = resolved.const_value() {
                return Some(v);
            }
            // optimizer.py:383-386: box.type == 'i' + IntBound + is_constant
            if resolved.type_() == majit_ir::Type::Int
                && let Some(b) = resolved.int_bound()
                && b.is_constant()
            {
                return Some(Value::Int(b.get_constant_int()));
            }
            None
        })
    }

    pub fn get_constant_int_box(&self, op: &Operand) -> Option<i64> {
        match self.get_constant_box(op)? {
            Value::Int(i) => Some(i),
            _ => None,
        }
    }

    pub fn get_constant_float_box(&self, op: &Operand) -> Option<f64> {
        match self.get_constant_box(op)? {
            Value::Float(f) => Some(f),
            _ => None,
        }
    }

    /// Constant pointer value of a `Ref`-typed operand (`GcRef.0`).  Quasi-
    /// immutable dependency objects (`optimize_QUASIIMMUT_FIELD`'s struct /
    /// namespace operand) are baked as `ConstPtr`, so their pointer must be
    /// read from `Value::Ref`, not `Value::Int`.  `Int` is accepted as a
    /// fallback for any path that already encodes the pointer as an integer.
    pub fn get_constant_ptr_box(&self, op: &Operand) -> Option<u64> {
        match self.get_constant_box(op)? {
            Value::Ref(r) => Some(r.0 as u64),
            Value::Int(i) => Some(i as u64),
            _ => None,
        }
    }

    /// `isinstance(opref, Const)` parity — True only when the OpRef itself
    /// is in the constant namespace. Does NOT walk the forwarding chain;
    /// a body-namespace OpRef forwarded to a Const via `make_constant` returns
    /// None here. Use `get_constant_box` for `getrawconstoption` semantics
    /// (includes IntBound synthesis and chain walk).
    pub fn isinstance_const(&self, opref: OpRef) -> Option<Value> {
        if !opref.is_constant() {
            return None;
        }
        // Const variants carry the value in the variant itself
        // (history.py `Const.value`).
        opref.inline_const_to_value()
    }

    /// `isinstance(opref, ConstInt)` parity — narrow check without chain walk.
    pub fn isinstance_const_int(&self, opref: OpRef) -> Option<i64> {
        match self.isinstance_const(opref)? {
            Value::Int(i) => Some(i),
            _ => None,
        }
    }

    /// optimizer.py `constant_fold(op)`:
    ///
    /// ```python
    /// def constant_fold(self, op):
    ///     self.protect_speculative_operation(op)
    ///     argboxes = [self.get_constant_box(op.getarg(i))
    ///                 for i in range(op.numargs())]
    ///     return execute_nonspec_const(self.cpu, None,
    ///                                    op.getopnum(), argboxes,
    ///                                    op.getdescr(), op.type)
    /// ```
    ///
    /// Returns `None` only when:
    ///  - `supports_guard_gc_type == false` and the op is a memory-
    ///    reading fold (array/string/unicode).  Upstream `optimizer.py:
    ///    822-825` relies on "we don't unroll in that case"; pyre's
    ///    `constant_fold` runs outside the unroll pass too, so this
    ///    gate is placed here at the call site, NOT inside
    ///    `protect_speculative_operation` (which matches upstream as a
    ///    plain `()` function).
    ///  - Helper-internal `Ok(None)` for OVF/shift/divide-by-zero/
    ///    non-finite cast (see `pure.rs`'s `propagate_forward`).
    ///    Every other path panics (caller-invariant, NotImplemented).
    pub fn constant_fold(&self, op: &Op) -> Option<Value> {
        // optimizer.py:822-825: "if cpu.supports_guard_gc_type is
        // false, we can't really do this check at all, but then we
        // don't unroll in that case."  Gate memory-reading ops here
        // so protect_speculative_operation stays a plain () function.
        if !majit_gc::supports_guard_gc_type() {
            use majit_ir::OpCode;
            if matches!(
                op.opcode,
                OpCode::GetarrayitemGcPureI
                    | OpCode::GetarrayitemGcPureR
                    | OpCode::GetarrayitemGcPureF
                    | OpCode::ArraylenGc
                    | OpCode::Strgetitem
                    | OpCode::Strlen
                    | OpCode::Unicodegetitem
                    | OpCode::Unicodelen
            ) {
                return None;
            }
        }
        self.protect_speculative_operation(op);
        // protect_speculative_operation defers an `InvalidLoop` instead of
        // raising; bail before reading the (now unvalidated) memory so the
        // fold never dereferences an ill-typed speculative pointer.
        if self.has_pending_invalid_loop() {
            return None;
        }
        let mut argboxes: Vec<Value> = Vec::with_capacity(op.num_args());
        for i in 0..op.num_args() {
            argboxes.push(
                self.get_constant_box(&op.arg(i).get_box_replacement(false))
                    .expect("constant_fold: arg must be Const (pure.rs:993-1006 pre-check)"),
            );
        }
        match crate::executor::execute_nonspec_const(
            self.cpu.as_ref(),
            op.opcode,
            &argboxes,
            op.getdescr().as_ref(),
            op.result_type(),
        ) {
            Ok(folded) => folded,
            Err(crate::executor::NoConstExecutor) => panic!(
                "execute_nonspec_const: no helper registered for opcode {:?} \
                 (executor.py:610 NotImplementedError) args={:?}",
                op.opcode, argboxes
            ),
        }
    }

    /// optimizer.py `protect_speculative_operation(op)` — when
    /// constant-folding a pure operation that reads memory from a
    /// gcref, validate the gcref is non-null and of a valid type;
    /// signal `InvalidLoop` otherwise.
    ///
    /// Returns `()` — matching upstream's Python `def protect_
    /// speculative_operation(self, op):` which has no return value.
    /// Either returns normally (validation passed) or records a deferred
    /// `InvalidLoop` signal on the context (via `signal_invalid_loop`),
    /// which `constant_fold` observes to bail out and the driver surfaces
    /// as `Err(InvalidLoop)` at its next barrier (`unroll.py:119-123`).
    ///
    /// The `supports_guard_gc_type == false` gate that was previously
    /// inside this function has been moved to `constant_fold` (the
    /// only caller), matching upstream's architectural invariant:
    /// *"if cpu.supports_guard_gc_type is false, we can't really do
    /// this check at all, but then we don't unroll in that case"*
    /// (optimizer.py).
    ///
    /// Caller-invariant violations (missing box, descr, wrong Value
    /// variant) panic — upstream would `AttributeError`.
    ///
    /// Branches mirror the upstream `if / elif / elif / elif / else`
    /// chain line-for-line:
    ///  - pure GETFIELD_GC_PURE_*  → `protect_speculative_field`
    ///  - GETARRAYITEM_GC_PURE_* / ARRAYLEN_GC → `protect_speculative_array`
    ///  - STRGETITEM / STRLEN → `protect_speculative_string`
    ///  - UNICODEGETITEM / UNICODELEN → `protect_speculative_unicode`
    ///  - default → no validation needed (return early).
    ///
    /// For the get*item branches, `cpu.bh_arraylen_gc / bh_strlen /
    /// bh_unicodelen` reads the container length and the routine
    /// checks `0 <= index < length`.  When `bh_strlen / bh_unicodelen`
    /// returns `None` (pyre has no fold-time str/unicode layout), the
    /// bounds check is skipped — equivalent to RPython where the
    /// optimizer falls back to runtime evaluation in that case.
    fn protect_speculative_operation(&self, op: &Op) {
        use majit_ir::OpCode;

        let opnum = op.opcode;

        let descr = op.getdescr();
        if opnum.is_getfield() {
            // optimizer.py pure-getfield branch.
            let gcref = match self
                .get_constant_box(&op.arg(0).get_box_replacement(false))
                .expect("protect_speculative_operation: arg0 must be Const")
            {
                Value::Ref(r) => r,
                v => unreachable!(
                    "GETFIELD_GC_PURE_* arg0 must be a gcref (Value::Ref); got {:?}",
                    v
                ),
            };
            let fd = descr
                .as_ref()
                .and_then(|d| d.as_field_descr())
                .expect("GETFIELD_GC_PURE_* descr must be a FieldDescr");
            if let Err(reason) = self.cpu.protect_speculative_field(gcref, fd) {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] {reason} — field={:?} descr={}",
                        fd.field_name(),
                        fd.repr_of_descr()
                    );
                }
                self.signal_invalid_loop(reason);
            }
            return;
        }

        let arraylength: i64 = if matches!(
            opnum,
            OpCode::GetarrayitemGcPureI
                | OpCode::GetarrayitemGcPureR
                | OpCode::GetarrayitemGcPureF
                | OpCode::ArraylenGc
        ) {
            // optimizer.py:834-841 array branch.
            let array = match self
                .get_constant_box(&op.arg(0).get_box_replacement(false))
                .expect("protect_speculative_operation: array arg0 must be Const")
            {
                Value::Ref(r) => r,
                v => unreachable!(
                    "GETARRAYITEM_GC_PURE_* / ARRAYLEN_GC arg0 must be a gcref; got {:?}",
                    v
                ),
            };
            let ad = descr
                .as_ref()
                .and_then(|d| d.as_array_descr())
                .expect("array op descr must be an ArrayDescr");
            if self.cpu.protect_speculative_array(array, ad).is_err() {
                self.signal_invalid_loop("protect_speculative_array");
                return;
            }
            if opnum == OpCode::ArraylenGc {
                return;
            }
            self.cpu
                .bh_arraylen_gc(array, ad)
                .expect("bh_arraylen_gc must succeed after protect_speculative_array")
        } else if matches!(opnum, OpCode::Strgetitem | OpCode::Strlen) {
            // optimizer.py:843-848 string branch.
            let string = match self
                .get_constant_box(&op.arg(0).get_box_replacement(false))
                .expect("protect_speculative_operation: string arg0 must be Const")
            {
                Value::Ref(r) => r,
                v => unreachable!("STRGETITEM / STRLEN arg0 must be a gcref; got {:?}", v),
            };
            if self.cpu.protect_speculative_string(string).is_err() {
                self.signal_invalid_loop("protect_speculative_string");
                return;
            }
            if opnum == OpCode::Strlen {
                return;
            }
            self.cpu
                .bh_strlen(string)
                .expect("bh_strlen must succeed after protect_speculative_string")
        } else if matches!(opnum, OpCode::Unicodegetitem | OpCode::Unicodelen) {
            // optimizer.py:850-855 unicode branch.
            let unicode = match self
                .get_constant_box(&op.arg(0).get_box_replacement(false))
                .expect("protect_speculative_operation: unicode arg0 must be Const")
            {
                Value::Ref(r) => r,
                v => unreachable!(
                    "UNICODEGETITEM / UNICODELEN arg0 must be a gcref; got {:?}",
                    v
                ),
            };
            if self.cpu.protect_speculative_unicode(unicode).is_err() {
                self.signal_invalid_loop("protect_speculative_unicode");
                return;
            }
            if opnum == OpCode::Unicodelen {
                return;
            }
            self.cpu
                .bh_unicodelen(unicode)
                .expect("bh_unicodelen must succeed after protect_speculative_unicode")
        } else {
            // optimizer.py:857-858 else: return — nothing to validate.
            return;
        };

        // optimizer.py:860-862 shared bounds check:
        //   index = self.get_constant_box(op.getarg(1)).getint()
        //   if not (0 <= index < arraylength): raise SpeculativeError
        let index = match self
            .get_constant_box(&op.arg(1).get_box_replacement(false))
            .expect("protect_speculative_operation: arg1 must be Const")
        {
            Value::Int(i) => i,
            v => unreachable!(
                "GETARRAYITEM / STRGETITEM / UNICODEGETITEM arg1 must be an int index; got {:?}",
                v
            ),
        };
        if !(0 <= index && index < arraylength) {
            self.signal_invalid_loop("index out of bounds for constant-fold");
        }
    }

    /// Look up the producing `Op` for an OpRef in `new_operations`.
    /// Returns `None` for inputargs, constants, and OpRefs not yet emitted.
    ///
    /// RPython equivalent: holding a reference to the producing `Box`
    /// itself (every Box is a Python object, so identity lookup is the
    /// `is` operator — O(1)). pyre's flat `OpRef(u32)` cannot mirror
    /// that in O(1) without an auxiliary index; mutation patterns on
    /// `new_operations` (in-place `replace_new_operation` from
    /// `optimizer.rs`'s `emit_operation` and `rewrite.rs`'s
    /// `optimize_guard_class` / `optimize_guard_value`, plus
    /// `remove(jump_idx)` in `optimizer.rs`'s
    /// `optimize_with_constants_and_inputs_at`) make a maintained
    /// `pos_to_index` brittle.
    /// unifies on this single API. A maintained index would need to be kept
    /// in sync with every in-place replacement and removal site.
    pub fn op_at(&self, opref: OpRef) -> Option<&Op> {
        if let Some(op) = self
            .new_operations
            .iter()
            .rev()
            .find(|op| op.pos().get() == opref)
        {
            return Some(op);
        }
        // Phase 1 emit-op fallback (history.py:220 box.type parity for
        // cross-phase OpRefs). Reverse scan mirrors `new_operations` so a
        // later replacement of the same `pos` wins. Returned `&Op` is
        // safe to read for `.type_` and other intrinsic attributes; arg
        // / descr fields refer to Phase 1's namespace and should not be
        // dereferenced through this path (Phase 2 callers only consume
        // `op.type_` via `opref_type`).
        self.phase1_emit_ops
            .iter()
            .rev()
            .find(|op| op.pos().get() == opref)
            .map(|rc| rc.as_ref())
    }

    /// optimizer.py: clear_newoperations()
    /// Clear the output operation list (used when restarting optimization).
    pub fn clear_newoperations(&mut self) {
        self.new_operations.clear();
        self.new_operations_index.clear();
        // Reset next_pos to the iteration's first fresh OpRef position
        // (right after the inputarg slice in the OpRef namespace), but
        // never below the prior iteration's watermark. The context
        // survives across iterations (e.g. Phase 2 final_ctx reused as
        // `jump_ctx` for short-preamble inlining); `reserve_pos` only
        // skips constants, so the previous iteration's typed slots must
        // remain reserved to keep `alloc_op_position` from handing back
        // a pos that already names an emitted op.
        let base = self.inputarg_base + self.num_inputs;
        self.next_pos = self.next_pos.max(base);
        self.const_infos.clear();
    }

    /// resoperation.py: `op.type` parity. The typed OpRef enum
    /// encodes `box.type` (`AbstractValue.type` ∈ {`'i'`, `'r'`, `'f'`,
    /// `'v'`}) directly in the variant tag, so reading the tag is the
    /// line-by-line equivalent of upstream `box.type`. The fall-through
    /// arms cover residual cases — ops that have not yet been emitted,
    /// inputarg slots, and PtrInfo-derived Ref typing. Raw-pointer
    /// `ConstInt` Boxes keep `op.type == 'i'` and become `ConstPtrInfo`
    /// through `getrawptrinfo` per `info.py`.
    ///
    /// Returns `None` only when none of the above sources have type
    /// information for the OpRef. Callers must treat `None` like
    /// RPython's "unknown type" path and avoid making structural
    /// assumptions about it.
    pub fn opref_type(&self, opref: OpRef) -> Option<majit_ir::Type> {
        let resolved = self.get_replacement_opref(opref);
        // 0. Inputarg slot (recorder-side `InputArg{Int,Ref,Float}.tp`,
        //    history.py:220 parity per resoperation.py InputArgInt/727/739).  allow-line-citation
        //    `inputarg_types[idx]` is the canonical Box.type source
        //    for slot positions — a cross-phase caller that minted the
        //    OpRef with `input_arg_int(idx)` for a Ref-typed slot would
        //    mismatch the variant tag against the recorder's actual
        //    type, so consult the side-table first for inputarg
        //    positions.  Returns `None` for non-inputarg OpRefs, which
        //    falls through to the variant-tag step.
        if let Some(tp) = self.inputarg_type(resolved) {
            return Some(tp);
        }
        // 1. RPython `AbstractValue.type` (resoperation.py) parity. The
        //    OpRef enum encodes `box.type` directly in the variant tag
        //    (`ConstInt`/`InputArgInt`/`IntOp` → Int, etc.), so reading
        //    the tag is the line-by-line equivalent of upstream `box.type`.
        //    `OpRef::None` returns `None` here and falls through.
        if let Some(tp) = resolved.ty() {
            return Some(tp);
        }
        // 2. Producing op's intrinsic `type_` (resoperation.py:1693
        //    `res.type = result_type` in `create_class_for_op`, i.e.
        //    `opclasses[opnum].type`). `Op::new` populates this at
        //    construction; this is the primary fast path after allocation.
        //    Reached only when `resolved` is `None`/`TempVar` (every other
        //    variant is typed by step 1); inline-Const collapsed the old
        //    pool-indexed "seeded constant without a typed variant" case
        //    into step 1, so no separate const-value fall-through remains.
        if let Some(op) = self.op_at(resolved)
            && op.type_ != majit_ir::Type::Void
        {
            return Some(op.type_);
        }
        // 3. PtrInfo-derived type (box.type parity for
        //    virtual heads across phase boundaries). Phase 1 virtualizes
        //    NewWithVtable / New / NewArray etc. by returning
        //    `OptimizationResult::Remove` from the relevant `optimize_*`
        //    method (virtualize.py:208-225) — the op never lands in
        //    `new_operations` and is therefore absent from
        //    `phase1_emit_ops`. Phase 2 imports the virtual head's
        //    `PtrInfo` via `setinfo_from_preamble` (unroll.py) but
        //    starts with a fresh `value_types` map, so the prior four
        //    sources all miss. RPython preserves the type intrinsically
        //    on the Box object; pyre recovers it from the PtrInfo variant
        //    because every variant maps to a unique RPython box type.
        //    Ref-typed: Virtual / VirtualArray / VirtualStruct /
        //    VirtualArrayStruct / Array / Struct / Instance / Constant /
        //    NonNull / Virtualizable / Str (RPython instances of
        //    AbstractStructPtrInfo / ArrayPtrInfo / StrPtrInfo carry 'r').
        //    Int-typed: VirtualRawBuffer / VirtualRawSlice
        //    (info.py RawBufferPtrInfo + getrawptrinfo() — these
        //    describe raw pointers stored in 'i' Boxes).
        let resolved_box = self.get_box_replacement_operand_opt(opref);
        if let Some(info) = resolved_box.as_ref().and_then(Operand::ptr_info) {
            return Some(match &*info {
                crate::optimizeopt::info::PtrInfo::VirtualRawBuffer(_)
                | crate::optimizeopt::info::PtrInfo::VirtualRawSlice(_) => majit_ir::Type::Int,
                _ => majit_ir::Type::Ref,
            });
        }
        None
    }

    /// Read-only `box.type` lookup for inputarg-slot OpRefs.  Returns
    /// `Some(tp)` when `opref` falls in either the current context's
    /// own inputarg range `[inputarg_base, inputarg_base + num_inputs)`
    /// (RPython invariant) or the shared low range `[0, num_inputs)`
    /// (Phase 1's inputarg slot OpRefs referenced from Phase 2 via
    /// `imported_label_args`).  Returns `None` for constants, sentinels,
    /// out-of-range OpRefs, Void-typed slots, or empty `inputarg_types`
    /// (test contexts that bypass `setup_optimizations`).
    ///
    /// `[0, num_inputs)` fallback: in RPython each `InputArgInt`/
    /// `InputArgRef`/`InputArgFloat` (resoperation.py/727/739) Box
    /// carries `.type` intrinsically, so Phase 2 reads the same
    /// `box.type` regardless of which iteration's TraceIterator produced
    /// the box.  Pyre's flat `OpRef(u32)` namespace separates Phase 1
    /// inputargs (at `[0, num_inputs)`) from Phase 2 inputargs (at
    /// `[phase2_inputarg_base, phase2_inputarg_base + num_inputs)`), but
    /// `Optimizer.trace_inputargs` is identical between phases
    /// (single recorder source).  Indexing the same `inputarg_types`
    /// Vec by raw position recovers Phase 1 slot types from Phase 2
    /// without a separate side-table (history.py:220 parity).
    ///
    /// TODO: pyre stores the InputArg type on a
    /// graph-level side-table instead of a per-producer InputArg
    /// variant tag because the producer layout splits ResOp / InputArg /
    /// Const at construction time only.  Retiring this helper requires
    /// stamping the type onto the InputArg producer so an `Operand::type_()`
    /// read on an existing producer is sufficient.  Until then this lookup
    /// is the read-only counterpart of `resoperation.py/727/739
    /// InputArg{Int,Ref,Float}.type` — it must not materialize a fresh
    /// operand, because the materialization path (`materialize_operand_at`)
    /// keys the new operand's type off the OpRef variant tag and a Phase 2
    /// context referencing a Phase 1 low slot can mismatch that tag against
    /// the canonical `inputarg_types[idx]`.
    pub fn inputarg_type(&self, opref: OpRef) -> Option<majit_ir::Type> {
        if opref.is_none() || opref.is_constant() {
            return None;
        }
        let raw = opref.raw();
        let ni = self.num_inputs as usize;
        let idx = if raw >= self.inputarg_base && (raw - self.inputarg_base) < self.num_inputs {
            (raw - self.inputarg_base) as usize
        } else if self.inputarg_base > 0 && raw < self.num_inputs {
            // Phase 1 inputarg slot accessed from a non-Phase-1 context
            // (Phase 2 / bridge).  RPython resolves these through Box
            // identity; flat-OpRef pyre uses the shared `inputarg_types`
            // Vec — no materialization, just the recorder-seeded type
            // table.
            raw as usize
        } else {
            return None;
        };
        if idx >= ni {
            return None;
        }
        let opref = *self.inputargs.get(idx)?;
        let tp = opref.ty()?;
        if tp == majit_ir::Type::Void {
            None
        } else {
            Some(tp)
        }
    }

    /// Look up the declared type of inputarg slot `idx` (zero-based) from
    /// the `inputargs` Vec seeded by `setup_optimizations`. Returns
    /// `None` if the slot is out of range, the type is `Void`, or the Vec
    /// has not been populated. Reads `inputargs[idx].ty()` — each entry
    /// is a typed `OpRef::input_arg_*` (optimizer.py:34 parity).
    pub fn inputarg_type_at(&self, idx: usize) -> Option<majit_ir::Type> {
        let opref = *self.inputargs.get(idx)?;
        let tp = opref.ty()?;
        if tp == majit_ir::Type::Void {
            None
        } else {
            Some(tp)
        }
    }

    /// Strict counterpart to `inputarg_type_at`. Panics when the slot is
    /// out of range, the variant yields `Void`, or `inputargs` was not
    /// populated by `setup_optimizations`. Mirrors RPython's
    /// `box.type` invariant (history.py:220).
    pub fn inputarg_type_at_strict(&self, idx: usize) -> majit_ir::Type {
        match self.inputargs.get(idx).and_then(|o| o.ty()) {
            Some(majit_ir::Type::Void) => panic!(
                "inputarg_type_at_strict: slot {idx} is Void; \
                 RPython invariant violated (history.py:220 box.type)"
            ),
            Some(tp) => tp,
            None => panic!(
                "inputarg_type_at_strict: slot {idx} out of range \
                 (inputargs.len() = {}); setup_optimizations did not \
                 seed the inputarg list",
                self.inputargs.len()
            ),
        }
    }

    /// Read-only access to the inputarg slot's typed `OpRef` (variant tag
    /// is `box.type`). Returns `None` when the slot is out of range.
    /// Mirrors PyPy `self.inputargs[idx]` (optimizer.py:34).
    pub fn inputarg_at(&self, idx: usize) -> Option<majit_ir::OpRef> {
        self.inputargs.get(idx).copied()
    }

    /// info.py `getrawptrinfo(op)` parity (line-by-line port).
    ///
    /// ```python
    /// def getrawptrinfo(op):
    ///     from rpython.jit.metainterp.optimizeopt.intutils import IntBound
    ///     assert op.type == 'i'
    ///     op = op.get_box_replacement()
    ///     assert op.type == 'i'
    ///     if isinstance(op, ConstInt):
    ///         return ConstPtrInfo(op)
    ///     fw = op.get_forwarded()
    ///     if isinstance(fw, IntBound):
    ///         return None
    ///     if fw is not None:
    ///         assert isinstance(fw, AbstractRawPtrInfo)
    ///         return fw
    ///     return None
    /// ```
    ///
    /// info.py `getrawptrinfo(op)` parity — orthodox return
    /// shape that preserves RPython `_forwarded` object identity.
    /// `PtrInfoHandle::Const(_)` for the `isinstance(op, ConstInt)`
    /// fresh `ConstPtrInfo` arm; `PtrInfoHandle::Live(rc)` for the
    /// `return fw` arm carrying the live `Rc<RefCell<PtrInfo>>` cell
    /// from the chain terminal's `_forwarded` slot.
    ///
    /// Callers that need an owned snapshot can call `.snapshot()`;
    /// callers that need identity/value parity (`same_info`, in-place
    /// mutation) use `.same_info()` / `.borrow()` / `.borrow_mut()`.
    pub fn getrawptrinfo_handle(&self, op: &Operand) -> Option<PtrInfoHandle> {
        use crate::optimizeopt::info::OpInfo;
        use majit_ir::forwarding::Forwarded;
        // info.py:867 — `assert op.type == 'i'`.
        debug_assert_eq!(
            op.type_(),
            majit_ir::Type::Int,
            "getrawptrinfo_handle: expected 'i'-typed operand"
        );
        // info.py:868 — `op = op.get_box_replacement()`.
        let terminal = op.get_box_replacement(false);
        // info.py:869 — `assert op.type == 'i'`.
        debug_assert_eq!(
            terminal.type_(),
            majit_ir::Type::Int,
            "getrawptrinfo_handle: terminal expected 'i'-typed operand"
        );
        // info.py — `if isinstance(op, ConstInt): return ConstPtrInfo(op)`.
        if let Some(Value::Int(bits)) = terminal.const_value() {
            return Some(PtrInfoHandle::Const(PtrInfo::Constant(majit_ir::GcRef(
                bits as usize,
            ))));
        }
        // info.py:872-878 line-by-line dispatch on the forwarded slot:
        //     fw = op.get_forwarded()
        //     if isinstance(fw, IntBound): return None
        //     if fw is not None:
        //         assert isinstance(fw, AbstractRawPtrInfo)
        //         return fw
        //     return None
        match &terminal.get_forwarded() {
            Forwarded::None => None,
            Forwarded::Info(OpInfo::IntBound(_)) => None,
            Forwarded::Info(OpInfo::Ptr(rc)) => Some(PtrInfoHandle::Live(std::rc::Rc::clone(rc))),
            // info.py `assert isinstance(fw, AbstractRawPtrInfo)` —
            // a non-Ptr, non-IntBound forwarded on an `'i'`-typed terminal
            // is a structural invariant violation upstream would crash on.
            Forwarded::Info(other) => panic!(
                "getrawptrinfo: forwarded must be IntBound or AbstractRawPtrInfo \
                 (info.py:876), got {:?}",
                std::mem::discriminant(other),
            ),
            // Terminal of `get_box_replacement(false)` can only be `None`
            // or `Info(_)` per the chain walker (forwarding.rs); a
            // `Forwarded::Const` terminal is materialized inline by the
            // walker into a fresh operand whose own slot is None.
            Forwarded::Const(_)
            | Forwarded::SmallConst(_)
            | Forwarded::SmallWide(_)
            | Forwarded::Op(_)
            | Forwarded::InputArg(_) => {
                unreachable!(
                    "getrawptrinfo: chain terminal must not carry Forwarded::Const \
                 (forwarding.rs get_box_replacement walker invariant)",
                )
            }
        }
    }

    /// info.py `getptrinfo(op)` parity (line-by-line port).
    ///
    /// ```python
    /// def getptrinfo(op):
    ///     if op.type == 'i':
    ///         return getrawptrinfo(op)
    ///     elif op.type == 'f':
    ///         return None
    ///     assert op.type == 'r'
    ///     op = get_box_replacement(op)
    ///     assert op.type == 'r'
    ///     if isinstance(op, ConstPtr):
    ///         return ConstPtrInfo(op)
    ///     fw = op.get_forwarded()
    ///     if fw is not None:
    ///         assert isinstance(fw, PtrInfo)
    ///         return fw
    ///     return None
    /// ```
    /// The Int arm delegates to `getrawptrinfo` per `info.py`.
    /// The Float arm short-circuits to `None`. The Void arm panics —
    /// `info.py assert op.type == 'r'` rejects Void boxes outright;
    /// no synthetic Void filler box exists that would smuggle a
    /// type-erased pointer through this helper.
    pub fn getptrinfo(&self, op: &Operand) -> Option<PtrInfo> {
        self.getptrinfo_handle(op).map(|h| h.snapshot())
    }

    /// info.py `getptrinfo(op)` parity — orthodox return
    /// shape that preserves RPython `_forwarded` object identity.
    /// See `getrawptrinfo_handle` for the variant semantics.
    pub fn getptrinfo_handle(&self, op: &Operand) -> Option<PtrInfoHandle> {
        use crate::optimizeopt::info::OpInfo;
        use majit_ir::forwarding::Forwarded;
        match op.type_() {
            // info.py — `if op.type == 'i': return getrawptrinfo(op)`.
            majit_ir::Type::Int => return self.getrawptrinfo_handle(op),
            // info.py:883-884 — `elif op.type == 'f': return None`.
            majit_ir::Type::Float => return None,
            // info.py:885 — `assert op.type == 'r'`.
            majit_ir::Type::Ref => {}
            majit_ir::Type::Void => panic!(
                "getptrinfo_handle: op.type == 'v' (info.py:885 `assert op.type == 'r'`); \
                 caller must guard on a typed box (Int/Ref/Float) — Void boxes \
                 carry no PtrInfo upstream",
            ),
        }
        // info.py:886-893 type 'r' arm:
        //   op = get_box_replacement(op)
        //   if isinstance(op, ConstPtr): return ConstPtrInfo(op)
        //   fw = op.get_forwarded()
        //   if fw is not None:
        //       assert isinstance(fw, PtrInfo)
        //       return fw
        //   return None
        let terminal = op.get_box_replacement(false);
        debug_assert_eq!(
            terminal.type_(),
            majit_ir::Type::Ref,
            "getptrinfo_handle: chain-walked replacement lost Ref type (got {:?})",
            terminal.type_(),
        );
        // info.py: if isinstance(op, ConstPtr): return ConstPtrInfo(op)
        if let Some(Value::Ref(gcref)) = terminal.const_value() {
            return Some(PtrInfoHandle::Const(PtrInfo::Constant(gcref)));
        }
        match &terminal.get_forwarded() {
            Forwarded::None => None,
            Forwarded::Info(OpInfo::Ptr(rc)) => Some(PtrInfoHandle::Live(std::rc::Rc::clone(rc))),
            // info.py `assert isinstance(fw, PtrInfo)` — a Ref-typed
            // terminal must not forward to IntBound / FloatConst / Unknown.
            Forwarded::Info(other) => panic!(
                "getptrinfo: forwarded must be PtrInfo (info.py:892), got {:?}",
                std::mem::discriminant(other),
            ),
            // Terminal of `get_box_replacement(false)` can only be `None`
            // or `Info(_)` per the chain walker (forwarding.rs); a
            // `Forwarded::Const` terminal is materialized inline by the
            // walker into a fresh operand whose own slot is None.
            Forwarded::Const(_)
            | Forwarded::SmallConst(_)
            | Forwarded::SmallWide(_)
            | Forwarded::Op(_)
            | Forwarded::InputArg(_) => {
                unreachable!(
                    "getptrinfo: chain terminal must not carry Forwarded::Const \
                 (forwarding.rs get_box_replacement walker invariant)",
                )
            }
        }
    }

    /// optimizer.py `Optimizer.getinfo`.
    ///
    /// ```python
    /// def getinfo(self, op):
    ///     if op.type == 'r':
    ///         return getptrinfo(op)
    ///     elif op.type == 'i':
    ///         if self.is_raw_ptr(op):
    ///             return getptrinfo(op)
    ///         return self.getintbound(op)
    ///     elif op.type == 'f':
    ///         if get_box_replacement(op).is_constant():
    ///             return info.FloatConstInfo(get_box_replacement(op))
    /// ```
    ///
    /// Reads `_forwarded` off `op` itself (after `get_box_replacement`), never
    /// by re-resolving `op.to_opref()` through the producer registry. A bound
    /// box that the OpRef map does not know still exports its PtrInfo /
    /// IntBound — the same object `unroll.py _expand_info` stores in
    /// `exported_infos`.
    pub fn getinfo(&self, op: &Operand) -> Option<crate::optimizeopt::info::OpInfo> {
        use crate::optimizeopt::info::{FloatConstInfo, OpInfo};
        match op.type_() {
            majit_ir::Type::Ref => self.getinfo_ptr(op),
            majit_ir::Type::Int => {
                if self.is_raw_ptr(op) {
                    self.getinfo_ptr(op)
                } else {
                    Some(self.getinfo_intbound(op))
                }
            }
            majit_ir::Type::Float => {
                let replaced = op.get_box_replacement(false);
                if replaced.is_constant() {
                    match replaced.const_value() {
                        Some(Value::Float(f)) => {
                            Some(OpInfo::FloatConstInfo(FloatConstInfo::new(f)))
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            }
            majit_ir::Type::Void => None,
        }
    }

    fn getinfo_ptr(&self, op: &Operand) -> Option<crate::optimizeopt::info::OpInfo> {
        use crate::optimizeopt::info::OpInfo;
        match self.getptrinfo_handle(op) {
            Some(PtrInfoHandle::Live(rc)) => Some(OpInfo::Ptr(rc)),
            Some(PtrInfoHandle::Const(pi)) => Some(OpInfo::ptr(pi)),
            None => None,
        }
    }

    /// optimizer.py `getintbound(op)` as the `getinfo` int arm: always
    /// returns an IntBound, lazily installing unbounded on a live box
    /// (`op.set_forwarded(intbound)`).
    fn getinfo_intbound(&self, op: &Operand) -> crate::optimizeopt::info::OpInfo {
        use crate::optimizeopt::info::OpInfo;
        match self.getintbound_handle(op) {
            IntBoundHandle::Live(rc) | IntBoundHandle::Const(rc) => OpInfo::IntBound(rc),
        }
    }

    /// virtualstate.py `GenerateGuardState.get_runtime_field(box, descr)`
    /// parity.
    ///
    /// ```python
    /// def get_runtime_field(self, box, descr):
    ///     struct = box.getref_base()
    ///     if descr.is_pointer_field():
    ///         return InputArgRef(self.cpu.bh_getfield_gc_r(struct, descr))
    ///     elif descr.is_float_field():
    ///         return InputArgFloat(self.cpu.bh_getfield_gc_f(struct, descr))
    ///     else:
    ///         return InputArgInt(self.cpu.bh_getfield_gc_i(struct, descr))
    /// ```
    ///
    /// Walks `runtime_box` to its `Value::Ref(gcref)` payload and reads
    /// the typed value at `gcref.raw() + descr.offset()` using
    /// `FieldDescr.field_size()` / `is_field_signed()` (the same
    /// (offset, size, sign) triple `Cpu::bh_getfield_gc_i` consumes on
    /// the backend — `majit-backend-cranelift/src/compiler.rs`). Wraps
    /// the read in a freshly
    /// allocated const OpRef matching `InputArg*` parity.
    ///
    /// Concrete-Ref extractor is `runtime_value_of` (mod.rs) which
    /// cascades box-forwarding chain → const_pool → stamped operand
    /// runtime value (the RPython `InputArg*.value` analog).
    /// Returns `None` when the OpRef does not resolve to a concrete
    /// non-null Ref, when the descr is not a FieldDescr, or when the
    /// runtime pointer is null.
    pub fn get_runtime_field(
        &mut self,
        runtime_box: OpRef,
        descr: &majit_ir::descr::DescrRef,
    ) -> Option<OpRef> {
        // virtualstate.py get_runtime_field `cpu.bh_getfield_gc_*(struct, descr)` reads the
        // field off the *struct* the runtime box points at. RPython's runtime
        // box is always an eagerly allocated object, so `getref_base()` yields
        // a concrete pointer. Under pyre's lazy boxing the runtime box can be
        // an unmaterialized virtual (state.rs NewWithVtable without a W_*Object
        // allocation), so `getref_base()` has no concrete pointer to read. The
        // faithful equivalent is to read the field's tracked box — the
        // `setfield_gc` source the optimizer recorded on the virtual — which is
        // exactly the value `bh_getfield_gc_*` would observe on the eager
        // object. Consult it before the concrete-Ref read; the concrete path is
        // unchanged for already-materialized runtime boxes.
        if let Some(opref) = self.get_virtual_runtime_field(runtime_box, descr) {
            return Some(opref);
        }
        // virtualstate.py get_runtime_item `box.getref_base()` — concrete Ref read.
        // `runtime_value_of` cascades const_pool → stamped operand value
        // (RPython `InputArg*.value` analog).
        let raw = match self.runtime_value_of(runtime_box)? {
            Value::Ref(gcref) if !gcref.is_null() => gcref.0 as i64,
            _ => return None,
        };
        let fd = descr.as_field_descr()?;
        let offset = fd.offset() as i64;
        let ptr = (raw as usize).wrapping_add(offset as usize);
        if raw == 0 {
            return None;
        }
        match fd.field_type() {
            Type::Ref => {
                let val = unsafe { (ptr as *const usize).read_unaligned() };
                Some(self.make_constant_ref(majit_ir::GcRef(val)))
            }
            Type::Float => {
                let val = unsafe { (ptr as *const f64).read_unaligned() };
                Some(self.make_constant_float(val))
            }
            Type::Int => {
                let size = fd.field_size();
                let sign = fd.is_field_signed();
                let val = unsafe {
                    match (size, sign) {
                        (1, true) => (ptr as *const i8).read_unaligned() as i64,
                        (1, false) => (ptr as *const u8).read_unaligned() as i64,
                        (2, true) => (ptr as *const i16).read_unaligned() as i64,
                        (2, false) => (ptr as *const u16).read_unaligned() as i64,
                        (4, true) => (ptr as *const i32).read_unaligned() as i64,
                        (4, false) => (ptr as *const u32).read_unaligned() as i64,
                        _ => (ptr as *const i64).read_unaligned(),
                    }
                };
                Some(self.make_constant_int(val))
            }
            _ => None,
        }
    }

    /// Lazy-boxing companion to `get_runtime_field` / `get_runtime_interiorfield`.
    ///
    /// virtualstate.py reads `cpu.bh_getfield_gc_*(struct, descr)` off the
    /// eagerly allocated struct the runtime box points at. When `runtime_box`
    /// resolves to an unmaterialized `PtrInfo::Virtual` / `VirtualStruct`
    /// instead of a concrete pointer (pyre keeps W_IntObject counters as
    /// virtuals — state.rs NewWithVtable with no allocation), there is no
    /// struct to read; the field's observed value lives on the virtual's
    /// tracked field box (the recorded `setfield_gc` source). This mirrors what
    /// `bh_getfield_gc_*` would observe on the eager object: look up the field
    /// slot via `info.getfield(index_in_parent)` (the same parent-local slot
    /// `enum_forced_boxes_for_entry` reads, in `virtualstate.rs`), resolve
    /// its own runtime value (`runtime_value_of`, the InputArg*.value analog),
    /// and wrap it in a fresh const OpRef matching the concrete-read return
    /// shape. Returns `None` when the runtime box is not a virtual, the slot is
    /// unset, or the slot carries no observed value.
    fn get_virtual_runtime_field(
        &mut self,
        runtime_box: OpRef,
        descr: &majit_ir::descr::DescrRef,
    ) -> Option<OpRef> {
        let fd = descr.as_field_descr()?;
        // virtualstate.py:149-151 `opinfo = getptrinfo(box); assert
        // opinfo.is_virtual()`. Read the field box only off a virtual struct
        // ptrinfo; a concrete/instance ptrinfo falls through to the eager read.
        let op = self.get_box_replacement_operand_opt(runtime_box)?;
        let info = self.getptrinfo(&op)?;
        // virtualstate.py `opinfo._fields[descr.get_index()]`. Storage
        // keys by the allocation SizeDescr list (`parent_list_slot_in`).
        let owner = match &info {
            crate::optimizeopt::info::PtrInfo::Virtual(v) => Some(&v.descr),
            crate::optimizeopt::info::PtrInfo::VirtualStruct(v) => Some(&v.descr),
            _ => None,
        };
        let slot = crate::optimizeopt::virtualize::parent_list_slot_in(owner, fd);
        let field_opref = match &info {
            crate::optimizeopt::info::PtrInfo::Virtual(_)
            | crate::optimizeopt::info::PtrInfo::VirtualStruct(_) => {
                info.getfield(slot)?.as_opref()?
            }
            _ => return None,
        };
        // virtualstate.py get_runtime_field: the read produces a fresh InputArg* carrying
        // the concrete field value. `runtime_value_of` reads the field box's
        // own observed value (the setfield_gc source's stamped value); wrap it
        // in the matching const OpRef.
        self.const_opref_from_runtime_value(field_opref)
    }

    /// Lazy-boxing companion to `get_runtime_interiorfield`.
    ///
    /// When `runtime_box` resolves to an unmaterialized `VirtualArrayStruct`,
    /// the interior field's observed value lives on the virtual's tracked
    /// element-field box (info.py `getinteriorfield_virtual`), not a
    /// concrete struct. Read it the same way `get_virtual_runtime_field` reads a
    /// struct field. Returns `None` when the runtime box is not a virtual
    /// array-struct, the slot is unset, or the slot carries no observed value.
    fn get_virtual_runtime_interiorfield(
        &mut self,
        runtime_box: OpRef,
        descr: &majit_ir::descr::DescrRef,
        i: usize,
    ) -> Option<OpRef> {
        let ifd = descr.as_interior_field_descr()?;
        let fd = ifd.field_descr();
        // virtualstate.py:321-322 reads `opinfo._fields[i][descr.get_index()]`:
        // the parent-local field slot index (descr.rs index_in_parent), with a
        // fallback to the global index when no parent SizeDescr is bound.
        let slot = fd
            .get_parent_descr()
            .map(|_| crate::optimizeopt::virtualize::parent_list_slot(fd))
            .unwrap_or_else(|| descr.index());
        let op = self.get_box_replacement_operand_opt(runtime_box)?;
        let info = self.getptrinfo(&op)?;
        if !matches!(
            info,
            crate::optimizeopt::info::PtrInfo::VirtualArrayStruct(_)
        ) {
            return None;
        }
        let field_opref = info.getinteriorfield_virtual(i, slot)?;
        self.const_opref_from_runtime_value(field_opref)
    }

    /// virtualstate.py get_runtime_field tail: wrap the field box's observed runtime value
    /// (`runtime_value_of`, the `InputArg*.value` analog) in a fresh const OpRef
    /// matching the `InputArg{Int,Float,Ref}` the concrete `bh_getfield_gc_*`
    /// read would produce. Returns `None` for an unobserved or void slot.
    fn const_opref_from_runtime_value(&mut self, field_opref: OpRef) -> Option<OpRef> {
        match self.runtime_value_of(field_opref)? {
            Value::Int(v) => Some(self.make_constant_int(v)),
            Value::Float(v) => Some(self.make_constant_float(v)),
            Value::Ref(v) => Some(self.make_constant_ref(v)),
            Value::Void => None,
        }
    }

    /// virtualstate.py `GenerateGuardState.get_runtime_item(box, descr, i)`
    /// parity.
    ///
    /// ```python
    /// def get_runtime_item(self, box, descr, i):
    ///     array = box.getref_base()
    ///     if descr.is_array_of_pointers():
    ///         return InputArgRef(self.cpu.bh_getarrayitem_gc_r(array, i, descr))
    ///     elif descr.is_array_of_floats():
    ///         return InputArgFloat(self.cpu.bh_getarrayitem_gc_f(array, i, descr))
    ///     else:
    ///         return InputArgInt(self.cpu.bh_getarrayitem_gc_i(array, i, descr))
    /// ```
    ///
    /// Reads `array_ptr + base_size + i * itemsize` per
    /// `ArrayDescr.base_size()` / `ArrayDescr.itemsize()` matching the
    /// backend `Cpu::bh_getarrayitem_gc_*`
    /// (`majit-backend-cranelift/src/compiler.rs`). Wraps
    /// the read in a freshly allocated const OpRef.
    ///
    /// Concrete-Ref extractor routes through `runtime_value_of`; see
    /// the matching note on `get_runtime_field` for the concrete-value
    /// population path.
    pub fn get_runtime_item(
        &mut self,
        runtime_box: OpRef,
        descr: &majit_ir::descr::DescrRef,
        i: usize,
    ) -> Option<OpRef> {
        // virtualstate.py get_runtime_item `box.getref_base()` — concrete Ref read.
        // `runtime_value_of` cascades const_pool → stamped operand value
        // (RPython `InputArg*.value` analog).
        let raw = match self.runtime_value_of(runtime_box)? {
            Value::Ref(gcref) if !gcref.is_null() => gcref.0 as i64,
            _ => return None,
        };
        let ad = descr.as_array_descr()?;
        let base_size = ad.base_size() as i64;
        let itemsize = ad.item_size() as i64;
        let offset = base_size + (i as i64) * itemsize;
        let ptr = (raw as usize).wrapping_add(offset as usize);
        match ad.item_type() {
            Type::Ref => {
                let val = unsafe { (ptr as *const usize).read_unaligned() };
                Some(self.make_constant_ref(majit_ir::GcRef(val)))
            }
            Type::Float => {
                let val = unsafe { (ptr as *const f64).read_unaligned() };
                Some(self.make_constant_float(val))
            }
            Type::Int => {
                let size = itemsize as usize;
                let sign = ad.is_item_signed();
                let val = unsafe {
                    match (size, sign) {
                        (1, true) => (ptr as *const i8).read_unaligned() as i64,
                        (1, false) => (ptr as *const u8).read_unaligned() as i64,
                        (2, true) => (ptr as *const i16).read_unaligned() as i64,
                        (2, false) => (ptr as *const u16).read_unaligned() as i64,
                        (4, true) => (ptr as *const i32).read_unaligned() as i64,
                        (4, false) => (ptr as *const u32).read_unaligned() as i64,
                        _ => (ptr as *const i64).read_unaligned(),
                    }
                };
                Some(self.make_constant_int(val))
            }
            _ => None,
        }
    }

    /// virtualstate.py `GenerateGuardState.get_runtime_interiorfield(box, descr, i)`
    /// parity.
    ///
    /// ```python
    /// def get_runtime_interiorfield(self, box, descr, i):
    ///     struct = box.getref_base()
    ///     if descr.is_pointer_field():
    ///         return InputArgRef(self.cpu.bh_getinteriorfield_gc_r(struct, i, descr))
    ///     elif descr.is_float_field():
    ///         return InputArgFloat(self.cpu.bh_getinteriorfield_gc_f(struct, i, descr))
    ///     else:
    ///         return InputArgInt(self.cpu.bh_getinteriorfield_gc_i(struct, i, descr))
    /// ```
    ///
    /// Reads at `struct_ptr + array.base_size() + i * array.item_size()
    /// + field.offset()` per `InteriorFieldDescr.array_descr()` +
    /// `field_descr()`. Matches the backend Cpu::bh_getinteriorfield_gc_*
    ///   shape (struct + element_index + interior-field).
    ///
    /// Concrete-Ref extractor routes through `runtime_value_of`; see
    /// `get_runtime_field` docstring.
    pub fn get_runtime_interiorfield(
        &mut self,
        runtime_box: OpRef,
        descr: &majit_ir::descr::DescrRef,
        i: usize,
    ) -> Option<OpRef> {
        // Lazy-boxing path: when the runtime box is an unmaterialized
        // `VirtualArrayStruct` the interior field's observed value lives on the
        // virtual's tracked element-field box rather than a concrete struct.
        // See `get_virtual_runtime_field` for the model this mirrors.
        if let Some(opref) = self.get_virtual_runtime_interiorfield(runtime_box, descr, i) {
            return Some(opref);
        }
        // virtualstate.py get_runtime_item `box.getref_base()` — concrete Ref read.
        // `runtime_value_of` cascades const_pool → stamped operand value
        // (RPython `InputArg*.value` analog).
        let raw = match self.runtime_value_of(runtime_box)? {
            Value::Ref(gcref) if !gcref.is_null() => gcref.0 as i64,
            _ => return None,
        };
        let ifd = descr.as_interior_field_descr()?;
        let ad = ifd.array_descr();
        let fd = ifd.field_descr();
        let element_offset = (ad.base_size() as i64) + (i as i64) * (ad.item_size() as i64);
        let offset = element_offset + (fd.offset() as i64);
        let ptr = (raw as usize).wrapping_add(offset as usize);
        match fd.field_type() {
            Type::Ref => {
                let val = unsafe { (ptr as *const usize).read_unaligned() };
                Some(self.make_constant_ref(majit_ir::GcRef(val)))
            }
            Type::Float => {
                let val = unsafe { (ptr as *const f64).read_unaligned() };
                Some(self.make_constant_float(val))
            }
            Type::Int => {
                let size = fd.field_size();
                let sign = fd.is_field_signed();
                let val = unsafe {
                    match (size, sign) {
                        (1, true) => (ptr as *const i8).read_unaligned() as i64,
                        (1, false) => (ptr as *const u8).read_unaligned() as i64,
                        (2, true) => (ptr as *const i16).read_unaligned() as i64,
                        (2, false) => (ptr as *const u16).read_unaligned() as i64,
                        (4, true) => (ptr as *const i32).read_unaligned() as i64,
                        (4, false) => (ptr as *const u32).read_unaligned() as i64,
                        _ => (ptr as *const i64).read_unaligned(),
                    }
                };
                Some(self.make_constant_int(val))
            }
            _ => None,
        }
    }

    /// model.py `cpu.cls_of_box(box)` parity:
    ///
    /// ```python
    /// def cls_of_box(self, box):
    ///     obj = lltype.cast_opaque_ptr(OBJECTPTR, box.getref_base())
    ///     return ConstInt(ptr2int(obj.typeptr))
    /// ```
    ///
    /// Walks the forwarding chain to its constant `Value::Ref(gcref)` payload
    /// (`box.getref_base()` parity) and dispatches `cpu.cls_of_box(raw)`
    /// through the `Cpu` trait object stored at `self.cpu`.  Falls back
    /// to the resolved box's per-type mixin slot (`RefOp._resref`,
    /// resoperation.py) when the forwarding chain has no terminal Const
    /// — live `InputArgRef` boxes with a tracer-recorded concrete value
    /// reach the typeptr via `cpu.bh_classof` on the matched gcref.
    /// Returns `None` when neither path produces a non-null gcref
    /// (`DefaultCpu::cls_of_box` reports both "no Ref" and "null gcref"
    /// as 0).
    ///
    /// Caller shape mirrors `optimizer.cpu.cls_of_box(box)` — every
    /// invocation (`info.rs`, `virtualstate.rs`, `rewrite.rs`,
    /// `bridgeopt.rs`) routes through `ctx.cls_of_box(box)`.  Future
    /// `bh_*` runtime calls will land on the same `Cpu` trait and lose
    /// the `OptContext::cls_of_box` wrapper as that surface fills out.
    pub fn cls_of_box(&self, op: &Operand) -> Option<i64> {
        // model.py `cpu.cls_of_box(box)` returns `ConstInt(ptr2int(
        // typeptr))` — the immortal vtable address as a plain integer, never
        // a traced ref. DefaultCpu walks the operand to its Const terminal
        // and dereferences the typeptr-at-offset-0. Returns 0 for
        // non-Ref / null.
        let typeptr = self.cpu.cls_of_box(op);
        if typeptr != 0 {
            return Some(typeptr);
        }
        // resoperation.py `RefOp._resref` fallback — when the
        // forwarding chain has no Const terminal, read the mixin slot
        // directly off the resolved operand. Typeptr is read with
        // `cpu.bh_classof`, the hook `cls_of_box` delegates to, so a
        // backend override still applies.
        let resolved = op.get_box_replacement(false);
        if resolved.const_value().is_some() {
            // Already had a Const terminal; cpu reported 0 because
            // non-Ref or null.  No mixin-slot fallback applies.
            return None;
        }
        let value = resolved.get_value()?;
        match value {
            Value::Ref(gcref) if !gcref.is_null() => {
                let typeptr = self.cpu.bh_classof(gcref);
                if typeptr == 0 { None } else { Some(typeptr) }
            }
            _ => None,
        }
    }

    /// info.py `getptrinfo(op).get_known_class(cpu)` parity.
    ///
    /// Delegates to `getptrinfo(&Operand)` + `PtrInfo::get_known_class` so
    /// constant pointers are handled via `cls_of_box` the same way
    /// `Instance` / `Virtual` read their stored `known_class`.
    pub fn get_known_class(&self, op: &Operand) -> Option<i64> {
        self.getptrinfo(op)?.get_known_class(self.cpu.as_ref())
    }

    /// optimizer.py `getnullness(op)` parity (line-by-line port).
    ///
    /// ```python
    /// def getnullness(self, op):
    ///     if op.type == 'r' or self.is_raw_ptr(op):
    ///         ptrinfo = getptrinfo(op)
    ///         if ptrinfo is None:
    ///             return info.INFO_UNKNOWN
    ///         return ptrinfo.getnullness()
    ///     elif op.type == 'i':
    ///         return self.getintbound(op).getnullness()
    ///     assert False
    /// ```
    ///
    /// Returns one of `INFO_NULL` / `INFO_NONNULL` / `INFO_UNKNOWN`
    /// (info.py) so callers can compare directly against the
    /// upstream constants.
    ///
    /// The `Type::Int` arm inlines `getintbound` (optimizer.py)
    /// operand-direct, preserving the lazy install of `IntBound.unbounded()`
    /// on first access via `set_forwarded_info` (interior mutability lets
    /// the method take `&self`).
    pub fn getnullness(&self, op: &Operand) -> i8 {
        use crate::optimizeopt::info::OpInfo;
        use majit_ir::forwarding::Forwarded;
        // optimizer.py: if op.type == 'r' or self.is_raw_ptr(op):
        //
        // `Box.type` is intrinsic in upstream — never Void. In pyre,
        // `materialize_operand_at` lazy-creates `Type::Void` phantom placeholders
        // for OpRefs the recorder has not yet typed; the chain walker
        // hop into the terminal Box (which carries the proper type via
        // the inline `Const` operand for Const targets) recovers the
        // RPython-intrinsic type. Read after chain walk so a phantom
        // forwarded to a typed Const still routes via the type arm.
        let resolved = op.get_box_replacement(false);
        let tp = resolved.type_();
        if matches!(tp, majit_ir::Type::Ref) || self.is_raw_ptr(op) {
            // optimizer.py:129-132 with info.py `getptrinfo` inlined.
            //
            // info.py: `r`-typed: walk the chain, synthesize
            // `ConstPtrInfo` for `ConstPtr`, else return the forwarded slot.
            // For `Type::Int` raw-pointer entries (this branch when
            // `is_raw_ptr(op)` returned true), the forwarded slot is a
            // `VirtualRaw{Buffer,Slice}` and `ptr_info()` reads it directly.
            let ptrinfo: Option<PtrInfo> = if let Some(Value::Ref(gcref)) = resolved.const_value() {
                // info.py: isinstance(op, ConstPtr): ConstPtrInfo(op)
                Some(PtrInfo::Constant(gcref))
            } else {
                // info.py:890-893: fw = op.get_forwarded(); return fw or None
                resolved.ptr_info().map(|r| r.clone())
            };
            // optimizer.py: if ptrinfo is None: INFO_UNKNOWN; else ptrinfo.getnullness()
            return match ptrinfo {
                None => INFO_UNKNOWN,
                Some(info) => info.getnullness(),
            };
        }
        // optimizer.py: elif op.type == 'i': return getintbound(op).getnullness()
        //
        // Void phantoms (untyped recorder placeholders) route through the
        // Int arm as the pyre equivalent of RPython's unknown-type tolerance
        // — the inlined `getintbound` side effect (line 110-113) installs
        // `IntBound.unbounded()` so subsequent reads agree.
        if matches!(tp, majit_ir::Type::Int | majit_ir::Type::Void) {
            // optimizer.py `getintbound` inlined operand-direct.
            // optimizer.py: op = get_box_replacement(op) — already
            // walked above (`resolved` shadows here for parity).
            // optimizer.py: if isinstance(op, ConstInt): from_constant
            if let Some(Value::Int(v)) = resolved.const_value() {
                return crate::optimizeopt::intutils::IntBound::from_constant(v).getnullness();
            }
            // optimizer.py:104-109: fw = op.get_forwarded(); branch on type.
            {
                let fw = resolved.get_forwarded();
                match &fw {
                    Forwarded::Info(OpInfo::IntBound(rc)) => return rc.borrow().getnullness(),
                    // optimizer.py:108-109: rare case (fw is RawBufferPtrInfo).
                    // optimizer.py:104-109 reads anything that is not an
                    // IntBound through the same "return unbounded" path.
                    Forwarded::Info(_) => {
                        return crate::optimizeopt::intutils::IntBound::unbounded().getnullness();
                    }
                    // `resolved` came out of `get_box_replacement(false)`,
                    // which walks `Op` / `InputArg` steps to exhaustion and
                    // materializes a `Const` target, so none of the three can
                    // still be the slot here. That holds because the slot owns
                    // its target (`forwarding.rs Forwarded::Op`); while it held
                    // a `Weak`, a dropped target ended the walk one hop early
                    // and this arm fired.
                    Forwarded::Const(_)
                    | Forwarded::SmallConst(_)
                    | Forwarded::SmallWide(_)
                    | Forwarded::Op(_)
                    | Forwarded::InputArg(_) => {
                        unreachable!("chain walker terminal")
                    }
                    Forwarded::None => {}
                }
            }
            // optimizer.py:110-113: intbound = unbounded; op.set_forwarded(intbound); return intbound
            let intbound = crate::optimizeopt::intutils::IntBound::unbounded();
            resolved.set_forwarded_info(OpInfo::int_bound(intbound.clone()));
            return intbound.getnullness();
        }
        // optimizer.py:135: assert False — Float / Void never reaches here in upstream.
        INFO_UNKNOWN
    }

    /// optimizer.py `is_raw_ptr(op)` parity (line-by-line port).
    ///
    /// ```python
    /// def is_raw_ptr(self, op):
    ///     fw = get_box_replacement(op).get_forwarded()
    ///     if isinstance(fw, info.AbstractRawPtrInfo):
    ///         return True
    ///     return False
    /// ```
    ///
    /// `AbstractRawPtrInfo` is the upstream base for `RawBufferPtrInfo`,
    /// `RawStructPtrInfo`, `RawSlicePtrInfo` (info.py). Of these:
    ///
    /// - `RawBufferPtrInfo` ↔ majit `PtrInfo::VirtualRawBuffer` (created
    ///   by `OptVirtualize` from `RAW_MALLOC_VARSIZE_CHAR`).
    /// - `RawSlicePtrInfo` ↔ majit `PtrInfo::VirtualRawSlice` (created
    ///   by `OptVirtualize::optimize_int_add` slice creator,
    ///   virtualize.py make_virtual_raw_slice).
    /// - `RawStructPtrInfo` is defined at info.py but never
    ///   instantiated anywhere in upstream (`grep -rn "RawStructPtrInfo("
    ///   rpython/jit/` returns only the class definition). It is dead
    ///   reservation code, so the absence of a majit variant is not a
    ///   parity gap.
    ///
    /// `ConstPtrInfo` is NOT a subclass of `AbstractRawPtrInfo` in
    /// upstream, so a constant raw-pointer `ConstInt` is `False` here
    /// (matches `isinstance(fw, AbstractRawPtrInfo)` returning `False`
    /// for `ConstPtrInfo`).
    pub fn is_raw_ptr(&self, op: &Operand) -> bool {
        let resolved = op.get_box_replacement(false);
        matches!(
            resolved.ptr_info().as_deref(),
            Some(PtrInfo::VirtualRawBuffer(_) | PtrInfo::VirtualRawSlice(_))
        )
    }

    /// Set PtrInfo without clearing forwarding.
    /// RPython parity: set PtrInfo at the terminal of opref's forwarding chain.
    /// In RPython, `box.set_forwarded(info)` sets info on the Box directly.
    /// `get_box_replacement(box)` then returns the terminal Box which has the info.
    /// In majit, we follow the Op chain to the terminal OpRef and set Info there.
    fn ensure_ptr_info_preserve_forwarding(&mut self, opref: OpRef, info: PtrInfo) {
        use crate::optimizeopt::info::OpInfo;
        let terminal = self.get_replacement_opref(opref);
        if terminal.is_constant() {
            return;
        }
        let b = self.materialize_operand_at(terminal);
        let already_set = !matches!(b.get_forwarded(), majit_ir::forwarding::Forwarded::None);
        if !already_set {
            b.set_forwarded_info(OpInfo::ptr(info));
        }
    }

    /// info.py `ConstPtrInfo._get_info(descr, optheap)` parity.
    ///
    /// ```python
    /// def _get_info(self, descr, optheap):
    ///     ref = self._const.getref_base()
    ///     if not ref:
    ///         raise InvalidLoop   # null protection
    ///     info = optheap.const_infos.get(ref, None)
    ///     if info is None:
    ///         info = StructPtrInfo(descr)
    ///         optheap.const_infos[ref] = info
    ///     return info
    /// ```
    ///
    /// majit's port: route through `getptrinfo` (which encapsulates the
    /// RPython `op.type` dispatch + `ConstPtrInfo` synthesis), then read
    /// `_const.getref_base()` from the resulting `PtrInfo::Constant`.
    /// Both `Value::Ref` constants and `Value::Int` constants tagged
    /// with a `Type::Ref` override hash to the same `const_infos` slot
    /// — the upstream invariant that any `ConstPtrInfo._get_info()`
    /// call on the same address returns the same shared
    /// `StructPtrInfo`.
    ///
    /// Returns `None` when `opref` is not a constant pointer at all
    /// (matching PyPy's `getrawptrinfo` returning `None` for non-pointer
    /// boxes — there's no `_get_info` to call), and also when a constant
    /// pointer resolves to a null `gcref`. In the null case it additionally
    /// records a deferred `InvalidLoop` signal on the context, mirroring
    /// `info.py`:
    ///
    /// ```python
    /// def _get_info(self, descr, optheap):
    ///     ref = self._const.getref_base()
    ///     if not ref:
    ///         raise InvalidLoop   # null protection
    /// ```
    ///
    /// The trace was constant-folding through a null base pointer, which
    /// is an impossible execution path; the driver aborts the trace at its
    /// next barrier so the JIT can retry with a different shape.
    /// Like `get_const_info_mut` but does NOT create an entry on miss.
    /// Returns `None` when:
    /// - `opref` is not a constant pointer
    /// - The constant is null
    /// - No `const_infos` entry has been created yet
    ///
    /// Used by array invalidation paths that only need to clear existing
    /// items, not install new PtrInfo variants.
    pub fn get_const_info_mut_if_exists(
        &mut self,
        opref: OpRef,
    ) -> Option<&mut crate::optimizeopt::info::PtrInfo> {
        // Resolve the position to its canonical box, then delegate to the
        // box-native form. `get_box_replacement_operand_opt` returns `None` for an
        // unresolved position (the old `and_then`/`_ => return None` arm).
        let op = self.get_box_replacement_operand_opt(opref)?;
        self.get_const_info_mut_if_exists_box(&op)
    }

    /// info.py `ConstPtrInfo._get_info(descr, optheap)` parity.
    ///
    /// `parent_descr` is the parent SizeDescr, passed so that the
    /// vacant-slot case creates `StructPtrInfo(descr)` (info.py)
    /// rather than a bare `PtrInfo::instance(None, None)`. Callers
    /// that don't have the parent descr (e.g. the field read path)
    /// extract it from the field descr via
    /// `descr.as_field_descr().get_parent_descr()`.
    pub fn get_const_info_mut(
        &mut self,
        opref: OpRef,
        parent_descr: Option<DescrRef>,
    ) -> Option<&mut crate::optimizeopt::info::PtrInfo> {
        // Resolve the position to its canonical box, then delegate to the
        // box-native form.
        let op = self.get_box_replacement_operand_opt(opref)?;
        self.get_const_info_mut_box(&op, parent_descr)
    }

    /// Box-native form of `get_const_info_mut_if_exists`: the caller already
    /// holds the (canonical) struct box, so `getptrinfo` chain-walks it
    /// directly (info.py `op = get_box_replacement(op)`). Returns `None`
    /// when the box is not a constant pointer, the constant is null, or no
    /// `const_infos` entry exists yet.
    pub fn get_const_info_mut_if_exists_box(
        &mut self,
        op: &Operand,
    ) -> Option<&mut crate::optimizeopt::info::PtrInfo> {
        let Some(gcref) = self.getptrinfo_handle(op).and_then(|h| h.constant_ref()) else {
            return None;
        };
        if gcref.is_null() {
            return None;
        }
        self.const_infos.get_mut(&gcref.0)
    }

    /// Box-native form of `get_const_info_mut` (info.py make_guards
    /// `ConstPtrInfo._get_info`): the caller holds the (canonical) struct
    /// box. Creates a `StructPtrInfo(parent_descr)` on miss; raises
    /// `InvalidLoop` on a null constant base (info.py).
    pub fn get_const_info_mut_box(
        &mut self,
        op: &Operand,
        parent_descr: Option<DescrRef>,
    ) -> Option<&mut crate::optimizeopt::info::PtrInfo> {
        use crate::optimizeopt::info::PtrInfo;
        // info.py:719: ref = self._const.getref_base()
        let Some(gcref) = self.getptrinfo_handle(op).and_then(|h| h.constant_ref()) else {
            return None;
        };
        // info.py:720-721: if not ref: raise InvalidLoop. Recorded as a
        // deferred signal (checked by the driver) and `None` returned so
        // callers take their existing "no info" path until the barrier aborts.
        if gcref.is_null() {
            self.signal_invalid_loop("ConstPtrInfo._get_info: null constant base pointer");
            return None;
        }
        let addr = gcref.0;
        // info.py:722-725: info = optheap.const_infos.get(ref, None)
        //                  if info is None: info = StructPtrInfo(descr)
        //                  optheap.const_infos[ref] = info
        Some(self.const_infos.entry(addr).or_insert_with(|| {
            // info.py: StructPtrInfo(descr)
            match parent_descr {
                Some(d) => PtrInfo::struct_ptr(d),
                None => PtrInfo::instance(None, None),
            }
        }))
    }

    /// info.py `ConstPtrInfo._get_array_info(descr, optheap)`
    /// parity:
    ///
    /// ```python
    /// def _get_array_info(self, descr, optheap):
    ///     ref = self._const.getref_base()
    ///     if not ref:
    ///         raise InvalidLoop   # null protection
    ///     info = optheap.const_infos.get(ref, None)
    ///     if info is None:
    ///         info = ArrayPtrInfo(descr)
    ///         optheap.const_infos[ref] = info
    ///     return info
    /// ```
    ///
    /// Companion to `get_const_info_mut` for the array path. Both share
    /// the same `const_infos` slot keyed by `gcref` — PyPy's invariant
    /// is that a given constant ref is used as either a struct base or
    /// an array base, never both. The Vacant entry inserts an
    /// `ArrayPtrInfo` (descr + `nonnegative` lenbound) so subsequent
    /// `setitem`/`getitem` calls land on the right variant.
    pub fn get_const_info_array_mut(
        &mut self,
        opref: OpRef,
        descr: DescrRef,
    ) -> Option<&mut crate::optimizeopt::info::PtrInfo> {
        // Resolve the position to its canonical box, then delegate to the
        // box-native form.
        let op = self.get_box_replacement_operand_opt(opref)?;
        self.get_const_info_array_mut_box(&op, descr)
    }

    /// Box-native form of `get_const_info_array_mut` (info.py _get_array_info
    /// `ConstPtrInfo._get_array_info`): the caller holds the (canonical)
    /// array box, so `getptrinfo` chain-walks it directly. Creates an
    /// `ArrayPtrInfo(descr)` with a `nonnegative` lenbound on miss; raises
    /// `InvalidLoop` on a null constant base (info.py).
    pub fn get_const_info_array_mut_box(
        &mut self,
        op: &Operand,
        descr: DescrRef,
    ) -> Option<&mut crate::optimizeopt::info::PtrInfo> {
        // info.py:729: ref = self._const.getref_base()
        let Some(gcref) = self.getptrinfo_handle(op).and_then(|h| h.constant_ref()) else {
            return None;
        };
        // info.py:730-731: if not ref: raise InvalidLoop. Deferred signal +
        // `None` return; the driver aborts at the next barrier.
        if gcref.is_null() {
            self.signal_invalid_loop("ConstPtrInfo._get_array_info: null constant base pointer");
            return None;
        }
        let addr = gcref.0;
        Some(self.const_infos.entry(addr).or_insert_with(|| {
            crate::optimizeopt::info::PtrInfo::array(
                descr,
                crate::optimizeopt::intutils::IntBound::nonnegative(),
            )
        }))
    }

    /// info.py `ConstPtrInfo.setfield` + info.py
    /// `AbstractStructPtrInfo.setfield` parity (line-by-line PyPy
    /// `structinfo.setfield(...)` routing).
    ///
    /// ```python
    /// # ConstPtrInfo
    /// def setfield(self, fielddescr, struct, op, optheap=None, cf=None):
    ///     info = self._get_info(fielddescr.get_parent_descr(), optheap)
    ///     info.setfield(fielddescr, struct, op, optheap=optheap, cf=cf)
    ///
    /// # AbstractStructPtrInfo
    /// def setfield(self, fielddescr, struct, op, optheap=None, cf=None):
    ///     self.init_fields(fielddescr.get_parent_descr(),
    ///                      fielddescr.get_index())
    ///     self._fields[fielddescr.get_index()] = op
    /// ```
    ///
    /// The Rust port routes both branches through one helper so heap.rs
    /// callers don't need to special-case the constant arg0 path. The
    /// constant case lands on `const_infos[gcref]`; the regular case
    /// runs `ensure_ptr_info_arg0(op).as_mut().setfield(...)`.
    pub fn structinfo_setfield(&mut self, op: &Op, field_idx: u32, value: &Operand) {
        let value = self.resolve_operand_operand(value);
        let arg0 = self.resolve_operand_operand(&op.arg(0));
        if arg0.const_value().is_some() {
            let parent_descr = op.with_field_descr(|fd| fd.get_parent_descr()).flatten();
            if let Some(info) = self.get_const_info_mut_box(&arg0, parent_descr.clone()) {
                if let Some(parent) = parent_descr {
                    info.init_fields(parent, field_idx as usize);
                }
                info.setfield(field_idx, value.clone());
            }
            return;
        }
        // A virtual's field list is addressed by position, and the key this
        // site carries is not one.  [`heap::OptHeap::field_slot_index`] answers
        // `typeptr` and `w_class` out of a band of their own, above every
        // position a parent's field list can hand out, precisely because they
        // resolve through no field list at all.  That is a workable key for the
        // association lists an `Instance`/`Struct` carries, but `force_box`
        // walks a virtual's `fields` back through `descr.get_all_fielddescrs()`
        // BY POSITION, so a banded slot there resolves to nothing.  A virtual
        // that owns a listed class word records it from
        // `virtualize.rs optimize_setfield_gc`, which reads the slot off the
        // layout; nothing is lost by declining here.
        let is_header_word = op
            .with_field_descr(|fd| fd.is_typeptr() || fd.is_w_class())
            .unwrap_or(false);
        // info.py AbstractStructPtrInfo.setfield: mutate `_fields`
        // in the PtrInfo object stored in the operand's `_forwarded` slot.
        // PyPy has the same single-object behavior via `box._forwarded`.
        let parent_descr = op.with_field_descr(|fd| fd.get_parent_descr()).flatten();
        self.with_ensured_ptr_info_arg0(op, |mut handle| {
            if let Some(mut pi) = handle.as_mut() {
                if is_header_word && pi.is_virtual() {
                    return;
                }
                // A header word names class identity, not a slot in a parent's
                // field list, so it contributes no layout to the receiver's
                // info either.  `ensure_ptr_info_arg0` already refuses to build
                // one from it -- its `is_w_class` arm returns the descr-less
                // `InstancePtrInfo()` that `GUARD_CLASS` builds -- and
                // `init_fields` is the one call that can put a descr back.  What
                // it would put back is the header word's `get_parent_descr()`:
                // the two-word `PyObject` shell, whose `vtable()` is 0 and whose
                // `is_object()` is therefore false.  An `InstancePtrInfo` is
                // only ever constructed from a parent that answers `is_object()`
                // (`optimizer.py ensure_ptr_info_arg0`), and
                // `InstancePtrInfo.make_guards` reads `self.descr.get_vtable()`
                // on the strength of that, so stamping a vtable-less shell there
                // publishes `GUARD_SUBCLASS[op, ConstInt(0)]` into the short
                // preamble -- a classptr the backend has no
                // `subclassrange_min/max` for.  RPython never reaches this:
                // `heaptracker.py` drops `typeptr` from the field list before
                // any descr is built, so `AbstractStructPtrInfo.setfield`'s
                // `init_fields(descr.get_parent_descr(), ...)` is only ever
                // handed an ordinary field's object parent -- the same invariant
                // `serialize_optheap` states for the bridge export when it skips
                // a cached field whose parent is not `is_object()`.  The value
                // is still cached: `heap::OptHeap::field_slot_index` answers a
                // header word out of a band of its own, and the association list
                // `setfield` writes needs no `_fields` sizing to hold it.
                if !is_header_word && let Some(parent) = parent_descr.clone() {
                    pi.init_fields(parent, field_idx as usize);
                }
                pi.setfield(field_idx, value.clone());
            }
        });
    }

    /// info.py `ConstPtrInfo.setitem` + info.py: ArrayPtrInfo
    /// `setitem` parity. Same shape as `structinfo_setfield` but routes
    /// through `_get_array_info` (`get_const_info_array_mut`) for the
    /// constant arg0 path so the const_infos slot is created as
    /// `PtrInfo::Array` rather than `PtrInfo::Instance`.
    pub fn arrayinfo_setitem(&mut self, op: &Op, index: usize, value: &Operand) {
        let value = self.resolve_operand_operand(value);
        let arg0 = self.resolve_operand_operand(&op.arg(0));
        if arg0.is_constant() || arg0.const_value().is_some() {
            if let Some(descr) = op.getdescr()
                && let Some(info) = self.get_const_info_array_mut_box(&arg0, descr)
            {
                info.setitem(index, value.clone());
            }
            return;
        }
        // info.py: ArrayPtrInfo.setitem: mutate `_items` in the PtrInfo object
        // stored in the operand's `_forwarded` slot.
        self.with_ensured_ptr_info_arg0(op, |mut handle| {
            if let Some(mut pi) = handle.as_mut() {
                pi.setitem(index, value.clone());
            }
        });
    }

    /// optimizer.py: make_nonnull(op) line-by-line port.
    ///
    /// ```python
    /// def make_nonnull(self, op):
    ///     op = self.get_box_replacement(op)
    ///     if op.is_constant():
    ///         return
    ///     if op.type == 'i':
    ///         # raw pointers
    ///         return
    ///     opinfo = op.get_forwarded()
    ///     if opinfo is not None:
    ///         assert opinfo.is_nonnull()
    ///         return
    ///     op.set_forwarded(info.NonNullPtrInfo())
    /// ```
    pub fn make_nonnull(&self, op: &Operand) {
        use crate::optimizeopt::info::OpInfo;
        // optimizer.py: op = self.get_box_replacement(op)
        let op = op.get_box_replacement(false);
        // optimizer.py:442-443: if op.is_constant(): return
        if op.is_constant() {
            return;
        }
        // optimizer.py:444-445: if op.type == 'i': return  (raw pointers)
        if matches!(op.type_(), majit_ir::Type::Int) {
            return;
        }
        // optimizer.py:446-449: opinfo = op.get_forwarded()
        //                       if opinfo is not None: ... return
        // After `get_box_replacement` walks the chain, the terminal box's
        // forwarded slot is either `Forwarded::None` or `Forwarded::Info(_)`
        // (Box variants are consumed during walk). The skip condition maps
        // directly to "Info present".
        if matches!(op.get_forwarded(), majit_ir::forwarding::Forwarded::Info(_)) {
            return;
        }
        // optimizer.py:451: op.set_forwarded(info.NonNullPtrInfo())
        op.set_forwarded_info(OpInfo::ptr(PtrInfo::NonNull { last_guard_pos: -1 }));
    }

    /// optimizer.py `ensure_ptr_info_arg0(op)` — direct line-by-line
    /// port that returns the same kind of value as PyPy.
    ///
    /// ```python
    /// def ensure_ptr_info_arg0(self, op):
    ///     from rpython.jit.metainterp.optimizeopt import vstring
    ///     arg0 = self.get_box_replacement(op.getarg(0))
    ///     if arg0.is_constant():
    ///         return info.ConstPtrInfo(arg0)
    ///     opinfo = arg0.get_forwarded()
    ///     if isinstance(opinfo, info.AbstractVirtualPtrInfo):
    ///         return opinfo
    ///     elif opinfo is not None:
    ///         last_guard_pos = opinfo.get_last_guard_pos()
    ///     else:
    ///         last_guard_pos = -1
    ///     assert opinfo is None or opinfo.__class__ is info.NonNullPtrInfo
    ///     opnum = op.opnum
    ///     if (rop.is_getfield(opnum) or opnum == rop.SETFIELD_GC or
    ///         opnum == rop.QUASIIMMUT_FIELD):
    ///         descr = op.getdescr()
    ///         parent_descr = descr.get_parent_descr()
    ///         if parent_descr.is_object():
    ///             opinfo = info.InstancePtrInfo(parent_descr)
    ///         else:
    ///             opinfo = info.StructPtrInfo(parent_descr)
    ///         opinfo.init_fields(parent_descr, descr.get_index())
    ///     elif (rop.is_getarrayitem(opnum) or opnum == rop.SETARRAYITEM_GC or
    ///           opnum == rop.ARRAYLEN_GC):
    ///         opinfo = info.ArrayPtrInfo(op.getdescr())
    ///     elif opnum in (rop.GUARD_CLASS, rop.GUARD_NONNULL_CLASS):
    ///         opinfo = info.InstancePtrInfo()
    ///     elif opnum in (rop.STRLEN,):
    ///         opinfo = vstring.StrPtrInfo(vstring.mode_string)
    ///     elif opnum in (rop.UNICODELEN,):
    ///         opinfo = vstring.StrPtrInfo(vstring.mode_unicode)
    ///     else:
    ///         assert False, "operations %s unsupported" % op
    ///     assert isinstance(opinfo, info.NonNullPtrInfo)
    ///     opinfo.last_guard_pos = last_guard_pos
    ///     arg0.set_forwarded(opinfo)
    ///     return opinfo
    /// ```
    ///
    /// Returns an [`EnsuredPtrInfo`] discriminating the constant arg0 path
    /// (`Constant(GcRef)` ↔ `info.ConstPtrInfo(arg0)`) from the regular
    /// path (`Forwarded(&mut PtrInfo)` ↔ `arg0.set_forwarded(opinfo); return
    /// opinfo`). Callers invoke methods on the return value directly,
    /// matching PyPy's `structinfo.setfield(...)` /
    /// `arrayinfo.getlenbound(...)` patterns.
    pub fn ensure_ptr_info_arg0(&mut self, op: &Op) -> EnsuredPtrInfo {
        // optimizer.py: arg0 = self.get_box_replacement(op.getarg(0))
        let arg0_box = op.arg(0).get_box_replacement(false);
        let arg0 = arg0_box.to_opref();
        // optimizer.py:465-466: if arg0.is_constant(): return info.ConstPtrInfo(arg0)
        //
        // PyPy's `info.ConstPtrInfo(arg0)` wraps the constant box itself,
        // which can be either a `ConstPtr` (Ref) or a `ConstInt` (raw
        // pointer). PyPy doesn't reject either at this point — downstream
        // code calls `_const.getref_base()` and raises `InvalidLoop` only
        // when the ref is null. The Rust port matches that permissive
        // contract: extract whatever GcRef we can (Ref → the gcref, raw
        // pointer Int → cast, anything else → null sentinel) and let the
        // downstream user decide whether to act on it.
        let arg0_const = arg0_box.const_value();
        if arg0.is_constant() || arg0_const.is_some() {
            let gcref = match arg0_const {
                Some(Value::Ref(g)) => g,
                Some(Value::Int(bits)) => majit_ir::GcRef(bits as usize),
                // Float / Void / no-constant fall back to a null sentinel —
                // PyPy's getref_base would return null and InvalidLoop guard
                // the dereference at the actual use site.
                _ => majit_ir::GcRef(0),
            };
            // info.py `ConstPtrInfo.getstrlen1(mode)`: clone the
            // resolver Arc into the EnsuredPtrInfo so subsequent
            // `getlenbound(Some(mode))` calls can ask the runtime for an
            // exact constant string length without re-borrowing self.
            let resolver = self.string_length_resolver.clone();
            return EnsuredPtrInfo::Constant {
                gcref,
                string_length_resolver: resolver,
            };
        }
        // optimizer.py:467-474:
        //     opinfo = arg0.get_forwarded()
        //     if isinstance(opinfo, info.AbstractVirtualPtrInfo):
        //         return opinfo
        //     elif opinfo is not None:
        //         last_guard_pos = opinfo.get_last_guard_pos()
        //     else:
        //         last_guard_pos = -1
        //     assert opinfo is None or opinfo.__class__ is info.NonNullPtrInfo
        //
        // The PyPy class hierarchy that drives the AbstractVirtualPtrInfo
        // early-return:
        //
        //     PtrInfo
        //       NonNullPtrInfo                       ← only this falls through
        //         AbstractVirtualPtrInfo
        //           AbstractStructPtrInfo
        //             InstancePtrInfo                ← Instance / Virtual
        //             StructPtrInfo                  ← Struct / VirtualStruct
        //           AbstractRawPtrInfo
        //             RawBufferPtrInfo               ← VirtualRawBuffer
        //             RawSlicePtrInfo                ← VirtualRawSlice
        //           ArrayPtrInfo                     ← Array / VirtualArray
        //             ArrayStructInfo                ← VirtualArrayStruct
        //         vstring.StrPtrInfo                 ← Str
        //       ConstPtrInfo                         ← Constant (handled before)
        //
        // optimizer.py `ensure_ptr_info_arg0` reads the forwarded info itself,
        // not a copy of its cached fields/items. End the immutable borrow
        // before returning the owner or installing the upgraded info.
        if matches!(
            arg0_box.ptr_info().as_deref(),
            Some(
                PtrInfo::Instance(_)
                    | PtrInfo::Virtual(_)
                    | PtrInfo::Struct(_)
                    | PtrInfo::VirtualStruct(_)
                    | PtrInfo::Array(_)
                    | PtrInfo::VirtualArray(_)
                    | PtrInfo::VirtualArrayStruct(_)
                    | PtrInfo::VirtualRawBuffer(_)
                    | PtrInfo::VirtualRawSlice(_)
                    | PtrInfo::Virtualizable(_)
                    | PtrInfo::Str(_)
            )
        ) {
            // optimizer.py:469: return opinfo. The matches! above required
            // arg0_box to carry a virtual/known PtrInfo, so the terminal
            // operand is already resolved — reuse it instead of re-minting.
            return EnsuredPtrInfo::Forwarded(arg0_box);
        }
        let last_guard_pos = if let Some(opinfo) = arg0_box.ptr_info() {
            // optimizer.py:474:
            //     assert opinfo is None or opinfo.__class__ is info.NonNullPtrInfo
            debug_assert!(
                matches!(&*opinfo, PtrInfo::NonNull { .. }),
                "ensure_ptr_info_arg0: existing non-virtual PtrInfo must be NonNullPtrInfo before upgrade, got {:?}",
                opinfo
            );
            // optimizer.py:471: last_guard_pos = opinfo.get_last_guard_pos()
            opinfo.last_guard_pos().unwrap_or(-1)
        } else {
            // optimizer.py:472-473: else: last_guard_pos = -1
            -1
        };
        // optimizer.py:475-495: dispatch on opcode to construct the right
        // PtrInfo class. The Rust port reuses PtrInfo factory constructors
        // (`PtrInfo::array`, `PtrInfo::instance`, `PtrInfo::struct_ptr`,
        // and the StrPtrInfo struct literal).
        let mut new_info = if op.opcode.is_getfield()
            || op.opcode == OpCode::SetfieldGc
            || op.opcode == OpCode::QuasiimmutField
        {
            // optimizer.py:476-484:
            //     descr = op.getdescr()
            //     parent_descr = descr.get_parent_descr()
            //     if parent_descr.is_object():
            //         opinfo = info.InstancePtrInfo(parent_descr)
            //     else:
            //         opinfo = info.StructPtrInfo(parent_descr)
            //     opinfo.init_fields(parent_descr, descr.get_index())
            let ensure_field_descr_arc = op
                .getdescr()
                .expect("ensure_ptr_info_arg0: field op without FieldDescr");
            let field_descr = ensure_field_descr_arc
                .as_field_descr()
                .expect("ensure_ptr_info_arg0: field op without FieldDescr");
            if field_descr.is_w_class() {
                // The `w_class`/typeptr header carries class identity, not a
                // value field, so it has no parent SizeDescr. arg0 is just a
                // non-null instance with no known layout — the same PtrInfo
                // GUARD_CLASS builds (optimizer.py:488-489). OptVirtualize folds
                // this read for virtuals (virtualize.rs `is_w_class`); a
                // non-virtual heap object's `__class__` read reaches here.
                PtrInfo::instance(None, None)
            } else {
                // optimizer.py:479-484: parent_descr.is_object() decides Instance vs Struct.
                let parent_descr = field_descr.get_parent_descr().unwrap_or_else(|| {
                    panic!(
                        "ensure_ptr_info_arg0: FieldDescr.get_parent_descr() returned None \
                         for opcode={:?} descr={:?} field_name={:?} index_in_parent={} \
                         offset={} field_type={:?}; the FieldDescr implementation must \
                         override get_parent_descr() for parity with optimizer.py:478",
                        op.opcode,
                        op.getdescr(),
                        field_descr.field_name(),
                        field_descr.index_in_parent(),
                        field_descr.offset(),
                        field_descr.field_type(),
                    )
                });
                let is_object = parent_descr
                    .as_size_descr()
                    .expect(
                        "ensure_ptr_info_arg0: FieldDescr.get_parent_descr() must point at a SizeDescr",
                    )
                    .is_object();
                let mut new_info = if is_object {
                    PtrInfo::instance(Some(parent_descr.clone()), None)
                } else {
                    PtrInfo::struct_ptr(parent_descr.clone())
                };
                // optimizer.py:484: opinfo.init_fields(parent_descr, descr.get_index())
                // info.py init_fields(parent_descr, index) sets self.descr
                // and pre-allocates _fields by parent slot count.
                new_info.init_fields(
                    parent_descr,
                    crate::optimizeopt::virtualize::parent_list_slot(field_descr) as usize,
                );
                new_info
            }
        } else if op.opcode.is_getarrayitem()
            || op.opcode == OpCode::SetarrayitemGc
            || op.opcode == OpCode::ArraylenGc
        {
            // optimizer.py:485-487: getarrayitem / setarrayitem_gc / arraylen_gc
            // → ArrayPtrInfo(op.getdescr())
            let descr = op
                .getdescr()
                .expect("ensure_ptr_info_arg0: array op without descr");
            PtrInfo::array(descr, crate::optimizeopt::intutils::IntBound::nonnegative())
        } else if op.opcode == OpCode::GuardClass || op.opcode == OpCode::GuardNonnullClass {
            // optimizer.py:488-489: guard_class / guard_nonnull_class
            // → InstancePtrInfo()
            PtrInfo::instance(None, None)
        } else if op.opcode == OpCode::Strlen {
            // optimizer.py:490-491: strlen → StrPtrInfo(mode_string)
            PtrInfo::Str(crate::optimizeopt::info::StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: -1,
                variant: crate::optimizeopt::info::VStringVariant::Ptr,
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            })
        } else if op.opcode == OpCode::Unicodelen {
            // optimizer.py:492-493: unicodelen → StrPtrInfo(mode_unicode)
            PtrInfo::Str(crate::optimizeopt::info::StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 1,
                length: -1,
                variant: crate::optimizeopt::info::VStringVariant::Ptr,
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            })
        } else {
            // optimizer.py:494-495: assert False, "operations %s unsupported"
            panic!("ensure_ptr_info_arg0: opcode {:?} unsupported", op.opcode);
        };
        // optimizer.py:496: assert isinstance(opinfo, info.NonNullPtrInfo)
        // — every constructed PtrInfo above is a NonNullPtrInfo subclass.
        // optimizer.py:497: opinfo.last_guard_pos = last_guard_pos
        new_info.set_last_guard_pos(last_guard_pos);
        // optimizer.py:498: arg0.set_forwarded(opinfo)
        // `arg0` is already `get_box_replacement`. `materialize_write_host`
        // reseats onto the producer at `to_opref()`, which is a different box
        // once the replacement is a forwarded target, so `CachedField._getfield`
        // (`get_box_replacement` + `getptrinfo`) misses the field. Write on the
        // replacement when it is a real box. An unbound operand has no slot
        // (`Operand::with_forwarding_host`); only then mint the host.
        use crate::optimizeopt::info::OpInfo;
        let arg0_box = if arg0_box.bound_op().is_some() || arg0_box.bound_inputarg().is_some() {
            arg0_box
        } else {
            self.materialize_write_host(arg0_box)
        };
        arg0_box.set_forwarded_info(OpInfo::ptr(new_info));
        // optimizer.py:499: return opinfo — hand back the operand so
        // subsequent mutations land on the authoritative slot.
        EnsuredPtrInfo::Forwarded(arg0_box)
    }

    /// optimizer.py: make_nonnull_str(op, mode) line-by-line port.
    ///
    /// ```python
    /// def make_nonnull_str(self, op, mode):
    ///     from rpython.jit.metainterp.optimizeopt import vstring
    ///     op = self.get_box_replacement(op)
    ///     if op.is_constant():
    ///         return
    ///     opinfo = op.get_forwarded()
    ///     if isinstance(opinfo, vstring.StrPtrInfo):
    ///         return
    ///     op.set_forwarded(vstring.StrPtrInfo(mode))
    /// ```
    pub fn make_nonnull_str(&self, op: &Operand, mode: u8) {
        use crate::optimizeopt::info::OpInfo;
        // optimizer.py: op = self.get_box_replacement(op)
        let op = op.get_box_replacement(false);
        // optimizer.py:457: if op.is_constant(): return
        if op.is_constant() {
            return;
        }
        // optimizer.py:459-460: opinfo = op.get_forwarded();
        //                       if isinstance(opinfo, vstring.StrPtrInfo): return
        if matches!(op.ptr_info().as_deref(), Some(PtrInfo::Str(_))) {
            return;
        }
        // optimizer.py:462: op.set_forwarded(vstring.StrPtrInfo(mode))
        op.set_forwarded_info(OpInfo::ptr(PtrInfo::Str(
            crate::optimizeopt::info::StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode,
                length: -1,
                variant: crate::optimizeopt::info::VStringVariant::Ptr,
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            },
        )));
    }

    /// rewrite.py:434-435: isinstance(old_guard_op.getdescr(),
    /// compile.ResumeAtPositionDescr).
    /// guard_pos is a _newoperations index (info.py get_last_guard).
    pub fn is_resume_at_position_guard(&self, guard_pos: i32) -> bool {
        if self.new_operations_drained || guard_pos < 0 {
            return false;
        }
        self.new_operations
            .get(guard_pos as usize)
            .and_then(|op| op.getdescr())
            .is_some_and(|descr| descr.is_resume_at_position())
    }

    /// Trace ConstPtr indexes this context owns while it is still in place.
    ///
    /// `trace_index` is idempotent across overlapping holders (`new_operations`
    /// and `resop_refs` share ops). `const_infos` keys are raw addresses;
    /// the values hold `_fields` and must still be walked (`info.py`
    /// `optheap.const_infos`).
    fn walk_live_const_ptrs(&mut self, visitor: &mut dyn FnMut(&mut GcRef)) {
        for op in &self.new_operations {
            walk_live_op(op, visitor);
        }
        for op in &self.input_ops {
            walk_live_op(op, visitor);
        }
        for op in &self.live_synthetics {
            walk_live_op(op, visitor);
        }
        for op in self.resop_refs.values() {
            walk_live_op(op, visitor);
        }
        for op in &self.phase1_emit_ops {
            walk_live_op(op, visitor);
        }
        for (_, op) in self.extra_operations_after.iter() {
            walk_live_op(op, visitor);
        }
        for queue in &self.extra_pending {
            for (_, op) in queue.iter() {
                walk_live_op(op, visitor);
            }
        }
        if let Some(op) = &self.patchguardop {
            walk_live_op(op, visitor);
        }
        if let Some(op) = &self.pending_finish_guard_postprocess {
            walk_live_op(op, visitor);
        }
        for r in &self.inputargs {
            r.trace_const_ptr(visitor);
        }
        // `resoperation.py` `RefOp._resref` is a GC field of the box. The
        // canonical InputArg boxes live in `inputarg_refs`; a bridge's are
        // reminted copies no recorder walk visits, so forward `.value` here.
        for ia in self.inputarg_refs.values() {
            if let Some(Value::Ref(mut r)) = ia.get_value() {
                visitor(&mut r);
                ia.set_value(Value::Ref(r));
            }
        }
        for entry in &self.exported_const_short_boxes {
            walk_live_op(&entry.op, visitor);
            if let Some(source) = &entry.source_op {
                walk_live_op(source, visitor);
            }
            entry.res.walk_const_ptr_refs(visitor);
            if let Some(same) = &entry.same_as_source {
                same.walk_const_ptr_refs(visitor);
            }
        }
        for pop in self.potential_extra_ops.values() {
            pop.op.walk_const_ptr_refs(visitor);
            walk_live_op(&pop.preamble_op, visitor);
            if let Some(same) = &pop.same_as_source {
                same.walk_const_ptr_refs(visitor);
            }
        }
        if let Some(preview) = self.preview_short_state.as_mut() {
            for entry in &preview.short_boxes {
                walk_live_op(&entry.op, visitor);
                if let Some(source) = &entry.source_op {
                    walk_live_op(source, visitor);
                }
                entry.res.walk_const_ptr_refs(visitor);
                if let Some(same) = &entry.same_as_source {
                    same.walk_const_ptr_refs(visitor);
                }
            }
            for r in &preview.short_inputargs {
                r.trace_const_ptr(visitor);
            }
            if let Some(state) = preview.args_state.as_mut() {
                state.0.walk_const_ptr_refs_mut(visitor);
                for r in &state.1 {
                    r.trace_const_ptr(visitor);
                }
                for r in &state.2 {
                    r.trace_const_ptr(visitor);
                }
                for operand in &state.4 {
                    operand.walk_const_ptr_refs(visitor);
                }
            }
        }
        for info in self.const_infos.values_mut() {
            info.walk_const_ptr_refs_mut(visitor);
        }
    }

    /// Drain the emitted-operation buffer and invalidate every PtrInfo guard
    /// position that named the old buffer.
    pub(crate) fn take_new_operations(&mut self) -> Vec<majit_ir::OpRc> {
        self.new_operations_drained = true;
        std::mem::take(&mut self.new_operations)
    }

    /// Take ownership of PtrInfo, replacing with None.
    /// Used by force_box to mutate info in-place (RPython parity).
    pub fn take_ptr_info(&self, op: &Operand) -> Option<PtrInfo> {
        use crate::optimizeopt::info::OpInfo;
        use majit_ir::forwarding::Forwarded;
        let resolved = op.get_box_replacement(false);
        // Read terminal's `_forwarded` slot; clone the PtrInfo (if any),
        // drop the Ref borrow, then clear the slot via interior
        // mutability. Const targets are no-op-cleared by
        // `clear_forwarded` per AbstractValue invariant.
        let info = {
            let fw = resolved.get_forwarded();
            match &fw {
                Forwarded::Info(OpInfo::Ptr(rc)) => Some(rc.borrow().clone()),
                _ => None,
            }
        };
        if info.is_some() {
            resolved.clear_forwarded();
        }
        info
    }

    pub fn set_ptr_info(&self, op: &Operand, info: PtrInfo) {
        use crate::optimizeopt::info::OpInfo;
        // Walk chain and write through the terminal slot. Const targets
        // (whose chain walker landed on an inline `Const`) silently
        // no-op via `set_forwarded_info`'s upstream invariant — Const has
        // no _forwarded slot so any write would assert.
        let resolved = op.get_box_replacement(false);
        if resolved.is_constant() {
            return;
        }
        resolved.set_forwarded_info(OpInfo::ptr(info));
    }

    /// optimizer.py: replace_op_with(old, new_op, ctx)
    /// Replace old opref AND emit the new op.
    pub fn replace_op_with(&mut self, old: OpRef, new_op: Op) -> OpRef {
        let new_ref = self.emit(new_op);
        let b_old = self.get_box_replacement_operand(old);
        let b_new = self.get_box_replacement_operand(new_ref);
        self.make_equal_to(&b_old, &b_new);
        new_ref
    }
}

#[cfg(test)]
pub(crate) fn seed_guard_snapshots_with<F>(
    ops: &[Op],
    mut snapshot_for_guard: F,
) -> (Vec<Op>, SnapshotBoxes)
where
    F: FnMut(&Op) -> Vec<OpRef>,
{
    let mut seeded = ops.to_vec();
    let mut snapshots: SnapshotBoxes = Vec::new();
    let mut next_resume_pos = 0i32;
    for op in seeded.iter_mut().filter(|op| op.opcode.is_guard()) {
        let snapshot_boxes = snapshot_for_guard(op);
        let resume_pos = if op.rd_resume_position() >= 0
            && !snapshot_contains(&snapshots, op.rd_resume_position())
        {
            op.rd_resume_position()
        } else {
            while snapshot_contains(&snapshots, next_resume_pos) {
                next_resume_pos += 1;
            }
            let resume_pos = next_resume_pos;
            next_resume_pos += 1;
            resume_pos
        };
        op.set_rd_resume_position(resume_pos);
        snapshot_insert(
            &mut snapshots,
            resume_pos,
            snapshot_boxes.into_iter().map(SnapshotBox::from).collect(),
        );
    }
    (seeded, snapshots)
}

/// Test fixture helper for optimizer tests whose guard resume state is
/// intentionally irrelevant.  RPython would still have a
/// `capture_resumedata()` entry for such a guard; the explicit empty list
/// models an empty active frame snapshot rather than deriving anything from
/// `guard.fail_args`.
#[cfg(test)]
pub(crate) fn seed_empty_guard_snapshots(ops: &[Op]) -> (Vec<Op>, SnapshotBoxes) {
    seed_guard_snapshots_with(ops, |_| Vec::new())
}

#[cfg(test)]
mod opt_context_pos_tests {
    use super::*;
    use majit_ir::OpRef;
    use majit_ir::resoperation::{Op, OpCode};
    use std::rc::Rc;

    /// `make_constant` stores an i32 ConstInt as `Forwarded::SmallConst`.
    /// `allocate_next_pos_raw` must treat that as a claimed position.
    #[test]
    fn reserve_pos_skips_a_smallconst_forwarded_position() {
        use majit_ir::forwarding::ForwardingHost;
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 0, 0, 0);
        let pos = OpRef::int_op(5);
        let op = Op::new(OpCode::SameAsI, &[]);
        op.pos().set(pos);
        op.set_forwarded_const(majit_ir::Const::from_value(majit_ir::Value::Int(7)));
        assert!(
            matches!(
                op.forwarded().borrow(),
                majit_ir::forwarding::Forwarded::SmallConst(_)
            ),
            "i32 ConstInt must pack as SmallConst"
        );
        ctx.resop_refs.insert(pos, OpRc::new(op));
        ctx.next_pos = 5;
        let reserved = ctx.reserve_pos_typed(majit_ir::Type::Int);
        assert_eq!(
            reserved.raw(),
            6,
            "SmallConst-forwarded position 5 must stay claimed"
        );
    }

    #[test]
    fn take_preamble_forwarded_opinfo_reads_smallconst() {
        use majit_ir::forwarding::ForwardingHost;
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 0, 0, 0);
        let pos = OpRef::int_op(5);
        let op = Op::new(OpCode::SameAsI, &[]);
        op.pos().set(pos);
        op.set_forwarded_const(majit_ir::Const::from_value(majit_ir::Value::Int(7)));
        ctx.resop_refs.insert(pos, OpRc::new(op));
        let info = ctx
            .take_preamble_forwarded_opinfo(pos)
            .expect("SmallConst must produce preamble IntBound info");
        match info {
            crate::optimizeopt::info::OpInfo::IntBound(ib) => {
                assert_eq!(ib.borrow().lower, 7);
                assert_eq!(ib.borrow().upper, 7);
            }
            other => panic!("expected IntBound, got {other:?}"),
        }
        assert!(
            matches!(
                ctx.read_forwarded(pos),
                None | Some(majit_ir::forwarding::Forwarded::None)
            ),
            "consumed SmallConst forwarding must be cleared"
        );
    }
}

#[cfg(test)]
mod getstrlen_force_tests {
    use super::*;
    use crate::optimizeopt::info::{PtrInfo, StrPtrInfo, VStringVariant};
    use majit_ir::{OpCode, OpRef};

    /// vstring.py `StrPtrInfo.getstrlen`: `optstring.emit_extra(lengthop)`
    /// during force (`in_final_emission`) lands in `Optimizer.emit`
    /// immediately, so the STRLEN precedes NEWSTR. `emit_extra_rc` would
    /// queue it after the NEWSTR/COPYSTRCONTENT that consume it.
    #[test]
    fn getstrlen_during_force_emits_strlen_immediately() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(10, 0, 0, 50);
        ctx.in_final_emission = true;
        let arg = OpRef::ref_op(1);
        let arg_box = ctx.materialize_operand_at(arg);
        ctx.set_ptr_info(
            &arg_box,
            PtrInfo::Str(StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: -1,
                variant: VStringVariant::Ptr,
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );
        let _ = ctx.getstrlen_opref(arg, 0);
        assert!(
            ctx.extra_operations_after.is_empty(),
            "force-time getstrlen must emit STRLEN immediately"
        );
        assert!(
            ctx.new_operations
                .iter()
                .any(|op| op.opcode == OpCode::Strlen),
            "STRLEN must land in new_operations during force"
        );
    }
}

#[cfg(test)]
mod boxref_forwarding_tests {
    //! Operand `_forwarded` invariants: the four writers (`set_ptr_info`,
    //! `setintbound`, `make_constant`, `make_equal_to`) install PyPy-style
    //! forwarding state on the authoritative forwarding slot.
    use super::*;
    use crate::history::test_support::{bound_inputarg_operand, bound_resop_operand};
    use crate::optimizeopt::info::{OpInfo, PtrInfo};
    use crate::optimizeopt::intutils::IntBound;
    use majit_ir::forwarding::Forwarded as BoxForwarded;
    use majit_ir::{InputArgRc, OpRef, Type, Value};

    fn ctx_with_two_int_boxes() -> (OptContext, Operand, Operand, Vec<InputArgRc>) {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 2, 0, 2);
        let (b0, ia0) = bound_inputarg_operand(Type::Int, 0);
        let (b1, ia1) = bound_inputarg_operand(Type::Int, 1);
        ctx.seed_boxes_canonical(&[b0.clone(), b1.clone()]);
        (ctx, b0, b1, vec![ia0, ia1])
    }

    /// `make_equal_to(old, new)` plants an `InputArg`-target chain step on
    /// `old`'s `_forwarded` slot (`optimizer.py op.set_forwarded(newop)`
    /// — `newop` is an `AbstractInputArg` here), and `get_box_replacement`
    /// (`resoperation.py`) walks to an operand bound to `new`'s
    /// `AbstractInputArg` identity. The walker materialises a transient
    /// operand wrapping the same `InputArgRc`, so identity is checked via
    /// the bound handle, not outer `Rc<Box>` pointer equality.
    #[test]
    fn h3_1_replace_op_mirrors_box_forward() {
        let (mut ctx, b0, b1, ia_holder) = ctx_with_two_int_boxes();
        ctx.make_equal_to(&b0, &b1);
        assert!(matches!(b0.get_forwarded(), BoxForwarded::InputArg(_)));
        let walked = b0.get_box_replacement(false);
        assert!(majit_ir::InputArgRc::ptr_eq(
            &walked
                .bound_inputarg()
                .expect("walked terminal carries bound InputArg"),
            &ia_holder[1],
        ));
    }

    #[test]
    fn replacement_operand_walks_to_the_forwarded_terminal() {
        let (ctx, b0, _b1, _ia_holder) = ctx_with_two_int_boxes();
        let foreign = InputArgRc::new(majit_ir::InputArg::from_type(Type::Int, 1));
        b0.set_forwarded_inputarg(&foreign);

        let resolved = ctx.get_box_replacement_operand(OpRef::input_arg_typed(0, Type::Int));
        assert!(majit_ir::InputArgRc::ptr_eq(
            &resolved
                .bound_inputarg()
                .expect("walked terminal carries bound InputArg"),
            &foreign,
        ));
    }

    /// Forward-reference dup-materialization regression: a
    /// consumer that binds its operand to a position's stand-in BEFORE the
    /// producer at that position is emitted must, after the producer emits,
    /// resolve through to the emitted producer — not freeze on the orphaned
    /// stand-in. This is the `test_guard_true_int_lt_enables_add_ovf_removal`
    /// shape reduced to OptContext primitives: GUARD_TRUE (consumer) is
    /// dispatched and arg-bound to the pos-2 stand-in before INT_LT (producer)
    /// emits its `new_operations` clone. Without the `emit` `set_forwarded_op`
    /// link the box-native walk of the consumer arg terminates on the stand-in
    /// while the OpRef path resolves the producer, tripping `resolve_operand_operand`'s
    /// divergence witness (panic under `cfg(debug_assertions)`).
    #[test]
    fn s2_forward_reference_consumer_follows_emitted_producer() {
        use majit_ir::resoperation::{Op, OpCode};
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 2, 0, 2);
        let (b0, _ia0) = bound_inputarg_operand(Type::Int, 0);
        let (b1, _ia1) = bound_inputarg_operand(Type::Int, 1);
        ctx.seed_boxes_canonical(&[b0.clone(), b1.clone()]);

        // pos 2 = an INT_LT result not yet emitted; mint the stand-in the
        // recorder / `bind_input_resops` seeds into resop_refs.
        let pos2 = OpRef::int_op(2);
        let _standin = ctx.mint_synthetic_resop(pos2, Type::Int);

        // A consumer (GUARD_TRUE) dispatched before the producer binds its
        // operand to the stand-in — the forward reference.
        let consumer_arg = ctx.materialize_operand_at(pos2);

        // The producer emits at pos 2.
        let mut producer = Op::new(OpCode::IntLt, &[b0.clone(), b1.clone()]);
        producer.pos().set(pos2);
        ctx.emit(producer);

        // The frozen consumer arg must resolve to the emitted producer; the
        // call also exercises resolve_operand_operand's native walk, which
        // would diverge without the emit rebind.
        let resolved = ctx.resolve_operand_operand(&consumer_arg);
        let producer_b = ctx
            .find_producer_op(pos2)
            .expect("producer emitted at pos 2");
        assert!(
            resolved
                .bound_op()
                .is_some_and(|op| OpRc::ptr_eq(&op, &producer_b)),
            "consumer must resolve to the emitted producer, not the orphaned stand-in"
        );
    }

    /// `box.clear_forwarded()` resets a previously-set forwarding slot
    /// back to `Forwarded::None`.  PyPy has no `make_equal_to(op, None)`
    /// path; chain reset happens on the box directly.
    #[test]
    fn h3_1_clear_forwarded_resets_box_forward() {
        let (mut ctx, b0, b1, _ia_holder) = ctx_with_two_int_boxes();
        ctx.make_equal_to(&b0, &b1);
        b0.clear_forwarded();
        assert!(matches!(b0.get_forwarded(), BoxForwarded::None));
    }

    /// `optimizer.py make_equal_to` Info transfer parity: when
    /// `old` carries `Forwarded::IntBound(_)` and is forwarded to a
    /// non-constant `new`, the IntBound moves to `new`'s slot.
    #[test]
    fn h3_1_replace_op_transfers_int_bound_to_new() {
        let (mut ctx, b0, b1, ia_holder) = ctx_with_two_int_boxes();
        let bound = IntBound::from_constant(7);
        ctx.setintbound(&b0, &bound);
        ctx.make_equal_to(&b0, &b1);
        // After: old's IntBound transferred to new (PyPy:
        // `newop.set_forwarded(opinfo)`). old now forwards to new.
        match &b1.get_forwarded() {
            BoxForwarded::Info(OpInfo::IntBound(b)) => assert_eq!(b.borrow().lower, 7),
            other => panic!("operand[1] should carry IntBound, got {:?}", other),
        }
        // old's slot now points to new. Bound-InputArg target routes through
        // `set_forwarded_inputarg`, so the slot carries
        // `Forwarded::InputArg(Rc<InputArg>)`; chain walk lands on a
        // transient operand sharing `ia_holder[1]`'s identity.
        assert!(matches!(b0.get_forwarded(), BoxForwarded::InputArg(_)));
        let walked = b0.get_box_replacement(false);
        assert!(majit_ir::InputArgRc::ptr_eq(
            &walked
                .bound_inputarg()
                .expect("walked terminal carries bound InputArg"),
            &ia_holder[1],
        ));
    }

    /// `optimizer.py:400` guard: transfer is **skipped** when `new` is
    /// constant. PyPy short-circuits via `not newop.is_constant()`.
    #[test]
    fn h3_1_replace_op_skips_info_transfer_when_new_is_constant() {
        let (mut ctx, b0, _b1, _ia_holder) = ctx_with_two_int_boxes();
        // Seed an IntBound on old.
        let bound = IntBound::from_constant(42);
        ctx.setintbound(&b0, &bound);
        // Forward to an inline-Const target — history.py ConstInt.value
        // carries the value on the Box itself, no const_pool seed needed.
        let const_opref = OpRef::const_int(42);
        let b_const = ctx.materialize_operand_at(const_opref);
        ctx.make_equal_to(&b0, &b_const);
        // The IntBound on old is gone (overwritten by Forwarded::Op(const)).
        // Const targets do not carry transferred info — PyPy skips this case.
        match &b0.get_forwarded() {
            other if other.const_value() == Some(majit_ir::Value::Int(42)) => {}
            other => panic!("expected b0 to forward to Const, got {:?}", other),
        }
    }

    /// `set_ptr_info(opref, info)` mirrors `box.set_forwarded(PtrInfo)`.
    #[test]
    fn h3_1_set_ptr_info_mirrors_box_info() {
        // PtrInfo applies to ref-typed boxes.
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 1, 0, 1);
        let (b, _ia) = bound_inputarg_operand(Type::Ref, 0);
        ctx.seed_boxes_canonical(std::slice::from_ref(&b));
        let info = PtrInfo::NonNull { last_guard_pos: -1 };
        ctx.set_ptr_info(&b, info);
        match &b.get_forwarded() {
            BoxForwarded::Info(OpInfo::Ptr(rc))
                if matches!(&*rc.borrow(), PtrInfo::NonNull { .. }) => {}
            other => panic!("expected Info(Ptr(NonNull)), got {:?}", other),
        }
    }

    /// PyPy optimizer.py:432 parity: after
    /// `make_constant_arg(&box, Value::Ref(_))` writes the constant onto
    /// the InputArg's `_forwarded` slot, a subsequent `make_nonnull(opref)`
    /// MUST NOT overwrite the Const with `OpInfo::Ptr(NonNull)`.
    #[test]
    fn audit_a_make_nonnull_preserves_box_constant_slot() {
        use majit_ir::GcRef;
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 1, 0, 1);
        let (b, _ia) = bound_inputarg_operand(Type::Ref, 0);
        ctx.seed_boxes_canonical(std::slice::from_ref(&b));
        ctx.make_constant_arg(&b, Value::Ref(GcRef(0xdead_beef)));
        assert_eq!(
            b.get_forwarded().const_value(),
            Some(majit_ir::Value::Ref(GcRef(0xdead_beef))),
            "expected const-ref terminal post make_constant, got {:?}",
            b.get_forwarded()
        );
        // Use the operand form because `make_nonnull` writes to the box's
        // forwarded slot.
        ctx.make_nonnull(&b);
        assert_eq!(
            b.get_forwarded().const_value(),
            Some(majit_ir::Value::Ref(GcRef(0xdead_beef))),
            "make_nonnull must not overwrite the Const slot — got {:?}",
            b.get_forwarded()
        );
    }

    /// `resoperation.py get_box_replacement` + `history.py:188
    /// Const.is_constant()` parity: after the chain walker advances into
    /// a `Forwarded::Const(constval)` target, `is_constant()` on the
    /// terminal box reports True. Covers both encodings of "this slot is
    /// a known constant": (a) Const-namespace OpRef terminus, and (b)
    /// `Forwarded::Const(constval)` produced by `optimizer.py
    /// set_forwarded(constbox)` — equivalent to RPython's single
    /// `is_constant()` predicate after `get_box_replacement`.
    #[test]
    fn audit_a_chain_walker_reaches_constant_through_forwarded_const() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 2, 0, 2);
        let (b0, _ia0) = bound_inputarg_operand(Type::Int, 0);
        let (b1, _ia1) = bound_inputarg_operand(Type::Int, 1);
        ctx.seed_boxes_canonical(&[b0.clone(), b1.clone()]);
        // (a) Const-namespace OpRef terminates at a Const box.
        let const_opref = OpRef::const_int(7);
        let const_box = ctx.materialize_operand_at(const_opref);
        assert!(const_box.get_box_replacement(false).is_constant());
        // (b) `Forwarded::Const(constval)` chain on a non-Const-namespace OpRef.
        let b0_iarg = ctx.materialize_operand_at(OpRef::input_arg_int(0));
        ctx.make_equal_to(&b0_iarg, &const_box);
        let b0_after = ctx.materialize_operand_at(OpRef::input_arg_int(0));
        assert!(b0_after.get_box_replacement(false).is_constant());
        // `Forwarded::Const(constval)` planted directly via set_forwarded_const.
        b1.set_forwarded_const(majit_ir::Const::Int(42));
        let first = b1.get_box_replacement(false);
        let second = b1.get_box_replacement(false);
        assert!(first.is_constant());
        assert_eq!(
            first, second,
            "get_box_replacement must return the identity stored in _forwarded"
        );
        // Negative case: operand with no constant forwarding.
        let (nb, _ia_nb) = bound_inputarg_operand(Type::Int, 0);
        assert!(!nb.get_box_replacement(false).is_constant());
    }

    /// A stack-resident red InputArg must stay a TAGBOX. Folding it to
    /// `ConstPtr` drops it from resume liveboxes; the next `Vm::new` then
    /// has no InputArg to rebind.
    #[test]
    fn stack_resident_inputarg_is_not_folded_to_const() {
        use majit_ir::GcRef;
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 1, 0, 1);
        let (b0, ia0) = bound_inputarg_operand(Type::Ref, 0);
        let probe = 0usize;
        let stack_addr = std::ptr::addr_of!(probe) as usize;
        ia0.set_value(Value::Ref(GcRef(stack_addr)));
        ctx.seed_boxes_canonical(&[b0.clone()]);
        ctx.make_constant_box(&b0, Value::Ref(GcRef(stack_addr)));
        assert!(
            !b0.get_box_replacement(false).is_constant(),
            "stack-resident InputArg must not become ConstPtr"
        );
        let heap = GcRef(0x0000_0001_0000_0000);
        let (b1, ia1) = bound_inputarg_operand(Type::Ref, 1);
        ia1.set_value(Value::Ref(heap));
        ctx.seed_boxes_canonical(&[b1.clone()]);
        ctx.make_constant_box(&b1, Value::Ref(heap));
        assert!(
            b1.get_box_replacement(false).is_constant(),
            "heap InputArg stays foldable (PyPy GC identity)"
        );
    }

    /// `make_constant` mirrors PyPy optimizer.py
    /// `box.set_forwarded(constbox)` — Const variant.
    #[test]
    fn h3_1_make_constant_mirrors_box_info_constant() {
        let (mut ctx, b0, _b1, _ia_holder) = ctx_with_two_int_boxes();
        ctx.make_constant_arg(&b0, Value::Int(42));
        match &b0.get_forwarded() {
            other if other.const_value() == Some(majit_ir::Value::Int(42)) => {}
            other => panic!("expected Forwarded::Const(Int 42), got {:?}", other),
        }
    }

    /// `setintbound(opref, bound)` mirrors `box.set_forwarded(IntBound)`.
    #[test]
    fn h3_1_setintbound_mirrors_box_info() {
        let (mut ctx, b0, _b1, _ia_holder) = ctx_with_two_int_boxes();
        let bound = IntBound::from_constant(7);
        ctx.setintbound(&b0, &bound);
        match &b0.get_forwarded() {
            BoxForwarded::Info(OpInfo::IntBound(b)) => {
                let b = b.borrow();
                assert_eq!(b.lower, 7);
                assert_eq!(b.upper, 7);
            }
            other => panic!("expected Info(IntBound), got {:?}", other),
        }
    }

    /// `make_equal_to(old, ConstX)` mirrors onto `old_box.set_forwarded_const(
    /// const_value)`. Per RPython parity (`optimizer.py`,
    /// `history.py` ConstInt construction), the const target is built
    /// fresh from `const_pool[const_index]` per call site — no dedup, value
    /// equality via `same_constant`. The mirror must record the same Value
    /// as the seeded constant via `Forwarded::Const`.
    #[test]
    fn h3_4_replace_op_const_target_mirrors_value_box() {
        let (mut ctx, b0, _b1, _ia_holder) = ctx_with_two_int_boxes();
        let const_opref = OpRef::const_int(42);
        let b_const = ctx.materialize_operand_at(const_opref);
        ctx.make_equal_to(&b0, &b_const);
        match &b0.get_forwarded() {
            other if other.const_value() == Some(majit_ir::Value::Int(42)) => {}
            other => panic!("expected Forwarded::Const(Int 42), got {:?}", other),
        }
    }

    /// resoperation.py get_box_replacement(not_const=True) stops before
    /// stepping into a Const target. This is required for guard fail args:
    /// resume numbering encodes constants as TAGCONST, while backend liveboxes
    /// keep the runtime Box identity.
    #[test]
    fn get_box_replacement_not_const_stops_before_const_target() {
        let (mut ctx, b0, b1, _ia_holder) = ctx_with_two_int_boxes();
        let const_opref = OpRef::const_int(42);
        let b_const = ctx.materialize_operand_at(const_opref);
        ctx.make_equal_to(&b0, &b_const);

        assert_eq!(
            ctx.get_box_replacement_operand(OpRef::input_arg_typed(0, Type::Int))
                .to_opref(),
            const_opref
        );
        assert_eq!(
            ctx.get_box_replacement_impl(OpRef::input_arg_typed(0, Type::Int), true),
            OpRef::input_arg_typed(0, Type::Int)
        );

        ctx.make_equal_to(&b1, &b0);
        assert_eq!(
            ctx.get_box_replacement_operand(OpRef::input_arg_typed(1, Type::Int))
                .to_opref(),
            const_opref
        );
        assert_eq!(
            ctx.get_box_replacement_impl(OpRef::input_arg_typed(1, Type::Int), true),
            OpRef::input_arg_typed(0, Type::Int)
        );
    }

    /// Body optimization's inherited emit-position region
    /// `[0..phase2_inputarg_base)` carries placeholder resop hosts (preamble
    /// emit ops do NOT appear in the body trace iteration, so the body iter
    /// has no `cls()` allocation for them). Replicates the import_state
    /// pattern in `unroll.rs`'s `import_state`:
    ///
    ///   1. `make_equal_to(source_p2, target_p1)` writes
    ///      `source._forwarded = Box(placeholder_at_target_p1.raw)`.
    ///   2. Body import writes info via `set_ptr_info(target_p1, info)`,
    ///      setting `placeholder._forwarded = Info(info)`.
    ///   3. Reading source via `peek_ptr_info` walks
    ///      `source → placeholder` and sees the placeholder's info.
    ///
    /// PyPy parity is preserved structurally even though the preamble's actual
    /// Box is not shared with the body context: the placeholder absorbs body
    /// import writes the same way the preamble Box would in PyPy.
    #[test]
    fn h3_4_phase2_placeholder_forwarding_yields_consistent_reads() {
        // Layout: indices 0..2 are preamble emit-position placeholders,
        // indices 2..4 are body inputarg operands. PyPy `box.type`
        // invariant prevents `make_equal_to(Ref, Void)` (cross-type forward),
        // so place Ref-typed boxes on both sides — the test models a
        // preamble RefOp result acting as the import target.
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 4, 0, 4);
        let (placeholder_target, _op_target) = bound_resop_operand(Type::Ref, 0);
        let (placeholder_other, _op_other) = bound_resop_operand(Type::Ref, 1);
        let (source_box, _ia_source) = bound_inputarg_operand(Type::Ref, 2);
        let (other_box, _ia_other) = bound_inputarg_operand(Type::Ref, 3);
        ctx.seed_boxes_canonical(&[
            placeholder_target.clone(),
            placeholder_other.clone(),
            source_box.clone(),
            other_box.clone(),
        ]);

        // operand-first chain walker reconstructs the variant tag from
        // `box.type_()`; placeholders and source are both Ref, so use the
        // typed factories that match.
        let target_p1 = OpRef::ref_op(0);
        let source_p2 = OpRef::input_arg_ref(2);

        // Step 1: import_state's `source.set_forwarded(target)` equivalent.
        ctx.make_equal_to(&source_box, &placeholder_target);

        // Step 2: setinfo_from_preamble's terminal write.
        // `setinfo_from_preamble(source, info)` first walks the chain via
        // `get_box_replacement` which returns `target_p1`,
        // then calls `set_ptr_info(target_p1, info)`. Replicate the
        // post-walk write directly.
        let info = PtrInfo::NonNull { last_guard_pos: -1 };
        let target_p1_box = ctx.materialize_operand_at(target_p1);
        ctx.set_ptr_info(&target_p1_box, info.clone());

        // Read via operand path: walk source's chain to placeholder.
        let source_p2_box = ctx.get_box_replacement_operand(source_p2);
        let via_box = ctx
            .peek_ptr_info(&source_p2_box)
            .expect("operand path must see info");
        assert!(matches!(via_box, PtrInfo::NonNull { .. }));

        // Chain walk lands on target_p1.
        let resolved = ctx.get_replacement_opref(source_p2);
        assert_eq!(resolved, target_p1);

        // Placeholder Box absorbed the mirror write, so its _forwarded now
        // carries the info — equivalent to PyPy's preamble Box receiving
        // setinfo_from_preamble.
        match &placeholder_target.get_forwarded() {
            BoxForwarded::Info(OpInfo::Ptr(rc))
                if matches!(&*rc.borrow(), PtrInfo::NonNull { .. }) => {}
            other => panic!(
                "placeholder must carry Info(NonNull) after set_ptr_info, got {:?}",
                other
            ),
        }
    }

    /// Complementary to
    /// `h3_4_phase2_placeholder_forwarding_yields_consistent_reads`. Pre-import
    /// (no `setinfo_from_preamble` call), reading `target_p1` info via either
    /// path returns None — consistent within pyre. PyPy parity here depends on
    /// `ExportedState.exported_infos` (`unroll.py` canonical field)
    /// carrying every preamble op info the body needs; the placeholder cannot
    /// fabricate preamble info that wasn't exported. PyPy itself uses the same
    /// serialization map for the import (PyPy's body optimizer reads exported_infos
    /// → setinfo_from_preamble too), so structural narrowness here matches
    /// PyPy's own dispatch.
    #[test]
    fn h3_4_phase2_placeholder_without_import_returns_none_consistently() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 4, 0, 4);
        // Same Ref-typed alignment as the sibling test: forwarding a Ref
        // source to the placeholder requires placeholder type to match
        // (PyPy `box.type` invariant; `make_equal_to` cross-type assertion).
        let (placeholder_target, _op_target) = bound_resop_operand(Type::Ref, 0);
        let (placeholder_other, _op_other) = bound_resop_operand(Type::Ref, 1);
        let (source_box, _ia_source) = bound_inputarg_operand(Type::Ref, 2);
        let (other_box, _ia_other) = bound_inputarg_operand(Type::Ref, 3);
        ctx.seed_boxes_canonical(&[
            placeholder_target.clone(),
            placeholder_other.clone(),
            source_box.clone(),
            other_box.clone(),
        ]);

        let target_p1 = OpRef::ref_op(0);
        let source_p2 = OpRef::input_arg_ref(2);

        // import_state's make_equal_to fires, but the body import chose NOT to import
        // info (e.g. exported_infos didn't carry an entry for target_p1).
        ctx.make_equal_to(&source_box, &placeholder_target);

        // operand-path reader: chain walks source → placeholder → None.
        let source_p2_box = ctx.get_box_replacement_operand(source_p2);
        assert!(ctx.peek_ptr_info(&source_p2_box).is_none());

        // Legacy Vec reader: chain walks source → target_p1 → None
        // (the body's fresh Vec has no entry for target_p1).
        let resolved = ctx.get_replacement_opref(source_p2);
        assert_eq!(resolved, target_p1);

        // Placeholder Box was not mutated (no info import fired) — still None.
        assert!(matches!(
            placeholder_target.get_forwarded(),
            BoxForwarded::None
        ));
    }

    /// With the canonical slot seeded and no forwarding, the
    /// operand-returning reader resolves the slot's bound InputArg.
    /// `resoperation.py:57-68` walker terminates on `None` immediately.
    #[test]
    fn h3_2b_get_box_replacement_operand_opt_returns_pool_entry_when_no_forward() {
        let (ctx, _b0, _b1, ia_holder) = ctx_with_two_int_boxes();
        let got = ctx
            .get_box_replacement_operand_opt(OpRef::input_arg_typed(0, Type::Int))
            .expect("canonical store resolves the slot");
        // No forwarding: the resolver materialises a fresh terminal operand
        // bound to the same `InputArgRc` as the seeded slot.
        assert!(majit_ir::InputArgRc::ptr_eq(
            &got.bound_inputarg()
                .expect("resolved terminal carries bound InputArg"),
            &ia_holder[0],
        ));
    }

    /// With a forwarding chain installed via `make_equal_to`, the
    /// operand walker reaches the terminal Box (`b1`). RPython parity:
    /// `optimizer.py box.set_forwarded(newop)` → reader walks until
    /// `Forwarded::None` and returns the last Box. The walker materialises
    /// a transient operand wrapping `b1`'s bound `InputArgRc`, so terminal
    /// identity is checked via the shared `InputArg` handle rather than
    /// outer `Rc<Box>` pointer equality.
    #[test]
    fn h3_2b_get_box_replacement_operand_opt_walks_forwarded_chain() {
        let (mut ctx, b0, b1, ia_holder) = ctx_with_two_int_boxes();
        ctx.make_equal_to(&b0, &b1);
        let got = ctx
            .get_box_replacement_operand_opt(OpRef::input_arg_typed(0, Type::Int))
            .expect("bound box resolves");
        assert!(majit_ir::InputArgRc::ptr_eq(
            &got.bound_inputarg()
                .expect("walked terminal carries bound InputArg"),
            &ia_holder[1],
        ));
        // b0 itself is not the terminal.
        assert_ne!(got.to_opref(), b0.to_opref());
    }

    /// With no seeded canonical stores (test/retrace baseline) the
    /// operand-returning reader returns `None`; the OpRef-returning walker
    /// cannot resolve a Box identity without a bound producer either.
    #[test]
    fn h3_2b_get_box_replacement_operand_opt_returns_none_when_pool_empty() {
        let ctx = OptContext::with_num_inputs_and_start_pos(0, 2, 0, 2);
        // No seeded producer: no Box identity to resolve.
        assert!(
            ctx.get_box_replacement_operand_opt(OpRef::int_op(0))
                .is_none()
        );
    }

    /// `OpRef::NONE` sentinel returns `None` — the operand reader
    /// has no Box to root the walk on. The OpRef-returning reader handles
    /// the sentinel independently by returning it unchanged.
    #[test]
    fn h3_2b_get_box_replacement_operand_opt_handles_none_sentinel() {
        let (ctx, _b0, _b1, _ia_holder) = ctx_with_two_int_boxes();
        assert!(ctx.get_box_replacement_operand_opt(OpRef::NONE).is_none());
    }

    /// When the chain terminates at `Forwarded::Info(_)`, the
    /// walker returns the Box that holds the Info — the forwarding-chain
    /// walker stops before descending into Info, matching
    /// PyPy `resoperation.py:60 isinstance(next, AbstractInfo)`.
    #[test]
    fn h3_2b_get_box_replacement_operand_opt_stops_at_info_terminal() {
        let (mut ctx, b0, _b1, ia_holder) = ctx_with_two_int_boxes();
        ctx.setintbound(&b0, &IntBound::from_constant(7));
        let got = ctx
            .get_box_replacement_operand_opt(OpRef::input_arg_typed(0, Type::Int))
            .expect("canonical store resolves the slot");
        // Walker terminates at the slot (its `_forwarded` is Info, not a
        // chain step); the resolved operand shares b0's bound InputArg.
        assert!(majit_ir::InputArgRc::ptr_eq(
            &got.bound_inputarg()
                .expect("resolved terminal carries bound InputArg"),
            &ia_holder[0],
        ));
    }

    // operand-path helpers `is_virtual` / `is_nonnull` read the same
    // `_forwarded` slot that PyPy's getptrinfo() inspects.

    fn ctx_with_one_ref_box() -> (OptContext, Operand) {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 1, 0, 1);
        let (b, ia) = bound_inputarg_operand(Type::Ref, 0);
        ctx.seed_boxes_canonical(std::slice::from_ref(&b));
        // Register the InputArgRc as the slot's canonical host so the
        // context's readers resolve to it across the test body.
        ctx.inputarg_refs = std::iter::once((ia.index, ia)).collect();
        (ctx, b)
    }

    #[derive(Debug)]
    struct DummySizeDescr;
    impl majit_ir::Descr for DummySizeDescr {}

    #[test]
    fn h3_2c_is_virtual_matches_legacy_when_pool_plumbed() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        let info = PtrInfo::Virtual(crate::optimizeopt::info::VirtualInfo {
            descr: std::sync::Arc::new(DummySizeDescr),
            known_class: None,
            ob_type_descr: None,
            fields: Default::default(),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });
        ctx.set_ptr_info(&b, info);
        assert!(ctx.peek_ptr_info(&b).is_some_and(|i| i.is_virtual()));
        assert!(ctx.is_virtual(&b));
    }

    #[test]
    fn h3_2c_is_virtual_returns_false_for_nonnull_only() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(&b, PtrInfo::NonNull { last_guard_pos: -1 });
        assert!(!ctx.is_virtual(&b));
    }

    #[test]
    fn h3_2c_is_virtual_returns_false_for_unset() {
        let (ctx, b) = ctx_with_one_ref_box();
        assert!(!ctx.is_virtual(&b));
    }

    #[test]
    fn h3_2c_is_nonnull_matches_set_info() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(&b, PtrInfo::NonNull { last_guard_pos: -1 });
        assert!(ctx.peek_ptr_info(&b).is_some_and(|i| i.is_nonnull()));
        assert!(ctx.is_nonnull(&b));
    }

    #[test]
    fn h3_2c_is_nonnull_returns_false_for_unset() {
        let (ctx, b) = ctx_with_one_ref_box();
        assert!(!ctx.is_nonnull(&b));
    }

    #[test]
    fn h3_2c_peek_intbound_box_matches_legacy_when_pool_plumbed() {
        let (mut ctx, b0, _b1, _ia_holder) = ctx_with_two_int_boxes();
        ctx.setintbound(&b0, &IntBound::from_constant(42));
        let legacy = ctx
            .peek_intbound(OpRef::input_arg_int(0))
            .expect("legacy bound");
        let via_box = ctx.peek_intbound_box(&b0).expect("box bound");
        assert!(legacy.is_constant());
        assert_eq!(legacy.get_constant_int(), 42);
        assert!(via_box.is_constant());
        assert_eq!(via_box.get_constant_int(), 42);
    }

    #[test]
    fn h3_2c_peek_intbound_box_returns_none_for_unset() {
        let (ctx, b0, _b1, _ia_holder) = ctx_with_two_int_boxes();
        assert!(ctx.peek_intbound_box(&b0).is_none());
    }

    #[test]
    fn h3_2c_last_guard_pos_matches_legacy_when_pool_plumbed() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(&b, PtrInfo::NonNull { last_guard_pos: 5 });
        assert_eq!(ctx.last_guard_pos(&b), Some(5));
        assert_eq!(
            ctx.peek_ptr_info(&b).and_then(|i| i.get_last_guard_pos()),
            Some(5)
        );
    }

    #[test]
    fn drained_operations_invalidate_recorded_guard_positions() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        let guard = OpRc::new(Op::new(OpCode::GuardNonnull, std::slice::from_ref(&b)));
        ctx.push_new_operation(guard);
        ctx.set_ptr_info(&b, PtrInfo::NonNull { last_guard_pos: 0 });
        assert!(ctx.get_last_guard(&b).is_some());

        let _old_operations = ctx.take_new_operations();
        ctx.push_new_operation(OpRc::new(Op::new(OpCode::IntAdd, &[])));

        assert!(ctx.get_last_guard(&b).is_none());
        assert!(ctx.last_guard_pos(&b).is_none());
        assert!(!ctx.is_resume_at_position_guard(0));
    }

    #[test]
    fn h3_2c_last_guard_pos_returns_none_for_unset() {
        let (ctx, b) = ctx_with_one_ref_box();
        assert!(ctx.last_guard_pos(&b).is_none());
    }

    #[test]
    fn h3_2c_last_guard_pos_returns_none_when_no_recorded_guard() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        // info.py last_guard_pos == -1 → get_last_guard_pos returns None.
        ctx.set_ptr_info(&b, PtrInfo::NonNull { last_guard_pos: -1 });
        assert!(ctx.last_guard_pos(&b).is_none());
    }

    #[test]
    fn h3_2c_is_virtualizable_via_box_matches_legacy_when_pool_plumbed() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(
            &b,
            PtrInfo::Virtualizable(crate::optimizeopt::info::VirtualizableFieldState {
                fields: Vec::new(),
                field_descrs: Vec::new(),
                arrays: Vec::new(),
                heap_fields: Vec::new(),
                last_guard_pos: -1,
            }),
        );
        assert!(ctx.is_virtualizable(&b));
        assert!(matches!(
            ctx.peek_ptr_info(&b),
            Some(PtrInfo::Virtualizable(_))
        ));
    }

    #[test]
    fn h3_2c_is_virtualizable_returns_false_for_nonnull_only() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(&b, PtrInfo::NonNull { last_guard_pos: -1 });
        assert!(!ctx.is_virtualizable(&b));
    }

    #[test]
    fn h3_2c_is_virtualizable_returns_false_for_unset() {
        let (ctx, b) = ctx_with_one_ref_box();
        assert!(!ctx.is_virtualizable(&b));
    }

    #[test]
    fn h3_2c_has_ptr_info_matches_set_info() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(&b, PtrInfo::NonNull { last_guard_pos: -1 });
        assert!(ctx.has_ptr_info(&b));
        assert!(ctx.peek_ptr_info(&b).is_some());
    }

    #[test]
    fn h3_2c_has_ptr_info_returns_false_for_unset() {
        let (ctx, b) = ctx_with_one_ref_box();
        assert!(!ctx.has_ptr_info(&b));
    }

    #[test]
    fn h3_2c_peek_ptr_info_returns_set_info() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(&b, PtrInfo::NonNull { last_guard_pos: 5 });
        let via_box = ctx.peek_ptr_info(&b).expect("box clone");
        assert!(matches!(via_box, PtrInfo::NonNull { last_guard_pos: 5 }));
    }

    #[test]
    fn h3_2c_peek_ptr_info_returns_none_for_unset() {
        let (ctx, b) = ctx_with_one_ref_box();
        assert!(ctx.peek_ptr_info(&b).is_none());
    }

    // `with_ptr_info_mut(box, |info| ...)` runs a closure against the
    // `&mut PtrInfo` stored on `box._forwarded::Info` so subsequent
    // operand-path readers (`peek_ptr_info`, `last_guard_pos`) see
    // the mutation.

    #[test]
    fn h3_2c_with_ptr_info_mut_mirrors_after_mutation_when_pool_plumbed() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(&b, PtrInfo::NonNull { last_guard_pos: 0 });
        // Pre-condition: operand snapshot matches legacy at pos 0.
        assert_eq!(ctx.last_guard_pos(&b), Some(0));
        // Mutate inner state via closure.
        let returned = ctx
            .with_ptr_info_mut(&b, |info| {
                info.set_last_guard_pos(42);
                "ok"
            })
            .expect("closure runs");
        assert_eq!(returned, "ok");
        // Post-condition: operand snapshot reflects mutation (mirror ran).
        assert_eq!(ctx.last_guard_pos(&b), Some(42));
        assert_eq!(
            ctx.peek_ptr_info(&b).and_then(|i| i.get_last_guard_pos()),
            Some(42)
        );
    }

    #[test]
    fn h3_2c_with_ptr_info_mut_returns_none_for_unset() {
        let (ctx, b) = ctx_with_one_ref_box();
        // No PtrInfo installed at OpRef(0).
        let invoked = std::cell::Cell::new(false);
        let result = ctx.with_ptr_info_mut(&b, |_info| {
            invoked.set(true);
        });
        assert!(result.is_none());
        assert!(!invoked.get(), "closure must not run when info is absent");
    }

    /// `PtrInfoHandle::Live` preserves RPython `_forwarded` object
    /// identity: two handles cloned from the same `_forwarded` cell
    /// satisfy `same_info` and observe each other's in-place
    /// mutations — Python `is`-equivalent semantics for non-ConstPtrInfo.
    #[test]
    fn ptr_info_handle_live_identity_propagates_mutation() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(&b, PtrInfo::NonNull { last_guard_pos: 0 });
        let h1 = ctx
            .getptrinfo_handle(&b)
            .expect("Live handle for installed PtrInfo");
        let h2 = ctx
            .getptrinfo_handle(&b)
            .expect("second call must return another clone of the same cell");
        assert!(
            h1.same_info(&h2),
            "two handles into the same _forwarded cell must satisfy same_info"
        );
        // Mutation through h1 visible through h2 — RPython
        // `opinfo._known_class = ...` propagation.
        {
            let mut m = h1.borrow_mut().expect("Live handle borrows mutably");
            m.set_last_guard_pos(99);
        }
        assert_eq!(
            h2.borrow().get_last_guard_pos(),
            Some(99),
            "h2 must observe h1's mutation (shared Rc cell)"
        );
    }

    /// `ConstPtrInfo.same_info` is value-based: RPython overrides the
    /// base identity check and compares `_const.same_constant(other._const)`.
    #[test]
    fn ptr_info_handle_const_arms_use_constptr_same_info() {
        use majit_ir::GcRef;
        let h1 = PtrInfoHandle::Const(PtrInfo::Constant(GcRef(0x1000)));
        let h2 = PtrInfoHandle::Const(PtrInfo::Constant(GcRef(0x1000)));
        let h3 = PtrInfoHandle::Const(PtrInfo::Constant(GcRef(0x2000)));
        assert!(
            h1.same_info(&h2),
            "two ConstPtrInfo handles for the same const must be same_info"
        );
        assert!(!h1.same_info(&h3), "different constants are not same_info");
    }

    /// The ConstPtrInfo override applies even when one side is a live
    /// `_forwarded` cell carrying the ConstPtrInfo-equivalent payload.
    #[test]
    fn ptr_info_handle_const_and_live_constant_use_constptr_same_info() {
        use crate::optimizeopt::info::OpInfo;
        use majit_ir::{GcRef, Type};

        let (b, _op) = bound_resop_operand(Type::Ref, 0);
        b.set_forwarded_info(OpInfo::ptr(PtrInfo::Constant(GcRef(0x1000))));
        let live = PtrInfoHandle::Live(
            b.ptr_info_handle()
                .expect("live forwarded ConstPtrInfo-equivalent handle"),
        );
        let same_const = PtrInfoHandle::Const(PtrInfo::Constant(GcRef(0x1000)));
        let different_const = PtrInfoHandle::Const(PtrInfo::Constant(GcRef(0x2000)));

        assert!(same_const.same_info(&live));
        assert!(live.same_info(&same_const));
        assert!(!different_const.same_info(&live));
    }

    /// `getintbound_handle` lazy-installs `IntBound::unbounded` on first
    /// access (mirroring `optimizer.py:110-112 op.set_forwarded(IntBound())`)
    /// and subsequent calls return the same `Live` cell — `Rc::ptr_eq`
    /// holds and mutation through one handle propagates to the other.
    #[test]
    fn int_bound_handle_live_identity_propagates_mutation() {
        use majit_ir::Type;

        let mut ctx = OptContext::with_num_inputs(0, 0);
        let (b, _op) = bound_resop_operand(Type::Int, 0);
        let h1 = ctx.getintbound_handle(&b);
        let h2 = ctx.getintbound_handle(&b);
        assert!(
            h1.ptr_eq(&h2),
            "Live handles for the same box must share the same Rc cell"
        );

        // Mutation through h1 visible through h2 — RPython
        // `getintbound(box).intersect(b)` propagation.
        {
            let mut m = h1.borrow_mut().expect("Live handle accepts mutable borrow");
            let _ = m.make_ge_const(42);
        }
        assert_eq!(
            h2.borrow().lower,
            42,
            "h2 must observe h1's make_ge mutation (shared Rc cell)"
        );
    }

    /// `IntBoundHandle::Const` arms (synthesized from `ConstInt`) are
    /// independent objects — fresh `IntBound::from_constant(v)` each
    /// time, never `Rc::ptr_eq`.
    #[test]
    fn int_bound_handle_const_arms_are_not_ptr_eq() {
        use majit_ir::Value;

        let mut ctx = OptContext::with_num_inputs(0, 0);
        let c = Operand::const_from_value(Value::Int(7));
        let h1 = ctx.getintbound_handle(&c);
        let h2 = ctx.getintbound_handle(&c);
        assert!(
            !h1.ptr_eq(&h2),
            "Const arms must be fresh independent objects, never ptr_eq"
        );
        assert_eq!(h1.borrow().lower, 7);
        assert_eq!(h2.borrow().lower, 7);
    }

    /// `optimizer.py:102-103 return IntBound.from_constant(...)` —
    /// PyPy yields a *mutable* fresh `IntBound` for ConstInt. The Rust
    /// Const arm mirrors that: `borrow_mut()` succeeds and the mutation
    /// is observable through the same handle (and any clones of it),
    /// while a SEPARATE `getintbound_handle` call on the same ConstInt
    /// produces an independent cell that does NOT see the mutation.
    #[test]
    fn int_bound_handle_const_arm_is_locally_mutable() {
        use majit_ir::Value;

        let mut ctx = OptContext::with_num_inputs(0, 0);
        let c = Operand::const_from_value(Value::Int(7));
        let h = ctx.getintbound_handle(&c);
        // Direct field mutation through the RefMut — `make_ge_const`
        // would reject 20 on `from_constant(7)` (empty interval); the
        // parity claim is "borrow_mut succeeds and writes land in the
        // cell", not "any arbitrary IntBound method succeeds".
        {
            let mut m = h
                .borrow_mut()
                .expect("Const arm must accept mutable borrow (optimizer.py:102)");
            m.upper = 20;
        }
        assert_eq!(
            h.borrow().upper,
            20,
            "Const arm mutation must be visible through the same handle"
        );
        // A fresh getintbound_handle call mints an independent cell —
        // mutations on `h` do not leak across calls (PyPy: each
        // `IntBound.from_constant(7)` is a distinct object).
        let h_fresh = ctx.getintbound_handle(&c);
        assert_eq!(
            h_fresh.borrow().upper,
            7,
            "Fresh Const handle must not observe prior handle's mutation"
        );
    }

    /// `resoperation.py get_box_replacement` walks the
    /// `_forwarded` chain until it hits a terminal that is not a Box
    /// forward.  After two consecutive `make_equal_to(a, b)` /
    /// `make_equal_to(b, c)` calls, reading `getptrinfo_handle(&a)`
    /// must return a handle to the same Rc cell that
    /// `getptrinfo_handle(&c)` returns — the chain walker resolves
    /// `a → b → c` and the PtrInfo installed earliest on `a` has
    /// transferred through both steps via the OpInfo clone.
    #[test]
    fn chain_walk_preserves_ptr_info_rc_identity_across_two_hops() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 3, 0, 3);
        let (a, _ia_a) = bound_inputarg_operand(Type::Ref, 0);
        let (b, _ia_b) = bound_inputarg_operand(Type::Ref, 1);
        let (c, _ia_c) = bound_inputarg_operand(Type::Ref, 2);
        ctx.seed_boxes_canonical(&[a.clone(), b.clone(), c.clone()]);
        ctx.set_ptr_info(&a, PtrInfo::NonNull { last_guard_pos: 7 });

        ctx.make_equal_to(&a, &b);
        ctx.make_equal_to(&b, &c);

        let h_a = ctx
            .getptrinfo_handle(&a)
            .expect("chain a -> b -> c must surface c's _forwarded slot");
        let h_c = ctx
            .getptrinfo_handle(&c)
            .expect("c carries the transferred PtrInfo");
        assert!(
            h_a.same_info(&h_c),
            "chain walker must land on the same Rc cell that lives on c"
        );

        // Verify the original last_guard_pos survived both transfers.
        assert_eq!(h_c.borrow().get_last_guard_pos(), Some(7));
    }

    /// `optimizer.py make_equal_to` transfers the `_forwarded`
    /// IntBound from `op` to `newop` by writing the same Python object
    /// into `newop.set_forwarded(...)`.  Counterpart of
    /// [`replace_op_preserves_ptr_info_rc_identity`] for the IntBound
    /// cell.
    #[test]
    fn replace_op_preserves_int_bound_rc_identity() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 2, 0, 2);
        let (old_box, _ia_old) = bound_inputarg_operand(Type::Int, 0);
        let (new_box, _ia_new) = bound_inputarg_operand(Type::Int, 1);
        ctx.seed_boxes_canonical(&[old_box.clone(), new_box.clone()]);
        ctx.setintbound(
            &old_box,
            &crate::optimizeopt::intutils::IntBound::unbounded(),
        );

        let old_handle = ctx.getintbound_handle(&old_box);
        assert!(matches!(old_handle, IntBoundHandle::Live(_)));

        ctx.make_equal_to(&old_box, &new_box);
        let new_handle = ctx.getintbound_handle(&new_box);
        assert!(
            old_handle.ptr_eq(&new_handle),
            "make_equal_to must transfer the same Rc cell for IntBound"
        );

        // Mutation through new_handle visible through old_handle.
        let _ = new_handle.borrow_mut().unwrap().make_ge_const(99);
        assert_eq!(
            old_handle.borrow().lower,
            99,
            "old must observe new's make_ge mutation (shared Rc)"
        );
    }

    /// `optimizer.py make_equal_to` transfers the `_forwarded`
    /// PtrInfo from `op` to `newop` by writing the same Python object
    /// into `newop.set_forwarded(...)`.  pyre's `make_equal_to` clones
    /// the `OpInfo` enum, but since `OpInfo::Ptr` holds an `Rc`, the
    /// clone shares the same cell — so after `make_equal_to(old, new)`
    /// the handles obtained from `old` and `new` satisfy `ptr_eq`
    /// and downstream mutation on one is visible through the other.
    #[test]
    fn replace_op_preserves_ptr_info_rc_identity() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 2, 0, 2);
        let (old_box, _ia_old) = bound_inputarg_operand(Type::Ref, 0);
        let (new_box, _ia_new) = bound_inputarg_operand(Type::Ref, 1);
        ctx.seed_boxes_canonical(&[old_box.clone(), new_box.clone()]);
        ctx.set_ptr_info(&old_box, PtrInfo::NonNull { last_guard_pos: 0 });

        let old_handle = ctx
            .getptrinfo_handle(&old_box)
            .expect("install populated _forwarded on old");
        ctx.make_equal_to(&old_box, &new_box);
        let new_handle = ctx
            .getptrinfo_handle(&new_box)
            .expect("PtrInfo transferred to new via clone of Rc cell");
        assert!(
            old_handle.same_info(&new_handle),
            "make_equal_to must transfer the same Rc cell (RPython _forwarded share)"
        );

        // Mutation through new_handle visible through old_handle —
        // they share the same Rc<RefCell<PtrInfo>>.
        new_handle
            .borrow_mut()
            .expect("Live handle accepts mutation")
            .set_last_guard_pos(123);
        assert_eq!(
            old_handle.borrow().get_last_guard_pos(),
            Some(123),
            "old's view of the transferred info must see new's mutation"
        );
    }

    /// `box.clear_forwarded()` resets `_forwarded` directly.  After the
    /// call, `old`'s slot is `None` and any previously-stored IntBound is
    /// unreachable.
    #[test]
    fn clear_forwarded_drops_int_bound() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 1, 0, 1);
        let (old_box, ia) = bound_inputarg_operand(Type::Int, 0);
        ctx.seed_boxes_canonical(std::slice::from_ref(&old_box));
        ctx.inputarg_refs = std::iter::once((ia.index, ia)).collect();
        ctx.setintbound(
            &old_box,
            &crate::optimizeopt::intutils::IntBound::unbounded(),
        );
        assert!(matches!(
            ctx.getintbound_handle(&old_box),
            IntBoundHandle::Live(_),
        ));

        old_box.clear_forwarded();
        assert!(matches!(
            &old_box.get_forwarded(),
            majit_ir::forwarding::Forwarded::None,
        ));
    }

    /// For an emitted ResOp the producing `Op.type_`,
    /// the minted OpRef variant tag, and `opref_type` all derive from the
    /// single source `opcode.result_type()` (resoperation.py:1693
    /// `res.type = result_type` in `create_class_for_op`). This regresses
    /// if a producer mints a variant tag disagreeing with `op.type_`, or if
    /// `opref_type` stops reading the variant tag / producing op — i.e. if
    /// any retired type side-table (`value_types` / `prev_phase_value_types`
    /// / `renamed_inputarg_types`) were reintroduced as a competing source.
    #[test]
    fn slice_0_8_emitted_op_type_is_single_source_of_truth() {
        use majit_ir::{Op, OpCode};
        let mut ctx = OptContext::with_num_inputs(8, 0);
        let cases: &[(OpCode, Type)] = &[
            (OpCode::SameAsI, Type::Int),
            (OpCode::SameAsR, Type::Ref),
            (OpCode::SameAsF, Type::Float),
        ];
        for &(opcode, ty) in cases {
            let op = Op::new(opcode, &[]);
            // `Op::new` seeds `type_` from `opcode.result_type()`.
            assert_eq!(op.type_, ty, "Op.type_ must equal opcode.result_type()");
            let pos: OpRef = ctx.emit(op);
            // `emit` reserves a typed pos, so the variant tag encodes the type.
            assert_eq!(
                pos.ty(),
                Some(ty),
                "minted OpRef variant tag must encode the op result type"
            );
            // The unified reader agrees, proving it reads the variant tag /
            // producing op.type_ and not any retired side-table.
            assert_eq!(
                ctx.opref_type(pos),
                Some(ty),
                "opref_type must agree with the variant tag / op.type_"
            );
            // The producing op recovered by position carries the same type_.
            let producer = ctx.op_at(pos).expect("emitted op must be findable by pos");
            assert_eq!(
                producer.type_, ty,
                "producing op.type_ must match the variant tag / opref_type"
            );
        }
    }

    /// A peeled body names its input args by the recorder slot, while
    /// `OptUnroll.import_state` (`source.set_forwarded(target)`) writes
    /// that iteration's value onto the host at `inputarg_base + slot`.
    /// RPython writes onto the one box the body operations hold; the flat
    /// `OpRef` namespace splits it in two, so reading the recorder slot on
    /// its own loses the import and a slot the preamble proved constant
    /// reaches the body as an unknown.
    #[test]
    fn a_peeled_body_reads_the_slot_the_import_forwarded() {
        use majit_ir::operand::Operand;
        const BASE: u32 = 71;
        const SLOT: u32 = 5;
        const NUM_INPUTS: usize = 11;
        let mut ctx = OptContext::with_num_inputs_and_start_pos(
            8,
            NUM_INPUTS,
            BASE,
            BASE + NUM_INPUTS as u32,
        );
        let recorder = majit_ir::InputArg::from_type_rc(Type::Ref, SLOT);
        let host = majit_ir::InputArg::from_type_rc(Type::Ref, BASE + SLOT);
        let host_arg = Operand::from_bound_inputarg(&host);
        host_arg.set_forwarded_const(majit_ir::Const::from_gcref(majit_ir::GcRef(0)));
        ctx.register_carried_host(&Operand::from_bound_inputarg(&recorder));
        ctx.register_carried_host(&host_arg);

        let body_arg = Operand::from_bound_inputarg(&recorder);
        let resolved = ctx
            .resolve_operand_operand_opt(&body_arg)
            .expect("the imported host resolves");
        assert!(
            ctx.getptrinfo(&resolved).is_some_and(|info| info.is_null()),
            "the body must read the null the preamble proved, not its own slot"
        );
    }

    /// A slot whose import forwards to a value the BODY produces must keep
    /// naming the label arg.
    ///
    /// `OptUnroll.import_state`'s `source.set_forwarded(target)` unifies the label arg with
    /// the value the preamble jumps with, and those are one runtime value only
    /// at loop entry. A result produced inside the trace is redefined every
    /// iteration, so substituting it for a body operand turns "the value this
    /// iteration started with" into "the value it ends with" — and a
    /// virtualized field store is re-resolved late, at the loop-closing jump,
    /// which is where that difference becomes a wrong answer: `head =
    /// Node(i, head)` stores the node the body just allocated into that node's
    /// own `next`, and the list walk never terminates.
    #[test]
    fn a_peeled_body_keeps_its_label_arg_when_the_import_is_body_produced() {
        use majit_ir::operand::Operand;
        const BASE: u32 = 71;
        const SLOT: u32 = 5;
        const NUM_INPUTS: usize = 11;
        let mut ctx = OptContext::with_num_inputs_and_start_pos(
            8,
            NUM_INPUTS,
            BASE,
            BASE + NUM_INPUTS as u32,
        );
        let recorder = majit_ir::InputArg::from_type_rc(Type::Ref, SLOT);
        let host = majit_ir::InputArg::from_type_rc(Type::Ref, BASE + SLOT);
        let host_arg = Operand::from_bound_inputarg(&host);
        ctx.register_carried_host(&Operand::from_bound_inputarg(&recorder));
        ctx.register_carried_host(&host_arg);

        // The import lands on a result the trace produces — what a loop-carried
        // `head = Node(i, head)` forwards to, in place of the constant the
        // sibling test proves is still honoured.
        let produced = ctx.materialize_operand_at(OpRef::ref_op(4_242));
        ctx.make_equal_to(&host_arg, &produced);

        let body_arg = Operand::from_bound_inputarg(&recorder);
        let resolved = ctx
            .resolve_operand_operand_opt(&body_arg)
            .expect("the body arg resolves");
        assert!(
            resolved.same_box(&body_arg),
            "a body operand must keep its own label arg when the import names a \
             value the loop redefines, not the body's own result"
        );
    }

    /// A slot whose import forwards to another input arg must keep naming the
    /// label arg.
    ///
    /// An input arg is fixed for the whole loop, so it is tempting to let one
    /// stand for another. It cannot: substituting one label arg for another
    /// says the two slots hold the same value on *every* entry, which is a
    /// claim about the loop rather than about this iteration. A slot a
    /// residual writes behind the trace's back — a frame local reached through
    /// the running frame's `f_locals` — then reads as loop-invariant, and both
    /// the label arg and the reload after the call fall away, so the loop
    /// keeps answering the value it was compiled with.
    #[test]
    fn a_peeled_body_keeps_its_label_arg_when_the_import_is_another_input_arg() {
        use majit_ir::operand::Operand;
        const BASE: u32 = 71;
        const SLOT: u32 = 5;
        const OTHER: u32 = 7;
        const NUM_INPUTS: usize = 11;
        let mut ctx = OptContext::with_num_inputs_and_start_pos(
            8,
            NUM_INPUTS,
            BASE,
            BASE + NUM_INPUTS as u32,
        );
        let recorder = majit_ir::InputArg::from_type_rc(Type::Ref, SLOT);
        let host = majit_ir::InputArg::from_type_rc(Type::Ref, BASE + SLOT);
        let host_arg = Operand::from_bound_inputarg(&host);
        let other = majit_ir::InputArg::from_type_rc(Type::Ref, BASE + OTHER);
        let other_arg = Operand::from_bound_inputarg(&other);
        ctx.register_carried_host(&Operand::from_bound_inputarg(&recorder));
        ctx.register_carried_host(&host_arg);
        ctx.register_carried_host(&other_arg);

        // The import lands on a sibling label arg, in place of the constant
        // `a_peeled_body_reads_the_slot_the_import_forwarded` proves is
        // honoured.
        ctx.make_equal_to(&host_arg, &other_arg);

        let body_arg = Operand::from_bound_inputarg(&recorder);
        let resolved = ctx
            .resolve_operand_operand_opt(&body_arg)
            .expect("the body arg resolves");
        assert!(
            resolved.same_box(&body_arg),
            "a body operand must keep its own label arg when the import names \
             another slot, which agrees with it only on entry"
        );
    }

    /// The counterpart of `a_peeled_body_reads_the_slot_the_import_forwarded`:
    /// a preamble numbers its input args from zero, so slot `i` is the whole
    /// address and there is no second host to consult.
    #[test]
    fn a_preamble_reads_its_own_input_arg_slot() {
        use majit_ir::operand::Operand;
        const SLOT: u32 = 5;
        let mut ctx = OptContext::with_num_inputs_and_start_pos(8, 11, 0, 11);
        let recorder = majit_ir::InputArg::from_type_rc(Type::Ref, SLOT);
        // The position a shifted read would land on if the base were not 0.
        let decoy = majit_ir::InputArg::from_type_rc(Type::Ref, SLOT + 11);
        let decoy_arg = Operand::from_bound_inputarg(&decoy);
        decoy_arg.set_forwarded_const(majit_ir::Const::from_gcref(majit_ir::GcRef(0)));
        ctx.register_carried_host(&Operand::from_bound_inputarg(&recorder));
        ctx.register_carried_host(&decoy_arg);

        let body_arg = Operand::from_bound_inputarg(&recorder);
        let resolved = ctx
            .resolve_operand_operand_opt(&body_arg)
            .expect("the recorder slot resolves to itself");
        assert!(
            resolved.same_box(&body_arg),
            "a preamble must not read another slot's import"
        );
    }
}

#[cfg(test)]
mod constant_ptr_info_tests {
    //! info.py + 865-894 ConstPtrInfo / getptrinfo / getrawptrinfo
    //! parity tests for the typed-Int constant override path. RPython
    //! treats `ConstInt` (raw pointer) and `ConstPtr` uniformly via
    //! `_const.getref_base()`; majit must do the same regardless of how
    //! the constant pool stored the bits (`Value::Ref` vs `Value::Int`
    //! with a `Type::Ref` override).
    use super::*;
    use crate::optimizeopt::info::{PtrInfo, RawBufferPtrInfo, RawSlicePtrInfo, VStringVariant};
    use majit_ir::{GcRef, OpRef, Type, Value};
    use std::borrow::Cow;

    /// info.py getptrinfo(ConstPtr) → ConstPtrInfo(op).
    /// A `Value::Ref` constant must be wrapped in `PtrInfo::Constant`.
    #[test]
    fn getptrinfo_returns_constant_for_value_ref() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::ref_op(10_000);
        let b = ctx.materialize_operand_at(opref);
        ctx.seed_constant(&b, Value::Ref(GcRef(0xdead_beef)));
        match ctx.getptrinfo(&b) {
            Some(PtrInfo::Constant(g)) => assert_eq!(g.0, 0xdead_beef),
            other => panic!("expected ConstPtrInfo(0xdeadbeef), got {other:?}"),
        }
    }

    /// info.py getrawptrinfo(ConstInt) → ConstPtrInfo(op).
    /// Every ConstInt reaching `getrawptrinfo` is treated as a raw
    /// pointer (the caller has selected the helper because the
    /// `'i'`-typed box is intended as a pointer). The wrapped GcRef
    /// carries the int bits.
    #[test]
    fn getptrinfo_wraps_int_constant_as_const_ptr_info() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::int_op(10_002);
        let b = ctx.materialize_operand_at(opref);
        ctx.seed_constant(&b, Value::Int(42));
        match ctx.getptrinfo(&b) {
            Some(PtrInfo::Constant(g)) => assert_eq!(g.0, 42),
            other => panic!("expected ConstPtrInfo(42), got {other:?}"),
        }
    }

    /// info.py ConstPtrInfo._get_info(descr, optheap) parity:
    /// the same constant must always resolve to the same shared
    /// `const_infos[ref]` slot. Calling `get_const_info_mut` twice on a
    /// `Value::Ref` constant returns identical info — and a mutation
    /// observed via the second call confirms the slot identity.
    #[test]
    fn const_info_mut_returns_same_slot_for_value_ref() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::ref_op(10_004);
        let seed_box = ctx.materialize_operand_at(opref);
        ctx.seed_constant(&seed_box, Value::Ref(GcRef(0xa5a5_a5a5)));
        // First lookup: install Instance via the Vacant entry path,
        // then mark a known class so the second lookup observes it.
        {
            let info = ctx
                .get_const_info_mut(opref, None)
                .expect("Ref constant should have const_infos slot");
            *info = PtrInfo::known_class(0x1111_2222, true);
        }
        // Second lookup: the slot must contain the previously written
        // PtrInfo, not a freshly minted Instance.
        let info = ctx
            .get_const_info_mut(opref, None)
            .expect("Ref constant should still have const_infos slot");
        match info {
            PtrInfo::Instance(iinfo) => {
                assert_eq!(iinfo.known_class, Some(0x1111_2222));
            }
            other => panic!("expected Instance(known_class=Some) after re-lookup, got {other:?}"),
        }
    }

    /// info.py:719-720 `if not ref: raise InvalidLoop` — null protection.
    /// `get_const_info_mut` records a deferred `InvalidLoop` signal and
    /// returns `None` when the constant pointer resolves to a null `gcref`.
    /// The driver aborts the impossible trace shape at the next barrier so
    /// the JIT can retry; the Rust port mirrors that contract without
    /// unwinding (so it works under `panic=abort`).
    #[test]
    fn const_info_mut_signals_invalid_loop_on_null_value_ref_constant() {
        let mut ctx = OptContext::new(0);
        let ref_null = OpRef::ref_op(10_007);
        let seed_box = ctx.materialize_operand_at(ref_null);
        ctx.seed_constant(&seed_box, Value::Ref(GcRef(0)));
        let info = ctx.get_const_info_mut(ref_null, None);
        assert!(info.is_none(), "null constant base must not yield ptr info");
        let invalid = ctx
            .take_invalid_loop()
            .expect("expected pending InvalidLoop signal");
        assert!(invalid.0.contains("null constant base pointer"));
    }

    /// `Value::Int(0)` reaches `getrawptrinfo` as `ConstPtrInfo(NULL)`
    /// per `info.py`, then trips the null-constant InvalidLoop
    /// protection at `get_const_info_mut`. Mirrors the `Value::Ref(0)`
    /// case — null-pointer protection is uniform regardless of the
    /// underlying constant tag.
    #[test]
    fn const_info_mut_signals_invalid_loop_on_null_int_constant() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::int_op(10_010);
        let seed_box = ctx.materialize_operand_at(opref);
        ctx.seed_constant(&seed_box, Value::Int(0));
        let info = ctx.get_const_info_mut(opref, None);
        assert!(info.is_none(), "null constant base must not yield ptr info");
        let invalid = ctx
            .take_invalid_loop()
            .expect("expected pending InvalidLoop signal");
        assert!(invalid.0.contains("null constant base pointer"));
    }

    /// optimizer.py `is_raw_ptr(op)` parity for
    /// `info.RawSlicePtrInfo`: once a raw slice PtrInfo is present, it
    /// must be classified as an `AbstractRawPtrInfo` exactly like its
    /// parent raw buffer.
    #[test]
    fn is_raw_ptr_accepts_virtual_raw_slice() {
        let mut ctx = OptContext::new(0);
        let parent = OpRef::ref_op(10_010);
        let slice = OpRef::ref_op(10_011);

        let parent_box = ctx.materialize_operand_at(parent);
        let slice_box = ctx.materialize_operand_at(slice);
        ctx.set_ptr_info(
            &parent_box,
            PtrInfo::VirtualRawBuffer(RawBufferPtrInfo::new(0, 32, None)),
        );
        ctx.set_ptr_info(
            &slice_box,
            PtrInfo::VirtualRawSlice(RawSlicePtrInfo {
                offset: 8,
                parent: parent_box,
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );

        let parent_box = ctx.get_box_replacement_operand(parent);
        let slice_box = ctx.get_box_replacement_operand(slice);
        assert!(ctx.is_raw_ptr(&parent_box));
        assert!(ctx.is_raw_ptr(&slice_box));
    }

    /// vstring.py `StrPtrInfo.__init__(mode, is_virtual=False, length=-1)`
    /// parity for non-virtual strings: `make_nonnull_str()` must install
    /// a base `StrPtrInfo`, not one of the virtual subclasses.
    #[test]
    fn make_nonnull_str_initializes_ptr_variant() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::ref_op(10_012);
        // Synthetic-OpRef test fixture: lazy-allocate the operand so the
        // operand-direct `make_nonnull_str` can write through it. Production
        // callers obtain the box via `get_box_replacement_operand_opt`.
        let op_box = ctx.materialize_operand_at(opref);

        ctx.make_nonnull_str(&op_box, 0);

        match ctx.peek_ptr_info(&op_box) {
            Some(PtrInfo::Str(sinfo)) => {
                assert_eq!(sinfo.mode, 0);
                assert_eq!(sinfo.length, -1);
                assert!(sinfo.lenbound.is_none());
                assert!(matches!(sinfo.variant, VStringVariant::Ptr));
            }
            other => panic!("expected base StrPtrInfo, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod ensure_ptr_info_arg0_tests {
    //! optimizer.py `ensure_ptr_info_arg0` parity tests.
    //!
    //! Each test mirrors a single PyPy branch in `ensure_ptr_info_arg0`:
    //! the constant arg0 path, the AbstractVirtualPtrInfo early-return path,
    //! the NonNullPtrInfo upgrade path, and the assertion that fires on
    //! unexpected forwarded info shapes.
    use super::*;
    use crate::optimizeopt::info::{ArrayPtrInfo, EnsuredPtrInfo, PtrInfo};
    use crate::optimizeopt::intutils::IntBound;
    use majit_ir::{Descr, DescrRef, GcRef, Op, OpCode, OpRc, OpRef, SizeDescr, Type, Value};
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestSizeDescr {
        index: u32,
        is_object: bool,
    }

    impl Descr for TestSizeDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
            Some(self)
        }
    }

    impl SizeDescr for TestSizeDescr {
        fn size(&self) -> usize {
            64
        }
        fn type_id(&self) -> u32 {
            self.index
        }
        fn is_immutable(&self) -> bool {
            false
        }
        fn is_object(&self) -> bool {
            self.is_object
        }
    }

    fn struct_parent_descr() -> DescrRef {
        Arc::new(TestSizeDescr {
            index: 0xFFFF_0000,
            is_object: false,
        })
    }

    fn instance_parent_descr() -> DescrRef {
        Arc::new(TestSizeDescr {
            index: 0xFFFF_0001,
            is_object: true,
        })
    }

    #[derive(Debug)]
    struct TestFieldDescr {
        index: u32,
        parent: DescrRef,
    }

    impl Descr for TestFieldDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn as_field_descr(&self) -> Option<&dyn majit_ir::FieldDescr> {
            Some(self)
        }
    }

    impl majit_ir::FieldDescr for TestFieldDescr {
        fn offset(&self) -> usize {
            0
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> majit_ir::Type {
            majit_ir::Type::Int
        }
        fn index_in_parent(&self) -> usize {
            0
        }
        fn get_parent_descr(&self) -> Option<DescrRef> {
            Some(self.parent.clone())
        }
    }

    fn field_op_with_parent(parent: DescrRef) -> Op {
        // history.py:182 GetfieldGc receiver is a Ref box; arg0 must
        // carry the Ref variant tag (resoperation.py RefOp).
        let descr: DescrRef = Arc::new(TestFieldDescr { index: 0, parent });
        let mut op = Op::with_descr(
            OpCode::GetfieldGcI,
            &[crate::history::test_support::rooted_inputarg_operand(
                Type::Ref,
                0,
            )],
            descr,
        );
        op.pos().set(OpRef::int_op(1));
        op
    }

    fn array_op() -> Op {
        // ArraylenGc receiver is a Ref box.
        let descr: DescrRef = Arc::new(TestSizeDescr {
            index: 7,
            is_object: false,
        });
        let mut op = Op::with_descr(
            OpCode::ArraylenGc,
            &[crate::history::test_support::rooted_inputarg_operand(
                Type::Ref,
                0,
            )],
            descr,
        );
        op.pos().set(OpRef::int_op(1));
        op
    }

    /// optimizer.py:465-466: `if arg0.is_constant(): return info.ConstPtrInfo(arg0)`
    /// Constant `Value::Ref` arg0 → `EnsuredPtrInfo::Constant(gcref)`.
    #[test]
    fn ensure_ptr_info_arg0_returns_constant_for_value_ref() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let op = field_op_with_parent(struct_parent_descr());
        let seed_box = op.arg(0);
        ctx.seed_constant(&seed_box, Value::Ref(GcRef(0xdead_beef)));
        let info = ctx.ensure_ptr_info_arg0(&op);
        match info {
            EnsuredPtrInfo::Constant { gcref, .. } => assert_eq!(gcref.0, 0xdead_beef),
            _ => panic!("expected EnsuredPtrInfo::Constant"),
        }
    }

    /// optimizer.py:465-466 parity for plain `Value::Int` constants — PyPy
    /// returns `info.ConstPtrInfo(arg0)` regardless of the box's exact type.
    /// majit's port mirrors that by returning `Constant(GcRef(bits))`; null
    /// or unsafe pointers are filtered downstream by `_get_info`'s null
    /// protection (info.py).
    #[test]
    fn ensure_ptr_info_arg0_returns_constant_for_value_int() {
        // optimizer.py:465-466 PyPy parity: even Value::Int seeded at the
        // GetfieldGc receiver slot is interpreted as a ptr (ConstPtrInfo).
        // The Box class is still Ref because the receiver position is Ref;
        // the inner i64 value just happens to be tagged Int by the trace.
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let op = field_op_with_parent(struct_parent_descr());
        let seed_box = op.arg(0);
        ctx.seed_constant(&seed_box, Value::Int(1));
        let info = ctx.ensure_ptr_info_arg0(&op);
        assert!(matches!(info, EnsuredPtrInfo::Constant { .. }));
    }

    /// info.py `ConstPtrInfo.getlenbound(mode_string)` returns
    /// `IntBound.from_constant(length)` when `getstrlen1(mode)` knows the
    /// exact length. The Rust port consults the `string_length_resolver`
    /// hook the host runtime registered on `OptContext`.
    #[test]
    fn ensure_ptr_info_arg0_constant_string_returns_exact_length_via_resolver() {
        use std::sync::Arc;
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        // Resolver pretends every constant has byte-string length 5 in
        // mode_string and unicode length 7 in mode_unicode.
        ctx.string_length_resolver = Some(Arc::new(|gcref: GcRef, mode: u8| {
            assert_eq!(gcref.0, 0xC0FE);
            match mode {
                0 => Some(5),
                1 => Some(7),
                _ => None,
            }
        }));
        let op = {
            let descr: DescrRef = Arc::new(TestSizeDescr {
                index: 1,
                is_object: false,
            });
            let mut op = Op::with_descr(
                OpCode::Strlen,
                &[crate::history::test_support::rooted_inputarg_operand(
                    Type::Ref,
                    0,
                )],
                descr,
            );
            op.pos().set(OpRef::int_op(1));
            op
        };
        let seed_box = op.arg(0);
        ctx.seed_constant(&seed_box, Value::Ref(GcRef(0xC0FE)));
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let bound = info
            .getlenbound(Some(0))
            .expect("constant string length should resolve");
        assert_eq!(bound.lower, 5);
        assert_eq!(bound.upper, 5);
        let bound = info
            .getlenbound(Some(1))
            .expect("constant unicode length should resolve");
        assert_eq!(bound.lower, 7);
        assert_eq!(bound.upper, 7);
    }

    /// info.py:799-801 `if length < 0: return IntBound.nonnegative()` —
    /// no resolver registered → conservative nonnegative fallback.
    #[test]
    fn ensure_ptr_info_arg0_constant_string_falls_back_to_nonnegative_without_resolver() {
        use std::sync::Arc;
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let op = {
            let descr: DescrRef = Arc::new(TestSizeDescr {
                index: 1,
                is_object: false,
            });
            let mut op = Op::with_descr(
                OpCode::Strlen,
                &[crate::history::test_support::rooted_inputarg_operand(
                    Type::Ref,
                    0,
                )],
                descr,
            );
            op.pos().set(OpRef::int_op(1));
            op
        };
        let seed_box = op.arg(0);
        ctx.seed_constant(&seed_box, Value::Ref(GcRef(0x1234)));
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let bound = info
            .getlenbound(Some(0))
            .expect("nonnegative fallback should be Some");
        assert_eq!(bound.lower, IntBound::nonnegative().lower);
        assert!(!bound.is_constant());
    }

    /// optimizer.py:475-484 GETFIELD branch with `parent_descr.is_object() == false`
    /// → `info.StructPtrInfo(parent_descr)`. The Rust port returns the
    /// freshly-installed `PtrInfo::Struct` via `Forwarded(&mut PtrInfo)`.
    #[test]
    fn ensure_ptr_info_arg0_constructs_struct_for_non_object_field() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        // Keep a strong reference to the parent alive for the duration
        // of the test: `SimpleFieldDescr::parent_descr` is a
        // `Weak<DescrRef>` (breaks the cycle between SizeDescr.all_fielddescrs
        // and FieldDescr.parent_descr), so the test must hold the parent
        // Arc until `get_parent_descr()` has been called.
        let _parent = struct_parent_descr();
        let op = field_op_with_parent(_parent.clone());
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let pi = info.as_mut().expect("Forwarded variant expected");
        assert!(matches!(&*pi, PtrInfo::Struct(_)));
    }

    /// optimizer.py:480-484 GETFIELD branch with `parent_descr.is_object() == true`
    /// → `info.InstancePtrInfo(parent_descr)`.
    #[test]
    fn ensure_ptr_info_arg0_constructs_instance_for_object_field() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let _parent = instance_parent_descr();
        let op = field_op_with_parent(_parent.clone());
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let pi = info.as_mut().expect("Forwarded variant expected");
        assert!(matches!(&*pi, PtrInfo::Instance(_)));
    }

    /// optimizer.py:485-487 ARRAYLEN_GC branch → `info.ArrayPtrInfo(descr)`.
    /// The PyPy primitive returns the same arrayinfo across calls so
    /// callers can read `arrayinfo.getlenbound(None)` directly. The Rust
    /// port mirrors that and the `getlenbound` call resolves to the
    /// pre-installed `nonnegative` lenbound on the freshly-built ArrayPtrInfo.
    #[test]
    fn ensure_ptr_info_arg0_arraylen_returns_array_with_nonnegative_lenbound() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let op = array_op();
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let bound = info
            .getlenbound(None)
            .expect("ArrayPtrInfo.getlenbound(None) should be Some");
        assert_eq!(bound.lower, IntBound::nonnegative().lower);
    }

    /// info.py `ConstPtrInfo.getlenbound(mode)` returns
    /// `IntBound.nonnegative()` whenever `getstrlen1(mode)` produces a
    /// negative length. info.py makes `mode is None` (no
    /// vstring mode) one of those cases via the `else: return -1`
    /// branch. The Rust port must therefore answer `Some(nonnegative())`
    /// — not `None` — for `Constant.getlenbound(None)` so the
    /// ARRAYLEN_GC postprocess on a constant array still propagates a
    /// non-negative bound.
    #[test]
    fn ensure_ptr_info_arg0_constant_arraylen_returns_nonnegative() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let op = array_op();
        let seed_box = op.arg(0);
        ctx.seed_constant(&seed_box, Value::Ref(GcRef(0xfeed)));
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let bound = info
            .getlenbound(None)
            .expect("ConstPtrInfo.getlenbound(None) must mirror PyPy nonnegative fallback");
        assert_eq!(bound.lower, IntBound::nonnegative().lower);
        assert_eq!(bound.upper, IntBound::nonnegative().upper);
    }

    /// optimizer.py:467-469 `if isinstance(opinfo, AbstractVirtualPtrInfo):
    /// return opinfo` parity. A second call must return the SAME PtrInfo
    /// (verified by mutating via the first call and observing the mutation
    /// via the second). PyPy's structinfo identity is the test of record;
    /// the Rust port checks via state preserved across calls.
    #[test]
    fn ensure_ptr_info_arg0_returns_existing_array_unchanged() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let op = array_op();
        // Production binds trace arguments into the context before the pass
        // runs.  Seed this fixture with the operation's exact receiver Box so
        // both calls address that same `_forwarded` slot.
        ctx.seed_boxes_canonical(&[op.arg(0)]);
        // First call constructs the ArrayPtrInfo and tightens the lenbound
        // through the helper.
        {
            let mut info = ctx.ensure_ptr_info_arg0(&op);
            let mut handle = info.as_mut().expect("expected fresh ArrayPtrInfo");
            if let PtrInfo::Array(arr) = &mut *handle {
                let _ = arr.lenbound.make_gt_const(7);
            } else {
                panic!("expected fresh ArrayPtrInfo");
            }
        }
        // Second call returns the same ArrayPtrInfo (lenbound preserved).
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let mut handle = info.as_mut().expect("second call must still return Array");
        match &mut *handle {
            PtrInfo::Array(ArrayPtrInfo { lenbound, .. }) => {
                assert!(
                    lenbound.lower >= 8,
                    "second call must return the previously-mutated ArrayPtrInfo (lower={})",
                    lenbound.lower
                );
            }
            _ => panic!("second call must still return Array"),
        }
    }

    /// optimizer.py:470-474 `elif opinfo is not None: ...; assert opinfo is
    /// None or opinfo.__class__ is info.NonNullPtrInfo`. A pre-existing
    /// NonNullPtrInfo flows through the upgrade path; its `last_guard_pos`
    /// is preserved on the freshly-installed PtrInfo.
    #[test]
    fn ensure_ptr_info_arg0_upgrades_nonnull_to_struct() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let _parent = struct_parent_descr();
        let op = field_op_with_parent(_parent.clone());
        // Pre-install a NonNullPtrInfo with a specific last_guard_pos.
        let pos0_box = op.arg(0);
        ctx.set_ptr_info(&pos0_box, PtrInfo::NonNull { last_guard_pos: 7 });
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let mut handle = info.as_mut().expect("expected upgraded Struct, got None");
        match &mut *handle {
            pi @ PtrInfo::Struct(_) => {
                assert_eq!(pi.last_guard_pos(), Some(7));
            }
            other => panic!("expected upgraded Struct, got {other:?}"),
        }
    }

    /// optimizer.py:474 assertion: an unexpected forwarded info shape (e.g.
    /// an op-forwarding redirect that resolved to a non-PtrInfo state)
    /// must NOT silently overwrite. We seed an `Instance` PtrInfo, then
    /// hand it a field op with a different parent — the early-return path
    /// hits, and the existing Instance is returned without overwrite.
    #[test]
    fn ensure_ptr_info_arg0_does_not_overwrite_existing_instance() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let op = field_op_with_parent(struct_parent_descr());
        let pos0_box = op.arg(0);
        ctx.set_ptr_info(
            &pos0_box,
            PtrInfo::instance(Some(instance_parent_descr()), Some(0xc0de)),
        );
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let mut handle = info
            .as_mut()
            .expect("expected Instance preserved, got None");
        match &mut *handle {
            PtrInfo::Instance(_) => {} // unchanged
            other => panic!("expected Instance preserved, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod rd_virtual_info_builder_tests {
    use super::*;
    use crate::walkvirtual::VirtualVisitor;
    use majit_ir::{BoxEnv, Descr, DescrRef, FieldDescr, SizeDescr, Type};
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestSizeDescr {
        index: u32,
        type_id: u32,
        is_object: bool,
    }

    impl Descr for TestSizeDescr {
        fn index(&self) -> u32 {
            self.index
        }

        fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
            Some(self)
        }
    }

    impl SizeDescr for TestSizeDescr {
        fn size(&self) -> usize {
            16
        }

        fn type_id(&self) -> u32 {
            self.type_id
        }

        fn is_immutable(&self) -> bool {
            false
        }

        fn is_object(&self) -> bool {
            self.is_object
        }
    }

    #[derive(Debug)]
    struct TestFieldDescr {
        index: u32,
        offset: usize,
        field_size: usize,
        field_type: Type,
        is_pointer: bool,
    }

    impl Descr for TestFieldDescr {
        fn index(&self) -> u32 {
            self.index
        }

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

        fn is_pointer_field(&self) -> bool {
            self.is_pointer
        }
    }

    #[test]
    fn visit_virtual_preserves_field_descr_indices() {
        let mut builder = RdVirtualInfoBuilder;
        let size_descr: DescrRef = Arc::new(TestSizeDescr {
            index: 0x3000_0001,
            type_id: 7,
            is_object: true,
        });
        let field0: DescrRef = Arc::new(TestFieldDescr {
            index: 0x1000_0123,
            offset: 16,
            field_size: 8,
            field_type: Type::Int,
            is_pointer: false,
        });
        let field1: DescrRef = Arc::new(TestFieldDescr {
            index: 0x1000_0456,
            offset: 24,
            field_size: 8,
            field_type: Type::Ref,
            is_pointer: true,
        });

        let Some(majit_ir::RdVirtualInfo::VirtualInfo { fielddescrs, .. }) =
            builder.visit_virtual(&size_descr, &[], &[field0.clone(), field1.clone()])
        else {
            panic!("expected VirtualInfo");
        };

        assert_eq!(fielddescrs[0].index, field0.index());
        assert_eq!(fielddescrs[1].index, field1.index());
    }

    #[test]
    fn visit_virtual_classifies_pointer_field_by_is_pointer_field() {
        // `setfield` / `is_pointer_field`: FLAG_POINTER is independent of
        // `field_type()`. A GC pointer whose IR type is Int still snapshots
        // as Ref so `visit_vstruct` / `visit_virtual` decode via decode_ref.
        let mut builder = RdVirtualInfoBuilder;
        let size_descr: DescrRef = Arc::new(TestSizeDescr {
            index: 0x3000_0002,
            type_id: 9,
            is_object: true,
        });
        let struct_descr: DescrRef = Arc::new(TestSizeDescr {
            index: 0x3000_0003,
            type_id: 11,
            is_object: false,
        });
        let pointer_as_int: DescrRef = Arc::new(TestFieldDescr {
            index: 0x1000_0abc,
            offset: 24,
            field_size: 8,
            field_type: Type::Int,
            is_pointer: true,
        });
        let true_int: DescrRef = Arc::new(TestFieldDescr {
            index: 0x1000_0def,
            offset: 16,
            field_size: 8,
            field_type: Type::Int,
            is_pointer: false,
        });

        let Some(majit_ir::RdVirtualInfo::VirtualInfo { fielddescrs, .. }) = builder.visit_virtual(
            &size_descr,
            &[],
            &[pointer_as_int.clone(), true_int.clone()],
        ) else {
            panic!("expected VirtualInfo");
        };
        assert_eq!(fielddescrs[0].field_type, Type::Ref);
        assert_eq!(fielddescrs[0].offset, 24);
        assert_eq!(fielddescrs[0].field_size, 8);
        assert_eq!(fielddescrs[1].field_type, Type::Int);
        assert_eq!(fielddescrs[1].offset, 16);

        let Some(majit_ir::RdVirtualInfo::VStructInfo { fielddescrs, .. }) =
            builder.visit_vstruct(&struct_descr, &[], &[pointer_as_int, true_int])
        else {
            panic!("expected VStructInfo");
        };
        assert_eq!(fielddescrs[0].field_type, Type::Ref);
        assert_eq!(fielddescrs[0].offset, 24);
        assert_eq!(fielddescrs[0].field_size, 8);
        assert_eq!(fielddescrs[1].field_type, Type::Int);
        assert_eq!(fielddescrs[1].offset, 16);
    }

    // Parent SizeDescr list and visitor all_fielddescrs in different
    // order: storage keys by parent_list_slot, resume snapshots the
    // visitor list. ArrayStructInfo._compute_index / parent_list_slot.
    #[derive(Debug)]
    struct PairingSizeDescr {
        index: u32,
        type_id: u32,
        is_object: bool,
        fields: std::sync::OnceLock<Vec<Arc<dyn FieldDescr>>>,
    }

    impl Descr for PairingSizeDescr {
        fn index(&self) -> u32 {
            self.index
        }

        fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
            Some(self)
        }
    }

    impl SizeDescr for PairingSizeDescr {
        fn size(&self) -> usize {
            32
        }

        fn type_id(&self) -> u32 {
            self.type_id
        }

        fn is_immutable(&self) -> bool {
            false
        }

        fn is_object(&self) -> bool {
            self.is_object
        }

        fn all_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
            self.fields.get().map(|v| v.as_slice()).unwrap_or(&[])
        }
    }

    #[derive(Debug)]
    struct PairingFieldDescr {
        index: u32,
        offset: usize,
        field_size: usize,
        field_type: Type,
        is_pointer: bool,
        name: &'static str,
        parent: DescrRef,
    }

    impl Descr for PairingFieldDescr {
        fn index(&self) -> u32 {
            self.index
        }

        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for PairingFieldDescr {
        fn offset(&self) -> usize {
            self.offset
        }

        fn field_size(&self) -> usize {
            self.field_size
        }

        fn field_type(&self) -> Type {
            self.field_type
        }

        fn is_pointer_field(&self) -> bool {
            self.is_pointer
        }

        fn field_name(&self) -> &str {
            self.name
        }

        fn get_parent_descr(&self) -> Option<DescrRef> {
            Some(self.parent.clone())
        }
    }

    #[derive(Debug)]
    struct PairingDummyArray;

    impl Descr for PairingDummyArray {
        fn as_array_descr(&self) -> Option<&dyn majit_ir::ArrayDescr> {
            Some(self)
        }
    }

    impl majit_ir::ArrayDescr for PairingDummyArray {
        fn base_size(&self) -> usize {
            8
        }

        fn item_size(&self) -> usize {
            16
        }

        fn type_id(&self) -> u32 {
            0
        }

        fn item_type(&self) -> Type {
            Type::Void
        }

        fn is_array_of_structs(&self) -> bool {
            true
        }
    }

    static PAIRING_DUMMY_ARRAY: PairingDummyArray = PairingDummyArray;

    #[derive(Debug)]
    struct PairingInteriorFieldDescr {
        field: Arc<PairingFieldDescr>,
    }

    impl Descr for PairingInteriorFieldDescr {
        fn as_interior_field_descr(&self) -> Option<&dyn majit_ir::InteriorFieldDescr> {
            Some(self)
        }
    }

    impl majit_ir::InteriorFieldDescr for PairingInteriorFieldDescr {
        fn array_descr(&self) -> &dyn majit_ir::ArrayDescr {
            &PAIRING_DUMMY_ARRAY
        }

        fn field_descr(&self) -> &dyn FieldDescr {
            self.field.as_ref()
        }
    }

    #[derive(Debug)]
    struct PairingArrayDescr {
        interiors: std::sync::OnceLock<Vec<DescrRef>>,
    }

    impl Descr for PairingArrayDescr {
        fn as_array_descr(&self) -> Option<&dyn majit_ir::ArrayDescr> {
            Some(self)
        }
    }

    impl majit_ir::ArrayDescr for PairingArrayDescr {
        fn base_size(&self) -> usize {
            8
        }

        fn item_size(&self) -> usize {
            16
        }

        fn type_id(&self) -> u32 {
            3
        }

        fn item_type(&self) -> Type {
            Type::Void
        }

        fn is_array_of_structs(&self) -> bool {
            true
        }

        fn get_all_interiorfielddescrs(&self) -> Option<&[DescrRef]> {
            self.interiors.get().map(|v| v.as_slice())
        }
    }

    fn reversed_parent_int_ref_fields() -> (DescrRef, Arc<PairingFieldDescr>, Arc<PairingFieldDescr>)
    {
        let parent_sd = Arc::new(PairingSizeDescr {
            index: 0x3000_0100,
            type_id: 21,
            is_object: true,
            fields: std::sync::OnceLock::new(),
        });
        let parent: DescrRef = parent_sd.clone();
        let int_fd = Arc::new(PairingFieldDescr {
            index: 0x1000_1001,
            offset: 16,
            field_size: 8,
            field_type: Type::Int,
            is_pointer: false,
            name: "ival",
            parent: parent.clone(),
        });
        let ref_fd = Arc::new(PairingFieldDescr {
            index: 0x1000_1002,
            offset: 24,
            field_size: 8,
            field_type: Type::Int,
            is_pointer: true,
            name: "rval",
            parent: parent.clone(),
        });
        parent_sd
            .fields
            .set(vec![
                ref_fd.clone() as Arc<dyn FieldDescr>,
                int_fd.clone() as Arc<dyn FieldDescr>,
            ])
            .unwrap();
        (parent, int_fd, ref_fd)
    }

    #[test]
    fn get_virtual_fields_pairs_parent_list_slot_not_enumerate() {
        // Parent list [ref, int] vs visitor list [int, ref]. Storage keys
        // by parent_list_slot; fieldnums must follow the visitor list so
        // AbstractVirtualStructInfo.setfields decodes each fieldnum with
        // the descr at the same index.
        let (_parent, int_fd, ref_fd) = reversed_parent_int_ref_fields();
        assert_eq!(
            crate::optimizeopt::virtualize::parent_list_slot(int_fd.as_ref()),
            1
        );
        assert_eq!(
            crate::optimizeopt::virtualize::parent_list_slot(ref_fd.as_ref()),
            0
        );

        let visitor_sd = Arc::new(PairingSizeDescr {
            index: 0x3000_0101,
            type_id: 22,
            is_object: true,
            fields: std::sync::OnceLock::new(),
        });
        visitor_sd
            .fields
            .set(vec![
                int_fd.clone() as Arc<dyn FieldDescr>,
                ref_fd.clone() as Arc<dyn FieldDescr>,
            ])
            .unwrap();
        let visitor: DescrRef = visitor_sd;

        let mut ctx = OptContext::with_num_inputs(16, 0);
        let int_val = ctx.materialize_operand_at(majit_ir::OpRef::int_op(50));
        let ref_val = ctx.materialize_operand_at(majit_ir::OpRef::ref_op(51));
        let virtual_op = majit_ir::OpRef::ref_op(7);
        let virtual_box = ctx.materialize_operand_at(virtual_op);
        let mut fields = majit_ir::ptr_info::VirtualFieldList::new();
        // Storage keys follow the allocation/visitor SizeDescr list
        // (parent_list_slot_in), not the field's nested parent list.
        fields.push((0, int_val.clone()));
        fields.push((1, ref_val.clone()));
        ctx.set_ptr_info(
            &virtual_box,
            PtrInfo::Virtual(crate::optimizeopt::info::VirtualInfo {
                descr: visitor.clone(),
                known_class: None,
                ob_type_descr: None,
                fields,
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );

        let env = OptBoxEnv { ctx: &ctx };
        let vf = env
            .get_virtual_fields(virtual_op)
            .expect("virtual must expose fields");
        let int_opref = env.get_box_replacement(int_val.to_opref());
        let ref_opref = env.get_box_replacement(ref_val.to_opref());
        assert_eq!(vf.field_oprefs, vec![int_opref, ref_opref]);

        let visitor_descrs: Vec<DescrRef> =
            vec![int_fd.clone() as DescrRef, ref_fd.clone() as DescrRef];
        let mut builder = RdVirtualInfoBuilder;
        let Some(majit_ir::RdVirtualInfo::VirtualInfo { fielddescrs, .. }) =
            builder.visit_virtual(&visitor, &[], &visitor_descrs)
        else {
            panic!("expected VirtualInfo");
        };
        assert_eq!(fielddescrs[0].field_type, Type::Int);
        assert_eq!(fielddescrs[1].field_type, Type::Ref);
        assert_eq!(env.get_type(vf.field_oprefs[0]), Type::Int);
        assert_eq!(env.get_type(vf.field_oprefs[1]), Type::Ref);
    }

    #[test]
    fn pointer_field_snapshot_kind_matches_stored_box_type() {
        // virtualize.py optimize_SETFIELD_GC stores
        // get_box_replacement(arg1) as-is. resume.py decoder.setfield
        // classifies by is_pointer_field; decode_box asserts
        // box.type == kind. A FLAG_POINTER field's SETFIELD value is
        // already a Ref box (descr.py get_type_flag maps a GC Ptr to
        // FLAG_POINTER). snapshot_field_descr_info writes that same
        // split, so the snapshot kind and the stored opref type agree.
        let (_parent, _int_fd, ref_fd) = reversed_parent_int_ref_fields();
        assert!(ref_fd.is_pointer_field());

        let visitor_sd = Arc::new(PairingSizeDescr {
            index: 0x3000_0102,
            type_id: 23,
            is_object: true,
            fields: std::sync::OnceLock::new(),
        });
        visitor_sd
            .fields
            .set(vec![ref_fd.clone() as Arc<dyn FieldDescr>])
            .unwrap();
        let visitor: DescrRef = visitor_sd;

        let mut ctx = OptContext::with_num_inputs(16, 0);
        let ref_val = ctx.materialize_operand_at(majit_ir::OpRef::ref_op(51));
        let virtual_op = majit_ir::OpRef::ref_op(8);
        let virtual_box = ctx.materialize_operand_at(virtual_op);
        let mut fields = majit_ir::ptr_info::VirtualFieldList::new();
        fields.push((0, ref_val.clone()));
        ctx.set_ptr_info(
            &virtual_box,
            PtrInfo::Virtual(crate::optimizeopt::info::VirtualInfo {
                descr: visitor.clone(),
                known_class: None,
                ob_type_descr: None,
                fields,
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );

        let env = OptBoxEnv { ctx: &ctx };
        let vf = env
            .get_virtual_fields(virtual_op)
            .expect("virtual must expose fields");
        let mut builder = RdVirtualInfoBuilder;
        let visitor_descrs: Vec<DescrRef> = vec![ref_fd.clone() as DescrRef];
        let Some(majit_ir::RdVirtualInfo::VirtualInfo { fielddescrs, .. }) =
            builder.visit_virtual(&visitor, &[], &visitor_descrs)
        else {
            panic!("expected VirtualInfo");
        };
        assert_eq!(fielddescrs[0].field_type, Type::Ref);
        assert_eq!(env.get_type(vf.field_oprefs[0]), Type::Ref);
        assert_eq!(
            super::snapshot_field_descr_info(&(ref_fd as DescrRef)).field_type,
            Type::Ref
        );
    }

    #[test]
    fn get_virtual_fields_pairs_varraystruct_interior_by_parent_list_slot() {
        // Interior visitor list [int, ref]; inner FieldDescr parent list
        // [ref, int]. ArrayStructInfo._compute_index uses
        // get_field_descr().get_index(); pyre stores via parent_list_slot.
        let (_parent, int_fd, ref_fd) = reversed_parent_int_ref_fields();
        let int_ifd: DescrRef = Arc::new(PairingInteriorFieldDescr {
            field: int_fd.clone(),
        });
        let ref_ifd: DescrRef = Arc::new(PairingInteriorFieldDescr {
            field: ref_fd.clone(),
        });
        let array_sd = Arc::new(PairingArrayDescr {
            interiors: std::sync::OnceLock::new(),
        });
        array_sd
            .interiors
            .set(vec![int_ifd.clone(), ref_ifd.clone()])
            .unwrap();
        let arraydescr: DescrRef = array_sd;

        let mut ctx = OptContext::with_num_inputs(16, 0);
        let int_val = ctx.materialize_operand_at(majit_ir::OpRef::int_op(60));
        let ref_val = ctx.materialize_operand_at(majit_ir::OpRef::ref_op(61));
        let virtual_op = majit_ir::OpRef::ref_op(8);
        let virtual_box = ctx.materialize_operand_at(virtual_op);
        let mut elem = majit_ir::ptr_info::VirtualFieldList::new();
        elem.push((1, int_val.clone()));
        elem.push((0, ref_val.clone()));
        ctx.set_ptr_info(
            &virtual_box,
            PtrInfo::VirtualArrayStruct(crate::optimizeopt::info::ArrayStructInfo {
                descr: arraydescr.clone(),
                fielddescrs: vec![int_ifd.clone(), ref_ifd.clone()],
                element_fields: vec![elem],
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );

        let env = OptBoxEnv { ctx: &ctx };
        let vf = env
            .get_virtual_fields(virtual_op)
            .expect("array struct must expose fields");
        let int_opref = env.get_box_replacement(int_val.to_opref());
        let ref_opref = env.get_box_replacement(ref_val.to_opref());
        assert_eq!(vf.field_oprefs, vec![int_opref, ref_opref]);

        let mut builder = RdVirtualInfoBuilder;
        let Some(majit_ir::RdVirtualInfo::VArrayStructInfo {
            field_types,
            fielddescrs,
            ..
        }) = builder.visit_varraystruct(&arraydescr, 1, &[], &[int_ifd, ref_ifd])
        else {
            panic!("expected VArrayStructInfo");
        };
        assert_eq!(field_types, vec![1u8, 0u8]);
        assert_eq!(fielddescrs.len(), 2);
        assert_eq!(env.get_type(vf.field_oprefs[0]), Type::Int);
        assert_eq!(env.get_type(vf.field_oprefs[1]), Type::Ref);
    }

    fn same_offset_empty_name_fields() -> (
        DescrRef,
        Arc<PairingFieldDescr>,
        Arc<PairingFieldDescr>,
        Arc<PairingFieldDescr>,
        Arc<PairingFieldDescr>,
    ) {
        let parent_sd = Arc::new(PairingSizeDescr {
            index: 0x3000_0200,
            type_id: 31,
            is_object: true,
            fields: std::sync::OnceLock::new(),
        });
        let parent: DescrRef = parent_sd.clone();
        let mk = |index, is_pointer| {
            Arc::new(PairingFieldDescr {
                index,
                offset: 16,
                field_size: 8,
                field_type: Type::Int,
                is_pointer,
                name: "",
                parent: parent.clone(),
            })
        };
        let listed_ref = mk(0x1000_1002, true);
        let listed_int = mk(0x1000_1001, false);
        let visitor_int = mk(0x1000_10aa, false);
        let visitor_ref = mk(0x1000_10bb, true);
        parent_sd
            .fields
            .set(vec![
                listed_ref.clone() as Arc<dyn FieldDescr>,
                listed_int.clone() as Arc<dyn FieldDescr>,
            ])
            .unwrap();
        (parent, listed_int, listed_ref, visitor_int, visitor_ref)
    }

    #[test]
    fn get_virtual_fields_pairs_same_offset_empty_name_int_ref() {
        // Flattened unnamed leaves at one offset. The field's nested parent
        // list is [ref, int]; the allocation/visitor list is [int, ref].
        // Storage keys follow parent_list_slot_in(visitor) so Int is
        // slot 0 and AbstractVirtualStructInfo.setfields does not
        // decode_int the Ref TAGCONST.
        let (_parent, _listed_int, _listed_ref, visitor_int, visitor_ref) =
            same_offset_empty_name_fields();
        assert_eq!(
            crate::optimizeopt::virtualize::parent_list_slot(visitor_int.as_ref()),
            1
        );
        assert_eq!(
            crate::optimizeopt::virtualize::parent_list_slot(visitor_ref.as_ref()),
            0
        );

        let visitor_sd = Arc::new(PairingSizeDescr {
            index: 0x3000_0201,
            type_id: 32,
            is_object: true,
            fields: std::sync::OnceLock::new(),
        });
        visitor_sd
            .fields
            .set(vec![
                visitor_int.clone() as Arc<dyn FieldDescr>,
                visitor_ref.clone() as Arc<dyn FieldDescr>,
            ])
            .unwrap();
        let visitor: DescrRef = visitor_sd;
        assert_eq!(
            crate::optimizeopt::virtualize::parent_list_slot_in(
                Some(&visitor),
                visitor_int.as_ref()
            ),
            0
        );
        assert_eq!(
            crate::optimizeopt::virtualize::parent_list_slot_in(
                Some(&visitor),
                visitor_ref.as_ref()
            ),
            1
        );

        let mut ctx = OptContext::with_num_inputs(16, 0);
        let int_val = ctx.materialize_operand_at(majit_ir::OpRef::int_op(70));
        let ref_val = ctx.materialize_operand_at(majit_ir::OpRef::ref_op(71));
        let virtual_op = majit_ir::OpRef::ref_op(9);
        let virtual_box = ctx.materialize_operand_at(virtual_op);
        let mut fields = majit_ir::ptr_info::VirtualFieldList::new();
        fields.push((0, int_val.clone()));
        fields.push((1, ref_val.clone()));
        ctx.set_ptr_info(
            &virtual_box,
            PtrInfo::Virtual(crate::optimizeopt::info::VirtualInfo {
                descr: visitor.clone(),
                known_class: None,
                ob_type_descr: None,
                fields,
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );

        let env = OptBoxEnv { ctx: &ctx };
        let vf = env
            .get_virtual_fields(virtual_op)
            .expect("virtual must expose fields");
        let int_opref = env.get_box_replacement(int_val.to_opref());
        let ref_opref = env.get_box_replacement(ref_val.to_opref());
        assert_eq!(vf.field_oprefs, vec![int_opref, ref_opref]);

        let visitor_descrs: Vec<DescrRef> = vec![
            visitor_int.clone() as DescrRef,
            visitor_ref.clone() as DescrRef,
        ];
        let mut builder = RdVirtualInfoBuilder;
        let Some(majit_ir::RdVirtualInfo::VirtualInfo { fielddescrs, .. }) =
            builder.visit_virtual(&visitor, &[], &visitor_descrs)
        else {
            panic!("expected VirtualInfo");
        };
        assert_eq!(fielddescrs[0].field_type, Type::Int);
        assert_eq!(fielddescrs[1].field_type, Type::Ref);
        assert_eq!(env.get_type(vf.field_oprefs[0]), Type::Int);
        assert_eq!(env.get_type(vf.field_oprefs[1]), Type::Ref);
    }
}

#[cfg(test)]
mod intbound_invariant_tests {
    use super::*;
    use crate::optimizeopt::intutils::IntBound;
    use majit_ir::{GcRef, OpRef, Value};

    #[test]
    #[should_panic]
    fn getintbound_rejects_non_int_boxes() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::ref_op(20_000);
        let seed_box = ctx.materialize_operand_at(opref);
        ctx.seed_constant(&seed_box, Value::Ref(GcRef(0xdead_beef)));
        let _ = {
            let __mb = ctx.materialize_operand_at(opref);
            ctx.getintbound_handle(&__mb).borrow().clone()
        };
    }

    #[test]
    #[should_panic]
    fn setintbound_rejects_non_int_boxes() {
        let ctx = OptContext::new(0);
        // setintbound asserts `op.type_()` is Int/Void per optimizer.py.
        // A Ref-typed operand should trigger the panic.
        let ref_op = crate::history::test_support::rooted_inputarg_operand(majit_ir::Type::Ref, 0);
        ctx.setintbound(&ref_op, &IntBound::nonnegative());
    }
}

#[cfg(test)]
mod imported_short_preamble_fallback_tests {
    use super::*;
    use majit_ir::{Op, OpCode, OpRc, OpRef};

    #[test]
    fn force_op_from_preamble_replays_pop_without_builder_lookup() {
        // 2 Ref inputargs for the body label — typical loop-body shape.
        let mut ctx =
            OptContext::with_inputarg_types(16, &[majit_ir::Type::Ref, majit_ir::Type::Ref]);
        ctx.initialize_imported_short_preamble_builder(
            &[OpRef::input_arg_ref(0), OpRef::input_arg_ref(1)],
            &[OpRef::int_op(7), OpRef::int_op(8)],
            &[],
        );

        let mut replay_op = Op::new(
            OpCode::IntAdd,
            &[
                crate::history::test_support::rooted_resop_operand(majit_ir::Type::Int, 7),
                crate::history::test_support::rooted_resop_operand(majit_ir::Type::Int, 8),
            ],
        );
        replay_op.pos().set(OpRef::int_op(14));
        // shortpreamble.py non-invented PureOp.produce_op: `op = self.res`.
        // pop.op carries the body-visible OpRef directly (no forwarding chain
        // installed for non-invented Pure).
        let pop = crate::optimizeopt::info::PreambleOp {
            op: majit_ir::operand::Operand::bound_from_opref(OpRef::int_op(41)),
            invented_name: false,
            preamble_op: OpRc::new(replay_op),
            same_as_source: None,
        };

        let forced = ctx.force_op_from_preamble_op(&pop).to_opref();
        assert_eq!(forced, OpRef::int_op(41));

        let sp = ctx
            .build_imported_short_preamble()
            .expect("imported short preamble builder should exist");
        assert_eq!(sp.ops.len(), 1);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntAdd);
        assert_eq!(
            sp.ops[0]
                .op
                .args_slice()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![OpRef::int_op(7), OpRef::int_op(8)]
        );
        assert_eq!(sp.ops[0].op.pos().get(), OpRef::int_op(14));
    }
}

#[cfg(test)]
mod opt_box_env_tests {
    use super::*;
    use crate::optimizeopt::info::VirtualInfo;
    use majit_ir::{BoxEnv, DescrRef, GcRef, OpRef};
    use std::sync::Arc;

    #[derive(Debug)]
    struct DummySizeDescr;

    impl majit_ir::Descr for DummySizeDescr {
        fn index(&self) -> u32 {
            0
        }

        fn clone_descr(&self) -> Option<DescrRef> {
            Some(Arc::new(DummySizeDescr))
        }

        fn as_size_descr(&self) -> Option<&dyn majit_ir::SizeDescr> {
            Some(self)
        }
    }

    impl majit_ir::SizeDescr for DummySizeDescr {
        fn size(&self) -> usize {
            24
        }

        fn vtable(&self) -> usize {
            0x1234
        }

        fn type_id(&self) -> u32 {
            7
        }

        fn is_immutable(&self) -> bool {
            false
        }
    }

    #[test]
    fn opt_box_env_is_virtual_ref_follows_box_replacement() {
        let mut ctx = OptContext::with_num_inputs(16, 0);
        let source = OpRef::ref_op(12);
        let target = OpRef::ref_op(21);
        let source_box = ctx.materialize_operand_at(source);
        let target_box = ctx.materialize_operand_at(target);
        ctx.make_equal_to(&source_box, &target_box);
        let resolved = OptBoxEnv { ctx: &ctx }.get_box_replacement_operand(source);
        ctx.set_ptr_info(
            &target_box,
            PtrInfo::Virtual(VirtualInfo {
                descr: Arc::new(DummySizeDescr),
                known_class: Some(0x1234),
                ob_type_descr: None,
                fields: Default::default(),
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );

        let env = OptBoxEnv { ctx: &ctx };
        assert!(
            env.is_virtual_ref(&resolved),
            "forwarded snapshot boxes must classify as virtual via replacement"
        );
    }

    #[test]
    fn native_box_env_reads_live_constant_info_and_typed_constants() {
        let mut ctx = OptContext::with_num_inputs(16, 0);
        let producer = ctx.materialize_operand_at(OpRef::ref_op(21));
        let env = OptBoxEnv { ctx: &ctx };
        let resolved = env.get_box_replacement_operand(producer.to_opref());
        assert!(!env.is_const(&resolved));
        ctx.set_ptr_info(&producer, PtrInfo::Constant(GcRef(0x1234)));
        assert!(env.is_const(&resolved));
        assert_eq!(env.get_const(&resolved), (0x1234, Type::Ref));
        for value in [
            Value::Int(i64::MAX),
            Value::Float(-0.0),
            Value::Ref(GcRef::NULL),
        ] {
            let constant = Operand::const_from_value(value);
            assert!(env.is_const(&constant));
            assert_eq!(env.get_const(&constant), const_value_as_word(value));
        }
    }

    #[test]
    fn virtual_resume_cache_lives_on_the_forwarded_info() {
        let mut ctx = OptContext::with_num_inputs(16, 0);
        let source = OpRef::ref_op(12);
        let target = OpRef::ref_op(21);
        let source_box = ctx.materialize_operand_at(source);
        let target_box = ctx.materialize_operand_at(target);
        ctx.make_equal_to(&source_box, &target_box);
        ctx.set_ptr_info(
            &target_box,
            PtrInfo::Virtual(VirtualInfo {
                descr: Arc::new(DummySizeDescr),
                known_class: Some(0x1234),
                ob_type_descr: None,
                fields: Default::default(),
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );
        let env = OptBoxEnv { ctx: &ctx };
        let first = env.make_virtual_info(source, vec![1, 2]).unwrap();
        assert!(env.virtual_info_would_be_reused(target, &[1, 2]));
        let same = env.make_virtual_info(target, vec![1, 2]).unwrap();
        assert!(std::rc::Rc::ptr_eq(&first, &same));
        {
            let info = target_box.ptr_info().unwrap();
            let cached = info.cached_vinfo().unwrap().borrow();
            assert!(std::rc::Rc::ptr_eq(cached.as_ref().unwrap(), &first));
        }
        let changed = env.make_virtual_info(source, vec![2, 1]).unwrap();
        assert!(!std::rc::Rc::ptr_eq(&first, &changed));
        assert!(first.equals(&[1, 2]));
        assert!(env.virtual_info_would_be_reused(target, &[2, 1]));
    }

    #[test]
    fn materialize_operand_at_lazy_materialises_inputarg_for_empty_inputarg_slot() {
        // `resoperation.py AbstractInputArg` and
        // `resoperation.py AbstractResOp` are distinct classes
        // upstream — a Box materialised against `OpRef::InputArg*`
        // must be `is_inputarg()` so the chain walker reconstructs
        // the same variant on the round-trip through the
        // forwarding chain.  Prior to the per-variant empty-slot
        // path, materialisation always emitted `new_resop`,
        // silently demoting the inputarg.
        // `with_inputarg_types` would seed the inputarg slots
        // eagerly, defeating the lazy-materialisation check.  Use
        // `with_num_inputs(_, 0)` to get an empty pool then hand an
        // `OpRef::InputArg*` in directly — the regression we are
        // covering is exactly the path where the empty slot must
        // mint a `new_inputarg`.
        let mut ctx = OptContext::with_num_inputs(8, 0);
        let arg = OpRef::input_arg_typed(0, majit_ir::Type::Int);
        let materialised = ctx.materialize_operand_at(arg);
        let ia = materialised
            .bound_inputarg()
            .expect("empty InputArg* slot lazy-materialised the wrong host kind");
        assert_eq!(ia.index, 0);
        assert_eq!(ia.tp.get(), majit_ir::Type::Int);

        // Re-entering must resolve to the same canonical `_forwarded`
        // host (`resoperation.py AbstractInputArg._forwarded`) —
        // identity lives on the bound `InputArgRc`.
        let second = ctx.materialize_operand_at(arg);
        assert!(
            majit_ir::InputArgRc::ptr_eq(
                &ia,
                &second.bound_inputarg().expect("second bound to InputArg"),
            ),
            "second materialize_operand_at must resolve to the same InputArg host",
        );
    }

    #[test]
    fn materialize_operand_at_lazy_materialises_resop_for_empty_resop_slot() {
        // Companion to the InputArg case — a producer-less `OpRef::int_op(_)`
        // is minted at allocation (`mint_box_at`, the `ResOperation(...)`
        // analog) and then found by `materialize_operand_at`.
        let mut ctx = OptContext::with_num_inputs(8, 0);
        let result = OpRef::int_op(3);
        let minted = ctx.mint_box_at(result);
        let op = minted
            .bound_op()
            .expect("empty ResOp slot mint_box_at bound the wrong host kind");
        assert_eq!(op.pos().get(), result);
        let second = ctx.materialize_operand_at(result);
        assert!(
            majit_ir::OpRc::ptr_eq(&op, &second.bound_op().expect("second bound to ResOp"),),
            "materialize_operand_at must resolve to the mint_box_at host",
        );
    }
}
