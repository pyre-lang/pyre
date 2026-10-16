//! Owner of `ConstPtr.value`.
//!
//! `history.py` `ConstPtr` is one GC object. Holders store the box; the
//! collector writes `value`. `OpRef` is a `Copy` word, so the box's
//! address cannot live in the word: a minor would stale every copy.
//! The word is an index into this table. The table is the single
//! mutable `value` slot. `gcreftracer.GcTable` is the same shape for
//! addresses baked into compiled code.
//!
//! A slot is a root only while a live holder traces it
//! (`trace_index`). `history.py` drops a `ConstPtr` with the trace that
//! referenced the box; walking every slot would keep a dead trace's
//! referent alive for the process. Compiled loops keep their own
//! `GcTable` roots. A major collection stamps the slots extra-root
//! walkers still hold and [`sweep_untraced`] frees the rest onto a
//! free list `intern` reuses. Index 0 stays null. The table and
//! [`MAJOR_LIVE`] are process-global: a major that walked only one
//! thread's extra areas must not sweep. Two live `install_gc_box`
//! collectors at once is not a supported configuration.
//!
//! Intern key is `gc_id_or_identityhash` (`minimark.py`
//! `id_or_identityhash`), not the address. The key is recorded at
//! intern time and is not recomputed during a collection.

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicUsize, Ordering};

use parking_lot::Mutex;

use crate::value::{GcRef, gc_current_object_address, gc_id_or_identityhash};

/// Live word of `addr`. `gc_current_object_address` follows a nursery
/// forwarding stub (`_trace_drag_out`) without treating the address as a
/// root. Unset (tests, no collector) is identity.
fn live_gcref(addr: GcRef) -> GcRef {
    if addr.is_null() {
        addr
    } else {
        GcRef(gc_current_object_address(addr.0))
    }
}

struct Table {
    /// Index 0 is null and is never interned.
    slots: Vec<GcRef>,
    /// Intern key recorded with the slot. `minimark.py`
    /// `id_or_identityhash` is not recomputed on a stale address.
    hashes: Vec<u64>,
    /// `(identity hash, index)`, sorted by hash then index.
    by_hash: Vec<(u64, u32)>,
    /// `(current address, index)`, sorted by address then index.
    /// `history.py` `ConstPtr.value` after a move is this word.
    by_addr: Vec<(u64, u32)>,
    /// `begin_wave` generation that last traced this slot. Wave 0 does
    /// not dedup: a unit test that never opens a wave still forwards.
    marks: Vec<u32>,
    /// [`MAJOR_LIVE`] generation a holder last stamped. A slot whose
    /// generation is older than the current major is dead at sweep.
    live: Vec<u32>,
    /// Reclaimed indexes `intern` reuses. Index 0 is never queued.
    free: Vec<u32>,
    /// Indexes interned since the last [`collect_young_const_ptrs`] that
    /// owned the referent. `resumecode.py` `YOUNG_NUMBERINGS` for `ConstPtr`:
    /// the intern-to-holder window. An `OpRef` Copy index is not a GCREF
    /// root; extra-area `trace_index` is registered later. A minor in that
    /// window must still forward the slot (`incminimark.py`
    /// `collect_oldrefs_to_nursery`). That walk is per MiniMark; a collector
    /// that does not own the referent requeues the index.
    young: Vec<u32>,
    /// `1` while `idx` sits in [`Table::young`]. Cleared when the list drains.
    young_mark: Vec<u8>,
}

/// Non-zero while a collection (or a test helper) is forwarding.
/// Slots already traced in this wave are not written twice.
static WAVE: AtomicU32 = AtomicU32::new(0);

/// Next id `Wave::enter` publishes. Separate from `WAVE` so dropping a
/// guard can restore the enclosing wave without reusing an id `marks`
/// still holds. `0` is never issued. After the counter wraps, the next
/// id is published only once `marks` has been cleared.
static NEXT_WAVE: AtomicU32 = AtomicU32::new(1);

impl Table {
    fn new() -> Self {
        Self {
            slots: vec![GcRef::NULL],
            hashes: vec![0],
            by_hash: Vec::new(),
            by_addr: Vec::new(),
            marks: vec![0],
            live: vec![0],
            free: Vec::new(),
            young: Vec::new(),
            young_mark: vec![0],
        }
    }
}

static TABLE: Mutex<Table> = Mutex::new(Table {
    slots: Vec::new(),
    hashes: Vec::new(),
    by_hash: Vec::new(),
    by_addr: Vec::new(),
    marks: Vec::new(),
    live: Vec::new(),
    free: Vec::new(),
    young: Vec::new(),
    young_mark: Vec::new(),
});
static READY: AtomicBool = AtomicBool::new(false);

// Reentrancy on the walking thread. Another test thread must not
// observe it: the table mutex is what serializes slot updates.
thread_local! {
    static WALKING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn walking() -> bool {
    WALKING.with(|cell| cell.get())
}

fn set_walking(value: bool) {
    WALKING.with(|cell| cell.set(value));
}

/// Lock-free mirror of `Table.slots` for `history.py` `ConstPtr.getref_base`.
///
/// `resolve` runs once per constant ref in a blackhole resume. Taking
/// the intern mutex there is not the field load upstream emits. The
/// directory grows under the intern mutex; a published directory and
/// each published chunk pointer stay valid for lock-free readers, so a
/// replaced directory is leaked. The table mutex still covers intern,
/// the hash index, reclamation, and the forwarding walk.
const SLOT_CHUNK: usize = 256;

struct SlotChunk {
    slots: [AtomicUsize; SLOT_CHUNK],
}

struct ChunkDir {
    n: usize,
    ptrs: *mut AtomicPtr<SlotChunk>,
}

static DIR: AtomicPtr<ChunkDir> = AtomicPtr::new(std::ptr::null_mut());

/// `history.py` drops a `ConstPtr` with the last holder. A major that
/// traces every live holder then frees a slot no holder stamped.
/// Process-global: one generation for the whole table. Sound when the
/// stamping walk covered every holder (STW, or a single-mutator process).
static MAJOR_LIVE: AtomicU32 = AtomicU32::new(1);

fn current_live() -> u32 {
    MAJOR_LIVE.load(Ordering::Relaxed)
}

/// Open a major-live generation. Slots interned or `trace_index`'d
/// after this call survive [`sweep_untraced`]; older stamps die.
pub fn begin_major_live() {
    let cur = MAJOR_LIVE.load(Ordering::Relaxed);
    let mut next = cur.wrapping_add(1);
    if next == 0 {
        next = 1;
    }
    MAJOR_LIVE.store(next, Ordering::Relaxed);
}

fn grow_dir(min_len: usize) {
    let old = DIR.load(Ordering::Acquire);
    let old_len = if old.is_null() {
        0
    } else {
        unsafe { (*old).n }
    };
    if min_len <= old_len {
        return;
    }
    let new_len = min_len.next_power_of_two().max(8);
    let chunks: Vec<AtomicPtr<SlotChunk>> = (0..new_len)
        .map(|_| AtomicPtr::new(std::ptr::null_mut()))
        .collect();
    if !old.is_null() {
        let old_ptrs = unsafe { (*old).ptrs };
        for i in 0..old_len {
            chunks[i].store(
                unsafe { (*old_ptrs.add(i)).load(Ordering::Relaxed) },
                Ordering::Relaxed,
            );
        }
    }
    let ptrs = Box::into_raw(chunks.into_boxed_slice()) as *mut AtomicPtr<SlotChunk>;
    let fresh = Box::into_raw(Box::new(ChunkDir { n: new_len, ptrs }));
    DIR.store(fresh, Ordering::Release);
}

fn publish_slot(index: usize, addr: GcRef) {
    if index == 0 {
        return;
    }
    let ci = index / SLOT_CHUNK;
    let mut dir = DIR.load(Ordering::Acquire);
    if dir.is_null() || ci >= unsafe { (*dir).n } {
        grow_dir(ci + 1);
        dir = DIR.load(Ordering::Acquire);
    }
    let slot = unsafe { (*dir).ptrs.add(ci) };
    let mut chunk = unsafe { (*slot).load(Ordering::Acquire) };
    if chunk.is_null() {
        let fresh = Box::into_raw(Box::new(SlotChunk {
            slots: std::array::from_fn(|_| AtomicUsize::new(0)),
        }));
        match unsafe {
            (*slot).compare_exchange(
                std::ptr::null_mut(),
                fresh,
                Ordering::Release,
                Ordering::Acquire,
            )
        } {
            Ok(_) => chunk = fresh,
            Err(existing) => {
                unsafe { drop(Box::from_raw(fresh)) };
                chunk = existing;
            }
        }
    }
    unsafe {
        (*chunk).slots[index % SLOT_CHUNK].store(addr.0, Ordering::Release);
    }
}

fn table() -> parking_lot::MutexGuard<'static, Table> {
    let mut guard = TABLE.lock();
    if !READY.load(Ordering::Acquire) {
        if guard.slots.is_empty() {
            *guard = Table::new();
        }
        READY.store(true, Ordering::Release);
    }
    guard
}

/// Index of `addr` in the table. Null is 0. Same referent, same index.
///
/// intern is not a root: `history.py` `ConstPtr` dies with the holder.
/// The intern-to-holder window is [`Table::young`], traced by
/// [`collect_young_const_ptrs`] / [`trace_young_const_ptrs`].
pub fn intern(addr: GcRef) -> u32 {
    if addr.is_null() {
        return 0;
    }
    // Follow the stub before hashing. `minimark.py` `id_or_identityhash`
    // already follows it, so a find-hit is the dest slot; storing the
    // from-space address would `stamp_reuse` dest with a gap corpse
    // (`arena_reset` mode 0). `history.py` `ConstPtr.value` is the live
    // word.
    let addr = live_gcref(addr);
    crate::gcref_diag(addr.0, "intern-before-hash");
    let hash = gc_id_or_identityhash(addr.0) as u64;
    crate::gcref_diag(addr.0, "intern-after-hash");
    let mut guard = table();
    intern_under(&mut guard, addr, hash)
}

/// Intern `addr` and drain [`Table::young`] under one table lock.
///
/// A sibling `MiniMarkGC` minor in the same process would otherwise
/// `take_young` this slot before the caller traces it. Two live
/// collectors at once is not a supported configuration; cargo test
/// still starts many `MiniMarkGC` values in one process.
pub fn intern_and_take_young(addr: GcRef) -> (u32, Vec<u32>) {
    if addr.is_null() {
        let mut guard = table();
        return (0, take_young_under(&mut guard));
    }
    let addr = live_gcref(addr);
    crate::gcref_diag(addr.0, "intern-before-hash");
    let hash = gc_id_or_identityhash(addr.0) as u64;
    crate::gcref_diag(addr.0, "intern-after-hash");
    let mut guard = table();
    let idx = intern_under(&mut guard, addr, hash);
    let young = take_young_under(&mut guard);
    (idx, young)
}

fn intern_under(guard: &mut Table, addr: GcRef, hash: u64) -> u32 {
    let addr = live_gcref(addr);
    if let Some(idx) = find(guard, hash, addr.0) {
        stamp_reuse(guard, idx, addr, hash);
        note_young(guard, idx);
        return idx;
    }
    // `find` keys `by_hash`. A nursery intern that hashed the nursery
    // word (`id_or_identityhash_reentrant` / a busy allocator box)
    // records a different key than the old-gen intern of the same
    // object. `history.py` `ConstPtr` is one object: the live word
    // already in a slot is that intern (`same_constant` compares
    // `value`).
    if let Some(idx) = find_by_addr(guard, addr.0) {
        stamp_reuse(guard, idx, addr, hash);
        note_young(guard, idx);
        return idx;
    }
    let idx = if let Some(free_idx) = guard.free.pop() {
        let i = free_idx as usize;
        guard.slots[i] = addr;
        guard.hashes[i] = hash;
        guard.marks[i] = 0;
        guard.live[i] = 0;
        free_idx
    } else {
        let idx = guard.slots.len() as u32;
        guard.slots.push(addr);
        guard.hashes.push(hash);
        guard.marks.push(0);
        guard.live.push(0);
        idx
    };
    insert_pair(&mut guard.by_hash, hash, idx);
    insert_pair(&mut guard.by_addr, addr.0 as u64, idx);
    note_young(guard, idx);
    // Publish before releasing the intern lock so another thread that
    // finds this index cannot `resolve` a still-zero mirror.
    publish_slot(idx as usize, addr);
    idx
}

/// Open a forwarding wave. A second trace of the same slot in this
/// wave does not call the visitor. Drop restores the enclosing wave.
/// The id itself comes from [`NEXT_WAVE`]. A wrapping `u32` would
/// reissue an old generation while `marks` still holds it, and
/// [`claim_wave`] would skip that slot; the wrap path zeros `marks`
/// before publishing the next id. The collector holds one guard across
/// a root walk so `drag_out_root` writes each live `ConstPtr.value` once.
pub struct Wave {
    prev: u32,
}

impl Wave {
    pub fn enter() -> Self {
        let prev = WAVE.load(Ordering::Relaxed);
        let next = alloc_wave_id();
        WAVE.store(next, Ordering::Relaxed);
        Wave { prev }
    }
}

/// Publish the next nonzero generation.
///
/// `0` stays reserved for "no wave". The id `u32::MAX` is issued once;
/// the caller that then observes `0` clears `Table::marks` while holding
/// the table lock, then stores `1`. `claim_wave` takes that same lock, so
/// it cannot treat a reissued generation as already traced.
fn alloc_wave_id() -> u32 {
    loop {
        let cur = NEXT_WAVE.load(Ordering::Relaxed);
        if cur == 0 {
            let mut guard = table();
            if NEXT_WAVE.load(Ordering::Relaxed) == 0 {
                guard.marks.fill(0);
                NEXT_WAVE.store(1, Ordering::Relaxed);
            }
            continue;
        }
        let new = cur.wrapping_add(1);
        if NEXT_WAVE
            .compare_exchange(cur, new, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            return cur;
        }
    }
}

impl Drop for Wave {
    fn drop(&mut self) {
        WAVE.store(self.prev, Ordering::Release);
    }
}

/// Forward one live slot. Holders call this; the slot is not a root
/// merely because `intern` recorded it.
///
/// A nested visitor (one that collects) is skipped: the outer call is
/// already updating slots, matching [`walk`]'s `WALKING` guard.
pub fn trace_index(index: u32, visitor: &mut dyn FnMut(&mut GcRef)) {
    if index == 0 || walking() {
        return;
    }
    let idx = index as usize;
    let mut slot = {
        let mut guard = table();
        if idx >= guard.slots.len() || guard.slots[idx].is_null() {
            return;
        }
        if idx >= guard.live.len() {
            guard.live.resize(idx + 1, 0);
        }
        guard.live[idx] = current_live();
        if !claim_wave(&mut guard.marks, idx) {
            return;
        }
        guard.slots[idx]
    };
    struct Clear;
    impl Drop for Clear {
        fn drop(&mut self) {
            set_walking(false);
        }
    }
    set_walking(true);
    let _clear = Clear;
    visitor(&mut slot);
    let mut guard = table();
    if idx >= guard.slots.len() {
        return;
    }
    let old = guard.slots[idx].0 as u64;
    guard.slots[idx] = slot;
    let new = slot.0 as u64;
    if old != new {
        remove_pair(&mut guard.by_addr, old, index);
        insert_pair(&mut guard.by_addr, new, index);
    }
    publish_slot(idx, slot);
}

/// `true` when this wave has not yet traced `idx`. Wave 0 always claims.
fn claim_wave(marks: &mut Vec<u32>, idx: usize) -> bool {
    let wave = WAVE.load(Ordering::Relaxed);
    if wave == 0 {
        return true;
    }
    if idx >= marks.len() {
        marks.resize(idx + 1, 0);
    }
    if marks[idx] == wave {
        return false;
    }
    marks[idx] = wave;
    true
}

/// Write the forwarded address of an existing index.
///
/// `MIFrame.set_forwarded_ref_value` publishes one move into the box.
/// The identity hash recorded at intern time does not change.
pub fn set_slot(index: u32, addr: GcRef) {
    if index == 0 {
        return;
    }
    let mut guard = table();
    let i = index as usize;
    let Some(slot) = guard.slots.get_mut(i) else {
        return;
    };
    let old = slot.0 as u64;
    *slot = addr;
    if old != addr.0 as u64 {
        remove_pair(&mut guard.by_addr, old, index);
        insert_pair(&mut guard.by_addr, addr.0 as u64, index);
    }
    publish_slot(i, addr);
}

/// Current address of index `index`.
///
/// `history.py` `ConstPtr.getref_base` reads the box field. The mirror
/// slot is an atomic so this does not take the intern mutex.
#[inline]
pub fn resolve(index: u32) -> GcRef {
    if index == 0 {
        return GcRef::NULL;
    }
    let i = index as usize;
    let ci = i / SLOT_CHUNK;
    let dir = DIR.load(Ordering::Acquire);
    if dir.is_null() {
        return GcRef::NULL;
    }
    if ci >= unsafe { (*dir).n } {
        return GcRef::NULL;
    }
    let chunk = unsafe { (*(*dir).ptrs.add(ci)).load(Ordering::Acquire) };
    if chunk.is_null() {
        return GcRef::NULL;
    }
    GcRef(unsafe { (*chunk).slots[i % SLOT_CHUNK].load(Ordering::Acquire) })
}

/// Index whose current value is `addr`.
///
/// `intern` records one slot per referent. A root walk that has already
/// read `ConstPtr.value` uses this to name that slot without recomputing
/// `gc_id_or_identityhash` (a stale nursery address must not be hashed).
pub fn index_of_current(addr: GcRef) -> Option<u32> {
    if addr.is_null() {
        return None;
    }
    let guard = table();
    guard
        .slots
        .iter()
        .position(|slot| slot.0 == addr.0)
        .map(|idx| idx as u32)
        .filter(|idx| *idx != 0)
}

/// Forward every non-null slot. Nested walk (a visitor that collects)
/// is a no-op: the outer walk is already updating the slots.
pub fn walk(visitor: &mut dyn FnMut(&mut GcRef)) {
    if walking() {
        return;
    }
    struct Clear;
    impl Drop for Clear {
        fn drop(&mut self) {
            set_walking(false);
        }
    }
    set_walking(true);
    let _clear = Clear;
    let mut idx = 1usize;
    loop {
        let (index, mut slot) = {
            let mut guard = table();
            let found = loop {
                if idx >= guard.slots.len() {
                    break None;
                }
                if guard.slots[idx].is_null() || !claim_wave(&mut guard.marks, idx) {
                    idx += 1;
                    continue;
                }
                let slot = guard.slots[idx];
                let index = idx as u32;
                idx += 1;
                break Some((index, slot));
            };
            match found {
                Some(pair) => pair,
                None => return,
            }
        };
        visitor(&mut slot);
        let mut guard = table();
        let i = index as usize;
        if i < guard.slots.len() {
            let old = guard.slots[i].0 as u64;
            guard.slots[i] = slot;
            let new = slot.0 as u64;
            if old != new {
                remove_pair(&mut guard.by_addr, old, index);
                insert_pair(&mut guard.by_addr, new, index);
            }
            publish_slot(i, slot);
        }
    }
}

/// Drain [`Table::young`] and `trace_index` each slot.
///
/// Tests and a collector that owns every interned referent use this.
/// A MiniMark that shares the process-global table with another collector
/// uses [`take_owned_young`]: `incminimark.py` `collect_oldrefs_to_nursery`
/// walks one MiniMark's remembered set.
pub fn collect_young_const_ptrs(visitor: &mut dyn FnMut(&mut GcRef)) {
    let young = take_young();
    for idx in young {
        trace_index(idx, visitor);
    }
}

/// `trace_index` every young intern without draining the list.
///
/// Major seed / rescan: intern during incremental mark still has to sit
/// in the remembered set until the pre-sweep drain. `trace_live_numberings`
/// is the NUMBERING counterpart that does not clear `YOUNG_NUMBERINGS`.
pub fn trace_young_const_ptrs(visitor: &mut dyn FnMut(&mut GcRef)) {
    for idx in young_indexes() {
        trace_index(idx, visitor);
    }
}

/// Snapshot [`Table::young`] without draining it.
pub fn young_indexes() -> Vec<u32> {
    table().young.clone()
}

/// Drain [`Table::young`].
pub fn take_young() -> Vec<u32> {
    let mut guard = table();
    take_young_under(&mut guard)
}

fn take_young_under(guard: &mut Table) -> Vec<u32> {
    let taken = std::mem::take(&mut guard.young);
    for &idx in &taken {
        let i = idx as usize;
        if i < guard.young_mark.len() {
            guard.young_mark[i] = 0;
        }
    }
    taken
}

/// Remove from [`Table::young`] the indexes whose current value `belongs`
/// accepts. The rest stay queued. Holds the intern lock across the
/// partition so a concurrent MiniMark cannot drain a foreign intern
/// (`incminimark.py` `collect_oldrefs_to_nursery` is per collector).
///
/// `belongs` must not intern, trace, or collect: it runs under the intern lock.
pub fn take_owned_young(belongs: &mut dyn FnMut(GcRef) -> bool) -> Vec<u32> {
    let mut guard = table();
    let taken = take_young_under(&mut guard);
    let mut mine = Vec::new();
    for idx in taken {
        let addr = guard
            .slots
            .get(idx as usize)
            .copied()
            .unwrap_or(GcRef::NULL);
        if addr.is_null() {
            continue;
        }
        if belongs(addr) {
            mine.push(idx);
        } else {
            note_young(&mut guard, idx);
        }
    }
    mine
}

/// Return indexes to the intern-to-holder remembered set.
pub fn requeue_young(indexes: &[u32]) {
    if indexes.is_empty() {
        return;
    }
    let mut guard = table();
    for &idx in indexes {
        note_young(&mut guard, idx);
    }
}

fn note_young(guard: &mut Table, idx: u32) {
    if idx == 0 {
        return;
    }
    let i = idx as usize;
    if i >= guard.young_mark.len() {
        guard.young_mark.resize(i + 1, 0);
    }
    if guard.young_mark[i] != 0 {
        return;
    }
    guard.young_mark[i] = 1;
    guard.young.push(idx);
}

/// Free a slot no holder stamped in this major. `history.py` drops
/// the `ConstPtr` when no trace, descr, or op holds the box.
///
/// Call only after a complete holder census. A major that walked one
/// thread's extra areas would recycle another thread's live indexes.
pub fn sweep_untraced() {
    sweep_untraced_if(&mut |_| true);
}

/// [`sweep_untraced`] limited to referents `belongs` accepts.
///
/// A MiniMark major must not free another collector's interned slot.
/// `history.py` `ConstPtr` lives in that heap; the process-global table
/// is the `Copy` stand-in for the box field.
pub fn sweep_untraced_if(belongs: &mut dyn FnMut(GcRef) -> bool) {
    let live = current_live();
    let mut guard = table();
    let mut idx = 1usize;
    while idx < guard.slots.len() {
        if !guard.slots[idx].is_null() && guard.live[idx] != live && belongs(guard.slots[idx]) {
            free_slot(&mut guard, idx);
        }
        idx += 1;
    }
}

/// Drop intern slots whose live word `belongs` accepts.
///
/// `history.py` `ConstPtr` lives in the MiniMark heap that interned it.
/// The process-global table is the `Copy` stand-in for the box field, so
/// destroying that heap must drop those slots. A later MiniMark whose
/// nursery reuses the mapping would otherwise `drag_out` a leftover as
/// `collect_oldrefs_to_nursery`. Does not bump [`begin_major_live`] and
/// does not drain [`Table::young`] of other slots.
pub fn drop_slots_if(belongs: &mut dyn FnMut(GcRef) -> bool) {
    let mut guard = table();
    let mut idx = 1usize;
    while idx < guard.slots.len() {
        if !guard.slots[idx].is_null() && belongs(guard.slots[idx]) {
            free_slot(&mut guard, idx);
        }
        idx += 1;
    }
}

fn free_slot(guard: &mut Table, idx: usize) {
    let hash = guard.hashes[idx];
    remove_pair(&mut guard.by_hash, hash, idx as u32);
    remove_pair(&mut guard.by_addr, guard.slots[idx].0 as u64, idx as u32);
    guard.slots[idx] = GcRef::NULL;
    guard.hashes[idx] = 0;
    guard.marks[idx] = 0;
    guard.live[idx] = 0;
    if idx < guard.young_mark.len() {
        guard.young_mark[idx] = 0;
    }
    publish_slot(idx, GcRef::NULL);
    guard.free.push(idx as u32);
}

fn insert_pair(pairs: &mut Vec<(u64, u32)>, key: u64, idx: u32) {
    let pos = pairs.partition_point(|&(k, i)| (k, i) < (key, idx));
    pairs.insert(pos, (key, idx));
}

fn remove_pair(pairs: &mut Vec<(u64, u32)>, key: u64, idx: u32) {
    let start = pairs.partition_point(|&(k, _)| k < key);
    for i in start..pairs.len() {
        if pairs[i].0 != key {
            break;
        }
        if pairs[i].1 == idx {
            pairs.remove(i);
            return;
        }
    }
}

fn stamp_reuse(guard: &mut Table, idx: u32, addr: GcRef, hash: u64) {
    let i = idx as usize;
    // Find-hit is not a root. `history.py` `ConstPtr` lives only while a
    // holder traces `value`. The interned word is still the live address
    // (`same_constant` compares `value`); rewrite a from-space slot so
    // the next intern of this referent reuses this index. `intern`
    // already stored dest (`live_gcref`); a from-space intern must not
    // write the gap corpse over dest.
    if guard.slots[i] != addr {
        let old = guard.slots[i].0 as u64;
        guard.slots[i] = addr;
        publish_slot(i, addr);
        if old != addr.0 as u64 {
            remove_pair(&mut guard.by_addr, old, idx);
            insert_pair(&mut guard.by_addr, addr.0 as u64, idx);
        }
    }
    if guard.hashes.get(i).copied() != Some(hash) {
        retarget_hash(guard, idx, hash);
    }
}

fn retarget_hash(guard: &mut Table, idx: u32, new_hash: u64) {
    let i = idx as usize;
    let old = guard.hashes[i];
    if old == new_hash {
        return;
    }
    remove_pair(&mut guard.by_hash, old, idx);
    guard.hashes[i] = new_hash;
    insert_pair(&mut guard.by_hash, new_hash, idx);
}

fn find_by_addr(table: &Table, addr: usize) -> Option<u32> {
    let key = addr as u64;
    let start = table.by_addr.partition_point(|&(a, _)| a < key);
    for &(a, idx) in &table.by_addr[start..] {
        if a != key {
            break;
        }
        return Some(idx);
    }
    None
}

fn find(table: &Table, hash: u64, addr: usize) -> Option<u32> {
    let start = table.by_hash.partition_point(|&(h, _)| h < hash);
    let mut identity_hit = None;
    let mut identity_hits = 0u32;
    for &(h, idx) in &table.by_hash[start..] {
        if h != hash {
            break;
        }
        if table.slots.get(idx as usize).map(|s| s.0) == Some(addr) {
            return Some(idx);
        }
        identity_hits += 1;
        identity_hit = Some(idx);
    }
    // `id_or_identityhash` is unique per object. After a move the
    // live address misses the slot, but the recorded hash still names
    // the one intern. Several hits would be a hash collision.
    if identity_hits == 1 {
        identity_hit
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use parking_lot::Mutex;

    use super::*;

    /// `walk` claims every non-null slot for the current [`Wave`]
    /// (`claim_wave`). These tests share that process-global table, so a
    /// parallel `walk` inside another test's wave marks its slot,
    /// `trace_index` returns without the visitor, and `resolve` still
    /// returns the interned address.
    static TEST_SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn intern_is_stable_and_null_is_zero() {
        let _serial = TEST_SERIAL.lock();
        assert_eq!(intern(GcRef::NULL), 0);
        assert!(resolve(0).is_null());
        let a = GcRef(0x1111_0000);
        let i = intern(a);
        assert_eq!(intern(a), i);
        assert_eq!(resolve(i), a);
        let b = GcRef(0x2222_0000);
        assert_ne!(intern(b), i);
    }

    #[test]
    fn visitor_may_call_index_of_current_without_deadlocking() {
        let _serial = TEST_SERIAL.lock();
        let addr = GcRef(0x6E6A_DEAD_0100);
        let idx = intern(addr);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut found_trace = None;
            trace_index(idx, &mut |slot| {
                found_trace = index_of_current(*slot);
            });
            let mut found_walk = None;
            walk(&mut |slot| {
                if slot.0 == addr.0 {
                    found_walk = index_of_current(*slot);
                }
            });
            let _ = tx.send((found_trace, found_walk));
        });
        match rx.recv_timeout(std::time::Duration::from_secs(2)) {
            Ok((found_trace, found_walk)) => {
                assert_eq!(found_trace, Some(idx));
                assert_eq!(found_walk, Some(idx));
            }
            Err(_) => panic!("const_ptr_table visitor deadlock: index_of_current under TABLE"),
        }
    }

    #[test]
    fn intern_reuses_the_slot_when_the_referent_moved() {
        let _serial = TEST_SERIAL.lock();
        struct ResetHook;
        impl Drop for ResetHook {
            fn drop(&mut self) {
                crate::set_gc_id_or_identityhash(None);
            }
        }
        let _reset = ResetHook;
        fn stable_id(addr: usize) -> usize {
            if addr == 0x5555_0000 || addr == 0x5555_0080 {
                0x1D_0001
            } else {
                addr
            }
        }
        crate::set_gc_id_or_identityhash(Some(stable_id));
        let idx = intern(GcRef(0x5555_0000));
        assert_eq!(intern(GcRef(0x5555_0080)), idx);
        assert_eq!(resolve(idx), GcRef(0x5555_0080));
    }

    #[test]
    fn intern_of_a_from_space_address_does_not_overwrite_dest() {
        let _serial = TEST_SERIAL.lock();
        struct ResetHook;
        impl Drop for ResetHook {
            fn drop(&mut self) {
                crate::set_gc_id_or_identityhash(None);
                crate::set_gc_current_object_address(None);
            }
        }
        let _reset = ResetHook;
        const FROM: usize = 0x5555_4000;
        const TO: usize = 0x5555_4080;
        fn stable_id(addr: usize) -> usize {
            if addr == FROM || addr == TO {
                0x1D_0002
            } else {
                addr
            }
        }
        fn current(addr: usize) -> usize {
            if addr == FROM { TO } else { addr }
        }
        crate::set_gc_id_or_identityhash(Some(stable_id));
        crate::set_gc_current_object_address(Some(current));
        let idx = intern(GcRef(TO));
        assert_eq!(intern(GcRef(FROM)), idx);
        assert_eq!(resolve(idx), GcRef(TO));
        let idx_from = intern(GcRef(FROM));
        assert_eq!(idx_from, idx);
        assert_eq!(resolve(idx_from), GcRef(TO));
    }

    #[test]
    fn intern_reuses_the_slot_when_the_identity_hash_is_the_address() {
        let _serial = TEST_SERIAL.lock();
        // Default hook: identity is the address. A nursery intern then
        // an intern of the forwarded word must be one ConstPtr.
        let from = GcRef(0x5555_1000);
        let to = GcRef(0x5555_1080);
        let idx = intern(from);
        set_slot(idx, to);
        assert_eq!(intern(to), idx);
        assert_eq!(resolve(idx), to);
        assert_ne!(intern(GcRef(0x5555_2000)), idx);
    }

    #[test]
    fn intern_reuses_the_slot_after_walk_when_hashes_differ() {
        let _serial = TEST_SERIAL.lock();
        let from = GcRef(0x5555_3000);
        let to = GcRef(0x5555_3080);
        let idx = intern(from);
        walk(&mut |slot| {
            if slot.0 == from.0 {
                *slot = to;
            }
        });
        assert_eq!(intern(to), idx);
        assert_eq!(resolve(idx), to);
    }

    #[test]
    fn drop_slots_if_frees_matching_slots_and_leaves_the_rest() {
        let _serial = TEST_SERIAL.lock();
        let keep = intern(GcRef(0x6E6D_0000));
        let drop_me = intern(GcRef(0x6E6D_1000));
        drop_slots_if(&mut |g| g.0 == 0x6E6D_1000);
        assert_eq!(resolve(keep), GcRef(0x6E6D_0000));
        assert!(resolve(drop_me).is_null());
        intern(GcRef(0x6E6D_1000));
        drop_slots_if(&mut |g| (0x6E6D_0000..0x6E6D_2000).contains(&g.0));
    }

    #[test]
    fn walk_forwards_the_slot_the_index_names() {
        let _serial = TEST_SERIAL.lock();
        let addr = GcRef(0x3333_0000);
        let idx = intern(addr);
        walk(&mut |slot| {
            if slot.0 == addr.0 {
                *slot = GcRef(0x4444_0000);
            }
        });
        assert_eq!(resolve(idx), GcRef(0x4444_0000));
    }

    #[test]
    fn a_later_wave_forwards_a_slot_the_previous_wave_traced() {
        let _serial = TEST_SERIAL.lock();
        let first = GcRef(0x6E6A_ED90_0100);
        let idx = intern(first);
        {
            let _wave = Wave::enter();
            trace_index(idx, &mut |slot| {
                if slot.0 == first.0 {
                    *slot = GcRef(0x6E6A_ED90_0180);
                }
            });
        }
        {
            let _wave = Wave::enter();
            trace_index(idx, &mut |slot| {
                if slot.0 == 0x6E6A_ED90_0180 {
                    *slot = GcRef(0x6E6A_ED90_0200);
                }
            });
        }
        assert_eq!(resolve(idx), GcRef(0x6E6A_ED90_0200));
    }

    #[test]
    fn wrapped_wave_id_does_not_skip_a_slot_marked_with_that_generation() {
        let _serial = TEST_SERIAL.lock();
        let addr = GcRef(0x6E6A_ED90_0A01);
        let idx = intern(addr);
        {
            let mut guard = table();
            let slot = idx as usize;
            if slot >= guard.marks.len() {
                guard.marks.resize(slot + 1, 0);
            }
            // Generation 1 is the id the wrap path publishes next.
            guard.marks[slot] = 1;
        }
        NEXT_WAVE.store(0, Ordering::Relaxed);
        {
            let _wave = Wave::enter();
            let mut visited = false;
            trace_index(idx, &mut |slot| {
                assert_eq!(slot.0, addr.0);
                visited = true;
            });
            assert!(visited);
        }
    }

    /// Sweep unstamped slots whose live word is in `lo..=hi`.
    ///
    /// `incminimark.py` `mass_free_incremental` and
    /// `free_unvisited_rawmalloc_objects_step` free objects in this
    /// MiniMark's arenas. `sweep_untraced_if` is the table counterpart.
    /// The table is process-global, so `sweep_untraced` would recycle a
    /// foreign intern still held in an `OpRef`.
    fn sweep_untraced_owned(lo: usize, hi: usize) {
        sweep_untraced_if(&mut |g| (lo..=hi).contains(&g.0));
    }

    /// Close this test's intern-to-holder window and free its unstamped
    /// slots. `take_owned_young` is `collect_oldrefs_to_nursery` for one
    /// MiniMark; a foreign intern stays queued.
    fn reclaim_unheld(lo: usize, hi: usize) {
        let young = take_owned_young(&mut |g| (lo..=hi).contains(&g.0));
        for idx in young {
            trace_index(idx, &mut |_| {});
        }
        begin_major_live();
        sweep_untraced_owned(lo, hi);
    }

    #[test]
    fn intern_grows_past_the_old_chunk_directory_cap() {
        let _serial = TEST_SERIAL.lock();
        const N: usize = SLOT_CHUNK * 1024 + 1;
        let base = 0x6E6B_C100usize;
        let last_word = base + (N - 1) * 16;
        reclaim_unheld(base, last_word);
        let first = intern(GcRef(base));
        for i in 1..N {
            intern(GcRef(base + i * 16));
        }
        let last_addr = GcRef(last_word);
        let last = intern(last_addr);
        assert_eq!(resolve(first), GcRef(base));
        assert_eq!(resolve(last), last_addr);
        assert_ne!(first, last);
        reclaim_unheld(base, last_word);
    }

    #[test]
    fn intern_reuses_slots_across_major_sweeps_past_the_old_cap() {
        let _serial = TEST_SERIAL.lock();
        const BATCH: usize = 80_000;
        const ROUNDS: usize = 4;
        let base = 0x6E6B_D000usize;
        let last_word = base + (ROUNDS - 1) * 1_000_000 + (BATCH - 1) * 16;
        reclaim_unheld(base, last_word);
        let mut lifetime = 0usize;
        for round in 0..ROUNDS {
            begin_major_live();
            for i in 0..BATCH {
                intern(GcRef(base + round * 1_000_000 + i * 16));
            }
            sweep_untraced_owned(base, last_word);
            lifetime += BATCH;
            begin_major_live();
            sweep_untraced_owned(base, last_word);
        }
        assert!(
            lifetime > SLOT_CHUNK * 1024,
            "lifetime interned {lifetime} must exceed the old 256×1024 cap"
        );
        let probe = intern(GcRef(0x6E6B_E001));
        assert_eq!(resolve(probe), GcRef(0x6E6B_E001));
        reclaim_unheld(base, last_word);
    }

    #[test]
    fn a_held_slot_survives_a_major_and_keeps_its_index() {
        let _serial = TEST_SERIAL.lock();
        let addr = GcRef(0x6E6B_F010);
        reclaim_unheld(addr.0, addr.0);
        let idx = intern(addr);
        begin_major_live();
        trace_index(idx, &mut |_| {});
        sweep_untraced_owned(addr.0, addr.0);
        assert_eq!(resolve(idx), addr);
        assert_eq!(intern(addr), idx);
        reclaim_unheld(addr.0, addr.0);
    }

    #[test]
    fn intern_find_hit_does_not_keep_a_slot_without_trace_index() {
        let _serial = TEST_SERIAL.lock();
        let addr = GcRef(0x6E6B_F110);
        reclaim_unheld(addr.0, addr.0);
        let idx = intern(addr);
        begin_major_live();
        assert_eq!(intern(addr), idx);
        sweep_untraced_owned(addr.0, addr.0);
        assert!(
            resolve(idx).is_null(),
            "find-hit intern is not a holder; sweep must free the slot"
        );
        reclaim_unheld(addr.0, addr.0);
    }

    #[test]
    fn intern_find_hit_survives_when_the_young_list_is_walked() {
        let _serial = TEST_SERIAL.lock();
        let addr = GcRef(0x6E6B_F118);
        reclaim_unheld(addr.0, addr.0);
        let idx = intern(addr);
        begin_major_live();
        assert_eq!(intern(addr), idx);
        let young = take_owned_young(&mut |g| g.0 == addr.0);
        for young_idx in young {
            trace_index(young_idx, &mut |_| {});
        }
        sweep_untraced_owned(addr.0, addr.0);
        assert_eq!(resolve(idx), addr);
        assert_eq!(intern(addr), idx);
        reclaim_unheld(addr.0, addr.0);
    }

    /// `collect_oldrefs_to_nursery` is per MiniMark. A collector that
    /// does not own the referent must leave the intern queued for the
    /// MiniMark that allocated it.
    #[test]
    fn collect_young_leaves_slots_the_collector_does_not_own() {
        let _serial = TEST_SERIAL.lock();
        reclaim_unheld(0x6E6C_0010, 0x6E6C_0020);
        let mine = intern(GcRef(0x6E6C_0010));
        let other = intern(GcRef(0x6E6C_0020));
        let mine_idxs = take_owned_young(&mut |g| g.0 == 0x6E6C_0010);
        for idx in mine_idxs {
            trace_index(idx, &mut |slot| {
                if slot.0 == 0x6E6C_0010 {
                    *slot = GcRef(0x6E6C_0018);
                }
            });
        }
        assert_eq!(resolve(mine), GcRef(0x6E6C_0018));
        assert_eq!(resolve(other), GcRef(0x6E6C_0020));
        assert!(
            young_indexes().contains(&other),
            "the foreign intern must stay in the remembered set"
        );
        reclaim_unheld(0x6E6C_0010, 0x6E6C_0020);
    }

    /// A holder that copies the index (`OpRef` is `Copy`) and never calls
    /// `trace_index` loses the slot. `ResumeDataDirectReader::decode_ref`
    /// then sees `ConstPtr.getref_base` as null. `walk_rd_consts_refs` is
    /// the extra-root walk that stamps `ResumeGuardDescr.rd_consts`.
    #[test]
    fn a_copied_index_without_trace_index_is_freed_at_major() {
        let _serial = TEST_SERIAL.lock();
        let addr = GcRef(0x6E6B_F210);
        reclaim_unheld(addr.0, addr.0);
        let holder = intern(addr);
        begin_major_live();
        sweep_untraced_owned(addr.0, addr.0);
        assert!(
            resolve(holder).is_null(),
            "unwalked holder must not keep the slot"
        );
        reclaim_unheld(addr.0, addr.0);
    }
}
