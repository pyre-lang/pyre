//! RPython parity module for `rpython/jit/metainterp/counter.py`.

use crate::warmstate::BaseJitCell;
use majit_ir::IndexMapExt;
use std::cell::{Ref, RefCell};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// counter.py: JitCounter — float-based 5-way associative timetable.
///
/// Direct port of rpython/jit/metainterp/counter.py.
/// Uses f32 time values (0.0 to 1.0) instead of integer counts.
/// tick(hash, increment) adds increment; fires when >= 1.0.
/// 5-way associative cache indexed by _get_index(hash), matched by
/// _get_subhash(hash). MRU promotion via _swap.
///
/// counter.py DEFAULT_SIZE = 2048
pub const DEFAULT_SIZE: usize = 2048;

/// counter.py ENTRY: 5 (f32 time, u16 subhash) pairs per bucket.
const ASSOCIATIVITY: usize = 5;

/// counter.py UINT32MAX = 2 ** 32 - 1
const UINT32MAX: u64 = 0xFFFF_FFFF;

static MINOR_COLLECTION_STEP: AtomicUsize = AtomicUsize::new(0);
static DECAY_GENERATION: AtomicUsize = AtomicUsize::new(0);

/// counter.py invoke_after_minor_collection
///
/// This runs inside a minor collection, so it must remain allocation-free and
/// must not touch the counter table or acquire a lock.
fn invoke_after_minor_collection() {
    let step = MINOR_COLLECTION_STEP.fetch_add(1, Ordering::Relaxed) + 1;
    if step == 32 {
        MINOR_COLLECTION_STEP.store(0, Ordering::Relaxed);
        DECAY_GENERATION.fetch_add(1, Ordering::Relaxed);
    }
}

/// One timetable entry: 5-way associative (time, subhash) pairs.
/// counter.py ENTRY struct.
#[derive(Clone)]
struct Entry {
    /// counter.py: times — f32 timing values, 0.0 to 1.0.
    times: [f32; ASSOCIATIVITY],
    /// counter.py: subhashes — lower 16 bits of the hash.
    subhashes: [u16; ASSOCIATIVITY],
}

impl Default for Entry {
    fn default() -> Self {
        Entry {
            times: [0.0; ASSOCIATIVITY],
            subhashes: [0; ASSOCIATIVITY],
        }
    }
}

/// Timetable body of `counter.py JitCounter`.
struct JitCounterInner {
    /// counter.py `JitCounter.__init__` `self.size`
    size: usize,
    /// counter.py `JitCounter.__init__` `self.shift`
    shift: u32,
    /// counter.py `JitCounter.__init__` `self.timetable`
    timetable: Vec<Entry>,
    /// counter.py `JitCounter.__init__` `self._nexthash`
    _nexthash: u64,
    /// counter.py `JitCounter.set_decay` `decay_by_mult` — f64 (Python float).
    decay_by_mult: f64,
    /// Last `DECAY_GENERATION` this counter applied. Each counter tracks its
    /// own, so one thread's tick cannot consume another counter's decay.
    last_decay_generation: usize,
    /// counter.py `JitCounter.__init__` `self.celltable = [None] * size` —
    /// the table of JitCell entries, recording already-compiled loops.
    /// Each slot holds the HEAD of a linked list of cells; walk
    /// `BaseJitCell::next` for the rest of a chain.
    ///
    /// The slot is `_get_index(hash)`, which "truncates the hash to 32 bits,
    /// and then keep the *highest* remaining bits" — the same call the
    /// timetable is indexed with. Truncation means one slot collects every
    /// green key whose hash agrees in those bits, not only the keys whose
    /// hashes are equal, so a chain mixes unrelated green keys by design;
    /// `counter.py` calls the result "non-lossy" because nothing is evicted
    /// to make room, not because a slot belongs to one key. Every driver's
    /// `WarmEnterState` files its cells here (`warmspot.py`
    /// `WarmRunnerDesc.jitcounter` is one object on the runner); a driver
    /// tells its own cells apart by [`BaseJitCell::jitdriver_sd`], the
    /// `isinstance(cell, JitCell)` of `warmstate.py` `get_jitcell`.
    ///
    /// A cell inside a chain is named by its own [`BaseJitCell::cell_key`],
    /// never by the slot. Upstream separates a slot's occupants with
    /// `cell.comparekey(*greenargs)` (warmstate.py), which needs the greens;
    /// a reader holding only a `u64` separates them with the cell key, because
    /// a cell key still carries the full hash the index truncation dropped.
    celltable: Vec<Option<Rc<BaseJitCell>>>,
    /// Minted cell keys → the raw bucket hash the cell lives in.
    ///
    /// Only cells that could not take their bucket's raw hash appear here, so
    /// this map is empty for every workload that never chains a bucket, and
    /// [`JitCounter::bucket_of`] short-circuits on `is_empty()`. The
    /// invariant it maintains is: **`bucket_of(cell_key)` is the bucket the
    /// cell lives in** — trivially `cell_key` itself for an unminted key,
    /// because an unminted key is only ever assigned inside the bucket whose
    /// hash it equals.
    ///
    /// Upstream has no such map: it hands the cell object itself on and never
    /// re-derives it from a number. The `u64` cell-key currency pyre's callers
    /// hold is what this and [`JitCounterInner::mint_serial`] serve.
    minted: crate::FxIndexMap<u64, u64>,
    /// Serial feeding [`JitCounterInner::mint_cell_key`]'s candidate sequence.
    mint_serial: u64,
}

/// counter.py JitCounter
///
/// `Clone` is another handle to the same timetable: `warmspot.py`
/// `WarmRunnerDesc.jitcounter` is one object on the runner, and every
/// `WarmEnterState` reads it (`warmstate.py` `_compute_threshold`).
#[derive(Clone)]
pub struct JitCounter {
    inner: Rc<RefCell<JitCounterInner>>,
}

impl JitCounterInner {
    /// counter.py __init__(self, size=DEFAULT_SIZE, translator=None)
    fn new(size: usize) -> Self {
        majit_gc::register_after_minor_collection_hook(invoke_after_minor_collection);
        let mut shift = 16u32;
        while (UINT32MAX >> shift) != (size as u64 - 1) {
            shift += 1;
            assert!(shift < 999, "size is not a power of two <= 2**16");
        }
        JitCounterInner {
            size,
            shift,
            timetable: vec![Entry::default(); size],
            _nexthash: 0,
            decay_by_mult: 1.0,
            last_decay_generation: DECAY_GENERATION.load(Ordering::Relaxed),
            celltable: std::iter::repeat_with(|| None).take(size).collect(),
            minted: crate::FxIndexMap::default(),
            mint_serial: 0,
        }
    }

    /// counter.py compute_threshold
    pub fn compute_threshold(&self, threshold: u32) -> f64 {
        if threshold == 0 {
            return 0.0;
        }
        1.0_f64 / (threshold as f64 - 0.001)
    }

    /// counter.py `self.size = size` — the entry count both tables are
    /// indexed with. `counter.py JitCounter.__init__` sizes the timetable and the celltable
    /// from this one number, so a table living outside this struct has to read
    /// it from here rather than pick its own.
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.size
    }

    /// counter.py _get_index
    ///
    /// ```text
    ///  def _get_index(self, hash):
    ///      hash32 = r_uint(r_uint32(hash))  # mask off the bits higher than 32
    ///      index = hash32 >> self.shift     # shift, resulting in a value < size
    ///      return index                     # return the result as a r_uint
    /// ```
    ///
    /// Public because the timetable is not the only table this indexes:
    /// `counter.py` reads the celltable through the same call, and the
    /// two tables must agree about which entry a hash names.
    #[inline(always)]
    pub fn _get_index(&self, hash: u64) -> usize {
        let hash32 = hash as u32 as u64;
        (hash32 >> self.shift) as usize
    }

    /// counter.py _get_subhash
    #[inline(always)]
    fn _get_subhash(hash: u64) -> u16 {
        (hash & 0xFFFF) as u16
    }

    /// counter.py fetch_next_hash
    pub fn fetch_next_hash(&mut self) -> u64 {
        let result = self._nexthash;
        self._nexthash =
            result.wrapping_add(1 | (1u64 << self.shift) | (1u64 << (self.shift - 16)));
        result
    }

    /// counter.py _swap
    #[inline(always)]
    fn _swap(entry: &mut Entry, n: usize) -> usize {
        if entry.times[n] > entry.times[n + 1] {
            n + 1
        } else {
            entry.times.swap(n, n + 1);
            entry.subhashes.swap(n, n + 1);
            n
        }
    }

    /// counter.py _tick_slowpath
    fn _tick_slowpath(entry: &mut Entry, subhash: u16) -> usize {
        if entry.subhashes[1] == subhash {
            Self::_swap(entry, 0)
        } else if entry.subhashes[2] == subhash {
            Self::_swap(entry, 1)
        } else if entry.subhashes[3] == subhash {
            Self::_swap(entry, 2)
        } else if entry.subhashes[4] == subhash {
            Self::_swap(entry, 3)
        } else {
            let mut n = 4;
            while n > 0 && entry.times[n - 1] == 0.0 {
                n -= 1;
            }
            entry.subhashes[n] = subhash;
            entry.times[n] = 0.0;
            n
        }
    }

    /// Test-only read of what `tick` would answer, without counting.
    #[cfg(test)]
    pub fn would_tick_fire(&self, hash: u64, increment: f64) -> bool {
        let elapsed = DECAY_GENERATION
            .load(Ordering::Relaxed)
            .wrapping_sub(self.last_decay_generation);
        let index = self._get_index(hash);
        let subhash = Self::_get_subhash(hash);
        let entry = &self.timetable[index];
        for i in 0..ASSOCIATIVITY {
            if entry.subhashes[i] == subhash {
                // This predicate is &self, so decay-adjust the read instead of
                // mutating the table to apply the pending generations. Step
                // through them one at a time rather than raising the multiplier
                // to `elapsed`: every step rounds back to f32, and this answer
                // has to be the one a `tick` would give once it drains the same
                // generations. Narrow the multiplier the way decay_all_counters
                // does, for the same reason.
                let mult = self.decay_by_mult as f32;
                let mut time = entry.times[i];
                for _ in 0..elapsed {
                    time *= mult;
                }
                return time as f64 + increment >= 1.0;
            }
        }
        increment >= 1.0
    }

    /// counter.py tick(self, hash, increment)
    #[inline(always)]
    pub fn tick(&mut self, hash: u64, increment: f64) -> bool {
        // counter.py `invoke_after_minor_collection` applies the decay synchronously inside the minor
        // collection, where the hook closes over the process's one JitCounter.
        // pyre defers it to the next table access instead: the counter is
        // reached through the `JIT_DRIVER` cell in eval.rs, whose accessor
        // mints a `&'static mut JitDriverPair`, and a minor collection can be
        // triggered by an allocation the metainterp makes while already holding
        // one. Decaying from inside the collector would alias it.
        //
        // Each JitCounter keeps its own last-seen generation because JIT_DRIVER
        // is thread-local, giving each mutator thread its own counter.
        // Every elapsed interval is applied before every mutating table access.
        // A value written after a collection is therefore not retro-decayed.
        self.apply_pending_decay();

        let index = self._get_index(hash);
        let subhash = Self::_get_subhash(hash);
        let entry = &mut self.timetable[index];

        let n = if entry.subhashes[0] == subhash {
            0
        } else {
            Self::_tick_slowpath(entry, subhash)
        };

        // counter.py `JitCounter.tick`: counter = float(p_entry.times[n]) + increment
        let counter: f64 = entry.times[n] as f64 + increment;
        if counter < 1.0 {
            // counter.py `JitCounter.tick`: p_entry.times[n] = r_singlefloat(counter)
            entry.times[n] = counter as f32;
            false
        } else {
            // counter.py: self.reset(hash); return True
            self.reset(hash);
            true
        }
    }

    /// counter.py change_current_fraction(hash, new_fraction)
    pub fn change_current_fraction(&mut self, hash: u64, new_fraction: f64) {
        self.apply_pending_decay();

        let index = self._get_index(hash);
        let subhash = Self::_get_subhash(hash);
        let entry = &mut self.timetable[index];

        let mut n = 0;
        while n < 4 && entry.subhashes[n] != subhash && entry.times[n] != 0.0 {
            n += 1;
        }
        while n > 0 {
            n -= 1;
            entry.subhashes[n + 1] = entry.subhashes[n];
            entry.times[n + 1] = entry.times[n];
        }
        entry.subhashes[0] = subhash;
        entry.times[0] = new_fraction as f32;
    }

    /// counter.py reset(hash)
    pub fn reset(&mut self, hash: u64) {
        self.apply_pending_decay();

        let index = self._get_index(hash);
        let subhash = Self::_get_subhash(hash);
        let entry = &mut self.timetable[index];
        for i in 0..ASSOCIATIVITY {
            if entry.subhashes[i] == subhash {
                entry.times[i] = 0.0;
            }
        }
    }

    /// counter.py set_decay(decay)
    pub fn set_decay(&mut self, decay: i32) {
        self.apply_pending_decay();

        let clamped = decay.clamp(0, 1000);
        self.decay_by_mult = 1.0_f64 - (clamped as f64 * 0.001);
    }

    /// Inverse of [`Self::set_decay`] for `set_param(None)` inherit.
    pub fn decay(&self) -> i32 {
        ((1.0 - self.decay_by_mult) / 0.001).round() as i32
    }

    /// counter.py decay_all_counters()
    ///
    /// counter.py hands `decay_by_mult` to `pypy__decay_jit_counters`,
    /// whose C body narrows it with `float f = (float)f1` once and then
    /// multiplies each entry in single precision. Narrowing the multiplier here
    /// rather than the product keeps those bits: widening the entry to f64,
    /// multiplying, and narrowing back rounds twice, and the two disagree
    /// whenever the exact product lands near an f32 tie.
    pub fn decay_all_counters(&mut self) {
        let mult = self.decay_by_mult as f32;
        for entry in &mut self.timetable {
            for time in &mut entry.times {
                *time *= mult;
            }
        }
    }

    /// Apply every 32-collection interval that elapsed since this counter
    /// last looked. counter.py `invoke_after_minor_collection` decays inside the collection, so a
    /// pending decay must land before anything reads or writes the table —
    /// otherwise it would decay values written after the collection.
    fn apply_pending_decay(&mut self) {
        let generation = DECAY_GENERATION.load(Ordering::Relaxed);
        while self.last_decay_generation != generation {
            self.last_decay_generation = self.last_decay_generation.wrapping_add(1);
            self.decay_all_counters();
        }
    }
    /// counter.py lookup_chain(hash)
    ///
    /// ```text
    ///  def lookup_chain(self, hash):
    ///      return self.celltable[self._get_index(hash)]
    /// ```
    ///
    /// One load of the slot. Callers that need an owned `Rc` clone it;
    /// `warmstate.py maybe_compile_and_run` keeps the pointer.
    #[inline(always)]
    fn lookup_chain(&self, hash: u64) -> &Option<Rc<BaseJitCell>> {
        &self.celltable[self._get_index(hash)]
    }

    /// counter.py install_new_cell(hash, newcell)
    ///
    /// ```text
    ///  def install_new_cell(self, hash, newcell):
    ///      index = self._get_index(hash)
    ///      cell = self.celltable[index]
    ///      keep = newcell
    ///      while cell is not None:
    ///          nextcell = cell.next
    ///          if not cell.should_remove_jitcell():
    ///              cell.next = keep
    ///              keep = cell
    ///          cell = nextcell
    ///      self.celltable[index] = keep
    /// ```
    ///
    /// Before the relink, a new cell is named: it takes the bucket's raw hash
    /// as its cell key when no live cell holds that key, and a minted one
    /// otherwise ([`Self::mint_cell_key`]). Upstream needs no equivalent
    /// because it hands the cell object itself on and never re-derives it
    /// from a number.
    fn install_new_cell(&mut self, hash: u64, newcell: Option<Rc<BaseJitCell>>) {
        if let Some(cell) = &newcell {
            if cell.cell_key.get().is_none() {
                let cell_key = if self.cell_by_key(cell.jitdriver_sd, hash).is_none() {
                    hash
                } else {
                    self.mint_cell_key(hash)
                };
                cell.cell_key.set(Some(cell_key));
            }
            // The slot being filed IS the bucket, whichever of the two keys
            // above the cell ended up with. Only the newly filed cell gets it:
            // the chain members relinked below keep their own, which is what
            // the truncated index put them here in spite of.
            cell.cell_bucket.set(hash);
        }
        let index = self._get_index(hash);
        let mut cell = self.celltable[index].take();
        let mut keep = newcell;
        while let Some(c) = cell {
            let nextcell = c.next.take();
            if !c.should_remove_jitcell() {
                *c.next.borrow_mut() = keep;
                keep = Some(c);
            } else {
                self.forget_cell_key(&c);
            }
            cell = nextcell;
        }
        self.celltable[index] = keep;
    }

    /// counter.py cleanup_chain(hash)
    ///
    /// ```text
    ///  def cleanup_chain(self, hash):
    ///      self.reset(hash)
    ///      self.install_new_cell(hash, None)
    /// ```
    fn cleanup_chain(&mut self, hash: u64) {
        self.reset(hash);
        self.install_new_cell(hash, None);
    }

    /// The bucket a cell key lives in.
    ///
    /// An unminted cell key IS its bucket's raw hash (it is only ever handed
    /// out inside that bucket), so the common path costs one `is_empty()`
    /// test and no lookup at all.
    #[inline]
    fn bucket_of(&self, cell_key: u64) -> u64 {
        if self.minted.is_empty() {
            return cell_key;
        }
        self.minted.get(&cell_key).copied().unwrap_or(cell_key)
    }

    /// The cell of driver `jitdriver_sd` named by `cell_key`, or `None` if
    /// that driver holds no cell under the key.
    ///
    /// A key is the hash-form of a driver's greens, so it names a cell only
    /// together with the driver — `isinstance(cell, JitCell)` again. Two
    /// drivers may file a cell under the same raw hash; a minted key is
    /// unique across drivers ([`Self::mint_cell_key`]).
    fn cell_by_key(&self, jitdriver_sd: usize, cell_key: u64) -> Option<Rc<BaseJitCell>> {
        let mut cell = self.lookup_chain(self.bucket_of(cell_key)).clone();
        while let Some(c) = cell {
            if c.jitdriver_sd == jitdriver_sd && c.cell_key.get() == Some(cell_key) {
                return Some(c);
            }
            cell = c.next.borrow().clone();
        }
        None
    }

    /// Whether any driver's cell holds `cell_key`.
    fn key_is_held(&self, cell_key: u64) -> bool {
        let mut cell = self.lookup_chain(self.bucket_of(cell_key)).clone();
        while let Some(c) = cell {
            if c.cell_key.get() == Some(cell_key) {
                return true;
            }
            cell = c.next.borrow().clone();
        }
        false
    }

    /// Mint a cell key for a cell whose bucket's raw hash is already taken.
    ///
    /// The candidate must name no live cell and no live minted key. A
    /// candidate equal to some OTHER bucket's raw hash is admissible: that
    /// hash may later be claimed by a green key, which then finds its raw
    /// hash taken, sees no typed match, and mints in turn, so the two stay
    /// distinct cells.
    fn mint_cell_key(&mut self, bucket: u64) -> u64 {
        loop {
            self.mint_serial = self.mint_serial.wrapping_add(1);
            // The serial must reach the fold as the VALUE, not pre-mixed into
            // the accumulator: `green_uhash_step(bucket ^ serial, Int, serial)`
            // folds `(bucket ^ serial) ^ serial`, which is `bucket` again, so
            // every retry re-proposes one number and the second mint in a
            // bucket cannot terminate. Folded here, distinct serials give
            // distinct candidates — xor and the odd multiplier are both
            // bijections — so the retry walks a fresh number each time and the
            // finite live set bounds it.
            let candidate = majit_ir::green_uhash_step(
                bucket,
                majit_ir::GreenType::Int,
                self.mint_serial as i64,
            );
            if candidate != bucket
                && !self.key_is_held(candidate)
                && !self.minted.contains_key(&candidate)
            {
                self.minted.insert(candidate, bucket);
                return candidate;
            }
        }
    }

    /// Retire a dropped cell's minted key so [`Self::mint_cell_key`] may reuse
    /// the number and `bucket_of` stops answering for a cell that is gone.
    ///
    /// Unminted keys need no retiring: they equal their bucket hash, so they
    /// are recomputable from greens and are re-taken by the next cell that
    /// installs into an empty bucket.
    fn forget_cell_key(&mut self, cell: &BaseJitCell) {
        if let Some(key) = cell.cell_key.get()
            && !self.minted.is_empty()
        {
            self.minted.swap_remove(&key);
        }
    }

    /// `install_new_cell(hash, None)` over every slot: drop each cell whose
    /// `should_remove_jitcell` answers true. Returns how many were dropped.
    fn gc_cells(&mut self) -> usize {
        let mut removed = 0;
        for index in 0..self.celltable.len() {
            let mut cell = self.celltable[index].take();
            let mut keep = None;
            while let Some(c) = cell {
                let nextcell = c.next.take();
                if !c.should_remove_jitcell() {
                    *c.next.borrow_mut() = keep;
                    keep = Some(c);
                } else {
                    removed += 1;
                    self.forget_cell_key(&c);
                }
                cell = nextcell;
            }
            self.celltable[index] = keep;
        }
        removed
    }
}

impl JitCounter {
    /// counter.py __init__(self, size=DEFAULT_SIZE, translator=None)
    pub fn new(size: usize) -> Self {
        JitCounter {
            inner: Rc::new(RefCell::new(JitCounterInner::new(size))),
        }
    }

    /// True when `other` is the same timetable object.
    /// `warmspot.py` `WarmRunnerDesc.jitcounter` is one object on the runner.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }

    /// counter.py compute_threshold
    pub fn compute_threshold(&self, threshold: u32) -> f64 {
        self.inner.borrow().compute_threshold(threshold)
    }

    /// counter.py `self.size = size`
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.inner.borrow().size()
    }

    /// counter.py _get_index
    #[inline(always)]
    pub fn _get_index(&self, hash: u64) -> usize {
        self.inner.borrow()._get_index(hash)
    }

    /// counter.py fetch_next_hash
    pub fn fetch_next_hash(&mut self) -> u64 {
        self.inner.borrow_mut().fetch_next_hash()
    }

    /// counter.py _swap
    #[inline(always)]
    fn _swap(entry: &mut Entry, n: usize) -> usize {
        JitCounterInner::_swap(entry, n)
    }

    /// Test-only read of what `tick` would answer, without counting.
    #[cfg(test)]
    pub fn would_tick_fire(&self, hash: u64, increment: f64) -> bool {
        self.inner.borrow().would_tick_fire(hash, increment)
    }

    /// counter.py tick(self, hash, increment)
    #[inline(always)]
    pub fn tick(&mut self, hash: u64, increment: f64) -> bool {
        self.inner.borrow_mut().tick(hash, increment)
    }

    /// counter.py change_current_fraction(hash, new_fraction)
    pub fn change_current_fraction(&mut self, hash: u64, new_fraction: f64) {
        self.inner
            .borrow_mut()
            .change_current_fraction(hash, new_fraction)
    }

    /// counter.py reset(hash)
    pub fn reset(&mut self, hash: u64) {
        self.inner.borrow_mut().reset(hash)
    }

    /// counter.py set_decay(decay)
    pub fn set_decay(&mut self, decay: i32) {
        self.inner.borrow_mut().set_decay(decay)
    }

    /// Inverse of [`Self::set_decay`] for `set_param(None)` inherit.
    pub fn decay(&self) -> i32 {
        self.inner.borrow().decay()
    }

    /// counter.py decay_all_counters()
    pub fn decay_all_counters(&mut self) {
        self.inner.borrow_mut().decay_all_counters()
    }

    /// counter.py lookup_chain(hash) — the head of the chain at `hash`'s
    /// table slot; walk `.next` for the rest.
    ///
    /// A `Ref` of the table entry, matching the pointer load
    /// `return self.celltable[self._get_index(hash)]`. The compiled portal
    /// (`warmstate.py maybe_compile_and_run`) keeps that borrow; callers that
    /// need an owned `Rc` clone it from the `Ref`.
    #[inline(always)]
    pub fn lookup_chain(&self, hash: u64) -> Option<Ref<'_, Rc<BaseJitCell>>> {
        Ref::filter_map(self.inner.borrow(), |inner| {
            inner.lookup_chain(hash).as_ref()
        })
        .ok()
    }

    /// counter.py install_new_cell(hash, newcell).
    pub fn install_new_cell(&self, hash: u64, newcell: Option<Rc<BaseJitCell>>) {
        self.inner.borrow_mut().install_new_cell(hash, newcell)
    }

    /// counter.py cleanup_chain(hash).
    pub fn cleanup_chain(&self, hash: u64) {
        self.inner.borrow_mut().cleanup_chain(hash)
    }

    /// The bucket a cell key lives in; see `JitCounterInner::bucket_of`.
    #[inline]
    pub fn bucket_of(&self, cell_key: u64) -> u64 {
        self.inner.borrow().bucket_of(cell_key)
    }

    /// The cell of driver `jitdriver_sd` named by `cell_key`.
    #[inline]
    pub fn cell_by_key(&self, jitdriver_sd: usize, cell_key: u64) -> Option<Rc<BaseJitCell>> {
        self.inner.borrow().cell_by_key(jitdriver_sd, cell_key)
    }

    /// Drop every cell whose `should_remove_jitcell` answers true.
    pub fn gc_cells(&self) -> usize {
        self.inner.borrow_mut().gc_cells()
    }

    /// Visit every cell of every chain. `f` runs under the table borrow, so
    /// it must not reach back into this counter.
    pub fn for_each_cell(&self, mut f: impl FnMut(&Rc<BaseJitCell>)) {
        let inner = self.inner.borrow();
        for slot in &inner.celltable {
            let mut cur = slot.clone();
            while let Some(cell) = cur {
                f(&cell);
                cur = cell.next.borrow().clone();
            }
        }
    }

    /// How many table slots hold a chain at all. The table is sized once, so
    /// `celltable.len()` says nothing about how many green keys are filed;
    /// fixtures that assert "one bucket" mean one OCCUPIED slot.
    #[cfg(test)]
    pub(crate) fn occupied_buckets(&self) -> usize {
        self.inner
            .borrow()
            .celltable
            .iter()
            .filter(|slot| slot.is_some())
            .count()
    }
}

/// counter.py DeterministicJitCounter — test-only, NOT_RPYTHON.
///
/// RPython: subclasses JitCounter, overrides _get_index to return the
/// raw hash (identity — no collision), uses a defaultdict timetable.
/// Rust: uses a IndexMap<u64, Entry> to mirror the defaultdict approach.
pub struct DeterministicJitCounter {
    entries: indexmap::IndexMap<u64, Entry>,
}

impl Default for DeterministicJitCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl DeterministicJitCounter {
    /// counter.py DeterministicJitCounter.__init__
    pub fn new() -> Self {
        DeterministicJitCounter {
            entries: indexmap::IndexMap::new(),
        }
    }

    /// counter.py _get_index — identity (no hash collision).
    #[inline(always)]
    fn _get_index(hash: u64) -> u64 {
        hash
    }

    /// counter.py _get_subhash
    #[inline(always)]
    fn _get_subhash(hash: u64) -> u16 {
        (hash & 0xFFFF) as u16
    }

    /// counter.py compute_threshold
    pub fn compute_threshold(&self, threshold: u32) -> f64 {
        if threshold == 0 {
            return 0.0;
        }
        1.0_f64 / (threshold as f64 - 0.001)
    }

    /// counter.py tick — same logic but using identity _get_index.
    pub fn tick(&mut self, hash: u64, increment: f64) -> bool {
        let key = Self::_get_index(hash);
        let subhash = Self::_get_subhash(hash);
        let entry = self.entries.entry_or_insert_with(key, Entry::default);

        let n = if entry.subhashes[0] == subhash {
            0
        } else if entry.subhashes[1] == subhash {
            JitCounter::_swap(entry, 0)
        } else if entry.subhashes[2] == subhash {
            JitCounter::_swap(entry, 1)
        } else if entry.subhashes[3] == subhash {
            JitCounter::_swap(entry, 2)
        } else if entry.subhashes[4] == subhash {
            JitCounter::_swap(entry, 3)
        } else {
            let mut n = 4;
            while n > 0 && entry.times[n - 1] == 0.0 {
                n -= 1;
            }
            entry.subhashes[n] = subhash;
            entry.times[n] = 0.0;
            n
        };

        let counter: f64 = entry.times[n] as f64 + increment;
        if counter < 1.0 {
            entry.times[n] = counter as f32;
            false
        } else {
            self.reset(hash);
            true
        }
    }

    /// counter.py reset
    pub fn reset(&mut self, hash: u64) {
        let key = Self::_get_index(hash);
        let subhash = Self::_get_subhash(hash);
        if let Some(entry) = self.entries.get_mut(&key) {
            for i in 0..ASSOCIATIVITY {
                if entry.subhashes[i] == subhash {
                    entry.times[i] = 0.0;
                }
            }
        }
    }

    /// counter.py decay_all_counters — no-op for deterministic counter.
    pub fn decay_all_counters(&mut self) {}

    /// counter.py _clear_all
    pub fn _clear_all(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    static DECAY_GENERATION_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn advance_decay_generation(intervals: usize) {
        DECAY_GENERATION.fetch_add(intervals, Ordering::Relaxed);
    }

    fn counter_time(counter: &JitCounter, hash: u64) -> f32 {
        let inner = counter.inner.borrow();
        let index = inner._get_index(hash);
        let subhash = JitCounterInner::_get_subhash(hash);
        let entry = &inner.timetable[index];
        for i in 0..ASSOCIATIVITY {
            if entry.subhashes[i] == subhash {
                return entry.times[i];
            }
        }
        0.0
    }

    #[test]
    fn test_basic_counting() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(3);
        assert!(!counter.tick(42, increment));
        assert!(!counter.tick(42, increment));
        assert!(counter.tick(42, increment));
    }

    #[test]
    fn test_different_hashes() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(3);
        let shift = counter.inner.borrow().shift;
        let h1 = 1u64 << shift;
        let h2 = 2u64 << shift;
        assert!(!counter.tick(h1, increment));
        assert!(!counter.tick(h2, increment));
        assert!(!counter.tick(h1, increment));
        assert!(counter.tick(h1, increment));
        assert!(!counter.tick(h2, increment));
    }

    #[test]
    fn test_reset() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(3);
        let h = 1u64 << counter.inner.borrow().shift;
        counter.tick(h, increment);
        counter.tick(h, increment);
        counter.reset(h);
        assert!(!counter.tick(h, increment));
        assert!(!counter.tick(h, increment));
        assert!(counter.tick(h, increment));
    }

    #[test]
    fn test_decay() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(10);
        let h = 1u64 << counter.inner.borrow().shift;
        for _ in 0..8 {
            counter.tick(h, increment);
        }
        // default decay_by_mult = 1.0 (no decay). Set decay first.
        counter.set_decay(40); // decay_by_mult = 0.96
        // time ≈ 8 * (1/10) = 0.8, decay by 0.96 → 0.768
        counter.decay_all_counters();
        // Verify via a tick that doesn't fire (need ~0.232 more to reach 1.0)
        let inner = counter.inner.borrow();
        let index = inner._get_index(h);
        let subhash = JitCounterInner::_get_subhash(h);
        let entry = &inner.timetable[index];
        let mut time = 0.0f32;
        for i in 0..ASSOCIATIVITY {
            if entry.subhashes[i] == subhash {
                time = entry.times[i];
                break;
            }
        }
        assert!(time > 0.7 && time < 0.8, "time={}", time);
    }

    #[test]
    fn test_tick_applies_every_pending_decay_generation() {
        let _generation_guard = DECAY_GENERATION_TEST_LOCK.lock();
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        counter.set_decay(40);
        let h = 3u64 << counter.inner.borrow().shift;
        counter.change_current_fraction(h, 0.5);

        advance_decay_generation(2);
        let increment = 0.001;
        assert!(!counter.tick(h, increment));

        // pypy__decay_jit_counters narrows the multiplier once and multiplies
        // in single precision; two elapsed intervals are two such multiplies.
        let mult = 0.96f64 as f32;
        let expected = ((0.5f32 * mult * mult) as f64 + increment) as f32;
        let actual = counter_time(&counter, h);
        assert!((actual - expected).abs() < 1.0e-6, "actual={actual}");
    }

    #[test]
    fn test_change_current_fraction_is_not_retro_decayed() {
        let _generation_guard = DECAY_GENERATION_TEST_LOCK.lock();
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        counter.set_decay(40);
        let h = 4u64 << counter.inner.borrow().shift;
        counter.change_current_fraction(h, 0.5);

        advance_decay_generation(1);
        counter.change_current_fraction(h, 0.98);
        let increment = 0.001;
        assert!(!counter.tick(h, increment));

        let expected = (0.98f32 as f64 + increment) as f32;
        let actual = counter_time(&counter, h);
        assert!((actual - expected).abs() < 1.0e-6, "actual={actual}");
    }

    #[test]
    fn test_auto_reset_on_fire() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(3);
        let h = 1u64 << counter.inner.borrow().shift;
        assert!(!counter.tick(h, increment));
        assert!(!counter.tick(h, increment));
        assert!(counter.tick(h, increment));
        assert!(!counter.tick(h, increment));
        assert!(!counter.tick(h, increment));
        assert!(counter.tick(h, increment));
    }

    #[test]
    fn test_fetch_next_hash() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let h1 = counter.fetch_next_hash();
        let h2 = counter.fetch_next_hash();
        assert_ne!(h1, h2);
        assert_ne!(counter._get_index(h1), counter._get_index(h2));
    }

    #[test]
    fn test_change_current_fraction() {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        let increment = counter.compute_threshold(100);
        let h = 1u64 << counter.inner.borrow().shift;
        counter.change_current_fraction(h, 0.98);
        // 0.98 + ~0.01 = ~0.99, not enough; two more ticks → ~1.0
        assert!(!counter.tick(h, increment));
        assert!(counter.tick(h, increment));
    }

    #[test]
    fn test_size_parameter() {
        let counter = JitCounter::new(1024);
        assert_eq!(counter.size(), 1024);
        // 0xFFFFFFFF >> shift = 1023 → shift = 22
        assert_eq!(counter.inner.borrow().shift, 22);
    }

    #[test]
    fn clone_shares_the_timetable() {
        // warmspot.py WarmRunnerDesc.jitcounter is one object; Clone is
        // another handle, not a second table.
        let _generation_guard = DECAY_GENERATION_TEST_LOCK.lock();
        let mut a = JitCounter::new(DEFAULT_SIZE);
        let mut b = a.clone();
        assert!(a.ptr_eq(&b));
        a.set_decay(77);
        assert_eq!(b.decay(), 77);
        assert!(!a.tick(42, 0.5));
        assert!(b.would_tick_fire(42, 0.5));
        b.set_decay(40);
        assert_eq!(a.decay(), 40);
    }
}
