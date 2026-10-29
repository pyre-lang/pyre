//! `rpython/jit/metainterp/memmgr.py` parity.
//!
//! `MemoryManager` is the **sole long-living strong reference** to
//! compiled `JitCellToken` objects (`memmgr.py`):
//!
//! > All the long-lived references to LoopToken are weakrefs (see
//! > JitCell in warmstate.py), apart from the 'alive_loops' set in
//! > MemoryManager, which is the only (long-living) place that keeps
//! > them alive. If a loop was not called for long enough, then it is
//! > removed from 'alive_loops'. It will soon be freed by the GC.
//! > `LoopToken.__del__` calls the method `cpu.free_loop_and_bridges()`.
//!
//! In pyre this is achieved by `alive_loops` storing
//! `Arc<JitCellToken>`. When an entry is removed, the Arc drops; when
//! the last strong reference goes, `JitCellToken::drop` calls
//! `CompiledLoopToken::free_loop_and_bridges` (`llmodel.py`) and the
//! CLT `Drop` bumps `total_freed_*` (`model.py` `CompiledLoopToken.__del__`).
//!
//! Both long-lived handles on a compiled token are now weak —
//! `BaseJitCell::loop_token` (`warmstate.py`) and
//! `CompiledEntry::token` — so `alive_loops` is the sole long-lived strong
//! owner and removing an entry here is what actually frees the token.
//!
//! The corollary every producer owes: a token must be registered HERE before
//! any cell or table is given a handle to it, or it dies with the producer's
//! own temporary and the cell it was installed on reads back as "never
//! compiled" rather than failing loudly. `compile.py:566-567`
//! (`send_loop_to_backend`) and `compile.py`
//! (`compile_tmp_callback`) are the two places upstream discharges it.

use indexmap::IndexMap;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use majit_backend::JitCellToken;

/// Body of `memmgr.py` `class MemoryManager`. Pyre also pins the
/// retrace/unroll parameters here, mirroring RPython's lazy attribute
/// writes via `warmstate.py set_param_retrace_limit set_param_*`. RPython treats them
/// as Python `int` attributes; pyre declares them as typed fields and
/// initializes them to the `rlib/jit.py PARAMETERS` defaults.
struct MemoryManagerInner {
    /// `memmgr.py` `self.current_generation = r_int64(1)`.
    current_generation: i64,
    /// `memmgr.py` `self.max_age = max_age` — set by
    /// `set_max_age` (`memmgr.py`).  `<= 0` disables eviction.
    max_age: i64,
    /// `memmgr.py:39` `self.next_check = r_int64(-1)`.  Generation
    /// at which `_kill_old_loops_now` next fires; `-1` means
    /// "eviction disabled" (`memmgr.py`).
    next_check: i64,
    /// `memmgr.py` `self.check_frequency = -1`.  Number of
    /// generations between successive `_kill_old_loops_now` sweeps.
    /// `-1` is "uninitialized"; `set_max_age` derives a real value
    /// (`int(sqrt(max_age))` by default per `memmgr.py`).
    check_frequency: i64,
    /// How many times this manager has let go of loops: each
    /// `_kill_old_loops_now` that evicted something, and each
    /// `release_all_loops`. A strong handle dropped here can be a cell's last,
    /// after which its weak reference no longer upgrades and the cell stops
    /// answering runnable — so this is one input of
    /// `MetaInterp::runnable_generation`.
    evictions: u64,

    /// `memmgr.py` `self.alive_loops = {}` — a dict keyed on the
    /// looptoken object itself.  In Rust the dict key uses the Arc's
    /// pointer-id (stable while the Arc lives) and the value IS the
    /// strong Arc reference: removing the entry drops the Arc, and
    /// dropping the last strong reference fires `JitCellToken::drop`
    /// (mirrors `LoopToken.__del__` calling `cpu.free_loop_and_bridges`
    /// at `memmgr.py:13-14`).
    ///
    /// **Pointer-key soundness:** `*const JitCellToken` is used **only
    /// as an associative-container key**. The Arc value held alongside guarantees the
    /// pointee is alive for the lifetime of the entry, so pointer
    /// identity is stable until removal.
    alive_loops: indexmap::IndexMap<*const JitCellToken, Arc<JitCellToken>>,

    /// `warmstate.py` `set_param_retrace_limit` writes here.
    /// `unroll.py:215` reader.
    retrace_limit: u32,
    /// `warmstate.py` `set_param_max_retrace_guards`.
    /// `unroll.py:265` reader.
    max_retrace_guards: u32,
    /// `warmstate.py` `set_param_max_unroll_loops`.
    /// `pyjitpl.py:2946` reader.
    max_unroll_loops: u32,
    /// `warmstate.py` `set_param_max_unroll_recursion`.
    /// `pyjitpl.py:1404` reader.
    max_unroll_recursion: u32,
}

/// `memmgr.py` `class MemoryManager`.
///
/// `Clone` is another handle to the same manager: `warmspot.py`
/// `WarmRunnerDesc.__init__` creates one `memmgr.MemoryManager()` on the
/// runner, and every `WarmEnterState` reaches it as
/// `self.warmrunnerdesc.memory_manager` (`set_param_loop_longevity`,
/// `get_assembler_token`, `execute_assembler`).
#[derive(Clone)]
pub struct MemoryManager {
    inner: Rc<RefCell<MemoryManagerInner>>,
}

impl MemoryManagerInner {
    /// `memmgr.py` `MemoryManager.__init__`. Note RPython splits
    /// init from `set_max_age`; pyre takes `max_age` upfront for
    /// ergonomics — `set_max_age` later overwrites it just like the
    /// upstream call sequence at `warmspot.py` /
    /// `set_user_param('loop_longevity=...')`.
    fn new(max_age: i64) -> Self {
        let mut mgr = MemoryManagerInner {
            // memmgr.py:38 current_generation = r_int64(1)
            current_generation: 1,
            // memmgr.py:39 next_check = r_int64(-1)
            next_check: -1,
            // memmgr.py:26 check_frequency = -1
            check_frequency: -1,
            max_age: 0,
            alive_loops: indexmap::IndexMap::new(),
            evictions: 0,
            // rlib/jit.py PARAMETERS defaults.
            retrace_limit: 0,
            max_retrace_guards: 15,
            max_unroll_loops: 0,
            max_unroll_recursion: 7,
        };
        // memmgr.py set_max_age — derives next_check / check_frequency.
        mgr.set_max_age(max_age, 0);
        mgr
    }

    /// `memmgr.py` `set_max_age(max_age, check_frequency=0)`.
    ///
    /// ```python
    /// def set_max_age(self, max_age, check_frequency=0):
    ///     if max_age <= 0:
    ///         self.next_check = r_int64(-1)
    ///     else:
    ///         self.max_age = max_age
    ///         if check_frequency <= 0:
    ///             check_frequency = int(math.sqrt(max_age))
    ///         self.check_frequency = check_frequency
    ///         self.next_check = self.current_generation + 1
    /// ```
    pub fn set_max_age(&mut self, max_age: i64, check_frequency: i64) {
        if max_age <= 0 {
            self.next_check = -1;
        } else {
            self.max_age = max_age;
            let cf = if check_frequency <= 0 {
                (max_age as f64).sqrt() as i64
            } else {
                check_frequency
            };
            self.check_frequency = cf;
            self.next_check = self.current_generation + 1;
        }
    }

    /// Read accessor.  Used by `set_user_param('loop_longevity', ...)`
    /// readback in `warmstate.rs`.
    pub fn max_age(&self) -> i64 {
        self.max_age
    }

    /// Pyre-only readback for `get_param("loop_longevity")`.
    ///
    /// `set_max_age(<= 0)` matches `memmgr.py self.next_check = -1`
    /// without touching `max_age`, so the raw `max_age` field stays at
    /// the previous positive value. RPython has no `get_param` for
    /// `loop_longevity`, so the field staleness is invisible upstream.
    /// Pyre adds the readback, so report 0 when eviction is disabled
    /// (`next_check == -1`) — matching the user-facing semantics of
    /// `set_user_param('loop_longevity=0')`.
    pub fn loop_longevity_param(&self) -> i64 {
        if self.next_check == -1 {
            0
        } else {
            self.max_age
        }
    }

    /// `memmgr.py` `keep_loop_alive(looptoken)`.
    ///
    /// ```python
    /// def keep_loop_alive(self, looptoken):
    ///     if looptoken.generation != self.current_generation:
    ///         looptoken.generation = self.current_generation
    ///         self.alive_loops[looptoken] = None
    /// ```
    pub fn keep_loop_alive(&mut self, looptoken: &Arc<JitCellToken>) {
        if looptoken.generation.get() != self.current_generation {
            looptoken.generation.set(self.current_generation);
            let key: *const JitCellToken = Arc::as_ptr(looptoken);
            self.alive_loops
                .entry(key)
                .or_insert_with(|| Arc::clone(looptoken));
        }
    }

    /// `memmgr.py` `next_generation`.
    ///
    /// ```python
    /// def next_generation(self):
    ///     self.current_generation += 1
    ///     if self.current_generation == self.next_check:
    ///         self._kill_old_loops_now()
    ///         self.next_check = self.current_generation + self.check_frequency
    /// ```
    ///
    /// Returns the evicted token objects, where upstream returns `None`:
    /// `LoopToken.__del__` (`memmgr.py`) lets the CLT run
    /// `cpu.free_loop_and_bridges` once `alive_loops` drops the token.
    /// Pyre does the same in `JitCellToken::drop` /
    /// `CompiledLoopToken::drop`. The list lets
    /// `pyjitpl::try_to_free_some_loops` retire the matching
    /// `compiled_loops` entries, whose metadata cannot live on the
    /// token (crate split).
    ///
    /// The return value is a `Vec<Arc<JitCellToken>>` rather than
    /// `Vec<u64>` (green_keys) so the caller can match by **token-object
    /// identity** (`Arc::ptr_eq`) — mirroring `memmgr.py`'s
    /// `del self.alive_loops[looptoken]`, which keys on the looptoken
    /// itself.  Returning green_keys would let an evicted stale
    /// looptoken kick out the *current* compiled token at the same
    /// green_key (the recompile case where `compiled_loops[gk].token`
    /// has already been replaced).
    pub fn next_generation(&mut self) -> Vec<Arc<JitCellToken>> {
        self.current_generation += 1;
        if self.current_generation == self.next_check {
            let evicted = self._kill_old_loops_now();
            self.next_check = self.current_generation + self.check_frequency;
            evicted
        } else {
            Vec::new()
        }
    }

    /// `memmgr.py` `_kill_old_loops_now`.  RPython:
    /// ```python
    /// debug_start("jit-mem-collect")
    /// oldtotal = len(self.alive_loops)
    /// debug_print("Current generation:", self.current_generation)
    /// debug_print("Loop tokens before:", oldtotal)
    /// max_generation = self.current_generation - (self.max_age - 1)
    /// for looptoken in self.alive_loops.keys():
    ///     if (0 <= looptoken.generation < max_generation
    ///         or looptoken.invalidated):
    ///         del self.alive_loops[looptoken]
    /// newtotal = len(self.alive_loops)
    /// debug_print("Loop tokens freed: ", oldtotal - newtotal)
    /// debug_print("Loop tokens left:  ", newtotal)
    /// debug_stop("jit-mem-collect")
    /// ```
    /// Pyre uses `IndexMap::retain` to fuse the iterate + delete steps.
    /// Output is routed through [`crate::debug`] (`debug_start /
    /// debug_print / debug_stop`) so the `jit-mem-collect` section
    /// brackets match PyPy's `rlib/debug.py` wire format and can be
    /// consumed by `rpython/tool/logparser.py` without prefix munging.
    fn _kill_old_loops_now(&mut self) -> Vec<Arc<JitCellToken>> {
        let _scope = crate::debug::scope("jit-mem-collect");
        let log = crate::debug::have_debug_prints();
        let oldtotal = if log { self.alive_loops.len() } else { 0 };
        if log {
            crate::debug::debug_print(&format!("Current generation: {}", self.current_generation));
            crate::debug::debug_print(&format!("Loop tokens before: {oldtotal}"));
        }
        let max_generation = self.current_generation - (self.max_age - 1);
        // memmgr.py:70-73 `for looptoken in self.alive_loops.keys(): if
        // (0 <= looptoken.generation < max_generation or
        // looptoken.invalidated): del self.alive_loops[looptoken]`.
        // Pyre returns the looptoken Arcs themselves so the caller can
        // match by Arc identity (cf. RPython's `del` keying on the
        // looptoken object) when dropping `compiled_loops` entries.
        let mut evicted_tokens = Vec::new();
        self.alive_loops.retain(|_key, token| {
            let token_gen = token.generation.get();
            let invalidated = token.invalidated.load(Ordering::Relaxed);
            let evict = (0 <= token_gen && token_gen < max_generation) || invalidated;
            if evict {
                evicted_tokens.push(Arc::clone(token));
            }
            !evict
        });
        // memmgr.py `_kill_old_loops_now` only deletes the `alive_loops`
        // entry; a token still reachable from a live jumper through
        // `_keepalive_jitcell_tokens` stays alive. Break only garbage
        // `record_jump_to` cycles so `JitCellToken::drop` can run.
        self.trial_delete_keepalive_tokens(&evicted_tokens);
        if !evicted_tokens.is_empty() {
            self.evictions = self.evictions.wrapping_add(1);
        }
        if log {
            let newtotal = self.alive_loops.len();
            crate::debug::debug_print(&format!("Loop tokens freed: {}", oldtotal - newtotal));
            crate::debug::debug_print(&format!("Loop tokens left: {newtotal}"));
        }
        evicted_tokens
    }

    /// `memmgr.py` `release_all_loops`.
    ///
    /// ```python
    /// debug_start("jit-mem-releaseall")
    /// debug_print("Loop tokens cleared:", len(self.alive_loops))
    /// self.alive_loops.clear()
    /// debug_stop("jit-mem-releaseall")
    /// ```
    pub fn release_all_loops(&mut self) -> Vec<Arc<JitCellToken>> {
        let _scope = crate::debug::scope("jit-mem-releaseall");
        crate::debug::debug_print(&format!("Loop tokens cleared: {}", self.alive_loops.len()));
        // memmgr.py `release_all_loops` drops every `alive_loops` entry
        // and lets the GC free unreachable tokens. Drain first so the
        // trial deletion below can tell a still-held token from garbage.
        // The returned Arcs let `pyjitpl::try_to_free_some_loops` retire
        // matching `compiled_loops` entries the same way
        // `_kill_old_loops_now` does.
        let evicted_tokens: Vec<Arc<JitCellToken>> = std::mem::take(&mut self.alive_loops)
            .into_iter()
            .map(|(_, token)| token)
            .collect();
        self.trial_delete_keepalive_tokens(&evicted_tokens);
        self.evictions = self.evictions.wrapping_add(1);
        evicted_tokens
    }

    /// Clear `keepalive_tokens` only on tokens that are garbage after
    /// leaving `alive_loops`.
    ///
    /// `record_jump_to` stores a strong `Arc` in `keepalive_tokens`, so
    /// two loops that jump to each other form a cycle that removing the
    /// `alive_loops` entry will not drop. memmgr.py `_kill_old_loops_now`
    /// / `release_all_loops` only delete that entry ("It will soon be
    /// freed by the GC"): a token still reachable from a live token
    /// through `_keepalive_jitcell_tokens` stays alive together with
    /// everything it reaches. Trial-delete the evicted set so a chain
    /// `C -> A -> B` with `C` still in `alive_loops` keeps A's hold on B.
    fn trial_delete_keepalive_tokens(&self, evicted_tokens: &[Arc<JitCellToken>]) {
        let mut candidates: Vec<Arc<JitCellToken>> = Vec::new();
        for token in evicted_tokens {
            let ptr = Arc::as_ptr(token);
            if self.alive_loops.contains_key(&ptr) {
                continue;
            }
            if candidates.iter().any(|t| Arc::as_ptr(t) == ptr) {
                continue;
            }
            candidates.push(Arc::clone(token));
        }
        let mut i = 0;
        while i < candidates.len() {
            let target_ptrs: Vec<*const JitCellToken> = candidates[i]
                .keepalive_tokens
                .lock()
                .iter()
                .map(Arc::as_ptr)
                .collect();
            let mut new_ptrs = Vec::new();
            for ptr in target_ptrs {
                if self.alive_loops.contains_key(&ptr) {
                    continue;
                }
                if candidates.iter().any(|t| Arc::as_ptr(t) == ptr) {
                    continue;
                }
                if new_ptrs.contains(&ptr) {
                    continue;
                }
                new_ptrs.push(ptr);
            }
            if !new_ptrs.is_empty() {
                let clones: Vec<Arc<JitCellToken>> = candidates[i]
                    .keepalive_tokens
                    .lock()
                    .iter()
                    .filter(|t| new_ptrs.contains(&Arc::as_ptr(t)))
                    .cloned()
                    .collect();
                candidates.extend(clones);
            }
            i += 1;
        }

        let mut live = vec![false; candidates.len()];
        for (idx, token) in candidates.iter().enumerate() {
            let ptr = Arc::as_ptr(token);
            let mut internal = 0usize;
            for held in evicted_tokens {
                if Arc::as_ptr(held) == ptr {
                    internal += 1;
                }
            }
            for held in &candidates {
                if Arc::as_ptr(held) == ptr {
                    internal += 1;
                }
            }
            for other in &candidates {
                if Arc::as_ptr(other) == ptr {
                    continue;
                }
                for kept in other.keepalive_tokens.lock().iter() {
                    if Arc::as_ptr(kept) == ptr {
                        internal += 1;
                    }
                }
            }
            if Arc::strong_count(token) > internal && !live[idx] {
                live[idx] = true;
                mark_keepalive_reachable_live(token, &candidates, &mut live);
            }
        }
        for token in self.alive_loops.values() {
            mark_keepalive_reachable_live(token, &candidates, &mut live);
        }
        for (idx, token) in candidates.iter().enumerate() {
            if !live[idx] {
                token.keepalive_tokens.lock().clear();
            }
        }
    }

    /// The counter [`Self::evictions`] documents.
    pub fn eviction_generation(&self) -> u64 {
        self.evictions
    }

    /// Number of loops currently tracked.  Test/debug accessor; no
    /// upstream counterpart since RPython's `alive_loops` is a Python
    /// dict whose `len()` is read directly.
    pub fn alive_count(&self) -> usize {
        self.alive_loops.len()
    }

    /// `memmgr.py` `MemoryManager.current_generation` read accessor.  Test/debug
    /// only; production code reads the field directly.
    pub fn current_generation(&self) -> i64 {
        self.current_generation
    }

    /// Test/debug accessor — `looptoken in self.alive_loops` upstream.
    pub fn contains(&self, looptoken: &Arc<JitCellToken>) -> bool {
        self.alive_loops.contains_key(&Arc::as_ptr(looptoken))
    }
}

impl MemoryManager {
    /// `memmgr.py` `MemoryManager.__init__` + `set_max_age(max_age)`.
    pub fn new(max_age: i64) -> Self {
        MemoryManager {
            inner: Rc::new(RefCell::new(MemoryManagerInner::new(max_age))),
        }
    }

    /// True when `other` is the same manager object
    /// (`warmrunnerdesc.memory_manager` is one object on the runner).
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }

    /// `memmgr.py` `set_max_age(max_age, check_frequency=0)`.
    pub fn set_max_age(&mut self, max_age: i64, check_frequency: i64) {
        self.inner
            .borrow_mut()
            .set_max_age(max_age, check_frequency)
    }

    /// `memmgr.py` `self.max_age`.
    pub fn max_age(&self) -> i64 {
        self.inner.borrow().max_age()
    }

    /// Pyre-only readback for `get_param("loop_longevity")`.
    pub fn loop_longevity_param(&self) -> i64 {
        self.inner.borrow().loop_longevity_param()
    }

    /// `memmgr.py` `keep_loop_alive(looptoken)`.
    pub fn keep_loop_alive(&mut self, looptoken: &Arc<JitCellToken>) {
        self.inner.borrow_mut().keep_loop_alive(looptoken)
    }

    /// `memmgr.py` `next_generation`; returns the evicted tokens (see
    /// [`MemoryManagerInner::next_generation`]).
    pub fn next_generation(&mut self) -> Vec<Arc<JitCellToken>> {
        self.inner.borrow_mut().next_generation()
    }

    /// `memmgr.py` `release_all_loops`.
    pub fn release_all_loops(&mut self) -> Vec<Arc<JitCellToken>> {
        self.inner.borrow_mut().release_all_loops()
    }

    /// The counter `MemoryManagerInner::evictions` documents.
    pub fn eviction_generation(&self) -> u64 {
        self.inner.borrow().eviction_generation()
    }

    /// `len(self.alive_loops)`.
    pub fn alive_count(&self) -> usize {
        self.inner.borrow().alive_count()
    }

    /// `memmgr.py` `MemoryManager.current_generation`.
    pub fn current_generation(&self) -> i64 {
        self.inner.borrow().current_generation()
    }

    /// `memmgr.py` `MemoryManager.next_check`.
    pub fn next_check(&self) -> i64 {
        self.inner.borrow().next_check
    }

    /// `looptoken in self.alive_loops`.
    pub fn contains(&self, looptoken: &Arc<JitCellToken>) -> bool {
        self.inner.borrow().contains(looptoken)
    }

    /// `warmstate.py` `set_param_retrace_limit` target.
    pub fn retrace_limit(&self) -> u32 {
        self.inner.borrow().retrace_limit
    }

    pub fn set_retrace_limit(&mut self, value: u32) {
        self.inner.borrow_mut().retrace_limit = value;
    }

    /// `warmstate.py` `set_param_max_retrace_guards` target.
    pub fn max_retrace_guards(&self) -> u32 {
        self.inner.borrow().max_retrace_guards
    }

    pub fn set_max_retrace_guards(&mut self, value: u32) {
        self.inner.borrow_mut().max_retrace_guards = value;
    }

    /// `warmstate.py` `set_param_max_unroll_loops` target.
    pub fn max_unroll_loops(&self) -> u32 {
        self.inner.borrow().max_unroll_loops
    }

    pub fn set_max_unroll_loops(&mut self, value: u32) {
        self.inner.borrow_mut().max_unroll_loops = value;
    }

    /// `warmstate.py` `set_param_max_unroll_recursion` target.
    pub fn max_unroll_recursion(&self) -> u32 {
        self.inner.borrow().max_unroll_recursion
    }

    pub fn set_max_unroll_recursion(&mut self, value: u32) {
        self.inner.borrow_mut().max_unroll_recursion = value;
    }
}

fn mark_keepalive_reachable_live(
    start: &JitCellToken,
    candidates: &[Arc<JitCellToken>],
    live: &mut [bool],
) {
    let mut work: Vec<*const JitCellToken> = start
        .keepalive_tokens
        .lock()
        .iter()
        .map(Arc::as_ptr)
        .collect();
    while let Some(ptr) = work.pop() {
        let Some(idx) = candidates.iter().position(|t| Arc::as_ptr(t) == ptr) else {
            continue;
        };
        if live[idx] {
            continue;
        }
        live[idx] = true;
        work.extend(
            candidates[idx]
                .keepalive_tokens
                .lock()
                .iter()
                .map(Arc::as_ptr),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keepalive_holds(jumper: &Arc<JitCellToken>, target: &Arc<JitCellToken>) -> bool {
        jumper
            .keepalive_tokens
            .lock()
            .iter()
            .any(|t| Arc::ptr_eq(t, target))
    }

    #[test]
    fn evicted_tokens_release_keepalive_cycle() {
        let a = Arc::new(JitCellToken::new(1));
        let b = Arc::new(JitCellToken::new(2));
        a.record_jump_to(Arc::clone(&b));
        b.record_jump_to(Arc::clone(&a));

        let mut mgr = MemoryManager::new(1);
        mgr.keep_loop_alive(&a);
        mgr.keep_loop_alive(&b);

        // Observer Arcs would look like an executing frame / JitCell hold
        // and keep the cycle. Drop them so this is a pure garbage cycle.
        let weak_a = Arc::downgrade(&a);
        let weak_b = Arc::downgrade(&b);
        drop(a);
        drop(b);

        let mut evicted = Vec::new();
        while mgr.alive_count() > 0 {
            evicted.extend(mgr.next_generation());
        }
        drop(evicted);

        assert!(weak_a.upgrade().is_none());
        assert!(weak_b.upgrade().is_none());
    }

    #[test]
    fn evicted_chain_kept_by_alive_jumper() {
        let a = Arc::new(JitCellToken::new(1));
        let b = Arc::new(JitCellToken::new(2));
        let c = Arc::new(JitCellToken::new(3));
        c.record_jump_to(Arc::clone(&a));
        a.record_jump_to(Arc::clone(&b));

        // max_age=2, check_frequency=1: first sweep keeps gen=1 tokens,
        // second sweep evicts them while a gen=2 token stays in alive_loops.
        let mut mgr = MemoryManager::new(2);
        mgr.keep_loop_alive(&a);
        mgr.keep_loop_alive(&b);
        assert!(mgr.next_generation().is_empty());
        mgr.keep_loop_alive(&c);

        let weak_a = Arc::downgrade(&a);
        let weak_b = Arc::downgrade(&b);
        drop(a);
        drop(b);

        let evicted = mgr.next_generation();
        assert!(evicted.iter().any(|t| Arc::as_ptr(t) == weak_a.as_ptr()));
        assert!(evicted.iter().any(|t| Arc::as_ptr(t) == weak_b.as_ptr()));
        assert!(mgr.contains(&c));
        drop(evicted);

        let a = weak_a.upgrade().expect("C still jumps to A");
        let b = weak_b.upgrade().expect("A still jumps to B");
        assert!(keepalive_holds(&a, &b));
    }

    #[test]
    fn evicted_token_held_externally_keeps_keepalive() {
        let a = Arc::new(JitCellToken::new(1));
        let b = Arc::new(JitCellToken::new(2));
        a.record_jump_to(Arc::clone(&b));

        let mut mgr = MemoryManager::new(1);
        mgr.keep_loop_alive(&a);
        mgr.keep_loop_alive(&b);

        let executing = Arc::clone(&a);
        drop(a);
        let weak_b = Arc::downgrade(&b);
        drop(b);

        let mut evicted = Vec::new();
        while mgr.alive_count() > 0 {
            evicted.extend(mgr.next_generation());
        }
        drop(evicted);

        let b = weak_b
            .upgrade()
            .expect("executing frame keeps A's jump target");
        assert!(keepalive_holds(&executing, &b));
    }

    #[test]
    fn release_all_loops_garbage_cycle_and_external_hold() {
        let a = Arc::new(JitCellToken::new(1));
        let b = Arc::new(JitCellToken::new(2));
        a.record_jump_to(Arc::clone(&b));
        b.record_jump_to(Arc::clone(&a));

        let mut mgr = MemoryManager::new(1);
        mgr.keep_loop_alive(&a);
        mgr.keep_loop_alive(&b);
        let weak_a = Arc::downgrade(&a);
        let weak_b = Arc::downgrade(&b);
        drop(a);
        drop(b);
        mgr.release_all_loops();
        assert!(weak_a.upgrade().is_none());
        assert!(weak_b.upgrade().is_none());

        let held = Arc::new(JitCellToken::new(3));
        let target = Arc::new(JitCellToken::new(4));
        held.record_jump_to(Arc::clone(&target));
        let mut mgr = MemoryManager::new(1);
        mgr.keep_loop_alive(&held);
        mgr.keep_loop_alive(&target);
        let executing = Arc::clone(&held);
        drop(held);
        let weak_target = Arc::downgrade(&target);
        drop(target);
        mgr.release_all_loops();
        let target = weak_target
            .upgrade()
            .expect("externally held token keeps its targets");
        assert!(keepalive_holds(&executing, &target));
    }

    #[test]
    fn release_all_loops_returns_every_alive_token() {
        let a = Arc::new(JitCellToken::new(1));
        let b = Arc::new(JitCellToken::new(2));
        let mut mgr = MemoryManager::new(1);
        mgr.keep_loop_alive(&a);
        mgr.keep_loop_alive(&b);
        let evicted = mgr.release_all_loops();
        assert_eq!(evicted.len(), 2);
        assert!(evicted.iter().any(|t| Arc::ptr_eq(t, &a)));
        assert!(evicted.iter().any(|t| Arc::ptr_eq(t, &b)));
        assert_eq!(mgr.alive_count(), 0);
    }
}
