//! Interpreter-path GC integration (`PYRE_GC_INTERP`).
//!
//! The `PYRE_GC_INTERP=0` rollback path creates interpreter boxes through
//! [`crate::lltype::malloc_typed`] (`w_int_new` / `w_float_new`), whose
//! `alloc_with_gc_header` is a bare `std::alloc::alloc` that is never tracked by
//! the collector and never freed — a permanent leak. The default path below
//! instead makes those boxes collector-owned. JIT-compiled boxing already uses
//! the managed nursery.
//!
//! The faithful fix is RPython's model: allocate young objects in the moving
//! nursery and let the allocator trigger a minor collection when it fills. pyre
//! cannot do that yet — the interpreter has no shadowstack pass, so a moving
//! collection would relocate any live `PyObjectRef` held only on the Rust stack
//! of a bytecode handler and leave it dangling (documented at
//! `pyre/pyre-jit/src/eval.rs` `pyre_object_gc_collect_trampoline`).
//!
//! This module is the safe stepping stone: route the interpreter's int/float
//! allocations through the *non-moving* old-gen (`try_gc_alloc_stable`, the same
//! path set/type/instance boxes already use), so they become GC-tracked without
//! the move hazard, and trigger a full mark-sweep at a bytecode-dispatch
//! safepoint (loop top, where the only live refs are in the frame and reachable
//! through the registered `pyframe` root walker). When to collect is not
//! decided here: the safepoint asks the collector's own `threshold_reached`
//! (incminimark.py), which weighs everything the collector is
//! responsible for against the threshold `set_major_threshold_from`
//! (incminimark.py) derived from the last major's survivors. Upstream
//! asks that question in the allocator; pyre asks it here because here is where
//! it can act on the answer.
//!
//! Enabled by default on every target. RPython never has a mode in which
//! ordinary interpreter boxes sit outside the collector, and the full native
//! synthetic corpus exercises this born-old stepping stone without a semantic
//! or root-liveness failure. `PYRE_GC_INTERP=0` remains the explicit rollback
//! switch while the translated shadow-stack pass replaces the stepping stone
//! with ordinary moving-nursery allocation.

use std::cell::Cell;
use std::ffi::OsStr;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU8, Ordering};

/// Tri-state: 0 = not yet read from env, 1 = disabled, 2 = enabled.
static STATE: AtomicU8 = AtomicU8::new(0);

/// A semantic old-gen collection requested by a fork child callback.
///
/// This is process-global because the request belongs to the post-fork
/// interpreter, not to the native thread that happened to execute the hook.
/// Keep it outside the eval-breaker word while a re-entrant callback is too
/// deep to expose a complete root set; [`note_eval_activation_exit`] re-arms
/// the breaker once that callback has unwound.
static EXPLICIT_OLDGEN_REQUEST: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// Number of interpreter eval-loop activations currently on this
    /// thread's call stack (both the plain `eval_loop` and the JIT
    /// `eval_loop_jit`). The dispatch-loop safepoint consults it so
    /// a collection only fires at the OUTERMOST activation — see
    /// [`at_outermost_activation`].
    static EVAL_NESTING: Cell<u32> = const { Cell::new(0) };
}

/// RAII guard that counts one interpreter eval-loop activation. Construct it
/// at the top of `eval_loop` / `eval_loop_jit`; the matching `Drop` rewinds
/// the depth on every exit path (normal return, `?`, unwind). `armed` records
/// whether the ordinary interpreter-allocation GC integration is enabled;
/// nesting itself is always tracked because an explicit post-fork request must
/// obey the same root-completeness rule when that feature is off.
pub struct EvalActivationGuard {
    armed: bool,
}

/// Count one eval-loop activation on this thread.
///
/// Writes the runtime-mutable `EVAL_NESTING` thread-local, not a build-time
/// constant, so the JIT residualizes the call instead of tracing into it
/// (`@dont_look_inside`, the [`at_outermost_activation`] sibling). No
/// arguments, `()` result, and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn note_eval_activation_enter() {
    EVAL_NESTING.with(|d| d.set(d.get() + 1));
}

/// Rewind one eval-loop activation. The [`note_eval_activation_enter`] twin;
/// same residual contract.
#[majit_macros::dont_look_inside]
pub fn note_eval_activation_exit() {
    let depth = EVAL_NESTING.with(|d| {
        let depth = d.get().saturating_sub(1);
        d.set(depth);
        depth
    });
    if depth <= 2 && EXPLICIT_OLDGEN_REQUEST.load(Ordering::Acquire) {
        majit_gc::collector::request_deferred_major_collection();
    }
    // A nested Python eval just returned. Its frame is unrooted
    // (`pyframe.py class PyFrame(W_Root)`). The dispatch-loop poll refuses
    // while `EVAL_NESTING` is ≥ 3, so a recursive call never collects on
    // its own bytecodes; arm `EB_GC` so the caller's next opcode — after
    // CALL has finished — runs `do_collect_oldgen_nonmoving`.
    if enabled()
        && collect_enabled()
        && crate::gc_hook::try_gc_isenabled()
        && crate::gc_hook::try_gc_major_threshold_reached()
    {
        majit_gc::collector::request_deferred_major_collection();
    }
}

impl EvalActivationGuard {
    #[inline]
    pub fn enter() -> Self {
        let armed = enabled();
        note_eval_activation_enter();
        Self { armed }
    }

    #[inline]
    pub fn armed(&self) -> bool {
        self.armed
    }
}

impl Drop for EvalActivationGuard {
    #[inline]
    fn drop(&mut self) {
        note_eval_activation_exit();
    }
}

/// Whether the safepoint may fire at this eval-loop nesting depth.
///
/// Depth 1 = module-level loop, depth 2 = one called function's loop.
/// Both are safe: a Python CALL opcode completes before re-entering the
/// eval loop, so no opcode handler holds a Rust-stack `PyObjectRef` across
/// the boundary; the pyframe root walker sees the full `CURRENT_FRAME` /
/// `f_backref` chain.
///
/// Depth ≥ 3 = a callback re-entry from inside an opcode handler (e.g.
/// FOR_ITER → `__getitem__`, exception handler, `_sre.sub` callable,
/// `sorted` key). The outer handler holds live `PyObjectRef`s on the
/// Rust stack that the walker cannot reach — a collection would free
/// still-reachable old-gen objects. Block the safepoint there.
///
/// Reads the runtime-mutable `EVAL_NESTING` thread-local, not a build-time
/// constant, so the JIT residualizes the call instead of tracing into it
/// (`@dont_look_inside`, the [`enabled`] sibling). The `-> bool` return fits
/// a single word and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn at_outermost_activation() -> bool {
    EVAL_NESTING.with(|d| d.get() <= 2)
}

/// Tri-state for the safepoint collection, gated by `PYRE_GC_INTERP_COLLECT`
/// (default on when `PYRE_GC_INTERP` is on). Lets us A/B routing-only vs
/// routing+collection while diagnosing root-completeness.
static COLLECT_STATE: AtomicU8 = AtomicU8::new(0);

/// Dispatches between two `threshold_reached` queries.
///
/// This is a poll rate, not a collection policy: the safepoint holds no model
/// of the heap, and this value cannot make a collection happen that the
/// collector did not ask for. It exists only because reaching the collector
/// costs a thread-local borrow — or, with no backend GC box installed, a
/// `gc_op` compare-and-exchange — which is too much to spend on every bytecode
/// dispatch. At a few tens of bytes allocated per dispatch the interval spans
/// far less than `min_heap_size` (incminimark.py), the smallest threshold
/// the collector will ever set, so no answer is reached late enough to matter.
const POLL_INTERVAL: u32 = 1024;

/// `executioncontext.py ActionFlag._ticker`: dispatches left before the next
/// one asks the collector.
///
/// One process-wide counter, decremented by every dispatch that has the
/// safepoint armed, as `bytecode_trace` decrements the single `actionflag`
/// ticker under the GIL. Relaxed loads and stores: the GIL orders dispatches,
/// and a lost decrement only moves a poll by one dispatch.
static POLL_TICKER: AtomicI32 = AtomicI32::new(POLL_INTERVAL as i32);

/// `executioncontext.py ActionFlag.decrement_ticker(by) < 0` — whether this
/// dispatch is the one that asks the collector.
///
/// Reads the runtime-mutable `POLL_TICKER`, not a build-time constant, so the
/// JIT residualizes the call instead of tracing into it (`@dont_look_inside`,
/// the [`at_outermost_activation`] sibling). The `-> bool` return fits a
/// single word and it cannot raise.
#[majit_macros::dont_look_inside]
#[inline]
pub fn poll_due() -> bool {
    let ticker = POLL_TICKER.load(Ordering::Relaxed) - 1;
    if ticker < 0 {
        POLL_TICKER.store(POLL_INTERVAL as i32, Ordering::Relaxed);
        true
    } else {
        POLL_TICKER.store(ticker, Ordering::Relaxed);
        false
    }
}

/// Whether interpreter allocations are routed through the GC and the
/// dispatch-loop safepoint is armed. This is on by default, matching RPython's
/// single collector-owned object model; `PYRE_GC_INTERP=0` is the temporary
/// rollback switch for the born-old interpreter stepping stone. Reads the env
/// once, then caches.
///
/// Reads (and lazily initialises) the runtime `STATE` atomic. The env
/// read is idempotent and cached, so `@elidable` (`rlib/jit.py`) can
/// fold a traced result; `@dont_look_inside` would leave a residual
/// `CallI` in every compiled loop.
#[majit_macros::elidable_cannot_raise]
pub fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => init_interp_state(),
    }
}

/// First-touch halves of [`enabled`] and [`collect_enabled`], out of line.
///
/// `env::var_os` allocates and unwinds, so leaving it inlined puts its frame
/// on whoever inlines the switch — and that is [`safepoint`], which the
/// dispatch loop calls on every bytecode.  Cold and never inlined, the switch
/// itself is an atomic load and a branch, which is all upstream's equivalent
/// gate is (`bytecode_trace` reads a field of the prebuilt `ActionFlag`).
///
/// Each names its own variable at the `env::var_os` call, which is what marks
/// the gate live for `gate_triage_complete`.
#[cold]
#[inline(never)]
fn init_interp_state() -> bool {
    store_from_env(&STATE, std::env::var_os("PYRE_GC_INTERP"))
}

#[cold]
#[inline(never)]
fn init_collect_state() -> bool {
    store_from_env(&COLLECT_STATE, std::env::var_os("PYRE_GC_INTERP_COLLECT"))
}

fn store_from_env(state: &AtomicU8, value: Option<std::ffi::OsString>) -> bool {
    let on = enabled_from_env(value.as_deref());
    state.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
}

/// Pure half of [`enabled`] and [`collect_enabled`], kept separate so the
/// default-on contract is testable without racing process-global environment
/// mutation, and shared so the two switches cannot drift apart.
fn enabled_from_env(value: Option<&OsStr>) -> bool {
    // Exactly `0`, and nothing else, takes the rollback path. An empty value is
    // what a launcher produces from an unset variable (`PYRE_GC_INTERP=$FOO`),
    // and treating it as the off switch would route those processes onto the
    // born-old stepping stone, which leaks every interpreter allocation.
    value != Some(OsStr::new("0"))
}

/// Whether a collection reaching this safepoint right now would be performed.
///
/// The allocator asks it too, before arming a deferred major request, so the
/// two cannot drift: a request armed past a safepoint that will refuse is
/// re-armed by every following born-old allocation, because the threshold
/// stays reached until a major completes
/// (`majit_gc::collector::set_deferred_major_request_probe`).
///
/// `gc.disable()` is among the questions. This is the automatic path, the one
/// `incminimark.py` returns from while the flag is clear; only an
/// explicit `gc.collect()` carries `force_enabled` past it. The collection this
/// safepoint runs is `do_collect_oldgen_nonmoving`, which drives a full cycle
/// whatever the flag says, so the flag has to be read here or a
/// `gc.disable()` region would still be collected.
///
/// Plain rather than `@dont_look_inside`: both callers are already outside
/// traced code — [`safepoint`] is itself a residual, and the allocator reaches
/// this through an installed `fn` pointer — and each of the four questions it
/// asks carries the attribute in its own right.
pub fn would_collect() -> bool {
    enabled()
        && collect_enabled()
        && crate::gc_hook::try_gc_isenabled()
        && at_outermost_activation()
}

/// Dispatch-loop safepoint: when the collector says it has reached the
/// threshold it set for its next major, run a non-moving old-gen-only major.
/// A no-op when the flag is off or no collection hook is installed.
///
/// The decision is entirely the collector's. `threshold_reached`
/// (incminimark.py) compares `get_total_memory_used()` — every byte
/// the collector is responsible for, headers included, whatever allocated them
/// — against the threshold `set_major_threshold_from` (incminimark.py)
/// derived from the last major's survivors under `major_collection_threshold`,
/// `growth_rate_max`, `max_delta`, `min_heap_size` and `max_heap_size`. Running
/// the collection re-derives it, because `do_collect_oldgen_nonmoving` drives
/// `major_collection_step` to the end of the cycle and `finish_incremental_cycle`
/// sets the next threshold there.
///
/// The safepoint deliberately keeps no counter of its own. Upstream asks this
/// question in the allocator, where every allocation passes; a second tally
/// kept beside the collector could only ever see the subset of allocations
/// that remembered to report, and would answer for a heap that is not the one
/// being collected.
///
/// Born-old allocations ask it in the allocator too, as `external_malloc`
/// (incminimark.py) does, and hand the answer here through the
/// eval-breaker word (`majit_gc::collector::take_deferred_major_request`).
/// That path is what reaches compiled code: a trace runs its loop without
/// returning to this dispatch loop, so a poll placed here alone is never
/// executed while one is hot, and old-gen allocations made from inside it
/// would go unanswered until the loop exited. The armed bit fails the trace's
/// back-edge poll instead, which deopts to this loop and lands on the taker
/// above. The allocator arms it only when [`would_collect`] holds, so the
/// request cannot outlive the conditions that let this safepoint answer it.
///
/// The collection is `try_gc_collect_oldgen` — it seeds roots, marks, and
/// sweeps ONLY the old generation, never touching the nursery (not moved, not
/// freed). So unlike `do_collect_full` (whose leading moving minor would
/// relocate a Rust-stack nursery `PyObjectRef` with no shadowstack root) it is
/// safe with a live nursery, and the `nursery_used == 0` gate is dropped — the
/// reclamation can now fire under an active JIT, exactly when the interp-path
/// old-gen leak accumulates. Reachability stays exact: the marker walks through
/// nursery objects (`old -> nursery -> old` edges are followed) so no live old
/// object is freed. A suspended compiled trace is no obstacle either: a trace
/// suspends by making a residual call into the interpreter, and a call site is
/// where the backend emits a gcmap, so `enumerate_root_walker_values` reaches
/// the suspended frame's live refs by expanding each jitframe on the JF shadow
/// stack through `trace_libc_jitframe`. That is the same basis on which
/// upstream collects with compiled frames on the stack. The safepoint fires
/// at the first two eval activations and no deeper
/// ([`at_outermost_activation`]), so a Python callback re-entered from inside
/// an opcode handler — whose Rust-stack roots the pyframe walker cannot see —
/// never triggers it.
///
/// Dispatches to the installed threshold and collection hooks, neither a
/// build-time constant, so the JIT residualizes the call instead of tracing
/// into it (`@dont_look_inside`, the [`enabled`] sibling). A `()` return has no
/// discriminant to erase and it cannot raise. A dispatch with nothing pending
/// reaches only the two request flags and the two cached feature switches
/// before returning. The flags are read ahead of the switches on purpose — an
/// explicit request has to be serviced even with the routing feature off,
/// which is why [`note_eval_activation_exit`] tracks nesting unconditionally.
#[majit_macros::dont_look_inside]
pub fn safepoint() {
    // Take any request the old-gen allocator armed, and take it before the
    // decision below: the bit is a compiled loop's deopt trigger, so one left
    // armed by a safepoint that declined to collect fires again on the next
    // back edge, and the next.
    let requested = majit_gc::collector::take_deferred_major_request();
    if EXPLICIT_OLDGEN_REQUEST.load(Ordering::Acquire) {
        if !at_outermost_activation() {
            // Leave the semantic request pending outside the breaker word.
            // Re-arming here would deopt every compiled back edge until the
            // native callback unwinds; the activation guard does it once on
            // the eligible transition instead.
            return;
        }
        if EXPLICIT_OLDGEN_REQUEST.swap(false, Ordering::AcqRel) {
            crate::gc_hook::try_gc_collect_oldgen();
            return;
        }
    }
    // The request already carries the collector's threshold answer — it was
    // armed by `external_malloc`'s equivalent in the allocator — so it does
    // not wait for the poll interval. Recheck the semantic gates because
    // `gc.disable()` or a callback activation may have appeared since the
    // allocator armed it.
    if requested {
        if would_collect() {
            crate::gc_hook::try_gc_collect_oldgen();
        }
        return;
    }

    // incminimark asks the comparatively expensive `gcflag_extra` / heap
    // threshold questions from allocation, not from every interpreter
    // dispatch. Pyre's dispatch safepoint is the safe stepping stone until
    // the translated shadow-stack pass lets the allocator collect directly,
    // but its polling interval must retain that shape: check the two cached
    // feature switches, then spend the GC hook and activation queries only on
    // the one dispatch that actually polls.
    if !enabled() || !collect_enabled() || !poll_due() {
        return;
    }
    collect_if_threshold_reached();
}

/// `executioncontext.py bytecode_trace` for the dispatch loops: decrement the
/// action ticker inline and leave the dispatch only for the slow path —
/// `actionflag.decrement_ticker(decr_by) < 0` → `action_dispatcher`.
///
/// `breaker` is the dispatch's own read of the eval-breaker word. An armed
/// request (`EB_GC`) goes to [`safepoint`] at once, as before: it carries the
/// allocator's threshold answer, and an explicit request always travels with
/// it ([`request_oldgen_collection`], [`note_eval_activation_exit`]).
/// Otherwise `EB_GC_INTERP` is the process-stable "poll armed" switch, and
/// only the dispatch that crosses the ticker pays a call.
#[inline]
pub fn dispatch_safepoint(breaker: usize) {
    if breaker & majit_ir::eval_breaker_word::EB_GC != 0 {
        safepoint();
    } else if breaker & majit_ir::eval_breaker_word::EB_GC_INTERP != 0 && poll_due() {
        poll_safepoint();
    }
}

/// Periodic half of `executioncontext.py action_dispatcher`: the action
/// ticker has wrapped, or an urgent `EB_GC` fired it. Runs next to the
/// other `PeriodicAsyncAction`s. `poll_due` is not decremented here; the
/// bytecode countdown is `ActionFlag.decrement_ticker`.
#[cold]
pub fn on_ticker_wrap() {
    let breaker = majit_ir::eval_breaker_word::load();
    if breaker & majit_ir::eval_breaker_word::EB_GC != 0 {
        safepoint();
    } else if breaker & majit_ir::eval_breaker_word::EB_GC_INTERP != 0 {
        poll_safepoint();
    }
}

/// The `action_dispatcher` half of [`dispatch_safepoint`]: the ticker has
/// already crossed zero, so ask the collector without decrementing it again.
#[majit_macros::dont_look_inside]
#[cold]
pub fn poll_safepoint() {
    if !enabled() || !collect_enabled() {
        return;
    }
    collect_if_threshold_reached();
}

fn collect_if_threshold_reached() {
    if crate::gc_hook::try_gc_isenabled()
        && at_outermost_activation()
        && crate::gc_hook::try_gc_major_threshold_reached()
    {
        crate::gc_hook::try_gc_collect_oldgen();
    }
}

/// Ask the next root-complete bytecode dispatch to run a non-moving old-gen
/// collection.  Unlike [`majit_gc::collect_full`], this cannot relocate a
/// nursery reference held by the caller while it unwinds back to the loop.
#[majit_macros::dont_look_inside]
pub fn request_oldgen_collection() {
    EXPLICIT_OLDGEN_REQUEST.store(true, Ordering::Release);
    majit_gc::collector::request_deferred_major_collection();
}

/// Whether the safepoint actually collects. Off via `PYRE_GC_INTERP_COLLECT=0`
/// to isolate the allocation routing from the collection while diagnosing.
///
/// Reads (and lazily initialises) the runtime-mutable `COLLECT_STATE` atomic,
/// not a build-time constant, so the JIT residualizes the call instead of
/// tracing into it (`@dont_look_inside`, the [`enabled`] sibling). The
/// `-> bool` return fits a single word and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn collect_enabled() -> bool {
    match COLLECT_STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => init_collect_state(),
    }
}

#[cfg(test)]
mod tests {
    use super::enabled_from_env;
    use std::ffi::OsStr;

    #[test]
    fn interpreter_gc_is_default_on_with_explicit_off_switch() {
        assert!(enabled_from_env(None));
        assert!(!enabled_from_env(Some(OsStr::new("0"))));
        assert!(enabled_from_env(Some(OsStr::new("1"))));
        // An unset variable expanded by a launcher arrives as an empty value;
        // it must not select the leaking rollback path, nor — through the
        // `collect_enabled` sharing this predicate — silence the safepoint.
        assert!(enabled_from_env(Some(OsStr::new(""))));
    }
}
