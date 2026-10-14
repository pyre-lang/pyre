//! `pypy/module/gc/interp_gc.py`: collection control and the app-level
//! finalizer lock.
//!
//! The `get_count`, `set_threshold`/`get_threshold`, `set_debug`/`get_debug`,
//! `is_tracked`, `is_finalized` and `freeze`/`unfreeze`/`get_freeze_count`
//! bodies at the end answer to 3.14 alone; see the module doc in `mod.rs`.

use majit_gc::GcStepTransition;
use pyre_object::*;

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

use super::hook::{STATE_SCANNING, STATE_USERDEL, new_collect_step_stats};

/// `interp_gc.py` tracks a process-wide `enabled` flag on the GC
/// frontend; pyre has no generational threshold knob, but
/// `gc.isenabled()` should reflect the most recent `enable`/`disable`
/// call so callers that toggle and re-read the state stay consistent.
static GC_ENABLED: AtomicBool = AtomicBool::new(true);

/// The collector debug word.
///
/// `[3.14-spec]` PyPy exposes no such knob, and 3.14 requires the value to be
/// interpreter-owned and shared by all threads, so the word is kept exactly:
/// a caller can bracket a collection and restore the prior flags.  It drives
/// nothing, and `DEBUG_SAVEALL` is the flag that shows why.  3.14 retains what
/// the *cyclic* collector found unreachable, which is a small set precisely
/// because refcounting already reclaimed the acyclic garbage before it ran.
/// Here there is no refcount, so the population reaching the sweep's
/// free-or-keep callback is everything that died since the last major —
/// hundreds of objects across a dozen type ids inside the single collection
/// `test_saveall` brackets, where it expects one — and no filter at that
/// callback can recover the distinction, because a would-this-have-died-by-
/// refcount answer is never computed.  The remaining flags describe a
/// per-object cycle report this collector likewise does not produce.
static GC_DEBUG: AtomicI64 = AtomicI64::new(0);

/// The collection thresholds `gc.get_threshold()` reports.
///
/// `[3.14-spec]` A remembered round trip, where PyPy binds no threshold
/// surface at all.  The values drive nothing, and the reason is a unit
/// mismatch rather than an absence: what schedules a collection here is a byte
/// reading — `get_total_memory_used` against `next_major_collection_threshold`
/// — while `threshold0` is a count of container allocations, and the only
/// knob retunable after construction, `set_max_heap_size`, is a byte ceiling
/// too.  An old-gen live-*object* count does exist (`live_objects`, kept by
/// the arena collection), so the honest statement is that no knob shares
/// `threshold0`'s unit, not that nothing is counted.  Pointing a count at a
/// byte knob would silently mean something neither 3.14 nor PyPy means.  So
/// `set_threshold` stores what it was given and `get_threshold` hands the same
/// tuple back, which is the part of the pair's behaviour a caller can
/// observe.  All three are kept, including the third, whose round
/// trip 3.14 preserves even though its own incremental collector sizes no
/// third generation.  The initial values are the ones a fresh interpreter
/// starts with.
static GC_THRESHOLD: [AtomicI64; 3] =
    [AtomicI64::new(2000), AtomicI64::new(10), AtomicI64::new(10)];

/// How many generations this module reports.  `get_threshold` answers a
/// three-tuple and `set_threshold` writes three slots, so three is the count
/// every generation argument is checked against — the same `NUM_GENERATIONS`
/// `gc_collect_impl` and `gc_get_objects_impl` bound theirs by.  The collector
/// underneath has two physical generations; the middle one is simply always
/// empty, which is what `do_get_objects` already answers for it.
pub(super) const NUM_GENERATIONS: i64 = 3;

/// `rgc.py is_done__states`: a major collection has finished when the
/// step ended in the starting state *and* did not start there. A collector
/// with no work to do reports `(0, 0)`, which is not the end of anything.
pub(super) fn is_done_states(oldstate: u8, newstate: u8) -> bool {
    GcStepTransition {
        old_state: oldstate,
        new_state: newstate,
    }
    .is_done()
}

/// `interp_gc.py StepCollector.finalizing`. `space.fromcache` owns one
/// instance per object space upstream; pyre has one process-wide object space,
/// so the corresponding state is shared rather than thread-local.
///
/// The atomic carries storage and visibility, not the state transition. The
/// rgil is what serializes load -> collector step -> store: a mutator holds it
/// across the whole builtin body (`rgil.rs`'s `acquire_fast_path` takes it with
/// a single-owner CAS, and only an explicit release such as
/// `call_external_function` gives it up), and nothing on this path releases it.
/// So a second mutator cannot also
/// observe `false` and start a major step.
///
/// The USERDEL drain is the deliberate exception, and matches
/// `StepCollector.do`: the flag stays set until `_run_finalizers` returns,
/// so app-level `__del__` code can yield the GIL or re-enter `collect_step`.
/// Both are safe because a queue entry is popped before its callback runs
/// (`_run_finalizers` takes `next_dead()` first, and the collector's
/// `finalizer_next_dead` is a `pop_front`), so an interleaved or nested drain
/// cannot invoke the same entry twice. A lock held across the drain would not
/// help and would invert against the GIL: its holder yields inside `__del__`,
/// a second thread takes the GIL and blocks on the lock, and the holder can
/// never reacquire the GIL.
static STEP_FINALIZING: AtomicBool = AtomicBool::new(false);

fn user_del_action() -> Option<&'static mut pyre_interpreter::executioncontext::UserDelAction> {
    let action = pyre_interpreter::executioncontext::space_user_del_action();
    if action.is_null() {
        None
    } else {
        Some(unsafe { &mut *action })
    }
}

/// Release the mirrors the collection queued, the way `interp_gc.py collect`
/// ends with `_rawrefcount_perform` — "perform dealloc callbacks now, instead
/// of waiting for the next AsyncAction to fire".  A `tp_dealloc` is then part
/// of the `gc.collect()` that freed its object rather than of whatever runs
/// next; `PyObjDeallocAction` stays registered and still drains what an
/// automatic collection queues.
#[cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]
fn run_cpyext_deallocs_now() {
    pyre_interpreter::cpyext::pyobject::drain_dead();
}

/// The builds with no mirrors to release — upstream reaches its
/// `_rawrefcount_perform` only under `usemodules.cpyext`.
#[cfg(not(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
)))]
fn run_cpyext_deallocs_now() {}

/// `interp_gc.py _run_finalizers`: run the queued finalizers now, re-enabling
/// them for the duration when the app level disabled them.
pub(super) fn _run_finalizers() -> Result<(), pyre_interpreter::PyError> {
    let Some(uda) = user_del_action() else {
        return Ok(());
    };
    let temp_reenable = !uda.enabled_at_app_level;
    if temp_reenable {
        __enable_finalizers_impl()?;
    }
    if let Some(uda) = user_del_action() {
        uda._run_finalizers();
    }
    if temp_reenable {
        __disable_finalizers_impl();
    }
    Ok(())
}

/// `_run_finalizers` for interpreter code outside this module.
// Its work is `UserDelAction._run_finalizers`, which carries
// `@jit.dont_look_inside` (executioncontext.py). The bracket around it reads
// the space's `UserDelAction` through a process-global cell, runtime state the
// translated trace cannot read, so the whole drain stays one residual call,
// as for `executioncontext::may_ignore_finalizer`. A residual call has no
// error channel, so when an app-level `enable_finalizers` already released
// the lock `gc.disable` took, the queue drains under the lock depth it finds.
#[majit_macros::dont_look_inside]
pub(crate) fn run_finalizers_now() {
    if _run_finalizers().is_err()
        && let Some(uda) = user_del_action()
    {
        uda._run_finalizers();
    }
}

/// `interp_gc.py collect`.
#[pyre_interpreter::pyre_function]
pub(super) fn collect(
    #[default(w_int_new(NUM_GENERATIONS - 1))] generation: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // `interp_gc.py collect` unwraps the optional generation as an int
    // and then ignores it, because the frontend it belongs to has no
    // generations to select between.  This one does: `NUM_GENERATIONS`
    // publishes the mapping, `get_objects` already selects on it, and
    // `get_count` reports per generation.  So the argument is bounded
    // the way `gc_collect_impl` bounds it and then passed on to
    // `incminimark.py collect(gen)`, whose generations are the same
    // ones -- a minor at 0, a started major at 1, a full major at 2.
    //
    // The default is the oldest generation, so a bare `gc.collect()`
    // is the full collection it has always been.
    let generation = pyre_interpreter::baseobjspace::int_w(
        pyre_interpreter::baseobjspace::space_index(generation)?,
    )?;
    if !(0..NUM_GENERATIONS).contains(&generation) {
        return Err(pyre_interpreter::PyError::value_error("invalid generation"));
    }
    pyre_interpreter::baseobjspace::clear_method_cache();
    pyre_interpreter::objspace::std::mapdict::clear_map_attr_cache();
    pyre_object::gc_hook::try_gc_collect(generation);
    _run_finalizers()?;
    run_cpyext_deallocs_now();
    // The return value is the caller-observable axis and is an int.
    // A collector that never counts unreachable objects has no count
    // to report, so the constant `interp_gc.py:48` carries is what it
    // answers.  `extra_tests/snippets/stdlib_gc.py` pins the type.
    Ok(w_int_new(0))
}

/// `interp_gc.py collect_step`, running `StepCollector.do`.
#[pyre_interpreter::pyre_function]
pub(super) fn collect_step() -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // interp_gc.py StepCollector: the app-level finalizer drain
    // is a virtual fifth state after the collector has returned to
    // SCANNING.
    if STEP_FINALIZING.load(Ordering::Acquire) {
        _run_finalizers()?;
        STEP_FINALIZING.store(false, Ordering::Release);
        return new_collect_step_stats(STATE_USERDEL, STATE_SCANNING, true);
    }

    let (oldstate, mut newstate) = pyre_object::gc_hook::try_gc_collect_step();
    if is_done_states(oldstate, newstate) {
        newstate = STATE_USERDEL;
        STEP_FINALIZING.store(true, Ordering::Release);
    }
    new_collect_step_stats(oldstate, newstate, false)
}

/// `interp_gc.py enable`.
pub(super) fn enable(_args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    pyre_object::gc_hook::try_gc_set_enabled(true);
    GC_ENABLED.store(true, Ordering::Relaxed);
    if let Some(uda) = user_del_action()
        && !uda.enabled_at_app_level
    {
        uda.enabled_at_app_level = true;
        __enable_finalizers_impl()?;
    }
    Ok(w_none())
}

/// `interp_gc.py disable`.
pub(super) fn disable(_args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    pyre_object::gc_hook::try_gc_set_enabled(false);
    GC_ENABLED.store(false, Ordering::Relaxed);
    if let Some(uda) = user_del_action()
        && uda.enabled_at_app_level
    {
        uda.enabled_at_app_level = false;
        __disable_finalizers_impl();
    }
    Ok(w_none())
}

/// `interp_gc.py isenabled`.
pub(super) fn isenabled(_args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let enabled = match user_del_action() {
        Some(action) => action.enabled_at_app_level,
        None => GC_ENABLED.load(Ordering::Relaxed),
    };
    Ok(w_bool_from(enabled))
}

/// `interp_gc.py enable_finalizers`.
#[pyre_interpreter::pyre_function]
pub(super) fn enable_finalizers() -> Result<(), pyre_interpreter::PyError> {
    // Unlike gc.enable(), an unmatched enable is an error rather than a
    // no-op. Before UserDelAction is installed there cannot have been a
    // matching disable, so that is the same zero lock depth.
    let Some(uda) = user_del_action().filter(|uda| uda.finalizers_lock_count > 0) else {
        return Err(pyre_interpreter::PyError::value_error(
            "finalizers are already enabled",
        ));
    };
    uda.finalizers_lock_count -= 1;
    if uda.finalizers_lock_count == 0
        && let Some(pending) = uda.pending_with_disabled_del.take()
    {
        // The list just left its GC-visible UserDelAction slot; keep every
        // entry rooted while the finalizers run (upstream clears the
        // GC-visible list as it progresses).
        let _roots = pyre_object::gc_roots::push_roots();
        for &obj in pending.iter() {
            let _ = pyre_object::gc_roots::pin_root(obj);
        }
        let root_end = pyre_object::gc_roots::shadow_stack_len();
        let root_base = root_end - pending.len();
        for index in 0..pending.len() {
            uda._call_finalizer(pyre_object::gc_roots::shadow_stack_get(root_base + index));
        }
    }
    Ok(())
}

/// `interp_gc.py disable_finalizers`.
#[pyre_interpreter::pyre_function]
pub(super) fn disable_finalizers() {
    // The lock is recursive and deliberately independent of gc.isenabled().
    if let Some(uda) = user_del_action() {
        uda.finalizers_lock_count += 1;
        if uda.pending_with_disabled_del.is_none() {
            uda.pending_with_disabled_del = Some(Vec::new());
        }
    }
}

// `set_threshold(threshold0, threshold1=None, threshold2=None)` — the
// optional tail leaves no single natural arity, so the body enforces
// the count itself.
pub(super) fn set_threshold(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let positional = args;
    // CPython 3.14 `gc.set_threshold(threshold0[, threshold1[,
    // threshold2]])` writes only the positions it was given, and
    // parses every argument before writing any of them.
    if positional.is_empty() || positional.len() > 3 {
        return Err(pyre_interpreter::PyError::type_error(
            "gc.set_threshold requires 1 to 3 arguments",
        ));
    }
    // Read every value before storing any, so a non-integer in the
    // tail leaves the previous thresholds untouched.  An omitted
    // trailing value keeps the threshold it already had.
    let mut given = Vec::with_capacity(positional.len());
    for &w_value in positional {
        // The index protocol, so an object carrying only `__int__` is
        // a TypeError rather than a silent conversion.
        given.push(pyre_interpreter::builtins::space_index_w(w_value)?);
    }
    for (slot, value) in GC_THRESHOLD.iter().zip(given) {
        slot.store(value, Ordering::Relaxed);
    }
    Ok(w_none())
}

pub(super) fn get_threshold(
    _args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    Ok(w_tuple_new(
        GC_THRESHOLD
            .iter()
            .map(|slot| w_int_new(slot.load(Ordering::Relaxed)))
            .collect(),
    ))
}

// `gc_get_count_impl` reads three fields and only the first is an
// object count: element 0 is tracked-container allocations minus
// deallocations since generation 0 was collected, while elements 1 and
// 2 count *collections* -- generation-0 collections since generation 1
// was collected, and generation-1 collections since generation 2 was.
// Collecting a generation zeroes its own count and every younger one.
//
// Under the generation mapping `NUM_GENERATIONS` already publishes --
// 0 is the nursery, 1 the generation this collector keeps empty, 2
// what is not in the nursery -- a minor collection is the generation-0
// one and a major collects both older generations at once.  So element
// 1 is the minors run since the last major, which is what `collect(0)`
// moves.  Element 2 is exact at zero: `collect(1)` is `collect(0)` plus
// "start the major now if one is not already running", so asking for
// the middle generation runs no collection of its own for element 2 to
// count -- there is no middle generation holding anything to reclaim.
//
// Element 0 stays zero because no counter can be truthful here.  The
// allocation seam is keyed by a majit type id and nothing else
// (`try_gc_alloc(type_id, payload_size)`), and the tracked predicate is
// not a function of that key -- `cpython_object_is_gc` reaches the
// object's type and, for a type object, the object itself, so one type
// id covers both a tracked heap type and an untracked static one.
// Even a decidable bit would undercount: every backend emits the
// nursery bump inline and merges several objects into one, so compiled
// code allocates without passing any counter site, and a virtualized
// allocation is removed outright.  Counting by walking instead is what
// `gc.get_objects` costs, four orders of magnitude above this call.
pub(super) fn get_count(_args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(w_int_new(0));
    fields.push(w_int_new(
        majit_gc::active_minor_collections_since_major() as i64
    ));
    fields.push(w_int_new(0));
    Ok(w_tuple_new(fields.take()))
}

pub(super) fn get_debug(_args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    Ok(w_int_new(GC_DEBUG.load(Ordering::Relaxed)))
}

pub(super) fn set_debug(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // `gc_set_debug_impl` parses a C int through the index protocol.
    // Convert before storing so a failed conversion leaves the old
    // process-wide word untouched.
    let flags = pyre_interpreter::baseobjspace::c_int_w(
        pyre_interpreter::baseobjspace::space_index(args[0])?,
    )?;
    GC_DEBUG.store(flags as i64, Ordering::Relaxed);
    Ok(w_none())
}

pub(super) fn is_tracked(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // CPython 3.14 `PyObject_GC_IsTracked` first requires
    // `_PyObject_IS_GC`, then asks the collector's tracked state. Host
    // fallback objects are outside MiniMark and retain their type-level
    // answer; managed objects must not bypass `GCBase.is_tracked`.
    let eligible = pyre_interpreter::typedef::cpython_object_is_gc(args[0]);
    let tracked = !majit_gc::gc_owns_object(args[0] as usize)
        || majit_gc::is_tracked(majit_ir::GcRef(args[0] as usize));
    Ok(w_bool_from(eligible && tracked))
}

pub(super) fn is_finalized(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    Ok(w_bool_from(majit_gc::gc_finalizer_has_run(
        args[0] as usize,
    )))
}

// `[3.14-spec]` `gc.freeze()` moves the surviving objects into a
// permanent generation that later collections skip; it is a pre-fork
// hint, not a semantic guarantee.  PyPy binds none of the three.  The
// collector has no permanent generation, so freezing and unfreezing
// are no-ops and the frozen count is the truthful zero.
//
// Rooting the live set instead would not be that operation under
// another name: a frozen object in 3.14 is skipped by the cyclic
// collector but still reclaimed by refcount, while a rooted one is
// immortal until `unfreeze` and has its `__del__` deferred until then.
// That is a third behaviour, matching neither side, and the whole live
// set is what it would apply to.
pub(super) fn freeze(_args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    Ok(w_none())
}

pub(super) fn unfreeze(_args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    Ok(w_none())
}

pub(super) fn get_freeze_count(
    _args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    Ok(w_int_new(0))
}
