//! Thread state and the GIL -- PyPy `cpyext/pystate.py`.
//!
//! A thread holds the GIL for as long as it runs pyre code, so an extension
//! that blocks has to give it up (`PyEval_SaveThread`, which
//! `Py_BEGIN_ALLOW_THREADS` expands to) and one that calls in from a thread of
//! its own has to take it (`PyGILState_Ensure`).  Both are the runtime's own
//! boundary guards; [`crate::module::thread`] owns them, because a build
//! without this layer exports `PyGILState_Ensure` and `PyGILState_Release` too
//! -- cffi's embedding header calls them.
//!
//! `PyThreadState` is opaque.  Upstream keeps one per execution context and
//! records whether it is the current one (`pystate.py PyThreadState_Get`,
//! `:228 PyThreadState_Swap` through `ec.cpyext_threadstate_is_current`); the
//! same pair here is per thread, since that is what an extension can name.

use std::cell::{Cell, UnsafeCell};
use std::ffi::c_int;
use std::sync::atomic::{AtomicU64, Ordering};

use super::pyobject::{self, CPyObject};

/// The interpreter an extension is handed a pointer to.
///
/// Opaque, and for the same reason `CPyThreadState` is: what it stands for is
/// reached through entry points rather than through the struct.
#[repr(C)]
pub struct CPyInterpreterState {
    _private: u8,
}

/// The one interpreter, which every thread runs in.
///
/// Nothing is kept behind the handle: it exists so that an extension asking
/// which interpreter it is in gets an answer to compare and to pass on, and
/// the address of a static gives one that is stable and never NULL.
static INTERPRETER: CPyInterpreterState = CPyInterpreterState { _private: 0 };

/// The per-thread state an extension is handed a pointer to.
///
/// `interp` and `dict` are the fields an extension reads directly, which is
/// how it asks which interpreter the thread runs in and where to keep its own
/// per-thread state without a call; everything else this stands for is reached
/// through an entry point.  `pystate.py` gives its `PyThreadState` the same
/// three, and `_status` is the bitfield `struct _ts` publishes beside them --
/// cffi clears `bound_gilstate` on a state it is about to delete.
#[repr(C)]
pub struct CPyThreadState {
    interp: *mut CPyInterpreterState,
    /// The namespace [`PyThreadState_GetDict`] answers with, minted on first
    /// read and released by [`PyThreadState_Clear`].
    dict: *mut CPyObject,
    /// What [`PyThreadState_GetID`] answers with.
    id: u64,
    /// The eight one-bit flags and their padding, as one 32-bit unit.  Nothing
    /// here reads them; they exist so that an extension writing one writes
    /// into storage of its own rather than past the end of the block.
    status: u32,
}

/// The layout is written out twice -- here and in
/// `include/pyre3.15t/pystate.h` -- so a field added to one without the other
/// stops compiling rather than writing somewhere unclaimed.
const _: () = {
    assert!(std::mem::offset_of!(CPyThreadState, interp) == 0);
    assert!(std::mem::offset_of!(CPyThreadState, dict) == 8);
    assert!(std::mem::offset_of!(CPyThreadState, id) == 16);
    assert!(std::mem::offset_of!(CPyThreadState, status) == 24);
};

/// `State.threadstate_count` -- what a state's id is taken from, counted up
/// so that no two live states share one.
static THREAD_STATE_COUNT: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// Stable for as long as the thread lives, which is as long as a
    /// `PyThreadState *` naming it is valid.
    static THREAD_STATE: UnsafeCell<CPyThreadState> = UnsafeCell::new(CPyThreadState {
        interp: &INTERPRETER as *const CPyInterpreterState as *mut CPyInterpreterState,
        dict: std::ptr::null_mut(),
        // `new_thread_state` counts from one, so every state that exists has
        // an id an extension can key its own per-thread data by.
        id: THREAD_STATE_COUNT.fetch_add(1, Ordering::Relaxed) + 1,
        status: 0,
    });
    /// `ec.cpyext_threadstate_is_current`.  Swapping NULL in clears it, which
    /// is how an extension says the thread state it was handed is no longer the
    /// one in force.
    static STATE_IS_CURRENT: Cell<bool> = const { Cell::new(true) };
}

/// This thread's state, whether or not it is the current one.
fn thread_state() -> *mut CPyThreadState {
    THREAD_STATE.with(|state| state.get())
}

/// Record whether this thread's state is the current one.
///
/// The GIL entry points that make it so live in [`crate::module::thread`],
/// because a build without this layer exports them too; the flag they set is
/// this one.
pub(crate) fn set_threadstate_is_current(current: bool) {
    STATE_IS_CURRENT.with(|state| state.set(current));
}

/// `pystate.py PyEval_SaveThread` — drop the GIL, and answer the state the
/// matching [`PyEval_RestoreThread`] is to be handed back.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyEval_SaveThread() -> *mut CPyThreadState {
    let state = thread_state();
    STATE_IS_CURRENT.with(|current| current.set(false));
    crate::module::thread::save_thread();
    state
}

/// `pystate.py PyEval_RestoreThread` — retake the GIL.
///
/// The argument is the state [`PyEval_SaveThread`] answered with, which for any
/// one thread is the only state it can be: the release is bound to the thread
/// that took it, so there is nothing to switch to.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyEval_RestoreThread(_state: *mut CPyThreadState) {
    crate::module::thread::restore_thread();
    STATE_IS_CURRENT.with(|current| current.set(true));
}

/// `PyEval_AcquireThread(tstate)` — take the GIL and make `tstate` current.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyEval_AcquireThread(state: *mut CPyThreadState) {
    unsafe { PyEval_RestoreThread(state) };
}

/// `PyEval_ReleaseThread(tstate)` — the drop half of
/// [`PyEval_AcquireThread`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyEval_ReleaseThread(_state: *mut CPyThreadState) {
    unsafe { PyEval_SaveThread() };
}

/// `pystate.py PyThreadState_Get`.
///
/// There is no state to answer with while this thread's is not current, and no
/// way to say so: the return is a state and every caller reads it.  So the
/// process ends here rather than one dereference later.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyThreadState_Get() -> *mut CPyThreadState {
    if !STATE_IS_CURRENT.with(|current| current.get()) {
        super::pyerrors::fatal_error(
            Some("PyThreadState_Get"),
            "the function must be called with the GIL held, after Python \
             initialization and before Python finalization, but the GIL is \
             released (the current Python thread state is NULL)",
        );
    }
    thread_state()
}

/// `pystate.py _PyThreadState_UncheckedGet` — [`PyThreadState_Get`]
/// without the fatal error, so NULL when this thread's state is not current.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyThreadState_UncheckedGet() -> *mut CPyThreadState {
    if STATE_IS_CURRENT.with(|current| current.get()) {
        thread_state()
    } else {
        std::ptr::null_mut()
    }
}

/// `pystate.py PyThreadState_Swap` — make `state` the current one and
/// answer the one it replaces, or NULL if there was none.
///
/// Upstream refuses to switch to a *different* thread state, which is the only
/// use this can be put to that it cannot serve; a thread can name no state but
/// its own, so NULL ("no state is current") is the only other argument.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyThreadState_Swap(state: *mut CPyThreadState) -> *mut CPyThreadState {
    let previous = if STATE_IS_CURRENT.with(|current| current.get()) {
        thread_state()
    } else {
        std::ptr::null_mut()
    };
    STATE_IS_CURRENT.with(|current| current.set(!state.is_null()));
    previous
}

/// The namespace a thread state answers with, minted on first read.
///
/// `InterpreterState.new_thread_state` mints one with the state itself
/// (`ts.c_dict = make_ref(space, space.newdict())`).  A state here is a
/// thread-local that exists before there is an interpreter to ask for a dict,
/// so the mint is deferred to the first reader; the reference it takes is the
/// one [`PyThreadState_Clear`] gives back.
///
/// # Safety
/// `state` must be null or a live thread state.
unsafe fn state_dict(state: *mut CPyThreadState) -> *mut CPyObject {
    if state.is_null() {
        return std::ptr::null_mut();
    }
    let dict = unsafe { (*state).dict };
    if !dict.is_null() {
        return dict;
    }
    let dict = pyobject::make_ref(pyre_object::dictmultiobject::w_dict_new());
    unsafe { (*state).dict = dict };
    dict
}

/// `pystate.py PyThreadState_GetDict` — the namespace an extension keeps its
/// own per-thread state in, as a borrowed reference.
///
/// NULL when this thread's state is not the current one, which upstream
/// reports the same way and which the documented contract asks for: a caller
/// is to read NULL as "no thread state is available" rather than as an error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyThreadState_GetDict() -> *mut CPyObject {
    if !STATE_IS_CURRENT.with(|current| current.get()) {
        return std::ptr::null_mut();
    }
    unsafe { state_dict(thread_state()) }
}

/// `pystate.py _PyThreadState_GetDict` — the same namespace, named by state
/// rather than taken from the current one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyThreadState_GetDict(state: *mut CPyThreadState) -> *mut CPyObject {
    unsafe { state_dict(state) }
}

/// `pystate.py PyThreadState_Clear` — give back what the state holds.
///
/// Upstream also leaves the thread and drops its execution context; a state
/// here is the thread's own thread-local and outlives every caller, so what is
/// left to release is the namespace.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyThreadState_Clear(state: *mut CPyThreadState) {
    if state.is_null() {
        return;
    }
    unsafe {
        pyobject::decref((*state).dict);
        (*state).dict = std::ptr::null_mut();
    }
}

/// `pystate.py PyThreadState_Delete` — upstream's body is empty, and for the
/// same reason: the storage belongs to the thread, not to the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyThreadState_Delete(_state: *mut CPyThreadState) {}

/// `pystate.py PyThreadState_DeleteCurrent`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyThreadState_DeleteCurrent() {}

/// `src/pythonrun.c PyThreadState_GetInterpreter` — the one field read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyThreadState_GetInterpreter(
    state: *mut CPyThreadState,
) -> *mut CPyInterpreterState {
    match state.is_null() {
        true => std::ptr::null_mut(),
        false => unsafe { (*state).interp },
    }
}

/// `PyThreadState_GetUnchecked()` — the name [`_PyThreadState_UncheckedGet`]
/// was published under from 3.13, and the same answer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyThreadState_GetUnchecked() -> *mut CPyThreadState {
    unsafe { _PyThreadState_UncheckedGet() }
}

/// `PyThreadState_GetID(tstate)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyThreadState_GetID(state: *mut CPyThreadState) -> u64 {
    match state.is_null() {
        true => 0,
        false => unsafe { (*state).id },
    }
}

/// The namespace [`PyInterpreterState_GetDict`] answers with.
///
/// One interpreter, so one namespace, and it lives as long as the process:
/// the reference taken when it is minted is never given back, which is what
/// makes the borrowed answer good for as long as the caller holds the
/// interpreter.
static INTERPRETER_DICT: super::ForkMutex<usize> = super::ForkMutex::new(0);

/// The lock word can be held by a thread the child does not have; the mirror
/// the payload names is still mapped, because `fork` copies the address space.
pub(super) unsafe fn after_fork_child() {
    unsafe { INTERPRETER_DICT.reinit_after_fork() };
}

/// `PyInterpreterState_GetDict(interp)` — a namespace shared by every thread
/// of an interpreter, as a borrowed reference.
///
/// cffi keeps its `@ffi.def_extern()` table here, so the answer has to be the
/// same block on every call from every thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyInterpreterState_GetDict(
    interp: *mut CPyInterpreterState,
) -> *mut CPyObject {
    if interp.is_null() {
        return std::ptr::null_mut();
    }
    let mut held = INTERPRETER_DICT.lock();
    if *held == 0 {
        *held = pyobject::make_ref(pyre_object::dictmultiobject::w_dict_new()) as usize;
    }
    *held as *mut CPyObject
}

/// `pystate.py PyGILState_Check` — whether this thread holds the GIL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyGILState_Check() -> c_int {
    crate::module::thread::gilstate_check() as c_int
}

/// `PyGILState_GetThisThreadState()` — this thread's state, or NULL when it has
/// never held the GIL and so has no state of pyre's to name.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyGILState_GetThisThreadState() -> *mut CPyThreadState {
    if crate::module::thread::gilstate_check() {
        thread_state()
    } else {
        std::ptr::null_mut()
    }
}

/// `PyInterpreterState_Get()` — the interpreter the calling thread runs in.
///
/// There is no interpreter to answer with while this thread's state is not
/// current, and no way to say so, so this ends the process where
/// [`PyThreadState_Get`] does and for the same reason.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyInterpreterState_Get() -> *mut CPyInterpreterState {
    if !STATE_IS_CURRENT.with(|current| current.get()) {
        super::pyerrors::fatal_error(
            Some("PyInterpreterState_Get"),
            "the function must be called with the GIL held, after Python \
             initialization and before Python finalization, but the GIL is \
             released (the current Python thread state is NULL)",
        );
    }
    &INTERPRETER as *const CPyInterpreterState as *mut CPyInterpreterState
}

/// `PyInterpreterState_GetID(interp)` — the identifier an interpreter is
/// known by, which is 0 for the one pyre runs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyInterpreterState_GetID(interp: *mut CPyInterpreterState) -> i64 {
    if interp.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    0
}

/// `pystate.py PyEval_InitThreads` — threads are always available, so
/// upstream's `setup_threads` has nothing left to do here.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyEval_InitThreads() {}

/// `pystate.py PyEval_ThreadsInitialized`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyEval_ThreadsInitialized() -> c_int {
    1
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyEval_SaveThread as *const ());
    std::hint::black_box(PyEval_RestoreThread as *const ());
    std::hint::black_box(PyEval_AcquireThread as *const ());
    std::hint::black_box(PyEval_ReleaseThread as *const ());
    std::hint::black_box(PyThreadState_Get as *const ());
    std::hint::black_box(_PyThreadState_UncheckedGet as *const ());
    std::hint::black_box(PyThreadState_Swap as *const ());
    std::hint::black_box(PyGILState_Check as *const ());
    std::hint::black_box(PyGILState_GetThisThreadState as *const ());
    std::hint::black_box(PyInterpreterState_Get as *const ());
    std::hint::black_box(PyInterpreterState_GetID as *const ());
    std::hint::black_box(PyEval_InitThreads as *const ());
    std::hint::black_box(PyEval_ThreadsInitialized as *const ());
    std::hint::black_box(PyThreadState_GetDict as *const ());
    std::hint::black_box(_PyThreadState_GetDict as *const ());
    std::hint::black_box(PyThreadState_Clear as *const ());
    std::hint::black_box(PyThreadState_Delete as *const ());
    std::hint::black_box(PyThreadState_DeleteCurrent as *const ());
    std::hint::black_box(PyThreadState_GetInterpreter as *const ());
    std::hint::black_box(PyThreadState_GetUnchecked as *const ());
    std::hint::black_box(PyThreadState_GetID as *const ());
    std::hint::black_box(PyInterpreterState_GetDict as *const ());
}
