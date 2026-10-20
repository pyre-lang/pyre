//! Frame mirrors -- PyPy `cpyext/frameobject.py`.
//!
//! A frame is the one interpreter object whose mirror carries fields of its
//! own.  Everything else this runtime defines is exactly a [`CPyObject`], and
//! an extension reaches its contents through calls; a frame is different
//! because an extension builds one itself and then writes its line number,
//! which is a field assignment no entry point stands in for.
//!
//! The block therefore has to be [`CPyFrameObject`]-sized rather than
//! `PyObject`-sized, which is what [`basicsize`] tells the allocator through
//! `tp_basicsize`.  Getting that wrong is not a failure that reports itself:
//! the writes below would land past the end of a header-sized block.  [`fields`] is
//! the one place that turns a mirror into a frame mirror, and it refuses on a
//! block that was not sized for one.

use super::pyobject::{self, CPyObject, REFCNT_FROM_PYPY};
use super::typeobject::CPyTypeObject;
use pyre_object::{PY_NULL, PyObjectRef};
use std::collections::HashSet;
use std::ffi::c_int;
use std::hash::BuildHasherDefault;

/// C-visible `PyFrameObject`, the twin of `struct _frame` in
/// `include/pyre3.15t/frameobject.h`.
///
/// Field order is `frameobject.py PyFrameObjectFields`.
#[repr(C)]
pub struct CPyFrameObject {
    pub ob_base: CPyObject,
    pub f_code: *mut CPyObject,
    pub f_globals: *mut CPyObject,
    pub f_locals: *mut CPyObject,
    pub f_lineno: c_int,
    pub f_back: *mut CPyFrameObject,
}

/// The layout is written out twice — here and in the header — so each offset
/// is pinned in both, and a field added to one without the other stops
/// compiling rather than writing somewhere unclaimed.
///
/// `pyre/pyrex/tests/fixtures/cpyext_frames.c` carries the C half.
const _: () = {
    assert!(std::mem::offset_of!(CPyFrameObject, ob_base) == 0);
    assert!(std::mem::offset_of!(CPyFrameObject, f_code) == size_of::<CPyObject>());
    assert!(
        std::mem::offset_of!(CPyFrameObject, f_globals)
            == size_of::<CPyObject>() + size_of::<usize>()
    );
    assert!(
        std::mem::offset_of!(CPyFrameObject, f_locals)
            == size_of::<CPyObject>() + 2 * size_of::<usize>()
    );
    assert!(
        std::mem::offset_of!(CPyFrameObject, f_lineno)
            == size_of::<CPyObject>() + 3 * size_of::<usize>()
    );
    assert!(
        std::mem::offset_of!(CPyFrameObject, f_back)
            == size_of::<CPyObject>() + 4 * size_of::<usize>()
    );
    assert!(size_of::<CPyFrameObject>() == size_of::<CPyObject>() + 5 * size_of::<usize>());
};

type PendingSet = super::address_table::HeldSet;

/// Mirrors [`PyFrame_New`] handed out that have no interpreter frame yet.
use super::address_table::{AddressTable, hold};

static PENDING: AddressTable<PendingSet> =
    AddressTable::new(HashSet::with_hasher(BuildHasherDefault::new()));

pub(super) unsafe fn after_fork_child() {
    unsafe { PENDING.reinit_after_fork() };
}

/// The interpreter's frame type, or null before it is built.
fn frame_type() -> PyObjectRef {
    crate::typedef::gettypeobject(&crate::pyframe::FRAME_TYPE)
}

fn is_frame(w_obj: PyObjectRef) -> bool {
    !w_obj.is_null() && unsafe { pyre_object::py_type_check(w_obj, &crate::pyframe::FRAME_TYPE) }
}

/// What `tp_basicsize` a synthesized mirror of `w_type` carries —
/// `frameobject.py basicstruct=PyFrameObject.TO` for the frame type, and
/// the `PyObject`-sized default for every other type this runtime defines.
pub(super) fn basicsize(w_type: PyObjectRef) -> isize {
    let frame_type = frame_type();
    match !frame_type.is_null() && std::ptr::eq(w_type, frame_type) {
        true => size_of::<CPyFrameObject>() as isize,
        false => 0,
    }
}

/// A mirror's frame fields, or `None` when it is not a frame mirror.
///
/// The type is what decides it.  A size test would answer yes for any
/// C-defined type whose own storage reaches this far, and reading its fields
/// as a frame's is exactly the corruption this module exists to prevent — the
/// first fixture to declare a large enough `tp_basicsize` found that out.
///
/// The size is then asserted rather than tested: a block that is a frame's and
/// too small for one means [`basicsize`] and the allocator have come apart,
/// and the writes below would land past its end.
fn fields(raw: *mut CPyObject) -> Option<*mut CPyFrameObject> {
    if raw.is_null() {
        return None;
    }
    let tp = unsafe { (*raw).ob_type };
    // `as_pyobj` rather than a realizing read: this runs while a mirror is
    // being deallocated, where nothing may build an interpreter object.  A
    // frame mirror cannot exist before its type's does, so a null answer here
    // is a definite no.
    let frame_tp = pyobject::as_pyobj(frame_type()) as *mut CPyTypeObject;
    if tp.is_null() || frame_tp.is_null() || !std::ptr::eq(tp, frame_tp) {
        return None;
    }
    assert!(
        unsafe { (*tp).tp_basicsize } >= size_of::<CPyFrameObject>() as isize,
        "a frame mirror was allocated at the plain PyObject size"
    );
    Some(raw as *mut CPyFrameObject)
}

/// Fill a freshly allocated mirror of `w_obj` when `w_obj` is a frame —
/// `frameobject.py frame_attach`.
///
/// Reached from `pyobject::ensure_mirror`, so an extension handed a frame this
/// runtime is executing reads the same fields as one it built itself.
pub(super) fn attach(raw: *mut CPyObject, w_obj: PyObjectRef) {
    if !is_frame(w_obj) {
        return;
    }
    let Some(py_frame) = fields(raw) else {
        return;
    };
    // `make_ref` allocates, so the frame is read back through the mirror
    // before each field rather than kept in a local: the block's address does
    // not move and the frame's does.  What `make_ref` answers with is a block
    // of its own, which is why those are held.
    let frame = || unsafe { (*raw).ob_pyre_link } as *mut crate::pyframe::PyFrame;
    let code = pyobject::make_ref(unsafe { (*frame()).fget_f_code() });
    let globals = pyobject::make_ref(unsafe { (*frame()).get_w_globals() });
    let locals = pyobject::make_ref(unsafe { (*frame()).get_w_locals() });
    // The line as a number.  `fget_f_lineno` is the property, which answers an
    // object and `None` where there is no line; `frame_attach`'s own comment
    // names `get_last_lineno` as what the field is owed.
    let lineno = unsafe { (*frame()).get_last_lineno() } as c_int;
    let back = unsafe { (*frame()).get_f_back() } as PyObjectRef;
    let back = pyobject::make_ref(back) as *mut CPyFrameObject;
    unsafe {
        (*py_frame).f_code = code;
        (*py_frame).f_globals = globals;
        (*py_frame).f_locals = locals;
        (*py_frame).f_lineno = lineno;
        (*py_frame).f_back = back;
    }
}

/// Drop the record that a frame mirror is still owed its fields.
pub(super) fn forget_pending(raw: *mut CPyObject) {
    PENDING.discard(raw as usize);
}

/// Release the references a frame mirror owns — `frameobject.py:52-60
/// frame_dealloc`.
///
/// Reached from `pyobject::dealloc`, which runs it while the block is still
/// live and before anything can hand the address back out.
pub(super) fn forget_block(raw: *mut CPyObject) {
    let Some(py_frame) = fields(raw) else {
        return;
    };
    unsafe {
        for reference in [
            (*py_frame).f_code,
            (*py_frame).f_globals,
            (*py_frame).f_locals,
            (*py_frame).f_back as *mut CPyObject,
        ] {
            pyobject::decref(reference);
        }
        (*py_frame).f_code = std::ptr::null_mut();
        (*py_frame).f_globals = std::ptr::null_mut();
        (*py_frame).f_locals = std::ptr::null_mut();
        (*py_frame).f_back = std::ptr::null_mut();
    }
}

/// Build the interpreter frame a mirror [`PyFrame_New`] handed out stands for
/// — `frameobject.py frame_realize`.
///
/// Reached from `pyobject::realize`, so the frame is built at the first point
/// something reads the mirror as a value.  What C wrote up to then is what it
/// carries, which is why upstream's "must not be modified after this call"
/// holds here too: a later `f_lineno` write reaches the block alone.
pub(super) fn realize_pending(raw: *mut CPyObject) {
    if raw.is_null() {
        return;
    }
    {
        let mut pending = PENDING.lock();
        if !pending.remove(&(raw as usize)) {
            return;
        }
    }
    let Some(py_frame) = fields(raw) else {
        return;
    };
    let w_code = unsafe { pyobject::from_ref((*py_frame).f_code) };
    let w_globals = unsafe { pyobject::from_ref((*py_frame).f_globals) };
    if w_code.is_null() || w_globals.is_null() {
        return;
    }
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    let _ = roots.pin_root(w_code);
    let _ = roots.pin_root(w_globals);
    // `frameobject.py:72 space.FrameClass(space, code, w_globals,
    // outer_func=None)` — no arguments and no closure, so the locals array is
    // whatever the code object's own counts ask for.
    let Ok(frame) = crate::pyframe::PyFrame::try_new_for_call_with_closure_and_globals_obj(
        roots.get(base) as *const (),
        &[],
        roots.get(base + 1),
        crate::call::take_last_exec_ctx(),
        PY_NULL,
        crate::pyframe::FrameLocalsArrayAllocation::NurseryGc,
    ) else {
        return;
    };
    let mut boxed = crate::pyframe::FrameBox::new(frame);
    // `frameobject.py:75 d.f_lineno = rffi.getintfield(py_frame,
    // 'c_f_lineno')` — the line the extension wrote before handing the frame
    // on, which is the whole reason it asked for one.
    boxed.getorcreatedebug(-1).f_lineno = unsafe { (*py_frame).f_lineno } as isize;
    if !boxed.is_gc_owned() {
        // A frame the collector does not own is freed when this handle dies,
        // and the mirror would outlive it.  Only the pre-hook bootstrap window
        // allocates that way, and no extension is loaded in it.
        return;
    }
    let refcnt = unsafe { (*raw).ob_refcnt };
    pyobject::link_allocated(
        boxed.into_raw() as PyObjectRef,
        raw,
        REFCNT_FROM_PYPY + refcnt,
    );
}

/// `frameobject.py PyFrame_New` — a frame the caller fills in and hands
/// to [`super::pytraceback::PyTraceBack_Here`].
///
/// The frame itself is not built here: `locals` is routinely NULL and the line
/// number is written afterwards, so the interpreter frame is built at the
/// first read instead ([`realize_pending`]).
///
/// # Safety
/// `code`, `globals` and `locals` must be null or live mirrors.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFrame_New(
    _tstate: *mut super::pystate::CPyThreadState,
    code: *mut CPyObject,
    globals: *mut CPyObject,
    locals: *mut CPyObject,
) -> *mut CPyFrameObject {
    super::object::realize_all([code, globals, locals]);
    let w_code = unsafe { pyobject::from_ref(code) };
    if w_code.is_null() || !unsafe { pyre_object::py_type_check(w_code, &crate::pycode::CODE_TYPE) }
    {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let w_frame_type = frame_type();
    if w_frame_type.is_null() {
        return unsafe { super::pyerrors::PyErr_NoMemory() } as *mut CPyFrameObject;
    }
    let ob_type = pyobject::borrow_mirror(w_frame_type) as *mut CPyTypeObject;
    let raw = pyobject::allocate_raw(size_of::<CPyFrameObject>(), true) as *mut CPyObject;
    if raw.is_null() {
        return unsafe { super::pyerrors::PyErr_NoMemory() } as *mut CPyFrameObject;
    }
    unsafe {
        // One reference, the caller's; no link share, because there is nothing
        // linked yet.
        (*raw).ob_refcnt = 1;
        (*raw).ob_pyre_link = PY_NULL;
        (*raw).ob_type = ob_type;
    }
    PENDING.lock().insert(hold(raw as usize));
    let py_frame = raw as *mut CPyFrameObject;
    unsafe {
        (*py_frame).f_code = pyobject::make_ref(w_code);
        (*py_frame).f_globals = pyobject::make_ref(pyobject::from_ref(globals));
        (*py_frame).f_locals = pyobject::make_ref(pyobject::from_ref(locals));
    }
    py_frame
}

/// `frameobject.py PyTraceBack_Here` — prepend `frame` to the pending
/// exception's traceback.
///
/// `-1` for no pending exception is upstream's answer and CPython's: there is
/// nothing to attach a traceback to.
///
/// # Safety
/// `frame` must be null or a live frame mirror.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTraceBack_Here(frame: *mut CPyFrameObject) -> c_int {
    if !super::pyerrors::has_pending_error() {
        return -1;
    }
    let w_frame = unsafe { pyobject::from_ref(frame as *mut CPyObject) };
    if !is_frame(w_frame) {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let Some(w_exception) = super::pyerrors::pending_exception() else {
        return -1;
    };
    unsafe {
        crate::pytraceback::record_application_traceback(
            w_exception,
            w_frame as *mut crate::pyframe::PyFrame,
            0,
        )
    };
    0
}
