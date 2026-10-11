use pyre_object::PyObjectRef;
use pyre_object::interp_exceptions::{
    ExcKind, exc_kind_name, w_exception_new, w_exception_new_wtf8,
};
use ruff_text_size::Ranged;
use rustpython_compiler::{ast, parser};
use rustpython_wtf8::Wtf8Buf;
use std::collections::HashSet;
use std::io::Write;

/// Short-lived normalizer for the three-argument generator `throw` surface.
///
/// This is not an interpreter exception carrier: it only converts the legacy
/// `(type, value, traceback)` input spelling into one application exception
/// before evaluation resumes.  Live exception propagation uses
/// [`OperationError`] below.
#[derive(Debug, Clone)]
pub(crate) struct ExceptionNormalization {
    pub w_type: PyObjectRef,
    pub w_value: PyObjectRef,
    pub _application_traceback: Option<PyObjectRef>,
}

/// Test whether a live exception object is an instance of StopIteration.
/// The class cache is populated only after registry lookup succeeds, so a
/// call before exception-class initialisation returns false without caching
/// absence forever.
#[majit_macros::dont_look_inside]
pub fn exception_object_matches_stop_iteration(exc_object: PyObjectRef) -> bool {
    static STOP_ITERATION_CLASS: pyre_object::gc_roots::RootedOnceRef =
        pyre_object::gc_roots::RootedOnceRef::new();
    let stop_iteration = match STOP_ITERATION_CLASS.get() {
        Some(class) => class,
        None => {
            let Some(class) = crate::builtins::lookup_exc_class("StopIteration") else {
                return false;
            };
            STOP_ITERATION_CLASS.set(class);
            class
        }
    };
    crate::eval::check_exc_match_against(exc_object, stop_iteration)
}

/// Test whether a live exception object is an instance of StopAsyncIteration.
/// Deliberately a separate function rather than a shared parameterised lookup:
/// `exception_object_matches_stop_iteration` is recognised by its body shape in
/// `majit-translate front::result_exc` and addressed by its symbol path in
/// `jit_fnaddr`, both of which a refactor would break silently.
#[majit_macros::dont_look_inside]
pub fn exception_object_matches_stop_async_iteration(exc_object: PyObjectRef) -> bool {
    static STOP_ASYNC_ITERATION_CLASS: pyre_object::gc_roots::RootedOnceRef =
        pyre_object::gc_roots::RootedOnceRef::new();
    let stop_async_iteration = match STOP_ASYNC_ITERATION_CLASS.get() {
        Some(class) => class,
        None => {
            let Some(class) = crate::builtins::lookup_exc_class("StopAsyncIteration") else {
                return false;
            };
            STOP_ASYNC_ITERATION_CLASS.set(class);
            class
        }
    };
    crate::eval::check_exc_match_against(exc_object, stop_async_iteration)
}

impl ExceptionNormalization {
    pub fn new(w_type: PyObjectRef, w_value: PyObjectRef) -> Self {
        Self {
            w_type,
            w_value,
            _application_traceback: None,
        }
    }

    pub fn get_w_value(&self, _space: PyObjectRef) -> PyObjectRef {
        let _ = _space;
        self.w_value
    }

    /// pypy/interpreter/error.py `normalize_exception`.
    ///
    /// ```python
    /// def normalize_exception(self, space):
    ///     w_type = self.w_type
    ///     w_value = self.get_w_value(space)
    ///     if space.exception_is_valid_obj_as_class_w(w_type):
    ///         if space.is_w(w_value, space.w_None):
    ///             w_value = space.call_function(w_type)
    ///             w_type = self._exception_getclass(space, w_value)
    ///         else:
    ///             w_valuetype = space.exception_getclass(w_value)
    ///             if space.exception_issubclass_w(w_valuetype, w_type):
    ///                 w_type = w_valuetype
    ///             else:
    ///                 if space.isinstance_w(w_value, space.w_tuple):
    ///                     w_value = space.call(w_type, w_value)
    ///                 else:
    ///                     w_value = space.call_function(w_type, w_value)
    ///                 w_type = self._exception_getclass(space, w_value)
    ///         if self._application_traceback:
    ///             from pypy.interpreter.pytraceback import PyTraceback
    ///             from pypy.module.exceptions.interp_exceptions import W_BaseException
    ///             tb = self._application_traceback
    ///             if (isinstance(w_value, W_BaseException) and
    ///                 isinstance(tb, PyTraceback)):
    ///                 # traceback hasn't escaped yet
    ///                 w_value.w_traceback = tb
    ///             else:
    ///                 # traceback has escaped
    ///                 space.setattr(w_value, space.newtext("__traceback__"),
    ///                               self.get_w_traceback(space))
    ///     else:
    ///         w_inst = w_type
    ///         w_instclass = self._exception_getclass(space, w_inst)
    ///         if not space.is_w(w_value, space.w_None):
    ///             raise oefmt(space.w_TypeError, ...)
    ///         w_value = w_inst
    ///         w_type = w_instclass
    ///     self.w_type = w_type
    ///     self._w_value = w_value
    ///     return w_value
    /// ```
    ///
    /// Mutates `self` to install the normalized `(w_type, w_value)` and
    /// returns the new `w_value`.
    ///
    /// `pypy/interpreter/error.py normalize_exception` traceback
    /// attach: the W_BaseException-typed fast path writes
    /// `w_value.w_traceback = tb` directly; the generic fallback is
    /// `space.setattr(w_value, "__traceback__", tb)`.  Pyre's
    /// W_BaseException grew the typed `w_traceback` slot, so the
    /// fast path is now reachable — `w_exception_set_traceback` is
    /// invoked when `w_value` is an exception instance, falling back
    /// to the generic setattr for anything else.
    #[majit_macros::unroll_safe]
    pub fn normalize_exception(&mut self, space: PyObjectRef) -> Result<PyObjectRef, PyError> {
        let mut w_type = self.w_type;
        let mut w_value = self.get_w_value(space);
        unsafe {
            if crate::baseobjspace::exception_is_valid_obj_as_class_w(w_type) {
                if w_value.is_null() || w_value == pyre_object::w_none() {
                    // error.py:208-210 (Class, None): instantiate Class()
                    // `error.py OperationError.normalize_exception` reads
                    // `w_type` again after each application-level constructor
                    // call.  The RPython GC transform restores that livevar;
                    // native Rust must perform the same pop-roots writeback.
                    w_value = pyre_object::with_roots!(w_type =>
                        crate::baseobjspace::call_function(w_type, &[])
                    );
                    if w_value.is_null() {
                        return Err(crate::call::take_call_error()
                            .unwrap_or_else(|| PyError::type_error("constructor failed")));
                    }
                    w_type = pyre_object::with_roots!(w_value =>
                        self._exception_getclass_from_call(w_type, w_value)
                    )?;
                } else {
                    // error.py:212 w_valuetype = space.exception_getclass(w_value)
                    let w_valuetype = crate::baseobjspace::exception_getclass(w_value);
                    if !w_valuetype.is_null()
                        && crate::baseobjspace::exception_issubclass_w(w_valuetype, w_type)
                    {
                        // error.py:213-215 (Class, inst): use inst's exact class
                        w_type = w_valuetype;
                    } else {
                        // error.py:217 if space.isinstance_w(w_value, space.w_tuple):
                        let w_tuple_cls =
                            crate::typedef::gettypeobject(&pyre_object::pyobject::TUPLE_TYPE);
                        if !w_tuple_cls.is_null()
                            && crate::baseobjspace::isinstance_w(w_value, w_tuple_cls)
                        {
                            // error.py:218-220 (Class, tuple): Class(*tuple)
                            let items = pyre_object::w_tuple_items_copy_as_vec(w_value);
                            // The snapshot is a plain `Vec` and roots nothing:
                            // on a specialised tuple its elements are freshly
                            // re-boxed, and `call_function` runs application
                            // code.  Publish them and read the live words back
                            // at the call.
                            let item_roots = pyre_object::gc_roots::push_roots();
                            let item_base = item_roots.publish(&items);
                            item_roots.normalize(item_base, items.len());
                            let items: Vec<pyre_object::PyObjectRef> = (0..items.len())
                                .map(|i| item_roots.get(item_base + i))
                                .collect();
                            w_value = pyre_object::with_roots!(w_type =>
                                crate::baseobjspace::call_function(w_type, &items)
                            );
                        } else {
                            // error.py:221-223 (Class, x): Class(x)
                            w_value = pyre_object::with_roots!(w_type =>
                                crate::baseobjspace::call_function(w_type, &[w_value])
                            );
                        }
                        if w_value.is_null() {
                            return Err(crate::call::take_call_error()
                                .unwrap_or_else(|| PyError::type_error("constructor failed")));
                        }
                        w_type = pyre_object::with_roots!(w_value =>
                            self._exception_getclass_from_call(w_type, w_value)
                        )?;
                    }
                }
                // error.py:225-236 traceback attach — fast path writes
                // `w_value.w_traceback = tb` when `w_value` is a
                // `W_BaseException`, otherwise falls through to the
                // generic `space.setattr(w_value, "__traceback__", tb)`.
                if let Some(tb) = self._application_traceback {
                    if pyre_object::is_exception(w_value) {
                        pyre_object::interp_exceptions::w_exception_set_traceback(w_value, tb);
                    } else {
                        pyre_object::with_roots!(w_type, w_value =>
                            crate::baseobjspace::setattr_str(w_value, "__traceback__", tb)
                        )?;
                    }
                }
            } else {
                // error.py (inst, None) — `raise inst`
                let w_inst = w_type;
                let w_instclass = self._exception_getclass(w_inst)?;
                if !w_value.is_null() && w_value != pyre_object::w_none() {
                    return Err(PyError::type_error(
                        "instance exception may not have a separate value",
                    ));
                }
                w_value = w_inst;
                w_type = w_instclass;
            }
        }
        self.w_type = w_type;
        self.w_value = w_value;
        Ok(w_value)
    }

    /// pypy/interpreter/error.py `_exception_getclass(space, w_inst, what="exceptions")`.
    ///
    /// ```python
    /// def _exception_getclass(self, space, w_inst, what="exceptions"):
    ///     w_type = space.exception_getclass(w_inst)
    ///     if not space.exception_is_valid_class_w(w_type):
    ///         raise oefmt(space.w_TypeError, ...)
    ///     return w_type
    /// ```
    fn _exception_getclass(&self, w_inst: PyObjectRef) -> Result<PyObjectRef, PyError> {
        let w_type = crate::baseobjspace::exception_getclass(w_inst);
        if w_type.is_null() || !unsafe { crate::baseobjspace::exception_is_valid_class_w(w_type) } {
            return Err(PyError::type_error(
                "exceptions must derive from BaseException",
            ));
        }
        Ok(w_type)
    }

    /// Python 3.14 exception normalization diagnostic for a class call whose
    /// `__new__` returned a non-exception.  This is the class-construction
    /// counterpart of `_exception_getclass`; direct `raise non_exception`
    /// retains the ordinary "exceptions must derive" wording.
    fn _exception_getclass_from_call(
        &self,
        w_constructor: PyObjectRef,
        w_inst: PyObjectRef,
    ) -> Result<PyObjectRef, PyError> {
        let w_type = crate::baseobjspace::exception_getclass(w_inst);
        if w_type.is_null() || !unsafe { crate::baseobjspace::exception_is_valid_class_w(w_type) } {
            return Err(exception_from_call_type_error(w_constructor, w_inst));
        }
        Ok(w_type)
    }
}

/// The TypeError raised when instantiating a raised exception class yields
/// something that is not a `BaseException` instance:
/// "calling C should have returned an instance of BaseException, not T".
/// A direct `raise non_exception`, which never calls a constructor, keeps
/// the ordinary "exceptions must derive from BaseException" wording.
pub fn exception_from_call_type_error(w_constructor: PyObjectRef, w_inst: PyObjectRef) -> PyError {
    // Read the class off `w_inst` before anything allocates: `py_repr_wtf8`
    // runs Python and can drive a collection, and `w_inst` is whatever the
    // constructor answered — a list moves.  `w_type` and `w_constructor` are
    // type objects, which do not.
    let mut w_type = crate::baseobjspace::exception_getclass(w_inst);
    let constructor =
        pyre_object::with_roots!(w_type => unsafe { crate::display::py_repr_wtf8(w_constructor) })
            .unwrap_or_else(|_| Wtf8Buf::from_string("<exception class>".to_string()));
    let unknown = || Wtf8Buf::from_string("<unknown type>".to_string());
    let returned_type = if w_type.is_null() {
        unknown()
    } else {
        unsafe { crate::display::py_repr_wtf8(w_type) }.unwrap_or_else(|_| unknown())
    };
    PyError::type_error(crate::display::wtf8_format!(
        "calling ",
        constructor,
        " should have returned an instance of BaseException, not ",
        returned_type
    ))
}

/// `pypy/interpreter/error.py _break_context_cycle` parity —
/// Floyd cycle-detection over the `__context__` chain, breaking the
/// loop by writing `None` into the offending link before the new
/// `w_context` is attached.
fn _break_context_cycle(w_value: PyObjectRef, w_context: PyObjectRef) -> Result<(), PyError> {
    let mut w_rabbit = w_context;
    let mut w_tortoise = w_context;
    let mut update_tortoise_toggle = false;
    loop {
        if !unsafe { pyre_object::is_exception(w_rabbit) } {
            return Err(PyError::new(
                crate::PyErrorKind::SystemError,
                "not an instance of Exception".to_string(),
            ));
        }
        let w_next = unsafe { pyre_object::interp_exceptions::w_exception_get_context(w_rabbit) };
        if w_next.is_null() || unsafe { pyre_object::is_none(w_next) } {
            break;
        }
        if std::ptr::eq(w_next, w_value) {
            // `:497-498` — `w_rabbit.descr_setcontext(space, space.w_None)`
            // (a real None, mirroring PyPy's "internal None" → user-
            // visible None via the typed slot).
            unsafe {
                pyre_object::interp_exceptions::w_exception_set_context(
                    w_rabbit,
                    pyre_object::w_none(),
                )
            };
            break;
        }
        w_rabbit = w_next;
        if std::ptr::eq(w_rabbit, w_tortoise) {
            // `:502-503` — pre-existing cycle; don't set anything to None.
            break;
        }
        if update_tortoise_toggle {
            if !unsafe { pyre_object::is_exception(w_tortoise) } {
                return Err(PyError::new(
                    crate::PyErrorKind::RuntimeError,
                    "not an instance of Exception".to_string(),
                ));
            }
            w_tortoise =
                unsafe { pyre_object::interp_exceptions::w_exception_get_context(w_tortoise) };
        }
        update_tortoise_toggle = !update_tortoise_toggle;
    }
    Ok(())
}

/// `_PyErr_FormatFromCause`: a `SystemError` carrying `cause` as both its
/// `__context__` and its `__cause__`.
///
/// A callee that reported success while leaving an exception set has already
/// lost the raise site, so the exception it left behind is the only
/// description of what went wrong and is chained onto the report rather than
/// discarded.  `_Py_CheckFunctionResult` raises through here, and so does the
/// C-API shim's own success-with-exception check.
pub(crate) fn system_error_from_cause(message: String, mut cause: PyError) -> PyError {
    let roots = pyre_object::gc_roots::push_roots();
    let cause_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(cause.to_exc_object());
    let mut error = PyError::new(PyErrorKind::SystemError, message);
    let error_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(error.to_exc_object());
    let w_cause = pyre_object::gc_roots::shadow_stack_get(cause_slot);
    let w_error = pyre_object::gc_roots::shadow_stack_get(error_slot);
    unsafe {
        pyre_object::interp_exceptions::w_exception_set_context(w_error, w_cause);
        pyre_object::interp_exceptions::w_exception_set_cause(w_error, w_cause);
        pyre_object::interp_exceptions::w_exception_set_suppress_context(w_error, true);
    }
    unsafe { PyError::from_exc_object(w_error) }
}

/// Whether [`chain_context`] would write `exc.__context__`.
///
/// A residual call that takes `exc` forces a still-virtual exception even
/// when the body would return without storing.  The in-trace catch
/// compensation uses this to skip that call on a no-op (self-reraise,
/// already-stamped context, no active handler).
pub fn chain_context_would_write(exc: PyObjectRef, active: PyObjectRef) -> bool {
    if exc.is_null()
        || active.is_null()
        || std::ptr::eq(exc, active)
        || unsafe { pyre_object::is_none(active) }
        || !unsafe { pyre_object::is_exception(exc) }
        || !unsafe { pyre_object::is_exception(active) }
    {
        return false;
    }
    unsafe { pyre_object::interp_exceptions::w_exception_get_context(exc) }.is_null()
}

/// Record `active` as `exc.__context__` when `exc` is raised while `active`
/// is the exception currently being handled — the shared primitive for both
/// explicit (`raise`) and implicit (builtin/operator) raises.  Only writes
/// when no `__context__` is already stamped (so an exception that already
/// carries one — e.g. an explicit `raise ... from` or a re-raise — is left
/// untouched) and skips self-context; any cycle through the context chain is
/// broken first via `_break_context_cycle`.
pub fn chain_context(exc: PyObjectRef, active: PyObjectRef) {
    if !chain_context_would_write(exc, active) {
        return;
    }
    let _ = _break_context_cycle(exc, active);
    unsafe { pyre_object::interp_exceptions::w_exception_set_context(exc, active) };
}

#[derive(Debug, Clone)]
pub struct ClearedOpErr;

#[derive(Debug, Clone)]
pub struct OpErrFmtNoArgs;

/// Result type for Python operations.
pub type PyResult = Result<PyObjectRef, PyError>;

/// Rust-side display text for [`PyError`].
///
/// A dedicated enum, not `Option<Wtf8Buf>`: a bare `Option` shares
/// `Some.__pos_0` with every other `Option` instantiation and unions
/// the string payload with unrelated types. `FromExcObject` is a unit
/// variant so `from_exc_object` does not construct a `Wtf8Buf`.
#[derive(Debug, Clone)]
pub enum DisplayMessage {
    FromExcObject,
    Text(Wtf8Buf),
}

/// `OperationError` (`pypy/interpreter/error.py`) as a GC object.
///
/// The class word is [`PYERROR_TYPE`]. `message` is a nullable rstr `STR`:
/// null is [`DisplayMessage::FromExcObject`]; otherwise the pointer holds the
/// WTF-8 bytes. The accessor [`PyError::display_message`] rebuilds the enum.
#[repr(C)]
pub struct PyErrorObject {
    pub ob_header: pyre_object::PyObject,
    pub kind: PyErrorKind,
    /// Exception class. `error.py OperationError.w_type`.
    ///
    /// Null only before the exception classes are registered (bootstrap);
    /// [`kind`](Self::kind) is then the fallback tag
    /// [`PyError::matches_stop_iteration`] still reads.
    pub w_type: pyre_object::PyObjectRef,
    /// Nullable rstr `STR`. Null means the display text comes from
    /// `exc_object` ([`DisplayMessage::FromExcObject`]). Otherwise the
    /// pointer holds the WTF-8 bytes. The object does not own a `Wtf8Buf`.
    pub message: pyre_object::gc_hook::GCREF,
    /// `error.py OperationError._w_value`: the exception value, a message
    /// object, or null until [`PyError::get_w_value`] /
    /// [`PyError::normalize_exception`] instantiate `w_type(w_value)` and
    /// cache the instance here.
    pub exc_object: pyre_object::PyObjectRef,
    /// `pypy/interpreter/pyopcode.py handle_operation_error(..., attach_tb=True)`
    /// parity.  RERAISE opcode (`pyopcode.py RERAISE`)
    /// surfaces the operror as `RaiseWithExplicitTraceback` which
    /// PyPy's bytecode dispatcher routes through
    /// `handle_operation_error(attach_tb=False)` to avoid stamping a
    /// spurious traceback frame on the cleanup path.  Pyre routes the
    /// same intent through this field: the `reraise` opcode
    /// (`eval.rs::reraise`) clears `attach_tb` so the
    /// re-fired `handle_exception` skips
    /// `record_application_traceback`.  Default `true` for any
    /// normally-constructed PyError; reraise flips it off.
    pub attach_tb: bool,
    /// `pypy/interpreter/error.py record_context` parity — the
    /// once-only latch for implicit `__context__` chaining:
    ///
    /// ```python
    /// def record_context(self, space, ec):
    ///     if self._context_recorded:
    ///         return
    ///     last = ec.sys_exc_info()
    ///     try:
    ///         if last is not None:
    ///             self.chain_exceptions(space, last)
    ///     finally:
    ///         self._context_recorded = True
    /// ```
    ///
    /// The latch is load-bearing whenever the recorded answer was "no
    /// context": a null `__context__` cannot distinguish an error that
    /// recorded none from one not yet recorded, so without it an outer
    /// frame handling something of its own would hand that to an error
    /// merely unwinding through.  Set on the first dispatch regardless of
    /// whether an active exception was found, mirroring the `finally`.
    pub context_recorded: bool,
    /// `pypy/interpreter/pyopcode.py handle_operation_error(..., reraise_lasti=-1)`
    /// parity.  RERAISE N reads the original raise-site lasti from the
    /// value stack and carries it through `RaiseWithExplicitTraceback`
    /// so the next exception-table dispatch can push the original
    /// raise-site offset (not the RERAISE instruction itself) as the
    /// `lasti` value, and so the no-handler propagation path can
    /// restore `last_instr` for correct `f_lineno`.  `-1` means "no
    /// reraise lasti carried" (default for primary raises).
    pub reraise_lasti: i32,
    /// The `name` context attribute for a freshly raised NameError or
    /// AttributeError (Python 3.10+): the undefined name (NameError) or
    /// the failed attribute name (AttributeError).  Exception
    /// materialisation is deferred to `to_exc_object`, so the raise site
    /// records the context here (`PY_NULL` = unset) and `to_exc_object`
    /// stamps it onto the built instance — the lazy equivalent of
    /// setting `exc.name` right after the raise
    /// (`_PyEval_FormatExcCheckArg` / `set_attribute_error_context`).
    pub w_name_context: pyre_object::PyObjectRef,
    /// The `obj` context attribute for a freshly raised AttributeError
    /// (Python 3.10+): the object whose attribute lookup failed.
    /// Carried and applied alongside `w_name_context`; `PY_NULL` = unset.
    pub w_obj_context: pyre_object::PyObjectRef,
}

/// RPython `Exception` (`HOST_ENV` / `standardexceptions`). Distinct from
/// the Python `Exception` vtable (`interp_exceptions::EXC_EXCEPTION_TYPE`).
/// [`PYERROR_TYPE`] (`OperationError`) subclasses this so catch-all
/// `Exception` and typed `OperationError` both match, and `OverflowError`
/// does not.
pub static HOST_EXCEPTION_TYPE: pyre_object::PyType =
    pyre_object::pyobject::new_pytype("Exception");

/// Class word of [`PyErrorObject`]. Subclasses [`HOST_EXCEPTION_TYPE`].
pub static PYERROR_TYPE: pyre_object::PyType = pyre_object::pyobject::new_pytype("OperationError");

/// Runtime GC tid. `build_gc` stamps this after the module classes so no
/// earlier id moves.
pub static PYERROR_GC_TYPE_ID_CELL: pyre_object::lltype::TypeIdCell =
    pyre_object::lltype::TypeIdCell::auto();

pub const PYERROR_OBJECT_SIZE: usize = std::mem::size_of::<PyErrorObject>();
pub const PYERROR_W_TYPE_OFFSET: usize = std::mem::offset_of!(PyErrorObject, w_type);
pub const PYERROR_MESSAGE_OFFSET: usize = std::mem::offset_of!(PyErrorObject, message);
pub const PYERROR_EXC_OBJECT_OFFSET: usize = std::mem::offset_of!(PyErrorObject, exc_object);
pub const PYERROR_W_NAME_CONTEXT_OFFSET: usize =
    std::mem::offset_of!(PyErrorObject, w_name_context);
pub const PYERROR_W_OBJ_CONTEXT_OFFSET: usize = std::mem::offset_of!(PyErrorObject, w_obj_context);

/// GC type id of [`HOST_EXCEPTION_TYPE`]: one past the module classes.
pub fn host_exception_gc_type_id() -> u32 {
    crate::MODULE_FIRST_TYPE_ID + crate::module_gc_types().len() as u32
}

/// GC type id of [`PyErrorObject`]: one past [`HOST_EXCEPTION_TYPE`].
///
/// `build_gc` registers this id and stores it in [`PYERROR_GC_TYPE_ID_CELL`].
/// Allocation and the subclass-range tables both use this expression, so a
/// raise before the cell is stamped still names the id the driver will
/// register.
pub fn pyerror_gc_type_id() -> u32 {
    host_exception_gc_type_id() + 1
}

impl pyre_object::lltype::GcType for PyErrorObject {
    #[inline]
    fn type_id() -> u32 {
        let id = PYERROR_GC_TYPE_ID_CELL.get();
        if id == pyre_object::lltype::TypeIdCell::UNASSIGNED {
            pyerror_gc_type_id()
        } else {
            id
        }
    }
    const SIZE: usize = PYERROR_OBJECT_SIZE;
}

/// One-word handle to [`PyErrorObject`]. `Clone` aliases the pointer, which is
/// `OperationError` semantics.
#[repr(transparent)]
pub struct PyError(pyre_object::PyObjectRef);

impl Clone for PyError {
    fn clone(&self) -> Self {
        PyError(self.0)
    }
}

impl std::ops::Deref for PyError {
    type Target = PyErrorObject;
    fn deref(&self) -> &PyErrorObject {
        unsafe { &*(self.0 as *mut PyErrorObject) }
    }
}

impl std::fmt::Debug for PyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyError")
            .field("kind", &self.kind)
            .field("w_type", &self.w_type)
            .field("message", &self.display_message())
            .finish()
    }
}

/// The STR allocation and the byte copy are host plumbing. Callers keep
/// the pointer result; the body is not a flow graph.
#[majit_macros::dont_look_inside]
fn message_str_from_wtf8(text: &rustpython_wtf8::Wtf8) -> pyre_object::gc_hook::GCREF {
    use pyre_object::lowlevel_string::{
        LOWLEVEL_STR_BASE_SIZE, LOWLEVEL_STRING_CHARS_OFFSET, bh_alloc_lowlevel_string,
    };
    let bytes = text.as_bytes();
    let raw = bh_alloc_lowlevel_string(bytes.len(), LOWLEVEL_STR_BASE_SIZE, 1);
    if raw == 0 {
        return std::ptr::null_mut();
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            (raw as *mut u8).add(LOWLEVEL_STRING_CHARS_OFFSET),
            bytes.len(),
        );
    }
    raw as pyre_object::gc_hook::GCREF
}

/// Inverse of [`message_str_from_wtf8`]. `Wtf8::from_bytes` is not a
/// translated operation; the bytes were written from a `Wtf8`.
#[majit_macros::dont_look_inside]
fn wtf8_from_message_str(ptr: pyre_object::gc_hook::GCREF) -> rustpython_wtf8::Wtf8Buf {
    use pyre_object::lowlevel_string::{LOWLEVEL_STRING_CHARS_OFFSET, bh_lowlevel_string_len};
    if ptr.is_null() {
        return rustpython_wtf8::Wtf8Buf::new();
    }
    let len = bh_lowlevel_string_len(ptr as i64);
    let bytes = unsafe {
        std::slice::from_raw_parts((ptr as *const u8).add(LOWLEVEL_STRING_CHARS_OFFSET), len)
    };
    rustpython_wtf8::Wtf8::from_bytes(bytes)
        .expect("message STR bytes were written from a Wtf8")
        .to_owned()
}

/// Allocate one [`PyErrorObject`] on the managed heap.
///
/// `malloc_typed_managed` (`gct_fv_gc_malloc` / `framework.py init_gc_object`).
/// A text message is a second allocation, so the handle is pinned across it
/// and reloaded after: `shadowstack.py expand_pop_roots` writes the forwarded
/// pointer back into the local.
#[majit_macros::dont_look_inside]
fn make_pyerror(
    kind: PyErrorKind,
    message: DisplayMessage,
    exc_object: pyre_object::PyObjectRef,
    attach_tb: bool,
    context_recorded: bool,
    reraise_lasti: i32,
    w_name_context: pyre_object::PyObjectRef,
    w_obj_context: pyre_object::PyObjectRef,
    w_type: pyre_object::PyObjectRef,
) -> PyError {
    // Resolve the class from `kind` when the caller did not supply one.
    // Before exception-class initialisation the lookup is null and `kind`
    // remains the bootstrap fallback (`error.py OperationError.w_type`
    // is otherwise always set).
    let w_type = if w_type.is_null() {
        pyre_object::interp_exceptions::lookup_exc_class_for_kind(kind_to_exc_kind(kind))
    } else {
        w_type
    };
    let obj = pyre_object::lltype::malloc_typed_managed(PyErrorObject {
        ob_header: pyre_object::PyObject {
            ob_type: &PYERROR_TYPE,
            w_class: pyre_object::pyobject::get_instantiate(&PYERROR_TYPE),
        },
        kind,
        w_type,
        message: std::ptr::null_mut(),
        exc_object,
        attach_tb,
        context_recorded,
        reraise_lasti,
        w_name_context,
        w_obj_context,
    });
    let mut err = PyError(obj as pyre_object::PyObjectRef);
    if let DisplayMessage::Text(text) = message {
        let roots = pyre_object::gc_roots::push_roots();
        let slot = err.pin(&roots);
        let message_str = message_str_from_wtf8(&text);
        err.reload(&roots, slot);
        err.set_message_ptr(message_str);
    }
    err
}

impl PyError {
    /// The display text. Null storage is [`DisplayMessage::FromExcObject`].
    ///
    /// `Wtf8::from_bytes` is not a translated operation. Callers that only
    /// need the enum keep this residual.
    #[majit_macros::dont_look_inside]
    pub fn display_message(&self) -> DisplayMessage {
        if self.message.is_null() {
            DisplayMessage::FromExcObject
        } else {
            DisplayMessage::Text(wtf8_from_message_str(self.message))
        }
    }

    fn write_barrier(&self) {
        pyre_object::gc_hook::try_gc_write_barrier(self.0 as pyre_object::gc_hook::GCREF);
    }

    fn raw(&self) -> *mut PyErrorObject {
        self.0 as *mut PyErrorObject
    }

    pub fn set_kind(&mut self, kind: PyErrorKind) {
        unsafe { (*self.raw()).kind = kind };
    }

    pub fn set_w_type(&mut self, w_type: pyre_object::PyObjectRef) {
        unsafe { (*self.raw()).w_type = w_type };
        self.write_barrier();
    }

    pub fn set_message_ptr(&mut self, message: pyre_object::gc_hook::GCREF) {
        unsafe { (*self.raw()).message = message };
        self.write_barrier();
    }

    pub fn set_display_message(&mut self, message: DisplayMessage) {
        let ptr = match message {
            DisplayMessage::FromExcObject => std::ptr::null_mut(),
            DisplayMessage::Text(text) => {
                let roots = pyre_object::gc_roots::push_roots();
                let slot = self.pin(&roots);
                let ptr = message_str_from_wtf8(&text);
                self.reload(&roots, slot);
                ptr
            }
        };
        self.set_message_ptr(ptr);
    }

    pub fn set_exc_object(&mut self, exc_object: pyre_object::PyObjectRef) {
        unsafe { (*self.raw()).exc_object = exc_object };
        self.write_barrier();
    }

    pub fn set_attach_tb(&mut self, attach_tb: bool) {
        unsafe { (*self.raw()).attach_tb = attach_tb };
    }

    pub fn set_context_recorded(&mut self, context_recorded: bool) {
        unsafe { (*self.raw()).context_recorded = context_recorded };
    }

    pub fn set_reraise_lasti(&mut self, reraise_lasti: i32) {
        unsafe { (*self.raw()).reraise_lasti = reraise_lasti };
    }

    pub fn set_w_name_context(&mut self, w_name_context: pyre_object::PyObjectRef) {
        unsafe { (*self.raw()).w_name_context = w_name_context };
        self.write_barrier();
    }

    pub fn set_w_obj_context(&mut self, w_obj_context: pyre_object::PyObjectRef) {
        unsafe { (*self.raw()).w_obj_context = w_obj_context };
        self.write_barrier();
    }

    pub fn as_raw(&self) -> *mut PyErrorObject {
        self.raw()
    }

    /// Rebuild a handle from a `pin_root` word.
    pub(crate) fn from_raw(raw: pyre_object::PyObjectRef) -> Self {
        PyError(raw)
    }

    /// Pin this handle on `roots` and return that slot.
    ///
    /// `pin_root` is a safepoint, so the word written back is the one it
    /// returns. A later collecting call forwards the slot only;
    /// [`Self::reload`] stores it, as `shadowstack.py expand_pop_roots`
    /// writes every live variable after `pop_roots`.
    pub fn pin(&mut self, roots: &pyre_object::gc_roots::RootScope) -> usize {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        self.0 = roots.pin_root(self.0);
        slot
    }

    /// Write the forwarded handle out of `slot`.
    ///
    /// `shadowstack.py gc_restore_root` / `expand_pop_roots`: the local read
    /// after the call is the slot, not the word from before the call.
    pub fn reload(&mut self, roots: &pyre_object::gc_roots::RootScope, slot: usize) {
        self.0 = roots.get(slot);
    }

    /// `error.py OperationError.w_type`, resolving from [`kind`] when the
    /// class was not stored (bootstrap, before exception-class registration).
    fn resolved_w_type(&self) -> pyre_object::PyObjectRef {
        if !self.w_type.is_null() {
            return self.w_type;
        }
        pyre_object::interp_exceptions::lookup_exc_class_for_kind(kind_to_exc_kind(self.kind))
    }

    /// True when `_w_value` is already a `BaseException` instance.
    fn has_exception_instance(&self) -> bool {
        !self.exc_object.is_null() && unsafe { pyre_object::is_exception(self.exc_object) }
    }
}

/// The one interpreter-level exception carrier, corresponding to
/// `pypy.interpreter.error.OperationError`.
///
/// `PyError` remains the Rust-facing compatibility name used by existing
/// `Result<T, PyError>` signatures and by the source translator; both names
/// denote this same value and therefore cannot diverge in type, value,
/// traceback, or context state.
pub type OperationError = PyError;

/// Whether `co_filename` names one of the modules
/// [`PyError::remove_traceback_module_frames`] was asked to drop.
///
/// A name matches outright, as upstream's `co_filename not in module_names`
/// does, and also matches as a trailing path component: upstream's callers
/// name frozen modules, whose `co_filename` is the fixed `<frozen ...>`
/// pseudo-name, while the same module read from disk carries an
/// installation-dependent absolute path that no fixed string can spell.
fn traceback_frame_names_module(co_filename: &str, module_names: &[&str]) -> bool {
    let normalized = co_filename.replace('\\', "/");
    module_names.iter().any(|name| {
        normalized == *name
            || (normalized.len() > name.len()
                && normalized.ends_with(name)
                && normalized.as_bytes()[normalized.len() - name.len() - 1] == b'/')
    })
}

/// [`PyError::to_exc_object`] behind the residual-call ABI:
/// one pointer in, one pointer out, and a body the codewriter does not
/// look inside.
///
/// The lowered raise path emits this spelling instead of the method
/// (`front/result_exc.rs`'s callee rule), so materialising the exception
/// keeps its shadow-stack roots, `w_exception_new_empty_impl`, and the
/// WTF-8 and allocation machinery under it out of the caller's JitCode.
/// A jitcode that merely *can* raise otherwise carries all of it, and one
/// un-lowered call anywhere in that body refuses the whole descent.
///
/// The pointer spellings are load-bearing: the call-target trampoline
/// reads word-sized arguments and cannot see through a reference or the
/// `PyObjectRef` alias.
///
/// # Safety
/// `err` must point to a live `PyError` no other borrow aliases.
#[majit_macros::dont_look_inside]
pub unsafe fn pyerror_to_exc_object(err: *mut PyError) -> *mut pyre_object::PyObject {
    unsafe { (*err).to_exc_object() }
}

/// [`PyError::type_error`] and [`PyError::to_exc_object`] fused into one
/// residual call, so a raise site carries neither body.
///
/// The two-call form leaves `PyError::new` in the caller's JitCode as a
/// transparent constructor with no host symbol, and every descent that
/// reaches it is refused. Fusing them removes the constructor from the
/// caller entirely: the front rule
/// (`front/result_exc.rs::fuse_kind_ctor_raise`) rewrites the
/// `PyError::type_error` call to this one and drops the successor's
/// `pyerror_to_exc_object`.
///
/// Message word of a fused `PyError::<kind>(msg)` raise
/// (`front/result_exc.rs` `fuse_kind_ctor_raise`).
///
/// The front rule fires only where the message resolves to a string literal,
/// and a literal's one-word `r` constant is `Ptr(STR)`
/// (`rstr.py StringRepr.convert_const`) -- the payload `oefmt` takes before
/// `space.newtext(msg)`, never a `W_UnicodeObject`.
fn fused_raise_message(msg: *mut pyre_object::PyObject) -> Wtf8Buf {
    unsafe {
        debug_assert!(
            !msg.is_null() && !pyre_object::is_str(msg),
            "fused raise message must be a STR payload, not a W_UnicodeObject"
        );
        pyre_object::unicodeobject::utf8_payload_wtf8(
            msg as *mut pyre_object::unicodeobject::UnicodeValueStorage,
        )
        .to_owned()
    }
}

/// `w_msg` is the message word, not a `&str`: `&str` is a two-word aggregate
/// with no one-word residual-call ABI -- the reason `stack_underflow_error`
/// stays unpublished (`jit_fnaddr.rs`).
///
/// # Safety
/// `w_msg` is a live rstr `STR` payload.
#[majit_macros::dont_look_inside]
pub unsafe fn pyerror_type_error_to_exc_object(
    w_msg: *mut pyre_object::PyObject,
) -> *mut pyre_object::PyObject {
    PyError::type_error(fused_raise_message(w_msg)).to_exc_object()
}

/// The zero-division twin of [`pyerror_type_error_to_exc_object`].
///
/// `int_floordiv` and `int_mod` both raise `PyError::zero_division` with the
/// literal `ZERO_DIVISION_MSG`.  Fusing that constructor with exception
/// materialisation keeps the non-RPython Rust `PyError` aggregate out of the
/// generated operator JitCode, so the orthodox descent carries the same
/// `W_BaseException` value that PyPy's `OperationError` path exposes.
///
/// # Safety
/// `w_msg` is a live rstr `STR` payload.
#[majit_macros::dont_look_inside]
pub unsafe fn pyerror_zero_division_to_exc_object(
    w_msg: *mut pyre_object::PyObject,
) -> *mut pyre_object::PyObject {
    PyError::zero_division(fused_raise_message(w_msg)).to_exc_object()
}

/// Literal-message `ValueError` twin of [`pyerror_zero_division_to_exc_object`].
///
/// `long_lshift` / `long_rshift` (`descr_lshift` / `descr_rshift`) raise
/// `PyError::value_error("negative shift count")` before `rbigint.lshift` /
/// `rbigint.rshift`. Fusing that constructor keeps the `PyError` aggregate,
/// whose `Wtf8Buf` is not a one-word `STR`, out of the operator JitCode.
///
/// # Safety
/// `w_msg` is a live rstr `STR` payload.
#[majit_macros::dont_look_inside]
pub unsafe fn pyerror_value_error_to_exc_object(
    w_msg: *mut pyre_object::PyObject,
) -> *mut pyre_object::PyObject {
    PyError::value_error(fused_raise_message(w_msg)).to_exc_object()
}

/// Literal-message `IndexError` twin of [`pyerror_value_error_to_exc_object`].
///
/// `getitem_str` (`descr_getitem` after `getindex_w`) raises
/// `PyError::index_error("string index out of range")` when `_getitem_result`
/// finds no code point. Fusing that constructor keeps the `PyError` aggregate,
/// whose `Wtf8Buf` is not a one-word `STR`, out of the subscript JitCode.
///
/// # Safety
/// `w_msg` is a live rstr `STR` payload.
#[majit_macros::dont_look_inside]
pub unsafe fn pyerror_index_error_to_exc_object(
    w_msg: *mut pyre_object::PyObject,
) -> *mut pyre_object::PyObject {
    PyError::index_error(fused_raise_message(w_msg)).to_exc_object()
}

impl PyError {
    /// Visit this handle by address so a moved [`PyErrorObject`] is relocated
    /// in place. The collector traces `w_type`, `message`, `exc_object`,
    /// `w_name_context` and `w_obj_context` through the registered offsets.
    /// `walk_raw_exception_roots` stays: those offsets do not cover the
    /// off-GC `malloc_typed` children of the W exception.
    pub fn walk_gc_refs(&mut self, visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
        if self.0.is_null() {
            return;
        }
        unsafe {
            visitor(&mut *(&mut self.0 as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef));
        }
        // `exc_object` is read through the handle the visitor just forwarded.
        let exc = self.exc_object;
        if !exc.is_null() && unsafe { pyre_object::is_exception(exc) } {
            unsafe { crate::eval::walk_raw_exception_roots(exc, visitor) };
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyErrorKind {
    TypeError,
    ValueError,
    ZeroDivisionError,
    NameError,
    UnboundLocalError,
    IndexError,
    KeyError,
    AttributeError,
    RuntimeError,
    StopIteration,
    StopAsyncIteration,
    OverflowError,
    ArithmeticError,
    ImportError,
    /// Subclass of ImportError raised when a module cannot be found.
    ModuleNotFoundError,
    NotImplementedError,
    AssertionError,
    /// Raised by `_weakref` when a proxy is dereferenced after the
    /// referent has been collected.
    /// pypy/module/_weakref/interp__weakref.py
    /// `oefmt(space.w_ReferenceError, "weakly referenced object no longer exists")`.
    ReferenceError,
    /// BaseException subclass — raised inside generators by gen.close().
    /// Not a subclass of Exception, so `except Exception` does not catch it.
    GeneratorExit,
    RecursionError,
    /// Internal parity marker for pypy/interpreter/pycode.py BytecodeCorruption
    /// `BytecodeCorruption`. This should not be raised as a user-level
    /// Python exception; it signals malformed bytecode in the interpreter.
    BytecodeCorruption,
    /// Base class for all operating-system errors.
    /// pypy/module/exceptions/interp_exceptions.py W_OSError.
    OSError,
    /// Subclass of OSError raised when a file or directory is not found.
    FileNotFoundError,
    /// pypy/interpreter/baseobjspace.py DescrMismatch.
    ///
    /// ```python
    /// class DescrMismatch(Exception):
    ///     pass
    /// ```
    ///
    /// Internal control-flow exception raised by `space.descr_self_interp_w`
    /// when a descriptor's typecheck wrapper sees an instance of the wrong
    /// class. Caught by `GetSetProperty.descr_property_get/set/del` which
    /// then re-raises a user-visible TypeError via `descr_call_mismatch`.
    DescrMismatch,
    /// Raised by sys.exit(). Not a subclass of Exception.
    SystemExit,
    /// rpython/jit/metainterp/compile.py `memory_error = MemoryError()`
    /// — raised by `PropagateExceptionDescr.handle_fail` when a JIT
    /// malloc helper returns NULL (true OOM).  Not raised by user code
    /// in pyre; the runtime allocator never returns NULL except on
    /// genuine system OOM.
    MemoryError,
    /// `pypy/module/exceptions/interp_exceptions.py W_SystemError` —
    /// PyPy uses this for interpreter-internal invariant violations
    /// (e.g. `chain_exceptions` rejecting non-BaseException context
    /// at `error.py:429`).
    SystemError,
    /// Internal JIT trace-abort signal. This is not a Python exception
    /// class and must be intercepted before user-visible error handling.
    TraceAbort,
    /// `pypy/module/exceptions/interp_exceptions.py W_LookupError`
    /// — intermediate parent of IndexError / KeyError.  Distinct
    /// PyErrorKind variant so `from_exc_object(LookupError)` /
    /// `to_exc_object()` / `render_exception` round-trip preserves
    /// the exact class rather than collapsing it onto IndexError or
    /// Exception.
    LookupError,
    /// `pypy/module/exceptions/interp_exceptions.py W_UnicodeError`
    /// — intermediate parent of UnicodeDecodeError / UnicodeEncodeError,
    /// itself a subclass of ValueError.
    UnicodeError,
    UnicodeDecodeError,
    UnicodeEncodeError,
    /// `pypy/module/exceptions/interp_exceptions.py W_UnicodeTranslateError
    /// W_UnicodeTranslateError` — subclass of UnicodeError.
    /// Identity-only port: dedicated PyErrorKind / ExcKind / PyType
    /// so render_exception preserves the class.  W_UnicodeTranslateError's
    /// 5-arg `(object, start, end, reason)` `__init__` and custom
    /// `__str__` are TODO.
    UnicodeTranslateError,
    /// `pypy/module/exceptions/interp_exceptions.py W_SyntaxError` —
    /// raised by `compile`/`exec`/`eval`/`ast.parse` on malformed source.
    /// Its location and message fields live in the flattened
    /// `W_BaseException` layout and mirror `W_SyntaxError`'s `w_*` slots.
    SyntaxError,
    /// Raised when a buffer-related operation cannot proceed, e.g.
    /// `PyByteArray_Resize` resizing a `bytearray` whose storage backs a
    /// live `memoryview` ("Existing exports of data: object cannot be
    /// re-sized").  Direct subclass of Exception.
    BufferError,
    /// Direct Exception subclass raised when a stream ends before a complete
    /// value is available.
    EOFError,
}

impl PyError {
    pub fn new(kind: PyErrorKind, message: impl Into<Wtf8Buf>) -> Self {
        make_pyerror(
            kind,
            DisplayMessage::Text(message.into()),
            std::ptr::null_mut(),
            true,
            false,
            -1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    }

    /// `error.py OperationError(w_type, w_value)`.
    ///
    /// `w_value` is the message object, `space.w_None`, or null. The
    /// exception instance is built later by [`Self::normalize_exception`] /
    /// [`Self::get_w_value`], which cache it in `exc_object` (`_w_value`).
    pub fn from_type_and_value(
        w_type: pyre_object::PyObjectRef,
        w_value: pyre_object::PyObjectRef,
    ) -> Self {
        let kind = pyre_object::interp_exceptions::kind_of_canonical_exc_class(w_type)
            .map(Self::kind_from_exc)
            .unwrap_or(PyErrorKind::RuntimeError);
        let w_value = if w_value.is_null() || unsafe { pyre_object::is_none(w_value) } {
            pyre_object::PY_NULL
        } else {
            w_value
        };
        // The class and the value outlive the carrier malloc only as roots:
        // a collection rewrites the slot, not this frame's copy of the
        // pointer. `w_None` is dropped above so `_w_value is None` matches
        // `OperationError.setup(w_type)` with no value.
        let _roots = pyre_object::gc_roots::push_roots();
        let type_slot = pyre_object::gc_roots::shadow_stack_len();
        if !w_type.is_null() {
            let _ = pyre_object::gc_roots::pin_root(w_type);
        }
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        if !w_value.is_null() {
            let _ = pyre_object::gc_roots::pin_root(w_value);
        }
        let w_type = if w_type.is_null() {
            w_type
        } else {
            pyre_object::gc_roots::shadow_stack_get(type_slot)
        };
        let w_value = if w_value.is_null() {
            w_value
        } else {
            pyre_object::gc_roots::shadow_stack_get(value_slot)
        };
        make_pyerror(
            kind,
            DisplayMessage::FromExcObject,
            w_value,
            true,
            false,
            -1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            w_type,
        )
    }

    /// Bool view of [`Self::display_message`]. The `Wtf8Buf` match stays
    /// inside the residual helper.
    #[majit_macros::dont_look_inside]
    fn has_display_message(&self) -> bool {
        matches!(self.display_message(), DisplayMessage::Text(m) if !m.is_empty())
    }

    /// Owned copy of the display text. `Wtf8Buf::new` is host plumbing.
    #[majit_macros::dont_look_inside]
    fn display_message_buf(&self) -> Wtf8Buf {
        match self.display_message() {
            DisplayMessage::Text(m) => m,
            DisplayMessage::FromExcObject => Wtf8Buf::new(),
        }
    }

    /// `pypy/interpreter/pyopcode.py FOR_ITER` tests iterator exhaustion with
    /// a Python-level MRO match. A flat `PyErrorKind` tag cannot express
    /// multiple inheritance, so only exact-tagged errors use the free fast
    /// path. An internally built error has no cached exception object and can
    /// only name a builtin, making its tag authoritative without materialising
    /// an object. The slow-path class cache is populated only after registry
    /// lookup succeeds, so a call before exception-class initialisation
    /// returns false without caching absence forever. A caller that keeps the
    /// handle uses [`Self::matches_stop_iteration_keep`]: `framework.py
    /// pop_roots` reloads every live variable, and this bool-only entry does
    /// not.
    pub fn matches_stop_iteration(&self) -> bool {
        PyError(self.0).matches_stop_kind()
    }

    /// The StopAsyncIteration twin of [`matches_stop_iteration`], with the same
    /// tag fast path and object slow path.
    ///
    /// [`matches_stop_iteration`]: Self::matches_stop_iteration
    pub fn matches_stop_async_iteration(&self) -> bool {
        PyError(self.0).matches_stop_async_kind()
    }

    /// Tag fast path, then a pinned slow path. The class match can collect,
    /// so the handle is on the shadow stack for that call and is not live
    /// again afterwards.
    fn matches_stop_kind(self) -> bool {
        if self.kind == PyErrorKind::StopIteration {
            return true;
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let handle = pyre_object::gc_roots::pin_root(self.0);
        let exc = unsafe { (*(handle as *mut PyErrorObject)).exc_object };
        if exc.is_null() || unsafe { !pyre_object::is_exception(exc) } {
            return false;
        }
        exception_object_matches_stop_iteration(exc)
    }

    fn matches_stop_async_kind(self) -> bool {
        if self.kind == PyErrorKind::StopAsyncIteration {
            return true;
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let handle = pyre_object::gc_roots::pin_root(self.0);
        let exc = unsafe { (*(handle as *mut PyErrorObject)).exc_object };
        if exc.is_null() || unsafe { !pyre_object::is_exception(exc) } {
            return false;
        }
        exception_object_matches_stop_async_iteration(exc)
    }

    /// Same answer as [`matches_stop_iteration`], plus the handle reloaded
    /// from the shadow stack so the caller can keep using it.
    pub fn matches_stop_iteration_keep(self) -> (bool, Self) {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let handle = pyre_object::gc_roots::pin_root(self.0);
        let tagged =
            unsafe { (*(handle as *mut PyErrorObject)).kind } == PyErrorKind::StopIteration;
        let exc = unsafe { (*(handle as *mut PyErrorObject)).exc_object };
        let stop = if tagged {
            true
        } else if exc.is_null() || unsafe { !pyre_object::is_exception(exc) } {
            false
        } else {
            exception_object_matches_stop_iteration(exc)
        };
        (stop, PyError(pyre_object::gc_roots::shadow_stack_get(slot)))
    }

    /// Same answer as [`matches_stop_async_iteration`], plus the handle
    /// reloaded from the shadow stack.
    ///
    /// [`matches_stop_async_iteration`]: Self::matches_stop_async_iteration
    pub fn matches_stop_async_iteration_keep(self) -> (bool, Self) {
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let handle = pyre_object::gc_roots::pin_root(self.0);
        let tagged =
            unsafe { (*(handle as *mut PyErrorObject)).kind } == PyErrorKind::StopAsyncIteration;
        let exc = unsafe { (*(handle as *mut PyErrorObject)).exc_object };
        let stop = if tagged {
            true
        } else if exc.is_null() || unsafe { !pyre_object::is_exception(exc) } {
            false
        } else {
            exception_object_matches_stop_async_iteration(exc)
        };
        (stop, PyError(pyre_object::gc_roots::shadow_stack_get(slot)))
    }

    pub fn type_error(msg: impl Into<Wtf8Buf>) -> Self {
        Self::new(PyErrorKind::TypeError, msg)
    }

    pub fn attribute_error(msg: impl Into<Wtf8Buf>) -> Self {
        Self::new(PyErrorKind::AttributeError, msg)
    }

    /// `set_attribute_error_context` parity — an AttributeError raised by
    /// a failed attribute lookup, carrying the failed attribute `name`
    /// and the `obj` it was looked up on so `e.name` / `e.obj` read back
    /// once the instance is materialised (Python 3.10+).
    pub fn attribute_error_with_context(
        msg: impl Into<Wtf8Buf>,
        w_obj: PyObjectRef,
        name: &str,
    ) -> Self {
        let mut err = Self::new(PyErrorKind::AttributeError, msg);
        // `w_obj` lives only in this Rust argument. `w_str_new_managed`
        // collects under stress, and the precise walker does not see the
        // local, so pin it and store the slot's word — a young receiver
        // that moved would otherwise be written into `w_obj_context` as
        // the vacated address.
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        if !w_obj.is_null() {
            let _ = pyre_object::gc_roots::pin_root(w_obj);
        }
        let err_slot = err.pin(&_roots);
        let w_name = pyre_object::w_str_new_managed(name);
        err.reload(&_roots, err_slot);
        err.set_w_name_context(w_name);
        err.set_w_obj_context(if w_obj.is_null() {
            w_obj
        } else {
            pyre_object::gc_roots::shadow_stack_get(obj_slot)
        });
        err
    }

    /// PyPy `error.py enrich_attribute_error` / CPython 3.14
    /// `_PyObject_SetAttributeErrorContext`: an `AttributeError` escaping a
    /// public attribute lookup gets the receiver and requested name when (and
    /// only when) neither slot was already supplied by an inner lookup or by
    /// the exception constructor.  In particular, an explicit `name=None` or
    /// `obj=None` is a populated slot in CPython and must not be overwritten.
    pub fn enrich_attribute_error(&mut self, w_obj: PyObjectRef, w_name: PyObjectRef) {
        if self.kind != PyErrorKind::AttributeError || w_obj.is_null() || w_name.is_null() {
            return;
        }
        if !self.has_exception_instance() {
            if self.w_name_context.is_null() && self.w_obj_context.is_null() {
                self.set_w_name_context(w_name);
                self.set_w_obj_context(w_obj);
            }
            return;
        }
        unsafe {
            let current_name =
                pyre_object::interp_exceptions::w_exception_get_name(self.exc_object);
            let current_obj =
                pyre_object::interp_exceptions::w_exception_get_attr_obj(self.exc_object);
            if current_name.is_null() && current_obj.is_null() {
                pyre_object::interp_exceptions::w_exception_set_name(self.exc_object, w_name);
                pyre_object::interp_exceptions::w_exception_set_attr_obj(self.exc_object, w_obj);
            }
        }
    }

    /// String-taking companion for pyre's optimized `getattr_str` path. PyPy
    /// already carries a wrapped `w_name`; pyre delays that allocation until
    /// an un-enriched AttributeError actually escapes. Root the receiver and
    /// error payload while constructing it because the precise collector does
    /// not scan this Rust `PyError` local.
    pub fn enrich_attribute_error_str(&mut self, w_obj: PyObjectRef, name: &str) {
        if self.kind != PyErrorKind::AttributeError || w_obj.is_null() {
            return;
        }
        let missing = if !self.has_exception_instance() {
            self.w_name_context.is_null() && self.w_obj_context.is_null()
        } else {
            unsafe {
                pyre_object::interp_exceptions::w_exception_get_name(self.exc_object).is_null()
                    && pyre_object::interp_exceptions::w_exception_get_attr_obj(self.exc_object)
                        .is_null()
            }
        };
        if !missing {
            return;
        }

        let _roots = pyre_object::gc_roots::push_roots();
        let handle_slot = self.pin(&_roots);
        let exc_slot = if !self.has_exception_instance() {
            None
        } else {
            let slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(self.exc_object);
            Some(slot)
        };
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_obj);
        let w_name = pyre_object::w_str_new_managed(name);
        self.reload(&_roots, handle_slot);
        if let Some(slot) = exc_slot {
            self.set_exc_object(pyre_object::gc_roots::shadow_stack_get(slot));
        }
        let w_obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        self.enrich_attribute_error(w_obj, w_name);
    }

    pub fn value_error(msg: impl Into<Wtf8Buf>) -> Self {
        Self::new(PyErrorKind::ValueError, msg)
    }

    /// Retag the materialised exception to a subclass that shares its
    /// layout, the way `os_error_family_new` picks the errno subclass:
    /// the `ExcKind` (and so the storage) stays put and only `w_class`
    /// moves.  Used for `IndentationError` / `TabError`, which are
    /// `SyntaxError` subclasses with no storage of their own.
    pub fn retag_exception_class(&mut self, class_name: &str) {
        let Some(w_target) = crate::builtins::lookup_exc_class(class_name) else {
            return;
        };
        // `set_w_type` write-barriers the handle (`llop.gc_writebarrier`).
        // Pin the handle and the class across it the way
        // `enrich_attribute_error_str` pins across `w_str_new_managed`.
        let _roots = pyre_object::gc_roots::push_roots();
        let handle_slot = self.pin(&_roots);
        let target_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = _roots.pin_root(w_target);
        self.set_w_type(_roots.get(target_slot));
        self.reload(&_roots, handle_slot);
        if !self.has_exception_instance() {
            return;
        }
        // Fresh instance: no recorded `w_class?` read can name it yet.
        let w_target = _roots.get(target_slot);
        unsafe {
            (*(self.exc_object as *mut pyre_object::PyObject)).w_class = w_target;
        }
    }

    pub fn syntax_error(msg: impl Into<Wtf8Buf>) -> Self {
        Self::new(PyErrorKind::SyntaxError, msg)
    }

    /// Build a compile-time SyntaxError carrying the parser location so
    /// `e.msg` / `e.filename` / `e.lineno` / `e.offset` / `e.text` /
    /// `e.end_lineno` / `e.end_offset` read back once the instance is
    /// materialised.  `args` becomes `(msg, (filename, lineno, offset,
    /// text, end_lineno, end_offset))`, the shape
    /// `interp_exceptions.py W_SyntaxError.descr_init` stores and
    /// the dedicated `W_SyntaxError` slots and `args_w` both receive. `text` is the
    /// offending source line (`None` → `text` reads back as `None`).
    #[allow(clippy::too_many_arguments)]
    pub fn syntax_error_located(
        msg: impl Into<Wtf8Buf>,
        filename: &rustpython_wtf8::Wtf8,
        lineno: i64,
        offset: i64,
        end_lineno: i64,
        end_offset: i64,
        text: Option<&str>,
    ) -> Self {
        let message = msg.into();
        // Root the fresh exception across the args/details allocations: `exc`
        // lives only in this Rust local while `w_str_new` / `w_int_new` /
        // `w_tuple_new` / `w_list_new` run, so a collection there could sweep
        // the unrooted exception before `w_exception_set_args` writes it.
        let _roots = pyre_object::gc_roots::push_roots();
        let exc = w_exception_new_wtf8(ExcKind::SyntaxError, &message);
        let exc_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(exc);
        // Build the `(filename, lineno, offset, text, end_lineno, end_offset)`
        // details tuple element by element. Each is pinned as it is created and
        // reloaded before the tuple is assembled: an int / str allocation for a
        // later element can trigger a minor collection that relocates an earlier
        // young element (the filename or text string, or an uncached line/col
        // int), leaving its raw local pointing at the old address.
        let filename_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_from_wtf8_managed(
            filename.to_wtf8_buf(),
        ));
        let lineno_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(lineno));
        let offset_slot = pyre_object::gc_roots::shadow_stack_len();
        // CPython 3.14 keeps the start offset numeric even for the tokenizer's
        // two sentinel values: 0 for an encoding-cookie range beginning
        // before the first character, and -1 for a location-less codec
        // failure.  Only absent *end* positions use `None` below.
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(offset));
        let text_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(match text {
            Some(t) => pyre_object::w_str_new_managed(t),
            None => pyre_object::w_none(),
        });
        let end_lineno_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(wrap_pos(end_lineno));
        let end_offset_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(wrap_pos(end_offset));
        let rooted_exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_syntax_filename(
                rooted_exc,
                pyre_object::gc_roots::shadow_stack_get(filename_slot),
            );
            pyre_object::interp_exceptions::w_exception_set_syntax_lineno(
                rooted_exc,
                pyre_object::gc_roots::shadow_stack_get(lineno_slot),
            );
            pyre_object::interp_exceptions::w_exception_set_syntax_offset(
                rooted_exc,
                pyre_object::gc_roots::shadow_stack_get(offset_slot),
            );
            pyre_object::interp_exceptions::w_exception_set_syntax_text(
                rooted_exc,
                pyre_object::gc_roots::shadow_stack_get(text_slot),
            );
            pyre_object::interp_exceptions::w_exception_set_syntax_end_lineno(
                rooted_exc,
                pyre_object::gc_roots::shadow_stack_get(end_lineno_slot),
            );
            pyre_object::interp_exceptions::w_exception_set_syntax_end_offset(
                rooted_exc,
                pyre_object::gc_roots::shadow_stack_get(end_offset_slot),
            );
        }
        let details = pyre_object::w_tuple_new(vec![
            pyre_object::gc_roots::shadow_stack_get(filename_slot),
            pyre_object::gc_roots::shadow_stack_get(lineno_slot),
            pyre_object::gc_roots::shadow_stack_get(offset_slot),
            pyre_object::gc_roots::shadow_stack_get(text_slot),
            pyre_object::gc_roots::shadow_stack_get(end_lineno_slot),
            pyre_object::gc_roots::shadow_stack_get(end_offset_slot),
        ]);
        // Root the details tuple across the message allocation below before it
        // becomes `args[1]`.
        let details_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(details);
        let w_msg = pyre_object::w_str_from_wtf8_managed(message.clone());
        let msg_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_msg);
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_syntax_msg(
                pyre_object::gc_roots::shadow_stack_get(exc_slot),
                pyre_object::gc_roots::shadow_stack_get(msg_slot),
            );
        }
        let args_list = pyre_object::interp_exceptions::w_exception_args_new(vec![
            pyre_object::gc_roots::shadow_stack_get(msg_slot),
            pyre_object::gc_roots::shadow_stack_get(details_slot),
        ]);
        let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
        unsafe { pyre_object::interp_exceptions::w_exception_set_args(exc, args_list) };
        // Leave the display message empty so `message_text` derives it from
        // `exc_object` via the SyntaxError `descr_str`, which appends the
        // `(filename, line N)` suffix.
        make_pyerror(
            PyErrorKind::SyntaxError,
            DisplayMessage::FromExcObject,
            exc,
            true,
            false,
            -1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    }

    /// Replace the filename in a located `SyntaxError` with an already-built
    /// Python string.
    ///
    /// RustPython's compiler currently accepts `&str` paths, while PyPy's
    /// `space.fsdecode_w()` can preserve surrogateescaped filesystem bytes.
    /// Callers may therefore compile with a lossy Rust path but restore the
    /// authoritative Python filename object in the exception details tuple.
    /// Rebuild the immutable tuple through shadow-stack slots so every child
    /// remains valid if either allocation triggers a moving collection.
    pub fn replace_syntax_error_filename(&mut self, w_filename: PyObjectRef) {
        if self.kind != PyErrorKind::SyntaxError
            || self.exc_object.is_null()
            || w_filename.is_null()
        {
            return;
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let self_slot = self.pin(&_roots);
        let base = pyre_object::gc_roots::pin_roots(&[self.exc_object, w_filename]);
        let storage = unsafe {
            pyre_object::interp_exceptions::w_exception_get_args_storage(
                pyre_object::gc_roots::shadow_stack_get(base),
            )
        };
        if storage.is_null() || unsafe { pyre_object::interp_exceptions::rlist_len(storage) } < 2 {
            return;
        }
        let w_msg = unsafe { pyre_object::interp_exceptions::rlist_getitem(storage, 0) };
        if w_msg.is_null() {
            return;
        }
        let details = unsafe { pyre_object::interp_exceptions::rlist_getitem(storage, 1) };
        if details.is_null() {
            return;
        }
        if unsafe { !pyre_object::is_tuple(details) || pyre_object::w_tuple_len(details) < 6 } {
            return;
        }
        let _ = pyre_object::gc_roots::pin_root(w_msg);
        for index in 1..6 {
            let Some(item) = (unsafe { pyre_object::w_tuple_getitem(details, index) }) else {
                return;
            };
            let _ = pyre_object::gc_roots::pin_root(item);
        }
        let rebuilt_details = pyre_object::w_tuple_new(vec![
            pyre_object::gc_roots::shadow_stack_get(base + 1),
            pyre_object::gc_roots::shadow_stack_get(base + 3),
            pyre_object::gc_roots::shadow_stack_get(base + 4),
            pyre_object::gc_roots::shadow_stack_get(base + 5),
            pyre_object::gc_roots::shadow_stack_get(base + 6),
            pyre_object::gc_roots::shadow_stack_get(base + 7),
        ]);
        let _ = pyre_object::gc_roots::pin_root(rebuilt_details);
        let rebuilt_args = pyre_object::interp_exceptions::w_exception_args_new(vec![
            pyre_object::gc_roots::shadow_stack_get(base + 2),
            pyre_object::gc_roots::shadow_stack_get(base + 8),
        ]);
        self.reload(&_roots, self_slot);
        self.set_exc_object(pyre_object::gc_roots::shadow_stack_get(base));
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_args(self.exc_object, rebuilt_args);
            pyre_object::interp_exceptions::w_exception_set_syntax_filename(
                self.exc_object,
                pyre_object::gc_roots::shadow_stack_get(base + 1),
            );
        };
    }

    pub fn zero_division(msg: impl Into<Wtf8Buf>) -> Self {
        Self::new(PyErrorKind::ZeroDivisionError, msg)
    }

    pub fn overflow_error(msg: impl Into<Wtf8Buf>) -> Self {
        Self::new(PyErrorKind::OverflowError, msg)
    }

    pub fn runtime_error(msg: impl Into<Wtf8Buf>) -> Self {
        Self::new(PyErrorKind::RuntimeError, msg)
    }

    pub fn not_implemented(msg: impl Into<Wtf8Buf>) -> Self {
        Self::new(PyErrorKind::NotImplementedError, msg)
    }

    pub fn system_error(msg: impl Into<Wtf8Buf>) -> Self {
        Self::new(PyErrorKind::SystemError, msg)
    }

    /// [`Self::oefmt_name_error`] for a name the caller holds as text, which
    /// is how the eval loop reports an undefined one
    /// (`_PyEval_FormatExcCheckArg` parity).
    pub fn oefmt_name_error_text(msg: impl Into<Wtf8Buf>, name: &str) -> Self {
        Self::oefmt_name_error(msg, pyre_object::w_str_new_managed(name))
    }

    /// `error.py oefmt_name_error` — a NameError carrying the undefined name so
    /// `e.name` reads back once the instance is materialised (Python 3.10+);
    /// `pyopcode.py _load_global_failed` passes `w_varname` the same way.
    pub fn oefmt_name_error(msg: impl Into<Wtf8Buf>, w_name: pyre_object::PyObjectRef) -> Self {
        let mut err = Self::new(PyErrorKind::NameError, msg);
        err.set_w_name_context(w_name);
        err
    }

    /// UnboundLocalError(NameError) for an unbound local.  The undefined name
    /// rides only in the message: `format_exc_check_arg` stamps the `name`
    /// slot when the raised class is exactly `NameError`, so an
    /// `UnboundLocalError` reaches Python with `name` still `None`.
    pub fn unbound_local_error(msg: impl Into<Wtf8Buf>) -> Self {
        Self::new(PyErrorKind::UnboundLocalError, msg)
    }

    /// A ModuleNotFoundError carrying the unresolved module `name` so
    /// `e.name` reads back once the instance is materialised — the import
    /// machinery raises `ModuleNotFoundError(msg, name=fullname)`.  The
    /// `name` rides the shared `w_n` slot (ImportError / NameError /
    /// AttributeError), stamped by `to_exc_object`.
    pub fn module_not_found_with_name(msg: impl Into<Wtf8Buf>, name: &str) -> Self {
        Self::module_not_found_with_name_obj(msg, pyre_object::w_str_new_managed(name))
    }

    /// `module_not_found_with_name` for a name with no `&str` spelling, so the
    /// caller supplies the name object itself.
    pub fn module_not_found_with_name_obj(
        msg: impl Into<Wtf8Buf>,
        w_name: pyre_object::PyObjectRef,
    ) -> Self {
        let mut err = Self::new(PyErrorKind::ModuleNotFoundError, msg);
        err.set_w_name_context(w_name);
        err
    }

    pub fn internal_trace_abort(reason: impl Into<Wtf8Buf>) -> Self {
        let mut err = Self::new(PyErrorKind::TraceAbort, reason);
        err.set_attach_tb(false);
        err
    }

    pub fn key_error(msg: impl Into<Wtf8Buf>) -> Self {
        Self::new(PyErrorKind::KeyError, msg)
    }

    /// `baseobjspace.py raise_key_error(w_key)` parity — builds
    /// `KeyError(w_key)` with the key itself (not a stringified copy)
    /// as args[0], so callers reading `e.args[0]` get back the missing
    /// key object.  The display message is the key's repr, matching
    /// what `KeyError.__str__` yields.
    pub fn key_error_with_key(key: PyObjectRef) -> Self {
        // Root the caller's key and the fresh exception across the repr,
        // exception, and args allocations below: they live only in Rust
        // locals, which the precise collector does not scan, so a collection
        // inside `py_repr` / `w_exception_new` / `w_list_new` could sweep the
        // key or the exception before `w_exception_set_args` stamps them.
        // The key is pinned before `py_repr` because that repr itself
        // allocates.
        let _roots = pyre_object::gc_roots::push_roots();
        let key_slot = pyre_object::gc_roots::shadow_stack_len();
        let key = pyre_object::gc_roots::pin_root(key);
        let message = if key.is_null() {
            Wtf8Buf::from_string("<null>".to_string())
        } else {
            unsafe { crate::display::py_repr_wtf8(key) }
                .unwrap_or_else(|_| Wtf8Buf::from_string("<unrepresentable>".to_string()))
        };
        let exc_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_exception_new_wtf8(ExcKind::KeyError, &message));
        // The exception is read back out of its slot at every use: the pin
        // keeps it alive without keeping it in place, and the allocations
        // below can relocate it.
        let exc = || pyre_object::gc_roots::shadow_stack_get(exc_slot);
        // Reload the key after the repr / exception allocations: the pin
        // keeps it alive, but a minor collection may have relocated the
        // young key, leaving the raw local pointing at the old address.
        let key = pyre_object::gc_roots::shadow_stack_get(key_slot);
        if !key.is_null() {
            let args_list = pyre_object::interp_exceptions::w_exception_args_new(vec![key]);
            unsafe { pyre_object::interp_exceptions::w_exception_set_args(exc(), args_list) };
        }
        make_pyerror(
            PyErrorKind::KeyError,
            DisplayMessage::Text(message),
            exc(),
            true,
            false,
            -1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    }

    pub fn index_error(msg: impl Into<Wtf8Buf>) -> Self {
        Self::new(PyErrorKind::IndexError, msg)
    }

    pub fn lookup_error(msg: impl Into<Wtf8Buf>) -> Self {
        Self::new(PyErrorKind::LookupError, msg)
    }

    pub fn os_error(msg: impl Into<Wtf8Buf>) -> Self {
        Self::new(PyErrorKind::OSError, msg)
    }

    /// Rust's OS-error `Display` is `"{strerror} (os error {errno})"`;
    /// recover the bare strerror the way `rposix.strerror` reports it.
    pub fn clean_strerror(errno: i32) -> String {
        // `from_raw_os_error` reads its argument as a Win32 code on Windows,
        // where these errnos are the POSIX ones `io_error_posix_errno`
        // translated to, so it would describe an unrelated system error (17 is
        // EEXIST but also ERROR_NOT_SAME_DEVICE).  The C runtime keeps the
        // errno-keyed table `strerror` reports from.
        #[cfg(all(windows, feature = "host_env"))]
        if let Some(msg) = rustpython_host_env::errno::strerror_string(errno) {
            return msg;
        }
        #[cfg(all(windows, not(feature = "host_env")))]
        {
            let msg = unsafe { libc::strerror(errno) };
            if !msg.is_null() {
                return unsafe { std::ffi::CStr::from_ptr(msg) }
                    .to_string_lossy()
                    .into_owned();
            }
        }
        // wasm32 has no error table behind `from_raw_os_error`, which answers
        // every code with the same placeholder.
        #[cfg(target_arch = "wasm32")]
        if let Some(msg) = crate::builtins::wasm_errno::strerror(errno) {
            return msg.to_string();
        }
        let full = std::io::Error::from_raw_os_error(errno).to_string();
        match full.rfind(" (os error ") {
            Some(idx) => full[..idx].to_string(),
            None => full,
        }
    }

    /// The `errno` and `strerror` a failed `std::io` call reports as, the pair
    /// `pymain_run_file_obj` names an unopenable script with.  The error's own
    /// `Display` gives the Win32 failure in the language the OS is localised
    /// to; `strerror` answers from the C runtime's table instead, so the text
    /// is the same on every host.
    pub fn io_errno_strerror(e: &std::io::Error) -> (i32, String) {
        let errno = crate::builtins::io_error_posix_errno(e, 0);
        (errno, Self::clean_strerror(errno))
    }

    /// `error.py:_wrap_oserror2` — build the OSError a failed syscall raises:
    /// `args = (errno, strerror)`, the errno-specific subclass (`ERRNO_MAP`),
    /// the clean strerror, and the `.errno` / `.strerror` / `.filename` slots.
    /// `w_filename` is the affected path, or `PY_NULL` for none.
    pub fn os_error_syscall(errno: i32, w_filename: PyObjectRef) -> Self {
        Self::os_error_syscall2(errno, w_filename, pyre_object::PY_NULL)
    }

    /// `os_error_syscall` for the two-path syscalls (`rename`, `replace`,
    /// `link`, `symlink`): `w_filename2` becomes the `.filename2` slot, which
    /// `str(e)` renders as the `" -> 'dest'"` suffix.
    pub fn os_error_syscall2(
        errno: i32,
        w_filename: PyObjectRef,
        w_filename2: PyObjectRef,
    ) -> Self {
        Self::os_error_from_parts(
            None,
            errno,
            None,
            Self::clean_strerror(errno),
            w_filename,
            w_filename2,
        )
    }

    /// `os_error_syscall2` for a failure reported by a Win32 API: the code is a
    /// Win32 error, not an errno.  It is kept in `.winerror`, `.strerror` is the
    /// system's own message for it, and `.errno` — with the subclass picked from
    /// it — comes from the errmap, so `str(e)` reads `[WinError 183] ...: 'path'`
    /// (`PyErr_SetExcFromWindowsErrWithFilenameObjects`).
    #[cfg(windows)]
    pub fn os_error_win32_syscall2(
        winerror: i32,
        w_filename: PyObjectRef,
        w_filename2: PyObjectRef,
    ) -> Self {
        Self::os_error_win32_from_parts(None, winerror, w_filename, w_filename2)
    }

    /// `PyErr_SetExcFromWindowsErr(exc, winerror)` for an `exc` the caller
    /// names, which keeps that class instead of the one the errno would pick.
    #[cfg(windows)]
    pub fn os_error_win32_with_class(class_name: &'static str, winerror: i32) -> Self {
        Self::os_error_win32_from_parts(
            Some(class_name),
            winerror,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        )
    }

    #[cfg(windows)]
    fn os_error_win32_from_parts(
        class_name: Option<&'static str>,
        winerror: i32,
        w_filename: PyObjectRef,
        w_filename2: PyObjectRef,
    ) -> Self {
        let errno = crate::builtins::winerror_to_errno(winerror as i64) as i32;
        let strerror = Self::win32_strerror(winerror);
        Self::os_error_from_parts(
            class_name,
            errno,
            Some(winerror),
            strerror,
            w_filename,
            w_filename2,
        )
    }

    /// The message a Win32 error code names, with the trailing newline and
    /// full stop `FormatMessageW` appends removed.  Reporting the bare code is
    /// the answer for a code the system cannot describe.
    #[cfg(windows)]
    pub fn win32_strerror(winerror: i32) -> String {
        #[cfg(feature = "host_env")]
        if let Some(message) =
            rustpython_host_env::windows::format_error_message(Some(winerror as u32))
        {
            return message
                .trim_end_matches(|c: char| c <= ' ' || c == '.')
                .to_string();
        }
        format!("Windows Error 0x{winerror:x}")
    }

    /// The body both syscall constructors share.  `winerror` is `None` off
    /// Windows, where no code but the errno exists.  `class_name` is an
    /// OSError subclass the caller raises explicitly; `None` picks the class
    /// from the errno, as constructing plain `OSError` does.
    fn os_error_from_parts(
        class_name: Option<&'static str>,
        errno: i32,
        winerror: Option<i32>,
        strerror: String,
        w_filename: PyObjectRef,
        w_filename2: PyObjectRef,
    ) -> Self {
        let subclass =
            class_name.or_else(|| crate::builtins::os_error_errno_subclass(errno as i64));
        // The dedicated FileNotFoundError kind carries its own str/repr; the
        // other errno subclasses share OSError's and differ only by class.
        let (kind, exc_kind) = if matches!(subclass, Some("FileNotFoundError")) {
            (PyErrorKind::FileNotFoundError, ExcKind::FileNotFoundError)
        } else {
            (PyErrorKind::OSError, ExcKind::OSError)
        };
        let message = match winerror {
            Some(code) => format!("[WinError {code}] {strerror}"),
            None => format!("[Errno {errno}] {strerror}"),
        };
        // Root the caller's filename and the fresh exception across the
        // exception and args allocations below: they live only in Rust locals,
        // which the precise collector does not scan, so a collection inside
        // `w_exception_new` / `w_list_new` could sweep them before the setters
        // stamp them onto `exc`.
        let _roots = pyre_object::gc_roots::push_roots();
        let pin = |obj: PyObjectRef| -> Option<usize> {
            if obj.is_null() {
                return None;
            }
            let slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(obj);
            Some(slot)
        };
        let filename_slot = pin(w_filename);
        let filename2_slot = pin(w_filename2);
        // The exception is read back out of its slot at every use below rather
        // than out of a Rust local: each setter is preceded by an allocation,
        // and the pin keeps the object alive without keeping it in place.
        let exc_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_exception_new(exc_kind, &message));
        let exc = || pyre_object::gc_roots::shadow_stack_get(exc_slot);
        // Retag to the errno subclass through `w_class`, like
        // `os_error_family_new`: the subclasses share OSError's layout, so
        // the ExcKind stays OSError and only the class differs.
        if let Some(w_target) = subclass.and_then(crate::builtins::lookup_exc_class) {
            // Fresh instance: no recorded `w_class?` read can name it yet.
            unsafe {
                (*(exc() as *mut pyre_object::PyObject)).w_class = w_target;
            }
        }
        // `PyErr_SetExcFromWindowsErrWithFilenameObjects` builds the exception
        // from `(errno, strerror, filename, winerror, filename2)`, and
        // `oserror_init` trims that back to the leading pair only once there is
        // a filename to carry the rest.  A Win32 failure naming no file keeps
        // all five, which is what lets `.winerror` survive the round trip
        // through `args` that a pickle or a re-raise makes.
        //
        // Each element is pinned as it is built and read back only once the
        // last one exists: they are young objects held in a Rust local, so
        // allocating the next one can relocate the ones already in hand.
        let mut arg_slots = Vec::with_capacity(5);
        {
            let mut push = |obj: PyObjectRef| {
                let slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(obj);
                arg_slots.push(slot);
            };
            push(pyre_object::w_int_new(errno as i64));
            push(pyre_object::w_str_new_managed(&strerror));
            if let Some(code) = winerror.filter(|_| filename_slot.is_none()) {
                push(pyre_object::w_none());
                push(pyre_object::w_int_new(code as i64));
                push(pyre_object::w_none());
            }
        }
        let args_list = pyre_object::interp_exceptions::w_exception_args_new(
            arg_slots
                .iter()
                .map(|&slot| pyre_object::gc_roots::shadow_stack_get(slot))
                .collect(),
        );
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_args(exc(), args_list);
            // Each value is built before the setter runs, so the exception the
            // setter is handed is the one that exists after that allocation
            // rather than the one that existed before it.
            let w_errno = pyre_object::w_int_new(errno as i64);
            pyre_object::interp_exceptions::w_exception_set_errno(exc(), w_errno);
            let w_strerror = pyre_object::w_str_new_managed(&strerror);
            pyre_object::interp_exceptions::w_exception_set_strerror(exc(), w_strerror);
            #[cfg(windows)]
            if let Some(code) = winerror {
                let w_winerror = pyre_object::w_int_new(code as i64);
                pyre_object::interp_exceptions::w_exception_set_winerror(exc(), w_winerror);
            }
            // Reload the paths after the args allocations: the pins keep them
            // alive, but a minor collection may have relocated the young
            // strings, leaving the raw locals pointing at the old addresses.
            if let Some(slot) = filename_slot {
                let w_filename = pyre_object::gc_roots::shadow_stack_get(slot);
                pyre_object::interp_exceptions::w_exception_set_filename(exc(), w_filename);
            }
            if let Some(slot) = filename2_slot {
                let w_filename2 = pyre_object::gc_roots::shadow_stack_get(slot);
                pyre_object::interp_exceptions::w_exception_set_filename2(exc(), w_filename2);
            }
        }
        // Leave the display message empty so `message_text` derives it from
        // `exc_object` via `W_OSError.descr_str`, which appends the
        // `: 'filename'` suffix; the bare "[Errno N] strerror" would bypass
        // it and the uncaught-traceback header would drop the filename.
        make_pyerror(
            kind,
            DisplayMessage::FromExcObject,
            exc(),
            true,
            false,
            -1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    }

    /// Raise the structured OSError for `errno` (no filename). The caller's
    /// platform message is superseded by the errno-derived strerror.
    pub fn os_error_with_errno(errno: i32, _msg: impl Into<Wtf8Buf>) -> Self {
        Self::os_error_syscall(errno, pyre_object::PY_NULL)
    }

    /// `PyErr_SetFromErrno(exc)` for an exception class outside the OSError
    /// family: `args` becomes the same `(errno, strerror)` pair, but with no
    /// `__str__` override to fold it into `"[Errno N] strerror"` the message
    /// is the args tuple's own repr — `OverflowError: (34, 'Result too
    /// large')`.  A libm `ERANGE` out of `float.__pow__` is reported this way;
    /// `floatobject.py:941` instead raises a plain `"float power"` message,
    /// which 3.14 does not produce.
    pub fn errno_pair(kind: PyErrorKind, exc_kind: ExcKind, errno: i32) -> Self {
        let strerror = Self::clean_strerror(errno);
        // Root the fresh exception across the args allocations: `exc` lives
        // only in this Rust local while `w_int_new` / `w_str_new` /
        // `w_list_new` run, so a collection there could sweep the unrooted
        // exception before `w_exception_set_args` writes through it.
        let _roots = pyre_object::gc_roots::push_roots();
        let exc_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_exception_new(exc_kind, &strerror));
        // Everything below is read back out of its slot: `w_list_new_object`
        // takes elements its caller has already pinned, and each pin keeps its
        // object alive without keeping it in place.
        let exc = || pyre_object::gc_roots::shadow_stack_get(exc_slot);
        let errno_slot = exc_slot + 1;
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(errno as i64));
        let strerror_slot = errno_slot + 1;
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(&strerror));
        let args_list = pyre_object::interp_exceptions::w_exception_args_new(vec![
            pyre_object::gc_roots::shadow_stack_get(errno_slot),
            pyre_object::gc_roots::shadow_stack_get(strerror_slot),
        ]);
        unsafe { pyre_object::interp_exceptions::w_exception_set_args(exc(), args_list) };
        // Leave the display message empty so `message_text` derives it
        // from `exc_object`, whose `descr_str` renders the two-element
        // `args` as a tuple repr rather than as a bare string.
        make_pyerror(
            kind,
            DisplayMessage::FromExcObject,
            exc(),
            true,
            false,
            -1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    }

    /// Raise an OSError carrying the C-level `(errno, strerror)` pair,
    /// matching `OSError.__init__`'s 2-argument form: `args` becomes
    /// `(errno, strerror)`, `str(e)` is `"[Errno N] strerror"`, and
    /// `e.errno` / `e.strerror` read back the two values.  Like
    /// `OSError(errno, strerror)`, every errno in CPython's/PyPy's errno map
    /// selects its concrete subclass (EAGAIN → BlockingIOError, etc.).
    pub fn os_error_errno_strerror(errno: i32, strerror: impl Into<Wtf8Buf>) -> Self {
        let strerror = strerror.into();
        let subclass = crate::builtins::os_error_errno_subclass(errno as i64);
        let kind = if matches!(subclass, Some("FileNotFoundError")) {
            PyErrorKind::FileNotFoundError
        } else {
            PyErrorKind::OSError
        };
        let exc_kind = if matches!(subclass, Some("FileNotFoundError")) {
            ExcKind::FileNotFoundError
        } else {
            ExcKind::OSError
        };
        // `strerror` reaches Python as `e.strerror` and as `args[1]`, so it is
        // WTF-8 and the assembled `[Errno N] ...` line has to be too.
        let mut message = Wtf8Buf::from_string(format!("[Errno {errno}] "));
        message.push_wtf8(&strerror);
        // Root the fresh exception across the args allocation below: `exc`
        // lives only in this Rust local while `w_int_new` / `w_str_new` /
        // `w_list_new` run, so a collection there could sweep the unrooted
        // exception before `w_exception_set_args` writes through it.
        let _roots = pyre_object::gc_roots::push_roots();
        let exc_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_exception_new_wtf8(exc_kind, &message));
        // Everything below is read back out of its slot: `w_list_new_object`
        // takes elements its caller has already pinned, and each pin keeps its
        // object alive without keeping it in place.
        let exc = || pyre_object::gc_roots::shadow_stack_get(exc_slot);
        if let Some(w_target) = subclass.and_then(crate::builtins::lookup_exc_class) {
            // Fresh instance: no recorded `w_class?` read can name it yet.
            unsafe {
                (*(exc() as *mut pyre_object::PyObject)).w_class = w_target;
            }
        }
        let errno_slot = exc_slot + 1;
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(errno as i64));
        let strerror_slot = errno_slot + 1;
        let _ =
            pyre_object::gc_roots::pin_root(pyre_object::w_str_from_wtf8_managed(strerror.clone()));
        let args_list = pyre_object::interp_exceptions::w_exception_args_new(vec![
            pyre_object::gc_roots::shadow_stack_get(errno_slot),
            pyre_object::gc_roots::shadow_stack_get(strerror_slot),
        ]);
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_args(exc(), args_list);
            let w_errno = pyre_object::w_int_new(errno as i64);
            pyre_object::interp_exceptions::w_exception_set_errno(exc(), w_errno);
            let w_strerror = pyre_object::w_str_from_wtf8_managed(strerror.clone());
            pyre_object::interp_exceptions::w_exception_set_strerror(exc(), w_strerror);
        }
        make_pyerror(
            kind,
            DisplayMessage::Text(message),
            exc(),
            true,
            false,
            -1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    }

    /// `error.py:new_import_error` — `ImportError(msg, name=name, path=path)`.
    /// A failed `from X import Y` (pyopcode.py import_from) raises through
    /// this so `.name` (the package) and `.path` (its file) are readable,
    /// which `PyError::new(ImportError, msg)` cannot carry.
    pub fn new_import_error(
        msg: impl Into<Wtf8Buf>,
        w_name: PyObjectRef,
        w_path: PyObjectRef,
    ) -> Self {
        Self::new_import_error_name_from(msg, w_name, w_path, pyre_object::PY_NULL)
    }

    /// `_PyErr_SetImportErrorWithNameFrom` — the same error carrying the name
    /// the `from` list asked for.  `traceback.py`
    /// `TracebackException.__init__` gates its spelling
    /// suggestion on `.name_from` being other than `None`, so only a raise that
    /// stamps it can offer one.
    pub fn new_import_error_name_from(
        msg: impl Into<Wtf8Buf>,
        w_name: PyObjectRef,
        w_path: PyObjectRef,
        w_name_from: PyObjectRef,
    ) -> Self {
        let message = msg.into();
        // Root the caller's name/path and the fresh exception across the
        // exception and message allocations below: they live only in Rust
        // locals, which the precise collector does not scan, so a collection
        // inside `w_exception_new` / `w_str_new` could sweep them before the
        // setters stamp them onto `exc`.
        let _roots = pyre_object::gc_roots::push_roots();
        let name_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_name);
        let path_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_path);
        let name_from_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_name_from);
        let exc_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ =
            pyre_object::gc_roots::pin_root(w_exception_new_wtf8(ExcKind::ImportError, &message));
        // The exception is read back out of its slot at every use: the pin
        // keeps it alive without keeping it in place, and the message
        // allocation below can relocate it.
        let exc = || pyre_object::gc_roots::shadow_stack_get(exc_slot);
        // `ImportError.__init__` mirrors args[0] into the dedicated `msg`
        // slot; the prebuilt-instance path bypasses it, so stamp it here.
        let w_msg = if message.is_empty() {
            pyre_object::w_none()
        } else {
            pyre_object::w_str_from_wtf8_managed(message.clone())
        };
        // Reload name/path after the message allocation: the pins keep them
        // alive, but a minor collection may have relocated the young objects,
        // leaving these raw locals stale.
        let w_name = pyre_object::gc_roots::shadow_stack_get(name_slot);
        let w_path = pyre_object::gc_roots::shadow_stack_get(path_slot);
        let w_name_from = pyre_object::gc_roots::shadow_stack_get(name_from_slot);
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_import_msg(exc(), w_msg);
            pyre_object::interp_exceptions::w_exception_set_name(exc(), w_name);
            pyre_object::interp_exceptions::w_exception_set_import_path(exc(), w_path);
            pyre_object::interp_exceptions::w_exception_set_import_name_from(exc(), w_name_from);
        }
        make_pyerror(
            PyErrorKind::ImportError,
            DisplayMessage::Text(message),
            exc(),
            true,
            false,
            -1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    }

    /// pypy/module/_weakref/interp__weakref.py — raised by `force()`
    /// when the referent of a proxy is no longer alive.
    pub fn reference_error(msg: impl Into<Wtf8Buf>) -> Self {
        Self::new(PyErrorKind::ReferenceError, msg)
    }

    pub fn recursion_error(msg: impl Into<Wtf8Buf>) -> Self {
        Self::new(PyErrorKind::RecursionError, msg)
    }

    /// rpython/jit/metainterp/compile.py `memory_error = MemoryError()`
    /// — module-level singleton instance the JIT raises through
    /// `PropagateExceptionDescr.handle_fail` when a malloc helper
    /// returns NULL.
    pub fn memory_error(msg: impl Into<Wtf8Buf>) -> Self {
        Self::new(PyErrorKind::MemoryError, msg)
    }

    pub fn stop_iteration() -> Self {
        Self::new(PyErrorKind::StopIteration, String::new())
    }

    pub fn stop_async_iteration() -> Self {
        Self::new(PyErrorKind::StopAsyncIteration, String::new())
    }

    /// `error.py OperationError.get_w_value`.
    ///
    /// Returns `_w_value`, computing a text object from the display
    /// message when the field is still null (`space.newtext`). Does not
    /// instantiate `w_type`; that is [`Self::normalize_exception`].
    pub fn get_w_value(&mut self, _space: PyObjectRef) -> PyObjectRef {
        let _ = _space;
        if !self.exc_object.is_null() {
            return self.exc_object;
        }
        if !self.has_display_message() {
            return pyre_object::w_none();
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let handle_slot = self.pin(&_roots);
        let w_value = pyre_object::w_str_from_wtf8_managed(self.display_message_buf());
        self.reload(&_roots, handle_slot);
        self.set_exc_object(w_value);
        self.exc_object
    }

    /// Convert to a W_BaseException for pushing onto the value stack.
    /// Reuses the cached object from from_exc_object() if available, otherwise
    /// materialises it once and memoises it into `self.exc_object` so repeat
    /// calls return the same instance (`get_w_value` write-once, error.py:349).
    ///
    /// Mirrors `pypy/interpreter/error.py:OperationError.get_w_value`'s
    /// upgrade-to-exception-instance path: when the OperationError
    /// carries a raw message (the `oefmt` shape), the materialised
    /// exception instance gets `args = (msg,)` per
    /// `pypy/module/exceptions/interp_exceptions.py
    /// W_BaseException.descr_init` — `self.args_w = args_w`.  Pyre
    /// stores `args_w` as the fixed-size item array (`ll_fixed_newlist`),
    /// so a raw message stamps that one-element array and `str(e)`,
    /// `repr(e)`, and `e.args == (msg,)` follow `descr_str` /
    /// `descr_getargs`.
    pub fn to_exc_object(&mut self) -> PyObjectRef {
        if self.has_exception_instance() {
            return self.exc_object;
        }
        // `from_type_and_value` / `oefmt`: `_w_value` is the message object
        // and the instance is `w_type(w_value)`. `PyError::new` still carries
        // a display-message rstr and uses the kind path below, which does
        // not run application-level constructors.
        if !self.w_type.is_null() && !self.has_display_message() {
            if let Ok(w_value) = self.normalize_exception(pyre_object::PY_NULL) {
                if !w_value.is_null() && unsafe { pyre_object::is_exception(w_value) } {
                    return w_value;
                }
            }
        }
        // Root the deferred name/obj context references across the exception
        // allocation below: they live only in this Rust `PyError`, which the
        // precise collector does not scan, so a collection inside
        // `w_exception_new` could sweep them before they are stamped onto `exc`
        // at the `set_name` / `set_attr_obj` calls.
        let _roots = pyre_object::gc_roots::push_roots();
        let handle_slot = self.pin(&_roots);
        let name_ctx_slot = pyre_object::gc_roots::shadow_stack_len();
        if !self.w_name_context.is_null() {
            let _ = pyre_object::gc_roots::pin_root(self.w_name_context);
        }
        let obj_ctx_slot = pyre_object::gc_roots::shadow_stack_len();
        if !self.w_obj_context.is_null() {
            let _ = pyre_object::gc_roots::pin_root(self.w_obj_context);
        }
        // `w_exception_new_empty`, not `w_exception_new`: the message block
        // below builds `args_w` itself (it must keep the message object in a
        // shadow-stack slot for the `ImportError` `msg` stamp), so letting the
        // message helper build a first string + list here only to overwrite
        // both allocated two W_Str/W_List pairs per raise and threw one away.
        let exc_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(
            pyre_object::interp_exceptions::w_exception_new_empty(self.to_exc_kind()),
        );
        // The exception is a nursery object (`alloc_exception_nursery` →
        // `collect_and_reserve`). `pin_root` keeps it alive without keeping
        // the local in place: `w_exception_args_new` collects, so every use
        // below re-reads the slot.
        let exc = || pyre_object::gc_roots::shadow_stack_get(exc_slot);
        self.reload(&_roots, handle_slot);
        if self.has_display_message() {
            let msg_slot = pyre_object::gc_roots::shadow_stack_len();
            let msg = pyre_object::w_str_from_wtf8_managed(self.display_message_buf());
            let _ = pyre_object::gc_roots::pin_root(msg);
            let msg = pyre_object::gc_roots::shadow_stack_get(msg_slot);
            let args_list = pyre_object::interp_exceptions::w_exception_args_new(vec![msg]);
            unsafe { pyre_object::interp_exceptions::w_exception_set_args(exc(), args_list) };
            self.reload(&_roots, handle_slot);
            // `ImportError` / `ModuleNotFoundError` expose the message through a
            // dedicated `msg` slot (`ImportError.__init__` stores `args[0]`
            // there). The raw-message raise path bypasses `__init__`, so stamp
            // `msg` here too — otherwise `e.msg` reads back `None`.
            if matches!(
                self.kind,
                PyErrorKind::ImportError | PyErrorKind::ModuleNotFoundError
            ) {
                // Reload `msg` after the args allocation: the pin keeps it
                // alive, but `w_exception_args_new` may have relocated the young string,
                // leaving this raw local pointing at the old address.
                let msg = pyre_object::gc_roots::shadow_stack_get(msg_slot);
                unsafe { pyre_object::interp_exceptions::w_exception_set_import_msg(exc(), msg) };
            }
            // `W_SyntaxError.descr_init` likewise stores `args_w[0]` in its
            // dedicated `w_msg` field.  Internal parser/codegen errors use
            // this raw-message path, so mirror that store here.
            if self.kind == PyErrorKind::SyntaxError {
                let msg = pyre_object::gc_roots::shadow_stack_get(msg_slot);
                unsafe { pyre_object::interp_exceptions::w_exception_set_syntax_msg(exc(), msg) };
            }
            // `W_SystemExit.descr_init`: a single constructor argument is `code`.
            if self.kind == PyErrorKind::SystemExit {
                let msg = pyre_object::gc_roots::shadow_stack_get(msg_slot);
                unsafe { pyre_object::interp_exceptions::w_exception_set_code(exc(), msg) };
            }
        }
        // Stamp the deferred `name` / `obj` context onto the freshly
        // materialised NameError / AttributeError instance, the lazy
        // equivalent of `_PyEval_FormatExcCheckArg` /
        // `set_attribute_error_context` (Python 3.10+).
        // Reload the deferred context refs after the exception / args
        // allocations: the pins keep them alive, but a minor collection may
        // have relocated the young objects, leaving the stored fields stale.
        // Store each reloaded address back on the error as well. The pins
        // above pop when this scope ends, and `walk_gc_refs` keeps forwarding
        // both fields afterwards, so leaving the pre-move address here would
        // hand the collector a moved-from slot.
        self.reload(&_roots, handle_slot);
        if !self.w_name_context.is_null() {
            let w_name_context = pyre_object::gc_roots::shadow_stack_get(name_ctx_slot);
            self.set_w_name_context(w_name_context);
            unsafe { pyre_object::interp_exceptions::w_exception_set_name(exc(), w_name_context) };
        }
        if !self.w_obj_context.is_null() {
            let w_obj_context = pyre_object::gc_roots::shadow_stack_get(obj_ctx_slot);
            self.set_w_obj_context(w_obj_context);
            unsafe {
                pyre_object::interp_exceptions::w_exception_set_attr_obj(exc(), w_obj_context)
            };
        }
        // Write-once memo (`get_w_value` self.w_value): cache the materialised
        // instance so a second call returns the same object (identity `e1 is e2`)
        // instead of a fresh allocation.
        self.set_exc_object(exc());
        self.exc_object
    }

    /// `error.py OperationError.normalize_exception` is `@jit.unroll_safe`.
    /// Instantiates `w_type(w_value)` on first need and caches the instance
    /// in `exc_object` (`_w_value`).
    #[majit_macros::unroll_safe]
    pub fn normalize_exception(&mut self, space: PyObjectRef) -> Result<PyObjectRef, PyError> {
        if self.has_exception_instance() {
            return Ok(self.exc_object);
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let handle_slot = self.pin(&_roots);
        let mut w_type = self.resolved_w_type();
        if w_type.is_null() {
            // Bootstrap: exception classes are not registered yet, so
            // `w_type` stays null and `kind` names the class. The instance
            // is built by [`Self::to_exc_object`]'s kind path.
            self.reload(&_roots, handle_slot);
            return Err(PyError::type_error("exception class not initialised"));
        }
        let type_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_type);
        let w_value = self.get_w_value(space);
        self.reload(&_roots, handle_slot);
        w_type = pyre_object::gc_roots::shadow_stack_get(type_slot);
        let mut norm = ExceptionNormalization::new(w_type, w_value);
        let w_value = pyre_object::with_roots!(w_type => norm.normalize_exception(space))?;
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_value);
        self.reload(&_roots, handle_slot);
        let w_value = pyre_object::gc_roots::shadow_stack_get(value_slot);
        self.set_w_type(norm.w_type);
        self.set_exc_object(w_value);
        if !self.w_name_context.is_null() {
            unsafe {
                pyre_object::interp_exceptions::w_exception_set_name(w_value, self.w_name_context)
            };
        }
        if !self.w_obj_context.is_null() {
            unsafe {
                pyre_object::interp_exceptions::w_exception_set_attr_obj(
                    w_value,
                    self.w_obj_context,
                )
            };
        }
        Ok(w_value)
    }

    /// `error.py` — `OperationError.get_traceback`:
    ///
    /// ```python
    /// def get_traceback(self):
    ///     tb = self._application_traceback
    ///     if tb is not None and isinstance(tb, PyTraceback):
    ///         tb.frame.mark_as_escaped()
    ///     return tb
    /// ```
    ///
    /// Upstream keeps the chain head in an `_application_traceback` field
    /// beside `_w_value`, so it can answer while `_w_value` is still `None`.
    /// This carrier keeps it on the materialised instance's `w_traceback`
    /// slot instead, for the reason [`to_exc_object`] exists at all: a
    /// propagating Rust `PyError` is memcpy'd at every `?` and so cannot be
    /// registered as a GC root, while the instance can — `set_in_flight_exception`
    /// publishes it and the collector then reaches the whole chain through it.
    /// An unmaterialised error therefore has no chain to report, which is the
    /// same answer upstream's `None` gives: nothing has recorded a frame yet.
    ///
    /// 3.14 does not defer the instance either — `_PyErr_SetObject` comments
    /// "We must normalize the value right now" and calls
    /// `_PyErr_CreateException` before storing anything — so building it at
    /// the first unwind, rather than never, is the closer of the two.
    ///
    /// Reading marks the head frame escaped, upstream's documented side
    /// effect — see [`crate::pytraceback::mark_traceback_escaped`] for which
    /// single-frame case that covers.
    pub fn get_traceback(&self) -> PyObjectRef {
        let tb = self.application_traceback();
        unsafe { crate::pytraceback::mark_traceback_escaped(tb) };
        tb
    }

    /// The `_application_traceback` field read itself, without
    /// [`get_traceback`](Self::get_traceback)'s escape mark.  Upstream reads
    /// the field directly wherever the chain must not escape, and goes through
    /// `get_traceback` everywhere else.
    fn application_traceback(&self) -> PyObjectRef {
        if !self.has_exception_instance() {
            return pyre_object::PY_NULL;
        }
        unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(self.exc_object) }
    }

    /// `error.py` — `OperationError.has_any_traceback`:
    ///
    /// ```python
    /// def has_any_traceback(self):
    ///     return self._application_traceback is not None
    /// ```
    ///
    /// The plain field test, so it neither materialises nor marks anything
    /// escaped; [`get_traceback`](Self::get_traceback) is the reading form.
    pub fn has_any_traceback(&self) -> bool {
        !self.application_traceback().is_null()
    }

    /// `error.py` — `OperationError.got_any_traceback`, upstream's second
    /// spelling of the same test, kept so both call sites port across.
    pub fn got_any_traceback(&self) -> bool {
        self.has_any_traceback()
    }

    /// `error.py` — `OperationError.get_w_traceback`:
    ///
    /// ```python
    /// def get_w_traceback(self, space):
    ///     """Return a traceback or w_None. """
    ///     tb = self.get_traceback()
    ///     if tb is None:
    ///         return space.w_None
    ///     return tb
    /// ```
    pub fn get_w_traceback(&self, _space: PyObjectRef) -> PyObjectRef {
        let tb = self.get_traceback();
        if tb.is_null() {
            return pyre_object::w_none();
        }
        tb
    }

    /// `error.py` — `OperationError.set_traceback`:
    ///
    /// ```python
    /// def set_traceback(self, traceback):
    ///     """Set the current traceback."""
    ///     self._application_traceback = traceback
    /// ```
    ///
    /// The store is the one operation that cannot be answered without the
    /// instance, since that is where [`get_traceback`](Self::get_traceback)
    /// reads it back from, so this is where an error still carrying only a
    /// message materialises.  Callers that go on to allocate must sequence
    /// their own allocation after this one: the instance is what roots the
    /// chain, so a node built before it exists is unreachable.
    pub fn set_traceback(&mut self, mut w_traceback: PyObjectRef) {
        let exc = pyre_object::with_roots!(w_traceback => self.to_exc_object());
        if exc.is_null() {
            return;
        }
        unsafe { pyre_object::interp_exceptions::w_exception_set_traceback(exc, w_traceback) };
    }

    /// `error.py` — `OperationError.remove_traceback_module_frames`:
    ///
    /// ```python
    /// def remove_traceback_module_frames(self, *module_names):
    ///     tb = self._application_traceback
    ///     while tb is not None and isinstance(tb, PyTraceback):
    ///         if tb.frame.pycode.co_filename not in module_names:
    ///             break
    ///         tb = tb.next
    ///     self._application_traceback = tb
    /// ```
    ///
    /// Drops the leading, outermost run of traceback entries raised from one
    /// of `module_names`, so machinery a module implements in Python does not
    /// surface in an error it re-raises; the first entry from anywhere else
    /// stops the walk and the rest of the chain is kept intact.
    ///
    /// [`traceback_frame_names_module`] is the membership test, which widens
    /// upstream's `not in module_names` by one case pyre needs.
    pub fn remove_traceback_module_frames(&mut self, module_names: &[&str]) {
        let exc = self.to_exc_object();
        if exc.is_null() {
            return;
        }
        unsafe {
            use pyre_object::gc_roots::{
                pin_root, push_roots, shadow_stack_get, shadow_stack_len, shadow_stack_set,
            };

            // `co_filename` is realised by `code_get_field`, which allocates
            // and can therefore collect.  A traceback node emitted by compiled
            // code is nursery-resident, so a collection moves it and a raw
            // cursor carried across the call would name reclaimed memory —
            // both when the walk steps to `w_next` and when the survivor is
            // republished below.  Keep the cursor in a root slot and re-read it
            // after every call that can allocate, the discipline
            // `write_traceback_chain` already follows for its own walk.
            let _roots = push_roots();
            let handle_slot = self.pin(&_roots);
            let exc_slot = shadow_stack_len();
            let exc = pin_root(exc);
            // Read the chain off the pinned address, not off
            // `self.exc_object`: `pin_root` normalizes, so a collection during
            // the publish leaves the field naming where the exception used to
            // be.  Store the pinned address back for the same reason
            // `to_exc_object` does with the deferred context refs.
            // The handle itself moves with that collection; reload it before
            // the field write.
            self.reload(&_roots, handle_slot);
            self.set_exc_object(exc);
            let tb_slot = shadow_stack_len();
            let _ = pin_root(pyre_object::interp_exceptions::w_exception_get_traceback(
                exc,
            ));
            let code_slot = shadow_stack_len();
            let _ = pin_root(pyre_object::PY_NULL);
            loop {
                // `while tb is not None and isinstance(tb, PyTraceback)`.
                let tb = shadow_stack_get(tb_slot);
                if tb.is_null()
                    || pyre_object::is_none(tb)
                    || !crate::pytraceback::is_pytraceback(tb)
                {
                    break;
                }
                let w_code = crate::pytraceback::w_pytraceback_get_w_code(tb);
                if w_code.is_null() {
                    break;
                }
                shadow_stack_set(code_slot, w_code);
                let names_a_module =
                    crate::pycode::code_get_field(shadow_stack_get(code_slot), "co_filename")
                        .ok()
                        .filter(|f| pyre_object::is_str(*f))
                        // A module imported from a path with no UTF-8 spelling
                        // carries a surrogate escape in `co_filename`; it is
                        // not one of the named modules either way.
                        .and_then(|f| pyre_object::w_str_get_value_opt(f))
                        .is_some_and(|f| traceback_frame_names_module(f, module_names));
                if !names_a_module {
                    break;
                }
                let tb = shadow_stack_get(tb_slot);
                shadow_stack_set(tb_slot, crate::pytraceback::w_pytraceback_get_w_next(tb));
            }
            // `self._application_traceback = tb`.  `exc_object` still names the
            // pinned exception, so the store lands on the same instance the
            // walk read from.  `code_get_field` allocated during the walk;
            // reload the handle before using it.
            self.reload(&_roots, handle_slot);
            debug_assert_eq!(self.exc_object, shadow_stack_get(exc_slot));
            self.set_traceback(shadow_stack_get(tb_slot));
        }
    }

    /// pypy/interpreter/error.py `OperationError.match`.
    ///
    /// ```python
    /// def match(self, space, w_check_class):
    ///     "Check if this application-level exception matches 'w_check_class'."
    ///     return space.exception_match(self.w_type, w_check_class)
    /// ```
    ///
    /// Upstream reads `self.w_type` and calls `space.exception_match`.
    /// The instance is not materialised.
    pub fn match_(&mut self, _space: PyObjectRef, mut w_check_class: PyObjectRef) -> bool {
        let _ = _space;
        let _roots = pyre_object::gc_roots::push_roots();
        let handle_slot = self.pin(&_roots);
        let check_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_check_class);
        self.reload(&_roots, handle_slot);
        let w_type = self.resolved_w_type();
        if w_type.is_null() {
            return false;
        }
        let type_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_type);
        let matched = crate::baseobjspace::exception_match(
            pyre_object::gc_roots::shadow_stack_get(type_slot),
            pyre_object::gc_roots::shadow_stack_get(check_slot),
        );
        self.reload(&_roots, handle_slot);
        matched
    }

    /// pypy/interpreter/error.py `OperationError.async`.
    ///
    /// ```python
    /// def async(self, space):
    ///     "Check if this is an exception that should better not be caught."
    ///     return (self.match(space, space.w_SystemExit) or
    ///             self.match(space, space.w_KeyboardInterrupt))
    ///     # note: an extra case is added in OpErrFmtNoArgs
    /// ```
    ///
    /// Named `r#async` because `async` is a Rust keyword, the same escape
    /// `typedef::r#type` uses for `space.type`.
    ///
    /// The guard sites all read the same way: an error is about to be
    /// swallowed and turned into a fallback answer, and this says which
    /// errors must not be. The two named classes are the ones a program has
    /// no business converting into `None` or `NotImplemented` — an exit
    /// request and a Ctrl-C.
    ///
    /// `KeyboardInterrupt` has no `ExcKind`. When `w_type` is set, `match`
    /// answers from the class without materialising. When `w_type` is still
    /// null (bootstrap), `kind` is the fallback: nothing unmaterialised is
    /// a `KeyboardInterrupt`.
    pub fn r#async(&mut self, space: PyObjectRef) -> bool {
        if !self.resolved_w_type().is_null() {
            return ["SystemExit", "KeyboardInterrupt"].iter().any(|name| {
                crate::builtins::lookup_exc_class(name)
                    .is_some_and(|w_class| self.match_(space, w_class))
            });
        }
        self.kind == PyErrorKind::SystemExit
    }

    /// pypy/interpreter/error.py `chain_exceptions`.
    ///
    /// ```python
    /// def chain_exceptions(self, space, context):
    ///     w_value = self.normalize_exception(space)
    ///     w_context = context.normalize_exception(space)
    ///     if not space.is_w(w_value, w_context):
    ///         if not isinstance(w_value, W_BaseException):
    ///             raise oefmt(space.w_SystemError,
    ///                         "not an instance of Exception: %T", w_value)
    ///         if w_value.w_context is None:
    ///             _break_context_cycle(space, w_value, w_context)
    ///             w_value.descr_setcontext(space, w_context)
    /// ```
    ///
    /// The `w_context is None` test, the cycle break and the write are the
    /// three steps [`chain_context`] already performs on a pair of normalized
    /// exceptions, so they are taken there rather than spelled twice; what
    /// stays here is the part that reports instead of skipping.
    pub fn chain_exceptions(
        &mut self,
        mut space: PyObjectRef,
        context: &mut OperationError,
    ) -> Result<(), PyError> {
        let mut w_value = pyre_object::with_roots!(space => self.normalize_exception(space))?;
        let w_context = pyre_object::with_roots!(w_value => context.normalize_exception(space))?;
        if std::ptr::eq(w_value, w_context) {
            return Ok(());
        }
        if !unsafe { pyre_object::is_exception(w_value) } {
            return Err(PyError::new(
                crate::PyErrorKind::SystemError,
                "not an instance of Exception".to_string(),
            ));
        }
        chain_context(w_value, w_context);
        Ok(())
    }

    /// `error.py OperationError.record_context`: chain whatever exception is
    /// currently being handled as this one's `__context__`, once.
    ///
    /// ```python
    /// def record_context(self, space, ec):
    ///     if self._context_recorded:
    ///         return
    ///     last = ec.sys_exc_info()
    ///     try:
    ///         if last is not None:
    ///             self.chain_exceptions(space, last)
    ///     finally:
    ///         self._context_recorded = True
    /// ```
    ///
    /// Upstream reads `ec.sys_exc_info()` itself.  The source is a parameter
    /// here because an exception thrown *into* a resumed generator asks the
    /// same question of the flat slot instead — see [`ContextSource`].  The
    /// mark is set whether or not anything was found, which is the `finally`:
    /// a null `__context__` cannot say whether it was recorded as absent or
    /// never recorded at all.
    pub fn record_context(&mut self, source: crate::eval::ContextSource) {
        if self.context_recorded {
            return;
        }
        let last = match source {
            crate::eval::ContextSource::GeneratorChain => crate::eval::get_sys_exception(),
            crate::eval::ContextSource::ResumedFrameOnly => crate::eval::get_current_exception(),
        };
        chain_context(self.exc_object, last);
        self.set_context_recorded(true);
    }

    /// `error.py OperationError.set_cause`.
    ///
    /// ```python
    /// def set_cause(self, space, w_cause):
    ///     if w_cause is None:
    ///         return
    ///     if space.is_none(w_cause):
    ///         pass
    ///     else:
    ///         self._exception_getclass(space, w_cause, "exception causes")
    ///     w_value = self.get_w_value(space)
    ///     space.setattr(w_value, space.newtext("__cause__"), w_cause)
    /// ```
    ///
    /// The `setattr` lands on `descr_setcause`, which stamps
    /// `suppress_context` beside the slot; both writes are spelled out
    /// because the slot is reached directly.
    pub fn set_cause(
        &mut self,
        space: PyObjectRef,
        mut w_cause: PyObjectRef,
    ) -> Result<(), PyError> {
        if w_cause.is_null() {
            return Ok(());
        }
        let roots = pyre_object::gc_roots::push_roots();
        let cause_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_cause);
        let handle_slot = self.pin(&roots);
        let w_cause = pyre_object::gc_roots::shadow_stack_get(cause_slot);
        if !unsafe { pyre_object::is_none(w_cause) } {
            let w_cause_type =
                crate::typedef::r#type(w_cause).map_or(pyre_object::PY_NULL, |p| p.as_ptr());
            if w_cause_type.is_null()
                || !unsafe { crate::baseobjspace::exception_is_valid_class_w(w_cause_type) }
            {
                return Err(PyError::type_error(
                    "exception causes must derive from BaseException",
                ));
            }
        }
        self.reload(&roots, handle_slot);
        let w_value = self.normalize_exception(space)?;
        let w_cause = pyre_object::gc_roots::shadow_stack_get(cause_slot);
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_cause(w_value, w_cause);
            pyre_object::interp_exceptions::w_exception_set_suppress_context(w_value, true);
        }
        Ok(())
    }

    /// `error.py OperationError.chain_exceptions_from_cause`.
    ///
    /// ```python
    /// def chain_exceptions_from_cause(self, space, exception):
    ///     self.chain_exceptions(space, exception)
    ///     self.set_cause(space, exception.get_w_value(space))
    ///     self.record_context(space, space.getexecutioncontext())
    /// ```
    ///
    /// `getexecutioncontext` is the `sys_exc_info` source, so the recording
    /// asks [`ContextSource::GeneratorChain`].
    pub fn chain_exceptions_from_cause(
        &mut self,
        mut space: PyObjectRef,
        exception: &mut OperationError,
    ) -> Result<(), PyError> {
        let roots = pyre_object::gc_roots::push_roots();
        let self_slot = self.pin(&roots);
        let exception_slot = exception.pin(&roots);
        let chained = pyre_object::with_roots!(space => self.chain_exceptions(space, exception));
        self.reload(&roots, self_slot);
        exception.reload(&roots, exception_slot);
        chained?;
        let w_cause = pyre_object::with_roots!(space => exception.normalize_exception(space));
        self.reload(&roots, self_slot);
        let w_cause = w_cause?;
        self.set_cause(space, w_cause)?;
        self.record_context(crate::eval::ContextSource::GeneratorChain);
        Ok(())
    }

    /// pypy/interpreter/error.py `write_unraisable`, with the
    /// `with_traceback=False` first-line shape.
    pub fn write_unraisable(
        &mut self,
        space: PyObjectRef,
        where_desc: &rustpython_wtf8::Wtf8,
        w_object: PyObjectRef,
    ) {
        self.write_unraisable_with_traceback(space, where_desc, w_object, false);
    }

    /// `error.py OperationError.write_unraisable`, including the
    /// `with_traceback` spelling selected by callers such as
    /// `interp_magic.write_unraisable`.
    pub fn write_unraisable_with_traceback(
        &mut self,
        mut space: PyObjectRef,
        where_desc: &rustpython_wtf8::Wtf8,
        mut w_object: PyObjectRef,
        with_traceback: bool,
    ) {
        let roots = pyre_object::gc_roots::push_roots();
        let self_slot = self.pin(&roots);
        let mut w_value = pyre_object::with_roots!(space, w_object => {
            self.reload(&roots, self_slot);
            self.normalize_exception(space)
        })
        .unwrap_or_else(|_| {
            self.reload(&roots, self_slot);
            self.to_exc_object()
        });
        let mut w_type = crate::baseobjspace::exception_getclass(w_value);
        if w_type.is_null() {
            w_type = pyre_object::w_none();
        }
        // `error.py` — `w_tb = self.get_w_traceback(space)`, the slot
        // read with its escape mark.
        let mut w_tb = if !w_value.is_null() && unsafe { pyre_object::is_exception(w_value) } {
            unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(w_value) }
        } else {
            pyre_object::PY_NULL
        };
        unsafe { crate::pytraceback::mark_traceback_escaped(w_tb) };
        if w_tb.is_null() {
            w_tb = pyre_object::w_none();
        }
        let mut w_object = if w_object.is_null() {
            pyre_object::w_none()
        } else {
            w_object
        };
        // The description names an object — a thread's callable, a ctypes
        // callback — so it may hold a lone surrogate and is carried as WTF-8.
        let mut first_line = if where_desc.is_empty() {
            Wtf8Buf::new()
        } else if where_desc.as_bytes().starts_with(b"Exception ignored ") {
            where_desc.to_wtf8_buf()
        } else if with_traceback {
            crate::display::wtf8_format!("Exception ignored ", where_desc)
        } else {
            crate::display::wtf8_format!("Exception ignored in: ", where_desc)
        };
        // vm.py UnraisableHookArgs also carries `extra_line`; pyre's hook-args
        // structseq targets the 5-field shape, so only the default printer
        // receives the Rust-side extra line.
        let mut hook_fields = pyre_object::gc_roots::RootedItems::new();
        hook_fields.push(w_type);
        hook_fields.push(w_value);
        hook_fields.push(w_tb);
        // Pin the object before the managed first-line mint; the structseq
        // field order is type, value, traceback, line, object.
        hook_fields.push(w_object);
        hook_fields.push(if first_line.is_empty() {
            pyre_object::w_none()
        } else {
            pyre_object::w_str_from_wtf8_managed(first_line.clone())
        });
        let live = hook_fields.take();
        let mut hook_args = pyre_object::with_roots!(space, w_object, w_tb, w_type, w_value =>
            crate::_structseq::new_instance(
                unraisable_hook_args_type(),
                vec![live[0], live[1], live[2], live[4], live[3]],
            )
        );
        if let Some(sys_mod) = crate::importing::get_interpreter_sys_module()
            && let Ok(mut w_hook) = pyre_object::with_roots!(
                hook_args, space, w_object, w_tb, w_type, w_value =>
                crate::baseobjspace::getattr_str(sys_mod, "unraisablehook")
            )
            && !w_hook.is_null()
            && !unsafe { pyre_object::is_none(w_hook) }
        {
            match pyre_object::with_roots!(space, w_hook =>
                crate::call::call_function_impl_result(w_hook, &[hook_args])
            ) {
                Ok(_) => return,
                Err(mut hook_err) => {
                    first_line =
                        Wtf8Buf::from_string("Exception ignored in sys.unraisablehook".to_string());
                    w_object = w_hook;
                    let hook_roots = pyre_object::gc_roots::push_roots();
                    let hook_err_slot = hook_err.pin(&hook_roots);
                    let hook_value = pyre_object::with_roots!(space, w_object => {
                        hook_err.reload(&hook_roots, hook_err_slot);
                        hook_err.normalize_exception(space)
                    })
                    .unwrap_or_else(|_| {
                        hook_err.reload(&hook_roots, hook_err_slot);
                        hook_err.to_exc_object()
                    });
                    w_type = crate::baseobjspace::exception_getclass(hook_value);
                    if w_type.is_null() {
                        w_type = pyre_object::w_none();
                    }
                    // `error.py` — same `get_w_traceback` read
                    // for the exception the hook itself raised.
                    w_tb = if !hook_value.is_null()
                        && unsafe { pyre_object::is_exception(hook_value) }
                    {
                        unsafe {
                            pyre_object::interp_exceptions::w_exception_get_traceback(hook_value)
                        }
                    } else {
                        pyre_object::PY_NULL
                    };
                    unsafe { crate::pytraceback::mark_traceback_escaped(w_tb) };
                    if w_tb.is_null() {
                        w_tb = pyre_object::w_none();
                    }
                    Self::write_unraisable_default(
                        space,
                        w_type,
                        hook_value,
                        w_tb,
                        &first_line,
                        w_object,
                        "",
                    );
                    return;
                }
            }
        }
        Self::write_unraisable_default(space, w_type, w_value, w_tb, &first_line, w_object, "");
    }

    /// `PySys_WriteStderr` -- one line of a report, on `sys.stderr` when there
    /// is one and on the host's stream when there is not.
    ///
    /// The one caller is in `cpyext`, which the same condition builds.
    #[cfg(all(
        feature = "cpyext",
        not(feature = "sandbox"),
        any(target_os = "macos", target_os = "linux")
    ))]
    pub(crate) fn write_report_line(buf: &[u8]) {
        if !Self::write_to_sys_stderr(buf) {
            emit_report_to_host_stderr(buf);
        }
    }

    /// Write WTF-8 `buf` through the live `sys.stderr`, so the text spends that
    /// stream's codec and error handler. False for a stream that is missing,
    /// `None` or refuses the write, leaving the caller its own descriptor sink.
    fn write_to_sys_stderr(buf: &[u8]) -> bool {
        let Some(sys_mod) = crate::importing::get_interpreter_sys_module() else {
            return false;
        };
        let stderr = match crate::baseobjspace::getattr_str(sys_mod, "stderr") {
            Ok(w) if !w.is_null() && !unsafe { pyre_object::is_none(w) } => w,
            _ => return false,
        };
        // Every writer into the buffer emits WTF-8, filenames included.  A
        // decode failure would mean a byte none of them put there; read it the
        // way a filesystem name is read so it keeps its escape instead of
        // folding to U+FFFD, and so this sink and the fd sink agree.
        let _roots = pyre_object::gc_roots::push_roots();
        let stream_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(stderr);
        let w_text = match rustpython_wtf8::Wtf8::from_bytes(buf) {
            Some(text) => pyre_object::w_str_from_wtf8_managed(text.to_wtf8_buf()),
            None => {
                pyre_object::w_str_from_wtf8_managed(crate::gateway::fsdecode_filename_wtf8(buf))
            }
        };
        let _ = pyre_object::gc_roots::pin_root(w_text);
        let text_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let result = crate::baseobjspace::call_method(
            pyre_object::gc_roots::shadow_stack_get(stream_slot),
            "write",
            &[pyre_object::gc_roots::shadow_stack_get(text_slot)],
        );
        if result.is_null() {
            let _ = crate::call::take_call_error();
            return false;
        }
        true
    }

    /// pypy/interpreter/error.py `write_unraisable_default`.
    pub fn write_unraisable_default(
        space: PyObjectRef,
        w_type: PyObjectRef,
        mut w_value: PyObjectRef,
        mut w_tb: PyObjectRef,
        first_line: &rustpython_wtf8::Wtf8,
        w_object: PyObjectRef,
        extra_line: &str,
    ) {
        let _ = (space, w_type);
        let mut first_line = if first_line.is_empty() {
            Wtf8Buf::from_string("Exception ignored in".to_string())
        } else {
            first_line.to_wtf8_buf()
        };
        // `errors.c write_unraisable_exc_file` ends the stated message here,
        // whether or not an object follows it.
        first_line.push_str(":");
        if !w_object.is_null() && !unsafe { pyre_object::is_none(w_object) } {
            let objrepr = pyre_object::with_roots!(w_tb, w_value =>
                unsafe { crate::display::py_repr_wtf8(w_object) }
            )
            .unwrap_or_else(|_| Wtf8Buf::from_string("<object repr() failed>".to_string()));
            first_line.push_str(" ");
            first_line.push_wtf8(&objrepr);
        }
        let extra_line = if extra_line.is_empty() {
            "\n".to_string()
        } else {
            format!("{extra_line}:\n")
        };
        let mut buf = Vec::new();
        // The stream takes text, and the printer spells a lone surrogate as
        // the escape it prints elsewhere rather than folding it to U+FFFD.
        let _ = buf.write_all(first_line.as_bytes());
        if !extra_line.is_empty() {
            let _ = write!(&mut buf, "{extra_line}");
        }
        if !w_value.is_null() && unsafe { pyre_object::is_exception(w_value) } {
            if !w_tb.is_null() && !unsafe { pyre_object::is_none(w_tb) } {
                unsafe { pyre_object::interp_exceptions::w_exception_set_traceback(w_value, w_tb) };
            }
            let err = unsafe { PyError::from_exc_object(w_value) };
            let _ = write_exception(&mut buf, &err, true);
        }
        if !Self::write_to_sys_stderr(&buf) {
            emit_report_to_host_stderr(&buf);
        }
    }

    fn to_exc_kind(&self) -> ExcKind {
        kind_to_exc_kind(self.kind)
    }
}

fn kind_to_exc_kind(kind: PyErrorKind) -> ExcKind {
    match kind {
        PyErrorKind::TypeError => ExcKind::TypeError,
        PyErrorKind::ValueError => ExcKind::ValueError,
        PyErrorKind::ZeroDivisionError => ExcKind::ZeroDivisionError,
        PyErrorKind::NameError => ExcKind::NameError,
        PyErrorKind::UnboundLocalError => ExcKind::UnboundLocalError,
        PyErrorKind::IndexError => ExcKind::IndexError,
        PyErrorKind::KeyError => ExcKind::KeyError,
        PyErrorKind::AttributeError => ExcKind::AttributeError,
        PyErrorKind::RuntimeError => ExcKind::RuntimeError,
        PyErrorKind::StopIteration => ExcKind::StopIteration,
        PyErrorKind::StopAsyncIteration => ExcKind::StopAsyncIteration,
        PyErrorKind::OverflowError => ExcKind::OverflowError,
        PyErrorKind::ArithmeticError => ExcKind::ArithmeticError,
        PyErrorKind::ImportError => ExcKind::ImportError,
        PyErrorKind::ModuleNotFoundError => ExcKind::ModuleNotFoundError,
        PyErrorKind::NotImplementedError => ExcKind::NotImplementedError,
        PyErrorKind::AssertionError => ExcKind::AssertionError,
        PyErrorKind::ReferenceError => ExcKind::ReferenceError,
        PyErrorKind::GeneratorExit => ExcKind::GeneratorExit,
        PyErrorKind::RecursionError => ExcKind::RecursionError,
        // Internal-only marker. If it escapes to object-space conversion,
        // degrade to RuntimeError rather than inventing a new exception type.
        PyErrorKind::BytecodeCorruption => ExcKind::RuntimeError,
        PyErrorKind::OSError => ExcKind::OSError,
        PyErrorKind::FileNotFoundError => ExcKind::FileNotFoundError,
        // DescrMismatch is a control-flow exception caught by
        // GetSetProperty.descr_property_get/set/del. If it ever escapes
        // to user code without being converted to TypeError it surfaces
        // as a TypeError, matching PyPy's eventual descr_call_mismatch.
        PyErrorKind::DescrMismatch => ExcKind::TypeError,
        PyErrorKind::SystemExit => ExcKind::SystemExit,
        PyErrorKind::MemoryError => ExcKind::MemoryError,
        PyErrorKind::SystemError => ExcKind::SystemError,
        PyErrorKind::TraceAbort => ExcKind::RuntimeError,
        PyErrorKind::LookupError => ExcKind::LookupError,
        PyErrorKind::UnicodeError => ExcKind::UnicodeError,
        PyErrorKind::UnicodeDecodeError => ExcKind::UnicodeDecodeError,
        PyErrorKind::UnicodeEncodeError => ExcKind::UnicodeEncodeError,
        PyErrorKind::UnicodeTranslateError => ExcKind::UnicodeTranslateError,
        PyErrorKind::SyntaxError => ExcKind::SyntaxError,
        PyErrorKind::BufferError => ExcKind::BufferError,
        PyErrorKind::EOFError => ExcKind::EOFError,
    }
}

impl PyError {
    /// Create a PyError from a W_BaseException.
    ///
    /// # Safety
    /// `obj` must point to a valid `W_BaseException`.
    pub unsafe fn from_exc_object(obj: PyObjectRef) -> Self {
        unsafe {
            let kind = pyre_object::interp_exceptions::w_exception_get_kind(obj);
            // The display string is derived lazily from `exc_object`
            // (`message_text`), never here: conversion runs on every
            // raise propagation, and stringifying the args eagerly would
            // execute their `__str__` at raise time instead of at
            // display time.
            make_pyerror(
                Self::kind_from_exc(kind),
                DisplayMessage::FromExcObject,
                obj,
                true,
                false,
                -1,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                crate::baseobjspace::exception_getclass(obj),
            )
        }
    }

    pub fn kind_from_exc(kind: ExcKind) -> PyErrorKind {
        match kind {
            ExcKind::TypeError => PyErrorKind::TypeError,
            ExcKind::ValueError => PyErrorKind::ValueError,
            ExcKind::ZeroDivisionError => PyErrorKind::ZeroDivisionError,
            ExcKind::NameError => PyErrorKind::NameError,
            ExcKind::UnboundLocalError => PyErrorKind::UnboundLocalError,
            ExcKind::IndexError => PyErrorKind::IndexError,
            ExcKind::KeyError => PyErrorKind::KeyError,
            ExcKind::AttributeError => PyErrorKind::AttributeError,
            ExcKind::RuntimeError => PyErrorKind::RuntimeError,
            ExcKind::StopIteration => PyErrorKind::StopIteration,
            ExcKind::StopAsyncIteration => PyErrorKind::StopAsyncIteration,
            ExcKind::OverflowError => PyErrorKind::OverflowError,
            ExcKind::ArithmeticError => PyErrorKind::ArithmeticError,
            ExcKind::ImportError => PyErrorKind::ImportError,
            ExcKind::ModuleNotFoundError => PyErrorKind::ModuleNotFoundError,
            ExcKind::NotImplementedError => PyErrorKind::NotImplementedError,
            ExcKind::AssertionError => PyErrorKind::AssertionError,
            ExcKind::ReferenceError => PyErrorKind::ReferenceError,
            ExcKind::GeneratorExit => PyErrorKind::GeneratorExit,
            ExcKind::RecursionError => PyErrorKind::RecursionError,
            ExcKind::BaseException | ExcKind::Exception => PyErrorKind::RuntimeError,
            ExcKind::OSError => PyErrorKind::OSError,
            ExcKind::FileNotFoundError => PyErrorKind::FileNotFoundError,
            // `pypy/module/exceptions/interp_exceptions.py:418`
            // W_UnicodeError = _new_exception('UnicodeError',
            // W_ValueError, ...).  Each Unicode subclass round-trips
            // through its own PyErrorKind variant so render_exception
            // preserves the exact class.
            ExcKind::UnicodeError => PyErrorKind::UnicodeError,
            ExcKind::UnicodeDecodeError => PyErrorKind::UnicodeDecodeError,
            ExcKind::UnicodeEncodeError => PyErrorKind::UnicodeEncodeError,
            // `pypy/module/exceptions/interp_exceptions.py W_UnicodeTranslateError`
            // W_UnicodeTranslateError — subclass of UnicodeError.
            ExcKind::UnicodeTranslateError => PyErrorKind::UnicodeTranslateError,
            ExcKind::SystemExit => PyErrorKind::SystemExit,
            ExcKind::MemoryError => PyErrorKind::MemoryError,
            ExcKind::SystemError => PyErrorKind::SystemError,
            // `pypy/module/exceptions/interp_exceptions.py:474`
            // W_LookupError = _new_exception('LookupError', W_Exception,
            // ...) — intermediate parent of IndexError / KeyError.
            ExcKind::LookupError => PyErrorKind::LookupError,
            ExcKind::SyntaxError => PyErrorKind::SyntaxError,
            ExcKind::BufferError => PyErrorKind::BufferError,
            ExcKind::EOFError => PyErrorKind::EOFError,
        }
    }

    /// The exception text without the class name.  Rust-constructed
    /// errors carry it in `message`; object-backed errors
    /// (`from_exc_object`) derive it from `exc_object` on demand —
    /// `W_BaseException.descr_str` formats `args_w`, so the args'
    /// `__str__` runs at display time, not at raise time.
    pub fn message_text(&self) -> String {
        if self.has_display_message() || self.exc_object.is_null() {
            // Display-side only: `message` is WTF-8 and may hold a lone
            // surrogate, so spend the escape the stream would spend rather
            // than folding it to U+FFFD.  A caller that needs the value
            // itself wants [`message_wtf8`].
            return crate::display::wtf8_display_string(
                self.display_message_buf(),
                "<unprintable>",
            );
        }
        // Infallible Display-side context: a raising `__str__` degrades to
        // the placeholder rather than propagating, and a lone surrogate is
        // backslash-escaped rather than panicking.
        unsafe { crate::display::py_str_display(self.exc_object) }
    }

    /// [`message_text`] without the display encode — the text as a value.
    ///
    /// The exception's own `args[0]` is minted from this, so an unpaired
    /// surrogate in a message that names a user string has to survive here;
    /// `message_text` is for the diagnostic streams only.
    pub fn message_wtf8(&self) -> Wtf8Buf {
        if self.has_display_message() || self.exc_object.is_null() {
            return self.display_message_buf();
        }
        unsafe { crate::display::py_str_wtf8(self.exc_object) }
            .unwrap_or_else(|_| Wtf8Buf::from_string("<unprintable>".to_string()))
    }

    pub fn render_exception(&self) -> String {
        crate::display::wtf8_display_string(self.render_exception_wtf8(), "<unprintable>")
    }

    /// [`render_exception`](Self::render_exception) as the report's own WTF-8,
    /// for a writer assembling a buffer whose sink spends the encode.
    pub(crate) fn render_exception_wtf8(&self) -> Wtf8Buf {
        let name = exc_object_class_name(self.exc_object)
            .or_else(|| {
                if self.w_type.is_null() || unsafe { pyre_object::is_none(self.w_type) } {
                    None
                } else {
                    Some(unsafe { pyre_object::w_type_get_name(self.w_type) }.to_string())
                }
            })
            .unwrap_or_else(|| exc_kind_name(self.to_exc_kind()).to_string());
        let message = self.message_wtf8();
        if message.is_empty() {
            return Wtf8Buf::from_string(name);
        }
        crate::display::wtf8_format!(name, ": ", message)
    }
}

/// `error.py wrap_pos` — a zero position means "unknown" and reads back as
/// `None`; every other value keeps its integer.  `lineno` deliberately skips
/// this: a location-less parser error still reports line 0.
fn wrap_pos(num: i64) -> PyObjectRef {
    if num == 0 {
        pyre_object::w_none()
    } else {
        pyre_object::w_int_new(num)
    }
}

fn unraisable_hook_args_type() -> PyObjectRef {
    static TYPE: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    TYPE.get_or_init(|| {
        crate::_structseq::make_struct_seq(
            "sys.UnraisableHookArgs",
            &[
                "exc_type",
                "exc_value",
                "exc_traceback",
                "err_msg",
                "object",
            ],
        )
    })
}

/// Resolve an exception instance's actual Python class name for display.
///
/// `crate::typedef::type` trusts the instance's `w_class`, which the
/// `__new__` wrapper sets to the exact class that was raised — including
/// user subclasses (`class MyErr(ValueError)`) and class-only builtins
/// such as `KeyboardInterrupt` whose `ExcKind` tag degrades to
/// `BaseException`.  Reading the class name here makes the printed header
/// match `type(e).__name__` instead of the coarse kind tag.  Returns
/// `None` for a null pointer or non-exception object so callers fall back
/// to the kind-based name.
fn exc_object_class_name(exc: PyObjectRef) -> Option<String> {
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return None;
    }
    crate::typedef::r#type(exc).map(|tp| unsafe {
        let w_type = tp.as_ptr();
        if !pyre_object::w_type_is_cpython_heaptype(w_type) {
            return pyre_object::w_type_get_name(w_type).to_string();
        }
        let qualname = pyre_object::w_type_get_qualname(w_type).to_string();
        let module = crate::baseobjspace::lookup_in_type(w_type, "__module__")
            .filter(|value| pyre_object::is_str(*value))
            .and_then(|value| {
                crate::baseobjspace::str_utf8_w(value)
                    .ok()
                    .map(str::to_string)
            });
        match module.as_deref() {
            Some(module) if module != "builtins" && module != "__main__" => {
                format!("{module}.{qualname}")
            }
            _ => qualname,
        }
    })
}

impl std::fmt::Display for PyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.render_exception())
    }
}

pub fn write_exception<W: Write>(
    writer: &mut W,
    err: &PyError,
    include_traceback: bool,
) -> std::io::Result<()> {
    if !include_traceback {
        writer.write_all(err.render_exception_wtf8().as_bytes())?;
        return writer.write_all(b"\n");
    }
    if !err.exc_object.is_null() && unsafe { pyre_object::is_exception(err.exc_object) } {
        // The instance carries the whole report: the cause/context chain, the
        // group tree, the notes and the suggestion suffix all hang off it, so
        // render it through the same structured walk `_PyErr_Display` uses
        // rather than re-deriving a flat header here.
        let _roots = pyre_object::gc_roots::push_roots();
        let exc_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(err.exc_object);
        let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
        let mut context = ExceptionPrintContext::new(exc);
        return write_exception_object_recursive(writer, exc, &mut context);
    }
    // A Rust-side error that never materialised an instance carries no chain,
    // no group and no frame list — only the header.
    writeln!(writer, "Traceback (most recent call last):")?;
    write_traceback_chain(writer, err)?;
    writer.write_all(err.render_exception_wtf8().as_bytes())?;
    writer.write_all(b"\n")
}

/// `pythonrun.c _PyErr_Display`'s first attempt, which runs before the printer
/// [`write_exception_from_parts`] is.  The stdlib renderer is where a report
/// gains what only Python-level code produces: `_find_keyword_typos` rewrites a
/// bare `invalid syntax` into one naming the keyword the author meant, reading
/// the `_metadata` the parser hangs on the exception, and nothing in this file
/// offers that.
///
/// `false` says the module was not reachable, the attribute was absent or not
/// callable, or the call raised -- upstream's `fallback:` label, where the
/// pending failure is dropped and the caller prints the exception itself.  A
/// shutdown that has already taken `traceback` apart arrives as that same
/// `false`, which is what keeps a late report printable at all.
///
/// The traceback is installed on the exception first because that is the only
/// place the stdlib renderer looks for it; [`write_exception_from_parts`]
/// performs the same install, and doing it twice writes the slot once.
pub fn display_through_traceback_module(exc_value: PyObjectRef, exc_tb: PyObjectRef) -> bool {
    if exc_value.is_null() || !unsafe { pyre_object::is_exception(exc_value) } {
        return false;
    }
    // A bounded heap has one `MemoryError` to give and aborts on the breach
    // after it.  Reaching the stdlib renderer imports `traceback`, whose module
    // body is a nested dispatch loop whose first safepoint collects over a heap
    // already at its limit -- so the report this route exists to write is
    // exactly what that abort takes away.  Declining leaves the built-in
    // renderer below, which reaches stderr without running a module body.
    if majit_gc::gc_max_heap_already_raised() {
        return false;
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let exc_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(exc_value);
    if !exc_tb.is_null()
        && !unsafe { pyre_object::is_none(exc_tb) }
        && unsafe { crate::pytraceback::is_pytraceback(exc_tb) }
        && unsafe {
            pyre_object::interp_exceptions::w_exception_get_traceback(
                pyre_object::gc_roots::shadow_stack_get(exc_slot),
            )
            .is_null()
        }
    {
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_traceback(
                pyre_object::gc_roots::shadow_stack_get(exc_slot),
                exc_tb,
            );
        }
    }
    let ec = crate::call::getexecutioncontext();
    let Ok(module) = crate::importing::importhook(
        rustpython_wtf8::Wtf8::new("traceback"),
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        0,
        ec,
    ) else {
        return false;
    };
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(module);
    let Ok(printer) = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        "_print_exception_bltin",
    ) else {
        return false;
    };
    if printer.is_null() || !crate::baseobjspace::callable_w(printer) {
        return false;
    }
    let printer_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(printer);
    let result = crate::baseobjspace::call_function(
        pyre_object::gc_roots::shadow_stack_get(printer_slot),
        &[pyre_object::gc_roots::shadow_stack_get(exc_slot)],
    );
    if result.is_null() {
        let _ = crate::call::take_call_error();
        return false;
    }
    true
}

/// CPython 3.14 `_PyErr_Display(file, exc_type, exc_value, exc_tb)` shape used
/// by `_thread._excepthook`.
///
/// `Python/pythonrun.c:_PyErr_Display` installs a supplied traceback only when
/// the exception has none, then displays the traceback resident on the
/// exception.  An existing traceback therefore wins and the one-time
/// installation remains observable through `exc_value.__traceback__`.
pub fn write_exception_from_parts<W: Write>(
    writer: &mut W,
    exc_value: PyObjectRef,
    exc_tb: PyObjectRef,
) -> std::io::Result<()> {
    let _roots = pyre_object::gc_roots::push_roots();
    let exc_value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(exc_value);
    let exc_tb_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(exc_tb);
    let exc_value = pyre_object::gc_roots::shadow_stack_get(exc_value_slot);
    if exc_value.is_null() || !unsafe { pyre_object::is_exception(exc_value) } {
        if !exc_value.is_null() && unsafe { pyre_object::is_none(exc_value) } {
            return writeln!(writer, "NoneType: None");
        }
        let value_type = crate::typedef::r#type(exc_value)
            .map(|tp| unsafe { pyre_object::w_type_get_name(tp.as_ptr()).to_string() })
            .unwrap_or_else(|| "NULL".to_string());
        return writeln!(
            writer,
            "TypeError: print_exception(): Exception expected for value, {value_type} found"
        );
    }

    let exc_tb = pyre_object::gc_roots::shadow_stack_get(exc_tb_slot);
    if !exc_tb.is_null()
        && !unsafe { pyre_object::is_none(exc_tb) }
        && unsafe { crate::pytraceback::is_pytraceback(exc_tb) }
        && unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(exc_value).is_null() }
    {
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_traceback(exc_value, exc_tb);
        }
    }

    // `lib-python/3/traceback.py TracebackException.__init__`
    // records every visited exception before following cause/context edges.
    // Keep the same set for the complete structured render so cycles terminate
    // and ExceptionGroup leaves share the chain census.
    let exc_value = pyre_object::gc_roots::shadow_stack_get(exc_value_slot);
    let mut context = ExceptionPrintContext::new(exc_value);
    write_exception_object_recursive(writer, exc_value, &mut context)
}

/// Render an uncaught compile-time SyntaxError as the top-level banner
/// (`print_error_text`): the `File "…", line N` header, the offending
/// source line, a caret under the offending column span, and
/// `<Class>: <msg>` (the raw `msg`, not `str(e)`'s `(file, line N)`
/// suffix — the header already carries the location).  There is no
/// traceback frame chain: a malformed source never began executing.
/// Falls back to the plain header when the instance lacks structured
/// location (a location-less codegen error).
pub fn write_syntax_error<W: Write>(writer: &mut W, err: &PyError) -> std::io::Result<()> {
    let exc = err.exc_object;
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        writer.write_all(err.render_exception_wtf8().as_bytes())?;
        return writer.write_all(b"\n");
    }
    write_syntax_error_object(writer, exc)
}

fn write_syntax_error_object<W: Write>(writer: &mut W, exc: PyObjectRef) -> std::io::Result<()> {
    let _roots = pyre_object::gc_roots::push_roots();
    let exc_slot = pyre_object::gc_roots::shadow_stack_len();
    let exc = pyre_object::gc_roots::pin_root(exc);
    let attr = |name: &str| {
        let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
        crate::baseobjspace::syntax_error_attr(exc, name)
    };
    // `filename` is the compiled file's own name, so it carries whatever
    // surrogate escapes the filesystem encoding produced, and `msg` may quote
    // it.  Both go out as the report's own WTF-8; the sink spends the
    // `backslashreplace` encode.
    let wtf8_of = |w: PyObjectRef| -> Option<Wtf8Buf> {
        (!w.is_null() && unsafe { pyre_object::is_str(w) })
            .then(|| unsafe { pyre_object::w_str_get_wtf8(w) }.to_owned())
    };
    // Code points in a WTF-8 byte run: every byte that is not a continuation
    // byte starts one, and a lone surrogate is a single three-byte run, so it
    // counts as the one column the caret arithmetic treats it as.
    let code_point_len = |bytes: &[u8]| bytes.iter().filter(|b| (*b & 0xC0) != 0x80).count();
    let int_of = |w: PyObjectRef| -> Option<i64> {
        (!w.is_null() && unsafe { pyre_object::is_int(w) })
            .then(|| unsafe { pyre_object::intobject::w_int_get_value(w) })
    };
    // `traceback.py _format_syntax_error`: the header goes out whenever the
    // instance carries a line number, with the filename rendered through
    // `str()` -- a `bytes` filename shows as `b'…'` -- and `<string>` standing
    // in for one that is falsy. Without a line number there is no header and
    // the filename becomes a ` (…)` suffix on the message line instead.
    //
    // Each read below takes the attribute off the pinned exception again
    // rather than holding one reference across a call that re-enters Python.
    let filename_is_set = {
        let w = attr("filename");
        !w.is_null() && !unsafe { pyre_object::is_none(w) }
    };
    let filename_is_truthy =
        filename_is_set && crate::baseobjspace::is_true(attr("filename")).unwrap_or(true);
    let filename_text = filename_is_set.then(|| {
        unsafe { crate::py_str_wtf8(attr("filename")) }
            .unwrap_or_else(|_| Wtf8Buf::from_string(String::new()))
    });
    let lineno = int_of(attr("lineno"));
    let mut filename_suffix = None;
    if let Some(lineno) = lineno {
        let shown = match (&filename_text, filename_is_truthy) {
            (Some(text), true) => text.as_bytes(),
            _ => b"<string>".as_slice(),
        };
        writer.write_all(b"  File \"")?;
        writer.write_all(shown)?;
        writeln!(writer, "\", line {lineno}")?;
    } else {
        filename_suffix = filename_text;
    }
    if let Some(text) = wtf8_of(attr("text")) {
        // A constructor-supplied `text` is any str, so it may hold a lone
        // surrogate even though a compiled line cannot.  It goes out as the
        // report's own WTF-8 like the filename above; the trims below cut only
        // ASCII, which leaves the run well-formed.
        let bytes = text.as_bytes();
        let end = bytes.len()
            - bytes
                .iter()
                .rev()
                .take_while(|b| **b == b'\n' || **b == b'\r')
                .count();
        let raw = &bytes[..end];
        let lead_bytes = raw.iter().take_while(|b| b.is_ascii_whitespace()).count();
        let shown = &raw[lead_bytes..];
        let lead = code_point_len(&raw[..lead_bytes]);
        writer.write_all(b"    ")?;
        writer.write_all(shown)?;
        writer.write_all(b"\n")?;
        if let Some(offset) = int_of(attr("offset")).filter(|&offset| offset > 0) {
            // Traceback formatting treats the stored columns as display
            // columns and clamps malformed constructor-supplied offsets to
            // the displayed line.  Offsets at or below zero suppress the
            // marker entirely.
            let displayed_len = code_point_len(raw);
            let display_col = |offset: i64| ((offset as usize) - 1).min(displayed_len);
            let start_col = display_col(offset);
            let end_col = int_of(attr("end_offset"))
                .filter(|&e| e >= offset)
                .map_or(start_col, display_col);
            let caret_start = start_col.saturating_sub(lead);
            let caret_len = end_col.saturating_sub(start_col).max(1);
            let mut caret = String::from("    ");
            caret.push_str(&" ".repeat(caret_start));
            caret.push_str(&"^".repeat(caret_len));
            writeln!(writer, "{caret}")?;
        }
    }
    let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
    let name = exc_object_class_name(exc).unwrap_or_else(|| "SyntaxError".to_string());
    match wtf8_of(attr("msg")) {
        Some(msg) if !msg.is_empty() => {
            write!(writer, "{name}: ")?;
            writer.write_all(msg.as_bytes())?;
        }
        _ => write!(writer, "{name}")?,
    }
    if let Some(suffix) = filename_suffix {
        writer.write_all(b" (")?;
        writer.write_all(suffix.as_bytes())?;
        writer.write_all(b")")?;
    }
    writer.write_all(b"\n")
}

/// Emit `write_syntax_error` through the mediated stderr seam, mirroring
/// `eprint_exception`.
pub fn eprint_syntax_error(err: &PyError) {
    let mut buf: Vec<u8> = Vec::new();
    let _ = write_syntax_error(&mut buf, err);
    emit_report_to_host_stderr(&buf);
}

/// `lib-python/3/traceback.py _ExceptionPrintContext` and
/// `:1468-1565 TracebackException.format` state.  CPython shares one visited
/// set across the complete cause/context/ExceptionGroup tree.
struct ExceptionPrintContext {
    seen: HashSet<usize>,
    exception_group_depth: usize,
    need_close: bool,
}

impl ExceptionPrintContext {
    fn new(exc: PyObjectRef) -> Self {
        let mut seen = HashSet::new();
        seen.insert(pyre_object::gc_hook::gc_identity_hash(exc as usize));
        Self {
            seen,
            exception_group_depth: 0,
            need_close: false,
        }
    }
}

fn write_exception_object_recursive<W: Write>(
    writer: &mut W,
    exc: PyObjectRef,
    context: &mut ExceptionPrintContext,
) -> std::io::Result<()> {
    // `TracebackException.__init__` uses an explicit queue for chain capture.
    // Keep the same non-recursive shape so an application-created chain cannot
    // consume the native Rust stack.  Store shadow-stack slot indices, not raw
    // GC pointers: formatting calls arbitrary `__str__` / attribute code.
    let _roots = pyre_object::gc_roots::push_roots();
    let first_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(exc);
    let mut chain: Vec<(usize, Option<&'static str>)> = Vec::new();
    let mut current_slot = first_slot;
    loop {
        let current = pyre_object::gc_roots::shadow_stack_get(current_slot);
        let cause = unsafe { pyre_object::interp_exceptions::w_exception_get_cause(current) };
        let cause_slot = pin_unseen_exception(cause, context);
        let (older_slot, banner) = if let Some(cause_slot) = cause_slot {
            (
                Some(cause_slot),
                Some("\nThe above exception was the direct cause of the following exception:\n\n"),
            )
        } else {
            let current = pyre_object::gc_roots::shadow_stack_get(current_slot);
            let suppress = unsafe {
                pyre_object::interp_exceptions::w_exception_get_suppress_context(current)
            };
            let previous =
                unsafe { pyre_object::interp_exceptions::w_exception_get_context(current) };
            if !suppress {
                if let Some(previous_slot) = pin_unseen_exception(previous, context) {
                    (
                        Some(previous_slot),
                        Some(
                            "\nDuring handling of the above exception, another exception occurred:\n\n",
                        ),
                    )
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            }
        };
        chain.push((current_slot, banner));
        match older_slot {
            Some(slot) => current_slot = slot,
            None => break,
        }
    }

    for (slot, banner) in chain.into_iter().rev() {
        if let Some(banner) = banner {
            emit_exception_text(writer, banner.as_bytes(), context, b'|')?;
        }
        if let Some(exceptions) = exception_group_item_slots(slot) {
            write_exception_group(writer, slot, exceptions, context)?;
        } else {
            write_plain_exception_object(writer, slot, context)?;
        }
    }
    Ok(())
}

fn pin_unseen_exception(
    candidate: PyObjectRef,
    context: &mut ExceptionPrintContext,
) -> Option<usize> {
    if candidate.is_null() || unsafe { pyre_object::is_none(candidate) } {
        return None;
    }
    let slot = pyre_object::gc_roots::shadow_stack_len();
    let candidate = pyre_object::gc_roots::pin_root(candidate);
    let identity = pyre_object::gc_hook::gc_identity_hash(candidate as usize);
    context.seen.insert(identity).then_some(slot)
}

fn write_plain_exception_object<W: Write>(
    writer: &mut W,
    exc_slot: usize,
    context: &ExceptionPrintContext,
) -> std::io::Result<()> {
    let mut rendered = Vec::new();
    let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
    let tb = unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(exc) };
    if !tb.is_null() {
        writeln!(rendered, "Traceback (most recent call last):")?;
        write_traceback_chain_from_tb(&mut rendered, tb)?;
    }
    let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
    if unsafe { pyre_object::interp_exceptions::w_exception_get_kind(exc) } == ExcKind::SyntaxError
    {
        write_syntax_error_object(&mut rendered, exc)?;
    } else {
        rendered.write_all(render_rooted_exc_object_wtf8(exc_slot).as_bytes())?;
        rendered.write_all(b"\n")?;
    }
    let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
    write_exception_notes(&mut rendered, exc)?;
    emit_exception_text(writer, &rendered, context, b'|')
}

fn exception_group_item_slots(exc_slot: usize) -> Option<Vec<usize>> {
    let base_group = crate::builtins::lookup_exc_class("BaseExceptionGroup")?;
    let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
    if !crate::baseobjspace::isinstance(exc, base_group).ok()? {
        return None;
    }
    let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
    let exceptions = crate::baseobjspace::getattr_str(exc, "exceptions").ok()?;
    let tuple_slot = pyre_object::gc_roots::shadow_stack_len();
    let exceptions = pyre_object::gc_roots::pin_root(exceptions);
    if !unsafe { pyre_object::is_tuple(exceptions) } {
        return None;
    }
    let len = unsafe { pyre_object::w_tuple_len(exceptions) };
    let mut slots = Vec::with_capacity(len);
    for index in 0..len {
        let exceptions = pyre_object::gc_roots::shadow_stack_get(tuple_slot);
        let child = unsafe { pyre_object::w_tuple_getitem(exceptions, index as i64) }?;
        let child_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(child);
        slots.push(child_slot);
    }
    Some(slots)
}

fn write_exception_group<W: Write>(
    writer: &mut W,
    exc_slot: usize,
    exceptions: Vec<usize>,
    context: &mut ExceptionPrintContext,
) -> std::io::Result<()> {
    const MAX_GROUP_WIDTH: usize = 15;
    const MAX_GROUP_DEPTH: usize = 10;

    if context.exception_group_depth > MAX_GROUP_DEPTH {
        return emit_exception_text(writer, b"... (max_group_depth is 10)\n", context, b'|');
    }
    let is_toplevel = context.exception_group_depth == 0;
    if is_toplevel {
        context.exception_group_depth += 1;
    }

    let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
    let tb = unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(exc) };
    if !tb.is_null() {
        emit_exception_text(
            writer,
            b"Exception Group Traceback (most recent call last):\n",
            context,
            if is_toplevel { b'+' } else { b'|' },
        )?;
        let mut traceback = Vec::new();
        write_traceback_chain_from_tb(&mut traceback, tb)?;
        emit_exception_text(writer, &traceback, context, b'|')?;
    }

    let mut header = Vec::new();
    header.write_all(render_rooted_exc_object_wtf8(exc_slot).as_bytes())?;
    header.write_all(b"\n")?;
    let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
    write_exception_notes(&mut header, exc)?;
    emit_exception_text(writer, &header, context, b'|')?;

    let shown = exceptions.len().min(MAX_GROUP_WIDTH);
    let count = if exceptions.len() > MAX_GROUP_WIDTH {
        shown + 1
    } else {
        shown
    };
    for child_slot in exceptions.iter().take(shown) {
        let child = pyre_object::gc_roots::shadow_stack_get(*child_slot);
        context
            .seen
            .insert(pyre_object::gc_hook::gc_identity_hash(child as usize));
    }
    context.need_close = false;
    for index in 0..count {
        let last = index + 1 == count;
        if last {
            context.need_close = true;
        }
        let truncated = index >= MAX_GROUP_WIDTH;
        let title = if truncated {
            "...".to_string()
        } else {
            (index + 1).to_string()
        };
        writeln!(
            writer,
            "{}{}+---------------- {title} ----------------",
            " ".repeat(2 * context.exception_group_depth),
            if index == 0 { "+-" } else { "  " },
        )?;
        context.exception_group_depth += 1;
        if truncated {
            let remaining = exceptions.len() - MAX_GROUP_WIDTH;
            let suffix = if remaining == 1 { "" } else { "s" };
            emit_exception_text(
                writer,
                format!("and {remaining} more exception{suffix}\n").as_bytes(),
                context,
                b'|',
            )?;
        } else {
            let child = pyre_object::gc_roots::shadow_stack_get(exceptions[index]);
            write_exception_object_recursive(writer, child, context)?;
        }
        if last && context.need_close {
            writeln!(
                writer,
                "{}+------------------------------------",
                " ".repeat(2 * context.exception_group_depth)
            )?;
            context.need_close = false;
        }
        context.exception_group_depth -= 1;
    }

    if is_toplevel {
        debug_assert_eq!(context.exception_group_depth, 1);
        context.exception_group_depth = 0;
    }
    Ok(())
}

fn emit_exception_text<W: Write>(
    writer: &mut W,
    bytes: &[u8],
    context: &ExceptionPrintContext,
    margin: u8,
) -> std::io::Result<()> {
    let mut prefix = vec![b' '; 2 * context.exception_group_depth];
    if context.exception_group_depth != 0 {
        prefix.extend_from_slice(&[margin, b' ']);
    }
    for line in bytes.split_inclusive(|byte| *byte == b'\n') {
        writer.write_all(&prefix)?;
        writer.write_all(line)?;
    }
    Ok(())
}

/// `Lib/traceback.py:171-200 TracebackException._format_*` parity —
/// walk `__cause__` (or `__context__` when `__suppress_context__` is
/// False) chains and emit each older exception with the appropriate
/// bridging banner before the current one.
#[allow(dead_code)]
fn write_chained_context<W: Write>(writer: &mut W, exc: PyObjectRef) -> std::io::Result<()> {
    let mut seen = HashSet::new();
    seen.insert(exc as usize);
    write_chained_context_inner(writer, exc, &mut seen)
}

#[allow(dead_code)]
fn write_chained_context_inner<W: Write>(
    writer: &mut W,
    exc: PyObjectRef,
    seen: &mut HashSet<usize>,
) -> std::io::Result<()> {
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return Ok(());
    }
    let cause = unsafe { pyre_object::interp_exceptions::w_exception_get_cause(exc) };
    let context = unsafe { pyre_object::interp_exceptions::w_exception_get_context(exc) };
    let suppress = unsafe { pyre_object::interp_exceptions::w_exception_get_suppress_context(exc) };

    // `traceback.py:184-191` — `__cause__` wins over `__context__`;
    // `__suppress_context__` (set by `raise X from None`) hides
    // `__context__` only when no explicit cause was attached.
    let (mut older, banner) = if !cause.is_null()
        && !unsafe { pyre_object::is_none(cause) }
        && seen.insert(cause as usize)
    {
        (
            cause,
            "\nThe above exception was the direct cause of the following exception:\n",
        )
    } else if !suppress
        && !context.is_null()
        && !unsafe { pyre_object::is_none(context) }
        && seen.insert(context as usize)
    {
        (
            context,
            "\nDuring handling of the above exception, another exception occurred:\n",
        )
    } else {
        return Ok(());
    };

    pyre_object::with_roots!(older => write_chained_context_inner(writer, older, seen))?;
    write_single_exception(writer, older)?;
    writeln!(writer, "{}", banner)?;
    Ok(())
}

/// Print one W_BaseException as `Traceback ...\n<frames>\n<header>`.
/// Used by `write_chained_context` when recursing through chained
/// __cause__ / __context__ predecessors.
#[allow(dead_code)]
fn write_single_exception<W: Write>(writer: &mut W, mut exc: PyObjectRef) -> std::io::Result<()> {
    writeln!(writer, "Traceback (most recent call last):")?;
    pyre_object::with_roots!(exc => write_traceback_chain_from_exc(writer, exc))?;
    let rendered = pyre_object::with_roots!(exc => render_exc_object_wtf8(exc));
    writer.write_all(rendered.as_bytes())?;
    writer.write_all(b"\n")?;
    write_exception_notes(writer, exc)
}

/// `Lib/traceback.py:_format_final_exc_line` notes block — each entry of
/// `e.__notes__` is printed on its own line after the exception
/// header.  Notes are stored in the exception's instance dict as a
/// list of strings by `BaseException.add_note` (PEP 678).
fn write_exception_notes<W: Write>(writer: &mut W, exc: PyObjectRef) -> std::io::Result<()> {
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return Ok(());
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let exc_slot = pyre_object::gc_roots::shadow_stack_len();
    let exc = pyre_object::gc_roots::pin_root(exc);
    let notes = match crate::baseobjspace::getattr_str(exc, "__notes__") {
        Ok(v) if !v.is_null() => v,
        Ok(_) => return Ok(()),
        Err(err) if err.kind == PyErrorKind::AttributeError => return Ok(()),
        Err(mut err) => {
            // `traceback.py TracebackException.__init__` preserves
            // failures raised by a custom `__notes__` descriptor as a
            // synthetic note instead of silently discarding the diagnostic.
            let err_obj = err.to_exc_object();
            let _ = pyre_object::gc_roots::pin_root(err_obj);
            let err_obj = pyre_object::gc_roots::shadow_stack_get(
                pyre_object::gc_roots::shadow_stack_len() - 1,
            );
            let rendered = unsafe { crate::display::py_repr_wtf8(err_obj) }
                .unwrap_or_else(|_| Wtf8Buf::from_string("<exception repr() failed>".to_string()));
            writer.write_all(b"Ignored error getting __notes__: ")?;
            writer.write_all(rendered.as_bytes())?;
            return writer.write_all(b"\n");
        }
    };
    let notes_slot = pyre_object::gc_roots::shadow_stack_len();
    let notes = pyre_object::gc_roots::pin_root(notes);
    let is_list = unsafe { crate::baseobjspace::isinstance_list_w(notes) };
    let is_tuple = unsafe {
        pyre_object::is_tuple(notes)
            || crate::typedef::gettypefor(&pyre_object::TUPLE_TYPE).is_some_and(|tuple_type| {
                crate::baseobjspace::isinstance_w(notes, tuple_type.as_ptr())
            })
    };
    let is_sequence = is_list || is_tuple || notes_is_abc_sequence(notes_slot);
    if is_sequence
        && !unsafe { crate::baseobjspace::isinstance_str_w(notes) }
        && !unsafe { crate::baseobjspace::isinstance_bytes_w(notes) }
    {
        let mut item_slots = Vec::new();
        if is_list || is_tuple {
            let len = if is_list {
                unsafe { pyre_object::w_list_len(notes) }
            } else {
                unsafe { pyre_object::w_tuple_len(notes) }
            } as i64;
            for i in 0..len {
                let notes = pyre_object::gc_roots::shadow_stack_get(notes_slot);
                let item = if is_list {
                    unsafe { pyre_object::w_list_getitem(notes, i) }
                } else {
                    unsafe { pyre_object::w_tuple_getitem(notes, i) }
                }
                .unwrap_or(pyre_object::PY_NULL);
                if !item.is_null() {
                    let item_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = pyre_object::gc_roots::pin_root(item);
                    item_slots.push(item_slot);
                }
            }
        } else {
            let notes = pyre_object::gc_roots::shadow_stack_get(notes_slot);
            if let Ok(items) = crate::builtins::collect_iterable(notes) {
                for item in items {
                    let item_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = pyre_object::gc_roots::pin_root(item);
                    item_slots.push(item_slot);
                }
            }
        }
        for item_slot in item_slots {
            let item = pyre_object::gc_roots::shadow_stack_get(item_slot);
            let s = unsafe { crate::display::py_str_wtf8(item) }
                .unwrap_or_else(|_| Wtf8Buf::from_string("<unprintable note>".to_string()));
            for line in s.as_bytes().split(|byte| *byte == b'\n') {
                writer.write_all(line)?;
                writer.write_all(b"\n")?;
            }
        }
        return Ok(());
    }
    let notes = pyre_object::gc_roots::shadow_stack_get(notes_slot);
    let rendered = unsafe { crate::display::py_repr_wtf8(notes) }
        .unwrap_or_else(|_| Wtf8Buf::from_string("<exception repr() failed>".to_string()));
    writer.write_all(rendered.as_bytes())?;
    writer.write_all(b"\n")
}

fn notes_is_abc_sequence(notes_slot: usize) -> bool {
    let module = crate::importing::get_sys_module("collections.abc").or_else(|| {
        crate::importing::importhook(
            rustpython_wtf8::Wtf8::new("collections.abc"),
            pyre_object::w_none(),
            pyre_object::w_none(),
            0,
            crate::call::take_last_exec_ctx(),
        )
        .ok()?;
        crate::importing::get_sys_module("collections.abc")
    });
    let Some(module) = module else {
        return false;
    };
    let Ok(sequence) = crate::baseobjspace::getattr_str(module, "Sequence") else {
        return false;
    };
    let sequence_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(sequence);
    let notes = pyre_object::gc_roots::shadow_stack_get(notes_slot);
    let sequence = pyre_object::gc_roots::shadow_stack_get(sequence_slot);
    crate::baseobjspace::isinstance(notes, sequence).unwrap_or(false)
}

fn render_rooted_exc_object_wtf8(exc_slot: usize) -> Wtf8Buf {
    let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return Wtf8Buf::from_string("<no exception>".to_string());
    }
    let name = exc_object_class_name(exc).unwrap_or_else(|| {
        let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
        let kind = unsafe { pyre_object::interp_exceptions::w_exception_get_kind(exc) };
        exc_kind_name(kind).to_string()
    });
    let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
    let msg = unsafe {
        crate::display::py_str_wtf8(exc)
            .unwrap_or_else(|_| Wtf8Buf::from_string("<exception str() failed>".to_string()))
    };
    let mut rendered = Wtf8Buf::from_string(name);
    if !msg.as_bytes().is_empty() {
        rendered.push_str(": ");
        rendered.push_wtf8(&msg);
    }
    if let Some(suffix) = exception_suggestion(exc_slot) {
        rendered.push_str(&suffix);
    }
    rendered
}

#[allow(dead_code)]
fn render_exc_object_wtf8(exc: PyObjectRef) -> Wtf8Buf {
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return Wtf8Buf::from_string("<no exception>".to_string());
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let exc_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(exc);
    render_rooted_exc_object_wtf8(exc_slot)
}

pub const MAX_SUGGESTION_CANDIDATES: usize = 750;
const MAX_SUGGESTION_STRING_SIZE: usize = 40;
const SUGGESTION_MOVE_COST: usize = 2;
const SUGGESTION_CASE_COST: usize = 1;

fn dict_string_keys(dict: PyObjectRef, names: &mut Vec<String>) {
    if dict.is_null() || !unsafe { pyre_object::is_dict(dict) } {
        return;
    }
    for (key, _) in unsafe { pyre_object::w_dict_items(dict) } {
        if unsafe { pyre_object::is_str(key) } {
            // A name carrying a lone surrogate has no `&str` view, and the
            // distance below is computed over `char`s.  Drop it from the
            // candidate set rather than reading it as UTF-8.
            if let Some(name) = unsafe { pyre_object::w_str_get_value_opt(key) } {
                names.push(name.to_string());
            }
        }
    }
}

fn suggestion_distance(a: &str, b: &str, max_cost: usize) -> usize {
    // `levenshtein_distance` reads each name through `PyUnicode_AsUTF8AndSize`
    // and indexes the buffer it gets, so every length and every position here
    // is a UTF-8 byte.  Measuring code points instead moves the ratio cutoff
    // and the distance itself for any name outside ASCII, which is a different
    // ranking rather than the same one in other units.
    let mut a: &[u8] = a.as_bytes();
    let mut b: &[u8] = b.as_bytes();
    if a == b {
        return 0;
    }
    // Trim away common affixes.
    while a.first() == b.first() && !a.is_empty() {
        a = &a[1..];
        b = &b[1..];
    }
    while a.last() == b.last() && !a.is_empty() {
        a = &a[..a.len() - 1];
        b = &b[..b.len() - 1];
    }
    if a.is_empty() || b.is_empty() {
        return SUGGESTION_MOVE_COST * (a.len() + b.len());
    }
    if a.len() > MAX_SUGGESTION_STRING_SIZE || b.len() > MAX_SUGGESTION_STRING_SIZE {
        return max_cost + 1;
    }
    if b.len() < a.len() {
        std::mem::swap(&mut a, &mut b);
    }
    if (b.len() - a.len()) * SUGGESTION_MOVE_COST > max_cost {
        return max_cost + 1;
    }
    let mut row: Vec<usize> = (1..=a.len())
        .map(|index| index * SUGGESTION_MOVE_COST)
        .collect();
    let mut result = 0;
    for (bindex, bbyte) in b.iter().enumerate() {
        let mut distance = bindex * SUGGESTION_MOVE_COST;
        result = distance;
        let mut minimum = usize::MAX;
        for (index, abyte) in a.iter().enumerate() {
            // `_substitution_cost` folds case over ASCII alone, so a pair that
            // differs only outside that range costs a full move.
            let case_equal = abyte.eq_ignore_ascii_case(bbyte);
            let substitute = distance
                + if abyte == bbyte {
                    0
                } else if case_equal {
                    SUGGESTION_CASE_COST
                } else {
                    SUGGESTION_MOVE_COST
                };
            distance = row[index];
            result = (result.min(distance) + SUGGESTION_MOVE_COST).min(substitute);
            row[index] = result;
            minimum = minimum.min(result);
        }
        if minimum > max_cost {
            return max_cost + 1;
        }
    }
    result
}

pub fn best_suggestion(candidates: &[String], wrong_name: &str) -> Option<String> {
    if candidates.len() >= MAX_SUGGESTION_CANDIDATES {
        return None;
    }
    // The ratio cutoff below is `_Py_CalculateSuggestions`', which sizes each
    // name by the UTF-8 buffer `PyUnicode_AsUTF8AndSize` returned for it.
    let wrong_len = wrong_name.len();
    // No candidate has been measured yet, so nothing caps the first one but
    // its own ratio test below.  A name longer than `MAX_SUGGESTION_STRING_SIZE`
    // is not rejected here: `suggestion_distance` applies that limit to what is
    // left after the common affixes are trimmed, so a long name still matches a
    // candidate it mostly shares.
    let mut best_distance = usize::MAX;
    let mut suggestion = None;
    for candidate in candidates {
        if candidate == wrong_name {
            continue;
        }
        let candidate_len = candidate.len();
        // No more than 1/3 of the involved characters should need changed.
        let mut max_distance = (candidate_len + wrong_len + 3) * SUGGESTION_MOVE_COST / 6;
        // Don't take matches we've already beaten.
        max_distance = max_distance.min(best_distance.saturating_sub(1));
        let distance = suggestion_distance(wrong_name, candidate, max_distance);
        if distance > max_distance {
            continue;
        }
        if suggestion.is_none() || distance < best_distance {
            suggestion = Some(candidate.clone());
            best_distance = distance;
        }
    }
    suggestion
}

fn traceback_last_frame(tb: PyObjectRef) -> Option<*mut crate::pyframe::PyFrame> {
    if tb.is_null() {
        return None;
    }
    let tb_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(tb);
    let mut current = pyre_object::gc_roots::shadow_stack_get(tb_slot);
    let mut last = pyre_object::PY_NULL;
    while !current.is_null() && unsafe { crate::pytraceback::is_pytraceback(current) } {
        last = current;
        current = unsafe { crate::pytraceback::w_pytraceback_get_w_next(current) };
    }
    if last.is_null() {
        return None;
    }
    let frame = unsafe { crate::pytraceback::w_pytraceback_get_frame(last) };
    (!frame.is_null()).then_some(frame)
}

fn traceback_self_is(tb: PyObjectRef, obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let Some(frame) = traceback_last_frame(tb) else {
        return false;
    };
    let anchor = crate::eval::FrameAnchor::new(unsafe { &mut *frame });
    let Ok(locals) = unsafe { &mut *anchor.live() }.getdictscope() else {
        return false;
    };
    let locals_slot = pyre_object::gc_roots::shadow_stack_len();
    let locals = pyre_object::gc_roots::pin_root(locals);
    let Some(self_obj) = (unsafe { pyre_object::w_dict_getitem_str(locals, "self") }) else {
        return false;
    };
    let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    std::ptr::eq(self_obj, obj)
}

fn object_dir_strings(obj: PyObjectRef) -> Option<Vec<String>> {
    if obj.is_null() || unsafe { pyre_object::is_none(obj) } {
        return None;
    }
    let slot = pyre_object::gc_roots::shadow_stack_len();
    let obj = pyre_object::gc_roots::pin_root(obj);
    let names = crate::builtins::builtin_dir(&[obj]).ok()?;
    let names_slot = pyre_object::gc_roots::shadow_stack_len();
    let names = pyre_object::gc_roots::pin_root(names);
    let items = crate::builtins::collect_iterable(names).ok()?;
    Some(
        items
            .into_iter()
            .filter(|item| unsafe { pyre_object::is_str(*item) })
            .filter_map(|item| {
                unsafe { pyre_object::w_str_get_wtf8(item).as_str() }
                    .ok()
                    .map(str::to_owned)
            })
            .collect(),
    )
}

fn stdlib_module_name(name: &str) -> bool {
    let Some(sys) = crate::importing::get_interpreter_sys_module() else {
        return false;
    };
    let Ok(names) = crate::baseobjspace::getattr_str(sys, "stdlib_module_names") else {
        return false;
    };
    if !unsafe { pyre_object::is_set_or_frozenset(names) } {
        return false;
    }
    let names_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(names);
    let name = pyre_object::w_str_new_managed(name);
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(name);
    let names = pyre_object::gc_roots::shadow_stack_get(names_slot);
    let name = pyre_object::gc_roots::shadow_stack_get(name_slot);
    crate::baseobjspace::contains(names, name).unwrap_or(false)
}

/// `traceback.py:1096-1114,1601-1684` enhanced error messages.
fn exception_suggestion(exc_slot: usize) -> Option<String> {
    let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
    let kind = unsafe { pyre_object::interp_exceptions::w_exception_get_kind(exc) };
    let wrong = match kind {
        ExcKind::ImportError => unsafe {
            pyre_object::interp_exceptions::w_exception_get_import_name_from(exc)
        },
        ExcKind::NameError | ExcKind::AttributeError => unsafe {
            pyre_object::interp_exceptions::w_exception_get_name(exc)
        },
        _ => return None,
    };
    if wrong.is_null()
        || unsafe { pyre_object::is_none(wrong) }
        || !unsafe { pyre_object::is_str(wrong) }
    {
        return None;
    }
    // The distance below is computed over `char`s, so a name carrying a lone
    // surrogate has nothing to compare; answer no suggestion instead of
    // reading it as UTF-8.
    let wrong_name = unsafe { pyre_object::w_str_get_value_opt(wrong) }?.to_string();
    let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
    let tb = unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(exc) };
    let tb_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(tb);
    let suggestion = match kind {
        ExcKind::AttributeError => {
            let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
            let obj = unsafe { pyre_object::interp_exceptions::w_exception_get_attr_obj(exc) };
            let obj_slot = pyre_object::gc_roots::shadow_stack_len();
            let obj = pyre_object::gc_roots::pin_root(obj);
            object_dir_strings(obj).and_then(|mut names| {
                let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
                let tb = pyre_object::gc_roots::shadow_stack_get(tb_slot);
                if !wrong_name.starts_with('_') && !traceback_self_is(tb, obj) {
                    names.retain(|name| !name.starts_with('_'));
                }
                best_suggestion(&names, &wrong_name)
            })
        }
        ExcKind::ImportError => {
            let exc = pyre_object::gc_roots::shadow_stack_get(exc_slot);
            let module_name = unsafe { pyre_object::interp_exceptions::w_exception_get_name(exc) };
            let module_name =
                if module_name.is_null() || !unsafe { pyre_object::is_str(module_name) } {
                    None
                } else {
                    unsafe { pyre_object::w_str_get_value_opt(module_name) }
                };
            match module_name {
                None => None,
                Some(module_name) => {
                    let module_name = module_name.to_string();
                    crate::importing::get_sys_module(&module_name)
                        .or_else(|| {
                            crate::importing::importhook(
                                rustpython_wtf8::Wtf8::new(module_name.as_str()),
                                pyre_object::w_none(),
                                pyre_object::w_none(),
                                0,
                                crate::call::take_last_exec_ctx(),
                            )
                            .ok()
                        })
                        .and_then(object_dir_strings)
                        .and_then(|mut names| {
                            if !wrong_name.starts_with('_') {
                                names.retain(|name| !name.starts_with('_'));
                            }
                            best_suggestion(&names, &wrong_name)
                        })
                }
            }
        }
        ExcKind::NameError => {
            let frame = traceback_last_frame(pyre_object::gc_roots::shadow_stack_get(tb_slot))?;
            let anchor = crate::eval::FrameAnchor::new(unsafe { &mut *frame });
            let mut names = Vec::new();
            if let Ok(locals) = unsafe { &mut *anchor.live() }.getdictscope() {
                let locals_slot = pyre_object::gc_roots::shadow_stack_len();
                let locals = pyre_object::gc_roots::pin_root(locals);
                dict_string_keys(locals, &mut names);
                if let Some(self_obj) = unsafe { pyre_object::w_dict_getitem_str(locals, "self") } {
                    let self_slot = pyre_object::gc_roots::shadow_stack_len();
                    let self_obj = pyre_object::gc_roots::pin_root(self_obj);
                    if crate::baseobjspace::getattr_str(self_obj, &wrong_name).is_ok() {
                        return Some(format!(". Did you mean: 'self.{wrong_name}'?"));
                    }
                }
            }
            let frame = unsafe { &*anchor.live() };
            dict_string_keys(frame.get_w_globals(), &mut names);
            let frame = unsafe { &*anchor.live() };
            dict_string_keys(frame.fget_f_builtins(), &mut names);
            best_suggestion(&names, &wrong_name)
        }
        _ => None,
    };
    let mut suffix = suggestion.map(|value| format!(". Did you mean: '{value}'?"));
    if kind == ExcKind::NameError && stdlib_module_name(&wrong_name) {
        let import = format!("Did you forget to import '{wrong_name}'?");
        match suffix.as_mut() {
            Some(text) => text.push_str(&format!(" Or {import}")),
            None => suffix = Some(format!(". {import}")),
        }
    }
    suffix
}

/// `pypy/interpreter/error.py print_app_tb_only` parity —
/// walk the chained `PyTraceback` head→tail (outermost → innermost)
/// and print each frame as `File "<path>", line N, in <name>` plus the
/// source line.  Silently no-ops when `err.exc_object` is null, the
/// traceback slot is empty, or the source file can't be read.
fn write_traceback_chain<W: Write>(writer: &mut W, err: &PyError) -> std::io::Result<()> {
    if err.exc_object.is_null() {
        return Ok(());
    }
    write_traceback_chain_from_exc(writer, err.exc_object)
}

fn write_traceback_chain_from_exc<W: Write>(
    writer: &mut W,
    exc: PyObjectRef,
) -> std::io::Result<()> {
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return Ok(());
    }
    let tb = unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(exc) };
    write_traceback_chain_from_tb(writer, tb)
}

/// `traceback.py:_RECURSIVE_CUTOFF` — a frame repeating the same
/// `(filename, lineno, name)` is printed at most this many times before the
/// rest collapse into a single `[Previous line repeated N more times]`.
const RECURSIVE_CUTOFF: usize = 3;

/// `traceback.py:StackSummary.format` collapse line, emitted when a run of
/// identical frames just ended (or the walk finished on one).  A run at or
/// below the cutoff was printed in full and needs no marker.
fn write_repeated_frames<W: Write>(writer: &mut W, count: usize) -> std::io::Result<()> {
    if count <= RECURSIVE_CUTOFF {
        return Ok(());
    }
    let count = count - RECURSIVE_CUTOFF;
    let plural = if count > 1 { "s" } else { "" };
    writeln!(
        writer,
        "  [Previous line repeated {count} more time{plural}]"
    )
}

fn write_traceback_chain_from_tb<W: Write>(
    writer: &mut W,
    mut tb: PyObjectRef,
) -> std::io::Result<()> {
    let _roots = pyre_object::gc_roots::push_roots();
    // `StackSummary.format` dedup state: the previous frame's identity and how
    // many consecutive frames have carried it.
    let mut last: Option<(Vec<u8>, Option<i64>, rustpython_wtf8::Wtf8Buf)> = None;
    let mut repeats: usize = 0;
    while !tb.is_null() {
        let tb_slot = pyre_object::gc_roots::shadow_stack_len();
        tb = pyre_object::gc_roots::pin_root(tb);
        let current_tb = pyre_object::gc_roots::shadow_stack_get(tb_slot);
        if !unsafe { crate::pytraceback::is_pytraceback(current_tb) } {
            break;
        }
        let w_code = unsafe { crate::pytraceback::w_pytraceback_get_w_code(current_tb) };
        // `FrameSummary.lineno` is whatever `tb_lineno` answered, `None`
        // included: the node is still printed, with `None` where the line
        // would go and with no source line under it, because `_set_lines`
        // returns early for a summary that has no line to read.
        let lineno = unsafe { crate::pytraceback::w_pytraceback_get_lineno(current_tb) };
        let lasti = unsafe { crate::pytraceback::w_pytraceback_get_lasti(current_tb) };
        let unknown_name = || {
            let mut name = rustpython_wtf8::Wtf8Buf::new();
            name.push_str("<unknown>");
            name
        };
        let (filename, funcname, location) = if w_code.is_null() {
            (b"<unknown>".to_vec(), unknown_name(), None)
        } else {
            // `w_code` is a GC-rooted `PyCode` pointer captured
            // at `record_application_traceback` time; the inner
            // `CodeObject` lives as long as `w_code` is reachable.
            let code_obj = unsafe { crate::w_code_get_ptr(w_code) } as *const crate::CodeObject;
            if code_obj.is_null() {
                (b"<unknown>".to_vec(), unknown_name(), None)
            } else {
                let code = unsafe { &*code_obj };
                // `lasti` is a byte offset; `locations` is indexed by
                // instruction.
                let location = usize::try_from(lasti / 2)
                    .ok()
                    .and_then(|index| crate::pycode::code_locations(code).get(index))
                    .map(|(start, end)| {
                        (
                            start.line.get(),
                            end.line.get(),
                            start.character_offset.get().saturating_sub(1),
                            end.character_offset.get().saturating_sub(1),
                        )
                    });
                let w_name = unsafe { crate::pycode::w_code_name_obj(w_code) };
                let mut funcname = rustpython_wtf8::Wtf8Buf::new();
                if w_name.is_null() {
                    funcname.push_str("<unknown>");
                } else {
                    funcname.push_wtf8(unsafe { pyre_object::w_str_get_wtf8(w_name) });
                }
                (
                    unsafe { crate::pycode::code_filename_bytes(w_code) },
                    funcname,
                    location,
                )
            }
        };
        let key = (filename, lineno, funcname);
        if last.as_ref() != Some(&key) {
            write_repeated_frames(writer, repeats)?;
            last = Some(key.clone());
            repeats = 0;
        }
        repeats += 1;
        if repeats > RECURSIVE_CUTOFF {
            let current_tb = pyre_object::gc_roots::shadow_stack_get(tb_slot);
            tb = unsafe { crate::pytraceback::w_pytraceback_get_w_next(current_tb) };
            continue;
        }
        let (filename, lineno, funcname) = key;
        let shown_lineno = lineno.map_or_else(|| String::from("None"), |line| line.to_string());
        // The name is *reported* as the text it decodes to, with the same error
        // handler that read it off the filesystem, so a byte with no UTF-8
        // spelling reaches the stream as its escape.  Writing the raw bytes
        // instead would leave the report a mix of those bytes and the WTF-8
        // around them, which no sink can encode as a whole.  Opening the source
        // still wants the bytes, which `filename` keeps.
        let shown_filename = crate::gateway::fsdecode_filename_wtf8(&filename);
        writer.write_all(b"  File \"")?;
        writer.write_all(shown_filename.as_bytes())?;
        writer.write_all(b"\", line ")?;
        writer.write_all(shown_lineno.as_bytes())?;
        writer.write_all(b", in ")?;
        writer.write_all(funcname.as_bytes())?;
        writer.write_all(b"\n")?;
        // `FrameSummary._set_lines` collects every line the failing
        // instruction spans, so a statement written across several lines (a
        // class body, a multi-line call) shows all of them, dedented by the
        // indentation they share.
        if let Some((start_line, end_line, _, _)) = location
            && lineno.and_then(|line| usize::try_from(line).ok()) == Some(start_line)
            && end_line > start_line
        {
            let span: Vec<String> = (start_line..=end_line)
                .map(|n| {
                    frame_source_line(tb_slot, &filename, n as i64)
                        .map_or(String::new(), |l| l.trim_end().to_string())
                })
                .collect();
            if !span.iter().all(|line| line.trim().is_empty()) {
                for line in dedent_lines(&span) {
                    writeln!(writer, "    {line}")?;
                }
            }
        } else if let Some(line) =
            lineno.and_then(|line| frame_source_line(tb_slot, &filename, line))
        {
            let raw_line = line.trim_end_matches(['\n', '\r']);
            let shown_line = raw_line.trim_start();
            writeln!(writer, "    {shown_line}")?;

            if let Some((start_line, end_line, start_col, end_col)) = location
                && lineno.and_then(|line| usize::try_from(line).ok()) == Some(start_line)
                && end_line == start_line
                && end_col > start_col
                && start_col <= raw_line.len()
                && end_col <= raw_line.len()
                && raw_line.is_char_boundary(start_col)
                && raw_line.is_char_boundary(end_col)
            {
                let lead = raw_line.len() - shown_line.len();
                let anchors = traceback_anchors(raw_line, start_col, end_col);
                // `traceback.py:_should_show_carets` — a plain caret span covering
                // the whole stripped line (nothing but whitespace before the start
                // or after the end, and no binary-op/subscript sub-anchor) carries
                // no information, so it is suppressed. A `<name> = <call>(…)` or
                // `return <call>(…)` whose call value exactly covers the span is
                // likewise suppressed even though it has a call sub-anchor.
                let before_ws_only = raw_line
                    .get(..start_col)
                    .unwrap_or("")
                    .trim_start()
                    .is_empty();
                let after_ws_only = raw_line.get(end_col..).unwrap_or("").trim_end().is_empty();
                let full_line_call = spawns_full_line_call(
                    shown_line,
                    start_col.saturating_sub(lead),
                    end_col.saturating_sub(lead),
                );
                if !full_line_call && (anchors.is_some() || !before_ws_only || !after_ws_only) {
                    let mut caret_line = String::from("    ");
                    caret_line.push_str(&" ".repeat(start_col.saturating_sub(lead)));
                    for col in start_col..end_col {
                        let marker = if let Some((lo, hi)) = anchors {
                            if (lo..hi).contains(&col) { '^' } else { '~' }
                        } else {
                            '^'
                        };
                        caret_line.push(marker);
                    }
                    writeln!(writer, "{caret_line}")?;
                }
            }
        }
        let current_tb = pyre_object::gc_roots::shadow_stack_get(tb_slot);
        tb = unsafe { crate::pytraceback::w_pytraceback_get_w_next(current_tb) };
    }
    write_repeated_frames(writer, repeats)
}

/// `textwrap.dedent` — drop the longest leading-whitespace prefix shared by
/// every non-blank line; whitespace-only lines normalise to empty.
fn dedent_lines(lines: &[String]) -> Vec<String> {
    let mut prefix: Option<&str> = None;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let indent = &line[..line.len() - line.trim_start().len()];
        prefix = Some(match prefix {
            None => indent,
            Some(common) => {
                let shared = common
                    .bytes()
                    .zip(indent.bytes())
                    .take_while(|(a, b)| a == b)
                    .count();
                &common[..shared]
            }
        });
    }
    let prefix = prefix.unwrap_or("");
    lines
        .iter()
        .map(|line| {
            if line.trim().is_empty() {
                String::new()
            } else {
                line[prefix.len()..].to_string()
            }
        })
        .collect()
}

/// `traceback.py:_should_show_carets` special case: a `<name> = <call>(…)` or
/// `return <call>(…)` statement whose call value exactly spans the failing
/// instruction (byte offsets `seg_start..seg_end` into `line`, the already-
/// dedented source line). Such a caret would merely re-underline the whole
/// right-hand side, so it is suppressed.
fn spawns_full_line_call(line: &str, seg_start: usize, seg_end: usize) -> bool {
    let Ok(parsed) = parser::parse_module(line) else {
        return false;
    };
    let body = &parsed.syntax().body;
    let [stmt] = body.as_slice() else {
        return false;
    };
    let call = match stmt {
        ast::Stmt::Assign(assign) => {
            if assign.targets.len() != 1
                || !matches!(assign.targets.first(), Some(ast::Expr::Name(_)))
                || !matches!(assign.value.as_ref(), ast::Expr::Call(_))
            {
                return false;
            }
            assign.value.range()
        }
        ast::Stmt::Return(ret) => match ret.value.as_deref() {
            Some(ast::Expr::Call(call)) if matches!(call.func.as_ref(), ast::Expr::Name(_)) => {
                call.range()
            }
            _ => return false,
        },
        _ => return false,
    };
    call.start().to_usize() == seg_start && call.end().to_usize() == seg_end
}

/// Return the secondary-marker span used by `traceback.py` for an expression.
fn traceback_anchors(raw_line: &str, start_col: usize, end_col: usize) -> Option<(usize, usize)> {
    let segment = raw_line.get(start_col..end_col)?;
    let wrapped = format!("(\n{segment}\n)");
    let parsed = parser::parse_expression(&wrapped).ok()?;
    let expr = parsed.syntax().body.as_ref();

    let map_offset = |offset: usize| {
        offset
            .checked_sub(2)
            .and_then(|offset| offset.checked_add(start_col))
            .map(|offset| offset.clamp(start_col, end_col))
    };
    let map_range = |lo: usize, hi: usize| {
        let lo = map_offset(lo)?;
        let hi = map_offset(hi)?;
        (lo < hi).then_some((lo, hi))
    };

    match expr {
        ast::Expr::BinOp(bin_op) => {
            let left_end = bin_op.left.range().end().to_usize();
            let right_start = bin_op.right.range().start().to_usize();
            if left_end > right_start || right_start > wrapped.len() {
                return None;
            }
            let between = wrapped.as_bytes().get(left_end..right_start)?;
            let op_relative = between
                .iter()
                .position(|byte| !byte.is_ascii_whitespace() && *byte != b')')?;
            let op_start = left_end.checked_add(op_relative)?;
            let mut op_end = op_start;
            while op_end < right_start {
                let byte = *wrapped.as_bytes().get(op_end)?;
                if byte.is_ascii_whitespace()
                    || byte == b')'
                    || byte.is_ascii_alphanumeric()
                    || byte == b'_'
                {
                    break;
                }
                op_end = op_end.checked_add(1)?;
            }
            map_range(op_start, op_end)
        }
        ast::Expr::Subscript(subscript) => map_range(
            subscript.value.range().end().to_usize(),
            expr.range().end().to_usize(),
        ),
        ast::Expr::Call(call) => map_range(
            call.func.range().end().to_usize(),
            expr.range().end().to_usize(),
        ),
        _ => None,
    }
}

/// `FrameSummary._set_lines` — the filename-keyed `linecache.getline` first
/// and, for a synthetic `<…>` name, the code-keyed `_getline_from_code`
/// fallback that reaches a source published by `linecache._register_code`
/// (the `-c` command and the interactive prompt, neither of which has a file).
///
/// The code object is read back off the traceback rooted at `tb_slot` rather
/// than passed in: the filesystem lookup runs first and can collect, so a
/// pointer captured ahead of it would not survive.
fn frame_source_line(tb_slot: usize, filename: &[u8], lineno: i64) -> Option<String> {
    if let Some(line) = read_source_line(filename, lineno) {
        return Some(line);
    }
    if filename.first() != Some(&b'<') {
        return None;
    }
    let w_code = unsafe {
        crate::pytraceback::w_pytraceback_get_w_code(pyre_object::gc_roots::shadow_stack_get(
            tb_slot,
        ))
    };
    if w_code.is_null() {
        return None;
    }
    read_registered_source_line(w_code, lineno)
}

/// `linecache._getline_from_code` — the `_interactive_cache` entry a
/// `_register_code` call published for this code object, or `None` when there
/// is none.  The entry carries its lines with the trailing newline, which both
/// callers already trim.
///
/// `sys.modules` is only consulted, never imported into: the cache is empty
/// unless something already imported `linecache` to register a source, so an
/// absent module is an empty cache — and rendering a traceback must not run an
/// import.
fn read_registered_source_line(w_code: PyObjectRef, lineno: i64) -> Option<String> {
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_code);
    let code_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let linecache = crate::importing::get_sys_module("linecache")?;
    let _ = pyre_object::gc_roots::pin_root(linecache);
    let linecache_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    // Each allocation can collect, so every operand is rooted before the next
    // one is built rather than left in a temporary.
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(lineno));
    let lineno_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let getline = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(linecache_slot),
        "_getline_from_code",
    )
    .ok()?;
    let _ = pyre_object::gc_roots::pin_root(getline);
    let getline_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let line = crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(getline_slot),
        &[
            pyre_object::gc_roots::shadow_stack_get(code_slot),
            pyre_object::gc_roots::shadow_stack_get(lineno_slot),
        ],
    )
    .ok()?;
    let line = crate::baseobjspace::text_w(line).ok()?;
    (!line.is_empty()).then(|| line.to_string())
}

/// Open `filename` and return its `lineno`-th line (1-indexed).  Returns
/// `None` for synthetic / unreadable sources — matches PyPy's silent
/// `linecache.getline` fallback at `error.py`.
///
/// The name arrives as the filesystem bytes the code object carries, because
/// `linecache.getline` is handed `co_filename` itself: a path spelling a byte
/// with no UTF-8 form still has to open the file it names, which the lossy
/// spelling no longer does.
fn read_source_line(filename: &[u8], lineno: i64) -> Option<String> {
    if lineno <= 0 || filename.is_empty() || filename.first() == Some(&b'<') {
        return None;
    }
    #[cfg(all(feature = "host_env", unix))]
    let path = {
        use std::os::unix::ffi::OsStrExt;
        std::path::Path::new(std::ffi::OsStr::from_bytes(filename))
    };
    // Elsewhere the platform has no bytes spelling of a path, so a name that is
    // not valid UTF-8 simply does not resolve.
    #[cfg(all(feature = "host_env", not(unix)))]
    let lossy = String::from_utf8_lossy(filename);
    #[cfg(all(feature = "host_env", not(unix)))]
    let path = std::path::Path::new(lossy.as_ref());
    #[cfg(feature = "host_env")]
    {
        // Read through the import machinery's source provider, not std::fs:
        // under sandbox that routes the read through the seam to the controller
        // VFS, so a guest-controlled traceback path cannot leak a host file.
        // On wasm32 the provider is the embedder's — the host-FS bridge for the
        // native-host build, `NullSourceProvider` in a browser — so the same
        // call renders the offending line wherever one is actually reachable.
        let bytes = crate::importing::read_source_bytes(path).ok()?;
        // `linecache` opens with `tokenize.open`, so the BOM and the PEP 263
        // cookie decide the encoding; a plain UTF-8 read drops the whole line
        // for a file declaring anything else.  The name reaches the decode only
        // to report a failure this caller discards, so the lossy spelling of a
        // path with no UTF-8 form is enough there.
        let content = crate::compile::decode_source_bytes(
            &bytes,
            &crate::gateway::fsdecode_filename_wtf8(filename),
            false,
        )
        .ok()?;
        content
            .lines()
            .nth((lineno - 1) as usize)
            .map(|s| s.to_string())
    }
    #[cfg(not(feature = "host_env"))]
    {
        // Sandbox-intentional: PyPy's `error.py linecache.getline`
        // also returns silently when the source can't be read; with
        // host_env off the interpreter must not reach `std::fs`
        // directly, so we treat every source as unreadable and let the
        // traceback render `^^^` markers without the offending line.
        let _ = (filename, lineno);
        None
    }
}

/// Write a rendered report to the host's stderr, spending the encode the
/// stream object would have.
///
/// A report is assembled as WTF-8 throughout, which is what `sys.stderr.write`
/// takes: a lone surrogate reaches the stream as itself and its
/// `errors='backslashreplace'` decides how it is spelled, exactly as
/// `PyErr_Display` leaves that to the stream.  A raw fd has no codec behind it,
/// so a caller writing to one owes that encode itself -- putting the WTF-8
/// bytes straight on the stream would emit a sequence that is not valid UTF-8.
pub(crate) fn emit_report_to_host_stderr(buf: &[u8]) {
    match rustpython_wtf8::Wtf8::from_bytes(buf) {
        Some(report) => {
            let text = crate::display::wtf8_display_string(report.to_wtf8_buf(), "<unprintable>");
            crate::host_seam::emit_stderr(text.as_bytes());
        }
        // Every writer into the buffer emits WTF-8, so this is a byte none of
        // them put there.  Read it the way a filesystem name is read, then
        // spend the same encode the arm above does: the byte survives as its
        // escape, and this sink and the `sys.stderr` sink agree on the report.
        None => {
            let decoded = crate::gateway::fsdecode_filename_wtf8(buf);
            let text = crate::display::wtf8_display_string(decoded, "<unprintable>");
            crate::host_seam::emit_stderr(text.as_bytes());
        }
    }
}

/// `display_exception`'s `stderr = sys.stderr` — the banner lines its failure
/// arm prints go to the stream the application configured, not to the host seam
/// beneath it.  A stream set to `None` disables the report the way
/// `PyErr_Display` does; only a missing `sys` or a failing lookup falls through
/// to the seam, because treating the two alike leaks the write past a
/// configured sink.
pub(crate) fn emit_report_to_sys_stderr(buf: &[u8]) {
    let stream = crate::importing::get_interpreter_sys_module()
        .and_then(|sys| crate::baseobjspace::getattr_str(sys, "stderr").ok());
    let Some(stream) = stream else {
        emit_report_to_host_stderr(buf);
        return;
    };
    if stream.is_null() || unsafe { pyre_object::is_none(stream) } {
        return;
    }
    // The write runs Python, so the stream is published and read back from its
    // slot: a Rust local would still name the address it had before the
    // allocation below moved the object.
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(stream);
    // Every caller passes a literal this file holds, so the bytes are
    // well-formed WTF-8; the lossy read is the last resort for a byte no writer
    // put there.
    let w_text = match rustpython_wtf8::Wtf8::from_bytes(buf) {
        Some(text) => pyre_object::w_str_from_wtf8_managed(text.to_wtf8_buf()),
        None => pyre_object::w_str_new_managed(&String::from_utf8_lossy(buf)),
    };
    let _ = pyre_object::gc_roots::pin_root(w_text);
    let text_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    // Read the stream and text back after that allocation, not before it.
    let result = crate::baseobjspace::call_method(
        pyre_object::gc_roots::shadow_stack_get(sp),
        "write",
        &[pyre_object::gc_roots::shadow_stack_get(text_slot)],
    );
    if result.is_null() {
        let _ = crate::call::take_call_error();
        emit_report_to_host_stderr(buf);
    }
}

/// `originalexcepthook` for an exception a Rust caller holds: `sys.excepthook`'s
/// default is the renderer that reaches the live `sys.stderr`, and
/// `display_exception` hands it each report its failure arm prints.  A hook that
/// cannot render leaves the host seam as the last sink, which is where a report
/// with nowhere else to go belongs.
fn report_through_default_excepthook(failure: &mut PyError) {
    let roots = pyre_object::gc_roots::push_roots();
    let failure_slot = failure.pin(&roots);
    let exc = failure.to_exc_object();
    failure.reload(&roots, failure_slot);
    if exc.is_null() {
        eprint_exception(failure, true);
        return;
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(exc);
    let w_type =
        crate::baseobjspace::exception_getclass(pyre_object::gc_roots::shadow_stack_get(sp));
    let _ = pyre_object::gc_roots::pin_root(if w_type.is_null() {
        pyre_object::w_none()
    } else {
        w_type
    });
    let w_tb = unsafe {
        pyre_object::interp_exceptions::w_exception_get_traceback(
            pyre_object::gc_roots::shadow_stack_get(sp),
        )
    };
    unsafe { crate::pytraceback::mark_traceback_escaped(w_tb) };
    let _ = pyre_object::gc_roots::pin_root(if w_tb.is_null() {
        pyre_object::w_none()
    } else {
        w_tb
    });
    let reported = crate::builtins::sys_excepthook(&[
        pyre_object::gc_roots::shadow_stack_get(sp + 1),
        pyre_object::gc_roots::shadow_stack_get(sp),
        pyre_object::gc_roots::shadow_stack_get(sp + 2),
    ]);
    failure.reload(&roots, failure_slot);
    if reported.is_err() {
        eprint_exception(failure, true);
    }
}

/// `pythonrun.c _PyErr_Print` for a caller holding the exception rather than
/// the thread's indicator: every top-level run reports through `sys.excepthook`,
/// so an application's own hook is what reports the failure it was installed
/// for, and the default hook is what carries the report to the stdlib renderer
/// -- which is where a `SyntaxError` gains the misspelled-keyword suggestion
/// that no printer in this file produces.
///
/// `false` says there was no hook to reach, which is `_PyErr_PrintEx`'s
/// `sys.excepthook is missing` arm, and the caller prints the exception itself.
/// A hook that runs and fails is reported here instead: the failure and the
/// original both go through the default hook, which is `display_exception`'s
/// `originalexcepthook`, so this answers `true` and the caller prints nothing.
pub fn print_exception_via_excepthook(err: &mut PyError) -> bool {
    let Some(sys) = crate::importing::get_interpreter_sys_module() else {
        return false;
    };
    let _roots = pyre_object::gc_roots::push_roots();
    let err_slot = err.pin(&_roots);
    let sys_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(sys);
    // Materialising the instance is what gives the hook something to report.
    // `pin_root` is a safepoint; `expand_pop_roots` reloads the carrier
    // before it is read again. Each of the three arguments is pinned as it
    // is taken.
    err.reload(&_roots, err_slot);
    let exc = err.to_exc_object();
    if exc.is_null() {
        return false;
    }
    let exc_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(exc);
    let w_type =
        crate::baseobjspace::exception_getclass(pyre_object::gc_roots::shadow_stack_get(exc_slot));
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(if w_type.is_null() {
        pyre_object::w_none()
    } else {
        w_type
    });
    let w_tb = unsafe {
        pyre_object::interp_exceptions::w_exception_get_traceback(
            pyre_object::gc_roots::shadow_stack_get(exc_slot),
        )
    };
    unsafe { crate::pytraceback::mark_traceback_escaped(w_tb) };
    let tb_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(if w_tb.is_null() {
        pyre_object::w_none()
    } else {
        w_tb
    });
    // `app_main.py display_exception` sets the sys.last_xxx attributes before
    // it resolves the hook, so a hook that inspects them -- a post-mortem
    // debugger is the reason they exist -- finds the exception it was handed.
    // `last_exc` names the exception itself and the three beside it are the
    // triple that predates it, as `PyErr_PrintEx` stores them.  A store that
    // fails leaves that one name alone.
    for (name, slot) in [
        ("last_exc", exc_slot),
        ("last_type", type_slot),
        ("last_value", exc_slot),
        ("last_traceback", tb_slot),
    ] {
        let stored = crate::baseobjspace::setattr_str(
            pyre_object::gc_roots::shadow_stack_get(sys_slot),
            name,
            pyre_object::gc_roots::shadow_stack_get(slot),
        );
        drop(stored);
    }
    // `display_exception` resolves the hook only once those stores are in
    // place, so a `sys` whose attribute lookup runs Python -- a module class
    // with its own `__getattribute__` -- reads the names already written.
    // A missing name is a hook that is not there; `None` is a hook that is,
    // and calling it is what reports it.
    let Ok(hook) = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(sys_slot),
        "excepthook",
    ) else {
        return false;
    };
    if hook.is_null() {
        return false;
    }
    let hook_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(hook);
    let arguments = [
        pyre_object::gc_roots::shadow_stack_get(type_slot),
        pyre_object::gc_roots::shadow_stack_get(exc_slot),
        pyre_object::gc_roots::shadow_stack_get(tb_slot),
    ];
    let reported = crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(hook_slot),
        &arguments,
    );
    if let Err(mut failure) = reported {
        // `display_exception`'s `except BaseException` arm: the hook is named
        // as the thing that failed and its own report follows, both on the live
        // `sys.stderr` the arm reads -- an application that replaced the stream
        // is where its diagnostics belong, and the host seam beneath it would
        // miss a capture buffer entirely.
        let failure_slot = failure.pin(&_roots);
        emit_report_to_sys_stderr(b"Error calling sys.excepthook:\n");
        failure.reload(&_roots, failure_slot);
        report_through_default_excepthook(&mut failure);
        emit_report_to_sys_stderr(b"\nOriginal exception was:\n");
        // The arm falls out of the `try` into `originalexcepthook(etype,
        // evalue, etraceback)`, so the original is reported from here onto that
        // same stream.  Leaving it to the caller would split one report across
        // two sinks, because the caller's printer writes to the seam.
        if crate::builtins::sys_excepthook(&[
            pyre_object::gc_roots::shadow_stack_get(type_slot),
            pyre_object::gc_roots::shadow_stack_get(exc_slot),
            pyre_object::gc_roots::shadow_stack_get(tb_slot),
        ])
        .is_err()
        {
            err.reload(&_roots, err_slot);
            eprint_exception(err, true);
        }
        return true;
    }
    true
}

pub fn eprint_exception(err: &PyError, include_traceback: bool) {
    // Buffer then emit through the host_seam so the traceback rides the same
    // mediated stderr as sys.stderr under sandbox (raw fd 2 would bypass the
    // controller / corrupt nothing but escape the seam).
    let mut buf: Vec<u8> = Vec::new();
    let _ = write_exception(&mut buf, err, include_traceback);
    emit_report_to_host_stderr(&buf);
}

/// `app_main.py handle_sys_exit` — `exitcode = e.code`; `None` exits
/// 0; otherwise `int(exitcode)`, and a value `int()` rejects is printed to
/// stderr with exit status 1. `e.code` itself is `args[0]` for a 1-arg raise
/// and the whole args tuple otherwise (`interp_exceptions.py descr_init
/// W_SystemExit.descr_init`). A `SystemExit` with no object behind it has no
/// `code` attribute beyond the class default `None`, i.e. a success exit.
///
/// Lives here rather than in a launcher because pyre has two of them —
/// `pyrex` and the wasm `run_python` entry — and both terminate on the same
/// rule.
pub fn system_exit_code(err: &PyError) -> i32 {
    let exc = err.exc_object;
    if exc.is_null() {
        return 0;
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let exc_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(exc);
    let code_name = pyre_object::unicodeobject::intern_str_value("code");
    let pair = pyre_object::gc_roots::pin_roots(&[
        pyre_object::gc_roots::shadow_stack_get(exc_slot),
        code_name,
    ]);
    let mut code = match crate::getattr(
        pyre_object::gc_roots::shadow_stack_get(pair),
        pyre_object::gc_roots::shadow_stack_get(pair + 1),
    ) {
        Ok(c) => c,
        Err(_) => return 1,
    };
    if unsafe { pyre_object::is_none(code) } {
        return 0;
    }
    // `pylifecycle.c _Py_HandleSystemExit` tests `PyLong_Check(exc)` — there is
    // no `int()` coercion, so a float or any other non-integer code is printed
    // rather than converted.
    if unsafe { pyre_object::is_int_or_long(code) } {
        // `exitcode = (int)PyLong_AsLong(exc)`: a value too wide for a machine
        // word leaves the -1 `PyLong_AsLong` returns on overflow, and the
        // narrowing to `int` is a plain truncation. So `SystemExit(10**100)`
        // and `SystemExit(-1)` both exit 255.
        return crate::baseobjspace::int_w(code).unwrap_or(-1) as i32;
    }
    // `pylifecycle.c _Py_HandleSystemExit` writes the code object through
    // `sys.stderr`, so the message spends that stream's codec and its
    // `backslashreplace` handler: under `PYTHONIOENCODING=latin-1`,
    // `sys.exit("h\xe9")` reaches the descriptor as `h\xe9` rather than as the
    // utf-8 spelling of it.
    if let Ok(text) = pyre_object::with_roots!(code => unsafe { crate::py_str_wtf8(code) }) {
        let mut buf = text.as_bytes().to_vec();
        buf.push(b'\n');
        if pyre_object::with_roots!(code => PyError::write_to_sys_stderr(&buf)) {
            return 1;
        }
    }
    // The host's stderr has no codec behind it, so a stream that is missing or
    // refuses the write leaves the text to spend a `backslashreplace` encode
    // here instead.
    let text = unsafe { crate::display::py_str_display(code) };
    crate::host_seam::emit_stderr(format!("{text}\n").as_bytes());
    1
}

/// pypy/interpreter/error.py `ClearedOpErr`.
///
/// ```python
/// class ClearedOpErr:
///     def __init__(self, space):
///         self.operr = OperationError(space.w_None, space.w_None)
/// ```
///
/// Both halves are `space.w_None`, so the cleared carrier names no exception
/// at all — it is the value "nothing is being handled", not an error.  Here
/// that is `w_None` in the exception slot: `normalize_exception` refuses it, so
/// a sentinel that escaped cannot be reported as a raised exception.  The
/// `kind` tag has no "no exception" spelling and is never read while the slot
/// holds `w_None`.
pub fn get_cleared_operation_error(_space: PyObjectRef) -> OperationError {
    let _ = _space;
    let mut operr = PyError::from_type_and_value(pyre_object::w_none(), pyre_object::w_none());
    operr.set_w_type(pyre_object::w_none());
    operr.set_exc_object(pyre_object::w_none());
    operr.set_attach_tb(false);
    operr
}

/// pypy/interpreter/error.py `get_converted_unexpected_exception`.
///
/// ```python
/// except KeyboardInterrupt:
///     return OperationError(space.w_KeyboardInterrupt, space.w_None)
/// except MemoryError:
///     return OperationError(space.w_MemoryError, space.w_None)
/// except rstackovf.StackOverflow as e:
///     rstackovf.check_stack_overflow()
///     return oefmt(space.w_RecursionError, "maximum recursion depth exceeded")
/// except RuntimeError:   # not on top of py.py
///     return OperationError(space.w_RuntimeError, space.w_None)
/// except:
///     return OperationError(space.w_SystemError, space.newtext(
///         "unexpected internal exception (please report a bug): %r%s"
///         % (e, extra)))
/// ```
///
/// The four typed arms re-raise `e` to sort it by RPython exception class.
/// A `dyn Error` carries no such class: an interrupt reaches pyre through the
/// signal module rather than as a host error, an allocation failure aborts the
/// process instead of unwinding, and a stack overflow is caught by the
/// recursion guard before any host error exists.  So every host error lands in
/// the `except:` arm, which is also the one that reads `e` — reporting the
/// failure instead of discarding it.  `extra` names the traceback
/// `debug_print_traceback` dumps, and nothing dumps one here.
///
/// `@jit.dont_look_inside` (`error.py get_converted_unexpected_exception`).
#[majit_macros::dont_look_inside]
pub fn get_converted_unexpected_exception(
    _space: PyObjectRef,
    error: &dyn std::error::Error,
) -> OperationError {
    let _ = _space;
    PyError::new(
        crate::PyErrorKind::SystemError,
        format!("unexpected internal exception (please report a bug): {error:?}"),
    )
}

/// pypy/interpreter/error.py `decompose_valuefmt`.
///
/// Every format code has a literal part on each side, so a format string that
/// ends in one still contributes a trailing empty part. `_compute_value`
/// relies on that — it walks the parts and the formats in lockstep — and
/// `get_operrcls2` asserts the resulting relationship.
pub fn decompose_valuefmt(valuefmt: &str) -> (Vec<String>, Vec<String>) {
    const FMTS: &str = "8NRSTds";

    let mut strings = Vec::new();
    let mut formats = Vec::new();
    let mut current = String::new();

    let mut iter = valuefmt.chars();
    while let Some(ch) = iter.next() {
        if ch == '%' {
            match iter.next() {
                Some('%') => current.push('%'),
                Some(spec) if FMTS.contains(spec) => {
                    strings.push(std::mem::take(&mut current));
                    formats.push(spec.to_string());
                }
                // Upstream raises `ValueError` here, but `oefmt` is
                // `@specialize.arg(1)`: `valuefmt` is a constant per call site
                // and the class this feeds is built during translation, so the
                // raise aborts the build rather than a running program.  A
                // panic is that same "this call site is malformed" report at
                // the earliest point this port can make it.
                Some(_) | None => {
                    panic!("invalid format string (only %8, %N, %R, %S, %T, %d or %s supported)")
                }
            }
        } else {
            current.push(ch);
        }
    }

    // Unconditionally, so `"%s"`-terminated formats keep the empty tail part
    // that makes `strings` one longer than `formats`.
    strings.push(current);

    assert!(!formats.is_empty(), "unsupported: no % command found");
    (strings, formats)
}

pub fn get_operrcls2(valuefmt: &str) -> (PyObjectRef, Vec<String>) {
    let (strings, formats) = decompose_valuefmt(valuefmt);
    assert_eq!(strings.len(), formats.len() + 1);
    (std::ptr::null_mut(), strings)
}

pub fn get_operr_class(valuefmt: &str) -> (PyObjectRef, Vec<String>) {
    get_operrcls2(valuefmt)
}

/// `OperationError(w_type, w_error)` over the instance a class call produced.
///
/// A constructor that raised reports its own failure: upstream's
/// `space.call_function` propagates out of `exception_from_errno` and
/// `_wrap_oserror2_impl` rather than being replaced by the error they were
/// assembling.  Only a call that answered with something that is not an
/// exception has no instance to carry, and takes `fallback`.
fn operation_error_from_instance(
    w_type: PyObjectRef,
    called: Result<PyObjectRef, PyError>,
    fallback: impl FnOnce() -> OperationError,
) -> OperationError {
    match called {
        Ok(w_error) if !w_error.is_null() && unsafe { pyre_object::is_exception(w_error) } => {
            PyError::from_type_and_value(w_type, w_error)
        }
        Err(err) => err,
        Ok(_) => fallback(),
    }
}

/// `OperationError(w_type, space.newtext(message))`. The instance is built
/// later by [`PyError::normalize_exception`].
///
/// A null class falls back to the message alone — a value cannot be reported
/// through a class that is not there.
fn operation_error_from_class(w_type: PyObjectRef, message: Wtf8Buf) -> OperationError {
    if w_type.is_null() {
        return PyError::runtime_error(message);
    }
    // `OpErrFmt` stores the formatted text on the carrier and builds the
    // application `str` in `get_w_value`. Minting `space.newtext` here
    // paid a Python string on every raise, including an `except TypeError`
    // that never reads `e`. The class still has to outlive the carrier
    // malloc, so it is pinned across it.
    let kind = pyre_object::interp_exceptions::kind_of_canonical_exc_class(w_type)
        .map(PyError::kind_from_exc)
        .unwrap_or(PyErrorKind::RuntimeError);
    let _roots = pyre_object::gc_roots::push_roots();
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = _roots.pin_root(w_type);
    make_pyerror(
        kind,
        DisplayMessage::Text(message),
        std::ptr::null_mut(),
        true,
        false,
        -1,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        _roots.get(type_slot),
    )
}

/// pypy/interpreter/error.py `oefmt`.
///
/// ```python
/// def oefmt(w_type, valuefmt, *args):
///     if not len(args):
///         return OpErrFmtNoArgs(w_type, valuefmt)
///     OpErrFmt, strings = get_operr_class(valuefmt)
///     return OpErrFmt(w_type, strings, *args)
/// ```
///
/// The two `OpErrFmt` classes exist so RPython can defer `%`-interleaving until
/// something reads the message; `get_operrcls2` cannot mint a class here, so
/// the work `OpErrFmt._compute_value` does happens up front instead, and what
/// is left is the step upstream takes in both arms — build the error **from
/// `w_type`**.
///
/// Each format code keeps the operand kind that
/// `get_operrcls2.OpErrFmt._compute_value` expects, so conversion is selected
/// by the code rather than by one common display implementation.
pub enum FmtArg<'a> {
    /// `%s`: text copied as-is.
    Text(&'a str),
    /// `%d`: an integer rendered in decimal.
    Int(i64),
    /// `%8`: raw bytes decoded as UTF-8 with replacement.
    Bytes(&'a [u8]),
    /// `%R`, `%S`, `%T`, and `%N`: a wrapped object.
    Obj(PyObjectRef),
}

impl FmtArg<'_> {
    fn kind_name(&self) -> &'static str {
        match self {
            Self::Text(_) => "Text",
            Self::Int(_) => "Int",
            Self::Bytes(_) => "Bytes",
            Self::Obj(_) => "Obj",
        }
    }
}

fn expected_fmt_arg(format: char) -> &'static str {
    match format {
        's' => "Text or Bytes",
        'd' => "Int",
        '8' => "Bytes",
        'R' | 'S' | 'T' | 'N' => "Obj",
        _ => unreachable!("decompose_valuefmt accepted an unknown format code"),
    }
}

/// The shared `%R` / `%S` conversion and rescue from
/// `get_operrcls2.OpErrFmt._compute_value`.
fn format_obj_as_utf8(obj: PyObjectRef, format: char) -> Wtf8Buf {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = _roots.base();
    let _ = _roots.pin_root(obj);
    let obj = || _roots.get(obj_slot);

    let (rendered, fallback) = if format == 'R' {
        (
            unsafe { crate::display::py_repr_wtf8(obj()) },
            "<repr raised>",
        )
    } else {
        (
            unsafe { crate::display::py_str_wtf8(obj()) },
            "<str raised>",
        )
    };
    match rendered {
        // `space.utf8_w(w_s)` hands back the string's own buffer, which is what
        // `py_repr_wtf8` already returned, so only a repr or str that *raised*
        // takes the rescue below.  Reading it back through a `&str` view
        // instead would divert a lone surrogate — legal in that buffer — into
        // the rescue as well, reporting `<repr raised>` for a repr that
        // succeeded.  The buffer is carried on rather than converted, so the
        // surrogate reaches the message intact.
        Ok(rendered) => rendered,
        Err(mut error) => {
            error.write_unraisable(
                pyre_object::w_none(),
                rustpython_wtf8::Wtf8::new("exception formatting"),
                obj(),
            );
            Wtf8Buf::from_string(fallback.to_string())
        }
    }
}

/// `W_Root.getname`: read `__name__` and answer `?` when the lookup raises —
/// for any `OperationError`, not just the two an attribute miss produces.
///
/// The name is read off the string's own buffer for the reason
/// [`format_obj_as_utf8`] gives: a `&str` view would turn a lone surrogate in
/// `__name__` into a raise, and this function reports a raise as `?`.
fn format_obj_name(obj: PyObjectRef) -> Wtf8Buf {
    match crate::baseobjspace::getattr_str(obj, "__name__")
        .and_then(crate::baseobjspace::text_wtf8_w)
    {
        Ok(name) => name.to_owned(),
        Err(_) => Wtf8Buf::from_string("?".to_string()),
    }
}

/// `space.type(w_obj).getname(space)` — the `%T` spelling.
///
/// The Python-visible type, which is `w_class`; `ob_type` names the storage a
/// value is laid out as, and a class deriving from a builtin shares its
/// storage while presenting its own name.
pub fn type_name_of(w_obj: PyObjectRef) -> String {
    // A tagged int immediate is an exact builtin int; skip the ob_type deref.
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(w_obj) {
        return "int".to_string();
    }
    unsafe {
        match crate::typedef::r#type(w_obj) {
            Some(tp) => pyre_object::w_type_get_name(tp.as_ptr()).to_string(),
            None => (*(*w_obj).ob_type).name.to_string(),
        }
    }
}

/// `error.py oefmt` — build an exception whose message is `valuefmt` with each
/// format code filled from its own argument.
///
/// ```python
/// @specialize.arg(1)
/// def oefmt(w_type, valuefmt, *args):
///     """Equivalent to OperationError(w_type, space.newtext(valuefmt % args)).
///     More efficient in the (common) case where the value is not actually
///     needed."""
/// ```
///
/// The conversions run here rather than behind the message accessor, and that
/// is a decision rather than an omission.  Upstream returns an `OpErrFmt`
/// storing the operands and formats only when `get_w_value` asks
/// `_compute_value`, which is the RPython saving its docstring advertises: an
/// error caught by an interpreter-level guard and discarded never pays for its
/// `%R`.  3.14 does not defer.  `_PyErr_FormatV` renders the whole string
/// before it ever names the exception —
///
/// ```c
/// /* Issue #23571: PyUnicode_FromFormatV() must not be called with an
///    exception set, it calls arbitrary Python code like PyObject_Repr() */
/// _PyErr_Clear(tstate);
/// string = PyUnicode_FromFormatV(format, vargs);
/// ```
///
/// — so `%R`'s `PyObject_Repr` runs at the raise, not at the read.  Measured on
/// 3.14.2 with a `__repr__` that logs, against `types.GenericAlias(obj, ...)[int]`
/// (`genericaliasobject.c` `"%R is not a generic class"`): one call at
/// construction, none at `str(e)`.  Deferring would move that side effect and
/// its unraisable to a different moment than 3.14 picks.
///
/// The upstream saving is also narrower than it looks on 3.x: `except` pushes
/// the instance whether or not the handler binds it, so the first
/// application-level handler forces `_compute_value` anyway.  What is left is
/// the interpreter-internal swallow — `findattr`, the `async` guards — and
/// taking it would need the operands rooted for a `PyError` in flight, which
/// nothing can do: Rust memcpies the carrier at every `?`, the same
/// constraint that keeps the traceback chain on the instance
/// ([`PyError::get_traceback`]).
pub fn oefmt(w_type: PyObjectRef, valuefmt: &str, args: &[FmtArg<'_>]) -> OperationError {
    // `OpErrFmtNoArgs(w_type, valuefmt)` — no argument, so the format string is
    // the message verbatim, `%` sequences and all.  Nothing on the way to the
    // error runs application-level code, so the class needs no root here.
    if args.is_empty() {
        // The format string is the message verbatim. `from_string(to_string())`
        // copied it into a `String` only to copy it again.
        let mut message = Wtf8Buf::new();
        message.push_str(valuefmt);
        return operation_error_from_class(w_type, message);
    }
    // `OpErrFmt, strings = get_operr_class(valuefmt)`, then the walk
    // `_compute_value` makes over them: a literal part, the argument
    // belonging to the format code that followed it, and so on to the
    // trailing part `decompose_valuefmt` guarantees.
    let (strings, formats) = decompose_valuefmt(valuefmt);
    assert_eq!(
        args.len(),
        formats.len(),
        "oefmt argument count mismatch: {} format codes but {} arguments",
        formats.len(),
        args.len()
    );

    // The class and every object operand form one livevar set, published in
    // one pass and normalized once.  `pin_root` per operand would instead
    // query the GC between publications, and that query is a safepoint: it
    // can wait through a foreign collection which moves the operands still
    // held only in `args`, which is not a root.  `publish_roots` states the
    // rule, and `DictOperationGuard::new` follows it for the same reason:
    // its `refs` is a caller-owned native slice no collection rewrites.
    let _roots = pyre_object::gc_roots::push_roots();
    let type_slot = _roots.publish(std::slice::from_ref(&w_type));
    let mut rooted_obj_count = 0;
    for arg in args {
        if let FmtArg::Obj(obj) = arg {
            _roots.publish(std::slice::from_ref(obj));
            rooted_obj_count += 1;
        }
    }
    _roots.normalize(type_slot, 1 + rooted_obj_count);
    // The class is read back out of its slot after the conversions, which is
    // why it is in the set at all: `%R`, `%S` and `%N` each run
    // application-level code between this point and the error below.
    let first_obj_slot = type_slot + 1;
    let mut next_obj_slot = first_obj_slot;
    // The message is a `Wtf8Buf` for the same reason `_compute_value`
    // joins `space.utf8_w` results: `%R`, `%S` and `%N` can each produce a
    // lone surrogate, which no `&str` can hold.
    let mut message = Wtf8Buf::new();
    for (i, part) in strings[..formats.len()].iter().enumerate() {
        message.push_str(part);
        let format = formats[i].chars().next().unwrap();
        let arg = &args[i];
        match (format, arg) {
            ('d', FmtArg::Int(value)) => message.push_str(&value.to_string()),
            ('s', FmtArg::Text(value)) => message.push_str(value),
            // `_compute_value`'s final `else` — `%s` over a byte string,
            // which it keeps verbatim after only checking the length — and
            // `%8`, which it decodes with `'replace'`.  Both run with
            // `allow_surrogates`, so a byte string that is already WTF-8
            // reaches the message unchanged, encoded surrogates included;
            // decoding it as UTF-8 would turn each into `U+FFFD`.
            //
            // Bytes that are not WTF-8 at all are where the two part company
            // upstream: `%8` replaces them, which this matches, while `%s`
            // keeps them and subtracts the bad run from the reported length
            // instead.  No `Wtf8Buf` can hold them, so `%s` replaces them too.
            ('s', FmtArg::Bytes(value)) | ('8', FmtArg::Bytes(value)) => {
                match rustpython_wtf8::Wtf8::from_bytes(value) {
                    Some(value) => message.push_wtf8(value),
                    None => message.push_str(&String::from_utf8_lossy(value)),
                }
            }
            ('T', FmtArg::Obj(_)) => {
                let obj = _roots.get(next_obj_slot);
                next_obj_slot += 1;
                message.push_str(&type_name_of(obj));
            }
            ('N', FmtArg::Obj(_)) => {
                let obj = _roots.get(next_obj_slot);
                next_obj_slot += 1;
                message.push_wtf8(&format_obj_name(obj));
            }
            ('R' | 'S', FmtArg::Obj(_)) => {
                let obj = _roots.get(next_obj_slot);
                next_obj_slot += 1;
                message.push_wtf8(&format_obj_as_utf8(obj, format));
            }
            _ => panic!(
                "oefmt format %{format} requires {} argument, got {}",
                expected_fmt_arg(format),
                arg.kind_name()
            ),
        }
    }
    debug_assert_eq!(next_obj_slot - first_obj_slot, rooted_obj_count);
    message.push_str(&strings[formats.len()]);
    operation_error_from_class(_roots.get(type_slot), message)
}

pub fn debug_print(text: &str, file: Option<&mut dyn Write>, _newline: bool) {
    if let Some(file) = file {
        let _ = file.write_all(text.as_bytes());
    }
}

/// pypy/interpreter/error.py `exception_from_errno`.
///
/// ```python
/// def exception_from_errno(space, w_type, errno):
///     msg, lgt = strerror(errno)
///     w_error = space.call_function(w_type, space.newint(errno),
///                                   space.newtext(msg, lgt))
///     return OperationError(w_type, w_error)
/// ```
pub fn exception_from_errno(
    _space: PyObjectRef,
    w_type: PyObjectRef,
    errno: i32,
) -> OperationError {
    let _ = _space;
    let msg = PyError::clean_strerror(errno);
    if w_type.is_null() {
        // No class to call: `OSError(errno, strerror)` is what the family's
        // own constructor would have produced.
        return PyError::os_error_with_errno(errno, "");
    }
    // Each argument is read back out of its root slot after the last
    // allocation: pinning keeps the object alive, but a collection updates the
    // slot rather than this frame's copy of the pointer. `exception_from_errno`
    // then builds `OperationError(w_type, w_error)` after `space.call_function`,
    // so the class is re-read from its slot after that call too.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = _roots.base();
    let w_type = _roots.pin_root(w_type);
    let w_errno = pyre_object::w_int_new(errno as i64);
    let w_errno = _roots.pin_root(w_errno);
    let w_msg = pyre_object::w_str_new_managed(&msg);
    let w_msg = _roots.pin_root(w_msg);
    let (w_type, w_errno, w_msg) = (_roots.get(base), _roots.get(base + 1), _roots.get(base + 2));
    let w_error = crate::call::call_function_impl_result(w_type, &[w_errno, w_msg]);
    let w_type = _roots.get(base);
    operation_error_from_instance(w_type, w_error, || PyError::os_error_with_errno(errno, ""))
}

/// pypy/interpreter/error.py `exception_from_saved_errno`.
///
/// ```python
/// def exception_from_saved_errno(space, w_type):
///     from rpython.rlib.rposix import get_saved_errno
///     errno = get_saved_errno()
///     return exception_from_errno(space, w_type, errno)
/// ```
///
/// `rposix.get_saved_errno` reads the errno RPython stashed around the last
/// external call; the host `errno` this build reads back through
/// `last_os_error` is the same thread-local the C library wrote.
pub fn exception_from_saved_errno(space: PyObjectRef, w_type: PyObjectRef) -> OperationError {
    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
    exception_from_errno(space, w_type, errno)
}

/// pypy/interpreter/error.py `new_exception_class`.
///
/// ```python
/// def new_exception_class(space, name, w_bases=None, w_dict=None):
///     if '.' in name:
///         module, name = name.rsplit('.', 1)
///     else:
///         module = None
///     if w_bases is None:
///         w_bases = space.newtuple([space.w_Exception])
///     elif not space.isinstance_w(w_bases, space.w_tuple):
///         w_bases = space.newtuple([w_bases])
///     if w_dict is None:
///         w_dict = space.newdict()
///     w_exc = space.call_function(
///         space.w_type, space.newtext(name), w_bases, w_dict)
///     if module:
///         space.setattr(w_exc, space.newtext("__module__"), space.newtext(module))
///     return w_exc
/// ```
pub fn new_exception_class(
    _space: PyObjectRef,
    name: &str,
    w_bases: Option<PyObjectRef>,
    w_dict: Option<PyObjectRef>,
) -> PyObjectRef {
    let _ = _space;
    let (module, name) = match name.rsplit_once('.') {
        Some((module, name)) => (Some(module), name),
        None => (None, name),
    };
    // Every argument below is pinned and then read back out of its slot: a
    // collection during a later allocation rewrites the root SLOT and leaves
    // this frame's copy of the pointer behind, the way `alloc_dict_object`
    // re-reads `dstorage` across its header malloc.
    let _roots = pyre_object::gc_roots::push_roots();
    let w_bases = match w_bases {
        None => pyre_object::w_tuple_new(vec![crate::typedef::gettypeobject(
            &pyre_object::interp_exceptions::EXC_EXCEPTION_TYPE,
        )]),
        Some(w_bases) if unsafe { !pyre_object::is_tuple(w_bases) } => {
            let slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = _roots.pin_root(w_bases);
            pyre_object::w_tuple_new(vec![_roots.get(slot)])
        }
        Some(w_bases) => w_bases,
    };
    let bases_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = _roots.pin_root(w_bases);
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = _roots.pin_root(w_dict.unwrap_or_else(pyre_object::w_dict_new));
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = _roots.pin_root(pyre_object::w_str_new_managed(name));
    let args = [
        _roots.get(name_slot),
        _roots.get(bases_slot),
        _roots.get(dict_slot),
    ];
    // `space.call_function` propagates its failure; this signature answers with
    // NULL instead, so the error travels the bare-return convention's channel —
    // `set_call_error`, which `take_call_error` hands back — rather than being
    // dropped for a caller that then reads whatever an earlier call left.
    let w_exc = match crate::call::call_function_impl_result(crate::typedef::w_type(), &args) {
        Ok(w_exc) => w_exc,
        Err(err) => {
            crate::call::set_call_error(err);
            return std::ptr::null_mut();
        }
    };
    // `if module:` — a name like `".Name"` splits to an empty prefix, which is
    // falsy upstream, so the attribute is left alone rather than set to `""`.
    if let Some(module) = module.filter(|module| !module.is_empty()) {
        let exc_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = _roots.pin_root(w_exc);
        // Pinned like every other operand here: `setattr_str` allocates, and a
        // collection rewrites the root SLOT rather than this frame's copy.
        let module_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = _roots.pin_root(pyre_object::w_str_new_managed(module));
        // `space.setattr` propagates its failure through the same bare-return
        // channel as the class call above.
        if let Err(err) = crate::baseobjspace::setattr_str(
            _roots.get(exc_slot),
            "__module__",
            _roots.get(module_slot),
        ) {
            crate::call::set_call_error(err);
            return std::ptr::null_mut();
        }
        return _roots.get(exc_slot);
    }
    w_exc
}

/// pypy/interpreter/error.py `wrap_oserror2`, with the body
/// `_wrap_oserror2_impl` holds.
///
/// ```python
/// if w_exception_class is None:
///     w_exc = space.w_OSError
/// else:
///     w_exc = w_exception_class
/// ...
/// errno = e.errno
/// if errno == EINTR:
///     space.getexecutioncontext().checksignals()
/// msg, lgt = strerror(errno)
/// w_error = space.call_function(w_exc, w_errno, w_msg, w_filename,
///                               w_winerror, w_filename2)
/// operror = OperationError(space.type(w_error), w_error)
/// ```
///
/// The `eintr_retry` half of the double API has no caller here, so this is the
/// `eintr_retry=False` arm: EINTR still runs `checksignals`, and the error is
/// returned rather than raised.  `w_winerror` stays `None` — the Windows arm
/// keys on `rwin32.WIN32 and isinstance(e, WindowsError)`, and this build
/// resolves a host error to its POSIX errno through `io_error_posix_errno` on
/// every platform.
pub fn wrap_oserror2(
    space: PyObjectRef,
    error: &(dyn std::error::Error + 'static),
    w_filename: Option<PyObjectRef>,
    w_exception_class: Option<PyObjectRef>,
    w_filename2: Option<PyObjectRef>,
) -> OperationError {
    let _ = space;
    // `assert isinstance(e, OSError)` — a host error that is not an io error
    // carries no errno, so it reports as the generic failure it is.
    let Some(io_error) = error.downcast_ref::<std::io::Error>() else {
        return PyError::os_error(error.to_string());
    };
    let errno = crate::builtins::io_error_posix_errno(io_error, 0);
    // wasm32 has neither `libc::EINTR` nor the `signal` module
    // (`module/mod.rs` gates `pub mod signal` off), so there is no handler for
    // the retry to check for — the same gate `builtins::eintr_retry_with`
    // carries.
    #[cfg(not(target_arch = "wasm32"))]
    if errno == libc::EINTR {
        if let Err(err) = crate::module::signal::interp_signal::checksignals_now() {
            return err;
        }
    }
    let msg = PyError::clean_strerror(errno);
    let Some(w_exc) = w_exception_class else {
        // `space.w_OSError` with both names the caller kept.  The custom-class
        // arm below hands the constructor `w_filename2`, and the default class
        // takes it in the same position, so a two-path syscall keeps its
        // `.filename2` slot and the `" -> 'dest'"` suffix `str(e)` renders.
        return PyError::os_error_syscall2(
            errno,
            w_filename.unwrap_or(pyre_object::PY_NULL),
            w_filename2.unwrap_or(pyre_object::PY_NULL),
        );
    };
    // Pinned, then re-read from the slots: a collection during a later
    // allocation rewrites the root SLOT and not this frame's copy of the
    // pointer.
    let _roots = pyre_object::gc_roots::push_roots();
    let filename = w_filename.unwrap_or_else(pyre_object::w_none);
    let filename2 = w_filename2.unwrap_or_else(pyre_object::w_none);
    let live_base = _roots.pin_roots(&[w_exc, filename, filename2]);
    // `error.py _wrap_oserror2_impl` builds `w_errno` before `w_msg`.
    // Evaluating both allocators in one `pin_roots` leaves the int
    // unrooted across `w_str_new_managed`.
    let extra = _roots.pin_roots(&[pyre_object::w_int_new(errno as i64)]);
    let _ = _roots.pin_roots(&[pyre_object::w_str_new_managed(&msg)]);
    // The five the constructor takes, in `_wrap_oserror2_impl`'s order.  A
    // subclass reading `filename2` sees the argument it declares rather than a
    // short call.
    let args = [
        _roots.get(extra),
        _roots.get(extra + 1),
        _roots.get(live_base + 1),
        pyre_object::w_none(),
        _roots.get(live_base + 2),
    ];
    operation_error_from_instance(
        _roots.get(live_base),
        crate::call::call_function_impl_result(_roots.get(live_base), &args),
        || PyError::os_error_syscall(errno, pyre_object::PY_NULL),
    )
}

/// pypy/interpreter/error.py `wrap_oserror`.
///
/// ```python
/// w_filename = None
/// w_filename2 = None
/// if filename is not None:
///     w_filename = space.newfilename(filename)
///     if filename2 is not None:
///         w_filename2 = space.newfilename(filename2)
/// return wrap_oserror2(space, e, w_filename, w_exception_class,
///                      w_filename2, eintr_retry)
/// ```
///
/// The second name is wrapped only inside the first's arm, so a caller that
/// passes `filename2` alone reports neither — upstream's nesting, kept.
pub fn wrap_oserror(
    space: PyObjectRef,
    error: &(dyn std::error::Error + 'static),
    filename: Option<&str>,
    w_exception_class: Option<PyObjectRef>,
    filename2: Option<&str>,
) -> OperationError {
    let Some(filename) = filename else {
        return wrap_oserror2(space, error, None, w_exception_class, None);
    };
    // The first name is rooted across the second's allocation and read back
    // from its slot afterwards.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = _roots.base();
    let _ = _roots.pin_root(pyre_object::w_str_new_managed(filename));
    let w_filename2 = filename2.map(|name| {
        let _ = _roots.pin_root(pyre_object::w_str_new_managed(name));
        _roots.get(base + 1)
    });
    wrap_oserror2(
        space,
        error,
        Some(_roots.get(base)),
        w_exception_class,
        w_filename2,
    )
}

#[cfg(test)]
mod tests {
    use super::{PyError, PyErrorKind, write_exception};

    #[test]
    fn render_exception_omits_empty_message_separator() {
        let err = PyError::new(PyErrorKind::StopIteration, "");
        assert_eq!(err.render_exception(), "StopIteration");
    }

    #[test]
    fn decompose_valuefmt_keeps_one_more_part_than_formats() {
        // `error.py get_operrcls2` asserts `len(strings) == len(formats) + 1`
        // on what `decompose_valuefmt` hands back — including when a format
        // code sits at either end.
        for valuefmt in [
            "cannot do %s to %s",
            "%s leads",
            "trails %s",
            "%N takes %R and %T",
        ] {
            let (strings, formats) = super::decompose_valuefmt(valuefmt);
            assert_eq!(
                strings.len(),
                formats.len() + 1,
                "{valuefmt:?} -> {strings:?} / {formats:?}"
            );
        }
    }

    #[test]
    fn decompose_valuefmt_rejects_no_format_command() {
        for valuefmt in ["no format codes", ""] {
            let panic = std::panic::catch_unwind(|| super::decompose_valuefmt(valuefmt));
            assert!(panic.is_err(), "accepted {valuefmt:?}");
        }
    }

    #[test]
    #[should_panic(
        expected = "invalid format string (only %8, %N, %R, %S, %T, %d or %s supported)"
    )]
    fn decompose_valuefmt_rejects_an_invalid_code() {
        let _ = super::decompose_valuefmt("cannot use %q");
    }

    #[test]
    fn decompose_valuefmt_folds_a_doubled_percent() {
        let (strings, formats) = super::decompose_valuefmt("100%% of %s");
        assert_eq!(strings, vec!["100% of ".to_string(), String::new()]);
        assert_eq!(formats, vec!["s".to_string()]);
    }

    #[test]
    fn oefmt_renders_format_string_and_arguments() {
        let cases: &[(&str, &str, &[super::FmtArg], &str)] = &[
            (
                "verbatim",
                "dictionary is empty",
                &[],
                "RuntimeError: dictionary is empty",
            ),
            (
                "one_text",
                "cannot add %s",
                &[super::FmtArg::Text("int")],
                "RuntimeError: cannot add int",
            ),
            (
                "two_texts",
                "cannot add %s to %s",
                &[super::FmtArg::Text("int"), super::FmtArg::Text("str")],
                "RuntimeError: cannot add int to str",
            ),
            (
                "scalar",
                "%d %s",
                &[super::FmtArg::Int(-42), super::FmtArg::Text("items")],
                "RuntimeError: -42 items",
            ),
        ];
        for (name, fmt, args, want) in cases {
            let err = super::oefmt(std::ptr::null_mut(), fmt, args);
            assert_eq!(err.render_exception(), *want, "case {name}");
        }
    }

    #[test]
    fn oefmt_takes_a_byte_string_for_percent_s() {
        // `error.py _compute_value`'s final `else`, which
        // `interpreter/test/test_error.py test_oefmt_utf8` exercises: a UTF-8
        // byte string reaches `%s` and reads back as the same text a `str`
        // operand would have produced.
        let text = "abc \u{e0}\u{e8}\u{ec}\u{f2}\u{f9}";
        let from_bytes = super::oefmt(
            std::ptr::null_mut(),
            "%s",
            &[super::FmtArg::Bytes(text.as_bytes())],
        );
        let from_text = super::oefmt(std::ptr::null_mut(), "%s", &[super::FmtArg::Text(text)]);
        assert_eq!(from_bytes.message_wtf8(), from_text.message_wtf8());
        assert_eq!(
            from_bytes.render_exception(),
            format!("RuntimeError: {text}")
        );
    }

    #[test]
    fn oefmt_keeps_a_surrogate_encoded_in_a_byte_string() {
        // WTF-8 for a lone `U+D800`.  `_compute_value` decodes `%8` with
        // `allow_surrogates` and hands `%s` its bytes untouched, so neither
        // conversion replaces this.
        let bytes = [0xEDu8, 0xA0, 0x80];
        let expected = rustpython_wtf8::Wtf8::from_bytes(&bytes)
            .expect("the encoding of a lone surrogate is WTF-8")
            .to_owned();
        for valuefmt in ["%s", "%8"] {
            let err = super::oefmt(
                std::ptr::null_mut(),
                valuefmt,
                &[super::FmtArg::Bytes(&bytes)],
            );
            assert_eq!(err.message_wtf8(), expected, "for {valuefmt}");
        }
    }

    #[test]
    fn oefmt_replaces_invalid_utf8_for_percent_8() {
        let err = super::oefmt(
            std::ptr::null_mut(),
            "bad bytes: %8",
            &[super::FmtArg::Bytes(b"left\xffright")],
        );
        assert_eq!(
            err.render_exception(),
            "RuntimeError: bad bytes: left\u{fffd}right"
        );
    }

    #[test]
    #[should_panic(expected = "oefmt format %d requires Int argument, got Text")]
    fn oefmt_rejects_an_operand_of_the_wrong_kind() {
        let _ = super::oefmt(
            std::ptr::null_mut(),
            "count: %d",
            &[super::FmtArg::Text("not an integer")],
        );
    }

    #[test]
    fn write_exception_includes_traceback_header() {
        let err = PyError::type_error("bad operand");
        let mut out = Vec::new();
        write_exception(&mut out, &err, true).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("Traceback (most recent call last):"));
        assert!(text.contains("TypeError: bad operand"));
    }

    #[test]
    fn traceback_frame_names_module_matches_frozen_and_installed_paths() {
        let cases: &[(&str, &str, &[&str], bool)] = &[
            (
                "frozen_exact",
                "<frozen importlib._bootstrap>",
                &["<frozen importlib._bootstrap>", "importlib/_bootstrap.py"],
                true,
            ),
            (
                "frozen_longer_prefix",
                "<frozen importlib._bootstrap_external>",
                &["<frozen importlib._bootstrap>", "importlib/_bootstrap.py"],
                false,
            ),
            (
                "installed_tail",
                "/usr/lib/python3.14/importlib/_bootstrap.py",
                &["importlib/_bootstrap.py"],
                true,
            ),
            (
                "installed_tail_windows",
                r"C:\Python314\Lib\importlib\_bootstrap.py",
                &["importlib/_bootstrap.py"],
                true,
            ),
            (
                "needs_separator",
                "/app/my_importlib/_bootstrap.py",
                &["importlib/_bootstrap.py"],
                false,
            ),
            (
                "needs_separator_prefix",
                "/app/notimportlib/_bootstrap.py",
                &["importlib/_bootstrap.py"],
                false,
            ),
            (
                "application_frame",
                "/app/main.py",
                &["<frozen importlib._bootstrap>", "importlib/_bootstrap.py"],
                false,
            ),
            ("application_frame_empty_names", "/app/main.py", &[], false),
        ];
        for (name, filename, modules, want) in cases {
            assert_eq!(
                super::traceback_frame_names_module(filename, modules),
                *want,
                "case {name}",
            );
        }
    }

    #[test]
    fn an_unmaterialised_error_reports_no_traceback() {
        // `_application_traceback` starts as `None` upstream, and pyre reads
        // the chain off the exception instance, which a message-only error has
        // not built.  Neither read may allocate one to find that out.
        let err = super::PyError::type_error("bad operand");
        assert!(err.exc_object.is_null());
        assert!(err.get_traceback().is_null());
        assert!(!err.has_any_traceback());
        assert!(!err.got_any_traceback());
        assert!(err.exc_object.is_null(), "the reads must not materialise");
    }

    #[test]
    fn an_unmaterialised_error_answers_async_from_its_kind() {
        // The guard sites ask this on the swallow path of every miss, so it
        // must not build an instance to answer.  `kind` settles both
        // disjuncts before materialisation: nothing unmaterialised is a
        // `KeyboardInterrupt`, which has no `ExcKind` at all.
        let mut ordinary = super::PyError::type_error("bad operand");
        assert!(!ordinary.r#async(pyre_object::PY_NULL));
        assert!(
            ordinary.exc_object.is_null(),
            "the read must not materialise"
        );

        let mut exiting = super::PyError::new(super::PyErrorKind::SystemExit, "0");
        assert!(exiting.r#async(pyre_object::PY_NULL));
        assert!(
            exiting.exc_object.is_null(),
            "the read must not materialise"
        );
    }

    #[test]
    fn attribute_error_with_context_keeps_the_receiver_through_materialisation() {
        crate::typedef::init_typeobjects();
        crate::test_hooks::install_hash_hook();
        let obj = pyre_object::w_list_new(vec![]);
        let mut err = PyError::attribute_error_with_context("missing", obj, "absent");
        assert!(!err.w_obj_context.is_null());
        assert!(!err.w_name_context.is_null());
        let exc = err.to_exc_object();
        assert!(!exc.is_null());
        let stored = unsafe { pyre_object::interp_exceptions::w_exception_get_attr_obj(exc) };
        assert!(!stored.is_null());
        let name = unsafe { pyre_object::interp_exceptions::w_exception_get_name(exc) };
        assert!(!name.is_null());
    }

    /// A collection copies the carrier and writes the new address into the
    /// rooted slot. Reload must follow that slot, so the GC fields
    /// are read from the copy rather than from the pre-move object.
    #[test]
    fn pinned_carrier_reloads_every_gc_field_a_walk_rewrites() {
        let dummy = |addr: usize| addr as pyre_object::PyObjectRef;
        let mut err = PyError::attribute_error("missing");
        err.set_w_type(dummy(0x0400));
        err.set_exc_object(dummy(0x1000));
        err.set_w_name_context(dummy(0x2000));
        err.set_w_obj_context(dummy(0x3000));
        let roots = pyre_object::gc_roots::push_roots();
        let base = err.pin(&roots);
        let handle = err.as_raw();
        // A moving collection copies the object, then stores the new
        // address in the rooted slot. Adding to the pointer in place
        // lands inside the old object and reads the next field.
        let forwarded =
            pyre_object::lltype::malloc_typed_managed(unsafe { std::ptr::read(handle) });
        pyre_object::gc_roots::walk_shadow_stack(|slot| {
            if std::ptr::eq(*slot, handle as pyre_object::PyObjectRef) {
                *slot = forwarded as pyre_object::PyObjectRef;
            }
        });
        err.reload(&roots, base);
        assert_eq!(err.as_raw(), forwarded);
        assert_eq!(err.w_type as usize, 0x0400);
        assert_eq!(err.exc_object as usize, 0x1000);
        assert_eq!(err.w_name_context as usize, 0x2000);
        assert_eq!(err.w_obj_context as usize, 0x3000);
    }

    #[test]
    fn match_compares_w_type_without_materialising() {
        let _ = crate::builtins::new_builtin_module_dict();
        let mut err = super::PyError::type_error("bad operand");
        let type_error = crate::builtins::lookup_exc_class("TypeError").expect("TypeError");
        let value_error = crate::builtins::lookup_exc_class("ValueError").expect("ValueError");
        assert!(err.match_(pyre_object::PY_NULL, type_error));
        assert!(!err.match_(pyre_object::PY_NULL, value_error));
        assert!(
            err.exc_object.is_null(),
            "match_ must not instantiate w_type(w_value)"
        );
    }
}
