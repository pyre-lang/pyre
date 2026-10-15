//! Callbacks — PyPy: `pypy/module/_cffi_backend/ccallback.py`.

use majit_jitcode::rffi::{RFFI_ALT_ERRNO, RFFI_ERR_ALL};
use pyre_interpreter::{PyError, PyErrorKind};
use pyre_object::PyObjectRef;

use super::cdataobj::{self, W_CData};
use super::ctypeobj::{self, W_CType};

/// The libffi surface a callback needs, which is a dependency only where the
/// crate declares it; `ctypefunc.rs` splits its `cif` module the same way.
#[cfg(any(unix, windows))]
mod ffi {
    use majit_rlib::clibffi::{self, ffi_arg};

    /// The width libffi widens a return value narrower than a word to.
    pub const SIZE_OF_FFI_ARG: usize = std::mem::size_of::<ffi_arg>();

    /// # Safety
    /// `ptr` is null or a closure `c_ffi_closure_alloc` returned and nothing
    /// else holds it.
    pub unsafe fn closure_free(ptr: *mut libc::c_void) {
        if !ptr.is_null() {
            unsafe { clibffi::c_ffi_closure_free(ptr) };
        }
    }
}

#[cfg(not(any(unix, windows)))]
mod ffi {
    /// `ffi_arg` is a machine word on every target libffi defines it for, and
    /// the result encoder below is shared with those targets.
    pub const SIZE_OF_FFI_ARG: usize = std::mem::size_of::<usize>();

    /// # Safety
    /// No closure can be allocated here, so `ptr` is always null.
    pub unsafe fn closure_free(_ptr: *mut libc::c_void) {}
}

use ffi::SIZE_OF_FFI_ARG;

/// Raw state owned by one callback.  Keeping it out of [`W_CData`] leaves the
/// common cdata layout unchanged; the callback flavor stores this block's
/// address in its otherwise-unused `length` slot.
pub(crate) struct CallbackSideBlock {
    raw_closure: *mut libc::c_void,
    raw_code: *mut libc::c_void,
    error_bytes: Vec<u8>,
}

/// Release the raw `Closure` owner from `W_CDataCallback`'s sweep destructor.
///
/// # Safety
/// `raw` is zero or the unique address returned by `Box::into_raw` for a
/// [`CallbackSideBlock`].
pub(crate) unsafe fn free_callback_side_block(raw: i64, code: *mut u8) {
    if raw == 0 {
        return;
    }
    let side = unsafe { Box::from_raw(raw as usize as *mut CallbackSideBlock) };
    debug_assert_eq!(side.raw_code.cast::<u8>(), code);
    unsafe { ffi::closure_free(side.raw_closure) };
}

fn callback_arg(w_callback: PyObjectRef) -> Result<&'static mut W_CData, PyError> {
    W_CData::from_obj(w_callback)
        .filter(|cdata| cdata.flavor == cdataobj::FLAVOR_CALLBACK)
        .ok_or_else(|| PyError::system_error("callback payload is not a W_CDataCallback"))
}

fn side_block(callback: &W_CData) -> &'static CallbackSideBlock {
    unsafe { &*(callback.length as usize as *const CallbackSideBlock) }
}

struct UnpublishedClosure(*mut libc::c_void);

impl Drop for UnpublishedClosure {
    fn drop(&mut self) {
        unsafe { ffi::closure_free(self.0) };
    }
}

/// `ccallback.py make_callback` and `W_CDataCallback.__init__`.
#[cfg(any(unix, windows))]
pub fn make_callback(
    w_ctype: PyObjectRef,
    w_callable: PyObjectRef,
    w_error: PyObjectRef,
    w_onerror: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for &value in &[w_ctype, w_callable, w_error, w_onerror] {
        let _ = roots.pin_root(value);
    }

    let mut raw_code = std::ptr::null_mut();
    let raw_closure = unsafe {
        majit_rlib::clibffi::c_ffi_closure_alloc(
            std::mem::size_of::<majit_rlib::clibffi::ffi_closure>(),
            &mut raw_code,
        )
    };
    if raw_closure.is_null() || raw_code.is_null() {
        if !raw_closure.is_null() {
            unsafe { majit_rlib::clibffi::c_ffi_closure_free(raw_closure) };
        }
        return Err(PyError::new(
            PyErrorKind::MemoryError,
            "libffi failed to allocate a callback closure",
        ));
    }
    let mut unpublished = UnpublishedClosure(raw_closure);

    let functype = function_ctype(roots.get(base))?;
    if !pyre_interpreter::baseobjspace::callable_w(roots.get(base + 1)) {
        return Err(PyError::type_error(format!(
            "expected a callable object, not {}",
            pyre_interpreter::type_methods::arg_type_name(roots.get(base + 1))
        )));
    }
    let onerror = roots.get(base + 3);
    let has_onerror = if unsafe { pyre_object::is_none(onerror) } {
        false
    } else {
        if !pyre_interpreter::baseobjspace::callable_w(onerror) {
            return Err(PyError::type_error(format!(
                "expected a callable object for 'onerror', not {}",
                pyre_interpreter::type_methods::arg_type_name(onerror)
            )));
        }
        true
    };

    let fresult = ctypeobj::ctype_arg(functype.ctitem)?;
    let error_size = if fresult.size < 0 {
        0
    } else if fresult.has(ctypeobj::CTypeFlags::PRIMITIVE_INTEGER)
        && (fresult.size as usize) < SIZE_OF_FFI_ARG
    {
        SIZE_OF_FFI_ARG
    } else {
        fresult.size as usize
    };
    let mut error_bytes = vec![0u8; error_size];
    if unsafe { !pyre_object::is_none(roots.get(base + 2)) } {
        unsafe {
            convert_from_object_fficallback(fresult, error_bytes.as_mut_ptr(), roots.get(base + 2))?
        };
    }

    let side = Box::new(CallbackSideBlock {
        raw_closure,
        raw_code,
        error_bytes,
    });
    let raw_side = Box::into_raw(side);
    unpublished.0 = std::ptr::null_mut();
    let w_callback = cdataobj::new_cdata_callback(
        raw_code as usize,
        roots.get(base),
        roots.get(base + 1),
        if has_onerror {
            roots.get(base + 3)
        } else {
            pyre_object::PY_NULL
        },
        raw_side as usize as i64,
    );
    let callback_slot = base + 4;
    let _ = roots.pin_root(w_callback);

    let functype = function_ctype(callback_arg(roots.get(callback_slot))?.ctype)?;
    if functype.cif_descr == 0 {
        return Err(PyError::not_implemented(format!(
            "{}: callback with unsupported argument or return type or with '...'",
            functype.name()
        )));
    }
    let userdata = super::hide_reveal::hide_callback(roots.get(callback_slot));
    let status = unsafe {
        majit_rlib::clibffi::c_ffi_prep_closure_loc(
            raw_closure.cast(),
            functype.cif_descr as majit_rlib::clibffi::FFI_CIFP,
            invoke_callback,
            userdata.cast(),
            raw_code,
        )
    };
    if status != majit_rlib::clibffi::FFI_OK {
        return Err(PyError::system_error(
            "libffi failed to build this callback",
        ));
    }
    if unsafe {
        (*raw_closure.cast::<majit_rlib::clibffi::ffi_closure>()).user_data != userdata.cast()
    } {
        return Err(PyError::system_error(
            "ffi_prep_closure(): bad user_data (it seems that the version of the libffi library seen at runtime is different from the 'ffi.h' file seen at compile-time)",
        ));
    }
    Ok(roots.get(callback_slot))
}

#[cfg(not(any(unix, windows)))]
pub fn make_callback(
    _w_ctype: PyObjectRef,
    _w_callable: PyObjectRef,
    _w_error: PyObjectRef,
    _w_onerror: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    Err(PyError::not_implemented(
        "this platform has no libffi, so a callback cannot be built",
    ))
}

fn function_ctype(w_ctype: PyObjectRef) -> Result<&'static mut W_CType, PyError> {
    let ctype = ctypeobj::ctype_arg(w_ctype)?;
    if ctype.kind != ctypeobj::KIND_FUNC {
        return Err(PyError::type_error("expected a function ctype"));
    }
    Ok(ctype)
}

/// `W_ExternPython.prepare_args_tuple` for a libffi-decoded callback.
unsafe fn prepare_args_tuple(
    w_callback: PyObjectRef,
    ll_args: *mut *mut libc::c_void,
) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    let _ = roots.pin_root(w_callback);
    let functype = function_ctype(callback_arg(roots.get(base))?.ctype)?;
    let fargs = super::ctypefunc::fargs_of(functype);
    let fargs_base = base + 1;
    for &w_farg in &fargs {
        let _ = roots.pin_root(w_farg);
    }
    let args_base = fargs_base + fargs.len();
    for i in 0..fargs.len() {
        let farg = ctypeobj::ctype_arg(roots.get(fargs_base + i))?;
        let ll_arg = unsafe { ll_args.add(i).read() }.cast::<u8>();
        let w_arg = unsafe { ctypeobj::convert_to_object(farg, ll_arg as usize)? };
        let _ = roots.pin_root(w_arg);
    }
    let mut args_w = Vec::with_capacity(fargs.len());
    for i in 0..fargs.len() {
        args_w.push(roots.get(args_base + i));
    }
    Ok(pyre_object::w_tuple_new(args_w))
}

unsafe fn convert_result(
    w_callback: PyObjectRef,
    ll_res: *mut u8,
    w_res: PyObjectRef,
) -> Result<(), PyError> {
    let functype = function_ctype(callback_arg(w_callback)?.ctype)?;
    let fresult = ctypeobj::ctype_arg(functype.ctitem)?;
    unsafe { convert_from_object_fficallback(fresult, ll_res, w_res) }
}

/// `convert_from_object_fficallback(..., encode_result_for_libffi=True)`.
unsafe fn convert_from_object_fficallback(
    fresult: &W_CType,
    mut ll_res: *mut u8,
    mut w_res: PyObjectRef,
) -> Result<(), PyError> {
    if fresult.kind == ctypeobj::KIND_VOID {
        if unsafe { !pyre_object::is_none(w_res) } {
            return Err(PyError::type_error(
                "callback with the return type 'void' must return None",
            ));
        }
        return Ok(());
    }
    let small_result = fresult.size >= 0 && (fresult.size as usize) < SIZE_OF_FFI_ARG;
    if small_result && fresult.has(ctypeobj::CTypeFlags::PRIMITIVE_INTEGER) {
        if fresult.kind == ctypeobj::KIND_PRIM_SIGNED && !fresult.has(ctypeobj::CTypeFlags::ENUM) {
            unsafe {
                pyre_object::with_roots!(w_res => ctypeobj::convert_from_object(fresult, ll_res as usize, w_res))?
            };
            let value = super::misc::as_long(w_res)?;
            return unsafe {
                super::misc::write_raw_signed_data(ll_res as usize, value, SIZE_OF_FFI_ARG as i64)
            };
        }
        unsafe { std::ptr::write_bytes(ll_res, 0, SIZE_OF_FFI_ARG) };
        if cfg!(target_endian = "big") {
            ll_res = unsafe { ll_res.add(SIZE_OF_FFI_ARG - fresult.size as usize) };
        }
    }
    unsafe { ctypeobj::convert_from_object(fresult, ll_res as usize, w_res) }
}

fn write_error_return_value(callback: &W_CData, ll_res: *mut u8) {
    if ll_res.is_null() {
        return;
    }
    let error = &side_block(callback).error_bytes;
    unsafe { std::ptr::copy_nonoverlapping(error.as_ptr(), ll_res, error.len()) };
}

fn print_error(
    w_callback: PyObjectRef,
    error: &mut PyError,
    extra_line: &str,
) -> Result<(), PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let error_slot = error.pin(&roots);
    let callback_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_callback);
    let keepalive = callback_arg(roots.get(callback_slot))?.w_keepalive;
    let keepalive_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(keepalive);
    let w_repr = pyre_interpreter::builtins::builtin_repr(&[roots.get(keepalive_slot)])?;
    let repr_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_repr);
    let repr = unsafe { pyre_object::unicodeobject::w_str_get_wtf8(roots.get(repr_slot)) };
    let where_desc = if extra_line.is_empty() {
        pyre_interpreter::display::wtf8_format!("Exception ignored from cffi callback ", repr)
    } else {
        pyre_interpreter::display::wtf8_format!(
            "Exception ignored from cffi callback ",
            repr,
            extra_line
        )
    };
    error.reload(&roots, error_slot);
    error.write_unraisable(pyre_object::w_none(), &where_desc, pyre_object::PY_NULL);
    Ok(())
}

fn handle_applevel_exception(
    w_callback: PyObjectRef,
    error: PyError,
    ll_res: *mut u8,
    extra_line: &str,
) {
    let roots = pyre_object::gc_roots::push_roots();
    let mut error = error;
    let error_slot = error.pin(&roots);
    let callback_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_callback);
    // `pin_root` is a safepoint. `expand_pop_roots` reloads the carrier
    // before the next read; slots taken after the pin do not alias it.
    error.reload(&roots, error_slot);
    write_error_return_value(
        callback_arg(roots.get(callback_slot)).expect("root remains a callback"),
        ll_res,
    );
    let w_onerror = callback_arg(roots.get(callback_slot))
        .expect("root remains a callback")
        .w_destructor;
    if w_onerror.is_null() {
        let _ = print_error(roots.get(callback_slot), &mut error, extra_line);
        return;
    }
    let onerror_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_onerror);
    error.reload(&roots, error_slot);
    let w_value = match error.normalize_exception(pyre_object::w_none()) {
        Ok(value) => value,
        Err(mut normalization_error) => {
            let _ = print_error(
                roots.get(callback_slot),
                &mut normalization_error,
                extra_line,
            );
            return;
        }
    };
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_value);
    let w_type = pyre_interpreter::baseobjspace::exception_getclass(roots.get(value_slot));
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_type);
    error.reload(&roots, error_slot);
    let mut w_tb = error.get_traceback();
    if w_tb.is_null() {
        // A raise that never entered `handle_operation_error` still
        // owes `onerror` a traceback. Stamp the live callback frame.
        let frame = pyre_interpreter::eval::current_frame();
        if !frame.is_null() && !error.exc_object.is_null() {
            unsafe {
                pyre_interpreter::pytraceback::record_application_traceback(
                    error.exc_object,
                    frame,
                    (*frame).last_instr as i64,
                );
            }
            error.reload(&roots, error_slot);
            w_tb = error.get_traceback();
        }
        if w_tb.is_null() {
            w_tb = pyre_object::w_none();
        }
    }
    let tb_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_tb);
    match pyre_interpreter::call::call_function_impl_result(
        roots.get(onerror_slot),
        &[
            roots.get(type_slot),
            roots.get(value_slot),
            roots.get(tb_slot),
        ],
    ) {
        Ok(w_res) if unsafe { !pyre_object::is_none(w_res) } => {
            let res_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(w_res);
            if let Err(conversion_error) =
                unsafe { convert_result(roots.get(callback_slot), ll_res, roots.get(res_slot)) }
            {
                let mut conversion_error = conversion_error;
                let conversion_error_slot = conversion_error.pin(&roots);
                error.reload(&roots, error_slot);
                let _ = print_error(roots.get(callback_slot), &mut error, extra_line);
                conversion_error.reload(&roots, conversion_error_slot);
                print_onerror_exception(conversion_error);
            }
        }
        Ok(_) => {}
        Err(onerror_error) => {
            let mut onerror_error = onerror_error;
            let onerror_error_slot = onerror_error.pin(&roots);
            error.reload(&roots, error_slot);
            let _ = print_error(roots.get(callback_slot), &mut error, extra_line);
            onerror_error.reload(&roots, onerror_error_slot);
            print_onerror_exception(onerror_error);
        }
    }
}

fn print_onerror_exception(mut error: PyError) {
    let roots = pyre_object::gc_roots::push_roots();
    let error_slot = error.pin(&roots);
    let w_value = error
        .normalize_exception(pyre_object::w_none())
        .unwrap_or_else(|_| {
            error.reload(&roots, error_slot);
            error.to_exc_object()
        });
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_value);
    let w_type = pyre_interpreter::baseobjspace::exception_getclass(roots.get(value_slot));
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_type);
    error.reload(&roots, error_slot);
    let mut w_tb = error.get_traceback();
    if w_tb.is_null() {
        w_tb = pyre_object::w_none();
    }
    let tb_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_tb);
    PyError::write_unraisable_default(
        pyre_object::w_none(),
        roots.get(type_slot),
        roots.get(value_slot),
        roots.get(tb_slot),
        rustpython_wtf8::Wtf8::new(""),
        pyre_object::w_none(),
        "\nDuring the call to 'onerror', another exception occurred:\n\n",
    );
}

unsafe fn do_invoke(w_callback: PyObjectRef, ll_res: *mut u8, ll_args: *mut *mut libc::c_void) {
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    let _ = roots.pin_root(w_callback);
    let w_args = match unsafe { prepare_args_tuple(roots.get(base), ll_args) } {
        Ok(args) => args,
        Err(error) => {
            handle_applevel_exception(roots.get(base), error, ll_res, "");
            return;
        }
    };
    let _ = roots.pin_root(w_args);
    let _ = roots.pin_root(
        callback_arg(roots.get(base))
            .expect("root remains a callback")
            .w_keepalive,
    );
    let arguments =
        match pyre_interpreter::argument::Arguments::frompacked(Some(roots.get(base + 1)), None) {
            Ok(arguments) => arguments,
            Err(error) => {
                handle_applevel_exception(roots.get(base), error, ll_res, "");
                return;
            }
        };
    let w_res = match pyre_interpreter::baseobjspace::call_args(roots.get(base + 2), &arguments) {
        Ok(result) => result,
        Err(error) => {
            handle_applevel_exception(roots.get(base), error, ll_res, "");
            return;
        }
    };
    let _ = roots.pin_root(w_res);
    if let Err(error) = unsafe { convert_result(roots.get(base), ll_res, roots.get(base + 3)) } {
        handle_applevel_exception(
            roots.get(base),
            error,
            ll_res,
            ", trying to convert the result back to C",
        );
    }
}

#[cfg(any(unix, windows))]
unsafe extern "C" fn invoke_callback(
    _ffi_cif: majit_rlib::clibffi::FFI_CIFP,
    ll_res: majit_rlib::rffi::VOIDP,
    ll_args: majit_rlib::clibffi::VOIDPP,
    ll_userdata: majit_rlib::rffi::VOIDP,
) {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    super::cerrno::_errno_after(RFFI_ERR_ALL | RFFI_ALT_ERRNO);
    if !ll_res.is_null() {
        unsafe { std::ptr::write_bytes(ll_res.cast::<u8>(), 0, SIZE_OF_FFI_ARG) };
    }
    let invoked = catch_unwind(AssertUnwindSafe(|| {
        let _callback =
            pyre_interpreter::module::thread::enter_external_callback_from_foreign_thread();
        if let Some(w_callback) = super::hide_reveal::reveal_callback(ll_userdata.cast::<u8>()) {
            let roots = pyre_object::gc_roots::push_roots();
            let base = roots.base();
            let _ = roots.pin_root(w_callback);
            unsafe { do_invoke(roots.get(base), ll_res.cast::<u8>(), ll_args) };
        } else {
            pyre_interpreter::host_seam::emit_stderr(
                b"SystemError: invoking a callback that was already freed\n",
            );
        }
    }));
    if invoked.is_err() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let _callback =
                pyre_interpreter::module::thread::enter_external_callback_from_foreign_thread();
            if let Some(w_callback) = super::hide_reveal::reveal_callback(ll_userdata.cast::<u8>())
                && let Ok(callback) = callback_arg(w_callback)
            {
                write_error_return_value(callback, ll_res.cast::<u8>());
            }
        }));
    }
    super::cerrno::_errno_before(RFFI_ERR_ALL | RFFI_ALT_ERRNO);
}
