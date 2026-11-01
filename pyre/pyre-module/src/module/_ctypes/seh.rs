//! Trapping the machine-level exceptions a foreign call raises.
//!
//! A structured exception is delivered by the OS through a chain of frame
//! handlers, not by anything the language runtime unwinds, so a foreign
//! function that reads address 0x20 does not return an error to its caller —
//! it takes the process down unless a `__try`/`__except` further out claims
//! it.  `_ctypes_callproc` has that `__try` around `ffi_call`
//! (`callproc.c:950-959`); [`guard`] is the same fence with the call it wraps
//! written in Rust, and [`exception`] is `SetException`.
//!
//! Everywhere else the fence is nothing: a POSIX fault arrives as a signal and
//! the same call raises no structured exception at all.

/// Run `body` inside the fence, reporting a structured exception it raised as
/// the OSError `SetException` would have set.
#[cfg(not(all(windows, target_env = "msvc")))]
pub(super) fn guard<T, F: FnOnce() -> T>(body: F) -> Result<T, pyre_interpreter::PyError> {
    Ok(body())
}

#[cfg(all(windows, target_env = "msvc"))]
pub(super) use msvc::{access_violation_reading, guard};

#[cfg(all(windows, target_env = "msvc"))]
mod msvc {
    use std::ffi::c_void;

    /// What the filter in `seh.c` copies out of the `EXCEPTION_RECORD`.
    #[repr(C)]
    #[derive(Default)]
    struct Record {
        code: u32,
        info: [u64; 2],
        ninfo: u32,
    }

    #[cfg(all(windows, target_env = "msvc"))]
    majit_rlib::rffi::external_compilation_info! {
        const SEH_ECI = {
            separate_module_files: ["src/module/_ctypes/seh.c"],
        };
    }

    // `seh.c` — `__try { body(arg); } __except (filter) {}`, returning 1
    // when the filter took an exception and filling `out` with it.
    #[cfg(all(windows, target_env = "msvc"))]
    majit_rlib::rffi::llexternal!(
        pyre_seh_guard = "pyre_seh_guard",
        [unsafe extern "C" fn(*mut c_void), *mut c_void, *mut Record],
        i32,
        compilation_info = SEH_ECI,
        _nowrapper = true,
        sandboxsafe = true
    );

    /// The exception the fence would have made of a read at `address` that
    /// faulted, for a read that is refused before it is attempted rather than
    /// one the fence caught: nothing dereferences the address, so no
    /// structured exception is ever raised to be filtered.
    pub(in super::super) fn access_violation_reading(address: usize) -> pyre_interpreter::PyError {
        exception(&Record {
            code: rustpython_host_env::faulthandler::EXCEPTION_ACCESS_VIOLATION,
            info: [0, address as u64],
            ninfo: 2,
        })
    }

    pub(in super::super) fn guard<T, F: FnOnce() -> T>(
        body: F,
    ) -> Result<T, pyre_interpreter::PyError> {
        // The fence takes one plain function pointer, so the closure and its
        // result travel in a cell this frame owns.  `__except` unwinds
        // everything above `pyre_seh_guard` and leaves that frame — and this
        // one — standing, so the cell is still readable afterwards; what the
        // unwound frames held is not, which is why nothing is carried out of
        // the body except through the cell.
        struct Cell<F, T> {
            body: Option<F>,
            out: Option<T>,
        }

        unsafe extern "C" fn trampoline<F: FnOnce() -> T, T>(arg: *mut c_void) {
            let cell = unsafe { &mut *arg.cast::<Cell<F, T>>() };
            let body = cell.body.take().expect("the fence calls its body once");
            cell.out = Some(body());
        }

        let mut cell = Cell {
            body: Some(body),
            out: None,
        };
        let mut record = Record::default();
        let raised = unsafe {
            pyre_seh_guard(
                trampoline::<F, T>,
                std::ptr::from_mut(&mut cell).cast(),
                &mut record,
            )
        };
        match cell.out {
            Some(value) if raised == 0 => Ok(value),
            _ => Err(exception(&record)),
        }
    }

    /// `SetException` (`callproc.c:290-441`).  A structured exception code is
    /// a Win32 error code and most of them are left to
    /// `PyErr_SetFromWindowsErr`; the ones named here either carry
    /// information the code alone does not, or describe a fault the system
    /// message does not.
    fn exception(record: &Record) -> pyre_interpreter::PyError {
        use rustpython_host_env::faulthandler as fh;

        let code = record.code as i32;
        let message = match code {
            x if x == fh::EXCEPTION_ACCESS_VIOLATION as i32 => {
                // `ExceptionInformation[0]` is the access that faulted and
                // `[1]` the address it named.  `%p` is ill-defined, so
                // `PyUnicode_FromFormat` puts the `0x` in front of it itself
                // and leaves the digits as the CRT printed them — as wide as
                // a pointer, in upper case.
                let verb = if record.info[0] == 0 {
                    "reading"
                } else {
                    "writing"
                };
                format!(
                    "exception: access violation {verb} 0x{:0width$X}",
                    record.info[1],
                    width = size_of::<usize>() * 2
                )
            }
            x if x == fh::EXCEPTION_BREAKPOINT as i32 => {
                "exception: breakpoint encountered".to_string()
            }
            x if x == fh::EXCEPTION_DATATYPE_MISALIGNMENT as i32 => {
                "exception: datatype misalignment".to_string()
            }
            x if x == fh::EXCEPTION_SINGLE_STEP as i32 => "exception: single step".to_string(),
            x if x == fh::EXCEPTION_ARRAY_BOUNDS_EXCEEDED as i32 => {
                "exception: array bounds exceeded".to_string()
            }
            x if x == fh::EXCEPTION_FLT_DENORMAL_OPERAND as i32 => {
                "exception: floating-point operand denormal".to_string()
            }
            x if x == fh::EXCEPTION_FLT_DIVIDE_BY_ZERO as i32 => {
                "exception: float divide by zero".to_string()
            }
            x if x == fh::EXCEPTION_FLT_INEXACT_RESULT as i32 => {
                "exception: float inexact".to_string()
            }
            x if x == fh::EXCEPTION_FLT_INVALID_OPERATION as i32 => {
                "exception: float invalid operation".to_string()
            }
            x if x == fh::EXCEPTION_FLT_OVERFLOW as i32 => "exception: float overflow".to_string(),
            x if x == fh::EXCEPTION_FLT_STACK_CHECK as i32 => {
                "exception: stack over/underflow".to_string()
            }
            x if x == fh::EXCEPTION_STACK_OVERFLOW as i32 => {
                "exception: stack overflow".to_string()
            }
            x if x == fh::EXCEPTION_FLT_UNDERFLOW as i32 => {
                "exception: float underflow".to_string()
            }
            x if x == fh::EXCEPTION_INT_DIVIDE_BY_ZERO as i32 => {
                "exception: integer divide by zero".to_string()
            }
            x if x == fh::EXCEPTION_INT_OVERFLOW as i32 => {
                "exception: integer overflow".to_string()
            }
            x if x == fh::EXCEPTION_PRIV_INSTRUCTION as i32 => {
                "exception: privileged instruction".to_string()
            }
            x if x == fh::EXCEPTION_NONCONTINUABLE_EXCEPTION as i32 => {
                "exception: nocontinuable".to_string()
            }
            _ => {
                return pyre_interpreter::PyError::os_error_win32_syscall2(
                    code,
                    pyre_object::PY_NULL,
                    pyre_object::PY_NULL,
                );
            }
        };
        pyre_interpreter::PyError::os_error(message)
    }
}
