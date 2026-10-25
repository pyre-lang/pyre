//! `pypy/module/_cffi_backend/errorbox.py`: on Windows, the text an
//! unraisable cffi callback error writes to `sys.stderr` is also shown in a
//! message box.

use pyre_interpreter::PyError;
use pyre_object::PyObjectRef;

/// `errorbox.py start_error_capture`.
#[cfg(windows)]
#[majit_macros::dont_look_inside]
pub fn start_error_capture() -> Result<PyObjectRef, PyError> {
    win::start_error_capture()
}

/// `errorbox.py stop_error_capture`.
#[cfg(windows)]
#[majit_macros::dont_look_inside]
pub fn stop_error_capture(w_done: PyObjectRef) -> Result<(), PyError> {
    win::stop_error_capture(w_done)
}

/// `errorbox.py start_error_capture` when `MESSAGEBOX` is false.
#[cfg(not(windows))]
pub fn start_error_capture() -> Result<PyObjectRef, PyError> {
    Ok(pyre_object::PY_NULL)
}

/// `errorbox.py stop_error_capture` when `MESSAGEBOX` is false.
#[cfg(not(windows))]
pub fn stop_error_capture(nothing: PyObjectRef) -> Result<(), PyError> {
    let _ = nothing;
    Ok(())
}

#[cfg(windows)]
mod win {
    use super::*;
    use core::ffi::c_void;
    use core::sync::atomic::{AtomicPtr, Ordering};
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{CreateThread, Sleep};
    use windows_sys::Win32::UI::WindowsAndMessaging::{MB_ICONERROR, MB_OK, MessageBoxW};

    // `errorbox.py MODULE`.

    /// `_cffi_bootstrap_text`.
    static CFFI_BOOTSTRAP_TEXT: AtomicPtr<u16> = AtomicPtr::new(core::ptr::null_mut());

    /// `_cffi_errorbox1`.
    fn cffi_errorbox1() -> i32 {
        CFFI_BOOTSTRAP_TEXT
            .compare_exchange(
                core::ptr::null_mut(),
                core::ptr::without_provenance_mut(1),
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .is_ok() as i32
    }

    /// `_cffi_bootstrap_dialog`.
    unsafe extern "system" fn cffi_bootstrap_dialog(_ignored: *mut c_void) -> u32 {
        let caption: Vec<u16> = "PyPy: Python-CFFI error"
            .encode_utf16()
            .chain(core::iter::once(0))
            .collect();
        unsafe {
            // may be interrupted if the whole process is closing
            Sleep(666);
            MessageBoxW(
                core::ptr::null_mut(),
                CFFI_BOOTSTRAP_TEXT.load(Ordering::SeqCst),
                caption.as_ptr(),
                MB_OK | MB_ICONERROR,
            );
        }
        CFFI_BOOTSTRAP_TEXT.store(core::ptr::null_mut(), Ordering::SeqCst);
        0
    }

    /// `_cffi_errorbox`: show a dialog box, but in a background thread, and
    /// never show multiple dialog boxes at once.
    fn cffi_errorbox(text: majit_rlib::rffi::CWCHARP) {
        CFFI_BOOTSTRAP_TEXT.store(text, Ordering::SeqCst);
        unsafe {
            let h = CreateThread(
                core::ptr::null(),
                0,
                Some(cffi_bootstrap_dialog),
                core::ptr::null(),
                0,
                core::ptr::null_mut(),
            );
            if !h.is_null() {
                CloseHandle(h);
            }
        }
    }

    /// `Message.text_p`. `space.fromcache(Message)` makes one per space,
    /// and the product has one space, the process-wide
    /// `baseobjspace::OBJECT_SPACE`, so this static is its `Message`.
    static TEXT_P: AtomicPtr<u16> = AtomicPtr::new(core::ptr::null_mut());

    /// The `Message.start_error_capture` `space.appexec` source, as a
    /// function so that `of` and `fl` are its locals, captured by `write`
    /// and `done`.
    const START_ERROR_CAPTURE: &str = "\
def start_error_capture():
    import sys
    class FileLike:
        def write(self, x):
            try:
                of.write(x)
            except:
                pass
            self.buf += x
    fl = FileLike()
    fl.buf = ''
    of = sys.stderr
    sys.stderr = fl
    def done():
        sys.stderr = of
        return fl.buf
    return done
";

    /// `Message.start_error_capture`.
    pub(super) fn start_error_capture() -> Result<PyObjectRef, PyError> {
        if cffi_errorbox1() != 1 {
            return Ok(pyre_object::PY_NULL);
        }
        // `space.appexec`: define the app-level function, then call it.
        let roots = pyre_object::gc_roots::push_roots();
        let globals_slot = roots.pin_roots(&[pyre_object::dictmultiobject::w_dict_new()]);
        let code = pyre_interpreter::compile::compile_exec(START_ERROR_CAPTURE)
            .map_err(|_| PyError::system_error("could not compile start_error_capture"))?;
        let code_slot = roots.pin_roots(&[
            pyre_interpreter::pycode::box_code_object_with_hidden_applevel(code, true),
        ]);
        pyre_interpreter::builtins::exec_or_eval(
            roots.get(code_slot),
            roots.get(globals_slot),
            pyre_object::PY_NULL,
            false,
            pyre_object::PY_NULL,
        )?;
        let w_func = unsafe {
            pyre_object::dictmultiobject::w_dict_getitem_str(
                roots.get(globals_slot),
                "start_error_capture",
            )
        }
        .expect("the source defines start_error_capture");
        pyre_interpreter::call::call_function_impl_result(w_func, &[])
    }

    /// `Message.stop_error_capture`.
    pub(super) fn stop_error_capture(w_done: PyObjectRef) -> Result<(), PyError> {
        if w_done.is_null() {
            return Ok(());
        }
        let roots = pyre_object::gc_roots::push_roots();
        let text_slot = roots.pin_roots(&[pyre_interpreter::call::call_function_impl_result(
            w_done,
            &[],
        )?]);
        // `len_w` first: `text_wtf8_w` borrows the str's storage. The
        // internal (WTF-8) bytes keep lone surrogates, as `space.utf8_w`.
        let len = pyre_interpreter::baseobjspace::len_w(roots.get(text_slot))? as usize;
        let utf8 = pyre_interpreter::baseobjspace::text_wtf8_w(roots.get(text_slot))?;
        let p = majit_rlib::rffi::utf82wcharp(utf8.as_bytes(), len, false).map_err(|e| {
            pyre_interpreter::error::get_converted_unexpected_exception(pyre_object::PY_NULL, &e)
        })?;
        let old = TEXT_P.swap(p, Ordering::SeqCst);
        if !old.is_null() {
            unsafe { majit_rlib::rffi::free_wcharp(old, false) };
        }
        // `self.text_p = p` keeps the buffer alive for the dialog thread.
        cffi_errorbox(p);
        Ok(())
    }
}
