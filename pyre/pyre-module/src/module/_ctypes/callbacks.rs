//! C-callable callback thunks for `_CFuncPtr` instances.

use pyre_object::PyObjectRef;

mod imp {
    use super::PyObjectRef;
    use crate::module::_ctypes::callproc::{CTypeLayout, CallbackThunk, FfiType};
    use crate::module::_ctypes::{callproc, cdata, cfield, funcptr};
    use core::ffi::c_void;
    use majit_rlib::clibffi::{self, LibFFIError, LibFFIErrorKind, UserCallback, UserData, VOIDPP};
    use majit_rlib::rffi::{INT, VOIDP};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    const THUNK_KEY: &str = "_thunk";

    struct ThunkUserdata {
        /// GC-rooted slot holding `*mut Weakref` to the owning `CFuncPtr`.
        slot: *mut usize,
        result_width: usize,
    }

    struct StoredThunk {
        #[allow(dead_code)]
        thunk: CallbackThunk,
        #[allow(dead_code)]
        userdata: Box<ThunkUserdata>,
        slot: Box<usize>,
        rooted: bool,
    }

    impl Drop for StoredThunk {
        fn drop(&mut self) {
            if self.rooted {
                let slot_ptr =
                    std::ptr::from_mut(&mut *self.slot).cast::<pyre_object::gc_hook::GCREF>();
                pyre_object::gc_hook::try_gc_remove_root(slot_ptr);
                self.rooted = false;
            }
        }
    }

    pub(super) fn build_thunk(
        obj: PyObjectRef,
    ) -> Result<Option<usize>, pyre_interpreter::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        let current_obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
        let argtypes = funcptr::resolve_argtypes(current_obj()).unwrap_or_default();
        let argtypes_slot = pyre_object::gc_roots::pin_roots(&argtypes);
        let ffi_arg_types = (0..argtypes.len())
            .map(|i| ffi_type_for_arg(pyre_object::gc_roots::shadow_stack_get(argtypes_slot + i)))
            .collect::<Result<Vec<_>, _>>()?;
        let restype =
            funcptr::resolve_restype(current_obj()).map_err(|_| invalid_callback_result_type())?;
        let ffi_res_type = ffi_type_for_ret(&restype)?;
        let result_width = ffi_result_slot_width(&restype);
        let flags = funcptr::funcptr_flags(current_obj()) as INT;

        let wref = unsafe { pyre_object::weakref::w_weakref_new(current_obj()) };
        let mut slot = Box::new(wref as usize);
        let slot_ptr = std::ptr::from_mut(&mut *slot).cast::<usize>();
        let rooted = unsafe {
            pyre_object::gc_hook::try_gc_add_root(slot_ptr.cast::<pyre_object::gc_hook::GCREF>())
        };

        let userdata = Box::new(ThunkUserdata {
            slot: slot_ptr,
            result_width,
        });
        let additional_arg = std::ptr::from_ref(&*userdata) as isize;
        let thunk = CallbackThunk::new(
            ffi_arg_types,
            ffi_res_type,
            thunk_user_callback,
            additional_arg,
            flags,
        )
        .map_err(map_libffi_error)?;
        let code = thunk.code_ptr();

        let stored = Box::new(StoredThunk {
            thunk,
            userdata,
            slot,
            rooted,
        });
        let stored_ptr = Box::into_raw(stored) as usize;
        store_thunk_ptr(current_obj(), stored_ptr)?;
        pyre_interpreter::executioncontext::register_finalizer(current_obj());
        Ok(Some(code))
    }

    pub(super) fn free_thunk(obj: PyObjectRef) {
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = _roots.pin_roots(&[obj]);
        let Some(stored) = funcptr::instance_get(_roots.get(obj_slot), THUNK_KEY) else {
            return;
        };
        if !unsafe { pyre_object::is_int(stored) } {
            return;
        }
        let ptr = unsafe { pyre_object::w_int_get_value(stored) } as usize;
        if ptr == 0 {
            return;
        }
        funcptr::clear_thunk_ptr(_roots.get(obj_slot));
        unsafe { drop(Box::from_raw(ptr as *mut StoredThunk)) };
    }

    fn store_thunk_ptr(obj: PyObjectRef, ptr: usize) -> Result<(), pyre_interpreter::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        let w_ptr = pyre_object::w_int_new(ptr as i64);
        let ptr_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_ptr);
        let d = pyre_interpreter::baseobjspace::getdict_native(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
        );
        if d.is_null() {
            return Err(pyre_interpreter::PyError::type_error(
                "CFuncPtr instance has no instance dict",
            ));
        }
        unsafe {
            pyre_object::w_dict_setitem_str(
                d,
                THUNK_KEY,
                pyre_object::gc_roots::shadow_stack_get(ptr_slot),
            )
        };
        Ok(())
    }

    fn map_libffi_error(err: LibFFIError) -> pyre_interpreter::PyError {
        match err.kind {
            LibFFIErrorKind::ClosureAlloc => pyre_interpreter::PyError::memory_error(""),
            LibFFIErrorKind::PrepClosure => pyre_interpreter::PyError::runtime_error(format!(
                "ffi_prep_closure failed with {}",
                err.status
            )),
            LibFFIErrorKind::PrepCif | LibFFIErrorKind::PrepCifVar => {
                pyre_interpreter::PyError::runtime_error(format!(
                    "ffi_prep_cif failed with {}",
                    err.status
                ))
            }
        }
    }

    fn ffi_type_for_arg(at: PyObjectRef) -> Result<FfiType, pyre_interpreter::PyError> {
        if funcptr::argtype_is_pointer_kind(at) {
            return Ok(callproc::ffi_pointer_type());
        }
        if let Some(tc) = cdata::type_code_of(at)
            && let Some(ty) = callproc::ffi_type_from_code(&tc)
        {
            return Ok(ty);
        }
        ffi_type_for_layout(&funcptr::build_layout(at)?)
    }

    fn ffi_type_for_layout(layout: &CTypeLayout) -> Result<FfiType, pyre_interpreter::PyError> {
        layout.to_ffi_type().map_err(|e| match e {
            callproc::CallError::UnknownTypeCode(_) => {
                pyre_interpreter::PyError::type_error("unknown callback argument type")
            }
            other => funcptr::call_error_to_py(other),
        })
    }

    fn ffi_result_width(ret: &funcptr::Ret) -> usize {
        match ret {
            funcptr::Ret::Void => 0,
            funcptr::Ret::Code(code) if code == "g" => clibffi::longdouble_size(),
            funcptr::Ret::Code(code) => cfield::simple_type_size(code).unwrap_or(0),
            funcptr::Ret::Pointer(_) => cfield::pointer_size(),
            funcptr::Ret::Aggregate(_) => 0,
        }
    }

    const FFI_ARG_SIZE: usize = std::mem::size_of::<clibffi::ffi_arg>();

    fn is_float_code(code: &str) -> bool {
        matches!(code, "f" | "d" | "g")
    }

    fn ffi_result_slot_width(ret: &funcptr::Ret) -> usize {
        let width = ffi_result_width(ret);
        match ret {
            funcptr::Ret::Code(code) if !is_float_code(code) => width.max(FFI_ARG_SIZE),
            _ => width,
        }
    }

    fn ffi_result_offset(ret: &funcptr::Ret, n: usize) -> usize {
        if cfg!(target_endian = "big") {
            ffi_result_slot_width(ret).saturating_sub(n)
        } else {
            0
        }
    }

    fn ffi_type_for_ret(ret: &funcptr::Ret) -> Result<FfiType, pyre_interpreter::PyError> {
        match ret {
            funcptr::Ret::Void => Ok(callproc::ffi_void_type()),
            funcptr::Ret::Code(c) => {
                callproc::ffi_type_from_code(c).ok_or_else(invalid_callback_result_type)
            }
            funcptr::Ret::Pointer(_) | funcptr::Ret::Aggregate(_) => {
                Err(invalid_callback_result_type())
            }
        }
    }

    fn invalid_callback_result_type() -> pyre_interpreter::PyError {
        pyre_interpreter::PyError::type_error("invalid result type for callback function")
    }

    unsafe fn thunk_user_callback(args: VOIDPP, result: VOIDP, userdata: *mut UserData) {
        let addarg = unsafe { (*userdata).addarg };
        let userdata = unsafe { &*(addarg as *const ThunkUserdata) };
        if userdata.result_width != 0 {
            unsafe { std::ptr::write_bytes(result.cast::<u8>(), 0, userdata.result_width) };
        }
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            thunk_callback_inner(result, args as *const *const c_void, userdata)
        }));
    }

    const _: UserCallback = thunk_user_callback;

    unsafe fn thunk_callback_inner(
        result: VOIDP,
        args: *const *const c_void,
        userdata: &ThunkUserdata,
    ) {
        let _callback =
            pyre_interpreter::module::thread::enter_external_callback_from_foreign_thread();
        let wref = unsafe { *userdata.slot } as *const pyre_object::weakref::Weakref;
        let obj = unsafe { pyre_object::weakref::w_weakref_deref(wref) };
        if obj.is_null() {
            return;
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        let current_obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
        let flags = funcptr::funcptr_flags(current_obj());
        let use_errno = flags & funcptr::FUNCFLAG_USE_ERRNO != 0;
        let use_last_error = flags & funcptr::FUNCFLAG_USE_LASTERROR != 0;
        let call_result = match decode_args(current_obj(), args) {
            Ok(decoded) => {
                callproc::with_callback_errno_preserved(use_errno, use_last_error, || {
                    let callable = funcptr::instance_get(current_obj(), funcptr::CALLABLE_KEY)
                        .ok_or_else(|| {
                            pyre_interpreter::PyError::type_error("callback has no callable")
                        })?;
                    let _ = pyre_object::gc_roots::pin_root(callable);
                    let callable_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                    pyre_interpreter::call::call_function_impl_result(
                        pyre_object::gc_roots::shadow_stack_get(callable_slot),
                        &decoded,
                    )
                })
            }
            Err(error) => Err(error),
        };
        let written = match funcptr::callback_result(current_obj(), call_result) {
            Ok(value) => {
                let _ = pyre_object::gc_roots::pin_root(value);
                let value_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                write_result(
                    current_obj(),
                    pyre_object::gc_roots::shadow_stack_get(value_slot),
                    result,
                )
            }
            Err(error) => Err(error),
        };
        if let Err(mut error) = written {
            error.write_unraisable(
                pyre_object::w_none(),
                &rustpython_wtf8::Wtf8Buf::from_string(
                    "Exception ignored while converting result of ctypes callback function"
                        .to_string(),
                ),
                pyre_object::PY_NULL,
            );
        }
    }

    fn decode_args(
        obj: PyObjectRef,
        args: *const *const c_void,
    ) -> Result<Vec<PyObjectRef>, pyre_interpreter::PyError> {
        let argtypes = funcptr::resolve_argtypes(obj).unwrap_or_default();
        let argtypes_slot = pyre_object::gc_roots::pin_roots(&argtypes);
        let mut decoded_slots = Vec::with_capacity(argtypes.len());
        for i in 0..argtypes.len() {
            let at = pyre_object::gc_roots::shadow_stack_get(argtypes_slot + i);
            let p = unsafe { *args.add(i) };
            {
                let value = decode_arg(at, p)?;
                decoded_slots.push(pyre_object::gc_roots::shadow_stack_len());
                let _ = pyre_object::gc_roots::pin_root(value);
            }
        }
        Ok(decoded_slots
            .into_iter()
            .map(pyre_object::gc_roots::shadow_stack_get)
            .collect())
    }

    fn decode_arg(
        at: PyObjectRef,
        p: *const c_void,
    ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        if funcptr::is_funcptr_type(at) {
            let address = unsafe { *(p as *const usize) };
            let inst = pyre_interpreter::call::type_call_instantiate(at, &[])?;
            let inst_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(inst);
            funcptr::store_funcptr_addr(
                pyre_object::gc_roots::shadow_stack_get(inst_slot),
                address,
            )?;
            return Ok(pyre_object::gc_roots::shadow_stack_get(inst_slot));
        }
        if let Some(tc) = cdata::type_code_of(at)
            && !funcptr::is_simple_subclass(at)
        {
            if tc == "O" {
                let num = unsafe { *(p as *const usize) };
                return Ok(cdata::pyobj_container_get(num).unwrap_or_else(pyre_object::w_none));
            }
            return Ok(cdata::decoded_to_pyobject(unsafe {
                cfield::callback_arg_value(Some(&tc), p)
            }));
        }
        if let Some(size) = cdata::ctype_size_of(at) {
            let inst = pyre_interpreter::call::type_call_instantiate(at, &[])?;
            let inst_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(inst);
            let bytes = unsafe { cfield::borrow_memory(p.cast::<u8>(), size) };
            cdata::cdata_write(pyre_object::gc_roots::shadow_stack_get(inst_slot), 0, bytes);
            return Ok(pyre_object::gc_roots::shadow_stack_get(inst_slot));
        }
        Err(pyre_interpreter::PyError::type_error(
            "cannot build parameter",
        ))
    }

    fn write_result(
        mut obj: PyObjectRef,
        mut value: PyObjectRef,
        result: *mut c_void,
    ) -> Result<(), pyre_interpreter::PyError> {
        match pyre_object::with_roots!(obj, value => funcptr::resolve_restype(obj))? {
            funcptr::Ret::Void => Ok(()),
            funcptr::Ret::Code(c) => {
                let bytes = cdata::encode_value_into(&c, value, obj, "result")?;
                let ret = funcptr::Ret::Code(c);
                let n = bytes.len().min(ffi_result_width(&ret));
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        result.cast::<u8>().add(ffi_result_offset(&ret, n)),
                        n,
                    );
                }
                Ok(())
            }
            funcptr::Ret::Pointer(_) | funcptr::Ret::Aggregate(_) => {
                Err(invalid_callback_result_type())
            }
        }
    }
}

/// Build a C-callable thunk for the callback `CFuncPtr` instance `obj` and
/// return its code address.
pub(super) fn build_thunk(obj: PyObjectRef) -> Result<Option<usize>, pyre_interpreter::PyError> {
    imp::build_thunk(obj)
}

/// `CThunkObject_dealloc` / `W_CallbackPtr.free` — drop the thunk owned by
/// `obj`.
pub(super) fn free_thunk(obj: PyObjectRef) {
    imp::free_thunk(obj)
}
