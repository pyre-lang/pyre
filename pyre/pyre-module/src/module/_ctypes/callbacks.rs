//! C-callable callback thunks for `_CFuncPtr` instances.

use pyre_object::PyObjectRef;

#[cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows",
        target_os = "android"
    ),
    not(any(target_env = "musl", target_env = "sgx"))
))]
mod imp {
    use super::PyObjectRef;
    use crate::module::_ctypes::callproc::{CTypeLayout, CallbackThunk, FfiType};
    use crate::module::_ctypes::{callproc, cdata, cfield, funcptr};
    use core::ffi::c_void;
    use majit_rlib::clibffi::{CALLBACK_TP, FFI_CIFP, VOIDPP};
    use majit_rlib::rffi::VOIDP;
    use parking_lot::Mutex;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::LazyLock;

    struct ThunkUserdata {
        /// Address of the GC-rooted slot holding the `CFuncPtr` instance: the
        /// slot, not the instance. A minor collection moves the instance and
        /// rewrites the slot, so the live pointer is read through it at call
        /// time.
        slot: *mut usize,
        /// Size of the libffi result slot the callback CIF returns through —
        /// one `ffi_arg` word for an integral result narrower than that.  Kept
        /// here so the outer panic boundary can initialize the whole slot
        /// before any Python/runtime work is attempted.
        result_width: usize,
    }

    struct StoredThunk {
        #[allow(dead_code)]
        slot: Box<usize>,
        #[allow(dead_code)]
        thunk: CallbackThunk<ThunkUserdata>,
    }

    unsafe impl Send for StoredThunk {}

    static CALLBACK_THUNKS: LazyLock<Mutex<Vec<StoredThunk>>> =
        LazyLock::new(|| Mutex::new(Vec::new()));

    pub(super) fn build_thunk(
        obj: PyObjectRef,
    ) -> Result<Option<usize>, pyre_interpreter::PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        let current_obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
        let argtypes = funcptr::resolve_argtypes(current_obj()).unwrap_or_default();
        // The whole livevar set is published before the first forwarding query:
        // `pin_root` in a loop queries after the first write, and that query is
        // a safepoint at which the entries still only in the `Vec` can move.
        let argtypes_slot = pyre_object::gc_roots::pin_roots(&argtypes);
        let ffi_arg_types = (0..argtypes.len())
            .map(|i| ffi_type_for_arg(pyre_object::gc_roots::shadow_stack_get(argtypes_slot + i)))
            .collect::<Result<Vec<_>, _>>()?;
        let restype =
            funcptr::resolve_restype(current_obj()).map_err(|_| invalid_callback_result_type())?;
        let ffi_res_type = ffi_type_for_ret(&restype)?;
        let result_width = ffi_result_slot_width(&restype);

        let mut slot = Box::new(current_obj() as usize);
        let slot_ptr = (&mut *slot) as *mut usize;
        let root_slot = slot_ptr as *mut pyre_object::gc_hook::GCREF;
        unsafe { pyre_object::gc_hook::try_gc_add_root(root_slot) };

        let thunk = CallbackThunk::new(
            ffi_arg_types,
            ffi_res_type,
            Box::new(ThunkUserdata {
                slot: slot_ptr,
                result_width,
            }),
            thunk_callback,
        );
        let code = thunk.code_ptr();

        // A foreign caller may retain a callback pointer indefinitely; pyre has
        // no proof point for unregistering it, so the rooted slot and closure
        // intentionally live for the process lifetime.
        CALLBACK_THUNKS.lock().push(StoredThunk { slot, thunk });
        Ok(Some(code))
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

    /// Lower the same recursive layout foreign calls use to the callback CIF.
    /// Struct field kinds determine register classification on every major
    /// ABI; a byte-struct of the same size is not equivalent.
    fn ffi_type_for_layout(layout: &CTypeLayout) -> Result<FfiType, pyre_interpreter::PyError> {
        match layout {
            CTypeLayout::Simple(code) => {
                let mut buf = [0u8; 4];
                callproc::ffi_type_from_code(code.encode_utf8(&mut buf)).ok_or_else(|| {
                    pyre_interpreter::PyError::type_error("unknown callback argument type")
                })
            }
            CTypeLayout::Pointer => Ok(callproc::ffi_pointer_type()),
            CTypeLayout::Struct { fields, .. } => fields
                .iter()
                .map(ffi_type_for_layout)
                .collect::<Result<Vec<_>, _>>()
                .map(FfiType::structure),
            CTypeLayout::Array {
                element, length, ..
            } => Ok(callproc::ffi_repeat_type(
                ffi_type_for_layout(element)?,
                *length,
            )),
            CTypeLayout::Union { size, .. } | CTypeLayout::Opaque { size } => {
                Ok(callproc::ffi_byte_struct(*size))
            }
        }
    }

    fn ffi_result_width(ret: &funcptr::Ret) -> usize {
        match ret {
            funcptr::Ret::Void => 0,
            // ctypes long double (`g`) is declared to libffi as `double`, so
            // use the CIF width rather than its 16-byte cdata storage size
            // on Unix.
            funcptr::Ret::Code(code) if code == "g" => 8,
            funcptr::Ret::Code(code) => cfield::simple_type_size(code).unwrap_or(0),
            funcptr::Ret::Pointer(_) => cfield::pointer_size(),
            funcptr::Ret::Aggregate(_) => 0,
        }
    }

    /// Width of `ffi_arg`, the word a libffi closure reserves for an integral
    /// return however narrow the declared result is.
    const FFI_ARG_SIZE: usize = std::mem::size_of::<usize>();

    /// A floating-point result code, which libffi returns through its own
    /// register file and never widens to `ffi_arg`.
    fn is_float_code(code: &str) -> bool {
        matches!(code, "f" | "d" | "g")
    }

    /// Bytes of the closure's return slot the result owns.  An integral result
    /// narrower than `ffi_arg` still owns the whole word, so that is what gets
    /// cleared and what the value is placed within.
    fn ffi_result_slot_width(ret: &funcptr::Ret) -> usize {
        let width = ffi_result_width(ret);
        match ret {
            funcptr::Ret::Code(code) if !is_float_code(code) => width.max(FFI_ARG_SIZE),
            _ => width,
        }
    }

    /// Offset of an `n`-byte value inside its `ffi_arg` return slot: a
    /// big-endian ABI keeps the value at the end of the word
    /// (`callbacks.c:234-239`).
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

    unsafe extern "C" fn thunk_callback(
        _cif: FFI_CIFP,
        result: VOIDP,
        args: VOIDPP,
        userdata: VOIDP,
    ) {
        let userdata = unsafe { &*(userdata as *const ThunkUserdata) };
        if userdata.result_width != 0 {
            unsafe { std::ptr::write_bytes(result.cast::<u8>(), 0, userdata.result_width) };
        }
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            thunk_callback_inner(result, args as *const *const c_void, userdata)
        }));
    }

    const _: CALLBACK_TP = thunk_callback;

    unsafe fn thunk_callback_inner(
        result: VOIDP,
        args: *const *const c_void,
        userdata: &ThunkUserdata,
    ) {
        let _callback =
            pyre_interpreter::module::thread::enter_external_callback_from_foreign_thread();
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(unsafe { *userdata.slot } as PyObjectRef);
        let current_obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
        let use_errno = funcptr::funcptr_flags(current_obj()) & funcptr::FUNCFLAG_USE_ERRNO != 0;
        let call_result = match decode_args(current_obj(), args) {
            Ok(decoded) => callproc::with_callback_errno_preserved(use_errno, || {
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
            }),
            Err(error) => Err(error),
        };
        // A callback has nowhere to raise: the frame above it is foreign.  The
        // call's own failure was already reported by `callback_result`, which
        // substitutes a zero result; what is left here is a result the declared
        // `restype` cannot represent.
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
        // Published as one set for the reason `pin_roots` documents: pinning
        // them one at a time makes the first query a safepoint at which the
        // still-unpinned entries can move.
        let argtypes_slot = pyre_object::gc_roots::pin_roots(&argtypes);
        // `decode_arg` may itself pin construction intermediates (aggregate
        // and function-pointer instances), so decoded results are not
        // necessarily contiguous on the shadow stack.  Record each result's
        // actual slot instead of deriving `base + index`.
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
        // A `_CFuncPtr` can expose the pointer type code too, but Python must
        // receive a callable function-pointer instance, not the integer that
        // the generic simple decoder would produce.  CPython's
        // `ConvParam`/`PyCFuncPtrType` path likewise resolves this before the
        // scalar fallback.
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
                // The word is the container slot, not the object: an address
                // parked in callback memory would dangle once its target moved.
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
                // The return slot is one `ffi_arg` word: a narrow value goes
                // into a word `thunk_callback` already cleared rather than
                // beside whatever the closure left there, and nothing wider
                // than the CIF's own type may be written.  `g` is the case that
                // overruns — it encodes to a long double while
                // `ffi_type_from_code` maps it to the `f64` the slot holds.
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

#[cfg(not(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows",
        target_os = "android"
    ),
    not(any(target_env = "musl", target_env = "sgx"))
)))]
mod imp {
    use super::PyObjectRef;

    pub(super) fn build_thunk(
        _obj: PyObjectRef,
    ) -> Result<Option<usize>, pyre_interpreter::PyError> {
        Ok(None)
    }
}

/// Build a C-callable thunk for the callback `CFuncPtr` instance `obj` and
/// return its code address, or `None` when this platform has no libffi closure
/// support.
pub(super) fn build_thunk(obj: PyObjectRef) -> Result<Option<usize>, pyre_interpreter::PyError> {
    imp::build_thunk(obj)
}
