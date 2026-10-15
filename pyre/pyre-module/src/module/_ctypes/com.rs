//! The COM half of `_CFuncPtr` — a method reached through an interface
//! pointer's vtable rather than through a symbol.
//!
//! Such a method has no address of its own: the constructor is given a vtable
//! slot number, and the interface pointer the call is made on is what the slot
//! is read out of, so the first call argument is always that pointer.  When
//! the constructor was also handed the interface id, a failed status is
//! reported as a `COMError` carrying whatever the callee's error object says
//! about it; without one the restype's own `_check_retval_` has the last word.

use super::cdata;
use super::cfield;
use pyre_object::PyObjectRef;
use rustpython_host_env::ctypes as host_ctypes;

/// Vtable slots.  Every interface starts with the three `IUnknown` ones, so
/// the numbering below continues from there for each of the two interfaces an
/// error is read through.
const QUERY_INTERFACE: usize = 0;
const RELEASE: usize = 2;
/// `ISupportErrorInfo`.
const INTERFACE_SUPPORTS_ERROR_INFO: usize = 3;
/// `IErrorInfo`.
const GET_GUID: usize = 3;
const GET_SOURCE: usize = 4;
const GET_DESCRIPTION: usize = 5;
const GET_HELP_FILE: usize = 6;
const GET_HELP_CONTEXT: usize = 7;

/// `IID_ISupportErrorInfo` — {DF0B3D60-548F-101B-8E65-08002B2BD119} in the
/// byte order a `GUID` has in memory.
static IID_ISUPPORTERRORINFO: [u8; 16] = [
    0x60, 0x3d, 0x0b, 0xdf, 0x8f, 0x54, 0x1b, 0x10, 0x8e, 0x65, 0x08, 0x00, 0x2b, 0x2b, 0xd1, 0x19,
];

/// The address in slot `index` of the vtable `this` points at, or 0 when
/// there is nothing to call there.
///
/// A COM method is not an import — there is no name to link against, only a
/// function pointer the object publishes — so reading the slot and calling it
/// is the whole calling convention.
fn slot(this: usize, index: usize) -> usize {
    match host_ctypes::resolve_com_vtable_entry(this, index) {
        Ok(entry) => entry.0 as usize,
        Err(_) => 0,
    }
}

/// Call a slot that takes only `this`.  A missing slot answers the failure
/// every caller here treats as "nothing more to read".
unsafe fn call1(this: usize, index: usize) -> i32 {
    let f = slot(this, index);
    if f == 0 {
        return -1;
    }
    let f: unsafe extern "system" fn(usize) -> i32 = unsafe { std::mem::transmute(f) };
    unsafe { f(this) }
}

unsafe fn call2(this: usize, index: usize, a: usize) -> i32 {
    let f = slot(this, index);
    if f == 0 {
        return -1;
    }
    let f: unsafe extern "system" fn(usize, usize) -> i32 = unsafe { std::mem::transmute(f) };
    unsafe { f(this, a) }
}

unsafe fn call3(this: usize, index: usize, a: usize, b: usize) -> i32 {
    let f = slot(this, index);
    if f == 0 {
        return -1;
    }
    let f: unsafe extern "system" fn(usize, usize, usize) -> i32 =
        unsafe { std::mem::transmute(f) };
    unsafe { f(this, a, b) }
}

/// The `this` pointer and the code address a COM method call runs at.
pub(super) fn call_target(
    index: i64,
    inargs: &[PyObjectRef],
) -> Result<(usize, usize), pyre_interpreter::PyError> {
    let Some(&this) = inargs.first() else {
        return Err(pyre_interpreter::PyError::value_error(
            "native com method call without 'this' parameter",
        ));
    };
    if !cdata::is_cdata_instance(this) {
        return Err(pyre_interpreter::PyError::type_error(
            "Expected a COM this pointer as first argument",
        ));
    }
    let this_ptr = cfield::read_pointer_from_buffer(cdata::cdata_bytes(this).unwrap_or(&[]));
    match host_ctypes::resolve_com_vtable_entry(this_ptr, index.max(0) as usize) {
        Ok(entry) => Ok((this_ptr, entry.0 as usize)),
        Err(host_ctypes::ComMethodError::NullComPointer) => Err(
            pyre_interpreter::PyError::value_error("NULL COM pointer access"),
        ),
        Err(host_ctypes::ComMethodError::NullVtablePointer) => Err(
            pyre_interpreter::PyError::value_error("COM method call without VTable"),
        ),
        Err(host_ctypes::ComMethodError::NullFunctionPointer) => Err(
            pyre_interpreter::PyError::value_error("NULL function pointer"),
        ),
    }
}

/// What the callee's error object says about a failed call.  All-empty when it
/// has nothing to say, which is also what a callee that carries no error
/// object at all leaves behind.
struct ErrorInfo {
    guid: [u8; 16],
    description: usize,
    source: usize,
    help_file: usize,
    help_context: u32,
}

impl Drop for ErrorInfo {
    fn drop(&mut self) {
        for bstr in [self.description, self.source, self.help_file] {
            if bstr != 0 {
                host_ctypes::sys_free_string(bstr as *const u16);
            }
        }
    }
}

/// An object only carries error info when it answers to `ISupportErrorInfo`
/// for the very interface the failed call was made on, so both that question
/// and the interface's own answer come before anything is read.
fn collect_error_info(this: usize, iid: usize) -> ErrorInfo {
    let mut info = ErrorInfo {
        guid: [0; 16],
        description: 0,
        source: 0,
        help_file: 0,
        help_context: 0,
    };
    unsafe {
        let mut psei = 0usize;
        if call3(
            this,
            QUERY_INTERFACE,
            IID_ISUPPORTERRORINFO.as_ptr() as usize,
            std::ptr::addr_of_mut!(psei) as usize,
        ) < 0
        {
            return info;
        }
        let supports = call2(psei, INTERFACE_SUPPORTS_ERROR_INFO, iid);
        call1(psei, RELEASE);
        if supports < 0 {
            return info;
        }
        // `S_FALSE` means the thread has no error object, and only `S_OK` says
        // one came back.
        let Some(pei) = host_ctypes::get_error_info() else {
            return info;
        };
        let pei = pei as usize;
        call2(
            pei,
            GET_DESCRIPTION,
            std::ptr::addr_of_mut!(info.description) as usize,
        );
        call2(pei, GET_GUID, info.guid.as_mut_ptr() as usize);
        call2(
            pei,
            GET_HELP_CONTEXT,
            std::ptr::addr_of_mut!(info.help_context) as usize,
        );
        call2(
            pei,
            GET_HELP_FILE,
            std::ptr::addr_of_mut!(info.help_file) as usize,
        );
        call2(
            pei,
            GET_SOURCE,
            std::ptr::addr_of_mut!(info.source) as usize,
        );
        call1(pei, RELEASE);
    }
    info
}

/// The `str` a BSTR holds, or `None` for a null one.
fn bstr_str(bstr: usize) -> PyObjectRef {
    if bstr == 0 {
        return pyre_object::w_none();
    }
    cdata::bstr_to_pyobject(&bstr.to_ne_bytes())
}

/// The ProgID `guid` is registered under, or `None` when it is registered
/// under none — which is also what a guid nobody filled in answers.
fn prog_id(guid: &[u8; 16]) -> PyObjectRef {
    let Some(progid) = host_ctypes::prog_id_from_clsid(guid) else {
        return pyre_object::w_none();
    };
    let mut len = 0;
    while unsafe { *progid.add(len) } != 0 {
        len += 1;
    }
    let value = pyre_object::w_str_from_wtf8_managed(rustpython_wtf8::Wtf8Buf::from_wide(unsafe {
        std::slice::from_raw_parts(progid, len)
    }));
    host_ctypes::co_task_mem_free(progid.cast());
    value
}

/// `GetComError` — the `COMError` a failed COM method call raises.
pub(super) fn error(hresult: i32, iid: usize, this: usize) -> pyre_interpreter::PyError {
    // Every read below is a COM method call, and a COM method call must not
    // hold the interpreter: the callee is free to wait on another thread.
    let info = {
        let _blocked = pyre_interpreter::module::thread::before_external_block();
        collect_error_info(this, iid)
    };
    // Each value allocates, so each is pinned as it is built rather than
    // gathered into a list the next allocation could strand.
    let _roots = pyre_object::gc_roots::push_roots();
    let details_base = pyre_object::gc_roots::shadow_stack_len();
    for bstr in [info.description, info.source, info.help_file] {
        let _ = pyre_object::gc_roots::pin_root(bstr_str(bstr));
    }
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(info.help_context as i64));
    let _ = pyre_object::gc_roots::pin_root(prog_id(&info.guid));
    let details = pyre_object::w_tuple_new(
        (0..5)
            .map(|i| pyre_object::gc_roots::shadow_stack_get(details_base + i))
            .collect(),
    );
    let details_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(details);
    let text = match host_ctypes::format_error_message(Some(hresult as u32)) {
        Some(message) => pyre_object::w_str_new_managed(&message),
        None => pyre_object::w_none(),
    };
    let text_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(text);
    let hresult_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(hresult as i64));
    // Without the class there is nothing to raise but the report the id-less
    // form would have given, which is the same status by another name.
    let win32_error = || {
        pyre_interpreter::PyError::os_error_win32_syscall2(
            hresult,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        )
    };
    let Some(cls) = super::interp_ctypes::comerror_type() else {
        return win32_error();
    };
    let args = [
        pyre_object::gc_roots::shadow_stack_get(hresult_slot),
        pyre_object::gc_roots::shadow_stack_get(text_slot),
        pyre_object::gc_roots::shadow_stack_get(details_slot),
    ];
    match pyre_interpreter::call::type_call_instantiate(cls, &args) {
        Ok(instance) => {
            let instance_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(instance);
            let mut error = win32_error();
            error.set_exc_object(pyre_object::gc_roots::shadow_stack_get(instance_slot));
            error
        }
        Err(error) => error,
    }
}
