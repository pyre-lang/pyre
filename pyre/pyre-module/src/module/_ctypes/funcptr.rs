//! `CFuncPtr` (imported as `_CFuncPtr`) — the foreign-function object.
//!
//! `__new__` resolves and stores a function-pointer address (`"_ptr"`) from a
//! `(name, dll)` pair, a bare integer address, or nothing (NULL).  `__call__`
//! marshals scalar Python arguments into libffi values and performs the call
//! through `host_env`, then decodes the scalar result.
//!
//! All host/FFI work is delegated to `rustpython_host_env::ctypes`.  Arguments
//! are marshalled into `CallArg`s and the return type into a `CallRet`, then the
//! call runs through the single `call` entry point, which performs the libffi
//! call and decodes the result.  By-reference arguments — `byref()` carriers,
//! `_Pointer`/`Array` instances, and pointer-typed cdata — lower to
//! `CallArg::Pointer(addr)`; by-value struct/union arguments and returns lower
//! to `CallArg::Aggregate` / `CallRet::Aggregate`; a pointer-typed `restype`
//! wraps the returned address in a fresh instance.

use super::cdata::{self, TypeCode};
use super::stginfo::{self, ParamFunc};
use pyre_object::PyObjectRef;
use rustpython_host_env::ctypes as host_ctypes;

/// `_flags_ & FUNCFLAG_USE_ERRNO` — swap the ctypes-local errno around the call.
pub(super) const FUNCFLAG_USE_ERRNO: i64 = host_ctypes::FUNCFLAG_USE_ERRNO as i64;

/// `_flags_ & FUNCFLAG_USE_LASTERROR` — swap the ctypes-local last error
/// around the call, so `ctypes.get_last_error()` reports what the callee set
/// and no call made since can have overwritten it.
pub(super) const FUNCFLAG_USE_LASTERROR: i64 = host_ctypes::FUNCFLAG_USE_LASTERROR as i64;

/// The `restype` a CDLL function has when none is declared: `c_int`.
const DEFAULT_RESTYPE_CODE: TypeCode = TypeCode::from_char('i');

/// Reserved instance-dict keys.
const PTR_KEY: &str = "_ptr";
const RESTYPE_KEY: &str = "_restype";
const ARGTYPES_KEY: &str = "_argtypes";
pub(super) const CALLABLE_KEY: &str = "_callable";
const ERRCHECK_KEY: &str = "_errcheck";
const PARAMFLAGS_KEY: &str = "_paramflags";
/// Reserved instance-dict key holding what the declarations have settled about
/// this function's calls — see [`Settled`].
const PLAN_KEY: &str = "_plan";
#[cfg(windows)]
const INDEX_KEY: &str = "_index";
#[cfg(windows)]
const IID_KEY: &str = "_iid";

/// `paramflags` direction bits: an argument the caller supplies, one the callee
/// writes through, and the locale id that always comes from the default.
const PARAMFLAG_FIN: i64 = 0x1;
const PARAMFLAG_FOUT: i64 = 0x2;
const PARAMFLAG_FLCID: i64 = 0x4;
const PARAMFLAG_DIRECTION: i64 = PARAMFLAG_FIN | PARAMFLAG_FOUT | PARAMFLAG_FLCID;
const PARAMFLAG_FIN_FLCID: i64 = PARAMFLAG_FIN | PARAMFLAG_FLCID;
const PARAMFLAG_FIN_FOUT: i64 = PARAMFLAG_FIN | PARAMFLAG_FOUT;
const INTERNAL_CAST_ADDR: usize = 1;
const INTERNAL_STRING_AT_ADDR: usize = 2;
const INTERNAL_WSTRING_AT_ADDR: usize = 3;
const INTERNAL_MEMORYVIEW_AT_ADDR: usize = 4;
const INTERNAL_PYBYTES_FROMSTRINGANDSIZE: usize = 5;
const INTERNAL_PYOS_SNPRINTF: usize = 6;
#[cfg(windows)]
const INTERNAL_PYERR_SETFROMWINDOWSERR: usize = 7;

static CFUNCPTR_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

/// The native `CFuncPtr` type object (cached, `hasdict=true`).
pub(super) fn cfuncptr_type() -> PyObjectRef {
    CFUNCPTR_TYPE_OBJ.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type_with_base(
            "CFuncPtr",
            init_cfuncptr_type,
            cdata::cdata_type(),
        );
        unsafe {
            pyre_object::typeobject::w_type_set_hasdict(tp, true);
            // PyPy `function.py:CFuncPtr.__metaclass__ = CFuncPtrType`.
            (*tp).w_class = super::metaclass::pycfuncptrtype_type();
        }
        super::finish_cpython_type(tp, "_ctypes", true)
    })
}

fn init_cfuncptr_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let mut ns = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_store!(
        ns,
        "__new__",
        pyre_interpreter::typedef::make_new_descr(cfuncptr_new)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "__call__",
        pyre_interpreter::make_builtin_function("__call__", cfuncptr_call)
    );
    // `restype` / `argtypes` — settable data descriptors with class-attr
    // fallback to `_restype_` / `_argtypes_`.
    let restype_get =
        pyre_interpreter::make_builtin_function_with_arity("restype", restype_getter, 2);
    let restype_get_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(restype_get);
    let restype_set =
        pyre_interpreter::make_builtin_function_with_arity("restype", restype_setter, 3);
    let restype_set_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(restype_set);
    let restype_del =
        pyre_interpreter::make_builtin_function_with_arity("restype", restype_deleter, 2);
    let restype_del_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(restype_del);
    // Later pins forward the earlier functions in their slots. Pass those
    // slots into the descriptor (`ShadowStackFrameworkGCTransformer.pop_roots`).
    ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
    pyre_interpreter::__pyre_store!(
        ns,
        "restype",
        pyre_interpreter::typedef::make_getset_property_named(
            pyre_object::gc_roots::shadow_stack_get(restype_get_slot),
            pyre_object::gc_roots::shadow_stack_get(restype_set_slot),
            pyre_object::gc_roots::shadow_stack_get(restype_del_slot),
            "restype"
        )
    );
    let argtypes_get =
        pyre_interpreter::make_builtin_function_with_arity("argtypes", argtypes_getter, 2);
    let argtypes_get_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(argtypes_get);
    let argtypes_set =
        pyre_interpreter::make_builtin_function_with_arity("argtypes", argtypes_setter, 3);
    let argtypes_set_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(argtypes_set);
    let argtypes_del =
        pyre_interpreter::make_builtin_function_with_arity("argtypes", argtypes_deleter, 2);
    let argtypes_del_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(argtypes_del);
    ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
    pyre_interpreter::__pyre_store!(
        ns,
        "argtypes",
        pyre_interpreter::typedef::make_getset_property_named(
            pyre_object::gc_roots::shadow_stack_get(argtypes_get_slot),
            pyre_object::gc_roots::shadow_stack_get(argtypes_set_slot),
            pyre_object::gc_roots::shadow_stack_get(argtypes_del_slot),
            "argtypes"
        )
    );
    let errcheck_get =
        pyre_interpreter::make_builtin_function_with_arity("errcheck", errcheck_getter, 2);
    let errcheck_get_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(errcheck_get);
    let errcheck_set =
        pyre_interpreter::make_builtin_function_with_arity("errcheck", errcheck_setter, 3);
    let errcheck_set_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(errcheck_set);
    let errcheck_del =
        pyre_interpreter::make_builtin_function_with_arity("errcheck", errcheck_deleter, 2);
    let errcheck_del_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(errcheck_del);
    ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
    pyre_interpreter::__pyre_store!(
        ns,
        "errcheck",
        pyre_interpreter::typedef::make_getset_property_named(
            pyre_object::gc_roots::shadow_stack_get(errcheck_get_slot),
            pyre_object::gc_roots::shadow_stack_get(errcheck_set_slot),
            pyre_object::gc_roots::shadow_stack_get(errcheck_del_slot),
            "errcheck"
        )
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "__repr__",
        pyre_interpreter::make_builtin_function("__repr__", cfuncptr_repr)
    );
    pyre_interpreter::__pyre_store!(
        ns,
        "__bool__",
        pyre_interpreter::make_builtin_function("__bool__", cfuncptr_bool)
    );
}

/// A COM method says which vtable slot it is; everything else reports the
/// default `<T object at 0x...>`.
fn cfuncptr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let obj = args[0];
    let name = cdata::value_type_name(obj);
    let address = pyre_interpreter::display::repr_gc_addr(obj);
    #[cfg(windows)]
    if let Some(index) = com_index(obj) {
        return Ok(pyre_object::w_str_new_managed(&format!(
            "<COM method offset {index}: {name} at {address}>"
        )));
    }
    Ok(pyre_object::w_str_new_managed(&format!(
        "<{name} object at {address}>"
    )))
}

/// A COM method is true whether or not it holds an address, because the vtable
/// slot it names is all it ever needed.
fn cfuncptr_bool(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let obj = args[0];
    #[cfg(windows)]
    if com_index(obj).is_some() {
        return Ok(pyre_object::w_bool_from(true));
    }
    Ok(pyre_object::w_bool_from(funcptr_addr(obj) != 0))
}

/// The vtable slot a COM method lives in, or `None` for an ordinary function.
#[cfg(windows)]
fn com_index(obj: PyObjectRef) -> Option<i64> {
    let index = instance_get(obj, INDEX_KEY)?;
    Some(unsafe { pyre_object::w_int_get_value(index) } - 0x1000)
}

// ── construction ──────────────────────────────────────────────────────

/// `_CFuncPtr.__new__(cls, arg=None)`.
fn cfuncptr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if args.is_empty() || !unsafe { pyre_object::is_type(args[0]) } {
        return Err(pyre_interpreter::PyError::type_error(
            "CFuncPtr.__new__(): not enough arguments",
        ));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let cls_slot = pyre_object::gc_roots::pin_roots(args);
    let _ = pyre_object::gc_roots::shadow_stack_get(cls_slot);
    let rest: Vec<_> = (1..args.len())
        .map(|i| pyre_object::gc_roots::shadow_stack_get(cls_slot + i))
        .collect();
    let (pos, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(&rest);
    reject_kwargs(kwargs)?;
    let mut callback = pyre_object::PY_NULL;
    let mut paramflags = pyre_object::PY_NULL;
    #[cfg(windows)]
    let mut com_index: Option<i64> = None;
    #[cfg(windows)]
    let mut com_iid = pyre_object::PY_NULL;
    let addr: usize = match pos.first().copied() {
        None => 0,
        // `(name, dll)`, with paramflags allowed after it.
        Some(a) if unsafe { pyre_object::is_tuple(a) } => {
            paramflags = pos.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            // `resolve_from_tuple` allocates.  The callback word, the
            // paramflags word, and the tuple itself are pinned together
            // before that, then read back.
            let keep = pyre_object::gc_roots::pin_roots(&[callback, paramflags, a]);
            let addr = resolve_from_tuple(pyre_object::gc_roots::shadow_stack_get(keep + 2))?;
            callback = pyre_object::gc_roots::shadow_stack_get(keep);
            paramflags = pyre_object::gc_roots::shadow_stack_get(keep + 1);
            addr
        }
        // `index, name` — a COM method, with paramflags and then the interface
        // id allowed after it.  It has no address: the vtable of whatever
        // interface pointer the call is made on is where the method lives.
        #[cfg(windows)]
        Some(a) if pos.len() >= 2 && unsafe { pyre_object::is_int(a) } => {
            if !unsafe { pyre_object::is_str(pos[1]) } {
                return Err(pyre_interpreter::PyError::type_error(format!(
                    "argument 2 must be str, not {}",
                    cdata::value_type_name(pos[1])
                )));
            }
            com_index = Some(unsafe { pyre_object::w_int_get_value(a) });
            paramflags = pos.get(2).copied().unwrap_or(pyre_object::PY_NULL);
            com_iid = pos.get(3).copied().unwrap_or(pyre_object::PY_NULL);
            0
        }
        Some(a) if unsafe { pyre_object::is_none(a) } => 0,
        Some(a) if unsafe { pyre_object::is_int(a) } => {
            (unsafe { pyre_object::w_int_get_value(a) }) as usize
        }
        Some(a) => {
            if !pyre_interpreter::baseobjspace::callable_w(a) {
                return Err(pyre_interpreter::PyError::type_error(
                    "argument must be callable or integer function address",
                ));
            }
            callback = a;
            0
        }
    };
    // Every arm above writes `callback` and `paramflags`, so the pin has to
    // follow the match: an earlier pin is killed by those assignments.
    let keep = pyre_object::gc_roots::pin_roots(&[callback, paramflags]);
    let argtypes = type_argtypes(pyre_object::gc_roots::shadow_stack_get(cls_slot));
    let paramflags = validate_paramflags(
        pyre_object::gc_roots::shadow_stack_get(keep + 1),
        argtypes.as_deref(),
    )?;
    let callback = pyre_object::gc_roots::shadow_stack_get(keep);
    let callback_slot = if callback.is_null() {
        None
    } else {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(callback);
        Some(slot)
    };
    // Building the instance allocates, so what gets stored on it is read back
    // out of a root slot once it exists.
    let paramflags_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(paramflags);
    #[cfg(windows)]
    let iid_slot = {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(if com_iid.is_null() {
            pyre_object::w_none()
        } else {
            com_iid
        });
        slot
    };
    let obj = cdata::new_cdata_obj_from_bytes(
        pyre_object::gc_roots::shadow_stack_get(cls_slot),
        host_ctypes::pointer_size(),
        &[],
    )?;
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    store_funcptr_addr(pyre_object::gc_roots::shadow_stack_get(obj_slot), addr)?;
    if let Some(callback_slot) = callback_slot {
        let current_obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        let d = pyre_interpreter::baseobjspace::getdict_native(current_obj);
        let _ = pyre_object::gc_roots::pin_root(d);
        let dict_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let callback = pyre_object::gc_roots::shadow_stack_get(callback_slot);
        unsafe {
            pyre_object::w_dict_setitem_str(
                pyre_object::gc_roots::shadow_stack_get(dict_slot),
                CALLABLE_KEY,
                callback,
            )
        };
        if let Some(code) =
            super::callbacks::build_thunk(pyre_object::gc_roots::shadow_stack_get(obj_slot))?
        {
            store_funcptr_addr(pyre_object::gc_roots::shadow_stack_get(obj_slot), code)?;
        }
    }
    let paramflags = pyre_object::gc_roots::shadow_stack_get(paramflags_slot);
    if !unsafe { pyre_object::is_none(paramflags) } {
        instance_set(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            PARAMFLAGS_KEY,
            paramflags,
        );
    }
    #[cfg(windows)]
    if let Some(index) = com_index {
        let index = pyre_object::w_int_new(index + 0x1000);
        instance_set(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            INDEX_KEY,
            index,
        );
        let iid = pyre_object::gc_roots::shadow_stack_get(iid_slot);
        if !unsafe { pyre_object::is_none(iid) } {
            instance_set(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
                IID_KEY,
                iid,
            );
        }
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(obj_slot))
}

/// The borrow a marshalled argument is handed to the call as; the borrow ends
/// with the call, and `owned` outlives it.
fn owned_arg_as_call_arg(arg: &OwnedArg) -> host_ctypes::CallArg<'_> {
    match arg {
        OwnedArg::Typed(code, buf) => host_ctypes::CallArg::Typed {
            code: code.as_str(),
            buffer: buf.as_slice(),
        },
        OwnedArg::Int(v) => host_ctypes::CallArg::Int(*v),
        OwnedArg::Double(v) => host_ctypes::CallArg::Double(*v),
        OwnedArg::Pointer(v) => host_ctypes::CallArg::Pointer(*v),
        OwnedArg::Aggregate(layout, buf) => host_ctypes::CallArg::Aggregate {
            layout,
            buffer: buf.as_slice(),
        },
    }
}

fn call_error(e: host_ctypes::CallError) -> pyre_interpreter::PyError {
    match e {
        host_ctypes::CallError::NullFunctionPointer => {
            pyre_interpreter::PyError::value_error("NULL function pointer")
        }
        host_ctypes::CallError::UnknownTypeCode(c) => {
            pyre_interpreter::PyError::type_error(format!("unsupported type code {c:?}"))
        }
        host_ctypes::CallError::BufferTooSmall { expected, got } => {
            pyre_interpreter::PyError::value_error(format!(
                "aggregate argument buffer too small: expected {expected}, got {got}"
            ))
        }
    }
}

/// `(name, dll)` → resolved symbol address.  `dll._handle` is the integer
/// library handle; `name` is the symbol string/bytes.
fn resolve_from_tuple(t: PyObjectRef) -> Result<usize, pyre_interpreter::PyError> {
    let name_obj = unsafe { pyre_object::w_tuple_getitem(t, 0) };
    let dll_obj = unsafe { pyre_object::w_tuple_getitem(t, 1) };
    let (Some(mut name_obj), Some(dll_obj)) = (name_obj, dll_obj) else {
        return Err(pyre_interpreter::PyError::type_error(
            "CFuncPtr constructor requires a (name, dll) pair",
        ));
    };
    let handle_obj = pyre_object::with_roots!(name_obj => pyre_interpreter::baseobjspace::getattr_str(dll_obj, "_handle"))?;
    let handle = pyre_object::with_roots!(name_obj => pyre_interpreter::baseobjspace::int_w(handle_obj))?
        as usize;
    let name_bytes: Vec<u8> = if unsafe { pyre_object::is_str(name_obj) } {
        unsafe { pyre_object::w_str_get_wtf8(name_obj) }
            .as_bytes()
            .to_vec()
    } else if unsafe { pyre_object::is_bytes(name_obj) } {
        unsafe { pyre_object::bytesobject::w_bytes_data(name_obj) }.to_vec()
    } else {
        return Err(pyre_interpreter::PyError::type_error(
            "function name must be string or bytes (ordinals not supported)",
        ));
    };
    match name_bytes.as_slice() {
        b"PyBytes_FromStringAndSize" => return Ok(INTERNAL_PYBYTES_FROMSTRINGANDSIZE),
        b"PyOS_snprintf" => return Ok(INTERNAL_PYOS_SNPRINTF),
        #[cfg(windows)]
        b"PyErr_SetFromWindowsErr" => return Ok(INTERNAL_PYERR_SETFROMWINDOWSERR),
        _ => {}
    }
    super::interp_ctypes::lookup_symbol(handle, &name_bytes).map_err(|e| {
        use rustpython_host_env::ctypes::LookupSymbolError as E;
        if matches!(e, E::LibraryNotFound) {
            return pyre_interpreter::PyError::value_error("library not found");
        }
        // A symbol name arrives as bytes, so it is decoded the way a name the
        // host handed us is: `format!` would fold a byte with no UTF-8
        // spelling to U+FFFD and report a symbol nobody asked for.
        let mut msg = rustpython_wtf8::Wtf8Buf::from_string("function '".to_string());
        msg.push_wtf8(&pyre_interpreter::gateway::fsdecode_filename_wtf8(
            &name_bytes,
        ));
        msg.push_str("' not found");
        pyre_interpreter::PyError::attribute_error(msg)
    })
}

// ── restype / argtypes descriptors ────────────────────────────────────

pub(super) fn instance_get(obj: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    // The dictionary `getdict` would answer is a view onto the instance's own
    // mapdict storage and the read goes through to it either way, so this asks
    // the storage directly and leaves the view unbuilt.
    pyre_interpreter::baseobjspace::getdictvalue_native(obj, key)
}

fn instance_set(obj: PyObjectRef, key: &str, mut value: PyObjectRef) {
    let d = pyre_object::with_roots!(value => pyre_interpreter::baseobjspace::getdict_native(obj));
    if d.is_null() {
        return;
    }
    // Every key here is one a [`CallPlan`] was settled from, so the write goes
    // with the plan dropped — `_setargtypes` / `_setrestype` drop `self._ptr`
    // for the same reason.  Either store can grow the dict and so move it, and
    // the value is live across the first, so both go through root slots.
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(d);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(value);
    unsafe {
        pyre_object::w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            key,
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        );
        drop_settled(pyre_object::gc_roots::shadow_stack_get(dict_slot));
    }
}

// ── settled call plan ─────────────────────────────────────────────────

/// Arguments a call hands over without a heap allocation of its own.  A wider
/// signature is rare enough that the allocation does not matter.
const INLINE_CALL_ARGS: usize = 8;

/// The most declared arguments a [`CallPlan`] carries.  A signature wider than
/// this settles nothing and re-derives per call, which costs only the functions
/// nobody calls in a loop.
const PLAN_MAX_ARGS: usize = 8;

/// What the declarations have settled about a function's calls.
#[derive(Clone, Copy)]
enum Settled {
    /// The plan a repeat call runs from.
    Plan(CallPlan),
    /// There is nothing to settle: a paramflag, an `errcheck`, a
    /// `_check_retval_`, a COM index or an aggregate in the signature puts the
    /// call past what a plan describes.  Recorded so that the finding is made
    /// once rather than at every call.
    Unplannable,
}

/// Everything about a call that the declarations fix, derived once.
///
/// `_ctypes/function.py` keeps the prepared call in `self._ptr` and drops it in
/// `_setargtypes` / `_setrestype`; this is the same cache under the same
/// invalidation — [`instance_set`] and [`store_funcptr_addr`] are every write
/// that can change an answer in it.  A class-level `_restype_` / `_argtypes_` /
/// `_flags_` write goes through neither, so the record also carries the type's
/// version tag and is re-derived once that moves.
///
/// The carrier is an opaque `bytes` in the instance dict rather than a Rust
/// side table so that it holds no object reference across a collection and dies
/// with the instance it describes.
#[derive(Clone, Copy)]
struct CallPlan {
    addr: usize,
    flags: i64,
    /// `Ret::Void` or `Ret::Code` only: the other two name a type, and a plan
    /// carries no object references.
    ret: Ret,
    args: [ArgKind; PLAN_MAX_ARGS],
    arg_count: usize,
}

impl CallPlan {
    /// The declared argument shapes, in order.
    fn kinds(&self) -> &[ArgKind] {
        &self.args[..self.arg_count]
    }
}

/// Byte layout of the stored record: a kind byte, the type version tag it was
/// settled under, then — for a plan — the call's fixed values and one record
/// per declared argument.
mod plan_bytes {
    /// Leading byte: this instance settles into no plan.
    pub(super) const UNPLANNABLE: u8 = 0;
    /// Leading byte: a plan follows.
    pub(super) const PLAN: u8 = 1;
    /// Argument record byte: a pointer-kind argtype.
    pub(super) const ARG_POINTER: u8 = 0;
    /// Argument record byte: a simple argtype, followed by its code.
    pub(super) const ARG_SIMPLE: u8 = 1;
    /// Return record byte: the call returns nothing.
    pub(super) const RET_VOID: u8 = 0;
    /// Return record byte: a simple return, followed by its code.
    pub(super) const RET_CODE: u8 = 1;
}

/// The plan `obj` has settled into, or `None` when it has settled nothing yet.
fn settled(mut obj: PyObjectRef) -> Option<Settled> {
    let stored = pyre_object::with_roots!(obj => instance_get(obj, PLAN_KEY))?;
    if !unsafe { pyre_object::bytesobject::is_bytes(stored) } {
        return None;
    }
    let record = unsafe { pyre_object::bytesobject::w_bytes_data(stored) };
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    decode_settled(record, unsafe {
        pyre_interpreter::baseobjspace::w_type_version_tag(cls)
    })
}

/// Read a stored record back, refusing one settled under a different version of
/// the type.
fn decode_settled(record: &[u8], version_tag: u64) -> Option<Settled> {
    let mut cursor = Cursor::new(record);
    let kind = cursor.byte()?;
    if cursor.u64()? != version_tag || version_tag == 0 {
        return None;
    }
    if kind == plan_bytes::UNPLANNABLE {
        return Some(Settled::Unplannable);
    }
    if kind != plan_bytes::PLAN {
        return None;
    }
    let addr = cursor.u64()? as usize;
    let flags = cursor.u64()? as i64;
    let ret = match cursor.byte()? {
        plan_bytes::RET_VOID => Ret::Void,
        plan_bytes::RET_CODE => Ret::Code(TypeCode::from_padded(cursor.code()?)),
        _ => return None,
    };
    let arg_count = cursor.byte()? as usize;
    if arg_count > PLAN_MAX_ARGS {
        return None;
    }
    let mut args = [ArgKind::Pointer; PLAN_MAX_ARGS];
    for slot in args.iter_mut().take(arg_count) {
        *slot = match cursor.byte()? {
            plan_bytes::ARG_POINTER => ArgKind::Pointer,
            plan_bytes::ARG_SIMPLE => ArgKind::Simple(TypeCode::from_padded(cursor.code()?)),
            _ => return None,
        };
    }
    Some(Settled::Plan(CallPlan {
        addr,
        flags,
        ret,
        args,
        arg_count,
    }))
}

/// A reader over a stored record that answers `None` rather than panicking on a
/// record shorter than it claims to be.
struct Cursor<'a> {
    record: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(record: &'a [u8]) -> Self {
        Cursor { record, at: 0 }
    }

    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let bytes = self.record.get(self.at..self.at + N)?.try_into().ok()?;
        self.at += N;
        Some(bytes)
    }

    fn byte(&mut self) -> Option<u8> {
        self.take::<1>().map(|bytes| bytes[0])
    }

    fn u64(&mut self) -> Option<u64> {
        self.take().map(u64::from_ne_bytes)
    }

    fn code(&mut self) -> Option<[u8; TypeCode::CAPACITY]> {
        self.take()
    }
}

/// The record [`decode_settled`] reads.
fn encode_settled(settled: Settled, version_tag: u64) -> Vec<u8> {
    let mut record = Vec::new();
    let plan = match settled {
        Settled::Unplannable => {
            record.push(plan_bytes::UNPLANNABLE);
            record.extend_from_slice(&version_tag.to_ne_bytes());
            return record;
        }
        Settled::Plan(plan) => plan,
    };
    record.push(plan_bytes::PLAN);
    record.extend_from_slice(&version_tag.to_ne_bytes());
    record.extend_from_slice(&(plan.addr as u64).to_ne_bytes());
    record.extend_from_slice(&(plan.flags as u64).to_ne_bytes());
    match plan.ret {
        Ret::Code(code) => {
            record.push(plan_bytes::RET_CODE);
            record.extend_from_slice(&code.padded());
        }
        // `Ret::Pointer` / `Ret::Aggregate` never reach a plan; both name a
        // type, and `store_settled` refuses them.
        _ => record.push(plan_bytes::RET_VOID),
    }
    record.push(plan.arg_count as u8);
    for kind in plan.kinds() {
        match kind {
            ArgKind::Simple(code) => {
                record.push(plan_bytes::ARG_SIMPLE);
                record.extend_from_slice(&code.padded());
            }
            // Same: `store_settled` refuses an aggregate argtype.
            _ => record.push(plan_bytes::ARG_POINTER),
        }
    }
    record
}

/// Drop whatever `dict` had settled, so the next call derives it again.
///
/// # Safety
/// `dict` must be a live instance dictionary.
unsafe fn drop_settled(dict: PyObjectRef) {
    // Dropping one that was never settled is not an error.
    unsafe { pyre_object::dictmultiobject::w_dict_delitem_str(dict, PLAN_KEY) };
}

/// Record what this call settled, so the next one runs from it.
fn store_settled(obj: PyObjectRef, settled: Settled) {
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let version_tag = unsafe { pyre_interpreter::baseobjspace::w_type_version_tag(cls) };
    // A type with no version tag cannot be told apart from one that has since
    // changed, so nothing is settled against it.
    if version_tag == 0 {
        return;
    }
    let record = encode_settled(settled, version_tag);
    // The `bytes` allocation moves the instance, so the dict is read off the
    // instance only after it, out of a root slot.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let stored = pyre_object::bytesobject::w_bytes_from_bytes(&record);
    let stored_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(stored);
    let d = pyre_interpreter::baseobjspace::getdict_native(
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
    );
    if d.is_null() {
        return;
    }
    unsafe {
        pyre_object::w_dict_setitem_str(
            d,
            PLAN_KEY,
            pyre_object::gc_roots::shadow_stack_get(stored_slot),
        )
    };
}

fn restype_getter(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let mut obj = args[1];
    if let Some(v) = pyre_object::with_roots!(obj => instance_get(obj, RESTYPE_KEY)) {
        return Ok(v);
    }
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    Ok(
        unsafe { pyre_interpreter::baseobjspace::lookup_in_type(cls, "_restype_") }
            .unwrap_or_else(pyre_object::w_none),
    )
}

fn restype_setter(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    instance_set(args[1], RESTYPE_KEY, args[2]);
    Ok(pyre_object::w_none())
}

/// `_ctypes_CFuncPtr_restype_set_impl`'s `value == NULL` branch clears both
/// `restype` and the `_check_retval_` checker it was derived from, leaving the
/// getter to answer from the class again.
fn restype_deleter(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    instance_del(args[1], RESTYPE_KEY);
    Ok(pyre_object::w_none())
}

fn argtypes_getter(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let mut obj = args[1];
    if let Some(v) = pyre_object::with_roots!(obj => instance_get(obj, ARGTYPES_KEY)) {
        return Ok(v);
    }
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    Ok(
        unsafe { pyre_interpreter::baseobjspace::lookup_in_type(cls, "_argtypes_") }
            .unwrap_or_else(pyre_object::w_none),
    )
}

fn argtypes_setter(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let value = args[2];
    // `_argtypes_` must be a sequence of types; a bare type (`fn.argtypes =
    // c_int`) or other non-sequence is rejected rather than silently ignored.
    if !unsafe { pyre_object::is_none(value) } && seq_to_vec(value).is_none() {
        return Err(pyre_interpreter::PyError::type_error(
            "argtypes must be a sequence of types",
        ));
    }
    // Paramflags describe the argtypes one for one, so replacing the argtypes
    // has to leave that description true.  The whole argument slice is pinned
    // first: both lookups allocate, and the receiver and the new value are
    // read back from its slots.
    let _roots = pyre_object::gc_roots::push_roots();
    let args_base = pyre_object::gc_roots::pin_roots(args);
    if let Some(paramflags) = instance_get(
        pyre_object::gc_roots::shadow_stack_get(args_base + 1),
        PARAMFLAGS_KEY,
    ) {
        let flags_slot = pyre_object::gc_roots::pin_roots(&[paramflags]);
        let seq = argtypes_seq(pyre_object::gc_roots::shadow_stack_get(args_base + 2));
        validate_paramflags(
            pyre_object::gc_roots::shadow_stack_get(flags_slot),
            seq.as_deref(),
        )?;
    }
    instance_set(
        pyre_object::gc_roots::shadow_stack_get(args_base + 1),
        ARGTYPES_KEY,
        pyre_object::gc_roots::shadow_stack_get(args_base + 2),
    );
    Ok(pyre_object::w_none())
}

/// `_ctypes_CFuncPtr_argtypes_set_impl` treats a NULL operand exactly as it
/// treats `None`: the argtypes and their converters are cleared.
fn argtypes_deleter(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    instance_del(args[1], ARGTYPES_KEY);
    Ok(pyre_object::w_none())
}

fn errcheck_getter(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    Ok(instance_get(args[1], ERRCHECK_KEY).unwrap_or_else(pyre_object::w_none))
}

/// The only thing ever done with an errcheck is calling it, so anything that
/// cannot be called is refused — `None` included, which is why clearing one is
/// spelled `del`.
fn errcheck_setter(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let value = args[2];
    if !pyre_interpreter::baseobjspace::callable_w(value) {
        return Err(pyre_interpreter::PyError::type_error(
            "the errcheck attribute must be callable",
        ));
    }
    instance_set(args[1], ERRCHECK_KEY, value);
    Ok(pyre_object::w_none())
}

fn errcheck_deleter(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    instance_del(args[1], ERRCHECK_KEY);
    Ok(pyre_object::w_none())
}

/// Drop one of the per-instance slots a [`CallPlan`] is settled from.  Every
/// key here reaches the plan, so the delete drops it the way [`instance_set`]
/// does; deleting one that was never set is not an error.
fn instance_del(obj: PyObjectRef, key: &str) {
    let d = pyre_interpreter::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return;
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_delitem_str(d, key);
        drop_settled(d);
    }
}

/// Reject keyword arguments: ctypes foreign calls and `_CFuncPtr(...)` take
/// only positional arguments, so a stray `fn(x, foo=1)` is an error rather
/// than a silently dropped `foo`.
fn reject_kwargs(kwargs: Option<PyObjectRef>) -> Result<(), pyre_interpreter::PyError> {
    let Some(kw) = kwargs else { return Ok(()) };
    for (key_obj, _) in unsafe { pyre_object::w_dict_items(kw) } {
        if unsafe { pyre_object::is_str(key_obj) }
            && unsafe { pyre_object::w_str_get_wtf8(key_obj) } == "__pyre_kw__"
        {
            continue;
        }
        return Err(pyre_interpreter::PyError::type_error(
            "call takes no keyword arguments",
        ));
    }
    Ok(())
}

// ── call ──────────────────────────────────────────────────────────────

/// Resolved return-type selector.
#[derive(Clone, Copy)]
pub(super) enum Ret {
    Void,
    Code(TypeCode),
    /// A pointer metaclass type (`POINTER(T)`): the result address is wrapped
    /// in a fresh instance of this type.
    Pointer(PyObjectRef),
    /// A by-value struct/union type: the returned aggregate bytes are copied
    /// into a fresh instance of this type.
    Aggregate(PyObjectRef),
}

pub(super) fn resolve_restype(obj: PyObjectRef) -> Result<Ret, pyre_interpreter::PyError> {
    let mut cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let rt = pyre_object::with_roots!(cls => instance_get(obj, RESTYPE_KEY))
        .or_else(|| unsafe { pyre_interpreter::baseobjspace::lookup_in_type(cls, "_restype_") });
    match rt {
        // CDLL functions default to c_int when no restype is set.
        None => Ok(Ret::Code(DEFAULT_RESTYPE_CODE)),
        Some(o) if unsafe { pyre_object::is_none(o) } => Ok(Ret::Void),
        Some(mut o) => {
            // A `_Pointer` subtype returns a live pointer instance; a
            // struct/union subtype returns a by-value aggregate instance.
            if let Some(info) = stginfo::stginfo_of(o) {
                match pyre_object::with_roots!(o => stginfo::stginfo_paramfunc(info)) {
                    ParamFunc::Pointer => return Ok(Ret::Pointer(o)),
                    ParamFunc::Struct | ParamFunc::Union => return Ok(Ret::Aggregate(o)),
                    _ => {}
                }
            }
            let tc = cdata::type_code_of(o)
                .ok_or_else(|| pyre_interpreter::PyError::type_error("invalid restype"))?;
            Ok(Ret::Code(tc))
        }
    }
}

/// `restype._check_retval_` — the callable a return value passes through
/// before the caller sees it.  `HRESULT` declares one so that a failed status
/// raises `OSError` rather than being handed back as a negative number.
fn resolve_checker(obj: PyObjectRef) -> Option<PyObjectRef> {
    let mut cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let rt = pyre_object::with_roots!(cls => instance_get(obj, RESTYPE_KEY))
        .or_else(|| unsafe { pyre_interpreter::baseobjspace::lookup_in_type(cls, "_restype_") })?;
    if !unsafe { pyre_object::is_type(rt) } {
        return None;
    }
    unsafe { pyre_interpreter::baseobjspace::lookup_in_type(rt, "_check_retval_") }
}

/// `errcheck(result, self, arguments)` — the last word on what a call returns.
/// Handing back the argument tuple unchanged means "nothing to say", which is
/// `None` here; anything else replaces the result outright, `out` parameters
/// included.
fn apply_errcheck(
    self_obj: PyObjectRef,
    result: PyObjectRef,
    inargs: &[PyObjectRef],
) -> Result<Option<PyObjectRef>, pyre_interpreter::PyError> {
    // The receiver, the result, and every positional argument are published
    // before the first lookup.  Sequential pins would let that lookup move
    // the arguments before they were on the root stack.
    let n_inargs = inargs.len();
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::publish_roots(&[self_obj, result]);
    let inargs_base = pyre_object::gc_roots::publish_roots(inargs);
    pyre_object::gc_roots::normalize_roots(base, 2 + n_inargs);
    let Some(errcheck) = instance_get(pyre_object::gc_roots::shadow_stack_get(base), ERRCHECK_KEY)
    else {
        return Ok(None);
    };
    let errcheck_slot = pyre_object::gc_roots::pin_roots(&[errcheck]);
    let mut inargs_now = vec![pyre_object::PY_NULL; n_inargs];
    pyre_object::gc_roots::shadow_stack_copy_range(inargs_base, &mut inargs_now);
    let arguments = pyre_object::w_tuple_new(inargs_now);
    let arguments_slot = pyre_object::gc_roots::pin_roots(&[arguments]);
    let value = pyre_interpreter::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(errcheck_slot),
        &[
            pyre_object::gc_roots::shadow_stack_get(base + 1),
            pyre_object::gc_roots::shadow_stack_get(base),
            pyre_object::gc_roots::shadow_stack_get(arguments_slot),
        ],
    )?;
    Ok((!std::ptr::eq(
        value,
        pyre_object::gc_roots::shadow_stack_get(arguments_slot),
    ))
    .then_some(value))
}

/// Wrap a returned pointer value `p` in a fresh instance of pointer type `rt`.
fn wrap_pointer_result(
    rt: PyObjectRef,
    p: usize,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let mut obj = pyre_object::w_instance_new(rt);
    let d = pyre_object::with_roots!(obj => pyre_interpreter::baseobjspace::getdict_native(obj));
    if d.is_null() {
        return Err(pyre_interpreter::PyError::type_error(
            "pointer instance has no dict",
        ));
    }
    // The instance dict moves, and the bytearray allocated below is a
    // collection point, so the dict word is read back out of a root slot at the
    // store.  The null check stays ahead of the bracket, leaving the error path
    // rootless.  The instance is rooted here too because nothing else
    // references it until it is returned.
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(d);
    let obj_slot = dict_slot + 1;
    let _ = pyre_object::gc_roots::pin_root(obj);
    let psize = host_ctypes::pointer_size();
    let ba = pyre_object::w_bytearray_new(psize);
    let bytes = host_ctypes::simple_storage_value_to_bytes_endian(
        "P",
        host_ctypes::SimpleStorageValue::Pointer(p),
        false,
    );
    let n = bytes.len().min(psize);
    unsafe {
        pyre_object::w_bytearray_data_mut(ba)[..n].copy_from_slice(&bytes[..n]);
        pyre_object::w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            "_b_",
            ba,
        );
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(obj_slot))
}

/// The `_argtypes_` sequence as a Vec, or `None` when unset (ConvParam
/// defaults apply).
pub(super) fn resolve_argtypes(mut obj: PyObjectRef) -> Option<Vec<PyObjectRef>> {
    match pyre_object::with_roots!(obj => instance_get(obj, ARGTYPES_KEY)) {
        Some(at) => argtypes_seq(at),
        None => type_argtypes(unsafe { pyre_object::w_instance_get_type(obj) }),
    }
}

/// The `_argtypes_` declared on type `cls` — what a constructor has to check
/// paramflags against, the instance having none of its own yet.
fn type_argtypes(cls: PyObjectRef) -> Option<Vec<PyObjectRef>> {
    argtypes_seq(unsafe { pyre_interpreter::baseobjspace::lookup_in_type(cls, "_argtypes_") }?)
}

/// A settled `argtypes` value as a Vec; `None` means unset.
fn argtypes_seq(at: PyObjectRef) -> Option<Vec<PyObjectRef>> {
    if unsafe { pyre_object::is_none(at) } {
        return None;
    }
    seq_to_vec(at)
}

/// The name a type reports, for a message about the type itself rather than
/// about a value of it.
fn type_display_name(obj: PyObjectRef) -> String {
    if unsafe { pyre_object::is_type(obj) } {
        return unsafe { pyre_object::w_type_get_name(obj) }.to_string();
    }
    cdata::value_type_name(obj)
}

/// `_check_outarg_type` — the callee writes an `out` parameter through the
/// argument, so the argtype has to be something there is a through: a pointer
/// or array type, or one of the simple codes that is already an address.
fn check_outarg_type(mut at: PyObjectRef, index: usize) -> Result<(), pyre_interpreter::PyError> {
    if let Some(info) = stginfo::stginfo_of(at)
        && matches!(
            pyre_object::with_roots!(at => stginfo::stginfo_paramfunc(info)),
            ParamFunc::Pointer | ParamFunc::Array
        )
    {
        return Ok(());
    }
    if matches!(cdata::type_code_of(at).as_deref(), Some("P" | "z" | "Z")) {
        return Ok(());
    }
    Err(pyre_interpreter::PyError::type_error(format!(
        "'out' parameter {index} must be a pointer type, not {}",
        type_display_name(at)
    )))
}

/// `_validate_paramflags` — one paramflag per argtype, each an
/// `(int [,string [,value]])` tuple naming a direction that is actually
/// implemented.  Returns the tuple to store, or `None` for "no paramflags",
/// which is also what an absent `_argtypes_` leaves the check with nothing to
/// say about.
fn validate_paramflags(
    mut paramflags: PyObjectRef,
    argtypes: Option<&[PyObjectRef]>,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if paramflags.is_null() || unsafe { pyre_object::is_none(paramflags) } {
        return Ok(pyre_object::w_none());
    }
    if !unsafe { pyre_object::is_tuple(paramflags) } {
        return Err(pyre_interpreter::PyError::type_error(
            "paramflags must be a tuple or None",
        ));
    }
    let Some(argtypes) = argtypes else {
        return Ok(paramflags);
    };
    let len = unsafe { pyre_object::w_tuple_len(paramflags) };
    if len != argtypes.len() {
        return Err(pyre_interpreter::PyError::value_error(
            "paramflags must have the same length as argtypes",
        ));
    }
    let malformed = || {
        pyre_interpreter::PyError::type_error(
            "paramflags must be a sequence of (int [,string [,value]]) tuples",
        )
    };
    for (i, &at) in argtypes.iter().enumerate() {
        let item =
            unsafe { pyre_object::w_tuple_getitem(paramflags, i as i64) }.ok_or_else(malformed)?;
        if !unsafe { pyre_object::is_tuple(item) } {
            return Err(malformed());
        }
        let item_len = unsafe { pyre_object::w_tuple_len(item) };
        if !(1..=3).contains(&item_len) {
            return Err(malformed());
        }
        let flag = unsafe { pyre_object::w_tuple_getitem(item, 0) }.ok_or_else(malformed)?;
        if !unsafe { pyre_object::is_int(flag) } {
            return Err(malformed());
        }
        if item_len > 1 {
            let name = unsafe { pyre_object::w_tuple_getitem(item, 1) }.ok_or_else(malformed)?;
            if !unsafe { pyre_object::is_none(name) } && !unsafe { pyre_object::is_str(name) } {
                return Err(malformed());
            }
        }
        let flag = unsafe { pyre_object::w_int_get_value(flag) };
        match flag & PARAMFLAG_DIRECTION {
            PARAMFLAG_FOUT => pyre_object::with_roots!(paramflags => check_outarg_type(at, i + 1))?,
            0 | PARAMFLAG_FIN | PARAMFLAG_FIN_FLCID | PARAMFLAG_FIN_FOUT => {}
            _ => {
                return Err(pyre_interpreter::PyError::type_error(format!(
                    "paramflag value {flag} not supported"
                )));
            }
        }
    }
    Ok(paramflags)
}

fn seq_to_vec(obj: PyObjectRef) -> Option<Vec<PyObjectRef>> {
    if unsafe { pyre_object::is_tuple(obj) } {
        let n = unsafe { pyre_object::w_tuple_len(obj) };
        Some(
            (0..n as i64)
                .filter_map(|i| unsafe { pyre_object::w_tuple_getitem(obj, i) })
                .collect(),
        )
    } else if unsafe { pyre_object::is_list(obj) } {
        let n = unsafe { pyre_object::w_list_len(obj) };
        Some(
            (0..n as i64)
                .filter_map(|i| unsafe { pyre_object::w_list_getitem(obj, i) })
                .collect(),
        )
    } else {
        None
    }
}

pub(super) fn funcptr_flags(obj: PyObjectRef) -> i64 {
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    match unsafe { pyre_interpreter::baseobjspace::lookup_in_type(cls, "_flags_") } {
        Some(o) if unsafe { pyre_object::is_int(o) } => unsafe { pyre_object::w_int_get_value(o) },
        _ => 0,
    }
}

pub(super) fn funcptr_addr(obj: PyObjectRef) -> usize {
    instance_get(obj, PTR_KEY)
        .filter(|o| unsafe { pyre_object::is_int(*o) })
        .map(|o| unsafe { pyre_object::w_int_get_value(o) } as usize)
        .unwrap_or(0)
}

/// Store a function address in both `_ptr` and the pointer-sized CData buffer.
/// The object and boxed integer are rooted independently because either the
/// integer allocation or the dict write may move the instance.
pub(super) fn store_funcptr_addr(
    obj: PyObjectRef,
    addr: usize,
) -> Result<(), pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let w_addr = pyre_object::w_int_new(addr as i64);
    let addr_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_addr);
    let current_obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    let d = pyre_interpreter::baseobjspace::getdict_native(current_obj);
    if d.is_null() {
        return Err(pyre_interpreter::PyError::type_error(
            "CFuncPtr instance has no instance dict",
        ));
    }
    let _ = pyre_object::gc_roots::pin_root(d);
    let dict_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    unsafe {
        pyre_object::w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            PTR_KEY,
            pyre_object::gc_roots::shadow_stack_get(addr_slot),
        );
        // The address a plan calls through is the one being replaced here.
        drop_settled(pyre_object::gc_roots::shadow_stack_get(dict_slot));
    };
    let bytes = host_ctypes::simple_storage_value_to_bytes_endian(
        "P",
        host_ctypes::SimpleStorageValue::Pointer(addr),
        false,
    );
    cdata::cdata_write(pyre_object::gc_roots::shadow_stack_get(obj_slot), 0, &bytes);
    Ok(())
}

/// Owned argument data whose buffers must outlive the borrowed `CallArg`s
/// handed to `call`.
enum OwnedArg {
    Typed(TypeCode, Vec<u8>),
    Int(i32),
    Double(f64),
    Pointer(usize),
    /// A by-value aggregate: its recursive layout and a copy of its bytes.
    Aggregate(host_ctypes::CTypeLayout, Vec<u8>),
}

pub(super) fn is_simple_subclass(ty: PyObjectRef) -> bool {
    let bases = unsafe { pyre_object::typeobject::w_type_get_bases(ty) };
    !bases.is_null()
        && unsafe { pyre_object::w_tuple_getitem(bases, 0) }
            .is_some_and(|base| cdata::type_code_of(base).is_some())
}

fn callback_argument(
    ty: PyObjectRef,
    value: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if is_simple_subclass(ty) {
        pyre_interpreter::call::type_call_instantiate(ty, &[value])
    } else {
        Ok(value)
    }
}

pub(super) fn callback_result(
    mut obj: PyObjectRef,
    result: Result<PyObjectRef, pyre_interpreter::PyError>,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let mut result = match result {
        Ok(value) => value,
        Err(error) => {
            let unknown = || rustpython_wtf8::Wtf8Buf::from_string("<unknown>".to_string());
            pyre_object::with_roots!(obj => {
                let roots = pyre_object::gc_roots::push_roots();
                let mut error = error;
                let error_slot = error.pin(&roots);
                let callable = instance_get(obj, CALLABLE_KEY).unwrap_or(pyre_object::PY_NULL);
                let rendered = if callable.is_null() {
                    unknown()
                } else {
                    unsafe { pyre_interpreter::display::py_repr_wtf8(callable) }
                        .unwrap_or_else(|_| unknown())
                };
                error.reload(&roots, error_slot);
                error.write_unraisable(
                    pyre_object::w_none(),
                    &pyre_interpreter::display::wtf8_format!(
                        "Exception ignored while calling ctypes callback function ",
                        rendered
                    ),
                    pyre_object::PY_NULL,
                );
            });
            pyre_object::w_int_new(0)
        }
    };
    match pyre_object::with_roots!(obj, result => resolve_restype(obj))? {
        Ret::Void => Ok(pyre_object::w_none()),
        Ret::Code(code) => {
            let bytes = cdata::encode_value_into(&code, result, obj, "result")?;
            Ok(cdata::decode_slot(&code, &bytes))
        }
        Ret::Pointer(_) | Ret::Aggregate(_) => Ok(result),
    }
}

fn call_python_callback(
    obj: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // The callable and the argument slice both have to be on the root stack
    // before the first lookup.  Each argument is read from its slot at the
    // conversion that consumes it, because that conversion can allocate.
    let n_args = args.len();
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::publish_roots(&[obj]);
    let args_base = pyre_object::gc_roots::publish_roots(args);
    pyre_object::gc_roots::normalize_roots(obj_slot, 1 + n_args);
    let callable = instance_get(
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
        CALLABLE_KEY,
    )
    .ok_or_else(|| pyre_interpreter::PyError::type_error("callback has no callable"))?;
    let callable_slot = pyre_object::gc_roots::pin_roots(&[callable]);
    let argtypes =
        resolve_argtypes(pyre_object::gc_roots::shadow_stack_get(obj_slot)).unwrap_or_default();
    if n_args != argtypes.len() {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "this function takes {} arguments ({} given)",
            argtypes.len(),
            n_args,
        )));
    }
    let converted_base = pyre_object::gc_roots::shadow_stack_len();
    for (i, ty) in argtypes.into_iter().enumerate() {
        let converted =
            callback_argument(ty, pyre_object::gc_roots::shadow_stack_get(args_base + i))?;
        let _ = pyre_object::gc_roots::publish_roots(&[converted]);
    }
    pyre_object::gc_roots::normalize_roots(converted_base, n_args);
    let mut converted = vec![pyre_object::PY_NULL; n_args];
    pyre_object::gc_roots::shadow_stack_copy_range(converted_base, &mut converted);
    let result = pyre_interpreter::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(callable_slot),
        &converted,
    );
    callback_result(pyre_object::gc_roots::shadow_stack_get(obj_slot), result)
}

/// `_CFuncPtr.__call__(self, *args)`.
fn cfuncptr_call(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "__call__ requires self",
        ));
    }
    // A keyword argument only ever names a paramflag, and one that names
    // nothing is not an error — it simply goes unread.
    let (inargs, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(&args[1..]);
    // self, the positional arguments, and the keyword dict are published
    // before the first allocating call.  Later steps read those slots; the
    // originals are not.
    let n_inargs = inargs.len();
    let has_kwargs = kwargs.is_some();
    let kwargs_word = kwargs.unwrap_or(pyre_object::PY_NULL);
    let _roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::publish_roots(&[args[0]]);
    let inargs_base = pyre_object::gc_roots::publish_roots(inargs);
    let kwargs_slot = pyre_object::gc_roots::publish_roots(&[kwargs_word]);
    pyre_object::gc_roots::normalize_roots(self_slot, 1 + n_inargs + 1);
    let reload_inargs = || -> Vec<PyObjectRef> {
        let mut reloaded = vec![pyre_object::PY_NULL; n_inargs];
        pyre_object::gc_roots::shadow_stack_copy_range(inargs_base, &mut reloaded);
        reloaded
    };
    let settled_already = settled(pyre_object::gc_roots::shadow_stack_get(self_slot));
    // A plan that turns out not to describe this call after all leaves the
    // general path to derive it again and record what it finds.
    let mut settled_stands = settled_already.is_some();
    if let Some(Settled::Plan(plan)) = settled_already {
        match call_settled(
            pyre_object::gc_roots::shadow_stack_get(self_slot),
            &plan,
            &reload_inargs(),
        )? {
            Some(value) => return Ok(value),
            None => settled_stands = false,
        }
    }
    let addr = funcptr_addr(pyre_object::gc_roots::shadow_stack_get(self_slot));
    if addr == 0
        && instance_get(
            pyre_object::gc_roots::shadow_stack_get(self_slot),
            CALLABLE_KEY,
        )
        .is_some()
    {
        return call_python_callback(
            pyre_object::gc_roots::shadow_stack_get(self_slot),
            &reload_inargs(),
        );
    }
    match addr {
        INTERNAL_CAST_ADDR => return internal_cast(&reload_inargs()),
        INTERNAL_STRING_AT_ADDR => return internal_string_at(&reload_inargs()),
        INTERNAL_WSTRING_AT_ADDR => return internal_wstring_at(&reload_inargs()),
        INTERNAL_MEMORYVIEW_AT_ADDR => return internal_memoryview_at(&reload_inargs()),
        INTERNAL_PYBYTES_FROMSTRINGANDSIZE => {
            return internal_pybytes_fromstringandsize(&reload_inargs());
        }
        INTERNAL_PYOS_SNPRINTF => return internal_pyos_snprintf(&reload_inargs()),
        #[cfg(windows)]
        INTERNAL_PYERR_SETFROMWINDOWSERR => {
            return internal_pyerr_setfromwindowserr(&reload_inargs());
        }
        _ => {}
    }

    // A COM method has no address of its own: the interface pointer passed as
    // the first argument is what its vtable is read out of, and that pointer
    // then leads the call.
    #[cfg(windows)]
    let com = match com_index(pyre_object::gc_roots::shadow_stack_get(self_slot)) {
        Some(index) => Some(super::com::call_target(index, &reload_inargs())?),
        None => None,
    };
    #[cfg(not(windows))]
    let com: Option<(usize, usize)> = None;

    let argtypes = resolve_argtypes(pyre_object::gc_roots::shadow_stack_get(self_slot));
    let paramflags = instance_get(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        PARAMFLAGS_KEY,
    );
    let has_paramflags = paramflags.is_some();
    let paramflags_word = paramflags.unwrap_or(pyre_object::PY_NULL);
    let _paramflags_slot = pyre_object::gc_roots::pin_roots(&[paramflags_word]);
    let kwargs_now = has_kwargs.then(|| pyre_object::gc_roots::shadow_stack_get(kwargs_slot));
    let mut callargs = build_callargs(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        argtypes.as_deref(),
        &reload_inargs(),
        kwargs_now,
        com.is_some(),
    )?;
    let n_call_args = callargs.args.len();
    let call_args_base = {
        let call_args = callargs.args.as_slice();
        pyre_object::gc_roots::pin_roots(call_args)
    };

    // Marshal arguments into owned scalar data.  `keepalive` owns any
    // null-terminated `bytes` copies that pointer args address; `owned` owns
    // the typed buffers.  Both must outlive the borrowed `SimpleArg`s below.
    let mut owned: Vec<OwnedArg> = Vec::with_capacity(n_call_args + 1);
    let mut keepalive: Vec<Vec<u8>> = Vec::new();
    if let Some((this_ptr, _)) = com {
        owned.push(OwnedArg::Pointer(this_ptr));
    }

    // What each declared argtype lowers to, kept so that a repeat call can run
    // from it instead of asking the types again.
    let mut kinds: Vec<ArgKind> = Vec::new();
    match &argtypes {
        Some(argtypes) => {
            for (i, &at) in argtypes.iter().enumerate() {
                if i >= n_call_args {
                    return Err(pyre_interpreter::PyError::type_error(format!(
                        "this function takes at least {} argument(s)",
                        argtypes.len()
                    )));
                }
                let kind = classify_argtype(at).ok_or_else(|| {
                    pyre_interpreter::PyError::type_error("argtype has no valid '_type_'")
                })?;
                kinds.push(kind);
                owned.push(marshal_typed_arg(
                    pyre_object::gc_roots::shadow_stack_get(call_args_base + i),
                    at,
                    kind,
                    &mut keepalive,
                )?);
            }
            // Variadic tail (printf-style): arguments past the declared
            // argtypes are marshalled by the default conversion rules.
            for i in argtypes.len().min(n_call_args)..n_call_args {
                owned.push(marshal_default_arg(
                    pyre_object::gc_roots::shadow_stack_get(call_args_base + i),
                    &mut keepalive,
                )?);
            }
        }
        None => {
            for i in 0..n_call_args {
                owned.push(marshal_default_arg(
                    pyre_object::gc_roots::shadow_stack_get(call_args_base + i),
                    &mut keepalive,
                )?);
            }
        }
    }

    let ret = resolve_restype(pyre_object::gc_roots::shadow_stack_get(self_slot))?;
    let flags = funcptr_flags(pyre_object::gc_roots::shadow_stack_get(self_slot));
    let result = invoke(call_address(addr, com), &owned, ret, flags)?;
    // `owned` / `keepalive` must outlive the call above.
    drop(keepalive);
    // A COM method constructed with an interface id answers the plain status,
    // and asks the callee what went wrong when that status is a failure; the
    // restype never gets a look in.
    let checker = resolve_checker(pyre_object::gc_roots::shadow_stack_get(self_slot));
    let has_checker = checker.is_some();
    let checker_word = checker.unwrap_or(pyre_object::PY_NULL);
    let checker_slot = pyre_object::gc_roots::pin_roots(&[checker_word]);
    let value = match com_status(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        com,
        &result,
    ) {
        Some(status) => status?,
        None => {
            let value = build_return_value(ret, result)?;
            if has_checker {
                pyre_interpreter::call::call_function_impl_result(
                    pyre_object::gc_roots::shadow_stack_get(checker_slot),
                    &[value],
                )?
            } else {
                value
            }
        }
    };
    let value_slot = pyre_object::gc_roots::pin_roots(&[value]);
    let errcheck = instance_get(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        ERRCHECK_KEY,
    );
    let has_errcheck = errcheck.is_some();
    let errcheck_word = errcheck.unwrap_or(pyre_object::PY_NULL);
    let _errcheck_slot = pyre_object::gc_roots::pin_roots(&[errcheck_word]);
    let mut call_args_now = vec![pyre_object::PY_NULL; n_call_args];
    pyre_object::gc_roots::shadow_stack_copy_range(call_args_base, &mut call_args_now);
    let applied = apply_errcheck(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        pyre_object::gc_roots::shadow_stack_get(value_slot),
        &call_args_now,
    )?;
    pyre_object::gc_roots::shadow_stack_copy_range(call_args_base, &mut call_args_now);
    callargs.args = call_args_now;
    let mut value = match applied {
        Some(forced) => forced,
        None => build_result(
            pyre_object::gc_roots::shadow_stack_get(value_slot),
            &callargs,
        )?,
    };
    if !settled_stands {
        let value_out = pyre_object::gc_roots::pin_roots(&[value]);
        store_settled(
            pyre_object::gc_roots::shadow_stack_get(self_slot),
            plan_of_call(
                addr,
                flags,
                ret,
                &kinds,
                com.is_some() || has_checker || has_errcheck || has_paramflags,
            ),
        );
        value = pyre_object::gc_roots::shadow_stack_get(value_out);
    }
    Ok(value)
}

/// The address a call is made through: a COM method's comes out of the
/// interface vtable, everything else calls the stored `_ptr`.
fn call_address(addr: usize, com: Option<(usize, usize)>) -> usize {
    match com {
        Some((_, method)) => method,
        None => addr,
    }
}

/// What this call settles, for the next one to run from.
///
/// `past_a_plan` is the caller's finding that something outside the argument
/// and return shapes — a COM index, a `_check_retval_`, an `errcheck`, a
/// paramflag — has a say in the answer or in which arguments the call is made
/// with.
fn plan_of_call(
    addr: usize,
    flags: i64,
    ret: Ret,
    kinds: &[ArgKind],
    past_a_plan: bool,
) -> Settled {
    let describable = !past_a_plan
        && kinds.len() <= PLAN_MAX_ARGS
        && matches!(ret, Ret::Void | Ret::Code(_))
        && !kinds.iter().any(|kind| *kind == ArgKind::Aggregate);
    if !describable {
        return Settled::Unplannable;
    }
    let mut args = [ArgKind::Pointer; PLAN_MAX_ARGS];
    args[..kinds.len()].copy_from_slice(kinds);
    Settled::Plan(CallPlan {
        addr,
        flags,
        ret,
        args,
        arg_count: kinds.len(),
    })
}

/// Hand the marshalled arguments to the foreign function and answer its raw
/// result.
fn invoke(
    addr: usize,
    owned: &[OwnedArg],
    ret: Ret,
    flags: i64,
) -> Result<host_ctypes::CallValue, pyre_interpreter::PyError> {
    // Build the aggregate return layout (if any) up front so it outlives the
    // borrowed `CallRet` handed to the call.
    let ret_layout = match &ret {
        Ret::Aggregate(ty) => Some(build_layout(*ty)?),
        _ => None,
    };
    let restype = match &ret {
        Ret::Void => host_ctypes::CallRet::Void,
        Ret::Code(c) => host_ctypes::CallRet::Code(c.as_str()),
        Ret::Pointer(_) => host_ctypes::CallRet::Code("P"),
        Ret::Aggregate(_) => host_ctypes::CallRet::Aggregate(
            ret_layout.as_ref().expect("aggregate restype has a layout"),
        ),
    };

    // Borrow the owned data as `CallArg`s; these borrows end with the call.
    // A signature no wider than the inline row is handed over without a heap
    // allocation of its own, which is otherwise a whole allocation per call.
    let mut inline = [host_ctypes::CallArg::Int(0); INLINE_CALL_ARGS];
    let mut spilled: Vec<host_ctypes::CallArg> = Vec::new();
    let host_args: &[host_ctypes::CallArg] = if owned.len() <= INLINE_CALL_ARGS {
        for (slot, arg) in inline.iter_mut().zip(owned) {
            *slot = owned_arg_as_call_arg(arg);
        }
        &inline[..owned.len()]
    } else {
        spilled.extend(owned.iter().map(owned_arg_as_call_arg));
        &spilled
    };

    let options = host_ctypes::CallOptions {
        use_errno: flags & FUNCFLAG_USE_ERRNO != 0,
        use_last_error: flags & FUNCFLAG_USE_LASTERROR != 0,
    };
    // `clibffi.py:350-352` declares `ffi_call` with the default
    // `releasegil='auto'`, i.e. released: the callee is arbitrary foreign code
    // and may block on another Python thread's progress.  The caller's `owned`
    // / `keepalive` hold the marshalled buffers across the released window.
    // Being arbitrary foreign code is also why the call goes inside
    // `seh::guard`, which is what stands between a faulting callee and the
    // process.
    let result = {
        let _blocked = pyre_interpreter::module::thread::before_external_block();
        super::seh::guard(|| host_ctypes::call(addr, host_args, restype, options))
    };
    result?.map_err(call_error)
}

/// `__call__` for a function whose declarations have already settled into a
/// [`CallPlan`] — `_getfuncptr` answering out of `self._ptr`.
///
/// Answers `None` when the plan turns out not to describe this call after all,
/// which leaves the general path to derive it again.
fn call_settled(
    self_obj: PyObjectRef,
    plan: &CallPlan,
    inargs: &[PyObjectRef],
) -> Result<Option<PyObjectRef>, pyre_interpreter::PyError> {
    let kinds = plan.kinds();
    if inargs.len() < kinds.len() {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "this function takes at least {} argument(s)",
            kinds.len()
        )));
    }
    // `inargs` is the caller's native copy; resolving the argtypes and
    // marshalling an argument can collect, so each is read from its slot.
    let _roots = pyre_object::gc_roots::push_roots();
    let inargs_base = _roots.pin_roots(inargs);
    // Only a pointer argtype is still wanted as an object; a simple one is
    // fully described by the code the plan carries.
    let argtypes = if kinds.iter().any(|kind| *kind == ArgKind::Pointer) {
        let argtypes = resolve_argtypes(self_obj);
        if argtypes.as_ref().is_none_or(|at| at.len() < kinds.len()) {
            return Ok(None);
        }
        argtypes
    } else {
        None
    };
    let mut owned: Vec<OwnedArg> = Vec::with_capacity(inargs.len());
    let mut keepalive: Vec<Vec<u8>> = Vec::new();
    for i in 0..inargs.len() {
        let arg = _roots.get(inargs_base + i);
        owned.push(match kinds.get(i) {
            Some(ArgKind::Simple(tc)) => marshal_simple_arg(arg, *tc)?,
            Some(_) => {
                let at = argtypes.as_ref().expect("a pointer argtype was resolved")[i];
                OwnedArg::Pointer(pointer_argument_addr(arg, at, &mut keepalive)?)
            }
            // Variadic tail (printf-style): arguments past the declared
            // argtypes are marshalled by the default conversion rules.
            None => marshal_default_arg(arg, &mut keepalive)?,
        });
    }
    let result = invoke(plan.addr, &owned, plan.ret, plan.flags)?;
    // `owned` / `keepalive` must outlive the call above.
    drop(keepalive);
    build_return_value(plan.ret, result).map(Some)
}

/// The status word a returned scalar carries.  A COM method answers an
/// `HRESULT` whatever its declared restype would otherwise make of the bytes.
#[allow(dead_code)]
fn scalar_int_result(result: &host_ctypes::CallValue) -> i32 {
    let mut word = [0u8; 4];
    if let host_ctypes::CallValue::Scalar(bytes) = result {
        let n = bytes.len().min(word.len());
        word[..n].copy_from_slice(&bytes[..n]);
    }
    i32::from_ne_bytes(word)
}

/// The status a COM method constructed with an interface id answers, and the
/// callee's own account of a failed one.  `None` for a method without an id
/// and for every ordinary function, which go through the restype instead.
///
/// A COM method only exists on Windows — `_ctypes.c` builds the whole form
/// under `MS_WIN32` — so off it there is no status to read and `com` is
/// already `None` at every call.
#[cfg(windows)]
fn com_status(
    obj: PyObjectRef,
    com: Option<(usize, usize)>,
    result: &host_ctypes::CallValue,
) -> Option<Result<PyObjectRef, pyre_interpreter::PyError>> {
    let (this_ptr, iid) = com_error_iid(obj, com)?;
    let hresult = scalar_int_result(result);
    Some(if hresult < 0 {
        Err(super::com::error(hresult, iid, this_ptr))
    } else {
        Ok(pyre_object::w_int_new(hresult as i64))
    })
}

#[cfg(not(windows))]
fn com_status(
    _obj: PyObjectRef,
    _com: Option<(usize, usize)>,
    _result: &host_ctypes::CallValue,
) -> Option<Result<PyObjectRef, pyre_interpreter::PyError>> {
    None
}

/// The `this` pointer and interface id a failed COM call reports through, when
/// the method was constructed with an id at all.
#[cfg(windows)]
fn com_error_iid(obj: PyObjectRef, com: Option<(usize, usize)>) -> Option<(usize, usize)> {
    let (this_ptr, _) = com?;
    let iid = instance_get(obj, IID_KEY)?;
    let (addr, len) = if unsafe { pyre_object::is_bytes(iid) } {
        let data = unsafe { pyre_object::bytesobject::w_bytes_data(iid) };
        (data.as_ptr() as usize, data.len())
    } else {
        (cdata::cdata_addr(iid)?, cdata::cdata_len(iid)?)
    };
    // Anything that is not `GUID`-sized is not an interface id, and is passed
    // over the way an unrecognised one is.
    (len == 16).then_some((this_ptr, addr))
}

/// The arguments a call is made with, and which of them the caller gets back.
struct CallArgs {
    args: Vec<PyObjectRef>,
    /// Arguments the callee alone fills in, by index.
    outmask: u32,
    /// Arguments the caller supplies and the callee writes back through.
    inoutmask: u32,
    numretvals: u32,
}

/// `_build_callargs` — with no `paramflags` the call is made with the
/// arguments the caller passed (less the COM `this`, which leads the call
/// separately) and nothing comes back through them.  With `paramflags` every
/// declared argtype gets a value: from the call, from a keyword, from a
/// default, or from a fresh instance for the callee to write into.
fn build_callargs(
    self_obj: PyObjectRef,
    argtypes: Option<&[PyObjectRef]>,
    inargs: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
    is_com: bool,
) -> Result<CallArgs, pyre_interpreter::PyError> {
    let passed = if is_com { &inargs[1..] } else { inargs };
    let plain = |args: Vec<PyObjectRef>| CallArgs {
        args,
        outmask: 0,
        inoutmask: 0,
        numretvals: 0,
    };
    // Publish the receiver, the keyword dict, and the positional arguments
    // before the paramflags lookup.  A pin of only the dict would let that
    // lookup move the arguments.
    let n_passed = passed.len();
    let has_kwargs = kwargs.is_some();
    let kwargs_word = kwargs.unwrap_or(pyre_object::PY_NULL);
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = roots.publish(&[self_obj]);
    let kwargs_slot = roots.publish(&[kwargs_word]);
    let passed_base = roots.publish(passed);
    roots.normalize(self_slot, 2 + n_passed);
    let passed_now = || -> Vec<PyObjectRef> {
        let mut reloaded = vec![pyre_object::PY_NULL; n_passed];
        pyre_object::gc_roots::shadow_stack_copy_range(passed_base, &mut reloaded);
        reloaded
    };
    let kwargs_now = || has_kwargs.then(|| roots.get(kwargs_slot));
    let paramflags = instance_get(roots.get(self_slot), PARAMFLAGS_KEY);
    let (Some(paramflags), Some(argtypes)) = (paramflags, argtypes) else {
        return Ok(plain(passed_now()));
    };
    if argtypes.is_empty() || !unsafe { pyre_object::is_tuple(paramflags) } {
        return Ok(plain(passed_now()));
    }
    let mut out = plain(Vec::with_capacity(argtypes.len()));
    // `out_parameter` instantiates the argtype, which is arbitrary Python, so
    // every value already collected lives in a root slot across it; the list
    // is read back out of those slots once the loop is done.
    // `paramflags` is the tuple the loop indexes on every turn, and the same
    // arbitrary Python runs between two of those reads.  A tuple relocates, so
    // it is pinned ahead of `base` -- the values below want a contiguous run
    // starting there -- and read back at each index.
    let paramflags_slot = roots.pin_roots(&[paramflags]);
    let base = pyre_object::gc_roots::shadow_stack_len();
    let mut index = 0;
    for (i, &at) in argtypes.iter().enumerate() {
        let malformed = || {
            pyre_interpreter::PyError::value_error(
                "paramflags must have the same length as argtypes",
            )
        };
        let item = unsafe {
            pyre_object::w_tuple_getitem(
                pyre_object::gc_roots::shadow_stack_get(paramflags_slot),
                i as i64,
            )
        }
        .ok_or_else(malformed)?;
        if !unsafe { pyre_object::is_tuple(item) } {
            return Err(malformed());
        }
        let item_len = unsafe { pyre_object::w_tuple_len(item) };
        let flag = unsafe { pyre_object::w_tuple_getitem(item, 0) }.ok_or_else(malformed)?;
        let flag = unsafe { pyre_object::w_int_get_value(flag) };
        let name = (item_len > 1)
            .then(|| unsafe { pyre_object::w_tuple_getitem(item, 1) })
            .flatten()
            .filter(|&n| unsafe { pyre_object::is_str(n) })
            .and_then(|n| {
                pyre_interpreter::baseobjspace::str_utf8_w(n)
                    .ok()
                    .map(str::to_string)
            });
        let defval = if item_len > 2 {
            unsafe { pyre_object::w_tuple_getitem(item, 2) }
        } else {
            None
        };
        let value = match flag & PARAMFLAG_DIRECTION {
            // A locale id never comes from the call.
            PARAMFLAG_FIN_FLCID => defval.unwrap_or_else(|| pyre_object::w_int_new(0)),
            PARAMFLAG_FOUT => {
                out.outmask |= param_bit(i);
                out.numretvals += 1;
                match defval {
                    Some(defval) => defval,
                    None => out_parameter(at)?,
                }
            }
            direction => {
                if direction == PARAMFLAG_FIN_FOUT {
                    out.inoutmask |= param_bit(i);
                    out.numretvals += 1;
                }
                get_arg(
                    &mut index,
                    name.as_deref(),
                    defval,
                    &passed_now(),
                    kwargs_now(),
                )?
            }
        };
        let value = pyre_object::gc_roots::pin_root(value);
        out.args.push(value);
    }
    for (i, arg) in out.args.iter_mut().enumerate() {
        *arg = pyre_object::gc_roots::shadow_stack_get(base + i);
    }
    Ok(out)
}

/// The `1 << i` bit `_build_callargs` sets in its two `int` masks.  A
/// parameter past the width of that word has no bit of its own, which is the
/// range [`build_result`] reads back.
fn param_bit(i: usize) -> u32 {
    1u32.checked_shl(i as u32).unwrap_or(0)
}

/// `_get_arg` — the next positional argument, else the keyword of that name,
/// else the declared default.
fn get_arg(
    index: &mut usize,
    name: Option<&str>,
    defval: Option<PyObjectRef>,
    inargs: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if *index < inargs.len() {
        let value = inargs[*index];
        *index += 1;
        return Ok(value);
    }
    if let (Some(kwargs), Some(name)) = (kwargs, name)
        && let Some(value) = unsafe { pyre_object::w_dict_getitem_str(kwargs, name) }
    {
        *index += 1;
        return Ok(value);
    }
    if let Some(defval) = defval {
        return Ok(defval);
    }
    Err(match name {
        Some(name) => {
            pyre_interpreter::PyError::type_error(format!("required argument '{name}' missing"))
        }
        None => pyre_interpreter::PyError::type_error("not enough arguments"),
    })
}

/// The instance an `out` parameter is given for the callee to write into.  The
/// argtype points at the thing written, so that is what gets built — an array
/// type being its own such thing.
fn out_parameter(at: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let info = stginfo::stginfo_of(at);
    if info.is_some_and(|info| stginfo::stginfo_paramfunc(info) == ParamFunc::Array) {
        return pyre_interpreter::call::type_call_instantiate(at, &[]);
    }
    match info.and_then(stginfo::stginfo_proto) {
        Some(proto) if unsafe { pyre_object::is_type(proto) } => {
            pyre_interpreter::call::type_call_instantiate(proto, &[])
        }
        // A simple pointer code points at no type there is an instance of.
        _ => Err(pyre_interpreter::PyError::type_error(format!(
            "{} 'out' parameter must be passed as default value",
            type_display_name(at)
        ))),
    }
}

/// `_build_result` — with no `out` parameters the call's own result stands;
/// with one it is that parameter, and with several a tuple of them.
fn build_result(
    result: PyObjectRef,
    callargs: &CallArgs,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if callargs.numretvals == 0 {
        return Ok(result);
    }
    // `__ctypes_from_outparam__` is arbitrary Python and the tuple allocates,
    // so every value still wanted lives in a root slot across both.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    for &arg in &callargs.args {
        let _ = pyre_object::gc_roots::pin_root(arg);
    }
    let returned_base = pyre_object::gc_roots::shadow_stack_len();
    let mut returned = 0;
    for i in 0..callargs.args.len().min(u32::BITS as usize) {
        let bit = 1 << i;
        let value = if callargs.inoutmask & bit != 0 {
            pyre_object::gc_roots::shadow_stack_get(base + i)
        } else if callargs.outmask & bit != 0 {
            from_outparam(pyre_object::gc_roots::shadow_stack_get(base + i))?
        } else {
            continue;
        };
        let _ = pyre_object::gc_roots::pin_root(value);
        returned += 1;
        if returned == callargs.numretvals as usize {
            break;
        }
    }
    if returned == 1 {
        return Ok(pyre_object::gc_roots::shadow_stack_get(returned_base));
    }
    Ok(pyre_object::w_tuple_new(
        (0..returned)
            .map(|i| pyre_object::gc_roots::shadow_stack_get(returned_base + i))
            .collect(),
    ))
}

fn from_outparam(value: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let method = pyre_interpreter::baseobjspace::getattr_str(value, "__ctypes_from_outparam__")?;
    pyre_interpreter::call::call_function_impl_result(method, &[])
}

/// The Python value a returned `CallValue` becomes under return type `ret`.
fn build_return_value(
    ret: Ret,
    result: host_ctypes::CallValue,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    match ret {
        Ret::Pointer(rt) => {
            let p = match result {
                host_ctypes::CallValue::Pointer(p) => p,
                host_ctypes::CallValue::Scalar(b) => host_ctypes::read_pointer_from_buffer(&b),
                _ => 0,
            };
            wrap_pointer_result(rt, p)
        }
        Ret::Aggregate(ty) => {
            let bytes = match result {
                host_ctypes::CallValue::Aggregate(b) => b,
                _ => Vec::new(),
            };
            make_aggregate_instance(ty, &bytes)
        }
        Ret::Void => Ok(cdata::decoded_to_pyobject(host_ctypes::DecodedValue::None)),
        Ret::Code(c) => {
            // Reconstruct the raw result bytes and read them as a slot of that
            // type: a scalar carries its register image, a pointer-code result
            // its address bytes.
            let bytes = match result {
                host_ctypes::CallValue::Void => {
                    return Ok(cdata::decoded_to_pyobject(host_ctypes::DecodedValue::None));
                }
                host_ctypes::CallValue::Scalar(b) | host_ctypes::CallValue::Aggregate(b) => b,
                host_ctypes::CallValue::Pointer(p) => p.to_ne_bytes().to_vec(),
            };
            Ok(cdata::decode_slot(&c, &bytes))
        }
    }
}

fn argument_address(mut obj: PyObjectRef) -> Result<usize, pyre_interpreter::PyError> {
    if unsafe { pyre_object::is_none(obj) } {
        return Ok(0);
    }
    if unsafe { pyre_object::is_int(obj) } {
        return Ok(pyre_interpreter::baseobjspace::int_w(obj)? as usize);
    }
    if unsafe { pyre_object::is_bytes(obj) } {
        return Ok(unsafe { pyre_object::bytesobject::w_bytes_data(obj) }.as_ptr() as usize);
    }
    if is_funcptr_instance(obj) {
        return Ok(funcptr_addr(obj));
    }
    if cdata::is_cdata_instance(obj) {
        let mut cls = unsafe { pyre_object::w_instance_get_type(obj) };
        if let Some(info) = stginfo::stginfo_of(cls)
            && pyre_object::with_roots!(cls, obj => stginfo::stginfo_paramfunc(info))
                == ParamFunc::Pointer
        {
            return Ok(host_ctypes::read_pointer_from_buffer(
                cdata::cdata_bytes(obj).unwrap_or(&[]),
            ));
        }
        if cdata::type_code_of(cls).is_some_and(|tc| cdata::is_pointer_code(&tc)) {
            return Ok(host_ctypes::read_pointer_from_buffer(
                cdata::cdata_bytes(obj).unwrap_or(&[]),
            ));
        }
        return cdata::cdata_addr(obj)
            .ok_or_else(|| pyre_interpreter::PyError::type_error("ctypes instance has no buffer"));
    }
    if super::interp_ctypes::is_carg(obj) {
        return Ok(super::interp_ctypes::carg_ptr(obj));
    }
    Err(pyre_interpreter::PyError::type_error(
        "wrong type: expected bytes, integer address, ctypes instance, or None",
    ))
}

fn internal_cast(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if args.len() < 3 || !unsafe { pyre_object::is_type(args[2]) } {
        return Err(pyre_interpreter::PyError::type_error(
            "cast() argument 2 must be a pointer type",
        ));
    }
    let target = args[2];
    // `_ctypes.c cast_check_pointertype`: a pointer type, a function-pointer
    // type, or a simple type whose code is one of the pointer-shaped ones.
    let is_pointer = stginfo::stginfo_of(target)
        .is_some_and(|i| stginfo::stginfo_paramfunc(i) == ParamFunc::Pointer)
        || is_funcptr_type(target)
        || cdata::type_code_of(target)
            .is_some_and(|tc| matches!(tc.as_str(), "s" | "P" | "z" | "U" | "Z" | "X" | "O"));
    if !is_pointer {
        return Err(pyre_interpreter::PyError::type_error(
            "cast() argument 2 must be a pointer type",
        ));
    }
    // The address read and the instance allocation both move.  The whole
    // argument slice is pinned once; the source, the type, and the kept
    // object are read from those slots afterwards.
    let _roots = pyre_object::gc_roots::push_roots();
    let args_base = pyre_object::gc_roots::pin_roots(args);
    let address = argument_address(pyre_object::gc_roots::shadow_stack_get(args_base))?;
    let result = pyre_interpreter::call::type_call_instantiate(
        pyre_object::gc_roots::shadow_stack_get(args_base + 2),
        &[],
    )?;
    let result_slot = pyre_object::gc_roots::pin_roots(&[result]);
    if is_funcptr_type(pyre_object::gc_roots::shadow_stack_get(args_base + 2)) {
        // A foreign function keeps its address beside the buffer, and that is
        // the copy a call reads.
        store_funcptr_addr(
            pyre_object::gc_roots::shadow_stack_get(result_slot),
            address,
        )?;
    } else {
        let bytes = host_ctypes::simple_storage_value_to_bytes_endian(
            "P",
            host_ctypes::SimpleStorageValue::Pointer(address),
            false,
        );
        cdata::cdata_write(
            pyre_object::gc_roots::shadow_stack_get(result_slot),
            0,
            &bytes,
        );
    }
    if cdata::is_cdata_instance(pyre_object::gc_roots::shadow_stack_get(args_base + 1)) {
        cdata::share_objects_for_cast(
            pyre_object::gc_roots::shadow_stack_get(result_slot),
            pyre_object::gc_roots::shadow_stack_get(args_base + 1),
        );
    } else {
        cdata::keep_ref(
            pyre_object::gc_roots::shadow_stack_get(result_slot),
            "1",
            pyre_object::gc_roots::shadow_stack_get(args_base + 1),
        );
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(result_slot))
}

/// What a `string_at`/`wstring_at` read that `host_env` refuses to attempt
/// raises.  Neither one checks the address it is handed — `strlen`,
/// `PyBytes_FromStringAndSize` and `PyUnicode_FromWideChar` are simply given
/// the pointer — so NULL is a bad address like any other and the fence names
/// it the way it names the rest.
#[cfg(all(windows, target_env = "msvc"))]
fn declined_read(address: usize) -> pyre_interpreter::PyError {
    super::seh::access_violation_reading(address)
}

/// Where no fence can catch a fault there is no access violation to report,
/// so a read `host_env` refuses to attempt raises the refusal itself rather
/// than taking the process down over it.
#[cfg(not(all(windows, target_env = "msvc")))]
fn declined_read(_address: usize) -> pyre_interpreter::PyError {
    pyre_interpreter::PyError::value_error("NULL pointer access")
}

fn internal_string_at(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "string_at() missing address",
        ));
    }
    let size = args
        .get(1)
        .copied()
        .map(pyre_interpreter::baseobjspace::int_w)
        .transpose()?
        .unwrap_or(-1);
    // `PyBytes_FromStringAndSize` rejects an impossible size before the
    // pointer converter runs.  Keep that ordering for huge explicit sizes.
    if size > isize::MAX as i64 / 2 {
        return Err(pyre_interpreter::PyError::memory_error("size too large"));
    }
    let address = argument_address(args[0])?;
    // `PyBytes_FromStringAndSize` copies nothing for a zero size, so the
    // address is never dereferenced and need not be readable at all.
    if size == 0 {
        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&[]));
    }
    // Only `-1` asks for the length to be measured; any other negative size
    // reaches `PyBytes_FromStringAndSize`, which refuses it before it copies,
    // so the address is not read for that either.
    if size < -1 {
        return Err(pyre_interpreter::PyError::system_error(
            "Negative size passed to PyBytes_FromStringAndSize",
        ));
    }
    // `_string_at = PYFUNCTYPE(py_object, c_void_p, c_int)(_string_at_addr)`,
    // so the read runs as an ordinary foreign call and the `__try` around
    // `ffi_call` is what stands between a bad address and the process.
    let read = super::seh::guard(|| host_ctypes::string_at(address, size as isize))?;
    let value = match read {
        Ok(value) => value,
        Err(host_ctypes::StringAtError::NullPointer) => return Err(declined_read(address)),
        Err(host_ctypes::StringAtError::TooLong) => {
            return Err(pyre_interpreter::PyError::memory_error("size too large"));
        }
    };
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&value))
}

fn internal_wstring_at(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "wstring_at() missing address",
        ));
    }
    let size = args
        .get(1)
        .copied()
        .map(pyre_interpreter::baseobjspace::int_w)
        .transpose()?
        .unwrap_or(-1);
    if size > isize::MAX as i64 / std::mem::size_of::<libc::wchar_t>() as i64 {
        return Err(pyre_interpreter::PyError::overflow_error("size too large"));
    }
    let address = argument_address(args[0])?;
    // `PyUnicode_FromWideChar` copies nothing for a zero size, so the address
    // is never dereferenced and need not be readable at all.
    if size == 0 {
        return Ok(pyre_object::w_str_new(""));
    }
    // A negative size other than `-1` reaches `find_maxchar_surrogates` as the
    // range `[ptr, ptr + size)`, whose end precedes its start, so nothing is
    // read and `PyUnicode_New` refuses the length.  An address low enough for
    // that end to wrap past it is the exception: there the scan runs away from
    // `ptr`, which is what the read below does for a negative size as well.
    if size < -1 {
        let end =
            address.wrapping_add((size as isize as usize).wrapping_mul(size_of::<libc::wchar_t>()));
        if end <= address {
            return Err(pyre_interpreter::PyError::system_error(
                "Negative size passed to PyUnicode_New",
            ));
        }
    }
    let read = super::seh::guard(|| host_ctypes::wstring_at(address, size as isize))?;
    let value = match read {
        Ok(value) => value,
        Err(host_ctypes::StringAtError::NullPointer) => return Err(declined_read(address)),
        Err(host_ctypes::StringAtError::TooLong) => {
            return Err(pyre_interpreter::PyError::overflow_error("size too large"));
        }
    };
    Ok(pyre_object::w_str_from_wtf8_managed(value))
}

fn internal_memoryview_at(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if args.len() < 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "memoryview_at() needs address and size",
        ));
    }
    let w_address = args[0];
    let mut w_size = args[1];
    let mut w_readonly = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
    let address = pyre_object::with_roots!(w_size, w_readonly => argument_address(w_address))?;
    if !unsafe { pyre_object::is_int(w_size) || pyre_object::is_long(w_size) } {
        return Err(pyre_interpreter::PyError::type_error(
            "size must be an integer",
        ));
    }
    let size =
        pyre_object::with_roots!(w_readonly => pyre_interpreter::baseobjspace::int_w(w_size))
            .map_err(|_| pyre_interpreter::PyError::value_error("size is too large"))?;
    if size < 0 {
        return Err(pyre_interpreter::PyError::value_error(
            "size must not be negative",
        ));
    }
    let readonly = if w_readonly.is_null() {
        false
    } else {
        pyre_interpreter::baseobjspace::is_true(w_readonly)?
    };
    let roots = pyre_object::gc_roots::push_roots();
    let fmt_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(pyre_object::w_str_new_managed("B"));
    let w_obj = pyre_object::w_none();
    let mv = pyre_object::memoryview::w_memoryview_alloc_header(false, false);
    let mv_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(mv);
    let view = pyre_object::bufferview::BufferView::Raw {
        backing: pyre_object::buffer::Buffer::External {
            w_obj,
            address,
            size: size as usize,
            readonly,
        },
        w_obj,
        w_fmt: roots.get(fmt_slot),
        itemsize: 1,
        length: size,
    };
    let view = pyre_object::memoryview::bufferview_alloc(view);
    unsafe { pyre_object::memoryview::w_memoryview_set_view(roots.get(mv_slot), view) };
    Ok(roots.get(mv_slot))
}

fn internal_pybytes_fromstringandsize(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if args.len() < 2 || !unsafe { pyre_object::is_bytes(args[0]) } {
        return Err(pyre_interpreter::PyError::type_error(
            "PyBytes_FromStringAndSize needs string and size",
        ));
    }
    let bytes = unsafe { pyre_object::bytesobject::w_bytes_data(args[0]) };
    let size = pyre_interpreter::baseobjspace::int_w(args[1])?.max(0) as usize;
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(
        &bytes[..size.min(bytes.len())],
    ))
}

/// `PyErr_SetFromWindowsErr(ierr)` — raise the OSError a Win32 error code
/// names, or the one `GetLastError()` names when the code is 0.  The parameter
/// is a C `int`, so a code with the top bit set arrives as the negative number
/// `.winerror` reports; a code the system has no message for is spelled
/// `Windows Error 0x<code>`, which is what `test_windows_message` reads.
#[cfg(windows)]
fn internal_pyerr_setfromwindowserr(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // `PyErr_SetExcFromWindowsErrWithFilenameObjects` reads `GetLastError()`
    // when the code it is handed is 0, so an explicit zero says the same
    // thing as no argument at all rather than naming `ERROR_SUCCESS`.
    let code = match args.first() {
        Some(&arg) => pyre_interpreter::baseobjspace::int_w(arg)? as i32,
        None => 0,
    };
    let code = if code == 0 {
        std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
    } else {
        code
    };
    Err(pyre_interpreter::PyError::os_error_win32_syscall2(
        code,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
    ))
}

/// `_ctypes.call_function(address, args)` — call an address that carries no
/// argtypes, no restype and no flags, so every argument converts by the
/// default rules and the result is read as a C `int`.  `call_cdeclfunction` is
/// the same call under `FUNCFLAG_CDECL`, which on this architecture is the
/// only calling convention there is.
pub(super) fn call_function(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let (Some(&addr), Some(&(mut arguments))) = (args.first(), args.get(1)) else {
        return Err(pyre_interpreter::PyError::type_error(
            "call_function() takes exactly 2 arguments",
        ));
    };
    if !unsafe { pyre_object::is_tuple(arguments) } {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "argument 2 must be tuple, not {}",
            cdata::value_type_name(arguments)
        )));
    }
    let addr = pyre_object::with_roots!(arguments => cdata::pointer_word(addr))?;
    let mut keepalive: Vec<Vec<u8>> = Vec::new();
    let owned = seq_to_vec(arguments)
        .expect("a tuple")
        .into_iter()
        .map(|arg| marshal_default_arg(arg, &mut keepalive))
        .collect::<Result<Vec<_>, _>>()?;
    let host_args: Vec<host_ctypes::CallArg> = owned.iter().map(owned_arg_as_call_arg).collect();
    let result = {
        let _blocked = pyre_interpreter::module::thread::before_external_block();
        super::seh::guard(|| {
            host_ctypes::call(
                addr,
                &host_args,
                host_ctypes::CallRet::Code("i"),
                host_ctypes::CallOptions::default(),
            )
        })
    };
    drop(keepalive);
    let bytes = match result?.map_err(call_error)? {
        host_ctypes::CallValue::Scalar(b) => b,
        host_ctypes::CallValue::Pointer(p) => p.to_ne_bytes().to_vec(),
        _ => Vec::new(),
    };
    Ok(cdata::decoded_to_pyobject(host_ctypes::decode_type_code(
        "i", &bytes,
    )))
}

fn internal_pyos_snprintf(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if args.len() < 3
        || !cdata::is_cdata_instance(args[0])
        || !unsafe { pyre_object::is_bytes(args[2]) }
    {
        return Err(pyre_interpreter::PyError::type_error(
            "PyOS_snprintf needs buffer, size and format",
        ));
    }
    // `args` is the gateway's native copy; `int_w` and `cdata_write` can
    // collect, so every argument is read from its slot.
    let _roots = pyre_object::gc_roots::push_roots();
    let args_base = _roots.pin_roots(args);
    let capacity =
        pyre_interpreter::baseobjspace::int_w(_roots.get(args_base + 1))?.max(0) as usize;
    let format =
        unsafe { pyre_object::bytesobject::w_bytes_data(_roots.get(args_base + 2)) }.to_vec();
    let mut rendered = Vec::new();
    let mut arg = 3usize;
    let mut i = 0usize;
    while i < format.len() {
        if format[i] == b'%' && i + 1 < format.len() && matches!(format[i + 1], b's' | b'd') {
            if arg >= args.len() {
                return Err(pyre_interpreter::PyError::type_error(
                    "not enough arguments for format string",
                ));
            }
            let value = _roots.get(args_base + arg);
            arg += 1;
            if format[i + 1] == b's' {
                if !unsafe { pyre_object::is_bytes(value) } {
                    return Err(pyre_interpreter::PyError::type_error("%s requires bytes"));
                }
                rendered
                    .extend_from_slice(unsafe { pyre_object::bytesobject::w_bytes_data(value) });
            } else {
                rendered.extend_from_slice(
                    pyre_interpreter::baseobjspace::int_w(value)?
                        .to_string()
                        .as_bytes(),
                );
            }
            i += 2;
        } else {
            rendered.push(format[i]);
            i += 1;
        }
    }
    let write_len = rendered.len().min(capacity.saturating_sub(1));
    cdata::cdata_write(_roots.get(args_base), 0, &rendered[..write_len]);
    if capacity > 0 {
        cdata::cdata_write(_roots.get(args_base), write_len, &[0]);
    }
    Ok(pyre_object::w_int_new(rendered.len() as i64))
}

/// The `StgInfo.paramfunc` of a cdata instance's type, or
/// [`ParamFunc::Other`] when it carries no `StgInfo`.
fn cdata_paramfunc(obj: PyObjectRef) -> ParamFunc {
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    stginfo::stginfo_of(cls)
        .map(stginfo::stginfo_paramfunc)
        .unwrap_or(ParamFunc::Other)
}

/// What a declared argument type lowers to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ArgKind {
    /// A pointer metaclass type, an array type — which decays — or a simple
    /// pointer code like `P`/`z`/`Z`.
    Pointer,
    /// A struct or union, passed by value.
    Aggregate,
    /// A simple type, passed as the bytes its code describes.
    Simple(TypeCode),
}

/// Settle what argument type `at` lowers to.
///
/// The three questions a marshalled argument asks — pointer? aggregate? which
/// code? — all read the same `StgInfo` and the same `_type_`, so they are
/// asked together and the reads happen once.
pub(super) fn classify_argtype(at: PyObjectRef) -> Option<ArgKind> {
    let info = stginfo::stginfo_of(at);
    let paramfunc = info.map_or(ParamFunc::Other, stginfo::stginfo_paramfunc);
    if info.is_some_and(|i| stginfo::stginfo_flags(i) & stginfo::TYPEFLAG_ISPOINTER != 0)
        || paramfunc == ParamFunc::Array
    {
        return Some(ArgKind::Pointer);
    }
    let code = cdata::type_code_of(at);
    if code.is_some_and(|c| cdata::is_pointer_code(&c)) || is_funcptr_type(at) {
        return Some(ArgKind::Pointer);
    }
    if matches!(paramfunc, ParamFunc::Struct | ParamFunc::Union) {
        return Some(ArgKind::Aggregate);
    }
    code.map(ArgKind::Simple)
}

/// Whether argument type `at` lowers to a pointer (a pointer metaclass type,
/// an array type — which decays — or a simple pointer code like `P`/`z`/`Z`).
pub(super) fn argtype_is_pointer_kind(at: PyObjectRef) -> bool {
    classify_argtype(at) == Some(ArgKind::Pointer)
}

/// Whether `t` is a concrete foreign-function type (`CFUNCTYPE`/`WINFUNCTYPE`).
pub(super) fn is_funcptr_type(t: PyObjectRef) -> bool {
    !t.is_null()
        && unsafe { pyre_object::is_type(t) }
        && !std::ptr::eq(t, cfuncptr_type())
        && unsafe { pyre_interpreter::baseobjspace::lookup_in_type(t, "_flags_") }.is_some()
}

fn is_funcptr_instance(obj: PyObjectRef) -> bool {
    !obj.is_null() && unsafe { pyre_interpreter::baseobjspace::isinstance_w(obj, cfuncptr_type()) }
}

/// Append the field types declared in one `_fields_` sequence to `out`
/// (2-tuples only; bit fields are rejected at class-definition time).
fn collect_field_types(
    fields: PyObjectRef,
    out: &mut Vec<PyObjectRef>,
) -> Result<(), pyre_interpreter::PyError> {
    let items = seq_to_vec(fields)
        .ok_or_else(|| pyre_interpreter::PyError::type_error("_fields_ must be a sequence"))?;
    for it in items {
        if !unsafe { pyre_object::is_tuple(it) } {
            return Err(pyre_interpreter::PyError::type_error(
                "_fields_ entries must be tuples",
            ));
        }
        let ft = unsafe { pyre_object::w_tuple_getitem(it, 1) }.unwrap_or(pyre_object::PY_NULL);
        if ft.is_null() || !unsafe { pyre_object::is_type(ft) } {
            return Err(pyre_interpreter::PyError::type_error(
                "field type must be a ctypes type",
            ));
        }
        out.push(ft);
    }
    Ok(())
}

/// The full, base-first field types of a struct/union type.  A subclass's
/// `_fields_` lists only its own fields, so the inherited prefix is gathered by
/// walking the MRO from the least-derived ancestor down to `t`.
fn struct_field_types(t: PyObjectRef) -> Result<Vec<PyObjectRef>, pyre_interpreter::PyError> {
    let mro = unsafe { pyre_object::typeobject::w_type_get_mro(t) };
    if mro.is_null() {
        return Err(pyre_interpreter::PyError::type_error(
            "struct/union type has no '_fields_'",
        ));
    }
    let mut out = Vec::new();
    let mut found = false;
    for &cls in unsafe { (*mro).as_slice() }.iter().rev() {
        if let Some(fields) = pyre_interpreter::type_dict_lookup(cls, "_fields_") {
            found = true;
            collect_field_types(fields, &mut out)?;
        }
    }
    if !found {
        return Err(pyre_interpreter::PyError::type_error(
            "struct/union type has no '_fields_'",
        ));
    }
    Ok(out)
}

/// Build the recursive `CTypeLayout` of a ctypes type, driven by its `StgInfo`
/// `paramfunc`: simple → code, pointer → `Pointer`, array → element layout +
/// length, struct/union → per-field layouts from `_fields_`.
pub(super) fn build_layout(
    mut t: PyObjectRef,
) -> Result<host_ctypes::CTypeLayout, pyre_interpreter::PyError> {
    use host_ctypes::CTypeLayout;
    let mut info = stginfo::stginfo_of(t)
        .ok_or_else(|| pyre_interpreter::PyError::type_error("type has no ctypes layout info"))?;
    let size = pyre_object::with_roots!(info, t => stginfo::stginfo_size(info));
    let paramfunc = pyre_object::with_roots!(info, t => stginfo::stginfo_paramfunc(info));
    match paramfunc {
        ParamFunc::Simple => {
            let tc = cdata::type_code_of(t).ok_or_else(|| {
                pyre_interpreter::PyError::type_error("simple type has no '_type_'")
            })?;
            let ch = tc
                .as_char()
                .ok_or_else(|| pyre_interpreter::PyError::type_error("empty '_type_' code"))?;
            Ok(CTypeLayout::Simple(ch))
        }
        ParamFunc::Pointer => Ok(CTypeLayout::Pointer),
        ParamFunc::Array => {
            let element = pyre_object::with_roots!(info => stginfo::stginfo_proto(info))
                .ok_or_else(|| {
                    pyre_interpreter::PyError::type_error("array type has no element type")
                })?;
            let element = pyre_object::with_roots!(info => build_layout(element))?;
            Ok(CTypeLayout::Array {
                element: Box::new(element),
                length: stginfo::stginfo_length(info),
                size,
            })
        }
        ParamFunc::Struct | ParamFunc::Union => {
            let mut fields = Vec::new();
            for ft in struct_field_types(t)? {
                fields.push(build_layout(ft)?);
            }
            if paramfunc == ParamFunc::Union {
                Ok(CTypeLayout::Union { fields, size })
            } else {
                Ok(CTypeLayout::Struct { fields, size })
            }
        }
        _ => Ok(CTypeLayout::Opaque { size }),
    }
}

/// Marshal a by-value aggregate argument `arg` of type `at`: build the layout
/// and snapshot the instance's buffer bytes (padded to the layout size).
fn marshal_aggregate_arg(
    mut arg: PyObjectRef,
    at: PyObjectRef,
) -> Result<OwnedArg, pyre_interpreter::PyError> {
    let layout = pyre_object::with_roots!(arg => build_layout(at))?;
    let bytes = cdata::cdata_bytes(arg).ok_or_else(|| {
        pyre_interpreter::PyError::type_error(
            "by-value aggregate argument is not a ctypes instance",
        )
    })?;
    let buf = host_ctypes::copy_to_sized_bytes(bytes, layout.size());
    Ok(OwnedArg::Aggregate(layout, buf))
}

/// Create a fresh instance of aggregate type `ty` whose owned buffer holds the
/// returned `bytes`.
fn make_aggregate_instance(
    mut ty: PyObjectRef,
    bytes: &[u8],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let size = stginfo::stginfo_of(ty)
        .map(stginfo::stginfo_size)
        .unwrap_or(bytes.len());
    let mut ba = pyre_object::with_roots!(ty => pyre_object::w_bytearray_new(size));
    let n = bytes.len().min(size);
    unsafe {
        pyre_object::w_bytearray_data_mut(ba)[..n].copy_from_slice(&bytes[..n]);
    }
    let mut obj = pyre_object::w_instance_new(ty);
    let d =
        pyre_object::with_roots!(ba, obj => pyre_interpreter::baseobjspace::getdict_native(obj));
    if d.is_null() {
        return Err(pyre_interpreter::PyError::type_error(
            "aggregate instance has no instance dict",
        ));
    }
    unsafe { pyre_object::w_dict_setitem_str(d, "_b_", ba) };
    Ok(obj)
}

/// Marshal one argument that has an explicit `argtype` `at`, already settled
/// into the shape `kind` names.
fn marshal_typed_arg(
    arg: PyObjectRef,
    at: PyObjectRef,
    kind: ArgKind,
    keepalive: &mut Vec<Vec<u8>>,
) -> Result<OwnedArg, pyre_interpreter::PyError> {
    match kind {
        ArgKind::Pointer => Ok(OwnedArg::Pointer(pointer_argument_addr(
            arg, at, keepalive,
        )?)),
        ArgKind::Aggregate => marshal_aggregate_arg(arg, at),
        ArgKind::Simple(tc) => marshal_simple_arg(arg, tc),
    }
}

/// Marshal one argument against the simple type code `tc`.
///
/// `PyCSimpleType.from_param` hands an instance of the argtype itself straight
/// through and converts anything else, so a mismatched cdata cannot be
/// reinterpreted through the wrong argtype.
fn marshal_simple_arg(
    arg: PyObjectRef,
    tc: TypeCode,
) -> Result<OwnedArg, pyre_interpreter::PyError> {
    if let Some(bytes) = cdata::same_type_bytes(&tc, arg) {
        return Ok(OwnedArg::Typed(tc, bytes));
    }
    let buf = cdata::encode_value(&tc, arg)?;
    Ok(OwnedArg::Typed(tc, buf))
}

/// Marshal one argument with no explicit `argtype` (ConvParam defaults).
fn marshal_default_arg(
    arg: PyObjectRef,
    keepalive: &mut Vec<Vec<u8>>,
) -> Result<OwnedArg, pyre_interpreter::PyError> {
    // `byref()` carrier → the address it wraps.
    if super::interp_ctypes::is_carg(arg) {
        return Ok(OwnedArg::Pointer(super::interp_ctypes::carg_ptr(arg)));
    }
    if is_funcptr_instance(arg) {
        return Ok(OwnedArg::Pointer(funcptr_addr(arg)));
    }
    // A scalar cdata is passed by value.
    if cdata::is_simplecdata_instance(arg) {
        let cls = unsafe { pyre_object::w_instance_get_type(arg) };
        let tc = cdata::type_code_of(cls).ok_or_else(|| {
            pyre_interpreter::PyError::type_error("argument type has no '_type_'")
        })?;
        let buf = cdata::cdata_bytes(arg).unwrap_or(&[]).to_vec();
        return Ok(OwnedArg::Typed(tc, buf));
    }
    // Aggregate / pointer cdata: arrays and pointers decay to a pointer; a
    // struct/union with no `byref()` is passed by value.
    if cdata::is_cdata_instance(arg) {
        match cdata_paramfunc(arg) {
            ParamFunc::Pointer => {
                return Ok(OwnedArg::Pointer(host_ctypes::read_pointer_from_buffer(
                    cdata::cdata_bytes(arg).unwrap_or(&[]),
                )));
            }
            ParamFunc::Array => {
                return Ok(OwnedArg::Pointer(cdata::cdata_addr(arg).unwrap_or(0)));
            }
            ParamFunc::Struct | ParamFunc::Union => {
                let cls = unsafe { pyre_object::w_instance_get_type(arg) };
                return marshal_aggregate_arg(arg, cls);
            }
            _ => {}
        }
    }
    if unsafe { pyre_object::is_none(arg) } {
        Ok(OwnedArg::Pointer(0))
    } else if unsafe { pyre_object::is_bytes(arg) } {
        Ok(OwnedArg::Pointer(bytes_pointer_addr(arg, keepalive)))
    } else if unsafe { pyre_object::is_str(arg) } {
        Ok(OwnedArg::Pointer(wstr_pointer_addr(arg, keepalive)))
    } else if unsafe { pyre_object::is_float(arg) } {
        Ok(OwnedArg::Double(pyre_interpreter::baseobjspace::float_w(
            arg,
        )?))
    } else if unsafe { pyre_object::is_int(arg) } {
        Ok(OwnedArg::Int(
            pyre_interpreter::baseobjspace::int_w(arg)? as i32
        ))
    } else {
        Err(pyre_interpreter::PyError::type_error(
            "Don't know how to convert parameter",
        ))
    }
}

/// The address a pointer-kind argtype takes its argument as.
///
/// `PyCPointerType.from_param` accepts an instance of the type a `POINTER(T)`
/// argtype points at by taking its address, which is what lets `f(c_int(5))`
/// fill in a callee's `int *` — the address of the box, not the number in it.
fn pointer_argument_addr(
    mut arg: PyObjectRef,
    at: PyObjectRef,
    keepalive: &mut Vec<Vec<u8>>,
) -> Result<usize, pyre_interpreter::PyError> {
    if let Some(mut info) = stginfo::stginfo_of(at)
        && pyre_object::with_roots!(arg, info => stginfo::stginfo_paramfunc(info))
            == ParamFunc::Pointer
        && let Some(proto) = pyre_object::with_roots!(arg => stginfo::stginfo_proto(info))
        && unsafe { pyre_object::is_type(proto) }
        && unsafe { pyre_interpreter::baseobjspace::isinstance_w(arg, proto) }
        && let Some(addr) = pyre_object::with_roots!(arg => cdata::cdata_addr(arg))
    {
        return Ok(addr);
    }
    resolve_pointer_addr(arg, keepalive)
}

/// Resolve the address a pointer-kind argument lowers to: `byref()` carriers,
/// `_Pointer`/`Array`/`Structure` instances, pointer-typed scalars, bytes, an
/// integer address, or `None`.
pub(super) fn resolve_pointer_addr(
    arg: PyObjectRef,
    keepalive: &mut Vec<Vec<u8>>,
) -> Result<usize, pyre_interpreter::PyError> {
    if super::interp_ctypes::is_carg(arg) {
        return Ok(super::interp_ctypes::carg_ptr(arg));
    }
    if is_funcptr_instance(arg) {
        return Ok(funcptr_addr(arg));
    }
    if cdata::is_simplecdata_instance(arg) {
        // A pointer-typed scalar stores the target address in its buffer.
        return Ok(host_ctypes::read_pointer_from_buffer(
            cdata::cdata_bytes(arg).unwrap_or(&[]),
        ));
    }
    if cdata::is_cdata_instance(arg) {
        // `_Pointer` → stored address; `Array`/`Structure` → buffer address.
        return Ok(match cdata_paramfunc(arg) {
            ParamFunc::Pointer => {
                host_ctypes::read_pointer_from_buffer(cdata::cdata_bytes(arg).unwrap_or(&[]))
            }
            _ => cdata::cdata_addr(arg).unwrap_or(0),
        });
    }
    if unsafe { pyre_object::is_none(arg) } {
        Ok(0)
    } else if unsafe { pyre_object::pyobject::is_int_or_long(arg) } {
        Ok(cdata::pointer_word(arg)?)
    } else if unsafe { pyre_object::is_bytes(arg) } {
        Ok(bytes_pointer_addr(arg, keepalive))
    } else if unsafe { pyre_object::is_str(arg) } {
        Ok(wstr_pointer_addr(arg, keepalive))
    } else {
        Err(pyre_interpreter::PyError::type_error(
            "expected bytes, integer address, ctypes instance, or None",
        ))
    }
}

/// Null-terminate a `bytes` payload, keep the copy alive, and return the
/// address of the copy.
fn bytes_pointer_addr(arg: PyObjectRef, keepalive: &mut Vec<Vec<u8>>) -> usize {
    let raw = unsafe { pyre_object::bytesobject::w_bytes_data(arg) };
    keepalive.push(host_ctypes::clone_as_null_terminated(raw));
    // The inner Vec's heap buffer is stable even if `keepalive` reallocates.
    keepalive.last().unwrap().as_ptr() as usize
}

/// Copy a `str` into a NUL-terminated `wchar_t` buffer, keep the copy alive,
/// and return its address.  `_conv_param` reaches the same place by building
/// a `c_wchar_p(arg)` and taking its ffi parameter, and `ConvParam` by
/// `PyUnicode_AsWideCharString` with the buffer held in a capsule for the
/// duration of the call; `c_wchar_p`'s own setter (`encode_value_into`, the
/// `Z` arm) spells the copy the same way.
fn wstr_pointer_addr(arg: PyObjectRef, keepalive: &mut Vec<Vec<u8>>) -> usize {
    let raw = unsafe { pyre_object::w_str_get_wtf8(arg) };
    keepalive.push(host_ctypes::clone_wchar_null_terminated(raw));
    keepalive.last().unwrap().as_ptr() as usize
}
