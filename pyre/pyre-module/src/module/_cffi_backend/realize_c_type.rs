//! C parser opcode realization — PyPy: `pypy/module/_cffi_backend/realize_c_type.py`.

use pyre_interpreter::PyError;
use pyre_object::PyObjectRef;
use std::ffi::CStr;
use std::sync::{Condvar, Mutex, OnceLock};

use super::ctypeobj;
use super::ffi_obj::W_FFIObject;
use super::{newtype, parse_c_type};

const NAMES: [Option<&str>; parse_c_type::NUM_PRIM] = [
    None,
    Some("_Bool"),
    Some("char"),
    Some("signed char"),
    Some("unsigned char"),
    Some("short"),
    Some("unsigned short"),
    Some("int"),
    Some("unsigned int"),
    Some("long"),
    Some("unsigned long"),
    Some("long long"),
    Some("unsigned long long"),
    Some("float"),
    Some("double"),
    Some("long double"),
    Some("wchar_t"),
    Some("int8_t"),
    Some("uint8_t"),
    Some("int16_t"),
    Some("uint16_t"),
    Some("int32_t"),
    Some("uint32_t"),
    Some("int64_t"),
    Some("uint64_t"),
    Some("intptr_t"),
    Some("uintptr_t"),
    Some("ptrdiff_t"),
    Some("size_t"),
    Some("ssize_t"),
    Some("int_least8_t"),
    Some("uint_least8_t"),
    Some("int_least16_t"),
    Some("uint_least16_t"),
    Some("int_least32_t"),
    Some("uint_least32_t"),
    Some("int_least64_t"),
    Some("uint_least64_t"),
    Some("int_fast8_t"),
    Some("uint_fast8_t"),
    Some("int_fast16_t"),
    Some("uint_fast16_t"),
    Some("int_fast32_t"),
    Some("uint_fast32_t"),
    Some("int_fast64_t"),
    Some("uint_fast64_t"),
    Some("intmax_t"),
    Some("uintmax_t"),
    Some("_cffi_float_complex_t"),
    Some("_cffi_double_complex_t"),
    Some("char16_t"),
    Some("char32_t"),
];

#[derive(Default)]
struct RealizeLockState {
    owner: Option<std::thread::ThreadId>,
    rec_level: usize,
}

/// `RealizeCache.__enter__` / `__exit__`.
struct RealizeLock {
    state: Mutex<RealizeLockState>,
    ready: Condvar,
}

impl RealizeLock {
    fn new() -> Self {
        Self {
            state: Mutex::new(RealizeLockState::default()),
            ready: Condvar::new(),
        }
    }

    fn enter(&self) -> RealizeGuard<'_> {
        let current = std::thread::current().id();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.owner.as_ref().is_some_and(|owner| *owner != current) {
            let _blocked = pyre_interpreter::module::thread::before_external_block();
            state = self
                .ready
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state.owner = Some(current);
        state.rec_level += 1;
        let rec_level = state.rec_level;
        RealizeGuard {
            lock: self,
            rec_level,
        }
    }
}

struct RealizeGuard<'a> {
    lock: &'a RealizeLock,
    rec_level: usize,
}

impl Drop for RealizeGuard<'_> {
    fn drop(&mut self) {
        let mut state = self
            .lock
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.rec_level -= 1;
        if state.rec_level == 0 {
            state.owner = None;
            self.lock.ready.notify_one();
        }
    }
}

fn realize_lock() -> &'static RealizeLock {
    static LOCK: OnceLock<RealizeLock> = OnceLock::new();
    LOCK.get_or_init(RealizeLock::new)
}

/// Keep the parser's shared output and realization under the recursive lock.
pub fn with_realize_lock<T>(f: impl FnOnce() -> Result<T, PyError>) -> Result<T, PyError> {
    let _guard = realize_lock().enter();
    f()
}

fn ffi_error(message: impl Into<String>) -> PyError {
    let message = message.into();
    // `newtype.py detect_custom_layout` raises `oefmt(w_FFIError, ...)`,
    // which is `OperationError(w_type, space.newtext(msg))`.
    let cls = newtype::ffi_error();
    let _roots = pyre_object::gc_roots::push_roots();
    let cls_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(cls);
    let w_value = pyre_object::w_str_new_managed(&message);
    PyError::from_type_and_value(pyre_object::gc_roots::shadow_stack_get(cls_slot), w_value)
}

fn c_string(ptr: *const core::ffi::c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn signed_size(value: usize) -> i64 {
    value as isize as i64
}

fn ffi_arg(w_ffi: PyObjectRef) -> Result<&'static mut W_FFIObject, PyError> {
    W_FFIObject::from_obj(w_ffi).ok_or_else(|| PyError::type_error("expected an FFI object"))
}

/// `RealizeCache.get_file_struct`.
fn get_file_struct() -> PyObjectRef {
    static FILE_STRUCT: pyre_object::gc_roots::RootedOnceRef =
        pyre_object::gc_roots::RootedOnceRef::new();
    FILE_STRUCT.get_or_init(|| newtype::new_struct_type("FILE"))
}

/// `get_primitive_type`.
pub fn get_primitive_type(w_ffi: PyObjectRef, num: isize) -> Result<PyObjectRef, PyError> {
    if !(0..parse_c_type::NUM_PRIM as isize).contains(&num) {
        return Err(match num {
            parse_c_type::UNKNOWN_PRIM => ffi_error(
                "primitive integer type with an unexpected size (or not an integer type at all)",
            ),
            parse_c_type::UNKNOWN_FLOAT_PRIM => ffi_error(
                "primitive floating-point type with an unexpected size (or not a float type at all)",
            ),
            parse_c_type::UNKNOWN_LONG_DOUBLE => ffi_error(
                "primitive floating-point type is 'long double', not supported for now with the syntax 'typedef double... xxx;'",
            ),
            _ => PyError::not_implemented(format!("prim={num}")),
        });
    }
    let num = num as usize;
    if num == parse_c_type::PRIM_VOID {
        return Ok(newtype::new_void_type());
    }
    let _ = w_ffi;
    newtype::new_primitive_type(NAMES[num].expect("every non-void primitive has a name"))
}

/// `get_array_type`.
fn get_array_type(
    w_ffi: PyObjectRef,
    opcodes: *mut parse_c_type::OpcodeT,
    itemindex: isize,
    length: i64,
) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let item_slot = roots.base();
    let _ = roots.pin_root(realize_c_type(w_ffi, opcodes, itemindex)?);
    let ptr_slot = item_slot + 1;
    let _ = roots.pin_root(newtype::new_pointer_type(roots.get(item_slot))?);
    newtype::new_array_type(roots.get(ptr_slot), length)
}

/// `realize_global_int`.
pub fn realize_global_int(
    w_ffi: PyObjectRef,
    g: parse_c_type::GlobalS,
    gindex: isize,
) -> Result<PyObjectRef, PyError> {
    type Fetch = unsafe extern "C" fn(*mut parse_c_type::GetConstS) -> core::ffi::c_int;
    let fetch: Fetch = unsafe { core::mem::transmute(g.address) };
    let ffi = ffi_arg(w_ffi)?;
    let mut value = parse_c_type::GetConstS {
        value: 0,
        ctx: unsafe { &(*ffi.ctxobj).ctx },
        gindex: gindex as core::ffi::c_int,
    };
    let neg = unsafe { fetch(&mut value) };
    let raw = value.value;
    match neg {
        0 if raw <= i64::MAX as u64 => Ok(pyre_object::w_int_new(raw as i64)),
        0 => Ok(pyre_object::longobject::w_long_new(
            pyre_interpreter::PyBigInt::from_u128(raw as u128),
        )),
        1 => Ok(pyre_object::w_int_new(raw as i64)),
        2 => Err(ffi_error(format!(
            "the C compiler says '{}' is equal to {} (0x{:x}), but the cdef disagrees",
            c_string(g.name),
            raw,
            raw
        ))),
        _ => Err(ffi_error(format!(
            "the C compiler says '{}' is equal to {}, but the cdef disagrees",
            c_string(g.name),
            raw as i64
        ))),
    }
}

/// Temporary representation of an `OP_FUNCTION` that has not yet been
/// required to be a pointer-to-function.
#[pyre_interpreter::pyre_class("_cffi_backend.__RawFuncType")]
// `W_RawFuncType._immutable_fields_`: `prepare_nostruct_fnptr` fills these
// three in once, before any caller can read them, and nothing writes them
// again.  A call through a promoted raw function type folds its
// `nostruct_ctype` — and with it the function type's `cif_descr` — to a trace
// constant only because of this declaration.
#[majit_macros::jit_immutable_fields("nostruct_ctype", "nostruct_locs", "nostruct_nargs")]
#[derive(Default)]
pub struct W_RawFuncType {
    pub opcodes: *mut parse_c_type::OpcodeT,
    pub base_index: i64,
    pub ctfuncptr: PyObjectRef,
    pub nostruct_ctype: PyObjectRef,
    pub nostruct_locs: PyObjectRef,
    pub nostruct_nargs: i64,
}

static RAW_FUNC_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

fn raw_func_type() -> PyObjectRef {
    RAW_FUNC_TYPE_OBJ.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type_with_layout(
            "_cffi_backend.__RawFuncType",
            |_| {},
            pyre_interpreter::typedef::w_object(),
            <W_RawFuncType as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_RawFuncType as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        unsafe {
            pyre_object::w_type_set_disallow_instantiation(tp);
            pyre_object::w_type_set_acceptable_as_base_class(tp, false);
        }
        tp
    })
}

fn new_raw_func(opcodes: *mut parse_c_type::OpcodeT, base_index: isize) -> PyObjectRef {
    let _ = raw_func_type();
    W_RawFuncType::allocate_stable(W_RawFuncType {
        opcodes,
        base_index: base_index as i64,
        ..Default::default()
    })
}

impl W_RawFuncType {
    /// `W_RawFuncType._unpack`.
    fn unpack(
        &self,
        w_ffi: PyObjectRef,
    ) -> Result<(Vec<PyObjectRef>, PyObjectRef, bool, i64), PyError> {
        let roots = pyre_object::gc_roots::push_roots();
        let ffi_slot = roots.base();
        let _ = roots.pin_root(w_ffi);
        let mut index = self.base_index as isize;
        let op = unsafe { *self.opcodes.offset(index) };
        assert_eq!(parse_c_type::getop(op), parse_c_type::OP_FUNCTION);
        let fret_slot = ffi_slot + 1;
        let _ = roots.pin_root(realize_c_type(
            roots.get(ffi_slot),
            self.opcodes,
            parse_c_type::getarg(op),
        )?);
        index += 1;
        let mut num_args = 0isize;
        while parse_c_type::getop(unsafe { *self.opcodes.offset(index + num_args) })
            != parse_c_type::OP_FUNCTION_END
        {
            num_args += 1;
        }
        let end = unsafe { *self.opcodes.offset(index + num_args) };
        let endarg = parse_c_type::getarg(end);
        let ellipsis = endarg & 1 != 0;
        let encoded_abi = endarg & !1;
        let abi = match encoded_abi {
            0 => super::interp_cffi_backend::default_abi() as i64,
            2 => super::ctypefunc::stdcall_abi()
                .unwrap_or_else(|| super::interp_cffi_backend::default_abi() as i64),
            _ => return Err(ffi_error(format!("abi number {encoded_abi} not supported"))),
        };
        let args_slot = pyre_object::gc_roots::shadow_stack_len();
        for i in 0..num_args {
            let w_arg = realize_c_type(roots.get(ffi_slot), self.opcodes, index + i)?;
            let _ = roots.pin_root(w_arg);
        }
        let fargs = (0..num_args as usize)
            .map(|i| roots.get(args_slot + i))
            .collect();
        Ok((fargs, roots.get(fret_slot), ellipsis, abi))
    }

    /// `W_RawFuncType.unwrap_as_fnptr`.
    pub fn unwrap_as_fnptr(w_raw: PyObjectRef, w_ffi: PyObjectRef) -> Result<PyObjectRef, PyError> {
        let roots = pyre_object::gc_roots::push_roots();
        let raw_slot = roots.base();
        let _ = roots.pin_root(w_raw);
        let ffi_slot = raw_slot + 1;
        let _ = roots.pin_root(w_ffi);
        let raw = W_RawFuncType::from_obj(roots.get(raw_slot))
            .ok_or_else(|| PyError::system_error("expected a raw function type"))?;
        if raw.ctfuncptr.is_null() {
            let (fargs, fret, ellipsis, abi) = raw.unpack(roots.get(ffi_slot))?;
            let ctfuncptr = newtype::build_function_type(&fargs, fret, ellipsis, abi)?;
            W_RawFuncType::from_obj(roots.get(raw_slot))
                .expect("the rooted raw function keeps its layout")
                .ctfuncptr = ctfuncptr;
            pyre_object::gc_hook::try_gc_write_barrier_managed(
                roots.get(raw_slot) as pyre_object::gc_hook::GCREF
            );
        }
        Ok(W_RawFuncType::from_obj(roots.get(raw_slot))
            .expect("the rooted raw function keeps its layout")
            .ctfuncptr)
    }

    /// `W_RawFuncType.unwrap_as_fnptr_in_elidable`.
    pub fn unwrap_as_fnptr_in_elidable(&self) -> Result<PyObjectRef, PyError> {
        if self.ctfuncptr.is_null() {
            Err(PyError::system_error("raw function type was not forced"))
        } else {
            Ok(self.ctfuncptr)
        }
    }

    /// `W_RawFuncType.prepare_nostruct_fnptr`.
    pub fn prepare_nostruct_fnptr(w_raw: PyObjectRef, w_ffi: PyObjectRef) -> Result<(), PyError> {
        let roots = pyre_object::gc_roots::push_roots();
        let raw_slot = roots.base();
        let _ = roots.pin_root(w_raw);
        let ffi_slot = raw_slot + 1;
        let _ = roots.pin_root(w_ffi);
        let raw = W_RawFuncType::from_obj(roots.get(raw_slot))
            .ok_or_else(|| PyError::system_error("expected a raw function type"))?;
        if !raw.nostruct_ctype.is_null() {
            return Ok(());
        }
        let (mut fargs, mut fret, ellipsis, abi) = raw.unpack(roots.get(ffi_slot))?;
        let fret_slot = ffi_slot + 1;
        let _ = roots.pin_root(fret);
        let mut locs = vec![0u8; fargs.len()];
        for i in 0..fargs.len() {
            let ct = ctypeobj::ctype_arg(fargs[i])?;
            if ct.is_struct_or_union() || ct.kind == ctypeobj::KIND_PRIM_COMPLEX {
                fargs[i] = newtype::new_pointer_type(fargs[i])?;
                locs[i] = b'A';
            }
        }
        fret = roots.get(fret_slot);
        let fret_ct = ctypeobj::ctype_arg(fret)?;
        if fret_ct.is_struct_or_union() || fret_ct.kind == ctypeobj::KIND_PRIM_COMPLEX {
            fret = newtype::new_pointer_type(fret)?;
            fargs.insert(0, fret);
            locs.insert(0, b'R');
            fret = newtype::new_void_type();
        }
        let nostruct_ctype = newtype::build_function_type(&fargs, fret, ellipsis, abi)?;
        let raw = W_RawFuncType::from_obj(roots.get(raw_slot))
            .expect("the rooted raw function keeps its layout");
        raw.nostruct_ctype = nostruct_ctype;
        if locs.iter().any(|&c| c != 0) {
            raw.nostruct_locs = pyre_object::bytesobject::w_bytes_from_bytes(&locs);
        }
        raw.nostruct_nargs = fargs.len() as i64 - i64::from(locs.first() == Some(&b'R'));
        pyre_object::gc_hook::try_gc_write_barrier_managed(
            roots.get(raw_slot) as pyre_object::gc_hook::GCREF
        );
        Ok(())
    }

    /// `W_RawFuncType.repr_fn_type`.
    pub fn repr_fn_type(&self, w_ffi: PyObjectRef, repl: &str) -> Result<String, PyError> {
        let (fargs, fret, ellipsis, _) = self.unpack(w_ffi)?;
        let mut argnames: Vec<&str> = fargs
            .iter()
            .map(|&arg| ctypeobj::ctype_arg(arg).map(|ct| ct.name()))
            .collect::<Result<_, _>>()?;
        if ellipsis {
            argnames.push("...");
        }
        let fret = ctypeobj::ctype_arg(fret)?;
        let at = fret.name_position as usize;
        let mut replacement = repl.to_string();
        if !replacement.is_empty() && !fret.name()[..at].ends_with('*') {
            replacement.insert(0, ' ');
        }
        Ok(format!(
            "{}{}({}){}",
            &fret.name()[..at],
            replacement,
            argnames.join(", "),
            &fret.name()[at..]
        ))
    }

    /// `W_RawFuncType.unexpected_fn_type`.
    pub fn unexpected_fn_type(&self, w_ffi: PyObjectRef) -> PyError {
        let name = self
            .repr_fn_type(w_ffi, "")
            .unwrap_or_else(|_| "<function>".to_string());
        ffi_error(format!(
            "the type '{name}' is a function type, not a pointer-to-function type"
        ))
    }
}

fn cached_type(ffi: &W_FFIObject, index: isize) -> Option<PyObjectRef> {
    if ffi.cached_types.is_null() || index < 0 {
        return None;
    }
    unsafe { pyre_object::listobject::w_list_getitem(ffi.cached_types, index as i64) }
        .filter(|&item| !item.is_null() && !unsafe { pyre_object::pyobject::is_none(item) })
}

fn set_cached_type(ffi: &W_FFIObject, index: isize, value: PyObjectRef) {
    if !ffi.cached_types.is_null() && index >= 0 {
        assert!(unsafe {
            pyre_object::listobject::w_list_setitem(ffi.cached_types, index as i64, value)
        });
    }
}

/// `realize_c_type`.
pub fn realize_c_type(
    w_ffi: PyObjectRef,
    opcodes: *mut parse_c_type::OpcodeT,
    index: isize,
) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let ffi_slot = roots.base();
    let _ = roots.pin_root(w_ffi);
    let x_slot = ffi_slot + 1;
    let _ = roots.pin_root(realize_c_type_or_func(roots.get(ffi_slot), opcodes, index)?);
    let x = roots.get(x_slot);
    if ctypeobj::ctype_at(x).is_some() {
        return Ok(x);
    }
    let raw = W_RawFuncType::from_obj(x)
        .ok_or_else(|| PyError::system_error("realization returned an unknown object"))?;
    Err(raw.unexpected_fn_type(roots.get(ffi_slot)))
}

/// `_realize_name`.
fn realize_name(prefix: &str, src_name: *const core::ffi::c_char) -> String {
    let name = c_string(src_name);
    let bytes = name.as_bytes();
    if bytes.first() == Some(&b'$')
        && bytes.get(1) != Some(&b'$')
        && !bytes.get(1).is_some_and(u8::is_ascii_digit)
    {
        name[1..].to_string()
    } else {
        format!("{prefix}{name}")
    }
}

/// `_realize_c_struct_or_union`.
fn realize_c_struct_or_union(w_ffi: PyObjectRef, sindex: isize) -> Result<PyObjectRef, PyError> {
    if sindex == -1 {
        return Ok(get_file_struct());
    }
    let roots = pyre_object::gc_roots::push_roots();
    let ffi_slot = roots.base();
    let _ = roots.pin_root(w_ffi);
    let ffi = ffi_arg(roots.get(ffi_slot))?;
    let ctx = unsafe { &(*ffi.ctxobj).ctx };
    if sindex < 0 || sindex >= ctx.num_struct_unions as isize {
        return Err(PyError::system_error("struct/union index out of range"));
    }
    let s = unsafe { *ctx.struct_unions.offset(sindex) };
    let type_index = s.type_index as isize;
    if let Some(found) = cached_type(ffi, type_index) {
        return Ok(found);
    }
    let c_flags = s.flags;
    let first_field = s.first_field_index;
    let mut lazy = false;
    let x = if c_flags & parse_c_type::CffiTypeFlags::EXTERNAL.bits() == 0 {
        let (kind, prefix) = if c_flags & parse_c_type::CffiTypeFlags::UNION.bits() != 0 {
            (ctypeobj::KIND_UNION, "union ")
        } else {
            (ctypeobj::KIND_STRUCT, "struct ")
        };
        let name = realize_name(prefix, s.name);
        let x = if name == "struct _IO_FILE" {
            get_file_struct()
        } else if kind == ctypeobj::KIND_UNION {
            newtype::new_union_type(&name)
        } else {
            newtype::new_struct_type(&name)
        };
        if c_flags & parse_c_type::CffiTypeFlags::OPAQUE.bits() == 0 {
            assert!(first_field >= 0);
            let ct = ctypeobj::ctype_arg(x)?;
            ctypeobj::set_size(ct, signed_size(s.size));
            ct.align = s.alignment as i64;
            ct.lazy_ffi = roots.get(ffi_slot);
            ct.lazy_sindex = sindex as i64;
            pyre_object::gc_hook::try_gc_write_barrier_managed(x as pyre_object::gc_hook::GCREF);
            lazy = true;
        } else {
            assert!(first_field < 0);
        }
        x
    } else {
        assert!(first_field < 0);
        let Some(x) = fetch_external_struct_or_union(&s, roots.get(ffi_slot))? else {
            let kind = if c_flags & parse_c_type::CffiTypeFlags::UNION.bits() != 0 {
                "union"
            } else {
                "struct"
            };
            return Err(ffi_error(format!(
                "'{kind} {}' should come from ffi.include() but was not found",
                c_string(s.name)
            )));
        };
        let ct = ctypeobj::ctype_arg(x)?;
        if c_flags & parse_c_type::CffiTypeFlags::OPAQUE.bits() == 0 && ct.size < 0 {
            let kind = if c_flags & parse_c_type::CffiTypeFlags::UNION.bits() != 0 {
                "union"
            } else {
                "struct"
            };
            let name = c_string(s.name);
            return Err(PyError::not_implemented(format!(
                "'{kind} {name}' is opaque in the ffi.include(), but no longer in the ffi doing the include (workaround: don't use ffi.include() but duplicate the declarations of everything using {kind} {name})"
            )));
        }
        x
    };
    let x_slot = ffi_slot + 1;
    let x = roots.pin_root(x);
    set_cached_type(ffi_arg(roots.get(ffi_slot))?, type_index, x);
    if lazy
        && signed_size(s.size) == -2
        && let Err(error) = do_realize_lazy_struct(x)
    {
        set_cached_type(
            ffi_arg(roots.get(ffi_slot))?,
            type_index,
            pyre_object::w_none(),
        );
        return Err(error);
    }
    Ok(roots.get(x_slot))
}

/// `_realize_c_enum`.
fn realize_c_enum(w_ffi: PyObjectRef, eindex: isize) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let ffi_slot = roots.base();
    let _ = roots.pin_root(w_ffi);
    let ffi = ffi_arg(roots.get(ffi_slot))?;
    let ctx = unsafe { &(*ffi.ctxobj).ctx };
    if eindex < 0 || eindex >= ctx.num_enums as isize {
        return Err(PyError::system_error("enum index out of range"));
    }
    let e = unsafe { *ctx.enums.offset(eindex) };
    let type_index = e.type_index as isize;
    if let Some(found) = cached_type(ffi, type_index) {
        return Ok(found);
    }
    let base_slot = ffi_slot + 1;
    let _ = roots.pin_root(get_primitive_type(
        roots.get(ffi_slot),
        e.type_prim as isize,
    )?);
    let names_slot = base_slot + 1;
    let _ = roots.pin_root(pyre_object::w_list_new(Vec::new()));
    let values_slot = names_slot + 1;
    let _ = roots.pin_root(pyre_object::w_list_new(Vec::new()));
    let enumerators = c_string(e.enumerators);
    if !enumerators.is_empty() {
        for enname in enumerators.split(',') {
            let gindex = parse_c_type::search_in_globals(ctx, enname);
            assert!(gindex >= 0);
            let g = unsafe { *ctx.globals.offset(gindex) };
            assert_eq!(parse_c_type::getop(g.type_op), parse_c_type::OP_ENUM);
            assert_eq!(parse_c_type::getarg(g.type_op), -1);
            let part_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(pyre_object::w_str_new(enname));
            let _ = roots.pin_root(realize_global_int(roots.get(ffi_slot), g, gindex)?);
            unsafe {
                pyre_object::listobject::w_list_append(roots.get(names_slot), roots.get(part_slot));
                pyre_object::listobject::w_list_append(
                    roots.get(values_slot),
                    roots.get(part_slot + 1),
                );
            }
        }
    }
    let name = realize_name("enum ", e.name);
    let w_ctype = newtype::new_enum_type(
        &name,
        roots.get(names_slot),
        roots.get(values_slot),
        roots.get(base_slot),
    )?;
    set_cached_type(ffi_arg(roots.get(ffi_slot))?, type_index, w_ctype);
    Ok(w_ctype)
}

/// `realize_c_type_or_func`.
pub fn realize_c_type_or_func(
    w_ffi: PyObjectRef,
    opcodes: *mut parse_c_type::OpcodeT,
    index: isize,
) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let ffi_slot = roots.base();
    let _ = roots.pin_root(w_ffi);
    let ffi = ffi_arg(roots.get(ffi_slot))?;
    let from_ffi = opcodes == unsafe { (*ffi.ctxobj).ctx.types };
    if from_ffi && let Some(found) = cached_type(ffi, index) {
        return Ok(found);
    }
    let guard = realize_lock().enter();
    if from_ffi && let Some(found) = cached_type(ffi_arg(roots.get(ffi_slot))?, index) {
        return Ok(found);
    }
    if guard.rec_level > 1000 {
        return Err(PyError::runtime_error(
            "type-building recursion too deep or infinite.  This is known to occur e.g. in ``struct s { void(*callable)(struct s); }''.  Please report if you get this error and really need support for your case.",
        ));
    }
    let op = unsafe { *opcodes.offset(index) };
    let x = realize_c_type_or_func_now(roots.get(ffi_slot), op, opcodes, index)?;
    if from_ffi {
        if let Some(old) = cached_type(ffi_arg(roots.get(ffi_slot))?, index) {
            assert_eq!(old, x);
        }
        set_cached_type(ffi_arg(roots.get(ffi_slot))?, index, x);
    }
    Ok(x)
}

/// `realize_c_type_or_func_now`.
fn realize_c_type_or_func_now(
    w_ffi: PyObjectRef,
    op: parse_c_type::OpcodeT,
    opcodes: *mut parse_c_type::OpcodeT,
    index: isize,
) -> Result<PyObjectRef, PyError> {
    match parse_c_type::getop(op) {
        parse_c_type::OP_PRIMITIVE => get_primitive_type(w_ffi, parse_c_type::getarg(op)),
        parse_c_type::OP_POINTER => {
            let roots = pyre_object::gc_roots::push_roots();
            let ffi_slot = roots.base();
            let _ = roots.pin_root(w_ffi);
            let value_slot = ffi_slot + 1;
            let _ = roots.pin_root(realize_c_type_or_func(
                roots.get(ffi_slot),
                opcodes,
                parse_c_type::getarg(op),
            )?);
            let value = roots.get(value_slot);
            if ctypeobj::ctype_at(value).is_some() {
                newtype::new_pointer_type(value)
            } else if W_RawFuncType::from_obj(value).is_some() {
                W_RawFuncType::unwrap_as_fnptr(value, roots.get(ffi_slot))
            } else {
                Err(PyError::not_implemented("unknown pointer target"))
            }
        }
        parse_c_type::OP_ARRAY => {
            get_array_type(w_ffi, opcodes, parse_c_type::getarg(op), unsafe {
                *opcodes.offset(index + 1) as isize as i64
            })
        }
        parse_c_type::OP_OPEN_ARRAY => get_array_type(w_ffi, opcodes, parse_c_type::getarg(op), -1),
        parse_c_type::OP_STRUCT_UNION => realize_c_struct_or_union(w_ffi, parse_c_type::getarg(op)),
        parse_c_type::OP_ENUM => realize_c_enum(w_ffi, parse_c_type::getarg(op)),
        parse_c_type::OP_FUNCTION => Ok(new_raw_func(opcodes, index)),
        parse_c_type::OP_NOOP => realize_c_type_or_func(w_ffi, opcodes, parse_c_type::getarg(op)),
        parse_c_type::OP_TYPENAME => {
            let ffi = ffi_arg(w_ffi)?;
            let ctx = unsafe { &(*ffi.ctxobj).ctx };
            let tindex = parse_c_type::getarg(op);
            if tindex < 0 || tindex >= ctx.num_typenames as isize {
                return Err(PyError::system_error("typename index out of range"));
            }
            let typename = unsafe { *ctx.typenames.offset(tindex) };
            realize_c_type_or_func(w_ffi, ctx.types, typename.type_index as isize)
        }
        case => Err(PyError::not_implemented(format!("op={case}"))),
    }
}

/// `do_realize_lazy_struct`.
pub fn do_realize_lazy_struct(w_ctype: PyObjectRef) -> Result<(), PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let ctype_slot = roots.base();
    let _ = roots.pin_root(w_ctype);
    let ct = ctypeobj::ctype_arg(roots.get(ctype_slot))?;
    assert!(ct.is_struct_or_union());
    if ct.lazy_ffi.is_null() {
        return Ok(());
    }
    let ffi_slot = ctype_slot + 1;
    let _ = roots.pin_root(ct.lazy_ffi);
    let ffi = ffi_arg(roots.get(ffi_slot))?;
    let ctx = unsafe { &(*ffi.ctxobj).ctx };
    let sindex = ctypeobj::ctype_arg(roots.get(ctype_slot))?.lazy_sindex as isize;
    let s = unsafe { *ctx.struct_unions.offset(sindex) };
    assert_ne!(ctypeobj::ctype_arg(roots.get(ctype_slot))?.size, -1);
    let fields_slot = ffi_slot + 1;
    let _ = roots.pin_root(pyre_object::w_list_new(Vec::new()));
    for i in 0..s.num_fields as isize {
        let fld = unsafe { *ctx.fields.offset(s.first_field_index as isize + i) };
        let field_name = c_string(fld.name);
        let field_size = signed_size(fld.field_size);
        let field_offset = signed_size(fld.field_offset);
        let case = parse_c_type::getop(fld.field_type_op);
        let fbitsize = match case {
            parse_c_type::OP_NOOP => -1,
            parse_c_type::OP_BITFIELD => {
                assert!(field_size >= 0);
                field_size
            }
            _ => return Err(PyError::not_implemented(format!("field op={case}"))),
        };
        let part_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(realize_c_type(
            roots.get(ffi_slot),
            ctx.types,
            parse_c_type::getarg(fld.field_type_op),
        )?);
        let w_ctf = roots.get(part_slot);
        if field_offset == -1 {
            assert!(field_size == -1 || fbitsize >= 0);
        } else {
            newtype::detect_custom_layout(
                ctypeobj::ctype_arg(roots.get(ctype_slot))?,
                newtype::StructFlags::STD_FIELD_POS.bits(),
                ctypeobj::ctype_arg(w_ctf)?.size,
                field_size,
                &format!("wrong size for field '{field_name}'"),
            )?;
        }
        let _ = roots.pin_root(pyre_object::w_str_new_managed(&field_name));
        let _ = roots.pin_root(pyre_object::w_int_new(fbitsize));
        let _ = roots.pin_root(pyre_object::w_int_new(field_offset));
        let tuple_slot = part_slot + 4;
        let _ = roots.pin_root(pyre_object::w_tuple_new(vec![
            roots.get(part_slot + 1),
            roots.get(part_slot),
            roots.get(part_slot + 2),
            roots.get(part_slot + 3),
        ]));
        unsafe {
            pyre_object::listobject::w_list_append(roots.get(fields_slot), roots.get(tuple_slot))
        };
    }
    let mut sflags = 0;
    if s.flags & parse_c_type::CffiTypeFlags::CHECK_FIELDS.bits() != 0 {
        sflags |= newtype::StructFlags::STD_FIELD_POS.bits();
    }
    if s.flags & parse_c_type::CffiTypeFlags::PACKED.bits() != 0 {
        sflags |= newtype::StructFlags::PACKED.bits();
    }
    let old_size = ctypeobj::ctype_arg(roots.get(ctype_slot))?.size;
    let old_align = ctypeobj::ctype_arg(roots.get(ctype_slot))?.align;
    ctypeobj::set_size(ctypeobj::ctype_arg(roots.get(ctype_slot))?, -1);
    if let Err(error) = newtype::complete_struct_or_union(
        roots.get(ctype_slot),
        roots.get(fields_slot),
        signed_size(s.size),
        s.alignment as i64,
        sflags,
        0,
    ) {
        let ct = ctypeobj::ctype_arg(roots.get(ctype_slot))?;
        ctypeobj::set_size(ct, old_size);
        ct.align = old_align;
        return Err(error);
    }
    let ct = ctypeobj::ctype_arg(roots.get(ctype_slot))?;
    ct.lazy_ffi = pyre_object::PY_NULL;
    ct.lazy_sindex = -1;
    pyre_object::gc_hook::try_gc_write_barrier_managed(
        roots.get(ctype_slot) as pyre_object::gc_hook::GCREF
    );
    Ok(())
}

/// `_fetch_external_struct_or_union`.
fn fetch_external_struct_or_union(
    s: &parse_c_type::StructUnionS,
    w_ffi: PyObjectRef,
) -> Result<Option<PyObjectRef>, PyError> {
    let name = c_string(s.name);
    let roots = pyre_object::gc_roots::push_roots();
    let ffi_slot = roots.base();
    let _ = roots.pin_root(w_ffi);
    let ffi = ffi_arg(roots.get(ffi_slot))?;
    if ffi.included_ffis_libs.is_null() {
        return Ok(None);
    }
    let list_slot = ffi_slot + 1;
    let _ = roots.pin_root(ffi.included_ffis_libs);
    let included = pyre_interpreter::baseobjspace::fixedview(roots.get(list_slot), -1)?;
    let included_slot = pyre_object::gc_roots::shadow_stack_len();
    for &item in &included {
        let _ = roots.pin_root(item);
    }
    for i in 0..included.len() {
        let pair = pyre_interpreter::baseobjspace::fixedview(roots.get(included_slot + i), 2)?;
        let pair_slot = pyre_object::gc_roots::shadow_stack_len();
        for &item in &pair {
            let _ = roots.pin_root(item);
        }
        let w_ffi1 = roots.get(pair_slot);
        let ffi1 = ffi_arg(w_ffi1)?;
        let ctx1 = unsafe { &(*ffi1.ctxobj).ctx };
        let sindex = parse_c_type::search_in_struct_unions(ctx1, &name);
        if sindex < 0 {
            continue;
        }
        let s1 = unsafe { *ctx1.struct_unions.offset(sindex) };
        if s1.flags
            & (parse_c_type::CffiTypeFlags::EXTERNAL.bits()
                | parse_c_type::CffiTypeFlags::UNION.bits())
            == s.flags & parse_c_type::CffiTypeFlags::UNION.bits()
        {
            return realize_c_struct_or_union(w_ffi1, sindex).map(Some);
        }
        if unsafe { pyre_object::listobject::w_list_len(ffi1.included_ffis_libs) } > 0
            && let Some(result) = fetch_external_struct_or_union(s, w_ffi1)?
        {
            return Ok(Some(result));
        }
    }
    Ok(None)
}
