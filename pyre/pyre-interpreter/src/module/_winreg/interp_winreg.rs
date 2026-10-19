use majit_rlib::rbigint::RBigInt as BigInt;
use pyre_object::*;
use rustpython_host_env::winreg::{self as host_reg, HKEY};
use rustpython_wtf8::{Wtf8, Wtf8Buf};
use widestring::{WideCStr, WideCString};

/// A raw handle/pointer value → a Python int (`PyLong_FromVoidPtr`).  The
/// predefined roots sign-extend past `i64::MAX` on 64-bit, so route those
/// through a long.
fn int_from_ptr(raw: HKEY) -> PyObjectRef {
    int_from_u64(raw as usize as u64)
}

/// `PyLong_FromUnsignedLongLong` — a 64-bit value one past `i64::MAX`
/// reads back as the large positive it is rather than as a negative.
fn int_from_u64(value: u64) -> PyObjectRef {
    if value <= i64::MAX as u64 {
        w_int_new(value as i64)
    } else {
        pyre_object::longobject::w_long_new(BigInt::from(value))
    }
}

// ── W_HKEY ──

// [3.14-spec] Two observables of the handle type follow `PC/winreg.c` at
// v3.14.6, where `interp_winreg.py W_HKEY.typedef` differs: the type is named
// `winreg.PyHKEY` (`pyhkey_type_spec.name`) rather than `winreg.HKEYType`,
// and it cannot be instantiated (`Py_TPFLAGS_DISALLOW_INSTANTIATION`) where
// `new_HKEY` builds one from an int.  Windows only; evidence is the C source
// at the tag.  The text follows `W_HKEY.descr_repr`, answering both `repr`
// and `str`: the tag's `Py_tp_str` without a `Py_tp_repr` has no carrier
// here, because `str()` does not dispatch a typed class's own `__str__`.
#[crate::pyre_class("winreg.PyHKEY", cpython_heaptype)]
pub struct W_HKEY {
    /// `W_HKEY.hkey` — the raw handle, null once closed or detached.
    hkey: HKEY,
}

/// `W_HKEY.Close.__doc__`.
const CLOSE_DOC: &str =
    "Closes the underlying Windows handle.\n\nIf the handle is already closed, no error is raised.";

/// `W_HKEY.Detach.__doc__`.
const DETACH_DOC: &str = "Detaches the Windows handle from the handle object.\n\nThe result is the value of the handle before it is detached.  If the\nhandle is already detached, this will return zero.\n\nAfter calling this function, the handle is effectively invalidated,\nbut the handle is not closed.  You would call this function when you\nneed the underlying win32 handle to exist beyond the lifetime of the\nhandle object.";

#[crate::pyre_methods]
impl W_HKEY {
    fn __bool__(&self) -> bool {
        !self.hkey.is_null()
    }

    /// `W_HKEY.descr_handle_get` — the handle as the integer it points at.
    /// `PyHKEY_memberlist` reads the same field through a `Py_T_INT` member,
    /// which on a 64-bit host truncates it to the low word; the property
    /// answers the whole pointer, so it agrees with `int(key)` for every
    /// handle rather than only for one below 2**31.
    #[getter]
    fn handle(&self) -> PyObjectRef {
        int_from_ptr(self.hkey)
    }

    fn __repr__(&self) -> String {
        format!("<PyHKEY:0x{:x}>", self.hkey as usize)
    }

    fn __int__(&self) -> PyObjectRef {
        int_from_ptr(self.hkey)
    }

    fn __enter__(&self) -> PyObjectRef {
        self as *const Self as PyObjectRef
    }

    fn __exit__(
        &mut self,
        _exc_type: PyObjectRef,
        _exc_value: PyObjectRef,
        _traceback: PyObjectRef,
    ) -> Result<(), pyre_interpreter::PyError> {
        close_key_obj(self as *mut Self as PyObjectRef)
    }

    fn Close(&mut self) -> Result<(), pyre_interpreter::PyError> {
        close_key_obj(self as *mut Self as PyObjectRef)
    }

    /// [3.14-spec] `W_HKEY.Detach` clears the handle before
    /// `space.audit("winreg.PyHKEY.Detach")`; `winreg_HKEYType_Detach_impl`
    /// at v3.14.6 audits first, then reads the handle it returns and clears
    /// it, so a refused call leaves the key holding (and later closing) its
    /// handle, and a hook that closed the key makes `Detach` return 0.
    /// Windows only; evidence is the C source at the tag.
    fn Detach(&mut self) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        audit_key("winreg.PyHKEY.Detach", &[AuditArg::Key(self.hkey)])?;
        // A hook that closed the key cleared the field through its object
        // pointer, which `&mut self` does not let the compiler see; read it
        // afresh.
        let key = unsafe { core::ptr::read_volatile(&raw const self.hkey) };
        self.hkey = core::ptr::null_mut();
        Ok(int_from_ptr(key))
    }
}

impl W_HKEY {
    /// `W_HKEY.__init__` — `register_finalizer`, so a key nobody closes is
    /// closed once it is unreachable.  The finalizer queue wants the
    /// object's definitive address, hence the stable allocation.
    fn new(hkey: HKEY) -> PyObjectRef {
        let obj = Self::allocate_stable(W_HKEY {
            ob: PyObject::default(),
            hkey,
        });
        pyre_interpreter::executioncontext::register_finalizer(obj);
        obj
    }

    /// `W_HKEY._finalize_` — `Close`, ignoring errors.
    pub fn _finalize_(&mut self) {
        let _ = close_key_obj(self as *mut Self as PyObjectRef);
    }
}

/// The GC type `W_HKEY` registers, in `build_gc` order after the
/// interpreter's own classes.
pub(crate) fn gc_types(types: &mut Vec<pyre_interpreter::importing::ModuleGcType>) {
    use pyre_interpreter::importing::{ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    types.push(ModuleGcType {
        descriptor: <W_HKEY as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: ModuleGcLayout::PyreClass {
            memory_pressure_offset: None,
        },
        destructor: None,
    });
}

/// `hkey_w` — a key argument is a `W_HKEY`, or an integer handle (the
/// predefined `HKEY_*` roots are published as their pointer value; a
/// negative one is cast rather than refused).
///
/// [3.14-spec] `hkey_w` refuses `None` everywhere; `PyHKEY_Close` at v3.14.6
/// calls `PyHKEY_AsHKEY(st, ob_handle, &key, TRUE)`, so `CloseKey(None)` reads
/// it as the null handle and closes nothing.  `none_ok` is that one
/// boundary.  Windows only; evidence is the C source at the tag.
fn as_hkey(obj: PyObjectRef, none_ok: bool) -> Result<HKEY, pyre_interpreter::PyError> {
    if unsafe { is_none(obj) } {
        if none_ok {
            return Ok(core::ptr::null_mut());
        }
        return Err(pyre_interpreter::PyError::type_error(
            "None is not a valid HKEY in this context",
        ));
    }
    if let Some(w_hkey) = W_HKEY::from_obj(obj) {
        return Ok(w_hkey.hkey);
    }
    if unsafe { is_int_or_long(obj) } {
        // [3.14-spec] `hkey_w` picks the branch with
        // `space.lt(w_hkey, space.newint(0))`, which runs an int subclass's
        // `__lt__`; `PyHKEY_AsHKEY` at v3.14.6 converts with
        // `PyLong_AsVoidPtr`, which reads the value's sign and calls nothing.
        // Windows only; evidence is the C source at the tag.
        let negative = unsafe {
            if is_int(obj) {
                w_int_get_value(obj) < 0
            } else {
                w_long_get_value(obj).sign() == majit_rlib::rbigint::RBigIntSign::Minus
            }
        };
        let raw = if negative {
            pyre_interpreter::baseobjspace::int_w(obj)? as usize
        } else {
            pyre_interpreter::baseobjspace::uint_w(obj)? as usize
        };
        return Ok(raw as HKEY);
    }
    Err(pyre_interpreter::PyError::type_error(
        "The object is not a PyHKEY object",
    ))
}

/// `CloseKey` — `RegCloseKey` when the key holds a handle, raising its
/// failure, and a `W_HKEY` left holding the null handle once that succeeds.
fn close_key_obj(w_hkey: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let hkey = as_hkey(w_hkey, true)?;
    if !hkey.is_null() {
        let ret = host_reg::close_key(hkey);
        if ret != 0 {
            return Err(win_err(ret));
        }
    }
    if let Some(w_hkey) = W_HKEY::from_obj(w_hkey) {
        w_hkey.hkey = core::ptr::null_mut();
    }
    Ok(())
}

/// One argument of a `winreg.*` audit event.
enum AuditArg<'a> {
    Obj(PyObjectRef),
    /// `space.newint(rffi.cast(rffi.SIZE_T, hkey))`.
    Key(HKEY),
    Int(i64),
    /// A `"u"` argument: a new `str` built from the converted wide name.
    Wide(&'a WideCStr),
    /// A `"u#"` argument: a new `str` built from the converted units.
    Units(&'a [u16]),
}

/// A `"u"` argument whose `wchar_t *` may be NULL, which `Py_BuildValue`
/// turns into `None`.
fn audit_name(name: Option<&WideCStr>) -> AuditArg<'_> {
    name.map_or_else(|| AuditArg::Obj(w_none()), AuditArg::Wide)
}

/// `space.audit(event, args_w)`.  The objects are built only when a hook is
/// armed, each rooted before the next one is allocated.
///
/// Every caller converts its text arguments to wide strings before the
/// audit, the order the clinic converters (`PyUnicode_AsWideCharString`) run
/// in ahead of each `_impl`; the converted names are owned, so a hook that
/// collects cannot move the bytes out from under them.
fn audit_key(event: &str, args: &[AuditArg<'_>]) -> Result<(), pyre_interpreter::PyError> {
    if !pyre_interpreter::module::sys::vm::audit_hooks_armed() {
        return Ok(());
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    for arg in args {
        let w = match arg {
            AuditArg::Obj(w) => *w,
            _ => PY_NULL,
        };
        let _ = pyre_object::gc_roots::pin_root(w);
    }
    for (i, arg) in args.iter().enumerate() {
        let w = match arg {
            AuditArg::Obj(_) => continue,
            AuditArg::Key(key) => int_from_ptr(*key),
            AuditArg::Int(value) => w_int_new(*value),
            AuditArg::Wide(text) => w_str_from_wtf8_managed(Wtf8Buf::from_wide(text.as_slice())),
            AuditArg::Units(units) => w_str_from_wtf8_managed(Wtf8Buf::from_wide(units)),
        };
        pyre_object::gc_roots::shadow_stack_set(base + i, w);
    }
    let args_w: Vec<PyObjectRef> = (0..args.len())
        .map(|i| pyre_object::gc_roots::shadow_stack_get(base + i))
        .collect();
    pyre_interpreter::module::sys::vm::audit(event, &args_w)
}

/// A new key handle, then `space.audit("winreg.OpenKey/result",
/// [space.newint(w_ret.as_int())])`; a hook that refuses leaves the handle to
/// the finalizer.
fn new_key_audited(out: HKEY) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let ret_slot = pyre_object::gc_roots::pin_roots(&[W_HKEY::new(out)]);
    audit_key("winreg.OpenKey/result", &[AuditArg::Key(out)])?;
    Ok(pyre_object::gc_roots::shadow_stack_get(ret_slot))
}

/// A positional-only call of fixed arity — `_PyArg_CheckPositional`
/// reports the count it wanted against the count it got.
/// Bound scope: omitted slots are `PY_NULL`.
fn exact_args<'a>(
    args: &'a [PyObjectRef],
    name: &str,
    count: usize,
) -> Result<&'a [PyObjectRef], pyre_interpreter::PyError> {
    let given = args.iter().filter(|value| !value.is_null()).count();
    if given != count {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{name} expected {count} arguments, got {given}"
        )));
    }
    Ok(args)
}

/// `METH_O` — the single key the seven one-argument calls take, whose
/// count is checked by the call machinery rather than by a converter.
fn single_arg(args: &[PyObjectRef], name: &str) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let given = args.iter().filter(|value| !value.is_null()).count();
    if given != 1 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "winreg.{name}() takes exactly one argument ({given} given)"
        )));
    }
    Ok(args[0])
}

/// The four `key, sub_key, …` boundaries the clinic exposes by keyword.
/// The first two are required; the last two default differently per
/// boundary, so an unset slot comes back for the caller to fill.
///
/// The order the three refusals come in is `_PyArg_UnpackKeywords`':
/// too many positionals first, then a required slot still empty — which
/// is why `OpenKey(bogus=1)` names `key` as missing rather than `bogus`
/// as unexpected — and the unknown keyword last.
fn bind_key_args(
    args: &[PyObjectRef],
    name: &str,
    params: [&str; 4],
) -> Result<[Option<PyObjectRef>; 4], pyre_interpreter::PyError> {
    // Bound scope: four named slots (`PY_NULL` omitted). The first two
    // are required; the last two default per boundary.
    let given = args.iter().filter(|o| !o.is_null()).count();
    if given > params.len() {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{name}() takes at most {} arguments ({given} given)",
            params.len()
        )));
    }
    let mut bound = [None; 4];
    for (index, key) in params.iter().enumerate() {
        let value = args.get(index).copied().filter(|o| !o.is_null());
        if value.is_none() && index < 2 {
            return Err(pyre_interpreter::PyError::type_error(format!(
                "{name}() missing required argument '{key}' (pos {})",
                index + 1
            )));
        }
        bound[index] = value;
    }
    Ok(bound)
}

/// A rejected string argument. The clinic names the parameter where the
/// boundary takes keywords and its 1-based position where it does not, and
/// names neither at `ExpandEnvironmentStrings`, so `label` arrives already
/// spelled the way the message wants it — `'sub_key'`, `2`, or nothing.
fn text_arg_error(
    func: &str,
    label: &str,
    accept_none: bool,
    obj: PyObjectRef,
) -> pyre_interpreter::PyError {
    let none = if accept_none { " or None" } else { "" };
    let label = if label.is_empty() {
        String::new()
    } else {
        format!("{label} ")
    };
    // `_PyArg_BadArgument` names `None` by the value rather than by
    // `NoneType`, which is the only object it spells that way.
    let got = if unsafe { is_none(obj) } {
        "None".to_string()
    } else {
        pyre_interpreter::error::type_name_of(obj)
    };
    pyre_interpreter::PyError::type_error(format!(
        "{func}() argument {label}must be str{none}, not {got}"
    ))
}

/// `Py_UNICODE(accept={str, NoneType})`.
fn opt_text(
    obj: PyObjectRef,
    func: &str,
    label: &str,
) -> Result<Option<&'static Wtf8>, pyre_interpreter::PyError> {
    if unsafe { is_none(obj) } {
        return Ok(None);
    }
    if unsafe { is_str(obj) } {
        return Ok(Some(unsafe { w_str_get_wtf8(obj) }));
    }
    Err(text_arg_error(func, label, true, obj))
}

/// `Py_UNICODE` — the same argument where `None` is not one of the
/// answers.
fn req_text(
    obj: PyObjectRef,
    func: &str,
    label: &str,
) -> Result<&'static Wtf8, pyre_interpreter::PyError> {
    if unsafe { is_str(obj) } {
        return Ok(unsafe { w_str_get_wtf8(obj) });
    }
    Err(text_arg_error(func, label, false, obj))
}

/// `PyUnicode_AsWideCharString` — a text argument as the terminated
/// UTF-16 a Win32 `W` call reads.
///
/// The registry is UTF-16 native, so an unpaired surrogate is a code unit
/// like any other in a key or value name: the text is read as WTF-8, and
/// `encode_wide` hands every code point straight to its own units, which
/// is what carries such a surrogate across unchanged.  A NUL is the one
/// unit that cannot travel, because the terminator is what marks the end;
/// the size-less `PyUnicode_AsWideCharString` the clinic's `Py_UNICODE`
/// converter calls refuses it rather than truncating there.
fn wide(text: &Wtf8) -> Result<WideCString, pyre_interpreter::PyError> {
    WideCString::from_vec(text.encode_wide().collect::<Vec<u16>>())
        .map_err(|_| pyre_interpreter::PyError::value_error("embedded null character"))
}

/// The name a sub-key argument spells; absent and `None` both mean the key
/// itself, which the Win32 calls take as the empty name.
fn wide_or_empty(text: Option<&Wtf8>) -> Result<WideCString, pyre_interpreter::PyError> {
    wide(text.unwrap_or_default())
}

// The four wordings `longobject.c` gives a value outside an unsigned
// width. The `unsigned long` pair names `int` for the negative and `long`
// for the overflow; both are its own, and neither follows from the other.
const NEGATIVE_UNSIGNED_INT: &str = "can't convert negative value to unsigned int";
const TOO_BIG_UNSIGNED_LONG: &str = "Python int too large to convert to C unsigned long";
const NEGATIVE_UNSIGNED: &str = "can't convert negative int to unsigned";
const TOO_BIG_UNSIGNED: &str = "int too big to convert";

/// `PyLong_AsUnsignedLong` and `PyLong_AsUnsignedLongLong` differ only in
/// the width they fit into and in what they call a value outside it. The
/// argument is already known to be an `int`.
fn unsigned_w(
    value: PyObjectRef,
    max: u64,
    negative: &str,
    too_big: &str,
) -> Result<u64, pyre_interpreter::PyError> {
    let raw = if unsafe { is_bool(value) } {
        (unsafe { pyre_object::boolobject::w_bool_get_value(value) }) as i64 as u64
    } else if unsafe { is_int(value) } {
        let signed = unsafe { pyre_object::intobject::w_int_get_value(value) };
        if signed < 0 {
            return Err(pyre_interpreter::PyError::overflow_error(negative));
        }
        signed as u64
    } else {
        let big = unsafe { pyre_object::w_long_get_value(value) };
        if big.get_sign() < 0 {
            return Err(pyre_interpreter::PyError::overflow_error(negative));
        }
        if pyre_object::longobject::jit_bigint_to_u64_fits(big) == 0 {
            return Err(pyre_interpreter::PyError::overflow_error(too_big));
        }
        pyre_object::longobject::jit_bigint_to_u64_value(big)
    };
    if raw > max {
        return Err(pyre_interpreter::PyError::overflow_error(too_big));
    }
    Ok(raw)
}

/// `_PyLong_UnsignedLong_Converter` — a `DWORD` argument: an access mask,
/// a reserved word, or a value type.
fn dword_arg(obj: PyObjectRef) -> Result<u32, pyre_interpreter::PyError> {
    let value = pyre_interpreter::baseobjspace::space_index(obj)?;
    unsigned_w(
        value,
        u64::from(u32::MAX),
        NEGATIVE_UNSIGNED_INT,
        TOO_BIG_UNSIGNED_LONG,
    )
    .map(|raw| raw as u32)
}

/// `_PyLong_AsInt` — the `index` the two Enum calls count with.
fn int_arg(obj: PyObjectRef) -> Result<i32, pyre_interpreter::PyError> {
    let value = pyre_interpreter::baseobjspace::space_index(obj)?;
    // `space_index` answers an int, so the only reading left to fail is
    // the width.
    let raw = pyre_interpreter::baseobjspace::int_w(value).map_err(|_| {
        pyre_interpreter::PyError::overflow_error("Python int too large to convert to C int")
    })?;
    i32::try_from(raw).map_err(|_| {
        pyre_interpreter::PyError::overflow_error("Python int too large to convert to C int")
    })
}
/// The `OSError` a failed `Reg*` call raises.  These report `LSTATUS`
/// codes, which are Win32 error codes, so the code is kept in `.winerror`
/// and `str(e)` opens `[WinError 2]` rather than `[Errno 2]`
/// (`PyErr_SetFromWindowsErrWithFunction`) -- the errno and its subclass
/// are derived from it.
fn win_err(code: u32) -> pyre_interpreter::PyError {
    pyre_interpreter::PyError::os_error_win32_syscall2(
        code as i32,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
    )
}

fn check(code: u32) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if code != 0 {
        Err(win_err(code))
    } else {
        Ok(w_none())
    }
}

fn utf16_units(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

/// `Reg2Py` — registry bytes + type → a Python value.
fn reg2py(data: &[u8], typ: u32) -> PyObjectRef {
    match typ {
        host_reg::REG_DWORD => {
            let value = if data.len() >= 4 {
                u32::from_le_bytes([data[0], data[1], data[2], data[3]])
            } else {
                0
            };
            w_int_new(value as i64)
        }
        host_reg::REG_QWORD => {
            let value = if data.len() >= 8 {
                u64::from_le_bytes(data[..8].try_into().unwrap())
            } else {
                0
            };
            int_from_u64(value)
        }
        host_reg::REG_SZ | host_reg::REG_EXPAND_SZ => {
            let units = utf16_units(data);
            let end = units.iter().position(|&c| c == 0).unwrap_or(units.len());
            w_str_from_wtf8_managed(Wtf8Buf::from_wide(&units[..end]))
        }
        host_reg::REG_MULTI_SZ => {
            // `countStrings`/`fixupMultiSZ` — one trailing terminator ends
            // the block and every run before it is a string, the empty
            // ones included; a block that is nothing but terminators reads
            // back as that many empty strings.
            let units = utf16_units(data);
            let end = match units.last() {
                None => 0,
                Some(&0) => units.len() - 1,
                Some(_) => units.len(),
            };
            let mut items = pyre_object::gc_roots::RootedItems::new();
            let mut start = 0;
            while start < end {
                let mut stop = start;
                while stop < end && units[stop] != 0 {
                    stop += 1;
                }
                items.push(w_str_from_wtf8_managed(Wtf8Buf::from_wide(
                    &units[start..stop],
                )));
                start = stop + 1;
            }
            w_list_new(items.take())
        }
        _ => {
            if data.is_empty() {
                w_none()
            } else {
                pyre_object::bytesobject::w_bytes_from_bytes(data)
            }
        }
    }
}

fn encode_utf16z(text: &Wtf8) -> Vec<u8> {
    let mut units: Vec<u16> = text.encode_wide().collect();
    units.push(0);
    units.iter().flat_map(|u| u.to_le_bytes()).collect()
}

/// What `Py2Reg` reports when it answers `FALSE`: either an exception of
/// its own — a value outside the width it is being written at, or one no
/// buffer can be taken from — or no exception at all, which the caller
/// turns into a `ValueError` naming neither.
enum Py2RegError {
    Raised(pyre_interpreter::PyError),
    Unconvertible,
}

impl From<pyre_interpreter::PyError> for Py2RegError {
    fn from(error: pyre_interpreter::PyError) -> Self {
        Py2RegError::Raised(error)
    }
}

/// `Py2Reg` — a Python value + type → registry bytes.
fn py2reg(value: PyObjectRef, typ: u32) -> Result<Vec<u8>, Py2RegError> {
    match typ {
        host_reg::REG_DWORD => {
            if unsafe { is_none(value) } {
                return Ok(0u32.to_le_bytes().to_vec());
            }
            if unsafe { !is_int_or_long(value) } {
                return Err(Py2RegError::Unconvertible);
            }
            let d = unsigned_w(
                value,
                u64::from(u32::MAX),
                NEGATIVE_UNSIGNED_INT,
                TOO_BIG_UNSIGNED_LONG,
            )? as u32;
            Ok(d.to_le_bytes().to_vec())
        }
        host_reg::REG_QWORD => {
            if unsafe { is_none(value) } {
                return Ok(0u64.to_le_bytes().to_vec());
            }
            if unsafe { !is_int_or_long(value) } {
                return Err(Py2RegError::Unconvertible);
            }
            let d = unsigned_w(value, u64::MAX, NEGATIVE_UNSIGNED, TOO_BIG_UNSIGNED)?;
            Ok(d.to_le_bytes().to_vec())
        }
        host_reg::REG_SZ | host_reg::REG_EXPAND_SZ => {
            // [3.14-spec] `convert_to_regdata` stores `2 * len + 1` bytes for a
            // str and a single byte for `None`, so the value `QueryValueEx`
            // reads back is an odd length with half a terminator; `Py2Reg` at
            // v3.14.6 stores `(len + 1) * sizeof(wchar_t)` and two zero bytes
            // for `None`, a whole UTF-16 terminator either way.  Windows only;
            // evidence is the C source at the tag.
            if unsafe { is_none(value) } {
                return Ok(vec![0, 0]);
            }
            if unsafe { !is_str(value) } {
                return Err(Py2RegError::Unconvertible);
            }
            Ok(encode_utf16z(unsafe { w_str_get_wtf8(value) }))
        }
        host_reg::REG_MULTI_SZ => {
            let len = if unsafe { is_none(value) } {
                0
            } else if unsafe { is_list(value) } {
                unsafe { w_list_len(value) }
            } else {
                return Err(Py2RegError::Unconvertible);
            };
            let mut out = Vec::new();
            for i in 0..len {
                let item = unsafe { w_list_getitem(value, i as i64) };
                let Some(item) = item else { continue };
                if unsafe { !is_str(item) } {
                    return Err(Py2RegError::Unconvertible);
                }
                let units: Vec<u16> = unsafe { w_str_get_wtf8(item) }.encode_wide().collect();
                out.extend(units.iter().flat_map(|u| u.to_le_bytes()));
                out.extend_from_slice(&[0, 0]);
            }
            out.extend_from_slice(&[0, 0]);
            Ok(out)
        }
        _ => {
            // REG_BINARY / REG_NONE and the rest — a bytes-like value, or
            // `None` for an empty write.
            if unsafe { is_none(value) } {
                return Ok(Vec::new());
            }
            // `w_value.buffer_w(space, space.BUF_SIMPLE)`; any exporter of a
            // simple buffer qualifies, `BufferInterfaceNotFound` is the
            // TypeError.
            match pyre_interpreter::baseobjspace::simple_buffer_bytes(value)
                .map_err(Py2RegError::Raised)?
            {
                Some(view) => Ok(view.into_bytes()),
                None => Err(Py2RegError::Raised(pyre_interpreter::PyError::type_error(
                    format!(
                        "Objects of type '{}' can not be used as binary registry values",
                        pyre_interpreter::error::type_name_of(value)
                    ),
                ))),
            }
        }
    }
}

// ── module functions ──
/// `OpenKey` and `OpenKeyEx` are one implementation under two names, and
/// each names itself in what it refuses.
///
/// [3.14-spec] `interp_winreg.py` reads the name argument of `OpenKey`,
/// `OpenKeyEx`, `CreateKey`, `CreateKeyEx`, `SetValue`, `SetValueEx` and
/// `DeleteValue` with `space.utf8_w`, so `None` is refused there;
/// `PC/winreg.c` at v3.14.6 declares each of them
/// `Py_UNICODE(accept={str, NoneType})`, so `None` names the key itself or
/// its default value.  `opt_text` is that converter (`QueryValue`,
/// `QueryValueEx` and `ConnectRegistry` take `None` on both sides).
/// Windows only; evidence is the C source at the tag.
fn open_key_named(
    args: &[PyObjectRef],
    name: &str,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let bound = bind_key_args(args, name, ["key", "sub_key", "reserved", "access"])?;
    let key = as_hkey(bound[0].expect("key is required"), false)?;
    let sub_key = opt_text(bound[1].expect("sub_key is required"), name, "'sub_key'")?;
    let named = sub_key.is_some();
    let wide_sub = wide_or_empty(sub_key)?;
    let reserved = bound[2].map(int_arg).transpose()?.unwrap_or(0) as u32;
    let access = bound[3]
        .map(dword_arg)
        .transpose()?
        .unwrap_or(host_reg::KEY_READ);
    audit_key(
        "winreg.OpenKey",
        &[
            AuditArg::Key(key),
            audit_name(named.then_some(&*wide_sub)),
            AuditArg::Int(access as i64),
        ],
    )?;
    let mut out: HKEY = core::ptr::null_mut();
    let rc = unsafe { host_reg::open_key_ex(key, &wide_sub, reserved, access, &mut out) };
    if rc != 0 {
        return Err(win_err(rc));
    }
    new_key_audited(out)
}

fn open_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    open_key_named(args, "OpenKey")
}

fn open_key_ex(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    open_key_named(args, "OpenKeyEx")
}

fn create_key_ex(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // `CreateKey`'s form with the mask to open it under, which is the only
    // way to reach a 32-bit view (`KEY_WOW64_32KEY`) of a key.
    let bound = bind_key_args(
        args,
        "CreateKeyEx",
        ["key", "sub_key", "reserved", "access"],
    )?;
    let key = as_hkey(bound[0].expect("key is required"), false)?;
    let sub_key = opt_text(
        bound[1].expect("sub_key is required"),
        "CreateKeyEx",
        "'sub_key'",
    )?;
    let named = sub_key.is_some();
    let wide_sub = wide_or_empty(sub_key)?;
    let reserved = bound[2].map(int_arg).transpose()?.unwrap_or(0) as u32;
    let access = bound[3]
        .map(dword_arg)
        .transpose()?
        .unwrap_or(host_reg::KEY_WRITE);
    audit_key(
        "winreg.CreateKey",
        &[
            AuditArg::Key(key),
            audit_name(named.then_some(&*wide_sub)),
            AuditArg::Int(access as i64),
        ],
    )?;
    let mut out: HKEY = core::ptr::null_mut();
    let rc = unsafe {
        host_reg::create_key_ex(
            key,
            &wide_sub,
            reserved,
            core::ptr::null_mut(),
            host_reg::REG_OPTION_NON_VOLATILE,
            access,
            core::ptr::null(),
            &mut out,
            core::ptr::null_mut(),
        )
    };
    if rc != 0 {
        return Err(win_err(rc));
    }
    new_key_audited(out)
}

fn delete_key_ex(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // The access mask comes before `reserved` here, the other way round
    // from CreateKeyEx.
    let bound = bind_key_args(
        args,
        "DeleteKeyEx",
        ["key", "sub_key", "access", "reserved"],
    )?;
    let key = as_hkey(bound[0].expect("key is required"), false)?;
    let sub_key = wide(req_text(
        bound[1].expect("sub_key is required"),
        "DeleteKeyEx",
        "'sub_key'",
    )?)?;
    let access = bound[2]
        .map(dword_arg)
        .transpose()?
        .unwrap_or(host_reg::KEY_WOW64_64KEY);
    let reserved = bound[3].map(int_arg).transpose()?.unwrap_or(0) as u32;
    // [3.14-spec] `DeleteKeyEx` in `interp_winreg.py` audits with access 0;
    // `winreg_DeleteKeyEx_impl` at v3.14.6 passes the mask it was given
    // (`"nun"`, key, sub_key, access).  Windows only; evidence is the C
    // source at the tag.
    audit_key(
        "winreg.DeleteKey",
        &[
            AuditArg::Key(key),
            AuditArg::Wide(&sub_key),
            AuditArg::Int(access as i64),
        ],
    )?;
    check(unsafe { host_reg::delete_key_ex(key, &sub_key, access, reserved) })
}

fn load_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // Restores a hive a `SaveKey` wrote.  Both it and SaveKey need
    // SE_RESTORE_NAME/SE_BACKUP_NAME, so an ordinary account gets
    // ERROR_PRIVILEGE_NOT_HELD rather than a result.
    let pos = exact_args(args, "LoadKey", 3)?;
    let key = as_hkey(pos[0], false)?;
    let sub_key = req_text(pos[1], "LoadKey", "2")?;
    let file_name = req_text(pos[2], "LoadKey", "3")?;
    let (sub_key, file_name) = (wide(sub_key)?, wide(file_name)?);
    audit_key(
        "winreg.LoadKey",
        &[
            AuditArg::Key(key),
            AuditArg::Wide(&sub_key),
            AuditArg::Wide(&file_name),
        ],
    )?;
    check(unsafe { host_reg::load_key(key, &sub_key, &file_name) })
}

fn save_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let pos = exact_args(args, "SaveKey", 2)?;
    let key = as_hkey(pos[0], false)?;
    let file_name = wide(req_text(pos[1], "SaveKey", "2")?)?;
    audit_key(
        "winreg.SaveKey",
        &[AuditArg::Key(key), AuditArg::Wide(&file_name)],
    )?;
    check(unsafe { host_reg::save_key(key, &file_name) })
}

fn close_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    close_key_obj(single_arg(args, "CloseKey")?)?;
    Ok(w_none())
}

/// `winreg_QueryValue_impl` — the key's unnamed value, read as text.
///
/// `host_reg::query_default_value` answers a `String`, so a `REG_SZ` whose
/// units hold an unpaired surrogate has no spelling it can give back; the
/// units are read here instead and carried across as themselves
/// (`PyUnicode_FromWideChar`).
///
/// [3.14-spec] `interp_winreg.QueryValue` reads through `RegQueryValueW`;
/// `winreg_QueryValue_impl` at v3.14.6 opens the sub-key and reads its
/// unnamed value with `RegQueryValueExW`, as this does.  Measured with
/// python3.14 and pypy3: a key without an unnamed value reads `''` (pypy3:
/// `'\x00'`), a `REG_DWORD` or `REG_EXPAND_SZ` one raises `OSError` with
/// errno 22 (pypy3: 13), and a 3000-unit value reads back whole (pypy3:
/// `UnicodeDecodeError`).
fn query_default_value(
    key: HKEY,
    sub_key: &WideCStr,
) -> Result<Wtf8Buf, pyre_interpreter::PyError> {
    use windows_sys::Win32::Foundation::{
        ERROR_FILE_NOT_FOUND, ERROR_INVALID_DATA, ERROR_INVALID_HANDLE, ERROR_MORE_DATA,
    };
    // The performance data root answers no default value, and the refusal
    // is spelled here rather than left to the registry.
    if key == host_reg::HKEY_PERFORMANCE_DATA {
        return Err(win_err(ERROR_INVALID_HANDLE));
    }
    // An absent or empty sub-key names the key itself; anything else is
    // opened for the read and closed again after it.
    let child_key = if sub_key.is_empty() {
        None
    } else {
        let mut out: HKEY = core::ptr::null_mut();
        let rc =
            unsafe { host_reg::open_key_ex(key, sub_key, 0, host_reg::KEY_QUERY_VALUE, &mut out) };
        if rc != 0 {
            return Err(win_err(rc));
        }
        Some(out)
    };
    let target = child_key.unwrap_or(key);
    // `WCHAR buf[256]`, whose byte size is what the first call is given.
    let mut buffer = vec![0u8; 256 * size_of::<u16>()];
    let mut typ = 0u32;
    let result = loop {
        let mut size = buffer.len() as u32;
        let rc = unsafe {
            host_reg::query_value_ex(target, None, &mut typ, buffer.as_mut_ptr(), &mut size)
        };
        if rc == ERROR_MORE_DATA {
            // The failed call reports the size it wanted, so the retry
            // asks for exactly that.
            buffer.resize(size as usize, 0);
            continue;
        }
        // A key with no unnamed value reads back as the empty string
        // rather than as the error the call reports for it.
        if rc == ERROR_FILE_NOT_FOUND {
            break Ok(Wtf8Buf::new());
        }
        if rc != 0 {
            break Err(win_err(rc));
        }
        if typ != host_reg::REG_SZ {
            break Err(win_err(ERROR_INVALID_DATA));
        }
        let units = utf16_units(&buffer[..size as usize]);
        let end = units
            .iter()
            .position(|&unit| unit == 0)
            .unwrap_or(units.len());
        break Ok(Wtf8Buf::from_wide(&units[..end]));
    };
    if let Some(child) = child_key {
        host_reg::close_key(child);
    }
    result
}

fn query_value(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let pos = exact_args(args, "QueryValue", 2)?;
    let key = as_hkey(pos[0], false)?;
    let sub_key = opt_text(pos[1], "QueryValue", "2")?;
    let named = sub_key.is_some();
    let wide_sub = wide_or_empty(sub_key)?;
    // [3.14-spec] `QueryValue` in `interp_winreg.py` audits the absent value
    // name as `''`; `winreg_QueryValue_impl` at v3.14.6 builds `"nuu"` from a
    // NULL name, which `Py_BuildValue` turns into `None` (and `QueryValueEx`
    // likewise for its absent sub-key).  Windows only; evidence is the C
    // source at the tag.
    audit_key(
        "winreg.QueryValue",
        &[
            AuditArg::Key(key),
            audit_name(named.then_some(&*wide_sub)),
            AuditArg::Obj(w_none()),
        ],
    )?;
    Ok(w_str_from_wtf8_managed(query_default_value(
        key, &wide_sub,
    )?))
}

fn query_value_ex(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let pos = exact_args(args, "QueryValueEx", 2)?;
    let key = as_hkey(pos[0], false)?;
    let value_name = opt_text(pos[1], "QueryValueEx", "2")?;
    let named = value_name.is_some();
    let name = wide_or_empty(value_name)?;
    audit_key(
        "winreg.QueryValue",
        &[
            AuditArg::Key(key),
            AuditArg::Obj(w_none()),
            audit_name(named.then_some(&*name)),
        ],
    )?;
    use windows_sys::Win32::Foundation::ERROR_MORE_DATA;
    let mut data_size = 0u32;
    let ret = unsafe {
        host_reg::query_value_ex(
            key,
            Some(&name),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            &mut data_size,
        )
    };
    let mut buf_size = data_size;
    if ret == ERROR_MORE_DATA {
        // `bufSize = 256`, otherwise the size can be 0.
        buf_size = 256;
    } else if ret != 0 {
        return Err(win_err(ret));
    }
    loop {
        let mut data_buf = vec![0u8; buf_size as usize];
        let mut ret_type = 0u32;
        // Every read offers the whole buffer: the size the previous call
        // reported back can be larger than what is allocated.
        data_size = buf_size;
        let ret = unsafe {
            host_reg::query_value_ex(
                key,
                Some(&name),
                &mut ret_type,
                data_buf.as_mut_ptr(),
                &mut data_size,
            )
        };
        if ret == ERROR_MORE_DATA {
            // Resize and retry.
            buf_size = buf_size
                .checked_mul(2)
                .ok_or_else(|| win_err(ERROR_MORE_DATA))?;
            continue;
        }
        if ret != 0 {
            return Err(win_err(ret));
        }
        let length = (data_size as usize).min(data_buf.len());
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(reg2py(&data_buf[..length], ret_type));
        fields.push(w_int_new(ret_type as i64));
        return Ok(w_tuple_new(fields.take()));
    }
}

fn enum_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let pos = exact_args(args, "EnumKey", 2)?;
    let key = as_hkey(pos[0], false)?;
    let index = int_arg(pos[1])?;
    audit_key(
        "winreg.EnumKey",
        &[AuditArg::Key(key), AuditArg::Int(index as i64)],
    )?;
    let index = index as u32;
    // Registry key names are capped at 255 chars (winnt.h MAX_KEY_LENGTH).
    let mut buffer = [0u16; 257];
    let mut len = buffer.len() as u32;
    let rc = unsafe { host_reg::enum_key_ex(key, index, buffer.as_mut_ptr(), &mut len) };
    if rc != 0 {
        return Err(win_err(rc));
    }
    Ok(w_str_from_wtf8_managed(Wtf8Buf::from_wide(
        &buffer[..len as usize],
    )))
}

fn enum_value(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let pos = exact_args(args, "EnumValue", 2)?;
    let key = as_hkey(pos[0], false)?;
    let index = int_arg(pos[1])?;
    audit_key(
        "winreg.EnumValue",
        &[AuditArg::Key(key), AuditArg::Int(index as i64)],
    )?;
    let index = index as u32;
    let mut max_name = 0u32;
    let mut max_data = 0u32;
    let rc = unsafe {
        host_reg::query_info_key(
            key,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            &mut max_name,
            &mut max_data,
        )
    };
    if rc != 0 {
        return Err(win_err(rc));
    }
    // The two lengths the key reports leave no room for the terminators.
    // For a dynamic key (the performance hive) the reported data size is
    // useless, so `ERROR_MORE_DATA` quadruples the data buffer and retries.
    let mut buf_name_size = max_name + 1;
    let mut buf_data_size = max_data + 1;
    let mut name = vec![0u16; buf_name_size as usize];
    let mut data = vec![0u8; buf_data_size as usize];
    loop {
        let mut name_len = buf_name_size;
        let mut data_len = buf_data_size;
        let mut typ = 0u32;
        let rc = unsafe {
            host_reg::enum_value(
                key,
                index,
                name.as_mut_ptr(),
                &mut name_len,
                &mut typ,
                data.as_mut_ptr(),
                &mut data_len,
            )
        };
        if rc == windows_sys::Win32::Foundation::ERROR_MORE_DATA {
            buf_data_size = buf_data_size
                .checked_mul(4)
                .ok_or_else(|| win_err(windows_sys::Win32::Foundation::ERROR_MORE_DATA))?;
            data.resize(buf_data_size as usize, 0);
            continue;
        }
        if rc != 0 {
            return Err(win_err(rc));
        }
        // `Py_BuildValue("uOi", retValueBuf, …)` reads the name up to
        // its terminator rather than to the length the call reports back,
        // which a key that enumerates without a reliable count -- the
        // performance hive is one -- spells one character short.
        let end = name
            .iter()
            .position(|&unit| unit == 0)
            .unwrap_or(name.len());
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(w_str_from_wtf8_managed(Wtf8Buf::from_wide(&name[..end])));
        fields.push(reg2py(&data[..data_len as usize], typ));
        fields.push(w_int_new(typ as i64));
        return Ok(w_tuple_new(fields.take()));
    }
}

fn query_info_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let key = as_hkey(single_arg(args, "QueryInfoKey")?, false)?;
    audit_key("winreg.QueryInfoKey", &[AuditArg::Key(key)])?;
    match host_reg::query_info_key_full(key) {
        Ok(info) => {
            let mut fields = pyre_object::gc_roots::RootedItems::new();
            fields.push(w_int_new(info.sub_keys as i64));
            fields.push(w_int_new(info.values as i64));
            fields.push(w_int_new(info.last_write_time as i64));
            Ok(w_tuple_new(fields.take()))
        }
        Err(code) => Err(win_err(code)),
    }
}

/// [3.14-spec] `FlushKey` guards the call with `if hkey`, so `FlushKey(0)`
/// is a no-op; `winreg_FlushKey_impl` at v3.14.6 calls `RegFlushKey`
/// unconditionally, and python3.14 raises `OSError: [WinError 6]` for
/// `FlushKey(0)`. Windows only; evidence is the C source at the tag and that
/// run.
fn flush_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let key = as_hkey(single_arg(args, "FlushKey")?, false)?;
    check(host_reg::flush_key(key))
}

fn create_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let pos = exact_args(args, "CreateKey", 2)?;
    let key = as_hkey(pos[0], false)?;
    let sub_key = opt_text(pos[1], "CreateKey", "2")?;
    let named = sub_key.is_some();
    let wide_sub = wide_or_empty(sub_key)?;
    audit_key(
        "winreg.CreateKey",
        &[
            AuditArg::Key(key),
            audit_name(named.then_some(&*wide_sub)),
            AuditArg::Int(host_reg::KEY_WRITE as i64),
        ],
    )?;
    let mut out: HKEY = core::ptr::null_mut();
    let rc = unsafe { host_reg::create_key(key, &wide_sub, &mut out) };
    if rc != 0 {
        return Err(win_err(rc));
    }
    new_key_audited(out)
}

fn set_value(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // SetValue only writes REG_SZ, and the type is read before it is
    // judged, so a non-integer is an integer's complaint rather than the
    // wrong-type one.  Every argument is spelled before any of them is
    // judged, though, so a `value` that is no string is that argument's
    // complaint even when the type is wrong too.
    let pos = exact_args(args, "SetValue", 4)?;
    let key = as_hkey(pos[0], false)?;
    let sub_key_text = opt_text(pos[1], "SetValue", "2")?;
    let named = sub_key_text.is_some();
    let sub_key = wide_or_empty(sub_key_text)?;
    // `dword_arg` can run `__index__`, which can collect; the value is read
    // back from its root afterwards.
    let _roots = pyre_object::gc_roots::push_roots();
    let value_slot = pyre_object::gc_roots::pin_roots(&[pos[3]]);
    let typ = dword_arg(pos[2])?;
    let value = req_text(
        pyre_object::gc_roots::shadow_stack_get(value_slot),
        "SetValue",
        "4",
    )?;
    if typ != host_reg::REG_SZ {
        return Err(pyre_interpreter::PyError::type_error(
            "type must be winreg.REG_SZ",
        ));
    }
    // `(length + 1) * sizeof(WCHAR)` — the stored `REG_SZ` carries its
    // terminator, and the byte count has to fit a `DWORD`.
    let units: Vec<u16> = value.encode_wide().chain(std::iter::once(0)).collect();
    if u32::try_from(units.len() * size_of::<u16>()).is_err() {
        return Err(pyre_interpreter::PyError::overflow_error(
            "value is too long",
        ));
    }
    audit_key(
        "winreg.SetValue",
        &[
            AuditArg::Key(key),
            audit_name(named.then_some(&*sub_key)),
            AuditArg::Int(typ as i64),
            AuditArg::Units(&units[..units.len() - 1]),
        ],
    )?;
    if key == host_reg::HKEY_PERFORMANCE_DATA {
        use windows_sys::Win32::Foundation::ERROR_INVALID_HANDLE;
        return Err(win_err(ERROR_INVALID_HANDLE));
    }
    check(host_reg::set_default_value(
        key,
        &sub_key,
        typ,
        &widestring::WideString::from_vec(units),
    ))
}

fn set_value_ex(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let pos = exact_args(args, "SetValueEx", 5)?;
    let key = as_hkey(pos[0], false)?;
    let wide_name = opt_text(pos[1], "SetValueEx", "2")?.map(wide).transpose()?;
    // `dword_arg` and `py2reg` can run Python, which can collect; the value
    // is read back from its root after each.
    let _roots = pyre_object::gc_roots::push_roots();
    let value_slot = pyre_object::gc_roots::pin_roots(&[pos[4]]);
    // `reserved` is documented as "can be anything" and zero is what
    // reaches the API, so it is read but not converted.
    let typ = dword_arg(pos[3])?;
    let data =
        py2reg(pyre_object::gc_roots::shadow_stack_get(value_slot), typ).map_err(|error| {
            match error {
                Py2RegError::Raised(error) => error,
                Py2RegError::Unconvertible => pyre_interpreter::PyError::value_error(
                    "Could not convert the data to the specified type.",
                ),
            }
        })?;
    audit_key(
        "winreg.SetValue",
        &[
            AuditArg::Key(key),
            audit_name(wide_name.as_deref()),
            AuditArg::Int(typ as i64),
            AuditArg::Obj(pyre_object::gc_roots::shadow_stack_get(value_slot)),
        ],
    )?;
    let rc = unsafe {
        host_reg::set_value_ex(
            key,
            wide_name.as_deref(),
            typ,
            data.as_ptr(),
            data.len() as u32,
        )
    };
    if rc != 0 {
        return Err(win_err(rc));
    }
    Ok(w_none())
}

fn delete_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let pos = exact_args(args, "DeleteKey", 2)?;
    let key = as_hkey(pos[0], false)?;
    let sub_key = wide(req_text(pos[1], "DeleteKey", "2")?)?;
    audit_key(
        "winreg.DeleteKey",
        &[
            AuditArg::Key(key),
            AuditArg::Wide(&sub_key),
            AuditArg::Int(0),
        ],
    )?;
    check(unsafe { host_reg::delete_key(key, &sub_key) })
}

fn delete_value(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let pos = exact_args(args, "DeleteValue", 2)?;
    let key = as_hkey(pos[0], false)?;
    let wide_name = opt_text(pos[1], "DeleteValue", "2")?
        .map(wide)
        .transpose()?;
    audit_key(
        "winreg.DeleteValue",
        &[AuditArg::Key(key), audit_name(wide_name.as_deref())],
    )?;
    check(unsafe { host_reg::delete_value(key, wide_name.as_deref()) })
}

fn connect_registry(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let pos = exact_args(args, "ConnectRegistry", 2)?;
    let wide_computer = opt_text(pos[0], "ConnectRegistry", "1")?
        .map(wide)
        .transpose()?;
    let key = as_hkey(pos[1], false)?;
    audit_key(
        "winreg.ConnectRegistry",
        &[audit_name(wide_computer.as_deref()), AuditArg::Key(key)],
    )?;
    let mut out: HKEY = core::ptr::null_mut();
    let rc = unsafe { host_reg::connect_registry(wide_computer.as_deref(), key, &mut out) };
    if rc != 0 {
        return Err(win_err(rc));
    }
    Ok(W_HKEY::new(out))
}

/// `ExpandEnvironmentStringsW`, twice: the first call reports the size the
/// second one writes into.  `None` is the call having failed, which the
/// caller spells with the code the system left behind.
///
/// The wide reader keeps unpaired surrogates (`PyUnicode_FromWideChar`).
fn expand_environment_strings_wide(input: &WideCStr) -> Option<Wtf8Buf> {
    host_reg::expand_environment_strings_wide(input)
        .ok()
        .map(|units| Wtf8Buf::from_wide(&units))
}

fn expand_environment_strings(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // The only boundary whose refusal names neither a parameter nor a
    // position, because it has just the one argument.
    let w_source = single_arg(args, "ExpandEnvironmentStrings")?;
    let input = wide(req_text(w_source, "ExpandEnvironmentStrings", "")?)?;
    // [3.14-spec] `ExpandEnvironmentStrings` in `interp_winreg.py` raises the
    // event under the name `winreg.ConnectRegistry` with the argument object;
    // `winreg_ExpandEnvironmentStrings_impl` at v3.14.6 raises
    // `winreg.ExpandEnvironmentStrings` with `"u"` built from the converted
    // wide string, so a hook sees a plain `str` even for a `str` subclass.
    // Windows only; evidence is the C source at the tag.
    audit_key("winreg.ExpandEnvironmentStrings", &[AuditArg::Wide(&input)])?;
    match expand_environment_strings_wide(&input) {
        Some(value) => Ok(w_str_from_wtf8_managed(value)),
        None => Err(win_err(
            std::io::Error::last_os_error().raw_os_error().unwrap_or(0) as u32,
        )),
    }
}

fn disable_reflection_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let key = as_hkey(single_arg(args, "DisableReflectionKey")?, false)?;
    audit_key("winreg.DisableReflectionKey", &[AuditArg::Key(key)])?;
    check(host_reg::disable_reflection_key(key))
}

fn enable_reflection_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let key = as_hkey(single_arg(args, "EnableReflectionKey")?, false)?;
    audit_key("winreg.EnableReflectionKey", &[AuditArg::Key(key)])?;
    check(host_reg::enable_reflection_key(key))
}

fn query_reflection_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let key = as_hkey(single_arg(args, "QueryReflectionKey")?, false)?;
    audit_key("winreg.QueryReflectionKey", &[AuditArg::Key(key)])?;
    let mut disabled: i32 = 0;
    let rc = unsafe { host_reg::query_reflection_key(key, &mut disabled) };
    if rc != 0 {
        return Err(win_err(rc));
    }
    Ok(w_bool_from(disabled != 0))
}

pub fn install(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // The handle type is bound under one name only: the class calls itself
    // `PyHKEY`, and `winreg` publishes it as `HKEYType`.
    let w_hkey_type = type_object();
    unsafe { pyre_object::w_type_set_disallow_instantiation(w_hkey_type) };
    crate::__pyre_put_new!(ns_slot, "HKEYType", w_hkey_type);
    // `Py_tp_doc`, and the signature line clinic writes ahead of each
    // `PyHKEY_methods` docstring. The type has just been built by the line
    // above and nothing has read it yet, so the class dict takes them
    // directly.
    unsafe {
        let ty = type_object();
        let class_ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ =
            pyre_object::gc_roots::pin_root(pyre_object::w_type_get_dict_ptr(ty) as PyObjectRef);
        crate::__pyre_put_new!(
            class_ns_slot,
            "__doc__",
            w_str_new(
                "PyHKEY Object - A Python object, representing a win32 registry key.\n\nThis object wraps a Windows HKEY object, automatically closing it when\nthe object is destroyed.  To guarantee cleanup, you can call either\nthe Close() method on the PyHKEY, or the CloseKey() method.\n\nAll functions which accept a handle object also accept an integer --\nhowever, use of the handle object is encouraged.\n\nFunctions:\nClose() - Closes the underlying handle.\nDetach() - Returns the integer Win32 handle, detaching it from the object\n\nProperties:\nhandle - The integer Win32 handle.\n\nOperations:\n__bool__ - Handles with an open object return true, otherwise false.\n__int__ - Converting a handle to an integer returns the Win32 handle.\n__enter__, __exit__ - Context manager support for 'with' statement,\nautomatically closes handle.",
            )
        );
        for (method, signature, docstring) in [
            ("Close", "($self, /)", Some(CLOSE_DOC)),
            ("Detach", "($self, /)", Some(DETACH_DOC)),
            ("__enter__", "($self, /)", None),
            (
                "__exit__",
                "($self, exc_type, exc_value, traceback, /)",
                None,
            ),
        ] {
            let Some(function) = pyre_object::w_dict_getitem_str(
                pyre_object::gc_roots::shadow_stack_get(class_ns_slot),
                method,
            ) else {
                continue;
            };
            // `interp2app` reads the docstring off the method; nothing has
            // asked the fresh function for its `__doc__` yet, so the code
            // object still answers it.
            if let Some(docstring) = docstring {
                let code = pyre_interpreter::function::function_get_code(function);
                (*(code as *mut pyre_interpreter::gateway::BuiltinCode)).docstring =
                    Some(docstring);
            }
            let function_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(function);
            let signature_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_str_new(signature));
            pyre_interpreter::function::fset_func_text_signature(
                pyre_object::gc_roots::shadow_stack_get(function_slot),
                pyre_object::gc_roots::shadow_stack_get(signature_slot),
            );
        }
    }
    // `winreg.error` is `OSError` itself rather than a module-specific
    // class, so `except winreg.error` catches what the Reg* calls raise.
    if let Some(os_error) = pyre_interpreter::builtins::lookup_exc_class("OSError") {
        crate::__pyre_put_new!(ns_slot, "error", os_error);
    }
    // The predefined roots are exposed as their full (sign-extended)
    // pointer value, matching `PyLong_FromVoidPtr` — this overrides the
    // 32-bit fallbacks the always-present `int_constants` set.
    for (name, root) in [
        ("HKEY_CLASSES_ROOT", host_reg::HKEY_CLASSES_ROOT),
        ("HKEY_CURRENT_USER", host_reg::HKEY_CURRENT_USER),
        ("HKEY_LOCAL_MACHINE", host_reg::HKEY_LOCAL_MACHINE),
        ("HKEY_USERS", host_reg::HKEY_USERS),
        ("HKEY_PERFORMANCE_DATA", host_reg::HKEY_PERFORMANCE_DATA),
        ("HKEY_CURRENT_CONFIG", host_reg::HKEY_CURRENT_CONFIG),
        ("HKEY_DYN_DATA", host_reg::HKEY_DYN_DATA),
    ] {
        crate::__pyre_put_new!(ns_slot, name, int_from_ptr(root));
    }
    // Each call carries the clinic's own signature line and
    // docstring, which is what `help(winreg.X)` and
    // `inspect.signature` read back.
    for (name, func, signature, doc) in [
        (
            "OpenKey",
            open_key as fn(&[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError>,
            "($module, /, key, sub_key, reserved=0, access=winreg.KEY_READ)",
            "Opens the specified key.\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n  sub_key\n    A string that identifies the sub_key to open.\n  reserved\n    A reserved integer that must be zero.  Default is zero.\n  access\n    An integer that specifies an access mask that describes the desired\n    security access for the key.  Default is KEY_READ.\n\nThe result is a new handle to the specified key.\nIf the function fails, an OSError exception is raised.",
        ),
        (
            "OpenKeyEx",
            open_key_ex,
            "($module, /, key, sub_key, reserved=0, access=winreg.KEY_READ)",
            "Opens the specified key.\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n  sub_key\n    A string that identifies the sub_key to open.\n  reserved\n    A reserved integer that must be zero.  Default is zero.\n  access\n    An integer that specifies an access mask that describes the desired\n    security access for the key.  Default is KEY_READ.\n\nThe result is a new handle to the specified key.\nIf the function fails, an OSError exception is raised.",
        ),
        (
            "CreateKeyEx",
            create_key_ex,
            "($module, /, key, sub_key, reserved=0,\n            access=winreg.KEY_WRITE)",
            "Creates or opens the specified key.\n\n  key\n    An already open key, or one of the predefined HKEY_* constants.\n  sub_key\n    The name of the key this method opens or creates.\n  reserved\n    A reserved integer, and must be zero.  Default is zero.\n  access\n    An integer that specifies an access mask that describes the\n    desired security access for the key. Default is KEY_WRITE.\n\nIf key is one of the predefined keys, sub_key may be None. In that case,\nthe handle returned is the same key handle passed in to the function.\n\nIf the key already exists, this function opens the existing key\n\nThe return value is the handle of the opened key.\nIf the function fails, an OSError exception is raised.",
        ),
        (
            "DeleteKeyEx",
            delete_key_ex,
            "($module, /, key, sub_key, access=winreg.KEY_WOW64_64KEY,\n            reserved=0)",
            "Deletes the specified key (intended for 64-bit OS).\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n  sub_key\n    A string that must be the name of a subkey of the key identified by\n    the key parameter. This value must not be None, and the key may not\n    have subkeys.\n  access\n    An integer that specifies an access mask that describes the\n    desired security access for the key. Default is KEY_WOW64_64KEY.\n  reserved\n    A reserved integer, and must be zero.  Default is zero.\n\nWhile this function is intended to be used for 64-bit OS, it is also\n available on 32-bit systems.\n\nThis method can not delete keys with subkeys.\n\nIf the function succeeds, the entire key, including all of its values,\nis removed.  If the function fails, an OSError exception is raised.\nOn unsupported Windows versions, NotImplementedError is raised.",
        ),
        (
            "LoadKey",
            load_key,
            "($module, key, sub_key, file_name, /)",
            "Insert data into the registry from a file.\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n  sub_key\n    A string that identifies the sub-key to load.\n  file_name\n    The name of the file to load registry data from.  This file must\n    have been created with the SaveKey() function.  Under the file\n    allocation table (FAT) file system, the filename may not have an\n    extension.\n\nCreates a subkey under the specified key and stores registration\ninformation from a specified file into that subkey.\n\nA call to LoadKey() fails if the calling process does not have the\nSE_RESTORE_PRIVILEGE privilege.\n\nIf key is a handle returned by ConnectRegistry(), then the path\nspecified in fileName is relative to the remote computer.\n\nThe MSDN docs imply key must be in the HKEY_USER or HKEY_LOCAL_MACHINE\ntree.",
        ),
        (
            "SaveKey",
            save_key,
            "($module, key, file_name, /)",
            "Saves the specified key, and all its subkeys to the specified file.\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n  file_name\n    The name of the file to save registry data to.  This file cannot\n    already exist. If this filename includes an extension, it cannot be\n    used on file allocation table (FAT) file systems by the LoadKey(),\n    ReplaceKey() or RestoreKey() methods.\n\nIf key represents a key on a remote computer, the path described by\nfile_name is relative to the remote computer.\n\nThe caller of this method must possess the SeBackupPrivilege\nsecurity privilege.  This function passes NULL for security_attributes\nto the API.",
        ),
        (
            "CloseKey",
            close_key,
            "($module, hkey, /)",
            "Closes a previously opened registry key.\n\n  hkey\n    A previously opened key.\n\nNote that if the key is not closed using this method, it will be\nclosed when the hkey object is destroyed by Python.",
        ),
        (
            "QueryValue",
            query_value,
            "($module, key, sub_key, /)",
            "Retrieves the unnamed value for a key.\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n  sub_key\n    A string that holds the name of the subkey with which the value\n    is associated.  If this parameter is None or empty, the function\n    retrieves the value set by the SetValue() method for the key\n    identified by key.\n\nValues in the registry have name, type, and data components. This method\nretrieves the data for a key's first value that has a NULL name.\nBut since the underlying API call doesn't return the type, you'll\nprobably be happier using QueryValueEx; this function is just here for\ncompleteness.",
        ),
        (
            "QueryValueEx",
            query_value_ex,
            "($module, key, name, /)",
            "Retrieves the type and value of a specified sub-key.\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n  name\n    A string indicating the value to query.\n\nBehaves mostly like QueryValue(), but also returns the type of the\nspecified value name associated with the given open registry key.\n\nThe return value is a tuple of the value and the type_id.",
        ),
        (
            "EnumKey",
            enum_key,
            "($module, key, index, /)",
            "Enumerates subkeys of an open registry key.\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n  index\n    An integer that identifies the index of the key to retrieve.\n\nThe function retrieves the name of one subkey each time it is called.\nIt is typically called repeatedly until an OSError exception is\nraised, indicating no more values are available.",
        ),
        (
            "EnumValue",
            enum_value,
            "($module, key, index, /)",
            "Enumerates values of an open registry key.\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n  index\n    An integer that identifies the index of the value to retrieve.\n\nThe function retrieves the name of one subkey each time it is called.\nIt is typically called repeatedly, until an OSError exception\nis raised, indicating no more values.\n\nThe result is a tuple of 3 items:\n  value_name\n    A string that identifies the value.\n  value_data\n    An object that holds the value data, and whose type depends\n    on the underlying registry type.\n  data_type\n    An integer that identifies the type of the value data.",
        ),
        (
            "QueryInfoKey",
            query_info_key,
            "($module, key, /)",
            "Returns information about a key.\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n\nThe result is a tuple of 3 items:\nAn integer that identifies the number of sub keys this key has.\nAn integer that identifies the number of values this key has.\nAn integer that identifies when the key was last modified (if available)\nas 100's of nanoseconds since Jan 1, 1600.",
        ),
        (
            "FlushKey",
            flush_key,
            "($module, key, /)",
            "Writes all the attributes of a key to the registry.\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n\nIt is not necessary to call FlushKey to change a key.  Registry changes\nare flushed to disk by the registry using its lazy flusher.  Registry\nchanges are also flushed to disk at system shutdown.  Unlike\nCloseKey(), the FlushKey() method returns only when all the data has\nbeen written to the registry.\n\nAn application should only call FlushKey() if it requires absolute\ncertainty that registry changes are on disk.  If you don't know whether\na FlushKey() call is required, it probably isn't.",
        ),
        (
            "CreateKey",
            create_key,
            "($module, key, sub_key, /)",
            "Creates or opens the specified key.\n\n  key\n    An already open key, or one of the predefined HKEY_* constants.\n  sub_key\n    The name of the key this method opens or creates.\n\nIf key is one of the predefined keys, sub_key may be None. In that case,\nthe handle returned is the same key handle passed in to the function.\n\nIf the key already exists, this function opens the existing key.\n\nThe return value is the handle of the opened key.\nIf the function fails, an OSError exception is raised.",
        ),
        (
            "SetValue",
            set_value,
            "($module, key, sub_key, type, value, /)",
            "Associates a value with a specified key.\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n  sub_key\n    A string that names the subkey with which the value is associated.\n  type\n    An integer that specifies the type of the data.  Currently this must\n    be REG_SZ, meaning only strings are supported.\n  value\n    A string that specifies the new value.\n\nIf the key specified by the sub_key parameter does not exist, the\nSetValue function creates it.\n\nValue lengths are limited by available memory. Long values (more than\n2048 bytes) should be stored as files with the filenames stored in\nthe configuration registry to help the registry perform efficiently.\n\nThe key identified by the key parameter must have been opened with\nKEY_SET_VALUE access.",
        ),
        (
            "SetValueEx",
            set_value_ex,
            "($module, key, value_name, reserved, type, value, /)",
            "Stores data in the value field of an open registry key.\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n  value_name\n    A string containing the name of the value to set, or None.\n  reserved\n    Can be anything - zero is always passed to the API.\n  type\n    An integer that specifies the type of the data, one of:\n    REG_BINARY -- Binary data in any form.\n    REG_DWORD -- A 32-bit number.\n    REG_DWORD_LITTLE_ENDIAN -- A 32-bit number in little-endian format. Equivalent to REG_DWORD\n    REG_DWORD_BIG_ENDIAN -- A 32-bit number in big-endian format.\n    REG_EXPAND_SZ -- A null-terminated string that contains unexpanded\n                     references to environment variables (for example,\n                     %PATH%).\n    REG_LINK -- A Unicode symbolic link.\n    REG_MULTI_SZ -- A sequence of null-terminated strings, terminated\n                    by two null characters.  Note that Python handles\n                    this termination automatically.\n    REG_NONE -- No defined value type.\n    REG_QWORD -- A 64-bit number.\n    REG_QWORD_LITTLE_ENDIAN -- A 64-bit number in little-endian format. Equivalent to REG_QWORD.\n    REG_RESOURCE_LIST -- A device-driver resource list.\n    REG_SZ -- A null-terminated string.\n  value\n    A string that specifies the new value.\n\nThis method can also set additional value and type information for the\nspecified key.  The key identified by the key parameter must have been\nopened with KEY_SET_VALUE access.\n\nTo open the key, use the CreateKeyEx() or OpenKeyEx() methods.\n\nValue lengths are limited by available memory. Long values (more than\n2048 bytes) should be stored as files with the filenames stored in\nthe configuration registry to help the registry perform efficiently.",
        ),
        (
            "DeleteKey",
            delete_key,
            "($module, key, sub_key, /)",
            "Deletes the specified key.\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n  sub_key\n    A string that must be the name of a subkey of the key identified by\n    the key parameter. This value must not be None, and the key may not\n    have subkeys.\n\nThis method can not delete keys with subkeys.\n\nIf the function succeeds, the entire key, including all of its values,\nis removed.  If the function fails, an OSError exception is raised.",
        ),
        (
            "DeleteValue",
            delete_value,
            "($module, key, value, /)",
            "Removes a named value from a registry key.\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n  value\n    A string that identifies the value to remove.",
        ),
        (
            "ConnectRegistry",
            connect_registry,
            "($module, computer_name, key, /)",
            "Establishes a connection to the registry on another computer.\n\n  computer_name\n    The name of the remote computer, of the form r\"\\\\computername\".  If\n    None, the local computer is used.\n  key\n    The predefined key to connect to.\n\nThe return value is the handle of the opened key.\nIf the function fails, an OSError exception is raised.",
        ),
        (
            "ExpandEnvironmentStrings",
            expand_environment_strings,
            "($module, string, /)",
            "Expand environment vars.",
        ),
        (
            "DisableReflectionKey",
            disable_reflection_key,
            "($module, key, /)",
            "Disables registry reflection for 32bit processes running on a 64bit OS.\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n\nWill generally raise NotImplementedError if executed on a 32bit OS.\n\nIf the key is not on the reflection list, the function succeeds but has\nno effect.  Disabling reflection for a key does not affect reflection\nof any subkeys.",
        ),
        (
            "EnableReflectionKey",
            enable_reflection_key,
            "($module, key, /)",
            "Restores registry reflection for the specified disabled key.\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n\nWill generally raise NotImplementedError if executed on a 32bit OS.\nRestoring reflection for a key does not affect reflection of any\nsubkeys.",
        ),
        (
            "QueryReflectionKey",
            query_reflection_key,
            "($module, key, /)",
            "Returns the reflection state for the specified key as a bool.\n\n  key\n    An already open key, or any one of the predefined HKEY_* constants.\n\nWill generally raise NotImplementedError if executed on a 32bit OS.",
        ),
    ] {
        let function = match name {
            "OpenKey" | "OpenKeyEx" | "CreateKeyEx" => {
                // `OpenKey`/`CreateKeyEx`: reserved=0, access=KEY_READ/KEY_WRITE
                let code = pyre_interpreter::gateway::builtin_code_new_with_signature(
                    name,
                    func,
                    Some(doc),
                    pyre_interpreter::Signature::new(
                        vec!["key", "sub_key", "reserved", "access"],
                        None,
                        None,
                        0,
                        0,
                    ),
                );
                pyre_interpreter::gateway::function_set_null_positional_defaults(
                    pyre_interpreter::function_new_with_fixed_code(
                        code as *const (),
                        name.to_string(),
                        pyre_object::PY_NULL,
                    ),
                    2,
                )
            }
            "DeleteKeyEx" => {
                // `DeleteKeyEx`: access=KEY_WOW64_64KEY, reserved=0
                let code = pyre_interpreter::gateway::builtin_code_new_with_signature(
                    name,
                    func,
                    Some(doc),
                    pyre_interpreter::Signature::new(
                        vec!["key", "sub_key", "access", "reserved"],
                        None,
                        None,
                        0,
                        0,
                    ),
                );
                pyre_interpreter::gateway::function_set_null_positional_defaults(
                    pyre_interpreter::function_new_with_fixed_code(
                        code as *const (),
                        name.to_string(),
                        pyre_object::PY_NULL,
                    ),
                    2,
                )
            }
            "CloseKey"
            | "QueryInfoKey"
            | "FlushKey"
            | "ExpandEnvironmentStrings"
            | "DisableReflectionKey"
            | "EnableReflectionKey"
            | "QueryReflectionKey" => {
                pyre_interpreter::make_builtin_function_with_arity_and_doc(name, func, 1, doc)
            }
            // [3.14-spec] `interp_winreg.py` exposes these (and `LoadKey`,
            // `SetValue`) through plain `interp2app` signatures, so pypy3
            // takes `QueryValueEx(hkey=..., subkey=...)`; the clinic
            // signatures at v3.14.6 end in `/` and python3.14 raises
            // `takes no keyword arguments`.  Measured on both.
            "SaveKey" | "QueryValue" | "QueryValueEx" | "EnumKey" | "EnumValue" | "CreateKey"
            | "DeleteKey" | "DeleteValue" | "ConnectRegistry" => {
                pyre_interpreter::make_builtin_function_with_arity_and_doc(name, func, 2, doc)
            }
            "LoadKey" => {
                pyre_interpreter::make_builtin_function_with_arity_and_doc(name, func, 3, doc)
            }
            "SetValue" => {
                pyre_interpreter::make_builtin_function_with_arity_and_doc(name, func, 4, doc)
            }
            "SetValueEx" => {
                let code = pyre_interpreter::gateway::builtin_code_new_with_signature(
                    name,
                    func,
                    Some(doc),
                    pyre_interpreter::Signature::new(
                        vec!["key", "value_name", "reserved", "type", "value"],
                        None,
                        None,
                        0,
                        5,
                    ),
                );
                pyre_interpreter::function_new_with_fixed_code(
                    code as *const (),
                    name.to_string(),
                    pyre_object::PY_NULL,
                )
            }
            _ => pyre_interpreter::make_builtin_function_with_doc(name, func, doc),
        };
        let function_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(function);
        let signature_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new(signature));
        unsafe {
            pyre_interpreter::function::fset_func_text_signature(
                pyre_object::gc_roots::shadow_stack_get(function_slot),
                pyre_object::gc_roots::shadow_stack_get(signature_slot),
            );
        }
        crate::__pyre_put_new!(
            ns_slot,
            name,
            pyre_object::gc_roots::shadow_stack_get(function_slot)
        );
    }
}
