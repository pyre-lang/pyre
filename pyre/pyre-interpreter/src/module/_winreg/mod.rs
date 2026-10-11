//! winreg module — PyPy: pypy/module/_winreg/ (`applevel_name = 'winreg'`).
//!
//! `importlib._bootstrap_external` does an eager `import winreg` on every
//! `sys.platform == "win32"` build, so the module must exist for the import
//! machinery — and therefore `import site` — to come up.  The integer
//! constants are always present; the `PyHKEY` handle object and the
//! Open/Query/Enum/Set/Delete key & value functions are registered when the
//! `host_env` feature is on (they delegate every Win32 `Reg*` call to
//! `rustpython_host_env::winreg`).
#[cfg(feature = "host_env")]
macro_rules! reg_int {
    (HKEY_CLASSES_ROOT) => {
        rustpython_host_env::winreg::HKEY_CLASSES_ROOT as u32 as i64
    };
    (HKEY_CURRENT_USER) => {
        rustpython_host_env::winreg::HKEY_CURRENT_USER as u32 as i64
    };
    (HKEY_LOCAL_MACHINE) => {
        rustpython_host_env::winreg::HKEY_LOCAL_MACHINE as u32 as i64
    };
    (HKEY_USERS) => {
        rustpython_host_env::winreg::HKEY_USERS as u32 as i64
    };
    (HKEY_PERFORMANCE_DATA) => {
        rustpython_host_env::winreg::HKEY_PERFORMANCE_DATA as u32 as i64
    };
    (HKEY_CURRENT_CONFIG) => {
        rustpython_host_env::winreg::HKEY_CURRENT_CONFIG as u32 as i64
    };
    (HKEY_DYN_DATA) => {
        rustpython_host_env::winreg::HKEY_DYN_DATA as u32 as i64
    };
    ($name:ident) => {
        rustpython_host_env::winreg::$name as i64
    };
}

#[cfg(not(feature = "host_env"))]
macro_rules! reg_int {
    (HKEY_CLASSES_ROOT) => {
        0x8000_0000u32 as i64
    };
    (HKEY_CURRENT_USER) => {
        0x8000_0001u32 as i64
    };
    (HKEY_LOCAL_MACHINE) => {
        0x8000_0002u32 as i64
    };
    (HKEY_USERS) => {
        0x8000_0003u32 as i64
    };
    (HKEY_PERFORMANCE_DATA) => {
        0x8000_0004u32 as i64
    };
    (HKEY_CURRENT_CONFIG) => {
        0x8000_0005u32 as i64
    };
    (HKEY_DYN_DATA) => {
        0x8000_0006u32 as i64
    };
    (KEY_QUERY_VALUE) => {
        0x0001 as i64
    };
    (KEY_SET_VALUE) => {
        0x0002 as i64
    };
    (KEY_CREATE_SUB_KEY) => {
        0x0004 as i64
    };
    (KEY_ENUMERATE_SUB_KEYS) => {
        0x0008 as i64
    };
    (KEY_NOTIFY) => {
        0x0010 as i64
    };
    (KEY_CREATE_LINK) => {
        0x0020 as i64
    };
    (KEY_WOW64_64KEY) => {
        0x0100 as i64
    };
    (KEY_WOW64_32KEY) => {
        0x0200 as i64
    };
    (KEY_WRITE) => {
        0x0002_0006 as i64
    };
    (KEY_READ) => {
        0x0002_0019 as i64
    };
    (KEY_EXECUTE) => {
        0x0002_0019 as i64
    };
    (KEY_ALL_ACCESS) => {
        0x000F_003F as i64
    };
    (REG_NONE) => {
        0 as i64
    };
    (REG_SZ) => {
        1 as i64
    };
    (REG_EXPAND_SZ) => {
        2 as i64
    };
    (REG_BINARY) => {
        3 as i64
    };
    (REG_DWORD) => {
        4 as i64
    };
    (REG_DWORD_LITTLE_ENDIAN) => {
        4 as i64
    };
    (REG_DWORD_BIG_ENDIAN) => {
        5 as i64
    };
    (REG_LINK) => {
        6 as i64
    };
    (REG_MULTI_SZ) => {
        7 as i64
    };
    (REG_RESOURCE_LIST) => {
        8 as i64
    };
    (REG_FULL_RESOURCE_DESCRIPTOR) => {
        9 as i64
    };
    (REG_RESOURCE_REQUIREMENTS_LIST) => {
        10 as i64
    };
    (REG_QWORD) => {
        11 as i64
    };
    (REG_QWORD_LITTLE_ENDIAN) => {
        11 as i64
    };
    (REG_OPTION_RESERVED) => {
        0 as i64
    };
    (REG_OPTION_NON_VOLATILE) => {
        0 as i64
    };
    (REG_OPTION_VOLATILE) => {
        1 as i64
    };
    (REG_OPTION_CREATE_LINK) => {
        2 as i64
    };
    (REG_OPTION_BACKUP_RESTORE) => {
        4 as i64
    };
    (REG_OPTION_OPEN_LINK) => {
        8 as i64
    };
    (REG_CREATED_NEW_KEY) => {
        1 as i64
    };
    (REG_OPENED_EXISTING_KEY) => {
        2 as i64
    };
    (REG_WHOLE_HIVE_VOLATILE) => {
        1 as i64
    };
    (REG_REFRESH_HIVE) => {
        2 as i64
    };
    (REG_NO_LAZY_FLUSH) => {
        4 as i64
    };
    (REG_NOTIFY_CHANGE_NAME) => {
        1 as i64
    };
    (REG_NOTIFY_CHANGE_ATTRIBUTES) => {
        2 as i64
    };
    (REG_NOTIFY_CHANGE_LAST_SET) => {
        4 as i64
    };
    (REG_NOTIFY_CHANGE_SECURITY) => {
        8 as i64
    };
    (REG_LEGAL_CHANGE_FILTER) => {
        0x1000_000F as i64
    };
    (REG_LEGAL_OPTION) => {
        0x001F as i64
    };
}

pyre_interpreter::py_module! {
    "winreg",
    int_constants: {
        // Predefined root handles (winreg.h). Python surfaces these as the
        // unsigned handle value, so cast through u32.
        "HKEY_CLASSES_ROOT" => reg_int!(HKEY_CLASSES_ROOT),
        "HKEY_CURRENT_USER" => reg_int!(HKEY_CURRENT_USER),
        "HKEY_LOCAL_MACHINE" => reg_int!(HKEY_LOCAL_MACHINE),
        "HKEY_USERS" => reg_int!(HKEY_USERS),
        "HKEY_PERFORMANCE_DATA" => reg_int!(HKEY_PERFORMANCE_DATA),
        "HKEY_CURRENT_CONFIG" => reg_int!(HKEY_CURRENT_CONFIG),
        "HKEY_DYN_DATA" => reg_int!(HKEY_DYN_DATA),

        // Access-right masks (winnt.h).
        "KEY_QUERY_VALUE" => reg_int!(KEY_QUERY_VALUE),
        "KEY_SET_VALUE" => reg_int!(KEY_SET_VALUE),
        "KEY_CREATE_SUB_KEY" => reg_int!(KEY_CREATE_SUB_KEY),
        "KEY_ENUMERATE_SUB_KEYS" => reg_int!(KEY_ENUMERATE_SUB_KEYS),
        "KEY_NOTIFY" => reg_int!(KEY_NOTIFY),
        "KEY_CREATE_LINK" => reg_int!(KEY_CREATE_LINK),
        "KEY_WOW64_64KEY" => reg_int!(KEY_WOW64_64KEY),
        "KEY_WOW64_32KEY" => reg_int!(KEY_WOW64_32KEY),
        "KEY_WRITE" => reg_int!(KEY_WRITE),
        "KEY_READ" => reg_int!(KEY_READ),
        "KEY_EXECUTE" => reg_int!(KEY_EXECUTE),
        "KEY_ALL_ACCESS" => reg_int!(KEY_ALL_ACCESS),

        // Value types (winnt.h REG_*).
        "REG_NONE" => reg_int!(REG_NONE),
        "REG_SZ" => reg_int!(REG_SZ),
        "REG_EXPAND_SZ" => reg_int!(REG_EXPAND_SZ),
        "REG_BINARY" => reg_int!(REG_BINARY),
        "REG_DWORD" => reg_int!(REG_DWORD),
        "REG_DWORD_LITTLE_ENDIAN" => reg_int!(REG_DWORD_LITTLE_ENDIAN),
        "REG_DWORD_BIG_ENDIAN" => reg_int!(REG_DWORD_BIG_ENDIAN),
        "REG_LINK" => reg_int!(REG_LINK),
        "REG_MULTI_SZ" => reg_int!(REG_MULTI_SZ),
        "REG_RESOURCE_LIST" => reg_int!(REG_RESOURCE_LIST),
        "REG_FULL_RESOURCE_DESCRIPTOR" => reg_int!(REG_FULL_RESOURCE_DESCRIPTOR),
        "REG_RESOURCE_REQUIREMENTS_LIST" => reg_int!(REG_RESOURCE_REQUIREMENTS_LIST),
        "REG_QWORD" => reg_int!(REG_QWORD),
        "REG_QWORD_LITTLE_ENDIAN" => reg_int!(REG_QWORD_LITTLE_ENDIAN),

        // Key-creation options and disposition (winnt.h).
        "REG_OPTION_RESERVED" => reg_int!(REG_OPTION_RESERVED),
        "REG_OPTION_NON_VOLATILE" => reg_int!(REG_OPTION_NON_VOLATILE),
        "REG_OPTION_VOLATILE" => reg_int!(REG_OPTION_VOLATILE),
        "REG_OPTION_CREATE_LINK" => reg_int!(REG_OPTION_CREATE_LINK),
        "REG_OPTION_BACKUP_RESTORE" => reg_int!(REG_OPTION_BACKUP_RESTORE),
        "REG_OPTION_OPEN_LINK" => reg_int!(REG_OPTION_OPEN_LINK),
        "REG_CREATED_NEW_KEY" => reg_int!(REG_CREATED_NEW_KEY),
        "REG_OPENED_EXISTING_KEY" => reg_int!(REG_OPENED_EXISTING_KEY),
        "REG_WHOLE_HIVE_VOLATILE" => reg_int!(REG_WHOLE_HIVE_VOLATILE),
        "REG_REFRESH_HIVE" => reg_int!(REG_REFRESH_HIVE),
        "REG_NO_LAZY_FLUSH" => reg_int!(REG_NO_LAZY_FLUSH),

        // RegNotifyChangeKeyValue filters (winnt.h).
        "REG_NOTIFY_CHANGE_NAME" => reg_int!(REG_NOTIFY_CHANGE_NAME),
        "REG_NOTIFY_CHANGE_ATTRIBUTES" => reg_int!(REG_NOTIFY_CHANGE_ATTRIBUTES),
        "REG_NOTIFY_CHANGE_LAST_SET" => reg_int!(REG_NOTIFY_CHANGE_LAST_SET),
        "REG_NOTIFY_CHANGE_SECURITY" => reg_int!(REG_NOTIFY_CHANGE_SECURITY),
        // Both masks carry a bit whose own name `winreg` does not publish:
        // `REG_NOTIFY_THREAD_AGNOSTIC` in the filter, `REG_OPTION_DONT_VIRTUALIZE`
        // in the option word.
        "REG_LEGAL_CHANGE_FILTER" => reg_int!(REG_LEGAL_CHANGE_FILTER),
        "REG_LEGAL_OPTION" => reg_int!(REG_LEGAL_OPTION),
    },
    extra_init: |ns| {
        #[cfg(feature = "host_env")]
        imp::install(ns);
    }
}

#[cfg(feature = "host_env")]
mod imp {
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

    // ── PyHKEY handle object ──
    // The raw `HKEY` is stored on the instance dict under `_handle` as its
    // pointer value; `int(key)` and passing a key where a handle is expected
    // read it back.  A closed handle is left as 0.
    // Qualified, so the type reports `winreg` as its `__module__` the way a
    // static type with a dotted `tp_name` does; `__name__` stays `PyHKEY`.
    pyre_interpreter::py_class! {
        "winreg.PyHKEY",
        methods: {
            #[doc = "Closes the underlying Windows handle.\n\nIf the handle is already closed, no error is raised."]
            fn Close(self_obj: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
                let raw = take_handle(self_obj);
                if !raw.is_null() {
                    host_reg::close_key(raw);
                }
                Ok(())
            }
            #[doc = "Detaches the Windows handle from the handle object.\n\nThe result is the value of the handle before it is detached.  If the\nhandle is already detached, this will return zero.\n\nAfter calling this function, the handle is effectively invalidated,\nbut the handle is not closed.  You would call this function when you\nneed the underlying win32 handle to exist beyond the lifetime of the\nhandle object."]
            fn Detach(self_obj: PyObjectRef) -> PyObjectRef {
                int_from_ptr(take_handle(self_obj))
            }
            fn __int__(self_obj: PyObjectRef) -> PyObjectRef {
                int_from_ptr(get_handle(self_obj))
            }
            fn __bool__(self_obj: PyObjectRef) -> bool {
                !get_handle(self_obj).is_null()
            }
            fn __enter__(self_obj: PyObjectRef) -> PyObjectRef {
                self_obj
            }
            fn __exit__(
                self_obj: PyObjectRef,
                _exc_type: PyObjectRef,
                _exc_value: PyObjectRef,
                _traceback: PyObjectRef,
            ) -> Result<bool, pyre_interpreter::PyError> {
                let raw = take_handle(self_obj);
                if !raw.is_null() {
                    host_reg::close_key(raw);
                }
                Ok(false)
            }
            fn __str__(self_obj: PyObjectRef) -> PyObjectRef {
                // `PyHKEY_strFunc` — `PyUnicode_FromFormat("<PyHKEY:%p>", …)`.
                // `%p` is the host's own pointer spelling, which on Windows is
                // the full width in upper-case digits with no prefix of its
                // own; `PyUnicode_FromFormat` is what puts the `0x` in front.
                // The type keeps `object.__repr__`, so `repr` and `str` of a
                // handle read differently.
                let raw = get_handle(self_obj) as usize;
                w_str_new_managed(&format!(
                    "<PyHKEY:0x{raw:0width$X}>",
                    width = core::mem::size_of::<usize>() * 2
                ))
            }
        },
        properties: {
            fn handle(_descr: PyObjectRef, self_obj: PyObjectRef) -> PyObjectRef {
                // `W_HKEY.descr_handle_get` — the handle as the integer it
                // points at. `PyHKEY_memberlist` reads the same field through
                // a `Py_T_INT` member, which on a 64-bit host truncates it to
                // the low word; the property answers the whole pointer, so it
                // agrees with `int(key)` for every handle rather than only for
                // one below 2**31.
                int_from_ptr(get_handle(self_obj))
            }
        }
    }

    fn store_handle(obj: PyObjectRef, raw: HKEY) {
        let d = pyre_interpreter::baseobjspace::getdict_native(obj);
        if !d.is_null() {
            unsafe { w_dict_setitem_str(d, "_handle", int_from_ptr(raw)) };
        }
    }

    fn get_handle(obj: PyObjectRef) -> HKEY {
        stored_handle(obj).unwrap_or(core::ptr::null_mut())
    }

    /// Read the stored handle and reset it to 0 (used by `Close`/`Detach`).
    fn take_handle(obj: PyObjectRef) -> HKEY {
        let raw = get_handle(obj);
        store_handle(obj, core::ptr::null_mut());
        raw
    }

    fn make_pyhkey(raw: HKEY) -> PyObjectRef {
        let obj = w_instance_new(type_object());
        store_handle(obj, raw);
        obj
    }

    /// The stored `_handle` value, if `obj` is a `PyHKEY`.  `uint_w` reads both
    /// the small-int and long forms a handle pointer may take.
    fn stored_handle(obj: PyObjectRef) -> Option<HKEY> {
        let d = pyre_interpreter::baseobjspace::getdict_native(obj);
        if d.is_null() {
            return None;
        }
        let value = unsafe { w_dict_getitem_str(d, "_handle") }?;
        pyre_interpreter::baseobjspace::uint_w(value)
            .ok()
            .map(|u| u as usize as HKEY)
    }

    /// `PyHKEY_Check` — the handle object this module hands out, and nothing
    /// that merely happens to carry a `_handle` entry.
    fn is_pyhkey(obj: PyObjectRef) -> bool {
        pyre_interpreter::typedef::r#type(obj).map(|w_type| w_type.as_ptr()) == Some(type_object())
    }

    /// `PyHKEY_AsHKEY` — a key argument is a `PyHKEY`, an integer handle (the
    /// predefined `HKEY_*` roots are published as their pointer value), or
    /// `None` at `CloseKey`, the one boundary that reads it as the null
    /// handle rather than as a mistake.
    fn as_hkey(obj: PyObjectRef, none_ok: bool) -> Result<HKEY, pyre_interpreter::PyError> {
        if unsafe { is_none(obj) } {
            if none_ok {
                return Ok(core::ptr::null_mut());
            }
            return Err(pyre_interpreter::PyError::type_error(
                "None is not a valid HKEY in this context",
            ));
        }
        if is_pyhkey(obj) {
            return Ok(get_handle(obj));
        }
        if unsafe { is_int_or_long(obj) } {
            return Ok(pyre_interpreter::baseobjspace::uint_w(obj)? as usize as HKEY);
        }
        Err(pyre_interpreter::PyError::type_error(
            "The object is not a PyHKEY object",
        ))
    }

    /// The wording a boundary with no keyword table gives one anyway. It
    /// qualifies itself with the module, which the clinic's keyword binder
    /// does not.
    fn no_keywords(name: &str) -> pyre_interpreter::PyError {
        pyre_interpreter::PyError::type_error(format!("winreg.{name}() takes no keyword arguments"))
    }

    /// A positional-only call of fixed arity — `_PyArg_CheckPositional`
    /// reports the count it wanted against the count it got.
    fn exact_args<'a>(
        args: &'a [PyObjectRef],
        name: &str,
        count: usize,
    ) -> Result<&'a [PyObjectRef], pyre_interpreter::PyError> {
        let (pos, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
        if pyre_interpreter::builtins::has_real_kwargs(kwargs) {
            return Err(no_keywords(name));
        }
        if pos.len() != count {
            return Err(pyre_interpreter::PyError::type_error(format!(
                "{name} expected {count} arguments, got {}",
                pos.len()
            )));
        }
        Ok(pos)
    }

    /// `METH_O` — the single key the seven one-argument calls take, whose
    /// count is checked by the call machinery rather than by a converter.
    fn single_arg(
        args: &[PyObjectRef],
        name: &str,
    ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let (pos, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
        if pyre_interpreter::builtins::has_real_kwargs(kwargs) {
            return Err(no_keywords(name));
        }
        if pos.len() != 1 {
            return Err(pyre_interpreter::PyError::type_error(format!(
                "winreg.{name}() takes exactly one argument ({} given)",
                pos.len()
            )));
        }
        Ok(pos[0])
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
        let (pos, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
        if pos.len() > params.len() {
            return Err(pyre_interpreter::PyError::type_error(format!(
                "{name}() takes at most {} arguments ({} given)",
                params.len(),
                pos.len()
            )));
        }
        let mut bound = [None; 4];
        for (index, key) in params.iter().enumerate() {
            let value = pyre_interpreter::builtins::bind_pos_or_kw(
                pos,
                kwargs,
                index,
                key,
                name,
                index + 1,
            )?;
            if value.is_none() && index < 2 {
                return Err(pyre_interpreter::PyError::type_error(format!(
                    "{name}() missing required argument '{key}' (pos {})",
                    index + 1
                )));
            }
            bound[index] = value;
        }
        pyre_interpreter::builtins::kwarg_reject_unknown(kwargs, &params, name)?;
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
                if unsafe { is_none(value) } {
                    // No string is still a terminated one.
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
                if unsafe { pyre_object::bytesobject::is_bytes_like(value) } {
                    Ok(unsafe { pyre_object::bytesobject::bytes_like_data(value) }.to_vec())
                } else {
                    Err(Py2RegError::Raised(pyre_interpreter::PyError::type_error(
                        format!(
                            "Objects of type '{}' can not be used as binary registry values",
                            pyre_interpreter::error::type_name_of(value)
                        ),
                    )))
                }
            }
        }
    }

    // ── module functions ──
    /// `OpenKey` and `OpenKeyEx` are one implementation under two names, and
    /// each names itself in what it refuses.
    fn open_key_named(
        args: &[PyObjectRef],
        name: &str,
    ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let bound = bind_key_args(args, name, ["key", "sub_key", "reserved", "access"])?;
        let key = as_hkey(bound[0].expect("key is required"), false)?;
        let wide_sub = wide_or_empty(opt_text(
            bound[1].expect("sub_key is required"),
            name,
            "'sub_key'",
        )?)?;
        let reserved = bound[2].map(int_arg).transpose()?.unwrap_or(0) as u32;
        let access = bound[3]
            .map(dword_arg)
            .transpose()?
            .unwrap_or(host_reg::KEY_READ);
        let mut out: HKEY = core::ptr::null_mut();
        let rc = unsafe { host_reg::open_key_ex(key, &wide_sub, reserved, access, &mut out) };
        if rc != 0 {
            return Err(win_err(rc));
        }
        Ok(make_pyhkey(out))
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
        let wide_sub = wide_or_empty(opt_text(
            bound[1].expect("sub_key is required"),
            "CreateKeyEx",
            "'sub_key'",
        )?)?;
        let reserved = bound[2].map(int_arg).transpose()?.unwrap_or(0) as u32;
        let access = bound[3]
            .map(dword_arg)
            .transpose()?
            .unwrap_or(host_reg::KEY_WRITE);
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
        Ok(make_pyhkey(out))
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
        let sub_key = req_text(
            bound[1].expect("sub_key is required"),
            "DeleteKeyEx",
            "'sub_key'",
        )?;
        let access = bound[2]
            .map(dword_arg)
            .transpose()?
            .unwrap_or(host_reg::KEY_WOW64_64KEY);
        let reserved = bound[3].map(int_arg).transpose()?.unwrap_or(0) as u32;
        check(unsafe { host_reg::delete_key_ex(key, &wide(sub_key)?, access, reserved) })
    }

    fn load_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        // Restores a hive a `SaveKey` wrote.  Both it and SaveKey need
        // SE_RESTORE_NAME/SE_BACKUP_NAME, so an ordinary account gets
        // ERROR_PRIVILEGE_NOT_HELD rather than a result.
        let pos = exact_args(args, "LoadKey", 3)?;
        let key = as_hkey(pos[0], false)?;
        let sub_key = req_text(pos[1], "LoadKey", "2")?;
        let file_name = req_text(pos[2], "LoadKey", "3")?;
        check(unsafe { host_reg::load_key(key, &wide(sub_key)?, &wide(file_name)?) })
    }

    fn save_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let pos = exact_args(args, "SaveKey", 2)?;
        let key = as_hkey(pos[0], false)?;
        let file_name = req_text(pos[1], "SaveKey", "2")?;
        check(unsafe { host_reg::save_key(key, &wide(file_name)?) })
    }

    fn close_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        // `PyHKEY_Close` takes `None` as the null handle, so closing what was
        // never opened is not an error.
        let obj = single_arg(args, "CloseKey")?;
        let raw = if is_pyhkey(obj) {
            take_handle(obj)
        } else {
            as_hkey(obj, true)?
        };
        if !raw.is_null() {
            host_reg::close_key(raw);
        }
        Ok(w_none())
    }

    /// `winreg_QueryValue_impl` — the key's unnamed value, read as text.
    ///
    /// `host_reg::query_default_value` answers a `String`, so a `REG_SZ` whose
    /// units hold an unpaired surrogate has no spelling it can give back; the
    /// units are read here instead and carried across as themselves
    /// (`PyUnicode_FromWideChar`).
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
            let rc = unsafe {
                host_reg::open_key_ex(key, sub_key, 0, host_reg::KEY_QUERY_VALUE, &mut out)
            };
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
        let wide_sub = wide_or_empty(opt_text(pos[1], "QueryValue", "2")?)?;
        Ok(w_str_from_wtf8_managed(query_default_value(
            key, &wide_sub,
        )?))
    }

    fn query_value_ex(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let pos = exact_args(args, "QueryValueEx", 2)?;
        let key = as_hkey(pos[0], false)?;
        let name = wide_or_empty(opt_text(pos[1], "QueryValueEx", "2")?)?;
        match host_reg::query_value_bytes(key, &name) {
            Ok((data, typ)) => {
                let mut fields = pyre_object::gc_roots::RootedItems::new();
                fields.push(reg2py(&data, typ));
                fields.push(w_int_new(typ as i64));
                Ok(w_tuple_new(fields.take()))
            }
            Err(code) => Err(win_err(code)),
        }
    }

    fn enum_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let pos = exact_args(args, "EnumKey", 2)?;
        let key = as_hkey(pos[0], false)?;
        let index = int_arg(pos[1])? as u32;
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
        let index = int_arg(pos[1])? as u32;
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
        // The two lengths the key reports leave no room for the terminators,
        // and a key whose values change size under the enumeration — the
        // performance hive is one, reporting nothing and answering megabytes —
        // reports neither. `ERROR_MORE_DATA` is the answer to a buffer too
        // small, and the data one doubles until the value fits.
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
                    .checked_mul(2)
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

    fn flush_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        check(host_reg::flush_key(as_hkey(
            single_arg(args, "FlushKey")?,
            false,
        )?))
    }

    fn create_key(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let pos = exact_args(args, "CreateKey", 2)?;
        let key = as_hkey(pos[0], false)?;
        let wide_sub = wide_or_empty(opt_text(pos[1], "CreateKey", "2")?)?;
        let mut out: HKEY = core::ptr::null_mut();
        let rc = unsafe { host_reg::create_key(key, &wide_sub, &mut out) };
        if rc != 0 {
            return Err(win_err(rc));
        }
        Ok(make_pyhkey(out))
    }

    fn set_value(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        // SetValue only writes REG_SZ, and the type is read before it is
        // judged, so a non-integer is an integer's complaint rather than the
        // wrong-type one.  Every argument is spelled before any of them is
        // judged, though, so a `value` that is no string is that argument's
        // complaint even when the type is wrong too.
        let pos = exact_args(args, "SetValue", 4)?;
        let key = as_hkey(pos[0], false)?;
        let sub_key = wide_or_empty(opt_text(pos[1], "SetValue", "2")?)?;
        let typ = dword_arg(pos[2])?;
        let value = req_text(pos[3], "SetValue", "4")?;
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
        let name = opt_text(pos[1], "SetValueEx", "2")?;
        // `reserved` is documented as "can be anything" and zero is what
        // reaches the API, so it is read but not converted.
        let typ = dword_arg(pos[3])?;
        let data = py2reg(pos[4], typ).map_err(|error| match error {
            Py2RegError::Raised(error) => error,
            Py2RegError::Unconvertible => pyre_interpreter::PyError::value_error(
                "Could not convert the data to the specified type.",
            ),
        })?;
        let wide_name = name.map(wide).transpose()?;
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
        let sub_key = req_text(pos[1], "DeleteKey", "2")?;
        check(unsafe { host_reg::delete_key(key, &wide(sub_key)?) })
    }

    fn delete_value(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let pos = exact_args(args, "DeleteValue", 2)?;
        let key = as_hkey(pos[0], false)?;
        let name = opt_text(pos[1], "DeleteValue", "2")?;
        let wide_name = name.map(wide).transpose()?;
        check(unsafe { host_reg::delete_value(key, wide_name.as_deref()) })
    }

    fn connect_registry(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let pos = exact_args(args, "ConnectRegistry", 2)?;
        let computer = opt_text(pos[0], "ConnectRegistry", "1")?;
        let key = as_hkey(pos[1], false)?;
        let wide_computer = computer.map(wide).transpose()?;
        let mut out: HKEY = core::ptr::null_mut();
        let rc = unsafe { host_reg::connect_registry(wide_computer.as_deref(), key, &mut out) };
        if rc != 0 {
            return Err(win_err(rc));
        }
        Ok(make_pyhkey(out))
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
        let input = req_text(
            single_arg(args, "ExpandEnvironmentStrings")?,
            "ExpandEnvironmentStrings",
            "",
        )?;
        match expand_environment_strings_wide(&wide(input)?) {
            Some(value) => Ok(w_str_from_wtf8_managed(value)),
            None => Err(win_err(
                std::io::Error::last_os_error().raw_os_error().unwrap_or(0) as u32,
            )),
        }
    }

    fn disable_reflection_key(
        args: &[PyObjectRef],
    ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        check(host_reg::disable_reflection_key(as_hkey(
            single_arg(args, "DisableReflectionKey")?,
            false,
        )?))
    }

    fn enable_reflection_key(
        args: &[PyObjectRef],
    ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        check(host_reg::enable_reflection_key(as_hkey(
            single_arg(args, "EnableReflectionKey")?,
            false,
        )?))
    }

    fn query_reflection_key(
        args: &[PyObjectRef],
    ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let key = as_hkey(single_arg(args, "QueryReflectionKey")?, false)?;
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
        crate::__pyre_put_new!(ns_slot, "HKEYType", type_object());
        // `Py_tp_doc`, and the signature line clinic writes ahead of each
        // `PyHKEY_methods` docstring. The type has just been built by the line
        // above and nothing has read it yet, so the class dict takes them
        // directly.
        unsafe {
            let ty = type_object();
            let class_ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(
                pyre_object::w_type_get_dict_ptr(ty) as PyObjectRef
            );
            crate::__pyre_put_new!(
                class_ns_slot,
                "__doc__",
                w_str_new(
                    "PyHKEY Object - A Python object, representing a win32 registry key.\n\nThis object wraps a Windows HKEY object, automatically closing it when\nthe object is destroyed.  To guarantee cleanup, you can call either\nthe Close() method on the PyHKEY, or the CloseKey() method.\n\nAll functions which accept a handle object also accept an integer --\nhowever, use of the handle object is encouraged.\n\nFunctions:\nClose() - Closes the underlying handle.\nDetach() - Returns the integer Win32 handle, detaching it from the object\n\nProperties:\nhandle - The integer Win32 handle.\n\nOperations:\n__bool__ - Handles with an open object return true, otherwise false.\n__int__ - Converting a handle to an integer returns the Win32 handle.\n__enter__, __exit__ - Context manager support for 'with' statement,\nautomatically closes handle.",
                )
            );
            for (method, signature) in [
                ("Close", "($self, /)"),
                ("Detach", "($self, /)"),
                ("__enter__", "($self, /)"),
                ("__exit__", "($self, exc_type, exc_value, traceback, /)"),
            ] {
                let Some(function) = pyre_object::w_dict_getitem_str(
                    pyre_object::gc_roots::shadow_stack_get(class_ns_slot),
                    method,
                ) else {
                    continue;
                };
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
            let function = pyre_interpreter::make_builtin_function_with_doc(name, func, doc);
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
}
