use std::fmt;

use pyre_object::pyobject::{
    BOOL_TYPE, ELLIPSIS_TYPE, FLOAT_TYPE, INSTANCE_TYPE, INT_TYPE, LONG_TYPE, MODULE_TYPE,
    NONE_TYPE, PyObjectRef, PyType, STR_TYPE, TYPE_TYPE,
};
use rustpython_wtf8::{Wtf8, Wtf8Buf};

use crate::{
    BUILTIN_CODE_TYPE, BUILTIN_FUNCTION_TYPE, FUNCTION_TYPE, METHOD_DESCRIPTOR_TYPE,
    builtin_code_name, function_get_name, function_get_qualname,
};

/// Address half of `W_Root.getaddrstring`: `2 * HUGEVAL_BYTES` lower-case
/// hex digits, zero-padded to the pointer width, after a literal `0x`.
///
/// `PyUnicode_FromFormat`'s `%p` hands the pointer to the platform `printf`
/// instead, which pads and uppercases on MSVC and does neither on glibc; no
/// `lib-python/3` assertion pins either spelling (the repr tests match
/// `0x[0-9a-fA-F]+`), so the text follows `getaddrstring` on every host.
///
/// The bits are whatever the caller already computed. A Rust struct that is
/// not a moving GC object — `PyFrame` (born with `try_gc_alloc_stable_raw`)
/// and the `_thread` lock types (born with `malloc_typed_stable`) — passes
/// its own address here. A GC object that can move prints
/// [`repr_gc_addr`] instead, so the text is `getaddrstring`'s id rather than
/// the nursery address it currently occupies.
pub fn repr_addr(addr: usize) -> String {
    format!("0x{addr:0width$x}", width = size_of::<usize>() * 2)
}

/// Address half of `W_Root.getaddrstring` for a GC object.
///
/// `getaddrstring` renders `space.id(self)`. For an object with no
/// `immutable_unique_id` that is `compute_unique_id`, which incminimark
/// implements as `id_or_identityhash`: the shadow a nursery object will be
/// copied to, or the object's own address once it is old. That is the same
/// number `ObjSpace.id` / `builtin_id` returns, so `hex(id(obj))` and this
/// text stay equal across a minor collection.
///
/// Allocating the id shadow can collect. The receiver is rooted across that
/// call and the hash reads the reloaded word — a copy taken before the
/// shadow allocation addresses the pre-move object, and `id_or_identityhash`
/// refuses a forwarded header.
pub fn repr_gc_addr(obj: PyObjectRef) -> String {
    let _roots = pyre_object::gc_roots::push_roots();
    let live = pyre_object::gc_roots::pin_root(obj);
    repr_addr(pyre_object::gc_hook::gc_identity_hash(live as usize))
}

/// Try to call a dunder method (__repr__, __str__, etc.) on an instance,
/// returning the raw result object when it is a `str`.
pub(crate) unsafe fn try_call_dunder_obj(
    obj: PyObjectRef,
    name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    unsafe {
        if !pyre_object::is_instance(obj) {
            return Ok(None);
        }
        // Resolving `name` walks the MRO, which allocates: an uncached type
        // computes its `__mro__` on the spot and the namespace probe mints its
        // key.  Publish the receiver first and read it back for the call —
        // a copy taken before a collection addresses the pre-move object, and
        // the descriptor rejects that as an instance of the wrong type.
        let _roots = pyre_object::gc_roots::push_roots();
        let root_base = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        let obj = || pyre_object::gc_roots::shadow_stack_get(root_base);
        let Some(method) = crate::baseobjspace::lookup(obj(), name) else {
            return Ok(None);
        };
        if method.is_null() {
            return Ok(None);
        }
        let _ = pyre_object::gc_roots::pin_root(method);
        let method = || pyre_object::gc_roots::shadow_stack_get(root_base + 1);
        // A raising `__repr__`/`__str__` propagates; a non-string return is a
        // TypeError (`object.c slot_tp_repr` / `slot_tp_str`).
        let w_type = crate::typedef::r#type(obj()).map_or(pyre_object::PY_NULL, |p| p.as_ptr());
        let result = crate::baseobjspace::get_and_call_function(method(), obj(), w_type, &[])?;
        if pyre_object::is_str(result) {
            return Ok(Some(result));
        }
        Err(dunder_returned_non_string(name, result))
    }
}

pub(crate) unsafe fn try_call_dunder_obj_above_object(
    obj: PyObjectRef,
    name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    unsafe {
        if !pyre_object::is_instance(obj) {
            return Ok(None);
        }
        let Some(w_type) = crate::typedef::r#type(obj) else {
            return Ok(None);
        };
        // Published across the allocating MRO walk, as in `try_call_dunder_obj`.
        let _roots = pyre_object::gc_roots::push_roots();
        let root_base = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        let obj = || pyre_object::gc_roots::shadow_stack_get(root_base);
        let Some((src, method)) =
            crate::baseobjspace::lookup_where_with_method_cache(w_type.as_ptr(), name)
        else {
            return Ok(None);
        };
        if method.is_null() || std::ptr::eq(src, crate::typedef::w_object()) {
            return Ok(None);
        }
        let _ = pyre_object::gc_roots::pin_root(method);
        let method = || pyre_object::gc_roots::shadow_stack_get(root_base + 1);
        let result =
            crate::baseobjspace::get_and_call_function(method(), obj(), w_type.as_ptr(), &[])?;
        if pyre_object::is_str(result) {
            return Ok(Some(result));
        }
        Err(dunder_returned_non_string(name, result))
    }
}

/// WTF-8 carrying variant of [`try_call_dunder`]: dispatches `__str__` /
/// `__repr__` on an instance and preserves a surrogate-bearing result
/// instead of folding it through a `&str` (which would panic).
unsafe fn try_call_dunder_wtf8(
    obj: PyObjectRef,
    name: &str,
) -> Result<Option<Wtf8Buf>, crate::PyError> {
    unsafe {
        Ok(try_call_dunder_obj(obj, name)?
            .map(|result| pyre_object::w_str_get_wtf8(result).to_wtf8_buf()))
    }
}

/// `TypeError: __repr__ returned non-string (type X)` for a dunder whose
/// override returned a non-`str` (CPython 3.14 `slot_tp_repr`).
unsafe fn dunder_returned_non_string(name: &str, result: PyObjectRef) -> crate::PyError {
    let type_name = match unsafe { crate::typedef::r#type(result) } {
        Some(tp) => unsafe { pyre_object::w_type_get_name(tp.as_ptr()) }.to_string(),
        None => "object".to_string(),
    };
    crate::PyError::type_error(format!("{name} returned non-string (type {type_name})"))
}

/// `floatobject.py W_FloatObject.descr_repr` — the shortest decimal string
/// that round-trips to `val` (lowercase `nan`/`inf`, signed two-digit
/// exponents, `.0` on integral values). Delegates to the shortest-repr
/// formatter in `rustpython_literal::float`.
pub fn format_float_repr(val: f64) -> String {
    rustpython_literal::float::to_string(val)
}

/// RPython `rfloat.formatd` residual ABI for the generated JIT.
///
/// Upstream returns one low-level `STR` GC pointer (`rstr.py` `rpy_string`).
/// The host formatter returns a three-word Rust `String`, so the source
/// translator retargets only that formatter call to this wrapper, which
/// copies the bytes into a fresh `mallocstr` payload.
#[majit_macros::dont_look_inside]
pub fn jit_format_float_repr_rstr(val: f64) -> *mut pyre_object::unicodeobject::Utf8Str {
    let text = rustpython_literal::float::to_string(val);
    pyre_object::unicodeobject::alloc_utf8_payload(text.as_bytes(), true)
}

/// `unicodedb.isprintable` in PyPy's `rutf8.make_utf8_escape_function` is a
/// pure generated-table lookup.  Keep the same scalar call boundary around
/// pyre's Unicode database provider, including the upstream guarantee that a
/// lone surrogate is not printable.
#[majit_macros::elidable_cannot_raise]
fn repr_codepoint_is_printable(code: u32) -> bool {
    char::from_u32(code).is_some_and(rustpython_unicode::classify::is_printable)
}

/// `rutf8.make_utf8_escape_function(pass_printable=True, quotes=True)` —
/// choose the outer quote, then append printable code points or lowercase
/// `\x` / `\u` / `\U` escapes to a StringBuilder.  PyPy marks the generated
/// `_repr_function` `@jit.elidable`; preserve that call policy and keep the
/// loop in interpreter source so source translation sees its real semantics.
#[majit_macros::elidable]
pub fn format_wtf8_repr(s: &Wtf8) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let quote = if s.as_bytes().contains(&b'\'') && !s.as_bytes().contains(&b'"') {
        b'"'
    } else {
        b'\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote as char);
    let mut pos = 0;
    while pos < s.len() {
        let code = pyre_object::rutf8::codepoint_at_pos(s, pos) as u32;
        if code == quote as u32 || code == b'\\' as u32 {
            out.push('\\');
            out.push(code as u8 as char);
        } else if code == b'\t' as u32 {
            out.push_str("\\t");
        } else if code == b'\n' as u32 {
            out.push_str("\\n");
        } else if code == b'\r' as u32 {
            out.push_str("\\r");
        } else if repr_codepoint_is_printable(code) {
            // A printable code point is necessarily a Unicode scalar (the
            // helper rejects surrogates), so the conversion cannot fail.
            out.push(char::from_u32(code).expect("printable code point is a scalar"));
        } else {
            let digits = if code >= 0x10000 {
                out.push_str("\\U");
                8
            } else if code >= 0x100 {
                out.push_str("\\u");
                4
            } else {
                out.push_str("\\x");
                2
            };
            let mut shift = digits * 4;
            while shift != 0 {
                shift -= 4;
                out.push(HEX[((code >> shift) & 0x0f) as usize] as char);
            }
        }
        pos = pyre_object::rutf8::next_codepoint_pos(s, pos);
    }
    out.push(quote as char);
    out
}

/// `bytearrayobject.py W_BytearrayObject.descr_repr` — `bytearray(b'...')`.
/// The outer quote prefers `'`, flipping to `"` only when the data holds a
/// `'` but no `"`; the body always backslash-escapes `'` and `\` and never
/// escapes `"`, so a `'` survives as `\'` even under a `"` outer quote.
pub(crate) fn bytearray_repr_string(data: &[u8], class_name: &str) -> String {
    let has_single = data.contains(&b'\'');
    let has_double = data.contains(&b'"');
    let quote = if has_single && !has_double {
        b'"'
    } else {
        b'\''
    };
    let mut out = String::with_capacity(data.len() + 14);
    out.push_str(class_name);
    out.push_str("(b");
    out.push(quote as char);
    for &c in data {
        match c {
            b'\'' | b'\\' => {
                out.push('\\');
                out.push(c as char);
            }
            b'\t' => out.push_str("\\t"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            0x20..=0x7e => out.push(c as char),
            _ => out.push_str(&format!("\\x{c:02x}")),
        }
    }
    out.push(quote as char);
    out.push(')');
    out
}

/// `bytesobject.py::string_escape_encode(data, quotes=True)` — bytes repr.
/// PyPy chooses `"` only when the input contains `'` and not `"`, escapes
/// the chosen quote and backslash, names tab/newline/carriage-return, and
/// renders every other non-printable byte through the constant lowercase hex
/// table.  Keep this loop in interpreter source so translation sees the same
/// StringBuilder/character operations as PyPy instead of an opaque external
/// Rust formatting iterator.
#[majit_macros::elidable]
pub(crate) fn bytes_repr_string(data: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let quote = if data.contains(&b'\'') && !data.contains(&b'"') {
        b'"'
    } else {
        b'\''
    };
    let mut out = String::with_capacity(data.len() + 2);
    out.push('b');
    out.push(quote as char);
    for &c in data {
        if c == b'\\' || c == quote {
            out.push('\\');
            out.push(c as char);
        } else if c == b'\t' {
            out.push_str("\\t");
        } else if c == b'\r' {
            out.push_str("\\r");
        } else if c == b'\n' {
            out.push_str("\\n");
        } else if !(0x20..0x7f).contains(&c) {
            out.push_str("\\x");
            out.push(HEX[(c >> 4) as usize] as char);
            out.push(HEX[(c & 0x0f) as usize] as char);
        } else {
            out.push(c as char);
        }
    }
    out.push(quote as char);
    out
}

fn repr_active() -> Option<&'static std::cell::RefCell<Vec<PyObjectRef>>> {
    let ec = crate::call::ensure_executioncontext();
    if ec.is_null() {
        return None;
    }
    Some(unsafe { &(*ec).repr_active })
}

/// Record `obj` as mid-repr on this execution context, or report `false`
/// when it already is (`Py_ReprEnter`).  The set lives on the EC
/// (`objspace.py get_objects_in_repr` / `cpyext Py_ReprEnter`).
#[majit_macros::dont_look_inside]
pub(crate) fn repr_enter(obj: PyObjectRef) -> bool {
    let Some(active) = repr_active() else {
        return false;
    };
    let mut active = active.borrow_mut();
    if active.contains(&obj) {
        false
    } else {
        active.push(obj);
        true
    }
}

/// Length of the mid-repr set. The `Ref` deref stays in this residual.
/// `0` is the absent-context answer; a recorded object makes the length
/// at least one, so the caller can subtract without matching `Option`.
#[majit_macros::dont_look_inside]
fn repr_active_len() -> usize {
    match repr_active() {
        Some(active) => active.borrow().len(),
        None => 0,
    }
}

/// Drop `obj` from the mid-repr set (`Py_ReprLeave`) — see [`repr_enter`].
#[majit_macros::dont_look_inside]
pub(crate) fn repr_leave(obj: PyObjectRef) {
    let Some(active) = repr_active() else {
        return;
    };
    let mut active = active.borrow_mut();
    if let Some(pos) = active.iter().rposition(|&entry| entry == obj) {
        active.remove(pos);
    }
}

/// Drop the entry `repr_enter` pushed at `index` — see [`repr_leave`].
///
/// The guard leaves by position rather than by value because its own copy of
/// the object is a Rust local the collector does not update, so matching on it
/// would miss a forwarded entry and leave the set holding a finished repr.
fn repr_leave_at(index: usize) {
    let Some(active) = repr_active() else {
        return;
    };
    let mut active = active.borrow_mut();
    if index < active.len() {
        active.remove(index);
    }
}

/// RAII cycle guard.  `enter` returns `None` when `obj` is already being
/// repr'd on this thread — the caller emits the `...` placeholder — and
/// otherwise records `obj`, removing it again when the guard drops.
pub struct ReprGuard(Option<usize>);

impl ReprGuard {
    pub fn enter(obj: PyObjectRef) -> Option<ReprGuard> {
        if !repr_enter(obj) {
            return None;
        }
        let len = repr_active_len();
        Some(ReprGuard(Some(len.wrapping_sub(1))))
    }
}

impl Drop for ReprGuard {
    fn drop(&mut self) {
        if let Some(index) = self.0 {
            repr_leave_at(index);
        }
    }
}

/// `dictmultiobject.py descr_repr` — `{k: v, ...}`.  Iterates
/// `w_dict_items` (which routes through `is_module_dict`), guarded against
/// self-recursion.  Shared by the `py_repr` dict fast path and the dict
/// type's `__repr__` method (so dict-subclass instances and `super().
/// __repr__()` format their backing the same way).
///
/// # Safety
/// `obj` must be a real `W_DictObject` (caller resolves any subclass
/// backing via `resolve_dict_backing` first).
pub unsafe fn dict_repr(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    let Some(_guard) = ReprGuard::enter(obj) else {
        return Ok(Wtf8Buf::from_string("{...}".to_string()));
    };
    let entries = pyre_object::w_dict_items(obj);
    // The pairs live in a native Vec the collector does not walk, and every
    // key's and value's `__repr__` runs Python.  Pin them and read each one
    // back at its use, the way `list_repr` re-reads its container: a copy
    // taken before a collection addresses the pre-move object.
    let _roots = pyre_object::gc_roots::push_roots();
    let mut flat: Vec<PyObjectRef> = Vec::with_capacity(entries.len() * 2);
    for &(k, v) in entries.as_slice() {
        flat.push(k);
        flat.push(v);
    }
    let pair_base = pyre_object::gc_roots::pin_roots(&flat);
    let mut out = Wtf8Buf::new();
    out.push_str("{");
    for i in 0..entries.len() {
        // `dictmultiobject.py:388` joins the pairs by position, so a key or
        // value whose `__repr__` answers `""` still gets its separator.
        if i != 0 {
            out.push_str(", ");
        }
        let key = pyre_object::gc_roots::shadow_stack_get(pair_base + i * 2);
        out.push_wtf8(&py_repr_wtf8(key)?);
        out.push_str(": ");
        let value = pyre_object::gc_roots::shadow_stack_get(pair_base + i * 2 + 1);
        out.push_wtf8(&py_repr_wtf8(value)?);
    }
    out.push_str("}");
    Ok(out)
}

/// `frozendict_repr`: empty is `Name()`, otherwise `Name({...})` wrapping
/// the dict-body repr. `Name` is `Py_TYPE(self)->tp_name`.
///
/// # Safety
/// `obj` must be a live exact `frozendict` (`W_DictObject` payload).
pub unsafe fn frozendict_repr(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    let name = if let Some(w_type) = crate::typedef::r#type(obj) {
        pyre_object::w_type_get_name(w_type.as_ptr())
    } else {
        "frozendict"
    };
    if pyre_object::w_dict_len(obj) == 0 {
        let mut out = Wtf8Buf::new();
        out.push_str(name);
        out.push_str("()");
        return Ok(out);
    }
    let body = dict_repr(obj)?;
    let mut out = Wtf8Buf::new();
    out.push_str(name);
    out.push_str("(");
    out.push_wtf8(&body);
    out.push_str(")");
    Ok(out)
}

/// `listobject.py W_ListObject.descr_repr`, factored out so the TypeDef slot
/// and the native `py_repr` fast path use the same recursion guard and item
/// walk. Calling the base descriptor on a subclass must not redispatch its
/// overriding `__repr__`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn list_repr(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    let Some(_guard) = ReprGuard::enter(obj) else {
        return Ok(Wtf8Buf::from_string("[...]".to_string()));
    };
    // Each item's `__repr__` runs Python, so the list itself moves under the
    // walk; re-read it from the shadow stack before every element fetch.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
    let n = pyre_object::w_list_len(obj());
    let mut out = Wtf8Buf::new();
    out.push_str("[");
    for i in 0..n {
        // `listobject.py:217-221` stops rather than skipping when an item is
        // gone, since an item's `__repr__` may have shortened the list.
        let Some(item) = pyre_object::w_list_getitem(obj(), i as i64) else {
            break;
        };
        // The separator goes by position, not by how much has been written:
        // an item whose `__repr__` answers `""` still takes a slot.
        if i != 0 {
            out.push_str(", ");
        }
        out.push_wtf8(&py_repr_wtf8(item)?);
    }
    out.push_str("]");
    Ok(out)
}

/// `tupleobject.py W_AbstractTupleObject.descr_repr`. This is the base slot
/// body, so it formats tuple storage directly even when invoked through
/// `super().__repr__()` on a subtype.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn tuple_repr(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    let Some(_guard) = ReprGuard::enter(obj) else {
        return Ok(Wtf8Buf::from_string("(...)".to_string()));
    };
    // Same as `list_repr`: an item's `__repr__` moves the tuple under the walk.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
    let n = pyre_object::w_tuple_len(obj());
    let mut out = Wtf8Buf::new();
    out.push_str("(");
    for i in 0..n {
        if let Some(item) = pyre_object::w_tuple_getitem(obj(), i as i64) {
            // `tupleobject.py:114` joins by position, so an item whose
            // `__repr__` answers `""` still gets its separator.
            if i != 0 {
                out.push_str(", ");
            }
            out.push_wtf8(&py_repr_wtf8(item)?);
        }
    }
    if n == 1 {
        out.push_str(",");
    }
    out.push_str(")");
    Ok(out)
}

/// Format a PyObjectRef for debug display.
///
/// # Safety
/// `obj` must be a valid pointer to a known Python object type.
/// Format an `int`/`long`/`float`/`bool` storage object with its builtin
/// `repr` (which equals its `str` for these types).  Returns `None` for
/// any other storage type.  Shared by `py_repr`'s leaf path and `py_str`'s
/// fallback so a builtin leaf subclass that overrides only `__repr__`
/// still `str()`s via the inherited builtin `tp_str`.
unsafe fn builtin_leaf_repr_string(
    obj: PyObjectRef,
    tp: *const PyType,
) -> Result<Option<String>, crate::PyError> {
    unsafe {
        Ok(if std::ptr::eq(tp, &INT_TYPE as *const PyType) {
            // A machine int is at most 19 digits, below the 640 floor
            // `sys.set_int_max_str_digits` accepts, so it never trips the
            // conversion-length limit the `long` arm has to check.
            Some(format!("{}", pyre_object::intobject::w_int_get_value(obj)))
        } else if std::ptr::eq(tp, &FLOAT_TYPE as *const PyType) {
            let float_obj = obj as *const pyre_object::floatobject::W_FloatObject;
            Some(format_float_repr((*float_obj).floatval))
        } else if std::ptr::eq(tp, &pyre_object::COMPLEX_TYPE as *const PyType) {
            Some(crate::typedef::complex_repr_string(
                pyre_object::w_complex_get_real(obj),
                pyre_object::w_complex_get_imag(obj),
            ))
        } else if std::ptr::eq(tp, &LONG_TYPE as *const PyType) {
            // `long_to_decimal_string` enforces the
            // `sys.set_int_max_str_digits` conversion-length limit, so the
            // guard belongs to the conversion itself rather than to the
            // `__repr__` descriptor that is only one of its callers.
            Some(crate::builtins::int_to_decimal_string(obj)?)
        } else if std::ptr::eq(tp, &BOOL_TYPE as *const PyType) {
            let bool_obj = obj as *const pyre_object::boolobject::W_BoolObject;
            Some(
                if (*bool_obj).intval != 0 {
                    "True"
                } else {
                    "False"
                }
                .to_string(),
            )
        } else {
            None
        })
    }
}

/// Dispatch a user-defined `__repr__`/`__str__` override for a builtin leaf
/// subclass instance.  `int`/`float`/`str`/... user subclasses carry a
/// `_getusercls` typeptr. Callers pass `layout_base` of that typeptr, so
/// the formatters below see the builtin storage type and would ignore a
/// subclass override.  Returns `Some`
/// only when the dunder resolves above `object` (whose inherited default
/// must fall through to the builtin formatting instead of re-entering).
/// `builtin_subclass_dunder` returning the raw `str` result object so a
/// WTF-8-preserving caller (`py_str_wtf8`) can read a lone-surrogate result
/// via `w_str_get_wtf8` instead of the panicking `w_str_get_value`.
pub(crate) unsafe fn builtin_subclass_dunder_obj(
    obj: PyObjectRef,
    tp: *const PyType,
    name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    unsafe {
        let tp = pyre_object::pyobject::layout_base(tp);
        let is_leaf = std::ptr::eq(tp, &INT_TYPE as *const PyType)
            || std::ptr::eq(tp, &LONG_TYPE as *const PyType)
            || std::ptr::eq(tp, &FLOAT_TYPE as *const PyType)
            || std::ptr::eq(tp, &pyre_object::COMPLEX_TYPE as *const PyType)
            || std::ptr::eq(tp, &BOOL_TYPE as *const PyType)
            || std::ptr::eq(tp, &STR_TYPE as *const PyType)
            || std::ptr::eq(tp, &pyre_object::LIST_TYPE as *const PyType)
            || pyre_object::is_tuple(obj)
            || pyre_object::is_any_dict(obj)
            || pyre_object::is_set_or_frozenset(obj)
            || std::ptr::eq(tp, &pyre_object::interp_array::ARRAY_TYPE as *const PyType)
            || std::ptr::eq(
                tp,
                &pyre_object::bytearrayobject::BYTEARRAY_TYPE as *const PyType,
            );
        if !is_leaf {
            return Ok(None);
        }
        // An exact builtin is not a subclass instance: its `w_class` is the
        // canonical type object, so the MRO walk below would resolve the
        // builtin's own dunder and re-dispatch it through a full call rather
        // than the native formatting this function exists to defer to.
        if pyre_object::is_exact_builtin_instance(obj) {
            return Ok(None);
        }
        let w_class = (*obj).w_class;
        if w_class.is_null() || !pyre_object::is_type(w_class) {
            return Ok(None);
        }
        // Only a subclass can redirect the dunder: it keeps the builtin
        // `ob_type` and retags `w_class` (`typedef::subclass_to_tag`), which
        // is exactly what `is_exact_builtin_instance` tests. An exact
        // instance resolves the dunder to the builtin the caller is about to
        // run natively, and the builtin types are immutable, so the two MRO
        // walks and the descriptor call below can only reproduce it.
        //
        // `long` is the one leaf where the descriptor does more than the leaf
        // formatter: `longobject.py descr_repr` also enforces
        // `sys.set_int_max_str_digits`, and that check sits in the descriptor
        // rather than in the conversion, so an exact `long` keeps going
        // through it. A machine `int` cannot reach any settable limit — 19
        // digits against a floor of 640.
        if !std::ptr::eq(tp, &LONG_TYPE as *const PyType)
            && pyre_object::is_exact_builtin_instance(obj)
        {
            return Ok(None);
        }
        // Published across the allocating MRO walk, as in `try_call_dunder_obj`.
        let _roots = pyre_object::gc_roots::push_roots();
        let root_base = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        let obj = || pyre_object::gc_roots::shadow_stack_get(root_base);
        let Some((src, found)) = crate::baseobjspace::lookup_where_with_method_cache(w_class, name)
        else {
            return Ok(None);
        };
        // `object`'s inherited default is not a leaf override — fall through
        // so the builtin formatting runs (and `object.__repr__` does not
        // re-enter through this path). An explicit
        // `Subclass.__str__ = object.__str__` is different: its owner is the
        // subclass and the descriptor must run, allowing object.__str__ to
        // delegate to the subclass's __repr__.
        let w_object = crate::typedef::w_object();
        if std::ptr::eq(src, w_object) {
            return Ok(None);
        }
        let _ = pyre_object::gc_roots::pin_root(found);
        let found = || pyre_object::gc_roots::shadow_stack_get(root_base + 1);
        // A raising override propagates; a non-string return is a TypeError.
        let r = crate::builtins::call_and_check(found(), &[obj()])?;
        if pyre_object::is_str(r) {
            return Ok(Some(r));
        }
        Err(dunder_returned_non_string(name, r))
    }
}

/// `setobject.py W_BaseSetObject.descr_repr` / `setrepr`.
///
/// This is the native descriptor body, separate from `space.repr`'s special
/// method dispatch.  In particular, `set.__repr__(subclass_instance)` must
/// format the backing set instead of redispatching the subclass override that
/// called it.  The copied item vector is rooted because each recursive repr
/// can collect and move every item still waiting in it.
pub(crate) unsafe fn set_repr_wtf8(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        let current_obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);

        let is_frozen = pyre_object::is_frozenset(current_obj());
        let is_exact_set =
            pyre_object::is_exact_type(current_obj(), &pyre_object::setobject::SET_TYPE);
        let class_name = crate::typedef::r#type(current_obj())
            .map(|w_type| pyre_object::w_type_get_name(w_type.as_ptr()).to_string())
            .unwrap_or_else(|| {
                if is_frozen {
                    "frozenset".to_string()
                } else {
                    "set".to_string()
                }
            });
        let Some(_guard) = ReprGuard::enter(current_obj()) else {
            return Ok(Wtf8Buf::from_string(format!("{class_name}(...)")));
        };
        let items = pyre_object::w_set_items(current_obj());
        let item_base = pyre_object::gc_roots::pin_roots(&items);
        let mut out = Wtf8Buf::new();
        if items.is_empty() {
            out.push_str(&class_name);
            out.push_str("()");
            return Ok(out);
        }
        if !is_exact_set {
            out.push_str(&class_name);
            out.push_str("(");
        }
        out.push_str("{");
        for index in 0..items.len() {
            if index != 0 {
                out.push_str(", ");
            }
            let item = pyre_object::gc_roots::shadow_stack_get(item_base + index);
            out.push_wtf8(&py_repr_wtf8(item)?);
        }
        out.push_str("}");
        if !is_exact_set {
            out.push_str(")");
        }
        Ok(out)
    }
}

/// Resolve a class object's special method on its metaclass.
///
/// PyPy: `space.lookup(w_obj, name)` uses `space.type(w_obj)`, so a class
/// receives an EnumType/user-metaclass override before the native `type`
/// implementation. The builtin `type`/`object` definitions are terminals and
/// deliberately left to the native formatting path to avoid re-entry.
pub(crate) unsafe fn type_metaclass_dunder_obj(
    obj: PyObjectRef,
    name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    unsafe {
        if !pyre_object::is_type(obj) {
            return Ok(None);
        }
        let Some(metaclass) = crate::typedef::r#type(obj) else {
            return Ok(None);
        };
        if !pyre_object::is_type(metaclass.as_ptr()) {
            return Ok(None);
        }
        // Published across the allocating MRO walk, as in `try_call_dunder_obj`.
        let _roots = pyre_object::gc_roots::push_roots();
        let root_base = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        let obj = || pyre_object::gc_roots::shadow_stack_get(root_base);
        let Some((src, method)) =
            crate::baseobjspace::lookup_where_with_method_cache(metaclass.as_ptr(), name)
        else {
            return Ok(None);
        };
        if std::ptr::eq(src, crate::typedef::w_type())
            || std::ptr::eq(src, crate::typedef::w_object())
        {
            return Ok(None);
        }
        let _ = pyre_object::gc_roots::pin_root(method);
        let method = || pyre_object::gc_roots::shadow_stack_get(root_base + 1);
        let result = crate::builtins::call_and_check(method(), &[obj()])?;
        if pyre_object::is_str(result) {
            return Ok(Some(result));
        }
        Err(dunder_returned_non_string(name, result))
    }
}

/// Dispatch a user-defined `__str__` / `__repr__` override on an
/// exception subclass.  The builtin `descr_str` / `descr_repr` are
/// handled natively in `py_str` / `py_repr`, but a Python subclass
/// (`class E(Exception): def __str__(self): ...`) installs its own
/// method that must win, the same way `str(e)` dispatches it in PyPy.
/// Returns `None` only when `__str__`/`__repr__` resolves to the builtin
/// `BaseException` / `object` registration (no override). A raising override
/// propagates, and a non-string result raises `TypeError`.
/// `exc_user_dunder` variant returning the raw `str` result object so a
/// WTF-8-preserving caller (`exception_descr_str_wtf8`) can read the
/// lone-surrogate-carrying bytes directly. A raising override propagates and
/// a non-string result raises `TypeError`, matching descroperation.py's
/// `space.str` / `space.repr` result check.
pub(crate) unsafe fn exc_user_dunder_obj(
    obj: PyObjectRef,
    name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    unsafe {
        let w_class = (*obj).w_class;
        if w_class.is_null() || !pyre_object::is_type(w_class) {
            return Ok(None);
        }
        // Published across the allocating MRO walk, as in `try_call_dunder_obj`.
        let _roots = pyre_object::gc_roots::push_roots();
        let root_base = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        let obj = || pyre_object::gc_roots::shadow_stack_get(root_base);
        let Some((src, method)) =
            crate::baseobjspace::lookup_where_with_method_cache(w_class, name)
        else {
            return Ok(None);
        };
        // `object`'s and `BaseException`'s registrations are the two the
        // native formatting stands in for, and so are the `descr_str`
        // builtins the exception classes install on top of them — calling
        // any of those back from here would recurse.  A builtin that the
        // native path does *not* implement, such as
        // `BaseExceptionGroup.__str__`, still has to be dispatched.
        if method.is_null() || std::ptr::eq(src, crate::typedef::w_object()) {
            return Ok(None);
        }
        if crate::builtins::is_native_exception_dunder(method) {
            return Ok(None);
        }
        if let Some(base) = crate::builtins::lookup_exc_class("BaseException")
            && std::ptr::eq(src, base)
        {
            return Ok(None);
        }
        let _ = pyre_object::gc_roots::pin_root(method);
        let method = || pyre_object::gc_roots::shadow_stack_get(root_base + 1);
        let r = crate::builtins::call_and_check(method(), &[obj()])?;
        if pyre_object::is_str(r) {
            return Ok(Some(r));
        }
        Err(dunder_returned_non_string(name, r))
    }
}

/// Dispatch a user-defined `__repr__` / `__str__` override on a
/// `types.ModuleType` subclass. `layout_base` of `MODULE_USER_TYPE` is
/// `MODULE_TYPE`, so the formatting in `py_repr` / `py_str` reaches this
/// helper for a subclass as well as an exact module. The subclass carries
/// its Python class in `w_class`. Returns `None` when the method resolves
/// to the base `module` registration or `object` (no override); a raising
/// override propagates and a non-string result raises `TypeError`.
unsafe fn module_user_dunder_obj(
    obj: PyObjectRef,
    name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    unsafe {
        let w_class = (*obj).w_class;
        if w_class.is_null() || !pyre_object::is_type(w_class) {
            return Ok(None);
        }
        let module_class = crate::typedef::gettypeobject(&MODULE_TYPE);
        if std::ptr::eq(w_class, module_class) {
            return Ok(None);
        }
        // Published across the allocating MRO walk, as in `try_call_dunder_obj`.
        let _roots = pyre_object::gc_roots::push_roots();
        let root_base = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        let obj = || pyre_object::gc_roots::shadow_stack_get(root_base);
        let Some((src, method)) =
            crate::baseobjspace::lookup_where_with_method_cache(w_class, name)
        else {
            return Ok(None);
        };
        if method.is_null()
            || std::ptr::eq(src, module_class)
            || std::ptr::eq(src, crate::typedef::w_object())
        {
            return Ok(None);
        }
        let _ = pyre_object::gc_roots::pin_root(method);
        let method = || pyre_object::gc_roots::shadow_stack_get(root_base + 1);
        let r = crate::builtins::call_and_check(method(), &[obj()])?;
        if pyre_object::is_str(r) {
            return Ok(Some(r));
        }
        Err(dunder_returned_non_string(name, r))
    }
}

/// `space.repr` — the whole type dispatch, answering the encoded bytes.
///
/// `listobject.py _listrepr_inner` assembles a container's repr in a
/// `rutf8.Utf8StringBuilder` from each item's `space.utf8_len_w(space.repr(...))`,
/// so a lone surrogate an item wrote survives being nested. A `Wtf8Buf` is the
/// buffer that can hold the same thing here; a Rust `String` cannot, so every
/// caller reads the WTF-8 rather than a `String` round trip of it.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
/// The array module installs one function pointer. Calling it here
/// is a residual: the pointer is an integer at the annotation layer
/// and has no pre-rtyper callable shape.
#[majit_macros::dont_look_inside]
#[inline(never)]
unsafe fn array_repr_hook(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    let hooks = crate::importing::optional_module_hooks().expect("array repr hook");
    (hooks.array_repr_wtf8)(obj)
}

pub unsafe fn py_repr_wtf8(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    // A tagged immediate must be formatted before `ob_type` touches it as a
    // pointer; `repr` of a plain `int` is its
    // decimal value. Gated on `CAN_BE_TAGGED` (default false).
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return Ok(Wtf8Buf::from_string(format!(
            "{}",
            pyre_object::tagged_int::untag_int(obj)
        )));
    }
    // The recursive container branches below (dict/list/tuple/set/deque/
    // slice/range) re-enter `py_repr_wtf8` on each element in native Rust with
    // no Python frame push, so a deeply nested structure blows the C stack
    // before any frame-level check fires. Guard the stack here so
    // `repr(deeply_nested)` raises RecursionError instead of overflowing.
    crate::stack_check::stack_check()?;
    if obj.is_null() {
        return Ok(Wtf8Buf::from_string("NULL".to_string()));
    }
    unsafe {
        // Each dispatch below resolves a dunder through an MRO walk and may
        // run a Python override; both allocate, so the object this function
        // goes on to format natively has to be re-read from the shadow stack
        // rather than carried in a native local across them.  `ob_type` names
        // a static type, so it survives the moves the object itself makes.
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        let obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
        // The formatting below is keyed on the payload layout, which a
        // `_getusercls` class shares with the builtin it was made from.
        let tp = pyre_object::pyobject::layout_base((*obj()).ob_type);
        // A builtin leaf subclass keeps `ob_type` at the canonical storage
        // type but carries the Python class in `w_class`; dispatch its
        // `__repr__` override before the `ob_type`-keyed formatting below.
        if let Some(r) = builtin_subclass_dunder_obj(obj(), tp, "__repr__")? {
            return Ok(pyre_object::w_str_get_wtf8(r).to_wtf8_buf());
        }
        // A class object is an instance of its metaclass.  PyPy's
        // `space.repr` therefore resolves `__repr__` on that metaclass before
        // `type`'s native `<class ...>` representation.  This is observable
        // for EnumType and any user metaclass defining `__repr__`.
        if let Some(result) = type_metaclass_dunder_obj(obj(), "__repr__")? {
            return Ok(pyre_object::w_str_get_wtf8(result).to_wtf8_buf());
        }
        let formatted = if let Some(s) = builtin_leaf_repr_string(obj(), tp)? {
            s
        } else if pyre_object::interp_array::is_array(obj())
            && crate::importing::optional_module_hooks().is_some()
        {
            return array_repr_hook(obj());
        } else if std::ptr::eq(tp, &pyre_object::pyobject::LIST_TYPE as *const PyType) {
            return list_repr(obj());
        } else if pyre_object::is_tuple(obj()) {
            // `pyre_object::is_tuple` covers `TUPLE_TYPE` plus the
            // arity-2 specialisations (`SPECIALISED_TUPLE_{II,FF,OO}_TYPE`,
            // `pypy/objspace/std/specialisedtupleobject.py makespecialisedtuple`).
            // Without this union dispatch the specialised variants
            // (returned by `w_tuple_new(items)` whenever `items.len() == 2`)
            // would fall through to the generic `<{name} object at ...>`
            // fallback — visible as `<tuple object at 0x...>` on
            // `print(e.args)` for two-arg exception constructors.
            //
            // structseq instances (`_structseq.py structseqtype`)
            // are tuple subclasses with `w_class` pointing at a custom
            // type that installs its own `__repr__`.  Route them
            // through the subclass dunder before the generic tuple
            // formatting so `repr(pwd_entry)` prints
            // `'pwd.struct_passwd(pw_name=..., ...)'` instead of the
            // bare tuple form.  Plain `tuple()` keeps the fast path
            // because its `w_class` is the canonical tuple type.
            let w_class = (*obj()).w_class;
            let tuple_class = crate::typedef::gettypeobject(&pyre_object::pyobject::TUPLE_TYPE);
            if !w_class.is_null() && !std::ptr::eq(w_class, tuple_class) {
                // structseq instances are tuple subclasses with ob_type ==
                // TUPLE_TYPE, so reach for a subclass __repr__ via the MRO.
                // `tuple` itself installs no `__repr__` dict entry (it is
                // handled natively below), so a plain tuple subclass
                // resolves `__repr__` to `object` — fall through to the
                // tuple formatting in that case rather than printing the
                // generic `<object at ...>`.
                if let Some((src, method)) =
                    crate::baseobjspace::lookup_where_with_method_cache(w_class, "__repr__")
                    && !std::ptr::eq(src, crate::typedef::w_object())
                    && !method.is_null()
                {
                    // The walk above allocates: re-read the receiver so the
                    // descriptor is handed the object at its current home.
                    // A raising override propagates; a non-string return is
                    // a TypeError like every other `__repr__` override.
                    let r = crate::builtins::call_and_check(method, &[obj()])?;
                    if pyre_object::is_str(r) {
                        return Ok(pyre_object::w_str_get_wtf8(r).to_wtf8_buf());
                    }
                    return Err(dunder_returned_non_string("__repr__", r));
                }
            }
            return tuple_repr(obj());
        } else if unsafe { pyre_object::is_frozendict(obj()) } {
            return unsafe { frozendict_repr(obj()) };
        } else if unsafe { pyre_object::is_dict(obj()) } {
            return unsafe { dict_repr(obj()) };
        } else if pyre_object::sliceobject::is_slice(obj()) {
            // `pypy/objspace/std/sliceobject.py descr_repr` —
            // `slice(%r, %r, %r)`.
            let mut out = Wtf8Buf::new();
            out.push_str("slice(");
            out.push_wtf8(&py_repr_wtf8(pyre_object::sliceobject::w_slice_get_start(
                obj(),
            ))?);
            out.push_str(", ");
            out.push_wtf8(&py_repr_wtf8(pyre_object::sliceobject::w_slice_get_stop(
                obj(),
            ))?);
            out.push_str(", ");
            out.push_wtf8(&py_repr_wtf8(pyre_object::sliceobject::w_slice_get_step(
                obj(),
            ))?);
            out.push_str(")");
            return Ok(out);
        } else if pyre_object::is_bytes_like(obj()) {
            // `bytesobject.py W_BytesObject.descr_repr` — ASCII-printable
            // bytes pass through, control/high bytes use `\xNN`, and the
            // outer quote prefers `'`, flipping to `"` when the data holds a
            // `'` but no `"`.
            let data = pyre_object::bytes_like_data(obj()).to_vec();
            if pyre_object::bytearrayobject::is_bytearray(obj()) {
                // `bytearrayobject.py W_BytearrayObject.descr_repr` differs
                // from the bytes form: it chooses the same outer quote but
                // always backslash-escapes an inner `'` (never `"`), so the
                // shared bytes escaper cannot express it.
                bytearray_repr_string(&data, "bytearray")
            } else {
                bytes_repr_string(&data)
            }
        } else if pyre_object::is_set_or_frozenset(obj()) {
            return set_repr_wtf8(obj());
        } else if std::ptr::eq(tp, &STR_TYPE as *const PyType) {
            format_wtf8_repr(&pyre_object::w_str_get_wtf8(obj()).to_wtf8_buf())
        } else if std::ptr::eq(tp, &NONE_TYPE as *const PyType) {
            "None".to_string()
        } else if std::ptr::eq(
            tp,
            &pyre_object::pyobject::NOTIMPLEMENTED_TYPE as *const PyType,
        ) {
            "NotImplemented".to_string()
        } else if std::ptr::eq(tp, &ELLIPSIS_TYPE as *const PyType) {
            "Ellipsis".to_string()
        } else if std::ptr::eq(tp, &BUILTIN_CODE_TYPE as *const PyType) {
            // Raw BuiltinCode objects (Code-level, not normally user-visible)
            let name = builtin_code_name(obj());
            format!("<code {name}>")
        } else if std::ptr::eq(tp, &crate::function::SLOT_WRAPPER_TYPE as *const PyType) {
            let name = function_get_name(obj()).to_string();
            let owner = crate::function::fget_func_objclass(obj())?;
            let owner_name = pyre_object::w_type_get_name(owner);
            format!("<slot wrapper '{name}' of '{owner_name}' objects>")
        } else if std::ptr::eq(
            tp,
            &crate::function::METHOD_DESCRIPTOR_TYPE as *const PyType,
        ) {
            let name = function_get_name(obj()).to_string();
            let owner = crate::function::fget_func_objclass(obj())?;
            let owner_name = pyre_object::w_type_get_name(owner);
            format!("<method '{name}' of '{owner_name}' objects>")
        } else if std::ptr::eq(tp, &BUILTIN_FUNCTION_TYPE as *const PyType) {
            // function.py BuiltinFunction.descr_function_repr.  Same text
            // the `__repr__` this type registers in `typedef.rs` produces;
            // this native arm is the one `repr()` actually reaches.
            let name = function_get_name(obj()).to_string();
            let w_self = crate::function::function_get_self_or_none(obj());
            crate::function::builtin_function_repr_text(&name, w_self)
        } else if std::ptr::eq(tp, &FUNCTION_TYPE as *const PyType) {
            // function.py Function.descr_function_repr —
            // `self.getrepr(space, 'function %s' % self.qualname)`, and
            // `baseobjspace.py getrepr` appends ` at 0x<addr>`.  Exact
            // builtin values take this fast path instead of dispatching
            // through the `__repr__` the type registers in `typedef.rs`, so it
            // must produce the same address-bearing text.
            // `format!` would render the WTF-8 qualname through `Display`,
            // which substitutes U+FFFD for a lone surrogate.
            let mut repr = Wtf8Buf::from_string("<function ".to_string());
            repr.push_wtf8(&function_get_qualname(obj()));
            // `function_get_qualname` can allocate the fallback string.
            repr.push_str(&format!(" at {}>", repr_gc_addr(obj())));
            return Ok(repr);
        } else if unsafe { pyre_object::is_exception(obj()) } {
            // A user subclass that overrides `__repr__` shadows the builtin
            // `W_BaseException.descr_repr`; dispatch it before the native
            // formatting below.
            if let Some(r) = exc_user_dunder_obj(obj(), "__repr__")? {
                return Ok(pyre_object::w_str_get_wtf8(r).to_wtf8_buf());
            }
            // `W_BaseException.descr_repr` reads `len(self.args_w)`.
            // Zero items format as `()`. One item is `repr(args_w[0])`
            // with no trailing comma. Several items are
            // `repr(space.newtuple(self.args_w))`, which separates with
            // `", "` and also has no trailing comma. `BaseException_repr`
            // uses `%R` on the stored args tuple for every length other
            // than one. The class name is `type(obj).__name__`.
            let class_name = if let Some(cls) = crate::typedef::r#type(obj()) {
                // `w_type_get_name_obj` is the accessor the `__name__` getter
                // reads, so the two answers cannot drift.  A class registered
                // as `"termios.error"` is named `error` in the module
                // `termios` — `new_exception_class` splits the dotted name
                // before it calls `type` — while `w_type_get_name` keeps the
                // undivided registration name, which spells the module twice
                // here.
                let w_name = unsafe { pyre_object::w_type_get_name_obj(cls.as_ptr()) };
                pyre_object::w_str_get_wtf8(w_name).to_wtf8_buf()
            } else {
                Wtf8Buf::from_string(
                    pyre_object::interp_exceptions::exc_kind_name(
                        pyre_object::w_exception_get_kind(obj()),
                    )
                    .to_string(),
                )
            };
            // The name object above is minted on its first read of a class, so
            // that read is a collection point and the receiver comes back off
            // the shadow stack.
            let stored =
                unsafe { pyre_object::interp_exceptions::w_exception_get_args_storage(obj()) };
            let len = if stored.is_null() {
                0
            } else {
                unsafe { pyre_object::interp_exceptions::rlist_len(stored) }
            };
            let mut inner = Wtf8Buf::new();
            if len == 1 {
                let first = unsafe { pyre_object::interp_exceptions::rlist_getitem(stored, 0) };
                let first_slot = pyre_object::gc_roots::pin_roots(&[first]);
                inner.push_wtf8(&py_repr_wtf8(pyre_object::gc_roots::shadow_stack_get(
                    first_slot,
                ))?);
            } else if len > 1 {
                // `descr_repr` calls `repr(space.newtuple(self.args_w))`.
                // The header is nursery. Pin it before an item `__repr__`
                // collects, and read each element back through the pin.
                let args_obj =
                    unsafe { pyre_object::interp_exceptions::w_exception_get_args(obj()) };
                if !args_obj.is_null() && pyre_object::is_tuple(args_obj) {
                    let _args_roots = pyre_object::gc_roots::push_roots();
                    let args_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = pyre_object::gc_roots::pin_root(args_obj);
                    let args_obj = || pyre_object::gc_roots::shadow_stack_get(args_slot);
                    let n = pyre_object::w_tuple_len(args_obj());
                    for i in 0..n {
                        if let Some(item) = pyre_object::w_tuple_getitem(args_obj(), i as i64) {
                            // `repr(tuple)` separates by position, so an
                            // argument whose `__repr__` answers `""` still
                            // takes a slot.
                            if i != 0 {
                                inner.push_str(", ");
                            }
                            inner.push_wtf8(&py_repr_wtf8(item)?);
                        }
                    }
                }
            }
            let mut out = Wtf8Buf::new();
            out.push_wtf8(&class_name);
            out.push_str("(");
            out.push_wtf8(&inner);
            out.push_str(")");
            return Ok(out);
        } else if std::ptr::eq(tp, &TYPE_TYPE as *const PyType) {
            let name = crate::baseobjspace::type_repr_qualified_name(obj());
            return Ok(wtf8_format!("<class '", name, "'>"));
        } else if std::ptr::eq(tp, &pyre_object::UNION_TYPE as *const PyType) {
            // PyPy: UnionType.__repr__ → " | ".join([_repr_item(x) for x in self.__args__])
            let args = pyre_object::w_union_get_args(obj());
            let n = pyre_object::w_tuple_len(args);
            // An argument's repr can move the args tuple; read each item back
            // from the slot.
            let _union_roots = pyre_object::gc_roots::push_roots();
            let args_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(args);
            let mut parts = Vec::with_capacity(n);
            for i in 0..n {
                if let Some(item) = pyre_object::w_tuple_getitem(
                    pyre_object::gc_roots::shadow_stack_get(args_slot),
                    i as i64,
                ) {
                    // `_repr_item_union` (`_pypy_generic_alias.py`) —
                    // `type(None)` renders as `None`; a bare `None` may
                    // still reach here from direct construction paths.
                    if pyre_object::is_none(item)
                        || std::ptr::eq(
                            item,
                            crate::typedef::gettypeobject(&pyre_object::NONE_TYPE),
                        )
                    {
                        parts.push(Wtf8Buf::from_string("None".to_string()));
                    } else {
                        parts.push(crate::_pypy_generic_alias::repr_item(item)?);
                    }
                }
            }
            let mut joined = Wtf8Buf::new();
            let mut first = true;
            for part in parts.as_slice() {
                if !first {
                    joined.push_str(" | ");
                }
                joined.push_wtf8(part);
                first = false;
            }
            return Ok(joined);
        } else if std::ptr::eq(tp, &pyre_object::GENERIC_ALIAS_TYPE as *const PyType) {
            // GenericAlias.__repr__ (`_pypy_generic_alias.py`).
            return crate::_pypy_generic_alias::repr(obj());
        } else if std::ptr::eq(tp, &MODULE_TYPE as *const PyType) {
            // A `types.ModuleType` subclass carries its class in `w_class`; a
            // subclass `__repr__` override wins over the native module
            // formatting.
            if let Some(r) = module_user_dunder_obj(obj(), "__repr__")? {
                return Ok(pyre_object::w_str_get_wtf8(r).to_wtf8_buf());
            } else {
                return crate::typedef::module_repr_string(obj());
            }
        } else if std::ptr::eq(
            tp,
            &pyre_object::pyobject::MAPPING_PROXY_TYPE as *const PyType,
        ) {
            // `pypy/objspace/std/dictproxyobject.py descr_repr` →
            // `b"mappingproxy(%s)" % space.utf8_w(space.repr(self.w_mapping))`.
            let inner = pyre_object::w_dict_proxy_get_mapping(obj());
            let mut out = Wtf8Buf::new();
            out.push_str("mappingproxy(");
            out.push_wtf8(&py_repr_wtf8(inner)?);
            out.push_str(")");
            return Ok(out);
        } else if pyre_object::typedef::is_getset_property(obj()) {
            // CPython 3.14 `PyGetSetDescr_Type.tp_repr`.
            crate::typedef::getset_descriptor_repr(obj())
        } else if pyre_object::is_member(obj()) {
            // CPython 3.14 `PyMemberDescr_Type.tp_repr = member_repr`.
            // Member descriptors are native-layout objects with no `w_class`,
            // so the generic builtin-dunder fallback below cannot discover
            // their registered __repr__ method.
            crate::typedef::member_descriptor_repr(obj())
        } else if std::ptr::eq(
            tp,
            &pyre_object::dictmultiobject::DICT_KEYS_TYPE as *const PyType,
        ) || std::ptr::eq(
            tp,
            &pyre_object::dictmultiobject::DICT_VALUES_TYPE as *const PyType,
        ) || std::ptr::eq(
            tp,
            &pyre_object::dictmultiobject::DICT_ITEMS_TYPE as *const PyType,
        ) {
            // `dictmultiobject.py viewrepr`: the view itself participates in
            // the shared identity recursion set and emits `...` on re-entry.
            // This is distinct from the owning dict's `{...}` placeholder: a
            // dict may contain one of its own values/items views.
            let Some(_guard) = ReprGuard::enter(obj()) else {
                return Ok(Wtf8Buf::from_string("...".to_string()));
            };
            let kind = pyre_object::dictmultiobject::w_dict_view_get_kind(obj());
            let label = match kind {
                pyre_object::dictmultiobject::DictViewKind::Keys => "dict_keys",
                pyre_object::dictmultiobject::DictViewKind::Values => "dict_values",
                pyre_object::dictmultiobject::DictViewKind::Items => "dict_items",
            };
            let snapshot = crate::type_methods::dict_view_snapshot(obj());
            // The snapshot is a native Vec the collector does not walk, and an
            // item's `__repr__` runs Python.  Pin it and read each element back
            // from the shadow stack.
            let _snapshot_roots = pyre_object::gc_roots::push_roots();
            let item_base = pyre_object::gc_roots::pin_roots(&snapshot);
            let mut out = Wtf8Buf::new();
            out.push_str(label);
            out.push_str("([");
            for i in 0..snapshot.len() {
                if i != 0 {
                    out.push_str(", ");
                }
                let item = pyre_object::gc_roots::shadow_stack_get(item_base + i);
                out.push_wtf8(&py_repr_wtf8(item)?);
            }
            out.push_str("])");
            return Ok(out);
        } else if pyre_object::is_w_range(obj()) {
            // `functional.py W_Range.descr_repr` —
            // `range(start, stop)`, with the step appended only when
            // it is not 1.  Bounds may be bignum, so render each wrapped
            // int rather than a machine word.
            // `range_obj_to_bigint` of a machine int and each `repr` collect;
            // the fields come back off the shadow stack.
            let _range_roots = pyre_object::gc_roots::push_roots();
            let (start, stop, step) = pyre_object::w_range_fields(obj());
            let field_base = pyre_object::gc_roots::pin_roots(&[start, stop, step]);
            let step_is_one = pyre_object::range_obj_to_bigint(
                pyre_object::gc_roots::shadow_stack_get(field_base + 2),
            )
            .int_eq(1);
            let mut out = Wtf8Buf::new();
            out.push_str("range(");
            out.push_wtf8(&py_repr_wtf8(pyre_object::gc_roots::shadow_stack_get(
                field_base,
            ))?);
            out.push_str(", ");
            out.push_wtf8(&py_repr_wtf8(pyre_object::gc_roots::shadow_stack_get(
                field_base + 1,
            ))?);
            if !step_is_one {
                out.push_str(", ");
                out.push_wtf8(&py_repr_wtf8(pyre_object::gc_roots::shadow_stack_get(
                    field_base + 2,
                ))?);
            }
            out.push_str(")");
            return Ok(out);
        } else if pyre_object::interp_sre::is_sre_pattern(obj()) {
            // `pypy/module/_sre/interp_sre.py W_SRE_Pattern.repr_w`.
            return crate::module::_sre::interp_sre::sre_pattern_repr_str(obj());
        } else if pyre_object::interp_sre::is_sre_match(obj()) {
            // `pypy/module/_sre/interp_sre.py W_SRE_Match.repr_w`.
            return crate::module::_sre::interp_sre::sre_match_repr_str(obj());
        } else if pyre_object::memoryview::is_w_memoryview(obj()) {
            // `memoryobject.py descr_repr` — `<memory at 0x...>`, or
            // `<released memory at 0x...>` once the view is released.
            let label = if pyre_object::memoryview::w_memoryview_released(obj()) {
                "released memory"
            } else {
                "memory"
            };
            format!("<{label} at {}>", repr_gc_addr(obj()))
        } else if std::ptr::eq(tp, &INSTANCE_TYPE as *const PyType) {
            // Try __repr__ first, then __str__
            if let Some(w) = try_call_dunder_wtf8(obj(), "__repr__")? {
                return Ok(w);
            }
            if let Some(w) = try_call_dunder_wtf8(obj(), "__str__")? {
                return Ok(w);
            }
            let name = crate::baseobjspace::getfulltypename(obj());
            let addr = repr_gc_addr(obj());
            return Ok(wtf8_format!("<", name, format!(" object at {addr}>")));
        } else {
            // A builtin type carrying its own `__repr__` dict entry (e.g.
            // `_struct.Struct`) — dispatch it before the generic
            // `<name object at 0x...>` fallback.  Mirrors the tuple-subclass
            // path above.
            let w_class = (*obj()).w_class;
            if !w_class.is_null()
                && let Some((src, method)) =
                    crate::baseobjspace::lookup_where_with_method_cache(w_class, "__repr__")
                && !std::ptr::eq(src, crate::typedef::w_object())
                && !method.is_null()
            {
                // The walk above allocates: re-read the receiver so the
                // descriptor is handed the object at its current home.
                let r = crate::builtins::call_and_check(method, &[obj()])?;
                if pyre_object::is_str(r) {
                    return Ok(pyre_object::w_str_get_wtf8(r).to_wtf8_buf());
                }
                return Err(dunder_returned_non_string("__repr__", r));
            }
            let name = crate::baseobjspace::getfulltypename(obj());
            let addr = repr_gc_addr(obj());
            return Ok(wtf8_format!("<", name, format!(" object at {addr}>")));
        };
        Ok(Wtf8Buf::from_string(formatted))
    }
}

/// One piece of a [`wtf8_format!`] message.
///
/// Text a Rust literal or `format!` produced is already `str`; text that came
/// from a Python object — a `repr`, a name, a filename — may hold a lone
/// surrogate and is carried as WTF-8. Both push their own bytes, so nothing on
/// the way to the message goes through `Display`.
pub trait Wtf8Piece {
    fn push_onto(&self, out: &mut Wtf8Buf);
}

impl Wtf8Piece for str {
    fn push_onto(&self, out: &mut Wtf8Buf) {
        out.push_str(self);
    }
}

impl Wtf8Piece for String {
    fn push_onto(&self, out: &mut Wtf8Buf) {
        out.push_str(self);
    }
}

impl Wtf8Piece for Wtf8 {
    fn push_onto(&self, out: &mut Wtf8Buf) {
        out.push_wtf8(self);
    }
}

impl Wtf8Piece for Wtf8Buf {
    fn push_onto(&self, out: &mut Wtf8Buf) {
        out.push_wtf8(self);
    }
}

impl<T: Wtf8Piece + ?Sized> Wtf8Piece for &T {
    fn push_onto(&self, out: &mut Wtf8Buf) {
        (**self).push_onto(out);
    }
}

/// `format!` for a message that interpolates text with no `str` spelling.
///
/// `format!` renders every argument through `Display`, and `Display for Wtf8`
/// substitutes U+FFFD for a lone surrogate — so a message naming a `repr`, a
/// `__qualname__` or a filename silently loses it. Here the literal chunks stay
/// `format!` calls and the WTF-8 pieces are concatenated as themselves.
#[macro_export]
macro_rules! wtf8_format {
    ($($piece:expr),+ $(,)?) => {{
        let mut buf = rustpython_wtf8::Wtf8Buf::new();
        $($crate::display::Wtf8Piece::push_onto(&$piece, &mut buf);)+
        buf
    }};
}
pub use wtf8_format;

/// Format for str() — tries __str__ first, then __repr__.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn py_str_wtf8(mut obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    unsafe {
        // `str` of a tagged `int` immediate is its decimal value; format
        // it before `ob_type` deref. Gated on
        // `CAN_BE_TAGGED` (default false).
        if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
            return Ok(Wtf8Buf::from_string(format!(
                "{}",
                pyre_object::tagged_int::untag_int(obj)
            )));
        }
        // The native `__str__` implementations reached below re-enter
        // `py_str_wtf8` without pushing a Python frame — a one-element
        // `BaseException.args` holding the exception itself recurses here
        // forever. Guard the stack so `str(e)` raises RecursionError instead of
        // overflowing.
        crate::stack_check::stack_check()?;
        if obj.is_null() {
            return Ok(Wtf8Buf::from_string("NULL".to_string()));
        }
        // Keyed on the payload layout, which a `_getusercls` class shares
        // with the builtin it was made from.
        let tp = pyre_object::pyobject::layout_base((*obj).ob_type);
        // For strings, return the value directly (no quotes).
        if std::ptr::eq(tp, &STR_TYPE as *const PyType) {
            if let Some(r) =
                pyre_object::with_roots!(obj => builtin_subclass_dunder_obj(obj, tp, "__str__"))?
            {
                return Ok(pyre_object::w_str_get_wtf8(r).to_wtf8_buf());
            }
            return Ok(pyre_object::w_str_get_wtf8(obj).to_wtf8_buf());
        }
        if std::ptr::eq(tp, &INSTANCE_TYPE as *const PyType) {
            if let Some(w) = pyre_object::with_roots!(obj => try_call_dunder_wtf8(obj, "__str__"))?
            {
                return Ok(w);
            }
            if let Some(w) = pyre_object::with_roots!(obj => try_call_dunder_wtf8(obj, "__repr__"))?
            {
                return Ok(w);
            }
        }
        if unsafe { pyre_object::is_exception(obj) } {
            // `exception_descr_str_wtf8` / `exception_kind_str_wtf8` run
            // `__str__` / `__index__` and can collect.  Pin the receiver so
            // the later arm does not stringify a from-space word
            // (`interp_exceptions.py W_BaseException.descr_str`).
            let _roots = pyre_object::gc_roots::push_roots();
            let obj_slot = pyre_object::gc_roots::pin_roots(&[obj]);
            let obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
            if let Some(w) = exception_descr_str_wtf8(obj())? {
                return Ok(w);
            }
            if let Some(w) = exception_kind_str_wtf8(obj())? {
                return Ok(w);
            }
            // A user subclass that overrides `__str__` shadows the builtin
            // `W_BaseException.descr_str`; dispatch it before the generic
            // args formatting below.  The kind arms above already handled
            // the Unicode / OSError / KeyError `__str__` overrides, so a
            // non-overridden exception here resolves `__str__` to the
            // BaseException builtin and falls through unchanged.
            return base_exception_str_wtf8(obj());
        }
        // `int`/`float`/... define no `tp_str`, so `str()` falls back to
        // `repr()` (a `__str__` override wins, otherwise the `__repr__`
        // override or builtin formatting from `py_repr`).  `str` itself
        // has its own `tp_str` and is handled by the `STR_TYPE` branch
        // above, so this fallthrough never reaches a bare-`str` subclass.
        if let Some(r) =
            pyre_object::with_roots!(obj => builtin_subclass_dunder_obj(obj, tp, "__str__"))?
        {
            return Ok(pyre_object::w_str_get_wtf8(r).to_wtf8_buf());
        }
        // A class object is an instance of its metaclass, so `space.str`
        // resolves `__str__` on that metaclass before falling back to the
        // `<class ...>` representation.  `type` defines no `__str__`, so a
        // metaclass without an override resolves to `object`'s and declines
        // here, leaving `py_repr_wtf8` below to produce the native text.  The
        // override may return a lone surrogate, so read it as WTF-8 rather
        // than folding to `&str`.
        if let Some(result) =
            pyre_object::with_roots!(obj => type_metaclass_dunder_obj(obj, "__str__"))?
        {
            return Ok(pyre_object::w_str_get_wtf8(result).to_wtf8_buf());
        }
        // A `types.ModuleType` subclass `__str__` override wins; without one,
        // `str` falls back to `__repr__` through `py_repr`.
        if pyre_object::is_module(obj)
            && let Some(r) =
                pyre_object::with_roots!(obj => module_user_dunder_obj(obj, "__str__"))?
        {
            return Ok(pyre_object::w_str_get_wtf8(r).to_wtf8_buf());
        }
        py_repr_wtf8(obj)
    }
}

/// `pypy/module/exceptions/interp_exceptions.py descr_str
/// W_BaseException.descr_str`:
///
/// ```python
/// def descr_str(self, space):
///     lgt = len(self.args_w)
///     if lgt == 0:
///         return space.newtext('')
///     elif lgt == 1:
///         return space.str(self.args_w[0])
///     else:
///         return space.str(space.newtuple(self.args_w))
/// ```
///
/// PyPy reads `self.args_w` on every call so `e.args = (...)` mutations are
/// reflected by subsequent `str(e)` reads.
///
/// # Safety
/// `obj` must be a live `W_BaseException`.
pub unsafe fn base_exception_str_wtf8(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    // `space.str(self.args_w[0])` re-enters `py_str_wtf8` on the element, which
    // for `e.args = (e,)` lands back here. The re-entry pushes no Python frame
    // and sits in tail position, so neither the frame counter nor the stack
    // pointer moves and `stack_check` alone cannot see the cycle. Spend a
    // recursion unit for this dispatch level, as the equally frameless
    // `A.__call__ = A()` chain does.
    let _depth = crate::call::enter_native_dispatch();
    unsafe {
        let args = pyre_object::interp_exceptions::w_exception_get_args(obj);
        if args.is_null() {
            return Ok(Wtf8Buf::new());
        }
        if !pyre_object::is_tuple(args) {
            return py_str_wtf8(args);
        }
        let n: usize = pyre_object::w_tuple_len(args);
        if n == 0 {
            return Ok(Wtf8Buf::new());
        }
        if n == 1 {
            let first = pyre_object::w_tuple_getitem(args, 0).unwrap_or(args);
            return py_str_wtf8(first);
        }
        py_str_wtf8(args)
    }
}

/// The `descr_str` overrides the builtin exception classes register on top of
/// `W_BaseException.descr_str`, dispatched on the instance's `ExcKind` because
/// pyre flattens PyPy's subclasses into the single `W_BaseException` struct.
/// `None` means the instance's class inherits the base `descr_str`.
///
/// # Safety
/// `obj` must be a live `W_BaseException`.
#[allow(dead_code)]
pub(crate) unsafe fn exception_kind_str(
    obj: PyObjectRef,
) -> Result<Option<String>, crate::PyError> {
    Ok(unsafe { exception_kind_str_wtf8(obj) }?.map(|w| w.to_string_lossy().into_owned()))
}

pub(crate) unsafe fn exception_kind_str_wtf8(
    obj: PyObjectRef,
) -> Result<Option<Wtf8Buf>, crate::PyError> {
    unsafe {
        // `pypy/module/exceptions/interp_exceptions.py descr_str`
        // `W_UnicodeTranslateError.descr_str`,
        // `:1061-1071` `W_UnicodeDecodeError.descr_str`,
        // `:1175-1191` `W_UnicodeEncodeError.descr_str` — each
        // typedef registers `__str__ = interp2app(descr_str)`,
        // overriding the inherited `W_BaseException.descr_str`.
        // Dispatched on `ExcKind` because Pyre flattens the three
        // PyPy subclasses into the single `W_BaseException`
        // struct.
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::pin_roots(&[obj]);
        let obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
        let kind = unsafe { pyre_object::w_exception_get_kind(obj()) };
        match kind {
            pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError => {
                return unicode_translate_error_str(obj()).map(Some);
            }
            pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError => {
                return unicode_decode_error_str(obj()).map(Some);
            }
            pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError => {
                return unicode_encode_error_str(obj()).map(Some);
            }
            // `interp_exceptions.py key_error_str` — one item is
            // `space.repr(self.args_w[0])`, so `str(KeyError('k'))` is
            // `"'k'"`. `KeyError_str` reprs that same item. Zero items
            // and every other length match `W_BaseException.descr_str`.
            pyre_object::interp_exceptions::ExcKind::KeyError => {
                let stored = pyre_object::interp_exceptions::w_exception_get_args_storage(obj());
                let len = if stored.is_null() {
                    0
                } else {
                    pyre_object::interp_exceptions::rlist_len(stored)
                };
                if len == 1 {
                    let first = pyre_object::interp_exceptions::rlist_getitem(stored, 0);
                    let first_slot = pyre_object::gc_roots::pin_roots(&[first]);
                    return Ok(Some(py_repr_wtf8(
                        pyre_object::gc_roots::shadow_stack_get(first_slot),
                    )?));
                }
            }
            // `OSError_str` and `W_OSError.descr_str` read only the
            // slots. The 2-argument form renders as `"[Errno N] strerror"`,
            // extended with `": 'filename'"` and `" -> 'filename2'"`
            // when those slots are set. Constructor `os_error_fill_slots`
            // fills errno and strerror for the 2..=5 argument forms.
            // `OSError_str` tests `myerrno && strerror` as pointers, so a
            // deleted `_Py_T_OBJECT` member (`PyMember_SetOne` stores
            // NULL) falls through to `BaseException_str`. A stored `None`
            // stays present (`[Errno 2] None`). `readwrite_attrproperty_w`
            // has no `fdel`, so PyPy raises on `del e.strerror`.
            // `descr_str` has no `@jit` hint. Both slots absent and no
            // filename falls through to `W_BaseException.descr_str`.
            pyre_object::interp_exceptions::ExcKind::OSError
            | pyre_object::interp_exceptions::ExcKind::FileNotFoundError => {
                let present = |slot: pyre_object::PyObjectRef| -> Option<pyre_object::PyObjectRef> {
                    if slot.is_null() { None } else { Some(slot) }
                };
                let w_errno = present(pyre_object::interp_exceptions::w_exception_get_errno(obj()));
                let w_strerror = present(pyre_object::interp_exceptions::w_exception_get_strerror(
                    obj(),
                ));
                // `W_OSError.descr_str`: a Windows error code takes
                // priority over the errno, but only where there is something
                // to render it with — a filename, which spells a missing
                // strerror as `None`, or a strerror on its own.  With neither,
                // the code is not reported at all and the errno arms below
                // answer, exactly as they do without one.
                //
                // A stored `None` is present. `OSError_str` tests the filename
                // pointer, and `W_OSError.descr_str` is true for `space.w_None`
                // because that object is not the null slot. Only `PY_NULL` is
                // absent, which is why a constructor argument of `None` (never
                // stored by `os_error_fill_slots`) still renders without a suffix.
                let w_winerror = pyre_object::interp_exceptions::w_exception_get_winerror(obj());
                let w_filename = present(pyre_object::interp_exceptions::w_exception_get_filename(
                    obj(),
                ));
                let has_errno = w_errno.is_some();
                let has_strerror = w_strerror.is_some();
                let has_filename = w_filename.is_some();
                let field_base = pyre_object::gc_roots::pin_roots(&[
                    w_errno.unwrap_or_else(pyre_object::w_none),
                    w_strerror.unwrap_or_else(pyre_object::w_none),
                    w_winerror,
                    w_filename.unwrap_or_else(pyre_object::w_none),
                ]);
                let w_winerror = pyre_object::gc_roots::shadow_stack_get(field_base + 2);
                if !w_winerror.is_null() && (has_filename || has_strerror) {
                    let mut out = Wtf8Buf::new();
                    out.push_str("[WinError ");
                    out.push_wtf8(&py_str_wtf8(pyre_object::gc_roots::shadow_stack_get(
                        field_base + 2,
                    ))?);
                    out.push_str("] ");
                    out.push_wtf8(&py_str_wtf8(pyre_object::gc_roots::shadow_stack_get(
                        field_base + 1,
                    ))?);
                    if has_filename {
                        out.push_str(": ");
                        out.push_wtf8(&py_repr_wtf8(pyre_object::gc_roots::shadow_stack_get(
                            field_base + 3,
                        ))?);
                        let w_filename2 = present(
                            pyre_object::interp_exceptions::w_exception_get_filename2(obj()),
                        );
                        if let Some(fname2) = w_filename2 {
                            out.push_str(" -> ");
                            out.push_wtf8(&py_repr_wtf8(fname2)?);
                        }
                    }
                    return Ok(Some(out));
                }
                // `OSError_str` takes the filename form whenever that pointer
                // is set, and a null errno or strerror is rendered as `None`.
                // `W_OSError.descr_str` substitutes an empty string for a null
                // slot, so `OSError()` plus `filename = "a"` prints
                // `[Errno ] : 'a'` there. `descr_str` has no `@jit` hint.
                if has_filename || (has_errno && has_strerror) {
                    let errno = py_str_wtf8(pyre_object::gc_roots::shadow_stack_get(field_base))?;
                    let strerror =
                        py_str_wtf8(pyre_object::gc_roots::shadow_stack_get(field_base + 1))?;
                    let mut out = Wtf8Buf::new();
                    out.push_str("[Errno ");
                    out.push_wtf8(&errno);
                    out.push_str("] ");
                    out.push_wtf8(&strerror);
                    if has_filename {
                        let w_filename2 = present(
                            pyre_object::interp_exceptions::w_exception_get_filename2(obj()),
                        );
                        let has_fname2 = w_filename2.is_some();
                        let _fname2_roots = pyre_object::gc_roots::push_roots();
                        let fname2_slot =
                            _fname2_roots.pin_roots(&[w_filename2.unwrap_or(pyre_object::PY_NULL)]);
                        out.push_str(": ");
                        out.push_wtf8(&py_repr_wtf8(pyre_object::gc_roots::shadow_stack_get(
                            field_base + 3,
                        ))?);
                        if has_fname2 {
                            out.push_str(" -> ");
                            out.push_wtf8(&py_repr_wtf8(_fname2_roots.get(fname2_slot))?);
                        }
                        return Ok(Some(out));
                    }
                    return Ok(Some(out));
                }
            }
            // `SyntaxError_str` — location suffix, including a non-str
            // `msg`. Shared with IndentationError / TabError (same kind).
            pyre_object::interp_exceptions::ExcKind::SyntaxError => {
                if let Some(w) = exception_descr_str_wtf8(obj())? {
                    return Ok(Some(w));
                }
            }
            // `ImportError_str`, also `ModuleNotFoundError`'s `tp_str`.
            pyre_object::interp_exceptions::ExcKind::ImportError
            | pyre_object::interp_exceptions::ExcKind::ModuleNotFoundError => {
                if let Some(w) = import_error_exact_msg_wtf8(obj()) {
                    return Ok(Some(w));
                }
            }
            _ => {}
        }
        Ok(None)
    }
}

/// `str(obj)` for diagnostic display (traceback headers / messages written to
/// stderr): like [`py_str`], but a lone surrogate is backslash-escaped
/// (`\udcXX`, the `backslashreplace` handler stderr uses) and a raising
/// `__str__` degrades to a placeholder, so rendering a diagnostic never panics.
///
/// # Safety
/// `obj` must be a valid object.
pub unsafe fn py_str_display(obj: PyObjectRef) -> String {
    unsafe {
        let w = match py_str_wtf8(obj) {
            Ok(w) => w,
            Err(_) => return "<unprintable>".to_string(),
        };
        wtf8_display_string(w, "<unprintable>")
    }
}

/// `str(obj)` rendered for a terminal, like [`py_str_display`], but a raising
/// `__str__` is reported to the caller instead of degrading to a placeholder.
///
/// For text that is a value the user supplied rather than a diagnostic about
/// one -- a prompt, say -- `"<unprintable>"` is the wrong answer: the caller
/// has its own fallback and needs to know the read failed to reach it.
///
/// # Safety
/// `obj` must be a valid object.
pub unsafe fn py_str_display_result(obj: PyObjectRef) -> Result<String, crate::PyError> {
    let rendered = unsafe { py_str_wtf8(obj) }?;
    Ok(wtf8_display_string(rendered, "<unprintable>"))
}

/// The text a WTF-8 diagnostic becomes on the way to stderr.
///
/// `sys.stderr` carries `errors='backslashreplace'`, so an unpaired surrogate
/// leaves as the six characters `\uXXXX` rather than as the three WTF-8 bytes
/// behind it — which are not valid UTF-8 and would reach a consumer as
/// replacement characters. The escape is local so a diagnostic raised before
/// the codec is initialized still keeps the surrounding text.
pub(crate) fn wtf8_display_string(rendered: Wtf8Buf, _fallback: &str) -> String {
    if let Ok(s) = rendered.as_str() {
        return s.to_owned();
    }
    // Unpaired surrogates are the only non-UTF-8 WTF-8 sequences. Escape
    // them here: `encode` needs the codec initialized, and a diagnostic
    // raised before that must still keep the surrounding text.
    let mut out = String::with_capacity(rendered.len());
    for cp in rendered.code_points() {
        let u = cp.to_u32();
        if let Some(ch) = char::from_u32(u) {
            out.push(ch);
        } else {
            out.push_str(&format!("\\u{u:04x}"));
        }
    }
    out
}

/// Index `my_basename` slices from: past the last `SEP`, which is `/`, or
/// `\` on Windows.
///
/// [3.14-spec] `W_SyntaxError.descr_str` calls app-level
/// `os.path.basename`, which on Windows also splits on `/` and strips a
/// drive; `exceptions.c my_basename` at v3.14.6 scans for `SEP` alone, so
/// `str(SyntaxError('m', ('C:/a/b.py', 1, 0, '')))` keeps the whole name on
/// Windows. Off Windows the two agree. Evidence is the C source at the tag.
fn syntax_error_basename_start(path: &[u8]) -> usize {
    let sep = if cfg!(windows) { b'\\' } else { b'/' };
    let mut offset = 0;
    let mut index = 0;
    while index < path.len() {
        if path[index] == sep {
            offset = index + 1;
        }
        index += 1;
    }
    offset
}

/// The WTF-8 carrying subset of `W_BaseException.descr_str`: a base
/// exception whose `args_w` is a single `str` stringifies to that str
/// verbatim (`interp_exceptions.py space.str(self.args_w[0])`).
/// Returns `None` for every other shape — no args, multiple args, a
/// non-`str` arg, or the Unicode/`KeyError` kinds whose `descr_str`
/// overrides are ASCII-only — letting `py_str_wtf8` fall back to
/// `py_str`.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
unsafe fn exception_descr_str_wtf8(
    mut obj: PyObjectRef,
) -> Result<Option<Wtf8Buf>, crate::PyError> {
    unsafe {
        // A user subclass that overrides `__str__` shadows the builtin
        // `W_BaseException.descr_str`; dispatch it (preserving WTF-8)
        // before the single-`str`-arg fast path below, matching `py_str`.
        if let Some(r) = pyre_object::with_roots!(obj => exc_user_dunder_obj(obj, "__str__"))? {
            return Ok(Some(pyre_object::w_str_get_wtf8(r).to_wtf8_buf()));
        }
        let kind = pyre_object::w_exception_get_kind(obj);
        if matches!(
            kind,
            pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError
                | pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError
                | pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError
                | pyre_object::interp_exceptions::ExcKind::KeyError
        ) {
            return Ok(None);
        }
        // `SyntaxError_str` appends a location whenever `filename` is a
        // str (a subclass counts; the stored text is used, not
        // `str(filename)`) or `lineno` is an exact int. `msg` is always
        // `str(msg)`, and a missing msg is None, so `None` / `5` still
        // print `None (a.py, line 1)` / `5 (a.py, line 1)`.
        // `W_SyntaxError.descr_str` returns `str(msg)` immediately when
        // `type(msg) is not str`, and it only treats an exact str as a
        // filename. That method has no `@jit` hint.
        // `PyLong_AsLongAndOverflow`'s overflow flag is ignored, so a number
        // too wide for a C long prints as `-1`. The tail is `my_basename`:
        // the text after the last `SEP` (`/`, or `\` on Windows).
        //
        // [3.14-spec] `W_SyntaxError.descr_str` substitutes `???` for a
        // falsy filename (`self.filename or "???"`) and prints
        // `lines N-M` when `end_lineno` is larger; `exceptions.c
        // SyntaxError_str` at v3.14.6 passes the empty name through
        // `my_basename` unchanged and formats `line %ld` from `lineno`
        // alone. Evidence is the C source at the tag.
        if kind == pyre_object::interp_exceptions::ExcKind::SyntaxError {
            let mut w_msg = crate::baseobjspace::syntax_error_attr(obj, "msg");
            let w_lineno = crate::baseobjspace::syntax_error_attr(obj, "lineno");
            let mut w_filename = crate::baseobjspace::syntax_error_attr(obj, "filename");
            // `int_w` reaches `__int__`. `msg` and `filename` stay live, so
            // publish them with the receiver and pass the forwarded lineno.
            let lineno = if pyre_object::pyobject::is_exact_type(w_lineno, &INT_TYPE) {
                let roots = pyre_object::gc_roots::push_roots();
                let base = roots.pin_roots(&[obj, w_msg, w_filename, w_lineno]);
                let lineno = crate::baseobjspace::int_w(roots.get(base + 3))
                    .ok()
                    .and_then(|lineno| core::ffi::c_long::try_from(lineno).ok())
                    .map_or(-1, i64::from);
                obj = roots.get(base);
                w_msg = roots.get(base + 1);
                w_filename = roots.get(base + 2);
                Some(lineno)
            } else {
                None
            };
            let lineno_str = lineno.map(|lineno| Wtf8Buf::from_string(format!("line {lineno}")));
            let filename_tail = if pyre_object::is_str(w_filename) {
                let fbuf = pyre_object::w_str_get_wtf8(w_filename).to_wtf8_buf();
                let start = syntax_error_basename_start(fbuf.as_bytes());
                Some(fbuf[start..].to_wtf8_buf())
            } else {
                None
            };
            if filename_tail.is_none() && lineno_str.is_none() {
                return Ok(Some(
                    pyre_object::with_roots!(obj, w_msg => py_str_wtf8(w_msg))?,
                ));
            }
            // Basename and the line number are sampled before `str(msg)`:
            // `%S` runs after `my_basename` and `PyLong_AsLongAndOverflow`.
            let mut out = pyre_object::with_roots!(obj, w_msg => py_str_wtf8(w_msg))?;
            let extra = match (filename_tail, lineno_str) {
                (Some(mut inner), Some(line)) => {
                    inner.push_str(", ");
                    inner.push_wtf8(&line);
                    Some(inner)
                }
                (Some(inner), None) => Some(inner),
                (None, Some(line)) => Some(line),
                (None, None) => None,
            };
            if let Some(inner) = extra {
                out.push_str(" (");
                out.push_wtf8(&inner);
                out.push_str(")");
            }
            return Ok(Some(out));
        }
        // Exact-str `msg` wins over `args`. A subclass, `None`, or a null
        // slot falls through to `BaseException_str`.
        if matches!(
            kind,
            pyre_object::interp_exceptions::ExcKind::ImportError
                | pyre_object::interp_exceptions::ExcKind::ModuleNotFoundError
        ) && let Some(text) = import_error_exact_msg_wtf8(obj)
        {
            return Ok(Some(text));
        }
        let args = pyre_object::interp_exceptions::w_exception_get_args(obj);
        if args.is_null() || !pyre_object::is_tuple(args) {
            return Ok(None);
        }
        if pyre_object::w_tuple_len(args) != 1 {
            return Ok(None);
        }
        let first = pyre_object::w_tuple_getitem(args, 0).unwrap_or(args);
        // A tagged `int` immediate is never a `str`; skip the `ob_type` deref
        // (which would read the immediate as a pointer) and fall back to
        // `py_str`, which formats the tagged value directly.
        if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(first) {
            return Ok(None);
        }
        if first.is_null() || !std::ptr::eq((*first).ob_type, &STR_TYPE as *const PyType) {
            return Ok(None);
        }
        Ok(Some(pyre_object::w_str_get_wtf8(first).to_wtf8_buf()))
    }
}

/// `ImportError_str` returns an exact `str` `msg` (not a subclass) and
/// otherwise lets `BaseException_str` render `args`. `W_ImportError` has
/// no `descr_str`; `W_BaseException.descr_str` always stringifies
/// `args_w` and has no `@jit` hint. `ModuleNotFoundError` shares
/// `ImportError_str`.
unsafe fn import_error_exact_msg_wtf8(obj: PyObjectRef) -> Option<Wtf8Buf> {
    unsafe {
        let w_msg = pyre_object::interp_exceptions::w_exception_get_import_msg(obj);
        if w_msg.is_null() || !pyre_object::pyobject::is_exact_type(w_msg, &STR_TYPE) {
            return None;
        }
        Some(pyre_object::w_str_get_wtf8(w_msg).to_wtf8_buf())
    }
}

/// Format an `int` `%d` position slot from a `W_BaseException`
/// typed Unicode*Error position field.  `descr_init`'s typecheck
/// admits `int` (including subclasses), so a successfully-initialised
/// instance always yields a number here.  After a writer-driven
/// mutation through `readwrite_attrproperty_w`, however, the slot may
/// hold any object — PyPy's appexec-driven `"%d" % w_start` raises
/// `TypeError` on non-int values.  Pyre's `py_str` cannot propagate
/// `PyError` from inside `descr_str`, so the closest behavior is
/// Python's `"%s" % value` (str-coerced) for the failure case: that
/// keeps the original value visible in the formatted message instead
/// of silently substituting `0`.  `Ok(i64)` carries a numeric value
/// (used for `end - 1` arithmetic and the `end == start + 1` shape
/// check); `Err(String)` carries the pre-formatted str-coerced
/// fallback for direct interpolation into the message.
unsafe fn unicode_err_int_slot(mut stored: PyObjectRef) -> Result<i64, Wtf8Buf> {
    unsafe {
        if stored.is_null() || pyre_object::is_none(stored) {
            // Never set / explicit None — PyPy class-default `w_start
            // = None`.  `"%d" % None` raises, but in pyre py_str
            // cannot raise; surface "None" so the bad state is at
            // least visible.
            return Err(Wtf8Buf::from_string("None".to_string()));
        }
        // `int_w` walks the __int__/__index__ protocol, so int
        // subclasses with stored intval (`class MyInt(int): pass`,
        // `True`/`False`) and any object implementing __index__ all
        // resolve to the numeric value — matching PyPy's
        // `"%d" % value` semantics.
        if let Ok(v) = pyre_object::with_roots!(stored => crate::baseobjspace::int_w(stored)) {
            return Ok(v);
        }
        // `descr_str` deliberately str-coerces rather than raising; a raising
        // `__str__` on the mutated slot degrades to empty here.
        Err(py_str_wtf8(stored).unwrap_or_default())
    }
}

/// Format an `str` `%s` slot (encoding / reason) from a typed
/// Unicode*Error field. `UnicodeDecodeError_str` / `UnicodeEncodeError_str`
/// call `PyObject_Str` on the slot, which spells a NULL pointer as
/// `<NULL>`. A stored `None` stringifies as `None`. `descr_init`
/// rejects a non-str at construction; this helper covers a later
/// mutation (`e.encoding = 42`, `e.reason = None`).
unsafe fn unicode_err_str_slot(stored: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    unsafe {
        if stored.is_null() {
            return Ok(Wtf8Buf::from_string("<NULL>".to_owned()));
        }
        // `__str__` below can collect; pin before the exact-type probe so a
        // nursery encoding/reason is forwarded rather than stringified as a
        // from-space word (`interp_exceptions.py` `%s` coerce).
        let _roots = pyre_object::gc_roots::push_roots();
        let stored_slot = pyre_object::gc_roots::pin_roots(&[stored]);
        let stored = pyre_object::gc_roots::shadow_stack_get(stored_slot);
        if pyre_object::is_exact_type(stored, &pyre_object::STR_TYPE) {
            return Ok(pyre_object::w_str_get_wtf8(stored).to_wtf8_buf());
        }
        // `%s` propagates an exception raised by the value's `__str__`.
        py_str_wtf8(pyre_object::gc_roots::shadow_stack_get(stored_slot))
    }
}

/// Single-char `%d`-slot formatter: takes the `(Ok|Err)` from
/// `unicode_err_int_slot` and renders either the `int` or the
/// str-coerced fallback verbatim.
fn unicode_err_int_repr(slot: &Result<i64, Wtf8Buf>) -> Wtf8Buf {
    match slot {
        Ok(v) => Wtf8Buf::from_string(v.to_string()),
        Err(s) => s.clone(),
    }
}

/// `check_unicode_error_attribute` after `UnicodeDecodeError_str` rereads
/// `object`. A NULL slot is "not set"; a present non-bytes / non-str
/// value, `None` included, is the `must be a bytes` / `must be a string`
/// TypeError. `W_UnicodeDecodeError.descr_str` treats `None` as empty
/// and has no `@jit` hint.
unsafe fn unicode_err_require_object(
    w_object: PyObjectRef,
    as_bytes: bool,
) -> Result<(), crate::PyError> {
    if w_object.is_null() {
        return Err(crate::PyError::type_error(
            "UnicodeError 'object' attribute is not set",
        ));
    }
    let ok = if as_bytes {
        unsafe { pyre_object::is_bytes_like(w_object) }
    } else {
        unsafe { pyre_object::is_str(w_object) }
    };
    if !ok {
        let expected = if as_bytes { "a bytes" } else { "a string" };
        return Err(crate::PyError::type_error(format!(
            "UnicodeError 'object' attribute must be {expected}"
        )));
    }
    Ok(())
}

/// Single-item form only when the offending slice is one unit inside
/// the object.  `UnicodeDecodeError_str`, `UnicodeEncodeError_str`
/// and `UnicodeTranslateError_str` share the guard
/// `start >= 0 && start < len && end >= 0 && end <= len && end == start + 1`
/// (3.14.6).  `W_UnicodeDecodeError.descr_str`,
/// `W_UnicodeEncodeError.descr_str` and
/// `W_UnicodeTranslateError.descr_str` have no guard: they index the
/// object directly, so an out-of-range or negative `start` raises
/// `IndexError` there.
fn unicode_err_index_in_range(start: i64, end: i64, len: usize) -> bool {
    // `start < len` bounds `start` below the object length before
    // `start + 1` is evaluated, so the sum cannot leave the range.
    let len = len as i64;
    start >= 0 && start < len && end >= 0 && end <= len && end == start + 1
}

/// `end - 1` for the plural message: matches PyPy's `self.end - 1`.
/// On an int slot, arithmetic; on the str-coerced fallback, the
/// value is embedded verbatim so the message still reflects what the
/// user actually stored.
fn unicode_err_end_minus_one_repr(slot: &Result<i64, Wtf8Buf>) -> Wtf8Buf {
    match slot {
        Ok(v) => Wtf8Buf::from_string((*v).wrapping_sub(1).to_string()),
        Err(s) => s.clone(),
    }
}

/// `pypy/module/exceptions/interp_exceptions.py descr_str
/// W_UnicodeTranslateError.descr_str`:
///
/// ```python
/// if self.object is None:
///     return ""
/// if self.end == self.start + 1:
///     badchar = ord(self.object[self.start])
///     if badchar <= 0xff:
///         return "can't translate character '\\x%02x' in position %d: %s"
///     ...
/// return "can't translate characters in position %d-%d: %s"
/// ```
///
/// `UnicodeTranslateError_str` returns empty only when `object` is
/// NULL. A stored `None` is present and `check_unicode_error_attribute`
/// then raises `must be a string`. `W_UnicodeTranslateError.descr_str`
/// returns empty when the slot is None and has no `@jit` hint.
///
/// Non-int `start`/`end` are rendered via `"%s"`-style str-coercion
/// (`unicode_err_int_slot`) in the range form.  The single-character
/// form is taken only when [`unicode_err_index_in_range`] holds:
/// `UnicodeTranslateError_str` (3.14.6) requires the index inside the
/// object, and `W_UnicodeTranslateError.descr_str` has no such guard.
unsafe fn unicode_translate_error_str(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let obj = pyre_object::gc_roots::pin_root(obj);
        let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let initial = pyre_object::interp_exceptions::w_exception_get_object(obj);
        // `UnicodeTranslateError_str` returns empty when `object` is NULL.
        // `W_UnicodeTranslateError.descr_str` returns empty when it is None.
        if initial.is_null() {
            return Ok(Wtf8Buf::new());
        }
        // Each of these three reads can run Python — `int_w` walks
        // `__index__` and `unicode_err_str_slot` calls `__str__` — so the
        // receiver is refetched from its slot before every one of them
        // rather than carried in a local across them.
        let start_slot =
            unicode_err_int_slot(pyre_object::interp_exceptions::w_exception_get_start(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
            ));
        let end_slot = unicode_err_int_slot(pyre_object::interp_exceptions::w_exception_get_end(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
        ));
        let reason = unicode_err_str_slot(pyre_object::interp_exceptions::w_exception_get_reason(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
        ))?;
        // Formatting `reason` can run arbitrary Python and mutate the
        // exception.  CPython 3.14 rereads `object` before indexing it.
        let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        let w_object = pyre_object::interp_exceptions::w_exception_get_object(obj);
        unicode_err_require_object(w_object, false)?;
        let start_repr = unicode_err_int_repr(&start_slot);
        // `UnicodeTranslateError_str` (3.14.6) takes the single-character
        // form only when the slice is inside the object.
        // `W_UnicodeTranslateError.descr_str` has no guard.
        let len = pyre_object::w_str_get_wtf8(w_object).code_points().count();
        let single_char = matches!(
            (&start_slot, &end_slot),
            (Ok(s), Ok(e)) if unicode_err_index_in_range(*s, *e, len)
        );
        if single_char {
            let start = *start_slot.as_ref().expect("single_char gated on Ok");
            // The guard puts `start` on a real code point.  Read it
            // through the surrogate-aware WTF-8 view: the bad character
            // is frequently a lone surrogate, which `w_str_get_value`
            // cannot hold.
            let badchar = pyre_object::w_str_get_wtf8(w_object)
                .code_points()
                .nth(usize::try_from(start).expect("in-range start"))
                .expect("in-range start")
                .to_u32();
            let badchar_repr = if badchar <= 0xff {
                format!("'\\x{:02x}'", badchar)
            } else if badchar <= 0xffff {
                format!("'\\u{:04x}'", badchar)
            } else {
                format!("'\\U{:08x}'", badchar)
            };
            let mut out = wtf8_format!(
                format!("can't translate character {badchar_repr} in position "),
                start_repr,
                ": ",
            );
            out.push_wtf8(&reason);
            return Ok(out);
        }
        let mut out = wtf8_format!(
            "can't translate characters in position ",
            start_repr,
            "-",
            unicode_err_end_minus_one_repr(&end_slot),
            ": ",
        );
        out.push_wtf8(&reason);
        Ok(out)
    }
}

/// `pypy/module/exceptions/interp_exceptions.py descr_str
/// W_UnicodeDecodeError.descr_str`:
///
/// ```python
/// if self.object is None: return ""
/// if self.end == self.start + 1:
///     return "'%s' codec can't decode byte 0x%02x in position %d: %s"%(
///         self.encoding, self.object[self.start], self.start, self.reason)
/// return "'%s' codec can't decode bytes in position %d-%d: %s" % (
///     self.encoding, self.start, self.end - 1, self.reason)
/// ```
///
/// Non-int `start`/`end` fall back to `"%s"`-style str-coercion in
/// the range form.  The single-byte form is taken only when
/// [`unicode_err_index_in_range`] holds: `UnicodeDecodeError_str`
/// (3.14.6) requires the index inside the object, and
/// `W_UnicodeDecodeError.descr_str` has no such guard.
unsafe fn unicode_decode_error_str(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let obj = pyre_object::gc_roots::pin_root(obj);
        let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let initial = pyre_object::interp_exceptions::w_exception_get_object(obj);
        // `UnicodeDecodeError_str` returns empty when `object` is NULL.
        // `W_UnicodeDecodeError.descr_str` returns empty when it is None.
        if initial.is_null() {
            return Ok(Wtf8Buf::new());
        }
        let encoding =
            unicode_err_str_slot(pyre_object::interp_exceptions::w_exception_get_encoding(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
            ))?;
        // Each of these three reads can run Python — `int_w` walks
        // `__index__` and `unicode_err_str_slot` calls `__str__` — so the
        // receiver is refetched from its slot before every one of them
        // rather than carried in a local across them.
        let start_slot =
            unicode_err_int_slot(pyre_object::interp_exceptions::w_exception_get_start(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
            ));
        let end_slot = unicode_err_int_slot(pyre_object::interp_exceptions::w_exception_get_end(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
        ));
        let reason = unicode_err_str_slot(pyre_object::interp_exceptions::w_exception_get_reason(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
        ))?;
        let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        let w_object = pyre_object::interp_exceptions::w_exception_get_object(obj);
        unicode_err_require_object(w_object, true)?;
        let start_repr = unicode_err_int_repr(&start_slot);
        // `UnicodeDecodeError_str` (3.14.6) takes the single-byte form
        // only when the slice is inside the object.
        // `W_UnicodeDecodeError.descr_str` has no guard.
        let data = pyre_object::bytes_like_data(w_object);
        let single_char = matches!(
            (&start_slot, &end_slot),
            (Ok(s), Ok(e)) if unicode_err_index_in_range(*s, *e, data.len())
        );
        if single_char {
            let start = *start_slot.as_ref().expect("single_char gated on Ok");
            let byte = data[usize::try_from(start).expect("in-range start")];
            let mut out = Wtf8Buf::new();
            out.push_str("'");
            out.push_wtf8(&encoding);
            out.push_str(&format!(
                "' codec can't decode byte 0x{byte:02x} in position ",
            ));
            out.push_wtf8(&start_repr);
            out.push_str(": ");
            out.push_wtf8(&reason);
            return Ok(out);
        }
        let mut out = Wtf8Buf::new();
        out.push_str("'");
        out.push_wtf8(&encoding);
        out.push_str("' codec can't decode bytes in position ");
        out.push_wtf8(&start_repr);
        out.push_str("-");
        out.push_wtf8(&unicode_err_end_minus_one_repr(&end_slot));
        out.push_str(": ");
        out.push_wtf8(&reason);
        Ok(out)
    }
}

/// `W_UnicodeEncodeError.descr_str` — same single/range split as
/// `W_UnicodeTranslateError` but prefixed with the encoding name.
/// The single-character form is taken only when
/// [`unicode_err_index_in_range`] holds: `UnicodeEncodeError_str`
/// (3.14.6) requires the index inside the object, and
/// `W_UnicodeEncodeError.descr_str` has no such guard.
unsafe fn unicode_encode_error_str(obj: PyObjectRef) -> Result<Wtf8Buf, crate::PyError> {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let obj = pyre_object::gc_roots::pin_root(obj);
        let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let initial = pyre_object::interp_exceptions::w_exception_get_object(obj);
        // `UnicodeEncodeError_str` returns empty when `object` is NULL.
        // `W_UnicodeEncodeError.descr_str` returns empty when it is None.
        if initial.is_null() {
            return Ok(Wtf8Buf::new());
        }
        let encoding =
            unicode_err_str_slot(pyre_object::interp_exceptions::w_exception_get_encoding(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
            ))?;
        // Each of these three reads can run Python — `int_w` walks
        // `__index__` and `unicode_err_str_slot` calls `__str__` — so the
        // receiver is refetched from its slot before every one of them
        // rather than carried in a local across them.
        let start_slot =
            unicode_err_int_slot(pyre_object::interp_exceptions::w_exception_get_start(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
            ));
        let end_slot = unicode_err_int_slot(pyre_object::interp_exceptions::w_exception_get_end(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
        ));
        let reason = unicode_err_str_slot(pyre_object::interp_exceptions::w_exception_get_reason(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
        ))?;
        let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        let w_object = pyre_object::interp_exceptions::w_exception_get_object(obj);
        unicode_err_require_object(w_object, false)?;
        let start_repr = unicode_err_int_repr(&start_slot);
        // `UnicodeEncodeError_str` (3.14.6) takes the single-character
        // form only when the slice is inside the object.
        // `W_UnicodeEncodeError.descr_str` has no guard.
        let len = pyre_object::w_str_get_wtf8(w_object).code_points().count();
        let single_char = matches!(
            (&start_slot, &end_slot),
            (Ok(s), Ok(e)) if unicode_err_index_in_range(*s, *e, len)
        );
        if single_char {
            let start = *start_slot.as_ref().expect("single_char gated on Ok");
            // The guard puts `start` on a real code point.  Read it
            // through the surrogate-aware WTF-8 view: the bad character
            // is frequently a lone surrogate, which `w_str_get_value`
            // cannot hold.
            let badchar = pyre_object::w_str_get_wtf8(w_object)
                .code_points()
                .nth(usize::try_from(start).expect("in-range start"))
                .expect("in-range start")
                .to_u32();
            let badchar_repr = if badchar <= 0xff {
                format!("'\\x{:02x}'", badchar)
            } else if badchar <= 0xffff {
                format!("'\\u{:04x}'", badchar)
            } else {
                format!("'\\U{:08x}'", badchar)
            };
            let mut out = Wtf8Buf::new();
            out.push_str("'");
            out.push_wtf8(&encoding);
            out.push_str(&format!(
                "' codec can't encode character {badchar_repr} in position ",
            ));
            out.push_wtf8(&start_repr);
            out.push_str(": ");
            out.push_wtf8(&reason);
            return Ok(out);
        }
        let mut out = Wtf8Buf::new();
        out.push_str("'");
        out.push_wtf8(&encoding);
        out.push_str("' codec can't encode characters in position ");
        out.push_wtf8(&start_repr);
        out.push_str("-");
        out.push_wtf8(&unicode_err_end_minus_one_repr(&end_slot));
        out.push_str(": ");
        out.push_wtf8(&reason);
        Ok(out)
    }
}

/// Display wrapper for PyObjectRef.
pub struct PyDisplay(pub PyObjectRef);

impl fmt::Display for PyDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_null() {
            write!(f, "NULL")
        } else {
            // `Display` cannot surface a `PyError`; a raising `__str__` in a
            // diagnostic output context degrades to a placeholder rather than
            // propagating (the user-facing `print()`/`str()` paths thread the
            // error through `py_str`).
            let s = unsafe { py_str_wtf8(self.0) }
                .map(|w| wtf8_display_string(w, "<unprintable>"))
                .unwrap_or_else(|_| "<exception in __str__>".to_string());
            write!(f, "{s}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        bytes_repr_string, format_float_repr, format_wtf8_repr, jit_format_float_repr_rstr,
        syntax_error_basename_start,
    };
    use rustpython_wtf8::{CodePoint, Wtf8Buf};

    #[test]
    fn bytes_repr_matches_pypy_quote_and_escape_edges() {
        assert_eq!(bytes_repr_string(b""), "b''");
        assert_eq!(bytes_repr_string(b"a'b"), "b\"a'b\"");
        assert_eq!(bytes_repr_string(b"a\"b"), "b'a\"b'");
        assert_eq!(bytes_repr_string(b"a'\"b"), "b'a\\'\"b'");
        assert_eq!(
            bytes_repr_string(&[0, 9, 10, 13, 31, 32, 92, 126, 127, 255]),
            "b'\\x00\\t\\n\\r\\x1f \\\\~\\x7f\\xff'"
        );
    }

    #[test]
    fn unicode_repr_matches_pypy_quote_printable_and_surrogate_edges() {
        assert_eq!(format_wtf8_repr(Wtf8Buf::from("a'b").as_ref()), "\"a'b\"");
        assert_eq!(format_wtf8_repr(Wtf8Buf::from("a\"b").as_ref()), "'a\"b'");
        assert_eq!(
            format_wtf8_repr(Wtf8Buf::from("\0\t\n\r ☃\u{e0020}").as_ref()),
            "'\\x00\\t\\n\\r ☃\\U000e0020'"
        );
        let mut surrogate = Wtf8Buf::from("x");
        surrogate.push(CodePoint::from_u32(0xd800).unwrap());
        assert_eq!(format_wtf8_repr(&surrogate), "'x\\ud800'");
    }

    #[test]
    fn float_repr_residual_returns_the_same_lowlevel_string_bytes() {
        for value in [0.0, -0.0, 1.5, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let s = jit_format_float_repr_rstr(value);
            let bytes = unsafe { pyre_object::unicodeobject::utf8_payload_bytes(s) };
            assert_eq!(bytes, format_float_repr(value).as_bytes());
        }
    }

    #[test]
    fn syntax_error_basename_splits_on_sep_only() {
        assert_eq!(syntax_error_basename_start(b""), 0);
        assert_eq!(syntax_error_basename_start(b"enc.py"), 0);
        assert_eq!(syntax_error_basename_start(b"C:foo.py"), 0);
        #[cfg(not(windows))]
        {
            assert_eq!(syntax_error_basename_start(b"dir/enc.py"), 4);
            assert_eq!(syntax_error_basename_start(b"dir/"), 4);
            assert_eq!(syntax_error_basename_start(br"dir\enc.py"), 0);
        }
        #[cfg(windows)]
        {
            assert_eq!(syntax_error_basename_start(b"dir/enc.py"), 0);
            assert_eq!(syntax_error_basename_start(br"dir\enc.py"), 4);
            assert_eq!(syntax_error_basename_start(br"dir\"), 4);
        }
    }
}
