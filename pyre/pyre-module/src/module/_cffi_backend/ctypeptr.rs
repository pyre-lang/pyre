//! Pointers and arrays — PyPy:
//! `pypy/module/_cffi_backend/ctypeptr.py` and `ctypearray.py`.
//!
//! `W_CTypePtrOrArray`, `W_CTypePtrBase`, `W_CTypePointer` and
//! `W_CTypeArray` all share `W_CType`'s typedef, so their overrides are the
//! `match` arms below rather than four classes.

use pyre_interpreter::PyError;
use pyre_object::PyObjectRef;
use std::collections::HashSet;
use std::ffi::{CString, c_char, c_int, c_void};
use std::sync::{Mutex, OnceLock};

use super::cdataobj::{self, W_CData};
use super::ctypeobj::{self, W_CType};
use super::misc;

/// `W_CTypePtrBase.convert_to_object` — the pointer word at `cdata`, boxed.
///
/// # Safety
/// `cdata` must be readable for the width of a pointer.
pub unsafe fn pointer_convert_to_object(ct: &W_CType, cdata: *const u8) -> PyObjectRef {
    // `rffi.cast(rffi.CCHARPP, cdata)[0]` — `getarrayitem` of a raw
    // pointer array, which `rewrite_op_getarrayitem` keeps as
    // `getarrayitem_raw` / `raw_load`. The oopspec leaf is
    // `cdataobj::raw_read_ptr`; inlining `read_unaligned` here left
    // that rustc helper in the `_CDataBase` call descent.
    let target = cdataobj::raw_read_ptr(cdata as usize);
    cdataobj::new_cdata(target, ct.as_object())
}

/// `W_CTypeArray.convert_to_object` — the array itself, not a copy.  An
/// array of unknown length has no length to give the result, so it decays to
/// the pointer type, exactly as the comment there says.
///
/// # Safety
/// `cdata` must point into the array's storage.
pub unsafe fn array_convert_to_object(ct: &W_CType, cdata: *const u8) -> PyObjectRef {
    let w_ctype = if ct.length < 0 {
        ct.ctptr
    } else {
        ct.as_object()
    };
    cdataobj::new_cdata(cdata as usize, w_ctype)
}

/// `W_CTypePtrBase.convert_from_object`.
///
/// # Safety
/// `cdata` must be writable for the width of a pointer.
pub unsafe fn pointer_convert_from_object(
    ct: &W_CType,
    cdata: *mut u8,
    w_ob: PyObjectRef,
) -> Result<(), PyError> {
    let Some(source) = W_CData::from_obj(w_ob) else {
        return Err(unsafe {
            PyError::from_exc_object(ctypeobj::convert_error(ct, &"cdata pointer", w_ob))
        });
    };
    // Spelled as a `match` rather than `Option::ok_or_else`: the closure
    // the combinator takes is a callee of its own, and a traced cdata
    // call would stop at it.
    let mut other = match ctypeobj::ctype_at(source.ctype) {
        Some(ct) => ct,
        None => return Err(PyError::system_error("cdata without a ctype")),
    };
    if other.kind == ctypeobj::KIND_ARRAY {
        other = match ctypeobj::ctype_at(other.ctptr) {
            Some(ct) => ct,
            None => return Err(PyError::system_error("array without a pointer type")),
        };
    }
    // `W_CTypeFunc` and `W_CTypePointer` are both `W_CTypePtrBase` upstream.
    if !matches!(other.kind, ctypeobj::KIND_POINTER | ctypeobj::KIND_FUNC) {
        return Err(unsafe {
            PyError::from_exc_object(ctypeobj::convert_error(ct, &"compatible pointer", w_ob))
        });
    }
    if !std::ptr::eq(ct as *const W_CType, other as *const W_CType) {
        if ct.has(ctypeobj::CTypeFlags::VOID_PTR) || other.has(ctypeobj::CTypeFlags::VOID_PTR) {
            // A cast from or to 'void *' is always allowed.
        } else if ct.has(ctypeobj::CTypeFlags::VOIDCHAR_PTR)
            || other.has(ctypeobj::CTypeFlags::VOIDCHAR_PTR)
        {
            // 'char *' is accepted either way for backward compatibility;
            // between two single-byte pointers it is silent, otherwise it
            // warns that the acceptance will end.
            if !(ct.has(ctypeobj::CTypeFlags::ONEBYTE_PTR)
                && other.has(ctypeobj::CTypeFlags::ONEBYTE_PTR))
            {
                implicit_cast_warning(other, ct)?;
            }
        } else {
            return Err(unsafe {
                PyError::from_exc_object(ctypeobj::convert_error(ct, &"compatible pointer", w_ob))
            });
        }
    }
    cdataobj::raw_write_ptr(cdata as usize, source.ptr);
    Ok(())
}

/// `W_CTypePtrBase.convert_from_object` — the `char *` compatibility warning.
/// `oefmt`/`space.warn` formatting stays off the look-inside graph.
#[majit_macros::dont_look_inside]
pub(crate) fn implicit_cast_warning(other: &W_CType, ct: &W_CType) -> Result<(), PyError> {
    pyre_interpreter::warn::warn_category(
        &format!(
            "implicit cast from '{}' to '{}' will be forbidden in the future (check that the types are as you expect; use an explicit ffi.cast() if they are correct)",
            other.name(),
            ct.name()
        ),
        "UserWarning",
        1,
    )
}

/// `W_CTypePtrOrArray._convert_array_from_listview` — `oefmt`.
#[majit_macros::dont_look_inside]
pub(crate) fn too_many_array_initializers(ct: &W_CType, got: usize) -> PyObjectRef {
    PyError::index_error(format!(
        "too many initializers for '{}' (got {})",
        ct.name(),
        got
    ))
    .to_exc_object()
}

/// `W_CTypePtrOrArray.convert_array_from_object` — `oefmt` for a bytes initializer.
#[majit_macros::dont_look_inside]
pub(crate) fn initializer_string_too_long(ct: &W_CType, n: i64) -> PyObjectRef {
    PyError::index_error(format!(
        "initializer string is too long for '{}' (got {n} characters)",
        ct.name()
    ))
    .to_exc_object()
}

/// `W_CTypePtrOrArray.convert_array_from_object` — `oefmt` for a unicode initializer.
#[majit_macros::dont_look_inside]
pub(crate) fn initializer_unicode_too_long(ct: &W_CType, n: i64) -> PyObjectRef {
    PyError::index_error(format!(
        "initializer unicode string is too long for '{}' (got {n} characters)",
        ct.name()
    ))
    .to_exc_object()
}

/// `W_CTypePtrOrArray._must_be_string_of_zero_or_one`.
#[majit_macros::dont_look_inside]
pub(crate) fn bool_array_not_zero_or_one() -> PyObjectRef {
    PyError::value_error("an array of _Bool can only contain \\x00 or \\x01").to_exc_object()
}

/// `W_CTypeArray.convert_from_object`.
///
/// # Safety
/// `cdata` must be writable for the array's size.
pub unsafe fn array_convert_from_object(
    ct: &W_CType,
    cdata: *mut u8,
    w_ob: PyObjectRef,
) -> Result<(), PyError> {
    // The fast path: an array of exactly this type is a straight copy.
    if let Some(source) = W_CData::from_obj(w_ob)
        && source.ctype == ct.as_object()
    {
        let item = item_of(ct)?;
        let length = source.array_length()?;
        // `ctypearray.py convert_from_object` — `rffi.c_memcpy`, an
        // external call.
        misc::raw_memcopy_opaque(source.ptr, cdata as usize, (item.size * length) as usize);
        return Ok(());
    }
    unsafe { convert_array_from_object(ct, cdata, w_ob) }
}

/// `W_CTypePtrOrArray.convert_array_from_object`.
///
/// # Safety
/// `cdata` must be writable for as many items as `w_ob` supplies.
pub unsafe fn convert_array_from_object(
    ct: &W_CType,
    cdata: *mut u8,
    w_ob: PyObjectRef,
) -> Result<(), PyError> {
    let item = item_of(ct)?;
    if unsafe { pyre_object::pyobject::is_list(w_ob) || pyre_object::pyobject::is_tuple(w_ob) } {
        let items = pyre_interpreter::baseobjspace::unpackiterable(w_ob, -1)?;
        if !ct.within_bounds(items.len() as i64) {
            return Err(unsafe {
                PyError::from_exc_object(too_many_array_initializers(ct, items.len()))
            });
        }
        let roots = pyre_object::gc_roots::push_roots();
        let base = pyre_object::gc_roots::pin_roots(&items);
        let items: Vec<_> = (0..items.len()).map(|i| roots.get(base + i)).collect();
        if unsafe { super::ctypeprim::pack_list_of_items(item, cdata, &items)? } {
            return Ok(());
        }
        for i in 0..items.len() {
            let element = unsafe { cdata.offset(i as isize * item.size as isize) };
            unsafe { ctypeobj::convert_from_object(item, element as usize, roots.get(base + i))? };
        }
        return Ok(());
    }
    if ct.has(ctypeobj::CTypeFlags::ACCEPT_STR) {
        if !unsafe { pyre_object::bytesobject::is_bytes(w_ob) } {
            return Err(unsafe {
                PyError::from_exc_object(ctypeobj::convert_error(
                    ct,
                    &"bytes or list or tuple",
                    w_ob,
                ))
            });
        }
        let s = unsafe { pyre_object::bytesobject::w_bytes_data(w_ob) };
        let n = s.len() as i64;
        if ct.length >= 0 && n > ct.length {
            return Err(unsafe { PyError::from_exc_object(initializer_string_too_long(ct, n)) });
        }
        if item.kind == ctypeobj::KIND_PRIM_BOOL && s.iter().any(|&c| c > 1) {
            return Err(unsafe { PyError::from_exc_object(bool_array_not_zero_or_one()) });
        }
        unsafe {
            // `copy_string_to_raw` — residual memcpy of the initializer.
            misc::raw_memcopy_opaque(s.as_ptr() as usize, cdata as usize, s.len());
            if n != ct.length {
                cdata.offset(n as isize).write(0);
            }
        }
        return Ok(());
    }
    if item.kind == ctypeobj::KIND_PRIM_UNICHAR {
        if !unsafe { pyre_object::unicodeobject::is_str(w_ob) } {
            return Err(unsafe {
                PyError::from_exc_object(ctypeobj::convert_error(
                    ct,
                    &"unicode or list or tuple",
                    w_ob,
                ))
            });
        }
        let value = unsafe { pyre_object::w_str_get_wtf8(w_ob) };
        let n = if item.size == 2 {
            super::wchar_helper::utf8_size_as_char16(value)
        } else {
            value.code_points().count() as i64
        };
        if ct.length >= 0 && n > ct.length {
            return Err(unsafe { PyError::from_exc_object(initializer_unicode_too_long(ct, n)) });
        }
        let add_final_zero = n != ct.length;
        unsafe {
            if item.size == 2 {
                super::wchar_helper::utf8_to_char16(value, cdata, n, add_final_zero)?;
            } else {
                super::wchar_helper::utf8_to_char32(value, cdata, n, add_final_zero)?;
            }
        }
        return Ok(());
    }
    Err(unsafe { PyError::from_exc_object(ctypeobj::convert_error(ct, &"list or tuple", w_ob)) })
}

/// `W_CTypePtrOrArray.cast`.
pub fn cast(mut w_ctype: PyObjectRef, mut w_ob: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    if ct.size < 0 {
        return Err(PyError::type_error(format!(
            "cannot cast to '{}'",
            ct.name()
        )));
    }
    // `W_CTypePointer.cast`: casting a stream to a `FILE *` opens one over it.
    if ct.has(ctypeobj::CTypeFlags::FILE_PTR) {
        let file = pyre_object::with_roots!(w_ctype, w_ob => prepare_file(w_ob))?;
        if !file.is_null() {
            return Ok(cdataobj::new_cdata(file as usize, w_ctype));
        }
    }
    let value = if let Some(source) = W_CData::from_obj(w_ob)
        && ctypeobj::ctype_at(source.ctype).is_some_and(|it| it.is_ptr_or_array())
    {
        source.ptr
    } else {
        pyre_object::with_roots!(w_ctype => misc::as_unsigned_long(w_ob, false))? as usize
    };
    Ok(cdataobj::new_cdata(value, w_ctype))
}

/// `W_CTypePointer.newp`.
pub fn pointer_newp(
    w_ctype: PyObjectRef,
    w_init: PyObjectRef,
    w_allocator: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    pointer_newp_with_allocator(
        w_ctype,
        w_init,
        super::allocator::W_Allocator::from_obj(w_allocator),
    )
}

pub(crate) fn pointer_newp_with_allocator(
    w_ctype: PyObjectRef,
    w_init: PyObjectRef,
    mut allocator: Option<&mut super::allocator::W_Allocator>,
) -> Result<PyObjectRef, PyError> {
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    let item = item_of(ct)?;
    let w_item = ct.ctitem;
    let mut datasize = item.size;
    if datasize < 0 {
        return Err(PyError::type_error(format!(
            "cannot instantiate ctype '{}' of unknown size",
            ct.name()
        )));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let init_slot = roots.pin_roots(&[w_init, w_ctype, w_item]);
    let ctype_slot = init_slot + 1;
    let item_slot = init_slot + 2;
    let cdata_slot = init_slot + 3;
    if item.is_struct_or_union() {
        // `newp` on a struct-or-union pointer hands back a co-owner of the
        // cdata that really holds the struct, so `p[0]` is that object.
        let mut varsize_length = -1;
        item.force_lazy_struct()?;
        if item.has(ctypeobj::CTypeFlags::WITH_VAR_ARRAY) {
            if !unsafe { pyre_object::pyobject::is_none(roots.get(init_slot)) } {
                datasize = unsafe {
                    super::ctypestruct::convert_struct_from_object(
                        item,
                        std::ptr::null_mut(),
                        roots.get(init_slot),
                        datasize,
                    )?
                };
            }
            varsize_length = datasize;
        }
        let w_structobj = super::allocator::allocate(
            allocator.as_deref_mut(),
            datasize,
            roots.get(item_slot),
            varsize_length,
        )?;
        let struct_slot = cdata_slot;
        let _ = roots.pin_root(w_structobj);
        let ptr = cdataobj::cdata_arg(roots.get(struct_slot))?.ptr;
        let w_cdata =
            cdataobj::new_cdata_ptr_to_struct(ptr, roots.get(ctype_slot), roots.get(struct_slot));
        let ptr_slot = struct_slot + 1;
        let _ = roots.pin_root(w_cdata);
        if !unsafe { pyre_object::pyobject::is_none(roots.get(init_slot)) } {
            let cdata = cdataobj::cdata_arg(roots.get(ptr_slot))?;
            let item = ctypeobj::ctype_arg(roots.get(item_slot))?;
            unsafe {
                ctypeobj::convert_from_object(item, cdata.ptr, roots.get(init_slot))?;
            }
        }
        return Ok(roots.get(ptr_slot));
    }
    if ct.is_char_or_unichar_ptr_or_array() {
        // Room for the null character `newp` always adds.
        datasize *= 2;
    }
    let w_cdata = super::allocator::allocate(allocator, datasize, roots.get(ctype_slot), -1)?;
    let _ = roots.pin_root(w_cdata);
    let w_init = roots.get(init_slot);
    if !unsafe { pyre_object::pyobject::is_none(w_init) } {
        let cdata = cdataobj::cdata_arg(roots.get(cdata_slot))?;
        let ct = ctypeobj::ctype_arg(cdata.ctype)?;
        let item = item_of(ct)?;
        unsafe { ctypeobj::convert_from_object(item, cdata.ptr, w_init)? };
    }
    Ok(roots.get(cdata_slot))
}

/// `W_CTypeArray.newp`.
pub fn array_newp(
    w_ctype: PyObjectRef,
    w_init: PyObjectRef,
    w_allocator: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    let roots = pyre_object::gc_roots::push_roots();
    let init_slot = roots.pin_roots(&[w_init, w_allocator, w_ctype]);
    let allocator_slot = init_slot + 1;
    let ctype_slot = init_slot + 2;
    let (w_init, datasize, length) = if ct.size < 0 {
        let (w_init, length) = new_array_length(ct, roots.get(init_slot))?;
        let item = item_of(ct)?;
        let datasize = length.checked_mul(item.size).ok_or_else(array_overflow)?;
        (w_init, datasize, length)
    } else {
        (roots.get(init_slot), ct.size, ct.length)
    };
    let init_slot2 = init_slot + 3;
    let _ = roots.pin_root(w_init);
    let w_cdata = super::allocator::allocate(
        super::allocator::W_Allocator::from_obj(roots.get(allocator_slot)),
        datasize,
        roots.get(ctype_slot),
        length,
    )?;
    let cdata_slot = init_slot2 + 1;
    let _ = roots.pin_root(w_cdata);
    let w_init = roots.get(init_slot2);
    if !unsafe { pyre_object::pyobject::is_none(w_init) } {
        let cdata = cdataobj::cdata_arg(roots.get(cdata_slot))?;
        let ct = ctypeobj::ctype_arg(cdata.ctype)?;
        unsafe { array_convert_from_object(ct, cdata.ptr as *mut u8, w_init)? };
    }
    Ok(roots.get(cdata_slot))
}

/// `W_CTypeArray.get_new_array_length`.
pub fn new_array_length(
    ct: &W_CType,
    mut w_value: PyObjectRef,
) -> Result<(PyObjectRef, i64), PyError> {
    unsafe {
        if pyre_object::pyobject::is_list(w_value) || pyre_object::pyobject::is_tuple(w_value) {
            let length = pyre_object::with_roots!(w_value => pyre_interpreter::runtime_ops::sequence_len(w_value))?
                as i64;
            return Ok((w_value, length));
        }
        if pyre_object::bytesobject::is_bytes(w_value) {
            // A string initializer carries its own null terminator.
            let s = pyre_object::bytesobject::w_bytes_data(w_value);
            return Ok((w_value, s.len() as i64 + 1));
        }
        if pyre_object::unicodeobject::is_str(w_value) {
            let value = pyre_object::w_str_get_wtf8(w_value);
            let item = item_of(ct)?;
            let length: i64 = if item.size == 2 {
                value
                    .code_points()
                    .map(|p| if p.to_u32() > 0xFFFF { 2 } else { 1 })
                    .sum()
            } else {
                value.code_points().count() as i64
            };
            return Ok((w_value, length + 1));
        }
    }
    // `__index__` is arbitrary Python, so the name the TypeError reports is
    // read off the object before the conversion rather than after it.
    let got = pyre_interpreter::type_methods::arg_type_name(w_value);
    let length =
        pyre_interpreter::baseobjspace::index_int_w_preserve_negative(w_value).map_err(|e| {
            if e.kind == pyre_interpreter::PyErrorKind::TypeError {
                PyError::type_error(format!(
                    "expected new array length or list/tuple/str, not {got}"
                ))
            } else {
                e
            }
        })?;
    if length < 0 {
        return Err(PyError::value_error("negative array length"));
    }
    Ok((pyre_object::w_none(), length))
}

fn array_overflow() -> PyError {
    PyError::overflow_error("array size would overflow a ssize_t")
}

/// `W_CTypePointer.add` and `W_CTypeArray.add`.
pub fn add(w_ctype: PyObjectRef, cdata: *mut u8, i: i64) -> Result<PyObjectRef, PyError> {
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    let item = item_of(ct)?;
    let mut item_size = item.size;
    if item_size < 0 {
        if ct.has(ctypeobj::CTypeFlags::VOID_PTR) {
            item_size = 1;
        } else {
            return Err(PyError::type_error(format!(
                "ctype '{}' points to items of unknown size",
                ct.name()
            )));
        }
    }
    let target = unsafe { cdata.offset((i * item_size) as isize) };
    // An array's arithmetic result is a pointer, not another array.
    let w_result_ctype = if ct.kind == ctypeobj::KIND_ARRAY {
        ct.ctptr
    } else {
        w_ctype
    };
    Ok(cdataobj::new_cdata(target as usize, w_result_ctype))
}

/// `W_CTypePtrOrArray.string`.
pub fn string(mut w_cdata: PyObjectRef, maxlen: i64) -> Result<PyObjectRef, PyError> {
    let cdata = pyre_object::with_roots!(w_cdata => cdataobj::cdata_arg(w_cdata))?;
    let ct = ctypeobj::ctype_at(cdata.ctype)
        .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
    let item = item_of(ct)?;
    if !item.is_primitive() || item.kind == ctypeobj::KIND_PRIM_BOOL {
        return Err(ctypeobj::unexpected_string_argument(ct));
    }
    if cdata.ptr == 0 {
        let w_repr = pyre_interpreter::builtins::builtin_repr(&[w_cdata])?;
        return Err(PyError::runtime_error(
            pyre_interpreter::display::wtf8_format!("cannot use string() on ", unsafe {
                pyre_object::w_str_get_wtf8(w_repr)
            }),
        ));
    }
    let mut length = maxlen;
    if length < 0 && ct.kind == ctypeobj::KIND_ARRAY {
        length = cdata.array_length()?;
    }
    // A pointer to a one-byte type builds a `bytes` up to the first NUL.
    if item.size == 1 {
        let bytes = unsafe { read_until_nul(cdata.ptr as *const u8, length) };
        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(bytes));
    }
    if ct.is_unichar_ptr_or_array() {
        let measured = unsafe {
            if item.size == 2 {
                super::wchar_helper::measure_length_16(cdata.ptr as *const u8, length)?
            } else {
                super::wchar_helper::measure_length_32(cdata.ptr as *const u8, length)?
            }
        };
        return unpack_ptr(ct, item, cdata.ptr as *mut u8, measured);
    }
    Err(ctypeobj::unexpected_string_argument(ct))
}

/// `rffi.charp2str` / `charp2strn` — the bytes before the first NUL, and at
/// most `length` of them when one was given.
///
/// # Safety
/// `ptr` must be readable up to the first NUL, or for `length` bytes.
unsafe fn read_until_nul(ptr: *const u8, length: i64) -> &'static [u8] {
    let mut n = 0usize;
    let limit = if length < 0 {
        usize::MAX
    } else {
        length as usize
    };
    while n < limit && unsafe { ptr.add(n).read() } != 0 {
        n += 1;
    }
    unsafe { std::slice::from_raw_parts(ptr, n) }
}

/// `W_CType.unpack_ptr` and the primitive fast paths over it.
pub fn unpack_ptr(
    ct: &W_CType,
    item: &W_CType,
    ptr: *mut u8,
    length: i64,
) -> Result<PyObjectRef, PyError> {
    if item.size < 0 {
        return Err(PyError::value_error(format!(
            "'{}' points to items of unknown size",
            ct.name()
        )));
    }
    // `W_CTypePrimitiveChar.unpack_ptr` — a `bytes`, not a list.
    if item.kind == ctypeobj::KIND_PRIM_CHAR {
        let bytes = unsafe { std::slice::from_raw_parts(ptr, length.max(0) as usize) };
        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(bytes));
    }
    // `W_CTypePrimitiveUniChar.unpack_ptr` — a `str`.
    if item.kind == ctypeobj::KIND_PRIM_UNICHAR {
        return unpack_wide_string(item, ptr, length);
    }
    if let Some(w_list) = unsafe { super::ctypeprim::unpack_list_of_items(item, ptr, length)? } {
        return Ok(w_list);
    }
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for i in 0..length {
        let element = unsafe { ptr.offset((i * item.size) as isize) };
        let _ = roots.pin_root(unsafe { ctypeobj::convert_to_object(item, element as usize)? });
    }
    let items = (0..length as usize).map(|i| roots.get(base + i)).collect();
    Ok(pyre_object::w_list_new(items))
}

/// `wchar_helper.utf8_from_char16` / `utf8_from_char32`.
fn unpack_wide_string(item: &W_CType, ptr: *mut u8, length: i64) -> Result<PyObjectRef, PyError> {
    let (out, _) = unsafe {
        if item.size == 2 {
            super::wchar_helper::utf8_from_char16(ptr, length)?
        } else {
            super::wchar_helper::utf8_from_char32(ptr, length)?
        }
    };
    Ok(pyre_object::w_str_from_wtf8_managed(out))
}

/// `W_CTypePtrOrArray.ctitem`, which every pointer and array has.
pub fn item_of(ct: &W_CType) -> Result<&'static mut W_CType, PyError> {
    match ctypeobj::ctype_at(ct.ctitem) {
        Some(item) => Ok(item),
        None => Err(PyError::system_error(
            "pointer or array without an item type",
        )),
    }
}

// ── passing a pointer as a call argument ────────────────────────────────

/// `W_CTypePointer.convert_argument_from_object` — write the pointer into the
/// exchange slot and record, in the byte before it, what the call owes
/// afterwards.
///
/// # Safety
/// `cdata` must be an exchange-buffer argument slot with a byte of its own
/// just before it.
pub unsafe fn pointer_convert_argument_from_object(
    ct: &W_CType,
    cdata: *mut u8,
    mut w_ob: PyObjectRef,
) -> Result<bool, PyError> {
    use super::ctypefunc::{MUSTFREE_FREE, MUSTFREE_NOTHING, set_mustfree_flag};

    let mut result = MUSTFREE_NOTHING;
    if W_CData::from_obj(w_ob).is_none() {
        if ct.has(ctypeobj::CTypeFlags::ACCEPT_STR)
            && let Some(offset) = super::func::OffsetInBytes::from_obj(w_ob)
        {
            // `process_str_from_offset_in_bytes` hands C the string's own
            // characters only where `STR.chars` carries the final null
            // `extra_item_after_alloc` reserves.  A `BytesBlock` reserves
            // none, so this is that function's copy arm: the tail from
            // `offset`, null-terminated, freed after the call.
            let value = unsafe { pyre_object::bytesobject::w_bytes_data(offset.w_bytes) };
            let tail = usize::try_from(offset.offset)
                .ok()
                .and_then(|start| value.get(start..))
                .unwrap_or(&[]);
            let buf = cdataobj::raw_alloc(tail.len() as i64 + 1, false)?;
            unsafe {
                std::ptr::copy_nonoverlapping(tail.as_ptr(), buf as *mut u8, tail.len());
                (buf as *mut u8).add(tail.len()).write(0);
                cdata.cast::<usize>().write_unaligned(buf);
            }
            unsafe { set_mustfree_flag(cdata, MUSTFREE_FREE) };
            return Ok(true);
        }
        if ct.has(ctypeobj::CTypeFlags::ACCEPT_STR)
            && unsafe { pyre_object::bytesobject::is_bytes(w_ob) }
        {
            // A `bytes` passed to a `char *` argument reaches C as a copy
            // with its own null terminator; RPython instead pins or hands
            // over the string's own characters, which pyre has no equivalent
            // of because the collector may move them.
            return unsafe { accept_movable_str(ct, cdata, w_ob) };
        }
        result = unsafe {
            pyre_object::with_roots!(w_ob => prepare_pointer_call_argument(ct, cdata, w_ob))?
        };
    }
    if result == MUSTFREE_NOTHING {
        unsafe { pointer_convert_from_object(ct, cdata, w_ob)? };
    }
    unsafe { set_mustfree_flag(cdata, result) };
    // `convert_argument_from_object` answers with the flag itself, and the
    // caller widens the cleanup window to any non-zero one.
    Ok(result != MUSTFREE_NOTHING)
}

/// `W_CTypePointer.accept_movable_str`.
///
/// # Safety
/// As [`pointer_convert_argument_from_object`].
unsafe fn accept_movable_str(
    ct: &W_CType,
    cdata: *mut u8,
    w_ob: PyObjectRef,
) -> Result<bool, PyError> {
    use super::ctypefunc::{MUSTFREE_FREE, set_mustfree_flag};

    let value = unsafe { pyre_object::bytesobject::w_bytes_data(w_ob) };
    if item_of(ct)?.kind == ctypeobj::KIND_PRIM_BOOL {
        must_be_string_of_zero_or_one(value)?;
    }
    let buf = cdataobj::raw_alloc(value.len() as i64 + 1, false)?;
    unsafe {
        // `misc.py write_string_as_charp` is `@jit.dont_look_inside`; the
        // copy is its opaque half, taken because the collector may move the
        // bytes and there is no non-moving pin.
        misc::raw_memcopy_opaque(value.as_ptr() as usize, buf, value.len());
        (buf as *mut u8).add(value.len()).write(0);
        cdataobj::raw_write_ptr(cdata as usize, buf);
        set_mustfree_flag(cdata, MUSTFREE_FREE);
    }
    Ok(true)
}

/// `W_CTypePointer._must_be_string_of_zero_or_one`.
fn must_be_string_of_zero_or_one(value: &[u8]) -> Result<(), PyError> {
    if value.iter().any(|&c| c > 1) {
        return Err(PyError::value_error(
            "an array of _Bool can only contain \\x00 or \\x01",
        ));
    }
    Ok(())
}

/// `W_CTypePointer._prepare_pointer_call_argument` — a list, tuple, bytes or
/// str argument becomes an array this call owns for its duration.
///
/// # Safety
/// As [`pointer_convert_argument_from_object`].
unsafe fn prepare_pointer_call_argument(
    ct: &W_CType,
    cdata: *mut u8,
    mut w_init: PyObjectRef,
) -> Result<u8, PyError> {
    use super::ctypefunc::{MUSTFREE_FREE, MUSTFREE_NOTHING};

    let item = item_of(ct)?;
    let length = unsafe {
        if pyre_object::pyobject::is_list(w_init) || pyre_object::pyobject::is_tuple(w_init) {
            pyre_object::with_roots!(w_init => pyre_interpreter::runtime_ops::sequence_len(w_init))?
                as i64
        } else if pyre_object::bytesobject::is_bytes(w_init) {
            // From a string, we add the null terminator.
            pyre_object::bytesobject::w_bytes_data(w_init).len() as i64 + 1
        } else if pyre_object::unicodeobject::is_str(w_init) {
            let value = pyre_object::w_str_get_wtf8(w_init);
            let n: i64 = if item.size == 2 {
                value
                    .code_points()
                    .map(|p| if p.to_u32() > 0xFFFF { 2 } else { 1 })
                    .sum()
            } else {
                value.code_points().count() as i64
            };
            n + 1
        } else if ct.has(ctypeobj::CTypeFlags::FILE_PTR) {
            let file = prepare_file(w_init)?;
            if file.is_null() {
                return Ok(MUSTFREE_NOTHING);
            }
            cdataobj::raw_write_ptr(cdata as usize, file as usize);
            return Ok(super::ctypefunc::MUSTFREE_FILE);
        } else {
            return Ok(MUSTFREE_NOTHING);
        }
    };
    let itemsize = if item.size > 0 {
        item.size
    } else if item.kind == ctypeobj::KIND_VOID {
        1
    } else {
        return Ok(MUSTFREE_NOTHING);
    };
    let datasize = length
        .checked_mul(itemsize)
        .filter(|&n| n >= 0)
        .ok_or_else(|| PyError::overflow_error("array size would overflow a ssize_t"))?;
    let buf = cdataobj::raw_alloc(datasize, true)?;
    if let Err(e) = unsafe { convert_array_from_object(ct, buf as *mut u8, w_init) } {
        cdataobj::raw_free(buf);
        return Err(e);
    }
    cdataobj::raw_write_ptr(cdata as usize, buf);
    Ok(MUSTFREE_FREE)
}

// ── the C `FILE` a stream is cast to ────────────────────────────────────

unsafe extern "C" {
    #[cfg_attr(windows, link_name = "_fdopen")]
    #[cfg_attr(not(windows), link_name = "fdopen")]
    fn rffi_fdopen(fd: c_int, mode: *const c_char) -> *mut c_void;
    #[cfg_attr(windows, link_name = "_dup")]
    #[cfg_attr(not(windows), link_name = "dup")]
    fn rffi_dup(fd: c_int) -> c_int;
    #[cfg_attr(windows, link_name = "_close")]
    #[cfg_attr(not(windows), link_name = "close")]
    fn rffi_close(fd: c_int) -> c_int;
    fn setbuf(stream: *mut c_void, buf: *mut c_char);
    fn fclose(stream: *mut c_void) -> c_int;
}

/// The slot `W_IOBase.cffi_fileobj` is here: the address `fdopen` returned,
/// as an ordinary integer.
const CFFI_FILEOBJ_SLOT: &str = "__cffi_fileobj__";

/// Every C `FILE` this module has opened.  A stream names its handle by
/// address, and only an address this module minted is ever closed, so a
/// value written into the slot from application code is inert.
fn open_files() -> &'static Mutex<HashSet<usize>> {
    static OPEN: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();
    OPEN.get_or_init(|| Mutex::new(HashSet::new()))
}

fn lock_open_files() -> std::sync::MutexGuard<'static, HashSet<usize>> {
    open_files()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The C `FILE` this stream already holds, if it holds one this module made.
fn held_file(w_fileobj: PyObjectRef) -> Option<usize> {
    let w_held = pyre_interpreter::baseobjspace::getdictvalue_native(w_fileobj, CFFI_FILEOBJ_SLOT)?;
    if !unsafe { pyre_object::pyobject::is_int(w_held) } {
        return None;
    }
    let address = unsafe { pyre_object::intobject::w_int_get_value(w_held) } as usize;
    lock_open_files().contains(&address).then_some(address)
}

/// `W_CTypePointer.prepare_file` — a stream answers with the C `FILE` over
/// it, and anything else with a null pointer.
fn prepare_file(mut w_ob: PyObjectRef) -> Result<*mut c_void, PyError> {
    if !pyre_object::with_roots!(w_ob => pyre_interpreter::baseobjspace::isinstance(
        w_ob,
        pyre_interpreter::module::_io::io_base_type(),
    ))? {
        return Ok(std::ptr::null_mut());
    }
    prepare_file_argument(w_ob)
}

/// `ctypeptr.py prepare_file_argument`.
fn prepare_file_argument(w_fileobj: PyObjectRef) -> Result<*mut c_void, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let file_slot = roots.base();
    let _ = roots.pin_root(w_fileobj);
    pyre_interpreter::baseobjspace::call_method_result(roots.get(file_slot), "flush", &[])?;
    if let Some(held) = held_file(roots.get(file_slot)) {
        return Ok(held as *mut c_void);
    }

    let w_fd =
        pyre_interpreter::baseobjspace::call_method_result(roots.get(file_slot), "fileno", &[])?;
    let fd_slot = file_slot + 1;
    let _ = roots.pin_root(w_fd);
    let fd = pyre_interpreter::baseobjspace::int_w(roots.get(fd_slot))?;
    if fd < 0 {
        return Err(PyError::value_error("file has no OS file descriptor"));
    }
    let w_mode = pyre_interpreter::baseobjspace::getattr_str(roots.get(file_slot), "mode")?;
    let mode_slot = fd_slot + 1;
    let _ = roots.pin_root(w_mode);
    let mode = pyre_interpreter::baseobjspace::text_w(roots.get(mode_slot))?;
    let mode = CString::new(mode.as_bytes())
        .map_err(|_| PyError::value_error("embedded null character"))?;

    // `os.dup` — the C `FILE` gets a descriptor of its own, so closing either
    // side leaves the other usable.
    let fd = unsafe { rffi_dup(fd as c_int) };
    if fd < 0 {
        return Err(PyError::os_error_with_errno(errno(), "dup failed"));
    }
    let llf = unsafe { rffi_fdopen(fd, mode.as_ptr()) };
    if llf.is_null() {
        let saved = errno();
        unsafe { rffi_close(fd) };
        return Err(PyError::os_error_with_errno(saved, "fdopen failed"));
    }
    unsafe { setbuf(llf, std::ptr::null_mut()) };

    let address = llf as usize;
    lock_open_files().insert(address);
    let handle_slot = mode_slot + 1;
    let _ = roots.pin_root(pyre_object::w_int_new(address as i64));
    pyre_interpreter::baseobjspace::setdictvalue(
        roots.get(file_slot),
        CFFI_FILEOBJ_SLOT,
        roots.get(handle_slot),
    )?;
    Ok(llf)
}

/// `CffiFileObj.close`, which `W_IOBase.close_w` runs before its flush.
pub fn close_cffi_fileobj(mut w_fileobj: PyObjectRef) {
    let Some(address) = pyre_object::with_roots!(w_fileobj => held_file(w_fileobj)) else {
        return;
    };
    lock_open_files().remove(&address);
    let _ = pyre_interpreter::baseobjspace::setdictvalue(
        w_fileobj,
        CFFI_FILEOBJ_SLOT,
        pyre_object::w_none(),
    );
    unsafe { fclose(address as *mut c_void) };
}

/// `rposix.get_saved_errno()`.
fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}
