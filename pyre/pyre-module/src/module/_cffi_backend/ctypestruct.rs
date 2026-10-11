//! Structs and unions — PyPy: `pypy/module/_cffi_backend/ctypestruct.py`.
//!
//! `W_CTypeStruct` and `W_CTypeUnion` differ only in the `kind` they report
//! and in how `complete_struct_or_union` lays their fields out, so both are
//! [`ctypeobj::KIND_STRUCT`] / [`ctypeobj::KIND_UNION`] arms here.  `W_CField`
//! is a type of its own, as in PyPy.

use pyre_interpreter::PyError;
use pyre_object::PyObjectRef;

use super::cdataobj::{self, W_CData};
use super::ctypeobj::{self, W_CType};
use super::misc;

/// `W_CField.BS_REGULAR`.
pub const BS_REGULAR: i64 = -1;
/// `W_CField.BS_EMPTY_ARRAY`.
pub const BS_EMPTY_ARRAY: i64 = -2;
/// `W_CField.BF_IGNORE_IN_CTOR`.
pub const BF_IGNORE_IN_CTOR: i64 = 0x01;

/// `ctypestruct.py W_CField`.
#[pyre_interpreter::pyre_class("_cffi_backend.CField")]
#[derive(Default)]
pub struct W_CField {
    /// `W_CField.ctype`.
    pub ctype: PyObjectRef,
    /// `W_CField.offset`.
    pub offset: i64,
    /// `W_CField.bitshift` — non-negative for a bitfield, otherwise
    /// [`BS_REGULAR`] or [`BS_EMPTY_ARRAY`].
    pub bitshift: i64,
    /// `W_CField.bitsize`.
    pub bitsize: i64,
    /// `W_CField.flags` — the `BF_*` above.
    pub flags: i64,
}

impl W_CField {
    /// `W_CField.is_bitfield`.
    pub fn is_bitfield(&self) -> bool {
        self.bitshift >= 0
    }

    fn ctype_ref(&self) -> Result<&'static mut W_CType, PyError> {
        ctypeobj::ctype_at(self.ctype).ok_or_else(|| PyError::system_error("field without a ctype"))
    }
}

/// `W_CField.__init__`.  A field is born non-moving for the same reason a
/// ctype is: the struct that owns it hands out borrowed references to it.
pub fn new_cfield(
    w_ctype: PyObjectRef,
    offset: i64,
    bitshift: i64,
    bitsize: i64,
    flags: i64,
) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let ctype_slot = roots.base();
    let _ = roots.pin_root(w_ctype);
    W_CField::allocate_stable(W_CField {
        ctype: roots.get(ctype_slot),
        offset,
        bitshift,
        bitsize,
        flags,
        ..Default::default()
    })
}

/// `W_CField.make_shifted` — the same field seen from an enclosing struct.
pub fn make_shifted(field: &W_CField, offset: i64, fflags: i64) -> PyObjectRef {
    new_cfield(
        field.ctype,
        offset + field.offset,
        field.bitshift,
        field.bitsize,
        field.flags | fflags,
    )
}

/// `W_CField.read`.
///
/// # Safety
/// `cdata` must point at the start of the struct this field belongs to.
pub unsafe fn read(
    field: &W_CField,
    cdata: *mut u8,
    w_cdata: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let ct = field.ctype_ref()?;
    let p = unsafe { cdata.offset(field.offset as isize) };
    match field.bitshift {
        BS_REGULAR => unsafe { ctypeobj::convert_to_object(ct, p as usize) },
        BS_EMPTY_ARRAY => {
            // A variable-length array reads as far as the owning allocation
            // reaches; without one it decays to the pointer type.
            let item = super::ctypeptr::item_of(ct)?;
            if let Some(structobj) = cdataobj::structobj_of(w_cdata)
                && item.size > 0
            {
                let size = structobj.length - field.offset;
                if size >= 0 {
                    return Ok(cdataobj::new_cdata_sliced(
                        p as usize,
                        ct.as_object(),
                        size / item.size,
                    ));
                }
            }
            Ok(cdataobj::new_cdata(p as usize, ct.ctptr))
        }
        _ => unsafe { convert_bitfield_to_object(field, ct, p) },
    }
}

/// `W_CField.write`.
///
/// # Safety
/// `cdata` must point at the start of the struct this field belongs to.
pub unsafe fn write(field: &W_CField, cdata: *mut u8, w_ob: PyObjectRef) -> Result<(), PyError> {
    let ct = field.ctype_ref()?;
    let p = unsafe { cdata.offset(field.offset as isize) };
    if field.is_bitfield() {
        unsafe { convert_bitfield_from_object(field, ct, p, w_ob) }
    } else {
        unsafe { ctypeobj::convert_from_object(ct, p as usize, w_ob) }
    }
}

/// `W_CField.add_varsize_length`.
fn add_varsize_length(
    field: &W_CField,
    itemsize: i64,
    varsizelength: i64,
    optvarsize: i64,
) -> Result<i64, PyError> {
    let size = itemsize
        .checked_mul(varsizelength)
        .and_then(|varsize| field.offset.checked_add(varsize))
        .filter(|&size| size >= 0)
        .ok_or_else(|| PyError::overflow_error("array size would overflow a ssize_t"))?;
    Ok(size.max(optvarsize))
}

/// `W_CField.write_v` — the C99 var-sized-array case of a field write.
///
/// # Safety
/// `cdata` must point at the struct when `optvarsize` is -1; it is unused
/// otherwise, because that mode only measures.
unsafe fn write_v(
    field: &W_CField,
    cdata: *mut u8,
    w_ob: PyObjectRef,
    optvarsize: i64,
) -> Result<i64, PyError> {
    let ct = field.ctype_ref()?;
    let roots = pyre_object::gc_roots::push_roots();
    let ob_slot = roots.base();
    let _ = roots.pin_root(w_ob);
    if ct.kind == ctypeobj::KIND_ARRAY && ct.length < 0 {
        let item = super::ctypeptr::item_of(ct)?;
        // `get_new_array_length` reaches `__index__`, so the object it hands
        // back is what the write below must read.
        let (w_next, varsizelength) = super::ctypeptr::new_array_length(ct, roots.get(ob_slot))?;
        if optvarsize != -1 {
            return add_varsize_length(field, item.size, varsizelength, optvarsize);
        }
        // An integer initializer leaves the content uninitialized; it was
        // zeroed when the struct was allocated.
        if unsafe { pyre_object::pyobject::is_none(w_next) } {
            return Ok(optvarsize);
        }
        roots.set(ob_slot, w_next);
    }
    if optvarsize == -1 {
        unsafe { write(field, cdata, roots.get(ob_slot))? };
        return Ok(optvarsize);
    }
    if ct.is_struct_or_union()
        && ct.has(ctypeobj::CTypeFlags::WITH_VAR_ARRAY)
        && W_CData::from_obj(roots.get(ob_slot)).is_none()
    {
        let subsize = unsafe {
            convert_struct_from_object(ct, std::ptr::null_mut(), roots.get(ob_slot), ct.size)?
        };
        return add_varsize_length(field, 1, subsize, optvarsize);
    }
    Ok(optvarsize)
}

/// `W_CField.convert_bitfield_to_object`.
///
/// # Safety
/// `cdata` must be readable for the field ctype's size.
unsafe fn convert_bitfield_to_object(
    field: &W_CField,
    ct: &W_CType,
    cdata: *const u8,
) -> Result<PyObjectRef, PyError> {
    let bitsize = field.bitsize as u32;
    let raw = unsafe { misc::read_raw_unsigned_data(cdata as usize, ct.size)? };
    // A field as wide as its own type shifts by the full width.  The masks are
    // built with the shift the hardware performs — which leaves the operand
    // alone — because that is what the translated `r_ulonglong` arithmetic
    // these expressions come from compiles to.
    let valuemask = 1u64.wrapping_shl(bitsize).wrapping_sub(1);
    if ct.kind == ctypeobj::KIND_PRIM_SIGNED {
        let shiftforsign = 1u64 << (bitsize - 1);
        let value = ((raw >> field.bitshift).wrapping_add(shiftforsign)) & valuemask;
        return Ok(pyre_object::w_int_new(
            (value as i64).wrapping_sub(shiftforsign as i64),
        ));
    }
    if ct.kind != ctypeobj::KIND_PRIM_UNSIGNED && !ct.is_char_or_unichar() {
        return Err(PyError::system_error(format!(
            "cannot read the bit field '{}'",
            ct.name()
        )));
    }
    let value = (raw >> field.bitshift) & valuemask;
    // A char or unichar bitfield always fits, which is what the mixin's
    // `value_fits_long = True` says.
    if ct.is_char_or_unichar() {
        return Ok(pyre_object::w_int_new(value as i64));
    }
    Ok(super::ctypeprim::unsigned_as_object(ct, value))
}

/// `W_CField.convert_bitfield_from_object`.
///
/// # Safety
/// `cdata` must be writable for the field ctype's size.
unsafe fn convert_bitfield_from_object(
    field: &W_CField,
    ct: &W_CType,
    cdata: *mut u8,
    w_ob: PyObjectRef,
) -> Result<(), PyError> {
    let bitsize = field.bitsize as u32;
    let value = misc::as_long_long(w_ob)?;
    let is_signed = ct.kind == ctypeobj::KIND_PRIM_SIGNED;
    let (fmin, fmax) = if is_signed {
        // As in `convert_bitfield_to_object`, the full-width shift is the one
        // the hardware performs.
        let half = 1i64.wrapping_shl(bitsize - 1);
        let fmax = half.wrapping_sub(1);
        // "int x:1" is allowed to receive 1.
        (half.wrapping_neg(), if fmax == 0 { 1 } else { fmax })
    } else {
        (0, 1u64.wrapping_shl(bitsize).wrapping_sub(1) as i64)
    };
    if value < fmin || value > fmax {
        return Err(PyError::overflow_error(format!(
            "value {value} outside the range allowed by the bit field width: {fmin} <= x <= {fmax}"
        )));
    }
    let shift = field.bitshift as u32;
    let rawmask = 1u64
        .wrapping_shl(bitsize)
        .wrapping_sub(1)
        .wrapping_shl(shift);
    let rawvalue = (value as u64).wrapping_shl(shift);
    let mut raw = unsafe { misc::read_raw_unsigned_data(cdata as usize, ct.size)? };
    raw = (raw & !rawmask) | (rawvalue & rawmask);
    unsafe {
        if is_signed {
            misc::write_raw_signed_data(cdata as usize, raw as i64, ct.size)
        } else {
            misc::write_raw_unsigned_data(cdata as usize, raw, ct.size)
        }
    }
}

// ── the ctype's own overrides ───────────────────────────────────────────

/// `W_CTypeStructOrUnion.convert_to_object` — the struct in place, not a copy.
///
/// # Safety
/// `cdata` must point at a struct of this type.
pub unsafe fn convert_to_object(ct: &W_CType, cdata: *const u8) -> Result<PyObjectRef, PyError> {
    ct.check_complete(false)?;
    Ok(cdataobj::new_cdata(cdata as usize, ct.as_object()))
}

/// `W_CTypeStructOrUnion.copy_and_convert_to_object`.
///
/// # Safety
/// `source` must be readable for `ct.size` bytes.
pub unsafe fn copy_and_convert_to_object(
    ct: &W_CType,
    source: *const u8,
) -> Result<PyObjectRef, PyError> {
    ct.check_complete(false)?;
    unsafe { cdataobj::new_cdata_copy(ct.as_object(), source, ct.size) }
}

/// `W_CTypeStructOrUnion._copy_from_same`.
///
/// # Safety
/// `cdata` must be writable for `ct.size` bytes.
unsafe fn copy_from_same(ct: &W_CType, cdata: *mut u8, w_ob: PyObjectRef) -> bool {
    if let Some(source) = W_CData::from_obj(w_ob)
        && source.ctype == ct.as_object()
        && ct.size >= 0
    {
        misc::raw_memcopy(source.ptr, cdata as usize, ct.size as usize);
        return true;
    }
    false
}

/// `W_CTypeStructOrUnion.convert_from_object`.
///
/// # Safety
/// `cdata` must be writable for `ct.size` bytes.
pub unsafe fn convert_from_object(
    ct: &W_CType,
    cdata: *mut u8,
    w_ob: PyObjectRef,
) -> Result<(), PyError> {
    if unsafe { copy_from_same(ct, cdata, w_ob) } {
        return Ok(());
    }
    unsafe { convert_struct_from_object(ct, cdata, w_ob, -1)? };
    Ok(())
}

/// `W_CTypeStructOrUnion.convert_struct_from_object`.
///
/// With `optvarsize` at -1 this writes the struct at `cdata`; otherwise it
/// writes nothing and only reports how large a var-sized struct would be.
///
/// # Safety
/// `cdata` must be writable for `ct.size` bytes when `optvarsize` is -1.
pub unsafe fn convert_struct_from_object(
    ct: &W_CType,
    cdata: *mut u8,
    mut w_ob: PyObjectRef,
    optvarsize: i64,
) -> Result<i64, PyError> {
    pyre_object::with_roots!(w_ob => ct.force_lazy_struct())?;
    let mut optvarsize = optvarsize;
    let roots = pyre_object::gc_roots::push_roots();
    let ob_slot = roots.base();
    let _ = roots.pin_root(w_ob);
    let is_seq = unsafe {
        pyre_object::pyobject::is_list(roots.get(ob_slot))
            || pyre_object::pyobject::is_tuple(roots.get(ob_slot))
    };
    if is_seq {
        let items = pyre_interpreter::baseobjspace::unpackiterable(roots.get(ob_slot), -1)?;
        let items_slot = pyre_object::gc_roots::shadow_stack_len();
        for &item in &items {
            let _ = roots.pin_root(item);
        }
        let fields = fields_list_of(ct)?;
        let mut j = 0usize;
        for i in 0..items.len() {
            loop {
                let field = fields.get(j).and_then(|&f| W_CField::from_obj(f));
                match field {
                    Some(field) if field.flags & BF_IGNORE_IN_CTOR != 0 => j += 1,
                    Some(field) => {
                        optvarsize = unsafe {
                            write_v(field, cdata, roots.get(items_slot + i), optvarsize)?
                        };
                        j += 1;
                        break;
                    }
                    None => {
                        return Err(unsafe {
                            PyError::from_exc_object(too_many_struct_initializers(ct, items.len()))
                        });
                    }
                }
            }
        }
        return Ok(optvarsize);
    }
    if unsafe { pyre_object::pyobject::is_dict(roots.get(ob_slot)) } {
        let keys = pyre_interpreter::baseobjspace::fixedview(roots.get(ob_slot), -1)?;
        let keys_slot = pyre_object::gc_roots::shadow_stack_len();
        for &key in &keys {
            let _ = roots.pin_root(key);
        }
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(pyre_object::PY_NULL);
        for i in 0..keys.len() {
            let w_key = roots.get(keys_slot + i);
            let key = pyre_interpreter::baseobjspace::text_w(w_key)?;
            let Ok(field) = ctypeobj::getcfield(ct, key, "write") else {
                return Err(PyError::key_error_with_key(w_key));
            };
            roots.set(
                value_slot,
                pyre_interpreter::baseobjspace::getitem(
                    roots.get(ob_slot),
                    roots.get(keys_slot + i),
                )?,
            );
            optvarsize = unsafe { write_v(field, cdata, roots.get(value_slot), optvarsize)? };
        }
        return Ok(optvarsize);
    }
    let expected: &'static &'static str = if optvarsize == -1 {
        &"list or tuple or dict or struct-cdata"
    } else {
        &"list or tuple or dict"
    };
    Err(unsafe {
        PyError::from_exc_object(ctypeobj::convert_error(ct, expected, roots.get(ob_slot)))
    })
}

/// `W_CTypeStructOrUnion.convert_struct_from_object` — `oefmt`.
#[majit_macros::dont_look_inside]
pub(crate) fn too_many_struct_initializers(ct: &W_CType, got: usize) -> PyObjectRef {
    PyError::value_error(format!(
        "too many initializers for '{}' (got {})",
        ct.name(),
        got
    ))
    .to_exc_object()
}

/// The `W_CField`s of a completed struct, in declaration order.
pub fn fields_list_of(ct: &W_CType) -> Result<Vec<PyObjectRef>, PyError> {
    if ct.fields_list.is_null() {
        return Err(PyError::system_error(format!(
            "'{}' has no field list",
            ct.name()
        )));
    }
    Ok(unsafe { pyre_object::listobject::w_list_items_copy_as_vec(ct.fields_list) })
}

/// `W_CTypeStructOrUnion._fget('f')` — `[(name, field), ...]` in declaration
/// order.  A field the struct kept but never named — an anonymous member
/// spliced in from a nested struct — leaves its slot as `None`.
pub fn fget_fields(ct: &W_CType) -> Result<PyObjectRef, PyError> {
    if ct.size < 0 {
        return Ok(pyre_object::w_none());
    }
    ct.force_lazy_struct()?;
    let fields = fields_list_of(ct)?;
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for _ in 0..fields.len() {
        let _ = roots.pin_root(pyre_object::w_none());
    }
    for (name, w_field) in fields_dict_items(ct)? {
        if let Some(i) = fields.iter().position(|&f| f == w_field) {
            let mut pair = pyre_object::gc_roots::RootedItems::new();
            pair.push(pyre_object::w_str_new_managed(&name));
            pair.push(w_field);
            let w_pair = pyre_object::w_tuple_new(pair.take());
            roots.set(base + i, w_pair);
        }
    }
    let items = (0..fields.len()).map(|i| roots.get(base + i)).collect();
    Ok(pyre_object::w_list_new(items))
}

/// The `(name, field)` pairs of a completed struct's field dict.
fn fields_dict_items(ct: &W_CType) -> Result<Vec<(String, PyObjectRef)>, PyError> {
    if ct.fields_dict.is_null() {
        return Ok(Vec::new());
    }
    let keys = pyre_interpreter::baseobjspace::fixedview(ct.fields_dict, -1)?;
    let mut out = Vec::with_capacity(keys.len());
    for mut w_key in keys {
        let name =
            pyre_object::with_roots!(w_key => pyre_interpreter::baseobjspace::text_w(w_key))?
                .to_string();
        let w_value = pyre_interpreter::baseobjspace::getitem(ct.fields_dict, w_key)?;
        out.push((name, w_value));
    }
    Ok(out)
}

/// The name a completed struct's field dict knows this field by.  A nested
/// anonymous struct splices its fields in under those names, and a field the
/// dict never recorded — an anonymous bitfield — keeps none.
pub fn name_of_field(ct: &W_CType, mut w_field: PyObjectRef) -> Result<Option<String>, PyError> {
    for (name, w_value) in pyre_object::with_roots!(w_field => fields_dict_items(ct))? {
        if w_value == w_field {
            return Ok(Some(name));
        }
    }
    Ok(None)
}

/// `W_CTypeStructOrUnion.cdata_dir`.
pub fn cdata_dir(ct: &W_CType) -> Result<Vec<String>, PyError> {
    if ct.size < 0 {
        return Ok(Vec::new());
    }
    ct.force_lazy_struct()?;
    Ok(fields_dict_items(ct)?
        .into_iter()
        .map(|(name, _)| name)
        .collect())
}

/// `W_CTypeStructOrUnion.typeoffsetof_field`.
pub fn typeoffsetof_field(ct: &W_CType, fieldname: &str) -> Result<(PyObjectRef, i64), PyError> {
    ct.force_lazy_struct()?;
    let field = ctypeobj::getcfield(ct, fieldname, "read")
        .map_err(|_| PyError::key_error(fieldname.to_string()))?;
    if field.bitshift >= 0 {
        return Err(PyError::type_error("not supported for bitfields"));
    }
    Ok((field.ctype, field.offset))
}

// ── the Python type ─────────────────────────────────────────────────────

static CFIELD_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

/// `_cffi_backend.CField`.
pub fn cfield_type() -> PyObjectRef {
    CFIELD_TYPE_OBJ.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type_with_layout(
            "_cffi_backend.CField",
            init_cfield_type,
            pyre_interpreter::typedef::w_object(),
            <W_CField as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_CField as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        tp
    })
}

fn init_cfield_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    let store =
        |name: &str, value: PyObjectRef| pyre_interpreter::__pyre_put_new!(ns_slot, name, value);
    for (name, doc) in [
        ("type", "type"),
        ("offset", "offset"),
        ("bitshift", "bitshift"),
        ("bitsize", "bitsize"),
        ("flags", "flags"),
    ] {
        store(
            name,
            pyre_interpreter::typedef::make_getset_property_named_doc(
                make_fget(name),
                pyre_object::PY_NULL,
                pyre_object::PY_NULL,
                doc,
                name,
            ),
        );
    }
}

/// `interp_attrproperty` / `interp_attrproperty_w` for the five attributes.
/// A builtin function carries no closure, so each name needs its own reader.
fn make_fget(name: &'static str) -> PyObjectRef {
    macro_rules! fget {
        ($read:expr) => {
            pyre_interpreter::make_builtin_function_with_arity(
                name,
                |args| {
                    // `typedef.py self.fget(self, space, w_obj)`.
                    let w_self = args
                        .get(1)
                        .copied()
                        .ok_or_else(|| PyError::type_error("descriptor requires an instance"))?;
                    let field = W_CField::from_obj(w_self)
                        .ok_or_else(|| PyError::type_error("expected a CField object"))?;
                    Ok($read(field))
                },
                2,
            )
        };
    }
    match name {
        "type" => fget!(|f: &W_CField| f.ctype),
        "offset" => fget!(|f: &W_CField| pyre_object::w_int_new(f.offset)),
        "bitshift" => fget!(|f: &W_CField| pyre_object::w_int_new(f.bitshift)),
        "bitsize" => fget!(|f: &W_CField| pyre_object::w_int_new(f.bitsize)),
        _ => fget!(|f: &W_CField| pyre_object::w_int_new(f.flags)),
    }
}
