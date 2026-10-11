//! `array` module — the `array.array` type.
//!
//! PyPy: pypy/module/array/interp_array.py + reconstructor.py
//!
//! The element storage lives in `pyre_object::interp_array` (an off-GC
//! `*mut Vec<u8>` of native-order bytes).  This module supplies the
//! interpreter-level behaviour: range-checked packing of a Python object
//! into element bytes, the bound methods, the `__new__` constructor, and
//! the module-level `_array_reconstructor` unpickler.

use majit_rlib::rbigint::RBigInt as BigInt;
use pyre_interpreter::baseobjspace::{float_w, int_w, uint_w};
use pyre_interpreter::objspace::descroperation::{CompareOp, compare};
use pyre_interpreter::{PyError, PyErrorKind, PyResult, make_builtin_function_with_arity};
use pyre_object::interp_array as arr;
use pyre_object::{PY_NULL, PyObjectRef};
use rustpython_wtf8::{CodePoint, Wtf8Buf};

/// A fixed inline byte buffer for one packed element (≤ 8 bytes).
type Bytes = [u8; 8];

// ──────────────────────────────────────────────────────────────────────
// Packing: Python object → native-order element bytes (range-checked).
// ──────────────────────────────────────────────────────────────────────

/// Pack `w` into `out[..itemsize]` per `typecode`, returning the item size.
/// `interp_array.py W_Array.item_w` + the per-`TypeCode` overflow checks.
fn pack_into(typecode: u8, w: PyObjectRef, out: &mut Bytes) -> Result<usize, PyError> {
    fn signed_range(v: i64, lo: i64, hi: i64, name: &str) -> Result<(), PyError> {
        if v < lo {
            Err(PyError::overflow_error(format!(
                "{name} is less than minimum"
            )))
        } else if v > hi {
            Err(PyError::overflow_error(format!(
                "{name} is greater than maximum"
            )))
        } else {
            Ok(())
        }
    }
    /// The item converters take `PyNumber_Index` before the width check, so a
    /// value that is not an integer at all is refused by the protocol, which
    /// names the type it could not convert, rather than by the unwrap behind
    /// it, which states PyPy's own sentence.
    fn item_int_w(w: PyObjectRef) -> Result<i64, PyError> {
        if unsafe { pyre_object::is_int(w) || pyre_object::pyobject::is_long(w) } {
            return int_w(w);
        }
        int_w(pyre_interpreter::baseobjspace::space_index(w)?)
    }
    let n = match typecode {
        b'b' => {
            let v = item_int_w(w)?;
            signed_range(v, i8::MIN as i64, i8::MAX as i64, "signed char")?;
            out[..1].copy_from_slice(&(v as i8).to_ne_bytes());
            1
        }
        b'B' => {
            let v = item_int_w(w)?;
            signed_range(v, 0, u8::MAX as i64, "unsigned byte integer")?;
            out[..1].copy_from_slice(&(v as u8).to_ne_bytes());
            1
        }
        b'h' => {
            let v = item_int_w(w)?;
            signed_range(v, i16::MIN as i64, i16::MAX as i64, "signed short")?;
            out[..2].copy_from_slice(&(v as i16).to_ne_bytes());
            2
        }
        b'H' => {
            let v = item_int_w(w)?;
            signed_range(v, 0, u16::MAX as i64, "unsigned short")?;
            out[..2].copy_from_slice(&(v as u16).to_ne_bytes());
            2
        }
        b'i' => {
            let v = item_int_w(w)?;
            signed_range(v, i32::MIN as i64, i32::MAX as i64, "signed int")?;
            out[..4].copy_from_slice(&(v as i32).to_ne_bytes());
            4
        }
        b'I' => {
            let v = item_int_w(w)?;
            signed_range(v, 0, u32::MAX as i64, "unsigned int")?;
            out[..4].copy_from_slice(&(v as u32).to_ne_bytes());
            4
        }
        b'l' | b'q' => {
            // C long / long long on 64-bit — full i64 range; `int_w` itself
            // raises OverflowError outside it.
            let v = item_int_w(w)?;
            out[..8].copy_from_slice(&v.to_ne_bytes());
            8
        }
        b'L' | b'Q' => {
            // Unsigned 64-bit — `uint_w` handles bignums and rejects negatives.
            let v = if unsafe { pyre_object::is_int(w) || pyre_object::pyobject::is_long(w) } {
                uint_w(w)
            } else {
                uint_w(pyre_interpreter::baseobjspace::space_index(w)?)
            }
            .map_err(|error| {
                if error.kind == PyErrorKind::ValueError {
                    PyError::overflow_error("unsigned 8-byte integer is less than minimum")
                } else {
                    error
                }
            })?;
            out[..8].copy_from_slice(&v.to_ne_bytes());
            8
        }
        b'f' => {
            let v = float_w(w)? as f32;
            out[..4].copy_from_slice(&v.to_ne_bytes());
            4
        }
        b'd' => {
            let v = float_w(w)?;
            out[..8].copy_from_slice(&v.to_ne_bytes());
            8
        }
        b'u' | b'w' => {
            let cp = unicode_char_w(w)?;
            out[..4].copy_from_slice(&cp.to_ne_bytes());
            4
        }
        _ => return Err(PyError::value_error("bad typecode")),
    };
    Ok(n)
}

/// Extract a single Unicode code point from a length-1 str (`'u'` items).
fn unicode_char_w(w: PyObjectRef) -> Result<u32, PyError> {
    if !unsafe { pyre_object::is_str(w) } {
        return Err(PyError::type_error(
            "array item must be a unicode character, not a different type",
        ));
    }
    let s = unsafe { pyre_object::unicodeobject::w_str_get_wtf8(w) };
    let mut points = s.code_points();
    match (points.next(), points.next()) {
        (Some(c), None) => Ok(c.to_u32()),
        _ => Err(PyError::type_error(
            "array item must be a unicode character, not str",
        )),
    }
}

// ──────────────────────────────────────────────────────────────────────
// Element unpacking.
// ──────────────────────────────────────────────────────────────────────

/// PyPy `W_Array.w_getitem`: box one live array element, with the raw-integer
/// mode `compare_arrays` uses for unicode items.
///
/// [3.14-spec] PyPy reports its array-specific out-of-range sentence here.
/// `arraymodule.c` `u_getitem` / `w_getitem` at v3.14.6 instead call
/// `PyUnicode_FromOrdinal`, whose observable error is the `chr()` sentence
/// below. Keep PyPy's `integer_instead_of_char` shape: same-descriptor array
/// comparisons use it to compare even malformed raw code points without
/// trying to construct a Python string.
pub(crate) fn array_w_getitem(
    obj: PyObjectRef,
    index: usize,
    integer_instead_of_char: bool,
) -> PyResult {
    let typecode = unsafe { arr::w_array_typecode(obj) };
    if matches!(typecode, b'u' | b'w') {
        let itemsize = unsafe { arr::w_array_itemsize(obj) };
        let offset = index * itemsize;
        let bytes = unsafe { arr::w_array_bytes(obj) };
        let code = u32::from_ne_bytes(bytes[offset..offset + 4].try_into().unwrap());
        if integer_instead_of_char {
            // PyPy casts the `u` wchar_t through `lltype.Signed`, matching
            // v3.14.6's `u_compareitems(wchar_t)`. The 3.14 `w` descriptor is
            // Py_UCS4 and therefore keeps the full unsigned value.
            let value = if typecode == b'u' {
                i32::from_ne_bytes(code.to_ne_bytes()) as i64
            } else {
                code as i64
            };
            return Ok(pyre_object::w_int_new(value));
        }
        let point = CodePoint::from_u32(code)
            .ok_or_else(|| PyError::value_error("chr() arg not in range(0x110000)"))?;
        let mut one = Wtf8Buf::new();
        one.push(point);
        return Ok(pyre_object::unicodeobject::w_str_from_wtf8_managed(one));
    }
    Ok(unsafe { arr::w_array_unpack_item(obj, index) })
}

// Core mutation helpers.
fn array_check_resize(obj: PyObjectRef) -> Result<(), PyError> {
    if unsafe { arr::w_array_exports(obj) } != 0 {
        Err(PyError::new(
            PyErrorKind::BufferError,
            "cannot resize an array that is exporting buffers",
        ))
    } else {
        Ok(())
    }
}

/// Append one item through the public 3.14 `ins1` conversion sequence.
///
/// PyPy `W_Array.descr_append` converts `w_x` before `setlen`.
/// [3.14-spec] CPython v3.14.6 `Modules/arraymodule.c` `ins1` preserves that
/// ordering, but validates once at index -1, resizes from the length captured
/// before validation, then converts once more for the actual slot.  `append`
/// and `array_iter_extend` share this helper so callbacks see the same receiver
/// state on both paths.
fn array_append_value(obj: PyObjectRef, w_value: PyObjectRef) -> Result<(), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[obj, w_value]);
    let obj = pyre_object::gc_roots::shadow_stack_get(base);
    let old_len = unsafe { arr::w_array_len(obj) };
    let itemsize = unsafe { arr::w_array_itemsize(obj) };
    let typecode = unsafe { arr::w_array_typecode(obj) };

    let mut validated: Bytes = [0; 8];
    pack_into(
        typecode,
        pyre_object::gc_roots::shadow_stack_get(base + 1),
        &mut validated,
    )?;

    let obj = pyre_object::gc_roots::shadow_stack_get(base);
    array_check_resize(obj)?;
    let new_len = old_len
        .checked_add(1)
        .and_then(|len| len.checked_mul(itemsize))
        .ok_or_else(|| PyError::memory_error(""))?;
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    if new_len > vec.len() {
        vec.try_reserve(new_len - vec.len())
            .map_err(|_| PyError::memory_error(""))?;
    }
    vec.resize(new_len, 0);

    let mut packed: Bytes = [0; 8];
    let packed_len = pack_into(
        typecode,
        pyre_object::gc_roots::shadow_stack_get(base + 1),
        &mut packed,
    )?;
    let obj = pyre_object::gc_roots::shadow_stack_get(base);
    if old_len < unsafe { arr::w_array_len(obj) } {
        // CPython writes through the pointer captured after resize even when
        // the second conversion resized the receiver.  Preserve the visible
        // slot write while it remains live, without reproducing a stale raw
        // write when the callback cleared the receiver.
        let start = old_len * itemsize;
        let vec = unsafe { arr::w_array_vec_mut(obj) };
        vec[start..start + itemsize].copy_from_slice(&packed[..packed_len]);
    }
    Ok(())
}

/// Extend from any iterable (`W_Array.extend` / `_fromiterable`).
fn array_extend_iterable(
    obj: PyObjectRef,
    w_iterable: PyObjectRef,
    reject_different_array: bool,
) -> Result<(), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[obj, w_iterable]);

    // PyPy `W_Array.extend` handles an array source before `_fromiterable`:
    // compare the descriptors, size once, then raw-copy the source.  Like its
    // `oldlen` / `new` loop, keep the original source length when self is also
    // the iterable and copy from the resized receiver's leading range.
    let w_iterable = pyre_object::gc_roots::shadow_stack_get(base + 1);
    if unsafe { arr::is_array(w_iterable) } {
        let obj = pyre_object::gc_roots::shadow_stack_get(base);
        let dst_tc = unsafe { arr::w_array_typecode(obj) };
        let src_tc = unsafe { arr::w_array_typecode(w_iterable) };
        if dst_tc != src_tc {
            if reject_different_array {
                return Err(PyError::type_error(
                    "can only extend with array of same kind",
                ));
            }
        } else {
            let src_len = unsafe { arr::w_array_bytes(w_iterable) }.len();
            // CPython `array_do_extend` treats an empty source as a true
            // no-op, so it succeeds even while the receiver exports a buffer.
            if src_len == 0 {
                return Ok(());
            }
            let obj = pyre_object::gc_roots::shadow_stack_get(base);
            array_check_resize(obj)?;
            let dst_len = unsafe { arr::w_array_bytes(obj) }.len();
            let new_len = dst_len
                .checked_add(src_len)
                .ok_or_else(|| PyError::memory_error(""))?;
            if std::ptr::eq(obj, w_iterable) {
                let vec = unsafe { arr::w_array_vec_mut(obj) };
                vec.try_reserve(src_len)
                    .map_err(|_| PyError::memory_error(""))?;
                vec.resize(new_len, 0);
                // PyPy copies `srcbuf[0:new]` into the newly sized receiver;
                // CPython `array_do_extend` does the same with `memcpy`.
                // The source and destination byte ranges are adjacent.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        vec.as_ptr(),
                        vec.as_mut_ptr().add(dst_len),
                        src_len,
                    )
                };
                return Ok(());
            }

            // The objects are distinct and no Python runs between these two
            // borrows, so resizing the destination cannot move the source.
            let src_bytes = unsafe { arr::w_array_bytes(w_iterable) };
            let vec = unsafe { arr::w_array_vec_mut(obj) };
            vec.try_reserve(src_len)
                .map_err(|_| PyError::memory_error(""))?;
            vec.extend_from_slice(src_bytes);
            return Ok(());
        }
    }

    // PyPy `_fromiterable` and CPython `array_iter_extend` mint the iterator
    // before attempting any resize.  Consequently an empty iterable remains
    // a no-op under a live export, and iterator side effects precede the first
    // per-item resize check.
    let w_iterable = pyre_object::gc_roots::shadow_stack_get(base + 1);
    let w_iter = pyre_interpreter::baseobjspace::iter(w_iterable)?;
    let iter_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_iter);
    loop {
        match pyre_interpreter::baseobjspace::next(pyre_object::gc_roots::shadow_stack_get(
            iter_slot,
        )) {
            Ok(w_item) => {
                let obj = pyre_object::gc_roots::shadow_stack_get(base);
                array_append_value(obj, w_item)?;
            }
            Err(e) => {
                let (stop, e) = e.matches_stop_iteration_keep();
                if stop {
                    break;
                }
                return Err(e);
            }
        }
    }
    Ok(())
}

/// Append raw bytes (`frombytes`); length must be a multiple of itemsize.
fn array_frombytes(obj: PyObjectRef, bytes: &[u8]) -> Result<(), PyError> {
    let isz = unsafe { arr::w_array_itemsize(obj) };
    if !bytes.len().is_multiple_of(isz) {
        return Err(PyError::value_error(
            "bytes length not a multiple of item size",
        ));
    }
    // PyPy `_frombytes` returns before `setlen` when the copied buffer is
    // empty; CPython's `frombytes` likewise skips `array_resize` for n == 0.
    // An empty append therefore succeeds under an outstanding export, and a
    // malformed length above wins over that export's resize error.
    if bytes.is_empty() {
        return Ok(());
    }
    array_check_resize(obj)?;
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    vec.try_reserve(bytes.len())
        .map_err(|_| PyError::memory_error(""))?;
    vec.extend_from_slice(bytes);
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// __new__
// ──────────────────────────────────────────────────────────────────────

/// `array.__new__(cls, typecode, [initializer])` — `interp_array.py w_array`.
fn array_descr_new(args: &[PyObjectRef]) -> PyResult {
    let (pos, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
    if pos.len() < 2 {
        return Err(PyError::type_error(
            "array() takes at least 1 argument (0 given)",
        ));
    }
    if pos.len() > 3 {
        return Err(PyError::type_error(format!(
            "array() takes at most 2 arguments ({} given)",
            pos.len() - 1
        )));
    }
    // `pos` is a native slice into the flat ABI's argument buffer.  The gateway
    // keeps the arguments alive, but it does not rewrite this copy, and the 'u'
    // deprecation warning below runs Python: a `list` or `dict` initializer is
    // movable and leaves its pre-move address behind in the slice.  Pin the
    // arguments here, before that warning, and read the initializer back out of
    // its slot where it is used.
    let _pos_roots = pyre_object::gc_roots::push_roots();
    let pos_base = pyre_object::gc_roots::pin_roots(pos);
    let canonical = pyre_interpreter::typedef::gettypefor(&pyre_object::interp_array::ARRAY_TYPE)
        .map_or(PY_NULL, |ty| ty.as_ptr());
    let canonical_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(canonical);
    let cls = pyre_object::gc_roots::shadow_stack_get(pos_base);
    // PyPy's interp2app gateway leaves keywords available to a subtype's
    // overridden __init__.  The exact array type (and a subtype inheriting
    // array.__init__) rejects them later; __new__ itself must not mistake the
    // flat ABI's kwargs marker for the optional initializer.
    let init_matches = std::ptr::eq(cls, pyre_object::gc_roots::shadow_stack_get(canonical_slot))
        || unsafe {
            match (
                pyre_interpreter::baseobjspace::lookup_in_type(cls, "__init__"),
                pyre_interpreter::baseobjspace::lookup_in_type(
                    pyre_object::gc_roots::shadow_stack_get(canonical_slot),
                    "__init__",
                ),
            ) {
                (Some(sub), Some(base)) => std::ptr::eq(sub, base),
                (None, None) => true,
                _ => false,
            }
        };
    if init_matches && pyre_interpreter::builtins::has_real_kwargs(kwargs) {
        return Err(PyError::type_error(
            "array.array() takes no keyword arguments",
        ));
    }
    let w_typecode = pyre_object::gc_roots::shadow_stack_get(pos_base + 1);
    // typecode must be a 1-character str.
    if !unsafe { pyre_object::is_str(w_typecode) } {
        return Err(PyError::type_error(format!(
            "array() argument 1 must be a unicode character, not {}",
            pyre_interpreter::baseobjspace::object_functionstr_type_name(w_typecode)
        )));
    }
    // Measured in code points on the raw buffer: a lone surrogate is one
    // character with no `&str` spelling, so it reaches the typecode check
    // below rather than counting as three.
    let tc = unsafe { pyre_object::unicodeobject::w_str_get_wtf8(w_typecode) };
    let mut points = tc.code_points();
    let Some(first) = points.next().filter(|_| points.next().is_none()) else {
        return Err(PyError::type_error(format!(
            "array() argument 1 must be a unicode character, not a string of length {}",
            tc.code_points().count()
        )));
    };
    // Every typecode is ASCII, so a wider code point takes the same
    // "bad typecode" rejection as an unrecognised ASCII one.
    let typecode = u8::try_from(first.to_u32()).unwrap_or(0xff);
    let itemsize = arr::typecode_itemsize(typecode).ok_or_else(|| {
        PyError::value_error("bad typecode (must be b, B, u, w, h, H, i, I, l, L, q, Q, f or d)")
    })?;
    if typecode == b'u' {
        pyre_interpreter::warn::warn_deprecation(
            "The 'u' type code is deprecated and will be removed in Python 3.16",
        )?;
    }
    // The 'u' deprecation warning above can collect, so the subclass word is
    // re-read from the shadow stack rather than the pre-warning local.
    let cls = pyre_object::gc_roots::shadow_stack_get(pos_base);
    let obj = if !cls.is_null()
        && unsafe { pyre_object::is_type(cls) }
        && !std::ptr::eq(cls, pyre_object::gc_roots::shadow_stack_get(canonical_slot))
    {
        // `allocate_instance(W_ArrayUser, w_cls)` (`typedef.py` `_getusercls`),
        // then the user-finalizer enqueue `tag_subclass_instance` performs
        // after `user_setup`.
        let obj = arr::w_array_user_new(typecode, itemsize, cls);
        pyre_object::gc_hook::maybe_register_finalizer(obj);
        obj
    } else {
        arr::w_array_new(typecode, itemsize)
    };
    // Nothing refers to the fresh array yet and every initializer below runs
    // Python.  An array is stable, so the local stays a valid address; the pin
    // is what stops a major cycle sweeping it as unreachable.
    let obj = pyre_object::gc_roots::pin_root(obj);
    // Optional initializer.
    if pos.len() >= 3 {
        let w_init = pyre_object::gc_roots::shadow_stack_get(pos_base + 2);
        if unsafe { pyre_object::is_str(w_init) } {
            if matches!(typecode, b'u' | b'w') {
                array_fromunicode(obj, w_init)?;
            } else {
                return Err(PyError::type_error(format!(
                    "cannot use a str to initialize an array with typecode '{}'",
                    typecode as char
                )));
            }
        } else if unsafe { pyre_object::bytesobject::is_bytes_like(w_init) } {
            let bytes = unsafe { pyre_object::bytesobject::bytes_like_data(w_init) }.to_vec();
            array_frombytes(obj, &bytes)?;
        } else {
            if unsafe { arr::is_array(w_init) }
                && matches!(unsafe { arr::w_array_typecode(w_init) }, b'u' | b'w')
                && !matches!(typecode, b'u' | b'w')
            {
                return Err(PyError::type_error(format!(
                    "cannot use a unicode array to initialize an array with typecode '{}'",
                    typecode as char
                )));
            }
            array_extend_iterable(obj, w_init, false)?;
        }
    }
    Ok(obj)
}

/// `fromunicode` — append code points of a str to a `'u'` array.
fn array_fromunicode(obj: PyObjectRef, w_str: PyObjectRef) -> Result<(), PyError> {
    // [3.14-spec] Argument Clinic's `unicode` converter runs before
    // `array_array_fromunicode_impl`, whereas PyPy's `descr_fromunicode`
    // checks the receiver typecode before entering `fromsequence`.
    if !unsafe { pyre_object::is_str(w_str) } {
        return Err(PyError::type_error(format!(
            "fromunicode() argument must be str, not {}",
            pyre_interpreter::type_methods::clinic_arg_type_name(w_str)
        )));
    }
    if !matches!(unsafe { arr::w_array_typecode(obj) }, b'u' | b'w') {
        return Err(PyError::value_error(
            "fromunicode() may only be called on unicode type arrays ('u' or 'w')",
        ));
    }
    let s = unsafe { pyre_object::unicodeobject::w_str_get_wtf8(w_str) };
    let count = s.code_points().count();
    // PyPy's `fromsequence` performs no resize for an empty unicode string;
    // CPython's same-size `array_resize` also accepts it under a live export.
    if count == 0 {
        return Ok(());
    }
    array_check_resize(obj)?;
    let byte_count = count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| PyError::memory_error(""))?;
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    vec.try_reserve(byte_count)
        .map_err(|_| PyError::memory_error(""))?;
    for cp in s.code_points() {
        vec.extend_from_slice(&cp.to_u32().to_ne_bytes());
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// Indexing.
// ──────────────────────────────────────────────────────────────────────

/// Normalize an integer index against the receiver's live length.
///
/// PyPy `decode_index4(w_idx, self)` and CPython 3.14
/// `array_subscr` / `array_ass_subscr` both run `__index__` before reading the
/// length used for negative-index adjustment and bounds.  The conversion can
/// resize and collect either input, so own both through that boundary and read
/// the receiver back before consulting its length.
fn index_in_range(obj: PyObjectRef, w_index: PyObjectRef, what: &str) -> Result<usize, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[obj, w_index]);
    let w_index = pyre_object::gc_roots::shadow_stack_get(base + 1);
    if unsafe {
        !pyre_object::pyobject::is_int_or_long(w_index)
            && pyre_interpreter::baseobjspace::lookup(w_index, "__index__").is_none()
    } {
        // PyPy `ObjSpace.decode_index4` enters `getindex_w` only after the
        // slice branch.  [3.14-spec] `arraymodule.c::array_subscr` and
        // `array_ass_subscr` give the array-specific sentence to an operand
        // with no index slot; an existing slot's own failure still escapes.
        return Err(PyError::type_error("array indices must be integers"));
    }
    let w_indexed = pyre_interpreter::baseobjspace::space_index(w_index)?;
    let mut i = match int_w(w_indexed) {
        Ok(i) => i,
        Err(error) if error.kind == PyErrorKind::OverflowError => {
            // PyPy `ObjSpace.getindex_w` names the original operand when the
            // converted integer does not fit.  [3.14-spec]
            // `PyNumber_AsSsize_t(item, PyExc_IndexError)` changes only the
            // exception class selected by the array subscription boundary.
            return Err(PyError::new(
                PyErrorKind::IndexError,
                format!(
                    "cannot fit '{}' into an index-sized integer",
                    pyre_interpreter::baseobjspace::object_functionstr_type_name(w_index)
                ),
            ));
        }
        Err(error) => return Err(error),
    };
    let len = unsafe { arr::w_array_len(pyre_object::gc_roots::shadow_stack_get(base)) };
    if i < 0 {
        i += len as i64;
    }
    if i < 0 || i >= len as i64 {
        return Err(PyError::new(
            PyErrorKind::IndexError,
            format!("{what} index out of range"),
        ));
    }
    Ok(i as usize)
}

/// PyPy `W_SliceObject.unpack` followed by `adjust_indices(..., self.len)`.
/// Bound conversions may resize the array, so the live length is deliberately
/// read only after all three have completed.
fn array_slice_indices(
    obj: PyObjectRef,
    key: PyObjectRef,
) -> Result<(i64, i64, i64, i64), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[obj, key]);
    let key = pyre_object::gc_roots::shadow_stack_get(base + 1);
    let (start, stop, step) = pyre_interpreter::sliceobject::slice_unpack(
        unsafe { pyre_object::sliceobject::w_slice_get_start(key) },
        unsafe { pyre_object::sliceobject::w_slice_get_stop(key) },
        unsafe { pyre_object::sliceobject::w_slice_get_step(key) },
    )?;
    let len = unsafe { arr::w_array_len(pyre_object::gc_roots::shadow_stack_get(base)) } as i64;
    Ok(pyre_interpreter::sliceobject::slice_adjust_indices(
        start, stop, step, len,
    ))
}

fn array_getitem(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__getitem__", false)?;
    check_arity(args, 2, "array.__getitem__")?;
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(args);
    let obj = pyre_object::gc_roots::shadow_stack_get(base);
    let key = pyre_object::gc_roots::shadow_stack_get(base + 1);
    if unsafe { pyre_object::sliceobject::is_slice(key) } {
        let (start, _, step, n) = array_slice_indices(obj, key)?;
        let obj = pyre_object::gc_roots::shadow_stack_get(base);
        let isz = unsafe { arr::w_array_itemsize(obj) };
        let tc = unsafe { arr::w_array_typecode(obj) };
        let src = unsafe { arr::w_array_bytes(obj) }.to_vec();
        let mut out: Vec<u8> = Vec::with_capacity(n as usize * isz);
        let mut i = start;
        for k in 0..n {
            let off = i as usize * isz;
            out.extend_from_slice(&src[off..off + isz]);
            if k + 1 < n {
                i += step;
            }
        }
        return Ok(arr::w_array_from_bytes(tc, isz as u8, out));
    }
    let i = index_in_range(obj, key, "array")?;
    let obj = pyre_object::gc_roots::shadow_stack_get(base);
    array_w_getitem(obj, i, false)
}

fn array_setitem(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__setitem__", false)?;
    check_arity(args, 3, "array.__setitem__")?;
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(args);
    let obj = pyre_object::gc_roots::shadow_stack_get(base);
    let key = pyre_object::gc_roots::shadow_stack_get(base + 1);
    let w_value = pyre_object::gc_roots::shadow_stack_get(base + 2);
    if unsafe { pyre_object::sliceobject::is_slice(key) } {
        // `array_ass_subscr` names the operand it was handed; an array of
        // another kind passed that test and is refused by `PyErr_BadArgument`
        // behind it.
        if !unsafe { arr::is_array(w_value) } {
            return Err(PyError::type_error(format!(
                "can only assign array (not \"{}\") to array slice",
                pyre_interpreter::error::type_name_of(w_value)
            )));
        }
        let tc = unsafe { arr::w_array_typecode(obj) };
        if unsafe { arr::w_array_typecode(w_value) } != tc {
            return Err(PyError::type_error(
                "bad argument type for built-in operation",
            ));
        }
        let (start, stop, step, n) = array_slice_indices(obj, key)?;
        let obj = pyre_object::gc_roots::shadow_stack_get(base);
        let w_value = pyre_object::gc_roots::shadow_stack_get(base + 2);
        let isz = unsafe { arr::w_array_itemsize(obj) };
        let src = unsafe { arr::w_array_bytes(w_value) }.to_vec();
        let src_len = src.len() / isz;
        if step == 1 {
            // Contiguous: may resize.
            if src_len as i64 != n {
                array_check_resize(obj)?;
            }
            let vec = unsafe { arr::w_array_vec_mut(obj) };
            let lo = start as usize * isz;
            let hi = stop.max(start) as usize * isz;
            vec.splice(lo..hi, src.iter().copied());
        } else {
            if src_len as i64 != n {
                return Err(PyError::value_error(format!(
                    "attempt to assign array of size {src_len} to extended slice of size {n}"
                )));
            }
            let vec = unsafe { arr::w_array_vec_mut(obj) };
            let mut i = start;
            for k in 0..n {
                let dst = i as usize * isz;
                let s = k as usize * isz;
                vec[dst..dst + isz].copy_from_slice(&src[s..s + isz]);
                if k + 1 < n {
                    i += step;
                }
            }
        }
        return Ok(pyre_object::w_none());
    }
    let i = index_in_range(obj, key, "array assignment")?;
    let obj = pyre_object::gc_roots::shadow_stack_get(base);
    let tc = unsafe { arr::w_array_typecode(obj) };
    let isz = unsafe { arr::w_array_itemsize(obj) };
    let mut buf: Bytes = [0u8; 8];
    // pack_into may run user code (`__index__`/`__int__`/`__float__`) that
    // resizes the array mid-assignment (gh-142555); re-validate the slot
    // against the current length before writing.
    let n = pack_into(
        tc,
        pyre_object::gc_roots::shadow_stack_get(base + 2),
        &mut buf,
    )?;
    let obj = pyre_object::gc_roots::shadow_stack_get(base);
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    let end = i * isz + n;
    if end > vec.len() {
        return Err(PyError::new(
            PyErrorKind::IndexError,
            "array assignment index out of range".to_string(),
        ));
    }
    vec[i * isz..end].copy_from_slice(&buf[..n]);
    Ok(pyre_object::w_none())
}

fn array_delitem(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__delitem__", false)?;
    check_arity(args, 2, "array.__delitem__")?;
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(args);
    let obj = pyre_object::gc_roots::shadow_stack_get(base);
    let key = pyre_object::gc_roots::shadow_stack_get(base + 1);
    if unsafe { pyre_object::sliceobject::is_slice(key) } {
        let (start, _, step, n) = array_slice_indices(obj, key)?;
        if n == 0 {
            // No elements removed: leave the backing storage (and any exported
            // views over it) untouched rather than rebuilding the buffer.
            return Ok(pyre_object::w_none());
        }
        let obj = pyre_object::gc_roots::shadow_stack_get(base);
        array_check_resize(obj)?;
        let len = unsafe { arr::w_array_len(obj) };
        let isz = unsafe { arr::w_array_itemsize(obj) };
        // Collect element indices to drop, then rebuild the buffer.
        let mut drop_set: Vec<usize> = Vec::with_capacity(n as usize);
        let mut i = start;
        for k in 0..n {
            drop_set.push(i as usize);
            if k + 1 < n {
                i += step;
            }
        }
        drop_set.sort_unstable();
        let src = unsafe { arr::w_array_bytes(obj) }.to_vec();
        let mut out: Vec<u8> = Vec::with_capacity(src.len());
        let mut di = 0usize;
        for e in 0..len {
            if di < drop_set.len() && drop_set[di] == e {
                di += 1;
                continue;
            }
            out.extend_from_slice(&src[e * isz..e * isz + isz]);
        }
        let vec = unsafe { arr::w_array_vec_mut(obj) };
        *vec = out;
        return Ok(pyre_object::w_none());
    }
    let i = index_in_range(obj, key, "array assignment")?;
    let obj = pyre_object::gc_roots::shadow_stack_get(base);
    array_check_resize(obj)?;
    let isz = unsafe { arr::w_array_itemsize(obj) };
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    vec.drain(i * isz..i * isz + isz);
    Ok(pyre_object::w_none())
}

// ──────────────────────────────────────────────────────────────────────
// Sequence / list methods.
// ──────────────────────────────────────────────────────────────────────

fn array_len(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__len__", false)?;
    check_arity(args, 1, "array.__len__")?;
    Ok(pyre_object::w_int_new(
        unsafe { arr::w_array_len(args[0]) } as i64
    ))
}

fn array_iter(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__iter__", false)?;
    check_arity(args, 1, "array.__iter__")?;
    let len = unsafe { arr::w_array_len(args[0]) };
    Ok(pyre_object::w_seq_iter_new(args[0], len))
}

/// Preserve the exact positional signature that PyPy's interp2app gateway
/// enforces for these methods (`self` included in `expected_total`).
fn check_arity(args: &[PyObjectRef], expected_total: usize, name: &str) -> Result<(), PyError> {
    if args.len() != expected_total {
        let want = expected_total - 1;
        let noun = if want == 1 { "argument" } else { "arguments" };
        return Err(PyError::type_error(format!(
            "{name}() takes exactly {want} {noun} ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    Ok(())
}

/// The receiver check `descr_check` performs before any `array.array`
/// descriptor runs.  Without it an unbound call such as
/// `array.array.tolist(1)` reads a foreign object through the `W_Array`
/// layout: the typecode and itemsize getters hand back whatever bytes sit at
/// those offsets, and `w_array_len` divides the buffer length by an itemsize
/// that came from the same foreign memory.
///
/// Slot wrappers and method descriptors word both the missing-receiver and
/// the wrong-receiver message differently, so callers say which surface they
/// are, as they do for the list and tuple twins in `type_methods`.
fn require_array_receiver(
    args: &[PyObjectRef],
    name: &str,
    method_descriptor: bool,
) -> Result<PyObjectRef, PyError> {
    let Some(&receiver) = args.first() else {
        let message = if method_descriptor {
            format!("unbound method array.{name}() needs an argument")
        } else {
            format!("descriptor '{name}' of 'array.array' object needs an argument")
        };
        return Err(PyError::type_error(message));
    };
    if !unsafe { arr::is_array(receiver) } {
        let received = pyre_interpreter::baseobjspace::object_functionstr_type_name(receiver);
        let message = if method_descriptor {
            format!(
                "descriptor '{name}' for 'array.array' objects doesn't apply to a '{received}' object"
            )
        } else {
            format!(
                "descriptor '{name}' requires a 'array.array' object but received a '{received}'"
            )
        };
        return Err(PyError::type_error(message));
    }
    Ok(receiver)
}

fn check_arity_range(
    args: &[PyObjectRef],
    min_total: usize,
    max_total: usize,
    name: &str,
) -> Result<(), PyError> {
    if args.len() < min_total || args.len() > max_total {
        return Err(PyError::type_error(format!(
            "{name}() takes from {} to {} arguments ({} given)",
            min_total - 1,
            max_total - 1,
            args.len().saturating_sub(1)
        )));
    }
    Ok(())
}

fn array_append_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "append", true)?;
    pyre_interpreter::type_methods::reject_kwargs_of(Some("array"), args, "append")?;
    if args.len() != 2 {
        return Err(PyError::type_error(format!(
            "array.append() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    array_append_value(args[0], args[1])?;
    Ok(pyre_object::w_none())
}

fn array_extend_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "extend", true)?;
    let (positional, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
    let positional_given = if positional.is_empty() {
        0
    } else {
        positional.len() - 1
    };
    let supplied = positional_given + pyre_interpreter::builtins::real_kwarg_count(kwargs);
    // [3.14-spec] CPython v3.14.6 `array_array_extend`'s positional-only
    // clinic gateway checks the required positional count before diagnosing
    // its blank keyword slot; PyPy's interp2app gateway reports its own owner.
    if positional_given == 0 {
        return Err(PyError::type_error(
            "extend() takes exactly 1 positional argument (0 given)",
        ));
    }
    if supplied > 1 {
        return Err(PyError::type_error(format!(
            "extend() takes at most 1 argument ({supplied} given)"
        )));
    }
    array_extend_iterable(positional[0], positional[1], true)?;
    Ok(pyre_object::w_none())
}

/// The ssize gateway for `array.insert` / `array.pop` / `array.fromfile`.
///
/// PyPy's `W_ArrayBase.descr_insert` and `descr_pop` use `@unwrap_spec(...=int)`
/// and `descr_fromfile` uses `@unwrap_spec(n=int)` before entering the method
/// body.  CPython 3.14 exposes the narrower `__index__` protocol and its
/// Py_ssize_t overflow wording, so keep PyPy's gateway-before-body ordering
/// while applying the 3.14 observable contract.
fn array_ssize_index_w(w_index: PyObjectRef) -> Result<i64, PyError> {
    let w_index = pyre_interpreter::baseobjspace::space_index(w_index)?;
    pyre_interpreter::baseobjspace::int_w(w_index).map_err(|error| {
        if error.kind == PyErrorKind::OverflowError
            && error.message_text() == "int too large to convert to int"
        {
            PyError::overflow_error("Python int too large to convert to C ssize_t")
        } else {
            error
        }
    })
}

fn array_insert_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "insert", true)?;
    pyre_interpreter::type_methods::reject_kwargs_of(Some("array"), args, "insert")?;
    pyre_interpreter::type_methods::arity_exact_unpack(args, "insert", 2)?;
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(args);
    // `@unwrap_spec(idx=int)` runs before PyPy's `descr_insert` reads len.
    let mut i = array_ssize_index_w(pyre_object::gc_roots::shadow_stack_get(base + 1))?;
    let obj = pyre_object::gc_roots::shadow_stack_get(base);
    let old_len = unsafe { arr::w_array_len(obj) };
    if i < 0 {
        i += old_len as i64;
        if i < 0 {
            i = 0;
        }
    }
    if i > old_len as i64 {
        i = old_len as i64;
    }
    let i = i as usize;

    // PyPy `W_Array.descr_insert` converts `w_val` before `setlen`, then moves
    // the tail and writes that converted value.  [3.14-spec] CPython v3.14.6
    // `arraymodule.c::ins1` exposes the same pre-resize validation as append,
    // but validates a second time after resizing and moving the tail.  Keep
    // PyPy's gateway/index structure and reproduce that observable two-call
    // sequence.  In particular, a first conversion may mutate the receiver;
    // the resize still targets the length captured before it ran.
    let tc = unsafe { arr::w_array_typecode(obj) };
    let itemsize = unsafe { arr::w_array_itemsize(obj) };
    let mut validated: Bytes = [0u8; 8];
    pack_into(
        tc,
        pyre_object::gc_roots::shadow_stack_get(base + 2),
        &mut validated,
    )?;

    let obj = pyre_object::gc_roots::shadow_stack_get(base);
    array_check_resize(obj)?;
    let new_bytes = old_len
        .checked_add(1)
        .and_then(|len| len.checked_mul(itemsize))
        .ok_or_else(|| PyError::memory_error(""))?;
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    if new_bytes > vec.len() {
        vec.try_reserve(new_bytes - vec.len())
            .map_err(|_| PyError::memory_error(""))?;
    }
    vec.resize(new_bytes, 0);
    let start = i * itemsize;
    let end = old_len * itemsize;
    if i != old_len {
        vec.copy_within(start..end, start + itemsize);
    }

    let mut packed: Bytes = [0u8; 8];
    let packed_len = pack_into(
        tc,
        pyre_object::gc_roots::shadow_stack_get(base + 2),
        &mut packed,
    )?;
    let obj = pyre_object::gc_roots::shadow_stack_get(base);
    if i >= unsafe { arr::w_array_len(obj) } {
        return Err(PyError::new(
            PyErrorKind::IndexError,
            "array assignment index out of range",
        ));
    }
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    vec[start..start + itemsize].copy_from_slice(&packed[..packed_len]);
    Ok(pyre_object::w_none())
}

fn array_pop_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "pop", true)?;
    pyre_interpreter::type_methods::reject_kwargs_of(Some("array"), args, "pop")?;
    pyre_interpreter::type_methods::arity_at_most(args, "pop", 1)?;
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(args);
    // `@unwrap_spec(i=int)` runs before PyPy's `descr_pop`, including for an
    // empty receiver.  Re-read the receiver after this user-code boundary.
    let mut i = if args.len() >= 2 {
        array_ssize_index_w(pyre_object::gc_roots::shadow_stack_get(base + 1))?
    } else {
        -1
    };
    let obj = pyre_object::gc_roots::shadow_stack_get(base);
    let len = unsafe { arr::w_array_len(obj) };
    if len == 0 {
        return Err(PyError::new(
            PyErrorKind::IndexError,
            "pop from empty array".to_string(),
        ));
    }
    if i < 0 {
        i += len as i64;
    }
    if i < 0 || i >= len as i64 {
        return Err(PyError::new(
            PyErrorKind::IndexError,
            "pop index out of range".to_string(),
        ));
    }
    let isz = unsafe { arr::w_array_itemsize(obj) };
    let w_val = array_w_getitem(obj, i as usize, false)?;
    // PyPy `W_Array.descr_pop` boxes the item before `setlen` checks exports;
    // v3.14.6 `array_array_pop_impl` likewise calls `getarrayitem` before
    // `array_del_slice`. An invalid unicode value therefore wins over a live
    // export, and either error leaves the array untouched.
    array_check_resize(obj)?;
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    vec.drain(i as usize * isz..i as usize * isz + isz);
    Ok(w_val)
}

fn array_remove_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "remove", true)?;
    pyre_interpreter::type_methods::reject_kwargs_of(Some("array"), args, "remove")?;
    if args.len() != 2 {
        return Err(PyError::type_error(format!(
            "array.remove() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(args);
    let idx = array_index_count(
        pyre_object::gc_roots::shadow_stack_get(base),
        pyre_object::gc_roots::shadow_stack_get(base + 1),
        false,
    )?;
    match usize::try_from(idx) {
        Ok(i) => {
            let obj = pyre_object::gc_roots::shadow_stack_get(base);
            // [3.14-spec] CPython `array_array_remove_impl` hands the matched
            // index to `array_del_slice`; if `__eq__` already shrank the array
            // past it, deletion is an empty successful slice.  PyPy's
            // `W_ArrayBase.descr_remove` instead calls `descr_pop`, whose
            // bounds check raises.  Preserve the 3.14 observable no-op.
            if i >= unsafe { arr::w_array_len(obj) } {
                return Ok(pyre_object::w_none());
            }
            // PyPy reaches its resize through `descr_pop` only after the
            // comparison search.  CPython likewise runs comparisons before
            // `array_del_slice` rejects an exported receiver.
            array_check_resize(obj)?;
            let isz = unsafe { arr::w_array_itemsize(obj) };
            let vec = unsafe { arr::w_array_vec_mut(obj) };
            vec.drain(i * isz..i * isz + isz);
            Ok(pyre_object::w_none())
        }
        Err(_) => Err(PyError::value_error("array.remove(x): x not in array")),
    }
}

/// First matching index, or the total number of matches, via `==`.
fn array_index_count(obj: PyObjectRef, w_value: PyObjectRef, count: bool) -> Result<i64, PyError> {
    // PyPy `index_count_array` shares this loop between index-like searches
    // and count.  The receiver and needle stay red/live because equality is
    // user code that may resize and collect either object.
    //
    // [3.14-spec] `index_count_array` takes `stop = min(stop, arr.len)` once
    // before the loop, so an `__eq__` that appends is never searched, while
    // 3.14 keeps searching the array as it grows.  Measured on 3.14.2, an
    // `array('i', [1, 2, 3])` whose needle extends it by two 7s answers
    // `count()` 2, `in` True and a successful `remove()`; pypy3 7.3.20
    // answers 0, False and ValueError.  Re-reading is also what bounds the
    // storage here: a shrink leaves PyPy's separately allocated raw buffer
    // addressable, whereas this Vec's length is the bound, so the length is
    // read back at every loop boundary rather than snapshotted.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[obj, w_value]);
    let mut cnt = 0i64;
    let mut i = 0;
    loop {
        let obj = pyre_object::gc_roots::shadow_stack_get(base);
        if i >= unsafe { arr::w_array_len(obj) } {
            break;
        }
        let w_item = array_w_getitem(obj, i, false)?;
        let _elem_roots = pyre_object::gc_roots::push_roots();
        let w_item = pyre_object::gc_roots::pin_root(w_item);
        let w_value = pyre_object::gc_roots::shadow_stack_get(base + 1);
        if pyre_interpreter::baseobjspace::eq_w(w_item, w_value)? {
            if count {
                cnt += 1;
            } else {
                return Ok(i as i64);
            }
        }
        i += 1;
    }
    Ok(if count { cnt } else { -1 })
}

fn array_index_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "index", true)?;
    // PyPy's `W_ArrayBase.descr_index` keeps the search itself in
    // `index_count_array`, with `start`/`stop` unwrapped as indexes.  Keep
    // that shape below.  [3.14-spec] CPython 3.14's `array.array.index`
    // exposes those arguments as positional-only, however, and its
    // PyArg_UnpackTuple gateway reports the bare method name at the arity
    // boundary.  Reject the trailing kwargs marker before it can be mistaken
    // for `start` or `stop`.
    pyre_interpreter::type_methods::reject_kwargs_of(Some("array"), args, "index")?;
    pyre_interpreter::type_methods::arity_between(args, "index", 1, 3)?;
    // Optional start/stop, unwrapped via __index__, clamped like descr_index.
    // PyPy's interp2app gateway unwraps both bounds before `descr_index` reads
    // `self.len`.  Their `__index__` is user code: it can resize the array and
    // collect, so every later argument and the receiver itself is read back
    // from the rooted gateway stack copy.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(args);
    let mut start = if args.len() >= 3 {
        pyre_interpreter::sliceobject::eval_slice_index_not_none(
            pyre_object::gc_roots::shadow_stack_get(base + 2),
        )?
    } else {
        0
    };
    // [3.14-spec] An omitted stop is the largest index and growth from
    // `__eq__` stays searchable in 3.14, while `index_count_array` takes
    // `stop = min(stop, arr.len)` once before its loop and never revisits it.
    // Measured on 3.14.2, an `array('i', [1, 2, 3])` whose needle extends it
    // by two 7s answers `index()` 3; pypy3 7.3.20 raises ValueError.
    let mut stop = if args.len() >= 4 {
        pyre_interpreter::sliceobject::eval_slice_index_not_none(
            pyre_object::gc_roots::shadow_stack_get(base + 3),
        )?
    } else {
        i64::MAX
    };
    let obj = pyre_object::gc_roots::shadow_stack_get(base);
    let len = unsafe { arr::w_array_len(obj) } as i64;
    if start < 0 {
        start += len;
        if start < 0 {
            start = 0;
        }
    }
    if stop < 0 {
        stop += len;
        if stop < 0 {
            stop = 0;
        }
    }
    let mut i = start;
    while i < stop {
        let obj = pyre_object::gc_roots::shadow_stack_get(base);
        // `index_count_array` reads PyPy's separately allocated raw buffer;
        // shrinking `self.len` from `__eq__` leaves that storage addressable.
        // pyre's Vec length is the storage bound, so uphold
        // `w_array_unpack_item`'s safety contract explicitly.  CPython 3.14
        // likewise stops the search when comparison shrinks the array.
        if i >= unsafe { arr::w_array_len(obj) } as i64 {
            break;
        }
        let w_item = array_w_getitem(obj, i as usize, false)?;
        let _elem_roots = pyre_object::gc_roots::push_roots();
        let w_item = pyre_object::gc_roots::pin_root(w_item);
        let w_value = pyre_object::gc_roots::shadow_stack_get(base + 1);
        if pyre_interpreter::baseobjspace::eq_w(w_item, w_value)? {
            return Ok(pyre_object::w_int_new(i));
        }
        i += 1;
    }
    Err(PyError::value_error("array.index(x): x not in array"))
}

fn array_clear_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "clear", true)?;
    check_arity(args, 1, "array.clear")?;
    // descr_clear — empty the buffer, preserving typecode/itemsize.
    array_check_resize(args[0])?;
    unsafe { arr::w_array_vec_mut(args[0]) }.clear();
    Ok(pyre_object::w_none())
}

fn array_release_buffer(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__release_buffer__", false)?;
    pyre_interpreter::builtins::buffer_exporter_release_view(args[0], args[1])
}

fn array_buffer(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__buffer__", false)?;
    let mut obj = args[0];
    let flags = pyre_object::with_roots!(obj => pyre_interpreter::baseobjspace::c_int_w(args[1]))?;
    pyre_interpreter::builtins::w_memoryview_new_native_with_flags(obj, flags)
}

fn array_count_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "count", true)?;
    pyre_interpreter::type_methods::reject_kwargs_of(Some("array"), args, "count")?;
    if args.len() != 2 {
        return Err(PyError::type_error(format!(
            "array.count() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    Ok(pyre_object::w_int_new(array_index_count(
        args[0], args[1], true,
    )?))
}

fn array_reverse_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "reverse", true)?;
    check_arity(args, 1, "array.reverse")?;
    let obj = args[0];
    let isz = unsafe { arr::w_array_itemsize(obj) };
    let len = unsafe { arr::w_array_len(obj) };
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    let mut lo = 0usize;
    let mut hi = len;
    while lo < hi {
        hi -= 1;
        for k in 0..isz {
            vec.swap(lo * isz + k, hi * isz + k);
        }
        lo += 1;
    }
    Ok(pyre_object::w_none())
}

// ──────────────────────────────────────────────────────────────────────
// Conversion methods.
// ──────────────────────────────────────────────────────────────────────

fn array_tolist_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "tolist", true)?;
    check_arity(args, 1, "array.tolist")?;
    let obj = args[0];
    let len = unsafe { arr::w_array_len(obj) };
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::pin_roots(&[obj]);
    let items_base = pyre_object::gc_roots::shadow_stack_len();
    for i in 0..len {
        let item = array_w_getitem(pyre_object::gc_roots::shadow_stack_get(obj_slot), i, false)?;
        let _ = pyre_object::gc_roots::pin_root(item);
    }
    let mut items = Vec::with_capacity(len);
    for i in 0..len {
        items.push(pyre_object::gc_roots::shadow_stack_get(items_base + i));
    }
    Ok(pyre_object::w_list_new(items))
}

fn array_fromlist_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "fromlist", true)?;
    pyre_interpreter::type_methods::reject_kwargs_of(Some("array"), args, "fromlist")?;
    if args.len() != 2 {
        return Err(PyError::type_error(format!(
            "array.fromlist() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    // PyPy `W_ArrayBase.descr_fromlist` rejects non-lists before entering
    // `fromsequence`, then restores the old length if conversion fails.
    if !unsafe { pyre_object::is_list(args[1]) } {
        return Err(PyError::type_error("arg must be list"));
    }

    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(args);
    let item_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(PY_NULL);
    let obj = pyre_object::gc_roots::shadow_stack_get(base);
    let w_list = pyre_object::gc_roots::shadow_stack_get(base + 1);
    let n = unsafe { pyre_object::listobject::w_list_len(w_list) };
    if n == 0 {
        return Ok(pyre_object::w_none());
    }

    // Both PyPy `W_Array.fromsequence` and CPython
    // `array_array_fromlist_impl` size the destination before converting any
    // item.  That makes an existing export fail here, while an export created
    // by an item's `__index__` does not block writing the already-sized slot.
    array_check_resize(obj)?;
    let itemsize = unsafe { arr::w_array_itemsize(obj) };
    let old_bytes = unsafe { arr::w_array_bytes(obj) }.to_vec();
    let added = n
        .checked_mul(itemsize)
        .ok_or_else(|| PyError::memory_error(""))?;
    let new_len = old_bytes
        .len()
        .checked_add(added)
        .ok_or_else(|| PyError::memory_error(""))?;
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    vec.try_reserve(added)
        .map_err(|_| PyError::memory_error(""))?;
    vec.resize(new_len, 0);

    for i in 0..n {
        let w_list = pyre_object::gc_roots::shadow_stack_get(base + 1);
        let Some(w_item) = (unsafe { pyre_object::listobject::w_list_getitem(w_list, i as i64) })
        else {
            let obj = pyre_object::gc_roots::shadow_stack_get(base);
            array_restore_fromlist(obj, &old_bytes)?;
            return Err(PyError::runtime_error("list changed size during iteration"));
        };
        pyre_object::gc_roots::shadow_stack_set(item_slot, w_item);

        // [3.14-spec] CPython `array_array_fromlist_impl` uses PyList_GET_ITEM
        // even for list subclasses and checks the live list size after each
        // conversion.  PyPy's `fromsequence` can instead fall through to
        // `_fromiterable` for a subclass and its installed oracle snapshots
        // list-size changes.  Keep PyPy's pre-size/rollback structure, but use
        // the 3.14 observable direct-list gateway and mutation error.
        let obj = pyre_object::gc_roots::shadow_stack_get(base);
        let len_before = unsafe { arr::w_array_len(obj) };
        let slot = len_before
            .checked_sub(n)
            .and_then(|base| base.checked_add(i));
        let typecode = unsafe { arr::w_array_typecode(obj) };
        let mut packed: Bytes = [0; 8];
        let packed_len = match pack_into(
            typecode,
            pyre_object::gc_roots::shadow_stack_get(item_slot),
            &mut packed,
        ) {
            Ok(packed_len) => packed_len,
            Err(error) => {
                let obj = pyre_object::gc_roots::shadow_stack_get(base);
                array_restore_fromlist(obj, &old_bytes)?;
                return Err(error);
            }
        };

        let obj = pyre_object::gc_roots::shadow_stack_get(base);
        let live_len = unsafe { arr::w_array_len(obj) };
        if let Some(slot) = slot {
            if slot < live_len {
                let start = slot * itemsize;
                let vec = unsafe { arr::w_array_vec_mut(obj) };
                vec[start..start + itemsize].copy_from_slice(&packed[..packed_len]);
            }
        }

        let w_list = pyre_object::gc_roots::shadow_stack_get(base + 1);
        if unsafe { pyre_object::listobject::w_list_len(w_list) } != n {
            let obj = pyre_object::gc_roots::shadow_stack_get(base);
            array_restore_fromlist(obj, &old_bytes)?;
            return Err(PyError::runtime_error("list changed size during iteration"));
        }
    }
    Ok(pyre_object::w_none())
}

fn array_restore_fromlist(obj: PyObjectRef, old_bytes: &[u8]) -> Result<(), PyError> {
    // PyPy `W_ArrayBase.descr_fromlist` calls `setlen(s)` on every conversion
    // failure.  pyre's logical length is the Vec byte length, so restore the
    // corresponding pre-call byte prefix.  Respect a buffer export created by
    // conversion just as a fresh resize would.
    array_check_resize(obj)?;
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    vec.clear();
    vec.extend_from_slice(old_bytes);
    Ok(())
}

fn array_tobytes_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "tobytes", true)?;
    check_arity(args, 1, "array.tobytes")?;
    let bytes = unsafe { arr::w_array_bytes(args[0]) };
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(bytes))
}

fn array_frombytes_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "frombytes", true)?;
    check_arity(args, 2, "array.frombytes")?;
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(args);
    // PyPy `descr_frombytes` enters `bufferstr_w` / `acquire_readbuf` and
    // copies a BUF_SIMPLE view before `_frombytes`.  Use the same paired
    // acquisition rather than narrowing the public method to bytes and
    // bytearray.
    let source = pyre_object::gc_roots::shadow_stack_get(base + 1);
    let Some(buffer) = pyre_interpreter::baseobjspace::simple_buffer_bytes(source)? else {
        return Err(PyError::type_error(format!(
            "a bytes-like object is required, not '{}'",
            pyre_interpreter::type_methods::arg_type_name(source)
        )));
    };
    // [3.14-spec] `arraymodule.c::frombytes` rejects a typed Py_buffer even
    // though PyPy's `bufferstr_w` accepts its raw bytes.  The itemsize check is
    // observable for `memoryview(array('i'))` and `array('i')` exporters.
    if buffer.itemsize() != 1 {
        let mut error = PyError::type_error("a bytes-like object is required");
        let error_slot = error.pin(&_roots);
        buffer.release();
        error.reload(&_roots, error_slot);
        return Err(error);
    }
    let bytes = buffer.as_bytes().to_vec();
    // [3.14-spec] CPython keeps the Py_buffer export active through
    // `array_resize`, so `a.frombytes(a)` raises BufferError for an itemsize-1
    // array.  PyPy releases its copied buffer before `_frombytes`; retaining
    // this lease is the minimal lifetime difference forced by that visible
    // result.  Release errors remain unraisable and never replace `result`.
    match array_frombytes(pyre_object::gc_roots::shadow_stack_get(base), &bytes) {
        Ok(()) => {
            buffer.release();
            Ok(pyre_object::w_none())
        }
        Err(error) => {
            let mut error = error;
            let error_slot = error.pin(&_roots);
            buffer.release();
            error.reload(&_roots, error_slot);
            Err(error)
        }
    }
}

fn call_method(obj: PyObjectRef, name: &str, args: &[PyObjectRef]) -> PyResult {
    let result = pyre_interpreter::baseobjspace::call_method(obj, name, args);
    if result.is_null() {
        Err(pyre_interpreter::call::take_call_error()
            .unwrap_or_else(|| PyError::runtime_error("method call failed")))
    } else {
        Ok(result)
    }
}

fn array_fromfile_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "fromfile", true)?;
    check_arity(args, 3, "array.fromfile")?;
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.publish(args);
    roots.normalize(base, args.len());
    // PyPy `W_ArrayBase.descr_fromfile` runs its `@unwrap_spec(n=int)` gateway
    // before reading `self.itemsize`.  The callback may resize and collect the
    // receiver or file, so every input is reloaded from the published slots.
    let count = array_ssize_index_w(roots.get(base + 2))?;
    if count < 0 {
        return Err(PyError::value_error("negative count"));
    }
    let obj = roots.get(base);
    let size = (unsafe { arr::w_array_itemsize(obj) } as i64)
        .checked_mul(count)
        .ok_or_else(|| PyError::memory_error(""))?;
    // `ovfcheck(self.itemsize * n)` is a signed Py_ssize_t operation in PyPy;
    // do not let a product in usize's upper half wrap when it is boxed below.
    let size_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(pyre_object::w_int_new(size));
    let w_bytes = call_method(roots.get(base + 1), "read", &[roots.get(size_slot)])?;
    if !unsafe { pyre_object::is_bytes(w_bytes) } {
        return Err(PyError::type_error("read() didn't return bytes"));
    }
    let bytes = unsafe { pyre_object::bytesobject::bytes_like_data(w_bytes) }.to_vec();
    // PyPy `descr_fromfile` calls `_frombytes` before reporting its short-read
    // EOF, so an aligned prefix is appended and a non-aligned result fails in
    // `_frombytes` first.  [3.14-spec] `array_array_fromfile_impl` makes any
    // length unequal to the request an EOF (including an over-read) and uses
    // the byte-oriented sentence below.
    array_frombytes(roots.get(base), &bytes)?;
    if bytes.len() != size as usize {
        return Err(PyError::new(
            PyErrorKind::EOFError,
            "read() didn't return enough bytes",
        ));
    }
    Ok(pyre_object::w_none())
}

fn array_tofile_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "tofile", true)?;
    check_arity(args, 2, "array.tofile")?;
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[args[0], args[1]]);
    let w_bytes = array_tobytes_method(&[pyre_object::gc_roots::shadow_stack_get(base)])?;
    let bytes_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_bytes);
    call_method(
        pyre_object::gc_roots::shadow_stack_get(base + 1),
        "write",
        &[pyre_object::gc_roots::shadow_stack_get(bytes_slot)],
    )?;
    Ok(pyre_object::w_none())
}

fn array_tounicode_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "tounicode", true)?;
    check_arity(args, 1, "array.tounicode")?;
    let _roots = pyre_object::gc_roots::push_roots();
    let obj = pyre_object::gc_roots::pin_root(args[0]);
    let typecode = unsafe { arr::w_array_typecode(obj) };
    if !matches!(typecode, b'u' | b'w') {
        // [3.14-spec] PyPy `descr_tounicode` names only type `u`; v3.14.6
        // `array_array_tounicode_impl` exposes both unicode typecodes in the
        // public sentence now that `w` is part of the module surface.
        return Err(PyError::value_error(
            "tounicode() may only be called on unicode type arrays ('u' or 'w')",
        ));
    }
    let bytes = unsafe { arr::w_array_bytes(obj) };
    if typecode == b'w' {
        // `array_array_tounicode_impl` at v3.14.6 routes `w` through the
        // native-order UTF-32 decoder. Reuse PyPy's
        // `str_decode_utf_32_helper` port so BOM consumption, surrogate
        // rejection, error spans, and byte-order-specific codec names stay
        // identical to the public codec rather than growing a second loop.
        let (decoded, _, _) = pyre_interpreter::type_methods::decode_utf16_32_helper(
            bytes, true, None, "utf-32", "strict", true,
        )?;
        return Ok(pyre_object::unicodeobject::w_str_from_wtf8_managed(decoded));
    }

    // PyPy `descr_tounicode` uses `wcharpsize2utf8`, which admits lone
    // surrogates and raises for the first value above U+10ffff. The same
    // shape implements `PyUnicode_FromWideChar`, used for `u` by v3.14.6.
    let mut wb = Wtf8Buf::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let code = u32::from_ne_bytes(chunk.try_into().unwrap());
        let point = CodePoint::from_u32(code).ok_or_else(|| {
            PyError::value_error(format!(
                "character U+{code:x} is not in range [U+0000; U+10ffff]"
            ))
        })?;
        wb.push(point);
    }
    Ok(pyre_object::unicodeobject::w_str_from_wtf8_managed(wb))
}

fn array_fromunicode_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "fromunicode", true)?;
    check_arity(args, 2, "array.fromunicode")?;
    array_fromunicode(args[0], args[1])?;
    Ok(pyre_object::w_none())
}

// ──────────────────────────────────────────────────────────────────────
// Misc / dunder.
// ──────────────────────────────────────────────────────────────────────

fn array_contains_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__contains__", false)?;
    check_arity(args, 2, "array.__contains__")?;
    Ok(pyre_object::w_bool_from(
        array_index_count(args[0], args[1], false)? >= 0,
    ))
}

fn array_buffer_info_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "buffer_info", true)?;
    check_arity(args, 1, "array.buffer_info")?;
    let obj = args[0];
    let addr = unsafe { arr::w_array_buffer_address(obj) } as u64;
    let len = unsafe { arr::w_array_len(obj) } as i64;
    // PyPy boxes `_buffer_as_unsigned` as an unsigned integer; keep a pointer
    // whose top bit is set non-negative rather than narrowing it through i64.
    let w_addr = if addr <= i64::MAX as u64 {
        pyre_object::w_int_new(addr as i64)
    } else {
        pyre_object::longobject::w_long_new(BigInt::from(addr))
    };
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(w_addr);
    fields.push(pyre_object::w_int_new(len));
    Ok(pyre_object::w_tuple_new(fields.take()))
}

fn array_byteswap_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "byteswap", true)?;
    check_arity(args, 1, "array.byteswap")?;
    let obj = args[0];
    let isz = unsafe { arr::w_array_itemsize(obj) };
    if !matches!(isz, 1 | 2 | 4 | 8) {
        return Err(PyError::runtime_error(
            "don't know how to byteswap this array type",
        ));
    }
    let len = unsafe { arr::w_array_len(obj) };
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    for i in 0..len {
        vec[i * isz..i * isz + isz].reverse();
    }
    Ok(pyre_object::w_none())
}

fn array_copy_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__copy__", true)?;
    check_arity_range(args, 1, 2, "array.__copy__")?;
    let obj = args[0];
    let tc = unsafe { arr::w_array_typecode(obj) };
    let isz = unsafe { arr::w_array_itemsize(obj) } as u8;
    let bytes = unsafe { arr::w_array_bytes(obj) }.to_vec();
    Ok(arr::w_array_from_bytes(tc, isz, bytes))
}

/// `array.__repr__` formatting (`interp_array.py descr_repr`).  Shared with
/// `display::py_repr` so an array nested in a list / error / tuple formats
/// the same way.
pub fn array_repr_wtf8(obj: PyObjectRef) -> Result<rustpython_wtf8::Wtf8Buf, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let current_obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
    // PyPy `W_ArrayBase.descr_repr` obtains
    // `space.type(self).getname(space)`: the Python-visible `__name__`, not
    // the TypeDef's module-qualified registration name.  In particular the
    // builtin TypeDef is registered as `array.array` but formats as `array`,
    // while a subclass keeps its live (and assignable) `__name__`.
    let w_type = pyre_interpreter::typedef::r#type(current_obj())
        .expect("an array instance must have a Python type object");
    let class_name = unsafe {
        pyre_interpreter::baseobjspace::str_utf8_w(pyre_object::w_type_get_name_obj(
            w_type.as_ptr(),
        ))?
        .to_string()
    };
    let tc = unsafe { arr::w_array_typecode(current_obj()) } as char;
    let len = unsafe { arr::w_array_len(current_obj()) };
    if len == 0 {
        return Ok(rustpython_wtf8::Wtf8Buf::from_string(format!(
            "{class_name}('{tc}')"
        )));
    }
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    out.push_str(&format!("{class_name}('{tc}', "));
    if matches!(tc, 'u' | 'w') {
        // [3.14-spec] PyPy `descr_repr` catches this ValueError and prints a
        // `<character ...>` placeholder. v3.14.6 `array_repr` propagates any
        // failure from `array_array_tounicode_impl`, including the structured
        // UTF-32 error for `w`.
        let s = array_tounicode_method(&[current_obj()])?;
        out.push_wtf8(&unsafe { pyre_interpreter::display::py_repr_wtf8(s)? });
        out.push_str(")");
        return Ok(out);
    }
    out.push_str("[");
    for i in 0..len {
        if i != 0 {
            out.push_str(", ");
        }
        let w_item = array_w_getitem(current_obj(), i, false)?;
        out.push_wtf8(&unsafe { pyre_interpreter::display::py_repr_wtf8(w_item)? });
    }
    out.push_str("])");
    Ok(out)
}

fn array_repr_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__repr__", false)?;
    check_arity(args, 1, "array.__repr__")?;
    Ok(pyre_object::w_str_from_wtf8_managed(array_repr_wtf8(
        args[0],
    )?))
}

// `interp_array.py compare_arrays`: compare each element with the requested
// operation.  In particular, an unordered pair such as NaN is neither less
// nor greater; treating every non-equal/non-less pair as greater is wrong.
fn array_richcompare(mut a: PyObjectRef, mut b: PyObjectRef, op: u8) -> PyResult {
    if !unsafe { arr::is_array(a) } || !unsafe { arr::is_array(b) } {
        return Ok(pyre_object::w_not_implemented());
    }
    let la = unsafe { arr::w_array_len(a) };
    let lb = unsafe { arr::w_array_len(b) };
    if op == 0 && la != lb {
        return Ok(pyre_object::w_bool_from(false));
    }
    if op == 1 && la != lb {
        return Ok(pyre_object::w_bool_from(true));
    }
    let n = la.min(lb);
    let typecode_a = unsafe { arr::w_array_typecode(a) };
    let typecode_b = unsafe { arr::w_array_typecode(b) };
    // PyPy `compare_arrays` requests `integer_instead_of_char` so malformed
    // unicode storage remains comparable. [3.14-spec] v3.14.6 does that raw
    // comparison only when both descriptors match; differing typecodes take
    // `getarrayitem` and can therefore raise while boxing an invalid scalar.
    let integer_instead_of_char = typecode_a == typecode_b && matches!(typecode_a, b'u' | b'w');
    for i in 0..n {
        let mut ea =
            pyre_object::with_roots!(a, b => array_w_getitem(a, i, integer_instead_of_char))?;
        let mut eb =
            pyre_object::with_roots!(a, b, ea => array_w_getitem(b, i, integer_instead_of_char))?;
        match op {
            0 => {
                let r = pyre_object::with_roots!(a, b => compare(ea, eb, CompareOp::Eq))?;
                if !pyre_object::with_roots!(a, b => pyre_interpreter::baseobjspace::is_true(r))? {
                    return Ok(pyre_object::w_bool_from(false));
                }
            }
            1 => {
                let r = pyre_object::with_roots!(a, b => compare(ea, eb, CompareOp::Ne))?;
                if pyre_object::with_roots!(a, b => pyre_interpreter::baseobjspace::is_true(r))? {
                    return Ok(pyre_object::w_bool_from(true));
                }
            }
            2 | 4 => {
                let cmp = if op == 2 {
                    CompareOp::Lt
                } else {
                    CompareOp::Gt
                };
                let r = pyre_object::with_roots!(a, b, ea, eb => compare(ea, eb, cmp))?;
                if pyre_object::with_roots!(a, b, ea, eb => pyre_interpreter::baseobjspace::is_true(r))?
                {
                    return Ok(pyre_object::w_bool_from(true));
                }
                let r = pyre_object::with_roots!(a, b => compare(ea, eb, CompareOp::Eq))?;
                if !pyre_object::with_roots!(a, b => pyre_interpreter::baseobjspace::is_true(r))? {
                    return Ok(pyre_object::w_bool_from(false));
                }
            }
            3 | 5 => {
                let cmp = if op == 3 {
                    CompareOp::Le
                } else {
                    CompareOp::Ge
                };
                let r = pyre_object::with_roots!(a, b, ea, eb => compare(ea, eb, cmp))?;
                if !pyre_object::with_roots!(a, b, ea, eb => pyre_interpreter::baseobjspace::is_true(r))?
                {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let r = pyre_object::with_roots!(a, b => compare(ea, eb, CompareOp::Eq))?;
                if !pyre_object::with_roots!(a, b => pyre_interpreter::baseobjspace::is_true(r))? {
                    return Ok(pyre_object::w_bool_from(true));
                }
            }
            _ => unreachable!(),
        }
    }
    let result = match op {
        0 => true,
        1 => false,
        2 => la < lb,
        3 => la <= lb,
        4 => la > lb,
        5 => la >= lb,
        _ => unreachable!(),
    };
    Ok(pyre_object::w_bool_from(result))
}

fn array_eq_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__eq__", false)?;
    check_arity(args, 2, "array.__eq__")?;
    array_richcompare(args[0], args[1], 0)
}
fn array_ne_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__ne__", false)?;
    check_arity(args, 2, "array.__ne__")?;
    array_richcompare(args[0], args[1], 1)
}
fn array_lt_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__lt__", false)?;
    check_arity(args, 2, "array.__lt__")?;
    array_richcompare(args[0], args[1], 2)
}
fn array_le_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__le__", false)?;
    check_arity(args, 2, "array.__le__")?;
    array_richcompare(args[0], args[1], 3)
}
fn array_gt_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__gt__", false)?;
    check_arity(args, 2, "array.__gt__")?;
    array_richcompare(args[0], args[1], 4)
}
fn array_ge_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__ge__", false)?;
    check_arity(args, 2, "array.__ge__")?;
    array_richcompare(args[0], args[1], 5)
}

// Arithmetic.
fn array_add_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__add__", false)?;
    check_arity(args, 2, "array.__add__")?;
    let a = args[0];
    let b = args[1];
    if !unsafe { arr::is_array(b) } {
        return Err(PyError::type_error(format!(
            "can only append array (not \"{}\") to array",
            pyre_interpreter::error::type_name_of(b)
        )));
    }
    let tc = unsafe { arr::w_array_typecode(a) };
    if unsafe { arr::w_array_typecode(b) } != tc {
        return Err(PyError::type_error(
            "bad argument type for built-in operation",
        ));
    }
    let isz = unsafe { arr::w_array_itemsize(a) } as u8;
    let mut out = unsafe { arr::w_array_bytes(a) }.to_vec();
    out.extend_from_slice(unsafe { arr::w_array_bytes(b) });
    Ok(arr::w_array_from_bytes(tc, isz, out))
}

fn array_iadd_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__iadd__", false)?;
    check_arity(args, 2, "array.__iadd__")?;
    let a = args[0];
    array_check_resize(a)?;
    let b = args[1];
    // `array_inplace_concat` refuses a non-array itself and hands the
    // same-kind test to `array_do_extend`, which states the other sentence.
    if !unsafe { arr::is_array(b) } {
        return Err(PyError::type_error(format!(
            "can only extend array with array (not \"{}\")",
            pyre_interpreter::error::type_name_of(b)
        )));
    }
    if unsafe { arr::w_array_typecode(b) } != unsafe { arr::w_array_typecode(a) } {
        return Err(PyError::type_error(
            "can only extend with array of same kind",
        ));
    }
    let src = unsafe { arr::w_array_bytes(b) }.to_vec();
    let vec = unsafe { arr::w_array_vec_mut(a) };
    vec.extend_from_slice(&src);
    Ok(a)
}

fn array_repeat_bytes(obj: PyObjectRef, count: i64) -> PyResult {
    let tc = unsafe { arr::w_array_typecode(obj) };
    let isz = unsafe { arr::w_array_itemsize(obj) } as u8;
    let src = unsafe { arr::w_array_bytes(obj) };
    let n = count.max(0) as usize;
    // ovfcheck(oldlen * repeat) -> MemoryError on overflow (_mul_helper).
    let total = src
        .len()
        .checked_mul(n)
        .ok_or_else(|| PyError::memory_error(""))?;
    // The count came from Python, so the buffer is reserved fallibly: an
    // infallible reservation ends the process where `_mul_helper` raises.
    let mut out = pyre_interpreter::builtins::try_vec_with_capacity::<u8>(total)?;
    // `array_repeat` copies only for `oldbytes > 0`; without that, repeating an
    // empty array runs the count as a loop trip and never comes back.
    if !src.is_empty() {
        for _ in 0..n {
            out.extend_from_slice(src);
        }
    }
    Ok(arr::w_array_from_bytes(tc, isz, out))
}

/// `sequence_repeat`'s count.  `array` carries `sq_repeat` and no
/// `nb_multiply`, so the binary operator reaches the repeat with whatever it
/// was handed: an operand with no `__index__` at all is refused as a repeat,
/// naming its own type, rather than by the conversion behind that check.
fn array_repeat_count(w_count: PyObjectRef) -> Result<i64, PyError> {
    if unsafe { pyre_interpreter::baseobjspace::lookup(w_count, "__index__") }.is_none() {
        return Err(PyError::type_error(format!(
            "can't multiply sequence by non-int of type '{}'",
            pyre_interpreter::type_methods::arg_type_name(w_count)
        )));
    }
    pyre_interpreter::builtins::getindex_w(w_count)
}

fn array_rmul_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__rmul__", false)?;
    array_mul_method(args)
}

fn array_deepcopy_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__deepcopy__", true)?;
    array_copy_method(args)
}

fn array_mul_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__mul__", false)?;
    check_arity(args, 2, "array.__mul__")?;
    let mut obj = args[0];
    let count = pyre_object::with_roots!(obj => array_repeat_count(args[1]))?;
    array_repeat_bytes(obj, count)
}

fn array_imul_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__imul__", false)?;
    check_arity(args, 2, "array.__imul__")?;
    let mut obj = args[0];
    array_check_resize(obj)?;
    let count = pyre_object::with_roots!(obj => array_repeat_count(args[1]))?.max(0) as usize;
    let src = unsafe { arr::w_array_bytes(obj) }.to_vec();
    if count == 0 {
        unsafe { arr::w_array_vec_mut(obj) }.clear();
        return Ok(obj);
    }
    // ovfcheck(oldlen * repeat) -> MemoryError on overflow.
    let extra = src
        .len()
        .checked_mul(count - 1)
        .ok_or_else(|| PyError::memory_error(""))?;
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    vec.try_reserve_exact(extra)
        .map_err(|_| pyre_interpreter::builtins::reservation_failed())?;
    if !src.is_empty() {
        for _ in 1..count {
            vec.extend_from_slice(&src);
        }
    }
    Ok(obj)
}

// ──────────────────────────────────────────────────────────────────────
// Pickle: __reduce_ex__ + _array_reconstructor.
// ──────────────────────────────────────────────────────────────────────

const UNKNOWN_FORMAT: i64 = -1;

fn array_machine_format_code(typecode: u8, itemsize: usize) -> i64 {
    let big_endian = cfg!(target_endian = "big") as i64;
    match typecode {
        b'B' => 0,
        b'b' => 1,
        b'H' | b'I' | b'L' | b'Q' => match itemsize {
            2 => 2 + big_endian,
            4 => 6 + big_endian,
            8 => 10 + big_endian,
            _ => UNKNOWN_FORMAT,
        },
        b'h' | b'i' | b'l' | b'q' => match itemsize {
            2 => 4 + big_endian,
            4 => 8 + big_endian,
            8 => 12 + big_endian,
            _ => UNKNOWN_FORMAT,
        },
        b'f' => 14 + big_endian,
        b'd' => 16 + big_endian,
        b'u' | b'w' => match itemsize {
            2 => 18 + big_endian,
            4 => 20 + big_endian,
            _ => UNKNOWN_FORMAT,
        },
        _ => UNKNOWN_FORMAT,
    }
}

/// `interp_array.py:descr_reduce_ex`: protocols below 3 use the portable
/// list form; newer protocols use the module's canonical reconstructor and
/// native machine-format bytes.
fn array_reduce_ex_method(args: &[PyObjectRef]) -> PyResult {
    require_array_receiver(args, "__reduce_ex__", true)?;
    check_arity(args, 2, "array.__reduce_ex__")?;
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::pin_roots(&[args[0], args[1]]);
    let protocol = pyre_interpreter::baseobjspace::int_w(pyre_object::gc_roots::shadow_stack_get(
        obj_slot + 1,
    ))?;
    let obj = || pyre_object::gc_roots::shadow_stack_get(obj_slot);
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(
        pyre_interpreter::typedef::r#type(obj()).map_or(PY_NULL, |p| p.as_ptr()),
    );
    let typecode = unsafe { arr::w_array_typecode(obj()) };
    let tc = typecode as char;
    let typecode_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(&tc.to_string()));
    let w_dict = pyre_interpreter::baseobjspace::findattr_result(obj(), "__dict__")?
        .unwrap_or_else(pyre_object::w_none);
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_dict);
    let mformat = array_machine_format_code(typecode, unsafe { arr::w_array_itemsize(obj()) });
    if protocol < 3 || mformat == UNKNOWN_FORMAT {
        let w_items = array_tolist_method(&[obj()])?;
        let items_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_items);
        let ctor_args = pyre_object::w_tuple_new(vec![
            pyre_object::gc_roots::shadow_stack_get(typecode_slot),
            pyre_object::gc_roots::shadow_stack_get(items_slot),
        ]);
        let ctor_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(ctor_args);
        return Ok(pyre_object::w_tuple_new(vec![
            pyre_object::gc_roots::shadow_stack_get(type_slot),
            pyre_object::gc_roots::shadow_stack_get(ctor_slot),
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
        ]));
    }

    let module = pyre_interpreter::importing::get_sys_module("array")
        .ok_or_else(|| PyError::runtime_error("array module not initialized"))?;
    let reconstructor =
        pyre_interpreter::baseobjspace::getattr_str(module, "_array_reconstructor")?;
    let reconstructor_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(reconstructor);
    let w_bytes = array_tobytes_method(&[obj()])?;
    let bytes_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_bytes);
    let format_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(mformat));
    let ctor_args = pyre_object::w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(type_slot),
        pyre_object::gc_roots::shadow_stack_get(typecode_slot),
        pyre_object::gc_roots::shadow_stack_get(format_slot),
        pyre_object::gc_roots::shadow_stack_get(bytes_slot),
    ]);
    let ctor_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ctor_args);
    Ok(pyre_object::w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(reconstructor_slot),
        pyre_object::gc_roots::shadow_stack_get(ctor_slot),
        pyre_object::gc_roots::shadow_stack_get(dict_slot),
    ]))
}

fn decode_uint(bytes: &[u8], big_endian: bool) -> u64 {
    if big_endian {
        bytes
            .iter()
            .fold(0u64, |value, byte| (value << 8) | *byte as u64)
    } else {
        bytes
            .iter()
            .rev()
            .fold(0u64, |value, byte| (value << 8) | *byte as u64)
    }
}

fn reconstructor_descriptor(mformat: i64) -> (usize, bool, bool) {
    let big_endian = matches!(mformat, 3 | 5 | 7 | 9 | 11 | 13 | 15 | 17 | 19 | 21);
    let signed = matches!(mformat, 1 | 4 | 5 | 8 | 9 | 12 | 13);
    let size = match mformat {
        0 | 1 => 1,
        2..=5 => 2,
        6..=9 | 14 | 15 => 4,
        10..=13 | 16 | 17 => 8,
        18 | 19 => 2,
        20 | 21 => 4,
        _ => 0,
    };
    (size, signed, big_endian)
}

/// CPython 3.14 `array__array_reconstructor_impl`: when integer bytes came
/// from an architecture with a different C integer layout, select the native
/// integer typecode with the same width and signedness before construction.
fn reconstructor_integer_typecode(size: usize, signed: bool) -> u8 {
    match (size, signed) {
        (1, true) => b'b',
        (1, false) => b'B',
        (2, true) => b'h',
        (2, false) => b'H',
        (4, true) => b'i',
        (4, false) => b'I',
        // CPython scans all descriptors and retains the last match.  On the
        // supported 64-bit targets that is q/Q rather than long/unsigned long.
        (8, true) => b'q',
        (8, false) => b'Q',
        _ => unreachable!("validated machine-format integer width"),
    }
}

/// `reconstructor.py:array_reconstructor`: use native bytes on the matching
/// machine-format fast path, otherwise decode each source item before the
/// requested array class repacks it in native representation.
fn array_reconstructor(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 4 {
        return Err(PyError::type_error(
            "_array_reconstructor() takes exactly 4 arguments",
        ));
    }
    // `args` is the gateway's native copy; the conversions below can collect.
    let mut w_cls = args[0];
    let mut w_typecode = args[1];
    let mut w_mformat_arg = args[2];
    let mut w_bytes = args[3];
    if !unsafe { pyre_object::is_type(w_cls) } {
        return Err(PyError::type_error(format!(
            "_array_reconstructor() argument 1 must be type, not {}",
            pyre_interpreter::type_methods::clinic_arg_type_name(w_cls)
        )));
    }
    let array_type =
        pyre_interpreter::typedef::gettypeobject(&pyre_object::interp_array::ARRAY_TYPE);
    if !pyre_object::with_roots!(w_cls, w_typecode, w_mformat_arg, w_bytes =>
        pyre_interpreter::baseobjspace::issubclass(w_cls, array_type))?
    {
        return Err(PyError::type_error(format!(
            "{} is not a subtype of array.array",
            unsafe { pyre_object::w_type_get_name(w_cls) }
        )));
    }
    if !unsafe { pyre_object::is_str(w_typecode) } {
        return Err(PyError::type_error(format!(
            "_array_reconstructor() argument 2 must be a unicode character, not {}",
            pyre_interpreter::type_methods::clinic_arg_type_name(w_typecode)
        )));
    }
    let typecode_bytes =
        unsafe { pyre_object::unicodeobject::w_str_get_wtf8(w_typecode) }.as_bytes();
    if typecode_bytes.len() != 1 || arr::typecode_itemsize(typecode_bytes[0]).is_none() {
        return Err(PyError::value_error("invalid type code"));
    }
    let typecode = typecode_bytes[0];
    let itemsize = arr::typecode_itemsize(typecode).unwrap() as usize;
    // mformat_code: int in [MACHINE_FORMAT_CODE_MIN, MACHINE_FORMAT_CODE_MAX].
    // The `int` converter takes `PyNumber_Index`, which names the type it
    // could not convert.
    let w_mformat = pyre_object::with_roots!(w_cls, w_typecode, w_bytes =>
        pyre_interpreter::baseobjspace::space_index(w_mformat_arg))?;
    let mformat = pyre_object::with_roots!(w_cls, w_typecode, w_bytes =>
        pyre_interpreter::baseobjspace::int_w(w_mformat))?;
    if !(0..=21).contains(&mformat) {
        return Err(PyError::value_error(
            "third argument must be a valid machine format code.",
        ));
    }
    if !unsafe { pyre_object::bytesobject::is_bytes_like(w_bytes) } {
        return Err(PyError::type_error(format!(
            "fourth argument should be bytes, not {}",
            pyre_interpreter::error::type_name_of(w_bytes)
        )));
    }
    let bytes = unsafe { pyre_object::bytesobject::bytes_like_data(w_bytes) }.to_vec();
    if mformat == array_machine_format_code(typecode, itemsize) {
        let obj = array_descr_new(&[w_cls, w_typecode])?;
        array_frombytes(obj, &bytes)?;
        return Ok(obj);
    }

    if matches!(mformat, 18..=21) {
        let obj = pyre_object::with_roots!(w_bytes => array_descr_new(&[w_cls, w_typecode]))?;
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = _roots.pin_roots(&[obj, w_bytes]);
        let encoding = match mformat {
            18 => "utf-16-le",
            19 => "utf-16-be",
            20 => "utf-32-le",
            _ => "utf-32-be",
        };
        let w_encoding = pyre_object::w_str_new_managed(encoding);
        let decoded = call_method(_roots.get(obj_slot + 1), "decode", &[w_encoding])?;
        array_fromunicode(_roots.get(obj_slot), decoded)?;
        return Ok(_roots.get(obj_slot));
    }

    let (source_size, signed, big_endian) = reconstructor_descriptor(mformat);
    if bytes.len() % source_size != 0 {
        return Err(PyError::value_error(
            "bytes length not a multiple of item size",
        ));
    }
    let output_typecode = if mformat <= 13 {
        reconstructor_integer_typecode(source_size, signed)
    } else {
        typecode
    };
    let output_typecode_text = (output_typecode as char).to_string();
    let w_output_typecode =
        pyre_object::with_roots!(w_cls => pyre_object::w_str_new_managed(&output_typecode_text));
    let mut obj = array_descr_new(&[w_cls, w_output_typecode])?;
    for chunk in bytes.chunks_exact(source_size) {
        let w_item = if matches!(mformat, 14 | 15) {
            let raw: [u8; 4] = chunk.try_into().unwrap();
            let value = if big_endian {
                f32::from_be_bytes(raw)
            } else {
                f32::from_le_bytes(raw)
            };
            pyre_object::w_float_new(value as f64)
        } else if matches!(mformat, 16 | 17) {
            let raw: [u8; 8] = chunk.try_into().unwrap();
            let value = if big_endian {
                f64::from_be_bytes(raw)
            } else {
                f64::from_le_bytes(raw)
            };
            pyre_object::w_float_new(value)
        } else {
            let raw = decode_uint(chunk, big_endian);
            if signed {
                let shift = 64 - source_size * 8;
                pyre_object::w_int_new(((raw << shift) as i64) >> shift)
            } else if raw <= i64::MAX as u64 {
                pyre_object::w_int_new(raw as i64)
            } else {
                pyre_object::with_roots!(obj =>
                    pyre_object::longobject::w_long_new(BigInt::from(raw))
                )
            }
        };
        pyre_object::with_roots!(obj => array_append_value(obj, w_item))?;
    }
    Ok(obj)
}

// ──────────────────────────────────────────────────────────────────────
// Type / module registration.
// ──────────────────────────────────────────────────────────────────────

// CPython 3.14 `arraymodule.c` `arraytype_doc`.  PyPy's
// `W_ArrayBase.typedef` leaves its TypeDef doc empty; the public 3.14 value is
// observable through both `array.array.__doc__` and the type dictionary, so
// only this metadata string departs from the PyPy owner/implementation shape.
const ARRAY_TYPE_DOC: &str = concat!(
    "array(typecode [, initializer]) -> array\n",
    "\n",
    "Return a new array whose items are restricted by typecode, and\n",
    "initialized from the optional initializer value, which must be a list,\n",
    "string or iterable over elements of the appropriate type.\n",
    "\n",
    "Arrays represent basic values and behave very much like lists, except\n",
    "the type of objects stored in them is constrained. The type is specified\n",
    "at object creation time by using a type code, which is a single character.\n",
    "The following type codes are defined:\n",
    "\n",
    "    Type code   C Type             Minimum size in bytes\n",
    "    'b'         signed integer     1\n",
    "    'B'         unsigned integer   1\n",
    "    'u'         Unicode character  2 (see note)\n",
    "    'h'         signed integer     2\n",
    "    'H'         unsigned integer   2\n",
    "    'i'         signed integer     2\n",
    "    'I'         unsigned integer   2\n",
    "    'l'         signed integer     4\n",
    "    'L'         unsigned integer   4\n",
    "    'q'         signed integer     8 (see note)\n",
    "    'Q'         unsigned integer   8 (see note)\n",
    "    'f'         floating-point     4\n",
    "    'd'         floating-point     8\n",
    "\n",
    "NOTE: The 'u' typecode corresponds to Python's unicode character. On\n",
    "narrow builds this is 2-bytes on wide builds this is 4-bytes.\n",
    "\n",
    "NOTE: The 'q' and 'Q' type codes are only available if the platform\n",
    "C compiler used to build Python supports 'long long', or, on Windows,\n",
    "'__int64'.\n",
    "\n",
    "Methods:\n",
    "\n",
    "append() -- append a new item to the end of the array\n",
    "buffer_info() -- return information giving the current memory info\n",
    "byteswap() -- byteswap all the items of the array\n",
    "count() -- return number of occurrences of an object\n",
    "extend() -- extend array by appending multiple elements from an iterable\n",
    "fromfile() -- read items from a file object\n",
    "fromlist() -- append items from the list\n",
    "frombytes() -- append items from the string\n",
    "index() -- return index of first occurrence of an object\n",
    "insert() -- insert a new item into the array at a provided position\n",
    "pop() -- remove and return item (default last)\n",
    "remove() -- remove first occurrence of an object\n",
    "reverse() -- reverse the order of the items in the array\n",
    "tofile() -- write all items to a file object\n",
    "tolist() -- return the array converted to an ordinary list\n",
    "tobytes() -- return the array converted to a string\n",
    "\n",
    "Attributes:\n",
    "\n",
    "typecode -- the typecode character used to create the array\n",
    "itemsize -- the length in bytes of one array item\n",
);

/// Register all `array.array` methods/getsets into the type namespace.
pub fn init_array_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_put_new!(ns_slot, "__doc__", pyre_object::w_str_new(ARRAY_TYPE_DOC));
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__new__",
        pyre_interpreter::typedef::make_new_descr(array_descr_new)
    );
    let m = |name: &'static str, f: fn(&[PyObjectRef]) -> PyResult, arity: u16| {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            name,
            make_builtin_function_with_arity(name, f, arity)
        );
    };
    m("__len__", array_len, 1);
    m("__iter__", array_iter, 1);
    m("__getitem__", array_getitem, 2);
    m("__setitem__", array_setitem, 3);
    m("__delitem__", array_delitem, 2);
    m("__contains__", array_contains_method, 2);
    m("__repr__", array_repr_method, 1);
    m("__eq__", array_eq_method, 2);
    m("__ne__", array_ne_method, 2);
    m("__lt__", array_lt_method, 2);
    m("__le__", array_le_method, 2);
    m("__gt__", array_gt_method, 2);
    m("__ge__", array_ge_method, 2);
    m("__add__", array_add_method, 2);
    m("__iadd__", array_iadd_method, 2);
    m("__mul__", array_mul_method, 2);
    m("__rmul__", array_rmul_method, 2);
    m("__imul__", array_imul_method, 2);
    m("__reduce_ex__", array_reduce_ex_method, 2);
    // `append` owns its CPython 3.14 fixed-owner keyword/arity gateway.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "append",
        pyre_interpreter::make_builtin_function("append", array_append_method)
    );
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "extend",
        pyre_interpreter::make_builtin_function("extend", array_extend_method)
    );
    // `insert` uses the PyArg_UnpackTuple arity wording and the fixed
    // `array.insert` keyword owner, both supplied by its gateway body.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "insert",
        pyre_interpreter::make_builtin_function("insert", array_insert_method)
    );
    // `remove` owns its CPython 3.14 fixed-owner keyword/arity gateway.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "remove",
        pyre_interpreter::make_builtin_function("remove", array_remove_method)
    );
    // `index` accepts optional start/stop.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "index",
        pyre_interpreter::make_builtin_function("index", array_index_method)
    );
    // `count` owns its CPython 3.14 fixed-owner keyword/arity gateway.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "count",
        pyre_interpreter::make_builtin_function("count", array_count_method)
    );
    m("clear", array_clear_method, 1);
    m("__release_buffer__", array_release_buffer, 2);
    m("__buffer__", array_buffer, 2);
    m("reverse", array_reverse_method, 1);
    m("tolist", array_tolist_method, 1);
    // `fromlist` owns its CPython 3.14 fixed-owner keyword/arity gateway.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "fromlist",
        pyre_interpreter::make_builtin_function("fromlist", array_fromlist_method)
    );
    m("tobytes", array_tobytes_method, 1);
    m("frombytes", array_frombytes_method, 2);
    m("tofile", array_tofile_method, 2);
    m("fromfile", array_fromfile_method, 3);
    m("tounicode", array_tounicode_method, 1);
    m("fromunicode", array_fromunicode_method, 2);
    m("buffer_info", array_buffer_info_method, 1);
    m("byteswap", array_byteswap_method, 1);
    m("__copy__", array_copy_method, 1);
    m("__deepcopy__", array_deepcopy_method, 2);
    // CPython 3.14 arraymodule.c:2471 — Py_GenericAlias with METH_CLASS.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__class_getitem__",
        pyre_object::function::w_classmethod_new(pyre_interpreter::make_builtin_function(
            "__class_getitem__",
            pyre_interpreter::_pypy_generic_alias::generic_alias_class_getitem,
        ))
    );
    // `pop` accepts an optional index.
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "pop",
        pyre_interpreter::make_builtin_function("pop", array_pop_method)
    );
    // typecode / itemsize read-only properties.
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "typecode",
            pyre_object::w_property_new(
                make_builtin_function_with_arity(
                    "typecode",
                    |args| {
                        let obj = require_array_receiver(args, "typecode", true)?;
                        let tc = arr::w_array_typecode(obj) as char;
                        Ok(pyre_object::w_str_new_managed(&tc.to_string()))
                    },
                    1,
                ),
                PY_NULL,
                PY_NULL,
                PY_NULL,
            )
        )
    };
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "itemsize",
            pyre_object::w_property_new(
                make_builtin_function_with_arity(
                    "itemsize",
                    |args| {
                        let obj = require_array_receiver(args, "itemsize", true)?;
                        Ok(pyre_object::w_int_new(arr::w_array_itemsize(obj) as i64))
                    },
                    1,
                ),
                PY_NULL,
                PY_NULL,
                PY_NULL,
            )
        )
    };
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__weakref__",
        pyre_interpreter::typedef::make_weakref_descr(PY_NULL)
    );
}

/// `array` module init — `moduledef.py interpleveldefs`.
pub fn init_array_module(ns: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // `W_Array.typedef` (`interp_array.py`) is a module TypeDef.  Empty
    // startup does not import array. Pin `ns` first: this build collects
    // and can drop the GIL.
    pyre_interpreter::typedef::ensure_array_typeobjects();
    let type_obj = pyre_interpreter::typedef::gettypeobject(&pyre_object::interp_array::ARRAY_TYPE);
    {
        let _pyre_store_roots = ::pyre_object::gc_roots::push_roots();
        let __pyre_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(type_obj);
        unsafe {
            ::pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ::pyre_object::gc_roots::shadow_stack_get(ns_slot),
                "array",
                pyre_object::gc_roots::shadow_stack_get(__pyre_slot),
            )
        };
        unsafe {
            ::pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ::pyre_object::gc_roots::shadow_stack_get(ns_slot),
                "ArrayType",
                pyre_object::gc_roots::shadow_stack_get(__pyre_slot),
            )
        };
    };
    pyre_interpreter::__pyre_put_new!(ns_slot, "typecodes", pyre_object::w_str_new(arr::TYPECODES));
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "_array_reconstructor",
        pyre_interpreter::make_builtin_function("_array_reconstructor", array_reconstructor)
    );
    Ok(())
}

/// PyPy `pypy/module/array/moduledef.py:Module.startup`: array is a
/// virtual `MutableSequence`, registered only after the builtin module is in
/// `sys.modules` so importing `_collections_abc` cannot create a second array
/// module during a cycle.
pub fn startup_array_module(
    _module: pyre_object::PyObjectRef,
    execution_context: *const pyre_interpreter::PyExecutionContext,
) -> Result<(), pyre_interpreter::PyError> {
    let abc_module = pyre_interpreter::importing::importhook(
        rustpython_wtf8::Wtf8::new("_collections_abc"),
        pyre_object::PY_NULL,
        pyre_object::w_tuple_new(vec![pyre_object::w_str_new("MutableSequence")]),
        0,
        execution_context,
    )?;
    let mutable_sequence =
        pyre_interpreter::baseobjspace::getattr_str(abc_module, "MutableSequence")?;
    let register = pyre_interpreter::baseobjspace::getattr_str(mutable_sequence, "register")?;
    let array_type =
        pyre_interpreter::typedef::gettypeobject(&pyre_object::interp_array::ARRAY_TYPE);
    pyre_interpreter::call::call_function_impl_result(register, &[array_type])?;
    Ok(())
}
