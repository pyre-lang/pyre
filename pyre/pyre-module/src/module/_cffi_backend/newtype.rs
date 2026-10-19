//! Building ctypes — PyPy:
//! `pypy/module/_cffi_backend/newtype.py`.
//!
//! `newtype.py` memoises primitives and singleton ctypes strongly through
//! `UniqueCache`, while pointer, array and function ctypes are weakly cached.
//! The same ownership split matters here: pointer compatibility uses identity,
//! but an otherwise-unreferenced derived ctype must still be collectable.

use pyre_interpreter::PyError;
use pyre_object::PyObjectRef;
use std::collections::HashMap;
use std::ffi::{c_char, c_double, c_float, c_int, c_long, c_longlong, c_short};
use std::sync::{Mutex, OnceLock};

use super::ctypeobj::{self, W_CType};
use super::misc;

/// `newtype.py alignment(TYPE)` — `offsetof(struct {char x; TYPE y;}, y)`,
/// which is the type's alignment.
macro_rules! prim {
    ($kind:expr, $ty:ty) => {
        (
            $kind,
            std::mem::size_of::<$ty>() as i64,
            std::mem::align_of::<$ty>() as i64,
        )
    };
    ($kind:expr, $ty:ty, $rep:literal) => {
        (
            $kind,
            (std::mem::size_of::<$ty>() * $rep) as i64,
            std::mem::align_of::<$ty>() as i64,
        )
    };
}

/// `newtype.py PRIMITIVE_TYPES` — the kind, size and alignment of every name
/// `new_primitive_type` accepts.
fn primitive_types() -> &'static HashMap<&'static str, (i64, i64, i64)> {
    static TABLE: OnceLock<HashMap<&'static str, (i64, i64, i64)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        use ctypeobj::{
            KIND_PRIM_BOOL, KIND_PRIM_CHAR, KIND_PRIM_COMPLEX, KIND_PRIM_FLOAT,
            KIND_PRIM_LONGDOUBLE, KIND_PRIM_SIGNED, KIND_PRIM_UNICHAR, KIND_PRIM_UNSIGNED,
        };
        let mut t: HashMap<&'static str, (i64, i64, i64)> = HashMap::new();
        t.insert("char", prim!(KIND_PRIM_CHAR, c_char));
        let (wchar_size, wchar_align) = misc::wchar_layout();
        t.insert("wchar_t", (KIND_PRIM_UNICHAR, wchar_size, wchar_align));
        t.insert("signed char", prim!(KIND_PRIM_SIGNED, i8));
        t.insert("short", prim!(KIND_PRIM_SIGNED, c_short));
        t.insert("int", prim!(KIND_PRIM_SIGNED, c_int));
        t.insert("long", prim!(KIND_PRIM_SIGNED, c_long));
        t.insert("long long", prim!(KIND_PRIM_SIGNED, c_longlong));
        t.insert("unsigned char", prim!(KIND_PRIM_UNSIGNED, u8));
        t.insert("unsigned short", prim!(KIND_PRIM_UNSIGNED, c_short));
        t.insert("unsigned int", prim!(KIND_PRIM_UNSIGNED, c_int));
        t.insert("unsigned long", prim!(KIND_PRIM_UNSIGNED, c_long));
        t.insert("unsigned long long", prim!(KIND_PRIM_UNSIGNED, c_longlong));
        t.insert("float", prim!(KIND_PRIM_FLOAT, c_float));
        t.insert("double", prim!(KIND_PRIM_FLOAT, c_double));
        t.insert(
            "long double",
            (
                KIND_PRIM_LONGDOUBLE,
                misc::sizeof_long_double(),
                misc::alignof_long_double(),
            ),
        );
        t.insert("_Bool", prim!(KIND_PRIM_BOOL, bool));
        for name in ["float _Complex", "_cffi_float_complex_t"] {
            t.insert(name, prim!(KIND_PRIM_COMPLEX, c_float, 2));
        }
        for name in ["double _Complex", "_cffi_double_complex_t"] {
            t.insert(name, prim!(KIND_PRIM_COMPLEX, c_double, 2));
        }
        // `eptypesize` — the fixed-width names.  It resolves each one to a
        // real C type of that width and takes that type's alignment, which is
        // not the width on every target: a 32-bit x86 `int64_t` is 8 bytes
        // aligned to 4.  Rust's integers carry the same target alignments.
        for (name, kind, layout) in [
            ("int8_t", KIND_PRIM_SIGNED, prim!(0, i8)),
            ("uint8_t", KIND_PRIM_UNSIGNED, prim!(0, u8)),
            ("int16_t", KIND_PRIM_SIGNED, prim!(0, i16)),
            ("uint16_t", KIND_PRIM_UNSIGNED, prim!(0, u16)),
            ("int32_t", KIND_PRIM_SIGNED, prim!(0, i32)),
            ("uint32_t", KIND_PRIM_UNSIGNED, prim!(0, u32)),
            ("int64_t", KIND_PRIM_SIGNED, prim!(0, i64)),
            ("uint64_t", KIND_PRIM_UNSIGNED, prim!(0, u64)),
            ("int_least8_t", KIND_PRIM_SIGNED, prim!(0, i8)),
            ("uint_least8_t", KIND_PRIM_UNSIGNED, prim!(0, u8)),
            ("int_least16_t", KIND_PRIM_SIGNED, prim!(0, i16)),
            ("uint_least16_t", KIND_PRIM_UNSIGNED, prim!(0, u16)),
            ("int_least32_t", KIND_PRIM_SIGNED, prim!(0, i32)),
            ("uint_least32_t", KIND_PRIM_UNSIGNED, prim!(0, u32)),
            ("int_least64_t", KIND_PRIM_SIGNED, prim!(0, i64)),
            ("uint_least64_t", KIND_PRIM_UNSIGNED, prim!(0, u64)),
            ("char16_t", KIND_PRIM_UNICHAR, prim!(0, u16)),
            ("char32_t", KIND_PRIM_UNICHAR, prim!(0, u32)),
        ] {
            t.insert(name, (kind, layout.1, layout.2));
        }
        // The `fast` names are the platform's own widths, which C picks for
        // speed rather than size, so the C compiler is asked for them.
        for (name, kind, bits) in [
            ("int_fast8_t", KIND_PRIM_SIGNED, 8),
            ("uint_fast8_t", KIND_PRIM_UNSIGNED, 8),
            ("int_fast16_t", KIND_PRIM_SIGNED, 16),
            ("uint_fast16_t", KIND_PRIM_UNSIGNED, 16),
            ("int_fast32_t", KIND_PRIM_SIGNED, 32),
            ("uint_fast32_t", KIND_PRIM_UNSIGNED, 32),
            ("int_fast64_t", KIND_PRIM_SIGNED, 64),
            ("uint_fast64_t", KIND_PRIM_UNSIGNED, 64),
        ] {
            let (size, align) = misc::fast_int_layout(bits);
            t.insert(name, (kind, size, align));
        }
        t.insert("intptr_t", prim!(KIND_PRIM_SIGNED, isize));
        t.insert("uintptr_t", prim!(KIND_PRIM_UNSIGNED, usize));
        t.insert("size_t", prim!(KIND_PRIM_UNSIGNED, usize));
        t.insert("ssize_t", prim!(KIND_PRIM_SIGNED, isize));
        t.insert("ptrdiff_t", prim!(KIND_PRIM_SIGNED, isize));
        t.insert("intmax_t", prim!(KIND_PRIM_SIGNED, c_longlong));
        t.insert("uintmax_t", prim!(KIND_PRIM_UNSIGNED, c_longlong));
        t
    })
}

/// `UniqueCache.primitives`.
fn primitive_cache() -> &'static Mutex<HashMap<&'static str, usize>> {
    static CACHE: OnceLock<Mutex<HashMap<&'static str, usize>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `UniqueCache.ctvoid`.
static VOID_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();
/// `UniqueCache.ctvoidp`.
static VOID_POINTER_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();
/// `UniqueCache.ctchara`.
static CHAR_ARRAY_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

/// `newtype.py _new_primitive_type`.
pub fn new_primitive_type(name: &str) -> Result<PyObjectRef, PyError> {
    let Some((&key, &(kind, size, align))) = primitive_types().get_key_value(name) else {
        return Err(PyError::key_error(format!("{name:?}")));
    };
    {
        let cache = primitive_cache()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(&existing) = cache.get(key) {
            return Ok(existing as PyObjectRef);
        }
    }
    let word = std::mem::size_of::<isize>() as i64;
    let mut flags = 0;
    match kind {
        ctypeobj::KIND_PRIM_CHAR
        | ctypeobj::KIND_PRIM_UNICHAR
        | ctypeobj::KIND_PRIM_SIGNED
        | ctypeobj::KIND_PRIM_UNSIGNED
        | ctypeobj::KIND_PRIM_BOOL => {
            flags |= ctypeobj::CTypeFlags::PRIMITIVE_INTEGER.bits();
        }
        _ => {}
    }
    match kind {
        ctypeobj::KIND_PRIM_SIGNED => {
            if size <= word {
                flags |= ctypeobj::CTypeFlags::VALUE_FITS_LONG.bits();
            }
            if size < word {
                flags |= ctypeobj::CTypeFlags::VALUE_SMALLER_THAN_LONG.bits();
            }
        }
        ctypeobj::KIND_PRIM_UNSIGNED | ctypeobj::KIND_PRIM_BOOL => {
            if size < word {
                flags |= ctypeobj::CTypeFlags::VALUE_FITS_LONG.bits();
            }
            if size <= word {
                flags |= ctypeobj::CTypeFlags::VALUE_FITS_ULONG.bits();
            }
        }
        ctypeobj::KIND_PRIM_UNICHAR
            // `char16_t` and `char32_t` are always unsigned; only `wchar_t`
            // follows the platform's signedness.
            if key == "wchar_t" && wchar_is_signed() => {
                flags |= ctypeobj::CTypeFlags::SIGNED_WCHAR.bits();
            }
        _ => {}
    }
    let obj = ctypeobj::new_ctype(
        kind,
        size,
        key,
        key.len() as i64,
        align,
        pyre_object::PY_NULL,
        -1,
        flags,
    );
    ctypeobj::root_forever(obj);
    let mut cache = primitive_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Ok(*cache.entry(key).or_insert(obj as usize) as PyObjectRef)
}

/// `rfficache.signof_c_type('wchar_t')`.
fn wchar_is_signed() -> bool {
    // The C `wchar_t` is signed on the platforms whose `wchar_t` is `int`
    // and unsigned where it is `unsigned short` or `unsigned int`.
    cfg!(not(windows))
}

/// `newtype.py new_void_type`.
pub fn new_void_type() -> PyObjectRef {
    VOID_TYPE.get_or_init(|| {
        ctypeobj::new_ctype(
            ctypeobj::KIND_VOID,
            -1,
            "void",
            "void".len() as i64,
            -1,
            pyre_object::PY_NULL,
            -1,
            0,
        )
    })
}

/// `newtype.py _new_voidp_type`.
pub fn new_voidp_type() -> Result<PyObjectRef, PyError> {
    if let Some(cached) = VOID_POINTER_TYPE.get() {
        return Ok(cached);
    }
    let obj = new_pointer_type(new_void_type())?;
    Ok(VOID_POINTER_TYPE.get_or_init(|| obj))
}

/// `newtype.py _new_chara_type`.
pub fn new_chara_type() -> Result<PyObjectRef, PyError> {
    if let Some(cached) = CHAR_ARRAY_TYPE.get() {
        return Ok(cached);
    }
    let w_char = new_primitive_type("char")?;
    let w_charp = new_pointer_type(w_char)?;
    let obj = new_array_type(w_charp, -1)?;
    Ok(CHAR_ARRAY_TYPE.get_or_init(|| obj))
}

/// `newtype.py _new_pointer_type` — `W_CType._pointer_type`'s memo, which
/// `convert_from_object` relies on to decide pointer compatibility.
pub fn new_pointer_type(w_ctitem: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let ctitem = ctypeobj::ctype_arg(w_ctitem)?;
    let cached =
        unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref(ctitem.pointer_type) };
    if !cached.is_null() {
        return Ok(cached);
    }
    // `W_CTypePointer.__init__`: an array's pointer needs the parenthesised
    // spelling so `int(*)[5]` does not read as `int *[5]`.
    let extra = if ctitem.kind == ctypeobj::KIND_ARRAY {
        "(*)"
    } else {
        " *"
    };
    let (name, name_position) = ctitem.insert_name(extra, 2);
    let mut flags = ctypeobj::CTypeFlags::NONFUNC_POINTER_OR_ARRAY.bits();
    if ctitem.kind == ctypeobj::KIND_VOID {
        flags |= ctypeobj::CTypeFlags::VOID_PTR.bits() | ctypeobj::CTypeFlags::VOIDCHAR_PTR.bits();
    }
    if ctitem.kind == ctypeobj::KIND_PRIM_CHAR {
        flags |= ctypeobj::CTypeFlags::VOIDCHAR_PTR.bits();
    }
    if ctitem.size == 1 {
        flags |= ctypeobj::CTypeFlags::ONEBYTE_PTR.bits();
    }
    if ctitem.name() == "struct _IO_FILE" || ctitem.name() == "FILE" {
        flags |= ctypeobj::CTypeFlags::FILE_PTR.bits();
    }
    flags |= accept_str_flag(ctitem);
    let obj = ctypeobj::new_ctype(
        ctypeobj::KIND_POINTER,
        std::mem::size_of::<*const u8>() as i64,
        &name,
        name_position,
        -1,
        w_ctitem,
        -1,
        flags,
    );
    // Re-read the item: building the pointer allocated, and the memo must
    // land on the object the caller named.
    let roots = pyre_object::gc_roots::push_roots();
    let item_slot = roots.base();
    let _ = roots.pin_root(w_ctitem);
    let obj_slot = item_slot + 1;
    let _ = roots.pin_root(obj);
    let weak = pyre_object::weakref::w_gc_weakref_box_new_or_strong(roots.get(obj_slot));
    let weak_slot = obj_slot + 1;
    let _ = roots.pin_root(weak);
    let ctitem = ctypeobj::ctype_arg(roots.get(item_slot))?;
    ctitem.pointer_type = roots.get(weak_slot);
    pyre_object::gc_hook::try_gc_write_barrier_managed(
        roots.get(item_slot) as pyre_object::gc_hook::GCREF
    );
    Ok(roots.get(obj_slot))
}

/// `W_CTypePtrOrArray.accept_str` — whether a `bytes` initialises it.
fn accept_str_flag(ctitem: &W_CType) -> i64 {
    let accepts = ctitem.kind == ctypeobj::KIND_VOID
        || ctitem.kind == ctypeobj::KIND_PRIM_CHAR
        || (ctitem.has(ctypeobj::CTypeFlags::PRIMITIVE_INTEGER) && ctitem.size == 1);
    if accepts {
        ctypeobj::CTypeFlags::ACCEPT_STR.bits()
    } else {
        0
    }
}

/// `newtype.py _new_array_type`.
pub fn new_array_type(w_ctptr: PyObjectRef, length: i64) -> Result<PyObjectRef, PyError> {
    let ctptr = ctypeobj::ctype_arg(w_ctptr)?;
    if ctptr.kind != ctypeobj::KIND_POINTER {
        return Err(PyError::type_error("first arg must be a pointer ctype"));
    }
    if !ctptr.array_types.is_null()
        && let Some(weak) =
            unsafe { pyre_object::dictmultiobject::w_dict_getitem(ctptr.array_types, length) }
    {
        let cached = unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref(weak) };
        if !cached.is_null() {
            return Ok(cached);
        }
    }
    let ctitem = super::ctypeptr::item_of(ctptr)?;
    if ctitem.size < 0 {
        return Err(PyError::value_error(format!(
            "array item of unknown size: '{}'",
            ctitem.name()
        )));
    }
    let (arraysize, extra) = if length < 0 {
        (-1, "[]".to_string())
    } else {
        let arraysize = length
            .checked_mul(ctitem.size)
            .filter(|&n| n >= 0)
            .ok_or_else(|| PyError::overflow_error("array size would overflow a ssize_t"))?;
        (arraysize, format!("[{length}]"))
    };
    let (name, name_position) = ctitem.insert_name(&extra, 0);
    let flags = ctypeobj::CTypeFlags::NONFUNC_POINTER_OR_ARRAY.bits() | accept_str_flag(ctitem);
    let obj = ctypeobj::new_ctype(
        ctypeobj::KIND_ARRAY,
        arraysize,
        &name,
        name_position,
        -1,
        ctptr.ctitem,
        length,
        flags,
    );
    // `W_CTypeArray.ctptr`, re-read for the same reason as above.
    let array = ctypeobj::ctype_arg(obj)?;
    array.ctptr = w_ctptr;
    let roots = pyre_object::gc_roots::push_roots();
    let pointer_slot = roots.base();
    let _ = roots.pin_root(w_ctptr);
    let array_slot = pointer_slot + 1;
    let _ = roots.pin_root(obj);
    let weak = pyre_object::weakref::w_gc_weakref_box_new_or_strong(roots.get(array_slot));
    let weak_slot = array_slot + 1;
    let _ = roots.pin_root(weak);
    let ctptr = ctypeobj::ctype_arg(roots.get(pointer_slot))?;
    if ctptr.array_types.is_null() {
        ctptr.array_types = pyre_object::dictmultiobject::w_dict_new();
        pyre_object::gc_hook::try_gc_write_barrier_managed(
            roots.get(pointer_slot) as pyre_object::gc_hook::GCREF
        );
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem(
            ctptr.array_types,
            length,
            roots.get(weak_slot),
        )
    };
    Ok(roots.get(array_slot))
}

/// `W_CTypePointer.cache_array_type` — the `T[]` a slice of a `T *` reads
/// through.
pub fn cached_unbounded_array_type(w_ctptr: PyObjectRef) -> Result<PyObjectRef, PyError> {
    new_array_type(w_ctptr, -1)
}

/// `func.py new_array_type`'s length argument: `None` is "no length".
pub fn array_length_arg(w_length: PyObjectRef) -> Result<i64, PyError> {
    if unsafe { pyre_object::pyobject::is_none(w_length) } {
        return Ok(-1);
    }
    // `space.getindex_w(w_length, space.w_OverflowError)` — a length wider
    // than a `ssize_t` is an OverflowError rather than a silent clamp.
    let length = pyre_interpreter::baseobjspace::index_int_w_preserve_negative(w_length)?;
    if length < 0 {
        return Err(PyError::value_error("negative array length"));
    }
    Ok(length)
}

// ── structs and unions ──────────────────────────────────────────────────

bitflags::bitflags! {
    /// `newtype.py SF_*` struct-layout flags.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct StructFlags: i64 {
        const MSVC_BITFIELDS = 0x01;
        const GCC_ARM_BITFIELDS = 0x02;
        const GCC_BIG_ENDIAN = 0x04;
        const PACKED = 0x08;
        const GCC_X86_BITFIELDS = 0x10;
        const GCC_LITTLE_ENDIAN = 0x40;
        const STD_FIELD_POS = 0x80;
    }
}

/// `newtype.py SF_DEFAULT_PACKING`.
const SF_DEFAULT_PACKING: i64 = if cfg!(windows) { 8 } else { 0x4000_0000 };

/// `newtype.py DEFAULT_SFLAGS_PLATFORM`.
const DEFAULT_SFLAGS_PLATFORM: i64 = if cfg!(windows) {
    StructFlags::MSVC_BITFIELDS.bits()
} else if cfg!(any(target_arch = "arm", target_arch = "aarch64")) {
    StructFlags::GCC_ARM_BITFIELDS.bits()
} else {
    StructFlags::GCC_X86_BITFIELDS.bits()
};

/// `newtype.py DEFAULT_SFLAGS_ENDIAN`.
const DEFAULT_SFLAGS_ENDIAN: i64 = if cfg!(target_endian = "big") {
    StructFlags::GCC_BIG_ENDIAN.bits()
} else {
    StructFlags::GCC_LITTLE_ENDIAN.bits()
};

/// `newtype.py complete_sflags`.
fn complete_sflags(mut sflags: i64) -> i64 {
    if sflags
        & (StructFlags::MSVC_BITFIELDS
            | StructFlags::GCC_ARM_BITFIELDS
            | StructFlags::GCC_X86_BITFIELDS)
            .bits()
        == 0
    {
        sflags |= DEFAULT_SFLAGS_PLATFORM;
    }
    if sflags & (StructFlags::GCC_BIG_ENDIAN | StructFlags::GCC_LITTLE_ENDIAN).bits() == 0 {
        sflags |= DEFAULT_SFLAGS_ENDIAN;
    }
    sflags
}

/// `ffi_obj.py get_ffi_error` — the class `FFI.error` names.  `ffi_obj` is
/// not ported yet, so the class lives here until it has an owner to publish
/// it from; it is the object identity that matters, and there is one.
pub fn ffi_error() -> PyObjectRef {
    static FFI_ERROR: pyre_object::gc_roots::RootedOnceRef =
        pyre_object::gc_roots::RootedOnceRef::new();
    FFI_ERROR.get_or_init(|| {
        let w_exception = pyre_interpreter::builtins::lookup_exc_class("Exception")
            .expect("Exception must be installed before _cffi_backend init");
        pyre_interpreter::builtins::make_exc_type(
            "ffi.error",
            pyre_interpreter::builtins::exc_exception_new,
            w_exception,
        )
    })
}

/// `newtype.py new_struct_type`.
pub fn new_struct_type(name: &str) -> PyObjectRef {
    new_struct_or_union(ctypeobj::KIND_STRUCT, name)
}

/// `newtype.py new_union_type`.
pub fn new_union_type(name: &str) -> PyObjectRef {
    new_struct_or_union(ctypeobj::KIND_UNION, name)
}

/// `W_CTypeStructOrUnion.__init__` — born opaque, with the name as its own
/// insertion point.
fn new_struct_or_union(kind: i64, name: &str) -> PyObjectRef {
    ctypeobj::new_ctype(
        kind,
        -1,
        name,
        name.len() as i64,
        -1,
        pyre_object::PY_NULL,
        -1,
        0,
    )
}

/// `newtype.py detect_custom_layout`.
pub fn detect_custom_layout(
    ct: &mut W_CType,
    sflags: i64,
    cdef_value: i64,
    compiler_value: i64,
    msg: &str,
) -> Result<(), PyError> {
    if compiler_value == cdef_value {
        return Ok(());
    }
    if sflags & StructFlags::STD_FIELD_POS.bits() != 0 {
        let name = ct.name();
        // `newtype.py detect_custom_layout` raises through `oefmt(w_FFIError, ...)`:
        // the class and the text exist before OperationError.
        let msg_text = format!(
            "{name}: {msg} (cdef says {cdef_value}, but C compiler says {compiler_value}). fix it or use \"...;\" as the last field in the cdef for {name} to make it flexible"
        );
        // `newtype.py detect_custom_layout` raises `oefmt(w_FFIError, ...)`.
        let cls = ffi_error();
        let _roots = pyre_object::gc_roots::push_roots();
        let cls_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(cls);
        let w_value = pyre_object::w_str_new_managed(&msg_text);
        return Err(PyError::from_type_and_value(
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
            w_value,
        ));
    }
    ct.flags |= ctypeobj::CTypeFlags::CUSTOM_FIELD_POS.bits();
    Ok(())
}

/// `newtype.py roundup_bytes`.
fn roundup_bytes(bytes: i64, bit: i64) -> i64 {
    bytes + i64::from(bit > 0)
}

/// One entry of the `fields` argument: `(name, ctype[, bitsize[, offset]])`.
struct FieldDescr {
    fname: String,
    w_ftype: PyObjectRef,
    fbitsize: i64,
    foffset: i64,
}

/// `newtype.py complete_struct_or_union`.
pub fn complete_struct_or_union(
    mut w_ctype: PyObjectRef,
    w_fields: PyObjectRef,
    totalsize: i64,
    totalalignment: i64,
    sflags: i64,
    pack: i64,
) -> Result<(), PyError> {
    let sflags = complete_sflags(sflags);
    let (sflags, pack) = if sflags & StructFlags::PACKED.bits() != 0 {
        (sflags, 1)
    } else if pack <= 0 {
        (sflags, SF_DEFAULT_PACKING)
    } else {
        (sflags | StructFlags::PACKED.bits(), pack)
    };
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    if !ct.is_struct_or_union() || ct.size >= 0 {
        return Err(PyError::type_error(
            "first arg must be a non-initialized struct or union ctype",
        ));
    }
    let is_union = ct.kind == ctypeobj::KIND_UNION;

    // The field descriptors are read out first: each one is a tuple whose
    // unpacking allocates, and nothing below may hold a stale reference.
    let descrs = pyre_object::with_roots!(w_ctype => read_field_descrs(ct, w_fields))?;

    let roots = pyre_object::gc_roots::push_roots();
    // The ctype stays pinned across every allocation below, including the two
    // containers.  Their slots are the indices those pins return.
    let ctype_slot = roots.pin_roots(&[w_ctype]);
    let list_slot = roots.pin_roots(&[pyre_object::w_list_new(Vec::new())]);
    let dict_slot = roots.pin_roots(&[pyre_object::dictmultiobject::w_dict_new()]);
    let append = |w_field: PyObjectRef| unsafe {
        pyre_object::listobject::w_list_append(roots.get(list_slot), w_field);
    };
    let record = |name: &str, w_field: PyObjectRef| unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(roots.get(dict_slot), name, w_field);
    };

    let mut alignment = 1i64;
    // The real offset is `byteoffset + bitoffset * 8`, counted in bits.
    let mut byteoffset = 0i64;
    let mut bitoffset = 0i64;
    let mut byteoffsetmax = 0i64;
    let mut prev_bitfield_size = 0i64;
    let mut prev_bitfield_free = 0i64;
    ct.flags &= !ctypeobj::CTypeFlags::CUSTOM_FIELD_POS.bits();
    let mut with_var_array = false;
    let mut with_packed_change = false;

    for (i, descr) in descrs.iter().enumerate() {
        let ftype = ctypeobj::ctype_arg(descr.w_ftype)?;
        let fname = descr.fname.as_str();
        let fbitsize = descr.fbitsize;
        let foffset = descr.foffset;
        if unsafe {
            pyre_object::dictmultiobject::w_dict_getitem_str(roots.get(dict_slot), fname).is_some()
        } {
            return Err(PyError::key_error(format!(
                "duplicate field name '{fname}'"
            )));
        }
        if ftype.size < 0 {
            if ftype.kind == ctypeobj::KIND_ARRAY
                && fbitsize < 0
                && (i == descrs.len() - 1 || foffset != -1)
            {
                with_var_array = true;
            } else {
                return Err(PyError::type_error(format!(
                    "field '{}.{fname}' has ctype '{}' of unknown size",
                    ct.name(),
                    ftype.name()
                )));
            }
        } else if ftype.is_struct_or_union() {
            ftype.force_lazy_struct()?;
            // A var-sized array anywhere inside a field propagates outward,
            // so a struct holding such a struct is var-sized too.
            if ftype.has(ctypeobj::CTypeFlags::WITH_VAR_ARRAY) {
                with_var_array = true;
            }
        }

        if is_union {
            byteoffset = 0;
            bitoffset = 0;
        }

        // Skip the alignment bump for an anonymous bitfield, or under SF_PACKED.
        let falignorg = ftype.alignof()?;
        let falign = pack.min(falignorg);
        let mut do_align = true;
        if sflags & StructFlags::GCC_ARM_BITFIELDS.bits() == 0 && fbitsize >= 0 {
            do_align = if sflags & StructFlags::MSVC_BITFIELDS.bits() == 0 {
                // Anonymous bitfields of any size do not cause alignment.
                !fname.is_empty()
            } else {
                // Zero-sized bitfields do not cause alignment.
                fbitsize > 0
            };
        }
        if alignment < falign && do_align {
            alignment = falign;
        }

        let fflags = if is_union && i > 0 {
            super::ctypestruct::BF_IGNORE_IN_CTOR
        } else {
            0
        };

        if fbitsize < 0 {
            // Not a bitfield: the common case.
            let bs_flag = if ftype.kind == ctypeobj::KIND_ARRAY && ftype.length <= 0 {
                super::ctypestruct::BS_EMPTY_ARRAY
            } else {
                super::ctypestruct::BS_REGULAR
            };
            // Pad to the next byte, then to `falign` or `falignorg` bytes.
            byteoffset = roundup_bytes(byteoffset, bitoffset);
            bitoffset = 0;
            let byteoffsetorg = (byteoffset + falignorg - 1) & !(falignorg - 1);
            byteoffset = (byteoffset + falign - 1) & !(falign - 1);
            if byteoffsetorg != byteoffset {
                with_packed_change = true;
            }
            if foffset >= 0 {
                // A forced field position: the offset just computed only
                // decides whether the layout is a custom one.
                detect_custom_layout(
                    ct,
                    sflags,
                    byteoffset,
                    foffset,
                    &format!("wrong offset for field '{fname}'"),
                )?;
                byteoffset = foffset;
            }
            if fname.is_empty() && ftype.is_struct_or_union() {
                // A nested anonymous struct or union: its fields are spliced
                // in at this offset under their own names.
                for w_srcfld in super::ctypestruct::fields_list_of(ftype)? {
                    let srcfld = super::ctypestruct::W_CField::from_obj(w_srcfld)
                        .ok_or_else(|| PyError::system_error("field list holds a non-field"))?;
                    let w_fld = super::ctypestruct::make_shifted(srcfld, byteoffset, fflags);
                    let fld_slot = roots.pin_roots(&[w_fld]);
                    append(roots.get(fld_slot));
                    if let Some(name) = super::ctypestruct::name_of_field(ftype, w_srcfld)? {
                        record(&name, roots.get(fld_slot));
                    }
                }
                // Such a structure can never be passed by value.
                ct.flags |= ctypeobj::CTypeFlags::CUSTOM_FIELD_POS.bits();
            } else {
                let w_fld =
                    super::ctypestruct::new_cfield(descr.w_ftype, byteoffset, bs_flag, -1, fflags);
                let fld_slot = roots.pin_roots(&[w_fld]);
                append(roots.get(fld_slot));
                record(fname, roots.get(fld_slot));
            }
            if ftype.size >= 0 {
                byteoffset += ftype.size;
            }
            prev_bitfield_size = 0;
        } else {
            // A bitfield.
            if foffset >= 0 {
                return Err(PyError::type_error(format!(
                    "field '{}.{fname}' is a bitfield, but a fixed offset is specified",
                    ct.name()
                )));
            }
            if !(ftype.kind == ctypeobj::KIND_PRIM_SIGNED
                || ftype.kind == ctypeobj::KIND_PRIM_UNSIGNED
                || ftype.is_char_or_unichar())
            {
                return Err(PyError::type_error(format!(
                    "field '{}.{fname}' declared as '{}' cannot be a bit field",
                    ct.name(),
                    ftype.name()
                )));
            }
            if fbitsize > 8 * ftype.size {
                return Err(PyError::type_error(format!(
                    "bit field '{}.{fname}' is declared '{}:{fbitsize}', which exceeds the width of the type",
                    ct.name(),
                    ftype.name()
                )));
            }
            // Where the theoretical field covering a whole `ftype` starts;
            // the real bitfield lives inside it.
            let mut field_offset_bytes = byteoffset & !(falign - 1);
            if fbitsize == 0 {
                if !fname.is_empty() {
                    return Err(PyError::type_error(format!(
                        "field '{}.{fname}' is declared with :0",
                        ct.name()
                    )));
                }
                if sflags & StructFlags::MSVC_BITFIELDS.bits() == 0 {
                    // GCC's notion of "ftype :0;" pads to a value aligned for
                    // `ftype`.
                    if roundup_bytes(byteoffset, bitoffset) > field_offset_bytes {
                        field_offset_bytes += falign;
                    }
                    byteoffset = field_offset_bytes;
                    bitoffset = 0;
                }
                // MSVC's notion is mostly ignored: it only separates other
                // bitfields, forcing them into separate words.
                prev_bitfield_size = 0;
            } else {
                let mut bitshift;
                if sflags & StructFlags::MSVC_BITFIELDS.bits() == 0 {
                    // GCC's algorithm: the field can start where we are if it
                    // would fit entirely into an aligned `ftype` field.
                    let bits_already_occupied = (byteoffset - field_offset_bytes) * 8 + bitoffset;
                    if bits_already_occupied + fbitsize > 8 * ftype.size {
                        if sflags & StructFlags::PACKED.bits() != 0
                            && bits_already_occupied & 7 != 0
                        {
                            return Err(PyError::not_implemented(format!(
                                "with 'packed', gcc would compile field '{}.{fname}' to reuse some bits in the previous field",
                                ct.name()
                            )));
                        }
                        field_offset_bytes += falign;
                        byteoffset = field_offset_bytes;
                        bitoffset = 0;
                        bitshift = 0;
                    } else {
                        bitshift = bits_already_occupied;
                    }
                    bitoffset += fbitsize;
                    byteoffset += bitoffset >> 3;
                    bitoffset &= 7;
                } else {
                    // MSVC's algorithm: a bitfield takes the full width of its
                    // declared type, and shares bits only with a previous
                    // bitfield of the same size.
                    if prev_bitfield_size == ftype.size && prev_bitfield_free >= fbitsize {
                        bitshift = 8 * prev_bitfield_size - prev_bitfield_free;
                    } else {
                        byteoffset = roundup_bytes(byteoffset, bitoffset);
                        bitoffset = 0;
                        byteoffset = (byteoffset + falign - 1) & !(falign - 1);
                        byteoffset += ftype.size;
                        bitshift = 0;
                        prev_bitfield_size = ftype.size;
                        prev_bitfield_free = 8 * prev_bitfield_size;
                    }
                    prev_bitfield_free -= fbitsize;
                    field_offset_bytes = byteoffset - ftype.size;
                }
                if sflags & StructFlags::GCC_BIG_ENDIAN.bits() != 0 {
                    bitshift = 8 * ftype.size - fbitsize - bitshift;
                }
                if !fname.is_empty() {
                    let w_fld = super::ctypestruct::new_cfield(
                        descr.w_ftype,
                        field_offset_bytes,
                        bitshift,
                        fbitsize,
                        fflags,
                    );
                    let fld_slot = roots.pin_roots(&[w_fld]);
                    append(roots.get(fld_slot));
                    record(fname, roots.get(fld_slot));
                }
            }
        }
        byteoffsetmax = byteoffsetmax.max(roundup_bytes(byteoffset, bitoffset));
    }

    // As in C, a structure whose size would be zero is one byte instead; a
    // manually-specified total size of zero is still honoured.
    // The rounding is machine-word arithmetic: `newtype.py` computes it in
    // RPython `Signed`, which wraps.  A `char[sys.maxsize]` field rounds
    // `MAX + 1 - 1` at alignment one and has to come back as `MAX`.
    let alignedsize =
        (byteoffsetmax.wrapping_add(alignment).wrapping_sub(1) & !(alignment - 1)).max(1);
    let totalsize = if totalsize < 0 {
        alignedsize
    } else {
        detect_custom_layout(ct, sflags, alignedsize, totalsize, "wrong total size")?;
        if totalsize < byteoffsetmax {
            return Err(PyError::type_error(format!(
                "{} cannot be of size {totalsize}: there are fields at least up to {byteoffsetmax}",
                ct.name()
            )));
        }
        totalsize
    };
    let totalalignment = if totalalignment < 0 {
        alignment
    } else {
        detect_custom_layout(
            ct,
            sflags,
            alignment,
            totalalignment,
            "wrong total alignment",
        )?;
        totalalignment
    };

    ctypeobj::set_size(ct, totalsize);
    ct.align = totalalignment;
    ct.fields_list = roots.get(list_slot);
    ct.fields_dict = roots.get(dict_slot);
    if with_var_array {
        ct.flags |= ctypeobj::CTypeFlags::WITH_VAR_ARRAY.bits();
    }
    if with_packed_change {
        ct.flags |= ctypeobj::CTypeFlags::WITH_PACKED_CHANGE.bits();
    }
    // The field containers were just stored into the ctype.
    pyre_object::gc_hook::try_gc_write_barrier_managed(
        roots.get(ctype_slot) as pyre_object::gc_hook::GCREF
    );
    Ok(())
}

/// The `(name, ctype[, bitsize[, offset]])` tuples `complete_struct_or_union`
/// is given, read before anything allocates on their behalf.
fn read_field_descrs(ct: &W_CType, w_fields: PyObjectRef) -> Result<Vec<FieldDescr>, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let fields_slot = roots.base();
    let _ = roots.pin_root(w_fields);
    let fields_w = pyre_interpreter::baseobjspace::fixedview(roots.get(fields_slot), -1)?;
    let items_slot = pyre_object::gc_roots::shadow_stack_len();
    for &w_field in &fields_w {
        let _ = roots.pin_root(w_field);
    }
    let mut out = Vec::with_capacity(fields_w.len());
    for i in 0..fields_w.len() {
        let field_w = pyre_interpreter::baseobjspace::fixedview(roots.get(items_slot + i), -1)?;
        if !(2..=4).contains(&field_w.len()) {
            return Err(PyError::type_error("bad field descr"));
        }
        let part_slot = pyre_object::gc_roots::shadow_stack_len();
        for &w_part in &field_w {
            let _ = roots.pin_root(w_part);
        }
        let fname = pyre_interpreter::baseobjspace::text_w(roots.get(part_slot))?.to_string();
        let w_ftype = roots.get(part_slot + 1);
        if ctypeobj::ctype_at(w_ftype).is_none() {
            return Err(PyError::type_error(format!(
                "field '{}.{fname}' must be a ctype",
                ct.name()
            )));
        }
        let fbitsize = if field_w.len() > 2 {
            pyre_interpreter::baseobjspace::int_w(roots.get(part_slot + 2))?
        } else {
            -1
        };
        let foffset = if field_w.len() > 3 {
            pyre_interpreter::baseobjspace::int_w(roots.get(part_slot + 3))?
        } else {
            -1
        };
        out.push(FieldDescr {
            fname,
            w_ftype,
            fbitsize,
            foffset,
        });
    }
    Ok(out)
}

// ── enums ───────────────────────────────────────────────────────────────

/// `newtype.py new_enum_type`.
pub fn new_enum_type(
    name: &str,
    w_enumerators: PyObjectRef,
    w_enumvalues: PyObjectRef,
    w_basectype: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let base_slot = roots.pin_roots(&[w_basectype, w_enumvalues]);
    let enumvalues_slot = base_slot + 1;
    let enumerators_w = pyre_interpreter::baseobjspace::fixedview(w_enumerators, -1)?;
    let names_slot = pyre_object::gc_roots::shadow_stack_len();
    for &w in &enumerators_w {
        let _ = roots.pin_root(w);
    }
    let enumvalues_w = pyre_interpreter::baseobjspace::fixedview(roots.get(enumvalues_slot), -1)?;
    let values_slot = pyre_object::gc_roots::shadow_stack_len();
    for &w in &enumvalues_w {
        let _ = roots.pin_root(w);
    }
    if enumerators_w.len() != enumvalues_w.len() {
        return Err(PyError::value_error("tuple args must have the same size"));
    }
    let basectype = ctypeobj::ctype_arg(roots.get(base_slot))?;
    if basectype.kind != ctypeobj::KIND_PRIM_SIGNED
        && basectype.kind != ctypeobj::KIND_PRIM_UNSIGNED
    {
        return Err(PyError::type_error(
            "expected a primitive signed or unsigned base type",
        ));
    }
    // Writing each value through the base type is what detects an
    // out-of-range or badly typed one.
    let probe = super::cdataobj::raw_alloc(basectype.size, true)?;
    for i in 0..enumvalues_w.len() {
        let written = unsafe {
            ctypeobj::convert_from_object(
                ctypeobj::ctype_arg(roots.get(base_slot))?,
                probe as usize,
                roots.get(values_slot + i),
            )
        };
        if let Err(e) = written {
            super::cdataobj::raw_free(probe);
            return Err(e);
        }
    }
    super::cdataobj::raw_free(probe);

    let basectype = ctypeobj::ctype_arg(roots.get(base_slot))?;
    let w_ctype = ctypeobj::new_ctype(
        basectype.kind,
        basectype.size,
        name,
        name.len() as i64,
        basectype.align,
        pyre_object::PY_NULL,
        -1,
        basectype.flags | ctypeobj::CTypeFlags::ENUM.bits(),
    );
    let ctype_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_ctype);
    let e2v_slot = ctype_slot + 1;
    let _ = roots.pin_root(pyre_object::dictmultiobject::w_dict_new());
    let v2e_slot = e2v_slot + 1;
    let _ = roots.pin_root(pyre_object::dictmultiobject::w_dict_new());
    // Walking backwards is what lets an earlier duplicate win, as upstream's
    // reversed fill does.
    for i in (0..enumerators_w.len()).rev() {
        let w_name = roots.get(names_slot + i);
        let w_value = roots.get(values_slot + i);
        pyre_interpreter::baseobjspace::setitem(roots.get(e2v_slot), w_name, w_value)?;
        pyre_interpreter::baseobjspace::setitem(roots.get(v2e_slot), w_value, w_name)?;
    }
    let ct = ctypeobj::ctype_arg(roots.get(ctype_slot))?;
    ct.enumerators2values = roots.get(e2v_slot);
    ct.enumvalues2erators = roots.get(v2e_slot);
    pyre_object::gc_hook::try_gc_write_barrier_managed(
        roots.get(ctype_slot) as pyre_object::gc_hook::GCREF
    );
    Ok(roots.get(ctype_slot))
}

// ── function types ──────────────────────────────────────────────────────

/// `UniqueCache.functions` — a list of weak dictionaries keyed by the
/// identity hash.  Each dictionary carries at most one value for a hash;
/// collisions spill into the next dictionary and are resolved by comparing
/// the live function ctype's fields.
///
/// The native value is the stable address of a registered root slot, not the
/// weakref box's address: the collector moves the box and rewrites that slot.
type FunctionCache = Vec<HashMap<isize, usize>>;

fn function_cache() -> &'static Mutex<FunctionCache> {
    static CACHE: OnceLock<Mutex<FunctionCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

/// `newtype.py _func_key_hash` with `compute_identity_hash` represented by
/// stable ctype addresses.
fn function_key_hash(
    fargs: &[PyObjectRef],
    w_fresult: PyObjectRef,
    ellipsis: bool,
    abi: i64,
) -> isize {
    let mut hash = w_fresult as isize;
    for &w_arg in fargs {
        hash = hash.wrapping_mul(1_000_003) ^ w_arg as isize;
    }
    hash ^ isize::from(ellipsis).wrapping_add((2isize).wrapping_mul(abi as isize))
}

fn rooted_weakref_at(root_slot: usize) -> PyObjectRef {
    unsafe { *(root_slot as *const usize) as PyObjectRef }
}

fn function_type_matches(
    w_ctype: PyObjectRef,
    fargs: &[PyObjectRef],
    w_fresult: PyObjectRef,
    ellipsis: bool,
    abi: i64,
) -> bool {
    let Some(ctype) = ctypeobj::ctype_at(w_ctype) else {
        return false;
    };
    if ctype.kind != ctypeobj::KIND_FUNC
        || ctype.ctitem != w_fresult
        || ctype.has(ctypeobj::CTypeFlags::ELLIPSIS) != ellipsis
        || ctype.abi != abi
        || unsafe { pyre_object::w_tuple_len(ctype.fargs) } != fargs.len()
    {
        return false;
    }
    fargs.iter().enumerate().all(|(index, &expected)| {
        (unsafe { pyre_object::w_tuple_getitem(ctype.fargs, index as i64) }) == Some(expected)
    })
}

/// `newtype.py new_function_type`.
pub fn new_function_type(
    w_fargs: PyObjectRef,
    w_fresult: PyObjectRef,
    ellipsis: bool,
    abi: i64,
) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let result_slot = roots.base();
    let _ = roots.pin_root(w_fresult);
    let args_w = pyre_interpreter::baseobjspace::fixedview(w_fargs, -1)?;
    let mut fargs = Vec::with_capacity(args_w.len());
    for w_farg in args_w {
        let Some(farg) = ctypeobj::ctype_at(w_farg) else {
            return Err(PyError::type_error(
                "first arg must be a tuple of ctype objects",
            ));
        };
        // An array argument decays to the pointer it was built from.
        fargs.push(if farg.kind == ctypeobj::KIND_ARRAY {
            farg.ctptr
        } else {
            w_farg
        });
    }
    build_function_type(&fargs, roots.get(result_slot), ellipsis, abi)
}

/// `newtype.py _new_function_type` and `_build_function_type`.
pub fn build_function_type(
    fargs: &[PyObjectRef],
    w_fresult: PyObjectRef,
    ellipsis: bool,
    abi: i64,
) -> Result<PyObjectRef, PyError> {
    let func_hash = function_key_hash(fargs, w_fresult, ellipsis, abi);
    {
        let cache = function_cache()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for weakdict in cache.iter() {
            if let Some(&root_slot) = weakdict.get(&func_hash) {
                let existing = unsafe {
                    pyre_object::weakref::w_gc_weakref_box_or_strong_deref(rooted_weakref_at(
                        root_slot,
                    ))
                };
                if function_type_matches(existing, fargs, w_fresult, ellipsis, abi) {
                    return Ok(existing);
                }
            }
        }
    }
    let fresult = ctypeobj::ctype_arg(w_fresult)?;
    if (fresult.size < 0 && fresult.kind != ctypeobj::KIND_VOID)
        || fresult.kind == ctypeobj::KIND_ARRAY
    {
        return Err(PyError::type_error(
            if fresult.is_struct_or_union() && fresult.size < 0 {
                format!("result type '{}' is opaque", fresult.name())
            } else {
                format!("invalid result type: '{}'", fresult.name())
            },
        ));
    }
    // `W_CTypeFunc.__init__`.
    let (extra, xpos) = super::ctypefunc::compute_extra_text(fargs, ellipsis, abi);
    let (name, name_position) = fresult.insert_name(&extra, xpos);
    let w_ctype = ctypeobj::new_ctype(
        ctypeobj::KIND_FUNC,
        std::mem::size_of::<*const u8>() as i64,
        &name,
        name_position,
        -1,
        w_fresult,
        -1,
        if ellipsis {
            ctypeobj::CTypeFlags::ELLIPSIS.bits()
        } else {
            0
        },
    );
    // The argument slice is published with the new ctype and the result type
    // before the tuple allocation.  Later reads rebuild it from those slots.
    let n_fargs = fargs.len();
    let roots = pyre_object::gc_roots::push_roots();
    let ctype_slot = pyre_object::gc_roots::publish_roots(&[w_ctype, w_fresult]);
    let fresult_slot = ctype_slot + 1;
    let fargs_base = pyre_object::gc_roots::publish_roots(fargs);
    pyre_object::gc_roots::normalize_roots(ctype_slot, 2 + n_fargs);
    let fargs_now = || -> Vec<PyObjectRef> {
        let mut reloaded = vec![pyre_object::PY_NULL; n_fargs];
        pyre_object::gc_roots::shadow_stack_copy_range(fargs_base, &mut reloaded);
        reloaded
    };
    let fargs_tuple_slot = roots.pin_roots(&[pyre_object::w_tuple_new(fargs_now())]);
    let ct = ctypeobj::ctype_arg(roots.get(ctype_slot))?;
    ct.fargs = roots.get(fargs_tuple_slot);
    ct.abi = abi;
    pyre_object::gc_hook::try_gc_write_barrier_managed(
        roots.get(ctype_slot) as pyre_object::gc_hook::GCREF
    );
    if !ellipsis {
        // A function taking '...' is stored without a cif at all: its cif is
        // computed per call from the types actually passed.  For every other
        // one it is computed once, here.  A NotImplementedError is eaten; the
        // call itself raises it if one is ever made.
        match super::ctypefunc::build_cif_descr(&fargs_now(), roots.get(fresult_slot), abi, None) {
            Ok(cif) => ct.cif_descr = cif,
            Err(e) if e.kind == pyre_interpreter::PyErrorKind::NotImplementedError => {}
            Err(e) => return Err(e),
        }
    }
    let weak = pyre_object::weakref::w_gc_weakref_box_new_or_strong(roots.get(ctype_slot));
    let weak_slot = roots.pin_roots(&[weak]);
    let mut cache = function_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // A concurrent builder may have published the same live function type
    // while this one was being constructed.  Preserve PyPy's unique result.
    for weakdict in cache.iter() {
        if let Some(&existing_root_slot) = weakdict.get(&func_hash) {
            let existing = unsafe {
                pyre_object::weakref::w_gc_weakref_box_or_strong_deref(rooted_weakref_at(
                    existing_root_slot,
                ))
            };
            if function_type_matches(
                existing,
                &fargs_now(),
                roots.get(fresult_slot),
                ellipsis,
                abi,
            ) {
                return Ok(existing);
            }
        }
    }
    for weakdict in cache.iter_mut() {
        match weakdict.get(&func_hash).copied() {
            None => {
                let root_slot = ctypeobj::root_forever_slot(roots.get(weak_slot)) as usize;
                weakdict.insert(func_hash, root_slot);
                return Ok(roots.get(ctype_slot));
            }
            Some(existing_root_slot) => {
                let existing_weak = rooted_weakref_at(existing_root_slot);
                let dead = unsafe {
                    pyre_object::weakref::w_gc_weakref_box_or_strong_deref(existing_weak).is_null()
                };
                if dead {
                    if unsafe {
                        pyre_object::weakref::w_gc_weakref_box_retarget(
                            existing_weak,
                            roots.get(ctype_slot),
                        )
                    } {
                        return Ok(roots.get(ctype_slot));
                    }
                    let root_slot = ctypeobj::root_forever_slot(roots.get(weak_slot)) as usize;
                    weakdict.insert(func_hash, root_slot);
                    return Ok(roots.get(ctype_slot));
                }
            }
        }
    }
    let root_slot = ctypeobj::root_forever_slot(roots.get(weak_slot)) as usize;
    cache.push(HashMap::from([(func_hash, root_slot)]));
    Ok(roots.get(ctype_slot))
}
