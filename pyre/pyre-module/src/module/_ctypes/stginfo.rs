//! `StgInfo` — per-ctypes-type storage metadata (size/align/layout).
//!
//! pyre types have no type-data slot to hang this off, so the carrier is a
//! private native object stored in the ctypes type's **own**
//! dict under the reserved key `"__stginfo__"`.  `stginfo_of` reads it with a
//! *direct* dict get (never the MRO), so it is non-inherited by construction —
//! layout inheritance is explicit cloning in the metaclass.  Every reference it
//! holds (`proto`, `pointer_type`) is an ordinary dict entry, so it stays
//! GC-correct with no Rust-side side tables.

use pyre_object::PyObjectRef;

use super::cfield;
use super::ctypes_h;

/// Reserved key under which the carrier lives in a ctypes type's own dict.
const STGINFO_KEY: &str = "__stginfo__";

// Storage-flag bits (subset of the full flag set).
pub(super) const TYPEFLAG_ISPOINTER: i64 = ctypes_h::TYPEFLAG_ISPOINTER as i64;
pub(super) const TYPEFLAG_HASPOINTER: i64 = ctypes_h::TYPEFLAG_HASPOINTER as i64;
pub(super) const TYPEFLAG_HASUNION: i64 = 0x400;
pub(super) const DICTFLAG_FINAL: i64 = ctypes_h::DICTFLAG_FINAL as i64;

// Carrier field keys.
const K_SIZE: &str = "size";
const K_ALIGN: &str = "align";
const K_LENGTH: &str = "length";
const K_ELEMENT_SIZE: &str = "element_size";
const K_FLAGS: &str = "flags";
const K_PARAMFUNC: &str = "paramfunc";
const K_PROTO: &str = "proto";
const K_FORMAT: &str = "format";
const K_POINTER_TYPE: &str = "pointer_type";
const K_BIG_ENDIAN: &str = "big_endian";

static STGINFO_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

/// The private `StgInfo` carrier type (`hasdict=true`, never registered).
fn stginfo_type() -> PyObjectRef {
    STGINFO_TYPE_OBJ.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type("StgInfo", |_| {});
        unsafe { pyre_interpreter::objspace::std::mapdict::set_type_hasdict(tp, true) };
        tp
    })
}

/// `StgInfo.paramfunc` — the storage shape a ctypes type has.
///
/// The carrier keeps it spelled out as a string, so a shape is one comparison
/// on the way in and out rather than a fresh allocation at every read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ParamFunc {
    Simple,
    Array,
    Pointer,
    Struct,
    Union,
    /// A carrier with no `paramfunc`, or one naming a shape this build has no
    /// arm for.
    Other,
}

impl ParamFunc {
    /// The spelling the carrier stores.
    fn name(self) -> &'static str {
        match self {
            ParamFunc::Simple => "simple",
            ParamFunc::Array => "array",
            ParamFunc::Pointer => "pointer",
            ParamFunc::Struct => "struct",
            ParamFunc::Union => "union",
            ParamFunc::Other => "",
        }
    }

    fn from_name(name: &str) -> Self {
        match name {
            "simple" => ParamFunc::Simple,
            "array" => ParamFunc::Array,
            "pointer" => ParamFunc::Pointer,
            "struct" => ParamFunc::Struct,
            "union" => ParamFunc::Union,
            _ => ParamFunc::Other,
        }
    }
}

/// Field values for [`stginfo_new`].
pub(super) struct StgInfoData {
    pub size: usize,
    pub align: usize,
    pub length: usize,
    pub element_size: usize,
    pub flags: i64,
    pub paramfunc: ParamFunc,
    pub proto: Option<PyObjectRef>,
    pub format: Option<String>,
    pub big_endian: bool,
}

impl StgInfoData {
    /// A minimal simple/aggregate carrier of the given size and alignment.
    pub(super) fn new(size: usize, align: usize, paramfunc: ParamFunc) -> Self {
        StgInfoData {
            size,
            align: align.max(1),
            length: 0,
            element_size: 0,
            flags: 0,
            paramfunc,
            proto: None,
            format: None,
            big_endian: cfg!(target_endian = "big"),
        }
    }
}

fn dict_of(info: PyObjectRef) -> PyObjectRef {
    pyre_interpreter::baseobjspace::getdict_native(info)
}

fn set_int(info: PyObjectRef, key: &str, v: i64) {
    unsafe { pyre_object::w_dict_setitem_str(dict_of(info), key, pyre_object::w_int_new(v)) };
}

fn get_int(info: PyObjectRef, key: &str) -> i64 {
    match unsafe { pyre_object::w_dict_getitem_str(dict_of(info), key) } {
        Some(o) if unsafe { pyre_object::is_int(o) } => unsafe { pyre_object::w_int_get_value(o) },
        _ => 0,
    }
}

/// Build a fresh `StgInfo` carrier from `data`.
pub(super) fn stginfo_new(data: StgInfoData) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let info_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_instance_new(stginfo_type()));
    let info = || pyre_object::gc_roots::shadow_stack_get(info_slot);
    let d = || dict_of(info());
    unsafe {
        pyre_object::w_dict_setitem_str(d(), K_SIZE, pyre_object::w_int_new(data.size as i64));
        pyre_object::w_dict_setitem_str(d(), K_ALIGN, pyre_object::w_int_new(data.align as i64));
        pyre_object::w_dict_setitem_str(d(), K_LENGTH, pyre_object::w_int_new(data.length as i64));
        pyre_object::w_dict_setitem_str(
            d(),
            K_ELEMENT_SIZE,
            pyre_object::w_int_new(data.element_size as i64),
        );
        pyre_object::w_dict_setitem_str(d(), K_FLAGS, pyre_object::w_int_new(data.flags));
        pyre_object::w_dict_setitem_str(
            d(),
            K_PARAMFUNC,
            pyre_object::w_str_new_managed(data.paramfunc.name()),
        );
        pyre_object::w_dict_setitem_str(
            d(),
            K_PROTO,
            data.proto.unwrap_or_else(pyre_object::w_none),
        );
        pyre_object::w_dict_setitem_str(
            d(),
            K_FORMAT,
            match data.format {
                Some(s) => pyre_object::w_str_new_managed(&s),
                None => pyre_object::w_none(),
            },
        );
        pyre_object::w_dict_setitem_str(d(), K_POINTER_TYPE, pyre_object::w_none());
        pyre_object::w_dict_setitem_str(
            d(),
            K_BIG_ENDIAN,
            pyre_object::w_bool_from(data.big_endian),
        );
    }
    info()
}

/// Direct (non-MRO) lookup of `cls`'s own `StgInfo`, if any.
pub(super) fn stginfo_of(cls: PyObjectRef) -> Option<PyObjectRef> {
    if cls.is_null() || !unsafe { pyre_object::is_type(cls) } {
        return None;
    }
    let info = pyre_interpreter::type_dict_lookup(cls, STGINFO_KEY)?;
    (!unsafe { pyre_object::is_none(info) }).then_some(info)
}

/// Store `info` as `cls`'s `StgInfo` and invalidate the type cache.
pub(super) fn stginfo_set(mut cls: PyObjectRef, mut info: PyObjectRef) {
    pyre_object::with_roots!(cls, info => {
        if pyre_interpreter::type_dict_store(cls, STGINFO_KEY, info) {
            pyre_object::gc_hook::try_gc_write_barrier(cls as pyre_object::gc_hook::GCREF);
            unsafe { pyre_interpreter::baseobjspace::mutated(cls, STGINFO_KEY) };
        }
    });
}

pub(super) fn stginfo_size(info: PyObjectRef) -> usize {
    get_int(info, K_SIZE).max(0) as usize
}

pub(super) fn stginfo_align(info: PyObjectRef) -> usize {
    (get_int(info, K_ALIGN).max(1)) as usize
}

pub(super) fn stginfo_length(info: PyObjectRef) -> usize {
    get_int(info, K_LENGTH).max(0) as usize
}

/// The per-element byte size of an array `StgInfo` (0 for non-arrays).
pub(super) fn stginfo_element_size(info: PyObjectRef) -> usize {
    get_int(info, K_ELEMENT_SIZE).max(0) as usize
}

pub(super) fn stginfo_flags(info: PyObjectRef) -> i64 {
    get_int(info, K_FLAGS)
}

pub(super) fn stginfo_paramfunc(info: PyObjectRef) -> ParamFunc {
    match unsafe { pyre_object::w_dict_getitem_str(dict_of(info), K_PARAMFUNC) } {
        Some(o) if unsafe { pyre_object::is_str(o) } => {
            match unsafe { pyre_object::w_str_get_value_opt(o) } {
                Some(name) => ParamFunc::from_name(name),
                None => ParamFunc::Other,
            }
        }
        _ => ParamFunc::Other,
    }
}

pub(super) fn stginfo_format(info: PyObjectRef) -> Option<String> {
    match unsafe { pyre_object::w_dict_getitem_str(dict_of(info), K_FORMAT) } {
        Some(o) if unsafe { pyre_object::is_str(o) } => {
            pyre_interpreter::baseobjspace::str_utf8_w(o)
                .ok()
                .map(str::to_string)
        }
        _ => None,
    }
}

pub(super) fn stginfo_big_endian(info: PyObjectRef) -> bool {
    match unsafe { pyre_object::w_dict_getitem_str(dict_of(info), K_BIG_ENDIAN) } {
        Some(value) => pyre_interpreter::baseobjspace::is_true(value).unwrap_or(false),
        None => false,
    }
}

/// The pointed-to / element type (`proto`), or `None` when unset.
pub(super) fn stginfo_proto(info: PyObjectRef) -> Option<PyObjectRef> {
    match unsafe { pyre_object::w_dict_getitem_str(dict_of(info), K_PROTO) } {
        Some(o) if !unsafe { pyre_object::is_none(o) } => Some(o),
        _ => None,
    }
}

pub(super) fn stginfo_is_final(info: PyObjectRef) -> bool {
    stginfo_flags(info) & DICTFLAG_FINAL != 0
}

pub(super) fn stginfo_mark_final(mut info: PyObjectRef) {
    let f = pyre_object::with_roots!(info => stginfo_flags(info)) | DICTFLAG_FINAL;
    set_int(info, K_FLAGS, f);
}

/// The cached `__pointer_type__`, or `None` when unset.
pub(super) fn stginfo_pointer_type(info: PyObjectRef) -> Option<PyObjectRef> {
    match unsafe { pyre_object::w_dict_getitem_str(dict_of(info), K_POINTER_TYPE) } {
        Some(o) if !unsafe { pyre_object::is_none(o) } => Some(o),
        _ => None,
    }
}

pub(super) fn stginfo_set_pointer_type(info: PyObjectRef, mut ty: PyObjectRef) {
    let w_dict = pyre_object::with_roots!(ty => dict_of(info));
    unsafe { pyre_object::w_dict_setitem_str(w_dict, K_POINTER_TYPE, ty) };
}

// ── field size/align with the `_type_` fallback ───────────────────────

/// Byte size of a ctypes type `t`: its `StgInfo` size, else the simple-type
/// size derived from `_type_`.
pub(super) fn field_size_of(t: PyObjectRef) -> Option<usize> {
    if let Some(info) = stginfo_of(t) {
        return Some(stginfo_size(info));
    }
    let tc = super::cdata::type_code_of(t)?;
    cfield::simple_type_size(&tc)
}

/// Alignment of a ctypes type `t`: its `StgInfo` align, else the simple-type
/// alignment derived from `_type_`.
pub(super) fn field_align_of(t: PyObjectRef) -> Option<usize> {
    if let Some(info) = stginfo_of(t) {
        return Some(stginfo_align(info));
    }
    let tc = super::cdata::type_code_of(t)?;
    cfield::simple_type_align(&tc)
}
