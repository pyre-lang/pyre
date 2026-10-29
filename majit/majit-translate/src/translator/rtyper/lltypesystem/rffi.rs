//! RPython `rpython/rtyper/lltypesystem/rffi.py`.
//!
//! This slice exposes the exact lltype-facing surface that can be represented
//! with pyre's current `lltype` port: errno flags, C array/struct helper
//! constructors, callback pointer types, and the standard raw pointer aliases.
//! Width-specific C integer aliases and full wrapper-generation behavior remain
//! deferred until the corresponding RPython platform/type metadata is ported.
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use indexmap::IndexMap;
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::{Arc, LazyLock};

use crate::annotator::annrpython::RPythonAnnotator;
use crate::annotator::dictdef::DictDef;
use crate::annotator::model::{SomeInteger, SomeValue};
use crate::flowspace::model::{
    AlwaysInline, Block, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc,
    GraphRef, Hlvalue, Link, SpaceOperation, Variable, c_last_exception,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::llmemory;
use crate::translator::rtyper::lltypesystem::lltype::{
    _ptr, Array, FixedSizeArray, FuncType, GCREF, LowLevelType, LowLevelValue, MallocFlavor,
    OpaqueType, Ptr, PtrTarget, SomePtr, Struct, functionptr_with_external_name, malloc, typeOf,
};
use crate::translator::rtyper::lltypesystem::rordereddict::OrderedDictRepr;
use crate::translator::rtyper::lltypesystem::rstr::{
    STRPTR, chars_array_ptr_lltype_from_strptr, direct_call_funcobjs, graph_ref_for_funcptr_id,
    struct_lltype_from_strptr, sub_helper_funcptr_constant,
};
use crate::translator::rtyper::rint::signed_repr;
use crate::translator::rtyper::rmodel::{PtrRepr, Repr, lowlevel_type_const};
use crate::translator::rtyper::rtyper::{
    HighLevelOp, LowLevelFunction, RPythonTyper, constant_with_lltype, exception_args,
    helper_pygraph_from_graph, variable_with_lltype,
};

pub use majit_jitcode::rffi::{
    RFFI_ALT_ERRNO, RFFI_ERR_ALL, RFFI_ERR_NONE, RFFI_FULL_ERRNO, RFFI_FULL_ERRNO_ZERO,
    RFFI_FULL_LASTERROR, RFFI_READSAVED_ERRNO, RFFI_READSAVED_LASTERROR, RFFI_SAVE_ERRNO,
    RFFI_SAVE_LASTERROR, RFFI_SAVE_WSALASTERROR, RFFI_ZERO_ERRNO_BEFORE,
};

/// RPython `_isfunctype(TP)` (`rffi.py`).
pub fn _isfunctype(TP: &LowLevelType) -> bool {
    matches!(TP, LowLevelType::Ptr(ptr) if matches!(ptr.TO, PtrTarget::Func(_)))
}

/// RPython `_isllptr(p)` (`rffi.py`). The Rust signature accepts only
/// `lltype::_ptr`, so every value reaching this helper is a low-level pointer.
pub fn _isllptr(_p: &_ptr) -> bool {
    true
}

/// RPython `_IsLLPtrEntry(ExtRegistryEntry)` (`rffi.py`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct _IsLLPtrEntry;

/// RPython primitive aliases (`rffi.py:739-779`).
pub const CHAR: LowLevelType = LowLevelType::Char;
pub const DOUBLE: LowLevelType = LowLevelType::Float;
pub const LONGDOUBLE: LowLevelType = LowLevelType::LongFloat;
pub const FLOAT: LowLevelType = LowLevelType::SingleFloat;
pub const SIGNED: LowLevelType = LowLevelType::Signed;
pub const UNSIGNED: LowLevelType = LowLevelType::Unsigned;
pub const r_singlefloat: LowLevelType = LowLevelType::SingleFloat;
pub const NULL: Option<()> = None;

/// RPython `TYPES` seed list (`rffi.py`), narrowed only where the
/// platform-dependent `CompilationError` branch cannot be evaluated in this
/// static port.
pub static TYPES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    #[cfg_attr(windows, allow(unused_mut))]
    let mut types = vec![
        "short",
        "unsigned short",
        "int",
        "unsigned int",
        "long",
        "unsigned long",
        "signed char",
        "unsigned char",
        "long long",
        "unsigned long long",
        "size_t",
        "time_t",
        "wchar_t",
        "uintptr_t",
        "intptr_t",
        "void*",
    ];
    #[cfg(not(windows))]
    {
        types.extend([
            "mode_t",
            "pid_t",
            "ssize_t",
            "ptrdiff_t",
            "int_least8_t",
            "uint_least8_t",
            "int_least16_t",
            "uint_least16_t",
            "int_least32_t",
            "uint_least32_t",
            "int_least64_t",
            "uint_least64_t",
            "int_fast8_t",
            "uint_fast8_t",
            "int_fast16_t",
            "uint_fast16_t",
            "int_fast32_t",
            "uint_fast32_t",
            "int_fast64_t",
            "uint_fast64_t",
            "intmax_t",
            "uintmax_t",
        ]);
    }
    types
});

fn rffi_name_from_c_type(mut name: &str) -> String {
    if let Some(rest) = name.strip_prefix("unsigned") {
        name = rest.trim_start();
        format!("u{}", name.replace(' ', ""))
    } else {
        name.replace(' ', "")
    }
}

/// RPython `populate_inttypes()` (`rffi.py`).
pub fn populate_inttypes() -> Vec<String> {
    TYPES
        .iter()
        .map(|name| rffi_name_from_c_type(name))
        .collect()
}

/// RPython `setup()` (`rffi.py`), represented by the primitive
/// low-level types available before the full platform cache is ported.
pub fn setup() -> Vec<LowLevelType> {
    populate_inttypes()
        .into_iter()
        .map(|name| {
            if name == "void*" {
                (*VOIDP).clone()
            } else if name.starts_with('u') || name == "size_t" || name == "uintptr_t" {
                LowLevelType::Unsigned
            } else if name == "wchar_t" {
                LowLevelType::UniChar
            } else {
                LowLevelType::Signed
            }
        })
        .collect()
}

pub static NUMBER_TYPES: LazyLock<Vec<LowLevelType>> = LazyLock::new(setup);

/// RPython wrap-around integer class names (`rffi.py:578-584`). The current
/// lltype port carries the low-level primitive identity, not rarithmetic's
/// generated Python class object.
pub const r_int_real: LowLevelType = LowLevelType::Signed;
pub const r_uint_real: LowLevelType = LowLevelType::Unsigned;

fn ptr_to_array(of: LowLevelType, hints: Vec<(String, ConstValue)>) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(Array::with_hints(of, hints)),
    }))
}

fn nolength_hints() -> Vec<(String, ConstValue)> {
    vec![("nolength".into(), ConstValue::Bool(true))]
}

fn void_hints(render_as_const: bool) -> Vec<(String, ConstValue)> {
    let mut hints = nolength_hints();
    hints.push(("render_as_void".into(), ConstValue::Bool(true)));
    if render_as_const {
        hints.push(("render_as_const".into(), ConstValue::Bool(true)));
    }
    hints
}

fn const_hints() -> Vec<(String, ConstValue)> {
    let mut hints = nolength_hints();
    hints.push(("render_as_const".into(), ConstValue::Bool(true)));
    hints
}

/// `void *` (`rffi.py:746`), represented by upstream as a no-length char array
/// pointer with the `render_as_void` hint.
pub static VOIDP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(LowLevelType::Char, void_hints(false)));

/// `const void *` (`rffi.py:747`).
pub static CONST_VOIDP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(LowLevelType::Char, void_hints(true)));

/// `void **` (`rffi.py`).
pub static VOIDPP: LazyLock<LowLevelType> = LazyLock::new(|| CArrayPtr((*VOIDP).clone()));

/// `char *` (`rffi.py:754`).
pub static CCHARP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(LowLevelType::Char, nolength_hints()));

/// `const char *` (`rffi.py`).
pub static CONST_CCHARP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(LowLevelType::Char, const_hints()));

/// `wchar_t *` (`rffi.py`).
pub static CWCHARP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(LowLevelType::UniChar, nolength_hints()));

pub static DOUBLEP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(DOUBLE, nolength_hints()));
pub static FLOATP: LazyLock<LowLevelType> = LazyLock::new(|| ptr_to_array(FLOAT, nolength_hints()));
pub static LONGDOUBLEP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(LONGDOUBLE, nolength_hints()));
pub static SIGNEDP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(SIGNED, nolength_hints()));
pub static SIGNEDPP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array((*SIGNEDP).clone(), nolength_hints()));
pub static UNSIGNEDP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(UNSIGNED, nolength_hints()));
pub static CCHARPP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array((*CCHARP).clone(), nolength_hints()));
pub static CWCHARPP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array((*CWCHARP).clone(), nolength_hints()));

/// RPython `CStruct(name, *fields, **kwds)` (`rffi.py`).
///
/// Upstream prefixes every field with `c_` and adds the C rendering hints.
pub fn CStruct(name: &str, fields: Vec<(String, LowLevelType)>) -> Struct {
    CStruct_with_hints(name, fields, vec![])
}

pub fn CStruct_with_hints(
    name: &str,
    fields: Vec<(String, LowLevelType)>,
    mut hints: Vec<(String, ConstValue)>,
) -> Struct {
    hints.push(("external".into(), ConstValue::byte_str("C")));
    hints.push(("c_name".into(), ConstValue::byte_str(name)));
    let c_fields = fields
        .into_iter()
        .map(|(field, typ)| (format!("c_{field}"), typ))
        .collect();
    Struct::with_hints(name, c_fields, hints)
}

/// RPython `CStructPtr(*args, **kwds)` (`rffi.py`).
pub fn CStructPtr(name: &str, fields: Vec<(String, LowLevelType)>) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(CStruct(name, fields)),
    }))
}

/// RPython `CFixedArray(tp, size)` (`rffi.py`).
pub fn CFixedArray(tp: LowLevelType, size: usize) -> FixedSizeArray {
    FixedSizeArray::new(tp, size)
}

/// RPython `CArray(tp)` (`rffi.py`).
pub fn CArray(tp: LowLevelType) -> Array {
    Array::with_hints(tp, nolength_hints())
}

/// RPython `CArrayPtr(tp)` (`rffi.py`).
pub fn CArrayPtr(tp: LowLevelType) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(CArray(tp)),
    }))
}

/// RPython `CCallback(args, res)` (`rffi.py`).
pub fn CCallback(args: Vec<LowLevelType>, res: LowLevelType) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Func(FuncType { args, result: res }),
    }))
}

/// RPython `COpaque(...)` (`rffi.py`), narrowed to the type identity
/// available in the current `lltype::OpaqueType` port.
pub fn COpaque(name: Option<&str>) -> OpaqueType {
    OpaqueType::new(name.unwrap_or("C"))
}

/// RPython `COpaquePtr(*args, **kwds)` (`rffi.py`).
pub fn COpaquePtr(name: Option<&str>) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Opaque(COpaque(name)),
    }))
}

/// Minimal `llexternal` surface backed by the same function-pointer metadata
/// used by `extfunc.py`. Full wrapper generation remains with the broader
/// `rffi.py` runtime port.
pub fn llexternal(name: &str, args: Vec<LowLevelType>, result: LowLevelType) -> _ptr {
    functionptr_with_external_name(FuncType { args, result }, name, None::<String>)
}

/// RPython `make(STRUCT, **fields)` (`rffi.py`).
pub fn make(STRUCT: Struct, fields: Vec<(String, LowLevelValue)>) -> Result<_ptr, String> {
    let mut ptr = malloc(
        LowLevelType::Struct(Box::new(STRUCT)),
        None,
        MallocFlavor::Raw,
        false,
    )?;
    for (name, value) in fields {
        ptr.setattr(&name, value)?;
    }
    Ok(ptr)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CallbackHolder {
    pub callbacks: HashMap<String, bool>,
}

impl CallbackHolder {
    pub fn new() -> Self {
        CallbackHolder {
            callbacks: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeepaliveKeeper {
    pub stuff_to_keepalive: Vec<Option<ConstValue>>,
    pub free_positions: Vec<usize>,
}

/// RPython `_KEEPER_CACHE = {}` (`rffi.py`). This is intentionally a
/// side cache because upstream uses a module-level dict keyed by low-level
/// type for callback keepalive slots.
pub static _KEEPER_CACHE: LazyLock<Mutex<HashMap<LowLevelType, KeepaliveKeeper>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn _keeper_for_type(TP: LowLevelType) -> KeepaliveKeeper {
    let mut cache = _KEEPER_CACHE.lock();
    cache.entry(TP).or_default().clone()
}

pub fn register_keepalive(TP: LowLevelType, obj: ConstValue) -> usize {
    let mut cache = _KEEPER_CACHE.lock();
    let keeper = cache.entry(TP).or_default();
    if let Some(pos) = keeper.free_positions.pop() {
        keeper.stuff_to_keepalive[pos] = Some(obj);
        pos
    } else {
        let pos = keeper.stuff_to_keepalive.len();
        keeper.stuff_to_keepalive.push(Some(obj));
        pos
    }
}

pub fn get_keepalive_object(pos: usize, TP: LowLevelType) -> Option<ConstValue> {
    let cache = _KEEPER_CACHE.lock();
    cache
        .get(&TP)
        .and_then(|keeper| keeper.stuff_to_keepalive.get(pos).cloned())
        .flatten()
}

pub fn unregister_keepalive(pos: usize, TP: LowLevelType) {
    let mut cache = _KEEPER_CACHE.lock();
    if let Some(keeper) = cache.get_mut(&TP)
        && pos < keeper.stuff_to_keepalive.len()
    {
        keeper.stuff_to_keepalive[pos] = None;
        keeper.free_positions.push(pos);
    }
}

fn deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!(
        "lltypesystem.rffi.{name} — C wrapper/runtime buffer behavior deferred"
    ))
}

pub fn _make_wrapper_for() -> Result<(), TyperError> {
    Err(deferred("_make_wrapper_for"))
}

pub fn llexternal_use_eci() -> Result<(), TyperError> {
    Err(deferred("llexternal_use_eci"))
}

pub fn generate_macro_wrapper() -> Result<(), TyperError> {
    Err(deferred("generate_macro_wrapper"))
}

pub fn CExternVariable() -> Result<(), TyperError> {
    Err(deferred("CExternVariable"))
}

pub fn make_string_mappings() -> Result<(), TyperError> {
    Err(deferred("make_string_mappings"))
}

pub fn constcharp2str() -> Result<(), TyperError> {
    Err(deferred("constcharp2str"))
}

pub fn constcharpsize2str() -> Result<(), TyperError> {
    Err(deferred("constcharpsize2str"))
}

pub fn str2constcharp() -> Result<(), TyperError> {
    Err(deferred("str2constcharp"))
}

pub fn _deprecated_get_nonmovingbuffer() -> Result<(), TyperError> {
    Err(deferred("_deprecated_get_nonmovingbuffer"))
}

pub fn get_nonmovingbuffer() -> Result<(), TyperError> {
    Err(deferred("get_nonmovingbuffer"))
}

pub fn get_nonmovingbuffer_final_null() -> Result<(), TyperError> {
    Err(deferred("get_nonmovingbuffer_final_null"))
}

pub fn free_nonmovingbuffer() -> Result<(), TyperError> {
    Err(deferred("free_nonmovingbuffer"))
}

pub fn wcharpsize2utf8() -> Result<(), TyperError> {
    Err(deferred("wcharpsize2utf8"))
}

pub fn wcharp2utf8() -> Result<(), TyperError> {
    Err(deferred("wcharp2utf8"))
}

pub fn wcharp2utf8n() -> Result<(), TyperError> {
    Err(deferred("wcharp2utf8n"))
}

pub fn utf82wcharp() -> Result<(), TyperError> {
    Err(deferred("utf82wcharp"))
}

pub fn utf82wcharp_ex() -> Result<(), TyperError> {
    Err(deferred("utf82wcharp_ex"))
}

pub fn liststr2charpp() -> Result<(), TyperError> {
    Err(deferred("liststr2charpp"))
}

pub fn ll_liststr2charpp() -> Result<(), TyperError> {
    Err(deferred("ll_liststr2charpp"))
}

pub fn free_charpp() -> Result<(), TyperError> {
    Err(deferred("free_charpp"))
}

pub fn charpp2liststr() -> Result<(), TyperError> {
    Err(deferred("charpp2liststr"))
}

pub fn cast() -> Result<(), TyperError> {
    Err(deferred("cast"))
}

pub fn ptradd() -> Result<(), TyperError> {
    Err(deferred("ptradd"))
}

/// `rffi.py sizeof(tp)`: integer size when known. A `Struct` without a
/// numeric `_hints["size"]` is `llmemory.sizeof(tp)` (symbolic
/// `AddressOffset`). `FixedSizeArray` is `sizeof(OF) * length`.
pub fn sizeof(tp: &LowLevelType) -> Result<ConstValue, TyperError> {
    match tp {
        LowLevelType::FixedSizeArray(a) => match sizeof(&a.OF)? {
            ConstValue::Int(n) if n >= 0 => Ok(ConstValue::Int(n * a.length as i64)),
            ConstValue::AddressOffset(off) => off
                .mul(a.length as i64)
                .map(ConstValue::AddressOffset)
                .ok_or_else(|| TyperError::message(format!("sizeof({tp:?}): cannot scale offset"))),
            other => Err(TyperError::message(format!(
                "sizeof({tp:?}): unexpected element size {other:?}"
            ))),
        },
        LowLevelType::Struct(s) => match s._hints.get("size") {
            Some(ConstValue::Int(n)) if *n >= 0 => Ok(ConstValue::Int(*n)),
            _ => {
                // `llmemory.sizeof(tp)` — a symbolic result in this case.
                llmemory::sizeof(tp, None).map_err(TyperError::message)
            }
        },
        LowLevelType::Signed | LowLevelType::Ptr(_) | LowLevelType::Address => {
            Ok(ConstValue::Int(crate::layout::target_word_size() as i64))
        }
        LowLevelType::Char | LowLevelType::Bool => Ok(ConstValue::Int(1)),
        LowLevelType::UniChar => {
            // `rffi.py sizeof`: `r_wchar_t.BITS/8`. `setup()` maps
            // `wchar_t` to UniChar; `ll2ctypes.py` stores UniChar as
            // `c_uint16` when `sizeof(c_wchar) == 2` else `c_uint32`.
            Ok(ConstValue::Int(llmemory::sizeof_wchar_t()))
        }
        LowLevelType::Float => Ok(ConstValue::Int(8)),
        LowLevelType::SingleFloat => Ok(ConstValue::Int(4)),
        LowLevelType::LongFloat => Ok(ConstValue::Int(llmemory::sizeof_longfloat())),
        LowLevelType::Unsigned => Ok(ConstValue::Int(crate::layout::target_word_size() as i64)),
        LowLevelType::SignedLongLong | LowLevelType::UnsignedLongLong => Ok(ConstValue::Int(8)),
        LowLevelType::SignedLongLongLong | LowLevelType::UnsignedLongLongLong => {
            Ok(ConstValue::Int(16))
        }
        _ => Err(TyperError::message(format!("sizeof({tp:?}): not a Number"))),
    }
}

/// `rffi.py size_and_sign(tp)`: `(sizeof(tp), unsigned)`.
/// `_type.SIGNED` for Number types; non-Primitive is unsigned False;
/// `Signed` / FLOAT / DOUBLE / LONGDOUBLE / Address are False;
/// Char / UniChar / Bool are True. Upstream calls this on Numbers /
/// Ptr; a symbolic sizeof is an error here.
pub fn size_and_sign(tp: &LowLevelType) -> Result<(usize, bool), TyperError> {
    let size = match sizeof(tp)? {
        ConstValue::Int(n) if n >= 0 => n as usize,
        other => {
            return Err(TyperError::message(format!(
                "size_and_sign({tp:?}): need an integer size, got {other:?}"
            )));
        }
    };
    let unsigned = match tp {
        LowLevelType::Unsigned
        | LowLevelType::UnsignedLongLong
        | LowLevelType::UnsignedLongLongLong => true,
        LowLevelType::Signed | LowLevelType::SignedLongLong | LowLevelType::SignedLongLongLong => {
            false
        }
        _ if !tp.is_primitive() => false,
        LowLevelType::Float
        | LowLevelType::SingleFloat
        | LowLevelType::LongFloat
        | LowLevelType::Address => false,
        LowLevelType::Char | LowLevelType::UniChar | LowLevelType::Bool => true,
        _ => return Err(TyperError::message(format!("size_and_sign({tp:?})"))),
    };
    Ok((size, unsigned))
}

pub fn offsetof() -> Result<(), TyperError> {
    Err(deferred("offsetof"))
}

/// RPython `MakeEntry(ExtRegistryEntry)` (`rffi.py`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MakeEntry;

pub fn structcopy() -> Result<(), TyperError> {
    Err(deferred("structcopy"))
}

pub fn _get_structcopy_fn() -> Result<(), TyperError> {
    Err(deferred("_get_structcopy_fn"))
}

pub fn setintfield() -> Result<(), TyperError> {
    Err(deferred("setintfield"))
}

pub fn getintfield() -> Result<(), TyperError> {
    Err(deferred("getintfield"))
}

/// RPython scoped raw string buffer context managers (`rffi.py:1386-1492`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_str2charp;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_unicode2wcharp;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_utf82wcharp;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_nonmovingbuffer;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_view_charp;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_nonmoving_unicodebuffer;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_alloc_buffer;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_alloc_unicodebuffer;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_alloc_utf8buffer;

pub fn c_memcpy() -> Result<(), TyperError> {
    Err(deferred("c_memcpy"))
}

pub fn c_memset() -> Result<(), TyperError> {
    Err(deferred("c_memset"))
}

/// RPython `TEST_RAW_ADDR_KEEP_ALIVE = {}` (`rffi.py`).
pub static TEST_RAW_ADDR_KEEP_ALIVE: LazyLock<Mutex<HashMap<String, LowLevelType>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Synthesise `rffi.get_raw_address_of_string` (`rffi.py`) in the
/// `we_are_translated()` body:
///
/// ```python
/// if rgc.must_split_gc_address_space():
///     return _get_raw_address_buf_from_string(string)
/// if rgc.can_move(string):
///     string = rgc.move_out_of_nursery(string)
///     if rgc.can_move(string):
///         return _get_raw_address_buf_from_string(string)
/// lldata = llstr(string)
/// data_start = (cast_ptr_to_adr(lldata) + offsetof(STR, 'chars')
///               + itemoffsetof(STR.chars, 0))
/// data_start = cast(CCHARP, data_start)
/// data_start[len(string)] = '\x00'
/// return data_start
/// ```
///
/// `rgc.must_split_gc_address_space` (`SplitAddrSpaceEntry`) is a
/// translation-time constant; MiniMark is False, so the first buf
/// return is folded. `@jit.dont_look_inside` is `_jit_look_inside_ =
/// Some(false)`. The inner `if rgc.can_move` after
/// `move_out_of_nursery` `direct_call`s `_get_raw_address_buf_from_string`
/// as a graph-less residual (`functionptr_with_external_name`, the
/// `llexternal` sibling). The NUL is `setarrayitem` on `STR.chars`.
pub(crate) fn build_ll_get_raw_address_of_string_helper_graph(
    name: &str,
) -> Result<PyGraph, TyperError> {
    let str_struct = struct_lltype_from_strptr(&STRPTR)?;
    let chars_array_ty = match &str_struct {
        LowLevelType::Struct(body) => body
            ._flds
            .get("chars")
            .cloned()
            .ok_or_else(|| TyperError::message("STR struct has no chars field"))?,
        _ => {
            return Err(TyperError::message(
                "ll_get_raw_address_of_string: STR body is not a Struct",
            ));
        }
    };
    let offsetof_chars = llmemory::offsetof(&str_struct, "chars").map_err(TyperError::message)?;
    let itemoffset_chars =
        llmemory::itemoffsetof(&chars_array_ty, 0).map_err(TyperError::message)?;
    let chars_ptr_ty = chars_array_ptr_lltype_from_strptr(&STRPTR)?;

    let bool_const = |b: bool| constant_with_lltype(ConstValue::Bool(b), LowLevelType::Bool);
    let push = |block: &BlockRef, op: &str, args: Vec<Hlvalue>, result: Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            op,
            args,
            Hlvalue::Variable(result),
        ));
    };
    let switch_bool = |block: &BlockRef,
                       cond: Variable,
                       true_args: Vec<Hlvalue>,
                       true_tgt: BlockRef,
                       false_args: Vec<Hlvalue>,
                       false_tgt: BlockRef| {
        block.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
        block.closeblock(vec![
            Link::new(true_args, Some(true_tgt), Some(bool_const(true))).into_ref(),
            Link::new(false_args, Some(false_tgt), Some(bool_const(false))).into_ref(),
        ]);
    };

    let s = variable_with_lltype("s", STRPTR.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(s.clone())]);
    let return_var = variable_with_lltype("result", (*CCHARP).clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let s_moved = variable_with_lltype("s", STRPTR.clone());
    let moveblock = Block::shared(vec![Hlvalue::Variable(s_moved.clone())]);
    let s_imm = variable_with_lltype("s", STRPTR.clone());
    let addrblock = Block::shared(vec![Hlvalue::Variable(s_imm.clone())]);
    let s_buf = variable_with_lltype("s", STRPTR.clone());
    let bufblock = Block::shared(vec![Hlvalue::Variable(s_buf.clone())]);

    let can_move = variable_with_lltype("can_move", LowLevelType::Bool);
    push(
        &startblock,
        "gc_can_move",
        vec![Hlvalue::Variable(s.clone())],
        can_move.clone(),
    );
    switch_bool(
        &startblock,
        can_move,
        vec![Hlvalue::Variable(s.clone())],
        moveblock.clone(),
        vec![Hlvalue::Variable(s.clone())],
        addrblock.clone(),
    );

    let s2 = variable_with_lltype("s", STRPTR.clone());
    push(
        &moveblock,
        "gc_move_out_of_nursery",
        vec![Hlvalue::Variable(s_moved)],
        s2.clone(),
    );
    let can_move2 = variable_with_lltype("can_move", LowLevelType::Bool);
    push(
        &moveblock,
        "gc_can_move",
        vec![Hlvalue::Variable(s2.clone())],
        can_move2.clone(),
    );
    switch_bool(
        &moveblock,
        can_move2,
        vec![Hlvalue::Variable(s2.clone())],
        bufblock.clone(),
        vec![Hlvalue::Variable(s2)],
        addrblock.clone(),
    );

    let buf_funcptr = functionptr_with_external_name(
        FuncType {
            args: vec![STRPTR.clone()],
            result: (*CCHARP).clone(),
        },
        "_get_raw_address_buf_from_string",
        Some("_get_raw_address_buf_from_string".to_string()),
    );
    let buf_funcptr_lltype = LowLevelType::Ptr(Box::new(typeOf(&buf_funcptr)));
    let buf_result = variable_with_lltype("result", (*CCHARP).clone());
    push(
        &bufblock,
        "direct_call",
        vec![
            constant_with_lltype(ConstValue::LLPtr(Box::new(buf_funcptr)), buf_funcptr_lltype),
            Hlvalue::Variable(s_buf),
        ],
        buf_result.clone(),
    );
    bufblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(buf_result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let s_imm_chars = s_imm.clone();
    let adr = variable_with_lltype("adr", LowLevelType::Address);
    push(
        &addrblock,
        "cast_ptr_to_adr",
        vec![Hlvalue::Variable(s_imm)],
        adr.clone(),
    );
    let adr1 = variable_with_lltype("adr", LowLevelType::Address);
    push(
        &addrblock,
        "adr_add",
        vec![
            Hlvalue::Variable(adr),
            constant_with_lltype(
                ConstValue::AddressOffset(offsetof_chars),
                LowLevelType::Signed,
            ),
        ],
        adr1.clone(),
    );
    let adr2 = variable_with_lltype("adr", LowLevelType::Address);
    push(
        &addrblock,
        "adr_add",
        vec![
            Hlvalue::Variable(adr1),
            constant_with_lltype(
                ConstValue::AddressOffset(itemoffset_chars),
                LowLevelType::Signed,
            ),
        ],
        adr2.clone(),
    );
    let data_start = variable_with_lltype("data_start", (*CCHARP).clone());
    push(
        &addrblock,
        "cast_adr_to_ptr",
        vec![Hlvalue::Variable(adr2)],
        data_start.clone(),
    );
    let chars = variable_with_lltype("chars", chars_ptr_ty);
    push(
        &addrblock,
        "getsubstruct",
        vec![
            Hlvalue::Variable(s_imm_chars),
            constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void),
        ],
        chars.clone(),
    );
    let strlen = variable_with_lltype("length", LowLevelType::Signed);
    push(
        &addrblock,
        "getarraysize",
        vec![Hlvalue::Variable(chars.clone())],
        strlen.clone(),
    );
    let set_nul = variable_with_lltype("set_nul", LowLevelType::Void);
    push(
        &addrblock,
        "setarrayitem",
        vec![
            Hlvalue::Variable(chars),
            Hlvalue::Variable(strlen),
            constant_with_lltype(ConstValue::ByteStr(vec![0]), LowLevelType::Char),
        ],
        set_nul,
    );
    addrblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(data_start)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    func._jit_look_inside_ = Some(false);
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

fn helper_funcptr_hlvalue(
    rtyper: &RPythonTyper,
    helper: &LowLevelFunction,
) -> Result<Hlvalue, TyperError> {
    Ok(Hlvalue::Constant(sub_helper_funcptr_constant(
        rtyper, helper,
    )?))
}

/// Identity-bearing `CDefinedIntSymbolic` for `_fq_addr_from_string`
/// (`rgc.py FqTagEntry`: `CDefinedIntSymbolic(..., default=fq)`).
/// `framework.py get_finalizer_queue_index` reads `.default` off the tag.
fn fq_addr_from_string_tag_id() -> u64 {
    std::ptr::from_ref(&*_fq_addr_from_string) as u64
}

fn ccharp_array_lltype() -> Result<LowLevelType, TyperError> {
    match &*CCHARP {
        LowLevelType::Ptr(ptr) => match &ptr.TO {
            PtrTarget::Array(arr) => Ok(LowLevelType::Array(Box::new(arr.clone()))),
            other => Err(TyperError::message(format!(
                "CCHARP.TO is not Array: {other:?}"
            ))),
        },
        other => Err(TyperError::message(format!(
            "CCHARP is not a Ptr: {other:?}"
        ))),
    }
}

fn raw_flavor_no_track_const() -> Result<Hlvalue, TyperError> {
    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    let flags = IndexMap::from([
        (ConstValue::byte_str("flavor"), ConstValue::byte_str("raw")),
        (
            ConstValue::byte_str("track_allocation"),
            ConstValue::Bool(false),
        ),
    ]);
    HighLevelOp::inputconst(
        &LowLevelType::Void,
        &ConstValue::Dict(std::sync::Arc::new(flags)),
    )
    .map(Hlvalue::Constant)
}

fn signed_ccharp_dict_repr(rtyper: Rc<RPythonTyper>) -> Result<OrderedDictRepr, TyperError> {
    let ccharp_ptr = match &*CCHARP {
        LowLevelType::Ptr(ptr) => (**ptr).clone(),
        other => {
            return Err(TyperError::message(format!(
                "CCHARP is not a Ptr: {other:?}"
            )));
        }
    };
    let dictdef = DictDef::new(
        None,
        SomeValue::Integer(SomeInteger::default()),
        SomeValue::Ptr(SomePtr::new(ccharp_ptr.clone())),
        false,
        false,
        false,
    );
    OrderedDictRepr::new(
        rtyper,
        signed_repr() as Arc<dyn Repr>,
        Arc::new(PtrRepr::new(ccharp_ptr)) as Arc<dyn Repr>,
        dictdef,
        None,
        false,
        false,
    )
}

/// Convert `_fq_addr_from_string.raw_copies` (`rffi.py`) to the rtyped
/// `{Signed: CCHARP}` dict constant. `OrderedDictRepr.convert_const` of a
/// host `Dict` is the rtyper path (`rordereddict.py convert_const`); an
/// empty table is the immortal `ll_newdict_size(DICT, 0)` /
/// `ll_no_initial_index` container.
fn convert_fq_raw_copies_const(repr: &OrderedDictRepr) -> Result<Hlvalue, TyperError> {
    let host = {
        let fq = _fq_addr_from_string.lock();
        let mut items = IndexMap::new();
        for (k, ptr) in fq.raw_copies.iter() {
            items.insert(
                ConstValue::Int(*k as i64),
                ConstValue::LLPtr(Box::new(ptr.clone())),
            );
        }
        ConstValue::Dict(std::sync::Arc::new(items))
    };
    Ok(Hlvalue::Constant(repr.convert_const(&host)?))
}

/// Synthesise `copy_string_to_raw(src, ptrdst, srcstart, length)` (`rstr.py`)
/// MiniMark path (`must_split_gc_address_space()` is False):
///
/// ```python
/// asrc = _get_raw_buf(SRC_TP, src, srcstart)
/// adst = cast_ptr_to_adr(ptrdst)
/// adst = adst + itemoffsetof(typeOf(ptrdst).TO, 0)
/// raw_memcopy(asrc, adst, sizeof(Char) * length)
/// keepalive_until_here(src)
/// ```
///
/// `_get_raw_buf` (`rstr.py`) is `_always_inline_`: `cast_ptr_to_adr(src)
/// + offsetof(STR, 'chars') + itemoffsetof(STR.chars, 0) + sizeof(Char)*ofs`.
/// `@jit.oopspec('stroruni.copy_string_to_raw(src, ptrdst, srcstart, length)')`.
fn build_copy_string_to_raw_helper_graph(name: &str) -> Result<PyGraph, TyperError> {
    let str_struct = struct_lltype_from_strptr(&STRPTR)?;
    let chars_array_ty = match &str_struct {
        LowLevelType::Struct(body) => body
            ._flds
            .get("chars")
            .cloned()
            .ok_or_else(|| TyperError::message("STR struct has no chars field"))?,
        _ => {
            return Err(TyperError::message(
                "copy_string_to_raw: STR body is not a Struct",
            ));
        }
    };
    let offsetof_chars = llmemory::offsetof(&str_struct, "chars").map_err(TyperError::message)?;
    let itemoffset_chars =
        llmemory::itemoffsetof(&chars_array_ty, 0).map_err(TyperError::message)?;
    let sizeof_char = llmemory::sizeof(&LowLevelType::Char, None).map_err(TyperError::message)?;
    let dst_array_ty = ccharp_array_lltype()?;
    let itemoffset_dst = llmemory::itemoffsetof(&dst_array_ty, 0).map_err(TyperError::message)?;

    let signed_offset = |off: llmemory::AddressOffset| {
        constant_with_lltype(ConstValue::AddressOffset(off), LowLevelType::Signed)
    };
    let push = |block: &BlockRef, op: &str, args: Vec<Hlvalue>, result: Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            op,
            args,
            Hlvalue::Variable(result),
        ));
    };

    let src = variable_with_lltype("src", STRPTR.clone());
    let ptrdst = variable_with_lltype("ptrdst", (*CCHARP).clone());
    let srcstart = variable_with_lltype("srcstart", LowLevelType::Signed);
    let length = variable_with_lltype("length", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(src.clone()),
        Hlvalue::Variable(ptrdst.clone()),
        Hlvalue::Variable(srcstart.clone()),
        Hlvalue::Variable(length.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // `assert length >= 0` (`rstr.py copy_string_to_raw`). Python `assert`
    // becomes `debug_assert` in `removeassert.kill_assertion_link`.
    let ge0 = variable_with_lltype("ge0", LowLevelType::Bool);
    push(
        &startblock,
        "int_ge",
        vec![
            Hlvalue::Variable(length.clone()),
            constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
        ],
        ge0.clone(),
    );
    let assert_void = variable_with_lltype("v", LowLevelType::Void);
    push(
        &startblock,
        "debug_assert",
        vec![
            Hlvalue::Variable(ge0),
            constant_with_lltype(
                ConstValue::byte_str("assertion failed in copy_string_to_raw"),
                LowLevelType::Void,
            ),
        ],
        assert_void,
    );

    let src_adr = variable_with_lltype("asrc", LowLevelType::Address);
    push(
        &startblock,
        "cast_ptr_to_adr",
        vec![Hlvalue::Variable(src.clone())],
        src_adr.clone(),
    );
    let src_chars = variable_with_lltype("asrc", LowLevelType::Address);
    push(
        &startblock,
        "adr_add",
        vec![Hlvalue::Variable(src_adr), signed_offset(offsetof_chars)],
        src_chars.clone(),
    );
    let src_items = variable_with_lltype("asrc", LowLevelType::Address);
    push(
        &startblock,
        "adr_add",
        vec![
            Hlvalue::Variable(src_chars),
            signed_offset(itemoffset_chars),
        ],
        src_items.clone(),
    );
    let src_scale = variable_with_lltype("src_scale", LowLevelType::Signed);
    push(
        &startblock,
        "int_mul",
        vec![
            constant_with_lltype(sizeof_char.clone(), LowLevelType::Signed),
            Hlvalue::Variable(srcstart),
        ],
        src_scale.clone(),
    );
    let asrc = variable_with_lltype("asrc", LowLevelType::Address);
    push(
        &startblock,
        "adr_add",
        vec![Hlvalue::Variable(src_items), Hlvalue::Variable(src_scale)],
        asrc.clone(),
    );

    let dst_adr = variable_with_lltype("adst", LowLevelType::Address);
    push(
        &startblock,
        "cast_ptr_to_adr",
        vec![Hlvalue::Variable(ptrdst)],
        dst_adr.clone(),
    );
    let adst = variable_with_lltype("adst", LowLevelType::Address);
    push(
        &startblock,
        "adr_add",
        vec![Hlvalue::Variable(dst_adr), signed_offset(itemoffset_dst)],
        adst.clone(),
    );
    let nbytes = variable_with_lltype("nbytes", LowLevelType::Signed);
    push(
        &startblock,
        "int_mul",
        vec![
            constant_with_lltype(sizeof_char, LowLevelType::Signed),
            Hlvalue::Variable(length),
        ],
        nbytes.clone(),
    );
    let copied = variable_with_lltype("copied", LowLevelType::Void);
    push(
        &startblock,
        "raw_memcopy",
        vec![
            Hlvalue::Variable(asrc),
            Hlvalue::Variable(adst),
            Hlvalue::Variable(nbytes),
        ],
        copied,
    );
    let kept = variable_with_lltype("kept", LowLevelType::Void);
    push(&startblock, "keepalive", vec![Hlvalue::Variable(src)], kept);
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::None,
                LowLevelType::Void,
            ))],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    func._always_inline_ = AlwaysInline::True;
    // `rstr.py copy_string_to_raw`: `@jit.oopspec('stroruni.copy_string_to_raw(src, ptrdst, srcstart, length)')`.
    func.oopspec = Some("stroruni.copy_string_to_raw(src, ptrdst, srcstart, length)".to_string());
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "src".to_string(),
            "ptrdst".to_string(),
            "srcstart".to_string(),
            "length".to_string(),
        ],
        func,
    ))
}

fn copy_string_to_raw_helper(rtyper: &RPythonTyper) -> Result<LowLevelFunction, TyperError> {
    rtyper.lowlevel_helper_function_with_builder(
        "copy_string_to_raw",
        vec![
            STRPTR.clone(),
            (*CCHARP).clone(),
            LowLevelType::Signed,
            LowLevelType::Signed,
        ],
        LowLevelType::Void,
        move |_rtyper, _args, _result| build_copy_string_to_raw_helper_graph("copy_string_to_raw"),
    )
}

/// Synthesise `str2charp(s, track_allocation=False)` (`rffi.py`):
///
/// ```python
/// array = lltype.malloc(CCHARP.TO, len(s) + 1, flavor='raw',
///                       track_allocation=False)
/// i = len(s)
/// copy_string_to_raw(llstr(s), array, 0, i)
/// array[i] = '\x00'
/// return array
/// ```
fn build_str2charp_helper_graph(rtyper: &RPythonTyper, name: &str) -> Result<PyGraph, TyperError> {
    let chars_ptr_ty = chars_array_ptr_lltype_from_strptr(&STRPTR)?;
    let array_ty = ccharp_array_lltype()?;
    let raw_flags = raw_flavor_no_track_const()?;
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let copy_fn = copy_string_to_raw_helper(rtyper)?;
    let copy_ptr = helper_funcptr_hlvalue(rtyper, &copy_fn)?;
    let push = |block: &BlockRef, op: &str, args: Vec<Hlvalue>, result: Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            op,
            args,
            Hlvalue::Variable(result),
        ));
    };

    let s = variable_with_lltype("s", STRPTR.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(s.clone())]);
    let return_var = variable_with_lltype("result", (*CCHARP).clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let chars = variable_with_lltype("chars", chars_ptr_ty);
    push(
        &startblock,
        "getsubstruct",
        vec![
            Hlvalue::Variable(s.clone()),
            constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void),
        ],
        chars.clone(),
    );
    let length = variable_with_lltype("length", LowLevelType::Signed);
    push(
        &startblock,
        "getarraysize",
        vec![Hlvalue::Variable(chars)],
        length.clone(),
    );
    let size = variable_with_lltype("size", LowLevelType::Signed);
    push(
        &startblock,
        "int_add",
        vec![Hlvalue::Variable(length.clone()), signed_const(1)],
        size.clone(),
    );
    let array = variable_with_lltype("array", (*CCHARP).clone());
    push(
        &startblock,
        "malloc_varsize",
        vec![
            lowlevel_type_const(array_ty),
            raw_flags,
            Hlvalue::Variable(size),
        ],
        array.clone(),
    );
    let copied = variable_with_lltype("copied", LowLevelType::Void);
    push(
        &startblock,
        "direct_call",
        vec![
            copy_ptr,
            Hlvalue::Variable(s),
            Hlvalue::Variable(array.clone()),
            signed_const(0),
            Hlvalue::Variable(length.clone()),
        ],
        copied,
    );
    let set_nul = variable_with_lltype("set_nul", LowLevelType::Void);
    push(
        &startblock,
        "setarrayitem",
        vec![
            Hlvalue::Variable(array.clone()),
            Hlvalue::Variable(length),
            constant_with_lltype(ConstValue::ByteStr(vec![0]), LowLevelType::Char),
        ],
        set_nul,
    );
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(array)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
}

/// Synthesise `_get_raw_address_buf_from_string` (`rffi.py`):
///
/// ```python
/// key = objectmodel.compute_unique_id(string)
/// try:
///     ptr = _fq_addr_from_string.raw_copies[key]
/// except KeyError:
///     ptr = str2charp(string, track_allocation=False)
///     _fq_addr_from_string.raw_copies[key] = ptr
///     ll_string = llstr(string)
///     gcptr = lltype.cast_opaque_ptr(llmemory.GCREF, ll_string)
///     _fq_addr_from_string.register_finalizer(gcptr)
/// return ptr
/// ```
///
/// `compute_unique_id` is llop `gc_id`. The dict getitem is
/// `ll_dict_getitem` with the `c_last_exception` KeyError catch
/// `rtype_getitem` records (`has_implicit_exception(KeyError)` /
/// `exception_is_here`). `register_finalizer` is llop `gc_fq_register`
/// (`rgc.py FinalizerQueue.register_finalizer`); the tag is the
/// `CDefinedIntSymbolic` `FqTagEntry.specialize_call` yields
/// (`framework.py get_finalizer_queue_index` reads `.default`).
///
/// Returns the outer graph plus every callee graph hanging off its
/// `direct_call` funcptrs (`call.py` follows `funcobj.graph`).
pub(crate) fn build_get_raw_address_buf_from_string_family(
    name: &str,
) -> Result<(PyGraph, Vec<(String, GraphRef)>), TyperError> {
    let ann = RPythonAnnotator::new(None, None, None, false);
    let rtyper = Rc::new(RPythonTyper::new(&ann));
    rtyper.initialize_exceptiondata()?;
    let repr = signed_ccharp_dict_repr(Rc::clone(&rtyper))?;
    let raw_copies = convert_fq_raw_copies_const(&repr)?;
    let getitem_fn = repr.ll_dict_getitem_helper_on(&rtyper)?;
    let getitem = helper_funcptr_hlvalue(&rtyper, &getitem_fn)?;
    let setitem_fn = repr.ll_dict_setitem_helper_on(&rtyper)?;
    let setitem = helper_funcptr_hlvalue(&rtyper, &setitem_fn)?;
    let str2charp_fn = rtyper.lowlevel_helper_function_with_builder(
        "str2charp",
        vec![STRPTR.clone()],
        (*CCHARP).clone(),
        move |rtyper, _args, _result| build_str2charp_helper_graph(rtyper, "str2charp"),
    )?;
    let str2charp = helper_funcptr_hlvalue(&rtyper, &str2charp_fn)?;
    let keyerror = exception_args("KeyError")?;
    let fq_tag = constant_with_lltype(
        ConstValue::SpecTag(fq_addr_from_string_tag_id()),
        LowLevelType::Signed,
    );

    let push = |block: &BlockRef, op: &str, args: Vec<Hlvalue>, result: Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            op,
            args,
            Hlvalue::Variable(result),
        ));
    };

    let s = variable_with_lltype("s", STRPTR.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(s.clone())]);
    let return_var = variable_with_lltype("result", (*CCHARP).clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let s_miss = variable_with_lltype("s", STRPTR.clone());
    let key_miss = variable_with_lltype("key", LowLevelType::Signed);
    let missblock = Block::shared(vec![
        Hlvalue::Variable(s_miss.clone()),
        Hlvalue::Variable(key_miss.clone()),
    ]);

    let key = variable_with_lltype("key", LowLevelType::Signed);
    push(
        &startblock,
        "gc_id",
        vec![Hlvalue::Variable(s.clone())],
        key.clone(),
    );
    let ptr_hit = variable_with_lltype("ptr", (*CCHARP).clone());
    push(
        &startblock,
        "direct_call",
        vec![getitem, raw_copies.clone(), Hlvalue::Variable(key.clone())],
        ptr_hit.clone(),
    );
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Constant(c_last_exception()));
    let hit_link = Link::new(
        vec![Hlvalue::Variable(ptr_hit)],
        Some(graph.returnblock.clone()),
        None,
    )
    .into_ref();
    let mut miss_link = Link::new(
        vec![Hlvalue::Variable(s), Hlvalue::Variable(key)],
        Some(missblock.clone()),
        Some(keyerror[0].clone()),
    );
    miss_link.extravars(Some(keyerror[0].clone()), Some(keyerror[1].clone()));
    startblock.closeblock(vec![hit_link, miss_link.into_ref()]);

    let ptr_miss = variable_with_lltype("ptr", (*CCHARP).clone());
    push(
        &missblock,
        "direct_call",
        vec![str2charp, Hlvalue::Variable(s_miss.clone())],
        ptr_miss.clone(),
    );
    let set_void = variable_with_lltype("v", LowLevelType::Void);
    push(
        &missblock,
        "direct_call",
        vec![
            setitem,
            raw_copies,
            Hlvalue::Variable(key_miss),
            Hlvalue::Variable(ptr_miss.clone()),
        ],
        set_void,
    );
    let gcptr = variable_with_lltype("gcptr", (*GCREF).clone());
    push(
        &missblock,
        "cast_opaque_ptr",
        vec![Hlvalue::Variable(s_miss)],
        gcptr.clone(),
    );
    let fq_void = variable_with_lltype("v", LowLevelType::Void);
    push(
        &missblock,
        "gc_fq_register",
        vec![fq_tag, Hlvalue::Variable(gcptr)],
        fq_void,
    );
    missblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(ptr_miss)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    let callees = collect_callee_graphs(&rtyper, &graph);
    let py = helper_pygraph_from_graph(graph, vec!["s".to_string()], func);
    let _keep_ann = ann;
    let _keep_rtyper = rtyper;
    Ok((py, callees))
}

fn collect_callee_graphs(rtyper: &RPythonTyper, graph: &FunctionGraph) -> Vec<(String, GraphRef)> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    collect_callee_graphs_rec(rtyper, graph, &mut out, &mut seen);
    out
}

fn collect_callee_graphs_rec(
    rtyper: &RPythonTyper,
    graph: &FunctionGraph,
    out: &mut Vec<(String, GraphRef)>,
    seen: &mut HashSet<usize>,
) {
    for func in direct_call_funcobjs(graph) {
        let Some(graph_id) = func.graph else {
            continue;
        };
        if !seen.insert(graph_id) {
            continue;
        }
        let Some(gref) = graph_ref_for_funcptr_id(rtyper, graph_id) else {
            continue;
        };
        out.push((func._name.clone(), gref.clone()));
        collect_callee_graphs_rec(rtyper, &gref.borrow(), out, seen);
    }
}

/// RPython `_StrFinalizerQueue(rgc.FinalizerQueue)` (`rffi.py`).
/// `raw_copies` is `{id: CCHARP}` — each value is a raw `Ptr(Array(Char))`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct _StrFinalizerQueue {
    pub raw_copies: HashMap<usize, _ptr>,
}

/// RPython `_fq_addr_from_string = _StrFinalizerQueue()` (`rffi.py`).
pub static _fq_addr_from_string: LazyLock<Mutex<_StrFinalizerQueue>> =
    LazyLock::new(|| Mutex::new(_StrFinalizerQueue::default()));

#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(non_snake_case)]
pub struct CConstant {
    pub c_name: String,
    pub TP: LowLevelType,
}

impl CConstant {
    pub fn new(c_name: impl Into<String>, TP: LowLevelType) -> Self {
        CConstant {
            c_name: c_name.into(),
            TP,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ptr_array_of(value: &LowLevelType) -> &Array {
        let LowLevelType::Ptr(ptr) = value else {
            panic!("expected Ptr(Array), got {value:?}");
        };
        let PtrTarget::Array(array) = &ptr.TO else {
            panic!("expected Ptr(Array), got {value:?}");
        };
        array
    }

    #[test]
    fn errno_flags_match_upstream_bit_layout() {
        assert_eq!(RFFI_FULL_ERRNO, 3);
        assert_eq!(RFFI_FULL_ERRNO_ZERO, 5);
        assert_eq!(RFFI_FULL_LASTERROR, 24);
        assert_eq!(RFFI_ERR_ALL, 27);
        assert_eq!(RFFI_ALT_ERRNO, 64);
    }

    #[test]
    fn carray_builds_nolength_raw_array() {
        let array = CArray(LowLevelType::Char);
        assert_eq!(array.OF, LowLevelType::Char);
        assert_eq!(array._hints.get("nolength"), Some(&ConstValue::Bool(true)));
    }

    #[test]
    fn size_and_sign_raw_ptr_is_word_signed() {
        // `rffi.py size_and_sign`: a `Ptr` is `(LONG_BIT/8, unsigned=False)`.
        let word = crate::layout::target_word_size();
        assert_eq!(size_and_sign(&*CCHARP).unwrap(), (word, false));
        assert_eq!(
            size_and_sign(&CArrayPtr(LowLevelType::Signed)).unwrap(),
            (word, false)
        );
        assert_eq!(size_and_sign(&LowLevelType::Signed).unwrap(), (word, false));
        assert_eq!(size_and_sign(&LowLevelType::Char).unwrap(), (1, true));
        assert_eq!(
            size_and_sign(&LowLevelType::Unsigned).unwrap(),
            (word, true)
        );
        assert_eq!(
            size_and_sign(&LowLevelType::SignedLongLong).unwrap(),
            (8, false)
        );
        let four_signed =
            LowLevelType::FixedSizeArray(Box::new(CFixedArray(LowLevelType::Signed, 4)));
        assert_eq!(
            sizeof(&four_signed).unwrap(),
            ConstValue::Int((4 * word) as i64)
        );
        assert_eq!(size_and_sign(&four_signed).unwrap(), (4 * word, false));
        assert!(sizeof(&LowLevelType::Void).is_err());
        assert!(size_and_sign(&LowLevelType::Void).is_err());
    }

    #[test]
    fn sizeof_unichar_is_sizeof_wchar_t() {
        assert_eq!(
            sizeof(&LowLevelType::UniChar).unwrap(),
            ConstValue::Int(llmemory::sizeof_wchar_t())
        );
        assert_eq!(
            size_and_sign(&LowLevelType::UniChar).unwrap(),
            (llmemory::sizeof_wchar_t() as usize, true)
        );
    }

    #[test]
    fn sizeof_unprobed_struct_is_symbolic_item_offset() {
        let s = LowLevelType::Struct(Box::new(Struct::new(
            "S",
            vec![("f".to_string(), LowLevelType::Signed)],
        )));
        let expected = llmemory::sizeof(&s, None).unwrap();
        assert_eq!(sizeof(&s).unwrap(), expected);
        assert!(matches!(expected, ConstValue::AddressOffset(_)));
        assert!(size_and_sign(&s).is_err());

        let arr = LowLevelType::FixedSizeArray(Box::new(CFixedArray(s.clone(), 3)));
        let ConstValue::AddressOffset(base) = expected else {
            panic!("llmemory::sizeof(S, None) is AddressOffset");
        };
        let scaled = base.mul(3).expect("ItemOffset.__mul__ scales repeat");
        assert_eq!(sizeof(&arr).unwrap(), ConstValue::AddressOffset(scaled));
    }

    #[test]
    fn sizeof_struct_with_size_hint_is_int() {
        let hinted = LowLevelType::Struct(Box::new(Struct::with_hints(
            "Hinted",
            vec![("f".to_string(), LowLevelType::Signed)],
            vec![("size".to_string(), ConstValue::Int(24))],
        )));
        assert_eq!(sizeof(&hinted).unwrap(), ConstValue::Int(24));
        assert_eq!(size_and_sign(&hinted).unwrap(), (24, false));
    }

    #[test]
    fn functype_and_llptr_predicates_match_upstream_shape() {
        let callback = CCallback(vec![LowLevelType::Signed], LowLevelType::Void);
        assert!(_isfunctype(&callback));
        assert!(!_isfunctype(&LowLevelType::Signed));

        let ptr = llexternal("demo", vec![], LowLevelType::Void);
        assert!(_isllptr(&ptr));
    }

    #[test]
    fn inttype_population_preserves_rffi_names() {
        let names = populate_inttypes();
        assert!(names.contains(&"short".to_string()));
        assert!(names.contains(&"ushort".to_string()));
        assert!(names.contains(&"int".to_string()));
        assert!(names.contains(&"uint".to_string()));
        assert!(names.contains(&"void*".to_string()));
        assert!(NUMBER_TYPES.len() >= names.len());
        assert_eq!(r_int_real, LowLevelType::Signed);
        assert_eq!(r_uint_real, LowLevelType::Unsigned);
        assert_eq!(r_singlefloat, LowLevelType::SingleFloat);
        assert_eq!(NULL, None);
    }

    #[test]
    fn keepalive_cache_reuses_freed_slots_per_type() {
        let tp = LowLevelType::Signed;
        let pos = register_keepalive(tp.clone(), ConstValue::Int(41));
        assert_eq!(
            get_keepalive_object(pos, tp.clone()),
            Some(ConstValue::Int(41))
        );

        unregister_keepalive(pos, tp.clone());
        assert_eq!(get_keepalive_object(pos, tp.clone()), None);

        let reused = register_keepalive(tp.clone(), ConstValue::Int(42));
        assert_eq!(reused, pos);
        assert_eq!(get_keepalive_object(pos, tp), Some(ConstValue::Int(42)));
    }

    #[test]
    fn deferred_runtime_helpers_name_the_original_surface() {
        let err = make_string_mappings().expect_err("string runtime is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("make_string_mappings"));

        let err = generate_macro_wrapper().expect_err("macro wrapper is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("generate_macro_wrapper"));

        let err = ll_liststr2charpp().expect_err("char** runtime is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_liststr2charpp"));

        let err = c_memcpy().expect_err("native memcpy binding is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("c_memcpy"));

        let _scoped = scoped_alloc_buffer;
        let _finalizer = _StrFinalizerQueue::default();
    }

    #[test]
    fn build_ll_get_raw_address_of_string_emits_can_move_then_address_arithmetic() {
        let helper =
            build_ll_get_raw_address_of_string_helper_graph("ll_get_raw_address_of_string")
                .expect("build_ll_get_raw_address_of_string_helper_graph");
        assert_eq!(helper.func.name, "ll_get_raw_address_of_string");
        assert_eq!(helper.func._jit_look_inside_, Some(false));
        let inner = helper.graph.borrow();
        assert_eq!(inner.name, "ll_get_raw_address_of_string");
        assert_eq!(
            inner.func.as_ref().and_then(|f| f._jit_look_inside_),
            Some(false)
        );
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 1);
        let Hlvalue::Variable(v) = &startblock.inputargs[0] else {
            panic!("startblock input must be a Variable");
        };
        assert!(
            v.name().starts_with('s'),
            "input variable name = {:?}",
            v.name()
        );
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["gc_can_move"]);
        drop(startblock);

        let mut opnames: Vec<String> = Vec::new();
        for block in inner.iterblocks() {
            for op in &block.borrow().operations {
                opnames.push(op.opname.clone());
            }
        }
        assert_eq!(
            opnames
                .iter()
                .filter(|n| n.as_str() == "gc_can_move")
                .count(),
            2,
            "second gc_can_move after move_out_of_nursery missing: {opnames:?}"
        );
        assert!(
            opnames.iter().any(|n| n == "gc_move_out_of_nursery"),
            "gc_move_out_of_nursery missing: {opnames:?}"
        );

        let moveblock = inner
            .iterblocks()
            .into_iter()
            .find(|block| {
                block
                    .borrow()
                    .operations
                    .iter()
                    .any(|op| op.opname == "gc_move_out_of_nursery")
            })
            .expect("moveblock with gc_move_out_of_nursery");
        {
            let mb = moveblock.borrow();
            let move_ops: Vec<&str> = mb.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(
                move_ops,
                vec!["gc_move_out_of_nursery", "gc_can_move"],
                "moveblock must keep the second gc_can_move: {move_ops:?}"
            );
            let mut saw_buf = false;
            let mut saw_addr = false;
            let mut saw_except = false;
            for link in &mb.exits {
                let lb = link.borrow();
                let target = lb.target.as_ref().expect("link.target");
                if std::rc::Rc::ptr_eq(target, &inner.exceptblock) {
                    saw_except = true;
                }
                let exitcase_is_true = matches!(
                    lb.exitcase,
                    Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)
                );
                if exitcase_is_true {
                    saw_buf = target
                        .borrow()
                        .operations
                        .iter()
                        .any(|op| op.opname == "direct_call");
                } else {
                    saw_addr = target
                        .borrow()
                        .operations
                        .iter()
                        .any(|op| op.opname == "cast_ptr_to_adr");
                }
            }
            assert!(
                !saw_except,
                "second gc_can_move True arm must not raise AssertionError"
            );
            assert!(
                saw_buf,
                "second gc_can_move True arm must direct_call _get_raw_address_buf_from_string"
            );
            assert!(
                saw_addr,
                "second gc_can_move False arm must take the address-arithmetic path"
            );
        }

        assert!(
            opnames.iter().any(|n| n == "cast_ptr_to_adr"),
            "cast_ptr_to_adr missing: {opnames:?}"
        );
        assert_eq!(
            opnames.iter().filter(|n| n.as_str() == "adr_add").count(),
            2,
            "offsetof(STR, chars) + itemoffsetof(STR.chars, 0): {opnames:?}"
        );
        assert!(
            opnames.iter().any(|n| n == "cast_adr_to_ptr"),
            "cast_adr_to_ptr missing: {opnames:?}"
        );
        assert!(
            opnames.iter().any(|n| n == "direct_call"),
            "buf helper direct_call missing: {opnames:?}"
        );
        assert!(
            opnames.iter().any(|n| n == "setarrayitem"),
            "NUL setarrayitem missing: {opnames:?}"
        );

        let addrblock = inner
            .iterblocks()
            .into_iter()
            .find(|block| {
                block
                    .borrow()
                    .operations
                    .iter()
                    .any(|op| op.opname == "cast_ptr_to_adr")
            })
            .expect("addrblock with cast_ptr_to_adr");
        {
            let ab = addrblock.borrow();
            let addr_ops: Vec<&str> = ab.operations.iter().map(|op| op.opname.as_str()).collect();
            assert_eq!(
                addr_ops,
                vec![
                    "cast_ptr_to_adr",
                    "adr_add",
                    "adr_add",
                    "cast_adr_to_ptr",
                    "getsubstruct",
                    "getarraysize",
                    "setarrayitem",
                ],
                "addrblock must write NUL via setarrayitem on STR.chars: {addr_ops:?}"
            );
            let set = ab
                .operations
                .iter()
                .find(|op| op.opname == "setarrayitem")
                .expect("setarrayitem");
            assert_eq!(set.args.len(), 3);
            match &set.args[2] {
                Hlvalue::Constant(c) => {
                    assert_eq!(c.value, ConstValue::ByteStr(vec![0]));
                    assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Char));
                }
                other => panic!("NUL must be a Char constant, got {other:?}"),
            }
        }

        let mut saw_except_link = false;
        for block in inner.iterblocks() {
            for link in &block.borrow().exits {
                if let Some(target) = &link.borrow().target
                    && std::rc::Rc::ptr_eq(target, &inner.exceptblock)
                {
                    saw_except_link = true;
                }
            }
        }
        assert!(
            !saw_except_link,
            "helper graph must not link to exceptblock"
        );

        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some((*CCHARP).clone()),
            "get_raw_address_of_string returns CCHARP"
        );
    }

    #[test]
    fn pointer_aliases_have_rffi_array_hints() {
        let voidp = ptr_array_of(&VOIDP);
        assert_eq!(voidp.OF, LowLevelType::Char);
        assert_eq!(voidp._hints.get("nolength"), Some(&ConstValue::Bool(true)));
        assert_eq!(
            voidp._hints.get("render_as_void"),
            Some(&ConstValue::Bool(true))
        );

        let const_charp = ptr_array_of(&CONST_CCHARP);
        assert_eq!(const_charp.OF, LowLevelType::Char);
        assert_eq!(
            const_charp._hints.get("render_as_const"),
            Some(&ConstValue::Bool(true))
        );

        let charpp = ptr_array_of(&CCHARPP);
        assert_eq!(charpp.OF, (*CCHARP).clone());
    }

    #[test]
    fn cstruct_prefixes_fields_and_records_c_hints() {
        let c_struct = CStruct(
            "demo",
            vec![("x".into(), LowLevelType::Signed), ("y".into(), DOUBLE)],
        );
        assert_eq!(c_struct._names, vec!["c_x", "c_y"]);
        assert_eq!(
            c_struct._hints.get("external"),
            Some(&ConstValue::byte_str("C"))
        );
        assert_eq!(
            c_struct._hints.get("c_name"),
            Some(&ConstValue::byte_str("demo"))
        );
    }

    #[test]
    fn make_mallocs_raw_struct_and_populates_fields() {
        let s = Struct::new("point", vec![("x".into(), LowLevelType::Signed)]);
        let ptr = make(s, vec![("x".into(), LowLevelValue::Signed(7))])
            .expect("rffi.make should allocate a raw struct pointer");

        assert_eq!(ptr.getattr("x").unwrap(), LowLevelValue::Signed(7));
    }

    #[test]
    fn ccallback_returns_ptr_to_functype() {
        let callback = CCallback(vec![SIGNED], LowLevelType::Void);
        let LowLevelType::Ptr(ptr) = callback else {
            panic!("expected Ptr(Func)");
        };
        let PtrTarget::Func(func_t) = ptr.TO else {
            panic!("expected Ptr(Func)");
        };
        assert_eq!(func_t.args, vec![LowLevelType::Signed]);
        assert_eq!(func_t.result, LowLevelType::Void);
    }

    fn funcptr_name_and_graph(hlv: &Hlvalue) -> Option<(String, Option<usize>)> {
        use crate::translator::rtyper::lltypesystem::lltype::_ptr_obj;
        let Hlvalue::Constant(c) = hlv else {
            return None;
        };
        let ConstValue::LLPtr(ptr) = &c.value else {
            return None;
        };
        match ptr._obj0_value() {
            Ok(Some(_ptr_obj::Func(func))) => Some((func._name.clone(), func.graph)),
            _ => None,
        }
    }

    #[test]
    fn copy_string_to_raw_emits_minimark_raw_memcopy_and_keepalive() {
        let helper = build_copy_string_to_raw_helper_graph("copy_string_to_raw")
            .expect("build_copy_string_to_raw_helper_graph");
        assert_eq!(
            helper.func._always_inline_,
            crate::flowspace::model::AlwaysInline::True
        );
        let inner = helper.graph.borrow();
        let opnames: Vec<String> = inner
            .iterblocks()
            .into_iter()
            .flat_map(|block| {
                block
                    .borrow()
                    .operations
                    .iter()
                    .map(|op| op.opname.clone())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(
            opnames.iter().any(|n| n == "int_ge"),
            "assert length >= 0: {opnames:?}"
        );
        assert!(
            opnames.iter().any(|n| n == "debug_assert"),
            "assert length >= 0: {opnames:?}"
        );
        assert!(
            opnames.iter().any(|n| n == "cast_ptr_to_adr"),
            "cast_ptr_to_adr: {opnames:?}"
        );
        assert!(
            opnames.iter().filter(|n| n.as_str() == "adr_add").count() >= 3,
            "offsetof+itemoffsetof+sizeof*srcstart and dst itemoffsetof: {opnames:?}"
        );
        assert!(
            opnames.iter().any(|n| n == "raw_memcopy"),
            "raw_memcopy: {opnames:?}"
        );
        assert!(
            opnames.iter().any(|n| n == "keepalive"),
            "keepalive: {opnames:?}"
        );
    }

    #[test]
    fn str2charp_direct_calls_copy_string_to_raw() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let helper = build_str2charp_helper_graph(&rtyper, "str2charp").expect("str2charp");
        let inner = helper.graph.borrow();
        let mut saw_copy = false;
        for block in inner.iterblocks() {
            for op in &block.borrow().operations {
                if op.opname != "direct_call" {
                    continue;
                }
                let Some((name, graph)) = funcptr_name_and_graph(&op.args[0]) else {
                    continue;
                };
                if name == "copy_string_to_raw" {
                    assert!(graph.is_some(), "copy_string_to_raw funcptr carries graph");
                    saw_copy = true;
                }
            }
        }
        assert!(saw_copy, "str2charp must direct_call copy_string_to_raw");
        let _keep = (ann, rtyper);
    }

    #[test]
    fn convert_fq_raw_copies_const_preserves_nonempty_ccharp_values() {
        let array_ty = ccharp_array_lltype().expect("CCHARP.TO");
        let ptr = malloc(array_ty, Some(4), MallocFlavor::Raw, true).expect("raw CCHARP");
        let key = 0xC0FFEE_usize;
        {
            let mut fq = _fq_addr_from_string.lock();
            fq.raw_copies.insert(key, ptr.clone());
        }
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let repr = signed_ccharp_dict_repr(Rc::clone(&rtyper)).expect("signed CCHARP dict");
        let converted = convert_fq_raw_copies_const(&repr).expect("convert nonempty raw_copies");
        {
            let mut fq = _fq_addr_from_string.lock();
            fq.raw_copies.remove(&key);
        }
        let Hlvalue::Constant(c) = converted else {
            panic!("convert_fq_raw_copies_const must yield a Constant");
        };
        let ConstValue::LLPtr(dict_ptr) = &c.value else {
            panic!("converted dict must be LLPtr, got {:?}", c.value);
        };
        let live = match dict_ptr.getattr("num_live_items").expect("num_live_items") {
            LowLevelValue::Signed(n) => n,
            other => panic!("num_live_items must be Signed, got {other:?}"),
        };
        assert_eq!(live, 1, "nonempty cache inserts one entry");
        let entries = dict_ptr.getattr("entries").expect("entries");
        let LowLevelValue::Ptr(entries) = entries else {
            panic!("entries must be a pointer, got {entries:?}");
        };
        let entry = entries.getitem(0).expect("entries[0]");
        let (entry_key, entry_value) = match entry {
            LowLevelValue::InteriorPtr(ip) => (
                ip.getattr("key").expect("key"),
                ip.getattr("value").expect("value"),
            ),
            LowLevelValue::Ptr(p) => (
                p.getattr("key").expect("key"),
                p.getattr("value").expect("value"),
            ),
            other => panic!("entry must be a struct pointer, got {other:?}"),
        };
        assert!(
            matches!(entry_key, LowLevelValue::Signed(k) if k == key as i64),
            "key must be {key}, got {entry_key:?}"
        );
        let LowLevelValue::Ptr(value_ptr) = entry_value else {
            panic!("CCHARP value must be a pointer, got {entry_value:?}");
        };
        assert_eq!(
            value_ptr._hashable_identity(),
            ptr._hashable_identity(),
            "converted dict must hold the stored CCHARP"
        );
    }
}
