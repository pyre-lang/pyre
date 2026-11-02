//! RPython `rpython/rtyper/lltypesystem/rffi.py`.
//!
//! This slice exposes the exact lltype-facing surface that can be represented
//! with pyre's current `lltype` port: errno flags, C array/struct helper
//! constructors, callback pointer types, the standard raw pointer aliases,
//! and the platform C integer `Number` types created by `setup()`.
//! The program-level half of `rffi.py` (string and buffer helpers, `llexternal`
//! wrappers and their callbacks) is program code here: it lives in
//! `majit_rlib::rffi` and `majit_rlib::lltypesystem::rffi`, which the
//! translator reads as LLBC like the rest of the program.
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use indexmap::IndexMap;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::rc::Rc;
use std::sync::{Arc, LazyLock, OnceLock};

use crate::annotator::annrpython::RPythonAnnotator;
use crate::annotator::dictdef::DictDef;
use crate::annotator::model::{KnownType, SomeInteger, SomeValue};
use crate::flowspace::model::{
    AlwaysInline, Block, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc,
    GraphRef, Hlvalue, Link, SpaceOperation, Variable, c_last_exception,
};
use crate::flowspace::pygraph::PyGraph;
use crate::rlib::rarithmetic;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::llmemory;
use crate::translator::rtyper::lltypesystem::lltype::{
    _address, _ptr, _ptr_obj, Array, CastWhateverError, FixedSizeArray, FuncType, GCREF,
    LowLevelType, LowLevelValue, MallocFlavor, Number, Ptr, PtrTarget, SomePtr, Struct,
    build_number, cast_primitive, const_as_f64, functionptr_with_external_name, malloc, nullptr,
    truncated_from_const, typeOf, typeOf_const,
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
use crate::translator::rtyper::tool::rfficache::{
    PLATFORM, RffiCacheError, sizeof_c_type, sizeof_c_types,
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

/// RPython `TYPES` seed list (`rffi.py`). POSIX typedefs follow
/// `os.name != 'nt'` on the *target* (`layout::target_os`), not the host
/// `#[cfg]`. `wasm32-unknown-unknown` is `unknown` and keeps only builtin
/// C types plus pointer-width typedefs the rustc probe can answer.
pub static TYPES: LazyLock<Vec<&'static str>> = LazyLock::new(types_for_target);

fn types_for_target() -> Vec<&'static str> {
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
    // `rffi.py`: `try: sizeof_c_type('__int128_t', ignore_errors=True);
    // TYPES += ['__int128_t', '__uint128_t'] except CompilationError: pass`
    // before the POSIX `os.name != 'nt'` block. The rustc probe has no
    // `__int128_t` mapping, so a cross target skips both names.
    if sizeof_c_type("__int128_t", true).is_ok() {
        types.extend(["__int128_t", "__uint128_t"]);
    }
    let os = crate::layout::target_os();
    if target_os_is_unix(os) {
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
    } else if os == "unknown" {
        types.retain(|name| !matches!(*name, "time_t" | "wchar_t"));
    }
    types
}

fn target_os_is_unix(os: &str) -> bool {
    !matches!(os, "windows" | "unknown" | "none" | "uefi")
}

fn types_contains(c_name: &str) -> bool {
    TYPES.iter().any(|name| *name == c_name)
}

fn rffi_name_from_c_type(mut name: &str) -> String {
    if let Some(rest) = name.strip_prefix("unsigned") {
        name = rest.trim_start();
        format!("u{}", name.replace(' ', ""))
    } else {
        name.replace(' ', "")
    }
}

/// Signedness from `populate_inttypes` (`rffi.py`): `'unsigned'` prefix,
/// `size_t` / `uint*` / `__uint*` unsigned; `wchar_t` unsigned on Windows;
/// everything else signed (`void*` included).
fn rffi_inttype_signed(c_name: &str) -> bool {
    if c_name.starts_with("unsigned") {
        false
    } else if c_name == "size_t" || c_name.starts_with("uint") || c_name.starts_with("__uint") {
        false
    } else if c_name == "wchar_t" && crate::layout::target_os() == "windows" {
        false
    } else {
        true
    }
}

fn populate_platform_inttypes() -> Result<(), RffiCacheError> {
    let mut missing_c = Vec::new();
    let mut missing_name = Vec::new();
    let mut missing_signed = Vec::new();
    {
        let platform = PLATFORM.lock();
        for c_name in TYPES.iter().copied() {
            let name = rffi_name_from_c_type(c_name).to_uppercase();
            if platform.get(&name).is_none() {
                missing_signed.push(rffi_inttype_signed(c_name));
                missing_c.push(c_name);
                missing_name.push(name);
            }
        }
    }
    if missing_c.is_empty() {
        return Ok(());
    }
    let sizes = sizeof_c_types(&missing_c, false)?;
    let mut platform = PLATFORM.lock();
    for ((name, signed), size) in missing_name
        .iter()
        .zip(missing_signed)
        .zip(sizes.into_iter())
    {
        if platform.get(name).is_none() {
            platform.make_type(name, signed, size)?;
        }
    }
    Ok(())
}

/// RPython `populate_inttypes()` (`rffi.py`).
pub fn populate_inttypes() -> Vec<String> {
    if let Err(err) = populate_platform_inttypes() {
        panic!("{}", err);
    }
    TYPES
        .iter()
        .map(|name| rffi_name_from_c_type(name))
        .collect()
}

fn nolength_hints() -> Vec<(String, ConstValue)> {
    vec![("nolength".into(), ConstValue::Bool(true))]
}

/// One `setup()` row: `NAME`, `NAMEP`, and `r_<name>` (`rffi.py` `setup`).
///
/// `RffiGlobal` Deref needs `&'static LowLevelType` handles, so published
/// types stay interned statics. `intern_static` leaks a value only the
/// first time it is seen; repeated `setup()` calls on an unchanged
/// platform leak nothing, and the total leak is bounded by the number of
/// distinct published values. `rclass` is `KnownType` (Copy) and stored
/// by value.
#[derive(Copy, Clone)]
struct PublishedCInt {
    tp: &'static LowLevelType,
    tpp: &'static LowLevelType,
    rclass: KnownType,
}

struct Published {
    result: Vec<LowLevelType>,
    by_key: IndexMap<String, PublishedCInt>,
    /// `rffi.py` `os.name == 'nt'`: `PID_T`/`SSIZE_T`/`PTRDIFF_T` at
    /// `TYPES` construction and `MODE_T = INT` after `setup()`. Bare
    /// `NAME` aliases; not in `TYPES`, so no `NAMEP`/`r_*`.
    aliases: IndexMap<String, &'static LowLevelType>,
}

/// Module globals last written by `setup()` (`rffi.py` `globals()[...] =`).
static PUBLISHED: RwLock<Option<Published>> = RwLock::new(None);

/// `rffi.py` module initialization: `NUMBER_TYPES = setup()`, then the
/// `Signed→int` override and the `r_int_real` / `INT_real` /
/// `r_uint_real` / `UINT_real` registrations. Every public number-type
/// export forces this once.
static RFFI_MODULE: OnceLock<RffiModule> = OnceLock::new();

struct RffiModule {
    number_types: Vec<LowLevelType>,
    r_int_real: KnownType,
    int_real: LowLevelType,
    r_uint_real: KnownType,
    uint_real: LowLevelType,
    /// `globals()['r_long']` from the first `setup()` (`rffi.py`), taken
    /// before `numbertype_to_rclass[Signed] = int`. A later `setup()` may
    /// republish `r_long` as builtin `int` when `LONG` is `Signed`.
    #[allow(dead_code)]
    first_r_long: KnownType,
}

/// Process-global intern table for published `LowLevelType` values.
/// `setup` (`rffi.py`) rebinds module globals onto the platform's existing
/// type objects; this table is the Rust analogue that keeps `&'static`
/// identity without leaking on every `setup()`.
static INTERNED: LazyLock<Mutex<HashMap<LowLevelType, &'static LowLevelType>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Intern `tp` as `&'static LowLevelType`. Leaks only the first time a
/// given value is seen; later calls return the same pointer.
fn intern_static(tp: LowLevelType) -> &'static LowLevelType {
    let mut interned = INTERNED.lock();
    if let Some(&existing) = interned.get(&tp) {
        return existing;
    }
    let leaked: &'static LowLevelType = Box::leak(Box::new(tp.clone()));
    interned.insert(tp, leaked);
    leaked
}

fn insert_published(
    by_key: &mut IndexMap<String, PublishedCInt>,
    key: String,
    tp: LowLevelType,
    rclass: KnownType,
) -> LowLevelType {
    let tpp = CArrayPtr(tp.clone());
    let published = PublishedCInt {
        tp: intern_static(tp.clone()),
        tpp: intern_static(tpp),
        rclass,
    };
    by_key.insert(key, published);
    tp
}

/// `rffi.py` `os.name == 'nt'` `TYPES` else-branch (`PID_T` / `SSIZE_T` /
/// `PTRDIFF_T = lltype.Signed`) and later `MODE_T = INT`. Bare `NAME`
/// aliases only: the names are not in `TYPES` and get no `NAMEP` / `r_*`.
fn nt_fallback_type_aliases(
    types: &[&str],
    int_tp: LowLevelType,
) -> IndexMap<String, LowLevelType> {
    let mut aliases = IndexMap::new();
    if !types.iter().any(|name| *name == "pid_t") {
        aliases.insert("PID_T".into(), SIGNED);
    }
    if !types.iter().any(|name| *name == "ssize_t") {
        aliases.insert("SSIZE_T".into(), SIGNED);
    }
    if !types.iter().any(|name| *name == "ptrdiff_t") {
        aliases.insert("PTRDIFF_T".into(), SIGNED);
    }
    if !types.iter().any(|name| *name == "mode_t") {
        aliases.insert("MODE_T".into(), int_tp);
    }
    aliases
}

/// RPython `setup()` body (`rffi.py`): for every `populate_inttypes()`
/// name, bind `r_<name>`, `NAME`, and `NAMEP`. `os.name == 'nt'`
/// aliases from [`nt_fallback_type_aliases`] stay out of that family.
fn setup_body() -> Published {
    let names = populate_inttypes();
    let platform = PLATFORM.lock();
    let mut result = Vec::with_capacity(names.len());
    let mut by_key = IndexMap::with_capacity(names.len());
    for name in names {
        let key = name.to_uppercase();
        let tp = match platform.get(&key) {
            Some(nt) => nt.lltype.clone(),
            None => panic!("rffi.setup: {key} missing from Platform.types"),
        };
        // `globals()['r_' + name] = platform.numbertype_to_rclass[tp]`
        let rclass = platform
            .numbertype_to_rclass(&tp)
            .unwrap_or_else(|| panic!("rffi.setup: r_{name} missing from numbertype_to_rclass"));
        result.push(insert_published(&mut by_key, key, tp, rclass));
    }
    drop(platform);
    let int_tp = (*by_key
        .get("INT")
        .unwrap_or_else(|| panic!("rffi.setup: INT missing from published table"))
        .tp)
        .clone();
    let aliases = nt_fallback_type_aliases(TYPES.as_slice(), int_tp)
        .into_iter()
        .map(|(key, tp)| (key, intern_static(tp)))
        .collect();
    Published {
        result,
        by_key,
        aliases,
    }
}

fn published_table_cint(key: &str) -> PublishedCInt {
    let guard = PUBLISHED.read();
    let table = guard
        .as_ref()
        .unwrap_or_else(|| panic!("rffi.setup: published table missing"));
    table
        .by_key
        .get(key)
        .copied()
        .unwrap_or_else(|| panic!("rffi.setup: {key} is not in TYPES for this target"))
}

fn published_table_lltype(key: &str) -> &'static LowLevelType {
    let guard = PUBLISHED.read();
    let table = guard
        .as_ref()
        .unwrap_or_else(|| panic!("rffi.setup: published table missing"));
    if let Some(entry) = table.by_key.get(key) {
        return entry.tp;
    }
    if let Some(&tp) = table.aliases.get(key) {
        return tp;
    }
    if let Some(stem) = key.strip_suffix('P')
        && let Some(entry) = table.by_key.get(stem)
    {
        return entry.tpp;
    }
    panic!("rffi.setup: {key} is not in TYPES for this target");
}

fn published_cint(key: &str) -> PublishedCInt {
    let _ = rffi_module();
    published_table_cint(key)
}

fn published_lltype_ref(key: &str) -> &'static LowLevelType {
    let _ = rffi_module();
    published_table_lltype(key)
}

#[cfg(test)]
fn published_lltype(key: &str) -> LowLevelType {
    published_lltype_ref(key).clone()
}

#[cfg(test)]
fn platform_lltype(name: &str) -> LowLevelType {
    published_lltype(name)
}

/// `globals()[NAME]` / `globals()[NAMEP]` from `setup()` (`rffi.py`).
/// Each read forces rffi module init, then looks the name up in the
/// table last written by `setup()`.
#[derive(Copy, Clone, Debug)]
pub struct RffiGlobal(&'static str);

impl Deref for RffiGlobal {
    type Target = LowLevelType;

    fn deref(&self) -> &LowLevelType {
        published_lltype_ref(self.0)
    }
}

/// `setup()` (`rffi.py`) without forcing `rffi_module` first.
/// `rffi_module_init` calls this so public [`setup`] cannot recurse.
fn setup_unforced() -> Vec<LowLevelType> {
    let published = setup_body();
    let result = published.result.clone();
    *PUBLISHED.write() = Some(published);
    result
}

/// RPython `setup()` (`rffi.py`). Recomputes and replaces the published
/// globals table (`globals()['r_' + name]`, `NAME`, `NAMEP`).
///
/// Upstream module init (`NUMBER_TYPES = setup()`, then
/// `numbertype_to_rclass[Signed] = int`, then `r_int_real` / `INT_real` /
/// `r_uint_real` / `UINT_real`) always completes before an external
/// caller can invoke `setup()`. Public `setup()` therefore forces
/// `rffi_module` first; init uses [`setup_unforced`].
pub fn setup() -> Vec<LowLevelType> {
    let _ = rffi_module();
    setup_unforced()
}

/// `platform.numbertype_to_rclass[tp]` after rffi module init (`rffi.py`).
pub fn numbertype_to_rclass(tp: &LowLevelType) -> KnownType {
    let _ = rffi_module();
    PLATFORM
        .lock()
        .numbertype_to_rclass(tp)
        .unwrap_or_else(|| panic!("rffi.setup: no rclass for {tp:?}"))
}

/// `globals()['r_' + name]` (`rffi.py` `setup`). `name` is the
/// lower-case rffi name (`"uchar"`, `"int_least8_t"`, `"__int128_t"`).
/// Forces rffi module init, then reads the rclass snapshot `setup()`
/// published, not the live `platform.numbertype_to_rclass` map. Panics
/// when that name is not in `TYPES` for this target.
pub fn r_class(name: &str) -> KnownType {
    published_cint(&name.to_uppercase()).rclass
}

macro_rules! rffi_r_class {
    ($($fn:ident => $name:expr;)*) => {
        $(
            pub fn $fn() -> KnownType {
                r_class($name)
            }
        )*
    };
}

// Named `r_<name>` accessors `setup()` binds (`rffi.py`). `r_int`
// shadows `rarithmetic.r_int` inside this module the way `rffi.r_int`
// does upstream.
rffi_r_class! {
    r_signedchar => "signedchar";
    r_uchar => "uchar";
    r_short => "short";
    r_ushort => "ushort";
    r_int => "int";
    r_uint => "uint";
    r_long => "long";
    r_ulong => "ulong";
    r_longlong => "longlong";
    r_ulonglong => "ulonglong";
    r_size_t => "size_t";
    r_time_t => "time_t";
    r_wchar_t => "wchar_t";
    r_uintptr_t => "uintptr_t";
    r_intptr_t => "intptr_t";
    r_mode_t => "mode_t";
    r_pid_t => "pid_t";
    r_ssize_t => "ssize_t";
    r_ptrdiff_t => "ptrdiff_t";
    r_int_least8_t => "int_least8_t";
    r_uint_least8_t => "uint_least8_t";
    r_int_least16_t => "int_least16_t";
    r_uint_least16_t => "uint_least16_t";
    r_int_least32_t => "int_least32_t";
    r_uint_least32_t => "uint_least32_t";
    r_int_least64_t => "int_least64_t";
    r_uint_least64_t => "uint_least64_t";
    r_int_fast8_t => "int_fast8_t";
    r_uint_fast8_t => "uint_fast8_t";
    r_int_fast16_t => "int_fast16_t";
    r_uint_fast16_t => "uint_fast16_t";
    r_int_fast32_t => "int_fast32_t";
    r_uint_fast32_t => "uint_fast32_t";
    r_int_fast64_t => "int_fast64_t";
    r_uint_fast64_t => "uint_fast64_t";
    r_intmax_t => "intmax_t";
    r_uintmax_t => "uintmax_t";
    // `r___int128_t` / `r___uint128_t` (`rffi.py` `setup`). Panic when
    // `TYPES` does not contain `__int128_t` / `__uint128_t`.
    r___int128_t => "__int128_t";
    r___uint128_t => "__uint128_t";
}

/// `SIGN`/`BITS` of the rclass behind a rffi C integer `Number`.
/// `LowLevelType::Number` carries `KnownType::RInt`; when `build_number`
/// folded the C type onto a predefined singleton, the `Platform`
/// `NumberType` record supplies the same pair (`rfficache._make_type`).
fn rffi_c_int_sign_bits(c_name: &str, tp: &LowLevelType) -> (bool, u16) {
    match tp {
        LowLevelType::Number(n) => (n._type.sign, n._type.bits),
        _ => match PLATFORM.lock().get(c_name) {
            Some(nt) => (nt.signed, nt.bits as u16),
            None => panic!("rffi.setup: {c_name} missing from Platform.types"),
        },
    }
}

fn build_int_real(name: &'static str, c_name: &str, c_tp: &LowLevelType) -> KnownType {
    let (sign, bits) = rffi_c_int_sign_bits(c_name, c_tp);
    rarithmetic::build_int(Some(name), sign, bits, true).unwrap_or_else(|e| panic!("{e}"))
}

/// `r_int_real` / `INT_real` (or the UINT pair) (`rffi.py`):
/// `rarithmetic.build_int`, `lltype.build_number`, then
/// `platform.numbertype_to_rclass[TP_real] = r_*_real`.
fn register_number_real(number_name: &str, r_name: &'static str) -> (KnownType, LowLevelType) {
    let published = published_table_cint(number_name);
    let r_real = build_int_real(r_name, number_name, published.tp);
    let tp = build_number(Some(number_name), r_real)
        .unwrap_or_else(|e| panic!("lltype.build_number({number_name}, {r_name}): {e}"));
    PLATFORM.lock().set_numbertype_to_rclass(tp.clone(), r_real);
    (r_real, tp)
}

/// RPython module body after `setup` (`rffi.py`):
/// `NUMBER_TYPES = setup()`;
/// `platform.numbertype_to_rclass[lltype.Signed] = int`;
/// `r_int_real` / `INT_real` then `NUMBER_TYPES.append(INT_real)`;
/// same for `r_uint_real` / `UINT_real`.
/// Calls [`setup_unforced`] so this cannot re-enter `rffi_module`
/// through public [`setup`]. Reads `INT`/`UINT` from the table that
/// write just installed, so this cannot re-enter through `RffiGlobal`.
fn rffi_module_init() -> RffiModule {
    let mut number_types = setup_unforced();
    let first_r_long = published_table_cint("LONG").rclass;
    PLATFORM
        .lock()
        .set_numbertype_to_rclass(LowLevelType::Signed, KnownType::Int);
    let (r_int_real_cls, int_real) = register_number_real("INT", "r_int_real");
    number_types.push(int_real.clone());
    let (r_uint_real_cls, uint_real) = register_number_real("UINT", "r_uint_real");
    number_types.push(uint_real.clone());
    RffiModule {
        number_types,
        r_int_real: r_int_real_cls,
        int_real,
        r_uint_real: r_uint_real_cls,
        uint_real,
        first_r_long,
    }
}

fn rffi_module() -> &'static RffiModule {
    RFFI_MODULE.get_or_init(rffi_module_init)
}

/// `NUMBER_TYPES` (`rffi.py`): `setup()` result plus `INT_real` and `UINT_real`.
pub static NUMBER_TYPES: LazyLock<Vec<LowLevelType>> =
    LazyLock::new(|| rffi_module().number_types.clone());

/// `rarithmetic.build_int("r_int_real", r_int.SIGN, r_int.BITS, True)` (`rffi.py`).
/// `r_int` here is the rffi C-int class bound by `setup()`.
pub static r_int_real: LazyLock<KnownType> = LazyLock::new(|| rffi_module().r_int_real);

/// `lltype.build_number("INT", r_int_real)` (`rffi.py`).
pub static INT_real: LazyLock<LowLevelType> = LazyLock::new(|| rffi_module().int_real.clone());

/// `rarithmetic.build_int("r_uint_real", r_uint.SIGN, r_uint.BITS, True)` (`rffi.py`).
/// `r_uint` here is the rffi C-unsigned-int class bound by `setup()`.
pub static r_uint_real: LazyLock<KnownType> = LazyLock::new(|| rffi_module().r_uint_real);

/// `lltype.build_number("UINT", r_uint_real)` (`rffi.py`).
pub static UINT_real: LazyLock<LowLevelType> = LazyLock::new(|| rffi_module().uint_real.clone());

/// `globals()[NAME]` / `globals()[NAMEP]` from `setup()` (`rffi.py`).
/// The export panics when accessed if that name is not in `TYPES` for
/// this target (POSIX-only names on Windows/wasm32; `__int128_t` when
/// the compiler lacks it). `os.name == 'nt'` aliases (`PID_T` /
/// `SSIZE_T` / `PTRDIFF_T` / `MODE_T`) are bare `NAME` bindings from
/// [`nt_fallback_type_aliases`]; `NAMEP` / `r_*` still panic.
macro_rules! rffi_c_int {
    ($id:ident, $pid:ident) => {
        pub static $id: RffiGlobal = RffiGlobal(stringify!($id));
        pub static $pid: RffiGlobal = RffiGlobal(concat!(stringify!($id), "P"));
    };
}

rffi_c_int!(SIGNEDCHAR, SIGNEDCHARP);
rffi_c_int!(UCHAR, UCHARP);
rffi_c_int!(SHORT, SHORTP);
rffi_c_int!(USHORT, USHORTP);
rffi_c_int!(INT, INTP);
rffi_c_int!(UINT, UINTP);
rffi_c_int!(LONG, LONGP);
rffi_c_int!(ULONG, ULONGP);
rffi_c_int!(LONGLONG, LONGLONGP);
rffi_c_int!(ULONGLONG, ULONGLONGP);
rffi_c_int!(SIZE_T, SIZE_TP);
rffi_c_int!(TIME_T, TIME_TP);
rffi_c_int!(WCHAR_T, WCHAR_TP);
rffi_c_int!(UINTPTR_T, UINTPTR_TP);
rffi_c_int!(INTPTR_T, INTPTR_TP);

// Upstream `__INT128_T` / `__UINT128_T` (`rffi.py` `setup`). Panics when
// accessed if `TYPES` does not contain `__int128_t` / `__uint128_t`.
rffi_c_int!(__INT128_T, __INT128_TP);
rffi_c_int!(__UINT128_T, __UINT128_TP);

rffi_c_int!(INT_LEAST8_T, INT_LEAST8_TP);
rffi_c_int!(UINT_LEAST8_T, UINT_LEAST8_TP);
rffi_c_int!(INT_LEAST16_T, INT_LEAST16_TP);
rffi_c_int!(UINT_LEAST16_T, UINT_LEAST16_TP);
rffi_c_int!(INT_LEAST32_T, INT_LEAST32_TP);
rffi_c_int!(UINT_LEAST32_T, UINT_LEAST32_TP);
rffi_c_int!(INT_LEAST64_T, INT_LEAST64_TP);
rffi_c_int!(UINT_LEAST64_T, UINT_LEAST64_TP);
rffi_c_int!(INT_FAST8_T, INT_FAST8_TP);
rffi_c_int!(UINT_FAST8_T, UINT_FAST8_TP);
rffi_c_int!(INT_FAST16_T, INT_FAST16_TP);
rffi_c_int!(UINT_FAST16_T, UINT_FAST16_TP);
rffi_c_int!(INT_FAST32_T, INT_FAST32_TP);
rffi_c_int!(UINT_FAST32_T, UINT_FAST32_TP);
rffi_c_int!(INT_FAST64_T, INT_FAST64_TP);
rffi_c_int!(UINT_FAST64_T, UINT_FAST64_TP);
rffi_c_int!(INTMAX_T, INTMAX_TP);
rffi_c_int!(UINTMAX_T, UINTMAX_TP);

rffi_c_int!(SSIZE_T, SSIZE_TP);
rffi_c_int!(MODE_T, MODE_TP);
rffi_c_int!(PID_T, PID_TP);
rffi_c_int!(PTRDIFF_T, PTRDIFF_TP);

/// `INT_realP` (`rffi.py`).
pub static INT_realP: LazyLock<LowLevelType> = LazyLock::new(|| CArrayPtr((*INT_real).clone()));
/// `UINT_realP` sibling of `INT_realP`.
pub static UINT_realP: LazyLock<LowLevelType> = LazyLock::new(|| CArrayPtr((*UINT_real).clone()));

fn ptr_to_array(of: LowLevelType, hints: Vec<(String, ConstValue)>) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(Array::with_hints(of, hints)),
    }))
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

/// Interpreter-level `rffi.cast(TGT, value)` (`rffi.py` = `ll2ctypes.force_cast`).
/// Integer 0 to a `Ptr` is `nullptr` (`ctypes.c_void_p(0)`). A nonzero
/// integer becomes the `_ptr_obj::IntCast` address carrier
/// (`cast_int_to_ptr` / `force_cast` `c_void_p`). Primitive destinations
/// follow the numeric part of `ll2ctypes.force_cast` (ctypes wrap, not
/// `cast_primitive` / `rint_new` range checks).
pub fn cast_value(tgt: &LowLevelType, value: &ConstValue) -> Result<ConstValue, CastWhateverError> {
    if !(tgt.is_primitive() || matches!(tgt, LowLevelType::Ptr(_))) {
        return Err(CastWhateverError::TypeError(format!(
            "rffi.cast() first arg should be a TYPE, got {tgt:?}"
        )));
    }
    let value = unwrap_force_cast_source(value);
    if let Ok(orig) = typeOf_const(&value)
        && orig == *tgt
    {
        return Ok(value);
    }
    if tgt.is_primitive() {
        if let Ok(orig) = typeOf_const(&value)
            && orig.is_primitive()
        {
            return force_cast_numeric(tgt, &value);
        }
        return Err(CastWhateverError::TypeError(format!(
            "casting {value:?} to {tgt:?}"
        )));
    }
    let LowLevelType::Ptr(tgt_ptr) = tgt else {
        return Err(CastWhateverError::TypeError(format!(
            "rffi.cast() first arg should be a TYPE, got {tgt:?}"
        )));
    };
    let n = integer_bits_for_ptr_cast(&value)?;
    if n == 0 {
        return nullptr(tgt_ptr.TO.clone().into())
            .map(|p| ConstValue::LLPtr(Box::new(p)))
            .map_err(CastWhateverError::TypeError);
    }
    Ok(ConstValue::LLPtr(Box::new(_ptr::new_with_solid(
        *tgt_ptr.clone(),
        Ok(Some(_ptr_obj::IntCast(n))),
        true,
    ))))
}

fn unwrap_force_cast_source(value: &ConstValue) -> ConstValue {
    match value {
        ConstValue::AddressAsInt(adr) | ConstValue::LLAddress(adr) => match adr {
            _address::Null => ConstValue::Int(0),
            _address::IntCast(n) => ConstValue::Int(*n),
            _address::Fake(ptr) => {
                if ptr.nonzero() {
                    ConstValue::LLPtr(ptr.clone())
                } else {
                    ConstValue::Int(0)
                }
            }
        },
        ConstValue::SingleFloat(_) | ConstValue::LongFloat(_) => {
            r_singlefloat_or_r_longfloat_as_float(value)
        }
        other => other.clone(),
    }
}

/// `ll2ctypes.force_cast`: `if isinstance(value, r_singlefloat): value = float(value)`
/// and `elif isinstance(cvalue, r_longfloat): cvalue = cvalue.value`.
fn r_singlefloat_or_r_longfloat_as_float(value: &ConstValue) -> ConstValue {
    match value {
        ConstValue::SingleFloat(bits) => ConstValue::float(f32::from_bits(*bits) as f64),
        ConstValue::LongFloat(bits) => ConstValue::float(f64::from_bits(*bits)),
        other => other.clone(),
    }
}

/// ctypes integer wrap used by `ll2ctypes.force_cast`:
/// `cresulttype(cvalue).value` masks to `bits` and, for a signed
/// destination, sign-extends (`c_byte(255).value == -1`).
/// `rint_new` / `Number._cast` raise OverflowError on signed overflow,
/// so they are not this wrap.
fn wrap_exact_int(bits: u16, signed: bool, value: i128) -> i128 {
    let bits_u = if bits == 0 {
        0
    } else if bits >= 128 {
        value as u128
    } else {
        (value as u128) & ((1u128 << bits) - 1)
    };
    if signed && bits > 0 && bits < 128 {
        let shift = 128 - bits;
        ((bits_u << shift) as i128) >> shift
    } else {
        bits_u as i128
    }
}

/// `lltype.Number` destinations `force_cast` wraps with ctypes
/// (`Signed` / `Unsigned` / longlong / `LowLevelType::Number`).
fn integer_dest_bits_signed(tgt: &LowLevelType) -> Option<(u16, bool)> {
    match tgt {
        LowLevelType::Signed => Some((rarithmetic::long_bit(), true)),
        LowLevelType::Unsigned => Some((rarithmetic::long_bit(), false)),
        LowLevelType::SignedLongLong => Some((64, true)),
        LowLevelType::UnsignedLongLong => Some((64, false)),
        LowLevelType::SignedLongLongLong => Some((128, true)),
        LowLevelType::UnsignedLongLongLong => Some((128, false)),
        LowLevelType::Number(n) => Some((n._type.bits, n._type.sign)),
        _ => None,
    }
}

/// `ll2ctypes.ctypes2lltype` for an integer `RESTYPE`.
fn ctypes2lltype_integer(RESTYPE: &LowLevelType, wrapped: i128) -> ConstValue {
    match RESTYPE {
        LowLevelType::Signed | LowLevelType::SignedLongLong => ConstValue::Int(wrapped as i64),
        LowLevelType::Unsigned | LowLevelType::UnsignedLongLong => ConstValue::UInt(wrapped as u64),
        LowLevelType::SignedLongLongLong => ConstValue::Int128(wrapped),
        LowLevelType::UnsignedLongLongLong => ConstValue::UInt128(wrapped as u128),
        LowLevelType::Number(n) => ConstValue::rint(n._type, wrapped),
        _ => ConstValue::Int(wrapped as i64),
    }
}

/// Numeric part of `ll2ctypes.force_cast` after the pointer shortcut
/// and the `RESTYPE == TYPE1` identity. Integer destinations wrap with
/// ctypes (`cresulttype(cvalue).value`). Bool is `bool(cvalue)` of the
/// original source; Float/SingleFloat/LongFloat take the source's exact
/// value as a float (`cresulttype(cvalue).value`). Char/UniChar keep
/// `cast_primitive`.
fn force_cast_numeric(
    RESTYPE: &LowLevelType,
    value: &ConstValue,
) -> Result<ConstValue, CastWhateverError> {
    let value = r_singlefloat_or_r_longfloat_as_float(value);
    if let Some((bits, signed)) = integer_dest_bits_signed(RESTYPE) {
        let t = truncated_from_const(&value)?;
        let wrapped = wrap_exact_int(bits, signed, t.low as i128);
        return Ok(ctypes2lltype_integer(RESTYPE, wrapped));
    }
    match RESTYPE {
        LowLevelType::Bool => force_cast_to_bool(&value),
        LowLevelType::Float | LowLevelType::SingleFloat | LowLevelType::LongFloat => {
            force_cast_to_float(RESTYPE, &value)
        }
        _ => cast_primitive(RESTYPE, &value),
    }
}

/// `ll2ctypes.force_cast`: `RESTYPE == Bool` → `bool(cvalue)` after
/// `lltype2ctypes` (`Char`/`UniChar` → `ord`). Float keeps `f != 0.0`
/// (`force_cast(Bool, 0.5)` is True); integers and character codes
/// use `truncated_from_const` then `!= 0`.
fn force_cast_to_bool(value: &ConstValue) -> Result<ConstValue, CastWhateverError> {
    if matches!(value, ConstValue::Float(_)) {
        return Ok(ConstValue::Bool(const_as_f64(value)? != 0.0));
    }
    let t = truncated_from_const(value)?;
    Ok(ConstValue::Bool(t.low != 0))
}

/// `ll2ctypes.force_cast`: Float/SingleFloat/LongFloat destination is
/// `cresulttype(cvalue).value` — the source's exact value as a float.
/// Unsigned sources stay positive (`rint_value` for `ConstValue::RInt`).
fn force_cast_to_float(
    RESTYPE: &LowLevelType,
    value: &ConstValue,
) -> Result<ConstValue, CastWhateverError> {
    let f = const_as_f64(value)?;
    match RESTYPE {
        LowLevelType::Float => Ok(ConstValue::float(f)),
        LowLevelType::SingleFloat => Ok(ConstValue::SingleFloat((f as f32).to_bits())),
        LowLevelType::LongFloat => Ok(ConstValue::LongFloat(f.to_bits())),
        _ => Err(CastWhateverError::TypeError("unsupported cast".into())),
    }
}

fn integer_bits_for_ptr_cast(value: &ConstValue) -> Result<i64, CastWhateverError> {
    match value {
        ConstValue::Int(n) => Ok(*n),
        ConstValue::UInt(n) => Ok(*n as i64),
        ConstValue::Bool(false) => Ok(0),
        ConstValue::Bool(true) => Ok(1),
        ConstValue::Int128(n) => Ok(*n as i64),
        ConstValue::UInt128(n) => Ok(*n as i64),
        ConstValue::RInt(c, v, _) => Ok(rarithmetic::rint_value(*c, *v).as_i64()),
        ConstValue::LLPtr(p) if !p.nonzero() => Ok(0),
        _ => Err(CastWhateverError::TypeError(format!(
            "casting {value:?} to pointer"
        ))),
    }
}

/// `rffi.py sizeof` for a `lltype.Number`: `tp._type.BITS/8`.
///
/// Python 2 `/` on ints is floor, so a 65-bit class is 8 bytes and
/// `getkind` banks it as `'int'` when that is `<= sizeof(Signed)`.
/// Char and Bool return 1, SingleFloat returns 4, on dedicated arms of
/// [`sizeof`]; they are not Numbers.
pub fn sizeof_number(n: &Number) -> i64 {
    (n._type.bits as i64) / 8
}

/// Forced `build_int` class wrapped as `lltype.Number`. Tests use widths
/// (`65`, `128`) that `setup()` does not publish.
#[cfg(test)]
pub(crate) fn number_from_forced_int(name: &'static str, sign: bool, bits: u16) -> LowLevelType {
    let c = match rarithmetic::build_int(Some(name), sign, bits, true) {
        Ok(KnownType::RInt(c)) => c,
        other => panic!("expected RInt, got {other:?}"),
    };
    LowLevelType::Number(Box::new(Number {
        _name: name.into(),
        _type: c,
    }))
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
        LowLevelType::Signed | LowLevelType::Unsigned => {
            Ok(ConstValue::Int(crate::layout::target_signed_size() as i64))
        }
        LowLevelType::Ptr(_) | LowLevelType::Address => {
            Ok(ConstValue::Int(crate::layout::target_word_size() as i64))
        }
        LowLevelType::Char | LowLevelType::Bool => Ok(ConstValue::Int(1)),
        LowLevelType::UniChar => {
            // `rffi.py sizeof`: `r_wchar_t.BITS/8`. UniChar stays the
            // unicode code-point primitive; `WCHAR_T` is a Number.
            Ok(ConstValue::Int(llmemory::sizeof_wchar_t()))
        }
        LowLevelType::Float => Ok(ConstValue::Int(8)),
        LowLevelType::SingleFloat => Ok(ConstValue::Int(4)),
        LowLevelType::LongFloat => Ok(ConstValue::Int(llmemory::sizeof_longfloat())),
        LowLevelType::SignedLongLong | LowLevelType::UnsignedLongLong => Ok(ConstValue::Int(8)),
        LowLevelType::SignedLongLongLong | LowLevelType::UnsignedLongLongLong => {
            Ok(ConstValue::Int(16))
        }
        LowLevelType::Number(n) => Ok(ConstValue::Int(sizeof_number(n))),
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
        LowLevelType::Number(n) => !n._type.sign,
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

/// RPython `MakeEntry(ExtRegistryEntry)` (`rffi.py`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MakeEntry;

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
pub(crate) mod tests {
    use super::*;

    /// Serializes tests that call `setup()` (`rffi.py`) or observe the
    /// published table / `numbertype_to_rclass`. Poison recovery so one
    /// panic does not wedge later tests in this module.
    static RFFI_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    pub(crate) fn rffi_test_guard() -> std::sync::MutexGuard<'static, ()> {
        RFFI_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

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
    fn cast_value_nonzero_integer_to_voidp_is_intcast_and_primitives_cast() {
        let got = cast_value(&*VOIDP, &ConstValue::UInt(0x1000)).expect("nonzero → VOIDP");
        let ConstValue::LLPtr(ptr) = got else {
            panic!("expected VOIDP pointer, got {got:?}");
        };
        assert!(ptr.nonzero());
        assert!(matches!(
            ptr._obj0_value(),
            Ok(Some(_ptr_obj::IntCast(0x1000)))
        ));
        let signed = cast_value(&LowLevelType::Signed, &ConstValue::UInt(7)).expect("prim dest");
        assert_eq!(signed, ConstValue::Int(7));
        let null = cast_value(&*VOIDP, &ConstValue::Int(0)).expect("0 → nullptr");
        let ConstValue::LLPtr(p0) = null else {
            panic!("expected nullptr, got {null:?}");
        };
        assert!(!p0.nonzero());
    }

    fn number_class(tp: &LowLevelType) -> crate::annotator::model::RIntClass {
        match tp {
            LowLevelType::Number(n) => n._type,
            other => panic!("expected Number, got {other:?}"),
        }
    }

    /// `ll2ctypes.force_cast`: integer dests wrap (`c_byte(255).value == -1`),
    /// unsigned sources stay positive as float, float→int is `int()` then wrap.
    #[test]
    fn cast_value_force_cast_wraps_like_ctypes() {
        let _guard = rffi_test_guard();
        let signedchar = (*SIGNEDCHAR).clone();
        let short = (*SHORT).clone();
        let uchar = (*UCHAR).clone();
        assert_eq!(
            cast_value(&signedchar, &ConstValue::Int(255)).unwrap(),
            ConstValue::rint(number_class(&signedchar), -1)
        );
        assert_eq!(
            cast_value(&short, &ConstValue::Int(65535)).unwrap(),
            ConstValue::rint(number_class(&short), -1)
        );
        assert_eq!(
            cast_value(&uchar, &ConstValue::Int(300)).unwrap(),
            ConstValue::rint(number_class(&uchar), 44)
        );
        assert_eq!(
            cast_value(&LowLevelType::Float, &ConstValue::UInt(u64::MAX)).unwrap(),
            ConstValue::float(18446744073709551615.0)
        );
        assert_eq!(
            cast_value(&uchar, &ConstValue::float(-1.5)).unwrap(),
            ConstValue::rint(number_class(&uchar), 255)
        );
        assert_eq!(
            cast_value(&short, &ConstValue::Bool(true)).unwrap(),
            ConstValue::rint(number_class(&short), 1)
        );
    }

    #[test]
    fn cast_value_force_cast_bool_float_singlefloat_and_rint_ptr() {
        let _guard = rffi_test_guard();
        assert_eq!(
            cast_value(
                &LowLevelType::Signed,
                &ConstValue::SingleFloat(2.5f32.to_bits()),
            )
            .unwrap(),
            ConstValue::Int(2)
        );
        assert_eq!(
            cast_value(&LowLevelType::Bool, &ConstValue::float(0.5)).unwrap(),
            ConstValue::Bool(true)
        );
        assert_eq!(
            cast_value(&LowLevelType::Float, &ConstValue::UInt(u64::MAX)).unwrap(),
            ConstValue::float(18446744073709551615.0)
        );
        let null = cast_value(&*VOIDP, &ConstValue::rint(number_class(&*UCHAR), 0))
            .expect("force_cast(VOIDP, r_uchar(0))");
        let ConstValue::LLPtr(p0) = null else {
            panic!("expected nullptr, got {null:?}");
        };
        assert!(!p0.nonzero());
        // `lltype2ctypes`: Char/UniChar → `ord` before `bool(cvalue)`.
        assert_eq!(
            cast_value(&LowLevelType::Bool, &ConstValue::ByteStr(vec![0])).unwrap(),
            ConstValue::Bool(false)
        );
        assert_eq!(
            cast_value(&LowLevelType::Bool, &ConstValue::UniStr("\0".into())).unwrap(),
            ConstValue::Bool(false)
        );
        assert_eq!(
            cast_value(&LowLevelType::Bool, &ConstValue::UniChar(0)).unwrap(),
            ConstValue::Bool(false)
        );
        assert_eq!(
            cast_value(&LowLevelType::Bool, &ConstValue::ByteStr(vec![b'a'])).unwrap(),
            ConstValue::Bool(true)
        );
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
        // Signed follows `target_signed_size` (8 on wasm32); pointers stay word.
        let word = crate::layout::target_word_size();
        let signed = crate::layout::target_signed_size();
        assert_eq!(size_and_sign(&*CCHARP).unwrap(), (word, false));
        assert_eq!(
            size_and_sign(&CArrayPtr(LowLevelType::Signed)).unwrap(),
            (word, false)
        );
        assert_eq!(
            size_and_sign(&LowLevelType::Signed).unwrap(),
            (signed, false)
        );
        assert_eq!(size_and_sign(&LowLevelType::Char).unwrap(), (1, true));
        assert_eq!(
            size_and_sign(&LowLevelType::Unsigned).unwrap(),
            (signed, true)
        );
        assert_eq!(
            size_and_sign(&LowLevelType::SignedLongLong).unwrap(),
            (8, false)
        );
        let four_signed =
            LowLevelType::FixedSizeArray(Box::new(CFixedArray(LowLevelType::Signed, 4)));
        assert_eq!(
            sizeof(&four_signed).unwrap(),
            ConstValue::Int((4 * signed) as i64)
        );
        assert_eq!(size_and_sign(&four_signed).unwrap(), (4 * signed, false));
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
    fn sizeof_number_floors_bits_over_eight() {
        // `rffi.sizeof`: `tp._type.BITS/8`. Char/Bool/SingleFloat are
        // not Numbers (1 / 1 / 4 on dedicated arms).
        let n65 = number_from_forced_int("r_int65_rffi", true, 65);
        assert_eq!(sizeof(&n65).unwrap(), ConstValue::Int(8));
        match &n65 {
            LowLevelType::Number(n) => assert_eq!(sizeof_number(n), 8),
            other => panic!("expected Number, got {other:?}"),
        }
        assert_eq!(sizeof(&*UCHAR).unwrap(), ConstValue::Int(1));
        let n128 = number_from_forced_int("r_int128_rffi", true, 128);
        assert_eq!(sizeof(&n128).unwrap(), ConstValue::Int(16));
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
    fn r_int_real_has_c_int_width_and_is_a_fresh_class() {
        let _guard = rffi_test_guard();
        let (sign, bits) = rffi_c_int_sign_bits("INT", &*INT);
        match *r_int_real {
            KnownType::RInt(c) => assert_eq!(c.bits, bits),
            other => panic!("r_int_real is a forced class, got {other:?}"),
        }
        assert_ne!(
            *r_int_real,
            rarithmetic::build_int(None, sign, bits, false).unwrap()
        );
    }

    #[test]
    fn inttype_population_preserves_rffi_names() {
        let _guard = rffi_test_guard();
        let names = populate_inttypes();
        assert!(names.contains(&"short".to_string()));
        assert!(names.contains(&"ushort".to_string()));
        assert!(names.contains(&"int".to_string()));
        assert!(names.contains(&"uint".to_string()));
        assert!(names.contains(&"void*".to_string()));
        assert!(NUMBER_TYPES.len() >= names.len() + 2);
        assert_ne!(*UCHAR, UNSIGNED);
        assert_eq!(sizeof(&*UCHAR).unwrap(), ConstValue::Int(1));
        assert_eq!(sizeof(&*USHORT).unwrap(), ConstValue::Int(2));
        assert_eq!(sizeof(&*UINT).unwrap(), ConstValue::Int(4));
        assert_eq!(size_and_sign(&*UCHAR).unwrap(), (1, true));
        assert_eq!(size_and_sign(&*SHORT).unwrap(), (2, false));
        let (_, long_bits) = rffi_c_int_sign_bits("LONG", &*LONG);
        if long_bits == 64 {
            assert_eq!(*ULONG, UNSIGNED);
            assert_eq!(*LONG, SIGNED);
        }
        assert_ne!(*UINT_real, *UINT);
        match &*UCHAR {
            LowLevelType::Number(n) => {
                assert_eq!(
                    cast_primitive(&*UCHAR, &ConstValue::Int(-1)).unwrap(),
                    ConstValue::rint(n._type, 255)
                );
            }
            other => panic!("UCHAR is a Number, got {other:?}"),
        }
        let (int_sign, int_bits) = rffi_c_int_sign_bits("INT", &*INT);
        let (uint_sign, uint_bits) = rffi_c_int_sign_bits("UINT", &*UINT);
        match *r_int_real {
            KnownType::RInt(c) => assert_eq!(c.bits, int_bits),
            other => panic!("r_int_real is a forced class, got {other:?}"),
        }
        match *r_uint_real {
            KnownType::RInt(c) => assert_eq!(c.bits, uint_bits),
            other => panic!("r_uint_real is a forced class, got {other:?}"),
        }
        assert_ne!(
            *r_int_real,
            rarithmetic::build_int(None, int_sign, int_bits, false).unwrap()
        );
        assert_ne!(
            *r_uint_real,
            rarithmetic::build_int(None, uint_sign, uint_bits, false).unwrap()
        );
        assert_eq!(r_singlefloat, LowLevelType::SingleFloat);
        assert_eq!(NULL, None);
    }

    fn rclass_name(kt: KnownType) -> String {
        match kt {
            KnownType::Int => "int".to_string(),
            KnownType::Ruint => "r_uint".to_string(),
            KnownType::LongLong => "r_longlong".to_string(),
            KnownType::LongLongLong => "r_longlonglong".to_string(),
            KnownType::ULongLong => "r_ulonglong".to_string(),
            KnownType::ULongLongLong => "r_ulonglonglong".to_string(),
            KnownType::RInt(c) => c.name.to_string(),
            other => format!("{other:?}"),
        }
    }

    #[test]
    fn numbertype_to_rclass_is_keyed_by_lltype() {
        let _guard = rffi_test_guard();
        match numbertype_to_rclass(&*UCHAR) {
            KnownType::RInt(c) => {
                assert!(!c.sign);
                assert_eq!(c.bits, 8);
                assert_eq!(c.name.to_ascii_lowercase(), "r_uchar");
            }
            other => panic!("UCHAR rclass: {other:?}"),
        }
        assert_eq!(rclass_name(r_uchar()).to_ascii_lowercase(), "r_uchar");
        assert_eq!(r_class("uchar"), numbertype_to_rclass(&*UCHAR));
        assert_eq!(r_class("uchar"), r_uchar());
        assert_eq!(r_int(), numbertype_to_rclass(&*INT));
        assert_eq!(numbertype_to_rclass(&LowLevelType::Signed), KnownType::Int);
        match numbertype_to_rclass(&*INT_real) {
            KnownType::RInt(c) => assert_eq!(c.name, "r_int_real"),
            other => panic!("INT_real rclass: {other:?}"),
        }
        match numbertype_to_rclass(&*UINT_real) {
            KnownType::RInt(c) => assert_eq!(c.name, "r_uint_real"),
            other => panic!("UINT_real rclass: {other:?}"),
        }
    }

    #[test]
    fn number_types_void_star_is_integer_not_voidp() {
        let _guard = rffi_test_guard();
        let _ = &*NUMBER_TYPES;
        let void_star = PLATFORM
            .lock()
            .get("VOID*")
            .expect("setup populates VOID*")
            .lltype
            .clone();
        assert!(void_star.is_number());
        assert_eq!(
            sizeof(&void_star).unwrap(),
            ConstValue::Int(crate::layout::target_word_size() as i64)
        );
        match &void_star {
            LowLevelType::Number(n) => assert_eq!(n._name, "VOID*"),
            LowLevelType::Signed => {}
            other => panic!("VOID* is a Number, got {other:?}"),
        }
        assert!(NUMBER_TYPES.iter().any(|tp| *tp == void_star));
        assert!(!NUMBER_TYPES.iter().any(|tp| *tp == *VOIDP));
    }

    #[test]
    fn types_includes_int128_iff_sizeof_succeeds() {
        let _guard = rffi_test_guard();
        let sizeof_ok = sizeof_c_type("__int128_t", true).is_ok();
        assert_eq!(TYPES.iter().any(|name| *name == "__int128_t"), sizeof_ok);
        assert_eq!(TYPES.iter().any(|name| *name == "__uint128_t"), sizeof_ok);
        if sizeof_ok {
            let signed128 = platform_lltype("__INT128_T");
            assert_eq!(signed128, LowLevelType::SignedLongLongLong);
            assert_eq!(numbertype_to_rclass(&signed128), KnownType::LongLongLong);
            let unsigned128 = platform_lltype("__UINT128_T");
            assert_eq!(unsigned128, LowLevelType::UnsignedLongLongLong);
            assert_eq!(numbertype_to_rclass(&unsigned128), KnownType::ULongLongLong);
            assert_eq!(*__INT128_T, signed128);
            assert_eq!(*__UINT128_T, unsigned128);
            assert_eq!(r_class("__int128_t"), numbertype_to_rclass(&*__INT128_T));
            assert_eq!(r___int128_t(), r_class("__int128_t"));
        } else {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = &*__INT128_T;
                }))
                .is_err(),
                "__INT128_T must panic when TYPES lacks __int128_t"
            );
        }
    }

    #[test]
    fn touching_uchar_runs_full_rffi_module_init() {
        let _guard = rffi_test_guard();
        let _ = &*UCHAR;
        assert_eq!(numbertype_to_rclass(&LowLevelType::Signed), KnownType::Int);
        let long_class = rffi_module().first_r_long;
        let (_, long_bits) = rffi_c_int_sign_bits("LONG", &*LONG);
        if long_bits == 64 {
            assert_ne!(
                long_class,
                KnownType::Int,
                "r_class(\"long\") is the class setup() bound, not live Signed→int"
            );
            match long_class {
                KnownType::Rint => {}
                KnownType::RInt(c) => {
                    assert!(c.sign);
                    assert!(c.name.to_ascii_lowercase().contains("long"));
                }
                other => panic!("expected generated r_long class, got {other:?}"),
            }
        } else {
            assert_eq!(
                long_class,
                r_class("int"),
                "r_class(\"long\") shares the (signed, 32) class with r_class(\"int\")"
            );
        }
    }

    #[test]
    fn r_class_reads_setup_snapshot_not_live_map() {
        let _guard = rffi_test_guard();
        let _ = &*UCHAR;
        assert_eq!(numbertype_to_rclass(&LowLevelType::Signed), KnownType::Int);
        let long_class = rffi_module().first_r_long;
        let (_, long_bits) = rffi_c_int_sign_bits("LONG", &*LONG);
        if long_bits == 64 {
            assert_ne!(
                long_class,
                KnownType::Int,
                "r_class(\"long\") is the class setup() bound, not live Signed→int"
            );
            match long_class {
                KnownType::Rint => {}
                KnownType::RInt(c) => {
                    assert!(c.sign);
                    assert!(c.name.to_ascii_lowercase().contains("long"));
                }
                other => panic!("expected generated r_long class, got {other:?}"),
            }
        } else {
            assert_eq!(
                long_class,
                r_class("int"),
                "r_class(\"long\") shares the (signed, 32) class with r_class(\"int\")"
            );
        }
        let uchar_tp = (*UCHAR).clone();
        let uchar_class = r_class("uchar");
        PLATFORM
            .lock()
            .set_numbertype_to_rclass(uchar_tp.clone(), KnownType::Int);
        assert_eq!(r_class("uchar"), uchar_class);
        PLATFORM
            .lock()
            .set_numbertype_to_rclass(uchar_tp, uchar_class);
        let long_tp = (*LONG).clone();
        PLATFORM
            .lock()
            .set_numbertype_to_rclass(long_tp, long_class);
        let first = setup();
        let second = setup();
        assert_eq!(first, second);
        assert_eq!(r_class("uchar"), uchar_class);
        assert_eq!(r_class("long"), long_class);
        PLATFORM
            .lock()
            .set_numbertype_to_rclass(LowLevelType::Signed, KnownType::Int);
    }

    #[test]
    fn setup_twice_returns_same_static_uchar_pointers() {
        let _guard = rffi_test_guard();
        let _ = setup();
        let uchar: &LowLevelType = &*UCHAR;
        let ucharp: &LowLevelType = &*UCHARP;
        let _ = setup();
        assert!(
            std::ptr::eq(uchar, &*UCHAR),
            "setup() interned UCHAR must keep the same &'static pointer"
        );
        assert!(
            std::ptr::eq(ucharp, &*UCHARP),
            "setup() interned UCHARP must keep the same &'static pointer"
        );
    }

    #[test]
    fn external_setup_runs_after_signed_to_int_override() {
        let _guard = rffi_test_guard();
        let _ = setup();
        assert_eq!(numbertype_to_rclass(&LowLevelType::Signed), KnownType::Int);
    }

    #[test]
    fn int_least16_tp_is_nolength_array_ptr_on_unix() {
        let _guard = rffi_test_guard();
        if !types_contains("int_least16_t") {
            return;
        }
        let array = ptr_array_of(&INT_LEAST16_TP);
        assert_eq!(array.OF, *INT_LEAST16_T);
        assert_eq!(array._hints.get("nolength"), Some(&ConstValue::Bool(true)));
        assert_eq!(*INT_LEAST16_TP, CArrayPtr((*INT_LEAST16_T).clone()));
        assert_eq!(
            r_class("int_least16_t"),
            numbertype_to_rclass(&*INT_LEAST16_T)
        );
    }

    #[test]
    fn windows_aliases_match_rffi_nt_branch() {
        let _guard = rffi_test_guard();
        if crate::layout::target_os() == "windows" {
            assert_eq!(*PID_T, LowLevelType::Signed);
            assert_eq!(*SSIZE_T, LowLevelType::Signed);
            assert_eq!(*PTRDIFF_T, LowLevelType::Signed);
            assert_eq!(*MODE_T, *INT);
        } else if types_contains("pid_t") {
            assert_eq!(*PID_T, platform_lltype("PID_T"));
            assert_eq!(*SSIZE_T, platform_lltype("SSIZE_T"));
            assert_eq!(*PTRDIFF_T, platform_lltype("PTRDIFF_T"));
            assert_eq!(*MODE_T, platform_lltype("MODE_T"));
        }
    }

    #[test]
    fn nt_fallback_aliases_are_bare_type_aliases() {
        let _guard = rffi_test_guard();
        let windows_types = [
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
        let int_tp = LowLevelType::Signed;
        let aliases = nt_fallback_type_aliases(&windows_types, int_tp.clone());
        assert_eq!(aliases.get("PID_T"), Some(&SIGNED));
        assert_eq!(aliases.get("SSIZE_T"), Some(&SIGNED));
        assert_eq!(aliases.get("PTRDIFF_T"), Some(&SIGNED));
        assert_eq!(aliases.get("MODE_T"), Some(&int_tp));
        assert!(!aliases.contains_key("PID_TP"));
        assert!(!aliases.contains_key("SSIZE_TP"));
        assert!(!aliases.contains_key("PTRDIFF_TP"));
        assert!(!aliases.contains_key("MODE_TP"));
        assert!(!aliases.contains_key("r_pid_t"));
        assert!(!aliases.keys().any(|key| key.starts_with("r_")));
        assert_eq!(aliases.len(), 4);

        let unix_types = ["mode_t", "pid_t", "ssize_t", "ptrdiff_t"];
        assert!(nt_fallback_type_aliases(&unix_types, int_tp).is_empty());
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
