//! Process-wide registry of macro-emitted residual-call trampolines.
//!
//! `#[dont_look_inside]` / `#[elidable]` (and the rest of that family) emit
//! an `extern "C"` trampoline next to the annotated function. This slice
//! publishes `(path, address, arity)` for each trampoline whose ABI matches
//! what the codewriter records, so a residual callee has a real address
//! without a hand-maintained list.
//!
//! The address is captured in the defining crate (`fn as *const ()` in that
//! crate's static or constructor). Taking `fn as usize` in another crate can
//! produce a second wasm32 table slot; reading the stored pointer bits does
//! not.
//!
//! `distributed_slice` is not implemented for wasm32; that target fills
//! [`WASM_HELPER_FNADDRS`] from a constructor instead.
//!
//! `#[prebuilt_static]` on a `PyType` static publishes that class singleton
//! through [`PREBUILT_CLASS_STATICS`]. An `Atomic*` static publishes a
//! nullary getter through [`HELPER_FNADDRS`] instead.

/// A fieldless enum a residual call passes as its discriminant word.
/// # Safety: `from_discriminant` must return a valid value for every
/// discriminant of `Self`; implement it only through the derive.
pub unsafe trait FieldlessEnumArg: Copy {
    fn from_discriminant(d: i64) -> Self;
}

/// One residual-call trampoline the macros published.
#[derive(Clone, Copy)]
pub struct HelperFnAddr {
    /// `concat!(module_path!(), "::", stringify!(name))` of the annotated
    /// function — the `FunctionPath` a residual call names.
    pub path: &'static str,
    /// Trampoline pointer captured in the defining crate.
    addr: *const (),
    /// Argument count in source order, matching the trampoline.
    pub arity: u8,
}

// Safety: `path` is `'static` and `addr` is a process-global function
// pointer; sharing across threads is sound.
unsafe impl Sync for HelperFnAddr {}
unsafe impl Send for HelperFnAddr {}

impl HelperFnAddr {
    /// Build a registry row. Called from a `const` static initializer in the
    /// defining crate, where `fn as *const ()` is a valid initializer.
    pub const fn new(path: &'static str, addr: *const (), arity: u8) -> Self {
        Self { path, addr, arity }
    }

    /// Address captured in the defining crate, as a `usize`.
    pub fn get(&self) -> usize {
        self.addr as usize
    }
}

/// Link-time registry of every macro-published residual trampoline.
///
/// `distributed_slice` rejects wasm32, which carries the same set in
/// [`WASM_HELPER_FNADDRS`]; read both through [`for_each_helper_fnaddr`].
#[cfg(not(target_arch = "wasm32"))]
#[::linkme::distributed_slice]
pub static HELPER_FNADDRS: [HelperFnAddr] = [..];

/// The same registry on wasm32, populated at constructor time.
#[cfg(target_arch = "wasm32")]
pub static WASM_HELPER_FNADDRS: std::sync::Mutex<Vec<HelperFnAddr>> =
    std::sync::Mutex::new(Vec::new());

/// Append one trampoline to [`WASM_HELPER_FNADDRS`].
///
/// Called only from the constructor the macros emit. An entry appended after
/// the address table has been read is not published.
#[cfg(target_arch = "wasm32")]
pub fn register(path: &'static str, addr: *const (), arity: u8) {
    WASM_HELPER_FNADDRS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(HelperFnAddr::new(path, addr, arity));
}

/// Error a residual trampoline can report out of band.
///
/// RPython residual calls return the value and store the exception in
/// `exc_data`; the codewriter then records `GUARD_NO_EXCEPTION`. A trampoline
/// for `Result<T, E>` therefore returns `T` (or the zero of that kind) and
/// calls this on `Err`. The defining crate implements it for its exception
/// carrier — `majit` does not name that type.
pub trait ResidualError {
    fn publish_residual(self);
}

/// Visit every registered trampoline, whichever population the target carries.
pub fn for_each_helper_fnaddr(mut visit: impl FnMut(&HelperFnAddr)) {
    #[cfg(not(target_arch = "wasm32"))]
    for desc in HELPER_FNADDRS {
        visit(desc);
    }
    #[cfg(target_arch = "wasm32")]
    {
        let guard = WASM_HELPER_FNADDRS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for desc in guard.iter() {
            visit(desc);
        }
    }
}

/// One prebuilt class-singleton static the macros published.
#[derive(Clone, Copy)]
pub struct PrebuiltStaticAddr {
    /// `concat!(module_path!(), "::", stringify!(name))` of the annotated
    /// static — the path a class-singleton address table names.
    pub path: &'static str,
    /// Address of the static, captured in the defining crate.
    addr: *const (),
}

// Safety: `path` is `'static` and `addr` is a process-global static;
// sharing across threads is sound.
unsafe impl Sync for PrebuiltStaticAddr {}
unsafe impl Send for PrebuiltStaticAddr {}

impl PrebuiltStaticAddr {
    /// Build a registry row. Called from a `const` static initializer in the
    /// defining crate, where `&STATIC as *const ()` is a valid initializer.
    pub const fn new(path: &'static str, addr: *const ()) -> Self {
        Self { path, addr }
    }

    /// Address captured in the defining crate, as a `usize`.
    pub fn get(&self) -> usize {
        self.addr as usize
    }
}

/// Link-time registry of every macro-published prebuilt class singleton.
///
/// `distributed_slice` rejects wasm32, which carries the same set in
/// [`WASM_PREBUILT_CLASS_STATICS`]; read both through
/// [`for_each_prebuilt_class_static`].
#[cfg(not(target_arch = "wasm32"))]
#[::linkme::distributed_slice]
pub static PREBUILT_CLASS_STATICS: [PrebuiltStaticAddr] = [..];

/// The same registry on wasm32, populated at constructor time.
#[cfg(target_arch = "wasm32")]
pub static WASM_PREBUILT_CLASS_STATICS: std::sync::Mutex<Vec<PrebuiltStaticAddr>> =
    std::sync::Mutex::new(Vec::new());

/// Append one class singleton to [`WASM_PREBUILT_CLASS_STATICS`].
///
/// Called only from the constructor the macros emit. An entry appended after
/// the address table has been read is not published.
#[cfg(target_arch = "wasm32")]
pub fn register_prebuilt_class_static(path: &'static str, addr: *const ()) {
    WASM_PREBUILT_CLASS_STATICS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(PrebuiltStaticAddr::new(path, addr));
}

/// Reads the byte payload of one rstr `STR` word.
///
/// JIT code holds `&str` / `&Wtf8` as a single `Ref` word (a pointer to an
/// rstr `STR`). The defining frontend owns that layout, so it registers the
/// reader; `majit` does not name it.
pub type RstrPayloadFn = unsafe fn(i64) -> &'static [u8];

static RSTR_PAYLOAD: std::sync::OnceLock<RstrPayloadFn> = std::sync::OnceLock::new();

/// Frontend-registered rstr `STR` payload reader. First call wins; subsequent
/// calls are silently ignored to mirror `OnceLock::set`'s init-once contract.
/// Frontends register at JitDriver startup before any residual trampoline
/// rebuilds a `&str` or `&Wtf8` argument.
pub fn set_rstr_payload_reader(f: RstrPayloadFn) {
    let _ = RSTR_PAYLOAD.set(f);
}

/// Rebuild a shared object-pointer slice from one residual `Ref` word.
///
/// The word is the address of a length-prefixed `GcArray` (`lltype.py
/// GcArray`): a `usize` length header at offset 0, then the items. That is
/// the same shape as `majit_rlib` `TypedItemsBlock` and pyre-object
/// `ItemsBlock`. A zero word is the empty slice.
///
/// # Safety
/// `word` is 0, or a pointer to that layout whose items of type `T` are live
/// for `'a`.
pub unsafe fn object_gcarray_slice<'a, T>(word: i64) -> &'a [T] {
    if word == 0 {
        return &[];
    }
    let header = word as usize as *const usize;
    let len = unsafe { *header };
    let items = (word as usize + std::mem::size_of::<usize>()) as *const T;
    unsafe { std::slice::from_raw_parts(items, len) }
}

/// Bytes of the rstr `STR` word `word`, via the reader installed by
/// [`set_rstr_payload_reader`].
///
/// # Safety
/// `word` must be a pointer the registered reader accepts. A null word means
/// whatever that reader defines (pyre: the empty payload).
///
/// # Panics
/// Panics if no reader was registered.
pub unsafe fn rstr_payload(word: i64) -> &'static [u8] {
    match RSTR_PAYLOAD.get() {
        Some(reader) => unsafe { reader(word) },
        None => panic!(
            "rstr_payload: no STR payload reader registered — frontend must call \
             `set_rstr_payload_reader` at startup before a residual trampoline \
             rebuilds a `&str` or `&Wtf8` argument"
        ),
    }
}

/// Suffix that distinguishes one residual monomorph of a generic helper.
///
/// `annlowlevel.specialize` / `FunctionDesc.cachedgraph` names each
/// specialized graph; `getfunctionptr(graph)` is then a distinct funcptr
/// (`rpython/jit/codewriter/call.py` `get_jitcode_calldescr`). Charon
/// extracts with `monomorphize: false`, so a generic `dont_look_inside`
/// helper is one FunDecl. The residual CallPath appends this suffix,
/// built from the call site's type arguments, so two instantiations
/// cannot share an address.
pub fn residual_inst_suffix(type_names: &[&str]) -> String {
    if type_names.is_empty() {
        return String::new();
    }
    let mut out = String::from("__inst");
    for name in type_names {
        out.push('_');
        out.push_str(&residual_inst_type_token(name));
    }
    out
}

/// One path-segment token for a rustc `type_name` or Charon `spec_type_name`.
///
/// The input is reduced to a canonical spelling (full module paths, refs,
/// arrays, tuples and generics kept; whitespace dropped except one space
/// after `mut` / `const` / `dyn`) and then escaped injectively so two
/// distinct types cannot share a suffix.
pub fn residual_inst_type_token(name: &str) -> String {
    encode_canonical_type(&canonical_type_spelling(name))
}

/// rustc `type_name` spelling with whitespace collapsed so both renderers
/// agree: drop whitespace, then keep exactly one space after the keywords
/// `mut`, `const`, and `dyn`.
fn canonical_type_spelling(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut rest = name.trim();
    while !rest.is_empty() {
        let first = rest.chars().next().unwrap();
        if first.is_whitespace() {
            rest = rest.trim_start();
            continue;
        }
        if let Some((kw, after)) = match_type_keyword(rest, &out) {
            out.push_str(kw);
            rest = after.trim_start();
            if !rest.is_empty() {
                out.push(' ');
            }
            continue;
        }
        out.push(first);
        rest = &rest[first.len_utf8()..];
    }
    out
}

fn match_type_keyword<'a>(rest: &'a str, out: &str) -> Option<(&'static str, &'a str)> {
    let prev_ok = match out.chars().last() {
        None => true,
        Some(c) => !is_ident_char(c),
    };
    if !prev_ok {
        return None;
    }
    for kw in ["const", "mut", "dyn"] {
        if let Some(after) = rest.strip_prefix(kw) {
            let next_ok = match after.chars().next() {
                None => true,
                Some(c) => !is_ident_char(c),
            };
            if next_ok {
                return Some((kw, after));
            }
        }
    }
    None
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn encode_canonical_type(name: &str) -> String {
    let mut out = String::with_capacity(name.len().saturating_mul(2));
    let mut chars = name.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
            continue;
        }
        if c == '_' {
            out.push_str("__");
            continue;
        }
        if c == ':' && chars.peek() == Some(&':') {
            chars.next();
            out.push_str("_C");
            continue;
        }
        match c {
            '<' => out.push_str("_L"),
            '>' => out.push_str("_G"),
            ',' => out.push_str("_M"),
            '&' => out.push_str("_R"),
            '*' => out.push_str("_P"),
            ' ' => out.push_str("_S"),
            '[' => out.push_str("_B"),
            ']' => out.push_str("_E"),
            ';' => out.push_str("_Y"),
            '(' => out.push_str("_O"),
            ')' => out.push_str("_Q"),
            other => out.push_str(&format!("_x{:x}_", other as u32)),
        }
    }
    out
}

/// Visit every registered class singleton, whichever population the target carries.
pub fn for_each_prebuilt_class_static(mut visit: impl FnMut(&PrebuiltStaticAddr)) {
    #[cfg(not(target_arch = "wasm32"))]
    for desc in PREBUILT_CLASS_STATICS {
        visit(desc);
    }
    #[cfg(target_arch = "wasm32")]
    {
        let guard = WASM_PREBUILT_CLASS_STATICS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for desc in guard.iter() {
            visit(desc);
        }
    }
}

#[cfg(test)]
mod object_gcarray_slice_tests {
    use super::object_gcarray_slice;

    #[test]
    fn reads_length_prefixed_block_and_zero_word() {
        let a = 0xAAusize;
        let b = 0xBBusize;
        let c = 0xCCusize;
        let block: [usize; 4] = [3, a, b, c];
        let slice = unsafe { object_gcarray_slice::<usize>(block.as_ptr() as i64) };
        assert_eq!(slice, &[a, b, c]);
        let empty = unsafe { object_gcarray_slice::<usize>(0) };
        assert!(empty.is_empty());
    }
}

#[cfg(test)]
mod residual_inst_suffix_tests {
    use super::{residual_inst_suffix, residual_inst_type_token};

    #[test]
    fn tokens_encode_full_canonical_spelling() {
        assert_eq!(residual_inst_type_token("i64"), "i64");
        assert_eq!(residual_inst_type_token("&i64"), "_Ri64");
        assert_eq!(residual_inst_type_token("&mut i64"), "_Rmut_Si64");
        assert_eq!(residual_inst_type_token("*mut i64"), "_Pmut_Si64");
        assert_eq!(residual_inst_type_token("*const i64"), "_Pconst_Si64");
        assert_eq!(residual_inst_type_token("a::Key"), "a_CKey");
        assert_eq!(residual_inst_type_token("b::Key"), "b_CKey");
        assert_eq!(residual_inst_type_token("()"), "_O_Q");
        assert_eq!(
            residual_inst_type_token("pyre_object::identitydict::IdentityKey"),
            "pyre__object_Cidentitydict_CIdentityKey"
        );
        assert_eq!(
            residual_inst_type_token("&pyre_object::dictmultiobject::IntDictStorage"),
            "_Rpyre__object_Cdictmultiobject_CIntDictStorage"
        );
        assert_eq!(
            residual_inst_type_token("*mut pyre_object::pyobject::PyObject"),
            "_Pmut_Spyre__object_Cpyobject_CPyObject"
        );
        assert_eq!(
            residual_inst_type_token(
                "core::hash::BuildHasherDefault<pyre_object::dictmultiobject::IntKeyHasher>"
            ),
            "core_Chash_CBuildHasherDefault_Lpyre__object_Cdictmultiobject_CIntKeyHasher_G"
        );
        assert_eq!(
            residual_inst_type_token("Foo<A, B>"),
            residual_inst_type_token("Foo<A,B>")
        );
        assert_eq!(
            residual_inst_type_token("[T; N]"),
            residual_inst_type_token("[T;N]")
        );
        assert_eq!(
            residual_inst_type_token("&mut  T"),
            residual_inst_type_token("&mut T")
        );
    }

    #[test]
    fn tokens_are_injective_for_boundary_cases() {
        assert_ne!(
            residual_inst_type_token("i64"),
            residual_inst_type_token("&i64")
        );
        assert_ne!(
            residual_inst_type_token("&T"),
            residual_inst_type_token("&mut T")
        );
        assert_ne!(
            residual_inst_type_token("*mut T"),
            residual_inst_type_token("*const T")
        );
        assert_ne!(
            residual_inst_type_token("a::Key"),
            residual_inst_type_token("b::Key")
        );
        assert_ne!(
            residual_inst_type_token("Foo<A, B>"),
            residual_inst_type_token("Foo<A<B>>")
        );
        assert_ne!(
            residual_inst_type_token("hash_map"),
            residual_inst_type_token("hash::map")
        );
    }

    #[test]
    fn suffix_distinguishes_instantiations() {
        let int = residual_inst_suffix(&[
            "i64",
            "pyre_object::pyobject::PyObjectRef",
            "pyre_object::dictmultiobject::IntKeyHash",
        ]);
        let identity = residual_inst_suffix(&[
            "pyre_object::identitydict::IdentityKey",
            "pyre_object::pyobject::PyObjectRef",
            "std::collections::hash_map::RandomState",
        ]);
        assert_ne!(int, identity);
        assert!(int.starts_with("__inst_"), "{int}");
        assert!(!int.contains("::"), "{int}");
        assert!(identity.contains("IdentityKey"), "{identity}");
        assert!(identity.contains("RandomState"), "{identity}");
        assert!(int.contains("i64"), "{int}");
        assert!(!int.contains("IdentityKey"), "{int}");
    }
}
