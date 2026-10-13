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
