//! `rpython/rlib/rlocale.py`.
//!
//! `numeric_formatting` is the entry the number formatter reads. It shares
//! the `localeconv` walk with `_locale.localeconv`, so the grouping
//! `format(x, 'n')` uses and the grouping `locale.localeconv()` reports
//! come from the same read.
//!
//! `setlocale` raises [`LocaleError`] when the host refuses the setting.
//! The `_locale` builtin maps that error onto `locale.Error`.
//!
//! `rlocale.external` sets `sandboxsafe=True`.

#[cfg(all(unix, not(target_arch = "wasm32")))]
use crate::rffi::{CCHARP, INT};

/// `rlocale.py LocaleError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocaleError {
    /// `setlocale` returned a null pointer.
    Unsupported,
}

/// `localeconv` fields the `_locale` builtin turns into a dict.
///
/// Grouping bytes are `charp2str` of `lconv.grouping` / `mon_grouping`.
/// The trailing `0` `_w_copy_grouping` appends belongs to the Python list,
/// not to this walk.
#[derive(Debug, Clone)]
pub struct LocaleConv {
    pub decimal_point: Vec<u8>,
    pub thousands_sep: Vec<u8>,
    pub grouping: Vec<u8>,
    pub int_curr_symbol: Vec<u8>,
    pub currency_symbol: Vec<u8>,
    pub mon_decimal_point: Vec<u8>,
    pub mon_thousands_sep: Vec<u8>,
    pub mon_grouping: Vec<u8>,
    pub positive_sign: Vec<u8>,
    pub negative_sign: Vec<u8>,
    pub int_frac_digits: core::ffi::c_char,
    pub frac_digits: core::ffi::c_char,
    pub p_cs_precedes: core::ffi::c_char,
    pub p_sep_by_space: core::ffi::c_char,
    pub n_cs_precedes: core::ffi::c_char,
    pub n_sep_by_space: core::ffi::c_char,
    pub p_sign_posn: core::ffi::c_char,
    pub n_sign_posn: core::ffi::c_char,
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
crate::rffi::external_compilation_info! {
    const LOCALE_ECI = {
        includes: ["locale.h", "limits.h", "ctype.h", "wchar.h", "langinfo.h"],
    };
}

// `rlocale.localeconv` (`sandboxsafe=True`).
#[cfg(all(unix, not(target_arch = "wasm32"), feature = "host_env"))]
crate::rffi::llexternal!(
    pub localeconv = "localeconv",
    [],
    *mut libc::lconv,
    compilation_info = LOCALE_ECI,
    sandboxsafe = true
);

// `rlocale._setlocale` (`sandboxsafe=True`).
#[cfg(all(unix, not(target_arch = "wasm32"), feature = "host_env"))]
crate::rffi::llexternal!(
    pub _setlocale = "setlocale",
    [INT, CCHARP],
    CCHARP,
    compilation_info = LOCALE_ECI,
    sandboxsafe = true
);

// `rlocale._nl_langinfo` (`sandboxsafe=True`).
#[cfg(all(
    unix,
    not(target_arch = "wasm32"),
    feature = "host_env",
    not(any(target_os = "ios", target_os = "android", target_os = "redox"))
))]
crate::rffi::llexternal!(
    pub _nl_langinfo = "nl_langinfo",
    [INT],
    CCHARP,
    compilation_info = LOCALE_ECI,
    sandboxsafe = true
);

#[cfg(all(unix, not(target_arch = "wasm32"), feature = "host_env"))]
unsafe fn charp_to_bytes(ptr: *mut libc::c_char) -> Vec<u8> {
    if ptr.is_null() {
        Vec::new()
    } else {
        unsafe { crate::rffi::charp2str(ptr.cast()) }
    }
}

#[cfg(all(unix, not(target_arch = "wasm32"), feature = "host_env"))]
fn localeconv_from_lconv(lp: *mut libc::lconv) -> LocaleConv {
    if lp.is_null() {
        return LocaleConv {
            decimal_point: b".".to_vec(),
            thousands_sep: Vec::new(),
            grouping: Vec::new(),
            int_curr_symbol: Vec::new(),
            currency_symbol: Vec::new(),
            mon_decimal_point: Vec::new(),
            mon_thousands_sep: Vec::new(),
            mon_grouping: Vec::new(),
            positive_sign: Vec::new(),
            negative_sign: Vec::new(),
            int_frac_digits: 127,
            frac_digits: 127,
            p_cs_precedes: 127,
            p_sep_by_space: 127,
            n_cs_precedes: 127,
            n_sep_by_space: 127,
            p_sign_posn: 127,
            n_sign_posn: 127,
        };
    }
    unsafe {
        LocaleConv {
            decimal_point: charp_to_bytes((*lp).decimal_point),
            thousands_sep: charp_to_bytes((*lp).thousands_sep),
            grouping: charp_to_bytes((*lp).grouping),
            int_curr_symbol: charp_to_bytes((*lp).int_curr_symbol),
            currency_symbol: charp_to_bytes((*lp).currency_symbol),
            mon_decimal_point: charp_to_bytes((*lp).mon_decimal_point),
            mon_thousands_sep: charp_to_bytes((*lp).mon_thousands_sep),
            mon_grouping: charp_to_bytes((*lp).mon_grouping),
            positive_sign: charp_to_bytes((*lp).positive_sign),
            negative_sign: charp_to_bytes((*lp).negative_sign),
            int_frac_digits: (*lp).int_frac_digits,
            frac_digits: (*lp).frac_digits,
            p_cs_precedes: (*lp).p_cs_precedes,
            p_sep_by_space: (*lp).p_sep_by_space,
            n_cs_precedes: (*lp).n_cs_precedes,
            n_sep_by_space: (*lp).n_sep_by_space,
            p_sign_posn: (*lp).p_sign_posn,
            n_sign_posn: (*lp).n_sign_posn,
        }
    }
}

/// `rlocale.py numeric_formatting` / `numeric_formatting_impl`.
///
/// The decimal point, thousands separator, and grouping bytes `localeconv`
/// reports. Without `host_env` the C locale stands in (`b"."`, empty
/// separator, empty grouping). `rlocale.py` marks `localeconv`
/// `sandboxsafe`, so a `host_env` build reads the host even under sandbox.
pub fn numeric_formatting() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    #[cfg(all(unix, not(target_arch = "wasm32"), feature = "host_env"))]
    {
        let conv = localeconv_from_lconv(unsafe { localeconv() });
        return (conv.decimal_point, conv.thousands_sep, conv.grouping);
    }
    #[cfg(all(windows, feature = "host_env"))]
    {
        return rustpython_host_env::locale::localeconv_numeric();
    }
    #[cfg(not(all(
        any(all(unix, not(target_arch = "wasm32")), windows),
        feature = "host_env"
    )))]
    (b".".to_vec(), Vec::new(), Vec::new())
}

/// `rlocale.py setlocale`. A null host result is [`LocaleError::Unsupported`].
///
/// The Windows category-range check and the encoding-length check stay with
/// the `_locale` builtin: they raise `locale.Error` before this call, and
/// the CRT invalid-parameter handler would abort on a category outside
/// `LC_ALL..=LC_TIME`.
pub fn setlocale(category: i32, locale: Option<&std::ffi::CStr>) -> Result<Vec<u8>, LocaleError> {
    #[cfg(all(unix, not(target_arch = "wasm32"), feature = "host_env"))]
    {
        let ptr = match locale {
            None => unsafe { _setlocale(category, std::ptr::null_mut()) },
            Some(locale) => unsafe { _setlocale(category, locale.as_ptr().cast_mut().cast()) },
        };
        if ptr.is_null() {
            return Err(LocaleError::Unsupported);
        }
        return Ok(unsafe { crate::rffi::charp2str(ptr) });
    }
    #[cfg(all(windows, feature = "host_env"))]
    {
        return rustpython_host_env::locale::setlocale(category, locale)
            .ok_or(LocaleError::Unsupported);
    }
    #[cfg(not(all(
        any(all(unix, not(target_arch = "wasm32")), windows),
        feature = "host_env"
    )))]
    {
        let _ = (category, locale);
        Ok(b"C".to_vec())
    }
}

/// `localeconv` fields the `_locale` builtin turns into a dict.
#[cfg(all(
    any(all(unix, not(target_arch = "wasm32")), windows),
    feature = "host_env"
))]
pub fn localeconv_data() -> LocaleConv {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    {
        localeconv_from_lconv(unsafe { localeconv() })
    }
    #[cfg(windows)]
    {
        let lc = rustpython_host_env::locale::localeconv_data();
        LocaleConv {
            decimal_point: lc.decimal_point,
            thousands_sep: lc.thousands_sep,
            grouping: lc.grouping.iter().map(|&size| size as u8).collect(),
            int_curr_symbol: lc.int_curr_symbol,
            currency_symbol: lc.currency_symbol,
            mon_decimal_point: lc.mon_decimal_point,
            mon_thousands_sep: lc.mon_thousands_sep,
            mon_grouping: lc.mon_grouping.iter().map(|&size| size as u8).collect(),
            positive_sign: lc.positive_sign,
            negative_sign: lc.negative_sign,
            int_frac_digits: lc.int_frac_digits,
            frac_digits: lc.frac_digits,
            p_cs_precedes: lc.p_cs_precedes,
            p_sep_by_space: lc.p_sep_by_space,
            n_cs_precedes: lc.n_cs_precedes,
            n_sep_by_space: lc.n_sep_by_space,
            p_sign_posn: lc.p_sign_posn,
            n_sign_posn: lc.n_sign_posn,
        }
    }
}

/// `nl_langinfo(CODESET)`.
#[cfg(all(
    unix,
    not(target_arch = "wasm32"),
    feature = "host_env",
    not(any(target_os = "ios", target_os = "android", target_os = "redox"))
))]
pub fn nl_langinfo_codeset() -> Option<Vec<u8>> {
    nl_langinfo(libc::CODESET)
}

/// `rlocale.py nl_langinfo`. `None` is a null pointer from the host.
#[cfg(all(
    unix,
    not(target_arch = "wasm32"),
    feature = "host_env",
    not(any(target_os = "ios", target_os = "android", target_os = "redox"))
))]
pub fn nl_langinfo(item: libc::nl_item) -> Option<Vec<u8>> {
    let ptr = unsafe { _nl_langinfo(item as INT) };
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { crate::rffi::charp2str(ptr) })
    }
}

/// `GetACP`, as the integer the `cp<n>` encoding name is built from.
#[cfg(all(windows, feature = "host_env"))]
pub fn acp() -> u32 {
    rustpython_host_env::locale::acp()
}

/// `rlocale.py getdefaultlocale` language or territory component.
///
/// `lctype` is `LOCALE_SISO639LANGNAME` or `LOCALE_SISO3166CTRYNAME`. The
/// `cp<n>` encoding half stays with the builtin that formats [`acp`].
#[cfg(all(windows, feature = "host_env"))]
pub fn user_default_locale_component(lctype: u32) -> Option<String> {
    rustpython_host_env::locale::locale_info(
        rustpython_host_env::locale::user_default_lcid(),
        lctype,
    )
}

/// `rlocale.py isalpha`.
#[cfg(not(target_arch = "wasm32"))]
pub fn isalpha(c: i32) -> i32 {
    unsafe { libc::isalpha(c) }
}

/// `rlocale.py isupper`.
#[cfg(not(target_arch = "wasm32"))]
pub fn isupper(c: i32) -> i32 {
    unsafe { libc::isupper(c) }
}

/// `rlocale.py toupper`.
#[cfg(not(target_arch = "wasm32"))]
pub fn toupper(c: i32) -> i32 {
    unsafe { libc::toupper(c) }
}

/// `rlocale.py islower`.
#[cfg(not(target_arch = "wasm32"))]
pub fn islower(c: i32) -> i32 {
    unsafe { libc::islower(c) }
}

/// `rlocale.py tolower`.
#[cfg(not(target_arch = "wasm32"))]
pub fn tolower(c: i32) -> i32 {
    unsafe { libc::tolower(c) }
}

/// `rlocale.py isalnum`.
#[cfg(not(target_arch = "wasm32"))]
pub fn isalnum(c: i32) -> i32 {
    unsafe { libc::isalnum(c) }
}

#[cfg(test)]
fn defining_module() -> &'static str {
    module_path!()
}

#[cfg(test)]
mod tests {
    #[cfg(not(all(any(unix, windows), feature = "host_env")))]
    #[test]
    fn numeric_formatting_stands_in_for_the_c_locale() {
        let (decimal, sep, grouping) = super::numeric_formatting();
        assert_eq!(decimal, b".");
        assert!(sep.is_empty());
        assert!(grouping.is_empty());
    }

    #[test]
    fn defining_module_is_the_policy_key() {
        assert_eq!(super::defining_module(), "majit_rlib::rlocale");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn ctype_classifies_ascii() {
        assert_ne!(super::isalpha(i32::from(b'A')), 0);
        assert_ne!(super::isupper(i32::from(b'A')), 0);
        assert_eq!(super::toupper(i32::from(b'a')), i32::from(b'A'));
        assert_ne!(super::islower(i32::from(b'z')), 0);
        assert_eq!(super::tolower(i32::from(b'Z')), i32::from(b'z'));
        assert_ne!(super::isalnum(i32::from(b'7')), 0);
        assert_eq!(super::isalpha(i32::from(b'7')), 0);
    }
}
