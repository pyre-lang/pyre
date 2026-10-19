//! `rpython/rlib/rlocale.py`.
//!
//! `numeric_formatting` is the entry the number formatter reads.
//! `numeric_formatting_impl` reads `decimal_point`, `thousands_sep`, and
//! `grouping` from the `lconv` `localeconv` returns.
//!
//! `setlocale` raises [`LocaleError`] when the host refuses the setting.
//! The `_locale` builtin maps that error onto `locale.Error`.
//!
//! `rlocale.external` sets `sandboxsafe=True`.

#[cfg(all(
    any(all(unix, not(target_arch = "wasm32")), windows),
    feature = "host_env"
))]
use crate::rffi::{CCHARP, INT};

/// `rlocale.py LocaleError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocaleError {
    /// `setlocale`: the category is outside `LC_MIN..=LC_MAX`
    /// ("invalid locale category").
    InvalidCategory,
    /// `setlocale` returned a null pointer ("unsupported locale setting").
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

// `CConfig._compilation_info_` on win32: no `langinfo.h` (`HAVE_LANGINFO`)
// or `libintl.h` (`HAVE_LIBINTL`), and `windows.h` for `GetLocaleInfo`.
#[cfg(windows)]
crate::rffi::external_compilation_info! {
    const LOCALE_ECI = {
        includes: ["locale.h", "limits.h", "ctype.h", "wchar.h", "windows.h"],
    };
}

/// `CConfig.lconv` (`struct lconv`).
#[cfg(all(unix, not(target_arch = "wasm32")))]
#[allow(non_camel_case_types)]
pub type lconv = libc::lconv;

/// `CConfig.lconv` (`struct lconv`), as the MSVC `locale.h` lays it out:
/// the C fields, then the wide copies of the strings.
#[cfg(windows)]
#[allow(non_camel_case_types, non_snake_case)]
#[repr(C)]
pub struct lconv {
    pub decimal_point: *mut core::ffi::c_char,
    pub thousands_sep: *mut core::ffi::c_char,
    pub grouping: *mut core::ffi::c_char,
    pub int_curr_symbol: *mut core::ffi::c_char,
    pub currency_symbol: *mut core::ffi::c_char,
    pub mon_decimal_point: *mut core::ffi::c_char,
    pub mon_thousands_sep: *mut core::ffi::c_char,
    pub mon_grouping: *mut core::ffi::c_char,
    pub positive_sign: *mut core::ffi::c_char,
    pub negative_sign: *mut core::ffi::c_char,
    pub int_frac_digits: core::ffi::c_char,
    pub frac_digits: core::ffi::c_char,
    pub p_cs_precedes: core::ffi::c_char,
    pub p_sep_by_space: core::ffi::c_char,
    pub n_cs_precedes: core::ffi::c_char,
    pub n_sep_by_space: core::ffi::c_char,
    pub p_sign_posn: core::ffi::c_char,
    pub n_sign_posn: core::ffi::c_char,
    pub _W_decimal_point: crate::rffi::CWCHARP,
    pub _W_thousands_sep: crate::rffi::CWCHARP,
    pub _W_int_curr_symbol: crate::rffi::CWCHARP,
    pub _W_currency_symbol: crate::rffi::CWCHARP,
    pub _W_mon_decimal_point: crate::rffi::CWCHARP,
    pub _W_mon_thousands_sep: crate::rffi::CWCHARP,
    pub _W_positive_sign: crate::rffi::CWCHARP,
    pub _W_negative_sign: crate::rffi::CWCHARP,
}

/// `cConfig.LC_MIN` / `cConfig.LC_MAX`. Only the MSVC `locale.h` defines
/// them (`LC_ALL` / `LC_TIME`); elsewhere they are `None`.
#[cfg(windows)]
pub const LC_MIN: crate::rffi::INT = libc::LC_ALL;
#[cfg(windows)]
pub const LC_MAX: crate::rffi::INT = libc::LC_TIME;

// `rlocale.localeconv` (`sandboxsafe=True`).
#[cfg(all(
    any(all(unix, not(target_arch = "wasm32")), windows),
    feature = "host_env"
))]
crate::rffi::llexternal!(
    pub localeconv = "localeconv",
    [],
    *mut lconv,
    compilation_info = LOCALE_ECI,
    sandboxsafe = true
);

// `rlocale._setlocale` (`sandboxsafe=True`).
#[cfg(all(
    any(all(unix, not(target_arch = "wasm32")), windows),
    feature = "host_env"
))]
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
    not(any(target_os = "android", target_os = "redox"))
))]
crate::rffi::llexternal!(
    pub _nl_langinfo = "nl_langinfo",
    [INT],
    CCHARP,
    compilation_info = LOCALE_ECI,
    sandboxsafe = true
);

#[cfg(all(
    any(all(unix, not(target_arch = "wasm32")), windows),
    feature = "host_env"
))]
unsafe fn charp_to_bytes(ptr: *mut core::ffi::c_char) -> Vec<u8> {
    if ptr.is_null() {
        Vec::new()
    } else {
        unsafe { crate::rffi::charp2str(ptr.cast()) }
    }
}

#[cfg(all(
    any(all(unix, not(target_arch = "wasm32")), windows),
    feature = "host_env"
))]
fn localeconv_from_lconv(lp: *mut lconv) -> LocaleConv {
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
    #[cfg(all(
        any(all(unix, not(target_arch = "wasm32")), windows),
        feature = "host_env"
    ))]
    {
        // `numeric_formatting_impl` reads only these three `lconv` fields.
        let lp = unsafe { localeconv() };
        if lp.is_null() {
            return (b".".to_vec(), Vec::new(), Vec::new());
        }
        unsafe {
            (
                charp_to_bytes((*lp).decimal_point),
                charp_to_bytes((*lp).thousands_sep),
                charp_to_bytes((*lp).grouping),
            )
        }
    }
    #[cfg(not(all(
        any(all(unix, not(target_arch = "wasm32")), windows),
        feature = "host_env"
    )))]
    (b".".to_vec(), Vec::new(), Vec::new())
}

/// `rlocale.py setlocale`. A category outside `LC_MIN..=LC_MAX` is
/// [`LocaleError::InvalidCategory`] (the MSVC runtime's invalid-parameter
/// handler would end the process on one); a null host result is
/// [`LocaleError::Unsupported`].
pub fn setlocale(category: i32, locale: Option<&std::ffi::CStr>) -> Result<Vec<u8>, LocaleError> {
    #[cfg(windows)]
    if !(LC_MIN..=LC_MAX).contains(&category) {
        return Err(LocaleError::InvalidCategory);
    }
    #[cfg(all(
        any(all(unix, not(target_arch = "wasm32")), windows),
        feature = "host_env"
    ))]
    {
        let ptr = match locale {
            None => unsafe { _setlocale(category, std::ptr::null_mut()) },
            Some(locale) => unsafe { _setlocale(category, locale.as_ptr().cast_mut().cast()) },
        };
        if ptr.is_null() {
            return Err(LocaleError::Unsupported);
        }
        Ok(unsafe { crate::rffi::charp2str(ptr) })
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
    localeconv_from_lconv(unsafe { localeconv() })
}

/// `nl_langinfo(CODESET)`.
#[cfg(all(
    unix,
    not(target_arch = "wasm32"),
    feature = "host_env",
    not(any(target_os = "android", target_os = "redox"))
))]
pub fn nl_langinfo_codeset() -> Option<Vec<u8>> {
    nl_langinfo(libc::CODESET)
}

/// `rlocale.py nl_langinfo`. `None` is a null pointer from the host.
#[cfg(all(
    unix,
    not(target_arch = "wasm32"),
    feature = "host_env",
    not(any(target_os = "android", target_os = "redox"))
))]
pub fn nl_langinfo(item: libc::nl_item) -> Option<Vec<u8>> {
    let ptr = unsafe { _nl_langinfo(item as INT) };
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { crate::rffi::charp2str(ptr) })
    }
}

// `getdefaultlocale()` implementation for Windows.

/// `rlocale.LCID`.
#[cfg(windows)]
pub type LCID = crate::rwin32::DWORD;
/// `rlocale.LCTYPE`.
#[cfg(windows)]
pub type LCTYPE = crate::rwin32::DWORD;

/// `cConfig.LOCALE_USER_DEFAULT` — `MAKELCID(LANG_USER_DEFAULT, SORT_DEFAULT)`.
#[cfg(windows)]
pub const LOCALE_USER_DEFAULT: LCID = 0x0400;
/// `cConfig.LOCALE_SISO639LANGNAME`.
#[cfg(windows)]
pub const LOCALE_SISO639LANGNAME: LCTYPE = 0x0059;
/// `cConfig.LOCALE_SISO3166CTRYNAME`.
#[cfg(windows)]
pub const LOCALE_SISO3166CTRYNAME: LCTYPE = 0x005A;
/// `cConfig.LOCALE_IDEFAULTLANGUAGE`.
#[cfg(windows)]
pub const LOCALE_IDEFAULTLANGUAGE: LCTYPE = 0x0009;

// `rlocale.GetACP` (`calling_conv='win'`).
#[cfg(all(windows, feature = "host_env"))]
crate::rffi::llexternal!(
    pub GetACP = "GetACP",
    [],
    INT,
    compilation_info = LOCALE_ECI,
    calling_conv = "win",
    sandboxsafe = true
);

// `rlocale.GetLocaleInfo` (`calling_conv='win'`).
#[cfg(all(windows, feature = "host_env"))]
crate::rffi::llexternal!(
    pub GetLocaleInfo = "GetLocaleInfoA",
    [LCID, LCTYPE, crate::rwin32::LPSTR, INT],
    INT,
    compilation_info = LOCALE_ECI,
    calling_conv = "win",
    sandboxsafe = true
);

/// `rlocale.py getdefaultlocale` — `(language, encoding)`; `language` is
/// `None` when no locale name can be built.
#[cfg(all(windows, feature = "host_env"))]
pub fn getdefaultlocale() -> (Option<Vec<u8>>, Vec<u8>) {
    let encoding = format!("cp{}", unsafe { GetACP() }).into_bytes();

    const BUFSIZE: usize = 50;
    let mut buf_lang = [0u8; BUFSIZE];
    let mut buf_country = [0u8; BUFSIZE];

    let language = if unsafe {
        GetLocaleInfo(
            LOCALE_USER_DEFAULT,
            LOCALE_SISO639LANGNAME,
            buf_lang.as_mut_ptr(),
            BUFSIZE as INT,
        ) != 0
            && GetLocaleInfo(
                LOCALE_USER_DEFAULT,
                LOCALE_SISO3166CTRYNAME,
                buf_country.as_mut_ptr(),
                BUFSIZE as INT,
            ) != 0
    } {
        let lang = unsafe { crate::rffi::charp2str(buf_lang.as_mut_ptr()) };
        let country = unsafe { crate::rffi::charp2str(buf_country.as_mut_ptr()) };
        Some([&lang[..], b"_", &country[..]].concat())
    } else if unsafe {
        // If we end up here, this windows version didn't know about
        // ISO639/ISO3166 names (it's probably Windows 95).  Return the
        // Windows language identifier instead (a hexadecimal number)
        GetLocaleInfo(
            LOCALE_USER_DEFAULT,
            LOCALE_IDEFAULTLANGUAGE,
            buf_lang.as_mut_ptr(),
            BUFSIZE as INT,
        ) != 0
    } {
        let lang = unsafe { crate::rffi::charp2str(buf_lang.as_mut_ptr()) };
        Some([&b"0x"[..], &lang[..]].concat())
    } else {
        None
    };

    (language, encoding)
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
    #[cfg(not(all(
        any(all(unix, not(target_arch = "wasm32")), windows),
        feature = "host_env"
    )))]
    #[test]
    fn numeric_formatting_stands_in_for_the_c_locale() {
        let (decimal, sep, grouping) = super::numeric_formatting();
        assert_eq!(decimal, b".");
        assert!(sep.is_empty());
        assert!(grouping.is_empty());
    }

    #[cfg(all(windows, feature = "host_env"))]
    #[test]
    fn windows_lconv_getdefaultlocale_and_category_range() {
        let (language, encoding) = super::getdefaultlocale();
        assert!(encoding.starts_with(b"cp"));
        assert!(language.is_some_and(|name| !name.is_empty()));
        // The process starts in the "C" locale.
        let lc = super::localeconv_data();
        assert_eq!(lc.decimal_point, b".");
        assert_eq!(super::numeric_formatting().0, b".");
        assert_eq!(
            super::setlocale(super::LC_MAX + 1, None).unwrap_err(),
            super::LocaleError::InvalidCategory
        );
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
