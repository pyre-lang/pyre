//! `rpython/rlib/rlocale.py`.
//!
//! `numeric_formatting` is the entry the number formatter reads. It shares
//! the `localeconv` walk with `_locale.localeconv`, so the grouping
//! `format(x, 'n')` uses and the grouping `locale.localeconv()` reports
//! come from the same read.
//!
//! `setlocale` raises [`LocaleError`] when the host refuses the setting.
//! The `_locale` builtin maps that error onto `locale.Error`.

/// `rlocale.py LocaleError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocaleError {
    /// `setlocale` returned a null pointer.
    Unsupported,
}

/// `rlocale.py numeric_formatting` / `numeric_formatting_impl`.
///
/// The decimal point, thousands separator, and grouping bytes `localeconv`
/// reports. Without `host_env` the C locale stands in (`b"."`, empty
/// separator, empty grouping). `rlocale.py` marks `localeconv`
/// `sandboxsafe`, so a `host_env` build reads the host even under sandbox.
pub fn numeric_formatting() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    #[cfg(all(any(unix, windows), feature = "host_env"))]
    {
        return rustpython_host_env::locale::localeconv_numeric();
    }
    #[cfg(not(all(any(unix, windows), feature = "host_env")))]
    (b".".to_vec(), Vec::new(), Vec::new())
}

/// `rlocale.py setlocale`. A null host result is [`LocaleError::Unsupported`].
///
/// The Windows category-range check and the encoding-length check stay with
/// the `_locale` builtin: they raise `locale.Error` before this call, and
/// the CRT invalid-parameter handler would abort on a category outside
/// `LC_ALL..=LC_TIME`.
pub fn setlocale(category: i32, locale: Option<&std::ffi::CStr>) -> Result<Vec<u8>, LocaleError> {
    #[cfg(all(any(unix, windows), feature = "host_env"))]
    {
        return rustpython_host_env::locale::setlocale(category, locale)
            .ok_or(LocaleError::Unsupported);
    }
    #[cfg(not(all(any(unix, windows), feature = "host_env")))]
    {
        let _ = (category, locale);
        Ok(b"C".to_vec())
    }
}

/// `localeconv` fields the `_locale` builtin turns into a dict.
///
/// Grouping bytes stop at `0` or `CHAR_MAX` inside the host copy. The
/// trailing `0` `_w_copy_grouping` appends belongs to the Python list, not
/// to this walk.
#[cfg(all(any(unix, windows), feature = "host_env"))]
pub fn localeconv_data() -> rustpython_host_env::locale::LocaleConv {
    rustpython_host_env::locale::localeconv_data()
}

/// `nl_langinfo(CODESET)`.
#[cfg(all(
    unix,
    feature = "host_env",
    not(any(target_os = "ios", target_os = "android", target_os = "redox"))
))]
pub fn nl_langinfo_codeset() -> Option<Vec<u8>> {
    rustpython_host_env::locale::nl_langinfo_codeset()
}

/// `rlocale.py nl_langinfo`. `None` is a null pointer from the host.
#[cfg(all(
    not(target_arch = "wasm32"),
    unix,
    not(any(target_os = "ios", target_os = "android", target_os = "redox"))
))]
pub fn nl_langinfo(item: libc::nl_item) -> Option<Vec<u8>> {
    let ptr = unsafe { libc::nl_langinfo(item) };
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { std::ffi::CStr::from_ptr(ptr) }.to_bytes().to_vec())
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
