//! _locale implementation — PyPy: pypy/module/_locale/interp_locale.py
//!
//! Verbatim move of the inline block previously in importing.rs.

// Under sandbox, name libc through the seam facade so any direct syscall call
// in this module is a compile error (only types/constants/pure fns resolve).
#[cfg(feature = "sandbox")]
use crate::host_seam::sys as libc;

/// Raise `_locale.Error` with the supplied message.  Mirrors
/// `interp_locale.py make_error`.
/// The NUL-terminated buffer a collation call is made with: wide where the
/// collation is wide, bytes elsewhere.
///
/// `PyUnicode_AsWideCharString` raises on an embedded NUL before either
/// platform's collation sees the argument, so the rejection belongs to the
/// argument rather than to whichever collation the build ends up calling.
#[cfg(windows)]
type CollationArg = Vec<u16>;
#[cfg(not(windows))]
type CollationArg = std::ffi::CString;

/// `_Py_GetLocaleEncoding` — the codeset the active `LC_CTYPE` names.
///
/// `locale.py` takes `_locale.getencoding` when the module carries it and
/// falls back to `sys.getfilesystemencoding()` when it does not, and the two
/// are different answers wherever the filesystem encoding is not the locale
/// one — every Windows host, where PEP 529 makes the filesystem utf-8.
///
/// `_io.TextIOWrapper` reads it as well, for the `encoding="locale"` its
/// unspecified argument resolves to.
#[cfg(all(windows, not(feature = "sandbox")))]
pub(crate) fn locale_encoding() -> String {
    // The active ANSI code page, spelled `cp<n>` whatever it is: code page
    // 65001 answers `cp65001`, not `utf-8`.
    format!("cp{}", active_acp())
}

#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
fn active_acp() -> u32 {
    majit_rlib::rlocale::acp()
}

#[cfg(all(windows, not(feature = "host_env"), not(feature = "sandbox")))]
fn active_acp() -> u32 {
    // host_env owns GetACP; without it there is no ANSI code page to ask.
    65001
}

#[cfg(all(
    unix,
    feature = "host_env",
    not(feature = "sandbox"),
    not(any(target_os = "ios", target_os = "android", target_os = "redox"))
))]
pub(crate) fn locale_encoding() -> String {
    match majit_rlib::rlocale::nl_langinfo_codeset() {
        // An empty codeset answers utf-8: `nl_langinfo` returns one on macOS
        // when the `LC_CTYPE` locale is not supported.
        Some(bytes) if !bytes.is_empty() => String::from_utf8_lossy(&bytes).into_owned(),
        _ => "utf-8".to_string(),
    }
}

#[cfg(not(any(
    all(windows, not(feature = "sandbox")),
    all(
        unix,
        feature = "host_env",
        not(feature = "sandbox"),
        not(any(target_os = "ios", target_os = "android", target_os = "redox"))
    )
)))]
pub(crate) fn locale_encoding() -> String {
    // No host locale to ask, which is the answer `_Py_FORCE_UTF8_LOCALE`
    // builds give without asking one.
    "utf-8".to_string()
}

fn collation_arg(obj: pyre_object::PyObjectRef) -> Result<CollationArg, crate::PyError> {
    let text = crate::baseobjspace::str_utf8_w(obj)?.to_string();
    #[cfg(windows)]
    {
        if text.contains('\0') {
            return Err(crate::PyError::value_error("embedded null character"));
        }
        Ok(text.encode_utf16().chain([0]).collect())
    }
    #[cfg(not(windows))]
    {
        std::ffi::CString::new(text.into_bytes())
            .map_err(|_| crate::PyError::value_error("embedded null character"))
    }
}

/// The wide collation the runtime offers.  The narrow pair runs its arguments
/// through the active code page first, so a character the page cannot spell
/// collates as whatever replaced it, which puts an accented letter after `b`
/// instead of before it.
#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
unsafe extern "C" {
    fn wcscoll(s1: *const u16, s2: *const u16) -> i32;
    fn wcsxfrm(dst: *mut u16, src: *const u16, count: usize) -> usize;
}

/// Whether the runtime's locale parser would overrun its fixed code page
/// buffer on this name.  That buffer holds 16 wide characters including the
/// terminator; a longer encoding is a debug assertion rather than a rejection,
/// so a composite `LC_ALL` name reaches `setlocale`, comes back reporting
/// success, and leaves the offending category at "C".  Answer it here instead.
#[cfg(all(windows, feature = "host_env"))]
fn windows_encoding_too_long(locale: &str) -> bool {
    locale.split(';').any(|part| {
        let name = part.rsplit_once('=').map_or(part, |(_, name)| name);
        name.split_once('.')
            .is_some_and(|(_, encoding)| encoding.len() >= 16)
    })
}

#[cfg(all(any(unix, windows), feature = "host_env"))]
fn locale_error(message: &str) -> crate::PyError {
    let cls = crate::builtins::lookup_exc_class("locale.Error")
        .or_else(|| crate::builtins::lookup_exc_class("Exception"))
        .expect("Exception must be installed");
    let _roots = pyre_object::gc_roots::push_roots();
    let cls_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(cls);
    let w_value = pyre_object::w_str_new_managed(message);
    crate::PyError::from_type_and_value(pyre_object::gc_roots::shadow_stack_get(cls_slot), w_value)
}

#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
fn windows_default_locale_component(lctype: u32) -> Option<String> {
    majit_rlib::rlocale::user_default_locale_component(lctype)
}

/// Numeric/monetary locale parameters decoded into owned buffers, the
/// `lconv` fields `localeconv` exposes (`interp_locale.py`).
struct LocaleConvData {
    decimal_point: Vec<u8>,
    thousands_sep: Vec<u8>,
    grouping: Vec<i64>,
    int_curr_symbol: Vec<u8>,
    currency_symbol: Vec<u8>,
    mon_decimal_point: Vec<u8>,
    mon_thousands_sep: Vec<u8>,
    mon_grouping: Vec<i64>,
    positive_sign: Vec<u8>,
    negative_sign: Vec<u8>,
    int_frac_digits: i64,
    frac_digits: i64,
    p_cs_precedes: i64,
    p_sep_by_space: i64,
    n_cs_precedes: i64,
    n_sep_by_space: i64,
    p_sign_posn: i64,
    n_sign_posn: i64,
}

/// Build the `localeconv()` result dict.  String fields decode the host
/// bytes via `charp2uni` (utf-8 + surrogateescape, `interp_locale.py`);
/// grouping lists already carry the trailing 0 that `_w_copy_grouping`
/// appends to a non-empty grouping (`interp_locale.py`).
fn localeconv_to_dict(c: &LocaleConvData) -> pyre_object::PyObjectRef {
    // A `dict` header moves, and each of the eighteen insertions below
    // allocates: the value, the key string `w_dict_setitem_str` builds, and
    // the dict's own storage when it grows.  A minor collection anywhere in
    // that run relocates the dict, so the word is read back out of a root slot
    // for every insertion and for the return.
    let roots = pyre_object::gc_roots::push_roots();
    let dict_slot = roots.base();
    let _ = roots.pin_root(pyre_object::w_dict_new());
    // The value arrives already built.  Call arguments evaluate left to right,
    // so reading the dict slot inline with an allocating value expression
    // would read the slot first and hand over a pre-move word.
    let put = |k: &str, value: pyre_object::PyObjectRef| {
        // `grouping` is a `list`, which moves as well, and the key string is
        // allocated after this value is handed over.
        let value_root = pyre_object::gc_roots::push_roots();
        let value_slot = value_root.base();
        let _ = value_root.pin_root(value);
        // SAFETY: the slot holds the `W_DictObject` pinned above.
        unsafe {
            pyre_object::w_dict_setitem_str(roots.get(dict_slot), k, value_root.get(value_slot));
        }
    };
    let put_str = |k: &str, b: &[u8]| put(k, crate::typedef::charp2uni(b));
    let put_int = |k: &str, v: i64| put(k, pyre_object::w_int_new(v));
    let put_grouping = |k: &str, v: &[i64]| {
        let mut items = pyre_object::gc_roots::RootedItems::new();
        for &x in v {
            items.push(pyre_object::w_int_new(x));
        }
        put(k, pyre_object::w_list_new(items.take()))
    };
    put_str("decimal_point", &c.decimal_point);
    put_str("thousands_sep", &c.thousands_sep);
    put_grouping("grouping", &c.grouping);
    put_str("int_curr_symbol", &c.int_curr_symbol);
    put_str("currency_symbol", &c.currency_symbol);
    put_str("mon_decimal_point", &c.mon_decimal_point);
    put_str("mon_thousands_sep", &c.mon_thousands_sep);
    put_grouping("mon_grouping", &c.mon_grouping);
    put_str("positive_sign", &c.positive_sign);
    put_str("negative_sign", &c.negative_sign);
    put_int("int_frac_digits", c.int_frac_digits);
    put_int("frac_digits", c.frac_digits);
    put_int("p_cs_precedes", c.p_cs_precedes);
    put_int("p_sep_by_space", c.p_sep_by_space);
    put_int("n_cs_precedes", c.n_cs_precedes);
    put_int("n_sep_by_space", c.n_sep_by_space);
    put_int("p_sign_posn", c.p_sign_posn);
    put_int("n_sign_posn", c.n_sign_posn);
    roots.get(dict_slot)
}

/// POSIX "C" locale `localeconv` defaults for builds without libc; the
/// char-typed fields default to `CHAR_MAX` (no value provided).
#[cfg(not(all(any(unix, windows), feature = "host_env")))]
fn c_locale_conv() -> LocaleConvData {
    const CHAR_MAX: i64 = 127;
    LocaleConvData {
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
        int_frac_digits: CHAR_MAX,
        frac_digits: CHAR_MAX,
        p_cs_precedes: CHAR_MAX,
        p_sep_by_space: CHAR_MAX,
        n_cs_precedes: CHAR_MAX,
        n_sep_by_space: CHAR_MAX,
        p_sign_posn: CHAR_MAX,
        n_sign_posn: CHAR_MAX,
    }
}

/// `_locale` C-extension stub — PyPy: pypy/module/_locale/.
///
/// Provides the 'C' locale defaults so locale.py's `from _locale import *`
/// succeeds and Lib/locale.py exposes working `localeconv`/`setlocale`.
/// This mirrors the `except ImportError` fallback in the stdlib's
/// `locale` module, but routed through pyre's builtin-module registry
/// so a single import succeeds.
pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), crate::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let mut ns = pyre_object::gc_roots::pin_root(ns);
    // Locale category constants sourced from libc so the values match
    // the host (Linux: LC_CTYPE=0; macOS: LC_ALL=0, LC_CTYPE=2; ...).
    // Windows has a C runtime too, and its numbering is a third one again
    // (LC_ALL=0, LC_COLLATE=1, LC_CTYPE=2); a POSIX value passed to it names
    // a different category, so it must come from libc there as well.
    #[cfg(any(unix, windows))]
    {
        #[cfg(not(feature = "host_env"))]
        use libc as host_locale;
        #[cfg(feature = "host_env")]
        use rustpython_host_env::locale as host_locale;
        crate::__pyre_store!(
            ns,
            "LC_CTYPE",
            pyre_object::w_int_new(host_locale::LC_CTYPE as i64)
        );
        crate::__pyre_store!(
            ns,
            "LC_NUMERIC",
            pyre_object::w_int_new(host_locale::LC_NUMERIC as i64)
        );
        crate::__pyre_store!(
            ns,
            "LC_TIME",
            pyre_object::w_int_new(host_locale::LC_TIME as i64)
        );
        crate::__pyre_store!(
            ns,
            "LC_COLLATE",
            pyre_object::w_int_new(host_locale::LC_COLLATE as i64)
        );
        crate::__pyre_store!(
            ns,
            "LC_MONETARY",
            pyre_object::w_int_new(host_locale::LC_MONETARY as i64)
        );
        crate::__pyre_store!(
            ns,
            "LC_ALL",
            pyre_object::w_int_new(host_locale::LC_ALL as i64)
        );
    }
    // `LC_MESSAGES` is a POSIX category the MSVC CRT has no counterpart for.
    // `locale.py` appends it to `__all__` only if the name survived
    // its `from _locale import *`, so publishing it here on Windows puts a
    // category into `from locale import *` that no call can be made with.
    #[cfg(all(
        unix,
        feature = "host_env",
        not(any(target_os = "ios", target_os = "redox"))
    ))]
    crate::__pyre_store!(
        ns,
        "LC_MESSAGES",
        pyre_object::w_int_new(rustpython_host_env::locale::LC_MESSAGES as i64)
    );
    #[cfg(all(
        unix,
        not(all(feature = "host_env", not(any(target_os = "ios", target_os = "redox"))))
    ))]
    crate::__pyre_store!(
        ns,
        "LC_MESSAGES",
        pyre_object::w_int_new(libc::LC_MESSAGES as i64)
    );
    #[cfg(not(any(unix, windows)))]
    {
        crate::__pyre_store!(ns, "LC_CTYPE", pyre_object::w_int_new(0));
        crate::__pyre_store!(ns, "LC_NUMERIC", pyre_object::w_int_new(1));
        crate::__pyre_store!(ns, "LC_TIME", pyre_object::w_int_new(2));
        crate::__pyre_store!(ns, "LC_COLLATE", pyre_object::w_int_new(3));
        crate::__pyre_store!(ns, "LC_MONETARY", pyre_object::w_int_new(4));
        crate::__pyre_store!(ns, "LC_MESSAGES", pyre_object::w_int_new(5));
        crate::__pyre_store!(ns, "LC_ALL", pyre_object::w_int_new(6));
    }
    crate::__pyre_store!(ns, "CHAR_MAX", pyre_object::w_int_new(127));
    #[cfg(all(
        unix,
        not(any(target_os = "ios", target_os = "android", target_os = "redox"))
    ))]
    {
        crate::__pyre_store!(ns, "CODESET", pyre_object::w_int_new(libc::CODESET as i64));
        // The rest of `_localemodule.c langinfo_constants` — the `nl_item`
        // keys `nl_langinfo` takes.  Each host numbers them for itself: the
        // BSD headers count from 0 while glibc packs a category into the high
        // bits, so the values come from `libc` rather than a shared table.
        // `YESSTR`/`NOSTR` are left out, as they are upstream.
        for (name, val) in [
            ("D_T_FMT", libc::D_T_FMT as i64),
            ("D_FMT", libc::D_FMT as i64),
            ("T_FMT", libc::T_FMT as i64),
            ("T_FMT_AMPM", libc::T_FMT_AMPM as i64),
            ("AM_STR", libc::AM_STR as i64),
            ("PM_STR", libc::PM_STR as i64),
            ("DAY_1", libc::DAY_1 as i64),
            ("DAY_2", libc::DAY_2 as i64),
            ("DAY_3", libc::DAY_3 as i64),
            ("DAY_4", libc::DAY_4 as i64),
            ("DAY_5", libc::DAY_5 as i64),
            ("DAY_6", libc::DAY_6 as i64),
            ("DAY_7", libc::DAY_7 as i64),
            ("ABDAY_1", libc::ABDAY_1 as i64),
            ("ABDAY_2", libc::ABDAY_2 as i64),
            ("ABDAY_3", libc::ABDAY_3 as i64),
            ("ABDAY_4", libc::ABDAY_4 as i64),
            ("ABDAY_5", libc::ABDAY_5 as i64),
            ("ABDAY_6", libc::ABDAY_6 as i64),
            ("ABDAY_7", libc::ABDAY_7 as i64),
            ("MON_1", libc::MON_1 as i64),
            ("MON_2", libc::MON_2 as i64),
            ("MON_3", libc::MON_3 as i64),
            ("MON_4", libc::MON_4 as i64),
            ("MON_5", libc::MON_5 as i64),
            ("MON_6", libc::MON_6 as i64),
            ("MON_7", libc::MON_7 as i64),
            ("MON_8", libc::MON_8 as i64),
            ("MON_9", libc::MON_9 as i64),
            ("MON_10", libc::MON_10 as i64),
            ("MON_11", libc::MON_11 as i64),
            ("MON_12", libc::MON_12 as i64),
            ("ABMON_1", libc::ABMON_1 as i64),
            ("ABMON_2", libc::ABMON_2 as i64),
            ("ABMON_3", libc::ABMON_3 as i64),
            ("ABMON_4", libc::ABMON_4 as i64),
            ("ABMON_5", libc::ABMON_5 as i64),
            ("ABMON_6", libc::ABMON_6 as i64),
            ("ABMON_7", libc::ABMON_7 as i64),
            ("ABMON_8", libc::ABMON_8 as i64),
            ("ABMON_9", libc::ABMON_9 as i64),
            ("ABMON_10", libc::ABMON_10 as i64),
            ("ABMON_11", libc::ABMON_11 as i64),
            ("ABMON_12", libc::ABMON_12 as i64),
            ("ERA", libc::ERA as i64),
            ("ERA_D_FMT", libc::ERA_D_FMT as i64),
            ("ERA_D_T_FMT", libc::ERA_D_T_FMT as i64),
            ("ERA_T_FMT", libc::ERA_T_FMT as i64),
            ("ALT_DIGITS", libc::ALT_DIGITS as i64),
            ("RADIXCHAR", libc::RADIXCHAR as i64),
            ("THOUSEP", libc::THOUSEP as i64),
            ("YESEXPR", libc::YESEXPR as i64),
            ("NOEXPR", libc::NOEXPR as i64),
            ("CRNCYSTR", libc::CRNCYSTR as i64),
        ] {
            crate::__pyre_store!(ns, name, pyre_object::w_int_new(val));
        }
    }
    // `interp_locale.py W_Error = _new_exception('Error', W_Exception, 'locale error')`
    let mut exception_base = crate::builtins::lookup_exc_class("Exception")
        .expect("Exception must be installed before _locale init");
    let mut w_error = pyre_object::with_roots!(ns, exception_base => crate::builtins::new_exception_class(
        "locale.Error",
        crate::builtins::exc_exception_new,
        exception_base,
    ));
    crate::__pyre_store!(ns, "Error", w_error);

    // `_localemodule.c:_locale._getdefaultlocale` — this compatibility hook
    // is Windows-only.  The locale name is built from the user's ISO
    // language and territory and reports the active ANSI code page separately.
    // PyPy publishes the same two-item shape from `getdefaultlocale`.
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    crate::__pyre_store!(
        ns,
        "_getdefaultlocale",
        crate::make_builtin_function_with_arity(
            "_getdefaultlocale",
            |_| {
                use rustpython_host_env::locale::{
                    LOCALE_SISO639LANGNAME, LOCALE_SISO3166CTRYNAME,
                };
                let language = windows_default_locale_component(LOCALE_SISO639LANGNAME);
                let territory = windows_default_locale_component(LOCALE_SISO3166CTRYNAME);
                let mut fields = pyre_object::gc_roots::RootedItems::new();
                let locale = match (language, territory) {
                    (Some(language), Some(territory)) => {
                        pyre_object::w_str_new_managed(&format!("{language}_{territory}"))
                    }
                    _ => pyre_object::w_none(),
                };
                fields.push(locale);
                // The active ANSI code page, spelled `cp<n>` whatever it is:
                // code page 65001 answers `cp65001`, not `utf-8`.  Mapping it
                // to a codec name is `locale.getdefaultlocale`'s job, not this
                // hook's.
                let codepage = active_acp();
                fields.push(pyre_object::w_str_new_managed(&format!("cp{codepage}")));
                Ok(pyre_object::w_tuple_new(fields.take()))
            },
            0,
        )
    );

    // localeconv() — numeric/monetary parameters of the current locale.
    #[cfg(not(feature = "sandbox"))]
    crate::__pyre_store!(
        ns,
        "localeconv",
        crate::make_builtin_function_with_arity(
            "localeconv",
            |_| {
                #[cfg(all(any(unix, windows), feature = "host_env"))]
                {
                    let lc = majit_rlib::rlocale::localeconv_data();
                    // `_w_copy_grouping` (`interp_locale.py`): every byte
                    // of the C grouping string up to its NUL is one group size
                    // (a `CHAR_MAX` element stays `127`), then a trailing `0`
                    // is appended to a non-empty list. The trailing `0` is
                    // this function's own fixup and belongs to the app-level
                    // list, never to the grouping the formatter walks.
                    let grouping_of = |sizes: Vec<u8>| -> Vec<i64> {
                        let mut v: Vec<i64> = sizes.into_iter().map(i64::from).collect();
                        if !v.is_empty() {
                            v.push(0);
                        }
                        v
                    };
                    // The host copy retains CHAR_MAX. The Python adapter
                    // still owns PyPy's trailing-zero list convention.
                    let (grouping, mon_grouping) = (
                        grouping_of(lc.grouping.iter().map(|&size| size as u8).collect()),
                        grouping_of(lc.mon_grouping.iter().map(|&size| size as u8).collect()),
                    );
                    let data = LocaleConvData {
                        decimal_point: lc.decimal_point.clone(),
                        thousands_sep: lc.thousands_sep.clone(),
                        grouping,
                        int_curr_symbol: lc.int_curr_symbol.clone(),
                        currency_symbol: lc.currency_symbol.clone(),
                        mon_decimal_point: lc.mon_decimal_point.clone(),
                        mon_thousands_sep: lc.mon_thousands_sep.clone(),
                        mon_grouping,
                        positive_sign: lc.positive_sign.clone(),
                        negative_sign: lc.negative_sign.clone(),
                        int_frac_digits: lc.int_frac_digits as i64,
                        frac_digits: lc.frac_digits as i64,
                        p_cs_precedes: lc.p_cs_precedes as i64,
                        p_sep_by_space: lc.p_sep_by_space as i64,
                        n_cs_precedes: lc.n_cs_precedes as i64,
                        n_sep_by_space: lc.n_sep_by_space as i64,
                        p_sign_posn: lc.p_sign_posn as i64,
                        n_sign_posn: lc.n_sign_posn as i64,
                    };
                    Ok(localeconv_to_dict(&data))
                }
                #[cfg(not(all(any(unix, windows), feature = "host_env")))]
                {
                    Ok(localeconv_to_dict(&c_locale_conv()))
                }
            },
            0,
        )
    );
    // setlocale() mutates/reads the host locale (and $LANG/$LC_*).
    #[cfg(not(feature = "sandbox"))]
    crate::__pyre_store!(
        ns,
        "setlocale",
        crate::make_builtin_function("setlocale", |args| {
            // Argument contract is shared between host_env and the
            // no-libc fallback so invalid calls always raise instead of
            // silently echoing "C".
            if args.is_empty() || args.len() > 2 {
                return Err(crate::PyError::type_error(
                    "setlocale() takes 1 or 2 arguments",
                ));
            }
            if !unsafe { pyre_object::is_int(args[0]) } {
                return Err(crate::PyError::type_error(
                    "setlocale: category must be an integer",
                ));
            }
            // `args` is the gateway's native copy; `str_utf8_w` can collect.
            let mut w_category = args[0];
            let locale_str: Option<String> =
                if args.len() >= 2 && !unsafe { pyre_object::is_none(args[1]) } {
                    if !unsafe { pyre_object::is_str(args[1]) } {
                        return Err(crate::PyError::type_error(
                            "setlocale: locale must be a string or None",
                        ));
                    }
                    let text = pyre_object::with_roots!(
                        w_category => crate::baseobjspace::str_utf8_w(args[1])
                    )?;
                    Some(text.to_string())
                } else {
                    None
                };
            #[cfg(all(any(unix, windows), feature = "host_env"))]
            {
                let cat = (unsafe { pyre_object::w_int_get_value(w_category) }) as i32;
                // The MSVC runtime reports a category outside its own
                // `LC_MIN..=LC_MAX` through the invalid parameter handler,
                // whose default action ends the process before `setlocale`
                // can return.  Answer the range here so the call raises
                // instead.  That pair is `LC_ALL..=LC_TIME`, the numbering
                // the constants above are published from.
                #[cfg(windows)]
                if !(rustpython_host_env::locale::LC_ALL..=rustpython_host_env::locale::LC_TIME)
                    .contains(&cat)
                {
                    return Err(locale_error("invalid locale category"));
                }
                #[cfg(windows)]
                if locale_str.as_deref().is_some_and(windows_encoding_too_long) {
                    return Err(locale_error("unsupported locale setting"));
                }
                let c_locale = match locale_str.as_ref() {
                    Some(s) => Some(
                        std::ffi::CString::new(s.as_bytes())
                            .map_err(|_| crate::PyError::value_error("embedded null character"))?,
                    ),
                    None => None,
                };
                match majit_rlib::rlocale::setlocale(cat, c_locale.as_deref()) {
                    Ok(bytes) => Ok(pyre_object::w_str_new_managed(&String::from_utf8_lossy(
                        &bytes,
                    ))),
                    Err(majit_rlib::rlocale::LocaleError::Unsupported) => {
                        Err(locale_error("unsupported locale setting"))
                    }
                }
            }
            #[cfg(not(all(any(unix, windows), feature = "host_env")))]
            {
                // No libc available — every valid call resolves to the
                // POSIX "C" locale.  `locale_str` is dropped on purpose.
                let _ = (locale_str, w_category);
                Ok(pyre_object::w_str_new_managed("C"))
            }
        })
    );
    // nl_langinfo() reads the active-locale codeset/DB.  `moduledef.py`
    // publishes the name only where the host has langinfo, which is the
    // condition its `nl_item` constants are registered under above.
    #[cfg(all(
        not(feature = "sandbox"),
        unix,
        not(any(target_os = "ios", target_os = "android", target_os = "redox"))
    ))]
    crate::__pyre_store!(
        ns,
        "nl_langinfo",
        crate::make_builtin_function_with_arity(
            "nl_langinfo",
            |args| {
                #[cfg(all(
                    unix,
                    feature = "host_env",
                    not(any(target_os = "ios", target_os = "android", target_os = "redox"))
                ))]
                {
                    let item = if args.is_empty() {
                        libc::CODESET as i64
                    } else {
                        if !unsafe { pyre_object::is_int(args[0]) } {
                            return Err(crate::PyError::type_error(
                                "nl_langinfo: item must be an integer",
                            ));
                        }
                        unsafe { pyre_object::w_int_get_value(args[0]) }
                    };
                    if item == libc::CODESET as i64
                        && let Some(bytes) = majit_rlib::rlocale::nl_langinfo_codeset()
                    {
                        return Ok(crate::typedef::charp2uni(&bytes));
                    }
                    // `interp_locale.py:151-154` — unknown items raise
                    // ValueError("unsupported langinfo constant").  POSIX
                    // nl_langinfo never returns NULL for valid items, so a
                    // null return is treated as the unsupported case.
                    // `interp_locale.py:153` decodes via utf-8 + surrogateescape.
                    let Some(bytes) = majit_rlib::rlocale::nl_langinfo(item as _) else {
                        return Err(crate::PyError::value_error("unsupported langinfo constant"));
                    };
                    Ok(crate::typedef::charp2uni(&bytes))
                }
                #[cfg(not(all(
                    unix,
                    feature = "host_env",
                    not(any(target_os = "ios", target_os = "android", target_os = "redox"))
                )))]
                {
                    // No langinfo available → every constant counts as
                    // unsupported, matching `interp_locale.py:151-154`.
                    let _ = args;
                    Err(crate::PyError::value_error("unsupported langinfo constant"))
                }
            },
            1,
        )
    );
    crate::__pyre_store!(
        ns,
        "strcoll",
        crate::make_builtin_function_with_arity(
            "strcoll",
            |args| {
                if args.len() < 2
                    || !unsafe { pyre_object::is_str(args[0]) && pyre_object::is_str(args[1]) }
                {
                    return Err(crate::PyError::type_error(
                        "strcoll: arguments must be strings",
                    ));
                }
                // `args` is the gateway's native copy; `collation_arg` can
                // collect.
                let mut w_s2 = args[1];
                let c1 = pyre_object::with_roots!(w_s2 => collation_arg(args[0]))?;
                let c2 = collation_arg(w_s2)?;
                #[cfg(all(
                    any(unix, windows),
                    feature = "host_env",
                    not(feature = "sandbox")
                ))]
                {
                    #[cfg(windows)]
                    let ord = unsafe { wcscoll(c1.as_ptr(), c2.as_ptr()) } as i64;
                    #[cfg(not(windows))]
                    let ord = rustpython_host_env::locale::strcoll(&c1, &c2) as i64;
                    Ok(pyre_object::w_int_new(ord))
                }
                #[cfg(not(all(
                    any(unix, windows),
                    feature = "host_env",
                    not(feature = "sandbox")
                )))]
                {
                    // No libc collation available (or sandbox build) — fall back
                    // to lexical comparison.  Pure computation, no I/O; under
                    // sandbox this keeps the fixed "C" collation and never calls
                    // host libc collation, which would leak host LC_COLLATE.
                    let ord = match c1.cmp(&c2) {
                        std::cmp::Ordering::Less => -1,
                        std::cmp::Ordering::Equal => 0,
                        std::cmp::Ordering::Greater => 1,
                    };
                    Ok(pyre_object::w_int_new(ord))
                }
            },
            2,
        )
    );
    crate::__pyre_store!(
        ns,
        "strxfrm",
        crate::make_builtin_function_with_arity(
            "strxfrm",
            |args| {
                let s = args[0];
                if !unsafe { pyre_object::is_str(s) } {
                    return Err(crate::PyError::type_error("strxfrm() argument must be str"));
                }
                let c = collation_arg(s)?;
                #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
                {
                    // The transform is usually no longer than its argument, so
                    // start there; a return of `count` or more is the length the
                    // transform needs, with the buffer left indeterminate.
                    let mut buf = vec![0u16; c.len()];
                    let out = loop {
                        let n = unsafe { wcsxfrm(buf.as_mut_ptr(), c.as_ptr(), buf.len()) };
                        if n < buf.len() {
                            buf.truncate(n);
                            break buf;
                        }
                        buf = vec![0u16; n + 1];
                    };
                    Ok(pyre_object::w_str_from_wtf8_managed(
                        rustpython_wtf8::Wtf8Buf::from_wide(&out),
                    ))
                }
                #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
                {
                    let out = rustpython_host_env::locale::strxfrm(&c, c.as_bytes().len() + 1);
                    // `interp_locale.py` returns `space.newtext(val)` —
                    // a plain utf-8 decode (lossy), matching `setlocale`;
                    // unlike `localeconv`/`nl_langinfo` it does not apply
                    // surrogateescape.
                    Ok(pyre_object::w_str_new_managed(&String::from_utf8_lossy(
                        &out,
                    )))
                }
                #[cfg(not(all(
                    any(unix, windows),
                    feature = "host_env",
                    not(feature = "sandbox")
                )))]
                {
                    // No libc collation available (or sandbox build) — the
                    // transform is identity, keeping the fixed "C" locale and
                    // never reaching host libc strxfrm.
                    let _ = c;
                    Ok(s)
                }
            },
            1,
        )
    );
    // `_locale.getencoding` — `_Py_GetLocaleEncodingObject`.
    crate::__pyre_store!(
        ns,
        "getencoding",
        crate::make_builtin_function_with_arity(
            "getencoding",
            |_| Ok(pyre_object::w_str_new_managed(&locale_encoding())),
            0,
        )
    );
    // Upstream's sandbox module set (`pypyoption.py` `default_modules`) has
    // no `_locale`, so its entry points raise here.
    #[cfg(feature = "sandbox")]
    {
        fn locale_unavailable(
            _: &[pyre_object::PyObjectRef],
        ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
            Err(crate::host_seam::stub("this locale function"))
        }
        #[cfg(all(
            unix,
            not(any(target_os = "ios", target_os = "android", target_os = "redox"))
        ))]
        let stubbed: &[&str] = &["setlocale", "localeconv", "nl_langinfo"];
        #[cfg(not(all(
            unix,
            not(any(target_os = "ios", target_os = "android", target_os = "redox"))
        )))]
        let stubbed: &[&str] = &["setlocale", "localeconv"];
        for &name in stubbed {
            crate::__pyre_store!(
                ns,
                name,
                crate::make_builtin_function(name, locale_unavailable)
            );
        }
        // `_getdefaultlocale` reads the user's locale and the active ANSI code
        // page, so it is host state on the same terms.
        #[cfg(windows)]
        crate::__pyre_store!(
            ns,
            "_getdefaultlocale",
            crate::make_builtin_function_with_arity("_getdefaultlocale", locale_unavailable, 0)
        );
    }
    Ok(())
}
