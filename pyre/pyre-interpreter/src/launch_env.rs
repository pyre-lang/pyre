//! The launcher option block and the environment half of its resolution
//! (`config_read_env_vars`).
//!
//! `app_main.py` reads the `PYTHON*` variables itself, so the folding rules
//! belong beside the `SYS_*` statics they feed rather than in whichever
//! binary happens to own a command line. Two callers share them: the native
//! launcher, which parses an argv and then folds the process environment over
//! it; and the wasm32 embedding, which has neither — `std::env` there is
//! permanently empty, so its host installs the relevant entries through
//! [`set_launch_env`] and folds them over an all-default option block.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::LazyLock;

/// The launcher options, as parsed from a command line and then folded with
/// the environment by [`finalize`]. `app_main.py` keeps the same set in its
/// `options` dict before `sys` initialization reads them back.
#[derive(Clone, Default)]
pub struct LaunchFlags {
    pub inspect: bool,
    /// `-i` alone.  PYTHONINSPECT folds into `inspect` and stops there, and the
    /// difference is what decides whether a prompt opens over a stdin that is
    /// not a terminal: `stdin_is_interactive` is `isatty(0) || interactive`.
    pub interactive: bool,
    pub quiet: bool,
    pub no_site: bool,
    pub no_user_site: bool,
    pub ignore_environment: bool,
    pub isolated: bool,
    pub dev_mode: bool,
    /// `-X faulthandler`; `-X dev` and PYTHONFAULTHANDLER fold in during
    /// finalize.  Installs the fatal-signal handlers before user code runs, so
    /// a crash dumps the Python traceback it was in.
    pub faulthandler: bool,
    pub warn_default_encoding: bool,
    /// PYTHONLEGACYWINDOWSFSENCODING, which no command-line option spells.
    /// Read only on Windows, the only platform whose `PyPreConfig` carries it,
    /// and left false everywhere else.
    pub legacy_windows_fs_encoding: bool,
    /// CPython's PEP 528 compatibility switch.  PyPy owns standard-stream
    /// construction in `app_main.create_stdio`; on Windows the 3.14-visible
    /// contract additionally requires this environment flag to keep those
    /// streams on `_io.FileIO` instead of `_io._WindowsConsoleIO`.
    pub legacy_windows_stdio: bool,
    /// CPython 3.14 `code_debug_ranges == 0`, selected by either
    /// `-X no_debug_ranges` or the PYTHONNODEBUGRANGES presence flag.
    pub no_debug_ranges: bool,
    /// `None` until [`finalize`] resolves it; a command line that named
    /// `-X utf8` carries that value through instead.
    pub utf8_mode: Option<i64>,
    pub safe_path: bool,
    /// `-O` count on the command line; PYTHONOPTIMIZE folds in during finalize.
    pub optimize: i64,
    /// `-v` count; PYTHONVERBOSE folds in during finalize.  Read back as
    /// `sys.flags.verbose`, which is what `importlib._bootstrap`'s
    /// `_verbose_message` gates the import trace on.
    pub verbose: i64,
    /// `-d` count; PYTHONDEBUG folds in during finalize.  The parser has no
    /// debug output to emit, so the count exists to be reported.
    pub debug: i64,
    pub bytes_warning: i64,
    pub dont_write_bytecode: bool,
    /// `-X pycache_prefix[=PATH]`, three-state until [`finalize`] resolves it.
    /// `None` is an option the command line did not name, and is the only
    /// state PYTHONPYCACHEPREFIX still applies to; after finalize a `Some`
    /// always carries a non-empty path, which is what `sys.pycache_prefix`
    /// reports and what `_bootstrap_external.cache_from_source` writes under.
    ///
    /// The path keeps the host's own spelling for the same reason the option
    /// lists do: a directory name is not required to have a UTF-8 form.
    pub pycache_prefix: Option<std::ffi::OsString>,
    pub unbuffered: bool,
    /// Both option lists stay in the host's own spelling until sys module
    /// initialization decodes them, because neither is required to be text the
    /// host can spell in UTF-8: `-W $'ignore\xff'` reaches `sys.warnoptions` as
    /// `'ignore\udcff'`, and PYTHONWARNINGS carries the same bytes.
    pub warnoptions: Vec<std::ffi::OsString>,
    /// `app_main.py` passes the raw PYTHONIOENCODING value to initstdio after
    /// applying -E/-I. Keep it raw until stdio parses the optional errors part.
    pub stdio_encoding: Option<String>,
    /// Every raw `-X` value stays in a list until sys module initialization
    /// turns it into `sys._xoptions`.
    pub xoptions: Vec<std::ffi::OsString>,
}

/// Environment the launcher options resolve against, when the process
/// environment is not it. Installed once at startup; absent means read the
/// process environment through the host seam.
///
/// Values are raw bytes, not `String`: `_Py_GetEnv` tests presence on the
/// undecoded bytes, so a value that is not valid Unicode is still *set*. A
/// table of `String` could not represent that, and the name whose fold depends
/// on it — `PYTHONSAFEPATH` — would silently read as unset.
static LAUNCH_ENV: LazyLock<Mutex<Option<HashMap<String, Vec<u8>>>>> =
    LazyLock::new(|| Mutex::new(None));

/// Supply the launcher-relevant environment explicitly, for an embedding whose
/// own process environment is not the one the program should observe.
///
/// wasm32 has no environment at all — every lookup inside the module returns
/// unset regardless of the host process — so a guest resolving `-P`, `-O` or
/// PYTHONWARNINGS from it reads them all as absent. Passing the entries in
/// restores the fold. Once installed the table is authoritative: a name missing
/// from it is unset, never a fallback to the process environment.
pub fn set_launch_env(entries: impl IntoIterator<Item = (String, Vec<u8>)>) {
    *LAUNCH_ENV.lock() = Some(entries.into_iter().collect());
}

/// The names [`finalize`] reads, for a host that has to forward them to an
/// embedding with no environment of its own. Locale variables are included
/// because an unset chain resolves to the C locale, which coerces UTF-8 mode.
pub const LAUNCH_ENV_NAMES: &[&str] = &[
    "PYTHONSAFEPATH",
    "PYTHONNOUSERSITE",
    "PYTHONUNBUFFERED",
    "PYTHONDEVMODE",
    "PYTHONFAULTHANDLER",
    "PYTHONINSPECT",
    "PYTHONOPTIMIZE",
    "PYTHONVERBOSE",
    "PYTHONDEBUG",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONPYCACHEPREFIX",
    "PYTHONUTF8",
    "PYTHONWARNDEFAULTENCODING",
    "PYTHONLEGACYWINDOWSFSENCODING",
    "PYTHONLEGACYWINDOWSSTDIO",
    "PYTHONWARNINGS",
    "PYTHONIOENCODING",
    "PYTHONNODEBUGRANGES",
    "LC_ALL",
    "LC_CTYPE",
    "LANG",
];

/// The raw bytes of a variable, from the installed table when there is one and
/// otherwise from the process environment through the host seam. A seam error
/// reads as unset, the same as a name the environment does not carry.
fn read_raw(name: &str) -> Option<Vec<u8>> {
    if let Some(table) = LAUNCH_ENV.lock().as_ref() {
        return table.get(name).cloned();
    }
    crate::host_seam::getenv(name.as_bytes()).ok().flatten()
}

/// The decoded value, or `None` when unset *or* not valid Unicode — the
/// `env::var` contract. Only the free-text names resolve under it
/// (PYTHONWARNINGS, PYTHONIOENCODING and the locale chain); every name whose
/// fold turns on presence or on an exact spelling reads bytes instead, so that
/// an undecodable value stays distinguishable from an absent one.
fn read(name: &str) -> Option<String> {
    read_raw(name).and_then(|value| String::from_utf8(value).ok())
}

/// Environment bytes in the host's own spelling. The seam hands back what the
/// platform stores: bytes on unix, where any byte is legal, and the WTF-8 form
/// of the wide value on Windows, whose unpaired surrogates re-encode to the
/// code units they came from.
fn os_string_from_bytes(value: &[u8]) -> std::ffi::OsString {
    crate::gateway::os_string_from_fs_bytes(value)
}

/// Presence of a variable, without decoding it. `_Py_GetEnv` tests the raw
/// bytes, so a value that is not valid Unicode still counts as set. Both paths
/// preserve that: the seam hands back bytes, and the installed table stores
/// bytes for this reason.
fn is_set_nonempty(name: &str) -> bool {
    read_raw(name).is_some_and(|value| !value.is_empty())
}

/// Whether the effective `LC_CTYPE` is the legacy C/POSIX locale, which is what
/// coerces utf8_mode to 1; every named locale (en_US, C.UTF-8, …) leaves it 0.
fn locale_implies_utf8_mode() -> bool {
    // Windows has no such rule: `preconfig_init_utf8_mode` guards the whole
    // locale block with `#ifndef MS_WINDOWS`, so there the mode stays 0 unless
    // `-X utf8` or `PYTHONUTF8` asks for it. The variables the cascade below
    // reads are normally unset on Windows, and an unset chain resolves to C --
    // which would turn the mode on for every process on the platform whose
    // streams the mode most changes.
    if cfg!(windows) {
        return false;
    }
    // `_Py_SetLocaleFromEnv(LC_CTYPE)` and read the answer back, so the value
    // tested is the locale the C library actually installed rather than the
    // variables that asked for it. The two differ whenever the environment
    // names a locale the system cannot set: `setlocale` keeps C there, and
    // utf8_mode has to follow the installed locale, not the unusable name.
    //
    // Only when the process environment is the authority — an embedding that
    // supplied its own table is describing an environment this process does not
    // have, and `setlocale("")` would answer from the wrong one, so that path
    // keeps the string cascade below. wasm32 has no locale database at all and
    // reaches the cascade the same way.
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    if LAUNCH_ENV.lock().is_none() {
        let _ = majit_rlib::rlocale::setlocale(libc::LC_CTYPE, Some(c""));
        let effective = majit_rlib::rlocale::setlocale(libc::LC_CTYPE, None).ok();
        return matches!(effective.as_deref(), None | Some(b"C") | Some(b"POSIX"));
    }
    // An empty variable is treated as unset (`setlocale` POSIX semantics) and
    // falls through to the next, so `LC_ALL= LC_CTYPE=en_US.UTF-8` resolves to
    // en_US.UTF-8, not C.
    let read = |name: &str| read(name).filter(|v| !v.is_empty());
    let locale = read("LC_ALL")
        .or_else(|| read("LC_CTYPE"))
        .or_else(|| read("LANG"));
    // Only the legacy C/POSIX locale — or a fully unset chain, which resolves to
    // C — coerces utf8_mode to 1; every named locale (en_US, C.UTF-8, …) leaves
    // it 0.
    matches!(locale.as_deref(), None | Some("C") | Some("POSIX"))
}

/// Detail of a `preconfig_init_utf8_mode` failure, for the caller to report the
/// way its embedding reports a fatal pre-init error.
pub struct PreConfigError(pub &'static str);

fn resolve_utf8_mode(flags: &LaunchFlags) -> Result<i64, PreConfigError> {
    // `preconfig_init_utf8_mode` opens with this, ahead of every other source:
    // the legacy filesystem codec and UTF-8 mode cannot both be on, and the
    // codec turns the mode off whether `-X utf8` or `PYTHONUTF8` asked for it.
    if flags.legacy_windows_fs_encoding {
        return Ok(0);
    }
    if let Some(value) = flags.utf8_mode {
        return Ok(value);
    }
    if !flags.ignore_environment {
        // Compared as bytes: a value that does not decode is *invalid*, which is
        // fatal, not absent, which would silently fall through to the locale.
        if let Some(value) = read_raw("PYTHONUTF8")
            && !value.is_empty()
        {
            return match value.as_slice() {
                b"0" => Ok(0),
                b"1" => Ok(1),
                _ => Err(PreConfigError(
                    "invalid PYTHONUTF8 environment variable value",
                )),
            };
        }
    }
    Ok(if locale_implies_utf8_mode() { 1 } else { 0 })
}

/// `_Py_get_env_flag`: an integer environment flag. An unset or empty value is
/// absent; a clean non-negative integer is used as-is; anything else (trailing
/// junk, negative, overflow) counts as 1.
fn env_int_flag(name: &str) -> Option<u32> {
    let raw = read_raw(name)?;
    if raw.is_empty() {
        return None;
    }
    // Read the bytes, not a decoded string: `_Py_str_to_int` rejects undecodable
    // input exactly the way it rejects trailing junk, so both land on 1 rather
    // than on "unset".
    //
    // `strtol` steps over leading whitespace before the digits and then reports
    // where it stopped, so `_Py_str_to_int` takes a value padded on the left and
    // refuses one padded on the right.  Rust's parse refuses both, so the left
    // pad is trimmed here; the set is `isspace`'s, not `char::is_whitespace`'s,
    // which would also step over spacing the C function stops at.
    let parsed = std::str::from_utf8(&raw)
        .ok()
        .map(|value| value.trim_start_matches([' ', '\t', '\n', '\x0b', '\x0c', '\r']))
        .and_then(|value| value.parse::<i64>().ok());
    // `preconfig.c:554` rejects anything outside the C `int` range as
    // overflow, and an unreadable value is what becomes 1, so the ceiling is
    // INT_MAX rather than the width of the field this returns into.
    let value = match parsed {
        Some(v) if (0..=i64::from(i32::MAX)).contains(&v) => v as u32,
        _ => 1,
    };
    Some(value)
}

/// A repeatable option folded with its integer environment variable
/// (`app_main.py`'s `parse_env`, and `config_init_optimization_level` for the
/// `-O` case): the effective count is the larger of the two.  The count is kept
/// as a wide integer so `sys.flags` mirrors a large variable verbatim; the
/// compiler clamps the optimization level into a byte at read time.
fn resolve_counter(flags: &LaunchFlags, count: i64, name: &str) -> i64 {
    if flags.ignore_environment {
        return count;
    }
    match env_int_flag(name) {
        Some(value) => count.max(i64::from(value)),
        None => count,
    }
}

/// A boolean option folded with an *integer* environment flag: the variable
/// enables it for any value `_Py_get_env_flag` resolves nonzero, so an explicit
/// `NAME=0` leaves it off. The fold only ever sets, so the command-line spelling
/// survives both `-E` and a zero-valued variable.
fn fold_int_flag(flags: &LaunchFlags, set: bool, name: &str) -> bool {
    set || (!flags.ignore_environment && env_int_flag(name).is_some_and(|v| v != 0))
}

/// A boolean option folded with a *presence* environment flag, read through
/// `_Py_GetEnv` rather than `_Py_get_env_flag`: any non-empty value enables it —
/// `"0"` included — while an empty value counts as unset.
fn fold_presence_flag(flags: &LaunchFlags, set: bool, name: &str) -> bool {
    set || (!flags.ignore_environment && is_set_nonempty(name))
}

/// `app_main.py run_command_line` — `-X pycache_prefix` folded with
/// PYTHONPYCACHEPREFIX.  Naming the option at all settles the question, so
/// `-X pycache_prefix` and `-X pycache_prefix=` both leave the prefix unset
/// *and* keep the variable from supplying one; only an unnamed option reaches
/// the environment. An empty value never becomes a prefix, from either source,
/// and a value that is a relative path is stored the way it was written.
fn resolve_pycache_prefix(flags: &LaunchFlags) -> Option<std::ffi::OsString> {
    let named = match &flags.pycache_prefix {
        Some(value) => value.clone(),
        None if flags.ignore_environment => return None,
        // Read as bytes: a directory name is free text, so `read`'s `env::var`
        // contract would drop the whole variable for one undecodable byte
        // rather than carry the path `-X` would have carried.
        None => os_string_from_bytes(&read_raw("PYTHONPYCACHEPREFIX")?),
    };
    (!named.is_empty()).then_some(named)
}

/// Fold the environment over the options a command line named. An embedding
/// without a command line passes `LaunchFlags::default()`.
pub fn finalize(mut flags: LaunchFlags) -> Result<LaunchFlags, PreConfigError> {
    // Read in `preconfig_read`, so it takes the same integer fold as the
    // other variables there rather than the presence one: `=0` leaves the
    // filesystem codec on the PEP 529 pair.
    #[cfg(windows)]
    {
        flags.legacy_windows_fs_encoding = fold_int_flag(
            &flags,
            flags.legacy_windows_fs_encoding,
            "PYTHONLEGACYWINDOWSFSENCODING",
        );
        // [3.14-spec] `CmdLineTest.test_python_legacy_windows_stdio` observes
        // the raw standard-stream type in a fresh console.  PyPy's
        // `app_main.create_stdio` remains the owner of stream construction;
        // only CPython's compatibility flag is folded into that decision.
        flags.legacy_windows_stdio = fold_int_flag(
            &flags,
            flags.legacy_windows_stdio,
            "PYTHONLEGACYWINDOWSSTDIO",
        );
    }
    flags.utf8_mode = Some(resolve_utf8_mode(&flags)?);
    flags.optimize = resolve_counter(&flags, flags.optimize, "PYTHONOPTIMIZE");
    flags.verbose = resolve_counter(&flags, flags.verbose, "PYTHONVERBOSE");
    flags.debug = resolve_counter(&flags, flags.debug, "PYTHONDEBUG");
    // Which of the two shapes a name takes is per-variable, not a category:
    // PYTHONSAFEPATH and PYTHONINSPECT enable on a bare `=0`, PYTHONNOUSERSITE
    // and PYTHONUNBUFFERED do not.
    flags.dont_write_bytecode =
        fold_int_flag(&flags, flags.dont_write_bytecode, "PYTHONDONTWRITEBYTECODE");
    flags.no_user_site = fold_int_flag(&flags, flags.no_user_site, "PYTHONNOUSERSITE");
    flags.unbuffered = fold_int_flag(&flags, flags.unbuffered, "PYTHONUNBUFFERED");
    flags.safe_path = fold_presence_flag(&flags, flags.safe_path, "PYTHONSAFEPATH");
    flags.dev_mode = fold_presence_flag(&flags, flags.dev_mode, "PYTHONDEVMODE");
    // app_main.py run_command_line — the option, developer mode and the
    // variable all reach the same install, and the variable takes the presence
    // fold, so a `PYTHONFAULTHANDLER=0` enables it the way any other non-empty
    // value does.  Folded after dev_mode so the variable spelling of that one
    // carries here too.
    flags.faulthandler = fold_presence_flag(
        &flags,
        flags.faulthandler || flags.dev_mode,
        "PYTHONFAULTHANDLER",
    );
    // `interactive` is deliberately not folded: `-i` is the only spelling that
    // sets it, and `pymain_repl` is where the variable is read again anyway,
    // late enough that a program can still set it from Python.
    flags.inspect = fold_presence_flag(&flags, flags.inspect, "PYTHONINSPECT");
    // app_main.py:773-774 — the variable enables it on any non-empty value,
    // `0` included, the same way `-X warn_default_encoding` does.
    flags.warn_default_encoding = fold_presence_flag(
        &flags,
        flags.warn_default_encoding,
        "PYTHONWARNDEFAULTENCODING",
    );
    flags.no_debug_ranges =
        fold_presence_flag(&flags, flags.no_debug_ranges, "PYTHONNODEBUGRANGES");
    flags.pycache_prefix = resolve_pycache_prefix(&flags);
    flags.stdio_encoding = if flags.ignore_environment {
        None
    } else {
        read("PYTHONIOENCODING")
    };
    // pypy/interpreter/app_main.py:892-906 — lowest-precedence entries first;
    // the warnings module installs later entries ahead of earlier ones.
    let mut warnoptions: Vec<std::ffi::OsString> = Vec::new();
    if flags.dev_mode {
        warnoptions.push("default".into());
    }
    if !flags.ignore_environment {
        // Read as bytes: a filter is free text, so `read`'s `env::var`
        // contract would drop the whole variable for one undecodable byte
        // rather than carry the entry `-W` would have carried.
        if let Some(value) = read_raw("PYTHONWARNINGS")
            && !value.is_empty()
        {
            warnoptions.extend(value.split(|&b| b == b',').map(os_string_from_bytes));
        }
    }
    warnoptions.append(&mut flags.warnoptions);
    if flags.bytes_warning > 0 {
        warnoptions.push(if flags.bytes_warning > 1 {
            "error::BytesWarning".into()
        } else {
            "default::BytesWarning".into()
        });
    }
    flags.warnoptions = warnoptions;
    Ok(flags)
}
