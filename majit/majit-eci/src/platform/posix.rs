//! Port of `rpython/translator/platform/posix.py` (`BasePosix`).

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;

use super::{CompilerInvocation, prefixed_path};

pub(super) const RELEVANT_ENVIRON: &[&str] = &["CPATH", "LIBRARY_PATH", "C_INCLUDE_PATH"];

/// `BasePosix.exe_ext`.
pub(super) const EXE_EXT: &str = "";

/// `BasePosix._libs`.
pub(super) fn posix_libs(libraries: &[String]) -> Vec<OsString> {
    libraries
        .iter()
        .map(|lib| prefixed_path("-l", Path::new(lib)))
        .collect()
}

/// `BasePosix._libdirs`.
pub(super) fn posix_libdirs(library_dirs: &[PathBuf]) -> Vec<OsString> {
    assert!(
        library_dirs.iter().all(|dir| !dir.as_os_str().is_empty()),
        "empty library dir"
    );
    library_dirs
        .iter()
        .map(|dir| prefixed_path("-L", dir))
        .collect()
}

/// `BasePosix._includedirs`.
pub(super) fn posix_includedirs(include_dirs: &[PathBuf]) -> Vec<OsString> {
    assert!(
        include_dirs.iter().all(|dir| !dir.as_os_str().is_empty()),
        "empty include dir"
    );
    include_dirs
        .iter()
        .map(|dir| prefixed_path("-I", dir))
        .collect()
}

/// `BasePosix._compile_c_file` argument list (`-c` … `-o`).
pub(super) fn posix_compile_c_file_invocation(
    cc: &Path,
    cfile: &Path,
    compile_args: &[OsString],
    oname: &Path,
) -> CompilerInvocation {
    let mut args = vec![OsString::from("-c")];
    args.extend(compile_args.iter().cloned());
    args.push(cfile.as_os_str().to_os_string());
    args.push(OsString::from("-o"));
    args.push(oname.as_os_str().to_os_string());
    CompilerInvocation {
        program: cc.to_path_buf(),
        args,
        cwd: cfile
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .map(Path::to_path_buf),
    }
}

/// `BasePosix._args_for_shared`.
pub(super) fn posix_args_for_shared(args: Vec<OsString>) -> Vec<OsString> {
    let mut out = vec![OsString::from("-shared")];
    out.extend(args);
    out
}

/// `BasePosix._exportsymbols_link_flags` (non-mingw / non-cygwin).
pub(super) fn posix_export_dynamic() -> Vec<OsString> {
    vec![OsString::from("-Wl,--export-dynamic")]
}

/// `BasePosix._exportsymbols_link_flags` for mingw / gcc-on-nt / cygwin.
pub(super) fn posix_export_all_symbols() -> Vec<OsString> {
    vec![OsString::from("-Wl,--export-all-symbols")]
}

/// `BasePosix._pkg_config` raises `ValueError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueError(pub Vec<u8>);

impl fmt::Display for ValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display only; the payload stays the subprocess bytes.
        f.write_str(&String::from_utf8_lossy(&self.0))
    }
}

/// Python 2 `str.split()` whitespace bytes: space, tab, LF, CR, VT, FF.
const PY2_STR_WHITESPACE: &[u8] = &[b' ', b'\t', b'\n', b'\r', 0x0b, 0x0c];

fn is_py2_str_whitespace(b: u8) -> bool {
    PY2_STR_WHITESPACE.contains(&b)
}

fn split_py2_str(bytes: &[u8]) -> Vec<&[u8]> {
    bytes
        .split(|b| is_py2_str_whitespace(*b))
        .filter(|tok| !tok.is_empty())
        .collect()
}

fn rstrip_py2_bytes(bytes: &[u8]) -> &[u8] {
    let end = bytes
        .iter()
        .rposition(|b| !is_py2_str_whitespace(*b))
        .map(|i| i + 1)
        .unwrap_or(0);
    &bytes[..end]
}

/// Python 2 `str(OSError(errno, strerror))` is `[Errno <n>] <strerror>`.
fn py2_oserror_str(e: &std::io::Error) -> Vec<u8> {
    match e.raw_os_error() {
        Some(code) => {
            // std formats `from_raw_os_error` as `<strerror> (os error <code>)`.
            let formatted = std::io::Error::from_raw_os_error(code).to_string();
            let suffix = format!(" (os error {code})");
            let strerror = formatted
                .strip_suffix(&suffix)
                .unwrap_or(formatted.as_str());
            format!("[Errno {code}] {strerror}").into_bytes()
        }
        None => e.to_string().into_bytes(),
    }
}

/// `runsubprocess.py` replaces `_run` with the helper-process version when
/// `sys.platform != 'win32'` and `hasattr(os, 'fork')` and
/// `not os.getenv("PYPY_DONT_RUN_SUBPROCESS", None)` (empty string is false).
/// `hasattr(os, 'fork')` holds on unix only.
fn run_subprocess_uses_helper(dont_run: Option<OsString>) -> bool {
    cfg!(unix) && dont_run.is_none_or(|v| v.is_empty())
}

/// `runsubprocess.py` import-time `_run` selection; the helper spawn itself is
/// not modelled (the port runs `Command` directly), so this records only which
/// error shape `_run` raises.
static RUN_SUBPROCESS_HELPER: std::sync::LazyLock<bool> =
    LazyLock::new(|| run_subprocess_uses_helper(std::env::var_os("PYPY_DONT_RUN_SUBPROCESS")));

/// Helper-process `runsubprocess.py` `_run` raises
/// `OSError('%s: %s\nargs=%r' % (args[0], results[1], args))`;
/// `BasePosix._pkg_config` stores `err = str(e)`.
fn run_subprocess_oserror_str(
    executable: &OsStr,
    args: &[&str],
    e: &std::io::Error,
    helper: bool,
) -> Vec<u8> {
    let inner = py2_oserror_str(e);
    if !helper {
        return inner;
    }
    let exec_bytes = executable.as_encoded_bytes();
    let mut out = exec_bytes.to_vec();
    out.extend_from_slice(b": ");
    out.extend_from_slice(&inner);
    out.extend_from_slice(b"\nargs=(");
    out.extend_from_slice(py_bytes_repr(exec_bytes).as_bytes());
    out.extend_from_slice(b", [");
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            out.extend_from_slice(b", ");
        }
        out.extend_from_slice(py_bytes_repr(arg.as_bytes()).as_bytes());
    }
    out.extend_from_slice(b"], None, None)");
    out
}

fn pkg_config_failed_msg(lib: &str, opt: &str, err: &[u8], default_repr: &str) -> Vec<u8> {
    let mut msg = b"running 'pkg-config ".to_vec();
    msg.extend_from_slice(lib.as_bytes());
    msg.push(b' ');
    msg.extend_from_slice(opt.as_bytes());
    msg.extend_from_slice(b"' failed:\n");
    msg.extend_from_slice(rstrip_py2_bytes(err));
    msg.extend_from_slice(b"\nand the default ");
    msg.extend_from_slice(default_repr.as_bytes());
    msg.extend_from_slice(b" is not a valid directory");
    msg
}

fn pkg_config_no_valid_dir_msg(lib: &str, opt: &str, out: &[u8], err: &[u8]) -> Vec<u8> {
    let mut msg = b"'pkg-config ".to_vec();
    msg.extend_from_slice(lib.as_bytes());
    msg.push(b' ');
    msg.extend_from_slice(opt.as_bytes());
    msg.extend_from_slice(b"' returned no valid directory:\n");
    msg.extend_from_slice(rstrip_py2_bytes(out));
    msg.push(b'\n');
    msg.extend_from_slice(rstrip_py2_bytes(err));
    msg
}

fn os_string_from_bytes(bytes: Vec<u8>) -> OsString {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        OsString::from_vec(bytes)
    }
    #[cfg(not(unix))]
    {
        // Windows has no `_pkg_config` caller (`MingwPlatform._include_dirs_for_libffi`
        // returns `[]`; `MsvcPlatform` panics in `Platform._include_dirs_for_libffi`).
        OsString::from(String::from_utf8_lossy(&bytes).into_owned())
    }
}

/// Python 2 `str.__repr__` of a byte string.
pub(super) fn py_bytes_repr(bytes: &[u8]) -> String {
    let quote = if bytes.contains(&b'\'') && !bytes.contains(&b'"') {
        b'"'
    } else {
        b'\''
    };
    let mut out = String::new();
    out.push(quote as char);
    for &b in bytes {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'\'' if quote == b'\'' => out.push_str("\\'"),
            b'"' if quote == b'"' => out.push_str("\\\""),
            b'\t' => out.push_str("\\t"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b if b < 0x20 || b >= 0x7f => out.push_str(&format!("\\x{b:02x}")),
            _ => out.push(b as char),
        }
    }
    out.push(quote as char);
    out
}

/// Python 2 `%r` of a list of byte strings: `['/a', '/b']`.
fn py_bytes_list_repr(dirs: &[OsString]) -> String {
    let mut out = String::from("[");
    for (i, dir) in dirs.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&py_bytes_repr(dir.as_os_str().as_encoded_bytes()));
    }
    out.push(']');
    out
}

/// `BasePosix._pkg_config`.
pub(super) fn _pkg_config(
    lib: &str,
    opt: &str,
    default: Vec<OsString>,
    check_result_dir: bool,
) -> Result<Vec<OsString>, ValueError> {
    let pkg_config = std::env::var_os("PKG_CONFIG").unwrap_or_else(|| OsString::from("pkg-config"));
    let (ret, stdout, err) = match Command::new(&pkg_config).args([lib, opt]).output() {
        Ok(output) => {
            let ret = output.status.code().unwrap_or(1);
            (ret, output.stdout, output.stderr)
        }
        Err(e) => {
            // `OSError as e`: `err = str(e)`, `ret = 1`; `out` is unset.
            (
                1,
                Vec::new(),
                run_subprocess_oserror_str(&pkg_config, &[lib, opt], &e, *RUN_SUBPROCESS_HELPER),
            )
        }
    };
    let result = if ret != 0 {
        default
    } else {
        // strip compiler flags (`entry[2:]` drops the first two bytes)
        split_py2_str(&stdout)
            .into_iter()
            .map(|entry| {
                let stripped = if entry.len() >= 2 {
                    entry[2..].to_vec()
                } else {
                    Vec::new()
                };
                os_string_from_bytes(stripped)
            })
            .collect()
    };
    if result.is_empty() {
        // if pkg-config explicitly returned nothing, then
        // we assume it means no options are needed
    } else if check_result_dir {
        // check that at least one of the results is a valid dir
        let found = result.iter().any(|check| Path::new(check).is_dir());
        if !found {
            let msg = if ret != 0 {
                pkg_config_failed_msg(lib, opt, &err, &py_bytes_list_repr(&result))
            } else {
                pkg_config_no_valid_dir_msg(lib, opt, &stdout, &err)
            };
            return Err(ValueError(msg));
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_bytes_list_repr_matches_python2_bytes_list() {
        assert_eq!(py_bytes_repr(b"/a"), "'/a'");
        assert_eq!(py_bytes_repr(b"it's"), "\"it's\"");
        assert_eq!(py_bytes_repr(b"a'b\"c"), "'a\\'b\"c'");
        assert_eq!(
            py_bytes_repr(b"t\tn\nr\r\x01\xff\\"),
            r"'t\tn\nr\r\x01\xff\\'"
        );
        assert_eq!(py_bytes_list_repr(&[]), "[]");
        assert_eq!(
            py_bytes_list_repr(&[OsString::from("/a"), OsString::from("/b")]),
            "['/a', '/b']"
        );
        assert_eq!(py_bytes_list_repr(&[OsString::from("it's")]), "[\"it's\"]");
        assert_eq!(
            py_bytes_list_repr(&[OsString::from("a'b\"c")]),
            "['a\\'b\"c']"
        );
    }

    #[test]
    fn pkg_config_missing_lib_returns_default() {
        let default = vec![OsString::from("/the/default/include")];
        let result = _pkg_config(
            "majit-translate-no-such-libffi-pkg",
            "--cflags-only-I",
            default.clone(),
            false,
        )
        .expect("default without directory check");
        assert_eq!(result, default);
    }

    #[test]
    fn pkg_config_check_result_dir_accepts_existing_default() {
        let default = vec![crate::udir::udir().as_os_str().to_os_string()];
        let result = _pkg_config(
            "majit-translate-no-such-libffi-pkg",
            "--cflags-only-I",
            default.clone(),
            true,
        )
        .expect("existing default dir");
        assert_eq!(result, default);
    }

    #[test]
    fn pkg_config_check_result_dir_rejects_missing_default() {
        let default = vec![
            crate::udir::udir()
                .join("majit-translate-no-such-libffi-dir")
                .into_os_string(),
        ];
        let err = _pkg_config(
            "majit-translate-no-such-libffi-pkg",
            "--cflags-only-I",
            default,
            true,
        )
        .expect_err("missing default dir");
        assert!(
            err.to_string().contains("is not a valid directory"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn py2_oserror_str_matches_python2_oserror() {
        let bytes = py2_oserror_str(&std::io::Error::from_raw_os_error(2));
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.starts_with("[Errno 2] "), "{s}");
        assert!(!s.contains("(os error"), "{s}");
    }

    #[test]
    fn run_subprocess_oserror_str_matches_helper_and_direct() {
        use std::ffi::OsStr;
        use std::io;
        let e = io::Error::from_raw_os_error(2);
        let helper = run_subprocess_oserror_str(
            OsStr::new("pkg-config"),
            &["libffi", "--cflags-only-I"],
            &e,
            true,
        );
        let s = String::from_utf8_lossy(&helper);
        assert!(s.starts_with("pkg-config: [Errno 2] "), "{s}");
        assert!(
            s.ends_with("\nargs=('pkg-config', ['libffi', '--cflags-only-I'], None, None)"),
            "{s}"
        );
        let direct = run_subprocess_oserror_str(
            OsStr::new("pkg-config"),
            &["libffi", "--cflags-only-I"],
            &e,
            false,
        );
        assert_eq!(direct, py2_oserror_str(&e));
    }

    #[test]
    fn run_subprocess_uses_helper_matches_pypy_dont_run_subprocess() {
        assert_eq!(run_subprocess_uses_helper(Some("".into())), cfg!(unix));
        assert!(!run_subprocess_uses_helper(Some("1".into())));
        assert_eq!(run_subprocess_uses_helper(None), cfg!(unix));
    }

    #[cfg(unix)]
    #[test]
    fn pkg_config_value_error_keeps_non_utf8_err_byte() {
        let err = ValueError(pkg_config_failed_msg(
            "lib",
            "--cflags",
            b"boom\xff\n",
            "['/nope']",
        ));
        assert!(
            err.0.contains(&0xff),
            "failed msg dropped 0xff: {:?}",
            err.0
        );
        let err = ValueError(pkg_config_no_valid_dir_msg(
            "lib", "--cflags", b"/tmp", b"err\xff",
        ));
        assert!(
            err.0.contains(&0xff),
            "no-valid-dir msg dropped 0xff: {:?}",
            err.0
        );
    }
}
