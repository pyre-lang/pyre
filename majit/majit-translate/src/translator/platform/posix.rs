//! Port of `rpython/translator/platform/posix.py` (`BasePosix`).

use std::ffi::OsString;
use std::path::{Path, PathBuf};

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
    debug_assert!(
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
    debug_assert!(
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
