//! Port of `rpython/translator/platform/bsd.py` (`BSD`).

use std::ffi::OsString;
use std::path::PathBuf;

/// `BSD._preprocess_include_dirs` (`LOCALBASE` or `/usr/local`).
pub(super) fn preprocess_include_dirs(include_dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs = include_dirs.to_vec();
    dirs.push(localbase_or("/usr/local").join("include"));
    dirs
}

/// `BSD._preprocess_library_dirs`.
pub(super) fn preprocess_library_dirs(library_dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs = library_dirs.to_vec();
    dirs.push(localbase_or("/usr/local").join("lib"));
    dirs
}

pub(super) fn localbase_or(default: &str) -> PathBuf {
    PathBuf::from(std::env::var_os("LOCALBASE").unwrap_or_else(|| OsString::from(default)))
}
