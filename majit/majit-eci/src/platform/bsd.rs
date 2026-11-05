//! Port of `rpython/translator/platform/bsd.py` (`BSD`).

use std::ffi::OsString;
use std::path::PathBuf;

use super::Platform;

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

/// `BSD._include_dirs_for_libffi`.
pub(super) fn _include_dirs_for_libffi(
    _plat: &Platform,
) -> Result<Vec<OsString>, super::posix::ValueError> {
    Ok(_include_dirs_for_libffi_from(
        localbase_or("/usr/local").into_os_string(),
    ))
}

/// `os.path.join(LOCALBASE, "include")` (`BSD._include_dirs_for_libffi`).
fn _include_dirs_for_libffi_from(localbase: OsString) -> Vec<OsString> {
    vec![PathBuf::from(localbase).join("include").into_os_string()]
}

/// `BSD._library_dirs_for_libffi`.
pub(super) fn _library_dirs_for_libffi(
    _plat: &Platform,
) -> Result<Vec<OsString>, super::posix::ValueError> {
    Ok(_library_dirs_for_libffi_from(
        localbase_or("/usr/local").into_os_string(),
    ))
}

/// `os.path.join(LOCALBASE, "lib")` (`BSD._library_dirs_for_libffi`).
fn _library_dirs_for_libffi_from(localbase: OsString) -> Vec<OsString> {
    vec![PathBuf::from(localbase).join("lib").into_os_string()]
}

pub(super) fn localbase_or(default: &str) -> PathBuf {
    PathBuf::from(std::env::var_os("LOCALBASE").unwrap_or_else(|| OsString::from(default)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bsd_libffi_include_joins_empty_and_trailing_slash() {
        assert_eq!(
            _include_dirs_for_libffi_from(OsString::from("")),
            vec![OsString::from("include")]
        );
        assert_eq!(
            _include_dirs_for_libffi_from(OsString::from("/usr/local/")),
            vec![OsString::from("/usr/local/include")]
        );
    }
}
