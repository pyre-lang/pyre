//! Port of `rpython/tool/gcc_cache.py`.
//!
//! `cache_file_path` md5s source contents + `eci` + `platform.key()` under
//! `CACHE_DIR/<cachename>` (`translationoption.cache_dir`).
//! `build_executable_cache` hits return the stored stdout; misses compile
//! + execute, echo stderr, cache only when stderr is empty, and return
//! stdout. `try_compile_cache` caches `"True"` or the compiler error
//! text and does not execute. `try_atomic_write` is the pid-suffixed
//! temp file + rename.

use md5::{Digest, Md5};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use crate::config::translationoption::cache_dir;
use crate::translator::platform::{CompilationError, platform};
use crate::translator::tool::cbuild::ExternalCompilationInfo;

/// `cache_file_path(c_files, eci, cachename)`.
pub fn cache_file_path(
    c_files: &[impl AsRef<Path>],
    eci: &ExternalCompilationInfo,
    cachename: &str,
) -> Result<PathBuf, CompilationError> {
    cache_file_path_with_key(c_files, eci, cachename, &platform().key())
}

/// `cache_file_path` with an injected identity in place of `platform.key()`.
///
/// The rustc sizeof probe (`rfficache.sizeof_c_types_via_rustc`) is a
/// cross-target adaptation: the target has no C toolchain, so the cache
/// key uses a rustc identity instead of the host `Platform.key`.
pub(crate) fn cache_file_path_with_key(
    c_files: &[impl AsRef<Path>],
    eci: &ExternalCompilationInfo,
    cachename: &str,
    platform_key: &str,
) -> Result<PathBuf, CompilationError> {
    let cache_dir = cache_dir_for(cachename)?;
    let mut hasher = Md5::new();
    hasher.update((c_files.len() as u64).to_le_bytes());
    for c_file in c_files {
        let contents = fs::read(c_file.as_ref()).map_err(|err| CompilationError {
            out: String::new(),
            err: format!(
                "gcc_cache.cache_file_path: read {}: {err}",
                c_file.as_ref().display()
            ),
        })?;
        hash_bytes(&mut hasher, &contents);
    }
    hash_eci(&mut hasher, eci);
    hash_bytes(&mut hasher, platform_key.as_bytes());
    Ok(cache_dir.join(format!("{:x}", hasher.finalize())))
}

/// `build_executable_cache(c_files, eci, ignore_errors=False)`.
///
/// `ignore_errors` sets `Platform.log_errors = False` for the compile
/// and execute (`gcc_cache.build_executable_cache`). `log_errors` is a
/// public `AtomicBool` on `Platform`; there is no setter.
pub fn build_executable_cache(
    c_files: &[impl AsRef<Path>],
    eci: &ExternalCompilationInfo,
    ignore_errors: bool,
) -> Result<String, CompilationError> {
    let plat = platform();
    let path = cache_file_path(c_files, eci, "build_executable_cache")?;
    if let Ok(data) = fs::read_to_string(&path) {
        return Ok(data);
    }

    let _guard = ignore_errors.then(LogErrorsRestore::suppress);
    let exe = plat.compile(c_files, eci, None, true)?;
    let result = plat
        .execute(&exe, None, None, None)
        .map_err(|err| CompilationError {
            out: String::new(),
            err: err.message,
        })?;
    drop(_guard);
    let (out, should_cache) = executable_cache_record(&result.out, &result.err);
    if !result.err.is_empty() {
        eprint!("{}", result.err);
    }
    if should_cache {
        try_atomic_write(&path, &out)?;
    }
    Ok(out)
}

/// `try_compile_cache(c_files, eci)`.
///
/// Cache key from `cache_file_path(..., "try_compile_cache")`. Cached
/// `"True"` → `Ok`; cached error text → `Err(CompilationError(text, ""))`.
/// Miss calls `Platform.compile` (compile+link). `Platform` has no
/// compile-only (`standalone=False` / `_compile_o_files`) entry, so this
/// is not compile-without-link; it does not execute. Always sets
/// `Platform.log_errors = False` for the compile.
pub fn try_compile_cache(
    c_files: &[impl AsRef<Path>],
    eci: &ExternalCompilationInfo,
) -> Result<(), CompilationError> {
    let plat = platform();
    let path = cache_file_path(c_files, eci, "try_compile_cache")?;
    match fs::read_to_string(&path) {
        Ok(data) if data == "True" => return Ok(()),
        Ok(data) => {
            return Err(CompilationError {
                out: data,
                err: String::new(),
            });
        }
        Err(_) => {}
    }

    let compile_result = {
        let _guard = LogErrorsRestore::suppress();
        plat.compile(c_files, eci, None, true)
    };
    match compile_result {
        Ok(_) => {
            try_atomic_write(&path, "True")?;
            Ok(())
        }
        Err(err) => {
            let msg = format!("{}{}", err.out, err.err);
            if msg != "True" {
                try_atomic_write(&path, &msg)?;
            }
            Err(err)
        }
    }
}

/// After `platform.execute`: return `result.out` and cache only when
/// `result.err` is empty.
fn executable_cache_record(out: &str, err: &str) -> (String, bool) {
    (out.to_string(), err.is_empty())
}

/// `try_atomic_write(path, data)`.
///
/// Temp-file create/write errors propagate. `os.rename` failure means
/// another process won and is suppressed, as is temp-file cleanup
/// (`gcc_cache.try_atomic_write`).
pub fn try_atomic_write(path: &Path, data: &str) -> Result<(), CompilationError> {
    let tmp = path.with_file_name(format!(
        "{}~{}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ));
    fs::write(&tmp, data).map_err(|err| CompilationError {
        out: String::new(),
        err: format!("gcc_cache.try_atomic_write: {}: {err}", path.display()),
    })?;
    if fs::rename(&tmp, path).is_err() {
        let _ = fs::remove_file(&tmp);
    }
    Ok(())
}

/// `cache_file_path`: `CACHE_DIR/<cachename>` (`gcc_cache.cache_file_path`).
/// `py.path.local(CACHE_DIR).ensure(dir=1)` then
/// `cache_root.join(cachename).ensure(dir=1)`.
fn cache_dir_for(cachename: &str) -> Result<PathBuf, CompilationError> {
    let dir = cache_dir().join(cachename);
    fs::create_dir_all(&dir).map_err(|err| CompilationError {
        out: String::new(),
        err: format!("gcc_cache: create cache dir {}: {err}", dir.display()),
    })?;
    Ok(dir)
}

/// `platform.log_errors = False` for a compile, then restore the previous
/// value (`build_executable_cache` / `try_compile_cache`). `Platform` has
/// no setter; the public `AtomicBool` field is the store.
struct LogErrorsRestore {
    previous: bool,
}

impl LogErrorsRestore {
    fn suppress() -> Self {
        let plat = platform();
        let previous = plat.log_errors.load(Ordering::Relaxed);
        plat.log_errors.store(false, Ordering::Relaxed);
        Self { previous }
    }
}

impl Drop for LogErrorsRestore {
    fn drop(&mut self) {
        platform()
            .log_errors
            .store(self.previous, Ordering::Relaxed);
    }
}

/// `gcc_cache.cache_file_path` hashes `repr((filecontents, eci, platform.key()))`;
/// `ExternalCompilationInfo.__repr__` / `_value` covers every field. Field-name
/// tags keep a moved value from colliding across fields.
fn hash_eci(hasher: &mut Md5, eci: &ExternalCompilationInfo) {
    let ExternalCompilationInfo {
        pre_include_bits,
        includes,
        include_dirs,
        post_include_bits,
        libraries,
        library_dirs,
        separate_module_sources,
        separate_module_files,
        compile_extra,
        link_extra,
        frameworks,
        link_files,
        testonly_libraries,
        use_cpp_linker,
        platform,
    } = eci;
    hash_named_strs(hasher, "pre_include_bits", pre_include_bits);
    hash_named_strs(hasher, "includes", includes);
    hash_named_paths(hasher, "include_dirs", include_dirs);
    hash_named_strs(hasher, "post_include_bits", post_include_bits);
    hash_named_strs(hasher, "libraries", libraries);
    hash_named_paths(hasher, "library_dirs", library_dirs);
    hash_named_strs(hasher, "separate_module_sources", separate_module_sources);
    hash_named_paths(hasher, "separate_module_files", separate_module_files);
    hash_named_strs(hasher, "compile_extra", compile_extra);
    hash_named_strs(hasher, "link_extra", link_extra);
    hash_named_strs(hasher, "frameworks", frameworks);
    hash_named_paths(hasher, "link_files", link_files);
    hash_named_strs(hasher, "testonly_libraries", testonly_libraries);
    hash_bytes(hasher, b"use_cpp_linker");
    hasher.update([u8::from(*use_cpp_linker)]);
    hash_bytes(hasher, b"platform");
    match platform {
        Some(value) => {
            hasher.update([1]);
            hash_bytes(hasher, value.as_bytes());
        }
        None => hasher.update([0]),
    }
}

fn hash_named_paths(hasher: &mut Md5, name: &str, values: &[PathBuf]) {
    hash_bytes(hasher, name.as_bytes());
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hash_os(hasher, value.as_os_str());
    }
}

fn hash_named_strs(hasher: &mut Md5, name: &str, values: &[String]) {
    hash_bytes(hasher, name.as_bytes());
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hash_bytes(hasher, value.as_bytes());
    }
}

fn hash_bytes(hasher: &mut Md5, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hash_os(hasher: &mut Md5, value: &OsStr) {
    hash_bytes(hasher, value.as_encoded_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    fn write_temp(contents: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "majit-gcc-cache-src-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, contents).unwrap();
        path
    }

    fn write_temp_c(contents: &str) -> PathBuf {
        // `Platform._make_o_file` requires sources under the repo or `udir`.
        let dir = crate::tool::udir::udir().join("gcc_cache_tests");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!(
            "src-{}-{}.c",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, contents).unwrap();
        path
    }

    fn remove_compile_artifacts(src: &Path) {
        let _ = fs::remove_file(src);
        let _ = fs::remove_file(src.with_extension("o"));
        let _ = fs::remove_file(src.with_extension("obj"));
        let _ = fs::remove_file(src.with_extension(""));
        let _ = fs::remove_dir_all(src.with_extension("dSYM"));
    }

    #[test]
    fn build_executable_cache_result_matches_upstream_policy() {
        assert_eq!(
            executable_cache_record("hello\n", ""),
            ("hello\n".to_string(), true)
        );
        assert_eq!(executable_cache_record("", ""), ("".to_string(), true));
        assert_eq!(
            executable_cache_record("hello\n", "warning: x\n"),
            ("hello\n".to_string(), false)
        );
        assert_eq!(
            executable_cache_record("", "warning: x\n"),
            ("".to_string(), false)
        );
    }

    #[test]
    fn cache_file_path_differs_when_sources_or_keys_differ() {
        let eci = ExternalCompilationInfo::default();
        let a = write_temp("int main(void) { return 0; }\n");
        let b = write_temp("int main(void) { return 1; }\n");
        let same_a = write_temp("int main(void) { return 0; }\n");
        let path_a = cache_file_path_with_key(std::slice::from_ref(&a), &eci, "test", "key-1")
            .expect("path a");
        let path_b = cache_file_path_with_key(std::slice::from_ref(&b), &eci, "test", "key-1")
            .expect("path b");
        let path_a_other_key =
            cache_file_path_with_key(std::slice::from_ref(&a), &eci, "test", "key-2")
                .expect("path a other key");
        let path_same =
            cache_file_path_with_key(std::slice::from_ref(&same_a), &eci, "test", "key-1")
                .expect("path same");
        assert_ne!(path_a, path_b);
        assert_ne!(path_a, path_a_other_key);
        assert_eq!(path_a, path_same);
        let _ = fs::remove_file(&a);
        let _ = fs::remove_file(&b);
        let _ = fs::remove_file(&same_a);
    }

    #[test]
    fn try_atomic_write_replaces_the_target() {
        let path =
            std::env::temp_dir().join(format!("majit-gcc-cache-atomic-{}", std::process::id()));
        try_atomic_write(&path, "one").unwrap();
        try_atomic_write(&path, "two").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "two");
        let _ = fs::remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn try_atomic_write_errors_on_unwritable_temp_dir() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!(
            "majit-gcc-cache-nowrite-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
        let path = dir.join("target");
        let result = try_atomic_write(&path, "data");
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
        let _ = fs::remove_dir_all(&dir);
        assert!(result.is_err());
    }

    #[test]
    fn try_atomic_write_suppresses_rename_onto_directory() {
        let dir = std::env::temp_dir().join(format!(
            "majit-gcc-cache-rename-dir-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("occupied");
        fs::create_dir(&target).unwrap();
        try_atomic_write(&target, "data").unwrap();
        assert!(target.is_dir());
        let tmp = target.with_file_name(format!("occupied~{}", std::process::id()));
        assert!(!tmp.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_file_path_changes_when_any_eci_field_changes() {
        let src = write_temp("int x;\n");
        let files = std::slice::from_ref(&src);
        let base = ExternalCompilationInfo::default();
        let base_path = cache_file_path_with_key(files, &base, "test", "key").expect("base");
        let sentinel_path = PathBuf::from("/eci-sentinel");
        let sentinel = "eci-sentinel".to_string();
        let variants = [
            ExternalCompilationInfo {
                pre_include_bits: vec![sentinel.clone()],
                ..ExternalCompilationInfo::default()
            },
            ExternalCompilationInfo {
                includes: vec![sentinel.clone()],
                ..ExternalCompilationInfo::default()
            },
            ExternalCompilationInfo {
                include_dirs: vec![sentinel_path.clone()],
                ..ExternalCompilationInfo::default()
            },
            ExternalCompilationInfo {
                post_include_bits: vec![sentinel.clone()],
                ..ExternalCompilationInfo::default()
            },
            ExternalCompilationInfo {
                libraries: vec![sentinel.clone()],
                ..ExternalCompilationInfo::default()
            },
            ExternalCompilationInfo {
                library_dirs: vec![sentinel_path.clone()],
                ..ExternalCompilationInfo::default()
            },
            ExternalCompilationInfo {
                separate_module_sources: vec![sentinel.clone()],
                ..ExternalCompilationInfo::default()
            },
            ExternalCompilationInfo {
                separate_module_files: vec![sentinel_path.clone()],
                ..ExternalCompilationInfo::default()
            },
            ExternalCompilationInfo {
                compile_extra: vec![sentinel.clone()],
                ..ExternalCompilationInfo::default()
            },
            ExternalCompilationInfo {
                link_extra: vec![sentinel.clone()],
                ..ExternalCompilationInfo::default()
            },
            ExternalCompilationInfo {
                frameworks: vec![sentinel.clone()],
                ..ExternalCompilationInfo::default()
            },
            ExternalCompilationInfo {
                link_files: vec![sentinel_path],
                ..ExternalCompilationInfo::default()
            },
            ExternalCompilationInfo {
                testonly_libraries: vec![sentinel.clone()],
                ..ExternalCompilationInfo::default()
            },
            ExternalCompilationInfo {
                use_cpp_linker: true,
                ..ExternalCompilationInfo::default()
            },
            ExternalCompilationInfo {
                platform: Some(sentinel),
                ..ExternalCompilationInfo::default()
            },
        ];
        let mut seen = vec![base_path];
        for eci in &variants {
            let path = cache_file_path_with_key(files, eci, "test", "key").expect("variant");
            assert!(
                !seen.contains(&path),
                "changing any ExternalCompilationInfo field must change cache_file_path: {path:?}"
            );
            seen.push(path);
        }
        let _ = fs::remove_file(&src);
    }

    #[test]
    fn cache_file_path_nests_under_cache_dir_cachename() {
        let eci = ExternalCompilationInfo::default();
        let src = write_temp("int x;\n");
        let path = cache_file_path(std::slice::from_ref(&src), &eci, "try_compile_cache")
            .expect("cache path");
        let cache_name = path.parent().expect("cachename dir");
        assert_eq!(cache_name.file_name().unwrap(), "try_compile_cache");
        assert_eq!(cache_name.parent().unwrap(), cache_dir().as_path());
        let _ = fs::remove_file(&src);
    }

    #[test]
    fn try_compile_cache_stores_true_or_error_text() {
        let eci = ExternalCompilationInfo::default();
        let marker = format!(
            "/* {}-{} */",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        );
        let ok_src = write_temp_c(&format!(
            "{marker}\n#include <stdio.h>\nint main(void) {{ printf(\"hello\\n\"); return 0; }}\n"
        ));
        try_compile_cache(std::slice::from_ref(&ok_src), &eci).expect("valid C compiles");
        let ok_cache =
            cache_file_path(std::slice::from_ref(&ok_src), &eci, "try_compile_cache").unwrap();
        assert_eq!(fs::read_to_string(&ok_cache).unwrap(), "True");
        try_atomic_write(&ok_cache, "cached-failure").unwrap();
        let cached_err = try_compile_cache(std::slice::from_ref(&ok_src), &eci).unwrap_err();
        assert_eq!(cached_err.out, "cached-failure");
        assert_eq!(cached_err.err, "");

        let bad_src = write_temp_c(&format!("{marker}\nthis is not c\n"));
        let compile_err =
            try_compile_cache(std::slice::from_ref(&bad_src), &eci).expect_err("invalid C");
        let bad_cache =
            cache_file_path(std::slice::from_ref(&bad_src), &eci, "try_compile_cache").unwrap();
        let stored = fs::read_to_string(&bad_cache).unwrap();
        assert_ne!(stored, "True");
        assert_eq!(stored, format!("{}{}", compile_err.out, compile_err.err));
        let cached_again = try_compile_cache(std::slice::from_ref(&bad_src), &eci).unwrap_err();
        assert_eq!(cached_again.out, stored);
        assert_eq!(cached_again.err, "");

        remove_compile_artifacts(&ok_src);
        remove_compile_artifacts(&bad_src);
        let _ = fs::remove_file(&ok_cache);
        let _ = fs::remove_file(&bad_cache);
    }
}
