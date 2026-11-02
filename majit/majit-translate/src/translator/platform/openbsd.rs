//! Port of `rpython/translator/platform/openbsd.py` (`OpenBSD`, `OpenBSD_64`).

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use super::{FamilySpec, HostFamily, Platform, os_flags, posix, prefixed_path, split_env_flags};

/// `OpenBSD` / `OpenBSD_64`.
pub(super) fn openbsd_platform(host_triple: String, is_64: bool) -> Platform {
    let mut cflags = os_flags(["-O3", "-pthread", "-fomit-frame-pointer", "-D_BSD_SOURCE"]);
    cflags.extend(split_env_flags("CFLAGS"));
    let mut link_flags = split_env_flags("LDFLAGS");
    link_flags.extend(os_flags(["-pthread"]));
    Platform::from_spec(
        host_triple,
        FamilySpec {
            name: "openbsd".to_string(),
            class_name: if is_64 { "OpenBSD_64" } else { "OpenBSD" },
            family: HostFamily::Openbsd,
            relevant_environ: posix::RELEVANT_ENVIRON,
            cflags,
            link_flags,
            standalone_only: Vec::new(),
            shared_only: if is_64 {
                os_flags(["-fPIC"])
            } else {
                Vec::new()
            },
            extra_libs: Vec::new(),
            exe_ext: posix::EXE_EXT,
            so_ext: "so",
            link_path: PathBuf::new(),
            masm_path: PathBuf::new(),
        },
    )
}

/// `OpenBSD._libs`. Upstream `set(libraries + ("intl", "iconv"))` has
/// unspecified order; first-seen order is the deterministic choice.
pub(super) fn openbsd_libs(libraries: &[String]) -> Vec<OsString> {
    let mut seen = Vec::new();
    for lib in libraries
        .iter()
        .map(String::as_str)
        .chain(["intl", "iconv"])
    {
        if !seen.iter().any(|existing| existing == lib) {
            seen.push(lib.to_string());
        }
    }
    seen.into_iter()
        .filter(|lib| lib != "crypt" && lib != "dl" && lib != "rt")
        .map(|lib| prefixed_path("-l", Path::new(&lib)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::tool::cbuild::ExternalCompilationInfo;

    #[test]
    fn openbsd_cflags_include_bsd_source() {
        let plat = openbsd_platform("x86_64-unknown-openbsd".to_string(), true);
        let compile = plat._compile_args_from_eci(&ExternalCompilationInfo::default(), true);
        assert!(compile.iter().any(|flag| flag == "-D_BSD_SOURCE"));
        assert!(compile.iter().any(|flag| flag == "-pthread"));
        assert!(compile.iter().all(|flag| flag != "-Wall"));
    }

    #[test]
    fn openbsd_libs_drop_crypt_dl_rt_and_add_intl_iconv() {
        let libs = openbsd_libs(&["m".to_string(), "crypt".to_string(), "rt".to_string()]);
        let rendered: Vec<String> = libs
            .iter()
            .map(|lib| lib.to_string_lossy().into_owned())
            .collect();
        assert!(rendered.contains(&"-lm".to_string()));
        assert!(rendered.contains(&"-lintl".to_string()));
        assert!(rendered.contains(&"-liconv".to_string()));
        assert!(
            !rendered
                .iter()
                .any(|lib| lib == "-lcrypt" || lib == "-lrt" || lib == "-ldl")
        );
    }

    #[test]
    fn openbsd_libs_first_seen_set_then_filter() {
        let libs = openbsd_libs(&[
            "m".to_string(),
            "m".to_string(),
            "intl".to_string(),
            "crypt".to_string(),
        ]);
        let rendered: Vec<String> = libs
            .iter()
            .map(|lib| lib.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            rendered,
            vec![
                "-lm".to_string(),
                "-lintl".to_string(),
                "-liconv".to_string()
            ]
        );
    }
}
