//! Port of `rpython/translator/platform/freebsd.py` (`Freebsd`, `Freebsd_64`).

use std::path::PathBuf;

use super::{FamilySpec, HostFamily, Platform, os_flags, posix, split_env_flags};

/// `Freebsd` / `Freebsd_64`.
pub(super) fn freebsd_platform(host_triple: String, is_64: bool) -> Platform {
    let mut cflags = os_flags(["-O3", "-pthread", "-fomit-frame-pointer"]);
    cflags.extend(split_env_flags("CFLAGS"));
    let mut link_flags = os_flags(["-pthread"]);
    link_flags.extend(split_env_flags("LDFLAGS"));
    Platform::from_spec(
        host_triple,
        FamilySpec {
            name: "freebsd".to_string(),
            class_name: if is_64 { "Freebsd_64" } else { "Freebsd" },
            family: HostFamily::Freebsd,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::tool::cbuild::ExternalCompilationInfo;
    use std::path::PathBuf;

    #[test]
    fn freebsd_flags_are_not_linux_cflags() {
        let plat = freebsd_platform("x86_64-unknown-freebsd".to_string(), true);
        let compile = plat._compile_args_from_eci(&ExternalCompilationInfo::default(), true);
        assert!(compile.iter().any(|flag| flag == "-pthread"));
        assert!(compile.iter().any(|flag| flag == "-O3"));
        assert!(compile.iter().all(|flag| flag != "-Wall"));
        let link = plat._link_args_from_eci(&ExternalCompilationInfo::default(), true);
        assert!(link.iter().all(|flag| flag != "-lrt"));
        let shared = plat._compile_args_from_eci(&ExternalCompilationInfo::default(), false);
        assert!(shared.iter().any(|flag| flag == "-fPIC"));
    }

    #[test]
    fn freebsd_preprocess_appends_localbase() {
        let plat = freebsd_platform("x86_64-unknown-freebsd".to_string(), true);
        let dirs = plat.preprocess_include_dirs(&[PathBuf::from("/app/include")]);
        assert!(dirs.iter().any(|dir| dir.ends_with("include")));
        assert!(dirs.contains(&PathBuf::from("/app/include")));
    }
}
