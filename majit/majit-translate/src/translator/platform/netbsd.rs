//! Port of `rpython/translator/platform/netbsd.py` (`Netbsd`, `Netbsd_64`).

use std::path::PathBuf;

use super::bsd::localbase_or;
use super::{FamilySpec, HostFamily, Platform, os_flags, posix, split_env_flags};

/// `Netbsd` / `Netbsd_64`.
pub(super) fn netbsd_platform(host_triple: String, is_64: bool) -> Platform {
    let mut cflags = os_flags(["-O3", "-pthread", "-fomit-frame-pointer"]);
    cflags.extend(split_env_flags("CFLAGS"));
    let localbase = localbase_or("/usr/pkg");
    let mut link_flags = os_flags(["-pthread"]);
    let mut rpath = std::ffi::OsString::from("-Wl,-R");
    rpath.push(localbase.join("lib"));
    link_flags.push(rpath);
    link_flags.extend(split_env_flags("LDFLAGS"));
    Platform::from_spec(
        host_triple,
        FamilySpec {
            name: "netbsd".to_string(),
            class_name: if is_64 { "Netbsd_64" } else { "Netbsd" },
            family: HostFamily::Netbsd,
            relevant_environ: posix::RELEVANT_ENVIRON,
            cflags,
            link_flags,
            standalone_only: Vec::new(),
            shared_only: if is_64 {
                os_flags(["-fPIC"])
            } else {
                Vec::new()
            },
            extra_libs: os_flags(["-lrt"]),
            exe_ext: posix::EXE_EXT,
            so_ext: "so",
            link_path: PathBuf::new(),
            masm_path: PathBuf::new(),
        },
    )
}

/// `Netbsd._preprocess_include_dirs`.
pub(super) fn preprocess_include_dirs(include_dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs = include_dirs.to_vec();
    dirs.push(localbase_or("/usr/pkg").join("include"));
    dirs
}

/// `Netbsd._preprocess_library_dirs`.
pub(super) fn preprocess_library_dirs(library_dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs = library_dirs.to_vec();
    dirs.push(localbase_or("/usr/pkg").join("lib"));
    dirs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::tool::cbuild::ExternalCompilationInfo;
    use std::path::PathBuf;

    #[test]
    fn netbsd_link_flags_include_rpath_and_lrt() {
        let plat = netbsd_platform("x86_64-unknown-netbsd".to_string(), true);
        let link = plat._link_args_from_eci(&ExternalCompilationInfo::default(), true);
        let rendered: Vec<String> = link
            .iter()
            .map(|flag| flag.to_string_lossy().into_owned())
            .collect();
        assert!(rendered.iter().any(|flag| flag.starts_with("-Wl,-R")));
        assert!(rendered.iter().any(|flag| flag == "-lrt"));
        let compile = plat._compile_args_from_eci(&ExternalCompilationInfo::default(), true);
        assert!(compile.iter().any(|flag| flag == "-pthread"));
        assert!(compile.iter().all(|flag| flag != "-Wall"));
    }

    #[test]
    fn netbsd_preprocess_appends_pkg_include() {
        let dirs = preprocess_include_dirs(&[PathBuf::from("/app/include")]);
        assert_eq!(dirs[0], PathBuf::from("/app/include"));
        assert!(
            dirs.iter()
                .any(|dir| dir == &PathBuf::from("/usr/pkg/include") || dir.ends_with("include"))
        );
    }
}
