//! Port of `rpython/translator/platform/linux.py` (`Linux`) and
//! `rpython/translator/platform/arch/s390x.py` (`update_cflags`).

use std::ffi::OsString;
use std::path::PathBuf;

use super::{FamilySpec, HostFamily, Platform, os_flags, posix, split_env_flags};

/// `Linux.__init__` flags (`cflags` / `link_flags` / `extra_libs` / `shared_only`).
pub(super) fn linux_platform(host_triple: String, s390x: bool) -> Platform {
    let mut cflags = os_flags([
        "-O3",
        "-pthread",
        "-fomit-frame-pointer",
        "-Wall",
        "-Wno-unused",
        "-Wno-address",
        "-Wno-discarded-qualifiers",
        "-Wno-duplicate-decl-specifier",
    ]);
    cflags.extend(split_env_flags("CFLAGS"));
    if s390x {
        cflags = s390x_update_cflags(cflags);
    }
    let mut link_flags = os_flags(["-pthread"]);
    link_flags.extend(split_env_flags("LDFLAGS"));
    Platform::from_spec(
        host_triple,
        FamilySpec {
            name: "linux".to_string(),
            class_name: "Linux",
            family: HostFamily::Linux,
            relevant_environ: posix::RELEVANT_ENVIRON,
            cflags,
            link_flags,
            standalone_only: Vec::new(),
            shared_only: os_flags(["-fPIC"]),
            extra_libs: os_flags(["-lrt"]),
            exe_ext: posix::EXE_EXT,
            so_ext: "so",
            link_path: PathBuf::new(),
            masm_path: PathBuf::new(),
        },
    )
}

/// `s390x.update_cflags`.
pub(crate) fn s390x_update_cflags(mut cflags: Vec<OsString>) -> Vec<OsString> {
    let has_march = cflags
        .iter()
        .any(|flag| flag.as_encoded_bytes().starts_with(b"-march="));
    if !has_march {
        cflags.push(OsString::from("-march=zEC12"));
    }
    cflags.push(OsString::from("-m64"));
    cflags.push(OsString::from("-mzarch"));
    cflags
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::tool::cbuild::ExternalCompilationInfo;

    #[test]
    fn s390x_update_cflags_defaults_to_zec12_and_appends_m64() {
        let flags = s390x_update_cflags(os_flags(["-O3"]));
        assert_eq!(flags, os_flags(["-O3", "-march=zEC12", "-m64", "-mzarch"]));
    }

    #[test]
    fn s390x_update_cflags_keeps_existing_march() {
        let flags = s390x_update_cflags(os_flags(["-O3", "-march=z13"]));
        assert_eq!(flags, os_flags(["-O3", "-march=z13", "-m64", "-mzarch"]));
    }

    #[test]
    fn linux_link_args_include_lrt_and_pthread() {
        let plat = linux_platform("x86_64-unknown-linux-gnu".to_string(), false);
        let compile = plat._compile_args_from_eci(&ExternalCompilationInfo::default(), true);
        assert!(compile.iter().any(|flag| flag == "-pthread"));
        assert!(compile.iter().any(|flag| flag == "-Wall"));
        assert!(compile.iter().any(|flag| flag == "-fomit-frame-pointer"));
        let link = plat._link_args_from_eci(&ExternalCompilationInfo::default(), true);
        assert!(link.iter().any(|flag| flag == "-lrt"));
        assert!(link.iter().any(|flag| flag == "-pthread"));
        let shared = plat._compile_args_from_eci(&ExternalCompilationInfo::default(), false);
        assert!(shared.iter().any(|flag| flag == "-fPIC"));
        assert!(shared.iter().any(|flag| flag == "-fvisibility=hidden"));
    }
}
