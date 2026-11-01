//! Port of `rpython/translator/platform/darwin.py` (`Darwin`, `Darwin_arm64`,
//! `Darwin_x86_64`).

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use super::{FamilySpec, HostFamily, Platform, os_flags, posix};

/// `DARWIN_VERSION_MIN`.
const DARWIN_VERSION_MIN: &str = "-mmacosx-version-min=10.13";

/// `Darwin` / `Darwin_arm64` / `Darwin_x86_64` constructor.
pub(super) fn darwin_platform(host_triple: String) -> Platform {
    let arch = darwin_arch_flags(&host_triple);
    let mut cflags = os_flags([
        "-O3",
        "-fomit-frame-pointer",
        "-Wno-duplicate-decl-specifier",
        "-Wno-unguarded-availability-new",
        DARWIN_VERSION_MIN,
    ]);
    cflags.extend(arch.iter().cloned());
    let mut link_flags = os_flags([DARWIN_VERSION_MIN]);
    link_flags.extend(arch);
    Platform::from_spec(
        host_triple.clone(),
        FamilySpec {
            name: "darwin".to_string(),
            class_name: darwin_class_name(&host_triple),
            family: HostFamily::Darwin,
            relevant_environ: posix::RELEVANT_ENVIRON,
            cflags,
            link_flags,
            standalone_only: os_flags(["-mdynamic-no-pic"]),
            shared_only: Vec::new(),
            extra_libs: Vec::new(),
            exe_ext: posix::EXE_EXT,
            so_ext: "dylib",
            link_path: PathBuf::new(),
            masm_path: PathBuf::new(),
        },
    )
}

fn darwin_class_name(triple: &str) -> &'static str {
    if triple.contains("aarch64") || triple.contains("arm64") {
        "Darwin_arm64"
    } else if triple.contains("x86_64") {
        "Darwin_x86_64"
    } else if triple.contains("i386") || triple.contains("i686") {
        "Darwin_i386"
    } else if triple.contains("powerpc") || triple.contains("ppc") {
        "Darwin_PowerPC"
    } else {
        "Darwin"
    }
}

fn darwin_arch_flags(triple: &str) -> Vec<OsString> {
    if triple.contains("aarch64") || triple.contains("arm64") {
        os_flags(["-arch", "arm64"])
    } else if triple.contains("x86_64") {
        os_flags(["-arch", "x86_64"])
    } else if triple.contains("i386") || triple.contains("i686") {
        os_flags(["-arch", "i386"])
    } else if triple.contains("powerpc") || triple.contains("ppc") {
        os_flags(["-arch", "ppc"])
    } else {
        Vec::new()
    }
}

/// `Darwin._frameworks`.
pub(super) fn frameworks(frameworks: &[String]) -> Vec<OsString> {
    let mut args = Vec::with_capacity(frameworks.len() * 2);
    for name in frameworks {
        args.push(OsString::from("-framework"));
        args.push(OsString::from(name));
    }
    args
}

/// `Darwin._args_for_shared`.
pub(super) fn darwin_args_for_shared(
    shared_only: &[OsString],
    args: Vec<OsString>,
    exe_name: &Path,
) -> Vec<OsString> {
    let target_basename = exe_name
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_else(|| OsString::from("$(TARGET)"));
    let mut out = shared_only.to_vec();
    out.push(OsString::from("-dynamiclib"));
    out.push(OsString::from("-install_name"));
    let mut install_name = OsString::from("@rpath/");
    install_name.push(&target_basename);
    out.push(install_name);
    out.push(OsString::from("-undefined"));
    out.push(OsString::from("dynamic_lookup"));
    out.push(OsString::from("-flat_namespace"));
    out.push(OsString::from("-headerpad_max_install_names"));
    out.extend(args);
    out
}

/// `Darwin._exportsymbols_link_flags`.
pub(super) fn darwin_exportsymbols_link_flags() -> Vec<OsString> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::tool::cbuild::ExternalCompilationInfo;
    use std::path::PathBuf;

    #[test]
    fn darwin_link_args_include_frameworks_and_include_dirs() {
        let plat = darwin_platform("aarch64-apple-darwin".to_string());
        let inc = PathBuf::from("/tmp/darwin-inc");
        let eci = ExternalCompilationInfo {
            include_dirs: vec![inc.clone()],
            frameworks: vec!["Cocoa".to_string()],
            link_files: vec![PathBuf::from("/foo/bar.a")],
            ..ExternalCompilationInfo::default()
        };
        let args = plat._link_args_from_eci(&eci, true);
        let rendered: Vec<String> = args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert!(
            rendered
                .windows(2)
                .any(|pair| pair[0] == "-framework" && pair[1] == "Cocoa"),
            "expected -framework Cocoa in {rendered:?}"
        );
        assert!(rendered.iter().any(|arg| arg == "-I/tmp/darwin-inc"));
        assert!(rendered.iter().any(|arg| arg == "/foo/bar.a"));
        assert!(rendered.iter().any(|arg| arg == DARWIN_VERSION_MIN));
    }

    #[test]
    fn darwin_cpp_linker_is_clangxx() {
        let plat = darwin_platform("x86_64-apple-darwin".to_string());
        assert_eq!(plat.cpp_linker_name(), "clang++");
        let eci = ExternalCompilationInfo {
            use_cpp_linker: true,
            ..ExternalCompilationInfo::default()
        };
        assert_eq!(plat.cc_link_program(&eci), PathBuf::from("clang++"));
    }

    #[test]
    fn darwin_shared_args_use_dynamiclib_install_name() {
        let shared = darwin_args_for_shared(&[], Vec::new(), Path::new("/tmp/libfoo.dylib"));
        let rendered: Vec<String> = shared
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert!(rendered.contains(&"-dynamiclib".to_string()));
        assert!(rendered.contains(&"-install_name".to_string()));
        assert!(rendered.contains(&"@rpath/libfoo.dylib".to_string()));
    }
}
