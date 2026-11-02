//! Port of `rpython/translator/platform/windows.py` (`MsvcPlatform`,
//! `MingwPlatform`).

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use super::{CompilerInvocation, FamilySpec, HostFamily, Platform, os_flags, prefixed_path};

pub(super) const RELEVANT_ENVIRON: &[&str] = &["PATH", "INCLUDE", "LIB"];

/// `MsvcPlatform` (`Windows` / `Windows_x64`).
pub(super) fn msvc_platform(host_triple: String, x64: bool) -> Platform {
    let masm_name = if x64 { "ml64.exe" } else { "ml.exe" };
    Platform::from_spec(
        host_triple,
        FamilySpec {
            name: "msvc".to_string(),
            class_name: if x64 { "Windows_x64" } else { "Windows" },
            family: HostFamily::Msvc,
            relevant_environ: RELEVANT_ENVIRON,
            cflags: os_flags(["/MD", "/O2"]),
            link_flags: os_flags([
                "/nologo",
                "/LARGEADDRESSAWARE",
                "/STACK:3145728",
                "/MANIFEST:EMBED",
            ]),
            standalone_only: Vec::new(),
            shared_only: Vec::new(),
            extra_libs: Vec::new(),
            exe_ext: "exe",
            so_ext: "dll",
            link_path: PathBuf::from("link.exe"),
            masm_path: PathBuf::from(masm_name),
        },
    )
}

/// `MingwPlatform`.
pub(super) fn mingw_platform(host_triple: String) -> Platform {
    Platform::from_spec(
        host_triple,
        FamilySpec {
            name: "mingw32".to_string(),
            class_name: "MingwPlatform",
            family: HostFamily::Mingw,
            relevant_environ: RELEVANT_ENVIRON,
            cflags: os_flags(["-O3"]),
            link_flags: Vec::new(),
            standalone_only: Vec::new(),
            shared_only: Vec::new(),
            extra_libs: Vec::new(),
            exe_ext: "exe",
            so_ext: "dll",
            link_path: PathBuf::new(),
            masm_path: PathBuf::new(),
        },
    )
}

/// `MsvcPlatform.link` / `MsvcPlatform.masm`: `cc::windows_registry::find_tool`
/// then the directory of `cl.exe`.
pub(super) fn discover_msvc_sidecar(
    cc: &cc::Tool,
    target: &str,
    name: &str,
) -> (Option<cc::Tool>, PathBuf) {
    if let Some(tool) = cc::windows_registry::find_tool(target, name) {
        let path = tool.path().to_path_buf();
        return (Some(tool), path);
    }
    if let Some(parent) = cc.path().parent() {
        let candidate = parent.join(name);
        if candidate.is_file() {
            return (None, candidate);
        }
    }
    (None, PathBuf::from(name))
}

/// `MsvcPlatform._includedirs`.
pub(super) fn msvc_includedirs(include_dirs: &[PathBuf]) -> Vec<OsString> {
    debug_assert!(
        include_dirs.iter().all(|dir| !dir.as_os_str().is_empty()),
        "empty include dir"
    );
    include_dirs
        .iter()
        .map(|dir| prefixed_path("/I", dir))
        .collect()
}

/// `MsvcPlatform._libdirs`.
pub(super) fn msvc_libdirs(library_dirs: &[PathBuf]) -> Vec<OsString> {
    debug_assert!(
        library_dirs.iter().all(|dir| !dir.as_os_str().is_empty()),
        "empty library dir"
    );
    library_dirs
        .iter()
        .map(|dir| prefixed_path("/LIBPATH:", dir))
        .collect()
}

/// `MsvcPlatform._libs`.
pub(super) fn msvc_libs(libraries: &[String]) -> Vec<OsString> {
    let mut libs: Vec<OsString> = libraries
        .iter()
        .map(|lib| {
            let name = lib.strip_suffix(".dll").unwrap_or(lib);
            OsString::from(format!("{name}.lib"))
        })
        .collect();
    libs.push(OsString::from("kernel32.lib"));
    libs
}

/// `MsvcPlatform._compile_c_file`.
pub(super) fn msvc_compile_c_file_invocation(
    cc: &Path,
    cfile: &Path,
    compile_args: &[OsString],
    oname: &Path,
) -> CompilerInvocation {
    let mut args = vec![OsString::from("/nologo"), OsString::from("/c")];
    args.extend(compile_args.iter().cloned());
    if let Some(pdb_dir) = oname.parent().filter(|path| !path.as_os_str().is_empty()) {
        let mut fd = OsString::from("/Fd");
        fd.push(pdb_dir);
        fd.push("\\");
        args.push(fd);
    }
    args.push(prefixed_path("/Fo", oname));
    args.push(cfile.as_os_str().to_os_string());
    CompilerInvocation {
        program: cc.to_path_buf(),
        args,
        cwd: None,
    }
}

/// `MsvcPlatform._args_for_shared`.
pub(super) fn msvc_args_for_shared(args: Vec<OsString>) -> Vec<OsString> {
    let mut out = vec![OsString::from("/dll")];
    out.extend(args);
    out
}

/// `MsvcPlatform._link` (`self.link` / link.exe, `/out:`, `/incremental:no`,
/// `/MANIFEST:EMBED`).
pub(super) fn msvc_link_invocation(
    link: &Path,
    ofiles: &[PathBuf],
    link_args: &[OsString],
    standalone: bool,
    exe_name: &Path,
) -> CompilerInvocation {
    let mut args = vec![OsString::from("/nologo")];
    args.extend(ofiles.iter().map(|ofile| ofile.as_os_str().to_os_string()));
    args.extend(link_args.iter().cloned());
    args.push(prefixed_path("/out:", exe_name));
    args.push(OsString::from("/incremental:no"));
    if !standalone {
        args = msvc_args_for_shared(args);
    }
    args.push(OsString::from("/MANIFEST:EMBED"));
    CompilerInvocation {
        program: link.to_path_buf(),
        args,
        cwd: None,
    }
}

/// `MsvcPlatform._exportsymbols_link_flags` (inherits `Platform` → empty).
pub(super) fn msvc_exportsymbols_link_flags() -> Vec<OsString> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::platform::is_asm_source;
    use crate::translator::tool::cbuild::ExternalCompilationInfo;
    use std::path::Path;

    fn msvc_x64() -> Platform {
        msvc_platform("x86_64-pc-windows-msvc".to_string(), true)
    }

    fn render(args: &[OsString]) -> Vec<String> {
        args.iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn msvc_link_runs_link_exe_with_out_stack_and_manifest() {
        let plat = msvc_x64();
        assert_eq!(
            plat.link_path().file_name().unwrap(),
            "link.exe",
            "MsvcPlatform.link is link.exe"
        );
        let libdir = PathBuf::from(r"C:\libs");
        let eci = ExternalCompilationInfo {
            library_dirs: vec![libdir.clone()],
            libraries: vec!["foo".to_string()],
            link_files: vec![PathBuf::from(r"C:\prebuilt.obj")],
            ..ExternalCompilationInfo::default()
        };
        let link_args = plat._link_args_from_eci(&eci, true);
        let rendered_largs = render(&link_args);
        assert!(
            rendered_largs
                .iter()
                .any(|arg| arg == r"/LIBPATH:C:\libs" || arg.starts_with("/LIBPATH:")),
            "expected /LIBPATH: in {rendered_largs:?}"
        );
        assert!(rendered_largs.iter().any(|arg| arg == "/STACK:3145728"));
        assert!(rendered_largs.iter().any(|arg| arg == "/MANIFEST:EMBED"));
        assert!(rendered_largs.iter().any(|arg| arg == r"C:\prebuilt.obj"));
        assert!(rendered_largs.iter().any(|arg| arg == "foo.lib"));
        assert!(rendered_largs.iter().any(|arg| arg == "kernel32.lib"));

        let ofiles = [PathBuf::from(r"C:\tmp\main.obj")];
        let exe = PathBuf::from(r"C:\tmp\main.exe");
        let inv = plat.link_invocation(Path::new("cl.exe"), &ofiles, &link_args, true, &exe);
        assert_eq!(inv.program.file_name().unwrap(), "link.exe");
        let rendered = render(&inv.args);
        assert!(rendered.iter().any(|arg| arg == "/nologo"));
        assert!(rendered.iter().any(|arg| arg == r"/out:C:\tmp\main.exe"));
        assert!(rendered.iter().any(|arg| arg == "/incremental:no"));
        assert!(
            rendered
                .iter()
                .filter(|arg| *arg == "/MANIFEST:EMBED")
                .count()
                >= 2,
            "link_flags and _link both pass /MANIFEST:EMBED: {rendered:?}"
        );
        assert!(
            rendered.iter().all(|arg| !arg.starts_with("/Fe")),
            "MSVC _link uses link.exe /out:, not cl /Fe: {rendered:?}"
        );
        assert!(inv.cwd.is_none());
    }

    #[test]
    fn msvc_link_ignores_cpp_linker_and_uses_link_exe() {
        let plat = msvc_x64();
        let eci = ExternalCompilationInfo {
            use_cpp_linker: true,
            ..ExternalCompilationInfo::default()
        };
        let ofiles = [PathBuf::from(r"C:\tmp\a.obj")];
        let exe = PathBuf::from(r"C:\tmp\a.exe");
        let inv = plat.link_invocation(&plat.cc_link_program(&eci), &ofiles, &[], true, &exe);
        assert_eq!(inv.program.file_name().unwrap(), "link.exe");
    }

    #[test]
    fn msvc_shared_link_prepends_dll() {
        let plat = msvc_x64();
        let ofiles = [PathBuf::from(r"C:\tmp\a.obj")];
        let exe = PathBuf::from(r"C:\tmp\a.dll");
        let inv = plat.link_invocation(Path::new("cl.exe"), &ofiles, &[], false, &exe);
        assert_eq!(inv.args[0], OsString::from("/dll"));
        assert_eq!(inv.program.file_name().unwrap(), "link.exe");
    }

    #[test]
    fn msvc_asm_goes_to_masm_with_empty_extra_args() {
        let plat = msvc_x64();
        assert_eq!(plat.masm_path().file_name().unwrap(), "ml64.exe");
        let asm = PathBuf::from(r"C:\src\foo.ASM");
        assert!(is_asm_source(&asm));
        assert_eq!(
            plat.compiler_for_source(&asm).file_name().unwrap(),
            "ml64.exe"
        );
        assert_eq!(
            plat.compiler_for_source(Path::new(r"C:\src\foo.c"))
                .file_name()
                .map(|name| name.to_os_string()),
            Some(OsString::from("cl.exe"))
        );
        let oname = PathBuf::from(r"C:\src\foo.obj");
        let inv = msvc_compile_c_file_invocation(plat.masm_path(), &asm, &[], &oname);
        assert_eq!(inv.program.file_name().unwrap(), "ml64.exe");
        let rendered = render(&inv.args);
        assert_eq!(rendered[0], "/nologo");
        assert_eq!(rendered[1], "/c");
        assert!(
            rendered.iter().any(|arg| arg == r"/FoC:\src\foo.obj"),
            "expected /Fo in {rendered:?}"
        );
        assert_eq!(rendered.last().unwrap(), r"C:\src\foo.ASM");
        assert!(
            rendered.iter().all(|arg| arg != "/MD" && arg != "/O2"),
            "asm compile_args are empty: {rendered:?}"
        );
    }

    #[test]
    fn mingw_uses_posix_export_all_symbols() {
        let plat = mingw_platform("x86_64-pc-windows-gnu".to_string());
        let link = plat._link_args_from_eci(&ExternalCompilationInfo::default(), true);
        assert!(link.iter().any(|flag| flag == "-Wl,--export-all-symbols"));
        let compile = plat._compile_args_from_eci(&ExternalCompilationInfo::default(), true);
        assert!(compile.iter().any(|flag| flag == "-O3"));
        assert!(compile.iter().all(|flag| flag != "-pthread"));
    }
}
