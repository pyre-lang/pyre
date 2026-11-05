//! Port of `rpython/translator/platform/windows.py` (`MsvcPlatform`,
//! `MingwPlatform`).

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use super::{
    CompilerInvocation, FamilySpec, HostFamily, Platform, env_name_eq, host_c_compiler, os_flags,
    posix, prefixed_path, resolve_compiler_on_path, rpythonroot,
};

pub(super) const RELEVANT_ENVIRON: &[&str] = &["PATH", "INCLUDE", "LIB"];

/// `MsvcPlatform` returned by the `Windows` / `Windows_x64` factory functions.
/// Those names are factories, not classes; `class_name` is `MsvcPlatform`.
/// `x64` still selects `ml64.exe` vs `ml.exe`.
pub(super) fn msvc_platform(host_triple: String, x64: bool, cc: Option<String>) -> Platform {
    let masm_name = if x64 { "ml64.exe" } else { "ml.exe" };
    let mut plat = Platform::from_spec(
        host_triple,
        FamilySpec {
            name: "msvc".to_string(),
            class_name: "MsvcPlatform",
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
    );
    // `MsvcPlatform.__init__`: `patch_os_env(self.externals)` then
    // `self.c_environ = os.environ.copy()` always run. Process-wide
    // `set_var` is not used (unsafe in this edition and races other threads);
    // the captured mapping is the snapshot. `if cc is None:` runs
    // `find_msvc_env` (`host_c_compiler`) and
    // `self.c_environ.update(msvc_compiler_environ)` (`merged_compiler_environ`);
    // an empty/failed `msvc_compiler_environ` leaves the snapshot unchanged.
    // Then `Platform.__init__(self, 'cl.exe')` whether or not `find_msvc_env`
    // found an environment. `else: self.cc = cc`. After construction
    // `cc_command` is always `Some`.
    let patch = patch_os_env(&rpythonroot().join("externals"), |name| {
        std::env::var(name).ok()
    });
    let mut c_environ: Vec<(OsString, OsString)> = std::env::vars_os().collect();
    for (name, value) in patch {
        let name_os = OsString::from(&name);
        let value_os = OsString::from(value);
        if let Some((_, existing)) = c_environ
            .iter_mut()
            .find(|(key, _)| env_name_eq(key, name.as_str()))
        {
            *existing = value_os;
        } else {
            c_environ.push((name_os, value_os));
        }
    }
    if cc.is_none() {
        if let Ok(tool) = host_c_compiler(&plat.host_triple, None, Some(&c_environ)) {
            c_environ = merged_compiler_environ(Some(&c_environ), tool.env());
        }
        plat.cc_command = Some("cl.exe".to_string());
    } else {
        plat.cc_command = cc;
    }
    plat.c_environ = Some(c_environ);
    plat
}

/// `MingwPlatform`.
pub(super) fn mingw_platform(host_triple: String) -> Platform {
    Platform::from_spec(
        host_triple,
        FamilySpec {
            name: "mingw32".to_string(),
            class_name: "MingwPlatform",
            family: HostFamily::Mingw,
            relevant_environ: posix::RELEVANT_ENVIRON,
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

/// `MingwPlatform._include_dirs_for_libffi`.
pub(super) fn _include_dirs_for_libffi(
    _plat: &Platform,
) -> Result<Vec<OsString>, posix::ValueError> {
    Ok(Vec::new())
}

/// `MingwPlatform._library_dirs_for_libffi`.
pub(super) fn _library_dirs_for_libffi(
    _plat: &Platform,
) -> Result<Vec<OsString>, posix::ValueError> {
    Ok(Vec::new())
}

/// `_get_msvc_env`'s final loop. Keep only PATH/INCLUDE/LIB (`ukey` in
/// that set); every other key is dropped. The stored name is the
/// upper-cased `ukey`. PATH/INCLUDE/LIB append `existing` when non-empty
/// (`os.environ.get` after `patch_os_env`, or the process environment
/// when `existing` is `None`).
pub(super) fn msvc_compiler_environ(
    tool_env: &[(OsString, OsString)],
    existing: Option<&[(OsString, OsString)]>,
) -> Vec<(OsString, OsString)> {
    let mut env = Vec::new();
    for (key, value) in tool_env {
        // `_get_msvc_env`: `ukey = key.upper()` then
        // `if ukey in ('PATH', 'INCLUDE', 'LIB')`. `env_name_eq` covers the
        // Windows-insensitive map; `eq_ignore_ascii_case` is `str.upper`.
        let Some(name) = ["PATH", "INCLUDE", "LIB"].into_iter().find(|name| {
            env_name_eq(key, name) || key.to_str().is_some_and(|k| k.eq_ignore_ascii_case(name))
        }) else {
            continue;
        };
        let mut value = value.clone();
        let existing_value = match existing {
            Some(environ) => environ
                .iter()
                .find(|(k, _)| env_name_eq(k, name))
                .map(|(_, v)| v.clone()),
            None => std::env::var_os(name),
        };
        if let Some(existing_value) = existing_value.filter(|v| !v.is_empty()) {
            value = match value.to_str() {
                Some(s) => OsString::from(s.trim_end_matches(';')),
                None => value,
            };
            value.push(";");
            value.push(&existing_value);
        }
        env.push((OsString::from(name), value));
    }
    env
}

/// `MsvcPlatform.__init__`: `self.c_environ.update(msvc_compiler_environ)`.
/// `c_environ` is the snapshot after `patch_os_env` (or the process
/// environment when absent); `tool_env` is `cc.env()`.
pub(super) fn merged_compiler_environ(
    c_environ: Option<&[(OsString, OsString)]>,
    tool_env: &[(OsString, OsString)],
) -> Vec<(OsString, OsString)> {
    let mut merged = match c_environ {
        Some(environ) => environ.to_vec(),
        None => std::env::vars_os().collect(),
    };
    for (name, value) in msvc_compiler_environ(tool_env, c_environ) {
        if let Some((_, existing)) = merged.iter_mut().find(|(key, _)| match name.to_str() {
            Some(name_str) => env_name_eq(key, name_str),
            None => key == &name,
        }) {
            *existing = value;
        } else {
            merged.push((name, value));
        }
    }
    merged
}

/// `MsvcPlatform.link` / `MsvcPlatform.masm`: PATH from `self.c_environ`
/// (`MsvcPlatform.__init__` `self.c_environ.update(msvc_compiler_environ)`),
/// then `cc::windows_registry::find_tool`, then the directory of `cl.exe`.
pub(super) fn discover_msvc_sidecar(
    cc: &cc::Tool,
    target: &str,
    name: &str,
    c_environ: Option<&[(OsString, OsString)]>,
) -> (Option<cc::Tool>, PathBuf) {
    // `_run_subprocess(..., env=self.c_environ)` after
    // `self.c_environ.update(msvc_compiler_environ)`.
    if let Some(found) = resolve_compiler_on_path(Path::new(name), c_environ) {
        return (None, found);
    }
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

/// `patch_os_env` (`windows.py`). Returns the `(name, new_value)` pairs
/// upstream would write into `os.environ`. The `not in` tests are substring
/// checks; PATH/INCLUDE/LIB use `;` and `\bin`/`\include`/`\lib` suffixes.
pub(super) fn patch_os_env(
    externals: &Path,
    get: impl Fn(&str) -> Option<String>,
) -> Vec<(String, String)> {
    let externals = externals.to_string_lossy();
    let binpath = format!(r"{externals}\bin");
    let mut out = Vec::new();
    // `patch_os_env` `os.environ['PATH']`.
    let path = get("PATH").unwrap_or_else(|| panic!("KeyError: 'PATH'"));
    if !path.contains(&binpath) {
        out.push(("PATH".to_string(), format!("{binpath};{path}")));
    }
    let include = get("INCLUDE").unwrap_or_default();
    if !include.contains(externals.as_ref()) {
        out.push((
            "INCLUDE".to_string(),
            format!(r"{externals}\include;{include}"),
        ));
    }
    let lib = get("LIB").unwrap_or_default();
    if !lib.contains(externals.as_ref()) {
        out.push(("LIB".to_string(), format!(r"{externals}\lib;{lib}")));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExternalCompilationInfo;
    use crate::platform::is_asm_source;
    use std::path::Path;

    fn msvc_x64() -> Platform {
        msvc_platform("x86_64-pc-windows-msvc".to_string(), true, None)
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
        let eci = ExternalCompilationInfo {
            library_dirs: vec![r"C:\libs".to_string()],
            libraries: vec!["foo".to_string()],
            link_files: vec![r"C:\prebuilt.obj".to_string()],
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

    #[test]
    fn patch_os_env_prepends_missing_and_skips_present() {
        let ext = Path::new(r"C:\externals");
        let without_bin = |name: &str| match name {
            "PATH" => Some(r"C:\Windows".to_string()),
            _ => None,
        };
        let patched = patch_os_env(ext, without_bin);
        let path = patched
            .iter()
            .find(|(k, _)| k == "PATH")
            .map(|(_, v)| v.as_str());
        assert_eq!(path, Some(r"C:\externals\bin;C:\Windows"));
        let include = patched
            .iter()
            .find(|(k, _)| k == "INCLUDE")
            .map(|(_, v)| v.as_str());
        assert_eq!(include, Some(r"C:\externals\include;"));

        let already = |name: &str| match name {
            "PATH" => Some(r"C:\externals\bin;C:\Windows".to_string()),
            "INCLUDE" => Some(r"C:\externals\include;".to_string()),
            "LIB" => Some(r"C:\externals\lib;".to_string()),
            _ => None,
        };
        let patched = patch_os_env(ext, already);
        assert!(patched.iter().all(|(k, _)| k != "PATH"));
    }

    #[test]
    #[should_panic(expected = "KeyError: 'PATH'")]
    fn patch_os_env_requires_path() {
        let ext = Path::new(r"C:\externals");
        let _ = patch_os_env(ext, |_| None);
    }

    #[test]
    fn msvc_c_environ_is_full_snapshot_mingw_is_none() {
        let msvc = msvc_x64();
        let env = msvc.c_environ.as_ref().expect("MsvcPlatform.c_environ");
        assert!(
            env.iter().any(|(k, _)| super::env_name_eq(k, "PATH")),
            "captured c_environ includes PATH"
        );
        let mingw = mingw_platform("x86_64-pc-windows-gnu".to_string());
        assert!(mingw.c_environ.is_none());
    }

    #[test]
    fn msvc_explicit_cc_is_stored_and_skips_find_msvc_env() {
        let plat = msvc_platform(
            "x86_64-pc-windows-msvc".to_string(),
            true,
            Some("explicit-cl.exe".into()),
        );
        assert_eq!(plat.cc_command.as_deref(), Some("explicit-cl.exe"));
        assert!(plat.c_environ.is_some());
        let none = msvc_platform("x86_64-pc-windows-msvc".to_string(), true, None);
        assert_eq!(none.cc_command.as_deref(), Some("cl.exe"));
    }

    #[test]
    fn msvc_compiler_environ_keeps_only_path_include_lib() {
        let tool = [
            (OsString::from("PATH"), OsString::from(r"C:\vc\bin;")),
            (OsString::from("FOO"), OsString::from("x")),
        ];
        let existing = [(
            OsString::from("PATH"),
            OsString::from(r"C:\ext\bin;C:\Windows"),
        )];
        let out = msvc_compiler_environ(&tool, Some(&existing));
        let path = out
            .iter()
            .find(|(k, _)| env_name_eq(k, "PATH"))
            .map(|(_, v)| v.as_os_str());
        assert_eq!(
            path,
            Some(std::ffi::OsStr::new(r"C:\vc\bin;C:\ext\bin;C:\Windows"))
        );
        assert!(
            out.iter().all(|(k, _)| !env_name_eq(k, "FOO")),
            "FOO is dropped: {out:?}"
        );

        let tool = [(OsString::from("path"), OsString::from(r"C:\vc\bin"))];
        let out = msvc_compiler_environ(&tool, Some(&[]));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, OsString::from("PATH"));
        assert_eq!(out[0].1, OsString::from(r"C:\vc\bin"));

        let tool = [(OsString::from("INCLUDE"), OsString::from(r"C:\vc\include"))];
        let existing = [(OsString::from("PATH"), OsString::from(r"C:\Windows"))];
        let out = msvc_compiler_environ(&tool, Some(&existing));
        let include = out
            .iter()
            .find(|(k, _)| env_name_eq(k, "INCLUDE"))
            .map(|(_, v)| v.as_os_str());
        assert_eq!(include, Some(std::ffi::OsStr::new(r"C:\vc\include")));
    }

    #[test]
    fn merged_compiler_environ_appends_snapshot_path() {
        let snapshot = [
            (OsString::from("PATH"), OsString::from("/orig")),
            (OsString::from("X"), OsString::from("1")),
        ];
        let tool_env = [(OsString::from("PATH"), OsString::from("/tool:/orig"))];
        let merged = merged_compiler_environ(Some(&snapshot), &tool_env);
        let path = merged
            .iter()
            .find(|(k, _)| env_name_eq(k, "PATH"))
            .map(|(_, v)| v.as_os_str());
        assert_eq!(path, Some(std::ffi::OsStr::new("/tool:/orig;/orig")));
        let x = merged
            .iter()
            .find(|(k, _)| env_name_eq(k, "X"))
            .map(|(_, v)| v.as_os_str());
        assert_eq!(x, Some(std::ffi::OsStr::new("1")));
    }
}
