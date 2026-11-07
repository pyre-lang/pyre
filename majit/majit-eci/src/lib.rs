//! `rpython/translator/tool/cbuild.py`: `ExternalCompilationInfo` and the C
//! build step ll2ctypes runs for it (`compile_shared_lib`).

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

pub mod ansi_print;
pub mod build;
pub mod cdecl;
pub mod gcc_cache;
pub mod parse;
pub mod platform;
pub mod rffi_platform;
pub mod taskengine;
pub mod udir;

/// `rffi.py generate_macro_wrapper`. `args` and `result` are `gettype`
/// templates. `LowLevelDatabase.gettype` FuncType skips `Void` (the resolved
/// type), so `params` is the `(ty, name)` pairs whose `ty != "void @"`,
/// `"void"` when none remain; `generate_macro_wrapper` still calls the macro
/// with every `argN`. The function template is `result` with `@` replaced by
/// `@(<params>)`, and `cdecl` places `pypy_macro_wrapper_{name}`. A `void @`
/// result omits `return`.
pub fn generate_macro_wrapper(name: &str, macro_: &str, args: &[String], result: &str) -> String {
    let argnames: Vec<String> = (0..args.len()).map(|i| format!("arg{i}")).collect();
    let params: Vec<String> = args
        .iter()
        .zip(&argnames)
        .filter(|(ty, _)| ty.as_str() != "void @")
        .map(|(ty, name)| cdecl::cdecl(ty, name))
        .collect();
    let params = if params.is_empty() {
        "void".to_string()
    } else {
        params.join(", ")
    };
    let argnames = argnames.join(", ");
    let template = result.replace('@', &format!("@({params})"));
    let decl = cdecl::cdecl(&template, &format!("pypy_macro_wrapper_{name}"));
    if result == "void @" {
        format!("RPY_EXTERN {decl} {{ {macro_}({argnames}); }}")
    } else {
        format!("RPY_EXTERN {decl} {{ return {macro_}({argnames}); }}")
    }
}

/// `cbuild.py` `from_pkg_config` / `from_config_tool` `ImportError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportError {
    pub message: String,
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ImportError {}

/// `cbuild.py` `ExternalCompilationInfo`. Fields follow `_ATTRIBUTES`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ExternalCompilationInfo {
    pub pre_include_bits: Vec<String>,
    pub includes: Vec<String>,
    pub include_dirs: Vec<String>,
    pub post_include_bits: Vec<String>,
    pub libraries: Vec<String>,
    pub library_dirs: Vec<String>,
    pub separate_module_sources: Vec<String>,
    pub separate_module_files: Vec<String>,
    pub compile_extra: Vec<String>,
    pub link_extra: Vec<String>,
    pub frameworks: Vec<String>,
    pub link_files: Vec<String>,
    pub testonly_libraries: Vec<String>,
    pub use_cpp_linker: bool,
    /// `_EXTRA_ATTRIBUTES` `platform`.
    pub platform: Option<String>,
}

/// `ExternalCompilationInfo._ATTRIBUTES`.
pub const _ATTRIBUTES: &[&str] = &[
    "pre_include_bits",
    "includes",
    "include_dirs",
    "post_include_bits",
    "libraries",
    "library_dirs",
    "separate_module_sources",
    "separate_module_files",
    "compile_extra",
    "link_extra",
    "frameworks",
    "link_files",
    "testonly_libraries",
];

/// `ExternalCompilationInfo._DUPLICATES_OK`.
pub const _DUPLICATES_OK: &[&str] = &["compile_extra", "link_extra"];

/// `repr()` of a `str`: single quotes unless the text contains `'` and no `"`; escapes `\`, the chosen quote,
/// `\t`, `\n`, `\r`, and other characters below U+0020 or U+007F as `\xNN`.
fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c == '\u{007F}' => {
                const HEX: &[u8; 16] = b"0123456789abcdef";
                let n = c as u32;
                out.push('\\');
                out.push('x');
                out.push(HEX[(n >> 4) as usize] as char);
                out.push(HEX[(n & 0xf) as usize] as char);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `py.path.local.sysfind`.
fn sysfind(execonfigtool: &str) -> Option<PathBuf> {
    if execonfigtool.contains('/') || execonfigtool.contains('\\') {
        let path = Path::new(execonfigtool);
        return path.is_file().then(|| path.to_path_buf());
    }
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(execonfigtool);
        if candidate.is_file() {
            return Some(candidate);
        }
        #[cfg(windows)]
        {
            for ext in ["exe", "bat", "cmd", "com"] {
                let candidate = dir.join(format!("{execonfigtool}.{ext}"));
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

/// `py.process.cmdexec`.
fn cmdexec(program: &Path, args: &[&str]) -> String {
    let output = Command::new(program)
        .args(args)
        .output()
        .unwrap_or_else(|err| panic!("{err}"));
    if !output.status.success() {
        panic!("{}", String::from_utf8_lossy(&output.stderr));
    }
    String::from_utf8_lossy(&output.stdout).into_owned()
}

impl ExternalCompilationInfo {
    fn field(&self, name: &str) -> &[String] {
        match name {
            "pre_include_bits" => &self.pre_include_bits,
            "includes" => &self.includes,
            "include_dirs" => &self.include_dirs,
            "post_include_bits" => &self.post_include_bits,
            "libraries" => &self.libraries,
            "library_dirs" => &self.library_dirs,
            "separate_module_sources" => &self.separate_module_sources,
            "separate_module_files" => &self.separate_module_files,
            "compile_extra" => &self.compile_extra,
            "link_extra" => &self.link_extra,
            "frameworks" => &self.frameworks,
            "link_files" => &self.link_files,
            "testonly_libraries" => &self.testonly_libraries,
            other => panic!("unknown ExternalCompilationInfo field `{other}`"),
        }
    }

    fn field_mut(&mut self, name: &str) -> &mut Vec<String> {
        match name {
            "pre_include_bits" => &mut self.pre_include_bits,
            "includes" => &mut self.includes,
            "include_dirs" => &mut self.include_dirs,
            "post_include_bits" => &mut self.post_include_bits,
            "libraries" => &mut self.libraries,
            "library_dirs" => &mut self.library_dirs,
            "separate_module_sources" => &mut self.separate_module_sources,
            "separate_module_files" => &mut self.separate_module_files,
            "compile_extra" => &mut self.compile_extra,
            "link_extra" => &mut self.link_extra,
            "frameworks" => &mut self.frameworks,
            "link_files" => &mut self.link_files,
            "testonly_libraries" => &mut self.testonly_libraries,
            other => panic!("unknown ExternalCompilationInfo field `{other}`"),
        }
    }

    /// `ExternalCompilationInfo._value`: `_ATTRIBUTES + _EXTRA_ATTRIBUTES`
    /// in declaration order.
    pub fn _value(
        &self,
    ) -> (
        &[String],
        &[String],
        &[String],
        &[String],
        &[String],
        &[String],
        &[String],
        &[String],
        &[String],
        &[String],
        &[String],
        &[String],
        &[String],
        bool,
        Option<&str>,
    ) {
        (
            &self.pre_include_bits,
            &self.includes,
            &self.include_dirs,
            &self.post_include_bits,
            &self.libraries,
            &self.library_dirs,
            &self.separate_module_sources,
            &self.separate_module_files,
            &self.compile_extra,
            &self.link_extra,
            &self.frameworks,
            &self.link_files,
            &self.testonly_libraries,
            self.use_cpp_linker,
            self.platform.as_deref(),
        )
    }

    /// `ExternalCompilationInfo.merge`. Duplicate `others` are dropped first;
    /// `_DUPLICATES_OK` fields concatenate; every other field keeps first
    /// occurrence order without duplicates; `use_cpp_linker` is the `or`.
    /// Mixed `platform` panics (`ExternalCompilationInfo.merge`).
    pub fn merge(&self, others: &[&ExternalCompilationInfo]) -> ExternalCompilationInfo {
        let mut uniq: Vec<&ExternalCompilationInfo> = Vec::new();
        for &other in others {
            if !uniq.contains(&other) {
                uniq.push(other);
            }
        }

        let mut out = ExternalCompilationInfo {
            use_cpp_linker: self.use_cpp_linker || uniq.iter().any(|other| other.use_cpp_linker),
            ..ExternalCompilationInfo::default()
        };
        for name in _ATTRIBUTES {
            if _DUPLICATES_OK.contains(name) {
                let mut s = self.field(name).to_vec();
                for other in &uniq {
                    s.extend(other.field(name).iter().cloned());
                }
                *out.field_mut(name) = s;
            } else {
                let mut attr = Vec::new();
                for one in std::iter::once(self).chain(uniq.iter().copied()) {
                    for elem in one.field(name) {
                        if !attr.contains(elem) {
                            attr.push(elem.clone());
                        }
                    }
                }
                *out.field_mut(name) = attr;
            }
        }
        for other in &uniq {
            if other.platform != self.platform {
                panic!(
                    "Mixing ECI for different platforms {:?} and {:?}",
                    other.platform, self.platform
                );
            }
        }
        out.platform = self.platform.clone();
        out
    }

    /// `ExternalCompilationInfo.from_compiler_flags`.
    pub fn from_compiler_flags(flags: &str) -> Result<Self, String> {
        let mut pre_include_bits = Vec::new();
        let mut include_dirs = Vec::new();
        let mut compile_extra = Vec::new();
        for arg in flags.split_whitespace() {
            if let Some(dir) = arg.strip_prefix("-I") {
                include_dirs.push(dir.to_string());
            } else if let Some(macro_def) = arg.strip_prefix("-D") {
                let parts: Vec<&str> = macro_def.split('=').collect();
                let (macro_name, value) = match parts.as_slice() {
                    [name] => (*name, "1"),
                    [name, value] => (*name, *value),
                    _ => {
                        return Err("too many values to unpack (expected 2)".to_string());
                    }
                };
                if macro_name == "_XOPEN_SOURCE" {
                    continue;
                }
                pre_include_bits.push(format!("#define {macro_name} {value}"));
            } else if arg.starts_with("-L") || arg.starts_with("-l") {
                return Err(format!(
                    "linker flag found in compiler options: {}",
                    py_repr(arg)
                ));
            } else {
                compile_extra.push(arg.to_string());
            }
        }
        Ok(ExternalCompilationInfo {
            pre_include_bits,
            include_dirs,
            compile_extra,
            ..ExternalCompilationInfo::default()
        })
    }

    /// `ExternalCompilationInfo.from_linker_flags`.
    pub fn from_linker_flags(flags: &str) -> Result<Self, String> {
        let mut libraries = Vec::new();
        let mut library_dirs = Vec::new();
        let mut link_extra = Vec::new();
        for arg in flags.split_whitespace() {
            if let Some(dir) = arg.strip_prefix("-L") {
                library_dirs.push(dir.to_string());
            } else if let Some(lib) = arg.strip_prefix("-l") {
                libraries.push(lib.to_string());
            } else if arg.starts_with("-I") || arg.starts_with("-D") {
                return Err(format!(
                    "compiler flag found in linker options: {}",
                    py_repr(arg)
                ));
            } else {
                link_extra.push(arg.to_string());
            }
        }
        Ok(ExternalCompilationInfo {
            libraries,
            library_dirs,
            link_extra,
            ..ExternalCompilationInfo::default()
        })
    }

    /// `ExternalCompilationInfo.from_config_tool`.
    pub fn from_config_tool(execonfigtool: &str) -> Result<Self, ImportError> {
        let path = sysfind(execonfigtool).ok_or_else(|| ImportError {
            message: format!("cannot find {}", py_repr(execonfigtool)),
        })?;
        Self::run_config_tool(&path, &[])
    }

    /// `ExternalCompilationInfo.from_pkg_config`.
    pub fn from_pkg_config(pkgname: &str) -> Result<Self, ImportError> {
        let result = match Command::new("pkg-config")
            .args([pkgname, "--exists"])
            .status()
        {
            Ok(status) => status.code().unwrap_or(-1),
            Err(_) => -1,
        };
        if result != 0 {
            return Err(ImportError {
                message: format!("failed: 'pkg-config {pkgname} --exists'"),
            });
        }
        Self::run_config_tool(Path::new("pkg-config"), &[pkgname])
    }

    /// `ExternalCompilationInfo._run_config_tool`.
    fn run_config_tool(program: &Path, extra: &[&str]) -> Result<Self, ImportError> {
        let mut cflags_args = extra.to_vec();
        cflags_args.push("--cflags");
        let cflags = cmdexec(program, &cflags_args);
        let eci1 = Self::from_compiler_flags(cflags.trim()).unwrap_or_else(|err| panic!("{err}"));
        let mut libs_args = extra.to_vec();
        libs_args.push("--libs");
        let libs = cmdexec(program, &libs_args);
        let eci2 = Self::from_linker_flags(libs.trim()).unwrap_or_else(|err| panic!("{err}"));
        Ok(eci1.merge(&[&eci2]))
    }

    /// `ExternalCompilationInfo.write_c_header`.
    pub fn write_c_header(&self, out: &mut String) {
        out.push_str(include_str!("precommondefs.h"));
        out.push('\n');
        for piece in &self.pre_include_bits {
            out.push_str(piece);
            out.push('\n');
        }
        for path in &self.includes {
            out.push_str("#include <");
            out.push_str(path);
            out.push_str(">\n");
        }
        for piece in &self.post_include_bits {
            out.push_str(piece);
            out.push('\n');
        }
    }

    /// `ExternalCompilationInfo.convert_sources_to_files`.
    pub fn convert_sources_to_files(&self, cache_dir: &Path) -> ExternalCompilationInfo {
        if self.separate_module_sources.is_empty() {
            return self.clone();
        }
        let mut files = self.separate_module_files.clone();
        let mut num = 0;
        for source in &self.separate_module_sources {
            let filename = loop {
                let filename = cache_dir.join(format!("module_{num}.c"));
                num += 1;
                if !filename.exists() {
                    break filename;
                }
            };
            let mut body = String::new();
            self.write_c_header(&mut body);
            body.push_str(source);
            if !source.ends_with('\n') {
                body.push('\n');
            }
            if let Err(err) = std::fs::write(&filename, &body) {
                panic!("{}: {err}", filename.display());
            }
            match filename.to_str() {
                Some(path) => files.push(path.to_owned()),
                None => panic!("{}: path is not UTF-8", filename.display()),
            }
        }
        let mut out = self.clone();
        out.separate_module_sources = Vec::new();
        out.separate_module_files = files;
        out
    }
}

/// `ExternalCompilationInfo.__repr__`.
impl fmt::Display for ExternalCompilationInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let attrs = [
            format!("pre_include_bits={:?}", self.pre_include_bits),
            format!("includes={:?}", self.includes),
            format!("include_dirs={:?}", self.include_dirs),
            format!("post_include_bits={:?}", self.post_include_bits),
            format!("libraries={:?}", self.libraries),
            format!("library_dirs={:?}", self.library_dirs),
            format!("separate_module_sources={:?}", self.separate_module_sources),
            format!("separate_module_files={:?}", self.separate_module_files),
            format!("compile_extra={:?}", self.compile_extra),
            format!("link_extra={:?}", self.link_extra),
            format!("frameworks={:?}", self.frameworks),
            format!("link_files={:?}", self.link_files),
            format!("testonly_libraries={:?}", self.testonly_libraries),
            format!("use_cpp_linker={:?}", self.use_cpp_linker),
            format!("platform={:?}", self.platform),
        ];
        write!(f, "<ExternalCompilationInfo ({})>", attrs.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_header_and_sources() {
        let a = ExternalCompilationInfo {
            includes: vec!["stdio.h".into(), "stdlib.h".into()],
            include_dirs: vec!["inc".into(), "inc".into()],
            libraries: vec!["foo".into()],
            compile_extra: vec!["-O2".into()],
            link_extra: vec!["-lm".into()],
            pre_include_bits: vec!["#define FOO 1".into()],
            post_include_bits: vec!["typedef int X;".into()],
            use_cpp_linker: false,
            platform: Some("host".into()),
            ..ExternalCompilationInfo::default()
        };
        let b = ExternalCompilationInfo {
            includes: vec!["stdlib.h".into(), "math.h".into()],
            include_dirs: vec!["inc".into(), "other".into()],
            libraries: vec!["foo".into(), "bar".into()],
            compile_extra: vec!["-O2".into(), "-g".into()],
            link_extra: vec!["-lm".into(), "-ldl".into()],
            use_cpp_linker: true,
            platform: Some("host".into()),
            ..ExternalCompilationInfo::default()
        };
        let merged = a.merge(&[&b, &b]);
        assert_eq!(
            merged.includes,
            vec![
                "stdio.h".to_string(),
                "stdlib.h".to_string(),
                "math.h".to_string()
            ]
        );
        assert_eq!(
            merged.include_dirs,
            vec!["inc".to_string(), "other".to_string()]
        );
        assert_eq!(merged.libraries, vec!["foo".to_string(), "bar".to_string()]);
        assert_eq!(
            merged.compile_extra,
            vec!["-O2".to_string(), "-O2".to_string(), "-g".to_string()]
        );
        assert_eq!(
            merged.link_extra,
            vec!["-lm".to_string(), "-lm".to_string(), "-ldl".to_string()]
        );
        assert!(merged.use_cpp_linker);
        assert_eq!(merged.platform.as_deref(), Some("host"));

        let mut header = String::new();
        a.write_c_header(&mut header);
        let pre = include_str!("precommondefs.h");
        assert!(header.starts_with(pre), "{header}");
        let rest = &header[pre.len()..];
        assert_eq!(
            rest,
            "\n#define FOO 1\n#include <stdio.h>\n#include <stdlib.h>\ntypedef int X;\n"
        );

        let dir = std::env::temp_dir().join(format!("majit-eci-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("module_0.c"), "occupied\n").unwrap();
        let with_src = ExternalCompilationInfo {
            includes: vec!["string.h".into()],
            separate_module_sources: vec!["int f(void) { return 0; }".into()],
            ..ExternalCompilationInfo::default()
        };
        let converted = with_src.convert_sources_to_files(&dir);
        assert!(converted.separate_module_sources.is_empty());
        assert_eq!(converted.separate_module_files.len(), 1);
        let written = Path::new(&converted.separate_module_files[0]);
        assert_eq!(
            written.file_name().and_then(|n| n.to_str()),
            Some("module_1.c")
        );
        let body = std::fs::read_to_string(written).unwrap();
        assert!(body.starts_with(include_str!("precommondefs.h")), "{body}");
        assert!(body.ends_with("int f(void) { return 0; }\n"), "{body}");
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(
            generate_macro_wrapper(
                "FD_ISSET",
                "FD_ISSET",
                &["int @".into(), "fd_set *@".into()],
                "int @"
            ),
            "RPY_EXTERN int pypy_macro_wrapper_FD_ISSET(int arg0, fd_set *arg1) { return FD_ISSET(arg0, arg1); }"
        );
        assert_eq!(
            generate_macro_wrapper("FD_ZERO", "FD_ZERO", &["fd_set *@".into()], "void @"),
            "RPY_EXTERN void pypy_macro_wrapper_FD_ZERO(fd_set *arg0) { FD_ZERO(arg0); }"
        );
        assert_eq!(
            generate_macro_wrapper("f", "F", &["void @".into(), "int @".into()], "int @"),
            "RPY_EXTERN int pypy_macro_wrapper_f(int arg1) { return F(arg0, arg1); }"
        );

        let none: &[String] = &[];
        let resolve_extern = |path: &syn::Path, _: &[String]| -> Option<cdecl::Resolved> {
            let segs: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
            match segs.first().map(String::as_str) {
                Some("libc" | "std") => Some(cdecl::Resolved::Extern(segs)),
                _ => None,
            }
        };
        assert_eq!(
            cdecl::gettype(
                &syn::parse_str("*mut libc::stat").unwrap(),
                none,
                &resolve_extern
            )
            .unwrap(),
            "struct stat *@"
        );
        assert_eq!(
            cdecl::gettype(
                &syn::parse_str("*const libc::c_char").unwrap(),
                none,
                &resolve_extern
            )
            .unwrap(),
            "const char *@"
        );
        assert_eq!(
            cdecl::gettype(
                &syn::parse_str("*const *mut libc::c_char").unwrap(),
                none,
                &resolve_extern
            )
            .unwrap(),
            "char *const *@"
        );
        assert_eq!(
            cdecl::gettype(
                &syn::parse_str("Option<unsafe extern \"C\" fn(libc::c_int) -> libc::c_int>")
                    .unwrap(),
                none,
                &resolve_extern
            )
            .unwrap(),
            "int (*@)(int)"
        );
        let resolve_int = |path: &syn::Path, module_path: &[String]| -> Option<cdecl::Resolved> {
            if path.segments.len() == 1 && path.segments[0].ident == "INT" {
                Some(cdecl::Resolved::Alias(
                    syn::parse_str("std::ffi::c_int").unwrap(),
                    Vec::new(),
                ))
            } else {
                resolve_extern(path, module_path)
            }
        };
        assert_eq!(
            cdecl::gettype(&syn::parse_str("INT").unwrap(), none, &resolve_int).unwrap(),
            "int @"
        );
        let resolve_foo = |_: &syn::Path, _: &[String]| -> Option<cdecl::Resolved> {
            Some(cdecl::Resolved::LocalStruct("Foo".into()))
        };
        assert_eq!(
            cdecl::gettype(&syn::parse_str("Foo").unwrap(), none, &resolve_foo).unwrap(),
            "struct Foo @"
        );
        let array4 =
            cdecl::gettype(&syn::parse_str("[i32; 4]").unwrap(), none, &resolve_extern).unwrap();
        assert_eq!(array4, "int (@)[4]");
        assert_eq!(cdecl::cdecl(&array4, "x"), "int x[4]");
        assert_eq!(
            cdecl::gettype(
                &syn::parse_str("*mut [u8; 16]").unwrap(),
                none,
                &resolve_extern
            )
            .unwrap(),
            "unsigned char *@"
        );
        assert_eq!(
            cdecl::gettype(
                &syn::parse_str("*const [i32; 2]").unwrap(),
                none,
                &resolve_extern
            )
            .unwrap(),
            "const int *@"
        );
        let array_n = cdecl::gettype(&syn::parse_str("[i32; N]").unwrap(), none, &resolve_extern);
        assert!(
            array_n
                .as_ref()
                .err()
                .is_some_and(|e| e.starts_with("unsupported array length")),
            "{array_n:?}"
        );
        assert_eq!(
            cdecl::gettype(
                &syn::parse_str("*mut [i32; N]").unwrap(),
                none,
                &resolve_extern
            )
            .unwrap(),
            "int *@"
        );
        let resolve_a_foo = |path: &syn::Path, _: &[String]| -> Option<cdecl::Resolved> {
            if path.segments.len() != 1 {
                return None;
            }
            match path.segments[0].ident.to_string().as_str() {
                "A" => Some(cdecl::Resolved::Alias(
                    syn::parse_str("[i32; 4]").unwrap(),
                    vec![],
                )),
                "Foo" => Some(cdecl::Resolved::LocalStruct("Foo".into())),
                _ => None,
            }
        };
        assert_eq!(
            cdecl::gettype(&syn::parse_str("*mut A").unwrap(), none, &resolve_a_foo).unwrap(),
            "int *@"
        );
        assert_eq!(
            cdecl::gettype(&syn::parse_str("*const A").unwrap(), none, &resolve_a_foo).unwrap(),
            "const int *@"
        );
        let result_arr =
            cdecl::functype_result(&syn::parse_str("[i32; 4]").unwrap(), none, &resolve_a_foo);
        assert!(
            result_arr
                .as_ref()
                .err()
                .is_some_and(|e| e.starts_with("function result can only be primitive or pointer")),
            "{result_arr:?}"
        );
        let result_a = cdecl::functype_result(&syn::parse_str("A").unwrap(), none, &resolve_a_foo);
        assert!(
            result_a
                .as_ref()
                .err()
                .is_some_and(|e| e.starts_with("function result can only be primitive or pointer")),
            "{result_a:?}"
        );
        assert_eq!(
            cdecl::gettype(&syn::parse_str("*mut Foo").unwrap(), none, &resolve_a_foo).unwrap(),
            "struct Foo *@"
        );
        assert_eq!(
            cdecl::functype_result(&syn::parse_str("()").unwrap(), none, &resolve_a_foo).unwrap(),
            "void @"
        );
        assert_eq!(
            cdecl::gettype(
                &syn::parse_str("extern \"C\" fn(Foo, [i32; 4])").unwrap(),
                none,
                &resolve_a_foo
            )
            .unwrap(),
            "void (*@)(struct Foo, int [4])"
        );
        assert_eq!(
            cdecl::gettype(
                &syn::parse_str("extern \"C\" fn((), i32) -> i32").unwrap(),
                none,
                &resolve_extern
            )
            .unwrap(),
            "int (*@)(int)"
        );
        let resolve_v = |path: &syn::Path, _: &[String]| -> Option<cdecl::Resolved> {
            if path.segments.len() == 1 && path.segments[0].ident == "V" {
                Some(cdecl::Resolved::Alias(
                    syn::parse_str("()").unwrap(),
                    vec![],
                ))
            } else {
                None
            }
        };
        assert_eq!(
            cdecl::gettype(
                &syn::parse_str("extern \"C\" fn(V, i32) -> i32").unwrap(),
                none,
                &resolve_v
            )
            .unwrap(),
            "int (*@)(int)"
        );
        assert_eq!(
            cdecl::gettype(
                &syn::parse_str("extern \"C\" fn(())").unwrap(),
                none,
                &resolve_extern
            )
            .unwrap(),
            "void (*@)(void)"
        );
        let result_stat = cdecl::functype_result(
            &syn::parse_str("libc::stat").unwrap(),
            none,
            &resolve_extern,
        );
        assert_eq!(
            result_stat,
            Err("function result can only be primitive or pointer".into())
        );
        let resolve_rffi = |path: &syn::Path, _: &[String]| -> Option<cdecl::Resolved> {
            let segs: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
            if segs.first().map(String::as_str) == Some("majit_rlib") {
                Some(cdecl::Resolved::Extern(segs))
            } else {
                None
            }
        };
        assert_eq!(
            cdecl::gettype(
                &syn::parse_str("majit_rlib::rffi::INT").unwrap(),
                none,
                &resolve_rffi
            )
            .unwrap(),
            "int @"
        );
        assert_eq!(
            cdecl::gettype(
                &syn::parse_str("majit_rlib::rffi::CCHARP").unwrap(),
                none,
                &resolve_rffi
            )
            .unwrap(),
            "char *@"
        );
        #[cfg(all(target_pointer_width = "64", not(windows)))]
        {
            assert_eq!(
                cdecl::gettype(
                    &syn::parse_str("majit_rlib::rffi::LONG").unwrap(),
                    none,
                    &resolve_rffi
                )
                .unwrap(),
                "Signed @"
            );
            assert_eq!(
                cdecl::gettype(
                    &syn::parse_str("majit_rlib::rffi::SIZE_T").unwrap(),
                    none,
                    &resolve_rffi
                )
                .unwrap(),
                "size_t @"
            );
            assert_eq!(
                cdecl::gettype(
                    &syn::parse_str("majit_rlib::rffi::UNSIGNED").unwrap(),
                    none,
                    &resolve_rffi
                )
                .unwrap(),
                "size_t @"
            );
            assert_eq!(
                cdecl::gettype(
                    &syn::parse_str("majit_rlib::rffi::INT").unwrap(),
                    none,
                    &resolve_rffi
                )
                .unwrap(),
                "int @"
            );
            assert_eq!(
                cdecl::gettype(&syn::parse_str("usize").unwrap(), none, &resolve_extern).unwrap(),
                "size_t @"
            );
            assert_eq!(
                cdecl::gettype(
                    &syn::parse_str("libc::c_long").unwrap(),
                    none,
                    &resolve_extern
                )
                .unwrap(),
                "Signed @"
            );
            assert_eq!(
                cdecl::gettype(
                    &syn::parse_str("libc::ssize_t").unwrap(),
                    none,
                    &resolve_extern
                )
                .unwrap(),
                "Signed @"
            );
            assert_eq!(
                cdecl::gettype(
                    &syn::parse_str("libc::size_t").unwrap(),
                    none,
                    &resolve_extern
                )
                .unwrap(),
                "size_t @"
            );
            assert_eq!(
                cdecl::gettype(
                    &syn::parse_str("libc::uintptr_t").unwrap(),
                    none,
                    &resolve_extern
                )
                .unwrap(),
                "size_t @"
            );
            assert_eq!(
                cdecl::gettype(&syn::parse_str("u8").unwrap(), none, &resolve_extern).unwrap(),
                "unsigned char @"
            );
            assert_eq!(
                cdecl::gettype(
                    &syn::parse_str("majit_rlib::rffi::CHAR").unwrap(),
                    none,
                    &resolve_rffi
                )
                .unwrap(),
                "char @"
            );
        }
        let resolve_other_rffi = |path: &syn::Path, _: &[String]| -> Option<cdecl::Resolved> {
            let segs: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
            Some(cdecl::Resolved::Extern(segs))
        };
        assert_eq!(
            cdecl::gettype(
                &syn::parse_str("other_crate::rffi::INT").unwrap(),
                none,
                &resolve_other_rffi
            )
            .unwrap(),
            "INT @"
        );
        let resolve_cycle = |path: &syn::Path, _: &[String]| -> Option<cdecl::Resolved> {
            if path.segments.len() != 1 {
                return None;
            }
            match path.segments[0].ident.to_string().as_str() {
                "X" => Some(cdecl::Resolved::Alias(syn::parse_str("Y").unwrap(), vec![])),
                "Y" => Some(cdecl::Resolved::Alias(syn::parse_str("X").unwrap(), vec![])),
                _ => None,
            }
        };
        let cycle = cdecl::gettype(&syn::parse_str("*mut X").unwrap(), none, &resolve_cycle);
        assert!(
            cycle
                .as_ref()
                .err()
                .is_some_and(|e| e.starts_with("alias cycle")),
            "{cycle:?}"
        );
        let flags = ExternalCompilationInfo::from_compiler_flags(
            "-I/x -DFOO -DBAR=2 -D_XOPEN_SOURCE=700 -O2",
        )
        .unwrap();
        assert_eq!(flags.include_dirs, vec!["/x".to_string()]);
        assert_eq!(
            flags.pre_include_bits,
            vec!["#define FOO 1".to_string(), "#define BAR 2".to_string()]
        );
        assert_eq!(flags.compile_extra, vec!["-O2".to_string()]);
        assert!(ExternalCompilationInfo::from_compiler_flags("-DFOO=a=b").is_err());
        assert!(ExternalCompilationInfo::from_compiler_flags("-D_XOPEN_SOURCE=a=b").is_err());
        assert_eq!(
            ExternalCompilationInfo::from_compiler_flags("-lfoo"),
            Err("linker flag found in compiler options: '-lfoo'".to_string())
        );
        assert_eq!(
            ExternalCompilationInfo::from_compiler_flags("-lfoo'"),
            Err("linker flag found in compiler options: \"-lfoo'\"".to_string())
        );
        let link = ExternalCompilationInfo::from_linker_flags("-L/y -lbar -Wl,-z").unwrap();
        assert_eq!(link.library_dirs, vec!["/y".to_string()]);
        assert_eq!(link.libraries, vec!["bar".to_string()]);
        assert_eq!(link.link_extra, vec!["-Wl,-z".to_string()]);
        assert_eq!(
            ExternalCompilationInfo::from_linker_flags("-DX"),
            Err("compiler flag found in linker options: '-DX'".to_string())
        );
        assert_eq!(
            ExternalCompilationInfo::from_linker_flags("-DX'\""),
            Err(r#"compiler flag found in linker options: '-DX\'"'"#.to_string())
        );
        assert_eq!(
            ExternalCompilationInfo::from_linker_flags(r"-I\x"),
            Err(r"compiler flag found in linker options: '-I\\x'".to_string())
        );
    }

    #[test]
    fn from_missing_config_tool() {
        let err = ExternalCompilationInfo::from_config_tool("dxowqbncpqympqhe-config").unwrap_err();
        assert_eq!(
            err.message,
            "cannot find 'dxowqbncpqympqhe-config'".to_string()
        );
    }

    #[test]
    fn from_pkg_config() {
        let result = match std::process::Command::new("pkg-config")
            .args(["ncurses", "--exists"])
            .status()
        {
            Ok(status) => status.code().unwrap_or(-1),
            Err(_) => -1,
        };
        if result != 0 {
            return;
        }
        let eci = ExternalCompilationInfo::from_pkg_config("ncurses").unwrap();
        assert!(
            eci.libraries.iter().any(|lib| lib.contains("ncurses")),
            "{eci:?}"
        );
    }

    #[test]
    #[should_panic(expected = "Mixing ECI for different platforms")]
    fn merge_rejects_mixed_platforms() {
        let left = ExternalCompilationInfo {
            platform: Some("host".to_string()),
            ..ExternalCompilationInfo::default()
        };
        let right = ExternalCompilationInfo {
            platform: Some("other".to_string()),
            ..ExternalCompilationInfo::default()
        };
        let _ = left.merge(&[&right]);
    }

    #[test]
    fn repr_and_value_cover_every_attribute() {
        let eci = ExternalCompilationInfo {
            includes: vec!["stdio.h".to_string()],
            use_cpp_linker: true,
            platform: Some("host".to_string()),
            ..ExternalCompilationInfo::default()
        };
        let rendered = eci.to_string();
        assert!(rendered.starts_with("<ExternalCompilationInfo ("));
        assert!(rendered.contains("includes="));
        assert!(rendered.contains("use_cpp_linker=true"));
        assert!(rendered.contains("platform=Some(\"host\")"));
        let value = eci._value();
        assert_eq!(value.1, &["stdio.h".to_string()][..]);
        assert!(value.13);
        assert_eq!(value.14, Some("host"));
    }
}
