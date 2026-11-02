//! `rpython/translator/tool/cbuild.py`: `ExternalCompilationInfo` and the C
//! build step ll2ctypes runs for it (`compile_shared_lib`).

use std::path::Path;

pub mod build;
pub mod cdecl;
pub mod parse;

/// `rffi.py generate_macro_wrapper`. `args` and `result` are `gettype`
/// templates; the function template is `result` with `@` replaced by
/// `@(<params>)`, and `cdecl` places `pypy_macro_wrapper_{name}`. Empty
/// `args` is `void`; a `void @` result omits `return`.
pub fn generate_macro_wrapper(name: &str, macro_: &str, args: &[String], result: &str) -> String {
    let argnames: Vec<String> = (0..args.len()).map(|i| format!("arg{i}")).collect();
    let params = if args.is_empty() {
        "void".to_string()
    } else {
        args.iter()
            .zip(&argnames)
            .map(|(ty, name)| cdecl::cdecl(ty, name))
            .collect::<Vec<_>>()
            .join(", ")
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

/// `cbuild.py` `ExternalCompilationInfo`. Fields follow `_ATTRIBUTES`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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

    /// `ExternalCompilationInfo.merge`. Duplicate `others` are dropped first;
    /// `_DUPLICATES_OK` fields concatenate; every other field keeps first
    /// occurrence order without duplicates; `use_cpp_linker` is the `or`.
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
        out
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_header_and_sources() {
        let a = ExternalCompilationInfo {
            includes: vec!["stdio.h".into(), "stdlib.h".into()],
            libraries: vec!["foo".into()],
            compile_extra: vec!["-O2".into()],
            link_extra: vec!["-lm".into()],
            pre_include_bits: vec!["#define FOO 1".into()],
            post_include_bits: vec!["typedef int X;".into()],
            use_cpp_linker: false,
            ..ExternalCompilationInfo::default()
        };
        let b = ExternalCompilationInfo {
            includes: vec!["stdlib.h".into(), "math.h".into()],
            libraries: vec!["foo".into(), "bar".into()],
            compile_extra: vec!["-g".into()],
            link_extra: vec!["-ldl".into()],
            use_cpp_linker: true,
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
        assert_eq!(merged.libraries, vec!["foo".to_string(), "bar".to_string()]);
        assert_eq!(
            merged.compile_extra,
            vec!["-O2".to_string(), "-g".to_string()]
        );
        assert_eq!(
            merged.link_extra,
            vec!["-lm".to_string(), "-ldl".to_string()]
        );
        assert!(merged.use_cpp_linker);

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
    }
}
