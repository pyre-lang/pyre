//! `rpython/rtyper/tool/rffi_platform.py`: ask the C compiler from a
//! build script. Compilation uses `crate::gcc_cache`
//! (`build_executable_cache`, `try_compile_cache`) and
//! `crate::platform::CompilationError`; probe sources go under
//! `crate::udir::udir()` (`uniquefilepath`).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::ExternalCompilationInfo;
pub use crate::platform::CompilationError;

/// `rffi_platform.py` `C_HEADER`.
const C_HEADER: &str = r#"
#include <stdio.h>
#include <stddef.h>   /* for offsetof() */

void dump(char* key, int value) {
    printf("%s: %d\n", key, value);
}
"#;

/// The (sign, bits) class, `rffi.size_and_sign`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntType {
    pub signed: bool,
    pub bits: u32,
}

/// `rffi_platform.py` `CConfigEntry`.
pub enum CConfigEntry {
    /// `rffi_platform.py` `SimpleType`. The `ctype_hint` is always an integer
    /// type here, and `fixup_ctype` then lands on the class of the measured
    /// (size, sign), so the hint is not stored.
    SimpleType(String),
    /// `rffi_platform.py` `SizeOf`.
    SizeOf(String),
    /// `rffi_platform.py` `ConstantInteger`.
    ConstantInteger(String),
}

impl CConfigEntry {
    fn prepare_code(&self) -> Vec<String> {
        match self {
            CConfigEntry::SimpleType(name) => vec![
                format!("typedef {name} platcheck_t;"),
                String::new(),
                "platcheck_t x;".to_string(),
                "dump(\"size\",  sizeof(platcheck_t));".to_string(),
                "x = 0; x = ~x;".to_string(),
                "dump(\"unsigned\", x > 0);".to_string(),
            ],
            CConfigEntry::SizeOf(name) => {
                vec![format!("dump(\"size\",  sizeof({name}));")]
            }
            CConfigEntry::ConstantInteger(name) => vec![
                format!("if (({name}) <= 0) {{"),
                format!("    long long x = (long long)({name});"),
                "    printf(\"value: %lld\\n\", x);".to_string(),
                "} else {".to_string(),
                format!("    unsigned long long x = (unsigned long long)({name});"),
                "    printf(\"value: %llu\\n\", x);".to_string(),
                "}".to_string(),
            ],
        }
    }

    fn build_result(&self, info: &[(String, i128)]) -> Result<ConfigValue, CompilationError> {
        let get = |key: &str| {
            info.iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| *v)
                .unwrap_or_else(|| panic!("KeyError: {key}"))
        };
        match self {
            CConfigEntry::SimpleType(name) => {
                let size = get("size");
                let unsigned = get("unsigned") != 0;
                // `ctype_hint` is `rffi.INT` (4-byte signed). Keep it when
                // `(size, sign)` matches; otherwise `fixup_ctype`.
                let ctype = if size == 4 && !unsigned {
                    IntType {
                        signed: true,
                        bits: 32,
                    }
                } else {
                    fixup_ctype(name, size, unsigned)?
                };
                Ok(ConfigValue::Type(ctype))
            }
            CConfigEntry::SizeOf(_) => Ok(ConfigValue::Size(get("size") as u64)),
            CConfigEntry::ConstantInteger(_) => {
                Ok(ConfigValue::Int(expose_value_as_rpython(get("value"))?))
            }
        }
    }
}

/// `rffi_platform.py` `integer_class` / `fixup_ctype`. Members are
/// `SIGNEDCHAR`, `UCHAR`, `CHAR`, `SHORT`, `USHORT`, `INT`, `UINT`,
/// `INT_real`, `UINT_real`, `LONG`, `ULONG`, `LONGLONG`, `ULONGLONG`
/// (1, 2, 4, 8 bytes; no 128-bit type).
fn fixup_ctype(name: &str, size: i128, unsigned: bool) -> Result<IntType, CompilationError> {
    let bits = match size {
        1 => 8,
        2 => 16,
        4 => 32,
        8 => 64,
        _ => {
            return Err(CompilationError {
                out: String::new(),
                err: format!("conflicting field type for {name}"),
            });
        }
    };
    Ok(IntType {
        signed: !unsigned,
        bits,
    })
}

/// `rffi_platform.py` `expose_value_as_rpython`.
fn expose_value_as_rpython(value: i128) -> Result<i128, CompilationError> {
    if value >= i64::MIN as i128 && value <= u64::MAX as i128 {
        Ok(value)
    } else {
        Err(CompilationError {
            out: String::new(),
            err: format!("value {value} does not fit into any RPython integer type"),
        })
    }
}

/// `rffi_platform.py` `CConfigSingleEntry`.
pub enum CConfigSingleEntry {
    /// `rffi_platform.py` `Has`.
    Has(String),
}

impl CConfigSingleEntry {
    fn question(
        &self,
        ask_gcc: &mut dyn FnMut(&str) -> Result<(), CompilationError>,
    ) -> ConfigValue {
        match self {
            CConfigSingleEntry::Has(name) => {
                ConfigValue::Bool(ask_gcc(&format!("(void){name};")).is_ok())
            }
        }
    }
}

/// `rffi_platform.py` `configure` result values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigValue {
    Type(IntType),
    Size(u64),
    Bool(bool),
    Int(i128),
}

/// `rffi_platform.py` `CConfig`. Later attributes of the same name replace
/// earlier ones; `configure` sorts each list as `dir()` does.
pub struct CConfig {
    pub compilation_info: ExternalCompilationInfo,
    pub entries: Vec<(String, CConfigEntry)>,
    pub single_entries: Vec<(String, CConfigSingleEntry)>,
}

/// Collapse duplicate names (later wins) and sort as `dir()` does.
fn unique_sorted_by_name<T>(items: &[(String, T)]) -> Vec<(String, &T)> {
    let mut out: Vec<(String, &T)> = Vec::new();
    for (name, value) in items {
        if let Some(pos) = out.iter().position(|(n, _)| n == name) {
            out[pos].1 = value;
        } else {
            out.push((name.clone(), value));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// `rffi_platform.py` `configure`. The result has one value per name, in
/// `dir()` order. `ignore_errors` is passed to `configure_entries` /
/// `run_example_code` / `gcc_cache.build_executable_cache`. `TARGET` and
/// `HOST` both set and different is a cross build: `platform()` is the
/// host (`host_triple` reads `HOST`), and only `host` and `arm`
/// (`set_platform` / `translationoption.PLATFORMS`) exist.
pub fn configure(
    cconfig: &CConfig,
    ignore_errors: bool,
) -> Result<Vec<(String, ConfigValue)>, CompilationError> {
    if let (Ok(target), Ok(host)) = (std::env::var("TARGET"), std::env::var("HOST"))
        && target != host
    {
        return Err(CompilationError {
            out: String::new(),
            err: format!(
                "rffi_platform.configure: TARGET {target} is not the host {host}; probing it needs a platform for {target} (set_platform), and only host and arm exist"
            ),
        });
    }
    let entries = unique_sorted_by_name(&cconfig.entries);
    let single_entries = unique_sorted_by_name(&cconfig.single_entries);
    for (name, _) in &entries {
        if single_entries.iter().any(|(n, _)| n == name) {
            panic!("CConfig attribute {name} is both a CConfigEntry and a CConfigSingleEntry");
        }
    }
    let mut res = Vec::new();
    if !entries.is_empty() {
        let list: Vec<&CConfigEntry> = entries.iter().map(|(_, e)| *e).collect();
        let results = configure_entries(&list, &cconfig.compilation_info, ignore_errors)?;
        for ((key, _), result) in entries.iter().zip(results) {
            res.push((key.clone(), result));
        }
    }
    for (key, entry) in &single_entries {
        let mut writer = CWriter::new(cconfig.compilation_info.clone());
        writer.write_header();
        let mut writer = Some(writer);
        let result = entry.question(&mut |q| writer.take().unwrap().ask_gcc(q));
        res.push((key.clone(), result));
    }
    Ok(res)
}

/// `rffi_platform.py` `configure_entries`. The `ConfigResult` cache has no
/// second reader here, so call `build_result` directly. `ignore_errors`
/// is `run_example_code(..., ignore_errors=ignore_errors)`.
fn configure_entries(
    entries: &[&CConfigEntry],
    eci: &ExternalCompilationInfo,
    ignore_errors: bool,
) -> Result<Vec<ConfigValue>, CompilationError> {
    let mut writer = CWriter::new(eci.clone());
    writer.write_header();
    for (i, entry) in entries.iter().enumerate() {
        writer.write_entry(&i.to_string(), entry);
    }
    writer.start_main();
    for i in 0..entries.len() {
        writer.write_entry_main(&i.to_string());
    }
    writer.close()?;
    let infolist = run_example_code(&writer.path, eci, ignore_errors)?;
    assert_eq!(infolist.len(), entries.len());
    entries
        .iter()
        .zip(infolist)
        .map(|(entry, info)| entry.build_result(&info))
        .collect()
}

/// `rffi_platform.py` `_CWriter`.
struct CWriter {
    path: PathBuf,
    text: String,
    eci: ExternalCompilationInfo,
}

impl CWriter {
    fn new(eci: ExternalCompilationInfo) -> Self {
        CWriter {
            path: uniquefilepath(),
            text: String::new(),
            eci,
        }
    }

    fn write_header(&mut self) {
        self.eci.write_c_header(&mut self.text);
        self.text.push_str(C_HEADER);
        self.text.push('\n');
        self.text.push('\n');
    }

    fn write_entry(&mut self, key: &str, entry: &CConfigEntry) {
        self.text
            .push_str(&format!("void dump_section_{key}(void) {{\n"));
        for line in entry.prepare_code() {
            let line = if !line.is_empty() && !line.starts_with('#') {
                format!("\t{line}")
            } else {
                line
            };
            self.text.push_str(&line);
            self.text.push('\n');
        }
        self.text.push_str("}\n");
        self.text.push('\n');
    }

    fn write_entry_main(&mut self, key: &str) {
        self.text
            .push_str(&format!("\tprintf(\"-+- {key}\\n\");\n"));
        self.text.push_str(&format!("\tdump_section_{key}();\n"));
        self.text.push_str("\tprintf(\"---\\n\");\n");
    }

    fn start_main(&mut self) {
        self.text.push_str("int main(int argc, char *argv[]) {\n");
    }

    fn close(&mut self) -> Result<(), CompilationError> {
        self.text.push_str("\treturn 0;\n");
        self.text.push_str("}\n");
        fs::write(&self.path, &self.text).map_err(|err| CompilationError {
            out: String::new(),
            err: err.to_string(),
        })?;
        Ok(())
    }

    /// `rffi_platform.py` `_CWriter.ask_gcc`: `try_compile_cache([self.path], self.eci)`.
    fn ask_gcc(mut self, question: &str) -> Result<(), CompilationError> {
        self.start_main();
        self.text.push_str(question);
        self.text.push('\n');
        self.close()?;
        crate::gcc_cache::try_compile_cache(&[&self.path], &self.eci)
    }
}

/// `rffi_platform.py` `uniquefilepath(LAST=[0])`: `udir.join('platcheck_%d.c' % i)`.
fn uniquefilepath() -> PathBuf {
    static LAST: AtomicUsize = AtomicUsize::new(0);
    let i = LAST.fetch_add(1, Ordering::Relaxed);
    crate::udir::udir().join(format!("platcheck_{i}.c"))
}

/// `rffi_platform.py` `run_example_code`: `build_executable_cache(files, eci, ignore_errors=ignore_errors)` then parse stdout.
fn run_example_code(
    filepath: &Path,
    eci: &ExternalCompilationInfo,
    ignore_errors: bool,
) -> Result<Vec<Vec<(String, i128)>>, CompilationError> {
    let output = crate::gcc_cache::build_executable_cache(&[filepath], eci, ignore_errors)?;
    parse_example_output(&output)
}

/// `rffi_platform.py` `run_example_code` parsing loop.
fn parse_example_output(output: &str) -> Result<Vec<Vec<(String, i128)>>, CompilationError> {
    if !output.starts_with("-+- ") {
        return Err(CompilationError {
            out: String::new(),
            err: "run_example_code failed!".into(),
        });
    }
    let mut sections = Vec::new();
    let mut section: Option<Vec<(String, i128)>> = None;
    for line in output.lines() {
        let line = line.trim();
        if line.starts_with("-+- ") {
            section = Some(Vec::new());
        } else if line == "---" {
            let done = section.take().expect("---");
            sections.push(done);
        } else if !line.is_empty() {
            let section = section.as_mut().expect("line");
            let (key, value) = line.split_once(": ").unwrap_or_else(|| panic!("{line}"));
            let value: i128 = value.parse().unwrap_or_else(|_| panic!("{value}"));
            if let Some((_, slot)) = section.iter_mut().find(|(k, _)| k == key) {
                *slot = value;
            } else {
                section.push((key.to_string(), value));
            }
        }
    }
    Ok(sections)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rffi_platform_entries() {
        assert_eq!(
            CConfigEntry::SimpleType("mode_t".into()).prepare_code(),
            vec![
                "typedef mode_t platcheck_t;".to_string(),
                String::new(),
                "platcheck_t x;".to_string(),
                "dump(\"size\",  sizeof(platcheck_t));".to_string(),
                "x = 0; x = ~x;".to_string(),
                "dump(\"unsigned\", x > 0);".to_string(),
            ]
        );
        assert_eq!(
            CConfigEntry::SizeOf("long double".into()).prepare_code(),
            vec!["dump(\"size\",  sizeof(long double));".to_string()]
        );
        assert_eq!(
            CConfigEntry::ConstantInteger("EOF".into()).prepare_code(),
            vec![
                "if ((EOF) <= 0) {".to_string(),
                "    long long x = (long long)(EOF);".to_string(),
                "    printf(\"value: %lld\\n\", x);".to_string(),
                "} else {".to_string(),
                "    unsigned long long x = (unsigned long long)(EOF);".to_string(),
                "    printf(\"value: %llu\\n\", x);".to_string(),
                "}".to_string(),
            ]
        );

        let mut writer = CWriter::new(ExternalCompilationInfo::default());
        writer.write_header();
        writer.write_entry("0", &CConfigEntry::SizeOf("int".into()));
        writer.write_entry("1", &CConfigEntry::SizeOf("long".into()));
        writer.start_main();
        writer.write_entry_main("0");
        writer.write_entry_main("1");
        writer.close().unwrap();
        assert!(
            writer.text.contains("void dump_section_0(void) {"),
            "{}",
            writer.text
        );
        assert!(
            writer.text.contains("\tdump_section_1();"),
            "{}",
            writer.text
        );
        assert!(writer.text.ends_with("\treturn 0;\n}\n"), "{}", writer.text);

        let parsed =
            parse_example_output("-+- 0\nsize: 4\nunsigned: 1\n---\n-+- 1\nsize: 16\n---\n")
                .unwrap();
        assert_eq!(
            parsed,
            vec![
                vec![("size".into(), 4), ("unsigned".into(), 1)],
                vec![("size".into(), 16)],
            ]
        );
        let parsed_dup = parse_example_output("-+- 0\nsize: 4\nsize: 8\n---\n").unwrap();
        assert_eq!(parsed_dup, vec![vec![("size".into(), 8)]]);
        assert!(parse_example_output("oops").is_err());

        let st = CConfigEntry::SimpleType("mode_t".into());
        assert_eq!(
            st.build_result(&[("size".into(), 4), ("unsigned".into(), 1)])
                .unwrap(),
            ConfigValue::Type(IntType {
                signed: false,
                bits: 32
            })
        );
        assert_eq!(
            st.build_result(&[("size".into(), 8), ("unsigned".into(), 0)])
                .unwrap(),
            ConfigValue::Type(IntType {
                signed: true,
                bits: 64
            })
        );
        assert!(
            st.build_result(&[("size".into(), 16), ("unsigned".into(), 0)])
                .is_err()
        );

        #[cfg(unix)]
        {
            // `uniquefilepath` uses `udir()`; a leftover `PYPY_USESSION_DIR`
            // must not be required for gcc_cache to compile the probe.
            let previous_session = std::env::var_os("PYPY_USESSION_DIR");
            // SAFETY: this test is the only code touching `PYPY_USESSION_DIR`
            // around `configure`; `udir()` is a process-wide OnceLock.
            unsafe {
                std::env::remove_var("PYPY_USESSION_DIR");
            }
            let cconfig = CConfig {
                compilation_info: ExternalCompilationInfo {
                    includes: vec!["stdio.h".into()],
                    ..ExternalCompilationInfo::default()
                },
                entries: vec![
                    ("SIZE".into(), CConfigEntry::SizeOf("int".into())),
                    (
                        "TYPE".into(),
                        CConfigEntry::SimpleType("unsigned short".into()),
                    ),
                    ("EOF".into(), CConfigEntry::ConstantInteger("EOF".into())),
                ],
                single_entries: vec![
                    ("HAS".into(), CConfigSingleEntry::Has("printf".into())),
                    (
                        "NO".into(),
                        CConfigSingleEntry::Has("no_such_function_xyz".into()),
                    ),
                ],
            };
            let results = configure(&cconfig, false).unwrap();
            match previous_session {
                Some(value) => unsafe { std::env::set_var("PYPY_USESSION_DIR", value) },
                None => unsafe { std::env::remove_var("PYPY_USESSION_DIR") },
            }
            assert_eq!(
                results,
                vec![
                    ("EOF".into(), ConfigValue::Int(-1)),
                    ("SIZE".into(), ConfigValue::Size(4)),
                    (
                        "TYPE".into(),
                        ConfigValue::Type(IntType {
                            signed: false,
                            bits: 16
                        })
                    ),
                    ("HAS".into(), ConfigValue::Bool(true)),
                    ("NO".into(), ConfigValue::Bool(false)),
                ]
            );
        }
    }

    #[test]
    #[should_panic(
        expected = "CConfig attribute SIZE is both a CConfigEntry and a CConfigSingleEntry"
    )]
    fn cconfig_name_in_both_lists_panics() {
        let cconfig = CConfig {
            compilation_info: ExternalCompilationInfo::default(),
            entries: vec![("SIZE".into(), CConfigEntry::SizeOf("int".into()))],
            single_entries: vec![("SIZE".into(), CConfigSingleEntry::Has("printf".into()))],
        };
        let _ = configure(&cconfig, false);
    }
}
