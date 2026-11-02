//! RPython `rpython/rtyper/tool/rfficache.py` bootstrapping helpers.
//!
//! `sizeof_c_types` follows `pick_platform`: a native target runs `ask_gcc`
//! → `gcc_cache.build_executable_cache`; a cross target whose OS is
//! `"unknown"` (no C toolchain; `wasm32-unknown-unknown` today) uses a
//! rustc LLVM-IR probe of `core::ffi` sizes as pyre's only ported cross
//! platform; any other cross target is `ValueError("platform = %s")`.
//! The rustc cache uses `gcc_cache.cache_file_path` / `try_atomic_write`
//! with a rustc identity in place of `Platform.key`.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, OnceLock};

use crate::annotator::model::KnownType;
use crate::rlib::rarithmetic;
use crate::translator::rtyper::lltypesystem::lltype::{self, LowLevelType};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NumberType {
    pub name: String,
    pub signed: bool,
    pub bits: i64,
    pub lltype: LowLevelType,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Platform {
    types: HashMap<String, NumberType>,
    /// `Platform.numbertype_to_rclass` (`rfficache.py`): Number lltype →
    /// the integer class `build_int` returned. `LowLevelType` is `Hash + Eq`.
    numbertype_to_rclass: HashMap<LowLevelType, KnownType>,
}

/// Upstream module-global `platform = Platform()` (`rfficache.py`).
/// `rffi::populate_inttypes` fills it from `rffi::TYPES` and
/// [`sizeof_c_types`] on first use.
pub static PLATFORM: LazyLock<Mutex<Platform>> = LazyLock::new(|| Mutex::new(Platform::new()));

impl Platform {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn inttype_from_size(
        &mut self,
        name: &str,
        signed: bool,
        size: i64,
    ) -> Result<NumberType, RffiCacheError> {
        match self.types.get(name) {
            Some(existing) => Ok(existing.clone()),
            None => self.make_type(name, signed, size),
        }
    }

    /// `_make_type` (`rfficache.py`): `build_int` then `lltype.build_number`.
    pub fn make_type(
        &mut self,
        name: &str,
        signed: bool,
        size: i64,
    ) -> Result<NumberType, RffiCacheError> {
        let inttype =
            rarithmetic::build_int(Some(&format!("r_{name}")), signed, (size * 8) as u16, false)
                .map_err(RffiCacheError::new)?;
        let tp = lltype::build_number(Some(name), inttype).map_err(RffiCacheError::new)?;
        let number_type = NumberType {
            name: name.to_string(),
            signed,
            bits: size * 8,
            lltype: tp.clone(),
        };
        // `_make_type`: `self.numbertype_to_rclass[tp] = inttype`
        self.numbertype_to_rclass.insert(tp, inttype);
        self.types.insert(name.to_string(), number_type.clone());
        Ok(number_type)
    }

    pub fn populate_inttypes_from_sizes(
        &mut self,
        declarations: &[IntTypeDecl],
        sizes: &[i64],
    ) -> Result<(), RffiCacheError> {
        let missing = declarations
            .iter()
            .filter(|decl| !self.types.contains_key(decl.name))
            .collect::<Vec<_>>();
        if missing.len() != sizes.len() {
            return Err(RffiCacheError::new(format!(
                "rfficache.py: expected {} sizeof answers, got {}",
                missing.len(),
                sizes.len()
            )));
        }
        for (decl, size) in missing.into_iter().zip(sizes.iter().copied()) {
            self.make_type(decl.name, decl.signed, size)?;
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&NumberType> {
        self.types.get(name)
    }

    /// `platform.numbertype_to_rclass[tp]` (`rfficache.py` `_make_type`).
    pub fn numbertype_to_rclass(&self, tp: &LowLevelType) -> Option<KnownType> {
        self.numbertype_to_rclass.get(tp).copied()
    }

    /// `platform.numbertype_to_rclass[tp] = rclass` (`rffi.py` after `setup`).
    pub fn set_numbertype_to_rclass(&mut self, tp: LowLevelType, rclass: KnownType) {
        self.numbertype_to_rclass.insert(tp, rclass);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntTypeDecl {
    pub name: &'static str,
    pub c_name: &'static str,
    pub signed: bool,
}

impl IntTypeDecl {
    pub const fn new(name: &'static str, c_name: &'static str, signed: bool) -> Self {
        Self {
            name,
            c_name,
            signed,
        }
    }
}

pub fn default_includes(platform_name: &str) -> Vec<&'static str> {
    let mut includes = vec!["stdlib.h", "stdio.h", "sys/types.h"];
    if platform_name != "msvc" {
        includes.extend(["inttypes.h", "stddef.h"]);
    }
    includes
}

pub fn ask_gcc_source(question: &str, add_source: &str, platform_name: &str) -> String {
    let include_string = default_includes(platform_name)
        .into_iter()
        .map(|header| format!("#include <{header}>"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "// includes\n{include_string}\n\n{add_source}\n\n// checking code\nint main(void)\n{{\n   {question}\n   return (0);\n}}\n"
    )
}

pub fn sizeof_question(typenames_c: &[&str]) -> String {
    typenames_c
        .iter()
        .map(|c_typename| {
            format!("printf(\"sizeof {c_typename}=%ld\\n\", (long)sizeof({c_typename}));")
        })
        .collect::<Vec<_>>()
        .join("\n\t")
}

pub fn parse_sizeof_c_types(
    typenames_c: &[&str],
    answer: &str,
) -> Result<Vec<i64>, RffiCacheError> {
    let lines = answer.lines().collect::<Vec<_>>();
    if lines.len() != typenames_c.len() {
        return Err(RffiCacheError::new(format!(
            "rfficache.py: expected {} sizeof lines, got {}",
            typenames_c.len(),
            lines.len()
        )));
    }
    let mut result = Vec::with_capacity(typenames_c.len());
    for (line, c_typename) in lines.iter().zip(typenames_c.iter()) {
        let (label, value) = line
            .split_once('=')
            .ok_or_else(|| RffiCacheError::new(format!("rfficache.py: malformed line {line:?}")))?;
        let expected = format!("sizeof {c_typename}");
        if label != expected {
            return Err(RffiCacheError::new(format!(
                "rfficache.py: expected label {expected:?}, got {label:?}"
            )));
        }
        result.push(value.parse::<i64>().map_err(|e| {
            RffiCacheError::new(format!("rfficache.py: invalid sizeof value {value:?}: {e}"))
        })?);
    }
    Ok(result)
}

pub fn parse_sizeof_c_type(c_typename: &str, answer: &str) -> Result<i64, RffiCacheError> {
    parse_sizeof_c_types(&[c_typename], answer).map(|mut values| values.remove(0))
}

pub fn signof_question(c_typename: &str) -> String {
    format!("printf(\"sign {c_typename}=%d\\n\", (({c_typename}) -1) <= ({c_typename})0);")
}

pub fn parse_signof_c_type(c_typename: &str, answer: &str) -> Result<bool, RffiCacheError> {
    match answer.trim() {
        s if s == format!("sign {c_typename}=0") => Ok(false),
        s if s == format!("sign {c_typename}=1") => Ok(true),
        other => Err(RffiCacheError::new(format!(
            "rfficache.py: unexpected sign answer {other:?}"
        ))),
    }
}

/// `sizeof_c_type` (`rfficache.py`): `sizeof_c_types([c_typename], **kwds)[0]`.
pub fn sizeof_c_type(c_typename: &str, ignore_errors: bool) -> Result<i64, RffiCacheError> {
    sizeof_c_types(&[c_typename], ignore_errors).map(|mut values| values.remove(0))
}

/// Which sizeof oracle `sizeof_c_types` uses for `target` versus `host`.
///
/// Mirrors `pick_platform`: host stays on `ask_gcc`; the rustc probe stands
/// in for a target whose OS is `"unknown"` (no C toolchain); any other
/// cross target is `ValueError("platform = %s")`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SizeofOracle {
    Native,
    RustcProbe,
    NoCrossPlatform,
}

fn sizeof_oracle_for(target: &str, host: &str) -> SizeofOracle {
    if !crate::layout::is_cross_target(target, host) {
        SizeofOracle::Native
    } else if crate::layout::target_os_of(target) == "unknown" {
        SizeofOracle::RustcProbe
    } else {
        SizeofOracle::NoCrossPlatform
    }
}

/// `sizeof_c_types` (`rfficache.py`): `sizeof_question` then `ask_gcc` on a
/// native target. A cross target whose OS is `"unknown"` (no C toolchain)
/// uses [`sizeof_c_types_via_rustc`]. Any other cross target errors like
/// `pick_platform`'s `ValueError("platform = %s")`.
pub fn sizeof_c_types(
    typenames_c: &[&str],
    ignore_errors: bool,
) -> Result<Vec<i64>, RffiCacheError> {
    if typenames_c.is_empty() {
        return Ok(Vec::new());
    }
    let target = translation_target_triple();
    let host = translation_host_triple();
    match sizeof_oracle_for(&target, &host) {
        SizeofOracle::Native => {
            let question = sizeof_question(typenames_c);
            let answer = ask_gcc(&question, "", ignore_errors)?;
            parse_sizeof_c_types(typenames_c, &answer)
        }
        SizeofOracle::RustcProbe => sizeof_c_types_via_rustc(typenames_c, &target),
        SizeofOracle::NoCrossPlatform => Err(RffiCacheError::new(format!(
            "rfficache: platform = {target}: no cross platform for this target"
        ))),
    }
}

/// `ask_gcc(question, add_source="", ignore_errors=False)`: write
/// `ask_gcc_source` like upstream and call
/// `gcc_cache.build_executable_cache`.
pub fn ask_gcc(
    question: &str,
    add_source: &str,
    ignore_errors: bool,
) -> Result<String, RffiCacheError> {
    let platform = crate::translator::platform::platform();
    let source = ask_gcc_source(question, add_source, &platform.name);
    let c_file = write_gcctest_c(&source)?;
    crate::tool::gcc_cache::build_executable_cache(
        std::slice::from_ref(&c_file),
        &crate::translator::tool::cbuild::ExternalCompilationInfo::default(),
        ignore_errors,
    )
    .map_err(|err| RffiCacheError::new(format!("rfficache.ask_gcc: {err}")))
}

/// Cross-target substitute for `ask_gcc` when the target OS is `"unknown"`
/// and has no C toolchain (`wasm32-unknown-unknown` today; pyre's only
/// ported cross platform, standing in for `pick_platform`). Compiles a
/// `#![no_std]` lib whose `#[no_mangle]` statics are
/// `[u8; core::mem::size_of::<core::ffi::…>()]` and reads each `N` from
/// `rustc --emit=llvm-ir` (`[N x i8]`). Other cross targets do not reach
/// this path.
pub(crate) fn sizeof_c_types_via_rustc(
    typenames_c: &[&str],
    target_triple: &str,
) -> Result<Vec<i64>, RffiCacheError> {
    if typenames_c.is_empty() {
        return Ok(Vec::new());
    }
    let source = rustc_sizeof_lib_source(typenames_c, target_triple)?;
    let rustc = rustc_binary();
    let rustc_version = rustc_version_string(&rustc);
    let dir = fresh_probe_dir("rs")?;
    let src = dir.path.join("probe.rs");
    fs::write(&src, &source)
        .map_err(|err| RffiCacheError::new(format!("rfficache: write {}: {err}", src.display())))?;
    // Cross-target adaptation: rustc identity stands in for `Platform.key`.
    let key = rustc_identity_key(&rustc, &rustc_version, target_triple);
    let cache_path = crate::tool::gcc_cache::cache_file_path_with_key(
        std::slice::from_ref(&src),
        &crate::translator::tool::cbuild::ExternalCompilationInfo::default(),
        "rustc_sizeof_cache",
        &key,
    )
    .map_err(|err| RffiCacheError::new(err.to_string()))?;
    if let Ok(answer) = fs::read_to_string(&cache_path) {
        return parse_sizeof_c_types(typenames_c, &answer);
    }
    let answer = run_rustc_sizeof_probe(&source, typenames_c, target_triple, &rustc)?;
    let _ = crate::tool::gcc_cache::try_atomic_write(&cache_path, &answer);
    parse_sizeof_c_types(typenames_c, &answer)
}

fn rustc_identity_key(rustc: &str, version: &str, target_triple: &str) -> String {
    crate::translator::platform::format_platform_key(
        "rustc",
        Path::new(rustc),
        &[std::ffi::OsString::from(version)],
        &[],
        &[],
        target_triple,
    )
}

fn write_gcctest_c(source: &str) -> Result<PathBuf, RffiCacheError> {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let name = format!(
        "gcctest-{}-{}.c",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let path = crate::tool::udir::udir().join(name);
    fs::write(&path, format!("{source}\n")).map_err(|err| {
        RffiCacheError::new(format!(
            "rfficache.ask_gcc: write {}: {err}",
            path.display()
        ))
    })?;
    Ok(path)
}

fn translation_host_triple() -> String {
    crate::translator::platform::host_triple()
}

fn translation_target_triple() -> String {
    std::env::var("TARGET").unwrap_or_default()
}

fn rustc_binary() -> String {
    std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string())
}

/// `rustc -V`, once per process. Folded into the rustc-probe cache key as
/// the compiler identity analogue of `cc::Tool::args()`.
fn rustc_version_string(rustc: &str) -> String {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION
        .get_or_init(|| {
            Command::new(rustc)
                .arg("-V")
                .output()
                .ok()
                .and_then(|output| {
                    let text = String::from_utf8_lossy(&output.stdout);
                    let trimmed = text.trim();
                    (!trimmed.is_empty()).then(|| trimmed.to_string())
                })
                .unwrap_or_default()
        })
        .clone()
}

fn rust_ffi_type_for_c_name(c_name: &str) -> Option<&'static str> {
    Some(match c_name {
        "signed char" => "core::ffi::c_schar",
        "unsigned char" => "core::ffi::c_uchar",
        "short" => "core::ffi::c_short",
        "unsigned short" => "core::ffi::c_ushort",
        "int" => "core::ffi::c_int",
        "unsigned int" => "core::ffi::c_uint",
        "long" => "core::ffi::c_long",
        "unsigned long" => "core::ffi::c_ulong",
        "long long" => "core::ffi::c_longlong",
        "unsigned long long" => "core::ffi::c_ulonglong",
        "void*" => "*const core::ffi::c_void",
        "size_t" => "usize",
        "ssize_t" => "isize",
        "intptr_t" => "isize",
        "uintptr_t" => "usize",
        "ptrdiff_t" => "isize",
        _ => return None,
    })
}

fn sizeof_static_symbol(c_name: &str) -> String {
    let mut symbol = String::from("MAJIT_SIZEOF_");
    for ch in c_name.chars() {
        match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => symbol.push(ch.to_ascii_uppercase()),
            '*' => symbol.push('P'),
            _ => symbol.push('_'),
        }
    }
    symbol
}

fn rustc_sizeof_lib_source(
    typenames_c: &[&str],
    target_triple: &str,
) -> Result<String, RffiCacheError> {
    let mut source = String::from("#![no_std]\n");
    for c_name in typenames_c {
        let rust_ty = rust_ffi_type_for_c_name(c_name).ok_or_else(|| {
            RffiCacheError::new(format!(
                "rfficache: no rustc C ABI mapping for `{c_name}` on cross target `{target_triple}`"
            ))
        })?;
        let symbol = sizeof_static_symbol(c_name);
        source.push_str(&format!(
            "#[no_mangle]\npub static {symbol}: [u8; core::mem::size_of::<{rust_ty}>()] = [0; core::mem::size_of::<{rust_ty}>()];\n"
        ));
    }
    Ok(source)
}

fn run_rustc_sizeof_probe(
    source: &str,
    typenames_c: &[&str],
    target_triple: &str,
    rustc: &str,
) -> Result<String, RffiCacheError> {
    let dir = fresh_probe_dir("rs")?;
    let src = dir.path.join("probe.rs");
    let ll = dir.path.join("probe.ll");
    fs::write(&src, source)
        .map_err(|err| RffiCacheError::new(format!("rfficache: write {}: {err}", src.display())))?;
    let output = Command::new(rustc)
        .current_dir(&dir.path)
        .arg("--crate-type=lib")
        .arg("--target")
        .arg(target_triple)
        .arg("--emit=llvm-ir")
        .arg("--edition=2021")
        .arg("-o")
        .arg(path_file_name(&ll))
        .arg(path_file_name(&src))
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .output()
        .map_err(|err| RffiCacheError::new(format!("rfficache: failed to spawn {rustc}: {err}")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(RffiCacheError::new(format!(
            "rfficache: rustc sizeof probe for `{target_triple}` failed\n{stderr}{stdout}"
        )));
    }
    let ir = fs::read_to_string(&ll)
        .map_err(|err| RffiCacheError::new(format!("rfficache: read {}: {err}", ll.display())))?;
    let mut lines = Vec::with_capacity(typenames_c.len());
    for c_name in typenames_c {
        let symbol = sizeof_static_symbol(c_name);
        let size = parse_llvm_sizeof_static(&ir, &symbol).ok_or_else(|| {
            RffiCacheError::new(format!(
                "rfficache: missing LLVM IR sizeof static `{symbol}` for `{c_name}` on `{target_triple}`"
            ))
        })?;
        lines.push(format!("sizeof {c_name}={size}"));
    }
    Ok(lines.join("\n"))
}

fn parse_llvm_sizeof_static(ir: &str, symbol: &str) -> Option<i64> {
    for line in ir.lines() {
        let trimmed = line.trim_start();
        if llvm_line_defines_symbol(trimmed, symbol) {
            return array_len_i8(trimmed);
        }
    }
    None
}

fn llvm_line_defines_symbol(line: &str, symbol: &str) -> bool {
    for prefix in [
        format!("@{symbol}"),
        format!("@\"{symbol}\""),
        format!("@_{symbol}"),
    ] {
        if let Some(rest) = line.strip_prefix(&prefix) {
            return !matches!(
                rest.chars().next(),
                Some('A'..='Z' | 'a'..='z' | '0'..='9' | '_')
            );
        }
    }
    false
}

fn path_file_name(path: &Path) -> &std::ffi::OsStr {
    path.file_name().unwrap_or_else(|| path.as_os_str())
}

fn array_len_i8(line: &str) -> Option<i64> {
    let end = line.find(" x i8]")?;
    let start = line[..end].rfind('[')?;
    line[start + 1..end].trim().parse().ok()
}

struct ProbeDir {
    path: PathBuf,
}

impl Drop for ProbeDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn fresh_probe_dir(kind: &str) -> Result<ProbeDir, RffiCacheError> {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "majit-rfficache-probe-{kind}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).map_err(|err| {
        RffiCacheError::new(format!(
            "rfficache: create probe dir {}: {err}",
            path.display()
        ))
    })?;
    Ok(ProbeDir { path })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RffiCacheError {
    pub message: String,
}

impl RffiCacheError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for RffiCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RffiCacheError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ask_gcc_source_uses_platform_includes_like_upstream() {
        let unix = ask_gcc_source("printf(\"x\");", "typedef int foo;", "posix");
        assert!(unix.contains("#include <inttypes.h>"));
        assert!(unix.contains("#include <stddef.h>"));
        assert!(unix.contains("typedef int foo;"));
        assert!(unix.contains("printf(\"x\");"));

        let msvc = ask_gcc_source("", "", "msvc");
        assert!(!msvc.contains("#include <inttypes.h>"));
        assert!(!msvc.contains("#include <stddef.h>"));
    }

    #[test]
    fn sizeof_question_and_parser_match_rfficache_output_contract() {
        let question = sizeof_question(&["int", "long"]);
        assert_eq!(
            question,
            "printf(\"sizeof int=%ld\\n\", (long)sizeof(int));\n\tprintf(\"sizeof long=%ld\\n\", (long)sizeof(long));"
        );
        assert_eq!(
            parse_sizeof_c_types(&["int", "long"], "sizeof int=4\nsizeof long=8\n").unwrap(),
            vec![4, 8]
        );
    }

    #[test]
    fn sizeof_parser_rejects_mismatched_labels() {
        let err = parse_sizeof_c_type("int", "sizeof short=2\n").unwrap_err();
        assert!(err.message.contains("expected label"));
    }

    #[test]
    fn signof_question_and_parser_match_rfficache_output_contract() {
        assert_eq!(
            signof_question("char"),
            "printf(\"sign char=%d\\n\", ((char) -1) <= (char)0);"
        );
        assert!(parse_signof_c_type("char", "sign char=1\n").unwrap());
        assert!(!parse_signof_c_type("char", "sign char=0\n").unwrap());
    }

    #[test]
    fn platform_populate_inttypes_only_fills_missing_entries() {
        let mut platform = Platform::new();
        platform.inttype_from_size("INT", true, 4).unwrap();
        platform
            .populate_inttypes_from_sizes(
                &[
                    IntTypeDecl::new("INT", "int", true),
                    IntTypeDecl::new("ULONG", "unsigned long", false),
                ],
                &[8],
            )
            .unwrap();

        assert_eq!(platform.get("INT").unwrap().bits, 32);
        assert_eq!(platform.get("ULONG").unwrap().bits, 64);
        let ulong_tp = platform.get("ULONG").unwrap().lltype.clone();
        let ulong_rclass = platform
            .numbertype_to_rclass(&ulong_tp)
            .expect("numbertype_to_rclass[ULONG]");
        match ulong_rclass {
            KnownType::Ruint => {}
            KnownType::RInt(c) => {
                assert!(!c.sign);
                assert_eq!(c.bits, 64);
                assert_eq!(c.name.to_ascii_lowercase(), "r_ulong");
            }
            other => panic!("ULONG rclass: {other:?}"),
        }
        assert!(platform.get("INT").unwrap().lltype.is_number());
    }

    #[test]
    fn sizeof_c_type_err_is_not_cached_as_an_answer() {
        let first = sizeof_c_type("not_a_real_c_type_zzzz", true);
        assert!(first.is_err(), "{first:?}");
        let second = sizeof_c_type("not_a_real_c_type_zzzz", true);
        assert!(second.is_err(), "{second:?}");
        let rustc = sizeof_c_types_via_rustc(&["__int128_t"], "wasm32-unknown-unknown");
        assert!(
            rustc.is_err(),
            "cross-target rustc probe has no __int128_t mapping: {rustc:?}"
        );
        let rustc_again = sizeof_c_types_via_rustc(&["__int128_t"], "wasm32-unknown-unknown");
        assert!(rustc_again.is_err(), "{rustc_again:?}");
    }

    #[test]
    fn sizeof_c_types_matches_host_libc_abi() {
        let sizes = sizeof_c_types(&["int", "long", "void*", "short"], false).unwrap();
        assert_eq!(sizes[0], std::mem::size_of::<libc::c_int>() as i64);
        assert_eq!(sizes[1], std::mem::size_of::<libc::c_long>() as i64);
        assert_eq!(sizes[2], std::mem::size_of::<*const u8>() as i64);
        assert_eq!(sizes[3], std::mem::size_of::<libc::c_short>() as i64);
    }

    #[test]
    fn sizeof_oracle_for_wasm32_unknown_unknown_uses_rustc_probe() {
        assert_eq!(
            sizeof_oracle_for("wasm32-unknown-unknown", "aarch64-apple-darwin"),
            SizeofOracle::RustcProbe
        );
    }

    #[test]
    fn sizeof_oracle_for_cross_linux_has_no_cross_platform() {
        assert_eq!(
            sizeof_oracle_for("aarch64-unknown-linux-gnu", "aarch64-apple-darwin"),
            SizeofOracle::NoCrossPlatform
        );
    }

    #[test]
    fn sizeof_oracle_for_host_or_empty_target_is_native() {
        let h = "aarch64-apple-darwin";
        assert_eq!(sizeof_oracle_for(h, h), SizeofOracle::Native);
        assert_eq!(sizeof_oracle_for("", h), SizeofOracle::Native);
    }

    #[test]
    fn sizeof_c_types_via_rustc_wasm32_unknown_unknown() {
        let result = sizeof_c_types_via_rustc(
            &["long", "void*", "long long", "int"],
            "wasm32-unknown-unknown",
        );
        match result {
            Ok(sizes) => assert_eq!(sizes, vec![4, 4, 8, 4]),
            Err(err) => {
                let message = err.message.to_ascii_lowercase();
                let missing_core = message.contains("can't find crate")
                    || message.contains("couldn't find crate")
                    || message.contains("may not be installed")
                    || message.contains("could not find specification for target");
                if missing_core {
                    eprintln!("skip wasm32 rustc sizeof probe: {}", err.message);
                    return;
                }
                panic!("{}", err.message);
            }
        }
    }
}
