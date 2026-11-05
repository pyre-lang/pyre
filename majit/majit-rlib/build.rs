//! `clibffi` `cConfig` (`configure_simple_type`) and `dup3` (`rposix.py`
//! `HAVE_DUP3`) probes go through `majit_eci::rffi_platform`, and
//! `majit_eci::build::compile_crate_ecis` compiles each ECI's C sources
//! (`cbuild.py` `compile_shared_lib`).
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use majit_eci::ExternalCompilationInfo;
use majit_eci::rffi_platform::{CConfig, CConfigEntry, CConfigSingleEntry, ConfigValue, configure};

/// `clibffi.py` `base_names`.
const CLIBFFI_BASE_NAMES: &[&str] = &[
    "double",
    "uchar",
    "schar",
    "sshort",
    "ushort",
    "uint",
    "sint",
    "float",
    "longdouble",
    "pointer",
    "void",
    "sint8",
    "uint8",
    "sint16",
    "uint16",
    "sint32",
    "uint32",
    "sint64",
    "uint64",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let have_dup3 = have_dup3();
    majit_eci::build::compile_crate_ecis(&ExternalCompilationInfo {
        pre_include_bits: vec![format!("#define HAVE_DUP3 {}", have_dup3 as i32)],
        ..Default::default()
    });

    if target_has_clibffi() && std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default() != "msvc" {
        write_clibffi_cconfig();
    }
}

fn target_has_clibffi() -> bool {
    std::env::var("CARGO_CFG_UNIX").is_ok() || std::env::var("CARGO_CFG_WINDOWS").is_ok()
}

/// `clibffi.py` `add_simple_type` / `configure(CConfig)`. The probe ECI is
/// `clibffi.py`'s non-Windows `eci`: `includes=['ffi.h']`, macOS
/// `pre_include_bits` `MACOSX` / `USE_FFI_CLOSURE_ALLOC`,
/// `platform.library_dirs_for_libffi()` / `include_dirs_for_libffi()`,
/// and `libffi.a` in a library dir becomes `link_files` and clears
/// `libraries`. libffi-sys's bundled build on MSVC exports no include
/// directory for this probe.
fn write_clibffi_cconfig() {
    let mut entries = Vec::new();
    for base in CLIBFFI_BASE_NAMES {
        for field in ["size", "alignment", "type"] {
            let name = format!("ffi_type_{base}_{field}");
            let expr = format!("ffi_type_{base}.{field}");
            entries.push((name, CConfigEntry::ConstantInteger(expr)));
        }
    }
    let plat = majit_eci::platform::platform();
    let include_dirs = plat
        .include_dirs_for_libffi()
        .unwrap_or_else(|err| panic!("{err}"))
        .iter()
        .map(|dir| dir.to_string_lossy().into_owned())
        .collect();
    let library_dirs: Vec<String> = plat
        .library_dirs_for_libffi()
        .unwrap_or_else(|err| panic!("{err}"))
        .iter()
        .map(|dir| dir.to_string_lossy().into_owned())
        .collect();
    let pre_include_bits = if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "macos" {
        vec!["#define MACOSX\n#define USE_FFI_CLOSURE_ALLOC 1".into()]
    } else {
        vec![]
    };
    let mut libraries = vec!["ffi".to_string()];
    let mut link_files = Vec::new();
    for libdir in &library_dirs {
        let candidate = std::path::Path::new(libdir).join("libffi.a");
        if candidate.exists() {
            link_files.push(candidate.to_string_lossy().into_owned());
            libraries.clear();
        }
    }
    let cconfig = CConfig {
        compilation_info: ExternalCompilationInfo {
            pre_include_bits,
            includes: vec!["ffi.h".into()],
            include_dirs,
            library_dirs,
            libraries,
            link_files,
            ..Default::default()
        },
        entries,
        single_entries: vec![],
    };
    let results = configure(&cconfig, false).unwrap_or_else(|err| panic!("{err}"));
    let mut text = String::from(
        "// clibffi.py cConfig: rffi_platform.configure(CConfig), written by majit-rlib/build.rs.\n",
    );
    for (name, value) in results {
        let ConfigValue::Int(n) = value else {
            panic!("{name}: {value:?}");
        };
        let ty = if name.ends_with("_size") {
            if n < 0 || n > usize::MAX as i128 {
                panic!("{name}: {n}");
            }
            "usize"
        } else if name.ends_with("_alignment") || name.ends_with("_type") {
            if n < 0 || n > u16::MAX as i128 {
                panic!("{name}: {n}");
            }
            "u16"
        } else {
            panic!("{name}");
        };
        text.push_str(&format!("pub(crate) const {name}: {ty} = {n};\n"));
    }
    let path = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"))
        .join("clibffi_cconfig.rs");
    std::fs::write(&path, text).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
}

/// `rposix.py` `HAVE_DUP3 = rffi_platform.Has('dup3')`. `Has` builds a program
/// that names `dup3`; it fails when the headers do not declare it or the C
/// library does not provide it.
fn have_dup3() -> bool {
    if std::env::var("CARGO_CFG_UNIX").is_err() {
        return false;
    }
    let cconfig = CConfig {
        compilation_info: ExternalCompilationInfo {
            includes: vec!["unistd.h".into(), "fcntl.h".into()],
            ..Default::default()
        },
        entries: vec![],
        single_entries: vec![("HAVE_DUP3".into(), CConfigSingleEntry::Has("dup3".into()))],
    };
    let results = configure(&cconfig, false).unwrap_or_else(|err| panic!("{err}"));
    match results.as_slice() {
        [(_, ConfigValue::Bool(have))] => *have,
        _ => panic!("{results:?}"),
    }
}
