//! `majit_eci::build::compile_crate_ecis` compiles each ECI's C sources
//! (`cbuild.py` `compile_shared_lib`), including `cpyext` `api.py build_eci`.
//! The `_MSC_VER` probe stays here: `sys.version` names the C compiler the
//! build used.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    majit_eci::build::compile_crate_ecis(&majit_eci::ExternalCompilationInfo::default());
    let target = std::env::var("TARGET").unwrap_or_default();

    // `sys.version` names the C compiler the build used, and on an MSVC target
    // that name is the `MSC v.<_MSC_VER>` token
    // (`rpython/rlib/compilerinfo.py:22`).  `ctypes.util._get_build_version`
    // reads the number back out to decide which C runtime `find_library("c")`
    // may hand out; with no token at all it assumes MSVC 6 and answers
    // `msvcrt.dll`, a runtime this build does not share an `errno` with.  The
    // number comes from the preprocessor rather than a parsed banner: `/EP`
    // writes the expansion to stdout and nothing else.
    if target.ends_with("-pc-windows-msvc") {
        if let Some(msc_ver) = msc_ver() {
            println!("cargo:rustc-env=PYRE_MSC_VER={msc_ver}");
        }
    }
}

/// What the compiler makes of a preprocessor probe: `/EP` writes the
/// expansion to stdout and nothing else.  `None` when the probe cannot be
/// compiled at all.
fn expand(stem: &str, source: &str) -> Option<String> {
    let out_dir = std::env::var("OUT_DIR").ok()?;
    let probe = Path::new(&out_dir).join(format!("{stem}.c"));
    std::fs::write(&probe, source).ok()?;
    let output = cc::Build::new()
        .get_compiler()
        .to_command()
        .arg("/nologo")
        .arg("/EP")
        .arg(&probe)
        .output()
        .ok()?;
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// `_MSC_VER`, as the compiler itself expands it.  `None` when the probe
/// cannot be compiled, which leaves `sys.version` naming Rust alone.
fn msc_ver() -> Option<String> {
    let expanded = expand("pyre_msc_ver", "_MSC_VER\n")?;
    let digits: String = expanded.chars().filter(char::is_ascii_digit).collect();
    (!digits.is_empty()).then_some(digits)
}
