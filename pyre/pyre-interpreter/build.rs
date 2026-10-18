//! Build Python-runtime ABI shims and target-layout probes.
//!
//! Runtime-independent libraries and generated assets belong to
//! `pyre-native`; this script remains only where the interpreter ABI itself
//! determines the generated C symbols or cfg values.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target = std::env::var("TARGET").unwrap_or_default();

    // The variadic C-API entry points cannot have Rust bodies -- no Rust
    // compiler walks a `va_list` -- so they are C translation units compiled
    // into the interpreter, which is what `pypy/module/cpyext/src/` is.  The
    // predicate is the one `lib.rs` gates `mod cpyext` with, and `pyrex`'s
    // build script exports the symbols a loaded extension resolves against.
    if std::env::var_os("CARGO_FEATURE_CPYEXT").is_some()
        && std::env::var_os("CARGO_FEATURE_SANDBOX").is_none()
        && matches!(
            std::env::var("CARGO_CFG_TARGET_OS")
                .unwrap_or_default()
                .as_str(),
            "macos" | "linux"
        )
    {
        let cpyext_sources = [
            "src/cpyext/src/abstract.c",
            "src/cpyext/src/getargs.c",
            "src/cpyext/src/modsupport.c",
            "src/cpyext/src/mysnprintf.c",
            "src/cpyext/src/pyerrors.c",
            "src/cpyext/src/unicodeobject.c",
        ];
        for source in cpyext_sources {
            println!("cargo:rerun-if-changed={source}");
        }
        let include = Path::new("../../include/pyre3.14t");
        println!("cargo:rerun-if-changed={}", include.display());
        cc::Build::new()
            .files(cpyext_sources)
            .include(include)
            .compile("pyre_cpyext_c");
    }

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
