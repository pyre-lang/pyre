//! `rffi_platform` probes `clibffi` reads off the target's C compiler.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // `clibffi.py configure_simple_type('ffi_type_longdouble')` reads the
    // symbol's size/alignment/type. libffi defines that symbol only where its
    // configure saw `sizeof(long double) != sizeof(double)`; elsewhere
    // `ffi.h` makes `ffi_type_longdouble` a macro for `ffi_type_double`, and
    // naming the absent symbol is a link error. Ask the target's compiler the
    // same question libffi's configure asked.
    println!("cargo:rustc-check-cfg=cfg(clibffi_longdouble)");
    if matches!(
        std::env::var("CARGO_CFG_TARGET_OS")
            .unwrap_or_default()
            .as_str(),
        "linux" | "macos" | "windows" | "android"
    ) && !matches!(
        std::env::var("CARGO_CFG_TARGET_ENV")
            .unwrap_or_default()
            .as_str(),
        "musl" | "sgx"
    ) && long_double_is_its_own_type()
    {
        println!("cargo:rustc-cfg=clibffi_longdouble");
    }
}

/// Whether the target's `long double` is wider than its `double`, which is
/// what libffi's `HAVE_LONG_DOUBLE` records. The probe only has to compile:
/// a static assertion answers without running anything, so it holds when
/// cross-compiling too.
fn long_double_is_its_own_type() -> bool {
    let out = Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR")).join("longdouble-probe");
    if std::fs::create_dir_all(&out).is_err() {
        return false;
    }
    let source = out.join("probe.c");
    let assertion = "_Static_assert(sizeof(long double) != sizeof(double), \"same width\");\n";
    if std::fs::write(&source, assertion).is_err() {
        return false;
    }
    cc::Build::new()
        .file(&source)
        .out_dir(&out)
        .warnings(false)
        .cargo_metadata(false)
        .cargo_warnings(false)
        .try_compile("clibffi_longdouble_probe")
        .is_ok()
}
