//! `rffi_platform` probes `clibffi` and `dup3` (`rposix.py` `HAVE_DUP3`) read
//! off the target's C compiler, and `majit_eci::build::compile_crate_ecis`
//! compiles each ECI's C sources (`cbuild.py` `compile_shared_lib`).
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let have_dup3 = have_dup3();
    majit_eci::build::compile_crate_ecis(&majit_eci::ExternalCompilationInfo {
        pre_include_bits: vec![format!("#define HAVE_DUP3 {}", have_dup3 as i32)],
        ..Default::default()
    });

    // `clibffi.py configure_simple_type('ffi_type_longdouble')` reads the
    // symbol's size/alignment/type. libffi defines that symbol only where its
    // configure saw `sizeof(long double) != sizeof(double)`; elsewhere
    // `ffi.h` makes `ffi_type_longdouble` a macro for `ffi_type_double`, and
    // naming the absent symbol is a link error. Ask the target's compiler the
    // same question libffi's configure asked.
    println!("cargo:rustc-check-cfg=cfg(clibffi_longdouble)");
    if target_has_clibffi() && long_double_is_its_own_type() {
        println!("cargo:rustc-cfg=clibffi_longdouble");
    }
}

fn target_has_clibffi() -> bool {
    std::env::var("CARGO_CFG_UNIX").is_ok() || std::env::var("CARGO_CFG_WINDOWS").is_ok()
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

/// `rposix.py` `HAVE_DUP3 = rffi_platform.Has('dup3')`. Compile-only: an undeclared `dup3` is a
/// compile error.
fn have_dup3() -> bool {
    if std::env::var("CARGO_CFG_UNIX").is_err() {
        return false;
    }
    let out = Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR")).join("dup3-probe");
    if std::fs::create_dir_all(&out).is_err() {
        return false;
    }
    let source = out.join("probe.c");
    let eci = majit_eci::ExternalCompilationInfo {
        includes: vec!["unistd.h".into(), "fcntl.h".into()],
        ..Default::default()
    };
    let mut body = String::new();
    eci.write_c_header(&mut body);
    body.push_str("void *rpy_probe_dup3 = (void *)dup3;\n");
    if std::fs::write(&source, &body).is_err() {
        return false;
    }
    cc::Build::new()
        .file(&source)
        .out_dir(&out)
        .warnings(false)
        .cargo_metadata(false)
        .cargo_warnings(false)
        .try_compile("dup3_probe")
        .is_ok()
}
