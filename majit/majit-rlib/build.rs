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
    if target_has_clibffi() {
        compile_longdouble_helpers();
        if long_double_is_its_own_type() {
            println!("cargo:rustc-cfg=clibffi_longdouble");
        }
    }
}

fn target_has_clibffi() -> bool {
    std::env::var("CARGO_CFG_UNIX").is_ok() || std::env::var("CARGO_CFG_WINDOWS").is_ok()
}

/// `cfield.c g_set` / `g_get` — convert through C `long double` so the
/// buffer width is `sizeof(long double)` on every target.
fn compile_longdouble_helpers() {
    let out = Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR")).join("ld-helpers");
    if std::fs::create_dir_all(&out).is_err() {
        return;
    }
    let source = out.join("ld.c");
    let body = r#"
void clibffi_f64_to_longdouble(double x, void *dst) {
    long double y = x;
    __builtin_memcpy(dst, &y, sizeof(y));
}
double clibffi_longdouble_to_f64(const void *src) {
    long double y;
    __builtin_memcpy(&y, src, sizeof(y));
    return (double)y;
}
"#;
    if std::fs::write(&source, body).is_err() {
        return;
    }
    cc::Build::new()
        .file(&source)
        .out_dir(&out)
        .warnings(false)
        .cargo_warnings(false)
        .compile("clibffi_ld");
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
