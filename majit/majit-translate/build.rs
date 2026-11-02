//! Export the triple this crate is compiled for. A build script's `TARGET` is
//! the host running the translator (`HOST` is only set in *other* crates'
//! build.rs when they *use* majit-translate).
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TARGET");
    let host = std::env::var("TARGET").expect("cargo TARGET");
    println!("cargo:rustc-env=MAJIT_TRANSLATE_HOST={host}");
}
