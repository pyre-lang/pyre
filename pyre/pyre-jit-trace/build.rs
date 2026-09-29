//! Build script for `pyre-jit-trace`.
//!
//! Two outputs, never both: the translation prepass in `build/prepass.rs`
//! (the `prepass` feature, on by default), or compile-only placeholders for
//! the same `include!` / `include_bytes!` sites.
//!
//! The placeholders exist for LLBC extraction. Charon compiling `pyre-jit`
//! must compile its `pyre-jit-trace` dependency before `pyre-jit.ullbc`
//! exists, so nothing here can run then — and nothing needs to: the
//! artefacts only have to satisfy rustc so `pyre-jit`'s MIR is exposed. The
//! prepass itself is the reason this script build-depends on
//! `pyre-interpreter`, `majit-translate`, and under its feature `pyre-module`;
//! behind the `prepass` feature those are optional, so an extraction pass
//! (`--no-default-features`) does not compile a host copy of the interpreter
//! stack whose only purpose would be to write empty files.

#[cfg(feature = "prepass")]
#[path = "build/prepass.rs"]
mod prepass;

fn main() {
    println!("cargo::rerun-if-env-changed=MAJIT_LLBC_EXTRACTION");
    // The prepass runs inside this script, so its census switches only take
    // effect when the script itself reruns. Without these, setting one and
    // rebuilding replays a cached script and prints nothing — a reading that
    // is indistinguishable from a run with no declines. Declaring them means
    // a census is requested by setting the variable, not by first touching
    // this file.
    for census_switch in [
        "MAJIT_DECLINE_LOG",
        "MAJIT_MIR_FRONTEND_DEBUG",
        "MAJIT_RTYPER_VERBOSE",
    ] {
        println!("cargo::rerun-if-env-changed={census_switch}");
    }
    let extracting =
        std::env::var_os("MAJIT_LLBC_EXTRACTION").as_deref() == Some(std::ffi::OsStr::new("1"));
    #[cfg(not(feature = "prepass"))]
    if !extracting {
        // A consumer that turned `pyre-jit-trace`'s default features off
        // without enabling `prepass` would otherwise link a JIT with no
        // jitcodes and find out at runtime. `pyre-jit` forwards the feature;
        // a crate that depends on `pyre-jit` with `default-features = false`
        // must name `pyre-jit/prepass` itself.
        println!(
            "cargo::error=pyre-jit-trace is built without the `prepass` feature and \
             MAJIT_LLBC_EXTRACTION is not set: the generated JIT tables would be empty \
             placeholders. Enable `pyre-jit/prepass` (in `pyre-jit`'s defaults) on this \
             dependency edge."
        );
        std::process::exit(1);
    }
    if extracting {
        emit_llbc_extraction_placeholders();
        return;
    }
    #[cfg(feature = "prepass")]
    prepass::main();
}

/// Compile-only stand-ins for every prepass output.
///
/// None of these execute during extraction; they only need to satisfy
/// `include!` / `include_bytes!` so rustc can expose `pyre-jit`'s MIR. The
/// next normal Cargo build observes `MAJIT_LLBC_EXTRACTION` changing and
/// replaces every placeholder. Each empty `Vec` is serialized as `Vec<u8>`:
/// bincode writes an empty sequence as its length alone, so the element
/// type does not reach the bytes, and naming the real one would pull
/// `majit-translate` back into the build dependencies this path exists to
/// avoid.
fn emit_llbc_extraction_placeholders() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set");
    std::fs::write(
        format!("{out_dir}/jit_trace_gen.rs"),
        "pub const COMPILED_JIT_DRIVERS: &[(&str, usize)] = &[];\n\
         pub const CANONICAL_JITCODES: &[(&str, usize)] = &[];\n",
    )
    .unwrap();
    std::fs::write(format!("{out_dir}/jit_metadata.json"), b"{}\n").unwrap();
    std::fs::write(format!("{out_dir}/jitcodes.bin"), b"").unwrap();
    std::fs::write(
        format!("{out_dir}/jitcodes_index.bin"),
        bincode::serialize(&(Vec::<String>::new(), Vec::<String>::new(), vec![0_u32])).unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/indirectcalltargets.bin"),
        bincode::serialize(&Vec::<(usize, i64)>::new()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/jit_drivers.bin"),
        bincode::serialize(&Vec::<u8>::new()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/insns.bin"),
        bincode::serialize(&std::collections::BTreeMap::<String, u8>::new()).unwrap(),
    )
    .unwrap();
    std::fs::write(format!("{out_dir}/descrs.bin"), b"").unwrap();
    std::fs::write(
        format!("{out_dir}/callinfos.bin"),
        bincode::serialize(&Vec::<(u16, u32, i64)>::new()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/descrs_index.bin"),
        bincode::serialize(&(
            vec![0_u32],
            Vec::<u8>::new(),
            Vec::<u32>::new(),
            Vec::<u64>::new(),
        ))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(format!("{out_dir}/descr_layouts.bin"), b"").unwrap();
    std::fs::write(
        format!("{out_dir}/descr_layouts_index.bin"),
        bincode::serialize(&vec![0_u32]).unwrap(),
    )
    .unwrap();
    std::fs::write(format!("{out_dir}/effect_infos.bin"), b"").unwrap();
    std::fs::write(
        format!("{out_dir}/effect_infos_index.bin"),
        bincode::serialize(&vec![0_u32]).unwrap(),
    )
    .unwrap();
    std::fs::write(format!("{out_dir}/ei_descr_mints.bin"), b"").unwrap();
    std::fs::write(
        format!("{out_dir}/ei_descr_mints_index.bin"),
        bincode::serialize(&vec![0_u32]).unwrap(),
    )
    .unwrap();
    std::fs::write(format!("{out_dir}/ei_descr_stamps.bin"), b"").unwrap();
    std::fs::write(
        format!("{out_dir}/ei_descr_stamps_index.bin"),
        bincode::serialize(&vec![0_u32]).unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/ei_descr_lookups.bin"),
        bincode::serialize(&Vec::<u8>::new()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/field_mint_census.bin"),
        bincode::serialize(&majit_ir::descr::FieldMintCensus::default()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/liveness.bin"),
        bincode::serialize(&Vec::<u8>::new()).unwrap(),
    )
    .unwrap();
    for name in [
        "fnaddr_bindings.bin",
        "static_pytype_bindings.bin",
        "static_ref_bindings.bin",
    ] {
        std::fs::write(
            format!("{out_dir}/{name}"),
            bincode::serialize(&Vec::<(String, i64)>::new()).unwrap(),
        )
        .unwrap();
    }
    std::fs::write(
        format!("{out_dir}/symbolic_fnaddr_paths.bin"),
        bincode::serialize(&Vec::<(i64, String)>::new()).unwrap(),
    )
    .unwrap();
}
