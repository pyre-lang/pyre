//! Build script for pyre-jit: runs majit-translate on the active pyre
//! interpreter to auto-generate tracing code. This is the Rust
//! equivalent of RPython's translation pipeline.
//!
//! Analyzes all source files from:
//! - pyre-object (Python object types: W_IntObject, W_FloatObject, etc.)
//! - pyre-interpreter (object space, bytecode dispatch, eval loop)
//! - pyre-module (optional builtin modules: math, _csv, …), when linked

#[path = "../src/call_spec.rs"]
mod call_spec;
#[path = "../src/codegen_cache.rs"]
mod codegen_cache;
#[path = "../src/llbc_fingerprint.rs"]
mod llbc_fingerprint;
#[path = "../src/pypyjit_driver_layout.rs"]
mod pypyjit_driver_layout;
#[path = "../../pyre-interpreter/src/module/pypyjit/policy.rs"]
mod pypyjit_policy;
#[path = "../src/ullbc_semantic.rs"]
mod ullbc_semantic;
#[path = "../src/virtualizable_spec.rs"]
mod virtualizable_spec;

use codegen_cache::{LLBC_CRATES, LLBC_SUBDIR};
use walkdir::WalkDir;

/// The translation prepass churns the whole graph universe through short-lived
/// allocations; the system allocator keeps the freed spans and the process's
/// peak RSS tracks the churn rather than the live set.
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Bump whenever the bytes of a cached output change shape. `bincode` is not
/// self-describing, so a record written by an older generation is not detected
/// as stale -- it decodes, into the wrong fields.
const CODEGEN_CACHE_VERSION: &str = "pyre-jit-trace-codegen-cache-v23";
/// Retained cache entries, per version. An entry measures ~36 MB -- 32 MB of
/// it is `jit_metadata.json` -- so eight covers the configurations one checkout
/// switches between (native/wasm × release/dev) inside 300 MB.
///
/// A cache shared between worktrees (`PYRE_JIT_TRACE_CACHE_DIR`) has to hold
/// every checkout's working set at once, so its floor is the number of
/// checkouts times the local figure. Below that floor an active worktree ends
/// up with fewer entries than it had on its own, which is slower than not
/// sharing at all -- so the multiplier is spelled out rather than folded into a
/// literal, and `PYRE_JIT_TRACE_CACHE_ENTRIES` raises it for a machine carrying
/// more. The cap only evicts, so the disk cost follows the entries a machine
/// actually produces.
const CODEGEN_CACHE_MAX_ENTRIES_LOCAL: usize = 8;
const CODEGEN_CACHE_SHARED_CHECKOUTS: usize = 12;
const CODEGEN_CACHE_MAX_ENTRIES_SHARED: usize =
    CODEGEN_CACHE_SHARED_CHECKOUTS * CODEGEN_CACHE_MAX_ENTRIES_LOCAL;
/// A cache version nothing has touched for this long belongs to no working
/// set. Only consulted for a shared cache; see `prune_codegen_cache`.
const CODEGEN_CACHE_STALE_VERSION: std::time::Duration =
    std::time::Duration::from_secs(14 * 24 * 60 * 60);
/// Names the directory the two settings above are read out of.
const CODEGEN_CACHE_DIR_ENV: &str = "PYRE_JIT_TRACE_CACHE_DIR";
const CODEGEN_CACHE_ENTRIES_ENV: &str = "PYRE_JIT_TRACE_CACHE_ENTRIES";
/// Rewritten on every cache hit; its mtime is the entry's last-use stamp.
const CODEGEN_CACHE_USED_MARKER: &str = ".last-used";
const CODEGEN_OUTPUTS: &[&str] = &[
    "jit_trace_gen.rs",
    "jit_metadata.json",
    "jitcodes.bin",
    "jitcodes_index.bin",
    "indirectcalltargets.bin",
    "jit_drivers.bin",
    "insns.bin",
    "descrs.bin",
    "descrs_index.bin",
    "descr_layouts.bin",
    "descr_layouts_index.bin",
    "effect_infos.bin",
    "effect_infos_index.bin",
    "ei_descr_mints.bin",
    "ei_descr_mints_index.bin",
    "ei_descr_stamps.bin",
    "ei_descr_stamps_index.bin",
    "field_mint_census.bin",
    "liveness.bin",
    "fnaddr_bindings.bin",
    "symbolic_fnaddr_paths.bin",
    "static_pytype_bindings.bin",
    "static_ref_bindings.bin",
];

/// Outputs carrying this build-script process's own addresses, by
/// construction. ASLR moves them on every process, so two processes always
/// disagree and that disagreement is not a defect.
///
/// The three `*_bindings` tables are nothing but addresses: `fnaddr_bindings`
/// is `(path, build_fnaddr)` and the other two are `(name, build_addr)`,
/// captured so `runtime_fnaddr_patch` can re-pair them with the runtime's
/// addresses (see the comments at their write sites).
///
/// The other five carry the values those tables exist to repair — the
/// codewriter bakes `pyre_interpreter::jit_trace_fnaddrs()` addresses into
/// `JitCode.fnaddr` and funcptr/static-data entries of `JitCode.constants_i`,
/// which `runtime_fnaddr_patch::patch_constants_i_fnaddrs` and
/// `patch_static_addr_constants` overwrite after deserialization. They reach
/// `jitcodes.bin` directly, `descrs.bin` through `BhDescr::JitCode.fnaddr`
/// (`codewriter/assembler.rs`, `fnaddr: jitcode.fnaddr`),
/// `indirectcalltargets.bin` through its `(index, jitcode.fnaddr)` pairs, and
/// `jit_metadata.json` through the same pipeline serialized as JSON.
/// `descr_layouts.bin` carries the parent `BhSizeSpec.vtable` word.
///
/// Excluded from the cross-process verdict only. Within one process the
/// addresses are the same, so `DeterminismCheck::InProcess` still judges every
/// one of them — a difference there is real, and `jitcodes.bin` / `descrs.bin`
/// / `indirectcalltargets.bin` moving between two in-process generations is
/// the defect that mode was written for.
///
/// What decides the cross-process verdict after these exclusions:
/// `jit_trace_gen.rs`, `jitcodes_index.bin`, `jit_drivers.bin`, `insns.bin`,
/// `descrs_index.bin`, `descr_layouts_index.bin`, `effect_infos_index.bin`,
/// `ei_descr_mints.bin`, `ei_descr_mints_index.bin`, `ei_descr_stamps.bin`,
/// `ei_descr_stamps_index.bin`,
/// `liveness.bin`.
/// `jitcodes_index.bin` is the load-bearing one — it holds each jitcode's name
/// and its byte boundaries in `jitcodes.bin`, and an address is a fixed-width
/// `i64` there, so a change in jitcode population, order or body length still
/// moves it while an address change alone does not.
/// `descrs_index.bin` is likewise a pure byte-offset/kind table: `descrs.bin`
/// remains host-addressed, but its index contains no address and stays in the
/// cross-process verdict to judge descriptor population, order, lengths, and
/// setup-pass classification.
///
/// Caching them is still sound: a restore serves these tables and the
/// `constants_i` baked against them from the *same* generation, so they stay
/// consistent with each other.
const HOST_ADDRESSED_OUTPUTS: &[&str] = &[
    "jit_metadata.json",
    "jitcodes.bin",
    "indirectcalltargets.bin",
    "descrs.bin",
    "descr_layouts.bin",
    "effect_infos.bin",
    "fnaddr_bindings.bin",
    "static_pytype_bindings.bin",
    "static_ref_bindings.bin",
];

/// Outputs whose contents count work against process-global state and therefore
/// cannot be compared between two generations in the same process.
///
/// `field_mint_census.bin` includes `fieldless_size_shell_mints` and
/// `ei_identical`. The process-global GcCache and ei-descr ledger persist across
/// the two generations, so the second generation legitimately does less work:
/// it mints no fieldless shells and finds more identical entries already stored.
/// The reset at the start of each generation remains necessary so the file
/// describes the generation that wrote it instead of accumulating both runs.
///
/// Excluded from the in-process verdict only. A cross-process cache comparison
/// still judges the census, unlike [`HOST_ADDRESSED_OUTPUTS`], whose exclusions
/// apply only across processes.
const IN_PROCESS_STATEFUL_OUTPUTS: &[&str] = &["field_mint_census.bin"];

/// Opt-in self-check that the generated outputs are a function of the inputs
/// [`codegen_cache_key`] hashes. Deliberately not part of that key: it changes
/// nothing about what gets generated, it only compares two generations.
const DETERMINISM_CHECK_ENV: &str = "PYRE_CODEGEN_DETERMINISM_CHECK";
/// Subdirectory of `OUT_DIR` holding the second generation's outputs. Left in
/// place after the comparison so a reported difference can be diffed by hand.
const DETERMINISM_PROBE_SUBDIR: &str = "determinism-probe";
/// Changing this value re-runs the build script and nothing else — see
/// [`emit_rerun_directives`] for why the `cache` mode cannot be exercised
/// without it.
const DETERMINISM_NONCE_ENV: &str = "PYRE_CODEGEN_NONCE";

/// Which pair of generations [`codegen_outputs_match`] compares.
///
/// Off by default because it doubles a multi-minute prepass.
///
/// * `1` / `in-process` — generate a second time into
///   [`DETERMINISM_PROBE_SUBDIR`] and compare the two sets byte for byte.
/// * `cache` — compare this run's outputs against the entry already stored
///   under the same cache key, i.e. against a *different* process's bytes.
///   Reports that it compared nothing when no entry exists yet; editing this
///   file rekeys the cache, so that is the expected result of the first run
///   after any change here.
///
/// Neither mode subsumes the other. `in-process` catches whatever is carried
/// in mutable process state — a never-reset global registry, an allocation
/// address used as an order — but it shares this process's hash seeds with
/// the run it compares against, so a seed-derived order is invisible to it.
/// `cache` sees the seed class as well, and only ever compares against a key
/// that already has an entry.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DeterminismCheck {
    Off,
    InProcess,
    AgainstCache,
}

impl DeterminismCheck {
    /// Panics on an unrecognised value rather than reading it as `Off`: a
    /// misspelled gate that silently checks nothing reports exactly like a
    /// clean run.
    fn from_env() -> Self {
        match std::env::var(DETERMINISM_CHECK_ENV)
            .unwrap_or_default()
            .as_str()
        {
            "" | "0" => Self::Off,
            "1" | "in-process" => Self::InProcess,
            "cache" => Self::AgainstCache,
            other => {
                panic!("{DETERMINISM_CHECK_ENV}={other} is not one of: 0, 1, in-process, cache")
            }
        }
    }
}

const LOWERING_GATE_ENV: [&str; 5] = [
    "PYRE_DYN_INDIRECT",
    "MAJIT_FNPTR_INDIRECT",
    "MAJIT_MIR_FRAMESTATE",
    "PYRE_OPTION_RESIDUAL_NARROW",
    "PYRE_TUPLE_PER_SHAPE_CLASSDEF",
];

const ENGINE_ENV_ALIASES: &[(&str, &str)] = &[
    ("PYRE_RTYPER_VERBOSE", "MAJIT_RTYPER_VERBOSE"),
    ("PYRE_CALLEE_CENSUS", "MAJIT_CALLEE_CENSUS"),
    ("PYRE_CALLEE_CENSUS_ROWS", "MAJIT_CALLEE_CENSUS_ROWS"),
    ("PYRE_MIR_FRONTEND_LLBC", "MAJIT_MIR_FRONTEND_LLBC"),
    ("PYRE_PROFILE_PIPELINE", "MAJIT_PROFILE_PIPELINE"),
    ("PYRE_FNPTR_INDIRECT", "MAJIT_FNPTR_INDIRECT"),
    ("PYRE_MIR_FRAMESTATE", "MAJIT_MIR_FRAMESTATE"),
];

fn forward_engine_env_aliases() {
    for &(runtime_name, engine_name) in ENGINE_ENV_ALIASES {
        if std::env::var_os(engine_name).is_none()
            && let Some(value) = std::env::var_os(runtime_name)
        {
            // The build script has not created its analysis worker yet, so no
            // other thread can concurrently read or modify this process's
            // environment. The worker and majit libraries then see the
            // engine-owned spelling.
            unsafe { std::env::set_var(engine_name, value) };
        }
    }
}

/// Entry of the real build: the translation prepass over the LLBC set.
///
/// `build.rs` dispatches here only when the `prepass` feature is on and the
/// build is not an LLBC extraction; both other cases end in its placeholders.
pub fn main() {
    forward_engine_env_aliases();

    // Fail fast with an actionable message when the Charon-extracted LLBC
    // artefacts the codegen consumes are absent.  Without this, the missing
    // set surfaces deep inside `real_main` as a worker-thread `panic!`
    // (`build-script worker panicked: Any { .. }`) printed *below* the
    // 150-line source-path dump, where it is easy to miss.
    preflight_llbc_or_fail();

    // The codegen worker (`real_main`) runs the RPythonTyper
    // specialization (`specialize_legacy_graph_with_registry` → annotator
    // `complete_pending_blocks` / rtyper `specialize_more_blocks`).  Its
    // visit order is keyed off the annotator/rtyper worklist maps
    // (`genpendingblocks`, `annotated`, `all_blocks`, …), which are
    // insertion-ordered `IndexMap`s, so the order in which the callee
    // specialization chain is walked does not depend on the per-process
    // SipHash seed.
    //
    // That covers the walk order and nothing downstream of it.  It does not
    // establish that the emitted bytes are a function of the inputs: two
    // generations of the same sources have been observed to produce different
    // `jitcodes.bin` / `descrs.bin` / `ei_descr_mints.bin` while
    // `jitcodes_index.bin` held still, from ordering and population decided
    // after the walk (heap addresses used as a canonical order, a
    // process-global mint registry).  `DeterminismCheck` is the opt-in check;
    // an earlier reading of the paragraph above concluded "a single in-process
    // run suffices", and that conclusion is why no check existed to catch it.
    //
    // The 1 GiB thread stack is needed for syn's recursive parse of ~150
    // files plus the rtyper chain (on Windows the main thread's 1 MiB default
    // would STATUS_STACK_OVERFLOW).
    run_worker();
}

/// pyre's fallible-return carrier: a bare `PyError`, materialised through
/// the published `pyerror_to_exc_object` free function and rebuilt through
/// the `PyError::from_exc_object` associated fn.
///
/// `front::result_exc` lowers a `Result<T, E>`-returning graph into
/// exception edges; naming `E` is the consumer's job, so the identity of
/// pyre's own `E` lives here rather than in majit.
///
/// These conversions transport the application-level exception object, not
/// the complete interpreter-level carrier. `PyError::from_exc_object`
/// initializes `attach_tb` and `context_recorded` rather than restoring their
/// input values, so the conversion pair is not a lossless carrier round trip.
const PYRE_ERROR_CARRIER: majit_translate::ErrorCarrierSpec<'static> =
    majit_translate::ErrorCarrierSpec {
        carrier_path: "pyre_interpreter::error::PyError",
        carrier_class: "error::PyErrorObject",
        carrier_wrappers: &[],
        to_exc_object: Some(&["pyre_interpreter", "error", "pyerror_to_exc_object"]),
        from_exc_object: Some(("PyError", "from_exc_object")),
    };

fn analysis_llbc_paths(repo_root: &str) -> Vec<String> {
    if let Some(paths) = std::env::var_os("MAJIT_MIR_FRONTEND_LLBC") {
        let paths: Vec<String> = std::env::split_paths(&paths)
            .filter(|path| !path.as_os_str().is_empty())
            .map(|path| path.to_string_lossy().into_owned())
            .collect();
        if !paths.is_empty() {
            return paths;
        }
    }

    let llbc_dir = std::path::Path::new(repo_root)
        .join("build")
        .join(LLBC_SUBDIR);
    LLBC_CRATES
        .iter()
        .map(|crate_name| llbc_dir.join(format!("{crate_name}.ullbc")))
        .chain(
            llbc_layout_sidecars()
                .into_iter()
                .map(|name| llbc_dir.join(name)),
        )
        .map(|path| path.to_string_lossy().into_owned())
        .collect()
}

/// Pre-flight the LLBC prerequisite, mirroring the resolution order in
/// `majit-translate` (`build_semantic_program_via_active_frontend`):
/// honour the `MAJIT_MIR_FRONTEND_LLBC` override, else require the
/// `LLBC_CRATES` artefacts under `build/<LLBC_SUBDIR>/`. `pyre-jit` contains
/// the exact `eval::eval_loop_jit` portal; `pyre-module`, in a build that
/// links it, holds optional builtin-module graphs so look_inside/elidable survive
/// the crate split.
///
/// When neither resolves, emit a clean, copy-pasteable bootstrap message
/// and fail the build *before* the worker spawns — so the contributor
/// sees the exact steps to run instead of a worker-thread panic buried
/// under the source-file dump.  Auto-running the bootstrap from here is
/// deliberately avoided: `scripts/extract-llbc.py` shells out to a nested
/// `cargo build`, which would block on the outer build's target-directory
/// lock (deadlock), and a build script that downloads a toolchain breaks
/// hermetic / offline / CI builds.
///
/// That ban does not cover the stamp comparison in `fail_if_llbc_stale`:
/// `scripts/extract-llbc.py --fingerprint` returns before `extract` runs and
/// only performs a `cargo metadata` walk plus `git ls-files`, so it starts no
/// nested build and takes no target-directory lock.
/// The pin and the `<platform>/<version>` layout live in
/// `scripts/install-charon.py` (`CHARON_VERSION_DEFAULT`,
/// `default_charon_dest`). This only checks that binary so a missing-LLBC
/// error can omit the install step when it is already there.
fn charon_install_present(repo_root: &std::path::Path) -> bool {
    let pin = repo_root.join("scripts").join("install-charon.py");
    let Ok(text) = std::fs::read_to_string(&pin) else {
        return false;
    };
    let Some(version) = text.lines().find_map(|line| {
        let line = line.trim();
        let rest = line.strip_prefix("CHARON_VERSION_DEFAULT = ")?;
        Some(rest.trim_matches('"').to_string())
    }) else {
        return false;
    };
    let version = std::env::var("CHARON_VERSION").unwrap_or(version);
    if let Some(dest) = std::env::var_os("CHARON_DEST") {
        return std::path::PathBuf::from(dest)
            .join(charon_exe_name())
            .exists();
    }
    let Some(platform) = platform_key(std::env::consts::OS, std::env::consts::ARCH) else {
        return false;
    };
    let shared = std::env::var_os("PYRE_SHARED_BUILD")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| repo_root.join("..").join(".pyre-build"));
    shared
        .join("charon")
        .join(platform)
        .join(version)
        .join(charon_exe_name())
        .exists()
}

fn charon_exe_name() -> &'static str {
    if std::env::consts::OS == "windows" {
        "charon.exe"
    } else {
        "charon"
    }
}

fn preflight_llbc_or_fail() {
    // Explicit override: trust it and let the translator validate the
    // individual paths (its loader panics per-file with the bad path).
    if std::env::var_os("MAJIT_MIR_FRONTEND_LLBC")
        .map(|v| std::env::split_paths(&v).any(|p| !p.as_os_str().is_empty()))
        .unwrap_or(false)
    {
        return;
    }

    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let llbc_dir = repo_root.join("build").join(LLBC_SUBDIR);
    let mut missing: Vec<String> = LLBC_CRATES
        .iter()
        .map(|crate_name| format!("{crate_name}.ullbc"))
        .filter(|name| !llbc_dir.join(name).exists())
        .collect();
    // Cross-compiling additionally needs this target's layout sidecars:
    // Charon resolves struct layouts per target, and the artefacts above
    // carry the extraction host's.  Without them every descr field past the
    // first pointer names the wrong bytes on a target whose pointers are a
    // different width, which corrupts reads instead of failing.
    for name in llbc_layout_sidecars() {
        if !llbc_dir.join(&name).exists() {
            missing.push(name);
        }
    }
    if missing.is_empty() {
        // Present is not the same as current: the artefacts are frozen
        // snapshots (AGENTS.md:48) and nothing above compares them to the
        // sources they were extracted from.
        fail_if_llbc_stale(&repo_root);
        return;
    }

    let charon_present = charon_install_present(&repo_root);

    // `cargo::error=` lines (no embedded newlines) surface in Cargo's
    // error summary on modern Cargo and fail the build on their own; the
    // framed stderr block below stays readable on every toolchain, and the
    // explicit non-zero exit is the belt-and-suspenders stop for a Cargo
    // too old to recognise `cargo::error`.
    println!(
        "cargo::error=pyre-jit codegen needs the Charon-extracted LLBC artefacts, but build/{LLBC_SUBDIR}/ is missing: {}",
        missing.join(", "),
    );
    if !charon_present {
        println!("cargo::error=Install charon (one-time): scripts/install-charon.py");
    }
    // Cross-compiling: the driver's default `LAYOUT_TARGETS` only covers
    // wasm32, so any other target's sidecars are produced only when the
    // extraction is told this target explicitly.  Advertise it (harmless but
    // redundant for wasm32).  The override must also list the sidecar paths,
    // since `MAJIT_MIR_FRONTEND_LLBC` is treated as the complete input set and
    // returns before the sidecar check above.
    let sidecars = llbc_layout_sidecars();
    let (extract_cmd, override_extra) = if sidecars.is_empty() {
        ("scripts/extract-llbc.py".to_string(), String::new())
    } else {
        let target = std::env::var("TARGET").unwrap_or_default();
        (
            format!("LLBC_LAYOUT_TARGETS={target} scripts/extract-llbc.py"),
            sidecars
                .iter()
                .map(|name| format!(":/abs/{name}"))
                .collect::<String>(),
        )
    };
    println!("cargo::error=Extract the LLBC: {extract_cmd}");

    // The install step is only needed when the charon binary is absent;
    // with it present the fix is the single extract command.
    let install_line = if charon_present {
        String::new()
    } else {
        "   scripts/install-charon.py                            # one-time\n".to_string()
    };
    eprintln!(
        "\n\
========================================================================\n\
 pyre-jit-trace: JIT codegen prerequisite missing\n\
------------------------------------------------------------------------\n\
 The Charon-extracted LLBC artefacts are required but were not found:\n\
{}\n\
 Bootstrap (run from the repo root):\n\
{}\
   {}\n\
\n\
 …or point the build at existing artefacts:\n\
   export MAJIT_MIR_FRONTEND_LLBC={}{}\n\
========================================================================\n",
        missing
            .iter()
            .map(|name| format!("   build/{LLBC_SUBDIR}/{name}"))
            .collect::<Vec<_>>()
            .join("\n"),
        install_line,
        extract_cmd,
        // Spelled from `LLBC_CRATES` because setting this override returns
        // before the mandatory-artefact check: an example short one artefact
        // is a working command that silently drops that crate's bodies.
        LLBC_CRATES
            .iter()
            .map(|crate_name| format!("/abs/{crate_name}.ullbc"))
            .collect::<Vec<_>>()
            .join(":"),
        override_extra,
    );

    std::process::exit(1);
}

// The stamp and the `--fingerprint` stdout are the same `key=value` format
// written by the same producer, so both are parsed by `llbc_fingerprint`
// rather than once per reader here.  `tests/llbc_fingerprint_format_test.rs`
// pins that module against the driver's real output.
use llbc_fingerprint::{
    FingerprintFields, FreshnessMode, freshness_policy, parse_fingerprint_fields, platform_key,
    stamp_field,
};

/// Wait for the fingerprint oracle with a deadline.
///
/// A build script that blocks forever is strictly worse than a missing
/// warning, and `std` has no `wait_timeout`, so the child is polled until the
/// deadline. A timed-out or otherwise unobservable child is killed and reaped;
/// it must not retain Cargo locks after the build proceeds. Every non-answer —
/// spawn failure, non-zero exit, unparseable stdout — collapses to `None`. The
/// caller reports freshness as unknown without breaking offline, hermetic or
/// vendored builds.
fn llbc_fingerprint_output(mut child: std::process::Child) -> Option<FingerprintFields> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Ok(None) | Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    parse_fingerprint_fields(&stdout)
}

/// Ask the extraction driver what the current inputs hash to.
///
/// `scripts/llbc_extract.py` prints the same `key=value` lines it writes into
/// the stamp, and its `source=` / `closure=` / `external=` fields are the same
/// calls `stamp_for` records, so there is one implementation of each digest
/// and it is not this one.
/// Parsed by `llbc_fingerprint::parse_fingerprint_fields`, which reads the
/// fields by name. The driver may add more, as it did for `external=`; a
/// reader that models the output as one bare value stops answering as soon as
/// that happens.
///
/// All fields are parsed. `source=` is authoritative for in-repository inputs,
/// `closure=` is a conservative residual signal, and `external=` covers inputs
/// outside the repository.
///
/// `CARGO_FEATURES` and `LLBC_LAYOUT_TARGETS` are replayed out of the stamp:
/// `fingerprint_inputs` in `scripts/llbc_extract.py` walks the
/// dependency closure under the feature set and the cross-target layout set in
/// force at extraction time, so recomputing under this build's defaults would
/// report a difference the sources do not have.
fn llbc_current_fingerprint(
    repo_root: &std::path::Path,
    driver: &std::path::Path,
    crate_name: &str,
    features: &str,
    layout_targets: &str,
) -> Option<FingerprintFields> {
    for python in ["python3", "python"] {
        let spawned = std::process::Command::new(python)
            .arg(driver)
            .arg("--fingerprint")
            .arg(crate_name)
            .current_dir(repo_root)
            .env("CARGO_FEATURES", features)
            .env("LLBC_LAYOUT_TARGETS", layout_targets)
            .env("LLBC_DEST", repo_root.join("build").join(LLBC_SUBDIR))
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn();
        let Ok(child) = spawned else {
            continue;
        };
        return llbc_fingerprint_output(child);
    }
    None
}

/// Compare every `.ullbc` against the sources it was extracted from.
///
/// The prepass reads `build/llbc/*.ullbc`, never the Rust sources
/// (AGENTS.md:48), so a field inserted anywhere but last in a `#[repr(C)]`
/// struct shifts every following descr offset while the build stays green
/// if the artefact is not re-extracted. The existence test in
/// `preflight_llbc_or_fail` cannot see this. The cache key hashes the
/// artefact bytes, not the interpreter sources, so a source-only edit
/// still hits the cache — this check is what refuses that hit when the
/// artefact is stale.
///
/// The oracle is the extractor's own stamp. The extraction engine already
/// skips a crate whose stamp still matches, so this is the comparison the
/// producer trusts, evaluated by the consumer.
///
/// `source=`, `external=` and `artefacts=` are stale gates. `closure=`
/// preserves the old whole-cargo-closure hash, but a mismatch there alone
/// warns: the artefact's declared inputs did not move, while a trait impl
/// outside its file table can still affect method resolution. The standalone
/// `--check` uses the same three-way verdict.
///
/// `artefacts=` is the only one of the four that describes an OUTPUT. The
/// input fields answer "would a fresh extraction write different bytes"; a
/// build killed part-way through a Charon write leaves a short artefact under
/// a stamp whose inputs have not moved, and all three answer "current" over
/// it. Comparing it here, rather than only in `--check`, is what keeps a plain
/// `cargo build` from consuming those bytes.
///
/// A stale artefact fails the build.  It was warning-only, on the argument
/// that the build still works for everything whose layout did not move; the
/// measured cost of that leniency is the opposite — a binary built over a
/// stale artefact reads the *old* layout while the sources say otherwise, so
/// every measurement taken from it describes code that is not in the tree, and
/// the warning scrolls past inside `target/*/build/*/output` where nobody
/// reads it.  `PYRE_LLBC_STRICT=0` demotes it back to a warning for a
/// deliberately-stale working build, and `PYRE_LLBC_SKIP_FINGERPRINT_CHECK`
/// skips the comparison entirely.
///
/// `PYRE_LLBC_SKIP_FINGERPRINT_CHECK=1` opts out entirely — announcing that it
/// did, so opting out and passing stay distinguishable.  Both take exactly
/// `1`, any other value is reported rather than ignored, and setting both is
/// refused; see `llbc_fingerprint::freshness_policy` for why that is a
/// contradiction and not a precedence question.
///
/// Three outcomes per crate, on two channels: **fresh** is silent, **stale**
/// errors (or warns with strict mode disabled), and **unknown** — unreadable stamp, or an
/// oracle that did not answer — warns without ever being promoted.  Keeping
/// unknown off the silent channel is the point: otherwise "no warning" means
/// both *checked and current* and *nobody checked*, and a reader cannot tell
/// which build they got.  Cf. the same conflation in `llbc_extract.py`'s
/// direct-invocation no-op.
fn fail_if_llbc_stale(repo_root: &std::path::Path) {
    println!("cargo::rerun-if-env-changed=PYRE_LLBC_STRICT");
    println!("cargo::rerun-if-env-changed=PYRE_LLBC_SKIP_FINGERPRINT_CHECK");
    // A non-UTF8 value is not `1`, so lossy conversion lands it in the
    // unrecognised arm and it is reported rather than read as unset.
    let strict_var =
        std::env::var_os("PYRE_LLBC_STRICT").map(|value| value.to_string_lossy().into_owned());
    let skip_var = std::env::var_os("PYRE_LLBC_SKIP_FINGERPRINT_CHECK")
        .map(|value| value.to_string_lossy().into_owned());
    let policy = freshness_policy(strict_var.as_deref(), skip_var.as_deref());
    let policy_directive = match policy.mode {
        FreshnessMode::Refuse => "cargo::error",
        _ => "cargo::warning",
    };
    for line in &policy.diagnostics {
        println!("{policy_directive}={line}");
    }
    match policy.mode {
        // Both switches set. Refuse rather than pick one: see `freshness_policy`.
        FreshnessMode::Refuse => std::process::exit(1),
        FreshnessMode::Skip => return,
        FreshnessMode::Warn | FreshnessMode::Strict => {}
    }
    let driver = repo_root.join("scripts").join("extract-llbc.py");
    if !driver.is_file() {
        // An oracle that cannot be found cannot answer, which is the unknown
        // case — not the fresh one — and returning quietly here would spell
        // "nobody checked" exactly like "checked and current" for every crate
        // at once.  The checkouts this fires in (a vendored tree, a source
        // archive, a partial clone) did not set
        // `PYRE_LLBC_SKIP_FINGERPRINT_CHECK`, so they get an acknowledgement
        // rather than silence.  One line, and the exit status is untouched:
        // failing here would break the builds the fallback exists for.
        println!(
            "cargo::warning=LLBC FRESHNESS UNKNOWN: no crate was checked \
             (scripts/extract-llbc.py is not present); the artefacts in build/llbc may or may \
             not match the current sources"
        );
        return;
    }
    let llbc_dir = repo_root.join("build").join(LLBC_SUBDIR);
    // `(crate, field, recorded, current)`. One crate can be stale on both
    // fields, and which one moved is the difference between "your own sources
    // changed" and "a dependency in another checkout changed".
    let mut stale: Vec<(&str, &'static str, String, String)> = Vec::new();
    let mut closure_warnings: Vec<&str> = Vec::new();
    // Freshness has THREE outcomes, and the third used to be spelled the same
    // way as "fresh": silence.  A crate whose stamp is unreadable, or whose
    // oracle does not answer, is *unknown* — and "nobody checked" is not a
    // licence to read its field offsets, it is the absence of the check that
    // would grant one.
    let mut unknown: Vec<(&str, &'static str)> = Vec::new();
    // The machine that ran the extraction, so this is the machine running the
    // build script -- not `CARGO_CFG_TARGET_*`, which names what is being built
    // FOR: a wasm or cross build legitimately reads this host's artefacts.
    let host_platform = platform_key(std::env::consts::OS, std::env::consts::ARCH);
    for &crate_name in LLBC_CRATES {
        // An absent artefact is the bootstrap case, already reported by the
        // prepass; only an artefact that EXISTS can be silently trusted, so
        // only that case is worth calling unknown.
        let artefact_exists = llbc_dir.join(format!("{crate_name}.ullbc")).is_file();
        let stamp_path = llbc_dir.join(format!("{crate_name}.ullbc.fingerprint"));
        let Ok(stamp) = std::fs::read_to_string(&stamp_path) else {
            if artefact_exists {
                unknown.push((crate_name, "no .fingerprint stamp beside the artefact"));
            }
            continue;
        };
        let Some(recorded_source) = stamp_field(&stamp, "source=") else {
            unknown.push((crate_name, "stamp carries no source= line"));
            continue;
        };
        let Some(recorded_closure) = stamp_field(&stamp, "closure=") else {
            unknown.push((crate_name, "stamp carries no closure= line"));
            continue;
        };
        // Required, not defaulted: an empty `external=` is a positive claim
        // that nothing outside the repository was folded in, and a stamp too
        // old to carry the field never made it.
        let Some(recorded_external) = stamp_field(&stamp, "external=") else {
            unknown.push((crate_name, "stamp carries no external= line"));
            continue;
        };
        // Required for the same reason, and the only field here that describes
        // what is IN `build/llbc` rather than what went into it.  A stamp too
        // old to carry it cannot answer whether these are still the bytes
        // extraction wrote, and "nobody checked" is not a licence to read
        // their offsets.
        let Some(recorded_artefacts) = stamp_field(&stamp, "artefacts=") else {
            unknown.push((crate_name, "stamp carries no artefacts= line"));
            continue;
        };
        let Some(recorded_platform) = stamp_field(&stamp, "platform=") else {
            unknown.push((crate_name, "stamp carries no platform= line"));
            continue;
        };
        // Declining to compare this field must not cost the comparisons below.
        if let Some(host_platform) = host_platform
            && recorded_platform != host_platform
        {
            stale.push((
                crate_name,
                "platform",
                recorded_platform,
                host_platform.to_string(),
            ));
            continue;
        }
        let features = stamp_field(&stamp, "features=").unwrap_or_default();
        let layout_targets = stamp_field(&stamp, "layout_targets=").unwrap_or_default();
        let current =
            llbc_current_fingerprint(repo_root, &driver, crate_name, &features, &layout_targets);
        let Some(current) = current else {
            unknown.push((crate_name, "the fingerprint oracle did not answer"));
            continue;
        };
        let source_matches = recorded_source == current.source;
        if !source_matches {
            stale.push((crate_name, "source", recorded_source, current.source));
        }
        if recorded_external != current.external {
            stale.push((crate_name, "external", recorded_external, current.external));
        }
        if recorded_artefacts != current.artefacts {
            stale.push((
                crate_name,
                "artefacts",
                recorded_artefacts,
                current.artefacts,
            ));
        }
        if source_matches && recorded_closure != current.closure {
            closure_warnings.push(crate_name);
        }
    }
    // Reported as a warning and never promoted by `PYRE_LLBC_STRICT`: the
    // documented contract is that an unavailable oracle must not break offline,
    // hermetic or vendored builds, and failing here would break exactly the
    // builds the fallback exists for.  Saying so costs one line and removes the
    // ambiguity.  `PYRE_LLBC_SKIP_FINGERPRINT_CHECK` acknowledges and silences.
    for (crate_name, reason) in &unknown {
        println!(
            "cargo::warning=LLBC FRESHNESS UNKNOWN: {crate_name}.ullbc was not checked \
             ({reason}); it may or may not match the current sources"
        );
    }
    for crate_name in &closure_warnings {
        println!(
            "cargo::warning=LLBC CLOSURE MOVED: {crate_name}.ullbc's declared inputs are \
             unchanged, but something else in its dependency closure moved; if a trait impl \
             or proc macro changed, re-extract"
        );
    }
    if stale.is_empty() {
        return;
    }
    // The directive string is the only difference between the two modes, so it
    // is chosen once and the same lines go through it.  `cargo::warning=` and
    // `cargo::error=` each carry a single line with no embedded newline.
    let strict = policy.mode == FreshnessMode::Strict;
    let directive = if strict {
        "cargo::error"
    } else {
        "cargo::warning"
    };
    for (crate_name, field, recorded, current) in &stale {
        // Both sides are a per-file list, so printing them the way a digest is
        // printed would put kilobytes on one cargo line and still not say
        // which file moved.  `--check` renders that with `artefact_diff`.
        if *field == "artefacts" {
            println!(
                "{directive}=LLBC STALE: {crate_name}.ullbc, or a layout sidecar beside it, is \
                 not the bytes its stamp was written over; run \
                 `python3 scripts/extract-llbc.py --check` for which file and how it differs"
            );
            continue;
        }
        // The other fields are hashes of the tree; `platform` is not.
        let subject = if *field == "platform" {
            "this build is"
        } else {
            "the tree now has"
        };
        println!(
            "{directive}=LLBC STALE: {crate_name}.ullbc was extracted at {field}={recorded}, \
             {subject} {field}={current}"
        );
    }
    // Deduplicated: a crate stale on both fields contributed two lines above,
    // and the re-extraction takes it once.
    let mut stale_crates: Vec<&str> = Vec::new();
    for &(crate_name, ..) in &stale {
        if !stale_crates.contains(&crate_name) {
            stale_crates.push(crate_name);
        }
    }
    let crates = stale_crates.join(" ");
    println!(
        "{directive}=Field offsets read out of these artefacts may name the wrong bytes; \
         re-extract with: python3 scripts/extract-llbc.py {crates}"
    );
    if strict {
        std::process::exit(1);
    }
}

/// Run the codegen worker on a large-stack thread, propagating any panic so
/// the build fails loudly instead of emitting partial output.
fn run_worker() {
    std::thread::Builder::new()
        .stack_size(1024 * 1024 * 1024)
        .spawn(real_main)
        .expect("spawn build-script worker")
        .join()
        .expect("build-script worker panicked");
}

fn opcode_published_descr_members(
    descrs: &[majit_translate::jitcode::BhDescr],
) -> std::collections::BTreeSet<majit_ir::effectinfo::DescrSetMember> {
    use majit_ir::effectinfo::DescrSetMember;
    use majit_translate::jitcode::{BhDescr, BhSizeSpec};

    fn field_key(owner: &str, name: &str) -> String {
        if owner.is_empty() {
            return name.to_string();
        }
        name.strip_prefix(&format!("{owner}."))
            .unwrap_or(name)
            .to_string()
    }

    fn add_layout(members: &mut std::collections::BTreeSet<DescrSetMember>, layout: &BhSizeSpec) {
        if layout.type_id == 0 {
            return;
        }
        for field in &layout.all_fielddescrs {
            members.insert(DescrSetMember::Field {
                struct_id: layout.type_id,
                field_name: field.field_key().to_string(),
            });
        }
    }

    fn add_descr(members: &mut std::collections::BTreeSet<DescrSetMember>, descr: &BhDescr) {
        match descr {
            BhDescr::Field {
                parent: Some(parent),
                owner,
                name,
                ..
            } => {
                add_layout(members, parent);
                if parent.type_id != 0 {
                    members.insert(DescrSetMember::Field {
                        struct_id: parent.type_id,
                        field_name: field_key(owner, name),
                    });
                }
            }
            BhDescr::Array {
                type_id,
                interior_fields,
                ..
            } if *type_id != 0 => {
                members.insert(DescrSetMember::Array { array_id: *type_id });
                for field in interior_fields {
                    members.insert(DescrSetMember::InteriorField {
                        array_id: *type_id,
                        name: field.field.field_key().to_string(),
                    });
                }
            }
            BhDescr::InteriorField { array, field } => {
                add_descr(members, array);
                add_descr(members, field);
                if let (BhDescr::Array { type_id, .. }, BhDescr::Field { owner, name, .. }) =
                    (array.as_ref(), field.as_ref())
                    && *type_id != 0
                {
                    members.insert(DescrSetMember::InteriorField {
                        array_id: *type_id,
                        name: field_key(owner, name),
                    });
                }
            }
            BhDescr::Size {
                size,
                type_id,
                vtable,
                owner,
                all_fielddescrs,
                is_gc_managed,
            } => add_layout(
                members,
                &BhSizeSpec {
                    size: *size,
                    type_id: *type_id,
                    vtable: *vtable,
                    owner: owner.clone(),
                    is_gc_managed: *is_gc_managed,
                    headerless: owner == majit_translate::jitcode::HEADERLESS_SIZE_OWNER_MARKER,
                    all_fielddescrs: all_fielddescrs.clone(),
                },
            ),
            _ => {}
        }
    }

    let mut members = std::collections::BTreeSet::new();
    for descr in descrs {
        add_descr(&mut members, descr);
    }
    members
}

fn real_main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let pyre_base = format!("{manifest_dir}/..");
    let repo_root = format!("{manifest_dir}/../..");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Collect ALL source file paths from the active interpreter crates.
    // Only the paths are consumed — module-path derivation below plus
    // `cargo::rerun-if-changed` tracking; the graph bodies come from the
    // Charon-extracted LLBC set, so the file contents are never read.
    let source_dirs = [
        format!("{pyre_base}/pyre-object/src"),
        format!("{pyre_base}/pyre-interpreter/src"),
        // Optional builtins live in this crate; `should_lower_module`
        // admits `module::*` only when a matching source path is in
        // this census. Interpreter `module/mod.rs` currently contributes
        // the `module` root as well, but the moved graphs belong here.
        // The core translation leaves the crate out with its LLBC.
        #[cfg(feature = "pyre-module")]
        format!("{pyre_base}/pyre-module/src"),
    ];

    let mut source_paths = Vec::new();

    for dir in &source_dirs {
        source_paths.extend(majit_translate::module_path::collect_rs_files(dir));
    }

    // Include the canonical portal source so its module-qualified identity is
    // present in the analyzer manifest. Production configures the exact
    // `eval::eval_loop_jit` CallPath; there is no handler-level fallback.
    // Using the real portal preserves the dispatch loop that blackhole resume
    // needs to continue from a user bytecode PC.
    //
    // Single-file inclusion (not the whole `pyre-jit/src` tree)
    // because `eval_loop_jit` is the only function in pyre-jit that
    // belongs in the portal closure; the rest of pyre-jit
    // (codewriter, assembler, regalloc, etc.) is JIT infrastructure
    // that must NOT be analyzed as user code, and the orchestration
    // would inflate analysis time and risk pulling unrelated
    // helpers into `find_all_graphs(portal)` BFS.
    let eval_path = format!("{pyre_base}/pyre-jit/src/eval.rs");
    collect_single_file(&eval_path, &mut source_paths);

    // `materialize_virtual_from_rd` (`pyre-jit/src/eval.rs`) destructures
    // `majit_ir::RdVirtualInfo` enum variants whose named fields carry
    // primitive concretetypes (`kind: u8`, `descr_index: u32`, ...).
    // Without the enum declaration in the analyser source set the variant
    // arms surface as untyped, and the rtyper's exitswitch emission for
    // `match descr.kind { 0 => ..., 1 => ..., 2 => ..., }` falls back to
    // 'r' (Ref), which `flatten.rs`'s `insert_exits` rejects as
    // `switch exitswitch must be int`.  Single-file inclusion mirrors the
    // `pyre-jit/src/eval.rs` carve-out above: only the resoperation enum
    // declarations belong in the analyser closure, not the rest of
    // majit-ir which would inflate analysis time and pull JIT
    // infrastructure into the user-code BFS.
    let resoperation_path = format!("{repo_root}/majit/majit-ir/src/resoperation.rs");
    collect_single_file(&resoperation_path, &mut source_paths);

    eprintln!(
        "[pyre-jit-trace build.rs] collected {} source paths from {} dirs (+ pyre-jit/src/eval.rs): {:?}",
        source_paths.len(),
        source_dirs.len(),
        source_paths,
    );

    // Run analysis on ALL files with PyFrame virtualizable metadata.
    //
    // This keeps the generic analyzer closer to the proc-macro/runtime path:
    // graph rewrite can recognize `next_instr`, `valuestackdepth`, and
    // `locals_cells_stack_w[*]` as virtualizable accesses before legacy
    // TracePattern classification runs.
    //
    // `owner_root` below is also the `_virtualizable_` declaration upstream
    // reads off the class (`classdesc.get_param('_virtualizable_')`, consulted
    // by `rlib/jit.py`'s `hint` entry before it mints `access_directly`). The
    // pipeline derives the front end's root set from these descriptors, so
    // this table is the single place the frame is declared virtualizable.
    let analyze_config = majit_translate::AnalyzeConfig {
        pipeline: majit_translate::PipelineConfig {
            transform: majit_translate::GraphTransformConfig {
                vable_fields: virtualizable_spec::PYFRAME_VABLE_FIELDS
                    .iter()
                    .map(|(name, idx)| {
                        majit_translate::VirtualizableFieldDescriptor::new(
                            *name,
                            Some(virtualizable_spec::PYFRAME_VABLE_OWNER_ROOT.to_string()),
                            *idx,
                        )
                    })
                    .collect(),
                vable_arrays: virtualizable_spec::PYFRAME_VABLE_ARRAYS
                    .iter()
                    .map(|(name, idx)| {
                        // virtualizable.py — VirtualizableInfo.array_descrs[i] =
                        // cpu.arraydescrof(getattr(VTYPE, name).TO). Python frame
                        // locals are PyObjectRef pointers: itemsize is one
                        // target word (the build host's word would mis-stride
                        // `FixedObjectArray` on a narrower target), is_signed=false.
                        majit_translate::VirtualizableFieldDescriptor::new_with_arraydescr(
                            *name,
                            Some(virtualizable_spec::PYFRAME_VABLE_OWNER_ROOT.to_string()),
                            *idx,
                            majit_translate::layout::target_word_size(),
                            false, // is_signed: pointers are unsigned
                        )
                    })
                    .collect(),
                call_effects: build_call_effect_overrides(),
                // `rclass.CLASSTYPE` is embedded in the translated type-info
                // group, not allocated as an independent GC object.  Pyre's
                // `PyType` is likewise static raw storage: it has no
                // `GcHeader`, so fields such as `instantiate` must not acquire
                // a GC-managed parent descr merely because the pointer travels
                // through the Ref bank.
                struct_storage: vec![majit_translate::StructStorageDescriptor::raw(
                    "pyobject::PyType",
                )],
                ..Default::default()
            },
            // The bodies behind the opcode residuals the walker descends
            // (`specialize.rs try_walker_orthodox_descent`).  The portal
            // reaches none of them in source: `call_spec.rs` classifies the
            // eval loop's `binary_value` call as a residual, and the residual
            // is lowered to `jit_binary_value_from_tag`, whose body this is.
            // `is_op` (IS_OP), `format_simple_w` (FORMAT_SIMPLE,
            // `space.format`) and `convert_value` (CONVERT_VALUE) are the
            // codewriter's `inline_call` targets for those opcodes
            // (`flatten.rs inline_call_targets`).
            helper_graphs: vec![
                majit_translate::CallPath::from_segments([
                    "pyre_interpreter",
                    "runtime_ops",
                    "is_op",
                ]),
                majit_translate::CallPath::from_segments([
                    "pyre_interpreter",
                    "type_methods",
                    "format_simple_w",
                ]),
                majit_translate::CallPath::from_segments([
                    "pyre_interpreter",
                    "runtime_ops",
                    "convert_value",
                ]),
                majit_translate::CallPath::from_segments([
                    "pyre_interpreter",
                    "opcode_ops",
                    "binary_value_from_tag",
                ]),
                majit_translate::CallPath::from_segments([
                    "pyre_interpreter",
                    "opcode_ops",
                    "compare_value_from_tag",
                ]),
                // BINARY_SLICE residual `bh_binary_slice_fn`. The portal
                // never calls this split, so the descent looks the body up
                // by this graph key (`specialize.rs
                // try_walker_orthodox_binary_slice`). The wrapper holds
                // `push_roots`; this is the inner graph.
                majit_translate::CallPath::from_segments([
                    "pyre_interpreter",
                    "runtime_ops",
                    "binary_slice_values_inner",
                ]),
                // `w_list_getitem` is `dont_look_inside`. The getitem
                // descent looks this lock-free body up by name
                // (`jitcode_runtime.rs list_getitem_jitcode`). The portal
                // closure does not reach the split, so seed it like
                // `binary_slice_values_inner`.
                majit_translate::CallPath::from_segments([
                    "pyre_object",
                    "listobject",
                    "w_list_getitem_inner",
                ]),
                // `W_FastListIterObject.descr_next` is residual `jit_next`.
                // The portal never calls this split, so the descent looks the
                // body up by this one graph key.
                majit_translate::CallPath::from_segments([
                    "pyre_interpreter",
                    "baseobjspace",
                    "list_iter_descr_next",
                ]),
                // `W_Zip.next_w` arity two over tuple iterators. `jit_next`
                // never calls this split, so the descent looks the body up
                // by this graph key.
                majit_translate::CallPath::from_segments([
                    "pyre_interpreter",
                    "baseobjspace",
                    "zip_two_tuple_next",
                ]),
                // `format_int_or_long`'s decimal arm. The portal's
                // `format_w` residual never calls it, so the descent looks
                // the body up by this graph key.
                majit_translate::CallPath::from_segments([
                    "pyre_object",
                    "intobject",
                    "format_int_decimal",
                ]),
                // Step-1 range `FOR_ITER` descends this body. `space.next`
                // calls it, and that call stays residual, so the graph is
                // seeded here for the path lookup.
                majit_translate::CallPath::from_segments([
                    "pyre_object",
                    "functional",
                    "w_range_iter_one_arg_next",
                ]),
                majit_translate::CallPath::from_segments([
                    "pyre_object",
                    "functional",
                    "w_range_iter_step_one_next",
                ]),
                // BINARY_SUBSCR residualises `getitem`, so the portal may
                // not reach `getitem_list`'s look-inside call of
                // `w_list_getitem_inner`. The walker descends that body by
                // name (`list_getitem_jitcode`).
                majit_translate::CallPath::from_segments([
                    "pyre_object",
                    "listobject",
                    "w_list_getitem_inner",
                ]),
                // BUILD_MAP's `inline_call` target (`flatten.rs
                // inline_call_targets`). `build_map_from_refs` stores through
                // `w_dict_store_checked`, so the portal closure never names
                // this wrapper.
                majit_translate::CallPath::from_segments([
                    "pyre_interpreter",
                    "baseobjspace",
                    "dict_display_setitem",
                ]),
            ],
            // `support.py` `builtin_func_for_spec` / `inline_calls_to`
            // look the helper up under the single-segment impl name.
            // The graphs are registered under the same crate-qualified
            // spelling as `helper_graphs` above.
            builtin_graphs: vec![
                (
                    "_ll_2_int_floordiv".to_string(),
                    majit_translate::CallPath::from_segments([
                        "pyre_interpreter",
                        "objspace",
                        "descroperation",
                        "_ll_2_int_floordiv",
                    ]),
                ),
                (
                    "_ll_2_int_mod".to_string(),
                    majit_translate::CallPath::from_segments([
                        "pyre_interpreter",
                        "objspace",
                        "descroperation",
                        "_ll_2_int_mod",
                    ]),
                ),
            ],
            jit_drivers: vec![
                majit_translate::JitDriverSpec {
                    portal: majit_translate::CallPath::from_segments(["eval", "eval_loop_jit"]),
                    portal_runner: Some(majit_translate::CallPath::from_segments([
                        "call_jit",
                        "ll_portal_runner_shim",
                    ])),
                    greens: pypyjit_driver_layout::PYPYJIT_GREEN_VARS
                        .iter()
                        .map(|(name, _)| (*name).to_string())
                        .collect(),
                    reds: pypyjit_driver_layout::PYPYJIT_RED_VARS
                        .iter()
                        .map(|(name, _)| (*name).to_string())
                        .collect(),
                    green_kinds: pypyjit_driver_layout::PYPYJIT_GREEN_VARS
                        .iter()
                        .map(|(_, kind)| *kind)
                        .collect(),
                    red_kinds: pypyjit_driver_layout::PYPYJIT_RED_VARS
                        .iter()
                        .map(|(_, kind)| *kind)
                        .collect(),
                    autoreds: false,
                    virtualizables: vec![pypyjit_driver_layout::PYPYJIT_VIRTUALIZABLE.to_string()],
                    red_types: pypyjit_driver_layout::PYPYJIT_RED_TYPES
                        .iter()
                        .map(|name| (*name).to_string())
                        .collect(),
                    // warmspot.py find_portals → split_graph_and_record_jitdriver
                    split_portal: true,
                },
                majit_translate::JitDriverSpec {
                    // `warmspot.py split_graph_and_record_jitdriver` copy of
                    // `_unpackiterable_unknown_length`, cut at `jit_merge_point`.
                    // greens=['greenkey'], reds='auto' (baseobjspace.py
                    // `unpackiterable_driver`).
                    portal: majit_translate::CallPath::from_segments([
                        "baseobjspace",
                        "unpackiterable_portal",
                    ]),
                    portal_runner: Some(majit_translate::CallPath::from_segments([
                        "eval",
                        "ll_unpackiterable_portal_runner_shim",
                    ])),
                    greens: vec!["greenkey".to_string()],
                    reds: vec![],
                    green_kinds: vec![majit_ir::Type::Ref],
                    red_kinds: vec![],
                    autoreds: true,
                    virtualizables: vec![],
                    red_types: vec![],
                    // Already the split body; `rewrite_jit_merge_point` lives
                    // in `_unpackiterable_unknown_length` at source level.
                    split_portal: false,
                },
                majit_translate::JitDriverSpec {
                    // `generator.py` `generatorentry_driver`. Greens `pycode`,
                    // reds `gen`/`w_arg`, no virtualizable. The portal is the
                    // source-level cut at `split_before_jit_merge_point`:
                    // `generatorentry_portal` starts at the marker.
                    portal: majit_translate::CallPath::from_segments([
                        "baseobjspace",
                        "generatorentry_portal",
                    ]),
                    portal_runner: Some(majit_translate::CallPath::from_segments([
                        "eval",
                        "ll_generatorentry_portal_runner_shim",
                    ])),
                    greens: vec!["pycode".to_string()],
                    reds: vec!["gen".to_string(), "w_arg".to_string()],
                    green_kinds: vec![majit_ir::Type::Ref],
                    red_kinds: vec![majit_ir::Type::Ref, majit_ir::Type::Ref],
                    autoreds: false,
                    virtualizables: vec![],
                    red_types: vec![],
                    // `generator_send_ex` calls `generatorentry_portal`.
                    // `split_graph_and_record_jitdriver` copies the loop;
                    // `rewrite_jit_merge_point` leaves the original as the
                    // stub whose only call is this driver's `portal_runner`.
                    split_portal: true,
                },
            ],
            // pyre production registers no trait-dispatch families (#346).
            register_trait_families: Vec::new(),
        },
    };
    // warmspot.py `vinfos[VTYPEPTR] = VirtualizableInfo(self, VTYPEPTR)` —
    // pyre's runtime `VirtualizableInfo` constructor lives in the
    // `majit-metainterp` crate (`__build_virtualizable_info`) and runs
    // at `JitDriver::new` (jitdriver.rs) where the field offsets
    // resolved by `mem::offset_of!` are available.  build.rs cannot
    // import that crate (no metainterp build-dep, and the offsets are
    // a runtime fact), so the codewriter-side factory returns the
    // offset-free `CodewriterVirtualizableInfo` that
    // `codewriter_vinfo_from_config` builds from the `vable_fields` /
    // `vable_arrays` table above (field membership and indices only)
    // until the runtime metainterp setter overrides it.  TODO: documented at
    // `CallControl::make_virtualizable_infos`.
    let vinfo_factory: &majit_translate::VirtualizableInfoFactory<'_> = &|_jd_idx, vtype| {
        majit_translate::call::codewriter_vinfo_from_config(
            vtype,
            &analyze_config.pipeline.transform,
        )
    };
    // Link the optional-module crate into this process so its
    // `#[pyre_methods]` wrappers join `BUILTIN_WRAPPER_DESCRIPTORS`.
    // `jit_trace_fnaddrs` reads that slice; without the crate the
    // moved builtins publish no address and
    // `compute_builtin_wrapper_indirect_graphs` records
    // `no jitcode for address`. `register` is idempotent with the
    // crate's ctor.
    #[cfg(feature = "pyre-module")]
    pyre_module::register();
    let mut fnaddr_bindings = pyre_interpreter::jit_trace_fnaddrs();
    // This build script is the host, so it reads the linkme slice. The
    // wasm guest fills the same set from constructors. A missing row
    // here means the build-dep was dropped and every moved builtin
    // residualises at `no jitcode for address`.
    // `dont_look_inside` residuals the portal walk still executes.
    // Unbound, each one is a symbolic path hash and
    // `refuse_reachable_symbolic_residuals` aborts the trace.
    fnaddr_bindings.extend(pyre_interpreter::baseobjspace::generatorentry_fnaddrs());
    #[cfg(feature = "pyre-module")]
    if !fnaddr_bindings
        .iter()
        .any(|(path, _)| path.contains("pyre_module::module::") && path.contains("__majit_wrap_"))
    {
        panic!(
            "prepass fnaddr table has no pyre-module #[pyre_methods] wrappers; \
             pyre-module must be linked into this build script"
        );
    }
    // Prebuilt object-space singleton addresses (static `PyType` pointers
    // and dict-strategy refs).  `majit-translate` is the translation
    // layer and must not import `pyre-object`; the driver supplies these
    // across the translation boundary.  Resolved here in the same
    // build-script process the translator runs in, so the captured
    // addresses match a direct `&pyre_object::X` read at the codewriter
    // call site.
    let static_pytype_addrs = pyre_interpreter::jit_static_pytype_addrs();
    // This script is compiled for the host even when the crate it feeds is
    // built for another target, so a name bound here is only useful if the
    // runtime pool carries it too — `runtime_fnaddr_patch::
    // patch_static_addr_constants` rewrites only names it finds in both, and
    // an unmatched name keeps this process's address.  The `#[pyre_class]`
    // registry populates on every target for that reason; where the two pools
    // can still disagree is the module set, which this process cannot read for
    // another target, and `disarm_unpaired_build_addrs` writes zero over an
    // address of that kind so the call sites reading it decline.
    let registry_rows = pyre_interpreter::pyre_class_pytype_addrs();
    let registry_keys: std::collections::HashSet<&str> =
        registry_rows.iter().map(|&(key, _)| key).collect();
    // Counts the rows that survived, not the registry's size: the table
    // drops every registry row whose static a hand-written row already
    // names.
    let kept_from_registry = static_pytype_addrs
        .iter()
        .filter(|(key, _)| registry_keys.contains(key))
        .count();
    eprintln!(
        "[PREPASS statics] pytype rows = {} ({kept_from_registry} of {} \
         #[pyre_class] registry rows kept)",
        static_pytype_addrs.len(),
        registry_rows.len(),
    );
    // The same singletons keyed by their owning type's path, for the
    // `PyreClassPyTypeOf::PYTYPE` reads whose Charon path names every
    // trait impl in their module at once. See
    // `HostStaticAddrs::pytypes_by_struct`.
    let static_pytype_by_struct = pyre_interpreter::pyre_class_pytype_by_struct_addrs();
    let static_ref_addrs = pyre_interpreter::jit_static_ref_addrs();
    let static_int_values = pyre_interpreter::jit_static_int_values();
    let static_addrs = majit_translate::HostStaticAddrs {
        pytypes: &static_pytype_addrs,
        pytypes_by_struct: &static_pytype_by_struct,
        refs: &static_ref_addrs,
        int_values: &static_int_values,
        error_carrier: PYRE_ERROR_CARRIER,
        // This interpreter mutates a scalar field through an ordinary
        // assignment the lowering already reaches, so it declares no store
        // boundary.
        scalar_field_stores: &[],
    };
    // Per-source crate-stripped module paths — the analyzer-side
    // metadata (`front::mir`) records
    // `struct_origins[bare_name] = module_path`.  Aligns with
    // the runtime's `build_object_descr_group_with_def_path` qualified
    // def-path slot in `gc_cache._cache_size` so a future
    // `path_hash(canonical_struct_name)` analyzer hash lands on the
    // same Arc the runtime publishes (PyPy `cache[STRUCT]` lltype-
    // object identity, descr.py `get_size_descr`).
    let module_paths: Vec<String> = source_paths
        .iter()
        .map(|p| majit_translate::module_path::module_path_from_source_file(p))
        .collect();
    let module_path_refs: Vec<&str> = module_paths.iter().map(|s| s.as_str()).collect();

    emit_rerun_directives(&repo_root, &source_paths);

    let cache_key = codegen_cache_key(manifest_dir, &repo_root, &source_paths);
    let cache_dir = codegen_cache_dir(&repo_root, &cache_key);
    // The verbose prepass is a census over every attempted graph. Restoring
    // generated outputs would skip the analysis entirely and leave no
    // `PREPASS phaseA/phaseB fail` lines, contradicting the documented
    // `MAJIT_RTYPER_VERBOSE=1` workflow. The generated artifacts themselves do
    // not depend on verbosity, so an ordinary build may still reuse the cache.
    let verbose_prepass =
        std::env::var_os("MAJIT_RTYPER_VERBOSE").is_some_and(|value| value == "1");
    // The callee census is emitted from the analysis itself. A cache restore
    // would print no rows, which is indistinguishable from an empty census.
    let callee_census = std::env::var_os("MAJIT_CALLEE_CENSUS").is_some_and(|value| value == "1");
    // Restoring the outputs leaves the self-check nothing to compare, so it
    // bypasses the cache for the same reason the verbose prepass does.
    let determinism_check = DeterminismCheck::from_env();
    // These diagnostics are emitted while the analyzer is running. A cache
    // restore would make an enabled census look empty.
    let field_mint_trace = majit_ir::descr::field_mint_trace_enabled();
    let struct_layout_census =
        std::env::var_os("MAJIT_STRUCT_LAYOUT_CENSUS").is_some_and(|value| value == "1");
    // The decline census is likewise written by the analysis; a restore
    // leaves `check-rtyper-skip-subjects.py` a stderr with no census line.
    let decline_census = std::env::var_os("MAJIT_DECLINE_LOG")
        .is_some_and(|value| value != "0" && !value.is_empty());
    if !verbose_prepass
        && !decline_census
        && !callee_census
        && !field_mint_trace
        && !struct_layout_census
        && determinism_check == DeterminismCheck::Off
        && restore_codegen_cache(&cache_dir, &out_dir)
    {
        eprintln!(
            "[pyre-jit-trace build.rs] restored generated JIT trace artifacts from cache {}",
            cache_key
        );
        touch_codegen_cache_entry(&cache_dir);
        prune_codegen_cache(&repo_root, &cache_dir);
        return;
    }

    // One whole generation, writing every [`CODEGEN_OUTPUTS`] entry into the
    // directory it is handed. That directory is `OUT_DIR` for the build's own
    // outputs and the probe subdirectory for the second generation
    // `DeterminismCheck::InProcess` compares against; the parameter shadows
    // the outer `out_dir` so both calls read as "write into out_dir".
    let llbc_paths = analysis_llbc_paths(&repo_root);
    let llbc_path_refs: Vec<&str> = llbc_paths.iter().map(String::as_str).collect();
    let generate_into = |out_dir: &str| {
        majit_ir::descr::reset_field_mint_census();
        // `targetpypystandalone.py jitpolicy`: `return PyPyJitPolicy(pypy_hooks)`.
        let mut jitpolicy = pypyjit_policy::PyPyJitPolicy::new();
        let pipeline = majit_translate::analyze_multiple_pipeline_from_llbc_with_modules(
            &llbc_path_refs,
            &module_path_refs,
            &analyze_config,
            None,
            vinfo_factory,
            &fnaddr_bindings,
            static_addrs,
            Some(&mut jitpolicy),
        );

        // Generate tracing code from the canonical graph-first analysis result.
        let code = majit_translate::generate_trace_code_from_pipeline(&pipeline);

        std::fs::write(format!("{out_dir}/jit_trace_gen.rs"), &code).unwrap();

        // RPython runs effectinfo.compute_bitstrings during translation and
        // puts both outputs directly into the translated image: compact
        // bitstrings on each EffectInfo and `ei_index` on each descriptor.
        // Pyre's analyzer is a build-script process, so freeze that same
        // object graph into the serialized image before crossing the process
        // boundary. Keep the live pipeline untouched: trace-code generation
        // above consumes its Arc identities, while these clones are solely the
        // translated constants written below.
        let mut frozen_descrs = pipeline.descrs.clone();
        let mut frozen_jitcodes: Vec<std::sync::Arc<majit_translate::jitcode::JitCode>> = pipeline
            .jitcodes
            .iter()
            .map(|jitcode| std::sync::Arc::new((**jitcode).clone()))
            .collect();
        let mut frozen_mints = pipeline.ei_descr_mints.clone();
        let mut call_descrs: Vec<&mut majit_translate::jitcode::BhCallDescr> = Vec::new();
        for descr in &mut frozen_descrs {
            match descr {
                majit_translate::jitcode::BhDescr::Call { calldescr }
                | majit_translate::jitcode::BhDescr::JitCode { calldescr, .. } => {
                    call_descrs.push(calldescr)
                }
                _ => {}
            }
        }
        for jitcode in &mut frozen_jitcodes {
            call_descrs.push(
                &mut std::sync::Arc::get_mut(jitcode)
                    .expect("fresh frozen JitCode clone is unexpectedly shared")
                    .body_mut()
                    .calldescr,
            );
        }
        // `effectinfo.compute_bitstrings(self.cpu.setup_descrs())` sees every
        // CallDescr in GcCache, not only the subset an opcode or JitCode body
        // happens to serialize. Include that full call group so descriptors
        // minted by a non-opcode call still receive their upstream partition.
        // The serialized BhCallDescr copies below are appended only so their
        // translated object ids and bitstrings are written in place; ordinary
        // duplicates collapse through EffectInfo._cache identity.
        let mut cached_effect_infos: Vec<majit_ir::EffectInfo> = majit_ir::descr::gc_cache()
            .lock()
            .snapshot_calls()
            .into_iter()
            .filter_map(|descr| {
                descr.as_call_descr().map(|call| {
                    let mut effect_info = call.get_extra_info().clone();
                    // The frozen partition reads `descr_set_keys` plus scalar
                    // EI metadata. Drop its live-object frozenset projection
                    // immediately instead of retaining a second cumulative
                    // Arc graph until codegen ends. Serialized BhCallDescrs
                    // below keep their own complete values.
                    effect_info._readonly_descrs_fields = None;
                    effect_info._write_descrs_fields = None;
                    effect_info._readonly_descrs_arrays = None;
                    effect_info._write_descrs_arrays = None;
                    effect_info._readonly_descrs_interiorfields = None;
                    effect_info._write_descrs_interiorfields = None;
                    effect_info.single_write_descr_array = None;
                    effect_info.extradescrs = None;
                    effect_info
                })
            })
            .collect();
        let cached_effect_info_count = cached_effect_infos.len();
        let frozen_layout = {
            let mut effect_infos: Vec<&mut majit_ir::EffectInfo> =
                cached_effect_infos.iter_mut().collect();
            effect_infos.extend(
                call_descrs
                    .iter_mut()
                    .map(|calldescr| &mut calldescr.extra_info),
            );
            majit_ir::effectinfo::compute_frozen_bitstrings(&mut effect_infos)
        };
        let serialized_effect_info_ids: Vec<u32> = frozen_layout
            .effect_info_ids
            .iter()
            .copied()
            .skip(cached_effect_info_count)
            .collect();
        assert_eq!(call_descrs.len(), serialized_effect_info_ids.len());
        let mut frozen_effect_infos = std::collections::BTreeMap::new();
        for (calldescr, effect_info_id) in call_descrs
            .into_iter()
            .zip(serialized_effect_info_ids.into_iter())
        {
            calldescr.translated_effect_info_id = Some(effect_info_id);
            frozen_effect_infos
                .entry(effect_info_id)
                .or_insert_with(|| calldescr.extra_info.clone());
            // The translated image stores one EffectInfo object and lets every
            // CallDescr point at it. Keep BhCallDescr's host-side field for
            // codewriter use, but serialize only this inert placeholder once
            // the dense identity above has been assigned.
            calldescr.extra_info = majit_ir::EffectInfo::MOST_GENERAL.clone();
        }
        for entry in &mut frozen_mints {
            let category = match &entry.member {
                majit_ir::effectinfo::DescrSetMember::Field { .. } => 0,
                majit_ir::effectinfo::DescrSetMember::Array { .. } => 1,
                majit_ir::effectinfo::DescrSetMember::InteriorField { .. } => 2,
            };
            // A descriptor can be minted while write-analysis is still
            // assembling a concrete EI and then survive in GcCache after a
            // later unrepresentable member degrades that EI to
            // EF_RANDOM_EFFECTS. `setup_descrs()` still enumerates it, while
            // effectinfo.py leaves its `ei_index = sys.maxint` because no
            // final raw frozenset names it. Preserve that sentinel instead of
            // pretending the mint log is exactly the final raw-set union.
            entry.ei_index = frozen_layout.descr_indices[category]
                .get(&entry.member)
                .copied()
                .unwrap_or(u32::MAX);
        }
        // `effectinfo.py compute_bitstrings` is the one writer of
        // `descr.ei_index`. A serialized `BhDescr::Array` still holds the
        // index the codewriter's live descr had before this freeze, and the
        // runtime descr rebuilt from it is stamped with that index, so give
        // it the frozen partition's slot (or the sentinel) the frozen
        // bitstrings test.
        fn restamp_array_ei_index(
            descr: &mut majit_translate::jitcode::BhDescr,
            arrays: &std::collections::BTreeMap<majit_ir::effectinfo::DescrSetMember, u32>,
        ) {
            match descr {
                majit_translate::jitcode::BhDescr::Array {
                    type_id, ei_index, ..
                } => {
                    *ei_index = arrays
                        .get(&majit_ir::effectinfo::DescrSetMember::Array { array_id: *type_id })
                        .copied()
                        .filter(|_| *type_id != 0)
                        .unwrap_or(u32::MAX);
                }
                majit_translate::jitcode::BhDescr::InteriorField { array, .. } => {
                    restamp_array_ei_index(array, arrays);
                }
                _ => {}
            }
        }
        for descr in &mut frozen_descrs {
            restamp_array_ei_index(descr, &frozen_layout.descr_indices[1]);
        }

        // JSON metadata for debugging
        let mut frozen_pipeline = pipeline.clone();
        frozen_pipeline.descrs = frozen_descrs.clone();
        frozen_pipeline.jitcodes = frozen_jitcodes.clone();
        frozen_pipeline.ei_descr_mints = frozen_mints.clone();
        let json = serde_json::to_string_pretty(&frozen_pipeline).unwrap();
        std::fs::write(format!("{out_dir}/jit_metadata.json"), &json).unwrap();

        // `pipeline.ei_descr_mints` is the whole analyzer GcCache mint log.
        // Runtime has already replayed every member reachable through
        // `Assembler.descrs` before it consumes this table, just as PyPy's
        // one process has already populated those cache slots. Carry only an
        // `(object key, ei_index)` stamp for such members. An unset index
        // means `compute_bitstrings` found the descriptor in none of the six
        // final raw sets (`effectinfo.py::compute_bitstrings`); if no opcode
        // names it either, it is analyzer-only dead state and has no runtime
        // consumer.
        // The remaining full specs are therefore precisely the cache misses
        // that a frozen EffectInfo will look up and runtime must mint.
        let opcode_members = opcode_published_descr_members(&frozen_descrs);
        let mut frozen_stamps = Vec::new();
        frozen_mints.retain(|entry| {
            if opcode_members.contains(&entry.member) {
                if entry.ei_index != u32::MAX {
                    frozen_stamps.push((entry.member.clone(), entry.ei_index));
                }
                return false;
            }
            if entry.ei_index != u32::MAX {
                return true;
            }
            false
        });

        // Persist `pipeline.jitcodes` (RPython `all_jitcodes` returned by
        // codewriter.py `make_jitcodes`) as individually encoded entries plus a name/offset
        // index. Runtime materializes entries lazily into the shared
        // MetaInterpStaticData jitcodes store — same single-store model as
        // RPython warmspot.py `WarmRunnerDesc.__init__` `self.metainterp_sd.jitcodes =
        // codewriter.make_jitcodes()`.
        let mut paths = vec![String::new(); frozen_jitcodes.len()];
        for (path, jitcode) in &pipeline.jitcodes_by_path {
            paths[jitcode.index()] = path.canonical_key();
        }
        let reachable = majit_translate::artifacts::reachable_symbolic_residual_table(
            &frozen_jitcodes,
            &frozen_descrs,
        );
        let (index, jitcodes_bin) =
            majit_translate::artifacts::JitCodeIndex::encode(&frozen_jitcodes, paths, reachable)
                .expect("encode JitCode archive");
        let jitcodes_index_bin = bincode::serialize(&index).unwrap();
        std::fs::write(format!("{out_dir}/jitcodes.bin"), &jitcodes_bin).unwrap();
        std::fs::write(format!("{out_dir}/jitcodes_index.bin"), &jitcodes_index_bin).unwrap();
        // Keep the shell fnaddr beside each dense index.  A translated PyPy
        // binary already owns these JitCode shells as AOT objects; pyre must
        // not deserialize every body merely to build
        // `bytecode_for_address`'s runtime fnaddr dictionary.
        let indirectcalltargets: Vec<(usize, i64)> = pipeline
            .indirectcalltarget_indices
            .iter()
            .map(|&index| (index, pipeline.jitcodes[index].fnaddr))
            .collect();
        let indirectcalltargets_bin = bincode::serialize(&indirectcalltargets).unwrap();
        std::fs::write(
            format!("{out_dir}/indirectcalltargets.bin"),
            &indirectcalltargets_bin,
        )
        .unwrap();

        // Persist the explicit portal → main-JitCode mapping. Runtime consumes
        // this directly instead of rediscovering the portal through name or flag
        // scans.
        let jit_drivers_bin = bincode::serialize(&pipeline.jit_drivers).unwrap();
        std::fs::write(format!("{out_dir}/jit_drivers.bin"), &jit_drivers_bin).unwrap();

        // Persist the runtime opname → u8 table so
        // `JitCode.code` (assembler-local mapping) decodes back to the
        // canonical `(opname, argcodes)` shape at runtime (shadow dispatch,
        // IR diffing).  RPython equivalent: the table handed to
        // `BlackholeInterpBuilder::setup_insns` at metainterp startup
        // (pyjitpl.py `MetaInterpStaticData.setup_insns`).
        //
        // RPython parity (assembler.py `Assembler.write_insn` `self.insns.setdefault(key,
        // len(self.insns))`): the table is the assembler's emission-driven
        // dict, populated by `write_insn` calls during graph flattening.
        // Pyre's analog is `pipeline.insns`, snapshotted from
        // `codewriter.assembler.insns()` after `make_jitcodes` finishes
        // (`make_jitcodes` in `majit-translate/src/lib.rs`).  Each distinct key gets a
        // fresh byte; the forward map is injective.  blackhole.py
        // `bhimpl_goto_if_not_int_is_true = bhimpl_goto_if_not`
        // aliases the bhimpl handler under two Python attribute names
        // (`bhimpl_goto_if_not_int_is_true = bhimpl_goto_if_not`) but
        // does NOT register a second opname in `Assembler.insns`; the
        // alias is at the dispatch-function-name level only.  Pyre
        // therefore registers exactly one opname per byte; the runtime
        // inverse (`byte → opname`) is 1:1 and panics on duplicate-byte
        // collisions (`jitcode_runtime.rs:INSNS_BYTE_TO_OPNAME`).
        //
        // Serialize through a `BTreeMap` view so the byte output is stable
        // across processes (Rust's `HashMap` SipHash makes raw iteration
        // non-deterministic; RPython's Python dict is insertion-ordered).
        let insns_sorted: std::collections::BTreeMap<&String, &u8> =
            pipeline.insns.iter().collect();
        let insns_bin = bincode::serialize(&insns_sorted).unwrap();
        std::fs::write(format!("{out_dir}/insns.bin"), &insns_bin).unwrap();

        // RPython `blackhole.py self.setup_descrs(asm.descrs)` + `:102-103
        // def setup_descrs(self, descrs): self.descrs = descrs`. Persists the
        // build-time assembler's shared descr pool so that 'd'/'j' argcodes
        // in `JitCode.code` resolve at runtime via
        // `BlackholeInterpBuilder::setup_descrs(...)` — the single-store
        // model (same list consumed by every `BlackholeInterpreter` produced
        // by `acquire_interp`).
        // Encode entries independently so runtime can retain the one-pointer
        // table shape upstream gets from translation-time constants without
        // reconstituting every descriptor in this process.
        let mut descrs_bin = Vec::new();
        let mut descr_offsets = Vec::with_capacity(frozen_descrs.len() + 1);
        let mut descr_kinds = Vec::with_capacity(frozen_descrs.len());
        let mut descr_parent_layouts = Vec::with_capacity(frozen_descrs.len());
        let mut descr_size_type_ids = Vec::with_capacity(frozen_descrs.len());
        let mut canonical_layouts: Vec<std::sync::Arc<majit_translate::jitcode::BhSizeSpec>> =
            Vec::new();
        let mut conflicting_layouts: Vec<(u64, String)> = Vec::new();
        descr_offsets.push(0_u32);
        for descr in &frozen_descrs {
            // `GcCache.setup_descrs` publishes its structural descriptors
            // before call descriptors. `SwitchDictDescr` and the pyre opcode
            // adapters do not belong to any of GcCache's six dictionaries;
            // they remain solely in `Assembler.descrs` and are decoded lazily
            // by the opcode reader. Persist that three-way classification so
            // runtime setup does not materialize and immediately drop them.
            descr_kinds.push(match descr {
                majit_translate::jitcode::BhDescr::Call { .. }
                | majit_translate::jitcode::BhDescr::JitCode { .. } => 1_u8,
                majit_translate::jitcode::BhDescr::Field { .. }
                | majit_translate::jitcode::BhDescr::Array { .. }
                | majit_translate::jitcode::BhDescr::InteriorField { .. }
                | majit_translate::jitcode::BhDescr::Size { .. } => 0_u8,
                majit_translate::jitcode::BhDescr::Switch { .. }
                | majit_translate::jitcode::BhDescr::VableField { .. }
                | majit_translate::jitcode::BhDescr::VableArray { .. }
                | majit_translate::jitcode::BhDescr::VtableMethod { .. } => 2_u8,
            });
            // `GcCache.get_field_descr` attaches every FieldDescr to the
            // one SizeDescr in `GcCache._cache_size[STRUCT]`; the parent and
            // its `all_fielddescrs` are not copied into each field.  Preserve
            // that object graph across pyre's process boundary: serialize
            // each parent layout once, and put only its dense reference next
            // to the opcode-descr slot.  A linear ordered table is deliberate
            // here — it is the translated constant list, not a second
            // semantic cache beside GcCache.
            let mut wire_descr = descr.clone();
            let parent_layout = match descr {
                majit_translate::jitcode::BhDescr::Field {
                    parent: Some(parent),
                    ..
                } => {
                    let index = canonical_layouts
                        .iter()
                        .position(|known| {
                            if parent.type_id != 0 {
                                known.type_id == parent.type_id
                            } else {
                                known.same_descr_layout(parent)
                            }
                        })
                        .unwrap_or_else(|| {
                            canonical_layouts.push(parent.clone());
                            canonical_layouts.len() - 1
                        });
                    if !canonical_layouts[index].same_descr_layout(parent) {
                        if !conflicting_layouts
                            .iter()
                            .any(|(type_id, _)| *type_id == parent.type_id)
                        {
                            conflicting_layouts.push((
                                parent.type_id,
                                format!(
                                    "type_id={:#x}\n  first={:?}\n  later={:?}",
                                    parent.type_id, canonical_layouts[index], parent,
                                ),
                            ));
                        }
                    }
                    Some(u32::try_from(index).expect("too many canonical descriptor layouts"))
                }
                _ => None,
            };
            if let majit_translate::jitcode::BhDescr::Field { parent, .. } = &mut wire_descr {
                *parent = None;
            }
            descr_parent_layouts.push(parent_layout.unwrap_or(u32::MAX));
            let size_type_id = match descr {
                majit_translate::jitcode::BhDescr::Size { type_id, .. } => *type_id,
                _ => 0,
            };
            descr_size_type_ids.push(size_type_id);
            // `descr.py` `get_size_descr` reads the STRUCT row `finish_setup_descrs`
            // already numbered. A parentless size with a cache key is that row:
            // pack it the same way as `descr_layouts.bin`, not as a bincode enum.
            if let majit_translate::jitcode::BhDescr::Size {
                size,
                type_id,
                vtable,
                owner,
                all_fielddescrs,
                is_gc_managed,
            } = &wire_descr
                && *type_id != 0
            {
                majit_translate::jitcode::BhSizeSpec {
                    size: *size,
                    type_id: *type_id,
                    vtable: *vtable,
                    owner: owner.clone(),
                    is_gc_managed: *is_gc_managed,
                    headerless: owner == majit_translate::jitcode::HEADERLESS_SIZE_OWNER_MARKER,
                    all_fielddescrs: all_fielddescrs.clone(),
                }
                .pack_into(&mut descrs_bin);
            } else {
                descrs_bin.extend(bincode::serialize(&wire_descr).unwrap());
            }
            descr_offsets.push(
                u32::try_from(descrs_bin.len())
                    .expect("serialized descrs.bin exceeds the u32 offset range"),
            );
        }
        assert_eq!(descr_offsets.len(), frozen_descrs.len() + 1);
        assert_eq!(descr_kinds.len(), frozen_descrs.len());
        assert_eq!(descr_parent_layouts.len(), frozen_descrs.len());
        assert!(
            conflicting_layouts.is_empty(),
            "one STRUCT identity produced conflicting SizeDescr layouts:\n{}",
            conflicting_layouts
                .iter()
                .map(|(_, message)| message.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        );
        assert_eq!(descr_offsets.first().copied(), Some(0));
        assert_eq!(descr_offsets.last().copied(), Some(descrs_bin.len() as u32));
        assert!(descr_offsets.windows(2).all(|pair| pair[0] <= pair[1]));
        let descrs_index_bin = bincode::serialize(&(
            descr_offsets,
            descr_kinds,
            descr_parent_layouts,
            descr_size_type_ids,
        ))
        .unwrap();
        std::fs::write(format!("{out_dir}/descrs.bin"), &descrs_bin).unwrap();
        std::fs::write(format!("{out_dir}/descrs_index.bin"), &descrs_index_bin).unwrap();

        let mut descr_layouts_bin = Vec::new();
        let mut descr_layout_offsets = Vec::with_capacity(canonical_layouts.len() + 1);
        descr_layout_offsets.push(0_u32);
        for layout in &canonical_layouts {
            // `pyjitpl.py` `finish_setup_descrs` numbers the layout translation
            // already built. Pack that row; startup must not bincode it again.
            layout.pack_into(&mut descr_layouts_bin);
            descr_layout_offsets.push(
                u32::try_from(descr_layouts_bin.len())
                    .expect("serialized descr_layouts.bin exceeds the u32 offset range"),
            );
        }
        let descr_layouts_index_bin = bincode::serialize(&descr_layout_offsets).unwrap();
        std::fs::write(format!("{out_dir}/descr_layouts.bin"), &descr_layouts_bin).unwrap();
        std::fs::write(
            format!("{out_dir}/descr_layouts_index.bin"),
            &descr_layouts_index_bin,
        )
        .unwrap();

        // RPython's translated constants preserve EffectInfo object identity:
        // thousands of CallDescrs point at a small canonical object table.
        // Keep the same ownership instead of repeating each EffectInfo inside
        // every BhCallDescr wire record. Entries stay independently decodable
        // so runtime setup drops their structural key strings immediately.
        let mut effect_infos_bin = Vec::new();
        let mut effect_info_offsets = Vec::with_capacity(frozen_effect_infos.len() + 1);
        effect_info_offsets.push(0_u32);
        for (translated_id, effect_info) in &frozen_effect_infos {
            effect_infos_bin.extend(bincode::serialize(&(*translated_id, effect_info)).unwrap());
            effect_info_offsets.push(
                u32::try_from(effect_infos_bin.len())
                    .expect("serialized effect_infos.bin exceeds the u32 offset range"),
            );
        }
        let effect_infos_index_bin = bincode::serialize(&effect_info_offsets).unwrap();
        std::fs::write(format!("{out_dir}/effect_infos.bin"), &effect_infos_bin).unwrap();
        std::fs::write(
            format!("{out_dir}/effect_infos_index.bin"),
            &effect_infos_index_bin,
        )
        .unwrap();

        // `MAJIT_MINT_INDEX_CENSUS=1`: how many `fielddescrof` mints resolved a
        // slot for `index_in_parent`, and how many carried out the `0` they were
        // initialised with. The split the emitted descr cannot record — an
        // unwritten initialiser and a real slot-0 claim are the same bytes once
        // this function returns.
        //
        // Read in the minting build process: runtime statistics run in another
        // process and would see fresh zeroed atomics. A cached build emits no
        // warning, so force the build script to rerun when collecting this
        // census.
        if std::env::var_os("MAJIT_MINT_INDEX_CENSUS").is_some() {
            let [claimed, placeholder] = majit_ir::descr::GcCache::mint_index_census();
            println!(
                "cargo::warning=mint_index_census: claimed={claimed} placeholder={placeholder} \
                 (fielddescrof mints; placeholder=0 means no descr in this build carries an \
                 unresolved index_in_parent)"
            );
        }

        // The table above is RPython's `opcode_descrs` (pyjitpl.py
        // `MetaInterpStaticData.finish_setup` `self.setup_descrs(asm.descrs)`), not its `all_descrs` (`pyjitpl.py
        // self.cpu.setup_descrs()` = the full gccache walk at `descr.py`).
        // Upstream never has to distinguish them here because one gccache serves
        // one process, so `compute_bitstrings` unions descrs that are already
        // present. Pyre mints in this process and resolves in another, so the
        // raw-set members no opcode names would have no slot on the far side;
        // persist their mint arguments and let the runtime cache take the same
        // descr.py `get_field_descr` miss branch this one did.
        // Mint specs are one-shot setup inputs.  Keep the same indexed-entry
        // shape as `descrs.bin` so the runtime can deserialize and publish one
        // spec at a time instead of retaining all member/spec strings at the
        // peak of first-JIT setup.
        let mut ei_descr_mints_bin = Vec::new();
        let mut ei_descr_mint_offsets = Vec::with_capacity(frozen_mints.len() + 1);
        ei_descr_mint_offsets.push(0_u32);
        for entry in &frozen_mints {
            ei_descr_mints_bin.extend(bincode::serialize(entry).unwrap());
            ei_descr_mint_offsets.push(
                u32::try_from(ei_descr_mints_bin.len())
                    .expect("serialized ei_descr_mints.bin exceeds the u32 offset range"),
            );
        }
        let ei_descr_mints_index_bin = bincode::serialize(&ei_descr_mint_offsets).unwrap();
        std::fs::write(format!("{out_dir}/ei_descr_mints.bin"), &ei_descr_mints_bin).unwrap();
        std::fs::write(
            format!("{out_dir}/ei_descr_mints_index.bin"),
            &ei_descr_mints_index_bin,
        )
        .unwrap();

        let mut ei_descr_stamps_bin = Vec::new();
        let mut ei_descr_stamp_offsets = Vec::with_capacity(frozen_stamps.len() + 1);
        ei_descr_stamp_offsets.push(0_u32);
        for stamp in &frozen_stamps {
            ei_descr_stamps_bin.extend(bincode::serialize(stamp).unwrap());
            ei_descr_stamp_offsets.push(
                u32::try_from(ei_descr_stamps_bin.len())
                    .expect("serialized ei_descr_stamps.bin exceeds the u32 offset range"),
            );
        }
        let ei_descr_stamps_index_bin = bincode::serialize(&ei_descr_stamp_offsets).unwrap();
        std::fs::write(
            format!("{out_dir}/ei_descr_stamps.bin"),
            &ei_descr_stamps_bin,
        )
        .unwrap();
        std::fs::write(
            format!("{out_dir}/ei_descr_stamps_index.bin"),
            &ei_descr_stamps_index_bin,
        )
        .unwrap();

        // Analyzer-side descriptor producers run in this build-script process,
        // while the runtime formats the field-position report. Persist the
        // producer census beside the mint ledger.
        let field_mint_census_bin =
            bincode::serialize(&majit_ir::descr::field_mint_census_snapshot()).unwrap();
        std::fs::write(
            format!("{out_dir}/field_mint_census.bin"),
            &field_mint_census_bin,
        )
        .unwrap();

        // RPython `pyjitpl.py MetaInterpStaticData.finish_setup self.liveness_info = "".join(asm.all_liveness)`.
        // Persist the build-time assembler's shared `all_liveness` byte stream so a
        // runtime consumer re-tracing a build-time jitcode (whose `BC_LIVE` ops
        // carry offsets baked against this table) can install it into
        // `metainterp_sd.liveness_info` and resolve those offsets.
        let liveness_bin = bincode::serialize(&pipeline.all_liveness).unwrap();
        std::fs::write(format!("{out_dir}/liveness.bin"), &liveness_bin).unwrap();

        // RPython's translator AOT-compiles every helper into a single binary, so
        // `JitCode.fnaddr` / `constants_i` funcptrs are linker-resolved and stable
        // at runtime.  Pyre's `majit-translate` runs in `build.rs` — a separate
        // process from `pyre-dynasm` — so every fnaddr captured here is the
        // build-script process's address, which ASLR (and the divergent executable
        // layouts) invalidates at runtime.  Persist the `(path, build_fnaddr)`
        // table the codewriter consumed so the runtime patcher
        // (`runtime_fnaddr_patch::patch_constants_i_fnaddrs`) can pair each build
        // address with the matching runtime address from
        // `pyre_interpreter::jit_trace_fnaddrs()` and overwrite stale constants
        // before the walker invokes them.
        let fnaddr_bindings_owned: Vec<(String, i64)> = fnaddr_bindings
            .iter()
            .map(|(p, a)| ((*p).to_string(), *a))
            .collect();
        let fnaddr_bindings_bin = bincode::serialize(&fnaddr_bindings_owned).unwrap();
        std::fs::write(
            format!("{out_dir}/fnaddr_bindings.bin"),
            &fnaddr_bindings_bin,
        )
        .unwrap();
        // `symbolic_fnaddr_for_path` hashes have no reverse map in the runtime
        // process: the codewriter records them only while it runs, in the
        // build script. Persist that snapshot so a residual-call bail can
        // name the path. Hashes are path identity, not host addresses.
        std::fs::write(
            format!("{out_dir}/symbolic_fnaddr_paths.bin"),
            bincode::serialize(&pipeline.symbolic_fnaddr_paths).unwrap(),
        )
        .unwrap();

        // Same ASLR hazard for the static-data addresses the codewriter baked
        // into `constants_i` (host `PyType` singletons and prebuilt refs supplied
        // via `HostStaticAddrs`): the build-script process's `&pyre_object::X`
        // address does not survive into the runtime executable.  Persist the
        // `(name, build_addr)` tables so `runtime_fnaddr_patch::
        // patch_constants_i_static_addrs` can re-pair them with the runtime
        // addresses from `jit_static_pytype_addrs` / `jit_static_ref_addrs`.
        let pytype_bindings_owned: Vec<(String, i64)> = static_pytype_addrs
            .iter()
            .map(|(n, a)| ((*n).to_string(), *a))
            .collect();
        std::fs::write(
            format!("{out_dir}/static_pytype_bindings.bin"),
            bincode::serialize(&pytype_bindings_owned).unwrap(),
        )
        .unwrap();
        let ref_bindings_owned: Vec<(String, i64)> = static_ref_addrs
            .iter()
            .map(|(n, a)| ((*n).to_string(), *a))
            .collect();
        std::fs::write(
            format!("{out_dir}/static_ref_bindings.bin"),
            bincode::serialize(&ref_bindings_owned).unwrap(),
        )
        .unwrap();

        // Report
        eprintln!(
            "[pyre-jit-trace build.rs] canonical analysis: {} JIT drivers, {} functions, {} blocks, {} flat ops, {} all_jitcodes ({} bytes bodies + {} bytes index), generated {} bytes",
            pipeline.jit_drivers.len(),
            pipeline.functions.len(),
            pipeline.total_blocks,
            pipeline.total_ops,
            pipeline.jitcodes.len(),
            jitcodes_bin.len(),
            jitcodes_index_bin.len(),
            code.len(),
        );
    };

    generate_into(&out_dir);

    let reproducible = match determinism_check {
        DeterminismCheck::Off => true,
        DeterminismCheck::InProcess => {
            let probe_dir = std::path::Path::new(&out_dir).join(DETERMINISM_PROBE_SUBDIR);
            let _ = std::fs::remove_dir_all(&probe_dir);
            std::fs::create_dir_all(&probe_dir).expect("create the determinism probe directory");
            eprintln!(
                "[pyre-jit-trace build.rs] codegen determinism: generating a second time into {}",
                probe_dir.display()
            );
            // The second generation runs against whatever process-global
            // state the first one left behind. If that state makes it panic,
            // the outputs are not a function of the inputs either, so report
            // it rather than take down a build whose own outputs — the first
            // generation's, already written — are intact.
            let completed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                generate_into(&probe_dir.to_string_lossy())
            }))
            .is_ok();
            if completed {
                codegen_outputs_match(
                    "a second in-process generation",
                    std::path::Path::new(&out_dir),
                    &probe_dir,
                    false,
                )
            } else {
                report_determinism(&format!(
                    "the second in-process generation panicked (see the panic above), so \
                     nothing was compared; the codegen does not survive running twice in one \
                     process. Compare across processes with {DETERMINISM_CHECK_ENV}=cache \
                     instead"
                ));
                false
            }
        }
        DeterminismCheck::AgainstCache => {
            if CODEGEN_OUTPUTS
                .iter()
                .all(|name| cache_dir.join(name).is_file())
            {
                codegen_outputs_match(
                    &format!("the stored cache entry {cache_key}"),
                    &cache_dir,
                    std::path::Path::new(&out_dir),
                    true,
                )
            } else {
                // Compared nothing, which is neither a pass nor a failure —
                // say so rather than let an empty comparison read as clean.
                // Storing below is what gives the next run something to
                // compare against.
                report_determinism(&format!(
                    "cache key {cache_key} has no stored entry, so nothing was compared; \
                     re-run this build to compare against the entry it is about to store"
                ));
                true
            }
        }
    };
    if !reproducible {
        // What the refusal actually costs differs by mode, so only the
        // consequence is chosen and the report is written once. `AgainstCache`
        // reaches a `false` verdict only when an entry already exists — that
        // entry is what it compared against — and `store_codegen_cache`
        // returns early when it does, so there was never a store to refuse.
        // Saying "refusing to store them" there would tell a reader the bad
        // outputs were kept out of the cache while the unverified entry sits
        // in it, ready for the next ordinary build to restore.
        let consequence = match determinism_check {
            DeterminismCheck::AgainstCache => format!(
                "nothing was stored — the entry it disagreed with was already there and stays, \
                 so the next ordinary build restores it; remove {} to stop that",
                cache_dir.display()
            ),
            _ => "refusing to store them".to_string(),
        };
        report_determinism(&format!(
            "the outputs at cache key {cache_key} are not reproducible; {consequence}"
        ));
        return;
    }

    if let Err(e) = store_codegen_cache(&cache_dir, &out_dir) {
        eprintln!(
            "[pyre-jit-trace build.rs] warning: could not store generated JIT trace cache {}: {e}",
            cache_key
        );
        return;
    }
    touch_codegen_cache_entry(&cache_dir);
    prune_codegen_cache(&repo_root, &cache_dir);
}

fn build_call_effect_overrides() -> Vec<majit_translate::CallEffectOverride> {
    call_spec::PYFRAME_CALL_EFFECTS
        .iter()
        .map(|spec| {
            let target = match spec.target {
                call_spec::CallTargetSpec::Method {
                    name,
                    receiver_root,
                } => majit_translate::CallTarget::method(name, Some(receiver_root.to_string())),
                call_spec::CallTargetSpec::FunctionPath(segments) => {
                    majit_translate::CallTarget::function_path(segments.iter().copied())
                }
            };
            let effect = match spec.effect {
                call_spec::CallEffectKind::Elidable => majit_translate::CallEffectKind::Elidable,
                call_spec::CallEffectKind::Residual => majit_translate::CallEffectKind::Residual,
                call_spec::CallEffectKind::Declared(declared) => {
                    majit_translate::CallEffectKind::Declared(
                        majit_translate::DeclaredCallEffects {
                            extra: declared_extra_effect(declared.extra),
                            can_collect: declared.can_collect,
                            can_invalidate: declared.can_invalidate,
                        },
                    )
                }
            };
            majit_translate::CallEffectOverride::new(target, effect)
        })
        .collect()
}

/// Carry one declared `EF_*` across the crate boundary.
///
/// Two enums rather than one shared type because `call_spec.rs` is the
/// data-only contract pyre owns and `majit-translate` is the consumer; the
/// arms are enumerated so adding a member on either side is a compile error
/// here rather than a silent re-mapping.
fn declared_extra_effect(
    extra: call_spec::DeclaredExtraEffect,
) -> majit_translate::DeclaredExtraEffect {
    use call_spec::DeclaredExtraEffect as Spec;
    use majit_translate::DeclaredExtraEffect as Translate;
    match extra {
        Spec::ElidableCannotRaise => Translate::ElidableCannotRaise,
        Spec::LoopInvariant => Translate::LoopInvariant,
        Spec::CannotRaise => Translate::CannotRaise,
        Spec::ElidableOrMemoryError => Translate::ElidableOrMemoryError,
        Spec::ElidableCanRaise => Translate::ElidableCanRaise,
        Spec::CanRaise => Translate::CanRaise,
        Spec::ForcesVirtualOrVirtualizable => Translate::ForcesVirtualOrVirtualizable,
    }
}

fn emit_rerun_directives(repo_root: &str, source_paths: &[String]) {
    for path in source_paths {
        println!("cargo::rerun-if-changed={path}");
    }
    let inputs = codegen_cache::discover(
        std::path::Path::new(repo_root),
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
    );
    for rel in &inputs.src_trees {
        emit_rerun_if_changed_recursive(&format!("{repo_root}/{rel}"));
    }
    for rel in inputs.extra_files.iter().chain(&inputs.manifests) {
        println!("cargo::rerun-if-changed={repo_root}/{rel}");
    }
    // Neither reaches `codegen_cache_key` -- they choose where the cache lives
    // and how much of it is kept, not what it contains -- but cargo still has
    // to re-run this script when they move so the outputs land in, and are
    // pruned against, the directory now in force.
    println!("cargo::rerun-if-env-changed={CODEGEN_CACHE_DIR_ENV}");
    println!("cargo::rerun-if-env-changed={CODEGEN_CACHE_ENTRIES_ENV}");
    println!("cargo::rerun-if-env-changed=MAJIT_RTYPER_VERBOSE");
    println!("cargo::rerun-if-env-changed=MAJIT_CALLEE_CENSUS");
    println!("cargo::rerun-if-env-changed=MAJIT_CALLEE_CENSUS_ROWS");
    println!("cargo::rerun-if-env-changed=MAJIT_FIELD_MINT_TRACE");
    println!("cargo::rerun-if-env-changed=MAJIT_STRUCT_LAYOUT_CENSUS");
    println!("cargo::rerun-if-env-changed={DETERMINISM_CHECK_ENV}");
    for &(runtime_name, _) in ENGINE_ENV_ALIASES {
        println!("cargo::rerun-if-env-changed={runtime_name}");
    }
    // Re-runs this script without changing anything it hashes. That is the
    // only way to exercise `DeterminismCheck::AgainstCache`, which needs two
    // runs at the *same* cache key: with the gate value held constant Cargo
    // considers the script fresh and skips the second run, and any input edit
    // that would force one also rekeys the cache and leaves nothing to
    // compare against. Deliberately absent from `codegen_cache_key`.
    println!("cargo::rerun-if-env-changed={DETERMINISM_NONCE_ENV}");
    for key in LOWERING_GATE_ENV {
        println!("cargo::rerun-if-env-changed={key}");
    }
    // The mir-frontend analysis derives `jit_trace_gen.rs` from
    // the workspace LLBC artefacts or the `MAJIT_MIR_FRONTEND_LLBC`
    // override. Track both so re-extracting LLBC or repointing the override
    // invalidates Cargo's build-script cache and our content cache key.
    println!("cargo::rerun-if-env-changed=MAJIT_MIR_FRONTEND_LLBC");
    if let Some(paths) = std::env::var_os("MAJIT_MIR_FRONTEND_LLBC") {
        for path in std::env::split_paths(&paths) {
            if !path.as_os_str().is_empty() {
                println!("cargo::rerun-if-changed={}", path.display());
            }
        }
    }
    // Derived from `LLBC_CRATES` rather than repeated: a crate added there is a
    // production input, and a copy of the list that missed it would let cargo
    // skip this script after that artefact alone changed.
    for crate_name in LLBC_CRATES {
        println!("cargo::rerun-if-changed={repo_root}/build/{LLBC_SUBDIR}/{crate_name}.ullbc");
    }
    for sidecar in llbc_layout_sidecars() {
        println!("cargo::rerun-if-changed={repo_root}/build/{LLBC_SUBDIR}/{sidecar}");
    }
}

/// Layout-sidecar artefact names this build consumes, empty for a native
/// build.
///
/// `scripts/extract-llbc.py` re-extracts each crate for every cross target
/// and reduces the result to its `type_decls`; `majit-translate`'s
/// `auto_discover_workspace_llbc_paths` merges them ahead of the host
/// artefacts so their Charon-resolved field offsets win.
fn llbc_layout_sidecars() -> Vec<String> {
    let target = std::env::var("TARGET").unwrap_or_default();
    let host = std::env::var("HOST").unwrap_or_default();
    if !majit_translate::layout::is_cross_target(&target, &host) {
        return Vec::new();
    }
    // `pyre-jit` is absent: it has no sidecar (see the extraction driver's
    // spec for why), so only the object model gets cross-target layouts.
    // The naming convention lives in `majit_translate::layout` so it stays in
    // lockstep with `auto_discover_workspace_llbc_paths`.
    ["majit-rlib", "pyre-object", "pyre-interpreter"]
        .iter()
        .map(|crate_name| majit_translate::layout::layout_sidecar_filename(crate_name, &target))
        .collect()
}

/// The cache sits under `build/` rather than the Cargo target directory so
/// `cargo clean`, a fresh `CARGO_TARGET_DIR`, or a profile switch does not
/// re-pay the translation prepass — the same reason the prepass's own inputs
/// (`build/llbc/*.ullbc`) live there. `codegen_cache_key` already folds
/// HOST/TARGET/PROFILE/OPT_LEVEL, the feature set, the translator sources
/// and the LLBC content, so an entry is only ever served back to the
/// configuration that produced it.
/// The cache root when several checkouts are pointed at one, or `None` for the
/// per-worktree default.
///
/// Sharing is sound because `codegen_cache_key` derives the key from content
/// alone -- translator sources, `Cargo.lock`, the rustc version string, the
/// LLBC inputs -- and from repo-relative paths, so an entry stored by one
/// checkout is legible to every other on the same sources and artefacts. The
/// store is already safe to race on: it stages into a pid-suffixed sibling
/// and renames, and concedes without error when another process got there
/// first.
///
/// What that buys is capacity, not (usually) deduplication. Checkouts on
/// different branches disagree on the key -- twelve of them held 77 distinct
/// keys with none in common -- because the extracted LLBC alone differs
/// wherever the interpreter does. Two checkouts of the same tree now share
/// an entry: the key uses repo-relative paths. The remaining win of the
/// shared pool is that one LRU hands its slots to whichever checkouts are
/// building, instead of every idle one holding a local quota that nothing
/// is going to ask for.
///
/// Deliberately absent from `codegen_cache_key`: keying on where the cache
/// lives would give each worktree its own key again and defeat the point.
fn codegen_cache_shared_base() -> Option<std::path::PathBuf> {
    let dir = std::env::var_os(CODEGEN_CACHE_DIR_ENV)?;
    if dir.is_empty() {
        return None;
    }
    Some(std::path::PathBuf::from(dir))
}

fn codegen_cache_base(repo_root: &str) -> std::path::PathBuf {
    codegen_cache_shared_base()
        .unwrap_or_else(|| std::path::Path::new(repo_root).join("build/pyre-jit-trace-cache"))
}

fn codegen_cache_max_entries() -> usize {
    if let Some(n) = std::env::var(CODEGEN_CACHE_ENTRIES_ENV)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        return n.max(1);
    }
    if codegen_cache_shared_base().is_some() {
        CODEGEN_CACHE_MAX_ENTRIES_SHARED
    } else {
        CODEGEN_CACHE_MAX_ENTRIES_LOCAL
    }
}

/// Whether nothing under `version` has been used for `age`.
///
/// Reads each entry's `.last-used` marker, which a restore rewrites, and falls
/// back to the entry's own mtime; the newest of them dates the version.
fn codegen_cache_version_idle(version: &std::path::Path, age: std::time::Duration) -> bool {
    let newest = std::fs::read_dir(version)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            std::fs::metadata(entry.path().join(CODEGEN_CACHE_USED_MARKER))
                .or_else(|_| entry.metadata())
                .and_then(|meta| meta.modified())
                .ok()
        })
        .max();
    let stamp = match newest.or_else(|| {
        std::fs::metadata(version)
            .and_then(|meta| meta.modified())
            .ok()
    }) {
        Some(stamp) => stamp,
        // No readable timestamp: leave it alone rather than guess it is dead.
        None => return false,
    };
    stamp.elapsed().map(|since| since > age).unwrap_or(false)
}

fn codegen_cache_dir(repo_root: &str, cache_key: &str) -> std::path::PathBuf {
    codegen_cache_base(repo_root)
        .join(CODEGEN_CACHE_VERSION)
        .join(cache_key)
}

/// Record that `cache_dir` was used, so [`prune_codegen_cache`] evicts by last
/// use rather than by creation: a configuration built daily but keyed long ago
/// would otherwise be dropped ahead of one-off keys minted by a source edit.
fn touch_codegen_cache_entry(cache_dir: &std::path::Path) {
    let _ = std::fs::write(cache_dir.join(CODEGEN_CACHE_USED_MARKER), b"");
}

/// Drop entries beyond [`codegen_cache_max_entries`], least recently used
/// first, along with any directory left by an earlier `CODEGEN_CACHE_VERSION`.
/// Every distinct (target, profile, feature set, build-script binary, LLBC
/// content) combination mints a key and nothing removed them before, so the
/// directory grew without bound — 110 entries / 682 MB in one worktree.
///
/// Best effort throughout: a concurrent build whose entry is removed
/// mid-restore re-runs the prepass, because `restore_codegen_cache` copies
/// every output or reports failure — it never serves a partial set.
fn prune_codegen_cache(repo_root: &str, keep: &std::path::Path) {
    let base = codegen_cache_base(repo_root);
    let shared = codegen_cache_shared_base().is_some();
    if let Ok(versions) = std::fs::read_dir(&base) {
        for version in versions.flatten() {
            if version.file_name() == std::ffi::OsStr::new(CODEGEN_CACHE_VERSION) {
                continue;
            }
            // A worktree-local cache has one owner, so every other version is
            // this checkout's own leftovers and goes immediately. A shared one
            // is also read by checkouts sitting at commits whose version is
            // live *for them*: deleting it there would have each worktree
            // wiping the others' cache on every build. Wait for it to go
            // untouched instead.
            if shared && !codegen_cache_version_idle(&version.path(), CODEGEN_CACHE_STALE_VERSION) {
                continue;
            }
            let _ = std::fs::remove_dir_all(version.path());
        }
    }
    let Ok(entries) = std::fs::read_dir(base.join(CODEGEN_CACHE_VERSION)) else {
        return;
    };
    let mut by_last_use: Vec<(std::time::SystemTime, std::path::PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        // `store_codegen_cache` stages into a dot-prefixed sibling before its
        // rename; leaving those alone keeps a concurrent store intact.
        if path == keep || !path.is_dir() || entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let last_use = std::fs::metadata(path.join(CODEGEN_CACHE_USED_MARKER))
            .or_else(|_| std::fs::metadata(&path))
            .and_then(|meta| meta.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        by_last_use.push((last_use, path));
    }
    by_last_use.sort_by_key(|item| std::cmp::Reverse(item.0));
    // `keep` is excluded above and occupies one of the retained slots.
    let retained = codegen_cache_max_entries().saturating_sub(1);
    for (_, path) in by_last_use.into_iter().skip(retained) {
        let _ = std::fs::remove_dir_all(path);
    }
}

/// Compare two generated output sets file by file. True only when every
/// [`CODEGEN_OUTPUTS`] entry was read on both sides and is byte-identical: an
/// output that cannot be read on one side is an inconclusive comparison, and
/// an inconclusive comparison must not report as a clean one.
///
/// Reports through [`report_determinism`], which writes `cargo::warning` —
/// cargo surfaces that for a workspace crate's build script, where a bare
/// `println!` / `eprintln!` needs `-vv`, which is how a build-script line
/// documenting the cache restore went unread long enough to be mistaken for the
/// restore not happening — and repeats the same finding on stderr for a build
/// script invoked directly.
///
/// Names the identical outputs as well as the differing ones. Which files hold
/// still localises the cause faster than which files move: `jitcodes.bin`
/// moving while `jitcodes_index.bin` holds says the entries changed while
/// their names and boundaries did not, which rules out the walk order and
/// points at what is written into each entry. A size change and an
/// equal-size change are different findings too — one moved a count, the
/// other only an order — so both lengths are reported, not just that they
/// differ.
///
/// `cross_process` excuses [`HOST_ADDRESSED_OUTPUTS`] from the verdict; its
/// inverse excuses [`IN_PROCESS_STATEFUL_OUTPUTS`]. Both are still reported on
/// their own lines: a check that files a by-design difference as a defect is one
/// people learn to ignore, and then it reports nothing at all.
/// Emit one determinism finding on both streams.
///
/// `cargo::warning` is honoured on stdout only, while every line this build
/// script narrates is an `eprintln!` on stderr. A build script invoked directly
/// — the prepass rig runs one with the two streams captured to separate files —
/// therefore put its pass sentence on stderr and its entire failure report on
/// stdout, so filtering stderr for `codegen determinism` yielded the two
/// exclusion lines and no verdict at all. That reads as a quieter pass, and a
/// failing generation was taken for a passing one on exactly that reading. The
/// direct run also exits 0 either way, so the exit code does not correct it.
fn report_determinism(finding: &str) {
    println!("cargo::warning=codegen determinism: {finding}");
    eprintln!("[pyre-jit-trace build.rs] codegen determinism: {finding}");
}

fn codegen_outputs_match(
    label: &str,
    baseline: &std::path::Path,
    probe: &std::path::Path,
    cross_process: bool,
) -> bool {
    let mut identical: Vec<&str> = Vec::new();
    let mut differing: Vec<String> = Vec::new();
    let mut host_addressed: Vec<&str> = Vec::new();
    let mut in_process_stateful: Vec<&str> = Vec::new();
    let mut unreadable: Vec<String> = Vec::new();
    for name in CODEGEN_OUTPUTS {
        match (
            std::fs::read(baseline.join(name)),
            std::fs::read(probe.join(name)),
        ) {
            (Ok(a), Ok(b)) if a == b => identical.push(name),
            (Ok(_), Ok(_)) if cross_process && HOST_ADDRESSED_OUTPUTS.contains(name) => {
                host_addressed.push(name);
            }
            (Ok(_), Ok(_)) if !cross_process && IN_PROCESS_STATEFUL_OUTPUTS.contains(name) => {
                in_process_stateful.push(name);
            }
            (Ok(a), Ok(b)) => {
                let offset = a
                    .iter()
                    .zip(b.iter())
                    .position(|(x, y)| x != y)
                    .unwrap_or_else(|| a.len().min(b.len()));
                differing.push(format!(
                    "{name} ({} vs {} bytes, first difference at offset {offset})",
                    a.len(),
                    b.len()
                ));
            }
            (a, b) => unreadable.push(format!(
                "{name} (baseline {}, probe {})",
                if a.is_ok() { "read" } else { "unreadable" },
                if b.is_ok() { "read" } else { "unreadable" }
            )),
        }
    }
    if !host_addressed.is_empty() {
        eprintln!(
            "[pyre-jit-trace build.rs] codegen determinism: {} host-addressed output(s) differ \
             from {label} as expected, excluded from the verdict: {}",
            host_addressed.len(),
            host_addressed.join(" ")
        );
    }
    if !in_process_stateful.is_empty() {
        eprintln!(
            "[pyre-jit-trace build.rs] codegen determinism: {} process-stateful output(s) \
             differ from {label} as expected, excluded from the in-process verdict: {}",
            in_process_stateful.len(),
            in_process_stateful.join(" ")
        );
    }
    if differing.is_empty() && unreadable.is_empty() {
        eprintln!(
            "[pyre-jit-trace build.rs] codegen determinism: all {} reproducible outputs are \
             identical to {label}",
            identical.len()
        );
        return true;
    }
    for entry in &differing {
        report_determinism(&format!("DIFFERS from {label}: {entry}"));
    }
    for entry in &unreadable {
        report_determinism(&format!("NOT COMPARED against {label}: {entry}"));
    }
    report_determinism(&format!(
        "{} of {} outputs unchanged from {label}: {}",
        identical.len(),
        CODEGEN_OUTPUTS.len(),
        identical.join(" ")
    ));
    // Both exclusions already went to stderr above, in either outcome. Repeat
    // them as warnings so the cargo-visible report of a failure carries what
    // was left out of the verdict it just reported; a plain `println!` here
    // keeps them off stderr a second time.
    if !host_addressed.is_empty() {
        println!(
            "cargo::warning=codegen determinism: excluded as host addresses (expected to differ \
             across processes): {}",
            host_addressed.join(" ")
        );
    }
    if !in_process_stateful.is_empty() {
        println!(
            "cargo::warning=codegen determinism: excluded as process-stateful (expected to \
             differ across in-process generations): {}",
            in_process_stateful.join(" ")
        );
    }
    false
}

fn restore_codegen_cache(cache_dir: &std::path::Path, out_dir: &str) -> bool {
    if !CODEGEN_OUTPUTS
        .iter()
        .all(|name| cache_dir.join(name).is_file())
    {
        return false;
    }
    for name in CODEGEN_OUTPUTS {
        let src = cache_dir.join(name);
        let dst = std::path::Path::new(out_dir).join(name);
        if let Err(e) = std::fs::copy(&src, &dst) {
            eprintln!(
                "[pyre-jit-trace build.rs] warning: cache restore failed for {}: {e}",
                src.display()
            );
            return false;
        }
    }
    true
}

fn store_codegen_cache(cache_dir: &std::path::Path, out_dir: &str) -> std::io::Result<()> {
    if CODEGEN_OUTPUTS
        .iter()
        .all(|name| cache_dir.join(name).is_file())
    {
        return Ok(());
    }
    let Some(parent) = cache_dir.parent() else {
        return Ok(());
    };
    std::fs::create_dir_all(parent)?;
    let tmp_dir = parent.join(format!(
        ".{}.tmp-{}",
        cache_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("cache"),
        std::process::id()
    ));
    if tmp_dir.exists() {
        std::fs::remove_dir_all(&tmp_dir)?;
    }
    std::fs::create_dir_all(&tmp_dir)?;
    for name in CODEGEN_OUTPUTS {
        let src = std::path::Path::new(out_dir).join(name);
        let dst = tmp_dir.join(name);
        std::fs::copy(src, dst)?;
    }
    match std::fs::rename(&tmp_dir, cache_dir) {
        Ok(()) => Ok(()),
        Err(e) if cache_dir.exists() => {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            eprintln!("[pyre-jit-trace build.rs] cache already stored by another process: {e}");
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            Err(e)
        }
    }
}

fn codegen_cache_key(manifest_dir: &str, repo_root: &str, source_paths: &[String]) -> String {
    let mut h = CacheHasher::new();
    let repo_root = std::path::Path::new(repo_root);
    h.write_str(CODEGEN_CACHE_VERSION);
    for key in ["HOST", "TARGET", "PROFILE", "OPT_LEVEL"] {
        h.write_str(key);
        h.write_os(std::env::var_os(key));
    }
    let mut cargo_env: Vec<(String, String)> = std::env::vars()
        .filter(|(key, _)| {
            key.starts_with("CARGO_FEATURE_") || key.starts_with("CARGO_CFG_TARGET_")
        })
        .collect();
    cargo_env.sort();
    for (key, value) in cargo_env {
        h.write_str(&key);
        h.write_str(&value);
    }
    h.write_os(std::env::var_os("MAJIT_MIR_FRONTEND_LLBC"));
    for key in LOWERING_GATE_ENV {
        h.write_str(key);
        h.write_os(std::env::var_os(key));
    }

    // Hash the sources the prepass actually reads, not every workspace crate
    // the build-script binary happens to link. A walk of `majit/*/src` and
    // `pyre/*/src` rekeyed the cache on a `majit-gc` comment and re-ran the
    // analysis over unchanged LLBC. Interpreter/object/jit bodies reach the
    // prepass only through the artefacts `hash_llbc_inputs` hashes;
    // `fail_if_llbc_stale` refuses a source/artefact mismatch before the
    // key is consulted.
    //
    // Hash sources rather than the build-script executable's bytes. The
    // binary is not reproducible: recompiling it from unchanged sources —
    // which `cargo clean`, a fresh `CARGO_TARGET_DIR`, or a touched
    // `build.rs` all force — yields different bytes, so keying on them
    // rekeyed the cache on almost every build. `Cargo.lock` covers external
    // crate versions (`rustpython-compiler-core`, serde, bincode); the
    // compiler version string covers a toolchain that changes a type layout.
    hash_file_content(&mut h, repo_root, &repo_root.join("Cargo.lock"));
    h.write_str(&rustc_version_string());
    hash_file_content(
        &mut h,
        repo_root,
        &std::path::Path::new(manifest_dir).join("build.rs"),
    );
    // The whole `build/` directory, not `build.rs` alone. This prepass now
    // lives in `build/prepass.rs`, `#[path]`-included by that wrapper rather
    // than sitting under `src/`, so neither the line above nor the discovered
    // trees reach its bytes. Editing it recompiles the build script and
    // makes Cargo rerun it -- and then an unchanged key restores the previous
    // generated tables over the ones the rerun just wrote. Hashing the
    // directory covers a second module added beside it too.
    hash_rs_dir_content(
        &mut h,
        repo_root,
        &std::path::Path::new(manifest_dir).join("build"),
    );
    let inputs = codegen_cache::discover(repo_root, std::path::Path::new(manifest_dir));
    for rel in &inputs.src_trees {
        hash_rs_dir_content(&mut h, repo_root, &repo_root.join(rel));
    }
    for rel in inputs.extra_files.iter().chain(&inputs.manifests) {
        hash_file_content(&mut h, repo_root, &repo_root.join(rel));
    }

    // Paths only. The analyzer derives `module_path` from the filename; the
    // graph bodies come from LLBC. Hashing contents here re-ran the prepass
    // on every interpreter edit even when the artefact was unchanged.
    // Repo-relative so two checkouts of the same tree share a key.
    let mut rel_sources: Vec<String> = source_paths
        .iter()
        .map(|path| codegen_cache::repo_relative(repo_root, std::path::Path::new(path)))
        .collect();
    rel_sources.sort();
    h.write_str("source-paths");
    for path in &rel_sources {
        h.write_str(path);
    }
    hash_llbc_inputs(&mut h, repo_root);

    format!("{:016x}", h.finish())
}

/// `rustc -vV`, so a toolchain upgrade that changes codegen or a type layout
/// rekeys the cache. Falls back to a marker when the compiler cannot be run —
/// a missing string only makes the key coarser, never staler, because every
/// other input is still hashed.
fn rustc_version_string() -> String {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    std::process::Command::new(rustc)
        .arg("-vV")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
        .unwrap_or_else(|| "rustc-version-unavailable".to_string())
}

fn hash_llbc_inputs(h: &mut CacheHasher, repo_root: &std::path::Path) {
    // Hash the LLBC by a single forward pass that drops span line/col,
    // `files[].contents`, and `dest_file`. Those move when a comment is
    // edited or the checkout path changes, and hashing the raw JSON
    // re-ran the prepass over an unchanged translation. `source_text`
    // stays in the feed (the translator reads `static mut` off it).
    // Extract does not run this pass: it must not rescan the artefact.
    // The `.ullbc` set is ~660 MB; the read plus this walk is still
    // seconds, next to the tens of seconds of analysis it gates.
    if let Some(paths) = std::env::var_os("MAJIT_MIR_FRONTEND_LLBC") {
        for path in std::env::split_paths(&paths) {
            if !path.as_os_str().is_empty() {
                hash_ullbc_file(h, repo_root, &path);
            }
        }
        return;
    }
    // Same single source of truth as `emit_rerun_directives`: a crate missing
    // from this hash keys the codegen cache on a stale snapshot of it.
    for crate_name in LLBC_CRATES {
        hash_ullbc_file(
            h,
            repo_root,
            &repo_root
                .join("build")
                .join(LLBC_SUBDIR)
                .join(format!("{crate_name}.ullbc")),
        );
    }
    // Cross-target sidecars are build inputs too: they supply the target field
    // offsets `auto_discover_workspace_llbc_paths` merges in.  Without them in
    // the key, regenerating a sidecar (same Rust sources, same target — e.g.
    // after fixing the extraction flags) leaves the key unchanged, so
    // `restore_codegen_cache` would serve the descrs built from the old
    // offsets.  Empty on a native build.
    for sidecar in llbc_layout_sidecars() {
        hash_ullbc_file(
            h,
            repo_root,
            &repo_root.join("build").join(LLBC_SUBDIR).join(&sidecar),
        );
    }
}

fn hash_ullbc_file(h: &mut CacheHasher, repo_root: &std::path::Path, path: &std::path::Path) {
    h.write_str(&codegen_cache::repo_relative(repo_root, path));
    let Ok(bytes) = std::fs::read(path) else {
        h.write_str("missing");
        return;
    };
    h.write_str("semantic");
    ullbc_semantic::feed_semantic_ullbc(&bytes, &mut |chunk| h.write_bytes(chunk));
}

fn hash_rs_dir_content(h: &mut CacheHasher, repo_root: &std::path::Path, dir: &std::path::Path) {
    let mut paths = Vec::new();
    for entry in WalkDir::new(dir) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() || entry.path().extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        paths.push(entry.path().to_path_buf());
    }
    paths.sort();
    for path in paths {
        hash_file_content(h, repo_root, &path);
    }
}

fn hash_file_content(h: &mut CacheHasher, repo_root: &std::path::Path, path: &std::path::Path) {
    h.write_str(&codegen_cache::repo_relative(repo_root, path));
    let Ok(mut file) = std::fs::File::open(path) else {
        h.write_str("missing");
        return;
    };
    h.write_str("content");
    let mut buf = [0_u8; 64 * 1024];
    loop {
        match std::io::Read::read(&mut file, &mut buf) {
            Ok(0) => break,
            Ok(n) => h.write_bytes(&buf[..n]),
            Err(e) => {
                h.write_str("read-error");
                h.write_str(&e.to_string());
                break;
            }
        }
    }
}

/// Length-prefixed hashing wrapper over the std hasher.
///
/// `DefaultHasher` is fixed-key SipHash: deterministic within a given Rust
/// toolchain (no per-process seed), which is all the cache needs — a key
/// produced by one build matches the same build's stored entry. std does
/// not promise the algorithm is stable across Rust releases, so a toolchain
/// upgrade changes every key; that is fine here (a miss just regenerates,
/// and `rustc_version_string` is already in the key, so a toolchain
/// bump rekeys regardless). Inputs are length-prefixed so adjacent fields
/// cannot run together: `("ab", "c")` and `("a", "bc")` hash differently.
struct CacheHasher(std::collections::hash_map::DefaultHasher);

impl CacheHasher {
    fn new() -> Self {
        Self(std::collections::hash_map::DefaultHasher::new())
    }

    fn finish(&self) -> u64 {
        std::hash::Hasher::finish(&self.0)
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        // Fixed-width length prefix, then the payload — the prefix frames
        // the byte run so concatenation stays unambiguous.
        std::hash::Hasher::write_u64(&mut self.0, bytes.len() as u64);
        std::hash::Hasher::write(&mut self.0, bytes);
    }

    fn write_str(&mut self, value: &str) {
        self.write_bytes(value.as_bytes());
    }

    fn write_os(&mut self, value: Option<std::ffi::OsString>) {
        match value {
            Some(value) => self.write_str(&value.to_string_lossy()),
            None => self.write_str("<unset>"),
        }
    }
}

/// Collect a single `.rs` file by absolute path, appending to the same
/// vec the directory walk fills.  Used to thread
/// `pyre-jit/src/eval.rs` (the portal canonical)
/// into the analysis without including the rest of pyre-jit's JIT
/// infrastructure (codewriter, assembler, regalloc, ...).
fn collect_single_file(path: &str, paths: &mut Vec<String>) {
    match std::fs::metadata(path) {
        Ok(_) => paths.push(path.to_string()),
        Err(e) => {
            eprintln!("[pyre-jit-trace build.rs] warning: cannot read {path}: {e}");
        }
    }
}

fn emit_rerun_if_changed_recursive(dir: &str) {
    for entry in WalkDir::new(dir) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() || entry.path().extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        println!("cargo::rerun-if-changed={}", entry.path().display());
    }
}
