/// Tests that codegen produces valid wasm modules.
///
/// Most tests use wasmparser to validate emitted bytes. The terminal-decline
/// regression additionally executes the full wasm host and compares its Python
/// output with dynasm, because the old failure was a runtime pointer miscast.
///
/// One test thread on every OS. The single thread guards process-global
/// settings tests flip, such as the inline-trip helper slot and the
/// eager-merge byte cap.
#[cfg(not(target_arch = "wasm32"))]
mod serial_cpu_tests {
    extern "C" fn set_one_test_thread() {
        // SAFETY: constructor runs before `main`, single-threaded.
        unsafe { std::env::set_var("RUST_TEST_THREADS", "1") };
    }

    #[used]
    #[cfg_attr(
        any(target_os = "macos", target_os = "ios"),
        unsafe(link_section = "__DATA,__mod_init_func")
    )]
    #[cfg_attr(
        any(target_os = "linux", target_os = "android", target_os = "freebsd"),
        unsafe(link_section = ".init_array")
    )]
    #[cfg_attr(windows, unsafe(link_section = ".CRT$XCU"))]
    static SET_ONE_TEST_THREAD: extern "C" fn() = set_one_test_thread;
}

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use majit_backend_wasm::codegen;
use majit_backend_wasm::{gc_rewriter, rewrite_ops_for_gc_with};
use majit_ir::operand::Operand;
use majit_ir::{
    EffectInfo, InputArg, InputArgRc, Op, OpCode, OpRc, OpRef, RuntimeHelperKind, Type,
};
use smallvec::smallvec;
use wasmi::{Engine, Linker, Memory, MemoryType, Module, Store, Table, TableType, Val, ValType};

fn validate_wasm(bytes: &[u8]) {
    wasmparser::validate(bytes).expect("generated wasm should be valid");
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("majit/majit-backend-wasm must be below the workspace root")
        .to_path_buf()
}

/// The wasm-host module these runtime tests measure.
///
/// Only the snapshot path, never the raw `pyre_wasm.wasm` cargo output: the
/// `web` and `wasm-host` features of `pyre-wasm` build to that one filename and
/// overwrite each other, so a tree that last built `web` leaves a module there
/// which loads and runs but is not the one whose counters these tests pin.
/// `check.py` copies the wasm-host build here for exactly that reason.
/// A release runtime binary, named the way the host names an executable.
fn runtime_binary(root: &Path, name: &str) -> PathBuf {
    root.join("target/release")
        .join(format!("{name}{}", std::env::consts::EXE_SUFFIX))
}

fn wasm_host_module(root: &Path) -> PathBuf {
    root.join("target/wasm32-unknown-unknown/release/pyre_wasm.wasm-host.wasm")
}

fn run_runtime_program(
    binary: &Path,
    script: &Path,
    envs: &[(&str, &str)],
) -> std::process::Output {
    let mut command = Command::new(binary);
    command.arg(script);
    for &(key, value) in envs {
        command.env(key, value);
    }
    let output = command
        .output()
        .unwrap_or_else(|err| panic!("failed to run {}: {err}", binary.display()));
    // A child killed by a signal leaves every assertion below nothing to show:
    // both runtimes report their own failures before exiting — the wasm runner
    // keeps the run result precisely so a guest trap still prints its JIT
    // stats — so a failed status next to an empty stderr means the process
    // never reached any of that. Name the signal and the run that took it
    // here, once, instead of letting each call site blame output the child
    // never had the chance to write.
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = output.status.signal() {
            let env = envs
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join(" ");
            panic!(
                "{} died on signal {signal} running {}\nenv: {env}\nstderr:\n{}",
                binary.display(),
                script.display(),
                String::from_utf8_lossy(&output.stderr),
            );
        }
    }
    output
}

/// Assert a runtime subprocess exited 0, reporting enough to diagnose the
/// failure from a CI log alone.
///
/// `run_runtime_program` has already claimed a death by signal, so what
/// reaches here is a child that chose its own nonzero status. Which run it
/// was, what status it chose, and what it had written to stdout are all part
/// of that answer, and stderr alone carries none of them — a runner that
/// reports on stdout and exits 1 otherwise fails with an empty message.
#[track_caller]
fn assert_ran_ok(label: &str, output: &std::process::Output) {
    assert!(
        output.status.success(),
        "{label} failed: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Assert the two runtimes printed the same thing, up to the line ending
/// their hosts spell a newline with.
///
/// What is being compared is what the two backends computed. `pyre-dynasm`
/// runs on the host, where a text stream translates `\n` on the way out and
/// Windows spells the result `\r\n`; the wasm guest is its own platform and
/// spells it `\n`. Comparing the bytes would make every one of these tests
/// fail on a Windows host over the separator alone.
#[track_caller]
fn assert_same_stdout(label: &str, wasm: &std::process::Output, dynasm: &std::process::Output) {
    fn lines(stdout: &[u8]) -> String {
        String::from_utf8_lossy(stdout).replace("\r\n", "\n")
    }
    assert_eq!(
        lines(&wasm.stdout),
        lines(&dynasm.stdout),
        "{label} output diverged from dynasm:\n{}",
        String::from_utf8_lossy(&wasm.stderr),
    );
}

fn stat_value(stderr: &str, name: &str) -> u64 {
    stderr
        .split_whitespace()
        .find_map(|field| field.strip_prefix(&format!("{name}=")))
        .unwrap_or_else(|| panic!("missing {name}= in wasm JIT stats:\n{stderr}"))
        .parse()
        .unwrap_or_else(|err| panic!("invalid {name}= in wasm JIT stats: {err}\n{stderr}"))
}

/// CALL_ASSEMBLER must not refill a frame on the bump path. The inline
/// bump leaves `jf_gcmap` unset. Expected `memory.fill`s are the entry
/// home clear of marked slots, leftover `NEW_ARRAY_CLEAR` ZERO_ARRAY,
/// rewriter `ZeroArray` payload zeros of a freshly allocated Python
/// array, and the CA caller nulling the callee home range
/// (`home_slots * SLOT_SIZE` from the dispatch snapshot) before
/// publishing that snapshot's gcmap.
#[track_caller]
fn assert_no_call_assembler_frame_fill(stderr: &str) {
    let home_base = format!(
        "i32.load offset={}",
        majit_backend_wasm::failguard::WASM_CA_TARGET_HOME_SLOT_BASE_OFS
    );
    let home_slots = format!(
        "i32.load offset={}",
        majit_backend_wasm::failguard::WASM_CA_TARGET_HOME_SLOTS_OFS
    );
    let lines: Vec<_> = stderr.lines().map(str::trim).collect();
    for (index, line) in lines.iter().enumerate() {
        if *line != "memory.fill" {
            continue;
        }
        let is_entry_home_clear = index >= 5
            && lines[index - 5] == "local.get 0"
            && lines[index - 4].starts_with("i32.const ")
            && lines[index - 3] == "i32.add"
            && lines[index - 2] == "i32.const 0"
            && lines[index - 1].starts_with("i32.const ");
        // `emit_zero_bytes(base, HDR, n)`: payload after the header word.
        let is_headered_payload = index >= 5
            && lines[index - 5].starts_with("local.get ")
            && lines[index - 4] == "i32.const 8"
            && lines[index - 3] == "i32.add"
            && lines[index - 2] == "i32.const 0"
            && lines[index - 1].starts_with("i32.const ");
        // `emit_zero_bytes(base, 0, n)`: headerless / whole-object zero.
        let is_object_zero = index >= 3
            && lines[index - 3].starts_with("local.get ")
            && lines[index - 2] == "i32.const 0"
            && lines[index - 1].starts_with("i32.const ");
        // `emit_clear_reserved_homes`: size = home_slots << 3, skipped when
        // zero; dest = frame + FIRST_ITEM_OFFSET + home_slot_base (the
        // guest's wasm32 offset, not this host's). The three `local.get` of
        // the dispatch-entry pointer must name the same local, the loads must
        // be the snapshot home fields, and the fill length must be the local
        // the size was teed into.
        let ca_target = lines[index.saturating_sub(16)]
            .strip_prefix("local.get ")
            .filter(|s| !s.is_empty());
        let home_size = lines[index.saturating_sub(12)]
            .strip_prefix("local.tee ")
            .filter(|s| !s.is_empty());
        let is_ca_callee_home_null = index >= 16
            && ca_target.is_some()
            && lines[index - 15] == home_slots
            && lines[index - 14] == "i32.const 3"
            && lines[index - 13] == "i32.shl"
            && home_size.is_some()
            && lines[index - 11] == "i32.eqz"
            && lines[index - 10].starts_with("if")
            && lines[index - 9] == "else"
            && lines[index - 8].starts_with("local.get ")
            && lines[index - 7].starts_with("i32.const ")
            && lines[index - 6] == "i32.add"
            && lines[index - 5].strip_prefix("local.get ") == ca_target
            && lines[index - 4] == home_base
            && lines[index - 3] == "i32.add"
            && lines[index - 2] == "i32.const 0"
            && lines[index - 1].strip_prefix("local.get ") == home_size;
        // `OpCode::ZeroArray` after `emit_pending_zeros` (scales rewritten
        // to 1, start/size in bytes): dest = wrap(base) + start +
        // `ArrayDescr.base_size()`. PeepSink folds `wrap(const)` to
        // `i32.const` and drops a 0 add, so a trimmed start is two adds
        // and a 0 start (or 0 `base_size`) is one.
        let is_zero_array = index >= 8
            && lines[index - 8].starts_with("local.get ")
            && lines[index - 7] == "i32.wrap_i64"
            && lines[index - 6].starts_with("i32.const ")
            && lines[index - 5] == "i32.add"
            && lines[index - 4].starts_with("i32.const ")
            && lines[index - 3] == "i32.add"
            && lines[index - 2] == "i32.const 0"
            && lines[index - 1].starts_with("i32.const ");
        let is_zero_array_zero_start = index >= 6
            && lines[index - 6].starts_with("local.get ")
            && lines[index - 5] == "i32.wrap_i64"
            && lines[index - 4].starts_with("i32.const ")
            && lines[index - 3] == "i32.add"
            && lines[index - 2] == "i32.const 0"
            && lines[index - 1].starts_with("i32.const ");
        assert!(
            is_entry_home_clear
                || is_headered_payload
                || is_object_zero
                || is_ca_callee_home_null
                || is_zero_array
                || is_zero_array_zero_start,
            "recursive CA filled a nursery frame on the bump path:\n{stderr}"
        );
    }
}

#[test]
fn zero_array_payload_fill_is_not_a_frame_refill() {
    assert_no_call_assembler_frame_fill(
        "\
local.get 37
i32.wrap_i64
i32.const 4
i32.add
i32.const 4
i32.add
i32.const 0
i32.const 20
memory.fill
",
    );
}

#[test]
fn zero_array_zero_start_payload_fill_is_not_a_frame_refill() {
    assert_no_call_assembler_frame_fill(
        "\
local.get 37
i32.wrap_i64
i32.const 4
i32.add
i32.const 0
i32.const 24
memory.fill
",
    );
}

#[test]
fn call_assembler_home_range_fill_is_not_a_frame_refill() {
    assert_no_call_assembler_frame_fill(
        "\
local.get 267
i32.load offset=28
i32.const 3
i32.shl
local.tee 130
i32.eqz
if ;; label = @2
else
local.get 265
i32.const 32
i32.add
local.get 267
i32.load offset=24
i32.add
i32.const 0
local.get 130
memory.fill
",
    );
}

#[test]
#[should_panic(expected = "recursive CA filled a nursery frame")]
fn call_assembler_unrelated_dynamic_fill_is_rejected() {
    assert_no_call_assembler_frame_fill(
        "\
i32.const 100
i32.const 0
i32.const 16
i32.mul
memory.fill
",
    );
}

#[test]
#[should_panic(expected = "recursive CA filled a nursery frame")]
fn call_assembler_mismatched_target_local_fill_is_rejected() {
    assert_no_call_assembler_frame_fill(
        "\
local.get 267
i32.load offset=28
i32.const 3
i32.shl
local.tee 130
i32.eqz
if ;; label = @2
else
local.get 265
i32.const 32
i32.add
local.get 268
i32.load offset=24
i32.add
i32.const 0
local.get 130
memory.fill
",
    );
}

#[test]
#[should_panic(expected = "recursive CA filled a nursery frame")]
fn call_assembler_wrong_home_offset_fill_is_rejected() {
    assert_no_call_assembler_frame_fill(
        "\
local.get 267
i32.load offset=28
i32.const 3
i32.shl
local.tee 130
i32.eqz
if ;; label = @2
else
local.get 265
i32.const 32
i32.add
local.get 267
i32.load offset=8
i32.add
i32.const 0
local.get 130
memory.fill
",
    );
}

#[test]
#[ignore = "runtime integration test: needs the release pyre-dynasm, pyre-wasm-runner, and wasm-host module; \
            run via `cargo test -- --ignored` in the check.py job, which builds them"]
fn global_reassign_retraces_non_last_label_backedge_at_runtime() {
    let root = workspace_root();
    let dynasm = runtime_binary(&root, "pyre-dynasm");
    let wasm_runner = runtime_binary(&root, "pyre-wasm-runner");
    let wasm_module = wasm_host_module(&root);

    for artifact in [&dynasm, &wasm_runner, &wasm_module] {
        assert!(
            artifact.exists(),
            "runtime global-reassign regression needs {}; build the requested dynasm and wasm-host artifacts first",
            artifact.display()
        );
    }

    let module = wasm_module.to_str().expect("workspace paths must be UTF-8");
    {
        let bench = "global_reassign.py";
        let script = root.join("pyre/bench/synth").join(bench);
        let dynasm_run = run_runtime_program(&dynasm, &script, &[]);
        assert_ran_ok(&format!("dynasm {bench}"), &dynasm_run);
        let wasm_run = run_runtime_program(
            &wasm_runner,
            &script,
            &[
                ("PYRE_WASM_MODULE", module),
                ("PYRE_WASM_ENGINE", "wasmtime"),
                ("PYRE_WASM_JIT_STATS", "1"),
            ],
        );
        let stderr = String::from_utf8_lossy(&wasm_run.stderr);
        assert_ran_ok(&format!("wasm {bench}"), &wasm_run);
        assert_same_stdout(&format!("wasm {bench}"), &wasm_run, &dynasm_run);
        assert!(
            stat_value(&stderr, "compiles") > 1,
            "{bench} did not recompile after its global invalidation:\n{stderr}"
        );
        assert!(
            stat_value(&stderr, "gc_majors") < 10,
            "{bench} fell back to the allocating interpreter loop:\n{stderr}"
        );
    }
}

#[test]
#[ignore = "runtime integration test: needs the release pyre-dynasm, pyre-wasm-runner, and wasm-host module; \
            run via `cargo test -- --ignored` in the check.py job, which builds them"]
fn raise_catch_clear_root_does_not_cross_the_host_per_exception() {
    let root = workspace_root();
    let dynasm = runtime_binary(&root, "pyre-dynasm");
    let wasm_runner = runtime_binary(&root, "pyre-wasm-runner");
    let wasm_module = wasm_host_module(&root);
    let script = root.join("pyre/bench/raise_catch_loop.py");

    for artifact in [&dynasm, &wasm_runner, &wasm_module] {
        assert!(
            artifact.exists(),
            "runtime raise/catch regression needs {}; build the requested dynasm and wasm-host artifacts first",
            artifact.display()
        );
    }

    let dynasm_run = run_runtime_program(&dynasm, &script, &[]);
    assert_ran_ok("dynasm raise/catch", &dynasm_run);
    let module = wasm_module.to_str().expect("workspace paths must be UTF-8");
    let wasm_run = run_runtime_program(
        &wasm_runner,
        &script,
        &[
            ("PYRE_WASM_MODULE", module),
            ("PYRE_WASM_ENGINE", "wasmtime"),
            ("PYRE_WASM_JIT_STATS", "1"),
        ],
    );
    let stderr = String::from_utf8_lossy(&wasm_run.stderr);
    assert_ran_ok("wasm raise/catch", &wasm_run);
    assert_same_stdout("wasm raise/catch", &wasm_run, &dynasm_run);
    assert!(
        stat_value(&stderr, "jit_calls") < 100_000,
        "caught-exception root clearing crossed the host trampoline per iteration:\n{stderr}"
    );
}

#[test]
#[ignore = "runtime integration test: needs the release pyre-dynasm, pyre-wasm-runner, and wasm-host module; \
            run via `cargo test -- --ignored` in the check.py job, which builds them"]
fn recursive_call_assembler_does_not_refill_zeroed_nursery_frames() {
    let root = workspace_root();
    let dynasm = runtime_binary(&root, "pyre-dynasm");
    let wasm_runner = runtime_binary(&root, "pyre-wasm-runner");
    let wasm_module = wasm_host_module(&root);
    let script = root.join("pyre/bench/fib_recursive.py");

    for artifact in [&dynasm, &wasm_runner, &wasm_module] {
        assert!(
            artifact.exists(),
            "runtime recursive-CA regression needs {}; build the requested artifacts first",
            artifact.display()
        );
    }
    let dynasm_run = run_runtime_program(&dynasm, &script, &[]);
    assert_ran_ok("dynasm recursive fib", &dynasm_run);
    let module = wasm_module.to_str().expect("workspace paths must be UTF-8");
    let wasm_run = run_runtime_program(
        &wasm_runner,
        &script,
        &[
            ("PYRE_WASM_MODULE", module),
            ("PYRE_WASM_ENGINE", "wasmtime"),
            ("PYRE_WASM_JIT_STATS", "1"),
            ("PYRE_WASM_DUMP_ALL_TRACES", "1"),
            ("PYRE_WASM_NO_CACHE", "1"),
        ],
    );
    let stderr = String::from_utf8_lossy(&wasm_run.stderr);
    assert_ran_ok("wasm recursive fib", &wasm_run);
    assert_same_stdout("wasm recursive fib", &wasm_run, &dynasm_run);
    // `compiles` is the host's module-compile tally, one per loop and one per
    // bridge, and `BRIDGE_OK` counts the bridges the backend accepted — the
    // same event `bridges_compiled` counts, since both are bumped only on the
    // `Ok` side of `compile_bridge`. So both follow from the committed
    // `pyre/bench/fib_recursive.jitstats`: `loops_compiled=1` +
    // `bridges_compiled=7`. Re-record these two alongside that baseline.
    assert_eq!(stat_value(&stderr, "compiles"), 8);
    assert_eq!(stat_value(&stderr, "BRIDGE_OK"), 7);
    assert_no_call_assembler_frame_fill(&stderr);
}

#[test]
#[ignore = "runtime integration test: needs the release pyre-dynasm, pyre-wasm-runner, and wasm-host module; \
            run via `cargo test -- --ignored` in the check.py job, which builds them"]
fn fannkuch_blackhole_helpers_do_not_reflect_through_the_host() {
    let root = workspace_root();
    let dynasm = runtime_binary(&root, "pyre-dynasm");
    let wasm_runner = runtime_binary(&root, "pyre-wasm-runner");
    let wasm_module = wasm_host_module(&root);
    let script = root.join("pyre/bench/fannkuch.py");
    for artifact in [&dynasm, &wasm_runner, &wasm_module] {
        assert!(
            artifact.exists(),
            "runtime fannkuch regression needs {}; build the requested artifacts first",
            artifact.display()
        );
    }

    let dynasm_run = run_runtime_program(&dynasm, &script, &[]);
    assert_ran_ok("dynasm fannkuch", &dynasm_run);
    let module = wasm_module.to_str().expect("workspace paths must be UTF-8");
    let wasm_run = run_runtime_program(
        &wasm_runner,
        &script,
        &[
            ("PYRE_WASM_MODULE", module),
            ("PYRE_WASM_ENGINE", "wasmtime"),
            ("PYRE_WASM_JIT_STATS", "1"),
        ],
    );
    let stderr = String::from_utf8_lossy(&wasm_run.stderr);
    assert_ran_ok("wasm fannkuch", &wasm_run);
    assert_same_stdout("wasm fannkuch", &wasm_run, &dynasm_run);
    // `compiles` is the host's module-compile tally: one per loop, one per
    // accepted bridge, and one more each time a region merged into its owner
    // re-emits it. A byte-identical loop hits `compile_module_cached` and
    // increments `compile_cache_hits` instead, while `cl_ok` still counts
    // it. Loops follow `cl_ok`; bridges follow `BRIDGE_OK` (entered minus
    // declines), not the guest `bridges_compiled` snapshot. `reemit_ok` is
    // the merge count. The identity is compiles + cache hits.
    let loops = stat_value(&stderr, "cl_ok");
    let bridges = stat_value(&stderr, "BRIDGE_OK");
    let reemits = stat_value(&stderr, "reemit_ok");
    let compiles = stat_value(&stderr, "compiles");
    let cache_hits = stat_value(&stderr, "compile_cache_hits");
    assert!(
        reemits <= bridges,
        "more owner re-emissions ({reemits}) than bridges to have merged:\n{stderr}"
    );
    // A merge that re-emits an already-cached owner increments `reemit_ok`
    // without another host compile. ubuntu-24.04 read compiles=30 against
    // loops+bridges+1 with one cache hit.
    assert_eq!(
        compiles + cache_hits,
        loops + bridges + reemits,
        "compiles={compiles} cache_hits={cache_hits} loops={loops} \
         bridges={bridges} reemits={reemits}\n{stderr}"
    );
    assert!(
        stat_value(&stderr, "jit_calls") < 100,
        "uniform-i64 blackhole helpers still reflected through the host:\n{stderr}"
    );
}

#[test]
#[ignore = "runtime integration test: needs the release pyre-dynasm, pyre-wasm-runner, and wasm-host module; \
            run via `cargo test -- --ignored` in the check.py job, which builds them"]
fn terminal_declined_call_assembler_matches_dynasm_at_runtime() {
    let root = workspace_root();
    let dynasm = runtime_binary(&root, "pyre-dynasm");
    let wasm_runner = runtime_binary(&root, "pyre-wasm-runner");
    let wasm_module = wasm_host_module(&root);
    let script = root.join("pyre/oracle/ca_terminal_decline.py");

    for artifact in [&dynasm, &wasm_runner, &wasm_module] {
        assert!(
            artifact.exists(),
            "runtime CA regression needs {}; build the requested dynasm and wasm-host artifacts first",
            artifact.display()
        );
    }

    let dynasm_run = run_runtime_program(&dynasm, &script, &[]);
    assert_ran_ok("dynasm terminal-decline", &dynasm_run);
    let module = wasm_module.to_str().expect("workspace paths must be UTF-8");
    let wasm_run = run_runtime_program(
        &wasm_runner,
        &script,
        &[
            ("PYRE_WASM_MODULE", module),
            ("PYRE_WASM_ENGINE", "wasmtime"),
            ("PYRE_WASM_JIT_STATS", "1"),
            ("PYRE_WASM_FORCE_CA_TERMINAL_DECLINE", "1"),
        ],
    );
    let stderr = String::from_utf8_lossy(&wasm_run.stderr);
    assert_ran_ok("wasm terminal-decline", &wasm_run);
    assert_same_stdout("forced terminal-decline wasm", &wasm_run, &dynasm_run);
    assert!(
        stderr.contains("accepted_ca=") && !stderr.contains("accepted_ca=0"),
        "fixture did not compile its outer CALL_ASSEMBLER trace:\n{stderr}"
    );
    assert!(
        stderr.contains("forced_ca_terminal_decline=1"),
        "terminal-decline hook did not run after CA admission:\n{stderr}"
    );
}

#[test]
fn compile_bridge_terminal_decline_invalidates_ca_callers() {
    use majit_backend::Backend;
    use majit_backend_wasm::failguard::CompiledWasmLoop;

    let _serialized = HOST_COMPILE_LOCK.lock();
    let mut backend = majit_backend_wasm::WasmBackend::new();
    let token = std::sync::Arc::new(majit_backend::JitCellToken::new(1));
    let clt = std::sync::Arc::new(majit_backend::CompiledLoopToken::new(1));
    clt.set_loop_token_wref(std::sync::Arc::downgrade(&token));
    token.set_compiled_loop_token(Some(clt));
    let label_descr = majit_ir::make_loop_target_descr(70, false);
    backend
        .compile_loop(&host_loop_inputargs(), &host_loop_ops(&label_descr), &token)
        .expect("owner loop compiles");

    let caller_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let loop_ = token
            .compiled
            .get()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
            .expect("compiled wasm loop");
        loop_
            .ca_callers
            .borrow_mut()
            .push(std::sync::Arc::downgrade(&caller_flag));
        assert!(!loop_.ca_terminal_declined.get());
    }

    // One bridge input against two fail-arg types: `frame_entry_reads_live_positions`
    // declines this shape, which is a structural `Unsupported`.
    let fail_descr = HostFailDescr {
        fail_index: 0,
        arg_types: vec![Type::Int, Type::Int],
    };
    let err = backend
        .compile_bridge(
            &fail_descr,
            &[InputArg::from_type_rc(Type::Int, 40)],
            &host_bridge_ops(&label_descr),
            &token,
            None,
        )
        .expect_err("arity mismatch is a terminal decline");
    assert!(
        matches!(err, majit_backend::BackendError::Unsupported(_)),
        "got {err}"
    );
    assert!(
        caller_flag.load(std::sync::atomic::Ordering::Acquire),
        "CALL_ASSEMBLER callers of a terminally declined callee must be invalidated"
    );
    let loop_ = token
        .compiled
        .get()
        .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
        .expect("compiled wasm loop");
    assert!(loop_.ca_terminal_declined.get());
}

#[test]
#[ignore = "runtime integration test: needs the release pyre-dynasm, pyre-wasm-runner, and wasm-host module; \
            run via `cargo test -- --ignored` in the check.py job, which builds them"]
fn wasm_outlier_bridges_stay_compiled_at_runtime() {
    let root = workspace_root();
    let dynasm = runtime_binary(&root, "pyre-dynasm");
    let wasm_runner = runtime_binary(&root, "pyre-wasm-runner");
    let wasm_module = wasm_host_module(&root);

    for artifact in [&dynasm, &wasm_runner, &wasm_module] {
        assert!(
            artifact.exists(),
            "runtime outlier regression needs {}; build the requested dynasm and wasm-host artifacts first",
            artifact.display()
        );
    }

    let module = wasm_module.to_str().expect("workspace paths must be UTF-8");
    // A region that is no longer declined reaches compiled steady state through
    // one of two counters: an out-of-line bridge of its own (`BRIDGE_OK`), or,
    // when it closes back onto its owner's loop, a merge into the owner
    // (`inline_ok`). Inlining is the default and takes `exception_oserror_fields`,
    // so only the pair is a stable statement of "not declined".
    for (bench, compiled_counters) in [
        (
            "exception_oserror_fields.py",
            &["BRIDGE_OK", "inline_ok"][..],
        ),
        ("generator_tree_recursion.py", &["accepted_ca"][..]),
    ] {
        let script = root.join("pyre/bench/synth").join(bench);
        let dynasm_run = run_runtime_program(&dynasm, &script, &[]);
        assert_ran_ok(&format!("dynasm {bench}"), &dynasm_run);
        let wasm_run = run_runtime_program(
            &wasm_runner,
            &script,
            &[
                ("PYRE_WASM_MODULE", module),
                ("PYRE_WASM_ENGINE", "wasmtime"),
                ("PYRE_WASM_JIT_STATS", "1"),
            ],
        );
        let stderr = String::from_utf8_lossy(&wasm_run.stderr);
        assert_ran_ok(&format!("wasm {bench}"), &wasm_run);
        assert_same_stdout(&format!("wasm {bench}"), &wasm_run, &dynasm_run);
        let compiled: u64 = compiled_counters
            .iter()
            .map(|counter| stat_value(&stderr, counter))
            .sum();
        assert!(
            compiled > 0,
            "{bench} did not compile its formerly-declined bridge \
             (none of {compiled_counters:?} moved):\n{stderr}"
        );
        assert_eq!(
            stat_value(&stderr, "ml_unsafe_label"),
            0,
            "{bench} declined a LABEL resume:\n{stderr}"
        );
        assert_eq!(
            stat_value(&stderr, "decl_callasm"),
            0,
            "{bench} declined a CALL_ASSEMBLER bridge:\n{stderr}"
        );
    }
}

/// Production `compile_loop` / `compile_bridge` rewrite. Tests without an
/// installed collector would otherwise get the Boehm fallback
/// (`malloc_zero_filled=true`, no write-barrier descr); wasm production is
/// IncrementalMiniMark. `nursery` supplies the same addresses
/// `gc_rewriter()` would read off the live collector.
/// rewrite.py `get_box_replacement`: SameAsR/I is a forwarding edge.
/// rewrite.rs `resolve` only follows `record_result_mapping`, so fold
/// those aliases before the production rewrite sees the stores.
fn fold_same_as_replacements(ops: Vec<Op>) -> Vec<Op> {
    let mut map = std::collections::HashMap::<OpRef, OpRef>::new();
    let resolve = |r: OpRef, map: &std::collections::HashMap<OpRef, OpRef>| {
        let mut cur = r;
        for _ in 0..map.len().saturating_add(1) {
            match map.get(&cur).copied() {
                Some(next) if next != cur => cur = next,
                _ => break,
            }
        }
        cur
    };
    for op in &ops {
        if matches!(op.opcode, OpCode::SameAsI | OpCode::SameAsR) {
            let result = op.pos().get();
            if result != OpRef::NONE && !result.is_constant() {
                map.insert(result, resolve(op.arg(0).to_opref(), &map));
            }
        }
    }
    if map.is_empty() {
        return ops;
    }
    for op in &ops {
        for i in 0..op.num_args() {
            let r = op.arg(i).to_opref();
            let resolved = resolve(r, &map);
            if resolved != r {
                op.setarg(i, majit_ir::forwarding::bound_operand_from_opref(resolved));
            }
        }
    }
    ops
}

fn rewrite_ops_for_gc(
    ops: Vec<Op>,
    constants: &indexmap::IndexMap<u32, i64>,
    nursery: Option<&codegen::NurseryAllocParams>,
    alloc: &codegen::AllocHelpers,
    wb: &codegen::WriteBarrierHelpers,
) -> (Vec<Op>, indexmap::IndexMap<u32, i64>) {
    let ops = fold_same_as_replacements(ops);
    let mut rewriter = gc_rewriter();
    rewriter.malloc_zero_filled = false;
    let mut wb_descr = rewriter
        .wb_descr
        .take()
        .unwrap_or_else(majit_gc::WriteBarrierDescr::for_current_gc);
    if wb.cards_set == 0 {
        wb_descr.jit_wb_cards_set = 0;
        wb_descr.jit_wb_cards_set_singlebyte = 0;
    }
    rewriter.wb_descr = Some(wb_descr);
    if alloc.new_fn_ptr != 0 {
        rewriter.malloc_big_fixedsize_fn = alloc.new_fn_ptr;
    }
    if alloc.new_array_fn_ptr != 0 {
        rewriter.malloc_array_fn = alloc.new_array_fn_ptr;
        rewriter.malloc_array_nonstandard_fn = alloc.new_array_fn_ptr;
    }
    if let Some(na) = nursery {
        rewriter.nursery_free_addr = na.free_addr as usize;
        rewriter.nursery_top_addr = na.top_addr as usize;
        rewriter.max_nursery_size = na.large_threshold;
    } else {
        // No nursery on the module inputs: do not inherit a leftover
        // process-wide MiniMark size from earlier lib tests.
        rewriter.max_nursery_size = 0;
        rewriter.nursery_free_addr = 0;
        rewriter.nursery_top_addr = 0;
    }
    let (ops, constants, _table) = rewrite_ops_for_gc_with(&rewriter, ops, constants);
    (ops, constants)
}

fn ops_need_gc_rewrite(ops: &[Op]) -> bool {
    ops.iter().any(|op| {
        matches!(
            op.opcode,
            OpCode::New
                | OpCode::NewWithVtable
                | OpCode::NewArray
                | OpCode::NewArrayClear
                | OpCode::SetfieldGc
                | OpCode::SetarrayitemGc
        )
    })
}

fn rewrite_module_inputs(mut inputs: codegen::ModuleBuildInputs) -> codegen::ModuleBuildInputs {
    if ops_need_gc_rewrite(&inputs.ops)
        || inputs
            .inlined_bridges
            .iter()
            .any(|region| ops_need_gc_rewrite(&region.ops))
    {
        let nursery = inputs.nursery.clone();
        let (ops, constants) = rewrite_ops_for_gc(
            inputs.ops,
            &inputs.constants,
            nursery.as_ref(),
            &inputs.alloc,
            &inputs.wb,
        );
        inputs.ops = ops;
        inputs.constants = constants;
        for region in &mut inputs.inlined_bridges {
            let (ops, constants) = rewrite_ops_for_gc(
                std::mem::take(&mut region.ops),
                &region.constants,
                nursery.as_ref(),
                &inputs.alloc,
                &inputs.wb,
            );
            region.ops = ops;
            region.constants = constants;
        }
        let value_slots = codegen::frame_value_slots(&inputs.inputargs, &inputs.ops)
            .max(inputs.frame.value_slots);
        let homes = codegen::count_ref_homes(&inputs.inputargs, &inputs.ops)
            .max(inputs.frame.ordinary_home_slots());
        let label_refs = inputs.frame.label_ref_slots;
        inputs.frame = codegen::FrameGeometry::compact(value_slots, homes + label_refs, label_refs);
    }
    inputs
}

fn make_op(opcode: OpCode, args: &[OpRef], pos: OpRef) -> Op {
    let bx: Vec<Operand> = args.iter().map(|a| rb(*a)).collect();
    let op = Op::new(opcode, &bx);
    op.pos().set(pos);
    op
}

use majit_ir::forwarding::bound_operand_from_opref as rb;

fn rewrite_frontend_ops(
    inputargs: &[InputArgRc],
    ops: Vec<Op>,
) -> (Vec<Op>, indexmap::IndexMap<u32, i64>) {
    rewrite_frontend_ops_with_gc(inputargs, ops, majit_gc::collector::MiniMarkGC::new())
}

/// Run the same normalize + GC rewrite `compile_loop` runs, against `gc` as
/// the active collector, so the rewriter reads that collector's nursery
/// bounds and write-barrier descr.
fn rewrite_frontend_ops_with_gc(
    inputargs: &[InputArgRc],
    mut ops: Vec<Op>,
    mut gc: majit_gc::collector::MiniMarkGC,
) -> (Vec<Op>, indexmap::IndexMap<u32, i64>) {
    let num_inputs = inputargs.len() as u32;
    for (op_idx, op) in ops.iter().enumerate() {
        if op.result_type() != Type::Void && op.pos().get().is_none() {
            op.pos().set(OpRef::op_typed(
                num_inputs + op_idx as u32,
                op.result_type(),
            ));
        }
    }
    codegen::materialize_unbound_label_args(inputargs, &mut ops);
    let mut backend = majit_backend_wasm::WasmBackend::new();
    gc.register_type(majit_gc::TypeInfo::simple(16));
    backend.set_gc_allocator(Box::new(gc));
    let (ops, constants, _table) =
        majit_backend_wasm::rewrite_ops_for_gc(ops, &indexmap::IndexMap::new());
    (ops, constants)
}

fn make_guard(opcode: OpCode, args: &[OpRef], fail_args: &[OpRef]) -> Op {
    let bx: Vec<Operand> = args.iter().map(|a| rb(*a)).collect();
    let op = Op::new(opcode, &bx);
    op.setfailargs(fail_args.iter().copied().map(rb).collect());
    op
}

/// Calls `codegen::build_wasm_module` with the fixed test defaults (no
/// classptr map, no allocator/nursery, zero chaining slots, CA off) and
/// returns just the emitted bytes and guard exits — the only outputs the
/// tests assert on. `vtable_offset` and `gc_info` stay explicit because a
/// few tests vary them.
fn build_module(
    inputargs: &[InputArgRc],
    ops: &[Op],
    constants: &indexmap::IndexMap<u32, i64>,
    vtable_offset: Option<usize>,
    gc_info: &codegen::GuardGcTypeInfo,
) -> (Vec<u8>, Vec<codegen::GuardExit>) {
    build_module_with_frame(
        inputargs,
        ops,
        constants,
        vtable_offset,
        gc_info,
        codegen::FrameGeometry::fixed(),
    )
}

fn build_module_with_frame(
    inputargs: &[InputArgRc],
    ops: &[Op],
    constants: &indexmap::IndexMap<u32, i64>,
    vtable_offset: Option<usize>,
    gc_info: &codegen::GuardGcTypeInfo,
    frame: codegen::FrameGeometry,
) -> (Vec<u8>, Vec<codegen::GuardExit>) {
    build_module_with_ca(
        inputargs,
        ops,
        constants,
        vtable_offset,
        gc_info,
        frame,
        codegen::CaParams::default(),
    )
}

#[allow(clippy::too_many_arguments)]
fn build_module_with_ca(
    inputargs: &[InputArgRc],
    ops: &[Op],
    constants: &indexmap::IndexMap<u32, i64>,
    vtable_offset: Option<usize>,
    gc_info: &codegen::GuardGcTypeInfo,
    frame: codegen::FrameGeometry,
    ca: codegen::CaParams,
) -> (Vec<u8>, Vec<codegen::GuardExit>) {
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants: constants.clone(),
        vtable_offset,
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: gc_info.clone(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame,
        ca,
    };
    let (bytes, guards, _, _) = codegen::build_wasm_module(&rewrite_module_inputs(inputs))
        .expect("wasm codegen should succeed");
    (bytes, guards)
}

/// `build_module` with the most common variant: entry vtable_offset `Some(0)`
/// and a default (disabled) `GuardGcTypeInfo`.
fn build_module_default(
    inputargs: &[InputArgRc],
    ops: &[Op],
    constants: &indexmap::IndexMap<u32, i64>,
) -> (Vec<u8>, Vec<codegen::GuardExit>) {
    build_module(
        inputargs,
        ops,
        constants,
        Some(0),
        &codegen::GuardGcTypeInfo::default(),
    )
}

/// Locals of the module's FIRST code entry.
///
/// A trace that emits a label-parameter entry puts the narrow shim first and
/// the real body second, and the shim declares no locals. Use this only for
/// shapes `has_label_param_entry` rejects, or it reports the shim's zero.
fn emitted_local_count(bytes: &[u8]) -> u32 {
    wasmparser::Parser::new(0)
        .parse_all(bytes)
        .find_map(|payload| match payload.unwrap() {
            wasmparser::Payload::CodeSectionEntry(body) => Some(
                body.get_locals_reader()
                    .unwrap()
                    .into_iter()
                    .map(|local| local.unwrap().0)
                    .sum(),
            ),
            _ => None,
        })
        .expect("generated module must contain its trace function")
}

/// `(type arities, defined-function type indices, exported name -> func index)`.
///
/// Each type is reduced to `(params, results)` because that is all the entry
/// shape assertions need, and it keeps them readable next to the wasm text.
fn module_shape(bytes: &[u8]) -> (Vec<(usize, usize)>, Vec<u32>, HashMap<String, u32>) {
    let mut types = Vec::new();
    let mut functions = Vec::new();
    let mut exports = HashMap::new();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        match payload.unwrap() {
            wasmparser::Payload::TypeSection(reader) => {
                for group in reader {
                    for ty in group.unwrap().into_types() {
                        let func = ty.unwrap_func();
                        types.push((func.params().len(), func.results().len()));
                    }
                }
            }
            wasmparser::Payload::FunctionSection(reader) => {
                for idx in reader {
                    functions.push(idx.unwrap());
                }
            }
            wasmparser::Payload::ExportSection(reader) => {
                for export in reader {
                    let export = export.unwrap();
                    if export.kind == wasmparser::ExternalKind::Func {
                        exports.insert(export.name.to_string(), export.index);
                    }
                }
            }
            _ => {}
        }
    }
    (types, functions, exports)
}

#[test]
fn sparse_value_ids_declare_only_addressable_value_locals() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            OpRef::int_op(2),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(40),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(40), OpRef::const_int(1)],
            OpRef::int_op(900),
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(900))]),
    ];

    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    assert_eq!(
        emitted_local_count(&bytes),
        inputargs.len() as u32 + 3 + 6 + 2,
        "two input values and three sparse ids, plus fixed i64 and i32 locals"
    );
}

#[test]
fn same_as_reuses_its_source_local() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![
        make_op(
            OpCode::SameAsI,
            &[OpRef::input_arg_int(0)],
            OpRef::int_op(1),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))]),
    ];

    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    assert_eq!(
        emitted_local_count(&bytes),
        inputargs.len() as u32 + 1 + 6 + 2,
        "SameAsI shares input 0's local; only IntAdd allocates a result local"
    );
}

#[test]
fn same_as_does_not_alias_a_mutable_label_local() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![
        Op::new(OpCode::Label, &[rb(OpRef::input_arg_int(0))]),
        make_op(
            OpCode::SameAsI,
            &[OpRef::input_arg_int(0)],
            OpRef::int_op(1),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(2))]),
    ];

    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    assert_eq!(
        emitted_local_count(&bytes),
        inputargs.len() as u32 + 2 + 6 + 2,
        "SameAs must not share the phi local that JUMP rebinds"
    );
}

#[test]
fn same_as_before_label_defines_a_fresh_label_arg() {
    // Peeled header: preamble produces v1, SameAs copies it into the
    // LABEL-arg box, fall-through then reads that box. Without the
    // SameAs the backend declines (see the next test).
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        make_op(OpCode::SameAsI, &[OpRef::int_op(1)], OpRef::int_op(100)),
        Op::new(OpCode::Label, &[rb(OpRef::int_op(100))]),
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(100), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(2))]),
    ];

    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
}

#[test]
fn unbound_read_without_a_producer_declines() {
    // A LABEL arg with no producer is a peeled-header live-in and is
    // defined at the LABEL. A Finish that reads a never-written box
    // is not; that local would stay the zero wasm initializes it to.
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(100))]),
    ];

    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops,
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    };
    let error = match codegen::build_wasm_module(&inputs) {
        Ok(_) => panic!("a read with no producer and no pool entry must decline"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("read with no producing op and no"),
        "unexpected decline: {error}"
    );
}

#[test]
fn unbound_pool_float_operand_declares_an_f64_local() {
    let folded_float = OpRef::float_op(7);
    let ops = vec![Op::new(OpCode::Finish, &[rb(folded_float)])];
    let constants = indexmap::IndexMap::from([(folded_float.raw(), 3.5_f64.to_bits() as i64)]);

    let (bytes, _) = build_module_default(&[], &ops, &constants);
    validate_wasm(&bytes);
    let local_types = wasmparser::Parser::new(0)
        .parse_all(&bytes)
        .find_map(|payload| match payload.unwrap() {
            wasmparser::Payload::CodeSectionEntry(body) => Some(
                body.get_locals_reader()
                    .unwrap()
                    .into_iter()
                    .map(|local| local.unwrap().1)
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .expect("generated module must contain its trace function");
    assert!(
        local_types.contains(&wasmparser::ValType::F64),
        "the producer-less Float operand must declare an f64 local"
    );
}

/// A peeled loop header names live-ins as InputArgs that are not portal
/// inputargs and have no producing op. Those boxes are the LABEL's block
/// parameters (`consider_label`); they must not trip unbound-pool decline.
#[test]
fn label_livein_inputarg_is_defined_at_the_label() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let live_in = OpRef::input_arg_int(101);
    let ops = vec![
        Op::new(OpCode::Label, &[rb(OpRef::input_arg_int(0)), rb(live_in)]),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(1)], &[live_in]),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(1)), rb(live_in)]),
    ];
    let (bytes, guards) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
}

/// Same as `label_livein_inputarg_is_defined_at_the_label`, but the peeled
/// live-in is an `InputArgRef`. Declining every `Type::Ref` at LABEL
/// re-broke those traces (`fib_loop` / `int_loop`). Residual vable slots
/// are `RefOp`s and stay declined — see
/// `label_ref_phi_without_a_producer_declines`.
#[test]
fn label_livein_inputarg_ref_is_defined_at_the_label() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let live_in = OpRef::input_arg_ref(101);
    let ops = vec![
        Op::new(OpCode::Label, &[rb(OpRef::input_arg_int(0)), rb(live_in)]),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(1)], &[live_in]),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(1)), rb(live_in)]),
    ];
    let (bytes, guards) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
}

/// A LABEL live-in that is only a constants-map entry must still be seeded
/// in the prologue. Treating every LABEL arg as defined skipped that store,
/// so the local read as the zero wasm initializes it to.
#[test]
fn label_livein_pool_const_is_seeded_in_prologue() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let live_in = OpRef::input_arg_int(101);
    let ops = vec![
        Op::new(OpCode::Label, &[rb(OpRef::input_arg_int(0)), rb(live_in)]),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), live_in],
            OpRef::int_op(1),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(1)], &[live_in]),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(1)), rb(live_in)]),
    ];
    let mut constants = indexmap::IndexMap::new();
    constants.insert(live_in.raw(), 0x1111_2222_3333_4444);
    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    const SEED: i64 = 0x1111_2222_3333_4444u64 as i64;
    let mut saw_seed = false;
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            let mut pending = false;
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::I64Const { value } if value == SEED => pending = true,
                    wasmparser::Operator::LocalSet { .. } if pending => {
                        saw_seed = true;
                        break;
                    }
                    _ => pending = false,
                }
            }
        }
    }
    assert!(
        saw_seed,
        "prologue must materialize the LABEL live-in from the constants map"
    );
}

/// Count the direct `wasm_jit_write_barrier` table calls by their unique table
/// target immediate.  The direct lowering places that `i32.const` immediately
/// before its `call_indirect`.
fn direct_write_barrier_call_count(bytes: &[u8], target: i32) -> usize {
    let mut count = 0;
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            let mut target_on_stack = false;
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::I32Const { value } if value == target => {
                        target_on_stack = true;
                    }
                    wasmparser::Operator::I64Const { value } if value == i64::from(target) => {
                        target_on_stack = true;
                    }
                    wasmparser::Operator::CallIndirect { .. } if target_on_stack => {
                        count += 1;
                        target_on_stack = false;
                    }
                    _ => target_on_stack = false,
                }
            }
        }
    }
    count
}

fn build_module_with_write_barrier_target(
    inputargs: &[InputArgRc],
    ops: &[Op],
    write_barrier_target: i64,
) -> Vec<u8> {
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(write_barrier_target, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        // The allocated trace keeps both Ref inputs live across New; reserve
        // their homes in the shared helper geometry.  The value count is the
        // fail index, this trace's three-slot value area, and the trace-wide
        // GUARD_VALUE counter slot `normal_frame_value_slots` always reserves.
        frame: codegen::FrameGeometry::compact(5, 2, 0),
        ca: codegen::CaParams::default(),
    };
    let (bytes, _, _, _) = codegen::build_wasm_module(&rewrite_module_inputs(inputs))
        .expect("wasm codegen should succeed");
    bytes
}

fn build_module_with_gc_table(
    inputargs: &[InputArgRc],
    ops: &[Op],
    write_barrier_target: i64,
    gc_table_base: u32,
    gc_const_keys: &[usize],
) -> Vec<u8> {
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(write_barrier_target, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base,
        gc_const_keys: gc_const_keys.to_vec(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::compact(5, 2, 0),
        ca: codegen::CaParams::default(),
    };
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    bytes
}

fn module_with_collecting_calls(call_count: usize) -> Vec<u8> {
    const WB_TARGET: i64 = 127;
    let mut ops = Vec::with_capacity(call_count + 2);
    for i in 0..call_count {
        let call = make_op(
            OpCode::CallMayForceI,
            &[OpRef::const_int(42)],
            OpRef::int_op(1 + i as u32),
        );
        call.setdescr(majit_ir::descr::make_call_descr_full(
            0,
            vec![],
            Type::Int,
            false,
            8,
            EffectInfo::default(),
        ));
        ops.push(call);
    }
    let guard = Op::new(OpCode::GuardNotForced, &[]);
    guard.setfailargs(smallvec![rb(OpRef::input_arg_ref(0))]);
    ops.push(guard);
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_ref(0))]);
    finish.setfailargs(smallvec![rb(OpRef::input_arg_ref(0))]);
    ops.push(finish);
    // The barrier follows the frame reload, so the collector's reload
    // helper must be configured (`gcrootmap and wbdescr`).
    let mut ca = codegen::CaParams::default();
    ca.ca_reload_fn_ptr = 1;
    let inputs = codegen::ModuleBuildInputs {
        inputargs: vec![InputArg::from_type_rc(Type::Ref, 0)],
        ops,
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(WB_TARGET, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::compact(5, 2, 0),
        ca,
    };
    let (bytes, _, _, _) = codegen::build_wasm_module(&rewrite_module_inputs(inputs))
        .expect("wasm codegen should succeed");
    bytes
}

fn code_bodies(bytes: &[u8]) -> Vec<Vec<wasmparser::Operator<'_>>> {
    let mut bodies = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            bodies.push(
                body.get_operators_reader()
                    .unwrap()
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap(),
            );
        }
    }
    bodies
}

fn frame_barrier_flag_checks(ops: &[wasmparser::Operator], flag: i32) -> usize {
    ops.windows(4)
        .filter(|window| {
            matches!(
                window,
                [
                    wasmparser::Operator::I32Load8U { .. },
                    wasmparser::Operator::I32Const { value },
                    wasmparser::Operator::I32And,
                    wasmparser::Operator::If { .. }
                ] if *value == flag
            )
        })
        .count()
}

#[test]
fn jitframe_barrier_checks_flags_and_reserves_arity_one_for_zero_argument_residuals() {
    const WB_TARGET: i64 = 127;
    let bytes = module_with_collecting_calls(1);
    assert_eq!(direct_write_barrier_call_count(&bytes, WB_TARGET as i32), 1);
    validate_wasm(&bytes);
    // One site does not pay for a function, so the fastpath stays in the trace.
    let bodies = code_bodies(&bytes);
    assert_eq!(bodies.len(), 1);
    // _reload_frame_if_necessary's frame fastpath must guard the helper;
    // otherwise every collecting call searches the managed heap even when
    // the frame has already been remembered (or has a zeroed off-GC header).
    let flag = i32::from(codegen::WriteBarrierHelpers::for_current_gc(WB_TARGET, 0).if_flag);
    assert_eq!(frame_barrier_flag_checks(&bodies[0], flag), 1);
}

#[test]
fn jitframe_barrier_outlines_when_many_sites_collect() {
    const WB_TARGET: i64 = 127;
    // Past `outline_frame_write_barrier`: 6 * (17 - 2) > 17 + 40.
    let bytes = module_with_collecting_calls(6);
    validate_wasm(&bytes);
    let bodies = code_bodies(&bytes);
    assert_eq!(bodies.len(), 2);
    let flag = i32::from(codegen::WriteBarrierHelpers::for_current_gc(WB_TARGET, 0).if_flag);
    assert_eq!(frame_barrier_flag_checks(&bodies[0], flag), 0);
    assert_eq!(frame_barrier_flag_checks(&bodies[1], flag), 1);
    // The helper body is `FRAME_WB_BODY_OPS` plus the function `end`.
    assert_eq!(bodies[1].len(), 18);
    assert_eq!(direct_write_barrier_call_count(&bytes, WB_TARGET as i32), 1);
    let helper_calls = bodies[0]
        .iter()
        .filter(|op| matches!(op, wasmparser::Operator::Call { function_index: 1 }))
        .count();
    assert_eq!(helper_calls, 6);
}

fn func_import_count(bytes: &[u8]) -> u32 {
    let mut count = 0u32;
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::ImportSection(reader) = payload.unwrap() {
            for import in reader {
                if matches!(import.unwrap().ty, wasmparser::TypeRef::Func(_)) {
                    count += 1;
                }
            }
        }
    }
    count
}

fn body_has_loop(ops: &[wasmparser::Operator]) -> bool {
    ops.iter()
        .any(|op| matches!(op, wasmparser::Operator::Loop { .. }))
}

/// One `CallMayForce` so the ref input is live across a collecting call and
/// takes a home. `fn_ptr == 0` keeps the frame barrier inline-or-absent, and
/// arity 1 does not admit a spill helper, so an outlined exit is the only
/// extra function.
fn guard_exit_module(guard_count: usize, ref_fail: bool, exit_table_base: u32) -> Vec<u8> {
    let fail = if ref_fail {
        OpRef::input_arg_ref(0)
    } else {
        OpRef::input_arg_int(0)
    };
    let mut ops = Vec::with_capacity(guard_count + 2);
    if ref_fail {
        let call = make_op(
            OpCode::CallMayForceI,
            &[OpRef::const_int(42)],
            OpRef::int_op(1),
        );
        call.setdescr(majit_ir::descr::make_call_descr_full(
            0,
            vec![],
            Type::Int,
            false,
            8,
            EffectInfo::default(),
        ));
        ops.push(call);
    }
    for _ in 0..guard_count {
        ops.push(make_guard(
            OpCode::GuardTrue,
            &[OpRef::const_int(1)],
            &[fail],
        ));
    }
    let finish = Op::new(OpCode::Finish, &[rb(fail)]);
    finish.setfailargs(smallvec![rb(fail)]);
    ops.push(finish);
    let mut ca = codegen::CaParams::default();
    ca.exit_table_base = exit_table_base;
    let input = if ref_fail {
        InputArg::from_type_rc(Type::Ref, 0)
    } else {
        InputArg::from_type_rc(Type::Int, 0)
    };
    let inputs = codegen::ModuleBuildInputs {
        inputargs: vec![input],
        ops,
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::compact(8, 4, 0),
        ca,
    };
    let (bytes, _, _, _) = codegen::build_wasm_module(&rewrite_module_inputs(inputs))
        .expect("wasm codegen should succeed");
    bytes
}

#[test]
fn guard_exit_one_home_stays_inline() {
    // n=1 saves 12 operators. One site does not clear `GUARD_EXIT_OUTLINE_COST`.
    let bytes = guard_exit_module(1, true, 0x4000);
    validate_wasm(&bytes);
    let bodies = code_bodies(&bytes);
    assert_eq!(bodies.len(), 1);
    assert!(!body_has_loop(&bodies[0]));
}

#[test]
fn guard_exit_home_loads_outline_past_the_fixed_cost() {
    // 24 * 12 = 288, past the helper's fixed cost of 160.
    let bytes = guard_exit_module(24, true, 0x4000);
    validate_wasm(&bytes);
    let bodies = code_bodies(&bytes);
    assert_eq!(bodies.len(), 2);
    assert!(!body_has_loop(&bodies[0]));
    assert!(body_has_loop(&bodies[1]));
    let helper = func_import_count(&bytes) + 1;
    let calls = bodies[0]
        .iter()
        .filter(|op| matches!(op, wasmparser::Operator::Call { function_index } if *function_index == helper))
        .count();
    assert_eq!(calls, 24);
}

#[test]
fn guard_exit_without_homes_stays_inline() {
    let bytes = guard_exit_module(24, false, 0x4000);
    validate_wasm(&bytes);
    let bodies = code_bodies(&bytes);
    assert_eq!(bodies.len(), 1);
    assert!(!body_has_loop(&bodies[0]));
    let stores = bodies[0]
        .iter()
        .filter(|op| matches!(op, wasmparser::Operator::I64Store { .. }))
        .count();
    assert!(stores >= 24, "int fail args still spill at the site");
}

#[test]
fn write_barrier_elision_keeps_one_barrier_per_base() {
    use majit_ir::descr::{SimpleFieldDescr, SimpleSizeDescr};
    use std::sync::Arc;

    const WB_TARGET: i64 = 0x4a11;
    let pointer_field = Arc::new(SimpleFieldDescr::new(0, 0, 8, Type::Ref, false));
    let finish = Op::new(OpCode::Finish, &[]);

    let new_obj = make_op(OpCode::New, &[], OpRef::ref_op(2));
    new_obj.setdescr(Arc::new(SimpleSizeDescr::new(0, 16, 1)));
    let new_store_a = Op::new(
        OpCode::SetfieldGc,
        &[rb(OpRef::ref_op(2)), rb(OpRef::input_arg_ref(0))],
    );
    new_store_a.setdescr(pointer_field.clone());
    let new_store_b = Op::new(
        OpCode::SetfieldGc,
        &[rb(OpRef::ref_op(2)), rb(OpRef::input_arg_ref(1))],
    );
    new_store_b.setdescr(pointer_field.clone());
    let allocated = build_module_with_write_barrier_target(
        &[
            InputArg::from_type_rc(Type::Ref, 0),
            InputArg::from_type_rc(Type::Ref, 1),
        ],
        &[new_obj, new_store_a, new_store_b, finish.clone()],
        WB_TARGET,
    );
    validate_wasm(&allocated);
    // rewrite.rs `gen_malloc_nursery` `remember_wb`s the young result, so
    // both stores are elided. The deleted codegen New-arm used to emit 1.
    assert_eq!(
        direct_write_barrier_call_count(&allocated, WB_TARGET as i32),
        0,
        "gen_malloc_fixedsize remembers a young fixed-size malloc; neither store barriers"
    );

    let live_store_a = Op::new(
        OpCode::SetfieldGc,
        &[rb(OpRef::input_arg_ref(0)), rb(OpRef::input_arg_ref(1))],
    );
    live_store_a.setdescr(pointer_field.clone());
    let live_store_b = Op::new(
        OpCode::SetfieldGc,
        &[rb(OpRef::input_arg_ref(0)), rb(OpRef::input_arg_ref(2))],
    );
    live_store_b.setdescr(pointer_field);
    let repeated_livein = build_module_with_write_barrier_target(
        &[
            InputArg::from_type_rc(Type::Ref, 0),
            InputArg::from_type_rc(Type::Ref, 1),
            InputArg::from_type_rc(Type::Ref, 2),
        ],
        &[live_store_a, live_store_b, finish],
        WB_TARGET,
    );
    validate_wasm(&repeated_livein);
    assert_eq!(
        direct_write_barrier_call_count(&repeated_livein, WB_TARGET as i32),
        1,
        "repeated stores into one live-in base must emit one write-barrier call"
    );
}

/// rewrite.py `gen_malloc_nursery` `remember_write_barrier`: a nursery
/// New is young, so SETFIELD_GC on it emits no COND_CALL_GC_WB.
#[test]
fn nursery_new_seeds_write_barrier_applied() {
    const WB_TARGET: i64 = 0x4a11;
    let mut inputs = nursery_new_inputs(
        vec![
            new_with_gc_field(1, 53, 8),
            setfield_gc(1, OpRef::input_arg_int(0), 8),
            finish_int_arg0(),
        ],
        53,
    );
    inputs.wb = codegen::WriteBarrierHelpers::for_current_gc(WB_TARGET, 0);
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert_eq!(
        direct_write_barrier_call_count(&bytes, WB_TARGET as i32),
        0,
        "gen_malloc_nursery remembers the new object; first SETFIELD_GC is not a barrier"
    );
}

/// rewrite.py `gen_malloc_nursery` on a const-length array: SETARRAYITEM
/// of a Ref does not emit COND_CALL_GC_WB.
#[test]
fn nursery_const_newarray_seeds_write_barrier_applied() {
    const WB_TARGET: i64 = 0x4a11;
    let mut inputs = nursery_new_inputs(
        vec![
            plain_new_array(1, 53, 2),
            setarrayitem_gc(1, 0, OpRef::input_arg_int(0), 53),
            finish_int_arg0(),
        ],
        53,
    );
    inputs.wb = codegen::WriteBarrierHelpers::for_current_gc(WB_TARGET, 0);
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert_eq!(
        direct_write_barrier_call_count(&bytes, WB_TARGET as i32),
        0,
        "const-length nursery NEW_ARRAY remembers write_barrier_applied"
    );
}

#[test]
fn write_barrier_elision_follows_same_as_r_base() {
    use majit_ir::descr::SimpleFieldDescr;
    use std::sync::Arc;

    const WB_TARGET: i64 = 0x4a11;
    let pointer_field = Arc::new(SimpleFieldDescr::new(0, 0, 8, Type::Ref, false));
    let store_before_alias = Op::new(
        OpCode::SetfieldGc,
        &[rb(OpRef::input_arg_ref(0)), rb(OpRef::input_arg_ref(1))],
    );
    store_before_alias.setdescr(pointer_field.clone());
    let alias = make_op(
        OpCode::SameAsR,
        &[OpRef::input_arg_ref(0)],
        OpRef::ref_op(3),
    );
    let store_through_alias = Op::new(
        OpCode::SetfieldGc,
        &[rb(OpRef::ref_op(3)), rb(OpRef::input_arg_ref(2))],
    );
    store_through_alias.setdescr(pointer_field);

    let bytes = build_module_with_write_barrier_target(
        &[
            InputArg::from_type_rc(Type::Ref, 0),
            InputArg::from_type_rc(Type::Ref, 1),
            InputArg::from_type_rc(Type::Ref, 2),
        ],
        &[
            store_before_alias,
            alias,
            store_through_alias,
            Op::new(OpCode::Finish, &[]),
        ],
        WB_TARGET,
    );
    validate_wasm(&bytes);
    assert_eq!(
        direct_write_barrier_call_count(&bytes, WB_TARGET as i32),
        1,
        "stores through a SameAsR base share one applied write barrier"
    );
}

fn execute_ovf_trace_with_guard(
    opcode: OpCode,
    guard_opcode: OpCode,
    a: i64,
    b: i64,
) -> (i64, i64) {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let guard = Op::new(guard_opcode, &[]);
    guard.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))]);
    finish.setfailargs(smallvec![rb(OpRef::int_op(2))]);
    let ops = vec![
        make_op(
            opcode,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            OpRef::int_op(2),
        ),
        guard,
        finish,
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());

    let engine = Engine::default();
    let module = Module::new(&engine, &bytes).expect("generated trace should compile");
    let mut store = Store::new(&engine, ());
    let memory =
        Memory::new(&mut store, MemoryType::new(1, None)).expect("test memory should allocate");
    memory
        .write(
            &mut store,
            codegen::FRAME_SLOT_BASE as usize,
            &a.to_le_bytes(),
        )
        .unwrap();
    memory
        .write(
            &mut store,
            (codegen::FRAME_SLOT_BASE + 8) as usize,
            &b.to_le_bytes(),
        )
        .unwrap();
    let mut linker = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("generated trace should instantiate");
    instance
        .get_typed_func::<i32, i32>(&store, "trace")
        .unwrap()
        .call(&mut store, 0)
        .expect("generated trace should execute");

    let mut fail_index = [0; 8];
    let mut result = [0; 8];
    memory.read(&store, 0, &mut fail_index).unwrap();
    memory
        .read(&store, codegen::FRAME_SLOT_BASE as usize, &mut result)
        .unwrap();
    (i64::from_le_bytes(fail_index), i64::from_le_bytes(result))
}

fn execute_ovf_trace(opcode: OpCode, a: i64, b: i64) -> (i64, i64) {
    execute_ovf_trace_with_guard(opcode, OpCode::GuardNoOverflow, a, b)
}

/// Same trace as [`execute_ovf_trace`], but the constant `c` is an operand
/// rather than a second input argument, so the overflow check takes the
/// folded-bound form instead of the sign-comparison one. `const_first` puts
/// it on the left, which only addition accepts.
fn execute_ovf_trace_const(opcode: OpCode, a: i64, c: i64, const_first: bool) -> (i64, i64) {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let guard = Op::new(OpCode::GuardNoOverflow, &[]);
    guard.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish.setfailargs(smallvec![rb(OpRef::int_op(1))]);
    let args = if const_first {
        [OpRef::const_int(c), OpRef::input_arg_int(0)]
    } else {
        [OpRef::input_arg_int(0), OpRef::const_int(c)]
    };
    let ops = vec![make_op(opcode, &args, OpRef::int_op(1)), guard, finish];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());

    let engine = Engine::default();
    let module = Module::new(&engine, &bytes).expect("generated trace should compile");
    let mut store = Store::new(&engine, ());
    let memory =
        Memory::new(&mut store, MemoryType::new(1, None)).expect("test memory should allocate");
    memory
        .write(
            &mut store,
            codegen::FRAME_SLOT_BASE as usize,
            &a.to_le_bytes(),
        )
        .unwrap();
    let mut linker = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("generated trace should instantiate");
    instance
        .get_typed_func::<i32, i32>(&store, "trace")
        .unwrap()
        .call(&mut store, 0)
        .expect("generated trace should execute");

    let mut fail_index = [0; 8];
    let mut result = [0; 8];
    memory.read(&store, 0, &mut fail_index).unwrap();
    memory
        .read(&store, codegen::FRAME_SLOT_BASE as usize, &mut result)
        .unwrap();
    (i64::from_le_bytes(fail_index), i64::from_le_bytes(result))
}

/// A constant operand takes the folded-bound overflow check, so it needs the
/// same verdicts as the general form at the extremes — including the two
/// bounds that sit closest to overflowing themselves, `a - i64::MIN` and
/// `a + i64::MIN`.
#[test]
fn test_ovf_against_a_constant_matches_the_general_form() {
    // (opcode, a, c, overflows)
    let cases = [
        (OpCode::IntAddOvf, 10, 20, false),
        (OpCode::IntAddOvf, 5, 0, false),
        (OpCode::IntAddOvf, i64::MAX, 1, true),
        (OpCode::IntAddOvf, i64::MAX - 1, 1, false),
        (OpCode::IntAddOvf, 10, -20, false),
        (OpCode::IntAddOvf, i64::MIN, -1, true),
        (OpCode::IntAddOvf, -1, i64::MIN, true),
        (OpCode::IntAddOvf, 0, i64::MIN, false),
        (OpCode::IntSubOvf, 100, 58, false),
        (OpCode::IntSubOvf, 5, 0, false),
        (OpCode::IntSubOvf, i64::MIN, 1, true),
        (OpCode::IntSubOvf, i64::MIN + 1, 1, false),
        (OpCode::IntSubOvf, 10, -5, false),
        (OpCode::IntSubOvf, i64::MAX, -1, true),
        (OpCode::IntSubOvf, 0, i64::MIN, true),
        (OpCode::IntSubOvf, -1, i64::MIN, false),
    ];
    for (opcode, a, c, overflows) in cases {
        let (fail_index, result) = execute_ovf_trace_const(opcode, a, c, false);
        if overflows {
            assert_eq!(fail_index, 0, "{opcode:?}: {a} op {c} should guard-exit");
        } else {
            let expected = match opcode {
                OpCode::IntAddOvf => a.wrapping_add(c),
                _ => a.wrapping_sub(c),
            };
            assert_eq!(
                (fail_index, result),
                (1, expected),
                "{opcode:?}: {a} op {c}"
            );
        }
    }

    // Addition is commutative, so the constant is also accepted on the left.
    assert_eq!(
        execute_ovf_trace_const(OpCode::IntAddOvf, 10, 20, true),
        (1, 30)
    );
    assert_eq!(
        execute_ovf_trace_const(OpCode::IntAddOvf, i64::MAX, 1, true).0,
        0
    );
}

#[test]
fn test_int_add_ovf_guards_overflow() {
    for (a, b, expected) in [(10, 20, 30), (i64::MIN, 1, i64::MIN + 1)] {
        assert_eq!(execute_ovf_trace(OpCode::IntAddOvf, a, b), (1, expected));
    }
    assert_eq!(execute_ovf_trace(OpCode::IntAddOvf, i64::MAX, 1).0, 0);
}

#[test]
fn test_int_sub_ovf_guards_overflow() {
    for (a, b, expected) in [(100, 58, 42), (i64::MAX, 1, i64::MAX - 1)] {
        assert_eq!(execute_ovf_trace(OpCode::IntSubOvf, a, b), (1, expected));
    }
    assert_eq!(execute_ovf_trace(OpCode::IntSubOvf, i64::MIN, 1).0, 0);
}

#[test]
fn test_int_mul_ovf_fuses_its_adjacent_overflow_guard() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let guard = Op::new(OpCode::GuardNoOverflow, &[]);
    guard.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
    let ops = vec![
        make_op(
            OpCode::IntMulOvf,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            OpRef::int_op(2),
        ),
        guard,
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))]),
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);

    let (mut widened, mut i32_result_blocks) = (0usize, 0usize);
    count_operators(&bytes, |op| match op {
        wasmparser::Operator::I64ExtendI32U => widened += 1,
        wasmparser::Operator::If {
            blockty: wasmparser::BlockType::Type(wasmparser::ValType::I32),
        } => i32_result_blocks += 1,
        _ => {}
    });
    // Both arms of the signed-32 split answer the same one-bit predicate, so
    // the block yields it and the guard reads it off the stack -- the shape the
    // add and sub forms already take, and the one every sibling backend takes
    // by publishing a condition code instead of a stored flag.
    assert_eq!(i32_result_blocks, 1, "the split yields the guard predicate");
    assert_eq!(
        widened, 0,
        "a fused predicate never widens into a flag local"
    );
}

#[test]
fn test_int_mul_ovf_guard_overflow_inverts_on_both_split_arms() {
    // `GuardOverflow` fails when the multiply did NOT overflow, so the fused
    // predicate is the inverse of `GuardNoOverflow`'s on both arms of the
    // signed-32 split: the constant the fast arm yields flips, and the
    // full-width arm's sign-word comparison flips with it.
    for (a, b) in [(6_i64, 7_i64), (i32::MAX as i64 + 1, 2)] {
        assert_eq!(
            execute_ovf_trace_with_guard(OpCode::IntMulOvf, OpCode::GuardOverflow, a, b).0,
            0,
            "{a} * {b} does not overflow, so GuardOverflow exits"
        );
    }
    for (a, b) in [(i64::MIN, -1_i64), (1_i64 << 62, 3)] {
        assert_eq!(
            execute_ovf_trace_with_guard(OpCode::IntMulOvf, OpCode::GuardOverflow, a, b).0,
            1,
            "{a} * {b} overflows, so GuardOverflow passes"
        );
    }
}

#[test]
fn test_int_mul_ovf_guards_overflow() {
    for (a, b, expected) in [
        (6, 7, 42),
        (-9, -7, 63),
        (i32::MIN as i64, i32::MIN as i64, 1_i64 << 62),
        (
            i32::MAX as i64,
            i32::MAX as i64,
            (i32::MAX as i64) * (i32::MAX as i64),
        ),
        (i64::MIN, 1, i64::MIN),
        (i32::MAX as i64 + 1, 2, 1_i64 << 32),
    ] {
        assert_eq!(execute_ovf_trace(OpCode::IntMulOvf, a, b), (1, expected));
    }
    for (a, b) in [
        (i64::MIN, -1),
        (i64::MAX, 2),
        (1_i64 << 62, 3),
        (i32::MAX as i64 + 1, 1_i64 << 32),
    ] {
        assert_eq!(execute_ovf_trace(OpCode::IntMulOvf, a, b).0, 0);
    }
}

#[test]
fn test_int_mul_ovf_emits_signed32_fast_path_and_full_width_fallback() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let guard = Op::new(OpCode::GuardNoOverflow, &[]);
    guard.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
    let ops = vec![
        make_op(
            OpCode::IntMulOvf,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            OpRef::int_op(2),
        ),
        guard,
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))]),
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);

    let mut extend32_s = 0;
    let mut i64_mul = 0;
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::I64Extend32S => extend32_s += 1,
                    wasmparser::Operator::I64Mul => i64_mul += 1,
                    _ => {}
                }
            }
        }
    }
    assert_eq!(extend32_s, 2, "both factors need an exact signed-32 check");
    assert!(
        i64_mul > 1,
        "the software full-width overflow fallback must remain in the module"
    );
}

/// Each guard spills its own fail arguments before it branches to the shared
/// bridge-dispatch epilogue.
#[test]
fn test_guard_fail_args_spill_in_their_own_failure_arms() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let first_guard = Op::new(OpCode::GuardTrue, &[rb(OpRef::input_arg_int(0))]);
    first_guard.setfailargs(smallvec![
        rb(OpRef::input_arg_int(0)),
        rb(OpRef::input_arg_int(1)),
    ]);
    let second_guard = Op::new(OpCode::GuardFalse, &[rb(OpRef::input_arg_int(1))]);
    second_guard.setfailargs(smallvec![
        rb(OpRef::input_arg_int(1)),
        rb(OpRef::input_arg_int(0)),
    ]);
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]);
    let (bytes, guards) = build_module_default(
        &inputargs,
        &[first_guard, second_guard, finish],
        &indexmap::IndexMap::new(),
    );
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 3);

    let mut control_stack = Vec::new();
    let mut stores_per_guard_arm = Vec::new();
    let mut br_tables = 0;
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::If { .. } => control_stack.push(Some(0usize)),
                    wasmparser::Operator::Block { .. } | wasmparser::Operator::Loop { .. } => {
                        control_stack.push(None);
                    }
                    wasmparser::Operator::End => {
                        if let Some(Some(stores)) = control_stack.pop() {
                            stores_per_guard_arm.push(stores);
                        }
                    }
                    wasmparser::Operator::I64Store { .. } => {
                        if let Some(Some(stores)) =
                            control_stack.iter_mut().rev().find(|frame| frame.is_some())
                        {
                            *stores += 1;
                        }
                    }
                    wasmparser::Operator::BrTable { .. } => br_tables += 1,
                    _ => {}
                }
            }
        }
    }
    assert_eq!(stores_per_guard_arm, [3, 3]);
    assert_eq!(
        br_tables, 0,
        "guard exits must not use a selector dispatcher"
    );
}

#[test]
fn test_fused_integer_guard_true_uses_inverse_comparison_directly() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let compare = make_op(
        OpCode::IntLt,
        &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
        OpRef::int_op(2),
    );
    let guard = make_guard(
        OpCode::GuardTrue,
        &[OpRef::int_op(2)],
        &[OpRef::input_arg_int(0)],
    );
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(1))]);
    let (bytes, _) = build_module_default(
        &inputargs,
        &[compare, guard, finish],
        &indexmap::IndexMap::new(),
    );
    validate_wasm(&bytes);

    let mut ge_s = 0;
    let mut i32_eqz = 0;
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::I64GeS => ge_s += 1,
                    wasmparser::Operator::I32Eqz => i32_eqz += 1,
                    _ => {}
                }
            }
        }
    }
    assert_eq!(ge_s, 1, "IntLt guard failure should be emitted as IntGe");
    assert_eq!(i32_eqz, 0, "integer inverse must not materialize i32.eqz");
}

#[test]
fn test_cold_guard_recovery_preserves_nonzero_base_and_typed_bits() {
    const FAIL_INDEX_BASE: u32 = 37;
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Ref, 1),
        InputArg::from_type_rc(Type::Float, 2),
    ];
    let fail_args = smallvec![
        rb(OpRef::input_arg_ref(1)),
        rb(OpRef::input_arg_float(2)),
        rb(OpRef::input_arg_int(0)),
    ];
    let guard = Op::new(OpCode::GuardTrue, &[rb(OpRef::input_arg_int(0))]);
    guard.setfailargs(fail_args.clone());
    let finish = Op::new(OpCode::Finish, &fail_args);
    finish.setfailargs(fail_args);
    let ops = [guard, finish];
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: FAIL_INDEX_BASE,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    };
    let (bytes, guards, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    assert_eq!(guards[0].fail_index, FAIL_INDEX_BASE);

    let ref_bits = 0x1234_5678_i64;
    let float_bits = (-13.25_f64).to_bits() as i64;
    let mut store = Store::new(&Engine::default(), ());
    let engine = store.engine().clone();
    let module = Module::new(&engine, &bytes).expect("generated trace should compile");
    let memory = Memory::new(&mut store, MemoryType::new(1, None)).unwrap();
    for (slot, bits) in [0_i64, ref_bits, float_bits].into_iter().enumerate() {
        memory
            .write(
                &mut store,
                codegen::FRAME_SLOT_BASE as usize + slot * 8,
                &bits.to_le_bytes(),
            )
            .unwrap();
    }
    let mut linker = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    let instance = linker.instantiate_and_start(&mut store, &module).unwrap();
    instance
        .get_typed_func::<i32, i32>(&store, "trace")
        .unwrap()
        .call(&mut store, 0)
        .unwrap();

    let mut word = [0; 8];
    memory.read(&store, 0, &mut word).unwrap();
    assert_eq!(i64::from_le_bytes(word), FAIL_INDEX_BASE as i64);
    for (slot, expected) in [ref_bits, float_bits, 0].into_iter().enumerate() {
        memory
            .read(
                &store,
                codegen::FRAME_SLOT_BASE as usize + slot * 8,
                &mut word,
            )
            .unwrap();
        assert_eq!(i64::from_le_bytes(word), expected);
    }
}

#[test]
fn test_guard_overflow_uses_pending_flag() {
    assert_eq!(
        execute_ovf_trace_with_guard(OpCode::IntAddOvf, OpCode::GuardOverflow, i64::MAX, 1,).0,
        1
    );
    assert_eq!(
        execute_ovf_trace_with_guard(OpCode::IntAddOvf, OpCode::GuardOverflow, 1, 2).0,
        0
    );
}

#[test]
fn test_empty_trace() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![{
        let op = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]);
        op.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
        op
    }];
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    assert!(guards[0].is_finish);
}

/// Build, validate, instantiate and run `inputs` under wasmi, then read the
/// guard-exit index the trace wrote at offset 0 and the first three frame
/// slots.  Every inline-region repro drives the module exactly this way and
/// differs only in the exit index it expects.
fn run_inline_region_trace(inputs: &codegen::ModuleBuildInputs) -> (i64, i64, i64, i64) {
    let (bytes, _, _, _) = codegen::build_wasm_module(inputs).expect("non-header region merges");
    validate_wasm(&bytes);

    let engine = Engine::default();
    let module = Module::new(&engine, &bytes).expect("generated trace should compile");
    let mut store = Store::new(&engine, ());
    let memory =
        Memory::new(&mut store, MemoryType::new(2, None)).expect("test memory should allocate");
    memory
        .write(
            &mut store,
            codegen::FRAME_SLOT_BASE as usize,
            &0i64.to_le_bytes(),
        )
        .unwrap();
    let mut linker = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("generated trace should instantiate");
    instance
        .get_typed_func::<i32, i32>(&store, "trace")
        .unwrap()
        .call(&mut store, 0)
        .expect("generated trace should execute");

    let read = |off: u64| {
        let mut buf = [0u8; 8];
        memory.read(&store, off as usize, &mut buf).unwrap();
        i64::from_le_bytes(buf)
    };
    (
        read(0),
        read(codegen::FRAME_SLOT_BASE),
        read(codegen::FRAME_SLOT_BASE + 8),
        read(codegen::FRAME_SLOT_BASE + 16),
    )
}

/// A bridge whose merge is deferred carries an entry counter and calls back
/// once, on the entry that reaches the threshold — not before it, and not
/// again after. The counter is what separates a region worth its owner's
/// re-emission from one that is crossed a few thousand times and never pays it
/// back, so both halves of "exactly once, at the threshold" are the point.
///
/// The trip does not clear the source guard's dispatch cell. The bridge
/// stays the target until the host, already outside compiled code, installs
/// the merged module.
#[test]
fn a_deferred_merge_trips_once_at_its_threshold() {
    const COUNTER_ADDR: u32 = 0x10000;
    const CELLS_BASE: u32 = 0x10200;
    const CELL_INDEX: u32 = 2;
    const THRESHOLD: u64 = 3;

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(1)], &[OpRef::int_op(1)]),
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]),
    ];
    let mut inputs = inline_region_inputs(&inputargs, ops, Vec::new());
    inputs.inline_trip = Some(codegen::InlineTripProbe {
        counter_addr: COUNTER_ADDR,
        threshold: THRESHOLD,
        // Table slot 1 below; slot 0 stays null so a mis-selected slot traps
        // rather than calling the wrong function.
        trip_fn_ptr: 1,
        pending_slot: 77,
    });

    let (bytes, _, _, _) = codegen::build_wasm_module(&inputs).expect("the armed module builds");
    validate_wasm(&bytes);

    let engine = Engine::default();
    let module = Module::new(&engine, &bytes).expect("the armed module compiles");
    let mut store = Store::new(&engine, Vec::<i64>::new());
    let memory =
        Memory::new(&mut store, MemoryType::new(2, None)).expect("test memory should allocate");
    let trip = wasmi::Func::wrap(
        &mut store,
        |mut caller: wasmi::Caller<'_, Vec<i64>>, id: i64| {
            caller.data_mut().push(id);
            0i64
        },
    );
    let table = Table::new(
        &mut store,
        TableType::new(ValType::FuncRef, 2, None),
        Val::default(ValType::FuncRef),
    )
    .expect("test table should allocate");
    table
        .set(&mut store, 1, Val::from(trip))
        .expect("the trip callback occupies slot 1");
    let mut linker = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .define("env", "__indirect_function_table", table)
        .unwrap();
    // A live dispatch cell. The trip must not clear it: the bridge it names
    // is still the guard's target.
    let cell_addr = (CELLS_BASE + CELL_INDEX * 4) as usize;
    memory
        .write(&mut store, cell_addr, &7u32.to_le_bytes())
        .unwrap();
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("the armed module instantiates");
    let trace = instance
        .get_typed_func::<i32, i32>(&store, "trace")
        .unwrap();
    let cell = |store: &Store<Vec<i64>>| {
        let mut buf = [0u8; 4];
        memory.read(store, cell_addr, &mut buf).unwrap();
        u32::from_le_bytes(buf)
    };

    let counter = |store: &Store<Vec<i64>>| {
        let mut buf = [0u8; 8];
        memory.read(store, COUNTER_ADDR as usize, &mut buf).unwrap();
        u64::from_le_bytes(buf)
    };
    for entry in 1..=THRESHOLD + 2 {
        trace.call(&mut store, 0).expect("the trace runs");
        assert_eq!(counter(&store), entry, "every entry is counted");
        let fired = if entry < THRESHOLD { 0 } else { 1 };
        assert_eq!(
            store.data().len(),
            fired,
            "the callback fires on entry {THRESHOLD} and on no other"
        );
        assert_eq!(
            cell(&store),
            7,
            "the dispatch cell keeps naming the attached bridge after the trip"
        );
    }
    assert_eq!(store.data(), &[77], "the callback names its own merge");
}

/// The `ModuleBuildInputs` shape every inline-region test shares: a fixed
/// frame, no nursery, no census, and every base at zero.  Only the owner trace
/// and the regions merged into it vary between them, so a new field lands here
/// once instead of at each call site.
fn inline_region_inputs(
    inputargs: &[InputArgRc],
    ops: Vec<Op>,
    inlined_bridges: Vec<codegen::InlinedBridge>,
) -> codegen::ModuleBuildInputs {
    codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops,
        inlined_bridges,
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    }
}

#[test]
fn inlined_bridge_without_owner_loop_label_declines() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let guard = make_guard(
        OpCode::GuardTrue,
        &[OpRef::input_arg_int(0)],
        &[OpRef::input_arg_int(0)],
    );
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]);
    let inputs = inline_region_inputs(
        &inputargs,
        vec![guard, finish],
        vec![codegen::InlinedBridge {
            external_jump: None,
            source_fail_index: 0,
            outside_loop: false,
            trace_id: 1,
            inputargs: vec![InputArg::from_type_rc(Type::Int, 1)],
            ops: vec![Op::new(OpCode::Finish, &[])],
            gc_table_base: 0,
            gc_const_keys: Vec::new(),
            constants: indexmap::IndexMap::new(),
        }],
    );

    let error = match codegen::build_wasm_module(&inputs) {
        Ok(_) => panic!("a label-less owner cannot accept an inlined bridge"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("no local loop LABEL"));
}

fn test_ca_token() -> std::sync::Arc<majit_backend::JitCellToken> {
    std::sync::Arc::new(majit_backend::JitCellToken::new(0x5a5a))
}

/// A CALL_ASSEMBLER descr carrying a callee token, which the stock
/// `SimpleCallDescr` cannot: its `call_target_token` answers `None`, and a
/// merge declines on the missing token before it ever consults `ca.targets`.
#[derive(Debug)]
struct TargetTokenCallDescr {
    arg_types: Vec<Type>,
    result_type: Type,
    token: std::sync::Arc<majit_backend::JitCellToken>,
}

impl majit_ir::Descr for TargetTokenCallDescr {
    fn index(&self) -> u32 {
        u32::MAX
    }

    fn as_call_descr(&self) -> Option<&dyn majit_ir::descr::CallDescr> {
        Some(self)
    }

    fn as_loop_token_descr(&self) -> Option<&dyn majit_ir::LoopTokenDescr> {
        Some(self)
    }
}

impl majit_ir::LoopTokenDescr for TargetTokenCallDescr {
    fn loop_token_number(&self) -> i64 {
        self.token.number()
    }

    fn token_handle_any(&self) -> Option<&dyn std::any::Any> {
        Some(&self.token)
    }
}

impl majit_ir::descr::CallDescr for TargetTokenCallDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }

    fn result_type(&self) -> Type {
        self.result_type
    }

    fn result_size(&self) -> usize {
        8
    }

    fn call_target_token(&self) -> Option<i64> {
        Some(self.token.number())
    }

    fn get_extra_info(&self) -> &EffectInfo {
        static INFO: EffectInfo = EffectInfo::const_new(
            majit_ir::ExtraEffect::CanRaise,
            majit_ir::OopSpecIndex::None,
        );
        &INFO
    }
}

/// A region merged into an owner brings its own CALL_ASSEMBLER callee, but the
/// dedicated CA arm is selected by `ca.emit_ca` and bakes the callee geometry
/// out of `ca.targets` — both decided when the OWNER was compiled. An op that
/// misses that arm does not fail: it falls through to the ordinary
/// residual-call arm, which lowers arg 0 as an `__indirect_function_table`
/// slot, while a CALL_ASSEMBLER's arg 0 is the callee's first frame slot. That
/// calls whatever the slot happens to index and hands the answer back as the
/// callee's — a silent wrong result rather than a trap. Every trace's own ops
/// are screened for unsupported opcodes before compilation; the merged stream
/// is the one place that question is never re-asked, so the merge asks it.
///
/// Both ways of missing the arm are covered, and they are NOT the same test:
/// with no arm emitted at all the op never reaches `ca.targets`, while an owner
/// that does emit the arm still has no geometry for a callee it never compiled
/// against — and that second case would otherwise reach the CA arm's
/// `expect("CA op target must be registered")`.
#[test]
fn inlined_bridge_carrying_an_unarmed_call_assembler_declines() {
    fn build(
        region_ops: Vec<Op>,
        ca: codegen::CaParams,
    ) -> Result<Vec<u8>, majit_backend::BackendError> {
        let inputargs = vec![
            InputArg::from_type_rc(Type::Int, 0),
            InputArg::from_type_rc(Type::Int, 1),
        ];
        let owner_ops = vec![
            Op::new(
                OpCode::Label,
                &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_int(1))],
            ),
            make_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::const_int(1)],
                OpRef::int_op(2),
            ),
            make_guard(
                OpCode::GuardTrue,
                &[OpRef::int_op(2)],
                &[OpRef::int_op(2), OpRef::input_arg_int(1)],
            ),
            Op::new(
                OpCode::Jump,
                &[rb(OpRef::int_op(2)), rb(OpRef::input_arg_int(1))],
            ),
        ];
        let inputs = codegen::ModuleBuildInputs {
            inputargs: inputargs.to_vec(),
            ops: owner_ops,
            inlined_bridges: vec![codegen::InlinedBridge {
                external_jump: None,
                source_fail_index: 0,
                outside_loop: false,
                trace_id: 7,
                inputargs: vec![
                    InputArg::from_type_rc(Type::Int, 40),
                    InputArg::from_type_rc(Type::Int, 41),
                ],
                ops: region_ops,
                gc_table_base: 0,
                gc_const_keys: Vec::new(),
                constants: indexmap::IndexMap::new(),
            }],
            constants: indexmap::IndexMap::new(),
            vtable_offset: Some(0),
            classptr_to_typeid: HashMap::new(),
            guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
            alloc: codegen::AllocHelpers::default(),
            wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
            nursery: None,
            invalidated_flag_addr: 0,
            gc_table_base: 0,
            gc_const_keys: Vec::new(),
            fail_index_base: 0,
            bridge_cells_base: 0,
            guard_cell_addrs: Vec::new(),
            bridge_entry_arity: None,
            bridge_param_dispatch: false,
            trace_entry_census: None,
            inline_trip: None,
            external_jump_slot: 0,
            external_jump_wide_slot: 0,
            external_jump_key: 0,
            frame: codegen::FrameGeometry::fixed(),
            ca,
        };
        codegen::build_wasm_module(&inputs).map(|(bytes, _, _, _)| bytes)
    }

    /// The region the decline arms use, with `opcode` producing its one value.
    /// The loop-closing JUMP carries the region's own inputs, so the result
    /// type never reaches the owner's label and only the opcode varies.
    ///
    /// `token` gives the op a callee the owner could in principle have an arm
    /// for; without it the merge declines on the missing token and never
    /// consults `ca.targets` at all.
    fn region_ops(
        opcode: OpCode,
        result: OpRef,
        token: Option<std::sync::Arc<majit_backend::JitCellToken>>,
    ) -> Vec<Op> {
        let call = make_op(
            opcode,
            &[OpRef::input_arg_int(40), OpRef::input_arg_int(41)],
            result,
        );
        if let Some(token) = token {
            call.setdescr(std::sync::Arc::new(TargetTokenCallDescr {
                arg_types: vec![Type::Int],
                result_type: opcode.result_type(),
                token,
            }));
        }
        vec![
            call,
            Op::new(
                OpCode::Jump,
                &[rb(OpRef::input_arg_int(40)), rb(OpRef::input_arg_int(41))],
            ),
        ]
    }

    // The same region under an ordinary opcode, to pin that what declines
    // below is the CALL_ASSEMBLER and not the fixture.
    let plain = build(
        region_ops(OpCode::IntAdd, OpRef::int_op(42), None),
        codegen::CaParams::default(),
    )
    .expect("a loop-closing region with no CALL_ASSEMBLER merges into its owner");
    validate_wasm(&plain);

    for (opcode, result) in [
        (OpCode::CallAssemblerI, OpRef::int_op(42)),
        (OpCode::CallAssemblerR, OpRef::ref_op(42)),
    ] {
        for (case, token, ca) in [
            // No CA arm at all, so no table to consult.
            (
                "no arm",
                Some(test_ca_token()),
                codegen::CaParams::default(),
            ),
            // The arm is emitted, but this callee is not one the owner
            // compiled against, so its geometry is absent from `ca.targets`.
            (
                "arm without this callee",
                Some(test_ca_token()),
                codegen::CaParams {
                    emit_ca: true,
                    ..codegen::CaParams::default()
                },
            ),
            // A CALL_ASSEMBLER that names no callee at all.
            ("no callee token", None, codegen::CaParams::default()),
        ] {
            let error = match build(region_ops(opcode, result, token), ca) {
                Ok(_) => {
                    panic!("{opcode:?} with {case} must decline, not build a module")
                }
                Err(error) => error,
            };
            assert!(
                error.to_string().contains("no CALL_ASSEMBLER arm for"),
                "{case}: declined for the wrong reason: {error}"
            );
        }
    }
}

#[test]
fn call_assembler_accepts_float_and_void_result_locals() {
    fn build(opcode: OpCode, result: OpRef, input_type: Type) -> Vec<u8> {
        let token = test_ca_token();
        let inputargs = vec![InputArg::from_type_rc(input_type, 0)];
        let call = make_op(opcode, &[OpRef::input_arg_typed(0, input_type)], result);
        call.setdescr(std::sync::Arc::new(TargetTokenCallDescr {
            arg_types: vec![input_type],
            result_type: opcode.result_type(),
            token: std::sync::Arc::clone(&token),
        }));
        let finish_args = (result != OpRef::NONE).then(|| rb(result));
        let ops = vec![
            call,
            Op::new(
                OpCode::Finish,
                finish_args.as_ref().map_or(&[][..], std::slice::from_ref),
            ),
        ];
        let inputs = codegen::ModuleBuildInputs {
            inputargs,
            ops,
            inlined_bridges: Vec::new(),
            constants: indexmap::IndexMap::new(),
            vtable_offset: Some(0),
            classptr_to_typeid: HashMap::new(),
            guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
            alloc: codegen::AllocHelpers::default(),
            wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
            nursery: None,
            invalidated_flag_addr: 0,
            gc_table_base: 0,
            gc_const_keys: Vec::new(),
            fail_index_base: 0,
            bridge_cells_base: 0,
            guard_cell_addrs: Vec::new(),
            bridge_entry_arity: None,
            bridge_param_dispatch: false,
            trace_entry_census: None,
            inline_trip: None,
            external_jump_slot: 0,
            external_jump_wide_slot: 0,
            external_jump_key: 0,
            frame: codegen::FrameGeometry::fixed(),
            ca: codegen::CaParams {
                emit_ca: true,
                targets: HashMap::from([(
                    token.identity_key(),
                    codegen::CaTarget {
                        dispatch_entry: 1024,
                    },
                )]),
                deopt_helper_slot: 1,
                ca_push_fn_ptr: 2,
                ca_pop_fn_ptr: 3,
                ca_reload_fn_ptr: 4,
                ca_reload_caller_fn_ptr: 5,
                ..codegen::CaParams::default()
            },
        };
        codegen::build_wasm_module(&inputs)
            .expect("Float/Void CALL_ASSEMBLER should compile")
            .0
    }

    let float = build(OpCode::CallAssemblerF, OpRef::float_op(1), Type::Int);
    validate_wasm(&float);
    let mut float_reinterprets = 0;
    count_operators(&float, |op| {
        if matches!(op, wasmparser::Operator::F64ReinterpretI64) {
            float_reinterprets += 1;
        }
    });
    assert_eq!(float_reinterprets, 1);

    let void = build(OpCode::CallAssemblerN, OpRef::NONE, Type::Int);
    validate_wasm(&void);
}

#[test]
fn call_assembler_inlines_malloc_cond_varsize_frame() {
    let token = test_ca_token();
    let nursery_free = 0x1000_u32;
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let frame_ref = OpRef::int_op(2);
    let malloc = make_op(
        OpCode::CallMallocNurseryVarsizeFrame,
        &[OpRef::input_arg_int(0)],
        frame_ref,
    );
    let call = make_op(OpCode::CallAssemblerI, &[frame_ref], OpRef::int_op(1));
    call.setdescr(std::sync::Arc::new(TargetTokenCallDescr {
        arg_types: vec![Type::Int],
        result_type: Type::Int,
        token: std::sync::Arc::clone(&token),
    }));
    let ops = vec![
        malloc,
        call,
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]),
    ];
    let inputs = codegen::ModuleBuildInputs {
        inputargs,
        ops,
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers {
            new_fn_ptr: 0x11,
            ..codegen::AllocHelpers::default()
        },
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: Some(codegen::NurseryAllocParams {
            free_addr: nursery_free,
            top_addr: 0x1008,
            large_threshold: 4096,
            plain_tids: std::collections::HashSet::new(),
        }),
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::compact(4, 1, 0),
        ca: codegen::CaParams {
            emit_ca: true,
            targets: HashMap::from([(
                token.identity_key(),
                codegen::CaTarget {
                    dispatch_entry: 1024,
                },
            )]),
            deopt_helper_slot: 1,
            ca_push_fn_ptr: 2,
            ca_pop_fn_ptr: 3,
            ca_reload_fn_ptr: 4,
            ca_reload_caller_fn_ptr: 5,
            inline: Some(codegen::CaInlineParams {
                nursery_free_addr: nursery_free,
                nursery_top_addr: 0x1008,
                jf_top_addr: 0x2000,
                jf_limit_addr: 0x2008,
                jitframe_tid: 7,
                large_threshold: 4096,
            }),
            ..codegen::CaParams::default()
        },
    };
    let bytes = codegen::build_wasm_module(&inputs)
        .expect("inline CA frame alloc should compile")
        .0;
    validate_wasm(&bytes);
    let mut saw_nursery_free = false;
    let mut last_i32 = None;
    let mut fill_lengths = Vec::new();
    count_operators(&bytes, |op| match op {
        wasmparser::Operator::I32Const { value } => {
            if *value == nursery_free as i32 {
                saw_nursery_free = true;
            }
            last_i32 = Some(*value);
        }
        wasmparser::Operator::MemoryFill { .. } => {
            fill_lengths.push(last_i32.expect("memory.fill without a preceding i32.const"));
        }
        _ => {}
    });
    assert!(
        saw_nursery_free,
        "malloc_cond_varsize_frame must load nursery_free"
    );
    // The CA arm no longer memory.fills the frozen home range; the
    // callee key-0 prologue nulls only marked slots. The bump itself
    // must not refill `ca_frame_bytes`.
    let frame = codegen::FrameGeometry::compact(4, 1, 0);
    assert!(
        !fill_lengths.contains(&(frame.ca_frame_bytes as i32)),
        "inline CA bump must not memory.fill ca_frame_bytes; fills were {fill_lengths:?}"
    );
}

/// Key-0 nulls only the homes `build_home_gcmap` marks. Frozen
/// chain-padding is unmarked, so a 128-slot fill on every
/// CALL_ASSEMBLER is wasted work (`fib_recursive`).
#[test]
fn entry_prologue_does_not_null_unmarked_home_padding() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))])];
    let frame = codegen::FrameGeometry::compact(64, 128, 2);
    let home_fill_bytes = (frame.home_slots as i32) * 8;
    let inputs = codegen::ModuleBuildInputs {
        inputargs,
        ops,
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame,
        ca: codegen::CaParams {
            home_gcmap_ptr: 0x1000,
            ..codegen::CaParams::default()
        },
    };
    let bytes = codegen::build_wasm_module(&inputs)
        .expect("frozen-home entry prologue should compile")
        .0;
    validate_wasm(&bytes);
    let mut last_i32 = None;
    let mut fill_lengths = Vec::new();
    count_operators(&bytes, |op| match op {
        wasmparser::Operator::I32Const { value } => last_i32 = Some(*value),
        wasmparser::Operator::MemoryFill { .. } => {
            fill_lengths.push(last_i32.expect("memory.fill without a preceding i32.const"));
        }
        _ => {}
    });
    assert!(
        !fill_lengths.contains(&home_fill_bytes),
        "key-0 must not memory.fill unmarked frozen padding ({home_fill_bytes} bytes); fills were {fill_lengths:?}"
    );
    assert!(
        !fill_lengths.contains(&(frame.ca_frame_bytes as i32)),
        "must not refill ca_frame_bytes; fills were {fill_lengths:?}"
    );
}

/// The CA pop footer must not bake the caller's `GUARD_NOT_FORCED_2`.
/// After `redirect_call_assembler` the callee can differ, so the footer
/// loads `WasmCaDispatchEntry.has_guard_not_forced_2`, and the pop itself
/// is the shadowstack `SUB` (`_call_footer_shadowstack`) with no helper
/// call. A leftover caller `GUARD_NOT_FORCED` must not change that shape. The flag lives on the
/// dispatch cell so an in-flight pop sees a GNF2 bridge attached after
/// the pre-call snapshot was loaded.
#[test]
fn call_assembler_pop_reads_callee_gnf2_from_snapshot() {
    fn build(with_caller_gnf2: bool) -> Vec<u8> {
        let token = test_ca_token();
        let ca_pop_fn_ptr = 0x77_i64;
        let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
        let call = make_op(
            OpCode::CallAssemblerI,
            &[OpRef::input_arg_int(0)],
            OpRef::int_op(1),
        );
        call.setdescr(std::sync::Arc::new(TargetTokenCallDescr {
            arg_types: vec![Type::Int],
            result_type: Type::Int,
            token: std::sync::Arc::clone(&token),
        }));
        let mut ops = vec![call];
        if with_caller_gnf2 {
            ops.push(make_guard(
                OpCode::GuardNotForced2,
                &[],
                &[OpRef::int_op(1)],
            ));
        }
        ops.push(Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]));
        let inputs = codegen::ModuleBuildInputs {
            inputargs,
            ops,
            inlined_bridges: Vec::new(),
            constants: indexmap::IndexMap::new(),
            vtable_offset: Some(0),
            classptr_to_typeid: HashMap::new(),
            guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
            alloc: codegen::AllocHelpers::default(),
            wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
            nursery: None,
            invalidated_flag_addr: 0,
            gc_table_base: 0,
            gc_const_keys: Vec::new(),
            fail_index_base: 0,
            bridge_cells_base: 0,
            guard_cell_addrs: Vec::new(),
            bridge_entry_arity: None,
            bridge_param_dispatch: false,
            trace_entry_census: None,
            inline_trip: None,
            external_jump_slot: 0,
            external_jump_wide_slot: 0,
            external_jump_key: 0,
            frame: codegen::FrameGeometry::fixed(),
            ca: codegen::CaParams {
                emit_ca: true,
                targets: HashMap::from([(
                    token.identity_key(),
                    codegen::CaTarget {
                        dispatch_entry: 1024,
                    },
                )]),
                deopt_helper_slot: 1,
                ca_push_fn_ptr: 2,
                ca_pop_fn_ptr,
                ca_reload_fn_ptr: 4,
                ca_reload_caller_fn_ptr: 5,
                inline: Some(codegen::CaInlineParams {
                    nursery_free_addr: 0x1000,
                    nursery_top_addr: 0x1008,
                    jf_top_addr: 0x2000,
                    jf_limit_addr: 0x2008,
                    jitframe_tid: 7,
                    large_threshold: 4096,
                }),
                ..codegen::CaParams::default()
            },
        };
        codegen::build_wasm_module(&inputs)
            .expect("inline CA pop should compile")
            .0
    }

    fn mentions_pop_helper(bytes: &[u8]) -> bool {
        let mut saw = false;
        count_operators(bytes, |op| {
            if matches!(op, wasmparser::Operator::I32Const { value } if *value == 0x77) {
                saw = true;
            }
        });
        saw
    }

    fn loads_callee_gnf2(bytes: &[u8]) -> bool {
        let mut saw = false;
        count_operators(bytes, |op| {
            if matches!(
                op,
                wasmparser::Operator::I32Load { memarg }
                    if memarg.offset == majit_backend_wasm::failguard::WASM_CA_DISPATCH_HAS_GNF2_OFS
            ) {
                saw = true;
            }
        });
        saw
    }

    let without = build(false);
    validate_wasm(&without);
    assert!(
        loads_callee_gnf2(&without),
        "CA pop must load callee GUARD_NOT_FORCED_2 from the snapshot"
    );
    assert!(
        !mentions_pop_helper(&without),
        "the inline CA pop is `_call_footer_shadowstack`'s SUB, not a helper call"
    );

    let with = build(true);
    validate_wasm(&with);
    assert!(
        loads_callee_gnf2(&with),
        "caller GUARD_NOT_FORCED_2 must not replace the snapshot load"
    );
    assert!(
        !mentions_pop_helper(&with),
        "caller GUARD_NOT_FORCED_2 must not add a pop helper call"
    );
}

/// A region's value ids are its own trace's, so they collide with the owner's.
/// The merged stream has one local namespace, so an unrebased collision makes
/// the region's entry moves land in locals the owner still holds live across
/// the back edge. Where a region's numbering happens to start must therefore
/// not be observable in the emitted code.
#[test]
fn inlined_bridge_emission_is_independent_of_the_regions_own_numbering() {
    fn owner_ops() -> Vec<Op> {
        vec![
            // Defined before the LABEL and read after it, so it is live across
            // the back edge and is restored only on preamble/resume entry.
            make_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
                OpRef::int_op(2),
            ),
            Op::new(
                OpCode::Label,
                &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_int(1))],
            ),
            make_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::const_int(1)],
                OpRef::int_op(3),
            ),
            make_op(
                OpCode::IntLt,
                &[OpRef::int_op(3), OpRef::const_int(10)],
                OpRef::int_op(4),
            ),
            make_guard(
                OpCode::GuardTrue,
                &[OpRef::int_op(4)],
                &[OpRef::int_op(3), OpRef::input_arg_int(1)],
            ),
            make_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(1), OpRef::int_op(2)],
                OpRef::int_op(5),
            ),
            make_op(
                OpCode::IntLt,
                &[OpRef::int_op(5), OpRef::const_int(1000)],
                OpRef::int_op(6),
            ),
            make_guard(
                OpCode::GuardTrue,
                &[OpRef::int_op(6)],
                &[OpRef::int_op(3), OpRef::int_op(5)],
            ),
            Op::new(OpCode::Jump, &[rb(OpRef::int_op(3)), rb(OpRef::int_op(5))]),
        ]
    }

    // `base` picks where the region numbers its own values. `base = 2` makes
    // its first input arg share an id with the owner's loop-invariant
    // `int_op(2)`; `base = 40` clears every owner id.
    fn build(base: u32) -> Vec<u8> {
        let region_pool = indexmap::IndexMap::from([(base + 3, 0x5a5a_007)]);
        let first = build_with(base, region_pool.clone(), indexmap::IndexMap::new())
            .expect("a loop-closing region merges into its owner");
        // A region is RETAINED for the next re-emission, and `reemit_loop`
        // rebases the same retained copy every time, so the rebase must leave
        // it untouched. Building twice is what catches a rebase that wrote
        // through to the region it read.
        let second = build_with(base, region_pool, indexmap::IndexMap::new())
            .expect("a retained region rebases identically on re-emission");
        assert_eq!(first, second, "rebasing mutated the retained region");
        first
    }

    fn build_with(
        base: u32,
        region_constants: indexmap::IndexMap<u32, i64>,
        owner_constants: indexmap::IndexMap<u32, i64>,
    ) -> Result<Vec<u8>, majit_backend::BackendError> {
        let inputargs = vec![
            InputArg::from_type_rc(Type::Int, 0),
            InputArg::from_type_rc(Type::Int, 1),
        ];
        // `int_op(base + 3)` has no producing op: it is a folded value that
        // only the region's own constant pool binds, so the merge has to move
        // its pool key by the same offset it moves the read by.
        let region_ops = vec![
            make_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(base), OpRef::input_arg_int(base + 1)],
                OpRef::int_op(base + 2),
            ),
            make_op(
                OpCode::IntAdd,
                &[OpRef::int_op(base + 2), OpRef::int_op(base + 3)],
                OpRef::int_op(base + 4),
            ),
            Op::new(
                OpCode::Jump,
                &[
                    rb(OpRef::int_op(base + 4)),
                    rb(OpRef::input_arg_int(base + 1)),
                ],
            ),
        ];
        let inputs = codegen::ModuleBuildInputs {
            inputargs: inputargs.to_vec(),
            ops: owner_ops(),
            inlined_bridges: vec![codegen::InlinedBridge {
                external_jump: None,
                source_fail_index: 1,
                outside_loop: false,
                trace_id: 7,
                inputargs: vec![
                    InputArg::from_type_rc(Type::Int, base),
                    InputArg::from_type_rc(Type::Int, base + 1),
                ],
                ops: region_ops,
                gc_table_base: 0,
                gc_const_keys: Vec::new(),
                constants: region_constants,
            }],
            constants: owner_constants,
            vtable_offset: Some(0),
            classptr_to_typeid: HashMap::new(),
            guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
            alloc: codegen::AllocHelpers::default(),
            wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
            nursery: None,
            invalidated_flag_addr: 0,
            gc_table_base: 0,
            gc_const_keys: Vec::new(),
            fail_index_base: 0,
            bridge_cells_base: 0,
            guard_cell_addrs: Vec::new(),
            bridge_entry_arity: None,
            bridge_param_dispatch: false,
            trace_entry_census: None,
            inline_trip: None,
            external_jump_slot: 0,
            external_jump_wide_slot: 0,
            external_jump_key: 0,
            frame: codegen::FrameGeometry::fixed(),
            ca: codegen::CaParams::default(),
        };
        codegen::build_wasm_module(&inputs).map(|(bytes, _, _, _)| bytes)
    }

    let colliding = build(2);
    let disjoint = build(40);
    validate_wasm(&colliding);
    validate_wasm(&disjoint);
    assert_eq!(colliding, disjoint);

    // A key the owner pool carries inside the window the region is rebased
    // into names one of the REGION's ids once the merge is done, not the
    // owner value it was recorded for. Left in place it answers a read the
    // region's own pool declines, so the module builds on unrelated bits
    // instead of declining. Dropping the region's seed must therefore reach
    // the decline even though the owner pool has an entry at that position.
    let stale = build_with(
        2,
        indexmap::IndexMap::new(),
        indexmap::IndexMap::from([(12, 0x1234)]),
    );
    let error = match stale {
        Ok(_) => panic!("a stale owner key inside the region window answered the region's read"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("read with no producing op and no"),
        "unexpected decline: {error}"
    );
}

#[test]
fn test_int_add_loop() {
    // Label(i, sum) -> IntAdd(sum, i) -> IntAdd(i, 1) -> IntLt(i, 100)
    // -> GuardTrue -> Jump(new_i, new_sum)
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0), // i
        InputArg::from_type_rc(Type::Int, 1), // sum
    ];

    let const_1 = OpRef::const_int(1);
    let const_100 = OpRef::const_int(100);
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();

    let ops = vec![
        Op::new(
            OpCode::Label,
            &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_int(1))],
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(1), OpRef::input_arg_int(0)],
            OpRef::int_op(2),
        ), // sum + i
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), const_1],
            OpRef::int_op(3),
        ), // i + 1
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), const_100],
            OpRef::int_op(4),
        ), // i+1 < 100
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(4)],
            &[OpRef::int_op(3), OpRef::int_op(2)],
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(3)), rb(OpRef::int_op(2))]),
    ];

    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1); // one guard
    assert!(!guards[0].is_finish);
}

#[test]
fn label_int_phi_without_a_producer_still_compiles() {
    // A body LABEL can carry a virtualstate int box the peel never wrote.
    // Treating that id as unbound declined every wasm loop.
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let const_1 = OpRef::const_int(1);
    let const_100 = OpRef::const_int(100);
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let ops = vec![
        Op::new(
            OpCode::Label,
            &[
                rb(OpRef::input_arg_int(0)),
                rb(OpRef::input_arg_int(1)),
                rb(OpRef::int_op(85)),
            ],
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(1), OpRef::input_arg_int(0)],
            OpRef::int_op(2),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), const_1],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), const_100],
            OpRef::int_op(4),
        ),
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(4)],
            &[OpRef::int_op(3), OpRef::int_op(2), OpRef::int_op(85)],
        ),
        Op::new(
            OpCode::Jump,
            &[
                rb(OpRef::int_op(3)),
                rb(OpRef::int_op(2)),
                rb(OpRef::int_op(85)),
            ],
        ),
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
}

#[test]
fn label_ref_phi_without_a_producer_declines() {
    // A residual virtualizable Ref on the LABEL is not a phi wasm can
    // bind to a local. Compiling it stores null on failarg writeback.
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Ref, 1),
    ];
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let ops = vec![
        Op::new(
            OpCode::Label,
            &[rb(OpRef::input_arg_int(0)), rb(OpRef::ref_op(85))],
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(2), OpRef::const_int(100)],
            OpRef::int_op(3),
        ),
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(3)],
            &[OpRef::int_op(2), OpRef::ref_op(85)],
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(2)), rb(OpRef::ref_op(85))]),
    ];
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants,
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    };
    let error = match codegen::build_wasm_module(&inputs) {
        Ok(_) => panic!("an unbound LABEL Ref must not compile as a null local"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("no producing op"),
        "unexpected decline: {error}"
    );
}

#[test]
fn compute_home_gcmap_simple_loop_is_valid() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let const_1 = OpRef::const_int(1);
    let const_100 = OpRef::const_int(100);
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let ops = vec![
        Op::new(
            OpCode::Label,
            &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_int(1))],
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(1), OpRef::input_arg_int(0)],
            OpRef::int_op(2),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), const_1],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), const_100],
            OpRef::int_op(4),
        ),
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(4)],
            &[OpRef::int_op(3), OpRef::int_op(2)],
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(3)), rb(OpRef::int_op(2))]),
    ];
    let mut ca = codegen::CaParams::default();
    ca.compute_home_gcmap = true;
    let frame = codegen::FrameGeometry::compact(64, 128 + 2, 2);
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants: constants.clone(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame,
        ca,
    };
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("compute_home_gcmap loop should build");
    validate_wasm(&bytes);
}

#[test]
fn home_gcmap_union_call_validates_on_a_reload_only_residual_family() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let const_1 = OpRef::const_int(1);
    let const_100 = OpRef::const_int(100);
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let ops = vec![
        Op::new(
            OpCode::Label,
            &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_int(1))],
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(1), OpRef::input_arg_int(0)],
            OpRef::int_op(2),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), const_1],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), const_100],
            OpRef::int_op(4),
        ),
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(4)],
            &[OpRef::int_op(3), OpRef::int_op(2)],
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(3)), rb(OpRef::int_op(2))]),
    ];
    let mut ca = codegen::CaParams::default();
    ca.compute_home_gcmap = true;
    ca.ca_reload_fn_ptr = 1;
    let frame = codegen::FrameGeometry::compact(64, 128 + 2, 2);
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants,
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame,
        ca,
    };
    let (bytes, _, _, _) = codegen::build_wasm_module(&inputs)
        .expect("reload-only residual family must declare arity 2 for union");
    validate_wasm(&bytes);
}

/// assembler.py `push_gcmap` stores this site's map. The host union is
/// only for incomparable ordinary/LABEL extents; a live map that already
/// covers these bits must not `call_indirect` it. The pointer-eq sits in
/// front of that call so a keyed resume into the same module is a guest
/// compare.
#[test]
fn home_gcmap_publish_pointer_eq_guards_the_union_call() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let const_1 = OpRef::const_int(1);
    let const_100 = OpRef::const_int(100);
    let ops = vec![
        Op::new(
            OpCode::Label,
            &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_int(1))],
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(1), OpRef::input_arg_int(0)],
            OpRef::int_op(2),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), const_1],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), const_100],
            OpRef::int_op(4),
        ),
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(4)],
            &[OpRef::int_op(3), OpRef::int_op(2)],
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(3)), rb(OpRef::int_op(2))]),
    ];
    let mut ca = codegen::CaParams::default();
    ca.compute_home_gcmap = true;
    ca.ca_reload_fn_ptr = 1;
    let frame = codegen::FrameGeometry::compact(64, 128 + 2, 2);
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame,
        ca,
    };
    let (bytes, _, _, _) = codegen::build_wasm_module(&inputs)
        .expect("cover-check publish must still declare the union residual");
    validate_wasm(&bytes);
    let mut ptr_eqs = 0usize;
    let mut union_calls = 0usize;
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            for op in body.get_operators_reader().unwrap() {
                match op.unwrap() {
                    wasmparser::Operator::I64Eq => ptr_eqs += 1,
                    wasmparser::Operator::CallIndirect { .. } => union_calls += 1,
                    _ => {}
                }
            }
        }
    }
    let _ = ptr_eqs;
    assert_eq!(
        union_calls, 0,
        "per-site gcmaps replace the prologue union of a static home map"
    );
}

/// A re-emission that grew the LABEL-capture tail must still build when
/// asked to null only the newly marked slots before publishing.
#[test]
fn reemit_nulls_grown_label_homes_and_builds() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Ref, 1),
    ];
    let ops = vec![
        Op::new(OpCode::Label, &[rb(OpRef::input_arg_ref(0))]),
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::const_int(1)],
            &[OpRef::input_arg_ref(0), OpRef::input_arg_ref(1)],
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::input_arg_ref(0))]),
    ];
    let mut ca = codegen::CaParams::default();
    ca.compute_home_gcmap = true;
    ca.home_gcmap_has_prior = true;
    ca.home_gcmap_null_grown_labels = true;
    ca.home_gcmap_min_labels = 0;
    let frame = codegen::FrameGeometry::compact(16, 128 + 2, 2);
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame,
        ca,
    };
    let (bytes, _, _, used_labels) =
        codegen::build_wasm_module(&inputs).expect("re-emission grown LABEL tail should build");
    validate_wasm(&bytes);
    assert!(used_labels >= 1);
}

/// A peeled loop's start LABEL can still name an import_state source
/// (`InputArgRef(128)` on attr_delete) after every real use was rewritten
/// to the folded constant. That argument is the phi destination, not a
/// read of a never-written local. A later producer past 128 used to make
/// `unbound_pool_const_seeds` decline the whole module.
#[test]
fn label_arg_import_hole_is_not_an_unbound_read() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let const_1 = OpRef::const_int(1);
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let ops = vec![
        Op::new(
            OpCode::Label,
            &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_ref(128))],
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), const_1],
            OpRef::int_op(200),
        ),
        Op::new(
            OpCode::Jump,
            &[rb(OpRef::int_op(200)), rb(OpRef::input_arg_int(1))],
        ),
    ];
    let mut ca = codegen::CaParams::default();
    ca.compute_home_gcmap = true;
    let frame = codegen::FrameGeometry::compact(64, 128 + 2, 2);
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants: constants.clone(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame,
        ca,
    };
    let (bytes, _, _, _) = codegen::build_wasm_module(&inputs)
        .expect("stale LABEL import hole must not decline the module");
    validate_wasm(&bytes);
}

/// A real read of an unbound hole (not a LABEL phi name) is still declined.
#[test]
fn unbound_non_label_read_of_import_hole_is_declined() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let ops = vec![make_op(
        OpCode::IntAdd,
        &[OpRef::input_arg_int(0), OpRef::input_arg_ref(128)],
        OpRef::int_op(200),
    )];
    let err = match codegen::build_wasm_module(&codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants: constants.clone(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::compact(16, 8, 0),
        ca: codegen::CaParams::default(),
    }) {
        Err(err) => err,
        Ok(_) => panic!("a real read of the hole must still decline"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("InputArgRef(128)") && msg.contains("read with no producing op and no"),
        "expected unbound 128 decline, got {msg}"
    );
}

#[test]
fn test_folded_producer_value_seeds_unbound_local() {
    // The producer at pos 99 is not in the compiled stream. Its `_resint`
    // still holds the folded constant (`history.py IntFrontendOp`). Flattening
    // to `IntOp(99)` without a pool entry used to decline the module.
    let producer = OpRc::new(Op::new(OpCode::IntAdd, &[]));
    producer.pos().set(OpRef::int_op(99));
    let folded = Operand::from_bound_op(&producer);
    folded.set_forwarded_const(majit_ir::Const::Int(7));

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let add = Op::new(OpCode::IntAdd, &[folded, rb(OpRef::input_arg_int(0))]);
    add.pos().set(OpRef::int_op(1));
    let ops = vec![add, Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))])];
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, _) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(
        execute_simple_trace(&bytes, &[5]),
        12,
        "folded 7 + input 5 must execute as 12"
    );
}

#[test]
fn test_folded_producer_ref_is_interned() {
    // Same hole as the scalar case, but a leftover `_resref` must become
    // `LoadFromGcTable` (`rewrite.py remove_constptr`), not a baked pointer.
    let producer = OpRc::new(Op::new(OpCode::SameAsR, &[]));
    producer.pos().set(OpRef::ref_op(99));
    let folded = Operand::from_bound_op(&producer);
    folded.set_forwarded_const(majit_ir::Const::from_gcref(majit_ir::GcRef(0x1000)));

    let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
    let same = Op::new(OpCode::SameAsR, &[folded]);
    same.pos().set(OpRef::ref_op(1));
    let ops = vec![same, Op::new(OpCode::Finish, &[rb(OpRef::ref_op(1))])];
    // `compile_loop` rewrite intern runs this pass first; a leftover
    // `_resref` must become LoadFromGcTable before unbound-seed.
    let (ops, gcrefs) = majit_gc::rewrite::remove_ref_constants(&ops, 2);
    assert_eq!(gcrefs, vec![majit_ir::GcRef(0x1000)]);
    let load = ops
        .iter()
        .find(|op| op.opcode == OpCode::LoadFromGcTable)
        .expect("remove_constptr must emit LoadFromGcTable");
    assert_eq!(
        load.arg(0).const_value(),
        Some(majit_ir::Value::Int(0)),
        "LoadFromGcTable must use table index 0, ops={ops:?}"
    );
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, _) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
}

#[test]
fn test_stray_label_inputarg_is_seeded() {
    // Loop LABEL / JUMP carry an InputArg that is not in the token input
    // list (peeled fallthrough leftover). Must compile, not decline.
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Ref, 1),
    ];
    let ops = vec![
        Op::new(
            OpCode::Label,
            &[
                rb(OpRef::input_arg_ref(0)),
                rb(OpRef::input_arg_ref(1)),
                rb(OpRef::input_arg_ref(99)),
            ],
        ),
        make_op(
            OpCode::SameAsR,
            &[OpRef::input_arg_ref(0)],
            OpRef::ref_op(2),
        ),
        Op::new(
            OpCode::Jump,
            &[
                rb(OpRef::input_arg_ref(0)),
                rb(OpRef::input_arg_ref(1)),
                rb(OpRef::input_arg_ref(99)),
            ],
        ),
    ];
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, _) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
}

#[test]
fn test_external_jump_leftover_inputarg_declines() {
    // A leftover InputArg on a JUMP that does not land on a LABEL in this
    // stream is a live transfer. Seeding 0 would hand the target null.
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Ref, 1),
    ];
    let jump = Op::new(
        OpCode::Jump,
        &[
            rb(OpRef::input_arg_ref(0)),
            rb(OpRef::input_arg_ref(1)),
            rb(OpRef::input_arg_ref(99)),
        ],
    );
    jump.setdescr(majit_ir::descr::make_call_descr_full(
        99,
        vec![],
        Type::Void,
        false,
        8,
        EffectInfo::default(),
    ));
    let ops = vec![
        Op::new(
            OpCode::Label,
            &[rb(OpRef::input_arg_ref(0)), rb(OpRef::input_arg_ref(1))],
        ),
        jump,
    ];
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let err = match codegen::build_wasm_module(&codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops,
        inlined_bridges: Vec::new(),
        constants,
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    }) {
        Err(err) => err,
        Ok(_) => panic!("an external JUMP leftover must decline, not seed 0"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("no producing op")
            && (msg.contains("value[99]") || msg.contains("InputArgRef(99)")),
        "must decline the leftover JUMP InputArg, got {msg}"
    );
}

#[test]
fn test_stray_failarg_ref_declines() {
    // A failarg InputArg that is not a token input and not a LABEL
    // live-in has no producer. #1780 treats LABEL args as defined;
    // a snapshot-only leftover is still a null identity at deopt.
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Ref, 1),
    ];
    let guard = make_guard(
        OpCode::GuardTrue,
        &[OpRef::input_arg_ref(0)],
        &[
            OpRef::input_arg_ref(0),
            OpRef::input_arg_ref(1),
            OpRef::input_arg_ref(99),
        ],
    );
    let ops = vec![
        Op::new(
            OpCode::Label,
            &[rb(OpRef::input_arg_ref(0)), rb(OpRef::input_arg_ref(1))],
        ),
        guard,
        Op::new(
            OpCode::Jump,
            &[rb(OpRef::input_arg_ref(0)), rb(OpRef::input_arg_ref(1))],
        ),
    ];
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops,
        inlined_bridges: Vec::new(),
        constants,
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    };
    let error = match codegen::build_wasm_module(&inputs) {
        Ok(_) => panic!("a stray Ref failarg must decline, not compile as null"),
        Err(error) => error,
    };
    let msg = error.to_string();
    assert!(
        (msg.contains("value[99]") || msg.contains("InputArgRef(99)"))
            && msg.contains("no producing op"),
        "decline must name the stray failarg, got {msg}"
    );
}

#[test]
fn test_force_arm_accepts_constptr_failarg() {
    // `emit_force_arm` used to call `OpRef::raw()` on every failarg.
    // An inline `ConstPtr` (null or interned) has no raw index and panicked
    // while compiling a `CallMayForce` + `GuardNotForced` bridge.
    let call = make_op(
        OpCode::CallMayForceI,
        &[OpRef::const_int(42)],
        OpRef::int_op(1),
    );
    call.setdescr(majit_ir::descr::make_call_descr_full(
        0,
        vec![],
        Type::Int,
        false,
        8,
        EffectInfo::default(),
    ));
    let guard = Op::new(OpCode::GuardNotForced, &[]);
    guard.setfailargs(smallvec![
        rb(OpRef::input_arg_ref(0)),
        rb(OpRef::const_ptr(majit_ir::GcRef(0))),
    ]);
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish.setfailargs(smallvec![rb(OpRef::input_arg_ref(0))]);
    let bytes = build_module_with_write_barrier_target(
        &[InputArg::from_type_rc(Type::Ref, 0)],
        &[call, guard, finish],
        127,
    );
    validate_wasm(&bytes);
}

fn wasm_contains_i64_const(bytes: &[u8], value: i64) -> bool {
    let mut encoded = vec![0x42];
    let mut n = value;
    loop {
        let mut byte = (n as u8) & 0x7f;
        n >>= 7;
        let done = (n == 0 && byte & 0x40 == 0) || (n == -1 && byte & 0x40 != 0);
        if !done {
            byte |= 0x80;
        }
        encoded.push(byte);
        if done {
            break;
        }
    }
    bytes.windows(encoded.len()).any(|window| window == encoded)
}

#[test]
fn test_force_arm_rematerializes_constptr_from_gc_table() {
    // A movable non-null ConstPtr must not be stored as its compile-time
    // address in the untraced force slot. Tag the GC-table slot so
    // `dead_frame_from_forced_frame` reloads after a collection.
    let table_base = 0x2000u32;
    let call = make_op(
        OpCode::CallMayForceI,
        &[OpRef::const_int(42)],
        OpRef::int_op(1),
    );
    call.setdescr(majit_ir::descr::make_call_descr_full(
        0,
        vec![],
        Type::Int,
        false,
        8,
        EffectInfo::default(),
    ));
    let guard = Op::new(OpCode::GuardNotForced, &[]);
    guard.setfailargs(smallvec![
        rb(OpRef::input_arg_ref(0)),
        rb(OpRef::const_ptr(majit_ir::GcRef(0x1000))),
    ]);
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish.setfailargs(smallvec![rb(OpRef::input_arg_ref(0))]);
    let bytes = build_module_with_gc_table(
        &[InputArg::from_type_rc(Type::Ref, 0)],
        &[call, guard, finish],
        127,
        table_base,
        &[0x1000],
    );
    validate_wasm(&bytes);
    let tagged = i64::from(table_base) | 3;
    assert!(
        wasm_contains_i64_const(&bytes, tagged),
        "force slot must publish the tagged GC-table address {tagged:#x}"
    );
    assert!(
        !wasm_contains_i64_const(&bytes, 0x1000),
        "force slot must not bake the compile-time ConstPtr address"
    );
}

#[test]
fn test_force_arm_rebinding_keeps_compile_time_constptr_key() {
    // After a collection forwards the live slot, rebind must still find
    // the leftover ConstPtr by its compile-time address.
    let table = majit_gc::GcTable::from_gcrefs(&[majit_ir::GcRef(0x1000)]);
    table.trace(&mut |r| {
        if r.0 == 0x1000 {
            r.0 = 0x9000;
        }
    });
    let call = make_op(
        OpCode::CallMayForceI,
        &[OpRef::const_int(42)],
        OpRef::int_op(1),
    );
    call.setdescr(majit_ir::descr::make_call_descr_full(
        0,
        vec![],
        Type::Int,
        false,
        8,
        EffectInfo::default(),
    ));
    let guard = Op::new(OpCode::GuardNotForced, &[]);
    guard.setfailargs(smallvec![
        rb(OpRef::input_arg_ref(0)),
        rb(OpRef::const_ptr(majit_ir::GcRef(0x1000))),
    ]);
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish.setfailargs(smallvec![rb(OpRef::input_arg_ref(0))]);
    let bytes = build_module_with_gc_table(
        &[InputArg::from_type_rc(Type::Ref, 0)],
        &[call, guard, finish],
        127,
        table.base_addr() as u32,
        &[table.compile_key(0)],
    );
    validate_wasm(&bytes);
    let tagged = i64::from(table.base_addr() as u32) | 3;
    assert!(
        wasm_contains_i64_const(&bytes, tagged),
        "rebind must still tag the table slot for the original ConstPtr"
    );
    assert!(
        !wasm_contains_i64_const(&bytes, 0x1000),
        "must not fall through to the compile-time address"
    );
    assert!(
        !wasm_contains_i64_const(&bytes, 0x9000),
        "must not key the map on the forwarded slot"
    );
}

#[test]
fn test_force_arm_lookup_stays_on_the_emitting_table() {
    // Two tables can share a compile-time nursery address after a move.
    // Force-arm must tag the emitting region's slot, not the later table's.
    let owner = majit_gc::GcTable::from_gcrefs(&[majit_ir::GcRef(0x1000)]);
    owner.trace(&mut |r| {
        if r.0 == 0x1000 {
            r.0 = 0x9000;
        }
    });
    let later = majit_gc::GcTable::from_gcrefs(&[majit_ir::GcRef(0x1000)]);
    let call = make_op(
        OpCode::CallMayForceI,
        &[OpRef::const_int(42)],
        OpRef::int_op(1),
    );
    call.setdescr(majit_ir::descr::make_call_descr_full(
        0,
        vec![],
        Type::Int,
        false,
        8,
        EffectInfo::default(),
    ));
    let guard = Op::new(OpCode::GuardNotForced, &[]);
    guard.setfailargs(smallvec![
        rb(OpRef::input_arg_ref(0)),
        rb(OpRef::const_ptr(majit_ir::GcRef(0x1000))),
    ]);
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish.setfailargs(smallvec![rb(OpRef::input_arg_ref(0))]);
    let bytes = build_module_with_gc_table(
        &[InputArg::from_type_rc(Type::Ref, 0)],
        &[call, guard, finish],
        127,
        owner.base_addr() as u32,
        &[owner.compile_key(0)],
    );
    validate_wasm(&bytes);
    let owner_tagged = i64::from(owner.base_addr() as u32) | 3;
    let later_tagged = i64::from(later.base_addr() as u32) | 3;
    assert!(
        wasm_contains_i64_const(&bytes, owner_tagged),
        "owner emit must tag the owner table"
    );
    assert!(
        !wasm_contains_i64_const(&bytes, later_tagged),
        "a later table with the same compile_key must not win"
    );
}

#[test]
fn test_inputarg_observation_is_not_a_folded_scalar() {
    // An InputArg's get_value() is the sample seen while tracing. Seeding
    // that sample would freeze the first loop counter into later iterations.
    let ia = InputArg::new_int_rc(99);
    ia.set_value(majit_ir::Value::Int(7));
    let add = Op::new(
        OpCode::IntAdd,
        &[
            Operand::from_bound_inputarg(&ia),
            rb(OpRef::input_arg_int(0)),
        ],
    );
    add.pos().set(OpRef::int_op(1));
    let ops = vec![
        Op::new(OpCode::Label, &[rb(OpRef::input_arg_int(0))]),
        add,
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]),
    ];
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let err = match codegen::build_wasm_module(&codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops,
        inlined_bridges: Vec::new(),
        constants,
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    }) {
        Err(err) => err,
        Ok(_) => panic!("a body read of an InputArg observation must decline, not seed 7"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("no producing op")
            && (msg.contains("value[99]") || msg.contains("InputArgInt(99)")),
        "must decline the unbound InputArg, got {msg}"
    );
}

#[test]
fn test_float_ops() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Float, 0),
        InputArg::from_type_rc(Type::Float, 1),
    ];

    let ops = vec![
        make_op(
            OpCode::FloatAdd,
            &[OpRef::input_arg_float(0), OpRef::input_arg_float(1)],
            OpRef::float_op(2),
        ),
        make_op(
            OpCode::FloatSub,
            &[OpRef::input_arg_float(0), OpRef::input_arg_float(1)],
            OpRef::float_op(3),
        ),
        make_op(
            OpCode::FloatMul,
            &[OpRef::float_op(2), OpRef::float_op(3)],
            OpRef::float_op(4),
        ),
        make_op(
            OpCode::FloatTrueDiv,
            &[OpRef::float_op(4), OpRef::input_arg_float(0)],
            OpRef::float_op(5),
        ),
        make_op(OpCode::FloatNeg, &[OpRef::float_op(5)], OpRef::float_op(6)),
        make_op(OpCode::FloatAbs, &[OpRef::float_op(6)], OpRef::float_op(7)),
        make_op(
            OpCode::FloatLt,
            &[OpRef::input_arg_float(0), OpRef::input_arg_float(1)],
            OpRef::int_op(8),
        ),
        {
            let op = Op::new(OpCode::Finish, &[rb(OpRef::float_op(7))]);
            op.setfailargs(smallvec![rb(OpRef::float_op(7))]);
            op
        },
    ];

    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
}

#[test]
fn test_call_generates_import() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];

    let func_ptr = OpRef::const_int(42); // fake func_ptr
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();

    let ops = vec![
        make_op(
            OpCode::CallI,
            &[func_ptr, OpRef::input_arg_int(0)],
            OpRef::int_op(1),
        ),
        {
            let op = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
            op.setfailargs(smallvec![rb(OpRef::int_op(1))]);
            op
        },
    ];

    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);

    // The residual-call trampoline reads its scratch at `base + offset`, and
    // the scratch is the module-static call area, so every emitting module
    // takes the two-argument import.
    let parser = wasmparser::Parser::new(0);
    let mut has_jit_call = false;
    for payload in parser.parse_all(&bytes) {
        if let Ok(wasmparser::Payload::ImportSection(imports)) = payload {
            for import in imports {
                if let Ok(import) = import
                    && import.name == "jit_call_compact"
                {
                    has_jit_call = true;
                }
            }
        }
    }
    assert!(has_jit_call, "module should import jit_call_compact");
}

fn void_call(arg_types: Vec<Type>, args: &[OpRef], result_size: usize) -> Op {
    void_call_with_helper(arg_types, args, result_size, RuntimeHelperKind::None)
}

fn void_call_with_helper(
    arg_types: Vec<Type>,
    args: &[OpRef],
    result_size: usize,
    helper: RuntimeHelperKind,
) -> Op {
    let mut operands = vec![rb(OpRef::const_int(42))];
    operands.extend(args.iter().copied().map(rb));
    let op = Op::new(OpCode::CallN, &operands);
    let effect = EffectInfo {
        runtime_helper: helper,
        ..EffectInfo::default()
    };
    op.setdescr(majit_ir::descr::make_call_descr_full(
        0,
        arg_types,
        Type::Void,
        false,
        result_size,
        effect,
    ));
    op
}

fn import_func_type(bytes: &[u8], name: &str) -> Option<u32> {
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::ImportSection(imports) = payload.unwrap() {
            for import in imports {
                let import = import.unwrap();
                if import.name == name {
                    return match import.ty {
                        wasmparser::TypeRef::Func(type_idx) => Some(type_idx),
                        _ => None,
                    };
                }
            }
        }
    }
    None
}

fn has_table_import(bytes: &[u8]) -> bool {
    wasmparser::Parser::new(0).parse_all(bytes).any(|payload| {
        let Ok(wasmparser::Payload::ImportSection(imports)) = payload else {
            return false;
        };
        imports
            .into_iter()
            .any(|import| import.is_ok_and(|import| import.name == "__indirect_function_table"))
    })
}

fn indirect_call_types_and_drop_count(bytes: &[u8]) -> (Vec<(u32, u32)>, usize) {
    let mut indirect_calls = Vec::new();
    let mut drops = 0;
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::CallIndirect {
                        type_index,
                        table_index,
                        ..
                    } => indirect_calls.push((type_index, table_index)),
                    wasmparser::Operator::Drop => drops += 1,
                    _ => {}
                }
            }
        }
    }
    (indirect_calls, drops)
}

fn function_type(
    bytes: &[u8],
    type_index: usize,
) -> (Vec<wasmparser::ValType>, Vec<wasmparser::ValType>) {
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::TypeSection(types) = payload.unwrap() {
            let ty = types
                .into_iter_err_on_gc_types()
                .nth(type_index)
                .unwrap_or_else(|| panic!("missing function type {type_index}"))
                .unwrap();
            return (ty.params().to_vec(), ty.results().to_vec());
        }
    }
    panic!("module has no type section");
}

fn entry_function_type_index(bytes: &[u8]) -> usize {
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::FunctionSection(functions) = payload.unwrap() {
            return functions
                .into_iter()
                .next()
                .expect("module has an entry function")
                .expect("entry function type is valid") as usize;
        }
    }
    panic!("module has no function section");
}

#[test]
fn zero_arity_parameter_entry_is_structurally_type_zero() {
    // A published LABEL target is entered through type 0, `(i32) -> i32`.
    // The separate type index emitted for a zero-arity parameter bridge must
    // retain that same structural signature.
    let inputargs = Vec::new();
    let ops = vec![Op::new(OpCode::Label, &[]), Op::new(OpCode::Finish, &[])];
    let inputs = codegen::ModuleBuildInputs {
        inputargs,
        ops,
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 4,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: Some(0),
        bridge_param_dispatch: true,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    };
    let (bytes, _, _, _) = codegen::build_wasm_module(&inputs).unwrap();

    validate_wasm(&bytes);
    assert_eq!(
        function_type(&bytes, entry_function_type_index(&bytes)),
        function_type(&bytes, 0),
        "a published LABEL target's module entry must be structurally `(i32) -> i32`"
    );
}

fn parameter_entry_bridge_inputs(
    inputargs: Vec<InputArgRc>,
    ops: Vec<Op>,
    bridge_param_dispatch: bool,
) -> codegen::ModuleBuildInputs {
    codegen::ModuleBuildInputs {
        inputargs,
        ops,
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 4,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: Some(1),
        bridge_param_dispatch,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    }
}

fn mixed_arity_parameter_entry_ops() -> (Vec<InputArgRc>, Vec<Op>) {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::input_arg_int(0)],
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
    ];
    (inputargs, ops)
}

fn return_call_indirect_type_indices(bytes: &[u8]) -> Vec<u32> {
    let mut indices = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while let Ok(op) = operators.read() {
                if let wasmparser::Operator::ReturnCallIndirect { type_index, .. } = op {
                    indices.push(type_index);
                }
            }
        }
    }
    indices
}

#[test]
fn parameter_entry_does_not_arm_guard_param_dispatch() {
    // Assembler386.assemble_bridge rebuilds the entry from the source guard
    // (`rebuild_faillocs_from_descr`) and then write_pending_failure_recoveries
    // still emits generate_quick_failure for every guard of the bridge. The
    // entry arity is not the recovery-stub family: a parameter-entered module
    // whose own `bridge_param_dispatch` is off must use frame-entry dispatch
    // for those guards. Seeding `param_type_indices` with the entry arity
    // made emit_guard_exit look up the GuardTrue's live fail-arg count (2)
    // in a map that only held the entry arity (1).
    let (inputargs, ops) = mixed_arity_parameter_entry_ops();
    let inputs = parameter_entry_bridge_inputs(inputargs, ops, false);
    let (bytes, guards, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 2, "GuardTrue and Finish");
    let types = return_call_indirect_type_indices(&bytes);
    assert!(
        types.iter().any(|&t| t == 0),
        "frame-entry epilogue uses type 0, got {types:?}"
    );
    assert!(
        types.iter().all(|&t| t == 0),
        "no typed guard tail-call when param dispatch is off, got {types:?}"
    );
}

#[test]
fn parameter_entry_registers_distinct_guard_fail_arities() {
    // Same stream with param dispatch on: the GuardTrue's arity-2 recovery
    // is a different wasm type from the arity-1 entry, and both are declared.
    let (inputargs, ops) = mixed_arity_parameter_entry_ops();
    let inputs = parameter_entry_bridge_inputs(inputargs, ops, true);
    let (bytes, guards, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 2, "GuardTrue and Finish");
    let types = return_call_indirect_type_indices(&bytes);
    assert!(
        types.iter().any(|&t| t != 0),
        "arity-2 guard tail-call uses a declared parameter type, got {types:?}"
    );
}

#[test]
fn building_a_module_does_not_stamp_the_guard_cell_on_its_descr() {
    // The caller can still decline after `build_wasm_module` returns (the host
    // rejects the module), and a decline frees the cell array. The descr is
    // shared with the next compile of the same guard, which reuses a nonzero
    // `adr_jump_offset` as its cell, so the stamp waits for acceptance.
    #[derive(Debug)]
    struct CellDescr {
        types: Vec<Type>,
        adr: std::sync::atomic::AtomicUsize,
    }
    impl majit_ir::Descr for CellDescr {
        fn as_fail_descr(&self) -> Option<&dyn majit_ir::FailDescr> {
            Some(self)
        }
    }
    impl majit_ir::FailDescr for CellDescr {
        fn fail_index(&self) -> u32 {
            0
        }
        fn fail_arg_types(&self) -> &[Type] {
            &self.types
        }
        fn adr_jump_offset(&self) -> usize {
            self.adr.load(std::sync::atomic::Ordering::Relaxed)
        }
        fn set_adr_jump_offset(&self, offset: usize) {
            self.adr.store(offset, std::sync::atomic::Ordering::Relaxed);
        }
    }

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let guard = make_guard(
        OpCode::GuardTrue,
        &[OpRef::input_arg_int(0)],
        &[OpRef::input_arg_int(0)],
    );
    let descr = std::sync::Arc::new(CellDescr {
        types: vec![Type::Int],
        adr: std::sync::atomic::AtomicUsize::new(0),
    });
    guard.setdescr(descr.clone());
    let ops = vec![
        guard,
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
    ];
    let inputs = codegen::ModuleBuildInputs {
        inputargs,
        ops,
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 64,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    };
    let (bytes, guards, _, _) = codegen::build_wasm_module(&inputs).unwrap();

    validate_wasm(&bytes);
    assert_eq!(guards[0].bridge_cell, 64, "the guard exit carries its cell");
    assert_eq!(
        descr.adr.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "the descr names no cell until the caller accepts the module"
    );
}

#[test]
fn test_nullary_true_void_call_uses_indirect_call_without_drop() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![
        void_call(vec![], &[], 0),
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
    ];
    let (bytes, guards) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());

    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    assert!(has_table_import(&bytes));
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
    let (indirect_calls, drops) = indirect_call_types_and_drop_count(&bytes);
    assert_eq!(indirect_calls, vec![(1, 0)]);
    assert_eq!(drops, 0, "a genuine () -> () helper needs no result drop");
}

#[test]
fn test_true_void_int_ref_call_uses_void_result_type_without_drop() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Ref, 1),
    ];
    let ops = vec![
        void_call(
            vec![Type::Int, Type::Ref],
            &[OpRef::input_arg_int(0), OpRef::input_arg_ref(1)],
            0,
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
    ];
    let (bytes, guards) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());

    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    assert!(has_table_import(&bytes));
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
    let (indirect_calls, drops) = indirect_call_types_and_drop_count(&bytes);
    assert_eq!(indirect_calls, vec![(3, 0)]);
    assert_eq!(
        function_type(&bytes, 3),
        (
            vec![wasmparser::ValType::I64, wasmparser::ValType::I64],
            vec![]
        )
    );
    assert_eq!(drops, 0, "a genuine void call has no result to drop");
}

#[test]
fn an_allocation_is_followed_by_a_memory_error_check() {
    use majit_ir::descr::SimpleSizeDescr;
    use std::sync::Arc;

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let new_op = make_op(OpCode::New, &[], OpRef::ref_op(1));
    new_op.setdescr(Arc::new(SimpleSizeDescr::new(0, 32, 53)));
    let ops = vec![
        new_op,
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
    ];
    let (ops, constants) = rewrite_frontend_ops(&inputargs, ops);
    let (bytes, _guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);

    let mut seq: Vec<String> = Vec::new();
    count_operators(&bytes, |op| seq.push(format!("{op:?}")));
    // `rewrite.py` `_gen_call_malloc_gc` puts a NULL test after every
    // collecting malloc, and the peephole pass fuses the arm's store-then-read
    // of the result into a `local.tee`. Only the test itself is asserted: which
    // of the two failing arms follows it turns on whether the cpu has been
    // handed an `exit_frame_with_exception_descr_ref`, which a backend-only
    // module build has not.
    let checked = seq.windows(4).any(|w| {
        w[0].starts_with("CallIndirect")
            && w[1].starts_with("LocalTee")
            && w[2] == "I64Eqz"
            && w[3].starts_with("If")
    });
    assert!(checked, "the allocation emitted no NULL test: {seq:#?}");
}

#[test]
fn a_pointer_array_item_is_read_and_written_at_the_same_width() {
    use majit_ir::descr::SimpleArrayDescr;
    use std::sync::Arc;

    // A descriptor claiming the host's eight-byte pointer. The guest's is four,
    // and whichever of the two the arms take, the read and the write have to
    // take the same one: a four-byte read of an eight-byte write returns half a
    // pointer, and an eight-byte write over four-byte items clobbers the next.
    let descr: majit_ir::DescrRef = Arc::new(SimpleArrayDescr::new(0, 16, 8, 71, Type::Ref));
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
        InputArg::from_type_rc(Type::Ref, 2),
    ];
    let get = make_op(
        OpCode::GetarrayitemGcR,
        &[OpRef::input_arg_ref(0), OpRef::input_arg_int(1)],
        OpRef::ref_op(3),
    );
    get.setdescr(Arc::clone(&descr));
    let set = make_op(
        OpCode::SetarrayitemGc,
        &[
            OpRef::input_arg_ref(0),
            OpRef::input_arg_int(1),
            OpRef::input_arg_ref(2),
        ],
        OpRef::NONE,
    );
    set.setdescr(descr);
    let ops = vec![get, set, Op::new(OpCode::Finish, &[rb(OpRef::ref_op(3))])];
    let (ops, constants) = rewrite_frontend_ops(&inputargs, ops);
    let (bytes, _guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);

    // The entry loader reads every input from its frame slot, a Ref one
    // with `i64.load32_u`. Counting the same inputs finished without the
    // array ops leaves only the item read and the item write.
    let count_widths = |bytes: &[u8]| {
        let mut counts = [0usize; 4];
        count_operators(bytes, |op| match op {
            wasmparser::Operator::I64Load32U { .. } => counts[0] += 1,
            wasmparser::Operator::I64Load { .. } => counts[1] += 1,
            wasmparser::Operator::I64Store32 { .. } => counts[2] += 1,
            wasmparser::Operator::I64Store { .. } => counts[3] += 1,
            _ => {}
        });
        counts
    };
    let finish_only = vec![Op::new(OpCode::Finish, &[rb(OpRef::input_arg_ref(0))])];
    let (finish_only, finish_constants) = rewrite_frontend_ops(&inputargs, finish_only);
    let (entry_bytes, _guards) = build_module_default(&inputargs, &finish_only, &finish_constants);
    validate_wasm(&entry_bytes);
    let entry = count_widths(&entry_bytes);
    let all = count_widths(&bytes);
    let [narrow_loads, wide_loads, narrow_stores, wide_stores] =
        std::array::from_fn(|i| all[i].saturating_sub(entry[i]));
    assert!(
        narrow_loads + wide_loads >= 1 && narrow_stores + wide_stores >= 1,
        "GcLoadIndexedR / GcStoreIndexed must both access the item: loads=({narrow_loads},{wide_loads}) stores=({narrow_stores},{wide_stores})"
    );
    assert_eq!(
        (narrow_loads > 0),
        (narrow_stores > 0),
        "the item read and the item write moved different widths: loads=({narrow_loads},{wide_loads}) stores=({narrow_stores},{wide_stores})"
    );
}

#[test]
fn test_true_void_family_does_not_shift_new_call_type() {
    use majit_ir::descr::SimpleSizeDescr;
    use std::sync::Arc;

    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Ref, 1),
        InputArg::from_type_rc(Type::Int, 2),
    ];
    let new_op = make_op(OpCode::New, &[], OpRef::ref_op(3));
    new_op.setdescr(Arc::new(SimpleSizeDescr::new(0, 32, 53)));
    let ops = vec![
        void_call(
            vec![Type::Ref, Type::Ref],
            &[OpRef::input_arg_ref(0), OpRef::input_arg_ref(1)],
            0,
        ),
        new_op,
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(2))]),
    ];
    let (ops, constants) = rewrite_frontend_ops(&inputargs, ops);
    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);

    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    let (indirect_calls, drops) = indirect_call_types_and_drop_count(&bytes);
    // The void call and the `New`. Neither is followed by a frame reload:
    // `CaParams::default()` publishes no reload helper, which says the frame
    // this trace runs on is not one the jitframe shadow stack describes.
    assert_eq!(indirect_calls.len(), 2);
    assert_eq!(
        function_type(&bytes, indirect_calls[0].0 as usize),
        (
            vec![wasmparser::ValType::I64, wasmparser::ValType::I64],
            vec![]
        )
    );
    assert_eq!(
        function_type(&bytes, indirect_calls[1].0 as usize),
        (
            vec![wasmparser::ValType::I64, wasmparser::ValType::I64],
            vec![wasmparser::ValType::I64]
        )
    );
    assert_eq!(drops, 0);
}

#[test]
fn test_list_append_word_abi_and_new_type_indices_match_declared_i64_types() {
    use majit_ir::descr::SimpleSizeDescr;
    use std::sync::Arc;

    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Ref, 1),
        InputArg::from_type_rc(Type::Int, 2),
    ];
    let new_op = make_op(OpCode::New, &[], OpRef::ref_op(3));
    new_op.setdescr(Arc::new(SimpleSizeDescr::new(0, 32, 53)));
    let ops = vec![
        void_call_with_helper(
            vec![Type::Ref, Type::Ref],
            &[OpRef::input_arg_ref(0), OpRef::input_arg_ref(1)],
            8,
            RuntimeHelperKind::ListAppendValue,
        ),
        new_op,
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(2))]),
    ];
    let (ops, constants) = rewrite_frontend_ops(&inputargs, ops);
    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);

    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    let (indirect_calls, drops) = indirect_call_types_and_drop_count(&bytes);
    // No reload between them: `CaParams::default()` publishes no reload helper.
    assert_eq!(indirect_calls, vec![(3, 0), (3, 0)]);
    assert_eq!(
        function_type(&bytes, indirect_calls[0].0 as usize),
        (
            vec![wasmparser::ValType::I64, wasmparser::ValType::I64],
            vec![wasmparser::ValType::I64]
        ),
        "jit_list_append returns an ignored machine word"
    );
    assert_eq!(
        function_type(&bytes, indirect_calls[1].0 as usize),
        (
            vec![wasmparser::ValType::I64, wasmparser::ValType::I64],
            vec![wasmparser::ValType::I64]
        ),
        "New must reference its declared (i64, i64) -> i64 type"
    );
    assert_eq!(drops, 1, "only jit_list_append's result is discarded");
}

#[test]
fn test_true_void_float_arg_call_lowers_in_module() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Float, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let ops = vec![
        void_call(vec![Type::Float], &[OpRef::input_arg_float(0)], 0),
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(1))]),
    ];
    let (bytes, guards) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());

    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    assert!(has_table_import(&bytes));
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
    let (indirect_calls, drops) = indirect_call_types_and_drop_count(&bytes);
    assert_eq!(indirect_calls.len(), 1);
    assert_eq!(
        function_type(&bytes, indirect_calls[0].0 as usize),
        (vec![wasmparser::ValType::F64], vec![])
    );
    assert_eq!(drops, 0, "a genuine void call has no result to drop");
}

#[test]
fn test_void_word_abi_call_uses_i64_result_type_and_drop() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![
        void_call(vec![Type::Int], &[OpRef::input_arg_int(0)], 8),
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
    ];
    let (bytes, guards) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());

    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    assert!(has_table_import(&bytes));
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
    let (indirect_calls, drops) = indirect_call_types_and_drop_count(&bytes);
    // The call alone: `CaParams::default()` publishes no reload helper.
    assert_eq!(indirect_calls, vec![(2, 0)]);
    assert_eq!(
        function_type(&bytes, 2),
        (
            vec![wasmparser::ValType::I64],
            vec![wasmparser::ValType::I64]
        )
    );
    assert_eq!(drops, 1, "the ignored word result must be dropped");
}

/// A published reload helper is the statement that the running frame can move,
/// so the reload after a collecting call must appear exactly when one is
/// published. Emitting it unconditionally calls through whatever occupies the
/// unpublished slot and then installs its result as the frame pointer, which
/// is why both directions are asserted here rather than only the one this
/// backend's own host takes.
#[test]
fn test_frame_reload_is_emitted_only_when_a_reload_helper_is_published() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![
        void_call(vec![Type::Int], &[OpRef::input_arg_int(0)], 8),
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
    ];
    let build = |ca| {
        build_module_with_ca(
            &inputargs,
            &ops,
            &indexmap::IndexMap::new(),
            Some(0),
            &codegen::GuardGcTypeInfo::default(),
            codegen::FrameGeometry::fixed(),
            ca,
        )
        .0
    };

    let unpublished = build(codegen::CaParams::default());
    validate_wasm(&unpublished);
    assert_eq!(
        indirect_call_types_and_drop_count(&unpublished).0,
        vec![(2, 0)],
        "the call alone; a frame the host never pushed has nothing to reload from"
    );

    let published = build(codegen::CaParams {
        ca_reload_fn_ptr: 7,
        ..codegen::CaParams::default()
    });
    validate_wasm(&published);
    assert_eq!(
        indirect_call_types_and_drop_count(&published).0,
        vec![(2, 0), (1, 0)],
        "the reload follows the call whose collection may have moved the frame"
    );
}

#[test]
fn test_true_void_type_index_accounts_for_trampoline_type() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![
        make_op(
            OpCode::CallI,
            &[OpRef::const_int(43), OpRef::input_arg_int(0)],
            OpRef::int_op(1),
        ),
        void_call(vec![], &[], 0),
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]),
    ];
    let (bytes, guards) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());

    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), Some(1));
    let (indirect_calls, drops) = indirect_call_types_and_drop_count(&bytes);
    assert_eq!(indirect_calls, vec![(2, 0)]);
    assert_eq!(drops, 0);
}

#[test]
fn test_guard_types() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];

    let ops = vec![
        Op::new(
            OpCode::Label,
            &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_int(1))],
        ),
        // GuardTrue
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::input_arg_int(0)],
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
        ),
        // GuardFalse
        make_guard(
            OpCode::GuardFalse,
            &[OpRef::input_arg_int(1)],
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
        ),
        // GuardValue
        make_guard(
            OpCode::GuardValue,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            &[OpRef::input_arg_int(0)],
        ),
        // GuardNonnull
        make_guard(
            OpCode::GuardNonnull,
            &[OpRef::input_arg_int(0)],
            &[OpRef::input_arg_int(0)],
        ),
        // GuardIsnull
        make_guard(
            OpCode::GuardIsnull,
            &[OpRef::input_arg_int(1)],
            &[OpRef::input_arg_int(1)],
        ),
        // GuardNoOverflow (0 args)
        {
            let op = Op::new(OpCode::GuardNoOverflow, &[]);
            op.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
            op
        },
        // GuardNotInvalidated (0 args; address 0 in this generic smoke test)
        {
            let op = Op::new(OpCode::GuardNotInvalidated, &[]);
            op.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
            op
        },
        Op::new(
            OpCode::Jump,
            &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_int(1))],
        ),
    ];

    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 7);
}

/// GUARD_NOT_INVALIDATED must read the owning token's live AtomicBool rather
/// than being folded to an always-passing guard. The runtime regression is
/// covered by global_quasiimmut_invalidation.py; this pins the emitted load.
#[test]
fn test_guard_not_invalidated_loads_runtime_flag() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let guard = Op::new(OpCode::GuardNotInvalidated, &[]);
    guard.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
    let ops = vec![
        guard,
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
    ];
    let constants = indexmap::IndexMap::new();
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants: constants.clone(),
        vtable_offset: None,
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0x1000,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    };
    let (bytes, guards, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");

    validate_wasm(&bytes);
    assert_eq!(guards.len(), 2);
    let mut saw_flag_load = false;
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                if matches!(
                    operators.read().unwrap(),
                    wasmparser::Operator::I32Load8U { .. }
                ) {
                    saw_flag_load = true;
                }
            }
        }
    }
    assert!(saw_flag_load, "GUARD_NOT_INVALIDATED omitted its flag load");
}

/// GuardNoException loads the global exception slot and fails when it is set;
/// GuardException compares the pending exception type against the expected one
/// and, on match, binds the caught value into its result var and clears both
/// slots. Validates the emitted bytecode is well-formed (stack-balanced).
#[test]
fn test_exception_guards() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];

    let ops = vec![
        Op::new(OpCode::Label, &[rb(OpRef::input_arg_int(0))]),
        // GuardNoException — 0 args, fails when an exception is pending.
        {
            let op = Op::new(OpCode::GuardNoException, &[]);
            op.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
            op
        },
        // GuardException(expected_type) — caught value bound to int_op(1).
        {
            let op = Op::new(OpCode::GuardException, &[rb(OpRef::input_arg_int(0))]);
            op.pos().set(OpRef::int_op(1));
            op.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
            op
        },
        Op::new(OpCode::Jump, &[rb(OpRef::input_arg_int(0))]),
    ];

    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 2);
    let expected = majit_backend_wasm::jit_exc_type_addr() as i32;
    let mut saw_type_addr = false;
    count_operators(&bytes, |op| {
        if matches!(op, wasmparser::Operator::I32Const { value } if *value == expected) {
            saw_type_addr = true;
        }
    });
    assert!(
        saw_type_addr,
        "GuardNoException must load jit_exc_type_addr()={expected}, not a stale link-time estimate"
    );
}

/// GuardGcType contract in majit: arg0 = object ref, arg1 = expected
/// type_id. The wasm backend reads the GC header word at
/// `obj - GcHeader::SIZE` (matching the cranelift backend and
/// `majit_gc::header::GcHeader` layout) and compares the low
/// `TYPE_ID_BITS` against arg1. arg1 is an immediate type_id, NOT a
/// classptr — no `mem32[obj + 0]` read, no classptr→typeid lookup.
#[test]
fn test_guard_gc_type_uses_immediate_typeid() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];

    // Inline-Const carrying the immediate typeid 0x42
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();

    let ops = vec![
        Op::new(OpCode::Label, &[rb(OpRef::input_arg_int(0))]),
        make_guard(
            OpCode::GuardGcType,
            &[OpRef::input_arg_int(0), OpRef::const_int(0x42)],
            &[OpRef::input_arg_int(0)],
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::input_arg_int(0))]),
    ];

    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
}

/// Build a `GuardGcTypeInfo` matching what `WasmBackend::compile_loop`
/// derives from a real `GcLLDescr_framework`-equivalent allocator.
/// Mirrors `gc.py get_translated_info_for_typeinfo` /
/// `gc.py get_translated_info_for_guard_is_object` /
/// `x86/assembler.py Assembler386.genop_guard_guard_subclass cpu.subclassrange_min_offset`.
fn enabled_guard_gc_type_info() -> codegen::GuardGcTypeInfo {
    // Pretend the TYPE_INFO table sits at a small in-memory address;
    // wasm validation only checks the bytecode shape, not the actual
    // load addresses, so any value works for codegen testing.
    // majit `TypeEntry` stride = 64 bytes (the four-word TYPE_INFO plus the
    // four-word VARSIZE_TYPE_INFO/CLASSTYPE union tail).
    // shift_by = log2(64) = 6, sizeof_ti = rffi.sizeof(TYPE_INFO) = 32.
    // gc.py _setup_guard_is_object: T_IS_RPYTHON_INSTANCE
    // = 0x100000 (gctypelayout.py), packed little-endian into a
    // Signed word — byte at offset +2 carries the flag, mask = 0x10.
    codegen::GuardGcTypeInfo {
        supports_guard_gc_type: true,
        base_type_info: 0x1000,
        shift_by: 6,
        sizeof_ti: 32, // size_of::<TypeInfoLayout>()
        infobits_offset: 2,
        is_object_flag: 0x10,
        subclassrange_min_offset: 0, // offset within ClassTypeLayout
        ..Default::default()
    }
}

/// x86/assembler.py `genop_guard_guard_is_object` lowering —
/// the wasm backend's GUARD_IS_OBJECT arm emits the same MOV+addr_add
/// +TEST8+branch sequence. With `supports_guard_gc_type` enabled the
/// `assert` at line 1925 falls through and the rest of the lowering
/// runs; the resulting module must validate as legal wasm.
#[test]
fn test_guard_is_object_lowers_to_typeinfo_test() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];

    let ops = vec![
        Op::new(OpCode::Label, &[rb(OpRef::input_arg_int(0))]),
        make_guard(
            OpCode::GuardIsObject,
            &[OpRef::input_arg_int(0)],
            &[OpRef::input_arg_int(0)],
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::input_arg_int(0))]),
    ];

    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, guards) = build_module(
        &inputargs,
        &ops,
        &constants,
        Some(0),
        &enabled_guard_gc_type_info(),
    );
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
}

/// x86/assembler.py `genop_guard_guard_subclass` lowering —
/// the wasm backend's GUARD_SUBCLASS arm emits the gcremovetypeptr
/// branch (cpu.vtable_offset = None) when `vtable_offset` is `None`,
/// otherwise the vtable-load branch. With `supports_guard_gc_type`
/// enabled and the constant classptr's `(min, max)` pre-fetched, the
/// lowering runs to completion.
#[test]
fn test_guard_subclass_lowers_to_subclassrange_check() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];

    // model.py `cls_of_box()` returns `ConstInt(ptr2int(typeptr))` —
    // the emitted guard-class operand is the vtable address carried as a raw
    // integer (read with `op.getarg(1).getint()`, rewrite.py). Use the
    // inline ConstInt factory so the variant tag matches the backend reader.
    let class_constant = OpRef::const_int(0xCAFE);
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();

    let ops = vec![
        Op::new(OpCode::Label, &[rb(OpRef::input_arg_int(0))]),
        make_guard(
            OpCode::GuardSubclass,
            &[OpRef::input_arg_int(0), class_constant],
            &[OpRef::input_arg_int(0)],
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::input_arg_int(0))]),
    ];

    let mut info = enabled_guard_gc_type_info();
    // assembler.py:1971-1974: codegen-time
    // (vtable_ptr.subclassrange_min, vtable_ptr.subclassrange_max).
    info.subclass_ranges.insert(0xCAFE, (10, 20));

    // gcremovetypeptr branch: vtable_offset = None.
    let (bytes, guards) = build_module(&inputargs, &ops, &constants, None, &info);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);

    // vtable-load branch: vtable_offset = Some(...).
    let (bytes2, _) = build_module(&inputargs, &ops, &constants, Some(8), &info);
    validate_wasm(&bytes2);
}

#[test]
fn test_sameas_and_conversions() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];

    let ops = vec![
        make_op(
            OpCode::SameAsI,
            &[OpRef::input_arg_int(0)],
            OpRef::int_op(1),
        ),
        make_op(
            OpCode::CastIntToFloat,
            &[OpRef::input_arg_int(0)],
            OpRef::float_op(2),
        ),
        make_op(
            OpCode::CastFloatToInt,
            &[OpRef::float_op(2)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::CastIntToPtr,
            &[OpRef::input_arg_int(0)],
            OpRef::ref_op(4),
        ),
        make_op(OpCode::CastPtrToInt, &[OpRef::ref_op(4)], OpRef::int_op(5)),
        make_op(OpCode::IntNeg, &[OpRef::input_arg_int(0)], OpRef::int_op(6)),
        make_op(
            OpCode::IntInvert,
            &[OpRef::input_arg_int(0)],
            OpRef::int_op(7),
        ),
        make_op(
            OpCode::IntIsTrue,
            &[OpRef::input_arg_int(0)],
            OpRef::int_op(8),
        ),
        make_op(
            OpCode::IntIsZero,
            &[OpRef::input_arg_int(0)],
            OpRef::int_op(9),
        ),
        {
            let op = Op::new(OpCode::Finish, &[rb(OpRef::int_op(9))]);
            op.setfailargs(smallvec![rb(OpRef::int_op(9))]);
            op
        },
    ];

    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, _) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
}

#[test]
fn test_overflow_ops() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];

    let ops = vec![
        make_op(
            OpCode::IntAddOvf,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            OpRef::int_op(2),
        ),
        {
            let op = Op::new(OpCode::GuardNoOverflow, &[]);
            op.setfailargs(smallvec![rb(OpRef::int_op(2))]);
            op
        },
        make_op(
            OpCode::IntSubOvf,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            OpRef::int_op(3),
        ),
        {
            let op = Op::new(OpCode::GuardNoOverflow, &[]);
            op.setfailargs(smallvec![rb(OpRef::int_op(3))]);
            op
        },
        {
            let op = Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))]);
            op.setfailargs(smallvec![rb(OpRef::int_op(2))]);
            op
        },
    ];

    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 3); // 2 GuardNoOverflow + 1 Finish
}

#[test]
fn test_single_label_peeled_loop_validates() {
    // A single-label PEELED loop: a preamble op (the unrolled first iteration)
    // precedes the LABEL, so codegen wraps it in the resume-at-LABEL preamble-
    // skip dispatch (block $exit / $past_loader / $skip_preamble + br_if, with
    // the preamble at br-depth 2 and the body at 1). This validates the new
    // control-flow nesting and br depths via wasmparser.
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)]; // i
    let const_1 = OpRef::const_int(1);
    let const_100 = OpRef::const_int(100);
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();

    let ops = vec![
        // preamble (unrolled first iteration): i + 1 -> v1
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), const_1],
            OpRef::int_op(1),
        ),
        // loop header carrying v1 (single LABEL, with the preamble before it)
        Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]),
        // body: v1 + 1 -> v2 ; v2 < 100 -> v3 ; guard ; jump v2 back to LABEL
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), const_1],
            OpRef::int_op(2),
        ),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(2), const_100],
            OpRef::int_op(3),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(3)], &[OpRef::int_op(2)]),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(2))]),
    ];

    // Must be classified as single-label peeled (exercises the dispatch wrapper).
    assert!(codegen::is_single_label_peeled(&ops));

    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    assert!(!guards[0].is_finish);
}

/// Key 0 runs the preamble. Key `label ordinal + 1` enters that LABEL's
/// resume loader and reads the args from `FRAME_SLOT_BASE`, skipping the
/// preamble. `build_function`'s `br_table` and `front_target_dispatch_key`
/// share that numbering.
#[test]
fn peeled_loop_dispatch_key_enters_the_label_not_the_preamble() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1000)],
            OpRef::int_op(1),
        ),
        Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]),
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(1))]),
    ];
    assert!(codegen::is_single_label_peeled(&ops));
    let (bytes, _) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);

    let frame = codegen::FrameGeometry::fixed();
    assert_eq!(
        execute_trace_at_key(&bytes, &[7], frame.dispatch_key_ofs, 0),
        1007
    );
    assert_eq!(
        execute_trace_at_key(&bytes, &[7], frame.dispatch_key_ofs, 1),
        7
    );
}

/// Seventeen LABEL arguments, entered at key 1. The value area is sized from
/// `frame_value_slots` (no frozen 64-slot floor). Every argument must come
/// back through the checksum, which changes if the dispatch key overwrites one.
#[test]
fn wide_label_dispatch_key_preserves_every_argument() {
    const N: usize = 17;
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let mut ops = Vec::new();
    let mut label_args = vec![rb(OpRef::input_arg_int(0))];
    for k in 1..N {
        ops.push(make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(k as i64)],
            OpRef::int_op(k as u32),
        ));
        label_args.push(rb(OpRef::int_op(k as u32)));
    }
    ops.push(Op::new(OpCode::Label, &label_args));
    let mut acc = OpRef::input_arg_int(0);
    let mut next = N as u32;
    for k in 1..N {
        let scaled = OpRef::int_op(next);
        next += 1;
        ops.push(make_op(
            OpCode::IntMul,
            &[acc, OpRef::const_int(100)],
            scaled,
        ));
        let summed = OpRef::int_op(next);
        next += 1;
        ops.push(make_op(
            OpCode::IntAdd,
            &[scaled, OpRef::int_op(k as u32)],
            summed,
        ));
        acc = summed;
    }
    ops.push(Op::new(OpCode::Finish, &[rb(acc)]));
    ops.push(Op::new(OpCode::Jump, &label_args));

    let slots = codegen::frame_value_slots(&inputargs, &ops);
    let frame = codegen::FrameGeometry::compact(slots, 0, 0);
    let (bytes, _) = build_module_with_frame(
        &inputargs,
        &ops,
        &indexmap::IndexMap::new(),
        Some(0),
        &codegen::GuardGcTypeInfo::default(),
        frame,
    );
    validate_wasm(&bytes);

    let host: Vec<i64> = (10..10 + N as i64).collect();
    let mut expected = host[0];
    for &arg in &host[1..] {
        expected = expected.wrapping_mul(100).wrapping_add(arg);
    }
    assert_eq!(
        execute_trace_at_key(&bytes, &host, frame.dispatch_key_ofs, 1),
        expected
    );
}

#[test]
fn vec_guard_true_compiles() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![
        make_op(
            OpCode::IntLt,
            &[OpRef::input_arg_int(0), OpRef::const_int(10)],
            OpRef::int_op(1),
        ),
        make_guard(
            OpCode::VecGuardTrue,
            &[OpRef::int_op(1)],
            &[OpRef::input_arg_int(0)],
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
    ];
    let (bytes, guards) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 2);
    assert_eq!(execute_simple_trace(&bytes, &[3]), 3);
    assert_eq!(execute_simple_trace(&bytes, &[11]), 11);
}

/// `VecGuardFalse` fails when its argument is nonzero. Success and failure
/// write different words, so a flipped condition fails the assertion.
/// The argument is an input, so `next_op_can_accept_cc` does not fuse.
#[test]
fn vec_guard_false_direct_polarity() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![
        make_guard(
            OpCode::VecGuardFalse,
            &[OpRef::input_arg_int(0)],
            &[OpRef::const_int(20)],
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::const_int(10))]),
    ];
    let (bytes, guards) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 2);
    assert_eq!(execute_simple_trace(&bytes, &[0]), 10);
    assert_eq!(execute_simple_trace(&bytes, &[1]), 20);
}

/// The comparison is the guard's only reader, so `next_op_can_accept_cc`
/// fuses it into `push_guard_failure_cond`. `VecGuardFalse` fails when the
/// comparison is true.
#[test]
fn vec_guard_false_fused_comparison_polarity() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![
        make_op(
            OpCode::IntLt,
            &[OpRef::input_arg_int(0), OpRef::const_int(10)],
            OpRef::int_op(1),
        ),
        make_guard(
            OpCode::VecGuardFalse,
            &[OpRef::int_op(1)],
            &[OpRef::const_int(20)],
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::const_int(10))]),
    ];
    let (bytes, guards) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 2);
    assert_eq!(execute_simple_trace(&bytes, &[3]), 20);
    assert_eq!(execute_simple_trace(&bytes, &[11]), 10);
}

/// A `LoadFromGcTable` placed inside the loop body is emitted inside the loop.
///
/// `rewrite.py remove_constptr` caches one load per gc-table index,
/// but `rewrite.py emit_label` clears `gcrefs_recently_loaded` at
/// every LABEL, so a reference constant used after the LABEL is loaded again on
/// each iteration. The comment there rejects keeping the value alive across the
/// label ("don't spill it") as "the wrong level" — the backend emits the op
/// where the trace puts it and leaves that decision to the optimizer.
#[test]
fn gc_table_load_inside_a_loop_body_is_emitted_inside_the_loop() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]),
        make_op(
            OpCode::LoadFromGcTable,
            &[OpRef::const_int(0)],
            OpRef::ref_op(2),
        ),
        make_guard(
            OpCode::GuardNonnull,
            &[OpRef::ref_op(2)],
            &[OpRef::int_op(1)],
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(3))]),
    ];
    let gc_table_base = 4096;
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops: ops.iter().cloned().collect(),
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    };
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);

    let mut control_stack = Vec::new();
    let mut gc_table_address_on_stack = false;
    let mut loads_inside_loop = 0usize;
    let mut loads_outside_loop = 0usize;
    let mut saw_loop = false;
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::Loop { .. } => {
                        saw_loop = true;
                        control_stack.push(true);
                        gc_table_address_on_stack = false;
                    }
                    wasmparser::Operator::Block { .. } | wasmparser::Operator::If { .. } => {
                        control_stack.push(false);
                        gc_table_address_on_stack = false;
                    }
                    wasmparser::Operator::End => {
                        control_stack.pop();
                        gc_table_address_on_stack = false;
                    }
                    wasmparser::Operator::I32Const { value } if value == gc_table_base as i32 => {
                        gc_table_address_on_stack = true;
                    }
                    wasmparser::Operator::I64Load32U { .. } if gc_table_address_on_stack => {
                        if control_stack.contains(&true) {
                            loads_inside_loop += 1;
                        } else {
                            loads_outside_loop += 1;
                        }
                        gc_table_address_on_stack = false;
                    }
                    _ => gc_table_address_on_stack = false,
                }
            }
        }
    }
    // Without this, the counts below also hold for a body that emitted no loop
    // at all.
    assert!(saw_loop, "codegen emitted no loop for a looping trace");
    assert_eq!(
        loads_inside_loop, 1,
        "the in-loop LoadFromGcTable must be emitted inside the loop"
    );
    assert_eq!(
        loads_outside_loop, 0,
        "no gc-table load belongs outside the loop for this trace"
    );
}

/// A preamble `LoadFromGcTable` used only as a later guard failarg is
/// rematerialized at the spill. rewrite.py clears `gcrefs_recently_loaded`
/// at LABEL, so the local is not a LABEL arg and is stale after the first
/// collecting back edge — cranelift `resolve_failarg_opref`.
#[test]
fn preamble_gc_table_failarg_is_reloaded_inside_the_loop() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![
        make_op(
            OpCode::LoadFromGcTable,
            &[OpRef::const_int(0)],
            OpRef::ref_op(1),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        Op::new(OpCode::Label, &[rb(OpRef::int_op(2))]),
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(2)],
            &[OpRef::ref_op(1), OpRef::int_op(2)],
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(3))]),
    ];
    let gc_table_base = 4096;
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops,
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::compact(5, 3, 1),
        ca: codegen::CaParams::default(),
    };
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);

    let mut control_stack = Vec::new();
    let mut gc_table_address_on_stack = false;
    let mut loads_inside_loop = 0usize;
    let mut loads_outside_loop = 0usize;
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::Loop { .. } => {
                        control_stack.push(true);
                        gc_table_address_on_stack = false;
                    }
                    wasmparser::Operator::Block { .. } | wasmparser::Operator::If { .. } => {
                        control_stack.push(false);
                        gc_table_address_on_stack = false;
                    }
                    wasmparser::Operator::End => {
                        control_stack.pop();
                        gc_table_address_on_stack = false;
                    }
                    wasmparser::Operator::I32Const { value } if value == gc_table_base as i32 => {
                        gc_table_address_on_stack = true;
                    }
                    wasmparser::Operator::I64Load32U { .. } if gc_table_address_on_stack => {
                        if control_stack.contains(&true) {
                            loads_inside_loop += 1;
                        } else {
                            loads_outside_loop += 1;
                        }
                        gc_table_address_on_stack = false;
                    }
                    _ => gc_table_address_on_stack = false,
                }
            }
        }
    }
    assert_eq!(
        loads_outside_loop, 1,
        "the preamble LoadFromGcTable stays outside the loop"
    );
    assert_eq!(
        loads_inside_loop, 1,
        "the guard failarg must rematerialize the table load inside the loop"
    );
}

#[test]
fn test_peeled_label_captures_missing_ref_livein_in_frozen_frame() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        // The Ref input remains live in the body but is intentionally absent
        // from the semantic LABEL args: it must be restored from a GC-rooted
        // backend capture home on bridge re-entry.
        Op::new(OpCode::Label, &[rb(OpRef::int_op(2))]),
        make_guard(
            OpCode::GuardNonnull,
            &[OpRef::input_arg_ref(0)],
            &[OpRef::input_arg_ref(0), OpRef::int_op(2)],
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(3))]),
    ];

    assert!(codegen::is_resumable_peeled(&ops));
    assert_eq!(codegen::label_ref_capture_slots(&inputargs, &ops), 1);
    let ordinary_homes = codegen::count_ref_homes(&inputargs, &ops);
    let frame = codegen::FrameGeometry::compact(
        codegen::frame_value_slots(&inputargs, &ops),
        ordinary_homes + 1,
        1,
    );
    assert_eq!(
        codegen::label_resume_info(&inputargs, &ops, frame),
        vec![(true, true)]
    );
    assert_eq!(frame.ordinary_home_slots(), ordinary_homes);
    let (bytes, guards) = build_module_with_frame(
        &inputargs,
        &ops,
        &indexmap::IndexMap::new(),
        Some(0),
        &codegen::GuardGcTypeInfo::default(),
        frame,
    );
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
}

#[test]
fn test_multi_label_peeled_resumes_at_last_label_validates() {
    // A MULTI-label peeled loop: a preamble precedes an outer entry LABEL and
    // the inner loop-header LABEL. `is_single_label_peeled` is false (two
    // labels) but `is_resumable_peeled` is true, so codegen emits the SAME
    // resume-at-LABEL 3-block wrapper, resuming at the LAST label (where the
    // `loop` is). This proves the wrapper + br depths stay valid for a
    // multi-label source — the case `compile_bridge` newly accepts when a
    // loop-closing bridge targets that last label.
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)]; // i
    let const_1 = OpRef::const_int(1);
    let const_100 = OpRef::const_int(100);
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();

    let ops = vec![
        // preamble (unrolled first iteration): i + 1 -> v1
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), const_1],
            OpRef::int_op(1),
        ),
        // outer entry LABEL carrying v1 (no `loop` — a codegen no-op)
        Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]),
        // inner loop-header LABEL carrying v1 (the LAST label — `loop` here)
        Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]),
        // body: v1 + 1 -> v2 ; v2 < 100 -> v3 ; guard ; jump v2 back to LABEL
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), const_1],
            OpRef::int_op(2),
        ),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(2), const_100],
            OpRef::int_op(3),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(3)], &[OpRef::int_op(2)]),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(2))]),
    ];

    // Multi-label: NOT the single-label subset, but still resumable-peeled, so
    // the wrapper is emitted and resumes at the last label.
    assert!(!codegen::is_single_label_peeled(&ops));
    assert!(codegen::is_resumable_peeled(&ops));

    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    assert!(!guards[0].is_finish);
}

/// A resumable-peeled loop within the fixed arity emits two functions: the
/// narrow `trace` shim the host and CALL_ASSEMBLER keep entering, and the
/// `trace_wide` body a loop-closing JUMP can pass its arguments to.
///
/// `trace` must stay structurally `(i32) -> i32`: a loop's table slot is also
/// what the CALL_ASSEMBLER path calls as type 0, so widening it in place would
/// turn every such call into a trap rather than a decline.
#[test]
fn peeled_loop_exports_a_narrow_shim_beside_its_wide_entry() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let const_1 = OpRef::const_int(1);
    let const_100 = OpRef::const_int(100);
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();

    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), const_1],
            OpRef::int_op(1),
        ),
        Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]),
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), const_1],
            OpRef::int_op(2),
        ),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(2), const_100],
            OpRef::int_op(3),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(3)], &[OpRef::int_op(2)]),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(2))]),
    ];

    let frame = codegen::FrameGeometry::fixed();
    assert!(
        codegen::has_label_param_entry(&inputargs, &ops, frame, None),
        "this shape is the one the wide entry exists for; if the gate stops \
         accepting it the rest of these assertions prove nothing"
    );

    let (bytes, _) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);

    let (types, functions, exports) = module_shape(&bytes);
    let narrow = exports.get("trace").copied().expect("narrow entry export");
    let wide = exports
        .get("trace_wide")
        .copied()
        .expect("wide entry export");
    assert_eq!(wide, narrow + 1, "the wide entry follows the shim");

    // Exports index the whole function space, imports first; the type section
    // is indexed by defined function only.
    let imported = narrow;
    assert_eq!(
        types[functions[(narrow - imported) as usize] as usize],
        (1, 1),
        "trace must stay (i32) -> i32"
    );
    assert_eq!(
        types[functions[(wide - imported) as usize] as usize],
        (1 + majit_backend_wasm::FROZEN_LABEL_PARAM_ARITY, 1),
        "trace_wide takes frame_ptr plus one word per fixed label parameter"
    );

    // The shim reads from `FRAME_SLOT_BASE`, so its loads land in slots
    // 1..=FROZEN_LABEL_PARAM_ARITY and a frame holding exactly that many slots
    // is one short — its last load would run off the end.
    let exact = codegen::FrameGeometry {
        value_slots: majit_backend_wasm::FROZEN_LABEL_PARAM_ARITY,
        ..frame
    };
    assert!(
        !codegen::has_label_param_entry(&inputargs, &ops, exact, None),
        "a frame with FROZEN_LABEL_PARAM_ARITY slots must be rejected"
    );
    let one_more = codegen::FrameGeometry {
        value_slots: majit_backend_wasm::FROZEN_LABEL_PARAM_ARITY + 1,
        ..frame
    };
    assert!(
        codegen::has_label_param_entry(&inputargs, &ops, one_more, None),
        "one slot past the arity is the smallest frame the shim can read"
    );
}

/// A loop-closing bridge whose target published a wide entry hands the jump
/// arguments over as parameters. The narrow entry is a shim that loads
/// `FROZEN_LABEL_PARAM_ARITY` frame slots and tail-calls the wide one, so
/// writing the args into those slots first is a round trip through memory that
/// the target undoes instruction for instruction.
///
/// The dispatch key still travels through the frame: it selects the entry
/// `br_table` arm, which runs before any parameter is read.
#[test]
fn a_loop_closing_jump_passes_its_args_to_a_published_wide_entry() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let const_1 = OpRef::const_int(1);
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();

    // No LABEL of its own, so the terminal JUMP closes into a separate module.
    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), const_1],
            OpRef::int_op(1),
        ),
        Op::new(OpCode::Jump, &[rb(OpRef::int_op(1))]),
    ];

    let (narrow_bytes, _) = build_external_jump_module(&inputargs, &ops, &constants, 7, 3, 0);
    let (wide_bytes, _) = build_external_jump_module(&inputargs, &ops, &constants, 7, 3, 8);
    validate_wasm(&narrow_bytes);
    validate_wasm(&wide_bytes);

    let (narrow_stores, narrow_call) = external_jump_tail(&narrow_bytes);
    let (wide_stores, wide_call) = external_jump_tail(&wide_bytes);

    assert_eq!(
        narrow_call,
        Some((7, 0)),
        "without a wide entry the JUMP tail-calls the narrow slot as type 0"
    );
    assert_eq!(
        narrow_stores, 2,
        "the narrow arm writes one jump arg and the dispatch key"
    );
    assert_eq!(
        wide_stores, 1,
        "only the dispatch key still goes through the frame"
    );

    let (types, ..) = module_shape(&wide_bytes);
    let (wide_slot, wide_type) = wide_call.expect("the wide arm tail-calls its target");
    assert_eq!(wide_slot, 8, "the JUMP names the published wide slot");
    assert_eq!(
        types[wide_type as usize],
        (1 + majit_backend_wasm::FROZEN_LABEL_PARAM_ARITY, 1),
        "the call type is the target's parameter entry: frame_ptr plus the \
         fixed label parameters"
    );
}

/// A JUMP whose arity sits between the old 16-word ceiling and the frozen
/// granule still takes the wide parameter entry. Storing those args into
/// frame slots only to have the shim load them back is the round trip
/// `closing_jump` avoids by handing them in registers.
#[test]
fn a_jump_wider_than_sixteen_args_uses_the_wide_entry() {
    let n = 17usize;
    assert!(
        n <= majit_backend_wasm::FROZEN_LABEL_PARAM_ARITY,
        "17 must fit the published wide type"
    );
    let inputargs: Vec<_> = (0..n)
        .map(|i| InputArg::from_type_rc(Type::Int, i as u32))
        .collect();
    let jump_args: Vec<_> = (0..n).map(|i| rb(OpRef::input_arg_int(i as u32))).collect();
    let ops = vec![Op::new(OpCode::Jump, &jump_args)];
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();

    let (wide_bytes, _) = build_external_jump_module(&inputargs, &ops, &constants, 7, 3, 8);
    validate_wasm(&wide_bytes);
    let (wide_stores, wide_call) = external_jump_tail(&wide_bytes);
    assert_eq!(
        wide_stores, 1,
        "only the dispatch key still goes through the frame"
    );
    let (types, ..) = module_shape(&wide_bytes);
    let (wide_slot, wide_type) = wide_call.expect("the wide arm tail-calls its target");
    assert_eq!(wide_slot, 8, "the JUMP names the published wide slot");
    assert_eq!(
        types[wide_type as usize],
        (1 + majit_backend_wasm::FROZEN_LABEL_PARAM_ARITY, 1),
        "the call type is the target's parameter entry: frame_ptr plus the \
         fixed label parameters"
    );
}

/// A bridge on geometry GA tail-calls a LABEL of a loop on geometry GB.
/// Args land at GB's spill slots, the dispatch key at GB's key offset, and
/// the prologue compares `_check_frame_depth` against GB when that frame is
/// deeper. The running frame is pre-sized here; the compare immediate is the
/// growth the bridge would request.
#[test]
fn cross_geometry_jump_delivers_label_args() {
    let ga = codegen::FrameGeometry::compact(8, 2, 0);
    let gb = codegen::FrameGeometry::compact(40, 12, 2);
    assert!(gb.signed_item_count() > ga.signed_item_count());
    assert_ne!(ga.dispatch_key_ofs, gb.dispatch_key_ofs);
    assert_ne!(ga.home_slot_base, gb.home_slot_base);

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let bridge_ops = vec![Op::new(OpCode::Jump, &[rb(OpRef::input_arg_int(0))])];
    let descr = majit_ir::make_loop_target_descr(1, false);
    let label = Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]);
    label.setdescr(descr.clone());
    let back = Op::new(OpCode::Jump, &[rb(OpRef::int_op(1))]);
    back.setdescr(descr);
    let target_ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        label,
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]),
        back,
    ];

    let depth = ga.signed_item_count().max(gb.signed_item_count());
    let jump_ca = codegen::CaParams {
        external_jump_frame: Some(gb),
        ..codegen::CaParams::default()
    };
    let (bridge_bytes, _) = build_framed_module(
        &inputargs,
        &bridge_ops,
        &constants,
        ga,
        codegen::CaParams {
            realloc_fn_ptr: 9,
            frame_depth_items: depth,
            ..jump_ca.clone()
        },
        4,
        1,
        0,
    );
    validate_wasm(&bridge_bytes);
    let stores = i64_store_offsets(&bridge_bytes);
    assert!(
        stores.contains(&gb.spill_slot_ofs(0)),
        "narrow JUMP stores the label arg at the target spill, got {stores:?}"
    );
    assert!(
        stores.contains(&gb.dispatch_key_ofs),
        "dispatch key is stored at the target offset, got {stores:?}"
    );
    assert!(
        !stores.contains(&ga.dispatch_key_ofs),
        "the jumper's own key offset is not the target's"
    );
    assert!(
        i32_consts(&bridge_bytes).contains(&(depth as i32)),
        "prologue compares against max(bridge, target) item count"
    );

    let (target_bytes, _) = build_framed_module(
        &inputargs,
        &target_ops,
        &constants,
        gb,
        codegen::CaParams::default(),
        0,
        0,
        0,
    );
    validate_wasm(&target_bytes);

    let (run_bytes, _) =
        build_framed_module(&inputargs, &bridge_ops, &constants, ga, jump_ca, 4, 1, 0);
    validate_wasm(&run_bytes);
    let engine = Engine::default();
    let bridge_module = Module::new(&engine, &run_bytes).unwrap();
    let target_module = Module::new(&engine, &target_bytes).unwrap();
    let mut store = Store::new(&engine, ());
    let memory = Memory::new(&mut store, MemoryType::new(1, None)).unwrap();
    let table = Table::new(
        &mut store,
        TableType::new(ValType::FuncRef, 8, None),
        Val::default(ValType::FuncRef),
    )
    .unwrap();
    let mut linker = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .define("env", "__indirect_function_table", table)
        .unwrap();
    let target_inst = linker
        .instantiate_and_start(&mut store, &target_module)
        .expect("target loop instantiates");
    let target_func = target_inst.get_func(&store, "trace").expect("trace");
    table.set(&mut store, 4, target_func.into()).unwrap();
    let bridge_inst = linker
        .instantiate_and_start(&mut store, &bridge_module)
        .expect("bridge instantiates");

    // Items base sits after the jitframe header. `emit_store_header_word`
    // subtracts `FIRST_ITEM_OFFSET` from local 0, so a zero base is out of range.
    let frame_ptr = 4096i32;
    let arg = 41i64;
    memory
        .write(
            &mut store,
            frame_ptr as usize + ga.spill_slot_ofs(0) as usize,
            &arg.to_le_bytes(),
        )
        .unwrap();
    bridge_inst
        .get_typed_func::<i32, i32>(&store, "trace")
        .unwrap()
        .call(&mut store, frame_ptr)
        .expect("cross-geometry jump runs");

    let mut word = [0u8; 8];
    memory
        .read(
            &store,
            frame_ptr as usize + gb.spill_slot_ofs(0) as usize,
            &mut word,
        )
        .unwrap();
    assert_eq!(
        i64::from_le_bytes(word),
        arg,
        "LABEL resume returns the jump arg, not the preamble's increment"
    );
    memory
        .read(
            &store,
            frame_ptr as usize + gb.dispatch_key_ofs as usize,
            &mut word,
        )
        .unwrap();
    assert_eq!(i64::from_le_bytes(word), 1, "target dispatch key");
}

fn build_framed_module(
    inputargs: &[InputArgRc],
    ops: &[Op],
    constants: &indexmap::IndexMap<u32, i64>,
    frame: codegen::FrameGeometry,
    ca: codegen::CaParams,
    external_jump_slot: u32,
    external_jump_key: u32,
    external_jump_wide_slot: u32,
) -> (Vec<u8>, Vec<codegen::GuardExit>) {
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops: ops.to_vec(),
        inlined_bridges: Vec::new(),
        constants: constants.clone(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 1,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot,
        external_jump_wide_slot,
        external_jump_key,
        frame,
        ca,
    };
    let (bytes, guards, _, _) = codegen::build_wasm_module(&inputs).expect("wasm codegen");
    (bytes, guards)
}

fn i64_store_offsets(bytes: &[u8]) -> Vec<u64> {
    let mut offsets = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while let Ok(op) = operators.read() {
                if let wasmparser::Operator::I64Store { memarg } = op {
                    offsets.push(memarg.offset);
                }
            }
        }
    }
    offsets
}

fn i32_consts(bytes: &[u8]) -> Vec<i32> {
    let mut values = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while let Ok(op) = operators.read() {
                if let wasmparser::Operator::I32Const { value } = op {
                    values.push(value);
                }
            }
        }
    }
    values
}

/// `build_module_default` for a trace that closes into another module.
fn build_external_jump_module(
    inputargs: &[InputArgRc],
    ops: &[Op],
    constants: &indexmap::IndexMap<u32, i64>,
    external_jump_slot: u32,
    external_jump_key: u32,
    external_jump_wide_slot: u32,
) -> (Vec<u8>, Vec<codegen::GuardExit>) {
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops: ops.to_vec(),
        inlined_bridges: Vec::new(),
        constants: constants.clone(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 1,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot,
        external_jump_wide_slot,
        external_jump_key,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    };
    let (bytes, guards, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    (bytes, guards)
}

/// `i64.store` count of the module's body, and the `(table slot, type index)`
/// of the JUMP's `return_call_indirect`.
///
/// The FIRST one: the body's ops are emitted before the guard-dispatch
/// epilogue, whose own tail call is type 0 through a slot it loaded from a
/// cell rather than a baked constant.
fn external_jump_tail(bytes: &[u8]) -> (usize, Option<(i32, u32)>) {
    let mut stores = 0usize;
    let mut last_const: Option<i32> = None;
    let mut call = None;
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while let Ok(op) = operators.read() {
                match op {
                    wasmparser::Operator::I64Store { .. } => stores += 1,
                    wasmparser::Operator::I32Const { value } => last_const = Some(value),
                    wasmparser::Operator::ReturnCallIndirect { type_index, .. }
                        if call.is_none() =>
                    {
                        call = last_const.map(|slot| (slot, type_index));
                    }
                    _ => {}
                }
            }
        }
    }
    (stores, call)
}

#[test]
fn test_non_last_label_backedge_validates() {
    // Quasi-immutable invalidation can re-trace a loop with a wide entry
    // LABEL followed by a narrower peeled header, while the closing JUMP
    // targets the earlier entry label.  The LABEL/JUMP descr identity, not
    // source position, is the loop target and determines the parallel move.
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let wide_descr = majit_ir::make_loop_target_descr(10, false);
    let narrow_descr = majit_ir::make_loop_target_descr(11, false);

    let wide_label = Op::new(
        OpCode::Label,
        &[rb(OpRef::int_op(1)), rb(OpRef::input_arg_int(0))],
    );
    wide_label.setdescr(wide_descr.clone());
    let narrow_label = Op::new(OpCode::Label, &[rb(OpRef::int_op(2))]);
    narrow_label.setdescr(narrow_descr);
    let jump = Op::new(
        OpCode::Jump,
        &[rb(OpRef::int_op(3)), rb(OpRef::input_arg_int(0))],
    );
    jump.setdescr(wide_descr);

    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        wide_label,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        narrow_label,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        jump,
    ];

    // The wide entry label precedes the header, so it is a resume point; the
    // narrow one sits inside the `loop` and gets no block pair, which is what
    // keeps the wrapper's structured control flow well-formed here.
    assert!(codegen::is_resumable_peeled(&ops));
    assert_eq!(codegen::resumable_label_count(&ops), 1);
    let (bytes, _) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
}

#[test]
fn test_registration_loop_stamps_label_block_id() {
    // The bridge-side target-ordinal recovery rests on one fact: a LABEL and the
    // closing JUMP that targets it share their loop-target descr by Arc identity,
    // so the `label_block_id` `compile_loop` stamps on the LABEL is readable from
    // the JUMP's descr in `compile_bridge`. Reproduce that here: build two LABELs
    // each with its own descr, run the registration loop's stamping (ordinals 0,
    // 1), and confirm the JUMP that shares the second LABEL's descr reads back 1.
    let descr0 = majit_ir::make_loop_target_descr(10, false);
    let descr1 = majit_ir::make_loop_target_descr(11, false);

    let label0 = Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]);
    label0.setdescr(descr0.clone());
    let label1 = Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]);
    label1.setdescr(descr1.clone());
    // A loop-closing bridge's terminal JUMP carries the SAME descr Arc as the
    // label it targets (here, the second/last label).
    let jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(2))]);
    jump.setdescr(descr1.clone());

    // Registration loop (mirrors compile_loop): stamp each LABEL with its ordinal.
    for (ordinal, label) in [&label0, &label1].iter().enumerate() {
        let d = label.getdescr().expect("label has a descr");
        d.as_loop_target_descr()
            .expect("loop-target descr")
            .set_label_block_id(ordinal as u32);
    }

    // Recover the JUMP's target ordinal — it must equal the LAST label's (1),
    // via the shared Arc, NOT 0 (the default) or the first label's ordinal.
    let recovered = jump
        .getdescr()
        .and_then(|d| d.as_loop_target_descr().map(|t| t.label_block_id()));
    assert_eq!(recovered, Some(1));
}

/// Emitted `loop` count and entry-`br_table` arm count, the two things that say
/// whether a module took the resume-`loop` shape.
fn loop_and_br_table_shape(bytes: &[u8]) -> (usize, Vec<usize>) {
    let mut loops = 0usize;
    let mut tables = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::Loop { .. } => loops += 1,
                    wasmparser::Operator::BrTable { targets } => {
                        tables.push(targets.len() as usize)
                    }
                    _ => {}
                }
            }
        }
    }
    (loops, tables)
}

/// A two-label peeled owner (`preamble; LABEL0; segment; LABEL1(header); body;
/// JUMP->LABEL1`) with one inlined region, whose closing JUMP names `which`.
fn build_owner_with_region_closing_at(
    which: usize,
) -> Result<Vec<u8>, majit_backend::BackendError> {
    let descr0 = majit_ir::make_loop_target_descr(10, false);
    let descr1 = majit_ir::make_loop_target_descr(11, false);

    let label0 = Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]);
    label0.setdescr(descr0.clone());
    let label1 = Op::new(OpCode::Label, &[rb(OpRef::int_op(2))]);
    label1.setdescr(descr1.clone());
    let jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(3))]);
    jump.setdescr(descr1.clone());

    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        label0,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        label1,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), OpRef::const_int(100)],
            OpRef::int_op(4),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(4)], &[OpRef::int_op(3)]),
        jump,
    ];
    // Both LABELs precede the header, so both are resume points.
    assert!(codegen::is_resumable_peeled(&ops));
    assert_eq!(codegen::resumable_label_count(&ops), 2);

    let region_jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(11))]);
    region_jump.setdescr(if which == 0 { descr0 } else { descr1 });
    let region_ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(10), OpRef::const_int(1)],
            OpRef::int_op(11),
        ),
        region_jump,
    ];

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let inputs = inline_region_inputs(
        &inputargs,
        ops,
        vec![codegen::InlinedBridge {
            external_jump: None,
            source_fail_index: 0,
            outside_loop: false,
            trace_id: 1,
            inputargs: vec![InputArg::from_type_rc(Type::Int, 10)],
            ops: region_ops,
            gc_table_base: 0,
            gc_const_keys: Vec::new(),
            constants: indexmap::IndexMap::new(),
        }],
    );
    codegen::build_wasm_module(&inputs).map(|(bytes, _, _, _)| bytes)
}

/// A region closing at the loop HEADER `br`s to the `loop`, the long-standing
/// shape: one `loop`, and the entry `br_table` keeps its one arm per label plus
/// the fresh-entry bucket.
#[test]
fn region_closing_at_the_header_keeps_the_single_loop_shape() {
    let bytes = build_owner_with_region_closing_at(1).expect("header-closing region merges");
    validate_wasm(&bytes);
    let (loops, tables) = loop_and_br_table_shape(&bytes);
    assert_eq!(loops, 1, "only the loop header opens a `loop`");
    assert_eq!(
        tables,
        vec![3],
        "entry br_table: key 0 plus one arm per label"
    );
}

/// A region closing at a NON-header LABEL cannot `br` to the `loop` — that
/// would skip the segment between its label and the header. It re-enters the
/// entry dispatch instead, so the module grows a second `loop` around that
/// dispatch and a second `br_table` bucket per label (`num_labels + 1 + j`,
/// landing past label j's resume loader with the args already in locals).
#[test]
fn region_closing_at_a_non_header_label_wraps_the_dispatch_in_a_loop() {
    let bytes = build_owner_with_region_closing_at(0).expect("non-header region merges");
    validate_wasm(&bytes);
    let (loops, tables) = loop_and_br_table_shape(&bytes);
    assert_eq!(loops, 2, "the resume `loop` wraps the entry dispatch");
    assert_eq!(
        tables,
        vec![5],
        "entry br_table: key 0, one loader arm per label, one past-loader arm per label"
    );
}

/// Execute a two-label peeled loop whose inlined region closes at the NON-header
/// LABEL, and report the three values the exit guard spills. The region's
/// re-entry must land at label 0 and then run the segment between label 0 and
/// the header, exactly as an out-of-line bridge resuming at key 1 would.
///
///   preamble  v1 = v0 + 1
///   LABEL0    [v1]
///   segment   v2 = v1 + 1
///   LABEL1    [v2]                     <- header, the `loop`
///   body      v3 = v2 + 1
///             guard v3 > 10            <- fail 0, the region
///             guard v3 < 1000          <- fail 1, exits with [v3, v2, v1]
///             JUMP -> LABEL1 [v3]
///   region    v11 = v10 + 5000; JUMP -> LABEL0 [v11]
///
/// v0 = 0 makes v3 = 3, so the first guard fails into the region, which
/// re-enters at LABEL0 with 5003. The segment then makes v2 = 5004 and the body
/// v3 = 5005, which passes the first guard and fails the second.
#[test]
fn region_closing_at_a_non_header_label_reenters_and_runs_the_segment() {
    assert_eq!(run_non_header_region_repro(false), (5005, 5004, 5003));
}

/// The same trace with a Ref label arg carried through both LABELs and rebound
/// by the region's JUMP, so the region's back edge has to refresh a Ref home
/// exactly as the resume loader does.
#[test]
fn region_closing_at_a_non_header_label_carries_a_ref_label_arg() {
    assert_eq!(run_non_header_region_repro(true), (5005, 5004, 5003));
}

fn run_non_header_region_repro(with_ref: bool) -> (i64, i64, i64) {
    let descr0 = majit_ir::make_loop_target_descr(20, false);
    let descr1 = majit_ir::make_loop_target_descr(21, false);

    // v6 is a Ref inputarg threaded through both LABELs unchanged.
    let (label0_args, label1_args, jump_args, region_jump_args) = if with_ref {
        (
            vec![rb(OpRef::int_op(1)), rb(OpRef::ref_op(6))],
            vec![rb(OpRef::int_op(2)), rb(OpRef::ref_op(6))],
            vec![rb(OpRef::int_op(3)), rb(OpRef::ref_op(6))],
            vec![rb(OpRef::int_op(11)), rb(OpRef::ref_op(12))],
        )
    } else {
        (
            vec![rb(OpRef::int_op(1))],
            vec![rb(OpRef::int_op(2))],
            vec![rb(OpRef::int_op(3))],
            vec![rb(OpRef::int_op(11))],
        )
    };
    let label0 = Op::new(OpCode::Label, &label0_args);
    label0.setdescr(descr0.clone());
    let label1 = Op::new(OpCode::Label, &label1_args);
    label1.setdescr(descr1.clone());
    let jump = Op::new(OpCode::Jump, &jump_args);
    jump.setdescr(descr1);

    let region_failargs: Vec<_> = if with_ref {
        vec![OpRef::int_op(3), OpRef::ref_op(6)]
    } else {
        vec![OpRef::int_op(3)]
    };
    let guard_to_region = make_guard(OpCode::GuardTrue, &[OpRef::int_op(4)], &region_failargs);
    let guard_exit = make_guard(
        OpCode::GuardTrue,
        &[OpRef::int_op(5)],
        &[OpRef::int_op(3), OpRef::int_op(2), OpRef::int_op(1)],
    );
    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        label0,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        label1,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntGt,
            &[OpRef::int_op(3), OpRef::const_int(10)],
            OpRef::int_op(4),
        ),
        guard_to_region,
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), OpRef::const_int(1000)],
            OpRef::int_op(5),
        ),
        guard_exit,
        jump,
    ];
    assert_eq!(codegen::resumable_label_count(&ops), 2);

    let region_jump = Op::new(OpCode::Jump, &region_jump_args);
    region_jump.setdescr(descr0);
    let mut region_ops = vec![make_op(
        OpCode::IntAdd,
        &[OpRef::input_arg_int(10), OpRef::const_int(5000)],
        OpRef::int_op(11),
    )];
    if with_ref {
        // The region rebinds the Ref label arg to its own live-in copy, so the
        // back edge must move it AND refresh its home.
        region_ops.push(make_op(
            OpCode::SameAsR,
            &[OpRef::ref_op(13)],
            OpRef::ref_op(12),
        ));
    }
    region_ops.push(region_jump);

    let mut inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    if with_ref {
        inputargs.push(InputArg::from_type_rc(Type::Ref, 6));
    }
    let inputs = inline_region_inputs(
        &inputargs,
        ops,
        vec![codegen::InlinedBridge {
            external_jump: None,
            source_fail_index: 0,
            outside_loop: false,
            trace_id: 1,
            inputargs: if with_ref {
                vec![
                    InputArg::from_type_rc(Type::Int, 10),
                    InputArg::from_type_rc(Type::Ref, 13),
                ]
            } else {
                vec![InputArg::from_type_rc(Type::Int, 10)]
            },
            ops: region_ops,
            gc_table_base: 0,
            gc_const_keys: Vec::new(),
            constants: indexmap::IndexMap::new(),
        }],
    );
    let (exit_index, slot0, slot1, slot2) = run_inline_region_trace(&inputs);
    assert_eq!(exit_index, 1, "the second guard is the one that exits");
    (slot0, slot1, slot2)
}

/// Collect every `i32.const` / `i64.const` immediate in the emitted body, so a
/// test can assert which helper address the allocation arms baked in.
fn const_immediates(bytes: &[u8]) -> Vec<i64> {
    let mut out = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::I32Const { value } => out.push(value as i64),
                    wasmparser::Operator::I64Const { value } => out.push(value),
                    _ => {}
                }
            }
        }
    }
    out
}

#[test]
fn test_non_moving_descr_allocates_through_the_oldgen_helper() {
    use majit_ir::descr::{SimpleArrayDescr, SimpleSizeDescr};
    use std::sync::Arc;

    let size_descr = SimpleSizeDescr::new(0, 32, 53);
    size_descr.set_non_moving(true);
    let array_descr = SimpleArrayDescr::new(1, 8, 8, 55, Type::Ref);
    array_descr.set_non_moving(true);

    let new_op = make_op(OpCode::New, &[], OpRef::ref_op(1));
    new_op.setdescr(Arc::new(size_descr));
    let new_array_op = make_op(
        OpCode::NewArrayClear,
        &[OpRef::input_arg_int(0)],
        OpRef::ref_op(2),
    );
    new_array_op.setdescr(Arc::new(array_descr));
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]);
    finish.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let (ops, constants) = rewrite_frontend_ops(&inputargs, vec![new_op, new_array_op, finish]);
    let call_r_targets: Vec<i64> = ops
        .iter()
        .filter(|op| op.opcode == OpCode::CallR)
        .filter_map(|op| {
            op.arg(0).to_opref().inline_const_bits().or_else(|| {
                let r = op.arg(0).to_opref();
                r.is_constant()
                    .then(|| constants.get(&r.raw()).copied())
                    .flatten()
            })
        })
        .collect();
    let fixed_oldgen = majit_backend_wasm::wasm_malloc_big_fixedsize_oldgen as *const () as i64;
    let array_oldgen = majit_backend_wasm::wasm_malloc_array_nonstandard_oldgen as *const () as i64;
    let array_oldgen_std = majit_backend_wasm::wasm_malloc_array_oldgen as *const () as i64;
    assert!(
        call_r_targets.contains(&fixed_oldgen),
        "non_moving New must CallR wasm_malloc_big_fixedsize_oldgen (rewrite.rs gen_malloc_fixedsize): {call_r_targets:?}"
    );
    assert!(
        call_r_targets.contains(&array_oldgen) || call_r_targets.contains(&array_oldgen_std),
        "non_moving NewArrayClear must CallR an oldgen array helper: {call_r_targets:?}"
    );

    let mut inputs = inline_region_inputs(&inputargs, ops, vec![]);
    inputs.constants = constants;
    inputs.frame = codegen::FrameGeometry::compact(8, 4, 0);
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert_eq!(
        import_func_type(&bytes, "jit_call"),
        None,
        "oldgen malloc must be an in-module call_indirect, not jit_call"
    );
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
    assert!(
        has_table_import(&bytes),
        "oldgen malloc is call_indirect through the residual table"
    );
    let (calls, _) = indirect_call_types_and_drop_count(&bytes);
    assert!(
        !calls.is_empty(),
        "non_moving births must emit at least one in-module call_indirect"
    );
}

/// Minimal `FailDescr` for driving `Backend::compile_bridge` from the host.
/// `trace_id` defaults to 0, which is also a freshly compiled loop's, so the
/// bridge reads as a direct guard of the source loop.
#[derive(Debug)]
struct HostFailDescr {
    fail_index: u32,
    arg_types: Vec<Type>,
}

impl majit_ir::Descr for HostFailDescr {}

impl majit_ir::descr::FailDescr for HostFailDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }

    fn fail_arg_types(&self) -> &[Type] {
        &self.arg_types
    }
}

/// A two-input loop whose single LABEL sits at the entry, so
/// `stamp_and_publish_label_targets` publishes it as a re-enterable target and
/// a bridge closing onto `label_descr` resolves in-module.
fn host_loop_ops(label_descr: &std::sync::Arc<dyn majit_ir::Descr>) -> Vec<OpRc> {
    let label = OpRc::new(Op::new(
        OpCode::Label,
        &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_int(1))],
    ));
    label.setdescr(label_descr.clone());
    let advance = OpRc::new(make_op(
        OpCode::IntAdd,
        &[OpRef::input_arg_int(0), OpRef::const_int(1)],
        OpRef::int_op(2),
    ));
    let guard = OpRc::new(make_guard(
        OpCode::GuardTrue,
        &[OpRef::int_op(2)],
        &[OpRef::int_op(2), OpRef::input_arg_int(1)],
    ));
    // Bind the JUMP to the real producer, not to a synthetic stand-in: the
    // loop-carried advance is read off the arg's producing opcode.
    let jump = OpRc::new(Op::new(
        OpCode::Jump,
        &[
            Operand::from_bound_op(&advance),
            rb(OpRef::input_arg_int(1)),
        ],
    ));
    jump.setdescr(label_descr.clone());
    vec![label, advance, guard, jump]
}

/// A loop-closing bridge for the loop above: it advances one loop-carried
/// value and jumps back onto the owner's published label.
fn host_bridge_ops(label_descr: &std::sync::Arc<dyn majit_ir::Descr>) -> Vec<OpRc> {
    let advance = OpRc::new(make_op(
        OpCode::IntAdd,
        &[OpRef::input_arg_int(40), OpRef::const_int(1)],
        OpRef::int_op(42),
    ));
    let jump = OpRc::new(Op::new(
        OpCode::Jump,
        &[
            Operand::from_bound_op(&advance),
            rb(OpRef::input_arg_int(41)),
        ],
    ));
    jump.setdescr(label_descr.clone());
    vec![advance, jump]
}

/// Same close as [`host_bridge_ops`], plus `GUARD_NOT_INVALIDATED` so the
/// region would take the eager-merge arm if the owner were still small.
fn host_gni_bridge_ops(label_descr: &std::sync::Arc<dyn majit_ir::Descr>) -> Vec<OpRc> {
    let gni = OpRc::new(Op::new(OpCode::GuardNotInvalidated, &[]));
    gni.setfailargs(smallvec![
        rb(OpRef::input_arg_int(40)),
        rb(OpRef::input_arg_int(41)),
    ]);
    let mut ops = host_bridge_ops(label_descr);
    ops.insert(0, gni);
    ops
}

/// `host_loop_ops` plus a chain of `IntAdd`s. Each result feeds the next, and
/// the last feeds the guard and the JUMP, so none of the adds are dead. The
/// chain is long enough that the emitted module exceeds the eager-merge
/// ceiling (`DEFAULT_INLINE_EAGER_MAX_BYTES`, 4096).
fn host_oversized_loop_ops(label_descr: &std::sync::Arc<dyn majit_ir::Descr>) -> Vec<OpRc> {
    let mut ops = host_loop_ops(label_descr);
    let mut producer = ops[1].clone();
    // `host_loop_ops` is [label, advance, guard, jump]. The chain replaces
    // the guard and the JUMP so both read the last add.
    ops.truncate(2);
    const EXTRA: u32 = 700;
    for i in 0..EXTRA {
        let next = Op::new(
            OpCode::IntAdd,
            &[
                Operand::from_bound_op(&producer),
                rb(OpRef::input_arg_int(1)),
            ],
        );
        let pos = OpRef::int_op(3 + i);
        next.pos().set(pos);
        producer = OpRc::new(next);
        ops.push(producer.clone());
    }
    let carried = producer.pos().get();
    let guard = OpRc::new(make_guard(
        OpCode::GuardTrue,
        &[carried],
        &[carried, OpRef::input_arg_int(1)],
    ));
    let jump = OpRc::new(Op::new(
        OpCode::Jump,
        &[
            Operand::from_bound_op(&producer),
            rb(OpRef::input_arg_int(1)),
        ],
    ));
    jump.setdescr(label_descr.clone());
    ops.push(guard);
    ops.push(jump);
    ops
}

fn host_bridge_inputargs() -> Vec<InputArgRc> {
    vec![
        InputArg::from_type_rc(Type::Int, 40),
        InputArg::from_type_rc(Type::Int, 41),
    ]
}

/// The global fail-descr space is appended to under the assumption that no
/// other compile interleaves (`failguard.rs register_fail_descrs`), which the
/// single-threaded wasm host guarantees and a parallel test runner does not.
static HOST_COMPILE_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

fn host_loop_inputargs() -> Vec<InputArgRc> {
    vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ]
}

/// Header, preamble, and large-header owners all defer the inline trial
/// until the entry-count trip, including an invalidation-guard region.
#[test]
fn a_valid_owner_defers_the_inline_trial() {
    let cases = [
        ("header", false, false),
        ("preamble_invalidation", true, false),
        ("large_header_invalidation", false, true),
    ];
    for (name, preamble, large_header) in cases {
        assert_valid_owner_defers_inline_trial(name, preamble, large_header);
    }
}

fn assert_valid_owner_defers_inline_trial(case: &str, preamble: bool, large_header: bool) {
    use majit_backend::Backend;

    let _serialized = HOST_COMPILE_LOCK.lock();
    let mut backend = majit_backend_wasm::WasmBackend::new();
    // Through an `Arc`, with the compiled-loop token wired back to it the way
    // `make_jitcell_token` does: a deferred merge outlives the compile that
    // registered it, so it takes a strong handle on the owner from there.
    let token = std::sync::Arc::new(majit_backend::JitCellToken::new(1));
    let clt = std::sync::Arc::new(majit_backend::CompiledLoopToken::new(1));
    clt.set_loop_token_wref(std::sync::Arc::downgrade(&token));
    token.set_compiled_loop_token(Some(clt));
    let label_descr = majit_ir::make_loop_target_descr(70, false);
    let mut loop_ops = if large_header {
        host_oversized_loop_ops(&label_descr)
    } else {
        host_loop_ops(&label_descr)
    };
    let mut bridge_target = label_descr.clone();
    if preamble {
        bridge_target = majit_ir::make_loop_target_descr(73, false);
        let entry = OpRc::new(Op::new(
            OpCode::Label,
            &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_int(1))],
        ));
        entry.setdescr(bridge_target.clone());
        loop_ops.insert(
            0,
            OpRc::new(make_guard(
                OpCode::GuardTrue,
                &[OpRef::input_arg_int(0)],
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            )),
        );
        loop_ops.insert(0, entry);
    }
    let mut bridge_ops = host_bridge_ops(&bridge_target);
    if preamble || large_header {
        bridge_ops.insert(
            0,
            OpRc::new(make_guard(
                OpCode::GuardNotInvalidated,
                &[],
                &[OpRef::input_arg_int(40), OpRef::input_arg_int(41)],
            )),
        );
    }

    let owner = backend
        .compile_loop(&host_loop_inputargs(), &loop_ops, &token)
        .unwrap_or_else(|_| panic!("case {case}: the owner loop compiles"));
    if large_header {
        assert!(
            owner.code_size > 4096,
            "case {case}: owner module is {} bytes; the eager ceiling is 4096",
            owner.code_size
        );
    }
    assert!(!token.is_invalidated(), "case {case}");

    let fail_descr = HostFailDescr {
        fail_index: 0,
        arg_types: vec![Type::Int, Type::Int],
    };
    let declines_before = majit_backend_wasm::bridge_diag(50);
    let deferred_before = majit_backend_wasm::bridge_diag(54);
    let trials_before = majit_backend_wasm::bridge_diag(37);
    let invalidation_declines_before = majit_backend_wasm::bridge_diag(56);
    // A merge is only considered once there is a callback to act on the
    // entry-count trip; the guest publishes the real one.
    majit_backend_wasm::set_inline_trip_helper_slot(1);
    let compiled = backend.compile_bridge(
        &fail_descr,
        &host_bridge_inputargs(),
        &bridge_ops,
        &token,
        None,
    );
    majit_backend_wasm::set_inline_trip_helper_slot(0);
    compiled.unwrap_or_else(|_| panic!("case {case}: the loop-closing bridge compiles"));

    assert_eq!(
        majit_backend_wasm::bridge_diag(50),
        declines_before,
        "case {case}: a valid owner is not declined by the invalidation arm"
    );
    assert!(
        majit_backend_wasm::bridge_diag(54) > deferred_before,
        "case {case}: a valid region waits on the entry-count trip: {}",
        majit_backend_wasm::inline_declines()
    );
    assert_eq!(
        majit_backend_wasm::bridge_diag(37),
        trials_before,
        "case {case}: the deferred region has not tried to rebuild the owner yet"
    );
    assert_eq!(
        majit_backend_wasm::bridge_diag(56),
        invalidation_declines_before,
        "case {case}"
    );
    let generation = token.latest_bridge_invalidation_flag().unwrap();
    assert!(
        !generation.load(std::sync::atomic::Ordering::Acquire),
        "case {case}"
    );
    token.invalidate();
    assert!(token.is_invalidated(), "case {case}");
    assert!(
        generation.load(std::sync::atomic::Ordering::Acquire),
        "case {case}"
    );
}

/// After a peel has already grown the owner past the eager-merge ceiling, a
/// later invalidation-watched region used to refuse the eager arm and never
/// register a trip. That left `exception_loop_warmup`'s raise path as
/// a permanent crossing. The size check still refuses the unmeasured
/// re-emission; it now arms the same trip as the no-GNI deferral.
#[test]
fn an_oversized_owner_defers_a_gni_region_instead_of_dropping_it() {
    use majit_backend::Backend;

    let _serialized = HOST_COMPILE_LOCK.lock();
    let mut backend = majit_backend_wasm::WasmBackend::new();
    let token = std::sync::Arc::new(majit_backend::JitCellToken::new(1));
    let clt = std::sync::Arc::new(majit_backend::CompiledLoopToken::new(1));
    clt.set_loop_token_wref(std::sync::Arc::downgrade(&token));
    token.set_compiled_loop_token(Some(clt));
    let label_descr = majit_ir::make_loop_target_descr(71, false);

    let owner = backend
        .compile_loop(
            &host_loop_inputargs(),
            &host_oversized_loop_ops(&label_descr),
            &token,
        )
        .expect("the owner loop compiles");
    assert!(
        owner.code_size > 4096,
        "owner module is {} bytes; the eager ceiling is 4096",
        owner.code_size
    );

    let fail_descr = HostFailDescr {
        fail_index: 0,
        arg_types: vec![Type::Int, Type::Int],
    };
    let deferred_before = majit_backend_wasm::bridge_diag(54);
    let inline_ok_before = majit_backend_wasm::bridge_diag(32);
    let pending_before = majit_backend_wasm::pending_inline_count(&token);
    majit_backend_wasm::set_inline_trip_helper_slot(1);
    backend
        .compile_bridge(
            &fail_descr,
            &host_bridge_inputargs(),
            &host_gni_bridge_ops(&label_descr),
            &token,
            None,
        )
        .expect("the GNI bridge still compiles out of line");
    majit_backend_wasm::set_inline_trip_helper_slot(0);

    assert!(
        majit_backend_wasm::bridge_diag(54) > deferred_before,
        "an oversized owner waits on the entry-count trip, not an unmeasured merge: {}",
        majit_backend_wasm::inline_declines()
    );
    assert_eq!(
        majit_backend_wasm::bridge_diag(32),
        inline_ok_before,
        "an oversized owner is not merged before the trip"
    );
    assert!(
        majit_backend_wasm::pending_inline_count(&token) > pending_before,
        "the trip is armed so the raise path can still merge once hot"
    );
}

/// `model.py:145-152`, pinned upstream by `runner_test.py
/// test_guard_not_invalidated` steps 3 and 4: a bridge compiled AFTER
/// `invalidate_loop` starts valid, and only a LATER invalidation activates its
/// GUARD_NOT_INVALIDATED.
///
/// A merged region cannot honour that. It runs from the owner's module, whose
/// `invalidated_flag_addr` is the owner's root flag — already set once the
/// owner is invalidated — so the region would be dead the moment it is
/// installed. The inline arm therefore declines on an invalidated owner and
/// leaves the out-of-line path to mint the fresh, clear generation.
#[test]
fn a_bridge_compiled_after_the_owner_was_invalidated_starts_valid() {
    use majit_backend::Backend;

    let _serialized = HOST_COMPILE_LOCK.lock();
    let mut backend = majit_backend_wasm::WasmBackend::new();
    let token = majit_backend::JitCellToken::new(1);
    let label_descr = majit_ir::make_loop_target_descr(71, false);

    backend
        .compile_loop(&host_loop_inputargs(), &host_loop_ops(&label_descr), &token)
        .expect("the owner loop compiles");
    assert!(
        token.latest_bridge_invalidation_flag().is_none(),
        "no bridge generation exists before any bridge is compiled"
    );

    token.invalidate();
    assert!(token.is_invalidated());

    let fail_descr = HostFailDescr {
        fail_index: 0,
        arg_types: vec![Type::Int, Type::Int],
    };
    let declines_before = majit_backend_wasm::bridge_diag(50);
    backend
        .compile_bridge(
            &fail_descr,
            &host_bridge_inputargs(),
            &host_bridge_ops(&label_descr),
            &token,
            None,
        )
        .expect("the loop-closing bridge compiles out of line");

    assert!(
        majit_backend_wasm::bridge_diag(50) > declines_before,
        "the inline arm must decline an invalidated owner"
    );
    let generation = token
        .latest_bridge_invalidation_flag()
        .expect("the out-of-line path mints a generation for this bridge");
    assert!(
        !generation.load(std::sync::atomic::Ordering::Acquire),
        "a bridge compiled after invalidate_loop starts valid"
    );
}

/// The same two-label shape, but with a backend-only live-in captured at the
/// label the region resumes at: `v7` is produced before LABEL0, is not one of
/// its args, and is read in the loop body. The region's back edge must restore
/// it from its capture slot exactly as the entry `br_table`'s resume loader
/// does, so the body's `v3 + v7` reports 5005 + 100.
#[test]
fn region_closing_at_a_non_header_label_restores_that_labels_captures() {
    assert_eq!(run_non_header_capture_repro(), (5005, 5004, 5105));
}

fn run_non_header_capture_repro() -> (i64, i64, i64) {
    let descr0 = majit_ir::make_loop_target_descr(30, false);
    let descr1 = majit_ir::make_loop_target_descr(31, false);

    let label0 = Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]);
    label0.setdescr(descr0.clone());
    let label1 = Op::new(OpCode::Label, &[rb(OpRef::int_op(2))]);
    label1.setdescr(descr1.clone());
    let jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(3))]);
    jump.setdescr(descr1);

    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        // Produced before LABEL0, not one of its args, read after the header:
        // the capture plan must hold it across both resume paths.
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(100)],
            OpRef::int_op(7),
        ),
        label0,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        label1,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(3), OpRef::int_op(7)],
            OpRef::int_op(8),
        ),
        make_op(
            OpCode::IntGt,
            &[OpRef::int_op(3), OpRef::const_int(10)],
            OpRef::int_op(4),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(4)], &[OpRef::int_op(3)]),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), OpRef::const_int(1000)],
            OpRef::int_op(5),
        ),
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(5)],
            &[OpRef::int_op(3), OpRef::int_op(2), OpRef::int_op(8)],
        ),
        jump,
    ];
    assert_eq!(codegen::resumable_label_count(&ops), 2);

    let region_jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(11))]);
    region_jump.setdescr(descr0);
    let region_ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(10), OpRef::const_int(5000)],
            OpRef::int_op(11),
        ),
        region_jump,
    ];

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let inputs = inline_region_inputs(
        &inputargs,
        ops,
        vec![codegen::InlinedBridge {
            external_jump: None,
            source_fail_index: 0,
            outside_loop: false,
            trace_id: 1,
            inputargs: vec![InputArg::from_type_rc(Type::Int, 10)],
            ops: region_ops,
            gc_table_base: 0,
            gc_const_keys: Vec::new(),
            constants: indexmap::IndexMap::new(),
        }],
    );
    let (exit_index, slot0, slot1, slot2) = run_inline_region_trace(&inputs);
    assert_eq!(exit_index, 1, "the second guard is the one that exits");
    (slot0, slot1, slot2)
}

/// Two regions attached to the SAME owner, both closing at the NON-header
/// LABEL. Each region owns one of the blocks opened at the loop header, so the
/// guard that reaches it and the back edge it takes must both name that
/// region's own depth — region 0 innermost.
///
///   preamble  v1 = v0 + 1
///   LABEL0    [v1]
///   segment   v2 = v1 + 1
///   LABEL1    [v2]                     <- header, the `loop`
///   body      v3 = v2 + 1
///             guard v3 > 10            <- fail 0, region A
///             guard v3 > 6000          <- fail 1, region B
///             guard v3 < 100000        <- fail 2, exits with [v3, v2, v1]
///             JUMP -> LABEL1 [v3]
///   region A  v11 = v10 + 5000;  JUMP -> LABEL0 [v11]
///   region B  v21 = v20 + 50000; JUMP -> LABEL0 [v21]
#[test]
fn two_regions_closing_at_a_non_header_label_each_reenter_at_their_own_depth() {
    assert_eq!(run_two_non_header_regions_repro(), (100000, 99999, 55005));
}

fn run_two_non_header_regions_repro() -> (i64, i64, i64) {
    let descr0 = majit_ir::make_loop_target_descr(40, false);
    let descr1 = majit_ir::make_loop_target_descr(41, false);

    let label0 = Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]);
    label0.setdescr(descr0.clone());
    let label1 = Op::new(OpCode::Label, &[rb(OpRef::int_op(2))]);
    label1.setdescr(descr1.clone());
    let jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(3))]);
    jump.setdescr(descr1);

    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        label0,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        label1,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntGt,
            &[OpRef::int_op(3), OpRef::const_int(10)],
            OpRef::int_op(4),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(4)], &[OpRef::int_op(3)]),
        make_op(
            OpCode::IntGt,
            &[OpRef::int_op(3), OpRef::const_int(6000)],
            OpRef::int_op(5),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(5)], &[OpRef::int_op(3)]),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), OpRef::const_int(100000)],
            OpRef::int_op(6),
        ),
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(6)],
            &[OpRef::int_op(3), OpRef::int_op(2), OpRef::int_op(1)],
        ),
        jump,
    ];
    assert_eq!(codegen::resumable_label_count(&ops), 2);

    let region_a_jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(11))]);
    region_a_jump.setdescr(descr0.clone());
    let region_a = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(10), OpRef::const_int(5000)],
            OpRef::int_op(11),
        ),
        region_a_jump,
    ];
    let region_b_jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(21))]);
    region_b_jump.setdescr(descr0);
    let region_b = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(20), OpRef::const_int(50000)],
            OpRef::int_op(21),
        ),
        region_b_jump,
    ];

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let inputs = inline_region_inputs(
        &inputargs,
        ops,
        vec![
            codegen::InlinedBridge {
                external_jump: None,
                source_fail_index: 0,
                outside_loop: false,
                trace_id: 1,
                inputargs: vec![InputArg::from_type_rc(Type::Int, 10)],
                ops: region_a,
                gc_table_base: 0,
                gc_const_keys: Vec::new(),
                constants: indexmap::IndexMap::new(),
            },
            codegen::InlinedBridge {
                external_jump: None,
                source_fail_index: 1,
                outside_loop: false,
                trace_id: 2,
                inputargs: vec![InputArg::from_type_rc(Type::Int, 20)],
                ops: region_b,
                gc_table_base: 0,
                gc_const_keys: Vec::new(),
                constants: indexmap::IndexMap::new(),
            },
        ],
    );
    let (exit_index, slot0, slot1, slot2) = run_inline_region_trace(&inputs);
    assert_eq!(exit_index, 2, "the third guard is the one that exits");
    (slot0, slot1, slot2)
}

/// A region attached to a guard in the PEELED PREAMBLE, ahead of the loop
/// header LABEL. The per-region blocks the loop-body regions use are opened
/// inside the wasm `loop`, which the preamble has not entered, so such a
/// region needs its block outside that loop.
///
/// The trace increments once before LABEL0, once more between LABEL0 and the
/// header, and guards `v2 > 10` there. The first pass has `v2 == 2`, so the
/// guard fails into the region, which adds 5000 and re-enters at LABEL0 with
/// the sum. The second pass reaches the header with `v2 == 5003` and the loop
/// runs to its own exit guard, so the frame reports the region's arithmetic
/// rather than the header loader's zeroed slots.
#[test]
fn a_region_attached_to_a_preamble_guard_runs_its_body() {
    assert_eq!(run_preamble_region_repro(), (5004, 5003, 5002));
}

fn run_preamble_region_repro() -> (i64, i64, i64) {
    let inputs = preamble_region_inputs();
    let (exit_index, slot0, slot1, slot2) = run_inline_region_trace(&inputs);
    assert_eq!(exit_index, 1, "the second guard is the one that exits");
    (slot0, slot1, slot2)
}

/// llgraph/runner.py::invalidate_loop activates guards in attached traces too.
/// Mutating the token after module compilation changes its next execution.
#[test]
fn an_inlined_preamble_region_observes_later_token_invalidation() {
    let mut inputs = preamble_region_inputs();
    inputs.invalidated_flag_addr = 0x1000;
    inputs.inlined_bridges[0].ops.insert(
        0,
        make_guard(
            OpCode::GuardNotInvalidated,
            &[],
            &[OpRef::input_arg_int(10)],
        ),
    );
    let (bytes, _, _, _) = codegen::build_wasm_module(&inputs).unwrap();
    validate_wasm(&bytes);
    let engine = Engine::default();
    let module = Module::new(&engine, &bytes).unwrap();
    let mut store = Store::new(&engine, ());
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
    let mut linker = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    let instance = linker.instantiate_and_start(&mut store, &module).unwrap();
    let trace = instance
        .get_typed_func::<i32, i32>(&store, "trace")
        .unwrap();
    for (invalidated, exit_index, value) in [(0u8, 1i64, 5004i64), (1, 2, 2), (1, 2, 2)] {
        memory.write(&mut store, 0x1000, &[invalidated]).unwrap();
        memory
            .write(
                &mut store,
                codegen::FRAME_SLOT_BASE as usize,
                &0i64.to_le_bytes(),
            )
            .unwrap();
        trace.call(&mut store, 0).unwrap();
        let mut word = [0u8; 8];
        memory.read(&store, 0, &mut word).unwrap();
        assert_eq!(i64::from_le_bytes(word), exit_index);
        memory
            .read(&store, codegen::FRAME_SLOT_BASE as usize, &mut word)
            .unwrap();
        assert_eq!(i64::from_le_bytes(word), value);
    }
}

fn preamble_region_inputs() -> codegen::ModuleBuildInputs {
    let descr0 = majit_ir::make_loop_target_descr(50, false);
    let descr1 = majit_ir::make_loop_target_descr(51, false);

    let label0 = Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]);
    label0.setdescr(descr0.clone());
    let label1 = Op::new(OpCode::Label, &[rb(OpRef::int_op(2))]);
    label1.setdescr(descr1.clone());
    let jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(3))]);
    jump.setdescr(descr1);

    let guard_to_region = make_guard(OpCode::GuardTrue, &[OpRef::int_op(4)], &[OpRef::int_op(2)]);
    let guard_exit = make_guard(
        OpCode::GuardTrue,
        &[OpRef::int_op(5)],
        &[OpRef::int_op(3), OpRef::int_op(2), OpRef::int_op(1)],
    );
    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        label0,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        make_op(
            OpCode::IntGt,
            &[OpRef::int_op(2), OpRef::const_int(10)],
            OpRef::int_op(4),
        ),
        guard_to_region,
        label1,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), OpRef::const_int(1000)],
            OpRef::int_op(5),
        ),
        guard_exit,
        jump,
    ];
    assert_eq!(codegen::resumable_label_count(&ops), 2);

    let region_jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(11))]);
    region_jump.setdescr(descr0);
    let region_ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(10), OpRef::const_int(5000)],
            OpRef::int_op(11),
        ),
        region_jump,
    ];

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    inline_region_inputs(
        &inputargs,
        ops,
        vec![codegen::InlinedBridge {
            external_jump: None,
            source_fail_index: 0,
            outside_loop: true,
            trace_id: 1,
            inputargs: vec![InputArg::from_type_rc(Type::Int, 10)],
            ops: region_ops,
            gc_table_base: 0,
            gc_const_keys: Vec::new(),
            constants: indexmap::IndexMap::new(),
        }],
    )
}

/// Which of the two region-block families a guard's region belongs to. A guard
/// inside the `loop` reaches the blocks opened there; one in the peeled
/// preamble, which has not entered that `loop`, takes the blocks opened outside
/// it instead. The exit ordinal is what decides.
#[test]
fn a_preamble_guard_is_classified_apart_from_a_loop_body_one() {
    let descr0 = majit_ir::make_loop_target_descr(50, false);
    let descr1 = majit_ir::make_loop_target_descr(51, false);

    let label0 = Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]);
    label0.setdescr(descr0);
    let label1 = Op::new(OpCode::Label, &[rb(OpRef::int_op(2))]);
    label1.setdescr(descr1.clone());
    let jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(3))]);
    jump.setdescr(descr1);

    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        label0,
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(1), OpRef::const_int(50)],
            OpRef::int_op(6),
        ),
        // exit 0: in the preamble, between the two LABELs.
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(6)], &[OpRef::int_op(1)]),
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        label1,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), OpRef::const_int(100)],
            OpRef::int_op(4),
        ),
        // exit 1: in the loop body.
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(4)], &[OpRef::int_op(3)]),
        jump,
    ];

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let inputs = inline_region_inputs(&inputargs, ops, vec![]);
    assert!(codegen::source_guard_precedes_loop_label(&inputs.ops, 0));
    assert!(!codegen::source_guard_precedes_loop_label(&inputs.ops, 1));
}

/// One region in each family at once: region A hangs off a PREAMBLE guard, so
/// its block is opened outside the header `loop`, while region B hangs off a
/// loop-body guard and keeps its block inside it. Both close at LABEL0, so
/// both leave through the entry dispatch, and each has to reach its own block
/// from a different nesting depth.
///
/// B is listed first because merging is append-only and A's body is emitted
/// past the `end` of the loop B's body sits inside, so every outside-loop
/// region trails every inside-loop one.
///
///   entry     v1 = in0 + 1
///   LABEL0    [v1]
///             v2 = v1 + 1;  guard v2 > 10   -> region A
///   LABEL1    [v2]                             (loop header)
///             v3 = v2 + 1;  guard v3 > 6000 -> region B
///             guard v3 < 100000             -> exit
///             JUMP -> LABEL1 [v3]
///   region A  v11 = v10 + 5000;  JUMP -> LABEL0 [v11]
///   region B  v21 = v20 + 50000; JUMP -> LABEL0 [v21]
///
/// A fires once (v2 == 2), B once (v3 == 5004), and the loop then counts up to
/// its own exit, so the frame reports both regions' arithmetic.
#[test]
fn a_preamble_region_and_a_loop_body_region_each_reach_their_own_block() {
    assert_eq!(run_mixed_region_families_repro(), (100000, 99999, 55004));
}

fn run_mixed_region_families_repro() -> (i64, i64, i64) {
    let descr0 = majit_ir::make_loop_target_descr(60, false);
    let descr1 = majit_ir::make_loop_target_descr(61, false);

    let label0 = Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]);
    label0.setdescr(descr0.clone());
    let label1 = Op::new(OpCode::Label, &[rb(OpRef::int_op(2))]);
    label1.setdescr(descr1.clone());
    let jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(3))]);
    jump.setdescr(descr1);

    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        label0,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        make_op(
            OpCode::IntGt,
            &[OpRef::int_op(2), OpRef::const_int(10)],
            OpRef::int_op(6),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(6)], &[OpRef::int_op(2)]),
        label1,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntGt,
            &[OpRef::int_op(3), OpRef::const_int(6000)],
            OpRef::int_op(7),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(7)], &[OpRef::int_op(3)]),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), OpRef::const_int(100000)],
            OpRef::int_op(5),
        ),
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(5)],
            &[OpRef::int_op(3), OpRef::int_op(2), OpRef::int_op(1)],
        ),
        jump,
    ];
    assert_eq!(codegen::resumable_label_count(&ops), 2);
    assert!(codegen::source_guard_precedes_loop_label(&ops, 0));
    assert!(!codegen::source_guard_precedes_loop_label(&ops, 1));

    let region_a_jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(11))]);
    region_a_jump.setdescr(descr0.clone());
    let region_a = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(10), OpRef::const_int(5000)],
            OpRef::int_op(11),
        ),
        region_a_jump,
    ];
    let region_b_jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(21))]);
    region_b_jump.setdescr(descr0);
    let region_b = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(20), OpRef::const_int(50000)],
            OpRef::int_op(21),
        ),
        region_b_jump,
    ];

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let inputs = inline_region_inputs(
        &inputargs,
        ops,
        vec![
            codegen::InlinedBridge {
                external_jump: None,
                source_fail_index: 1,
                outside_loop: false,
                trace_id: 2,
                inputargs: vec![InputArg::from_type_rc(Type::Int, 20)],
                ops: region_b,
                gc_table_base: 0,
                gc_const_keys: Vec::new(),
                constants: indexmap::IndexMap::new(),
            },
            codegen::InlinedBridge {
                external_jump: None,
                source_fail_index: 0,
                outside_loop: true,
                trace_id: 1,
                inputargs: vec![InputArg::from_type_rc(Type::Int, 10)],
                ops: region_a,
                gc_table_base: 0,
                gc_const_keys: Vec::new(),
                constants: indexmap::IndexMap::new(),
            },
        ],
    );
    let (exit_index, slot0, slot1, slot2) = run_inline_region_trace(&inputs);
    assert_eq!(exit_index, 2, "the third guard is the one that exits");
    (slot0, slot1, slot2)
}

/// A preamble region resumes PAST its target LABEL's resume loader, which
/// restores that label's captures from slots the fall-through path writes as it
/// crosses the label — and a fresh entry clears them. A region whose guard
/// fails BEFORE the label it names has not written them, so the merge is
/// declined rather than resumed onto cleared slots.
#[test]
fn a_preamble_region_closing_at_a_label_past_its_guard_declines() {
    let descr0 = majit_ir::make_loop_target_descr(70, false);
    let descr1 = majit_ir::make_loop_target_descr(71, false);

    let label0 = Op::new(OpCode::Label, &[rb(OpRef::int_op(1))]);
    label0.setdescr(descr0.clone());
    let label1 = Op::new(OpCode::Label, &[rb(OpRef::int_op(2))]);
    label1.setdescr(descr1.clone());
    let jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(3))]);
    jump.setdescr(descr1.clone());

    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        label0,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        make_op(
            OpCode::IntGt,
            &[OpRef::int_op(2), OpRef::const_int(10)],
            OpRef::int_op(6),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(6)], &[OpRef::int_op(2)]),
        label1,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), OpRef::const_int(1000)],
            OpRef::int_op(5),
        ),
        make_guard(OpCode::GuardTrue, &[OpRef::int_op(5)], &[OpRef::int_op(3)]),
        jump,
    ];

    // The region names LABEL1, which sits after the guard it hangs off.
    let region_jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(11))]);
    region_jump.setdescr(descr1);
    assert!(!codegen::outside_region_labels_initialized(
        &ops,
        0,
        std::slice::from_ref(&region_jump)
    ));
    let earlier_jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(11))]);
    earlier_jump.setdescr(descr0);
    assert!(codegen::outside_region_labels_initialized(
        &ops,
        0,
        &[earlier_jump]
    ));
    let inputs = inline_region_inputs(
        &vec![InputArg::from_type_rc(Type::Int, 0)],
        ops,
        vec![codegen::InlinedBridge {
            external_jump: None,
            source_fail_index: 0,
            outside_loop: true,
            trace_id: 1,
            inputargs: vec![InputArg::from_type_rc(Type::Int, 10)],
            ops: vec![
                make_op(
                    OpCode::IntAdd,
                    &[OpRef::input_arg_int(10), OpRef::const_int(5000)],
                    OpRef::int_op(11),
                ),
                region_jump,
            ],
            gc_table_base: 0,
            gc_const_keys: Vec::new(),
            constants: indexmap::IndexMap::new(),
        }],
    );

    let error = match codegen::build_wasm_module(&inputs) {
        Ok(_) => panic!("the region's target LABEL is past its own guard"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("had not crossed"), "{error}");
}

/// A region closing at the loop HEADER must rebind the header LABEL's Ref arg
/// and refresh its home, exactly as the non-header back edge does.
///
/// `region_closing_at_a_non_header_label_carries_a_ref_label_arg` threads a Ref
/// that is already zero on entry, so it cannot tell a preserved Ref from a
/// zeroed one. This one carries a non-zero value into the loop and reads it back
/// out of the exit spill, which is what a dropped rebind actually looks like:
/// wasm offset 0 is valid linear memory, so the miscompile is silent.
#[test]
fn region_closing_at_the_header_preserves_a_ref_label_arg() {
    run_header_region_repro(true, RegionGuard::None);
}

/// A merged region that carries a GUARD of its own. `_jump_to_existing_trace`
/// prepends extra guards to a bridge that closes onto an existing loop, so a
/// region reaching the merge with guards is the ordinary case, not an exotic
/// one. Its guard must spill to its own exit, leaving the owner's loop-carried
/// locals — the Ref among them — untouched on the passing path.
#[test]
fn region_closing_at_the_header_carrying_its_own_guard() {
    run_header_region_repro(true, RegionGuard::Passes);
}

/// The same region whose own guard FAILS. A region reached through
/// `_jump_to_existing_trace` carries extra guards stamped with a
/// `ResumeAtPositionDescr` of the SOURCE trace, so the merged stream has to give
/// the region's guard its own exit index and spill the region's live values —
/// not the owner's.
#[test]
fn region_closing_at_the_header_exits_through_its_own_guard() {
    run_header_region_repro(true, RegionGuard::Fails);
}

/// The shape a real loop-closing bridge actually reaches the merge with: the
/// region's closing JUMP PERMUTES the header LABEL's args, two Refs whose homes
/// both have to be refreshed cross the back edge, and the region opens with a
/// guard of an inputarg against an inline `ConstPtr` — the extra guard
/// `_jump_to_existing_trace` prepends when a bridge closes onto an existing
/// loop.
#[test]
fn region_closing_at_the_header_permutes_two_ref_label_args() {
    const REF_A: i64 = 0x5EF0;
    const REF_B: i64 = 0x6EF0;
    let descr0 = majit_ir::make_loop_target_descr(40, false);
    let descr1 = majit_ir::make_loop_target_descr(41, false);

    let label0 = Op::new(
        OpCode::Label,
        &[
            rb(OpRef::int_op(1)),
            rb(OpRef::ref_op(6)),
            rb(OpRef::ref_op(7)),
        ],
    );
    label0.setdescr(descr0);
    let label1 = Op::new(
        OpCode::Label,
        &[
            rb(OpRef::int_op(2)),
            rb(OpRef::ref_op(6)),
            rb(OpRef::ref_op(7)),
        ],
    );
    label1.setdescr(descr1.clone());
    let jump = Op::new(
        OpCode::Jump,
        &[
            rb(OpRef::int_op(3)),
            rb(OpRef::ref_op(6)),
            rb(OpRef::ref_op(7)),
        ],
    );
    jump.setdescr(descr1.clone());

    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        label0,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        label1,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntGt,
            &[OpRef::int_op(3), OpRef::const_int(10)],
            OpRef::int_op(4),
        ),
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(4)],
            &[OpRef::int_op(3), OpRef::ref_op(6), OpRef::ref_op(7)],
        ),
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), OpRef::const_int(1000)],
            OpRef::int_op(5),
        ),
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(5)],
            &[OpRef::int_op(3), OpRef::ref_op(6), OpRef::ref_op(7)],
        ),
        jump,
    ];

    // The region swaps the two Ref label args on the way back.
    let region_jump = Op::new(
        OpCode::Jump,
        &[
            rb(OpRef::int_op(11)),
            rb(OpRef::ref_op(16)),
            rb(OpRef::ref_op(13)),
        ],
    );
    region_jump.setdescr(descr1);
    let region_ops = vec![
        make_op(
            OpCode::PtrEq,
            &[
                OpRef::ref_op(13),
                OpRef::const_ptr(majit_ir::GcRef(REF_A as usize)),
            ],
            OpRef::int_op(15),
        ),
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(15)],
            &[
                OpRef::input_arg_int(10),
                OpRef::ref_op(13),
                OpRef::ref_op(16),
            ],
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(10), OpRef::const_int(5000)],
            OpRef::int_op(11),
        ),
        region_jump,
    ];

    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Ref, 6),
        InputArg::from_type_rc(Type::Ref, 7),
    ];
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops,
        inlined_bridges: vec![codegen::InlinedBridge {
            external_jump: None,
            source_fail_index: 0,
            outside_loop: false,
            trace_id: 1,
            inputargs: vec![
                InputArg::from_type_rc(Type::Int, 10),
                InputArg::from_type_rc(Type::Ref, 13),
                InputArg::from_type_rc(Type::Ref, 16),
            ],
            ops: region_ops,
            gc_table_base: 0,
            gc_const_keys: Vec::new(),
            constants: indexmap::IndexMap::new(),
        }],
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    };
    let (bytes, _, _, _) = codegen::build_wasm_module(&inputs).expect("permuting region merges");
    validate_wasm(&bytes);

    let engine = Engine::default();
    let module = Module::new(&engine, &bytes).expect("generated trace should compile");
    let mut store = Store::new(&engine, ());
    let memory =
        Memory::new(&mut store, MemoryType::new(2, None)).expect("test memory should allocate");
    for (slot, value) in [(0u64, 0i64), (1, REF_A), (2, REF_B)] {
        memory
            .write(
                &mut store,
                (codegen::FRAME_SLOT_BASE + slot * 8) as usize,
                &value.to_le_bytes(),
            )
            .unwrap();
    }
    let mut linker = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("generated trace should instantiate");
    instance
        .get_typed_func::<i32, i32>(&store, "trace")
        .unwrap()
        .call(&mut store, 0)
        .expect("generated trace should execute");

    let read = |off: u64| {
        let mut buf = [0u8; 8];
        memory.read(&store, off as usize, &mut buf).unwrap();
        i64::from_le_bytes(buf)
    };
    assert_eq!(
        read(codegen::FRAME_SLOT_BASE),
        5004,
        "v3 after the re-entry"
    );
    assert_eq!(
        (
            read(codegen::FRAME_SLOT_BASE + 8),
            read(codegen::FRAME_SLOT_BASE + 16)
        ),
        (REF_B, REF_A),
        "the region's back edge must rebind both Ref label args, swapped"
    );
}

#[derive(Clone, Copy, PartialEq)]
enum RegionGuard {
    None,
    Passes,
    Fails,
}

/// A region whose closing JUMP carries FEWER args than the header LABEL cannot
/// rebind every loop-carried local. The back edge's parallel move is bounded by
/// `jump_args.len().min(label_args.len())`, so the unmatched label locals keep
/// whatever the failing iteration left in them. Refuse the merge instead.
#[test]
fn region_closing_at_the_header_with_a_short_jump_is_declined() {
    run_header_region_repro(false, RegionGuard::None);
}

fn run_header_region_repro(full_arity: bool, region_guard: RegionGuard) {
    const REF_IN: i64 = 0x5EF0;
    let descr0 = majit_ir::make_loop_target_descr(30, false);
    let descr1 = majit_ir::make_loop_target_descr(31, false);

    let label0 = Op::new(OpCode::Label, &[rb(OpRef::int_op(1)), rb(OpRef::ref_op(6))]);
    label0.setdescr(descr0);
    let label1 = Op::new(OpCode::Label, &[rb(OpRef::int_op(2)), rb(OpRef::ref_op(6))]);
    label1.setdescr(descr1.clone());
    let jump = Op::new(OpCode::Jump, &[rb(OpRef::int_op(3)), rb(OpRef::ref_op(6))]);
    jump.setdescr(descr1.clone());

    let guard_to_region = make_guard(
        OpCode::GuardTrue,
        &[OpRef::int_op(4)],
        &[OpRef::int_op(3), OpRef::ref_op(6)],
    );
    let guard_exit = make_guard(
        OpCode::GuardTrue,
        &[OpRef::int_op(5)],
        &[OpRef::int_op(3), OpRef::int_op(2), OpRef::ref_op(6)],
    );
    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(1),
        ),
        label0,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(1), OpRef::const_int(1)],
            OpRef::int_op(2),
        ),
        label1,
        make_op(
            OpCode::IntAdd,
            &[OpRef::int_op(2), OpRef::const_int(1)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntGt,
            &[OpRef::int_op(3), OpRef::const_int(10)],
            OpRef::int_op(4),
        ),
        guard_to_region,
        make_op(
            OpCode::IntLt,
            &[OpRef::int_op(3), OpRef::const_int(1000)],
            OpRef::int_op(5),
        ),
        guard_exit,
        jump,
    ];

    let region_jump_args = if full_arity {
        vec![rb(OpRef::int_op(11)), rb(OpRef::ref_op(12))]
    } else {
        vec![rb(OpRef::int_op(11))]
    };
    let region_jump = Op::new(OpCode::Jump, &region_jump_args);
    region_jump.setdescr(descr1);
    let mut region_ops = vec![make_op(
        OpCode::IntAdd,
        &[OpRef::input_arg_int(10), OpRef::const_int(5000)],
        OpRef::int_op(11),
    )];
    if region_guard != RegionGuard::None {
        // Passes on 5003 > 0; fails on 5003 > 100000.
        let threshold = if region_guard == RegionGuard::Passes {
            0
        } else {
            100_000
        };
        region_ops.push(make_op(
            OpCode::IntGt,
            &[OpRef::int_op(11), OpRef::const_int(threshold)],
            OpRef::int_op(14),
        ));
        region_ops.push(make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(14)],
            &[OpRef::int_op(11), OpRef::ref_op(13)],
        ));
    }
    region_ops.push(make_op(
        OpCode::SameAsR,
        &[OpRef::ref_op(13)],
        OpRef::ref_op(12),
    ));
    region_ops.push(region_jump);

    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Ref, 6),
    ];
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops,
        inlined_bridges: vec![codegen::InlinedBridge {
            external_jump: None,
            source_fail_index: 0,
            outside_loop: false,
            trace_id: 1,
            inputargs: vec![
                InputArg::from_type_rc(Type::Int, 10),
                InputArg::from_type_rc(Type::Ref, 13),
            ],
            ops: region_ops,
            gc_table_base: 0,
            gc_const_keys: Vec::new(),
            constants: indexmap::IndexMap::new(),
        }],
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    };
    let built = codegen::build_wasm_module(&inputs);
    if !full_arity {
        assert!(
            built.is_err(),
            "a region JUMP that cannot rebind every header LABEL arg must be refused, \
             not merged with a truncated parallel move"
        );
        return;
    }
    let (bytes, _, _, _) = built.expect("header region merges");
    validate_wasm(&bytes);

    let engine = Engine::default();
    let module = Module::new(&engine, &bytes).expect("generated trace should compile");
    let mut store = Store::new(&engine, ());
    let memory =
        Memory::new(&mut store, MemoryType::new(2, None)).expect("test memory should allocate");
    memory
        .write(
            &mut store,
            codegen::FRAME_SLOT_BASE as usize,
            &0i64.to_le_bytes(),
        )
        .unwrap();
    memory
        .write(
            &mut store,
            (codegen::FRAME_SLOT_BASE + 8) as usize,
            &REF_IN.to_le_bytes(),
        )
        .unwrap();
    let mut linker = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("generated trace should instantiate");
    instance
        .get_typed_func::<i32, i32>(&store, "trace")
        .unwrap()
        .call(&mut store, 0)
        .expect("generated trace should execute");

    let read = |off: u64| {
        let mut buf = [0u8; 8];
        memory.read(&store, off as usize, &mut buf).unwrap();
        i64::from_le_bytes(buf)
    };
    if region_guard == RegionGuard::Fails {
        assert_eq!(read(0), 2, "the region's own guard is the one that exits");
        assert_eq!(
            read(codegen::FRAME_SLOT_BASE),
            5003,
            "the region guard spills its own live int"
        );
        assert_eq!(
            read(codegen::FRAME_SLOT_BASE + 8),
            REF_IN,
            "the region guard spills the Ref it was entered with"
        );
        return;
    }
    assert_eq!(read(0), 1, "the second guard is the one that exits");
    assert_eq!(
        read(codegen::FRAME_SLOT_BASE),
        5004,
        "v3 after the re-entry"
    );
    assert_eq!(
        read(codegen::FRAME_SLOT_BASE + 8),
        5003,
        "v2 is the region's JUMP arg"
    );
    assert_eq!(
        read(codegen::FRAME_SLOT_BASE + 16),
        REF_IN,
        "the region's back edge must rebind the header LABEL's Ref arg"
    );
}

/// `compile.py make_a_counter_per_value` keys a GUARD_VALUE's jitcounter on the
/// pair (guard, failing value), and `regalloc.py consider_guard_value` hands it
/// `all_reg_indexes[x.value]` — a deadframe slot, not a position inside the
/// guard's fail arguments. The compared operand is a promoted value the resume
/// re-derives, so it is almost never one of those fail arguments; an exit that
/// can only name them leaves the stamp unwritten and the guard keeps the
/// per-guard hash, compiling a fresh bridge every `trace_eagerness` failures
/// for a value that never repeats.
///
/// So the exit publishes the operand itself, one slot past the last fail
/// argument, which is what `resolve_guard_value_operand` reads back through
/// `get_value_direct`.
#[test]
fn guard_value_parks_its_operand_past_every_exits_fail_args() {
    // `make_a_counter_per_value` (compile.py) keys the counter on
    // (guard, failing value), and `regalloc.py prepare_op_guard_value` names
    // the compared operand's deadframe slot. The operand here is not a fail
    // argument of its own guard, which is the ordinary shape for a promoted
    // value, so the exit has to park it somewhere.
    //
    // The wider GUARD_TRUE below is what pins WHERE. Parking at "one past MY
    // fail args" would put the word in slot 1, which is the GUARD_TRUE's
    // second fail argument — a slot another exit reads back as a fail value.
    // The slot is one per trace, past every exit's fail args.
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(7)],
            OpRef::int_op(1),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(100)],
            OpRef::int_op(2),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(5)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(4),
        ),
        // Two fail args, and it holds: this exit is never taken, it only makes
        // the trace's value area two slots wide.
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(4)],
            &[OpRef::int_op(2), OpRef::int_op(3)],
        ),
        make_guard(
            OpCode::GuardValue,
            &[OpRef::int_op(1), OpRef::const_int(999)],
            &[OpRef::int_op(2)],
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))]),
    ];
    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    assert_eq!(
        guards[0].counter_value_spill, None,
        "the GUARD_TRUE is not a GUARD_VALUE"
    );
    assert_eq!(
        guards[1].counter_value_spill,
        Some(OpRef::int_op(1)),
        "the GUARD_VALUE's own operand is not among its fail args"
    );
    validate_wasm(&bytes);

    let engine = Engine::default();
    let module = Module::new(&engine, &bytes).expect("generated trace should compile");
    let mut store = Store::new(&engine, ());
    let memory =
        Memory::new(&mut store, MemoryType::new(2, None)).expect("test memory should allocate");
    let mut linker = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("generated trace should instantiate");
    instance
        .get_typed_func::<i32, i32>(&store, "trace")
        .unwrap()
        .call(&mut store, 0)
        .expect("generated trace should execute");

    let read = |off: u64| {
        let mut buf = [0u8; 8];
        memory.read(&store, off as usize, &mut buf).unwrap();
        i64::from_le_bytes(buf)
    };
    assert_eq!(read(0), 1, "the GUARD_VALUE is the exit taken");
    assert_eq!(
        read(codegen::FRAME_SLOT_BASE),
        100,
        "slot 0 stays the guard's single fail argument"
    );
    assert_eq!(
        read(codegen::FRAME_SLOT_BASE + 8),
        0,
        "slot 1 belongs to the wider exit's fail args and this exit leaves it alone"
    );
    assert_eq!(
        read(codegen::FRAME_SLOT_BASE + 16),
        7,
        "the compared operand parks past every exit's fail args"
    );
}

#[test]
fn sparse_resume_positions_use_compact_exit_and_force_locations() {
    #[derive(Debug)]
    struct SparseDescr {
        locs: Vec<u16>,
        types: Vec<Type>,
        counter: std::sync::atomic::AtomicU32,
    }
    impl majit_ir::Descr for SparseDescr {
        fn as_fail_descr(&self) -> Option<&dyn majit_ir::FailDescr> {
            Some(self)
        }
    }
    impl majit_ir::FailDescr for SparseDescr {
        fn make_a_counter_per_value(&self, index: u32, _type_tag: u64) {
            self.counter
                .store(index, std::sync::atomic::Ordering::Relaxed);
        }
        fn fail_index(&self) -> u32 {
            0
        }
        fn rd_locs(&self) -> &[u16] {
            &self.locs
        }
        fn fail_arg_types(&self) -> &[Type] {
            &self.types
        }
    }

    // ResumeDataLoopMemo can leave a wide logical numbering with just two
    // live boxes. Neither normal exits nor force brackets may reserve the
    // intervening holes in the source token's physical frame.
    // A compared operand may be absent, present only in a logical hole, or
    // present both in an earlier hole and a later live position.
    for (force, counter_position) in [
        (false, None),
        (false, Some(3)),
        (false, Some(17)),
        (true, None),
    ] {
        let inputargs: Vec<_> = (0..3)
            .map(|i| InputArg::from_type_rc(Type::Int, i))
            .collect();
        let mut failargs = vec![OpRef::NONE; 100];
        failargs[17] = OpRef::input_arg_int(2);
        failargs[99] = OpRef::input_arg_int(1);
        if let Some(position) = counter_position {
            failargs[3] = OpRef::input_arg_int(0);
            failargs[position] = OpRef::input_arg_int(0);
        }
        let mut locs = vec![0xFFFF; 100];
        locs[17] = 17;
        locs[99] = 99;
        let guard = if force {
            make_guard(OpCode::GuardNotForced2, &[], &failargs)
        } else {
            make_guard(
                OpCode::GuardValue,
                &[OpRef::input_arg_int(0), OpRef::const_int(0)],
                &failargs,
            )
        };
        let descr = std::sync::Arc::new(SparseDescr {
            locs,
            types: vec![Type::Int; 100],
            counter: std::sync::atomic::AtomicU32::new(u32::MAX),
        });
        guard.setdescr(descr.clone());
        let ops = vec![
            guard,
            Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(1))]),
        ];
        assert_eq!(codegen::frame_value_slots(&inputargs, &ops), 5);
        let frame = codegen::FrameGeometry::compact(5, 0, 0);
        let (bytes, guards) = build_module_with_frame(
            &inputargs,
            &ops,
            &indexmap::IndexMap::new(),
            Some(0),
            &codegen::GuardGcTypeInfo::default(),
            frame,
        );
        assert_eq!(guards[0].fail_locs[17], Some(0));
        assert_eq!(guards[0].fail_locs[99], Some(1));
        assert!(guards[0].fail_locs[..17].iter().all(Option::is_none));
        let parked = !force && counter_position != Some(17);
        if !force {
            assert_eq!(
                descr.counter.load(std::sync::atomic::Ordering::Relaxed),
                if parked { 100 } else { 17 }
            );
        }
        if parked {
            assert_eq!(
                guards[0].fail_locs[100],
                Some(3),
                "counter coordinate remains logical"
            );
        }
        validate_wasm(&bytes);
        let engine = Engine::default();
        let module = Module::new(&engine, &bytes).unwrap();
        let mut store = Store::new(&engine, ());
        let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
        let mut linker = Linker::new(&engine);
        linker.define("env", "memory", memory).unwrap();
        let instance = linker.instantiate_and_start(&mut store, &module).unwrap();
        let base = 4096usize;
        for (i, value) in [5i64, 42, 7].iter().enumerate() {
            memory
                .write(&mut store, base + 8 + i * 8, &value.to_le_bytes())
                .unwrap();
        }
        instance
            .get_typed_func::<i32, i32>(&store, "trace")
            .unwrap()
            .call(&mut store, base as i32)
            .unwrap();
        let read = |offset: usize| {
            let mut bytes = [0u8; 8];
            memory.read(&store, base + offset, &mut bytes).unwrap();
            i64::from_le_bytes(bytes)
        };
        let offset = if force {
            frame.force_slot_base as usize
        } else {
            8
        };
        assert_eq!(
            read(offset),
            if counter_position == Some(17) { 5 } else { 7 }
        );
        assert_eq!(read(offset + 8), 42);
        if parked {
            assert_eq!(
                read(8 + 3 * 8),
                5,
                "counter operand survives compact spilling"
            );
        }
    }
}

/// A bridge executes in its source token's frame, whose offsets froze when that
/// token was compiled, and `compile_bridge` refuses a bridge whose
/// `frame_value_slots` exceeds `source_frame.value_slots` — a refusal
/// the guard descriptor makes permanent, so the guard blackholes for the
/// rest of the run.  A source loop that spills nothing must therefore already
/// carry the GUARD_VALUE counter slot: promoting a value is the ordinary way a
/// bridge acquires one, and the two traces below have the same value area, so a
/// conditional reservation would make the second one slot wider than the first.
#[test]
fn a_counterless_trace_reserves_the_slot_a_bridge_may_need() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    // Shared prefix: three values, a two-fail-arg exit, and a FINISH.
    let common = vec![
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(100)],
            OpRef::int_op(2),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(5)],
            OpRef::int_op(3),
        ),
        make_op(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(0), OpRef::const_int(1)],
            OpRef::int_op(4),
        ),
        make_guard(
            OpCode::GuardTrue,
            &[OpRef::int_op(4)],
            &[OpRef::int_op(2), OpRef::int_op(3)],
        ),
    ];

    let mut counterless = common.clone();
    counterless.push(Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))]));

    // The bridge shape: one more GUARD_VALUE whose compared operand is not
    // among its own fail args, so it needs the parked slot. Its single fail
    // argument keeps the value area at the same two slots.
    let mut with_counter = common;
    with_counter.push(make_op(
        OpCode::IntAdd,
        &[OpRef::input_arg_int(0), OpRef::const_int(7)],
        OpRef::int_op(1),
    ));
    with_counter.push(make_guard(
        OpCode::GuardValue,
        &[OpRef::int_op(1), OpRef::const_int(999)],
        &[OpRef::int_op(2)],
    ));
    with_counter.push(Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))]));

    let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();
    let (_, counterless_guards) = build_module_default(&inputargs, &counterless, &constants);
    let (_, counter_guards) = build_module_default(&inputargs, &with_counter, &constants);
    assert!(
        counterless_guards
            .iter()
            .all(|g| g.counter_value_spill.is_none()),
        "control: the first trace spills nothing, so a conditional reservation \
         would leave its frame one slot narrower"
    );
    assert!(
        counter_guards
            .iter()
            .any(|g| g.counter_value_spill.is_some()),
        "the second trace is the bridge shape that needs the parked slot"
    );

    assert_eq!(
        codegen::frame_value_slots(&inputargs, &counterless),
        codegen::frame_value_slots(&inputargs, &with_counter),
        "a source loop that spills nothing must still fit a bridge that does"
    );
}

fn count_i32_store8(bytes: &[u8]) -> u32 {
    let mut n = 0;
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                if matches!(
                    operators.read().unwrap(),
                    wasmparser::Operator::I32Store8 { .. }
                ) {
                    n += 1;
                }
            }
        }
    }
    n
}

const WB_FIELD: i64 = 0x4a11;
const WB_ARRAY: i64 = 0x4a22;

fn build_module_with_barrier_helpers(
    inputargs: &[InputArgRc],
    ops: &[Op],
    wb: codegen::WriteBarrierHelpers,
) -> Vec<u8> {
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.to_vec(),
        ops: ops.to_vec(),
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb,
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::compact(6, 3, 0),
        ca: codegen::CaParams::default(),
    };
    let (bytes, _, _, _) = codegen::build_wasm_module(&rewrite_module_inputs(inputs))
        .expect("write barrier module compiles");
    validate_wasm(&bytes);
    bytes
}

fn wb_helpers_without_cards() -> codegen::WriteBarrierHelpers {
    let mut descr = majit_gc::WriteBarrierDescr::for_current_gc();
    descr.jit_wb_cards_set = 0;
    descr.jit_wb_cards_set_byteofs = 0;
    descr.jit_wb_cards_set_singlebyte = 0;
    descr.jit_wb_card_page_shift = 0;
    codegen::WriteBarrierHelpers::new(WB_FIELD, WB_ARRAY, &descr)
}

fn ref_array_store(
    descr: &std::sync::Arc<majit_ir::descr::SimpleArrayDescr>,
    idx: OpRef,
    val: OpRef,
) -> Op {
    let op = make_op(
        OpCode::SetarrayitemGc,
        &[OpRef::input_arg_ref(0), idx, val],
        OpRef::NONE,
    );
    op.setdescr(descr.clone());
    op
}

fn two_ref_array_stores() -> (Vec<InputArgRc>, Vec<Op>) {
    use majit_ir::descr::SimpleArrayDescr;
    use std::sync::Arc;
    let descr = Arc::new(SimpleArrayDescr::new(1, 16, 8, 55, Type::Ref));
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
        InputArg::from_type_rc(Type::Ref, 2),
        InputArg::from_type_rc(Type::Ref, 3),
    ];
    let ops = vec![
        ref_array_store(&descr, OpRef::input_arg_int(1), OpRef::input_arg_ref(2)),
        ref_array_store(&descr, OpRef::const_int(5), OpRef::input_arg_ref(3)),
        Op::new(OpCode::Finish, &[]),
    ];
    (inputargs, ops)
}

/// `_write_barrier_fastpath(..., array=True)` / `WriteBarrierSlowPath`: a
/// SETARRAYITEM_GC of a Ref marks a card byte, and `gen_write_barrier_array`
/// remembers nothing, so the second store to the same base barriers again.
#[test]
fn array_write_barrier_marks_a_card_when_the_collector_has_them() {
    let (inputargs, ops) = two_ref_array_stores();
    let bytes = build_module_with_barrier_helpers(
        &inputargs,
        &ops,
        codegen::WriteBarrierHelpers::for_current_gc(WB_FIELD, WB_ARRAY),
    );
    assert!(
        count_i32_store8(&bytes) >= 2,
        "each array store marks its own card byte"
    );
    assert_eq!(
        direct_write_barrier_call_count(&bytes, WB_ARRAY as i32),
        2,
        "the CARDS_SET-clear arm calls jit_remember_young_pointer_from_array"
    );
    assert_eq!(
        direct_write_barrier_call_count(&bytes, WB_FIELD as i32),
        0,
        "array stores must not use the field write_barrier helper"
    );
}

/// `gen_write_barrier_array` falls through to `gen_write_barrier` when
/// `jit_wb_cards_set` is 0, and that arm *is* remembered.
#[test]
fn array_write_barrier_without_cards_is_the_plain_remembered_barrier() {
    let (inputargs, ops) = two_ref_array_stores();
    let bytes = build_module_with_barrier_helpers(&inputargs, &ops, wb_helpers_without_cards());
    assert_eq!(
        count_i32_store8(&bytes),
        0,
        "a collector with no cards has no card byte to mark"
    );
    assert_eq!(
        direct_write_barrier_call_count(&bytes, WB_ARRAY as i32),
        0,
        "jit_wb_cards_set == 0 never reaches the array helper"
    );
    assert_eq!(
        direct_write_barrier_call_count(&bytes, WB_FIELD as i32),
        1,
        "the fall-back arm is a plain write_barrier, so the second store elides"
    );
}

/// `handle_write_barrier_setarrayitem` opens with the same
/// `write_barrier_applied(val)` test `handle_write_barrier_setfield` does: a
/// base already remembered by a field barrier takes no array barrier at all.
#[test]
fn a_remembered_base_suppresses_a_later_array_barrier() {
    use majit_ir::descr::{SimpleArrayDescr, SimpleFieldDescr};
    use std::sync::Arc;

    let array = Arc::new(SimpleArrayDescr::new(1, 16, 8, 55, Type::Ref));
    let field = Op::new(
        OpCode::SetfieldGc,
        &[rb(OpRef::input_arg_ref(0)), rb(OpRef::input_arg_ref(2))],
    );
    field.setdescr(Arc::new(SimpleFieldDescr::new(0, 0, 8, Type::Ref, false)));
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
        InputArg::from_type_rc(Type::Ref, 2),
        InputArg::from_type_rc(Type::Ref, 3),
    ];
    let ops = vec![
        field,
        ref_array_store(&array, OpRef::input_arg_int(1), OpRef::input_arg_ref(3)),
        Op::new(OpCode::Finish, &[]),
    ];
    let bytes = build_module_with_barrier_helpers(
        &inputargs,
        &ops,
        codegen::WriteBarrierHelpers::for_current_gc(WB_FIELD, WB_ARRAY),
    );
    assert_eq!(
        direct_write_barrier_call_count(&bytes, WB_FIELD as i32),
        1,
        "the field store barriers once"
    );
    assert_eq!(
        direct_write_barrier_call_count(&bytes, WB_ARRAY as i32),
        0,
        "the array store finds the base already in write_barrier_applied"
    );
    assert_eq!(
        count_i32_store8(&bytes),
        0,
        "a suppressed barrier marks no card either"
    );
}

/// `gen_write_barrier_array`'s `known_length(v_base, LARGE)`: a NEW_ARRAY this
/// trace gave a statically short length takes the plain remembered barrier, so
/// the second store to it elides; one at the threshold keeps the card arm.
#[test]
fn a_statically_short_array_takes_the_plain_barrier() {
    use majit_ir::descr::SimpleArrayDescr;
    use std::sync::Arc;

    let descr = Arc::new(SimpleArrayDescr::new(1, 16, 8, 55, Type::Ref));
    let build = |length: i64| {
        let new = make_op(
            OpCode::NewArray,
            &[OpRef::const_int(length)],
            OpRef::ref_op(4),
        );
        new.setdescr(descr.clone());
        let store = |val: OpRef| {
            let op = make_op(
                OpCode::SetarrayitemGc,
                &[OpRef::ref_op(4), OpRef::input_arg_int(1), val],
                OpRef::NONE,
            );
            op.setdescr(descr.clone());
            op
        };
        let inputargs = vec![
            InputArg::from_type_rc(Type::Ref, 0),
            InputArg::from_type_rc(Type::Int, 1),
            InputArg::from_type_rc(Type::Ref, 2),
            InputArg::from_type_rc(Type::Ref, 3),
        ];
        let ops = vec![
            new,
            store(OpRef::input_arg_ref(2)),
            store(OpRef::input_arg_ref(3)),
            Op::new(OpCode::Finish, &[]),
        ];
        build_module_with_barrier_helpers(
            &inputargs,
            &ops,
            codegen::WriteBarrierHelpers::for_current_gc(WB_FIELD, WB_ARRAY),
        )
    };

    let short = build(129);
    assert_eq!(
        direct_write_barrier_call_count(&short, WB_ARRAY as i32),
        0,
        "a length under LARGE never reaches the array helper"
    );
    assert_eq!(
        direct_write_barrier_call_count(&short, WB_FIELD as i32),
        1,
        "the plain arm is remembered, so the second store elides"
    );

    let long = build(130);
    assert_eq!(
        direct_write_barrier_call_count(&long, WB_ARRAY as i32),
        2,
        "at LARGE the card arm takes over, and it remembers nothing"
    );
    assert_eq!(
        direct_write_barrier_call_count(&long, WB_FIELD as i32),
        0,
        "the card arm never calls the field helper"
    );
}

/// `(i64.mul, i64.shl)` counts. `load_supported_factors: &[1]` pre-scales
/// in IR as `IntMul` / `IntLshift`, which wasm lowers as i64 ops.
fn count_i64_scale_ops(bytes: &[u8]) -> (u32, u32) {
    let mut mul = 0;
    let mut shl = 0;
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                match operators.read().unwrap() {
                    wasmparser::Operator::I64Mul => mul += 1,
                    wasmparser::Operator::I64Shl => shl += 1,
                    _ => {}
                }
            }
        }
    }
    (mul, shl)
}

fn array_get_bytes(item_size: usize, index: OpRef) -> Vec<u8> {
    use majit_ir::descr::SimpleArrayDescr;
    use std::sync::Arc;

    let descr = Arc::new(SimpleArrayDescr::new(1, 16, item_size, 55, Type::Int));
    let get = make_op(
        OpCode::GetarrayitemGcI,
        &[OpRef::input_arg_ref(0), index],
        OpRef::int_op(2),
    );
    get.setdescr(descr);
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let ops = vec![get, Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))])];
    let (ops, constants) = rewrite_frontend_ops(&inputargs, ops);
    let (bytes, _) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    bytes
}

/// `load_supported_factors: &[1]`: the rewriter pre-multiplies a factor
/// other than 1 (`rewrite.rs` `cpu_simplify_scale`). Power-of-two factors
/// become `IntLshift`; others `IntMul`. Codegen of `GcLoadIndexed` with
/// factor 1 then has no extra scale.
#[test]
fn array_addr_uses_get_scale_instead_of_multiply() {
    let (mul1, shl1) = count_i64_scale_ops(&array_get_bytes(1, OpRef::input_arg_int(1)));
    assert_eq!(
        mul1, 0,
        "item_size == 1 skips the scale, like x86 getarrayitem"
    );
    assert_eq!(shl1, 0, "item_size == 1 is get_scale 0, not a shift");

    let (mul4, shl4) = count_i64_scale_ops(&array_get_bytes(4, OpRef::input_arg_int(1)));
    assert_eq!(mul4, 0, "item_size == 4 is pre-scaled with IntLshift");
    assert_eq!(shl4, 1, "item_size == 4 is an i64.shl of the index");

    let (mul8, shl8) = count_i64_scale_ops(&array_get_bytes(8, OpRef::input_arg_int(1)));
    assert_eq!(mul8, 0, "item_size == 8 is pre-scaled with IntLshift");
    assert_eq!(shl8, 1, "item_size == 8 is an i64.shl of the index");

    // item_size 3: rewrite.rs pre-scales with IntMul and keeps GcLoadIndexed
    // itemsize=3. wasm codegen has no 3-byte load (the deleted GetarrayitemGcI
    // arm did).
    {
        use majit_ir::descr::SimpleArrayDescr;
        use std::sync::Arc;
        let descr = Arc::new(SimpleArrayDescr::new(1, 16, 3, 55, Type::Int));
        let get = make_op(
            OpCode::GetarrayitemGcI,
            &[OpRef::input_arg_ref(0), OpRef::input_arg_int(1)],
            OpRef::int_op(2),
        );
        get.setdescr(descr);
        let inputargs = vec![
            InputArg::from_type_rc(Type::Ref, 0),
            InputArg::from_type_rc(Type::Int, 1),
        ];
        let ops = vec![get, Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))])];
        let (ops, constants) = rewrite_frontend_ops(&inputargs, ops);
        let inputs = inline_region_inputs(&inputargs, ops, vec![]);
        let mut inputs = inputs;
        inputs.constants = constants;
        match codegen::build_wasm_module(&inputs) {
            Err(majit_backend::BackendError::Unsupported(msg)) => {
                assert!(
                    msg.contains("item size 3"),
                    "GcLoadIndexedI of size 3 must be declined: {msg}"
                );
            }
            Ok(_) => panic!("item_size 3 should be Unsupported, got Ok"),
            Err(err) => panic!("item_size 3 should be Unsupported, got {err}"),
        }
    }

    let (mul_c, shl_c) = count_i64_scale_ops(&array_get_bytes(8, OpRef::const_int(5)));
    assert_eq!(
        (mul_c, shl_c),
        (0, 0),
        "a ConstInt index folds into the MemArg offset, rewrite.py GC_LOAD arm"
    );
}

/// `descr.py unpack_interiorfielddescr` / `rewrite.py transform_to_gc_load`:
/// GET/SETINTERIORFIELD must compile, not decline the trace.
#[test]
fn interior_field_ops_compile() {
    use majit_ir::descr::{
        ArrayDescr, FieldDescr, SimpleArrayDescr, SimpleFieldDescr, SimpleInteriorFieldDescr,
    };
    use std::sync::Arc;

    let interior = |item_size: usize| {
        let array: Arc<dyn ArrayDescr> =
            Arc::new(SimpleArrayDescr::new(1, 8, item_size, 55, Type::Int));
        let field: Arc<dyn FieldDescr> = Arc::new(SimpleFieldDescr::new(0, 4, 4, Type::Int, false));
        Arc::new(SimpleInteriorFieldDescr::new(0, array, field))
    };

    let get = make_op(
        OpCode::GetinteriorfieldGcI,
        &[OpRef::input_arg_ref(0), OpRef::input_arg_int(1)],
        OpRef::int_op(3),
    );
    get.setdescr(interior(16));
    let set = make_op(
        OpCode::SetinteriorfieldGc,
        &[
            OpRef::input_arg_ref(0),
            OpRef::input_arg_int(1),
            OpRef::input_arg_int(2),
        ],
        OpRef::NONE,
    );
    set.setdescr(interior(16));
    let raw = make_op(
        OpCode::SetinteriorfieldRaw,
        &[
            OpRef::input_arg_ref(0),
            OpRef::input_arg_int(1),
            OpRef::input_arg_int(2),
        ],
        OpRef::NONE,
    );
    raw.setdescr(interior(8));

    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
        InputArg::from_type_rc(Type::Int, 2),
    ];
    let ops = vec![
        set,
        raw,
        get,
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(3))]),
    ];
    let (ops, constants) = rewrite_frontend_ops(&inputargs, ops);
    let (bytes, guards) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    assert_eq!(guards.len(), 1);
    let (mul, shl) = count_i64_scale_ops(&bytes);
    // load_supported_factors: &[1] pre-multiplies power-of-two factors as
    // IntLshift: item_size 16 (set+get) and 8 (raw store).
    assert_eq!(mul, 0, "16 and 8 are shifts, not IMUL");
    assert_eq!(shl, 3, "two size-16 interior ops and one size-8 RAW store");
}

/// A ref-typed SETINTERIORFIELD_GC that the GC rewrite did not lower is
/// declined. Codegen must not store the pointer itself.
#[test]
#[should_panic(
    expected = "wasm codegen: SetinteriorfieldGc must have been lowered by rewrite_ops_for_gc"
)]
fn setinteriorfield_gc_ref_without_rewrite_is_rejected() {
    use majit_ir::descr::{
        ArrayDescr, FieldDescr, SimpleArrayDescr, SimpleFieldDescr, SimpleInteriorFieldDescr,
    };
    use std::sync::Arc;

    let array: Arc<dyn ArrayDescr> = Arc::new(SimpleArrayDescr::new(1, 8, 16, 55, Type::Ref));
    let field: Arc<dyn FieldDescr> = Arc::new(SimpleFieldDescr::new(0, 0, 4, Type::Ref, false));
    let interior = Arc::new(SimpleInteriorFieldDescr::new(0, array, field));
    let set = make_op(
        OpCode::SetinteriorfieldGc,
        &[
            OpRef::input_arg_ref(0),
            OpRef::input_arg_int(1),
            OpRef::input_arg_ref(2),
        ],
        OpRef::NONE,
    );
    set.setdescr(interior);
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
        InputArg::from_type_rc(Type::Ref, 2),
    ];
    let ops = vec![set, Op::new(OpCode::Finish, &[])];
    // Same module inputs as `build_module_default`, including its rewrite
    // gate. `SetinteriorfieldGc` is not in that gate, so the op reaches
    // codegen unlowered.
    let inputs = rewrite_module_inputs(inline_region_inputs(&inputargs, ops, vec![]));
    match codegen::build_wasm_module(&inputs) {
        Err(majit_backend::BackendError::Unsupported(msg)) => {
            assert!(
                msg.contains("SetinteriorfieldGc reached codegen without the GC rewrite"),
                "ref SETINTERIORFIELD_GC must be declined: {msg}"
            );
        }
        Ok(_) => panic!("ref SETINTERIORFIELD_GC without rewrite should be Unsupported, got Ok"),
        Err(err) => {
            panic!("ref SETINTERIORFIELD_GC without rewrite should be Unsupported, got {err}")
        }
    }
}

/// Count the operators a test cares about in the one code section entry.
fn count_operators(bytes: &[u8], mut f: impl FnMut(&wasmparser::Operator<'_>)) {
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                f(&operators.read().unwrap());
            }
        }
    }
}

/// x86/regalloc.py `consider_cond_call`, COND_CALL_VALUE arm: the call happens
/// when args[0] is 0/NULL, args[1] is the callee, and the result is args[0]
/// otherwise. Sharing the generic CALL arm called unconditionally through
/// args[0].
#[test]
fn cond_call_value_tests_its_value_and_calls_arg_one() {
    use majit_ir::descr::SimpleCallDescr;
    use std::sync::Arc;

    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let call = make_op(
        OpCode::CondCallValueI,
        &[
            OpRef::input_arg_int(0),
            OpRef::const_int(0x100),
            OpRef::input_arg_int(1),
        ],
        OpRef::int_op(2),
    );
    call.setdescr(Arc::new(SimpleCallDescr::new(
        0,
        vec![Type::Int],
        Type::Int,
        false,
        8,
        EffectInfo::default(),
    )));
    let ops = vec![call, Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))])];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    let (mut eqz, mut ifs) = (0, 0);
    count_operators(&bytes, |op| match op {
        wasmparser::Operator::I64Eqz => eqz += 1,
        wasmparser::Operator::If { .. } => ifs += 1,
        _ => {}
    });
    assert!(eqz >= 1, "COND_CALL_VALUE tests its value with i64.eqz");
    assert_eq!(ifs, 1, "the call sits under exactly one predicate");
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
}

/// x86/assembler.py `genop_guard_guard_nonnull_class` guards the class compare
/// with `CMP(ptr, 1)`; sharing the GUARD_CLASS arm let a NULL receiver be
/// decided by a read of linear-memory address 0, which on wasm does not trap.
#[test]
fn guard_nonnull_class_also_tests_for_null() {
    let build = |opcode: OpCode| {
        let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
        let ops = vec![
            make_guard(
                opcode,
                &[OpRef::input_arg_ref(0), OpRef::const_int(0x55)],
                &[OpRef::input_arg_ref(0)],
            ),
            Op::new(OpCode::Finish, &[rb(OpRef::input_arg_ref(0))]),
        ];
        let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
        validate_wasm(&bytes);
        let (mut eqz, mut ors) = (0, 0);
        count_operators(&bytes, |op| match op {
            wasmparser::Operator::I64Eqz => eqz += 1,
            wasmparser::Operator::I32Or => ors += 1,
            _ => {}
        });
        (eqz, ors)
    };
    assert_eq!(
        build(OpCode::GuardClass),
        (0, 0),
        "GUARD_CLASS is the class compare alone"
    );
    assert_eq!(
        build(OpCode::GuardNonnullClass),
        (1, 1),
        "GUARD_NONNULL_CLASS ors in a null test"
    );
}

/// pyjitpl.py `direct_call_release_gil` records `[savebox, funcbox] + args`, so
/// the callee is arg 1. The shared CALL arm read arg 0 — the `save_err`
/// immediate — as the callee and passed the real function as the first
/// argument.
#[test]
fn call_release_gil_takes_its_callee_from_arg_one() {
    use majit_ir::descr::SimpleCallDescr;
    use std::sync::Arc;

    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let call = make_op(
        OpCode::CallReleaseGilI,
        &[
            OpRef::const_int(0),     // save_err: no errno protocol
            OpRef::const_int(0x22),  // the real callee
            OpRef::input_arg_int(0), // its one argument
        ],
        OpRef::int_op(1),
    );
    call.setdescr(Arc::new(SimpleCallDescr::new(
        0x5100_0000,
        vec![Type::Int],
        Type::Int,
        true,
        8,
        EffectInfo::default(),
    )));
    let ops = vec![call, Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))])];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    let mut callee_consts = Vec::new();
    let mut call_indirects = 0;
    count_operators(&bytes, |op| match op {
        wasmparser::Operator::I32Const { value } => callee_consts.push(i64::from(*value)),
        wasmparser::Operator::CallIndirect { .. } => call_indirects += 1,
        _ => {}
    });
    assert!(
        callee_consts.contains(&0x22),
        "the callee table index is arg 1, not the save_err immediate"
    );
    assert!(
        call_indirects >= 1,
        "a uniform-word CALL_RELEASE_GIL must use the direct residual family"
    );
}

/// `raw_load_f` is a translated operation (`insns.rs` BC_RAW_LOAD_F), and the
/// int form was lowered while the float form declined the trace.
#[test]
fn raw_load_f_loads_a_double() {
    use majit_ir::descr::SimpleArrayDescr;
    use std::sync::Arc;

    let descr = Arc::new(SimpleArrayDescr::new(1, 0, 8, 55, Type::Float));
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let load = make_op(
        OpCode::RawLoadF,
        &[OpRef::input_arg_ref(0), OpRef::input_arg_int(1)],
        OpRef::float_op(2),
    );
    load.setdescr(descr);
    let ops = vec![load, Op::new(OpCode::Finish, &[rb(OpRef::float_op(2))])];
    let (ops, constants) = rewrite_frontend_ops(&inputargs, ops);
    let (bytes, _) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    let mut f64_loads = 0;
    count_operators(&bytes, |op| {
        if matches!(op, wasmparser::Operator::F64Load { .. }) {
            f64_loads += 1;
        }
    });
    assert_eq!(f64_loads, 1, "the float raw load is an f64.load");
}

#[test]
fn raw_load_const_offset_uses_memarg_displacement() {
    use majit_ir::descr::SimpleArrayDescr;
    use std::sync::Arc;

    const OFFSET: u64 = 24;
    let descr = Arc::new(SimpleArrayDescr::new(1, 0, 8, 55, Type::Int));
    let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
    let load = make_op(
        OpCode::RawLoadI,
        &[OpRef::input_arg_ref(0), OpRef::const_int(OFFSET as i64)],
        OpRef::int_op(1),
    );
    load.setdescr(descr);
    let ops = vec![load, Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))])];
    let (ops, constants) = rewrite_frontend_ops(&inputargs, ops);
    let (bytes, _) = build_module_default(&inputargs, &ops, &constants);
    validate_wasm(&bytes);
    let mut adds = 0;
    let mut displaced = 0;
    count_operators(&bytes, |op| match op {
        wasmparser::Operator::I32Add => adds += 1,
        wasmparser::Operator::I64Load { memarg } | wasmparser::Operator::I64Load32U { memarg }
            if memarg.offset == OFFSET =>
        {
            displaced += 1;
        }
        _ => {}
    });
    assert_eq!(displaced, 1, "the constant offset rides in MemArg");
    assert_eq!(
        adds, 0,
        "a nonnegative constant offset must not emit i32.add"
    );
}

/// resoperation.py `int_between(a, b, c)` is `a <= b < c`, signed.
#[test]
fn int_between_is_two_signed_comparisons() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
        InputArg::from_type_rc(Type::Int, 2),
    ];
    let ops = vec![
        make_op(
            OpCode::IntBetween,
            &[
                OpRef::input_arg_int(0),
                OpRef::input_arg_int(1),
                OpRef::input_arg_int(2),
            ],
            OpRef::int_op(3),
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(3))]),
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    let (mut le_s, mut lt_s, mut ands) = (0, 0, 0);
    count_operators(&bytes, |op| match op {
        wasmparser::Operator::I64LeS => le_s += 1,
        wasmparser::Operator::I64LtS => lt_s += 1,
        wasmparser::Operator::I32And => ands += 1,
        _ => {}
    });
    assert_eq!((le_s, lt_s, ands), (1, 1, 1), "a <= b, b < c, and");
}

#[test]
fn float_mod_uses_a_guest_side_typed_helper() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Float, 0),
        InputArg::from_type_rc(Type::Float, 1),
    ];
    let ops = vec![
        make_op(
            OpCode::FloatMod,
            &[OpRef::input_arg_float(0), OpRef::input_arg_float(1)],
            OpRef::float_op(2),
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::float_op(2))]),
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);

    assert!(has_table_import(&bytes));
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
    let (calls, _) = indirect_call_types_and_drop_count(&bytes);
    assert_eq!(calls.len(), 1);
    assert_eq!(
        function_type(&bytes, calls[0].0 as usize),
        (
            vec![wasmparser::ValType::F64, wasmparser::ValType::F64],
            vec![wasmparser::ValType::F64],
        )
    );
}

#[test]
fn check_memory_error_keeps_the_non_null_path_compiled() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let ops = vec![
        make_op(
            OpCode::CheckMemoryError,
            &[OpRef::input_arg_int(0)],
            OpRef::NONE,
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    let mut null_tests = 0;
    count_operators(&bytes, |op| {
        if matches!(op, wasmparser::Operator::I64Eqz) {
            null_tests += 1;
        }
    });
    assert_eq!(null_tests, 1);
}

#[test]
fn same_as_and_load_effective_address_do_not_decline() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let ops = vec![
        make_op(
            OpCode::SameAsR,
            &[OpRef::input_arg_ref(0)],
            OpRef::ref_op(2),
        ),
        make_op(
            OpCode::LoadEffectiveAddress,
            &[
                OpRef::ref_op(2),
                OpRef::input_arg_int(1),
                OpRef::const_int(12),
                OpRef::const_int(3),
            ],
            OpRef::int_op(3),
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(3))]),
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    assert_eq!(
        execute_simple_trace(&bytes, &[4096, 5]),
        4096 + (5 << 3) + 12
    );
}

fn execute_simple_trace(bytes: &[u8], inputs: &[i64]) -> i64 {
    execute_trace_at_key(
        bytes,
        inputs,
        codegen::FrameGeometry::fixed().dispatch_key_ofs,
        0,
    )
}

fn execute_trace_at_key(bytes: &[u8], inputs: &[i64], key_ofs: u64, dispatch_key: u32) -> i64 {
    let engine = Engine::default();
    let module = Module::new(&engine, bytes).expect("generated trace should compile");
    let mut store = Store::new(&engine, ());
    let memory =
        Memory::new(&mut store, MemoryType::new(1, None)).expect("test memory should allocate");
    for (index, value) in inputs.iter().enumerate() {
        memory
            .write(
                &mut store,
                codegen::FRAME_SLOT_BASE as usize + index * 8,
                &value.to_le_bytes(),
            )
            .unwrap();
    }
    memory
        .write(
            &mut store,
            key_ofs as usize,
            &i64::from(dispatch_key).to_le_bytes(),
        )
        .unwrap();
    let mut linker = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("generated trace should instantiate");
    instance
        .get_typed_func::<i32, i32>(&store, "trace")
        .unwrap()
        .call(&mut store, 0)
        .expect("generated trace should execute");
    let mut result = [0; 8];
    memory
        .read(&store, codegen::FRAME_SLOT_BASE as usize, &mut result)
        .unwrap();
    i64::from_le_bytes(result)
}

#[test]
fn int_signext_accepts_a_runtime_width() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let ops = vec![
        make_op(
            OpCode::IntSignext,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            OpRef::int_op(2),
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))]),
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);

    for (value, width, expected) in [
        (0xff, 1, -1),
        (0x7fff, 2, 0x7fff),
        (0x8000_0000, 4, -2_147_483_648),
        (0x80, 9, 0x80),
    ] {
        assert_eq!(
            execute_simple_trace(&bytes, &[value, width]),
            expected,
            "runtime-width int_signext({value:#x}, {width})"
        );
    }
}

#[test]
fn singlefloat_casts_round_trip_through_wasm_f32() {
    let inputargs = vec![InputArg::from_type_rc(Type::Float, 0)];
    let ops = vec![
        make_op(
            OpCode::CastFloatToSinglefloat,
            &[OpRef::input_arg_float(0)],
            OpRef::int_op(1),
        ),
        make_op(
            OpCode::CastSinglefloatToFloat,
            &[OpRef::int_op(1)],
            OpRef::float_op(2),
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::float_op(2))]),
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);

    let input = 1.234_567_890_123_f64;
    let result = f64::from_bits(execute_simple_trace(&bytes, &[input.to_bits() as i64]) as u64);
    assert_eq!(result, (input as f32) as f64);

    let mut casts = (0, 0, 0, 0);
    count_operators(&bytes, |op| match op {
        wasmparser::Operator::F32DemoteF64 => casts.0 += 1,
        wasmparser::Operator::I32ReinterpretF32 => casts.1 += 1,
        wasmparser::Operator::F32ReinterpretI32 => casts.2 += 1,
        wasmparser::Operator::F64PromoteF32 => casts.3 += 1,
        _ => {}
    });
    assert_eq!(casts, (1, 1, 1, 1));
}

#[test]
fn gc_rewrite_bare_load_store_execute_with_dynamic_offset() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
        InputArg::from_type_rc(Type::Int, 2),
    ];
    let ops = vec![
        make_op(
            OpCode::GcStore,
            &[
                OpRef::input_arg_ref(0),
                OpRef::input_arg_int(1),
                OpRef::input_arg_int(2),
                OpRef::const_int(4),
            ],
            OpRef::NONE,
        ),
        make_op(
            OpCode::GcLoadI,
            &[
                OpRef::input_arg_ref(0),
                OpRef::input_arg_int(1),
                OpRef::const_int(-4),
            ],
            OpRef::int_op(3),
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(3))]),
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    assert_eq!(execute_simple_trace(&bytes, &[4096, 12, -123]), -123);
}

#[test]
fn gc_store_halfword_at_negative_offset_uses_store16() {
    // GcStore offset -6, width 2. Negative offsets cannot use MemArg, so
    // the address is folded; width 2 is i64.store16.
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let ops = vec![
        make_op(
            OpCode::GcStore,
            &[
                OpRef::input_arg_ref(0),
                OpRef::const_int(-6),
                OpRef::input_arg_int(1),
                OpRef::const_int(2),
            ],
            OpRef::NONE,
        ),
        make_op(
            OpCode::GcLoadI,
            &[
                OpRef::input_arg_ref(0),
                OpRef::const_int(-6),
                OpRef::const_int(-2),
            ],
            OpRef::int_op(2),
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))]),
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    let mut store16 = 0;
    count_operators(&bytes, |op| {
        if matches!(op, wasmparser::Operator::I64Store16 { .. }) {
            store16 += 1;
        }
    });
    assert_eq!(store16, 1, "width-2 GcStore must emit i64.store16");
    assert_eq!(execute_simple_trace(&bytes, &[4096, 0x7BCD]), 0x7BCD);
}

#[test]
fn gc_rewrite_indexed_load_store_execute() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
        InputArg::from_type_rc(Type::Int, 2),
    ];
    let ops = vec![
        make_op(
            OpCode::GcStoreIndexed,
            &[
                OpRef::input_arg_ref(0),
                OpRef::input_arg_int(1),
                OpRef::input_arg_int(2),
                OpRef::const_int(4),
                OpRef::const_int(16),
                OpRef::const_int(4),
            ],
            OpRef::NONE,
        ),
        make_op(
            OpCode::GcLoadIndexedI,
            &[
                OpRef::input_arg_ref(0),
                OpRef::input_arg_int(1),
                OpRef::const_int(4),
                OpRef::const_int(16),
                OpRef::const_int(-4),
            ],
            OpRef::int_op(3),
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(3))]),
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    assert_eq!(execute_simple_trace(&bytes, &[4096, 3, -456]), -456);
}

#[test]
fn zero_array_uses_memory_fill_for_the_rewriter_range() {
    use majit_ir::descr::SimpleArrayDescr;
    use std::sync::Arc;

    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let store = make_op(
        OpCode::GcStore,
        &[
            OpRef::input_arg_ref(0),
            OpRef::const_int(16),
            OpRef::input_arg_int(1),
            OpRef::const_int(8),
        ],
        OpRef::NONE,
    );
    let zero = make_op(
        OpCode::ZeroArray,
        &[
            OpRef::input_arg_ref(0),
            OpRef::const_int(0),
            OpRef::const_int(8),
            OpRef::const_int(1),
            OpRef::const_int(1),
        ],
        OpRef::NONE,
    );
    zero.setdescr(Arc::new(SimpleArrayDescr::new(1, 16, 1, 55, Type::Int)));
    let load = make_op(
        OpCode::GcLoadI,
        &[
            OpRef::input_arg_ref(0),
            OpRef::const_int(16),
            OpRef::const_int(8),
        ],
        OpRef::int_op(2),
    );
    let ops = vec![
        store,
        zero,
        load,
        Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))]),
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    assert_eq!(
        execute_simple_trace(&bytes, &[4096, 0x1122_3344_5566_7788]),
        0
    );
    let mut fills = 0;
    count_operators(&bytes, |op| {
        if matches!(op, wasmparser::Operator::MemoryFill { .. }) {
            fills += 1;
        }
    });
    assert_eq!(fills, 1);
}

/// `genop_discard_cond_call`: CondCallN is a real conditional call, not a no-op.
#[test]
fn cond_call_reload_sits_on_the_arm_that_called() {
    use majit_ir::descr::SimpleCallDescr;
    use std::sync::Arc;

    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Ref, 1),
    ];
    let call = make_op(
        OpCode::CondCallN,
        &[
            OpRef::input_arg_int(0),
            OpRef::const_int(0x100),
            OpRef::input_arg_ref(1),
        ],
        OpRef::NONE,
    );
    call.setdescr(Arc::new(SimpleCallDescr::new(
        0,
        vec![Type::Ref],
        Type::Void,
        false,
        8,
        EffectInfo::default(),
    )));
    let ops = vec![
        call,
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_ref(1))]),
    ];
    let frame = codegen::FrameGeometry::compact(
        codegen::frame_value_slots(&inputargs, &ops),
        codegen::count_ref_homes(&inputargs, &ops),
        0,
    );
    let (bytes, _) = build_module_with_frame(
        &inputargs,
        &ops,
        &indexmap::IndexMap::new(),
        Some(0),
        &codegen::GuardGcTypeInfo::default(),
        frame,
    );
    validate_wasm(&bytes);
    let mut seq: Vec<String> = Vec::new();
    count_operators(&bytes, |op| seq.push(format!("{op:?}")));

    // The reload exists because the callee may collect, so it belongs on the
    // arm that called: between the `else` and the `end` that closes it, not
    // after the join where the untaken arm would pay for it too.
    let home = format!("offset: {},", frame.home_slot_base);
    let else_at = seq
        .iter()
        .position(|o| o == "Else")
        .expect("the cond call puts its trampoline in an else arm");
    let end_at = else_at
        + seq[else_at..]
            .iter()
            .position(|o| o == "End")
            .expect("the else arm closes");
    let reload_at = seq
        .iter()
        .position(|o| o.starts_with("I64Load ") && o.contains(&home))
        .expect("the cond call reloads the live Ref from its home slot");
    assert!(
        else_at < reload_at && reload_at < end_at,
        "the ref-home reload is outside the calling arm: {seq:#?}"
    );
}

#[test]
fn cond_call_n_emits_predicate_and_direct_call() {
    use majit_ir::descr::SimpleCallDescr;
    use std::sync::Arc;

    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let call = make_op(
        OpCode::CondCallN,
        &[
            OpRef::input_arg_int(0),
            OpRef::const_int(0x100),
            OpRef::input_arg_int(1),
        ],
        OpRef::NONE,
    );
    call.setdescr(Arc::new(SimpleCallDescr::new(
        0,
        vec![Type::Int],
        Type::Void,
        false,
        8,
        EffectInfo::default(),
    )));
    let ops = vec![
        call,
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    let (mut eqz, mut ifs) = (0, 0);
    count_operators(&bytes, |op| match op {
        wasmparser::Operator::I64Eqz => eqz += 1,
        wasmparser::Operator::If { .. } => ifs += 1,
        _ => {}
    });
    assert!(eqz >= 1, "CondCallN tests the predicate with i64.eqz");
    assert!(ifs >= 1, "CondCallN wraps the call in an if");
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
}

/// A COND_CALL whose table type is i32 does not match the descr word.
/// The direct `call_indirect` type is the descr's; this callee stays on `jit_call`.
#[test]
fn cond_call_n_i32_arg_keeps_trampoline() {
    use majit_ir::descr::SimpleCallDescr;
    use std::sync::Arc;

    let encoded = majit_backend_wasm::encode_func_sig(&[majit_backend_wasm::FuncSigVal::I32], None);
    with_table_sig(0x100, Some(encoded), || {
        let inputargs = vec![
            InputArg::from_type_rc(Type::Int, 0),
            InputArg::from_type_rc(Type::Ref, 1),
        ];
        let call = make_op(
            OpCode::CondCallN,
            &[
                OpRef::input_arg_int(0),
                OpRef::const_int(0x100),
                OpRef::input_arg_ref(1),
            ],
            OpRef::NONE,
        );
        call.setdescr(Arc::new(SimpleCallDescr::new(
            0,
            vec![Type::Ref],
            Type::Void,
            false,
            0,
            EffectInfo::default(),
        )));
        let ops = vec![
            call,
            Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
        ];
        let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
        validate_wasm(&bytes);
        assert_eq!(import_func_type(&bytes, "jit_call_compact"), Some(1));
        let (indirect_calls, _) = indirect_call_types_and_drop_count(&bytes);
        assert!(indirect_calls.is_empty());
    });
}

/// An i32 table result is not the descr word (`i64` / ref). Widening stays
/// in the host hook; the trace does not `call_indirect` that callee.
#[test]
fn cond_call_value_i32_result_keeps_trampoline() {
    use majit_ir::descr::SimpleCallDescr;
    use std::sync::Arc;

    let encoded =
        majit_backend_wasm::encode_func_sig(&[], Some(majit_backend_wasm::FuncSigVal::I32));
    for (opcode, result_signed) in [
        (OpCode::CondCallValueI, true),
        (OpCode::CondCallValueI, false),
        (OpCode::CondCallValueR, false),
    ] {
        with_table_sig(0x100, Some(encoded), || {
            let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
            let result = if opcode == OpCode::CondCallValueI {
                OpRef::int_op(1)
            } else {
                OpRef::ref_op(1)
            };
            let call = make_op(
                opcode,
                &[OpRef::input_arg_int(0), OpRef::const_int(0x100)],
                result,
            );
            call.setdescr(Arc::new(SimpleCallDescr::new(
                0,
                vec![],
                if opcode == OpCode::CondCallValueI {
                    Type::Int
                } else {
                    Type::Ref
                },
                result_signed,
                4,
                EffectInfo::default(),
            )));
            let ops = vec![call, Op::new(OpCode::Finish, &[rb(result)])];
            let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
            validate_wasm(&bytes);
            assert_eq!(import_func_type(&bytes, "jit_call_compact"), Some(1));
            let (indirect_calls, _) = indirect_call_types_and_drop_count(&bytes);
            assert!(indirect_calls.is_empty());
        });
    }
}

#[test]
fn cond_call_without_call_descr_is_unsupported() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let ops = vec![
        make_op(
            OpCode::CondCallN,
            &[
                OpRef::input_arg_int(0),
                OpRef::const_int(0x100),
                OpRef::input_arg_int(1),
            ],
            OpRef::NONE,
        ),
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
    ];
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.clone(),
        ops,
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    };
    assert!(
        matches!(
            codegen::build_wasm_module(&inputs),
            Err(majit_backend::BackendError::Unsupported(_))
        ),
        "COND_CALL without a call descr must decline"
    );
}

#[test]
fn descr_proven_cond_call_n_uses_direct_table_call() {
    use majit_ir::descr::SimpleCallDescr;
    use std::sync::Arc;

    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let call = make_op(
        OpCode::CondCallN,
        &[
            OpRef::input_arg_int(0),
            OpRef::const_int(42),
            OpRef::input_arg_int(1),
        ],
        OpRef::NONE,
    );
    call.setdescr(Arc::new(SimpleCallDescr::new(
        0,
        vec![Type::Int],
        Type::Void,
        false,
        8,
        EffectInfo::default(),
    )));
    let ops = vec![
        call,
        Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(1))]),
    ];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
    let (calls, drops) = indirect_call_types_and_drop_count(&bytes);
    assert_eq!(calls.len(), 1);
    assert_eq!(
        function_type(&bytes, calls[0].0 as usize),
        (
            vec![wasmparser::ValType::I64],
            vec![wasmparser::ValType::I64]
        )
    );
    assert_eq!(drops, 1, "CondCallN ignores the dummy word result");
}

#[test]
fn descr_proven_cond_call_value_uses_direct_table_call() {
    use majit_ir::descr::SimpleCallDescr;
    use std::sync::Arc;

    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let call = make_op(
        OpCode::CondCallValueI,
        &[
            OpRef::input_arg_int(0),
            OpRef::const_int(42),
            OpRef::input_arg_int(1),
        ],
        OpRef::int_op(2),
    );
    call.setdescr(Arc::new(SimpleCallDescr::new(
        0,
        vec![Type::Int],
        Type::Int,
        false,
        8,
        EffectInfo::default(),
    )));
    let ops = vec![call, Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))])];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
    let (calls, drops) = indirect_call_types_and_drop_count(&bytes);
    assert_eq!(calls.len(), 1);
    assert_eq!(
        function_type(&bytes, calls[0].0 as usize),
        (
            vec![wasmparser::ValType::I64],
            vec![wasmparser::ValType::I64]
        )
    );
    assert_eq!(drops, 0);
}

/// The two metadata opcodes both native backends skip must compile here too.
///
/// An opcode with no arm reaches the catch-all, and a void op's `pos` is
/// `OpRef::NONE`, which `raw_is_constant` rejects — so the catch-all declines
/// the whole trace rather than emitting nothing. `consider_jit_debug` is
/// `pass`, and `RECORD_KNOWN_RESULT` has no backend arm upstream at all
/// (`optimize_RECORD_KNOWN_RESULT` and simplify consume it), so neither owes
/// code and neither may cost the trace its compile.
#[test]
fn debug_and_record_metadata_opcodes_compile_instead_of_declining() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    for opcode in [OpCode::JitDebug, OpCode::RecordKnownResult] {
        let ops = vec![
            make_op(opcode, &[OpRef::input_arg_int(0)], OpRef::NONE),
            Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]),
        ];
        let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
        validate_wasm(&bytes);
    }
}

#[test]
fn fused_cond_call_n_consumes_the_comparison_i32() {
    use majit_ir::descr::SimpleCallDescr;
    use std::sync::Arc;

    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let compare = make_op(
        OpCode::IntLt,
        &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
        OpRef::int_op(2),
    );
    let call = make_op(
        OpCode::CondCallN,
        &[
            OpRef::int_op(2),
            OpRef::const_int(0x30),
            OpRef::input_arg_int(0),
        ],
        OpRef::NONE,
    );
    call.setdescr(Arc::new(SimpleCallDescr::new(
        0x5200_0000,
        vec![Type::Int],
        Type::Void,
        true,
        0,
        EffectInfo::default(),
    )));
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(1))]);
    let (bytes, _) = build_module_default(
        &inputargs,
        &[compare, call, finish],
        &indexmap::IndexMap::new(),
    );
    validate_wasm(&bytes);

    let mut i64_eqz = 0;
    let mut i64_lt_s = 0;
    let mut i64_extend = 0;
    count_operators(&bytes, |op| match op {
        wasmparser::Operator::I64Eqz => i64_eqz += 1,
        wasmparser::Operator::I64LtS => i64_lt_s += 1,
        wasmparser::Operator::I64ExtendI32U => i64_extend += 1,
        _ => {}
    });
    assert_eq!(i64_lt_s, 1, "the comparison must still be emitted");
    assert_eq!(
        i64_eqz, 0,
        "a fused CondCallN must not re-test a materialised boolean"
    );
    assert_eq!(
        i64_extend, 0,
        "a fused CondCallN must not widen the comparison into a local"
    );
}

#[test]
fn fused_cond_call_value_consumes_the_comparison_i32() {
    use majit_ir::descr::SimpleCallDescr;
    use std::sync::Arc;

    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let compare = make_op(
        OpCode::IntEq,
        &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
        OpRef::int_op(2),
    );
    let call = make_op(
        OpCode::CondCallValueI,
        &[
            OpRef::int_op(2),
            OpRef::const_int(0x30),
            OpRef::input_arg_int(0),
        ],
        OpRef::int_op(3),
    );
    call.setdescr(Arc::new(SimpleCallDescr::new(
        0x5200_0000,
        vec![Type::Int],
        Type::Int,
        true,
        8,
        EffectInfo::default(),
    )));
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::int_op(3))]);
    let (bytes, _) = build_module_default(
        &inputargs,
        &[compare, call, finish],
        &indexmap::IndexMap::new(),
    );
    validate_wasm(&bytes);

    let mut seq: Vec<String> = Vec::new();
    count_operators(&bytes, |op| seq.push(format!("{op:?}")));
    let eq_at = seq
        .iter()
        .position(|o| o == "I64Eq")
        .expect("the comparison must still be emitted");
    assert_eq!(seq.iter().filter(|o| *o == "I64Eq").count(), 1);
    assert_eq!(
        seq.get(eq_at + 1).map(String::as_str),
        Some("I64ExtendI32U"),
        "fused CondCallValue consumes the comparison i32 directly: {seq:#?}"
    );
    assert!(
        seq.get(eq_at + 2)
            .is_some_and(|o| o.starts_with("LocalTee")),
        "fused arm tees the result; unfused would LocalSet the compare: {seq:#?}"
    );
}

#[test]
fn fused_guard_isnull_uses_the_comparison_directly() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let compare = make_op(
        OpCode::IntEq,
        &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
        OpRef::int_op(2),
    );
    let guard = make_guard(
        OpCode::GuardIsnull,
        &[OpRef::int_op(2)],
        &[OpRef::input_arg_int(0)],
    );
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(1))]);
    let (bytes, _) = build_module_default(
        &inputargs,
        &[compare, guard, finish],
        &indexmap::IndexMap::new(),
    );
    validate_wasm(&bytes);
    let mut i64_eq = 0;
    let mut i64_extend = 0;
    count_operators(&bytes, |op| match op {
        wasmparser::Operator::I64Eq => i64_eq += 1,
        wasmparser::Operator::I64ExtendI32U => i64_extend += 1,
        _ => {}
    });
    assert_eq!(i64_eq, 1);
    assert_eq!(
        i64_extend, 0,
        "fused GuardIsnull must not materialize a local"
    );
}

#[test]
fn threadlocalref_get_lowers_through_the_tls_helper() {
    for offset in [-8, 0, 7, 8, 512, i64::MAX] {
        let mut inputs = inline_region_inputs(
            &[],
            vec![
                make_op(
                    OpCode::ThreadlocalrefGet,
                    &[OpRef::const_int(offset)],
                    OpRef::int_op(1),
                ),
                Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]),
            ],
            vec![],
        );
        inputs.inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
        inputs.alloc.threadlocal_fn_ptr = 0x66;
        let (bytes, _, _, _) =
            codegen::build_wasm_module(&inputs).expect("ThreadlocalrefGet should lower");
        validate_wasm(&bytes);

        inputs.alloc.threadlocal_fn_ptr = 0;
        assert!(
            matches!(
                codegen::build_wasm_module(&inputs),
                Err(majit_backend::BackendError::Unsupported(_))
            ),
            "ThreadlocalrefGet must decline without a TLS helper"
        );
    }
}

#[test]
fn cond_call_materializes_a_condition_reused_as_an_argument_or_callee() {
    use majit_ir::descr::SimpleCallDescr;
    use std::sync::Arc;
    for condition_is_callee in [false, true] {
        let condition = OpRef::int_op(2);
        let compare = make_op(
            OpCode::IntLt,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            condition,
        );
        let call = make_op(
            OpCode::CondCallN,
            &[
                condition,
                if condition_is_callee {
                    condition
                } else {
                    OpRef::const_int(1)
                },
                condition,
            ],
            OpRef::NONE,
        );
        call.setdescr(Arc::new(SimpleCallDescr::new(
            0x5200_0000,
            vec![Type::Int],
            Type::Void,
            true,
            0,
            EffectInfo::default(),
        )));
        let (bytes, _) = build_module_default(
            &[
                InputArg::from_type_rc(Type::Int, 0),
                InputArg::from_type_rc(Type::Int, 1),
            ],
            &[compare, call, Op::new(OpCode::Finish, &[])],
            &indexmap::IndexMap::new(),
        );
        validate_wasm(&bytes);
        let mut widens = 0;
        count_operators(&bytes, |op| {
            if matches!(op, wasmparser::Operator::I64ExtendI32U) {
                widens += 1;
            }
        });
        assert_eq!(widens, 1, "the shared condition must be materialized");
    }
}

#[test]
fn unlowered_virtual_refs_and_errno_calls_decline() {
    for opcode in [
        OpCode::VirtualRefI,
        OpCode::VirtualRefR,
        OpCode::VirtualRefFinish,
    ] {
        let op = make_op(
            opcode,
            &[OpRef::input_arg_ref(0), OpRef::const_int(1)],
            if opcode == OpCode::VirtualRefFinish {
                OpRef::NONE
            } else {
                OpRef::ref_op(1)
            },
        );
        let inputs = inline_region_inputs(
            &[InputArg::from_type_rc(Type::Ref, 0)],
            vec![op, Op::new(OpCode::Finish, &[])],
            vec![],
        );
        assert!(matches!(
            codegen::build_wasm_module(&inputs),
            Err(majit_backend::BackendError::Unsupported(_))
        ));
    }
    for save_err in [OpRef::const_int(1), OpRef::input_arg_int(0)] {
        let call = make_op(
            OpCode::CallReleaseGilI,
            &[save_err, OpRef::const_int(0x22)],
            OpRef::int_op(1),
        );
        let inputs = inline_region_inputs(
            &[InputArg::from_type_rc(Type::Int, 0)],
            vec![call, Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))])],
            vec![],
        );
        assert!(matches!(
            codegen::build_wasm_module(&inputs),
            Err(majit_backend::BackendError::Unsupported(_))
        ));
    }
}

#[test]
fn call_release_gil_save_err_lowers_through_the_errno_helpers() {
    let call = make_op(
        OpCode::CallReleaseGilI,
        &[OpRef::const_int(1), OpRef::const_int(0x22)],
        OpRef::int_op(1),
    );
    let mut inputs = inline_region_inputs(
        &[InputArg::from_type_rc(Type::Int, 0)],
        vec![call, Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))])],
        vec![],
    );
    inputs.alloc.write_real_errno_fn_ptr = 0x77;
    inputs.alloc.read_real_errno_fn_ptr = 0x78;
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("CALL_RELEASE_GIL with save_err should lower");
    validate_wasm(&bytes);
}

fn count_rewritten_malloc_nursery(ops: &[Op]) -> (usize, usize) {
    let mallocs = ops
        .iter()
        .filter(|op| op.opcode == OpCode::CallMallocNursery)
        .count();
    let npis = ops
        .iter()
        .filter(|op| op.opcode == OpCode::NurseryPtrIncrement)
        .count();
    (mallocs, npis)
}

#[test]
fn consecutive_allocations_home_and_reload_before_each_collection() {
    let refs = [OpRef::ref_op(1), OpRef::ref_op(2), OpRef::ref_op(3)];
    let mut ops: Vec<_> = refs
        .iter()
        .map(|&result| call_malloc_nursery(result.raw(), 32))
        .collect();
    ops.push(Op::new(OpCode::Finish, &refs.map(rb)));
    let mut inputs = inline_region_inputs(&[], ops, vec![]);
    inputs.frame = codegen::FrameGeometry::compact(5, 2, 0);
    inputs.alloc.new_fn_ptr = 1;
    inputs.nursery = None;
    let home_base = inputs.frame.home_slot_base as usize;
    let (bytes, _, homes, _) = codegen::build_wasm_module(&rewrite_module_inputs(inputs)).unwrap();
    assert_eq!(
        homes, 2,
        "the first two objects cross a subsequent allocation"
    );
    validate_wasm(&bytes);
    let engine = Engine::default();
    let module = Module::new(&engine, &bytes).unwrap();
    let mut store = Store::new(&engine, Vec::<i64>::new());
    let memory = Memory::new(&mut store, MemoryType::new(1, None)).unwrap();
    let alloc = wasmi::Func::wrap(
        &mut store,
        move |mut caller: wasmi::Caller<'_, Vec<i64>>, tid: i64, _size: i64| -> i64 {
            assert_eq!(tid, 0, "CallMallocNursery slow path stamps tid later");
            let prior = caller.data().clone();
            for (index, old) in prior.into_iter().enumerate() {
                let mut bits = [0; 8];
                memory
                    .read(&caller, home_base + index * 8, &mut bits)
                    .unwrap();
                assert_eq!(
                    i64::from_le_bytes(bits),
                    old,
                    "object must be homed before collecting"
                );
                let moved = old + 0x100;
                memory
                    .write(&mut caller, home_base + index * 8, &moved.to_le_bytes())
                    .unwrap();
                caller.data_mut()[index] = moved;
            }
            let result = 0x4000 + caller.data().len() as i64 * 0x1000;
            caller.data_mut().push(result);
            result
        },
    );
    let table = Table::new(
        &mut store,
        TableType::new(ValType::FuncRef, 2, None),
        Val::default(ValType::FuncRef),
    )
    .unwrap();
    table.set(&mut store, 1, Val::from(alloc)).unwrap();
    let mut linker = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .define("env", "__indirect_function_table", table)
        .unwrap();
    let instance = linker.instantiate_and_start(&mut store, &module).unwrap();
    instance
        .get_typed_func::<i32, i32>(&store, "trace")
        .unwrap()
        .call(&mut store, 0)
        .unwrap();
    assert_eq!(store.data().len(), 3);
    for (index, expected) in store.data().iter().enumerate() {
        let mut bits = [0; 8];
        memory
            .read(
                &store,
                codegen::FRAME_SLOT_BASE as usize + index * 8,
                &mut bits,
            )
            .unwrap();
        assert_eq!(
            i64::from_le_bytes(bits),
            *expected,
            "live local must reload its moved home"
        );
    }
}

fn nursery_new_inputs(ops: Vec<Op>, plain_tid: u32) -> codegen::ModuleBuildInputs {
    let mut plain_tids = std::collections::HashSet::new();
    plain_tids.insert(plain_tid);
    let inputs = codegen::ModuleBuildInputs {
        inputargs: vec![InputArg::from_type_rc(Type::Int, 0)],
        ops,
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers {
            new_fn_ptr: 0x11,
            new_array_fn_ptr: 0x22,
            headerless_fn_ptr: 0x55,
            threadlocal_fn_ptr: 0x66,
            fmod_fn_ptr: 0,
            write_real_errno_fn_ptr: 0x77,
            read_real_errno_fn_ptr: 0x78,
        },
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: Some(codegen::NurseryAllocParams {
            free_addr: 0x1000,
            top_addr: 0x1004,
            large_threshold: 4096,
            plain_tids,
        }),
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::fixed(),
        ca: codegen::CaParams::default(),
    };
    rewrite_module_inputs(inputs)
}

fn nursery_top_compare_count(bytes: &[u8]) -> usize {
    let mut nursery_top_compares = 0;
    count_operators(bytes, |op| {
        if matches!(op, wasmparser::Operator::I32GtU) {
            nursery_top_compares += 1;
        }
    });
    nursery_top_compares
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct NurseryArmStores {
    i64: usize,
    i32: usize,
    i64_32: usize,
    i64_16: usize,
    i64_off0: usize,
    i32_off0: usize,
}

fn bump_store(counts: &mut NurseryArmStores, op: &wasmparser::Operator<'_>) {
    match op {
        wasmparser::Operator::I64Store { memarg } => {
            counts.i64 += 1;
            if memarg.offset == 0 {
                counts.i64_off0 += 1;
            }
        }
        wasmparser::Operator::I32Store { memarg } => {
            counts.i32 += 1;
            if memarg.offset == 0 {
                counts.i32_off0 += 1;
            }
        }
        wasmparser::Operator::I64Store32 { .. } => counts.i64_32 += 1,
        wasmparser::Operator::I64Store16 { .. } => counts.i64_16 += 1,
        wasmparser::Operator::I32Store16 { .. } => counts.i64_16 += 1,
        _ => {}
    }
}

/// Stores in the `CallMallocNursery` overflow `if` (slow) and `else` (fast).
fn nursery_malloc_arm_stores(bytes: &[u8]) -> (NurseryArmStores, NurseryArmStores) {
    let mut slow = NurseryArmStores::default();
    let mut fast = NurseryArmStores::default();
    let mut pending_overflow = false;
    let mut depth = 0i32;
    let mut arm: Option<bool> = None; // false = slow, true = fast
    count_operators(bytes, |op| match op {
        wasmparser::Operator::I32GtU => pending_overflow = true,
        wasmparser::Operator::If { .. } if pending_overflow && arm.is_none() => {
            pending_overflow = false;
            arm = Some(false);
            depth = 1;
        }
        wasmparser::Operator::If { .. }
        | wasmparser::Operator::Block { .. }
        | wasmparser::Operator::Loop { .. }
            if arm.is_some() =>
        {
            pending_overflow = false;
            depth += 1;
        }
        wasmparser::Operator::Else if arm == Some(false) && depth == 1 => {
            arm = Some(true);
        }
        wasmparser::Operator::End if arm.is_some() => {
            depth -= 1;
            if depth == 0 {
                arm = None;
            }
        }
        other => {
            pending_overflow = false;
            match arm {
                Some(false) => bump_store(&mut slow, other),
                Some(true) => bump_store(&mut fast, other),
                None => {}
            }
        }
    });
    (slow, fast)
}

fn tid_gc_store(obj: u32, tid: i64) -> Op {
    make_op(
        OpCode::GcStore,
        &[
            OpRef::ref_op(obj),
            OpRef::const_int(-(std::mem::size_of::<usize>() as i64)),
            OpRef::const_int(tid),
            // `gen_initialize_tid`: the whole Signed `HDR.tid`.
            OpRef::const_int(std::mem::size_of::<usize>() as i64),
        ],
        OpRef::NONE,
    )
}

fn finish_int_arg0() -> Op {
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]);
    finish.setfailargs(smallvec![rb(OpRef::input_arg_int(0))]);
    finish
}

fn plain_new(result: u32, type_id: u32) -> Op {
    sized_new(result, type_id, 16)
}

fn sized_new(result: u32, type_id: u32, size: usize) -> Op {
    use majit_ir::descr::SimpleSizeDescr;
    use std::sync::Arc;
    let descr = SimpleSizeDescr::new(0, size, type_id);
    descr.set_non_moving(false);
    let op = make_op(OpCode::New, &[], OpRef::ref_op(result));
    op.setdescr(Arc::new(descr));
    op
}

fn plain_new_array(result: u32, type_id: u32, length: i64) -> Op {
    use majit_ir::descr::SimpleArrayDescr;
    use std::sync::Arc;
    let descr = SimpleArrayDescr::new(1, 16, 8, type_id, Type::Int);
    let op = make_op(
        OpCode::NewArray,
        &[OpRef::const_int(length)],
        OpRef::ref_op(result),
    );
    op.setdescr(Arc::new(descr));
    op
}

fn plain_new_array_clear(result: u32, type_id: u32, length: i64) -> Op {
    use majit_ir::descr::SimpleArrayDescr;
    use std::sync::Arc;
    let descr = SimpleArrayDescr::new(1, 16, 8, type_id, Type::Int);
    let op = make_op(
        OpCode::NewArrayClear,
        &[OpRef::const_int(length)],
        OpRef::ref_op(result),
    );
    op.setdescr(Arc::new(descr));
    op
}

fn setarrayitem_gc(array: u32, index: i64, value: OpRef, type_id: u32) -> Op {
    use majit_ir::descr::SimpleArrayDescr;
    use std::sync::Arc;
    let op = make_op(
        OpCode::SetarrayitemGc,
        &[OpRef::ref_op(array), OpRef::const_int(index), value],
        OpRef::NONE,
    );
    op.setdescr(Arc::new(SimpleArrayDescr::new(
        1,
        16,
        8,
        type_id,
        Type::Int,
    )));
    op
}

fn memory_fill_count(bytes: &[u8]) -> usize {
    let mut fills = 0;
    count_operators(bytes, |op| {
        if matches!(op, wasmparser::Operator::MemoryFill { .. }) {
            fills += 1;
        }
    });
    fills
}

/// rewrite.py: plain NEW_ARRAY never emits ZERO_ARRAY.
#[test]
fn inline_newarray_does_not_memory_fill() {
    let inputs = nursery_new_inputs(vec![plain_new_array(1, 53, 2), finish_int_arg0()], 53);
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert_eq!(
        memory_fill_count(&bytes),
        0,
        "NEW_ARRAY must not memory.fill; rewrite emits ZERO_ARRAY only for CLEAR"
    );
}

/// rewrite.py emit_pending_zeros: SETARRAYITEM covering 0..len-1
/// rewrites ZERO_ARRAY to a no-op.
#[test]
fn inline_newarray_clear_skips_fill_when_every_item_is_stored() {
    let inputs = nursery_new_inputs(
        vec![
            plain_new_array_clear(1, 53, 2),
            setarrayitem_gc(1, 0, OpRef::const_int(7), 53),
            setarrayitem_gc(1, 1, OpRef::const_int(8), 53),
            finish_int_arg0(),
        ],
        53,
    );
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert!(
        memory_fill_lengths(&bytes).iter().all(|&n| n == 0),
        "fully-written NEW_ARRAY_CLEAR is a ZERO_ARRAY no-op"
    );
}

#[test]
fn inline_newarray_clear_fills_when_an_item_is_unwritten() {
    let inputs = nursery_new_inputs(
        vec![
            plain_new_array_clear(1, 53, 2),
            setarrayitem_gc(1, 0, OpRef::const_int(7), 53),
            finish_int_arg0(),
        ],
        53,
    );
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert!(
        memory_fill_count(&bytes) >= 1,
        "partial NEW_ARRAY_CLEAR still ZERO_ARRAYs the unwritten tail"
    );
    let fills = memory_fill_lengths(&bytes);
    assert!(
        fills.iter().any(|&n| n == 8),
        "emit_pending_zeros trims the written prefix; leftover is one item (8 bytes), got {fills:?}"
    );
}

fn memory_fill_lengths(bytes: &[u8]) -> Vec<i32> {
    let mut last_const = None;
    let mut lengths = Vec::new();
    count_operators(bytes, |op| match op {
        wasmparser::Operator::I32Const { value } => last_const = Some(*value),
        wasmparser::Operator::MemoryFill { .. } => {
            if let Some(n) = last_const {
                lengths.push(n);
            }
        }
        _ => last_const = None,
    });
    lengths
}

/// rewrite.py flushes pending zeros at a guard, so a later SETARRAYITEM
/// does not cancel ZERO_ARRAY for the unwritten slot.
#[test]
fn inline_newarray_clear_fills_when_a_guard_splits_the_stores() {
    let guard = make_guard(
        OpCode::GuardTrue,
        &[OpRef::input_arg_int(0)],
        &[OpRef::input_arg_int(0)],
    );
    let inputs = nursery_new_inputs(
        vec![
            plain_new_array_clear(1, 53, 2),
            setarrayitem_gc(1, 0, OpRef::const_int(7), 53),
            guard,
            setarrayitem_gc(1, 1, OpRef::const_int(8), 53),
            finish_int_arg0(),
        ],
        53,
    );
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert!(
        memory_fill_count(&bytes) >= 1,
        "SETARRAYITEM after a guard must not rewrite ZERO_ARRAY to a no-op"
    );
}

fn new_with_gc_field(result: u32, type_id: u32, field_offset: usize) -> Op {
    use majit_ir::descr::{SimpleFieldDescr, SimpleSizeDescr};
    use std::sync::Arc;
    let fd = Arc::new(SimpleFieldDescr::new(0, field_offset, 4, Type::Ref, false));
    let descr = SimpleSizeDescr::new(0, 16, type_id).with_all_fielddescrs(vec![fd]);
    descr.set_non_moving(false);
    let op = make_op(OpCode::New, &[], OpRef::ref_op(result));
    op.setdescr(Arc::new(descr));
    op
}

fn setfield_gc(obj: u32, value: OpRef, offset: usize) -> Op {
    use majit_ir::descr::SimpleFieldDescr;
    use std::sync::Arc;
    let op = make_op(
        OpCode::SetfieldGc,
        &[OpRef::ref_op(obj), value],
        OpRef::NONE,
    );
    op.setdescr(Arc::new(SimpleFieldDescr::new(
        0,
        offset,
        4,
        Type::Ref,
        false,
    )));
    op
}

fn i32_store_const0_count(bytes: &[u8]) -> usize {
    let mut prev_const0 = false;
    let mut stores = 0;
    count_operators(bytes, |op| match op {
        wasmparser::Operator::I32Const { value: 0 }
        | wasmparser::Operator::I64Const { value: 0 } => prev_const0 = true,
        wasmparser::Operator::I32Store { .. } | wasmparser::Operator::I64Store { .. }
            if prev_const0 =>
        {
            stores += 1;
            prev_const0 = false;
        }
        _ => prev_const0 = false,
    });
    stores
}

/// rewrite.py clear_gc_fields: a New with no GC-pointer fields emits
/// no ZERO / NULL stores. IncrementalMiniMark does not fill the payload.
#[test]
fn inline_new_does_not_memory_fill() {
    let inputs = nursery_new_inputs(vec![plain_new(1, 53), finish_int_arg0()], 53);
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert_eq!(
        memory_fill_count(&bytes),
        0,
        "NEW must not memory.fill; rewrite only NULLs leftover GC fields"
    );
}

/// rewrite.py emit_pending_zeros: leftover gc_fielddescrs become
/// pointer-width NULL stores, not a payload fill.
#[test]
fn inline_new_nulls_unwritten_gc_field() {
    let inputs = nursery_new_inputs(vec![new_with_gc_field(1, 53, 8), finish_int_arg0()], 53);
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert_eq!(memory_fill_count(&bytes), 0, "NEW does not memory.fill");
    assert!(
        i32_store_const0_count(&bytes) >= 1,
        "unwritten GC field still gets the delayed NULL store"
    );
}

/// rewrite.py consider_setfield_gc drops a field SETFIELD_GC covers
/// before the next flush, so emit_pending_zeros is empty.
#[test]
fn inline_new_skips_null_when_setfield_covers() {
    let uncovered = nursery_new_inputs(vec![new_with_gc_field(1, 53, 8), finish_int_arg0()], 53);
    let covered = nursery_new_inputs(
        vec![
            new_with_gc_field(1, 53, 8),
            setfield_gc(1, OpRef::input_arg_int(0), 8),
            finish_int_arg0(),
        ],
        53,
    );
    let (uncovered_bytes, _, _, _) =
        codegen::build_wasm_module(&uncovered).expect("wasm codegen should succeed");
    let (covered_bytes, _, _, _) =
        codegen::build_wasm_module(&covered).expect("wasm codegen should succeed");
    validate_wasm(&uncovered_bytes);
    validate_wasm(&covered_bytes);
    assert_eq!(memory_fill_count(&covered_bytes), 0);
    assert!(
        i32_store_const0_count(&covered_bytes) < i32_store_const0_count(&uncovered_bytes),
        "SETFIELD_GC before the flush cancels the delayed NULL"
    );
}

/// rewrite.py `transform_to_gc_load` flushes pending zeros before
/// GETFIELD_GC, so a later SETFIELD_GC must not cancel the NULL the
/// load is about to observe.
#[test]
fn inline_new_nulls_gc_field_read_before_setfield() {
    let fd = {
        use majit_ir::descr::SimpleFieldDescr;
        use std::sync::Arc;
        Arc::new(SimpleFieldDescr::new(0, 8, 4, Type::Ref, false))
    };
    let getfield = make_op(OpCode::GetfieldGcR, &[OpRef::ref_op(1)], OpRef::ref_op(2));
    getfield.setdescr(fd);
    let uncovered = nursery_new_inputs(vec![new_with_gc_field(1, 53, 8), finish_int_arg0()], 53);
    let read_then_store = nursery_new_inputs(
        vec![
            new_with_gc_field(1, 53, 8),
            getfield,
            setfield_gc(1, OpRef::input_arg_int(0), 8),
            finish_int_arg0(),
        ],
        53,
    );
    let (uncovered_bytes, _, _, _) =
        codegen::build_wasm_module(&uncovered).expect("wasm codegen should succeed");
    let (read_bytes, _, _, _) =
        codegen::build_wasm_module(&read_then_store).expect("wasm codegen should succeed");
    validate_wasm(&uncovered_bytes);
    validate_wasm(&read_bytes);
    assert_eq!(
        i32_store_const0_count(&read_bytes),
        i32_store_const0_count(&uncovered_bytes),
        "GETFIELD_GC flushes pending NULLs; a later SETFIELD must not drop them"
    );
}

/// rewrite.py `handle_clear_array_contents` emits ZERO_ARRAY before a
/// later GETARRAYITEM, so a store after that read must not trim the fill.
#[test]
fn inline_newarray_clear_fills_when_an_item_is_read_before_store() {
    let getitem = {
        use majit_ir::descr::SimpleArrayDescr;
        use std::sync::Arc;
        let op = make_op(
            OpCode::GetarrayitemGcI,
            &[OpRef::ref_op(1), OpRef::const_int(0)],
            OpRef::int_op(2),
        );
        op.setdescr(Arc::new(SimpleArrayDescr::new(1, 16, 8, 53, Type::Int)));
        op
    };
    let inputs = nursery_new_inputs(
        vec![
            plain_new_array_clear(1, 53, 1),
            getitem,
            setarrayitem_gc(1, 0, OpRef::const_int(7), 53),
            finish_int_arg0(),
        ],
        53,
    );
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert!(
        memory_fill_count(&bytes) >= 1,
        "GETARRAYITEM of the new array must keep ZERO_ARRAY for the unread-then-stored slot"
    );
}

/// `handle_new` only skips the class-word NULL after it stamps `w_class`.
/// A class-word field with no `w_class_obj` still needs the delayed NULL.
#[test]
fn inline_new_nulls_class_word_when_w_class_is_not_stamped() {
    use majit_ir::descr::{SimpleFieldDescr, SimpleSizeDescr};
    use std::sync::Arc;
    let fd = Arc::new(SimpleFieldDescr::new(0, 8, 4, Type::Ref, false).with_class_word(true));
    let descr = SimpleSizeDescr::new(0, 16, 53).with_all_fielddescrs(vec![fd]);
    descr.set_non_moving(false);
    let new_op = make_op(OpCode::New, &[], OpRef::ref_op(1));
    new_op.setdescr(Arc::new(descr));
    let inputs = nursery_new_inputs(vec![new_op, finish_int_arg0()], 53);
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert!(
        i32_store_const0_count(&bytes) >= 1,
        "class-word field without a stamped w_class must still get the delayed NULL"
    );
}

/// rewrite.py `remember_setarrayitem_occurred` canonicalizes through
/// `get_box_replacement`, so SameAsR aliases of the array still trim.
#[test]
fn inline_newarray_clear_trims_stores_through_same_as_r() {
    let alias = make_op(OpCode::SameAsR, &[OpRef::ref_op(1)], OpRef::ref_op(2));
    let store0 = {
        use majit_ir::descr::SimpleArrayDescr;
        use std::sync::Arc;
        let op = make_op(
            OpCode::SetarrayitemGc,
            &[OpRef::ref_op(2), OpRef::const_int(0), OpRef::const_int(7)],
            OpRef::NONE,
        );
        op.setdescr(Arc::new(SimpleArrayDescr::new(1, 16, 8, 53, Type::Int)));
        op
    };
    let store1 = {
        use majit_ir::descr::SimpleArrayDescr;
        use std::sync::Arc;
        let op = make_op(
            OpCode::SetarrayitemGc,
            &[OpRef::ref_op(2), OpRef::const_int(1), OpRef::const_int(8)],
            OpRef::NONE,
        );
        op.setdescr(Arc::new(SimpleArrayDescr::new(1, 16, 8, 53, Type::Int)));
        op
    };
    let inputs = nursery_new_inputs(
        vec![
            plain_new_array_clear(1, 53, 2),
            alias,
            store0,
            store1,
            finish_int_arg0(),
        ],
        53,
    );
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert!(
        memory_fill_lengths(&bytes).iter().all(|&n| n == 0),
        "SETARRAYITEM through a SameAsR alias must still rewrite ZERO_ARRAY to a no-op"
    );
}

/// rewrite.py emits pending zeros at a guard, so a SETFIELD after the
/// guard does not cancel the NULL store.
#[test]
fn inline_new_nulls_gc_field_when_a_guard_splits_the_store() {
    let guard = make_guard(
        OpCode::GuardTrue,
        &[OpRef::input_arg_int(0)],
        &[OpRef::input_arg_int(0)],
    );
    let uncovered = nursery_new_inputs(vec![new_with_gc_field(1, 53, 8), finish_int_arg0()], 53);
    let split = nursery_new_inputs(
        vec![
            new_with_gc_field(1, 53, 8),
            guard,
            setfield_gc(1, OpRef::input_arg_int(0), 8),
            finish_int_arg0(),
        ],
        53,
    );
    let (uncovered_bytes, _, _, _) =
        codegen::build_wasm_module(&uncovered).expect("wasm codegen should succeed");
    let (split_bytes, _, _, _) =
        codegen::build_wasm_module(&split).expect("wasm codegen should succeed");
    validate_wasm(&split_bytes);
    assert_eq!(memory_fill_count(&split_bytes), 0);
    assert_eq!(
        i32_store_const0_count(&split_bytes),
        i32_store_const0_count(&uncovered_bytes),
        "SETFIELD_GC after a guard must not cancel the delayed NULL"
    );
}
#[test]
fn consecutive_new_ops_share_one_nursery_bump() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let frontend = vec![plain_new(1, 53), plain_new(2, 53), finish_int_arg0()];
    let (rewritten, _) = rewrite_frontend_ops(&inputargs, frontend.clone());
    let (mallocs, npis) = count_rewritten_malloc_nursery(&rewritten);
    assert_eq!(mallocs, 1, "gen_malloc_nursery merges consecutive New ops");
    assert_eq!(npis, 1);
    let inputs = nursery_new_inputs(frontend, 53);
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert_eq!(
        nursery_top_compare_count(&bytes),
        1,
        "the merged CallMallocNursery is one bump"
    );
}

#[test]
fn news_that_fill_the_nursery_threshold_keep_separate_bumps() {
    // Each half of `nursery_new_inputs`'s 4096 `large_threshold` still takes
    // the bump; the two together do not, so the batch cannot merge them.
    let frontend = vec![
        sized_new(1, 53, 2048),
        sized_new(2, 53, 2048),
        finish_int_arg0(),
    ];
    let inputs = nursery_new_inputs(frontend, 53);
    let (mallocs, _) = count_rewritten_malloc_nursery(&inputs.ops);
    assert_eq!(
        mallocs, 2,
        "combined aligned size must stay strictly below large_threshold"
    );
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert_eq!(nursery_top_compare_count(&bytes), 2);
}

#[test]
fn inlined_region_new_does_not_join_the_owners_nursery_batch() {
    let descr = majit_ir::make_loop_target_descr(10, false);
    let label = Op::new(OpCode::Label, &[rb(OpRef::input_arg_int(0))]);
    label.setdescr(descr.clone());
    let guard = make_guard(
        OpCode::GuardTrue,
        &[OpRef::input_arg_int(0)],
        &[OpRef::input_arg_int(0)],
    );
    let jump = Op::new(OpCode::Jump, &[rb(OpRef::input_arg_int(0))]);
    jump.setdescr(descr);
    let region_finish = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(10))]);
    region_finish.setfailargs(smallvec![rb(OpRef::input_arg_int(10))]);

    let owner_ia = vec![InputArg::from_type_rc(Type::Int, 0)];
    let region_ia = vec![InputArg::from_type_rc(Type::Int, 10)];
    let (owner_ops, owner_consts) =
        rewrite_frontend_ops(&owner_ia, vec![label, plain_new(2, 53), guard, jump]);
    let (region_ops, region_consts) =
        rewrite_frontend_ops(&region_ia, vec![plain_new(11, 53), region_finish]);
    let (owner_mallocs, _) = count_rewritten_malloc_nursery(&owner_ops);
    let (region_mallocs, _) = count_rewritten_malloc_nursery(&region_ops);
    assert_eq!(owner_mallocs, 1, "owner rewritten on its own");
    assert_eq!(
        region_mallocs, 1,
        "region rewritten on its own, not batched with owner"
    );

    let mut inputs = nursery_new_inputs(
        vec![Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))])],
        53,
    );
    inputs.inputargs = owner_ia;
    inputs.ops = owner_ops;
    inputs.constants = owner_consts;
    inputs.inlined_bridges = vec![codegen::InlinedBridge {
        external_jump: None,
        source_fail_index: 0,
        outside_loop: false,
        trace_id: 1,
        inputargs: region_ia,
        ops: region_ops,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        constants: region_consts,
    }];
    let inputs = rewrite_module_inputs(inputs);
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert_eq!(
        nursery_top_compare_count(&bytes),
        2,
        "an inlined region must not follow an owner-side nursery leader"
    );
}

#[test]
fn collecting_op_between_news_keeps_separate_nursery_bumps() {
    use majit_ir::descr::SimpleCallDescr;
    use std::sync::Arc;

    let call = Op::new(OpCode::CallN, &[rb(OpRef::const_int(0x99))]);
    call.setdescr(Arc::new(SimpleCallDescr::new(
        0x5300_0000,
        Vec::new(),
        Type::Void,
        true,
        0,
        EffectInfo::default(),
    )));
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let frontend = vec![plain_new(1, 53), call, plain_new(2, 53), finish_int_arg0()];
    let (rewritten, _) = rewrite_frontend_ops(&inputargs, frontend.clone());
    let (mallocs, _) = count_rewritten_malloc_nursery(&rewritten);
    assert_eq!(
        mallocs, 2,
        "a collecting op flushes the pending nursery batch"
    );
    let inputs = nursery_new_inputs(frontend, 53);
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert_eq!(nursery_top_compare_count(&bytes), 2);
}

#[test]
fn runtime_newarray_flushes_the_nursery_batch() {
    use majit_ir::descr::SimpleArrayDescr;
    use std::sync::Arc;

    let descr = SimpleArrayDescr::new(1, 16, 8, 55, Type::Int);
    let runtime = make_op(
        OpCode::NewArray,
        &[OpRef::input_arg_int(0)],
        OpRef::ref_op(2),
    );
    runtime.setdescr(Arc::new(descr));
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let frontend = vec![
        plain_new(1, 53),
        runtime,
        plain_new(3, 53),
        finish_int_arg0(),
    ];
    let (rewritten, _) = rewrite_frontend_ops(&inputargs, frontend.clone());
    let (mallocs, _) = count_rewritten_malloc_nursery(&rewritten);
    let call_r = rewritten
        .iter()
        .filter(|op| op.opcode == OpCode::CallR)
        .count();
    assert_eq!(
        mallocs, 2,
        "a runtime-length NewArray flushes the fixed nursery batch"
    );
    assert_eq!(
        call_r, 1,
        "nonstandard runtime NewArray takes CallR, not CallMallocNurseryVarsize"
    );
    let inputs = nursery_new_inputs(frontend, 53);
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert_eq!(
        nursery_top_compare_count(&bytes),
        2,
        "a runtime-length NewArray stays on the varsize path and flushes"
    );
}

#[test]
fn newstr_without_a_descr_injects_the_builtin_layout() {
    let newstr = make_op(OpCode::Newstr, &[OpRef::const_int(3)], OpRef::ref_op(1));
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let frontend = vec![newstr, finish_int_arg0()];
    let (rewritten, _) = rewrite_frontend_ops(&inputargs, frontend.clone());
    assert!(
        rewritten.iter().all(|op| op.opcode != OpCode::Newstr),
        "the compile path injects str_descr and the rewrite consumes Newstr"
    );
    assert!(
        rewritten.iter().any(|op| matches!(
            op.opcode,
            OpCode::CallMallocNursery | OpCode::CallMallocNurseryVarsize | OpCode::CallR
        )),
        "injected Newstr must become a malloc: {:?}",
        rewritten.iter().map(|op| op.opcode).collect::<Vec<_>>()
    );
    let inputs = nursery_new_inputs(frontend, 53);
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("Newstr without descr should inject and lower");
    validate_wasm(&bytes);
    let immediates = const_immediates(&bytes);
    assert!(
        immediates.contains(&0x22),
        "Newstr uses the array allocator"
    );
    let base_size = (2 * std::mem::size_of::<usize>() + 1) as i64;
    let len_offset = std::mem::size_of::<usize>() as i64;
    assert!(
        immediates.contains(&base_size) && immediates.contains(&len_offset),
        "the injected builtin str descr must supply the base size and length offset: {immediates:?}"
    );
}

fn call_malloc_nursery(result: u32, size: i64) -> Op {
    make_op(
        OpCode::CallMallocNursery,
        &[OpRef::const_int(size)],
        OpRef::ref_op(result),
    )
}

#[test]
fn call_malloc_nursery_uses_one_inline_bump() {
    let inputs = nursery_new_inputs(vec![call_malloc_nursery(1, 32), finish_int_arg0()], 53);
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert_eq!(nursery_top_compare_count(&bytes), 1);
    assert_eq!(
        memory_fill_count(&bytes),
        0,
        "CALL_MALLOC_NURSERY is the bump; rewrite emits zeros as later ops"
    );
}

/// rewrite.py `clear_gc_fields`: IncrementalMiniMark is
/// `malloc_zero_filled=false`, so a New with no leftover GC-pointer
/// fields emits no payload fill. The previous wasm lowering filled every
/// New; that is the skip this test pins.
#[test]
fn call_malloc_nursery_tid_store_is_one_fast_path_header_store() {
    let inputs = nursery_new_inputs(
        vec![
            call_malloc_nursery(1, 32),
            tid_gc_store(1, 53),
            finish_int_arg0(),
        ],
        53,
    );
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    let (slow, fast) = nursery_malloc_arm_stores(&bytes);
    // Fast path: bump (i32.store 0 of nursery_free) then one full-width
    // header store of the tid constant at the bump pointer.
    assert_eq!(fast.i32_off0, 1, "one nursery_free bump store: {fast:?}");
    assert_eq!(
        fast.i64_off0, 1,
        "exactly one header store between bump and join: {fast:?}"
    );
    assert_eq!(
        fast.i64, 1,
        "fast path header store is full-width: {fast:?}"
    );
    assert_eq!(fast.i64_32 + fast.i64_16, 0, "{fast:?}");
    // Slow path: the rewriter's tid store runs as-is, one native-word store.
    assert_eq!(
        slow.i64_off0, 1,
        "slow path keeps the one tid store: {slow:?}"
    );
    assert_eq!(slow.i64_32 + slow.i64_16, 0, "{slow:?}");
}

#[test]
fn call_malloc_nursery_without_tid_store_still_zeros_the_header() {
    let inputs = nursery_new_inputs(vec![call_malloc_nursery(1, 32), finish_int_arg0()], 53);
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    let (slow, fast) = nursery_malloc_arm_stores(&bytes);
    assert_eq!(fast.i32_off0, 1, "one nursery_free bump store: {fast:?}");
    assert_eq!(
        fast.i64_off0, 1,
        "fast path still zeros the header word: {fast:?}"
    );
    assert_eq!(fast.i64, 1, "{fast:?}");
    assert_eq!(
        slow.i64_32 + slow.i64_16,
        0,
        "slow path has no tid store to emit: {slow:?}"
    );
    assert_eq!(
        slow.i64_off0, 0,
        "slow path does not write the header: {slow:?}"
    );
}

#[test]
fn inline_nursery_new_does_not_fill_payload() {
    let inputs = nursery_new_inputs(vec![plain_new(1, 53), finish_int_arg0()], 53);
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert_eq!(
        memory_fill_count(&bytes),
        0,
        "inline New must not memory.fill; rewrite only NULLs leftover GC fields"
    );
}

fn vtable_new(result: u32, type_id: u32) -> Op {
    use majit_ir::descr::SimpleSizeDescr;
    use std::sync::Arc;
    let descr = SimpleSizeDescr::with_vtable(0, 24, type_id, 0x100);
    descr.set_non_moving(false);
    let op = make_op(OpCode::NewWithVtable, &[], OpRef::ref_op(result));
    op.setdescr(Arc::new(descr));
    op
}

#[test]
fn inline_nursery_new_with_vtable_skips_payload_fill() {
    // malloc_cond does not zero the payload. A NewWithVtable whose descr
    // has no leftover gc Refs (W_IntObject: only w_class, stamped here)
    // must not memory.fill — fannkuch does eight of these at JUMP.
    let inputs = nursery_new_inputs(vec![vtable_new(1, 53), finish_int_arg0()], 53);
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    let mut fills = 0;
    count_operators(&bytes, |op| {
        if matches!(op, wasmparser::Operator::MemoryFill { .. }) {
            fills += 1;
        }
    });
    assert_eq!(
        fills, 0,
        "NewWithVtable with no leftover gc Refs must not payload-fill (fills={fills})"
    );
}

#[test]
fn call_malloc_nursery_and_ptr_increment_share_one_bump() {
    let incr = make_op(
        OpCode::NurseryPtrIncrement,
        &[OpRef::ref_op(1), OpRef::const_int(32)],
        OpRef::ref_op(2),
    );
    let inputs = nursery_new_inputs(
        vec![call_malloc_nursery(1, 72), incr, finish_int_arg0()],
        53,
    );
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert_eq!(
        nursery_top_compare_count(&bytes),
        1,
        "rewrite already combined the batch; increment is pointer math"
    );
    let mut adds = 0;
    count_operators(&bytes, |op| {
        if matches!(op, wasmparser::Operator::I64Add) {
            adds += 1;
        }
    });
    assert!(adds >= 1, "NURSERY_PTR_INCREMENT is an i64 add");
}

#[test]
fn call_malloc_nursery_variants_lower() {
    use majit_ir::descr::SimpleArrayDescr;
    use std::sync::Arc;

    let headerless = make_op(
        OpCode::CallMallocNurseryHeaderless,
        &[OpRef::const_int(32)],
        OpRef::ref_op(1),
    );
    let inputs = nursery_new_inputs(vec![headerless, finish_int_arg0()], 53);
    let (bytes, _, _, _) = codegen::build_wasm_module(&inputs).expect("headerless should lower");
    validate_wasm(&bytes);
    assert_eq!(nursery_top_compare_count(&bytes), 1);
    assert!(const_immediates(&bytes).contains(&0x55));
    assert_eq!(
        memory_fill_count(&bytes),
        0,
        "CallMallocNurseryHeaderless fast path must not memory.fill"
    );

    let headerless_odd = make_op(
        OpCode::CallMallocNurseryHeaderless,
        &[OpRef::const_int(20)],
        OpRef::ref_op(1),
    );
    let inputs = nursery_new_inputs(vec![headerless_odd, finish_int_arg0()], 53);
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&inputs).expect("unaligned headerless should lower");
    validate_wasm(&bytes);
    assert!(
        const_immediates(&bytes).contains(&24),
        "headerless bump must 8-align so the next header is aligned"
    );

    let frame = make_op(
        OpCode::CallMallocNurseryVarsizeFrame,
        &[OpRef::const_int(60)],
        OpRef::ref_op(1),
    );
    let inputs = nursery_new_inputs(vec![frame, finish_int_arg0()], 53);
    let (bytes, _, _, _) = codegen::build_wasm_module(&inputs).expect("varsize frame should lower");
    validate_wasm(&bytes);
    assert_eq!(nursery_top_compare_count(&bytes), 1);
    assert!(const_immediates(&bytes).contains(&0x11));
    assert!(
        const_immediates(&bytes).contains(&64),
        "odd jfi_frame_size must 8-align before the nursery bump"
    );
    assert_eq!(
        memory_fill_count(&bytes),
        0,
        "CallMallocNurseryVarsizeFrame leaves home clears to the CA arm"
    );

    let varsize = make_op(
        OpCode::CallMallocNurseryVarsize,
        &[
            OpRef::const_int(0),
            OpRef::const_int(8),
            OpRef::const_int(4),
        ],
        OpRef::ref_op(1),
    );
    varsize.setdescr(Arc::new(SimpleArrayDescr::new(1, 16, 8, 53, Type::Int)));
    let inputs = nursery_new_inputs(vec![varsize, finish_int_arg0()], 53);
    let (bytes, _, _, _) = codegen::build_wasm_module(&inputs).expect("varsize should lower");
    validate_wasm(&bytes);
    assert_eq!(nursery_top_compare_count(&bytes), 1);
    assert!(
        const_immediates(&bytes).contains(&0x22),
        "oversize / full-nursery arm keeps the array helper"
    );
}

/// assembler.py `malloc_cond_varsize`: the slow path `push_gcmap`s before
/// calling the array helper, and a non-inlined `CALL_R` to the helper
/// pushes it too (callbuilder.py `push_gcmap`). A Ref live across the
/// allocation is then in `jf_gcmap` whenever the helper collects.
#[test]
fn call_malloc_nursery_varsize_helper_publishes_the_site_gcmap() {
    use majit_ir::descr::SimpleArrayDescr;
    use std::sync::Arc;

    let varsize = make_op(
        OpCode::CallMallocNurseryVarsize,
        &[
            OpRef::const_int(0),
            OpRef::const_int(8),
            OpRef::input_arg_int(0),
        ],
        OpRef::ref_op(2),
    );
    varsize.setdescr(Arc::new(SimpleArrayDescr::new(1, 16, 8, 53, Type::Int)));
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_ref(1))]);
    finish.setfailargs(smallvec![rb(OpRef::input_arg_ref(1))]);
    for inlined in [true, false] {
        let mut inputs = nursery_new_inputs(vec![varsize.clone(), finish.clone()], 53);
        if !inlined {
            inputs.nursery = None;
        }
        inputs.inputargs = vec![
            InputArg::from_type_rc(Type::Int, 0),
            InputArg::from_type_rc(Type::Ref, 1),
        ];
        inputs.frame = codegen::FrameGeometry::compact(
            codegen::frame_value_slots(&inputs.inputargs, &inputs.ops),
            codegen::count_ref_homes(&inputs.inputargs, &inputs.ops),
            0,
        );
        inputs.ca.compute_home_gcmap = true;
        inputs.ca.ca_reload_fn_ptr = 1;
        let (bytes, _, _, _) =
            codegen::build_wasm_module(&inputs).expect("varsize with a live Ref should lower");
        validate_wasm(&bytes);
        let gcmap_ofs = majit_backend::jitframe::JF_GCMAP_OFS as u64;
        let mut helper_calls = 0usize;
        for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
            let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() else {
                continue;
            };
            let mut published = false;
            let mut helper_pending = false;
            let mut last_const = None;
            for op in body.get_operators_reader().unwrap() {
                match op.unwrap() {
                    wasmparser::Operator::I32Const { value } => {
                        helper_pending |= value == 0x22;
                        last_const = Some(value as i64);
                    }
                    wasmparser::Operator::I64Const { value } => last_const = Some(value),
                    wasmparser::Operator::I32Store { memarg }
                    | wasmparser::Operator::I64Store { memarg }
                        if memarg.offset == gcmap_ofs =>
                    {
                        published = last_const.is_some_and(|v| v != 0);
                        last_const = None;
                    }
                    wasmparser::Operator::CallIndirect { .. } => {
                        if helper_pending {
                            helper_calls += 1;
                            assert!(
                                published,
                                "array helper runs with jf_gcmap unset (inlined={inlined})"
                            );
                        }
                        helper_pending = false;
                        last_const = None;
                    }
                    _ => last_const = None,
                }
            }
        }
        assert_eq!(helper_calls, if inlined { 2 } else { 1 });
    }
}

#[test]
fn inline_nursery_new_elides_the_barrier_like_gen_malloc_nursery() {
    use majit_ir::descr::{SimpleFieldDescr, SimpleSizeDescr};
    use std::sync::Arc;

    const WB_TARGET: i64 = 0x4a11;
    let size_descr = SimpleSizeDescr::new(0, 16, 53);
    let new_op = make_op(OpCode::New, &[], OpRef::ref_op(1));
    new_op.setdescr(Arc::new(size_descr));
    let store = Op::new(
        OpCode::SetfieldGc,
        &[rb(OpRef::ref_op(1)), rb(OpRef::input_arg_ref(0))],
    );
    store.setdescr(Arc::new(SimpleFieldDescr::new(0, 8, 8, Type::Ref, false)));
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(2))]);
    finish.setfailargs(smallvec![rb(OpRef::input_arg_int(2))]);

    let mut plain_tids = std::collections::HashSet::new();
    plain_tids.insert(53);
    let make_inputs = |nursery: Option<codegen::NurseryAllocParams>| codegen::ModuleBuildInputs {
        inputargs: vec![
            InputArg::from_type_rc(Type::Ref, 0),
            InputArg::from_type_rc(Type::Int, 2),
        ],
        ops: vec![new_op.clone(), store.clone(), finish.clone()],
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers {
            new_fn_ptr: 0x11,
            ..codegen::AllocHelpers::default()
        },
        wb: codegen::WriteBarrierHelpers::for_current_gc(WB_TARGET, 0),
        nursery,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::compact(5, 2, 0),
        ca: codegen::CaParams::default(),
    };
    let nursery = Some(codegen::NurseryAllocParams {
        free_addr: 0x1000,
        top_addr: 0x1004,
        large_threshold: 4096,
        plain_tids,
    });
    let (bytes, _, _, _) = codegen::build_wasm_module(&rewrite_module_inputs(make_inputs(nursery)))
        .expect("wasm codegen should succeed");
    validate_wasm(&bytes);
    assert_eq!(
        direct_write_barrier_call_count(&bytes, WB_TARGET as i32),
        0,
        "gen_malloc_nursery remembers the new object on both bump and overflow"
    );
    let (bytes, _, _, _) =
        codegen::build_wasm_module(&rewrite_module_inputs(make_inputs(None))).unwrap();
    assert_eq!(
        direct_write_barrier_call_count(&bytes, WB_TARGET as i32),
        0,
        "gen_malloc_fixedsize remembers a young fixed-size malloc"
    );
}
/// `emit_force_arm` publishes a guard's fail arguments while the bracketed
/// call is still on the stack. A constant among them has no home and no
/// local; it is published as its literal, like `emit_guard_fail_args_spill`
/// spills it.
#[test]
fn a_constant_fail_arg_in_a_force_bracket_is_published_as_a_literal() {
    let call = make_op(
        OpCode::CallMayForceI,
        &[OpRef::const_int(42)],
        OpRef::int_op(1),
    );
    call.setdescr(majit_ir::descr::make_call_descr_full(
        0,
        vec![],
        Type::Int,
        false,
        8,
        EffectInfo::default(),
    ));
    let guard = Op::new(OpCode::GuardNotForced, &[]);
    guard.setfailargs(smallvec![
        rb(OpRef::input_arg_ref(0)),
        rb(OpRef::const_int(1)),
        rb(OpRef::const_ptr(majit_ir::GcRef::NULL)),
    ]);
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_ref(0))]);
    finish.setfailargs(smallvec![rb(OpRef::input_arg_ref(0))]);
    let bytes = build_module_with_write_barrier_target(
        &[InputArg::from_type_rc(Type::Ref, 0)],
        &[call, guard, finish],
        127,
    );
    validate_wasm(&bytes);
}

/// A plain `CallN` does not publish `jf_force_descr`, even when
/// `GuardNotForced` follows it. `CallMayForce*` does. Dynasm's
/// `_store_force_index_if_next_guard` is gated on the call opcode the same
/// way; arming every such pair makes the guard fail for a callee that only
/// reads the frame.
#[test]
fn plain_call_before_guard_not_forced_does_not_arm_the_force_descr() {
    fn module(opcode: OpCode) -> Vec<u8> {
        let call = make_op(opcode, &[OpRef::const_int(42)], OpRef::int_op(1));
        call.setdescr(majit_ir::descr::make_call_descr_full(
            0,
            vec![],
            Type::Int,
            false,
            8,
            EffectInfo::default(),
        ));
        let guard = Op::new(OpCode::GuardNotForced, &[]);
        guard.setfailargs(smallvec![rb(OpRef::input_arg_ref(0))]);
        let finish = Op::new(OpCode::Finish, &[rb(OpRef::input_arg_ref(0))]);
        finish.setfailargs(smallvec![rb(OpRef::input_arg_ref(0))]);
        build_module_with_write_barrier_target(
            &[InputArg::from_type_rc(Type::Ref, 0)],
            &[call, guard, finish],
            127,
        )
    }
    let stores_force_descr = |bytes: &[u8]| {
        let mut pat = vec![0x36u8, 0x02];
        let mut offset = majit_backend::jitframe::JF_FORCE_DESCR_OFS as u64;
        loop {
            let mut byte = (offset & 0x7f) as u8;
            offset >>= 7;
            if offset != 0 {
                byte |= 0x80;
            }
            pat.push(byte);
            if offset == 0 {
                break;
            }
        }
        bytes
            .windows(pat.len())
            .any(|window| window == pat.as_slice())
    };
    let plain = module(OpCode::CallN);
    let may_force = module(OpCode::CallMayForceI);
    validate_wasm(&plain);
    validate_wasm(&may_force);
    assert!(
        !stores_force_descr(&plain),
        "CallN must not store jf_force_descr"
    );
    assert!(
        stores_force_descr(&may_force),
        "CallMayForce must store jf_force_descr before GuardNotForced"
    );
}

/// Post-rewrite `CondCallGcWb(obj)` + `GcStore` emits the same barrier helper
/// sequence a non-elided `SetfieldGc` would; `GcStore` itself stays
/// barrier-free.
#[test]
fn cond_call_gc_wb_emits_the_field_barrier() {
    const WB_TARGET: i64 = 0x4a11;
    let inputargs = [
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Ref, 1),
    ];
    let wb = make_op(
        OpCode::CondCallGcWb,
        &[OpRef::input_arg_ref(0)],
        OpRef::NONE,
    );
    let store = make_op(
        OpCode::GcStore,
        &[
            OpRef::input_arg_ref(0),
            OpRef::const_int(8),
            OpRef::input_arg_ref(1),
            OpRef::const_int(4),
        ],
        OpRef::NONE,
    );
    let finish = Op::new(OpCode::Finish, &[]);
    let with_wb = build_module_with_write_barrier_target(
        &inputargs,
        &[wb, store.clone(), finish.clone()],
        WB_TARGET,
    );
    let without_wb =
        build_module_with_write_barrier_target(&inputargs, &[store, finish], WB_TARGET);
    validate_wasm(&with_wb);
    validate_wasm(&without_wb);
    assert_eq!(
        direct_write_barrier_call_count(&without_wb, WB_TARGET as i32),
        0,
        "GcStore must stay barrier-free"
    );
    assert_eq!(
        direct_write_barrier_call_count(&with_wb, WB_TARGET as i32),
        1,
        "CondCallGcWb must emit the field write-barrier helper"
    );
}

/// Post-rewrite `CondCallGcWbArray(obj, index)` + `GcStoreIndexed` emits the
/// array barrier (`rewrite.py gen_write_barrier_array`), including card
/// marking when the collector has cards.
#[test]
fn cond_call_gc_wb_array_emits_the_array_barrier() {
    let inputargs = [
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Int, 1),
        InputArg::from_type_rc(Type::Ref, 2),
    ];
    let wb = make_op(
        OpCode::CondCallGcWbArray,
        &[OpRef::input_arg_ref(0), OpRef::input_arg_int(1)],
        OpRef::NONE,
    );
    let store = make_op(
        OpCode::GcStoreIndexed,
        &[
            OpRef::input_arg_ref(0),
            OpRef::input_arg_int(1),
            OpRef::input_arg_ref(2),
            OpRef::const_int(4),
            OpRef::const_int(16),
            OpRef::const_int(4),
        ],
        OpRef::NONE,
    );
    let finish = Op::new(OpCode::Finish, &[]);
    let wb_helpers = codegen::WriteBarrierHelpers::for_current_gc(WB_FIELD, WB_ARRAY);
    let with_wb = build_module_with_barrier_helpers(
        &inputargs,
        &[wb, store.clone(), finish.clone()],
        wb_helpers,
    );
    let without_wb = build_module_with_barrier_helpers(&inputargs, &[store, finish], wb_helpers);
    validate_wasm(&with_wb);
    validate_wasm(&without_wb);
    assert_eq!(
        direct_write_barrier_call_count(&without_wb, WB_ARRAY as i32),
        0,
        "GcStoreIndexed must stay barrier-free"
    );
    assert_eq!(
        direct_write_barrier_call_count(&with_wb, WB_ARRAY as i32),
        1,
        "CondCallGcWbArray must emit the array write-barrier helper"
    );
    assert!(
        count_i32_store8(&with_wb) >= 1,
        "CondCallGcWbArray must mark a card byte when the collector has cards"
    );
}

fn count_ops(bytes: &[u8], pred: impl Fn(&wasmparser::Operator<'_>) -> bool) -> usize {
    let mut n = 0;
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut operators = body.get_operators_reader().unwrap();
            while !operators.eof() {
                let op = operators.read().unwrap();
                if pred(&op) {
                    n += 1;
                }
            }
        }
    }
    n
}

fn call_i_two_ints() -> (Vec<InputArgRc>, Vec<Op>) {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Int, 1),
    ];
    let call = {
        let op = Op::new(
            OpCode::CallI,
            &[
                rb(OpRef::const_int(42)),
                rb(OpRef::input_arg_int(0)),
                rb(OpRef::input_arg_int(1)),
            ],
        );
        op.pos().set(OpRef::int_op(2));
        op.setdescr(majit_ir::descr::make_call_descr(
            vec![Type::Int, Type::Int],
            Type::Int,
            EffectInfo::default(),
        ));
        op
    };
    let ops = vec![call, Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))])];
    (inputargs, ops)
}

/// Host codegen tests have no function table. `encoded` stands in for the
/// type `jit_func_sig` would read from the guest table.
fn with_table_sig(addr: i64, encoded: Option<i64>, body: impl FnOnce()) {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            majit_backend_wasm::clear_test_residual_target_sigs();
        }
    }
    let _restore = Restore;
    majit_backend_wasm::clear_test_residual_target_sigs();
    if let Some(encoded) = encoded {
        majit_backend_wasm::set_test_residual_target_sig(addr, encoded);
    }
    body();
}

/// `descr.py` `create_call_stub`: the callee's FUNC is the calldescr's.
/// An i32-narrowed table type is a missing `word_publish` /
/// `residual_word_addr` on that target, not a second wasm ABI. Encoding 0
/// (no table type) is the legitimate `jit_call_host` case: multi-value /
/// sret, or an import outside the guest table.
#[test]
fn i32_table_type_is_not_the_calldescr_word() {
    use majit_backend_wasm::{FuncSigVal, WasmSig};
    let descr = WasmSig {
        params: vec![FuncSigVal::I64, FuncSigVal::I64],
        result: Some(FuncSigVal::I64),
    };
    let twin = WasmSig {
        params: vec![FuncSigVal::I32, FuncSigVal::I32],
        result: Some(FuncSigVal::I32),
    };
    assert_eq!(descr, descr);
    assert_ne!(twin, descr);
}

/// `descr.py CallDescr.create_call_stub` / `callbuilder.py load_result`:
/// a Ref residual is one word. wasm codegen emits `(i64…) -> i64`; a
/// two-word `(i64, i64)` result would be a different `call_indirect` type.
fn call_r_two_refs() -> (Vec<InputArgRc>, Vec<Op>) {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Ref, 0),
        InputArg::from_type_rc(Type::Ref, 1),
    ];
    let call = {
        let op = Op::new(
            OpCode::CallR,
            &[
                rb(OpRef::const_int(42)),
                rb(OpRef::input_arg_ref(0)),
                rb(OpRef::input_arg_ref(1)),
            ],
        );
        op.pos().set(OpRef::ref_op(2));
        op.setdescr(majit_ir::descr::make_call_descr(
            vec![Type::Ref, Type::Ref],
            Type::Ref,
            EffectInfo::default(),
        ));
        op
    };
    let ops = vec![call, Op::new(OpCode::Finish, &[rb(OpRef::ref_op(2))])];
    (inputargs, ops)
}

#[test]
fn call_r_lowers_as_one_word_i64_result() {
    let encoded = majit_backend_wasm::encode_func_sig(
        &[
            majit_backend_wasm::FuncSigVal::I64,
            majit_backend_wasm::FuncSigVal::I64,
        ],
        Some(majit_backend_wasm::FuncSigVal::I64),
    );
    with_table_sig(42, Some(encoded), || {
        let (inputargs, ops) = call_r_two_refs();
        let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
        validate_wasm(&bytes);
        assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
        let (indirect_calls, _) = indirect_call_types_and_drop_count(&bytes);
        assert_eq!(indirect_calls.len(), 1);
        let (params, results) = function_type(&bytes, indirect_calls[0].0 as usize);
        assert_eq!(
            params,
            vec![wasmparser::ValType::I64, wasmparser::ValType::I64]
        );
        assert_eq!(results, vec![wasmparser::ValType::I64]);
    });
}

#[test]
fn test_oracle_i64_call_lowers_in_module_without_vouch() {
    let encoded = majit_backend_wasm::encode_func_sig(
        &[
            majit_backend_wasm::FuncSigVal::I64,
            majit_backend_wasm::FuncSigVal::I64,
        ],
        Some(majit_backend_wasm::FuncSigVal::I64),
    );
    with_table_sig(42, Some(encoded), || {
        let (inputargs, ops) = call_i_two_ints();
        let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
        validate_wasm(&bytes);
        assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
        let (indirect_calls, _) = indirect_call_types_and_drop_count(&bytes);
        assert_eq!(indirect_calls.len(), 1);
    });
}

#[test]
fn test_oracle_i32_call_keeps_trampoline() {
    let encoded = majit_backend_wasm::encode_func_sig(
        &[
            majit_backend_wasm::FuncSigVal::I32,
            majit_backend_wasm::FuncSigVal::I32,
        ],
        Some(majit_backend_wasm::FuncSigVal::I32),
    );
    with_table_sig(42, Some(encoded), || {
        let (inputargs, ops) = call_i_two_ints();
        let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
        validate_wasm(&bytes);
        assert_eq!(import_func_type(&bytes, "jit_call_compact"), Some(1));
        let (indirect_calls, _) = indirect_call_types_and_drop_count(&bytes);
        assert!(indirect_calls.is_empty());
    });
}

#[test]
fn test_oracle_f32_int_bits_call_indirect() {
    let encoded = majit_backend_wasm::encode_func_sig(
        &[majit_backend_wasm::FuncSigVal::F32],
        Some(majit_backend_wasm::FuncSigVal::I64),
    );
    with_table_sig(42, Some(encoded), || {
        let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
        let call = {
            let op = Op::new(
                OpCode::CallI,
                &[rb(OpRef::const_int(42)), rb(OpRef::input_arg_int(0))],
            );
            op.pos().set(OpRef::int_op(1));
            op.setdescr(majit_ir::descr::make_call_descr(
                vec![Type::Int],
                Type::Int,
                EffectInfo::default(),
            ));
            op
        };
        let ops = vec![call, Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))])];
        let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
        validate_wasm(&bytes);
        assert_eq!(import_func_type(&bytes, "jit_call_compact"), Some(1));
        let (indirect_calls, _) = indirect_call_types_and_drop_count(&bytes);
        assert!(indirect_calls.is_empty());
    });
}

#[test]
fn test_oracle_singlefloat_result_returns_f32_bits() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let call = {
        let op = Op::new(
            OpCode::CallI,
            &[rb(OpRef::const_int(42)), rb(OpRef::input_arg_int(0))],
        );
        op.pos().set(OpRef::int_op(1));
        op.setdescr(majit_ir::descr::make_call_descr_full_with_result_class(
            0,
            vec![Type::Int],
            Type::Int,
            'S',
            false,
            4,
            EffectInfo::default(),
        ));
        op
    };
    let ops = vec![call, Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))])];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
    let (indirect_calls, _) = indirect_call_types_and_drop_count(&bytes);
    assert_eq!(indirect_calls.len(), 1);
    assert_eq!(
        function_type(&bytes, indirect_calls[0].0 as usize),
        (
            vec![wasmparser::ValType::I64],
            vec![wasmparser::ValType::F32]
        )
    );
    assert_eq!(
        count_ops(&bytes, |op| matches!(
            op,
            wasmparser::Operator::I32ReinterpretF32
        )),
        1
    );
}

#[test]
fn test_oracle_singlefloat_arg_passes_f32_bits() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let call = {
        let op = Op::new(
            OpCode::CallI,
            &[rb(OpRef::const_int(42)), rb(OpRef::input_arg_int(0))],
        );
        op.pos().set(OpRef::int_op(1));
        op.setdescr(majit_ir::descr::make_call_descr_full_with_classes(
            0,
            "S".to_string(),
            vec![Type::Int],
            Type::Int,
            'i',
            true,
            8,
            EffectInfo::default(),
        ));
        op
    };
    let ops = vec![call, Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))])];
    let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
    validate_wasm(&bytes);
    assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
    let (indirect_calls, _) = indirect_call_types_and_drop_count(&bytes);
    assert_eq!(indirect_calls.len(), 1);
    assert_eq!(
        function_type(&bytes, indirect_calls[0].0 as usize),
        (
            vec![wasmparser::ValType::F32],
            vec![wasmparser::ValType::I64]
        )
    );
    assert_eq!(
        count_ops(&bytes, |op| matches!(
            op,
            wasmparser::Operator::F32ReinterpretI32
        )),
        1
    );
}

#[test]
fn test_oracle_f32_float_keeps_trampoline() {
    let encoded = majit_backend_wasm::encode_func_sig(
        &[majit_backend_wasm::FuncSigVal::F32],
        Some(majit_backend_wasm::FuncSigVal::F32),
    );
    with_table_sig(42, Some(encoded), || {
        let inputargs = vec![InputArg::from_type_rc(Type::Float, 0)];
        let call = {
            let op = Op::new(
                OpCode::CallF,
                &[rb(OpRef::const_int(42)), rb(OpRef::input_arg_float(0))],
            );
            op.pos().set(OpRef::float_op(1));
            op.setdescr(majit_ir::descr::make_call_descr(
                vec![Type::Float],
                Type::Float,
                EffectInfo::default(),
            ));
            op
        };
        let ops = vec![call, Op::new(OpCode::Finish, &[rb(OpRef::float_op(1))])];
        let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
        validate_wasm(&bytes);
        assert_eq!(import_func_type(&bytes, "jit_call_compact"), Some(1));
        let (indirect_calls, _) = indirect_call_types_and_drop_count(&bytes);
        assert!(indirect_calls.is_empty());
    });
}

#[test]
fn test_oracle_f64_param_on_int_descr_keeps_trampoline() {
    let encoded = majit_backend_wasm::encode_func_sig(
        &[majit_backend_wasm::FuncSigVal::F64],
        Some(majit_backend_wasm::FuncSigVal::I64),
    );
    with_table_sig(42, Some(encoded), || {
        let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
        let call = {
            let op = Op::new(
                OpCode::CallI,
                &[rb(OpRef::const_int(42)), rb(OpRef::input_arg_int(0))],
            );
            op.pos().set(OpRef::int_op(1));
            op.setdescr(majit_ir::descr::make_call_descr(
                vec![Type::Int],
                Type::Int,
                EffectInfo::default(),
            ));
            op
        };
        let ops = vec![call, Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))])];
        let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
        validate_wasm(&bytes);
        assert_eq!(import_func_type(&bytes, "jit_call_compact"), Some(1));
        let (indirect_calls, _) = indirect_call_types_and_drop_count(&bytes);
        assert!(indirect_calls.is_empty());
        assert_eq!(
            count_ops(&bytes, |op| matches!(
                op,
                wasmparser::Operator::I64ExtendI32S
            )),
            0
        );
    });
}

#[test]
fn test_trampolined_narrow_signed_int_sign_extends() {
    let encoded = majit_backend_wasm::encode_func_sig(
        &[majit_backend_wasm::FuncSigVal::F64],
        Some(majit_backend_wasm::FuncSigVal::I32),
    );
    with_table_sig(42, Some(encoded), || {
        let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
        let call = {
            let op = Op::new(
                OpCode::CallI,
                &[rb(OpRef::const_int(42)), rb(OpRef::input_arg_int(0))],
            );
            op.pos().set(OpRef::int_op(1));
            op.setdescr(std::sync::Arc::new(majit_ir::descr::SimpleCallDescr::new(
                0,
                vec![Type::Int],
                Type::Int,
                true,
                4,
                EffectInfo::default(),
            )));
            op
        };
        let ops = vec![call, Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))])];
        let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
        validate_wasm(&bytes);
        assert_eq!(import_func_type(&bytes, "jit_call_compact"), Some(1));
        assert_eq!(
            count_ops(&bytes, |op| matches!(
                op,
                wasmparser::Operator::I64ExtendI32S
            )),
            1
        );
    });
}

#[test]
fn test_cond_call_value_mismatch_uses_trampoline() {
    let encoded = majit_backend_wasm::encode_func_sig(
        &[majit_backend_wasm::FuncSigVal::F64],
        Some(majit_backend_wasm::FuncSigVal::I32),
    );
    with_table_sig(42, Some(encoded), || {
        let inputargs = vec![
            InputArg::from_type_rc(Type::Int, 0),
            InputArg::from_type_rc(Type::Int, 1),
        ];
        let call = {
            let op = Op::new(
                OpCode::CondCallValueI,
                &[
                    rb(OpRef::input_arg_int(0)),
                    rb(OpRef::const_int(42)),
                    rb(OpRef::input_arg_int(1)),
                ],
            );
            op.pos().set(OpRef::int_op(2));
            op.setdescr(std::sync::Arc::new(majit_ir::descr::SimpleCallDescr::new(
                0,
                vec![Type::Int],
                Type::Int,
                true,
                4,
                EffectInfo::default(),
            )));
            op
        };
        let ops = vec![call, Op::new(OpCode::Finish, &[rb(OpRef::int_op(2))])];
        let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
        validate_wasm(&bytes);
        assert_eq!(import_func_type(&bytes, "jit_call_compact"), Some(1));
        assert_eq!(
            count_ops(&bytes, |op| matches!(
                op,
                wasmparser::Operator::I64ExtendI32S
            )),
            1
        );
    });
}

#[test]
fn residual_trampoline_accepts_a_full_call_area() {
    // More than MAX_CALL_ARGS cannot be published, so the host compiler emits
    // the descr type directly. A full area still takes the trampoline when
    // the published type is not that descr.
    let too_wide = vec![majit_backend_wasm::FuncSigVal::F64; codegen::MAX_CALL_ARGS + 1];
    assert_eq!(
        majit_backend_wasm::encode_func_sig(&too_wide, Some(majit_backend_wasm::FuncSigVal::I64)),
        0
    );
    let wide = vec![majit_backend_wasm::FuncSigVal::F64; codegen::MAX_CALL_ARGS];
    let encoded =
        majit_backend_wasm::encode_func_sig(&wide, Some(majit_backend_wasm::FuncSigVal::I64));
    assert_ne!(encoded, 0);
    with_table_sig(1, Some(encoded), || {
        let mut args = vec![rb(OpRef::const_int(1))];
        let mut types = Vec::new();
        for i in 0..codegen::MAX_CALL_ARGS {
            args.push(rb(OpRef::const_int(i as i64)));
            types.push(Type::Int);
        }
        let call = Op::new(OpCode::CallI, &args);
        call.pos().set(OpRef::int_op(1));
        call.setdescr(std::sync::Arc::new(majit_ir::descr::SimpleCallDescr::new(
            0,
            types,
            Type::Int,
            true,
            8,
            EffectInfo::default(),
        )));
        let ops = vec![call, Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))])];
        let (bytes, _) = build_module_default(&[], &ops, &indexmap::IndexMap::new());
        validate_wasm(&bytes);
        assert_eq!(import_func_type(&bytes, "jit_call_compact"), Some(1));
    });
}

#[test]
fn test_oracle_unknown_follows_descr_not_vouch_list() {
    with_table_sig(42, None, || {
        let (inputargs, ops) = call_i_two_ints();
        let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
        validate_wasm(&bytes);
        assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
    });
}

#[test]
fn test_calln_void_direct_when_table_matches_descr() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let call_n =
        |result_size: usize| void_call(vec![Type::Int], &[OpRef::input_arg_int(0)], result_size);
    let finish = || Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))]);

    let true_void =
        majit_backend_wasm::encode_func_sig(&[majit_backend_wasm::FuncSigVal::I64], None);
    let void_word = majit_backend_wasm::encode_func_sig(
        &[majit_backend_wasm::FuncSigVal::I64],
        Some(majit_backend_wasm::FuncSigVal::I64),
    );

    // result_size 0 + void table: direct void call, nothing to drop.
    with_table_sig(42, Some(true_void), || {
        let ops = vec![call_n(0), finish()];
        let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
        validate_wasm(&bytes);
        assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
        let (indirect_calls, drops) = indirect_call_types_and_drop_count(&bytes);
        assert_eq!(indirect_calls.len(), 1);
        assert_eq!(
            function_type(&bytes, indirect_calls[0].0 as usize),
            (vec![wasmparser::ValType::I64], vec![])
        );
        assert_eq!(drops, 0);
    });

    // result_size 8 + i64 table: direct word call, dummy result dropped.
    with_table_sig(42, Some(void_word), || {
        let ops = vec![call_n(8), finish()];
        let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
        validate_wasm(&bytes);
        assert_eq!(import_func_type(&bytes, "jit_call_compact"), None);
        let (indirect_calls, drops) = indirect_call_types_and_drop_count(&bytes);
        assert_eq!(indirect_calls.len(), 1);
        assert_eq!(
            function_type(&bytes, indirect_calls[0].0 as usize),
            (
                vec![wasmparser::ValType::I64],
                vec![wasmparser::ValType::I64]
            )
        );
        assert_eq!(drops, 1);
    });

    // Mismatched table types stay on jit_call. The descr alone names the type.
    with_table_sig(42, Some(true_void), || {
        let ops = vec![call_n(8), finish()];
        let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
        validate_wasm(&bytes);
        assert_eq!(import_func_type(&bytes, "jit_call_compact"), Some(1));
        let (indirect_calls, _) = indirect_call_types_and_drop_count(&bytes);
        assert!(indirect_calls.is_empty());
    });
    with_table_sig(42, Some(void_word), || {
        let ops = vec![call_n(0), finish()];
        let (bytes, _) = build_module_default(&inputargs, &ops, &indexmap::IndexMap::new());
        validate_wasm(&bytes);
        assert_eq!(import_func_type(&bytes, "jit_call_compact"), Some(1));
        let (indirect_calls, _) = indirect_call_types_and_drop_count(&bytes);
        assert!(indirect_calls.is_empty());
    });
}

#[test]
fn test_func_sig_codec_golden() {
    let known = 1_i64 << 63;
    assert_eq!(majit_backend_wasm::encode_func_sig(&[], None), known);
    assert_eq!(
        majit_backend_wasm::encode_func_sig(
            &[
                majit_backend_wasm::FuncSigVal::I64,
                majit_backend_wasm::FuncSigVal::I64
            ],
            Some(majit_backend_wasm::FuncSigVal::I64)
        ),
        known | 2 | (2 << 5) | (1 << 8) | (1 << 10)
    );
    assert_eq!(majit_backend_wasm::decode_func_sig(0), None);
}

/// A bridge whose value slots and Ref homes exceed the source geometry keeps
/// that geometry, appends the overflow, and `_check_frame_depth` reallocates
/// before the bridge reads or writes the tail. The host reads the exit words
/// from the frame the helper returned.
#[test]
#[ignore = "runtime integration test: executes a generated module through the indirect-call realloc helper"]
fn bridge_with_a_larger_frame_reallocates_and_returns_its_exit_values() {
    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Ref, 1),
        InputArg::from_type_rc(Type::Ref, 2),
    ];
    let add = make_op(
        OpCode::IntAdd,
        &[OpRef::input_arg_int(0), OpRef::const_int(1)],
        OpRef::int_op(3),
    );
    let guard = make_guard(
        OpCode::GuardNotForced,
        &[],
        &[OpRef::input_arg_ref(1), OpRef::input_arg_ref(2)],
    );
    let finish = Op::new(
        OpCode::Finish,
        &[
            rb(OpRef::int_op(3)),
            rb(OpRef::input_arg_int(0)),
            rb(OpRef::const_int(7)),
            rb(OpRef::const_int(8)),
            rb(OpRef::const_int(99)),
        ],
    );
    finish.setfailargs(smallvec![
        rb(OpRef::int_op(3)),
        rb(OpRef::input_arg_int(0)),
        rb(OpRef::const_int(7)),
        rb(OpRef::const_int(8)),
        rb(OpRef::const_int(99)),
    ]);
    let ops = vec![add, guard, finish];
    let source = codegen::FrameGeometry::compact(4, 1, 0);
    let needed_values = codegen::frame_value_slots(&inputargs, &ops);
    let needed_homes = codegen::count_ref_homes(&inputargs, &ops);
    assert!(needed_values > source.value_slots);
    assert!(needed_homes > source.ordinary_home_slots());
    // The stored loop frame is already tailed; this bridge extends it again.
    let tailed = source.extend(source.value_slots + 1, source.ordinary_home_slots());
    let frame = tailed.extend(needed_values, needed_homes);
    assert!(frame.has_tail());
    assert_eq!(frame.tail_base, tailed.tail_base);
    assert_eq!(frame.spill_slot_ofs(4), tailed.spill_slot_ofs(4));
    let mut ca = codegen::CaParams::default();
    ca.realloc_fn_ptr = 1;
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.clone(),
        ops: ops.clone(),
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame,
        ca,
    };
    let (bytes, _, homes, _) =
        codegen::build_wasm_module(&inputs).expect("extended bridge compiles");
    assert!(homes > source.ordinary_home_slots());
    validate_wasm(&bytes);
    let map = codegen::build_home_gcmap(frame, homes, 0);
    let sign = std::mem::size_of::<isize>();
    let bits = std::mem::size_of::<usize>() * 8;
    let tail_home = frame.home_ofs(source.ordinary_home_slots() as u64) as usize;
    let index = tail_home / sign;
    assert_ne!(
        map[1 + index / bits] & (1usize << (index % bits)),
        0,
        "gcmap marks the tail Ref home"
    );

    let engine = Engine::default();
    let module = Module::new(&engine, &bytes).expect("module");
    let mut store = Store::new(&engine, ());
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).expect("memory");
    let source_bytes = source.frame_bytes as usize;
    let seen_gcmap = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let seen_gcmap_slot = std::sync::Arc::clone(&seen_gcmap);
    let realloc = wasmi::Func::wrap(
        &mut store,
        move |mut caller: wasmi::Caller<'_, ()>, items: i64, depth: i64| -> i64 {
            let old = items as usize;
            let gcmap_at = old - majit_backend::jitframe::FIRST_ITEM_OFFSET
                + majit_backend::jitframe::JF_GCMAP_OFS as usize;
            let mut word = [0u8; 4];
            memory
                .read(&caller, gcmap_at, &mut word)
                .expect("jf_gcmap before realloc");
            seen_gcmap_slot.store(
                u32::from_le_bytes(word),
                std::sync::atomic::Ordering::Relaxed,
            );
            let new_base = 64 * 1024usize;
            let mut buf = vec![0u8; source_bytes];
            memory.read(&caller, old, &mut buf).expect("copy old frame");
            memory
                .write(&mut caller, new_base, &buf)
                .expect("write new frame");
            let mut len = [0u8; 8];
            len[..4].copy_from_slice(&(depth as u32).to_le_bytes());
            memory
                .write(&mut caller, new_base - sign, &len[..sign.min(8)])
                .expect("write new length");
            new_base as i64
        },
    );
    let table = Table::new(
        &mut store,
        TableType::new(ValType::FuncRef, 2, None),
        Val::default(ValType::FuncRef),
    )
    .expect("table");
    table.set(&mut store, 1, Val::from(realloc)).expect("slot");
    let mut linker = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .define("env", "__indirect_function_table", table)
        .unwrap();
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("instantiate");
    let old_base = 4096usize;
    let mut len = [0u8; 8];
    len[..4].copy_from_slice(&(source.signed_item_count() as u32).to_le_bytes());
    memory
        .write(&mut store, old_base - sign, &len[..sign.min(8)])
        .unwrap();
    let write_i64 = |store: &mut Store<()>, offset: usize, value: i64| {
        memory.write(store, offset, &value.to_le_bytes()).unwrap();
    };
    write_i64(&mut store, old_base + codegen::FRAME_SLOT_BASE as usize, 41);
    write_i64(
        &mut store,
        old_base + codegen::FRAME_SLOT_BASE as usize + 8,
        0x1111,
    );
    write_i64(
        &mut store,
        old_base + codegen::FRAME_SLOT_BASE as usize + 16,
        0x2222,
    );
    let new_base = instance
        .get_typed_func::<i32, i32>(&store, "trace")
        .unwrap()
        .call(&mut store, old_base as i32)
        .expect("bridge runs") as usize;
    assert_ne!(new_base, old_base, "the short frame was reallocated");
    assert_ne!(
        seen_gcmap.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "jf_gcmap is stored before wasm_realloc_frame"
    );
    let read_i64 = |offset: usize| {
        let mut buf = [0u8; 8];
        memory.read(&store, offset, &mut buf).unwrap();
        i64::from_le_bytes(buf)
    };
    let tail_slot = frame.spill_slot_ofs(4) as usize;
    assert!(tail_slot >= source.frame_bytes as usize);
    assert_eq!(read_i64(new_base + tail_slot), 99);
    assert_eq!(read_i64(new_base + tail_home), 0x2222);
}

/// A geometry that was not extended still records compact fail locations:
/// `spill_slot_index` is the identity, and the guard's `fail_locs` stay
/// `Some(0), Some(1), ...`.
#[test]
fn geometry_without_a_tail_keeps_identity_fail_locations() {
    let frame = codegen::FrameGeometry::compact(8, 1, 0);
    assert!(!frame.has_tail());
    for i in 0..8u64 {
        assert_eq!(frame.spill_slot_index(i), i);
        assert_eq!(frame.spill_slot_ofs(i), codegen::FRAME_SLOT_BASE + i * 8);
        assert_eq!(frame.force_slot_ofs(i), frame.force_slot_base + i * 8);
    }
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let guard = make_guard(
        OpCode::GuardTrue,
        &[OpRef::input_arg_int(0)],
        &[
            OpRef::input_arg_int(0),
            OpRef::const_int(7),
            OpRef::const_int(9),
        ],
    );
    let (_bytes, guards) = build_module_with_frame(
        &inputargs,
        &[guard],
        &indexmap::IndexMap::new(),
        Some(0),
        &codegen::GuardGcTypeInfo::default(),
        frame,
    );
    assert_eq!(
        guards[0].fail_locs,
        vec![Some(0), Some(1), Some(2)],
        "a frame without a tail records the compact spill index"
    );
}

/// Fail args past the source prefix are spilled in the tail. The host reads
/// each one back through the deadframe, and the guard gcmap marks the Ref.
#[test]
#[ignore = "runtime integration test: executes a tail-frame guard exit and decodes it through DeadFrame"]
fn tail_fail_args_decode_through_the_deadframe() {
    use majit_backend::Backend;
    use majit_backend::jitframe::{FIRST_ITEM_OFFSET, JitFrame, alloc_off_gc_jitframe};
    use majit_backend_wasm::failguard::{WasmFailDescr, WasmFrameData};

    let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
    let guard = make_guard(
        OpCode::GuardFalse,
        &[OpRef::const_int(1)],
        &[
            OpRef::const_int(11),
            OpRef::const_int(22),
            OpRef::const_int(33),
            OpRef::const_int(44),
            OpRef::input_arg_ref(0),
            OpRef::const_int(55),
        ],
    );
    guard.set_fail_arg_types(vec![
        Type::Int,
        Type::Int,
        Type::Int,
        Type::Int,
        Type::Ref,
        Type::Int,
    ]);
    let source = codegen::FrameGeometry::compact(4, 1, 0);
    let needed_values = codegen::frame_value_slots(&inputargs, &[guard.clone()]);
    let needed_homes = codegen::count_ref_homes(&inputargs, &[guard.clone()]);
    assert!(needed_values > source.value_slots);
    // Compile the guard against a geometry that already has a tail.
    let tailed = source.extend(source.value_slots + 1, needed_homes);
    let frame = tailed.extend(needed_values, needed_homes);
    assert!(frame.has_tail());
    assert_eq!(frame.tail_base, tailed.tail_base);
    assert_eq!(frame.spill_slot_ofs(4), tailed.spill_slot_ofs(4));
    assert_eq!(frame.force_slot_ofs(4), tailed.force_slot_ofs(4));
    assert!(frame.spill_slot_ofs(4) >= source.frame_bytes as u64);
    assert!(frame.spill_slot_ofs(5) > frame.spill_slot_ofs(4));
    let mut ca = codegen::CaParams::default();
    ca.realloc_fn_ptr = 1;
    let inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.clone(),
        ops: vec![guard],
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame,
        ca,
    };
    let (bytes, guards, _, _) = codegen::build_wasm_module(&inputs).expect("tail guard compiles");
    let exit = &guards[0];
    assert!(exit.fail_locs[4].unwrap() >= source.prefix_value_slots);
    assert!(exit.fail_locs[5].unwrap() > exit.fail_locs[4].unwrap());
    let sign = std::mem::size_of::<isize>();
    let bits = std::mem::size_of::<usize>() * 8;
    let ref_slot = exit.fail_locs[4].unwrap();
    let ref_bit = (codegen::FRAME_SLOT_BASE as usize + ref_slot * 8) / sign;
    let map = exit.exit_gcmap_ptr as *const usize;
    assert!(!map.is_null(), "the Ref fail arg publishes a gcmap");
    let n = unsafe { *map };
    let words = unsafe { std::slice::from_raw_parts(map, 1 + n) };
    assert_ne!(
        words[1 + ref_bit / bits] & (1usize << (ref_bit % bits)),
        0,
        "gcmap marks the tail Ref fail arg"
    );
    validate_wasm(&bytes);

    let engine = Engine::default();
    let module = Module::new(&engine, &bytes).expect("module");
    let mut store = Store::new(&engine, ());
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).expect("memory");
    let source_bytes = source.frame_bytes as usize;
    let grown = frame.frame_bytes as usize;
    let realloc = wasmi::Func::wrap(
        &mut store,
        move |mut caller: wasmi::Caller<'_, ()>, items: i64, depth: i64| -> i64 {
            let old = items as usize;
            let new_base = 64 * 1024usize;
            let mut buf = vec![0u8; source_bytes.max(grown)];
            memory
                .read(&caller, old, &mut buf[..source_bytes])
                .expect("copy old frame");
            memory
                .write(&mut caller, new_base, &buf[..grown])
                .expect("write new frame");
            let mut len = [0u8; 8];
            len[..4].copy_from_slice(&(depth as u32).to_le_bytes());
            memory
                .write(&mut caller, new_base - sign, &len[..sign.min(8)])
                .expect("write new length");
            new_base as i64
        },
    );
    let table = Table::new(
        &mut store,
        TableType::new(ValType::FuncRef, 2, None),
        Val::default(ValType::FuncRef),
    )
    .expect("table");
    table.set(&mut store, 1, Val::from(realloc)).expect("slot");
    let mut linker = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .define("env", "__indirect_function_table", table)
        .unwrap();
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("instantiate");
    let old_base = 4096usize;
    let mut len = [0u8; 8];
    len[..4].copy_from_slice(&(source.signed_item_count() as u32).to_le_bytes());
    memory
        .write(&mut store, old_base - sign, &len[..sign.min(8)])
        .unwrap();
    let ref_bits = 0xABC0i64;
    memory
        .write(
            &mut store,
            old_base + codegen::FRAME_SLOT_BASE as usize,
            &ref_bits.to_le_bytes(),
        )
        .unwrap();
    let new_base = instance
        .get_typed_func::<i32, i32>(&store, "trace")
        .unwrap()
        .call(&mut store, old_base as i32)
        .expect("guard exit") as usize;
    assert_ne!(new_base, old_base);

    let depth = frame.signed_item_count();
    let jf = alloc_off_gc_jitframe(JitFrame::alloc_size(depth));
    assert!(!jf.is_null());
    let mut buf = vec![0u8; grown];
    memory.read(&store, new_base, &mut buf).unwrap();
    unsafe {
        JitFrame::init(jf, std::ptr::null(), depth);
        let items = (jf as *mut u8).add(FIRST_ITEM_OFFSET);
        std::ptr::copy_nonoverlapping(buf.as_ptr(), items, grown);
    }
    let descr = std::sync::Arc::new(WasmFailDescr {
        fail_index: exit.fail_index,
        trace_id: 0,
        fail_arg_types: exit.fail_arg_types.clone(),
        fail_locs: exit.fail_locs.clone(),
        is_finish: false,
        force_args_offset: frame.force_slot_base as u32,
        force_tail_base: frame.force_tail_base() as u32,
        force_prefix_slots: frame.prefix_value_slots as u32,
        value_tail_index: frame.value_tail_index() as u32,
        force_gcmap_ptr: exit.exit_gcmap_ptr,
        bridge_cell: 0,
        fail_arg_advanced: Vec::new(),
        trace_ref_homes: 0,
        trace_label_homes: 0,
        param_dispatch: false,
        bridge_slot: std::sync::atomic::AtomicU32::new(0),
        meta_descr: None,
    });
    let data = WasmFrameData::from_live_frame(jf, descr, false, false, None);
    let dead = majit_backend::DeadFrame::Boxed(data);
    let backend = majit_backend_wasm::WasmBackend::new();
    assert_eq!(backend.get_int_value(&dead, 0), 11);
    assert_eq!(backend.get_int_value(&dead, 1), 22);
    assert_eq!(backend.get_int_value(&dead, 2), 33);
    assert_eq!(backend.get_int_value(&dead, 3), 44);
    assert_eq!(backend.get_ref_value(&dead, 4).0, ref_bits as usize);
    assert_eq!(backend.get_int_value(&dead, 5), 55);
    unsafe { majit_backend::jitframe::free_off_gc_jitframe(jf) };
}

/// A 0 from `wasm_realloc_frame` is the MemoryError arm of
/// `emit_memory_error_on_truthy`: local 0 is not replaced by that 0, and the
/// arm returns through `propagate_exception_descr`.
#[test]
fn realloc_zero_takes_the_memory_error_exit() {
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let source = codegen::FrameGeometry::compact(2, 0, 0);
    let frame = source.extend(4, 0);
    assert!(frame.has_tail());
    let mut ca = codegen::CaParams::default();
    ca.realloc_fn_ptr = 1;
    ca.attached.propagate_exception_descr = 0x1357;
    let inputs = codegen::ModuleBuildInputs {
        inputargs,
        ops: vec![Op::new(OpCode::Finish, &[rb(OpRef::input_arg_int(0))])],
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers::default(),
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame,
        ca,
    };
    let (bytes, _, _, _) = codegen::build_wasm_module(&inputs).expect("realloc module");
    validate_wasm(&bytes);

    // Walk the zero-check `if` in one pass. `Operator` borrows the parser.
    let mut prev_eqz = false;
    let mut depth = 0i32;
    let mut in_arm = false;
    let mut returns = false;
    let mut stores_descr = false;
    let mut sets_local0 = false;
    let mut arm_closed = false;
    let mut installs_nonzero = false;
    let mut saw_get_after_arm = false;
    count_operators(&bytes, |op| {
        if in_arm {
            match op {
                wasmparser::Operator::If { .. }
                | wasmparser::Operator::Block { .. }
                | wasmparser::Operator::Loop { .. } => depth += 1,
                wasmparser::Operator::End => {
                    depth -= 1;
                    if depth == 0 {
                        in_arm = false;
                        arm_closed = true;
                    }
                }
                wasmparser::Operator::Return => returns = true,
                wasmparser::Operator::I32Const { value } if *value == 0x1357 => {
                    stores_descr = true;
                }
                wasmparser::Operator::LocalSet { local_index } if *local_index == 0 => {
                    sets_local0 = true;
                }
                _ => {}
            }
            prev_eqz = false;
            return;
        }
        if arm_closed && !installs_nonzero {
            if saw_get_after_arm {
                installs_nonzero = matches!(op, wasmparser::Operator::LocalSet { local_index: 0 });
                saw_get_after_arm = false;
            } else {
                saw_get_after_arm = matches!(op, wasmparser::Operator::LocalGet { .. });
            }
        }
        let eqz = matches!(op, wasmparser::Operator::I32Eqz);
        if prev_eqz && matches!(op, wasmparser::Operator::If { .. }) && !arm_closed {
            in_arm = true;
            depth = 1;
        }
        prev_eqz = eqz;
    });
    assert!(
        arm_closed && returns && stores_descr,
        "realloc's 0 result never reached the MemoryError arm"
    );
    assert!(
        !sets_local0,
        "a 0 realloc result must not be installed into local 0"
    );
    assert!(
        installs_nonzero,
        "only a non-zero realloc result is installed into local 0"
    );
}

/// A Ref input lives across a collecting allocation that forwards the running
/// frame. The guest reloads local 0 from the shadow stack, and the host
/// decodes the int and the forwarded Ref from that frame.
#[test]
#[ignore = "runtime integration test: a collecting alloc moves the jitframe and the host decodes it"]
fn nursery_entry_frame_moves_and_the_host_decodes_ref_and_int() {
    use majit_backend::Backend;
    use majit_backend::jitframe::{FIRST_ITEM_OFFSET, JitFrame, alloc_off_gc_jitframe};
    use majit_backend_wasm::failguard::{WasmFailDescr, WasmFrameData};

    let inputargs = vec![
        InputArg::from_type_rc(Type::Int, 0),
        InputArg::from_type_rc(Type::Ref, 1),
    ];
    let malloc = call_malloc_nursery(2, 32);
    let guard = make_guard(
        OpCode::GuardFalse,
        &[OpRef::const_int(1)],
        &[
            OpRef::input_arg_int(0),
            OpRef::const_int(99),
            OpRef::input_arg_ref(1),
        ],
    );
    guard.set_fail_arg_types(vec![Type::Int, Type::Int, Type::Ref]);
    let mut inputs = codegen::ModuleBuildInputs {
        inputargs: inputargs.clone(),
        ops: vec![malloc, guard],
        inlined_bridges: Vec::new(),
        constants: indexmap::IndexMap::new(),
        vtable_offset: Some(0),
        classptr_to_typeid: HashMap::new(),
        guard_gc_type_info: codegen::GuardGcTypeInfo::default(),
        alloc: codegen::AllocHelpers {
            new_fn_ptr: 1,
            ..codegen::AllocHelpers::default()
        },
        wb: codegen::WriteBarrierHelpers::for_current_gc(0, 0),
        nursery: None,
        invalidated_flag_addr: 0,
        gc_table_base: 0,
        gc_const_keys: Vec::new(),
        fail_index_base: 0,
        bridge_cells_base: 0,
        guard_cell_addrs: Vec::new(),
        bridge_entry_arity: None,
        bridge_param_dispatch: false,
        trace_entry_census: None,
        inline_trip: None,
        external_jump_slot: 0,
        external_jump_wide_slot: 0,
        external_jump_key: 0,
        frame: codegen::FrameGeometry::compact(8, 2, 0),
        ca: codegen::CaParams {
            jf_top_addr: Some(0x2000),
            ..codegen::CaParams::default()
        },
    };
    inputs = rewrite_module_inputs(inputs);
    let value_slots = codegen::frame_value_slots(&inputs.inputargs, &inputs.ops).max(8);
    let homes = codegen::count_ref_homes(&inputs.inputargs, &inputs.ops).max(1);
    inputs.frame = codegen::FrameGeometry::compact(value_slots, homes, 0);
    let frame = inputs.frame;
    let home = frame.home_ofs(0) as usize;
    let (bytes, guards, homes, _) =
        codegen::build_wasm_module(&inputs).expect("moving-frame trace compiles");
    assert!(homes >= 1, "the Ref input is homed across the allocation");
    validate_wasm(&bytes);
    let exit = &guards[0];

    let engine = Engine::default();
    let module = Module::new(&engine, &bytes).expect("module");
    let mut store = Store::new(&engine, ());
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).expect("memory");
    let old_items = 4096usize;
    let new_obj = 64 * 1024usize;
    let new_items = new_obj + FIRST_ITEM_OFFSET;
    let frame_bytes = frame.frame_bytes as usize;
    let top_addr = 0x2000usize;
    let stack_top = 0x3000usize;
    let old_obj = old_items - FIRST_ITEM_OFFSET;
    memory
        .write(&mut store, top_addr, &(stack_top as u32).to_le_bytes())
        .unwrap();
    memory
        .write(&mut store, stack_top - 4, &(old_obj as u32).to_le_bytes())
        .unwrap();
    let int_bits = 41i64;
    let ref_bits = 0x1111i64;
    memory
        .write(
            &mut store,
            old_items + codegen::FRAME_SLOT_BASE as usize,
            &int_bits.to_le_bytes(),
        )
        .unwrap();
    memory
        .write(
            &mut store,
            old_items + codegen::FRAME_SLOT_BASE as usize + 8,
            &ref_bits.to_le_bytes(),
        )
        .unwrap();
    let alloc = wasmi::Func::wrap(
        &mut store,
        move |mut caller: wasmi::Caller<'_, ()>, _tid: i64, _size: i64| -> i64 {
            let mut buf = vec![0u8; frame_bytes];
            memory
                .read(&caller, old_items, &mut buf)
                .expect("copy frame");
            memory
                .write(&mut caller, new_items, &buf)
                .expect("write moved frame");
            let mut home_bits = [0u8; 8];
            memory
                .read(&caller, new_items + home, &mut home_bits)
                .expect("ref home");
            let forwarded = i64::from_le_bytes(home_bits) + 0x100;
            memory
                .write(&mut caller, new_items + home, &forwarded.to_le_bytes())
                .unwrap();
            let marker = 0xDEADi64;
            memory
                .write(
                    &mut caller,
                    old_items + codegen::FRAME_SLOT_BASE as usize,
                    &marker.to_le_bytes(),
                )
                .unwrap();
            memory
                .write(&mut caller, stack_top - 4, &(new_obj as u32).to_le_bytes())
                .unwrap();
            0x4000
        },
    );
    let table = Table::new(
        &mut store,
        TableType::new(ValType::FuncRef, 2, None),
        Val::default(ValType::FuncRef),
    )
    .expect("table");
    table.set(&mut store, 1, Val::from(alloc)).unwrap();
    let mut linker = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .define("env", "__indirect_function_table", table)
        .unwrap();
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("instantiate");
    let returned = instance
        .get_typed_func::<i32, i32>(&store, "trace")
        .unwrap()
        .call(&mut store, old_items as i32)
        .expect("trace") as usize;
    assert_eq!(returned, new_items, "local 0 reloaded the moved frame");
    let mut marker = [0u8; 8];
    memory
        .read(
            &store,
            old_items + codegen::FRAME_SLOT_BASE as usize,
            &mut marker,
        )
        .unwrap();
    assert_eq!(
        i64::from_le_bytes(marker),
        0xDEAD,
        "the guard spilled onto the moved frame"
    );

    let depth = frame.signed_item_count();
    let jf = alloc_off_gc_jitframe(JitFrame::alloc_size(depth));
    assert!(!jf.is_null());
    let mut buf = vec![0u8; frame_bytes];
    memory.read(&store, new_items, &mut buf).unwrap();
    unsafe {
        JitFrame::init(jf, std::ptr::null(), depth);
        let items = (jf as *mut u8).add(FIRST_ITEM_OFFSET);
        std::ptr::copy_nonoverlapping(buf.as_ptr(), items, frame_bytes);
    }
    let descr = std::sync::Arc::new(WasmFailDescr {
        fail_index: exit.fail_index,
        trace_id: 0,
        fail_arg_types: exit.fail_arg_types.clone(),
        fail_locs: exit.fail_locs.clone(),
        is_finish: false,
        force_args_offset: frame.force_slot_base as u32,
        force_tail_base: frame.force_tail_base() as u32,
        force_prefix_slots: frame.prefix_value_slots as u32,
        value_tail_index: if frame.has_tail() {
            frame.value_tail_index() as u32
        } else {
            0
        },
        force_gcmap_ptr: exit.exit_gcmap_ptr,
        bridge_cell: 0,
        fail_arg_advanced: Vec::new(),
        trace_ref_homes: 0,
        trace_label_homes: 0,
        param_dispatch: false,
        bridge_slot: std::sync::atomic::AtomicU32::new(0),
        meta_descr: None,
    });
    let data = WasmFrameData::from_live_frame(jf, descr, false, false, None);
    let dead = majit_backend::DeadFrame::Boxed(data);
    let backend = majit_backend_wasm::WasmBackend::new();
    assert_eq!(backend.get_int_value(&dead, 0), int_bits);
    assert_eq!(backend.get_int_value(&dead, 1), 99);
    assert_eq!(
        backend.get_ref_value(&dead, 2).0,
        (ref_bits + 0x100) as usize
    );
    unsafe { majit_backend::jitframe::free_off_gc_jitframe(jf) };
}
