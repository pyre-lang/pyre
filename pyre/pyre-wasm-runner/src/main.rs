//! Native host that runs the wasm32 build of pyre under wasmtime so the full
//! interpreter+JIT path executes outside a browser.
//!
//! It is the non-JS counterpart of `majit-backend-wasm/js/jit_glue.js` and
//! implements the same host-import contract:
//!
//!   * the main module (`pyre-wasm` built with `--features wasm-host`) imports
//!     `majit_host.{jit_compile_wasm, jit_execute_wasm, jit_free_wasm}` and
//!     exports `memory`, `__indirect_function_table`, `pyre_alloc`,
//!     `pyre_dealloc`, and `pyre_run_python`;
//!   * each JIT-emitted trace module imports `env.jit.memory` (shared with the
//!     main module) plus an optional `env.jit_call` trampoline, and exports a
//!     `trace` function `(i32) -> i32`;
//!   * the trampoline reads func_ptr / args from the frame call area, dispatches
//!     through the main module's indirect function table (a `fn as usize` is a
//!     table index on wasm32), and writes the result back.
//!
//! CLI: `pyre-wasm-runner <script.py>` runs the script and writes its output to
//! stdout, matching the `pyrex` backend interface `pyre/check.py` drives. The
//! wasm module is located via `$PYRE_WASM_MODULE` or `--module <path>`, else the
//! default release artifact path.

mod host_path;
mod memory_ceiling;
mod wasmi_host;

#[global_allocator]
static GLOBAL: memory_ceiling::ProcessAllocator = memory_ceiling::ProcessAllocator::new();

bitflags::bitflags! {
    /// Mirror of `pyre_jit_trace::trace::fbw_diag::RingFlags`.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct FbwRingFlags: u64 {
        const VALID = 1 << 0;
        const COMMITTED = 1 << 1;
        const BRIDGE = 1 << 2;
    }
}

use std::path::{Path, PathBuf};

use wasmtime::error::Context;
use wasmtime::{
    AsContext, Caller, Config, Engine, Error, Func, Instance, Linker, Memory, Module, OptLevel,
    Result, Store, Table,
};

use crate::host_path::{guest_path_to_host, host_path_to_guest};

// Frame call-area offsets — must match `majit-backend-wasm/src/codegen.rs`.
// Shared with `wasmi_host`, which mirrors the same call-area protocol.
pub(crate) const CALL_RESULT_OFS: usize = majit_backend_wasm_host::CALL_RESULT_OFS as usize;
// The arg count also lives at offset 2016, but the trampoline derives arity
// (and the exact value types) from the resolved function's wasm signature
// instead, which is authoritative on wasm32. Kept for layout documentation.
#[allow(dead_code)]
pub(crate) const CALL_NARGS_OFS: usize = 2016;

pub(crate) const DEFAULT_MODULE: &str = "target/wasm32-unknown-unknown/release/pyre_wasm.wasm";

/// Which wasm runtime executes the module. `wasmtime` (cranelift) is fast in
/// steady state but has a large cold module-compilation cost; [`load_main_module`]
/// caches that result as `.cwasm`. `wasmi` is a pure-Rust interpreter with
/// near-zero load cost but slower hot loops.
/// Selected by `--engine` or `$PYRE_WASM_ENGINE` (CLI wins), default wasmtime.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum WasmEngine {
    Wasmtime,
    Wasmi,
}

impl WasmEngine {
    fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "wasmtime" => Ok(WasmEngine::Wasmtime),
            "wasmi" => Ok(WasmEngine::Wasmi),
            other => Err(format!("unknown engine {other:?} (want wasmtime|wasmi)")),
        }
    }
}

/// Per-store host state shared by all import callbacks.
#[derive(Default)]
struct Host {
    jit: majit_backend_wasm_host::TraceState,
    /// Real stdlib root the wasm module's `pyre_host.*` imports read source
    /// from (`$PYRE_STDLIB`, forwarded by `pyre/check.py`). The wasm side
    /// seeds it on `sys.path`, so the host serves genuine absolute paths.
    stdlib_root: Option<String>,
    /// `PYRE_WASM_JIT_STATS` diagnostic counters: trace modules compiled /
    /// executed this run. Per-store (not a global static): the increment sites
    /// already hold the `Caller<Host>`, and the runner is single-threaded.
    jit_compile_count: u64,
    /// `PYRE_WASM_JIT_STATS` diagnostic: total wall time spent inside
    /// `Module::new` (wasmtime Cranelift-compiling guest-emitted trace
    /// modules), in nanoseconds. Isolates host-compile latency — the warmup
    /// tax — from steady-state execution.
    jit_compile_time_ns: u128,
    /// Bytes of trace-module wasm handed to `Module::new`. With
    /// `jit_compile_count` and `jit_compile_time_ns` it separates the fixed
    /// per-module cost of a cranelift compile from the part that scales with
    /// the emitted body.
    jit_compile_bytes: u64,
    jit_execute_count: u64,
    /// Diagnostic: per-op residual-call host crossings (`env.jit_call`
    /// trampoline invocations). Compare against `jit_execute_count` to test
    /// whether per-op crossings or per-guard-exit crossings dominate.
    jit_call_count: u64,
    /// Wall time inside `jit_call_trampoline`, in nanoseconds, with nested
    /// crossings subtracted out. A residual callee is itself a guest function
    /// (`bh_call_fn` and friends live in the module's table), so running one
    /// re-enters the guest and every crossing it makes lands inside this
    /// frame's sample. Summing raw samples counts those seconds once per
    /// enclosing frame: on `fib_recursive`, 15168 crossings nest 7 deep and
    /// the raw sum reads 621ms against a 191ms non-overlapping total.
    jit_call_time_ns: u128,
    /// Time the crossings nested inside the one currently running have taken.
    /// Saved and restored by each frame, so it always names the innermost.
    jit_call_child_ns: u128,
    /// Crossings currently on the stack, and the deepest that has ever been.
    jit_call_depth: u32,
    jit_call_depth_max: u32,
    /// Crossings entered with at least one already on the stack.
    jit_call_nested: u64,
    /// Diagnostic (PYRE_WASM_EXEC_TRACE=1): histogram of (trace func_id,
    /// guard-exit fail_index) over every host round-trip, so we can see which
    /// guard keeps returning to the host instead of chaining in-module.
    exec_hist: std::collections::BTreeMap<(u32, u32), u64>,
    /// `PYRE_WASM_GUEST_PROFILE` sampling profiler; taken/restored around each
    /// epoch tick so `sample` can borrow the store it lives in.
    guest_profiler: Option<wasmtime::GuestProfiler>,
    /// Trace modules by the table slot that names them, recorded only while
    /// `PYRE_WASM_GUEST_PROFILE` is armed. `GuestProfiler` takes its module
    /// list once at construction and has no way to be handed a module compiled
    /// later, so every sample that lands in a trace resolves to nothing and is
    /// dropped; this list is what `WasmBacktrace` frames are matched against
    /// instead.
    trace_modules: Vec<(wasmtime::Module, u32)>,
    /// Epoch samples whose innermost frame was in a trace module, by that
    /// module's table slot and the function index within it.
    trace_samples: std::collections::BTreeMap<(u32, u32), u64>,
    /// Epoch samples whose innermost frame was anywhere else — the interpreter
    /// itself, and any sample taken with no wasm frame on the stack.
    trace_samples_elsewhere: u64,
    /// Process ceiling applied to this store's linear memories.
    mem_limit: memory_ceiling::GuestMemoryLimit,
}

impl majit_backend_wasm_host::HostState for Host {
    fn traces(&self) -> &majit_backend_wasm_host::TraceState {
        &self.jit
    }
    fn traces_mut(&mut self) -> &mut majit_backend_wasm_host::TraceState {
        &mut self.jit
    }
}

/// Report `PYRE_*` / `MAJIT_*` settings the guest cannot see.
///
/// The module is built for `wasm32-unknown-unknown`, whose `std::env` is
/// permanently empty: every `std::env::var_os` reached from inside the guest
/// returns `None` regardless of this process's environment. That covers every
/// JIT knob and probe that lives in the guest crates — `PYRE_GC_INTERP`,
/// `MAJIT_STRICT`, `MAJIT_LOG`, the `PYRE_FBW_*` / `PYRE_P2_DIAG` /
/// `PYRE_JD1_DEBUG` / `PYRE_JD1_DUMP` diagnostics — so setting one here
/// changes nothing at all. Silently ignoring them is the trap: an A/B run
/// through such a knob measures the same build twice and reads as a result.
/// Name them instead, once, on stderr (`check.py` shows a run's stderr only
/// when it fails, so this is invisible to a green suite and present in every
/// failure dump and interactive run).
///
/// Where a guest-side probe has a host-side replacement, use that instead:
/// `PYRE_FBW_DEBUG_ABORT`'s decline census is `PYRE_WASM_FBW_CENSUS` here,
/// read back through the `pyre_fbw_census` export, and `MAJIT_GUARD_CENSUS`'s
/// per-guard deopt census is `PYRE_WASM_GUARD_CENSUS`, armed through
/// `pyre_jit_guard_census_enable` and read through `pyre_jit_guard_census`.
/// `PYRE_FBW_SPEC_CENSUS`'s per-fold tallies are `PYRE_WASM_SPEC_CENSUS`,
/// armed through `pyre_fbw_spec_census_enable` and read through
/// `pyre_fbw_spec_census`.
/// `PYRE_WASM_TRACE_ENTRY_CENSUS` similarly arms the emitted-module entry
/// census before tracing starts. `PYRE_LOOP_CENSUS` keeps its native spelling
/// instead of gaining a `PYRE_WASM_` twin, because `check.py` grades every
/// backend against the same `# pyre-check: selfcheck-compiles=` header and a
/// second name would make the wasm leg answer a differently-spelled question.
///
/// Exempt: the names this runner interprets host-side (`PYRE_WASM_*`,
/// `PYRE_STDLIB`, `MAJIT_STATS`, `PYRE_LOOP_CENSUS`), the knobs it forwards
/// into the guest (`PYRE_NO_JIT`, `PYRE_JIT`, `MAJIT_NO_BRIDGE`, `PYRE_NO_JD1`,
/// `PYRE_JD1`, `PYRE_JD1_NO_ENTER`, `PYRE_JD1_THRESHOLD`, `MAJIT_BRIDGE_BAIL`,
/// `MAJIT_MAX_BRIDGES`, `MAJIT_SKIP_BRIDGES`, `MAJIT_GC_STRESS`),
/// and `check.py`'s own `PYRE_CHECK_*` interpreter paths. A knob that later
/// becomes host-interpreted must be added here; the prefix match needs no
/// upkeep for new guest-side knobs.
fn warn_inert_guest_env() {
    const HOST_HANDLED: &[&str] = &[
        "PYRE_STDLIB",
        "MAJIT_STATS",
        "PYRE_LOOP_CENSUS",
        "PYRE_NO_JIT",
        "PYRE_JIT",
        "MAJIT_NO_BRIDGE",
        "PYRE_NO_JD1",
        "PYRE_JD1",
        "PYRE_JD1_NO_ENTER",
        "PYRE_JD1_THRESHOLD",
        "MAJIT_BRIDGE_BAIL",
        "MAJIT_MAX_BRIDGES",
        "MAJIT_SKIP_BRIDGES",
        "PYRE_MAX_MEMORY",
    ];
    // Forwarded into the guest through `pyre_set_env` / `pyre_env_names`,
    // not interpreted here. Presence of the name is the native contract.
    const GUEST_FORWARDED_GC: &[&str] = &["MAJIT_GC_STRESS"];
    // `to_string_lossy`, not `into_string().ok()`: a name the platform allows
    // but UTF-8 does not is still a setting the guest silently ignores, and
    // dropping it here would hide exactly the case worth reporting.
    let mut inert: Vec<String> = std::env::vars_os()
        .map(|(name, _)| name.to_string_lossy().into_owned())
        .filter(|name| name.starts_with("PYRE_") || name.starts_with("MAJIT_"))
        .filter(|name| {
            !name.starts_with("PYRE_WASM_")
                && !name.starts_with("PYRE_CHECK_")
                && !HOST_HANDLED.contains(&name.as_str())
                && !GUEST_FORWARDED_GC.contains(&name.as_str())
        })
        .collect();
    if inert.is_empty() {
        return;
    }
    inert.sort();
    inert.dedup();
    eprintln!(
        "[pyre-wasm-runner] ignoring {}: the wasm guest has no environment, so \
         guest-side knobs and probes do nothing here",
        inert.join(", ")
    );
}

fn main() {
    memory_ceiling::install();
    let t0_main = std::time::Instant::now();
    warn_inert_guest_env();
    let startup_trace = std::env::var_os("PYRE_WASM_STARTUP_TRACE").is_some();
    let mut module_path: Option<PathBuf> = None;
    let mut script: Option<PathBuf> = None;
    let mut inspect = false;
    let mut engine: Option<WasmEngine> = None;

    // `args_os`, not `args`: the latter unwraps the UTF-8 conversion, so a
    // script path spelling a byte with no UTF-8 form aborts the runner before
    // it can name the file. Every flag below is ASCII by construction, so a
    // value that does not convert is never one and takes the positional arm.
    let mut argv = std::env::args_os().skip(1);
    while let Some(arg) = argv.next() {
        let flag = arg.to_str().map(str::to_owned);
        match flag.as_deref() {
            Some("--inspect") => inspect = true,
            Some("--module") => {
                module_path = Some(PathBuf::from(
                    argv.next()
                        .unwrap_or_else(|| fatal("--module needs a path")),
                ))
            }
            Some("--engine") => {
                let v = argv
                    .next()
                    .unwrap_or_else(|| fatal("--engine needs a value"));
                let v = v
                    .to_str()
                    .unwrap_or_else(|| fatal("--engine needs a value"));
                engine = Some(WasmEngine::parse(v).unwrap_or_else(|e| fatal(&e)));
            }
            Some("-h") | Some("--help") => {
                eprintln!(
                    "usage: pyre-wasm-runner [--module <pyre_wasm.wasm>] \
                     [--engine wasmtime|wasmi] [--inspect] <script.py>"
                );
                std::process::exit(2);
            }
            Some(other) if other.starts_with('-') => fatal(&format!("unknown flag {other}")),
            _ => script = Some(PathBuf::from(arg)),
        }
    }

    let module_path = module_path
        .or_else(|| std::env::var_os("PYRE_WASM_MODULE").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODULE));

    // CLI `--engine` wins over `$PYRE_WASM_ENGINE`; default wasmtime.
    let engine = engine
        .or_else(|| {
            std::env::var("PYRE_WASM_ENGINE")
                .ok()
                .map(|v| WasmEngine::parse(&v).unwrap_or_else(|e| fatal(&e)))
        })
        .unwrap_or(WasmEngine::Wasmtime);

    // The wasm runs on the calling thread's stack (sync wasmtime), so the
    // deep interpreter recursion needs a large host stack to back the
    // generous `max_wasm_stack` set in `run`. wasmi runs on its own heap
    // stacks, but the large reservation is harmless there too.
    let worker = std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(move || -> std::result::Result<i32, String> {
            if inspect {
                return match engine {
                    WasmEngine::Wasmtime => {
                        inspect_module(&module_path).map(|()| 0).map_err(fmt_err)
                    }
                    WasmEngine::Wasmi => wasmi_host::inspect(&module_path),
                };
            }
            let script = script.ok_or_else(|| "no script given".to_string())?;
            let source = std::fs::read_to_string(&script)
                .map_err(|e| format!("read script {}: {e}", script.display()))?;
            match engine {
                WasmEngine::Wasmtime => run(&module_path, &source, &script).map_err(fmt_err),
                WasmEngine::Wasmi => wasmi_host::run(&module_path, &source, &script),
            }
        })
        .expect("spawn worker thread");

    match worker.join() {
        Ok(Ok(code)) => {
            if startup_trace {
                eprintln!(
                    "[startup] TOTAL(main-entry..worker-joined) {:.1}ms",
                    t0_main.elapsed().as_secs_f64() * 1e3
                );
            }
            std::process::exit(code)
        }
        Ok(Err(e)) => fatal(&e),
        Err(_) => fatal("worker thread panicked"),
    }
}

/// Format a wasmtime error with its `{:?}` chain for the fatal handler.
fn fmt_err(e: Error) -> String {
    format!("{e:?}")
}

fn fatal(msg: &str) -> ! {
    eprintln!("pyre-wasm-runner: {msg}");
    std::process::exit(1);
}

fn run(module_path: &Path, source: &str, script: &Path) -> Result<i32> {
    let mut config = Config::new();
    // Allow the interpreter's deep recursion before wasmtime raises a stack
    // overflow trap; the interpreter's own recursion limit normally fires
    // first. Kept below the worker thread's stack reservation. wasmtime
    // requires `max_wasm_stack <= async_stack_size` even for sync execution,
    // so the async stack is sized just above it.
    const WASM_STACK: usize = 256 * 1024 * 1024;
    config.max_wasm_stack(WASM_STACK);
    config.async_stack_size(WASM_STACK + 1024 * 1024);
    // Trace modules are compiled synchronously on the Python thread, and
    // call-heavy workloads materialize dozens of them before doing enough
    // work to amortize that pause.  Keep Cranelift's optimizing pipeline, but
    // ask it to optimize for size as well: smaller trace bodies reduce both
    // synchronous compile work and instruction-cache pressure.  This setting
    // belongs to the wasm host boundary; the shared frontend optimizer and IR
    // are unchanged.
    config.cranelift_opt_level(OptLevel::SpeedAndSize);
    // `ru_utime` sums every Cranelift worker. A bridge-heavy fixture
    // compiles dozens of few-kilobyte modules; the pool's extra user time
    // is what `inline_bignum_bridge_twoclamp` spent above dynasm (measured
    // 0.57s -> 0.36s, compile wall unchanged at ~120ms). One thread keeps
    // that user time on the compile itself. The main module arrives from
    // `.cwasm`, so this does not redo its cached compile.
    config.parallel_compilation(false);
    // JIT trace modules emit `return_call_indirect` to chain a loop-closing bridge
    // back into its loop at constant stack depth (the tail-call proposal).
    config.wasm_tail_call(true);
    // Optional instruction budget: PYRE_WASM_FUEL=N traps the guest after N fuel
    // units so a diagnostic run that livelocks (e.g. a buggy loop-closing bridge)
    // still reaches the stats readout instead of hanging forever.
    let fuel_limit: Option<u64> = std::env::var("PYRE_WASM_FUEL")
        .ok()
        .and_then(|s| s.parse().ok());
    // Fuel doubles as an exact executed-instruction counter, reported as
    // `wasm_ops` in the stats line below. It is the one wasm measurement that
    // does not move with machine load, which makes it the right thing to
    // compare a codegen change against. Metering costs time on every
    // instruction, so it stays confined to the two diagnostic modes that ask
    // for it — a timed run sets neither and pays nothing.
    //
    // `PYRE_WASM_JIT_STATS=nofuel` asks for the readout without the metering.
    // Every other counter on that line is measured the same way with or
    // without fuel, but `compile_ms` is host wall time around a cranelift
    // compile that fuel instrumentation itself makes slower — so the metered
    // run cannot say what trace compilation costs a production run. That
    // reading is what this spelling exists for; `wasm_ops` reports -1 under it.
    let jit_stats_value = std::env::var("PYRE_WASM_JIT_STATS").ok();
    let jit_stats = std::env::var_os("PYRE_WASM_JIT_STATS").is_some();
    let meter_fuel = fuel_limit.is_some()
        || (jit_stats && jit_stats_value.as_deref().map(str::trim) != Some("nofuel"));
    if meter_fuel {
        config.consume_fuel(true);
    }
    // Diagnostic: PYRE_WASM_GUEST_PROFILE=<out.json> writes a sampling profile
    // of the guest in the Firefox processed format (main-module symbols only;
    // JIT trace modules appear as omitted frames under `execute_token`). View
    // at https://profiler.firefox.com/. Samples land on epoch checkpoints
    // (function entries / loop back-edges), so call-dense small functions are
    // over-represented and bulk ops (memory.fill) are attributed to the next
    // checkpoint. Epoch interruption changes main-module codegen, so this mode
    // caches its module under its own `.cwasm` (see `cache_variant`).
    let guest_profile_out = std::env::var("PYRE_WASM_GUEST_PROFILE").ok();
    if guest_profile_out.is_some() {
        config.epoch_interruption(true);
    }
    let cache_variant = cache_variant(meter_fuel, guest_profile_out.is_some());
    // Diagnostic: PYRE_WASM_STARTUP_TRACE=1 prints a fixed-startup breakdown
    // (module load/deserialize, main-module instantiate, guest bootstrap+run)
    // so the warmup tax can be attributed to host module-load vs guest
    // interpreter bootstrap.
    let startup_trace = std::env::var_os("PYRE_WASM_STARTUP_TRACE").is_some();
    let mut startup_mark = std::time::Instant::now();
    let mut startup_lap = |label: &str| {
        if startup_trace {
            eprintln!(
                "[startup] {label} {:.1}ms",
                startup_mark.elapsed().as_secs_f64() * 1e3
            );
            startup_mark = std::time::Instant::now();
        }
    };
    let engine = Engine::new(&config)?;
    startup_lap("engine_new");

    let module = load_main_module(&engine, module_path, cache_variant)?;
    startup_lap("load_module");

    let mut store = Store::new(&engine, Host::default());
    store.data_mut().mem_limit.ceiling = memory_ceiling::bytes();
    if store.data().mem_limit.ceiling != 0 {
        store.limiter(|host| &mut host.mem_limit);
    }
    if guest_profile_out.is_some() {
        const SAMPLE_INTERVAL: std::time::Duration = std::time::Duration::from_micros(200);
        let profiler = wasmtime::GuestProfiler::new(
            &engine,
            "pyre-wasm",
            SAMPLE_INTERVAL,
            vec![("pyre_wasm".to_string(), module.clone())],
        )?;
        store.data_mut().guest_profiler = Some(profiler);
        store.set_epoch_deadline(1);
        store.epoch_deadline_callback(|mut ctx| {
            if let Some(mut p) = ctx.data_mut().guest_profiler.take() {
                p.sample(&ctx, std::time::Duration::ZERO);
                ctx.data_mut().guest_profiler = Some(p);
            }
            // The same tick, attributed a second way. A backtrace resolves its
            // frames through the store's module registry, which every trace
            // module joins when it is instantiated, so it reaches the code the
            // profiler's fixed module list cannot name.
            let hit = {
                let backtrace = wasmtime::WasmBacktrace::capture(&ctx);
                let host = ctx.data();
                backtrace.frames().first().and_then(|frame| {
                    host.trace_modules
                        .iter()
                        .find(|(module, _)| wasmtime::Module::same(module, frame.module()))
                        .map(|(_, slot)| (*slot, frame.func_index()))
                })
            };
            match hit {
                Some(key) => *ctx.data_mut().trace_samples.entry(key).or_insert(0) += 1,
                None => ctx.data_mut().trace_samples_elsewhere += 1,
            }
            Ok(wasmtime::UpdateDeadline::Continue(1))
        });
        let ticker_engine = engine.clone();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(SAMPLE_INTERVAL);
                ticker_engine.increment_epoch();
            }
        });
    }
    store.data_mut().stdlib_root = std::env::var("PYRE_STDLIB").ok();
    // Counting-only runs get the whole budget: `wasm_ops` is a subtraction, and
    // a run that exhausts its fuel reports the limit rather than its own cost.
    let fuel_start = meter_fuel.then(|| fuel_limit.unwrap_or(u64::MAX));
    if let Some(n) = fuel_start {
        store.set_fuel(n)?;
    }

    let linker = build_linker(&engine)?;
    let instance = linker
        .instantiate(&mut store, &module)
        .context("instantiate main module")?;
    startup_lap("instantiate");

    let memory = instance
        .get_memory(&mut store, "memory")
        .context("main module is missing its `memory` export")?;
    let table = instance
        .get_table(&mut store, "__indirect_function_table")
        .context("main module is missing its `__indirect_function_table` export (build with --export-table)")?;
    // The table's current size is the first slot a later `table.grow` will
    // return, i.e. the first JIT-trace id; everything below it is a main-module
    // function that must never be dispatched as, or freed like, a trace.
    let trace_base = table.size(&store);
    store.data_mut().jit.memory = Some(memory);
    store.data_mut().jit.table = Some(table);
    store.data_mut().jit.trace_base = trace_base;

    let alloc = instance.get_typed_func::<u32, u32>(&mut store, "pyre_alloc")?;
    let run_python = instance.get_typed_func::<(u32, u32), u64>(&mut store, "pyre_run_python")?;
    let dealloc = instance.get_typed_func::<(u32, u32), ()>(&mut store, "pyre_dealloc")?;
    if let Some(selector) = std::env::var("PYRE_WASM_FORCE_CA_TERMINAL_DECLINE")
        .ok()
        .filter(|value| !value.is_empty())
    {
        let selector = selector.parse::<u64>().context(
            "PYRE_WASM_FORCE_CA_TERMINAL_DECLINE must be 1 or a decimal JitCellToken number",
        )?;
        let set_force = instance
            .get_typed_func::<u64, ()>(&mut store, "pyre_set_force_ca_terminal_decline")
            .context("wasm module lacks terminal-decline regression hook")?;
        set_force.call(&mut store, selector)?;
    }

    // One environment for the guest. The launcher, the collector and the JIT
    // knobs all read it; the runner forwards every name the module asks for in
    // one blob. Values stay undecoded: `PYTHONSAFEPATH` is a presence flag on
    // the raw bytes, and a non-UTF-8 value is still set.
    //
    // A module predating `pyre_set_env` still understands `-P` through
    // `pyre_set_safe_path`.
    let env_names = instance
        .get_typed_func::<(), u64>(&mut store, "pyre_env_names")
        .ok();
    let set_env = instance
        .get_typed_func::<(u32, u32), ()>(&mut store, "pyre_set_env")
        .ok();
    if let (Some(names), Some(set_env)) = (env_names, set_env) {
        let packed = names.call(&mut store, ())?;
        let (nptr, nlen) = ((packed >> 32) as u32, packed as u32);
        let mut buf = vec![0u8; nlen as usize];
        memory.read(&store, nptr as usize, &mut buf)?;
        dealloc.call(&mut store, (nptr, nlen))?;

        let mut blob: Vec<u8> = Vec::new();
        for name in String::from_utf8_lossy(&buf)
            .split('\0')
            .filter(|name| !name.is_empty())
        {
            let Some(value) = std::env::var_os(name) else {
                continue;
            };
            if !blob.is_empty() {
                blob.push(0);
            }
            blob.extend_from_slice(name.as_bytes());
            blob.push(b'=');
            blob.extend_from_slice(&value.into_encoded_bytes());
        }

        let blen = blob.len() as u32;
        if blen == 0 {
            set_env.call(&mut store, (0, 0))?;
        } else {
            let p = alloc.call(&mut store, blen)?;
            memory.write(&mut store, p as usize, &blob)?;
            set_env.call(&mut store, (p, blen))?;
            dealloc.call(&mut store, (p, blen))?;
        }
    } else if std::env::var_os("PYTHONSAFEPATH").is_some_and(|value| !value.is_empty())
        && let Ok(set_safe_path) =
            instance.get_typed_func::<u32, ()>(&mut store, "pyre_set_safe_path")
    {
        set_safe_path.call(&mut store, 1)?;
    }

    // Name the script so the guest compiles it under its real path: that is
    // what a traceback prints, what its source-line lookup reads back through
    // `pyre_host.host_read`, and the directory that heads `sys.path`. Absent
    // on a module predating the export, leaving the guest on `<string>`.
    if let Ok(set_path) =
        instance.get_typed_func::<(u32, u32), ()>(&mut store, "pyre_set_script_path")
    {
        let name = host_path_to_guest(script).ok_or_else(|| {
            Error::msg(format!(
                "script path cannot be represented in guest: {}",
                script.display()
            ))
        })?;
        let nlen = name.len() as u32;
        if nlen != 0 {
            let p = alloc.call(&mut store, nlen)?;
            memory.write(&mut store, p as usize, &name)?;
            set_path.call(&mut store, (p, nlen))?;
            dealloc.call(&mut store, (p, nlen))?;
        }
    }

    // Arm the per-guard deopt census before the run, not after: it records on
    // the deopt path and stays off until asked. `MAJIT_GUARD_CENSUS` is inert
    // here for the usual reason, so the switch travels as a call. Absent on a
    // module predating the export, in which case the readout below finds
    // nothing and says so.
    if std::env::var_os("PYRE_WASM_GUARD_CENSUS").is_some()
        && let Ok(arm) =
            instance.get_typed_func::<(), ()>(&mut store, "pyre_jit_guard_census_enable")
    {
        arm.call(&mut store, ())?;
    }
    // The specialization census counts nothing until armed -- an unarmed run
    // is meant to do no extra work on the dispatch path -- so this must
    // precede the run rather than only the readout. `PYRE_FBW_SPEC_CENSUS` is
    // inert here for the usual reason.
    if std::env::var_os("PYRE_WASM_SPEC_CENSUS").is_some()
        && let Ok(arm) =
            instance.get_typed_func::<(), ()>(&mut store, "pyre_fbw_spec_census_enable")
    {
        arm.call(&mut store, ())?;
    }
    if std::env::var_os("PYRE_WASM_TRACE_ENTRY_CENSUS").is_some()
        && let Ok(arm) =
            instance.get_typed_func::<(), ()>(&mut store, "pyre_jit_trace_entry_census_enable")
    {
        arm.call(&mut store, ())?;
    }
    // The compile census records as each trace compiles, so it too has to be
    // armed before the run. It keeps its native name: `check.py` grades a
    // selfcheck fixture's declared shapes on every backend from one header.
    if std::env::var_os("PYRE_LOOP_CENSUS").is_some()
        && let Ok(arm) = instance.get_typed_func::<(), ()>(&mut store, "pyre_loop_census_enable")
    {
        arm.call(&mut store, ())?;
    }
    let src = source.as_bytes();
    let len = src.len() as u32;
    let in_ptr = if len == 0 {
        0
    } else {
        let p = alloc.call(&mut store, len)?;
        memory.write(&mut store, p as usize, src)?;
        p
    };

    // Keep the run result so the diagnostic stats can be read out even when the
    // guest traps (a panic, or a PYRE_WASM_FUEL exhaustion that interrupts a
    // livelock): the JIT counters are populated at compile time, before the run
    // finishes, and the diag exports just read statics that survive a trap.
    let run_result = run_python.call(&mut store, (in_ptr, len));
    startup_lap("run_python(bootstrap+script)");
    // After a fuel-exhaustion trap the store has no fuel, so the diagnostic
    // export calls below would themselves immediately trap and read as 0.
    // Refill so the readout reflects the real (compile-time) counter values.
    // Read the remaining budget first: `start - remaining` is the executed
    // instruction count, which makes "how much does one loop iteration cost"
    // answerable by differencing two runs at different trip counts. A run that
    // exhausted its budget reports `used == limit` and says nothing, so size the
    // limit above what the script needs.
    let wasm_ops = fuel_start.map(|start| {
        let remaining = store.get_fuel().unwrap_or(0);
        let used = start.saturating_sub(remaining);
        if let Some(limit) = fuel_limit {
            eprintln!("[fuel] used={used} remaining={remaining} limit={limit}");
        }
        let _ = store.set_fuel(u64::MAX);
        used
    });
    if let Some(path) = &guest_profile_out
        && let Some(p) = store.data_mut().guest_profiler.take()
    {
        match std::fs::File::create(path).context("create guest profile output") {
            Ok(f) => match p
                .finish(std::io::BufWriter::new(f))
                .context("write guest profile")
            {
                Ok(()) => eprintln!("[guest-profile] wrote {path}"),
                Err(err) => eprintln!("[guest-profile] failed to write {path}: {err:#}"),
            },
            Err(err) => eprintln!("[guest-profile] failed to create {path}: {err:#}"),
        }
        let host = store.data();
        let in_traces: u64 = host.trace_samples.values().sum();
        let total = in_traces + host.trace_samples_elsewhere;
        if total != 0 {
            eprintln!(
                "[trace-profile] samples={total} in_traces={in_traces} ({:.1}%) elsewhere={}",
                100.0 * in_traces as f64 / total as f64,
                host.trace_samples_elsewhere
            );
            let mut rows: Vec<_> = host.trace_samples.iter().collect();
            rows.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
            for ((slot, func), count) in rows.into_iter().take(20) {
                eprintln!(
                    "[trace-profile] slot={slot} func={func} samples={count} ({:.1}%)",
                    100.0 * *count as f64 / total as f64
                );
            }
        }
    }
    if std::env::var_os("PYRE_WASM_JIT_STATS").is_some() {
        let lin_mem = memory.data_size(&store);
        // Split linear-memory growth into GC-retained vs. host-heap: a leak that
        // shows up here but NOT in oldgen/nursery is a Rust-heap leak, not GC
        // false-retention. The exports are diagnostic; tolerate their absence.
        let gc_oldgen = instance
            .get_typed_func::<(), u64>(&mut store, "pyre_gc_oldgen_bytes")
            .and_then(|f| f.call(&mut store, ()))
            .unwrap_or(0);
        let gc_nursery = instance
            .get_typed_func::<(), u64>(&mut store, "pyre_gc_nursery_bytes")
            .and_then(|f| f.call(&mut store, ()))
            .unwrap_or(0);
        let gc_minors = instance
            .get_typed_func::<(), u64>(&mut store, "pyre_gc_minor_collections")
            .and_then(|f| f.call(&mut store, ()))
            .unwrap_or(0);
        let gc_majors = instance
            .get_typed_func::<(), u64>(&mut store, "pyre_gc_major_collections")
            .and_then(|f| f.call(&mut store, ()))
            .unwrap_or(0);
        // `heap-prof` builds only: net-live guest-heap bytes/count. Distinguishes
        // a true not-freed leak (live grows with executes) from fragmentation.
        let heap_live_bytes = instance
            .get_typed_func::<(), i64>(&mut store, "pyre_heap_live_bytes")
            .and_then(|f| f.call(&mut store, ()))
            .unwrap_or(-1);
        let heap_live_count = instance
            .get_typed_func::<(), i64>(&mut store, "pyre_heap_live_count")
            .and_then(|f| f.call(&mut store, ()))
            .unwrap_or(-1);
        // Per-size-class net-live histogram (heap-prof builds): bucket i covers
        // sizes (8*(i-1), 8*i]. Surfaces the exact leaking size class.
        if let Ok(bucket) = instance.get_typed_func::<u32, i64>(&mut store, "pyre_heap_bucket") {
            let mut parts = Vec::new();
            for i in 0u32..64 {
                let n = bucket.call(&mut store, i).unwrap_or(0);
                if n > 1000 {
                    parts.push(format!("≤{}B:{}", i * 8, n));
                }
            }
            if !parts.is_empty() {
                eprintln!("[jit-stats] heap_buckets {}", parts.join(" "));
            }
        }
        // The `Counters.ABORT_*` split behind `loops_aborted` (diagnostic).
        // The native backends get this from MAJIT_LOG's summary, which the
        // guest cannot read; without it a wasm-only `loops_aborted` regression
        // names no reason. Labels mirror
        // `majit_metainterp::jitprof::ABORT_COUNTER_KINDS`, which is
        // append-only, so a guest older than this runner just reads 0 for a
        // trailing label.
        if let Ok(ab) = instance.get_typed_func::<u32, u64>(&mut store, "pyre_jit_abort_diag") {
            let labels = [
                "too_long",
                "bridge_or_generic",
                "bad_loop",
                "escape",
                "force_quasiimmut",
                "segmented_trace",
            ];
            let mut parts = Vec::with_capacity(labels.len());
            for (i, lbl) in labels.iter().enumerate() {
                let n = ab.call(&mut store, i as u32).unwrap_or(0);
                parts.push(format!("{lbl}={n}"));
            }
            eprintln!("[jit-stats] abort_diag {}", parts.join(" "));
        }
        // compile_bridge outcome tallies (diagnostic). POSITIONAL MIRROR of
        // `majit_backend_wasm::BRIDGE_DIAG_LABELS`, which carries the legend
        // for every slot; the runner links no majit crate, so a slot added
        // there must be APPENDED here or its tally is bumped in the guest and
        // reported by nobody.
        if let Ok(diag) = instance.get_typed_func::<u32, u64>(&mut store, "pyre_jit_bridge_diag") {
            let labels = [
                "entered",
                "decl_callasm",
                "decl_multipeel",
                "decl_notdirect",
                "decl_refhome",
                "BRIDGE_OK",
                "loopclosing",
                "src_preamble",
                "ml_descr_none",
                "ml_unsafe_label",
                "ml_arity_mismatch",
                "decl_noadvance",
                "ca_cell_set",
                "ca_cells_zero",
                "accepted_ca",
                "unused_ca_trampoline",
                "forced_ca_terminal_decline",
                "ml_no_descr",
                "ml_unpublished",
                "pub_peeled",
                "pub_flat",
                "pub_flat_skipped",
                "label_retracted",
                "cl_entered",
                "cl_ok",
                "cl_decl_unsupported",
                "cl_decl_host_reject",
                "cell_set",
                "cell_missing",
                "cell_rebridge",
                "reemit_failed",
                "reemit_ok",
                "inline_ok",
                "inline_decl_not_direct",
                "inline_decl_not_loop_closing",
                "inline_decl_not_reemittable",
                "inline_decl_already_owned",
                "inline_decl_frame",
                "inline_decl_not_header",
                "inline_decl_no_loop_label",
                "inline_decl_value_layout",
                "inline_decl_ref_layout",
                "inline_decl_missing_label",
                "inline_decl_other",
                "bridge_param_ok",
                "bridge_param_decl_source_frame",
                "bridge_param_decl_arity",
                "bridge_param_label_suppressed",
                "inline_decl_label_resume_layout",
                "inline_decl_call_assembler",
                "inline_decl_owner_invalidated",
                "inline_foreign_jump",
                "inline_ok_outside_loop",
                "inline_decl_no_trip_helper",
                "inline_deferred",
                "inline_trip_fired",
                "inline_decl_defer_invalidation_guard",
                "ca_decl_opcode",
                "ca_decl_descr",
                "ca_decl_types",
                "ca_decl_unregistered",
                "ca_decl_materialize",
                "ca_decl_geometry",
                "ca_decl_terminal",
                "inline_decl_foreign_eager",
                "inline_decl_eager_too_large",
                "cl_decl_codegen",
            ];
            let mut parts = Vec::new();
            for (i, lbl) in labels.iter().enumerate() {
                let n = diag.call(&mut store, i as u32).unwrap_or(0);
                parts.push(format!("{lbl}={n}"));
            }
            eprintln!("[jit-stats] bridge_diag {}", parts.join(" "));
        }
        // `compile_loop` `Err` reason. `cl_entered` without `cl_ok` /
        // `cl_decl_*` means `build_wasm_module` declined and the string
        // never left the guest; this is the only host-visible copy.
        if let (Ok(len_fn), Ok(byte_fn)) = (
            instance.get_typed_func::<(), u32>(&mut store, "pyre_jit_last_compile_err_len"),
            instance.get_typed_func::<u32, u32>(&mut store, "pyre_jit_last_compile_err_byte"),
        ) {
            let len = len_fn.call(&mut store, ()).unwrap_or(0);
            if len > 0 {
                let mut bytes = Vec::with_capacity(len as usize);
                for i in 0..len {
                    bytes.push(byte_fn.call(&mut store, i).unwrap_or(0) as u8);
                }
                eprintln!(
                    "[jit-stats] last_compile_err {}",
                    String::from_utf8_lossy(&bytes)
                );
            }
        }
        // The portal's own decision tallies. POSITIONAL MIRROR of
        // `pyre_jit::eval::PORTAL_DIAG_LABELS`, but unlike `bridge_diag` above
        // the guest also exports the slot count, so a slot added there without
        // a label here is reported as `slot<N>` rather than silently dropped.
        if let Ok(diag) = instance.get_typed_func::<u32, u64>(&mut store, "pyre_jit_portal_diag") {
            let labels = ["can_enter_jit", "can_enter_jit_taken"];
            let guest_len = instance
                .get_typed_func::<(), u32>(&mut store, "pyre_jit_portal_diag_len")
                .ok()
                .and_then(|f| f.call(&mut store, ()).ok())
                .unwrap_or(labels.len() as u32) as usize;
            let mut parts = Vec::new();
            for i in 0..guest_len.max(labels.len()) {
                let n = diag.call(&mut store, i as u32).unwrap_or(0);
                match labels.get(i) {
                    Some(lbl) => parts.push(format!("{lbl}={n}")),
                    None => parts.push(format!("slot{i}={n}")),
                }
            }
            eprintln!("[jit-stats] portal_diag {}", parts.join(" "));
        }
        if let Ok(geometry) =
            instance.get_typed_func::<u32, u64>(&mut store, "pyre_jit_inline_geometry_diag")
        {
            let mut parts = Vec::new();
            for i in 0..3 {
                let packed = geometry.call(&mut store, i).unwrap_or(0);
                if packed != 0 {
                    let kind = match (packed >> 48) as u8 {
                        1 => "frame_value_slots",
                        2 => "ordinary_ref_homes",
                        3 => "label_resume_ref_slots",
                        4 => "label_resume_capture_slots",
                        _ => "unknown",
                    };
                    let needed = (packed >> 24) & 0x00ff_ffff;
                    let available = packed & 0x00ff_ffff;
                    parts.push(format!("{}:{kind}={needed}/{available}", i + 1));
                }
            }
            if !parts.is_empty() {
                let count = instance
                    .get_typed_func::<(), u64>(&mut store, "pyre_jit_inline_geometry_count")
                    .ok()
                    .and_then(|count| count.call(&mut store, ()).ok());
                match count {
                    Some(count) => eprintln!(
                        "[jit-stats] inline_geometry records={}/{} {}",
                        parts.len(),
                        count,
                        parts.join(" ")
                    ),
                    None => eprintln!("[jit-stats] inline_geometry {}", parts.join(" ")),
                }
            }
        }
        // Per-walk full-body-walk census (diagnostic). Prints the same record
        // the native backends print as `[fbw-census]` under PYRE_FBW_CENSUS,
        // which the guest cannot read. Slot layout in
        // `pyre_jit_trace::trace::fbw_diag`.
        //
        // The tallies BELOW the ring are not printed here: they live on the
        // `[jit-stats] fbw_diag` line in the MAJIT_STATS block, which is the
        // gate the native reader prints its copy under, so the two lines can be
        // diffed from runs made the same way. check.py never sets
        // PYRE_WASM_JIT_STATS, so printing them here would have kept them off
        // every gated wasm run.
        if let Ok(fbw) = instance.get_typed_func::<u32, u64>(&mut store, "pyre_fbw_diag") {
            // POSITIONAL MIRROR of the ring geometry and bit layout in
            // `pyre_jit_trace::trace::fbw_diag`. The runner links no pyre
            // crate and decodes the ring by arithmetic on these, so a stale
            // copy reads the wrong words and prints them as a walk outcome
            // instead of failing. Named rather than spelled inline so the
            // mirror is something a test can compare.
            const RING_BASE: u32 = 18;
            const RING_ENTRIES: u32 = 24;
            const RING_STRIDE: u32 = 5;
            const NAME_SLOTS: u32 = 4;
            const SHIFT_EFFECTS: u32 = 8;
            const SHIFT_JOURNAL: u32 = 24;
            const SHIFT_EXEC_MF: u32 = 40;
            const SHIFT_LEG: u32 = 56;
            const FIELD_MASK: u64 = 0xffff;
            let mut slot = |i: u32| fbw.call(&mut store, i).unwrap_or(0);
            let walks = slot(0);
            // `fbw_diag::record` keeps the FIRST `RING_ENTRIES` walks and lets
            // later ones bump the totals only, while the native reader prints
            // every walk it takes. A run past the ring therefore hands back a
            // tally that reads like the whole population and diffs short
            // against the native one, so name the shortfall instead of leaving
            // a `sort | uniq -c` over these lines silently truncated.
            if walks > u64::from(RING_ENTRIES) {
                eprintln!("[fbw-census] (first {RING_ENTRIES} walks of {walks})");
            }
            for entry in 0..RING_ENTRIES.min(walks as u32) {
                let base = RING_BASE + entry * RING_STRIDE;
                let mut end = String::new();
                for name_slot in 0..NAME_SLOTS {
                    let packed = slot(base + name_slot);
                    for byte in 0..8 {
                        let b = ((packed >> (byte * 8)) & 0xff) as u8;
                        if b != 0 {
                            end.push(b as char);
                        }
                    }
                }
                let flags = slot(base + NAME_SLOTS);
                if flags & FbwRingFlags::VALID.bits() == 0 {
                    continue;
                }
                let field = |shift: u32| (flags >> shift) & FIELD_MASK;
                eprintln!(
                    "[fbw-census] end={end} committed={} leg={} bridge={} exec_mf={} \
                     effects={} journaled={}",
                    flags & FbwRingFlags::COMMITTED.bits() != 0,
                    // The leg is the top byte, so the shift already isolates
                    // it; the mask keeps the width of the read on the page.
                    (flags >> SHIFT_LEG) & 0xff,
                    flags & FbwRingFlags::BRIDGE.bits() != 0,
                    field(SHIFT_EXEC_MF),
                    field(SHIFT_EFFECTS),
                    field(SHIFT_JOURNAL),
                );
            }
        }
        // must_compile / start_retrace gate tallies (diagnostic). Positional
        // mirror of `majit_metainterp::MC_DIAG_LABELS`; the runner links no
        // majit crate, so a new slot must be appended in both. Appending keeps
        // the existing indices, which is the only edit that cannot silently
        // rename every tally after it.
        if let Ok(mc) = instance.get_typed_func::<u32, u64>(&mut store, "pyre_jit_mc_diag") {
            let labels = [
                "mc_entered",
                "decl_shortcircuit",
                "descr0_skip",
                "busy_skip",
                "FIRED",
                "stack_full",
                "retrace_entered",
                "retrace_bailed",
                "cb_entered",
                "cb_invalidloop",
                "cb_retrace_req",
                "cb_arity_giveup",
                "sbt_entered",
                "sbt_not_faildescr",
                "sbt_no_jct",
                "sbt_no_meta",
                "sbt_cant_trace",
                "sbt_short_vals",
                "caro_backedge",
                "caro_funcentry",
                "caro_no_portal",
                "caro_no_merge_entry",
                "caro_not_tracing",
                "stfe_cell_busy",
                "stfe_dead_token",
                "stfe_tick",
                "toolong_suppressed",
                "wct_declined",
                "cl_hct_giveup",
                "ct_no_front_token",
                "ct_origin_loop_gone",
                "ct_entry_bridge_failed",
                "ct_no_entry_data",
                "ct_not_tracing",
                "ceb_no_loop",
                "ceb_dead_token",
                "ceb_invalidloop",
                "ceb_retrace_req",
                "ceb_backend_panic",
                "ceb_backend_err",
                "abrt_too_long",
                "abrt_bridge",
                "abrt_bad_loop",
                "abrt_escape",
                "abrt_force_qmut",
                "abrt_segmented",
                "abrt_other",
                "giveup_invalidloop",
                "giveup_compileloop_err",
                "giveup_bridge_aborted",
                "bridge_declined_close",
                "bridge_no_targets_close",
                "abort_after_declined",
                "ct_compile_bridge_false",
                "mp_shape_filtered",
                "retrace_mp_untyped",
                "close_hdr_fallback",
                "retrace_arity_giveup",
                "ptp_push",
                "ptp_pop",
                "forced_never_compiled",
                "mst_entered",
                "mst_sync_before_false",
                "mst_live_values_mismatch",
                "stfe_declined_compiled",
                "stfe_declined_tracing",
                "stfe_declined_tracing_stale",
                "bridge_unattempted_close",
                "xloop_close_decision_reached",
                "xloop_close_target_compiled",
                "xloop_close_published",
                "abrt_unclassified_default",
                "unroll_cancelled_invalid_loop",
                "unroll_free_retry_rescued",
                "unroll_free_retry_failed",
                "qmut_deps_simple_loop",
                "qmut_deps_entry_bridge",
                "qmut_deps_blackhole_arm",
                "retrace_close_resumed",
                "guard_resume_bridge_no_handoff",
                "abort_ceiling_banned",
                "abort_ceiling_refused",
                "bridge_abort_permanent",
                "guard_resume_decline_no_state",
                "guard_resume_decline_foreign_jitcode",
                "guard_resume_decline_replay_incomplete",
                "guard_resume_decline_regcount",
                "guard_resume_decline_unreadable_reg",
                "guard_resume_decline_unregistered_jitcode",
                "guard_resume_decline_no_inline_call_site",
                "guard_resume_decline_call_not_jitcode",
                "guard_resume_decline_callee_mismatch",
                "guard_resume_decline_return_slot",
                "guard_resume_decline_reserved_identity",
                "inline_frame_trim_blanked_live_int",
                "confirm_enter_jit_missing_key",
                "bridge_close_head_target_has_virtual_state",
                "peel_less_loop_label_is_const_specialized",
                "segmented_loop_label_is_const_specialized",
            ];
            let mut parts = Vec::new();
            for (i, lbl) in labels.iter().enumerate() {
                parts.push(format!(
                    "{lbl}={}",
                    mc.call(&mut store, i as u32).unwrap_or(0)
                ));
            }
            eprintln!("[jit-stats] mc_diag {}", parts.join(" "));
        }
        // Full-body-walk decline census: the `DispatchError` variant behind
        // every aborted walk, which is what `loops_aborted` counts. Unlike the
        // fixed-slot tallies above this one is name-keyed, so the guest hands
        // back `(ptr << 32) | len` into its own memory and the name is sliced
        // out here — `PYRE_FBW_DEBUG_ABORT`, the native channel for the same
        // attribution, never reaches a guest with no environment.
        if let (Ok(len_fn), Ok(name_fn), Ok(count_fn)) = (
            instance.get_typed_func::<(), u32>(&mut store, "pyre_jit_fbw_census_len"),
            instance.get_typed_func::<u32, u64>(&mut store, "pyre_jit_fbw_census_name"),
            instance.get_typed_func::<u32, u64>(&mut store, "pyre_jit_fbw_census_count"),
        ) {
            let n = len_fn.call(&mut store, ()).unwrap_or(0);
            let mut parts = Vec::new();
            for i in 0..n {
                let packed = name_fn.call(&mut store, i).unwrap_or(0);
                let count = count_fn.call(&mut store, i).unwrap_or(0);
                let (ptr, len) = ((packed >> 32) as usize, (packed & 0xffff_ffff) as usize);
                let data = memory.data(&store);
                let name = data
                    .get(ptr..ptr + len)
                    .and_then(|b| std::str::from_utf8(b).ok())
                    .unwrap_or("<unreadable>");
                parts.push(format!("{name}={count}"));
            }
            if !parts.is_empty() {
                eprintln!("[jit-stats] fbw_census {}", parts.join(" "));
            }
        }
        let guest_jit_execute_count = instance
            .get_typed_func::<(), u64>(&mut store, "pyre_jit_execute_count")
            .and_then(|f| f.call(&mut store, ()))
            .ok();
        let guest_jit_compile_count = instance
            .get_typed_func::<(), u64>(&mut store, "pyre_jit_compile_count")
            .and_then(|f| f.call(&mut store, ()))
            .ok();
        let guest_jit_compile_cache_hits = instance
            .get_typed_func::<(), u64>(&mut store, "pyre_jit_compile_cache_hits")
            .and_then(|f| f.call(&mut store, ()))
            .ok();
        let host = store.data();
        eprintln!(
            "[jit-stats] compiles={} compile_ms={:.1} compile_bytes={} executes={} jit_calls={} jit_call_ms={:.1} \
             jit_call_nested={} jit_call_depth_max={} \
             wasm_ops={} linear_mem={} gc_oldgen={} gc_nursery={} \
             gc_minors={} gc_majors={} heap_live_bytes={} heap_live_count={}",
            host.jit_compile_count,
            host.jit_compile_time_ns as f64 / 1.0e6,
            host.jit_compile_bytes,
            guest_jit_execute_count.unwrap_or(host.jit_execute_count),
            host.jit_call_count,
            host.jit_call_time_ns as f64 / 1.0e6,
            host.jit_call_nested,
            host.jit_call_depth_max,
            wasm_ops.map_or(-1, |n| n as i64),
            lin_mem,
            gc_oldgen,
            gc_nursery,
            gc_minors,
            gc_majors,
            heap_live_bytes,
            heap_live_count,
        );
        if let (Some(materialized), Some(cache_hits)) =
            (guest_jit_compile_count, guest_jit_compile_cache_hits)
        {
            eprintln!(
                "[jit-stats] materialized={} compile_cache_hits={}",
                materialized, cache_hits
            );
        }
        if !host.exec_hist.is_empty() {
            let mut v: Vec<_> = host.exec_hist.iter().collect();
            v.sort_by(|a, b| b.1.cmp(a.1));
            for ((func_id, fail_index), n) in v.into_iter().take(8) {
                eprintln!(
                    "[jit-stats] exec_hist func_id={func_id} fail_index={fail_index} roundtrips={n}"
                );
            }
        }
    }
    probe_call_hist_dump();
    // Emit the sign-stable badness counters under MAJIT_STATS (the same env
    // check.py sets for every backend), so the regression floor gates wasm on
    // the same fields as the native backends. Kept separate from the verbose
    // PYRE_WASM_JIT_STATS block above; which line they land on no longer
    // matters, since check.py merges every [jit-stats] line into one snapshot.
    // This line is the ONLY thing standing between a wasm-only descr-universe
    // regression and a green gate, because `_jit_stats_change` reads a
    // field absent from the current run as zero — the healthy value. So a
    // counter that cannot be read must not resolve to zero: a module that
    // predates these exports would then look perfectly healthy while gating on
    // nothing at all, which is the exact hole this block exists to close.
    if std::env::var_os("MAJIT_STATS").is_some() {
        let mut missing: Vec<&str> = Vec::new();
        let mut counter = |name: &'static str, missing: &mut Vec<&'static str>| match instance
            .get_typed_func::<(), u64>(&mut store, name)
            .and_then(|f| f.call(&mut store, ()))
        {
            Ok(v) => v,
            Err(_) => {
                missing.push(name);
                0
            }
        };
        let loops_compiled = counter("pyre_jit_loops_compiled", &mut missing);
        let bridges_compiled = counter("pyre_jit_bridges_compiled", &mut missing);
        let retraces_compiled = counter("pyre_jit_retraces_compiled", &mut missing);
        let loops_aborted = counter("pyre_jit_loops_aborted", &mut missing);
        let guard_failures = counter("pyre_jit_guard_failures", &mut missing);
        let back_edge_polls = counter("pyre_jit_back_edge_polls", &mut missing);
        let internal_compile_panics = counter("pyre_jit_internal_compile_panics", &mut missing);
        let descr_set_resolved = counter("pyre_jit_descr_set_resolved", &mut missing);
        let descr_set_absent = counter("pyre_jit_descr_set_absent", &mut missing);
        let descr_set_ambiguous = counter("pyre_jit_descr_set_ambiguous", &mut missing);
        let descr_set_stale_absent = counter("pyre_jit_descr_set_stale_absent", &mut missing);
        let field_pos_spec_checked = counter("pyre_jit_field_pos_spec_checked", &mut missing);
        let field_pos_spec_misplaced = counter("pyre_jit_field_pos_spec_misplaced", &mut missing);
        let field_pos_attached_checked =
            counter("pyre_jit_field_pos_attached_checked", &mut missing);
        let field_pos_attached_misplaced =
            counter("pyre_jit_field_pos_attached_misplaced", &mut missing);
        // The rest of the `field_position_census`. `field_pos_unresolved` is
        // the only counter that sees fields whose parent does not contain them:
        // both halves of the `attached_*` pair are counted inside a lookup that
        // must first succeed, so that population is missing from numerator and
        // denominator alike and the pair reads healthy while blind.
        //
        // Asking for them is what makes them gateable. The refusal below fires
        // only for names in THIS list, so a counter nobody asks for is not
        // caught by it — it simply never appears, and `_jit_stats_change` reads
        // an absent field as 0, the healthy value.
        let field_pos_parent_absent = counter("pyre_jit_field_pos_parent_absent", &mut missing);
        let field_pos_parent_empty = counter("pyre_jit_field_pos_parent_empty", &mut missing);
        let field_pos_rederived = counter("pyre_jit_field_pos_rederived", &mut missing);
        let field_pos_unresolved = counter("pyre_jit_field_pos_unresolved", &mut missing);
        // Every tally below the ring — the full-body-walk counts and the
        // admission-gate refusals — reached through the slot-indexed
        // `pyre_fbw_diag` export rather than a counter export of its own.
        //
        // POSITIONAL MIRROR of `pyre_jit_trace::trace::fbw_diag::LABELS` — the
        // same discipline as the mc_diag labels above, since the runner links
        // no pyre crate: a slot added there must be APPENDED here, and the two
        // arrays make this line and the native reader's
        // (`pyre/pyrex/src/lib.rs`) carry the same keys in the same order.
        // Printing a subset is what this replaces: the two readers used to
        // print disjoint halves, so five slots were bumped on native and
        // readable only on wasm, and five more the other way round.
        //
        // The second of each `*_latch` / `*_plain_fallback` pair is the
        // HAZARDOUS SUBSET of the first — `..._new_unjournaled` counts inside
        // `fbw_midbody_latch`, `..._unclean` inside
        // `fbw_escape_plain_fallback` — which is what the `subset/total`
        // fraction this line used to print said. They are named instead of
        // divided because check.py folds every `[jit-stats]` line into one flat
        // `key -> value` map and reads each value as an integer.
        //
        // Five of these are gated: slot 1 (`ROLLED_BACK_WITH_EFFECTS`, walks
        // that ended uncommitted after a residual had already run an
        // irreversible effect), slot 11 (`STORE_JOURNAL_ROLLBACK_FAILED`, the
        // store restores the rollback could not perform — the one journaled
        // effect the walk-end subtraction must not silently absorb), slot 19
        // (`FORITER_ITEM_DROPPED`, a consumed FOR_ITER item no leg handed back,
        // which costs the caller a whole iteration) and the two adoption
        // tallies. They carry the same keys the native backends print,
        // because `_jit_stats_change` compares by name and a name only one
        // backend emits gates nothing on the others; and they must not resolve
        // to 0 when the export is absent, which reads as healthy. Hence the one
        // lookup feeding the refusal below, and hence this line — not the
        // counter line — being where those five keys are emitted now.
        // The value array is built FROM the label array below, so a label
        // added without a slot (or the reverse) is impossible to write rather
        // than a `zip` that silently drops the tail.
        let fbw_labels = [
            "fbw_walks",
            "fbw_rolled_back_with_effects",
            "fbw_midbody_latch",
            "fbw_midbody_latch_new_unjournaled",
            "fbw_escape_plain_fallback",
            "fbw_escape_plain_fallback_unclean",
            "fbw_escape_portal_only",
            "fbw_escape_published_callee_only",
            "fbw_escape_portal_and_published_callee",
            "fbw_force_by_portal",
            "fbw_force_by_callee_only",
            "fbw_store_journal_rollback_failed",
            "fbw_blackhole_adopted_single_frame",
            "fbw_blackhole_adopted_multi_frame",
            "gate_declined_shape",
            "bridge_ec_from_portal_red",
            "bridge_ec_missing",
            "fbw_foriter_item_dropped",
        ];
        let fbw_slots = match instance
            .get_typed_func::<u32, u64>(&mut store, "pyre_fbw_diag")
            .and_then(|f| {
                let mut values = fbw_labels.map(|_| 0u64);
                for (slot, value) in values.iter_mut().enumerate() {
                    *value = f.call(&mut store, slot as u32)?;
                }
                Ok(values)
            }) {
            Ok(values) => values,
            Err(_) => {
                missing.push("pyre_fbw_diag");
                fbw_labels.map(|_| 0u64)
            }
        };
        if !missing.is_empty() {
            eprintln!(
                "pyre-wasm-runner: MAJIT_STATS is set but the module exports no {} — \
                 refusing to report a gated counter as 0, which reads as healthy",
                missing.join(", "),
            );
            std::process::exit(1);
        }
        let mut fbw = String::from("[jit-stats] fbw_diag");
        for (key, value) in fbw_labels.iter().zip(fbw_slots) {
            fbw.push_str(&format!(" {key}={value}"));
        }
        eprintln!("{fbw}");
        if let (Ok(len_fn), Ok(byte_fn)) = (
            instance.get_typed_func::<(), u32>(&mut store, "pyre_jit_last_compile_err_len"),
            instance.get_typed_func::<u32, u32>(&mut store, "pyre_jit_last_compile_err_byte"),
        ) {
            let len = len_fn.call(&mut store, ()).unwrap_or(0);
            if len > 0 {
                let mut bytes = Vec::with_capacity(len as usize);
                for i in 0..len {
                    bytes.push(byte_fn.call(&mut store, i).unwrap_or(0) as u8);
                }
                eprintln!(
                    "[jit-stats] last_compile_err {}",
                    String::from_utf8_lossy(&bytes)
                );
            }
        }
        if let Ok(diag) = instance.get_typed_func::<u32, u64>(&mut store, "pyre_jit_bridge_diag") {
            let cl_entered = diag.call(&mut store, 23).unwrap_or(0);
            let cl_ok = diag.call(&mut store, 24).unwrap_or(0);
            let cl_decl_unsupported = diag.call(&mut store, 25).unwrap_or(0);
            eprintln!(
                "[jit-stats] cl_diag entered={cl_entered} ok={cl_ok} decl_unsupported={cl_decl_unsupported}"
            );
        }
        eprintln!(
            "[jit-stats] loops_compiled={loops_compiled} \
             bridges_compiled={bridges_compiled} \
             retraces_compiled={retraces_compiled} \
             loops_aborted={loops_aborted} \
             guard_failures={guard_failures} \
             back_edge_polls={back_edge_polls} \
             internal_compile_panics={internal_compile_panics} \
             descr_set_resolved={descr_set_resolved} \
             descr_set_absent={descr_set_absent} \
             descr_set_ambiguous={descr_set_ambiguous} \
             descr_set_stale_absent={descr_set_stale_absent} \
             field_pos_spec_checked={field_pos_spec_checked} \
             field_pos_spec_misplaced={field_pos_spec_misplaced} \
             field_pos_attached_checked={field_pos_attached_checked} \
             field_pos_attached_misplaced={field_pos_attached_misplaced} \
             field_pos_parent_absent={field_pos_parent_absent} \
             field_pos_parent_empty={field_pos_parent_empty} \
             field_pos_rederived={field_pos_rederived} \
             field_pos_unresolved={field_pos_unresolved}"
        );
    }
    let packed = match run_result {
        Ok(p) => p,
        Err(e) => {
            // wasm32-unknown-unknown has no stderr, but pyre-wasm's panic hook
            // writes "panicked at …" into linear memory before the trap.
            // Recover the formatted message (the heap String, not the static
            // format template) so the real cause is visible.
            for msg in recover_panic_messages(memory.data(&store)) {
                eprintln!("pyre-wasm-runner: recovered panic: {msg}");
            }
            // Everything the run wrote to fd 2 sits in the guest's buffer until
            // `pyre_take_stderr` drains it, and the success path below is the
            // only place that used to do so — so a trap discarded the whole
            // channel at the one moment it carries the most: the diagnostics a
            // fatal path emits just before aborting. A trap does not poison the
            // store, so the export is still callable; ignore it if the module
            // predates the export or the call itself fails.
            if let Ok(bytes) = take_guest_stderr(&mut store, &instance, &memory)
                && !bytes.is_empty()
            {
                use std::io::Write;
                let _ = std::io::stderr().write_all(&bytes);
            }
            return Err(e);
        }
    };
    let out_ptr = (packed >> 32) as u32;
    let out_len = (packed & 0xffff_ffff) as u32;

    let mut out = vec![0u8; out_len as usize];
    if out_len != 0 {
        memory.read(&store, out_ptr as usize, &mut out)?;
        dealloc.call(&mut store, (out_ptr, out_len))?;
    }
    // The guest has no descriptors, so its fd-2 bytes and its exit status come
    // back through their own exports. Absent on a module predating them, in
    // which case the run keeps the old stdout-only, always-0 behaviour.
    let err_bytes = take_guest_stderr(&mut store, &instance, &memory)?;
    if let Ok(errors) =
        instance.get_typed_func::<(), u64>(&mut store, "pyre_jit_inline_trial_errors")
        && let Ok(packed) = errors.call(&mut store, ())
    {
        let (ptr, len) = ((packed >> 32) as u32, packed as u32);
        if len != 0 {
            let mut bytes = vec![0u8; len as usize];
            memory.read(&store, ptr as usize, &mut bytes)?;
            dealloc.call(&mut store, (ptr, len))?;
            eprintln!(
                "[jit-stats] inline_trial_errors {}",
                String::from_utf8_lossy(&bytes)
            );
        }
    }
    // The per-decline inline-bridge log rides the same string export shape.
    // `bridge_diag` already counts the reasons; this says which crossings they
    // left behind, so it is printed only with the rest of the stats.
    if std::env::var_os("PYRE_WASM_JIT_STATS").is_some()
        && let Ok(declines) =
            instance.get_typed_func::<(), u64>(&mut store, "pyre_jit_inline_declines")
        && let Ok(packed) = declines.call(&mut store, ())
    {
        let (ptr, len) = ((packed >> 32) as u32, packed as u32);
        if len != 0 {
            let mut bytes = vec![0u8; len as usize];
            memory.read(&store, ptr as usize, &mut bytes)?;
            dealloc.call(&mut store, (ptr, len))?;
            eprintln!(
                "[jit-stats] inline_declines {}",
                String::from_utf8_lossy(&bytes)
            );
        }
    }
    // `compile_loop` `Unsupported` reason. Guest eprintln is silent; slots
    // 25/26 do not cover `build_wasm_module`, so a `cl_entered` without
    // `cl_ok` otherwise names no opcode or layout check.
    if std::env::var_os("PYRE_WASM_JIT_STATS").is_some()
        && let Ok(last_err) =
            instance.get_typed_func::<(), u64>(&mut store, "pyre_jit_compile_loop_last_error")
        && let Ok(packed) = last_err.call(&mut store, ())
    {
        let (ptr, len) = ((packed >> 32) as u32, packed as u32);
        if len != 0 {
            let mut bytes = vec![0u8; len as usize];
            memory.read(&store, ptr as usize, &mut bytes)?;
            dealloc.call(&mut store, (ptr, len))?;
            eprintln!(
                "[jit-stats] compile_loop_last_error {}",
                String::from_utf8_lossy(&bytes)
            );
        }
    }
    // `PYRE_FBW_DEBUG_ABORT` cannot select the walker's decline census in the
    // guest, and its `eprintln!` would reach nothing anyway, so the census
    // comes back through its own export and is printed here. Absent on a
    // module predating the export, in which case there is nothing to print.
    if std::env::var_os("PYRE_WASM_FBW_CENSUS").is_some()
        && let Ok(census) = instance.get_typed_func::<(), u64>(&mut store, "pyre_fbw_census")
    {
        let census_result: Result<()> = (|| {
            let packed = census.call(&mut store, ())?;
            let (ptr, clen) = ((packed >> 32) as u32, (packed & 0xffff_ffff) as u32);
            if clen != 0 {
                let mut bytes = vec![0u8; clen as usize];
                memory.read(&store, ptr as usize, &mut bytes)?;
                dealloc.call(&mut store, (ptr, clen))?;
                eprint!("{}", String::from_utf8_lossy(&bytes));
            } else {
                eprintln!("[fbw-census] (no declines recorded)");
            }
            Ok(())
        })();
        if let Err(err) = census_result {
            eprintln!("pyre-wasm-runner: fbw census failed: {err}");
        }
    }
    // The specialization census armed before the run, printed in the shape
    // `pyrex` prints it natively so a fold's `fired` count compares directly
    // across backends. Without this the walker's specializations are
    // unobservable on wasm: the arm lives in the shared walker, but "it must
    // have fired here too" is an inference, and `check.py`'s `spec-folds`
    // gate has nothing to read.
    if std::env::var_os("PYRE_WASM_SPEC_CENSUS").is_some()
        && let Ok(census) = instance.get_typed_func::<(), u64>(&mut store, "pyre_fbw_spec_census")
    {
        let census_result: Result<()> = (|| {
            let packed = census.call(&mut store, ())?;
            let (ptr, clen) = ((packed >> 32) as u32, (packed & 0xffff_ffff) as u32);
            if clen != 0 {
                let mut bytes = vec![0u8; clen as usize];
                memory.read(&store, ptr as usize, &mut bytes)?;
                dealloc.call(&mut store, (ptr, clen))?;
                eprint!("{}", String::from_utf8_lossy(&bytes));
            } else {
                eprintln!("[spec-census] (census not armed)");
            }
            Ok(())
        })();
        if let Err(err) = census_result {
            eprintln!("pyre-wasm-runner: spec census failed: {err}");
        }
    }
    // The per-guard deopt census armed before the run, printed in the shape
    // `pyrex` prints it natively so the two are directly comparable. A bare
    // `guard_failures` total cannot say whether the excess is one guard the
    // guest keeps returning through or a spread of cold ones.
    if std::env::var_os("PYRE_WASM_GUARD_CENSUS").is_some() {
        match instance.get_typed_func::<(), u64>(&mut store, "pyre_jit_guard_census") {
            Ok(census) => {
                let census_result: Result<()> = (|| {
                    let packed = census.call(&mut store, ())?;
                    let (ptr, clen) = ((packed >> 32) as u32, (packed & 0xffff_ffff) as u32);
                    let mut bytes = vec![0u8; clen as usize];
                    if clen != 0 {
                        memory.read(&store, ptr as usize, &mut bytes)?;
                        dealloc.call(&mut store, (ptr, clen))?;
                    }
                    eprintln!("[jit-stats] {}", String::from_utf8_lossy(&bytes));
                    Ok(())
                })();
                if let Err(err) = census_result {
                    eprintln!("pyre-wasm-runner: guard census failed: {err}");
                }
            }
            Err(_) => eprintln!("[jit-stats] guard_census=unexported"),
        }
    }
    // The compile census armed before the run, printed verbatim: the same
    // `[loop-census]` lines a native run puts on stderr. A module predating
    // the export says so rather than reading as a run that compiled nothing,
    // which is what a selfcheck fixture would be failed on.
    if std::env::var_os("PYRE_LOOP_CENSUS").is_some() {
        match instance.get_typed_func::<(), u64>(&mut store, "pyre_loop_census") {
            Ok(census) => {
                let census_result: Result<()> = (|| {
                    let packed = census.call(&mut store, ())?;
                    let (ptr, clen) = ((packed >> 32) as u32, (packed & 0xffff_ffff) as u32);
                    if clen != 0 {
                        let mut bytes = vec![0u8; clen as usize];
                        memory.read(&store, ptr as usize, &mut bytes)?;
                        dealloc.call(&mut store, (ptr, clen))?;
                        eprint!("{}", String::from_utf8_lossy(&bytes));
                    }
                    Ok(())
                })();
                if let Err(err) = census_result {
                    eprintln!("pyre-wasm-runner: loop census failed: {err}");
                }
            }
            Err(_) => eprintln!("pyre-wasm-runner: loop_census=unexported"),
        }
    }
    if std::env::var_os("PYRE_WASM_TRACE_ENTRY_CENSUS").is_some() {
        match instance.get_typed_func::<(), u64>(&mut store, "pyre_jit_trace_entry_census") {
            Ok(census) => {
                let census_result: Result<()> = (|| {
                    let packed = census.call(&mut store, ())?;
                    let (ptr, clen) = ((packed >> 32) as u32, (packed & 0xffff_ffff) as u32);
                    if clen != 0 {
                        let mut bytes = vec![0u8; clen as usize];
                        memory.read(&store, ptr as usize, &mut bytes)?;
                        dealloc.call(&mut store, (ptr, clen))?;
                        eprint!("{}", String::from_utf8_lossy(&bytes));
                    }
                    Ok(())
                })();
                if let Err(err) = census_result {
                    eprintln!("pyre-wasm-runner: trace entry census failed: {err}");
                }
            }
            Err(_) => eprintln!("[trace-entry-census] unexported"),
        }
    }
    let exit_code = instance
        .get_typed_func::<(), i32>(&mut store, "pyre_exit_code")
        .and_then(|f| f.call(&mut store, ()))
        .unwrap_or(0);
    if len != 0 {
        dealloc.call(&mut store, (in_ptr, len))?;
    }

    use std::io::Write;
    std::io::stdout().write_all(&out)?;
    std::io::stdout().flush()?;
    if !err_bytes.is_empty() {
        std::io::stderr().write_all(&err_bytes)?;
    }
    startup_lap("dealloc+stdout");
    // Exit without running the wasmtime `Store`/`Module`/`Engine` destructors.
    // Dropping them frees ~40MB of mapped code + linear memory that the OS
    // reclaims on process exit anyway; on a short run that teardown is ~0.2s —
    // the dominant fixed startup tax, larger than the module load and far
    // larger than trace compilation. stdout is already flushed above and the
    // diagnostic stats print to (unbuffered) stderr, so nothing is lost.
    // `PYRE_WASM_FULL_TEARDOWN=1` restores the drops for leak diagnostics.
    if std::env::var_os("PYRE_WASM_FULL_TEARDOWN").is_none() {
        std::io::stderr().flush().ok();
        std::process::exit(exit_code);
    }
    Ok(exit_code)
}

/// Load the main module, using a compiled `<module>.cwasm` cache to skip
/// Cranelift recompilation of the main module on every process start.
///
/// The cache is the engine's own `Module::serialize` artifact, so it is only
/// usable by a byte-compatible engine build; `Module::deserialize_file`
/// validates that and errors on mismatch. `deserialize_file` runs trusted
/// precompiled native code and the `.cwasm` is otherwise independent of the
/// `.wasm` contents, so the cache is bound to the exact bytes it was produced
/// from: a sidecar `<module>.cwasm.sha256` records the SHA-256 of those bytes
/// along with the size and mtime they were read at, and the cache is
/// deserialized only when the module still answers to one of the two. A
/// rebuilt module or a pre-placed `.cwasm` therefore recompiles instead of
/// running a stale or untrusted artifact. Set `PYRE_WASM_NO_CACHE` to bypass
/// the cache entirely.
///
/// `variant` separates the artifacts of engine configurations that emit
/// different code for the same module; see [`cache_variant`].
fn load_main_module(engine: &Engine, module_path: &Path, variant: &str) -> Result<Module> {
    let cache_disabled = std::env::var_os("PYRE_WASM_NO_CACHE").is_some();
    probe_call_hist_note_module(module_path);
    let cache_path = cache_path_for(module_path, variant);
    let key_path = cache_key_path_for(module_path, variant);
    let stamp = file_stamp(module_path);

    // Reading and hashing the module is 28MB of work on a path that then
    // throws the bytes away, and every process start pays it. The sidecar
    // therefore also records the size and mtime the hash was taken from: while
    // those still describe the file, the bytes are the bytes that were hashed
    // and the hash is already known.
    if !cache_disabled
        && let Some(stamp) = stamp.as_deref()
        && cache_key_stamp_matches(&key_path, stamp)
        && let Some(module) = deserialize_cache(engine, &cache_path)
    {
        return Ok(module);
    }

    let wasm_bytes = std::fs::read(module_path)
        .with_context(|| format!("read wasm module {}", module_path.display()))?;
    let hash = wasm_content_hash(&wasm_bytes);

    if !cache_disabled
        && cache_key_matches(&key_path, &hash)
        && let Some(module) = deserialize_cache(engine, &cache_path)
    {
        // Same bytes under a stamp the sidecar does not name -- a rebuild that
        // reproduced the module, or a checkout that reset the mtime. Record
        // the stamp so the next run reaches the cache without hashing.
        write_cache_key(&key_path, &hash, stamp.as_deref());
        return Ok(module);
    }

    let module = Module::new(engine, &wasm_bytes[..])
        .with_context(|| format!("load wasm module {}", module_path.display()))?;
    if !cache_disabled && let Ok(bytes) = module.serialize() {
        // Best-effort: a failed cache write only forgoes the speedup. Publish
        // the artifact through a per-process temporary and rename it into
        // place, then write the key: a rename swaps the directory entry, so a
        // concurrent run either deserializes the previous artifact whole or
        // this one whole. Writing the file in place instead would truncate it
        // under whoever has it mapped, and the key would name a `.cwasm` that
        // is only half there.
        let staged = staged_cache_path_for(&cache_path);
        if std::fs::write(&staged, bytes).is_ok() {
            if std::fs::rename(&staged, &cache_path).is_ok() {
                write_cache_key(&key_path, &hash, stamp.as_deref());
            } else {
                let _ = std::fs::remove_file(&staged);
            }
        }
    }
    Ok(module)
}

/// Deserialize the cached artifact, or nothing if it cannot be trusted.
///
/// SAFETY: the artifact was produced by this runner's own engine via
/// `Module::serialize`; `deserialize_file` re-checks engine/version
/// compatibility and returns Err (not UB) if it cannot be trusted. The caller
/// has further proved, by hash or by stamp, that it was compiled from the
/// bytes the module path currently holds.
fn deserialize_cache(engine: &Engine, cache_path: &Path) -> Option<Module> {
    unsafe { Module::deserialize_file(engine, cache_path) }.ok()
}

/// The suffix separating the compiled artifacts of engine configurations that
/// emit different code for the same module. Both knobs below instrument every
/// guest instruction, so their modules are neither interchangeable with the
/// plain one nor with each other; sharing one cache path makes each switch
/// between modes reject the stored artifact, recompile the whole module and
/// rewrite the file the other mode just wrote.
fn cache_variant(meter_fuel: bool, epoch_interruption: bool) -> &'static str {
    match (meter_fuel, epoch_interruption) {
        (false, false) => ".speed-and-size",
        (true, false) => ".speed-and-size.fuel",
        (false, true) => ".speed-and-size.epoch",
        (true, true) => ".speed-and-size.fuel.epoch",
    }
}

#[test]
fn optimized_cache_variants_do_not_reuse_default_opt_level_artifacts() {
    let mut variants = Vec::new();
    for fuel in [false, true] {
        for epoch in [false, true] {
            let variant = cache_variant(fuel, epoch);
            assert!(variant.contains("speed-and-size"));
            assert!(!variants.contains(&variant));
            variants.push(variant);
        }
    }
}

/// `<module><variant>.cwasm` next to the module (full name kept, so
/// `pyre_wasm.wasm-host.wasm` → `pyre_wasm.wasm-host.wasm.cwasm`).
fn cache_path_for(module_path: &Path, variant: &str) -> PathBuf {
    let mut s = module_path.as_os_str().to_owned();
    s.push(variant);
    s.push(".cwasm");
    PathBuf::from(s)
}

/// Sidecar recording the SHA-256 of the `.wasm` its `.cwasm` was compiled from.
fn cache_key_path_for(module_path: &Path, variant: &str) -> PathBuf {
    let mut s = cache_path_for(module_path, variant).into_os_string();
    s.push(".sha256");
    PathBuf::from(s)
}

/// Where a `.cwasm` is written before being renamed over the real one. The pid
/// keeps two runs publishing the same variant from staging onto each other.
fn staged_cache_path_for(cache_path: &Path) -> PathBuf {
    let mut s = cache_path.as_os_str().to_owned();
    s.push(format!(".{}.tmp", std::process::id()));
    PathBuf::from(s)
}

/// Hex SHA-256 of the module bytes, used as the cache key.
fn wasm_content_hash(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;
    let digest = Sha256::digest(bytes);
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Size and mtime of the module file, in the spelling the key sidecar records.
///
/// Nothing distinguishes two builds that landed in the same nanosecond at the
/// same length, which is why this only ever admits a hash that was already
/// taken: it says the file has not been rewritten since, not what is in it.
fn file_stamp(module_path: &Path) -> Option<String> {
    let meta = std::fs::metadata(module_path).ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?;
    Some(format!(
        "{} {} {}",
        meta.len(),
        mtime.as_secs(),
        mtime.subsec_nanos()
    ))
}

/// The sidecar's hash line, then its stamp line if it has one.
fn read_cache_key(key_path: &Path) -> Option<(String, Option<String>)> {
    let text = std::fs::read_to_string(key_path).ok()?;
    let mut lines = text.lines();
    let hash = lines.next()?.trim().to_string();
    let stamp = lines.next().map(|line| line.trim().to_string());
    Some((hash, stamp))
}

fn write_cache_key(key_path: &Path, hash: &str, stamp: Option<&str>) {
    let mut text = String::from(hash);
    if let Some(stamp) = stamp {
        text.push('\n');
        text.push_str(stamp);
    }
    let _ = std::fs::write(key_path, text);
}

/// The cache is usable only if its key sidecar exists and matches `hash`.
fn cache_key_matches(key_path: &Path, hash: &str) -> bool {
    matches!(read_cache_key(key_path), Some((k, _)) if k == hash)
}

/// The cache is reachable without hashing only if the sidecar names the stamp
/// the module file still carries.
fn cache_key_stamp_matches(key_path: &Path, stamp: &str) -> bool {
    matches!(read_cache_key(key_path), Some((_, Some(s))) if s == stamp)
}

fn build_linker(engine: &Engine) -> Result<Linker<Host>> {
    let mut linker = Linker::new(engine);

    linker.func_wrap(
        "majit_host",
        "jit_compile_wasm",
        |mut caller: Caller<'_, Host>, bytes_ptr: u32, bytes_len: u32| -> u32 {
            match jit_compile(&mut caller, bytes_ptr, bytes_len) {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("[jit_compile_wasm] {e:?}");
                    0
                }
            }
        },
    )?;

    linker.func_wrap(
        "majit_host",
        "jit_replace_wasm",
        |mut caller: Caller<'_, Host>, func_id: u32, bytes_ptr: u32, bytes_len: u32| -> u32 {
            match jit_replace(&mut caller, func_id, bytes_ptr, bytes_len) {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("[jit_replace_wasm] {e:?}");
                    0
                }
            }
        },
    )?;

    linker.func_wrap(
        "majit_host",
        "jit_execute_wasm",
        |mut caller: Caller<'_, Host>, func_id: u32, frame_ptr: u32| -> u32 {
            match jit_execute(&mut caller, func_id, frame_ptr) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[jit_execute_wasm] {e:?}");
                    0
                }
            }
        },
    )?;

    linker.func_wrap(
        "majit_host",
        "jit_free_wasm",
        |mut caller: Caller<'_, Host>, func_id: u32| {
            let _ = majit_backend_wasm_host::free(&mut caller, func_id);
        },
    )?;

    // Reflective residual-call trampoline for the *recording* / blackhole
    // path. The compiled trace reaches residual targets through `env.jit_call`
    // on the child module; the in-module metainterp cannot reflect a function's
    // wasm type to build a matching `call_indirect`, so it routes residual
    // calls here instead. Reuses the same call-area protocol and signature
    // reflection as `jit_call_trampoline` (the child's `env.jit_call`), so a
    // residual target whose real signature is not the uniform `(i64…) -> i64`
    // (e.g. the void `set_current_exception_fn`, or the `-> i64`
    // `store_subscr_fn` invoked in a void context) is coerced correctly
    // rather than trapping on an indirect-call type mismatch.
    linker.func_wrap(
        "majit_host",
        "jit_call_host",
        |mut caller: Caller<'_, Host>, frame_ptr: u32| -> Result<()> {
            jit_call_trampoline(&mut caller, frame_ptr, CALL_RESULT_OFS as u32, "host")
        },
    )?;

    linker.func_wrap(
        "env",
        "jit_func_sig",
        |mut caller: Caller<'_, Host>, slot: i32| -> i64 {
            majit_backend_wasm_host::jit_func_sig_of_slot(&mut caller, slot)
        },
    )?;

    // Host-filesystem imports for the wasm-host build's module loader. The wasm32
    // module has no filesystem of its own; these serve module source from the
    // host's real stdlib (`$PYRE_STDLIB`). See `pyre-wasm`'s `host_fs_provider`.
    linker.func_wrap(
        "pyre_host",
        "host_stdlib_root",
        |mut caller: Caller<'_, Host>, buf_ptr: u32, buf_cap: u32| -> i64 {
            host_stdlib_root(&mut caller, buf_ptr, buf_cap)
        },
    )?;
    linker.func_wrap(
        "pyre_host",
        "host_is_dir",
        |mut caller: Caller<'_, Host>, path_ptr: u32, path_len: u32| -> u32 {
            match host_path(&mut caller, path_ptr, path_len) {
                Some(p) => PathBuf::from(p).is_dir() as u32,
                None => 0,
            }
        },
    )?;
    linker.func_wrap(
        "pyre_host",
        "host_file_size",
        |mut caller: Caller<'_, Host>, path_ptr: u32, path_len: u32| -> i64 {
            match host_path(&mut caller, path_ptr, path_len) {
                Some(p) => match std::fs::metadata(&p) {
                    Ok(m) if m.is_file() => m.len() as i64,
                    _ => -1,
                },
                None => -1,
            }
        },
    )?;
    linker.func_wrap(
        "pyre_host",
        "host_read",
        |mut caller: Caller<'_, Host>,
         path_ptr: u32,
         path_len: u32,
         buf_ptr: u32,
         buf_cap: u32|
         -> i64 { host_read(&mut caller, path_ptr, path_len, buf_ptr, buf_cap) },
    )?;
    linker.func_wrap(
        "pyre_host",
        "host_list_dir",
        |mut caller: Caller<'_, Host>,
         path_ptr: u32,
         path_len: u32,
         buf_ptr: u32,
         buf_cap: u32|
         -> i64 { host_list_dir(&mut caller, path_ptr, path_len, buf_ptr, buf_cap) },
    )?;

    linker.func_wrap(
        "pyre_host",
        "host_cwd",
        |mut caller: Caller<'_, Host>, buf_ptr: u32, buf_cap: u32| -> i64 {
            host_cwd(&mut caller, buf_ptr, buf_cap)
        },
    )?;
    linker.func_wrap(
        "pyre_host",
        "host_random",
        |mut caller: Caller<'_, Host>, buf_ptr: u32, buf_cap: u32| -> i32 {
            host_random(&mut caller, buf_ptr, buf_cap)
        },
    )?;

    // Host clock imports.  wasm32 has neither a `SystemTime` nor an `Instant`,
    // so `time` -- and every stdlib module that imports it -- reads the clock
    // here.  See `pyre-wasm`'s `host_clock`.
    linker.func_wrap("pyre_host", "host_time_ns", || -> i64 { host_time_ns() })?;
    linker.func_wrap("pyre_host", "host_monotonic_ns", || -> i64 {
        host_monotonic_ns()
    })?;
    linker.func_wrap("pyre_host", "host_sleep_ns", |nanos: i64| {
        host_sleep_ns(nanos)
    })?;

    Ok(linker)
}

/// `pyre_host.host_random`: fill the guest buffer with host entropy.
///
/// wasm32-unknown-unknown has no OS entropy of its own, so `os.urandom`,
/// `secrets` and the string hash key all end up here.
fn host_random(caller: &mut Caller<'_, Host>, buf_ptr: u32, buf_cap: u32) -> i32 {
    let Some(memory) = caller.data().jit.memory else {
        return -1;
    };
    let mut bytes = vec![0u8; buf_cap as usize];
    if getrandom::fill(&mut bytes).is_err() {
        return -1;
    }
    match memory.write(&mut *caller, buf_ptr as usize, &bytes) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// `pyre_host.host_cwd`: write the runner's working directory into the guest
/// buffer, reporting its length whether or not it fitted.
fn host_cwd(caller: &mut Caller<'_, Host>, buf_ptr: u32, buf_cap: u32) -> i64 {
    let Some(memory) = caller.data().jit.memory else {
        return -1;
    };
    let Ok(cwd) = std::env::current_dir() else {
        return -1;
    };
    let Some(bytes) = host_path_to_guest(&cwd) else {
        return -1;
    };
    if bytes.len() > buf_cap as usize {
        return bytes.len() as i64;
    }
    match memory.write(&mut *caller, buf_ptr as usize, &bytes) {
        Ok(()) => bytes.len() as i64,
        Err(_) => -1,
    }
}

/// `pyre_host.host_time_ns`: nanoseconds since the unix epoch.
///
/// The guest has no `SystemTime` of its own -- reading one panics there --
/// so the wall clock it reports is this one.
fn host_time_ns() -> i64 {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => d.as_nanos().min(i64::MAX as u128) as i64,
        Err(_) => -1,
    }
}

/// `pyre_host.host_monotonic_ns`: nanoseconds since the runner started.
///
/// The origin is taken at the first call and kept for the run, which is all
/// `time.monotonic()` asks of it: only differences are meaningful, and they
/// have to not go backwards.
fn host_monotonic_ns() -> i64 {
    static ORIGIN: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    ORIGIN
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_nanos()
        .min(i64::MAX as u128) as i64
}

/// `pyre_host.host_sleep_ns`: block the guest for `nanos` nanoseconds.
fn host_sleep_ns(nanos: i64) {
    if nanos > 0 {
        std::thread::sleep(std::time::Duration::from_nanos(nanos as u64));
    }
}

/// Read a host path argument out of wasm linear memory as the filesystem name
/// it spells.
///
/// The guest sends the bytes its own `OsStr` is. `guest_path_to_host` keeps
/// Unix bytes intact and decodes the absolute-path transport spelling used on
/// Windows; see `host_path_to_guest`.
fn host_path(
    caller: &mut Caller<'_, Host>,
    path_ptr: u32,
    path_len: u32,
) -> Option<std::ffi::OsString> {
    let memory = caller.data().jit.memory?;
    let mut bytes = vec![0u8; path_len as usize];
    memory.read(&*caller, path_ptr as usize, &mut bytes).ok()?;
    guest_path_to_host(bytes)
}

/// `pyre_host.host_stdlib_root`: write `$PYRE_STDLIB` into the wasm buffer.
fn host_stdlib_root(caller: &mut Caller<'_, Host>, buf_ptr: u32, buf_cap: u32) -> i64 {
    let Some(root) = caller.data().stdlib_root.clone() else {
        return -1;
    };
    let Some(bytes) = host_path_to_guest(Path::new(&root)) else {
        return -1;
    };
    if bytes.len() > buf_cap as usize {
        // Report the needed length without writing; the caller can retry.
        return bytes.len() as i64;
    }
    let Some(memory) = caller.data().jit.memory else {
        return -1;
    };
    if memory
        .write(&mut *caller, buf_ptr as usize, &bytes)
        .is_err()
    {
        return -1;
    }
    bytes.len() as i64
}

/// `pyre_host.host_read`: read the host file into the wasm-provided buffer.
fn host_read(
    caller: &mut Caller<'_, Host>,
    path_ptr: u32,
    path_len: u32,
    buf_ptr: u32,
    buf_cap: u32,
) -> i64 {
    let Some(path) = host_path(caller, path_ptr, path_len) else {
        return -1;
    };
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(_) => return -1,
    };
    let n = data.len().min(buf_cap as usize);
    let Some(memory) = caller.data().jit.memory else {
        return -1;
    };
    if memory
        .write(&mut *caller, buf_ptr as usize, &data[..n])
        .is_err()
    {
        return -1;
    }
    n as i64
}

/// `pyre_host.host_list_dir`: write the host directory's entry names into the
/// wasm buffer, NUL-separated, and answer the whole list's byte length.
///
/// The length is reported whether or not it fitted, so an undersized buffer is
/// a retry rather than a truncation — the same protocol `host_stdlib_root`
/// uses. Names travel as the filesystem's own bytes, NUL being the one byte no
/// name can contain, so one with no UTF-8 spelling still reaches the guest and
/// is decoded there the way every other target decodes a filename.
fn host_list_dir(
    caller: &mut Caller<'_, Host>,
    path_ptr: u32,
    path_len: u32,
    buf_ptr: u32,
    buf_cap: u32,
) -> i64 {
    let Some(path) = host_path(caller, path_ptr, path_len) else {
        return -1;
    };
    let Ok(entries) = std::fs::read_dir(&path) else {
        return -1;
    };
    let mut packed: Vec<u8> = Vec::new();
    for entry in entries {
        // An error raised part-way through the walk leaves a listing that is
        // short without saying so, which the guest would report as the whole
        // directory.
        let Ok(entry) = entry else {
            return -1;
        };
        if !packed.is_empty() {
            packed.push(0);
        }
        packed.extend_from_slice(entry.file_name().as_encoded_bytes());
    }
    if packed.len() > buf_cap as usize {
        return packed.len() as i64;
    }
    let Some(memory) = caller.data().jit.memory else {
        return -1;
    };
    if memory
        .write(&mut *caller, buf_ptr as usize, &packed)
        .is_err()
    {
        return -1;
    }
    packed.len() as i64
}

/// Compile and instantiate a JIT-emitted trace module, sharing the main
/// module's linear memory and wiring the `jit_call` trampoline.
fn jit_compile_trace(
    caller: &mut Caller<'_, Host>,
    bytes_ptr: u32,
    bytes_len: u32,
    kind: &str,
) -> Result<(Table, Func, Option<Func>, Module)> {
    caller.data_mut().jit_compile_count += 1;
    let memory = caller
        .data()
        .jit
        .memory
        .context("main memory not initialized")?;
    let table = caller
        .data()
        .jit
        .table
        .context("main table not initialized")?;

    let mut bytes = vec![0u8; bytes_len as usize];
    memory
        .read(&*caller, bytes_ptr as usize, &mut bytes)
        .context("read trace module bytes")?;

    let engine = caller.engine().clone();
    caller.data_mut().jit_compile_bytes += bytes.len() as u64;
    let compile_start = std::time::Instant::now();
    let module_result = Module::new(&engine, &bytes);
    let compile_elapsed = compile_start.elapsed();
    caller.data_mut().jit_compile_time_ns += compile_elapsed.as_nanos();
    // Diagnostic: PYRE_WASM_COMPILE_CENSUS=1 reports each cranelift compile
    // separately. The aggregate on the stats line cannot say what a re-emitted
    // owner costs against a first compile of the same trace, nor whether the
    // per-module cost is linear in the bytes, and both decide whether the
    // answer to compile latency is emitting less or emitting it elsewhere.
    if std::env::var_os("PYRE_WASM_COMPILE_CENSUS").is_some() {
        eprintln!(
            "[compile-census] kind={kind} bytes={} ms={:.3}",
            bytes.len(),
            compile_elapsed.as_secs_f64() * 1e3
        );
    }
    let module = match module_result {
        Ok(m) => m,
        Err(e) => {
            if std::env::var_os("PYRE_WASM_DUMP_BAD_TRACE").is_some() {
                let path = "/tmp/pyre_bad_trace.wasm";
                let _ = std::fs::write(path, &bytes);
                eprintln!("[jit_compile_wasm] dumped {} bytes to {path}", bytes.len());
                match wasmprinter::print_bytes(&bytes) {
                    Ok(wat) => eprintln!("--- WAT ---\n{wat}\n--- /WAT ---"),
                    Err(pe) => eprintln!("[jit_compile_wasm] wat print failed: {pe}"),
                }
            }
            return Err(e).context("compile trace module");
        }
    };

    let (trace, trace_wide, instance) =
        majit_backend_wasm_host::instantiate(caller, &module, |caller, frame, offset| {
            jit_call_trampoline(caller, frame, offset, "trace")
        })?;
    if std::env::var_os("PYRE_WASM_DUMP_ALL_TRACES").is_some() {
        let trace_id = instance
            .get_global(&mut *caller, "trace_entry_census_id")
            .and_then(|global| global.get(&mut *caller).i64())
            .map(|id| format!(" trace_id={id}"))
            .unwrap_or_default();
        match wasmprinter::print_bytes(&bytes) {
            Ok(wat) => eprintln!(
                "=== trace module{trace_id} ({} bytes) ===\n{wat}",
                bytes.len()
            ),
            Err(pe) => eprintln!("[jit_compile_wasm] wat print failed: {pe}"),
        }
    }
    Ok((table, trace, trace_wide, module))
}
/// Compile and instantiate a trace, then append its export to the trace table.
fn jit_compile(caller: &mut Caller<'_, Host>, bytes_ptr: u32, bytes_len: u32) -> Result<u32> {
    let (_table, trace, trace_wide, module) =
        jit_compile_trace(caller, bytes_ptr, bytes_len, "compile")?;
    let slot = majit_backend_wasm_host::publish(caller, trace, trace_wide)?;
    record_trace_module(caller, module, slot);
    Ok(slot)
}
/// Name a freshly installed trace module by its table slot, so an epoch sample
/// taken inside it can say which trace it was in. Only the profiling mode keeps
/// the list; every other run leaves it empty and pays nothing.
fn record_trace_module(caller: &mut Caller<'_, Host>, module: Module, slot: u32) {
    if caller.data().guest_profiler.is_none() {
        return;
    }
    let host = caller.data_mut();
    host.trace_modules.retain(|(_, s)| *s != slot);
    host.trace_modules.push((module, slot));
}

/// Compile and instantiate a trace, then replace an existing trace-table slot.
fn jit_replace(
    caller: &mut Caller<'_, Host>,
    func_id: u32,
    bytes_ptr: u32,
    bytes_len: u32,
) -> Result<u32> {
    if (func_id as u64) < caller.data().jit.trace_base {
        return Err(Error::msg("cannot replace a guest function"));
    }
    let (_table, trace, trace_wide, module) =
        jit_compile_trace(caller, bytes_ptr, bytes_len, "replace")?;
    let slot = majit_backend_wasm_host::replace(caller, func_id, trace, trace_wide)?;
    record_trace_module(caller, module, slot);
    Ok(slot)
}
/// Run a previously compiled trace, returning its guard-exit index.
fn jit_execute(caller: &mut Caller<'_, Host>, func_id: u32, frame_ptr: u32) -> Result<u32> {
    caller.data_mut().jit_execute_count += 1;
    let ret = majit_backend_wasm_host::execute(caller, func_id, frame_ptr)?;
    // Diagnostic: record which (trace, guard-exit fail_index) round-tripped.
    if std::env::var_os("PYRE_WASM_EXEC_TRACE").is_some()
        && let Some(mem) = caller.data().jit.memory
    {
        let fail_index = read_u32(&mem, &*caller, frame_ptr as usize);
        *caller
            .data_mut()
            .exec_hist
            .entry((func_id, fail_index))
            .or_insert(0) += 1;
    }
    Ok(ret)
}
/// Name a call-area FUNC field that indexes no live function, once per value.
///
/// The zero sentinel above is the only slot that legitimately has no function
/// behind it. Any other index that misses leaves nothing to call, and the 0
/// written in its place is indistinguishable from a null Ref the callee
/// returned -- so a call lowered against the wrong operand is answered with a
/// plausible value instead of a trap, and only surfaces much later as a wrong
/// result. Say the slot out loud here so it surfaces at the call instead.
fn report_dead_call_slot(func_ptr: u32) {
    static SEEN: parking_lot::Mutex<Option<std::collections::BTreeSet<u32>>> =
        parking_lot::Mutex::new(None);
    if SEEN
        .lock()
        .get_or_insert_with(Default::default)
        .insert(func_ptr)
    {
        eprintln!(
            "[warn] residual call area names function table slot {func_ptr}, \
             which holds no function; the call answers 0"
        );
    }
}

/// Dispatch a residual call requested by a running trace.
// PROBE(PYRE_WASM_CALL_HIST): temporary per-callee crossing histogram.
static PROBE_CALL_HIST: parking_lot::Mutex<
    Option<std::collections::BTreeMap<(&'static str, u32), u64>>,
> = parking_lot::Mutex::new(None);

/// Whether `PYRE_WASM_CALL_HIST` was set, read once.
///
/// `jit_call_trampoline` runs per residual call -- 15168 of them on
/// fib_recursive -- and `std::env::var_os` takes the process-wide environment
/// lock and scans it, so reading the variable there put that cost on every
/// crossing whether or not the probe was wanted.
fn probe_call_hist_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_WASM_CALL_HIST").is_some())
}

/// The module the histogram's slots index into, recorded so the dump can name
/// them. Only the path is kept: resolving costs a re-read and a parse of the
/// whole module, which is worth paying once at exit and never when the probe
/// is off.
static PROBE_CALL_HIST_MODULE: parking_lot::Mutex<Option<PathBuf>> = parking_lot::Mutex::new(None);

fn probe_call_hist_note_module(module_path: &Path) {
    if !probe_call_hist_enabled() {
        return;
    }
    *PROBE_CALL_HIST_MODULE.lock() = Some(module_path.to_path_buf());
}

/// Symbol for each function-table slot the histogram can report.
///
/// A slot is not a function index: the table is filled by the element
/// segments, so the slot is resolved to a function index through them and only
/// then to a name. Both index spaces count imports first, so the function
/// index an element segment holds is the one the name section names, with no
/// adjustment.
///
/// Every failure is silent and yields an empty map -- the caller falls back to
/// printing the bare slot, which is what this probe did before. A diagnostic
/// that cannot name a symbol is still worth its counts.
fn probe_call_hist_slot_names() -> std::collections::BTreeMap<u32, String> {
    use wasmparser::{ElementItems, ElementKind, Name, Operator, Payload};

    let mut out = std::collections::BTreeMap::new();
    let path = PROBE_CALL_HIST_MODULE.lock().clone();
    let Some(path) = path else { return out };
    let Ok(bytes) = std::fs::read(&path) else {
        return out;
    };

    let mut slot_to_func: std::collections::BTreeMap<u32, u32> = Default::default();
    let mut func_to_name: std::collections::BTreeMap<u32, String> = Default::default();
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        match payload {
            Ok(Payload::ElementSection(reader)) => {
                for element in reader {
                    let Ok(element) = element else { continue };
                    // Only an active segment puts entries in the table; a
                    // passive or declared one is never indexed by a call.
                    let ElementKind::Active { offset_expr, .. } = element.kind else {
                        continue;
                    };
                    // The table offset every real segment uses is a plain
                    // `i32.const`; anything else is not a shape this resolves.
                    let mut ops = offset_expr.get_operators_reader().into_iter();
                    let base = match ops.next() {
                        Some(Ok(Operator::I32Const { value })) => value as u32,
                        _ => continue,
                    };
                    let ElementItems::Functions(funcs) = element.items else {
                        continue;
                    };
                    for (i, func) in funcs.into_iter().enumerate() {
                        let Ok(func) = func else { continue };
                        slot_to_func.insert(base + i as u32, func);
                    }
                }
            }
            Ok(Payload::CustomSection(c)) if c.name() == "name" => {
                let reader = wasmparser::NameSectionReader::new(wasmparser::BinaryReader::new(
                    c.data(),
                    c.data_offset(),
                ));
                for subsection in reader {
                    let Ok(Name::Function(map)) = subsection else {
                        continue;
                    };
                    for naming in map {
                        let Ok(naming) = naming else { continue };
                        func_to_name.insert(
                            naming.index,
                            rustc_demangle::demangle(naming.name).to_string(),
                        );
                    }
                }
            }
            _ => {}
        }
    }

    for (slot, func) in slot_to_func {
        if let Some(name) = func_to_name.get(&func) {
            out.insert(slot, name.clone());
        }
    }
    out
}

pub(crate) fn probe_call_hist_dump() {
    if !probe_call_hist_enabled() {
        return;
    }
    let g = PROBE_CALL_HIST.lock();
    let Some(m) = g.as_ref() else { return };
    let total: u64 = m.values().sum();
    let mut v: Vec<_> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    eprintln!("[probe] call_hist total={total} distinct={}", v.len());
    let names = probe_call_hist_slot_names();
    for ((source, slot), n) in v.into_iter().take(15) {
        let pct = if total > 0 {
            *n as f64 * 100.0 / total as f64
        } else {
            0.0
        };
        let name = names.get(slot).map(String::as_str).unwrap_or("?");
        eprintln!("[probe] call_hist src={source} slot={slot} n={n} pct={pct:.1} name={name}");
    }
}

/// Time every residual crossing, including the paths that return early, so the
/// share of a run spent leaving the guest is measured rather than inferred from
/// `jit_call_count` alone.
///
/// The frame's own cost is its sample less whatever its nested crossings took
/// (see `Host::jit_call_time_ns`). The parent's accumulator lives in a local
/// for the duration, so the Rust call stack is the stack this needs.
fn jit_call_trampoline(
    caller: &mut Caller<'_, Host>,
    frame_ptr: u32,
    call_area_ofs: u32,
    source: &'static str,
) -> Result<()> {
    let host = caller.data_mut();
    let outer_child_ns = std::mem::take(&mut host.jit_call_child_ns);
    host.jit_call_nested += u64::from(host.jit_call_depth != 0);
    host.jit_call_depth += 1;
    host.jit_call_depth_max = host.jit_call_depth_max.max(host.jit_call_depth);
    let entered = std::time::Instant::now();
    let r = jit_call_trampoline_inner(caller, frame_ptr, call_area_ofs, source);
    let elapsed = entered.elapsed().as_nanos();
    let host = caller.data_mut();
    host.jit_call_depth -= 1;
    host.jit_call_time_ns += elapsed.saturating_sub(host.jit_call_child_ns);
    host.jit_call_child_ns = outer_child_ns + elapsed;
    r
}

fn jit_call_trampoline_inner(
    caller: &mut Caller<'_, Host>,
    frame_ptr: u32,
    call_area_ofs: u32,
    source: &'static str,
) -> Result<()> {
    caller.data_mut().jit_call_count += 1;
    majit_backend_wasm_host::residual_call(caller, frame_ptr, call_area_ofs, |slot, live| {
        if probe_call_hist_enabled() {
            *PROBE_CALL_HIST
                .lock()
                .get_or_insert_with(Default::default)
                .entry((source, slot))
                .or_insert(0) += 1;
        }
        if !live && slot != 0 {
            report_dead_call_slot(slot);
        }
    })
}
/// Drain the guest's fd-2 buffer through `pyre_take_stderr`.
///
/// The guest has no descriptors, so everything it writes to fd 2 accumulates in
/// linear memory until this export hands it over. Returns empty on a module
/// predating the export, which keeps the old stdout-only behaviour.
fn take_guest_stderr(
    store: &mut Store<Host>,
    instance: &Instance,
    memory: &Memory,
) -> Result<Vec<u8>> {
    let Ok(take_stderr) = instance.get_typed_func::<(), u64>(&mut *store, "pyre_take_stderr")
    else {
        return Ok(Vec::new());
    };
    let packed = take_stderr.call(&mut *store, ())?;
    let (ptr, len) = ((packed >> 32) as u32, (packed & 0xffff_ffff) as u32);
    if len == 0 {
        return Ok(Vec::new());
    }
    let mut bytes = vec![0u8; len as usize];
    memory.read(&*store, ptr as usize, &mut bytes)?;
    let dealloc = instance.get_typed_func::<(u32, u32), ()>(&mut *store, "pyre_dealloc")?;
    dealloc.call(&mut *store, (ptr, len))?;
    Ok(bytes)
}

/// Scan wasm linear memory for panic messages pyre-wasm's hook wrote there.
///
/// The hook prepends a literal `[pyre panic] ` to the panic info, and that
/// concatenation only exists at runtime (never as a static format template), so
/// matching the combined prefix recovers the real formatted message — including
/// the asserted values — rather than a `{…}`-placeholder template.
pub(crate) fn recover_panic_messages(data: &[u8]) -> Vec<String> {
    let needle = b"[pyre panic] panicked at";
    let mut out: Vec<String> = Vec::new();
    let mut from = 0;
    while let Some(rel) = data[from..].windows(needle.len()).position(|w| w == needle) {
        let pos = from + rel;
        let end = data[pos..]
            .iter()
            .position(|&b| b == 0)
            .map(|n| pos + n)
            .unwrap_or((pos + 400).min(data.len()));
        // Decode and cut at the first non-text byte (`U+FFFD` from lossy
        // decoding of trailing String capacity garbage).
        let text = String::from_utf8_lossy(&data[pos..end]);
        let text = text
            .split('\u{FFFD}')
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        if !text.is_empty() && !out.contains(&text) {
            out.push(text);
            if out.len() >= 4 {
                break;
            }
        }
        from = end + 1;
    }
    out
}

fn read_u32(mem: &Memory, store: impl AsContext, off: usize) -> u32 {
    let mut b = [0u8; 4];
    let _ = mem.read(store, off, &mut b);
    u32::from_le_bytes(b)
}

/// Dump the module's imports and exports, for debugging the host contract.
fn inspect_module(module_path: &PathBuf) -> Result<()> {
    // Match `run`'s engine configuration.
    let config = Config::new();
    let engine = Engine::new(&config)?;
    let module = Module::from_file(&engine, module_path)
        .with_context(|| format!("load wasm module {}", module_path.display()))?;
    println!("imports:");
    for import in module.imports() {
        println!(
            "  {}.{} : {:?}",
            import.module(),
            import.name(),
            import.ty()
        );
    }
    println!("exports:");
    for export in module.exports() {
        println!("  {} : {:?}", export.name(), export.ty());
    }
    Ok(())
}
