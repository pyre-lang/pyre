//! `--engine wasmi` execution path: runs the same wasm32 `pyre-wasm` module
//! under the pure-Rust `wasmi` interpreter instead of wasmtime.
//!
//! It satisfies the identical host-import contract as the wasmtime path in
//! `main.rs` (`majit_host.*`, `pyre_host.*`, the per-trace `env.memory` /
//! `env.jit_call` wiring, and the reflective residual-call trampoline). The
//! only difference is the runtime: wasmi does not compile the module on load,
//! so there is no per-process cranelift fixed cost — at the price of slower
//! steady-state execution. The call-area offsets and panic-message recovery
//! are shared with `main.rs`.

use std::collections::HashMap;
use std::path::Path;

use wasmi::{
    AsContext, AsContextMut, Caller, Config, Engine, Extern, F32, F64, Func, Linker, Memory,
    Module, Ref, Store, Table, Val, ValType,
};

use crate::CALL_RESULT_OFS;
use crate::host_path::{guest_path_to_host, host_path_to_guest};

/// Per-store host state, mirroring the wasmtime path's `Host`. wasmi needs the
/// engine handle stored too, because trace modules are compiled from inside an
/// import callback (`Caller`) that does not expose the engine directly.
#[derive(Default)]
struct Host {
    memory: Option<Memory>,
    table: Option<Table>,
    traces: HashMap<u32, Func>,
    next_id: u32,
    stdlib_root: Option<String>,
    engine: Option<Engine>,
    mem_limit: crate::memory_ceiling::GuestMemoryLimit,
}

fn estr(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Reported when wasmi's translator declines a main-module function. Not a pyre
/// bug; the program is runnable under wasmtime.
const WASMI_TRANSLATOR_DECLINE: &str = "wasmi could not translate this module (cmp+branch fusion assertion in wasmi 1.x); \
     run this program with `--engine wasmtime`";

/// True for panics raised by wasmi's own bytecode translator (e.g. the
/// `cmp+branch fusion must succeed` assertion in wasmi 1.x). Such a panic is a
/// wasmi limitation translating a module function, not a pyre bug; `run`
/// catch_unwinds it into the clean WASMI_TRANSLATOR_DECLINE error.
fn is_wasmi_translator_panic(msg: &str) -> bool {
    msg.contains("fusion must succeed") || msg.contains("translator")
}

/// Install a panic hook that suppresses the default report for wasmi translator
/// panics (which `run` catches and turns into a clean, actionable error) while
/// leaving every other panic's reporting intact — so the suppressed case does
/// not also print a confusing `panicked at …` line. Installed once per process;
/// re-entry would otherwise nest a new wrapper around the existing hook.
fn install_decline_hook() {
    static HOOK_INSTALLED: std::sync::Once = std::sync::Once::new();
    HOOK_INSTALLED.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let msg = info
                .payload()
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| info.payload().downcast_ref::<String>().cloned())
                .unwrap_or_default();
            if is_wasmi_translator_panic(&msg) {
                return;
            }
            default(info);
        }));
    });
}

pub fn run(module_path: &Path, source: &str, script: &Path) -> Result<i32, String> {
    install_decline_hook();
    let mut config = Config::default();
    // Raise the interpreter's value-stack / recursion ceilings so the deep
    // Python recursion the interpreter performs does not trip wasmi's limits
    // before the interpreter's own recursion guard fires. Both are caps (a
    // trap fires if exceeded) grown lazily, so generous values are cheap.
    config.set_max_recursion_depth(65536);
    config.set_max_stack_height(256 * 1024 * 1024);
    let engine = Engine::new(&config);

    let bytes = std::fs::read(module_path)
        .map_err(|e| format!("load wasm module {}: {e}", module_path.display()))?;
    let module = Module::new(&engine, &bytes[..]).map_err(|e| format!("compile module: {e}"))?;

    let mut store = Store::new(&engine, Host::default());
    store.data_mut().mem_limit.ceiling = crate::memory_ceiling::bytes();
    if store.data().mem_limit.ceiling != 0 {
        store.limiter(|host| &mut host.mem_limit);
    }
    store.data_mut().next_id = 1;
    store.data_mut().stdlib_root = std::env::var("PYRE_STDLIB").ok();
    store.data_mut().engine = Some(engine.clone());

    let linker = build_linker(&engine)?;
    let instantiated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        linker.instantiate_and_start(&mut store, &module)
    }));
    let instance = match instantiated {
        Ok(r) => r.map_err(|e| format!("instantiate main module: {e}"))?,
        Err(_) => return Err(WASMI_TRANSLATOR_DECLINE.to_string()),
    };

    let memory = instance
        .get_memory(&store, "memory")
        .ok_or("main module is missing its `memory` export")?;
    let table = instance
        .get_table(&store, "__indirect_function_table")
        .ok_or(
        "main module is missing its `__indirect_function_table` export (build with --export-table)",
    )?;
    store.data_mut().memory = Some(memory);
    store.data_mut().table = Some(table);

    let alloc = instance
        .get_typed_func::<u32, u32>(&store, "pyre_alloc")
        .map_err(estr)?;
    let run_python = instance
        .get_typed_func::<(u32, u32), u64>(&store, "pyre_run_python")
        .map_err(estr)?;
    let dealloc = instance
        .get_typed_func::<(u32, u32), ()>(&store, "pyre_dealloc")
        .map_err(estr)?;

    // One environment for the guest. The launcher, the collector and the JIT
    // knobs all read it; the runner forwards every name the module asks for in
    // one blob. Values stay undecoded: `PYTHONSAFEPATH` is a presence flag on
    // the raw bytes, and a non-UTF-8 value is still set.
    //
    // A module predating `pyre_set_env` still understands `-P` through
    // `pyre_set_safe_path`.
    let env_names = instance
        .get_typed_func::<(), u64>(&store, "pyre_env_names")
        .ok();
    let set_env = instance
        .get_typed_func::<(u32, u32), ()>(&store, "pyre_set_env")
        .ok();
    if let (Some(names), Some(set_env)) = (env_names, set_env) {
        let packed = names.call(&mut store, ()).map_err(estr)?;
        let (nptr, nlen) = ((packed >> 32) as u32, packed as u32);
        let mut buf = vec![0u8; nlen as usize];
        memory.read(&store, nptr as usize, &mut buf).map_err(estr)?;
        dealloc.call(&mut store, (nptr, nlen)).map_err(estr)?;

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
            set_env.call(&mut store, (0, 0)).map_err(estr)?;
        } else {
            let p = alloc.call(&mut store, blen).map_err(estr)?;
            memory.write(&mut store, p as usize, &blob).map_err(estr)?;
            set_env.call(&mut store, (p, blen)).map_err(estr)?;
            dealloc.call(&mut store, (p, blen)).map_err(estr)?;
        }
    } else if std::env::var_os("PYTHONSAFEPATH").is_some_and(|value| !value.is_empty())
        && let Ok(set_safe_path) = instance.get_typed_func::<u32, ()>(&store, "pyre_set_safe_path")
    {
        set_safe_path.call(&mut store, 1).map_err(estr)?;
    }

    // Name the script so the guest compiles it under its real path: that is
    // what a traceback prints, what its source-line lookup reads back through
    // `pyre_host.host_read`, and the directory that heads `sys.path`. Absent
    // on a module predating the export, leaving the guest on `<string>`.
    if let Ok(set_path) = instance.get_typed_func::<(u32, u32), ()>(&store, "pyre_set_script_path")
    {
        let name = host_path_to_guest(script).ok_or_else(|| {
            format!(
                "script path cannot be represented in guest: {}",
                script.display()
            )
        })?;
        let nlen = name.len() as u32;
        if nlen != 0 {
            let p = alloc.call(&mut store, nlen).map_err(estr)?;
            memory.write(&mut store, p as usize, &name).map_err(estr)?;
            set_path.call(&mut store, (p, nlen)).map_err(estr)?;
            dealloc.call(&mut store, (p, nlen)).map_err(estr)?;
        }
    }

    // Same `pyre_set_gc_env` channel as the wasmtime path: the guest has no
    // process environment, so `PYPY_GC_*` and `MAJIT_GC_STRESS` have to arrive
    // before the first allocation builds the collector.
    let gc_env_names = instance
        .get_typed_func::<(), u64>(&store, "pyre_gc_env_names")
        .ok();
    let set_gc_env = instance
        .get_typed_func::<(u32, u32), ()>(&store, "pyre_set_gc_env")
        .ok();
    if let (Some(names), Some(set_gc_env)) = (gc_env_names, set_gc_env) {
        let packed = names.call(&mut store, ()).map_err(estr)?;
        let (nptr, nlen) = ((packed >> 32) as u32, packed as u32);
        let mut buf = vec![0u8; nlen as usize];
        memory.read(&store, nptr as usize, &mut buf).map_err(estr)?;
        dealloc.call(&mut store, (nptr, nlen)).map_err(estr)?;

        let blob = String::from_utf8_lossy(&buf)
            .split('\0')
            .filter(|name| !name.is_empty())
            .filter_map(|name| {
                std::env::var(name)
                    .ok()
                    .map(|value| format!("{name}={value}"))
            })
            .collect::<Vec<_>>()
            .join("\0");
        let blen = blob.len() as u32;
        if blen != 0 {
            let p = alloc.call(&mut store, blen).map_err(estr)?;
            memory
                .write(&mut store, p as usize, blob.as_bytes())
                .map_err(estr)?;
            set_gc_env.call(&mut store, (p, blen)).map_err(estr)?;
            dealloc.call(&mut store, (p, blen)).map_err(estr)?;
        }
    }

    let src = source.as_bytes();
    let len = src.len() as u32;
    let in_ptr = if len == 0 {
        0
    } else {
        let p = alloc.call(&mut store, len).map_err(estr)?;
        memory.write(&mut store, p as usize, src).map_err(estr)?;
        p
    };

    // wasmi 1.x translates each module function lazily on first use, and its
    // translator can hit an internal assertion (`cmp+branch fusion must
    // succeed`) on some rustc-emitted functions. That is a wasmi limitation,
    // not a pyre bug, and unlike a JIT trace a main-module function cannot be
    // declined — so report it as a clean, actionable error instead of letting
    // the panic unwind the worker thread into an opaque "panicked" message.
    let called = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_python.call(&mut store, (in_ptr, len))
    }));
    let packed = match called {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            for msg in crate::recover_panic_messages(memory.data(&store)) {
                eprintln!("pyre-wasm-runner: recovered panic: {msg}");
            }
            return Err(estr(e));
        }
        Err(_) => return Err(WASMI_TRANSLATOR_DECLINE.to_string()),
    };
    let out_ptr = (packed >> 32) as u32;
    let out_len = (packed & 0xffff_ffff) as u32;

    let mut out = vec![0u8; out_len as usize];
    if out_len != 0 {
        memory
            .read(&store, out_ptr as usize, &mut out)
            .map_err(estr)?;
        dealloc.call(&mut store, (out_ptr, out_len)).map_err(estr)?;
    }
    // The guest has no descriptors, so its fd-2 bytes and its exit status come
    // back through their own exports. Absent on a module predating them, in
    // which case the run keeps the old stdout-only, always-0 behaviour.
    let mut err_bytes = Vec::new();
    if let Ok(take_stderr) = instance.get_typed_func::<(), u64>(&store, "pyre_take_stderr") {
        let packed = take_stderr.call(&mut store, ()).map_err(estr)?;
        let (ptr, elen) = ((packed >> 32) as u32, (packed & 0xffff_ffff) as u32);
        if elen != 0 {
            err_bytes = vec![0u8; elen as usize];
            memory
                .read(&store, ptr as usize, &mut err_bytes)
                .map_err(estr)?;
            dealloc.call(&mut store, (ptr, elen)).map_err(estr)?;
        }
    }
    let exit_code = instance
        .get_typed_func::<(), i32>(&store, "pyre_exit_code")
        .map_err(estr)
        .and_then(|f| f.call(&mut store, ()).map_err(estr))
        .unwrap_or(0);
    if len != 0 {
        dealloc.call(&mut store, (in_ptr, len)).map_err(estr)?;
    }

    use std::io::Write;
    std::io::stdout().write_all(&out).map_err(estr)?;
    std::io::stdout().flush().map_err(estr)?;
    if !err_bytes.is_empty() {
        std::io::stderr().write_all(&err_bytes).map_err(estr)?;
    }
    Ok(exit_code)
}

fn build_linker(engine: &Engine) -> Result<Linker<Host>, String> {
    let mut linker = Linker::new(engine);

    linker
        .func_wrap(
            "majit_host",
            "jit_compile_wasm",
            |mut caller: Caller<'_, Host>, bytes_ptr: u32, bytes_len: u32| -> u32 {
                match jit_compile(&mut caller, bytes_ptr, bytes_len) {
                    Ok(id) => id,
                    Err(e) => {
                        eprintln!("[jit_compile_wasm] {e}");
                        0
                    }
                }
            },
        )
        .map_err(estr)?;

    linker
        .func_wrap(
            "majit_host",
            "jit_replace_wasm",
            |mut caller: Caller<'_, Host>, func_id: u32, bytes_ptr: u32, bytes_len: u32| -> u32 {
                match jit_replace(&mut caller, func_id, bytes_ptr, bytes_len) {
                    Ok(id) => id,
                    Err(e) => {
                        eprintln!("[jit_replace_wasm] {e}");
                        0
                    }
                }
            },
        )
        .map_err(estr)?;

    linker
        .func_wrap(
            "majit_host",
            "jit_execute_wasm",
            |mut caller: Caller<'_, Host>, func_id: u32, frame_ptr: u32| -> u32 {
                match jit_execute(&mut caller, func_id, frame_ptr) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("[jit_execute_wasm] {e}");
                        0
                    }
                }
            },
        )
        .map_err(estr)?;

    linker
        .func_wrap(
            "majit_host",
            "jit_free_wasm",
            |mut caller: Caller<'_, Host>, func_id: u32| {
                caller.data_mut().traces.remove(&func_id);
            },
        )
        .map_err(estr)?;

    // Residual-call trampoline for the recording / blackhole path; see the
    // wasmtime path's `jit_call_host` for the full rationale.
    linker
        .func_wrap(
            "majit_host",
            "jit_call_host",
            |mut caller: Caller<'_, Host>, frame_ptr: u32| -> Result<(), wasmi::Error> {
                jit_call_trampoline(&mut caller, frame_ptr, CALL_RESULT_OFS as u32)
                    .map_err(wasmi::Error::new)
            },
        )
        .map_err(estr)?;

    linker
        .func_wrap(
            "env",
            "jit_func_sig",
            |mut caller: Caller<'_, Host>, slot: i32| -> i64 {
                jit_func_sig_of_slot(&mut caller, slot)
            },
        )
        .map_err(estr)?;

    // Host-filesystem imports serving the wasm-host build's module loader from
    // `$PYRE_STDLIB`. The wasm32 module has no filesystem of its own.
    linker
        .func_wrap(
            "pyre_host",
            "host_stdlib_root",
            |mut caller: Caller<'_, Host>, buf_ptr: u32, buf_cap: u32| -> i64 {
                host_stdlib_root(&mut caller, buf_ptr, buf_cap)
            },
        )
        .map_err(estr)?;
    linker
        .func_wrap(
            "pyre_host",
            "host_is_dir",
            |mut caller: Caller<'_, Host>, path_ptr: u32, path_len: u32| -> u32 {
                match host_path(&mut caller, path_ptr, path_len) {
                    Some(p) => std::path::PathBuf::from(p).is_dir() as u32,
                    None => 0,
                }
            },
        )
        .map_err(estr)?;
    linker
        .func_wrap(
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
        )
        .map_err(estr)?;
    linker
        .func_wrap(
            "pyre_host",
            "host_read",
            |mut caller: Caller<'_, Host>,
             path_ptr: u32,
             path_len: u32,
             buf_ptr: u32,
             buf_cap: u32|
             -> i64 { host_read(&mut caller, path_ptr, path_len, buf_ptr, buf_cap) },
        )
        .map_err(estr)?;
    linker
        .func_wrap(
            "pyre_host",
            "host_list_dir",
            |mut caller: Caller<'_, Host>,
             path_ptr: u32,
             path_len: u32,
             buf_ptr: u32,
             buf_cap: u32|
             -> i64 {
                host_list_dir(&mut caller, path_ptr, path_len, buf_ptr, buf_cap)
            },
        )
        .map_err(estr)?;

    linker
        .func_wrap(
            "pyre_host",
            "host_cwd",
            |mut caller: Caller<'_, Host>, buf_ptr: u32, buf_cap: u32| -> i64 {
                host_cwd(&mut caller, buf_ptr, buf_cap)
            },
        )
        .map_err(estr)?;
    linker
        .func_wrap(
            "pyre_host",
            "host_random",
            |mut caller: Caller<'_, Host>, buf_ptr: u32, buf_cap: u32| -> i32 {
                host_random(&mut caller, buf_ptr, buf_cap)
            },
        )
        .map_err(estr)?;

    // Host clock imports.  wasm32 has neither a `SystemTime` nor an `Instant`,
    // so `time` -- and every stdlib module that imports it -- reads the clock
    // here.  See `pyre-wasm`'s `host_clock`.
    linker
        .func_wrap("pyre_host", "host_time_ns", || -> i64 { host_time_ns() })
        .map_err(estr)?;
    linker
        .func_wrap("pyre_host", "host_monotonic_ns", || -> i64 {
            host_monotonic_ns()
        })
        .map_err(estr)?;
    linker
        .func_wrap("pyre_host", "host_sleep_ns", |nanos: i64| {
            host_sleep_ns(nanos)
        })
        .map_err(estr)?;

    Ok(linker)
}

/// `pyre_host.host_random`: fill the guest buffer with host entropy.
///
/// wasm32-unknown-unknown has no OS entropy of its own, so `os.urandom`,
/// `secrets` and the string hash key all end up here.
fn host_random(caller: &mut Caller<'_, Host>, buf_ptr: u32, buf_cap: u32) -> i32 {
    let Some(memory) = caller.data().memory else {
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
    let Some(memory) = caller.data().memory else {
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
    let memory = caller.data().memory?;
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
        return bytes.len() as i64;
    }
    let Some(memory) = caller.data().memory else {
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
    let Some(memory) = caller.data().memory else {
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
    let Some(memory) = caller.data().memory else {
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

/// Compile and instantiate a JIT-emitted trace module, sharing the main
/// module's linear memory and wiring the `jit_call` trampoline.
fn jit_compile_trace(
    caller: &mut Caller<'_, Host>,
    bytes_ptr: u32,
    bytes_len: u32,
) -> Result<Func, String> {
    let memory = caller.data().memory.ok_or("main memory not initialized")?;

    let mut bytes = vec![0u8; bytes_len as usize];
    memory
        .read(&*caller, bytes_ptr as usize, &mut bytes)
        .map_err(|e| format!("read trace module bytes: {e}"))?;

    let engine = caller.data().engine.clone().ok_or("engine not set")?;
    if std::env::var_os("PYRE_WASM_DUMP_ALL_TRACES").is_some() {
        match wasmprinter::print_bytes(&bytes) {
            Ok(wat) => eprintln!("=== trace module ({} bytes) ===\n{wat}", bytes.len()),
            Err(pe) => eprintln!("[jit_compile_wasm] wat print failed: {pe}"),
        }
    }
    let module = match Module::new(&engine, &bytes[..]) {
        Ok(m) => m,
        Err(e) => {
            if std::env::var_os("PYRE_WASM_DUMP_BAD_TRACE").is_some() {
                let path = std::env::temp_dir().join("pyre_bad_trace.wasm");
                let _ = std::fs::write(&path, &bytes);
                eprintln!(
                    "[jit_compile_wasm] dumped {} bytes to {}",
                    bytes.len(),
                    path.display()
                );
                match wasmprinter::print_bytes(&bytes) {
                    Ok(wat) => eprintln!("--- WAT ---\n{wat}\n--- /WAT ---"),
                    Err(pe) => eprintln!("[jit_compile_wasm] wat print failed: {pe}"),
                }
            }
            return Err(format!("compile trace module: {e}"));
        }
    };

    // A fresh trampoline per trace; it reads all state from `caller.data()`.
    let jit_call = Func::wrap(
        &mut *caller,
        |mut inner: Caller<'_, Host>, frame_ptr: i32| {
            if let Err(e) =
                jit_call_trampoline(&mut inner, frame_ptr as u32, CALL_RESULT_OFS as u32)
            {
                eprintln!("[jit_call] {e}");
            }
        },
    );
    let jit_call_compact = Func::wrap(
        &mut *caller,
        |mut inner: Caller<'_, Host>, frame_ptr: i32, call_area_ofs: i32| {
            if let Err(e) = jit_call_trampoline(&mut inner, frame_ptr as u32, call_area_ofs as u32)
            {
                eprintln!("[jit_call_compact] {e}");
            }
        },
    );

    // Supply imports by name; a trace that imports only `env.memory` simply
    // leaves the defined `env.jit_call` unused.
    let mut linker = Linker::new(&engine);
    linker
        .define("env", "memory", Extern::Memory(memory))
        .map_err(estr)?;
    linker
        .define("env", "jit_call", Extern::Func(jit_call))
        .map_err(estr)?;
    linker
        .define("env", "jit_call_compact", Extern::Func(jit_call_compact))
        .map_err(estr)?;
    let jit_func_sig = Func::wrap(
        &mut *caller,
        |mut inner: Caller<'_, Host>, slot: i32| -> i64 { jit_func_sig_of_slot(&mut inner, slot) },
    );
    linker
        .define("env", "jit_func_sig", Extern::Func(jit_func_sig))
        .map_err(estr)?;
    let instance = linker
        .instantiate_and_start(&mut *caller, &module)
        .map_err(|e| format!("instantiate trace module: {e}"))?;
    let trace = instance
        .get_func(&*caller, "trace")
        .ok_or("trace module is missing its `trace` export")?;

    Ok(trace)
}

fn jit_compile(
    caller: &mut Caller<'_, Host>,
    bytes_ptr: u32,
    bytes_len: u32,
) -> Result<u32, String> {
    let trace = jit_compile_trace(caller, bytes_ptr, bytes_len)?;
    let host = caller.data_mut();
    let id = host.next_id;
    host.next_id += 1;
    host.traces.insert(id, trace);
    Ok(id)
}

fn jit_replace(
    caller: &mut Caller<'_, Host>,
    func_id: u32,
    bytes_ptr: u32,
    bytes_len: u32,
) -> Result<u32, String> {
    if !caller.data().traces.contains_key(&func_id) {
        return Err(format!("jit_replace_wasm: unknown func id {func_id}"));
    }
    let trace = jit_compile_trace(caller, bytes_ptr, bytes_len)?;
    // The map owns the old function until this overwrite. Existing calls keep
    // their already-entered instance; later dispatches use this replacement.
    caller.data_mut().traces.insert(func_id, trace);
    Ok(func_id)
}

/// Run a previously compiled trace, returning its guard-exit index.
fn jit_execute(caller: &mut Caller<'_, Host>, func_id: u32, frame_ptr: u32) -> Result<u32, String> {
    let trace = *caller
        .data()
        .traces
        .get(&func_id)
        .ok_or_else(|| format!("jit_execute_wasm: unknown func id {func_id}"))?;

    // A wasmi translator panic during the trace's first call (lazy translation)
    // unwinds up through `run_python.call`'s catch_unwind in `run`, which turns
    // it into the clean WASMI_TRANSLATOR_DECLINE error. We do not catch it here:
    // synthesising a guard-exit index to "decline" a half-entered trace would
    // risk a silently wrong result, and a clean abort is the safe outcome.
    let mut results = [Val::I32(0)];
    trace
        .call(&mut *caller, &[Val::I32(frame_ptr as i32)], &mut results)
        .map_err(estr)?;
    Ok(match results[0] {
        Val::I32(x) => x as u32,
        _ => 0,
    })
}

/// Dispatch a residual call requested by a running trace.
fn jit_call_trampoline(
    caller: &mut Caller<'_, Host>,
    frame_ptr: u32,
    call_area_ofs: u32,
) -> Result<(), String> {
    let memory = caller.data().memory.ok_or("memory")?;
    let table = caller.data().table.ok_or("table")?;
    let call_area = frame_ptr as usize + call_area_ofs as usize;

    let func_ptr = read_u32(&memory, &*caller, call_area + 8);

    // `func_ptr == 0` is the "newstr" sentinel; without a host string
    // allocator it yields 0.
    if func_ptr == 0 {
        write_i64(&memory, &mut *caller, call_area, 0)?;
        return Ok(());
    }

    let func = match table.get(&*caller, func_ptr as u64) {
        Some(Val::FuncRef(fr)) => match func_of(&fr) {
            Some(f) => f,
            None => {
                write_i64(&memory, &mut *caller, call_area, 0)?;
                return Ok(());
            }
        },
        _ => {
            write_i64(&memory, &mut *caller, call_area, 0)?;
            return Ok(());
        }
    };

    let ty = func.ty(&*caller);
    let params: Vec<ValType> = ty.params().to_vec();
    let mut args: Vec<Val> = Vec::with_capacity(params.len());
    for (i, pty) in params.iter().enumerate() {
        let raw = read_i64(&memory, &*caller, call_area + 24 + i * 8);
        args.push(match *pty {
            ValType::I32 => Val::I32(raw as i32),
            ValType::I64 => Val::I64(raw),
            // Floats cross the call area as their raw bit pattern in an i64 slot.
            ValType::F32 => Val::F32(F32::from_bits(raw as u32)),
            ValType::F64 => Val::F64(F64::from_bits(raw as u64)),
            other => {
                return Err(format!("unsupported residual-call param type {other:?}"));
            }
        });
    }

    let mut results: Vec<Val> = ty
        .results()
        .iter()
        .map(|t| match *t {
            ValType::I64 => Val::I64(0),
            ValType::F32 => Val::F32(F32::from_bits(0)),
            ValType::F64 => Val::F64(F64::from_bits(0)),
            _ => Val::I32(0),
        })
        .collect();

    // Mirror the browser glue's try/catch: a trapping residual target is
    // reported as a zero result rather than aborting the whole run.
    if let Err(e) = func.call(&mut *caller, &args, &mut results) {
        eprintln!("[jit_call] residual target trapped: {e}");
        write_i64(&memory, &mut *caller, call_area, 0)?;
        return Ok(());
    }

    let result = match results.first() {
        Some(Val::I32(x)) => (*x as u32) as i64, // zero-extend; high word stays 0
        Some(Val::I64(x)) => *x,
        Some(Val::F64(x)) => x.to_bits() as i64,
        Some(Val::F32(x)) => (x.to_bits() as u64) as i64,
        _ => 0,
    };
    write_i64(&memory, &mut *caller, call_area, result)?;
    Ok(())
}

fn jit_func_sig_of_slot(caller: &mut Caller<'_, Host>, slot: i32) -> i64 {
    if slot == 0 {
        return 0;
    }
    let Some(table) = caller.data().table else {
        return 0;
    };
    let func = match table.get(&*caller, slot as u64) {
        Some(Val::FuncRef(fr)) => match func_of(&fr) {
            Some(f) => f,
            None => return 0,
        },
        _ => return 0,
    };
    let ty = func.ty(&*caller);
    if ty.params().len() > majit_backend_wasm_host::MAX_CALL_ARGS || ty.results().len() > 1 {
        return 0;
    }
    let mut params = Vec::with_capacity(ty.params().len());
    for param in ty.params() {
        params.push(match *param {
            ValType::I32 => majit_backend_wasm_host::FuncSigVal::I32,
            ValType::I64 => majit_backend_wasm_host::FuncSigVal::I64,
            ValType::F32 => majit_backend_wasm_host::FuncSigVal::F32,
            ValType::F64 => majit_backend_wasm_host::FuncSigVal::F64,
            _ => return 0,
        });
    }
    let result = match ty.results().first() {
        None => None,
        Some(ValType::I32) => Some(majit_backend_wasm_host::FuncSigVal::I32),
        Some(ValType::I64) => Some(majit_backend_wasm_host::FuncSigVal::I64),
        Some(ValType::F32) => Some(majit_backend_wasm_host::FuncSigVal::F32),
        Some(ValType::F64) => Some(majit_backend_wasm_host::FuncSigVal::F64),
        _ => return 0,
    };
    majit_backend_wasm_host::encode_func_sig(&params, result)
}

/// Resolve a funcref to its `Func`, copying the lightweight handle out.
fn func_of(fr: &Ref<Func>) -> Option<Func> {
    fr.val().copied()
}

fn read_u32(mem: &Memory, store: impl AsContext, off: usize) -> u32 {
    let mut b = [0u8; 4];
    let _ = mem.read(store, off, &mut b);
    u32::from_le_bytes(b)
}

fn read_i64(mem: &Memory, store: impl AsContext, off: usize) -> i64 {
    let mut b = [0u8; 8];
    let _ = mem.read(store, off, &mut b);
    i64::from_le_bytes(b)
}

fn write_i64(mem: &Memory, store: impl AsContextMut, off: usize, v: i64) -> Result<(), String> {
    mem.write(store, off, &v.to_le_bytes())
        .map_err(|e| format!("write call-area result: {e}"))
}

/// Dump the module's imports and exports, for debugging the host contract.
pub fn inspect(module_path: &Path) -> Result<i32, String> {
    let engine = Engine::default();
    let bytes = std::fs::read(module_path)
        .map_err(|e| format!("load wasm module {}: {e}", module_path.display()))?;
    let module = Module::new(&engine, &bytes[..]).map_err(|e| format!("compile module: {e}"))?;
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
    Ok(0)
}
