// `web` (wasm-bindgen) and `wasm-host` (C-ABI) export conflicting `run_python`
// surfaces; exactly one host binding may be active at a time.
#[cfg(all(feature = "web", feature = "wasm-host"))]
compile_error!("features `web` and `wasm-host` are mutually exclusive");

// The wasm-host C-ABI packs a result pointer and length into the high/low
// halves of a u64, which only round-trips with 32-bit pointers.
#[cfg(all(feature = "wasm-host", not(target_arch = "wasm32")))]
compile_error!("feature `wasm-host` requires target_arch = \"wasm32\"");

#[cfg(feature = "web")]
use wasm_bindgen::prelude::*;

// Native-host (`wasm-host`) builds target wasm32-unknown-unknown, which has no OS
// entropy. To avoid the wasm-bindgen-based `wasm_js` backend (whose imports a
// non-JS embedder cannot satisfy), getrandom is wired to its `custom` backend
// via `--cfg getrandom_backend="custom"`, which calls this hook.
//
// The embedder has entropy, so this asks it for the bytes: `os.urandom` and
// `secrets` are the same call, and a stream anyone can recompute is not what
// either of them promises. Only when the host declines does the SplitMix64
// fall-back below answer, which keeps the string hash key and the `random`
// module working on an embedder that supplies no entropy at all.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
mod custom_getrandom {
    use core::sync::atomic::{AtomicU64, Ordering};

    #[link(wasm_import_module = "pyre_host")]
    unsafe extern "C" {
        /// Fill `[buf, buf+cap)` with entropy; 0 on success, -1 if the host
        /// has none to give.
        fn host_random(buf_ptr: *mut u8, buf_cap: u32) -> i32;
    }

    static STATE: AtomicU64 = AtomicU64::new(0x9e37_79b9_7f4a_7c15);

    #[unsafe(no_mangle)]
    unsafe extern "Rust" fn __getrandom_v03_custom(
        dest: *mut u8,
        len: usize,
    ) -> Result<(), getrandom::Error> {
        if len == 0 {
            return Ok(());
        }
        if unsafe { host_random(dest, len as u32) } == 0 {
            return Ok(());
        }
        let mut i = 0;
        while i < len {
            let mut z = STATE
                .fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed)
                .wrapping_add(0x9e37_79b9_7f4a_7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            let bytes = z.to_le_bytes();
            let n = core::cmp::min(8, len - i);
            unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), dest.add(i), n) };
            i += n;
        }
        Ok(())
    }
}

#[cfg(any(feature = "web", feature = "wasm-host"))]
use pyre_interpreter::*;

#[cfg(any(feature = "web", feature = "wasm-host"))]
use std::cell::RefCell;
#[cfg(any(feature = "web", feature = "wasm-host"))]
use std::sync::Once;

// Diagnostic counting allocator (feature `heap-prof`): wraps the platform
// allocator and tracks net-live bytes/count so the host runner can tell a true
// not-freed leak (live grows linearly) from fragmentation (live flat,
// linear-memory still grows). realloc delegates to keep dlmalloc's in-place
// behaviour and accounts only the size delta. Not compiled into shipping builds.
#[cfg(feature = "heap-prof")]
mod heap_prof {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicI64, Ordering};

    pub(super) static LIVE_BYTES: AtomicI64 = AtomicI64::new(0);
    pub(super) static LIVE_COUNT: AtomicI64 = AtomicI64::new(0);

    // Net-live allocation count bucketed by size class: bucket i = sizes in
    // `(8*(i-1), 8*i]`, i.e. `(size+7)/8` clamped to 63. Bucket 63 aggregates
    // everything ≥ 504 bytes. Lets the host see exactly which size leaks.
    pub(super) const NBUCKETS: usize = 64;
    pub(super) static BUCKETS: [AtomicI64; NBUCKETS] = {
        const Z: AtomicI64 = AtomicI64::new(0);
        [Z; NBUCKETS]
    };

    #[inline]
    fn bucket(size: usize) -> usize {
        (size.div_ceil(8)).min(NBUCKETS - 1)
    }

    pub(super) struct CountingAlloc;

    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let p = unsafe { System.alloc(layout) };
            if !p.is_null() {
                LIVE_BYTES.fetch_add(layout.size() as i64, Ordering::Relaxed);
                LIVE_COUNT.fetch_add(1, Ordering::Relaxed);
                BUCKETS[bucket(layout.size())].fetch_add(1, Ordering::Relaxed);
            }
            p
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let p = unsafe { System.alloc_zeroed(layout) };
            if !p.is_null() {
                LIVE_BYTES.fetch_add(layout.size() as i64, Ordering::Relaxed);
                LIVE_COUNT.fetch_add(1, Ordering::Relaxed);
                BUCKETS[bucket(layout.size())].fetch_add(1, Ordering::Relaxed);
            }
            p
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) };
            LIVE_BYTES.fetch_sub(layout.size() as i64, Ordering::Relaxed);
            LIVE_COUNT.fetch_sub(1, Ordering::Relaxed);
            BUCKETS[bucket(layout.size())].fetch_sub(1, Ordering::Relaxed);
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let p = unsafe { System.realloc(ptr, layout, new_size) };
            if !p.is_null() {
                LIVE_BYTES.fetch_add(new_size as i64 - layout.size() as i64, Ordering::Relaxed);
                BUCKETS[bucket(layout.size())].fetch_sub(1, Ordering::Relaxed);
                BUCKETS[bucket(new_size)].fetch_add(1, Ordering::Relaxed);
            }
            p
        }
    }
}

#[cfg(feature = "heap-prof")]
#[global_allocator]
static HEAP_PROF_ALLOC: heap_prof::CountingAlloc = heap_prof::CountingAlloc;

// Host clock for the native-host (`wasm-host`) build.
//
// wasm32 has neither a `SystemTime` nor an `Instant`: both panic rather than
// answer, which kept `time` -- and the nine stdlib modules that import it --
// out of the guest entirely.  The runner owns a clock, so the guest asks it
// the same way it asks for module source.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
mod host_clock {
    use pyre_interpreter::module::time::interp_time::ClockProvider;

    #[link(wasm_import_module = "pyre_host")]
    unsafe extern "C" {
        /// Nanoseconds since the unix epoch, or -1 if the host has no clock.
        fn host_time_ns() -> i64;
        /// Nanoseconds on the host's monotonic clock, counted from an origin
        /// the host only has to keep fixed for the run.
        fn host_monotonic_ns() -> i64;
        /// Block the calling thread for `nanos` nanoseconds.
        fn host_sleep_ns(nanos: i64);
    }

    struct HostClock;

    impl ClockProvider for HostClock {
        fn wall_nanos(&self) -> i128 {
            unsafe { host_time_ns() as i128 }
        }
        fn monotonic_nanos(&self) -> i128 {
            unsafe { host_monotonic_ns() as i128 }
        }
        fn sleep_nanos(&self, nanos: u64) {
            // An i64 tops out at 292 years, which is longer than any sleep a
            // caller means, so a wider request is served as the longest one
            // the call can carry rather than wrapping into a short one.
            unsafe { host_sleep_ns(nanos.min(i64::MAX as u64) as i64) }
        }
    }

    pub fn install() {
        pyre_interpreter::module::time::interp_time::install_clock_provider(std::sync::Arc::new(
            HostClock,
        ));
    }
}

// Browser clock for the `web` build. `Date.now` / `performance.now` are the
// wall and monotonic clocks; wasm32's `SystemTime` panics, so without this
// `datetime.date.today()` is always 1970-01-01.
#[cfg(all(target_arch = "wasm32", feature = "web"))]
mod web_clock {
    use pyre_interpreter::module::time::interp_time::ClockProvider;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = Date, js_name = now)]
        fn date_now() -> f64;
        #[wasm_bindgen(js_namespace = performance, js_name = now)]
        fn performance_now() -> f64;
    }

    struct WebClock;

    impl ClockProvider for WebClock {
        fn wall_nanos(&self) -> i128 {
            (date_now() * 1_000_000.0) as i128
        }
        fn monotonic_nanos(&self) -> i128 {
            (performance_now() * 1_000_000.0) as i128
        }
        fn sleep_nanos(&self, _nanos: u64) {
            // The browser event loop cannot block. A playground `time.sleep`
            // returns immediately rather than freeze the tab.
        }
    }

    pub fn install() {
        pyre_interpreter::module::time::interp_time::install_clock_provider(std::sync::Arc::new(
            WebClock,
        ));
    }
}

// Host-filesystem source provider for the native-host (`wasm-host`) build.
//
// wasm32 has no filesystem, but the wasmtime runner does, so module source is
// read through `pyre_host.*` host imports the runner satisfies. The runner
// reports the real stdlib root (the `$PYRE_STDLIB` directory `pyre/check.py`
// forwards); seeding it on `sys.path` lets the SAME import machinery that runs
// on native resolve `import re` against genuine host paths. The browser/web
// build has no such host, so it embeds the stdlib instead (`wasm_vfs`).
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
mod host_fs_provider {
    use pyre_interpreter::importing::SourceProvider;
    use std::path::Path;

    #[link(wasm_import_module = "pyre_host")]
    unsafe extern "C" {
        /// Write the real stdlib root path into `[buf, buf+cap)`; return its
        /// byte length (without writing if it exceeds `cap`), or -1 if unset.
        fn host_stdlib_root(buf_ptr: *mut u8, buf_cap: u32) -> i64;
        /// 1 if `[path, path+len)` names a directory, else 0.
        fn host_is_dir(path_ptr: *const u8, path_len: u32) -> u32;
        /// Byte length of the regular file at `path`, or -1 if not a file.
        fn host_file_size(path_ptr: *const u8, path_len: u32) -> i64;
        /// Read the file at `path` into `[buf, buf+cap)`; return bytes written
        /// (clamped to `cap`), or -1 on error.
        fn host_read(path_ptr: *const u8, path_len: u32, buf_ptr: *mut u8, buf_cap: u32) -> i64;
        /// Write the directory's entry names into `[buf, buf+cap)`,
        /// NUL-separated, and return the whole list's byte length (without
        /// writing if it exceeds `cap`), or -1 if `path` is not a readable
        /// directory.
        fn host_list_dir(path_ptr: *const u8, path_len: u32, buf_ptr: *mut u8, buf_cap: u32)
        -> i64;
        /// Write the host's working directory into `[buf, buf+cap)`; return
        /// its byte length (without writing if it exceeds `cap`), or -1 if
        /// the host has none.
        fn host_cwd(buf_ptr: *mut u8, buf_cap: u32) -> i64;
    }

    struct HostFsProvider;

    /// Every path crosses as the bytes an `OsStr` is on this target, not as a
    /// text spelling of them, so a name with no UTF-8 spelling reaches the
    /// host intact -- the direction back, `host_list_dir`, already hands such
    /// names to the guest.
    impl SourceProvider for HostFsProvider {
        fn is_file(&self, path: &Path) -> bool {
            let p = path.as_os_str().as_encoded_bytes();
            unsafe { host_file_size(p.as_ptr(), p.len() as u32) >= 0 }
        }
        fn file_size(&self, path: &Path) -> std::io::Result<u64> {
            let p = path.as_os_str().as_encoded_bytes();
            let size = unsafe { host_file_size(p.as_ptr(), p.len() as u32) };
            match size < 0 {
                true => Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("{}", path.display()),
                )),
                false => Ok(size as u64),
            }
        }
        fn is_dir(&self, path: &Path) -> bool {
            let p = path.as_os_str().as_encoded_bytes();
            unsafe { host_is_dir(p.as_ptr(), p.len() as u32) != 0 }
        }
        fn list_dir(&self, path: &Path) -> std::io::Result<Vec<Vec<u8>>> {
            let p = path.as_os_str().as_encoded_bytes();
            let call = |buf: &mut [u8]| unsafe {
                host_list_dir(
                    p.as_ptr(),
                    p.len() as u32,
                    buf.as_mut_ptr(),
                    buf.len() as u32,
                )
            };
            let not_found =
                || std::io::Error::new(std::io::ErrorKind::NotFound, format!("{}", path.display()));
            let mut buf = Vec::new();
            // The host reports the whole list's length whether or not it fitted
            // and writes nothing when it did not, so an answer longer than the
            // buffer is a directory that grew between the two reads and has to
            // be asked again rather than parsed out of bytes the host never
            // touched.  Two rounds suffice unless it keeps growing, which is
            // what the bound is for.
            for _ in 0..8 {
                let size = call(&mut buf);
                if size < 0 {
                    return Err(not_found());
                }
                let size = size as usize;
                if size <= buf.len() {
                    buf.truncate(size);
                    if buf.is_empty() {
                        return Ok(Vec::new());
                    }
                    return Ok(buf.split(|byte| *byte == 0).map(<[u8]>::to_vec).collect());
                }
                buf = vec![0u8; size];
            }
            Err(std::io::Error::other(format!(
                "host_list_dir: {} kept growing",
                path.display()
            )))
        }
        fn cwd(&self) -> Option<Vec<u8>> {
            // The host reports the length whether or not it fitted and writes
            // nothing when it did not, so the second round reads a buffer
            // sized from the first answer.
            let mut buf = vec![0u8; 256];
            for _ in 0..4 {
                let size = unsafe { host_cwd(buf.as_mut_ptr(), buf.len() as u32) };
                if size < 0 {
                    return None;
                }
                let size = size as usize;
                if size <= buf.len() {
                    buf.truncate(size);
                    return Some(buf);
                }
                buf = vec![0u8; size];
            }
            None
        }
        fn read_to_bytes(&self, path: &Path) -> std::io::Result<Vec<u8>> {
            let p = path.as_os_str().as_encoded_bytes();
            let size = unsafe { host_file_size(p.as_ptr(), p.len() as u32) };
            if size < 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("{}", path.display()),
                ));
            }
            let mut buf = vec![0u8; size as usize];
            let n = unsafe {
                host_read(
                    p.as_ptr(),
                    p.len() as u32,
                    buf.as_mut_ptr(),
                    buf.len() as u32,
                )
            };
            if n < 0 {
                return Err(std::io::Error::other(format!(
                    "host_read failed: {}",
                    path.display()
                )));
            }
            buf.truncate(n as usize);
            Ok(buf)
        }
    }

    /// Query the host for the stdlib root, seed it on `sys.path`, and install
    /// the host-FS source provider.  Called once before the first import.
    pub fn install() {
        let mut buf = vec![0u8; 4096];
        let n = unsafe { host_stdlib_root(buf.as_mut_ptr(), buf.len() as u32) };
        if n > 0 && (n as usize) <= buf.len() {
            if let Ok(root) = std::str::from_utf8(&buf[..n as usize]) {
                pyre_interpreter::importing::add_sys_path(Path::new(root));
            }
        }
        pyre_interpreter::importing::install_source_provider(std::sync::Arc::new(HostFsProvider));
    }
}

/// Diagnostic (`heap-prof`): net-live guest-heap bytes (alloc − dealloc).
/// Linear growth here under steady re-entry is a true not-freed leak.
#[cfg(feature = "heap-prof")]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_heap_live_bytes() -> i64 {
    heap_prof::LIVE_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Diagnostic (`heap-prof`): net-live guest-heap allocation count.
/// `live_bytes / live_count` is the average leaked-object size.
#[cfg(feature = "heap-prof")]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_heap_live_count() -> i64 {
    heap_prof::LIVE_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Diagnostic (`heap-prof`): net-live allocation count in size-class bucket
/// `i` (sizes in `(8*(i-1), 8*i]`; bucket 63 = ≥504 B). Out-of-range → 0.
#[cfg(feature = "heap-prof")]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_heap_bucket(i: u32) -> i64 {
    let i = i as usize;
    if i < heap_prof::NBUCKETS {
        heap_prof::BUCKETS[i].load(std::sync::atomic::Ordering::Relaxed)
    } else {
        0
    }
}

/// Diagnostic-only: read a `compile_bridge` outcome tally from the wasm JIT
/// backend (index legend in `majit_backend_wasm::BRIDGE_DIAG`). Exported (not
/// an import) so it does not shift the module's function-index space, which
/// would break the JIT's baked `fn as usize` table indices. The host runner
/// prints these at `PYRE_WASM_JIT_STATS` time.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_bridge_diag(i: u32) -> u64 {
    majit_backend_wasm::bridge_diag(i as usize)
}

/// Last `compile_loop` `Err` string, byte `i`. The guest has no stderr, so
/// this is how the host reads `build_wasm_module`'s decline. Pair with
/// [`pyre_jit_last_compile_err_len`].
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_last_compile_err_byte(i: u32) -> u32 {
    majit_backend_wasm::last_compile_err_byte(i)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_last_compile_err_len() -> u32 {
    majit_backend_wasm::last_compile_err_len()
}

/// Packed `(kind, needed, available)` geometry for an inline-module trial that
/// did not fit its owner's frozen frame. The host formats this diagnostic only.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_inline_geometry_diag(i: u32) -> u64 {
    majit_backend_wasm::inline_geometry_diag(i as usize)
}

/// Number of inline geometry failures, including entries beyond the retained
/// diagnostic records.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_inline_geometry_count() -> u64 {
    majit_backend_wasm::inline_geometry_count()
}

/// Test-only control plane for the terminal-declined CALL_ASSEMBLER regression.
/// Exported rather than imported so it cannot perturb table/function indices
/// used by JIT-emitted modules. Zero disables it; see
/// `majit_backend_wasm::set_force_ca_terminal_decline` for selector semantics.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_set_force_ca_terminal_decline(selector: u64) {
    majit_backend_wasm::set_force_ca_terminal_decline(selector);
}

/// Diagnostic-only: number of JIT trace entries made from the guest.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_execute_count() -> u64 {
    majit_backend_wasm::jit_execute_count()
}

/// Diagnostic-only: modules that crossed the lazy materialization gate.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_compile_count() -> u64 {
    majit_backend_wasm::jit_compile_count()
}

/// Diagnostic-only: byte-identical module cache hits.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_compile_cache_hits() -> u64 {
    majit_backend_wasm::jit_compile_cache_hits()
}

/// Diagnostic-only: read a guard-failure → bridge-trace gate tally from the
/// metainterp (`majit_metainterp::MC_DIAG`). Same export-not-import rationale
/// as `pyre_jit_bridge_diag`.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_mc_diag(i: u32) -> u64 {
    majit_metainterp::mc_diag(i as usize)
}

/// Diagnostic-only: the portal's own decision tallies
/// (`pyre_jit::eval::PORTAL_DIAG`, legend in `PORTAL_DIAG_LABELS`).
///
/// Every other portal instrument is an env-gated print — `PYRE_PORTAL_METATRACE`,
/// `MAJIT_PCSEQ` — and the guest has no environment, so none of them can fire
/// here.  This export is the only channel that says whether the wasm portal was
/// entered, whether any back edge reached its `CloseLoop` arm, and which filter
/// consumed the ones that did.  Exported rather than imported for the same
/// reason as `pyre_jit_bridge_diag`: an import would shift the JIT's own
/// function-index space.
///
/// `..._len` is exported beside the values so the runner can report a slot this
/// crate added but its own legend does not name, instead of dropping it. The
/// existing `bridge_diag` mirror has no such guard and relies on both sides
/// being edited together.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_portal_diag(i: u32) -> u64 {
    pyre_jit::eval::portal_diag(i as usize)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_portal_diag_len() -> u32 {
    pyre_jit::eval::PORTAL_DIAG_LABELS.len() as u32
}

/// Diagnostic-only: the full-body-walk decline census
/// (`pyre_jit_trace::jitcode_dispatch::census_entries`), which names the
/// `DispatchError` variant behind every aborted walk. The map is always
/// populated; only its `[fbw-census]` print is gated on
/// `PYRE_FBW_DEBUG_ABORT`, which a wasm guest can neither read nor print, so
/// these three exports are the only channel that carries the attribution out.
///
/// The names are `&'static str` in guest memory: `..._name` returns
/// `(ptr << 32) | len` for the host to slice out of the exported memory, and
/// `..._count` returns that entry's tally. Both index the same sorted order as
/// `census_entries`, which is stable for a given call — read the length first
/// and do not interleave with anything that could record a new decline.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_fbw_census_len() -> u32 {
    pyre_jit_trace::jitcode_dispatch::census_entries().len() as u32
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_fbw_census_name(i: u32) -> u64 {
    match pyre_jit_trace::jitcode_dispatch::census_entries().get(i as usize) {
        Some((name, _)) => ((name.as_ptr() as u64) << 32) | name.len() as u64,
        None => 0,
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_fbw_census_count(i: u32) -> u64 {
    pyre_jit_trace::jitcode_dispatch::census_entries()
        .get(i as usize)
        .map_or(0, |&(_, count)| count as u64)
}

/// Sign-stable JIT counters that read 0 in a healthy run, the same badness
/// fields the native `[jit-stats]` line reports (`pyrex` maybe_print_jit_stats).
/// A nonzero value means a trace aborted or an internal compile bug degraded
/// the trace, never a tuning choice — so `check.py`'s regression floor can gate
/// wasm on them exactly as it gates the native backends. Exported (not imported)
/// so reading them cannot perturb the JIT's table/function indices.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_loops_aborted() -> u64 {
    pyre_jit::eval::driver_pair().0.get_stats().loops_aborted as u64
}

/// The `Counters.ABORT_*` split behind `pyre_jit_loops_aborted`, indexed as
/// `majit_metainterp::jitprof::ABORT_COUNTER_KINDS`. On the native backends
/// that breakdown comes from `MAJIT_LOG`'s summary, but the wasm32 guest has
/// no environment, so a `loops_aborted` regression that only reproduces here
/// otherwise carries no reason at all. Exported rather than imported for the
/// same reason as `pyre_jit_bridge_diag`. The host prints it at
/// `PYRE_WASM_JIT_STATS` time.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_abort_diag(i: u32) -> u64 {
    pyre_jit::eval::driver_pair().0.abort_diag(i as usize)
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_internal_compile_panics() -> u64 {
    pyre_jit::eval::driver_pair()
        .0
        .get_stats()
        .internal_compile_panics as u64
}

/// The rest of the native `[jit-stats]` line. These are not sign-stable, so
/// each is gated on its own terms rather than against zero, and the comparison
/// is exact in both directions — a baseline that stopped describing the tree
/// has to be re-recorded whichever way it moved. `loops_compiled` and
/// `retraces_compiled` name a FALL as the regression and a rise as the gain;
/// `guard_failures` is the other way round, with no band around it.
/// `bridges_compiled` has no gain direction at all: a rise and a fall are both
/// reported as regressions, because a fall is the dead-bridge class — guards
/// that stop earning a bridge re-enter the metainterp forever, and a collapse
/// to zero stayed invisible for as long as this counter was left ungated on
/// the grounds that it moves both ways under ordinary tuning.
///
/// They come out of the same `JitStats` the fields above read, and were simply
/// never exported. Their absence was not neutral: a count-valued field missing
/// from a recorded baseline is skipped silently, so a change that stopped
/// compiling a loop altogether aborted nothing and *lowered* `guard_failures` —
/// both remaining gates read the loss as an improvement.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_loops_compiled() -> u64 {
    pyre_jit::eval::driver_pair().0.get_stats().loops_compiled as u64
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_retraces_compiled() -> u64 {
    pyre_jit::eval::driver_pair()
        .0
        .get_stats()
        .retraces_compiled as u64
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_bridges_compiled() -> u64 {
    pyre_jit::eval::driver_pair().0.get_stats().bridges_compiled as u64
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_guard_failures() -> u64 {
    pyre_jit::eval::driver_pair().0.get_stats().guard_failures as u64
}

/// Unlike the four above, this one is exported to be *read*, not gated: it
/// counts back-edge polls that found the eval-breaker word armed, which tracks
/// the collection schedule rather than anything the compiler decided.
/// `check.py` keeps it out of `JITSTATS_SNAPSHOT_FIELDS` for that reason, and
/// the runner prints it so a `guard_failures` move can be read against it.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_back_edge_polls() -> u64 {
    pyre_jit::eval::driver_pair().0.get_stats().back_edge_polls as u64
}

/// The descr-universe invariants, the remaining `JITSTATS_BADNESS_FIELDS`. The
/// native backends print these from `descr_set_jit_stats`; the guest has no
/// stderr, so it exports the counts and the runner prints the line. Without
/// these the regression floor read every one as 0 on wasm — a missing field is
/// read as absent-and-therefore-zero — so a wasm-only rise passed the gate.
///
/// `stale_absent` re-asks the `AbsentContainer` question against the universe as
/// it now stands, so it is only meaningful once the run has published
/// everything; the runner calls these after the guest's entry point returns.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_descr_set_absent() -> u64 {
    pyre_jit::descr_set_counts().absent
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_descr_set_ambiguous() -> u64 {
    pyre_jit::descr_set_counts().ambiguous
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_descr_set_stale_absent() -> u64 {
    pyre_jit::descr_set_counts().stale_absent
}

/// The denominator, so a wasm run that resolves nothing cannot read the same as
/// one that resolves everything. Host-dependent, hence excluded from the
/// committed snapshot (`JITSTATS_SNAPSHOT_FIELDS`) — it is reported for
/// diagnosis, not gated.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_descr_set_resolved() -> u64 {
    pyre_jit::descr_set_counts().resolved
}

/// The producer-side field-position invariants, the other two
/// `JITSTATS_BADNESS_FIELDS` members, exported for the same reason as the
/// `descr_set_*` block above.
///
/// Not redundant with the native backends' reading of them: the invariant is
/// that `index_in_parent` names the slot the field's BYTE OFFSET occupies, and
/// wasm32 lays every struct out on a 4-byte word (`symbolic.py:12`). A
/// producer's ranking is a function of the target's offsets, so a 64-bit host
/// reading zero says nothing about wasm.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_field_pos_spec_misplaced() -> u64 {
    pyre_jit::field_position_counts().spec_misplaced
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_field_pos_attached_misplaced() -> u64 {
    pyre_jit::field_position_counts().attached_misplaced
}

/// The denominators for the two above, on the same footing as
/// `pyre_jit_descr_set_resolved`: reported so a run that checked nothing cannot
/// read the same as one that checked everything, host-dependent, not gated.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_field_pos_spec_checked() -> u64 {
    pyre_jit::field_position_counts().spec_checked
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_field_pos_attached_checked() -> u64 {
    pyre_jit::field_position_counts().attached_checked
}

/// The four `derive_index_in_parent` dispositions, the rest of the
/// `field_position_census`.
///
/// `unresolved` — parent resolved, non-empty, field absent from it — is the
/// population the `attached_*` pair structurally cannot see: both halves of
/// that pair are counted inside a lookup that must first succeed, so a field
/// its parent does not contain is counted by neither and numerator and
/// denominator read healthy together. It is the only counter that reports it.
///
/// Export all dispositions explicitly: the runner can reject a requested
/// missing symbol, but it cannot detect a counter it never requested. These
/// counts describe name resolution rather than target-dependent offsets, but
/// their values still depend on the workload.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_field_pos_parent_absent() -> u64 {
    pyre_jit::field_position_counts().parent_absent
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_field_pos_parent_empty() -> u64 {
    pyre_jit::field_position_counts().parent_empty
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_field_pos_rederived() -> u64 {
    pyre_jit::field_position_counts().rederived
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_jit_field_pos_unresolved() -> u64 {
    pyre_jit::field_position_counts().unresolved
}

#[cfg(any(feature = "web", feature = "wasm-host"))]
static PANIC_HOOK: Once = Once::new();

#[cfg(any(feature = "web", feature = "wasm-host"))]
fn install_panic_hook() {
    PANIC_HOOK.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            // The module is built with the default `panic=abort`, so a panic
            // ends the run. Record the formatted message in linear memory
            // before the abort; the runner's `recover_panic_messages` scans for
            // it (the browser glue surfaces it the same way) so the real cause
            // is visible rather than a bare trap.
            let msg = format!("[pyre panic] {info}");
            OUTPUT_BUF.with(|buf| buf.borrow_mut().push_str(&msg));
        }));
    });
}

#[cfg(any(feature = "web", feature = "wasm-host"))]
thread_local! {
    static OUTPUT_BUF: RefCell<String> = RefCell::new(String::new());
    /// fd-2 capture. wasm32 has no descriptors, so everything the interpreter
    /// writes to stderr — tracebacks, warnings, `sys.stderr.write` — is
    /// collected here and handed to the embedder separately from stdout.
    static ERR_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// Process status the run ends with: 0, or `SystemExit`'s code, or 1 for
    /// an uncaught exception. `targetpypystandalone.py entry_point` returns
    /// the same value; the embedder exits with it.
    static EXIT_CODE: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
    /// Path the source came from, if the embedder named one. The guest is
    /// handed source text, not a file, so without it every code object
    /// compiles as `<string>` and a traceback can name neither the file nor
    /// the offending line.
    static SCRIPT_PATH: RefCell<Option<String>> = const { RefCell::new(None) };
}

#[cfg(any(feature = "web", feature = "wasm-host"))]
fn install_wasm_print_hook() {
    pyre_interpreter::set_print_hook(|b| {
        // The hook is handed raw fd-1 bytes; this embedder returns a `String`
        // to its host, so the decode happens here rather than inside the
        // interpreter, where it would corrupt every embedder's output.
        OUTPUT_BUF.with(|buf| buf.borrow_mut().push_str(&String::from_utf8_lossy(b)));
    });
    pyre_interpreter::set_stderr_hook(|b| {
        ERR_BUF.with(|buf| buf.borrow_mut().extend_from_slice(b));
    });
}

/// Run a Python source string and return the output as a string.
///
/// Host-agnostic core shared by the `web` (wasm-bindgen) and `wasm-host`
/// (C-ABI) entry points below.
#[cfg(any(feature = "web", feature = "wasm-host"))]
fn run_python_impl(source: &str) -> String {
    install_panic_hook();
    // Published targets match the descr FUNC and take the word stub. The host
    // hook remains for a callee with no single wasm function type, or a host
    // import outside the guest table.
    #[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
    majit_backend_wasm::install_residual_host_call();
    // Optional-module rclass aliases must be installed before the collector
    // is built: `init_jit_hooks` / `build_gc` snapshots the alias census.
    pyre_module::register();
    // Eagerly install pyre-jit's hooks (pyrex real_main does the same at
    // boot): the dict `eq_w` / `hash_w` / `hash_str` /
    // `compares_by_identity` trampolines must be live before
    // `install_builtin_modules` / `import` builds the first str- or
    // object-keyed dict, not only after the first JIT-traced bytecode
    // (`dict_eq_hook::missing_hash_hook` fails fast otherwise).
    pyre_jit::eval::init_jit_hooks();
    // Resolve the launcher options before the first `import sys`, as
    // `pyrex real_main` does. There is no command line here, so every option
    // starts at its default and only the environment the embedder handed over
    // folds in — which is why the table has to be installed first.
    #[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
    {
        use pyre_interpreter::launch_env::{self, LaunchFlags};
        // The host installed one environment. The launcher fold still takes
        // its snapshot through `set_launch_env`; that snapshot is this map,
        // not a third table.
        launch_env::set_launch_env(majit_metainterp::jit_env::entries());
        // `-B`, because the seam this target sees a filesystem through reads
        // and does not write.  The source loader would otherwise try to cache
        // the bytecode of every module it compiles, and `_write_atomic` reaches
        // for `posix.open`, which no read-only `posix` publishes.
        let flags = LaunchFlags {
            dont_write_bytecode: true,
            ..LaunchFlags::default()
        };
        match launch_env::finalize(flags) {
            Ok(flags) => {
                pyre_interpreter::importing::set_no_site(flags.no_site);
                pyre_interpreter::importing::set_runtime_flags(&flags);
            }
            Err(e) => {
                // `preconfig_init_utf8_mode` is fatal before the interpreter
                // exists; the guest cannot exit a process, so it reports the
                // same banner on fd 2 and returns the same status.
                install_wasm_print_hook();
                OUTPUT_BUF.with(|buf| buf.borrow_mut().clear());
                ERR_BUF.with(|buf| buf.borrow_mut().clear());
                pyre_interpreter::host_seam::emit_stderr(
                    format!(
                        "Fatal Python error: preconfig_init_utf8_mode: {}\n\
                         Python runtime state: preinitializing\n\n",
                        e.0
                    )
                    .as_bytes(),
                );
                EXIT_CODE.with(|c| c.set(1));
                return String::new();
            }
        }
    }
    // `init_sys_path` registers the builtin modules and stages the startup
    // `sys.path[0]` -- the script's directory, or `""` for source handed over
    // without a name -- which `import_site` prepends once `site` has run, as
    // `pyrex` does.
    #[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
    {
        let script_dir = SCRIPT_PATH
            .with(|p| p.borrow().clone())
            .and_then(|p| std::path::Path::new(&p).parent().map(|d| d.to_path_buf()))
            .unwrap_or_default();
        pyre_interpreter::importing::init_sys_path(&script_dir, script_dir.as_os_str());
    }
    #[cfg(not(all(target_arch = "wasm32", feature = "wasm-host")))]
    pyre_interpreter::importing::install_builtin_modules();
    // Give the import machinery a source of module bytes. The browser has no
    // filesystem, so the web build serves the embedded stdlib closure from an
    // in-memory VFS; the native-host (`wasm-host`) build reads the host filesystem
    // through `pyre_host.*` imports the runner satisfies.
    #[cfg(feature = "web")]
    {
        pyre_interpreter::importing::mount_embedded_stdlib(std::path::Path::new("/lib-python/3"));
        #[cfg(target_arch = "wasm32")]
        web_clock::install();
    }
    #[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
    {
        host_fs_provider::install();
        host_clock::install();
    }
    install_wasm_print_hook();
    OUTPUT_BUF.with(|buf| buf.borrow_mut().clear());
    ERR_BUF.with(|buf| buf.borrow_mut().clear());
    EXIT_CODE.with(|c| c.set(0));

    let execution_context = std::rc::Rc::new(PyExecutionContext::default());
    // Seed the TLS execution-context slot (pyrex real_main does the same at
    // boot). `getexecutioncontext().gettopframe()` must be live so a residual
    // `bh_call_fn_impl` from a blackhole resume — e.g. a `print(...)` after a
    // JIT-compiled loop — can resolve its parent frame instead of tripping the
    // fail-fast topframe assert.
    pyre_interpreter::call::set_last_exec_ctx(std::rc::Rc::as_ptr(&execution_context));
    // objspace.py `space.user_del_action`: the queue every registered
    // finalizer's trigger fires into (`finalizer_queue_trigger` reads
    // `ec.user_del_action`).  Without it the pointer stays null and the trigger
    // returns silently, so NOTHING is ever finalized on this entry point — no
    // `__del__`, no generator `finally`, not even under an explicit
    // `gc.collect()`.  `pyrex::setup_exec_context` installs it at boot; this
    // entry point is the other launcher and needs the same step.
    unsafe {
        let ec_ptr = std::rc::Rc::as_ptr(&execution_context) as *mut PyExecutionContext;
        (*ec_ptr).install_user_del_action();
        // The same step for the other periodic action: without it the ticker
        // cell is unregistered and `CheckSignalAction` never runs, so a
        // handler `signal.signal` recorded has nothing to deliver it -- which
        // on this target is the only delivery there is.  It also installs the
        // startup `SIGINT` handler, so `getsignal(SIGINT)` names
        // `default_int_handler` here as it does natively.
        pyre_interpreter::module::signal::interp_signal::install_signal_handling(&mut *ec_ptr);
    }
    // Register the __build_class__ callback. Class construction resolves the
    // live frame from the execution-context slot seeded above.
    pyre_interpreter::call::register_build_class();
    // `app_main.py` creates `__main__` and registers it in `sys.modules`
    // before anything else runs, as `pyrex run_source` does: the module
    // aliases the globals dict the program will run in, so
    // `__main__.__dict__`, `globals()` and `function.__globals__` share one
    // identity, and `sys.modules['__main__']` / `import __main__` resolve.
    let (canonical, main_module) =
        pyre_interpreter::app_main::prepare_main_module(&execution_context);

    let script_path = SCRIPT_PATH.with(|p| p.borrow().clone());
    let main_file = script_path.clone();

    // `sys.argv[0]` is the script, as `pymain_run_file` sets it, and `''` when
    // the guest was handed source with no name -- which is what `Py_Initialize`
    // leaves behind and what `unittest.main()` reads for its program name.  An
    // empty `sys.argv` is not a shape any launcher produces.
    //
    // Last before `sys` is built: the list this hands over is held in a plain
    // cell until then, and nothing roots it while it waits there, so the fewer
    // allocations in between the better.
    pyre_interpreter::importing::set_sys_argv(&[std::ffi::OsString::from(
        script_path.unwrap_or_default(),
    )]);

    // Create `sys` before user code runs, as `run_script_path` does. Nothing on
    // this entry point imports it otherwise, and the interpreter reads it
    // directly for its own work: `_warnings.show_warning` writes through
    // `sys.stderr` and gives up silently when the module is absent, so every
    // warning raised before the first `import sys` disappeared.
    let _ = pyre_interpreter::importing::importhook(
        rustpython_wtf8::Wtf8::new("sys"),
        canonical,
        pyre_object::PY_NULL,
        0,
        pyre_interpreter::call::getexecutioncontext(),
    );

    // pylifecycle.c init_importlib, the step `run_script_path` runs before
    // `site`: install the importlib bootstrap so `builtins.__import__` routes
    // imports through `sys.meta_path` / `sys.path_hooks` from the first user
    // statement. Without it `sys.modules` holds no `_frozen_importlib` and
    // every import on this launcher is served natively, so `sys.meta_path`
    // stays empty and no module carries a real `__spec__`.
    //
    // A failure (no reachable stdlib) is non-fatal, as it is natively: the
    // native importer keeps serving imports, the minimal-importer role.
    if let Err(e) = pyre_interpreter::importing::init_importlib_bootstrap(
        canonical,
        pyre_interpreter::call::getexecutioncontext(),
    ) {
        pyre_interpreter::host_seam::emit_stderr(
            format!("pyre: importlib bootstrap failed: {}\n", e.message_text()).as_bytes(),
        );
    }

    // `__spec__` / `__package__` / `__doc__`, and `__file__` / `__cached__`
    // for a script run by path.
    pyre_interpreter::app_main::seed_main_module_attrs(
        canonical,
        main_module,
        main_file.as_deref(),
    );

    // `app_main.py` binds `__main__.__loader__` and then imports `site` before
    // the program runs, as `pyrex run_source` does.
    #[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
    {
        let ec_ptr = pyre_interpreter::call::getexecutioncontext();
        pyre_interpreter::app_main::seed_main_loader(
            canonical,
            main_file.as_deref(),
            false,
            ec_ptr,
        );
        if !pyre_interpreter::app_main::import_site(
            pyre_interpreter::importing::no_site_flag(),
            canonical,
            ec_ptr,
        ) {
            EXIT_CODE.with(|c| c.set(1));
            return OUTPUT_BUF.with(|buf| buf.borrow().clone());
        }
    }

    let filename = SCRIPT_PATH.with(|p| p.borrow().clone());
    let filename = filename.as_deref().unwrap_or("<string>");
    // Source is compiled only once startup is complete, the order
    // `pymain_run_python` runs in: `site` is imported ahead of every path that
    // reaches user source.
    let code =
        match compile_source_with_filename(source, Mode::Exec, filename) {
            Ok(code) => code,
            Err(e) => {
                // `pyrex::run_source` renders the same `File "…", line N` + caret
                // banner on stderr and exits 1.
                pyre_interpreter::eprint_syntax_error(
                    &pyre_interpreter::compile_err_to_syntax_error(e, source, Mode::Exec),
                );
                EXIT_CODE.with(|c| c.set(1));
                return String::new();
            }
        };

    let mut frame = match pyre_interpreter::pyframe::PyFrame::new_with_context_and_globals(
        code,
        execution_context,
        canonical,
    ) {
        Ok(frame) => frame,
        Err(e) => return format!("Error: {e}"),
    };
    // Nothing reaches a GC frame but the `CURRENT_FRAME` / `f_backref` chain,
    // which this one joins only when `eval_with_jit` installs it. Publish it
    // until then (pyrex `run_source` does the same).
    let _main_frame_root = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(
        &*frame as *const pyre_interpreter::pyframe::PyFrame as pyre_object::PyObjectRef,
    );

    // catch_unwind to capture panics from JIT as error messages
    let eval_result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pyre_jit::eval::eval_with_jit(&mut frame, None)
    })) {
        Ok(r) => r,
        Err(_) => {
            EXIT_CODE.with(|c| c.set(1));
            let panic_msg = OUTPUT_BUF.with(|buf| buf.borrow().clone());
            return if panic_msg.is_empty() {
                "[pyre] unknown panic".to_string()
            } else {
                panic_msg
            };
        }
    };

    // `app_main.py` `run_toplevel` prints an uncaught exception before
    // `space.finish`. `SystemExit` only sets the status: resolving `e.code`
    // walks the exception type, which `finalize_runtime` then collects.
    let eval_result = match eval_result {
        Ok(result) => Ok(result),
        Err(e) if e.kind == pyre_interpreter::PyErrorKind::SystemExit => {
            EXIT_CODE.with(|c| c.set(pyre_interpreter::system_exit_code(&e)));
            Err(e)
        }
        Err(mut e) => {
            if !pyre_interpreter::error::print_exception_via_excepthook(&mut e) {
                pyre_interpreter::eprint_exception(&e, true);
            }
            EXIT_CODE.with(|c| c.set(1));
            Err(e)
        }
    };
    // `baseobjspace.py` `finish`: threading shutdown, atexit, then stream
    // flush and module teardown. Same function the native launcher runs.
    // Only the one-shot `wasm-host` guest exits here; the `web` build keeps
    // one interpreter alive across `run_python` calls.
    #[cfg(feature = "wasm-host")]
    pyre_interpreter::shutdown::finalize_runtime(
        frame.get_w_globals(),
        pyre_interpreter::call::getexecutioncontext(),
    );

    let mut output = OUTPUT_BUF.with(|buf| buf.borrow().clone());

    if let Ok(result) = eval_result
        && !result.is_null()
        && !unsafe { pyre_object::is_none(result) }
    {
        if !output.is_empty() && !output.ends_with('\n') {
            output.push('\n');
        }
        output.push_str(&format!("{}", PyDisplay(result)));
    }

    output
}

/// Bytes the run wrote to fd 2, draining the buffer.
#[cfg(any(feature = "web", feature = "wasm-host"))]
fn take_stderr() -> Vec<u8> {
    ERR_BUF.with(|buf| std::mem::take(&mut *buf.borrow_mut()))
}

/// Browser / JS entry point: marshalled by wasm-bindgen.
///
/// The browser has one output channel, so the fd-2 text (traceback, warnings)
/// is appended to the returned string rather than handed over separately the
/// way the C-ABI surface does it.
#[cfg(feature = "web")]
#[wasm_bindgen]
pub fn run_python(source: &str) -> String {
    let mut output = run_python_impl(source);
    let err = take_stderr();
    if !err.is_empty() {
        if !output.is_empty() && !output.ends_with('\n') {
            output.push('\n');
        }
        output.push_str(&String::from_utf8_lossy(&err));
    }
    output
}

/// Native-host (`wasm-host` feature) C-ABI surface.
///
/// wasm-bindgen is unavailable without a JS runtime, so the embedder talks
/// to the module through plain exports over linear memory:
///   1. `pyre_alloc(len)` → reserve `len` bytes, write the UTF-8 source there;
///   2. `pyre_run_python(ptr, len)` → run it, returns a packed `u64`
///      (`hi32` = result pointer, `lo32` = result byte length);
///   3. read the UTF-8 result, then `pyre_dealloc(ptr, len)` both buffers;
///   4. `pyre_take_stderr()` → the same packed pair for the run's fd-2 bytes,
///      and `pyre_exit_code()` → the status to exit with.
///
/// `pyre_set_script_path(ptr, len)` may precede step 2 to name the file the
/// source came from, and `pyre_set_env(ptr, len)` to install the one environment
/// the launcher, the collector and the JIT knobs all read — the guest has none
/// of its own. `pyre_set_launch_env`, `pyre_set_gc_env` and `pyre_set_jit_env`
/// upsert into that same map for a host that still sends one blob per group.
#[cfg(feature = "wasm-host")]
mod host_abi {
    use super::run_python_impl;
    use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};

    // Buffers crossing the boundary are allocated and freed through the
    // global allocator with a `Layout::array::<u8>(len)` derived purely
    // from `len`, so the host only ever needs to remember the length to
    // free a buffer soundly.

    /// Reserve `len` bytes in linear memory and return a pointer the host
    /// can write into. Pair every call with `pyre_dealloc`.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_alloc(len: usize) -> *mut u8 {
        if len == 0 {
            return std::ptr::NonNull::<u8>::dangling().as_ptr();
        }
        // Layout::array can only fail on overflow, impossible for a real
        // wasm linear-memory size.
        let layout = Layout::array::<u8>(len).expect("pyre_alloc: size overflow");
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        ptr
    }

    /// Release a buffer previously handed out by `pyre_alloc` or returned
    /// by `pyre_run_python`.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_dealloc(ptr: *mut u8, len: usize) {
        if ptr.is_null() || len == 0 {
            return;
        }
        let layout = Layout::array::<u8>(len).expect("pyre_dealloc: size overflow");
        unsafe { dealloc(ptr, layout) }
    }

    /// Diagnostic: total bytes the GC holds in the old generation (promoted +
    /// raw/large old-gen objects), or 0 if no GC is installed. Read by the host
    /// runner after a run to split GC-retained memory from host-heap growth.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_gc_oldgen_bytes() -> u64 {
        pyre_jit::wasm_gc_heap_stats().0 as u64
    }

    /// Diagnostic: bytes currently filled in the GC nursery, or 0 if no GC is
    /// installed. Companion to [`pyre_gc_oldgen_bytes`].
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_gc_nursery_bytes() -> u64 {
        pyre_jit::wasm_gc_heap_stats().1 as u64
    }

    /// Diagnostic: minor collections run so far, or 0 if no GC is installed.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_gc_minor_collections() -> u64 {
        pyre_jit::wasm_gc_collection_counts().0 as u64
    }

    /// Diagnostic: major collections run so far, or 0 if no GC is installed.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_gc_major_collections() -> u64 {
        pyre_jit::wasm_gc_collection_counts().1 as u64
    }

    /// Run the UTF-8 Python source at `ptr[..len]`. Returns a packed
    /// `(result_ptr << 32) | result_len`; the result is a UTF-8 byte buffer
    /// the host must free with `pyre_dealloc`.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_run_python(ptr: *const u8, len: usize) -> u64 {
        let result = match guest_str(ptr, len) {
            Some(source) => run_python_impl(&source),
            None => "Error: input buffer out of wasm memory bounds".to_string(),
        };

        pack_into_guest(result.into_bytes())
    }

    /// Name the file the next `pyre_run_python` source came from. It becomes
    /// the compiled code's filename — hence what a traceback prints and what
    /// its source-line lookup reads — and its directory heads `sys.path`.
    /// Without it the guest only has source text, so both fall back to
    /// `<string>`.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_set_script_path(ptr: *const u8, len: usize) {
        let path = guest_str(ptr, len).filter(|s| !s.is_empty());
        super::SCRIPT_PATH.with(|p| *p.borrow_mut() = path);
    }

    /// Install the one guest environment, as NUL-separated `NAME=VALUE` records.
    /// Replaces the map. The launcher, the collector and the JIT knobs all read
    /// it. A VALUE that is not valid UTF-8 is carried through undecoded:
    /// `PYTHONSAFEPATH` is set by any nonempty value. A record without `=`, or
    /// with an empty or non-UTF-8 name, is dropped. Call before
    /// `pyre_run_python`: the collector and the JIT knobs cache on first read.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_set_env(ptr: *const u8, len: usize) {
        let Some(blob) = guest_bytes(ptr, len) else {
            return;
        };
        majit_metainterp::jit_env::install(parse_env_blob(&blob));
    }

    /// Every name the one environment is worth being given: launcher, collector
    /// and JIT knobs, NUL-separated. The host frees the buffer with
    /// `pyre_dealloc`.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_env_names() -> u64 {
        let mut names = pyre_interpreter::launch_env::LAUNCH_ENV_NAMES.to_vec();
        names.extend_from_slice(pyre_jit::GC_ENV_NAMES);
        names.extend_from_slice(pyre_jit::JIT_ENV_NAMES);
        pack_into_guest(names.join("\0").into_bytes())
    }

    /// Upsert launcher names into the one environment. Same blob as
    /// [`pyre_set_env`]. Kept so a host that still sends this group does not
    /// write a second table.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_set_launch_env(ptr: *const u8, len: usize) {
        let Some(blob) = guest_bytes(ptr, len) else {
            return;
        };
        majit_metainterp::jit_env::extend(parse_env_blob(&blob));
    }

    /// The names [`pyre_set_launch_env`] is worth being given, as NUL-separated
    /// records in a buffer the host must free with `pyre_dealloc`. Returned so
    /// a host does not have to keep its own copy of the list in step with the
    /// interpreter's.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_launch_env_names() -> u64 {
        pack_into_guest(
            pyre_interpreter::launch_env::LAUNCH_ENV_NAMES
                .join("\0")
                .into_bytes(),
        )
    }

    /// Supply the `PYPY_GC_*` / `MAJIT_GC_STRESS` variables the collector
    /// sizes itself from (and, for stress, whether it collects on every
    /// allocation), in the same NUL-separated `NAME=VALUE` form as
    /// [`pyre_set_launch_env`]. The guest has no environment, so without this
    /// the nursery, the major-collection threshold and the growth rates all
    /// take their built-in defaults however the host was configured — and the
    /// threshold decides when the collector arms the eval-breaker word, which
    /// every compiled loop's back edge polls through a real guard.
    /// `pyre_gc_env_names` lists the names that are read.
    ///
    /// The collector reads them once, when it is built, which the first
    /// allocation does — so this must precede `pyre_run_python`, not merely the
    /// script's own first line.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_set_gc_env(ptr: *const u8, len: usize) {
        let Some(blob) = guest_bytes(ptr, len) else {
            return;
        };
        majit_metainterp::jit_env::extend(parse_env_blob(&blob));
    }

    /// The names [`pyre_set_gc_env`] is worth being given, as NUL-separated
    /// records in a buffer the host must free with `pyre_dealloc`. Returned for
    /// the same reason as [`pyre_launch_env_names`]: so a host does not keep its
    /// own copy of the list in step with the collector.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_gc_env_names() -> u64 {
        pack_into_guest(pyre_jit::GC_ENV_NAMES.join("\0").into_bytes())
    }

    /// Supply the `PYRE_NO_JIT` / `PYRE_JIT` / `MAJIT_NO_BRIDGE` variables the JIT knobs
    /// resolve against, in the same NUL-separated `NAME=VALUE` form as
    /// [`pyre_set_gc_env`]. The guest has no environment, so without this both
    /// knobs read as unset however the host was configured — and they change
    /// what the guest executes, not merely what it prints. `pyre_jit_env_names`
    /// lists the names that are read.
    ///
    /// Each site caches the answer in a `OnceLock` on first read, which the
    /// first compiled path does — so this must precede `pyre_run_python`.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_set_jit_env(ptr: *const u8, len: usize) {
        let Some(blob) = guest_bytes(ptr, len) else {
            return;
        };
        majit_metainterp::jit_env::extend(parse_env_blob(&blob));
    }

    /// The names [`pyre_set_jit_env`] is worth being given, as NUL-separated
    /// records in a buffer the host must free with `pyre_dealloc`. Returned for
    /// the same reason as [`pyre_gc_env_names`]: so a host does not keep its
    /// own copy of the list in step with the knobs.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_jit_env_names() -> u64 {
        pack_into_guest(pyre_jit::JIT_ENV_NAMES.join("\0").into_bytes())
    }

    /// Set `-P` / PYTHONSAFEPATH on the one environment for the next
    /// `pyre_run_python`. Kept for a host predating [`pyre_set_env`]. A host
    /// that also sends the environment blob must call this second: the blob
    /// upserts the same name.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_set_safe_path(enabled: u32) {
        if enabled != 0 {
            majit_metainterp::jit_env::upsert("PYTHONSAFEPATH", b"1".to_vec());
        } else {
            majit_metainterp::jit_env::remove("PYTHONSAFEPATH");
        }
    }

    fn parse_env_blob(blob: &[u8]) -> Vec<(String, Vec<u8>)> {
        blob.split(|&b| b == 0)
            .filter_map(|record| {
                let eq = record.iter().position(|&b| b == b'=')?;
                let name = std::str::from_utf8(&record[..eq]).ok()?;
                (!name.is_empty()).then(|| (name.to_string(), record[eq + 1..].to_vec()))
            })
            .collect()
    }

    /// Copy `[ptr, ptr+len)` out of linear memory as UTF-8. The embedder
    /// supplies the pair raw, so a range escaping linear memory is rejected
    /// (`None`) before a slice is formed rather than being undefined
    /// behaviour; an empty buffer reads as the empty string.
    fn guest_str(ptr: *const u8, len: usize) -> Option<String> {
        guest_bytes(ptr, len).map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Copy `[ptr, ptr+len)` out of linear memory undecoded, for a caller whose
    /// payload is not text. Same bounds contract as [`guest_str`]: the embedder
    /// supplies the pair raw, so a range escaping linear memory is rejected
    /// (`None`) before a slice is formed rather than being undefined behaviour;
    /// an empty buffer reads as an empty slice.
    fn guest_bytes(ptr: *const u8, len: usize) -> Option<Vec<u8>> {
        if ptr.is_null() || len == 0 {
            return Some(Vec::new());
        }
        let mem_bytes = core::arch::wasm32::memory_size(0).saturating_mul(65536);
        match (ptr as usize).checked_add(len) {
            Some(end) if end <= mem_bytes => {
                Some(unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec())
            }
            _ => None,
        }
    }

    /// Take the bytes the run wrote to fd 2 (traceback, warnings,
    /// `sys.stderr.write`) and hand them over the same way as the stdout
    /// result: a packed `(ptr << 32) | len` the host frees with
    /// `pyre_dealloc`. Draining, so a second call returns an empty buffer.
    ///
    /// wasm32 has no descriptors, so without this export everything the
    /// interpreter writes to stderr is discarded.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_take_stderr() -> u64 {
        pack_into_guest(super::take_stderr())
    }

    /// The walker's decline census (`[fbw-census] <name>: <count>` lines) as
    /// the same packed pair, for the host to print under
    /// `PYRE_WASM_FBW_CENSUS`.
    ///
    /// `PYRE_FBW_DEBUG_ABORT` selects that dump on a native build, but the
    /// guest reads no environment and writes to no descriptor, so neither the
    /// switch nor the `eprintln!` exists here. The counters are accumulated
    /// either way; this is the only way to read them off a wasm run. Reading,
    /// not draining, unlike `pyre_take_stderr`.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_fbw_census() -> u64 {
        pack_into_guest(pyre_jit_trace::jitcode_dispatch::census_report().into_bytes())
    }

    /// Arm the specialization census before `pyre_run_python`.
    ///
    /// `PYRE_FBW_SPEC_CENSUS` selects it on a native build; the guest reads no
    /// environment, so the switch has to arrive as a call. It must precede the
    /// run — the gate latches whether it is enabled on the first dispatch it
    /// sees, so a census armed after that stays off.
    ///
    /// Unlike the decline census above, these counters are NOT accumulated
    /// unarmed: the gate's whole point is that an unarmed run does no extra
    /// work, so arming is what makes the readout non-empty.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_fbw_spec_census_enable() {
        pyre_jit_trace::jitcode_dispatch::spec_census_enable();
    }

    /// The armed specialization census (`[spec-census] ...` lines) as the same
    /// packed pair, for the host to print under `PYRE_WASM_SPEC_CENSUS`. Byte
    /// for byte what `pyrex` prints natively, so a fold's `fired` count
    /// compares directly across the two backends. Reading, not draining.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_fbw_spec_census() -> u64 {
        pack_into_guest(pyre_jit_trace::jitcode_dispatch::spec_census_summary().into_bytes())
    }

    /// Arm the per-guard deopt census before `pyre_run_python`.
    ///
    /// `MAJIT_GUARD_CENSUS` selects it on a native build; the guest reads no
    /// environment, so the switch has to arrive as a call. It must precede the
    /// run — the census records on the deopt path and is off until armed.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_jit_guard_census_enable() {
        majit_metainterp::guard_census_enable();
    }

    /// Arm the compile census before `pyre_run_python`, and collect its lines
    /// for [`pyre_loop_census`] to hand back.
    ///
    /// `PYRE_LOOP_CENSUS` selects it on a native build. The guest needs the
    /// call for both halves of the usual reason: it reads no environment, and
    /// its stderr is a sink, so the lines have to be buffered and read out
    /// rather than printed where they are made.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_loop_census_enable() {
        majit_metainterp::loop_census::enable();
    }

    /// The lines the armed compile census recorded, as the same packed pair.
    /// The runner prints them verbatim: they are the same
    /// `[loop-census] <arm> <name>` records a native run puts on stderr, so
    /// `check.py` reads one shape on every backend. Draining, so a host that
    /// reads twice does not see the run's traces twice.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_loop_census() -> u64 {
        pack_into_guest(majit_metainterp::loop_census::report().into_bytes())
    }

    /// Arm the per-trace, per-entry-resume-key census before tracing starts.
    /// The runner owns `PYRE_WASM_TRACE_ENTRY_CENSUS`; the wasm guest cannot
    /// read its environment, and arming before compilation keeps the counter
    /// instructions absent from every unarmed trace module.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_jit_trace_entry_census_enable() {
        majit_backend_wasm::trace_entry_census_enable();
    }

    /// The armed census as the same packed pair, for the host to print under
    /// `PYRE_WASM_GUARD_CENSUS`. Same `top` as `pyrex` prints natively, so the
    /// two lines compare directly. Reading, not draining.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_jit_guard_census() -> u64 {
        pack_into_guest(majit_metainterp::guard_census_summary(12).into_bytes())
    }

    /// Guest-memory trace-entry census readout. The runner prints every line
    /// verbatim, preserving one greppable record per nonzero `(trace_id, key)`.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_jit_trace_entry_census() -> u64 {
        pack_into_guest(majit_backend_wasm::trace_entry_census_summary().into_bytes())
    }

    /// Trial-build errors recorded while deciding whether a bridge can be
    /// merged into its loop module.  The host surfaces this diagnostic text.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_jit_inline_trial_errors() -> u64 {
        pack_into_guest(majit_backend_wasm::inline_trial_errors().into_bytes())
    }

    /// Why each loop-closing bridge was refused a merge into its owner, with
    /// the `(slot, key)` that joins each record against the trace-entry census.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_jit_inline_declines() -> u64 {
        pack_into_guest(majit_backend_wasm::inline_declines().into_bytes())
    }

    /// Last `compile_loop` decline reason. Guest `eprintln` never reaches the
    /// host, and slots 25/26 do not cover `build_wasm_module` `Unsupported`.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_jit_compile_loop_last_error() -> u64 {
        pack_into_guest(majit_backend_wasm::compile_loop_last_error().into_bytes())
    }

    /// Status the last `pyre_run_python` ended with: `SystemExit`'s code, 1 for
    /// an uncaught exception or a `SyntaxError`, else 0. The host exits with
    /// it, as `pyrex` does with `targetpypystandalone.py entry_point`'s
    /// return value.
    #[unsafe(no_mangle)]
    pub extern "C" fn pyre_exit_code() -> i32 {
        super::EXIT_CODE.with(|c| c.get())
    }

    /// Copy `bytes` into a fresh guest buffer and pack its pointer and length
    /// into the `(hi32, lo32)` pair every result-returning export uses.
    fn pack_into_guest(bytes: Vec<u8>) -> u64 {
        let len = bytes.len();
        let ptr = pyre_alloc(len);
        if len != 0 {
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, len) };
        }
        ((ptr as u64) << 32) | (len as u64)
    }
}

/// Diagnostic-only: read one slot of the pyre walker's per-walk census
/// (slot layout in `pyre_jit_trace::trace::fbw_diag`). The guest cannot read
/// `PYRE_FBW_CENSUS`, so this is the only walk-level observability the wasm
/// target has. Exported (not an import) for the same function-index reason as
/// `pyre_jit_bridge_diag`. The host runner prints these at
/// `PYRE_WASM_JIT_STATS` time.
#[cfg(all(target_arch = "wasm32", feature = "wasm-host"))]
#[unsafe(no_mangle)]
pub extern "C" fn pyre_fbw_diag(i: u32) -> u64 {
    pyre_jit_trace::trace::fbw_diag::get(i as usize)
}
