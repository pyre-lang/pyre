//! MiniMarkGC — the core collector implementing the GcAllocator trait.
//!
//! A generational copying collector with:
//! - Bump-pointer nursery for young objects
//! - ArenaCollection old gen plus rawmalloc fallback, with incremental major sweep
//! - Write barrier with remembered set for old-to-young pointers
//!
//! Modeled after incminimark's minor/major collection.
use majit_ir::GcRef;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
/// Monotonic source for the durations the collector reports to its hooks.
///
/// `wasm32-unknown-unknown` has no clock: `Instant::now` panics with "time not
/// implemented on this platform", which is also why the interpreter does not
/// install its `time` module on that target. Report a zero duration there
/// rather than taking the panic — `duration`, `duration_min`, and
/// `duration_max` only reach `hook.py`'s stats objects and the `total_gc_time`
/// counter, and no collection decision reads any of them.
struct GcClock {
    #[cfg(not(target_arch = "wasm32"))]
    start: std::time::Instant,
}

impl GcClock {
    fn start() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            start: std::time::Instant::now(),
        }
    }

    fn elapsed_secs(&self) -> f64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start.elapsed().as_secs_f64()
        }
        #[cfg(target_arch = "wasm32")]
        {
            0.0
        }
    }
}

use crate::GcFlags;
use crate::address_dict::AddressMap;
use crate::header::{GcHeader, header_of};
use crate::hook::GcHooks;
use crate::nursery::{DEFAULT_NURSERY_SIZE, Nursery};
use crate::oldgen::OldGen;
use crate::rawrefcount::{self, RrcList};
use crate::trace::{ClassTypeLayout, TypeEntry, TypeInfo, TypeInfoLayout, TypeRegistry};
use crate::{FinalizerTriggerFn, GcAllocator};

/// `inspector.py HeapDumper.BUFSIZE` raw signed-word writer. Keeping the
/// fixed-size buffer here avoids materializing the full heap dump in memory.
struct HeapDumpWriter {
    fd: i32,
    buffer: Vec<isize>,
}

// Some targets and host-seam failures provide no OS errno. Use the POSIX EIO
// value for those cases so every heap-dump failure still carries an error code.
pub const HEAP_DUMP_EIO: i32 = 5;

impl HeapDumpWriter {
    const BUFSIZE: usize = 8192;

    fn new(fd: i32) -> Self {
        Self {
            fd,
            buffer: Vec::with_capacity(Self::BUFSIZE),
        }
    }

    fn write(&mut self, value: isize) -> Result<(), i32> {
        self.buffer.push(value);
        if self.buffer.len() == Self::BUFSIZE {
            self.flush()?;
        }
        Ok(())
    }

    fn write_marker(&mut self) -> Result<(), i32> {
        for value in [0, 0, 0, -1] {
            self.write(value)?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), i32> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let byte_len = self.buffer.len() * std::mem::size_of::<isize>();
        // SAFETY: the initialized `isize` elements occupy exactly `byte_len`
        // bytes and remain borrowed for the duration of the write.
        let bytes =
            unsafe { std::slice::from_raw_parts(self.buffer.as_ptr().cast::<u8>(), byte_len) };
        let write_result = if let Some(result) = crate::try_heap_dump_write(self.fd, bytes) {
            result
        } else {
            // Neither `write` nor `_write` exists here, so no call was made and
            // `errno` still names some unrelated earlier one. Report the dump's own
            // failure code instead of reading a stale `errno`.
            #[cfg(not(any(unix, windows)))]
            {
                Err(HEAP_DUMP_EIO)
            }
            #[cfg(any(unix, windows))]
            {
                #[cfg(unix)]
                let written: isize = unsafe {
                    libc::write(
                        self.fd,
                        self.buffer.as_ptr().cast::<libc::c_void>(),
                        byte_len,
                    )
                };
                // The CRT entry point is `_write`, but `libc` exports it under the
                // POSIX name with a `#[link_name = "_write"]` alias, so the Rust
                // path is `libc::write` on this target as well. It takes a
                // `c_uint` count and returns `c_int`, unlike the `size_t`/`ssize_t`
                // unix signature above.
                #[cfg(windows)]
                let written: isize = unsafe {
                    libc::write(
                        self.fd,
                        self.buffer.as_ptr().cast::<libc::c_void>(),
                        byte_len as libc::c_uint,
                    ) as isize
                };
                if written < 0 {
                    Err(std::io::Error::last_os_error()
                        .raw_os_error()
                        .unwrap_or(HEAP_DUMP_EIO))
                } else {
                    Ok(written)
                }
            }
        };
        let written = write_result?;
        if written as usize != byte_len {
            return Err(HEAP_DUMP_EIO);
        }
        self.buffer.clear();
        Ok(())
    }
}

/// Host predicate answering whether the consumer of a deferred major request
/// would act on one right now. Unset = never arm.
///
/// A standing switch is not enough, because the consumer — the interpreter
/// dispatch-loop safepoint — refuses in two different ways. It refuses for the
/// whole process when the interpreter GC is off, and it refuses per-thread,
/// moment to moment, whenever the eval loop is nested too deep for the frame
/// walker to see the full root set. Arming through the second kind of refusal
/// is not merely wasted: `threshold_reached` stays true until a major
/// completes, so every subsequent born-old allocation re-arms, the compiled
/// back-edge poll fails continuously, and the guard accumulates bridges. Once
/// a bridge is attached at each level the chain closes on itself and compiled
/// code stops returning to the dispatch loop at all — which starves the rest of
/// the word, `EB_ASYNC` included, so a signal goes undelivered for as long as
/// the loop runs. Measured on a depth-3 loop: SIGINT delivery went from 0.11 s
/// to over 30 s.
///
/// So ask the consumer. The probe is called only after `threshold_reached`, and
/// only on the born-old path, so it costs nothing on the ordinary nursery
/// allocation. Left unset the threshold is answered the way it always was — by
/// the major progress every minor collection drives.
pub type DeferredMajorRequestProbeFn = fn() -> bool;

crate::global_hook!(static DEFERRED_MAJOR_REQUEST_PROBE: DeferredMajorRequestProbeFn);

/// Install the predicate gating born-old major requests, or `None` to stop
/// arming them. See [`DEFERRED_MAJOR_REQUEST_PROBE`].
pub fn set_deferred_major_request_probe(probe: Option<DeferredMajorRequestProbeFn>) {
    DEFERRED_MAJOR_REQUEST_PROBE.set(probe);
}

/// `MAJIT_GC_YOUNG_RAWMALLOC` — whether an oversized allocation from a
/// collecting entry point may be born YOUNG and non-moving
/// (`external_malloc(..., alloc_young=True)`) instead of straight into the old
/// generation.
///
/// On by default, because born-old is the deviation: upstream gives every
/// oversized `malloc_fixedsize`/`malloc_varsize` result `alloc_young=True`, and
/// pyre's born-old block can only be reclaimed by a major. `=0` restores the
/// born-old behaviour without a rebuild, which is what makes a defect that
/// appears only once these objects can die at a minor bisectable against a
/// single binary.
///
/// Read once and cached: the gate sits on the large-allocation arm, and
/// `std::env::var_os` takes the environment lock and scans it linearly on every
/// call. Same shape as [`crate::gc_lifetime_log_enabled`].
pub(crate) fn young_rawmalloc_enabled() -> bool {
    // wasm32-unknown-unknown has no process environment, so the guest cannot
    // read the variable and takes the default.
    #[cfg(target_arch = "wasm32")]
    {
        true
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var_os("MAJIT_GC_YOUNG_RAWMALLOC").as_deref()
                != Some(std::ffi::OsStr::new("0"))
        })
    }
}

/// Whether a request armed now would be acted on. False with no probe installed.
fn deferred_major_request_wanted() -> bool {
    DEFERRED_MAJOR_REQUEST_PROBE
        .get()
        .is_some_and(|probe| probe())
}

/// Consume a pending request, reporting whether one was armed.
///
/// Clear it whether or not the caller goes on to collect: the bit is the
/// compiled back edge's deopt trigger, so one the caller keeps fires again on
/// the next back edge, and the next.
pub fn take_deferred_major_request() -> bool {
    majit_ir::eval_breaker_word::take_gc()
}

/// Arm a major collection request for the next root-complete interpreter
/// dispatch.  Hosts use this when collection is required for semantics rather
/// than because an allocation crossed the collector's threshold.
pub fn request_deferred_major_collection() {
    majit_ir::eval_breaker_word::set_gc();
}

/// Configuration for the MiniMarkGC.
pub struct GcConfig {
    /// Nursery size in bytes.
    pub nursery_size: usize,
    /// incminimark.py:326/472 `debug_tiny_nursery` — the bytes
    /// `collect_and_reserve` leaves free after each reservation, so a minor
    /// collection runs roughly every `debug_tiny_nursery` bytes of allocation.
    /// `None` is upstream's `-1` (off).  Set only by the `PYPY_GC_NURSERY`
    /// clamp; an explicitly configured `nursery_size` is left alone.
    pub debug_tiny_nursery: Option<usize>,
    /// Maximum object size that can be allocated in the nursery.
    /// Larger objects go directly to old gen.
    pub large_object_threshold: usize,
    /// incminimark.py:275: card_page_indices (0 disables card marking).
    /// Must be a power of two.
    pub card_page_indices: u32,
    /// gc/base.py `post_setup` — `PYPY_GC_DEBUG`, the level the collector's
    /// own self-checks are gated on. 0 is off; 1 installs the rotating
    /// nurseries; 2 additionally runs `debug_check_consistency` after every
    /// minor collection.
    pub debug: u8,
    /// incminimark.py `gc_nursery_debug` — `PYPY_GC_NURSERY_DEBUG`. Fill the
    /// recycled nursery with garbage rather than zeroes, and rotate the arena
    /// when the debug ring is installed.
    pub gc_nursery_debug: bool,
    /// translationoption.py `taggedpointers` (default off). When set,
    /// a small `int` may be stored as an unboxed immediate with an odd
    /// low bit; the collector must then skip such fields rather than read
    /// them as object headers (`is_valid_gc_object`, gc/base.py).
    pub taggedpointers: bool,
}

/// incminimark.py `post_setup` allocates this many spare nurseries, so a
/// retired arena is not handed back until the whole ring has turned.
pub const DEBUG_ROTATING_NURSERIES: usize = 6;

/// The variables [`GcConfig`] and [`MiniMarkGC::with_config`] resolve against,
/// for an embedder that has to hand its environment over rather than share it.
///
/// Published here so such a host does not keep its own copy of the list in step
/// with the collector. `MAJIT_GC_STRESS` is presence-only (any value, including
/// empty, opts in); the rest are size/rate pins. Tracing knobs stay absent
/// because nothing in this file reads them.
pub const GC_ENV_NAMES: &[&str] = &[
    "PYPY_GC_NURSERY",
    "PYPY_GC_MAX_PINNED",
    "PYPY_GC_INCREMENT_STEP",
    "PYPY_GC_MAJOR_COLLECT",
    "PYPY_GC_GROWTH",
    "PYPY_GC_MIN",
    "PYPY_GC_MAX",
    "PYPY_GC_MAX_DELTA",
    "PYPY_GC_NURSERY_DEBUG",
    "PYPY_GC_DEBUG",
    "MAJIT_GC_STRESS",
];

/// Upsert [`GC_ENV_NAMES`] into the one environment ([`majit_ir::environ`]).
/// A later call keeps names another setter already wrote. Call before the
/// first allocation: the values are read once, when the collector is built.
/// [`majit_ir::environ::install`] is what replaces the map.
pub fn set_supplied_env(entries: Vec<(String, String)>) {
    majit_ir::environ::extend(
        entries
            .into_iter()
            .map(|(name, value)| (name, value.into_bytes())),
    );
}

/// `read_from_env`'s `os.environ.get`: the process environment, then the one
/// map the host installed for a guest that has none.
fn env_var(varname: &str) -> Option<String> {
    majit_ir::environ::env_var(varname)
}

/// Presence of `varname`. Matches `std::env::var_os(name).is_some()` natively
/// (empty counts as set) and the same name in the one installed environment
/// on a guest that has no process env. The `gc_stress` reader is the only
/// production call; without that feature the name still travels in
/// [`GC_ENV_NAMES`] so a host can forward it.
#[cfg_attr(not(feature = "gc_stress"), allow(dead_code))]
fn env_is_set(varname: &str) -> bool {
    majit_ir::environ::env_is_set(varname)
}

/// env.py `_read_float_and_factor_from_env`. Parse `varname` as a float
/// with an optional `k`/`m`/`g` size suffix (optionally followed by `b`/`B`),
/// returning `(value, factor)`. `None` mirrors PyPy's `(0.0, 0)` absent /
/// unparseable result, which callers treat as "unset". `float(realvalue)`
/// accepts `inf`/`nan`, so this parser passes them through unchanged;
/// non-finite handling happens at the `int`/`r_uint` conversion sites, as
/// upstream where `int(inf)`/`r_uint(inf)` raise.
fn read_float_and_factor_from_env(varname: &str) -> Option<(f64, f64)> {
    let raw = env_var(varname)?;
    let mut value = raw.trim();
    if value.is_empty() {
        return None;
    }
    // env.py:20-21 — a trailing b/B is dropped before the k/m/g check.
    if value.len() > 1 && matches!(value.as_bytes().last(), Some(b'b' | b'B')) {
        value = &value[..value.len() - 1];
    }
    if value.is_empty() {
        return None;
    }
    // env.py:22-31 — a k/m/g suffix sets the factor; otherwise factor 1.
    let (number, factor) = match value.as_bytes().last().copied() {
        Some(b'k') | Some(b'K') => (&value[..value.len() - 1], 1024.0),
        Some(b'm') | Some(b'M') => (&value[..value.len() - 1], 1024.0 * 1024.0),
        Some(b'g') | Some(b'G') => (&value[..value.len() - 1], 1024.0 * 1024.0 * 1024.0),
        _ => (value, 1.0),
    };
    let parsed = number.parse::<f64>().ok()?;
    Some((parsed, factor))
}

/// env.py `read_from_env` / `read_uint_from_env`: `value * factor` as an
/// integer byte count. `None` (unset / unparseable / non-positive) lets callers
/// fall back to the default, mirroring PyPy's `if x > 0` guards. PyPy's
/// `read_uint_from_env` r_uint-wraps a negative product to a huge positive; pyre
/// treats non-positive as unset, differing only on nonsensical negative input.
/// A non-finite product is treated as unset too: `int(inf)`/`r_uint(inf)` raise
/// upstream, so we fall back to the default instead of a `usize::MAX` byte count.
fn read_uint_from_env(varname: &str) -> Option<usize> {
    let (value, factor) = read_float_and_factor_from_env(varname)?;
    let bytes = value * factor;
    (bytes.is_finite() && bytes > 0.0).then_some(bytes as usize)
}

/// incminimark.py:516-528 `PYPY_GC_MAX_PINNED`.  Unlike the byte-sized GC
/// options this is a plain integer count: an absent or empty value selects the
/// nursery-derived default, while an invalid or negative value leaves the
/// constructor's initial zero in place and therefore disables pinning.
///
/// A value too large for `usize` saturates. `int()` is unbounded upstream, so
/// the limit it installs there is simply larger than any reachable pin count —
/// which is what `usize::MAX` is here. The field is a `usize`, so saturating is
/// the only representable reading of "a cap this program can never hit".
fn read_max_number_of_pinned_objects() -> Option<usize> {
    let raw = env_var("PYPY_GC_MAX_PINNED")?;
    let value = raw.trim();
    if value.is_empty() {
        return None;
    }
    if value.starts_with('-') {
        return Some(0);
    }
    Some(
        value
            .parse::<u128>()
            .map(|count| count.min(usize::MAX as u128) as usize)
            .unwrap_or(0),
    )
}

/// env.py `read_float_from_env`: the plain float, but only when no size
/// factor was given (`factor != 1` → unset). Callers apply their own `> 1.0`
/// threshold gate.
fn read_float_from_env(varname: &str) -> Option<f64> {
    let (value, factor) = read_float_and_factor_from_env(varname)?;
    if factor != 1.0 {
        return None;
    }
    Some(value)
}

/// env.py `get_darwin_sysctl_signed`: read a signed integer sysctl by
/// name; 0 on any error.  Upstream reuses this helper for FreeBSD's
/// `hw.usermem` probe as well as the Darwin cache/memory probes.
#[cfg(any(target_os = "macos", target_os = "freebsd"))]
fn get_sysctl_signed(name: &[u8]) -> i64 {
    let mut val: i64 = 0;
    let mut len = std::mem::size_of::<i64>();
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr() as *const libc::c_char,
            &mut val as *mut i64 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && len == std::mem::size_of::<i64>() {
        val
    } else {
        0
    }
}

/// env.py `get_L2cache_darwin`. Returns the performance-cluster
/// L2 plus the legacy L3 cache size via `sysctl`, or -1 when it cannot be
/// determined.  Apple documents lower performance-level indices as faster
/// cores, so `hw.perflevel0.l2cachesize` is the cache relevant to the cores
/// running the mutator.  Intel Macs do not expose that key and retain the
/// legacy `hw.l2cachesize` fallback.
#[cfg(target_os = "macos")]
fn get_l2cache() -> i64 {
    let mut l2cache = get_sysctl_signed(b"hw.perflevel0.l2cachesize\0");
    if l2cache <= 0 {
        l2cache = get_sysctl_signed(b"hw.l2cachesize\0");
    }
    let mangled = l2cache + get_sysctl_signed(b"hw.l3cachesize\0");
    if mangled > 0 { mangled } else { -1 }
}

/// env.py `get_L2cache_linux2_cpuinfo`: find the smallest cache-size
/// entry across CPUs.  The label must begin immediately after a newline, and
/// the value must be expressed in K/kilobytes, exactly like upstream.
fn l2cache_from_cpuinfo(data: &[u8], label: &[u8]) -> i64 {
    let mut smallest = i64::MAX;
    for (line_index, line) in data.split(|byte| *byte == b'\n').enumerate() {
        // `_findend(data, '\n' + label, ...)` does not inspect the first line.
        if line_index == 0 || !line.starts_with(label) {
            continue;
        }
        let mut pos = label.len();
        while pos < line.len() && matches!(line[pos], b' ' | b'\t') {
            pos += 1;
        }
        if line.get(pos) != Some(&b':') {
            continue;
        }
        pos += 1;
        while pos < line.len() && matches!(line[pos], b' ' | b'\t') {
            pos += 1;
        }
        let start = pos;
        while pos < line.len() && line[pos].is_ascii_digit() {
            pos += 1;
        }
        if start == pos {
            continue;
        }
        let Ok(number) = std::str::from_utf8(&line[start..pos])
            .unwrap_or("")
            .parse::<i64>()
        else {
            continue;
        };
        while pos < line.len() && matches!(line[pos], b' ' | b'\t') {
            pos += 1;
        }
        if !matches!(line.get(pos), Some(b'K' | b'k')) {
            continue;
        }
        if let Some(bytes) = number.checked_mul(1024) {
            smallest = smallest.min(bytes);
        }
    }
    if smallest < i64::MAX { smallest } else { -1 }
}

/// File half of env.py `get_L2cache_linux2_cpuinfo`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn get_l2cache_linux_cpuinfo(filename: &str, label: &[u8]) -> i64 {
    std::fs::read(filename)
        .ok()
        .map_or(-1, |data| l2cache_from_cpuinfo(&data, label))
}

/// Parse the leading decimal plus K/k form used by Linux sysfs cache sizes.
fn parse_sysfs_cache_size(data: &[u8]) -> Option<i64> {
    let end = data.iter().position(|byte| !byte.is_ascii_digit())?;
    if end == 0 || !matches!(data.get(end), Some(b'K' | b'k')) {
        return None;
    }
    std::str::from_utf8(&data[..end])
        .ok()?
        .parse::<i64>()
        .ok()?
        .checked_mul(1024)
}

/// env.py `get_L2cache_linux2_system_cpu_index`: walk every
/// `cpuN/cache/indexM`, take the smallest L2 and L3, then sum them.  Missing
/// either level deliberately overflows the signed sentinel to a non-positive
/// value, matching translated RPython's `sys.maxint` arithmetic and fallback.
fn get_l2cache_linux_system_cpu_index(sys_cpu_root: &str) -> i64 {
    let mut cpu = 0usize;
    let mut l2cache = i64::MAX;
    let mut l3cache = i64::MAX;
    loop {
        let mut index = 0usize;
        loop {
            let cachedir = format!("{sys_cpu_root}/cpu{cpu}/cache/index{index}");
            let Ok(level_data) = std::fs::read(format!("{cachedir}/level")) else {
                break;
            };
            let Ok(level) = std::str::from_utf8(&level_data)
                .unwrap_or("")
                .trim()
                .parse::<i64>()
            else {
                break;
            };
            if level != 2 && level != 3 {
                index += 1;
                continue;
            }
            let Ok(size_data) = std::fs::read(format!("{cachedir}/size")) else {
                break;
            };
            if let Some(number) = parse_sysfs_cache_size(&size_data) {
                if level == 2 {
                    l2cache = l2cache.min(number);
                } else {
                    l3cache = l3cache.min(number);
                }
            }
            index += 1;
        }
        if index == 0 {
            break;
        }
        cpu += 1;
    }
    let mangled = l2cache.wrapping_add(l3cache);
    if mangled > 0 { mangled } else { -1 }
}

/// env.py `get_L2cache_linux2_sparc`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn get_l2cache_linux_sparc(sys_cpu_root: &str) -> i64 {
    let mut cpu = 0usize;
    let mut smallest = i64::MAX;
    loop {
        let filename = format!("{sys_cpu_root}/cpu{cpu}/l2_cache_size");
        let Ok(data) = std::fs::read(filename) else {
            break;
        };
        let Ok(number) = std::str::from_utf8(&data)
            .unwrap_or("")
            .trim()
            .parse::<i64>()
        else {
            break;
        };
        smallest = smallest.min(number);
        cpu += 1;
    }
    if smallest < i64::MAX { smallest } else { -1 }
}

/// env.py `get_L2cache_linux2`: select the literal upstream probe by
/// machine architecture.  Rust's architecture spellings differ slightly from
/// `os.uname()[4]`, so include both forms where the target exists.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn get_l2cache_linux_for(arch: &str, cpuinfo: &str, sys_cpu_root: &str) -> i64 {
    if arch.ends_with("86") || arch == "x86_64" {
        return get_l2cache_linux_cpuinfo(cpuinfo, b"cache size");
    }
    if matches!(arch, "alpha" | "ppc" | "powerpc") {
        return get_l2cache_linux_cpuinfo(cpuinfo, b"L2 cache");
    }
    if matches!(arch, "ia64" | "aarch64") {
        return get_l2cache_linux_system_cpu_index(sys_cpu_root);
    }
    if matches!(arch, "parisc" | "parisc64") {
        return get_l2cache_linux_cpuinfo(cpuinfo, b"D-cache");
    }
    if matches!(arch, "sparc" | "sparc64") {
        return get_l2cache_linux_sparc(sys_cpu_root);
    }
    -1
}

#[cfg(target_os = "linux")]
fn get_l2cache() -> i64 {
    get_l2cache_linux_for(
        std::env::consts::ARCH,
        "/proc/cpuinfo",
        "/sys/devices/system/cpu",
    )
}

/// env.py `get_L2cache = globals().get('get_L2cache_' + sys.platform,
/// lambda: -1)`.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn get_l2cache() -> i64 {
    -1
}

/// env.py `best_nursery_size_for_L2cache`. About half the L2 cache,
/// but only when it exceeds 8MB (so an L3-inclusive value does not size the
/// nursery off L3); otherwise the 4MB unknown-cache fallback
/// (`NURSERY_SIZE_UNKNOWN_CACHE`, env.py = `DEFAULT_NURSERY_SIZE`).
fn best_nursery_size_for_l2cache(l2cache: i64) -> usize {
    if l2cache > 8 * 1024 * 1024 {
        (l2cache / 2) as usize
    } else {
        DEFAULT_NURSERY_SIZE
    }
}

/// env.py `estimate_best_nursery_size`.
fn estimate_best_nursery_size() -> usize {
    best_nursery_size_for_l2cache(get_l2cache())
}

/// incminimark.py `minsize = 2 * (self.nonlarge_max + 1)` — pyre stores
/// `nonlarge_max + 1` directly as the large-object cutoff, so the doubling is
/// of the cutoff itself.  A nursery below this cannot hold one non-large
/// object, which is why upstream never lets the environment configure one.
pub(crate) const LARGE_OBJECT_THRESHOLD: usize = (16384 + 512) * std::mem::size_of::<usize>();

/// incminimark.py:459-473 — the environment-configured nursery size, and the
/// `debug_tiny_nursery` budget a request below `minsize` turns into.
///
/// ```python
/// newsize = env.read_from_env('PYPY_GC_NURSERY')
/// if newsize <= 0:
///     newsize = env.estimate_best_nursery_size()
///     if newsize <= 0:
///         newsize = defaultsize
/// if newsize < minsize:
///     self.debug_tiny_nursery = newsize & ~(WORD-1)
///     newsize = minsize
/// ```
///
/// `PYPY_GC_NURSERY=1` means "collect on every malloc", not "a one-byte
/// nursery": the arena stays at `minsize` so every non-large object still
/// fits, and the requested size becomes the budget
/// `collect_and_reserve` leaves free after each reservation.  Dropping the
/// clamp and keeping the raw value instead hands out an arena smaller than a
/// single object.
fn default_nursery_size() -> (usize, Option<usize>) {
    // estimate_best_nursery_size never returns <= 0 (its floor is the 4MB
    // unknown-cache fallback), so the final `defaultsize` arm is unreachable.
    let newsize = read_uint_from_env("PYPY_GC_NURSERY").unwrap_or_else(estimate_best_nursery_size);
    let minsize = 2 * LARGE_OBJECT_THRESHOLD;
    if newsize < minsize {
        (minsize, Some(newsize & !(std::mem::size_of::<usize>() - 1)))
    } else {
        (newsize, None)
    }
}

/// env.py `addressable_size = float(2**63)` for a 64-bit host: the most
/// memory the process could address, used as the fallback / upper clamp when
/// the real total-memory probe is unavailable or larger.
const ADDRESSABLE_SIZE: f64 = if usize::BITS == 64 {
    9_223_372_036_854_775_808.0 // 2**63
} else {
    2_147_483_648.0 // 2**31
};

/// env.py `get_total_memory_darwin`. Clamp a sysctl-probed total:
/// fall back to the addressable size when the probe failed (`<= 0`) and cap it
/// at the addressable size otherwise.
#[cfg(any(target_os = "macos", target_os = "freebsd"))]
fn get_total_memory_sysctl(result: i64) -> f64 {
    if result <= 0 {
        ADDRESSABLE_SIZE
    } else {
        (result as f64).min(ADDRESSABLE_SIZE)
    }
}

/// env.py `get_total_memory_linux`. Read `/proc/meminfo`, parse the
/// `MemTotal:` line (kB) into a byte count, then clamp: fall back to the
/// addressable size on read/parse failure (`result < 0.0`) and cap it at the
/// addressable size otherwise. The `< 0.0` failure sentinel — NOT the darwin
/// `<= 0` — must be kept (a probed `MemTotal: 0` is degenerate but not the
/// "probe failed" marker).
#[cfg(target_os = "linux")]
fn get_total_memory_linux(filename: &str) -> f64 {
    let mut result = -1.0_f64;
    // env.py:74-80 `os.read(fd, 4096)`: `MemTotal:` is always the first line
    // of `/proc/meminfo`, so the first 4 KiB always contain it.
    if let Ok(buf) = std::fs::read(filename) {
        let buf = &buf[..buf.len().min(4096)];
        let prefix = b"MemTotal:";
        if buf.starts_with(prefix) {
            // env.py `_skipspace`: advance past ' ' / '\t' after the prefix.
            let mut start = prefix.len();
            while start < buf.len() && (buf[start] == b' ' || buf[start] == b'\t') {
                start += 1;
            }
            // env.py:85-86: take the leading ASCII-digit run.
            let mut stop = start;
            while stop < buf.len() && buf[stop].is_ascii_digit() {
                stop += 1;
            }
            if start < stop {
                let digits = std::str::from_utf8(&buf[start..stop]).unwrap_or("");
                if let Ok(kb) = digits.parse::<f64>() {
                    result = kb * 1024.0; // env.py:88 assume kB
                }
            }
        }
    }
    if result < 0.0 {
        ADDRESSABLE_SIZE
    } else {
        result.min(ADDRESSABLE_SIZE)
    }
}

/// env.py `get_total_memory`. Total physical memory in bytes.
/// Linux reads `/proc/meminfo`; macOS reads `hw.memsize`; FreeBSD reads
/// `hw.usermem`; every other platform returns the addressable size
/// (env.py).
#[cfg(target_os = "linux")]
fn get_total_memory() -> f64 {
    get_total_memory_linux("/proc/meminfo")
}

#[cfg(target_os = "macos")]
fn get_total_memory() -> f64 {
    get_total_memory_sysctl(get_sysctl_signed(b"hw.memsize\0"))
}

#[cfg(target_os = "freebsd")]
fn get_total_memory() -> f64 {
    get_total_memory_sysctl(get_sysctl_signed(b"hw.usermem\0"))
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "freebsd")))]
fn get_total_memory() -> f64 {
    ADDRESSABLE_SIZE
}

impl Default for GcConfig {
    fn default() -> Self {
        // large_object = (16384+512)*8 from incminimark for 64-bit
        let (nursery_size, debug_tiny_nursery) = default_nursery_size();
        GcConfig {
            nursery_size,
            debug_tiny_nursery,
            large_object_threshold: LARGE_OBJECT_THRESHOLD,
            card_page_indices: 128,
            debug: read_uint_from_env("PYPY_GC_DEBUG")
                .unwrap_or(0)
                .min(u8::MAX as usize) as u8,
            gc_nursery_debug: read_uint_from_env("PYPY_GC_NURSERY_DEBUG").is_some_and(|v| v != 0),
            taggedpointers: false,
        }
    }
}

/// Root set: a list of locations that hold GcRef values the GC must trace.
///
/// Each root is a pointer to a GcRef-sized slot. During collection,
/// the GC reads the GcRef from this slot, traces it, and writes back
/// the (possibly updated) value.
pub struct RootSet {
    /// Stack roots: mutable pointers to GcRef slots on the stack or in frames.
    roots: Vec<*mut GcRef>,
}

unsafe impl Send for RootSet {}

impl RootSet {
    pub fn new() -> Self {
        RootSet { roots: Vec::new() }
    }

    /// Add a root. The pointer must remain valid until removed.
    ///
    /// # Safety
    /// The caller must ensure the pointer remains valid for the lifetime of the root.
    pub unsafe fn add(&mut self, root: *mut GcRef) {
        self.roots.push(root);
    }

    /// Remove a root.
    pub fn remove(&mut self, root: *mut GcRef) {
        // Host root brackets are stack-shaped, matching RPython's
        // shadowstack push/pop discipline.  Keep the general out-of-order
        // fallback for callers that hold overlapping guards, but make the
        // overwhelmingly common matching-pop path O(1).
        if self.roots.last().copied() == Some(root) {
            self.roots.pop();
            return;
        }
        // `shadowstack.py` has no by-value removal: `push_stack`/`pop_stack`
        // move one `root_stack_top`, so the newest registration is the one
        // retired.  Either copy leaves the same roots here, but taking the
        // front one breaks that shape and rescans on every following removal.
        if let Some(pos) = self.roots.iter().rposition(|r| *r == root) {
            self.roots.swap_remove(pos);
        }
    }

    /// Clear all roots.
    pub fn clear(&mut self) {
        self.roots.clear();
    }

    /// Number of roots.
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }
}

impl Default for RootSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Default card page shift: each card covers 2^7 = 128 array elements.
pub const DEFAULT_CARD_PAGE_SHIFT: u32 = 7;

/// incminimark.py major_collection_step major-collection state machine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GcState {
    Scanning,
    Marking,
    Sweeping,
    Finalizing,
}

impl GcState {
    fn encoded(self) -> u8 {
        match self {
            Self::Scanning => crate::GcStepTransition::STATE_SCANNING,
            Self::Marking => crate::GcStepTransition::STATE_MARKING,
            Self::Sweeping => crate::GcStepTransition::STATE_SWEEPING,
            Self::Finalizing => crate::GcStepTransition::STATE_FINALIZING,
        }
    }
}

/// State for incremental major marking.
///
/// Instead of doing a full mark-sweep in one pause, the marking work
/// is spread across multiple minor collections. Each minor collection
/// piggybacks an incremental marking step that processes a bounded
/// number of objects from the gray stack.
struct IncrementalMarkState {
    /// incminimark.py `objects_to_trace`: objects discovered by the ordinary
    /// marking walk.
    gray_stack: Vec<usize>,
    /// incminimark.py `more_objects_to_trace`: objects exposed or modified by
    /// the mutator while MARKING.  Keeping this separate is the termination
    /// mechanism at incminimark.py:2453-2470: only when the ordinary worklist
    /// consumed less than half its step does the collector swap this list in
    /// and drain it without a budget.
    more_gray_stack: Vec<usize>,
    /// Number of objects marked so far in this cycle.
    objects_marked: usize,
    /// Target number of bytes to trace per increment.
    ///
    /// Mirrors incminimark.py:448,500-504 `gc_increment_step`, whose runtime
    /// default is `nursery_size * 4` when `PYPY_GC_INCREMENT_STEP` is unset.
    mark_budget_per_step: usize,
}

impl IncrementalMarkState {
    fn new(nursery_size: usize) -> Self {
        // incminimark.py:500-504: an explicit positive byte budget wins;
        // otherwise mark four nursery sizes per incremental step.
        let mark_budget_per_step = read_uint_from_env("PYPY_GC_INCREMENT_STEP")
            .unwrap_or_else(|| nursery_size.saturating_mul(4))
            .max(1);
        IncrementalMarkState {
            gray_stack: Vec::new(),
            more_gray_stack: Vec::new(),
            objects_marked: 0,
            mark_budget_per_step,
        }
    }
}

struct FinalizerHandler {
    deque: VecDeque<usize>,
    trigger: FinalizerTriggerFn,
}

/// The MiniMark generational GC.
#[allow(non_snake_case)] // _T_IS_RPYTHON_INSTANCE_BYTE keeps the RPython spelling
pub struct MiniMarkGC {
    /// The nursery (young generation).
    nursery: Nursery,
    /// gc.py:525-531 published nursery-top slot read by generated inline
    /// allocation paths. The atomic publishes limit changes to generated code
    /// and Rust-side readers. The Box keeps its baked address stable even if
    /// the containing GC value moves.
    published_nursery_top: Box<AtomicUsize>,
    /// The old generation.
    oldgen: OldGen,
    /// Type registry for tracing objects.
    pub types: TypeRegistry,
    /// The registered `JITFRAME` type, once the frontend has named it
    /// (`GcAllocator::set_jitframe_type_id`).
    jitframe_type_id: Option<u32>,
    /// Root set.
    pub roots: RootSet,
    /// incminimark.py:344 `old_objects_pointing_to_young = AddressStack()`.
    /// Used by minor collection: the (mostly non-young) objects that may
    /// contain a pointer to a young object.  The write barrier populates it,
    /// appending an object as it clears that object's
    /// `GCFLAG_TRACK_YOUNG_PTRS`.  A young array object may enter by
    /// temporary mistake; the next minor collection removes it again, in
    /// [`Self::remove_young_arrays_from_old_objects_pointing_to_young`].
    old_objects_pointing_to_young: Vec<usize>,
    /// `gcreftracer.py` `llop.gc_writebarrier(tr)` analogue. A `GCREFTRACER`
    /// is an ordinary old object upstream, so writing its slots puts *that*
    /// MiniMark's `old_objects_pointing_to_young` on the hook for one minor.
    /// pyre's tables are Rust-owned, so the same remembered-set entry lives
    /// here instead of a process-global list.
    pending_gc_tables: Vec<Weak<crate::GcTable>>,
    /// incminimark.py:355 `prebuilt_root_objects = AddressStack()`.
    /// Immortal objects enter exactly once, when their first pointer write
    /// clears NO_HEAP_PTRS in the write barrier.
    prebuilt_root_objects: Vec<usize>,
    /// High-water capacity of [`Self::enumerate_labeled_root_walker_values`]'s
    /// snapshot, so the next one asks the allocator once instead of growing
    /// through it.  `collect_roots` materializes nothing upstream — it calls
    /// `_collect_ref_stk` per root — so the reallocation ladder is pyre's own,
    /// and it runs over every registered root on every major cycle.  A `Cell`
    /// because the snapshot is a `&self` reader.
    root_snapshot_capacity: std::cell::Cell<usize>,
    /// `gc/base.py:_totalroots_rpy`, the non-shrinking raw-list capacity used
    /// by `inspector.py:get_rpy_roots`. The raw result deliberately retains
    /// NULL padding; `pypy/module/gc/referents.py:get_rpy_roots` filters it at
    /// the app-level boundary.
    totalroots_rpy: usize,
    /// incminimark.py: objects with GCFLAG_CARDS_SET bit.
    /// Card bits are stored inline before each object's GcHeader.
    /// This list tracks which objects have at least one card bit set.
    old_objects_with_cards_set: Vec<usize>,
    /// incminimark.py:414 `self.young_objects_with_weakrefs =
    /// self.AddressStack()`. Nursery-resident WEAKREF objects pending
    /// minor-cycle target invalidation. Populated by `alloc_with_type`
    /// when the requested type's `TypeInfo.is_weakref` is set
    /// (incminimark.py `if contains_weakptr:`). Drained at end of
    /// minor cycle by `invalidate_young_weakrefs`
    /// (incminimark.py, :3058-3105).
    young_objects_with_weakrefs: Vec<usize>,
    /// incminimark.py:415 `self.old_objects_with_weakrefs =
    /// self.AddressStack()`. Old-gen WEAKREF objects whose target is
    /// also in old-gen. Populated by `invalidate_young_weakrefs` for
    /// survivors and by direct old-gen weakref allocation. Drained by
    /// `invalidate_old_weakrefs` during the sweep phase of a major
    /// collection (incminimark.py); the major-side
    /// consumer lands in the major-collection sweep phase.
    old_objects_with_weakrefs: Vec<usize>,
    /// incminimark.py:407 `self.young_objects_with_destructors =
    /// self.AddressStack()`. Nursery-resident objects whose type carries
    /// a lightweight `TypeInfo.destructor`. Populated by
    /// `register_young_object_if_needed` at allocation (incminimark.py:689
    /// `if needs_finalizer:`). Drained at end of minor cycle by
    /// `deal_with_young_objects_with_destructors` (incminimark.py,
    /// :2884-2895): a dead object's destructor runs, a survivor moves to
    /// `old_objects_with_destructors`.
    young_objects_with_destructors: Vec<usize>,
    /// incminimark.py:408 `self.old_objects_with_destructors =
    /// self.AddressStack()`. Old-gen objects with a destructor, populated
    /// by promotion from the young list and by direct old-gen allocation.
    /// Drained by `deal_with_old_objects_with_destructors`
    /// (incminimark.py, :2897-2912) before the major sweep:
    /// a VISITED object survives, a dying one's destructor runs.
    old_objects_with_destructors: Vec<usize>,
    /// incminimark.py:388-390.  Registrations first enter the probably-young
    /// deque as `(object, finalizer-handler-index)` pairs and are promoted by
    /// the next minor collection.
    probably_young_objects_with_finalizers: VecDeque<(usize, isize)>,
    /// incminimark.py:392-393.  Old objects still waiting to become
    /// unreachable, paired with their translated FinalizerQueue handler.
    old_objects_with_finalizers: VecDeque<(usize, isize)>,
    /// gc/base.py finalizer handlers: one death deque and trigger per queue.
    finalizer_handlers: Vec<FinalizerHandler>,
    /// `GCBase.run_old_style_finalizers`: heavyweight finalizers use the
    /// historical queue index -1 and are invoked after new-style triggers.
    run_old_style_finalizers: VecDeque<usize>,
    finalizer_lock: bool,
    /// incminimark.py:394 `self.enabled = True`.
    enabled: bool,
    /// True while [`do_collect_oldgen_nonmoving`](MiniMarkGC::do_collect_oldgen_nonmoving)
    /// is running. A non-moving major skips the leading minor, so unlike
    /// `do_collect_full` it marks through a *populated* nursery: `mark_object`
    /// / `seed_major_root` set `GcFlags::GCFLAG_VISITED` on reachable nursery objects
    /// that no sweep then clears (the sweep only walks old-gen). While this is
    /// set, every *young* object greyed this cycle is recorded in
    /// `oldgen_nonmoving_young_marks` so VISITED can be cleared as the
    /// strictly-last step — otherwise `copy_nursery_object` would memcpy a
    /// stale VISITED bit into the next minor's promoted copy.
    ///
    /// Young raw-malloced blocks are in that set for the same reason and a
    /// worse consequence: they are promoted in place rather than copied, so a
    /// stale bit is not merely inherited, it is the same header the next major
    /// reads. See [`Self::note_nonmoving_young_mark`].
    oldgen_nonmoving_active: bool,
    /// Set while `do_collect_full` drives a cycle it seeded itself. No
    /// mutator runs between that seed and the end of MARKING, so every
    /// extra-root slot still holds the value the seed already marked.
    full_collect_owns_cycle: bool,
    /// The cycle now in sweep had its mark seam run with
    /// [`Self::oldgen_nonmoving_active`] set. Only that cycle may release
    /// unvisited young rawmallocs: a non-moving entry that first finishes a
    /// cycle already in `Sweeping` has not marked the young rawmallocs
    /// allocated after that cycle's `sweep_prepare`.
    oldgen_nonmoving_marked: bool,
    /// Nursery payload addresses greyed during the current non-moving major
    /// (only populated while `oldgen_nonmoving_active`). Drained by the final
    /// VISITED-clear pass. The boolean records whether this visit also added
    /// TRACK_YOUNG_PTRS, so the adaptation restores the young object's exact
    /// incoming flag contract afterwards.
    oldgen_nonmoving_young_marks: Vec<(usize, bool)>,
    /// incminimark.py:3160,3175-3182 — the raw-refcount lists, identity tables
    /// and dead queue, empty until the embedder calls
    /// [`rawrefcount_init`](MiniMarkGC::rawrefcount_init).
    rrc: rawrefcount::RawRefCount,
    /// Configuration.
    config: GcConfig,
    /// Count of minor collections performed.
    pub minor_collections: usize,
    /// Count of major collections performed.
    pub major_collections: usize,
    /// [`minor_collections`](Self::minor_collections) as it stood when the last
    /// major *cycle* ended, so the difference is the minors run since.
    ///
    /// No incminimark counterpart: it exists for `gc.get_count`, whose second
    /// element 3.14 defines as the generation-0 collections run since
    /// generation 1 was last collected. A major here collects both older
    /// generations, so it is what resets that count.
    ///
    /// Sampled at the FINALIZING -> SCANNING transition and not in
    /// `finish_incremental_cycle`, which is a sweep-to-finalize seam rather
    /// than the end: `do_collect_full` runs a minor before every remaining
    /// step, so one more still runs after that seam, and sampling there leaves
    /// `gc.collect()` reporting a minor the collection had not yet finished
    /// running. Measured, not assumed — see
    /// `minors_accumulate_until_a_major_finishes`.
    minor_collections_at_major_end: usize,
    /// `incminimark.py:self.hooks`, supplied by the translated standalone
    /// target and restricted to its allocation-free low-level surface.
    hooks: GcHooks,
    /// Hook accounting captured at the MARKING -> SWEEPING seam.
    stat_ac_arenas_count: usize,
    stat_rawmalloced_total_size: usize,
    /// incminimark.py:387 `total_gc_time`, in seconds.
    total_gc_time: f64,
    /// incminimark.py major_collection_step `gc_state`.
    gc_state: GcState,
    /// State for incremental major collection.
    incr_state: IncrementalMarkState,
    /// incminimark.py:304 `self.major_collection_threshold`. After a major
    /// collection, the next one is triggered once the total old-gen size
    /// grows to this many times the surviving size (TRANSLATION_PARAMS
    /// default 1.82; `PYPY_GC_MAJOR_COLLECT` override).
    major_collection_threshold: f64,
    /// incminimark.py:305 `self.growth_rate_max`. Caps how fast the
    /// next-major threshold may grow from one collection to the next
    /// (TRANSLATION_PARAMS default 1.4; `PYPY_GC_GROWTH` override).
    growth_rate_max: f64,
    /// incminimark.py:307,488,562 `self.min_heap_size`. Floor below which the
    /// next-major threshold is never set: `max(PYPY_GC_MIN or nursery*8,
    /// nursery*major_collection_threshold)`. This floor is what stops
    /// allocation-heavy, tiny-live-set workloads (e.g. recursive fib) from
    /// thrashing major cycles off a near-zero surviving baseline.
    min_heap_size: f64,
    /// incminimark.py:308 `self.max_heap_size` (`PYPY_GC_MAX`; 0.0 = unbounded).
    max_heap_size: f64,
    /// incminimark.py:310,498 `self.max_delta` (`PYPY_GC_MAX_DELTA`, else
    /// `0.125 * get_total_memory()`). Caps the absolute next-major threshold
    /// growth per cycle.
    max_delta: f64,
    /// incminimark.py:309 `self.max_heap_size_already_raised`. Set true the
    /// first time a bounded major collection finds the heap already over
    /// `max_heap_size`; a second such event aborts the process
    /// (`out_of_memory`, minimarkpage.py -> fatalerror) instead of raising
    /// another MemoryError. Only reachable when `max_heap_size > 0`
    /// (`PYPY_GC_MAX` set), so it stays false in the default unbounded config.
    ///
    /// Upstream sets this immediately before the raise, so it reads as "the
    /// program has already been given a MemoryError". Pyre's channels deliver
    /// later, so the flag alone would read as "one has been armed"; the fatal
    /// rung restores the original meaning by also requiring that neither
    /// channel is still holding an undelivered one.
    max_heap_size_already_raised: bool,
    /// Set by `finish_incremental_cycle` when a bounded major collection leaves
    /// the heap over `max_heap_size`. The allocation that triggered the
    /// collection (`alloc_with_type`) reads and clears this and returns NULL,
    /// so the JIT `CHECK_MEMORY_ERROR` path / interpreter allocation chokepoint
    /// raises `MemoryError` (incminimark.py:2603-2615 `raise MemoryError`,
    /// lowered to the NULL-return the backend already understands).
    oom_pending: bool,
    /// Whether the step now running signalled a max-heap `MemoryError`, on
    /// either channel.
    ///
    /// Upstream raises where the two channels are armed, so its
    /// `major_collection_step` never reaches the collect-step hook on a breach.
    /// Suppressing that hook therefore needs a record of the arming rather than
    /// a reading of a channel: `oom_pending` alone answers for one channel
    /// only, and the eval-breaker bit is process-global and may be armed by a
    /// breach this step had nothing to do with. Cleared at the top of every
    /// `major_collection_step`.
    oom_signalled_this_step: bool,
    /// Byte size of the allocation that triggered the current nursery-full
    /// collection, carried across `do_collect_nursery` so
    /// `finish_incremental_cycle` can pass it to `threshold_reached`
    /// (incminimark's `major_collection_step(reserving_size)` argument). Carried
    /// on the collector rather than threaded through the public
    /// `do_collect_nursery` signature and its many call sites; reset to 0 by the
    /// triggering allocation once the collection returns.
    pending_reserving_size: usize,
    /// incminimark.py:568 `self.next_major_collection_initial`. The
    /// pre-reservation threshold; `set_major_threshold_from` grows the next
    /// threshold relative to this value, bounded by `growth_rate_max`.
    next_major_collection_initial: f64,
    /// incminimark.py:569 `self.next_major_collection_threshold`. A new
    /// incremental major cycle starts once `get_total_memory_used()` reaches
    /// this value (`threshold_reached`).
    next_major_collection_threshold: f64,
    /// incminimark.py:2492,3011-3020 `self.kept_alive_by_finalizer`.
    ///
    /// During finalization ordering, unreachable object graphs are marked so
    /// their app-level finalizers can still run.  They remain physically live
    /// after this sweep, but upstream excludes exactly these bytes from the
    /// survivor baseline used to size the next major threshold; otherwise a
    /// workload consisting mostly of finalizable garbage grows that threshold
    /// forever (PyPy issue #2590).
    kept_alive_by_finalizer: usize,
    /// Bytes promoted to old gen since the current incremental cycle started.
    ///
    /// Mirrors incminimark's `size_objects_made_old`.
    bytes_made_old_since_cycle: usize,
    /// Promotion credit granted by completed major-GC steps within the current
    /// incremental cycle.
    ///
    /// Mirrors incminimark's `threshold_objects_made_old`.
    threshold_bytes_made_old: usize,
    /// incminimark.py:1816,2174,2232 `nursery_surviving_size`: allocator-size
    /// bytes promoted by the most recent minor collection.  The MARKING step
    /// traces at least twice this many bytes so a high-survival nursery cannot
    /// make the gray frontier grow faster than it is consumed.
    nursery_surviving_size: usize,
    /// `IncrementalMiniMarkGC.setup`: sorted addresses of pinned-object
    /// headers, followed by the nursery end.  The current header is held in
    /// `nursery_top`; `collect_and_reserve` consumes this deque to enter the
    /// next free gap without collecting.
    nursery_barriers: VecDeque<usize>,
    /// `IncrementalMiniMarkGC._minor_collection`: AddressStack of header
    /// addresses for pinned objects reached by this minor.  VISITED on the
    /// object header prevents duplicates, exactly as upstream does; pin state
    /// itself also lives in the header, never in an address side table.
    surviving_pinned_objects: Vec<usize>,
    /// `IncrementalMiniMarkGC.any_pinned_object_kept`: whether the preceding
    /// minor left at least one pinned object in the nursery.
    any_pinned_object_kept: bool,
    old_objects_pointing_to_pinned: Vec<usize>,
    /// incminimark.py `updated_old_objects_pointing_to_pinned`: set whenever
    /// the list above changes, and read by `writebarrier_before_copy` alone.
    updated_old_objects_pointing_to_pinned: bool,
    /// incminimark.py:311,430 `max_number_of_pinned_objects` and
    /// `pinned_objects_in_nursery`.
    max_number_of_pinned_objects: usize,
    pinned_objects_in_nursery: usize,
    /// minimark.py:338 `nursery_objects_shadows = AddressDict()`.
    /// Maps nursery object payload address → pre-allocated old-gen
    /// shadow payload address.  When `id()` or `identityhash()` is
    /// called on a nursery object, a shadow is allocated in old-gen
    /// and registered here.  `copy_nursery_object` copies to the
    /// shadow instead of a fresh allocation.  Cleared after each
    /// minor collection.
    nursery_objects_shadows: AddressMap<usize>,
    /// llsupport/gc.py get_typeid_from_classptr_if_gcremovetypeptr vtable→typeid mapping. RPython derives this
    /// arithmetically from the GC `type_info_group` base; pyre's GC
    /// keeps an explicit table because frontends register vtables
    /// independently of any translator pipeline. This has RPython
    /// AddressDict hashing: keys are addresses and insertion order is not
    /// observable at any call site.
    vtable_to_type_id: AddressMap<u32>,
    /// `gc.py _setup_guard_is_object` instance state.
    /// `_infobits_offset` is the byte offset of `infobits` inside
    /// `TYPE_INFO`; `_infobits_offset_plus` is the additional offset
    /// past leading zero bytes of the little-endian flag word.
    /// `_T_IS_RPYTHON_INSTANCE_BYTE` is the single non-zero byte
    /// pulled out of the flag word.
    ///
    /// Mirrors RPython's GcLLDescr_framework instance fields exactly
    /// — they're computed once at construction by `_setup_guard_is_object`
    /// and read by `get_translated_info_for_guard_is_object`. The
    /// `_T_IS_RPYTHON_INSTANCE_BYTE` field name keeps the RPython
    /// upper-case spelling so a `git grep` across the two trees
    /// finds the same name on both sides.
    _infobits_offset: usize,
    _infobits_offset_plus: usize,
    _T_IS_RPYTHON_INSTANCE_BYTE: u8,
    /// incminimark.py:315-317: log2(card_page_indices).
    /// 0 if card marking is disabled.
    card_page_shift: u32,
    /// When true, `alloc_with_type` forces a full collection before every
    /// allocation (see the `gc_stress` cargo feature). Off by default; opted in
    /// per instance via [`MiniMarkGC::set_stress_collect`]. Exists only under
    /// the feature so the default build carries no extra field or branch.
    #[cfg(feature = "gc_stress")]
    stress_collect: bool,
}

impl MiniMarkGC {
    /// Create a new GC with default configuration.
    pub fn new() -> Self {
        Self::with_config(GcConfig::default())
    }

    /// Create a new GC with custom configuration.
    pub fn with_config(config: GcConfig) -> Self {
        let nursery_size = config.nursery_size;
        // The host-allocation decay clock ticks once per nursery's worth of
        // host bytes (`nursery_clock`).
        crate::nursery_clock::set_quantum(nursery_size);

        // incminimark.py:261,268,475-481 — major_collection_threshold /
        // growth_rate_max from TRANSLATION_PARAMS, overridable by env.
        let major_collection_threshold = read_float_from_env("PYPY_GC_MAJOR_COLLECT")
            .filter(|v| *v > 1.0)
            .unwrap_or(1.82);
        let growth_rate_max = read_float_from_env("PYPY_GC_GROWTH")
            .filter(|v| *v > 1.0)
            .unwrap_or(1.4);
        // incminimark.py:483-488 — min_heap_size: PYPY_GC_MIN, else nursery*8.
        let mut min_heap_size = read_uint_from_env("PYPY_GC_MIN")
            .map(|v| v as f64)
            .unwrap_or(nursery_size as f64 * 8.0);
        // incminimark.py:490-492 — max_heap_size: PYPY_GC_MAX, else 0 (unbounded).
        let max_heap_size = read_uint_from_env("PYPY_GC_MAX")
            .map(|v| v as f64)
            .unwrap_or(0.0);
        // incminimark.py:494-498 — max_delta: PYPY_GC_MAX_DELTA, else
        // 0.125 * env.get_total_memory().
        let max_delta = read_uint_from_env("PYPY_GC_MAX_DELTA")
            .map(|v| v as f64)
            .unwrap_or_else(|| 0.125 * get_total_memory());
        // incminimark.py — allocate_nursery floors min_heap_size by
        // nursery_size * major_collection_threshold.
        min_heap_size = min_heap_size.max(nursery_size as f64 * major_collection_threshold);

        let mut nursery = Nursery::new(config.nursery_size);
        // incminimark.py `post_setup`: under `PYPY_GC_DEBUG` a retired nursery
        // is protected rather than reused, so a pointer left behind in one
        // faults on the next read instead of being answered by whatever was
        // allocated over it.
        if config.debug != 0 {
            nursery.install_debug_rotating_nurseries(DEBUG_ROTATING_NURSERIES);
        }
        nursery.set_nursery_debug(config.gc_nursery_debug);
        // incminimark.py:516-528. `nonlarge_max + 1` is the large-object
        // cutoff; pyre stores that cutoff directly in the configuration.
        let max_number_of_pinned_objects =
            read_max_number_of_pinned_objects().unwrap_or_else(|| {
                config
                    .nursery_size
                    .checked_div(config.large_object_threshold.saturating_mul(2))
                    .unwrap_or(0)
            });
        let published_nursery_top = Box::new(AtomicUsize::new(nursery.top_ptr() as usize));
        let mut gc = MiniMarkGC {
            nursery,
            published_nursery_top,
            oldgen: OldGen::new(),
            types: TypeRegistry::new(),
            jitframe_type_id: None,
            roots: RootSet::new(),
            old_objects_pointing_to_young: Vec::new(),
            pending_gc_tables: Vec::new(),
            prebuilt_root_objects: Vec::new(),
            root_snapshot_capacity: std::cell::Cell::new(0),
            totalroots_rpy: 0,
            old_objects_with_cards_set: Vec::new(),
            young_objects_with_weakrefs: Vec::new(),
            old_objects_with_weakrefs: Vec::new(),
            young_objects_with_destructors: Vec::new(),
            old_objects_with_destructors: Vec::new(),
            probably_young_objects_with_finalizers: VecDeque::new(),
            old_objects_with_finalizers: VecDeque::new(),
            finalizer_handlers: Vec::new(),
            run_old_style_finalizers: VecDeque::new(),
            finalizer_lock: false,
            enabled: true,
            oldgen_nonmoving_active: false,
            full_collect_owns_cycle: false,
            oldgen_nonmoving_marked: false,
            oldgen_nonmoving_young_marks: Vec::new(),
            rrc: rawrefcount::RawRefCount::default(),
            config,
            minor_collections: 0,
            major_collections: 0,
            minor_collections_at_major_end: 0,
            hooks: GcHooks,
            stat_ac_arenas_count: 0,
            stat_rawmalloced_total_size: 0,
            total_gc_time: 0.0,
            gc_state: GcState::Scanning,
            incr_state: IncrementalMarkState::new(nursery_size),
            major_collection_threshold,
            growth_rate_max,
            min_heap_size,
            max_heap_size,
            max_delta,
            max_heap_size_already_raised: false,
            oom_pending: false,
            oom_signalled_this_step: false,
            pending_reserving_size: 0,
            // incminimark.py:568-569 — both initialized to min_heap_size,
            // then refined by set_major_threshold_from(0.0) below.
            next_major_collection_initial: min_heap_size,
            next_major_collection_threshold: min_heap_size,
            kept_alive_by_finalizer: 0,
            bytes_made_old_since_cycle: 0,
            threshold_bytes_made_old: 0,
            nursery_surviving_size: 0,
            nursery_barriers: VecDeque::new(),
            surviving_pinned_objects: Vec::new(),
            any_pinned_object_kept: false,
            old_objects_pointing_to_pinned: Vec::new(),
            updated_old_objects_pointing_to_pinned: false,
            max_number_of_pinned_objects,
            pinned_objects_in_nursery: 0,
            nursery_objects_shadows: AddressMap::default(),
            vtable_to_type_id: AddressMap::default(),
            _infobits_offset: 0,
            _infobits_offset_plus: 0,
            _T_IS_RPYTHON_INSTANCE_BYTE: 0,
            card_page_shift: 0,
            // gc.py:603-617 has no analogue; seeded from the `MAJIT_GC_STRESS`
            // env var so a whole binary can be stressed without code edits,
            // while individual tests use `set_stress_collect`. Presence, not
            // a numeric parse: empty is on, matching `var_os(...).is_some()`.
            // `env_is_set` also sees the embedder table (`GC_ENV_NAMES`) so a
            // wasm guest with no process environment still opts in.
            #[cfg(feature = "gc_stress")]
            stress_collect: env_is_set("MAJIT_GC_STRESS"),
        };
        // incminimark.py:314-317
        if gc.config.card_page_indices > 0 {
            gc.card_page_shift = 0;
            while (1u32 << gc.card_page_shift) < gc.config.card_page_indices {
                gc.card_page_shift += 1;
            }
        }
        // incminimark.py — allocate_nursery finalizes the first threshold.
        gc.set_major_threshold_from(0.0, 0.0);
        gc._setup_guard_is_object();
        gc
    }

    /// Refresh the gc.py:525-531 published nursery-top slot from the real
    /// allocator bound.
    fn refresh_published_nursery_top(&mut self) {
        self.published_nursery_top
            .store(self.nursery.top_ptr() as usize, Ordering::Release);
    }

    // ── incminimark.py card_marking_words_for_length card marking geometry ──

    /// incminimark.py card_marking_words_for_length: number of machine words needed for card bits.
    fn card_marking_words_for_length(&self, length: usize) -> usize {
        const LONG_BIT: usize = usize::BITS as usize;
        const LONG_BIT_SHIFT: usize = usize::BITS.trailing_zeros() as usize;
        (length + (LONG_BIT << self.card_page_shift) - 1)
            >> (self.card_page_shift as usize + LONG_BIT_SHIFT)
    }

    /// incminimark.py card_marking_bytes_for_length: number of bytes needed for card bits.
    fn card_marking_bytes_for_length(&self, length: usize) -> usize {
        (length + (8 << self.card_page_shift) - 1) >> (self.card_page_shift as usize + 3)
    }

    /// incminimark.py get_card: address of card byte at `byteindex`.
    /// Card bytes are stored in reverse order before the GcHeader.
    /// `obj` is the payload address (after GcHeader).
    #[inline]
    fn get_card_ptr(obj: usize, byteindex: usize) -> *mut u8 {
        // addr_byte = obj - size_gc_header  (= GcHeader address)
        // return addr_byte + (~byteindex)   (= addr_byte - 1 - byteindex)
        (obj - GcHeader::SIZE - 1 - byteindex) as *mut u8
    }

    /// incminimark.py external_malloc: allocate a large object with optional card
    /// marker bits prepended before the GcHeader.
    ///
    /// `has_gc_ptrs_in_var` should be true for arrays containing GC pointers
    /// (i.e. `has_gcptr_in_varsize(typeid)` in RPython).
    ///
    /// This is the `alloc_young=False` arm;
    /// [`Self::try_alloc_young_nonmoving_with_cards`] is the other, and
    /// [`Self::alloc_varsize_nonmoving_prefer_young`] is the dispatch between
    /// them that upstream performs inside the one function.
    ///
    /// A card array arrives owing two things that the per-item barrier does not
    /// supply, because a card records a *position*: a bulk copy into it owes
    /// [`Self::writebarrier_before_copy`], and any move of items within it — a
    /// list insert, remove, splice or reverse — owes
    /// [`Self::writebarrier_before_move`], which is what generalizes the cards
    /// back to "any item may be young" before the positions change.
    ///
    /// The tail is [`finish_alloc_in_oldgen`](Self::finish_alloc_in_oldgen)'s,
    /// which is how upstream reaches it too: `external_malloc`'s arms rejoin at
    /// "Common code to fill the header and length of the object". A destructor
    /// or weakref type born here has no other route onto the old-generation
    /// lists that give it a destructor run and a `weakptr` invalidation, and
    /// the rawmalloc bytes have no other route into the major-progress
    /// accounting.
    ///
    /// The zero-fill is *not* upstream's — `external_malloc` says "the memory
    /// returned is not cleared", and for compiled code `clear_varsize_gc_fields`
    /// emits the `ZeroArray` that covers it. It is here because the alternative
    /// is a birth whose contract depends on which arm it took:
    /// [`try_alloc_young_nonmoving_with_cards`](Self::try_alloc_young_nonmoving_with_cards)
    /// clears, and so does the [`alloc_in_oldgen_clear`](Self::alloc_in_oldgen_clear)
    /// this stands beside in `alloc_with_type_slow`, so a caller of
    /// [`alloc_varsize_nonmoving_prefer_young`](Self::alloc_varsize_nonmoving_prefer_young)
    /// would otherwise get zeroed memory or dirty memory according to whether
    /// the young rawmalloc happened to take the request.
    pub fn alloc_in_oldgen_with_cards(
        &mut self,
        type_id: u32,
        total_size: usize,
        length: usize,
        has_gc_ptrs_in_var: bool,
    ) -> GcRef {
        let (card_header_bytes, extra_flags) =
            self.card_header_and_flags(total_size, length, has_gc_ptrs_in_var, false);
        let ptr = self
            .oldgen
            .alloc_with_card_header(total_size, card_header_bytes);
        let obj = self.finish_alloc_in_oldgen(type_id, total_size, ptr, extra_flags);
        Self::raw_memclear(obj, total_size);
        obj
    }

    /// incminimark.py, the card question `external_malloc` asks once
    /// for both of its arms. Returns `(cardheadersize, extra_flags)`.
    ///
    /// Three clauses deny cards, and the third is a *size* test: upstream
    /// spells it `raw_malloc_usage(totalsize) <= self.nonlarge_max`, and
    /// `large_object_threshold` holds `nonlarge_max + 1` (see
    /// [`LARGE_OBJECT_THRESHOLD`]), so equality is already the large arm. It is
    /// written here the way every other large-arm test in this file is
    /// written, so an object gets cards exactly when it took the arm that can
    /// hold them — a small object that reached rawmalloc only because the
    /// nursery was full must not come back carrying a card array.
    ///
    fn card_header_and_flags(
        &self,
        total_size: usize,
        length: usize,
        has_gc_ptrs_in_var: bool,
        alloc_young: bool,
    ) -> (usize, GcFlags) {
        if self.card_page_shift == 0
            || !has_gc_ptrs_in_var
            || total_size < self.config.large_object_threshold
        {
            return (0, GcFlags::empty());
        }
        let extra_words = self.card_marking_words_for_length(length);
        // incminimark.py `external_malloc`: `cardheadersize = WORD *
        // extra_words`.  On wasm32 `GcHeader` retains an eight-byte physical
        // prefix solely to align native Rust payloads, so round the logical
        // card bytes up to that physical alignment; the bytes adjacent to the
        // object remain the real card bitmap and any leading bytes are pad.
        let logical_card_bytes = std::mem::size_of::<usize>() * extra_words;
        let card_header_bytes = logical_card_bytes.div_ceil(GcHeader::ALIGN) * GcHeader::ALIGN;
        let mut extra_flags = GcFlags::GCFLAG_HAS_CARDS | GcFlags::GCFLAG_TRACK_YOUNG_PTRS;
        // incminimark.py:1032-1035: "if 'alloc_young', then we also
        // immediately set GCFLAG_CARDS_SET, but without adding the object to
        // 'old_objects_with_cards_set'.  In this way it should never be added
        // to that list as long as it is young."
        // `visit_young_rawmalloced_object` is what adds it, at promotion, and
        // it asserts the flag is already there.
        if alloc_young {
            extra_flags |= GcFlags::GCFLAG_CARDS_SET;
        }
        (card_header_bytes, extra_flags)
    }

    /// `gc.py _setup_guard_is_object` parity. Computes
    /// `(_infobits_offset, _infobits_offset_plus, _T_IS_RPYTHON_INSTANCE_BYTE)`
    /// once at construction time and stores them as instance state.
    /// Mirrors line-by-line:
    ///
    /// ```python
    /// def _setup_guard_is_object(self):
    ///     from rpython.memory.gctypelayout import GCData, T_IS_RPYTHON_INSTANCE
    ///     import struct
    ///     infobits_offset, _ = symbolic.get_field_token(
    ///         GCData.TYPE_INFO, 'infobits', True)
    ///     mask = struct.pack("l", T_IS_RPYTHON_INSTANCE)
    ///     assert mask.count('\x00') == len(mask) - 1
    ///     infobits_offset_plus = 0
    ///     while mask.startswith('\x00'):
    ///         infobits_offset_plus += 1
    ///         mask = mask[1:]
    ///     self._infobits_offset = infobits_offset
    ///     self._infobits_offset_plus = infobits_offset_plus
    ///     self._T_IS_RPYTHON_INSTANCE_BYTE = ord(mask[0])
    /// ```
    fn _setup_guard_is_object(&mut self) {
        // `symbolic.get_field_token(TYPE_INFO, 'infobits', True)`.
        let infobits_offset = TypeInfoLayout::INFOBITS_OFFSET;
        // `mask = struct.pack("l", T_IS_RPYTHON_INSTANCE)`.
        let mask = TypeInfoLayout::T_IS_RPYTHON_INSTANCE.to_le_bytes();
        // `assert mask.count('\x00') == len(mask) - 1`.
        let nonzero = mask.iter().filter(|&&b| b != 0).count();
        assert_eq!(
            nonzero, 1,
            "T_IS_RPYTHON_INSTANCE must occupy exactly one byte of \
             the packed Signed word (gc.py:610)"
        );
        // `while mask.startswith('\x00'): infobits_offset_plus += 1; mask = mask[1:]`.
        let mut infobits_offset_plus = 0usize;
        while infobits_offset_plus < mask.len() && mask[infobits_offset_plus] == 0 {
            infobits_offset_plus += 1;
        }
        self._infobits_offset = infobits_offset;
        self._infobits_offset_plus = infobits_offset_plus;
        self._T_IS_RPYTHON_INSTANCE_BYTE = mask[infobits_offset_plus];
    }

    /// Register a type and return its ID.
    pub fn register_type(&mut self, info: TypeInfo) -> u32 {
        self.types.register(info)
    }

    /// Check if an address is in the nursery.
    #[inline]
    pub fn is_in_nursery(&self, addr: usize) -> bool {
        debug_assert!(
            !self.is_tagged_immediate(addr),
            "odd-valued (i.e. tagged) pointer unexpected here"
        );
        self.nursery.contains(addr)
    }

    /// The tagged-immediate test used by `is_valid_gc_object`
    /// (gc/base.py). With `taggedpointers` set (enablement only;
    /// default off per `translationoption.py:185`), a small `int` may be
    /// stored as an unboxed immediate with an odd low bit; such a value is
    /// not a heap object and must not be read as an object header.
    #[inline]
    fn is_tagged_immediate(&self, addr: usize) -> bool {
        self.config.taggedpointers && (addr & 1 == 1)
    }

    /// gc/base.py `is_valid_gc_object`.
    #[inline]
    fn is_valid_gc_object(&self, addr: usize) -> bool {
        addr != 0 && !self.is_tagged_immediate(addr)
    }

    /// incminimark.py range-check parity: nursery membership is a pure
    /// range check; JIT inline nursery bump-alloc must produce GcRefs
    /// indistinguishable from the slow path's, so a side table would
    /// not stay in sync.
    #[inline]
    pub fn is_managed_heap_object(&self, addr: usize) -> bool {
        self.is_valid_gc_object(addr) && (self.nursery.contains(addr) || self.oldgen.contains(addr))
    }

    /// Resolve a PyObject header without asking the nursery/oldgen allocators.
    /// A registered vtable plus the exact matching header tid is the translated
    /// type-layout witness. The tid equality rejects headerless bootstrap
    /// objects whose preceding allocator word happens to look like flags.
    ///
    /// This is the ONLY admissible probe for an address outside both managed
    /// generations, and its caller (`do_write_barrier`) must go through it.
    /// Recording every `alloc_with_gc_header` result in an
    /// ownership table instead would make the answer exact, but that is a probe
    /// on the allocation fast path — the same shape measured at −15% in the GC
    /// box-probe experiment — so the tid witness stands.
    #[inline]
    fn registered_external_header(&self, addr: usize) -> Option<*mut GcHeader> {
        if addr < GcHeader::SIZE || !addr.is_multiple_of(GcHeader::ALIGN) {
            return None;
        }
        let vtable = unsafe { *(addr as *const usize) };
        let expected_type_id = *self.vtable_to_type_id.get(&vtable)?;
        let hdr = unsafe { header_of(addr) };
        if unsafe { (*hdr).type_id() } == expected_type_id {
            Some(hdr)
        } else {
            None
        }
    }

    /// Valid-object nursery check for arbitrary GC fields/roots.
    ///
    /// A `GcRef` names the payload; the header sits immediately before it
    /// (`header.rs` / incminimark `size_gc_header`). `is_in_nursery` is a
    /// range check on the payload, so a word equal to `nursery_start` (or
    /// any host pointer that reused a retired nursery page) would still
    /// pass it — and `copy_nursery_object` would then write a forwarding
    /// header at `addr - SIZE`, which is *outside* the arena. RPython
    /// never sees this because `_trace_drag_out` receives a typed GC
    /// pointer; pyre's `GcRef` is a raw `usize`.
    #[inline]
    fn is_nursery_object_start(&self, addr: usize) -> bool {
        if !self.is_valid_gc_object(addr) || !self.is_in_nursery(addr) {
            return false;
        }
        // Payload invariant: the header sits at `addr - SIZE` and must
        // lie inside the arena. A GcRef equal to `nursery_start` would
        // write *before* the mapping (ExtraHeap mutex / host allocator).
        addr >= (self.nursery.start_ptr() as usize).saturating_add(GcHeader::SIZE)
    }

    /// Allocate a fixed-size object with the given type ID and size (excluding header).
    /// Returns a GcRef pointing to the object payload (after the header).
    ///
    /// Split into an inline bump and an out-of-line tail like
    /// [`alloc_with_type_rooted`](Self::alloc_with_type_rooted).
    #[inline]
    pub fn alloc_with_type(&mut self, type_id: u32, payload_size: usize) -> GcRef {
        // gc_stress: force a full collection *before* allocating so that any
        // object live at this allocation point but unreachable from a
        // registered root or custom-trace path is moved/swept deterministically
        // — turning a latent dangling pointer into an immediate failure. The
        // collection runs before reserving the new object so the returned
        // pointer is never itself moved. Gated on both the `gc_stress` feature
        // (so default/release builds compile no branch) and the per-instance
        // `stress_collect` flag (so an opted-in GC stresses only the allocations
        // it owns, leaving unrelated suites — and their `minor_collections == 0`
        // assertions — untouched).
        #[cfg(feature = "gc_stress")]
        if self.stress_collect {
            self.do_collect_full();
        }

        let Some(total_size) = GcHeader::SIZE.checked_add(payload_size) else {
            return GcRef(0);
        };

        // incminimark.py `malloc_fixedsize`: a heavyweight old-style
        // finalizer can allocate, collect and resurrect, so its object must
        // never move. `finish_alloc_in_oldgen` performs the -1 registration.
        if self.type_has_old_style_finalizer(type_id) {
            return self.alloc_in_oldgen_clear(type_id, total_size);
        }

        if total_size < self.config.large_object_threshold {
            let ptr = self.nursery.alloc(total_size);
            if !ptr.is_null() {
                return self.finish_nursery_object(ptr, type_id);
            }
        }
        self.alloc_with_type_slow(type_id, total_size)
    }

    /// The outcomes [`alloc_with_type`](Self::alloc_with_type) leaves out of
    /// line: a large object, and a nursery too full to serve the bump.
    #[cold]
    #[inline(never)]
    fn alloc_with_type_slow(&mut self, type_id: u32, total_size: usize) -> GcRef {
        // incminimark.py `malloc_varsize` and :665 `malloc_fixedsize`:
        // an oversized object skips the nursery through
        // `external_malloc(..., alloc_young=True)`, so it is non-moving and
        // still YOUNG — the next minor frees it if nothing reached it. This
        // entry point already collects on its nursery arm below, so its callers
        // already present a complete root set at an allocation; the young birth
        // adds no requirement they do not already meet.
        if total_size >= self.config.large_object_threshold {
            if self.do_maybe_collect_for_external_malloc(total_size) {
                return GcRef(0);
            }
            // `external_malloc` (`incminimark.py`): `arena_malloc` returning
            // NULL is `MemoryError`, not a process abort. `rbigint.lshift`
            // of `1 << 10**18` is the public edge that depends on it. A young
            // birth that fails is that NULL; it does not fall back to an old
            // one, which callers that elide the barrier on a fresh result
            // (`remember_write_barrier`) could not see.
            if self.young_birth_allowed(type_id) {
                return self
                    .try_alloc_young_nonmoving_clear(type_id, total_size)
                    .unwrap_or(GcRef(0));
            }
            if let Some(obj) = self.try_alloc_in_oldgen(type_id, total_size) {
                Self::raw_memclear(obj, total_size);
                return obj;
            }
            return GcRef(0);
        }

        // `IncrementalMiniMarkGC.collect_and_reserve`: a failed bump may have
        // stopped at a pinned object rather than at the nursery end.  Consume
        // the already-built barrier deque before deciding to collect.
        let ptr = self.reserve_nursery_gap(total_size);
        if !ptr.is_null() {
            self.apply_debug_tiny_nursery();
            return self.finish_nursery_object(ptr, type_id);
        }

        // Nursery full: trigger minor collection and retry. Carry the triggering
        // allocation size so a bounded major collection applies the
        // PYPY_GC_MAX out-of-memory policy against it (threshold_reached).
        self.pending_reserving_size = total_size;
        self.do_collect_nursery();
        self.pending_reserving_size = 0;
        // incminimark.py:2603-2615 — a bounded major collection that leaves
        // the heap over `max_heap_size` asks this allocation to fail so the
        // caller raises MemoryError: NULL propagates to the compiled-code
        // `CHECK_MEMORY_ERROR` path. Never taken in the unbounded default
        // (`PYPY_GC_MAX` unset), so the fallback below is unchanged there.
        if std::mem::take(&mut self.oom_pending) {
            return GcRef(0);
        }
        let ptr = self.nursery.alloc(total_size);
        let mut ptr = if ptr.is_null() {
            self.reserve_nursery_gap(total_size)
        } else {
            ptr
        };
        if ptr.is_null() && Self::nursery_allocation_size(total_size) > self.nursery.size() {
            // Production incminimark keeps `nonlarge_max` below the
            // nursery size. Tiny backend tests can configure the two
            // independently; retain their external-allocation fallback
            // only for an object that cannot physically fit anywhere, born
            // young as the large arm above births it.
            if self.young_birth_allowed(type_id) {
                return self
                    .try_alloc_young_nonmoving_clear(type_id, total_size)
                    .unwrap_or(GcRef(0));
            }
            // Otherwise it is born old, and a `CALL_MALLOC_NURSERY` caller
            // then stamps the whole `HDR.tid` word over the old-generation
            // flags (see `make_tid_field_descr`).
            return self.alloc_in_oldgen_clear(type_id, total_size);
        }
        if ptr.is_null() {
            // `IncrementalMiniMarkGC.collect_and_reserve`: finalizer or
            // rawrefcount work run by the wrapper may have filled the nursery
            // again.  The second attempt is exactly `_minor_collection()` —
            // no major step and no second callback — so the retry sees an
            // empty nursery apart from pinned barriers.
            self.minor_collection_only();
            ptr = self.nursery.alloc(total_size);
            if ptr.is_null() {
                ptr = self.reserve_nursery_gap(total_size);
            }
        }
        assert!(
            !ptr.is_null(),
            "collect_and_reserve could not find nursery space for a non-large object"
        );
        self.apply_debug_tiny_nursery();
        self.finish_nursery_object(ptr, type_id)
    }

    /// `malloc_fixedsize` / `collect_and_reserve` with one native-stack root.
    ///
    /// RPython's GC transform puts live GC locals in a shadow-stack/stack-map,
    /// but the allocation fast path is still a plain nursery bump. Rust locals
    /// are not scanned, so callers pass the exceptional local slot explicitly.
    /// Register it only when the bump fails and a collection is actually
    /// required; this preserves the upstream fast path instead of doing a
    /// root-set add/remove for every allocation.
    ///
    /// Split into a `malloc_fast` body and its exceptional tail the way
    /// `framework.py:361-382` does: the copy it declares `inline=True` carries
    /// only the bump, so a caller pays a bump and a branch, while a large
    /// object or a full nursery costs a call. Without the split the whole
    /// function is a frame that saves ten callee-saved registers before it
    /// can reach a ten-instruction bump.
    ///
    /// # Safety
    /// `root` must point to a valid mutable `GcRef` slot until this call
    /// returns. `needs_write_barrier` must point to a valid mutable `bool`
    /// slot.
    #[inline]
    pub unsafe fn alloc_with_type_rooted(
        &mut self,
        type_id: u32,
        payload_size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.alloc_with_type_rooted_body::<false>(
                type_id,
                payload_size,
                root,
                1,
                needs_write_barrier,
            )
        }
    }

    /// [`alloc_with_type_rooted`](Self::alloc_with_type_rooted) for a type that
    /// carries neither a finalizer nor the weakref flag: `malloc_fast`
    /// (`framework.py`), the copy `gct_fv_gc_malloc` selects at
    /// `framework.py:830-838` for exactly that case.
    ///
    /// # Safety
    /// Same contract as [`alloc_with_type_rooted`](Self::alloc_with_type_rooted),
    /// plus `type_id` must name a type with no destructor and no weakref flag.
    #[inline]
    pub unsafe fn alloc_fast_with_type_rooted(
        &mut self,
        type_id: u32,
        payload_size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.alloc_with_type_rooted_body::<true>(
                type_id,
                payload_size,
                root,
                1,
                needs_write_barrier,
            )
        }
    }

    /// `malloc_fast` with the full `gc_push_roots(livevars)` span.  The
    /// common bump path touches none of the slots; only the collection tail
    /// registers and later reloads them, as `postprocess_inlining` arranges for
    /// RPython's expanded shadow-stack operations.
    ///
    /// # Safety
    /// `roots` must address `root_count` contiguous mutable [`GcRef`] slots
    /// until this call returns.
    #[inline]
    pub unsafe fn alloc_fast_with_type_roots(
        &mut self,
        type_id: u32,
        payload_size: usize,
        roots: *mut GcRef,
        root_count: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.alloc_with_type_rooted_body::<true>(
                type_id,
                payload_size,
                roots,
                root_count,
                needs_write_barrier,
            )
        }
    }

    /// # Safety
    /// See [`alloc_with_type_rooted`](Self::alloc_with_type_rooted); `FAST`
    /// additionally carries `malloc_fast`'s obligation on `type_id`.
    #[inline]
    unsafe fn alloc_with_type_rooted_body<const FAST: bool>(
        &mut self,
        type_id: u32,
        payload_size: usize,
        roots: *mut GcRef,
        root_count: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe { *needs_write_barrier = false };

        #[cfg(feature = "gc_stress")]
        if self.stress_collect {
            for i in 0..root_count {
                unsafe { self.roots.add(roots.add(i)) };
            }
            self.do_collect_full();
            for i in (0..root_count).rev() {
                self.roots.remove(unsafe { roots.add(i) });
            }
        }

        let Some(total_size) = GcHeader::SIZE.checked_add(payload_size) else {
            return GcRef(0);
        };

        if !FAST && self.type_has_old_style_finalizer(type_id) {
            unsafe { *needs_write_barrier = true };
            return self.alloc_in_oldgen_clear(type_id, total_size);
        }

        if total_size < self.config.large_object_threshold {
            let ptr = self.nursery.alloc(total_size);
            if !ptr.is_null() {
                return self.finish_bumped_nursery_object::<FAST>(ptr, type_id);
            }
        }
        unsafe {
            self.alloc_with_type_rooted_slow(
                type_id,
                total_size,
                roots,
                root_count,
                needs_write_barrier,
            )
        }
    }

    /// The outcomes [`alloc_with_type_rooted`] leaves out of line: a large
    /// object, and a nursery too full to serve the bump.
    ///
    /// # Safety
    /// Same contract as [`alloc_with_type_rooted`].
    #[cold]
    #[inline(never)]
    unsafe fn alloc_with_type_rooted_slow(
        &mut self,
        type_id: u32,
        total_size: usize,
        roots: *mut GcRef,
        root_count: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        // `needs_write_barrier` stays true for the young birth as well. The
        // young object carries no GCFLAG_TRACK_YOUNG_PTRS, so the barrier the
        // caller then emits finds the flag clear and does nothing — a cost, not
        // a hazard, and the alternative is a caller that has to know which of
        // the two births it got.
        //
        // `external_malloc` may run a moving minor before the new block
        // exists. The caller's live slot is not yet a nursery object, but it
        // may be the only reference to one; register it across that collection
        // the same way the nursery-full arm does.
        if total_size >= self.config.large_object_threshold {
            unsafe { *needs_write_barrier = true };
            for i in 0..root_count {
                unsafe { self.roots.add(roots.add(i)) };
            }
            let oom = self.do_maybe_collect_for_external_malloc(total_size);
            for i in (0..root_count).rev() {
                self.roots.remove(unsafe { roots.add(i) });
            }
            if oom {
                return GcRef(0);
            }
            // `external_malloc` (`incminimark.py`): `arena_malloc` returning
            // NULL is `MemoryError`, not a process abort. `rbigint.lshift`
            // of `1 << 10**18` is the public edge that depends on it. A young
            // birth that fails is that NULL, as in `alloc_with_type_slow`.
            if self.young_birth_allowed(type_id) {
                return self
                    .try_alloc_young_nonmoving_clear(type_id, total_size)
                    .unwrap_or(GcRef(0));
            }
            if let Some(obj) = self.try_alloc_in_oldgen(type_id, total_size) {
                Self::raw_memclear(obj, total_size);
                return obj;
            }
            return GcRef(0);
        }

        // `IncrementalMiniMarkGC.collect_and_reserve`: crossing a pinned
        // barrier does not collect, so the exceptional native-stack root does
        // not need registering for this branch.
        let ptr = self.reserve_nursery_gap(total_size);
        if !ptr.is_null() {
            self.apply_debug_tiny_nursery();
            return self.finish_nursery_object(ptr, type_id);
        }

        self.pending_reserving_size = total_size;
        for i in 0..root_count {
            unsafe { self.roots.add(roots.add(i)) };
        }
        self.do_collect_nursery();
        self.pending_reserving_size = 0;
        if std::mem::take(&mut self.oom_pending) {
            for i in (0..root_count).rev() {
                self.roots.remove(unsafe { roots.add(i) });
            }
            return GcRef(0);
        }
        let ptr = self.nursery.alloc(total_size);
        let mut ptr = if ptr.is_null() {
            self.reserve_nursery_gap(total_size)
        } else {
            ptr
        };
        if ptr.is_null() && Self::nursery_allocation_size(total_size) > self.nursery.size() {
            for i in (0..root_count).rev() {
                self.roots.remove(unsafe { roots.add(i) });
            }
            unsafe { *needs_write_barrier = true };
            return self.alloc_in_oldgen_clear(type_id, total_size);
        }
        if ptr.is_null() {
            // Keep the exceptional native-stack slot registered across the
            // second bare minor as well; it is precisely another moving
            // collection, even though no major/callback tail follows it.
            self.minor_collection_only();
            ptr = self.nursery.alloc(total_size);
            if ptr.is_null() {
                ptr = self.reserve_nursery_gap(total_size);
            }
        }
        for i in (0..root_count).rev() {
            self.roots.remove(unsafe { roots.add(i) });
        }
        assert!(
            !ptr.is_null(),
            "collect_and_reserve could not find nursery space for a non-large object"
        );
        self.apply_debug_tiny_nursery();
        self.finish_nursery_object(ptr, type_id)
    }

    /// incminimark.py, the tail of `collect_and_reserve`:
    ///
    /// ```python
    /// if self.debug_tiny_nursery >= 0:   # for debugging
    ///     if self.nursery_top - self.nursery_free > self.debug_tiny_nursery:
    ///         self.nursery_free = self.nursery_top - self.debug_tiny_nursery
    /// ```
    ///
    /// Runs after the reservation, so the object just handed out kept the
    /// whole arena and only what follows it is rationed.
    #[inline]
    fn apply_debug_tiny_nursery(&mut self) {
        let Some(budget) = self.config.debug_tiny_nursery else {
            return;
        };
        let free = self.nursery.free_ptr() as usize;
        let top = self.nursery.top_ptr() as usize;
        if top.saturating_sub(free) > budget {
            // SAFETY: `top - budget` is above `free`, which the reservation
            // just established is inside the nursery.
            unsafe { self.nursery.set_free_ptr((top - budget) as *mut u8) };
        }
    }

    /// The tail a successful nursery bump runs: `init_gc_object`, then
    /// incminimark.py's two registrations.
    ///
    /// `FAST` is `malloc_fast` (`framework.py:361-382`), the copy of
    /// `malloc_fixedsize` annotated `s_False, s_False, s_False` — under that
    /// annotation both `if`s constant-fold away, so the common allocation never
    /// reaches the type's flags at all. `gct_fv_gc_malloc`
    /// (`framework.py:830-838`) selects the copy only for a type with no
    /// finalizer, and never passes `contains_weakptr=True` for a fixed-size
    /// malloc: a WEAKREF is built by `gct_weakref_create` instead. The
    /// obligation rides with the caller here too, so the `ll_assert` the
    /// general body spells out becomes a debug assertion.
    #[inline]
    fn finish_bumped_nursery_object<const FAST: bool>(
        &mut self,
        ptr: *mut u8,
        type_id: u32,
    ) -> GcRef {
        if FAST {
            debug_assert!(
                !self.type_needs_young_registration(type_id),
                "malloc_fast served a type that needs a young weakref/destructor registration"
            );
            Self::init_nursery_object(ptr, type_id);
            return GcRef((ptr as usize) + GcHeader::SIZE);
        }
        self.finish_nursery_object(ptr, type_id)
    }

    /// The `malloc_fast` precondition, as the debug build checks it.
    fn type_needs_young_registration(&self, type_id: u32) -> bool {
        if (type_id as usize) >= self.types.len() {
            return false;
        }
        let info = self.types.get(type_id);
        info.is_weakref || info.destructor.is_some()
    }

    fn type_is_weakref(&self, type_id: u32) -> bool {
        (type_id as usize) < self.types.len() && self.types.get(type_id).is_weakref
    }

    /// The tail every non-`malloc_fast` nursery bump shares: `init_gc_object`
    /// plus the weakref and destructor registration of incminimark.py.
    #[inline]
    fn finish_nursery_object(&mut self, ptr: *mut u8, type_id: u32) -> GcRef {
        Self::init_nursery_object(ptr, type_id);
        let obj = GcRef((ptr as usize) + GcHeader::SIZE);
        self.register_young_object_if_needed(type_id, obj.0);
        obj
    }

    /// incminimark.py:687-693's pair of `if`s, reached the way upstream reaches
    /// them: with both flags already in hand.
    ///
    /// A WEAKREF (`T_IS_WEAKREF` in its TYPE_INFO) joins the young-weakref list
    /// so the next minor collection can invalidate the single `weakptr` slot at
    /// `weakref::WEAKPTR_OFFSET` inside the payload (gctypelayout.py) if its
    /// target dies. A type carrying a lightweight `TypeInfo.destructor` joins
    /// the young-destructor list, so that collection either runs the destructor
    /// or promotes the entry to `old_objects_with_destructors`.
    /// `obj_addr` is the payload base (post-header) in both cases.
    ///
    /// `malloc_fixedsize` takes `needs_finalizer` and `contains_weakptr` as
    /// arguments, so its tail costs two tests on values the caller supplied —
    /// and in the `malloc_fast` copy (`framework.py:371-382`, annotated
    /// `s_False, s_False, s_False`) both fold away entirely. pyre resolves them
    /// from the type table instead, which is one row lookup, not two: probing
    /// separately repeats the bound test, the table base load and the stride
    /// multiply that the first probe already performed.
    #[inline]
    fn register_young_object_if_needed(&mut self, type_id: u32, obj_addr: usize) {
        if (type_id as usize) >= self.types.len() {
            return;
        }
        let info = self.types.get(type_id);
        let (is_weakref, has_destructor) = (info.is_weakref, info.destructor.is_some());
        if is_weakref {
            self.young_objects_with_weakrefs.push(obj_addr);
        }
        if has_destructor {
            self.young_objects_with_destructors.push(obj_addr);
        }
    }

    /// incminimark.py `collect_and_reserve` pinned-barrier walk.
    ///
    /// A minor collection can leave `nursery_free` after the highest pinned
    /// object, where the tail is too short for the triggering allocation even
    /// though an earlier gap is large enough. Upstream walks the ordered
    /// `nursery_barriers` and reserves from the first fitting gap; it never
    /// changes a non-large malloc into an old-generation allocation. Preserve
    /// that young-result contract because the GC rewrite elides write barriers
    /// while initializing a fresh nursery object (`rewrite.py:911`).
    fn nursery_allocation_size(total_size: usize) -> usize {
        let align = crate::header::MEMORY_ALIGNMENT;
        total_size
            .max(GcHeader::MIN_NURSERY_OBJ_SIZE)
            .checked_add(align - 1)
            .map(|size| size & !(align - 1))
            .unwrap_or(usize::MAX)
    }

    /// `incminimark.py external_malloc` parity: both the variable
    /// portion and the final payload sum are overflow-checked.  The caller
    /// turns `None` into NULL so compiled code raises `MemoryError`.
    fn checked_varsize_payload_size(
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> Option<usize> {
        item_size
            .checked_mul(length)
            .and_then(|var_size| base_size.checked_add(var_size))
    }

    fn reserve_nursery_gap(&mut self, total_size: usize) -> *mut u8 {
        // `IncrementalMiniMarkGC.collect_and_reserve`: `nursery_top` is the
        // pinned header immediately in front of the failed bump, and the deque
        // contains the following pinned headers plus the nursery end.  Jump
        // over one object at a time and try the next gap.  There is deliberately
        // no pin lookup or barrier reconstruction on this path.
        while let Some(next_top) = self.nursery_barriers.pop_front() {
            let pinned_header = self.nursery.top_ptr() as usize;
            let pinned_obj = pinned_header + GcHeader::SIZE;
            debug_assert!(self.nursery.contains(pinned_obj));
            // The object may have been unpinned since the minor built this
            // deque. Upstream still consumes the stale barrier until the next
            // minor; skipping its bytes is harmless and avoids rebuilding the
            // deque on `unpin`.
            let type_id = unsafe { (*header_of(pinned_obj)).type_id() };
            let payload_size = self.size_for_typeid(pinned_obj, type_id, "pinned_barriers");
            let object_size = Self::nursery_allocation_size(GcHeader::SIZE + payload_size);
            // `size_for_typeid` decodes the pinned object's extent from its
            // header. A decode that overstates it would push free past the
            // barrier we are about to publish, and `Nursery::alloc` would then
            // hand out bytes beyond the gap. The barrier is the hard bound.
            let next_free = (pinned_header + object_size).min(next_top);
            unsafe {
                // Set the wider bound first so Nursery's pointer invariant is
                // maintained while free crosses the old (pinned) top.
                self.nursery.set_top_ptr(next_top as *const u8);
                self.nursery.set_free_ptr(next_free as *mut u8);
            }
            self.refresh_published_nursery_top();
            let ptr = self.nursery.alloc(total_size);
            if !ptr.is_null() {
                return ptr;
            }
        }
        std::ptr::null_mut()
    }

    /// Allocate without triggering collection.
    ///
    /// If the nursery cannot satisfy the request, this falls back directly to
    /// old-gen allocation so compiled code can keep running without needing
    /// stack-map-mediated collection.
    pub fn alloc_with_type_no_collect(&mut self, type_id: u32, payload_size: usize) -> GcRef {
        let Some(total_size) = GcHeader::SIZE.checked_add(payload_size) else {
            return GcRef(0);
        };

        if self.type_has_old_style_finalizer(type_id) {
            return self.alloc_in_oldgen_nursery_substitute(type_id, total_size);
        }

        // Both fallbacks stand in for the nursery bump below, which clears
        // nothing — the same reading `spill_to_oldgen_or_null` takes for this
        // function's fallible counterpart.
        //
        // The oversized arm is born OLD here, unlike `alloc_with_type_slow`'s
        // and `alloc_nursery_collecting_typed_rooted`'s, which take
        // `try_alloc_young_nonmoving_clear` — `external_malloc(...,
        // alloc_young=True)`, incminimark.py. The young birth does not
        // collect, so the no-collect contract is not what withholds it.
        //
        // What withholds it is the root set. Born old, an unrooted block is
        // reclaimed by the next major; born young, by the next MINOR. Neither
        // placement makes an unrooted block safe — the major sweeps it just as
        // the minor frees it — so this is a difference in how long a rooting
        // gap stays latent, and the callers here are compiled code whose
        // stack maps this branch exists to avoid consulting. Taking the young
        // arm turns every such gap into an immediate use-after-free, which is
        // why it waits on the shadow-stack coverage rather than on anything
        // in this function.
        if total_size >= self.config.large_object_threshold {
            return self.alloc_in_oldgen_nursery_substitute(type_id, total_size);
        }

        let ptr = self.nursery.alloc(total_size);
        if ptr.is_null() {
            return self.alloc_in_oldgen_nursery_substitute(type_id, total_size);
        }

        self.finish_nursery_object(ptr, type_id)
    }

    /// Fallible host-side counterpart of `alloc_with_type_no_collect`.
    ///
    /// The ordinary nursery bump remains allocation-free and therefore cannot
    /// fail. Oversized objects and nursery-full spill use rawmalloc and return
    /// NULL on failure, matching RPython's `MemoryError` path.
    pub fn try_alloc_with_type_no_collect(&mut self, type_id: u32, payload_size: usize) -> GcRef {
        let mut needs_write_barrier = true;
        unsafe {
            self.try_alloc_with_type_no_collect_with_placement(
                type_id,
                payload_size,
                &mut needs_write_barrier,
            )
        }
    }

    /// [`try_alloc_with_type_no_collect`](Self::try_alloc_with_type_no_collect)
    /// for a type that carries neither a finalizer nor the weakref flag:
    /// `malloc_fast` (`framework.py:361-382`).
    ///
    /// # Safety
    /// `type_id` must name a type with no destructor and no weakref flag.
    pub unsafe fn try_alloc_fast_with_type_no_collect(
        &mut self,
        type_id: u32,
        payload_size: usize,
    ) -> GcRef {
        let mut needs_write_barrier = true;
        unsafe {
            self.try_alloc_fast_with_type_no_collect_with_placement(
                type_id,
                payload_size,
                &mut needs_write_barrier,
            )
        }
    }

    /// Placement-reporting counterpart of
    /// [`try_alloc_with_type_no_collect`](Self::try_alloc_with_type_no_collect).
    ///
    /// `framework.py:28-61` propagates `no_write_barrier_needed` through a
    /// fresh nursery allocation. The no-collect path can instead spill to
    /// old-gen, so report that exceptional placement to the initializer.
    ///
    /// Split like [`alloc_with_type_rooted`](Self::alloc_with_type_rooted):
    /// the bump inlines into the caller, the old-gen spill is a call.
    ///
    /// # Safety
    /// `needs_write_barrier` must point to a valid mutable `bool` slot for the
    /// duration of this call.
    #[inline]
    pub unsafe fn try_alloc_with_type_no_collect_with_placement(
        &mut self,
        type_id: u32,
        payload_size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.try_alloc_with_type_no_collect_body::<false>(
                type_id,
                payload_size,
                needs_write_barrier,
            )
        }
    }

    /// [`try_alloc_with_type_no_collect_with_placement`](Self::try_alloc_with_type_no_collect_with_placement)
    /// for a type that carries neither a finalizer nor the weakref flag:
    /// `malloc_fast` (`framework.py:361-382`).
    ///
    /// # Safety
    /// Same contract as
    /// [`try_alloc_with_type_no_collect_with_placement`](Self::try_alloc_with_type_no_collect_with_placement),
    /// plus `type_id` must name a type with no destructor and no weakref flag.
    #[inline]
    pub unsafe fn try_alloc_fast_with_type_no_collect_with_placement(
        &mut self,
        type_id: u32,
        payload_size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.try_alloc_with_type_no_collect_body::<true>(
                type_id,
                payload_size,
                needs_write_barrier,
            )
        }
    }

    /// # Safety
    /// See
    /// [`try_alloc_with_type_no_collect_with_placement`](Self::try_alloc_with_type_no_collect_with_placement);
    /// `FAST` additionally carries `malloc_fast`'s obligation on `type_id`.
    #[inline]
    unsafe fn try_alloc_with_type_no_collect_body<const FAST: bool>(
        &mut self,
        type_id: u32,
        payload_size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe { *needs_write_barrier = true };
        let Some(total_size) = GcHeader::SIZE.checked_add(payload_size) else {
            return GcRef(0);
        };

        if !FAST && self.type_has_old_style_finalizer(type_id) {
            return self.spill_to_oldgen_or_null(type_id, total_size);
        }

        if total_size < self.config.large_object_threshold {
            let ptr = self.nursery.alloc(total_size);
            if !ptr.is_null() {
                let obj = self.finish_bumped_nursery_object::<FAST>(ptr, type_id);
                unsafe { *needs_write_barrier = false };
                return obj;
            }
        }
        // incminimark.py: a weakref is immutable and is born in the nursery,
        // so an old weakref never points at a young object. Spilling one into
        // old-gen while its target is still young leaves `weakptr` unfixed
        // across the next minor (`invalidate_young_weakrefs` only walks
        // nursery weakrefs). Refuse the spill; the caller collects and
        // retries, which promotes the rooted target first.
        if !FAST && self.type_is_weakref(type_id) {
            return GcRef(0);
        }
        self.spill_to_oldgen_or_null(type_id, total_size)
    }

    /// The old-gen spill
    /// [`try_alloc_with_type_no_collect_with_placement`](Self::try_alloc_with_type_no_collect_with_placement)
    /// leaves out of line, for a large object or a nursery with no room.
    ///
    /// Old, not young: [`Self::alloc_with_type_no_collect`] records why the
    /// no-collect paths withhold `external_malloc`'s `alloc_young` arm.
    ///
    /// A `ItemsBlock` past `large_object_threshold` — a list of roughly
    /// 16,900 items or more — reaches the old generation through here, and so
    /// stays resident until a major even when the list dies in the loop
    /// iteration that built it.
    #[cold]
    #[inline(never)]
    fn spill_to_oldgen_or_null(&mut self, type_id: u32, total_size: usize) -> GcRef {
        self.try_alloc_in_oldgen_nursery_substitute(type_id, total_size)
            .unwrap_or(GcRef(0))
    }

    /// Run the lightweight destructor registered for `obj_addr`'s type,
    /// if any. incminimark.py `call_destructor` analog: looks up the
    /// type's `TypeInfo.destructor` and calls it on the payload base.
    ///
    /// # Safety
    /// `obj_addr` must point at a live (not-yet-freed) object payload
    /// whose header still names a registered type id — true for a dead
    /// nursery object before `nursery.reset()` and a dying old-gen object
    /// before the first incremental old-gen sweep step.
    fn run_destructor(&self, obj_addr: usize) {
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        if (type_id as usize) >= self.types.len() {
            return;
        }
        if let Some(destructor) = self.types.get(type_id).destructor {
            unsafe { destructor(obj_addr) };
        }
    }

    /// `q_is_old_style_finalizer(typeid)` from gctypelayout.py.
    #[inline]
    fn type_has_old_style_finalizer(&self, type_id: u32) -> bool {
        (type_id as usize) < self.types.len()
            && self.types.get(type_id).old_style_finalizer.is_some()
    }

    /// `GCBase.mark_finalizer_to_run` keeps the historical -1 queue index for
    /// old-style finalizers and non-negative indexes for FinalizerQueue.
    fn register_finalizer_index(&mut self, fq_index: isize, obj: GcRef) {
        if obj.is_null() || !self.is_managed_heap_object(obj.0) {
            return;
        }
        // `register_finalizer` is contracted to run at most once per object
        // (`rgc.py`). A second deque entry survives the first
        // `deal_with_objects_with_finalizers` pass through `new_with_finalizer`
        // (incminimark.py) and delivers the object a second time on
        // the next major collection, so honour the contract here rather than
        // leaving it to every caller.
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { (*hdr).has_flag(GcFlags::FINALIZER_REGISTERED) } {
            return;
        }
        unsafe { (*hdr).set_flag(GcFlags::FINALIZER_REGISTERED) };
        // `oldgen.contains` is not the "is it old" question on its own: both
        // raw-malloc births share `try_rawmalloc_block`, so a young one is in
        // `rawmalloced_payloads` as well and answers yes. It has to be excluded
        // here, because filing it as old skips the GCFLAG_VISITED_RMY stamp
        // `deal_with_young_objects_with_finalizers` owes a raw-malloced member,
        // and `free_young_rawmalloced_objects` then releases the block at the
        // end of that minor with its address still on the old deque.
        if self.oldgen.contains(obj.0) && !self.is_young_rawmalloced(obj.0) {
            // Pyre's host allocations can be born directly in old-gen and an
            // explicit non-moving major intentionally skips the leading minor.
            // This is the post-`_trace_drag_out1` destination of PyPy's
            // probably-young deque, reached immediately for an already-old obj.
            self.old_objects_with_finalizers
                .push_back((obj.0, fq_index));
        } else {
            self.probably_young_objects_with_finalizers
                .push_back((obj.0, fq_index));
        }
    }

    /// Initialize a nursery object's header.
    fn init_nursery_object(header_ptr: *mut u8, type_id: u32) {
        let hdr = unsafe { &mut *(header_ptr as *mut GcHeader) };
        // Young objects do NOT have TRACK_YOUNG_PTRS (we assume any
        // young object can point to any other young object).
        *hdr = GcHeader::new(type_id);
    }

    /// Flags for an object born directly into the old generation, adding
    /// `GcFlags::GCFLAG_VISITED` while a major marking cycle is in progress.
    ///
    /// An object allocated straight into the old generation during MARKING (a
    /// large object or nursery-full fallback) enters the still-active page/raw
    /// lists that will be frozen at the MARKING->SWEEPING seam. The root scan has
    /// already happened, so allocate it black; allocations during SWEEPING need
    /// no VISITED workaround because incminimark.py and :2688-2694's
    /// list swaps keep them outside this cycle's candidates. Any young pointer
    /// later stored into a born-black object re-enters marking through the write
    /// barrier's remembered-set rescan.
    #[inline]
    fn oldgen_birth_flags(&self, base: GcFlags) -> GcFlags {
        if self.gc_state == GcState::Marking {
            base | GcFlags::GCFLAG_VISITED
        } else {
            base
        }
    }

    /// Walk everything reachable from the roots at the end of a minor and
    /// report each traced slot that still names an unpinned nursery address.
    ///
    /// Reachability is what makes the answer usable: a block the mutator
    /// abandoned — a grown-out-of mapdict storage, a replaced items block —
    /// keeps whatever it last held until the sweep frees it, and those stale
    /// words are not violations. Only declared slots are read, so a word of
    /// padding or a non-reference field that happens to fall inside the
    /// nursery range cannot be mistaken for one either.
    ///
    /// Returns `(holder, slot offset, value, parent)` per bad slot, where the
    /// parent is the object the holder was first reached through: a block with
    /// no header of its own is traced only via its owner's custom trace, so
    /// the owner is what the remembered set has to carry.
    fn bh_probe_stale_young_slots(&self) -> Vec<(usize, usize, usize, Option<usize>)> {
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut stale: Vec<(usize, usize, usize, Option<usize>)> = Vec::new();
        let mut pending: Vec<(GcRef, Option<usize>)> = self
            .enumerate_all_root_values()
            .into_iter()
            .map(|gcref| (gcref, None))
            .collect();
        while let Some((gcref, parent)) = pending.pop() {
            if gcref.is_null()
                || !self.is_managed_heap_object(gcref.0)
                || self.nursery.contains(gcref.0)
                // A young raw-malloced holder owes no barrier for the same
                // reason a nursery holder does not: the minor reaches it as an
                // object and forwards its slots itself. Without this it reads
                // as an old holder and every young reference it carries is
                // reported as a missing barrier.
                || self.is_young_rawmalloced(gcref.0)
                || !seen.insert(gcref.0)
            {
                continue;
            }
            let here = gcref.0;
            self.visit_referent_slots(here, &mut |slot| {
                let value = unsafe { *slot }.0;
                if value == 0 {
                    return;
                }
                if self.nursery.contains(value) && !self.is_pinned(GcRef(value)) {
                    stale.push((here, slot as usize - here, value, parent));
                    return;
                }
                pending.push((GcRef(value), Some(here)));
            });
        }
        stale
    }

    /// TEMPORARY DIAGNOSTIC (`MAJIT_GC_BH_PROBE`), run before a minor starts.
    ///
    /// A minor traces the roots and the remembered set and nothing else, so an
    /// old-generation object holding a young reference while off that set has
    /// its slot left behind: the value moves and the slot keeps the dead
    /// address. That is a missing write barrier, and the moment before the
    /// collection is the only one at which the evidence still exists — after
    /// it the slot is indistinguishable from one that was never written.
    ///
    /// Panics naming the holder, the slot, and the object the holder was first
    /// reached through, since a block with no header of its own is traced only
    /// through its owner and it is the owner that has to carry the barrier.
    fn bh_probe_check_barriers_before_minor(&mut self) {
        if !crate::bh_probe_enabled()
            || self.minor_collections < read_uint_from_env("MAJIT_GC_BH_PROBE_FROM").unwrap_or(0)
        {
            return;
        }
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut pending: Vec<(GcRef, Option<usize>)> = self
            .enumerate_all_root_values()
            .into_iter()
            .map(|gcref| (gcref, None))
            .collect();
        let mut bad: Vec<String> = Vec::new();
        while let Some((gcref, parent)) = pending.pop() {
            if gcref.is_null() || !self.is_managed_heap_object(gcref.0) || !seen.insert(gcref.0) {
                continue;
            }
            let here = gcref.0;
            // A young holder needs no barrier: the minor reaches it as an
            // object and forwards its slots itself, so only an old-generation
            // slot can be left behind. A young raw-malloced holder is in
            // `rawmalloced_payloads` like an old one, so it has to be asked
            // about separately — the same reading `bh_probe_stale_young_slots`
            // takes.
            let holder_is_old = self.oldgen.contains(here) && !self.is_young_rawmalloced(here);
            let remembered = self.old_objects_pointing_to_young.contains(&here);
            let parent_remembered =
                parent.is_some_and(|p| self.old_objects_pointing_to_young.contains(&p));
            self.visit_referent_slots(here, &mut |slot| {
                let value = unsafe { *slot }.0;
                if value == 0 {
                    return;
                }
                // Two ways a traced slot can already be wrong here: it names a
                // managed address whose header is not a type at all (the value
                // died and its memory was handed out again), or it names a
                // young object from an old holder the barrier never remembered.
                let managed = self.nursery.contains(value) || self.oldgen.contains(value);
                let bad_target = managed
                    && (unsafe { (*header_of(value)).type_id() } as usize) >= self.types.len();
                let unbarriered = holder_is_old
                    && !remembered
                    && !parent_remembered
                    && self.nursery.contains(value)
                    && !self.is_pinned(GcRef(value));
                if bad_target || unbarriered {
                    let tid = unsafe { (*header_of(here)).type_id() };
                    let parent_tid = parent.map(|p| unsafe { (*header_of(p)).type_id() });
                    bad.push(format!(
                        "{} holder={here:#x} tid={tid} old={holder_is_old} slot={:#x} \
                         offset={} value={value:#x} young={} track_young={} remembered={} \
                         barriered_ever={} | parent={:#x} tid={parent_tid:?} remembered={} \
                         barriered_ever={}",
                        if bad_target {
                            "BAD-TARGET"
                        } else {
                            "UNBARRIERED"
                        },
                        slot as usize,
                        slot as usize - here,
                        self.nursery.contains(value),
                        unsafe { (*header_of(here)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) },
                        remembered,
                        crate::bh_probe_was_barriered(here),
                        parent.unwrap_or(0),
                        parent_remembered,
                        parent.is_some_and(crate::bh_probe_was_barriered),
                    ));
                }
                if !bad_target {
                    pending.push((GcRef(value), Some(here)));
                }
            });
        }
        // The root walk only covers what `enumerate_all_root_values` names, and
        // an object the mutator is writing to is live whether or not that
        // enumeration reaches it. Sweep every recorded old-generation block of
        // the type under investigation instead, since the population that
        // matters is "old holders with a young reference and no barrier".
        // The remembered set is traced whether or not anything still points at
        // its entries, so it is its own population: an entry the root walk
        // never reached is a dead object the minor will still read.
        for index in 0..self.old_objects_pointing_to_young.len() {
            let here = self.old_objects_pointing_to_young[index];
            let root_reachable = seen.contains(&here);
            self.visit_referent_slots(here, &mut |slot| {
                let value = unsafe { *slot }.0;
                if value == 0 || !(self.nursery.contains(value) || self.oldgen.contains(value)) {
                    return;
                }
                if (unsafe { (*header_of(value)).type_id() } as usize) < self.types.len() {
                    return;
                }
                let tid = unsafe { (*header_of(here)).type_id() };
                bad.push(format!(
                    "REMEMBERED-BAD-TARGET holder={here:#x} tid={tid} \
                     root_reachable={root_reachable} slot={:#x} offset={} value={value:#x} \
                     young={} barriered_ever={}",
                    slot as usize,
                    slot as usize - here,
                    self.nursery.contains(value),
                    crate::bh_probe_was_barriered(here),
                ));
            });
        }
        if !bad.is_empty() {
            panic!(
                "BH PROBE: {} unbarriered old->young slot(s) before minor #{}\n  {}",
                bad.len(),
                self.minor_collections,
                bad.join("\n  ")
            );
        }
    }

    /// TEMPORARY DIAGNOSTIC (`MAJIT_GC_BH_PROBE`).
    ///
    /// At the end of a minor every live nursery object has been forwarded and
    /// every traced slot rewritten, so no surviving object may still name a
    /// nursery address. Accumulate the distinct `(type, offset)` classes of
    /// those that do, then panic once with the whole table — the wasm runner
    /// recovers a guest panic message out of linear memory, so this is the one
    /// diagnostic channel that reaches the console from inside the module.
    fn bh_probe_check_no_young_refs(&mut self) {
        if !crate::bh_probe_enabled() {
            return;
        }
        // How many distinct classes to gather before reporting, and how many
        // minors to keep scanning if fewer than that ever appear. Both are
        // overridable, because the interpreter fills the old generation long
        // before the JIT compiles anything: reporting at minor #1 can only ever
        // show pre-JIT allocations.
        let class_budget = read_uint_from_env("MAJIT_GC_BH_PROBE_CLASSES").unwrap_or(10);
        let report_at_minor = read_uint_from_env("MAJIT_GC_BH_PROBE_MINOR").unwrap_or(120);
        // The reachability walk below is a whole-heap traversal per minor, far
        // too slow to leave on for a whole run; skip the minors before the one
        // being investigated.
        if self.minor_collections < read_uint_from_env("MAJIT_GC_BH_PROBE_FROM").unwrap_or(0) {
            return;
        }
        const WORD: usize = std::mem::size_of::<usize>();
        // The recorded allocation size and origin of each old-generation block,
        // which the walk below does not carry.
        let blocks: std::collections::HashMap<usize, (usize, u8, &'static str)> =
            crate::with_bh_objects(|objects| {
                objects
                    .iter()
                    .map(|&(addr, payload_size, origin, phase)| {
                        (addr, (payload_size, origin, phase))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut fresh: Vec<crate::BhProbeViolation> = Vec::new();
        {
            for (addr, offset, word, parent) in self.bh_probe_stale_young_slots() {
                let (payload_size, origin, phase) =
                    blocks
                        .get(&addr)
                        .copied()
                        .unwrap_or((0, crate::BH_PROBE_ORIGIN_BORN_OLD, "?"));
                let tid = unsafe { (*header_of(addr)).type_id() };
                if (tid as usize) >= self.types.len() || crate::bh_probe_tid_ignored(tid) {
                    continue;
                }
                {
                    if crate::bh_probe_violation_seen(tid, offset, origin) {
                        continue;
                    }
                    let info = self.types.get(tid);
                    fresh.push(crate::BhProbeViolation {
                        minor: self.minor_collections,
                        origin,
                        phase,
                        holder: addr,
                        tid,
                        payload_size,
                        offset,
                        value: word,
                        forwarded: unsafe { (*header_of(word)).is_forwarded() },
                        type_size: info.size,
                        item_size: info.item_size,
                        length_offset: info.length_offset,
                        gc_ptr_offsets: info.gc_ptr_offsets.to_vec(),
                        items_have_gc_ptrs: info.items_have_gc_ptrs,
                        custom_trace: info.custom_trace.is_some(),
                        is_object: info.is_object,
                        // Only an `rclass.OBJECT` layout has a type pointer in
                        // its first word, and only a *registered* vtable is
                        // safe to dereference — a born-old block the allocator
                        // has handed out but its caller has not filled yet
                        // carries whatever the zero-fill left there.
                        holder_name: if info.is_object
                            && self
                                .vtable_to_type_id
                                .contains_key(&unsafe { *(addr as *const usize) })
                        {
                            crate::bh_probe_type_name(addr)
                        } else {
                            None
                        },
                        // The value moved, so its copy is a live object whose
                        // type names the field far better than an offset does.
                        value_name: {
                            let hdr = unsafe { header_of(word) };
                            let moved = unsafe { (*hdr).is_forwarded() }
                                .then(|| unsafe { GcHeader::forwarding_address(hdr) });
                            moved
                                .filter(|&m| {
                                    self.vtable_to_type_id
                                        .contains_key(&unsafe { *(m as *const usize) })
                                })
                                .and_then(crate::bh_probe_type_name)
                        },
                        neighbourhood: {
                            let lo = offset.saturating_sub(2 * WORD);
                            let hi = (offset + 3 * WORD).min(payload_size.max(offset + WORD));
                            (lo..hi)
                                .step_by(WORD)
                                .map(|o| (o, unsafe { *((addr + o) as *const usize) }))
                                .collect()
                        },
                        track_young_ptrs: unsafe {
                            (*header_of(addr)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS)
                        },
                        remembered: self.old_objects_pointing_to_young.contains(&addr),
                        barriered_ever: crate::bh_probe_was_barriered(addr),
                        traced_this_minor: crate::bh_probe_was_traced(addr),
                        store_sites: crate::bh_probe_store_sites(addr, offset),
                        parent,
                        parent_tid: parent.map(|p| unsafe { (*header_of(p)).type_id() }),
                        parent_name: parent
                            .filter(|&p| {
                                self.vtable_to_type_id
                                    .contains_key(&unsafe { *(p as *const usize) })
                            })
                            .and_then(crate::bh_probe_type_name),
                        parent_remembered: parent
                            .is_some_and(|p| self.old_objects_pointing_to_young.contains(&p)),
                        parent_barriered_ever: parent.is_some_and(crate::bh_probe_was_barriered),
                        parent_traced_this_minor: parent.is_some_and(crate::bh_probe_was_traced),
                    });
                }
            }
        }
        let total = crate::bh_probe_record_violations(fresh);
        if total >= class_budget || (total > 0 && self.minor_collections >= report_at_minor) {
            panic!(
                "{}",
                crate::bh_probe_violation_report(self.minor_collections)
            );
        }
    }

    /// Allocate directly in old gen (for large objects or post-collection
    /// fallback).
    ///
    /// The block comes back uninitialized, as `external_malloc`'s "fully
    /// initialized, but not zero-filled" (incminimark.py) does: with
    /// `malloc_zero_filled = False` (incminimark.py) no allocation tier
    /// clears, and a site that needs zeroed memory appends the clear itself —
    /// see [`alloc_in_oldgen_clear`](Self::alloc_in_oldgen_clear).
    /// `external_malloc`: if the heap is already over the next-major
    /// threshold, run `minor_collection_with_major_progress` *before* the
    /// allocation, with extrasize `totalsize + nursery_size/2`.
    ///
    /// Only collecting entry points call this. A no-collect path still
    /// cannot root the caller's native locals, so it keeps the deferred
    /// breaker in `finish_alloc_*`.
    ///
    /// Returns true when the collection armed `oom_pending`: the triggering
    /// allocation must fail rather than allocate past `PYPY_GC_MAX`.
    pub fn do_maybe_collect_for_external_malloc(&mut self, totalsize: usize) -> bool {
        if !self.threshold_reached(totalsize) {
            return false;
        }
        self.pending_reserving_size = totalsize.saturating_add(self.config.nursery_size / 2);
        self.minor_collection_with_major_progress(false);
        self.pending_reserving_size = 0;
        std::mem::take(&mut self.oom_pending)
    }

    fn alloc_in_oldgen(&mut self, type_id: u32, total_size: usize) -> GcRef {
        let ptr = self.oldgen.alloc(total_size);
        self.finish_alloc_in_oldgen(type_id, total_size, ptr, GcFlags::empty())
    }

    /// `framework.py gct_do_malloc_fixedsize_clear`: the plain malloc
    /// above, then the `emit_raw_memclear` the transform appends after it.
    ///
    /// This is the out-of-line tail of the entry points that stand in for the
    /// `_clear` malloc variant — [`alloc_with_type`](Self::alloc_with_type) and
    /// [`alloc_with_type_rooted`](Self::alloc_with_type_rooted), which compiled
    /// code reaches as `do_malloc_fixedsize_clear`, and `alloc_oldgen_typed`,
    /// whose resume materialization writes only the fields present in
    /// resumedata and needs the PyFrame owned-content fields it omits to read
    /// null for the destructor.
    ///
    /// Only the fallback arms take it, never the nursery bump they fall back
    /// from. `handle_new_fixedsize` (rewrite.py) reaches this variant
    /// on one arm only — `gen_malloc_fixedsize`, whose callee is
    /// `do_malloc_fixedsize_clear` (gc.py:481-489, 493-499). Its other arm,
    /// `gen_malloc_nursery`, is an inline bump with no memclear at all. What
    /// both arms share is the `clear_gc_fields` that follows, and that zeroes
    /// the GC-pointer fields only (rewrite.py:498-504) — so a payload byte
    /// outside those fields is the caller's to write on either arm.
    ///
    /// [`alloc_with_type_no_collect`](Self::alloc_with_type_no_collect) and the
    /// spill stand in for the plain `malloc_fixedsize` instead, and take
    /// [`alloc_in_oldgen`](Self::alloc_in_oldgen) directly. On the clearing
    /// axis only: `gct_do_malloc_fixedsize` appends no memclear, which is what
    /// those two want. They part from it on the collection axis, where its
    /// nursery overflow collects (incminimark.py) and they spill.
    fn alloc_in_oldgen_clear(&mut self, type_id: u32, total_size: usize) -> GcRef {
        let obj = self.alloc_in_oldgen(type_id, total_size);
        Self::raw_memclear(obj, total_size);
        obj
    }

    /// `framework.py emit_raw_memclear` — zero the block after the header.
    #[inline]
    fn raw_memclear(obj: GcRef, total_size: usize) {
        // SAFETY: `obj` names the payload of a block the allocator has just
        // returned, whose header it has already stamped.
        unsafe {
            std::ptr::write_bytes(
                obj.0 as *mut u8,
                0,
                total_size.saturating_sub(GcHeader::SIZE),
            );
        }
    }

    /// Old-gen spill of a nursery request. JIT helpers zero the payload they
    /// return; interpreter constructors write their own fields. Recycled
    /// bytes stay dirty on every target.
    #[inline]
    fn clear_nursery_substitute(_obj: GcRef, _total_size: usize) {}

    /// [`alloc_in_oldgen`](Self::alloc_in_oldgen) for a request the nursery
    /// could not serve; see [`clear_nursery_substitute`](Self::clear_nursery_substitute).
    fn alloc_in_oldgen_nursery_substitute(&mut self, type_id: u32, total_size: usize) -> GcRef {
        let obj = self.alloc_in_oldgen(type_id, total_size);
        Self::clear_nursery_substitute(obj, total_size);
        obj
    }

    /// [`try_alloc_in_oldgen`](Self::try_alloc_in_oldgen) for the same.
    fn try_alloc_in_oldgen_nursery_substitute(
        &mut self,
        type_id: u32,
        total_size: usize,
    ) -> Option<GcRef> {
        let obj = self.try_alloc_in_oldgen(type_id, total_size)?;
        Self::clear_nursery_substitute(obj, total_size);
        Some(obj)
    }

    /// Fallible old-gen allocation used by host allocation hooks. Upstream
    /// rawmalloc failure returns NULL so the caller can raise `MemoryError`.
    fn try_alloc_in_oldgen(&mut self, type_id: u32, total_size: usize) -> Option<GcRef> {
        let ptr = self.oldgen.try_alloc(total_size)?;
        Some(self.finish_alloc_in_oldgen(type_id, total_size, ptr, GcFlags::empty()))
    }

    /// `PYRE_GC_SIZE_AUDIT`: refuse a block too small for the type its header
    /// is about to claim.
    ///
    /// The header carries the type id and nothing else; the block's extent
    /// comes from the caller's `total_size`. When the two disagree the
    /// collector reads the neighbouring blocks as if they were this object's
    /// fields, and the first symptom appears an arbitrary number of
    /// collections later, in whichever object happened to follow. Ask at the
    /// stamp instead, where the caller is still on the stack.
    ///
    /// Varsize types are exempt: their items live past `size` and the length
    /// is not readable here.
    /// `PYRE_GC_SIZE_AUDIT`. Read once; `finish_alloc_in_oldgen` asks this
    /// together with the other birth diagnostics.
    fn allocation_size_audit_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var_os("PYRE_GC_SIZE_AUDIT").is_some())
    }

    /// One flag for every per-allocation diagnostic `finish_alloc_in_oldgen`
    /// used to test separately: `bh_probe_enabled`, `PYRE_GC_SIZE_AUDIT`,
    /// `gc_lifetime_log_enabled`, and `note_bh_object` (which is the probe).
    fn oldgen_birth_diagnostics_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            crate::bh_probe_enabled()
                || crate::gc_lifetime_log_enabled()
                || Self::allocation_size_audit_enabled()
        })
    }

    fn audit_allocation_size(&self, type_id: u32, total_size: usize, obj_addr: usize, kind: &str) {
        if !Self::allocation_size_audit_enabled() {
            return;
        }
        if (type_id as usize) >= self.types.len() {
            return;
        }
        let info = self.types.get(type_id);
        if info.item_size != 0 {
            return;
        }
        let declared = info.size;
        let payload = total_size.saturating_sub(GcHeader::SIZE);
        if payload >= declared {
            return;
        }
        panic!(
            "GC SIZE AUDIT: {kind} allocation at {obj_addr:#x} stamps type_id={type_id} \
             (declared payload {declared} bytes, {} gc offsets, is_object={}) but the block \
             carries only {payload} payload bytes (total_size={total_size}). \
             gc_state={:?} minors={} majors={}\nbacktrace:\n{}",
            info.gc_ptr_offsets.len(),
            info.is_object,
            self.gc_state,
            self.minor_collections,
            self.major_collections,
            std::backtrace::Backtrace::force_capture(),
        );
    }

    fn finish_alloc_in_oldgen(
        &mut self,
        type_id: u32,
        total_size: usize,
        ptr: *mut u8,
        extra_flags: GcFlags,
    ) -> GcRef {
        let hdr = unsafe { &mut *(ptr as *mut GcHeader) };
        // Old objects start with TRACK_YOUNG_PTRS set (they need write barrier).
        // incminimark.py:1080 ORs it in after the card question, so it reaches
        // the carded arm as well as `extra_flags = 0`.
        // An object born into the old generation while a major marking cycle is
        // in progress must also be allocated black (see `oldgen_birth_flags`).
        *hdr = GcHeader::with_flags(
            type_id,
            self.oldgen_birth_flags(extra_flags | GcFlags::GCFLAG_TRACK_YOUNG_PTRS),
        );
        // `size_objects_made_old` counts promotions, not old-gen births.
        // `external_malloc(..., alloc_young=False)` does not bump it.
        let obj_addr = (ptr as usize) + GcHeader::SIZE;
        // `malloc_fixedsize` reads destructor / finalizer / weakref off the
        // flags the caller already holds. With every birth diagnostic off and
        // a type that registers none of those, one flag and one type-table
        // read (`type_alloc_is_plain`) answer the same question.
        if !Self::oldgen_birth_diagnostics_enabled()
            && ((type_id as usize) >= self.types.len() || self.type_alloc_is_plain(type_id))
        {
            if self.threshold_reached(total_size) && deferred_major_request_wanted() {
                majit_ir::eval_breaker_word::set_gc();
            }
            return GcRef(obj_addr);
        }
        if crate::bh_probe_enabled() {
            let lo = self.nursery.start_ptr() as usize;
            crate::BH_PROBE_NURSERY_LO.store(lo, std::sync::atomic::Ordering::Relaxed);
            crate::BH_PROBE_NURSERY_HI.store(
                lo + self.nursery.size(),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        self.audit_allocation_size(type_id, total_size, obj_addr, "oldgen");
        if crate::gc_lifetime_log_enabled() {
            // Pairs with `[gc][free]`: whether a dangling reference names an
            // object freed after the referrer was born, or one already dead
            // when the referrer took it, decides between a missed marking edge
            // and a stale cached pointer, and the free line alone cannot say.
            eprintln!(
                "[gc][alloc] addr={obj_addr:#x} type_id={type_id} kind=oldgen state={:?}",
                self.gc_state
            );
        }
        if (type_id as usize) < self.types.len() {
            let info = self.types.get(type_id);
            // A destructor-bearing object that never passes through the
            // nursery (large object, or nursery-full fallback) is recorded
            // straight onto the old-destructor list so a later major
            // collection runs its destructor when it dies.
            if info.destructor.is_some() {
                self.old_objects_with_destructors.push(obj_addr);
            }
            // A weakref born directly in the old generation (large object or
            // nursery-full fallback) skips the young registration path, so
            // record it straight onto the old-weakref list. Without this its
            // `weakptr` is never invalidated and `weakref()` returns a
            // dangling pointer once the target dies.
            if info.is_weakref {
                self.old_objects_with_weakrefs.push(obj_addr);
            }
            // `q_is_old_style_finalizer` from the same `TypeInfo` read.
            if info.old_style_finalizer.is_some() {
                self.register_finalizer_index(-1, GcRef(obj_addr));
            }
        }
        crate::note_bh_object(
            obj_addr,
            total_size - GcHeader::SIZE,
            crate::BH_PROBE_ORIGIN_BORN_OLD,
        );
        // Collecting entries already ran `maybe_collect_for_external_malloc`
        // before the block exists. A no-collect path still cannot root the
        // caller's native locals, so it only arms the eval-breaker for the
        // dispatch loop, where the frame walker sees the whole root set.
        //
        // Only where that walk will actually happen: the threshold stays
        // reached until a major completes, so arming past a consumer that
        // refuses re-arms on every following allocation.
        // See [`DEFERRED_MAJOR_REQUEST_PROBE`].
        if self.threshold_reached(total_size) && deferred_major_request_wanted() {
            majit_ir::eval_breaker_word::set_gc();
        }
        GcRef(obj_addr)
    }

    /// Whether [`try_alloc_young_nonmoving_clear`](Self::try_alloc_young_nonmoving_clear)
    /// attempts a young birth for `type_id` at all: `external_malloc`'s
    /// `alloc_young=True`, unless `MAJIT_GC_YOUNG_RAWMALLOC=0` or the type
    /// refuses it.
    fn young_birth_allowed(&self, type_id: u32) -> bool {
        young_rawmalloc_enabled() && self.type_alloc_may_be_young(type_id)
    }

    /// incminimark.py `external_malloc(typeid, length, alloc_young=True)`,
    /// the birth [`alloc_in_oldgen`](Self::alloc_in_oldgen) is the
    /// `alloc_young=False` sibling of.
    ///
    /// The block is non-moving like every other rawmalloc block, and *young*:
    /// the minor collection that ends without reaching it frees it, exactly as
    /// it drops an unreached nursery object. That is what pyre has been
    /// missing — `alloc_in_oldgen`'s block can only be reclaimed by a major, so
    /// a large object allocated and abandoned inside one loop iteration stays
    /// resident for as many iterations as it takes the major threshold to
    /// arrive.
    ///
    /// Refused for a weakref type (see
    /// [`type_alloc_may_be_young`](Self::type_alloc_may_be_young)) and under
    /// `MAJIT_GC_YOUNG_RAWMALLOC=0` (see [`young_rawmalloc_enabled`]). A
    /// destructor is *not* refused.
    ///
    /// Returns `None` for either refusal and for an allocation failure.
    fn try_alloc_young_nonmoving_clear(
        &mut self,
        type_id: u32,
        total_size: usize,
    ) -> Option<GcRef> {
        if !self.young_birth_allowed(type_id) {
            return None;
        }
        let ptr = self.oldgen.try_alloc_young(total_size, 0)?;
        let obj = self.finish_alloc_young_nonmoving(type_id, total_size, ptr, GcFlags::empty());
        Self::raw_memclear(obj, total_size);
        Some(obj)
    }

    /// incminimark.py:1017-1078 with `alloc_young=True` and a card header:
    /// the varsize counterpart of
    /// [`try_alloc_young_nonmoving_clear`](Self::try_alloc_young_nonmoving_clear).
    ///
    /// Upstream reaches this through one function, because `external_malloc`
    /// takes `length` and `alloc_young` together and asks the card question
    /// before it branches on either. pyre split the two births into two
    /// entry points, and the split is what left the young one unable to carry
    /// cards at all — so a large pointer-bearing array had to be born old,
    /// which is the placement `try_alloc_young_nonmoving_clear` exists to
    /// avoid.
    ///
    /// The card bits are cleared by the allocator and the payload here, so a
    /// caller receives a zeroed array with no card set — GCFLAG_CARDS_SET is
    /// on the header from birth, but it says only "this object is young, do
    /// not go looking in `old_objects_with_cards_set` for it".
    ///
    /// Returns `None` on the same refusals its cardless sibling returns
    /// `None` on, plus a failed allocation; the caller falls back to
    /// [`Self::alloc_in_oldgen_with_cards`].
    fn try_alloc_young_nonmoving_with_cards(
        &mut self,
        type_id: u32,
        total_size: usize,
        length: usize,
        has_gc_ptrs_in_var: bool,
    ) -> Option<GcRef> {
        if !young_rawmalloc_enabled() || !self.type_alloc_may_be_young(type_id) {
            return None;
        }
        let (card_header_bytes, extra_flags) =
            self.card_header_and_flags(total_size, length, has_gc_ptrs_in_var, true);
        let ptr = self.oldgen.try_alloc_young(total_size, card_header_bytes)?;
        let obj = self.finish_alloc_young_nonmoving(type_id, total_size, ptr, extra_flags);
        Self::raw_memclear(obj, total_size);
        Some(obj)
    }

    /// The birth `external_malloc(typeid, length, alloc_young=True)` performs:
    /// young if the young rawmalloc arm will take it, old otherwise.
    ///
    /// Both arms ask [`Self::card_header_and_flags`], so an array that is too
    /// small for cards gets none whichever way it lands.
    pub fn alloc_varsize_nonmoving_prefer_young(
        &mut self,
        type_id: u32,
        total_size: usize,
        length: usize,
        has_gc_ptrs_in_var: bool,
    ) -> GcRef {
        if self.do_maybe_collect_for_external_malloc(total_size) {
            return GcRef(0);
        }
        self.try_alloc_young_nonmoving_with_cards(type_id, total_size, length, has_gc_ptrs_in_var)
            .unwrap_or_else(|| {
                self.alloc_in_oldgen_with_cards(type_id, total_size, length, has_gc_ptrs_in_var)
            })
    }

    /// Whether a type's allocation may take the young non-moving birth.
    ///
    /// The weakref flag is the one exclusion, and it is upstream's:
    /// `malloc_fixedsize`'s large arm opens with
    /// `ll_assert(not contains_weakptr, "'contains_weakptr' specified for a
    /// large object")`. A destructor is *not* excluded — that arm is reached
    /// only after the heavy-finalizer branch
    /// (`needs_finalizer and not is_finalizer_light`) has already returned, so
    /// a light destructor falls through to it, and `TypeInfo.destructor` is
    /// that light kind.
    ///
    /// A type id past the end of the table registers neither a destructor nor
    /// a weakref — `finish_alloc_in_oldgen` and `register_young_object_if_needed`
    /// both guard on the same bound — so it is plain by the same reading, not
    /// unknown.
    ///
    /// Upstream states that exclusion as an assertion rather than a branch,
    /// and it never fires: a WEAKREF is a fixed three-word structure, so no
    /// registered weakref type is oversized in the first place. The assertion
    /// is carried here as a debug one, because a live interpreter cannot be
    /// aborted on the strength of a condition that has never been reachable —
    /// but it is said out loud instead of being folded into the return value,
    /// so a type that ever does reach it is a failure to look at rather than a
    /// silent placement change. The release arm still declines, and born-old
    /// is the right answer to decline to: `invalidate_young_weakrefs` decides
    /// survival by reading a forwarding pointer, which a non-moving object
    /// never has, so a young weakref would read as dead while alive.
    #[inline]
    fn type_alloc_may_be_young(&self, type_id: u32) -> bool {
        if (type_id as usize) >= self.types.len() {
            return true;
        }
        let info = self.types.get(type_id);
        let contains_weakptr = info.is_weakref;
        if info.old_style_finalizer.is_some() {
            return false;
        }
        debug_assert!(
            !contains_weakptr,
            "'contains_weakptr' specified for a large object"
        );
        !contains_weakptr
    }

    /// The header and bookkeeping half of the young non-moving birth.
    ///
    /// `extra_flags` is `external_malloc`'s, which is 0 for the cardless arm
    /// this function was written for and `HAS_CARDS|TRACK_YOUNG_PTRS|CARDS_SET`
    /// for [`Self::try_alloc_young_nonmoving_with_cards`].
    fn finish_alloc_young_nonmoving(
        &mut self,
        type_id: u32,
        total_size: usize,
        ptr: *mut u8,
        extra_flags: GcFlags,
    ) -> GcRef {
        // incminimark.py:1013-1019: the non-card rawmalloc arm sets
        // `extra_flags = 0`, and line 1080 — the only place the young birth
        // does not reach — is what would add GCFLAG_TRACK_YOUNG_PTRS. A young
        // object needs no write barrier for a young pointer stored into it,
        // because the minor reaches it whichever way it is live. The flag
        // arrives when `visit_young_rawmalloced_object` has made it old and
        // the remembered walk sets it, exactly as it does for a nursery
        // survivor. The card arm is the exception and sets it at birth, so
        // that a card write on a still-young array is not silently dropped by
        // the barrier's TRACK_YOUNG_PTRS test.
        let hdr = unsafe { &mut *(ptr as *mut GcHeader) };
        // `YOUNG_RAWMALLOC` is this generation's membership; see the flag.
        *hdr = GcHeader::with_flags(type_id, extra_flags | GcFlags::YOUNG_RAWMALLOC);
        let obj_addr = (ptr as usize) + GcHeader::SIZE;
        if crate::gc_lifetime_log_enabled() {
            eprintln!(
                "[gc][alloc] addr={obj_addr:#x} type_id={type_id} kind=raw-young state={:?}",
                self.gc_state
            );
        }
        crate::note_bh_object(
            obj_addr,
            total_size - GcHeader::SIZE,
            crate::BH_PROBE_ORIGIN_BORN_YOUNG_RAW,
        );
        // `external_malloc` asks the threshold question for this arm as well;
        // `finish_alloc_in_oldgen` records why pyre can only defer the answer.
        //
        // `bytes_made_old_since_cycle` is deliberately NOT bumped here: nothing
        // was made old. `visit_young_rawmalloced_object` bumps it at promotion,
        // where upstream bumps `size_objects_made_old`.
        if self.threshold_reached(total_size) && deferred_major_request_wanted() {
            majit_ir::eval_breaker_word::set_gc();
        }
        // `malloc_fixedsize`'s tail runs for both arms: the large arm falls out
        // of the `if` into the same `if needs_finalizer:
        // self.young_objects_with_destructors.append(obj)`. A young rawmalloced
        // object owes that list exactly as a nursery one does.
        self.register_young_object_if_needed(type_id, obj_addr);
        GcRef(obj_addr)
    }

    /// incminimark.py `is_young_object`, second half.
    ///
    /// [`is_nursery_object_start`](Self::is_nursery_object_start) is the
    /// nursery bound alone, and it is the right question wherever the real
    /// subject is "can this move". Youth is the wider property, and only the
    /// sites upstream asks it at ask it.
    #[inline]
    pub(crate) fn is_young_rawmalloced(&self, addr: usize) -> bool {
        self.oldgen.young_rawmalloced_contains(addr)
    }

    /// incminimark.py `_visit_young_rawmalloced_object`.
    ///
    /// The minor reached a young rawmalloced object. Stamp
    /// `GCFLAG_VISITED_RMY` so `free_young_rawmalloced_objects` promotes it
    /// instead of freeing it, and put it on the lists an old object owes.
    /// Anything this collection never reaches ends without the flag and is
    /// freed at its end.
    fn visit_young_rawmalloced_object(&mut self, obj_addr: usize) {
        let hdr = unsafe { header_of(obj_addr) };
        if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_VISITED_RMY) } {
            return;
        }
        unsafe { (*hdr).set_flag(GcFlags::GCFLAG_VISITED_RMY) };
        // incminimark.py:2280-2283 `size_objects_made_old`: the promotion is
        // where these bytes become old, so it is where the major-progress
        // accounting must see them.
        let type_id = unsafe { (*hdr).type_id() };
        self.validate_type_id(type_id, obj_addr, "visit_young_rawmalloced");
        let payload = self.size_for_typeid(obj_addr, type_id, "visit_young_rawmalloced");
        self.bytes_made_old_since_cycle = self
            .bytes_made_old_since_cycle
            .saturating_add(GcHeader::SIZE + payload);
        // incminimark.py `_visit_young_rawmalloced_object`.
        // TRACK_YOUNG_PTRS is clear on an ordinary young rawmalloced birth, so
        // the first arm records it for the remembered walk.  A young card array
        // is the exception: it is born with TRACK_YOUNG_PTRS and CARDS_SET and
        // takes the second arm when promotion puts it on the dirty-card list.
        if unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) } {
            self.old_objects_pointing_to_young.push(obj_addr);
        }
        if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) } {
            debug_assert!(
                unsafe { (*hdr).has_flag(GcFlags::GCFLAG_CARDS_SET) },
                "young array: GCFLAG_HAS_CARDS without GCFLAG_CARDS_SET"
            );
            self.old_objects_with_cards_set.push(obj_addr);
        }
        // `oldgen_birth_flags` allocates black during MARKING because an object
        // entering the old generation then enters the lists this cycle will
        // sweep, after its root scan. A promotion is that same entry — but
        // unlike a fresh born-old block this object's fields are already
        // filled, so black alone would leave its children unmarked. Mark it and
        // re-queue it gray, the way `drag_out_root` does for a root first seen
        // during MARKING.
        if self.gc_state == GcState::Marking && unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_VISITED) }
        {
            unsafe { (*hdr).set_flag(GcFlags::GCFLAG_VISITED) };
            self.incr_state.more_gray_stack.push(obj_addr);
        }
    }

    /// incminimark.py `_add_to_more_objects_to_trace_if_black`.
    ///
    /// Upstream then clears `GCFLAG_VISITED` because its drain re-marks on
    /// visit. pyre's `mark_object` treats a push as already black, so the
    /// object stays VISITED across the re-push.
    fn add_to_more_objects_to_trace_if_black(&mut self, obj_addr: usize) {
        let hdr = unsafe { header_of(obj_addr) };
        if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_VISITED) } {
            self.incr_state.more_gray_stack.push(obj_addr);
        }
    }

    /// incminimark.py `_minor_collection` STATE_MARKING arm: black objects on
    /// `old_objects_pointing_to_young` / `old_objects_pointing_to_pinned` turn
    /// gray so the cycle retraces what they wrote since the last visit.
    fn regray_remembered_black_during_marking(&mut self) {
        debug_assert_eq!(self.gc_state, GcState::Marking);
        for index in 0..self.old_objects_pointing_to_young.len() {
            let obj_addr = self.old_objects_pointing_to_young[index];
            self.add_to_more_objects_to_trace_if_black(obj_addr);
        }
        for index in 0..self.old_objects_pointing_to_pinned.len() {
            let obj_addr = self.old_objects_pointing_to_pinned[index];
            self.add_to_more_objects_to_trace_if_black(obj_addr);
        }
    }

    /// incminimark.py `remove_young_arrays_from_old_objects_pointing_to_young`.
    ///
    /// An entry naming a young rawmalloced object is a contradiction: this
    /// minor visits it as an object, and if it dies the drain would be reading
    /// a freed header. Upstream drops those entries before the walk starts.
    fn remove_young_arrays_from_old_objects_pointing_to_young(&mut self) {
        let oldgen = &self.oldgen;
        self.old_objects_pointing_to_young
            .retain(|&obj_addr| !oldgen.young_rawmalloced_contains(obj_addr));
    }

    /// `IncrementalMiniMarkGC._minor_collection`: perform only the nursery
    /// collection, without major progress or the rawrefcount callback.
    ///
    /// The public wrapper below owns the stop-the-world guard.  Keeping this
    /// body separate is load-bearing for `collect_and_reserve`: if a finalizer
    /// or rawrefcount callback refills the nursery after the first collection,
    /// upstream performs one more bare `_minor_collection()` before retrying.
    fn minor_collection_body(&mut self) {
        let start = GcClock::start();
        let walk_all_mutators = crate::gc_sync::mutators_quiesced();
        if crate::majit_log_enabled() {
            eprintln!(
                "[gc][minor] start count={} remembered={} cards_set={}",
                self.minor_collections + 1,
                self.old_objects_pointing_to_young.len(),
                self.old_objects_with_cards_set.len(),
            );
        }
        self.minor_collections += 1;
        self.bh_probe_check_barriers_before_minor();
        // `IncrementalMiniMarkGC._minor_collection`: barriers describe the
        // allocation gaps from the preceding minor and are rebuilt from the
        // live pinned objects found by this one.
        self.nursery_barriers.clear();
        // `bytes_made_old_since_cycle` is the running sum of every
        // `copy_nursery_object` payload, so its delta across this collection is
        // exactly what was promoted out of the nursery. The major-collection
        // paths that reset it all run after the sample is taken below.
        let drain_sample = crate::drain_census_enabled()
            .then(|| (self.nursery.used(), self.bytes_made_old_since_cycle));
        // incminimark.py:1816: this is the survivor count for this minor only;
        // every promotion below adds its allocator-sized object extent.
        self.nursery_surviving_size = 0;
        // `IncrementalMiniMarkGC._minor_collection`: pinning does not keep an
        // object alive. Rebuild the AddressStack and count from traced edges.
        // The flag sampled here is the previous minor's, which
        // `collect_roots_in_nursery` turns into `use_jit_frame_stoppers`.
        let any_pinned_object_from_earlier = self.any_pinned_object_kept;
        self.surviving_pinned_objects.clear();
        self.pinned_objects_in_nursery = 0;
        self.any_pinned_object_kept = false;
        // incminimark.py:1787-1789: ahead of the re-gray and every root walk.
        if self.oldgen.has_young_rawmalloced() {
            self.remove_young_arrays_from_old_objects_pointing_to_young();
        }
        // incminimark.py `_minor_collection`: before any root walk, turn
        // already-black remembered / pinned-parent objects gray again so the
        // active marking cycle rescans what they wrote since the last visit.
        if self.gc_state == GcState::Marking {
            self.regray_remembered_black_during_marking();
        }
        crate::bh_probe_clear_traced();
        // Phase 1: Process roots — copy nursery objects they point to.
        // We use raw pointers to avoid borrow checker issues since
        // copy_nursery_object mutates oldgen/nursery.
        // Pinned objects are left in place (not copied to old gen).
        // Read the registered roots by index, as `seed_major_roots` does and
        // for the same reason: `drag_out_root` neither registers nor removes a
        // root, so the list cannot move under the walk, and copying it first
        // was one allocation and one pass over every registered root per minor
        // collection.
        let mut i = 0;
        while i < self.roots.roots.len() {
            let root_ptr = self.roots.roots[i];
            let gcref = unsafe { &mut *root_ptr };
            self.drag_out_root(gcref);
            i += 1;
        }

        // Phase 1b: Process shadow stack roots.
        // RPython gc.py: GcRootMap_shadowstack — walk the thread-local
        // shadow stack to find GC refs pushed by compiled JIT code.
        let mut visit_shadow_root = |gcref: &mut GcRef| {
            self.drag_out_root(gcref);
        };
        if walk_all_mutators {
            crate::shadow_stack::walk_all_roots(&mut visit_shadow_root);
        } else {
            crate::shadow_stack::walk_roots(&mut visit_shadow_root);
        }

        // Phase 1c: Process jitframe shadow stack roots.
        // `_call_header_shadowstack` pushes the frame and `root_walker.walk_roots`
        // hands each entry to `_trace_drag_out1`, which only moves a young
        // frame. An old frame's young refs are reached through the remembered
        // set: `llmodel.py execute_token` and `_reload_frame_if_necessary`
        // apply the write barrier to the frame, so `collect_oldrefs_to_nursery`
        // traces it with the jitframe custom trace (`jf_gcmap` bits).
        // Collect libc-jitframe slots so we can traverse/update them
        // after the walk finishes without reborrowing `self` inside the
        // tracer callback.
        let mut libc_jf_slots: Vec<*mut majit_ir::GcRef> = Vec::new();
        let mut visit_jf_root = |gcref: &mut GcRef| {
            if self.is_nursery_object_start(gcref.0) {
                self.drag_out_root(gcref);
            } else if self.is_young_rawmalloced(gcref.0) {
                // `_trace_drag_out` on a young rawmalloced frame: flag it
                // GCFLAG_VISITED_RMY so it survives, and queue it for the
                // remembered walk that traces its slots.
                self.visit_young_rawmalloced_object(gcref.0);
            } else if !gcref.is_null() && crate::shadow_stack::is_libc_jitframe(gcref.0) {
                // pyre dynasm extension: jitframes allocated via
                // `libc::calloc` in execute_token are neither in the
                // nursery nor the old gen. The registered libc-jitframe
                // tracer walks `jf_gcmap` bits to expose ref slots.
                crate::shadow_stack::trace_libc_jitframe(gcref.0, &mut |slot_ptr| {
                    libc_jf_slots.push(slot_ptr);
                });
            }
        };
        if walk_all_mutators {
            crate::shadow_stack::walk_all_jf_roots(&mut visit_jf_root);
        } else {
            crate::shadow_stack::walk_jf_roots(&mut visit_jf_root);
        }
        // The jitframes the backend is still holding as DEADFRAMES. A frame
        // reaches this walk exactly when it has left the shadow stack — the
        // compiled epilogue pops it before returning — and has not yet been
        // freed, which is the window in which the frontend reads its slots.
        // `llmodel.py grab_exc_value` reads those slots straight out of the frame, so
        // the interior refs are live for the whole window and nothing else in
        // this phase can see them. See `ActiveGcDeadFrameHooks`.
        crate::walk_active_live_deadframes(&mut |addr| {
            crate::shadow_stack::trace_libc_jitframe(addr, &mut |slot_ptr| {
                libc_jf_slots.push(slot_ptr);
            });
        });
        for slot_ptr in libc_jf_slots {
            let field_ref = unsafe { &mut *slot_ptr };
            self.drag_out_root(field_ref);
        }

        // Phase 1d: Process blackhole interpreter register banks.
        // blackhole.py BlackholeInterpreter.registers_r parity: each
        // active blackhole frame's ref register file is part of the GC
        // root set. RPython traces these via the RPython object graph
        // (Box arrays); pyre stores raw i64 in Vec<i64> so we walk the
        // explicit thread-local stack of register banks.
        let mut visit_bh_root = |gcref: &mut GcRef| {
            self.drag_out_root(gcref);
        };
        if walk_all_mutators {
            crate::shadow_stack::walk_all_bh_regs(&mut visit_bh_root);
        } else {
            crate::shadow_stack::walk_bh_regs(&mut visit_bh_root);
        }

        // blackhole resume construction roots (`resume.py blackhole_from_resumedata`): the
        // virtuals_cache + each frame's registers_r are filled by lazily
        // materializing virtuals before `run()` re-roots them via
        // `push_bh_regs`; forward any already-materialized nursery refs so a
        // later materialization's collection does not strand them.
        let mut visit_resume_root = |gcref: &mut GcRef| {
            self.drag_out_root(gcref);
        };
        if walk_all_mutators {
            crate::shadow_stack::walk_all_resume_ref_roots(&mut visit_resume_root);
        } else {
            crate::shadow_stack::walk_resume_ref_roots(&mut visit_resume_root);
        }

        // Phase 1e: framework.py `root_walker.walk_roots` parity — the
        // embedding runtime plugs a walker that visits
        // `PyFrame.locals_cells_stack_w` across the active f_backref chain
        // so nursery refs held only by a Python frame local or operand-stack
        // slot survive collection.
        //
        // Announce the collection kind so walkers mirroring incminimark's
        // prebuilt-object scanning can skip clean prebuilt structures during
        // a minor collection (incminimark.py:339-344
        // `old_objects_pointing_to_young`); restored to the conservative
        // Major default right after.
        //
        // `collect_roots_in_nursery` computes
        // `use_jit_frame_stoppers = not any_pinned_object_from_earlier` and
        // passes it as `is_minor`: a pinned object created before the previous
        // minor is still in the nursery and was never promoted, so the skip
        // would drop the only edge reaching it. Announce a full walk instead.
        let extra_root_walk_kind = if any_pinned_object_from_earlier {
            crate::shadow_stack::ExtraRootWalkKind::Major
        } else {
            crate::shadow_stack::ExtraRootWalkKind::Minor
        };
        crate::shadow_stack::set_extra_root_walk_kind(extra_root_walk_kind);
        let mut visit_extra_area = |gcref: &mut GcRef| {
            self.drag_out_root(gcref);
        };
        if walk_all_mutators {
            crate::shadow_stack::walk_all_extra_areas(&mut visit_extra_area);
        } else {
            crate::shadow_stack::walk_my_extra_areas(&mut visit_extra_area);
        }
        // Multi-registrar walker fan-out (rd_consts const-pool, etc.).
        crate::shadow_stack::walk_extra_roots(|gcref| {
            self.drag_out_root(gcref);
        });
        crate::shadow_stack::set_extra_root_walk_kind(
            crate::shadow_stack::ExtraRootWalkKind::Major,
        );

        // incminimark.py `_minor_collection`: old parents that reached a pinned
        // child in the previous minor must be traced again. The list is
        // swapped out after the root walk, whose callbacks record no parent,
        // and `copy_nursery_object` repopulates the fresh one only for parents
        // that still point to a pinned object.
        let old_parents_pointing_to_pinned =
            std::mem::take(&mut self.old_objects_pointing_to_pinned);
        for obj_addr in old_parents_pointing_to_pinned {
            self.trace_and_update_object(obj_addr, "minor_old_parent_pinned");
        }

        // incminimark.py:1834-1836: a mirror the C side still references roots
        // its linked object, so the P list joins the root walk before anything
        // decides which nursery objects die.
        if self.rrc.enabled {
            // Before the trace, not inside it: gathering the edges enters the
            // extension.
            self.rrc_refresh_c_edges();
            self.rrc_minor_collection_trace();
        }

        // incminimark.py:1838-1841: registered young finalizers all survive
        // this minor and move to the old-registration deque.
        if !self.probably_young_objects_with_finalizers.is_empty() {
            self.deal_with_young_objects_with_finalizers();
        }

        // GCREFTRACER write-barrier analogue: tables remembered on *this*
        // MiniMark, consumed once like `old_objects_pointing_to_young`.
        self.collect_pending_gc_tables_to_nursery();

        // incminimark.py:1843-1862: while True loop —
        // collect_cardrefs_to_nursery, then collect_oldrefs_to_nursery.
        // collect_oldrefs_to_nursery may cause new card-marked objects to
        // appear, so we must re-enter if old_objects_with_cards_set is
        // non-empty after processing the remembered set.
        loop {
            // incminimark.py:1846-1847
            if self.card_page_shift > 0 {
                self.collect_cardrefs_to_nursery();
            }

            // incminimark.py: collect_oldrefs_to_nursery.
            let mut idx = 0;
            loop {
                if idx >= self.old_objects_pointing_to_young.len() {
                    break;
                }
                let obj_addr = self.old_objects_pointing_to_young[idx];
                idx += 1;

                let hdr = unsafe { header_of(obj_addr) };
                // `_remember_young_pointer_inlined`: "addr cannot be in the
                // nursery, because nursery objects never have the flag
                // GCFLAG_TRACK_YOUNG_PTRS". Every producer of this list pushes
                // an old object, so no entry is a nursery object or the
                // forwarding stub the root walk left behind one.
                debug_assert!(
                    !unsafe { (*hdr).is_forwarded() } && !self.is_in_nursery(obj_addr),
                    "nursery object {obj_addr:#x} in old_objects_pointing_to_young"
                );

                // incminimark.py collect_oldrefs_to_nursery: set the flag
                // then trace. `seed_major_root` may already have armed
                // it on a jitframe spill; OR-ing it in is a no-op then.
                unsafe {
                    (*hdr).set_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
                }

                // Trace this old-gen object's fields and copy any nursery
                // objects they reference.
                self.trace_and_update_object(obj_addr, "minor_old_objects_pointing_to_young");
            }
            self.old_objects_pointing_to_young.clear();

            // incminimark.py:1859-1862: loop back if card-marked objects appeared.
            if self.card_page_shift > 0 && !self.old_objects_with_cards_set.is_empty() {
                continue;
            }
            // A block's C fields are edges no field walk above can reach, and
            // whether they root depends on whether the block's own object
            // survived — which only the settled walk answers.  Dragging one out
            // gives the walk more to do, so it re-enters here rather than
            // running once.
            if self.rrc_trace_c_edges_young() {
                continue;
            }
            // A mirror whose object died may have a finalizer to run first,
            // and it runs after this collection: its object comes out too.
            if self.rrc_claim_finalizers_young() {
                continue;
            }
            break;
        }

        // incminimark.py:1865-1867 — now that every live nursery
        // object has been forwarded out, walk the young weakref
        // list to update or invalidate each WEAKREF's `weakptr` slot.
        if !self.young_objects_with_weakrefs.is_empty() {
            self.invalidate_young_weakrefs();
        }

        // incminimark.py:1868-1869 — then run the destructor of each
        // dead nursery object (and promote survivors to the
        // old-destructor list). Must happen after every live nursery
        // object is forwarded (so survival is detectable) and before the
        // nursery reset below reclaims the backing bytes.
        if !self.young_objects_with_destructors.is_empty() {
            self.deal_with_young_objects_with_destructors();
        }

        // An embedder side table keyed by owner address cannot be left holding
        // a nursery key past this point: the reset below hands that address to
        // the next allocation, which would then answer to the dead owner's
        // entry.  Ask the same survival question `invalidate_young_weakrefs`
        // just answered, while the forwarding headers still read.
        // The trace above marked each live pin VISITED and appended its header
        // to `surviving_pinned_objects`. A pin that died this cycle must not
        // keep an address-keyed owner table entry alive.
        let nursery = &self.nursery;
        let oldgen = &self.oldgen;
        let mut classify_young_owner = |owner: usize| -> Option<usize> {
            if owner == 0 {
                return Some(owner);
            }
            if !nursery.contains(owner) {
                // A young rawmalloced owner is the other address this
                // collection can hand back to a later allocation. It never
                // moves, so a survivor keeps its key; one this collection did
                // not reach is about to be freed and must lose it.
                if oldgen.young_rawmalloced_contains(owner) {
                    let hdr = unsafe { header_of(owner) };
                    return unsafe { (*hdr).has_flag(GcFlags::GCFLAG_VISITED_RMY) }
                        .then_some(owner);
                }
                return Some(owner);
            }
            // A pinned object is alive and stayed put, so it never forwarded.
            let hdr = (owner - GcHeader::SIZE) as *const GcHeader;
            if unsafe {
                !(*hdr).is_forwarded()
                    && (*hdr).has_flag(GcFlags::GCFLAG_PINNED)
                    && (*hdr).has_flag(GcFlags::GCFLAG_VISITED)
            } {
                return Some(owner);
            }
            if unsafe { (*hdr).is_forwarded() } {
                Some(unsafe { GcHeader::forwarding_address(hdr) })
            } else {
                None
            }
        };
        crate::shadow_stack::reconcile_young_owner_tables(&mut classify_young_owner);

        // `IncrementalMiniMarkGC.record_pinned_object_with_shadow`: clear this
        // mapping now that every unpinned nursery object was forwarded (or
        // died), rebuilding it only from the surviving-pin AddressStack.
        // A pinned object's shadow is its stable identity address and MUST
        // survive the clear, else the next `id_or_identityhash`
        // re-allocates a fresh shadow and the identity hash changes.
        if !self.nursery_objects_shadows.is_empty() {
            if self.surviving_pinned_objects.is_empty() {
                self.nursery_objects_shadows.clear();
            } else {
                let mut new_shadows = AddressMap::default();
                let marking = self.gc_state == GcState::Marking;
                for &header_addr in &self.surviving_pinned_objects {
                    let obj_addr = header_addr + GcHeader::SIZE;
                    let Some(&shadow_addr) = self.nursery_objects_shadows.get(&obj_addr) else {
                        continue;
                    };
                    if marking {
                        // Safe because `pin` rejects types carrying GC
                        // pointers, as `record_pinned_object_with_shadow`
                        // requires before keeping the shadow black.
                        unsafe { (*header_of(shadow_addr)).set_flag(GcFlags::GCFLAG_VISITED) };
                    }
                    new_shadows.insert(obj_addr, shadow_addr);
                }
                self.nursery_objects_shadows = new_shadows;
            }
        }

        // incminimark.py:1886-1888 — every live nursery object has been
        // forwarded by now, so each young mirror's link either follows its
        // object out or reports that the object died.  Before the nursery reset
        // below, which invalidates the addresses this reads.
        if self.rrc.enabled {
            self.rrc_minor_collection_free();
        }

        // incminimark.py:1890-1893: walk the young raw-malloced objects and
        // either free them or make them old. Everything that reads survival off
        // the nursery — the weakref pass, the destructors, the owner tables,
        // the rawrefcount free — has already run, and the nursery reset that
        // invalidates those readings has not.
        if self.oldgen.has_young_rawmalloced() {
            self.oldgen.free_young_rawmalloced_objects();
        }

        if let Some((used_before, promoted_before)) = drain_sample {
            crate::drain_census_record(
                used_before,
                self.bytes_made_old_since_cycle
                    .saturating_sub(promoted_before),
                self.surviving_pinned_objects.len(),
                self.nursery.size(),
            );
        }
        self.bh_probe_check_no_young_refs();

        // `IncrementalMiniMarkGC._minor_collection`: reset only free gaps,
        // clear survivor VISITED bits, and build the sorted barrier deque.
        self.reset_nursery_with_pinned();
        // incminimark.py:1949-1951: the PINNED bit is reused on old objects as
        // PINNED_OBJECT_PARENT_KNOWN only within one minor collection.
        for &obj_addr in &self.old_objects_pointing_to_pinned {
            unsafe {
                (*header_of(obj_addr)).clear_flag(GcFlags::GCFLAG_PINNED_OBJECT_PARENT_KNOWN)
            };
        }
        self.refresh_published_nursery_top();

        // incminimark.py `_minor_collection`: `if self.DEBUG >= 2` — the whole
        // heap is walked, so it is gated a level above the rotating nurseries
        // rather than on `PYPY_GC_DEBUG` being set at all.
        if self.config.debug >= 2 {
            self.debug_check_consistency_at("minor");
        }

        // incminimark.py `self.root_walker.finished_minor_collection()`,
        // the callback framework.py:135-138 reads out of `_jit2gc`: after the
        // nursery is reset and accounted for, and before the timing and the
        // gc-minor hook below.
        crate::invoke_after_minor_collection_hook();

        // incminimark.py:1962-1974 — report the completed minor before the
        // wrapper advances the incremental major state machine.
        let duration = start.elapsed_secs();
        self.total_gc_time += duration;
        self.hooks.fire_gc_minor(
            duration,
            self.get_total_memory_used(),
            self.pinned_objects_in_nursery,
        );
    }

    /// `IncrementalMiniMarkGC._minor_collection`, including pyre's
    /// stop-the-world ownership but deliberately excluding the wrapper tail.
    fn minor_collection_only(&mut self) {
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        self.minor_collection_body();
    }

    /// Perform `minor_collection_with_major_progress(force_enabled=False)`.
    ///
    /// 1. Scan roots and evacuate the nursery.
    /// 2. Advance the incremental major until both target invariants hold.
    /// 3. Schedule rawrefcount work queued by either collection.
    pub fn do_collect_nursery(&mut self) {
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        self.minor_collection_body();
        self.run_major_progress_after_minor(false);
        self.rrc_invoke_callback();
        self.rrc_invoke_cpyext_drain();
    }

    /// incminimark.py `invalidate_young_weakrefs(self)`.
    ///
    /// For each WEAKREF object recorded in `young_objects_with_weakrefs`:
    ///   * If the weakref itself was not forwarded out, it died — skip.
    ///   * Otherwise read the `weakptr` slot off the forwarded payload.
    ///     If the target is a nursery object: forward → update slot;
    ///     not forwarded → invalidate slot (set to null). If the target
    ///     is in old-gen, the weakref survives — push onto
    ///     `old_objects_with_weakrefs` for the next major cycle.
    ///
    /// The young raw-malloced target is the second branch
    /// (incminimark.py): it survives without moving, so the slot
    /// needs no update and only its death has to be written. RPython's
    /// PyPy deliberately does not support weakrefs to pinned targets (the XXX
    /// immediately above this routine says so). Its prebuilt-target branch is
    /// part of the contract, however: a `GCFLAG_NO_HEAP_PTRS` target is
    /// immortal and never receives VISITED, so registering it for the next
    /// major would make that major clear a live weakref.
    fn invalidate_young_weakrefs(&mut self) {
        while let Some(obj_addr) = self.young_objects_with_weakrefs.pop() {
            // incminimark.py:3065-3066: if not forwarded → weakref died.
            let hdr_ptr = (obj_addr - GcHeader::SIZE) as *const GcHeader;
            if !unsafe { (*hdr_ptr).is_forwarded() } {
                continue;
            }
            let new_obj = unsafe { GcHeader::forwarding_address(hdr_ptr) };

            // incminimark.py:3069-3070: pointing_to = (obj + offset)
            //                                            .address[0]
            // pyre's WEAKREF struct keeps `weakptr` at offset 0
            // (gctypelayout.py, weakref.rs:WEAKPTR_OFFSET).
            let weakptr_slot = (new_obj + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef;
            let pointing_to = unsafe { (*weakptr_slot).0 };

            // Null targets need no update or follow-up tracking.
            if pointing_to == 0 {
                continue;
            }

            // incminimark.py:3083-3090: a young raw-malloced target survives
            // in place when this collection reached it, and its address does
            // not change either way — so the surviving case updates nothing and
            // only hands the weakref to the next major cycle.
            if self.is_young_rawmalloced(pointing_to) {
                if unsafe { (*header_of(pointing_to)).has_flag(GcFlags::GCFLAG_VISITED_RMY) } {
                    self.old_objects_with_weakrefs.push(new_obj);
                } else {
                    unsafe { (*weakptr_slot).0 = 0 };
                }
                continue;
            }

            // incminimark.py:3071-3079: pointing_to in nursery.
            if self.is_nursery_object_start(pointing_to) {
                let target_hdr = (pointing_to - GcHeader::SIZE) as *const GcHeader;
                if unsafe { (*target_hdr).is_forwarded() } {
                    let fwd = unsafe { GcHeader::forwarding_address(target_hdr) };
                    unsafe { (*weakptr_slot).0 = fwd };
                    // Target lives in old gen now — weakref also lives.
                    self.old_objects_with_weakrefs.push(new_obj);
                } else {
                    // Target dies; null out the weakptr and drop the
                    // weakref from any further tracking.
                    unsafe { (*weakptr_slot).0 = 0 };
                }
                continue;
            }

            // Target is in old-gen (or a foreign address pyre's GC
            // does not own). Either way, the minor cycle leaves the
            // weakptr untouched and the major cycle gets
            // the chance to invalidate it later.
            if self.oldgen.contains(pointing_to)
                && unsafe { !(*header_of(pointing_to)).has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS) }
            {
                self.old_objects_with_weakrefs.push(new_obj);
            }
        }
    }

    /// A non-moving major leaves nursery objects in place, so the normal
    /// minor-only `invalidate_young_weakrefs` pass does not run.  A live
    /// nursery weakref may nevertheless point at an old object that this
    /// major is about to sweep.  Clear that edge while old-gen VISITED bits
    /// still distinguish survivors, and leave the nursery bookkeeping list
    /// intact for the next moving minor.
    fn invalidate_young_weakrefs_for_nonmoving_major(&mut self) {
        debug_assert!(self.oldgen_nonmoving_active);
        for obj_addr in self.young_objects_with_weakrefs.iter().copied() {
            let weakref_hdr = unsafe { header_of(obj_addr) };
            if unsafe { !(*weakref_hdr).has_flag(GcFlags::GCFLAG_VISITED) } {
                continue;
            }
            let weakptr_slot = (obj_addr + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef;
            let pointing_to = unsafe { (*weakptr_slot).0 };
            if pointing_to == 0 || !self.oldgen.contains(pointing_to) {
                continue;
            }
            let target_hdr = unsafe { header_of(pointing_to) };
            if unsafe {
                !(*target_hdr).has_flag(GcFlags::GCFLAG_VISITED)
                    || (*target_hdr).has_flag(GcFlags::GCFLAG_FINALIZATION_ORDERING)
            } {
                unsafe { (*weakptr_slot).0 = 0 };
            }
        }
    }

    // RawRefCount — incminimark.py:3157-3409

    /// incminimark.py `rawrefcount_init`.
    ///
    /// Idempotent, as upstream's `if not self.rrc_enabled` is: whichever of the
    /// embedder's entry points runs first may call it.
    pub fn rawrefcount_init(&mut self, dealloc_trigger: rawrefcount::DeallocTriggerFn) {
        if !self.rrc.enabled {
            self.rrc = rawrefcount::RawRefCount {
                enabled: true,
                dealloc_trigger: Some(dealloc_trigger),
                ..Default::default()
            };
        }
    }

    /// `rrc_enabled` — whether [`rawrefcount_init`](Self::rawrefcount_init) has
    /// run.  Every phase call site is guarded on it.
    pub fn rawrefcount_enabled(&self) -> bool {
        self.rrc.enabled
    }

    /// Register the [`rawrefcount::CEdgeCensusFn`].
    ///
    /// Separate from [`rawrefcount_init`](Self::rawrefcount_init) because it is
    /// optional: an embedder whose blocks never reference each other has
    /// nothing for it to report, and without it every mirror is judged on its
    /// bare count, which is what upstream does.
    pub fn rawrefcount_set_c_edge_census(&mut self, census: rawrefcount::CEdgeCensusFn) {
        self.rrc.c_edge_census = Some(census);
    }

    /// Register the [`rawrefcount::FinalizerClaimFn`].
    pub fn rawrefcount_set_finalizer_claim(&mut self, claim: rawrefcount::FinalizerClaimFn) {
        self.rrc.finalizer_claim = Some(claim);
    }

    /// Register the allocator-matched release used by `_rrc_free` for an
    /// exact [`rawrefcount::REFCNT_FROM_PYPY_LIGHT`] mirror.
    pub fn rawrefcount_set_light_free(&mut self, free: rawrefcount::LightFreeFn) {
        self.rrc.light_free = Some(free);
    }

    /// Take this collection's edges from the embedder and count, per mirror,
    /// how many of its references the reporting blocks supply.
    ///
    /// The call enters the extension, so every caller must be positioned before
    /// the phase that reads the answer rather than inside it.
    fn rrc_refresh_c_edges(&mut self) {
        let Some(census) = self.rrc.c_edge_census else {
            return;
        };
        self.rrc.c_edges = census();
        self.rrc.c_discount.clear();
        for (_, referents) in &self.rrc.c_edges {
            for &referent in referents {
                *self.rrc.c_discount.entry(referent).or_insert(0) += 1;
            }
        }
    }

    /// Whether `mirror`'s count makes it root its linked object.
    ///
    /// `incminimark.py:_rrc_minor_trace` treats either exact link share as
    /// unreferenced. Every count above its corresponding share is one C holds,
    /// and holding it is what roots. The subtraction after that is
    /// [`rawrefcount::CEdgeCensusFn`]'s: a reference
    /// another block in the census supplies is not one from outside the heap,
    /// and whether *that* block's own object lives is settled by the trace, in
    /// [`Self::mark_c_edges`], rather than assumed here.
    fn rrc_roots_link(&self, mirror: usize) -> bool {
        let rc = unsafe { (*rawrefcount::pyobj(mirror)).ob_refcnt };
        let share = if rc >= rawrefcount::REFCNT_FROM_PYPY_LIGHT {
            rawrefcount::REFCNT_FROM_PYPY_LIGHT
        } else {
            rawrefcount::REFCNT_FROM_PYPY
        };
        let held = rc - share;
        held > self.rrc.c_discount.get(&mirror).copied().unwrap_or(0)
    }

    /// The interpreter object a census entry names, or zero once the collector
    /// has cleared the link.
    fn rrc_linked_object(mirror: usize) -> usize {
        unsafe { (*rawrefcount::pyobj(mirror)).ob_link }
    }

    /// Whether a major has decided `obj` lives — the question
    /// [`Self::_rrc_major_free`] asks of a mirror's link, asked one step
    /// earlier so that what a live block references can still be marked.
    fn rrc_object_marked(&self, obj: usize) -> bool {
        if obj == 0 {
            return false;
        }
        if !self.is_managed_heap_object(obj) {
            return true;
        }
        let hdr = unsafe { header_of(obj) };
        unsafe {
            (*hdr).has_flag(GcFlags::GCFLAG_VISITED)
                || (*hdr).has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS)
        }
    }

    /// Mark what the C fields of a block whose own object survived reference.
    ///
    /// The other end of the subtraction in [`Self::rrc_roots_link`]: a
    /// reference a census block supplies stops rooting on its own account, and
    /// this is where it roots again once that block's object is known to live.
    /// Which is knowable only here, after ordinary marking — before it, the
    /// question has no answer, and answering it anyway frees a live object that
    /// nothing but a C field references.
    ///
    /// Reading the gathered edges rather than calling `tp_traverse` is what
    /// keeps the extension off the collector's stack: the walk that produced
    /// them ran before this collection traced anything.
    ///
    /// Returns how many objects this marked, which is what makes it a step of a
    /// fixed point — a newly live block reports edges of its own.
    fn mark_c_edges(&mut self) -> usize {
        if !self.rrc.enabled || self.rrc.c_edges.is_empty() {
            return 0;
        }
        let edges = std::mem::take(&mut self.rrc.c_edges);
        let mut marked = 0usize;
        for (block, referents) in &edges {
            if !self.rrc_object_marked(Self::rrc_linked_object(*block)) {
                continue;
            }
            for &referent in referents {
                let obj = Self::rrc_linked_object(referent);
                if obj == 0 || self.rrc_object_marked(obj) {
                    continue;
                }
                self.seed_major_root(GcRef(obj), "rrc_c_edge");
                while let Some(addr) = self.incr_state.gray_stack.pop() {
                    self.mark_object(addr);
                    marked += 1;
                }
            }
        }
        self.rrc.c_edges = edges;
        marked
    }

    /// Whether this mirror's object has to outlive the collection: its
    /// finalizer is claimed here, or was claimed by an earlier collection whose
    /// drain has not run yet.
    ///
    /// The second case is what keeps the retention for as long as it is owed.
    /// The drain is an action the embedder runs between bytecodes, and a second
    /// collection can reach here before it has: freeing the block there would
    /// leave the queue naming one whose references are already gone.
    fn rrc_owes_finalizer(&mut self, mirror: usize) -> bool {
        if self.rrc.finalize_pending.contains(&mirror) {
            return true;
        }
        let Some(claim) = self.rrc.finalizer_claim else {
            return false;
        };
        if !claim(mirror) {
            return false;
        }
        self.rrc.finalize_pending.push_back(mirror);
        true
    }

    /// Claim the finalizer of every mirror this collection has found dead, and
    /// keep its object alive until the drain has run it.
    ///
    /// `gcmodule.c:finalize_garbage` runs before `delete_garbage` for the same
    /// reason this runs before the free pass: a finalizer reads the object it
    /// is handed, and the fields of the block it belongs to still reference
    /// what they referenced.  Marking the object here is what keeps both --
    /// [`Self::mark_c_edges`] re-grants the block's own references once its
    /// object is known to live, so the caller re-enters the fixed point and the
    /// whole of what the finalizer will reach comes with it.
    ///
    /// The free pass below then reads the mirror as surviving, which defers it
    /// by one collection rather than freeing it under the finalizer's feet.
    /// Every mirror whose object is dying is asked, not only the ones whose
    /// block the free pass would bring to zero: a block the C side still holds
    /// a reference to is deallocated later, and its finalizer reads the same
    /// fields whenever that is.
    /// [`rawrefcount::FinalizerClaimFn`] answers only once per block, so the
    /// next collection to find the object dead frees it.
    ///
    /// Returns how many objects this marked, which is what makes the caller
    /// re-enter the fixed point.
    fn rrc_claim_finalizers(&mut self) -> usize {
        if self.rrc.finalizer_claim.is_none() {
            return 0;
        }
        let mut marked = 0usize;
        for which in [RrcList::P, RrcList::O] {
            let list = match which {
                RrcList::P => std::mem::take(&mut self.rrc.p_list_old),
                RrcList::O => std::mem::take(&mut self.rrc.o_list_old),
            };
            for &mirror in &list {
                let obj = Self::rrc_linked_object(mirror);
                if obj == 0 || self.rrc_object_marked(obj) || !self.rrc_owes_finalizer(mirror) {
                    continue;
                }
                self.seed_major_root(GcRef(obj), "rrc_finalizer");
                while let Some(addr) = self.incr_state.gray_stack.pop() {
                    self.mark_object(addr);
                    marked += 1;
                }
            }
            match which {
                RrcList::P => self.rrc.p_list_old = list,
                RrcList::O => self.rrc.o_list_old = list,
            }
        }
        marked
    }

    /// Whether a minor has decided `obj` lives.
    ///
    /// Only a nursery object can die here, and it is exactly
    /// [`Self::_rrc_minor_free`]'s test — read before survivor VISITED bits are
    /// cleared, so the header identifies this collection's surviving pins.
    #[inline]
    fn is_surviving_pinned(&self, obj: usize) -> bool {
        if !self.is_nursery_object_start(obj) {
            return false;
        }
        let hdr = unsafe { &*header_of(obj) };
        !hdr.is_forwarded()
            && hdr.has_flag(GcFlags::GCFLAG_PINNED)
            && hdr.has_flag(GcFlags::GCFLAG_VISITED)
    }

    fn rrc_young_object_alive(&self, obj: usize) -> bool {
        if obj == 0 {
            return false;
        }
        if self.is_young_rawmalloced(obj) {
            // incminimark.py:3300-3306: a young raw-malloced object survives
            // without moving, and GCFLAG_VISITED_RMY is what says so.
            return unsafe { (*header_of(obj)).has_flag(GcFlags::GCFLAG_VISITED_RMY) };
        }
        if !self.is_nursery_object_start(obj) {
            return true;
        }
        if self.is_surviving_pinned(obj) {
            return true;
        }
        unsafe { (*header_of(obj)).is_forwarded() }
    }

    /// [`Self::mark_c_edges`] for a minor: drag out the nursery objects a
    /// surviving block's C fields reference.
    ///
    /// Returns whether anything was dragged out, which the caller turns into
    /// the fixed point by re-entering the walk that traces what this reached.
    fn rrc_trace_c_edges_young(&mut self) -> bool {
        if !self.rrc.enabled || self.rrc.c_edges.is_empty() {
            return false;
        }
        let edges = std::mem::take(&mut self.rrc.c_edges);
        let mut dragged = false;
        for (block, referents) in &edges {
            if !self.rrc_young_object_alive(Self::rrc_linked_object(*block)) {
                continue;
            }
            for &referent in referents {
                let obj = Self::rrc_linked_object(referent);
                if self.rrc_young_object_alive(obj) {
                    continue;
                }
                if self.is_young_rawmalloced(obj) {
                    // Young and reachable from a surviving block's C field, so
                    // it survives — but it does not move, so flagging it is the
                    // whole of dragging it out.
                    self.visit_young_rawmalloced_object(obj);
                    dragged = true;
                    continue;
                }
                if !self.is_nursery_object_start(obj) {
                    // Neither dead nor this collection's business: an old
                    // object survives every minor.
                    continue;
                }
                let mut root = GcRef(obj);
                self.drag_out_root(&mut root);
                dragged = true;
            }
        }
        self.rrc.c_edges = edges;
        dragged
    }

    /// [`Self::rrc_claim_finalizers`] for a minor: keep the nursery object of a
    /// mirror whose finalizer has yet to run out of this collection's reach.
    ///
    /// Returns whether anything was dragged out, which the caller turns into
    /// the fixed point by re-entering the walk that traces what this reached.
    fn rrc_claim_finalizers_young(&mut self) -> bool {
        if self.rrc.finalizer_claim.is_none() {
            return false;
        }
        let mut dragged = false;
        for which in [RrcList::P, RrcList::O] {
            let list = match which {
                RrcList::P => std::mem::take(&mut self.rrc.p_list_young),
                RrcList::O => std::mem::take(&mut self.rrc.o_list_young),
            };
            for &mirror in &list {
                let obj = Self::rrc_linked_object(mirror);
                if obj == 0
                    || self.rrc_young_object_alive(obj)
                    || !self.is_nursery_object_start(obj)
                    || !self.rrc_owes_finalizer(mirror)
                {
                    continue;
                }
                let mut root = GcRef(obj);
                self.drag_out_root(&mut root);
                dragged = true;
            }
            match which {
                RrcList::P => self.rrc.p_list_young = list,
                RrcList::O => self.rrc.o_list_young = list,
            }
        }
        dragged
    }

    /// incminimark.py `rawrefcount_create_link_pypy`.
    ///
    /// Upstream files a link by two independent questions — which list (young
    /// or old), and which identity table (nursery-keyed or not).  They come
    /// apart for a young raw-malloced object, which is young but not in the
    /// nursery: the list is chosen by `is_young_object`, the table by
    /// `is_in_nursery`, so such a link is on the young list and in the
    /// address-keyed table.
    ///
    /// The caller must already have added [`rawrefcount::REFCNT_FROM_PYPY`] to
    /// the mirror's count: that share is what this link is worth, and what
    /// [`Self::_rrc_free`] gives back when the object dies.
    pub fn rawrefcount_create_link_pypy(&mut self, obj: usize, mirror: usize) {
        debug_assert!(self.rrc.enabled, "rawrefcount.init not called");
        unsafe { (*rawrefcount::pyobj(mirror)).ob_link = obj };
        if self.is_in_nursery(obj) || self.is_young_rawmalloced(obj) {
            self.rrc.p_list_young.push(mirror);
        } else {
            self.rrc.p_list_old.push(mirror);
        }
        if self.is_in_nursery(obj) {
            self.rrc.p_dict_nurs.insert(obj, mirror);
        } else {
            self.rrc.p_dict.insert(obj, mirror);
        }
    }

    /// incminimark.py `rawrefcount_create_link_pyobj`.
    ///
    /// The mirror owns the interpreter object here, so no trace pass roots the
    /// object on the mirror's behalf and there is no identity table to keep.
    pub fn rawrefcount_create_link_pyobj(&mut self, obj: usize, mirror: usize) {
        debug_assert!(self.rrc.enabled, "rawrefcount.init not called");
        if self.is_in_nursery(obj) || self.is_young_rawmalloced(obj) {
            self.rrc.o_list_young.push(mirror);
        } else {
            self.rrc.o_list_old.push(mirror);
        }
        unsafe { (*rawrefcount::pyobj(mirror)).ob_link = obj };
        // there is no rrc_o_dict
    }

    /// incminimark.py `rawrefcount_mark_deallocating`.
    ///
    /// `marker` is not a GC object: it is the sentinel
    /// [`Self::rawrefcount_to_obj`] hands back while a mirror's deallocator
    /// runs, so C code that re-enters and asks for the interpreter object gets
    /// something it can recognise instead of a freed address.  A mirror is off
    /// every list by the time this is called, so no trace pass can mistake the
    /// sentinel for a reference.
    pub fn rawrefcount_mark_deallocating(&mut self, marker: usize, mirror: usize) {
        debug_assert!(self.rrc.enabled, "rawrefcount.init not called");
        unsafe { (*rawrefcount::pyobj(mirror)).ob_link = marker };
    }

    /// incminimark.py `rawrefcount_from_obj`.  Zero when unlinked.
    pub fn rawrefcount_from_obj(&self, obj: usize) -> usize {
        let dct = if self.is_in_nursery(obj) {
            &self.rrc.p_dict_nurs
        } else {
            &self.rrc.p_dict
        };
        dct.get(&obj).copied().unwrap_or(0)
    }

    /// incminimark.py `rawrefcount_to_obj`.
    pub fn rawrefcount_to_obj(&self, mirror: usize) -> usize {
        unsafe { (*rawrefcount::pyobj(mirror)).ob_link }
    }

    /// The next mirror whose finalizer the drain owes a call, or zero.
    ///
    /// [`Self::rrc_claim_finalizers`] filled the queue and kept each object
    /// alive for it; the block itself is freed by a later collection.
    pub fn rawrefcount_next_finalize(&mut self) -> usize {
        if self.rrc.enabled {
            self.rrc.finalize_pending.pop_front().unwrap_or(0)
        } else {
            0
        }
    }

    /// incminimark.py `rawrefcount_next_dead`.  Zero when the queue
    /// is empty.
    pub fn rawrefcount_next_dead(&mut self) -> usize {
        if self.rrc.enabled {
            self.rrc.dealloc_pending.pop().unwrap_or(0)
        } else {
            0
        }
    }

    /// incminimark.py `rrc_invoke_callback`.
    ///
    /// Called from the public collection entry points (`collect`,
    /// `collect_step`, `minor_collection_with_major_progress`), never from
    /// inside a phase: the collector is borrowed, so the callback may only
    /// schedule the drain, not perform it.
    fn rrc_invoke_callback(&mut self) {
        if self.rrc.enabled
            && !self.rrc.dealloc_pending.is_empty()
            && let Some(trigger) = self.rrc.dealloc_trigger
        {
            trigger();
        }
    }

    /// Schedule `drain_dead` for work PyPy's `rrc_invoke_callback` does not
    /// see: claimed `tp_finalize` blocks and C-only cycles.
    ///
    /// Pyre's spec is the free-threaded build only. This is not the
    /// GIL generational collector and must not grow into a second compiled
    /// GC. `tp_finalize` and C-side cycles are still observables there; a
    /// collection that queued only those two would leave the drain
    /// unscheduled if this sat inside `rrc_invoke_callback`. Called from
    /// the same entry points so that function stays dealloc-only.
    fn rrc_invoke_cpyext_drain(&mut self) {
        if self.rrc.enabled
            && (!self.rrc.finalize_pending.is_empty() || self.rrc.c_garbage)
            && let Some(trigger) = self.rrc.dealloc_trigger
        {
            self.rrc.c_garbage = false;
            trigger();
        }
    }

    /// incminimark.py `rrc_minor_collection_trace`.
    ///
    /// Every key in the nursery-keyed table is about to move, so the table is
    /// emptied here and refilled with the survivors' new addresses by
    /// [`Self::_rrc_minor_free`].
    fn rrc_minor_collection_trace(&mut self) {
        self.rrc.p_dict_nurs.clear();
        let young = std::mem::take(&mut self.rrc.p_list_young);
        for &mirror in &young {
            self._rrc_minor_trace(mirror);
        }
        self.rrc.p_list_young = young;
    }

    /// incminimark.py `_rrc_minor_trace`.
    ///
    /// The forwarded address is deliberately not written back to the link:
    /// upstream drags the object out through a scratch cell, and
    /// [`Self::_rrc_minor_free`] then reads the forwarding bit at the *old*
    /// address to decide whether the mirror survives.
    fn _rrc_minor_trace(&mut self, mirror: usize) {
        if !self.rrc_roots_link(mirror) {
            // Nothing outside the census references this mirror, so the linked
            // object may die.
            return;
        }
        let mut root = GcRef(unsafe { (*rawrefcount::pyobj(mirror)).ob_link });
        debug_assert!(!root.is_null(), "a mirror on the P list has a link");
        self.drag_out_root(&mut root);
    }

    /// incminimark.py `rrc_minor_collection_free`.
    ///
    /// `still_young` collects the entries that survived without leaving the
    /// nursery, which upstream has no case for; see [`Self::_rrc_minor_free`].
    fn rrc_minor_collection_free(&mut self) {
        debug_assert!(self.rrc.p_dict_nurs.is_empty(), "p_dict_nurs not empty 1");
        let mut young = std::mem::take(&mut self.rrc.p_list_young);
        let mut still_young = Vec::new();
        while let Some(mirror) = young.pop() {
            self._rrc_minor_free(mirror, RrcList::P, &mut still_young);
        }
        self.rrc.p_list_young = still_young;
        let mut young = std::mem::take(&mut self.rrc.o_list_young);
        let mut still_young = Vec::new();
        while let Some(mirror) = young.pop() {
            self._rrc_minor_free(mirror, RrcList::O, &mut still_young);
        }
        self.rrc.o_list_young = still_young;
    }

    /// incminimark.py `_rrc_minor_free`.
    fn _rrc_minor_free(&mut self, mirror: usize, list: RrcList, still_young: &mut Vec<usize>) {
        let obj = unsafe { (*rawrefcount::pyobj(mirror)).ob_link };
        // incminimark.py:3300-3312: a young raw-malloced target survives
        // without moving and is recognised by GCFLAG_VISITED_RMY rather than by
        // a forwarding pointer. Its link already names its final address, so a
        // survivor only changes lists.
        if self.is_young_rawmalloced(obj) {
            if unsafe { (*header_of(obj)).has_flag(GcFlags::GCFLAG_VISITED_RMY) } {
                match list {
                    RrcList::P => {
                        self.rrc.p_dict.insert(obj, mirror);
                        self.rrc.p_list_old.push(mirror);
                    }
                    RrcList::O => self.rrc.o_list_old.push(mirror),
                }
            } else {
                if list == RrcList::P {
                    // Dying young large object: it was keyed in `p_dict`, not
                    // in the nursery table this collection empties, so the
                    // entry has to be removed by hand.
                    self.rrc.p_dict.remove(&obj);
                }
                self._rrc_free(mirror);
            }
            return;
        }
        debug_assert!(
            self.is_nursery_object_start(obj),
            "a young rawrefcount list holds a non-nursery link"
        );
        let hdr = unsafe { header_of(obj) };
        if unsafe { (*hdr).is_forwarded() } {
            let moved = unsafe { GcHeader::forwarding_address(hdr) };
            unsafe { (*rawrefcount::pyobj(mirror)).ob_link = moved };
            match list {
                RrcList::P => {
                    // It was keyed in `p_dict_nurs`, which this collection
                    // emptied; at its new address it belongs in `p_dict`.
                    self.rrc.p_dict.insert(moved, mirror);
                    self.rrc.p_list_old.push(mirror);
                }
                RrcList::O => self.rrc.o_list_old.push(mirror),
            }
        } else if self.is_surviving_pinned(obj) {
            // A pinned object that was reached survives *in place*
            // (`copy_nursery_object` records it and returns the same address),
            // so there is no forwarding pointer and the link already names its
            // final address.  It is still a nursery address, so the entry stays
            // on the young list and in the nursery-keyed table.
            //
            // incminimark.py:3287-3299 reads only the forwarding bit, so a
            // pinned P-linked object reports there as dead.  `pin` refuses any
            // type carrying GC pointers, which is why the combination is rare
            // rather than impossible: a pinned byte buffer with a mirror is
            // exactly what `PyBytes_AsString` over a pinned buffer produces.
            // VISITED is still set on this collection's surviving pins here,
            // so an unreached pinned object still falls through to the free.
            if list == RrcList::P {
                self.rrc.p_dict_nurs.insert(obj, mirror);
            }
            still_young.push(mirror);
        } else {
            self._rrc_free(mirror);
        }
    }

    /// incminimark.py `_rrc_free`.
    ///
    /// The linked object has died, so the interpreter's share of the count goes
    /// away. `incminimark.py:_rrc_free` releases an exact LIGHT mirror directly
    /// with raw `free`; a LIGHT mirror with external references is merely
    /// unlinked. The ordinary share still queues a zero-count mirror for its
    /// type-specific C deallocator.
    fn _rrc_free(&mut self, mirror: usize) {
        let header = rawrefcount::pyobj(mirror);
        let mut rc = unsafe { (*header).ob_refcnt };
        debug_assert!(rc < rawrefcount::REFCNT_IMMORTAL);
        if rc >= rawrefcount::REFCNT_FROM_PYPY_LIGHT {
            rc -= rawrefcount::REFCNT_FROM_PYPY_LIGHT;
            if rc == 0 {
                let free = self
                    .rrc
                    .light_free
                    .expect("rawrefcount LIGHT mirror has no allocator release callback");
                free(mirror);
            } else {
                unsafe {
                    (*header).ob_refcnt = rc;
                    (*header).ob_link = 0;
                }
            }
            return;
        }
        debug_assert!(
            rc >= rawrefcount::REFCNT_FROM_PYPY,
            "rawrefcount refcount underflow"
        );
        debug_assert!(
            (rc as f64) < (rawrefcount::REFCNT_FROM_PYPY_LIGHT as f64 * 0.99),
            "refcount underflow from REFCNT_FROM_PYPY_LIGHT"
        );
        rc -= rawrefcount::REFCNT_FROM_PYPY;
        unsafe { (*header).ob_link = 0 };
        if rc == 0 {
            // incminimark.py:3339-3349 — a mirror at count 0 cannot sit and
            // wait for its deallocator: some extensions read the raw pointer
            // back and expect the deallocator to have run the moment the count
            // reached 0.  Queue it and leave it at 1, so the drain's own
            // release is what frees it.
            self.rrc.dealloc_pending.push(mirror);
            rc = 1;
        } else {
            // The link died and references are left over, so they come from
            // blocks whose own links died in this same pass — the ends of a
            // cycle the census made collectable.  Nothing here can break it:
            // the references live in C fields, and dropping one is the
            // extension's `tp_clear`.  Report that there is such work, which
            // an empty dealloc queue would otherwise not say.
            self.rrc.c_garbage = true;
        }
        unsafe { (*header).ob_refcnt = rc };
    }

    /// incminimark.py `rrc_major_collection_trace`.
    fn rrc_major_collection_trace(&mut self) {
        let old = std::mem::take(&mut self.rrc.p_list_old);
        for &mirror in &old {
            self._rrc_major_trace(mirror);
        }
        self.rrc.p_list_old = old;
    }

    /// incminimark.py `_rrc_major_trace`.
    fn _rrc_major_trace(&mut self, mirror: usize) {
        if !self.rrc_roots_link(mirror) {
            return;
        }
        let obj = unsafe { (*rawrefcount::pyobj(mirror)).ob_link };
        self.seed_major_root(GcRef(obj), "rrc_major_trace");
        self.drain_gray_stack();
    }

    /// A non-moving major has no leading minor, so a young object can already
    /// sit on `old_objects_with_finalizers` (a pinned identity copy from
    /// `deal_with_young_objects_with_finalizers`). incminimark never sees that:
    /// the MARKING seam asserts `probably_young_objects_with_finalizers` empty
    /// and then runs `deal_with_objects_with_finalizers` over old objects only.
    /// Move those young entries back so this pass leaves them for the next
    /// minor's `deal_with_young_objects_with_finalizers`.
    fn repark_young_finalizer_registrations(&mut self) {
        debug_assert!(self.oldgen_nonmoving_active);
        let mut kept = VecDeque::new();
        while let Some((addr, fq_index)) = self.old_objects_with_finalizers.pop_front() {
            if self.is_in_nursery(addr) || self.is_young_rawmalloced(addr) {
                self.probably_young_objects_with_finalizers
                    .push_back((addr, fq_index));
            } else {
                kept.push_back((addr, fq_index));
            }
        }
        self.old_objects_with_finalizers = kept;
    }

    /// incminimark.py `deal_with_young_objects_with_finalizers`: a young object
    /// registered with a finalizer survives the next minor whatever reaches it
    /// ("they all survive"), so upstream's major meets it only once it is on
    /// `old_objects_with_finalizers`, with its children marked by the
    /// finalization-order pass. A non-moving major runs without that minor.
    /// Leaving the young object unmarked would sweep the old objects only it
    /// names, and the minor would then keep it alive with fields pointing at
    /// freed memory. Mark from each one, as the minor will keep it.
    fn nonmoving_major_trace_young_finalizers(&mut self) {
        debug_assert!(self.oldgen_nonmoving_active);
        let young: Vec<usize> = self
            .probably_young_objects_with_finalizers
            .iter()
            .map(|&(addr, _)| addr)
            .collect();
        for addr in young {
            if !self.is_managed_heap_object(addr) {
                continue;
            }
            if unsafe { (*header_of(addr)).has_flag(GcFlags::GCFLAG_IGNORE_FINALIZER) } {
                continue;
            }
            self.seed_major_root(GcRef(addr), "nonmoving_major_young_finalizer");
        }
        self.drain_gray_stack();
    }

    /// A non-moving major runs with a live nursery and no leading minor
    /// (`do_collect_oldgen_nonmoving`), so `rrc_minor_collection_trace` never
    /// ran and the young P list still holds mirrors whose linked nursery object
    /// is reachable from C alone.  Nothing sweeps the nursery here, but an old
    /// object reachable only *through* such a nursery object would be, so seed
    /// the same roots the moving trace would have.  Nothing moves and no entry
    /// changes list: the young bookkeeping stays for the next moving minor.
    ///
    /// The counterpart for weak references is
    /// [`Self::invalidate_young_weakrefs_for_nonmoving_major`].
    fn rrc_nonmoving_major_trace_young(&mut self) {
        debug_assert!(self.oldgen_nonmoving_active);
        let young = std::mem::take(&mut self.rrc.p_list_young);
        for &mirror in &young {
            if !self.rrc_roots_link(mirror) {
                continue;
            }
            let obj = unsafe { (*rawrefcount::pyobj(mirror)).ob_link };
            self.seed_major_root(GcRef(obj), "rrc_nonmoving_major_young");
            self.drain_gray_stack();
        }
        self.rrc.p_list_young = young;
    }

    /// incminimark.py `rrc_major_collection_free`.
    ///
    /// Only the old half is rebuilt.  The nursery-keyed table belongs to the
    /// minor that will next empty it, and every entry it holds names a mirror
    /// still on the young list, which this pass does not walk.
    fn rrc_major_collection_free(&mut self) {
        // incminimark.py:3375 asserts the nursery-keyed table is empty here,
        // which holds upstream because a minor precedes every major step.
        // `do_collect_oldgen_nonmoving` is the one entry that deliberately runs
        // without one — that is its contract — so there the table is legitimately
        // populated.
        debug_assert!(
            self.oldgen_nonmoving_active || self.rrc.p_dict_nurs.is_empty(),
            "p_dict_nurs not empty 2"
        );
        self.rrc.p_dict.clear();
        let mut old = std::mem::take(&mut self.rrc.p_list_old);
        let mut surviving = Vec::with_capacity(old.len());
        while let Some(mirror) = old.pop() {
            self._rrc_major_free(mirror, &mut surviving, RrcList::P);
        }
        self.rrc.p_list_old = surviving;
        let mut old = std::mem::take(&mut self.rrc.o_list_old);
        let mut surviving = Vec::with_capacity(old.len());
        while let Some(mirror) = old.pop() {
            self._rrc_major_free(mirror, &mut surviving, RrcList::O);
        }
        self.rrc.o_list_old = surviving;
    }

    /// incminimark.py `_rrc_major_free`.
    fn _rrc_major_free(&mut self, mirror: usize, surviving: &mut Vec<usize>, list: RrcList) {
        // incminimark.py:3395-3401 — the mirror survives exactly if its object
        // did: VISITED means marking reached it, NO_HEAP_PTRS an immortal
        // object marking never has to reach.
        let obj = unsafe { (*rawrefcount::pyobj(mirror)).ob_link };
        let alive = if !self.is_managed_heap_object(obj) {
            // Upstream reaches the flag test unconditionally because every
            // RPython GC object has a header, prebuilt ones included — which is
            // what its GCFLAG_NO_HEAP_PTRS arm is for.  `malloc_typed` objects
            // here live outside both generations with no header at all, so
            // there is no flag word at that address to read.  Nothing sweeps
            // them either, so nothing can prove one dead.
            true
        } else {
            let hdr = unsafe { header_of(obj) };
            unsafe {
                (*hdr).has_flag(GcFlags::GCFLAG_VISITED)
                    || (*hdr).has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS)
            }
        };
        if alive {
            surviving.push(mirror);
            if list == RrcList::P {
                self.rrc.p_dict.insert(obj, mirror);
            }
        } else {
            self._rrc_free(mirror);
        }
    }

    /// incminimark.py `get_possibly_forwarded_tid`: header access that
    /// follows the forwarding pointer when a young object was already moved
    /// by this collection. A forwarded nursery header holds FORWARDED_MARKER,
    /// whose all-ones flag region fakes every flag (IGNORE_FINALIZER
    /// included), so a raw read there would silently drop a finalizer entry.
    ///
    /// Load-bearing today. Instances and generators are stable-allocated, but
    /// `list_descr_new` takes its header from `w_list_new` — the collecting
    /// *nursery* arm — and then calls `maybe_register_finalizer` for a
    /// builtin-layout subclass, so a `class L(list)` with `__del__` puts a
    /// nursery header on the queue. It gets more so when instance allocation
    /// converges back to the movable nursery (see `alloc_instance_object`).
    fn get_possibly_forwarded_header(&self, obj_addr: usize) -> *const GcHeader {
        let hdr = unsafe { header_of(obj_addr) };
        if self.is_nursery_object_start(obj_addr) && unsafe { (*hdr).is_forwarded() } {
            let fwd_addr = unsafe { GcHeader::forwarding_address(hdr) };
            unsafe { header_of(fwd_addr) }
        } else {
            hdr
        }
    }

    /// incminimark.py `deal_with_young_objects_with_finalizers`.
    fn deal_with_young_objects_with_finalizers(&mut self) {
        while let Some((obj_addr, fq_index)) =
            self.probably_young_objects_with_finalizers.pop_front()
        {
            let current = if self.is_nursery_object_start(obj_addr) {
                // incminimark.py:2918: the IGNORE_FINALIZER read must go
                // through `get_possibly_forwarded_tid` — the object may have
                // been forwarded by an earlier root walk this collection.
                let hdr = self.get_possibly_forwarded_header(obj_addr);
                if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_IGNORE_FINALIZER) } {
                    continue;
                }
                let mut root = GcRef(obj_addr);
                self.drag_out_root(&mut root);
                root.0
            } else {
                if !self.is_managed_heap_object(obj_addr) {
                    continue;
                }
                let hdr = unsafe { header_of(obj_addr) };
                if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_IGNORE_FINALIZER) } {
                    continue;
                }
                // incminimark.py:1838-1841 "they all survive": a registered
                // young finalizer is a root for this collection, and a young
                // rawmalloced one survives by being flagged rather than by
                // being copied out.
                if self.is_young_rawmalloced(obj_addr) {
                    self.visit_young_rawmalloced_object(obj_addr);
                }
                obj_addr
            };
            self.old_objects_with_finalizers
                .push_back((current, fq_index));
        }
    }

    /// incminimark.py `deal_with_young_objects_with_finalizers` drops a
    /// `GCFLAG_IGNORE_FINALIZER` member from the young deque (`continue`), so
    /// nothing keeps it alive and `free_young_rawmalloced_objects` may release
    /// its block. A non-moving major runs without that minor:
    /// [`Self::nonmoving_major_trace_young_finalizers`] leaves such a member
    /// unmarked, and this cycle then releases the block with the entry still
    /// on the deque. The next minor would read whatever was born at that
    /// address and file it as a finalizer object. Drop the young raw-malloced
    /// entries the minor would have dropped, before the blocks go.
    fn drop_ignored_young_rawmalloc_finalizers_nonmoving(&mut self) {
        debug_assert!(self.oldgen_nonmoving_active);
        if self.probably_young_objects_with_finalizers.is_empty() {
            return;
        }
        let queued = std::mem::take(&mut self.probably_young_objects_with_finalizers);
        for (obj_addr, fq_index) in queued {
            if self.is_young_rawmalloced(obj_addr)
                && unsafe { (*header_of(obj_addr)).has_flag(GcFlags::GCFLAG_IGNORE_FINALIZER) }
            {
                continue;
            }
            self.probably_young_objects_with_finalizers
                .push_back((obj_addr, fq_index));
        }
    }

    /// Young rawmallocs a non-moving major did not mark, before their blocks
    /// are released.
    ///
    /// The minor's [`Self::deal_with_young_objects_with_destructors`] reads
    /// `GCFLAG_VISITED_RMY` and promotes a survivor onto the old list. This
    /// cycle does not promote: a `GCFLAG_VISITED` survivor stays young, and
    /// its queue entry stays on the young list. Nursery entries are not this
    /// cycle's to run.
    fn deal_with_young_rawmalloc_destructors_nonmoving(&mut self) {
        debug_assert!(self.oldgen_nonmoving_active);
        if self.young_objects_with_destructors.is_empty() {
            return;
        }
        let queued = std::mem::take(&mut self.young_objects_with_destructors);
        for obj_addr in queued {
            if !self.is_young_rawmalloced(obj_addr) {
                self.young_objects_with_destructors.push(obj_addr);
                continue;
            }
            if unsafe { (*header_of(obj_addr)).has_flag(GcFlags::GCFLAG_VISITED) } {
                self.young_objects_with_destructors.push(obj_addr);
            } else {
                self.run_destructor(obj_addr);
            }
        }
    }

    /// Mirrors whose link is a young rawmalloc this non-moving major did not
    /// mark.
    ///
    /// [`Self::rrc_major_collection_free`] walks only the old lists and
    /// rebuilds `p_dict` from them. The young lists still name the block
    /// `free_unvisited_young_rawmalloced_keep_young` is about to release.
    /// `_rrc_minor_free` recognises that address only while the
    /// young-rawmalloc record exists, so the mirror is settled here: a
    /// `GCFLAG_VISITED` target stays young and keeps its list entry, and a
    /// young rawmalloc P link is keyed in `p_dict` again. A nursery link
    /// stays in `p_dict_nurs`. A dead one is unlinked the way
    /// `_rrc_minor_free` unlinks it.
    fn rrc_nonmoving_reclaim_young_rawmalloc(&mut self) {
        debug_assert!(self.oldgen_nonmoving_active);
        if !self.rrc.enabled {
            return;
        }
        for which in [RrcList::P, RrcList::O] {
            let list = match which {
                RrcList::P => std::mem::take(&mut self.rrc.p_list_young),
                RrcList::O => std::mem::take(&mut self.rrc.o_list_young),
            };
            let mut still_young = Vec::with_capacity(list.len());
            for mirror in list {
                let obj = unsafe { (*rawrefcount::pyobj(mirror)).ob_link };
                let young_raw = self.is_young_rawmalloced(obj);
                let dead_young_raw =
                    young_raw && unsafe { !(*header_of(obj)).has_flag(GcFlags::GCFLAG_VISITED) };
                if dead_young_raw {
                    if which == RrcList::P {
                        self.rrc.p_dict.remove(&obj);
                    }
                    self._rrc_free(mirror);
                } else {
                    still_young.push(mirror);
                    if which == RrcList::P && young_raw {
                        self.rrc.p_dict.insert(obj, mirror);
                    }
                }
            }
            match which {
                RrcList::P => self.rrc.p_list_young = still_young,
                RrcList::O => self.rrc.o_list_young = still_young,
            }
        }
    }

    /// incminimark.py `deal_with_young_objects_with_destructors`.
    ///
    /// "We can reasonably assume that destructors don't do anything fancy
    /// and *just* call them. Among other things they won't resurrect
    /// objects." For each recorded young object: if it was not forwarded
    /// out it died — run its destructor; otherwise it survived — move it
    /// (translated to its new old-gen address) to the old-destructor list
    /// so the next major collection can reclaim it.
    fn deal_with_young_objects_with_destructors(&mut self) {
        while let Some(obj_addr) = self.young_objects_with_destructors.pop() {
            // A young rawmalloced member reaches here because
            // `malloc_fixedsize`'s large arm falls into the same registration
            // tail. It never forwards, so the forwarding read below is not the
            // question to ask of it — `is_forwarded` states as much, opening
            // with `ll_assert(self.is_in_nursery(obj), "Can't forward an object
            // outside the nursery.")`. The witness the minor left for this kind
            // is GCFLAG_VISITED_RMY, already final at this point: both
            // `collect_roots_in_nursery` and the `collect_oldrefs_to_nursery`
            // drain run before this call, and
            // `free_young_rawmalloced_objects` reads the same flag a few steps
            // after it.
            if self.is_young_rawmalloced(obj_addr) {
                if unsafe { (*header_of(obj_addr)).has_flag(GcFlags::GCFLAG_VISITED_RMY) } {
                    // Promoted in place: the address does not change, so the
                    // old-generation list takes it unchanged.
                    self.old_objects_with_destructors.push(obj_addr);
                } else {
                    // Dead: run the destructor before
                    // `free_young_rawmalloced_objects` releases the block.
                    self.run_destructor(obj_addr);
                }
                continue;
            }
            let hdr_ptr = (obj_addr - GcHeader::SIZE) as *const GcHeader;
            if !unsafe { (*hdr_ptr).is_forwarded() } {
                // Dead: run the destructor before the nursery reset frees
                // the bytes.
                self.run_destructor(obj_addr);
            } else {
                // Surviving: track the promoted copy for the major cycle.
                let new_obj = unsafe { GcHeader::forwarding_address(hdr_ptr) };
                self.old_objects_with_destructors.push(new_obj);
            }
        }
    }

    /// minimark.py `_allocate_shadow(obj)`.
    /// Allocate a shadow copy in old-gen for a nursery object.  The
    /// shadow's tid is copied from the nursery object; for varsize
    /// objects the length field is also copied.  The nursery header
    /// gets GCFLAG_HAS_SHADOW so `_find_shadow` knows to look it up.
    fn allocate_shadow(&mut self, obj_addr: usize) -> usize {
        let hdr_ptr = (obj_addr - GcHeader::SIZE) as *const GcHeader;
        let type_id = unsafe { (*hdr_ptr).type_id() };
        self.validate_type_id(type_id, obj_addr, "allocate_shadow");
        let payload_size = self.size_for_typeid(obj_addr, type_id, "allocate_shadow");
        let total_size = GcHeader::SIZE + payload_size;
        let (item_size, length_offset) = {
            let type_info = self.types.get(type_id);
            (type_info.item_size, type_info.length_offset)
        };
        let shadow_hdr_ptr = self.oldgen.alloc(total_size);
        let shadow_obj = shadow_hdr_ptr as usize + GcHeader::SIZE;
        unsafe {
            (*(shadow_hdr_ptr as *mut GcHeader)).tid_and_flags = (*hdr_ptr).tid_and_flags;
            // `_allocate_shadow` copies the full tid and its XXX notes that a
            // pinned source leaves GCFLAG_PINNED (== PINNED_OBJECT_PARENT_KNOWN)
            // on the old-gen shadow. Upstream's reachable-only debug walk never
            // meets an unreferenced shadow; the MARKING all-allocated walk in
            // `debug_check_consistency_at` does, and asserts PINNED outside the
            // nursery. The bit means nothing on the shadow: the copy at the
            // next minor overwrites the whole header.
            (*(shadow_hdr_ptr as *mut GcHeader)).clear_flag(GcFlags::GCFLAG_PINNED);
            // The shadow lives in old-gen, so it carries TRACK_YOUNG_PTRS like
            // every other old-gen object — the write barrier tracks it through
            // that flag. (The source header was copied from the nursery object,
            // which does not carry it.)
            (*(shadow_hdr_ptr as *mut GcHeader)).set_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
            // A shadow reserved during marking is an old-gen object subject to
            // this cycle's future sweep; keep it black so marking does not
            // reclaim the reserved identity home before the object is copied in
            // (incminimark.py record_pinned_object_with_shadow). See
            // `oldgen_birth_flags`.
            if self.gc_state == GcState::Marking {
                (*(shadow_hdr_ptr as *mut GcHeader)).set_flag(GcFlags::GCFLAG_VISITED);
            }
            if item_size > 0 {
                // `find_shadow` has already rejected a forwarded header.
                // `is_forwarded` is nursery-only, and this object is still
                // the uncopied nursery original, so the length word is its own.
                *((shadow_obj + length_offset) as *mut usize) =
                    *((obj_addr + length_offset) as *const usize);
            }
            let nursery_hdr = (obj_addr - GcHeader::SIZE) as *mut GcHeader;
            (*nursery_hdr).set_flag(GcFlags::GCFLAG_HAS_SHADOW);
        }
        self.nursery_objects_shadows.insert(obj_addr, shadow_obj);
        shadow_obj
    }

    /// minimark.py `_find_shadow(obj)`.
    /// Return the shadow address for a nursery object, allocating one
    /// if this is the first request.
    fn find_shadow(&mut self, obj_addr: usize) -> usize {
        let hdr = unsafe { *((obj_addr - GcHeader::SIZE) as *const GcHeader) };
        // A forwarded header is `FORWARDED_MARKER`, whose every flag bit reads
        // set, so it would answer the test below and then die on the map lookup
        // under the shadow message.  `_find_shadow`'s precondition is that the
        // object has not been copied yet, so name the real fault here.
        assert!(
            !hdr.is_forwarded(),
            "stale pointer into the nursery: find_shadow reached a forwarded header at {obj_addr:#x}"
        );
        if hdr.has_flag(GcFlags::GCFLAG_HAS_SHADOW) {
            // incminimark.py:2855-2857 `ll_assert(shadow != NULL,
            // "GCFLAG_HAS_SHADOW but no shadow found")`.  HAS_SHADOW
            // guarantees the map holds the shadow; a missing entry is a GC
            // invariant violation, not a cue to silently allocate a second
            // shadow (which would hand out an unstable identity address).
            return self
                .nursery_objects_shadows
                .get(&obj_addr)
                .copied()
                .expect("GCFLAG_HAS_SHADOW but no shadow found");
        }
        self.allocate_shadow(obj_addr)
    }

    /// incminimark.py `move_out_of_nursery`: "called twice, it should return
    /// the same shadow object, and not creating another shadow object.  As a
    /// safety feature, when called on a non-nursery object, do nothing."
    ///
    /// `rgc.move_out_of_nursery`'s contract is what makes the eager copy sound:
    /// "Only use for immutable objects!"  The shadow is filled once and the
    /// nursery original may still be written to afterwards, so a mutable object
    /// would have the two copies disagree until the next minor.
    pub fn move_out_of_nursery(&mut self, obj_addr: usize) -> usize {
        if !self.is_in_nursery(obj_addr) {
            return obj_addr;
        }
        let shadow = self.find_shadow(obj_addr);
        let hdr = unsafe { header_of(obj_addr) };
        if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_SHADOW_INITIALIZED) } {
            return shadow;
        }
        unsafe { (*hdr).set_flag(GcFlags::GCFLAG_SHADOW_INITIALIZED) };
        let type_id = unsafe { (*hdr).type_id() };
        let payload_size = self.size_for_typeid(obj_addr, type_id, "move_out_of_nursery");
        // Payload only: `allocate_shadow` already wrote the shadow's own header
        // and the flags the old generation needs, and the copy must not undo
        // them.  The next minor reads SHADOW_INITIALIZED and skips its own copy
        // for the same reason.
        unsafe {
            std::ptr::copy_nonoverlapping(obj_addr as *const u8, shadow as *mut u8, payload_size)
        };
        shadow
    }

    /// minimark.py `id_or_identityhash(gcobj)`.
    /// Return a stable address usable as identity hash.  For nursery
    /// objects, returns the shadow's address (which is where the object
    /// will be copied to at the next minor collection).  For old-gen
    /// objects, returns the object's own address (old-gen objects don't
    /// move in mark-sweep).
    pub fn id_or_identityhash(&mut self, obj_addr: usize) -> usize {
        if self.is_valid_gc_object(obj_addr) && self.is_in_nursery(obj_addr) {
            return self.find_shadow(obj_addr);
        }
        obj_addr
    }

    /// Copy a single nursery object to old gen.
    /// If already forwarded, returns the forwarding address.
    /// Pinned objects are left in place and returned as-is.
    /// Abort with a diagnostic if `type_id` is out of range for the
    /// registered type table. The raw header read that produces it is
    /// layout/ASLR-dependent, so an out-of-range id must fail
    /// deterministically at the trace site with context rather than as
    /// a bare `entries[..]` index panic deep inside `types.get`.
    #[inline]
    fn validate_type_id(&self, type_id: u32, obj_addr: usize, site: &str) {
        if type_id as usize >= self.types.len() {
            self.report_invalid_type_id(type_id, obj_addr, site);
        }
    }

    /// Out-of-line half of [`validate_type_id`](Self::validate_type_id).
    ///
    /// A type id past the end of the type table has two very different
    /// causes, and the bare number does not separate them.  When the header
    /// word is `FORWARDED_MARKER` the object is not mistyped at all: it is a
    /// nursery corpse whose live copy sits at the forwarding address, and the
    /// caller reached it through a field nothing updated.  Say so, and give
    /// the copy plus the nursery geometry and collection cadence, so the
    /// surviving question is only which holder kept the stale pointer.
    #[cold]
    #[inline(never)]
    fn report_invalid_type_id(&self, type_id: u32, obj_addr: usize, site: &str) -> ! {
        let header_addr = obj_addr - GcHeader::SIZE;
        let nursery_start = self.nursery.start_ptr() as usize;
        let nursery_end = nursery_start + self.nursery.size();
        let in_nursery = (nursery_start..nursery_end).contains(&obj_addr);
        let header = unsafe { *(header_addr as *const GcHeader) };
        if header.is_forwarded() {
            let moved_to = unsafe { GcHeader::forwarding_address(header_addr as *const GcHeader) };
            let moved_to_type_id = self
                .is_managed_heap_object(moved_to)
                .then(|| unsafe { (*header_of(moved_to)).type_id() });
            panic!(
                "GC BUG: {site} was handed {obj_addr:#x}, a STALE address whose object a                  minor collection already moved to {moved_to:#x}                  (live copy type_id={moved_to_type_id:?}).                  nursery=[{nursery_start:#x}, {nursery_end:#x}) obj_in_nursery={in_nursery}                  offset_from_nursery_start={:#x} minor_collections={} major_collections={}.                  FORWARDED_MARKER sets every flag bit, so an inline TRACK_YOUNG_PTRS test                  cannot tell this corpse from a live old object — the defect is the holder                  that kept the pre-collection pointer, not this barrier.",
                obj_addr.wrapping_sub(nursery_start),
                self.minor_collections,
                self.major_collections,
            );
        }
        panic!(
            "GC BUG: invalid type_id={} at obj_addr={:#x} (header_addr={:#x}, nursery_start={:#x}, site={})              header_word={:#x} obj_in_nursery={} minor_collections={} major_collections={}",
            type_id,
            obj_addr,
            header_addr,
            nursery_start,
            site,
            header.tid_and_flags,
            in_nursery,
            self.minor_collections,
            self.major_collections,
        );
    }

    /// `get_possibly_forwarded_type_id`: follow a nursery corpse before
    /// reading its size. `set_forwarding_address` writes the live address
    /// at payload offset 0, which is `length_offset` for a varsize type
    /// whose length is its first field.
    ///
    /// `None` when the header is forwarded but the copy is not a live
    /// managed object. An object outside the nursery is never forwarded
    /// (`is_forwarded`).
    fn resolve_possibly_forwarded(&self, obj_addr: usize, type_id: u32) -> Option<(usize, u32)> {
        if !self.is_in_nursery(obj_addr) {
            return Some((obj_addr, type_id));
        }
        let hdr = unsafe { *header_of(obj_addr) };
        if !hdr.is_forwarded() {
            return Some((obj_addr, type_id));
        }
        let live = unsafe { GcHeader::forwarding_address(header_of(obj_addr)) };
        if !self.is_managed_heap_object(live) {
            return None;
        }
        let live_hdr = unsafe { *header_of(live) };
        if live_hdr.is_forwarded() {
            return None;
        }
        Some((live, live_hdr.type_id()))
    }

    /// `base.py _get_size_for_typeid` — the payload size of `obj_addr`,
    /// reading the length field when the type is varsize. `None` when the
    /// length cannot describe an allocation.
    ///
    /// The length is read from the object `get_possibly_forwarded_type_id`
    /// would hand to `get_size`. `_get_size_for_typeid` itself still loads
    /// the word raw; the forward step is the caller's, done here because
    /// this function is that caller for every size query.
    ///
    /// Upstream rounds the result here. Pyre's callers each apply their own
    /// rounding (nursery geometry, arena minimum, inspector alignment), so the
    /// rounding stays at the call sites.
    fn try_size_for_typeid(&self, obj_addr: usize, type_id: u32) -> Option<usize> {
        let (obj_addr, type_id) = self.resolve_possibly_forwarded(obj_addr, type_id)?;
        if (type_id as usize) >= self.types.len() {
            return None;
        }
        let type_info = self.types.get(type_id);
        if type_info.item_size == 0 {
            return Some(type_info.size);
        }
        let length = unsafe { *((obj_addr + type_info.length_offset) as *const usize) };
        type_info
            .item_size
            .checked_mul(length)
            .and_then(|items| type_info.size.checked_add(items))
            // No object can be larger than `isize::MAX` — `Layout` refuses to
            // describe one — so a larger result is a decode failure, not a
            // request the allocator could ever serve.
            .filter(|&size| size <= isize::MAX as usize - GcHeader::SIZE)
    }

    /// incminimark.py `shrink_array`.
    ///
    /// Records that a varsize object is shorter than it was, so that when it
    /// leaves the nursery it consumes less memory. The object keeps its
    /// address; the tail is simply never copied out, and the nursery reclaims
    /// it wholesale at the next reset the way it reclaims anything unreached.
    ///
    /// Writing the length field is the whole of the operation. Upstream also
    /// calls `llarena.arena_shrink_obj`, but that is `pass` in a translated
    /// build (`llarena.py llimpl_arena_shrink_obj`) — it exists for the
    /// fake arena's own reservation bookkeeping, which pyre's bump-pointer
    /// nursery has no counterpart to. Nothing walks the nursery linearly, so
    /// no reader has to be told about the gap.
    ///
    /// Declined for an object outside the nursery — upstream's comment names
    /// the consequence, "an array with GCFLAG_HAS_CARDS is never resized", and
    /// a carded array is rawmalloced and so already excluded — and for one
    /// with a shadow, whose shadow was allocated at the old length and would
    /// be partly stranded.
    ///
    /// `false` means the caller must allocate a smaller object and copy, which
    /// is exactly what `rgc.ll_shrink_array` does with that answer
    /// (`rgc.py`).
    pub fn shrink_array(&mut self, obj_addr: usize, smaller_length: usize) -> bool {
        if !self.is_in_nursery(obj_addr) {
            return false;
        }
        let hdr = unsafe { header_of(obj_addr) };
        if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_HAS_SHADOW) } {
            return false;
        }
        let type_id = unsafe { (*hdr).type_id() };
        if (type_id as usize) >= self.types.len() {
            return false;
        }
        let type_info = self.types.get(type_id);
        // `varsize_offset_to_length` is asked of a varsize type only; a fixed
        // one has no length field to write, and `length_offset` on it is the
        // registration default rather than an answer.
        if type_info.item_size == 0 {
            return false;
        }
        // A forwarded header is `FORWARDED_MARKER`, so every flag bit reads
        // set and the `GCFLAG_HAS_SHADOW` test above already returned. This
        // write therefore addresses the nursery original, not a corpse.
        let length_ptr = (obj_addr + type_info.length_offset) as *mut usize;
        assert!(
            smaller_length <= unsafe { *length_ptr },
            "shrink_array asked to grow an array: {smaller_length} > {}",
            unsafe { *length_ptr }
        );
        unsafe { *length_ptr = smaller_length };
        true
    }

    /// Panicking [`Self::try_size_for_typeid`], for the collector paths that
    /// are about to allocate or copy that many bytes.
    ///
    /// A varsize length is read straight out of the object, so a collector that
    /// reaches an object before its length field is initialized computes a size
    /// that describes nothing. Report the inputs here: downstream the allocator
    /// sees only the product, and fails on a `Layout` it cannot even build.
    fn size_for_typeid(&self, obj_addr: usize, type_id: u32, site: &str) -> usize {
        match self.try_size_for_typeid(obj_addr, type_id) {
            Some(size) => size,
            None => {
                // Same address `try_size_for_typeid` sized. A corpse that
                // resolved contributes the copy's length; one that did not
                // still reports the word at the caller's address.
                let (read_addr, read_tid) = self
                    .resolve_possibly_forwarded(obj_addr, type_id)
                    .unwrap_or((obj_addr, type_id));
                let type_info = self.types.get(if (read_tid as usize) < self.types.len() {
                    read_tid
                } else {
                    type_id
                });
                let length_addr = if (read_tid as usize) < self.types.len() {
                    read_addr
                } else {
                    obj_addr
                };
                let length = unsafe { *((length_addr + type_info.length_offset) as *const usize) };
                // On 64-bit, `set_forwarding_address` stores the live copy in
                // the first payload word (`obj_addr + 0`). Types with
                // `length_offset == 0` (`ItemsBlock.capacity`) then look like
                // a huge length if `is_forwarded` was skipped. On wasm32 the
                // live copy sits in the header padding, so a marker length
                // is a payload that was itself used as a header.
                let forwarded = unsafe { (*header_of(obj_addr)).is_forwarded() };
                panic!(
                    "GC BUG: varsize length describes no allocation: length={} (read at \
                     obj_addr={:#x} + length_offset={}) item_size={} fixed_size={} \
                     type_id={} header_addr={:#x} nursery_start={:#x} forwarded={} site={}",
                    length,
                    obj_addr,
                    type_info.length_offset,
                    type_info.item_size,
                    type_info.size,
                    type_id,
                    obj_addr - GcHeader::SIZE,
                    self.nursery.start_ptr() as usize,
                    forwarded,
                    site,
                );
            }
        }
    }

    fn copy_nursery_object(
        &mut self,
        obj_addr: usize,
        site: &str,
        // The child site names the slot kind; the root path identifies the producer.
        parent_site: &'static str,
        holder_addr: usize,
        slot_addr: usize,
    ) -> GcRef {
        // Keep the header access as a raw pointer rather than `&mut GcHeader`.
        // The later `alloc_and_copy` performs a raw read over this same byte
        // range, so re-materialize references only for scoped accesses.
        //
        // The header must sit inside this nursery. A payload at
        // `nursery_start` (or a host pointer that only *looks* in-range)
        // would write `obj_addr - SIZE` into the mapping before the
        // arena. RPython's `header(obj)` is only ever asked of a typed
        // GC payload.
        let start = self.nursery.start_ptr() as usize;
        if obj_addr < start.saturating_add(GcHeader::SIZE) || !self.is_in_nursery(obj_addr) {
            return GcRef(obj_addr);
        }
        let hdr_ptr = (obj_addr - GcHeader::SIZE) as *mut GcHeader;

        // `_trace_drag_out` tests forwarding before PINNED: the forwarding
        // marker has every flag bit set and must not be mistaken for a pin.
        if unsafe { (*hdr_ptr).is_forwarded() } {
            let fwd_addr = unsafe { GcHeader::forwarding_address(hdr_ptr) };
            return GcRef(fwd_addr);
        }

        // `IncrementalMiniMarkGC._trace_drag_out`: pin state lives on the
        // object header. Pinning is not a root; append the header address to
        // the survivor AddressStack only when a real traced edge reaches it.
        if unsafe { (*hdr_ptr).has_flag(GcFlags::GCFLAG_PINNED) } {
            // incminimark.py:2194-2199 records an old parent. The list outlives
            // the nursery reset at the end of this minor and is dereferenced in
            // the next one, so a nursery holder would become a read of recycled
            // nursery bytes; `is_managed_heap_object` alone accepts one.
            if holder_addr != 0
                && self.is_managed_heap_object(holder_addr)
                && !self.is_in_nursery(holder_addr)
            {
                let holder_hdr = unsafe { header_of(holder_addr) };
                if unsafe { !(*holder_hdr).has_flag(GcFlags::GCFLAG_PINNED_OBJECT_PARENT_KNOWN) } {
                    self.old_objects_pointing_to_pinned.push(holder_addr);
                    self.updated_old_objects_pointing_to_pinned = true;
                    unsafe { (*holder_hdr).set_flag(GcFlags::GCFLAG_PINNED_OBJECT_PARENT_KNOWN) };
                }
            }
            if unsafe { (*hdr_ptr).has_flag(GcFlags::GCFLAG_VISITED) } {
                return GcRef(obj_addr);
            }
            unsafe { (*hdr_ptr).set_flag(GcFlags::GCFLAG_VISITED) };
            self.surviving_pinned_objects
                .push(obj_addr - GcHeader::SIZE);
            self.pinned_objects_in_nursery += 1;
            self.any_pinned_object_kept = true;
            return GcRef(obj_addr);
        }

        let type_id = unsafe { (*hdr_ptr).type_id() };
        if type_id as usize >= self.types.len() {
            let holder_type_id = if holder_addr == 0 {
                None
            } else {
                Some(unsafe { (*header_of(holder_addr)).type_id() })
            };
            let mut holder_words = [0usize; 8];
            if holder_addr != 0 {
                for (index, word) in holder_words.iter_mut().enumerate() {
                    *word = unsafe { *((holder_addr as *const usize).add(index)) };
                }
            }
            let mut child_words = [0usize; 8];
            for (index, word) in child_words.iter_mut().enumerate() {
                *word = unsafe { *((obj_addr as *const usize).add(index)) };
            }
            let holder_hdr_tid_and_flags = if holder_addr == 0 {
                0
            } else {
                unsafe { (*header_of(holder_addr)).tid_and_flags }
            };
            panic!(
                "GC BUG: invalid type_id={} at obj_addr={:#x} \
                 (minor={}, header_addr={:#x}, nursery_start={:#x}, site={}, \
                 parent_site={}, \
                 nursery_free={:#x}, nursery_top={:#x}, holder_addr={:#x}, \
                 holder_type_id={:?}, holder_offset={:?}, holder_words={}, \
                 child_tid_and_flags={:#x}, child_flag_complement={:#x}, \
                 child_words={}, child_vtable_type_id={:?}, \
                 nearest_header={}, \
                 child_nursery_offset={:#x}, child_gen={}, holder_gen={}, \
                 holder_tid_and_flags={:#x}, holder_in_remembered={}, \
                 enclosing={}, extra_area={}, gc_state={:?}, minors={}, majors={})",
                type_id,
                obj_addr,
                self.minor_collections,
                obj_addr - GcHeader::SIZE,
                self.nursery.start_ptr() as usize,
                site,
                parent_site,
                self.nursery.free_ptr() as usize,
                self.nursery.top_ptr() as usize,
                holder_addr,
                holder_type_id,
                slot_addr.checked_sub(holder_addr),
                Self::hex_words(&holder_words),
                unsafe { (*hdr_ptr).tid_and_flags },
                // FORWARDED_MARKER sets every flag bit, so a corpse whose
                // 64-bit equality no longer holds names the cleared bit here.
                (!unsafe { (*hdr_ptr).flags() }.bits()) as u32,
                Self::hex_words(&child_words),
                self.vtable_to_type_id.get(&child_words[0]).copied(),
                self.describe_nearest_header(obj_addr),
                obj_addr.wrapping_sub(self.nursery.start_ptr() as usize),
                self.describe_generation(obj_addr),
                self.describe_generation(holder_addr),
                holder_hdr_tid_and_flags,
                self.old_objects_pointing_to_young.contains(&holder_addr),
                self.describe_enclosing_container(holder_addr, slot_addr, &holder_words),
                crate::shadow_stack::current_extra_area(),
                self.gc_state,
                self.minor_collections,
                self.major_collections,
            );
        }
        // Compute the actual payload size (for varsize objects, read the length).
        let actual_payload_size = self.size_for_typeid(obj_addr, type_id, site);

        let total_size = GcHeader::SIZE + actual_payload_size;
        let has_gc_ptrs = self.types.get(type_id).has_gc_ptrs;

        // minimark.py:1513-1519: if the object has a pre-allocated
        // shadow (from id() or identityhash()), copy into it instead
        // of a fresh allocation.
        let header_ptr = obj_addr - GcHeader::SIZE;
        let new_header_ptr = if unsafe { (*hdr_ptr).has_flag(GcFlags::GCFLAG_HAS_SHADOW) } {
            // incminimark.py:2213-2214 `ll_assert(newobj != NULL,
            // "GCFLAG_HAS_SHADOW but no shadow found")` — HAS_SHADOW
            // guarantees the shadow map entry exists.
            let shadow_obj = *self
                .nursery_objects_shadows
                .get(&obj_addr)
                .expect("GCFLAG_HAS_SHADOW but no shadow found");
            let shadow_hdr = (shadow_obj - GcHeader::SIZE) as *mut u8;
            // incminimark.py:2215-2221: if the shadow is already black
            // (VISITED), the memcpy below overwrites its flags with the
            // nursery object's (white) flags, so a sweep running during this
            // incremental cycle could reclaim it (see test_pin_id_bug).
            // Remember VISITED and re-apply it after the copy.
            let shadow_was_visited =
                unsafe { (*(shadow_hdr as *const GcHeader)).has_flag(GcFlags::GCFLAG_VISITED) };
            // incminimark.py `_trace_drag_out`: `if (tid &
            // GCFLAG_SHADOW_INITIALIZED) != 0: copy = False`.
            // `move_out_of_nursery` has already filled this shadow, and its
            // caller has been reading it ever since.  Copying again is not
            // merely redundant: HAS_SHADOW and SHADOW_INITIALIZED "have no
            // meaning in non-nursery objects", so the second copy is what would
            // carry the nursery flag word onto an object that has the right one.
            let shadow_initialized =
                unsafe { (*hdr_ptr).has_flag(GcFlags::GCFLAG_SHADOW_INITIALIZED) };
            // minimark.py:1518-1520: clear HAS_SHADOW before copy so the
            // flag does not propagate to the shadow object itself.
            unsafe { (*hdr_ptr).clear_flag(GcFlags::GCFLAG_HAS_SHADOW) };
            if !shadow_initialized {
                unsafe {
                    std::ptr::copy_nonoverlapping(header_ptr as *const u8, shadow_hdr, total_size);
                }
            }
            if shadow_was_visited {
                unsafe { (*(shadow_hdr as *mut GcHeader)).set_flag(GcFlags::GCFLAG_VISITED) };
                // The copy above replaced every field of an object the marker
                // has already accounted for as black, and the incoming values
                // are this cycle's first sight of them: the shadow was reserved
                // black before anything was copied into it (`allocate_shadow`),
                // so nothing has ever traced these slots.  A black object is
                // never pushed again by `grey_child`, so without this its
                // children stay white and the sweep frees them while the object
                // still points at them.  Re-greying a modified black object is
                // what `_add_to_more_objects_to_trace` (incminimark.py)
                // does for every other mid-cycle mutation.
                if self.gc_state == GcState::Marking {
                    self.incr_state.more_gray_stack.push(shadow_obj);
                }
            }
            shadow_hdr
        } else {
            unsafe {
                self.oldgen
                    .alloc_and_copy(header_ptr as *const u8, total_size)
            }
        };
        let new_obj_addr = new_header_ptr as usize + GcHeader::SIZE;
        if crate::gc_lifetime_log_enabled() {
            eprintln!(
                "[gc][alloc] addr={new_obj_addr:#x} type_id={type_id} kind=promotion state={:?}",
                self.gc_state
            );
        }
        crate::note_bh_object(
            new_obj_addr,
            total_size - GcHeader::SIZE,
            crate::BH_PROBE_ORIGIN_PROMOTED,
        );
        self.bytes_made_old_since_cycle =
            self.bytes_made_old_since_cycle.saturating_add(total_size);
        self.nursery_surviving_size = self.nursery_surviving_size.saturating_add(total_size);

        // Set TRACK_YOUNG_PTRS on the new old-gen object. (The source header was
        // copied from the nursery object, which does not carry it.)
        unsafe {
            let h = new_header_ptr as *mut GcHeader;
            (*h).set_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
        }

        // Install forwarding pointer in the nursery copy.
        unsafe {
            GcHeader::set_forwarding_address(hdr_ptr, new_obj_addr);
        }

        // If this object has GC pointers, add it to the work list so we
        // trace its fields and update nursery references.
        if has_gc_ptrs {
            // Clear TRACK_YOUNG_PTRS temporarily so the processing loop
            // can re-set it after processing.
            unsafe {
                (*(new_header_ptr as *mut GcHeader)).clear_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
            }
            self.old_objects_pointing_to_young.push(new_obj_addr);
            if crate::gc_lifetime_log_enabled() {
                eprintln!(
                    "[gc][remember] addr={:#x} type_id={} source=promotion state={:?}",
                    new_obj_addr, type_id, self.gc_state
                );
            }
        }

        GcRef(new_obj_addr)
    }

    /// llarena debug-fill parity: a GC-visible slot containing the nursery
    /// poison is an uninitialized reference, not an unmanaged pointer to skip.
    /// Check before nursery-range filtering so poison mode fails closed at the
    /// trace site and identifies the exact holder and slot.
    #[inline]
    fn assert_traced_slot_initialized(
        &self,
        field_ref: GcRef,
        slot_addr: usize,
        holder_addr: usize,
        site: &str,
        // The child site names the slot kind; the root path identifies the producer.
        parent_site: &'static str,
    ) {
        const NURSERY_POISON_WORD: usize = (usize::MAX / 0xff) * 0xaa;
        if self.nursery.poison_enabled() && field_ref.0 == NURSERY_POISON_WORD {
            let holder_type_id = if holder_addr == 0 {
                None
            } else {
                Some(unsafe { (*header_of(holder_addr)).type_id() })
            };
            let holder_offset = slot_addr.checked_sub(holder_addr);
            panic!(
                "GC BUG: traced slot contains nursery poison at slot_addr={:#x} holder_addr={:#x} holder_type_id={:?} holder_offset={:?} site={} parent_site={}",
                slot_addr, holder_addr, holder_type_id, holder_offset, site, parent_site,
            );
        }
    }

    /// incminimark.py `_trace_drag_out` + :2128-2143
    /// `_trace_drag_out1_marking_phase`, for a nursery object reached
    /// through a *root* slot during minor collection.
    ///
    /// Copies a nursery object out (updating `*gcref`) and, while major
    /// marking is active, greys any white root after that step — including an
    /// already-old object. This is the literal two-part shape of
    /// `_trace_drag_out1_marking_phase`: `_trace_drag_out(root, NULL)` first,
    /// then append `root.address[0]` iff it lacks VISITED/PINNED. The old-root
    /// half matters because stack roots can change after `collect_roots()`'s
    /// initial snapshot; every intervening minor reintroduces newly exposed
    /// white old objects to `more_objects_to_trace`.
    #[inline]
    fn drag_out_root(&mut self, gcref: &mut GcRef) {
        self.assert_traced_slot_initialized(
            *gcref,
            gcref as *mut GcRef as usize,
            0,
            "minor_root",
            "minor_root",
        );
        if self.is_nursery_object_start(gcref.0) {
            let slot_addr = gcref as *mut GcRef as usize;
            // incminimark.py `_trace_drag_out`: `is_in_nursery` is a range
            // check, then copy `root.address[0]`. Interiors are a publisher
            // defect, not a walk-site repair.
            *gcref =
                self.copy_nursery_object(gcref.0, "minor_root_target", "minor_root", 0, slot_addr);
        } else if self.is_young_rawmalloced(gcref.0) {
            // incminimark.py `_trace_drag_out`: an object outside the
            // nursery needs nothing changed, *except* that a young rawmalloced
            // one must be flagged so the end of this collection promotes rather
            // than frees it.
            self.visit_young_rawmalloced_object(gcref.0);
        }
        // `_trace_drag_out1_marking_phase`: inspect the live object's header
        // only while major marking is active, and append iff neither VISITED
        // nor PINNED is set.  In particular, the common scanning-state root
        // walk performs no address-table lookup at all.
        if self.gc_state == GcState::Marking && self.is_managed_heap_object(gcref.0) {
            let hdr = unsafe { header_of(gcref.0) };
            if unsafe {
                !(*hdr).has_flag(GcFlags::GCFLAG_VISITED)
                    && !(*hdr).has_flag(GcFlags::GCFLAG_PINNED)
            } {
                unsafe { (*hdr).set_flag(GcFlags::GCFLAG_VISITED) };
                self.incr_state.more_gray_stack.push(gcref.0);
                self.note_nonmoving_young_mark(gcref.0, false);
            }
        }
    }

    /// Trace an object's GC pointer fields and update any that point
    /// into the nursery by copying the target.
    fn trace_and_update_object(&mut self, obj_addr: usize, site: &'static str) {
        crate::bh_probe_note_traced(obj_addr);
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        self.validate_type_id(type_id, obj_addr, site);
        let custom_trace = self.types.get(type_id).custom_trace;

        // `GCBase.trace` / `_trace_slow_path`: a custom trace adds slots.
        // It does not replace the fixed `offsets_to_gc_pointers` walk or
        // the variable-part items below.
        if let Some(trace_fn) = custom_trace {
            // A custom trace names its own slots — for a JITFRAME, `jf_gcmap`
            // decides them, not the type table. When one of those slots does
            // not decode as an object, the discriminating question is whether
            // the *rest* of the same trace is sound: one bad slot among sound
            // ones is a bad published value, while a trace whose slots are
            // mostly unsound is a map that no longer describes the object.
            // Defer the first undecodable slot so the whole walk completes and
            // the panic can report both.
            let mut deferred: Option<(usize, usize)> = None;
            unsafe {
                trace_fn(obj_addr, &mut |slot_ptr: *mut GcRef| {
                    let field_ref = *slot_ptr;
                    self.assert_traced_slot_initialized(
                        field_ref,
                        slot_ptr as usize,
                        obj_addr,
                        "minor_custom_trace",
                        site,
                    );
                    if self.is_nursery_object_start(field_ref.0) {
                        // `_trace_drag_out` copies the slot value. An
                        // undecodable interior is deferred so the rest of
                        // this custom trace can be reported with it.
                        if self.nursery_start_decodes(field_ref.0) {
                            let new_ref = self.copy_nursery_object(
                                field_ref.0,
                                "minor_custom_trace_target",
                                site,
                                obj_addr,
                                slot_ptr as usize,
                            );
                            *slot_ptr = new_ref;
                            return;
                        }
                        if deferred.is_none() {
                            deferred = Some((slot_ptr as usize, field_ref.0));
                        }
                        return;
                    } else if self.is_young_rawmalloced(field_ref.0) {
                        self.visit_young_rawmalloced_object(field_ref.0);
                    }
                });
            }
            if let Some((slot_addr, field)) = deferred {
                let walk = self.describe_custom_trace_slots(obj_addr, trace_fn);
                eprintln!("GC BUG: custom-trace slot walk for holder={obj_addr:#x}: {walk}");
                self.copy_nursery_object(
                    field,
                    "minor_custom_trace_target",
                    site,
                    obj_addr,
                    slot_addr,
                );
            }
        }

        let type_info = self.types.get(type_id);
        let noffsets = type_info.gc_ptr_offsets.len();
        let n_var_offsets = type_info.var_gc_ptr_offsets.len();
        let items_have_gc_ptrs = type_info.items_have_gc_ptrs;
        let item_size = type_info.item_size;
        let length_offset = type_info.length_offset;
        let base_size = type_info.size;

        // Process fixed-part GC pointer fields.
        //
        // Read by index rather than over a copy of the offsets: `base.py:286-289
        // trace` walks `offsets_to_gc_pointers(typeid)[i]` in place, and this
        // loop runs once per traced object, so a `Vec` clone here was a malloc
        // and a memcpy per object on the minor collector's hot path.  The major
        // mark's equivalent already avoids it with a reusable buffer.
        for i in 0..noffsets {
            let offset = self.types.get(type_id).gc_ptr_offsets[i];
            let slot = (obj_addr + offset) as *mut GcRef;
            let field_ref = unsafe { *slot };
            self.assert_traced_slot_initialized(
                field_ref,
                slot as usize,
                obj_addr,
                "minor_fixed_field",
                site,
            );
            if self.is_nursery_object_start(field_ref.0) {
                let new_ref = self.copy_nursery_object(
                    field_ref.0,
                    "minor_fixed_field_target",
                    site,
                    obj_addr,
                    slot as usize,
                );
                unsafe {
                    *slot = new_ref;
                }
            } else if self.is_young_rawmalloced(field_ref.0) {
                self.visit_young_rawmalloced_object(field_ref.0);
            }
        }

        // Process variable-part items if they contain GC pointers.
        if items_have_gc_ptrs && item_size > 0 {
            // The holder is the live object. `validate_type_id` already
            // rejected a forwarded header: `FORWARDED_MARKER`'s type id sits
            // outside the table. `_trace_drag_out` returns before `get_size`
            // on a corpse and traces the copy later, as an old object.
            let length = unsafe { *((obj_addr + length_offset) as *const usize) };
            let items_start = obj_addr + base_size;
            for i in 0..length {
                let item = items_start + i * item_size;
                for j in 0..n_var_offsets {
                    let offset = self.types.get(type_id).var_gc_ptr_offsets[j];
                    let slot = (item + offset) as *mut GcRef;
                    let field_ref = unsafe { *slot };
                    self.assert_traced_slot_initialized(
                        field_ref,
                        slot as usize,
                        obj_addr,
                        "minor_varsize_item",
                        site,
                    );
                    if self.is_nursery_object_start(field_ref.0) {
                        let new_ref = self.copy_nursery_object(
                            field_ref.0,
                            "minor_varsize_item_target",
                            site,
                            obj_addr,
                            slot as usize,
                        );
                        unsafe {
                            *slot = new_ref;
                        }
                    } else if self.is_young_rawmalloced(field_ref.0) {
                        self.visit_young_rawmalloced_object(field_ref.0);
                    }
                }
            }
        }
    }

    // ── Incremental marking ──

    /// incminimark.py `get_total_memory_used`. Total memory the GC
    /// is responsible for, NOT counting the nursery: old-gen objects plus
    /// raw-malloced large objects. pyre's `oldgen.total_bytes()` already
    /// aggregates promoted objects and large/raw objects allocated straight
    /// into the old generation.
    fn get_total_memory_used(&self) -> usize {
        self.oldgen.total_bytes()
    }

    /// incminimark.py `threshold_reached`. True once the old-gen
    /// total has caught up to within `extra` of the next-major threshold,
    /// i.e. it is time to make incremental major-collection progress.
    fn threshold_reached(&self, extra: usize) -> bool {
        (self.next_major_collection_threshold - self.get_total_memory_used() as f64) < extra as f64
    }

    /// incminimark.py `set_major_threshold_from`. Set the next-major
    /// threshold, capping growth at `next_major_collection_initial *
    /// growth_rate_max`, flooring at `min_heap_size`, and bounding by
    /// `max_heap_size`. Returns whether the result was bounded by the heap max.
    fn set_major_threshold_from(&mut self, mut threshold: f64, reserving_size: f64) -> bool {
        let threshold_max = self.next_major_collection_initial * self.growth_rate_max;
        if threshold > threshold_max {
            threshold = threshold_max;
        }
        //
        threshold += reserving_size;
        if threshold < self.min_heap_size {
            threshold = self.min_heap_size;
        }
        //
        let bounded = if self.max_heap_size > 0.0 && threshold > self.max_heap_size {
            threshold = self.max_heap_size;
            true
        } else {
            false
        };
        //
        self.next_major_collection_initial = threshold;
        self.next_major_collection_threshold = threshold;
        bounded
    }

    /// Begin a new incremental marking cycle.
    ///
    /// Seeds the gray stack with all root-reachable old-gen objects.
    pub fn start_incremental_cycle(&mut self) {
        debug_assert_eq!(self.gc_state, GcState::Scanning);
        self.incr_state.gray_stack.clear();
        self.incr_state.more_gray_stack.clear();
        self.incr_state.objects_marked = 0;

        self.seed_major_roots();
        self.gc_state = GcState::Marking;
    }

    fn seed_major_root(&mut self, gcref: GcRef, site: &str) {
        self.seed_major_root_arming(gcref, site, true);
    }

    /// [`Self::seed_major_root`] for a value read out of the extra-root
    /// walker. Those slots are process tables (type dicts, module dict cells,
    /// builtin carriers), re-enumerated by every collection that needs them,
    /// and none of them is a JITFRAME, so an object that already carried
    /// `GCFLAG_TRACK_YOUNG_PTRS` is not armed into
    /// `old_objects_pointing_to_young`: `IncrementalMiniMarkGC.visit` leaves
    /// that list to the write barrier. One seeded without the flag (a young
    /// non-moving object, or one the barrier already recorded) is still armed,
    /// because the flag set here would otherwise hide its young pointers from
    /// the next minor.
    fn seed_major_extra_root(&mut self, gcref: GcRef, site: &str) {
        self.seed_major_root_arming(gcref, site, false);
    }

    fn seed_major_root_arming(&mut self, gcref: GcRef, site: &str, arm_remembered: bool) {
        // `incminimark.py _collect_obj` performs NO probe on the root
        // word — the type system guarantees every `Ptr(GcStruct)` reaching
        // `_collect_ref_stk` is a real GC object, so the only tests are non-null
        // and `is_in_nursery`. pyre's root banks are untyped `i64` slices
        // (`shadow_stack::walk_bh_regs`, `walk_resume_ref_roots`) that can carry
        // a stale word from a pooled blackhole interp, so the arena range check
        // is what stands in for that guarantee and must precede every
        // dereference.
        //
        // An object outside both generations reaches a major only through
        // `prebuilt_root_objects`, which `collect_nonstack_roots`
        // (`incminimark.py`) walks separately. `incminimark.py`
        // states the invariant that makes any other admission unsound: such an
        // object "should be in 'prebuilt_root_objects', and the GCFLAG_VISITED
        // will be reset at the end of the collection" — and the only reset pass
        // walks exactly that list.
        if gcref.is_null()
            || !self.is_managed_heap_object(gcref.0)
            || !self.may_enter_marking_worklist(gcref.0)
        {
            return;
        }
        let hdr = unsafe { header_of(gcref.0) };
        // A root pointing at freed or recycled memory decodes to a garbage
        // `type_id`. Without this check the address reaches the gray stack and
        // only fails once `mark_object` pops it, by which time the walker that
        // supplied it is no longer on the stack; `site` names that walker.
        self.validate_type_id(unsafe { (*hdr).type_id() }, gcref.0, site);
        // SAFETY: header_of returns a raw pointer; keep each access
        // short-lived to avoid creating overlapping exclusive borrows.
        let track_was_set = unsafe { (*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) };
        let newly_marked = unsafe {
            if !(*hdr).has_flag(GcFlags::GCFLAG_VISITED)
                && !(*hdr).has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS)
            {
                (*hdr).set_flag(GcFlags::GCFLAG_VISITED);
                // `IncrementalMiniMarkGC.visit`: the first visit makes the
                // object black *and* arms its write barrier.  The latter is
                // what lets a mutation between incremental steps turn the
                // black object gray again.
                if !self.is_in_nursery(gcref.0) {
                    (*hdr).set_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
                }
                true
            } else {
                false
            }
        };
        if newly_marked {
            self.incr_state.gray_stack.push(gcref.0);
            self.note_nonmoving_young_mark(gcref.0, !self.is_in_nursery(gcref.0) && !track_was_set);

            // incminimark.py:1322-1340 requires marking worklists to
            // contain no nursery objects after a minor. Pyre JITFRAME
            // gcmap spills are collector metadata, not SETFIELD_GC
            // writes: a frame first grayed in STATE_SCANNING can leave
            // the active shadow stack before the next minor without a
            // mutator barrier. Arm this newly seeded old root in the
            // existing old_objects_pointing_to_young shape once, so that
            // minor forwards any such spill before resetting nursery.
            if (arm_remembered || !track_was_set) && !self.is_in_nursery(gcref.0) {
                self.old_objects_pointing_to_young.push(gcref.0);
                if crate::gc_lifetime_log_enabled() {
                    eprintln!(
                        "[gc][remember] addr={:#x} type_id={} source=major_seed state={:?}",
                        gcref.0,
                        unsafe { (*hdr).type_id() },
                        self.gc_state
                    );
                }
            }
        }
    }

    /// `incminimark.py _collect_obj`: an object in the nursery is
    /// never appended to `objects_to_trace` — "such an object is handled by
    /// minor collections and shouldn't be specially handled by major
    /// collections" — and `visit` (:2797-2799) asserts
    /// `not self.is_in_nursery(obj)` on every popped entry.  The worklist
    /// outlives the mutator resuming, so a nursery address left on it is read
    /// back after the next `reset_nursery` has recycled those bytes: the
    /// header decodes to a garbage `type_id`.
    ///
    /// A young object stays reachable without the entry.  When the minor
    /// promotes it, `drag_out_root` greys the promoted copy
    /// (`_trace_drag_out1_marking_phase`); when it is only reachable from an
    /// old parent, that parent is in the remembered set and gets requeued
    /// while MARKING (`_add_to_more_objects_to_trace`).
    ///
    /// The non-moving oldgen major is the one mode that marks the nursery in
    /// place — it leaves those bytes untouched by contract, and
    /// [`Self::note_nonmoving_young_mark`] clears the marks afterwards.
    ///
    /// A young raw-malloced object is admitted, as upstream admits it, and the
    /// entry cannot outlive the object: it is only reachable from a root, which
    /// the next minor walks, or from an old parent, which the write barrier put
    /// on the remembered set — and either route runs
    /// [`Self::visit_young_rawmalloced_object`], whose MARKING arm greys it.
    #[inline]
    fn may_enter_marking_worklist(&self, addr: usize) -> bool {
        self.oldgen_nonmoving_active || !self.is_in_nursery(addr)
    }

    /// Record a *young* object greyed during a non-moving major so its stale
    /// `GcFlags::GCFLAG_VISITED` is cleared as the strictly-last collection step.
    /// No-op (just two range checks) outside a non-moving major; the normal
    /// incremental path runs after a minor, so nothing young survives to here.
    ///
    /// Both young shapes qualify: this major marks them in place. Nursery
    /// objects are never swept here (they must not move). Unvisited young
    /// rawmallocs are released after marking (`free_unvisited_young_rawmalloced_keep_young`);
    /// survivors stay young and still need the mark cleared, because they are
    /// not on the list the oldgen sweep clears. A leftover `VISITED` is read
    /// by the next major as "already greyed", which skips the object and
    /// leaves its children untraced.
    #[inline]
    fn note_nonmoving_young_mark(&mut self, addr: usize, armed_track_young_ptrs: bool) {
        if self.oldgen_nonmoving_active
            && (self.is_in_nursery(addr) || self.is_young_rawmalloced(addr))
        {
            self.oldgen_nonmoving_young_marks
                .push((addr, armed_track_young_ptrs));
        }
    }

    /// Snapshot the root walk shared by inspection and major marking. The
    /// finalizer lists are appended by the caller because
    /// `gc.enumerate_all_roots` includes live registered finalizers, whereas
    /// major marking must leave those objects to the finalization-order pass.
    fn enumerate_root_walker_values(&self) -> Vec<GcRef> {
        self.enumerate_labeled_root_walker_values()
            .into_iter()
            .map(|(gcref, _)| gcref)
            .collect()
    }

    /// Shadow stack, JitFrame and libc jitframe slots, live deadframe slots,
    /// blackhole registers, resume-construction roots, and mutator extra areas.
    ///
    /// Shared by [`Self::enumerate_labeled_root_walker_values`] and
    /// [`Self::rescan_major_stack_roots_black_and_drain`]. Registered roots and
    /// the extra-root walker are separate sets. `mutators_quiesced` selects
    /// `walk_all_*` over `walk_*`.
    fn walk_stack_shaped_roots(mut visit: impl FnMut(GcRef, &'static str)) {
        let walk_all_mutators = crate::gc_sync::mutators_quiesced();
        {
            let mut visit_shadow_root = |gcref: &mut GcRef| {
                visit(*gcref, "shadow_stack_root");
            };
            if walk_all_mutators {
                crate::shadow_stack::walk_all_roots(&mut visit_shadow_root);
            } else {
                crate::shadow_stack::walk_roots(&mut visit_shadow_root);
            }
        }
        {
            let mut visit_jf_root = |gcref: &mut GcRef| {
                if !gcref.is_null() && crate::shadow_stack::is_libc_jitframe(gcref.0) {
                    crate::shadow_stack::trace_libc_jitframe(gcref.0, &mut |slot_ptr| {
                        let field_ref = unsafe { *slot_ptr };
                        if !field_ref.is_null() {
                            visit(field_ref, "jitframe_slot");
                        }
                    });
                } else {
                    visit(*gcref, "jf_root");
                }
            };
            if walk_all_mutators {
                crate::shadow_stack::walk_all_jf_roots(&mut visit_jf_root);
            } else {
                crate::shadow_stack::walk_jf_roots(&mut visit_jf_root);
            }
        }
        // Same source the minor-collection root phase reads; enumerated here
        // too, so a root that exists for the collector is also a root this
        // listing reports. A listing that omitted it would say an object is
        // unreachable while the collector keeps it.
        crate::walk_active_live_deadframes(&mut |addr| {
            crate::shadow_stack::trace_libc_jitframe(addr, &mut |slot_ptr| {
                let field_ref = unsafe { *slot_ptr };
                if !field_ref.is_null() {
                    visit(field_ref, "deadframe_slot");
                }
            });
        });
        {
            let mut visit_bh_root = |gcref: &mut GcRef| {
                visit(*gcref, "blackhole_register");
            };
            if walk_all_mutators {
                crate::shadow_stack::walk_all_bh_regs(&mut visit_bh_root);
            } else {
                crate::shadow_stack::walk_bh_regs(&mut visit_bh_root);
            }
        }
        // Resume-construction roots (`BlackholeInterpreter` registers and the
        // in-flight `virtuals_cache`): see the minor-collection path for why
        // those slices must be seeded as roots.
        {
            let mut visit_resume_root = |gcref: &mut GcRef| {
                visit(*gcref, "resume_ref_root");
            };
            if walk_all_mutators {
                crate::shadow_stack::walk_all_resume_ref_roots(&mut visit_resume_root);
            } else {
                crate::shadow_stack::walk_resume_ref_roots(&mut visit_resume_root);
            }
        }
        {
            let mut visit_extra_area = |gcref: &mut GcRef| {
                visit(*gcref, crate::shadow_stack::current_extra_area());
            };
            if walk_all_mutators {
                crate::shadow_stack::walk_all_extra_areas(&mut visit_extra_area);
            } else {
                crate::shadow_stack::walk_my_extra_areas(&mut visit_extra_area);
            }
        }
    }

    /// [`Self::enumerate_root_walker_values`] with each value tagged by the
    /// walker that produced it, so a root carrying a freed address can be
    /// attributed to its source rather than to the marking loop that pops it.
    fn enumerate_labeled_root_walker_values(&self) -> Vec<(GcRef, &'static str)> {
        // incminimark.py collect_roots: root_walker.walk_roots()
        // walks the same root sets as minor collection.
        let mut result = Vec::with_capacity(self.root_snapshot_capacity.get());
        // Read the registered roots in place. The minor path copies this list
        // first because `drag_out_root` takes `&mut self` while it walks; here
        // the walk only reads, so the copy would be one allocation and one pass
        // over every registered root per collection for nothing.
        for &root_ptr in &self.roots.roots {
            result.push((unsafe { *root_ptr }, "registered_root"));
        }

        Self::walk_stack_shaped_roots(|gcref, site| result.push((gcref, site)));

        crate::shadow_stack::walk_extra_roots_labeled(|gcref, label| {
            result.push((*gcref, label));
        });
        // Monotone: the hint only ever grows, so a collection that happens to
        // see fewer roots does not send the next one back through the ladder.
        self.root_snapshot_capacity
            .set(self.root_snapshot_capacity.get().max(result.len()));
        result
    }

    /// `inspector.py:get_rpy_roots` / `gc.enumerate_all_roots`: snapshot all
    /// inspection roots in the order from `gc/base.py:enumerate_all_roots`.
    /// The returned Vec is raw system-allocator storage, matching
    /// inspector.py's preallocated raw root list.
    fn enumerate_all_root_values(&self) -> Vec<GcRef> {
        let mut result = self.enumerate_root_walker_values();
        result.extend(self.prebuilt_root_objects.iter().copied().map(GcRef));
        // incminimark.py `enum_live_with_finalizers`: registered
        // finalizer objects remain roots even before they become unreachable
        // and move to an app-visible death queue. AddressDeque.foreach(..., 2)
        // visits every object address while skipping each paired queue index.
        result.extend(
            self.probably_young_objects_with_finalizers
                .iter()
                .map(|&(addr, _)| GcRef(addr)),
        );
        result.extend(
            self.old_objects_with_finalizers
                .iter()
                .map(|&(addr, _)| GcRef(addr)),
        );

        // gc/base.py `enum_pending_finalizers`: objects already moved to a
        // death queue remain GC roots until app-level code pops them.
        let pending: Vec<usize> = self
            .finalizer_handlers
            .iter()
            .flat_map(|handler| handler.deque.iter().copied())
            .collect();
        for addr in pending {
            result.push(GcRef(addr));
        }
        result.extend(self.run_old_style_finalizers.iter().copied().map(GcRef));
        result
    }

    fn seed_major_roots(&mut self) {
        // incminimark.py collect_roots: root_walker.walk_roots()
        // seeds stack roots for a major marking cycle. Mirror the same
        // root sets as minor collection, but mark old objects instead of
        // copying nursery objects.
        // `prebuilt_root_objects.foreach(self._collect_obj, None)`
        // (incminimark.py) walks the stack in place.  Index rather than
        // clone: `seed_prebuilt_root` only sets header flags and pushes onto
        // the gray stack, so the list cannot move under the loop, and the
        // clone was a full copy of every prebuilt root per major cycle.
        let mut i = 0;
        while i < self.prebuilt_root_objects.len() {
            let addr = self.prebuilt_root_objects[i];
            self.seed_prebuilt_root(addr);
            i += 1;
        }
        // Each root source hands its values straight to `seed_major_root`, as
        // `collect_roots` hands `_collect_ref_stk` to `walk_roots`: upstream
        // materializes no list of roots, and one built here is a 16-byte write
        // and a second pass over every root of every major cycle.
        //
        // The registered roots are read by index for the reason the prebuilt
        // loop above is: `seed_major_root` neither registers nor removes a
        // root, so the list cannot move under the walk.
        let mut n_roots = 0usize;
        let mut census = RootCensus::new();
        let mut seed = |gc: &mut Self, gcref: GcRef, site: &'static str, extra: bool| {
            n_roots += 1;
            census.note(site, gcref);
            if extra {
                gc.seed_major_extra_root(gcref, site);
            } else {
                gc.seed_major_root(gcref, site);
            }
        };
        let mut i = 0;
        while i < self.roots.roots.len() {
            let slot = self.roots.roots[i];
            seed(self, unsafe { *slot }, "registered_root", false);
            i += 1;
        }
        Self::walk_stack_shaped_roots(|gcref, site| seed(self, gcref, site, false));
        crate::shadow_stack::walk_extra_roots_labeled(|gcref, label| {
            seed(self, *gcref, label, true);
        });
        // Objects already moved to a death queue remain ordinary roots until
        // app-level code pops them. Registered live finalizers are deliberately
        // absent here; incminimark's finalization-order pass decides whether
        // they survive or move to the death queue.
        let mut h = 0;
        while h < self.finalizer_handlers.len() {
            let mut j = 0;
            while j < self.finalizer_handlers[h].deque.len() {
                let addr = self.finalizer_handlers[h].deque[j];
                seed(self, GcRef(addr), "finalizer_death_queue", false);
                j += 1;
            }
            h += 1;
        }
        let mut k = 0;
        while k < self.run_old_style_finalizers.len() {
            let addr = self.run_old_style_finalizers[k];
            seed(self, GcRef(addr), "old_style_finalizer_death_queue", false);
            k += 1;
        }
        census.report(self.major_collections, self.prebuilt_root_objects.len());
        // Monotone, as `enumerate_labeled_root_walker_values` keeps it: the
        // inspection walks size their snapshot from this hint.
        self.root_snapshot_capacity
            .set(self.root_snapshot_capacity.get().max(n_roots));
    }

    /// incminimark.py `prebuilt_root_objects.foreach(_collect_obj)`.
    fn seed_prebuilt_root(&mut self, addr: usize) {
        let hdr = unsafe { header_of(addr) };
        debug_assert!(unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS) });
        if unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_VISITED) } {
            unsafe {
                (*hdr).set_flag(GcFlags::GCFLAG_VISITED);
                (*hdr).set_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
            }
            self.incr_state.gray_stack.push(addr);
        }
    }

    /// Whether the collector traverses references out of this object — the
    /// property CPython's `gc.is_tracked` reports.
    ///
    /// Heap ownership alone is the wrong test: an atomic object such as an int
    /// or a float is collector-allocated on some paths (`PYRE_GC_INTERP`, the
    /// JIT, wasm) and immortal on others, so ownership would make the same
    /// Python value answer differently per backend. Its registered type has no
    /// reference to traverse either way, which is the stable property.
    pub fn object_is_tracked(&self, obj_addr: usize) -> bool {
        if !self.is_managed_heap_object(obj_addr) {
            return false;
        }
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        self.validate_type_id(type_id, obj_addr, "object_is_tracked");
        let type_info = self.types.get(type_id);
        type_info.has_gc_ptrs
    }

    /// `inspector.py:get_rpy_referents`: trace one object's direct GC
    /// referents without changing collection state.
    fn visit_referents(&self, obj_addr: usize, visitor: &mut dyn FnMut(GcRef)) {
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        self.visit_referents_with_type_id(obj_addr, type_id, visitor);
    }

    /// `visit_referents`, but handing the visitor each slot's address instead
    /// of its value, so a caller can report where a bad reference is stored.
    fn visit_referent_slots(&self, obj_addr: usize, visitor: &mut dyn FnMut(*mut GcRef)) {
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        self.validate_type_id(type_id, obj_addr, "visit_referent_slots");
        let type_info = self.types.get(type_id);
        if let Some(trace_fn) = type_info.custom_trace {
            unsafe { trace_fn(obj_addr, visitor) };
        }
        for &offset in &type_info.gc_ptr_offsets {
            visitor((obj_addr + offset) as *mut GcRef);
        }
        if type_info.items_have_gc_ptrs && type_info.item_size > 0 {
            // `validate_type_id` ran on this header. A forwarded nursery
            // corpse fails that check (`get_possibly_forwarded_type_id`
            // would have replaced the address first) and never gets here.
            let length = unsafe { *((obj_addr + type_info.length_offset) as *const usize) };
            let items_start = obj_addr + type_info.size;
            for i in 0..length {
                let item = items_start + i * type_info.item_size;
                for &offset in &type_info.var_gc_ptr_offsets {
                    visitor((item + offset) as *mut GcRef);
                }
            }
        }
    }

    /// Return the inherited frontend-only edge that app-level inspection must
    /// omit.  RPython's class/type pointer is header metadata rather than a GC
    /// referent; pyre's managed mirror is declared once on the OBJECT root.
    fn app_level_inspector_hidden_edge_offset(&self, mut type_id: u32) -> Option<usize> {
        loop {
            let info = self.types.get(type_id);
            if let Some(offset) = info.app_level_inspector_hidden_edge_offset {
                return Some(offset);
            }
            type_id = info.parent?;
        }
    }

    /// Inspector trace for a value that may be a translated prebuilt/foreign
    /// object. RPython gives those objects a typeid through the prebuilt GC
    /// layout; pyre recovers it from the registered vtable mapping.  Unlike
    /// the collector trace, this omits the frontend's managed type-pointer
    /// mirror by its slot address, so a real item equal to the class object is
    /// still reported.
    fn visit_actual_inspector_referents(&self, obj: GcRef, visitor: &mut dyn FnMut(GcRef)) -> bool {
        let Some(type_id) = self.get_actual_typeid(obj) else {
            return false;
        };
        let hidden_offset = self.app_level_inspector_hidden_edge_offset(type_id);
        self.visit_referents_with_type_id_filtered(obj.0, type_id, hidden_offset, visitor);
        true
    }

    /// `visit_referents` for a caller that already resolved the type id,
    /// because a prebuilt object carries no header to read it back out of.
    /// The bounds check sits here rather than beside the header read so that
    /// both callers get its diagnostic instead of `TypeTable::get`'s bare
    /// index panic.
    fn visit_referents_with_type_id(
        &self,
        obj_addr: usize,
        type_id: u32,
        visitor: &mut dyn FnMut(GcRef),
    ) {
        self.visit_referents_with_type_id_filtered(obj_addr, type_id, None, visitor);
    }

    fn visit_referents_with_type_id_filtered(
        &self,
        obj_addr: usize,
        type_id: u32,
        ignored_offset: Option<usize>,
        visitor: &mut dyn FnMut(GcRef),
    ) {
        self.validate_type_id(type_id, obj_addr, "visit_referents");
        let type_info = self.types.get(type_id);
        if let Some(trace_fn) = type_info.custom_trace {
            unsafe {
                trace_fn(obj_addr, &mut |slot_ptr: *mut GcRef| {
                    if ignored_offset.is_some_and(|offset| slot_ptr as usize == obj_addr + offset) {
                        return;
                    }
                    let field_ref = *slot_ptr;
                    if !field_ref.is_null() {
                        visitor(field_ref);
                    }
                });
            }
        }
        for &offset in &type_info.gc_ptr_offsets {
            if ignored_offset == Some(offset) {
                continue;
            }
            let field_ref = unsafe { *((obj_addr + offset) as *const GcRef) };
            if !field_ref.is_null() {
                visitor(field_ref);
            }
        }
        if type_info.items_have_gc_ptrs && type_info.item_size > 0 {
            // Same gate as `visit_referent_slots`: the type id was taken
            // from this header and checked, so the address is not a corpse.
            let length = unsafe { *((obj_addr + type_info.length_offset) as *const usize) };
            let items_start = obj_addr + type_info.size;
            for i in 0..length {
                let item = items_start + i * type_info.item_size;
                for &offset in &type_info.var_gc_ptr_offsets {
                    let field_ref = unsafe { *((item + offset) as *const GcRef) };
                    if !field_ref.is_null() {
                        visitor(field_ref);
                    }
                }
            }
        }
    }

    /// `rpython/rlib/rgc.py do_get_objects`, using MiniMark's
    /// GCFLAG_EXTRA exactly as the translated RPython path does.
    pub fn do_get_objects(&mut self, generation: i8, visitor: &mut dyn FnMut(GcRef)) {
        if generation == 1 {
            return;
        }
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        let roots = self.enumerate_all_root_values();
        let mut pending = roots.clone();
        let mut result = Vec::new();
        while let Some(gcref) = pending.pop() {
            if gcref.is_null() || !self.is_managed_heap_object(gcref.0) {
                continue;
            }
            let hdr = unsafe { header_of(gcref.0) };
            if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_EXTRA) } {
                continue;
            }
            unsafe { (*hdr).set_flag(GcFlags::GCFLAG_EXTRA) };
            let is_requested_generation = match generation {
                -1 => true,
                // Youth, not nursery residency: a young raw-malloced object is
                // in generation 0 and dies at the next minor like any other
                // young object.
                0 => self.is_in_nursery(gcref.0) || self.is_young_rawmalloced(gcref.0),
                2 => !self.is_in_nursery(gcref.0) && !self.is_young_rawmalloced(gcref.0),
                _ => false,
            };
            let type_id = unsafe { (*hdr).type_id() };
            let info = self.types.get(type_id);
            if is_requested_generation
                && !unsafe { (*hdr).has_flag(GcFlags::GCFLAG_DUMMY) }
                && info.is_object
                && !info.hide_from_app_level_inspector
                && !info.has_no_typedef
            {
                result.push(gcref);
            }
            self.visit_referents(gcref.0, &mut |child| pending.push(child));
        }

        // rgc.clear_gcflag_extra(roots): repeat the root traversal and restore
        // every toggled header before returning to app-level code.
        let mut pending = roots;
        while let Some(gcref) = pending.pop() {
            if gcref.is_null() || !self.is_managed_heap_object(gcref.0) {
                continue;
            }
            let hdr = unsafe { header_of(gcref.0) };
            if unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_EXTRA) } {
                continue;
            }
            unsafe { (*hdr).clear_flag(GcFlags::GCFLAG_EXTRA) };
            self.visit_referents(gcref.0, &mut |child| pending.push(child));
        }
        for gcref in result {
            visitor(gcref);
        }
    }

    /// `inspector.py get_rpy_roots`: enumerate the raw root values
    /// exactly once, without the reachability traversal used by
    /// `rgc.do_get_objects`.
    pub fn do_get_rpy_roots(&mut self, visitor: &mut dyn FnMut(GcRef)) -> bool {
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        let roots = self.enumerate_all_root_values();
        if roots.len() > self.totalroots_rpy {
            self.totalroots_rpy = roots.len() + roots.len() / 8 + 10;
        }
        for root in roots.iter().copied() {
            visitor(root);
        }
        for _ in roots.len()..self.totalroots_rpy {
            visitor(GcRef::NULL);
        }
        true
    }

    /// `inspector.py get_rpy_referents`: trace only the direct raw
    /// fields of `obj`.  In contrast, `do_get_referents` below recursively
    /// looks through non-object implementation nodes.
    ///
    /// `inspector.py _do_append_rpy_referents` traces every gcref
    /// unconditionally, so heap membership cannot reject registered prebuilt
    /// objects. `get_actual_typeid` instead rejects null, tagged immediates,
    /// and unknown foreign pointers; the managed-only nursery check below is
    /// the only remaining `header_of` on this path.
    pub fn do_get_rpy_referents(&mut self, obj: GcRef, visitor: &mut dyn FnMut(GcRef)) -> bool {
        if obj.is_null() {
            return true;
        }
        let Some(type_id) = self.get_actual_typeid(obj) else {
            return true;
        };
        let is_managed = self.is_managed_heap_object(obj.0);
        if is_managed && self.is_in_nursery(obj.0) && unsafe { (*header_of(obj.0)).is_forwarded() }
        {
            return true;
        }
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        self.visit_referents_with_type_id(obj.0, type_id, visitor);
        true
    }

    /// `inspector.py HeapDumper`, preserving its two-phase walk:
    /// roots are emitted first, then `[0, 0, 0, -1]`, then the remaining
    /// reachable objects. `GCFLAG_EXTRA` is the forwarded/PtrInfo-equivalent
    /// visited slot used by upstream, so no address side table is introduced.
    pub fn do_dump_rpy_heap(&mut self, fd: i32) -> Result<bool, i32> {
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        let roots = self.enumerate_all_root_values();
        let mut root_pending = Vec::new();
        for &root in &roots {
            self.heap_dump_add(root, &mut root_pending);
        }
        let mut pending = Vec::new();
        let mut writer = HeapDumpWriter::new(fd);
        let result = (|| {
            while let Some(obj) = root_pending.pop() {
                self.heap_dump_writeobj(obj, &mut writer, &mut pending)?;
            }
            writer.write_marker()?;
            while let Some(obj) = pending.pop() {
                self.heap_dump_writeobj(obj, &mut writer, &mut pending)?;
            }
            writer.flush()?;
            Ok(true)
        })();

        // inspector.py finish_processing: restore GCFLAG_EXTRA by
        // repeating the reachability walk from roots, including on write error
        // so a bad fd cannot poison later inspector calls.
        self.heap_dump_clear_gcflag(roots);
        result
    }

    /// `incminimark.py raw_malloc_memory_pressure`, including the
    /// framework transform's object-owned field store
    /// (`gctransform/framework.py`). A negative size releases
    /// previously reported pressure just as upstream does.
    pub fn do_add_memory_pressure(&mut self, size: isize, object: GcRef) {
        // The forced-top store below rewrites the same published free/top words
        // compiled code reads inline, so park the other mutators for it the way
        // every other entry point that mutates published allocator state does.
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        if !object.is_null() && self.is_managed_heap_object(object.0) {
            let type_id = unsafe { (*header_of(object.0)).type_id() };
            if (type_id as usize) < self.types.len()
                && let Some(offset) = self.types.get(type_id).memory_pressure_offset
            {
                unsafe {
                    *((object.0 + offset) as *mut isize) = size;
                }
            }
        }

        // incminimark.py:1103-1106: account for the raw allocation plus a
        // small per-allocation overhead so many tiny allocations still drive
        // a major collection.
        self.next_major_collection_threshold -=
            (size as f64) + (2 * std::mem::size_of::<usize>()) as f64;
        if self.next_major_collection_threshold < 0.0 {
            // `IncrementalMiniMarkGC.raw_malloc_memory_pressure`: make the
            // next nursery allocation take `collect_and_reserve` by advancing
            // free to top.  The direction matters when top is a pinned-object
            // barrier: moving top backwards would discard that barrier.
            let top = self.nursery.top_ptr();
            unsafe { self.nursery.set_free_ptr(top.cast_mut()) };
        }
    }

    /// `inspector.py MemoryPressureCounter`: walk the root-reachable
    /// heap, summing each type's translated `special_memory_pressure` field.
    /// `GCFLAG_EXTRA` is the upstream walker slot; no address side table is
    /// introduced.
    pub fn do_count_memory_pressure(&mut self) -> isize {
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        let roots = self.enumerate_all_root_values();
        let mut pending = Vec::new();
        for &root in &roots {
            self.heap_dump_add(root, &mut pending);
        }
        let mut count = 0isize;
        while let Some(obj) = pending.pop() {
            let type_id = unsafe { (*header_of(obj.0)).type_id() };
            if (type_id as usize) < self.types.len()
                && let Some(offset) = self.types.get(type_id).memory_pressure_offset
            {
                count = count.wrapping_add(unsafe { *((obj.0 + offset) as *const isize) });
            }
            let mut referents = Vec::new();
            self.visit_referents(obj.0, &mut |child| referents.push(child));
            for child in referents {
                self.heap_dump_add(child, &mut pending);
            }
        }
        self.heap_dump_clear_gcflag(roots);
        count
    }

    fn heap_dump_add(&self, obj: GcRef, pending: &mut Vec<GcRef>) {
        if obj.is_null() || !self.is_managed_heap_object(obj.0) {
            return;
        }
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_EXTRA) } {
            return;
        }
        unsafe { (*hdr).set_flag(GcFlags::GCFLAG_EXTRA) };
        pending.push(obj);
    }

    fn heap_dump_writeobj(
        &self,
        obj: GcRef,
        writer: &mut HeapDumpWriter,
        pending: &mut Vec<GcRef>,
    ) -> Result<(), i32> {
        let type_id = unsafe { (*header_of(obj.0)).type_id() };
        writer.write(obj.0 as isize)?;
        writer.write((type_id as isize) + 1)?;
        writer.write(self.rpy_memory_usage(obj).unwrap_or(0) as isize)?;
        let mut referents = Vec::new();
        self.visit_referents(obj.0, &mut |child| referents.push(child));
        for child in referents {
            writer.write(child.0 as isize)?;
            self.heap_dump_add(child, pending);
        }
        writer.write(-1)
    }

    fn heap_dump_clear_gcflag(&self, roots: Vec<GcRef>) {
        let mut pending = roots;
        while let Some(obj) = pending.pop() {
            if obj.is_null() || !self.is_managed_heap_object(obj.0) {
                continue;
            }
            let hdr = unsafe { header_of(obj.0) };
            if unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_EXTRA) } {
                continue;
            }
            unsafe { (*hdr).clear_flag(GcFlags::GCFLAG_EXTRA) };
            self.visit_referents(obj.0, &mut |child| pending.push(child));
        }
    }

    /// `referents.py try_cast_gcref_to_w_root`.  The translated
    /// `T_IS_RPYTHON_INSTANCE` bit is `TypeInfo::is_object`.  Two flags then
    /// reject an instance: `hide_from_app_level_inspector` (a frame that has
    /// a typedef but is omitted from `gc.get_objects`) and `has_no_typedef`
    /// (a `W_Root` whose typedef is null).
    ///
    /// This is the predicate behind the `gc.get_objects` filter and the
    /// `get_rpy_*` wrap decision. The referents walk has one deliberate extra
    /// boundary: an OBJECT-layout value hidden from enumeration (the
    /// execution-frame shape described on
    /// `TypeInfo::hide_from_app_level_inspector`) still stops traversal.  A
    /// typedef-less `W_Root` does not: `try_cast_gcref_to_w_root` returns
    /// None and `_list_w_obj_referents` expands it.
    fn is_app_level_object_ref(&self, obj: GcRef) -> bool {
        // `referents.py rgc.get_gcflag_dummy(gcref)`: a dummy stands in for
        // an object the collector no longer holds, so it is never an app-level
        // object however its type reads.  Only a managed object carries the
        // header the flag lives in.
        if self.is_managed_heap_object(obj.0)
            && unsafe { (*header_of(obj.0)).has_flag(GcFlags::GCFLAG_DUMMY) }
        {
            return false;
        }
        let Some(type_id) = self.get_actual_typeid(obj) else {
            return false;
        };
        if type_id as usize >= self.types.len() {
            return false;
        }
        let info = self.types.get(type_id);
        info.is_object && !info.hide_from_app_level_inspector && !info.has_no_typedef
    }

    /// Whether app-level referents inspection must stop at `obj`.
    ///
    /// `referents._list_w_obj_referents` stops when
    /// `try_cast_gcref_to_w_root` returns a `W_Root`.  An execution frame is
    /// omitted from `gc.get_objects` but still has a typedef, so it remains
    /// a boundary.  A typedef-less `W_Root` (`TypeInfo::has_no_typedef`) is
    /// expanded, the same as any other non-`W_Root` gcref.
    fn is_app_level_referent_boundary(&self, obj: GcRef) -> bool {
        // Preserve every ordinary app-level boundary first, including
        // prebuilt/foreign W_Root objects which are not managed by this heap.
        if self.is_app_level_object_ref(obj) {
            return true;
        }
        if !self.is_managed_heap_object(obj.0)
            || unsafe { (*header_of(obj.0)).has_flag(GcFlags::GCFLAG_DUMMY) }
        {
            return false;
        }
        let Some(type_id) = self.get_actual_typeid(obj) else {
            return false;
        };
        if (type_id as usize) >= self.types.len() {
            return false;
        }
        let info = self.types.get(type_id);
        info.is_object && !info.has_no_typedef
    }

    /// `pypy/module/gc/referents.py _list_w_obj_referents`: visit the
    /// app-level objects `obj` refers to directly, looking through the
    /// interpreter-internal structs in between. A visited app-level object
    /// terminates that branch of the walk; anything else is expanded, so a
    /// list reports its items rather than its item array. GCFLAG_EXTRA keeps
    /// each node out of the walk twice and is restored before returning.
    pub fn do_get_referents(&mut self, obj: GcRef, visitor: &mut dyn FnMut(GcRef)) {
        if obj.is_null() || self.get_actual_typeid(obj).is_none() {
            return;
        }
        // A host-side raw local can still hold a just-forwarded nursery
        // address, whose header word is a forwarding pointer rather than
        // flags — toggling GCFLAG_EXTRA on it would write through that word.
        // Reject exactly that, not every young object: `referents.py _list_w_obj_referents`
        // does not require an empty nursery, and a live nursery object is an
        // ordinary app-level referent source.
        if self.is_in_nursery(obj.0) && unsafe { (*header_of(obj.0)).is_forwarded() } {
            return;
        }
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        let mut pending: Vec<GcRef> = Vec::new();
        let mut result: Vec<GcRef> = Vec::new();
        let mut i = 0usize;
        let mut parent = obj;
        loop {
            let mut children: Vec<GcRef> = Vec::new();
            self.visit_actual_inspector_referents(parent, &mut |child| children.push(child));
            for child in children {
                if child.is_null() || self.get_actual_typeid(child).is_none() {
                    continue;
                }
                if self.is_managed_heap_object(child.0) {
                    let hdr = unsafe { header_of(child.0) };
                    if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_EXTRA) } {
                        continue;
                    }
                    unsafe { (*hdr).set_flag(GcFlags::GCFLAG_EXTRA) };
                } else if pending.contains(&child) {
                    // RPython toggles GCFLAG_EXTRA on prebuilt objects too.
                    // Pyre's foreign wrappers have no GC header, so the same
                    // per-call identity set lives in the existing pending Vec.
                    continue;
                }
                pending.push(child);
            }
            // Walk the queue until a non-app-level node needs expanding; on
            // reaching the end without one, every branch has terminated.
            let mut expand = false;
            while i < pending.len() {
                parent = pending[i];
                i += 1;
                if self.is_app_level_referent_boundary(parent) {
                    result.push(parent);
                } else {
                    expand = true;
                    break;
                }
            }
            if !expand {
                break;
            }
        }
        for gcref in &pending {
            if self.is_managed_heap_object(gcref.0) {
                unsafe { (*header_of(gcref.0)).clear_flag(GcFlags::GCFLAG_EXTRA) };
            }
        }
        for gcref in result {
            visitor(gcref);
        }
    }

    /// Re-gray a root the end-of-marking rescan has already visited, or seed it
    /// when it is still white.
    ///
    /// `seed_major_root` leaves an object with `GCFLAG_VISITED` off the gray
    /// stack, so a black root that picked up a new child would not be traced
    /// again. A nursery root takes the seed path, which marks without queueing
    /// when the marking worklist must not hold it.
    fn regray_or_seed_major_root(&mut self, gcref: GcRef, site: &str) {
        if gcref.is_null() {
            return;
        }
        // incminimark.py keeps nursery objects out of a marking worklist, so a
        // nursery root goes through the seeding path.
        let regray = !self.is_in_nursery(gcref.0)
            && self.is_managed_heap_object(gcref.0)
            && unsafe { (*header_of(gcref.0)).has_flag(GcFlags::GCFLAG_VISITED) };
        if regray {
            self.incr_state.gray_stack.push(gcref.0);
        } else {
            self.seed_major_root(gcref, site);
        }
    }

    /// `collect_nonstack_roots` at the end of MARKING (`major_collection_step`).
    ///
    /// Owns `prebuilt_root_objects`, the extra-root walker,
    /// `walk_rescan_roots`, and the finalizer death queues. Already-black
    /// values are left black: `seed_major_root` does not re-trace a
    /// `GCFLAG_VISITED` object. The barrier-less stack-shaped sets are the
    /// ones that need regray, and they belong to
    /// [`Self::rescan_major_stack_roots_black_and_drain`].
    ///
    /// Upstream `collect_nonstack_roots` walks `prebuilt_root_objects`, the
    /// static roots (`root_walker.walk_roots`), and `enum_pending_finalizers`.
    fn rescan_major_nonstack_roots_and_drain(&mut self) {
        // In place, for the reason `seed_major_roots` gives.
        let mut i = 0;
        while i < self.prebuilt_root_objects.len() {
            let addr = self.prebuilt_root_objects[i];
            self.seed_prebuilt_root(addr);
            i += 1;
        }
        // `collect_nonstack_roots` repeats the static roots because the
        // mutator may have written them since `collect_roots`. A cycle that
        // `do_collect_full` seeded and is finishing in the same call has had
        // no mutator step, so the extra-root tables are the ones the seed
        // walked and every value in them is already `GCFLAG_VISITED`.
        if !self.full_collect_owns_cycle {
            crate::shadow_stack::walk_extra_roots(|gcref| {
                self.seed_major_extra_root(*gcref, "rescan_extra_root");
            });
        }
        // TLS exception cells live on the per-mutator frame area for the
        // first pass. Upstream's second `collect_nonstack_roots` still
        // repeats the non-stack carriers that can be written after the
        // snapshot; the rescan-only walkers are those three cells.
        crate::shadow_stack::walk_rescan_roots(|gcref| {
            self.seed_major_root(*gcref, "rescan_tls_exception");
        });
        let pending: Vec<usize> = self
            .finalizer_handlers
            .iter()
            .flat_map(|handler| handler.deque.iter().copied())
            .collect();
        for addr in pending {
            self.seed_major_root(GcRef(addr), "rescan_finalizer_death_queue");
        }
        while let Some(obj_addr) = self.incr_state.gray_stack.pop() {
            self.mark_object(obj_addr);
        }
    }

    /// Drive incremental major-collection progress after a minor collection.
    ///
    /// This follows incminimark's accounting rule: each major step grants
    /// `nursery_size / 2` bytes of promotion credit, and allocation-heavy
    /// minors may need multiple consecutive steps so old-gen growth does not
    /// outrun marking.
    fn run_major_progress_after_minor(&mut self, force_enabled: bool) {
        // incminimark.py `minor_collection_with_major_progress`: automatic
        // major progress after a minor stops while disabled; explicit
        // collect() passes force_enabled and stays ungated without flipping
        // `enabled`.
        if !self.enabled && !force_enabled {
            return;
        }
        let extrasize = self.pending_reserving_size;
        if self.gc_state == GcState::Scanning && !self.threshold_reached(extrasize) {
            return;
        }

        loop {
            self.major_collection_step();

            // incminimark.py:849-855 target (A1).
            if self.gc_state == GcState::Scanning {
                break;
            }

            // `minor_collection_with_major_progress` target (A2).  The
            // allocation being reserved does not exist yet but must already
            // fit under the credit line, hence the `threshold - extrasize`
            // comparison.  If promotions still outrun credit, PyPy performs
            // another bare `_minor_collection()` before the next major step;
            // repeating only the major step lets a high-survival nursery keep
            // growing faster than marking observes it.
            let mut threshold = self.threshold_bytes_made_old;
            if threshold >= extrasize {
                threshold -= extrasize;
                if self.bytes_made_old_since_cycle <= threshold {
                    break;
                }
            }
            self.minor_collection_body();
        }
    }

    /// incminimark.py `major_collection_step`.
    fn major_collection_step(&mut self) {
        let start = GcClock::start();
        let old_state = self.gc_state.encoded();
        self.oom_signalled_this_step = false;
        // incminimark.py `gc_step_until` runs `_minor_collection` (and its
        // MARKING `if_black` pass) before every `major_collection_step`.
        // `do_collect_oldgen_nonmoving` uses `gc_step_until_scanning` with no
        // minor, so a black remembered parent (TRACK clear, on
        // `old_objects_pointing_to_young`) would keep a white child across
        // this step. Re-grey those parents here, the way the skipped minor
        // would have.
        if self.oldgen_nonmoving_active && self.gc_state == GcState::Marking {
            self.regray_remembered_black_during_marking();
        }
        self.debug_check_consistency_at("major_step");

        // incminimark.py:2406-2436: each state-machine step grants half a
        // nursery of promotion credit.
        self.threshold_bytes_made_old = self
            .threshold_bytes_made_old
            .saturating_add(self.config.nursery_size / 2);

        match self.gc_state {
            GcState::Scanning => {
                // incminimark.py STATE_SCANNING.
                self.bytes_made_old_since_cycle = 0;
                self.threshold_bytes_made_old = self.config.nursery_size / 2;
                self.start_incremental_cycle();
            }
            GcState::Marking => {
                // `incremental_mark_step` performs the MARKING->SWEEPING seam
                // itself when the gray stack is exhausted.
                self.incremental_mark_step();
            }
            GcState::Sweeping => self.incremental_sweep_step(),
            GcState::Finalizing => {
                // incminimark.py:2623-2631: recursive collections from a
                // handler must see a collector ready to start a new scan.
                self.gc_state = GcState::Scanning;
                // The cycle is over here, which is what `gc.get_count`'s
                // second element counts from.
                self.minor_collections_at_major_end = self.minor_collections;
                self.execute_finalizer_triggers();
            }
        }

        // incminimark.py:2634-2644. A max-heap MemoryError exits upstream
        // before this site; pyre defers it instead, so suppress the transition
        // event when this step armed one — on either channel, since both stand
        // for the same raise.
        if !self.oom_signalled_this_step {
            let duration = start.elapsed_secs();
            self.total_gc_time += duration;
            self.hooks
                .fire_gc_collect_step(duration, old_state, self.gc_state.encoded());
        }
    }

    /// incminimark.py `debug_check_consistency`.
    ///
    /// Self-gated on the debug level rather than gated at its call sites, as
    /// upstream is: the body opens with `if self.DEBUG:`, so a run that did
    /// not ask for the checks pays one load and the checks are real
    /// assertions rather than `debug_assert!`s that a release build drops.
    /// `PYPY_GC_DEBUG` is the only way to arm them, and a run that sets it is
    /// asking to be aborted on a broken invariant.
    #[allow(dead_code)] // incminimark.py `debug_check_consistency`
    fn debug_check_consistency(&self) {
        self.debug_check_consistency_at("unspecified");
    }

    fn debug_check_consistency_at(&self, site: &'static str) {
        if self.config.debug == 0 {
            return;
        }
        // `do_collect_oldgen_nonmoving` skips the leading minor on purpose, so
        // young rawmalloced objects and young weakrefs are still live. Those
        // two asserts are "we just finished `_minor_collection`" facts
        // (`incminimark.py debug_check_consistency`); they do not hold here.
        if !self.oldgen_nonmoving_active {
            assert!(
                self.oldgen.young_rawmalloced_is_empty(),
                "young raw-malloced objects in a major collection"
            );
            assert!(
                self.young_objects_with_weakrefs.is_empty(),
                "young objects with weakrefs in a major collection"
            );
        }
        if self.oldgen.rawmalloc_sweep_pending() {
            assert_eq!(
                self.gc_state,
                GcState::Sweeping,
                "raw_malloc_might_sweep must be empty outside SWEEPING"
            );
        }
        if self.gc_state == GcState::Marking && !self.oldgen_nonmoving_active {
            // incminimark.py `_check_not_in_nursery`: ordinary marking never
            // queues nursery objects. A non-moving major marks the nursery in
            // place (`may_enter_marking_worklist`), so the assert does not
            // apply there.
            for &addr in self
                .incr_state
                .gray_stack
                .iter()
                .chain(self.incr_state.more_gray_stack.iter())
            {
                assert!(
                    !self.is_in_nursery(addr),
                    "'objects_to_trace' contains a nursery object"
                );
            }
        }
        // The reachable-graph walk (and the all-old-object MARKING walk) is
        // `DEBUG >= 2`, matching `_minor_collection`'s `if self.DEBUG >= 2`.
        // A full heap walk here changes collection timing enough to hide the
        // live-object-freed crash.
        if self.config.debug >= 2 {
            self.debug_check_reachable(site);
            if self.gc_state == GcState::Marking {
                self.oldgen.for_each_allocated_object(|addr| {
                    if self.is_managed_heap_object(addr) {
                        self.debug_check_object_at(addr, site);
                    }
                });
            }
        }
    }

    /// gc/base.py `debug_check_consistency`'s heap half — enumerate every root
    /// and trace the whole reachable graph, checking each object once.
    ///
    /// Roots include the shadow stack, jitframe gcmap slots, blackhole
    /// registers, resume construction, extra areas (compile-window among
    /// them), prebuilt/static objects, and registered finalizers
    /// (`enumerate_all_root_values`).
    ///
    /// Upstream keeps its seen set and pending stack as GC-side `AddressDict` /
    /// `AddressStack` because it has no other allocator; here they are ordinary
    /// Rust containers, which is the same structure without the bookkeeping.
    /// The parent map is the same kind of container: each object records the
    /// holder that reached it, so a slot that does not name an object can
    /// report that holder and the root the walk started from. `GCBase.trace`
    /// receives a typed pointer and would not load an integer; this walk
    /// validates before the load of the child header.
    fn debug_check_reachable(&self, site: &'static str) {
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut pending: Vec<usize> = Vec::new();
        // child -> (holder, slot address). A root has no entry.
        let mut parent: std::collections::HashMap<usize, (usize, usize)> =
            std::collections::HashMap::new();
        let labeled: Vec<(GcRef, &'static str)> = {
            let mut v = self.enumerate_labeled_root_walker_values();
            for root in self.enumerate_all_root_values() {
                v.push((root, "all_root"));
            }
            v
        };
        for (root, label) in labeled {
            if root.is_null() {
                continue;
            }
            // A root may name mapped host storage the production walks skip:
            // `walk_rbigint_parts_cache` yields headerless `std::alloc` digit
            // blocks, and a recorded trace `Ref` can hold `Function.name`
            // storage. See `debug_accept_host_storage_child`.
            if !self.debug_child_is_traceable(root.0) {
                if self.debug_accept_host_storage_child(root.0) {
                    continue;
                }
                panic!(
                    "debug_check_reachable: bad root label={label} value={:#x} value_state={} {} {} site={site}",
                    root.0,
                    self.debug_describe_bad_value(root.0),
                    self.debug_describe_external_miss(root.0),
                    self.debug_describe_type_layout(root.0),
                );
            }
            if seen.insert(root.0) {
                self.debug_check_object_at(root.0, site);
                pending.push(root.0);
            }
        }
        while let Some(obj_addr) = pending.pop() {
            let type_id = unsafe { (*header_of(obj_addr)).type_id() };
            if (type_id as usize) >= self.types.len() {
                continue;
            }
            let mut children: Vec<(usize, usize)> = Vec::new();
            unsafe {
                self.types.get(type_id).for_each_gc_ptr(obj_addr, |slot| {
                    let child = *slot;
                    if child.is_null() {
                        return;
                    }
                    // A custom trace reads through the slot after this visitor
                    // returns (an instance's boxed storage length, for one).
                    // After a minor no reachable slot may still name a
                    // forwarding stub or a retired nursery, so report it here,
                    // before the trace dereferences it.
                    let stale = self.nursery.contains_retired(child.0)
                        || (self.is_nursery_object_start(child.0)
                            && (*header_of(child.0)).is_forwarded());
                    if stale {
                        self.debug_panic_bad_child(
                            obj_addr,
                            type_id,
                            slot as usize,
                            child.0,
                            &parent,
                        );
                    }
                    children.push((child.0, slot as usize));
                });
            }
            for (child, slot_addr) in children {
                if seen.contains(&child) {
                    continue;
                }
                // A slot may name mapped host storage on purpose
                // (`Function.name`, see `debug_accept_host_storage_child`).
                // An unmapped word, a nursery address and a retired-nursery
                // address are the defects this walk reports.
                if !self.debug_child_is_traceable(child) {
                    if self.debug_accept_host_storage_child(child) {
                        continue;
                    }
                    self.debug_panic_bad_child(obj_addr, type_id, slot_addr, child, &parent);
                }
                seen.insert(child);
                parent.insert(child, (obj_addr, slot_addr));
                self.debug_check_object_at(child, site);
                pending.push(child);
            }
        }
    }

    /// Mapped host storage the collector never moves, as a root or a child.
    ///
    /// The minor and major traces skip a root or child `is_managed_heap_object`
    /// rejects. `Function.name` (`FUNCTION_NAME_OFFSET`) is such a child by
    /// design, in an immortal builtin (a `malloc_raw` String) and in a
    /// collector-owned function alike (`FunctionName::Borrowed`, the
    /// `co_name` of a `Box::into_raw`'d code object that is never freed). An
    /// unmapped word is not host storage: an integer in a GC slot is what
    /// this walk exists to report. Nursery addresses (an interior pointer
    /// included) and retired-nursery addresses are collector territory and
    /// never take this skip.
    fn debug_accept_host_storage_child(&self, child: usize) -> bool {
        self.debug_outside_collector(child) && self.debug_address_mapped(child)
    }

    /// Not in any nursery and not collector-owned.
    ///
    /// `nursery.contains` is the live arena; `contains_retired` is every arena
    /// `debug_rotate` has protected. Both fail the "not in any nursery" test
    /// a forwarding stub has to fail. `is_in_nursery` is not used here: it
    /// asserts on an odd word, and this predicate is how an integer is
    /// rejected. Young raw-malloced payloads are collector-owned;
    /// `try_alloc_young` also inserts them into the raw-malloc set
    /// `is_managed_heap_object` reads, and the young test stays so the two
    /// cannot drift apart.
    fn debug_outside_collector(&self, addr: usize) -> bool {
        addr != 0
            && !self.nursery.contains_retired(addr)
            && !self.nursery.contains(addr)
            && !self.is_young_rawmalloced(addr)
            && !self.is_managed_heap_object(addr)
    }

    /// A child the debug walk may read a header from.
    ///
    /// Nursery membership keeps the header inside the arena
    /// (`is_nursery_object_start`). Old-gen arenas and raw-malloc payloads are
    /// exact membership (`OldGeneration::contains`). `malloc_typed` and
    /// `malloc_typed_immortal` leaves (`None`, the other prebuilt singletons)
    /// sit outside both and are still real headers; `registered_external_header`
    /// is what admits one, and only after a mapped-page probe so an integer in
    /// the slot is not dereferenced. A forwarding stub fails both: after the
    /// collection the slot should name the copy, and the stub's page stays
    /// mapped when the nursery ring has not reused it.
    ///
    /// Nothing outside the collector is admitted by its header word alone: the
    /// allocator word in front of a host allocation can read as a registered
    /// type id with no flags. Such an address is host storage, a leaf here as
    /// it is to the production traces (`debug_accept_host_storage_child`).
    fn debug_child_is_traceable(&self, addr: usize) -> bool {
        if !self.is_valid_gc_object(addr) {
            return false;
        }
        if self.is_nursery_object_start(addr) || self.oldgen.contains(addr) {
            let hdr = unsafe { *header_of(addr) };
            return !hdr.is_forwarded();
        }
        self.debug_external_object(addr)
    }

    /// `alloc_with_gc_header` object outside the managed arenas.
    ///
    /// The page probe is the whole point: `registered_external_header` loads
    /// the payload's first word, which is the segfault this check exists to
    /// turn into a panic.
    fn debug_external_object(&self, addr: usize) -> bool {
        if addr < GcHeader::SIZE
            || !self.debug_address_mapped(addr)
            || !self.debug_address_mapped(addr - GcHeader::SIZE)
        {
            return false;
        }
        let hdr = unsafe { *header_of(addr) };
        if hdr.is_forwarded() {
            return false;
        }
        self.registered_external_header(addr).is_some()
    }

    /// Whether `addr` is in some mapped region. False on targets without
    /// `region::query`; the debug walk then refuses the address instead of
    /// loading it.
    fn debug_address_mapped(&self, addr: usize) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = self;
            // A retired nursery is still mapped, with `Protection::NONE`.
            // `query` succeeds and a header load then faults. Readable is
            // the predicate the load needs.
            match region::query(addr as *const u8) {
                Ok(info) => info.is_readable(),
                Err(_) => false,
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (self, addr);
            false
        }
    }

    /// `bh_probe_type_name` for a holder the walk has already accepted.
    ///
    /// `TypeInfo` carries no name. The namer reads `ob_type`, so it is only
    /// called for an `is_object` layout whose first word is a registered
    /// vtable — the same guard as the blackhole probe.
    fn debug_object_type_name(&self, addr: usize, type_id: u32) -> &'static str {
        if (type_id as usize) >= self.types.len() || !self.types.get(type_id).is_object {
            return "?";
        }
        let vtable = unsafe { *(addr as *const usize) };
        if !self.vtable_to_type_id.contains_key(&vtable) {
            return "?";
        }
        crate::bh_probe_type_name(addr).unwrap_or("?")
    }

    /// Walk the parent map to the object that entered the graph as a root.
    fn debug_reach_root(
        start: usize,
        parent: &std::collections::HashMap<usize, (usize, usize)>,
    ) -> usize {
        let mut addr = start;
        let limit = parent.len().saturating_add(1);
        for _ in 0..limit {
            match parent.get(&addr) {
                Some(&(up, _)) if up != addr => addr = up,
                _ => break,
            }
        }
        addr
    }

    /// Registered-type shape of a mapped header, for a root the vtable witness
    /// declined. `TypeInfo` has no name; the size and pointer offsets identify
    /// the layout.
    fn debug_describe_type_layout(&self, addr: usize) -> String {
        if addr < GcHeader::SIZE
            || !self.debug_address_mapped(addr)
            || !self.debug_address_mapped(addr - GcHeader::SIZE)
        {
            return "layout=unread".to_string();
        }
        let hdr = unsafe { *header_of(addr) };
        if hdr.is_forwarded() {
            return "layout=forwarded".to_string();
        }
        let tid = hdr.type_id();
        if (tid as usize) >= self.types.len() {
            return format!("layout=tid_oor flags={:#x}", hdr.flags().bits());
        }
        let info = self.types.get(tid);
        format!(
            "layout=size:{} is_object:{} has_gc_ptrs:{} offsets:{:?} item_size:{} \
             custom:{} flags={:#x}",
            info.size,
            info.is_object,
            info.has_gc_ptrs,
            info.gc_ptr_offsets,
            info.item_size,
            info.custom_trace.is_some(),
            hdr.flags().bits(),
        )
    }

    /// Why a mapped address was not `registered_external_header`.
    ///
    /// A root or child outside the arenas is admitted only when its first word
    /// is a registered vtable and the header tid matches. The miss names which
    /// half failed, and the type name when that vtable is one the namer can read.
    fn debug_describe_external_miss(&self, addr: usize) -> String {
        if !self.is_valid_gc_object(addr)
            || addr < GcHeader::SIZE
            || !self.debug_address_mapped(addr)
            || !self.debug_address_mapped(addr - GcHeader::SIZE)
        {
            return "external=unmapped".to_string();
        }
        let word0 = unsafe { *(addr as *const usize) };
        let vtable_tid = self.vtable_to_type_id.get(&word0).copied();
        let name = vtable_tid.and_then(|_| crate::bh_probe_type_name(addr));
        format!(
            "word0={word0:#x} vtable_tid={vtable_tid:?} name={}",
            name.unwrap_or("?")
        )
    }

    /// What a rejected value looks like, without loading an unmapped page.
    fn debug_describe_bad_value(&self, addr: usize) -> String {
        let generation = self.describe_generation(addr);
        if !self.is_valid_gc_object(addr) || addr < GcHeader::SIZE {
            return format!("invalid gen={generation}");
        }
        if !self.debug_address_mapped(addr) || !self.debug_address_mapped(addr - GcHeader::SIZE) {
            return format!("unmapped gen={generation}");
        }
        let hdr = unsafe { *header_of(addr) };
        if hdr.is_forwarded() {
            let fwd = unsafe { GcHeader::forwarding_address(header_of(addr)) };
            return format!("forwarded fwd={fwd:#x} gen={generation}");
        }
        format!("tid={} gen={generation}", hdr.type_id())
    }

    fn debug_panic_bad_child(
        &self,
        holder: usize,
        holder_tid: u32,
        slot_addr: usize,
        value: usize,
        parent: &std::collections::HashMap<usize, (usize, usize)>,
    ) -> ! {
        let offset = slot_addr.wrapping_sub(holder);
        let holder_type = self.debug_object_type_name(holder, holder_tid);
        let field = crate::bh_probe_field_name(holder_tid, offset);
        let root = Self::debug_reach_root(holder, parent);
        let parent_desc = match parent.get(&holder) {
            Some(&(parent_addr, parent_slot)) => {
                let parent_tid = unsafe { (*header_of(parent_addr)).type_id() };
                let parent_type = self.debug_object_type_name(parent_addr, parent_tid);
                format!(
                    "parent={parent_addr:#x} parent_tid={parent_tid} parent_type={parent_type} \
                     parent_slot={parent_slot:#x}"
                )
            }
            None => "parent=root".to_string(),
        };
        let root_desc = if root == holder {
            "root=holder".to_string()
        } else {
            let root_tid = unsafe { (*header_of(root)).type_id() };
            let root_type = self.debug_object_type_name(root, root_tid);
            format!("root={root:#x} root_tid={root_tid} root_type={root_type}")
        };
        let mut holder_words = [0usize; 8];
        if let Some(total) = self.try_object_total_size(holder) {
            let payload = total.saturating_sub(GcHeader::SIZE);
            let words = (payload / std::mem::size_of::<usize>()).min(holder_words.len());
            for (index, word) in holder_words.iter_mut().take(words).enumerate() {
                *word = unsafe { *((holder as *const usize).add(index)) };
            }
        }
        let enclosing = self.describe_enclosing_container(holder, slot_addr, &holder_words);
        let holder_layout = self.debug_describe_type_layout(holder);
        panic!(
            "debug_check_reachable: bad child holder={holder:#x} holder_tid={holder_tid} \
             holder_type={holder_type} slot_addr={slot_addr:#x} slot_offset={offset:#x} \
             field={field} value={value:#x} value_state={} {parent_desc} {root_desc} \
             enclosing={enclosing} {holder_layout} holder_words={:?}",
            self.debug_describe_bad_value(value),
            &holder_words,
        );
    }

    /// incminimark.py `debug_check_object`: after a collection nothing is left
    /// in the nursery but the pinned objects, and neither of the two flags the
    /// collection itself uses may survive it.  State-specific arms match
    /// `_debug_check_object_scanning` / `_marking` / `_sweeping` /
    /// `_finalizing`.
    fn debug_check_object_at(&self, obj_addr: usize, site: &'static str) {
        let hdr = unsafe { &*header_of(obj_addr) };
        if self.is_pinned(GcRef(obj_addr)) {
            assert!(
                self.is_in_nursery(obj_addr),
                "pinned object not in nursery at {obj_addr:#x}"
            );
            return;
        }
        // A non-moving major leaves nursery / young-rawmalloced objects in
        // place (flags=0, no TRACK_YOUNG_PTRS). The after-collection object
        // arms below do not apply to them.
        if self.oldgen_nonmoving_active
            && (self.is_in_nursery(obj_addr) || self.oldgen.young_rawmalloced_contains(obj_addr))
        {
            return;
        }
        assert!(
            !self.is_in_nursery(obj_addr),
            "object in nursery after collection at {obj_addr:#x}"
        );
        assert!(
            !hdr.has_flag(GcFlags::GCFLAG_VISITED_RMY),
            "GCFLAG_VISITED_RMY after collection at {obj_addr:#x}"
        );
        assert!(
            !hdr.has_flag(GcFlags::GCFLAG_PINNED),
            "GCFLAG_PINNED outside the nursery after collection at {obj_addr:#x}"
        );
        match self.gc_state {
            GcState::Scanning => self.debug_check_object_scanning(obj_addr),
            GcState::Marking => self.debug_check_object_marking(obj_addr, site),
            GcState::Sweeping => self.debug_check_object_sweeping(obj_addr),
            GcState::Finalizing => self.debug_check_object_scanning(obj_addr),
        }
    }

    /// `write_barrier` / `remember_young_pointer` clears TRACK_YOUNG_PTRS and
    /// must push `old_objects_pointing_to_young`. A black object with the flag
    /// clear is only legal while it is on that list; `collect_oldrefs_to_nursery`
    /// re-sets the flag. The bug class is the flag clear and the object absent
    /// from the list.
    fn debug_assert_track_young_ptrs_or_remembered(&self, obj_addr: usize, hdr: &GcHeader) {
        if hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS)
            || self.old_objects_pointing_to_young.contains(&obj_addr)
        {
            return;
        }
        panic!(
            "missing GCFLAG_TRACK_YOUNG_PTRS {}",
            self.debug_track_young_ptrs_context(obj_addr)
        );
    }

    /// Format tid / flags / remembered-set membership for a TRACK_YOUNG_PTRS
    /// failure so the first violation names whether `write_barrier` cleared the
    /// flag without `collect_oldrefs_to_nursery` re-arming it.
    fn debug_track_young_ptrs_context(&self, obj_addr: usize) -> String {
        if obj_addr == 0 {
            return "addr=0x0".to_string();
        }
        if !self.is_managed_heap_object(obj_addr) {
            return format!("addr={obj_addr:#x} unmanaged");
        }
        let hdr = unsafe { &*header_of(obj_addr) };
        format!(
            "addr={obj_addr:#x} tid={} tid_and_flags={:#x} TRACK_YOUNG_PTRS={} \
             VISITED={} HAS_CARDS={} CARDS_SET={} NO_HEAP_PTRS={} PINNED={} \
             in_old_objects_pointing_to_young={} in_gray={} in_more_gray={} \
             in_prebuilt_roots={} managed={} nursery={} oldgen={} young_raw={} gen={} \
             gc_state={:?} oldgen_nonmoving={} minors={} majors={}",
            hdr.type_id(),
            hdr.tid_and_flags,
            hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS),
            hdr.has_flag(GcFlags::GCFLAG_VISITED),
            hdr.has_flag(GcFlags::GCFLAG_HAS_CARDS),
            hdr.has_flag(GcFlags::GCFLAG_CARDS_SET),
            hdr.has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS),
            hdr.has_flag(GcFlags::GCFLAG_PINNED),
            self.old_objects_pointing_to_young.contains(&obj_addr),
            self.incr_state.gray_stack.contains(&obj_addr),
            self.incr_state.more_gray_stack.contains(&obj_addr),
            self.prebuilt_root_objects.contains(&obj_addr),
            self.is_managed_heap_object(obj_addr),
            self.nursery.contains(obj_addr),
            self.oldgen.contains(obj_addr),
            self.oldgen.young_rawmalloced_contains(obj_addr),
            self.describe_generation(obj_addr),
            self.gc_state,
            self.oldgen_nonmoving_active,
            self.minor_collections,
            self.major_collections,
        )
    }

    fn debug_is_visited_or_gray(&self, addr: usize) -> bool {
        if addr == 0 || !self.is_managed_heap_object(addr) {
            return true;
        }
        let hdr = unsafe { &*header_of(addr) };
        hdr.has_flag(GcFlags::GCFLAG_VISITED)
            || hdr.has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS)
            || self.is_pinned(GcRef(addr))
            || self.incr_state.gray_stack.contains(&addr)
            || self.incr_state.more_gray_stack.contains(&addr)
            || self.prebuilt_root_objects.contains(&addr)
    }

    /// `incminimark.py` `_debug_check_not_white` over the remembered set and
    /// the live root slots, at the MARKING→SWEEPING seam. A walk of every
    /// old object (custom_trace of every dict) hid the crash without
    /// panicking; this is the cheap set that still sees the same heap.
    fn debug_check_visited_children_not_white(&self, site: &'static str) {
        let check_obj = |addr: usize| {
            if !self.is_managed_heap_object(addr) {
                return;
            }
            let hdr = unsafe { &*header_of(addr) };
            if !hdr.has_flag(GcFlags::GCFLAG_VISITED) {
                return;
            }
            let type_id = hdr.type_id();
            if (type_id as usize) >= self.types.len() {
                return;
            }
            unsafe {
                self.types.get(type_id).for_each_gc_ptr(addr, |slot| {
                    let child = *slot;
                    if !child.is_null() {
                        self.debug_check_not_white(addr, slot as usize, child.0, site);
                    }
                });
            }
        };
        for &addr in &self.old_objects_pointing_to_young {
            check_obj(addr);
        }
        for &addr in &self.old_objects_pointing_to_pinned {
            check_obj(addr);
        }
        for (gcref, label) in self.enumerate_labeled_root_walker_values() {
            if gcref.is_null() {
                continue;
            }
            if self.is_managed_heap_object(gcref.0)
                && !self.debug_is_visited_or_gray(gcref.0)
                && !(self.is_in_nursery(gcref.0) && !self.oldgen_nonmoving_active)
            {
                panic!(
                    "white root at MARKING->SWEEPING site={site} label={label} {}",
                    self.debug_track_young_ptrs_context(gcref.0),
                );
            }
            check_obj(gcref.0);
        }
    }

    /// incminimark.py `_debug_check_object_marking`.
    fn debug_check_object_marking(&self, obj_addr: usize, site: &'static str) {
        let hdr = unsafe { &*header_of(obj_addr) };
        if !hdr.has_flag(GcFlags::GCFLAG_VISITED) {
            return;
        }
        let type_id = hdr.type_id();
        if (type_id as usize) >= self.types.len() {
            return;
        }
        let has_gc_ptrs = self.types.get(type_id).has_gc_ptrs;
        // pyre greys by setting VISITED at push (`seed_major_root`,
        // `grey_child`); upstream `visit` sets it when tracing. An object
        // still on the worklist is gray even though the flag is already on.
        // A write_barrier may have cleared TRACK_YOUNG_PTRS (and recorded the
        // object on old_objects_pointing_to_young) before `mark_object` pops
        // it and re-sets both flags together.
        if self.incr_state.gray_stack.contains(&obj_addr)
            || self.incr_state.more_gray_stack.contains(&obj_addr)
        {
            return;
        }
        if has_gc_ptrs {
            self.debug_assert_track_young_ptrs_or_remembered(obj_addr, hdr);
        }
        let mut children: Vec<(usize, usize)> = Vec::new();
        unsafe {
            self.types.get(type_id).for_each_gc_ptr(obj_addr, |slot| {
                let child = *slot;
                if !child.is_null() {
                    children.push((slot as usize, child.0));
                }
            });
        }
        for (slot_addr, child_addr) in children {
            self.debug_check_not_white(obj_addr, slot_addr, child_addr, site);
        }
    }

    /// incminimark.py `_debug_check_not_white`: a black object must not
    /// point at a white one.
    fn debug_check_not_white(
        &self,
        holder_addr: usize,
        slot_addr: usize,
        child_addr: usize,
        site: &'static str,
    ) {
        if !self.is_managed_heap_object(child_addr) {
            return;
        }
        if self.debug_is_visited_or_gray(child_addr)
            || (self.oldgen_nonmoving_active && self.is_in_nursery(child_addr))
        {
            return;
        }
        panic!(
            "black -> white pointer found site={site} \
             holder={} child={} slot_offset={:?}",
            self.debug_track_young_ptrs_context(holder_addr),
            self.debug_track_young_ptrs_context(child_addr),
            slot_addr.checked_sub(holder_addr),
        );
    }

    /// incminimark.py `_debug_check_object_sweeping`.
    fn debug_check_object_sweeping(&self, obj_addr: usize) {
        let hdr = unsafe { &*header_of(obj_addr) };
        let type_id = hdr.type_id();
        if (type_id as usize) < self.types.len() {
            let info = self.types.get(type_id);
            if info.has_gc_ptrs && !self.is_pinned(GcRef(obj_addr)) {
                self.debug_assert_track_young_ptrs_or_remembered(obj_addr, hdr);
            }
        }
        assert!(
            !hdr.has_flag(GcFlags::GCFLAG_FINALIZATION_ORDERING),
            "unexpected GCFLAG_FINALIZATION_ORDERING at {obj_addr:#x}"
        );
        assert!(
            !hdr.has_flag(GcFlags::GCFLAG_CARDS_SET),
            "unexpected GCFLAG_CARDS_SET at {obj_addr:#x}"
        );
        if hdr.has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS) {
            assert!(
                hdr.has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS),
                "GCFLAG_NO_HEAP_PTRS is set, but GCFLAG_TRACK_YOUNG_PTRS isn't! at {obj_addr:#x}"
            );
        }
    }

    /// incminimark.py `_debug_check_object_scanning`.
    fn debug_check_object_scanning(&self, obj_addr: usize) {
        let hdr = unsafe { &*header_of(obj_addr) };
        assert!(
            !hdr.has_flag(GcFlags::GCFLAG_VISITED),
            "unexpected GCFLAG_VISITED at {obj_addr:#x}"
        );
        self.debug_check_object_sweeping(obj_addr);
    }

    /// Perform one incremental marking step.
    ///
    /// Processes up to `mark_budget_per_step` bytes from the gray stack.
    /// Like incminimark, this is a byte budget, but we always process at least
    /// one object so very small budgets still make forward progress.
    /// Returns `true` if marking is complete (gray stack exhausted).
    pub fn incremental_mark_step(&mut self) -> bool {
        debug_assert_eq!(self.gc_state, GcState::Marking);
        // incminimark.py:2453-2457: a nursery with many survivors raises this
        // increment's byte budget so promotion cannot outrun tracing.
        let estimate = self
            .incr_state
            .mark_budget_per_step
            .max(self.nursery_surviving_size.saturating_mul(2))
            .max(1);
        let mut remaining = estimate;
        while let Some(obj_addr) = self.incr_state.gray_stack.pop() {
            let obj_size = self.object_total_size(obj_addr);
            self.mark_object(obj_addr);
            self.incr_state.objects_marked += 1;
            if obj_size > remaining {
                remaining = 0;
                break;
            }
            remaining -= obj_size;
        }

        // incminimark.py:2458-2470: if the ordinary frontier used less than
        // half the step, swap in every object exposed by concurrent mutation
        // and drain that worklist completely.  This deliberately trades some
        // incrementality for termination.
        if self.incr_state.gray_stack.is_empty()
            && remaining >= estimate / 2
            && !self.incr_state.more_gray_stack.is_empty()
        {
            std::mem::swap(
                &mut self.incr_state.gray_stack,
                &mut self.incr_state.more_gray_stack,
            );
            while let Some(obj_addr) = self.incr_state.gray_stack.pop() {
                self.mark_object(obj_addr);
                self.incr_state.objects_marked += 1;
            }
        }

        if self.incr_state.gray_stack.is_empty() && self.incr_state.more_gray_stack.is_empty() {
            self.finish_incremental_marking();
            return true;
        }
        false // more work to do
    }

    fn object_total_size(&self, obj_addr: usize) -> usize {
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        self.validate_type_id(type_id, obj_addr, "object_total_size");
        GcHeader::SIZE + self.size_for_typeid(obj_addr, type_id, "object_total_size")
    }

    /// `base.py get_size` / `inspector.py get_rpy_memory_usage
    /// get_rpy_memory_usage`.  The inspector reports the translated object
    /// size, not the collector header.  Fixed-size rows are already rounded by
    /// `gctypelayout.encode_type_shape`; variable-size rows are rounded after
    /// reading the live length.
    fn rpy_memory_usage(&self, obj: GcRef) -> Option<usize> {
        let type_id = self.get_actual_typeid(obj)?;
        if type_id as usize >= self.types.len() {
            return None;
        }
        let size = self.try_size_for_typeid(obj.0, type_id)?;
        if self.types.get(type_id).item_size == 0 {
            return Some(size);
        }
        let align_mask = GcHeader::ALIGN - 1;
        size.checked_add(align_mask).map(|size| size & !align_mask)
    }

    /// `inspector.py get_rpy_type_index`.  Pyre's compact type table
    /// stores one paired TYPE_INFO/CLASSTYPE row per type, so its zero-based
    /// row number differs from RPython's group-member index only by the dummy
    /// member that `TypeLayoutBuilder.make_type_info_group` installs at zero.
    fn rpy_type_index(&self, obj: GcRef) -> Option<usize> {
        let type_id = self.get_actual_typeid(obj)?;
        if type_id as usize >= self.types.len() {
            return None;
        }
        (type_id as usize).checked_add(1)
    }

    /// Total size of `addr`, or `None` when its header does not decode to a
    /// registered type. The panicking [`Self::object_total_size`] is unusable
    /// from a diagnostic that is already reporting a corrupt heap.
    fn try_object_total_size(&self, addr: usize) -> Option<usize> {
        let type_id = unsafe { (*header_of(addr)).type_id() } as usize;
        if type_id >= self.types.len() {
            return None;
        }
        let payload_size = self.try_size_for_typeid(addr, type_id as u32)?;
        Some(GcHeader::SIZE + payload_size)
    }

    /// The eight words read out of a holder or a child, on one line.
    ///
    /// The crash-triage extractors in `check.py` and `cpython_tests/run.py`
    /// keep only the first line of a panic body, and the two array formats Rust
    /// offers each give up something: `{:x?}` drops the `0x` every other
    /// address in the message carries, and `{:#x?}` puts each element on a line
    /// of its own, which is a line those extractors never read.
    fn hex_words(words: &[usize; 8]) -> String {
        let hex: Vec<String> = words.iter().map(|word| format!("{word:#x}")).collect();
        format!("[{}]", hex.join(", "))
    }

    /// Which generation an address falls in.
    ///
    /// A corrupt child separates into two unrelated defects depending on the
    /// answer: an old-gen holder naming a nursery child is a store that never
    /// reached the remembered set, while an old-gen child is a sweep that
    /// reclaimed a still-referenced object. The panic reports both ends so the
    /// two are never confused.
    fn describe_generation(&self, addr: usize) -> &'static str {
        if self.nursery.contains(addr) {
            "nursery"
        } else if self.is_young_rawmalloced(addr) {
            // Young and non-moving: it cannot be a stale forwarding address,
            // but a minor can still free it, so it is not the "oldgen" defect
            // either. Asked before `oldgen.contains`, which also answers yes.
            "young-rawmalloced"
        } else if self.oldgen.contains(addr) {
            "oldgen"
        } else {
            "outside"
        }
    }

    /// Whether the header in front of a nursery address decodes at all.
    ///
    /// `is_nursery_object_start` is a range check, so it answers "could be an
    /// object", not "is one". This is the cheap half of the question the
    /// `copy_nursery_object` panic answers in full.
    fn nursery_start_decodes(&self, obj_addr: usize) -> bool {
        let hdr = unsafe { *header_of(obj_addr) };
        hdr.is_forwarded() || (hdr.type_id() as usize) < self.types.len()
    }

    /// Every slot a custom trace names, with the state of the value in it.
    ///
    /// For a JITFRAME the slot set is whatever `jf_gcmap` says, so this is the
    /// only way to see the map the collector actually acted on.
    fn describe_custom_trace_slots(
        &self,
        obj_addr: usize,
        trace_fn: crate::trace::CustomTraceFn,
    ) -> String {
        let mut slots: Vec<(usize, usize)> = Vec::new();
        unsafe {
            trace_fn(obj_addr, &mut |slot_ptr: *mut GcRef| {
                slots.push((slot_ptr as usize, (*slot_ptr).0));
            });
        }
        let mut out = format!("{} slots:", slots.len());
        for (slot_addr, value) in slots {
            let offset = slot_addr.wrapping_sub(obj_addr);
            let tag = if value == 0 {
                "null".to_string()
            } else if !self.is_managed_heap_object(value) {
                "unmanaged".to_string()
            } else if !self.nursery_start_decodes(value) {
                format!("{}-BAD", self.describe_generation(value))
            } else {
                let hdr = unsafe { *header_of(value) };
                if hdr.is_forwarded() {
                    format!("{}-forwarded", self.describe_generation(value))
                } else {
                    format!("{}-tid{}", self.describe_generation(value), hdr.type_id())
                }
            };
            out.push_str(&format!(" [+{offset}]={value:#x}:{tag}"));
        }
        out
    }

    /// The nearest preceding word that decodes as an object header whose
    /// extent covers `obj_addr`.
    ///
    /// `is_nursery_object_start` is a bare range check, so a slot holding an
    /// *interior* address passes it and the collector then reads a payload
    /// word as a header. That is a different defect from a header that was
    /// genuinely clobbered, and the header word alone does not separate them:
    /// finding a real object that contains the address settles it, and its
    /// `interior_offset` names the field the slot actually points at.
    fn describe_nearest_header(&self, obj_addr: usize) -> String {
        let word = std::mem::size_of::<usize>();
        let floor = self.nursery.start_ptr() as usize;
        for back in 1..=64usize {
            let Some(candidate) = obj_addr.checked_sub(back * word) else {
                break;
            };
            if candidate < floor + GcHeader::SIZE {
                break;
            }
            let hdr = unsafe { *header_of(candidate) };
            if hdr.is_forwarded() {
                let fwd = unsafe { GcHeader::forwarding_address(header_of(candidate)) };
                // Follow the copy for size: the original's first payload
                // word is the forwarding address (`set_forwarding_address`).
                // Keep scanning unless that extent actually covers `obj_addr`.
                if !self.is_managed_heap_object(fwd) {
                    continue;
                }
                let fwd_hdr = unsafe { *header_of(fwd) };
                if fwd_hdr.is_forwarded() {
                    continue;
                }
                let Some(size) = self.try_size_for_typeid(fwd, fwd_hdr.type_id()) else {
                    continue;
                };
                if candidate + size <= obj_addr {
                    continue;
                }
                return format!("forwarded back={back}w candidate={candidate:#x} fwd={fwd:#x}");
            }
            let tid = hdr.type_id();
            if tid as usize >= self.types.len() {
                continue;
            }
            let Some(size) = self.try_size_for_typeid(candidate, tid) else {
                continue;
            };
            if candidate + size <= obj_addr {
                continue;
            }
            return format!(
                "tid={tid} back={back}w candidate={candidate:#x} size={size} \
                 interior_offset={} custom_trace={}",
                obj_addr - candidate,
                self.types.get(tid).custom_trace.is_some(),
            );
        }
        "none".to_string()
    }

    /// Which object actually owns `slot_addr`.
    ///
    /// A `custom_trace` may hand the collector slots that live outside the
    /// object it was called on — the mapdict storage block and the module-dict
    /// storage boxes are walked that way. When the child in such a slot is
    /// corrupt, the discriminating question is whether the *container* was
    /// reclaimed (its own header would be a freelist link) or the child was, and
    /// the panic's `holder_*` fields answer neither. Locate the container by
    /// testing each of the holder's leading words for an extent that covers the
    /// slot, and report its header.
    fn describe_enclosing_container(
        &self,
        holder_addr: usize,
        slot_addr: usize,
        holder_words: &[usize],
    ) -> String {
        if self.is_managed_heap_object(holder_addr)
            && let Some(size) = self.try_object_total_size(holder_addr)
            && (holder_addr..holder_addr + size).contains(&slot_addr)
        {
            return "self".to_string();
        }
        for (index, &word) in holder_words.iter().enumerate() {
            if !self.is_managed_heap_object(word) {
                continue;
            }
            let Some(size) = self.try_object_total_size(word) else {
                continue;
            };
            if !(word..word + size).contains(&slot_addr) {
                continue;
            }
            let hdr = unsafe { *header_of(word) };
            return format!(
                "word{index}@{word:#x} tid={} tid_and_flags={:#x} size={size} slot_offset={}",
                hdr.type_id(),
                hdr.tid_and_flags,
                slot_addr - word,
            );
        }
        "unknown".to_string()
    }

    /// Grey one child reference: if it is a managed, not-yet-visited heap
    /// object, set VISITED and push it onto the gray stack. The
    /// `is_managed_heap_object` guard mirrors `seed_major_root`: a
    /// `Ptr(GcStruct)` field can transiently point at memory outside the
    /// GC-managed heap during the L1/L2 stepping-stone state (e.g.
    /// `W_TupleObject.wrappeditems` → `std::alloc`'d ItemsBlock). In that
    /// window calling `header_of` on the field would dereference memory before
    /// the std::alloc'd block. Upstream RPython `_collect_obj`
    /// (incminimark.py) does not need this guard because RPython's
    /// type system guarantees every `Ptr(GcStruct)` is GC-managed; it converges
    /// away once every `gc_ptr_offsets` target is a real GC allocation.
    fn grey_child(&mut self, addr: usize, holder_addr: usize, slot_addr: usize, site: &str) {
        if addr == 0 || !self.is_managed_heap_object(addr) {
            return;
        }
        let hdr = unsafe { header_of(addr) };
        // incminimark `visit`: forwarding stubs exist only inside
        // `collect_nursery`, which updates every slot naming one before it
        // returns. A slot that still names a stub here is a missed root or
        // write barrier of the previous minor.
        debug_assert!(
            !unsafe { (*hdr).is_forwarded() },
            "forwarding stub reached major marking: child={addr:#x} holder={holder_addr:#x} \
             slot={slot_addr:#x} site={site}"
        );
        // incminimark `visit`: `ll_assert(not self.is_in_nursery(obj), ...)`.
        debug_assert!(
            self.may_enter_marking_worklist(addr),
            "nursery child reached major marking: child={addr:#x} holder={holder_addr:#x} \
             slot={slot_addr:#x} site={site}"
        );
        if self.may_enter_marking_worklist(addr) {
            let type_id = unsafe { (*hdr).type_id() };
            if type_id as usize >= self.types.len() {
                let holder_type_id = unsafe { (*header_of(holder_addr)).type_id() };
                let mut holder_words = [0usize; 8];
                for (index, word) in holder_words.iter_mut().enumerate() {
                    *word = unsafe { *((holder_addr as *const usize).add(index)) };
                }
                let mut child_words = [0usize; 8];
                for (index, word) in child_words.iter_mut().enumerate() {
                    *word = unsafe { *((addr as *const usize).add(index)) };
                }
                let child_vtable_type_id = self.vtable_to_type_id.get(&child_words[0]).copied();
                let holder_hdr = unsafe { *header_of(holder_addr) };
                panic!(
                    "GC BUG: invalid major child type_id={} at child_addr={:#x} \
                     holder_addr={:#x} holder_type_id={} slot_addr={:#x} \
                     holder_offset={:?} site={} holder_words={} \
                     child_vtable_type_id={:?} child_words={} \
                     holder_tid_and_flags={:#x} holder_in_remembered={} \
                     child_gen={} holder_gen={} \
                     enclosing={} gc_state={:?} minors={} majors={}",
                    type_id,
                    addr,
                    holder_addr,
                    holder_type_id,
                    slot_addr,
                    slot_addr.checked_sub(holder_addr),
                    site,
                    Self::hex_words(&holder_words),
                    child_vtable_type_id,
                    Self::hex_words(&child_words),
                    holder_hdr.tid_and_flags,
                    self.old_objects_pointing_to_young.contains(&holder_addr),
                    self.describe_generation(addr),
                    self.describe_generation(holder_addr),
                    self.describe_enclosing_container(holder_addr, slot_addr, &holder_words),
                    self.gc_state,
                    self.minor_collections,
                    self.major_collections,
                );
            }
            let track_was_set = unsafe { (*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) };
            if unsafe {
                !(*hdr).has_flag(GcFlags::GCFLAG_VISITED)
                    && !(*hdr).has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS)
            } {
                unsafe {
                    (*hdr).set_flag(GcFlags::GCFLAG_VISITED);
                    // `IncrementalMiniMarkGC.visit` sets both bits on the
                    // first visit.  A nursery object exists here only during
                    // pyre's non-moving-major adaptation; nursery objects do
                    // not participate in the old-object write barrier.
                    if !self.is_in_nursery(addr) {
                        (*hdr).set_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
                    }
                };
                self.incr_state.gray_stack.push(addr);
                self.note_nonmoving_young_mark(addr, !self.is_in_nursery(addr) && !track_was_set);
            }
        }
    }

    /// Mark a single object: trace its GC pointer fields and push
    /// unmarked children onto the gray stack.
    fn mark_object(&mut self, obj_addr: usize) {
        // The scalar trace descriptors are read out of the borrowed
        // `type_info`, and the two offset tables through raw slices, so the
        // `self.types` borrow is released before greying (which mutates
        // `self.incr_state.gray_stack`). Each child is then streamed straight
        // to the gray stack, as `trace` and `_trace_slow_path` stream to
        // `callback`: both index the type id's offset list in place and copy
        // nothing out of it.
        // incminimark.py `visit`: `ll_assert(not
        // self.is_in_nursery(obj), "nursery object in 'objects_to_trace'")`.
        debug_assert!(
            self.may_enter_marking_worklist(obj_addr),
            "nursery object {obj_addr:#x} in the marking worklist",
        );
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        // A non-moving major can trace a live nursery object without first
        // copying it into its reserved old-gen shadow.  Keep that unoccupied
        // shadow black, but do not push/trace it: only its header (and varsize
        // length) is initialized until copy_nursery_object overwrites the full
        // block.  Upstream normally reaches the shadow through
        // GCFLAG_HAS_SHADOW during the leading minor; this is the equivalent
        // for pyre's oldgen-only non-moving major.
        if self.is_in_nursery(obj_addr) {
            // Every flag bit of `FORWARDED_MARKER` reads set, so a worklist
            // entry a minor collection forwarded would pass the shadow test
            // below and then fail the map lookup under a message about the
            // shadow map.  The fault is the stale worklist entry; say so.
            assert!(
                !unsafe { (*header_of(obj_addr)).is_forwarded() },
                "stale major worklist entry: forwarded header at {obj_addr:#x}"
            );
            if unsafe { (*header_of(obj_addr)).has_flag(GcFlags::GCFLAG_HAS_SHADOW) } {
                let shadow_obj = *self
                    .nursery_objects_shadows
                    .get(&obj_addr)
                    .expect("GCFLAG_HAS_SHADOW but no shadow found");
                unsafe { (*header_of(shadow_obj)).set_flag(GcFlags::GCFLAG_VISITED) };
            }
        }
        let custom_trace;
        let (item_size, length_offset, fixed_size, items_have_gc_ptrs);
        // `offsets_to_gc_pointers(typeid)` and
        // `varsize_offsets_to_gcpointers_in_var_part(typeid)`, read per traced
        // object. `TypeRegistry::register` builds each entry's offset tables
        // before it appends the entry and never rewrites a registered one, so
        // the `Vec<usize>` buffers these name outlive the trace: growing
        // `TypeRegistry::entries` moves the `TypeInfo` structs, not the
        // allocations their offset Vecs own.  Holding the slices rather than
        // copying them per object is what lets `grey_child` take `&mut self`.
        let offsets: *const [usize];
        let var_offsets: *const [usize];
        {
            let type_info = self.types.get(type_id);
            custom_trace = type_info.custom_trace;
            item_size = type_info.item_size;
            length_offset = type_info.length_offset;
            fixed_size = type_info.size;
            items_have_gc_ptrs = type_info.items_have_gc_ptrs;
            offsets = &*type_info.gc_ptr_offsets;
            var_offsets = &*type_info.var_gc_ptr_offsets;
        }
        // SAFETY: the tables live in `TypeRegistry`, which outlives this call,
        // and nothing below registers or rewrites a type.
        let (offsets, var_offsets) = unsafe { (&*offsets, &*var_offsets) };

        // `GCBase.trace` / `_trace_slow_path`: `custom_trace_dispatcher`
        // when `has_custom_trace`, then the fixed `offsets_to_gc_pointers`
        // walk. Variable-part items are the other half of the slow path.
        if let Some(trace_fn) = custom_trace {
            unsafe {
                trace_fn(obj_addr, &mut |slot_ptr: *mut GcRef| {
                    let field_ref = *slot_ptr;
                    if !field_ref.is_null() {
                        self.grey_child(
                            field_ref.0,
                            obj_addr,
                            slot_ptr as usize,
                            "major_custom_trace",
                        );
                    }
                });
            }
        }
        // Fixed-part fields (count bounded by the struct's GC field count).
        for &offset in offsets {
            let field_ref = unsafe { *((obj_addr + offset) as *const GcRef) };
            if !field_ref.is_null() {
                self.grey_child(
                    field_ref.0,
                    obj_addr,
                    obj_addr + offset,
                    "major_fixed_field",
                );
            }
        }
        // Variable-part items: streamed one at a time, never buffered, so a
        // large GC-managed pointer array does not double the marking-side
        // peak memory or retain that capacity for the collector's lifetime.
        if items_have_gc_ptrs && item_size > 0 {
            // The nursery arm above asserts `!is_forwarded`. An old
            // object is never forwarded (`is_forwarded` requires the
            // nursery), so this length is the object's own.
            let length = unsafe { *((obj_addr + length_offset) as *const usize) };
            let items_start = obj_addr + fixed_size;
            for i in 0..length {
                let item = items_start + i * item_size;
                for &offset in var_offsets {
                    let slot = item + offset;
                    let field_ref = unsafe { *(slot as *const GcRef) };
                    if !field_ref.is_null() {
                        self.grey_child(field_ref.0, obj_addr, slot, "major_varsize_item");
                    }
                }
            }
        }
    }

    /// incminimark.py:1793-1799 + :2461-2470 — final snapshot-at-the-beginning
    /// re-scan before the sweep completes.
    ///
    /// Every old object modified since the last minor collection is recorded in
    /// `old_objects_pointing_to_young` (the write barrier's `old_objects_pointing_to_young`) with
    /// its `TRACK_YOUNG_PTRS` cleared. Such a store may install an `old -> old`
    /// edge to a still-white object. The per-minor black re-grey in
    /// `do_collect_nursery` only re-scans modifications up to the last minor; a
    /// cycle finished at a safepoint *between* minors (`do_collect_oldgen_nonmoving`
    /// branch A, `gc_step`) leaves the trailing delta unscanned. The later
    /// incremental sweep would otherwise free those white-but-reachable children.
    /// Re-grey every already-marked modified object and trace transitively so the
    /// delta is marked before the retain/sweep below. (pyre's marking convention
    /// sets VISITED at push time, so a black object is simply re-pushed; white
    /// entries are left for the normal frontier / retain to drop.)
    fn rescan_remembered_black_and_drain(&mut self) {
        let snapshot: Vec<usize> = self.old_objects_pointing_to_young.to_vec();
        for addr in snapshot {
            if unsafe { (*header_of(addr)).has_flag(GcFlags::GCFLAG_VISITED) } {
                self.incr_state.gray_stack.push(addr);
            }
        }
        while let Some(obj_addr) = self.incr_state.gray_stack.pop() {
            self.mark_object(obj_addr);
        }
    }

    /// End-of-marking rescan of the barrier-less stack-shaped roots.
    ///
    /// Owns the shadow stack, JitFrame / libc jitframe slots, live deadframe
    /// slots, blackhole register banks, resume-construction roots, and the
    /// mutator extra areas. Each value is re-grayed when it is already
    /// `GCFLAG_VISITED` outside the nursery, and seeded otherwise. Registered
    /// roots and the extra-root walker belong to
    /// [`Self::rescan_major_nonstack_roots_and_drain`].
    ///
    /// `major_collection_step` repeats only `collect_nonstack_roots` at the end
    /// of MARKING. Upstream's stack roots stay covered without this pass: a
    /// JITFRAME is nursery-allocated, so every minor re-traces it and a
    /// promotion during MARKING re-queues it, and a value stored into a stack
    /// slot mid-cycle was promoted during MARKING and is born black. pyre's
    /// stack-shaped sets are mutated with no write barrier and hold pre-cycle
    /// objects. An off-GC JitFrame is `alloc_zeroed` memory outside the heap,
    /// so its pointer stays valid across a collecting call while compiled code
    /// stores Refs into its gcmap slots (nursery-built frames get their
    /// pointer back from `_reload_frame_if_necessary` instead). The blackhole
    /// register banks and resume-construction roots are plain slices, and
    /// `seed_major_root` arms a newly seeded old root into the remembered set
    /// only once — the next minor drains that set and nothing re-arms it. A
    /// black root can therefore come to hold the only reference to a white
    /// object. Walking these sets once more here can only add survivors.
    fn rescan_major_stack_roots_black_and_drain(&mut self) {
        // Same split as the non-stack rescan: finish the walk, then mutate
        // the mark state. `walk_stack_shaped_roots` reads TLS root slots.
        let mut roots = Vec::new();
        Self::walk_stack_shaped_roots(|gcref, _site| roots.push(gcref));
        for gcref in roots {
            self.regray_or_seed_major_root(gcref, "marking_regray_root");
        }
        self.drain_gray_stack();
    }

    /// incminimark.py `visit_all_objects`: mark until the worklist is
    /// empty.  Every caller that seeds a root outside the incremental budget
    /// finishes it here rather than leaving work for the next step.
    fn drain_gray_stack(&mut self) {
        while let Some(obj_addr) = self.incr_state.gray_stack.pop() {
            self.mark_object(obj_addr);
        }
    }

    /// `visit_all_objects` over both `objects_to_trace` and
    /// `more_objects_to_trace`. A write barrier during MARKING may fill
    /// `more_gray_stack` after the primary stack was already empty.
    fn drain_marking_worklists(&mut self) {
        loop {
            if self.incr_state.gray_stack.is_empty() {
                if self.incr_state.more_gray_stack.is_empty() {
                    break;
                }
                std::mem::swap(
                    &mut self.incr_state.gray_stack,
                    &mut self.incr_state.more_gray_stack,
                );
            }
            self.drain_gray_stack();
        }
    }

    /// Mark conditional edges to a fixed point.
    ///
    /// Two kinds, asked together because each can answer the other. RPython
    /// sees an object's weakref lifeline as an ordinary field: marking the
    /// owner immediately marks the lifeline. Pyre's temporary carrier for
    /// builtin layouts is address-keyed, so it reports the same edges here,
    /// after ordinary marking has established which owners survived. The C
    /// fields of a `rawrefcount` block are conditional in the same way and for
    /// the same reason ([`Self::mark_c_edges`]). A newly marked value may itself
    /// make another owner live, hence the fixed point.
    fn mark_conditional_edges_to_fixed_point(&mut self) {
        loop {
            let mut classify_owner = |owner: usize| -> Option<usize> {
                if owner == 0 || !self.oldgen.contains(owner) {
                    return Some(owner);
                }
                let hdr = unsafe { header_of(owner) };
                unsafe { (*hdr).has_flag(GcFlags::GCFLAG_VISITED) }.then_some(owner)
            };
            let roots = crate::shadow_stack::mark_ephemeron_tables(&mut classify_owner);
            for root in roots {
                self.seed_major_root(root, "ephemeron_table_root");
            }
            // Progress is what this drain itself marked: `objects_marked` is
            // owned by `incremental_mark_step` and never moves here, so reading
            // it would end the walk after one pass and leave the second owner's
            // value unmarked while pruning keeps its now-live entry.  Seeding
            // an already-VISITED root pushes nothing, so an empty drain is
            // exactly "no owner became live in this pass".
            let mut drained = 0usize;
            while let Some(obj_addr) = self.incr_state.gray_stack.pop() {
                self.mark_object(obj_addr);
                drained += 1;
            }
            drained += self.mark_c_edges();
            if drained == 0 {
                break;
            }
        }
    }

    /// incminimark.py:2473-2533: finish MARKING and freeze this cycle's sweep
    /// candidates.  Every VISITED-dependent consumer runs before either raw or
    /// arena memory is freed.
    fn finish_incremental_marking(&mut self) {
        debug_assert_eq!(self.gc_state, GcState::Marking);
        // Re-scan the trailing snapshot-at-the-beginning delta (old objects
        // modified since the last minor) before the sweep, or a cycle finished
        // between minors frees reachable old->old targets.
        self.rescan_remembered_black_and_drain();
        // Barrier-less stack-root stores (JitFrame gcmap slots, blackhole
        // register banks, resume-construction roots): re-gray the black roots
        // before the sweep freezes survivors.
        self.rescan_major_stack_roots_black_and_drain();
        // incminimark.py:2478-2481: process-global/non-stack roots can grow
        // after the cycle's initial snapshot.  Rescan and trace them before
        // finalizers, weakrefs, and sweep inspect VISITED.
        self.rescan_major_nonstack_roots_and_drain();
        // A young finalizer object survives the minor this cycle skips.
        // Repark first so the seed below covers a young leftover already on
        // `old_objects_with_finalizers`; `deal_with_objects_with_finalizers`
        // then sees only old objects, as incminimark does after a minor.
        if self.oldgen_nonmoving_active {
            self.repark_young_finalizer_registrations();
            self.nonmoving_major_trace_young_finalizers();
        }
        // incminimark.py:2486-2487 — the P list is a root source like any
        // other, and it is consulted here, after the ordinary roots and before
        // anything reads VISITED to decide what dies.
        if self.rrc.enabled {
            // Before the trace, not inside it: gathering the edges enters the
            // extension.
            self.rrc_refresh_c_edges();
            self.rrc_major_collection_trace();
            if self.oldgen_nonmoving_active {
                self.rrc_nonmoving_major_trace_young();
            }
        }
        self.mark_conditional_edges_to_fixed_point();
        // Before anything reads VISITED to decide what dies, and after the
        // fixed point has settled which mirrors those are.
        if self.rrc.enabled && self.rrc_claim_finalizers() > 0 {
            self.mark_conditional_edges_to_fixed_point();
        }
        // incminimark.py:2492: this counter belongs to one finalization-order
        // pass.  `_bump_finalization_state_from_0_to_1` below repopulates it
        // with every otherwise-dead object retained for a finalizer.
        self.kept_alive_by_finalizer = 0;
        // incminimark.py:2961-2965 (and :2495-2499) — clear weak
        // pointers to dying objects before the sweep frees them. The
        // VISITED bit on every old-gen object is still meaningful at
        // this point; sweep below tears down `VISITED` per object as
        // it walks oldgen.
        if !self.old_objects_with_finalizers.is_empty() {
            self.deal_with_objects_with_finalizers();
        } else if !self.old_objects_with_weakrefs.is_empty() {
            self.invalidate_old_weakrefs();
        }
        if self.oldgen_nonmoving_active && !self.young_objects_with_weakrefs.is_empty() {
            self.invalidate_young_weakrefs_for_nonmoving_major();
        }
        // A non-moving major (do_collect_oldgen_nonmoving) runs with a live
        // nursery and no preceding minor, so the write barrier's
        // old_objects_pointing_to_young / old_objects_with_cards_set are still populated and
        // may name old objects this cycle is about to free. Drop the dead ones
        // (VISITED is still set on every survivor at this point) so the next
        // minor does not trace freed memory. A no-op for do_collect_full, whose
        // leading minor already drained both sets.
        self.old_objects_pointing_to_young
            .retain(|&addr| unsafe { (*header_of(addr)).has_flag(GcFlags::GCFLAG_VISITED) });
        self.old_objects_with_cards_set
            .retain(|&addr| unsafe { (*header_of(addr)).has_flag(GcFlags::GCFLAG_VISITED) });
        // Embedder side tables keyed by owner address hold their value as a
        // root, so an entry outlives its owner unless it is dropped here —
        // the same "the target died" question `invalidate_old_weakrefs` just
        // answered, asked on behalf of a table the collector cannot see.
        // An owner outside old-gen is either immortal (`malloc_typed`, no
        // header to read) or, under a non-moving major, still in the live
        // nursery; neither can be proven dead here, so both are kept.
        let mut classify_owner = |owner: usize| -> Option<usize> {
            if owner == 0 || !self.oldgen.contains(owner) {
                return Some(owner);
            }
            let hdr = unsafe { header_of(owner) };
            if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_VISITED) } {
                Some(owner)
            } else {
                None
            }
        };
        crate::shadow_stack::prune_ephemeron_tables(&mut classify_owner);
        // The same question for tables a single mutator owns in its own TLS.
        // Those cannot go through the global registration: it names no thread,
        // so a major driven here would leave every other mutator's dead-owner
        // entries pinned. Reach exactly as far as this collection's own root
        // walk did (`enumerate_root_walker_values`) — foreign TLS only while
        // this thread owns STW.
        if crate::gc_sync::mutators_quiesced() {
            crate::shadow_stack::prune_all_mutator_areas(&mut classify_owner);
        } else {
            crate::shadow_stack::prune_my_mutator_areas(&mut classify_owner);
        }
        // incminimark.py:2510-2511 — run destructors of dying old objects
        // before the sweep frees them (VISITED still distinguishes
        // survivors from the dying at this point). A non-moving major also
        // owes the minor's young-rawmalloc destructor pass: that minor did
        // not run, and the blocks are released after this seam.
        if self.oldgen_nonmoving_active {
            self.drop_ignored_young_rawmalloc_finalizers_nonmoving();
            self.deal_with_young_rawmalloc_destructors_nonmoving();
        }
        if !self.old_objects_with_destructors.is_empty() {
            self.deal_with_old_objects_with_destructors();
        }
        // incminimark.py:2512-2514. ArenaCollection's prepare moves active
        // pages to old_* lists, and OldGen swaps old_rawmalloced_objects into
        // raw_malloc_might_sweep. Promotions between sweep steps therefore
        // allocate into fresh lists and are not candidates in this cycle.
        //
        // incminimark.py `debug_check_consistency` asserts the young
        // raw-malloc list is empty by now, which holds for every major that ran
        // a minor first. `do_collect_oldgen_nonmoving` runs no minor — the
        // interpreter has no shadow-stack pass, so a minor there would relocate
        // nursery objects held only on the Rust stack — and marks the young
        // block in place instead, exactly as it marks the nursery in place.
        debug_assert!(
            self.oldgen_nonmoving_active || !self.oldgen.has_young_rawmalloced(),
            "young raw-malloced objects in a major collection"
        );
        // Finalizers and destructors run while still MARKING and may store
        // into a live black object. The write barrier re-greys it onto
        // `more_gray_stack`; drain that work before the sweep freezes
        // VISITED (`visit_all_objects` until both stacks are empty).
        self.drain_marking_worklists();
        if self.config.debug >= 2 {
            self.debug_check_visited_children_not_white("marking_sweep_seam");
        }
        self.oldgen.sweep_prepare();

        // incminimark.py: dead old parents must leave the pin-parent
        // stack before arena sweep invalidates their addresses. Survivors have
        // VISITED set at this point; the oldgen sweep clears it later.
        self.old_objects_pointing_to_pinned
            .retain(|&obj_addr| unsafe {
                (*header_of(obj_addr)).has_flag(GcFlags::GCFLAG_VISITED)
            });
        self.updated_old_objects_pointing_to_pinned = true;
        // incminimark.py:2528-2529 — VISITED still distinguishes survivors, so
        // this is where each mirror learns whether its object made it.
        if self.rrc.enabled {
            self.rrc_major_collection_free();
            if self.oldgen_nonmoving_active {
                self.rrc_nonmoving_reclaim_young_rawmalloc();
            }
        }
        self.oldgen_nonmoving_marked = self.oldgen_nonmoving_active;
        // incminimark.py:2531-2532 — snapshot the pre-sweep accounting after
        // the candidate sets have been frozen and before the state changes.
        self.stat_ac_arenas_count = self.oldgen.arenas_count();
        self.stat_rawmalloced_total_size = self.oldgen.rawmalloced_bytes();
        self.gc_state = GcState::Sweeping;
    }

    /// incminimark.py STATE_SWEEPING.
    fn incremental_sweep_step(&mut self) {
        debug_assert_eq!(self.gc_state, GcState::Sweeping);
        let done = if self.oldgen.rawmalloc_sweep_pending() {
            // incminimark.py:2537-2547: process a bounded number of rawmalloc
            // objects first. Even if this drains the stack, the arena half is
            // deliberately deferred to the next state-machine step.
            let limit = 3 * self.config.nursery_size / self.oldgen.small_request_threshold();
            // Upstream production geometry makes this non-zero. Pyre unit
            // tests use nurseries smaller than one rawmalloc threshold, so one
            // candidate is the minimum needed for forward progress while
            // retaining the incminimark.py:2543 upper-bound formula otherwise.
            self.oldgen.sweep_rawmalloc_step(limit.max(1));
            false
        } else {
            // incminimark.py: visit at most three nursery sizes of
            // frozen arena pages.
            let limit = 3 * self.config.nursery_size / self.oldgen.page_size();
            // As above, tiny test nurseries need one-page forward progress;
            // normal configurations use the literal incminimark.py:2553 value.
            self.oldgen.sweep_arenas_step(limit.max(1))
        };

        if done {
            self.finish_incremental_cycle();
        }
    }

    /// incminimark.py:2560-2621: complete SWEEPING, compute the next threshold
    /// from post-sweep accounting, then enter FINALIZING.
    fn finish_incremental_cycle(&mut self) {
        debug_assert_eq!(self.gc_state, GcState::Sweeping);
        debug_assert!(!self.oldgen.rawmalloc_sweep_pending());
        self.major_collections += 1;
        if crate::majit_log_enabled() {
            eprintln!(
                "[gc][major] complete count={} oldgen_bytes={}",
                self.major_collections,
                self.get_total_memory_used(),
            );
        }

        // incminimark.py:2563-2564: prebuilt objects are outside the arena
        // sweep, so reset their mark bit explicitly for the next cycle.
        for &addr in &self.prebuilt_root_objects {
            unsafe { (*header_of(addr)).clear_flag(GcFlags::GCFLAG_VISITED) };
        }
        // The non-moving major skipped the minor, whose last act is
        // `free_young_rawmalloced_objects`. Reachable young rawmallocs were
        // greyed in place (`GCFLAG_VISITED`) by the mark seam that set
        // `oldgen_nonmoving_marked`. Release the unvisited ones before the
        // threshold below, or the next cycle treats those dead bytes as
        // survivors and ratchets upward. Survivors stay young. A cycle that
        // was already sweeping when the non-moving entry began did not mark
        // them, so it must not release them.
        if self.oldgen_nonmoving_marked && self.oldgen.has_young_rawmalloced() {
            self.oldgen.free_unvisited_young_rawmalloced_keep_young();
        }
        self.oldgen_nonmoving_marked = false;
        // incminimark.py:2566-2577 — set the threshold for the next major
        // collection to `major_collection_threshold` times the surviving
        // size, but no more than `max_delta` above it, floored at
        // `min_heap_size` by set_major_threshold_from. incminimark.py
        // subtracts objects that survived only so their finalizers can run;
        // clamp at zero exactly as upstream does after the subtraction.
        let total_memory_used = self
            .get_total_memory_used()
            .saturating_sub(self.kept_alive_by_finalizer) as f64;
        // incminimark.py:2574-2577 — capped next-major threshold. `reserving_size`
        // is the byte size of the allocation that triggered this collection,
        // carried on the collector across `do_collect_nursery` (see
        // `pending_reserving_size`), matching the argument
        // `major_collection_step(reserving_size)` threads to both
        // `set_major_threshold_from` and `threshold_reached`.
        let reserving_size = self.pending_reserving_size;
        let bounded = self.set_major_threshold_from(
            (total_memory_used * self.major_collection_threshold)
                .min(total_memory_used + self.max_delta),
            reserving_size as f64,
        );
        self.bytes_made_old_since_cycle = 0;
        self.threshold_bytes_made_old = 0;

        // incminimark.py:2592-2600 — report post-sweep accounting after the
        // next threshold is computed, but before the max-heap check and the
        // transition into FINALIZING.
        self.hooks.fire_gc_collect(
            self.major_collections,
            self.stat_ac_arenas_count,
            self.oldgen.arenas_count(),
            self.oldgen.arenas_bytes(),
            self.stat_rawmalloced_total_size,
            self.oldgen.rawmalloced_bytes(),
            self.pinned_objects_in_nursery,
        );

        // incminimark.py:2601-2615 — max heap size (PYPY_GC_MAX). If the capped
        // threshold was bounded by `max_heap_size` and the heap has already
        // reached it, signal out-of-memory: the first time so the program gets
        // a chance to quit cleanly, and on a second occurrence by aborting the
        // process (`out_of_memory` -> fatalerror). `max_heap_size == 0`
        // (unbounded default) never sets `bounded`, so this is inert unless a
        // limit was set.
        //
        // Upstream signals it by raising, which unwinds through whichever of
        // the many drivers of a collection is on the stack. Pyre has to pick a
        // channel, because only one of those drivers can carry an exception,
        // and `reserving_size` already says which case this is: it is the size
        // of the allocation waiting on this collection, and it is nonzero for
        // exactly the two collecting nursery allocators.
        //
        //   * With an allocation waiting, fail it. `oom_pending` is read by
        //     that allocator the moment the collection returns, and its NULL
        //     becomes a `MemoryError` at the compiled `CHECK_MEMORY_ERROR`.
        //   * With none waiting — an explicit `gc.collect()`, a finalizer run,
        //     and above all the dispatch-loop safepoint, which is where the
        //     interpreter path's collections happen — there is nothing to fail
        //     and no frame that can raise, so the eval-breaker bit defers the
        //     exception to the next bytecode dispatch, exactly as `set_gc`
        //     defers the collection itself.
        //
        // The two are mutually exclusive on purpose. Arming both would leave
        // whichever channel went undelivered latched onto the *next* breach:
        // an `oom_pending` no allocation was waiting for is taken by the next
        // unrelated one, failing an allocation that breached nothing.
        //
        // The fatal rung asks whether the program has already had its
        // `MemoryError`. Upstream raises here, so setting the flag and handing
        // the program its exception are one event and the question answers
        // itself. Deferring delivery pulls them apart, and the two halves of
        // that are asked separately:
        //
        //   * *This* breach repeats one this thread has not been given yet —
        //     `do_collect_full` completes two cycles per call with no dispatch
        //     between them, so an explicit `gc.collect()` on a heap at its
        //     limit reaches exactly that. Nothing new has happened, so return.
        //     The question is per-thread: an exception owed to another thread
        //     says nothing about this breach, and treating it as this one's
        //     would let this thread allocate past the limit unremarked.
        //   * The *fatal* rung needs the exception to have landed, not merely
        //     to have been armed, or it ends the process before the program
        //     sees what the rung above promised it. So a breach while any
        //     thread is still owed one falls through and arms this thread too,
        //     which is what upstream's raise does for whichever thread ran the
        //     collection.
        //
        // Both channels clear as they deliver, so an armed one is the whole
        // test in either case.
        if bounded && self.threshold_reached(reserving_size) {
            if self.oom_pending || majit_ir::eval_breaker_word::memory_error_owed_here() {
                // Upstream raises again here, and never reaches the collect-step
                // hook; this return stands for that raise, so suppress it too.
                self.oom_signalled_this_step = true;
                self.gc_state = GcState::Scanning;
                return;
            }
            if self.max_heap_size_already_raised
                && !majit_ir::eval_breaker_word::memory_error_armed()
            {
                panic!("using too much memory, aborting");
            }
            self.max_heap_size_already_raised = true;
            self.oom_signalled_this_step = true;
            if reserving_size > 0 {
                self.oom_pending = true;
            } else {
                majit_ir::eval_breaker_word::set_memory_error();
            }
            // incminimark.py: STATE_SCANNING then an
            // immediate `raise MemoryError` exits `major_collection_step`
            // before the finalizing phase. Return before the queue-notification
            // triggers so none fire ahead of the `MemoryError` the two channels
            // above will raise.
            self.gc_state = GcState::Scanning;
            return;
        }

        self.gc_state = GcState::Finalizing;
    }

    /// incminimark.py `invalidate_old_weakrefs(self)`.
    ///
    /// For each old-gen WEAKREF recorded in `old_objects_with_weakrefs`:
    ///   * If the weakref struct itself was not marked → it dies; drop
    ///     it from the list (no slot mutation, sweep will reclaim it).
    ///   * Else read the `weakptr` slot and check the target's VISITED
    ///     bit. Live target → keep the weakref in a fresh list.
    ///     Dying target → null the slot and drop the weakref.
    ///
    /// incminimark.py also treats FINALIZATION_ORDERING targets as
    /// dying for weakref purposes even though finalizer-queue reachability has
    /// marked them VISITED; the check below preserves that ordering.
    fn invalidate_old_weakrefs(&mut self) {
        let entries = std::mem::take(&mut self.old_objects_with_weakrefs);
        let mut new_with_weakref = Vec::with_capacity(entries.len());
        for obj_addr in entries {
            let hdr_ptr = (obj_addr - GcHeader::SIZE) as *const GcHeader;
            // incminimark.py:3112: weakref itself not marked → dies.
            if unsafe { !(*hdr_ptr).has_flag(GcFlags::GCFLAG_VISITED) } {
                continue;
            }
            let weakptr_slot = (obj_addr + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef;
            let pointing_to = unsafe { (*weakptr_slot).0 };
            if pointing_to == 0 {
                continue;
            }
            // A target that is not an old-gen object is immortal
            // (`malloc_typed`): it carries no GcHeader to read a VISITED bit
            // from and never dies, so the weakref stays valid. Keep tracking
            // it with the slot intact, mirroring the foreign-target handling
            // in `invalidate_young_weakrefs`. `alloc_in_oldgen` registers
            // born-old weakrefs whose target may be such an immortal object
            // (e.g. a weakref to a type), so this is reachable; the nursery is
            // empty at this point, so any non-old-gen target is immortal.
            if !self.oldgen.contains(pointing_to) {
                new_with_weakref.push(obj_addr);
                continue;
            }
            // incminimark.py:3120-3121: queued finalizers carry VISITED so
            // they survive this cycle, but FINALIZATION_ORDERING still makes
            // their weakrefs observe death before `__del__` runs.
            let target_hdr = (pointing_to - GcHeader::SIZE) as *const GcHeader;
            if unsafe {
                (*target_hdr).has_flag(GcFlags::GCFLAG_VISITED)
                    && !(*target_hdr).has_flag(GcFlags::GCFLAG_FINALIZATION_ORDERING)
            } {
                new_with_weakref.push(obj_addr);
            } else {
                unsafe { (*weakptr_slot).0 = 0 };
            }
        }
        self.old_objects_with_weakrefs = new_with_weakref;
    }

    fn finalization_state(&self, obj_addr: usize) -> u8 {
        let hdr = unsafe { header_of(obj_addr) };
        let visited = unsafe { (*hdr).has_flag(GcFlags::GCFLAG_VISITED) };
        let ordering = unsafe { (*hdr).has_flag(GcFlags::GCFLAG_FINALIZATION_ORDERING) };
        match (visited, ordering) {
            (false, false) => 0,
            (false, true) => 1,
            (true, true) => 2,
            (true, false) => 3,
        }
    }

    /// Return the GC-managed outgoing references of one object.  This is the
    /// collector `trace()` callback shape used by incminimark's finalization
    /// ordering walk, kept separate from normal VISITED marking.
    fn finalizer_children(&self, obj_addr: usize) -> Vec<usize> {
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        let info = self.types.get(type_id);
        let mut children = Vec::new();
        if let Some(trace_fn) = info.custom_trace {
            unsafe {
                trace_fn(obj_addr, &mut |slot_ptr: *mut GcRef| {
                    let child = *slot_ptr;
                    if !child.is_null() && self.is_managed_heap_object(child.0) {
                        children.push(child.0);
                    }
                });
            }
        }
        for &offset in &info.gc_ptr_offsets {
            let child = unsafe { *((obj_addr + offset) as *const GcRef) };
            if !child.is_null() && self.is_managed_heap_object(child.0) {
                children.push(child.0);
            }
        }
        if info.items_have_gc_ptrs && info.item_size > 0 {
            // The type id was read from this header. A forwarded header's
            // type id is outside the table, so `types.get` does not reach
            // a corpse; `get_possibly_forwarded_type_id` is the path that
            // would have substituted the copy first.
            let length = unsafe { *((obj_addr + info.length_offset) as *const usize) };
            let items_start = obj_addr + info.size;
            for i in 0..length {
                let item = items_start + i * info.item_size;
                for &offset in &info.var_gc_ptr_offsets {
                    let child = unsafe { *((item + offset) as *const GcRef) };
                    if !child.is_null() && self.is_managed_heap_object(child.0) {
                        children.push(child.0);
                    }
                }
            }
        }
        children
    }

    fn recursively_clear_finalization_ordering(&mut self, obj_addr: usize) {
        let mut pending = vec![obj_addr];
        while let Some(addr) = pending.pop() {
            if self.finalization_state(addr) == 2 {
                unsafe { (*header_of(addr)).clear_flag(GcFlags::GCFLAG_FINALIZATION_ORDERING) };
                pending.extend(self.finalizer_children(addr));
            }
        }
    }

    /// How many objects are registered with a finalizer queue —
    /// `deal_with_objects_with_finalizers`' working set, young and old.
    ///
    /// The destructor lists are deliberately not counted. A destructor is
    /// RPython's *light* finalizer (`rgc.must_be_light_finalizer`): it may not
    /// run interpreter code, allocate or resurrect, so nothing the program can
    /// observe distinguishes the collection it runs in from a later one. In
    /// pyre they are drop glue — `function_destructor`, `type_object_destructor`,
    /// `pycode_destructor`, `storage_box_destructor` — so every function, type
    /// and code object is on those lists and a predicate that counted them
    /// would never answer zero.
    pub(crate) fn registered_finalizer_count(&self) -> usize {
        self.probably_young_objects_with_finalizers.len()
            + self.old_objects_with_finalizers.len()
            + self.run_old_style_finalizers.len()
    }

    /// Whether a collection could still hand control back to the program.
    ///
    /// `deal_with_objects_with_finalizers` is the one pass that does, and it is
    /// already guarded on its own queue being non-empty; `rawrefcount` answers
    /// for the whole of itself, because a collection there can claim a mirror's
    /// C finalizer and queue its deallocator, and which of its lists that
    /// reaches is the embedder's business rather than this predicate's. It is
    /// only ever enabled under `cpyext`.
    pub(crate) fn has_pending_death_callbacks(&self) -> bool {
        self.registered_finalizer_count() != 0 || self.rawrefcount_enabled()
    }

    /// A read-only reachability prepass for prompt-finalization boundaries.
    ///
    /// CPython releases a generator frame's locals by refcount, so only the
    /// graph behind those released edges can produce an immediate `__del__`.
    /// Pyre must collect to discover whether those objects are otherwise
    /// unreachable, but it need not collect when that graph contains no
    /// registered finalizer at all.  Walk the same traced edges as marking and
    /// use inspector.py's `GCFLAG_EXTRA` visited bit, restoring every bit
    /// before returning.  An enabled rawrefcount bridge keeps the sweep only
    /// when a reached object itself has a mirror — a process-wide rrc flag
    /// is not a per-graph witness.
    pub fn do_subgraph_has_pending_finalizer(&self, roots: &[GcRef]) -> bool {
        // A process-wide rawrefcount bridge is not a witness that *this*
        // released graph can run a death callback.  Walking the same edges
        // as marking, an enabled rrc only keeps the sweep when a reached
        // object itself has a mirror (`rawrefcount_from_obj`).
        if self.registered_finalizer_count() == 0 && !self.rawrefcount_enabled() {
            return false;
        }

        let mut pending = Vec::new();
        for &root in roots {
            self.heap_dump_add(root, &mut pending);
        }
        let marked_roots = pending.clone();
        let mut found = false;
        while let Some(obj) = pending.pop() {
            let hdr = unsafe { header_of(obj.0) };
            if unsafe {
                (*hdr).has_flag(GcFlags::FINALIZER_REGISTERED)
                    && !(*hdr).has_flag(GcFlags::FINALIZER_RUN)
                    && !(*hdr).has_flag(GcFlags::GCFLAG_IGNORE_FINALIZER)
            } {
                found = true;
            } else if self.rawrefcount_enabled() && self.rawrefcount_from_obj(obj.0) != 0 {
                found = true;
            }
            self.visit_referents(obj.0, &mut |child| self.heap_dump_add(child, &mut pending));
        }
        self.heap_dump_clear_gcflag(marked_roots);
        found
    }

    /// incminimark.py `deal_with_objects_with_finalizers`.
    fn deal_with_objects_with_finalizers(&mut self) {
        let mut new_with_finalizer = VecDeque::new();
        let mut marked = VecDeque::new();

        while let Some((obj_addr, fq_index)) = self.old_objects_with_finalizers.pop_front() {
            // The non-moving pass has no leading minor. A young leftover on
            // this deque belongs on `probably_young_objects_with_finalizers`
            // for `deal_with_young_objects_with_finalizers`, not on a death
            // queue as a nursery address. `repark_young_finalizer_registrations`
            // already moved these before the young-finalizer seed; keep the
            // same rule here so this pass cannot reintroduce one.
            if self.oldgen_nonmoving_active
                && (self.is_in_nursery(obj_addr) || self.is_young_rawmalloced(obj_addr))
            {
                self.probably_young_objects_with_finalizers
                    .push_back((obj_addr, fq_index));
                continue;
            }
            let hdr = unsafe { header_of(obj_addr) };
            if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_IGNORE_FINALIZER) } {
                continue;
            }
            if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_VISITED) } {
                new_with_finalizer.push_back((obj_addr, fq_index));
                continue;
            }

            marked.push_back((obj_addr, fq_index));
            let mut pending = vec![obj_addr];
            while let Some(addr) = pending.pop() {
                match self.finalization_state(addr) {
                    0 => {
                        unsafe {
                            (*header_of(addr)).set_flag(GcFlags::GCFLAG_FINALIZATION_ORDERING)
                        };
                        // incminimark.py:3011-3020
                        //
                        // Upstream reaches this seam only after a minor has
                        // emptied the nursery, so every `addr` contributes to
                        // `get_total_memory_used()`. Pyre's non-moving
                        // interpreter major deliberately skips that minor and
                        // may trace a young child here. The threshold metric
                        // likewise excludes nursery bytes, so subtract only
                        // the old-gen subset of the otherwise-dead graph.
                        if self.oldgen.contains(addr) {
                            self.kept_alive_by_finalizer = self
                                .kept_alive_by_finalizer
                                .wrapping_add(OldGen::allocation_size(
                                    self.object_total_size(addr),
                                ));
                        }
                        pending.extend(self.finalizer_children(addr));
                    }
                    2 => self.recursively_clear_finalization_ordering(addr),
                    _ => {}
                }
            }

            // `_recursively_bump_finalization_state_from_1_to_2`: enqueue the
            // root in normal marking and visit its complete object graph.
            self.seed_major_root(GcRef(obj_addr), "finalization_ordering_root");
            while let Some(addr) = self.incr_state.gray_stack.pop() {
                self.mark_object(addr);
            }
        }

        // Resurrecting a finalizer's object graph marks owners that the
        // caller's ephemeron pass had already written off, and an
        // address-keyed side table is not reached by the tracing that just
        // marked them — `_recursively_bump_finalization_state_from_1_to_2`
        // reaches a lifeline only because upstream stores it in an ordinary
        // field. Re-establish those edges here: `prune_ephemeron_tables` reads
        // the same VISITED bits below and would otherwise keep an entry whose
        // value nothing marked, leaving the table pointing into swept memory.
        self.mark_conditional_edges_to_fixed_point();

        // PyPy clears weakrefs while queued objects are in state 2.
        if !self.old_objects_with_weakrefs.is_empty() {
            self.invalidate_old_weakrefs();
        }

        while let Some((obj_addr, fq_index)) = marked.pop_front() {
            if self.finalization_state(obj_addr) == 2 {
                if fq_index == -1 {
                    self.run_old_style_finalizers.push_back(obj_addr);
                } else {
                    self.finalizer_handlers[fq_index as usize]
                        .deque
                        .push_back(obj_addr);
                }
                self.recursively_clear_finalization_ordering(obj_addr);
            } else {
                new_with_finalizer.push_back((obj_addr, fq_index));
            }
        }
        self.old_objects_with_finalizers = new_with_finalizer;
    }

    fn execute_finalizer_triggers(&mut self) {
        if self.finalizer_lock {
            return;
        }
        self.finalizer_lock = true;
        for handler in &self.finalizer_handlers {
            if !handler.deque.is_empty() {
                (handler.trigger)();
            }
        }
        // GCBase.execute_finalizers runs old-style callbacks only after every
        // new-style queue trigger. Pop before calling so a nested collection
        // cannot invoke the same object twice. PyPy's GC transform keeps the
        // popped `obj` local alive across `call_destructor`; register the Rust
        // local explicitly for that same re-entrant-collection contract.
        while let Some(obj_addr) = self.run_old_style_finalizers.pop_front() {
            let mut obj = GcRef(obj_addr);
            unsafe { self.roots.add(&mut obj) };
            let type_id = unsafe { (*header_of(obj.0)).type_id() };
            if (type_id as usize) < self.types.len()
                && let Some(finalizer) = self.types.get(type_id).old_style_finalizer
            {
                unsafe { finalizer(obj.0) };
            }
            self.roots.remove(&mut obj);
        }
        self.finalizer_lock = false;
    }

    /// incminimark.py `deal_with_old_objects_with_destructors`.
    ///
    /// Walk the old-destructor list: a VISITED (surviving) object is kept
    /// in a fresh list; a dying (not-VISITED) object's destructor runs
    /// before the imminent sweep frees its memory. Must run while VISITED
    /// bits are still meaningful — i.e. before incremental sweep steps clear
    /// them.
    fn deal_with_old_objects_with_destructors(&mut self) {
        let entries = std::mem::take(&mut self.old_objects_with_destructors);
        let mut new_objects = Vec::with_capacity(entries.len());
        for obj_addr in entries {
            let hdr_ptr = (obj_addr - GcHeader::SIZE) as *const GcHeader;
            if unsafe { (*hdr_ptr).has_flag(GcFlags::GCFLAG_VISITED) } {
                // surviving
                new_objects.push(obj_addr);
            } else {
                // dying
                self.run_destructor(obj_addr);
            }
        }
        self.old_objects_with_destructors = new_objects;
    }

    /// Whether an incremental marking cycle is currently in progress.
    pub fn is_incremental_marking(&self) -> bool {
        self.gc_state == GcState::Marking
    }

    /// Number of objects marked so far in the current incremental cycle.
    pub fn incremental_objects_marked(&self) -> usize {
        self.incr_state.objects_marked
    }

    /// incminimark.py `enable` / `disable` / `isenabled`.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn isenabled(&self) -> bool {
        self.enabled
    }

    /// Set the per-step marking budget in bytes.
    pub fn set_mark_budget(&mut self, budget: usize) {
        self.incr_state.mark_budget_per_step = budget;
    }

    /// Enable or disable `gc_stress` forced-collection-per-allocation for this
    /// instance. Only present under the `gc_stress` feature; callers gate use
    /// with `#[cfg(feature = "gc_stress")]`.
    #[cfg(feature = "gc_stress")]
    pub fn set_stress_collect(&mut self, on: bool) {
        self.stress_collect = on;
    }

    /// Perform a full (major) mark-sweep collection.
    ///
    /// 1. First do a minor collection to promote all live nursery objects.
    /// 2. Mark phase: trace all roots and transitively mark reachable objects.
    /// 3. Sweep phase: free all unmarked old-gen objects.
    pub fn do_collect_full(&mut self) {
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        // incminimark.py `minor_and_major_collection` /
        // `gc_step_until`: first finish the current major cycle, if any, with
        // a minor collection before every major step.  Besides discovering
        // new roots, that minor forwards every nursery edge held by the
        // in-progress marking state before the next gray object is consumed.
        self.gc_step_until_scanning_with_minors();

        // Minor collection first to empty the nursery.
        // This is the `_minor_collection()` performed by upstream's
        // `gc_step_until(STATE_MARKING)` before it starts the fresh cycle.
        // Upstream calls the low-level `_minor_collection`, which has no
        // major-progress or rawrefcount-callback tail.
        self.minor_collection_body();

        // The rest of that same `gc_step_until(STATE_MARKING)` iteration.
        // Reaching MARKING through `major_collection_step` is what runs the
        // STATE_SCANNING arm (incminimark.py), which resets
        // `size_objects_made_old` / `threshold_objects_made_old` before
        // `collect_roots`; calling `start_incremental_cycle` directly opened
        // the cycle with the finishing cycle's counters still in place.
        if self.gc_state == GcState::Scanning {
            self.major_collection_step();
            self.full_collect_owns_cycle = self.gc_state == GcState::Marking;
        }
        self.gc_step_until_scanning_with_minors();
        self.full_collect_owns_cycle = false;

        // incminimark.py:808.
        self.rrc_invoke_callback();
        self.rrc_invoke_cpyext_drain();
    }

    /// incminimark.py `minor_collection_with_major_progress`: "Do a minor
    /// collection.  Then, if the GC is enabled and there is already a major GC
    /// in progress, run at least one major collection step.  If there is no
    /// major GC but the threshold is reached, start a major GC."
    ///
    /// `do_collect_nursery` is that function with `force_enabled=False`.
    /// Explicit `collect(0/1)` passes `force_enabled=True` so major progress
    /// still runs while automatic progress is switched off, without flipping
    /// `enabled`.
    fn minor_collection_with_major_progress(&mut self, force_enabled: bool) {
        if !force_enabled {
            self.do_collect_nursery();
            return;
        }
        self.minor_collection_body();
        self.run_major_progress_after_minor(true);
        self.rrc_invoke_callback();
        self.rrc_invoke_cpyext_drain();
    }

    /// incminimark.py `collect(gen=2)`: "Do a minor (gen=0), start a major
    /// (gen=1), or do a full major (gen>=2) collection."
    ///
    /// The app-level `gc.collect(n)` does not reach this — `interp_gc.py
    /// collect` unwraps its generation and then asks for a full collection
    /// whatever it was.  This is the internal `rgc.collect(gen)` entry, whose
    /// callers want a named amount of work rather than all of it.
    pub fn do_collect(&mut self, generation: i64) {
        if generation >= 2 {
            // `minor_and_major_collection`, which ends with the same callback.
            self.do_collect_full();
            return;
        }
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        if generation < 0 {
            // "Dangerous! this makes no progress on the major GC cycle.  If
            // called too often, the memory usage will keep increasing, because
            // we'll never completely fill the nursery (and so never run
            // anything about the major collection)."
            self.minor_collection_body();
        } else {
            self.minor_collection_with_major_progress(true);
            // gen == 1 is gen == 0 plus "if no major collection is running,
            // then it forces one to start now".
            if generation == 1 && self.gc_state == GcState::Scanning {
                self.major_collection_step();
            }
        }
        self.rrc_invoke_callback();
        self.rrc_invoke_cpyext_drain();
    }

    /// incminimark.py `set_max_heap_size`.  `PYPY_GC_MAX` is read once at
    /// startup; this is the same limit set from inside a running process, and
    /// it pulls the pending major-collection thresholds down with it so a limit
    /// lowered below them takes effect before the next major rather than after.
    pub fn set_max_heap_size(&mut self, size: usize) {
        self.max_heap_size = size as f64;
        if self.max_heap_size > 0.0 {
            if self.max_heap_size < self.next_major_collection_initial {
                self.next_major_collection_initial = self.max_heap_size;
            }
            if self.max_heap_size < self.next_major_collection_threshold {
                self.next_major_collection_threshold = self.max_heap_size;
            }
        }
    }

    /// incminimark.py `max_heap_size_already_raised`, read from outside this
    /// module.  See `crate::gc_max_heap_already_raised` for what the answer is
    /// used to decide.
    pub fn max_heap_already_raised(&self) -> bool {
        self.max_heap_size_already_raised
    }

    /// incminimark.py `collect_step`.
    pub fn collect_step(&mut self) -> crate::GcStepTransition {
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        let old_state = self.gc_state;

        // `_minor_collection`, not `minor_collection_with_major_progress`:
        // the explicit `major_collection_step` below is the one and only
        // state transition performed by this call.
        self.minor_collection_body();
        self.major_collection_step();

        // incminimark.py:821.
        self.rrc_invoke_callback();
        self.rrc_invoke_cpyext_drain();

        crate::GcStepTransition {
            old_state: old_state.encoded(),
            new_state: self.gc_state.encoded(),
        }
    }

    /// Reclaim dead old-gen objects WITHOUT moving the nursery.
    ///
    /// A non-moving major: seed roots, mark transitively, and sweep only the
    /// old generation — skipping the leading minor that [`do_collect_full`]
    /// runs. The nursery is left byte-for-byte intact (not moved, not freed),
    /// so an unrooted Rust-stack `PyObjectRef` into the nursery cannot dangle
    /// — the exact hazard that blocks a moving minor at an interp safepoint
    /// where there is no shadowstack pass. Reachability stays exact because
    /// `seed_major_root` / `mark_object` gate on `is_managed_heap_object`,
    /// which INCLUDES the nursery, so an `old -> nursery -> old` live edge is
    /// fully followed and the target old object is marked before the sweep.
    ///
    /// Unlike a moving minor, nursery survivors keep their addresses, so a
    /// `GcFlags::GCFLAG_VISITED` set on a marked-through nursery object would otherwise
    /// survive into the next minor's promoted copy (`copy_nursery_object`
    /// memcpys the header verbatim). The strictly-last step clears VISITED on
    /// exactly the nursery objects greyed this cycle — after
    /// sweep completion, hence after `invalidate_old_weakrefs`,
    /// which reads a nursery target's VISITED to decide weakref survival. The
    /// remembered set is left untouched: a non-moving major does not consume
    /// `old -> young` edges, so the next minor still finds them.
    pub fn do_collect_oldgen_nonmoving(&mut self) {
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        self.oldgen_nonmoving_active = true;
        self.oldgen_nonmoving_young_marks.clear();

        // This entry skips the leading minor, so the MARKING arm of
        // `_minor_collection` (`_add_to_more_objects_to_trace_if_black` on
        // the remembered set) never runs. Black old objects from the
        // interrupted incremental cycle already had their children written
        // since the last visit. Re-grey those remembered parents so the rest
        // of this cycle traces those edges (`visit` / `mark_object`).
        if self.gc_state == GcState::Marking {
            self.regray_remembered_black_during_marking();
        }

        // `do_collect_full` / incminimark.py `gc_step_until` first finishes
        // an in-progress major, then starts a fresh cycle whose root snapshot
        // is taken at the explicit collection boundary.  The non-moving twin
        // owes the same ordering: merely finishing an older cycle can retain
        // an object which became unreachable after that cycle began.
        if self.gc_state != GcState::Scanning {
            self.gc_step_until_scanning();
            self.clear_oldgen_nonmoving_young_marks();
        }
        self.start_incremental_cycle();
        // Keep this oldgen-only entry stop-the-world: it may enter while
        // MARKING or SWEEPING, but always returns after the complete cycle.
        self.gc_step_until_scanning();

        // Strictly-last: clear VISITED on every young object greyed this cycle
        // (the oldgen sweep already cleared it on old-gen survivors).
        self.clear_oldgen_nonmoving_young_marks();
        self.oldgen_nonmoving_active = false;

        // This entry has no upstream counterpart, but it is a public collection
        // entry point and it can queue mirrors, so it owes the same schedule.
        self.rrc_invoke_callback();
        self.rrc_invoke_cpyext_drain();
    }

    /// Clear the nursery VISITED bits accumulated by one non-moving major.
    /// A fresh explicit cycle must begin with those bits clear or its marker
    /// will mistake the preceding cycle's young survivors for already-traced
    /// objects and skip their old-generation children.
    fn clear_oldgen_nonmoving_young_marks(&mut self) {
        let marks = std::mem::take(&mut self.oldgen_nonmoving_young_marks);
        for (addr, clear_track_young_ptrs) in marks {
            // Nothing moved and nothing young was freed, so each addr is still
            // where it was noted; the guard is defensive against a duplicate
            // already cleared.
            if self.is_in_nursery(addr) || self.is_young_rawmalloced(addr) {
                let hdr = unsafe { header_of(addr) };
                unsafe {
                    (*hdr).clear_flag(GcFlags::GCFLAG_VISITED);
                    if clear_track_young_ptrs {
                        (*hdr).clear_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
                    }
                };
            }
        }
    }

    fn gc_step_until_scanning(&mut self) {
        while self.gc_state != GcState::Scanning {
            self.major_collection_step();
        }
    }

    /// incminimark.py `gc_step_until`: explicit full collection
    /// performs a minor before every major state-machine step.  The ordinary
    /// non-moving oldgen entry deliberately uses [`gc_step_until_scanning`]
    /// instead because its contract is to leave the nursery byte-for-byte in
    /// place.
    fn gc_step_until_scanning_with_minors(&mut self) {
        while self.gc_state != GcState::Scanning {
            // Upstream calls the lower-level `_minor_collection` here and
            // then advances one explicit step.
            self.minor_collection_body();
            self.major_collection_step();
        }
    }

    /// incminimark.py write_barrier(addr_struct):
    /// if the object has GCFLAG_TRACK_YOUNG_PTRS, call remember_young_pointer.
    /// Nursery objects never have TRACK_YOUNG_PTRS, so the flag test alone
    /// selects the old-gen objects that may now point to young.
    ///
    /// Unlike incminimark — where every struct is GC-managed and the barrier
    /// only ever receives a header-bearing GC object — pyre runs a hybrid heap:
    /// host-side allocators (`w_list_new`, `w_dict_new`, …) fall back to a bare
    /// `Box::into_raw` when no GC hook is installed (bootstrap / import-time
    /// objects), producing PyObjects that are not in any managed generation and
    /// whose `obj - 8` word is Rust allocator metadata, not a `GcHeader`. Such
    /// an object legitimately reaches the interpreter barrier sites (a slice
    /// store on a bootstrap list, a namespace store on an import-time dict), so
    /// the barrier must ignore any address the GC does not own rather than read
    /// its non-header word. This centralizes the `try_gc_owns_object` guard that
    /// `object_array.rs` / `list_write_barrier` already apply per call site.
    ///
    /// The guard is equally load-bearing on the JIT path, which is why this
    /// entry point — not `jit_remember_young_pointer` — is what the backends'
    /// non-array COND_CALL_GC_WB helper calls for a JITFRAME base. An ordinary
    /// store's base takes `jit_remember_young_pointer` instead
    /// (`framework.py:538-544`), the unguarded entry the array barrier already
    /// reaches through `jit_remember_young_pointer_from_array`.
    /// `_reload_frame_if_necessary`
    /// (aarch64/assembler.py) re-applies the non-array barrier fast
    /// path to the *current jitframe* after every collecting helper call.
    /// Not every jitframe is nursery-allocated: the runner's entry frame, the
    /// realloc slowpath, and the JITFRAME nursery slowpath's fallback build
    /// frames off the GC — the class `shadow_stack::register_libc_jitframe`
    /// exists to track. Those go through `jitframe::alloc_off_gc_jitframe`,
    /// which reserves a zeroed header word so the inline test's read at
    /// `jit_wb_if_flag_byteofs` (negative, where a header would be) stays
    /// inside the allocation and finds the flag clear. Reaching this helper
    /// with such a frame therefore takes a set flag bit the allocator did not
    /// write; `is_managed_heap_object` is what still keeps the unmanaged block
    /// out of `old_objects_pointing_to_young`, where the next minor would decode a type id
    /// from it. `write_barrier_ignores_unmanaged_jitframe_with_flag_byte_set`
    /// pins that case.
    pub fn do_write_barrier(&mut self, obj: GcRef) {
        // incminimark's write_barrier receives a typed, non-null struct pointer.
        // pyre's GcRef is nullable (GcRef::NULL is the sentinel) and reaches the
        // safe `write_barrier`/`gc_write_barrier` entry points, so guard null
        // before reading `header_of(obj)`; the card variant guards it likewise.
        if obj.is_null() || self.is_in_nursery(obj.0) {
            return;
        }
        // incminimark `write_barrier` is the flag test alone. Both arms below
        // act only when this same word has `TRACK_YOUNG_PTRS` set, so a clear
        // bit answers for every family before the ownership question is
        // asked; a set bit still has to be owned or witnessed.
        if obj.0.is_multiple_of(GcHeader::ALIGN)
            && unsafe { !(*header_of(obj.0)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) }
        {
            return;
        }
        if self.is_managed_heap_object(obj.0) {
            let hdr = unsafe { header_of(obj.0) };
            if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) } {
                self.remember_young_pointer(obj);
            }
            return;
        }
        // `malloc_typed` bootstrap objects are incminimark's prebuilt family.
        // The witness is `registered_external_header`, not vtable membership
        // alone: `w_tuple_new` / `w_specialisedtuple_new_*` fall back to a bare
        // `Box::into_raw` when no GC is installed, and those objects carry the
        // very same registered `ob_header` at offset 0 with NO preceding
        // `GcHeader`. A vtable-only test therefore reads the allocator word in
        // front of the box and — if its `TRACK_YOUNG_PTRS` bit happens to be
        // set — has `remember_young_pointer` write flags back into that
        // metadata. The tid equality is what separates the two families.
        let Some(hdr) = self.registered_external_header(obj.0) else {
            return;
        };
        if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) } {
            self.remember_young_pointer(obj);
        }
    }

    /// `do_write_barrier` sibling for a pointer returned by this collector's
    /// allocation API.  RPython's barrier receives only GC-managed structs;
    /// the membership guard exists solely for pyre's mixed bootstrap heap and
    /// is redundant for a freshly allocated managed result.
    fn do_write_barrier_managed(&mut self, obj: GcRef) {
        if obj.is_null() {
            return;
        }
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) } {
            self.remember_young_pointer(obj);
        }
    }

    /// incminimark.py _remember_young_pointer_inlined(addr):
    /// append the object to the remembered set (`old_objects_pointing_to_young`)
    /// and clear GCFLAG_TRACK_YOUNG_PTRS. Callers have already verified the flag
    /// (the inline COND_CALL_GC_WB test, or `do_write_barrier`).
    /// `gcreftracer.py` `llop.gc_writebarrier(tr)`.
    pub fn remember_gc_table(&mut self, table: &Arc<crate::GcTable>) {
        self.pending_gc_tables.push(Arc::downgrade(table));
    }

    /// Drain this MiniMark's table remembered set. Same consume as
    /// `collect_oldrefs_to_nursery` on `old_objects_pointing_to_young`.
    fn collect_pending_gc_tables_to_nursery(&mut self) {
        let pending = std::mem::take(&mut self.pending_gc_tables);
        for table in pending.into_iter().filter_map(|w| w.upgrade()) {
            table.trace(&mut |r| self.drag_out_root(r));
        }
    }

    fn remember_young_pointer(&mut self, obj: GcRef) {
        // incminimark.py does not special-case STATE_SWEEPING in its barrier.
        // After the seam, every retained entry names a VISITED survivor. A new
        // barrier entry can only name an object the mutator can reach: an old
        // candidate already known live, a candidate already swept, or a new
        // object on the fresh lists. An unreachable white candidate cannot be
        // mutated and therefore cannot be re-added between sweep steps.
        let type_id = unsafe { (*header_of(obj.0)).type_id() };
        self.validate_type_id(type_id, obj.0, "remember_young_pointer_insert");
        // `_remember_young_pointer_inlined`: "We know that 'addr' cannot be
        // in the nursery, because nursery objects never have the flag
        // GCFLAG_TRACK_YOUNG_PTRS to start with."
        debug_assert!(
            !self.is_in_nursery(obj.0),
            "remember_young_pointer on nursery object {:#x}",
            obj.0
        );
        self.old_objects_pointing_to_young.push(obj.0);
        crate::bh_probe_note_barriered(obj.0);
        let hdr = unsafe { header_of(obj.0) };
        if crate::gc_lifetime_log_enabled() {
            eprintln!(
                "[gc][remember] addr={:#x} type_id={} source=write_barrier state={:?}",
                obj.0,
                unsafe { (*hdr).type_id() },
                self.gc_state
            );
        }
        unsafe {
            (*hdr).clear_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
            if (*hdr).has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS) {
                (*hdr).clear_flag(GcFlags::GCFLAG_NO_HEAP_PTRS);
                self.prebuilt_root_objects.push(obj.0);
            }
        }
        // `_add_to_more_objects_to_trace_if_black` is the MARKING arm of
        // `_minor_collection`. A store after that arm, when the next step is
        // a major that empties the worklist (`do_collect_oldgen_nonmoving`,
        // or the increment that calls `finish_incremental_marking`), would
        // otherwise leave new children white. Re-grey the modified black
        // object here so `visit` / `mark_object` retraces it.
        if self.gc_state == GcState::Marking {
            self.add_to_more_objects_to_trace_if_black(obj.0);
        }
    }

    /// incminimark.py write_barrier_from_array:
    /// called by non-JIT code. Checks TRACK_YOUNG_PTRS, then dispatches
    /// to remember_young_pointer_from_array2 (card path) or
    /// remember_young_pointer (generic).
    pub fn do_write_barrier_card(&mut self, obj: GcRef, index: usize, card_page_shift: u32) {
        // A non-GC (`Box::into_raw`) PyObject can reach the barrier here too;
        // ignore any address the GC does not own (see `do_write_barrier`).
        if obj.is_null() || !self.is_managed_heap_object(obj.0) {
            return;
        }
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) } {
            self.remember_young_pointer_from_array2(obj, index, card_page_shift);
        }
    }

    /// incminimark.py remember_young_pointer_from_array2:
    /// Called when TRACK_YOUNG_PTRS is set. If HAS_CARDS, marks the
    /// card WITHOUT clearing TRACK_YOUNG_PTRS. Otherwise falls back
    /// to the generic _remember_young_pointer_inlined.
    fn remember_young_pointer_from_array2(
        &mut self,
        obj: GcRef,
        index: usize,
        card_page_shift: u32,
    ) {
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) } {
            // No cards — fall back to the generic _remember_young_pointer_inlined.
            self.remember_young_pointer(obj);
            return;
        }
        // Card path: mark the specific card. TRACK_YOUNG_PTRS stays set
        // so that subsequent writes still enter the barrier.
        self.mark_card(obj, index, card_page_shift);
    }

    /// incminimark.py jit_remember_young_pointer_from_array:
    /// Minimal version called by the JIT when TRACK_YOUNG_PTRS is set
    /// but CARDS_SET is not. Tries to set CARDS_SET; otherwise falls
    /// back to remember_young_pointer (generic barrier).
    pub fn jit_remember_young_pointer_from_array(&mut self, obj: GcRef) {
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) } {
            // incminimark.py:1614-1615
            self.old_objects_with_cards_set.push(obj.0);
            unsafe { (*hdr).set_flag(GcFlags::GCFLAG_CARDS_SET) };
        } else {
            // No cards: the JIT already passed the inline flag test, so use
            // the corresponding remember_young_pointer path.
            self.jit_remember_young_pointer(obj);
        }
    }

    /// incminimark.py `writebarrier_before_copy`: "This has the same effect as
    /// calling writebarrier over each element in dest copied from source,
    /// except it might reset one of the following flags a bit too eagerly,
    /// which means we'll have a bit more objects to track, but being on the
    /// safe side."
    ///
    /// `false` is the answer "do it manually in ll_arraycopy" — the bulk form
    /// cannot express this copy and the caller owes a per-item barrier.
    /// Addresses are payload addresses, and `length` counts items.
    pub fn writebarrier_before_copy(
        &mut self,
        source_addr: usize,
        dest_addr: usize,
        source_start: usize,
        dest_start: usize,
        length: usize,
    ) -> bool {
        // "obscuuuure" upstream, and for the same reason here: entering a
        // parent on `old_objects_pointing_to_pinned` does not itself put that
        // parent through the barrier, so the flags this function reads below
        // are only trustworthy once every such entry has been.  Upstream notes
        // there should only be a small number of them.
        if self.updated_old_objects_pointing_to_pinned {
            let mut parents = std::mem::take(&mut self.old_objects_pointing_to_pinned);
            for &obj_addr in &parents {
                self.do_write_barrier(GcRef(obj_addr));
            }
            parents.append(&mut self.old_objects_pointing_to_pinned);
            self.old_objects_pointing_to_pinned = parents;
            self.updated_old_objects_pointing_to_pinned = false;
        }

        let source_hdr = unsafe { header_of(source_addr) };
        let dest_hdr = unsafe { header_of(dest_addr) };
        // The fast path of the write barrier: a destination already on the
        // remembered set needs nothing more.
        if unsafe { !(*dest_hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) } {
            return true;
        }

        if self.config.card_page_indices > 0
            && unsafe { (*source_hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) }
        {
            if unsafe { !(*source_hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) } {
                // "The source object may have random young pointers."
                return false;
            }
            if unsafe { !(*dest_hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) } {
                // "The dest object doesn't have cards.  Do it manually."
                return false;
            }
            debug_assert!(
                unsafe { !(*dest_hdr).has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS) },
                "prebuilt object with cards is not supported"
            );
            if unsafe { !(*source_hdr).has_flag(GcFlags::GCFLAG_CARDS_SET) } {
                // The source has no young pointers at all, so no card needs
                // marking.  A black destination still has to go gray and be
                // rescanned, which is what `collect_cardrefs_to_nursery` reads
                // CARDS_SET for at the end of a marking step.
                if self.gc_state == GcState::Marking
                    && unsafe { !(*dest_hdr).has_flag(GcFlags::GCFLAG_CARDS_SET) }
                {
                    self.old_objects_with_cards_set.push(dest_addr);
                    unsafe { (*dest_hdr).set_flag(GcFlags::GCFLAG_CARDS_SET) };
                }
                return true;
            }
            if source_start != 0 || dest_start != 0 {
                // Misaligned: the card index of an item differs between the
                // two objects, so the bits cannot be copied across.
                return false;
            }
            self.manually_copy_card_bits(source_addr, dest_addr, length);
            return true;
        }

        // While marking, the source's TRACK_YOUNG_PTRS says nothing: the
        // barrier is also what turns a black object gray again, so a source
        // that already lost the flag can still hold pointers this copy must
        // report.
        if unsafe { !(*source_hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) }
            || self.gc_state == GcState::Marking
        {
            self.remember_young_pointer(GcRef(dest_addr));
        }

        if unsafe { (*dest_hdr).has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS) }
            && unsafe { !(*source_hdr).has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS) }
        {
            unsafe { (*dest_hdr).clear_flag(GcFlags::GCFLAG_NO_HEAP_PTRS) };
            self.prebuilt_root_objects.push(dest_addr);
        }
        true
    }

    /// incminimark.py `writebarrier_before_move`: "If 'array_addr' uses cards,
    /// then this has the same effect as a call to the generic writebarrier,
    /// effectively generalizing the cards to 'any item may be young'."
    ///
    /// A move within one object carries its items across card boundaries while
    /// the card bits stay where they are, so the per-card record is no longer
    /// true of the object and only the whole-object record is.
    ///
    /// `alloc_varsize_typed` reaches
    /// `alloc_varsize_nonmoving_prefer_young`, so production list storage can
    /// carry cards. The sites that owe this call are the item moves in
    /// `pyre_object::listobject`'s `W_ListObject` — `object_insert`,
    /// `object_remove` and `object_drain` each shift items with a bare
    /// `ptr::copy`. `object_reverse` follows `rlist.py ll_reverse` and
    /// stores through `ll_list_obj_setitem_fast` (`setarrayitem_gc`): the
    /// host setter (`items_block_set_ref`) is `write_barrier_from_array`.
    /// `object_splice` is not on this list —
    /// it already calls `list_write_barrier` before its copies.
    ///
    /// Upstream reaches this barrier from those operations through
    /// `rgc.ll_arraymove`, which `ll_insert_nonneg`, `ll_pop_zero`,
    /// `ll_delitem_nonneg` and `ll_listdelslice_startstop` all call; pyre's
    /// list is not an rtyper-lowered list, so it has no such seam, and the
    /// three calls are written out at those sites as
    /// `listobject::list_before_move_barrier`.
    ///
    /// The order was not free. Those sites are prerequisites of a production
    /// card allocator, not siblings of it: with cards live and a move
    /// unbarriered, a minor scans only the dirty pages and a young pointer
    /// shifted into a clean one is collected while live. That is silent heap
    /// corruption, where the absence of cards makes the same code correct.
    pub fn writebarrier_before_move(&mut self, array_addr: usize) {
        if self.config.card_page_indices == 0 {
            return;
        }
        let hdr = unsafe { header_of(array_addr) };
        if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_CARDS_SET) } {
            self.do_write_barrier(GcRef(array_addr));
        }
    }

    /// incminimark.py `manually_copy_card_bits`.
    fn manually_copy_card_bits(&mut self, source_addr: usize, dest_addr: usize, length: usize) {
        debug_assert!(
            self.config.card_page_indices > 0,
            "non-positive card_page_indices"
        );
        let bytes = self.card_marking_bytes_for_length(length);
        let mut anybyte = 0u8;
        for i in 0..bytes {
            let byte = unsafe { *Self::get_card_ptr(source_addr, i) };
            anybyte |= byte;
            unsafe { *Self::get_card_ptr(dest_addr, i) |= byte };
        }
        if anybyte != 0 {
            let dest_hdr = unsafe { header_of(dest_addr) };
            if unsafe { !(*dest_hdr).has_flag(GcFlags::GCFLAG_CARDS_SET) } {
                self.old_objects_with_cards_set.push(dest_addr);
                unsafe { (*dest_hdr).set_flag(GcFlags::GCFLAG_CARDS_SET) };
            }
        }
    }

    /// incminimark.py:1574-1598: set the card bit for a given array index.
    /// Card bits are stored inline before the GcHeader. If CARDS_SET not
    /// yet set, add obj to `old_objects_with_cards_set` and set CARDS_SET.
    fn mark_card(&mut self, obj: GcRef, index: usize, card_page_shift: u32) {
        // incminimark.py:1576-1578
        let bitindex = index >> card_page_shift;
        let byteindex = bitindex >> 3;
        let bitmask: u8 = 1 << (bitindex & 7);

        // incminimark.py:1581-1584: if bit already set, return.
        let addr_byte = Self::get_card_ptr(obj.0, byteindex);
        let byte = unsafe { *addr_byte };
        if byte & bitmask != 0 {
            return;
        }

        // incminimark.py:1594: set the bit.
        unsafe {
            *addr_byte = byte | bitmask;
        }

        // incminimark.py:1596-1598: if CARDS_SET not set, track and set.
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_CARDS_SET) } {
            self.old_objects_with_cards_set.push(obj.0);
            unsafe { (*hdr).set_flag(GcFlags::GCFLAG_CARDS_SET) };
        }
    }

    /// Check whether a specific card of an object is dirty.
    /// Reads inline card bytes before the GcHeader.
    pub fn is_card_dirty(&self, obj: GcRef, card_index: usize) -> bool {
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) } {
            return false;
        }
        let byteindex = card_index >> 3;
        let bitmask: u8 = 1 << (card_index & 7);
        let addr_byte = Self::get_card_ptr(obj.0, byteindex);
        unsafe { *addr_byte & bitmask != 0 }
    }

    /// Return all dirty card indices for the given object.
    /// Reads inline card bytes before the GcHeader.
    pub fn dirty_cards(&self, obj: GcRef) -> Vec<usize> {
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) } {
            return Vec::new();
        }
        let type_id = unsafe { (*hdr).type_id() };
        let type_info = self.types.get(type_id);
        let length_offset = type_info.length_offset;
        // Card bytes belong to an old array. `is_forwarded` is nursery-only,
        // so this length word has not been overwritten with a forwarding address.
        let length = unsafe { *((obj.0 + length_offset) as *const usize) };
        let bytes = self.card_marking_bytes_for_length(length);
        let mut result = Vec::new();
        for bi in 0..bytes {
            let addr = Self::get_card_ptr(obj.0, bi);
            let cardbyte = unsafe { *addr };
            for bit in 0..8 {
                if cardbyte & (1 << bit) != 0 {
                    result.push(bi * 8 + bit);
                }
            }
        }
        result
    }

    /// incminimark.py: collect_cardrefs_to_nursery.
    /// Process `old_objects_with_cards_set` — for each object, scan
    /// dirty card ranges and copy nursery objects out.
    fn collect_cardrefs_to_nursery(&mut self) {
        while let Some(obj) = self.old_objects_with_cards_set.pop() {
            // incminimark.py:2016-2020
            let hdr = unsafe { header_of(obj) };
            debug_assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) });
            debug_assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_CARDS_SET) });
            unsafe { (*hdr).clear_flag(GcFlags::GCFLAG_CARDS_SET) };

            // incminimark.py:2023-2026
            let type_id = unsafe { (*hdr).type_id() };
            let type_info = self.types.get(type_id);
            let length_offset = type_info.length_offset;
            // `old_objects_with_cards_set` holds old arrays. A nursery
            // corpse is never on that list (`is_forwarded` asserts the
            // nursery), so the length word is the array's own.
            let length = unsafe { *((obj + length_offset) as *const usize) };
            let bytes = self.card_marking_bytes_for_length(length);

            // incminimark.py:2033-2039: if !TRACK_YOUNG_PTRS, object is also
            // in old_objects_pointing_to_young and will be fully traced.
            // Just clear card bytes.
            if unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) } {
                for bi in 0..bytes {
                    let p = Self::get_card_ptr(obj, bi);
                    unsafe {
                        *p = 0;
                    }
                }
                continue;
            }

            // incminimark.py:2041-2072: walk card bytes, trace dirty ranges.
            if type_info.items_have_gc_ptrs && type_info.item_size > 0 {
                let item_size = type_info.item_size;
                let base_size = type_info.size;
                let n_var_offsets = type_info.var_gc_ptr_offsets.len();
                let items_start = obj + base_size;
                let card_page_indices = 1usize << self.card_page_shift;

                let mut interval_start = 0usize;
                for bi in 0..bytes {
                    let p = Self::get_card_ptr(obj, bi);
                    let mut cardbyte = unsafe { *p };
                    unsafe {
                        *p = 0;
                    } // reset
                    let next_byte_start = interval_start + 8 * card_page_indices;

                    while cardbyte != 0 {
                        let mut interval_stop = interval_start + card_page_indices;
                        if cardbyte & 1 != 0 {
                            if interval_stop > length {
                                interval_stop = length;
                                if interval_stop <= interval_start {
                                    break;
                                }
                            }
                            // trace_and_drag_out_of_nursery_partial
                            for i in interval_start..interval_stop {
                                let item = items_start + i * item_size;
                                for j in 0..n_var_offsets {
                                    let offset = self.types.get(type_id).var_gc_ptr_offsets[j];
                                    let slot = (item + offset) as *mut GcRef;
                                    let field_ref = unsafe { *slot };
                                    self.assert_traced_slot_initialized(
                                        field_ref,
                                        slot as usize,
                                        obj,
                                        "minor_dirty_card_item",
                                        "minor_dirty_card",
                                    );
                                    if self.is_nursery_object_start(field_ref.0) {
                                        let new_ref = self.copy_nursery_object(
                                            field_ref.0,
                                            "minor_dirty_card_item_target",
                                            "minor_dirty_card",
                                            obj,
                                            slot as usize,
                                        );
                                        unsafe {
                                            *slot = new_ref;
                                        }
                                    } else if self.is_young_rawmalloced(field_ref.0) {
                                        self.visit_young_rawmalloced_object(field_ref.0);
                                    }
                                }
                            }
                        }
                        interval_start = interval_stop;
                        cardbyte >>= 1;
                    }
                    interval_start = next_byte_start;
                }
            } else {
                // No GC ptrs in items — just clear card bytes.
                for bi in 0..bytes {
                    let p = Self::get_card_ptr(obj, bi);
                    unsafe {
                        *p = 0;
                    }
                }
            }

            // incminimark.py:2079-2083: if incremental marking, re-add so the
            // object's outgoing references are (re)scanned before the sweep.
            // incminimark clears GCFLAG_VISITED here because its trace-list
            // drain re-marks on visit; pyre's `mark_object` does not set VISITED
            // on the popped object (the marking convention sets it at *push*
            // time — see `seed_major_root` and the remembered-set rescan above),
            // so keep the object black across the re-push. Clearing it would
            // leave the still-reachable card object white for the incremental
            // sweep to reclaim.
            if self.gc_state == GcState::Marking {
                unsafe { (*hdr).set_flag(GcFlags::GCFLAG_VISITED) };
                self.incr_state.more_gray_stack.push(obj);
            }
        }
    }

    /// Clear inline card bytes for a given object.
    pub fn clear_cards(&mut self, obj_addr: usize) {
        let hdr = unsafe { header_of(obj_addr) };
        if unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) } {
            unsafe { (*hdr).clear_flag(GcFlags::GCFLAG_CARDS_SET) };
            return;
        }
        let type_id = unsafe { (*hdr).type_id() };
        let type_info = self.types.get(type_id);
        let length_offset = type_info.length_offset;
        // Same old-array precondition as `dirty_cards`: not a nursery corpse.
        let length = unsafe { *((obj_addr + length_offset) as *const usize) };
        let bytes = self.card_marking_bytes_for_length(length);
        for bi in 0..bytes {
            let p = Self::get_card_ptr(obj_addr, bi);
            unsafe {
                *p = 0;
            }
        }
        unsafe { (*hdr).clear_flag(GcFlags::GCFLAG_CARDS_SET) };
    }

    // ── JIT integration hooks ──

    /// Fast-path write barrier for JIT-compiled code.
    ///
    /// Called from JIT-compiled code when a write barrier fires.
    /// Adds the object directly to the remembered set without the
    /// full flag-check logic of `do_write_barrier()`, because the
    /// JIT has already determined that the barrier is needed (via
    /// the inline flag test emitted by COND_CALL_GC_WB).
    ///
    /// Equivalent to incminimark's `jit_remember_young_pointer()`: the JIT has
    /// already passed the inline GCFLAG_TRACK_YOUNG_PTRS test emitted by
    /// COND_CALL_GC_WB, so go straight to the inlined remember logic without
    /// re-testing the flag.
    pub fn jit_remember_young_pointer(&mut self, obj: GcRef) {
        self.remember_young_pointer(obj);
    }

    /// Returns true if the GC supports optimized conditional write barriers.
    ///
    /// When true, the JIT can emit COND_CALL_GC_WB (an inline flag test +
    /// conditional call) instead of a full write-barrier call. Nursery-based
    /// collectors always support this because the barrier check is a simple
    /// flag test on the object header.
    pub fn can_optimize_cond_call(&self) -> bool {
        true
    }

    /// `IncrementalMiniMarkGC.can_optimize_clean_setarrayitems`.
    ///
    /// A clean bulk store is barrier-free only when card marking is disabled.
    /// With cards, clearing array items without marking their pages can hide
    /// young references from the next minor collection.
    pub fn can_optimize_clean_setarrayitems(&self) -> bool {
        self.config.card_page_indices == 0
    }

    /// Perform one incremental GC step. Called from JIT safepoints.
    ///
    /// If an incremental major cycle should start, it is initiated. If a cycle
    /// is already in progress, one bounded MARKING or SWEEPING step is
    /// performed. Returns true if any GC work was done.
    ///
    /// incminimark.py `gc_step_until` / `debug_gc_step`: a minor before every
    /// `major_collection_step`. `collect_step` is the same pair plus the
    /// rawrefcount callback; this is the JIT-safepoint half without that tail.
    pub fn gc_step(&mut self) -> bool {
        if !self.enabled {
            return false;
        }
        if self.gc_state == GcState::Scanning && !self.threshold_reached(0) {
            return false;
        }
        // `collect_step` takes the same pause: a leading minor moves the
        // nursery, so every registered mutator has to be off the heap.
        let _stw = if crate::gc_sync::stw_required() {
            Some(crate::gc_sync::quiesce_mutators())
        } else {
            None
        };
        self.minor_collection_body();
        self.major_collection_step();
        true
    }

    /// `IncrementalMiniMarkGC._minor_collection` nursery-barrier rebuild.
    ///
    /// The surviving AddressStack contains header addresses. Sort it, reset
    /// only the free ranges between those headers, and leave `nursery_top` at
    /// the first pin while the deque owns the remaining pins plus nursery end.
    fn reset_nursery_with_pinned(&mut self) {
        let nursery_start = self.nursery.start_ptr() as usize;
        let nursery_end = nursery_start + self.nursery.size();
        let mut surviving_pinned_objects = std::mem::take(&mut self.surviving_pinned_objects);
        surviving_pinned_objects.sort_unstable();
        debug_assert_eq!(
            self.pinned_objects_in_nursery,
            surviving_pinned_objects.len()
        );

        let mut barriers = VecDeque::with_capacity(surviving_pinned_objects.len() + 1);
        let mut previous_end = nursery_start;
        for header_addr in surviving_pinned_objects {
            assert!(
                header_addr >= previous_end,
                "pinned objects encountered in backwards order"
            );
            self.nursery.reset_range(previous_end, header_addr);

            let obj_addr = header_addr + GcHeader::SIZE;
            let type_id = unsafe { (*header_of(obj_addr)).type_id() };
            let payload_size = self.size_for_typeid(obj_addr, type_id, "pinned_barriers");
            let object_size = Self::nursery_allocation_size(GcHeader::SIZE + payload_size);
            unsafe { (*header_of(obj_addr)).clear_flag(GcFlags::GCFLAG_VISITED) };
            barriers.push_back(header_addr);
            previous_end = header_addr + object_size;
        }
        self.nursery.reset_range(previous_end, nursery_end);
        // `_minor_collection` rotates only on the no-pinned-objects arm: a
        // pinned object stays where it is, and the arena holding it is about
        // to be made inaccessible.
        if self.config.gc_nursery_debug && barriers.is_empty() {
            self.nursery.debug_rotate();
        }
        // Read the arena back rather than reusing the entry values: a rotation
        // moved it, and both the barrier below and the `nursery_free` at the
        // tail of this function have to name the arena now installed.
        let nursery_start = self.nursery.start_ptr() as usize;
        let nursery_end = nursery_start + self.nursery.size();
        barriers.push_back(nursery_end);
        self.nursery_barriers = barriers;

        let first_top = self
            .nursery_barriers
            .pop_front()
            .expect("nursery end is always a barrier");
        unsafe {
            self.nursery.set_top_ptr(first_top as *const u8);
            self.nursery.set_free_ptr(nursery_start as *mut u8);
        }
    }

    /// incminimark.py `pin`.
    pub fn pin(&mut self, obj: GcRef) -> bool {
        if self.pinned_objects_in_nursery >= self.max_number_of_pinned_objects {
            return false;
        }
        if obj.is_null() || !self.is_nursery_object_start(obj.0) {
            return false;
        }
        if self.is_pinned(obj) {
            return false;
        }
        let type_id = unsafe { (*header_of(obj.0)).type_id() };
        self.validate_type_id(type_id, obj.0, "pin");
        let info = self.types.get(type_id);
        // gctypelayout.py `q_cannot_pin`: GC-pointer-bearing types,
        // weakrefs, and `customdata` (custom trace / destructor / finalizer)
        // cannot be pinned. A registered finalizer is represented by the
        // per-object flag in pyre's dynamic finalizer-queue surface.
        if info.has_gc_ptrs
            || info.is_weakref
            || info.custom_trace.is_some()
            || info.destructor.is_some()
            || unsafe { (*header_of(obj.0)).has_flag(GcFlags::FINALIZER_REGISTERED) }
        {
            return false;
        }
        unsafe {
            (*header_of(obj.0)).set_flag(GcFlags::GCFLAG_PINNED);
        }
        self.pinned_objects_in_nursery += 1;
        true
    }

    /// incminimark.py `unpin`.
    pub fn unpin(&mut self, obj: GcRef) {
        assert!(self.is_pinned(obj), "unpin: object is already not pinned");
        unsafe {
            (*header_of(obj.0)).clear_flag(GcFlags::GCFLAG_PINNED);
        }
        self.pinned_objects_in_nursery -= 1;
    }

    /// Check if an object is currently pinned.
    pub fn is_pinned(&self, obj: GcRef) -> bool {
        if obj.is_null() || !self.is_in_nursery(obj.0) {
            return false;
        }
        let hdr = unsafe { &*header_of(obj.0) };
        !hdr.is_forwarded() && hdr.has_flag(GcFlags::GCFLAG_PINNED)
    }

    /// `rgc.get_gcflag_extra`: read RPython's temporary traversal bit from a
    /// managed or translated-prebuilt object.
    pub fn get_gcflag_extra(&self, obj: GcRef) -> bool {
        assert!(!obj.is_null(), "get_gcflag_extra(NULL)");
        self.gcflag_header(obj)
            .is_some_and(|hdr| unsafe { (*hdr).has_flag(GcFlags::GCFLAG_EXTRA) })
    }

    /// `rgc.toggle_gcflag_extra`: toggle the traversal bit in place. As in
    /// upstream, the argument must be a non-NULL GC object.
    pub fn toggle_gcflag_extra(&self, obj: GcRef) {
        assert!(!obj.is_null(), "toggle_gcflag_extra(NULL)");
        let hdr = self
            .gcflag_header(obj)
            .expect("toggle_gcflag_extra called for a non-GC object");
        unsafe {
            if (*hdr).has_flag(GcFlags::GCFLAG_EXTRA) {
                (*hdr).clear_flag(GcFlags::GCFLAG_EXTRA);
            } else {
                (*hdr).set_flag(GcFlags::GCFLAG_EXTRA);
            }
        }
    }

    /// `rgc.get_gcflag_dummy`: whether this is the low-level dummy value used
    /// by translated hash tables.
    pub fn get_gcflag_dummy(&self, obj: GcRef) -> bool {
        if obj.is_null() {
            return false;
        }
        self.gcflag_header(obj)
            .is_some_and(|hdr| unsafe { (*hdr).has_flag(GcFlags::GCFLAG_DUMMY) })
    }

    fn gcflag_header(&self, obj: GcRef) -> Option<*mut GcHeader> {
        if self.is_managed_heap_object(obj.0) {
            Some(unsafe { header_of(obj.0) })
        } else {
            self.registered_external_header(obj.0)
        }
    }

    /// Number of objects in the remembered set (for testing / diagnostics).
    pub fn old_objects_pointing_to_young_len(&self) -> usize {
        self.old_objects_pointing_to_young.len()
    }
}

impl Default for MiniMarkGC {
    fn default() -> Self {
        Self::new()
    }
}

impl GcAllocator for MiniMarkGC {
    fn remember_gc_table(&mut self, table: &Arc<crate::GcTable>) {
        MiniMarkGC::remember_gc_table(self, table);
    }

    fn writebarrier_before_move(&mut self, obj: GcRef) {
        MiniMarkGC::writebarrier_before_move(self, obj.0);
    }

    fn writebarrier_before_copy(
        &mut self,
        source: GcRef,
        dest: GcRef,
        source_start: usize,
        dest_start: usize,
        length: usize,
    ) -> bool {
        MiniMarkGC::writebarrier_before_copy(
            self,
            source.0,
            dest.0,
            source_start,
            dest_start,
            length,
        )
    }

    fn shrink_array(&mut self, addr: usize, smaller_length: usize) -> bool {
        MiniMarkGC::shrink_array(self, addr, smaller_length)
    }

    fn varsize_layout(&self, obj: GcRef) -> Option<crate::GcVarSizeLayout> {
        if obj.is_null() || !self.is_managed_heap_object(obj.0) {
            return None;
        }
        let type_id = unsafe { (*header_of(obj.0)).type_id() };
        if type_id as usize >= self.types.len() {
            return None;
        }
        let info = self.types.get(type_id);
        (info.item_size != 0).then_some(crate::GcVarSizeLayout {
            base_size: info.size,
            item_size: info.item_size,
            items_have_gc_ptrs: info.items_have_gc_ptrs,
        })
    }

    fn debug_validate_oldgen_freeblocks(&self, site: &str) {
        self.oldgen.debug_validate_freeblocks(site);
    }

    fn alloc_nursery(&mut self, size: usize) -> GcRef {
        self.alloc_with_type(0, size)
    }

    fn alloc_nursery_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        self.alloc_with_type(type_id, size)
    }

    unsafe fn alloc_nursery_collecting_typed_rooted(
        &mut self,
        type_id: u32,
        size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe { self.alloc_with_type_rooted(type_id, size, root, needs_write_barrier) }
    }

    unsafe fn alloc_fast_nursery_collecting_typed_rooted(
        &mut self,
        type_id: u32,
        size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe { self.alloc_fast_with_type_rooted(type_id, size, root, needs_write_barrier) }
    }

    unsafe fn alloc_fast_nursery_collecting_typed_roots(
        &mut self,
        type_id: u32,
        size: usize,
        roots: *mut GcRef,
        root_count: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.alloc_fast_with_type_roots(type_id, size, roots, root_count, needs_write_barrier)
        }
    }

    fn alloc_nursery_no_collect(&mut self, size: usize) -> GcRef {
        self.alloc_with_type_no_collect(0, size)
    }

    fn alloc_nursery_no_collect_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        self.alloc_with_type_no_collect(type_id, size)
    }

    fn try_alloc_nursery_no_collect_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        self.try_alloc_with_type_no_collect(type_id, size)
    }

    unsafe fn try_alloc_fast_nursery_no_collect_typed(
        &mut self,
        type_id: u32,
        size: usize,
    ) -> GcRef {
        unsafe { self.try_alloc_fast_with_type_no_collect(type_id, size) }
    }

    unsafe fn try_alloc_nursery_no_collect_typed_with_placement(
        &mut self,
        type_id: u32,
        size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.try_alloc_with_type_no_collect_with_placement(type_id, size, needs_write_barrier)
        }
    }

    unsafe fn try_alloc_fast_nursery_no_collect_typed_with_placement(
        &mut self,
        type_id: u32,
        size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.try_alloc_fast_with_type_no_collect_with_placement(
                type_id,
                size,
                needs_write_barrier,
            )
        }
    }

    fn alloc_varsize(&mut self, base_size: usize, item_size: usize, length: usize) -> GcRef {
        let Some(payload_size) = Self::checked_varsize_payload_size(base_size, item_size, length)
        else {
            return GcRef(0);
        };
        self.alloc_with_type(0, payload_size)
    }

    fn alloc_varsize_typed(
        &mut self,
        type_id: u32,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef {
        let Some(payload_size) = Self::checked_varsize_payload_size(base_size, item_size, length)
        else {
            return GcRef(0);
        };
        let Some(total_size) = GcHeader::SIZE.checked_add(payload_size) else {
            return GcRef(0);
        };
        if (type_id as usize) >= self.types.len() {
            return GcRef(0);
        }
        let type_info = self.types.get(type_id);
        let registered_varsize = type_info.item_size != 0;
        let length_offset = type_info.length_offset;
        let items_have_gc_ptrs = type_info.items_have_gc_ptrs;
        let obj = if total_size >= self.config.large_object_threshold {
            // incminimark.py `malloc_varsize`: the oversized arm calls
            // `external_malloc(typeid, length, alloc_young=True)`, whose card
            // decision comes from the registered type layout.  Routing through
            // generic `alloc_with_type` loses both `length` and that predicate,
            // so a production large pointer array is born cardless even though
            // the collector's card allocator exists.
            self.alloc_varsize_nonmoving_prefer_young(
                type_id,
                total_size,
                length,
                items_have_gc_ptrs,
            )
        } else {
            self.alloc_with_type(type_id, payload_size)
        };
        // incminimark.py `malloc_varsize` common/young paths both initialize
        // `varsize_offset_to_length(typeid)` before returning.  The backend
        // slow helper must not receive a typed array whose walker still sees
        // the allocator's zero-fill as its length.
        if !obj.is_null() && registered_varsize {
            unsafe { *((obj.0 + length_offset) as *mut usize) = length };
        }
        obj
    }

    fn alloc_varsize_no_collect(
        &mut self,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef {
        let Some(payload_size) = Self::checked_varsize_payload_size(base_size, item_size, length)
        else {
            return GcRef(0);
        };
        self.alloc_with_type_no_collect(0, payload_size)
    }

    fn alloc_oldgen_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        let Some(total_size) = GcHeader::SIZE.checked_add(size) else {
            return GcRef(0);
        };
        // `GcLLDescr_framework._bh_malloc` reaches translated
        // `do_malloc_fixedsize_clear`, whose rawmalloc failure is returned to
        // the exception transformer as NULL and becomes `MemoryError`.
        // Blackhole host calls need the same fallible edge; the infallible
        // `oldgen.alloc()` path aborts the process before `_get_method` can
        // deliver that exception.
        let Some(obj) = self.try_alloc_in_oldgen(type_id, total_size) else {
            return GcRef(0);
        };
        Self::raw_memclear(obj, total_size);
        obj
    }

    fn alloc_young_nonmoving_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        let Some(total_size) = GcHeader::SIZE.checked_add(size) else {
            return GcRef(0);
        };
        // incminimark.py `external_malloc`: `threshold_reached(raw_malloc_usage(totalsize))`
        // then `minor_collection_with_major_progress`. A compiled loop whose
        // other allocations are virtualized never fills the nursery, so this
        // is the collection those young rawmalloced frames would otherwise
        // never see. `raw_malloc_usage` is identity on a byte size.
        if self.do_maybe_collect_for_external_malloc(total_size) {
            return GcRef(0);
        }
        // incminimark.py `external_malloc(..., alloc_young=True)`; a refused
        // young birth (weakref type, old-style finalizer,
        // `MAJIT_GC_YOUNG_RAWMALLOC=0`) is born old instead.
        if let Some(obj) = self.try_alloc_young_nonmoving_clear(type_id, total_size) {
            return obj;
        }
        self.alloc_oldgen_typed(type_id, size)
    }

    fn alloc_young_nonmoving_typed_no_collect(&mut self, type_id: u32, size: usize) -> GcRef {
        let Some(total_size) = GcHeader::SIZE.checked_add(size) else {
            return GcRef(0);
        };
        // The same young birth without `external_malloc`'s
        // `minor_collection_with_major_progress` in front of it, for the
        // reason `alloc_oldgen_typed` gives: the host callers this entry
        // exists for hold the fields they are about to store on the Rust
        // stack, where a collection run here would leave them stale. The
        // threshold question is still asked — `finish_alloc_young_nonmoving`
        // raises the deferred major request, and the interpreter safepoint
        // answers it where the root set is known.
        if let Some(obj) = self.try_alloc_young_nonmoving_clear(type_id, total_size) {
            return obj;
        }
        self.alloc_oldgen_typed(type_id, size)
    }

    fn collection_counts(&self) -> (usize, usize) {
        (self.minor_collections, self.major_collections)
    }

    fn minor_collections_since_major(&self) -> usize {
        self.minor_collections - self.minor_collections_at_major_end
    }

    fn get_write_barrier_descr(&self) -> Option<crate::WriteBarrierDescr> {
        let mut descr = crate::WriteBarrierDescr::for_current_gc();
        if self.card_page_shift == 0 {
            descr.jit_wb_cards_set = 0;
            descr.jit_wb_card_page_shift = 0;
            descr.jit_wb_cards_set_byteofs = 0;
            descr.jit_wb_cards_set_singlebyte = 0;
        } else {
            descr.jit_wb_card_page_shift = self.card_page_shift;
        }
        Some(descr)
    }

    fn type_alloc_is_plain(&self, type_id: u32) -> bool {
        (type_id as usize) < self.types.len() && {
            let info = self.types.get(type_id);
            info.destructor.is_none() && info.old_style_finalizer.is_none() && !info.is_weakref
        }
    }

    fn is_managed_heap_object(&self, addr: usize) -> bool {
        MiniMarkGC::is_managed_heap_object(self, addr)
    }

    fn is_nursery_object(&self, addr: usize) -> bool {
        self.is_nursery_object_start(addr)
    }

    fn nursery_bounds(&self) -> Option<(usize, usize)> {
        // `debug_rotate` installs the ring's front arena on every minor
        // collection that finds no pinned survivor, so the range is then not
        // fixed for the life of this allocator and no published pair can
        // describe it.  `None` keeps every query on `is_nursery_object`,
        // which reads the arena now installed.  Both terms are the pair
        // `reset_nursery_with_pinned` tests before it rotates, and neither
        // has a writer after construction.
        if self.config.gc_nursery_debug && self.nursery.debug_rotating_nurseries() != 0 {
            return None;
        }
        let start = self.nursery.start_ptr() as usize;
        Some((start, start + self.nursery.size()))
    }

    fn taggedpointers(&self) -> bool {
        self.config.taggedpointers
    }

    fn write_barrier(&mut self, obj: GcRef) {
        self.do_write_barrier(obj);
    }

    fn write_barrier_managed(&mut self, obj: GcRef) {
        self.do_write_barrier_managed(obj);
    }

    fn write_barrier_from_array(&mut self, obj: GcRef, index: usize) {
        // A non-GC (`Box::into_raw`) PyObject can reach the barrier here too;
        // ignore any address the GC does not own (see `do_write_barrier`).
        if obj.is_null() || !self.is_managed_heap_object(obj.0) {
            return;
        }
        // incminimark.py `write_barrier_from_array`: TRACK_YOUNG_PTRS, then
        // `card_page_indices > 0` → `remember_young_pointer_from_array2`,
        // else `remember_young_pointer`.
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { (*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) } {
            if self.config.card_page_indices > 0 {
                self.remember_young_pointer_from_array2(obj, index, self.card_page_shift);
            } else {
                self.remember_young_pointer(obj);
            }
        }
    }

    fn jit_remember_young_pointer_from_array(&mut self, obj: GcRef) {
        self.jit_remember_young_pointer_from_array(obj);
    }

    fn remember_young_pointer_from_array2(
        &mut self,
        obj: GcRef,
        index: usize,
        card_page_shift: u32,
    ) {
        self.remember_young_pointer_from_array2(obj, index, card_page_shift);
    }

    fn collect_nursery(&mut self) {
        self.do_collect_nursery();
    }

    fn collect_full(&mut self) {
        self.do_collect_full();
    }

    fn collect_generation(&mut self, generation: i64) {
        self.do_collect(generation);
    }

    fn collect_step(&mut self) -> crate::GcStepTransition {
        self.collect_step()
    }

    fn get_objects(&mut self, generation: i8, visitor: &mut dyn FnMut(GcRef)) {
        self.do_get_objects(generation, visitor)
    }

    fn get_referents(&mut self, obj: GcRef, visitor: &mut dyn FnMut(GcRef)) {
        self.do_get_referents(obj, visitor)
    }

    fn subgraph_has_pending_finalizer(&mut self, roots: &[GcRef]) -> bool {
        self.do_subgraph_has_pending_finalizer(roots)
    }

    fn is_tracked(&mut self, obj: GcRef) -> bool {
        self.object_is_tracked(obj.0)
    }

    fn get_gcflag_extra(&mut self, obj: GcRef) -> bool {
        MiniMarkGC::get_gcflag_extra(self, obj)
    }

    fn toggle_gcflag_extra(&mut self, obj: GcRef) {
        MiniMarkGC::toggle_gcflag_extra(self, obj)
    }

    fn get_gcflag_dummy(&mut self, obj: GcRef) -> bool {
        MiniMarkGC::get_gcflag_dummy(self, obj)
    }

    fn get_rpy_memory_usage(&mut self, obj: GcRef) -> Option<usize> {
        self.rpy_memory_usage(obj)
    }

    fn get_rpy_type_index(&mut self, obj: GcRef) -> Option<usize> {
        self.rpy_type_index(obj)
    }

    fn get_rpy_roots(&mut self, visitor: &mut dyn FnMut(GcRef)) -> bool {
        self.do_get_rpy_roots(visitor)
    }

    fn get_rpy_referents(&mut self, obj: GcRef, visitor: &mut dyn FnMut(GcRef)) -> bool {
        self.do_get_rpy_referents(obj, visitor)
    }

    fn dump_rpy_heap(&mut self, fd: i32) -> Result<bool, i32> {
        self.do_dump_rpy_heap(fd)
    }

    fn get_typeids_text(&self) -> Option<Vec<u8>> {
        Some(self.types.typeids_text())
    }

    fn get_typeids_list(&self) -> Option<Vec<usize>> {
        Some(self.types.typeids_list())
    }

    fn add_memory_pressure(&mut self, size: isize, object: GcRef) {
        self.do_add_memory_pressure(size, object);
    }

    fn maybe_collect_for_external_malloc(&mut self, totalsize: usize) -> bool {
        self.do_maybe_collect_for_external_malloc(totalsize)
    }

    fn total_memory_pressure(&mut self) -> isize {
        self.do_count_memory_pressure()
    }

    fn is_app_level_object(&mut self, obj: GcRef) -> bool {
        self.is_app_level_object_ref(obj)
    }

    fn collect_oldgen_nonmoving(&mut self) {
        self.do_collect_oldgen_nonmoving();
    }

    fn enable(&mut self) {
        self.enable();
    }

    fn disable(&mut self) {
        self.disable();
    }

    fn isenabled(&self) -> bool {
        self.isenabled()
    }

    fn register_finalizer(&mut self, fq_index: usize, obj: GcRef, trigger: FinalizerTriggerFn) {
        if obj.is_null() || !self.is_managed_heap_object(obj.0) {
            return;
        }
        assert!(fq_index <= self.finalizer_handlers.len());
        if fq_index == self.finalizer_handlers.len() {
            self.finalizer_handlers.push(FinalizerHandler {
                deque: VecDeque::new(),
                trigger,
            });
        }
        let fq_index = isize::try_from(fq_index).expect("finalizer queue index exceeds Signed");
        self.register_finalizer_index(fq_index, obj);
    }

    fn ignore_finalizer(&mut self, obj: GcRef) {
        // The same admission the finalizer queues apply
        // (`deal_with_young_objects_with_finalizers` drops an entry whose
        // object is outside both generations): an address the collector does
        // not own has no header to stamp, and dropping the hint only costs the
        // finalizer that would have run anyway.
        if obj.is_null() || !self.is_managed_heap_object(obj.0) {
            return;
        }
        let hdr = unsafe { header_of(obj.0) };
        unsafe { (*hdr).set_flag(GcFlags::GCFLAG_IGNORE_FINALIZER) };
    }

    fn finalizer_next_dead(&mut self, fq_index: usize) -> Option<GcRef> {
        self.finalizer_handlers
            .get_mut(fq_index)
            .and_then(|handler| handler.deque.pop_front())
            .map(GcRef)
    }

    fn id_or_identityhash(&mut self, obj_addr: usize) -> usize {
        self.id_or_identityhash(obj_addr)
    }

    unsafe fn add_root(&mut self, root: *mut GcRef) {
        unsafe { self.roots.add(root) };
    }

    fn remove_root(&mut self, root: *mut GcRef) {
        self.roots.remove(root);
    }

    fn nursery_free(&self) -> *mut u8 {
        self.nursery.free_ptr()
    }

    fn nursery_free_addr(&self) -> usize {
        self.nursery.free_addr()
    }

    fn nursery_top(&self) -> *const u8 {
        self.nursery.top_ptr()
    }

    fn nursery_top_addr(&self) -> usize {
        self.published_nursery_top.as_ptr() as usize
    }

    fn max_nursery_object_size(&self) -> usize {
        self.config.large_object_threshold
    }

    fn card_page_shift(&self) -> u32 {
        self.card_page_shift
    }

    fn jit_remember_young_pointer(&mut self, obj: GcRef) {
        self.jit_remember_young_pointer(obj);
    }

    fn can_optimize_cond_call(&self) -> bool {
        self.can_optimize_cond_call()
    }

    fn can_optimize_clean_setarrayitems(&self) -> bool {
        self.can_optimize_clean_setarrayitems()
    }

    fn gc_step(&mut self) -> bool {
        self.gc_step()
    }

    fn pin(&mut self, obj: GcRef) -> bool {
        self.pin(obj)
    }

    fn unpin(&mut self, obj: GcRef) {
        self.unpin(obj);
    }

    fn is_pinned(&self, obj: GcRef) -> bool {
        self.is_pinned(obj)
    }

    fn register_type(&mut self, info: TypeInfo) -> u32 {
        self.types.register(info)
    }

    fn type_count(&self) -> usize {
        self.types.len()
    }

    fn heap_byte_stats(&self) -> (usize, usize) {
        (self.get_total_memory_used(), self.nursery.used())
    }

    fn gc_memory_stats(&self) -> crate::GcMemoryStats {
        // incminimark.py get_stats. The nursery contribution is its reserved
        // capacity, not its current bump-pointer fill.
        let nursery_size = self.config.nursery_size;
        crate::GcMemoryStats {
            total_gc_memory: self.get_total_memory_used() + nursery_size,
            total_allocated_memory: self.oldgen.total_allocated_bytes() + nursery_size,
            peak_memory: self.oldgen.peak_used_bytes() + nursery_size,
            peak_allocated_memory: self.oldgen.peak_allocated_bytes() + nursery_size,
            total_arena_memory: self.oldgen.arenas_bytes(),
            total_rawmalloced_memory: self.oldgen.rawmalloced_bytes(),
            peak_arena_memory: self.oldgen.peak_arena_bytes(),
            peak_rawmalloced_memory: self.oldgen.peak_rawmalloced_bytes(),
            nursery_size,
            total_gc_time_ms: (self.total_gc_time * 1000.0) as usize,
        }
    }

    fn major_threshold_reached(&self) -> bool {
        self.threshold_reached(0)
    }

    fn type_size(&self, type_id: u32) -> Option<usize> {
        if (type_id as usize) < self.types.len() {
            Some(self.types.get(type_id).size)
        } else {
            None
        }
    }

    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, classptr: usize) -> Option<u32> {
        // gc.py GcLLDescr_framework
        //   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
        // RPython derives the typeid arithmetically:
        //   expected_typeid = classptr - sizeof_ti - type_info_group
        // pyre keeps an explicit vtable→type_id table populated via
        // register_vtable_for_type, mirroring the same contract.
        self.vtable_to_type_id.get(&classptr).copied()
    }

    fn register_vtable_for_type(&mut self, vtable: usize, type_id: u32) {
        self.vtable_to_type_id.insert(vtable, type_id);
    }

    /// gctypelayout.py `encode_type_shapes_now` parity:
    /// closes the type-registration phase on the underlying
    /// `TypeRegistry`. The materialized `type_info_group` base
    /// address is stable from this point on, and every is_object
    /// type's preorder `subclassrange_{min,max}` is computed via
    /// `assign_inheritance_ids` (normalizecalls.py).
    fn freeze_types(&mut self) {
        self.types.freeze_types();
    }

    fn assign_inheritance_ids_now(&mut self) {
        self.types.assign_inheritance_ids_now();
    }

    fn types_frozen(&self) -> bool {
        self.types.is_frozen()
    }

    /// Owns a `TypeRegistry`, so a shape id can always be resolved against
    /// this allocator — including before anything has been registered in it.
    fn has_type_registry(&self) -> bool {
        true
    }

    fn set_jitframe_type_id(&mut self, id: u32) {
        assert!(
            (id as usize) < self.types.len(),
            "JITFRAME type id {id} is not registered on this collector"
        );
        self.jitframe_type_id = Some(id);
    }

    fn jitframe_type_id(&self) -> Option<u32> {
        self.jitframe_type_id
    }

    /// gc.py `GcLLDescr_framework.supports_guard_gc_type = True`.
    /// MiniMarkGC owns a `TypeRegistry` that materializes a `TYPE_INFO`
    /// table on demand, so the framework-equivalent flag is always
    /// true here. (Boehm-equivalent stubs override the trait default
    /// and leave it `false`.)
    fn supports_guard_gc_type(&self) -> bool {
        true
    }

    /// gc.py `check_is_object` parity.
    ///
    /// Implemented line-by-line as the RPython source:
    ///     typeid = self.get_actual_typeid(gcptr)
    ///     base_type_info, shift_by, sizeof_ti = (
    ///         self.get_translated_info_for_typeinfo())
    ///     infobits_offset, IS_OBJECT_FLAG = (
    ///         self.get_translated_info_for_guard_is_object())
    ///     p = base_type_info + (typeid << shift_by) + infobits_offset
    ///     p = rffi.cast(rffi.CCHARP, p)
    ///     return (ord(p[0]) & IS_OBJECT_FLAG) != 0
    ///
    /// `get_actual_typeid` is the only majit-specific adaptation: managed
    /// objects carry a `GcHeader` immediately before the payload, while
    /// pyre's host-allocated foreign objects keep their classptr at
    /// offset 0 and rely on the `vtable→type_id` table populated via
    /// `register_vtable_for_type`.
    fn check_is_object(&self, gcref: GcRef) -> bool {
        if gcref.is_null() {
            return false;
        }
        // gc.py: typeid = self.get_actual_typeid(gcptr).
        let Some(typeid) = self.get_actual_typeid(gcref) else {
            return false;
        };
        // gc.py:636-637: gc_ll_descr lookups.
        let (base_type_info, shift_by, _sizeof_ti) = self.get_translated_info_for_typeinfo();
        let (infobits_offset, is_object_flag) = self.get_translated_info_for_guard_is_object();
        // gc.py:640-642: addr arithmetic + byte test.
        let typeid = typeid as usize;
        if typeid >= self.types.len() {
            return false;
        }
        let p = base_type_info + (typeid << shift_by) + infobits_offset;
        let byte = unsafe { *(p as *const u8) };
        (byte & is_object_flag) != 0
    }

    /// gc/base.py `is_valid_gc_object` tagged-immediate test.
    /// Delegates to the inherent guard the collector uses internally,
    /// exposing it through the trait so backend-agnostic callers can ask
    /// whether an odd-valued constant address is an unboxed immediate.
    fn is_tagged_immediate(&self, addr: usize) -> bool {
        MiniMarkGC::is_tagged_immediate(self, addr)
    }

    /// incminimark.py `can_move`: nursery membership alone. Pinning
    /// temporarily prevents a move but deliberately does not change this
    /// answer; callers use it to decide whether pinning is meaningful.
    fn can_move(&self, gcref: GcRef) -> bool {
        if gcref.is_null() || self.is_tagged_immediate(gcref.0) {
            return false;
        }
        self.is_in_nursery(gcref.0)
    }

    /// gc.py `get_translated_info_for_typeinfo` parity.
    /// Returns `(type_info_group_base, shift_by, sizeof_ti)` — the
    /// base address of the materialized `type_info_group` table, the
    /// SIB-style scale `genop_guard_guard_is_object` /
    /// `genop_guard_guard_subclass` apply to the typeid register
    /// (x86/assembler.py, 1967), and `rffi.sizeof(TYPE_INFO)`.
    ///
    /// RPython's 64-bit port sets `shift_by = 0` because its typeid
    /// is already `GROUP_MEMBER_OFFSET`, i.e. the raw byte offset of
    /// the member inside the group struct
    /// (translator/c/src/llgroup.h:36). majit keeps typeid as a
    /// small-integer index (returned by `register_type`) and encodes
    /// the per-entry stride here: `shift_by = log2(TypeEntry::STRIDE)`
    /// so the backend formula
    /// `base + (typeid << shift_by) + offset` lands on the correct
    /// `TypeEntry[typeid]` field. `sizeof_ti` stays equal to
    /// `rffi.sizeof(TYPE_INFO)` because that is the distance to the
    /// paired `CLASSTYPE` entry the `genop_guard_guard_subclass`
    /// formula needs (`+ sizeof_ti + offset2`
    /// x86/assembler.py, gctypelayout.py add_vtable_after_typeinfo
    /// `add_vtable_after_typeinfo`).
    fn get_translated_info_for_typeinfo(&self) -> (usize, u8, usize) {
        let table = self.types.type_info_table();
        let base = table.as_ptr() as usize;
        (base, TypeEntry::SHIFT_BY, TypeInfoLayout::SIZE_OF_TI)
    }

    /// gc.py `get_translated_info_for_guard_is_object` parity.
    /// Reads the instance state populated by `_setup_guard_is_object`
    /// and returns `(infobits_offset + infobits_offset_plus,
    /// T_IS_RPYTHON_INSTANCE_BYTE)`. Mirrors:
    ///
    /// ```python
    /// def get_translated_info_for_guard_is_object(self):
    ///     infobits_offset = rffi.cast(lltype.Signed, self._infobits_offset)
    ///     infobits_offset += self._infobits_offset_plus
    ///     return (infobits_offset, self._T_IS_RPYTHON_INSTANCE_BYTE)
    /// ```
    fn get_translated_info_for_guard_is_object(&self) -> (usize, u8) {
        let infobits_offset = self._infobits_offset + self._infobits_offset_plus;
        (infobits_offset, self._T_IS_RPYTHON_INSTANCE_BYTE)
    }

    /// x86/assembler.py `cpu.subclassrange_min_offset` parity.
    /// Byte offset of `subclassrange_min` inside `ClassTypeLayout`,
    /// the paired `rclass.CLASSTYPE` member that immediately follows
    /// the `TYPE_INFO` entry in the type_info_group
    /// (gctypelayout.py add_vtable_after_typeinfo).
    fn subclassrange_min_offset(&self) -> usize {
        ClassTypeLayout::SUBCLASSRANGE_MIN_OFFSET
    }

    /// x86/assembler.py:1971-1974 codegen-time bounds lookup parity.
    /// Resolves `classptr` through the registered vtable→typeid map and
    /// returns the type's `(subclassrange_min, subclassrange_max)`.
    fn subclass_range(&self, classptr: usize) -> Option<(i64, i64)> {
        let type_id = self.vtable_to_type_id.get(&classptr).copied()?;
        let info = self.types.get(type_id);
        Some((info.subclassrange_min, info.subclassrange_max))
    }

    /// Companion to `subclass_range` keyed by typeid. Used by the
    /// executor's `GuardSubclass` arm after it resolves `value.typeptr`
    /// via `get_actual_typeid` (llgraph/runner.py).
    fn typeid_subclass_range(&self, typeid: u32) -> Option<(i64, i64)> {
        if (typeid as usize) >= self.types.len() {
            return None;
        }
        let info = self.types.get(typeid);
        Some((info.subclassrange_min, info.subclassrange_max))
    }

    /// gc.py `get_actual_typeid` parity.
    ///
    /// RPython reads the typeid from the GC header word
    /// (`llop.extract_ushort(llgroup.HALFWORD, hdr.tid)`). MiniMarkGC's
    /// managed objects use the same layout — `GcHeader::SIZE` bytes
    /// before the payload, type id in the lower `TYPE_ID_BITS`. pyre's
    /// host-allocated "foreign" objects do not carry a `GcHeader`; for
    /// those we recover the type id by reading the classptr (offset 0)
    /// and looking it up in `vtable_to_type_id`, which is what
    /// `register_vtable_for_type` populated.
    fn get_actual_typeid(&self, gcref: GcRef) -> Option<u32> {
        if gcref.is_null() {
            return None;
        }
        // gc/base.py `is_valid_gc_object`: a tagged immediate carries no
        // header/classptr — reading offset 0 would deref the value bits.
        // Mirrors `can_move`'s guard.
        if self.is_tagged_immediate(gcref.0) {
            return None;
        }
        if self.is_managed_heap_object(gcref.0) {
            let header_addr = gcref.0.wrapping_sub(crate::header::GcHeader::SIZE);
            let header: crate::header::GcHeader = unsafe { *(header_addr as *const _) };
            return Some(header.type_id());
        }
        // Foreign object — read classptr at offset 0 and consult the
        // vtable→type_id table populated via register_vtable_for_type.
        let vtable = unsafe { *(gcref.0 as *const usize) };
        self.vtable_to_type_id.get(&vtable).copied()
    }

    /// Companion to `check_is_object` keyed by typeid. Reads the
    /// `T_IS_RPYTHON_INSTANCE` bit directly from the materialized
    /// TYPE_INFO entry (gctypelayout.py).
    fn typeid_is_object(&self, typeid: u32) -> Option<bool> {
        if (typeid as usize) >= self.types.len() {
            return None;
        }
        Some(self.types.get(typeid).is_object)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static SHADOW_STACK_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
    static SUPPLIED_ENV_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    /// A supplied environment answers a name the process does not define, and
    /// yields to one it does. The absent name is not in [`GC_ENV_NAMES`], so a
    /// concurrently built collector cannot see this table. `PATH` is read, not
    /// written: a parallel test must not mutate the process environment.
    #[test]
    fn supplied_env_fills_in_only_what_the_process_lacks() {
        let _guard = SUPPLIED_ENV_TEST_LOCK.lock();
        let absent = "MAJIT_TEST_SUPPLIED_ENV_ABSENT";
        let present = "PATH";
        let process_value = std::env::var(present).expect("PATH");
        majit_ir::environ::install(Vec::new());

        assert_eq!(read_uint_from_env(absent), None);
        set_supplied_env(vec![
            (absent.to_string(), "1m".to_string()),
            (present.to_string(), "4m".to_string()),
        ]);
        assert_eq!(read_uint_from_env(absent), Some(1024 * 1024));
        assert_eq!(env_var(present).as_deref(), Some(process_value.as_str()));

        majit_ir::environ::install(Vec::new());
        assert_eq!(read_uint_from_env(absent), None);
    }

    /// Presence matches `var_os.is_some()`: an empty supplied value still
    /// counts, and a unique name cannot collide with `MAJIT_GC_STRESS` that
    /// `with_config` reads.
    #[test]
    fn supplied_env_presence_matches_var_os() {
        let _guard = SUPPLIED_ENV_TEST_LOCK.lock();
        let name = "MAJIT_TEST_SUPPLIED_PRESENCE";
        majit_ir::environ::install(Vec::new());
        assert!(
            !env_is_set(name),
            "a unique name absent from the process and the embedder table is unset"
        );
        set_supplied_env(vec![(name.to_string(), String::new())]);
        assert!(
            env_is_set(name),
            "empty supplied value is still present, matching var_os.is_some"
        );
        majit_ir::environ::install(Vec::new());
        assert!(!env_is_set(name));
    }

    /// An array length whose eight-byte items carry a `total_size` past
    /// `test_gc(4096)`'s `large_object_threshold`. Cards are denied below it
    /// (incminimark.py:1019), so a fixture that means to exercise them has to
    /// ask for an array that would really have been rawmalloced.
    const CARD_ARRAY_LEN: usize = 512;

    /// Helper: create a GC with a small nursery for testing.
    fn test_gc(nursery_size: usize) -> MiniMarkGC {
        MiniMarkGC::with_config(GcConfig {
            nursery_size,
            large_object_threshold: nursery_size / 2,
            ..GcConfig::default()
        })
    }

    /// The ring's front arena is installed on every minor collection, so a
    /// rotating allocator has no range that stays fixed for its lifetime and
    /// nothing for `publish_singleton_nursery` to capture.
    #[test]
    fn a_rotating_nursery_publishes_no_fixed_bounds() {
        if !crate::nursery::HAS_PROTECT {
            return;
        }
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 4096,
            large_object_threshold: 2048,
            debug: 1,
            gc_nursery_debug: true,
            ..GcConfig::default()
        });

        let before = GcAllocator::nursery_bounds(&gc);
        gc.do_collect_nursery();
        let after = GcAllocator::nursery_bounds(&gc);

        assert_eq!(
            before, after,
            "the arena moved, so a pair published before the collection would name a retired one"
        );
        assert_eq!(after, None, "a rotating arena has no publishable range");
    }

    /// incminimark.py `_minor_collection`: on the `gc_nursery_debug` arm with
    /// no pinned survivor, the recycled arena is retired and the next one in
    /// the ring takes its place.
    #[test]
    fn a_minor_collection_rotates_the_nursery_under_gc_nursery_debug() {
        if !crate::nursery::HAS_PROTECT {
            return;
        }
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 4096,
            large_object_threshold: 2048,
            debug: 1,
            gc_nursery_debug: true,
            ..GcConfig::default()
        });
        assert_eq!(
            gc.nursery.debug_rotating_nurseries(),
            DEBUG_ROTATING_NURSERIES
        );

        let before = gc.nursery.start_ptr() as usize;
        gc.do_collect_nursery();
        let after = gc.nursery.start_ptr() as usize;

        assert_ne!(before, after, "the minor collection retired the arena");
        assert_eq!(
            gc.nursery.free_ptr() as usize,
            after,
            "allocation resumes at the start of the arena now installed"
        );
        assert_eq!(
            gc.published_nursery_top.load(Ordering::Relaxed),
            after + gc.nursery.size(),
            "compiled code reads the published top, which must follow"
        );
    }

    /// `post_setup` allocates the ring only `if self.DEBUG and
    /// llarena.has_protect`, so an ordinary run keeps its one arena and
    /// `debug_rotate_nursery` finds nothing to rotate to.
    #[test]
    fn no_ring_is_allocated_without_the_debug_level() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 4096,
            large_object_threshold: 2048,
            gc_nursery_debug: true,
            ..GcConfig::default()
        });
        assert_eq!(gc.config.debug, 0);
        assert_eq!(gc.nursery.debug_rotating_nurseries(), 0);

        let before = gc.nursery.start_ptr() as usize;
        gc.do_collect_nursery();
        assert_eq!(gc.nursery.start_ptr() as usize, before);
    }

    /// incminimark.py `free_young_rawmalloced_objects`: an
    /// oversized object nothing reached is freed by the minor, not carried to
    /// the next major.
    #[test]
    fn large_object_dies_at_a_minor() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let large = gc.config.large_object_threshold + 64;
        let baseline = gc.oldgen.rawmalloced_bytes();

        let obj = gc.alloc_with_type(tid, large);
        assert!(!obj.is_null());
        assert!(
            !gc.is_in_nursery(obj.0),
            "an oversized object skips the nursery"
        );
        assert!(
            gc.is_young_rawmalloced(obj.0),
            "an oversized object from a collecting entry point is born young"
        );
        assert!(gc.oldgen.rawmalloced_bytes() > baseline);

        gc.do_collect_nursery();

        assert!(
            !gc.is_young_rawmalloced(obj.0),
            "the young list is emptied by every minor"
        );
        assert_eq!(
            gc.oldgen.rawmalloced_bytes(),
            baseline,
            "an unreached young rawmalloced object is freed by the minor"
        );
    }

    /// `external_malloc` (incminimark.py) tests
    /// `threshold_reached(raw_malloc_usage(totalsize))` before a young
    /// rawmalloc birth and runs `minor_collection_with_major_progress` when
    /// it holds. A compiled loop that virtualizes every nursery allocation
    /// never fills the nursery, so this entry is the only collection trigger
    /// those frames have.
    #[test]
    fn young_nonmoving_allocations_are_reclaimed_once_the_threshold_is_reached() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let payload = 16;
        let nursery_used_before = gc.nursery.used();
        let minor_before = gc.collection_counts().0;
        let mut saw_collection = false;
        let mut live_after_collect = 0usize;
        for _ in 0..4096 {
            let obj = gc.alloc_young_nonmoving_typed(tid, payload);
            assert!(!obj.is_null());
            assert_eq!(
                gc.nursery.used(),
                nursery_used_before,
                "this entry must not bump the nursery"
            );
            if gc.collection_counts().0 > minor_before {
                saw_collection = true;
                live_after_collect = gc.oldgen.rawmalloced_bytes();
                break;
            }
        }
        assert!(
            saw_collection,
            "threshold_reached must run a minor before the heap grows without bound"
        );
        assert!(
            live_after_collect > 0,
            "the allocating object itself is born after the minor"
        );
        let one = OldGen::allocation_size(GcHeader::SIZE + payload);
        assert!(
            live_after_collect <= one * 4,
            "unrooted young rawmallocs die at the threshold minor, leftover={live_after_collect}"
        );
    }

    /// The compiled-code entry point.  `dynasm_malloc_array` and
    /// `dynasm_malloc_str` reach the collector through
    /// `GcAllocator::alloc_varsize_typed`, so an oversized array built inside a
    /// trace takes the same young birth `alloc_with_type` does.  Pinning it
    /// here keeps the two arms from drifting: the varsize wrapper computes a
    /// payload size and then defers, and a future short-circuit that skipped
    /// the shared body would silently put compiled allocations back in the
    /// old generation.
    #[test]
    fn an_oversized_varsize_allocation_is_born_young() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let item_size = std::mem::size_of::<usize>();
        let length = gc.config.large_object_threshold / item_size + 8;
        let baseline = gc.oldgen.rawmalloced_bytes();

        let obj = GcAllocator::alloc_varsize_typed(&mut gc, tid, item_size, item_size, length);
        assert!(!obj.is_null());
        assert!(
            gc.is_young_rawmalloced(obj.0),
            "an oversized varsize allocation is born young"
        );

        gc.do_collect_nursery();

        assert_eq!(
            gc.oldgen.rawmalloced_bytes(),
            baseline,
            "and dies at the next minor when nothing reached it"
        );
    }

    /// The public typed varsize entry must retain `external_malloc`'s card
    /// decision; otherwise the private card allocator exists only in tests and
    /// compiled pointer arrays are silently born cardless.
    #[test]
    fn an_oversized_pointer_array_uses_the_production_card_path() {
        let mut gc = test_gc(4096);
        let item_size = std::mem::size_of::<GcRef>();
        let tid = gc.register_type(TypeInfo::varsize(item_size, item_size, 0, true, Vec::new()));
        let length = CARD_ARRAY_LEN;
        let obj = GcAllocator::alloc_varsize_typed(&mut gc, tid, item_size, item_size, length);

        let hdr = unsafe { header_of(obj.0) };
        assert!(gc.is_young_rawmalloced(obj.0));
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) });
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_CARDS_SET) });
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
        assert_eq!(unsafe { *(obj.0 as *const usize) }, length);
    }

    #[test]
    fn a_rooted_young_card_array_forwards_a_child_on_its_marked_card() {
        let mut gc = test_gc(4096);
        let item_size = std::mem::size_of::<GcRef>();
        let array_tid =
            gc.register_type(TypeInfo::varsize(item_size, item_size, 0, true, Vec::new()));
        let child_tid = gc.register_type(TypeInfo::simple(item_size));
        let mut array = GcAllocator::alloc_varsize_typed(
            &mut gc,
            array_tid,
            item_size,
            item_size,
            CARD_ARRAY_LEN,
        );
        let child = gc.alloc_with_type(child_tid, item_size);
        unsafe { *((array.0 + item_size) as *mut GcRef) = child };
        gc.remember_young_pointer_from_array2(array, 0, gc.card_page_shift);
        unsafe { gc.add_root(&mut array) };

        gc.do_collect_nursery();

        let forwarded_child = unsafe { *((array.0 + item_size) as *const GcRef) };
        assert!(gc.is_managed_heap_object(forwarded_child.0));
        assert!(!gc.is_in_nursery(forwarded_child.0));
        assert_ne!(forwarded_child, child);
        gc.remove_root(&mut array);
    }

    /// The other side of the entry-point split, pinned because it is what
    /// decides how much of the interpreter the young birth reaches.
    ///
    /// `alloc_with_type_no_collect` exists for a caller that holds the
    /// returned pointer on the Rust stack without registering it as a root.
    /// That caller cannot tolerate a collection *at* the allocation, and it
    /// cannot tolerate one just afterwards either — a young birth would let
    /// the next minor free the block while the unrooted local still names it.
    /// So the born-old arm here is not an oversight the young birth forgot to
    /// convert; it is the property the entry point is for.
    ///
    /// This is also the reason the young arm changes nothing an interpreter
    /// workload can observe today. The oversized populations the host
    /// allocates reach the collector on this side of the split — a list's
    /// items block through `try_gc_alloc` (`object_array.rs`
    /// `alloc_items_block_gc`), an rbigint's digits through
    /// `alloc_oldgen_typed` — while the host callers of the *collecting*
    /// entry (`listobject.rs`, `unicodeobject.rs`, `weakref.rs`) all pass a
    /// fixed object-header size that is never oversized. Compiled code is the
    /// door that does reach it, through `alloc_varsize_typed` above.
    #[test]
    fn the_no_collect_entry_keeps_its_born_old_arm_when_oversized() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let large = gc.config.large_object_threshold + 64;

        let obj = gc.alloc_with_type_no_collect(tid, large);
        assert!(!obj.is_null());
        assert!(
            !gc.is_young_rawmalloced(obj.0),
            "the no-collect entry births an oversized object old"
        );
        let born_old_bytes = gc.oldgen.rawmalloced_bytes();

        // The same type at the same size through the *collecting* entry, so
        // neither assertion can pass by `is_young_rawmalloced` answering no to
        // everything: the split is between the two entry points, not between
        // two sizes or two types.
        let collecting = gc.alloc_with_type(tid, large);
        assert!(
            gc.is_young_rawmalloced(collecting.0),
            "the collecting entry births the same allocation young"
        );

        gc.do_collect_nursery();
        assert_eq!(
            gc.oldgen.rawmalloced_bytes(),
            born_old_bytes,
            "one minor, two fates: the young block dies unreached and the \
             born-old one survives with no root at all -- which is exactly \
             what the no-collect entry's unrooted Rust local depends on"
        );
    }

    /// incminimark.py `register_finalizer` appends to
    /// `probably_young_objects_with_finalizers` unconditionally, and that deque
    /// is what makes the registration a root for the next minor.
    /// `deal_with_young_objects_with_finalizers` is already written for a
    /// raw-malloced member: it stamps GCFLAG_VISITED_RMY instead of copying the
    /// object out, then hands it to the old deque itself.
    #[test]
    fn a_young_large_finalizer_registration_survives_its_first_minor() {
        fn trigger() {}

        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let large = gc.config.large_object_threshold + 64;

        let obj = gc.alloc_with_type(tid, large);
        assert!(
            gc.is_young_rawmalloced(obj.0),
            "the collecting entry births an oversized allocation young"
        );
        let registered_bytes = gc.oldgen.rawmalloced_bytes();
        GcAllocator::register_finalizer(&mut gc, 0, obj, trigger);

        // Deliberately unrooted: the registration is the only thing that can
        // keep this block alive, which is the property being tested.
        gc.do_collect_nursery();

        assert_eq!(
            gc.oldgen.rawmalloced_bytes(),
            registered_bytes,
            "a registered finalizer is a root for the minor, so the block is \
             promoted -- not freed with its address left on the old deque"
        );
        assert!(
            !gc.is_young_rawmalloced(obj.0),
            "and the promotion moves it off the young list rather than leaving \
             it to be freed by a later minor"
        );
    }

    /// The other half of the same rule: reached means promoted, and promoted
    /// means the SAME address — a young rawmalloced object never moves.
    #[test]
    fn rooted_large_object_is_promoted_in_place_and_then_dies() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let large = gc.config.large_object_threshold + 64;
        let baseline = gc.oldgen.rawmalloced_bytes();

        let mut root = gc.alloc_with_type(tid, large);
        let born_at = root.0;
        unsafe { gc.roots.add(&mut root) };
        gc.do_collect_nursery();

        assert_eq!(root.0, born_at, "a non-moving object keeps its address");
        assert!(!gc.is_young_rawmalloced(root.0), "it is old now");
        assert!(
            gc.oldgen.rawmalloced_bytes() > baseline,
            "a reached object survives its minor"
        );
        assert!(
            unsafe { !(*header_of(root.0)).has_flag(GcFlags::GCFLAG_VISITED_RMY) },
            "incminimark.py:2663 clears the flag on the survivor"
        );
        // A second minor must not free it either: it is on
        // `old_rawmalloced_objects` now, which only a major sweeps.
        gc.roots.clear();
        gc.do_collect_nursery();
        assert!(gc.oldgen.rawmalloced_bytes() > baseline);
    }

    /// A young raw-malloced object greyed by the non-moving major must leave
    /// it with the flags it arrived with.
    #[test]
    fn a_young_rawmalloced_object_leaves_a_nonmoving_major_with_its_flags_pristine() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let large = gc.config.large_object_threshold + 64;

        let mut obj = gc.alloc_with_type(tid, large);
        let addr = obj.0;
        unsafe { gc.roots.add(&mut obj) };
        assert!(gc.is_young_rawmalloced(addr));

        gc.do_collect_oldgen_nonmoving();

        assert!(
            gc.is_young_rawmalloced(addr),
            "the non-moving major neither moves nor promotes a young block"
        );
        let hdr = unsafe { header_of(addr) };
        assert!(
            unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_VISITED) },
            "and it leaves no mark behind: a VISITED that survives the cycle \
             is read by the NEXT major as `already greyed`, which skips the \
             object and leaves its children untraced"
        );
    }

    /// An unrooted young rawmalloc is garbage when the call that allocated it
    /// has returned (`pyframe.py class PyFrame(W_Root)`). `incminimark.py`
    /// frees it at the minor that leads every major; this entry skipped that
    /// minor and must still release the block, which does not move.
    #[test]
    fn nonmoving_major_frees_unvisited_young_rawmalloced_objects() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let large = gc.config.large_object_threshold + 64;
        let baseline = gc.oldgen.rawmalloced_bytes();

        let dead = gc.alloc_with_type(tid, large);
        let dead_addr = dead.0;
        assert!(gc.is_young_rawmalloced(dead_addr));
        assert!(gc.oldgen.rawmalloced_bytes() > baseline);

        // Not added to `roots`: the block is unreachable.
        gc.do_collect_oldgen_nonmoving();

        assert!(
            !gc.is_young_rawmalloced(dead_addr),
            "an unvisited young rawmalloc must be released; leaving it for a \
             minor that the interpreter may never run retains one PyFrame per call"
        );
        assert_eq!(gc.oldgen.rawmalloced_bytes(), baseline);
        assert!(!gc.oldgen.has_young_rawmalloced());
    }

    /// A non-moving entry that finds the collector already sweeping finishes
    /// that cycle before it marks anything. Young rawmallocs born after that
    /// cycle's `sweep_prepare` were not marked by it, so the drain must leave
    /// them in place. The fresh cycle then keeps a rooted one and releases
    /// an unrooted one.
    #[test]
    fn a_young_rawmalloc_born_during_sweep_survives_until_the_fresh_cycle() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let total_size = GcHeader::SIZE + 16;
        for _ in 0..900 {
            gc.alloc_in_oldgen_clear(tid, total_size);
        }

        gc.start_incremental_cycle();
        assert!(gc.incremental_mark_step());
        assert_eq!(gc.gc_state, GcState::Sweeping);

        // The heap is already past the test nursery's threshold. The births
        // below must not collect before the entry under test runs.
        gc.next_major_collection_threshold = 1_000_000_000.0;
        let large = gc.config.large_object_threshold + 64;
        let one = OldGen::allocation_size(GcHeader::SIZE + large);
        let mut rooted = gc.alloc_with_type(tid, large);
        let rooted_addr = rooted.0;
        unsafe { gc.roots.add(&mut rooted) };
        let unrooted = gc.alloc_with_type(tid, large);
        let unrooted_addr = unrooted.0;
        assert!(gc.is_young_rawmalloced(rooted_addr));
        assert!(gc.is_young_rawmalloced(unrooted_addr));
        assert_eq!(gc.oldgen.rawmalloced_bytes(), one * 2);
        assert_eq!(gc.gc_state, GcState::Sweeping);

        gc.do_collect_oldgen_nonmoving();

        assert_eq!(gc.gc_state, GcState::Scanning);
        assert!(!gc.oldgen_nonmoving_active);
        assert!(!gc.oldgen_nonmoving_marked);
        assert!(
            gc.is_young_rawmalloced(rooted_addr),
            "a rooted young rawmalloc allocated after sweep_prepare is not a \
             candidate of the cycle being drained"
        );
        assert_eq!(gc.oldgen.rawmalloced_bytes(), one);
        assert!(
            !gc.is_young_rawmalloced(unrooted_addr),
            "the fresh cycle still releases an unvisited young rawmalloc"
        );
        assert_eq!(gc.oldgen.object_count(), 1);
        gc.roots.clear();
    }

    /// The minor's destructor pass reads `GCFLAG_VISITED_RMY` and would treat
    /// a non-moving survivor as dead. This entry keeps a `GCFLAG_VISITED`
    /// young rawmalloc on the young list, runs the destructor of an unvisited
    /// one, and does not touch a nursery entry.
    #[test]
    fn nonmoving_major_destructs_only_unvisited_young_rawmallocs() {
        let _guard = DESTRUCTOR_TEST_LOCK.lock();
        DESTRUCTOR_RUNS.store(0, std::sync::atomic::Ordering::SeqCst);
        DESTRUCTOR_LAST_ADDR.store(0, std::sync::atomic::Ordering::SeqCst);

        let mut gc = test_gc(4096);
        gc.next_major_collection_threshold = 1_000_000_000.0;
        let tid = gc.register_type(TypeInfo::with_destructor(16, counting_destructor));
        let large = gc.config.large_object_threshold + 64;

        let nursery = gc.alloc_with_type(tid, 16);
        assert!(gc.is_in_nursery(nursery.0));
        let dead = gc.alloc_with_type(tid, large);
        let dead_addr = dead.0;
        let mut live = gc.alloc_with_type(tid, large);
        let live_addr = live.0;
        unsafe { gc.roots.add(&mut live) };
        assert!(gc.is_young_rawmalloced(dead_addr));
        assert!(gc.is_young_rawmalloced(live_addr));
        assert!(gc.young_objects_with_destructors.contains(&nursery.0));
        assert!(gc.young_objects_with_destructors.contains(&dead_addr));
        assert!(gc.young_objects_with_destructors.contains(&live_addr));

        gc.do_collect_oldgen_nonmoving();

        assert_eq!(destructor_runs(), 1);
        assert_eq!(
            DESTRUCTOR_LAST_ADDR.load(std::sync::atomic::Ordering::SeqCst),
            dead_addr
        );
        assert!(!gc.is_young_rawmalloced(dead_addr));
        assert!(!gc.young_objects_with_destructors.contains(&dead_addr));
        assert!(!gc.old_objects_with_destructors.contains(&dead_addr));

        assert!(gc.is_in_nursery(nursery.0));
        assert!(gc.young_objects_with_destructors.contains(&nursery.0));
        assert!(gc.is_young_rawmalloced(live_addr));
        assert!(gc.young_objects_with_destructors.contains(&live_addr));
        assert!(!gc.old_objects_with_destructors.contains(&live_addr));
        assert!(gc.old_objects_with_destructors.is_empty());
        gc.roots.clear();
    }

    /// Young rawmalloc mirrors are on the young lists. The old-list free does
    /// not see them, and freeing the block first would leave `ob_link` and
    /// `p_dict` naming released memory. A `GCFLAG_VISITED` target stays young
    /// and keeps its mirror; an unvisited one is unlinked first.
    #[test]
    fn nonmoving_major_unlinks_dead_young_rawmalloc_mirrors() {
        let mut gc = rrc_test_gc();
        gc.next_major_collection_threshold = 1_000_000_000.0;
        let tid = gc.register_type(TypeInfo::object(64));
        let large = gc.config.large_object_threshold + 64;

        let nursery = gc.alloc_with_type(tid, 64);
        assert!(gc.is_in_nursery(nursery.0));
        let nursery_mirror = test_mirror(rawrefcount::REFCNT_FROM_PYPY);
        gc.rawrefcount_create_link_pypy(nursery.0, nursery_mirror);

        let dead_p = gc.alloc_with_type(tid, large);
        let dead_p_addr = dead_p.0;
        let dead_p_mirror = test_mirror(rawrefcount::REFCNT_FROM_PYPY);
        gc.rawrefcount_create_link_pypy(dead_p_addr, dead_p_mirror);

        let dead_o = gc.alloc_with_type(tid, large);
        let dead_o_addr = dead_o.0;
        let dead_o_mirror = test_mirror(rawrefcount::REFCNT_FROM_PYPY);
        gc.rawrefcount_create_link_pyobj(dead_o_addr, dead_o_mirror);

        let held = gc.alloc_with_type(tid, large);
        let held_addr = held.0;
        let held_mirror = test_mirror(rawrefcount::REFCNT_FROM_PYPY + 1);
        gc.rawrefcount_create_link_pypy(held_addr, held_mirror);

        let mut rooted = gc.alloc_with_type(tid, large);
        let rooted_addr = rooted.0;
        unsafe { gc.roots.add(&mut rooted) };
        let rooted_mirror = test_mirror(rawrefcount::REFCNT_FROM_PYPY);
        gc.rawrefcount_create_link_pyobj(rooted_addr, rooted_mirror);

        assert!(gc.rrc.p_list_young.contains(&dead_p_mirror));
        assert!(gc.rrc.o_list_young.contains(&dead_o_mirror));
        assert!(gc.rrc.p_list_young.contains(&held_mirror));
        assert!(gc.rrc.o_list_young.contains(&rooted_mirror));
        assert_eq!(gc.rawrefcount_from_obj(dead_p_addr), dead_p_mirror);
        assert_eq!(gc.rawrefcount_from_obj(held_addr), held_mirror);
        assert_eq!(gc.rawrefcount_from_obj(nursery.0), nursery_mirror);

        gc.do_collect_oldgen_nonmoving();

        assert!(!gc.is_young_rawmalloced(dead_p_addr));
        assert!(!gc.is_young_rawmalloced(dead_o_addr));
        assert_eq!(mirror_link(dead_p_mirror), 0);
        assert_eq!(mirror_link(dead_o_mirror), 0);
        assert_eq!(gc.rawrefcount_from_obj(dead_p_addr), 0);
        assert!(!gc.rrc.p_list_young.contains(&dead_p_mirror));
        assert!(!gc.rrc.p_list_old.contains(&dead_p_mirror));
        assert!(!gc.rrc.o_list_young.contains(&dead_o_mirror));
        assert!(!gc.rrc.o_list_old.contains(&dead_o_mirror));
        let mut queued = vec![gc.rawrefcount_next_dead(), gc.rawrefcount_next_dead()];
        queued.sort_unstable();
        let mut expected = vec![dead_p_mirror, dead_o_mirror];
        expected.sort_unstable();
        assert_eq!(queued, expected);
        assert_eq!(gc.rawrefcount_next_dead(), 0);

        assert!(gc.is_young_rawmalloced(held_addr));
        assert_eq!(mirror_link(held_mirror), held_addr);
        assert_eq!(gc.rawrefcount_from_obj(held_addr), held_mirror);
        assert!(gc.rrc.p_list_young.contains(&held_mirror));
        assert!(!gc.rrc.p_list_old.contains(&held_mirror));

        assert!(gc.is_young_rawmalloced(rooted_addr));
        assert_eq!(mirror_link(rooted_mirror), rooted_addr);
        assert!(gc.rrc.o_list_young.contains(&rooted_mirror));
        assert!(!gc.rrc.o_list_old.contains(&rooted_mirror));

        assert!(gc.is_in_nursery(nursery.0));
        assert_eq!(mirror_link(nursery_mirror), nursery.0);
        assert_eq!(gc.rawrefcount_from_obj(nursery.0), nursery_mirror);
        assert!(gc.rrc.p_list_young.contains(&nursery_mirror));
        gc.roots.clear();
    }

    /// The consequence of that leak, two collections downstream.
    ///
    /// `grey_child` pushes only when `VISITED` is clear, so an object carrying
    /// a stale one is never traced by the next major and a child reachable
    /// solely through it is swept while live. The parent is reached AS A CHILD
    /// here — a rooted parent is greyed by the root walk, which does not
    /// consult the flag, so rooting it would test nothing. The child is
    /// allocated after the non-moving major on purpose: one reachable during
    /// it would carry the same stale flag and survive for the wrong reason.
    #[test]
    fn a_stale_visited_from_a_nonmoving_major_does_not_strand_the_next_majors_children() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let ptr_tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let large = gc.config.large_object_threshold + 64;

        // An old, rooted grandparent, so the parent below is reached through
        // `grey_child` rather than through the root walk.
        let mut grandparent = gc.alloc_with_type(ptr_tid, ptr_size);
        unsafe { *(grandparent.0 as *mut GcRef) = GcRef::NULL };
        unsafe { gc.roots.add(&mut grandparent) };
        gc.do_collect_nursery();
        let grandparent_addr = grandparent.0;
        assert!(!gc.is_in_nursery(grandparent_addr));

        let parent = gc.alloc_with_type(ptr_tid, large);
        let parent_addr = parent.0;
        unsafe { *(parent_addr as *mut GcRef) = GcRef::NULL };
        assert!(gc.is_young_rawmalloced(parent_addr));
        gc.do_write_barrier(grandparent);
        unsafe { *(grandparent_addr as *mut GcRef) = parent };

        // The cycle that stamps the parent. Its child does not exist yet.
        gc.do_collect_oldgen_nonmoving();

        let child = gc.alloc_with_type(ptr_tid, large);
        unsafe { *(child.0 as *mut GcRef) = GcRef::NULL };
        let with_child = gc.oldgen.rawmalloced_bytes();
        unsafe { *(parent_addr as *mut GcRef) = child };

        // Promote both. The parent carries whatever the major left on it.
        gc.do_collect_nursery();
        assert!(!gc.is_young_rawmalloced(parent_addr));

        // The next major, with the parent as the child's only route.
        gc.do_collect(2);

        assert_eq!(
            gc.oldgen.rawmalloced_bytes(),
            with_child,
            "the child is reachable only through the parent, so a major that \
             skipped greying the parent frees it while it is live"
        );
    }

    /// The write barrier is the only route by which an old parent keeps a
    /// young child alive, and it has to work for the non-moving young child
    /// too — the parent's entry is what makes the minor reach it.
    #[test]
    fn large_object_held_by_a_barriered_old_parent_survives() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let holder_tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let leaf_tid = gc.register_type(TypeInfo::simple(16));
        let large = gc.config.large_object_threshold + 64;

        // Promote the holder first, so the store below is a real old->young
        // edge rather than a young->young one.
        let mut holder = gc.alloc_with_type(holder_tid, ptr_size);
        unsafe { *(holder.0 as *mut GcRef) = GcRef::NULL };
        unsafe { gc.roots.add(&mut holder) };
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(holder.0));

        let baseline = gc.oldgen.rawmalloced_bytes();
        let child = gc.alloc_with_type(leaf_tid, large);
        assert!(gc.is_young_rawmalloced(child.0));
        gc.do_write_barrier(holder);
        unsafe { *(holder.0 as *mut GcRef) = child };

        gc.do_collect_nursery();
        assert!(
            gc.oldgen.rawmalloced_bytes() > baseline,
            "a young rawmalloced child of a barriered old parent survives"
        );
        assert_eq!(unsafe { *(holder.0 as *const GcRef) }.0, child.0);

        gc.roots.clear();
    }

    /// A weakref type is refused the young large birth: `malloc_fixedsize`'s
    /// large arm asserts `not contains_weakptr`, and `invalidate_young_weakrefs`
    /// decides survival by reading a forwarding pointer a non-moving object
    /// never has.
    #[test]
    #[cfg_attr(
        debug_assertions,
        should_panic(expected = "'contains_weakptr' specified for a large object")
    )]
    fn a_weakref_type_is_refused_the_young_large_birth() {
        let mut gc = test_gc(4096);
        let mut info = TypeInfo::simple(16);
        info.is_weakref = true;
        let tid = gc.register_type(info);
        let large = gc.config.large_object_threshold + 64;

        // Two tests wearing one name, because the exclusion is carried as a
        // debug assertion: with `debug_assertions` the call must fail with
        // upstream's own message, and without them it must still decline to
        // the born-old placement rather than take the young birth.
        let obj = gc.alloc_with_type(tid, large);
        assert!(!gc.is_young_rawmalloced(obj.0), "born old");
        assert!(
            unsafe { (*header_of(obj.0)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) },
            "and with the old generation's birth flags"
        );
    }

    /// A *light* destructor does take the young birth — `malloc_fixedsize`
    /// refuses only `needs_finalizer and not is_finalizer_light` — and the
    /// minor that never reaches it runs the destructor before
    /// `free_young_rawmalloced_objects` releases the block.
    #[test]
    fn an_unreached_large_destructor_object_is_destroyed_at_the_minor() {
        let _guard = DESTRUCTOR_TEST_LOCK.lock();
        DESTRUCTOR_RUNS.store(0, std::sync::atomic::Ordering::SeqCst);

        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_destructor(16, counting_destructor));
        let large = gc.config.large_object_threshold + 64;
        let baseline = gc.oldgen.rawmalloced_bytes();

        let obj = gc.alloc_with_type(tid, large);
        assert!(
            gc.is_young_rawmalloced(obj.0),
            "a light destructor reaches the large arm's alloc_young=True"
        );
        assert_eq!(gc.young_objects_with_destructors, vec![obj.0]);

        gc.collect_nursery();

        assert_eq!(destructor_runs(), 1, "unreached: the destructor runs");
        assert_eq!(
            DESTRUCTOR_LAST_ADDR.load(std::sync::atomic::Ordering::SeqCst),
            obj.0
        );
        assert!(gc.old_objects_with_destructors.is_empty());
        assert_eq!(
            gc.oldgen.rawmalloced_bytes(),
            baseline,
            "and the block is freed by the same minor"
        );
    }

    /// The other half: reached means promoted in place, the destructor does
    /// *not* run, and the entry moves to the old-destructor list at the same
    /// address — a non-moving object has no forwarding address to translate.
    #[test]
    fn a_reached_large_destructor_object_is_promoted_not_destroyed() {
        let _guard = DESTRUCTOR_TEST_LOCK.lock();
        DESTRUCTOR_RUNS.store(0, std::sync::atomic::Ordering::SeqCst);

        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_destructor(16, counting_destructor));
        let large = gc.config.large_object_threshold + 64;

        let mut root = gc.alloc_with_type(tid, large);
        let born_at = root.0;
        unsafe { gc.roots.add(&mut root) };

        gc.collect_nursery();

        assert_eq!(destructor_runs(), 0, "reached: the destructor must not run");
        assert_eq!(root.0, born_at, "a non-moving object keeps its address");
        assert!(!gc.is_young_rawmalloced(root.0), "it is old now");
        assert_eq!(gc.old_objects_with_destructors, vec![born_at]);
        gc.roots.clear();
    }

    /// A young rawmalloced object is generation 0, and `gc.get_objects(2)`
    /// must not report it as old while it can still die at a minor.
    #[test]
    fn a_young_rawmalloced_object_is_generation_zero() {
        let mut gc = test_gc(4096);
        let mut info = TypeInfo::simple(16);
        info.is_object = true;
        let tid = gc.register_type(info);
        let large = gc.config.large_object_threshold + 64;

        let mut root = gc.alloc_with_type(tid, large);
        let addr = root.0;
        unsafe { gc.roots.add(&mut root) };
        assert!(gc.is_young_rawmalloced(addr));
        assert!(!gc.is_in_nursery(addr));

        let collect = |gc: &mut MiniMarkGC, generation: i8| {
            let mut seen = Vec::new();
            gc.do_get_objects(generation, &mut |obj| seen.push(obj.0));
            seen
        };
        assert!(
            collect(&mut gc, 0).contains(&addr),
            "born into generation 0"
        );
        assert!(
            !collect(&mut gc, 2).contains(&addr),
            "not old while a minor can still free it"
        );

        gc.do_collect_nursery();

        assert_eq!(root.0, addr, "a non-moving object keeps its address");
        assert!(
            collect(&mut gc, 2).contains(&addr),
            "the promotion moves it"
        );
        assert!(!collect(&mut gc, 0).contains(&addr));

        gc.roots.clear();
    }

    #[test]
    fn root_set_removes_lifo_and_out_of_order_roots() {
        let mut roots = RootSet::new();
        let mut a = GcRef(1);
        let mut b = GcRef(2);
        let mut c = GcRef(3);
        let (a, b, c) = (
            &mut a as *mut GcRef,
            &mut b as *mut GcRef,
            &mut c as *mut GcRef,
        );
        unsafe {
            roots.add(a);
            roots.add(b);
            roots.add(c);
        }

        roots.remove(c);
        assert_eq!(roots.roots, vec![a, b]);
        roots.remove(a);
        assert_eq!(roots.roots, vec![b]);
        roots.remove(b);
        assert!(roots.is_empty());
    }

    #[test]
    fn root_set_out_of_order_remove_prefers_the_newest_duplicate() {
        let mut roots = RootSet::new();
        let mut a = GcRef(1);
        let mut b = GcRef(2);
        let (a, b) = (&mut a as *mut GcRef, &mut b as *mut GcRef);
        unsafe {
            roots.add(a);
            roots.add(b);
            roots.add(a);
            roots.add(b);
        }

        roots.remove(a);
        assert_eq!(roots.roots, vec![a, b, b]);
        roots.remove(b);
        assert_eq!(roots.roots, vec![a, b]);
        roots.remove(b);
        roots.remove(a);
        assert!(roots.is_empty());
    }

    #[test]
    fn gc_memory_stats_follow_incminimark_selectors() {
        let nursery_size = 4096;
        let mut gc = test_gc(nursery_size);
        let initial = GcAllocator::gc_memory_stats(&gc);
        assert_eq!(initial.nursery_size, nursery_size);
        assert_eq!(initial.total_gc_memory, nursery_size);
        assert_eq!(initial.total_allocated_memory, nursery_size);
        assert_eq!(initial.total_gc_time_ms, 0);

        let tid = gc.register_type(TypeInfo::object(64));
        let _small = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + 64);
        let _large = gc.alloc_in_oldgen_clear(
            tid,
            gc.oldgen.small_request_threshold() + std::mem::size_of::<usize>(),
        );
        let after = GcAllocator::gc_memory_stats(&gc);
        assert_eq!(
            after.total_gc_memory,
            gc.get_total_memory_used() + nursery_size
        );
        assert_eq!(after.total_arena_memory, gc.oldgen.arenas_bytes());
        assert_eq!(
            after.total_rawmalloced_memory,
            gc.oldgen.rawmalloced_bytes()
        );
        assert!(after.peak_memory >= after.total_gc_memory);
        assert!(after.peak_allocated_memory >= after.total_allocated_memory);

        gc.do_collect_nursery();
        assert!(GcAllocator::gc_memory_stats(&gc).total_gc_time_ms <= 60_000);
    }

    #[test]
    fn memory_pressure_matches_framework_field_and_incminimark_threshold() {
        let word = std::mem::size_of::<isize>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(word).with_memory_pressure_offset(0));
        assert_ne!(
            crate::trace::encode_type_shape(gc.types.get(tid), tid)
                & TypeInfoLayout::T_HAS_MEMORY_PRESSURE,
            0
        );

        let mut object = gc.alloc_with_type(tid, word);
        unsafe { gc.roots.add(&mut object) };
        let threshold = gc.next_major_collection_threshold;
        gc.do_add_memory_pressure(123, object);
        assert_eq!(unsafe { *(object.0 as *const isize) }, 123);
        assert_eq!(gc.do_count_memory_pressure(), 123);
        assert_eq!(
            gc.next_major_collection_threshold,
            threshold - 123.0 - (2 * std::mem::size_of::<usize>()) as f64
        );

        // The object-bearing transform uses bare_setfield, not +=, and a
        // negative estimate releases the object's previously owned pressure.
        gc.do_add_memory_pressure(-7, object);
        assert_eq!(gc.do_count_memory_pressure(), -7);

        // Upstream forces the next nursery slow path once pressure drives the
        // threshold below zero.
        gc.next_major_collection_threshold = 0.0;
        let top = gc.nursery.top_ptr();
        gc.do_add_memory_pressure(1, GcRef::NULL);
        assert_eq!(gc.nursery.free_ptr().cast_const(), top);
        assert_eq!(
            gc.published_nursery_top.load(Ordering::Acquire),
            top as usize
        );
        gc.roots.clear();
    }

    /// A born-old allocation that crosses the next-major threshold asks for a
    /// collection the way `external_malloc` (incminimark.py) does — and
    /// asks the consumer first, every time.
    ///
    /// The refusing arm is the load-bearing one. The threshold stays reached
    /// until a major completes, so arming past a consumer that will not answer
    /// re-arms on every following allocation; the compiled back-edge poll then
    /// fails continuously and the guard grows a bridge chain that stops
    /// returning to the dispatch loop, starving the rest of the eval-breaker
    /// word — signals included.
    #[test]
    fn born_old_allocation_past_the_threshold_asks_the_consumer_every_time() {
        fn refuse() -> bool {
            false
        }
        fn accept() -> bool {
            true
        }

        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::object(64));
        // Put the threshold within reach of any single allocation.
        gc.next_major_collection_threshold = gc.get_total_memory_used() as f64;
        let size = GcHeader::SIZE + 64;
        majit_ir::eval_breaker_word::take_gc();

        set_deferred_major_request_probe(None);
        gc.alloc_in_oldgen_clear(tid, size);
        assert!(
            !majit_ir::eval_breaker_word::take_gc(),
            "no consumer installed, so no request should be armed"
        );

        set_deferred_major_request_probe(Some(refuse));
        for _ in 0..4 {
            gc.alloc_in_oldgen_clear(tid, size);
        }
        assert!(
            !majit_ir::eval_breaker_word::take_gc(),
            "the consumer refused, so none of the four allocations may arm"
        );

        set_deferred_major_request_probe(Some(accept));
        gc.alloc_in_oldgen_clear(tid, size);
        assert!(majit_ir::eval_breaker_word::take_gc());

        set_deferred_major_request_probe(None);
        majit_ir::eval_breaker_word::take_gc();
    }

    #[test]
    fn get_objects_walks_referents_filters_rpython_structs_and_generations() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let mut holder_info = TypeInfo::object(ptr_size);
        holder_info.has_gc_ptrs = true;
        holder_info.gc_ptr_offsets = vec![0];
        let holder_tid = gc.register_type(holder_info);
        let raw_tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let leaf_tid = gc.register_type(TypeInfo::object(ptr_size));

        let leaf = gc.alloc_with_type(leaf_tid, ptr_size);
        let raw = gc.alloc_with_type(raw_tid, ptr_size);
        let mut holder = gc.alloc_with_type(holder_tid, ptr_size);
        unsafe {
            *(raw.0 as *mut GcRef) = leaf;
            *(holder.0 as *mut GcRef) = raw;
            gc.roots.add(&mut holder);
        }

        let get_objects = |gc: &mut MiniMarkGC, generation| {
            let mut result = Vec::new();
            gc.do_get_objects(generation, &mut |gcref| result.push(gcref));
            result
        };
        let all = get_objects(&mut gc, -1);
        assert_eq!(all.len(), 2);
        assert!(all.contains(&holder));
        assert!(all.contains(&leaf));
        assert_eq!(get_objects(&mut gc, 0).len(), 2);
        assert!(get_objects(&mut gc, 1).is_empty());
        assert!(get_objects(&mut gc, 2).is_empty());
        for object in [holder, raw, leaf] {
            assert!(!unsafe { (*header_of(object.0)).has_flag(GcFlags::GCFLAG_EXTRA) });
        }

        gc.do_collect_nursery();
        let raw = unsafe { *(holder.0 as *const GcRef) };
        let leaf = unsafe { *(raw.0 as *const GcRef) };
        assert!(get_objects(&mut gc, 0).is_empty());
        let old = get_objects(&mut gc, 2);
        assert_eq!(old.len(), 2);
        assert!(old.contains(&holder));
        assert!(old.contains(&leaf));
        for object in [holder, raw, leaf] {
            assert!(!unsafe { (*header_of(object.0)).has_flag(GcFlags::GCFLAG_EXTRA) });
        }
        gc.roots.clear();
    }

    #[cfg(unix)]
    #[test]
    fn dump_rpy_heap_uses_native_words_root_marker_and_type_indexes() {
        use std::os::fd::AsRawFd;

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let holder_tid = gc.register_type(TypeInfo::object_with_gc_ptrs(ptr_size, vec![0]));
        let leaf_tid = gc.register_type(TypeInfo::object(ptr_size));
        let leaf = gc.alloc_with_type(leaf_tid, ptr_size);
        let mut holder = gc.alloc_with_type(holder_tid, ptr_size);
        unsafe {
            *(holder.0 as *mut GcRef) = leaf;
            gc.roots.add(&mut holder);
        }

        let path = std::env::temp_dir().join(format!(
            "majit-gc-heap-dump-{}-{:p}",
            std::process::id(),
            &gc
        ));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)
            .unwrap();
        assert_eq!(gc.do_dump_rpy_heap(file.as_raw_fd()), Ok(true));
        drop(file);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(bytes.len() % std::mem::size_of::<isize>(), 0);
        let words: Vec<isize> = bytes
            .chunks_exact(std::mem::size_of::<isize>())
            .map(|chunk| isize::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        let marker = words.windows(4).position(|w| w == [0, 0, 0, -1]).unwrap();

        let holder_pos = words
            .iter()
            .position(|&word| word == holder.0 as isize)
            .unwrap();
        assert!(holder_pos < marker);
        assert_eq!(words[holder_pos + 1], holder_tid as isize + 1);
        assert_eq!(words[holder_pos + 2], ptr_size as isize);
        assert_eq!(words[holder_pos + 3], leaf.0 as isize);
        let leaf_pos = marker
            + 4
            + words[marker + 4..]
                .iter()
                .position(|&word| word == leaf.0 as isize)
                .unwrap();
        assert!(leaf_pos > marker);
        assert_eq!(words[leaf_pos + 1], leaf_tid as isize + 1);
        assert_eq!(words[leaf_pos + 3], -1);

        assert_eq!(gc.do_dump_rpy_heap(-1), Err(libc::EBADF));
        for object in [holder, leaf] {
            assert!(!unsafe { (*header_of(object.0)).has_flag(GcFlags::GCFLAG_EXTRA) });
        }
        assert!(gc.types.typeids_text().starts_with(b"member0"));
        assert_eq!(gc.types.typeids_list(), vec![0, 0, 1]);
        gc.roots.clear();
    }

    /// A hidden OBJECT is omitted from `get_objects` but remains a direct
    /// referent boundary (the execution-frame case). A DUMMY is not a live
    /// Python object and is still looked through, matching
    /// `referents._list_w_obj_referents`.
    #[test]
    fn get_referents_stops_at_hidden_object_and_looks_through_dummy() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let object_tid = gc.register_type(TypeInfo::object_with_gc_ptrs(ptr_size, vec![0]));
        let mut hidden_info = TypeInfo::object_with_gc_ptrs(ptr_size, vec![0]);
        hidden_info.hide_from_app_level_inspector = true;
        let hidden_tid = gc.register_type(hidden_info);

        // holder -> hidden -> leaf, and holder -> dummy -> tail.
        let leaf = gc.alloc_with_type(object_tid, ptr_size);
        let tail = gc.alloc_with_type(object_tid, ptr_size);
        let hidden = gc.alloc_with_type(hidden_tid, ptr_size);
        let dummy = gc.alloc_with_type(object_tid, ptr_size);
        let mut holder = gc.alloc_with_type(object_tid, ptr_size);
        unsafe {
            (*header_of(dummy.0)).set_flag(GcFlags::GCFLAG_DUMMY);
            *(hidden.0 as *mut GcRef) = leaf;
            *(dummy.0 as *mut GcRef) = tail;
            *(holder.0 as *mut GcRef) = hidden;
            gc.roots.add(&mut holder);
        }

        let mut referents = Vec::new();
        gc.do_get_referents(holder, &mut |gcref| referents.push(gcref));
        assert_eq!(referents, vec![hidden]);
        assert!(!gc.is_app_level_object_ref(hidden));
        assert!(gc.is_app_level_referent_boundary(hidden));

        unsafe { *(holder.0 as *mut GcRef) = dummy };
        let mut referents = Vec::new();
        gc.do_get_referents(holder, &mut |gcref| referents.push(gcref));
        assert_eq!(referents, vec![tail]);

        for object in [holder, hidden, dummy, leaf, tail] {
            assert!(!unsafe { (*header_of(object.0)).has_flag(GcFlags::GCFLAG_EXTRA) });
        }
        gc.roots.clear();
    }

    /// A typedef-less OBJECT (`referents.py try_cast_gcref_to_w_root` returns
    /// None) is expanded by `get_referents` and omitted from `get_objects`.
    /// Unlike `hide_from_app_level_inspector`, it is not a referent boundary.
    #[test]
    fn get_referents_looks_through_typedef_less_object() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let object_tid = gc.register_type(TypeInfo::object_with_gc_ptrs(ptr_size, vec![0]));
        let cell_tid = gc.register_type(
            TypeInfo::object_subclass_with_gc_ptrs(ptr_size, object_tid, vec![0])
                .without_app_level_typedef(),
        );

        let value = gc.alloc_with_type(object_tid, ptr_size);
        let cell = gc.alloc_with_type(cell_tid, ptr_size);
        let mut holder = gc.alloc_with_type(object_tid, ptr_size);
        unsafe {
            *(cell.0 as *mut GcRef) = value;
            *(holder.0 as *mut GcRef) = cell;
            gc.roots.add(&mut holder);
        }

        let mut referents = Vec::new();
        gc.do_get_referents(holder, &mut |gcref| referents.push(gcref));
        assert_eq!(referents, vec![value]);
        assert!(!gc.is_app_level_object_ref(cell));
        assert!(!gc.is_app_level_referent_boundary(cell));

        let mut objects = Vec::new();
        gc.do_get_objects(-1, &mut |gcref| objects.push(gcref));
        assert!(!objects.contains(&cell));
        assert!(objects.contains(&value));
        assert!(objects.contains(&holder));
        for object in [holder, cell, value] {
            assert!(!unsafe { (*header_of(object.0)).has_flag(GcFlags::GCFLAG_EXTRA) });
        }
        gc.roots.clear();
    }

    #[test]
    fn get_referents_looks_through_rpython_structs_and_stops_at_objects() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let pair_info = || TypeInfo::with_gc_ptrs(2 * ptr_size, vec![0, ptr_size]);
        let raw_tid = gc.register_type(pair_info());
        let mut object_info = pair_info();
        object_info.is_object = true;
        let object_tid = gc.register_type(object_info);

        // holder -> raw -> {near, far}; `far` in turn points back at `holder`,
        // which must not be reported: the walk stops at the first object.
        let near = gc.alloc_with_type(object_tid, 2 * ptr_size);
        let far = gc.alloc_with_type(object_tid, 2 * ptr_size);
        let raw = gc.alloc_with_type(raw_tid, 2 * ptr_size);
        let mut holder = gc.alloc_with_type(object_tid, 2 * ptr_size);
        unsafe {
            *(raw.0 as *mut GcRef) = near;
            *((raw.0 + ptr_size) as *mut GcRef) = far;
            *(far.0 as *mut GcRef) = holder;
            *(holder.0 as *mut GcRef) = raw;
            gc.roots.add(&mut holder);
        }

        let mut referents = Vec::new();
        gc.do_get_referents(holder, &mut |gcref| referents.push(gcref));
        assert_eq!(referents.len(), 2);
        assert!(referents.contains(&near));
        assert!(referents.contains(&far));
        assert!(!referents.contains(&raw));
        assert!(!referents.contains(&holder));
        for object in [holder, raw, near, far] {
            assert!(!unsafe { (*header_of(object.0)).has_flag(GcFlags::GCFLAG_EXTRA) });
        }

        // A self-referential object terminates instead of looping forever.
        let mut referents = Vec::new();
        gc.do_get_referents(far, &mut |gcref| referents.push(gcref));
        assert_eq!(referents, vec![holder]);
        gc.roots.clear();
    }

    #[test]
    fn get_referents_hides_inherited_frontend_type_edge_by_slot() {
        unsafe fn trace_pair(obj_addr: usize, visit: &mut dyn FnMut(*mut GcRef)) {
            visit(obj_addr as *mut GcRef);
            visit((obj_addr + std::mem::size_of::<GcRef>()) as *mut GcRef);
        }

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let root_tid = gc.register_type(
            TypeInfo::object_with_gc_ptrs(2 * ptr_size, vec![0])
                .with_app_level_inspector_hidden_edge(0),
        );
        let holder_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
            2 * ptr_size,
            root_tid,
            trace_pair,
        ));
        let class = gc.alloc_with_type(root_tid, 2 * ptr_size);
        let item = gc.alloc_with_type(root_tid, 2 * ptr_size);
        let mut holder = gc.alloc_with_type(holder_tid, 2 * ptr_size);
        unsafe {
            *(holder.0 as *mut GcRef) = class;
            *((holder.0 + ptr_size) as *mut GcRef) = item;
            gc.roots.add(&mut holder);
        }

        let mut referents = Vec::new();
        gc.do_get_referents(holder, &mut |gcref| referents.push(gcref));
        assert_eq!(referents, vec![item]);

        // Filtering is by the header slot, not by pointer identity: an actual
        // payload entry equal to the class object remains visible.
        unsafe { *((holder.0 + ptr_size) as *mut GcRef) = class };
        let mut referents = Vec::new();
        gc.do_get_referents(holder, &mut |gcref| referents.push(gcref));
        assert_eq!(referents, vec![class]);
        for object in [holder, class, item] {
            assert!(!unsafe { (*header_of(object.0)).has_flag(GcFlags::GCFLAG_EXTRA) });
        }
        gc.roots.clear();
    }

    /// `GCBase.trace` visits a custom trace and `offsets_to_gc_pointers`.
    /// A young object reachable only through a fixed offset of such a type
    /// survives a minor collection. `object_subclass_with_custom_trace`
    /// does not inherit parent offsets: those hooks already walk the base.
    #[test]
    fn custom_trace_and_fixed_offsets_are_both_traced() {
        unsafe fn trace_middle_word(obj_addr: usize, visit: &mut dyn FnMut(*mut GcRef)) {
            let word = std::mem::size_of::<GcRef>();
            visit((obj_addr + word) as *mut GcRef);
        }

        let word = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let parent_tid = gc.register_type(TypeInfo::object_with_gc_ptrs(2 * word, vec![0]));
        let holder_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs_and_custom_trace(
            3 * word,
            parent_tid,
            vec![2 * word],
            trace_middle_word,
        ));
        let plain_tid = gc.register_type(TypeInfo::object_subclass_with_custom_trace(
            2 * word,
            parent_tid,
            trace_middle_word,
        ));
        let leaf_tid = gc.register_type(TypeInfo::simple(word));

        {
            let holder_info = gc.types.get(holder_tid);
            assert!(holder_info.inherits_parent_offsets);
            assert_eq!(holder_info.gc_ptr_offsets, vec![0, 2 * word]);
            let mut storage = vec![0u8; 3 * word];
            let obj_addr = storage.as_mut_ptr() as usize;
            let mut slots = Vec::new();
            unsafe {
                holder_info.for_each_gc_ptr(obj_addr, |slot| slots.push(slot as usize));
            }
            assert_eq!(slots.len(), 3);
            assert!(slots.contains(&obj_addr));
            assert!(slots.contains(&(obj_addr + word)));
            assert!(slots.contains(&(obj_addr + 2 * word)));
        }
        assert!(gc.types.get(plain_tid).gc_ptr_offsets.is_empty());
        assert!(!gc.types.get(plain_tid).inherits_parent_offsets);

        let leaf = gc.alloc_with_type(leaf_tid, word);
        unsafe {
            *(leaf.0 as *mut usize) = 0xC0FFEE;
        }
        let mut holder = gc.alloc_with_type(holder_tid, 3 * word);
        unsafe {
            *(holder.0 as *mut GcRef) = GcRef::NULL;
            *((holder.0 + word) as *mut GcRef) = GcRef::NULL;
            *((holder.0 + 2 * word) as *mut GcRef) = leaf;
            gc.roots.add(&mut holder);
        }
        let leaf_addr = leaf.0;
        gc.do_collect_nursery();
        let kept = unsafe { *((holder.0 + 2 * word) as *const GcRef) };
        assert_ne!(kept.0, leaf_addr);
        assert!(!gc.is_in_nursery(kept.0));
        assert!(gc.oldgen.contains(kept.0));
        assert_eq!(unsafe { *(kept.0 as *const usize) }, 0xC0FFEE);
        assert!(unsafe { *(holder.0 as *const GcRef) }.is_null());
        assert!(unsafe { *((holder.0 + word) as *const GcRef) }.is_null());
        gc.roots.clear();
    }

    #[test]
    fn get_referents_reports_prebuilt_foreign_objects() {
        #[repr(C)]
        struct ForeignLeaf {
            vtable: usize,
        }

        #[repr(C)]
        struct ForeignHolder {
            vtable: usize,
            child: GcRef,
        }

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let leaf_vtable = 0x1000usize;
        let holder_vtable = 0x2000usize;
        let leaf_tid = gc.register_type(TypeInfo::object(ptr_size));
        let holder_tid =
            gc.register_type(TypeInfo::object_with_gc_ptrs(2 * ptr_size, vec![ptr_size]));
        crate::GcAllocator::register_vtable_for_type(&mut gc, leaf_vtable, leaf_tid);
        crate::GcAllocator::register_vtable_for_type(&mut gc, holder_vtable, holder_tid);

        let leaf = Box::new(ForeignLeaf {
            vtable: leaf_vtable,
        });
        let holder = Box::new(ForeignHolder {
            vtable: holder_vtable,
            child: GcRef((&*leaf as *const ForeignLeaf) as usize),
        });
        let holder_ref = GcRef((&*holder as *const ForeignHolder) as usize);
        let leaf_ref = GcRef((&*leaf as *const ForeignLeaf) as usize);

        let mut referents = Vec::new();
        gc.do_get_referents(holder_ref, &mut |gcref| referents.push(gcref));
        assert_eq!(referents, vec![leaf_ref]);
    }

    #[test]
    fn rpy_inspector_keeps_raw_roots_and_referents_unexpanded() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let raw_tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let object_tid = gc.register_type(TypeInfo::object_with_gc_ptrs(ptr_size, vec![0]));
        let mut hidden_info = TypeInfo::object(ptr_size);
        hidden_info.hide_from_app_level_inspector = true;
        let hidden_tid = gc.register_type(hidden_info);

        let leaf = gc.alloc_with_type(object_tid, ptr_size);
        let raw = gc.alloc_with_type(raw_tid, ptr_size);
        let hidden = gc.alloc_with_type(hidden_tid, ptr_size);
        let mut holder = gc.alloc_with_type(object_tid, ptr_size);
        unsafe {
            *(raw.0 as *mut GcRef) = leaf;
            *(holder.0 as *mut GcRef) = raw;
            gc.roots.add(&mut holder);
        }

        let mut roots = Vec::new();
        assert!(gc.do_get_rpy_roots(&mut |root| roots.push(root)));
        assert!(roots.contains(&holder));
        assert_eq!(roots.last(), Some(&GcRef::NULL));
        assert!(roots.len() > roots.iter().filter(|root| !root.is_null()).count());

        let mut referents = Vec::new();
        assert!(gc.do_get_rpy_referents(holder, &mut |child| { referents.push(child) }));
        assert_eq!(referents, vec![raw]);
        assert!(gc.is_app_level_object_ref(holder));
        assert!(!gc.is_app_level_object_ref(raw));
        assert!(!gc.is_app_level_object_ref(hidden));
        gc.roots.clear();
    }

    #[test]
    fn is_tracked_follows_the_type_not_the_heap_the_instance_landed_in() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let atomic_tid = gc.register_type(TypeInfo::object(ptr_size));
        let holder_tid = gc.register_type(TypeInfo::object_subclass_with_gc_ptrs(
            ptr_size,
            atomic_tid,
            vec![0],
        ));

        let atomic = gc.alloc_with_type(atomic_tid, ptr_size);
        let mut holder = gc.alloc_with_type(holder_tid, ptr_size);
        unsafe {
            *(holder.0 as *mut GcRef) = atomic;
            gc.roots.add(&mut holder);
        }
        assert!(!gc.object_is_tracked(atomic.0));
        assert!(gc.object_is_tracked(holder.0));

        // The same two types answer the same way once promoted out of the
        // nursery — the heap the instance landed in must not matter.
        gc.do_collect_nursery();
        let atomic = unsafe { *(holder.0 as *const GcRef) };
        assert!(!gc.object_is_tracked(atomic.0));
        assert!(gc.object_is_tracked(holder.0));

        // An address the collector does not own is never tracked.
        let mut off_heap = 0usize;
        assert!(!gc.object_is_tracked(&mut off_heap as *mut usize as usize));
        gc.roots.clear();
    }

    #[test]
    fn public_rgc_flags_use_the_objects_header_without_a_side_table() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::object(32));
        let obj = gc.alloc_with_type(tid, 32);

        assert!(!gc.get_gcflag_extra(obj));
        gc.toggle_gcflag_extra(obj);
        assert!(gc.get_gcflag_extra(obj));
        gc.toggle_gcflag_extra(obj);
        assert!(!gc.get_gcflag_extra(obj));

        assert!(!gc.get_gcflag_dummy(obj));
        unsafe { (*header_of(obj.0)).set_flag(GcFlags::GCFLAG_DUMMY) };
        assert!(gc.get_gcflag_dummy(obj));
    }

    #[test]
    fn rpy_introspection_reports_payload_size_and_positive_type_group_index() {
        let mut gc = test_gc(4096);
        let fixed_tid = gc.register_type(TypeInfo::simple(24));
        let var_tid = gc.register_type(TypeInfo::varsize(8, 3, 0, false, Vec::new()));

        let fixed = gc.alloc_with_type(fixed_tid, 24);
        let var = gc.alloc_varsize_typed(var_tid, 8, 3, 2);
        unsafe { *(var.0 as *mut usize) = 2 };

        // inspector.py:get_rpy_memory_usage calls get_size_incl_hash, whose
        // incminimark implementation excludes the GC header. Variable objects
        // are rounded for the following arena allocation (8 + 2 * 3 -> 16).
        assert_eq!(gc.rpy_memory_usage(fixed), Some(24));
        assert_eq!(gc.rpy_memory_usage(var), Some(16));
        assert_eq!(gc.object_total_size(fixed.0), GcHeader::SIZE + 24);

        // TypeLayoutBuilder reserves group member zero as a dummy.
        assert_eq!(gc.rpy_type_index(fixed), Some(1));
        assert_eq!(gc.rpy_type_index(var), Some(2));
        assert_eq!(gc.rpy_memory_usage(GcRef::NULL), None);
        assert_eq!(gc.rpy_type_index(GcRef::NULL), None);
    }

    #[test]
    fn get_objects_includes_live_objects_registered_with_finalizers() {
        fn trigger() {}

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let object_tid = gc.register_type(TypeInfo::object(ptr_size));
        let object = gc.alloc_with_type(object_tid, ptr_size);
        gc.register_finalizer(0, object, trigger);

        let mut objects = Vec::new();
        gc.do_get_objects(-1, &mut |gcref| objects.push(gcref));
        assert!(objects.contains(&object));
    }

    #[test]
    fn write_barrier_descr_exposes_minimark_card_geometry() {
        let gc = test_gc(4096);
        let descr = gc
            .get_write_barrier_descr()
            .expect("MiniMark must expose its write-barrier descriptor");
        assert_eq!(
            descr.jit_wb_if_flag,
            GcFlags::GCFLAG_TRACK_YOUNG_PTRS.bits()
        );
        assert_eq!(descr.jit_wb_cards_set, GcFlags::GCFLAG_CARDS_SET.bits());
        assert_eq!(descr.jit_wb_card_page_shift, gc.card_page_shift);
    }

    #[test]
    fn write_barrier_descr_disables_cards_with_collector_config() {
        let gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 4096,
            large_object_threshold: 2048,
            card_page_indices: 0,
            ..GcConfig::default()
        });
        let descr = gc
            .get_write_barrier_descr()
            .expect("MiniMark must retain its generic write barrier");
        assert_eq!(
            descr.jit_wb_if_flag,
            GcFlags::GCFLAG_TRACK_YOUNG_PTRS.bits()
        );
        assert_eq!(descr.jit_wb_cards_set, 0);
        assert_eq!(descr.jit_wb_card_page_shift, 0);
    }

    #[test]
    fn test_basic_alloc() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_with_type(0, 16);
        assert!(!obj.is_null());
        assert!(gc.is_in_nursery(obj.0));
    }

    #[test]
    fn collect_and_reserve_uses_gap_before_pinned_object() {
        fn arrange_pinned_tail(gc: &mut MiniMarkGC) -> (u32, GcRef) {
            let pinned_tid = gc.register_type(TypeInfo::simple(440));
            let result_tid = gc.register_type(TypeInfo::simple(400));
            let _dead_prefix = gc.alloc_with_type(pinned_tid, 440);
            let pinned = gc.alloc_with_type(pinned_tid, 440);
            assert!(gc.pin(pinned));
            (result_tid, pinned)
        }

        // incminimark.py collect_and_reserve walks back to the free gap before a pinned
        // object. The old fallback returned an old-gen object here, violating
        // the fresh nursery allocation contract used by the GC rewrite.
        let mut gc = test_gc(1024);
        let (result_tid, mut pinned) = arrange_pinned_tail(&mut gc);
        unsafe { gc.roots.add(&mut pinned) };
        let result = gc.alloc_with_type(result_tid, 400);
        assert!(gc.is_in_nursery(result.0));
        assert!(result.0 < pinned.0);
        assert_eq!(
            unsafe { &*(gc.nursery_top_addr() as *const AtomicUsize) }.load(Ordering::Acquire),
            pinned.0 - GcHeader::SIZE,
            "compiled allocators must stop at the same pinned barrier",
        );
        gc.roots.clear();

        // The rooted slow path has the same collect-and-reserve contract and
        // must continue reporting that initialization needs no write barrier.
        let mut gc = test_gc(1024);
        let (result_tid, mut pinned) = arrange_pinned_tail(&mut gc);
        unsafe { gc.roots.add(&mut pinned) };
        let mut root = GcRef::NULL;
        let mut needs_write_barrier = true;
        let result = unsafe {
            gc.alloc_with_type_rooted(result_tid, 400, &mut root, &mut needs_write_barrier)
        };
        assert!(gc.is_in_nursery(result.0));
        assert!(result.0 < pinned.0);
        assert!(!needs_write_barrier);
        gc.roots.clear();
    }

    #[test]
    fn collect_and_reserve_consumes_minor_built_barriers_in_order() {
        let mut gc = test_gc(2048);
        gc.max_number_of_pinned_objects = 2;
        let pinned_tid = gc.register_type(TypeInfo::simple(440));
        let result_tid = gc.register_type(TypeInfo::simple(400));
        let _dead_prefix = gc.alloc_with_type(pinned_tid, 440);
        let mut first_pin = gc.alloc_with_type(pinned_tid, 440);
        let _dead_middle = gc.alloc_with_type(pinned_tid, 440);
        let mut second_pin = gc.alloc_with_type(pinned_tid, 440);
        assert!(gc.pin(first_pin));
        assert!(gc.pin(second_pin));
        unsafe {
            gc.roots.add(&mut first_pin);
            gc.roots.add(&mut second_pin);
        }

        // The tail cannot satisfy this request, so one minor rebuilds barriers
        // and serves the free prefix before the first pin.
        let before = gc.minor_collections;
        let prefix = gc.alloc_with_type(result_tid, 400);
        assert_eq!(gc.minor_collections, before + 1);
        assert!(prefix.0 < first_pin.0);

        // `unpin` does not rebuild `nursery_barriers`. Just like upstream,
        // collect_and_reserve consumes that now-stale first barrier, skips the
        // intact object bytes, and enters the gap before the second pin without
        // running another minor collection.
        gc.unpin(first_pin);
        let between = gc.alloc_with_type(result_tid, 400);
        assert_eq!(gc.minor_collections, before + 1);
        assert!(between.0 > first_pin.0);
        assert!(between.0 < second_pin.0);

        gc.roots.clear();
    }

    /// The max-heap tests read and write the process-global eval-breaker bit,
    /// and each one starts by normalising it. Ownership of an armed
    /// `MemoryError` is per-thread now, so a normalising call no longer clears
    /// one a concurrently running test armed — serialise them instead, the way
    /// the destructor and shadow-stack tests serialise their own shared state.
    static MAX_HEAP_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    /// incminimark.py:2601-2615 `PYPY_GC_MAX` out-of-memory policy: a bounded
    /// major collection over `max_heap_size` signals OOM the first time
    /// (`oom_pending` so the triggering allocation returns NULL) and aborts on
    /// the second occurrence. `max_heap_size` below `min_heap_size` makes any
    /// threshold bounded, so the decision fires on an otherwise empty heap.
    #[test]
    fn bounded_max_heap_size_signals_oom_then_aborts() {
        let _guard = MAX_HEAP_TEST_LOCK.lock();
        let mut gc = test_gc(4096);
        // PYPY_GC_MAX = 1 byte (below min_heap_size), so set_major_threshold_from
        // caps at 1 and reports `bounded`.
        gc.max_heap_size = 1.0;
        // The allocation that triggered the collection is larger than the
        // remaining headroom (1 - total_memory_used), so threshold_reached holds.
        gc.pending_reserving_size = 4096;
        // The dispatch-loop channel is process-global; start from a known state
        // so the assertion below reads this breach and not an earlier one.
        majit_ir::eval_breaker_word::take_memory_error();

        // First bounded breach: flag + signal, no abort.
        gc.gc_state = GcState::Sweeping;
        gc.finish_incremental_cycle();
        assert!(
            gc.max_heap_size_already_raised,
            "first bounded breach records max_heap_size_already_raised"
        );
        assert!(
            std::mem::take(&mut gc.oom_pending),
            "first bounded breach asks the allocation to fail (NULL)"
        );
        // The allocation is waiting and will take `oom_pending` the moment this
        // collection returns, so the deferred channel must stay clear. Arming
        // both would leave one latched onto an allocation that breached nothing.
        assert!(
            !majit_ir::eval_breaker_word::take_memory_error(),
            "an allocation-driven breach owes the dispatch loop nothing"
        );

        // Second bounded breach aborts (out_of_memory -> fatalerror == panic).
        // The take above is what the waiting allocation does the moment the
        // collection returns, and it is what makes this a second arrival.
        gc.pending_reserving_size = 4096;
        gc.gc_state = GcState::Sweeping;
        let second = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gc.finish_incremental_cycle();
        }));
        assert!(second.is_err(), "second bounded breach aborts the process");
    }

    /// A breach that lands while the previous `MemoryError` is still owed is
    /// the same arrival, not a second one.
    ///
    /// Upstream cannot reach this state: it raises where it sets the flag, so
    /// the program holds the exception before any further collection can run.
    /// Pyre's channels deliver later, and a driver that completes two cycles
    /// without returning to a dispatch — `do_collect_full`, i.e. an explicit
    /// `gc.collect()` — puts a whole second cycle inside that window. Aborting
    /// there would end the process on a heap limit the program was never told
    /// about.
    #[test]
    fn a_breach_while_the_memory_error_is_still_owed_is_not_a_second_arrival() {
        let _guard = MAX_HEAP_TEST_LOCK.lock();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(4096));
        let obj = gc.alloc_with_type(tid, 4096);
        assert!(!obj.is_null());
        gc.max_heap_size = 1.0;
        gc.pending_reserving_size = 0;
        majit_ir::eval_breaker_word::take_memory_error();

        gc.gc_state = GcState::Sweeping;
        gc.finish_incremental_cycle();
        assert!(gc.max_heap_size_already_raised);

        // Same breach again with the bit still armed: no abort, and the bit is
        // left alone for the dispatch loop that has yet to run.
        gc.gc_state = GcState::Sweeping;
        gc.finish_incremental_cycle();
        assert_eq!(gc.gc_state, GcState::Scanning);
        assert!(
            majit_ir::eval_breaker_word::take_memory_error(),
            "the undelivered MemoryError is still owed after the second breach"
        );

        // Delivered now, so the next breach is a genuine second arrival.
        gc.gc_state = GcState::Sweeping;
        let third = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gc.finish_incremental_cycle();
        }));
        assert!(
            third.is_err(),
            "a breach after delivery takes the fatal rung"
        );
        // Whatever that breach armed on its way out belongs to no one now.
        majit_ir::eval_breaker_word::take_memory_error();
    }

    /// A breach on a thread that is owed nothing is new, however loudly another
    /// thread's exception is still pending.
    ///
    /// The two questions the ladder asks are deliberately different widths. "Is
    /// this the same arrival?" is about the breaching thread, because an
    /// exception owed to some other thread says nothing about this collection —
    /// reading the process-wide summary there would let this thread allocate on
    /// past the limit with nothing owed to it. "May the process die now?" is
    /// process-wide, because the fatal rung must not fire while any thread is
    /// still to be told.
    #[test]
    fn a_breach_owed_to_another_thread_arms_this_one_rather_than_silencing_it() {
        let _guard = MAX_HEAP_TEST_LOCK.lock();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(4096));
        let obj = gc.alloc_with_type(tid, 4096);
        assert!(!obj.is_null());
        majit_ir::eval_breaker_word::take_memory_error();

        // Another thread breaches first and has not reached its own dispatch
        // loop yet — the window a stop-the-world collection opens, since its
        // guard resumes the other mutators before the collecting one returns.
        // It stays parked until this thread has breached, and then delivers,
        // so nothing is left owed to a thread that no longer exists.
        let (armed_tx, armed_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let other = std::thread::spawn(move || {
            majit_ir::eval_breaker_word::set_memory_error();
            armed_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            assert!(
                majit_ir::eval_breaker_word::take_memory_error(),
                "the other thread's own exception is still owed to it"
            );
        });
        armed_rx.recv().unwrap();
        // Burn the one grace event on that thread's behalf, so this breach can
        // only reach the fatal rung or the arming below.
        gc.max_heap_size_already_raised = true;

        gc.max_heap_size = 1.0;
        gc.pending_reserving_size = 0;
        gc.gc_state = GcState::Sweeping;
        gc.finish_incremental_cycle();

        assert_eq!(
            gc.gc_state,
            GcState::Scanning,
            "the fatal rung must not fire while a thread is still owed one"
        );
        assert!(
            majit_ir::eval_breaker_word::take_memory_error(),
            "this thread breached and had none owed to it, so it is owed one now"
        );
        assert!(
            majit_ir::eval_breaker_word::memory_error_armed(),
            "delivering this thread's does not clear the one the other is owed"
        );

        release_tx.send(()).unwrap();
        other.join().unwrap();
        assert!(
            !majit_ir::eval_breaker_word::memory_error_armed(),
            "the last owner to deliver clears the summary"
        );
    }

    /// The other half of the same policy: a collection with no allocation
    /// waiting on it.
    ///
    /// This is the dispatch-loop safepoint's collection — where the interpreter
    /// path's majors actually happen — and also `gc.collect()`'s. Upstream
    /// raises out of all of them alike; pyre has nothing to fail here, so the
    /// breach has to reach the eval-breaker instead or it is silent, and the
    /// program's only sign of a full heap becomes the abort on the next one.
    #[test]
    fn a_breach_with_no_allocation_waiting_defers_to_the_dispatch_loop() {
        let _guard = MAX_HEAP_TEST_LOCK.lock();
        let mut gc = test_gc(4096);
        // What separates this from the test above: no allocation is waiting, so
        // `threshold_reached(0)` decides on the heap as it stands, and the heap
        // has to actually hold something for it to decide yes. One object past
        // `large_object_threshold` is born in the old generation, which is what
        // `get_total_memory_used` counts.
        let tid = gc.register_type(TypeInfo::simple(4096));
        let obj = gc.alloc_with_type(tid, 4096);
        assert!(!obj.is_null());
        gc.max_heap_size = 1.0;
        gc.pending_reserving_size = 0;
        majit_ir::eval_breaker_word::take_memory_error();

        gc.gc_state = GcState::Sweeping;
        gc.finish_incremental_cycle();
        assert!(
            gc.max_heap_size_already_raised,
            "a driverless breach still burns the one grace event"
        );
        assert!(
            !gc.oom_pending,
            "there is no allocation to fail, so nothing may be latched for the next one"
        );
        assert!(
            majit_ir::eval_breaker_word::take_memory_error(),
            "a driverless breach owes the dispatch loop a MemoryError"
        );
    }

    /// The default (unbounded, `max_heap_size == 0`) config never sets `bounded`,
    /// so a completed major collection leaves the OOM signals untouched.
    #[test]
    fn unbounded_heap_never_signals_oom() {
        let _guard = MAX_HEAP_TEST_LOCK.lock();
        let mut gc = test_gc(4096);
        assert_eq!(gc.max_heap_size, 0.0);
        gc.pending_reserving_size = 4096;
        gc.gc_state = GcState::Sweeping;
        gc.finish_incremental_cycle();
        assert!(!gc.max_heap_size_already_raised);
        assert!(!gc.oom_pending);
        assert_eq!(gc.gc_state, GcState::Finalizing);
    }

    /// incminimark.py:2617-2631 keeps finalizer execution in its own major
    /// step. In particular, a handler cannot run in the SWEEPING ->
    /// FINALIZING transition; the following step first publishes SCANNING and
    /// only then invokes it, so a recursive collection can start safely.
    #[test]
    fn finalizer_triggers_run_in_a_distinct_finalizing_step() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static TRIGGERS: AtomicUsize = AtomicUsize::new(0);
        fn trigger() {
            TRIGGERS.fetch_add(1, Ordering::Relaxed);
        }

        TRIGGERS.store(0, Ordering::Relaxed);
        let mut gc = test_gc(4096);
        gc.finalizer_handlers.push(FinalizerHandler {
            deque: VecDeque::from([0xfeed]),
            trigger,
        });
        gc.gc_state = GcState::Sweeping;

        gc.finish_incremental_cycle();
        assert_eq!(gc.gc_state, GcState::Finalizing);
        assert_eq!(TRIGGERS.load(Ordering::Relaxed), 0);

        gc.major_collection_step();
        assert_eq!(gc.gc_state, GcState::Scanning);
        assert_eq!(TRIGGERS.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn explicit_collect_step_runs_one_minor_and_one_major_transition() {
        use crate::GcStepTransition as Step;

        let mut gc = test_gc(4096);
        gc.disable();
        let mut transitions = Vec::new();
        loop {
            let transition = gc.collect_step();
            transitions.push((transition.old_state, transition.new_state));
            if transition.is_done() {
                break;
            }
        }
        assert_eq!(
            transitions,
            vec![
                (Step::STATE_SCANNING, Step::STATE_MARKING),
                (Step::STATE_MARKING, Step::STATE_SWEEPING),
                (Step::STATE_SWEEPING, Step::STATE_FINALIZING),
                (Step::STATE_FINALIZING, Step::STATE_SCANNING),
            ]
        );
        assert!(!gc.isenabled());
        assert_eq!(gc.minor_collections, 4);
    }

    #[test]
    fn published_nursery_top_tracks_real_top_across_minor_collection() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let published_addr = gc.nursery_top_addr();
        let real_top = gc.nursery_top() as usize;
        assert_eq!(
            unsafe { &*(published_addr as *const AtomicUsize) }.load(Ordering::Acquire),
            real_top
        );

        // Prove the collection reset republishes the real limit instead of
        // merely leaving the construction-time value untouched.
        unsafe { &*(published_addr as *const AtomicUsize) }.store(0, Ordering::Release);
        gc.collect_nursery();
        assert_eq!(gc.nursery_top_addr(), published_addr);
        assert_eq!(
            unsafe { &*(published_addr as *const AtomicUsize) }.load(Ordering::Acquire),
            real_top
        );

        let obj = gc.alloc_nursery(16);
        assert!(!obj.is_null());
        assert!(gc.is_in_nursery(obj.0));
    }

    #[test]
    fn test_multiple_allocs() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let mut refs = Vec::new();
        for _ in 0..10 {
            refs.push(gc.alloc_with_type(0, 16));
        }

        // All should be non-null and distinct.
        for i in 0..refs.len() {
            assert!(!refs[i].is_null());
            for j in (i + 1)..refs.len() {
                assert_ne!(refs[i], refs[j]);
            }
        }
    }

    #[test]
    fn test_large_object_skips_the_nursery() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(1024));

        // 1024 > large_object_threshold (512), so it is rawmalloced rather
        // than bumped. Which generation it is born into is the separate
        // question `large_object_dies_at_a_minor` asks: from a collecting
        // entry point the answer is YOUNG and non-moving, so this asserts only
        // what both births share.
        let obj = gc.alloc_with_type(0, 1024);
        assert!(!obj.is_null());
        assert!(!gc.is_in_nursery(obj.0));
    }

    #[test]
    fn test_nursery_collection_basic() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(16));

        // Allocate an object and root it.
        let obj = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(obj.0));

        // Write something to the object payload.
        unsafe {
            *(obj.0 as *mut u64) = 0xDEADBEEF;
        }

        // Root it.
        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Trigger collection.
        gc.collect_nursery();

        // The root should now point to old gen.
        assert!(!gc.is_in_nursery(root.0));
        assert!(!root.is_null());

        // The data should be preserved.
        let val = unsafe { *(root.0 as *const u64) };
        assert_eq!(val, 0xDEADBEEF);

        gc.roots.clear();
    }

    #[test]
    fn test_unrooted_object_dies() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(16));

        // Allocate but don't root.
        let _obj = gc.alloc_with_type(0, 16);

        // Collection should run without issues.
        gc.collect_nursery();

        // Nursery is reset, the object is gone.
        assert_eq!(gc.nursery.used(), 0);
    }

    #[test]
    fn test_fill_nursery_triggers_collection() {
        let mut gc = test_gc(256);
        gc.register_type(TypeInfo::simple(16));

        // Keep allocating until we must have triggered at least one collection.
        for _ in 0..100 {
            gc.alloc_with_type(0, 16);
        }
        assert!(gc.minor_collections > 0);
    }

    #[test]
    fn rooted_collecting_alloc_keeps_fast_bump_in_nursery() {
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(16));
        let mut child = gc.alloc_with_type(tid, 16);
        let roots_before = gc.roots.len();
        let mut needs_write_barrier = true;

        let parent = unsafe {
            gc.alloc_with_type_rooted(tid, 16, &mut child as *mut GcRef, &mut needs_write_barrier)
        };

        assert!(gc.is_in_nursery(child.0));
        assert!(gc.is_in_nursery(parent.0));
        assert!(!needs_write_barrier);
        assert_eq!(gc.minor_collections, 0);
        assert_eq!(gc.roots.len(), roots_before);
    }

    #[test]
    fn rooted_collecting_alloc_forwards_child_only_on_slow_path() {
        let mut gc = test_gc(256);
        let tid = gc.register_type(TypeInfo::simple(16));
        let mut child = gc.alloc_with_type(tid, 16);

        // A 16-byte payload plus the GC header occupies 24 aligned bytes.
        // Leave less than one object free without collecting.
        while gc.nursery.remaining() >= GcHeader::SIZE + 16 {
            let filler = gc.alloc_with_type_no_collect(tid, 16);
            assert!(gc.is_in_nursery(filler.0));
        }
        let roots_before = gc.roots.len();
        let mut needs_write_barrier = true;

        let parent = unsafe {
            gc.alloc_with_type_rooted(tid, 16, &mut child as *mut GcRef, &mut needs_write_barrier)
        };

        assert_eq!(gc.minor_collections, 1);
        assert!(!gc.is_in_nursery(child.0));
        assert!(gc.is_in_nursery(parent.0));
        assert!(!needs_write_barrier);
        assert_eq!(gc.roots.len(), roots_before);
    }

    #[test]
    fn rooted_collecting_alloc_forwards_every_livevar_in_span() {
        let mut gc = test_gc(256);
        let tid = gc.register_type(TypeInfo::simple(16));
        let mut roots = [gc.alloc_with_type(tid, 16), gc.alloc_with_type(tid, 16)];
        let old_roots = roots;

        while gc.nursery.remaining() >= GcHeader::SIZE + 16 {
            let filler = gc.alloc_with_type_no_collect(tid, 16);
            assert!(gc.is_in_nursery(filler.0));
        }
        let roots_before = gc.roots.len();
        let mut needs_write_barrier = true;

        let parent = unsafe {
            gc.alloc_fast_with_type_roots(
                tid,
                16,
                roots.as_mut_ptr(),
                roots.len(),
                &mut needs_write_barrier,
            )
        };

        assert_eq!(gc.minor_collections, 1);
        assert!(roots.iter().all(|root| !gc.is_in_nursery(root.0)));
        assert_ne!(roots[0], old_roots[0]);
        assert_ne!(roots[1], old_roots[1]);
        assert!(gc.is_in_nursery(parent.0));
        assert!(!needs_write_barrier);
        assert_eq!(gc.roots.len(), roots_before);
    }

    #[test]
    fn rooted_collecting_alloc_reports_oldgen_creation_barrier() {
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(16));
        let mut child = gc.alloc_with_type(tid, 16);
        let mut needs_write_barrier = false;
        let payload_size = gc.config.large_object_threshold;

        let parent = unsafe {
            gc.alloc_with_type_rooted(
                tid,
                payload_size,
                &mut child as *mut GcRef,
                &mut needs_write_barrier,
            )
        };

        assert!(!gc.is_in_nursery(parent.0));
        assert!(needs_write_barrier);
    }

    #[test]
    fn no_collect_alloc_reports_nursery_and_oldgen_placement() {
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(16));
        let mut needs_write_barrier = true;

        let young = unsafe {
            gc.try_alloc_with_type_no_collect_with_placement(tid, 16, &mut needs_write_barrier)
        };
        assert!(gc.is_in_nursery(young.0));
        assert!(!needs_write_barrier);

        while gc.nursery.remaining() >= GcHeader::SIZE + 16 {
            let filler = gc.alloc_with_type_no_collect(tid, 16);
            assert!(gc.is_in_nursery(filler.0));
        }
        let spilled = unsafe {
            gc.try_alloc_with_type_no_collect_with_placement(tid, 16, &mut needs_write_barrier)
        };
        assert!(!gc.is_in_nursery(spilled.0));
        assert!(needs_write_barrier);

        let oversized = unsafe {
            gc.try_alloc_with_type_no_collect_with_placement(
                tid,
                gc.config.large_object_threshold,
                &mut needs_write_barrier,
            )
        };
        assert!(!gc.is_in_nursery(oversized.0));
        assert!(needs_write_barrier);
    }

    /// incminimark.py `malloc_fixedsize` and gc.py
    /// `can_use_nursery_malloc`: `large_object` is `nonlarge_max + 1`, an
    /// exclusive boundary rather than the largest admitted nursery size.
    #[test]
    fn large_object_threshold_itself_takes_the_large_arm() {
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(16));
        let boundary = gc.config.large_object_threshold;

        let below = gc.alloc_with_type(tid, boundary - GcHeader::SIZE - GcHeader::ALIGN);
        assert!(gc.is_in_nursery(below.0));

        let at_boundary = gc.alloc_with_type(tid, boundary - GcHeader::SIZE);
        assert!(!gc.is_in_nursery(at_boundary.0));
        assert!(gc.is_young_rawmalloced(at_boundary.0));
    }

    // Lightweight destructor tests (incminimark.py
    // deal_with_young_objects_with_destructors /
    // deal_with_old_objects_with_destructors)
    //
    // The counting destructor below deliberately does NOT `drop_in_place`
    // the dummy payload (the test allocations are raw bytes, not a real
    // `Drop` type); it only records that the collector dispatched it on
    // the right object at the right time. A serializing lock keeps the
    // shared counter races-free under cargo's parallel test runner.

    static DESTRUCTOR_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
    static DESTRUCTOR_RUNS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    static DESTRUCTOR_LAST_ADDR: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    unsafe fn counting_destructor(obj_addr: usize) {
        DESTRUCTOR_RUNS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        DESTRUCTOR_LAST_ADDR.store(obj_addr, std::sync::atomic::Ordering::SeqCst);
    }

    fn destructor_runs() -> usize {
        DESTRUCTOR_RUNS.load(std::sync::atomic::Ordering::SeqCst)
    }

    #[test]
    fn destructor_runs_on_nursery_death() {
        let _guard = DESTRUCTOR_TEST_LOCK.lock();
        DESTRUCTOR_RUNS.store(0, std::sync::atomic::Ordering::SeqCst);

        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_destructor(16, counting_destructor));

        let obj = gc.alloc_with_type(tid, 16);
        // Recorded on the young-destructor list at allocation.
        assert_eq!(gc.young_objects_with_destructors, vec![obj.0]);

        // Not rooted → dies on the next minor collection.
        gc.collect_nursery();

        assert_eq!(destructor_runs(), 1);
        assert_eq!(
            DESTRUCTOR_LAST_ADDR.load(std::sync::atomic::Ordering::SeqCst),
            obj.0
        );
        // Both lists drained: dead object never reaches the old list.
        assert!(gc.young_objects_with_destructors.is_empty());
        assert!(gc.old_objects_with_destructors.is_empty());
    }

    #[test]
    fn destructor_not_run_on_survival_then_run_on_oldgen_death() {
        let _guard = DESTRUCTOR_TEST_LOCK.lock();
        DESTRUCTOR_RUNS.store(0, std::sync::atomic::Ordering::SeqCst);

        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_destructor(16, counting_destructor));

        let obj = gc.alloc_with_type(tid, 16);
        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Survives the minor → promoted, NOT destructed; moves young→old list.
        gc.collect_nursery();
        assert_eq!(destructor_runs(), 0);
        assert!(!gc.is_in_nursery(root.0));
        assert!(gc.young_objects_with_destructors.is_empty());
        assert_eq!(gc.old_objects_with_destructors, vec![root.0]);

        // Unroot → dies at the next major collection.
        gc.roots.clear();
        gc.collect_full();
        assert_eq!(destructor_runs(), 1);
        assert!(gc.old_objects_with_destructors.is_empty());
    }

    #[test]
    fn destructor_survives_multiple_minors_no_double_count() {
        let _guard = DESTRUCTOR_TEST_LOCK.lock();
        DESTRUCTOR_RUNS.store(0, std::sync::atomic::Ordering::SeqCst);

        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_destructor(16, counting_destructor));

        let obj = gc.alloc_with_type(tid, 16);
        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Multiple minors while rooted: promoted once, never re-destructed,
        // and the old list does not accumulate duplicates.
        gc.collect_nursery();
        gc.collect_nursery();
        gc.collect_nursery();
        assert_eq!(destructor_runs(), 0);
        assert_eq!(gc.old_objects_with_destructors, vec![root.0]);

        gc.roots.clear();
        gc.collect_full();
        assert_eq!(destructor_runs(), 1);
    }

    #[test]
    fn destructor_on_a_large_alloc_runs_on_major_death_after_promotion() {
        let _guard = DESTRUCTOR_TEST_LOCK.lock();
        DESTRUCTOR_RUNS.store(0, std::sync::atomic::Ordering::SeqCst);

        // large_object_threshold = nursery/2 = 512; a 1024-byte payload
        // bypasses the nursery. `malloc_fixedsize`'s large arm births it
        // *young*, so the entry starts on the young list and only reaches the
        // old one by surviving a minor.
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_destructor(1024, counting_destructor));

        let mut root = gc.alloc_with_type(tid, 1024);
        let obj_addr = root.0;
        assert!(!gc.is_in_nursery(obj_addr));
        assert_eq!(gc.young_objects_with_destructors, vec![obj_addr]);
        assert!(gc.old_objects_with_destructors.is_empty());

        unsafe { gc.roots.add(&mut root) };
        gc.collect_nursery();
        assert_eq!(destructor_runs(), 0);
        assert!(gc.young_objects_with_destructors.is_empty());
        assert_eq!(gc.old_objects_with_destructors, vec![obj_addr]);

        // Unrooted → reclaimed by the major collection.
        gc.roots.clear();
        gc.collect_full();
        assert_eq!(destructor_runs(), 1);
        assert!(gc.old_objects_with_destructors.is_empty());
    }

    #[test]
    fn no_destructor_means_no_list_entry() {
        let _guard = DESTRUCTOR_TEST_LOCK.lock();
        let mut gc = test_gc(4096);
        // A plain type (no destructor) is never recorded on either list.
        let tid = gc.register_type(TypeInfo::simple(16));
        gc.alloc_with_type(tid, 16);
        assert!(gc.young_objects_with_destructors.is_empty());
        assert!(gc.old_objects_with_destructors.is_empty());
    }

    #[test]
    fn test_write_barrier() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(16));

        // Allocate a large object (goes to old gen).
        let old_obj = gc.alloc_in_oldgen_clear(0, GcHeader::SIZE + 16);
        assert!(!gc.is_in_nursery(old_obj.0));

        // The old object should have TRACK_YOUNG_PTRS.
        let hdr = unsafe { header_of(old_obj.0) };
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });

        // Write barrier clears the flag and adds to remembered set.
        gc.do_write_barrier(old_obj);
        assert!(unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
        assert_eq!(gc.old_objects_pointing_to_young.len(), 1);

        // Second call: flag already cleared, should not add again.
        gc.do_write_barrier(old_obj);
        assert_eq!(gc.old_objects_pointing_to_young.len(), 1);
    }

    #[test]
    fn prebuilt_write_barrier_registers_root_once_and_traces_children() {
        let mut gc = test_gc(1024);
        let child_tid = gc.register_type(TypeInfo::simple(16));
        let prebuilt_tid = gc.register_type(TypeInfo::with_gc_ptrs(
            2 * std::mem::size_of::<usize>(),
            vec![std::mem::size_of::<usize>()],
        ));
        let vtable = 0x1234_5678usize;
        crate::GcAllocator::register_vtable_for_type(&mut gc, vtable, prebuilt_tid);

        let child = gc.alloc_with_type(child_tid, 16);
        let prebuilt = crate::header::alloc_with_gc_header([vtable, child.0], prebuilt_tid);
        let prebuilt_ref = GcRef(prebuilt as usize);
        let hdr = unsafe { header_of(prebuilt_ref.0) };
        // `alloc_with_gc_header` is pyre's runtime host allocator, not a
        // translated prebuilt (see its docstring), so it leaves the flags
        // clear. Stamp incminimark's `init_gc_object_immortal` shape by hand
        // to exercise the lifecycle this test is about.
        unsafe {
            (*hdr).set_flag(GcFlags::GCFLAG_NO_HEAP_PTRS);
            (*hdr).set_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
        }
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS) });

        gc.do_write_barrier(prebuilt_ref);
        assert!(unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_NO_HEAP_PTRS) });
        assert_eq!(gc.old_objects_pointing_to_young, vec![prebuilt_ref.0]);
        assert_eq!(gc.prebuilt_root_objects, vec![prebuilt_ref.0]);

        gc.do_collect_nursery();
        let promoted_child = unsafe { GcRef((*prebuilt)[1]) };
        assert_ne!(promoted_child, child);
        assert!(gc.oldgen.contains(promoted_child.0));

        gc.do_write_barrier(prebuilt_ref);
        assert_eq!(gc.prebuilt_root_objects, vec![prebuilt_ref.0]);
        gc.collect_full();
        assert!(gc.oldgen.contains(promoted_child.0));
        assert!(unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_VISITED) });
    }

    // incminimark gates the write barrier solely on GCFLAG_TRACK_YOUNG_PTRS:
    // every old-gen producer sets it, and the barrier appends on the flag test
    // alone. These lock that invariant down across all four producers, plus a
    // zeroed-header case (no flag) that the barrier must skip.

    #[test]
    fn track_young_ptrs_set_on_direct_oldgen_alloc() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_in_oldgen_clear(0, GcHeader::SIZE + 16);
        assert!(unsafe { (*header_of(obj.0)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
    }

    #[test]
    fn track_young_ptrs_set_on_card_alloc() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::varsize(8, ptr_size, 0, true, Vec::new()));
        let length = CARD_ARRAY_LEN;
        let total_size = GcHeader::SIZE + 8 + ptr_size * length;
        let obj = gc.alloc_in_oldgen_with_cards(tid, total_size, length, true);
        assert!(unsafe { (*header_of(obj.0)).has_flag(GcFlags::GCFLAG_HAS_CARDS) });
        assert!(unsafe { (*header_of(obj.0)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
    }

    /// incminimark.py `shrink_array` on the object it accepts: the
    /// nursery array keeps its address and reports the smaller length, which
    /// is what a caller returning `p` unchanged relies on.
    #[test]
    fn shrink_array_records_a_smaller_length_in_place() {
        let mut gc = test_gc(4096);
        // A STR-shaped type: a fixed `hash` word, then the length word the
        // varsize part is measured by.
        let tid = gc.register_type(TypeInfo::varsize(16, 8, 8, false, Vec::new()));
        let obj = gc.alloc_with_type(tid, 16 + 8 * 12);
        assert!(gc.is_in_nursery(obj.0));
        unsafe { *((obj.0 + 8) as *mut usize) = 12 };
        unsafe { *(obj.0 as *mut usize) = 0xdead_beef };

        assert!(gc.shrink_array(obj.0, 5));

        assert_eq!(unsafe { *((obj.0 + 8) as *const usize) }, 5);
        // The fixed part is untouched: upstream copies it only on the fallback
        // path, because the accepted path never moves the object.
        assert_eq!(unsafe { *(obj.0 as *const usize) }, 0xdead_beef);
    }

    /// incminimark.py:1169-1170: "Only objects in the nursery can be
    /// resized."  An old-generation array is declined, and that answer is what
    /// sends `rgc.ll_shrink_array` down its allocate-and-copy path.
    #[test]
    fn shrink_array_declines_an_old_generation_array() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(16, 8, 8, false, Vec::new()));
        let obj = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + 16 + 8 * 12);
        assert!(!gc.is_in_nursery(obj.0));
        unsafe { *((obj.0 + 8) as *mut usize) = 12 };

        assert!(!gc.shrink_array(obj.0, 5));

        assert_eq!(unsafe { *((obj.0 + 8) as *const usize) }, 12);
    }

    /// incminimark.py: a nursery object with GCFLAG_HAS_SHADOW is
    /// not resized either, "as this would potentially loose part of the memory
    /// in the already-allocated shadow".
    #[test]
    fn shrink_array_declines_an_object_with_a_shadow() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(16, 8, 8, false, Vec::new()));
        let obj = gc.alloc_with_type(tid, 16 + 8 * 12);
        unsafe { *((obj.0 + 8) as *mut usize) = 12 };
        unsafe { (*header_of(obj.0)).set_flag(GcFlags::GCFLAG_HAS_SHADOW) };

        assert!(!gc.shrink_array(obj.0, 5));

        assert_eq!(unsafe { *((obj.0 + 8) as *const usize) }, 12);
    }

    /// A fixed-size type has no length field, so there is nothing to record.
    /// Upstream reaches the write through `varsize_offset_to_length`, which
    /// such a type has no answer for.
    #[test]
    fn shrink_array_declines_a_fixed_size_object() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_with_type(tid, 16);
        assert!(gc.is_in_nursery(obj.0));

        assert!(!gc.shrink_array(obj.0, 1));
    }

    /// A shrunk nursery array consumes less when it leaves the nursery: the
    /// promotion copy is sized from the length word the shrink wrote.
    #[test]
    fn a_shrunk_array_is_promoted_at_its_smaller_size() {
        let _guard = SHADOW_STACK_TEST_LOCK.lock();
        crate::shadow_stack::clear();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(16, 8, 8, false, Vec::new()));
        let obj = gc.alloc_with_type(tid, 16 + 8 * 100);
        unsafe { *(obj.0 as *mut usize) = 0xdead_beef };
        unsafe { *((obj.0 + 8) as *mut usize) = 100 };
        assert!(gc.shrink_array(obj.0, 3));
        let root = crate::shadow_stack::OwnerRootGuard::new(obj);
        gc.bytes_made_old_since_cycle = 0;

        gc.do_collect_nursery();

        let moved = root.get();
        assert!(gc.oldgen.contains(moved.0));
        assert_eq!(unsafe { *(moved.0 as *const usize) }, 0xdead_beef);
        assert_eq!(unsafe { *((moved.0 + 8) as *const usize) }, 3);
        // The promotion sizes the copy from the length word, so the shrink is
        // what the old generation is charged for.
        assert_eq!(
            gc.bytes_made_old_since_cycle,
            GcHeader::SIZE + 16 + 8 * 3,
            "the promotion copied the pre-shrink size"
        );
    }

    /// incminimark.py:1032-1035, the `alloc_young` arm of the card question.
    ///
    /// A young card array is born with GCFLAG_CARDS_SET already on, and stays
    /// off `old_objects_with_cards_set` while it is young. That flag is not
    /// decoration: `visit_young_rawmalloced_object` asserts it before it puts
    /// the promoted array on that list.
    #[test]
    fn a_young_card_array_is_born_with_cards_set_and_off_the_dirty_list() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(8, 8, 0, true, Vec::new()));
        let total_size = GcHeader::SIZE + 8 + 8 * CARD_ARRAY_LEN;

        let obj = gc
            .try_alloc_young_nonmoving_with_cards(tid, total_size, CARD_ARRAY_LEN, true)
            .expect("young rawmalloc is enabled and the type is not a weakref");

        let hdr = unsafe { header_of(obj.0) };
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) });
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_CARDS_SET) });
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
        assert!(gc.is_young_rawmalloced(obj.0));
        assert!(gc.old_objects_with_cards_set.is_empty());
        // The allocator cleared the card bytes and `raw_memclear` the payload.
        assert_eq!(unsafe { *MiniMarkGC::get_card_ptr(obj.0, 0) }, 0);
        assert_eq!(unsafe { *(obj.0 as *const usize) }, 0);
    }

    /// The young arm asks the same size question the old one does, so a small
    /// array born young carries no card header — and then no CARDS_SET, which
    /// is what keeps `visit_young_rawmalloced_object`'s assertion honest.
    #[test]
    fn a_small_young_array_is_denied_cards_and_carries_no_flags() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(8, 8, 0, true, Vec::new()));
        let total_size = GcHeader::SIZE + 8 + 8 * 4;
        assert!(total_size < gc.config.large_object_threshold);

        let obj = gc
            .try_alloc_young_nonmoving_with_cards(tid, total_size, 4, true)
            .expect("young rawmalloc is enabled and the type is not a weakref");

        let hdr = unsafe { header_of(obj.0) };
        assert!(!unsafe { (*hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) });
        assert!(!unsafe { (*hdr).has_flag(GcFlags::GCFLAG_CARDS_SET) });
        assert!(!unsafe { (*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
        assert!(gc.is_young_rawmalloced(obj.0));
    }

    /// `external_malloc`'s two arms share one tail — "Common code to fill the
    /// header and length of the object" — so the old-gen card birth owes every
    /// registration and the zero-fill that `alloc_in_oldgen_clear` owes.
    ///
    /// The arm became reachable when `alloc_varsize_typed` started routing its
    /// oversized requests through `alloc_varsize_nonmoving_prefer_young`, whose
    /// fallback this is: it is taken whenever the young rawmalloc is refused
    /// (`MAJIT_GC_YOUNG_RAWMALLOC=0`, a weakref type) or fails.
    #[test]
    fn the_oldgen_card_birth_clears_and_takes_the_shared_oldgen_tail() {
        let mut gc = test_gc(4096);
        let item_size = std::mem::size_of::<GcRef>();
        let base_size = 8;
        let length = CARD_ARRAY_LEN;
        let total_size = GcHeader::SIZE + base_size + item_size * length;
        assert!(total_size >= gc.config.large_object_threshold);
        let tid = gc.register_type(
            TypeInfo::varsize(base_size, item_size, 0, true, Vec::new())
                .with_destructor_fn(counting_destructor),
        );
        let bytes_before = gc.bytes_made_old_since_cycle;
        // `arena_malloc(allocsize, 0)` — "the memory returned is not cleared".
        // Hand this allocation a really dirty block, so the zero assertion
        // below can fail; a fresh `alloc::alloc` reads zero on its own and
        // would pass whether or not the clear ran.
        gc.oldgen.set_poison_on_alloc(true);

        let obj = gc.alloc_in_oldgen_with_cards(tid, total_size, length, true);

        let hdr = unsafe { header_of(obj.0) };
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) });
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
        // `do_malloc_varsize_clear`: the block the caller gets is zeroed, so
        // the tracer reads item slots and not whatever the allocator held.
        let payload =
            unsafe { std::slice::from_raw_parts(obj.0 as *const u8, total_size - GcHeader::SIZE) };
        assert!(
            payload.iter().all(|&b| b == 0),
            "the card block was handed back unzeroed"
        );
        // Reached only through the shared tail: a destructor type born on this
        // arm has no other route onto the list a major collection drains.
        assert!(gc.old_objects_with_destructors.contains(&obj.0));
        // `allocsize = cardheadersize + round_up_for_allocation(totalsize)`.
        let card_header_bytes =
            std::mem::size_of::<usize>() * gc.card_marking_words_for_length(length);
        assert!(card_header_bytes > 0);
        assert_eq!(
            gc.bytes_made_old_since_cycle, bytes_before,
            "an old-gen birth is not a promotion"
        );
    }

    /// `external_malloc`'s card predicate has no `length > 0` clause.  A
    /// large fixed prefix with an empty variable part therefore carries the
    /// flags with a zero-byte card header; no valid item access can ask for a
    /// card byte.
    #[test]
    fn a_large_zero_length_array_keeps_upstreams_card_flags() {
        let mut gc = test_gc(4096);
        let base_size = gc.config.large_object_threshold;
        let tid = gc.register_type(TypeInfo::varsize(base_size, 8, 0, true, Vec::new()));
        let obj = gc.alloc_in_oldgen_with_cards(tid, base_size, 0, true);

        let hdr = unsafe { header_of(obj.0) };
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) });
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
        assert_eq!(gc.card_marking_words_for_length(0), 0);
    }

    /// incminimark.py:1019, the size clause of the card question: an object
    /// small enough for the arena arm is denied cards even when its type has
    /// pointers in the varsize part. `spill_to_oldgen_or_null` reaches this
    /// entry point with exactly such an object whenever the nursery is full,
    /// and a card array over it would be read by `card_byte_ptr` as bytes the
    /// allocation never reserved.
    #[test]
    fn a_small_array_is_denied_cards_and_still_tracks_young_ptrs() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(8, ptr_size, 0, true, Vec::new()));
        let length = 4usize;
        let total_size = GcHeader::SIZE + 8 + ptr_size * length;
        assert!(total_size < gc.config.large_object_threshold);
        assert!(gc.config.card_page_indices > 0);

        let obj = gc.alloc_in_oldgen_with_cards(tid, total_size, length, true);

        assert!(!unsafe { (*header_of(obj.0)).has_flag(GcFlags::GCFLAG_HAS_CARDS) });
        assert!(unsafe { (*header_of(obj.0)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
    }

    #[test]
    fn track_young_ptrs_set_on_nursery_promotion() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));
        let mut root = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(root.0));
        unsafe { gc.roots.add(&mut root) };
        gc.collect_nursery();
        // `root` now holds the promoted old-gen address.
        assert!(!gc.is_in_nursery(root.0));
        assert!(unsafe { (*header_of(root.0)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
        gc.roots.clear();
    }

    #[test]
    fn track_young_ptrs_set_on_shadow_alloc() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_with_type(0, 16);
        let shadow = gc.allocate_shadow(obj.0);
        assert!(unsafe { (*header_of(shadow)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
    }

    #[test]
    fn born_old_typed_payload_is_zero_filled() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(24));
        let obj = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + 24);
        let payload = unsafe { std::slice::from_raw_parts(obj.0 as *const u8, 24) };
        assert!(payload.iter().all(|&byte| byte == 0));
    }

    #[test]
    fn nonmoving_major_keeps_unoccupied_shadow_until_minor_copy() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_with_type(tid, 16);
        let shadow = gc.allocate_shadow(obj.0);
        let mut root = obj;
        unsafe { gc.roots.add(&mut root) };

        // The shadow payload has not been copied yet.  A non-moving major must
        // retain the reserved block without tracing its uninitialized fields.
        gc.do_collect_oldgen_nonmoving();
        assert_eq!(gc.oldgen.object_count(), 1);
        assert!(gc.oldgen.contains(shadow));
        assert!(!unsafe { (*header_of(shadow)).has_flag(GcFlags::GCFLAG_VISITED) });

        // The following minor occupies exactly that reserved identity home.
        gc.do_collect_nursery();
        assert_eq!(root.0, shadow);
        gc.roots.clear();
    }

    #[test]
    fn write_barrier_skips_object_without_track_young_ptrs() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(16));
        // A fabricated header with TRACK_YOUNG_PTRS clear is skipped. Nursery
        // objects have the same flag state but return through the range check.
        let mut buf = Box::new([0u64; 2]);
        let base = buf.as_mut_ptr() as usize;
        unsafe { *(base as *mut GcHeader) = GcHeader::new(0) };
        let payload = base + GcHeader::SIZE;
        gc.do_write_barrier(GcRef(payload));
        assert_eq!(gc.old_objects_pointing_to_young.len(), 0);
        drop(buf);
    }

    #[test]
    fn write_barrier_null_is_noop() {
        // GcRef::NULL reaches the safe write_barrier entry; it must not read a
        // header at `0 - GcHeader::SIZE`.
        let mut gc = test_gc(1024);
        gc.do_write_barrier(GcRef(0));
        assert_eq!(gc.old_objects_pointing_to_young.len(), 0);
    }

    #[test]
    fn write_barrier_ignores_unmanaged_jitframe_with_flag_byte_set() {
        // `_reload_frame_if_necessary` (aarch64/assembler.py,
        // x86/assembler.py) re-applies the NON-array write-barrier fast
        // path to the current jitframe after a collecting helper call, so a
        // plain COND_CALL_GC_WB on the frame reaches the generic barrier at
        // runtime.
        //
        // A jitframe allocated off the GC is not in any managed generation,
        // so the word in front of it is not a header this collector owns.
        // `jitframe::alloc_off_gc_jitframe` keeps that word reserved and zeroed,
        // which is what stops the inline test from entering the helper at all;
        // this test covers the residue — the helper being entered anyway, with
        // the flag bit set, for a block the GC does not manage. Without the
        // `is_managed_heap_object` guard in `do_write_barrier`, that block
        // would enter `old_objects_pointing_to_young` and the next minor would decode a type
        // id from those bytes.
        let mut gc = test_gc(1024);

        // Same shape the off-GC jitframe allocator produces: one reserved
        // leading word, then the fixed `JitFrame` words plus the trailing slot
        // array, zero-filled.
        let slots = 32usize;
        let frame_size = std::mem::size_of::<usize>() * (7 + 1 + slots);
        let layout =
            std::alloc::Layout::from_size_align(GcHeader::SIZE + frame_size, GcHeader::SIZE)
                .unwrap();
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!base.is_null());
        let frame = unsafe { base.add(GcHeader::SIZE) };
        let obj = frame as usize;

        // The frame is a fresh host allocation, so it cannot overlap either
        // managed generation — that is precisely what the guard detects.
        assert!(!gc.nursery.contains(obj));
        assert!(!gc.oldgen.contains(obj));

        // Set the bit the inline COND_CALL_GC_WB test reads, so the helper is
        // entered exactly as it is when the bytes before a real malloc'd
        // frame happen to carry TRACK_YOUNG_PTRS.
        let descr = crate::WriteBarrierDescr::for_current_gc();
        let flag_byte = unsafe { frame.offset(descr.jit_wb_if_flag_byteofs as isize) };
        unsafe { *flag_byte |= descr.jit_wb_if_flag_singlebyte };
        let flag_byte_before = unsafe { *flag_byte };

        gc.do_write_barrier(GcRef(obj));

        assert_eq!(gc.old_objects_pointing_to_young.len(), 0);
        // `remember_young_pointer` clears TRACK_YOUNG_PTRS, so an unchanged
        // byte also proves the barrier never wrote through the fake header.
        assert_eq!(unsafe { *flag_byte }, flag_byte_before);

        unsafe { std::alloc::dealloc(base, layout) };
    }

    #[test]
    fn write_barrier_ignores_headerless_external_object_with_registered_vtable() {
        // `w_tuple_new` / `w_specialisedtuple_new_*` fall back to a bare
        // `Box::into_raw` when `try_gc_alloc_stable` finds no installed
        // collector, and the struct they build still starts with `ob_header` —
        // the very vtable `register_vtable_for_type` knows. Such an object is
        // in no managed generation, so `do_write_barrier` reaches its
        // bootstrap-prebuilt arm; a vtable-only test there would read the
        // allocator word in FRONT of the box as a `GcHeader` and, on a set
        // `TRACK_YOUNG_PTRS` bit, have `remember_young_pointer` write flags
        // back into that metadata. `registered_external_header`'s tid equality
        // is what rejects it.
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(2 * std::mem::size_of::<usize>()));
        let vtable = 0x1234_5678usize;
        crate::GcAllocator::register_vtable_for_type(&mut gc, vtable, tid);

        // One leading word standing in for the allocator metadata a real
        // `Box` keeps there, then the headerless object itself.
        let layout = std::alloc::Layout::from_size_align(
            GcHeader::SIZE + 2 * std::mem::size_of::<usize>(),
            GcHeader::SIZE,
        )
        .unwrap();
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!base.is_null());
        let obj = unsafe { base.add(GcHeader::SIZE) } as usize;
        unsafe { (obj as *mut usize).write(vtable) };
        assert!(!gc.is_managed_heap_object(obj));

        // Make the fake header in front of the object look live: both the
        // barrier's flag bit and a type id that is registered — but NOT this
        // object's, which is the whole discriminator.
        let other_tid = gc.register_type(TypeInfo::simple(8));
        assert_ne!(other_tid, tid);
        let fake = base as *mut GcHeader;
        unsafe {
            fake.write(GcHeader::new(other_tid));
            (*fake).set_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
        }
        let fake_bits_before = unsafe { (*fake).tid_and_flags };

        gc.do_write_barrier(GcRef(obj));

        assert_eq!(gc.old_objects_pointing_to_young.len(), 0);
        // `remember_young_pointer` clears TRACK_YOUNG_PTRS, so unchanged bits
        // also prove the barrier never wrote through the fake header.
        assert_eq!(unsafe { (*fake).tid_and_flags }, fake_bits_before);

        unsafe { std::alloc::dealloc(base, layout) };
    }

    #[test]
    fn write_barrier_skips_forwarded_nursery_address() {
        // incminimark.py:1510-1512: nursery objects never carry
        // GCFLAG_TRACK_YOUNG_PTRS.  After a moving minor the old nursery
        // header is a forwarding word, not a live header whose flag bits may
        // be tested.  A host raw local can briefly retain that old address.
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(16));
        let nursery_obj = gc.alloc_with_type(tid, 16);
        let stale_addr = nursery_obj;
        let mut root = nursery_obj;
        unsafe { gc.roots.add(&mut root) };

        gc.do_collect_nursery();
        assert_ne!(root, stale_addr);
        assert!(gc.is_in_nursery(stale_addr.0));
        assert!(!gc.is_in_nursery(root.0));

        gc.do_write_barrier(stale_addr);
        assert!(gc.old_objects_pointing_to_young.is_empty());
        gc.roots.clear();
    }

    #[test]
    fn test_nursery_collection_with_pointers() {
        // Object layout: one GcRef field at offset 0 (payload = 8 bytes).
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        // Create two objects: parent -> child.
        let child = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        let parent = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());

        // Write child's address into parent's first field.
        unsafe {
            *(parent.0 as *mut GcRef) = child;
        }

        // Root only the parent.
        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Trigger collection.
        gc.collect_nursery();

        // Parent should have survived.
        assert!(!gc.is_in_nursery(root.0));
        assert!(!root.is_null());

        // The pointer field should now point to the child's new location.
        let child_ref = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(child_ref.0));
        assert!(!child_ref.is_null());

        gc.roots.clear();
    }

    #[test]
    fn test_forwarding_dedup() {
        // Two roots pointing to the same nursery object should get the
        // same forwarded address.
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        let shared = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        let mut root1 = shared;
        let mut root2 = shared;

        unsafe {
            gc.roots.add(&mut root1);
            gc.roots.add(&mut root2);
        }

        gc.collect_nursery();

        // Both roots should point to the same old-gen location.
        assert_eq!(root1, root2);
        assert!(!gc.is_in_nursery(root1.0));

        gc.roots.clear();
    }

    #[test]
    fn test_major_collection() {
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Allocate some objects and root one.
        let obj1 = gc.alloc_with_type(tid, 16);
        let _obj2 = gc.alloc_with_type(tid, 16); // unreachable

        let mut root = obj1;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Full collection: promotes to old gen and sweeps.
        gc.collect_full();

        assert!(!root.is_null());
        assert!(!gc.is_in_nursery(root.0));

        gc.roots.clear();
    }

    #[test]
    fn test_major_collection_frees_unreachable_old_objects() {
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Promote two objects to old gen.
        let obj1 = gc.alloc_with_type(tid, 16);
        let obj2 = gc.alloc_with_type(tid, 16);
        let mut root1 = obj1;
        let mut root2 = obj2;
        unsafe {
            gc.roots.add(&mut root1);
            gc.roots.add(&mut root2);
        }
        gc.collect_nursery();
        assert!(!gc.is_in_nursery(root1.0));
        assert!(!gc.is_in_nursery(root2.0));
        assert_eq!(gc.oldgen.object_count(), 2);

        // Now unroot obj2 and do a full collection.
        gc.roots.remove(&mut root2);
        gc.collect_full();

        // Only obj1 should survive.
        assert_eq!(gc.oldgen.object_count(), 1);
        assert!(!root1.is_null());

        gc.roots.clear();
    }

    #[test]
    fn test_repeated_collections() {
        let mut gc = test_gc(512);
        let tid = gc.register_type(TypeInfo::simple(16));

        let mut root = GcRef::NULL;
        unsafe {
            gc.roots.add(&mut root);
        }

        for i in 0..50 {
            let obj = gc.alloc_with_type(tid, 16);
            // Write a marker value.
            unsafe {
                *(obj.0 as *mut u64) = i as u64;
            }
            root = obj;

            if i % 10 == 0 {
                gc.collect_nursery();
                // Root should survive and preserve its value.
                if !root.is_null() {
                    let val = unsafe { *(root.0 as *const u64) };
                    assert_eq!(val, i as u64);
                }
            }
        }

        gc.roots.clear();
    }

    #[test]
    fn test_gc_allocator_trait() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(32));

        let obj = gc.alloc_nursery(32);
        assert!(!obj.is_null());

        let varobj = gc.alloc_varsize(8, 4, 10);
        assert!(!varobj.is_null());

        gc.collect_nursery();
        gc.collect_full();
    }

    #[test]
    fn varsize_allocation_overflow_returns_null() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        // incminimark.py external_malloc turns either multiplication or
        // addition overflow into MemoryError; the JIT allocator reports that
        // condition as NULL to its CHECK_MEMORY_ERROR caller.
        assert!(gc.alloc_varsize(0, 2, usize::MAX).is_null());
        assert!(gc.alloc_varsize_typed(tid, usize::MAX, 1, 1).is_null());
        assert!(gc.alloc_varsize_no_collect(0, usize::MAX, 2).is_null());

        // Include the GC header in the same checked-size contract.
        assert!(gc.alloc_nursery(usize::MAX).is_null());
        assert!(gc.alloc_nursery_no_collect(usize::MAX).is_null());
        assert!(gc.alloc_oldgen_typed(tid, usize::MAX).is_null());
    }

    /// A varsize length is read out of the object, so a collector reaching an
    /// object before its length is initialized computes a size that describes
    /// nothing.  Both shapes must be rejected — in particular the second, where
    /// the multiplication does *not* overflow and the size is merely far past
    /// anything `Layout` can express.
    #[test]
    fn varsize_length_that_describes_no_allocation_is_rejected() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(16, 8, 0, false, Vec::new()));

        // The length field lives at offset 0, so a local stands in for the
        // object: nothing here allocates, and nothing is dereferenced beyond it.
        let length = std::cell::Cell::new(0usize);
        let obj_addr = length.as_ptr() as usize;

        length.set(4);
        assert_eq!(gc.try_size_for_typeid(obj_addr, tid), Some(16 + 8 * 4));

        length.set(usize::MAX);
        assert_eq!(gc.try_size_for_typeid(obj_addr, tid), None);

        length.set(isize::MAX as usize / 8);
        assert_eq!(gc.try_size_for_typeid(obj_addr, tid), None);
    }

    #[test]
    #[should_panic(expected = "varsize length describes no allocation")]
    fn varsize_length_that_describes_no_allocation_names_its_inputs() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(16, 8, 0, false, Vec::new()));
        let length = std::cell::Cell::new(usize::MAX);
        gc.size_for_typeid(length.as_ptr() as usize, tid, "test");
    }

    /// `get_possibly_forwarded_type_id` then `get_size`. The nursery corpse's
    /// first payload word is the forwarding address (`set_forwarding_address`),
    /// and that word is the length when `length_offset == 0`.
    #[test]
    fn forwarded_varsize_length_is_read_from_the_copy() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(16, 8, 0, false, Vec::new()));
        let corpse = gc.alloc_varsize_typed(tid, 16, 8, 1);
        let live = gc.alloc_oldgen_typed(tid, 16 + 8 * 4);
        assert!(gc.is_in_nursery(corpse.0));
        assert!(!gc.is_in_nursery(live.0));
        unsafe {
            *((live.0) as *mut usize) = 4;
            GcHeader::set_forwarding_address(header_of(corpse.0), live.0);
        }
        assert_eq!(gc.try_size_for_typeid(corpse.0, tid), Some(16 + 8 * 4));
    }

    /// llsupport/gc.py GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr
    /// pyre's GC stores an explicit vtable→type_id table; verify that
    /// register_vtable_for_type / get_typeid_from_classptr_if_gcremovetypeptr
    /// round-trip via the GcAllocator trait.
    #[test]
    fn test_gcremovetypeptr_vtable_lookup() {
        let mut gc = test_gc(4096);
        let int_tid = gc.register_type(TypeInfo::simple(16));
        let float_tid = gc.register_type(TypeInfo::simple(16));

        // Register two distinct vtables
        let int_vtable: usize = 0x1234_5670;
        let float_vtable: usize = 0x1234_5680;
        crate::GcAllocator::register_vtable_for_type(&mut gc, int_vtable, int_tid);
        crate::GcAllocator::register_vtable_for_type(&mut gc, float_vtable, float_tid);

        // Round-trip
        assert_eq!(
            crate::GcAllocator::get_typeid_from_classptr_if_gcremovetypeptr(&gc, int_vtable),
            Some(int_tid)
        );
        assert_eq!(
            crate::GcAllocator::get_typeid_from_classptr_if_gcremovetypeptr(&gc, float_vtable),
            Some(float_tid)
        );
        // Unknown classptr → None
        assert_eq!(
            crate::GcAllocator::get_typeid_from_classptr_if_gcremovetypeptr(&gc, 0xCAFEBABE),
            None
        );
    }

    #[test]
    fn test_alloc_nursery_no_collect_does_not_trigger_collection() {
        let mut gc = test_gc(64);
        gc.register_type(TypeInfo::simple(24));

        let obj1 = gc.alloc_nursery_no_collect(24);
        let obj2 = gc.alloc_nursery_no_collect(24);

        assert!(!obj1.is_null());
        assert!(!obj2.is_null());
        assert_eq!(gc.minor_collections, 0);
        // The second allocation may have fallen back to old gen, but it must
        // still succeed without forcing a collection.
        assert_ne!(obj1, obj2);
    }

    #[test]
    fn test_alloc_varsize_no_collect_does_not_trigger_collection() {
        let mut gc = test_gc(64);
        gc.register_type(TypeInfo::simple(32));

        let obj = gc.alloc_varsize_no_collect(16, 8, 8);

        assert!(!obj.is_null());
        assert_eq!(gc.minor_collections, 0);
        // This request is too large for the tiny nursery, so no-collect mode
        // must have used the old generation instead.
        assert!(!gc.is_in_nursery(obj.0));
    }

    #[test]
    fn test_write_barrier_with_collection() {
        // Scenario: old object points to young object, write barrier ensures
        // the young object survives collection.
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        // Create an old-gen object.
        let old_obj = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + std::mem::size_of::<GcRef>());

        // Create a young object.
        let young_obj = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(young_obj.0 as *mut u64) = 0x42424242;
        }

        // Store young ref into old object's field.
        unsafe {
            *(old_obj.0 as *mut GcRef) = young_obj;
        }
        // Write barrier.
        gc.do_write_barrier(old_obj);

        // Root only the old object.
        let mut root = old_obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Collect.
        gc.collect_nursery();

        // The old object's field should be updated to the new location.
        let child_ref = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(child_ref.0));
        assert!(!child_ref.is_null());
        let val = unsafe { *(child_ref.0 as *const u64) };
        assert_eq!(val, 0x42424242);

        gc.roots.clear();
    }

    #[test]
    fn test_chain_of_pointers() {
        // Test a chain: root -> A -> B -> C, all in nursery.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        let c = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(c.0 as *mut GcRef) = GcRef::NULL;
        }

        let b = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(b.0 as *mut GcRef) = c;
        }

        let a = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(a.0 as *mut GcRef) = b;
        }

        let mut root = a;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.collect_nursery();

        // Verify the entire chain survived.
        assert!(!gc.is_in_nursery(root.0));
        let new_b = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(new_b.0));
        assert!(!new_b.is_null());
        let new_c = unsafe { *(new_b.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(new_c.0));
        assert!(!new_c.is_null());
        let tail = unsafe { *(new_c.0 as *const GcRef) };
        assert!(tail.is_null());

        gc.roots.clear();
    }

    #[test]
    fn test_major_collection_with_graph() {
        // Test major collection with a graph: root -> A -> B, root -> C (unreachable D).
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        // Type with one pointer field.
        let tid1 = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        // Type with two pointer fields.
        let tid2 = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size * 2, vec![0, ptr_size]));

        let b = gc.alloc_with_type(tid1, ptr_size);
        unsafe {
            *(b.0 as *mut GcRef) = GcRef::NULL;
        }

        let a = gc.alloc_with_type(tid1, ptr_size);
        unsafe {
            *(a.0 as *mut GcRef) = b;
        }

        let c = gc.alloc_with_type(tid1, ptr_size);
        unsafe {
            *(c.0 as *mut GcRef) = GcRef::NULL;
        }

        let d = gc.alloc_with_type(tid1, ptr_size);
        unsafe {
            *(d.0 as *mut GcRef) = GcRef::NULL;
        }
        let _ = d; // unreachable

        // Root object points to both A and C.
        let root_obj = gc.alloc_with_type(tid2, ptr_size * 2);
        unsafe {
            *(root_obj.0 as *mut GcRef) = a;
            *((root_obj.0 + ptr_size) as *mut GcRef) = c;
        }

        let mut root = root_obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Promote to old gen.
        gc.collect_nursery();

        // Now do full collection. A, B, C should survive; D should be freed.
        // We have 5 objects in old gen (root, A, B, C, D).
        // Wait, D was unreachable from root, so it wouldn't have been promoted.
        // Actually it was in the nursery unreachable, so it was wiped on nursery reset.
        assert_eq!(gc.oldgen.object_count(), 4); // root, A, B, C

        gc.collect_full();

        // All 4 are reachable from the root, so all survive.
        assert_eq!(gc.oldgen.object_count(), 4);

        gc.roots.clear();
    }

    #[test]
    fn major_mark_skips_unmanaged_field_target() {
        // An old-gen object's `gc_ptr_offsets` field may point at a
        // `std::alloc`-backed (non-GC) block while some object fields are
        // still host allocations rather than managed-GC allocations.
        // The major-mark walker must skip such fields rather than
        // dereferencing memory before the unmanaged block as a
        // `GcHeader`. This test catches regression of the
        // `is_managed_heap_object` guard in `mark_object`.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Allocate parent in old-gen. Field pointer will be set to a
        // raw `Box::into_raw` block — what `std::alloc` does for an
        // unmigrated `wrappeditems` / `items` slot today.
        let parent = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        let unmanaged = Box::into_raw(Box::new(0xCAFEBABEu64));
        unsafe {
            *(parent.0 as *mut GcRef) = GcRef(unmanaged as usize);
        }

        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Major collection — without the guard this would crash on
        // `header_of(unmanaged)` reading invalid memory.
        gc.collect_full();

        // Parent survived because it is rooted; the unmanaged target
        // was correctly skipped.
        assert_eq!(gc.oldgen.object_count(), 1);

        gc.roots.clear();
        unsafe {
            drop(Box::from_raw(unmanaged));
        }
    }

    #[test]
    fn major_mark_skips_unmanaged_varsize_item_target() {
        // Same scenario, varsize variant: an `items_have_gc_ptrs = true`
        // array can hold unmanaged pointers while item storage is still
        // host-allocated. The mark
        // walker's variable-part loop must also guard with
        // `is_managed_heap_object`.
        let ptr_size = std::mem::size_of::<GcRef>();
        let length_offset = 0;
        let base_size = 8; // length field (Signed)
        let length = 2usize;

        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(
            base_size,
            ptr_size,
            length_offset,
            true,
            Vec::new(),
        ));

        let total_payload = base_size + ptr_size * length;
        let parent = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + total_payload);
        unsafe {
            *(parent.0 as *mut usize) = length;
        }
        let unmanaged_a = Box::into_raw(Box::new(0xDEADBEEFu64));
        let unmanaged_b = Box::into_raw(Box::new(0xFEEDFACEu64));
        unsafe {
            *((parent.0 + base_size) as *mut GcRef) = GcRef(unmanaged_a as usize);
            *((parent.0 + base_size + ptr_size) as *mut GcRef) = GcRef(unmanaged_b as usize);
        }

        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.collect_full();
        assert_eq!(gc.oldgen.object_count(), 1);

        gc.roots.clear();
        unsafe {
            drop(Box::from_raw(unmanaged_a));
            drop(Box::from_raw(unmanaged_b));
        }
    }

    #[test]
    fn varsize_struct_item_offsets_survive_minor_major_and_inspection() {
        let word = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(8192);
        let array_tid = gc.register_type(TypeInfo::varsize_with_gc_ptr_offsets(
            word,
            3 * word,
            0,
            vec![word, 2 * word],
            Vec::new(),
        ));
        let leaf_tid = gc.register_type(TypeInfo::simple(word));
        let mut array = GcAllocator::alloc_varsize_typed(&mut gc, array_tid, word, 3 * word, 2);
        let items = array.0 + word;
        let markers = [0xA11CEusize, 0xB0Busize, 0xCAFEusize, 0xD00Dusize];
        for (index, marker) in markers.into_iter().enumerate() {
            let child = gc.alloc_with_type(leaf_tid, word);
            unsafe { *(child.0 as *mut usize) = marker };
            let item = items + (index / 2) * 3 * word;
            let offset = (1 + index % 2) * word;
            unsafe { *((item + offset) as *mut GcRef) = child };
        }
        unsafe { gc.roots.add(&mut array) };

        gc.do_collect_nursery();

        let mut referents = Vec::new();
        gc.visit_referents(array.0, &mut |child| referents.push(child));
        assert_eq!(referents.len(), markers.len());
        for (child, marker) in referents.iter().zip(markers) {
            assert_eq!(unsafe { *(child.0 as *const usize) }, marker);
        }

        gc.do_collect_full();
        let mut referents_after_major = Vec::new();
        gc.visit_referents(array.0, &mut |child| referents_after_major.push(child));
        assert_eq!(referents_after_major.len(), markers.len());
        for (child, marker) in referents_after_major.iter().zip(markers) {
            assert_eq!(unsafe { *(child.0 as *const usize) }, marker);
        }
        gc.roots.clear();
    }

    #[test]
    fn unknown_struct_array_descr_tid_is_walked_after_pre_freeze_register() {
        use majit_ir::descr::{
            ArrayDescr, ArrayFlag, Descr, FieldDescr, GcCache, LLType, SimpleArrayDescr,
            SimpleFieldDescr, SimpleInteriorFieldDescr, UNSET_GC_TYPE_ID, try_downcast_arc,
        };
        use majit_ir::{DescrRef, Type};
        use std::sync::Arc;

        let mut gc = test_gc(8192);
        let word = std::mem::size_of::<GcRef>();
        // `gctypelayout.py make_type_info_group` dummy at typeid 0.
        let _dummy = gc.register_type(TypeInfo::simple(word));
        let mut cache = GcCache::new();
        let cache_key = 0x0A11CEu64;
        let descr = cache.get_array_descr(
            LLType::Array(cache_key),
            std::mem::size_of::<usize>(),
            3 * std::mem::size_of::<usize>(),
            ArrayFlag::Struct,
            Type::Ref,
            false,
            0,
            false,
            '\x00',
        );
        assert_eq!(descr.as_array_descr().unwrap().type_id(), UNSET_GC_TYPE_ID);
        let array_arc: Arc<dyn ArrayDescr> = try_downcast_arc::<SimpleArrayDescr>(descr.clone())
            .expect("SimpleArrayDescr")
            as Arc<dyn ArrayDescr>;
        let field: Arc<dyn FieldDescr> = Arc::new(SimpleFieldDescr::new(
            0,
            std::mem::size_of::<usize>(),
            std::mem::size_of::<usize>(),
            Type::Ref,
            false,
        ));
        let interior: DescrRef = Arc::new(SimpleInteriorFieldDescr::new(0, array_arc, field));
        descr
            .as_array_descr()
            .unwrap()
            .set_all_interiorfielddescrs(vec![interior]);

        cache.register_unresolved_array_tids(|base, item, len, offs, is_ptr| {
            let var_offsets = if !offs.is_empty() {
                offs
            } else if is_ptr {
                vec![0]
            } else {
                Vec::new()
            };
            if let Some(tid) = gc.types.find_varsize_type(base, item, len, &var_offsets) {
                return tid;
            }
            gc.register_type(TypeInfo::varsize_with_gc_ptr_offsets(
                base,
                item,
                len,
                var_offsets,
                Vec::new(),
            ))
        });
        let array_tid = descr.as_array_descr().unwrap().type_id();
        assert_ne!(array_tid, UNSET_GC_TYPE_ID);
        assert_ne!(array_tid, 0);

        let leaf_tid = gc.register_type(TypeInfo::simple(word));
        let mut array = GcAllocator::alloc_varsize_typed(&mut gc, array_tid, word, 3 * word, 1);
        let child = gc.alloc_with_type(leaf_tid, word);
        unsafe { *(child.0 as *mut usize) = 0xA11CE };
        let items = array.0 + word;
        unsafe { *((items + word) as *mut GcRef) = child };
        unsafe { gc.roots.add(&mut array) };

        gc.do_collect_nursery();
        let mut referents = Vec::new();
        gc.visit_referents(array.0, &mut |child| referents.push(child));
        assert_eq!(referents.len(), 1);
        assert_eq!(unsafe { *(referents[0].0 as *const usize) }, 0xA11CE);

        gc.do_collect_full();
        let mut referents_after_major = Vec::new();
        gc.visit_referents(array.0, &mut |child| referents_after_major.push(child));
        assert_eq!(referents_after_major.len(), 1);
        assert_eq!(
            unsafe { *(referents_after_major[0].0 as *const usize) },
            0xA11CE
        );
        gc.roots.clear();
    }

    #[test]
    fn test_data_integrity_across_collections() {
        // Allocate objects with distinctive data, collect, verify data.
        let mut gc = test_gc(2048);
        // Type: 32 bytes payload, one GcRef at offset 0, then 24 bytes of data.
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(32, vec![0]));

        let child = gc.alloc_with_type(tid, 32);
        unsafe {
            *(child.0 as *mut GcRef) = GcRef::NULL;
            *((child.0 + 8) as *mut u64) = 0xAAAA_BBBB_CCCC_DDDD;
            *((child.0 + 16) as *mut u64) = 0x1111_2222_3333_4444;
            *((child.0 + 24) as *mut u64) = 0x5555_6666_7777_8888;
        }

        let parent = gc.alloc_with_type(tid, 32);
        unsafe {
            *(parent.0 as *mut GcRef) = child;
            *((parent.0 + 8) as *mut u64) = 0xCAFE_BABE_DEAD_BEEF;
        }

        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.collect_nursery();

        // Verify parent data.
        let pdata = unsafe { *((root.0 + 8) as *const u64) };
        assert_eq!(pdata, 0xCAFE_BABE_DEAD_BEEF);

        // Verify child data.
        let child_ref = unsafe { *(root.0 as *const GcRef) };
        assert!(!child_ref.is_null());
        let c1 = unsafe { *((child_ref.0 + 8) as *const u64) };
        let c2 = unsafe { *((child_ref.0 + 16) as *const u64) };
        let c3 = unsafe { *((child_ref.0 + 24) as *const u64) };
        assert_eq!(c1, 0xAAAA_BBBB_CCCC_DDDD);
        assert_eq!(c2, 0x1111_2222_3333_4444);
        assert_eq!(c3, 0x5555_6666_7777_8888);

        gc.roots.clear();
    }

    #[test]
    fn marking_minor_root_greys_white_old_object() {
        // incminimark.py `_trace_drag_out1_marking_phase` checks
        // the root after `_trace_drag_out` even when it was already old.
        // This is how a stack root exposed after the major's initial root
        // snapshot re-enters the marking worklist at the next minor.
        let mut gc = test_gc(2048);
        let tid = gc.register_type(TypeInfo::simple(8));
        let old = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + 8);
        let hdr = unsafe { header_of(old.0) };
        assert!(!unsafe { (*hdr).has_flag(GcFlags::GCFLAG_VISITED) });

        gc.gc_state = GcState::Marking;
        let mut root = old;
        gc.drag_out_root(&mut root);

        assert_eq!(root, old);
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_VISITED) });
        assert!(gc.incr_state.gray_stack.is_empty());
        assert_eq!(gc.incr_state.more_gray_stack.pop(), Some(old.0));
    }

    // ── Card marking tests ──
    // incminimark.py card bytes layout: [card_bytes...][GcHeader][payload]

    /// Helper: register a varsize array type and allocate in old gen with
    /// inline card bytes. Returns (obj, array_length).
    fn alloc_card_array(gc: &mut MiniMarkGC, array_length: usize) -> (GcRef, usize) {
        let ptr_size = std::mem::size_of::<GcRef>();
        let arr_tid = gc.register_type(TypeInfo::varsize(8, ptr_size, 0, true, Vec::new()));
        let total_size = GcHeader::SIZE + 8 + ptr_size * array_length;
        let obj = gc.alloc_in_oldgen_with_cards(arr_tid, total_size, array_length, true);
        // Write the length field.
        unsafe {
            *(obj.0 as *mut usize) = array_length;
        }
        (obj, array_length)
    }

    /// incminimark.py `write_barrier_from_array`: with cards configured, only
    /// the card for `index` is marked. TRACK_YOUNG_PTRS stays set.
    #[test]
    fn write_barrier_from_array_marks_only_the_card_for_the_index() {
        let mut gc = test_gc(4096);
        assert!(gc.config.card_page_indices > 0);
        let (obj, _) = alloc_card_array(&mut gc, 512);
        let hdr = unsafe { header_of(obj.0) };
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) });
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });

        crate::GcAllocator::write_barrier_from_array(&mut gc, obj, 200);

        let card_idx = 200 >> DEFAULT_CARD_PAGE_SHIFT;
        assert!(
            gc.is_card_dirty(obj, card_idx as usize),
            "card for index 200 should be dirty"
        );
        assert!(!gc.is_card_dirty(obj, 0), "card 0 should stay clean");
        assert!(
            unsafe { (*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) },
            "card path leaves TRACK_YOUNG_PTRS set"
        );
        assert!(
            gc.old_objects_pointing_to_young.is_empty(),
            "card path does not whole-object-remember"
        );
    }

    #[test]
    fn test_card_marking_basic() {
        let mut gc = test_gc(4096);
        let (obj, _) = alloc_card_array(&mut gc, 512);
        let hdr = unsafe { header_of(obj.0) };
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) });

        // Card-marking write barrier: mark card for index 5.
        gc.do_write_barrier_card(obj, 5, DEFAULT_CARD_PAGE_SHIFT);

        // The card at index 5 >> 7 = 0 should be dirty.
        assert!(
            gc.is_card_dirty(obj, 0),
            "card 0 should be dirty after writing index 5"
        );
        assert!(
            unsafe { (*hdr).has_flag(GcFlags::GCFLAG_CARDS_SET) },
            "CARDS_SET flag should be set"
        );

        // Mark another index in a different card.
        gc.do_write_barrier_card(obj, 200, DEFAULT_CARD_PAGE_SHIFT);
        let card_idx = 200 >> DEFAULT_CARD_PAGE_SHIFT;
        assert!(
            gc.is_card_dirty(obj, card_idx as usize),
            "card for index 200 should be dirty"
        );
    }

    #[test]
    fn test_card_marking_clear_after_collection() {
        let mut gc = test_gc(4096);
        let (obj, array_length) = alloc_card_array(&mut gc, 512);
        let hdr = unsafe { header_of(obj.0) };
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) });

        // external_malloc is not zero-filled.  Initialize every item that
        // dirty-card tracing is allowed to read, as production allocation
        // rewrites do before publishing the array.
        let ptr_size = std::mem::size_of::<GcRef>();
        let items_start = obj.0 + 8;
        for i in 0..array_length {
            unsafe {
                *((items_start + i * ptr_size) as *mut GcRef) = GcRef::NULL;
            }
        }

        // Mark some cards.
        gc.do_write_barrier_card(obj, 0, DEFAULT_CARD_PAGE_SHIFT);
        gc.do_write_barrier_card(obj, 200, DEFAULT_CARD_PAGE_SHIFT);
        assert!(
            unsafe { (*hdr).has_flag(GcFlags::GCFLAG_CARDS_SET) },
            "CARDS_SET should be set before collection"
        );

        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Minor collection clears card bytes.
        gc.do_collect_nursery();

        let hdr = unsafe { header_of(root.0) };
        assert!(
            unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_CARDS_SET) },
            "CARDS_SET flag should be cleared after collection"
        );
        assert!(
            !gc.is_card_dirty(root, 0),
            "card 0 should be cleared after collection"
        );

        gc.roots.clear();
    }

    #[test]
    fn test_card_marking_dirty_cards_list() {
        let mut gc = test_gc(4096);
        let (obj, _) = alloc_card_array(&mut gc, 512);

        // Mark cards for indices in different card pages.
        gc.do_write_barrier_card(obj, 0, DEFAULT_CARD_PAGE_SHIFT);
        gc.do_write_barrier_card(obj, 128, DEFAULT_CARD_PAGE_SHIFT);
        gc.do_write_barrier_card(obj, 256, DEFAULT_CARD_PAGE_SHIFT);

        let dirty = gc.dirty_cards(obj);
        assert_eq!(dirty, vec![0, 1, 2], "should have cards 0, 1, 2 dirty");
    }

    #[test]
    fn test_card_marking_fallback_without_has_cards() {
        // Object without HAS_CARDS should fall back to remembered set.
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_in_oldgen_clear(0, GcHeader::SIZE + 16);
        // Do NOT set HAS_CARDS — no card header space allocated.

        gc.do_write_barrier_card(obj, 5, DEFAULT_CARD_PAGE_SHIFT);

        // Should fall back to full remembered set.
        assert_eq!(
            gc.old_objects_pointing_to_young.len(),
            1,
            "should add to remembered set"
        );
        assert!(
            gc.old_objects_with_cards_set.is_empty(),
            "should not add to old_objects_with_cards_set without HAS_CARDS"
        );
    }

    #[test]
    fn test_card_clear_individual() {
        let mut gc = test_gc(4096);
        let (obj1, _) = alloc_card_array(&mut gc, 512);
        let (obj2, _) = alloc_card_array(&mut gc, 512);

        gc.do_write_barrier_card(obj1, 0, DEFAULT_CARD_PAGE_SHIFT);
        gc.do_write_barrier_card(obj2, 0, DEFAULT_CARD_PAGE_SHIFT);

        // Clear only obj1's cards.
        gc.clear_cards(obj1.0);

        assert!(!gc.is_card_dirty(obj1, 0), "obj1 cards should be cleared");
        assert!(
            gc.is_card_dirty(obj2, 0),
            "obj2 cards should still be dirty"
        );
    }

    // ── Incremental marking tests ──

    #[test]
    fn test_incremental_marking_basic() {
        // Start an incremental cycle, run steps, and verify completion.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Promote some objects to old gen via minor collection.
        let obj1 = gc.alloc_with_type(tid, 16);
        let obj2 = gc.alloc_with_type(tid, 16);
        let mut root1 = obj1;
        let mut root2 = obj2;
        unsafe {
            gc.roots.add(&mut root1);
            gc.roots.add(&mut root2);
        }
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(root1.0));
        assert!(!gc.is_in_nursery(root2.0));

        // Manually start an incremental cycle.
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // Run marking steps until complete.
        let mut steps = 0;
        while gc.is_incremental_marking() {
            gc.incremental_mark_step();
            steps += 1;
            if steps > 100 {
                panic!("incremental marking did not complete");
            }
        }

        // Marking should have processed the 2 root objects.
        assert!(gc.incremental_objects_marked() >= 2);

        gc.roots.clear();
    }

    #[test]
    fn test_incremental_marking_piggyback() {
        // Verify that incremental marking progresses during nursery collections.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Promote several objects to old gen by rooting each and collecting.
        let mut roots_storage = vec![GcRef::NULL; 10];
        for r in roots_storage.iter_mut() {
            unsafe {
                gc.roots.add(r);
            }
        }
        for r in roots_storage.iter_mut() {
            let obj = gc.alloc_with_type(tid, 16);
            *r = obj;
        }
        gc.do_collect_nursery();
        for r in &roots_storage {
            assert!(!gc.is_in_nursery(r.0));
        }

        // Start an incremental cycle with a tiny budget so it takes
        // multiple steps.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // Each nursery collection should advance the marking.
        let marked_before = gc.incremental_objects_marked();
        gc.do_collect_nursery();
        // After one nursery collection with budget=1, we should have
        // marked at least one more object (if any remained).
        assert!(
            gc.incremental_objects_marked() > marked_before,
            "marking should advance during nursery collection"
        );

        gc.roots.clear();
    }

    #[test]
    fn minor_forwards_nursery_references_held_by_gray_objects() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let holder_tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let leaf_tid = gc.register_type(TypeInfo::simple(16));

        let holder = gc.alloc_with_type(holder_tid, ptr_size);
        unsafe { *(holder.0 as *mut GcRef) = GcRef::NULL };
        let mut root = holder;
        unsafe { gc.roots.add(&mut root) };
        gc.do_collect_nursery();

        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert_eq!(gc.incr_state.gray_stack, vec![root.0]);

        let young = gc.alloc_with_type(leaf_tid, 16);
        unsafe {
            *(young.0 as *mut u64) = 0xC011_EC70;
            // Model a collector-owned JITFRAME gcmap spill: it does not pass
            // through the ordinary mutator write barrier.
            *(root.0 as *mut GcRef) = young;
        }
        assert!(gc.old_objects_pointing_to_young.contains(&root.0));

        gc.disable();
        gc.do_collect_nursery();
        gc.enable();

        let forwarded = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(forwarded.0));
        assert_eq!(unsafe { *(forwarded.0 as *const u64) }, 0xC011_EC70);
        gc.do_collect_full();
        gc.roots.clear();
    }

    #[test]
    fn test_major_progress_repeats_minor_when_promotions_outpace_credit() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Build an old-gen chain large enough that a single budget=1 marking
        // step cannot finish it.
        let mut prev = GcRef::NULL;
        for _ in 0..6 {
            let obj = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
            }
            prev = obj;
        }
        let mut root = prev;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(!gc.incr_state.gray_stack.is_empty());

        // Simulate a minor collection that promoted more than one step's worth
        // of objects so incminimark-style accounting demands extra progress.
        gc.bytes_made_old_since_cycle = gc.config.nursery_size;
        gc.threshold_bytes_made_old = 0;
        let minors_before = gc.minor_collections;
        gc.run_major_progress_after_minor(false);

        assert!(
            gc.minor_collections > minors_before,
            "target A2 failure must run another bare minor before its next major step"
        );
        assert!(
            gc.incremental_objects_marked() >= 2
                || gc.threshold_bytes_made_old > gc.config.nursery_size / 2,
            "major progress should take multiple steps when promoted bytes outpace credit"
        );

        gc.roots.clear();
    }

    #[test]
    fn reserving_size_is_subtracted_from_major_progress_credit() {
        let word = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(word, vec![0]));

        let mut previous = GcRef::NULL;
        for _ in 0..4 {
            let object = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + word);
            unsafe { *(object.0 as *mut GcRef) = previous };
            previous = object;
        }
        let mut root = previous;
        unsafe { gc.roots.add(&mut root) };
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();

        // One step grants 512 bytes. The promoted total fits that unadjusted
        // credit, but not the 512 - 256 bytes left after accounting for the
        // allocation that is waiting for `collect_and_reserve`.
        gc.bytes_made_old_since_cycle = gc.config.nursery_size / 2 - 1;
        gc.threshold_bytes_made_old = 0;
        gc.pending_reserving_size = gc.config.nursery_size / 4;
        let minors_before = gc.minor_collections;

        gc.run_major_progress_after_minor(false);

        assert!(gc.minor_collections > minors_before);
        gc.pending_reserving_size = 0;
        gc.roots.clear();
    }

    #[test]
    fn reserving_more_than_credit_requires_progress_even_without_promotions() {
        // `IncrementalMiniMarkGC.minor_collection_with_major_progress` only
        // tests target A2 after `threshold >= extrasize`. Saturating the
        // subtraction would let zero promoted bytes satisfy a negative budget.
        for (extrasize, extra_minors) in [(0, 0), (512, 0), (513, 1), (1024, 1), (1025, 2)] {
            let word = std::mem::size_of::<GcRef>();
            let mut gc = test_gc(1024);
            let tid = gc.register_type(TypeInfo::with_gc_ptrs(word, vec![0]));
            let mut root = GcRef::NULL;
            for _ in 0..8 {
                let object = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + word);
                unsafe { *(object.0 as *mut GcRef) = root };
                root = object;
            }
            unsafe { gc.roots.add(&mut root) };
            gc.set_mark_budget(1);
            gc.start_incremental_cycle();
            gc.bytes_made_old_since_cycle = 0;
            gc.threshold_bytes_made_old = 0;
            gc.pending_reserving_size = extrasize;
            let minors_before = gc.minor_collections;

            gc.run_major_progress_after_minor(false);

            assert_eq!(gc.gc_state, GcState::Marking);
            assert_eq!(gc.bytes_made_old_since_cycle, 0);
            assert_eq!(
                gc.minor_collections - minors_before,
                extra_minors,
                "reservation {extrasize} must fit the credit even without promotions"
            );
            assert!(gc.threshold_bytes_made_old >= extrasize);
            gc.pending_reserving_size = 0;
            gc.roots.clear();
        }
    }

    #[test]
    fn test_incremental_marking_budget() {
        // Each step should process at most `mark_budget_per_step` objects.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Create a chain of 10 objects so marking has plenty of work.
        let mut prev = GcRef::NULL;
        let mut roots = Vec::new();
        for _ in 0..10 {
            let obj = gc.alloc_with_type(tid, ptr_size);
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
            }
            prev = obj;
            roots.push(prev);
        }

        // Root the head of the chain.
        let mut head = prev;
        unsafe {
            gc.roots.add(&mut head);
        }

        // Promote all to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(head.0));

        // Start incremental cycle with a tiny byte budget so each step still
        // processes exactly one object.  Clear the previous minor's survivor
        // sample: incminimark normally raises the step budget to twice that
        // sample, which is covered independently below.
        gc.set_mark_budget(2);
        gc.nursery_surviving_size = 0;
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // First step: marks at most 1 object.
        let done = gc.incremental_mark_step();
        assert!(!done, "should not be done after marking only 1 out of 10");
        assert_eq!(gc.incremental_objects_marked(), 1);

        // Second step: marks 1 more.
        let done = gc.incremental_mark_step();
        assert!(!done, "should not be done after 2 total");
        assert_eq!(gc.incremental_objects_marked(), 2);

        gc.roots.clear();
    }

    #[test]
    fn incremental_marking_uses_twice_the_latest_minor_survivors_as_its_budget() {
        // incminimark.py:2453-2457: even an explicitly tiny increment must
        // keep up with a high-survival minor collection.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        let mut head = GcRef::NULL;
        for _ in 0..4 {
            let obj = gc.alloc_with_type(tid, ptr_size);
            unsafe { *(obj.0 as *mut GcRef) = head };
            head = obj;
        }
        unsafe { gc.roots.add(&mut head) };
        gc.do_collect_nursery();
        assert!(gc.nursery_surviving_size > 0);

        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        gc.incremental_mark_step();

        assert_eq!(gc.incremental_objects_marked(), 4);
        gc.roots.clear();
    }

    #[test]
    fn incremental_marking_drains_the_separate_mutation_worklist() {
        // incminimark.py:2458-2470: an empty primary list means the ordinary
        // walk consumed less than half the step, so more_objects_to_trace is
        // swapped in and drained completely.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let modified = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + 16);
        unsafe { (*header_of(modified.0)).set_flag(GcFlags::GCFLAG_VISITED) };

        gc.gc_state = GcState::Marking;
        gc.incr_state.gray_stack.clear();
        gc.incr_state.more_gray_stack.push(modified.0);
        gc.nursery_surviving_size = 0;

        assert!(gc.incremental_mark_step());
        assert_eq!(gc.incremental_objects_marked(), 1);
        assert!(gc.incr_state.gray_stack.is_empty());
        assert!(gc.incr_state.more_gray_stack.is_empty());
    }

    #[test]
    fn write_barrier_during_marking_regreys_a_black_parent() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let parent = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        let child = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        unsafe {
            (*header_of(parent.0)).set_flag(GcFlags::GCFLAG_VISITED);
            (*header_of(parent.0)).set_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
            *(parent.0 as *mut GcRef) = child;
        }
        gc.gc_state = GcState::Marking;
        gc.incr_state.gray_stack.clear();
        gc.incr_state.more_gray_stack.clear();
        gc.do_write_barrier(parent);
        assert!(
            gc.incr_state.more_gray_stack.contains(&parent.0),
            "MARKING write barrier must re-grey a black parent"
        );
        gc.drain_marking_worklists();
        assert!(unsafe { (*header_of(child.0)).has_flag(GcFlags::GCFLAG_VISITED) });
    }

    #[test]
    fn test_incremental_marking_completes() {
        // A full incremental cycle (start -> repeated steps -> sweep)
        // produces the same result as a stop-the-world full collection:
        // unreachable old-gen objects are freed.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Create reachable and unreachable old-gen objects.
        let reachable = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(reachable.0 as *mut GcRef) = GcRef::NULL;
        }
        let unreachable = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(unreachable.0 as *mut GcRef) = GcRef::NULL;
        }

        let mut root = reachable;
        let mut root2 = unreachable;
        unsafe {
            gc.roots.add(&mut root);
            gc.roots.add(&mut root2);
        }

        // Promote both to old gen.
        gc.do_collect_nursery();
        assert_eq!(gc.oldgen.object_count(), 2);

        // Unroot the unreachable one.
        gc.roots.remove(&mut root2);

        // Run incremental cycle with budget=1 to force multiple steps.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();

        // Drive the cycle to completion.
        let mut iterations = 0;
        while gc.is_incremental_marking() {
            gc.incremental_mark_step();
            iterations += 1;
            if iterations > 100 {
                panic!("incremental marking did not complete");
            }
        }

        // Finish: sweep unreachable objects.
        gc.gc_step_until_scanning();

        // Only the reachable object should survive.
        assert_eq!(gc.oldgen.object_count(), 1);
        assert!(!root.is_null());

        gc.roots.clear();
    }

    #[test]
    fn incremental_sweep_spans_steps_and_preserves_interleaved_promotion() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let total_size = GcHeader::SIZE + 16;

        // More than two arena pages of white old objects ensure that the
        // one-page budget used by this tiny test nursery cannot finish sweep.
        for _ in 0..900 {
            gc.alloc_in_oldgen_clear(tid, total_size);
        }

        gc.start_incremental_cycle();
        assert!(gc.incremental_mark_step());
        assert_eq!(gc.gc_state, GcState::Sweeping);
        gc.major_collection_step();
        assert_eq!(gc.gc_state, GcState::Sweeping);

        // Disable automatic major progress only while this minor promotes its
        // survivor. The promotion allocates on ArenaCollection's fresh active
        // page lists created by mass_free_prepare, not the frozen old_* lists.
        let young = gc.alloc_with_type(tid, 16);
        unsafe { *(young.0 as *mut u64) = 0x0517_EA5E };
        let mut root = young;
        unsafe { gc.roots.add(&mut root) };
        gc.disable();
        gc.do_collect_nursery();
        gc.enable();
        assert!(!gc.is_in_nursery(root.0));

        gc.gc_step_until_scanning();
        assert_eq!(gc.gc_state, GcState::Scanning);
        assert_eq!(gc.major_collections, 1);
        assert_eq!(gc.oldgen.object_count(), 1);
        assert_eq!(unsafe { *(root.0 as *const u64) }, 0x0517_EA5E);
        gc.roots.clear();
    }

    /// `gc.get_count`'s second element: minors accumulate and a finished major
    /// is what resets them.
    ///
    /// Driven through `do_collect_full`, the operation `gc.collect()` runs, so
    /// the state the interpreter reports afterwards is the state asserted
    /// here. Stepping by hand would count differently: the reset lives in
    /// `finish_incremental_cycle`, and a full collection runs minors around
    /// its steps.
    #[test]
    fn minors_accumulate_until_a_major_finishes() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        assert_eq!(gc.minor_collections_since_major(), 0);

        gc.do_collect_nursery();
        gc.do_collect_nursery();
        assert_eq!(gc.minor_collections_since_major(), 2);

        // A survivor, so the major has something to trace.
        let mut root = gc.alloc_with_type(tid, 16);
        unsafe { gc.roots.add(&mut root) };

        gc.do_collect_full();
        assert!(gc.major_collections >= 1, "a full collection runs a major");
        assert_eq!(
            gc.minor_collections_since_major(),
            0,
            "a finished major resets the minors counted since the last one"
        );

        gc.do_collect_nursery();
        assert_eq!(gc.minor_collections_since_major(), 1);
        gc.roots.clear();
    }

    /// A GC armed for the debug checks, with the rotation ring the level also
    /// installs.
    fn debug_gc(nursery_size: usize) -> MiniMarkGC {
        MiniMarkGC::with_config(GcConfig {
            nursery_size,
            large_object_threshold: nursery_size / 2,
            debug: 2,
            ..GcConfig::default()
        })
    }

    /// The heap half of the check reaches an object through the root walk, so
    /// a flag the collection should have cleared is caught on the object
    /// rather than only on the collector's own lists.
    ///
    /// This is what makes the walk non-vacuous: without it a root enumeration
    /// that returned nothing would pass every run.
    #[test]
    #[should_panic(expected = "GCFLAG_VISITED_RMY after collection")]
    fn the_debug_walk_reaches_a_promoted_object() {
        let mut gc = debug_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_with_type(tid, 16);
        let mut root = obj;
        unsafe { gc.roots.add(&mut root) };
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(root.0), "the object promoted");

        unsafe { (*header_of(root.0)).set_flag(GcFlags::GCFLAG_VISITED_RMY) };
        gc.debug_check_consistency_at("unspecified");
    }

    /// The same walk over a heap nothing has broken reports nothing, so the
    /// test above is failing on the flag rather than on the walk itself.
    #[test]
    fn the_debug_walk_passes_a_clean_heap() {
        let mut gc = debug_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_with_type(tid, 16);
        let mut root = obj;
        unsafe { gc.roots.add(&mut root) };
        gc.do_collect_nursery();
        gc.debug_check_consistency_at("unspecified");
        gc.roots.clear();
    }

    /// incminimark.py `_debug_check_not_white`: a black parent must not point
    /// at a white child. The walk starts from roots, so rooting the parent is
    /// enough to reach the edge.
    #[test]
    #[should_panic(expected = "black -> white pointer found")]
    fn debug_check_marking_rejects_black_to_white() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = debug_gc(4096);
        let child_tid = gc.register_type(TypeInfo::simple(16));
        let parent_tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let child = gc.alloc_in_oldgen_clear(child_tid, GcHeader::SIZE + 16);
        let parent = gc.alloc_in_oldgen_clear(parent_tid, GcHeader::SIZE + ptr_size);
        unsafe { *(parent.0 as *mut GcRef) = child };
        let mut root = parent;
        unsafe { gc.roots.add(&mut root) };
        gc.gc_state = GcState::Marking;
        unsafe {
            (*header_of(parent.0)).set_flag(GcFlags::GCFLAG_VISITED);
            (*header_of(parent.0)).set_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
        }
        gc.debug_check_consistency_at("unspecified");
    }

    /// A headerless digit body is not a heap object. Its length word and the
    /// digit 1 must not be read as a header or as a pointer.
    #[test]
    fn debug_check_ignores_headerless_digit_body() {
        let mut gc = debug_gc(4096);
        let block = Box::leak(Box::new([1usize, 1usize]));
        let mut root = GcRef(block.as_mut_ptr() as usize);
        unsafe { gc.roots.add(&mut root) };
        gc.debug_check_consistency_at("unspecified");
        gc.roots.clear();
    }

    /// Without the level the body returns before the first assertion, so the
    /// same broken flag goes unreported.
    #[test]
    fn the_debug_checks_are_off_without_the_level() {
        let mut gc = test_gc(4096);
        assert_eq!(gc.config.debug, 0);
        let tid = gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_with_type(tid, 16);
        let mut root = obj;
        unsafe { gc.roots.add(&mut root) };
        gc.do_collect_nursery();

        unsafe { (*header_of(root.0)).set_flag(GcFlags::GCFLAG_VISITED_RMY) };
        gc.debug_check_consistency_at("unspecified");
        unsafe { (*header_of(root.0)).clear_flag(GcFlags::GCFLAG_VISITED_RMY) };
        gc.roots.clear();
    }

    fn panic_message(err: Box<dyn std::any::Any + Send>) -> String {
        if let Some(msg) = err.downcast_ref::<String>() {
            return msg.clone();
        }
        if let Some(msg) = err.downcast_ref::<&str>() {
            return (*msg).to_string();
        }
        "<non-string panic>".to_string()
    }

    /// One old object whose GC-pointer slot holds an integer. The walk must
    /// name that object, the slot, the integer, and the root — and must not
    /// load the integer as a header.
    #[test]
    fn debug_check_names_the_holder_of_a_bad_child() {
        let mut gc = debug_gc(4096);
        let ptr_size = std::mem::size_of::<usize>();
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let holder = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        let bad = 0x0de0_b6b3_a764_0000usize;
        unsafe { *(holder.0 as *mut usize) = bad };
        let mut root = holder;
        unsafe { gc.roots.add(&mut root) };
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gc.debug_check_consistency();
        }));
        gc.roots.clear();
        let msg = panic_message(err.expect_err("an integer child must panic"));
        assert!(msg.contains("debug_check_reachable: bad child"), "{msg}");
        assert!(msg.contains(&format!("holder={:#x}", holder.0)), "{msg}");
        assert!(msg.contains(&format!("holder_tid={tid}")), "{msg}");
        assert!(msg.contains("slot_offset=0x0"), "{msg}");
        assert!(msg.contains(&format!("value={bad:#x}")), "{msg}");
        assert!(msg.contains("unmapped"), "{msg}");
        assert!(msg.contains("root=holder"), "{msg}");
        assert!(msg.contains("parent=root"), "{msg}");
    }

    /// The parent map walks past the holder to the root that reached it.
    #[test]
    fn debug_check_names_the_root_that_reached_a_bad_child() {
        let mut gc = debug_gc(4096);
        let ptr_size = std::mem::size_of::<usize>();
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let total = GcHeader::SIZE + ptr_size;
        let root_obj = gc.alloc_in_oldgen_clear(tid, total);
        let mid = gc.alloc_in_oldgen_clear(tid, total);
        let holder = gc.alloc_in_oldgen_clear(tid, total);
        let bad = 0x0de0_b6b3_a764_0000usize;
        unsafe {
            *(root_obj.0 as *mut usize) = mid.0;
            *(mid.0 as *mut usize) = holder.0;
            *(holder.0 as *mut usize) = bad;
        }
        let mut root = root_obj;
        unsafe { gc.roots.add(&mut root) };
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gc.debug_check_consistency();
        }));
        gc.roots.clear();
        let msg = panic_message(err.expect_err("an integer child must panic"));
        assert!(msg.contains(&format!("holder={:#x}", holder.0)), "{msg}");
        assert!(msg.contains(&format!("parent={:#x}", mid.0)), "{msg}");
        assert!(msg.contains(&format!("root={:#x}", root_obj.0)), "{msg}");
        assert!(msg.contains(&format!("value={bad:#x}")), "{msg}");
    }

    /// A managed child is still checked. Rejecting every non-null slot would
    /// make the walk vacuous.
    #[test]
    fn debug_check_follows_a_managed_child() {
        let mut gc = debug_gc(4096);
        let ptr_size = std::mem::size_of::<usize>();
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let total = GcHeader::SIZE + ptr_size;
        let child = gc.alloc_in_oldgen_clear(tid, total);
        let holder = gc.alloc_in_oldgen_clear(tid, total);
        unsafe {
            *(holder.0 as *mut usize) = child.0;
            (*header_of(child.0)).set_flag(GcFlags::GCFLAG_VISITED_RMY);
        }
        let mut root = holder;
        unsafe { gc.roots.add(&mut root) };
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gc.debug_check_consistency();
        }));
        unsafe { (*header_of(child.0)).clear_flag(GcFlags::GCFLAG_VISITED_RMY) };
        gc.roots.clear();
        let msg = panic_message(err.expect_err("the child's flag must still be checked"));
        assert!(msg.contains("GCFLAG_VISITED_RMY"), "{msg}");
    }

    /// `malloc_typed` leaves are outside the arenas and still real objects.
    /// The walk follows one instead of reporting it as a broken slot.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn debug_check_follows_a_registered_external_child() {
        let mut gc = debug_gc(4096);
        let ptr_size = std::mem::size_of::<usize>();
        let holder_tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let ext_tid = gc.register_type(TypeInfo::simple(2 * ptr_size));
        let vtable = 0x51_7E_usize;
        gc.register_vtable_for_type(vtable, ext_tid);
        let ext = crate::header::alloc_with_gc_header([vtable, 0usize], ext_tid);
        let holder = gc.alloc_in_oldgen_clear(holder_tid, GcHeader::SIZE + ptr_size);
        unsafe { *(holder.0 as *mut usize) = ext as usize };
        let mut root = holder;
        unsafe { gc.roots.add(&mut root) };
        gc.debug_check_consistency();
        gc.roots.clear();
    }

    /// A host `alloc_with_gc_header` that is not an OBJECT layout has no
    /// vtable in word 0, so nothing but its header word says it is an object,
    /// and that word is what a host allocation's neighbour can imitate. The
    /// production traces skip a root outside the collector, and so does this
    /// walk: an integer in such a box's slot is not reached.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn debug_check_leaves_a_non_object_host_root_untraced() {
        let mut gc = debug_gc(4096);
        let ptr_size = std::mem::size_of::<usize>();
        // Keep the test type off tid 0, the object root.
        let _ = gc.register_type(TypeInfo::simple(ptr_size));
        let ext_tid = gc.register_type(TypeInfo::with_gc_ptrs(2 * ptr_size, vec![ptr_size]));
        let ext = crate::header::alloc_with_gc_header([0usize, 0usize], ext_tid);
        unsafe { *(ext as *mut usize).add(1) = 0x0de0_b6b3_a764_0000usize };
        let mut root = GcRef(ext as usize);
        unsafe { gc.roots.add(&mut root) };
        gc.debug_check_consistency();
        gc.roots.clear();
    }

    /// A slot that still names the pre-move address is the missing-barrier
    /// shape. The stub is mapped, so the failure has to say it is forwarded
    /// rather than treating the page as unmapped.
    #[test]
    fn debug_check_rejects_a_forwarding_stub() {
        let mut gc = debug_gc(4096);
        let ptr_size = std::mem::size_of::<usize>();
        let young_tid = gc.register_type(TypeInfo::simple(16));
        let holder_tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let young = gc.alloc_with_type(young_tid, 16);
        let copy = gc.alloc_in_oldgen_clear(young_tid, GcHeader::SIZE + 16);
        let holder = gc.alloc_in_oldgen_clear(holder_tid, GcHeader::SIZE + ptr_size);
        unsafe {
            GcHeader::set_forwarding_address(header_of(young.0), copy.0);
            *(holder.0 as *mut usize) = young.0;
        }
        let mut root = holder;
        unsafe { gc.roots.add(&mut root) };
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gc.debug_check_consistency();
        }));
        gc.roots.clear();
        let msg = panic_message(err.expect_err("a forwarding stub must panic"));
        assert!(msg.contains("debug_check_reachable: bad child"), "{msg}");
        assert!(msg.contains(&format!("holder={:#x}", holder.0)), "{msg}");
        assert!(msg.contains(&format!("value={:#x}", young.0)), "{msg}");
        assert!(msg.contains("forwarded"), "{msg}");
        assert!(msg.contains(&format!("fwd={:#x}", copy.0)), "{msg}");
    }

    /// Host block shaped like a prebuilt digit array: tid 0 in the word
    /// before the payload, `payload0` as the first payload word. The
    /// `GCFLAG_VISITED_RMY` bit makes `debug_check_object` fail if a walk
    /// treats the block as a GC object instead of leaving it alone.
    fn host_block_like_prebuilt_digits(payload0: usize) -> usize {
        let total = GcHeader::SIZE + std::mem::size_of::<usize>();
        let layout =
            std::alloc::Layout::from_size_align(total, GcHeader::ALIGN).expect("host block layout");
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!raw.is_null(), "host block allocation");
        let mut hdr = GcHeader::new(0);
        hdr.set_flag(GcFlags::GCFLAG_VISITED_RMY);
        unsafe {
            (raw as *mut GcHeader).write(hdr);
            let payload = raw.add(GcHeader::SIZE);
            (payload as *mut usize).write(payload0);
            payload as usize
        }
    }

    thread_local! {
        static DEBUG_IMMORTAL_RAW_ROOT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    fn walk_debug_immortal_raw_root(visitor: &mut dyn FnMut(&mut GcRef)) {
        let addr = DEBUG_IMMORTAL_RAW_ROOT.with(|cell| cell.get());
        if addr != 0 {
            let mut root = GcRef(addr);
            visitor(&mut root);
        }
    }

    struct ClearImmortalRawRoot;
    impl Drop for ClearImmortalRawRoot {
        fn drop(&mut self) {
            DEBUG_IMMORTAL_RAW_ROOT.with(|cell| cell.set(0));
        }
    }

    /// A GC slot may name mapped host storage, in an immortal holder (a
    /// builtin's `malloc_raw` name) and in a collector-owned one (a borrowed
    /// code-object name). An unmapped word in the same slot still panics.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn debug_check_accepts_a_host_storage_child() {
        let mut gc = debug_gc(4096);
        let ptr_size = std::mem::size_of::<usize>();
        // Keep the test type off tid 0, the object root.
        let _ = gc.register_type(TypeInfo::simple(ptr_size));
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let holder = crate::header::alloc_with_gc_header([0usize], tid);
        let raw = host_block_like_prebuilt_digits(0x1);
        unsafe { *(holder as *mut usize) = raw };
        let mut root = GcRef(holder as usize);
        unsafe { gc.roots.add(&mut root) };
        gc.debug_check_consistency();

        gc.roots.clear();
        let owned = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        unsafe { *(owned.0 as *mut usize) = raw };
        let mut owned_root = owned;
        unsafe { gc.roots.add(&mut owned_root) };
        gc.debug_check_consistency();

        let bad = 0x0de0_b6b3_a764_0000usize;
        unsafe { *(owned.0 as *mut usize) = bad };
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gc.debug_check_consistency();
        }));
        gc.roots.clear();
        let msg = panic_message(err.expect_err("an unmapped word must still panic"));
        assert!(msg.contains("debug_check_reachable: bad child"), "{msg}");
        assert!(msg.contains(&format!("holder={:#x}", owned.0)), "{msg}");
        assert!(msg.contains(&format!("value={bad:#x}")), "{msg}");
    }

    /// A slot naming host storage whose preceding allocator word reads as a
    /// registered type id with no flags is still a leaf: only a root may be
    /// admitted by that header guess. The host payload holds an integer where
    /// the guessed type would put a pointer.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn debug_check_does_not_guess_a_header_for_a_host_storage_child() {
        let mut gc = debug_gc(4096);
        let ptr_size = std::mem::size_of::<usize>();
        let _ = gc.register_type(TypeInfo::simple(ptr_size));
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let raw = host_block_like_prebuilt_digits(0x1a);
        unsafe { (header_of(raw) as *mut GcHeader).write(GcHeader::new(tid)) };
        let owned = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        unsafe { *(owned.0 as *mut usize) = raw };
        let mut root = owned;
        unsafe { gc.roots.add(&mut root) };
        gc.debug_check_consistency();
        gc.roots.clear();
    }

    /// `walk_immortal_store_roots` may yield a `std::alloc` digit block, and
    /// any root may name mapped host storage. A forwarding stub, a retired
    /// nursery address, and a collector-owned object still panic under the
    /// immortal-store label as under any other.
    #[test]
    fn debug_check_accepts_an_immortal_store_raw_root() {
        let _guard = SHADOW_STACK_TEST_LOCK.lock();
        let _clear = ClearImmortalRawRoot;
        crate::shadow_stack::register_extra_root_walker(
            walk_debug_immortal_raw_root,
            "immortal_store_roots",
        );

        let mut gc = debug_gc(4096);
        let raw = host_block_like_prebuilt_digits(0x1);

        let mut ordinary = GcRef(raw);
        unsafe { gc.roots.add(&mut ordinary) };
        gc.debug_check_consistency();
        gc.roots.clear();

        DEBUG_IMMORTAL_RAW_ROOT.with(|cell| cell.set(raw));
        gc.debug_check_consistency();
        DEBUG_IMMORTAL_RAW_ROOT.with(|cell| cell.set(0));

        let tid = gc.register_type(TypeInfo::simple(16));
        let young = gc.alloc_with_type(tid, 16);
        let copy = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + 16);
        unsafe { GcHeader::set_forwarding_address(header_of(young.0), copy.0) };
        DEBUG_IMMORTAL_RAW_ROOT.with(|cell| cell.set(young.0));
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gc.debug_check_consistency();
        }));
        DEBUG_IMMORTAL_RAW_ROOT.with(|cell| cell.set(0));
        let msg = panic_message(err.expect_err("a forwarding stub root must panic"));
        assert!(msg.contains("debug_check_reachable: bad root"), "{msg}");
        assert!(msg.contains("label=immortal_store_roots"), "{msg}");
        assert!(msg.contains("forwarded"), "{msg}");

        let old = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + 16);
        unsafe { (*header_of(old.0)).set_flag(GcFlags::GCFLAG_VISITED_RMY) };
        DEBUG_IMMORTAL_RAW_ROOT.with(|cell| cell.set(old.0));
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gc.debug_check_consistency();
        }));
        unsafe { (*header_of(old.0)).clear_flag(GcFlags::GCFLAG_VISITED_RMY) };
        DEBUG_IMMORTAL_RAW_ROOT.with(|cell| cell.set(0));
        let msg = panic_message(err.expect_err("a collector-owned root must still be checked"));
        assert!(msg.contains("GCFLAG_VISITED_RMY"), "{msg}");

        if crate::nursery::HAS_PROTECT {
            let mut rotating = MiniMarkGC::with_config(GcConfig {
                nursery_size: 4096,
                large_object_threshold: 2048,
                debug: 2,
                gc_nursery_debug: true,
                ..GcConfig::default()
            });
            let rot_tid = rotating.register_type(TypeInfo::simple(16));
            let obj = rotating.alloc_with_type(rot_tid, 16);
            let stale = obj.0;
            let mut root = obj;
            unsafe { rotating.roots.add(&mut root) };
            rotating.do_collect_nursery();
            rotating.roots.clear();
            assert!(
                rotating.nursery.contains_retired(stale),
                "the collected arena must be the retired one"
            );
            DEBUG_IMMORTAL_RAW_ROOT.with(|cell| cell.set(stale));
            let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                rotating.debug_check_consistency();
            }));
            DEBUG_IMMORTAL_RAW_ROOT.with(|cell| cell.set(0));
            let msg = panic_message(err.expect_err("a retired nursery root must panic"));
            assert!(msg.contains("debug_check_reachable: bad root"), "{msg}");
            assert!(msg.contains("label=immortal_store_roots"), "{msg}");
            assert!(msg.contains(&format!("value={stale:#x}")), "{msg}");
        }
    }

    /// A headerless digit block whose preceding allocator word reads as a
    /// registered type id with no flags. The walk must not read that word as
    /// a header, under the immortal-store label or any other: the block is
    /// host storage, and tracing it would read the digit as a pointer.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn debug_check_does_not_guess_a_header_for_a_host_storage_root() {
        let _guard = SHADOW_STACK_TEST_LOCK.lock();
        let _clear = ClearImmortalRawRoot;
        crate::shadow_stack::register_extra_root_walker(
            walk_debug_immortal_raw_root,
            "immortal_store_roots",
        );

        let mut gc = debug_gc(4096);
        let ptr_size = std::mem::size_of::<usize>();
        let _ = gc.register_type(TypeInfo::simple(ptr_size));
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let raw = host_block_like_prebuilt_digits(0x0de0_b6b3_a764_0000);
        unsafe { (header_of(raw) as *mut GcHeader).write(GcHeader::new(tid)) };

        DEBUG_IMMORTAL_RAW_ROOT.with(|cell| cell.set(raw));
        gc.debug_check_consistency();
        DEBUG_IMMORTAL_RAW_ROOT.with(|cell| cell.set(0));

        let mut ordinary = GcRef(raw);
        unsafe { gc.roots.add(&mut ordinary) };
        gc.debug_check_consistency();
        gc.roots.clear();
    }

    /// The same integer is invisible when the level is off: the body returns
    /// before the walk, which is what keeps an unset `PYPY_GC_DEBUG` on the
    /// ordinary path.
    #[test]
    fn debug_check_ignores_a_bad_child_when_the_level_is_off() {
        let mut gc = test_gc(4096);
        assert_eq!(gc.config.debug, 0);
        let ptr_size = std::mem::size_of::<usize>();
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let holder = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        unsafe { *(holder.0 as *mut usize) = 0x0de0_b6b3_a764_0000 };
        let mut root = holder;
        unsafe { gc.roots.add(&mut root) };
        gc.debug_check_consistency();
        gc.roots.clear();
    }

    /// No `#[cfg(debug_assertions)]`: `debug_check_consistency` self-gates on
    /// the debug level and asserts for real, so the check survives a release
    /// build — which is what makes `PYPY_GC_DEBUG` worth setting there.
    #[test]
    #[should_panic(expected = "raw_malloc_might_sweep must be empty outside SWEEPING")]
    fn rawmalloc_sweep_candidates_require_sweeping_state() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 4096,
            large_object_threshold: 2048,
            debug: 1,
            ..GcConfig::default()
        });
        let tid = gc.register_type(TypeInfo::simple(16));
        let raw_size = gc.oldgen.small_request_threshold() + std::mem::size_of::<usize>();
        gc.alloc_in_oldgen_clear(tid, raw_size);
        gc.oldgen.sweep_prepare();
        assert!(gc.oldgen.rawmalloc_sweep_pending());
        assert_eq!(gc.gc_state, GcState::Scanning);
        gc.debug_check_consistency_at("unspecified");
    }

    #[test]
    fn stop_the_world_full_collection_drains_marking_and_sweeping() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let live = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + 16);
        let mut root = live;
        unsafe { gc.roots.add(&mut root) };

        let raw_size = gc.oldgen.small_request_threshold() + std::mem::size_of::<usize>();
        for _ in 0..20 {
            gc.alloc_in_oldgen_clear(tid, raw_size);
        }
        let majors_before = gc.major_collections;

        gc.do_collect_full();

        assert_eq!(gc.gc_state, GcState::Scanning);
        assert!(!gc.oldgen.rawmalloc_sweep_pending());
        assert_eq!(gc.major_collections, majors_before + 1);
        assert_eq!(gc.oldgen.object_count(), 1);
        assert_eq!(root.0, live.0);
        gc.roots.clear();
    }

    /// With `set_stress_collect(true)`, `alloc_with_type` forces a full
    /// collection on every allocation. A large nursery is used so that no
    /// collection would fire naturally across these allocations — the
    /// collection counter therefore isolates the flag's effect. A rooted
    /// nursery object must be forwarded out of the nursery and stay valid.
    #[test]
    #[cfg(feature = "gc_stress")]
    fn test_gc_stress_forces_collection_per_alloc() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1 << 20); // 1 MiB nursery: no natural collection here.
        gc.set_stress_collect(true);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        let obj = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj.0 as *mut GcRef) = GcRef::NULL;
        }
        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root as *mut GcRef);
        }

        let before = gc.minor_collections;
        for _ in 0..5 {
            let _ = gc.alloc_with_type(tid, ptr_size);
        }
        assert!(
            gc.minor_collections >= before + 5,
            "stress_collect must force a collection on every allocation"
        );

        // The rooted object was forwarded; the root slot must now hold a
        // valid, non-nursery address.
        assert!(!root.is_null());
        assert!(
            !gc.is_in_nursery(root.0),
            "rooted object should have been promoted out of the nursery"
        );

        gc.roots.clear();
    }

    /// Isolation guarantee: with the `gc_stress` feature compiled in but
    /// `stress_collect` left off (the default), a large-nursery GC performs no
    /// forced collection — so suites that assert `minor_collections == 0` are
    /// unaffected even when the feature is enabled workspace-wide.
    #[test]
    #[cfg(feature = "gc_stress")]
    fn test_gc_stress_off_by_default() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1 << 20);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        for _ in 0..5 {
            let _ = gc.alloc_with_type(tid, ptr_size);
        }
        assert_eq!(
            gc.minor_collections, 0,
            "stress_collect defaults off: no collection without opt-in"
        );
    }

    /// A forwarded header is a moved object, not an unknown type.  The
    /// barrier's type-id validation must say so and name the live copy: the
    /// bare out-of-range number reads the same whether the header is a corpse
    /// marker or genuine corruption, and only the first has a holder to go
    /// find.
    #[test]
    #[should_panic(expected = "already moved to")]
    fn write_barrier_on_a_forwarded_object_names_the_live_copy() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1 << 20);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        let obj = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj.0 as *mut GcRef) = GcRef::NULL;
        }
        let stale = obj.0;
        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root as *mut GcRef);
        }
        gc.do_collect_nursery();
        gc.roots.clear();
        assert_ne!(root.0, stale, "the collection must move the object");
        assert!(
            unsafe { (*header_of(stale)).is_forwarded() },
            "the moved-from header must carry the forwarding marker"
        );

        // What a holder that kept the pre-collection pointer does next.
        gc.remember_young_pointer(GcRef(stale));
    }

    /// `_remember_young_pointer_inlined`: a nursery object never enters
    /// `old_objects_pointing_to_young`. A barrier that fires on one (a
    /// flag byte read through a stale pointer) fails at the producer.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "remember_young_pointer on nursery object")]
    fn remember_young_pointer_rejects_a_nursery_object() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1 << 20);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        let obj = gc.alloc_with_type(tid, ptr_size);
        assert!(gc.is_in_nursery(obj.0));
        gc.remember_young_pointer(obj);
    }

    #[test]
    fn test_incremental_gc_under_allocation_pressure() {
        // Allocate many objects forming a linked list, promote to old gen,
        // then run an incremental major cycle with a tiny budget while
        // continuing to allocate. Verify data integrity throughout.

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1024);
        // Object layout: [next: GcRef][data: u64] = 16 bytes, GC ptr at offset 0
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size + 8, vec![0]));

        // Build a linked list of 20 objects.
        let mut prev = GcRef::NULL;
        let mut all_roots: Vec<GcRef> = Vec::new();
        for i in 0..20u64 {
            let obj = gc.alloc_with_type(tid, ptr_size + 8);
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
                *((obj.0 + ptr_size) as *mut u64) = 0xA000 + i;
            }
            prev = obj;
            all_roots.push(obj);
        }

        // Root only the head of the list.
        let mut head = prev;
        unsafe {
            gc.roots.add(&mut head);
        }

        // Promote the whole list to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(head.0));

        // Start an incremental cycle with budget = 2 so it takes many steps.
        gc.set_mark_budget(2);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // Interleave incremental marking steps with new allocations.
        let mut step_count = 0;
        while gc.is_incremental_marking() {
            // Allocate a few new (short-lived) objects to maintain pressure.
            for _ in 0..5 {
                let tmp = gc.alloc_with_type(tid, ptr_size + 8);
                unsafe {
                    *(tmp.0 as *mut GcRef) = GcRef::NULL;
                    *((tmp.0 + ptr_size) as *mut u64) = 0xFFFF;
                }
            }
            let done = gc.incremental_mark_step();
            step_count += 1;
            if done {
                break;
            }
            assert!(step_count < 200, "incremental marking should converge");
        }

        // Complete the cycle.
        gc.gc_step_until_scanning();
        assert!(gc.major_collections > 0);

        // Walk the list from the head and verify all data values.
        let mut cursor = head;
        let mut count = 0;
        while !cursor.is_null() {
            let data = unsafe { *((cursor.0 + ptr_size) as *const u64) };
            // data should be 0xA000 + (19 - count) because the list was
            // built in reverse.
            assert_eq!(
                data,
                0xA000 + (19 - count) as u64,
                "data corruption detected at node {count}"
            );
            cursor = unsafe { *(cursor.0 as *const GcRef) };
            count += 1;
        }
        assert_eq!(count, 20, "entire list should be reachable");

        gc.roots.clear();
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_card_marking_under_write_pressure() {
        // Allocate a large array in old gen, write GC refs into many slots,
        // and verify that card marking accurately tracks the dirty ranges
        // so that nursery objects stored in the array survive collection.

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(2048);

        // Element type: simple 16-byte object.
        let elem_tid = gc.register_type(TypeInfo::simple(16));

        // Array type: varsize, base_size = 8 (length field at offset 0),
        // each item is a GcRef (item_size = ptr_size), items have GC ptrs.
        let arr_tid = gc.register_type(TypeInfo::varsize(
            8,          // base_size (length field)
            ptr_size,   // item_size
            0,          // length_offset
            true,       // items_have_gc_ptrs
            Vec::new(), // no fixed GC ptr fields
        ));

        let array_length = 512usize;
        let total_payload = 8 + ptr_size * array_length;

        // Allocate with inline card bytes (incminimark.py:1027 parity).
        let arr = gc.alloc_in_oldgen_with_cards(
            arr_tid,
            GcHeader::SIZE + total_payload,
            array_length,
            true,
        );

        // Write the length field.
        unsafe {
            *(arr.0 as *mut usize) = array_length;
        }

        let hdr = unsafe { header_of(arr.0) };
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_HAS_CARDS) });

        // Initialize all array slots to NULL.
        let items_start = arr.0 + 8;
        for i in 0..array_length {
            unsafe {
                *((items_start + i * ptr_size) as *mut GcRef) = GcRef::NULL;
            }
        }

        // Write nursery objects into scattered array positions and trigger
        // card-marking write barriers.
        let write_indices: Vec<usize> = vec![0, 1, 5, 64, 127, 128, 200, 255, 256, 400, 511];
        let mut expected_cards: Vec<usize> = Vec::new();
        let mut nursery_objs: Vec<(usize, GcRef)> = Vec::new();

        for &idx in &write_indices {
            let obj = gc.alloc_with_type(elem_tid, 16);
            // Write a distinctive marker.
            unsafe {
                *(obj.0 as *mut u64) = 0xCAFE_0000 + idx as u64;
            }
            // Store into the array.
            unsafe {
                *((items_start + idx * ptr_size) as *mut GcRef) = obj;
            }
            // Write barrier with card marking.
            gc.do_write_barrier_card(arr, idx, DEFAULT_CARD_PAGE_SHIFT);
            let card = idx >> DEFAULT_CARD_PAGE_SHIFT;
            if !expected_cards.contains(&card) {
                expected_cards.push(card);
            }
            nursery_objs.push((idx, obj));
        }

        // Verify the correct cards are dirty.
        let mut dirty_set: Vec<usize> = gc.dirty_cards(arr);
        dirty_set.sort();
        expected_cards.sort();
        assert_eq!(
            dirty_set, expected_cards,
            "dirty card set should match expected cards"
        );

        // Root the array so the collection traces it via card scanning.
        let mut root = arr;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Trigger nursery collection — this should use card scanning
        // to find and promote the nursery objects stored in the array.
        gc.do_collect_nursery();

        // After collection, all stored objects should be promoted and
        // their data should be intact.
        for &(idx, _orig) in &nursery_objs {
            let slot_ref = unsafe { *((items_start + idx * ptr_size) as *const GcRef) };
            assert!(
                !slot_ref.is_null(),
                "array slot {idx} should not be null after collection"
            );
            assert!(
                !gc.is_in_nursery(slot_ref.0),
                "array slot {idx} should be promoted to old gen"
            );
            let marker = unsafe { *(slot_ref.0 as *const u64) };
            assert_eq!(
                marker,
                0xCAFE_0000 + idx as u64,
                "data in array slot {idx} should be preserved"
            );
        }

        // Cards should be cleared after collection.
        let hdr = unsafe { header_of(root.0) };
        assert!(
            unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_CARDS_SET) },
            "CARDS_SET should be cleared after collection"
        );
        assert!(
            gc.old_objects_with_cards_set.is_empty(),
            "old_objects_with_cards_set should be empty after collection"
        );
        assert!(
            unsafe { (*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) },
            "TRACK_YOUNG_PTRS should be re-set after collection"
        );

        // Now do a second round of writes to verify card marking works
        // again after collection.
        let more_indices = vec![10, 300, 450];
        for &idx in &more_indices {
            let obj = gc.alloc_with_type(elem_tid, 16);
            unsafe {
                *(obj.0 as *mut u64) = 0xBEEF_0000 + idx as u64;
                *((items_start + idx * ptr_size) as *mut GcRef) = obj;
            }
            gc.do_write_barrier_card(arr, idx, DEFAULT_CARD_PAGE_SHIFT);
        }

        gc.do_collect_nursery();

        for &idx in &more_indices {
            let slot_ref = unsafe { *((items_start + idx * ptr_size) as *const GcRef) };
            assert!(!slot_ref.is_null());
            assert!(!gc.is_in_nursery(slot_ref.0));
            let marker = unsafe { *(slot_ref.0 as *const u64) };
            assert_eq!(marker, 0xBEEF_0000 + idx as u64);
        }

        gc.roots.clear();
    }

    // ── Write barrier + incremental marking interaction tests ──

    #[test]
    fn first_major_visit_arms_child_for_incremental_write_barrier() {
        let word = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let holder_tid = gc.register_type(TypeInfo::with_gc_ptrs(word, vec![0]));
        let leaf_tid = gc.register_type(TypeInfo::simple(word));

        let grandparent = gc.alloc_in_oldgen_clear(holder_tid, GcHeader::SIZE + word);
        let holder = gc.alloc_in_oldgen_clear(holder_tid, GcHeader::SIZE + word);
        let white_child = gc.alloc_in_oldgen_clear(leaf_tid, GcHeader::SIZE + word);
        unsafe {
            *(grandparent.0 as *mut GcRef) = holder;
            *(holder.0 as *mut GcRef) = GcRef::NULL;
        }
        let mut root = grandparent;
        unsafe { gc.roots.add(&mut root) };

        gc.start_incremental_cycle();
        gc.old_objects_pointing_to_young.clear();
        let root_work = gc.incr_state.gray_stack.pop().expect("root was seeded");
        gc.mark_object(root_work);
        let holder_work = gc.incr_state.gray_stack.pop().expect("child was greyed");
        assert_eq!(holder_work, holder.0);
        gc.mark_object(holder_work);

        assert!(unsafe { (*header_of(holder.0)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
        assert!(!unsafe { (*header_of(white_child.0)).has_flag(GcFlags::GCFLAG_VISITED) });

        gc.do_write_barrier(holder);
        unsafe { *(holder.0 as *mut GcRef) = white_child };
        assert!(gc.old_objects_pointing_to_young.contains(&holder.0));

        while gc.gc_state != GcState::Scanning {
            gc.major_collection_step();
        }
        assert!(gc.oldgen.contains(white_child.0));
        gc.roots.clear();
    }

    #[test]
    fn major_visit_ignores_no_heap_ptrs_objects_without_stamping_visited() {
        let word = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let holder_tid = gc.register_type(TypeInfo::with_gc_ptrs(word, vec![0]));
        let leaf_tid = gc.register_type(TypeInfo::simple(word));
        let holder = gc.alloc_in_oldgen_clear(holder_tid, GcHeader::SIZE + word);
        let clean_prebuilt = gc.alloc_in_oldgen_clear(leaf_tid, GcHeader::SIZE + word);
        unsafe {
            *(holder.0 as *mut GcRef) = clean_prebuilt;
            (*header_of(clean_prebuilt.0)).set_flag(GcFlags::GCFLAG_NO_HEAP_PTRS);
        }
        let mut root = holder;
        unsafe { gc.roots.add(&mut root) };

        gc.start_incremental_cycle();
        let root_work = gc.incr_state.gray_stack.pop().expect("root was seeded");
        gc.mark_object(root_work);

        assert!(!unsafe { (*header_of(clean_prebuilt.0)).has_flag(GcFlags::GCFLAG_VISITED) });
        assert!(!gc.incr_state.gray_stack.contains(&clean_prebuilt.0));
        gc.roots.clear();
    }

    #[test]
    fn test_write_barrier_during_incremental_marking() {
        // Verify that write barriers fired during an active incremental
        // marking cycle correctly add old-gen objects to the remembered set.
        // The remembered set is processed at the next minor collection,
        // ensuring mutated old-gen objects are re-scanned.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        // Object layout: [field: GcRef] = ptr_size bytes, GC ptr at offset 0
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Create 3 old-gen objects (A, B, C) with no pointers initially.
        let obj_a = gc.alloc_with_type(tid, ptr_size);
        let obj_b = gc.alloc_with_type(tid, ptr_size);
        let obj_c = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_a.0 as *mut GcRef) = GcRef::NULL;
            *(obj_b.0 as *mut GcRef) = GcRef::NULL;
            *(obj_c.0 as *mut GcRef) = GcRef::NULL;
        }

        let mut root_a = obj_a;
        let mut root_b = obj_b;
        let mut root_c = obj_c;
        unsafe {
            gc.roots.add(&mut root_a);
            gc.roots.add(&mut root_b);
            gc.roots.add(&mut root_c);
        }

        // Promote all to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(root_a.0));
        assert!(!gc.is_in_nursery(root_b.0));
        assert!(!gc.is_in_nursery(root_c.0));

        // All old-gen objects should have TRACK_YOUNG_PTRS set.
        assert!(unsafe { (*header_of(root_a.0)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
        assert!(unsafe { (*header_of(root_b.0)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
        assert!(unsafe { (*header_of(root_c.0)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });

        // Start an incremental marking cycle with budget=1 so it stays
        // active across multiple steps.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // `seed_major_root` arms all newly gray roots for the next minor so
        // collector-owned JITFRAME spills cannot strand nursery references.
        // Reset that one-time collector setup here to isolate the ordinary
        // mutator write-barrier behavior this test covers.
        gc.old_objects_pointing_to_young.clear();
        unsafe {
            (*header_of(root_a.0)).set_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
            (*header_of(root_b.0)).set_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
            (*header_of(root_c.0)).set_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS);
        }

        // During marking, perform write barriers on A and B.
        gc.do_write_barrier(root_a);
        gc.do_write_barrier(root_b);

        // A and B should be in the remembered set.
        assert!(
            gc.old_objects_pointing_to_young.contains(&root_a.0),
            "write barrier should add A to remembered set during marking"
        );
        assert!(
            gc.old_objects_pointing_to_young.contains(&root_b.0),
            "write barrier should add B to remembered set during marking"
        );
        // C was not written to, so it shouldn't be in remembered set.
        assert!(
            !gc.old_objects_pointing_to_young.contains(&root_c.0),
            "C should not be in remembered set"
        );

        // TRACK_YOUNG_PTRS should be cleared on A and B.
        assert!(!unsafe { (*header_of(root_a.0)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
        assert!(!unsafe { (*header_of(root_b.0)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
        // C still has it.
        assert!(unsafe { (*header_of(root_c.0)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });

        // Drive the incremental cycle to completion via nursery collections.
        for _ in 0..50 {
            if !gc.is_incremental_marking() {
                break;
            }
            gc.do_collect_nursery();
        }

        // No objects should be lost — all 3 are still rooted and should
        // survive the sweep.
        gc.do_collect_full();
        assert_eq!(
            gc.oldgen.object_count(),
            3,
            "all 3 rooted objects should survive full collection"
        );

        gc.roots.clear();
    }

    #[test]
    fn test_mutation_during_incremental_preserves_reachability() {
        // During an incremental marking cycle, mutate an old-gen object to
        // point to a newly promoted object (D). The write barrier ensures D
        // is reachable through the remembered set, so D survives the sweep.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        // Object layout: [next: GcRef] = ptr_size bytes, GC ptr at offset 0
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Build a chain A→B→C in the nursery.
        let obj_c = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_c.0 as *mut GcRef) = GcRef::NULL;
        }
        let obj_b = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_b.0 as *mut GcRef) = obj_c;
        }
        let obj_a = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_a.0 as *mut GcRef) = obj_b;
        }

        let mut root_a = obj_a;
        unsafe {
            gc.roots.add(&mut root_a);
        }

        // Promote A→B→C to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(root_a.0));
        let promoted_b = unsafe { *(root_a.0 as *const GcRef) };
        let promoted_c = unsafe { *(promoted_b.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(promoted_b.0));
        assert!(!gc.is_in_nursery(promoted_c.0));

        // Start incremental marking with a small budget.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // Do one marking step — only partially through the graph.
        gc.incremental_mark_step();

        // Now allocate D in the nursery.
        let obj_d = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_d.0 as *mut GcRef) = GcRef::NULL;
        }

        // Promote D to old gen by rooting it temporarily.
        let mut root_d = obj_d;
        unsafe {
            gc.roots.add(&mut root_d);
        }
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(root_d.0));

        // Mutate A to point to D instead of B (A→D).
        // The write barrier ensures A is in the remembered set.
        gc.do_write_barrier(root_a);
        unsafe {
            *(root_a.0 as *mut GcRef) = root_d;
        }

        // Remove D's direct root — D is now only reachable via A→D.
        gc.roots.remove(&mut root_d);

        // Complete the incremental cycle: drive marking to completion
        // and then sweep.
        gc.do_collect_full();

        // A and D should survive (A is rooted, D reachable via A→D).
        // B and C may or may not survive depending on marking order,
        // but D MUST survive.
        let d_addr = root_d.0;
        let a_field = unsafe { *(root_a.0 as *const GcRef) };
        assert_eq!(
            a_field.0, d_addr,
            "A should still point to D after collection"
        );

        // Verify D is actually alive by checking it's still in old gen.
        assert!(
            !a_field.is_null(),
            "D must survive collection — it's reachable via A"
        );

        gc.roots.clear();
    }

    #[test]
    fn test_nursery_alloc_during_incremental_marking() {
        // Allocate new nursery objects between incremental marking steps.
        // Trigger nursery collections that piggyback marking steps.
        // Verify that newly allocated objects are correctly promoted and
        // that the original old-gen object graph remains intact.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1024); // small nursery to force frequent collections
        // Object layout: [next: GcRef][data: u64] = ptr_size + 8 bytes
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size + 8, vec![0]));

        // Create initial old-gen objects (a small chain of 5).
        let mut prev = GcRef::NULL;
        for i in 0..5u64 {
            let obj = gc.alloc_with_type(tid, ptr_size + 8);
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
                *((obj.0 + ptr_size) as *mut u64) = 0xBB00 + i;
            }
            prev = obj;
        }

        let mut head = prev;
        unsafe {
            gc.roots.add(&mut head);
        }

        // Promote the chain to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(head.0));
        let old_count_after_promote = gc.oldgen.object_count();
        assert_eq!(old_count_after_promote, 5);

        // Start incremental marking with budget=1.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        let minor_before = gc.minor_collections;

        // Allocate many nursery objects between marking steps. The small
        // nursery (1024 bytes) forces nursery collections that piggyback
        // incremental marking steps.
        for i in 0..100u64 {
            let tmp = gc.alloc_with_type(tid, ptr_size + 8);
            unsafe {
                *(tmp.0 as *mut GcRef) = GcRef::NULL;
                *((tmp.0 + ptr_size) as *mut u64) = 0xDD00 + i;
            }
        }

        let minor_during = gc.minor_collections - minor_before;
        assert!(
            minor_during >= 2,
            "should have triggered multiple nursery collections, got {minor_during}"
        );

        // The incremental marking should have advanced via piggybacking.
        assert!(
            gc.incremental_objects_marked() > 0,
            "piggybacked marking should have processed some objects"
        );

        // Drive any remaining incremental marking to completion and sweep.
        // Use do_collect_full which correctly handles an in-progress cycle.
        gc.do_collect_full();
        assert!(gc.major_collections > 0);

        // Verify the original chain is intact — the 5 rooted objects
        // survived the incremental major cycle.
        let mut cursor = head;
        let mut count = 0;
        while !cursor.is_null() {
            let data = unsafe { *((cursor.0 + ptr_size) as *const u64) };
            assert_eq!(
                data,
                0xBB00 + (4 - count) as u64,
                "original chain data corrupted at node {count}"
            );
            cursor = unsafe { *(cursor.0 as *const GcRef) };
            count += 1;
        }
        assert_eq!(count, 5, "entire chain should be reachable after cycle");

        gc.roots.clear();
    }

    #[test]
    fn full_collection_finishes_active_cycle_then_runs_fresh_cycle() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let mut rooted = gc.alloc_with_type(tid, 16);
        unsafe { gc.roots.add(&mut rooted) };
        gc.do_collect_nursery();

        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());
        let major_before = gc.major_collections;

        gc.do_collect_full();

        assert_eq!(gc.major_collections, major_before + 2);
        assert!(gc.oldgen.contains(rooted.0));
        gc.roots.clear();
    }

    #[test]
    fn full_collection_forwards_nursery_references_held_by_gray_objects() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let holder_tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let leaf_tid = gc.register_type(TypeInfo::simple(16));

        let holder = gc.alloc_with_type(holder_tid, ptr_size);
        unsafe { *(holder.0 as *mut GcRef) = GcRef::NULL };
        let mut root = holder;
        unsafe { gc.roots.add(&mut root) };
        gc.do_collect_nursery();

        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert_eq!(gc.incr_state.gray_stack, vec![root.0]);

        let young = gc.alloc_with_type(leaf_tid, 16);
        unsafe {
            *(young.0 as *mut u64) = 0xF011_C011_EC70;
            *(root.0 as *mut GcRef) = young;
        }

        // incminimark.py `minor_and_major_collection` finishes the active
        // cycle through `gc_step_until`, whose leading minor forwards this
        // edge before the gray holder is consumed.
        gc.do_collect_full();

        let forwarded = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(forwarded.0));
        assert_eq!(unsafe { *(forwarded.0 as *const u64) }, 0xF011_C011_EC70);
        assert!(gc.oldgen.contains(forwarded.0));
        gc.roots.clear();
    }

    // ── JIT integration hook tests ──

    #[test]
    fn test_jit_remember_young_pointer() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        // Allocate an old-gen object.
        let obj = gc.alloc_in_oldgen_clear(0, GcHeader::SIZE + 16);
        assert!(!gc.is_in_nursery(obj.0));

        // Initially TRACK_YOUNG_PTRS is set.
        let hdr = unsafe { header_of(obj.0) };
        assert!(unsafe { (*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });

        // JIT fast-path barrier: clears flag and adds to remembered set.
        gc.jit_remember_young_pointer(obj);

        assert!(unsafe { !(*hdr).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
        assert_eq!(gc.old_objects_pointing_to_young_len(), 1);

        // Calling again adds a second entry (JIT fast-path does not
        // deduplicate; the collector handles this during minor collection).
        gc.jit_remember_young_pointer(obj);
        assert_eq!(gc.old_objects_pointing_to_young_len(), 2);
    }

    #[test]
    fn test_jit_remember_young_pointer_survives_collection() {
        // Verify that the remembered-set entry from jit_remember_young_pointer
        // causes a young object to survive minor collection.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        // Create an old-gen parent and a young child.
        let parent = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + std::mem::size_of::<GcRef>());
        let child = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(child.0 as *mut u64) = 0xABCD_1234;
            *(parent.0 as *mut GcRef) = child;
        }

        // Use the JIT hook instead of do_write_barrier.
        gc.jit_remember_young_pointer(parent);

        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.collect_nursery();

        // The child should have been promoted.
        let child_ref = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(child_ref.0));
        assert!(!child_ref.is_null());
        let val = unsafe { *(child_ref.0 as *const u64) };
        assert_eq!(val, 0xABCD_1234);

        gc.roots.clear();
    }

    #[test]
    fn test_can_optimize_cond_call() {
        let gc = test_gc(4096);
        assert!(gc.can_optimize_cond_call());
    }

    #[test]
    fn clean_setarrayitems_optimization_follows_card_configuration() {
        let cards = test_gc(4096);
        assert!(!cards.can_optimize_clean_setarrayitems());

        let no_cards = MiniMarkGC::with_config(GcConfig {
            nursery_size: 4096,
            large_object_threshold: 2048,
            card_page_indices: 0,
            ..GcConfig::default()
        });
        assert!(no_cards.can_optimize_clean_setarrayitems());
    }

    /// env.py `get_L2cache_linux2_cpuinfo`: use the smallest valid
    /// per-CPU K/k value, ignore another unit and malformed lines, and retain
    /// the upstream `_findend('\n' + label)` first-line behavior.
    #[test]
    fn linux_cpuinfo_cache_probe_matches_upstream_parser() {
        let data = b"cache size : 1 KB\n\
processor : 0\n\
cache size : 32768 KB\n\
cache size : 16 MB\n\
cache size = 8 KB\n\
processor : 1\n\
cache size\t: 8192 kB\n";
        assert_eq!(l2cache_from_cpuinfo(data, b"cache size"), 8 * 1024 * 1024);
        assert_eq!(l2cache_from_cpuinfo(data, b"L2 cache"), -1);
    }

    /// env.py `get_L2cache_linux2_system_cpu_index`: L2 and L3 are
    /// minimized independently across CPUs and then added.
    #[test]
    fn linux_sysfs_cache_probe_sums_smallest_l2_and_l3() {
        static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);
        let serial = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "majit-gc-cache-fixture-{}-{serial}",
            std::process::id()
        ));

        let entries = [
            (0, 0, "1\n", "64K\n"),
            (0, 1, "2\n", "2048K\n"),
            (0, 2, "3\n", "8192K\n"),
            (1, 0, "2\n", "1024K\n"),
            (1, 1, "3\n", "4096K\n"),
        ];
        for (cpu, index, level, size) in entries {
            let dir = root.join(format!("cpu{cpu}/cache/index{index}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("level"), level).unwrap();
            std::fs::write(dir.join("size"), size).unwrap();
        }

        let result = get_l2cache_linux_system_cpu_index(root.to_str().unwrap());
        assert_eq!(result, 5 * 1024 * 1024);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn linux_sysfs_cache_size_requires_leading_decimal_kilobytes() {
        assert_eq!(parse_sysfs_cache_size(b"2048K\n"), Some(2 * 1024 * 1024));
        assert_eq!(parse_sysfs_cache_size(b"512k\n"), Some(512 * 1024));
        assert_eq!(parse_sysfs_cache_size(b"2M\n"), None);
        assert_eq!(parse_sysfs_cache_size(b" K\n"), None);
    }

    #[test]
    fn test_estimate_best_nursery_size() {
        // env.py:443-456 — half the L2 cache when it exceeds 8MB, else the
        // 4MB unknown-cache fallback. Either way it must never return a size
        // below the fallback, so the major-collection floor (nursery*8) is
        // always well-defined.
        let est = estimate_best_nursery_size();
        assert!(est >= DEFAULT_NURSERY_SIZE, "estimate {est} below fallback");
        // best_nursery_size_for_l2cache mirrors env.py's strict `> 8MB` test.
        assert_eq!(best_nursery_size_for_l2cache(-1), DEFAULT_NURSERY_SIZE);
        assert_eq!(
            best_nursery_size_for_l2cache(8 * 1024 * 1024),
            DEFAULT_NURSERY_SIZE
        );
        assert_eq!(
            best_nursery_size_for_l2cache(32 * 1024 * 1024),
            16 * 1024 * 1024
        );
    }

    #[test]
    fn test_can_optimize_cond_call_via_trait() {
        let gc = test_gc(4096);
        let alloc: &dyn GcAllocator = &gc;
        assert!(alloc.can_optimize_cond_call());
        assert!(!alloc.can_optimize_clean_setarrayitems());
    }

    #[test]
    fn test_gc_step_no_work_when_old_gen_small() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        // With an almost-empty old gen, gc_step should do nothing.
        assert!(!gc.gc_step());
        assert!(!gc.is_incremental_marking());
    }

    #[test]
    fn test_gc_step_respects_enabled_flag() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));
        gc.alloc_in_oldgen_clear(0, GcHeader::SIZE + 16);
        gc.next_major_collection_threshold = 0.0;
        assert!(gc.threshold_reached(0));

        gc.disable();
        assert!(!gc.gc_step());
        assert_eq!(gc.gc_state, GcState::Scanning);

        gc.enable();
        assert!(gc.gc_step());
    }

    #[test]
    fn test_gc_step_triggers_incremental() {
        let mut gc = test_gc(256);
        gc.register_type(TypeInfo::simple(16));

        // Force a full collection to establish the next-major threshold baseline.
        let obj = gc.alloc_with_type(0, 16);
        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }
        gc.collect_full();

        // Fill old gen to trigger the major-cycle ratio threshold.
        // Allocate many objects directly in old gen.
        for _ in 0..200 {
            gc.alloc_in_oldgen_clear(0, GcHeader::SIZE + 16);
        }

        // gc_step should now start an incremental cycle and do work.
        let did_work = gc.gc_step();
        assert!(did_work);

        gc.roots.clear();
    }

    #[test]
    fn test_gc_step_advances_marking() {
        // Nursery must be large enough to hold all 20 chain links without an
        // auto-collect mid-loop; otherwise intermediate links get dropped and
        // `prev` ends up chaining through stale nursery addresses (reset-zero
        // bytes parse as tid=0 / `simple(16)`, which causes `copy_nursery_object`
        // to over-read past the nursery end).
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Build a chain of old-gen objects so there's marking work to do.
        let mut prev = GcRef::NULL;
        let ptr_tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));
        for _ in 0..20 {
            let obj = gc.alloc_with_type(ptr_tid, std::mem::size_of::<GcRef>());
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
            }
            prev = obj;
        }

        let mut root = prev;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Promote everything to old gen.
        gc.collect_nursery();

        // Add enough old-gen objects to cross the next-major threshold. The
        // threshold is floored at `min_heap_size = nursery_size * 8` = 32KB
        // for this 4096-byte nursery (incminimark.py:488,562), so the old gen
        // must exceed 32KB before `gc_step` will start a cycle — a few hundred
        // small objects is intentionally not enough under the parity-correct
        // floor.
        for _ in 0..4000 {
            gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + 16);
        }

        // First step: start cycle.
        let work1 = gc.gc_step();
        assert!(work1);
        let marked_after_1 = gc.incremental_objects_marked();

        // Second step: advance marking further.
        if gc.is_incremental_marking() {
            let work2 = gc.gc_step();
            assert!(work2);
            assert!(gc.incremental_objects_marked() >= marked_after_1);
        }

        gc.roots.clear();
    }

    // ── Pin / Unpin tests ──

    #[test]
    fn test_pin_prevents_nursery_move() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(obj.0));

        // Write a marker value.
        unsafe {
            *(obj.0 as *mut u64) = 0xCAFE_BABE;
        }

        // Pin the object and root it.
        assert!(gc.pin(obj));
        assert!(gc.is_pinned(obj));

        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Trigger minor collection.
        gc.do_collect_nursery();

        // The root should still point to the same nursery address.
        assert_eq!(root.0, obj.0);
        assert!(gc.is_in_nursery(root.0));

        // Data should be intact.
        let val = unsafe { *(root.0 as *const u64) };
        assert_eq!(val, 0xCAFE_BABE);

        gc.roots.clear();
    }

    #[test]
    fn test_can_move_nursery_pin_and_bounds() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        // rgc.py can_move — a young nursery object can still move.
        let obj = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(obj.0));
        assert!(gc.can_move(obj));

        // incminimark.py can_move answers by nursery membership even while
        // the object's PINNED flag temporarily prevents an actual move.
        assert!(gc.pin(obj));
        assert!(gc.can_move(obj));
        gc.unpin(obj);
        assert!(gc.can_move(obj));

        // Null and out-of-nursery addresses never move (rgc.py:231
        // "with non-moving GCs, it is always False").
        assert!(!gc.can_move(GcRef(0)));
        assert!(!gc.can_move(GcRef(0xdead_0000)));
    }

    #[test]
    fn test_unpin_allows_move() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(obj.0));

        // Write a marker.
        unsafe {
            *(obj.0 as *mut u64) = 0xDEAD_BEEF;
        }

        // Pin, then unpin.
        assert!(gc.pin(obj));
        gc.unpin(obj);
        assert!(!gc.is_pinned(obj));

        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Collection should now move the object to old gen.
        gc.do_collect_nursery();

        // The root should now point to old gen (different address).
        assert!(!gc.is_in_nursery(root.0));
        assert_ne!(root.0, obj.0);

        // Data should be preserved after the move.
        let val = unsafe { *(root.0 as *const u64) };
        assert_eq!(val, 0xDEAD_BEEF);

        gc.roots.clear();
    }

    #[test]
    fn test_is_pinned_query() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(0, 16);

        // Not pinned by default.
        assert!(!gc.is_pinned(obj));

        // Pin it.
        assert!(gc.pin(obj));
        assert!(gc.is_pinned(obj));

        // incminimark.py rejects a second pin rather than keeping a
        // reference count whose first unpin could invalidate the second pin.
        assert!(!gc.pin(obj));

        // Null cannot be pinned.
        assert!(!gc.pin(GcRef(0)));
        assert!(!gc.is_pinned(GcRef(0)));

        // Unpin.
        gc.unpin(obj);
        assert!(!gc.is_pinned(obj));
    }

    #[test]
    fn test_pin_enforces_capacity_and_cannot_pin_type_query() {
        unsafe fn noop_destructor(_obj_addr: usize) {}
        fn trigger() {}

        let mut gc = test_gc(4096);
        gc.max_number_of_pinned_objects = 1;
        let plain_tid = gc.register_type(TypeInfo::simple(16));
        let gcptr_tid = gc.register_type(TypeInfo::with_gc_ptrs(16, vec![0]));
        let weakref_tid = gc.register_type(TypeInfo::weakref());
        let destructor_tid = gc.register_type(TypeInfo::with_destructor(16, noop_destructor));

        let first = gc.alloc_with_type(plain_tid, 16);
        let second = gc.alloc_with_type(plain_tid, 16);
        assert!(gc.pin(first));
        assert_eq!(gc.pinned_objects_in_nursery, 1);
        // incminimark.py:1122-1123 checks the configured count first.
        assert!(!gc.pin(second));
        gc.unpin(first);
        assert_eq!(gc.pinned_objects_in_nursery, 0);

        // gctypelayout.py `q_cannot_pin`.
        let with_gcptr = gc.alloc_with_type(gcptr_tid, 16);
        let weakref = gc.alloc_with_type(weakref_tid, crate::weakref::SIZEOF_WEAKREF);
        let with_destructor = gc.alloc_with_type(destructor_tid, 16);
        assert!(!gc.pin(with_gcptr));
        assert!(!gc.pin(weakref));
        assert!(!gc.pin(with_destructor));

        // Pyre's finalizer queue is registered dynamically rather than
        // encoded as RPython `customdata`, so the header bit supplies the same
        // rejection once registration has occurred.
        let finalizable = gc.alloc_with_type(plain_tid, 16);
        GcAllocator::register_finalizer(&mut gc, 0, finalizable, trigger);
        assert!(!gc.pin(finalizable));
    }

    #[test]
    fn test_unreachable_pin_is_not_an_implicit_root() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_with_type(tid, 16);
        assert!(gc.pin(obj));

        // incminimark.py:1777-1785: the survivor stack is populated only by
        // traced references. With no such reference, the pinned object dies.
        gc.do_collect_nursery();
        // `obj` is now a dangling GC address. Upstream's mode-0 arena reset
        // deliberately leaves recycled bytes untouched, so asking `_is_pinned`
        // about that dead address has no defined answer; the live-pin counter
        // and absence of a barrier are the observable collector state.
        assert_eq!(gc.pinned_objects_in_nursery, 0);
        assert!(gc.nursery_barriers.is_empty());
    }

    #[test]
    fn test_old_parent_is_retraced_while_it_points_to_a_pin() {
        let mut gc = test_gc(4096);
        let child_tid = gc.register_type(TypeInfo::simple(16));
        let parent_tid = gc.register_type(TypeInfo::with_gc_ptrs(16, vec![0]));
        let child = gc.alloc_with_type(child_tid, 16);
        let mut parent = gc.alloc_with_type(parent_tid, 16);
        unsafe { *(parent.0 as *mut GcRef) = child };
        assert!(gc.pin(child));
        unsafe { gc.roots.add(&mut parent) };

        // The parent promotes, but its pinned child stays at the same nursery
        // address and records the old parent for the following minor.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(parent.0));
        assert!(gc.is_pinned(child));
        assert_eq!(unsafe { *(parent.0 as *const GcRef) }, child);
        assert_eq!(gc.old_objects_pointing_to_pinned, vec![parent.0]);

        // No remembered-set entry is needed: the dedicated PyPy parent list
        // revisits the edge and keeps the child alive for another cycle.
        gc.do_collect_nursery();
        assert!(gc.is_pinned(child));
        assert_eq!(gc.old_objects_pointing_to_pinned, vec![parent.0]);

        // Once that edge disappears, the next minor does not rediscover the
        // child and pinning alone no longer preserves it.
        unsafe { *(parent.0 as *mut GcRef) = GcRef::NULL };
        gc.do_collect_nursery();
        // `child` is no longer a valid GC reference; mode-0 arena reset does
        // not promise to rewrite its stale header bytes.
        assert_eq!(gc.pinned_objects_in_nursery, 0);
        assert!(gc.nursery_barriers.is_empty());
        assert!(gc.old_objects_pointing_to_pinned.is_empty());
        gc.roots.clear();
    }

    /// `collect_roots_in_nursery` computes
    /// `use_jit_frame_stoppers = not any_pinned_object_from_earlier` and passes
    /// it to `walk_roots` as `is_minor`. A pin that survived the previous minor
    /// was never promoted and still sits at its nursery address, so the walkers
    /// that skip a clean area on a minor have to be told to walk everything.
    #[test]
    fn a_surviving_pin_makes_the_next_minor_announce_a_full_extra_root_walk() {
        // The walker registry has no removal and every test binary thread
        // shares it, so record on the collecting thread only.
        thread_local! {
            static SEEN: std::cell::RefCell<Vec<crate::shadow_stack::ExtraRootWalkKind>> =
                const { std::cell::RefCell::new(Vec::new()) };
        }
        fn record(_visit: &mut dyn FnMut(&mut GcRef)) {
            SEEN.with(|seen| {
                seen.borrow_mut()
                    .push(crate::shadow_stack::extra_root_walk_kind())
            });
        }

        let _guard = SHADOW_STACK_TEST_LOCK.lock();
        crate::shadow_stack::clear();
        crate::shadow_stack::register_extra_root_walker(record, "test_pin_walk");
        SEEN.with(|seen| seen.borrow_mut().clear());

        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let mut obj = gc.alloc_with_type(tid, 16);
        assert!(gc.pin(obj));
        unsafe { gc.roots.add(&mut obj) };

        // Nothing was pinned before this one, so the first minor still
        // announces the skip; it discovers the pin and leaves it in place.
        gc.do_collect_nursery();
        assert!(gc.is_pinned(obj));

        // The pin now predates the previous minor: it is "from earlier".
        gc.do_collect_nursery();
        assert!(gc.is_pinned(obj));

        gc.roots.clear();
        assert_eq!(
            SEEN.with(|seen| seen.borrow().clone()),
            vec![
                crate::shadow_stack::ExtraRootWalkKind::Minor,
                crate::shadow_stack::ExtraRootWalkKind::Major,
            ]
        );
    }

    /// An old parent that is also a jitframe root must stay in the list: the
    /// root walk does not trace an old frame, so the parent is found only by
    /// the drained copy of the list and re-recorded into the fresh one.
    #[test]
    fn test_old_parent_that_is_a_jitframe_root_stays_in_the_list() {
        let _guard = SHADOW_STACK_TEST_LOCK.lock();
        crate::shadow_stack::clear();
        let mut gc = test_gc(4096);
        let child_tid = gc.register_type(TypeInfo::simple(16));
        let parent_tid = gc.register_type(TypeInfo::with_gc_ptrs(16, vec![0]));
        let child = gc.alloc_with_type(child_tid, 16);
        let mut parent = gc.alloc_with_type(parent_tid, 16);
        unsafe { *(parent.0 as *mut GcRef) = child };
        assert!(gc.pin(child));
        unsafe { gc.roots.add(&mut parent) };

        gc.do_collect_nursery();
        assert!(gc.oldgen.contains(parent.0));
        assert!(gc.is_pinned(child));
        assert_eq!(gc.old_objects_pointing_to_pinned, vec![parent.0]);

        // Publish the promoted parent as a jitframe root as well.
        crate::shadow_stack::push_jf(parent);
        gc.do_collect_nursery();
        assert_eq!(gc.old_objects_pointing_to_pinned, vec![parent.0]);
        assert!(gc.is_pinned(child));

        // The edge is still the child's only reference, so losing the parent
        // record above would let this minor reclaim it.
        gc.do_collect_nursery();
        assert!(gc.is_pinned(child));
        assert_eq!(unsafe { *(parent.0 as *const GcRef) }, child);

        crate::shadow_stack::pop_jf_to(0);
        gc.roots.clear();
    }

    #[test]
    fn test_incremental_cycle_root_walk_skips_non_gc_roots() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(tid, 16);
        let mut rooted = obj;
        let mut external = GcRef(Box::into_raw(Box::new(0xDEADBEEFu64)) as usize);
        unsafe {
            gc.roots.add(&mut rooted);
            gc.roots.add(&mut external);
        }

        gc.do_collect_nursery();
        gc.start_incremental_cycle();
        while !gc.gc_step() {}

        assert!(!rooted.is_null());
        assert!(gc.oldgen.contains(rooted.0));

        gc.roots.clear();
        unsafe {
            drop(Box::from_raw(external.0 as *mut u64));
        }
    }

    #[test]
    fn test_full_collection_root_walk_skips_non_gc_roots() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(tid, 16);
        let mut rooted = obj;
        let mut external = GcRef(Box::into_raw(Box::new(0xFEEDFACEu64)) as usize);
        unsafe {
            gc.roots.add(&mut rooted);
            gc.roots.add(&mut external);
        }

        gc.do_collect_nursery();
        gc.do_collect_full();

        assert!(!rooted.is_null());
        assert!(gc.oldgen.contains(rooted.0));

        gc.roots.clear();
        unsafe {
            drop(Box::from_raw(external.0 as *mut u64));
        }
    }

    /// A minor collection rewrites the SLOT, so reading through the position
    /// yields the promoted address.
    ///
    /// `walk_stack_root` (`shadowstack.py`) hands the collector the slot
    /// itself — `invoke(..., addr)` where `addr` is the stack address, not the
    /// value — which is what lets the forwarded address be stored back into it.
    /// Every holder that addresses its root by position therefore sees the move
    /// without being told about it.
    #[test]
    fn minor_collection_rewrites_an_owner_root_slot() {
        let _guard = SHADOW_STACK_TEST_LOCK.lock();
        crate::shadow_stack::clear();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(tid, 16);
        assert!(gc.is_in_nursery(obj.0));
        unsafe {
            *(obj.0 as *mut u64) = 0xD00D_D00D;
        }
        let root = crate::shadow_stack::OwnerRootGuard::new(obj);

        gc.do_collect_nursery();

        let moved = root.get();
        assert_ne!(
            moved, obj,
            "a surviving nursery object is promoted, so this run cannot tell a \
             rewritten slot from an unmoved object"
        );
        assert!(gc.oldgen.contains(moved.0));
        assert_eq!(unsafe { *(moved.0 as *const u64) }, 0xD00D_D00D);
    }

    /// The HOLDER may move. Addressing a root by position is what makes that
    /// true, and it is the whole difference from registering the address of the
    /// field that holds the pointer: a registered field address is only valid
    /// while its owner stays put, so an owner that can be returned by value has
    /// to be pinned behind a heap allocation first.
    #[test]
    fn an_owner_root_survives_its_holder_being_moved() {
        let _guard = SHADOW_STACK_TEST_LOCK.lock();
        crate::shadow_stack::clear();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(tid, 16);
        assert!(gc.is_in_nursery(obj.0));
        unsafe {
            *(obj.0 as *mut u64) = 0xFEED_FEED;
        }

        // Three distinct addresses for the same guard, one of them the heap,
        // with a collection between the last move and the read.
        let root = crate::shadow_stack::OwnerRootGuard::new(obj);
        let moved_once = root;
        let mut holders = vec![moved_once];
        gc.do_collect_nursery();

        let moved = holders.pop().unwrap().get();
        assert_ne!(moved, obj);
        assert!(gc.oldgen.contains(moved.0));
        assert_eq!(unsafe { *(moved.0 as *const u64) }, 0xFEED_FEED);
    }

    #[test]
    fn test_incremental_cycle_marks_jitframe_shadow_stack_roots() {
        let _guard = SHADOW_STACK_TEST_LOCK.lock();
        crate::shadow_stack::clear();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(tid, 16);
        let depth = crate::shadow_stack::jf_depth();
        crate::shadow_stack::push_jf(obj);

        gc.do_collect_nursery();
        let rooted = crate::shadow_stack::jf_top_ptr();
        assert!(gc.oldgen.contains(rooted.0));

        gc.start_incremental_cycle();
        gc.gc_step_until_scanning();

        assert!(gc.oldgen.contains(rooted.0));
        crate::shadow_stack::pop_jf_to(depth);
    }

    #[test]
    fn test_full_collection_marks_jitframe_shadow_stack_roots() {
        let _guard = SHADOW_STACK_TEST_LOCK.lock();
        crate::shadow_stack::clear();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(tid, 16);
        let depth = crate::shadow_stack::jf_depth();
        crate::shadow_stack::push_jf(obj);

        gc.do_collect_nursery();
        let rooted = crate::shadow_stack::jf_top_ptr();
        assert!(gc.oldgen.contains(rooted.0));

        gc.do_collect_full();

        assert!(gc.oldgen.contains(rooted.0));
        crate::shadow_stack::pop_jf_to(depth);
    }

    // Note: a `test_minor_root_walk_skips_interior_nursery_pointers`
    // test used to live here. It was predicated on a majit-local
    // extension where the nursery tracked exact object-start addresses
    // and `is_nursery_object_start` filtered out interior pointers.
    //
    // incminimark.py is_in_nursery is a pure range check
    //     return self.nursery <= addr < self.nursery + self.nursery_size
    // RPython guarantees that GC roots are exact object pointers (the
    // shadow stack only ever carries exact pointers produced by the
    // compiler), so interior-pointer filtering is not part of the GC
    // contract. The test disagreed with that contract and was removed
    // to keep majit-gc structurally aligned with RPython.
    //
    // `drag_out_root` copies `root.address[0]` (`_trace_drag_out`).
    // An interior slot is a publisher defect and must panic, not snap.

    #[test]
    #[should_panic(expected = "invalid type_id")]
    fn test_minor_root_walk_does_not_snap_interior_pointers() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(32));
        let obj = gc.alloc_with_type(tid, 32);
        unsafe {
            *(obj.0 as *mut u64) = 0x42;
            *((obj.0 + 8) as *mut u64) = 0x42;
        }
        let mut exact = obj;
        gc.drag_out_root(&mut exact);
        let mut interior = GcRef(obj.0 + 16);
        gc.drag_out_root(&mut interior);
    }

    /// incminimark.py:3068-3079 dead-target branch. A WEAKREF whose
    /// target is a nursery object with no GC root must have its
    /// `weakptr` slot invalidated to 0 after the minor cycle.
    #[test]
    fn test_minor_invalidate_young_weakref_dead_target() {
        let mut gc = test_gc(4096);
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        // Allocate the target first, then a WEAKREF whose payload's
        // `weakptr` slot points to that target. Neither object is
        // rooted in `gc.roots`; only the WEAKREF itself is rooted so
        // the minor cycle keeps the WEAKREF alive long enough to
        // observe the post-invalidation slot.
        let target = gc.alloc_with_type(target_tid, 16);
        let wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        unsafe {
            *((wref.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = target;
        }

        let mut wref_root = wref;
        unsafe { gc.roots.add(&mut wref_root) };

        gc.do_collect_nursery();

        // The WEAKREF survived → forwarded into old-gen.
        assert!(gc.oldgen.contains(wref_root.0));
        // Target had no root and was not reachable through the WEAKREF
        // (the collector does not trace weakptr), so it died — the
        // slot must read null.
        let after = unsafe {
            crate::weakref::ll_weakref_deref(wref_root.0 as *const crate::weakref::Weakref)
        };
        assert!(after.is_null(), "dead-target weakptr should read null");

        gc.roots.clear();
    }

    /// incminimark.py:3071-3074 live-target branch. A WEAKREF whose
    /// target survives the minor cycle (because something else
    /// rooted it) must have its `weakptr` slot rewritten to the
    /// target's new old-gen address.
    #[test]
    fn test_minor_invalidate_young_weakref_live_target() {
        let mut gc = test_gc(4096);
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        let target = gc.alloc_with_type(target_tid, 16);
        let wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        unsafe {
            *((wref.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = target;
        }

        // Root both the target and the weakref so both forward.
        let mut target_root = target;
        let mut wref_root = wref;
        unsafe {
            gc.roots.add(&mut target_root);
            gc.roots.add(&mut wref_root);
        }

        gc.do_collect_nursery();

        // Both forwarded out.
        assert!(gc.oldgen.contains(target_root.0));
        assert!(gc.oldgen.contains(wref_root.0));
        // The weakref's slot now reads the target's forwarded address.
        let after = unsafe {
            crate::weakref::ll_weakref_deref(wref_root.0 as *const crate::weakref::Weakref)
        };
        assert_eq!(after.0, target_root.0);

        gc.roots.clear();
    }

    /// `collect_oldrefs_to_nursery`: a young WEAKREF reached only through a
    /// remembered old object's custom trace, whose target is live via a root,
    /// must have `weakptr` rewritten (`invalidate_young_weakrefs`). Extra-rooting
    /// the WEAKREF itself would copy it outside that path; word 0 is `weakptr`
    /// and forwarding overwrites it on the nursery corpse.
    ///
    /// Interned strings live in `rweakvaldict::WeakDict<StrKey>` (`WEAK_INTERN`);
    /// `ll_set_nonnull` write-barriers the `entries` array so a minor traces a
    /// young intern WEAKREF the same way (`intern_publish`). This fixture uses
    /// a custom-trace holder as that remembered old object.
    #[test]
    fn test_minor_custom_trace_young_weakref_live_target() {
        use std::cell::Cell;
        thread_local! {
            static WEAKREF_SLOT: Cell<usize> = const { Cell::new(0) };
        }
        unsafe fn trace_weakref_slot(_obj_addr: usize, f: &mut dyn FnMut(*mut GcRef)) {
            WEAKREF_SLOT.with(|slot| {
                let addr = slot.get();
                if addr == 0 {
                    return;
                }
                let mut gcref = GcRef(addr);
                f(&mut gcref);
                slot.set(gcref.0);
            });
        }

        let mut gc = test_gc(4096);
        let holder_tid = gc.register_type(TypeInfo::with_custom_trace(8, trace_weakref_slot));
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        let holder = gc.alloc_in_oldgen_clear(holder_tid, GcHeader::SIZE + 8);
        let target = gc.alloc_with_type(target_tid, 16);
        let wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        unsafe {
            *((wref.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = target;
        }
        WEAKREF_SLOT.with(|slot| slot.set(wref.0));
        gc.do_write_barrier(holder);

        let mut target_root = target;
        unsafe {
            gc.roots.add(&mut target_root);
        }

        gc.do_collect_nursery();

        let wref_after = WEAKREF_SLOT.with(|slot| slot.get());
        assert!(
            gc.oldgen.contains(wref_after),
            "custom-trace must copy the young WEAKREF"
        );
        let after = unsafe {
            crate::weakref::ll_weakref_deref(wref_after as *const crate::weakref::Weakref)
        };
        assert_eq!(
            after.0, target_root.0,
            "live target's weakptr must be forwarded"
        );

        gc.roots.clear();
        WEAKREF_SLOT.with(|slot| slot.set(0));
    }

    /// incminimark.py `invalidate_young_weakrefs`: a clean prebuilt target is
    /// immortal but intentionally never marked VISITED. Carrying its weakref
    /// into `old_objects_with_weakrefs` would therefore clear a live edge at
    /// the next major collection.
    #[test]
    fn young_weakref_to_no_heap_ptrs_target_skips_old_weakref_queue() {
        let mut gc = test_gc(4096);
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        let target = gc.alloc_in_oldgen_clear(target_tid, GcHeader::SIZE + 16);
        unsafe { (*header_of(target.0)).set_flag(GcFlags::GCFLAG_NO_HEAP_PTRS) };
        let wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        unsafe {
            *((wref.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = target;
        }

        let mut wref_root = wref;
        unsafe { gc.roots.add(&mut wref_root) };
        gc.do_collect_nursery();

        assert!(gc.oldgen.contains(wref_root.0));
        assert!(gc.old_objects_with_weakrefs.is_empty());
        let after = unsafe {
            crate::weakref::ll_weakref_deref(wref_root.0 as *const crate::weakref::Weakref)
        };
        assert_eq!(after, target);
        gc.roots.clear();
    }

    /// incminimark.py:3065-3066 "weakref itself dies" branch. The
    /// WEAKREF object has no root → the cycle treats the entry as a
    /// no-op (no panic, no UB) and the bookkeeping list ends up empty.
    #[test]
    fn test_minor_invalidate_young_weakref_self_dies() {
        let mut gc = test_gc(4096);
        let wref_tid = gc.register_type(TypeInfo::weakref());

        let _wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        // No root on the WEAKREF itself.
        assert_eq!(gc.young_objects_with_weakrefs.len(), 1);

        gc.do_collect_nursery();

        // The cycle drained the list and didn't push the dead weakref
        // into the old-side queue.
        assert!(gc.young_objects_with_weakrefs.is_empty());
        assert!(gc.old_objects_with_weakrefs.is_empty());
    }

    /// incminimark.py:3116-3122 dying-target branch in
    /// `invalidate_old_weakrefs`. After a full GC where the WEAKREF's
    /// target is no longer rooted, the weakptr slot reads null and
    /// the bookkeeping list drops the weakref.
    #[test]
    fn test_major_invalidate_old_weakref_dead_target() {
        let mut gc = test_gc(4096);
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        let target = gc.alloc_with_type(target_tid, 16);
        let wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        unsafe {
            *((wref.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = target;
        }

        // Root all three (target, wref) and pre-promote them to old
        // gen via a minor collection so the major cycle sees both as
        // oldgen-resident.
        let mut target_root = target;
        let mut wref_root = wref;
        unsafe {
            gc.roots.add(&mut target_root);
            gc.roots.add(&mut wref_root);
        }
        gc.do_collect_nursery();
        assert!(gc.oldgen.contains(target_root.0));
        assert!(gc.oldgen.contains(wref_root.0));
        assert_eq!(gc.old_objects_with_weakrefs.len(), 1);

        // Drop the target root; keep wref rooted. Full GC should
        // sweep the target and null out wref's weakptr slot.
        gc.roots.remove(&mut target_root);
        gc.do_collect_full();

        let after = unsafe {
            crate::weakref::ll_weakref_deref(wref_root.0 as *const crate::weakref::Weakref)
        };
        assert!(after.is_null(), "dead-target old weakptr should read null");
        // Weakref survived but its entry was popped from the list
        // because its (now-null) slot has nothing left to clear.
        assert!(gc.old_objects_with_weakrefs.is_empty());

        gc.roots.clear();
    }

    /// incminimark.py:3120-3121 live-target keep branch. When both
    /// the WEAKREF and its target survive the major mark, the
    /// weakref carries over into the next cycle's bookkeeping list.
    #[test]
    fn test_major_invalidate_old_weakref_live_target() {
        let mut gc = test_gc(4096);
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        let target = gc.alloc_with_type(target_tid, 16);
        let wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        unsafe {
            *((wref.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = target;
        }

        let mut target_root = target;
        let mut wref_root = wref;
        unsafe {
            gc.roots.add(&mut target_root);
            gc.roots.add(&mut wref_root);
        }

        gc.do_collect_full();

        let after = unsafe {
            crate::weakref::ll_weakref_deref(wref_root.0 as *const crate::weakref::Weakref)
        };
        assert_eq!(after.0, target_root.0);
        assert_eq!(gc.old_objects_with_weakrefs.len(), 1);

        gc.roots.clear();
    }

    /// `gc_step_until` runs `_minor_collection`'s MARKING `if_black` before
    /// every major step. `do_collect_oldgen_nonmoving` finishes an in-progress
    /// MARKING cycle with no minor; a black remembered parent must still be
    /// retraced so a white old child stored since the last visit survives.
    #[test]
    fn nonmoving_major_regreys_remembered_black_parents() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        let child = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        unsafe { *(child.0 as *mut GcRef) = GcRef::NULL };
        let parent = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        unsafe { *(parent.0 as *mut GcRef) = GcRef::NULL };
        let mut root = parent;
        unsafe { gc.roots.add(&mut root) };

        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert_eq!(gc.gc_state, GcState::Marking);
        let mut steps = 0;
        while gc.gc_state == GcState::Marking
            && (gc.incr_state.gray_stack.contains(&parent.0)
                || gc.incr_state.more_gray_stack.contains(&parent.0))
        {
            gc.incremental_mark_step();
            steps += 1;
            assert!(steps < 50, "parent should leave the marking worklist");
        }
        if gc.gc_state != GcState::Marking {
            gc.roots.clear();
            return;
        }
        assert!(unsafe { (*header_of(parent.0)).has_flag(GcFlags::GCFLAG_VISITED) });
        assert!(unsafe { !(*header_of(child.0)).has_flag(GcFlags::GCFLAG_VISITED) });

        gc.do_write_barrier(parent);
        unsafe { *(parent.0 as *mut GcRef) = child };
        assert!(gc.old_objects_pointing_to_young.contains(&parent.0));

        gc.do_collect_oldgen_nonmoving();
        assert_eq!(
            gc.oldgen.object_count(),
            2,
            "white old child stored into a black remembered parent must survive the non-moving finish"
        );
        gc.roots.clear();
    }

    /// A non-moving major (`do_collect_oldgen_nonmoving`) sweeps dead old-gen
    /// objects while a populated nursery is left byte-for-byte intact, marks
    /// through nursery objects to keep `old -> nursery -> old` survivors, and
    /// clears `VISITED` off every greyed nursery object as the strictly-last
    /// step — otherwise a later minor promotion would memcpy a stale VISITED
    /// bit into the promoted copy.
    #[test]
    fn nonmoving_major_marks_through_nursery_and_clears_visited() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // q: an old-gen leaf reachable ONLY through a nursery object.
        let q = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        unsafe { *(q.0 as *mut GcRef) = GcRef(0) };
        // n: nursery object pointing at q (nursery -> old edge).
        let n = gc.alloc_with_type(tid, ptr_size);
        assert!(gc.is_in_nursery(n.0));
        unsafe { *(n.0 as *mut GcRef) = q };
        // o: old-gen root pointing at n (old -> nursery edge).
        let o = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        unsafe { *(o.0 as *mut GcRef) = n };
        // d: unreachable old-gen object that must be swept.
        let d = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        unsafe { *(d.0 as *mut GcRef) = GcRef(0) };

        let mut root = o;
        unsafe { gc.roots.add(&mut root) };
        assert_eq!(gc.oldgen.object_count(), 3); // o, q, d

        gc.do_collect_oldgen_nonmoving();

        // o and q survive (o -> n -> q); d swept. The nursery is untouched.
        assert_eq!(gc.oldgen.object_count(), 2);
        assert!(gc.is_in_nursery(n.0), "nursery object must not move/free");
        assert_eq!(unsafe { *(n.0 as *const GcRef) }, q, "n -> q edge intact");

        // The must-fix: no greyed nursery object retains VISITED.
        let n_hdr = unsafe { header_of(n.0) };
        assert!(
            unsafe { !(*n_hdr).has_flag(GcFlags::GCFLAG_VISITED) },
            "stale nursery VISITED would memcpy into the next minor's promotion"
        );
        // q (old-gen survivor) had VISITED cleared by the oldgen sweep.
        let q_hdr = unsafe { header_of(q.0) };
        assert!(unsafe { !(*q_hdr).has_flag(GcFlags::GCFLAG_VISITED) });

        gc.roots.clear();
    }

    /// A young object registered with a finalizer survives the next minor even
    /// when nothing reaches it, so a non-moving major that runs before that
    /// minor must keep the old objects it names. Sweeping them left the
    /// promoted object's fields pointing at freed memory, which its finalizer
    /// then read (`_socket.socket.__del__` reading an unboxed `_fd` block).
    #[test]
    fn nonmoving_major_keeps_old_children_of_an_unreachable_young_finalizer() {
        fn trigger() {}

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // q: an old-gen leaf named only by the young finalizer object.
        let q = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        unsafe { *(q.0 as *mut GcRef) = GcRef(0) };
        // f: unrooted nursery object with a registered finalizer.
        let f = gc.alloc_with_type(tid, ptr_size);
        assert!(gc.is_in_nursery(f.0));
        unsafe { *(f.0 as *mut GcRef) = q };
        GcAllocator::register_finalizer(&mut gc, 0, f, trigger);
        assert_eq!(gc.oldgen.object_count(), 1);

        gc.do_collect_oldgen_nonmoving();

        assert_eq!(
            gc.oldgen.object_count(),
            1,
            "the old child of a young finalizer object must survive"
        );
        assert!(gc.is_in_nursery(f.0), "nursery object must not move/free");
        let f_hdr = unsafe { header_of(f.0) };
        assert!(
            unsafe { !(*f_hdr).has_flag(GcFlags::GCFLAG_VISITED) },
            "no greyed nursery object retains VISITED"
        );

        // The next minor keeps f and it still names q.
        gc.do_collect_nursery();
        let f = gc
            .old_objects_with_finalizers
            .iter()
            .map(|&(addr, _)| addr)
            .next()
            .expect("the minor moves the young finalizer to the old list");
        assert_eq!(unsafe { *(f as *const GcRef) }, q);
    }

    /// An explicit non-moving collection must use roots observed at the call,
    /// not merely finish an incremental cycle whose root snapshot predates a
    /// release.  This is the non-moving counterpart of `do_collect_full`'s
    /// initial `gc_step_until_scanning_with_minors` followed by a fresh cycle.
    #[test]
    fn nonmoving_major_starts_fresh_after_finishing_an_in_progress_cycle() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let object = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + 16);
        let mut root = object;
        unsafe { gc.roots.add(&mut root) };

        // Seed the in-progress cycle while `object` is live, then release it.
        gc.major_collection_step();
        assert_ne!(gc.gc_state, GcState::Scanning);
        gc.roots.clear();

        gc.do_collect_oldgen_nonmoving();

        assert_eq!(
            gc.oldgen.object_count(),
            0,
            "the explicit collection's fresh root snapshot must sweep object"
        );
    }

    /// A non-moving major must run `invalidate_old_weakrefs` (reads the
    /// target's VISITED) BEFORE the nursery-VISITED clear, so an old weakref
    /// whose target is a live nursery object is kept, not spuriously nulled.
    #[test]
    fn nonmoving_major_keeps_live_nursery_weakref() {
        let mut gc = test_gc(4096);
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        // n: nursery-resident weakref target, kept live by a root.
        let n = gc.alloc_with_type(target_tid, 16);
        assert!(gc.is_in_nursery(n.0));
        // w: weakref born directly in the old generation, pointing at n.
        // `alloc_in_oldgen` records born-old weakrefs onto
        // `old_objects_with_weakrefs`, so no manual registration is needed.
        let w = gc.alloc_in_oldgen_clear(wref_tid, GcHeader::SIZE + crate::weakref::SIZEOF_WEAKREF);
        unsafe { *((w.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = n };

        let mut w_root = w;
        let mut n_root = n;
        unsafe {
            gc.roots.add(&mut w_root);
            gc.roots.add(&mut n_root);
        }

        gc.do_collect_oldgen_nonmoving();

        // n was marked-through (live), so the weakref slot is NOT nulled:
        // invalidate_old_weakrefs read n's VISITED before the clear pass ran.
        let after =
            unsafe { crate::weakref::ll_weakref_deref(w_root.0 as *const crate::weakref::Weakref) };
        assert_eq!(
            after.0, n_root.0,
            "live nursery weakref target must survive"
        );
        assert_eq!(gc.old_objects_with_weakrefs.len(), 1);
        assert!(gc.is_in_nursery(n_root.0));
        let n_hdr = unsafe { header_of(n_root.0) };
        assert!(unsafe { !(*n_hdr).has_flag(GcFlags::GCFLAG_VISITED) });

        gc.roots.clear();
    }

    #[test]
    fn nonmoving_major_invalidates_live_nursery_weakref_to_dead_old_target() {
        let mut gc = test_gc(4096);
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        let target = gc.alloc_in_oldgen_clear(target_tid, GcHeader::SIZE + 16);
        let wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        unsafe {
            *((wref.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = target;
        }
        let mut wref_root = wref;
        unsafe { gc.roots.add(&mut wref_root) };

        gc.do_collect_oldgen_nonmoving();

        let after = unsafe {
            crate::weakref::ll_weakref_deref(wref_root.0 as *const crate::weakref::Weakref)
        };
        assert_eq!(after.0, 0);
        assert_eq!(gc.young_objects_with_weakrefs, vec![wref.0]);
        gc.roots.clear();
    }

    #[test]
    fn nonmoving_major_keeps_live_nursery_weakref_to_live_old_target() {
        let mut gc = test_gc(4096);
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        let target = gc.alloc_in_oldgen_clear(target_tid, GcHeader::SIZE + 16);
        let wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        unsafe {
            *((wref.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = target;
        }
        let mut target_root = target;
        let mut wref_root = wref;
        unsafe {
            gc.roots.add(&mut target_root);
            gc.roots.add(&mut wref_root);
        }

        gc.do_collect_oldgen_nonmoving();

        let after = unsafe {
            crate::weakref::ll_weakref_deref(wref_root.0 as *const crate::weakref::Weakref)
        };
        assert_eq!(after.0, target_root.0);
        assert_eq!(gc.young_objects_with_weakrefs, vec![wref.0]);
        gc.roots.clear();
    }

    /// incminimark.py:3112 "weakref itself not marked" branch.
    /// Dropping the wref root before the major cycle drops the entry
    /// silently — no slot mutation, no panic.
    #[test]
    fn test_major_invalidate_old_weakref_self_dies() {
        let mut gc = test_gc(4096);
        let target_tid = gc.register_type(TypeInfo::simple(16));
        let wref_tid = gc.register_type(TypeInfo::weakref());

        let target = gc.alloc_with_type(target_tid, 16);
        let wref = gc.alloc_with_type(wref_tid, crate::weakref::SIZEOF_WEAKREF);
        unsafe {
            *((wref.0 + crate::weakref::WEAKPTR_OFFSET) as *mut GcRef) = target;
        }

        let mut target_root = target;
        let mut wref_root = wref;
        unsafe {
            gc.roots.add(&mut target_root);
            gc.roots.add(&mut wref_root);
        }
        gc.do_collect_nursery();
        assert_eq!(gc.old_objects_with_weakrefs.len(), 1);

        gc.roots.remove(&mut wref_root);
        gc.do_collect_full();

        assert!(gc.old_objects_with_weakrefs.is_empty());
        gc.roots.clear();
    }

    #[test]
    fn finalizer_queue_keeps_dead_object_until_it_is_popped() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static TRIGGERS: AtomicUsize = AtomicUsize::new(0);
        fn trigger() {
            TRIGGERS.fetch_add(1, Ordering::Relaxed);
        }

        TRIGGERS.store(0, Ordering::Relaxed);
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + 16);
        let mut root = obj;
        unsafe { gc.roots.add(&mut root) };
        GcAllocator::register_finalizer(&mut gc, 0, obj, trigger);

        // A live registered object stays on the registration deque.
        gc.do_collect_oldgen_nonmoving();
        assert!(GcAllocator::finalizer_next_dead(&mut gc, 0).is_none());
        assert!(gc.oldgen.contains(obj.0));

        // Once unreachable it moves to the death queue and is kept alive for
        // app-level `__del__` execution.
        gc.roots.remove(&mut root);
        gc.do_collect_oldgen_nonmoving();
        assert_eq!(TRIGGERS.load(Ordering::Relaxed), 1);
        assert!(gc.oldgen.contains(obj.0));
        assert_eq!(GcAllocator::finalizer_next_dead(&mut gc, 0), Some(obj));

        // Popping removes the queue root; the following major reclaims it.
        gc.do_collect_oldgen_nonmoving();
        // ArenaCollection::contains is intentionally an arena-range query and
        // can stay true for a freed block while the current arena is retained.
        // The allocator's exact live count verifies reclamation here.
        assert_eq!(gc.oldgen.object_count(), 0);
    }

    /// After `do_collect_oldgen_nonmoving` with a young finalizer object, no
    /// death-queue or `old_objects_with_finalizers` entry is a nursery address.
    /// incminimark only runs `deal_with_objects_with_finalizers` over old
    /// objects after a minor emptied the nursery; the non-moving pass leaves
    /// young registrations for `deal_with_young_objects_with_finalizers`.
    #[test]
    fn nonmoving_major_leaves_young_finalizer_objects_off_the_old_deques() {
        fn trigger() {}

        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(tid, 16);
        assert!(gc.is_in_nursery(obj.0));
        GcAllocator::register_finalizer(&mut gc, 0, obj, trigger);

        // A young leftover already on the old deque, as a pinned identity copy
        // from `deal_with_young_objects_with_finalizers` would leave.
        let leftover = gc.alloc_with_type(tid, 16);
        assert!(gc.is_in_nursery(leftover.0));
        GcAllocator::register_finalizer(&mut gc, 0, leftover, trigger);
        gc.probably_young_objects_with_finalizers
            .retain(|&(addr, _)| addr != leftover.0);
        gc.old_objects_with_finalizers
            .retain(|&(addr, _)| addr != leftover.0);
        gc.old_objects_with_finalizers.push_back((leftover.0, 0));

        gc.do_collect_oldgen_nonmoving();

        assert!(
            gc.finalizer_handlers[0]
                .deque
                .iter()
                .all(|&addr| !gc.is_in_nursery(addr)),
            "death queue must not hold a nursery address"
        );
        assert!(
            gc.run_old_style_finalizers
                .iter()
                .all(|&addr| !gc.is_in_nursery(addr)),
            "old-style death queue must not hold a nursery address"
        );
        assert!(
            gc.old_objects_with_finalizers
                .iter()
                .all(|&(addr, _)| !gc.is_in_nursery(addr)),
            "old_objects_with_finalizers must not hold a nursery address"
        );
        assert!(
            gc.probably_young_objects_with_finalizers
                .iter()
                .any(|&(addr, _)| addr == obj.0),
            "a young registration stays on probably_young"
        );
        assert!(
            gc.probably_young_objects_with_finalizers
                .iter()
                .any(|&(addr, _)| addr == leftover.0),
            "a young leftover is reparked onto probably_young"
        );
        assert!(gc.is_in_nursery(obj.0));
        assert!(gc.is_in_nursery(leftover.0));
    }

    /// incminimark.py `malloc_fixedsize(..., is_finalizer_light=False)` and
    /// GCBase.execute_finalizers: an old-style finalizer is born old, runs only
    /// after finalization ordering, and leaves the resurrectable object alive
    /// until a later major cycle.
    #[test]
    fn old_style_finalizer_is_born_old_and_runs_from_its_distinct_queue() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static RUNS: AtomicUsize = AtomicUsize::new(0);
        unsafe fn finalizer(_obj_addr: usize) {
            RUNS.fetch_add(1, Ordering::Relaxed);
        }

        RUNS.store(0, Ordering::Relaxed);
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_old_style_finalizer(16, finalizer));
        let obj = gc.alloc_with_type(tid, 16);

        assert!(!gc.is_in_nursery(obj.0));
        assert_eq!(
            gc.old_objects_with_finalizers,
            VecDeque::from([(obj.0, -1)])
        );
        assert!(gc.young_objects_with_destructors.is_empty());

        gc.do_collect_oldgen_nonmoving();
        assert_eq!(RUNS.load(Ordering::Relaxed), 1);
        assert!(gc.run_old_style_finalizers.is_empty());
        assert_eq!(gc.oldgen.object_count(), 1);

        gc.do_collect_oldgen_nonmoving();
        assert_eq!(RUNS.load(Ordering::Relaxed), 1);
        assert_eq!(gc.oldgen.object_count(), 0);
    }

    #[test]
    fn finalizer_subgraph_prefilter_follows_container_edges() {
        fn trigger() {}

        let mut gc = test_gc(4096);
        let leaf_tid = gc.register_type(TypeInfo::simple(16));
        let holder_tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));
        let unrelated = gc.alloc_in_oldgen_clear(leaf_tid, GcHeader::SIZE + 16);
        let finalizable = gc.alloc_in_oldgen_clear(leaf_tid, GcHeader::SIZE + 16);
        let holder =
            gc.alloc_in_oldgen_clear(holder_tid, GcHeader::SIZE + std::mem::size_of::<GcRef>());
        unsafe { *(holder.0 as *mut GcRef) = finalizable };
        GcAllocator::register_finalizer(&mut gc, 0, finalizable, trigger);

        assert!(!gc.do_subgraph_has_pending_finalizer(&[unrelated]));
        assert!(gc.do_subgraph_has_pending_finalizer(&[holder]));
        gc.rawrefcount_init(rrc_test_trigger);
        assert!(
            gc.rawrefcount_enabled(),
            "rrc must not make an unrelated subgraph look finalizable"
        );
        assert!(!gc.do_subgraph_has_pending_finalizer(&[unrelated]));
        assert!(gc.do_subgraph_has_pending_finalizer(&[holder]));
        // The inspector visited bit is scratch state, not a semantic mark.
        for obj in [unrelated, finalizable, holder] {
            assert!(unsafe { !(*header_of(obj.0)).has_flag(GcFlags::GCFLAG_EXTRA) });
        }
    }

    /// incminimark.py:2492,2570-2573,3011-3020: objects marked solely to
    /// make a finalizer runnable remain allocated after the sweep, but their
    /// bytes are not survivor growth for the next-major threshold.  Counting
    /// them made a workload of mostly-finalizable garbage raise its threshold
    /// forever (the upstream issue #2590 invariant).
    #[test]
    fn finalizer_kept_graph_is_excluded_from_next_major_threshold() {
        fn trigger() {}

        let mut gc = test_gc(4096);
        let ptr_size = std::mem::size_of::<GcRef>();
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let allocation_size = OldGen::allocation_size(GcHeader::SIZE + ptr_size);

        // Remove the production threshold floor and growth cap so this test
        // observes the survivor baseline itself.  One ordinary rooted object
        // remains real live data; the unreachable finalizer object and its
        // child survive only for finalization ordering.
        gc.min_heap_size = 0.0;
        gc.next_major_collection_initial = 1_000_000_000.0;
        gc.next_major_collection_threshold = 1_000_000_000.0;
        gc.max_delta = f64::MAX;

        let live = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        let child = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        let finalizable = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        unsafe {
            *(live.0 as *mut GcRef) = GcRef::NULL;
            *(child.0 as *mut GcRef) = GcRef::NULL;
            *(finalizable.0 as *mut GcRef) = child;
        }
        let mut live_root = live;
        unsafe { gc.roots.add(&mut live_root) };
        GcAllocator::register_finalizer(&mut gc, 0, finalizable, trigger);

        gc.do_collect_oldgen_nonmoving();

        assert_eq!(gc.kept_alive_by_finalizer, 2 * allocation_size);
        assert_eq!(gc.get_total_memory_used(), 3 * allocation_size);
        let expected = allocation_size as f64 * gc.major_collection_threshold;
        assert_eq!(gc.next_major_collection_threshold, expected);
        assert_eq!(
            GcAllocator::finalizer_next_dead(&mut gc, 0),
            Some(finalizable)
        );
    }

    /// `do_collect_oldgen_nonmoving` is a pyre adaptation: unlike upstream's
    /// major seam it can order a finalizer graph while nursery objects still
    /// exist. Nursery bytes are absent from `get_total_memory_used`, so they
    /// must also be absent from the amount subtracted from that metric.
    #[test]
    fn finalizer_threshold_accounting_excludes_live_nursery_children() {
        fn trigger() {}

        let mut gc = test_gc(4096);
        let ptr_size = std::mem::size_of::<GcRef>();
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let allocation_size = OldGen::allocation_size(GcHeader::SIZE + ptr_size);
        gc.min_heap_size = 0.0;
        gc.next_major_collection_initial = 1_000_000_000.0;
        gc.next_major_collection_threshold = 1_000_000_000.0;
        gc.max_delta = f64::MAX;

        let live = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        let finalizable = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        let young_child = gc.alloc_with_type(tid, ptr_size);
        assert!(gc.is_in_nursery(young_child.0));
        unsafe {
            *(live.0 as *mut GcRef) = GcRef::NULL;
            *(young_child.0 as *mut GcRef) = GcRef::NULL;
            *(finalizable.0 as *mut GcRef) = young_child;
        }
        gc.do_write_barrier(finalizable);
        let mut live_root = live;
        unsafe { gc.roots.add(&mut live_root) };
        GcAllocator::register_finalizer(&mut gc, 0, finalizable, trigger);

        gc.do_collect_oldgen_nonmoving();

        assert_eq!(gc.kept_alive_by_finalizer, allocation_size);
        assert_eq!(gc.get_total_memory_used(), 2 * allocation_size);
        let expected = allocation_size as f64 * gc.major_collection_threshold;
        assert_eq!(gc.next_major_collection_threshold, expected);
        assert!(gc.is_in_nursery(young_child.0));
    }

    /// `rgc.py` contracts `register_finalizer` to run at most once per
    /// object. Registering twice used to leave two deque entries: the first
    /// major delivers the object and re-appends the survivor through
    /// `new_with_finalizer` (incminimark.py), so the next major
    /// delivers it again and the app-level `__del__` runs a second time.
    #[test]
    fn register_finalizer_delivers_an_object_once_however_often_it_is_registered() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static TRIGGERS: AtomicUsize = AtomicUsize::new(0);
        fn trigger() {
            TRIGGERS.fetch_add(1, Ordering::Relaxed);
        }

        TRIGGERS.store(0, Ordering::Relaxed);
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + 16);
        let mut root = obj;
        unsafe { gc.roots.add(&mut root) };

        // The construction site registers, then a later out-of-band caller
        // asks again without knowing that.
        GcAllocator::register_finalizer(&mut gc, 0, obj, trigger);
        GcAllocator::register_finalizer(&mut gc, 0, obj, trigger);
        assert_eq!(gc.old_objects_with_finalizers.len(), 1);

        gc.roots.remove(&mut root);
        gc.do_collect_oldgen_nonmoving();
        assert_eq!(TRIGGERS.load(Ordering::Relaxed), 1);
        assert_eq!(GcAllocator::finalizer_next_dead(&mut gc, 0), Some(obj));
        assert_eq!(GcAllocator::finalizer_next_dead(&mut gc, 0), None);

        // No survivor was carried into the next cycle, so the object is not
        // delivered a second time.
        gc.do_collect_oldgen_nonmoving();
        assert_eq!(TRIGGERS.load(Ordering::Relaxed), 1);
        assert_eq!(GcAllocator::finalizer_next_dead(&mut gc, 0), None);
    }

    /// incminimark.py `ignore_finalizer`: the object stays on the queue, and
    /// `deal_with_objects_with_finalizers` is what passes over it.
    #[test]
    fn ignore_finalizer_withholds_a_registered_object_from_its_queue() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static TRIGGERS: AtomicUsize = AtomicUsize::new(0);
        fn trigger() {
            TRIGGERS.fetch_add(1, Ordering::Relaxed);
        }

        TRIGGERS.store(0, Ordering::Relaxed);
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + 16);
        let mut root = obj;
        unsafe { gc.roots.add(&mut root) };
        GcAllocator::register_finalizer(&mut gc, 0, obj, trigger);
        assert_eq!(gc.old_objects_with_finalizers.len(), 1);

        GcAllocator::ignore_finalizer(&mut gc, obj);
        assert!(unsafe { (*header_of(obj.0)).has_flag(GcFlags::GCFLAG_IGNORE_FINALIZER) });

        gc.roots.remove(&mut root);
        gc.do_collect_oldgen_nonmoving();
        assert_eq!(TRIGGERS.load(Ordering::Relaxed), 0);
        assert_eq!(GcAllocator::finalizer_next_dead(&mut gc, 0), None);
    }

    /// A non-moving major releases an unreachable young raw-malloced object
    /// whose finalizer is ignored. Its entry has to leave the young deque with
    /// it: the next minor would otherwise file whatever is born at that
    /// address as a finalizer object.
    #[test]
    fn nonmoving_major_drops_an_ignored_young_rawmalloc_finalizer_entry() {
        fn trigger() {}

        let mut gc = test_gc(4096);
        gc.next_major_collection_threshold = 1_000_000_000.0;
        let tid = gc.register_type(TypeInfo::simple(16));
        let large = gc.config.large_object_threshold + 64;

        let dead = gc.alloc_with_type(tid, large);
        let mut kept = gc.alloc_with_type(tid, large);
        unsafe { gc.roots.add(&mut kept) };
        assert!(gc.is_young_rawmalloced(dead.0));
        assert!(gc.is_young_rawmalloced(kept.0));
        GcAllocator::register_finalizer(&mut gc, 0, dead, trigger);
        GcAllocator::register_finalizer(&mut gc, 0, kept, trigger);
        assert_eq!(gc.probably_young_objects_with_finalizers.len(), 2);
        GcAllocator::ignore_finalizer(&mut gc, dead);

        gc.do_collect_oldgen_nonmoving();

        assert!(!gc.is_young_rawmalloced(dead.0));
        let queued: Vec<usize> = gc
            .probably_young_objects_with_finalizers
            .iter()
            .map(|&(addr, _)| addr)
            .collect();
        assert_eq!(queued, vec![kept.0]);
        assert!(gc.old_objects_with_finalizers.is_empty());
        gc.roots.clear();
    }

    /// An address the collector does not own has no header to stamp, so the
    /// hint is dropped rather than written into whatever precedes it.
    #[test]
    fn ignore_finalizer_drops_an_unmanaged_address() {
        let mut gc = test_gc(4096);
        let outside = Box::into_raw(Box::new([0usize; 4])) as usize;
        GcAllocator::ignore_finalizer(&mut gc, GcRef(outside));
        GcAllocator::ignore_finalizer(&mut gc, GcRef::NULL);
        unsafe { drop(Box::from_raw(outside as *mut [usize; 4])) };
    }

    /// incminimark.py `writebarrier_before_copy`: with no cards involved, a
    /// source that has already lost TRACK_YOUNG_PTRS may hold young pointers,
    /// so the destination joins the remembered set.
    #[test]
    fn writebarrier_before_copy_remembers_dest_when_source_is_untracked() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let source = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        let dest = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        unsafe { (*header_of(source.0)).clear_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) };
        let before = gc.old_objects_pointing_to_young.len();

        assert!(gc.writebarrier_before_copy(source.0, dest.0, 0, 0, 1));

        assert_eq!(gc.old_objects_pointing_to_young.len(), before + 1);
        assert!(gc.old_objects_pointing_to_young.contains(&dest.0));
        assert!(unsafe { !(*header_of(dest.0)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
    }

    /// A source that still carries TRACK_YOUNG_PTRS has no young pointers to
    /// hand over, so outside the marking phase the copy needs nothing.
    #[test]
    fn writebarrier_before_copy_is_a_noop_for_a_tracked_source() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let source = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        let dest = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        assert_eq!(gc.gc_state, GcState::Scanning);
        let before = gc.old_objects_pointing_to_young.len();

        assert!(gc.writebarrier_before_copy(source.0, dest.0, 0, 0, 1));

        assert_eq!(gc.old_objects_pointing_to_young.len(), before);
        assert!(unsafe { (*header_of(dest.0)).has_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) });
    }

    /// A destination already on the remembered set is the fast path: nothing
    /// below it runs, and the answer is still "handled".
    #[test]
    fn writebarrier_before_copy_fast_paths_an_untracked_dest() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let source = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        let dest = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        unsafe { (*header_of(source.0)).clear_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) };
        unsafe { (*header_of(dest.0)).clear_flag(GcFlags::GCFLAG_TRACK_YOUNG_PTRS) };
        let before = gc.old_objects_pointing_to_young.len();

        assert!(gc.writebarrier_before_copy(source.0, dest.0, 0, 0, 1));

        assert_eq!(gc.old_objects_pointing_to_young.len(), before);
    }

    /// The card arms: a source with cards and a destination without cannot be
    /// served in bulk, and the caller is told to do it per item.
    #[test]
    fn writebarrier_before_copy_declines_a_card_source_with_a_plain_dest() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        let source = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        let dest = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + ptr_size);
        unsafe { (*header_of(source.0)).set_flag(GcFlags::GCFLAG_HAS_CARDS) };
        assert!(gc.config.card_page_indices > 0);

        assert!(!gc.writebarrier_before_copy(source.0, dest.0, 0, 0, 1));
    }

    /// Two card arrays whose copy is misaligned: the card index of an item
    /// differs between them, so the bits cannot be carried across.
    #[test]
    fn writebarrier_before_copy_declines_a_misaligned_card_copy() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(8, 8, 0, true, vec![]));
        let source = gc.alloc_in_oldgen_with_cards(
            tid,
            GcHeader::SIZE + 8 + 8 * CARD_ARRAY_LEN,
            CARD_ARRAY_LEN,
            true,
        );
        let dest = gc.alloc_in_oldgen_with_cards(
            tid,
            GcHeader::SIZE + 8 + 8 * CARD_ARRAY_LEN,
            CARD_ARRAY_LEN,
            true,
        );
        unsafe { (*header_of(source.0)).set_flag(GcFlags::GCFLAG_CARDS_SET) };

        assert!(!gc.writebarrier_before_copy(source.0, dest.0, 1, 0, 64));
    }

    /// `manually_copy_card_bits`: an aligned copy between two card arrays
    /// carries the source's marks onto the destination and enters it on
    /// `old_objects_with_cards_set`.
    #[test]
    fn writebarrier_before_copy_copies_card_bits_across() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(8, 8, 0, true, vec![]));
        let source = gc.alloc_in_oldgen_with_cards(
            tid,
            GcHeader::SIZE + 8 + 8 * CARD_ARRAY_LEN,
            CARD_ARRAY_LEN,
            true,
        );
        let dest = gc.alloc_in_oldgen_with_cards(
            tid,
            GcHeader::SIZE + 8 + 8 * CARD_ARRAY_LEN,
            CARD_ARRAY_LEN,
            true,
        );
        assert!(unsafe { (*header_of(source.0)).has_flag(GcFlags::GCFLAG_HAS_CARDS) });
        assert!(unsafe { (*header_of(dest.0)).has_flag(GcFlags::GCFLAG_HAS_CARDS) });
        unsafe { *MiniMarkGC::get_card_ptr(source.0, 0) = 0b0000_0101 };
        unsafe { (*header_of(source.0)).set_flag(GcFlags::GCFLAG_CARDS_SET) };
        gc.old_objects_with_cards_set.clear();

        assert!(gc.writebarrier_before_copy(source.0, dest.0, 0, 0, 64));

        assert_eq!(unsafe { *MiniMarkGC::get_card_ptr(dest.0, 0) }, 0b0000_0101);
        assert!(unsafe { (*header_of(dest.0)).has_flag(GcFlags::GCFLAG_CARDS_SET) });
        assert_eq!(gc.old_objects_with_cards_set, vec![dest.0]);
    }

    /// incminimark.py `writebarrier_before_move`: the card bits stay put while
    /// the items move past them, so the per-card record is replaced by the
    /// whole-object one.
    #[test]
    fn writebarrier_before_move_generalizes_a_marked_card_array() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(8, 8, 0, true, vec![]));
        let array = gc.alloc_in_oldgen_with_cards(
            tid,
            GcHeader::SIZE + 8 + 8 * CARD_ARRAY_LEN,
            CARD_ARRAY_LEN,
            true,
        );
        unsafe { (*header_of(array.0)).set_flag(GcFlags::GCFLAG_CARDS_SET) };
        let before = gc.old_objects_pointing_to_young.len();

        gc.writebarrier_before_move(array.0);

        assert_eq!(gc.old_objects_pointing_to_young.len(), before + 1);
        assert!(gc.old_objects_pointing_to_young.contains(&array.0));
    }

    /// Type id 0 names a real type -- the first one registered -- so an
    /// allocation that "has no type" is recorded as that one and traced with
    /// its layout.  The JIT's varsize helpers used to do exactly this.
    #[test]
    fn type_id_zero_is_the_first_registered_type_not_an_absent_one() {
        let mut gc = test_gc(4096);
        let first = gc.register_type(TypeInfo::with_gc_ptrs(24, vec![0]));
        assert_eq!(first, 0, "the first registration owns id 0");
        let tid = gc.register_type(TypeInfo::varsize(8, 8, 0, true, vec![]));

        let untyped = gc.alloc_varsize(8, 8, 4);
        let typed = gc.alloc_varsize_typed(tid, 8, 8, 4);

        assert_eq!(unsafe { (*header_of(untyped.0)).type_id() }, first);
        assert_eq!(unsafe { (*header_of(typed.0)).type_id() }, tid);
        // The untyped array now claims a fixed-size type (`item_size == 0`,
        // so no item is ever walked) that declares a reference at offset 0 --
        // which in the array is its own length field.
        assert_eq!(gc.types.get(first).item_size, 0);
        assert_eq!(gc.types.get(first).gc_ptr_offsets, vec![0]);
        assert_ne!(gc.types.get(tid).item_size, 0);
    }

    /// The host reaches this barrier as a `GcAllocator`, never as a
    /// `MiniMarkGC`, so the forwarding impl is what production actually runs.
    /// The trait's own default is a no-op, which would silently drop the call.
    #[test]
    fn the_trait_entry_reaches_the_collectors_before_move_barrier() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(8, 8, 0, true, vec![]));
        let array = gc.alloc_in_oldgen_with_cards(
            tid,
            GcHeader::SIZE + 8 + 8 * CARD_ARRAY_LEN,
            CARD_ARRAY_LEN,
            true,
        );
        unsafe { (*header_of(array.0)).set_flag(GcFlags::GCFLAG_CARDS_SET) };
        let before = gc.old_objects_pointing_to_young.len();

        let dynamic: &mut dyn crate::GcAllocator = &mut gc;
        dynamic.writebarrier_before_move(array);

        assert_eq!(gc.old_objects_pointing_to_young.len(), before + 1);
        assert!(gc.old_objects_pointing_to_young.contains(&array.0));
    }

    /// The four-argument arraymove residual recovers its array token from the
    /// same TYPE_INFO row the collector uses to walk the variable part.
    #[test]
    fn varsize_layout_reports_registered_array_shape() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(24, 16, 8, true, vec![]));
        let array = gc.alloc_varsize_typed(tid, 24, 16, 3);

        let dynamic: &dyn crate::GcAllocator = &gc;
        assert_eq!(
            dynamic.varsize_layout(array),
            Some(crate::GcVarSizeLayout {
                base_size: 24,
                item_size: 16,
                items_have_gc_ptrs: true,
            })
        );
    }

    /// An array with no card marked has nothing the move could invalidate.
    #[test]
    fn writebarrier_before_move_is_a_noop_without_cards_set() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(8, 8, 0, true, vec![]));
        let array = gc.alloc_in_oldgen_with_cards(
            tid,
            GcHeader::SIZE + 8 + 8 * CARD_ARRAY_LEN,
            CARD_ARRAY_LEN,
            true,
        );
        let before = gc.old_objects_pointing_to_young.len();

        gc.writebarrier_before_move(array.0);

        assert_eq!(gc.old_objects_pointing_to_young.len(), before);
    }

    /// incminimark.py `move_out_of_nursery`: "called twice, it should return
    /// the same shadow object, and not creating another shadow object.  As a
    /// safety feature, when called on a non-nursery object, do nothing."
    #[test]
    fn move_out_of_nursery_fills_one_shadow_and_the_minor_keeps_it() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_with_type(tid, 16);
        assert!(gc.is_in_nursery(obj.0));
        unsafe { *(obj.0 as *mut u64) = 0xfeed_face };

        let shadow = gc.move_out_of_nursery(obj.0);
        assert_ne!(shadow, obj.0);
        assert!(!gc.is_in_nursery(shadow));
        assert!(unsafe { (*header_of(obj.0)).has_flag(GcFlags::GCFLAG_SHADOW_INITIALIZED) });
        assert_eq!(unsafe { *(shadow as *const u64) }, 0xfeed_face);
        assert_eq!(gc.move_out_of_nursery(obj.0), shadow);

        // The nursery original may be written after the shadow was filled;
        // upstream's contract ("Only use for immutable objects!") is what makes
        // the promotion's skipped copy sound, so the shadow keeps what it had.
        unsafe { *(obj.0 as *mut u64) = 0xdead_beef };
        let mut root = obj;
        unsafe { gc.roots.add(&mut root) };
        gc.do_collect_nursery();
        assert_eq!(root.0, shadow);
        assert_eq!(unsafe { *(shadow as *const u64) }, 0xfeed_face);
        gc.roots.clear();
    }

    /// An old-gen object is already where a shadow would put it.
    #[test]
    fn move_out_of_nursery_answers_a_non_nursery_object_unchanged() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + 16);
        assert_eq!(gc.move_out_of_nursery(obj.0), obj.0);
        assert!(unsafe { !(*header_of(obj.0)).has_flag(GcFlags::GCFLAG_SHADOW_INITIALIZED) });
    }

    /// `external_malloc` uses `force_enabled=False`, so a disabled collector
    /// still minors and then returns without starting a major.
    #[test]
    fn maybe_collect_for_external_malloc_respects_disable() {
        let mut gc = test_gc(4096);
        gc.next_major_collection_threshold = 0.0;
        gc.disable();
        let minors = gc.minor_collections;
        let majors = gc.major_collections;

        assert!(gc.threshold_reached(16));
        assert!(!gc.do_maybe_collect_for_external_malloc(16));

        assert_eq!(gc.minor_collections, minors + 1);
        assert_eq!(gc.major_collections, majors);
        assert_eq!(gc.gc_state, GcState::Scanning);
        assert!(!gc.isenabled());
    }

    /// incminimark.py `collect(gen)`: a negative generation is the minor that
    /// deliberately makes no major progress.
    #[test]
    fn do_collect_with_a_negative_generation_runs_a_minor_only() {
        let mut gc = test_gc(4096);
        let minors = gc.minor_collections;
        let majors = gc.major_collections;

        gc.do_collect(-1);

        assert_eq!(gc.minor_collections, minors + 1);
        assert_eq!(gc.major_collections, majors);
        assert_eq!(gc.gc_state, GcState::Scanning);
    }

    /// `gen == 1` "is like gen == 0, but if no major collection is running,
    /// then it forces one to start now".
    #[test]
    fn do_collect_with_generation_one_starts_a_major_cycle() {
        let mut gc = test_gc(4096);
        assert_eq!(gc.gc_state, GcState::Scanning);
        let minors = gc.minor_collections;

        gc.do_collect(1);

        assert_eq!(gc.minor_collections, minors + 1);
        assert_ne!(
            gc.gc_state,
            GcState::Scanning,
            "gen == 1 forces a cycle to start"
        );
    }

    /// `gen >= 2` is the complete collection, and it leaves no cycle running.
    #[test]
    fn do_collect_with_generation_two_completes_a_cycle() {
        let mut gc = test_gc(4096);
        let majors = gc.major_collections;

        gc.do_collect(2);

        assert!(gc.major_collections > majors);
        assert_eq!(gc.gc_state, GcState::Scanning);
    }

    /// incminimark.py `set_max_heap_size`: a limit below the pending
    /// thresholds pulls them down with it.
    #[test]
    fn set_max_heap_size_pulls_the_pending_thresholds_down() {
        let mut gc = test_gc(4096);
        gc.next_major_collection_initial = 8192.0;
        gc.next_major_collection_threshold = 8192.0;

        gc.set_max_heap_size(1024);

        assert_eq!(gc.max_heap_size, 1024.0);
        assert_eq!(gc.next_major_collection_initial, 1024.0);
        assert_eq!(gc.next_major_collection_threshold, 1024.0);

        // A larger limit does not raise a threshold that is already lower.
        gc.set_max_heap_size(65536);
        assert_eq!(gc.max_heap_size, 65536.0);
        assert_eq!(gc.next_major_collection_threshold, 1024.0);

        // Unbounded leaves them alone entirely.
        gc.set_max_heap_size(0);
        assert_eq!(gc.max_heap_size, 0.0);
        assert_eq!(gc.next_major_collection_threshold, 1024.0);
    }

    // ── rawrefcount (incminimark.py:3157-3409) ──────────────────────

    /// A mirror block lives outside the GC heap at a fixed address, which is
    /// the whole point of the link.  Leaked deliberately: the collector holds
    /// the address in its lists for as long as the test's collector lives.
    fn test_mirror(refcnt: isize) -> usize {
        Box::into_raw(Box::new(rawrefcount::PyObjHeader {
            ob_refcnt: refcnt,
            ob_link: 0,
        })) as usize
    }

    fn mirror_refcnt(mirror: usize) -> isize {
        unsafe { (*rawrefcount::pyobj(mirror)).ob_refcnt }
    }

    fn mirror_link(mirror: usize) -> usize {
        unsafe { (*rawrefcount::pyobj(mirror)).ob_link }
    }

    thread_local! {
        /// Per-thread so the count belongs to one test: the trigger is a bare
        /// `fn`, and a `static` would be shared with every test the harness
        /// runs concurrently.  The collector calls it on the calling thread.
        static RRC_TRIGGER_FIRED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        static RRC_LIGHT_FREED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    fn rrc_test_trigger() {
        RRC_TRIGGER_FIRED.with(|fired| fired.set(fired.get() + 1));
    }

    fn rrc_test_light_free(mirror: usize) {
        unsafe { drop(Box::from_raw(rawrefcount::pyobj(mirror))) };
        RRC_LIGHT_FREED.with(|freed| freed.set(freed.get() + 1));
    }

    fn rrc_test_gc() -> MiniMarkGC {
        let mut gc = test_gc(1 << 16);
        gc.rawrefcount_init(rrc_test_trigger);
        gc.rawrefcount_set_light_free(rrc_test_light_free);
        assert!(gc.rawrefcount_enabled());
        gc
    }

    /// incminimark.py _rrc_minor_trace and :3284-3318: a count above the link share is
    /// a reference the C side holds, and it keeps the linked object alive
    /// across a minor with no interpreter root at all.  The link then follows
    /// the object to its new address, and the identity table is re-keyed there.
    #[test]
    fn a_c_referenced_mirror_roots_its_object_across_a_minor() {
        let mut gc = rrc_test_gc();
        let tid = gc.register_type(TypeInfo::object(64));
        let object = gc.alloc_with_type(tid, 64);
        assert!(gc.is_in_nursery(object.0), "premise: the object is young");

        let mirror = test_mirror(rawrefcount::REFCNT_FROM_PYPY + 1);
        gc.rawrefcount_create_link_pypy(object.0, mirror);
        assert_eq!(gc.rawrefcount_from_obj(object.0), mirror);

        gc.do_collect_nursery();

        let moved = mirror_link(mirror);
        assert_ne!(moved, 0, "the mirror was unlinked");
        assert_ne!(moved, object.0, "premise: a surviving nursery object moves");
        assert!(!gc.is_in_nursery(moved));
        assert_eq!(gc.rawrefcount_from_obj(moved), mirror);
        assert_eq!(gc.rawrefcount_to_obj(mirror), moved);
        assert_eq!(gc.rawrefcount_next_dead(), 0, "nothing died");
    }

    /// incminimark.py:3264-3265 and :3320-3354: a count of exactly the link
    /// share means nothing but the link references the mirror, so the object
    /// may die.  The mirror is then unlinked, queued, and left at 1 so the
    /// drain's own release is what frees it.
    #[test]
    fn a_mirror_at_the_link_share_lets_its_object_die_and_is_queued() {
        let mut gc = rrc_test_gc();
        let tid = gc.register_type(TypeInfo::object(64));
        let object = gc.alloc_with_type(tid, 64);
        let mirror = test_mirror(rawrefcount::REFCNT_FROM_PYPY);
        gc.rawrefcount_create_link_pypy(object.0, mirror);

        RRC_TRIGGER_FIRED.with(|fired| fired.set(0));
        gc.do_collect_nursery();

        assert_eq!(mirror_link(mirror), 0, "a dead mirror keeps no link");
        assert_eq!(
            mirror_refcnt(mirror),
            1,
            "incminimark.py:3352 leaves a queued mirror at 1"
        );
        assert_eq!(gc.rawrefcount_next_dead(), mirror);
        assert_eq!(gc.rawrefcount_next_dead(), 0, "the queue drains once");
        assert_eq!(
            RRC_TRIGGER_FIRED.with(|fired| fired.get()),
            1,
            "incminimark.py:3248-3250 schedules the drain for a non-empty queue"
        );
    }

    /// incminimark.py `rrc_invoke_callback` watches only `rrc_dealloc_pending`.
    #[test]
    fn rrc_invoke_callback_ignores_finalize_and_c_garbage() {
        let mut gc = rrc_test_gc();
        gc.rrc.finalize_pending.push_back(0x100);
        gc.rrc.c_garbage = true;
        RRC_TRIGGER_FIRED.with(|fired| fired.set(0));

        gc.rrc_invoke_callback();

        assert_eq!(RRC_TRIGGER_FIRED.with(|fired| fired.get()), 0);
        assert!(
            gc.rrc.c_garbage,
            "the dealloc-only callback does not clear c_garbage"
        );
        assert_eq!(gc.rrc.finalize_pending.len(), 1);
    }

    /// A collection that queued only `tp_finalize` still owes `drain_dead`.
    #[test]
    fn rrc_invoke_cpyext_drain_fires_for_finalize_pending() {
        let mut gc = rrc_test_gc();
        gc.rrc.finalize_pending.push_back(0x100);
        RRC_TRIGGER_FIRED.with(|fired| fired.set(0));

        gc.rrc_invoke_cpyext_drain();

        assert_eq!(RRC_TRIGGER_FIRED.with(|fired| fired.get()), 1);
        assert_eq!(
            gc.rawrefcount_next_finalize(),
            0x100,
            "the sibling only schedules; the embedder drains"
        );
    }

    /// A C-only cycle sets `c_garbage` with an empty dealloc queue.
    #[test]
    fn rrc_invoke_cpyext_drain_fires_for_c_garbage() {
        let mut gc = rrc_test_gc();
        gc.rrc.c_garbage = true;
        RRC_TRIGGER_FIRED.with(|fired| fired.set(0));

        gc.rrc_invoke_cpyext_drain();

        assert_eq!(RRC_TRIGGER_FIRED.with(|fired| fired.get()), 1);
        assert!(
            !gc.rrc.c_garbage,
            "the sibling consumes the flag when it schedules"
        );
    }

    /// Both functions fire independently when both kinds of work are queued.
    #[test]
    fn rrc_invoke_pair_fires_once_each_when_both_queues_are_full() {
        let mut gc = rrc_test_gc();
        gc.rrc.dealloc_pending.push(0x10);
        gc.rrc.finalize_pending.push_back(0x20);
        RRC_TRIGGER_FIRED.with(|fired| fired.set(0));

        gc.rrc_invoke_callback();
        gc.rrc_invoke_cpyext_drain();

        assert_eq!(RRC_TRIGGER_FIRED.with(|fired| fired.get()), 2);
        assert_eq!(gc.rawrefcount_next_dead(), 0x10);
        assert_eq!(gc.rawrefcount_next_finalize(), 0x20);
    }

    /// incminimark.py `_rrc_free`: LIGHT means the mirror needs no C
    /// deallocator. With external references remaining, subtract the light
    /// share and unlink it without putting it on the deallocation queue.
    #[test]
    fn a_light_mirror_is_unlinked_without_deallocator_queueing() {
        let mut gc = rrc_test_gc();
        let tid = gc.register_type(TypeInfo::object(64));
        let object = gc.alloc_with_type(tid, 64);
        let mirror = test_mirror(rawrefcount::REFCNT_FROM_PYPY_LIGHT + 2);
        gc.rawrefcount_create_link_pyobj(object.0, mirror);

        gc.do_collect_nursery();

        assert_eq!(mirror_link(mirror), 0);
        assert_eq!(mirror_refcnt(mirror), 2);
        assert_eq!(gc.rawrefcount_next_dead(), 0);
    }

    /// `incminimark.py _rrc_free`: an exact LIGHT share needs no deallocator
    /// queue and returns the raw mirror through the embedder's allocator.
    #[test]
    fn an_exact_light_mirror_is_freed_immediately() {
        let mut gc = rrc_test_gc();
        let tid = gc.register_type(TypeInfo::object(64));
        let object = gc.alloc_with_type(tid, 64);
        let mirror = test_mirror(rawrefcount::REFCNT_FROM_PYPY_LIGHT);
        gc.rawrefcount_create_link_pyobj(object.0, mirror);

        RRC_LIGHT_FREED.with(|freed| freed.set(0));
        gc.do_collect_nursery();

        assert_eq!(RRC_LIGHT_FREED.with(|freed| freed.get()), 1);
        assert_eq!(gc.rawrefcount_next_dead(), 0);
    }

    /// incminimark.py:3359-3375 and :3397-3409: the same two outcomes, decided
    /// by a major collection over the old list.
    #[test]
    fn a_major_frees_only_the_mirrors_whose_object_died() {
        let mut gc = rrc_test_gc();
        let tid = gc.register_type(TypeInfo::object(64));

        let mut kept = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + 64);
        let dying = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + 64);
        assert!(!gc.is_in_nursery(kept.0) && !gc.is_in_nursery(dying.0));

        let kept_mirror = test_mirror(rawrefcount::REFCNT_FROM_PYPY + 1);
        let dying_mirror = test_mirror(rawrefcount::REFCNT_FROM_PYPY);
        gc.rawrefcount_create_link_pypy(kept.0, kept_mirror);
        gc.rawrefcount_create_link_pypy(dying.0, dying_mirror);
        unsafe { gc.roots.add(&mut kept) };

        gc.do_collect_full();

        assert_eq!(mirror_link(kept_mirror), kept.0);
        assert_eq!(gc.rawrefcount_from_obj(kept.0), kept_mirror);
        assert_eq!(mirror_link(dying_mirror), 0);
        assert_eq!(mirror_refcnt(dying_mirror), 1);
        assert_eq!(gc.rawrefcount_next_dead(), dying_mirror);
        assert_eq!(gc.rawrefcount_next_dead(), 0);

        gc.roots.remove(&mut kept);
    }

    /// A pinned object survives a minor without moving, so there is no
    /// forwarding pointer for its mirror's link to follow — and reading only
    /// the forwarding bit, as incminimark.py:3287-3299 does, would report it
    /// dead and free a mirror the C side still holds.
    #[test]
    fn a_pinned_object_keeps_its_mirror_across_a_minor() {
        let mut gc = rrc_test_gc();
        // `pin` refuses anything carrying GC pointers, so this is the shape a
        // pinnable object has: a leaf byte block.
        let tid = gc.register_type(TypeInfo::simple(64));
        let mut object = gc.alloc_with_type(tid, 64);
        assert!(gc.pin(object), "premise: this object can be pinned");
        // Pinning does not root: something must still reach it this collection.
        unsafe { gc.roots.add(&mut object) };

        let mirror = test_mirror(rawrefcount::REFCNT_FROM_PYPY + 1);
        gc.rawrefcount_create_link_pypy(object.0, mirror);

        gc.do_collect_nursery();

        assert_eq!(
            mirror_link(mirror),
            object.0,
            "a pinned object does not move, so the link must not change"
        );
        assert!(gc.is_in_nursery(mirror_link(mirror)));
        assert_eq!(gc.rawrefcount_from_obj(object.0), mirror);
        assert_eq!(gc.rawrefcount_next_dead(), 0, "the mirror was freed");

        // Still tracked as young, so the next minor finds it again.
        gc.do_collect_nursery();
        assert_eq!(gc.rawrefcount_from_obj(object.0), mirror);
        assert_eq!(gc.rawrefcount_next_dead(), 0);

        gc.roots.remove(&mut object);
    }

    /// incminimark.py rawrefcount_mark_deallocating: the deallocating marker replaces the link, so
    /// a re-entrant lookup during a deallocator cannot hand back an address the
    /// collector has already reclaimed.
    #[test]
    fn mark_deallocating_replaces_the_link_with_the_sentinel() {
        let mut gc = rrc_test_gc();
        let tid = gc.register_type(TypeInfo::object(64));
        let object = gc.alloc_in_oldgen_clear(tid, GcHeader::SIZE + 64);
        let mirror = test_mirror(rawrefcount::REFCNT_FROM_PYPY + 1);
        gc.rawrefcount_create_link_pypy(object.0, mirror);

        let marker = 0x0DEA_DFFF_usize;
        gc.rawrefcount_mark_deallocating(marker, mirror);
        assert_eq!(gc.rawrefcount_to_obj(mirror), marker);
    }

    /// A non-moving major runs no leading minor, so the young P list is the
    /// only thing that can report a C-referenced nursery object — and an old
    /// object reachable only through one would otherwise be swept.
    ///
    /// Differential, because "is this address still live?" has no sound direct
    /// oracle once the arena has been swept: the same heap is built twice and
    /// differs only in whether the mirror carries a C reference, so the
    /// surviving bytes are attributable to nothing else.
    #[test]
    fn a_nonmoving_major_keeps_an_old_object_reachable_only_through_a_c_root() {
        fn survivors_with(mirror_refcnt: isize) -> usize {
            let word = std::mem::size_of::<usize>();
            let mut gc = rrc_test_gc();
            let leaf = gc.register_type(TypeInfo::object(word));
            let holder = gc.register_type(TypeInfo::object_with_gc_ptrs(word, vec![0]));

            let old = gc.alloc_in_oldgen_clear(leaf, GcHeader::SIZE + word);
            let young = gc.alloc_with_type(holder, word);
            assert!(gc.is_in_nursery(young.0) && !gc.is_in_nursery(old.0));
            // The old object is reachable from the nursery object and from
            // nowhere else; the nursery object is reachable from the mirror
            // and from nowhere else.
            unsafe { *(young.0 as *mut GcRef) = old };
            gc.rawrefcount_create_link_pypy(young.0, test_mirror(mirror_refcnt));

            gc.do_collect_oldgen_nonmoving();
            assert_eq!(gc.rawrefcount_next_dead(), 0, "the nursery is not swept");
            gc.get_total_memory_used()
        }

        let referenced = survivors_with(rawrefcount::REFCNT_FROM_PYPY + 1);
        let unreferenced = survivors_with(rawrefcount::REFCNT_FROM_PYPY);
        assert!(
            referenced > unreferenced,
            "the old object was swept out from under a live C reference \
             ({referenced} bytes held with a C reference, {unreferenced} without)"
        );
    }
}

/// Per-walker tally of what one major cycle re-seeded, behind
/// `MAJIT_GC_ROOT_CENSUS`.
///
/// `collect_roots` costs one `seed_major_root` per root, and each of those
/// asks `is_managed_heap_object` — so the root count, not the live-object
/// count, is what a major on a small heap spends its time on. The census
/// names the walker responsible so a count far above the live-object count
/// can be traced to the table that registers it.
struct RootCensus {
    on: bool,
    /// Label, roots seen, and how many of them were null. Linear: there are a
    /// handful of labels and the walk appends to none of them.
    rows: Vec<(&'static str, usize, usize)>,
}

impl RootCensus {
    fn new() -> Self {
        Self {
            on: std::env::var_os("MAJIT_GC_ROOT_CENSUS").is_some(),
            rows: Vec::new(),
        }
    }

    fn note(&mut self, site: &'static str, gcref: GcRef) {
        if !self.on {
            return;
        }
        let null = usize::from(gcref.is_null());
        match self.rows.iter_mut().find(|(label, _, _)| *label == site) {
            Some(row) => {
                row.1 += 1;
                row.2 += null;
            }
            None => self.rows.push((site, 1, null)),
        }
    }

    fn report(&mut self, major: usize, prebuilt: usize) {
        if !self.on {
            return;
        }
        self.rows.sort_by(|a, b| b.1.cmp(&a.1));
        let total: usize = self.rows.iter().map(|(_, seen, _)| seen).sum();
        let mut line = format!("[jit-gc-diag] major={major} roots={total} prebuilt={prebuilt}");
        for (label, seen, null) in &self.rows {
            line.push_str(&format!(" {label}={seen}/{null}null"));
        }
        eprintln!("{line}");
    }
}
