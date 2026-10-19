//! `rpython/rlib/debug.py` parity — PYPYLOG-style debug scope and print
//! API shared across the metainterp, optimizer, and backends.
//!
//! PyPy structures runtime tracing through `debug_start(category)` /
//! `debug_stop(category)` brackets with intervening `debug_print(...)`
//! lines; the wire format is
//!
//! ```text
//! [<ts>] {<category>
//! <messages>
//! [<ts>] <category>}
//! ```
//!
//! Tooling (`rpython/tool/logparser.py`) parses this format directly. Pyre
//! emits the same wire shape so log captures cross-tool with PyPy when
//! `MAJIT_LOG` is set (pyre's `PYPYLOG` analog).
//!
//! Single-event sites — the common case in Pyre's metainterp/optimizeopt/
//! backends — use [`log_one`], which opens a single-line section,
//! emits the body, and closes the section in one call. Multi-message
//! pairs use [`scope`] (RAII) wrapping repeated [`debug_print`] calls.
//!
//! # Known divergences from `rpython/rlib/debug.py` / PYPYLOG
//!
//! 1. **Closed.** `MAJIT_LOG` is the category-prefix filter from
//!    `debug_print.c` `pypy_debug_open` / `startswithoneof` /
//!    `pypy_debug_start` / `pypy_have_debug_prints`. Accepted syntax:
//!    - `1` — every category, nested `debug_print` included.
//!    - `:file`, or an empty prefix list — every category. The filename
//!      is parsed and dropped.
//!    - `cat`, `cat1,cat2`, or `cat1,cat2:file` — `debug_start`'s category
//!      is emitted when it starts with one of the prefixes. The filename
//!      is parsed and dropped.
//!    - `+file` — `debug_profile`: `{cat` / `cat}` brackets for every
//!      category, ready bit left clear so nested `debug_print` stays off.
//!
//!    `have_debug_prints` is the current thread's ready bit.
//!    `debug_start` shifts it and sets the low bit when the category is
//!    accepted; `debug_stop` shifts it back. A prefix list starts the bit
//!    at 0 so direct print sites stay quiet outside an accepted section.
//!    `1`, an empty prefix list, and `debug_profile` start it at -1, which
//!    is `pypy_have_debug_prints`'s initial value (top-level prints on;
//!    `debug_profile` clears the low bit inside a section).
//!
//!    A colon-less value other than `1` is a prefix list. Upstream
//!    `pypy_debug_open` treats that shape as `debug_profile` plus a
//!    filename; the filename sink does not exist here, and
//!    `MAJIT_LOG=jit-summary` has to name a category. Leading `+` is what
//!    selects `debug_profile` (`+fname` in `pypy_debug_open`).
//!
//! 2. **No translated/untranslated split.** RPython's `_log_capture`
//!    versus `_log` distinction lets untranslated tests assert against
//!    captured sections without writing to stderr. Pyre always writes to
//!    `stderr` via `eprintln!` — tests that need to assert on log output
//!    redirect `stderr` at the OS level.
//!
//! 3. **Strict `debug_stop` nesting.** RPython's `DebugLog.debug_stop`
//!    raises on mismatch; Pyre [`debug_stop`] panics with the same intent.
//!    This is intentional, but it does mean a mid-stack panic propagates
//!    through any `debug_start`/`debug_stop` pair that was already opened
//!    (the RAII [`scope`] guard absorbs this by closing in `Drop`; bare
//!    `debug_start`/`debug_stop` callers must use `try/finally`-equivalent
//!    unwind discipline). The category stack is updated for sections the
//!    filter does not print, so a filtered `debug_stop` still pairs.
//!
//! 4. **Always stderr, no `:filename` sink.** PyPy's `PYPYLOG=…:my.log`
//!    redirects to a file; pyre parses the `:filename` portion and drops
//!    it. External tools that consume PYPYLOG-formatted output can capture
//!    pyre's `stderr` directly — the wire format is identical.
//!
//! The ready word is thread-local. `pypy_have_debug_prints` is one C
//! global; a second thread's `debug_start` must not clear this thread's
//! `eprintln!` gates.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

/// Parsed `MAJIT_LOG` value. `debug_print.c` `pypy_debug_open`.
#[derive(Clone, Debug)]
struct DebugLogConfig {
    /// Unset or empty: logging is off, including top-level prints.
    enabled: bool,
    /// `debug_print.c` `debug_profile`: brackets for every category, body off.
    debug_profile: bool,
    /// `None` accepts every category (`1`, `:file`, empty prefix list).
    /// `Some` is the comma-separated prefix list.
    prefixes: Option<Vec<String>>,
}

/// `debug_print.c` `pypy_debug_open` filter parse, with the `:filename`
/// sink dropped and a colon-less value read as prefixes (see divergence 1).
fn parse_majit_log(value: Option<&str>) -> DebugLogConfig {
    let Some(raw) = value else {
        return disabled_log();
    };
    if raw.is_empty() {
        // `pypy_debug_open`: an empty value is the same as unset.
        return disabled_log();
    }
    // Leading '+' forces `debug_profile` and ignores any colon; the rest
    // is a filename and is dropped.
    if let Some(_filename) = raw.strip_prefix('+') {
        return DebugLogConfig {
            enabled: true,
            debug_profile: true,
            prefixes: None,
        };
    }
    let prefix_part = match raw.split_once(':') {
        Some((prefix, _file)) => prefix,
        None if raw == "1" => {
            return accept_all_log();
        }
        None => raw,
    };
    if prefix_part.is_empty() || prefix_part == "1" {
        return accept_all_log();
    }
    DebugLogConfig {
        enabled: true,
        debug_profile: false,
        prefixes: Some(prefix_part.split(',').map(str::to_string).collect()),
    }
}

fn disabled_log() -> DebugLogConfig {
    DebugLogConfig {
        enabled: false,
        debug_profile: false,
        prefixes: None,
    }
}

fn accept_all_log() -> DebugLogConfig {
    DebugLogConfig {
        enabled: true,
        debug_profile: false,
        prefixes: None,
    }
}

fn global_config() -> &'static DebugLogConfig {
    static CONFIG: OnceLock<DebugLogConfig> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let raw = std::env::var_os("MAJIT_LOG").map(|value| value.to_string_lossy().into_owned());
        parse_majit_log(raw.as_deref())
    })
}

fn with_config<R>(f: impl FnOnce(&DebugLogConfig) -> R) -> R {
    #[cfg(test)]
    {
        let overridden = CONFIG_OVERRIDE.with(|slot| slot.borrow().clone());
        if let Some(cfg) = overridden {
            return f(&cfg);
        }
    }
    f(global_config())
}

/// `debug_print.c` `startswithoneof`: `category` starts with one comma-separated
/// prefix. An empty prefix matches every category.
fn category_starts_with_one_of(category: &str, prefixes: &[String]) -> bool {
    prefixes
        .iter()
        .any(|prefix| category.starts_with(prefix.as_str()))
}

/// `debug_print.c` `oneofstartswith`: one prefix starts with `query`.
/// An empty query does not match; the C loop never treats `""` as a hit.
fn one_prefix_starts_with(prefixes: &[String], query: &str) -> bool {
    if query.is_empty() {
        return false;
    }
    prefixes.iter().any(|prefix| prefix.starts_with(query))
}

/// Whether this `debug_start` category sets the ready bit.
/// `debug_profile` never does: brackets are printed, nested prints stay off.
fn section_accepted(cfg: &DebugLogConfig, category: &str) -> bool {
    if !cfg.enabled || cfg.debug_profile {
        return false;
    }
    match &cfg.prefixes {
        None => true,
        Some(prefixes) => category_starts_with_one_of(category, prefixes),
    }
}

/// `pypy_have_debug_prints` starts at -1. A prefix list starts at 0 so
/// `have_debug_prints` is false until `debug_start` accepts a category —
/// direct `eprintln!` sites are not nested in a section, and a filter
/// must silence them.
fn initial_ready(cfg: &DebugLogConfig) -> i64 {
    if cfg.enabled && cfg.prefixes.is_none() {
        -1
    } else {
        0
    }
}

struct ThreadDebugState {
    /// `debug_print.c` `pypy_have_debug_prints`, per thread.
    ready: i64,
    inited: bool,
    stack: Vec<&'static str>,
}

thread_local! {
    static STATE: RefCell<ThreadDebugState> = const { RefCell::new(ThreadDebugState {
        ready: 0,
        inited: false,
        stack: Vec::new(),
    }) };
    #[cfg(test)]
    static CONFIG_OVERRIDE: RefCell<Option<DebugLogConfig>> = const { RefCell::new(None) };
}

fn with_debug_state<R>(f: impl FnOnce(&mut ThreadDebugState, &DebugLogConfig) -> R) -> R {
    with_config(|cfg| {
        STATE.with(|cell| {
            let mut state = cell.borrow_mut();
            if !state.inited {
                state.ready = initial_ready(cfg);
                state.inited = true;
            }
            f(&mut state, cfg)
        })
    })
}

/// Whether `MAJIT_LOG` currently allows a `debug_print` on this thread.
///
/// Same answer as [`have_debug_prints`]: the ready bit `debug_start` sets
/// and `debug_stop` restores. Direct `eprintln!` sites call this so a
/// prefix filter silences them without each site naming a category.
///
/// `debug_print.h` `OP_HAVE_DEBUG_PRINTS(r)` inlines one load of
/// `pypy_have_debug_prints`.
#[inline(always)]
pub fn majit_log_enabled() -> bool {
    have_debug_prints()
}

/// Wall-clock origin used as the PyPy `read_timestamp()` analog.
fn ts_origin() -> Instant {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    *ORIGIN.get_or_init(Instant::now)
}

/// `rlib/rtimer.py read_timestamp()` analog — monotonic nanosecond
/// counter rendered as PyPy's hex `[ts]` prefix.
fn read_timestamp() -> u128 {
    ts_origin().elapsed().as_nanos()
}

/// `debug_print.c` `pypy_have_debug_prints` is one C `long`.
/// `OP_HAVE_DEBUG_PRINTS(r) r = (pypy_have_debug_prints & 1)` is one load.
///
/// 0 = uninit, 1 = `MAJIT_LOG` unset (`pypy_have_debug_prints` stays 0),
/// 2 = logging on (the ready bit lives in TLS so a second thread's
/// `debug_start` does not clear this thread's gates).
static HAVE_DEBUG_PRINTS_FLAG: AtomicU8 = AtomicU8::new(0);

/// `rlib/debug.py have_debug_prints()` — true when the next
/// `debug_print` would be emitted. That is the low bit of this thread's
/// ready word (`debug_print.c` `OP_HAVE_DEBUG_PRINTS`).
///
/// `debug_print.h` `OP_HAVE_DEBUG_PRINTS(r) r = (pypy_have_debug_prints & 1)`.
#[inline(always)]
pub fn have_debug_prints() -> bool {
    #[cfg(test)]
    {
        if CONFIG_OVERRIDE.with(|slot| slot.borrow().is_some()) {
            return with_debug_state(|state, _| (state.ready & 1) != 0);
        }
    }
    match HAVE_DEBUG_PRINTS_FLAG.load(Ordering::Relaxed) {
        1 => false,
        2 => with_debug_state(|state, _| (state.ready & 1) != 0),
        _ => have_debug_prints_init(),
    }
}

#[cold]
#[inline(never)]
fn have_debug_prints_init() -> bool {
    let enabled = global_config().enabled;
    HAVE_DEBUG_PRINTS_FLAG.store(if enabled { 2 } else { 1 }, Ordering::Relaxed);
    if !enabled {
        return false;
    }
    with_debug_state(|state, _| (state.ready & 1) != 0)
}

/// `rlib/debug.py have_debug_prints_for(prefix)` — true when the
/// configured prefix list overlaps `prefix`.
///
/// `debug_print.c` `pypy_have_debug_prints_for`: either some configured
/// prefix starts with `prefix`, or `prefix` starts with some configured
/// prefix. `debug_profile` and a disabled log are false. An accept-all
/// log (`1`, empty prefix) is true. This reads the filter, not the
/// open-section stack.
pub fn have_debug_prints_for(prefix: &str) -> bool {
    with_config(|cfg| {
        if !cfg.enabled || cfg.debug_profile {
            return false;
        }
        match &cfg.prefixes {
            None => true,
            Some(prefixes) => {
                one_prefix_starts_with(prefixes, prefix)
                    || category_starts_with_one_of(prefix, prefixes)
            }
        }
    })
}

/// `rlib/debug.py debug_start(category)` — open a logging section.
///
/// `debug_print.c` `pypy_debug_start`: shift the ready word, set the low
/// bit when the category is accepted, and emit `[<ts>] {<category>` for
/// an accepted category or for every category in `debug_profile`. A
/// rejected category shifts the word without setting the bit and emits
/// nothing. The category is still pushed so [`debug_stop`] can pair.
pub fn debug_start(category: &'static str) {
    let print = with_debug_state(|state, cfg| {
        if !cfg.enabled {
            return false;
        }
        state.ready <<= 1;
        let accepted = section_accepted(cfg, category);
        if accepted {
            state.ready |= 1;
        }
        state.stack.push(category);
        cfg.debug_profile || accepted
    });
    if print {
        eprintln!("[{:x}] {{{}", read_timestamp(), category);
    }
}

/// `rlib/debug.py debug_stop(category)` — close the matching section
/// opened by [`debug_start`].
///
/// `debug_print.c` `pypy_debug_stop` prints `[<ts>] <category>}` when
/// `debug_profile` is set or the ready bit is set, then shifts the ready
/// word back. Mismatched stops panic, mirroring `DebugLog.debug_stop`.
pub fn debug_stop(category: &'static str) {
    let print = with_debug_state(|state, cfg| {
        if !cfg.enabled {
            return false;
        }
        match state.stack.last() {
            Some(top) if *top == category => {
                state.stack.pop();
            }
            Some(top) => panic!(
                "debug_stop({category:?}) does not match the most recent debug_start({top:?})"
            ),
            None => panic!("debug_stop({category:?}) with no matching debug_start"),
        }
        let print = cfg.debug_profile || (state.ready & 1) != 0;
        state.ready >>= 1;
        print
    });
    if print {
        eprintln!("[{:x}] {}}}", read_timestamp(), category);
    }
}

/// `rlib/debug.py debug_print(*args)` — emit a single line inside
/// the currently-open section. No-op when [`have_debug_prints`] is false.
/// Pyre callers format the message themselves and pass the result here.
pub fn debug_print(msg: &str) {
    if !have_debug_prints() {
        return;
    }
    eprintln!("{msg}");
}

/// RAII scope guard returned by [`scope`]: panics still drop through
/// `Drop` so `debug_stop` always pairs with the opening `debug_start`.
#[must_use = "drop the guard to fire the matching debug_stop"]
pub struct DebugScope {
    category: &'static str,
}

impl Drop for DebugScope {
    fn drop(&mut self) {
        debug_stop(self.category);
    }
}

/// Convenience: open a `debug_start(category)` scope returning a guard
/// that fires the matching `debug_stop` on drop. Mirrors PyPy's
/// typical `debug_start … try: … finally: debug_stop` pattern.
pub fn scope(category: &'static str) -> DebugScope {
    debug_start(category);
    DebugScope { category }
}

/// Emit a single body line wrapped in a `debug_start`/`debug_stop`
/// section of the given category. Equivalent to PyPy's common pattern
/// of `debug_start`, `debug_print`, `debug_stop`.
///
/// The category is opened before the body is gated, so a filter that
/// accepts `category` still emits the section when the caller is outside
/// any section. Callers that format a heavy `msg` can skip that work
/// with [`have_debug_prints_for`] when the filter cannot accept it.
pub fn log_one(category: &'static str, msg: &str) {
    debug_start(category);
    debug_print(msg);
    debug_stop(category);
}

#[cfg(test)]
#[must_use = "dropping the guard clears the per-thread log override"]
struct TestLogGuard;

#[cfg(test)]
impl Drop for TestLogGuard {
    fn drop(&mut self) {
        CONFIG_OVERRIDE.with(|slot| *slot.borrow_mut() = None);
        STATE.with(|cell| {
            *cell.borrow_mut() = ThreadDebugState {
                ready: 0,
                inited: false,
                stack: Vec::new(),
            };
        });
    }
}

#[cfg(test)]
fn set_log_spec_for_test(value: Option<&str>) -> TestLogGuard {
    let cfg = parse_majit_log(value);
    CONFIG_OVERRIDE.with(|slot| *slot.borrow_mut() = Some(cfg.clone()));
    STATE.with(|cell| {
        *cell.borrow_mut() = ThreadDebugState {
            ready: initial_ready(&cfg),
            inited: true,
            stack: Vec::new(),
        };
    });
    TestLogGuard
}

#[cfg(test)]
mod tests {
    use super::{
        debug_start, debug_stop, have_debug_prints, have_debug_prints_for, parse_majit_log,
        set_log_spec_for_test,
    };

    #[test]
    fn filter_jit_summary_accepts_summary_and_rejects_tracing() {
        let _guard = set_log_spec_for_test(Some("jit-summary"));
        assert!(have_debug_prints_for("jit-summary"));
        assert!(!have_debug_prints_for("jit-tracing"));
        assert!(!have_debug_prints());

        debug_start("jit-summary");
        assert!(have_debug_prints());
        debug_stop("jit-summary");
        assert!(!have_debug_prints());

        debug_start("jit-tracing");
        assert!(!have_debug_prints());
        debug_stop("jit-tracing");
        assert!(!have_debug_prints());

        // Prefix match, not equality: `jit-sum` accepts `jit-summary`.
        drop(_guard);
        let _guard = set_log_spec_for_test(Some("jit-sum"));
        debug_start("jit-summary");
        assert!(have_debug_prints());
        debug_stop("jit-summary");
        debug_start("jit-tracing");
        assert!(!have_debug_prints());
        debug_stop("jit-tracing");
    }

    #[test]
    fn value_one_accepts_all_categories() {
        let _guard = set_log_spec_for_test(Some("1"));
        assert!(have_debug_prints());
        assert!(have_debug_prints_for("jit-summary"));
        assert!(have_debug_prints_for("jit-tracing"));
        debug_start("jit-tracing");
        assert!(have_debug_prints());
        debug_stop("jit-tracing");
        assert!(have_debug_prints());

        let all = parse_majit_log(Some("1"));
        assert!(all.enabled);
        assert!(!all.debug_profile);
        assert!(all.prefixes.is_none());
        let from_file = parse_majit_log(Some("1:ignored"));
        assert!(from_file.prefixes.is_none());
        assert!(!from_file.debug_profile);

        let empty_prefix = parse_majit_log(Some(":file"));
        assert!(empty_prefix.enabled);
        assert!(empty_prefix.prefixes.is_none());
        assert!(!empty_prefix.debug_profile);
        drop(_guard);
        let _guard = set_log_spec_for_test(Some(":file"));
        assert!(have_debug_prints());
        debug_start("jit-tracing");
        assert!(have_debug_prints());
        debug_stop("jit-tracing");
    }

    #[test]
    fn comma_prefixes_with_file_sink_parse_to_two_prefixes() {
        let cfg = parse_majit_log(Some("a,b:file"));
        assert!(cfg.enabled);
        assert!(!cfg.debug_profile);
        assert_eq!(
            cfg.prefixes.as_deref(),
            Some(["a".to_string(), "b".to_string()].as_slice())
        );
        assert!(parse_majit_log(None).prefixes.is_none());
        assert!(!parse_majit_log(None).enabled);
        assert!(!parse_majit_log(Some("")).enabled);

        let _guard = set_log_spec_for_test(Some("a,b:file"));
        assert!(have_debug_prints_for("a"));
        assert!(have_debug_prints_for("b"));
        assert!(!have_debug_prints_for("file"));
        debug_start("a-cat");
        assert!(have_debug_prints());
        debug_stop("a-cat");
        debug_start("zzz");
        assert!(!have_debug_prints());
        debug_stop("zzz");
    }

    #[test]
    fn nested_start_stop_restores_ready_state() {
        // `myc,cat2` accepts `mycat` and `cat2`, rejects `other`.
        // The child ready bit must not stick, and a rejected parent must
        // not hide an accepted child (`pypy_debug_start`'s shift).
        let _guard = set_log_spec_for_test(Some("myc,cat2"));
        assert!(!have_debug_prints());

        debug_start("mycat");
        assert!(have_debug_prints());
        debug_start("other");
        assert!(!have_debug_prints());
        debug_start("cat2");
        assert!(have_debug_prints());
        debug_stop("cat2");
        assert!(!have_debug_prints());
        debug_stop("other");
        assert!(have_debug_prints());
        debug_stop("mycat");
        assert!(!have_debug_prints());

        // Accepted child inside a rejected parent, then both restored.
        debug_start("other");
        assert!(!have_debug_prints());
        debug_start("cat2");
        assert!(have_debug_prints());
        debug_stop("cat2");
        assert!(!have_debug_prints());
        debug_stop("other");
        assert!(!have_debug_prints());
    }

    #[test]
    fn profile_mode_keeps_brackets_and_clears_the_ready_bit() {
        let cfg = parse_majit_log(Some("+out.log"));
        assert!(cfg.enabled);
        assert!(cfg.debug_profile);
        assert!(cfg.prefixes.is_none());
        assert!(parse_majit_log(Some("+a:b")).debug_profile);

        let _guard = set_log_spec_for_test(Some("+out.log"));
        assert!(have_debug_prints());
        assert!(!have_debug_prints_for("jit-summary"));
        debug_start("jit-tracing");
        assert!(!have_debug_prints());
        debug_start("jit-summary");
        assert!(!have_debug_prints());
        debug_stop("jit-summary");
        assert!(!have_debug_prints());
        debug_stop("jit-tracing");
        assert!(have_debug_prints());
    }
}
