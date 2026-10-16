//! **Heap allocations per compiled-code entry** — the instrument for majit's
//! warm entry path.
//!
//! ## Why this exists, and why it is not a `#[test]` in an ordinary file
//!
//! Reading the entry path statically has mispredicted this number twice. The
//! only way to know what a warm entry costs is to count what the allocator is
//! asked for while one is happening, so this file counts it.
//!
//! Four properties are load-bearing and each one is a separate failure mode:
//!
//! 1. **The driver must outlive the measured calls.** A fixture that builds its
//!    `JitDriver` inside the annotated function measures a driver that never
//!    gets warm: every call re-traces from zero and the figure it reports is
//!    the cost of *compiling*, not of *entering*. The fixture below takes
//!    `&mut JitDriver` as a parameter, so `main` binds one driver and calls the
//!    interpreter many times against it — the bind-once/call-many regime the
//!    steady-state cost lives in.
//!
//! 2. **The window must open at the compiled entry.** `MetaInterp`'s
//!    `on_compiled_entry` hook fires at the one point in `back_edge_internal`
//!    where every decline has already returned and the compiled body runs on
//!    the next statement, so one call of the hook is exactly one entry. The
//!    meter reads the allocation counter *in* the hook, which makes the
//!    entry-to-entry delta a measurement of one whole round trip (assemble the
//!    entry arguments, run the artifact, decode the exit, restore the state,
//!    come back round to the next merge point) without needing a second probe
//!    point in production code.
//!
//! 3. **libtest's thread pool races a process-global counter.** This target is
//!    `harness = false` (see `Cargo.toml`), so there is no thread pool and no
//!    second test running beside this one; and the counter that is *reported*
//!    is thread-local, so an allocation made off-thread cannot land in it. A
//!    second process-global counter is kept purely so the difference can be
//!    reported as `off-thread` — evidence that the isolation did something,
//!    rather than an assertion that it did.
//!
//! 4. **A GC must be installed.** The entry path allocates its jitframe from
//!    the nursery when there is a GC to allocate it from and out of the Rust
//!    heap when there is not, and the two differ by a whole allocation per
//!    entry. A fixture with no GC measures a frame shape that no shipped
//!    configuration uses. [`install_gc`] puts the meter on the configuration
//!    pyre and cel run.
//!
//! ## Per-site attribution
//!
//! The counter alone gives a number, not a diagnosis. For attribution the
//! allocator captures a backtrace at each allocation made inside an armed
//! window and buckets it by the innermost majit frame. Capturing a backtrace
//! itself allocates, so a re-entrancy flag routes those straight to `System`
//! uncounted and unattributed — without it the allocator recurses into itself.
//!
//! Attribution runs in its own short window, separate from the counting one:
//! symbolizing a backtrace per allocation is orders of magnitude slower than
//! the entry it describes, and a counted window that slow would be measuring
//! the profiler.
//!
//! ## Running it
//!
//! ```text
//! cargo test -p majit-metainterp --features dynasm --test allocs_per_compiled_entry
//! cargo test -p majit-metainterp --features cranelift --test allocs_per_compiled_entry
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use majit_metainterp::JitDriver;

// the meter

std::thread_local! {
    /// Allocations made by THIS thread. `const`-initialised and holding a
    /// destructor-free `Cell`, so the allocator can touch it without recursing
    /// through lazy TLS setup or registering a TLS destructor.
    static LOCAL_ALLOCS: Cell<u64> = const { Cell::new(0) };
    /// Set while the allocator is capturing a backtrace. Allocations made by
    /// the capture itself must not be counted, attributed, or recursed into.
    static IN_PROBE: Cell<bool> = const { Cell::new(false) };
    /// `innermost majit frame -> allocations charged to it`, filled only while
    /// [`ATTRIBUTING`] is set.
    static SITES: RefCell<BTreeMap<String, u64>> = RefCell::new(BTreeMap::new());
}

/// Allocations made by the whole process. Only ever read as `global - local`,
/// which is the count that could NOT have been the subject.
static GLOBAL_ALLOCS: AtomicU64 = AtomicU64::new(0);

/// Whether the allocator should symbolize and bucket. Off for the counted
/// window, on for the (much shorter) attribution window.
static ATTRIBUTING: AtomicBool = AtomicBool::new(false);

struct Counting;

#[inline]
fn bump() {
    GLOBAL_ALLOCS.fetch_add(1, Ordering::Relaxed);
    // `try_with`, not `with`: an allocation during thread teardown must not
    // turn into a panic inside the allocator.
    let _ = LOCAL_ALLOCS.try_with(|c| c.set(c.get() + 1));
    if ATTRIBUTING.load(Ordering::Relaxed) {
        attribute();
    }
}

/// Charge this allocation to the innermost majit frame on the stack.
#[inline(never)]
fn attribute() {
    let already = IN_PROBE.try_with(|c| c.replace(true)).unwrap_or(true);
    if already {
        return;
    }
    let site = innermost_majit_frame();
    let _ = SITES.try_with(|sites| {
        if let Ok(mut sites) = sites.try_borrow_mut() {
            *sites.entry(site).or_insert(0) += 1;
        }
    });
    let _ = IN_PROBE.try_with(|c| c.set(false));
}

/// The innermost frame that is neither the meter nor allocator plumbing, as a
/// `symbol (file:line)` string.
///
/// Keyed on what to SKIP rather than on a `majit` prefix. A prefix filter reads
/// as more precise and is in fact blinder: it buckets everything it does not
/// recognise — a `std` collection growing inside a backend, a frame whose
/// symbol the unwinder could not name — into one anonymous pile, and that pile
/// was the largest row in the first run of this file. Skipping the containers
/// instead names every site or says plainly that it could not.
fn innermost_majit_frame() -> String {
    let text = std::backtrace::Backtrace::force_capture().to_string();
    let mut pending: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("at ") {
            if let Some(symbol) = pending.take() {
                return format!("{symbol} ({})", trim_path(rest));
            }
            continue;
        }
        // Frame lines look like `  12: majit_metainterp::foo::bar`.
        let Some((_, symbol)) = line.split_once(": ") else {
            continue;
        };
        if is_plumbing(symbol) {
            pending = None;
        } else {
            pending = Some(symbol.to_string());
        }
    }
    match pending {
        Some(symbol) => symbol,
        None => String::from("<unnamed frame>"),
    }
}

/// Frames that describe the CONTAINER an allocation went into, or the meter
/// that observed it, rather than the code that asked for it.
fn is_plumbing(symbol: &str) -> bool {
    const CONTAINERS: &[&str] = &[
        "__rust",
        "_rjem_",
        "alloc::",
        "<alloc::",
        "core::",
        "<core::",
        "std::",
        "<std::",
        "<fn(",
        "hashbrown",
        "<hashbrown",
        "rustc_hash",
        "indexmap",
        "<indexmap",
        "smallvec",
        "<smallvec",
        "vecset",
        "<vecset",
        "bit_set",
        "<bit_set",
    ];
    // The meter's own frames, named ONE BY ONE rather than by the crate they
    // live in. This file holds two very different kinds of code: the meter,
    // which sits above every allocation and must never name itself; and the
    // `#[jit_interp]` fixture, which IS entry-path code and whose expansion is
    // one of the sites worth attributing. A `contains("allocs_per_compiled_
    // entry")` filter cannot tell them apart, and swallowed 6 of the 11
    // allocations per entry into an anonymous bucket when this was first run.
    const METER: &[&str] = &[
        "allocs_per_compiled_entry::bump",
        "allocs_per_compiled_entry::attribute",
        "allocs_per_compiled_entry::innermost_majit_frame",
        "allocs_per_compiled_entry::is_plumbing",
        "allocs_per_compiled_entry::trim_path",
        "allocs_per_compiled_entry::Counting",
        "<allocs_per_compiled_entry::Counting",
    ];
    METER.iter().any(|m| symbol.starts_with(m)) || CONTAINERS.iter().any(|c| symbol.starts_with(c))
}

/// Keep the last two path components, which is enough to identify a file
/// without pinning the absolute build directory into the report.
fn trim_path(at: &str) -> String {
    let parts: Vec<&str> = at.rsplit('/').take(2).collect();
    parts.into_iter().rev().collect::<Vec<_>>().join("/")
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    // Forwarded rather than left to the default (alloc + copy + dealloc) so
    // `System.realloc`'s in-place growth survives; one realloc is one
    // allocation event.
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

#[inline]
fn local_allocs() -> u64 {
    LOCAL_ALLOCS.with(Cell::get)
}

// what the entry hook records

/// Entries seen since the last [`reset_entries`].
static ENTRIES: AtomicU64 = AtomicU64::new(0);
/// The thread-local allocation counter as read at the FIRST entry of the
/// current window, and at the LAST. Their difference over
/// `ENTRIES - 1` gaps is the per-entry round-trip cost, measured entry
/// boundary to entry boundary so no interpreter prologue or epilogue is
/// half-counted.
static FIRST_AT_ENTRY: AtomicU64 = AtomicU64::new(0);
static LAST_AT_ENTRY: AtomicU64 = AtomicU64::new(0);

fn reset_entries() {
    ENTRIES.store(0, Ordering::Relaxed);
    FIRST_AT_ENTRY.store(0, Ordering::Relaxed);
    LAST_AT_ENTRY.store(0, Ordering::Relaxed);
}

/// Installed once on the persistent driver. Allocates nothing: two relaxed
/// atomics and a `Cell` read.
fn on_entry(_green_key: u64, _target_pc: usize) {
    let now = local_allocs();
    if ENTRIES.fetch_add(1, Ordering::Relaxed) == 0 {
        FIRST_AT_ENTRY.store(now, Ordering::Relaxed);
    }
    LAST_AT_ENTRY.store(now, Ordering::Relaxed);
}

/// `(entries, allocations between the first and last entry of the window)`.
fn window() -> (u64, u64) {
    let entries = ENTRIES.load(Ordering::Relaxed);
    let spanned = LAST_AT_ENTRY.load(Ordering::Relaxed) - FIRST_AT_ENTRY.load(Ordering::Relaxed);
    (entries, spanned)
}

// the GC

/// Install the process GC, because WHERE the jitframe comes from is part of
/// the configuration this file measures.
///
/// `llmodel.py malloc_jitframe` allocates the frame through the GC, and
/// `jitframe.py` makes JITFRAME an ordinary varsize GcStruct — so
/// upstream's per-entry frame is a nursery bump, which no `#[global_allocator]`
/// ever sees. The cranelift backend reproduces that only when a JITFRAME type
/// id is registered; with no GC at all it falls back to a `Vec<i64>`, and a
/// fixture that measured the fallback would be reporting a frame the shipped
/// configuration never allocates. pyre and cel reach the nursery branch through
/// `init_gc_subsystem` (pyre-jit `eval.rs`), which is the three calls below.
///
/// The nursery size is spelled out rather than left to `GcConfig::default`:
/// the default is `estimate_best_nursery_size`, which reads the host's cache
/// size, and a file whose whole point is an exact pinned integer must not take
/// a host-dependent input. Nothing here ever collects — the entry path
/// allocates its frames through the no-collect path and the fixture's state is
/// int-only — so the nursery only ever fills, and 4 MB is two orders of
/// magnitude more than the run needs. If a run ever did exhaust it, the spill
/// to old-gen is a `rawmalloc` and the assertion at the bottom would fail with
/// the spill named in the per-site table.
fn install_gc() {
    use majit_gc::collector::{GcConfig, MiniMarkGC};
    majit_gc::gc_sync::store_singleton(Box::new(MiniMarkGC::with_config(GcConfig {
        nursery_size: 1 << 22,
        ..GcConfig::default()
    })));
    // Takes the GIL, which every `gc_sync::gc_op` under the entry path asserts
    // this thread holds. The meter owns its process (`harness = false`), so it
    // holds the GIL for the whole run and never has to hand it back.
    majit_gc::gc_sync::register_thread();
    // Both backends require the descr's JITFRAME id before the install
    // (`check_jitframe_descr`): a collector with a type table allocates every
    // entry frame under that shape, which is the production frame source this
    // file meters. Registering the shape on the singleton and telling the
    // descr its id is what `eval.rs build_gc` does.
    majit_gc::gc_sync::gc_op(|gc| {
        let jitframe_tid =
            majit_gc::GcAllocator::register_type(gc, majit_backend::jitframe::jitframe_type_info());
        majit_gc::GcAllocator::set_jitframe_type_id(gc, jitframe_tid);
        majit_metainterp::opencoder::register_trace_ops_gc_type(gc);
    });
    #[cfg(feature = "cranelift")]
    majit_backend_cranelift::install_gc_standalone();
    #[cfg(all(feature = "dynasm", not(feature = "cranelift")))]
    majit_backend_dynasm::runner::install_gc_standalone();
}

// the fixture: one machine, one persistent driver

pub type Bytecode = [u8];

const C_ADD: u8 = 1; // acc += n; n -= 1
const C_BR_COND: u8 = 2; // [C_BR_COND, off]: if n != 0 jump
const C_RETURN: u8 = 3;

struct CounterState {
    acc: i64,
    n: i64,
}

/// `acc = n + (n-1) + ... + 1`.
///
/// THE DRIVER IS A PARAMETER, and that is the whole reason this fixture exists
/// rather than reusing one of the `#[jit_interp]` fixtures next door. Every
/// other machine in this crate spells `let mut driver = JitDriver::new(..)`
/// inside the annotated function, which makes one call one driver: the artifact
/// is compiled and thrown away within the call that paid for it, and no call
/// ever *enters* an already-warm artifact. That fixture shape cannot express
/// the regime this file measures no matter how it is driven.
#[majit_macros::jit_interp(
    state = CounterState,
    env = Bytecode,
    auto_calls = true,
    greens = [pc, program],
    state_fields = {
        acc: int,
        n: int,
    },
)]
#[allow(unused_assignments, unused_variables)]
pub fn mainloop(
    mut driver: &mut JitDriver<CounterState>,
    program: &Bytecode,
    inputarg: i64,
) -> i64 {
    let mut pc: usize = 0;
    let mut state = CounterState {
        acc: 0,
        n: inputarg,
    };

    while pc < program.len() {
        jit_merge_point!(driver, program, pc; state);

        let opcode = program[pc];
        pc += 1;

        match opcode {
            C_ADD => {
                state.acc = state.acc + state.n;
                state.n = state.n - 1;
            }
            C_BR_COND => {
                let offset = program[pc] as i8 as i64;
                let target = ((pc as i64) + offset + 1) as usize;
                pc += 1;
                if state.n != 0 {
                    if target <= pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            C_RETURN => break,
            _ => {}
        }
    }

    state.acc
}

fn counter_program() -> Vec<u8> {
    vec![
        C_ADD, // 0
        C_BR_COND, 253,      // 1  (-3 -> back to 0)
        C_RETURN, // 3
    ]
}

/// Iterations per call. Small on purpose: the figure this file reports is a
/// per-ENTRY cost, and a long inner loop amortizes the entry over iterations
/// that never touch the entry path.
const N: i64 = 8;

/// Back edges before the merge point traces. Matches the in-tree fixtures.
const THRESHOLD: u32 = 4;

fn expected() -> i64 {
    N * (N + 1) / 2
}

// report

fn backend() -> &'static str {
    if cfg!(feature = "cranelift") {
        "cranelift"
    } else if cfg!(feature = "dynasm") {
        "dynasm"
    } else {
        "none"
    }
}

fn profile() -> &'static str {
    if cfg!(debug_assertions) {
        "dev"
    } else {
        "release"
    }
}

fn main() {
    install_gc();
    let program = counter_program();

    // Bind ONCE. Everything below calls into this driver; nothing rebuilds it.
    let mut driver: JitDriver<CounterState> = JitDriver::new(THRESHOLD);
    {
        use majit_metainterp::JitState as _;
        // `MetaInterp::install_canonical_liveness` takes `Arc::get_mut` on the
        // staticdata and so must run exactly once, before anything clones it.
        // The per-call fixtures get away with calling it every call only
        // because their driver is per-call too.
        CounterState { acc: 0, n: 0 }
            .build_meta(0, &program)
            .install_canonical_liveness(&mut driver);
    }
    driver.set_on_compiled_entry(on_entry);

    // Warm: trace, compile, and take every guard-failure bridge, so the window
    // below sees an artifact that is already whole.
    const WARMUP: usize = 512;
    for _ in 0..WARMUP {
        let got = mainloop(&mut driver, &program, N);
        assert_eq!(
            got,
            expected(),
            "the fixture must compute its own answer; a wrong result means the \
             allocation figure below was taken off a broken run"
        );
    }
    let warm_entries = ENTRIES.load(Ordering::Relaxed);
    assert!(
        warm_entries > 0,
        "no compiled entry happened in {WARMUP} warm-up calls, so this run \
         measures the INTERPRETER and not the entry path. Either the fixture \
         stopped compiling or `on_compiled_entry` stopped firing; a number \
         reported from here would be about neither."
    );

    // the counted window
    const CALLS: usize = 256;
    reset_entries();
    let before_local = local_allocs();
    let before_global = GLOBAL_ALLOCS.load(Ordering::Relaxed);
    for _ in 0..CALLS {
        black_box(mainloop(&mut driver, &program, N));
    }
    let call_allocs = local_allocs() - before_local;
    let off_thread =
        (GLOBAL_ALLOCS.load(Ordering::Relaxed) - before_global).saturating_sub(call_allocs);
    let (entries, spanned) = window();

    // the attribution window
    // Separate and short: symbolizing a backtrace per allocation costs far
    // more than the entry it describes, so a counted window run this way
    // would be measuring the profiler.
    const ATTRIBUTED_CALLS: usize = 8;
    reset_entries();
    SITES.with(|s| s.borrow_mut().clear());
    ATTRIBUTING.store(true, Ordering::Relaxed);
    for _ in 0..ATTRIBUTED_CALLS {
        black_box(mainloop(&mut driver, &program, N));
    }
    ATTRIBUTING.store(false, Ordering::Relaxed);
    let (attributed_entries, _) = window();
    let sites: Vec<(String, u64)> = SITES.with(|s| {
        let mut v: Vec<(String, u64)> = s.borrow().iter().map(|(k, n)| (k.clone(), *n)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v
    });

    println!("allocations per compiled-code entry — majit warm entry path");
    println!(
        "backend={} profile={} threshold={THRESHOLD} n={N} warmup={WARMUP} calls={CALLS}",
        backend(),
        profile()
    );
    println!();
    println!("  calls in window          {CALLS}");
    println!("  compiled entries         {entries}");
    println!("  allocations (thread)     {call_allocs}");
    println!("  allocations (off-thread) {off_thread}");
    println!(
        "  per call                 {:.3}",
        call_allocs as f64 / CALLS as f64
    );
    println!(
        "  per entry, entry-to-entry {:.3}   ({spanned} allocations across {} gaps)",
        spanned as f64 / (entries.saturating_sub(1)).max(1) as f64,
        entries.saturating_sub(1)
    );
    println!();
    println!("per-site attribution over {ATTRIBUTED_CALLS} calls / {attributed_entries} entries:");
    let attributed: u64 = sites.iter().map(|(_, n)| *n).sum();
    for (site, count) in &sites {
        println!(
            "  {count:>6}  {:>6.3}/entry  {site}",
            *count as f64 / attributed_entries.max(1) as f64
        );
    }
    println!("  {attributed:>6}  total attributed");

    assert_eq!(
        off_thread, 0,
        "another thread allocated inside the measured window, so the \
         thread-local isolation is not doing what the header claims"
    );
    assert_eq!(
        entries as usize, CALLS,
        "the window must be one compiled entry per call for the per-call and \
         per-entry columns above to be the same measurement. {entries} entries \
         over {CALLS} calls means the fixture changed shape and the two columns \
         are no longer comparable."
    );

    // The number this file exists to pin. An EXACT equality, not a bound: a
    // `<=` passes just as happily on a run that stopped entering compiled code
    // altogether, and a `>=` cannot see the improvement it was written to
    // protect. The per-site table above is the breakdown of this number, and
    // the table is printed on every run precisely so that a failure here comes
    // with its own diagnosis instead of a bare integer.
    assert_eq!(
        call_allocs as usize,
        ALLOCS_PER_ENTRY * CALLS,
        "warm entry allocated {:.3} times per entry; the pinned figure is \
         {ALLOCS_PER_ENTRY}. Read the per-site table printed above: it names \
         every allocation by symbol and file:line, so the row that moved says \
         which site changed. Re-pin `ALLOCS_PER_ENTRY` only after deciding the \
         move is intended.",
        call_allocs as f64 / CALLS as f64
    );
}

/// Heap allocations per warm compiled-code entry, as measured by this file.
///
/// One number for both backends: since `malloc_jitframe` allocates against
/// the descr, the entry frame is a nursery bump on either backend and the
/// remaining rows are backend-neutral. The figure was per-backend while
/// dynasm still built its entry frame off the GC.
///
/// All four allocations are shared by both backends:
///
/// | n | site |
/// |---|------|
/// | 2 | `CounterState::extract_live_into` — TWO separate `extract_live()` calls, one `Vec<i64>` each: `JitState::extract_live_values`'s default (`jit_state.rs`) calls it and then `live_value_types`, whose default calls it AGAIN. A two-int-scalar state builds each Vec in one allocation (capacity 0→4, no regrowth); the 2 is a doubled traversal, not a regrowth |
/// | 1 | `JitState::extract_live_values` default body in `jit_state.rs` — the `Vec<Value>` it collects into |
/// | 1 | `JitState::live_value_types` default body in `jit_state.rs` — the `Vec<Type>` it collects into |
///
/// Four shared rows have gone since:
///
/// - `<[Type]>::to_vec`, the exit's type list copied out of the failing descr
///   into the exit layout. `CompiledExitLayout::exit_types` is now inline
///   storage wide enough for the exits a steady entry produces, so the copy
///   costs the allocator nothing until an exit is wider than that.
/// - `<[Value]>::to_vec`, the FINISH values copied out of the run result. The
///   FINISH arm returns immediately after publishing them, so nothing reads
///   the source and the values are moved instead.
/// - `run_compiled_detailed_with_values_at_dispatch_key` — `values`, the run
///   result's own machine-word output buffer.
/// - the same function's `typed_values`, its typed twin. Both are now inline
///   storage on the result, sized to the exit widths a steady entry returns,
///   so decoding an exit asks the allocator for nothing. The buffers could not
///   be pooled and reused instead: each one leaves the function owned — into
///   the finish latch, into the public run outcome, into the guard arm's own
///   copy — so there is nothing to hand back.
///
/// ⚠ ALL FOUR ARE A PROPERTY OF THIS FIXTURE'S STATE SHAPE, not of the
/// entry path in general. `codegen_state.rs`'s `generate_state_fields_jit_state`
/// emits the non-allocating
/// `extract_live_values_into` / `live_value_types_into` overrides only when the
/// state has a ref scalar, a float scalar or a virt array; a state of int
/// scalars only — which `CounterState` is — gets neither, and falls back to the
/// allocating `JitState` defaults. A state that clears that gate pays 4 fewer.
///
/// `dynasm` adds NOTHING, and its four are the four shared rows above.
///
/// It used to add three. The three that went:
///
/// - `majit_backend::jitframe::alloc_off_gc_jitframe`, the JITFRAME itself.
///   `malloc_jitframe` allocates under the descr's JITFRAME type id, and this
///   fixture registers the shape ([`install_gc`]), so the frame is a
///   `nursery_free` bump (`jitframe.py`) and costs the process
///   allocator nothing — the same row cranelift lost when its frame moved to
///   the nursery.
///
/// - `raw_values`, a copy of every jitframe slot taken because
///   `DynasmBackend::execute_token` freed the frame before returning.
///   `llmodel.py` reads those slots out of the frame itself, so the
///   copy had no upstream counterpart; the frame now lives as long as the
///   deadframe does and the accessors read it in place.
/// - the `Box` inside `DeadFrame::Boxed` holding the deadframe
///   `DynasmBackend::execute_token` returns. It held the frame POINTER, not a
///   copy of the frame, and what it bought was type erasure across a crate
///   boundary: `majit-backend` could not name a type declared in
///   `majit-backend-dynasm`. That constraint has no upstream counterpart —
///   `llsupport/jitframe.py` sits below every machine backend and
///   `llmodel.py` names `jitframe.JITFRAMEPTR` directly — so the deadframe
///   type moved down into `majit-backend` and `DeadFrame` holds it by value in
///   a variant of its own.
///
/// `cranelift` adds NOTHING either — the backends agree now that both take
/// the entry frame from `malloc_jitframe` against the same descr.
///
/// It used to add four. The four that went:
///
/// - `deadframe_from_jitframe` — the `Box<JitFrameDeadFrame>` inside
///   `DeadFrame`. The box was not the `llmodel.py` `cast_opaque_ptr` it was
///   annotated as; it was a PIN. The deadframe's frame pointer was rooted by
///   registering the address of the field holding it, and a field address is
///   only a valid root while its owner stays put, so the owner had to be given
///   a fixed address before it could be returned. The pointer now lives in a
///   root slot addressed by POSITION (`shadowstack.py push_stack`), which the
///   collector rewrites in place, so the holder may move and `DeadFrame` holds
///   the frame by value.
/// - `CraneliftBackend::execute_token_with_dispatch_key` unwrapped the
///   metainterp's `&[Value]` into an owned `Vec<i64>` before the frame
///   existed. `llmodel.py` unwraps each argument *at* the store into
///   its frame slot, so upstream never holds a second list; the arguments are
///   now handed down as `FrameInputs::Values` and unwrapped against the frame.
/// - `JitExecResult::extract_outputs` copied every exit slot out of the frame
///   on every exit, and the only two consumers are branches — the
///   CALL_ASSEMBLER sentinel, which reads slot 0, and external-JUMP re-entry.
///   `llmodel.py` reads slots out of the frame through the accessors,
///   so the copy is now taken only where it is consumed. This is the same
///   removal `raw_values` got on dynasm.
/// - `<i64 as SpecFromElem>::from_elem`, the `Vec<i64>` behind
///   `JitFrameDeadFrame::_heap_owner`. That was the no-GC branch of
///   `run_compiled_code_inner`, reached because this fixture had no GC at all;
///   see [`install_gc`]. The branch itself remains for a backend running
///   without a collector.
const ALLOCS_PER_ENTRY: usize = 4;
