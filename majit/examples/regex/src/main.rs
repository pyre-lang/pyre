//! Marked-regex matching, from "An Efficient and Elegant Regular Expression
//! Matcher in Python" and "A JIT for Regular Expression Matching".
//!
//! A regex is a tree; each node holds one mutable `marked` bit, and one
//! `shift(c, mark)` per input character propagates the marks left to right.
//! Two things a reader should not have to guess: **the marks live in the
//! nodes**, not in the matcher, and `empty` — does this subexpression accept
//! the empty string? — is **computed once, at lowering time**, so the matcher
//! only reads it.
//!
//! `regex` builds and lowers the tree; `interp` is the plain matcher over the
//! lowered graph; `jit_interp` and `shortcircuit` are the two JIT portals.
//! `shortcircuit` is the faithful port of the post's own source, which writes
//! `shift` with `and`/`or`; `jit_interp` is the `&`/`|` variant part 2 of the
//! series recommends. `rpython_original/` runs the post's RPython through
//! RPython's own JIT and settles which is which — the traces agree op for op
//! on every structural count, and the guard counts (RPython 27, `shortcircuit`
//! 26, `jit_interp` 1) name the faithful one.
//!
//! This file is also the benchmark, and its measurement discipline is half of
//! what it has to say. Every number printed below without a disclaimer was
//! measured in this process, in this run: each row is the median of
//! [`REPEATS`] timed runs with its min-max range beside it, taken at three
//! input lengths, and every ratio is one measured row divided by another.
//! Nothing is a constant carried over from an earlier session. What this run
//! could not measure — the post's own 2010 figures — is printed under a heading
//! that says so, and no ratio above it is computed from any of it.

#[cfg(feature = "alloc-census")]
pub mod alloc_census;
pub mod gc;
pub mod interp;
pub mod jit_interp;
pub mod regex;
pub mod rstr;
pub mod shortcircuit;

#[cfg(feature = "alloc-census")]
#[global_allocator]
static ALLOC: majit_gc::HostNurseryClock<alloc_census::Counting> =
    majit_gc::HostNurseryClock(alloc_census::Counting);

// The census wraps this same allocator. Selecting `fast-alloc` must not
// silently select System when the timing build leaves the counters out.
#[cfg(all(feature = "fast-alloc", not(feature = "alloc-census")))]
#[global_allocator]
static ALLOC: majit_gc::HostNurseryClock<mimalloc::MiMalloc> =
    majit_gc::HostNurseryClock(mimalloc::MiMalloc);

#[cfg(not(any(feature = "alloc-census", feature = "fast-alloc")))]
#[global_allocator]
static ALLOC: majit_gc::HostNurseryClock<std::alloc::System> =
    majit_gc::HostNurseryClock(std::alloc::System);

use regex::{NodeRec, bench_regex, bench_regex_left, count, depth, lower, nonmatching, vectors};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Timed runs per row. The example's own history is why this is not 1: the
/// same binary on this machine produced a 4.8x and a 5.5x minutes apart, and a
/// single run cannot tell which of those the machine actually does.
const REPEATS: usize = 5;

/// The post benchmarks "a random string (of varying lengths)". Three lengths
/// two orders of magnitude apart is enough to see whether a row reports a
/// per-character rate at all, which is the only reading under which the rows
/// can be compared to each other or to anybody else.
const LENGTHS: [usize; 3] = [1 << 12, 1 << 16, 1 << 20];

/// Keep the published benchmark's three-length sweep by default, but let the
/// allocation census select a short, deterministic diagnostic input.  The
/// latter counts per character and does not gain information by spending
/// minutes repeating the same allocation pattern at 1 MiB.
fn lengths() -> Vec<usize> {
    let Some(value) = std::env::var_os("PYRE_REGEX_LENGTHS") else {
        return LENGTHS.to_vec();
    };
    let value = value.to_string_lossy();
    let parsed: Vec<usize> = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .unwrap_or_else(|_| panic!("PYRE_REGEX_LENGTHS contains a non-usize: {part:?}"))
        })
        .collect();
    assert!(
        !parsed.is_empty() && parsed.iter().all(|&length| length > 0),
        "PYRE_REGEX_LENGTHS must contain one or more positive comma-separated lengths"
    );
    parsed
}

/// `PYRE_REGEX_ROWS=<comma-separated row indexes>` runs only those rows of
/// [`NAMES`]. A process-wide counter such as `/usr/bin/time -l`'s
/// `instructions retired` then charges one row alone, which is the reading
/// that survives a loaded machine. Deselected rows are absent, not synthetic
/// NaNs, so they cannot enter the summary's spreads or ratios.
fn row_selected(index: usize) -> bool {
    static ROWS: std::sync::OnceLock<Option<Vec<usize>>> = std::sync::OnceLock::new();
    ROWS.get_or_init(|| {
        let value = std::env::var_os("PYRE_REGEX_ROWS")?;
        let value = value.to_string_lossy();
        Some(
            value
                .split(',')
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .map(|part| {
                    part.parse::<usize>().unwrap_or_else(|_| {
                        panic!("PYRE_REGEX_ROWS contains a non-usize: {part:?}")
                    })
                })
                .collect(),
        )
    })
    .as_ref()
    .is_none_or(|rows| rows.contains(&index))
}

/// The `{N}` of the post's benchmark regex `(a|b)*a(a|b){20}a(a|b)*`.
const N: usize = 20;

/// The generator seed. Fixed, so every run of this binary — and every port
/// that reimplements `nonmatching` — scans exactly the same bytes.
const SEED: u64 = 42;

/// Back-edge count before a portal records. The tests in `jit_interp` grade
/// the loop compiled at this threshold, so the benchmark times that same loop.
const THRESHOLD: u32 = 3;

/// A row whose fastest and slowest length differ by more than this is not
/// reporting a per-character rate, and the difference is a finding rather than
/// noise.
const FLAT_ENOUGH: f64 = 1.10;

const NAMES: [&str; 3] = [
    "Rust interp over NodeRec, no JIT",
    "majit JIT, `&`/`|` variant",
    "majit JIT, the post's and/or",
];
const R_INTERP: usize = 0;
const R_JIT: usize = 1;

/// Characters scanned per second.
fn rate(chars: usize, secs: f64) -> f64 {
    chars as f64 / secs
}

fn commas(x: f64) -> String {
    let n = x.round() as u64;
    let d = n.to_string();
    let mut out = String::new();
    for (i, c) in d.chars().enumerate() {
        if i > 0 && (d.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// What the run can say about the machine without shelling out: the target the
/// binary was built for, plus the two runtime facts `std` will hand over.
///
/// The post records its own hardware — "Intel Core2 Duo P8400 2.26GHz" — and a
/// chars/s figure means nothing without it.
fn machine() -> String {
    let cores =
        std::thread::available_parallelism().map_or_else(|_| "?".to_string(), |n| n.to_string());
    let backend = if cfg!(feature = "dynasm") {
        "dynasm"
    } else if cfg!(feature = "cranelift") {
        "cranelift"
    } else {
        "none"
    };
    let profile = if cfg!(debug_assertions) {
        "debug — NOT a benchmark build"
    } else {
        "release"
    };
    format!(
        "machine, read at runtime: {} {}, {} logical cores, {}-bit pointers, \
         majit backend {backend}, {profile} build",
        std::env::consts::ARCH,
        std::env::consts::OS,
        cores,
        usize::BITS,
    )
}

/// One implementation measured at one input length.
struct Row {
    name: &'static str,
    /// chars/s, one per timed run, ascending.
    rates: Vec<f64>,
    /// Loops the JIT compiled across the timed runs. 0 for a row that never
    /// enters the JIT.
    compiles: usize,
    /// What the timed runs allocated. Deterministic where `rates` is not, so
    /// this is the half of the row that grades a per-deopt cost change on a
    /// machine under load. See [`alloc_census`].
    #[cfg(feature = "alloc-census")]
    alloc: alloc_census::Census,
    /// Exact-size allocation fingerprints for this row.  Kept empty unless
    /// explicitly requested so the ordinary census remains one compact line.
    #[cfg(feature = "alloc-census")]
    alloc_sizes: Vec<String>,
}

impl Row {
    /// The reported number: median, not mean.
    ///
    /// Benchmark noise is one-sided. A scheduler preemption, an interrupt, a
    /// page fault or a thermal step can only ever make a run slower — nothing
    /// makes a run faster than the machine can go. A mean folds every such
    /// excursion into the answer, so one bad run moves it and the figure
    /// reported is a run that never happened. The median is a run that did
    /// happen, and it is the one a rerun reproduces. The min-max range is
    /// printed beside it so the spread is visible instead of taken on trust: a
    /// median with a 2x range under it is not a measurement either, and the
    /// reader should be able to see that without being told.
    fn median(&self) -> f64 {
        let n = self.rates.len();
        if n % 2 == 1 {
            self.rates[n / 2]
        } else {
            (self.rates[n / 2 - 1] + self.rates[n / 2]) / 2.0
        }
    }
    fn min(&self) -> f64 {
        self.rates[0]
    }
    fn max(&self) -> f64 {
        self.rates[self.rates.len() - 1]
    }
}

/// Time `run` over `s`, [`REPEATS`] times.
///
/// One untimed run first: the first pass over a freshly built buffer takes the
/// page faults and the cold branch predictors, and those are not what the row
/// is about.
///
/// Every run's answer is asserted, timed ones included. That assertion is also
/// the hygiene this file used to spell as "a fresh tree per row": a mark left
/// behind by an earlier run would surface here as a spurious match, because
/// the input is built not to match.
fn bench(
    name: &'static str,
    root: *mut NodeRec,
    s: &[u8],
    counter: Option<&AtomicUsize>,
    mut run: impl FnMut(*mut NodeRec, &[u8]) -> bool,
) -> Option<Row> {
    let index = NAMES
        .iter()
        .position(|candidate| *candidate == name)
        .expect("every row is named in NAMES");
    if !row_selected(index) {
        return None;
    }
    assert!(
        !run(root, s),
        "{name}: the benchmark input is supposed NOT to match"
    );
    #[cfg(feature = "alloc-census")]
    alloc_census::rearm_trace(name);
    let before = counter.map_or(0, |c| c.load(Ordering::Relaxed));
    #[cfg(feature = "alloc-census")]
    let alloc_before = alloc_census::read();
    #[cfg(feature = "alloc-census")]
    let sizes_before = alloc_census::read_sizes();
    let mut rates = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let t0 = Instant::now();
        let hit = run(root, s);
        let secs = t0.elapsed().as_secs_f64();
        assert!(!hit, "{name}: the benchmark input is supposed NOT to match");
        rates.push(rate(s.len(), secs));
    }
    #[cfg(feature = "alloc-census")]
    let alloc = alloc_before.since(alloc_census::read());
    #[cfg(feature = "alloc-census")]
    let alloc_sizes = if std::env::var_os("PYRE_CENSUS_HISTOGRAM").is_some() {
        let sizes_after = alloc_census::read_sizes();
        let oversize = sizes_before.oversize_since(&sizes_after);
        let rows = sizes_before.since(&sizes_after);
        alloc_census::report_rows(&rows, oversize, s.len(), REPEATS, 16)
    } else {
        Vec::new()
    };
    let compiles = counter.map_or(0, |c| c.load(Ordering::Relaxed) - before);
    rates.sort_by(f64::total_cmp);
    Some(Row {
        name,
        rates,
        compiles,
        #[cfg(feature = "alloc-census")]
        alloc,
        #[cfg(feature = "alloc-census")]
        alloc_sizes,
    })
}

/// The peeled loop body of a recorded trace: everything from the last `Label`
/// on. A compiled loop is preamble plus peeled body and only the body runs per
/// input character, so grading the whole thing charges the body for the reads
/// the preamble hoisted out of it.
fn peeled(body: &[majit_ir::OpCode]) -> &[majit_ir::OpCode] {
    match body.iter().rposition(|op| *op == majit_ir::OpCode::Label) {
        Some(i) => &body[i..],
        None => body,
    }
}

/// The census a compiled body is read by. `getfield_gc_r` is the subject and
/// zero is the claim — the tree walk's pointer reads are gone — and the total
/// is printed beside it because a body of a handful of ops would also have no
/// pointer reads and would mean the walk never entered the trace at all.
fn census(body: &[majit_ir::OpCode]) -> String {
    use majit_ir::OpCode;
    let n = |op: OpCode| body.iter().filter(|o| **o == op).count();
    format!(
        "{:>4} ops: {} getfield_gc_r, {} getfield_gc_i, {} setfield_gc, {} int_eq, {} guard_value",
        body.len(),
        n(OpCode::GetfieldGcR),
        n(OpCode::GetfieldGcI),
        n(OpCode::SetfieldGc),
        n(OpCode::IntEq),
        n(OpCode::GuardValue),
    )
}

fn main() {
    #[cfg(feature = "alloc-census")]
    alloc_census::configure_trace_from_env();
    // Before anything runs compiled code. `target.py --opt=jit` is a translated
    // binary and carries a collector; this is majit's side of that.
    gc::install();
    let started = Instant::now();
    println!("{}", machine());
    println!();

    // ── correctness first: a fast wrong matcher is not a result ────────────
    let mut bad = 0;
    let total = vectors().len();
    for (re, s, want) in vectors() {
        let got = interp::matches(lower(&re), s.as_bytes());
        if got != want {
            bad += 1;
            println!("  FAIL {s:?}: got {got}, want {want}");
        }
    }
    println!("vectors: {}/{total} pass", total - bad);

    let balanced = lower(&bench_regex(N));
    let left_assoc = lower(&bench_regex_left(N));
    println!(
        "(a|b)*a(a|b){{{N}}}a(a|b)*  balanced: {} nodes, depth {}",
        count(balanced),
        depth(balanced)
    );
    println!(
        "(a|b)*a(a|b){{{N}}}a(a|b)*  left    : {} nodes, depth {}",
        count(left_assoc),
        depth(left_assoc)
    );

    // One tree per row, for the whole sweep. Two rows may never share a tree:
    // a mark is node state, so whatever one row left behind the other would
    // read. Re-lowering per length would only leak more graphs — `matches`
    // clears the marks it leaves, and `bench` asserts every answer, which is
    // the thing that breaks first if that ever stops being true.
    let roots = [
        lower(&bench_regex(N)),
        lower(&bench_regex(N)),
        lower(&bench_regex(N)),
    ];
    // warmspot.py owns one JitDriver for a portal. Keep each portal instance
    // across the whole sweep so compiled loops are reused between matches.
    let mut masking_matcher = jit_interp::Matcher::new(roots[1], THRESHOLD);
    let mut branching_matcher = shortcircuit::Matcher::new(roots[2], THRESHOLD);
    masking_matcher.register_gc_roots();
    branching_matcher.register_gc_roots();

    println!();
    println!("timed rows: {REPEATS} runs each, median (min - max).");
    println!("the input is nonmatching(len, {N}, {SEED}) — it matches nothing, so the matcher");
    println!("must read every character and no early exit can hide the per-character cost.");

    let mut sweep: Vec<(usize, [Option<Row>; 3])> = Vec::new();
    for len in lengths() {
        let s = nonmatching(len, N, SEED);
        assert!(
            !interp::matches(balanced, &s),
            "the benchmark input is supposed NOT to match"
        );

        let interp_row = bench(NAMES[0], roots[0], &s, None, |r, s| interp::matches(r, s));
        let masking_row = bench(
            NAMES[1],
            roots[1],
            &s,
            Some(&jit_interp::COMPILES),
            |_r, s| masking_matcher.matches(s),
        );
        let branching_row = bench(
            NAMES[2],
            roots[2],
            &s,
            Some(&shortcircuit::COMPILES),
            |_r, s| branching_matcher.matches(s),
        );
        let rows = [interp_row, masking_row, branching_row];

        println!();
        println!("  {} chars, no match as intended:", commas(len as f64));
        for (name, row) in NAMES.iter().zip(&rows) {
            let Some(row) = row else {
                println!("    {name:<32}: not selected");
                continue;
            };
            let compiled = if row.compiles > 0 {
                format!(
                    "   {:.1} loops compiled per run",
                    row.compiles as f64 / REPEATS as f64
                )
            } else {
                String::new()
            };
            println!(
                "    {:<32}: {:>12} chars/s  ({:>12} - {:>12}){}",
                row.name,
                commas(row.median()),
                commas(row.min()),
                commas(row.max()),
                compiled,
            );
            #[cfg(feature = "alloc-census")]
            println!("    {:<32}  {}", "", row.alloc.per_char(len, REPEATS));
            #[cfg(feature = "alloc-census")]
            for size in &row.alloc_sizes {
                println!("    {:<32}  {size}", "");
            }
        }
        sweep.push((len, rows));
    }

    // ── is any of it a per-character rate? ─────────────────────────────────
    println!();
    println!("rate against input length, median chars/s. The algorithm is O(m*n) with no");
    println!("per-character allocation, so a flat row is the expectation and a row that is not");
    println!("flat is reporting something other than the cost of one character.");
    print!("  {:<32}", "implementation");
    for (len, _) in &sweep {
        print!("{:>14}", commas(*len as f64));
    }
    println!("{:>9}", "spread");

    // Per row: the spread across the lengths, and which length was fastest —
    // the direction is what separates the two explanations below.
    let mut shape: Vec<Option<(f64, usize)>> = Vec::new();
    for (i, name) in NAMES.iter().enumerate() {
        print!("  {name:<32}");
        let (mut lo, mut hi, mut best) = (f64::MAX, 0.0f64, 0usize);
        for (j, (_, rows)) in sweep.iter().enumerate() {
            if let Some(row) = &rows[i] {
                let m = row.median();
                lo = lo.min(m);
                if m > hi {
                    hi = m;
                    best = j;
                }
                print!("{:>14}", commas(m));
            } else {
                print!("{:>14}", "-");
            }
        }
        if hi == 0.0 {
            println!("{:>9}", "n/a");
            shape.push(None);
        } else {
            println!("{:>8.2}x", hi / lo);
            shape.push(Some((hi / lo, best)));
        }
    }

    let not_flat: Vec<usize> = (0..NAMES.len())
        .filter(|i| shape[*i].is_some_and(|shape| shape.0 > FLAT_ENOUGH))
        .collect();
    if !not_flat.is_empty() {
        println!();
        println!("  FINDING: not every row reports a per-character rate.");
        for i in &not_flat {
            let (spread, best) = shape[*i].expect("not_flat contains measured rows");
            println!(
                "    {:<32} spread {:>6.2}x, fastest at {:>9} chars",
                NAMES[*i],
                spread,
                commas(sweep[best].0 as f64),
            );
        }
        println!("    A row fastest at the LONGEST input is amortizing a cost that is fixed per");
        println!(
            "    `matches` call. The JitDriver now follows warmspot.py ownership and survives"
        );
        println!("    the whole sweep, so `loops compiled per run` distinguishes a real new");
        println!("    specialization from fixed per-call setup.");
        println!("    A row fastest at the SHORTEST input has two candidates and this sweep does");
        println!("    not separate them. The tree is 93 nodes at every length, but the input");
        println!("    buffer grows from 4 KiB to 1 MiB, so the longest length is not reading out");
        println!("    of the same level of cache as the shortest. And a machine doing other work");
        println!("    at the same time costs a long run more than a short one — that one shows in");
        println!("    the min-max ranges above, so read the range before the median. An otherwise");
        println!("    idle machine is the control that tells the two apart.");
    }

    // ── what the JIT rows actually compiled ────────────────────────────────
    let jit_body = jit_interp::LAST_BODY.lock().unwrap().clone();
    let sc_body = shortcircuit::last_peeled_body();
    if !jit_body.is_empty() || !sc_body.is_empty() {
        println!();
        println!(
            "the peeled loop body of the last loop each portal compiled in this run — this is"
        );
        println!("what runs per input character, and 0 getfield_gc_r is the experiment's claim:");
        if !jit_body.is_empty() {
            println!("  {:<32}: {}", NAMES[1], census(peeled(&jit_body)));
        }
        if !sc_body.is_empty() {
            println!("  {:<32}: {}", NAMES[2], census(&sc_body));
        }
    }

    // ── the summary, in the post's shape ───────────────────────────────────
    let (len, rows) = sweep.last().expect("lengths() is not empty");
    let slowest = rows
        .iter()
        .flatten()
        .min_by(|a, b| a.median().total_cmp(&b.median()))
        .expect("PYRE_REGEX_ROWS selected no benchmark rows");
    println!();
    println!(
        "SUMMARY — (a|b)*a(a|b){{{N}}}a(a|b)* over {} chars, median of {REPEATS} runs.",
        commas(*len as f64)
    );
    println!("Every ratio divides this run's numbers by this run's own slowest measured row —");
    println!("the post's own speedup column divides by its slowest row too, which for it was");
    println!("pure Python. Here the denominator is:");
    println!("  {}", slowest.name);
    println!(
        "  {:<32}{:>14}{:>12}",
        "implementation", "chars/s", "speedup"
    );
    for (name, row) in NAMES.iter().zip(rows) {
        let Some(row) = row else {
            println!("  {name:<32}{:>26}", "not selected");
            continue;
        };
        println!(
            "  {:<32}{:>14}{:>11.1}x",
            row.name,
            commas(row.median()),
            row.median() / slowest.median()
        );
    }
    println!("  {:<32}{:>26}", "CPython `re`", "not ported here");

    // ── the two ratios the post makes its claims from ──────────────────────
    println!();
    println!("the two ratios the post's headline claims are made of:");
    if let (Some(jit), Some(aot)) = (&rows[R_JIT], &rows[R_INTERP]) {
        println!(
            "  majit JIT / the same matcher compiled ahead of time : {:>5.1}x",
            jit.median() / aot.median()
        );
    } else {
        println!("  majit JIT / the same matcher compiled ahead of time :   n/a");
    }
    println!("      the comparable quantity in the post is its own 16,500,000 / 720,000 = 22.9x,");
    println!("      on its 2010 machine. Both sides divide a JIT that specialized on the regex");
    println!("      by the same algorithm compiled ahead of time — here `rustc -O`, there");
    println!("      RPython translated to C. The absolute rows are not comparable across");
    println!("      sixteen years and two instruction sets; the ratio is.");
    println!("  majit JIT / CPython `re`                            : not ported here");
    println!("      the post's second ratio, and this example does not reproduce it. It would");
    println!("      not be like-for-like anyway: `re` is a backtracking engine that may bail");
    println!("      out of a non-match early, while the marked matcher always scans the whole");
    println!("      string, so the two do different amounts of work per character.");

    // ── everything below here this run did NOT measure ─────────────────────
    println!();
    println!("NOT MEASURED BY THIS RUN — the post's other rows. This example compares itself");
    println!("against the RPython original and nothing else, so the pure-Python, C++ and");
    println!("CPython `re` rows have no port here and no number below is one of theirs. The");
    println!("comparison that IS owed is a trace comparison, not a speed one:");
    println!("  pypy majit/examples/regex/rpython_original/runner.py {N} 4096");
    println!();
    println!("NOT MEASURED BY THIS RUN, AND NOT THIS MACHINE — the 2010 post's own figures, on");
    println!("its own hardware (Intel Core2 Duo P8400, 2.26GHz). Sixteen years and a different");
    println!("instruction set away; only its internal ratios are comparable to anything above:");
    println!("  pure Python                      :       12,200 chars/s");
    println!("  RPython translated to C          :      720,000 chars/s");
    println!("  CPython `re`                     :    2,500,000 chars/s");
    println!("  RPython + JIT, regex green       :   16,500,000 chars/s");
    println!("  its JIT-over-no-JIT ratio        :         22.9x");
    println!("  its JIT-over-`re` ratio          :          6.6x");

    println!();
    println!(
        "this run took {:.1}s of wall clock, all of it inside the rows above.",
        started.elapsed().as_secs_f64()
    );

    // The bridge row, on stderr so it never enters the tables above. `BRIDGES`
    // is what the guards earned; `BRIDGE_OPS` is how much trace each carried,
    // and the quotient is the unit a per-bridge cost divides by.
    {
        use std::sync::atomic::Ordering::Relaxed;
        let bridges = shortcircuit::BRIDGES.load(Relaxed);
        if bridges != 0 {
            eprintln!(
                "bridges {bridges}, ops {}, {:.1} ops/bridge",
                shortcircuit::BRIDGE_OPS.load(Relaxed),
                shortcircuit::BRIDGE_OPS.load(Relaxed) as f64 / bridges as f64,
            );
        }
        // `MiniMarkGC.collection_counts` — real minor collections, not the
        // `HostNurseryClock` byte quantum. `PYPYLOG=gc-minor` is not wired
        // on this example; the count is the stats line beside the bridges.
        let (minors, _) = majit_gc::gc_sync::gc_op(|gc| {
            use majit_gc::GcAllocator;
            gc.collection_counts()
        });
        eprintln!("minor collections {minors}");
    }

    if bad != 0 {
        std::process::exit(1);
    }
}
