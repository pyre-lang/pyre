//! A bounded heap surfaces as a `MemoryError` before it aborts.
//!
//! incminimark.py's module docstring states the `PYPY_GC_MAX` policy as a
//! ladder: the GC *"will first collect more often, then raise an RPython
//! MemoryError, and if that is not enough, crash the program with a fatal
//! error"*. The middle rung is the only one a program can act on, and
//! `major_collection_step` is where upstream raises it.
//!
//! Pyre reaches that rung differently, and this test exists because the
//! difference is invisible to any unit test. Upstream's `raise MemoryError`
//! unwinds out of whichever driver ran the collection; pyre's collections are
//! mostly driven from the dispatch-loop safepoint, which returns `()` and
//! cannot raise, so the collector defers the exception through the eval-breaker
//! word and the dispatch loop raises it. Only an end-to-end run exercises that
//! hand-off — a collector unit test can assert the flag and a dispatch-loop
//! unit test can assert the raise, and both can pass while nothing joins them.
//!
//! What is asserted, and what is deliberately not. The middle rung's promise is
//! that the limit *becomes a Python exception*; it is not a promise that the
//! program survives one. After the first breach the live set is by definition
//! at the limit, so anything the program does next can reach the rung below —
//! upstream says as much where it raises, and aborts on the second arrival for
//! exactly that reason. So the exception itself is the observable, and nothing
//! downstream of it is: no `except` clause, no marker the program prints, no
//! particular exit status.
//!
//! That rules out running the program as `-c`, and the reason is worth
//! recording because two CI rounds were spent on it. The exception reaches the
//! interpreter's own traceback printer, which is a handler like any other:
//! `eprint_exception` renders the whole report into a buffer and writes nothing
//! until it is complete. For a `-c` program the filename is `<string>`, so
//! `frame_source_line` falls past its filesystem branch into
//! `read_registered_source_line`, which calls app-level `linecache` — a nested
//! dispatch loop whose first safepoint drives a whole major cycle over a heap
//! the breach left at its limit. The abort unwinds out of the half-built buffer
//! and stderr receives nothing at all. Read from a real file the same program
//! prints its report, because `read_source_line` answers from the filesystem
//! and no Python runs between the raise and the write. Measured on both, while
//! both raise, deliver, and reach the printer identically.
//!
//! The stderr codec is the same trap wearing different clothes, and it is why
//! every arm here names `PYTHONIOENCODING`. A built-in encoder renders the
//! report in Rust; a legacy code page is an app-level `encodings.*` module, so
//! encoding the buffer *is* a nested dispatch loop, with the same safepoint and
//! the same major cycle over the same full heap. Measured on this shape, the
//! report survives exactly when the encoder is built in, and utf8_mode decides
//! nothing beyond which encoder that leaves:
//!
//! ```text
//!   stderr cp949, utf8_mode 0     no report, fatal rung only
//!   stderr cp949, utf8_mode 1     no report, fatal rung only
//!   stderr utf-8, utf8_mode 0     MemoryError printed
//!   stderr ascii, utf8_mode 0     MemoryError printed
//! ```
//!
//! So the arms pin utf-8 for the same reason they name a file rather than
//! `-c`: to leave the middle rung as the only thing the run can fail at. What
//! it does not do is make the gap go away — a Windows host on its ANSI code
//! page still cannot print this traceback, and nothing here claims otherwise.
//!
//! An `except MemoryError` in the program is no better, and that is what the
//! second round established: `os._exit` is about as small as a handler gets,
//! and it still lost the race on two of three hosts, because the exception
//! object's own allocation re-arms the collection request and the next
//! safepoint breaches again before the handler's first call returns.
//!
//! `--heapsize` rather than `PYPY_GC_MAX` because it also covers the launcher
//! plumbing (`take_heapsize_option` -> `gc_set_max_heap_size`), which the env
//! var bypasses.
//!
//! Do not invoke Cargo from this test: the parent `cargo test` holds the target
//! directory lock. `CARGO_BIN_EXE_pyre-dynasm` supplies the binary Cargo already
//! built for this integration test.

// `pyre-dynasm` is `required-features = ["dynasm"]`, so under
// `--no-default-features` the binary does not exist and `CARGO_BIN_EXE_…` is
// unset — which would be a compile error rather than a skip. Compile this file
// to nothing in that configuration instead.
#![cfg(feature = "dynasm")]

use std::path::PathBuf;
use std::process::{Command, Output};

/// The dynasm-backend `pyre` binary cargo built for this test.
const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

/// 96 MiB — comfortably above what interpreter startup needs, so a breach is
/// the program's doing, and far below what `ROUNDS` objects need.
const HEAP_LIMIT: usize = 96 * 1024 * 1024;

/// 8 KiB of items per object.
const WORDS: usize = 1024;

/// Allocations attempted: `ROUNDS * WORDS * 8` = 200 MiB, twice the limit, so
/// the limited arms breach well before the end and the unbounded control still
/// finishes in under a second. Bounded rather than `while True` so an arm that
/// never raises terminates with a diagnosis instead of hanging.
const ROUNDS: usize = 25_000;

/// 160 KiB of items per object, which puts each one over the collector's
/// `large_object_threshold` of `(16384 + 512) * 8` bytes and so onto a
/// different allocation path from `WORDS` — see the large-object test.
const WORDS_LARGE: usize = 20_000;

/// `ROUNDS_LARGE * WORDS_LARGE * 8` = 320 MiB, again well past the limit.
const ROUNDS_LARGE: usize = 2_000;

/// Keeps every object reachable, so the live set really does grow — a dead one
/// would be reclaimed and the limit never reached.
///
/// Nothing catches the `MemoryError`: the traceback the runtime prints for it
/// is the report, and it is written by the runtime before any further Python
/// runs. See the module docs for why a handler cannot be relied on here.
const GROW: &str = "\
data = []
for _ in range(ROUNDS):
    data.append([0] * WORDS)
print('completed')
";

/// [`run_shape`] at the small shape, which is what every arm but the
/// large-object one wants.
fn run(env: &[(&str, &str)], args: &[&str]) -> Output {
    run_shape(env, args, ROUNDS, WORDS)
}

/// Run [`GROW`] at the given shape and collect the child's output. `env` and
/// `args` select the dispatch loop and the heap limit.
///
/// The program is written to a file and named by path rather than passed with
/// `-c`, because a `<string>` filename sends the traceback printer through
/// app-level `linecache` — see the module docs. `CARGO_TARGET_TMPDIR` is
/// Cargo's own scratch directory for integration tests, and each call takes a
/// fresh name: the rust test threads still race the writes, so a shared name
/// would let one arm read a file another is still writing.
fn run_shape(env: &[(&str, &str)], args: &[&str], rounds: usize, words: usize) -> Output {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    static CHILD: std::sync::Mutex<()> = std::sync::Mutex::new(());

    let src = GROW
        .replace("ROUNDS", &rounds.to_string())
        .replace("WORDS", &words.to_string());
    let serial = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push(format!("heap_limit_{rounds}_{words}_{serial}.py"));
    std::fs::write(&path, &src).expect("write the program under test");

    // One child at a time: four debug `pyre-dynasm` processes on a 7 GiB
    // runner make `malloc_typed` of the MemoryError singleton
    // (`compile.py` `memory_error`) return null, and `handle_alloc_error`
    // skips the MiniMark middle rung. `PYRE_MAX_MEMORY=0` leaves only
    // `--heapsize` as the bound; the default 8 GiB process ceiling sits
    // above the runner and never refuses before host malloc fails.
    let _one_child = CHILD.lock().unwrap_or_else(|e| e.into_inner());
    let mut cmd = Command::new(PYRE);
    cmd.args(args);
    cmd.arg(&path);
    // Ahead of `env`, so an arm can still name its own.
    cmd.env("PYTHONIOENCODING", "utf-8");
    cmd.env("PYRE_MAX_MEMORY", "0");
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.output().expect("spawn pyre-dynasm")
}

/// A failure message carrying the child's status, its stdout, and the tail of
/// its stderr — an assertion here fails in a subprocess, so the parent has to
/// quote it or the reason is lost.
fn report(what: &str, out: &Output) -> String {
    format!(
        "{what} exited {:?}\n--- stdout ---\n{}\n--- stderr tail ---\n{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .rev()
            .take(20)
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

/// Assert a `--heapsize` arm reached the middle rung: the loop did not finish,
/// and the limit was reported as a Python exception rather than only as the
/// fatal rung below it.
fn assert_caught(what: &str, out: &Output) {
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("completed"),
        "{}",
        report(
            &format!("{what}: the limited run was not bounded at all"),
            out
        )
    );
    // `MemoryError` and not merely `Traceback`: the fatal rung below prints
    // `using too much memory, aborting`, and a run that reaches only that one
    // has skipped the rung this test is about. The two can both appear — the
    // report is written first, and whatever the process does afterwards on a
    // heap still at its limit is not this test's business.
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("MemoryError"),
        "{}",
        report(
            &format!("{what}: the limit never became a Python exception"),
            out
        )
    );
}

/// The control. Without a limit the same program runs to completion, so a
/// `MemoryError` in the limited arms is attributable to the limit and not to
/// the allocation volume, the host's memory, or the script.
#[test]
fn the_same_program_completes_when_the_heap_is_unbounded() {
    let out = run(&[], &[]);
    assert!(out.status.success(), "{}", report("control", &out));
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("completed"),
        "{}",
        report("control did not finish", &out)
    );
}

/// The middle rung, on the JIT dispatch loop (the default).
///
/// Reaching it is the whole point: before the breach was routed to a dispatch
/// loop, this run's only sign of the limit was
/// `out_of_memory("using too much memory, aborting")` on the *second* breach,
/// with the rung that names a Python exception skipped entirely.
#[test]
fn a_bounded_heap_reports_a_memory_error_rather_than_only_aborting() {
    let limit = HEAP_LIMIT.to_string();
    let out = run(&[], &["--heapsize", &limit]);
    assert_caught("jit loop", &out);
}

/// Objects past `large_object_threshold` are not exempt from the limit.
///
/// They take a different allocation path — `alloc_with_type_slow` hands them
/// to `alloc_in_oldgen_clear` and returns, ahead of the point where a nursery
/// allocation would consult `oom_pending` — so "the policy is enforced" does
/// not follow from the two tests above, which only ever allocate under the
/// threshold. It holds for a different reason, and that reason is worth
/// pinning: the breach is detected in the collector, by the safepoint-driven
/// collection that observes the old generation growing, rather than in an
/// allocator's return path. Routing the exception through the eval-breaker
/// word is what makes that detection point able to report at all, so this
/// shape would regress with the small one if the channel were removed —
/// but it would also regress alone if delivery were moved back into the
/// nursery allocator, which is the case the tests above cannot see.
///
/// Both dispatch loops in one test: the seam they share is already covered
/// per-loop above, and what is new here is only the allocation path.
#[test]
fn objects_over_the_large_object_threshold_are_bounded_too() {
    // Its own control, for the reason the small shape has one, plus one it
    // does not: this shape asks the host for 320 MiB, and a host that cannot
    // supply that would fail the assertions below for a reason that has
    // nothing to do with `--heapsize`.
    let control = run_shape(&[], &[], ROUNDS_LARGE, WORDS_LARGE);
    assert!(
        control.status.success() && String::from_utf8_lossy(&control.stdout).contains("completed"),
        "{}",
        report("large objects, unbounded control did not finish", &control)
    );

    let limit = HEAP_LIMIT.to_string();
    for env in [&[][..], &[("PYRE_JIT", "0")][..]] {
        let out = run_shape(env, &["--heapsize", &limit], ROUNDS_LARGE, WORDS_LARGE);
        let what = if env.is_empty() {
            "large objects, jit loop"
        } else {
            "large objects, plain loop"
        };
        assert_caught(what, &out);
    }
}

/// The same, on the plain evaluator.
///
/// The two dispatch loops carry the delivery separately, so a fix landing in
/// one and not the other reads as green here unless both are run. `PYRE_JIT=0`
/// selects the plain one.
#[test]
fn the_plain_evaluator_reports_it_too() {
    let limit = HEAP_LIMIT.to_string();
    let out = run(&[("PYRE_JIT", "0")], &["--heapsize", &limit]);
    assert_caught("plain loop", &out);
}
