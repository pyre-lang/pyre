//! The bridge-carrier drain must decide "too deep to compile" before it runs
//! the callee, not after.
//!
//! `drive_bridge_carrier_walk` drives a reconstructed inline callee as a
//! sub-walk and keeps only a clean `SubReturn`. Every other outcome falls to an
//! abort tail whose non-adopted leg calls `fbw_store_journal_rollback()`. That
//! rollback restores the journals and nothing else: a residual call's heap
//! writes move the effect odometer and push no journal entry, so they stand
//! while the blackhole replays the region that produced them, and the replay
//! applies them a second time.
//!
//! The carrier-depth cap is one predicate that used to be read *after* the
//! sub-walk, in `want_compile`. A carrier over the cap therefore ran the callee
//! and then took that rollback. Before recursive carriers were separated from
//! pyre's generic call-chain cap, lowering the cap made this program reach that
//! arm and corrupt its accumulator rather than crash:
//!
//! ```text
//! PYRE_FBW_MULTIFRAME_DEPTH=1  ->  acc: got 262173, want 578965
//! ```
//!
//! `PYRE_FBW_MULTIFRAME_DEPTH` is a supported knob (`fbw_state.rs`, clamped
//! 1..7). PyPy `_opimpl_recursive_call` bounds recursion only with
//! `max_unroll_recursion`, so recursive carriers now bypass pyre's additional
//! generic cap. This fixture pins both that bypass and the original safety
//! property: no abort may roll back over effects the drain already executed.
//!
//! Do not invoke Cargo from this test: the parent `cargo test` holds the target
//! directory lock. `CARGO_BIN_EXE_pyre-dynasm` supplies the binary Cargo already
//! built for this integration test.

// `pyre-dynasm` is `required-features = ["dynasm"]`, so under
// `--no-default-features` the binary does not exist and `CARGO_BIN_EXE_…` is
// unset -- a compile error rather than a skip. Compile this file to nothing in
// that configuration instead.
#![cfg(feature = "dynasm")]

use std::path::PathBuf;
use std::process::{Command, Output};

const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

/// Recursion whose inlined callee escapes its own frame, so the guard that
/// fails carries a multi-frame inline chain and the drain reconstructs it.
///
/// `sys._getframe(0)` forces the callee's virtualizable; the `n > 300` arm and
/// the recursion are both load-bearing, and `acc` is checked against the value
/// the interpreter produces, so a replayed store shows up as a wrong total
/// rather than as a crash.
const RECURSE: &str = r#"
import sys

MOD = 1000003
ROUNDS = 9000
ESCAPE_ABOVE = 300


def rec(n, seen):
    if n <= 1:
        return n
    if n % 2 == 0:
        r = (rec(n // 2, seen) * 3 + 7) % MOD
    else:
        r = (rec(n - 1, seen) + n * 5) % MOD
    if n > ESCAPE_ABOVE:
        seen[0] += 1
        frame = sys._getframe(0)
        if frame.f_lineno < 0:
            seen[1] += 1
        if r != r:
            seen[1] += 1
    return r


acc = 0
seen = [0, 0]
for i in range(1, ROUNDS + 1):
    n = (i * 37) % 211 + 2
    if i > ROUNDS - 2000:
        n = n * 31 + 1
    acc = (acc + rec(n, seen)) % MOD

print(f"acc={acc}")
print(f"escaped={seen[0]}")
print(f"disagreed={seen[1]}")
"#;

/// The accumulator the interpreter computes. Any tier that answers differently
/// has lost or repeated a store.
const WANT_ACC: &str = "acc=578965";

fn run(env: &[(&str, &str)]) -> Output {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push("bridge_carrier_depth_decline.py");
    std::fs::write(&path, RECURSE).expect("write the program under test");

    let mut cmd = Command::new(PYRE);
    cmd.arg(&path);
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.output().expect("spawn pyre-dynasm")
}

/// An assertion here fails in a subprocess, so the parent has to quote the
/// child's output or the reason is lost.
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

fn assert_answers(what: &str, out: &Output) {
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{}", report(what, out));
    assert!(
        stdout.contains(WANT_ACC),
        "{what} answered wrong\n{}",
        report(what, out)
    );
    assert!(
        stdout.contains("disagreed=0"),
        "{what} read a local back changed\n{}",
        report(what, out)
    );
    // Without the escaping arm the program never builds a carrier and the test
    // would pass while measuring nothing.
    assert!(
        !stdout.contains("escaped=0"),
        "{what} never escaped a frame, so it exercises no carrier\n{}",
        report(what, out)
    );
}

#[test]
fn the_answer_is_the_same_at_every_carrier_depth_cap() {
    assert_answers("default depth cap", &run(&[]));
    // The cap's minimum. Reading the depth after the sub-walk answered 262173
    // here, because the over-cap carrier ran the callee and then rolled back
    // over stores no journal held.
    assert_answers(
        "PYRE_FBW_MULTIFRAME_DEPTH=1",
        &run(&[("PYRE_FBW_MULTIFRAME_DEPTH", "1")]),
    );
}

#[test]
fn a_recursive_carrier_bypasses_the_generic_depth_decline() {
    // Non-vacuity: `the_answer_is_the_same_at_every_carrier_depth_cap` would
    // also pass if the generic cap silently stopped applying to this program.
    // The lowered cap must not record `OverMultiframeDepth`, and the escape
    // arm plus the interpreter accumulator still prove the recursive body ran.
    // `census_dump` reprints the whole map on every record, so the count is
    // the value after the colon, never the number of occurrences.
    let out = run(&[
        ("PYRE_FBW_MULTIFRAME_DEPTH", "1"),
        ("PYRE_FBW_DEBUG_ABORT", "1"),
        ("MAJIT_STATS", "1"),
    ]);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let declines = text
        .lines()
        .filter_map(|l| l.strip_prefix("[fbw-census] P2Drain::OverMultiframeDepth: "))
        .filter_map(|n| n.trim().parse::<u64>().ok())
        .max()
        .unwrap_or(0);
    assert_eq!(
        declines,
        0,
        "a recursive carrier was rejected by pyre's generic depth cap\n{}",
        report("PYRE_FBW_MULTIFRAME_DEPTH=1 census", &out)
    );
    // `_getframe().f_lineno` now reads the compiled last_instr store, so this
    // fixture no longer forces a multi-frame blackhole terminal. Portal
    // interpret may compile the recursive callee at function entry
    // (`fbw_walks=0`). The depth-cap bypass is the `OverMultiframeDepth: 0`
    // count above; either FBW adopted the terminal or interpret compiled.
    let adopted = text
        .lines()
        .any(|l| l.starts_with("[fbw-blackhole] adopted multi-frame terminal"));
    let compiled = text
        .lines()
        .any(|l| l.contains("loops_compiled=") && !l.contains("loops_compiled=0"));
    assert!(
        adopted || compiled,
        "neither the FBW drain nor portal interpret handled the recursive carrier\n{}",
        report("PYRE_FBW_MULTIFRAME_DEPTH=1 census", &out)
    );
    assert_answers("PYRE_FBW_MULTIFRAME_DEPTH=1 census", &out);
    // The rollback the arm exists to avoid: an abort that ran effects and found
    // no image to adopt.
    let dirty = text
        .lines()
        .filter(|l| l.starts_with("[p2-drain-abort]"))
        .filter(|l| l.contains("adopted=false") && !l.contains("effects=0"))
        .count();
    assert_eq!(
        dirty,
        0,
        "a drain rolled back over executed effects\n{}",
        report("PYRE_FBW_MULTIFRAME_DEPTH=1 census", &out)
    );
}
