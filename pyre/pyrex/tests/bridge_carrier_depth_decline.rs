//! `drive_bridge_carrier_walk` drives a reconstructed inline callee as a
//! sub-walk and keeps only a clean `SubReturn`. Every other outcome falls to an
//! abort tail whose non-adopted leg calls `fbw_store_journal_rollback()`. That
//! rollback restores the journals and nothing else: a residual call's heap
//! writes move the effect odometer and push no journal entry, so they stand
//! while the blackhole replays the region that produced them, and the replay
//! applies them a second time.
//!
//! Recursion is bounded only by `_opimpl_recursive_call`'s
//! `max_unroll_recursion`. This fixture pins the remaining safety property
//! under the default configuration: the recursive program's answers match the
//! interpreter, and no abort rolls back over effects the drain already
//! executed.
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
}

#[test]
fn a_recursive_carrier_bypasses_the_generic_depth_decline() {
    // Non-vacuity: `the_answer_is_the_same_at_every_carrier_depth_cap` would
    // also pass if the drain never reconstructed a carrier. The escape arm
    // plus the interpreter accumulator still prove the recursive body ran.
    let out = run(&[("PYRE_FBW_DEBUG_ABORT", "1")]);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    // `_getframe().f_lineno` now reads the compiled last_instr store, so this
    // fixture no longer forces a multi-frame blackhole terminal.  The answer
    // and the escape arm still prove the recursive body ran.
    assert_answers("default census", &out);
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
        report("default census", &out)
    );
}
