//! `DescrOperation.next` keeps `w_obj` as a livevar across
//! `space.lookup('__next__')` and `get_and_call_function`. A method-cache miss
//! can collect (`compute_mro`, intern, cache fill). pyre's `next` takes the
//! iterator by value; the caller's `with_roots!` reloads its own slot only, so
//! this frame has to publish the word (`push_roots` / `pop_roots`) and reread
//! it after the lookup.
//!
//! `PYPY_GC_NURSERY=262144` is `debug_tiny_nursery` just under minsize. That
//! size SIGSEGVs the product binary on this path (site `import *`, and a
//! user-class iterator after a cache-miss lookup). A harness that only names
//! the default nursery, or nursery=1, does not reach it.
//!
//! Do not invoke Cargo from this test: the parent `cargo test` holds the
//! target directory lock. `CARGO_BIN_EXE_pyre-dynasm` supplies the binary
//! Cargo already built for this integration test.

#![cfg(feature = "dynasm")]

use std::path::PathBuf;
use std::process::Command;

const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

/// A user-class iterator so `next` takes the generic `is_instance` tail
/// (`lookup_in_type_where` `__next__`) rather than a typed builtin arm.
/// The list of young objects fills the nursery so that lookup can collect.
const PROGRAM: &str = "\
class I:
    def __init__(self, n):
        self.n = n
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        v = self.i
        self.i = self.i + 1
        return v

junk = []
k = 0
while k < 3000:
    junk.append(object())
    k = k + 1

it = I(5)
total = 0
while True:
    try:
        total = total + next(it)
    except StopIteration:
        break
if total != 10:
    raise SystemExit('sum')
print('OK')
";

fn run(label: &str, env: &[(&str, &str)]) -> String {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push(format!("next_lookup_moving_gc_roots_{label}.py"));
    std::fs::write(&path, PROGRAM).expect("write the program under test");

    let mut cmd = Command::new(PYRE);
    cmd.arg(&path);
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run the program under test");
    assert!(
        out.status.success(),
        "{label}: {}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr),
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

const TINY: &[(&str, &str)] = &[
    ("PYPY_GC_NURSERY", "262144"),
    ("PYPY_GC_NURSERY_DEBUG", "1"),
    ("MAJIT_GC_NURSERY_POISON", "1"),
];

#[test]
fn tiny_nursery_next_lookup_keeps_iterator() {
    assert_eq!(run("262k", TINY), "OK");
}

#[test]
fn tiny_nursery_next_lookup_keeps_iterator_no_jit() {
    let mut env = TINY.to_vec();
    env.push(("PYRE_NO_JIT", "1"));
    assert_eq!(run("262k-nojit", &env), "OK");
}
