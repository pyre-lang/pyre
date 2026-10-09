//! `list(set)` takes `EmptyListStrategy._extend_from_iterable` through
//! `w_list_try_extend_empty_from_set`, which reloads both operands after the
//! set-view lock. A small nursery makes a collection during that window
//! deterministic on a host whose default nursery would hold the whole run.

#![cfg(feature = "dynasm")]

use std::path::PathBuf;
use std::process::Command;

const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

const PROGRAM: &str = "\
s = set()
i = 0
while i < 50:
    s.add(i)
    i += 1
print(sum(list(s)))
";

fn run(label: &str, env: &[(&str, &str)]) -> String {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push(format!("list_set_extend_rooting_{label}.py"));
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

#[test]
fn default_nursery_list_from_int_set() {
    assert_eq!(run("default", &[]), "1225");
}

#[test]
fn small_nursery_list_from_int_set() {
    assert_eq!(run("64k", &[("PYPY_GC_NURSERY", "64K")]), "1225");
}
