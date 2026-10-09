//! `set(existing_set)` copies storage through `w_set_copy_storage_from` →
//! `get_storage_copy` → `RDict::clone` → `alloc_entries`. That malloc
//! collects; both sets are nursery-born, so the copy has to write through
//! the reloaded destination.
//!
//! `PYPY_GC_NURSERY=64K` makes a minor collection land inside that window
//! on a host whose default nursery would otherwise hold the whole run.

#![cfg(feature = "dynasm")]

use std::path::PathBuf;
use std::process::Command;

const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

const PROGRAM: &str = "\
def copy_ok(src):
    dst = set(src)
    return dst == src and dst is not src and len(dst) == len(src)

s = set()
i = 0
while i < 200:
    s.add(i)
    i += 1
if not copy_ok(s):
    raise SystemExit('int')

s2 = set()
i = 0
while i < 200:
    s2.add((i,))
    i += 1
if not copy_ok(s2):
    raise SystemExit('obj')

if not copy_ok(set()):
    raise SystemExit('empty')
print('ok')
";

fn run(label: &str, env: &[(&str, &str)]) -> String {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push(format!("set_copy_storage_rooting_{label}.py"));
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
fn default_nursery_set_copy() {
    assert_eq!(run("default", &[]), "ok");
}

#[test]
fn small_nursery_set_copy() {
    assert_eq!(run("64k", &[("PYPY_GC_NURSERY", "64K")]), "ok");
}
