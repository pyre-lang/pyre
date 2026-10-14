//! A float subclass's `__float__` may allocate while `complex()` still holds
//! the method and the type. `complex_space_float` roots both across that call
//! and reloads the receiver afterwards.
//!
//! `PYPY_GC_NURSERY=64K` makes a minor collection land inside the window on
//! a host whose default nursery would otherwise hold the whole run.

#![cfg(feature = "dynasm")]

use std::path::PathBuf;
use std::process::Command;

const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

const PROGRAM: &str = "\
class F(float):
    def __float__(self):
        [object() for _ in range(200)]
        return float.__float__(self)

total = 0
i = 0
while i < 200:
    total += complex(F(1.5)).real
    i += 1
print(total)
";

fn run(label: &str, env: &[(&str, &str)]) -> String {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push(format!("complex_float_subclass_rooting_{label}.py"));
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
fn default_nursery_complex_float_subclass() {
    assert_eq!(run("default", &[]), "300.0");
}

#[test]
fn small_nursery_complex_float_subclass() {
    assert_eq!(run("64k", &[("PYPY_GC_NURSERY", "64K")]), "300.0");
}
