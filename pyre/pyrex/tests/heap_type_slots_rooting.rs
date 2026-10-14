//! A slotted subclass of a freshly created heap type holds that base across
//! `type_dict_store` (member inserts, `__dict__` / `__weakref__` descriptors).
//! Those stores grow the type dict and collect; the base is nursery-born, so
//! `create_all_slots` has to reread it from its shadow-stack slot.
//!
//! `PYPY_GC_NURSERY=65536` makes a minor collection land inside that window
//! on a host whose default nursery would otherwise hold the whole run.

#![cfg(feature = "dynasm")]

use std::path::PathBuf;
use std::process::Command;

const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

const PROGRAM: &str = "\
import _weakref
ok = 0
i = 0
while i < 40:
    class Base:
        __slots__ = ()
    names = ['s%d' % j for j in range(40)]
    class C(Base):
        __slots__ = names + ['__dict__', '__weakref__']
    if C.__base__ is not Base:
        raise SystemExit('base')
    if not hasattr(C, '__dict__') or not hasattr(C, '__weakref__'):
        raise SystemExit('flags')
    c = C()
    c.s0 = i
    c.extra = i + 1
    if c.s0 != i or c.__dict__['extra'] != i + 1:
        raise SystemExit('slots')
    r = _weakref.ref(c)
    if r() is not c:
        raise SystemExit('weakref')
    ok += 1
    i += 1
print(ok)
";

fn run(label: &str, env: &[(&str, &str)]) -> String {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push(format!("heap_type_slots_rooting_{label}.py"));
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
fn default_nursery_heap_type_slots() {
    assert_eq!(run("default", &[]), "40");
}

#[test]
fn small_nursery_heap_type_slots() {
    assert_eq!(run("64k", &[("PYPY_GC_NURSERY", "65536")]), "40");
}
