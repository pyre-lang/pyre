//! A recorder value slot's concrete Ref is not a GC root. Live FrontendOps
//! are forwarded through `walk_active_trace_refs`; a dead one keeps the
//! address it had. `Trace::materialize_ops` then copies every slot through
//! `Op::set_value`, which interns `Value::Ref` (`const_ptr_table::intern` →
//! `id_or_identityhash` / `_find_shadow`). A collection inside `exec`'s
//! dict-subclass `setdefault` leaves a stale nursery word in a dead slot,
//! and compiling the warm loop hashes it as a header.
//!
//! `PYPY_GC_NURSERY=262144` is `debug_tiny_nursery` just under minsize.
//! `MAJIT_GC_NURSERY_POISON=1` fills recycled nursery with `0xAA`, whose
//! flag byte includes `GCFLAG_HAS_SHADOW`, so the stale intern panics in
//! `find_shadow` instead of copying a recycled header.
//!
//! Do not invoke Cargo from this test: the parent `cargo test` holds the
//! target directory lock. `CARGO_BIN_EXE_pyre-dynasm` supplies the binary
//! Cargo already built for this integration test.

#![cfg(all(feature = "dynasm", feature = "pyre-module"))]

use std::path::PathBuf;
use std::process::Command;

const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

/// `exec` plants `__builtins__` with `setdefault` on the caller's mapping, so
/// a dict-subclass override runs and can collect while the tracer still holds
/// dead FrontendOp `_resref` words. The warm loop is what `finish_and_compile`
/// materializes.
const PROGRAM: &str = r#"
import gc

try:
    import pypyjit
    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


class SetdefaultNS(dict):
    def setdefault(self, key, value=None):
        [{} for _ in range(200)]
        gc.collect()
        return dict.setdefault(self, key, value)


total = 0
for _ in range(200):
    namespace = SetdefaultNS()
    namespace["src"] = {"x": 1}
    exec("r = src.get('x')", namespace)
    total += namespace.get("r") or 0
if total != 200:
    raise SystemExit("warm loop")
print("OK")
"#;

fn run(label: &str, env: &[(&str, &str)]) -> String {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push(format!(
        "exec_namespace_constptr_intern_gc_roots_{label}.py"
    ));
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
fn tiny_nursery_exec_compile_does_not_intern_a_dead_value_slot() {
    assert_eq!(run("262k", TINY), "OK");
}

#[test]
fn tiny_nursery_exec_compile_does_not_intern_a_dead_value_slot_no_jit() {
    let mut env = TINY.to_vec();
    env.push(("PYRE_NO_JIT", "1"));
    assert_eq!(run("262k-nojit", &env), "OK");
}
