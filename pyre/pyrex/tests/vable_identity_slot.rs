//! A traced frame must never store an operand into the virtualizable identity.
//!
//! `virtualizable_boxes` is laid out `[static fields][array slots][identity]`,
//! and the trailing identity entry is the box `_nonstandard_virtualizable`
//! compares every later vable op against. A store that lands on it does not
//! corrupt one value — it renames the standard virtualizable, after which the
//! compiled trace writes `PyFrame` fields into whatever object the store
//! carried and dereferences that object's `locals_cells_stack_w`.
//!
//! The reachable producer was STORE_DEREF: its codewriter lowering read the
//! cell with the LOAD_FAST *opcode* (a `getarrayitem_vable_r` plus a
//! `pushvalue`) while its own operand still occupied the top of the stack, so
//! the scratch push addressed one slot past `co_stacksize` — exactly the
//! identity entry. A class body is the shape that reaches it with no headroom:
//! `class C: def m(self): pass` compiles to `co_stacksize=1` with a single
//! `__classdict__` cellvar, and `STORE_DEREF __classdict__` runs with the
//! `LOAD_LOCALS` result on the stack.
//!
//! The failure is a SIGSEGV inside JIT-generated code with nothing on stdout or
//! stderr, so this test scores the **exit status**. A harness that only
//! inspected output would call the crash a pass.
//!
//! The trip count has to clear the function-entry trace threshold (~1619), so
//! it is deliberately well above it rather than tuned to it.
//!
//! Do not invoke Cargo from this test: the parent `cargo test` holds the target
//! directory lock. `CARGO_BIN_EXE_pyre-dynasm` supplies the binary Cargo already
//! built for this integration test.

// `pyre-dynasm` is `required-features = ["dynasm"]`, so under
// `--no-default-features` the binary does not exist and `CARGO_BIN_EXE_…` is
// unset — which would be a compile error rather than a skip. Compile this file
// to nothing in that configuration instead.
#![cfg(feature = "dynasm")]

use std::io::Write;
use std::process::Command;

const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

/// Run `source` through the dynasm binary and return `(exit code, stdout)`.
///
/// The exit code is `None` for a signal death — which is the whole point of
/// this test, so it is reported rather than unwrapped.
fn run(name: &str, source: &str) -> (Option<i32>, String) {
    let dir =
        std::env::temp_dir().join(format!("pyre-vable-identity-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join(format!("{name}.py"));
    let mut file = std::fs::File::create(&path).expect("scratch script");
    file.write_all(source.as_bytes()).expect("write script");
    drop(file);

    let out = Command::new(PYRE)
        .arg(&path)
        .output()
        .expect("run pyre-dynasm");
    let _ = std::fs::remove_dir_all(&dir);
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

/// A class body carrying a method, entered often enough to be compiled as a
/// function-entry trace.
const CLASS_BODY_IN_LOOP: &str = "\
for i in range(3000):
    class C:
        def m(self):
            return 41 + 1
print(C.__name__, C().m())
";

/// The same cell store outside a class body, so a regression that reaches
/// STORE_DEREF generally — not only through `__classdict__` — is separable.
const CELL_STORE_IN_LOOP: &str = "\
def outer(n):
    total = 0
    for i in range(n):
        x = i * 2
        def inner():
            return x
        total += inner()
    return total
print(outer(3000))
";

#[test]
fn class_body_in_a_loop_survives_being_compiled() {
    let (code, stdout) = run("class_body", CLASS_BODY_IN_LOOP);
    assert_eq!(
        code,
        Some(0),
        "a compiled class body must exit cleanly; `None` is a signal death \
         (the identity-slot clobber segfaults inside JIT code and prints nothing). \
         stdout: {stdout:?}",
    );
    assert_eq!(stdout.trim(), "C 42");
}

#[test]
fn cell_store_in_a_loop_survives_being_compiled() {
    let (code, stdout) = run("cell_store", CELL_STORE_IN_LOOP);
    assert_eq!(
        code,
        Some(0),
        "a compiled STORE_DEREF loop must exit cleanly. stdout: {stdout:?}",
    );
    assert_eq!(stdout.trim(), "8997000");
}

/// `reached_loop_header` continue-tracing nulls the dead `locals_cells_stack_w`
/// tail (`null_vable_dead_array_tail`). That store is
/// `_opimpl_setarrayitem_vable`: box then `synchronize_virtualizable_at`, so
/// the tracing heap agrees before the next `getfield_vable_r`.
#[test]
fn loop_header_after_call_ex_keeps_dead_stack_tail_synchronized() {
    let program = "\
import pypyjit
pypyjit.set_param(\"function_threshold=1\")
def f():
    pass
def run():
    func, args, kwargs = f, (), {}
    func(*args, **kwargs)
    n = 0
    while n < 1:
        n += 1
run()
print(\"ok\")
";
    let mut command = Command::new(PYRE);
    for flag in [
        "MAJIT_STATS",
        "MAJIT_LOG",
        "PYRE_MC_DIAG",
        "PYRE_GC_DIAG",
        "PYRE_FBW_SPEC_CENSUS",
        "PYRE_CELL_CENSUS",
        "PYRE_FBW_DEPTH_CENSUS",
    ] {
        command.env_remove(flag);
    }
    let output = command
        .args(["-c", program])
        .output()
        .expect("run pyre-dynasm");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "CALL_FUNCTION_EX then a later loop_header must keep the dead stack \
         tail synchronized; a box-only CONST_NULL trips \
         check_synchronized_virtualizable. stdout: {stdout:?} stderr: {stderr}"
    );
    assert_eq!(stdout.trim(), "ok");
    assert!(stderr.is_empty(), "{stderr}");
}
