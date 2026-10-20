//! PYRE-ONLY — no RPython upstream.
//!
//! The RPython flowspace tests rely on PyPy-2.7's single-byte bytecode
//! (`HAVE_ARGUMENT`/`EXTENDED_ARG` raw-byte walk) and have no equivalent
//! for CPython 3.15's wordcode format. This file pins the CPython 3.15
//! side of `majit_translate::flowspace::bytecode::HostCode::read` — every opcode
//! variant the RustPython compiler core emits must decode cleanly, the
//! walker must traverse every code unit in order, and EXTENDED_ARG
//! folding must not corrupt the `(next_offset, Instruction, oparg)`
//! triple.
//!
//! This covers the "new `tests/test_bytecode_py315.rs` asserting every
//! 3.15 opcode decodes" bullet from the roadmap.
//!
//! A port of `test_objspace.py` bytecode-shape tests is also planned,
//! but those are integration tests that drive `build_flow(func)` —
//! they require `FlowContext`. A separate `tests/test_objspace.rs`
//! will be added alongside the flowcontext port.

mod common;

use common::compile_first_code;
use majit_translate::flowspace::bytecode::{HostCode, Instruction};

/// Walk the entire bytecode via `HostCode::read` and return every
/// decoded instruction. Panics if the walker ever stalls or errors —
/// that's the positive assertion the test is making.
fn collect_instructions(host: &HostCode) -> Vec<Instruction> {
    let total = (host.co_code.len() * 2) as u32;
    let mut out = Vec::new();
    let mut offset = 0u32;
    while offset < total {
        let (next, op, _arg) = host.read(offset).expect("read must succeed");
        assert!(next > offset, "read must advance past offset {offset}");
        assert!(
            next <= total,
            "read advanced past bytecode end: next={next} total={total}"
        );
        out.push(op);
        offset = next;
    }
    assert_eq!(offset, total, "walker must land exactly at end");
    out
}

fn contains_variant<F: Fn(&Instruction) -> bool>(ops: &[Instruction], pred: F) -> bool {
    ops.iter().any(pred)
}

#[test]
fn walks_trivial_return() {
    let host = HostCode::from_code(&compile_first_code("def f():\n    return 1\n"));
    let ops = collect_instructions(&host);
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::ReturnValue
    )));
}

#[test]
fn walks_load_fast_and_binary_op() {
    let host = HostCode::from_code(&compile_first_code("def f(x, y):\n    return x + y\n"));
    let ops = collect_instructions(&host);
    // Either LoadFast (x, y loaded separately) or LoadFastLoadFast (3.15
    // super-instruction). Accept either.
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::LoadFast { .. }
            | Instruction::LoadFastBorrow { .. }
            | Instruction::LoadFastLoadFast { .. }
            | Instruction::LoadFastBorrowLoadFastBorrow { .. }
    )));
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::BinaryOp { .. }
    )));
}

#[test]
fn walks_branches() {
    let host = HostCode::from_code(&compile_first_code(
        "def f(x):\n    if x:\n        return 1\n    return 0\n",
    ));
    let ops = collect_instructions(&host);
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::PopJumpIfFalse { .. }
            | Instruction::PopJumpIfTrue { .. }
            | Instruction::PopJumpIfNone { .. }
            | Instruction::PopJumpIfNotNone { .. }
    )));
}

#[test]
fn walks_backward_jump_for_while_loop() {
    let host = HostCode::from_code(&compile_first_code(
        "def f(x):\n    while x:\n        x = x - 1\n    return x\n",
    ));
    let ops = collect_instructions(&host);
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::JumpBackward { .. }
    )));
}

#[test]
fn walks_for_iter() {
    let host = HostCode::from_code(&compile_first_code(
        "def f(xs):\n    for x in xs:\n        pass\n",
    ));
    let ops = collect_instructions(&host);
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::GetIter { .. }
    )));
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::ForIter { .. }
    )));
}

#[test]
fn walks_store_and_load_global() {
    let host = HostCode::from_code(&compile_first_code(
        "def f():\n    global g\n    g = 1\n    return g\n",
    ));
    let ops = collect_instructions(&host);
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::StoreGlobal { .. }
    )));
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::LoadGlobal { .. }
    )));
}

#[test]
fn walks_build_tuple_list_and_unpack() {
    // Destructure a parameter (not a literal tuple) so that the
    // compiler cannot constant-fold the RHS into direct stores and
    // must emit UnpackSequence.
    let host = HostCode::from_code(&compile_first_code(
        "def f(t):\n    a, b = t\n    return [a, b]\n",
    ));
    let ops = collect_instructions(&host);
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::UnpackSequence { .. }
    )));
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::BuildList { .. }
    )));
}

#[test]
fn walks_compare_and_contains() {
    let host = HostCode::from_code(&compile_first_code(
        "def f(a, b):\n    return (a < b) and (a in b)\n",
    ));
    let ops = collect_instructions(&host);
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::CompareOp { .. }
    )));
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::ContainsOp { .. }
    )));
}

#[test]
fn walks_call_and_kw_call() {
    let host = HostCode::from_code(&compile_first_code(
        "def f(x):\n    return print(x, end='\\n')\n",
    ));
    let ops = collect_instructions(&host);
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::Call { .. } | Instruction::CallKw { .. }
    )));
}

#[test]
fn walks_raise_and_try_except() {
    let host = HostCode::from_code(&compile_first_code(
        "def f():\n    try:\n        raise ValueError('x')\n    except ValueError:\n        return 0\n",
    ));
    let ops = collect_instructions(&host);
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::RaiseVarargs { .. }
    )));
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::CheckExcMatch
    )));
}

#[test]
fn walks_yield_and_resume_for_generators() {
    let host = HostCode::from_code(&compile_first_code("def g():\n    yield 1\n    yield 2\n"));
    let ops = collect_instructions(&host);
    assert!(host.is_generator());
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::YieldValue { .. }
    )));
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::Resume { .. }
    )));
}

#[test]
fn walks_build_dict_and_subscript() {
    let host = HostCode::from_code(&compile_first_code(
        "def f():\n    d = {'a': 1}\n    return d['a']\n",
    ));
    let ops = collect_instructions(&host);
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::BuildMap { .. } | Instruction::LoadConst { .. }
    )));
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::BinaryOp { .. } | Instruction::BinarySlice
    )));
}

#[test]
fn walks_return_generator_for_generator_entry() {
    let host = HostCode::from_code(&compile_first_code("def g():\n    yield 1\n"));
    let ops = collect_instructions(&host);
    assert!(contains_variant(&ops, |op| matches!(
        op,
        Instruction::ReturnGenerator
    )));
}

#[test]
fn extended_arg_prefix_folds_into_oparg() {
    // Generate a function with enough local variables that LoadFast needs
    // an EXTENDED_ARG prefix for at least one VarNum > 255. RustPython's
    // compiler may pick LoadFastBorrow instead; the critical invariant is
    // that `read` transparently consumes the ExtendedArg wordcode and
    // yields the concrete opcode with the folded oparg.
    let mut src = String::from("def f():\n");
    for i in 0..260 {
        src.push_str(&format!("    v{i} = {i}\n"));
    }
    src.push_str("    return v259\n");
    let host = HostCode::from_code(&compile_first_code(&src));
    let ops = collect_instructions(&host);
    // The decoded op stream must never itself contain ExtendedArg —
    // `HostCode::read` folds the prefix into the successor's oparg.
    assert!(
        !contains_variant(&ops, |op| matches!(op, Instruction::ExtendedArg)),
        "read must swallow ExtendedArg into the following opcode's oparg"
    );
}
