//! `MAJIT_FNPTR_INDIRECT` — how a call through a host-registered `fn`
//! pointer lowers, in both states of the switch.
//!
//! Its own test binary, holding exactly one `#[test]`, because the switch is
//! read from the process environment at lowering time: a `set_var` inside
//! `test_mir_frontend.rs` would race the seventeen tests libtest runs
//! alongside it. One test in one process can toggle the cause and watch the
//! instrument respond, which is the whole point of the file.

mod common;

use common::lower_function;
use common::{CORPUS_LLBC, load_llbc};
use majit_charon_reader::Llbc;
use majit_translate::model::{CallTarget, OpKind};

/// The two corpus fixtures that call through a `fn`-pointer field: the
/// direct spelling and the one-hop `Option<fn-ptr>` deref.
const FIXTURES: [&str; 2] = ["host_registry_dispatch", "host_registry_dispatch_optional"];

#[derive(Debug, Default, PartialEq, Eq)]
struct Shape {
    /// `OpKind::IndirectCall { graphs: None }` — `indirect_call` with an
    /// unknown PBC family.
    indirect_unknown_family: usize,
    /// `Call(FunctionPath["__dyn_call"])` — the placeholder.
    dyn_call: usize,
}

fn shape_of(llbc: &Llbc, name: &str) -> Shape {
    let graph = lower_function(llbc, name).expect("lowering");
    let mut shape = Shape::default();
    for b in &graph.blocks {
        for op in &b.operations {
            match &op.kind {
                OpKind::IndirectCall { graphs, args, .. } => {
                    // `None`, not `Some([])`, is load-bearing and asserted
                    // here rather than left to `is_some()`: `graphs is None`
                    // is upstream's "cannot follow the indirect call"
                    // (`call.py:105`/`137`, `jtransform.py rewrite_op_indirect_call`) and the  allow-line-citation
                    // one value that keeps every family analyzer on its top
                    // result.  `Some([])` reads as an *empty* family and
                    // collapses canraise / can_invalidate /
                    // forces_virtualizable to their bottom result.
                    assert_eq!(
                        graphs.as_deref(),
                        None,
                        "{name}: a runtime-installed callback has no \
                         statically recoverable PBC family",
                    );
                    // The callee moves out of the argument list into the op's
                    // own `funcptr` slot, so the arity is one below the
                    // `__dyn_call` spelling of the same call.
                    assert_eq!(
                        args.len(),
                        1,
                        "{name}: the callee lives in `funcptr`, not in args[0]",
                    );
                    shape.indirect_unknown_family += 1;
                }
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if segments.last().map(String::as_str) == Some("__dyn_call") => {
                    assert_eq!(
                        args.len(),
                        2,
                        "{name}: __dyn_call threads the callee as args[0]",
                    );
                    shape.dyn_call += 1;
                }
                _ => {}
            }
        }
    }
    shape
}

/// Off (the default) the call reaches the `__dyn_call` placeholder; on, it
/// reaches `IndirectCall` with no family.
///
/// The switch is opt-in rather than on-by-default for a reason recorded at
/// `front::mir::fnptr_indirect_enabled`: turning it on propagates
/// `EF_RANDOM_EFFECTS` out of the newly visible indirect calls and trips
/// `getcalldescr`'s elidable post-condition on pyre's
/// `#[elidable_cannot_raise]` bigint helpers.  This test pins the lowering
/// itself so that stays a pyre-side blocker and not a silent regression in
/// the lowering.
#[test]
fn fnptr_indirect_switch_moves_the_lowering_in_both_directions() {
    let llbc = load_llbc(CORPUS_LLBC);

    // SAFETY: `set_var` is unsafe in Rust 2024 because concurrent
    // environment mutation races.  This file's test binary holds exactly
    // one test, so nothing else in the process reads or writes the
    // environment while it runs.
    unsafe { std::env::remove_var("MAJIT_FNPTR_INDIRECT") };
    for name in FIXTURES {
        assert_eq!(
            shape_of(&llbc, name),
            Shape {
                indirect_unknown_family: 0,
                dyn_call: 1,
            },
            "{name}: default (switch unset) keeps the placeholder",
        );
    }

    unsafe { std::env::set_var("MAJIT_FNPTR_INDIRECT", "1") };
    for name in FIXTURES {
        assert_eq!(
            shape_of(&llbc, name),
            Shape {
                indirect_unknown_family: 1,
                dyn_call: 0,
            },
            "{name}: switch on lowers the callback to indirect_call",
        );
    }

    unsafe { std::env::remove_var("MAJIT_FNPTR_INDIRECT") };
}
