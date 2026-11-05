//! `_immutable_fields_` on a `#[jit_interp]` machine, end to end.
//!
//! The plumbing tests next door assert that the declaration reaches the minted
//! descr.  That is necessary and not sufficient: what the declaration is FOR is
//! `OptHeap::optimize_getfield`'s fold, which fires only when the descr is
//! always-pure AND the base is a constant.  A chain of reads is where the two
//! meet — promote the root once and each declared read folds to a constant,
//! which makes the NEXT read's base constant too, so the whole walk collapses.
//!
//! So this is an A/B: two structurally identical machines over two structurally
//! identical node types, differing only in whether `left` is declared.  Without
//! the declaration each level costs its own `getfield_gc_r`; with it, none do.
//!
//! The undeclared machine is the denominator, and it is asserted FIRST. A low
//! read count is also what a census over a trace that was never recorded
//! produces, and a fixture that cannot fail in the other direction has not
//! measured the fold.
//!
//! The two optimized traces, as measured:
//!
//! ```text
//! declared: Label GetfieldGcR GuardValue GetfieldGcI IntAdd IntAdd IntSub
//!           SetfieldGc IntIsTrue GuardTrue | Label IntAdd ... Jump
//! plain:    Label GetfieldGcR GuardValue GetfieldGcR GetfieldGcR GetfieldGcI
//!           IntAdd IntAdd IntSub SetfieldGc IntIsTrue GuardTrue | Label ... Jump
//! ```
//!
//! One `GetfieldGcR` per link without the declaration; with it, only the read
//! whose base is the red `state.root` survives and the two below it are gone.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use majit_ir::OpCode;
use majit_metainterp::JitDriver;

/// The chain the machines walk.  Three links, so a walk that folds only its
/// first level is still distinguishable from one that folds all of them.
const DEPTH: usize = 3;
const TICKS: i64 = 400;

// ── The declaring node type ────────────────────────────────────────────────

/// `marked` is deliberately left out: it is the mutable state, and a fixture in
/// which every field folded would not show that the fold is selective.
#[majit_macros::jit_immutable_fields("left")]
#[repr(C)]
struct DeclaredNode {
    marked: i64,
    left: *mut DeclaredNode,
}

/// Byte-for-byte the same shape, declaring nothing.
#[repr(C)]
struct PlainNode {
    marked: i64,
    left: *mut PlainNode,
}

macro_rules! machine {
    (
        $node:ident, $state:ident, $dispatch:ident, $compiles:ident, $compiled:ident,
        $chain:ident, $expected:ident
    ) => {
        static $compiles: AtomicUsize = AtomicUsize::new(0);
        static $compiled: Mutex<Vec<OpCode>> = Mutex::new(Vec::new());

        struct $state {
            root: usize,
            acc: i64,
            ticks: i64,
        }

        #[majit_macros::jit_interp(
            state = $state,
            env = Bytecode,
            greens = [pc, program],
            state_fields = {
                root: ref($node),
                acc: int,
                ticks: int,
            },
            ref_fields = { $node::left => $node },
            int_fields = { $node::marked => i64 },
        )]
        #[allow(unused_assignments, unused_variables)]
        fn $dispatch(program: &Bytecode, threshold: u32, root: usize, ticks: i64) -> i64 {
            let mut driver: JitDriver<$state> = JitDriver::new(threshold);
            driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
                $compiles.fetch_add(1, Ordering::Relaxed);
                *$compiled.lock() = opcodes.to_vec();
            });
            let mut pc: usize = 0;
            let mut state = $state {
                root,
                acc: 0i64,
                ticks,
            };
            {
                use majit_metainterp::JitState as _;
                state
                    .build_meta(0, program)
                    .install_canonical_liveness(&mut driver);
            }

            while pc < program.len() {
                jit_merge_point!(driver, program, pc; state);
                let opcode = program[pc];
                pc += 1;
                match opcode {
                    OP_WALK => {
                        // One promote, on the FIRST link only.  `state.root` is
                        // a red ref, so that read emits whatever is declared;
                        // the promote is what makes its result a constant.
                        // Every link below it is a declared read off a constant
                        // base, which is exactly the fold's precondition — so
                        // with the declaration in place they cost nothing and
                        // without it each one emits its own reload.
                        let a = state.root.left;
                        majit_metainterp::jit::promote(a);
                        let b = a.left;
                        let c = b.left;
                        // The mutable field, read and written off the deepest
                        // node: the store the walk exists to reach.
                        let m = c.marked;
                        c.marked = m + 1i64;
                        state.acc += m;
                    }
                    OP_TICK => {
                        state.ticks -= 1i64;
                        if state.ticks != 0 {
                            can_enter_jit!(driver, 0usize, &mut state, program, || {});
                            pc = 0;
                            continue;
                        }
                    }
                    OP_HALT => break,
                    _ => break,
                }
            }
            state.acc
        }

        /// A leaked chain: the machine takes a raw address and the graph must
        /// outlive every trace that folded a node of it into a constant.
        fn $chain() -> usize {
            let mut head: *mut $node = std::ptr::null_mut();
            for _ in 0..=DEPTH {
                head = Box::leak(Box::new($node {
                    marked: 0,
                    left: head,
                })) as *mut $node;
            }
            head as usize
        }

        /// Iteration `i` reads the mark left by iteration `i - 1`.
        fn $expected() -> i64 {
            (0..TICKS).sum()
        }
    };
}

pub type Bytecode = [u8];

const OP_WALK: u8 = 1;
const OP_TICK: u8 = 2;
const OP_HALT: u8 = 3;
const PROGRAM: [u8; 3] = [OP_WALK, OP_TICK, OP_HALT];

machine!(
    DeclaredNode,
    DeclaredState,
    dispatch_declared,
    DECLARED_COMPILES,
    DECLARED_COMPILED,
    declared_chain,
    declared_expected
);
machine!(
    PlainNode,
    PlainState,
    dispatch_plain,
    PLAIN_COMPILES,
    PLAIN_COMPILED,
    plain_chain,
    plain_expected
);

fn census(ops: &[OpCode], want: OpCode) -> usize {
    ops.iter().filter(|op| **op == want).count()
}

/// Both machines must answer the same as their own interpreter first.  A trace
/// census over wrong code grades the wrong thing.
#[test]
fn both_machines_agree_warm_and_cold() {
    let cold = dispatch_declared(&PROGRAM, u32::MAX, declared_chain(), TICKS);
    let warm = dispatch_declared(&PROGRAM, 8, declared_chain(), TICKS);
    assert_eq!(cold, declared_expected(), "the cold arm states the answer");
    assert_eq!(warm, cold, "the declared machine's compiled walk disagreed");

    let cold = dispatch_plain(&PROGRAM, u32::MAX, plain_chain(), TICKS);
    let warm = dispatch_plain(&PROGRAM, 8, plain_chain(), TICKS);
    assert_eq!(cold, plain_expected());
    assert_eq!(
        warm, cold,
        "the undeclared machine's compiled walk disagreed"
    );
}

#[test]
fn a_declared_ref_field_folds_where_an_undeclared_one_reloads() {
    let _ = dispatch_plain(&PROGRAM, 8, plain_chain(), TICKS);
    assert!(
        PLAIN_COMPILES.load(Ordering::Relaxed) > 0,
        "the undeclared machine compiled nothing, so there is no denominator",
    );
    let plain = PLAIN_COMPILED.lock().clone();
    let plain_reads = census(&plain, OpCode::GetfieldGcR);
    assert!(
        plain_reads > 0,
        "the undeclared walk emitted no GetfieldGcR at all, so the census below \
         cannot tell a fold from a walk that was never traced. Trace: {plain:#?}",
    );

    let _ = dispatch_declared(&PROGRAM, 8, declared_chain(), TICKS);
    assert!(
        DECLARED_COMPILES.load(Ordering::Relaxed) > 0,
        "the declared machine compiled nothing",
    );
    let declared = DECLARED_COMPILED.lock().clone();
    let declared_reads = census(&declared, OpCode::GetfieldGcR);

    assert!(
        declared_reads < plain_reads,
        "declaring `left` immutable changed nothing: {declared_reads} reads \
         declared vs {plain_reads} undeclared. Trace: {declared:#?}",
    );
    assert_eq!(
        declared_reads, 1,
        "only the read off `state.root` should survive — its base is a red ref, \
         so nothing folds it. The two links below it are declared reads off a \
         constant base and must both be gone: {declared:#?}",
    );
}

/// The mark is the mutable state and must survive as a store.  A fixture whose
/// every read folded would also pass the census above by folding the whole
/// body away.
#[test]
fn the_undeclared_mutable_field_still_stores() {
    let _ = dispatch_declared(&PROGRAM, 8, declared_chain(), TICKS);
    let declared = DECLARED_COMPILED.lock().clone();
    assert!(
        census(&declared, OpCode::SetfieldGc) > 0,
        "`marked` is not declared; its store must remain: {declared:#?}",
    );
}
