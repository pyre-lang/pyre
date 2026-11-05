//! A `#[jit_inline]` helper that calls itself.
//!
//! `codewriter.py CodeWriter.make_jitcodes` emits one JitCode per graph and
//! links callees by index; `CallControl.get_jitcode` mints and caches an empty
//! JitCode under the graph BEFORE the body is written, so a graph that calls
//! itself links to the object it is already registered under. Nothing here did
//! that: the inline lowering emitted a plain Rust call to the helper's builder,
//! so `f` calling `f` recursed until the stack was gone — at jitcode-BUILD
//! time, during warmup, before any tracing.
//!
//! It is not a caching problem. `add_sub_jitcode` pushes an `Arc<JitCode>` into
//! the CURRENT builder's own descr pool, so direct self-recursion asks for a
//! jitcode whose pool holds an `Arc` to itself. The identity now comes from
//! `Arc::new_cyclic`, and the self-edge is recorded as a `Weak` — which is what
//! keeps it from being a cycle of owning references that never drops.
//!
//! Reaching the first assertion at all is the first result: before this, the
//! process died in `dispatch_sum(..)` with a stack overflow.
//!
//! The compiled loop, measured (identical on dynasm and cranelift): four
//! `value` reads and four `next` reads in the preamble, no call of any kind.
//!
//! ```text
//! Label GetfieldGcI GetfieldGcR GetfieldGcI GetfieldGcR GetfieldGcI GetfieldGcR
//!       GetfieldGcI GetfieldGcR IntAdd IntAdd IntAdd IntAdd IntSub IntIsTrue GuardTrue
//! Label GetfieldGcR GetfieldGcI GetfieldGcR GetfieldGcI GetfieldGcR
//!       IntAdd IntAdd IntAdd IntAdd IntSub IntIsTrue GuardTrue Jump
//! ```

use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use majit_ir::OpCode;
use majit_metainterp::JitDriver;

static COMPILES: AtomicUsize = AtomicUsize::new(0);
static COMPILED: Mutex<Vec<OpCode>> = Mutex::new(Vec::new());

/// The chain the helper walks. Kept under `MAX_INLINE_DEPTH` so this fixture
/// measures the build-time defect and nothing else.
const LINKS: i64 = 4;
const TICKS: i64 = 400;

#[repr(C)]
struct Link {
    value: i64,
    next: *mut Link,
}

/// Recursive by construction: the `n > 0` arm calls this same function.
#[majit_macros::jit_inline(
    ref_params = { node: ref(Link) },
    ref_fields = { Link::next => Link },
    int_fields = { Link::value => i64 },
)]
fn sum_chain(node: usize, n: i64) -> i64 {
    if n <= 0i64 {
        0i64
    } else {
        let v = node.value;
        let rest = node.next;
        v + sum_chain(rest, n - 1i64)
    }
}

pub type Bytecode = [u8];

struct SumState {
    head: usize,
    acc: i64,
    ticks: i64,
}

const OP_SUM: u8 = 1;
const OP_TICK: u8 = 2;
const OP_HALT: u8 = 3;
const PROGRAM: [u8; 3] = [OP_SUM, OP_TICK, OP_HALT];

#[majit_macros::jit_interp(
    state = SumState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        head: ref(Link),
        acc: int,
        ticks: int,
    },
    ref_fields = { Link::next => Link },
    int_fields = { Link::value => i64 },
    calls = { sum_chain => inline_int },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_sum(program: &Bytecode, threshold: u32, head: usize, ticks: i64) -> i64 {
    let mut driver: JitDriver<SumState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
        *COMPILED.lock() = opcodes.to_vec();
    });
    let mut pc: usize = 0;
    let mut state = SumState {
        head,
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
            OP_SUM => {
                state.acc += sum_chain(state.head, LINKS);
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

/// Link `i` holds `i + 1`, so a walk that stops early or repeats a link sums
/// differently.
fn chain() -> usize {
    let mut head: *mut Link = std::ptr::null_mut();
    for i in (0..LINKS).rev() {
        head = Box::leak(Box::new(Link {
            value: i + 1,
            next: head,
        })) as *mut Link;
    }
    head as usize
}

fn expected() -> i64 {
    TICKS * (1..=LINKS).sum::<i64>()
}

#[test]
fn a_self_recursive_inline_helper_builds_and_runs() {
    // Getting here is the first assertion; the old lowering never returned.
    let cold = dispatch_sum(&PROGRAM, u32::MAX, chain(), TICKS);
    assert_eq!(cold, expected(), "the cold arm states the answer");

    let warm = dispatch_sum(&PROGRAM, 8, chain(), TICKS);
    assert_eq!(
        warm, cold,
        "the compiled recursive walk disagreed with the interpreter",
    );
    assert!(
        COMPILES.load(Ordering::Relaxed) > 0,
        "nothing compiled, so the warm answer above was the interpreter's too",
    );
}

/// The walk must actually be IN the trace.
///
/// A recursive helper that failed to link would degrade to a residual call, and
/// the answer would still be right — so agreement alone does not say the
/// self-edge resolved.
#[test]
fn the_recursive_walk_is_traced_rather_than_called_out_to() {
    let _ = dispatch_sum(&PROGRAM, 8, chain(), TICKS);
    let body = COMPILED.lock().clone();
    assert!(
        !body.is_empty(),
        "no compiled loop was recorded, so this census is vacuous",
    );
    let value_reads = body.iter().filter(|op| **op == OpCode::GetfieldGcI).count();
    assert!(
        value_reads >= LINKS as usize,
        "the walk contributed {value_reads} `node.value` reads, fewer than the \
         {LINKS} links it recurses over, so it was not fully traced: {body:#?}",
    );
    assert!(
        !body.iter().any(|op| format!("{op:?}").starts_with("Call")),
        "the walk left a residual call behind, so the self-edge did not \
         resolve to a jitcode the tracer could descend into: {body:#?}",
    );
}
