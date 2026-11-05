//! A value-position `match` whose arms name constants.
//!
//! syn parses `TAG_A => ..` and `other => ..` as the same `Pat::Ident`, and
//! `lower_match_value` read every such arm as the catch-all binding. Three
//! constant arms therefore produced NO guarded arms and a `default_arm` set to
//! whichever came last, so the jitcode computed that one arm for every
//! discriminant — while the concrete interpreter, which is real Rust, stayed
//! right. The trace was wrong and only a warm-versus-cold answer could say so.
//!
//! The sibling `lower_dispatch_chain` never had this: it reads its arms with
//! `extract_pat_value_tokens`, which emits `#path as i64` for the user crate to
//! resolve. `lower_match_value` now reads them the same way, and treats a
//! `Pat::Ident` as a binding only when its name holds a lower-case letter.
//!
//! The answer is the subject; the guard census is the denominator. A machine
//! that stopped lowering the arm altogether would agree with the interpreter
//! too — by being the interpreter.
//!
//! The compiled loop, measured:
//!
//! ```text
//! IntAnd GuardValue IntEq GuardTrue IntAdd IntAdd IntSub IntIsTrue GuardTrue Jump
//! ```
//!
//! `IntEq` + `GuardTrue` is the arm the trace took, guarded. Under the old
//! lowering there was no comparison in the body at all. The tags are given
//! non-zero values so that comparison stays spelled `IntEq`: against zero the
//! optimizer rewrites it to `IntIsZero` and the census would have to name both.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use majit_ir::OpCode;
use majit_metainterp::JitDriver;
use majit_metainterp::virt_array::VirtArray;

static COMPILES: AtomicUsize = AtomicUsize::new(0);
static COMPILED: Mutex<Vec<OpCode>> = Mutex::new(Vec::new());

/// Distinct per arm and distinct from every tag, so a sum can only come out
/// right if each tag reached its own arm.
const TAG_A: i64 = 11;
const TAG_B: i64 = 22;
const TAG_C: i64 = 33;

/// The last constant arm. Under the old lowering this was the value every tag
/// produced in the trace.
const VALUE_C: i64 = 500;

#[majit_macros::jit_inline]
fn classify(tag: i64) -> i64 {
    match tag {
        TAG_A => 5i64,
        TAG_B => 50i64,
        TAG_C => VALUE_C,
        _ => 9000i64,
    }
}

struct TagState {
    tags: VirtArray<i64>,
    pos: i64,
    acc: i64,
    ticks: i64,
}

pub type Bytecode = [u8];

const OP_CLASSIFY: u8 = 1;
const OP_TICK: u8 = 2;
const OP_HALT: u8 = 3;
const PROGRAM: [u8; 3] = [OP_CLASSIFY, OP_TICK, OP_HALT];

/// Four tags, cycled: every arm including the `_` default is reached.
const TAGS: [i64; 4] = [TAG_A, TAG_B, TAG_C, 7];

const TICKS: i64 = 400;

#[majit_macros::jit_interp(
    state = TagState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        tags: [int; virt],
        pos: int,
        acc: int,
        ticks: int,
    },
    calls = { classify => inline_int },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_tags(program: &Bytecode, threshold: u32, ticks: i64) -> i64 {
    let mut driver: JitDriver<TagState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, _after, opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
        *COMPILED.lock() = opcodes.to_vec();
    });
    let mut pc: usize = 0;
    let mut state = TagState {
        tags: VirtArray::from_slice(&TAGS),
        pos: 0i64,
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
            OP_CLASSIFY => {
                let i = state.pos & 3i64;
                let t = state.tags[i as usize];
                state.acc += classify(t);
                state.pos += 1i64;
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

/// What the interpreter alone computes, spelled independently of `classify`.
fn expected() -> i64 {
    let per_tag = [5i64, 50, VALUE_C, 9000];
    let full = TICKS / 4;
    let rest = TICKS % 4;
    let mut total = full * per_tag.iter().sum::<i64>();
    for (i, v) in per_tag.iter().enumerate() {
        if (i as i64) < rest {
            total += v;
        }
    }
    total
}

#[test]
fn the_traced_match_sends_each_tag_to_its_own_arm() {
    let cold = dispatch_tags(&PROGRAM, u32::MAX, TICKS);
    assert_eq!(cold, expected(), "the cold arm states the answer");

    let warm = dispatch_tags(&PROGRAM, 8, TICKS);
    assert_eq!(
        warm, cold,
        "the compiled match disagreed with the interpreter. \
         {} per pass would be every tag taking the last constant arm",
        VALUE_C,
    );
    assert!(
        COMPILES.load(Ordering::Relaxed) > 0,
        "nothing compiled, so the warm answer above was the interpreter's too",
    );
}

/// The denominator: the arms are in the trace as guarded comparisons.
#[test]
fn the_constant_arms_are_guarded_in_the_trace() {
    let _ = dispatch_tags(&PROGRAM, 8, TICKS);
    let body = COMPILED.lock().clone();
    assert!(
        !body.is_empty(),
        "no compiled loop was recorded, so this census is vacuous",
    );
    let eqs = body.iter().filter(|op| **op == OpCode::IntEq).count();
    assert!(
        eqs > 0,
        "the match lowered to no comparison at all, so one arm was taken \
         unconditionally: {body:#?}",
    );
}
