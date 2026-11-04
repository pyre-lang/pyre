//! A field declaration that describes nothing the machine reaches.
//!
//! `int_fields` and `ref_fields` are consulted by key — `"StructType::field"`,
//! built from the *declared* type of the base an access goes through, not from
//! the runtime object. A key naming a struct no access site is typed as
//! therefore matches nothing, emits nothing, and is silently accepted.
//!
//! Both halves of that are the problem. The entry contributes no descr width,
//! so it is dead; and `field_scalar_tokens`' `const _: fn(&S) -> T` witness is
//! emitted only on a match, so the claim it makes about the field is never
//! checked either. Measured on a real consumer: 36 entries declaring a `u32`
//! field as `u8` compiled clean, while the same misdeclaration on a key that
//! *is* consulted produced 13 `E0308`s. An unconsulted entry is not merely
//! untidy — it is a false statement the build agrees to.
//!
//! `pool_arrays` already rejects a base name that resolves to no `ref(_)`
//! state field. This is that check's missing sibling, reporting rather than
//! rejecting for the reason the gate's own message gives.

use majit_metainterp::{Assembler, JitDriver};

#[repr(C)]
struct Cell {
    /// Consulted: the machine reads it through a `ref(Cell)` state field.
    count: u32,
}

pub type Bytecode = [u8];

const OP_READ: u8 = 1;
const OP_TICK: u8 = 2;
const OP_HALT: u8 = 3;

/// Never the declared type of any access base in this machine. A key naming it
/// cannot match, whatever it claims about the field.
#[expect(dead_code)] // named only by the `int_fields` key below
#[repr(C)]
struct Unreached {
    count: u32,
}

// Named as the `jit_interp` state type; the macro lowers the interpreter body
// into jitcode and never runs it, so nothing constructs the struct.
#[expect(dead_code)]
struct CellState {
    cell: usize,
    total: i64,
    ticks: i64,
}

#[majit_macros::jit_interp(
    state = CellState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        cell: ref(Cell),
        total: int,
        ticks: int,
    },
    int_fields = {
        // Consulted: `state.cell.count` is an access through `ref(Cell)`.
        Cell::count => u32,
        // Not consulted, and deliberately a lie about the width: `Unreached`
        // is never an access base, so the key never matches and the witness
        // that would reject `u8` is never emitted.
        Unreached::count => u8,
    },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_cell(program: &Bytecode, threshold: u32, cell: usize, ticks: i64) -> i64 {
    let mut driver: JitDriver<CellState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = CellState {
        cell,
        total: 0,
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
            OP_READ => {
                state.total = state.total + state.cell.count as i64;
            }
            OP_TICK => {
                state.ticks -= 1i64;
                if state.ticks != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
            }
            OP_HALT => {
                break;
            }
            _ => break,
        }
    }
    state.total
}

fn install() {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
    __prebuild_jitcode_liveness_dispatch_cell(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    let _ = __dispatch_jitcode_dispatch_cell(&mut asm, 0i64).expect("dispatch lower must succeed");
}

fn keys() -> Vec<String> {
    install();
    majit_metainterp::unconsulted_field_declarations()
        .into_iter()
        .filter(|e| e.interp == "CellState")
        .map(|e| e.key.to_string())
        .collect()
}

/// The subject, with its control beside it.
///
/// The control is not decoration: an empty report and a report of *everything*
/// look the same from the subject alone, and a key set that listed `Cell::count`
/// too would mean the consultation is never recorded rather than that this
/// declaration is unused.
#[test]
fn a_declaration_no_access_site_consults_is_reported() {
    let keys = keys();
    assert!(
        keys.iter().any(|key| key == "Unreached::count"),
        "a key naming a struct no access is typed as matched nothing and \
         emitted nothing, including the witness that would have caught its \
         `u8`; keys={keys:?}",
    );
    assert!(
        !keys.iter().any(|key| key == "Cell::count"),
        "control: `state.cell.count` is a real access through `ref(Cell)`, so \
         its declaration is consulted. If this fails the recording is firing \
         for every key and the subject above means nothing; keys={keys:?}",
    );
}

/// The gate must separate its two failures, and name the third possibility.
///
/// An unconsulted key is a stale declaration OR a live one whose only arm
/// refused to lower. The message carries the degraded arms for that reason —
/// a reader who cannot tell those apart will delete a declaration that is
/// still needed.
#[test]
fn the_gate_reports_the_declaration_and_the_degraded_arms_together() {
    install();

    let failure = std::panic::catch_unwind(|| {
        majit_metainterp::assert_no_unconsulted_field_declarations("CellState")
    })
    .expect_err("this fixture declares one unconsulted key on purpose");
    let message = panic_message(&failure);
    assert!(
        message.contains("no access site consulted"),
        "the gate must fail on the declaration; got {message:?}",
    );
    assert!(
        message.contains("arms degraded"),
        "…and must show the degraded arms beside it, since a key used only in \
         a refused arm is unconsulted for a different reason; got {message:?}",
    );

    let uninstalled = std::panic::catch_unwind(|| {
        majit_metainterp::assert_no_unconsulted_field_declarations("NoSuchState")
    })
    .expect_err("a machine that was never installed cannot be graded");
    let uninstalled = panic_message(&uninstalled);
    assert!(
        uninstalled.contains("never installed"),
        "an absent portal must fail on the denominator, not pass because its \
         unconsulted list is empty; got {uninstalled:?}",
    );
}

/// `#[jit_inline]` carries its own `int_fields` / `ref_fields`, and this is
/// where the population is.
///
/// A machine declares a handful of keys; a consumer's helpers repeat theirs at
/// every site, so a key matching nothing at one site matches nothing at dozens.
/// The first version of this check covered only the `#[jit_interp]` portal and
/// reported exactly one key on the consumer that motivated it, while the 36
/// entries it was written for sat on the other surface untouched — the same
/// asymmetry these two macros produce on every input.
#[majit_macros::jit_inline(
    ref_params = {
        cell: ref(Cell),
    },
    int_fields = {
        Cell::count => u32,
        Unreached::count => u8,
    },
)]
fn bump_cell(cell: usize) {
    cell.count = cell.count + 1u32;
}

#[test]
fn an_inline_helpers_own_declaration_is_reported_under_the_helper() {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
    let _ = __majit_inline_jitcode_bump_cell_with_asm(&mut asm);

    let keys: Vec<String> = majit_metainterp::unconsulted_field_declarations()
        .into_iter()
        .filter(|e| e.interp == "bump_cell")
        .map(|e| e.key.to_string())
        .collect();
    assert!(
        keys.iter().any(|key| key == "Unreached::count"),
        "the helper's own unconsulted key must be reported under the helper's \
         name, not the enclosing machine's; keys={keys:?}",
    );
    assert!(
        !keys.iter().any(|key| key == "Cell::count"),
        "control: `cell.count` is a real access through `ref(Cell)`; keys={keys:?}",
    );
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_default()
}
