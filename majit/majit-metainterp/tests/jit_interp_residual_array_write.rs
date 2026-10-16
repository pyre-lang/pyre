//! `residual_writes = { <ref>.<field>[] => [...] }` — the element half.
//!
//! A plain `residual_writes` entry names the FIELD, which invalidates a cached
//! `getfield_gc_r` of the array base.  That is the wrong invalidation for a
//! buffer mutated in place: the base does not change, so the next element load
//! is served from `OptHeap`'s separate array cache and reads a value from
//! before the call.  The `[]` suffix declares the ELEMENTS instead.
//!
//! This fixture holds the declaration to the descr it must produce.  The shape
//! it has to carry is pinned by `array_write_set_invalidation.rs`; what is
//! tested here is that the declaration reaches the emitted call at all.

use majit_metainterp::jitcode::CanonicalBhDescr;
use majit_metainterp::{Assembler, JitDriver};

/// The array-backed pool the residual mutates.  `data` holds the buffer BASE
/// POINTER — `size` is the sibling length field, since a headerless buffer has
/// no length word to read.
#[repr(C)]
struct ArrStack {
    data: *mut i64,
    size: usize,
}

/// An opaque in-place mutator of `stack.data`'s elements.  The body is
/// irrelevant to what this fixture asserts — the JIT never looks inside a
/// residual — but it must exist for the concrete interpreter path.
extern "C" fn jit_scramble(stack: usize) {
    let stack = stack as *mut ArrStack;
    if stack.is_null() {
        return;
    }
    // SAFETY: the fixture never runs the interpreter, so this is only reached
    // through the null guard above.
    unsafe {
        let n = (*stack).size;
        for i in 0..n {
            *(*stack).data.add(i) = 0;
        }
    }
}

struct ArrTestState {
    a: i64,
    sel: usize,
}

const OP_NOP: u8 = 0;
const OP_SCRAMBLE: u8 = 1;

pub type Bytecode = [u8];

#[majit_macros::jit_interp(
    state = ArrTestState,
    env = Bytecode,
    state_fields = { a: int, sel: ref(ArrStack) },
    greens = [],
    array_fields = { ArrStack::data => i64 },
    calls = { jit_scramble => residual_void },
    residual_writes = {
        sel.data[] => [jit_scramble],
    },
)]
#[allow(unused_assignments, unused_variables)]
fn arr_minimal(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<ArrTestState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = ArrTestState { a: 0, sel: 0 };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while pc < program.len() {
        jit_merge_point!();
        let opcode = program[pc];
        pc += 1;
        match opcode {
            OP_NOP => {}
            OP_SCRAMBLE => jit_scramble(state.sel),
            _ => break,
        }
    }
    state.a
}

fn build_dispatch_jitcode(asm: &mut Assembler) -> majit_metainterp::JitCode {
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
    __prebuild_jitcode_liveness_arr_minimal(asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_arr_minimal(asm, 0i64)
        .expect("dispatch lower must succeed for the array-write fixture")
}

/// Every emitted call's `EffectInfo`, across the dispatch jitcode and each
/// per-arm sub-jitcode — the residual lives in an arm, not the portal.
fn all_call_effect_infos(dispatch_jc: &majit_metainterp::JitCode) -> Vec<majit_ir::EffectInfo> {
    fn collect(jc: &majit_metainterp::JitCode, out: &mut Vec<majit_ir::EffectInfo>) {
        for descr in &jc.exec.descrs {
            if let Some(CanonicalBhDescr::Call { calldescr }) = descr.as_bh_descr() {
                out.push(calldescr.extra_info.clone());
            }
            if let Some(sub) = descr.as_jitcode() {
                collect(sub, out);
            }
        }
    }
    let mut out = Vec::new();
    collect(dispatch_jc, &mut out);
    out
}

/// The shape `Assembler::add_raw_int_array_descr_signed(item_size, signed)`
/// interns for a `stack.data[i]` read — what the declared write has to reach.
fn element_read_descr_of(item_size: usize, is_item_signed: bool) -> majit_ir::DescrRef {
    majit_ir::descr::make_array_descr_from_lltype_shape(
        0,
        /* base_size */ 0,
        item_size,
        None,
        majit_ir::Type::Int,
        false,
        false,
        is_item_signed,
        /* is_gc_managed */ false,
        None,
        false,
        u32::MAX,
        Vec::new(),
    )
}

/// The `i64` element array both fixtures below declare.
fn element_read_descr() -> majit_ir::DescrRef {
    element_read_descr_of(std::mem::size_of::<i64>(), true)
}

#[test]
fn the_bracket_suffix_reaches_the_emitted_calls_effect_info() {
    let mut asm = Assembler::new();
    let dispatch_jc = build_dispatch_jitcode(&mut asm);
    let eis = all_call_effect_infos(&dispatch_jc);
    assert!(
        !eis.is_empty(),
        "the fixture must emit at least one residual call, or this test \
         cannot observe anything"
    );
    let read = element_read_descr();
    assert!(
        eis.iter().any(|ei| ei.writes_array_descr_by_shape(&read)),
        "`sel.data[] => [jit_scramble]` must give the emitted call a write set \
         reaching the element descr the read interns; without it the trace \
         keeps serving a `getarrayitem_gc_i` from before the call"
    );
}

/// The negative half: the flag has to be what selects the array write set, not
/// something every residual gets.  A `residual_writes` declaration WITHOUT the
/// `[]` suffix is a field write, and must leave the element cache alone.
///
/// This fixture is the same shape as the one above minus the brackets, so a
/// change that made every residual write every array would fail here while the
/// test above still passed.
mod without_the_bracket_suffix {
    use super::{ArrStack, Bytecode, JitDriver, OP_NOP, OP_SCRAMBLE, jit_scramble};
    use majit_metainterp::Assembler;

    struct FieldOnlyState {
        a: i64,
        sel: usize,
    }

    #[majit_macros::jit_interp(
        state = FieldOnlyState,
        env = Bytecode,
        state_fields = { a: int, sel: ref(ArrStack) },
        greens = [],
        array_fields = { ArrStack::data => i64 },
        calls = { jit_scramble => residual_void },
        residual_writes = {
            sel.data => [jit_scramble],
        },
    )]
    #[allow(unused_assignments, unused_variables)]
    fn field_only(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<FieldOnlyState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = FieldOnlyState { a: 0, sel: 0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_NOP => {}
                OP_SCRAMBLE => jit_scramble(state.sel),
                _ => break,
            }
        }
        state.a
    }

    #[test]
    fn a_field_declaration_leaves_the_element_cache_alone() {
        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
        __prebuild_jitcode_liveness_field_only(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_field_only(&mut asm, 0i64)
            .expect("dispatch lower must succeed for the field-only fixture");
        let eis = super::all_call_effect_infos(&dispatch_jc);
        assert!(
            !eis.is_empty(),
            "the fixture must emit at least one residual call"
        );
        let read = super::element_read_descr();
        assert!(
            eis.iter().all(|ei| !ei.writes_array_descr_by_shape(&read)),
            "`sel.data` without the `[]` suffix declares a write to the base \
             field, not to the elements; declaring it must not invalidate the \
             element cache"
        );
    }
}

/// Signedness is part of the shape key, so the declaration and the reads must
/// derive it from the SAME element type.  The fixtures above are all `i64`,
/// where a hard-coded "signed" write spec agrees with the read by accident;
/// this one is `u8`, where it does not.
mod unsigned_elements {
    use super::{Bytecode, JitDriver, OP_NOP, OP_SCRAMBLE};
    use majit_metainterp::Assembler;

    #[repr(C)]
    struct ByteStack {
        data: *mut u8,
        size: usize,
    }

    extern "C" fn jit_scramble_bytes(stack: usize) {
        let stack = stack as *mut ByteStack;
        if stack.is_null() {
            return;
        }
        // SAFETY: the fixture never runs the interpreter, so this is only
        // reached through the null guard above.
        unsafe {
            let n = (*stack).size;
            for i in 0..n {
                *(*stack).data.add(i) = 0;
            }
        }
    }

    struct ByteState {
        a: i64,
        sel: usize,
    }

    #[majit_macros::jit_interp(
        state = ByteState,
        env = Bytecode,
        state_fields = { a: int, sel: ref(ByteStack) },
        greens = [],
        array_fields = { ByteStack::data => u8 },
        calls = { jit_scramble_bytes => residual_void },
        residual_writes = {
            sel.data[] => [jit_scramble_bytes],
        },
    )]
    #[allow(unused_assignments, unused_variables)]
    fn byte_elements(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<ByteState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = ByteState { a: 0, sel: 0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_NOP => {}
                OP_SCRAMBLE => jit_scramble_bytes(state.sel),
                _ => break,
            }
        }
        state.a
    }

    #[test]
    fn a_u8_declaration_reaches_the_unsigned_read_and_not_a_signed_one() {
        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
        __prebuild_jitcode_liveness_byte_elements(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_byte_elements(&mut asm, 0i64)
            .expect("dispatch lower must succeed for the u8-element fixture");
        let eis = super::all_call_effect_infos(&dispatch_jc);
        assert!(
            !eis.is_empty(),
            "the fixture must emit at least one residual call"
        );

        let unsigned_read = super::element_read_descr_of(1, false);
        assert!(
            eis.iter()
                .any(|ei| ei.writes_array_descr_by_shape(&unsigned_read)),
            "a `u8` element declaration must reach the UNSIGNED element descr \
             its own reads intern; a hard-coded signed write spec misses it \
             entirely and the trace keeps serving the pre-call element"
        );

        // The control that makes the assertion above non-vacuous: the write set
        // is not matching everything. A signed byte array is the shape the old
        // hard-coded spelling produced, and it is a different array.
        let signed_read = super::element_read_descr_of(1, true);
        assert!(
            eis.iter()
                .all(|ei| !ei.writes_array_descr_by_shape(&signed_read)),
            "the declaration must not reach a signed byte array — if it does, \
             the shape predicate is too loose to have detected the mismatch"
        );
    }
}
