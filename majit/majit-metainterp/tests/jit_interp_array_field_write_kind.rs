//! An array field is an INT field holding a raw buffer address, and both
//! producers that describe it have to say so.
//!
//! Two places in the lowering register a description of the same member.  The
//! array base read (`emit_array_field_base`) registers it as an unsigned word:
//! the field holds a buffer base, and the read is a `getfield_gc_i`
//! (`history.py` `getkind`: a Ptr whose `TO._gckind == 'raw'` is kind `'int'`;
//! `descr.py` `get_type_flag`: raw Ptr → FLAG_UNSIGNED).  A
//! `residual_writes = { <ref>.<field> => [..] }` declaration — the FIELD form,
//! no `[]` — rebuilds the same layout to mint the write set's descr.
//!
//! `get_field_descr` caches by `(struct, fieldname)` and is cache-OR-mint: the
//! first producer to reach a slot wins, and a later one describing the field
//! differently gets the cached descr back with only a collision counter to show
//! for it.  So a write-set producer that asked the wrong map about the field's
//! kind does not fail; it either loses silently, or wins and leaves every
//! `getfield_gc_i` of that base carrying a signed eight-byte integer descr.
//! Which of those happens is decided by the order the two producers were
//! emitted in.
//!
//! This fixture pins the write-set producer alone.  Nothing here reads
//! `sel.data[i]`, so the array base producer never runs and the cache slot for
//! `PointerFieldStack::data` holds exactly what the write-set path decided —
//! order cannot mask the answer.

use majit_metainterp::{Assembler, JitDriver};

/// Reached only through the write-set declaration below, so its cache slot has
/// exactly one producer.  A struct any other fixture also touched would let the
/// other producer mint the slot and this test would pass without meaning it.
#[repr(C)]
struct PointerFieldStack {
    data: *mut i64,
    size: usize,
    /// An eight-byte field that is not `Copy`, named in the write set below but
    /// declared in none of the field maps, so it takes the undeclared default
    /// and the macro emits its width witness over it.
    ///
    /// A witness written as `|s| s.field` returns the field by value and so
    /// moves out of a shared borrow, which only compiles when the field is
    /// `Copy`. This member is the standing check that it is not written that
    /// way: if it regresses, this crate stops compiling rather than quietly
    /// refusing a struct that has nothing wrong with it.
    generation: Generation,
}

/// Deliberately not `Copy` and deliberately eight bytes.
#[repr(transparent)]
struct Generation(i64);

/// An opaque in-place mutator.  The body is irrelevant — the JIT never looks
/// inside a residual — but it has to exist for the concrete path.
extern "C" fn jit_scramble_pointer_field(stack: usize) {
    let stack = stack as *mut PointerFieldStack;
    if stack.is_null() {
        return;
    }
    unsafe {
        (*stack).size = (*stack).size;
        (*stack).generation = Generation((*stack).generation.0 + 1);
    }
}

pub type Bytecode = [u8];

const OP_NOP: u8 = 0;
const OP_SCRAMBLE: u8 = 1;

struct PointerFieldState {
    a: i64,
    sel: usize,
}

#[majit_macros::jit_interp(
    state = PointerFieldState,
    env = Bytecode,
    state_fields = { a: int, sel: ref(PointerFieldStack) },
    greens = [],
    array_fields = { PointerFieldStack::data => i64 },
    int_fields = {
        // Declared for its width, not its kind. `usize` is four bytes on a
        // 32-bit target, and naming a field in a write set is what sends it
        // through the undeclared-scalar default, whose witness demands eight —
        // so leaving it undeclared fails macro expansion on wasm32 before any
        // of this runs. The control below still reads whatever kind the
        // write-set path decided.
        PointerFieldStack::size => usize,
    },
    calls = { jit_scramble_pointer_field => residual_void },
    residual_writes = {
        // `size` is here only so the scalar control below has a descr to read.
        // Naming a field in a write-set layout is what mints its descr, and
        // without one the control skips: it asserted nothing, which is the
        // failure mode it exists to rule out for the array field.
        sel.data => [jit_scramble_pointer_field],
        sel.size => [jit_scramble_pointer_field],
        sel.generation => [jit_scramble_pointer_field],
    },
)]
#[allow(unused_assignments, unused_variables)]
fn pointer_field_only(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<PointerFieldState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = PointerFieldState { a: 0, sel: 0 };
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
            OP_SCRAMBLE => jit_scramble_pointer_field(state.sel),
            _ => break,
        }
    }
    state.a
}

fn build() -> majit_metainterp::JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
    __prebuild_jitcode_liveness_pointer_field_only(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_pointer_field_only(&mut asm, 0i64)
        .expect("dispatch lower must succeed for the pointer-field fixture")
}

/// `(field_type, field_size, is_signed)` the cache holds for `data`, or `None`
/// if nothing registered it.
fn cached_data_field() -> Option<(majit_ir::Type, usize, bool)> {
    use majit_ir::descr::FieldDescr as _;
    let type_id = majit_metainterp::__majit_struct_type_id::<PointerFieldStack>(false);
    let cache = majit_ir::descr::gc_cache().lock();
    let descr = cache
        ._cache_field
        .get(&majit_ir::descr::LLType::Struct(type_id))
        .and_then(|fields| fields.get("data"))
        .cloned()?;
    Some((
        descr.field_type(),
        descr.field_size(),
        descr.is_field_signed(),
    ))
}

/// The subject: a write-set declaration naming an array field describes a
/// raw buffer base address.
///
/// Before this was fixed the write-set path asked only `ref_fields` whether the
/// member is a pointer.  An array field is declared in `array_fields`, so the
/// answer was no and the field went into the write set as an eight-byte SIGNED
/// INTEGER — the undeclared-scalar default — for a member holding `*mut i64`.
/// `history.py` `getkind` banks that Ptr as `'int'`; `descr.py` `get_type_flag`
/// marks it FLAG_UNSIGNED, one target word wide. The sign and size assertions
/// are what distinguish the raw-address descr from that default.
#[test]
fn a_write_set_declaration_over_an_array_field_describes_a_raw_address() {
    let _jc = build();
    let (field_type, field_size, is_signed) = cached_data_field()
        .expect("the fixture's `residual_writes` declaration must register `data`");
    assert_eq!(
        field_type,
        majit_ir::Type::Int,
        "`sel.data` holds a raw buffer base address, so `history.py` `getkind` \
         of that Ptr (`TO._gckind == 'raw'`) is kind 'int'; the write set's \
         descr must be an Int",
    );
    assert_eq!(
        field_size,
        std::mem::size_of::<usize>(),
        "a raw address is one target word wide, not the eight bytes the \
         undeclared-scalar default claims (they differ on wasm32)",
    );
    assert!(
        !is_signed,
        "`descr.py` `get_type_flag` marks a raw Ptr FLAG_UNSIGNED; the \
         undeclared-scalar default is SIGNED, and the sign flag is part of \
         what `describes_same_field` compares when a second producer arrives",
    );
}

/// The control, without which the assertion above cannot fail for the reason it
/// names: a non-array field of the same struct must still be described as the
/// scalar it is.
///
/// A fix that applied the raw-address descr to every write-set field would
/// satisfy the subject and be exactly as wrong in the other direction.
#[test]
fn a_scalar_field_of_the_same_struct_is_still_a_scalar() {
    use majit_ir::descr::FieldDescr as _;
    let _jc = build();
    let type_id = majit_metainterp::__majit_struct_type_id::<PointerFieldStack>(false);
    let cache = majit_ir::descr::gc_cache().lock();
    let size_field = cache
        ._cache_field
        .get(&majit_ir::descr::LLType::Struct(type_id))
        .and_then(|fields| fields.get("size"))
        .cloned();
    // Require the slot rather than skipping when it is absent. This test read
    // `if let Some(..)` and `size` was in no declaration, so the assertion below
    // never ran once — a control that does not execute rules nothing out, which
    // is exactly what it was written to prevent for the array field. The
    // fixture now names `size` in its write set so the descr exists.
    let descr = size_field.expect(
        "`size` must be registered for this control to assert anything; the \
         fixture's `residual_writes` names it",
    );
    assert_eq!(
        descr.field_type(),
        majit_ir::Type::Int,
        "`size` is a scalar; only the field a pointer declaration names may \
         become a Ref",
    );
}
