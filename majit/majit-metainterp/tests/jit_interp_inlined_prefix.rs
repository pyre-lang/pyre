//! A field declared on an inlined leading substructure, reached through the
//! structs that embed it.
//!
//! `rclass.py:548` builds every instance struct as `MkStruct(name, ('super',
//! rbase.object_type), *own_fields)`, and `rclass.py getfield
//! InstanceRepr.getfield` resolves a field against the repr that OWNS it,
//! recursing to `self.rbase` with `force_cast=True` when it is not its own.
//! `jtransform.py:881` therefore always reads the descr off the declaring
//! struct, which is what gives one physical field one descriptor across a whole
//! family. `inlined_prefix` is that relationship declared to the macro.
//!
//! **What makes this file a gate is that it compiles at all.** `Prefixed` and
//! `PrefixedWithTail` below have no `head` and no `size` field: those words
//! live in the `Header` each embeds, and Rust knows it. Every access in the
//! machine spells `state.<ref>.head`, so without the redirect the expansion
//! emits `(*(p as *mut Prefixed)).head` on the concrete arm and
//! `offset_of!(Prefixed, head)` on the JIT arm, and this crate fails to build.
//! An empty assertion list would still be a real test here; the assertions
//! below cover what compiling alone does not.
//!
//! The two structs are deliberately different sizes — 16 and 24 — because the
//! point of the mechanism is that they keep their own identities and their own
//! sizes while SHARING the descriptors for the words they have in common.

use majit_metainterp::{Assembler, JitDriver};

#[repr(C)]
struct Node {
    value: i64,
    next: *mut Node,
}

/// The leading substructure. It, and only it, declares `head` and `size`.
#[repr(C)]
struct Header {
    head: *mut Node,
    size: u32,
}

/// Embeds the header and adds nothing: 16 bytes.
#[repr(C)]
struct Prefixed {
    base: Header,
}

/// Embeds the header and adds a word of its own: 24 bytes. `extra` is what
/// `heaptracker.py:68-69`'s recursion exists to number correctly — it must land
/// after the header's two fields, not at the slot `head` already occupies for
/// the same object.
#[repr(C)]
struct PrefixedWithTail {
    base: Header,
    extra: *mut Node,
}

const OP_NOP: u8 = 0;
const OP_READ_SIZE: u8 = 1;
const OP_MOVE_HEAD: u8 = 2;
const OP_HEAD_TO_EXTRA: u8 = 3;

pub type Bytecode = [u8];

// Named as the `jit_interp` state type; the macro lowers the interpreter body
// into jitcode and never runs it, so nothing constructs the struct.
#[expect(dead_code)]
struct PrefixState {
    a: i64,
    one: usize,
    two: usize,
}

#[majit_macros::jit_interp(
    state = PrefixState,
    env = Bytecode,
    state_fields = { a: int, one: ref(Prefixed), two: ref(PrefixedWithTail) },
    greens = [],
    ref_fields = {
        Header::head => Node,
        Node::next => Node,
        PrefixedWithTail::extra => Node,
    },
    int_fields = { Header::size => u32 },
    inlined_prefix = {
        Prefixed::base => Header,
        PrefixedWithTail::base => Header,
    },
)]
#[allow(unused_assignments, unused_variables)]
fn prefixed_minimal(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<PrefixState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = PrefixState {
        a: 0,
        one: 0,
        two: 0,
    };
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
            // An integer field of the header, reached through the smaller
            // struct: `Header::size`, not `Prefixed::size`.
            OP_READ_SIZE => state.a += state.one.size as i64,
            // The same header word read through one struct and written through
            // the other. One descriptor or two is exactly the difference
            // between this forwarding and not.
            OP_MOVE_HEAD => state.two.head = state.one.head,
            // The outer struct's OWN field beside the inherited one.
            OP_HEAD_TO_EXTRA => state.two.extra = state.two.head,
            _ => break,
        }
    }
    state.a
}

fn build_prefixed_dispatch_jitcode(asm: &mut Assembler) -> majit_metainterp::JitCode {
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
    __prebuild_jitcode_liveness_prefixed_minimal(asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_prefixed_minimal(asm, 0i64)
        .expect("dispatch lower must succeed for the inlined-prefix fixture")
}

/// The two structs keep their own sizes.
///
/// Nothing in the mechanism merges type identities: only the FIELD descriptors
/// of the shared header unify. Checked here from the Rust types the emitted
/// `size_of` witnesses read, so a change that collapsed the two onto one
/// identity would have to change these numbers first.
#[test]
fn the_embedding_structs_keep_their_own_sizes() {
    assert_eq!(std::mem::size_of::<Header>(), 16);
    assert_eq!(std::mem::size_of::<Prefixed>(), 16);
    assert_eq!(std::mem::size_of::<PrefixedWithTail>(), 24);
    assert_eq!(std::mem::offset_of!(PrefixedWithTail, extra), 16);
    // The offset every redirected access assumes, and the one the expansion
    // asserts for itself — `lltype.py _first_struct` admits an inlined substructure
    // only as the leading field.
    assert_eq!(std::mem::offset_of!(Prefixed, base), 0);
    assert_eq!(std::mem::offset_of!(PrefixedWithTail, base), 0);
}

/// Lowering a machine whose every field access is redirected reports no
/// disagreement about the words it registers.
///
/// The header's fields are registered twice over: once by the accesses that
/// resolve onto `Header` itself, and once as the leading entries of each
/// embedding struct's layout (`heaptracker.py:68-69`). Those two descriptions
/// of one word are built by different code paths, and a `Redescribed` conflict
/// is what a disagreement between them looks like.
#[test]
fn an_inlined_prefix_registers_one_description_of_each_shared_word() {
    let before = majit_metainterp::struct_layout_registrations();
    let mut asm = Assembler::new();
    let _ = build_prefixed_dispatch_jitcode(&mut asm);

    // The denominator. An empty conflict list is also what a run that
    // registered nothing produces.
    assert!(
        majit_metainterp::struct_layout_registrations() > before,
        "the fixture registered no struct layout, so the conflict check below \
         grades nothing",
    );
    assert_eq!(
        majit_metainterp::struct_layout_conflicts(),
        vec![],
        "the header's fields are described both by the accesses that resolve \
         onto it and by the leading entries of each embedding struct's layout; \
         a disagreement between those two means the prefix entries do not match \
         the declarations they were built from",
    );
}

/// Every arm lowered.
///
/// This is the failure mode when the JIT lowerer and the concrete-path
/// rewriters disagree about which struct declares a field: the arm the lowerer
/// gave up on becomes an abort stub rather than a build error, so the machine
/// still runs and still produces answers — just never from compiled code.
#[test]
fn every_redirected_arm_lowers() {
    let mut asm = Assembler::new();
    let _ = build_prefixed_dispatch_jitcode(&mut asm);

    let census = majit_metainterp::dispatch_arm_census();
    let counted = census
        .iter()
        .find(|entry| entry.interp == "PrefixState")
        .unwrap_or_else(|| panic!("the portal must record its arm count; census={census:?}"));
    assert_eq!(
        counted.arms, 4,
        "OP_NOP, OP_READ_SIZE, OP_MOVE_HEAD and OP_HEAD_TO_EXTRA are counted; \
         the `_` default is not; census={census:?}"
    );

    majit_metainterp::assert_no_degraded_dispatch_arms("PrefixState");
}
