//! A `str` state field on a `split_dispatch` machine is a vable ref field.
//!
//! `jtransform.py` `rewrite_op_getfield` on a virtualizable field of kind
//! `r`, and `virtualizable.py` `VirtualizableInfo` static fields of any
//! kind, treat `str` as a GC ref of the virtualizable. An arm that reads
//! `state.s` and writes an int scalar must therefore emit
//! `getfield_vable_r` / `setfield_vable_i` inside the sub-JitCode, and
//! the compiled loop must match a pure interpreter run. The regex
//! example (`majit/examples/regex`) is the source shape: `s: str` on
//! the state struct as a `usize` `rstr.STR` pointer, read through
//! `ll_strlen` / `ll_strgetitem`.

use core::sync::atomic::{AtomicU32, Ordering};
use majit_metainterp::jitcode::insns::{
    BC_GETFIELD_VABLE_R, BC_INLINE_CALL, BC_INLINE_CALL_IR_I, BC_INLINE_CALL_IR_V,
    BC_LOAD_STATE_FIELD_REF, BC_SETFIELD_VABLE_I, BC_STRGETITEM,
};
use majit_metainterp::{Assembler, JitCode, JitDriver};

pub type Bytecode = [u8];

const OP_ADD: u8 = 1;
const OP_LOOP: u8 = 2;
const N: i64 = 40;
const FIRST_BYTE: u8 = b'A';

struct SplitStr {
    acc: i64,
    /// `rstr.STR` pointer (`ll_strlen` / `ll_strgetitem`).
    s: usize,
    ticks: i64,
}

static COMPILES: AtomicU32 = AtomicU32::new(0);

/// Header in front of the character bytes, matching `rstr.STR`.
#[repr(C)]
struct Header {
    hash: isize,
    length: usize,
}

/// Owned `rstr.STR` bytes. The pointer stays valid for the life of this value.
struct RpyStr {
    buf: Vec<u8>,
}

impl RpyStr {
    fn from_bytes(bytes: &[u8]) -> Self {
        // `rstr.STR`: hash word, length word, chars, plus the extra NUL
        // `extra_item_after_alloc` counts in the token basesize.
        let chars_offset = std::mem::size_of::<Header>();
        let mut buf = vec![0u8; chars_offset + bytes.len() + 1];
        unsafe {
            let header = buf.as_mut_ptr() as *mut Header;
            (*header).hash = 0;
            (*header).length = bytes.len();
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buf.as_mut_ptr().add(chars_offset),
                bytes.len(),
            );
        }
        Self { buf }
    }

    fn as_usize(&self) -> usize {
        self.buf.as_ptr() as usize
    }
}

/// `len(s)` on an `rstr.STR`. The trace records `Strlen` for this call.
fn ll_strlen(s: usize) -> i64 {
    assert_ne!(s, 0, "ll_strlen: null string");
    unsafe {
        (s.wrapping_add(std::mem::offset_of!(Header, length)) as *const usize).read_unaligned()
            as i64
    }
}

/// `s[i]` as a byte, zero-extended. The trace records `Strgetitem`.
fn ll_strgetitem(s: usize, index: i64) -> i64 {
    let len = ll_strlen(s);
    assert!(
        index >= 0 && index < len,
        "ll_strgetitem: index {index} outside 0..{len}"
    );
    unsafe {
        (s.wrapping_add(std::mem::size_of::<Header>())
            .wrapping_add(index as usize) as *const u8)
            .read_unaligned() as i64
    }
}

#[majit_macros::jit_interp(
    state = SplitStr,
    env = Bytecode,
    split_dispatch = true,
    greens = [pc, program],
    state_fields = { acc: int, s: str, ticks: int },
)]
#[allow(unused_assignments, unused_variables)]
fn split_str_sum(program: &Bytecode, threshold: u32, s: usize, ticks: i64) -> i64 {
    let mut driver: JitDriver<SplitStr> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _b, _a, _opcodes| {
        COMPILES.fetch_add(1, Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let mut state = SplitStr { acc: 0, s, ticks };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while pc < program.len() {
        jit_merge_point!(driver, program, pc; state);
        let opcode = program[pc];
        match opcode {
            OP_ADD => {
                state.acc = state.acc + ll_strgetitem(state.s, 0);
                pc += 1;
            }
            OP_LOOP => {
                state.ticks = state.ticks - 1i64;
                if state.ticks != 0 {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
                pc += 1;
            }
            _ => break,
        }
    }
    state.acc
}

fn program() -> Vec<u8> {
    vec![OP_ADD, OP_LOOP]
}

fn build_dispatch_jitcode() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
    __prebuild_jitcode_liveness_split_str_sum(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_split_str_sum(&mut asm, 0i64).expect("dispatch lower must succeed")
}

fn dispatch_subs(jc: &JitCode) -> impl Iterator<Item = &JitCode> {
    jc.exec
        .descrs
        .iter()
        .filter_map(|descr| descr.as_jitcode().map(|sub| sub.as_ref()))
}

#[test]
fn split_dispatch_add_arm_reads_str_as_vable_ref() {
    let dispatch_jc = build_dispatch_jitcode();
    majit_metainterp::assert_no_degraded_dispatch_arms("SplitStr");

    let dispatch_has_inline = dispatch_jc.code.contains(&BC_INLINE_CALL)
        || dispatch_jc.code.contains(&BC_INLINE_CALL_IR_I)
        || dispatch_jc.code.contains(&BC_INLINE_CALL_IR_V);
    assert!(
        dispatch_has_inline,
        "split_dispatch must emit an arm inline_call; dispatch={:?}",
        dispatch_jc.code
    );

    let readers: Vec<&JitCode> = dispatch_subs(&dispatch_jc)
        .filter(|code| {
            code.code.contains(&BC_GETFIELD_VABLE_R) && code.code.contains(&BC_STRGETITEM)
        })
        .collect();
    assert!(
        !readers.is_empty(),
        "the OP_ADD arm sub-JitCode must emit getfield_vable_r then strgetitem; \
         dispatch={:?} subs={:?}",
        dispatch_jc.code,
        dispatch_subs(&dispatch_jc)
            .map(|c| &c.code)
            .collect::<Vec<_>>()
    );
    assert!(
        readers
            .iter()
            .any(|code| code.code.contains(&BC_SETFIELD_VABLE_I)),
        "the OP_ADD arm sub-JitCode must emit setfield_vable_i for acc; subs={:?}",
        readers.iter().map(|c| &c.code).collect::<Vec<_>>()
    );
    assert!(
        !dispatch_jc.code.contains(&BC_LOAD_STATE_FIELD_REF)
            && dispatch_subs(&dispatch_jc)
                .all(|code| !code.code.contains(&BC_LOAD_STATE_FIELD_REF)),
        "s must not remain an identity-slot load_state_field_ref; dispatch={:?} subs={:?}",
        dispatch_jc.code,
        dispatch_subs(&dispatch_jc)
            .map(|c| &c.code)
            .collect::<Vec<_>>()
    );
}

#[test]
fn compiled_split_dispatch_str_read_matches_the_interpreter() {
    let input = RpyStr::from_bytes(&[FIRST_BYTE, b'B', b'C']);
    let program = program();
    let expected = FIRST_BYTE as i64 * N;

    let interp = split_str_sum(&program, u32::MAX, input.as_usize(), N);
    assert_eq!(interp, expected, "interpreter adds s[0] once per loop trip");

    COMPILES.store(0, Ordering::Relaxed);
    let compiled = split_str_sum(&program, 2, input.as_usize(), N);
    assert!(
        COMPILES.load(Ordering::Relaxed) >= 1,
        "the split_dispatch loop must compile"
    );
    assert_eq!(
        compiled, interp,
        "compiled sub-JitCode state.s[0] must match the interpreter"
    );
}
