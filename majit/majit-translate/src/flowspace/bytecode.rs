//! Bytecode handling classes and functions for use by the flow space.
//!
//! RPython basis: `rpython/flowspace/bytecode.py`.
//!
//! ## TODO: bytecode enum source
//!
//! Upstream `bytecode.py:6` does `from opcode import EXTENDED_ARG,
//! HAVE_ARGUMENT` — the opcode definitions come from **host Python's
//! stdlib `opcode` module**, not from RPython itself. RPython wraps
//! the host table in `rpython/tool/stdlib_opcode.py`.
//!
//! Rust has no equivalent host-language opcode module, so we reach for
//! the externally-owned `rustpython-compiler-core` crate, which plays
//! the same role: a neutral third-party source of CPython-3.15 opcode
//! definitions (`Instruction`, `CodeObject`, `ConstantData`, ...).
//! majit-translate wraps it here in `HostCode`, exactly as RPython
//! wraps `opcode`.
//!
//! This is NOT a dependency on pyre: `rustpython-compiler-core` is
//! external to both `majit/` and `pyre/`, preserving the `rpython/` ⊥
//! `pypy/` boundary at the Rust-crate layer.
//!
//! There is no standalone `OpCode` type: RustPython's `Instruction`
//! enum already carries the opcode tag inline (each variant IS an
//! opcode), so we re-export `Instruction` on its own rather than
//! inventing a parallel `OpCode` alias.

pub use rustpython_compiler_core::bytecode::{
    BinaryOperator, CodeFlags, CodeObject, CodeUnit, ComparisonOperator, ConstantData,
    ExceptionTableEntry, Instruction, MakeFunctionFlag, OpArg, OpArgState,
};

use std::sync::atomic::{AtomicU64, Ordering};

use super::argument::Signature;

/// Error raised when the bytecode stream cannot be decoded.
///
/// RPython basis: `rpython/flowspace/bytecode.py:BytecodeCorruption`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BytecodeCorruption(pub String);

impl BytecodeCorruption {
    pub fn new<S: Into<String>>(msg: S) -> Self {
        Self(msg.into())
    }
}

impl core::fmt::Display for BytecodeCorruption {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "BytecodeCorruption: {}", self.0)
    }
}

impl std::error::Error for BytecodeCorruption {}

bitflags::bitflags! {
    /// RPython `bytecode.py` `CO_*` compile flags.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct CoFlags: u32 {
        const NEWLOCALS = 0x0002;
        const VARARGS = 0x0004;
        const VARKEYWORDS = 0x0008;
        const GENERATOR = 0x0020;
    }
}

/// Compute `([list-of-arg-names], vararg-name-or-None, kwarg-name-or-None)`.
///
/// RPython basis: `rpython/flowspace/bytecode.py:cpython_code_signature`.
pub fn cpython_code_signature(code: &HostCode) -> Signature {
    let mut argcount = code.co_argcount as usize;
    let argnames: Vec<String> = code.co_varnames.iter().take(argcount).cloned().collect();
    let varargname = if code.co_flags & CoFlags::VARARGS.bits() != 0 {
        let name = code.co_varnames[argcount].clone();
        argcount += 1;
        Some(name)
    } else {
        None
    };
    let kwargname = if code.co_flags & CoFlags::VARKEYWORDS.bits() != 0 {
        let name = code.co_varnames[argcount].clone();
        Some(name)
    } else {
        None
    };
    Signature::new(argnames, varargname, kwargname)
}

/// A wrapper around a native code object of the host interpreter.
///
/// RPython basis: `rpython/flowspace/bytecode.py:HostCode`.
///
/// Fields copy the upstream `co_*` attribute names verbatim. The
/// `opnames` class-level table has no Rust analogue — Phase 3
/// `flowcontext.rs` dispatches on the `Instruction` enum directly,
/// which already carries the opcode tag. The upstream
/// `HostCode.opnames` string list is a stdlib-`opcode` module artifact
/// with no independent meaning.
///
/// ### Field-shape deviation (parity rule #1)
///
/// RPython's `__init__` stores `code` as the raw `co_code` byte string
/// because PyPy-2.7 carried a pre-wordcode bytecode format. The CPython
/// 3.15 equivalent in RustPython is `CodeUnits` (a
/// two-byte-per-instruction container); we store that directly as
/// `co_code` to avoid a second copy into raw bytes. The RPython walker
/// (F2.3 port of `HostCode.read`) becomes a `CodeUnits` iterator
/// instead of a byte-level `ord(co_code[offset])` loop.
#[derive(Clone, Debug)]
pub struct HostCode {
    /// Stable object identity for RPython `Hashable(codeobj)` parity.
    pub(crate) id: u64,
    pub co_argcount: u32,
    pub co_nlocals: u32,
    pub co_stacksize: u32,
    pub co_flags: u32,
    pub co_code: rustpython_compiler_core::bytecode::CodeUnits,
    pub consts: Vec<ConstantData>,
    pub names: Vec<String>,
    pub co_varnames: Vec<String>,
    pub co_freevars: Vec<String>,
    /// Upstream `HostCode.co_cellvars` — the tuple of variable names
    /// that form the function's enclosing cells (`__closure__`). Empty
    /// for every RPython-legal function; a non-empty tuple is the
    /// closure-detection signal used by `objspace._assert_rpythonic`
    /// (objspace.py:25).
    pub co_cellvars: Vec<String>,
    pub co_filename: String,
    pub co_name: String,
    pub co_firstlineno: u32,
    pub co_lnotab: Vec<u8>,
    pub exceptiontable: Box<[u8]>,
    pub signature: Signature,
}

impl PartialEq for HostCode {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for HostCode {}

static NEXT_HOST_CODE_ID: AtomicU64 = AtomicU64::new(1);

impl HostCode {
    pub(crate) fn fresh_identity() -> u64 {
        NEXT_HOST_CODE_ID.fetch_add(1, Ordering::Relaxed)
    }

    pub fn _hashable_identity(&self) -> u64 {
        self.id
    }

    /// RPython: `HostCode.__init__`.
    ///
    /// Structural note: RPython takes 13 positional arguments matching
    /// CPython's code-object constructor order. We mirror that order here
    /// so call-site translations stay line-by-line, even though a
    /// builder struct would be more idiomatic.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        argcount: u32,
        nlocals: u32,
        stacksize: u32,
        flags: u32,
        code: rustpython_compiler_core::bytecode::CodeUnits,
        consts: Vec<ConstantData>,
        names: Vec<String>,
        varnames: Vec<String>,
        filename: String,
        name: String,
        firstlineno: u32,
        lnotab: Vec<u8>,
        freevars: Vec<String>,
        cellvars: Vec<String>,
        exceptiontable: Box<[u8]>,
    ) -> Self {
        assert!(nlocals as i64 >= 0, "nlocals must be non-negative");
        let mut host = Self {
            id: Self::fresh_identity(),
            co_argcount: argcount,
            co_nlocals: nlocals,
            co_stacksize: stacksize,
            co_flags: flags,
            co_code: code,
            consts,
            names,
            co_varnames: varnames,
            co_freevars: freevars,
            co_cellvars: cellvars,
            co_filename: filename,
            co_name: name,
            co_firstlineno: firstlineno,
            co_lnotab: lnotab,
            exceptiontable,
            // Temporary placeholder until we fill it below — RPython
            // assigns inside the same __init__ call.
            signature: Signature::new(Vec::new(), None, None),
        };
        host.signature = cpython_code_signature(&host);
        host
    }

    /// Initialize the `HostCode` from a real (CPython) code object.
    ///
    /// RPython basis: `HostCode._from_code`.
    pub fn from_code(code: &CodeObject) -> Self {
        let nlocals = code
            .localspluskinds
            .len()
            .saturating_sub(code.cellvars.len())
            .saturating_sub(code.freevars.len()) as u32;
        let firstlineno = code.first_line_number.map_or(0, |n| n.get() as u32);
        Self::new(
            code.arg_count,
            nlocals,
            code.max_stackdepth,
            code.flags.bits(),
            code.instructions.clone(),
            code.constants.iter().cloned().collect(),
            code.names.to_vec(),
            code.varnames.to_vec(),
            code.source_path.clone(),
            code.obj_name.clone(),
            firstlineno,
            code.linetable.to_vec(),
            code.freevars.to_vec(),
            code.cellvars.to_vec(),
            code.exceptiontable.clone(),
        )
    }

    /// Total number of arguments passed into the frame, including
    /// `*vararg` and `**varkwarg` if they exist.
    ///
    /// RPython basis: `HostCode.formalargcount`.
    pub fn formalargcount(&self) -> usize {
        self.signature.scope_length()
    }

    /// RPython: `HostCode.is_generator`.
    pub fn is_generator(&self) -> bool {
        self.co_flags & CoFlags::GENERATOR.bits() != 0
    }

    /// Find the exception-table handler covering the given byte offset.
    ///
    /// `pypy/interpreter/pycode.py lookup_exceptiontable` parity:
    /// last-matching-wins, byte-offset I/O.  `HostCode.read` and flowspace
    /// operate in byte offsets (RPython's `co_code` walk convention), so
    /// the result is byte-offset natively.
    ///
    /// Mirrors `pyre_interpreter::pycode::lookup_exceptiontable`
    /// but lives here because `majit-translate` does not depend on
    /// `pyre-interpreter`; the varint decoder is small and the entry
    /// format is frozen by CPython 3.11 bytecode.
    pub fn find_exception_handler(&self, offset: u32) -> Option<ExceptionTableEntry> {
        if !offset.is_multiple_of(2) {
            return None;
        }
        let table = &self.exceptiontable;
        let n = table.len();
        let mut best: Option<ExceptionTableEntry> = None;
        let mut i = 0;
        while i < n {
            let (start_raw, ni) = decode_table_varint(table, i);
            let start = start_raw * 2;
            let (length_raw, ni) = decode_table_varint(table, ni);
            let length = length_raw * 2;
            let (target_raw, ni) = decode_table_varint(table, ni);
            let target = target_raw * 2;
            let (dl, ni) = decode_table_varint(table, ni);
            i = ni;
            if start <= offset && offset < start + length {
                best = Some(ExceptionTableEntry {
                    start,
                    end: start + length,
                    target,
                    depth: (dl >> 1) as u16,
                    push_lasti: (dl & 1) != 0,
                });
            } else if start > offset {
                break;
            }
        }
        best
    }

    /// Decode the entire exception table into byte-offset entries.
    pub fn decode_exception_table(&self) -> Vec<ExceptionTableEntry> {
        let table = &self.exceptiontable;
        let n = table.len();
        let mut entries = Vec::new();
        let mut i = 0;
        while i < n {
            let (start_raw, ni) = decode_table_varint(table, i);
            let start = start_raw * 2;
            let (length_raw, ni) = decode_table_varint(table, ni);
            let length = length_raw * 2;
            let (target_raw, ni) = decode_table_varint(table, ni);
            let target = target_raw * 2;
            let (dl, ni) = decode_table_varint(table, ni);
            i = ni;
            entries.push(ExceptionTableEntry {
                start,
                end: start + length,
                target,
                depth: (dl >> 1) as u16,
                push_lasti: (dl & 1) != 0,
            });
        }
        entries
    }
}

/// `pypy/interpreter/pycode.py _decode_varint`.  Reads 6 bits per
/// byte, MSB first; bit 6 is the continuation flag, bit 7 is the start-of-
/// entry marker (ignored — masked along with continuation via `& 63`).
#[inline]
fn decode_table_varint(table: &[u8], mut i: usize) -> (u32, usize) {
    let mut b = table[i] as u32;
    i += 1;
    let mut value = b & 63;
    while b & 64 != 0 {
        b = table[i] as u32;
        i += 1;
        value = (value << 6) | (b & 63);
    }
    (value, i)
}

impl HostCode {
    /// Decode the instruction starting at byte position `offset`.
    ///
    /// Returns `(next_offset, op, oparg)` where `next_offset` is the byte
    /// position of the following instruction, `op` is the decoded opcode,
    /// and `oparg` is the concatenated argument (with any preceding
    /// `EXTENDED_ARG` already folded in).
    ///
    /// RPython basis: `rpython/flowspace/bytecode.py:HostCode.read`.
    ///
    /// ### Deviation from RPython (parity rule #1)
    ///
    /// RPython returns the opcode **name** as a string (`opname`); we
    /// return the typed `Instruction` enum because CPython 3.15's
    /// wordcode format is enum-typed at the RustPython layer. The
    /// string name was used upstream to drive attribute lookup
    /// (`self.opcode_<NAME>(oparg)`), which is not idiomatic in Rust
    /// — the Phase 3 `flowcontext.rs` port matches on the enum
    /// directly. No information is lost.
    pub fn read(&self, offset: u32) -> Result<(u32, Instruction, u32), BytecodeCorruption> {
        let byte_offset = offset as usize;
        if !byte_offset.is_multiple_of(2) {
            return Err(BytecodeCorruption::new(format!(
                "unaligned bytecode offset {offset}"
            )));
        }
        let units: &[CodeUnit] = &self.co_code;
        let mut idx = byte_offset / 2;
        if idx >= units.len() {
            return Err(BytecodeCorruption::new(format!(
                "bytecode offset {offset} out of range (codeunits={})",
                units.len()
            )));
        }
        let mut arg_state = OpArgState::default();
        loop {
            let instruction_index = idx as u32;
            let unit = units[idx];
            let (op, oparg) = arg_state.get(unit);
            idx += 1;
            if !matches!(op, Instruction::ExtendedArg) {
                let next_offset = (idx * 2) as u32;
                let arg = absolutize_jump_target(instruction_index, &op, u32::from(oparg))
                    .unwrap_or_else(|| u32::from(oparg));
                return Ok((next_offset, op, arg));
            }
            if idx >= units.len() {
                return Err(BytecodeCorruption::new(
                    "ExtendedArg at end of bytecode stream",
                ));
            }
        }
    }
}

fn absolutize_jump_target(offset_units: u32, op: &Instruction, arg: u32) -> Option<u32> {
    let after = offset_units
        .checked_add(1)?
        .checked_add(op.cache_entries() as u32)?;
    let target_units = match op {
        Instruction::JumpForward { .. }
        | Instruction::PopJumpIfFalse { .. }
        | Instruction::PopJumpIfTrue { .. }
        | Instruction::PopJumpIfNone { .. }
        | Instruction::PopJumpIfNotNone { .. }
        | Instruction::ForIter { .. }
        | Instruction::Send { .. } => after.checked_add(arg)?,
        Instruction::JumpBackward { .. } | Instruction::JumpBackwardNoInterrupt { .. } => {
            after.checked_sub(arg)?
        }
        _ => return None,
    };
    target_units.checked_mul(2)
}

#[cfg(test)]
mod test {
    use super::*;
    use rustpython_compiler::{Mode, compile as rp_compile};

    fn compile_source(src: &str) -> CodeObject {
        rp_compile(src, Mode::Exec, "<flow>", Default::default()).expect("compile should succeed")
    }

    // RPython basis: test_objspace.py shapes these round-trips by calling
    // `build_flow_graph`. We pin the minimum field-copy invariants here
    // so F2.3 / F3 can rely on `HostCode::from_code` preserving upstream
    // attribute semantics. See F2.4 (upstream test ports) for the full
    // bytecode-shape coverage.

    #[test]
    fn from_code_preserves_basic_fields() {
        let code = compile_source("def f(x, y):\n    return x + y\n");
        // Extract the inner function's code object.
        let inner = code
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body should be a code constant");
        let host = HostCode::from_code(inner);
        assert_eq!(host.co_argcount, 2);
        assert_eq!(host.co_name, "f");
        assert_eq!(host.co_varnames[0], "x");
        assert_eq!(host.co_varnames[1], "y");
        assert_eq!(host.formalargcount(), 2);
        assert!(!host.is_generator());
        assert_eq!(host.signature.argnames, vec!["x".to_string(), "y".into()]);
        assert!(host.signature.varargname.is_none());
        assert!(host.signature.kwargname.is_none());
    }

    #[test]
    fn signature_tracks_varargs_and_kwargs() {
        let code = compile_source("def f(a, *args, **kw):\n    return a\n");
        let inner = code
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body");
        let host = HostCode::from_code(inner);
        assert_eq!(host.signature.argnames, vec!["a".to_string()]);
        assert_eq!(host.signature.varargname.as_deref(), Some("args"));
        assert_eq!(host.signature.kwargname.as_deref(), Some("kw"));
        assert_eq!(host.formalargcount(), 3);
    }

    #[test]
    fn is_generator_detects_yield() {
        let code = compile_source("def g():\n    yield 1\n");
        let inner = code
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body");
        let host = HostCode::from_code(inner);
        assert!(host.is_generator());
    }

    // RPython basis: bytecode.py:HostCode.read. Upstream has no standalone
    // test for `read`; the shape is verified indirectly through
    // test_objspace.py:build_flow_graph. These pin the wordcode-level
    // decoder directly so F3 (flowcontext.py port) can rely on the
    // `(next_offset, Instruction, oparg)` shape.

    #[test]
    fn read_walks_each_instruction_in_order() {
        let code = compile_source("def f(x):\n    return x\n");
        let inner = code
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body");
        let host = HostCode::from_code(inner);
        let mut offset = 0u32;
        let mut seen = Vec::new();
        let total = (host.co_code.len() * 2) as u32;
        while offset < total {
            let (next, op, _arg) = host.read(offset).expect("decoder");
            assert!(next > offset, "read must advance");
            seen.push(op);
            offset = next;
        }
        assert!(!seen.is_empty());
        // Last instruction of a `return x` body is ReturnValue on CPython
        // 3.15 wordcode.
        assert!(
            matches!(seen.last(), Some(Instruction::ReturnValue)),
            "expected terminal ReturnValue, got {:?}",
            seen.last()
        );
    }

    #[test]
    fn read_rejects_unaligned_offset() {
        let code = compile_source("def f():\n    return 1\n");
        let inner = code
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body");
        let host = HostCode::from_code(inner);
        let err = host.read(1).expect_err("unaligned offset must fail");
        assert!(
            err.to_string().contains("unaligned"),
            "expected unaligned diagnostic, got {err}"
        );
    }

    #[test]
    fn read_rejects_out_of_range_offset() {
        let code = compile_source("def f():\n    return 1\n");
        let inner = code
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body");
        let host = HostCode::from_code(inner);
        let past_end = (host.co_code.len() * 2) as u32;
        assert!(host.read(past_end).is_err());
    }

    #[test]
    fn read_absolutizes_relative_conditional_jump_targets() {
        let code = compile_source("def f(x):\n    if x:\n        return 1\n    return 0\n");
        let inner = code
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body");
        let host = HostCode::from_code(inner);
        let total = (host.co_code.len() * 2) as u32;
        let mut offset = 0u32;
        let mut saw_branch = false;
        while offset < total {
            let (next, op, arg) = host.read(offset).expect("decoder");
            if matches!(
                op,
                Instruction::PopJumpIfFalse { .. }
                    | Instruction::PopJumpIfTrue { .. }
                    | Instruction::PopJumpIfNone { .. }
                    | Instruction::PopJumpIfNotNone { .. }
            ) {
                assert!(arg > next, "relative jump target should be absolutized");
                saw_branch = true;
                break;
            }
            offset = next;
        }
        assert!(
            saw_branch,
            "expected a conditional jump in the function body"
        );
    }
}
