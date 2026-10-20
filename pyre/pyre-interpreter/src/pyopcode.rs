use crate::bytecode::{
    BinaryOperator, BuildSliceArgCount, CodeObject, CodeUnit, CommonConstant, ComparisonOperator,
    ConstantData, ConvertValueOparg, Instruction, IntrinsicFunction1, IntrinsicFunction2, Invert,
    MakeFunctionFlag, OpArg, OpArgState, RaiseKind, SpecialMethod,
};

use crate::{
    PyBigInt, PyError, SharedOpcodeHandler, opcode_build_list, opcode_build_list_from_arg,
    opcode_build_map, opcode_build_tuple, opcode_call, opcode_list_append, opcode_load_attr,
    opcode_make_function, opcode_store_attr, opcode_store_subscr, opcode_unpack_sequence,
};

use pyre_object::PyObjectRef;

pub type PyUnaryOpHandler = fn(&mut crate::pyframe::PyFrame);
pub type PyBinaryOpHandler = fn(&mut crate::pyframe::PyFrame);

#[allow(non_camel_case_types)]
pub struct __extend__;

pub struct ExitFrame;

pub struct Return;

pub struct Yield;

/// pypy/interpreter/pyopcode.py — `RaiseWithExplicitTraceback(operr, lasti)`.
///
/// Carries the original raise-site `lasti` (instruction byte offset) from a
/// `RERAISE N` so the next exception-table dispatch can push that offset as
/// the handler's `lasti` value instead of the RERAISE instruction itself.
/// `lasti == -1` means no reraise lasti (default for primary raises).
///
/// pyre also routes this value through `PyError.reraise_lasti` because the
/// interpreter dispatch loop uses a single `Err(PyError)` channel rather
/// than two distinct exception classes; this struct exists for line-by-line
/// parity surface.
pub struct RaiseWithExplicitTraceback {
    pub operr: crate::error::OperationError,
    pub lasti: i32,
}

pub struct SuspendedUnroller {
    pub kind: usize,
}

pub struct SReturnValue {
    pub w_returnvalue: PyObjectRef,
}

pub struct SApplicationException {
    pub operr: crate::error::OperationError,
}

pub struct SBreakLoop {
    pub singleton: bool,
}

pub struct SContinueLoop {
    pub jump_to: usize,
}

#[inline]
pub fn unaryoperation(_operationname: &str) -> PyUnaryOpHandler {
    let _ = _operationname;
    _stub_unaryoperation
}

#[inline]
pub fn binaryoperation(_operationname: &str) -> PyBinaryOpHandler {
    let _ = _operationname;
    _stub_binaryoperation
}

fn _stub_unaryoperation(_frame: &mut crate::pyframe::PyFrame) {
    let _ = _frame;
}

fn _stub_binaryoperation(_frame: &mut crate::pyframe::PyFrame) {
    let _ = _frame;
}

pub enum StepResult<V> {
    Continue,
    Return(V),
    CloseLoop {
        /// The back edge's target pc.  `PyFrame.jump_absolute`
        /// (`interp_jit.py`) reports a back edge by calling
        /// `pypyjitdriver.can_enter_jit` with named greens and reds and
        /// returning the pc; there is no argument vector at this layer.
        loop_header_pc: usize,
    },
    Yield(V),
}

pub fn decode_instruction_at(code: &CodeObject, pc: usize) -> Option<(Instruction, OpArg)> {
    // closure-free, Option-pattern-free `instructions.get(pc)` rewrite — see
    // execute_load_fast for the rationale; keeps the bounds check a plain
    // `lt + getitem` instead of walking into `Option<&CodeUnit>`.
    if pc >= code_instructions_len(code) {
        return None;
    }
    let code_unit = code.instructions[pc];
    let mut start = pc;
    while start > 0 {
        let prev = code.instructions[start - 1];
        if matches!(prev.op, Instruction::ExtendedArg) {
            start -= 1;
        } else {
            break;
        }
    }

    let mut arg_state = OpArgState::default();
    for idx in start..pc {
        let _ = arg_state.get(code.instructions[idx]);
    }
    Some(arg_state.get(code_unit))
}

/// Whether `code` contains a `FOR_ITER` opcode — i.e. it is loop-bearing
/// through a `for` loop.  Used to gate the nested-inline decline: a residual
/// reached inside a loop-bearing inline callee risks the FOR_ITER-exempt
/// double-advance on a refused Option-C delivery.
pub fn code_has_for_iter(code: &CodeObject) -> bool {
    let mut arg_state = OpArgState::default();
    for unit in code.instructions.iter().copied() {
        let (instruction, _) = arg_state.get(unit);
        if matches!(instruction, Instruction::ForIter { .. }) {
            return true;
        }
    }
    false
}

/// Whether `code` is directly self-recursive: it references its own name via
/// `LOAD_GLOBAL <own name>`, the shape a module-level function calling itself
/// compiles to.  A hot self-recursion forms a `CALL_ASSEMBLER` bridge, which
/// the nested-inline decline refuses (its moving-nursery callee frame cannot
/// survive the residual trampoline).  Misses indirect/closure recursion (a
/// conservative false-negative), never a false positive beyond a function that
/// names itself without calling (which only over-declines, still correct).
pub fn code_is_self_recursive(code: &CodeObject) -> bool {
    let own_name = code.obj_name.as_str();
    let mut arg_state = OpArgState::default();
    for unit in code.instructions.iter().copied() {
        let (instruction, op_arg) = arg_state.get(unit);
        if let Instruction::LoadGlobal { namei } = instruction {
            let name_idx = (namei.get(op_arg) as usize) >> 1;
            // closure-free, Option-pattern-free `names.get(name_idx)` rewrite —
            // keeps the bounds check a plain `lt + getitem`.
            if name_idx < code.names.len() && code.names[name_idx].as_str() == own_name {
                return true;
            }
        }
    }
    false
}

/// Whether every `RETURN_VALUE` in `code` returns a constant.
///
/// True for a function with no `return <expr>` at all: its implicit
/// `return None` compiles to `LOAD_COMMON_CONSTANT None; RETURN_VALUE`, and
/// that is the only return it has.  `CACHE` and `EXTENDED_ARG` units are
/// transparent, so a cached or extended `LOAD_CONST` still reads as the
/// producer.
///
/// The caller uses this to ask whether a value built in this frame can be
/// handed back to the frame's caller.  It answers only for the return value:
/// a store to a global, a cell, or a container the caller reads is an escape
/// this does not see, so a false answer means "cannot escape *by returning*",
/// not "cannot escape".
pub fn code_returns_only_constants(code: &CodeObject) -> bool {
    let mut previous: Option<Instruction> = None;
    let mut arg_state = OpArgState::default();
    for unit in code.instructions.iter().copied() {
        let (instruction, _) = arg_state.get(unit);
        match instruction {
            Instruction::ReturnValue => {
                if !matches!(
                    previous,
                    Some(Instruction::LoadConst { .. } | Instruction::LoadCommonConstant { .. })
                ) {
                    return false;
                }
            }
            Instruction::Cache | Instruction::ExtendedArg => continue,
            _ => {}
        }
        previous = Some(instruction);
    }
    true
}

/// pypy/interpreter/pyopcode.py dispatch_bytecode parity.
///
/// Returns the semantic opcode to execute for a dispatch step starting at
/// `pc`. If `pc` points at one or more `EXTENDED_ARG` prefixes, this walks
/// forward to the real opcode and returns its fully accumulated oparg along
/// with the real opcode PC.
pub fn decode_instruction_for_dispatch(
    code: &CodeObject,
    pc: usize,
) -> Result<(usize, Instruction, OpArg), crate::pycode::BytecodeCorruption> {
    let mut opcode_pc = pc;
    let (mut instruction, mut op_arg) =
        decode_instruction_at(code, opcode_pc).ok_or(crate::pycode::BytecodeCorruption)?;
    while matches!(instruction, Instruction::ExtendedArg) {
        opcode_pc += 1;
        let decoded =
            decode_instruction_at(code, opcode_pc).ok_or(crate::pycode::BytecodeCorruption)?;
        instruction = decoded.0;
        op_arg = decoded.1;
        // `Reserved` is excluded for the same reason the forward decoder
        // excludes it: an unknown opcode byte is preserved here so dispatch
        // reaches it and reports `SystemError: unknown opcode N`.
        if !matches!(instruction, Instruction::ExtendedArg)
            && instruction_discriminant(instruction) < 44
            && !matches!(instruction, Instruction::Reserved)
        {
            return Err(crate::pycode::BytecodeCorruption);
        }
    }
    Ok((opcode_pc, instruction, op_arg))
}

/// pypy/interpreter/pyopcode.py dispatch loop forward decode.
///
/// Single-pass decode for the hot dispatch path: returns the real opcode's
/// `(pc, instruction, op_arg)` for a `pc` that names either a logical
/// instruction start (a real opcode, or the first `ExtendedArg` of a prefix
/// run) or a real opcode sitting *past* its own `ExtendedArg` prefix.
///
/// `OpArgState` resets its accumulator after every real opcode, so the fully
/// accumulated oparg at a real opcode depends only on the contiguous
/// `ExtendedArg` run immediately preceding it. The interpreter reaches a real
/// opcode by fall-through from its `ExtendedArg` prefix (the logical start),
/// but the JIT dispatch loop enters at a coordinate normalized through
/// `skip_python_trivia_forward`, which advances past `ExtendedArg` units (they
/// are trivia in pyre's pc-indexed metadata keying) and so names the real
/// opcode *past* its prefix. This preserves the backward-scanning
/// `decode_instruction_at` contract for both: back up to the logical start
/// over any immediately-preceding `ExtendedArg` units, then accumulate forward
/// with a fresh state. The back-up loop body runs zero times in the common
/// no-prefix case (the unit before a real opcode is not `ExtendedArg`), so the
/// hot path stays a single forward pass.
///
/// Shares `decode_instruction_for_dispatch`'s `u8 < 44` malformed-chain guard
/// and returns `BytecodeCorruption` on out of bounds.
/// Both loops are bounded by the `ExtendedArg` prefix chain, which is at most three units.
#[inline]
#[majit_macros::unroll_safe]
pub fn decode_instruction_forward(
    code: &CodeObject,
    pc: usize,
) -> Result<(usize, Instruction, OpArg), crate::pycode::BytecodeCorruption> {
    // Back up over any `ExtendedArg` prefix so accumulation starts at the
    // logical instruction start. Zero iterations when `pc` is already a
    // logical start (no preceding `ExtendedArg`). The `start - 1 < len` guard
    // is a closure-free, Option-pattern-free rewrite of
    // `.get(start - 1).is_some_and(..)` (see execute_load_fast for the
    // rationale), keeping the bounds check a plain `lt + getitem`.
    let mut start = pc;
    while start > 0
        && start - 1 < code_instructions_len(code)
        && matches!(
            instruction_from_code_unit_word(code_unit_at(code, start - 1)),
            Instruction::ExtendedArg
        )
    {
        start -= 1;
    }

    let mut opcode_pc = start;
    let mut op_arg = 0u32;
    loop {
        if opcode_pc >= code_instructions_len(code) {
            return Err(crate::pycode::BytecodeCorruption);
        }
        let word = code_unit_at(code, opcode_pc);
        let instruction = instruction_from_code_unit_word(word);
        op_arg = (op_arg << 8) | u32::from(word >> 8);
        if matches!(instruction, Instruction::ExtendedArg) {
            opcode_pc += 1;
            continue;
        }
        if opcode_pc != start
            && instruction_discriminant(instruction) < 44
            && !matches!(instruction, Instruction::Reserved)
        {
            return Err(crate::pycode::BytecodeCorruption);
        }
        return Ok((opcode_pc, instruction, OpArg::new(op_arg)));
    }
}

/// Scalar-PC half of [`decode_instruction_forward`] for the JIT dispatch loop.
/// `usize::MAX` represents the same bytecode-corruption cases as the public decoder.
#[inline]
#[majit_macros::unroll_safe]
pub fn decode_instruction_forward_pc(code: &CodeObject, pc: usize) -> usize {
    let mut start = pc;
    while start > 0
        && start - 1 < code_instructions_len(code)
        && matches!(
            instruction_from_code_unit_word(code_unit_at(code, start - 1)),
            Instruction::ExtendedArg
        )
    {
        start -= 1;
    }

    let mut opcode_pc = start;
    loop {
        if opcode_pc >= code_instructions_len(code) {
            return usize::MAX;
        }
        let instruction = instruction_from_code_unit_word(code_unit_at(code, opcode_pc));
        if matches!(instruction, Instruction::ExtendedArg) {
            opcode_pc += 1;
            continue;
        }
        if opcode_pc != start
            && instruction_discriminant(instruction) < 44
            && !matches!(instruction, Instruction::Reserved)
        {
            return usize::MAX;
        }
        return opcode_pc;
    }
}

/// Scalar opcode/oparg half of [`decode_instruction_forward`] for JIT dispatch.
/// `u64::MAX` is the corruption sentinel; valid results use bits 0..8 for the
/// opcode and bits 8..40 for the accumulated oparg.
#[inline]
#[majit_macros::unroll_safe]
pub fn decode_instruction_forward_packed(code: &CodeObject, pc: usize) -> u64 {
    let mut start = pc;
    while start > 0
        && start - 1 < code_instructions_len(code)
        && matches!(
            instruction_from_code_unit_word(code_unit_at(code, start - 1)),
            Instruction::ExtendedArg
        )
    {
        start -= 1;
    }

    let mut opcode_pc = start;
    let mut op_arg = 0u32;
    loop {
        if opcode_pc >= code_instructions_len(code) {
            return u64::MAX;
        }
        let word = code_unit_at(code, opcode_pc);
        let instruction = instruction_from_code_unit_word(word);
        op_arg = (op_arg << 8) | u32::from(word >> 8);
        if matches!(instruction, Instruction::ExtendedArg) {
            opcode_pc += 1;
            continue;
        }
        if opcode_pc != start
            && instruction_discriminant(instruction) < 44
            && !matches!(instruction, Instruction::Reserved)
        {
            return u64::MAX;
        }
        return u64::from(word & 0xff) | (u64::from(op_arg) << 8);
    }
}

/// One walk of [`decode_instruction_forward`] for `eval_loop_jit`.
///
/// `pyopcode.py` `dispatch_bytecode` reads `co_code[next_instr]` once per
/// opcode. The PC projection and the opcode/oparg projection each repeat
/// that walk; this returns both from the single pass. `usize::MAX` /
/// `u64::MAX` are the corruption sentinels of the two projections.
#[inline]
pub fn decode_instruction_forward_dispatch(code: &CodeObject, pc: usize) -> (usize, u64) {
    let mut start = pc;
    while start > 0
        && start - 1 < code_instructions_len(code)
        && matches!(
            instruction_from_code_unit_word(code_unit_at(code, start - 1)),
            Instruction::ExtendedArg
        )
    {
        start -= 1;
    }

    let mut opcode_pc = start;
    let mut op_arg = 0u32;
    loop {
        if opcode_pc >= code_instructions_len(code) {
            return (usize::MAX, u64::MAX);
        }
        let word = code_unit_at(code, opcode_pc);
        let instruction = instruction_from_code_unit_word(word);
        op_arg = (op_arg << 8) | u32::from(word >> 8);
        if matches!(instruction, Instruction::ExtendedArg) {
            opcode_pc += 1;
            continue;
        }
        if opcode_pc != start
            && u8::from(instruction) < 44
            && !matches!(instruction, Instruction::Reserved)
        {
            return (usize::MAX, u64::MAX);
        }
        let packed = u64::from(word & 0xff) | (u64::from(op_arg) << 8);
        return (opcode_pc, packed);
    }
}

pub trait LocalOpcodeHandler: SharedOpcodeHandler {
    fn load_local_value(&mut self, idx: usize) -> Result<Self::Value, PyError>;
    fn load_local_checked_value(&mut self, idx: usize) -> Result<Self::Value, PyError> {
        let value = self.load_local_value(idx)?;
        self.guard_nonnull_value(value)?;
        Ok(value)
    }
    fn store_local_value(&mut self, idx: usize, value: Self::Value) -> Result<(), PyError>;
}

pub trait NamespaceOpcodeHandler: SharedOpcodeHandler {
    /// `pyopcode.py LOAD_NAME` / `:561 LOAD_GLOBAL` — `nameindex`
    /// is the `co_names` index decoded from the bytecode operand,
    /// passed through so `pycode._globals_caches[nameindex]`
    /// (`celldict.py:292`) can be consulted without re-resolving from
    /// `name`.  PyFrame uses it; JIT trace `MIFrame` ignores it.
    fn load_name_value(&mut self, name: &str, nameindex: usize) -> Result<Self::Value, PyError>;
    fn load_name_checked_value(
        &mut self,
        name: &str,
        nameindex: usize,
    ) -> Result<Self::Value, PyError> {
        let value = self.load_name_value(name, nameindex)?;
        self.guard_nonnull_value(value)?;
        Ok(value)
    }
    fn store_name_value(
        &mut self,
        name: &str,
        nameindex: usize,
        value: Self::Value,
    ) -> Result<(), PyError>;
    /// PyPy STORE_GLOBAL writes to `w_globals` (`pyopcode.py STORE_GLOBAL`).
    /// Default mirrors STORE_NAME so implementations that conflate the two
    /// namespaces (e.g. JIT trace `MIFrame`) keep their existing behaviour;
    /// PyFrame overrides to bypass `w_locals` and write directly to globals.
    fn store_global_value(
        &mut self,
        name: &str,
        nameindex: usize,
        value: Self::Value,
    ) -> Result<(), PyError> {
        self.store_name_value(name, nameindex, value)
    }
    /// PyPy LOAD_GLOBAL skips `w_locals` (`pyopcode.py LOAD_GLOBAL`).
    /// Default mirrors LOAD_NAME for the conflating implementations;
    /// PyFrame overrides to read from `w_globals` only.
    fn load_global_value(&mut self, name: &str, nameindex: usize) -> Result<Self::Value, PyError> {
        self.load_name_value(name, nameindex)
    }
    fn load_global_checked_value(
        &mut self,
        name: &str,
        nameindex: usize,
    ) -> Result<Self::Value, PyError> {
        let value = self.load_global_value(name, nameindex)?;
        self.guard_nonnull_value(value)?;
        Ok(value)
    }
    fn null_value(&mut self) -> Result<Self::Value, PyError>;
}

pub trait StackOpcodeHandler: SharedOpcodeHandler {
    fn swap_values(&mut self, depth: usize) -> Result<(), PyError>;
}

pub trait IterOpcodeHandler: SharedOpcodeHandler {
    /// PyPy `GET_ITER`: `w_iterable = popvalue(); pushvalue(space.iter(w_iterable))`.
    fn iter_value(&mut self, iterable: Self::Value) -> Result<Self::Value, PyError>;
    /// Legacy in-place conversion hook retained while out-of-tree opcode
    /// executors migrate to `iter_value`; the shared opcode no longer calls it.
    fn ensure_iter_value(&mut self, iter: Self::Value) -> Result<(), PyError>;
    // FOR_ITER drives a single space.next (pyopcode.py).
    // Ok(Some(v)) = next value; Ok(None) = StopIteration (exhausted);
    // Err(e) = a non-StopIteration exception (propagate).
    fn iter_next(&mut self, iter: Self::Value) -> Result<Option<Self::Value>, PyError>;
    fn guard_optional_value(
        &mut self,
        value: Self::Value,
        expect_some: bool,
    ) -> Result<(), PyError> {
        let _ = (value, expect_some);
        Ok(())
    }

    fn record_for_iter_guard(&mut self, next: Self::Value, continues: bool) -> Result<(), PyError> {
        self.guard_optional_value(next, continues)
    }

    fn record_for_iter_guard_exhausted(&mut self) -> Result<(), PyError> {
        Ok(())
    }

    /// Jump the anchored handler to `target` on FOR_ITER exhaustion.
    /// Deliberately takes no `self`: after `iter_next` the caller's `&mut Self`
    /// may name the abandoned copy, so the anchor is the only sound way back
    /// to the live one.
    fn on_iter_exhausted_anchored(anchor: &Self::Anchor, target: usize) -> Result<(), PyError>;
}

pub trait TruthOpcodeHandler: SharedOpcodeHandler {
    type Truth: Copy;

    fn truth_value(&mut self, value: Self::Value) -> Result<Self::Truth, PyError>;
    fn bool_value_from_truth(
        &mut self,
        truth: Self::Truth,
        negate: bool,
    ) -> Result<Self::Value, PyError>;
    fn unary_not_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError>;
}

pub trait ControlFlowOpcodeHandler: SharedOpcodeHandler {
    fn fallthrough_target(&mut self) -> usize;
    fn set_next_instr(&mut self, target: usize) -> Result<(), PyError>;
    /// `PyFrame.jump_absolute` (`pyopcode.py`) — the non-JIT base returns the
    /// pc and reports no back edge.  `PyFrame` overrides this with the
    /// `interp_jit.py` form, which does.
    fn close_loop(&mut self, _target: usize) -> Result<StepResult<Self::Value>, PyError> {
        Ok(StepResult::Continue)
    }

    fn finish_value(&mut self, value: Self::Value) -> Result<StepResult<Self::Value>, PyError> {
        Ok(StepResult::Return(value))
    }
}

pub trait BranchOpcodeHandler: TruthOpcodeHandler + ControlFlowOpcodeHandler {
    fn enter_branch_truth(&mut self, value: Self::Value) -> Result<(), PyError> {
        let _ = value;
        Ok(())
    }

    fn leave_branch_truth(&mut self) -> Result<(), PyError> {
        Ok(())
    }

    fn concrete_truth_as_bool(
        &mut self,
        value: Self::Value,
        truth: Self::Truth,
    ) -> Result<bool, PyError>;
    fn guard_truth_value(&mut self, truth: Self::Truth, expect_true: bool) -> Result<(), PyError> {
        let _ = (truth, expect_true);
        Ok(())
    }

    fn record_branch_guard(
        &mut self,
        value: Self::Value,
        truth: Self::Truth,
        concrete_truth: bool,
        other_target: usize,
    ) -> Result<(), PyError> {
        let _ = (value, other_target);
        self.guard_truth_value(truth, concrete_truth)
    }
}

pub trait ArithmeticOpcodeHandler: SharedOpcodeHandler {
    fn binary_value(
        &mut self,
        a: Self::Value,
        b: Self::Value,
        op: BinaryOperator,
    ) -> Result<Self::Value, PyError>;
    fn compare_value(
        &mut self,
        a: Self::Value,
        b: Self::Value,
        op: ComparisonOperator,
    ) -> Result<Self::Value, PyError>;
    fn unary_negative_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError>;
    fn unary_invert_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError>;
}

pub trait ConstantOpcodeHandler: SharedOpcodeHandler<Value = PyObjectRef> {
    fn int_constant(&mut self, value: i64) -> Result<Self::Value, PyError>;
    /// Interned box for `LOAD_SMALL_INT`. Default is a fresh
    /// [`Self::int_constant`]; `PyFrame` returns the process-lifetime
    /// identity so JUMP can carry `ConstPtr` the way `getconstant_w`
    /// does for `LOAD_CONST`.
    fn small_int_constant(&mut self, value: i64) -> Result<Self::Value, PyError> {
        self.int_constant(value)
    }
    fn bigint_constant(&mut self, value: &PyBigInt) -> Result<Self::Value, PyError>;
    fn float_constant(&mut self, value: f64) -> Result<Self::Value, PyError>;
    fn complex_constant(&mut self, re: f64, im: f64) -> Result<Self::Value, PyError>;
    fn bool_constant(&mut self, value: bool) -> Result<Self::Value, PyError>;
    fn str_constant(&mut self, value: &rustpython_wtf8::Wtf8) -> Result<Self::Value, PyError>;
    /// Bytes literal — pyre stores immutable bytes values as bytearray.
    /// PyPy: Bytes literals create W_BytesObject, but pyre lacks a separate
    /// bytes type so we route through W_BytearrayObject (mutable) and rely
    /// on call sites that need true immutability to make a copy.
    fn bytes_constant(&mut self, value: &[u8]) -> Result<Self::Value, PyError>;
    fn code_constant(&mut self, code: &CodeObject) -> Result<Self::Value, PyError>;
    /// `getconstant_w(index) -> co_consts_w[index]` (`pyopcode.py`).
    /// The default realizes from the compiler constant; `PyFrame` overrides it
    /// to return the object owned by `self.pycode.co_consts_w[index]`.
    fn constant_at(
        &mut self,
        index: crate::bytecode::oparg::ConstIdx,
        enclosing: &CodeObject,
    ) -> Result<Self::Value, PyError> {
        load_const_value(self, &enclosing.constants[index])
    }
    fn none_constant(&mut self) -> Result<Self::Value, PyError>;
    fn ellipsis_constant(&mut self) -> Result<Self::Value, PyError>;
    fn slice_constant(
        &mut self,
        start: Self::Value,
        stop: Self::Value,
        step: Self::Value,
    ) -> Result<Self::Value, PyError> {
        // Default: build as tuple. PyFrame overrides to create W_SliceObject.
        self.build_tuple(&[start, stop, step])
    }
    fn frozenset_constant(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        // Default: build as tuple. PyFrame overrides to create W_FrozenSetObject.
        self.build_tuple(items)
    }
}

/// Realize each nested constant and keep it rooted while later siblings
/// mint. Each element is freshly minted, then pinned; `take` reloads the
/// live words and the bracket stays open across the container constructor.
fn load_const_rooted_items<H: ConstantOpcodeHandler + ?Sized>(
    handler: &mut H,
    elements: &[ConstantData],
) -> Result<pyre_object::gc_roots::RootedItems, PyError> {
    let mut items = pyre_object::gc_roots::RootedItems::new();
    for element in elements {
        items.push(load_const_value(handler, element)?);
    }
    Ok(items)
}

fn load_const_value<H: ConstantOpcodeHandler + ?Sized>(
    handler: &mut H,
    constant: &ConstantData,
) -> Result<H::Value, PyError> {
    match constant {
        ConstantData::Integer { value } => {
            // The compiler's Malachite integer is a serialization/API seam,
            // never the interpreter arithmetic representation.  Convert once
            // to RPython's rbigint, then make the same machine-int decision
            // through `rbigint.toint()` that PyPy's integer consumers use.
            // This also keeps the generated JIT graph entirely on the
            // sanctioned compiler-bigint conversion residual instead of
            // pulling a foreign `BigInt::to_i64 -> Option<i64>` method into
            // translated code.
            let value = crate::compiler_bigint_to_rbigint(value);
            match value.toint() {
                Ok(value) => handler.int_constant(value),
                Err(_) => handler.bigint_constant(&value),
            }
        }
        ConstantData::Float { value } => handler.float_constant(*value),
        ConstantData::Boolean { value } => handler.bool_constant(*value),
        ConstantData::Str { value } => handler.str_constant(value),
        ConstantData::Tuple { elements } => {
            let items = load_const_rooted_items(handler, elements)?;
            handler.build_tuple(&items.take())
        }
        ConstantData::Code { code } => handler.code_constant(code),
        ConstantData::None => handler.none_constant(),
        ConstantData::Ellipsis => handler.ellipsis_constant(),
        ConstantData::Bytes { value } => handler.bytes_constant(value),
        ConstantData::Complex { value } => handler.complex_constant(value.re, value.im),
        ConstantData::Frozenset { elements } => {
            let items = load_const_rooted_items(handler, elements)?;
            handler.frozenset_constant(&items.take())
        }
        ConstantData::Slice { elements } => {
            // Slice constant → build start/stop/step via handler.slice_constant()
            let items = load_const_rooted_items(handler, &**elements)?;
            let taken = items.take();
            if taken.len() == 3 {
                handler.slice_constant(taken[0], taken[1], taken[2])
            } else {
                handler.build_tuple(&taken)
            }
        }
    }
}

pub fn opcode_load_const<H: ConstantOpcodeHandler + ?Sized>(
    handler: &mut H,
    constant: &ConstantData,
) -> Result<(), PyError> {
    let value = load_const_value(handler, constant)?;
    handler.push_value(value)
}

pub fn opcode_load_small_int<H: ConstantOpcodeHandler + ?Sized>(
    handler: &mut H,
    value: i64,
) -> Result<(), PyError> {
    let value = handler.small_int_constant(value)?;
    handler.push_value(value)
}

pub fn opcode_load_fast_checked<H: LocalOpcodeHandler + ?Sized>(
    handler: &mut H,
    idx: usize,
) -> Result<(), PyError> {
    let value = handler.load_local_checked_value(idx)?;
    handler.push_value(value)
}

pub fn opcode_load_fast_pair_checked<H: LocalOpcodeHandler + ?Sized>(
    handler: &mut H,
    idx1: usize,
    idx2: usize,
) -> Result<(), PyError> {
    let v1 = handler.load_local_checked_value(idx1)?;
    let v2 = handler.load_local_checked_value(idx2)?;
    handler.push_value(v1)?;
    handler.push_value(v2)
}

pub fn opcode_store_fast<H: LocalOpcodeHandler + ?Sized>(
    handler: &mut H,
    idx: usize,
) -> Result<(), PyError> {
    let value = handler.pop_value()?;
    handler.store_local_value(idx, value)
}

pub fn opcode_load_fast_load_fast<H: LocalOpcodeHandler + ?Sized>(
    handler: &mut H,
    idx1: usize,
    idx2: usize,
) -> Result<(), PyError> {
    let v1 = handler.load_local_value(idx1)?;
    let v2 = handler.load_local_value(idx2)?;
    handler.push_value(v1)?;
    handler.push_value(v2)
}

pub fn opcode_store_fast_load_fast<H: LocalOpcodeHandler + ?Sized>(
    handler: &mut H,
    store_idx: usize,
    load_idx: usize,
) -> Result<(), PyError> {
    let value = handler.pop_value()?;
    handler.store_local_value(store_idx, value)?;
    let loaded = handler.load_local_value(load_idx)?;
    handler.push_value(loaded)
}

pub fn opcode_store_fast_store_fast<H: LocalOpcodeHandler + ?Sized>(
    handler: &mut H,
    idx1: usize,
    idx2: usize,
) -> Result<(), PyError> {
    let v1 = handler.pop_value()?;
    let v2 = handler.pop_value()?;
    handler.store_local_value(idx1, v1)?;
    handler.store_local_value(idx2, v2)
}

/// The `nameindex` a caller passes when it addresses no `co_names_w` slot
/// (`pycode.py:127-129`).
///
/// `0` cannot say this — it is a valid index, and naming the first entry of the
/// name table is exactly the wrong answer.  Out of range for every table, so
/// `w_code_getname_w` resolves it to `PY_NULL` and the key is minted the way it
/// was before the table existed.
///
/// The caller that needs it is the implicit class-body `__class__` store, whose
/// name is a literal rather than a `co_names` entry.  (The JIT's
/// `bh_store_name_fn` / `bh_store_global_fn` residuals also hold no index, but
/// they are handed an already-wrapped `w_name` and enter through
/// `eval::store_name_value_w`, which needs no index at all.)
pub const NO_NAMEINDEX: usize = usize::MAX;

pub fn opcode_store_name<H: NamespaceOpcodeHandler + ?Sized>(
    handler: &mut H,
    name: &str,
    nameindex: usize,
) -> Result<(), PyError> {
    let value = handler.pop_value()?;
    handler.store_name_value(name, nameindex, value)
}

/// pypy/interpreter/pyopcode.py STORE_GLOBAL — writes the TOS into
/// `w_globals` regardless of `w_locals`.
pub fn opcode_store_global<H: NamespaceOpcodeHandler + ?Sized>(
    handler: &mut H,
    name: &str,
    nameindex: usize,
) -> Result<(), PyError> {
    let value = handler.pop_value()?;
    handler.store_global_value(name, nameindex, value)
}

pub fn opcode_load_name<H: NamespaceOpcodeHandler + ?Sized>(
    handler: &mut H,
    name: &str,
    nameindex: usize,
) -> Result<(), PyError> {
    let value = handler.load_name_checked_value(name, nameindex)?;
    handler.push_value(value)
}

pub fn opcode_load_global<H: NamespaceOpcodeHandler + ?Sized>(
    handler: &mut H,
    name: &str,
    nameindex: usize,
    push_null: bool,
) -> Result<(), PyError> {
    let value = handler.load_global_checked_value(name, nameindex)?;
    handler.push_value(value)?;
    if push_null {
        let null = handler.null_value()?;
        handler.push_value(null)?;
    }
    Ok(())
}

pub fn opcode_pop_top<H: SharedOpcodeHandler + ?Sized>(handler: &mut H) -> Result<(), PyError> {
    let _ = handler.pop_value()?;
    Ok(())
}

pub fn opcode_push_null<H: NamespaceOpcodeHandler + ?Sized>(
    handler: &mut H,
) -> Result<(), PyError> {
    let null = handler.null_value()?;
    handler.push_value(null)
}

pub fn opcode_copy_value<H: SharedOpcodeHandler + ?Sized>(
    handler: &mut H,
    depth: usize,
) -> Result<(), PyError> {
    let value = handler.peek_at(depth - 1)?;
    handler.push_value(value)
}

pub fn opcode_swap<H: StackOpcodeHandler + ?Sized>(
    handler: &mut H,
    depth: usize,
) -> Result<(), PyError> {
    handler.swap_values(depth)
}

pub fn opcode_get_iter<H: IterOpcodeHandler + ?Sized>(handler: &mut H) -> Result<(), PyError> {
    let iterable = handler.pop_value()?;
    // `PyFrame` can be the nursery frame materialised by an inlined
    // CALL_ASSEMBLER.  PyPy's GC transform keeps the `self` livevar rooted
    // across `space.iter()` and reloads it before `pushvalue`; the Rust borrow
    // alone still names the evacuated copy after `iter_value` collects.
    let anchor = handler.anchor();
    let iterator = handler.iter_value(iterable)?;
    H::push_anchored(&anchor, iterator)
}

pub fn opcode_for_iter<H: IterOpcodeHandler + ControlFlowOpcodeHandler + ?Sized>(
    handler: &mut H,
    target: usize,
) -> Result<(), PyError> {
    // Iterator sits under the index-or-NULL slot GET_ITER pushed.
    let iter = handler.peek_at(1)?;
    // `PyFrame` can be the nursery frame materialised by an inlined
    // CALL_ASSEMBLER.  After `iter_next` runs Python a minor collection may
    // relocate it, so later writes go through the live frame rather than the
    // abandoned copy.
    let anchor = handler.anchor();
    match handler.iter_next(iter)? {
        Some(next) => {
            let fallthrough = handler.fallthrough_target();
            // On guard failure this bytecode exits through the exhaustion path.
            handler.set_next_instr(target)?;
            handler.record_for_iter_guard(next, true)?;
            handler.set_next_instr(fallthrough)?;
            H::push_anchored(&anchor, next)
        }
        None => {
            handler.record_for_iter_guard_exhausted()?;
            H::on_iter_exhausted_anchored(&anchor, target)
        }
    }
}

pub fn opcode_unary_not<H: TruthOpcodeHandler + ?Sized>(handler: &mut H) -> Result<(), PyError> {
    let value = handler.pop_value()?;
    // `unaryoperation("not_")` keeps `self` live across `space.not_`.
    let anchor = handler.anchor();
    let result = handler.unary_not_value(value)?;
    H::push_anchored(&anchor, result)
}

pub fn opcode_binary_op<H: ArithmeticOpcodeHandler + ?Sized>(
    handler: &mut H,
    op: BinaryOperator,
) -> Result<(), PyError> {
    let b = handler.pop_value()?;
    let a = handler.pop_value()?;
    // `pyopcode.py binaryoperation`: `w_2 = popvalue(); w_1 = popvalue();
    // w_result = space.<op>(w_1, w_2)`. Those pops are livevars across
    // the operation, which may run `__xor__` / `__eq__` and collect.
    // The concrete handler pins both; anchor `self` for the result push.
    let anchor = handler.anchor();
    let result = handler.binary_value(a, b, op)?;
    H::push_anchored(&anchor, result)
}

pub fn opcode_compare_op<H: ArithmeticOpcodeHandler + ?Sized>(
    handler: &mut H,
    op: ComparisonOperator,
) -> Result<(), PyError> {
    let b = handler.pop_value()?;
    let a = handler.pop_value()?;
    // Same translated-livevar obligation as `opcode_binary_op`:
    // `binaryoperation()`'s pops stay live across `space.eq` / `lt` / …
    let anchor = handler.anchor();
    let result = handler.compare_value(a, b, op)?;
    H::push_anchored(&anchor, result)
}

pub fn opcode_unary_negative<H: ArithmeticOpcodeHandler + ?Sized>(
    handler: &mut H,
) -> Result<(), PyError> {
    let value = handler.pop_value()?;
    let anchor = handler.anchor();
    let result = handler.unary_negative_value(value)?;
    H::push_anchored(&anchor, result)
}

pub fn opcode_unary_invert<H: ArithmeticOpcodeHandler + ?Sized>(
    handler: &mut H,
) -> Result<(), PyError> {
    let value = handler.pop_value()?;
    let anchor = handler.anchor();
    let result = handler.unary_invert_value(value)?;
    H::push_anchored(&anchor, result)
}

fn opcode_pop_jump_if<H: BranchOpcodeHandler + ?Sized>(
    handler: &mut H,
    target: usize,
    jump_if_true: bool,
) -> Result<(), PyError> {
    let value = handler.pop_value()?;
    handler.enter_branch_truth(value)?;
    let truth = handler.truth_value(value)?;
    let concrete_truth = handler.concrete_truth_as_bool(value, truth)?;
    let should_jump = concrete_truth == jump_if_true;
    let fallthrough = handler.fallthrough_target();
    if !should_jump {
        handler.set_next_instr(target)?;
    }
    // The "other target" is the branch NOT taken during tracing.
    // On guard failure, the interpreter should jump to this target.
    let other_target = if should_jump { fallthrough } else { target };
    handler.record_branch_guard(value, truth, concrete_truth, other_target)?;
    handler.leave_branch_truth()?;
    let next_target = if should_jump { target } else { fallthrough };
    handler.set_next_instr(next_target)
}

pub fn opcode_pop_jump_if_false<H: BranchOpcodeHandler + ?Sized>(
    handler: &mut H,
    target: usize,
) -> Result<(), PyError> {
    opcode_pop_jump_if(handler, target, false)
}

pub fn opcode_pop_jump_if_true<H: BranchOpcodeHandler + ?Sized>(
    handler: &mut H,
    target: usize,
) -> Result<(), PyError> {
    opcode_pop_jump_if(handler, target, true)
}

pub fn opcode_jump_forward<H: ControlFlowOpcodeHandler + ?Sized>(
    handler: &mut H,
    target: usize,
) -> Result<(), PyError> {
    handler.set_next_instr(target)
}

pub fn opcode_jump_backward<H: ControlFlowOpcodeHandler + ?Sized>(
    handler: &mut H,
    target: usize,
) -> Result<StepResult<H::Value>, PyError> {
    handler.set_next_instr(target)?;
    handler.close_loop(target)
}

pub fn opcode_return_value<H: ControlFlowOpcodeHandler + ?Sized>(
    handler: &mut H,
) -> Result<StepResult<H::Value>, PyError> {
    let value = handler.pop_value()?;
    handler.finish_value(value)
}

pub trait OpcodeStepExecutor: SharedOpcodeHandler {
    fn load_const(&mut self, constant: &ConstantData) -> Result<(), PyError>
    where
        Self: ConstantOpcodeHandler,
    {
        opcode_load_const(self, constant)
    }

    /// No default: the trait-default `small_int_constant` → `int_constant`
    /// → `w_int_new` walk writes a translator-local `&INT_TYPE`. `PyFrame`
    /// returns the interned `w_small_int_const` box.
    fn load_small_int(&mut self, value: i64) -> Result<(), PyError>;

    fn load_fast_checked(&mut self, idx: usize) -> Result<(), PyError>
    where
        Self: LocalOpcodeHandler,
    {
        opcode_load_fast_checked(self, idx)
    }

    fn load_fast_pair_checked(&mut self, idx1: usize, idx2: usize) -> Result<(), PyError>
    where
        Self: LocalOpcodeHandler,
    {
        opcode_load_fast_pair_checked(self, idx1, idx2)
    }

    fn store_fast(&mut self, idx: usize) -> Result<(), PyError>
    where
        Self: LocalOpcodeHandler,
    {
        opcode_store_fast(self, idx)
    }

    fn load_fast_load_fast(&mut self, idx1: usize, idx2: usize) -> Result<(), PyError>
    where
        Self: LocalOpcodeHandler,
    {
        opcode_load_fast_load_fast(self, idx1, idx2)
    }

    fn store_fast_load_fast(&mut self, store_idx: usize, load_idx: usize) -> Result<(), PyError>
    where
        Self: LocalOpcodeHandler,
    {
        opcode_store_fast_load_fast(self, store_idx, load_idx)
    }

    fn store_fast_store_fast(&mut self, idx1: usize, idx2: usize) -> Result<(), PyError>
    where
        Self: LocalOpcodeHandler,
    {
        opcode_store_fast_store_fast(self, idx1, idx2)
    }

    fn store_name(&mut self, name: &str, nameindex: usize) -> Result<(), PyError>
    where
        Self: NamespaceOpcodeHandler,
    {
        opcode_store_name(self, name, nameindex)
    }

    fn store_global(&mut self, name: &str, nameindex: usize) -> Result<(), PyError>
    where
        Self: NamespaceOpcodeHandler,
    {
        opcode_store_global(self, name, nameindex)
    }

    fn load_name(&mut self, name: &str, nameindex: usize) -> Result<(), PyError>
    where
        Self: NamespaceOpcodeHandler,
    {
        opcode_load_name(self, name, nameindex)
    }

    fn load_global(&mut self, name: &str, nameindex: usize, push_null: bool) -> Result<(), PyError>
    where
        Self: NamespaceOpcodeHandler,
    {
        opcode_load_global(self, name, nameindex, push_null)
    }

    fn pop_top(&mut self) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        opcode_pop_top(self)
    }

    fn push_null(&mut self) -> Result<(), PyError>
    where
        Self: NamespaceOpcodeHandler,
    {
        opcode_push_null(self)
    }

    fn copy_value(&mut self, depth: usize) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        opcode_copy_value(self, depth)
    }

    fn swap(&mut self, depth: usize) -> Result<(), PyError>
    where
        Self: StackOpcodeHandler,
    {
        opcode_swap(self, depth)
    }

    fn binary_op(&mut self, op: BinaryOperator) -> Result<(), PyError>
    where
        Self: ArithmeticOpcodeHandler,
    {
        opcode_binary_op(self, op)
    }

    fn compare_op(&mut self, op: ComparisonOperator) -> Result<(), PyError>
    where
        Self: ArithmeticOpcodeHandler,
    {
        opcode_compare_op(self, op)
    }

    fn unary_negative(&mut self) -> Result<(), PyError>
    where
        Self: ArithmeticOpcodeHandler,
    {
        opcode_unary_negative(self)
    }

    fn unary_not(&mut self) -> Result<(), PyError>
    where
        Self: TruthOpcodeHandler,
    {
        opcode_unary_not(self)
    }

    fn unary_invert(&mut self) -> Result<(), PyError>
    where
        Self: ArithmeticOpcodeHandler,
    {
        opcode_unary_invert(self)
    }

    fn jump_forward(&mut self, target: usize) -> Result<(), PyError>
    where
        Self: ControlFlowOpcodeHandler,
    {
        opcode_jump_forward(self, target)
    }

    fn jump_backward(
        &mut self,
        target: usize,
    ) -> Result<StepResult<<Self as SharedOpcodeHandler>::Value>, PyError>
    where
        Self: ControlFlowOpcodeHandler,
    {
        opcode_jump_backward(self, target)
    }

    fn pop_jump_if_false(&mut self, target: usize) -> Result<(), PyError>
    where
        Self: BranchOpcodeHandler,
    {
        opcode_pop_jump_if_false(self, target)
    }

    fn pop_jump_if_true(&mut self, target: usize) -> Result<(), PyError>
    where
        Self: BranchOpcodeHandler,
    {
        opcode_pop_jump_if_true(self, target)
    }

    fn make_function(&mut self) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        opcode_make_function(self)
    }

    /// `SETUP_ANNOTATIONS` — ensure the current locals namespace exposes
    /// `__annotations__` as a dict for subsequent `STORE_SUBSCR` writes.
    /// Default no-op so non-PyFrame handlers (e.g. trace recorder) can
    /// ignore the opcode; PyFrame overrides this to do the actual work.
    fn setup_annotations(&mut self) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        Ok(())
    }

    /// CPython 3.14 WITH_EXCEPT_START — call __exit__ with the active
    /// exception inside a `with` block. Default no-op for the trace
    /// recorder; PyFrame overrides to actually invoke __exit__.
    fn with_except_start(&mut self) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        Ok(())
    }

    fn call(&mut self, nargs: usize) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        opcode_call(self, nargs)
    }

    fn return_value(&mut self) -> Result<StepResult<<Self as SharedOpcodeHandler>::Value>, PyError>
    where
        Self: ControlFlowOpcodeHandler,
    {
        opcode_return_value(self)
    }

    fn build_list(&mut self, size: usize) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        opcode_build_list(self, size)
    }

    fn build_list_from_arg(&mut self) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        opcode_build_list_from_arg(self)
    }

    fn build_tuple(&mut self, size: usize) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        opcode_build_tuple(self, size)
    }

    fn build_map(&mut self, size: usize) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        opcode_build_map(self, size)
    }

    fn store_subscr(&mut self) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        opcode_store_subscr(self)
    }

    fn list_append(&mut self, depth: usize) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        opcode_list_append(self, depth)
    }

    fn unpack_sequence(&mut self, count: usize) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        opcode_unpack_sequence(self, count)
    }

    fn load_attr(&mut self, name: &str) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        opcode_load_attr(self, name)
    }

    /// LOAD_ATTR non-method branch threaded with the bytecode `nameindex` for
    /// the interpreter mapdict attribute cache (pyopcode.py). The
    /// default ignores `nameindex` and runs the uncached path, so MIFrame and
    /// the JIT tracer keep identical behavior; PyFrame overrides it to consult
    /// `pycode._mapdict_caches[nameindex]` (only under `not we_are_jitted()`).
    fn load_attr_cached(&mut self, name: &str, _nameindex: usize) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        OpcodeStepExecutor::load_attr(self, name)
    }

    /// LOAD_ATTR with is_method=true. Default: push [attr, NULL].
    ///
    /// PyFrame overrides this to push [attr, self] for instance method
    /// calls. The JIT tracer uses the default (no runtime branch), so
    /// trace and concrete execution always agree in the shared path.
    fn load_method(&mut self, name: &str) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler + NamespaceOpcodeHandler,
    {
        let obj = self.pop_value()?;
        let attr_val = SharedOpcodeHandler::load_attr(self, obj, name)?;
        self.push_value(attr_val)?;
        let null = self.null_value()?;
        self.push_value(null)
    }

    /// LOAD_SPECIAL used by synchronous/asynchronous context managers.
    /// Like PyPy's BEFORE_WITH, this performs a type-MRO lookup and descriptor
    /// binding without consulting the instance's `__getattribute__`.
    fn load_special(&mut self, name: &str) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler + NamespaceOpcodeHandler,
    {
        let obj = self.pop_value()?;
        let attr = SharedOpcodeHandler::load_special_attr(self, obj, name)?;
        self.push_value(attr)?;
        let null = self.null_value()?;
        self.push_value(null)
    }

    fn store_attr(&mut self, name: &str) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        opcode_store_attr(self, name)
    }

    /// STORE_ATTR threaded with the bytecode `nameindex` for the interpreter
    /// mapdict attribute cache (pyopcode.py). The default ignores
    /// `nameindex` and runs the uncached path; PyFrame overrides it to consult
    /// `pycode._mapdict_caches[nameindex]` (only under `not we_are_jitted()`).
    fn store_attr_cached(&mut self, name: &str, _nameindex: usize) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        OpcodeStepExecutor::store_attr(self, name)
    }

    fn get_iter(&mut self, mode: u32) -> Result<(), PyError>
    where
        Self: IterOpcodeHandler + NamespaceOpcodeHandler,
    {
        if mode != 0 {
            self.get_yield_from_iter()?;
        } else {
            opcode_get_iter(self)?;
        }
        // GET_ITER then pushes the iterator's index-or-NULL slot.
        self.push_null()
    }

    fn for_iter(&mut self, target: usize) -> Result<(), PyError>
    where
        Self: IterOpcodeHandler + ControlFlowOpcodeHandler,
    {
        opcode_for_iter(self, target)
    }

    fn end_for(&mut self) -> Result<(), PyError> {
        Ok(())
    }

    fn pop_iter(&mut self) -> Result<(), PyError>
    where
        Self: SharedOpcodeHandler,
    {
        opcode_pop_top(self)?;
        opcode_pop_top(self)
    }

    // ── Closures / cells ──
    fn load_deref(&mut self, _idx: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error("load_deref not implemented"))
    }
    fn store_deref(&mut self, _idx: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error("store_deref not implemented"))
    }
    fn load_closure(&mut self, _idx: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error("load_closure not implemented"))
    }
    fn delete_deref(&mut self, _idx: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error("delete_deref not implemented"))
    }

    // ── Exception handling ──
    fn raise_varargs(&mut self, _argc: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error("raise_varargs not implemented"))
    }
    fn end_finally(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error("end_finally not implemented"))
    }
    fn exception_handler(&mut self) -> Result<(), PyError> {
        Ok(()) // no-op by default
    }

    // ── Import ──
    fn import_name(&mut self, _name: &str, _nameindex: usize, _flags: u32) -> Result<(), PyError> {
        Err(crate::PyError::type_error("import_name not implemented"))
    }
    fn import_from(&mut self, _name: &str) -> Result<(), PyError> {
        Err(crate::PyError::type_error("import_from not implemented"))
    }
    fn import_star(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error("import_star not implemented"))
    }

    // ── Stack manipulation ──
    fn rotate3(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error("rotate3 not implemented"))
    }

    // ── Delete operations ──
    fn delete_fast(&mut self, _idx: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error("delete_fast not implemented"))
    }
    fn delete_subscript(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "delete_subscript not implemented",
        ))
    }
    fn delete_attr(&mut self, _name: &str) -> Result<(), PyError> {
        Err(crate::PyError::type_error("delete_attr not implemented"))
    }
    fn delete_name(&mut self, _name: &str, _nameindex: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error("delete_name not implemented"))
    }
    fn delete_global(&mut self, _name: &str) -> Result<(), PyError> {
        Err(crate::PyError::type_error("delete_global not implemented"))
    }

    // Containment / identity
    fn contains_op(&mut self, _invert: crate::bytecode::Invert) -> Result<(), PyError> {
        Err(crate::PyError::type_error("contains_op not implemented"))
    }
    fn is_op(&mut self, _invert: crate::bytecode::Invert) -> Result<(), PyError> {
        Err(crate::PyError::type_error("is_op not implemented"))
    }

    // Exception handling
    fn push_exc_info(&mut self) -> Result<(), PyError> {
        Ok(())
    }
    fn pop_except(&mut self) -> Result<(), PyError> {
        Ok(())
    }
    fn check_exc_match(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "check_exc_match not implemented",
        ))
    }
    fn check_eg_match(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error("check_eg_match not implemented"))
    }
    /// `pypy/interpreter/pyopcode.py RERAISE`.
    ///
    /// `oparg` is the depth (0..) at which the original raise-site lasti
    /// integer sits on the value stack.  When `oparg > 0` the handler
    /// peeks lasti at `peekvalue(oparg)` and carries it through
    /// `RaiseWithExplicitTraceback(operr, reraise_lasti=...)`.  When
    /// `oparg == 0` no lasti is attached (default `-1`).
    fn reraise(&mut self, _oparg: u32) -> Result<(), PyError> {
        Err(crate::PyError::type_error("reraise not implemented"))
    }

    // Collections
    fn build_set(&mut self, _count: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error("build_set not implemented"))
    }
    fn build_slice(&mut self, _argc: crate::bytecode::BuildSliceArgCount) -> Result<(), PyError> {
        Err(crate::PyError::type_error("build_slice not implemented"))
    }
    fn build_string(&mut self, _count: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error("build_string not implemented"))
    }
    fn list_extend(&mut self, _i: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error("list_extend not implemented"))
    }
    fn set_add(&mut self, _i: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error("set_add not implemented"))
    }
    fn dict_merge(&mut self, _i: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error("dict_merge not implemented"))
    }
    fn dict_update(&mut self, _i: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error("dict_update not implemented"))
    }
    fn set_update(&mut self, _i: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error("set_update not implemented"))
    }
    fn map_add(&mut self, _i: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error("map_add not implemented"))
    }

    // Slicing
    fn binary_slice(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error("binary_slice not implemented"))
    }
    fn store_slice(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error("store_slice not implemented"))
    }

    // Common constants (CPython LOAD_COMMON_CONSTANT)
    fn load_common_constant(
        &mut self,
        _cc: crate::bytecode::CommonConstant,
    ) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "load_common_constant not implemented",
        ))
    }

    // Boolean
    fn to_bool(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error("to_bool not implemented"))
    }

    // None jumps
    fn pop_jump_if_none(&mut self, _target: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "pop_jump_if_none not implemented",
        ))
    }
    fn pop_jump_if_not_none(&mut self, _target: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "pop_jump_if_not_none not implemented",
        ))
    }

    // Closures 3.11+
    /// MAKE_CELL i — create a cell object in slot i.
    /// PyPy: pyframe.py MAKE_CELL
    fn make_cell(&mut self, _idx: usize) -> Result<(), PyError> {
        Ok(()) // default no-op for JIT tracer
    }
    fn copy_free_vars(&mut self, _count: usize) -> Result<(), PyError> {
        Ok(())
    }
    fn return_generator(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "return_generator not implemented",
        ))
    }

    // Call variants
    fn call_kw(&mut self, _argc: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error("call_kw not implemented"))
    }
    fn call_function_ex(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "call_function_ex not implemented",
        ))
    }

    // yield from / send
    fn get_yield_from_iter(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error("yield from not implemented"))
    }
    fn send_value(&mut self, _target: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error("send not implemented"))
    }
    fn end_send(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error("end_send not implemented"))
    }
    fn cleanup_throw(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error("cleanup throw not implemented"))
    }
    /// GET_AWAITABLE — replace TOS with its awaitable iterator (`__await__`).
    /// Overridden by the interpreter; the trace path declines (await suspends
    /// the frame and is never JIT-traced).
    fn get_awaitable(&mut self, _context: u32) -> Result<(), PyError> {
        Err(crate::PyError::type_error("get_awaitable not implemented"))
    }
    fn get_aiter(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error("get_aiter not implemented"))
    }
    fn get_anext(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error("get_anext not implemented"))
    }
    fn end_async_for(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error("end_async_for not implemented"))
    }

    // Class
    fn load_build_class(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "load_build_class not implemented",
        ))
    }
    /// The value half of [`Self::load_build_class`], without the push.
    /// The JIT's `LOAD_BUILD_CLASS` residual calls this so the lookup order
    /// stays in one place, the way `load_name_checked_value` serves
    /// `load_name`.
    fn load_build_class_value(&mut self) -> Result<<Self as SharedOpcodeHandler>::Value, PyError> {
        Err(crate::PyError::type_error(
            "load_build_class not implemented",
        ))
    }
    fn load_super_attr(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "load_super_attr not implemented",
        ))
    }
    fn load_super_attr_with(
        &mut self,
        _name_idx: usize,
        _is_method: bool,
        _is_two_arg: bool,
    ) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "load_super_attr not implemented",
        ))
    }
    fn load_locals(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error("load_locals not implemented"))
    }

    // String formatting
    fn format_simple(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error("format_simple not implemented"))
    }
    fn format_with_spec(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "format_with_spec not implemented",
        ))
    }
    fn convert_value(&mut self, _conv: crate::bytecode::ConvertValueOparg) -> Result<(), PyError> {
        Err(crate::PyError::type_error("convert_value not implemented"))
    }

    fn get_len(
        &mut self,
        _obj: <Self as SharedOpcodeHandler>::Value,
    ) -> Result<<Self as SharedOpcodeHandler>::Value, PyError> {
        Err(crate::PyError::type_error("get_len not implemented"))
    }
    fn load_fast_and_clear(&mut self, _idx: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "load_fast_and_clear not implemented",
        ))
    }
    fn set_function_attribute_with_flag(
        &mut self,
        _flag: crate::bytecode::MakeFunctionFlag,
    ) -> Result<(), PyError> {
        // Default: pop the attribute value and discard
        let _attr = self.pop_value()?;
        Ok(())
    }
    fn load_from_dict_or_globals(&mut self, _name: &str, _nameindex: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "load_from_dict_or_globals not implemented",
        ))
    }
    fn load_from_dict_or_deref(&mut self, _idx: usize, _name: &str) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "load_from_dict_or_deref not implemented",
        ))
    }
    fn match_stub(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "pattern matching not implemented",
        ))
    }
    // MATCH_MAPPING / MATCH_SEQUENCE / MATCH_KEYS / MATCH_CLASS (PEP 634).
    // The JIT tracer inherits these erroring defaults and declines a trace
    // that reaches a match statement; the interpreter overrides them.
    fn match_mapping(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "pattern matching not implemented",
        ))
    }
    fn match_sequence(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "pattern matching not implemented",
        ))
    }
    fn match_keys(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "pattern matching not implemented",
        ))
    }
    fn match_class(&mut self, _count: usize) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "pattern matching not implemented",
        ))
    }
    fn unpack_ex(&mut self, _args: crate::bytecode::UnpackExArgs) -> Result<(), PyError> {
        Err(crate::PyError::type_error("unpack_ex not implemented"))
    }

    /// CALL_INTRINSIC_1: single-argument intrinsic operations.
    /// Import a module by name, returning its module object.  Overridden by
    /// the interpreter; the trace path declines (typing intrinsics run at
    /// import time, never inside a JIT-traced loop).
    fn import_module(&mut self, name: &str) -> Result<Self::Value, PyError> {
        let _ = name;
        Err(crate::PyError::type_error("import_module not implemented"))
    }

    /// Build a PEP 695 type-parameter object by routing the single intrinsic
    /// operand through the named `_typing` helper.
    fn typing_intrinsic_1(&mut self, helper: &str) -> Result<(), PyError> {
        let arg = self.pop_value()?;
        let module = self.import_module("_typing")?;
        let func = SharedOpcodeHandler::load_attr(self, module, helper)?;
        let result = self.call_callable(func, &[arg])?;
        self.push_value(result)?;
        Ok(())
    }

    /// Two-operand variant: TOS is the second argument, TOS1 the first.
    fn typing_intrinsic_2(&mut self, helper: &str) -> Result<(), PyError> {
        let arg2 = self.pop_value()?;
        let arg1 = self.pop_value()?;
        let module = self.import_module("_typing")?;
        let func = SharedOpcodeHandler::load_attr(self, module, helper)?;
        let result = self.call_callable(func, &[arg1, arg2])?;
        self.push_value(result)?;
        Ok(())
    }

    fn prep_reraise_star(
        &mut self,
        _orig: Self::Value,
        _exceptions: Self::Value,
    ) -> Result<Self::Value, PyError> {
        Err(crate::PyError::type_error(
            "prep_reraise_star not implemented",
        ))
    }

    /// BUILD_TEMPLATE: pop the interpolations and strings tuples and build a
    /// `string.templatelib.Template`.  Overridden by the interpreter; the trace
    /// path declines (t-strings run at import time, never inside a JIT-traced
    /// loop).
    fn build_template_op(&mut self) -> Result<(), PyError> {
        Err(crate::PyError::type_error("BUILD_TEMPLATE not implemented"))
    }

    /// BUILD_INTERPOLATION: build a `string.templatelib.Interpolation` from the
    /// value/expression (and optional format spec) on the stack, with the
    /// conversion taken from the opcode oparg.  Overridden by the interpreter.
    fn build_interpolation_op(
        &mut self,
        _conversion: u32,
        _has_format_spec: bool,
    ) -> Result<(), PyError> {
        Err(crate::PyError::type_error(
            "BUILD_INTERPOLATION not implemented",
        ))
    }

    fn call_intrinsic_1(&mut self, func: IntrinsicFunction1) -> Result<(), PyError> {
        match func {
            IntrinsicFunction1::UnaryPositive => {
                // PyPy: UNARY_POSITIVE → space.pos(w_value)
                let val = self.pop_value()?;
                let result = self.unary_positive(val)?;
                self.push_value(result)?;
                Ok(())
            }
            IntrinsicFunction1::ListToTuple => {
                let val = self.pop_value()?;
                let result = self.list_to_tuple(val)?;
                self.push_value(result)?;
                Ok(())
            }
            IntrinsicFunction1::ImportStar => {
                // Module is TOS; import_star pops it internally
                self.import_star()?;
                let none = self.none_value()?;
                self.push_value(none)?;
                Ok(())
            }
            IntrinsicFunction1::Print => {
                // sys.displayhook(value)
                let val = self.pop_value()?;
                self.print_expr(val)?;
                let none = self.none_value()?;
                self.push_value(none)?;
                Ok(())
            }
            IntrinsicFunction1::StopIterationError => {
                // CPython: convert StopIteration to RuntimeError in generator context.
                // For now, just leave the value on the stack unchanged.
                Ok(())
            }
            IntrinsicFunction1::AsyncGenWrap => {
                // PyPy `YIELD_VALUE`: an async-generator yield is wrapped so
                // AsyncGenASend can distinguish it from an await suspension.
                let value = self.pop_value()?;
                let wrapped = self.async_gen_wrap(value)?;
                self.push_value(wrapped)?;
                Ok(())
            }
            // PEP 695 type-parameter construction (`class C[T]:`, `def f[T]()`).
            IntrinsicFunction1::TypeVar => self.typing_intrinsic_1("_intrinsic_typevar"),
            IntrinsicFunction1::ParamSpec => self.typing_intrinsic_1("_intrinsic_paramspec"),
            IntrinsicFunction1::TypeVarTuple => self.typing_intrinsic_1("_intrinsic_typevartuple"),
            IntrinsicFunction1::SubscriptGeneric => {
                self.typing_intrinsic_1("_intrinsic_subscript_generic")
            }
            IntrinsicFunction1::TypeAlias => self.typing_intrinsic_1("_intrinsic_typealias"),
            _ => Err(crate::PyError::type_error(format!(
                "intrinsic function {:?} not implemented",
                func
            ))),
        }
    }

    /// CALL_INTRINSIC_2: two-argument intrinsic operations.
    fn call_intrinsic_2(&mut self, func: IntrinsicFunction2) -> Result<(), PyError> {
        match func {
            IntrinsicFunction2::PrepReraiseStar => {
                let exceptions = self.pop_value()?;
                let orig = self.pop_value()?;
                let result = self.prep_reraise_star(orig, exceptions)?;
                self.push_value(result)?;
                Ok(())
            }
            IntrinsicFunction2::SetFunctionTypeParams => {
                // arg2 = type_params, arg1 = function
                // Set __type_params__ attribute on the function; push function back
                let type_params = self.pop_value()?;
                let function = self.peek_at(0)?;
                self.set_function_typeparams(function, type_params)?;
                // just leave the function on the stack
                Ok(())
            }
            // PEP 695 two-operand type-parameter intrinsics.
            IntrinsicFunction2::TypeVarWithBound => {
                self.typing_intrinsic_2("_intrinsic_typevar_with_bound")
            }
            IntrinsicFunction2::TypeVarWithConstraint => {
                self.typing_intrinsic_2("_intrinsic_typevar_with_constraints")
            }
            IntrinsicFunction2::SetTypeparamDefault => {
                self.typing_intrinsic_2("_intrinsic_set_typeparam_default")
            }
            _ => Err(crate::PyError::type_error(format!(
                "intrinsic function {:?} not implemented",
                func
            ))),
        }
    }

    // ── Intrinsic helper methods ──
    /// Declared without a body: a default would be the graph the tracer walks,
    /// and the implementing type's version would never be seen.  `+x` compiles
    /// to `CALL_INTRINSIC_1`, so this method is the whole route to `pos`.
    fn unary_positive(
        &mut self,
        val: <Self as SharedOpcodeHandler>::Value,
    ) -> Result<<Self as SharedOpcodeHandler>::Value, PyError>;
    fn list_to_tuple(
        &mut self,
        _val: <Self as SharedOpcodeHandler>::Value,
    ) -> Result<<Self as SharedOpcodeHandler>::Value, PyError> {
        Err(crate::PyError::type_error("list_to_tuple not implemented"))
    }
    fn async_gen_wrap(
        &mut self,
        _val: <Self as SharedOpcodeHandler>::Value,
    ) -> Result<<Self as SharedOpcodeHandler>::Value, PyError> {
        Err(crate::PyError::type_error("async_gen_wrap not implemented"))
    }
    fn print_expr(&mut self, _val: <Self as SharedOpcodeHandler>::Value) -> Result<(), PyError> {
        Err(crate::PyError::type_error("print_expr not implemented"))
    }
    fn none_value(&mut self) -> Result<<Self as SharedOpcodeHandler>::Value, PyError> {
        Err(crate::PyError::type_error("none_value not implemented"))
    }

    fn unsupported(
        &mut self,
        instruction: &Instruction,
    ) -> Result<StepResult<<Self as SharedOpcodeHandler>::Value>, PyError>;
}

/// Widen a `u32`-typed oparg to a signed `i64`.
///
/// The cast is the unsigned-to-signed word retype (`rarithmetic.intmask`).
/// `i64::from` kept the source unsigned annotation, which then failed to
/// merge with a signed argument of the same helper.
#[inline]
fn u32_as_i64(x: u32) -> i64 {
    x as i64
}

/// Widen a `u32`-typed oparg to host-pointer-sized `usize`. See
/// [`u32_as_i64`] for the parity rationale. The body uses
/// `usize::try_from(x).expect(...)` because Rust stdlib does not
/// provide a `From<u32> for usize` impl (u32 → usize is not
/// universally lossless: 16-bit hosts have `usize` smaller than u32).
/// On supported pyre targets (64-bit Darwin) the conversion is
/// always lossless, so the runtime `expect(...)` never trips. Drop
/// `const fn` because neither `usize::try_from` nor `Result::expect` is
/// yet stable as const.
#[inline]
fn u32_as_usize(x: u32) -> usize {
    usize::try_from(x).expect("u32 fits in usize on supported pyre targets (64-bit only)")
}

/// Widen a raw [`OpArg`] to host-pointer-sized `usize`. Used at
/// instruction sites that consume the oparg directly (no inner
/// `Arg<T>::get` call) — currently `JumpForward` /
/// `jump_target_forward` indirections in this module.
/// Composes `u32::from(arg)` (trait-method call lowered as
/// `simple_call(getattr(<u32>, "from"), arg)`) with
/// `usize::try_from(...).expect(…)` to keep the body adapter-friendly
/// — see [`u32_as_usize`] for the u32 → usize parity rationale.
#[inline]
fn op_arg_as_usize(arg: OpArg) -> usize {
    usize::try_from(u32::from(arg))
        .expect("u32 fits in usize on supported pyre targets (64-bit only)")
}

/// Walker-fold helper for `var_num.get(op_arg).as_usize()` — chains
/// the third-party `Arg::get` + `VarNum::as_usize` calls under a single
/// `#[elidable_cannot_raise]` first-party wrapper.  Without it, the
/// codewriter sees two unfolded `residual_call` ops with
/// `oopspec=None CanRaise`, and the walker's bounds-check `goto_if_not`
/// downstream aborts with `GotoIfNotValueNotConcrete`.  The wrapper's
/// body still emits a residual_call to the third-party helpers, but
/// the OUTER call site is tagged elidable so the walker's
/// `try_fold_pure_call_via_executor` fold path runs end-to-end.
///
/// Every handler that resolves an `Arg<VarNum>` to a localsplus index
/// goes through here, not only the `LOAD_FAST` family it is named for:
/// `STORE_FAST`, `DELETE_FAST`, `LOAD_FAST_AND_CLEAR`, `MAKE_CELL` and
/// the `*_DEREF` handlers all carry the same field type, and each raw
/// `var_num.get(op_arg).as_usize()` left in a handler body is one
/// unbindable third-party residual on that opcode's walked path.
#[inline]
#[majit_macros::elidable_cannot_raise]
pub fn load_fast_var_num_to_index(
    var_num: crate::bytecode::Arg<crate::bytecode::oparg::VarNum>,
    op_arg: OpArg,
) -> usize {
    var_num.get(op_arg).as_usize()
}

/// Walker-fold helpers for the paired-local opcode family
/// (`LoadFastLoadFast` / `StoreFastLoadFast` / `StoreFastStoreFast` /
/// `LoadFastBorrowLoadFastBorrow`) — same rationale as
/// [`load_fast_var_num_to_index`]: collapse the third-party
/// `Arg::get` + `VarNumPair::idx_*` + `VarNum::as_usize` chain under
/// a first-party `#[elidable_cannot_raise]` wrapper so the walker's
/// pure-call fold resolves the local index at trace time.
#[inline]
#[majit_macros::elidable_cannot_raise]
pub fn var_nums_to_first_index(
    var_nums: crate::bytecode::Arg<crate::bytecode::oparg::VarNums>,
    op_arg: OpArg,
) -> usize {
    var_nums.get(op_arg).idx_1().as_usize()
}

/// Second half of the paired-local index decode — see
/// [`var_nums_to_first_index`].
#[inline]
#[majit_macros::elidable_cannot_raise]
pub fn var_nums_to_second_index(
    var_nums: crate::bytecode::Arg<crate::bytecode::oparg::VarNums>,
    op_arg: OpArg,
) -> usize {
    var_nums.get(op_arg).idx_2().as_usize()
}

/// Rtyper residual helper for label oparg decode sites. Keep the
/// third-party `Arg::get` call inside a first-party scalar wrapper so
/// opcode handlers do not need to register the external generic method.
#[inline]
#[majit_macros::dont_look_inside]
pub fn label_arg_to_usize(
    delta: crate::bytecode::Arg<crate::bytecode::oparg::Label>,
    op_arg: OpArg,
) -> usize {
    keep_fnaddr_distinct(2);
    delta.get(op_arg).as_usize()
}

/// Decode a forward jump target as one residual scalar call. The
/// `CodeUnits::deref` slice stays inside the helper body and never
/// crosses the two-phase residual ABI.
#[inline]
#[majit_macros::dont_look_inside]
pub fn jump_target_forward_decoded(
    code: &CodeObject,
    next_instr: usize,
    delta: crate::bytecode::Arg<crate::bytecode::oparg::Label>,
    op_arg: OpArg,
) -> usize {
    jump_target_forward(&code.instructions, next_instr, delta.get(op_arg).as_usize())
}

/// Forward jump helper for opcodes whose target is already the raw
/// instruction oparg, not an `Arg<Label>` field.
#[inline]
#[majit_macros::dont_look_inside]
pub fn jump_target_forward_from_oparg(
    code: &CodeObject,
    next_instr: usize,
    op_arg: OpArg,
) -> usize {
    keep_fnaddr_distinct(3);
    jump_target_forward(&code.instructions, next_instr, op_arg_as_usize(op_arg))
}

/// Backward jump counterpart of [`jump_target_forward_decoded`].
#[inline]
#[majit_macros::dont_look_inside]
pub fn jump_target_backward_decoded(
    code: &CodeObject,
    next_instr: usize,
    delta: crate::bytecode::Arg<crate::bytecode::oparg::Label>,
    op_arg: OpArg,
) -> usize {
    jump_target_backward(&code.instructions, next_instr, delta.get(op_arg).as_usize())
}

/// Decode `BINARY_OP`'s enum oparg behind a first-party residual helper.
#[inline]
#[majit_macros::dont_look_inside]
pub fn binary_op_arg(
    op: crate::bytecode::Arg<crate::bytecode::oparg::BinaryOperator>,
    op_arg: OpArg,
) -> BinaryOperator {
    op.get(op_arg)
}

/// Decode `COMPARE_OP`'s enum oparg behind a first-party residual helper.
#[inline]
#[majit_macros::dont_look_inside]
pub fn comparison_op_arg(
    opname: crate::bytecode::Arg<crate::bytecode::oparg::ComparisonOperator>,
    op_arg: OpArg,
) -> ComparisonOperator {
    opname.get(op_arg)
}

/// Decode containment/identity inversion flags behind a first-party helper.
#[inline]
#[majit_macros::dont_look_inside]
pub fn invert_arg(
    invert: crate::bytecode::Arg<crate::bytecode::oparg::Invert>,
    op_arg: OpArg,
) -> Invert {
    invert.get(op_arg)
}

/// Decode `BUILD_SLICE`'s argument-count enum behind a first-party helper.
#[inline]
#[majit_macros::dont_look_inside]
pub fn build_slice_arg(
    argc: crate::bytecode::Arg<crate::bytecode::oparg::BuildSliceArgCount>,
    op_arg: OpArg,
) -> BuildSliceArgCount {
    argc.get(op_arg)
}

/// Decode `LOAD_COMMON_CONSTANT`'s enum oparg behind a first-party helper.
#[inline]
#[majit_macros::dont_look_inside]
pub fn common_constant_arg(
    idx: crate::bytecode::Arg<crate::bytecode::oparg::CommonConstant>,
    op_arg: OpArg,
) -> CommonConstant {
    idx.get(op_arg)
}

/// Decode `CONVERT_VALUE`'s enum oparg behind a first-party helper.
#[inline]
#[majit_macros::dont_look_inside]
pub fn convert_value_arg(
    conv: crate::bytecode::Arg<crate::bytecode::oparg::ConvertValueOparg>,
    op_arg: OpArg,
) -> ConvertValueOparg {
    conv.get(op_arg)
}

/// Materialise `tag` behind an optimisation barrier so the caller's machine
/// code carries an immediate no sibling shares.
///
/// The decode helpers below differ only in the phantom type of their `Arg<T>`
/// parameter, so several of them compile to byte-identical bodies. Each is a
/// residual-call target whose address `jit_fnaddr.rs` registers, and
/// `runtime_fnaddr_patch` re-pairs a build-time address with the runtime one
/// by that address alone — two functions folded onto a single address make
/// that pairing ambiguous and can send one callee's call to the other. A
/// linker that folds identical code (MSVC `/OPT:ICF`, on by default) is what
/// performs the fold, and which pair it picks moves with unrelated layout
/// changes, so the bodies have to differ by construction rather than by luck.
/// `drain_list_append` keeps its `#[inline(never)]` forwarding call for the
/// same reason; these have no callee to forward to, so they carry a datum.
#[inline(always)]
fn keep_fnaddr_distinct(tag: u32) {
    std::hint::black_box(tag);
}

/// Decode `LOAD_SPECIAL`'s enum oparg behind a first-party helper.
#[inline]
#[majit_macros::dont_look_inside]
pub fn special_method_arg(
    method: crate::bytecode::Arg<crate::bytecode::oparg::SpecialMethod>,
    op_arg: OpArg,
) -> SpecialMethod {
    keep_fnaddr_distinct(1);
    method.get(op_arg)
}

/// Decode `SET_FUNCTION_ATTRIBUTE`'s flag oparg behind a first-party helper.
#[inline]
#[majit_macros::dont_look_inside]
pub fn make_function_flag_arg(
    flag: crate::bytecode::Arg<crate::bytecode::oparg::MakeFunctionFlag>,
    op_arg: OpArg,
) -> MakeFunctionFlag {
    flag.get(op_arg)
}

/// Decode `CALL_INTRINSIC_1`'s enum oparg behind a first-party helper.
#[inline]
#[majit_macros::dont_look_inside]
pub fn intrinsic_function_1_arg(
    func: crate::bytecode::Arg<crate::bytecode::oparg::IntrinsicFunction1>,
    op_arg: OpArg,
) -> IntrinsicFunction1 {
    func.get(op_arg)
}

/// Decode `CALL_INTRINSIC_2`'s enum oparg behind a first-party helper.
#[inline]
#[majit_macros::dont_look_inside]
pub fn intrinsic_function_2_arg(
    func: crate::bytecode::Arg<crate::bytecode::oparg::IntrinsicFunction2>,
    op_arg: OpArg,
) -> IntrinsicFunction2 {
    func.get(op_arg)
}

/// Decode `RAISE_VARARGS` to the executor's compact usize kind.
#[inline]
#[majit_macros::dont_look_inside]
pub fn raise_kind_arg_as_usize(
    argc: crate::bytecode::Arg<crate::bytecode::oparg::RaiseKind>,
    op_arg: OpArg,
) -> usize {
    raise_kind_as_usize(argc.get(op_arg))
}

/// Walker-fold helper for `code.varnames.len()` — `Vec::len` is a std
/// method the analyzer cannot tag elidable from its definition site,
/// so the bounds-check upper bound reaches the walker as an unfolded
/// `residual_call`.  Same `#[elidable_cannot_raise]` rationale as
/// [`load_fast_var_num_to_index`].
#[inline]
#[majit_macros::elidable_cannot_raise]
pub fn code_varnames_len(code: &CodeObject) -> usize {
    code.varnames.len()
}

/// Length of the instruction array behind an `#[elidable_cannot_raise]`
/// first-party wrapper — same rationale as [`code_varnames_len`]. Lets the
/// decode funnel's bounds check lower to a plain `lt + getitem` on
/// `code.instructions[pc]` instead of a `slice::get(pc) -> Option<&CodeUnit>`
/// the walker cannot fold.
#[inline]
#[majit_macros::elidable_cannot_raise]
pub fn code_instructions_len(code: &CodeObject) -> usize {
    code.instructions.len()
}

/// Read one packed two-byte code unit through a scalar residual call. The
/// `CodeUnits::deref` slice stays inside the helper body and never crosses the
/// two-phase residual ABI.
#[inline]
#[majit_macros::dont_look_inside]
pub fn code_unit_at(code: &CodeObject, i: usize) -> u16 {
    let unit = code.instructions[i];
    u16::from(u8::from(unit.op)) | (u16::from(u8::from(unit.arg)) << 8)
}

/// Build an `OpArg` from its 32-bit value without calling into the
/// bytecode crate: `OpArg` is `#[repr(transparent)]` over `u32`, so the
/// same-layout transmute lowers to a register copy, where `OpArg::new` is a
/// call into a crate outside the extracted LLBC.
#[inline]
pub fn oparg_from_u32(value: u32) -> OpArg {
    // SAFETY: `OpArg` is `#[repr(transparent)] struct OpArg(u32)`; every
    // `u32` is a valid `OpArg`.
    unsafe { std::mem::transmute::<u32, OpArg>(value) }
}

#[inline]
fn instruction_discriminant(instruction: Instruction) -> u8 {
    // SAFETY: `Instruction` is `#[repr(u8)]`.
    unsafe { std::mem::transmute::<Instruction, u8>(instruction) }
}

#[inline]
fn instruction_from_code_unit_word(word: u16) -> Instruction {
    // SAFETY: `code_unit_at` obtains this byte from a live `Instruction` in a
    // `CodeUnit`, so its discriminant is valid and `Instruction` is repr(u8).
    unsafe { std::mem::transmute::<u8, Instruction>(word as u8) }
}

/// Extract a [`RaiseKind`]'s discriminant as `usize`. Same parity
/// rationale as [`u32_as_i64`] / [`u32_as_usize`]: upstream Python
/// has no enum-discriminant cast syntax, so the bare `kind as usize`
/// at a `RaiseVarargs` arm has no flowspace counterpart. `u32::from`
/// goes through the `From<RaiseKind> for u32` impl synthesized by
/// `rustpython-compiler-core`'s `oparg_enum!` macro
/// (`oparg.rs:215-219`); the u32 → usize widening uses
/// `usize::try_from(...).expect(…)` to match the rest of the
/// cast-removal helper family — see [`u32_as_usize`].
#[inline]
fn raise_kind_as_usize(kind: RaiseKind) -> usize {
    usize::try_from(u32::from(kind))
        .expect("u32 fits in usize on supported pyre targets (64-bit only)")
}

/// `Instruction::PopTop` handler, lifted out of `execute_opcode_step`'s
/// match so the Charon/MIR front-end emits a standalone per-opcode
/// graph the JIT dispatch can resolve by name rather than re-lowering
/// the match arm body through the syn-AST walker.  The dispatch arm is
/// the single tail-call `execute_pop_top(executor)`.
pub fn execute_pop_top<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.pop_top()?;
    Ok(StepResult::Continue)
}

// Per-opcode handlers lifted out of `execute_opcode_step`'s match (same
// seam as `execute_pop_top`): each dispatch arm becomes a single
// tail-call so the Charon/MIR front-end emits a standalone graph the JIT
// resolves by name instead of re-lowering the arm body through the
// syn-AST walker.
pub fn execute_push_null<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: NamespaceOpcodeHandler,
{
    executor.push_null()?;
    Ok(StepResult::Continue)
}

pub fn execute_unary_negative<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: ArithmeticOpcodeHandler,
{
    executor.unary_negative()?;
    Ok(StepResult::Continue)
}

pub fn execute_unary_not<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: TruthOpcodeHandler,
{
    executor.unary_not()?;
    Ok(StepResult::Continue)
}

pub fn execute_unary_invert<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: ArithmeticOpcodeHandler,
{
    executor.unary_invert()?;
    Ok(StepResult::Continue)
}

pub fn execute_get_iter<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: IterOpcodeHandler + NamespaceOpcodeHandler,
{
    let Instruction::GetIter { mode } = instruction else {
        unreachable!()
    };
    executor.get_iter(mode.get(op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_end_for<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.end_for()?;
    Ok(StepResult::Continue)
}

pub fn execute_pop_iter<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.pop_iter()?;
    Ok(StepResult::Continue)
}

pub fn execute_delete_subscr<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.delete_subscript()?;
    Ok(StepResult::Continue)
}

pub fn execute_push_exc_info<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.push_exc_info()?;
    Ok(StepResult::Continue)
}

pub fn execute_pop_except<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.pop_except()?;
    Ok(StepResult::Continue)
}

pub fn execute_check_exc_match<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.check_exc_match()?;
    Ok(StepResult::Continue)
}

pub fn execute_check_eg_match<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.check_eg_match()?;
    Ok(StepResult::Continue)
}

pub fn execute_to_bool<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.to_bool()?;
    Ok(StepResult::Continue)
}

pub fn execute_binary_slice<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.binary_slice()?;
    Ok(StepResult::Continue)
}

pub fn execute_store_slice<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.store_slice()?;
    Ok(StepResult::Continue)
}

pub fn execute_jump_forward<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
    next_instr: usize,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: ControlFlowOpcodeHandler,
{
    let Instruction::JumpForward { delta } = instruction else {
        unreachable!()
    };
    executor.jump_forward(jump_target_forward_decoded(code, next_instr, delta, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_jump_backward<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
    next_instr: usize,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: ControlFlowOpcodeHandler,
{
    let Instruction::JumpBackward { delta } = instruction else {
        unreachable!()
    };
    let step = executor.jump_backward(jump_target_backward_decoded(
        code, next_instr, delta, op_arg,
    ))?;
    // Rebuild the StepResult per variant so the Result-of-PyError
    // lowering sees a concrete `Ok(StepResult::_)` shell to rewrite. A
    // bare `Ok(step)` forward of the method's same-typed Result collapses
    // in MIR, leaving this scoped wrapper with no rewritable return.
    match step {
        StepResult::Continue => Ok(StepResult::Continue),
        StepResult::Return(value) => Ok(StepResult::Return(value)),
        StepResult::CloseLoop { loop_header_pc } => Ok(StepResult::CloseLoop { loop_header_pc }),
        StepResult::Yield(value) => Ok(StepResult::Yield(value)),
    }
}

pub fn execute_pop_jump_if_false<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
    next_instr: usize,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: BranchOpcodeHandler,
{
    let Instruction::PopJumpIfFalse { delta } = instruction else {
        unreachable!()
    };
    executor.pop_jump_if_false(jump_target_forward_decoded(code, next_instr, delta, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_pop_jump_if_true<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
    next_instr: usize,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: BranchOpcodeHandler,
{
    let Instruction::PopJumpIfTrue { delta } = instruction else {
        unreachable!()
    };
    executor.pop_jump_if_true(jump_target_forward_decoded(code, next_instr, delta, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_for_iter<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
    next_instr: usize,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: IterOpcodeHandler + ControlFlowOpcodeHandler,
{
    let Instruction::ForIter { delta } = instruction else {
        unreachable!()
    };
    executor.for_iter(jump_target_forward_decoded(code, next_instr, delta, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_pop_jump_if_none<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
    next_instr: usize,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::PopJumpIfNone { delta } = instruction else {
        unreachable!()
    };
    executor.pop_jump_if_none(jump_target_forward_decoded(code, next_instr, delta, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_pop_jump_if_not_none<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
    next_instr: usize,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::PopJumpIfNotNone { delta } = instruction else {
        unreachable!()
    };
    executor.pop_jump_if_not_none(jump_target_forward_decoded(code, next_instr, delta, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_jump_backward_no_interrupt<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
    next_instr: usize,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: ControlFlowOpcodeHandler,
{
    let Instruction::JumpBackwardNoInterrupt { delta } = instruction else {
        unreachable!()
    };
    let tgt = label_arg_to_usize(delta, op_arg);
    executor.set_next_instr(next_instr - tgt)?;
    Ok(StepResult::Continue)
}

pub fn execute_send<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    op_arg: OpArg,
    next_instr: usize,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let target = jump_target_forward_from_oparg(code, next_instr, op_arg);
    executor.send_value(target)?;
    Ok(StepResult::Continue)
}

pub fn execute_cleanup_throw<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.cleanup_throw()?;
    Ok(StepResult::Continue)
}

pub fn execute_match_stub<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.match_stub()?;
    Ok(StepResult::Continue)
}

pub fn execute_match_mapping<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.match_mapping()?;
    Ok(StepResult::Continue)
}

pub fn execute_match_sequence<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.match_sequence()?;
    Ok(StepResult::Continue)
}

pub fn execute_match_keys<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.match_keys()?;
    Ok(StepResult::Continue)
}

pub fn execute_match_class<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::MatchClass { count } = instruction else {
        unreachable!()
    };
    executor.match_class(count.get(op_arg) as usize)?;
    Ok(StepResult::Continue)
}

pub fn execute_yield_value<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let value = executor.pop_value()?;
    Ok(StepResult::Yield(value))
}

pub fn execute_get_len<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let obj = executor.peek_at(0)?;
    let len = executor.get_len(obj)?;
    executor.push_value(len)?;
    Ok(StepResult::Continue)
}

// Template strings (PEP 750) — `t"hello {name}"`. Stack: [strings, interps].
// PyPy has no equivalent; we consume the operands and push a 2-tuple
// that preserves the strings+interpolations structure. Sufficient for
// module import; real Template type semantics are not implemented.
pub fn execute_build_template<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    OpcodeStepExecutor::build_template_op(executor)?;
    Ok(StepResult::Continue)
}

pub fn execute_build_interpolation<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::BuildInterpolation { format } = instruction else {
        unreachable!()
    };
    let oparg_val = format.get(op_arg);
    let has_format_spec = (oparg_val & 1) != 0;
    // The conversion (`!s` / `!r` / `!a`) is encoded in the upper oparg bits.
    let conversion = oparg_val >> 2;
    OpcodeStepExecutor::build_interpolation_op(executor, conversion, has_format_spec)?;
    Ok(StepResult::Continue)
}

// LOAD_SPECIAL: pops obj, pushes (callable, self_or_null). Used by the
// `with` statement to load __enter__ / __exit__.
pub fn execute_load_special<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: NamespaceOpcodeHandler,
{
    let Instruction::LoadSpecial { method } = instruction else {
        unreachable!()
    };
    // Last arm is `_` (covers `SpecialMethod::AExit`) so the
    // Rust-AST adapter's variant cascade has the wildcard arm
    // it needs to close the final isinstance fork (Position-2
    // adaptation; the adapter cannot enumerate the variant
    // universe from `syn::ItemFn` alone).
    let name = match special_method_arg(method, op_arg) {
        SpecialMethod::Enter => "__enter__",
        SpecialMethod::Exit => "__exit__",
        SpecialMethod::AEnter => "__aenter__",
        _ => "__aexit__",
    };
    executor.load_special(name)?;
    Ok(StepResult::Continue)
}

pub fn execute_make_function<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    OpcodeStepExecutor::make_function(executor)?;
    Ok(StepResult::Continue)
}

pub fn execute_store_subscr<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    OpcodeStepExecutor::store_subscr(executor)?;
    Ok(StepResult::Continue)
}

pub fn execute_return_value<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: ControlFlowOpcodeHandler,
{
    let step = executor.return_value()?;
    // Rebuild the StepResult per variant so the Result-of-PyError
    // lowering sees a concrete `Ok(StepResult::_)` shell to rewrite. A
    // bare `Ok(step)` forward of the method's same-typed Result collapses
    // in MIR, leaving this scoped wrapper with no rewritable return.
    match step {
        StepResult::Continue => Ok(StepResult::Continue),
        StepResult::Return(value) => Ok(StepResult::Return(value)),
        StepResult::CloseLoop { loop_header_pc } => Ok(StepResult::CloseLoop { loop_header_pc }),
        StepResult::Yield(value) => Ok(StepResult::Yield(value)),
    }
}

pub fn execute_return_generator<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.return_generator()?;
    Ok(StepResult::Continue)
}

pub fn execute_call_function_ex<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.call_function_ex()?;
    Ok(StepResult::Continue)
}

pub fn execute_load_build_class<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.load_build_class()?;
    Ok(StepResult::Continue)
}

pub fn execute_setup_annotations<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.setup_annotations()?;
    Ok(StepResult::Continue)
}

pub fn execute_load_locals<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.load_locals()?;
    Ok(StepResult::Continue)
}

pub fn execute_format_simple<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.format_simple()?;
    Ok(StepResult::Continue)
}

pub fn execute_format_with_spec<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.format_with_spec()?;
    Ok(StepResult::Continue)
}

pub fn execute_end_send<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.end_send()?;
    Ok(StepResult::Continue)
}

pub fn execute_with_except_start<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    executor.with_except_start()?;
    Ok(StepResult::Continue)
}

pub fn execute_get_awaitable<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::GetAwaitable { r#where } = instruction else {
        unreachable!()
    };
    executor.get_awaitable(r#where.get(op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_get_aiter<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<E::Value>, PyError> {
    executor.get_aiter()?;
    Ok(StepResult::Continue)
}

pub fn execute_get_anext<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<E::Value>, PyError> {
    executor.get_anext()?;
    Ok(StepResult::Continue)
}

pub fn execute_end_async_for<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<E::Value>, PyError> {
    executor.end_async_for()?;
    Ok(StepResult::Continue)
}

// Handlers for arms that read an instruction-embedded operand: the
// dispatch forwards `instruction` (Copy) and `op_arg`, and the handler
// re-destructures its own variant (the match already proved which one).
pub fn execute_copy<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::Copy { i } = instruction else {
        unreachable!()
    };
    executor.copy_value(u32_as_usize(i.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_swap<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: StackOpcodeHandler,
{
    let Instruction::Swap { i } = instruction else {
        unreachable!()
    };
    executor.swap(u32_as_usize(i.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_store_fast<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: LocalOpcodeHandler,
{
    let Instruction::StoreFast { var_num } = instruction else {
        unreachable!()
    };
    executor.store_fast(load_fast_var_num_to_index(var_num, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_delete_fast<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::DeleteFast { var_num } = instruction else {
        unreachable!()
    };
    executor.delete_fast(load_fast_var_num_to_index(var_num, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_load_fast<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: LocalOpcodeHandler,
{
    let (Instruction::LoadFast { var_num } | Instruction::LoadFastBorrow { var_num }) = instruction
    else {
        unreachable!()
    };
    let idx = load_fast_var_num_to_index(var_num, op_arg);
    executor.load_fast_checked(idx)?;
    Ok(StepResult::Continue)
}

pub fn execute_load_fast_check<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: LocalOpcodeHandler,
{
    let Instruction::LoadFastCheck { var_num } = instruction else {
        unreachable!()
    };
    let idx = load_fast_var_num_to_index(var_num, op_arg);
    executor.load_fast_checked(idx)?;
    Ok(StepResult::Continue)
}

pub fn execute_load_fast_borrow_load_fast_borrow<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: LocalOpcodeHandler,
{
    let Instruction::LoadFastBorrowLoadFastBorrow { var_nums } = instruction else {
        unreachable!()
    };
    let idx1 = var_nums_to_first_index(var_nums, op_arg);
    let idx2 = var_nums_to_second_index(var_nums, op_arg);
    executor.load_fast_pair_checked(idx1, idx2)?;
    Ok(StepResult::Continue)
}

pub fn execute_load_fast_load_fast<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: LocalOpcodeHandler,
{
    let Instruction::LoadFastLoadFast { var_nums } = instruction else {
        unreachable!()
    };
    executor.load_fast_load_fast(
        var_nums_to_first_index(var_nums, op_arg),
        var_nums_to_second_index(var_nums, op_arg),
    )?;
    Ok(StepResult::Continue)
}

pub fn execute_store_fast_load_fast<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: LocalOpcodeHandler,
{
    let Instruction::StoreFastLoadFast { var_nums } = instruction else {
        unreachable!()
    };
    executor.store_fast_load_fast(
        var_nums_to_first_index(var_nums, op_arg),
        var_nums_to_second_index(var_nums, op_arg),
    )?;
    Ok(StepResult::Continue)
}

pub fn execute_store_fast_store_fast<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: LocalOpcodeHandler,
{
    let Instruction::StoreFastStoreFast { var_nums } = instruction else {
        unreachable!()
    };
    executor.store_fast_store_fast(
        var_nums_to_first_index(var_nums, op_arg),
        var_nums_to_second_index(var_nums, op_arg),
    )?;
    Ok(StepResult::Continue)
}

pub fn execute_load_small_int<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: ConstantOpcodeHandler,
{
    let Instruction::LoadSmallInt { i } = instruction else {
        unreachable!()
    };
    executor.load_small_int(u32_as_i64(i.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_list_append<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::ListAppend { i } = instruction else {
        unreachable!()
    };
    OpcodeStepExecutor::list_append(executor, u32_as_usize(i.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_unpack_sequence<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::UnpackSequence { count } = instruction else {
        unreachable!()
    };
    OpcodeStepExecutor::unpack_sequence(executor, u32_as_usize(count.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_build_list<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::BuildList { count } = instruction else {
        unreachable!()
    };
    OpcodeStepExecutor::build_list(executor, u32_as_usize(count.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_build_list_from_arg<E: OpcodeStepExecutor>(
    executor: &mut E,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    OpcodeStepExecutor::build_list_from_arg(executor)?;
    Ok(StepResult::Continue)
}

pub fn execute_build_tuple<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::BuildTuple { count } = instruction else {
        unreachable!()
    };
    OpcodeStepExecutor::build_tuple(executor, u32_as_usize(count.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_build_map<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::BuildMap { count } = instruction else {
        unreachable!()
    };
    OpcodeStepExecutor::build_map(executor, u32_as_usize(count.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_build_set<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::BuildSet { count } = instruction else {
        unreachable!()
    };
    executor.build_set(u32_as_usize(count.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_build_string<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::BuildString { count } = instruction else {
        unreachable!()
    };
    executor.build_string(u32_as_usize(count.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_build_slice<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::BuildSlice { argc } = instruction else {
        unreachable!()
    };
    executor.build_slice(build_slice_arg(argc, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_call<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::Call { argc } = instruction else {
        unreachable!()
    };
    executor.call(u32_as_usize(argc.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_call_kw<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::CallKw { argc } = instruction else {
        unreachable!()
    };
    executor.call_kw(u32_as_usize(argc.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_binary_op<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: ArithmeticOpcodeHandler,
{
    let Instruction::BinaryOp { op } = instruction else {
        unreachable!()
    };
    executor.binary_op(binary_op_arg(op, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_compare_op<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: ArithmeticOpcodeHandler,
{
    let Instruction::CompareOp { opname } = instruction else {
        unreachable!()
    };
    executor.compare_op(comparison_op_arg(opname, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_contains_op<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::ContainsOp { invert } = instruction else {
        unreachable!()
    };
    executor.contains_op(invert_arg(invert, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_is_op<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::IsOp { invert } = instruction else {
        unreachable!()
    };
    executor.is_op(invert_arg(invert, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_raise_varargs<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::RaiseVarargs { argc } = instruction else {
        unreachable!()
    };
    executor.raise_varargs(raise_kind_arg_as_usize(argc, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_reraise<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::Reraise { depth } = instruction else {
        unreachable!()
    };
    executor.reraise(depth.get(op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_list_extend<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::ListExtend { i } = instruction else {
        unreachable!()
    };
    executor.list_extend(u32_as_usize(i.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_set_add<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::SetAdd { i } = instruction else {
        unreachable!()
    };
    executor.set_add(u32_as_usize(i.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_dict_merge<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::DictMerge { i } = instruction else {
        unreachable!()
    };
    executor.dict_merge(u32_as_usize(i.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_dict_update<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::DictUpdate { i } = instruction else {
        unreachable!()
    };
    executor.dict_update(u32_as_usize(i.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_set_update<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::SetUpdate { i } = instruction else {
        unreachable!()
    };
    executor.set_update(u32_as_usize(i.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_map_add<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::MapAdd { i } = instruction else {
        unreachable!()
    };
    executor.map_add(u32_as_usize(i.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_make_cell<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::MakeCell { i } = instruction else {
        unreachable!()
    };
    executor.make_cell(load_fast_var_num_to_index(i, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_copy_free_vars<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::CopyFreeVars { n } = instruction else {
        unreachable!()
    };
    executor.copy_free_vars(u32_as_usize(n.get(op_arg)))?;
    Ok(StepResult::Continue)
}

pub fn execute_load_deref<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::LoadDeref { i } = instruction else {
        unreachable!()
    };
    executor.load_deref(load_fast_var_num_to_index(i, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_store_deref<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::StoreDeref { i } = instruction else {
        unreachable!()
    };
    executor.store_deref(load_fast_var_num_to_index(i, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_delete_deref<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::DeleteDeref { i } = instruction else {
        unreachable!()
    };
    executor.delete_deref(load_fast_var_num_to_index(i, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_load_common_constant<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::LoadCommonConstant { idx } = instruction else {
        unreachable!()
    };
    executor.load_common_constant(common_constant_arg(idx, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_convert_value<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::ConvertValue { oparg: conv } = instruction else {
        unreachable!()
    };
    executor.convert_value(convert_value_arg(conv, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_load_fast_and_clear<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::LoadFastAndClear { var_num } = instruction else {
        unreachable!()
    };
    executor.load_fast_and_clear(load_fast_var_num_to_index(var_num, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_set_function_attribute<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::SetFunctionAttribute { flag } = instruction else {
        unreachable!()
    };
    executor.set_function_attribute_with_flag(make_function_flag_arg(flag, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_call_intrinsic_1<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::CallIntrinsic1 { func } = instruction else {
        unreachable!()
    };
    executor.call_intrinsic_1(intrinsic_function_1_arg(func, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_call_intrinsic_2<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::CallIntrinsic2 { func } = instruction else {
        unreachable!()
    };
    executor.call_intrinsic_2(intrinsic_function_2_arg(func, op_arg))?;
    Ok(StepResult::Continue)
}

pub fn execute_unpack_ex<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::UnpackEx { counts } = instruction else {
        unreachable!()
    };
    executor.unpack_ex(counts.get(op_arg))?;
    Ok(StepResult::Continue)
}

// Handlers for arms that index the `CodeObject` constant/name pools: the
// dispatch forwards `code` alongside `instruction`/`op_arg`.
pub fn execute_load_const<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: ConstantOpcodeHandler,
{
    let Instruction::LoadConst { consti } = instruction else {
        unreachable!()
    };
    let const_idx = consti.get(op_arg);
    // `pyopcode.py LOAD_CONST` reads `getconstant_w(index)`
    // (`pyopcode.py co_consts_w[index]`): every constant, not just a
    // nested code object, comes from the enclosing PyCode's shared slot.
    let value = executor.constant_at(const_idx, code)?;
    executor.push_value(value)?;
    Ok(StepResult::Continue)
}

pub fn execute_store_name<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: NamespaceOpcodeHandler,
{
    let Instruction::StoreName { namei } = instruction else {
        unreachable!()
    };
    let idx = u32_as_usize(namei.get(op_arg));
    executor.store_name(code.names[idx].as_ref(), idx)?;
    Ok(StepResult::Continue)
}

pub fn execute_store_global<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: NamespaceOpcodeHandler,
{
    let Instruction::StoreGlobal { namei } = instruction else {
        unreachable!()
    };
    let idx = u32_as_usize(namei.get(op_arg));
    executor.store_global(code.names[idx].as_ref(), idx)?;
    Ok(StepResult::Continue)
}

pub fn execute_load_name<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: NamespaceOpcodeHandler,
{
    let Instruction::LoadName { namei } = instruction else {
        unreachable!()
    };
    let idx = u32_as_usize(namei.get(op_arg));
    executor.load_name(code.names[idx].as_ref(), idx)?;
    Ok(StepResult::Continue)
}

pub fn execute_load_global<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: NamespaceOpcodeHandler,
{
    let Instruction::LoadGlobal { namei } = instruction else {
        unreachable!()
    };
    let raw = u32_as_usize(namei.get(op_arg));
    let name_idx = raw >> 1;
    let push_null = (raw & 1) != 0;
    executor.load_global(code.names[name_idx].as_ref(), name_idx, push_null)?;
    Ok(StepResult::Continue)
}

pub fn execute_delete_name<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::DeleteName { namei } = instruction else {
        unreachable!()
    };
    let nameindex = u32_as_usize(namei.get(op_arg));
    executor.delete_name(code.names[nameindex].as_ref(), nameindex)?;
    Ok(StepResult::Continue)
}

pub fn execute_delete_global<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::DeleteGlobal { namei } = instruction else {
        unreachable!()
    };
    executor.delete_global(code.names[u32_as_usize(namei.get(op_arg))].as_ref())?;
    Ok(StepResult::Continue)
}

pub fn execute_delete_attr<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::DeleteAttr { namei } = instruction else {
        unreachable!()
    };
    executor.delete_attr(code.names[u32_as_usize(namei.get(op_arg))].as_ref())?;
    Ok(StepResult::Continue)
}

pub fn execute_store_attr<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::StoreAttr { namei } = instruction else {
        unreachable!()
    };
    let name_idx = u32_as_usize(namei.get(op_arg));
    executor.store_attr_cached(code.names[name_idx].as_ref(), name_idx)?;
    Ok(StepResult::Continue)
}

/// rustpython-codegen `compile.rs` emits `(name_idx << 2) | flags` for
/// `Instruction::ImportName`. rustpython-vm `frame.rs` reads
/// `names[arg >> 2]`. The low bits are the PEP 810 lazy-import flags
/// (`bytecodes.c` `IMPORT_NAME`: bit0 explicit lazy, bit1 forced eager).
#[inline]
pub fn import_name_index(packed: u32) -> usize {
    u32_as_usize(packed >> 2)
}

pub fn execute_import_name<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::ImportName { namei } = instruction else {
        unreachable!()
    };
    let packed = namei.get(op_arg);
    let name_idx = import_name_index(packed);
    executor.import_name(code.names[name_idx].as_ref(), name_idx, packed & 3)?;
    Ok(StepResult::Continue)
}

pub fn execute_import_from<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::ImportFrom { namei } = instruction else {
        unreachable!()
    };
    let name_idx = u32_as_usize(namei.get(op_arg));
    executor.import_from(code.names[name_idx].as_ref())?;
    Ok(StepResult::Continue)
}

pub fn execute_load_attr<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: NamespaceOpcodeHandler,
{
    let Instruction::LoadAttr { namei } = instruction else {
        unreachable!()
    };
    let attr = namei.get(op_arg);
    let name_idx = u32_as_usize(attr.name_idx());
    let name = code.names[name_idx].as_ref();
    if attr.is_method() {
        executor.load_method(name)?;
    } else {
        executor.load_attr_cached(name, name_idx)?;
    }
    Ok(StepResult::Continue)
}

pub fn execute_load_from_dict_or_globals<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::LoadFromDictOrGlobals { i } = instruction else {
        unreachable!()
    };
    let idx = u32_as_usize(i.get(op_arg));
    executor.load_from_dict_or_globals(code.names[idx].as_ref(), idx)?;
    Ok(StepResult::Continue)
}

pub fn execute_load_from_dict_or_deref<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::LoadFromDictOrDeref { i } = instruction else {
        unreachable!()
    };
    let idx = load_fast_var_num_to_index(i, op_arg);
    // `idx` is a localsplus offset (cell / free var), not a `co_names` index.
    let name = crate::pyframe::deref_name_and_kind(code, idx).0;
    executor.load_from_dict_or_deref(idx, name)?;
    Ok(StepResult::Continue)
}

pub fn execute_load_super_attr<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let Instruction::LoadSuperAttr { .. } = instruction else {
        unreachable!()
    };
    let raw = op_arg_as_usize(op_arg);
    let idx = raw >> 2;
    let is_method = (raw & 1) != 0;
    let is_two_arg = (raw & 2) != 0;
    executor.load_super_attr_with(idx, is_method, is_two_arg)?;
    Ok(StepResult::Continue)
}

// dont_look_inside: unsupported-opcode handler; aborts tracing, never hot.
#[majit_macros::dont_look_inside]
pub fn execute_unsupported<E: OpcodeStepExecutor>(
    executor: &mut E,
    instruction: Instruction,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError> {
    let step = executor.unsupported(&instruction)?;
    // Rebuild the StepResult per variant so the Result-of-PyError
    // lowering sees a concrete `Ok(StepResult::_)` shell to rewrite. A
    // bare `Ok(step)` forward of the method's same-typed Result collapses
    // in MIR, leaving this scoped wrapper with no rewritable return.
    match step {
        StepResult::Continue => Ok(StepResult::Continue),
        StepResult::Return(value) => Ok(StepResult::Return(value)),
        StepResult::CloseLoop { loop_header_pc } => Ok(StepResult::CloseLoop { loop_header_pc }),
        StepResult::Yield(value) => Ok(StepResult::Yield(value)),
    }
}

pub fn execute_opcode_step<E: OpcodeStepExecutor>(
    executor: &mut E,
    code: &CodeObject,
    instruction: Instruction,
    op_arg: OpArg,
    next_instr: usize,
) -> Result<StepResult<<E as SharedOpcodeHandler>::Value>, PyError>
where
    E: SharedOpcodeHandler
        + ConstantOpcodeHandler
        + LocalOpcodeHandler
        + NamespaceOpcodeHandler
        + StackOpcodeHandler
        + IterOpcodeHandler
        + TruthOpcodeHandler
        + ControlFlowOpcodeHandler
        + BranchOpcodeHandler
        + ArithmeticOpcodeHandler,
{
    match instruction {
        // CPython 3.14 accepts unknown bytes in `CodeType` / `code.replace`
        // and reports them only when dispatch reaches the instruction.  The
        // constructor stores the raw opcode byte in this placeholder's low
        // arg byte because compiler-core's enum cannot represent it.
        Instruction::Reserved => Err(PyError::new(
            crate::PyErrorKind::SystemError,
            format!("unknown opcode {}", u32::from(op_arg) & 0xff),
        )),
        Instruction::ExtendedArg
        | Instruction::Resume { .. }
        | Instruction::Nop
        | Instruction::Cache
        | Instruction::NotTaken => Ok(StepResult::Continue),

        Instruction::LoadConst { .. } => execute_load_const(executor, code, instruction, op_arg),

        Instruction::LoadSmallInt { .. } => execute_load_small_int(executor, instruction, op_arg),

        Instruction::LoadFast { .. } | Instruction::LoadFastBorrow { .. } => {
            execute_load_fast(executor, code, instruction, op_arg)
        }

        Instruction::LoadFastBorrowLoadFastBorrow { .. } => {
            execute_load_fast_borrow_load_fast_borrow(executor, code, instruction, op_arg)
        }

        Instruction::StoreFast { .. } => execute_store_fast(executor, instruction, op_arg),

        Instruction::LoadFastCheck { .. } => {
            execute_load_fast_check(executor, code, instruction, op_arg)
        }

        Instruction::LoadFastLoadFast { .. } => {
            execute_load_fast_load_fast(executor, instruction, op_arg)
        }

        Instruction::StoreFastLoadFast { .. } => {
            execute_store_fast_load_fast(executor, instruction, op_arg)
        }

        Instruction::StoreFastStoreFast { .. } => {
            execute_store_fast_store_fast(executor, instruction, op_arg)
        }

        Instruction::StoreName { .. } => execute_store_name(executor, code, instruction, op_arg),

        Instruction::StoreGlobal { .. } => {
            execute_store_global(executor, code, instruction, op_arg)
        }

        Instruction::LoadName { .. } => execute_load_name(executor, code, instruction, op_arg),

        Instruction::LoadGlobal { .. } => execute_load_global(executor, code, instruction, op_arg),

        Instruction::PopTop => execute_pop_top(executor),

        Instruction::PushNull => execute_push_null(executor),

        Instruction::Copy { .. } => execute_copy(executor, instruction, op_arg),

        Instruction::Swap { .. } => execute_swap(executor, instruction, op_arg),

        Instruction::BinaryOp { .. } => execute_binary_op(executor, instruction, op_arg),

        Instruction::CompareOp { .. } => execute_compare_op(executor, instruction, op_arg),

        Instruction::UnaryNegative => execute_unary_negative(executor),

        Instruction::UnaryNot => execute_unary_not(executor),

        Instruction::UnaryInvert => execute_unary_invert(executor),

        Instruction::JumpForward { .. } => {
            execute_jump_forward(executor, code, instruction, op_arg, next_instr)
        }

        Instruction::JumpBackward { .. } => {
            execute_jump_backward(executor, code, instruction, op_arg, next_instr)
        }

        Instruction::PopJumpIfFalse { .. } => {
            execute_pop_jump_if_false(executor, code, instruction, op_arg, next_instr)
        }

        Instruction::PopJumpIfTrue { .. } => {
            execute_pop_jump_if_true(executor, code, instruction, op_arg, next_instr)
        }

        Instruction::MakeFunction => execute_make_function(executor),

        Instruction::Call { .. } => execute_call(executor, instruction, op_arg),

        Instruction::ReturnValue => execute_return_value(executor),

        Instruction::BuildList { .. } => execute_build_list(executor, instruction, op_arg),

        Instruction::BuildListFromArg => execute_build_list_from_arg(executor),

        Instruction::BuildTuple { .. } => execute_build_tuple(executor, instruction, op_arg),

        Instruction::BuildMap { .. } => execute_build_map(executor, instruction, op_arg),

        Instruction::StoreSubscr => execute_store_subscr(executor),

        Instruction::ListAppend { .. } => execute_list_append(executor, instruction, op_arg),

        Instruction::UnpackSequence { .. } => {
            execute_unpack_sequence(executor, instruction, op_arg)
        }

        Instruction::GetIter { .. } => execute_get_iter(executor, instruction, op_arg),

        Instruction::ForIter { .. } => {
            execute_for_iter(executor, code, instruction, op_arg, next_instr)
        }

        Instruction::EndFor => execute_end_for(executor),

        Instruction::PopIter => execute_pop_iter(executor),

        Instruction::LoadAttr { .. } => execute_load_attr(executor, code, instruction, op_arg),

        Instruction::StoreAttr { .. } => execute_store_attr(executor, code, instruction, op_arg),

        // ── Generators ──
        Instruction::YieldValue { .. } => execute_yield_value(executor),

        // All other opcodes fall through to unsupported handler.
        // Remaining opcodes (closures, exceptions, imports) will be added
        // ── Closures / cells ──
        Instruction::LoadDeref { .. } => execute_load_deref(executor, instruction, op_arg),
        Instruction::StoreDeref { .. } => execute_store_deref(executor, instruction, op_arg),
        Instruction::DeleteDeref { .. } => execute_delete_deref(executor, instruction, op_arg),

        // ── Import ──
        Instruction::ImportName { .. } => execute_import_name(executor, code, instruction, op_arg),
        Instruction::ImportFrom { .. } => execute_import_from(executor, code, instruction, op_arg),

        // ── Containment / identity tests ──
        Instruction::ContainsOp { .. } => execute_contains_op(executor, instruction, op_arg),
        Instruction::IsOp { .. } => execute_is_op(executor, instruction, op_arg),

        // ── Delete subscript ──
        Instruction::DeleteSubscr => execute_delete_subscr(executor),

        // ── Exception handling (CPython 3.13) ──
        Instruction::PushExcInfo => execute_push_exc_info(executor),
        Instruction::PopExcept => execute_pop_except(executor),
        Instruction::CheckExcMatch => execute_check_exc_match(executor),
        Instruction::CheckEgMatch => execute_check_eg_match(executor),
        Instruction::RaiseVarargs { .. } => execute_raise_varargs(executor, instruction, op_arg),
        Instruction::Reraise { .. } => execute_reraise(executor, instruction, op_arg),

        // ── Collection operations ──
        Instruction::BuildSet { .. } => execute_build_set(executor, instruction, op_arg),
        Instruction::BuildSlice { .. } => execute_build_slice(executor, instruction, op_arg),
        Instruction::BuildString { .. } => execute_build_string(executor, instruction, op_arg),
        Instruction::BuildTemplate => execute_build_template(executor),
        Instruction::BuildInterpolation { .. } => {
            execute_build_interpolation(executor, instruction, op_arg)
        }
        Instruction::ListExtend { .. } => execute_list_extend(executor, instruction, op_arg),
        Instruction::SetAdd { .. } => execute_set_add(executor, instruction, op_arg),
        Instruction::DictMerge { .. } => execute_dict_merge(executor, instruction, op_arg),
        Instruction::DictUpdate { .. } => execute_dict_update(executor, instruction, op_arg),
        Instruction::SetUpdate { .. } => execute_set_update(executor, instruction, op_arg),
        Instruction::MapAdd { .. } => execute_map_add(executor, instruction, op_arg),

        // ── Slicing ──
        Instruction::BinarySlice => execute_binary_slice(executor),
        Instruction::StoreSlice => execute_store_slice(executor),

        // ── Boolean conversion ──
        Instruction::ToBool => execute_to_bool(executor),

        // ── None comparison jumps ──
        Instruction::PopJumpIfNone { .. } => {
            execute_pop_jump_if_none(executor, code, instruction, op_arg, next_instr)
        }
        Instruction::PopJumpIfNotNone { .. } => {
            execute_pop_jump_if_not_none(executor, code, instruction, op_arg, next_instr)
        }

        // ── Closure 3.11+ ──
        Instruction::MakeCell { .. } => execute_make_cell(executor, instruction, op_arg),
        Instruction::CopyFreeVars { .. } => execute_copy_free_vars(executor, instruction, op_arg),

        // ── Generator ──
        Instruction::ReturnGenerator => execute_return_generator(executor),

        // ── Function call variants ──
        Instruction::CallKw { .. } => execute_call_kw(executor, instruction, op_arg),
        Instruction::CallFunctionEx => execute_call_function_ex(executor),

        // ── Common constants ──
        Instruction::LoadCommonConstant { .. } => {
            execute_load_common_constant(executor, instruction, op_arg)
        }

        // ── Class support ──
        Instruction::LoadBuildClass => execute_load_build_class(executor),

        // ── Delete ops ──
        Instruction::DeleteFast { .. } => execute_delete_fast(executor, instruction, op_arg),
        Instruction::DeleteName { .. } => execute_delete_name(executor, code, instruction, op_arg),
        Instruction::DeleteGlobal { .. } => {
            execute_delete_global(executor, code, instruction, op_arg)
        }
        Instruction::DeleteAttr { .. } => execute_delete_attr(executor, code, instruction, op_arg),

        // ── Load super attr ──
        // CPython 3.12: stack = [global_super, class, self] → super(class, self).attr
        Instruction::LoadSuperAttr { .. } => {
            execute_load_super_attr(executor, code, instruction, op_arg)
        }

        // ── Misc ──
        // SETUP_ANNOTATIONS: ensure that the current locals namespace has
        // an `__annotations__` dict. The class body / module top-level
        // emits this once before any annotated assignment so STORE_SUBSCR
        // can populate it.
        Instruction::SetupAnnotations => execute_setup_annotations(executor),
        Instruction::LoadLocals => execute_load_locals(executor),

        // ── String formatting (f-strings) ──
        Instruction::FormatSimple => execute_format_simple(executor),
        Instruction::FormatWithSpec => execute_format_with_spec(executor),
        Instruction::ConvertValue { .. } => execute_convert_value(executor, instruction, op_arg),

        // ── Sequence matching ──
        Instruction::GetLen => execute_get_len(executor),

        // ── Loop / generator control ──
        Instruction::JumpBackwardNoInterrupt { .. } => {
            execute_jump_backward_no_interrupt(executor, instruction, op_arg, next_instr)
        }

        // ── Load fast and clear (comprehension scope) ──
        Instruction::LoadFastAndClear { .. } => {
            execute_load_fast_and_clear(executor, instruction, op_arg)
        }

        // ── Set function attribute (closure, annotations, etc.) ──
        Instruction::SetFunctionAttribute { .. } => {
            execute_set_function_attribute(executor, instruction, op_arg)
        }

        // ── Scoping ──
        Instruction::LoadFromDictOrGlobals { .. } => {
            execute_load_from_dict_or_globals(executor, code, instruction, op_arg)
        }
        Instruction::LoadFromDictOrDeref { .. } => {
            execute_load_from_dict_or_deref(executor, code, instruction, op_arg)
        }

        // ── Pattern matching (Python 3.10+) ──
        Instruction::MatchMapping => execute_match_mapping(executor),
        Instruction::MatchSequence => execute_match_sequence(executor),
        Instruction::MatchKeys => execute_match_keys(executor),
        Instruction::MatchClass { .. } => execute_match_class(executor, instruction, op_arg),

        // ── Unpack extended ──
        Instruction::UnpackEx { .. } => execute_unpack_ex(executor, instruction, op_arg),

        // ── Intrinsics ──
        Instruction::CallIntrinsic1 { .. } => {
            execute_call_intrinsic_1(executor, instruction, op_arg)
        }
        Instruction::CallIntrinsic2 { .. } => {
            execute_call_intrinsic_2(executor, instruction, op_arg)
        }

        // ── Await ──
        Instruction::GetAwaitable { .. } => execute_get_awaitable(executor, instruction, op_arg),

        Instruction::GetAiter => execute_get_aiter(executor),
        Instruction::GetAnext => execute_get_anext(executor),
        Instruction::EndAsyncFor => execute_end_async_for(executor),

        Instruction::Send { .. } => execute_send(executor, code, op_arg, next_instr),

        Instruction::EndSend => execute_end_send(executor),

        Instruction::CleanupThrow => execute_cleanup_throw(executor),

        // ── Misc stubs ──
        // Pops obj, pushes (callable, self_or_null).
        // Used by `with` statement to load __enter__ / __exit__.
        // RustPython: frame.rs LoadSpecial, delegates to get_special_method.
        // Pyre: delegate to load_method with the special method name.
        Instruction::LoadSpecial { .. } => execute_load_special(executor, instruction, op_arg),
        Instruction::ExitInitCheck => Ok(StepResult::Continue),
        // CPython 3.14 WITH_EXCEPT_START:
        //   val = TOS         (the exception)
        //   exit_func = stack[-4]
        //   res = exit_func(type(val), val, val.__traceback__)
        //   push(res)
        Instruction::WithExceptStart => execute_with_except_start(executor),

        _ => execute_unsupported(executor, instruction),
    }
}

pub fn jump_target_forward(instructions: &[CodeUnit], next_instr: usize, delta: usize) -> usize {
    skip_caches(instructions, next_instr) + delta
}

fn jump_target_backward(instructions: &[CodeUnit], next_instr: usize, delta: usize) -> usize {
    skip_caches(instructions, next_instr) - delta
}

/// The instruction `pos` belongs to: `pos` itself, or the opcode whose cache
/// region covers it.
///
/// The backward mirror of [`skip_caches`].  pyre materializes each cache word
/// as an `Instruction::Cache` unit and dispatches it as a no-op, so a frame's
/// `last_instr` can name one, and `next_instr() = last_instr + 1` steps into
/// the region rather than over it.  That is sound for dispatch and for every
/// resume coordinate, and it is exactly what a Python-visible offset must not
/// be: `f_lasti` names an instruction (`_PyInterpreterFrame_LASTI` is derived
/// from `instr_ptr`, which `TARGET(CALL)` sets before advancing over the
/// caches), and `frame.f_lineno = N` indexes `mark_stacks`, whose cache
/// entries keep `MARK_UNINITIALIZED`.
///
/// Bounded by the widest cache region in the instruction set, so the walk is
/// a few steps rather than a scan.
pub fn owning_instruction(instructions: &[CodeUnit], mut pos: usize) -> usize {
    while pos > 0 && pos < instructions.len() {
        let mut state = OpArgState::default();
        let (instruction, _) = state.get(instructions[pos]);
        if matches!(instruction, Instruction::Cache) {
            pos -= 1;
        } else {
            break;
        }
    }
    pos
}

pub fn skip_caches(instructions: &[CodeUnit], mut pos: usize) -> usize {
    while pos < instructions.len() {
        let mut state = OpArgState::default();
        let (instruction, _) = state.get(instructions[pos]);
        if matches!(instruction, Instruction::Cache) {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::{
        decode_instruction_at, decode_instruction_for_dispatch, decode_instruction_forward,
        decode_instruction_forward_dispatch, decode_instruction_forward_packed,
        decode_instruction_forward_pc,
    };
    use crate::bytecode::{CodeUnit, CodeUnits, Instruction, OpArgByte};
    use crate::{OpArgState, compile_exec};

    #[test]
    fn decode_instruction_across_extended_arg_prefix() {
        let source = (0..400)
            .map(|i| format!("v{i} = {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let code = compile_exec(&source).expect("compile failed");

        let target_pc = code
            .instructions
            .iter()
            .enumerate()
            .find_map(|(pc, unit)| {
                if pc > 0
                    && matches!(code.instructions[pc - 1].op, Instruction::ExtendedArg)
                    && !matches!(unit.op, Instruction::ExtendedArg)
                {
                    Some(pc)
                } else {
                    None
                }
            })
            .expect("expected an instruction with an ExtendedArg prefix");

        // decode_instruction_at matches a forward OpArgState decode at target_pc.
        let mut forward = OpArgState::default();
        let mut expected = None;
        for (pc, unit) in code.instructions.iter().copied().enumerate() {
            let decoded = forward.get(unit);
            if pc == target_pc {
                expected = Some(decoded);
                break;
            }
        }

        assert_eq!(
            decode_instruction_at(&code, target_pc).map(|(instruction, arg)| {
                (std::mem::discriminant(&instruction), u32::from(arg))
            }),
            expected
                .map(|(instruction, arg)| (std::mem::discriminant(&instruction), u32::from(arg)))
        );

        // decode_instruction_for_dispatch, starting on the ExtendedArg prefix,
        // absorbs the prefix and lands on the same target instruction.
        let prefix_pc = target_pc - 1;

        let (decoded_pc, decoded_instr, decoded_arg) =
            decode_instruction_for_dispatch(&code, prefix_pc).expect("dispatch decode failed");
        let (target_instr, target_arg) =
            decode_instruction_at(&code, target_pc).expect("target decode failed");

        assert_eq!(decoded_pc, target_pc);
        assert_eq!(
            std::mem::discriminant(&decoded_instr),
            std::mem::discriminant(&target_instr)
        );
        assert_eq!(u32::from(decoded_arg), u32::from(target_arg));

        // decode_instruction_forward, entered on the ExtendedArg prefix start,
        // reconstructs the identical dispatch triple with no backward scan.
        let (fwd_pc, fwd_instr, fwd_arg) =
            decode_instruction_forward(&code, prefix_pc).expect("forward decode failed");
        assert_eq!(fwd_pc, decoded_pc);
        assert_eq!(
            std::mem::discriminant(&fwd_instr),
            std::mem::discriminant(&decoded_instr)
        );
        assert_eq!(u32::from(fwd_arg), u32::from(decoded_arg));

        // Entered directly at target_pc (the real opcode past its ExtendedArg
        // prefix — the coordinate a jump/loop-header/resume lands on), the
        // decode backs up over the prefix and reconstructs the full triple,
        // including the extended arg bits, matching the prefix-start entry.
        let (fwd_target_pc, fwd_target_instr, fwd_target_arg) =
            decode_instruction_forward(&code, target_pc).expect("forward decode at target failed");
        assert_eq!(fwd_target_pc, target_pc);
        assert_eq!(
            std::mem::discriminant(&fwd_target_instr),
            std::mem::discriminant(&target_instr)
        );
        assert_eq!(u32::from(fwd_target_arg), u32::from(target_arg));
    }

    #[test]
    fn decode_instruction_for_dispatch_rejects_malformed_extended_arg_chain() {
        let code = compile_exec("x = 1").expect("compile failed");
        assert!(
            code.instructions.len() >= 2,
            "expected at least two instructions"
        );
        // The malformed-chain guard is `instruction_discriminant < 44`.
        // GET_ITER is opcode 70 and takes an arg, so ExtendedArg+GetIter
        // is a valid decode. END_FOR is opcode 9 and still trips the guard.
        unsafe {
            code.instructions.replace_op(0, Instruction::ExtendedArg);
            code.instructions.replace_op(1, Instruction::EndFor);
        }
        assert!(decode_instruction_for_dispatch(&code, 0).is_err());
    }

    /// An opcode byte the enum cannot spell decodes to `Reserved`, and the
    /// `unknown opcode N` report is raised by dispatch rather than by the
    /// decoder, so an `ExtendedArg` prefix in front of one must not be
    /// rejected as a malformed chain.
    #[test]
    fn decode_instruction_for_dispatch_keeps_reserved_after_extended_arg() {
        let code = compile_exec("x = 1").expect("compile failed");
        assert!(
            code.instructions.len() >= 2,
            "expected at least two instructions"
        );
        unsafe {
            code.instructions.replace_op(0, Instruction::ExtendedArg);
            code.instructions.replace_op(1, Instruction::Reserved);
        }
        let (opcode_pc, instruction, _) =
            decode_instruction_for_dispatch(&code, 0).expect("Reserved stays decodable");
        assert_eq!(opcode_pc, 1);
        assert!(matches!(instruction, Instruction::Reserved));
    }

    #[test]
    fn forward_decode_matches_full_scan_and_dispatch() {
        let source = (0..400)
            .map(|i| format!("v{i} = {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let code = compile_exec(&source).expect("compile failed");

        // Naive full-scan reference: a single OpArgState folded over the whole
        // instruction stream, recording the decoded (instruction, arg) at every
        // index.
        let full: Vec<_> = {
            let mut state = OpArgState::default();
            code.instructions
                .iter()
                .copied()
                .map(|unit| state.get(unit))
                .collect()
        };

        let mut prefix_starts = 0usize;
        let mut real_opcodes = 0usize;

        // Walk logical instructions. `i` is always a logical-instruction start
        // (a bare real opcode or the first ExtendedArg of a prefix run), never
        // mid-prefix, so it is a valid dispatch entry.
        let mut i = 0;
        while i < code.instructions.len() {
            let is_prefix_start = matches!(full[i].0, Instruction::ExtendedArg);

            // Real opcode pc: skip the contiguous ExtendedArg prefix forward.
            let mut real = i;
            while matches!(full[real].0, Instruction::ExtendedArg) {
                real += 1;
            }
            real_opcodes += 1;
            if is_prefix_start {
                prefix_starts += 1;
            }

            let (fwd_pc, fwd_instr, fwd_arg) =
                decode_instruction_forward(&code, i).expect("forward decode failed");
            let (disp_pc, disp_instr, disp_arg) =
                decode_instruction_for_dispatch(&code, i).expect("dispatch decode failed");

            // forward decode == naive full-scan at the resolved real opcode.
            assert_eq!(fwd_pc, real);
            assert_eq!(
                std::mem::discriminant(&fwd_instr),
                std::mem::discriminant(&full[real].0)
            );
            assert_eq!(u32::from(fwd_arg), u32::from(full[real].1));

            // forward decode == decode_instruction_for_dispatch.
            assert_eq!(fwd_pc, disp_pc);
            assert_eq!(
                std::mem::discriminant(&fwd_instr),
                std::mem::discriminant(&disp_instr)
            );
            assert_eq!(u32::from(fwd_arg), u32::from(disp_arg));

            // Entering directly at the real opcode `real` (the coordinate a
            // JIT jump/loop-header/resume targets — past any ExtendedArg
            // prefix) must reconstruct the identical triple: same pc, same
            // opcode, and the full accumulated arg. This is the getwidth
            // FOR_ITER case where `real > i`.
            let (real_pc, real_instr, real_arg) = decode_instruction_forward(&code, real)
                .expect("forward decode at real opcode failed");
            assert_eq!(real_pc, real);
            assert_eq!(
                std::mem::discriminant(&real_instr),
                std::mem::discriminant(&full[real].0)
            );
            assert_eq!(u32::from(real_arg), u32::from(full[real].1));

            i = real + 1;
        }

        assert!(
            prefix_starts > 0,
            "expected the 400-assignment source to emit an ExtendedArg prefix"
        );
        assert!(real_opcodes > 0);
    }

    #[test]
    fn decode_instruction_forward_rejects_malformed_extended_arg_chain() {
        let code = compile_exec("x = 1").expect("compile failed");
        assert!(
            code.instructions.len() >= 2,
            "expected at least two instructions"
        );
        // The malformed-chain guard is `instruction_discriminant < 44`.
        // GET_ITER is opcode 70 and takes an arg, so ExtendedArg+GetIter
        // is a valid decode. END_FOR is opcode 9 and still trips the guard.
        unsafe {
            code.instructions.replace_op(0, Instruction::ExtendedArg);
            code.instructions.replace_op(1, Instruction::EndFor);
        }
        assert!(decode_instruction_forward(&code, 0).is_err());
    }

    #[test]
    fn scalar_forward_decode_handles_two_extended_arg_prefixes() {
        let mut code = compile_exec("x = 1").expect("compile failed");
        code.instructions = CodeUnits::from([
            CodeUnit::new(Instruction::ExtendedArg, OpArgByte::new(0x01)),
            CodeUnit::new(Instruction::ExtendedArg, OpArgByte::new(0x02)),
            CodeUnit::new(Instruction::Reserved, OpArgByte::new(0x03)),
        ]);

        assert_eq!(decode_instruction_forward_pc(&code, 0), 2);
        assert_eq!(decode_instruction_forward_pc(&code, 2), 2);
        let expected = u64::from(u8::from(Instruction::Reserved)) | (0x01_02_03u64 << 8);
        assert_eq!(decode_instruction_forward_packed(&code, 0), expected);
        assert_eq!(decode_instruction_forward_packed(&code, 2), expected);
        assert_eq!(decode_instruction_forward_dispatch(&code, 0), (2, expected));
        assert_eq!(decode_instruction_forward_dispatch(&code, 2), (2, expected));
    }

    // A JIT jump/loop-header/resume coordinate points at the real opcode past
    // its ExtendedArg prefix (e.g. the getwidth outer FOR_ITER at a pc whose
    // preceding unit is EXTENDED_ARG). Entered there, the forward decode must
    // recover the full extended oparg, not just the real opcode's own byte.
    #[test]
    fn forward_decode_at_real_opcode_recovers_extended_arg() {
        // A loop over a long-enough body forces the FOR_ITER jump delta above
        // 255, so its oparg needs an EXTENDED_ARG prefix. The 300 statements
        // push the END_FOR/JUMP_BACKWARD past the one-byte range.
        let body = (0..300)
            .map(|i| format!("    x{i} = {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let source = format!("for e in seq:\n{body}\n");
        let code = compile_exec(&source).expect("compile failed");

        // Reference: naive full-scan decode at every index.
        let full: Vec<_> = {
            let mut state = OpArgState::default();
            code.instructions
                .iter()
                .copied()
                .map(|unit| state.get(unit))
                .collect()
        };

        // Find a real opcode (not ExtendedArg) whose immediately preceding
        // unit is ExtendedArg — the shape a JIT resume/loop-header pc lands on.
        let real_with_prefix = (1..code.instructions.len())
            .find(|&pc| {
                matches!(code.instructions[pc - 1].op, Instruction::ExtendedArg)
                    && !matches!(code.instructions[pc].op, Instruction::ExtendedArg)
            })
            .expect("expected a real opcode preceded by ExtendedArg");

        // The recovered arg must exceed the single-byte range, proving the
        // high byte from the prefix was folded in rather than dropped.
        let expected_arg = u32::from(full[real_with_prefix].1);
        assert!(
            expected_arg > 0xff,
            "test setup: expected an extended (>255) oparg, got {expected_arg}"
        );

        let (pc, instr, arg) = decode_instruction_forward(&code, real_with_prefix)
            .expect("forward decode at real opcode failed");
        assert_eq!(pc, real_with_prefix);
        assert_eq!(
            std::mem::discriminant(&instr),
            std::mem::discriminant(&full[real_with_prefix].0)
        );
        assert_eq!(u32::from(arg), expected_arg);

        // Equivalent to the backward-scanning dispatch decode at the same pc.
        let (disp_pc, disp_instr, disp_arg) =
            decode_instruction_for_dispatch(&code, real_with_prefix)
                .expect("dispatch decode failed");
        assert_eq!(pc, disp_pc);
        assert_eq!(
            std::mem::discriminant(&instr),
            std::mem::discriminant(&disp_instr)
        );
        assert_eq!(u32::from(arg), u32::from(disp_arg));
    }
}
