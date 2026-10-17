//! JIT-internal exception/control-flow types.
//!
//! Mirrors RPython's `jitexc.py`: exceptions raised and caught within
//! the JIT infrastructure, never exposed to user code.

use majit_ir::{GcRef, GreenKey, GreenType, Value};

/// jitexc.py JitException — base class for all JIT control flow.
///
/// In RPython these are Python exceptions. In Rust we model them as an enum
/// returned via `Result::Err(JitException)` from blackhole execution.
#[derive(Debug, Clone, PartialEq)]
pub enum JitException {
    /// jitexc.py DoneWithThisFrameVoid
    DoneWithThisFrameVoid,
    /// jitexc.py DoneWithThisFrameInt
    DoneWithThisFrameInt(i64),
    /// jitexc.py DoneWithThisFrameRef
    DoneWithThisFrameRef(GcRef),
    /// jitexc.py DoneWithThisFrameFloat
    DoneWithThisFrameFloat(f64),
    /// jitexc.py ExitFrameWithExceptionRef
    ExitFrameWithExceptionRef(GcRef),
    /// jitexc.py ContinueRunningNormally
    ContinueRunningNormally(Box<ContinueRunningNormallyArgs>),
    /// pyre-only: the blackhole stopped without finishing the frame.
    ///
    /// Reached from a pyre `abort_permanent` marker and from the two
    /// unresolved-callee refusals (`reject_symbolic_residual_call`,
    /// `reject_unresolved_inline_call`), none of which upstream has: RPython's
    /// jitcodes cover every operation the codewriter accepted, so a blackhole
    /// frame always ends in one of the variants above.
    ///
    /// It is deliberately NOT `DoneWithThisFrameVoid`.  That variant says the
    /// frame ran to its `return` and produced no value, which is a result the
    /// caller installs and the program observes; a bail says the opposite —
    /// the frame is mid-execution, its resume coordinate is already stamped
    /// into the interpreter frame, and the interpreter has to take it back.
    /// Spelling one as the other turned an abort into a Python-visible `None`.
    BailToInterpreter,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ContinueRunningNormallyArgs {
    pub green_int: Vec<i64>,
    pub green_ref: Vec<i64>,
    pub green_float: Vec<i64>,
    pub red_int: Vec<i64>,
    pub red_ref: Vec<i64>,
    pub red_float: Vec<i64>,
}

/// Result of a completed trace execution.
///
/// Mirrors DoneWithThisFrame{Void,Int,Ref,Float} in jitexc.py.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DoneWithThisFrame {
    /// jitexc.py DoneWithThisFrameVoid
    Void,
    /// jitexc.py DoneWithThisFrameInt
    Int(i64),
    /// jitexc.py DoneWithThisFrameRef
    Ref(GcRef),
    /// jitexc.py DoneWithThisFrameFloat
    Float(f64),
}

impl DoneWithThisFrame {
    /// Create from a typed Value.
    pub fn from_value(v: Value) -> Self {
        match v {
            Value::Int(i) => DoneWithThisFrame::Int(i),
            Value::Ref(r) => DoneWithThisFrame::Ref(r),
            Value::Float(f) => DoneWithThisFrame::Float(f),
            Value::Void => DoneWithThisFrame::Void,
        }
    }
}

impl std::fmt::Display for DoneWithThisFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Void => f.write_str("DoneWithThisFrameVoid"),
            Self::Int(v) => write!(f, "DoneWithThisFrameInt({v})"),
            Self::Ref(r) => write!(f, "DoneWithThisFrameRef({:#x})", r.0),
            Self::Float(v) => write!(f, "DoneWithThisFrameFloat({:#x})", v.to_bits()),
        }
    }
}

impl std::error::Error for DoneWithThisFrame {}

/// The trace exited with an exception.
///
/// Mirrors ExitFrameWithExceptionRef in jitexc.py.
#[derive(Debug, Clone, PartialEq)]
pub struct ExitFrameWithException {
    pub exc_value: GcRef,
}

impl ContinueRunningNormallyArgs {
    /// Green banks a merge point or header snapshot already grouped by kind.
    ///
    /// `warmspot.py handle_jitexception` reads `green_int` / `green_ref` /
    /// `green_float` in declaration order. The first int green is the
    /// portal pc (dispatch i0).
    pub fn from_green_banks(
        green_int: Vec<i64>,
        green_ref: Vec<i64>,
        green_float: Vec<i64>,
    ) -> Self {
        Self {
            green_int,
            green_ref,
            green_float,
            red_int: Vec::new(),
            red_ref: Vec::new(),
            red_float: Vec::new(),
        }
    }

    /// Portal pc: `green_int[0]`, the dispatch-entry i0 convention.
    pub fn portal_pc(&self) -> usize {
        let pc = *self
            .green_int
            .first()
            .expect("ContinueRunningNormally green_int[0] is the portal pc");
        usize::try_from(pc).unwrap_or(usize::MAX)
    }

    /// Copy the six banks into `self`, keeping each Vec's capacity.
    ///
    /// `blackhole.py` `recycle_merge_point_args` recovers the lists after
    /// `handle_jitexception` has read them. `Vec::clone` would drop that
    /// capacity; `Vec::clone_from` reuses it.
    pub fn copy_from_args(&mut self, other: &Self) {
        self.green_int.clone_from(&other.green_int);
        self.green_ref.clone_from(&other.green_ref);
        self.green_float.clone_from(&other.green_float);
        self.red_int.clone_from(&other.red_int);
        self.red_ref.clone_from(&other.red_ref);
        self.red_float.clone_from(&other.red_float);
    }

    /// Fill the green banks from a header snapshot, keeping capacity.
    pub fn copy_green_banks_from(&mut self, ints: &[i64], refs: &[i64], floats: &[i64]) {
        self.green_int.clear();
        self.green_int.extend_from_slice(ints);
        self.green_ref.clear();
        self.green_ref.extend_from_slice(refs);
        self.green_float.clear();
        self.green_float.extend_from_slice(floats);
        self.red_int.clear();
        self.red_ref.clear();
        self.red_float.clear();
    }

    /// Declaration-order greens from a register file, keeping capacity.
    ///
    /// `blackhole.py bhimpl_jit_merge_point` indexes the merge-point's
    /// green operands, not the JitCode register file by position. A
    /// loop-carried green lives in its `join_merge` header register.
    pub fn fill_greens_from_named_registers(
        &mut self,
        green_i: &[u8],
        green_r: &[u8],
        green_f: &[u8],
        registers_i: &[i64],
        registers_r: &[i64],
        registers_f: &[i64],
    ) {
        fn fill(dest: &mut Vec<i64>, bank: &[i64], regs: &[u8], what: &str) {
            dest.clear();
            dest.extend(regs.iter().map(|&reg| {
                *bank.get(reg as usize).unwrap_or_else(|| {
                    panic!(
                        "merge-point green {what} register {reg} is missing \
                         from the register file (blackhole.py bhimpl_jit_merge_point)"
                    )
                })
            }));
        }
        fill(&mut self.green_int, registers_i, green_i, "int");
        fill(&mut self.green_ref, registers_r, green_r, "ref");
        fill(&mut self.green_float, registers_f, green_f, "float");
        self.red_int.clear();
        self.red_ref.clear();
        self.red_float.clear();
    }
}

/// What `warmspot.py handle_jitexception` does after a compiled or
/// blackhole exit: re-enter the portal (`ContinueRunningNormally`),
/// return (`DoneWithThisFrame*`), or propagate
/// (`ExitFrameWithExceptionRef`). Distinct variants, not empty banks.
///
/// `ContinueRunningNormally` always carries the six lists
/// (`bhimpl_jit_merge_point` already built them). A blackhole
/// `DoneWithThisFrame*` / bail / exception that still runs the native
/// epilogue carries them too, so a loop-carried green written before
/// Halt is assigned. A compiled JUMP, FINISH, or a guard that stays in
/// assembler does not: `jtransform.py promote_greens` already passed,
/// and `warmspot.py handle_jitexception` reads no banks on
/// `DoneWithThisFrame*` from `execute_assembler`.
#[derive(Clone, Debug, PartialEq)]
pub enum PortalResume {
    /// jitexc.py ContinueRunningNormally
    ContinueRunningNormally(ContinueRunningNormallyArgs),
    /// jitexc.py DoneWithThisFrame*. `Some` only when a blackhole Halt
    /// still runs the native epilogue (`bhimpl_jit_merge_point` mapping).
    DoneWithThisFrame(Option<ContinueRunningNormallyArgs>),
    /// pyre-only BailToInterpreter
    BailToInterpreter(Option<ContinueRunningNormallyArgs>),
    /// jitexc.py ExitFrameWithExceptionRef
    ExitFrameWithException {
        pc: usize,
        args: Option<ContinueRunningNormallyArgs>,
    },
    /// Successful compiled-loop JUMP, a tracing start, or a counter tick.
    ///
    /// The JUMP path has already passed the loop header's `guard_value` on
    /// every loop-carried green (`jtransform.py promote_greens`), so the
    /// native locals match those greens. A tick does not run compiled
    /// code.
    ResumeAt(usize),
}

impl PortalResume {
    pub fn args(&self) -> Option<&ContinueRunningNormallyArgs> {
        match self {
            Self::ContinueRunningNormally(args) => Some(args),
            Self::DoneWithThisFrame(args)
            | Self::BailToInterpreter(args)
            | Self::ExitFrameWithException { args, .. } => args.as_ref(),
            Self::ResumeAt(_) => None,
        }
    }

    /// Take the banks so the driver can return them to
    /// `recycle_merge_point_args` capacity (`portal_resume_scratch`).
    pub fn into_args(self) -> Option<ContinueRunningNormallyArgs> {
        match self {
            Self::ContinueRunningNormally(args) => Some(args),
            Self::DoneWithThisFrame(args)
            | Self::BailToInterpreter(args)
            | Self::ExitFrameWithException { args, .. } => args,
            Self::ResumeAt(_) => None,
        }
    }

    /// `Some(pc)` keeps the native dispatch loop running. `None` ends it
    /// so the function epilogue runs (`DoneWithThisFrame*`).
    pub fn resume_pc(&self) -> Option<usize> {
        match self {
            Self::ContinueRunningNormally(args) => {
                let pc = args.portal_pc();
                (pc != usize::MAX).then_some(pc)
            }
            Self::DoneWithThisFrame(_) | Self::BailToInterpreter(_) => None,
            Self::ExitFrameWithException { pc, .. } => (*pc != usize::MAX).then_some(*pc),
            Self::ResumeAt(pc) => (*pc != usize::MAX).then_some(*pc),
        }
    }
}

/// Inverse of `majit_ir::GreenAsI64` for a plain owned value.
///
/// The `#[jit_interp]` merge-point expansion assigns a loop-local green
/// from the `ContinueRunningNormally` banks. `as _` does not compile for
/// `bool`. A reference cannot be rebuilt from bits, so `&T` / `&mut T`
/// are not implemented.
pub trait GreenFromI64: Sized {
    fn from_green_i64(bits: i64) -> Self;
}

macro_rules! impl_green_from_i64_int {
    ($($ty:ty),*) => {
        $(
            impl GreenFromI64 for $ty {
                #[inline(always)]
                fn from_green_i64(bits: i64) -> Self {
                    bits as $ty
                }
            }
        )*
    };
}

impl_green_from_i64_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl GreenFromI64 for bool {
    #[inline(always)]
    fn from_green_i64(bits: i64) -> Self {
        bits != 0
    }
}

impl GreenFromI64 for f64 {
    #[inline(always)]
    fn from_green_i64(bits: i64) -> Self {
        f64::from_bits(bits as u64)
    }
}

impl GreenFromI64 for f32 {
    #[inline(always)]
    fn from_green_i64(bits: i64) -> Self {
        // `GreenAsI64 for f32` stores `(self as f64).to_bits()`.
        f64::from_bits(bits as u64) as f32
    }
}

/// The loop is not vectorizable.
///
/// Mirrors NotAVectorizeableLoop in jitexc.py.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotAVectorizeableLoop;

/// The loop is not profitable to vectorize.
///
/// Mirrors NotAProfitableLoop in jitexc.py.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotAProfitableLoop;
