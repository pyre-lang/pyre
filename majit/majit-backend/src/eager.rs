//! Explicit, non-tracing compilation of backend-ready majit IR.
//!
//! This opt-in embedding API wraps the existing `AbstractCPU.compile_loop`
//! contract (`rpython/jit/backend/model.py`), just as upstream's
//! `runner_test.Runner.execute_operations` submits IR without a metainterp.
//! It is a user-requested API extension, not a replacement for PyPy's tracing,
//! optimization, guard/resume, or backend lifecycle machinery.
//!
//! The caller lowers its language to backend-ready operations (including
//! descriptors, guards and explicit exits). No sample inputs are executed,
//! no hot counters are consulted, and no optimizer or interpreter fallback
//! is implicitly run. This is in-process eager compilation, not object-file
//! AOT or a general-purpose CFG frontend.
//!
//! Pass an already configured backend: its runtime/GC, completion and exception
//! descriptors, and `setup_once` remain the embedding's responsibility. This
//! API deliberately does not replace those attachments on a shared PyPy CPU.
//! Execution retains the ordinary JITFRAME ABI and its costs.
//!
//! # Example
//!
//! The embedding supplies its initialized dynasm, Cranelift, or other
//! [`Backend`] and a fresh token from its existing token namespace:
//!
//! ```no_run
//! use std::sync::Arc;
//! use majit_backend::{Backend, JitCellToken};
//! use majit_backend::eager::CompiledIr;
//! use majit_ir::{ConstMap, InputArg, Op, OpCode, OpRc, OpRef, Type, Value};
//! use majit_ir::operand::Operand;
//! use majit_ir::descr::make_finish_descr;
//!
//! fn add_one(cpu: &mut dyn Backend, fresh_token: Arc<JitCellToken>) {
//!     let x = InputArg::new_int_rc(0);
//!     let add = OpRc::new(Op::new(OpCode::IntAdd, &[
//!         Operand::from_bound_inputarg(&x),
//!         Operand::from_opref(OpRef::const_int(1)),
//!     ]));
//!     add.pos().set(OpRef::int_op(1));
//!     let finish = OpRc::new(Op::with_descr(OpCode::Finish,
//!         &[Operand::from_bound_op(&add)],
//!         make_finish_descr(0, vec![Type::Int])));
//!     // SAFETY: well-formed integer-only IR, configured CPU and fresh token.
//!     let mut compiled = unsafe {
//!         CompiledIr::compile(cpu, fresh_token, &[x],
//!             &[add, finish], ConstMap::default())
//!     }.expect("backend supports integer addition");
//!     // SAFETY: no pointers, allocation, or runtime calls in this IR.
//!     let exit = unsafe { compiled.execute(&[Value::Int(41)]) }.unwrap();
//!     assert!(exit.is_finish);
//!     assert_eq!(exit.typed_outputs, vec![Value::Int(42)]);
//! }
//! ```

use std::sync::Arc;

use majit_ir::forwarding::ForwardingHost;
use majit_ir::{Const, ConstMap, InputArgRc, OpRc, Type, Value};

use crate::{AsmInfo, Backend, BackendError, JitCellToken, RawExecResult};

/// An eagerly compiled IR unit, tied to the backend that compiled it.
///
/// The exclusive borrow prevents backend replacement or reconfiguration while
/// this handle is live. The token keeps code and its backend-owned descriptors
/// alive. Drop the handle to use the backend's lower-level APIs again; retain
/// a clone of the supplied token if subsequent bridge compilation is needed.
/// Like the underlying runtime, this handle is execution-thread-affine.
///
/// ```compile_fail
/// use majit_backend::eager::CompiledIr;
/// fn require_send<T: Send>() {}
/// require_send::<CompiledIr<'static>>();
/// ```
pub struct CompiledIr<'backend> {
    backend: &'backend mut dyn Backend,
    token: Arc<JitCellToken>,
    input_types: Vec<Type>,
    asm_info: AsmInfo,
    // Backend is Send, but the runtime's GC/exception state is thread-affine.
    _thread_affinity: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl<'backend> CompiledIr<'backend> {
    /// Compile immediately using the supplied backend, without tracing.
    ///
    /// `token` must be fresh and its number unique in the embedding's token
    /// namespace, including tokens made by a coexisting tracing frontend.
    /// The caller supplies it rather than this API introducing a second ID
    /// allocator. Backend errors are returned unchanged, never interpreted as
    /// a request to execute an interpreter. A failed attempt consumes the token:
    /// use a fresh token for retries, as backend compilation can partially fill it.
    ///
    /// `constants` replaces the backend's pending constant pool for this
    /// submission and is cleared after either success or failure.
    ///
    /// # Safety
    ///
    /// The IR must satisfy the selected backend's `compile_loop` contract:
    /// valid typed operands, descriptors, exits and control-flow targets, with
    /// any optimizer forwarding already resolved. Runtime hooks and pointer
    /// constants must be valid, and GC roots, called functions and referenced
    /// tokens must remain alive for compilation and execution. This API is not
    /// an IR verifier or a sandbox for untrusted machine-level operations.
    pub unsafe fn compile(
        backend: &'backend mut dyn Backend,
        token: Arc<JitCellToken>,
        inputargs: &[InputArgRc],
        operations: &[OpRc],
        constants: ConstMap<Const>,
    ) -> Result<Self, BackendError> {
        if token.compiled.get().is_some() || token.inputarg_types.get().is_some() {
            return Err(BackendError::CompilationFailed(
                "eager compilation requires a fresh JitCellToken".into(),
            ));
        }
        if operations.is_empty() || inputargs.iter().any(|arg| arg.tp.get() == Type::Void) {
            return Err(BackendError::CompilationFailed(
                "eager compilation requires nonempty IR and non-void input arguments".into(),
            ));
        }
        let input_types = inputargs.iter().map(|arg| arg.tp.get()).collect();
        // compile.py send_loop_to_backend: forget_optimization_info(operations/inputargs).
        for iarg in inputargs {
            iarg.clear_forwarded();
        }
        for op in operations {
            op.clear_forwarded();
        }
        backend.set_constants_pool(constants);
        let result = backend.compile_loop(inputargs, operations, &token);
        backend.set_constants_pool(ConstMap::default());
        let asm_info = result?;
        // compile.py record_loop_or_bridge: clt.loop_token_wref = wref.
        // Concrete backends attach ResumeDescr.rd_loop_token during assembly;
        // complete that existing ownership chain without a metainterp registry.
        token
            .compiled_loop_token_expect()
            .set_loop_token_wref(Arc::downgrade(&token));
        backend.track_compiled_token(Arc::clone(&token));
        Ok(Self {
            backend,
            token,
            input_types,
            asm_info,
            _thread_affinity: std::marker::PhantomData,
        })
    }

    /// The signature in argument-list order, not IR register-number order.
    pub fn input_types(&self) -> &[Type] {
        &self.input_types
    }

    /// Backend-reported code metadata; not a portable callable function pointer.
    pub fn asm_info(&self) -> &AsmInfo {
        &self.asm_info
    }

    /// Execute once, returning the backend's completion, guard or exception exit.
    ///
    /// Arity/type mismatches are rejected before entering machine code.
    /// A guard exit remains an exit (`is_finish == false`), not a successful
    /// language result: resumption or bridge compilation belongs to the caller.
    /// There is no hidden warmup, retracing or fallback.
    ///
    /// # Safety
    ///
    /// The compilation contract must still hold. Pointer-valued arguments
    /// (including pointers encoded as integers) must be valid for the IR, and
    /// the embedding must maintain its runtime/GC roots on this thread.
    /// References returned in `RawExecResult` are not independently rooted:
    /// root them before another collecting operation or runtime teardown.
    pub unsafe fn execute(&mut self, args: &[Value]) -> Result<RawExecResult, ArgumentError> {
        if args.len() != self.input_types.len() {
            return Err(ArgumentError::Arity {
                expected: self.input_types.len(),
                actual: args.len(),
            });
        }
        for (index, (arg, &expected)) in args.iter().zip(&self.input_types).enumerate() {
            let actual = arg.get_type();
            if actual != expected {
                return Err(ArgumentError::Type {
                    index,
                    expected,
                    actual,
                });
            }
        }
        Ok(self.backend.execute_token_raw(&self.token, args))
    }
}

/// Invalid arguments rejected before compiled entry.
#[derive(Debug, PartialEq, Eq)]
pub enum ArgumentError {
    Arity {
        expected: usize,
        actual: usize,
    },
    Type {
        index: usize,
        expected: Type,
        actual: Type,
    },
}

impl std::fmt::Display for ArgumentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Arity { expected, actual } => {
                write!(f, "expected {expected} arguments, got {actual}")
            }
            Self::Type {
                index,
                expected,
                actual,
            } => {
                write!(f, "argument {index}: expected {expected:?}, got {actual:?}")
            }
        }
    }
}

impl std::error::Error for ArgumentError {}
