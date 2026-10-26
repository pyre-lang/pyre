//! Port of `rpython/rtyper/llinterp.py`.
//!
//! Upstream is 1477 LOC of two top-level types — `LLException` /
//! `LLFatalError` exceptions, plus the `LLInterpreter` and its
//! `LLFrame` opcode-dispatch loop. Driver task `task_llinterpret_lltype`
//! (`driver.py`) is the only consumer of `LLInterpreter` and
//! uses only:
//!
//! * Constructor: `LLInterpreter(translator.rtyper)` (`:67-82`).
//! * `eval_graph(graph, args)` (`:84-…`): walks a `FunctionGraph`,
//!   running each `SpaceOperation` through `op_<opname>` handlers.
//!
//! The full interpreter requires `rpython.rtyper.lltypesystem.{lltype,
//! llmemory, lloperation, llheap}` plus the full `LLFrame` opcode table.
//! This file starts the real leaf port with the upstream `LLFrame`
//! execution skeleton and a small straight-line operation subset.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::LazyLock;

use crate::annotator::annrpython::RPythonAnnotator;
use crate::flowspace::model::{
    BlockRef, ConstValue, Constant, FunctionGraph, GraphRef, Hlvalue, HostObject, LinkRef,
    SpaceOperation, Variable,
};
use crate::tool::ansi_print::AnsiLogger;
use crate::translator::backendopt::constfold::WE_ARE_JITTED_TAG_ID;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::llmemory;
use crate::translator::rtyper::lltypesystem::lltype::{
    _address, _func, _ptr, _ptr_obj, LowLevelType, LowLevelValue, MallocFlavor, PtrTarget, Struct,
    identityhash, isCompatibleType, malloc, typeOf,
};
use crate::translator::rtyper::rtyper::RPythonTyper;
use crate::translator::tool::taskengine::TaskError;

/// Upstream `log = AnsiLogger('llinterp')`; tests usually leave output disabled.
pub static log: LazyLock<AnsiLogger> = LazyLock::new(|| {
    let logger = AnsiLogger::new("llinterp");
    logger.set_output_disabled(true);
    logger
});

/// Port of upstream `class LLException(Exception)` at `:28-54`.
///
/// `error_value` defaults to upstream's `UNDEFINED_ERROR_VALUE`
/// sentinel — the local port encodes the sentinel as `None`.
#[derive(Debug, Clone)]
pub struct LLException {
    pub args: Vec<Rc<dyn Any>>,
    pub error_value: Option<Rc<dyn Any>>,
}

impl LLException {
    /// Upstream `__init__(*args, error_value=UNDEFINED_ERROR_VALUE)` at
    /// `:35-40`.
    pub fn new(args: Vec<Rc<dyn Any>>, error_value: Option<Rc<dyn Any>>) -> Self {
        Self { args, error_value }
    }
}

impl std::fmt::Display for LLException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<LLException>")
    }
}

impl std::error::Error for LLException {}

/// Port of upstream `class LLFatalError(Exception)` at `:56-58`.
#[derive(Debug, Clone)]
pub struct LLFatalError(pub Vec<String>);

impl std::fmt::Display for LLFatalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.join(": "))
    }
}

impl std::error::Error for LLFatalError {}

/// Port of upstream `class LLAssertFailure(Exception)` at `:60-61`.
#[derive(Debug, Clone)]
pub struct LLAssertFailure;

impl std::fmt::Display for LLAssertFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LLAssertFailure")
    }
}

impl std::error::Error for LLAssertFailure {}

/// Port of upstream `class PleaseOverwriteStoreException(Exception)`.
#[derive(Debug, Clone)]
pub struct PleaseOverwriteStoreException(pub String);

impl std::fmt::Display for PleaseOverwriteStoreException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PleaseOverwriteStoreException {}

/// RPython `checkptr(ptr)` (`llinterp.py`).
pub fn checkptr(ptr: &LowLevelValue) -> Result<(), TyperError> {
    match ptr {
        LowLevelValue::Ptr(_) => Ok(()),
        other => Err(TyperError::message(format!(
            "llinterp.checkptr: expected Ptr, got {other:?}"
        ))),
    }
}

/// RPython `checkadr(addr)` (`llinterp.py`).
pub fn checkadr(addr: &LowLevelValue) -> Result<(), TyperError> {
    match addr {
        LowLevelValue::Address(_) => Ok(()),
        other => Err(TyperError::message(format!(
            "llinterp.checkadr: expected Address, got {other:?}"
        ))),
    }
}

/// Concrete low-level value used by the local `LLFrame` port.
///
/// Upstream `LLFrame.bindings` stores arbitrary concrete Python objects.
/// Rust does not have that object model, so this private enum carries
/// the primitive shapes currently produced by local lltyped graphs.
#[derive(Clone, Debug, PartialEq)]
enum LLValue {
    Void,
    Int(i64),
    Bool(bool),
    Float(u64),
    Str(String),
    Ptr(Box<_ptr>),
    Address(_address),
    Const(ConstValue),
}

impl LLValue {
    fn into_any(self) -> Rc<dyn Any> {
        match self {
            LLValue::Void => Rc::new(()) as Rc<dyn Any>,
            LLValue::Int(n) => Rc::new(n) as Rc<dyn Any>,
            LLValue::Bool(b) => Rc::new(b) as Rc<dyn Any>,
            LLValue::Float(bits) => Rc::new(f64::from_bits(bits)) as Rc<dyn Any>,
            LLValue::Str(s) => Rc::new(s) as Rc<dyn Any>,
            LLValue::Ptr(p) => Rc::new(LowLevelValue::Ptr(p)) as Rc<dyn Any>,
            LLValue::Address(a) => Rc::new(LowLevelValue::Address(a)) as Rc<dyn Any>,
            LLValue::Const(c) => Rc::new(c) as Rc<dyn Any>,
        }
    }

    fn as_i64(&self, opname: &str) -> Result<i64, TaskError> {
        match self {
            LLValue::Int(n) => Ok(*n),
            other => Err(TaskError {
                message: format!("llinterp.py:405 {opname}: expected Signed, got {other:?}"),
            }),
        }
    }

    fn as_bool(&self, opname: &str) -> Result<bool, TaskError> {
        match self {
            LLValue::Bool(b) => Ok(*b),
            other => Err(TaskError {
                message: format!("llinterp.py:405 {opname}: expected Bool, got {other:?}"),
            }),
        }
    }

    fn as_f64(&self, opname: &str) -> Result<f64, TaskError> {
        match self {
            LLValue::Float(bits) => Ok(f64::from_bits(*bits)),
            other => Err(TaskError {
                message: format!(
                    "llinterp.py LLFrame.getval {opname}: expected Float, got {other:?}"
                ),
            }),
        }
    }

    fn as_ptr(&self, opname: &str) -> Result<&_ptr, TaskError> {
        match self {
            LLValue::Ptr(p) => Ok(p.as_ref()),
            other => Err(TaskError {
                message: format!(
                    "llinterp.py LLFrame.getval {opname}: expected Ptr, got {other:?}"
                ),
            }),
        }
    }

    fn as_address(&self, opname: &str) -> Result<&_address, TaskError> {
        match self {
            LLValue::Address(a) => Ok(a),
            other => Err(TaskError {
                message: format!(
                    "llinterp.py LLFrame.getval {opname}: expected Address, got {other:?}"
                ),
            }),
        }
    }

    fn truth(&self) -> bool {
        match self {
            LLValue::Void => false,
            LLValue::Int(n) => *n != 0,
            LLValue::Bool(b) => *b,
            LLValue::Float(bits) => f64::from_bits(*bits) != 0.0,
            LLValue::Str(s) => !s.is_empty(),
            LLValue::Ptr(p) => p.nonzero(),
            LLValue::Address(a) => a.nonzero(),
            LLValue::Const(ConstValue::None) => false,
            LLValue::Const(_) => true,
        }
    }
}

fn const_to_llvalue(c: &Constant) -> LLValue {
    match &c.value {
        ConstValue::Int(n) => LLValue::Int(*n),
        ConstValue::UInt(n) => LLValue::Int(*n as i64),
        ConstValue::Bool(b) => LLValue::Bool(*b),
        ConstValue::Float(bits) => LLValue::Float(*bits),
        // Item 2 string split: ByteStr is the Python 2 `str` shape and
        // is the only string carrier upstream `LLInterpreter` produces
        // here. UniStr falls through to the Const carrier so callers
        // see the type-mismatch shape rather than a silent rewrap.
        ConstValue::ByteStr(s) => LLValue::Str(String::from_utf8_lossy(s).into_owned()),
        ConstValue::None => LLValue::Void,
        ConstValue::LLPtr(p) => LLValue::Ptr(p.clone()),
        ConstValue::LLAddress(a) => LLValue::Address(a.clone()),
        // `_we_are_jitted` is `CDefinedIntSymbolic(default=0)`;
        // the llinterp runs the non-jitted arm.
        ConstValue::SpecTag(tag) if *tag == WE_ARE_JITTED_TAG_ID => match &c.concretetype {
            Some(LowLevelType::Bool) => LLValue::Bool(false),
            _ => LLValue::Int(0),
        },
        other => LLValue::Const(other.clone()),
    }
}

/// HostObject class carried on an etype / `llexitcase` Constant.
fn host_class_object(value: &ConstValue) -> Option<&HostObject> {
    match value {
        ConstValue::HostObject(obj) if obj.is_class() => Some(obj),
        _ => None,
    }
}

/// Inverse of the `LowLevelValue` arm of [`any_to_llvalue`]: convert an
/// `LLValue` into the `lltype::LowLevelValue` a heap cell stores. A Char is
/// carried as a one-char `Str` (see the `LowLevelValue::Char` arm below).
fn llvalue_to_lowlevel(value: &LLValue, opname: &str) -> Result<LowLevelValue, TaskError> {
    match value {
        LLValue::Void => Ok(LowLevelValue::Void),
        LLValue::Int(n) => Ok(LowLevelValue::Signed(*n)),
        LLValue::Bool(b) => Ok(LowLevelValue::Bool(*b)),
        LLValue::Float(bits) => Ok(LowLevelValue::Float(*bits)),
        LLValue::Ptr(p) => Ok(LowLevelValue::Ptr(p.clone())),
        LLValue::Address(a) => Ok(LowLevelValue::Address(a.clone())),
        LLValue::Str(s) if s.chars().count() == 1 => {
            Ok(LowLevelValue::Char(s.chars().next().expect("one char")))
        }
        other => Err(TaskError {
            message: format!("llinterp.py:405 {opname}: cannot store {other:?} into a heap cell"),
        }),
    }
}

/// Convert `value` to a `LowLevelValue` of the function's `RESULT` type
/// (`llinterp.py` `perform_call` returns the graph's own typed value).
fn llvalue_to_lowlevel_of_type(
    value: &LLValue,
    RESULT: &LowLevelType,
    opname: &str,
) -> Result<LowLevelValue, TaskError> {
    match (RESULT, value) {
        (LowLevelType::Unsigned, LLValue::Int(n)) => Ok(LowLevelValue::Unsigned(*n as u64)),
        (LowLevelType::UniChar, LLValue::Str(s)) if s.chars().count() == 1 => {
            Ok(LowLevelValue::UniChar(s.chars().next().expect("one char")))
        }
        (LowLevelType::UniChar, LLValue::Int(n)) => {
            let c = char::from_u32(*n as u32).ok_or_else(|| TaskError {
                message: format!("{opname}: {n} is not a valid UniChar"),
            })?;
            Ok(LowLevelValue::UniChar(c))
        }
        (LowLevelType::Char, LLValue::Str(s)) if s.chars().count() == 1 => {
            Ok(LowLevelValue::Char(s.chars().next().expect("one char")))
        }
        (LowLevelType::Char, LLValue::Int(n)) => {
            let c = char::from_u32(*n as u32).ok_or_else(|| TaskError {
                message: format!("{opname}: {n} is not a valid Char"),
            })?;
            Ok(LowLevelValue::Char(c))
        }
        (LowLevelType::SingleFloat, LLValue::Float(bits)) => Ok(LowLevelValue::SingleFloat(
            (f64::from_bits(*bits) as f32).to_bits(),
        )),
        (LowLevelType::LongFloat, LLValue::Float(bits)) => Ok(LowLevelValue::LongFloat(*bits)),
        _ => llvalue_to_lowlevel(value, opname),
    }
}

fn any_to_llvalue(value: &Rc<dyn Any>) -> Result<LLValue, TaskError> {
    if let Some(v) = value.downcast_ref::<LLValue>() {
        return Ok(v.clone());
    }
    if let Some(v) = value.downcast_ref::<ConstValue>() {
        return Ok(match v {
            ConstValue::Int(n) => LLValue::Int(*n),
            ConstValue::UInt(n) => LLValue::Int(*n as i64),
            ConstValue::Bool(b) => LLValue::Bool(*b),
            ConstValue::Float(bits) => LLValue::Float(*bits),
            ConstValue::ByteStr(s) => LLValue::Str(String::from_utf8_lossy(s).into_owned()),
            ConstValue::None => LLValue::Void,
            ConstValue::LLPtr(p) => LLValue::Ptr(p.clone()),
            ConstValue::LLAddress(a) => LLValue::Address(a.clone()),
            other => LLValue::Const(other.clone()),
        });
    }
    if let Some(v) = value.downcast_ref::<LowLevelValue>() {
        return match v {
            LowLevelValue::Void => Ok(LLValue::Void),
            LowLevelValue::Signed(n) => Ok(LLValue::Int(*n)),
            LowLevelValue::Unsigned(n) => Ok(LLValue::Int(*n as i64)),
            LowLevelValue::Bool(b) => Ok(LLValue::Bool(*b)),
            LowLevelValue::Float(bits) => Ok(LLValue::Float(*bits)),
            LowLevelValue::SingleFloat(bits) => Ok(LLValue::Float(*bits as u64)),
            LowLevelValue::LongFloat(bits) => Ok(LLValue::Float(*bits)),
            LowLevelValue::Char(c) | LowLevelValue::UniChar(c) => Ok(LLValue::Str(c.to_string())),
            LowLevelValue::Address(a) => Ok(LLValue::Address(a.clone())),
            LowLevelValue::Ptr(p) => Ok(LLValue::Ptr(p.clone())),
            LowLevelValue::InteriorPtr(ip) => {
                let p = ip._as_container_ptr(true).map_err(|e| TaskError {
                    message: format!("llinterp.py LLInterpreter.eval_graph InteriorPtr: {e}"),
                })?;
                Ok(LLValue::Ptr(Box::new(p)))
            }
            other => Err(TaskError {
                message: format!(
                    "llinterp.py:84 LLInterpreter.eval_graph — unsupported LowLevelValue {other:?}"
                ),
            }),
        };
    }
    if let Some(v) = value.downcast_ref::<i64>() {
        return Ok(LLValue::Int(*v));
    }
    if let Some(v) = value.downcast_ref::<bool>() {
        return Ok(LLValue::Bool(*v));
    }
    if let Some(v) = value.downcast_ref::<f64>() {
        return Ok(LLValue::Float(v.to_bits()));
    }
    if let Some(v) = value.downcast_ref::<String>() {
        return Ok(LLValue::Str(v.clone()));
    }
    if value.downcast_ref::<()>().is_some() {
        return Ok(LLValue::Void);
    }
    Err(TaskError {
        message: "llinterp.py:84 LLInterpreter.eval_graph — unsupported concrete argument type"
            .to_string(),
    })
}

fn field_name(val: &LLValue, opname: &str) -> Result<String, TaskError> {
    match val {
        LLValue::Str(s) => Ok(s.clone()),
        LLValue::Const(ConstValue::ByteStr(s)) => Ok(String::from_utf8_lossy(s).into_owned()),
        LLValue::Const(ConstValue::UniStr(s)) => Ok(s.clone()),
        other => Err(TaskError {
            message: format!("{opname}: expected a field name, got {other:?}"),
        }),
    }
}

/// Host Python shifts (`opimpl.py` `get_primitive_op_src` `int_lshift` /
/// `op_int_rshift` / `op_uint_lshift` / `op_uint_rshift`): `y < 0` raises
/// `ValueError("negative shift count")`. For `y >= 64`, lshift is 0 (low 64
/// bits of a big int); signed rshift is `-1` if `x < 0` else `0`; unsigned
/// rshift is 0.
fn signed_lshift(x: i64, y: i64) -> Result<i64, TaskError> {
    if y < 0 {
        Err(TaskError {
            message: "ValueError: negative shift count".to_string(),
        })
    } else if y >= 64 {
        Ok(0)
    } else {
        Ok(x.wrapping_shl(y as u32))
    }
}

fn signed_rshift(x: i64, y: i64) -> Result<i64, TaskError> {
    if y < 0 {
        Err(TaskError {
            message: "ValueError: negative shift count".to_string(),
        })
    } else if y >= 64 {
        Ok(if x < 0 { -1 } else { 0 })
    } else {
        Ok(x >> y)
    }
}

fn unsigned_lshift(x: u64, y: i64) -> Result<u64, TaskError> {
    if y < 0 {
        Err(TaskError {
            message: "ValueError: negative shift count".to_string(),
        })
    } else if y >= 64 {
        Ok(0)
    } else {
        Ok(x.wrapping_shl(y as u32))
    }
}

fn unsigned_rshift(x: u64, y: i64) -> Result<u64, TaskError> {
    if y < 0 {
        Err(TaskError {
            message: "ValueError: negative shift count".to_string(),
        })
    } else if y >= 64 {
        Ok(0)
    } else {
        Ok(x >> y)
    }
}

/// `opimpl.py` `op_cast_float_to_int(f)`: `intmask(int(f))`. `int(inf)`
/// raises `OverflowError`, `int(nan)` raises `ValueError`; finite values
/// truncate toward zero then wrap to the low 64 bits.
fn op_cast_float_to_int(f: f64) -> Result<i64, TaskError> {
    if f.is_infinite() {
        return Err(TaskError {
            message: "OverflowError: cannot convert float infinity to integer".to_string(),
        });
    }
    if f.is_nan() {
        return Err(TaskError {
            message: "ValueError: cannot convert float NaN to integer".to_string(),
        });
    }
    let t = f.trunc();
    if t.abs() < 2f64.powi(127) {
        Ok((t as i128) as i64)
    } else {
        Ok(0)
    }
}

fn lowlevel_to_llvalue(value: LowLevelValue, opname: &str) -> Result<LLValue, TaskError> {
    any_to_llvalue(&(Rc::new(value) as Rc<dyn Any>)).map_err(|e| TaskError {
        message: format!("{opname}: {}", e.message),
    })
}

fn dummy_interpreter() -> Rc<LLInterpreter> {
    let ann = RPythonAnnotator::new(None, None, None, false);
    let typer = Rc::new(RPythonTyper::new(&ann));
    Rc::new(LLInterpreter::new(typer, false, None))
}

/// `LLFrame.perform_call` starts `fobj = f._obj` (`_ptr._obj` →
/// `_getobj(check=True)`). Map a delayed-pointer error into `TaskError`.
fn func_from_ptr(p: &_ptr, opname: &str) -> Result<_func, TaskError> {
    match p._obj() {
        Ok(_ptr_obj::Func(f)) => Ok(f),
        Ok(other) => Err(TaskError {
            message: format!("{opname}: expected a function pointer, got {other:?}"),
        }),
        Err(_) => Err(TaskError {
            message: format!("{opname}: DelayedPointer"),
        }),
    }
}

/// `ll_math.py ll_math_frexp`: `if not isfinite(x) or not x: return (x, 0)`,
/// otherwise mantissa in `[0.5, 1)` (or `(-1, -0.5]`) with
/// `f == mant * 2**exp`. Subnormals start the unbiased exponent at
/// `-1021` so the smallest positive subnormal is `(0.5, -1073)`.
pub(crate) fn python_frexp(f: f64) -> (f64, i32) {
    if !f.is_finite() || f == 0.0 {
        return (f, 0);
    }
    let bits = f.to_bits();
    let sign = bits & (1u64 << 63);
    let exp_raw = ((bits >> 52) & 0x7ff) as i32;
    let mut frac = bits & ((1u64 << 52) - 1);
    if exp_raw == 0 {
        let mut e = -1021;
        while frac < (1u64 << 52) {
            frac <<= 1;
            e -= 1;
        }
        frac &= (1u64 << 52) - 1;
        let mant = f64::from_bits(sign | (1022u64 << 52) | frac);
        (mant, e)
    } else {
        let mant = f64::from_bits(sign | (1022u64 << 52) | frac);
        (mant, exp_raw - 1022)
    }
}

const TAKE_NEXT: f64 = 2147483648.0; // 2**31

/// `objectmodel._hash_float`.
pub(crate) fn hash_float_value(f: f64) -> i64 {
    if !f.is_finite() {
        if f.is_infinite() {
            if f < 0.0 { -271828 } else { 314159 }
        } else {
            0
        }
    } else if f == 0.0 {
        0
    } else {
        let (mut v, expo) = python_frexp(f);
        v *= TAKE_NEXT;
        let hipart = v as i64;
        v = (v - hipart as f64) * TAKE_NEXT;
        hipart + (v as i64) + ((expo as i64) << 15)
    }
}

/// `_hash_string` over UniChar ords, then `_ll_strhash`'s `0 → 29872897`
/// fixup (`LLHelpers.ll_strhash` / `objectmodel._hash_string`).
pub(crate) fn hash_unichar_ords(chars: &[char]) -> i64 {
    let Some(&first) = chars.first() else {
        return -1;
    };
    let mut x = (first as i64) << 7;
    for &c in chars {
        x = 1000003i64.wrapping_mul(x) ^ (c as i64);
    }
    x ^= chars.len() as i64;
    if x == 0 { 29872897 } else { x }
}

/// RPython `TUPLE_TYPE([Float, Signed])` (`rtuple.py`): `Ptr(GcStruct(
/// 'tuple2', ('item0', Float), ('item1', Signed), hints={immutable, noidentity}))`.
fn malloc_frexp_tuple(mant: f64, exp: i32) -> Result<_ptr, TaskError> {
    let body = Struct::gc_with_hints(
        "tuple2",
        vec![
            ("item0".into(), LowLevelType::Float),
            ("item1".into(), LowLevelType::Signed),
        ],
        vec![
            ("immutable".into(), ConstValue::Bool(true)),
            ("noidentity".into(), ConstValue::Bool(true)),
        ],
    );
    let mut p = malloc(
        LowLevelType::Struct(Box::new(body)),
        None,
        MallocFlavor::Gc,
        true,
    )
    .map_err(|e| TaskError { message: e })?;
    p.setattr("item0", LowLevelValue::Float(mant.to_bits()))
        .map_err(|e| TaskError { message: e })?;
    p.setattr("item1", LowLevelValue::Signed(exp as i64))
        .map_err(|e| TaskError { message: e })?;
    Ok(p)
}

fn hlvalue_concretetype(v: &Hlvalue) -> Option<LowLevelType> {
    match v {
        Hlvalue::Variable(var) => var.concretetype(),
        Hlvalue::Constant(c) => c.concretetype.clone(),
    }
}

/// `LLFrame.perform_call` (`llinterp.py`): `ARGS = typeOf(f).TO.ARGS`
/// must match `graph.getargs()` arity and `isCompatibleType` per slot.
fn validate_graph_funcptr_type(graph: &GraphRef, ARGS: &[LowLevelType]) -> Result<(), TaskError> {
    let graph_b = graph.borrow();
    let args_v = graph_b.getargs();
    if ARGS.len() != args_v.len() {
        return Err(TaskError {
            message: format!(
                "graph with {} args called with wrong func ptr type: {:?}",
                args_v.len(),
                ARGS
            ),
        });
    }
    for (T, v) in ARGS.iter().zip(args_v.iter()) {
        let Some(ct) = hlvalue_concretetype(v) else {
            let expected: Vec<_> = args_v.iter().map(hlvalue_concretetype).collect();
            return Err(TaskError {
                message: format!(
                    "graph with {:?} args called with wrong func ptr type: {:?}",
                    expected, ARGS
                ),
            });
        };
        if !isCompatibleType(T, &ct) {
            let expected: Vec<_> = args_v.iter().map(hlvalue_concretetype).collect();
            return Err(TaskError {
                message: format!(
                    "graph with {:?} args called with wrong func ptr type: {:?}",
                    expected, ARGS
                ),
            });
        }
    }
    Ok(())
}

fn funcptr_ARGS(p: &_ptr) -> Result<Vec<LowLevelType>, TaskError> {
    match &typeOf(p).TO {
        PtrTarget::Func(ft) => Ok(ft.args.clone()),
        other => Err(TaskError {
            message: format!("direct_call: expected a function pointer type, got {other:?}"),
        }),
    }
}

fn funcptr_RESULT(p: &_ptr) -> Result<LowLevelType, TaskError> {
    match &typeOf(p).TO {
        PtrTarget::Func(ft) => Ok(ft.result.clone()),
        other => Err(TaskError {
            message: format!("direct_call: expected a function pointer type, got {other:?}"),
        }),
    }
}

/// `fobj._callable(*args)` for a graphless external (`llexternal` /
/// external function with an `llimpl`). A low-level helper always carries
/// its graph and runs through `perform_call`'s graph arm.
fn invoke_callable(name: &str, args: &[LLValue]) -> Result<LLValue, TaskError> {
    let ll_args: Vec<LowLevelValue> = args
        .iter()
        .map(|a| llvalue_to_lowlevel(a, name))
        .collect::<Result<_, _>>()?;
    match name {
        "math.frexp" => {
            let LowLevelValue::Float(bits) = ll_args.first().ok_or_else(|| TaskError {
                message: "math.frexp: missing argument".to_string(),
            })?
            else {
                return Err(TaskError {
                    message: "math.frexp: expected a Float".to_string(),
                });
            };
            let (mant, exp) = python_frexp(f64::from_bits(*bits));
            Ok(LLValue::Ptr(Box::new(malloc_frexp_tuple(mant, exp)?)))
        }
        "frexp" => {
            // C `math_frexp(x, exp_p)` (`ll_math.py`): write the exponent
            // through the INTP raw array and return the mantissa.
            let LowLevelValue::Float(bits) = ll_args.first().ok_or_else(|| TaskError {
                message: "frexp: missing argument".to_string(),
            })?
            else {
                return Err(TaskError {
                    message: "frexp: expected a Float".to_string(),
                });
            };
            let LowLevelValue::Ptr(exp_p) = ll_args.get(1).ok_or_else(|| TaskError {
                message: "frexp: missing exponent pointer".to_string(),
            })?
            else {
                return Err(TaskError {
                    message: "frexp: expected an INTP exponent pointer".to_string(),
                });
            };
            let (mant, exp) = python_frexp(f64::from_bits(*bits));
            let mut exp_p = (**exp_p).clone();
            exp_p
                .setitem(0, LowLevelValue::Signed(exp as i64))
                .map_err(|e| TaskError { message: e })?;
            Ok(LLValue::Float(mant.to_bits()))
        }
        other => Err(TaskError {
            message: format!(
                "llinterp.py LLFrame.perform_call — graphless funcptr {other} has no _callable table entry"
            ),
        }),
    }
}

fn eval_graph_on_funcptr(
    graph: GraphRef,
    args: &[LowLevelValue],
    RESULT: &LowLevelType,
) -> Result<LowLevelValue, TaskError> {
    let ll_args: Vec<LLValue> = args
        .iter()
        .cloned()
        .map(|v| lowlevel_to_llvalue(v, "eval_funcptr"))
        .collect::<Result<_, _>>()?;
    if let Some(interp) = LLInterpreter::current_interpreter() {
        let frame = Rc::new(RefCell::new(LLFrame::new(graph, ll_args)));
        let result = frame.borrow_mut().eval(&interp, frame.clone())?;
        llvalue_to_lowlevel_of_type(&result, RESULT, "eval_funcptr")
    } else {
        let interp = dummy_interpreter();
        let any_args: Vec<Rc<dyn Any>> = args
            .iter()
            .cloned()
            .map(|v| Rc::new(v) as Rc<dyn Any>)
            .collect();
        let result = interp.eval_graph(graph as Rc<dyn Any>, any_args, false)?;
        let ll = any_to_llvalue(&result)?;
        llvalue_to_lowlevel_of_type(&ll, RESULT, "eval_funcptr")
    }
}

/// Run a stored adtmeth funcptr (`llinterp.py` `op_direct_call` /
/// `perform_call`): a graph goes through `LLInterpreter.eval_graph` (or
/// a subframe when already interpreting); a graphless funcptr goes
/// through its `_callable` name.
pub fn eval_funcptr(funcptr: &_ptr, args: &[LowLevelValue]) -> Result<LowLevelValue, TaskError> {
    let func = func_from_ptr(funcptr, "eval_funcptr")?;
    let RESULT = funcptr_RESULT(funcptr)?;
    if let Some(graph) = func.graph_ref() {
        let ARGS = funcptr_ARGS(funcptr)?;
        validate_graph_funcptr_type(&graph, &ARGS)?;
        eval_graph_on_funcptr(graph, args, &RESULT)
    } else if let Some(name) = &func._callable {
        let ll_args: Vec<LLValue> = args
            .iter()
            .cloned()
            .map(|v| lowlevel_to_llvalue(v, name))
            .collect::<Result<_, _>>()?;
        let result = invoke_callable(name, &ll_args)?;
        llvalue_to_lowlevel_of_type(&result, &RESULT, name)
    } else {
        Err(TaskError {
            message: format!(
                "llinterp.py LLFrame.perform_call — don't know how to execute {}",
                func._name
            ),
        })
    }
}

/// Port of upstream `class LLInterpreter(object)` at `:67-…`.
///
/// Slot layout matches upstream's `__init__` (`:72-82`). The first
/// executable slice covers `eval_graph` (`:84-126`) and the straight-line
/// parts of `LLFrame` (`:214-309`, `:405-525`).
pub struct LLInterpreter {
    /// Upstream `self.bindings = {}` at `:73`. Maps each Variable to
    /// its concrete value during a frame's `eval()`. Stored as
    /// `Vec<(Rc<dyn Any>, Rc<dyn Any>)>` because the bindings carry
    /// heterogeneous (Variable, [`LLValue`]) pairs that callers
    /// downcast at the read site. lltype is now ported under
    /// [`crate::translator::rtyper::lltypesystem::lltype`]; a
    /// follow-up could narrow the value slot to
    /// `lltype::LowLevelValue` once every read site is migrated
    /// off `Rc<dyn Any>::downcast_ref`.
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    pub bindings: RefCell<Vec<(Rc<dyn Any>, Rc<dyn Any>)>>,
    /// Upstream `self.typer = typer` at `:74`.
    pub typer: Rc<RPythonTyper>,
    /// Upstream `self.heap = llheap` at `:76`. Upstream points at the
    /// `rpython.rtyper.lltypesystem.llheap` module (the `malloc`/`free`
    /// surface). Pyre exposes that facade in
    /// [`crate::translator::rtyper::lltypesystem::llheap`], but this slot
    /// remains opaque until LLFrame opcode handlers need to dereference it.
    pub heap: Option<Rc<dyn Any>>,
    /// Upstream `self.exc_data_ptr = exc_data_ptr` at `:77`.
    pub exc_data_ptr: Option<Rc<dyn Any>>,
    /// Upstream `self.frame_stack = []` at `:78`.
    pub frame_stack: RefCell<Vec<Rc<RefCell<LLFrame>>>>,
    /// Upstream `self.tracer = None` (or `Tracer()`) at `:79-82`.
    pub tracer: Option<Rc<dyn Any>>,
    /// Upstream `self.frame_class = LLFrame` at `:80`. Upstream stores
    /// the class so subclasses can override; the Rust port has only
    /// the one [`LLFrame`] type, so the slot is documented but unused.
    pub frame_class: std::marker::PhantomData<LLFrame>,
    /// Upstream `self.traceback_frames = []` initialized at the top of
    /// `eval_graph` (`:91`). The local port carries the slot to keep
    /// the upstream shape; it is reset on each `eval_graph` entry.
    pub traceback_frames: RefCell<Vec<Rc<RefCell<LLFrame>>>>,
}

// Upstream module-level `LLInterpreter.current_interpreter = None`
// (`llinterp.py`) plus its mutation in `eval_graph` (`:99`):
// `LLInterpreter.current_interpreter = self`. Carried as a thread-local
// `Weak` pointer so the LLFrame can `current_interpreter()` without
// receiver plumbing, matching upstream's class-attribute lookup.
thread_local! {
    static CURRENT_INTERPRETER: std::cell::RefCell<std::rc::Weak<LLInterpreter>>
        = const { std::cell::RefCell::new(std::rc::Weak::new()) };
}

impl LLInterpreter {
    /// Upstream `LLInterpreter.current_interpreter` class-level slot
    /// (`llinterp.py`) read by `LLFrame` opcode handlers via
    /// `LLInterpreter.current_interpreter`.
    pub fn current_interpreter() -> Option<Rc<LLInterpreter>> {
        CURRENT_INTERPRETER.with(|cell| cell.borrow().upgrade())
    }

    /// Upstream `__init__(self, typer, tracing=True, exc_data_ptr=None)`
    /// at `:72-82`.
    pub fn new(typer: Rc<RPythonTyper>, tracing: bool, exc_data_ptr: Option<Rc<dyn Any>>) -> Self {
        // Upstream `:80-82`: `if tracing: self.tracer = Tracer()` else
        // `self.tracer = None`. The local `Tracer` port is not landed,
        // so the slot stays `None` regardless. The `tracing` arg is
        // accepted for signature parity but not yet observable —
        // documented TODO until `Tracer` lands.
        let _ = tracing;
        Self {
            bindings: RefCell::new(Vec::new()),
            typer,
            // Upstream `:76`: `self.heap = llheap`. The module facade exists,
            // but no local LLFrame path reads this slot yet.
            heap: None,
            exc_data_ptr,
            frame_stack: RefCell::new(Vec::new()),
            tracer: None,
            frame_class: std::marker::PhantomData,
            traceback_frames: RefCell::new(Vec::new()),
        }
    }

    /// Upstream `eval_graph(self, graph, args=(), recursive=False)` at
    /// `llinterp.py`.
    ///
    /// Mirrors upstream `:99` `LLInterpreter.current_interpreter = self`
    /// unconditionally — every public entry installs the thread-local
    /// pointer that `LLFrame` opcode handlers consult, then restores
    /// the previous value on exit (handling re-entrant evaluation).
    /// Takes `self: &Rc<Self>` because installing a `Weak<Self>` into
    /// the thread-local requires upgrading from a `Rc`, matching
    /// upstream's Python-object identity.
    pub fn eval_graph(
        self: &Rc<Self>,
        graph: Rc<dyn Any>,
        args: Vec<Rc<dyn Any>>,
        _recursive: bool,
    ) -> Result<Rc<dyn Any>, TaskError> {
        // Upstream `llframe = self.frame_class(graph, args, self)` at
        // `llinterp.py:85`. The local public API still accepts
        // `Rc<dyn Any>` because driver plumbing is not typed yet; the
        // executable leaf accepts the orthodox `GraphRef`.
        let graph = graph
            .downcast::<RefCell<FunctionGraph>>()
            .map_err(|_| TaskError {
                message:
                    "llinterp.py:84 LLInterpreter.eval_graph — expected flowspace::model::GraphRef"
                        .to_string(),
            })?;
        let args: Vec<LLValue> = args.iter().map(any_to_llvalue).collect::<Result<_, _>>()?;
        let llframe = Rc::new(RefCell::new(LLFrame::new(graph, args)));
        // Upstream `:91`: `self.traceback_frames = []`. Reset the slot
        // at every `eval_graph` entry (top-level call resets the stack).
        self.traceback_frames.borrow_mut().clear();
        // Upstream `:99`: `LLInterpreter.current_interpreter = self`.
        // Save the prior value so re-entrant `eval_graph` (recursive
        // calls into another graph) restores correctly on return.
        let prior = CURRENT_INTERPRETER.with(|cell| {
            let old = cell.borrow().clone();
            *cell.borrow_mut() = Rc::downgrade(self);
            old
        });
        let old_depth = self.frame_stack.borrow().len();
        let result = llframe.borrow_mut().eval(self, llframe.clone());
        debug_assert_eq!(self.frame_stack.borrow().len(), old_depth);
        CURRENT_INTERPRETER.with(|cell| {
            *cell.borrow_mut() = prior;
        });
        Ok(result?.into_any())
    }
}

/// Port of upstream `class LLFrame(object)` at `llinterp.py-…`.
pub struct LLFrame {
    pub graph: GraphRef,
    args: Vec<LLValue>,
    /// Upstream `self.bindings = {}` at `:221`.
    bindings: HashMap<Variable, LLValue>,
    pub curr_block: Option<BlockRef>,
    pub curr_operation_index: usize,
    pub alloca_objects: Vec<Rc<dyn Any>>,
    /// Upstream catches `LLException as e` at `:323` after an op
    /// raised; `e.args[0]` is the exception class and `e.args[1]` the
    /// instance (`llinterp.py`). The Rust port captures the
    /// same `(etype, evalue)` pair here as typed `LLValue`s; the next
    /// `eval_block` step consumes it via the `canraise` dispatch.
    pending_exception: Option<(LLValue, LLValue)>,
}

impl LLFrame {
    /// Upstream `LLFrame.__init__(graph, args, llinterpreter)` at
    /// `llinterp.py`.
    fn new(graph: GraphRef, args: Vec<LLValue>) -> Self {
        Self {
            graph,
            args,
            bindings: HashMap::new(),
            curr_block: None,
            curr_operation_index: 0,
            alloca_objects: Vec::new(),
            pending_exception: None,
        }
    }

    /// Upstream `clear` at `:233-234`.
    fn clear(&mut self) {
        self.bindings.clear();
    }

    /// Upstream `fillvars` at `:236-244`.
    fn fillvars(&mut self, block: &BlockRef, values: &[LLValue]) -> Result<(), TaskError> {
        let vars = block.borrow().inputargs.clone();
        if vars.len() != values.len() {
            return Err(TaskError {
                message: format!(
                    "llinterp.py:236 block received {} args, expected {}",
                    values.len(),
                    vars.len()
                ),
            });
        }
        for (var, val) in vars.iter().zip(values.iter()) {
            self.setvar(var, val.clone())?;
        }
        Ok(())
    }

    /// Upstream `setvar` at `:246-253`.
    fn setvar(&mut self, var: &Hlvalue, val: LLValue) -> Result<(), TaskError> {
        let Hlvalue::Variable(v) = var else {
            return Ok(());
        };
        let val = self.enforce_value(v.concretetype(), val)?;
        self.bindings.insert(v.clone(), val);
        Ok(())
    }

    /// Upstream `getval` at `:256-268`.
    fn getval(&self, varorconst: &Hlvalue) -> Result<LLValue, TaskError> {
        let val = match varorconst {
            Hlvalue::Constant(c) => const_to_llvalue(c),
            Hlvalue::Variable(v) => self.bindings.get(v).cloned().ok_or_else(|| TaskError {
                message: format!("llinterp.py:256 unbound variable {v}"),
            })?,
        };
        match varorconst {
            Hlvalue::Variable(v) => self.enforce_value(v.concretetype(), val),
            Hlvalue::Constant(c) => self.enforce_value(c.concretetype.clone(), val),
        }
    }

    fn enforce_value(
        &self,
        concretetype: Option<LowLevelType>,
        val: LLValue,
    ) -> Result<LLValue, TaskError> {
        match concretetype {
            None | Some(LowLevelType::Void) => Ok(val),
            Some(LowLevelType::Signed)
            | Some(LowLevelType::Unsigned)
            | Some(LowLevelType::SignedLongLong)
            | Some(LowLevelType::SignedLongLongLong)
            | Some(LowLevelType::UnsignedLongLong)
            | Some(LowLevelType::UnsignedLongLongLong) => match val {
                LLValue::Int(_) => Ok(val),
                LLValue::Bool(b) => Ok(LLValue::Int(i64::from(b))),
                other => Err(TaskError {
                    message: format!("llinterp.py:246 type error: expected integer, got {other:?}"),
                }),
            },
            Some(LowLevelType::Bool) => match val {
                LLValue::Bool(_) => Ok(val),
                other => Err(TaskError {
                    message: format!("llinterp.py:246 type error: expected Bool, got {other:?}"),
                }),
            },
            Some(LowLevelType::Float)
            | Some(LowLevelType::SingleFloat)
            | Some(LowLevelType::LongFloat) => match val {
                LLValue::Float(_) => Ok(val),
                other => Err(TaskError {
                    message: format!("llinterp.py:246 type error: expected Float, got {other:?}"),
                }),
            },
            Some(LowLevelType::Char) | Some(LowLevelType::UniChar) => match val {
                LLValue::Str(_) => Ok(val),
                other => Err(TaskError {
                    message: format!("llinterp.py:246 type error: expected Char, got {other:?}"),
                }),
            },
            Some(LowLevelType::Address) => match val {
                LLValue::Address(_) => Ok(val),
                other => Err(TaskError {
                    message: format!("llinterp.py:246 type error: expected Address, got {other:?}"),
                }),
            },
            Some(LowLevelType::Ptr(_)) => match val {
                LLValue::Ptr(_) => Ok(val),
                other => Err(TaskError {
                    message: format!("llinterp.py:246 type error: expected Ptr, got {other:?}"),
                }),
            },
            Some(_) => Ok(val),
        }
    }

    /// Upstream `eval` at `llinterp.py`.
    fn eval(
        &mut self,
        llinterpreter: &LLInterpreter,
        self_ref: Rc<RefCell<LLFrame>>,
    ) -> Result<LLValue, TaskError> {
        llinterpreter
            .frame_stack
            .borrow_mut()
            .push(self_ref.clone());
        let mut nextblock = self.graph.borrow().startblock.clone();
        let mut args = self.args.clone();
        let result = loop {
            self.clear();
            if let Err(e) = self.fillvars(&nextblock, &args) {
                break Err(e);
            }
            let (block, values) = match self.eval_block(nextblock.clone()) {
                Ok(next) => next,
                Err(e) => break Err(e),
            };
            match block {
                Some(b) => {
                    nextblock = b;
                    args = values;
                }
                None => {
                    break values.into_iter().next().ok_or_else(|| TaskError {
                        message: "llinterp.py:282 return block produced no value".to_string(),
                    });
                }
            }
        };
        let leavingframe = llinterpreter.frame_stack.borrow_mut().pop();
        debug_assert!(
            leavingframe
                .as_ref()
                .is_some_and(|frame| Rc::ptr_eq(frame, &self_ref))
        );
        result
    }

    /// Upstream `eval_block` at `llinterp.py`.
    fn eval_block(
        &mut self,
        block: BlockRef,
    ) -> Result<(Option<BlockRef>, Vec<LLValue>), TaskError> {
        self.curr_block = Some(block.clone());
        let operations = block.borrow().operations.clone();
        // Upstream `:316-326`: ops run in sequence; only the last op
        // of a `canraise` block is allowed to raise `LLException` —
        // earlier raises propagate as TaskError.
        let canraise = block.borrow().canraise();
        let last_index = operations.len().saturating_sub(1);
        for (i, op) in operations.iter().enumerate() {
            self.curr_operation_index = i;
            self.eval_operation(op)?;
            if self.pending_exception.is_some() && (!canraise || i != last_index) {
                let (etype, _) = self.pending_exception.take().unwrap();
                return Err(TaskError {
                    message: format!(
                        "llinterp.py:316 unhandled LLException at op {}: etype={:?}",
                        op.opname, etype
                    ),
                });
            }
        }

        let exits = block.borrow().exits.clone();
        if exits.is_empty() {
            let variables = block.borrow().getvariables();
            if variables.len() == 2 {
                return Err(TaskError {
                    message:
                        "llinterp.py:325 exception return blocks are not implemented in LLFrame"
                            .to_string(),
                });
            }
            if variables.len() != 1 {
                return Err(TaskError {
                    message: format!(
                        "llinterp.py:338 return block expected one result variable, got {}",
                        variables.len()
                    ),
                });
            }
            let result = self.getval(&Hlvalue::Variable(variables[0].clone()))?;
            return Ok((None, vec![result]));
        }

        let link = if block.borrow().exitswitch.is_none() {
            // Upstream `llinterp.py:365-368`: single-exit block.
            if exits.len() != 1 {
                return Err(TaskError {
                    message: format!(
                        "llinterp.py:350 single-exit block expected one exit, got {}",
                        exits.len()
                    ),
                });
            }
            exits[0].clone()
        } else if canraise {
            // Upstream `llinterp.py:369-384`: a `canraise` block has
            // its normal flow on `block.exits[0]` and exception
            // handlers on `block.exits[1:]`. When `e is None` (the op
            // did not raise) take `exits[0]`. Otherwise walk the
            // handler links, dispatching to the first one whose
            // `llexitcase` matches; populate the link's
            // `last_exception` / `last_exc_value` Variables so the
            // target block sees them.
            match self.pending_exception.take() {
                None => exits[0].clone(),
                Some((etype, evalue)) => self.dispatch_exception(&exits[1..], &etype, &evalue)?,
            }
        } else {
            self.choose_switch_link(&block, &exits)?
        };
        let target = link.borrow().target.clone().ok_or_else(|| TaskError {
            message: "llinterp.py:397 link target is unset".to_string(),
        })?;
        let mut values = Vec::with_capacity(link.borrow().args.len());
        for arg in &link.borrow().args {
            let arg = arg.as_ref().ok_or_else(|| TaskError {
                message: "llinterp.py:397 merge link carried undefined local".to_string(),
            })?;
            values.push(self.getval(arg)?);
        }
        Ok((Some(target), values))
    }

    /// Upstream `llinterp.py:371-384` exception-handler dispatch loop.
    /// Walks `block.exits[1:]`, returns the first link whose
    /// `llexitcase` is a (super)class of `etype`, populating the
    /// link's `last_exception` / `last_exc_value` Variables. If no
    /// handler matches the upstream `else: raise e` branch surfaces
    /// here as a TaskError carrying the etype for the caller.
    fn dispatch_exception(
        &mut self,
        handler_links: &[LinkRef],
        etype: &LLValue,
        evalue: &LLValue,
    ) -> Result<LinkRef, TaskError> {
        for link in handler_links {
            let exitcase = {
                let l = link.borrow();
                debug_assert!(
                    l.exitcase.is_some(),
                    "llinterp.py:376 exception handler link must have an exitcase"
                );
                l.llexitcase.clone().or_else(|| l.exitcase.clone())
            };
            if !self.exception_link_matches(exitcase.as_ref(), etype) {
                continue;
            }
            let (last_exception, last_exc_value) = {
                let l = link.borrow();
                (l.last_exception.clone(), l.last_exc_value.clone())
            };
            if let Some(var) = last_exception {
                self.setvar(&var, etype.clone())?;
            }
            if let Some(var) = last_exc_value {
                self.setvar(&var, evalue.clone())?;
            }
            return Ok(link.clone());
        }
        Err(TaskError {
            message: format!("llinterp.py:382 unhandled LLException: etype={etype:?}",),
        })
    }

    /// Upstream `LLFrame.eval_block` calls
    /// `op_direct_call(exdata.fn_exception_match, cls, link.llexitcase)`.
    /// `exceptiondata.fn_exception_match` is `rclass.py ll_issubclass` over
    /// rtyped class vtables; etype / `llexitcase` Constants here carry
    /// HostObject classes, so the same subclass test runs on those.
    fn exception_link_matches(&self, llexitcase: Option<&Hlvalue>, etype: &LLValue) -> bool {
        let Some(llexitcase) = llexitcase else {
            return false;
        };
        let exitcase = match llexitcase {
            Hlvalue::Constant(c) => host_class_object(&c.value),
            _ => None,
        };
        let cls = match etype {
            LLValue::Const(c) => host_class_object(c),
            _ => None,
        };
        match (cls, exitcase) {
            (Some(cls), Some(exitcase)) => self.fn_exception_match(cls, exitcase),
            _ => false,
        }
    }

    /// `exceptiondata.fn_exception_match` / `rclass.py ll_issubclass`.
    /// True when `cls` is a subclass of `exitcase` (a superclass handler
    /// catches a subclass).
    fn fn_exception_match(&self, cls: &HostObject, exitcase: &HostObject) -> bool {
        cls.is_subclass_of(exitcase)
    }

    /// `LLFrame.make_llexception`: store the standard exception
    /// class/instance as `pending_exception` so `eval_block` routes it
    /// through `dispatch_exception`. The pair is the same
    /// `exception_args` HostObject class/instance
    /// `close_ovfcheck_block` puts on an OverflowError exit, so
    /// `fn_exception_match` walks the HostObject class hierarchy.
    fn make_llexception(&mut self, name: &str) -> Result<(), TaskError> {
        let args =
            crate::translator::rtyper::rtyper::exception_args(name).map_err(|e| TaskError {
                message: format!("LLFrame.make_llexception: {e}"),
            })?;
        let (etype, evalue) = match args.as_slice() {
            [Hlvalue::Constant(cls), Hlvalue::Constant(inst)] => {
                (const_to_llvalue(cls), const_to_llvalue(inst))
            }
            _ => {
                return Err(TaskError {
                    message: format!(
                        "LLFrame.make_llexception: {name} did not yield class/instance"
                    ),
                });
            }
        };
        self.pending_exception = Some((etype, evalue));
        Ok(())
    }

    /// `LLFrame.op_int_add_ovf` / `op_int_sub_ovf` / `op_int_mul_ovf`:
    /// `ovfcheck` then `make_llexception` on overflow. The result is
    /// `exceptiontransform.default_error_value` for Signed (0).
    fn op_int_ovf(
        &mut self,
        opname: &str,
        vals: &[LLValue],
        op: fn(i64, i64) -> Option<i64>,
    ) -> Result<LLValue, TaskError> {
        match op(vals[0].as_i64(opname)?, vals[1].as_i64(opname)?) {
            Some(n) => Ok(LLValue::Int(n)),
            None => {
                self.make_llexception("OverflowError")?;
                Ok(LLValue::Int(0))
            }
        }
    }

    fn choose_switch_link(
        &self,
        block: &BlockRef,
        exits: &[LinkRef],
    ) -> Result<LinkRef, TaskError> {
        let switch = block.borrow().exitswitch.clone().ok_or_else(|| TaskError {
            message: "llinterp.py:359 exitswitch is unset".to_string(),
        })?;
        let llexitvalue = self.getval(&switch)?;
        // Upstream `llinterp.py`: separate the `"default"`
        // exitcase (if any) from the regular cases, search regular
        // cases first, fall back to the default link if no match.
        let (defaultexit, nondefaultexits) = match exits.last() {
            Some(last) if exitcase_is_default(&last.borrow().exitcase) => {
                (Some(last.clone()), &exits[..exits.len() - 1])
            }
            _ => (None, exits),
        };
        for link in nondefaultexits {
            let llexitcase = {
                let link = link.borrow();
                link.llexitcase
                    .as_ref()
                    .and_then(exitcase_value)
                    .or_else(|| link.exitcase.as_ref().and_then(exitcase_value))
            };
            if llexitcase.is_some_and(|case| llvalue_matches_const(&llexitvalue, &case)) {
                return Ok(link.clone());
            }
        }
        if let Some(link) = defaultexit {
            return Ok(link);
        }
        Err(TaskError {
            message: format!("llinterp.py:399 exit case {llexitvalue:?} not found"),
        })
    }

    /// Upstream `eval_operation` at `llinterp.py`.
    fn eval_operation(&mut self, operation: &SpaceOperation) -> Result<(), TaskError> {
        let vals = operation
            .args
            .iter()
            .map(|arg| self.getval(arg))
            .collect::<Result<Vec<_>, _>>()?;
        let retval = self.getoperationhandler(&operation.opname, &vals)?;
        self.setvar(&operation.result, retval)
    }

    /// Upstream `getoperationhandler` at `llinterp.py`, with
    /// only the local primitive fold subset installed.
    fn getoperationhandler(
        &mut self,
        opname: &str,
        vals: &[LLValue],
    ) -> Result<LLValue, TaskError> {
        match opname {
            "same_as" => vals.first().cloned().ok_or_else(|| TaskError {
                message: "llinterp.py:405 same_as expected one argument".to_string(),
            }),
            "int_add" => Ok(LLValue::Int(
                vals[0]
                    .as_i64(opname)?
                    .wrapping_add(vals[1].as_i64(opname)?),
            )),
            "int_add_ovf" => self.op_int_ovf(opname, vals, i64::checked_add),
            "int_add_nonneg_ovf" => {
                let y = vals[1].as_i64(opname)?;
                debug_assert!(
                    y >= 0,
                    "LLFrame.op_int_add_nonneg_ovf: y must be nonnegative"
                );
                let _ = y;
                self.op_int_ovf(opname, vals, i64::checked_add)
            }
            "int_sub_ovf" => self.op_int_ovf(opname, vals, i64::checked_sub),
            "int_mul_ovf" => self.op_int_ovf(opname, vals, i64::checked_mul),
            "int_sub" => Ok(LLValue::Int(
                vals[0]
                    .as_i64(opname)?
                    .wrapping_sub(vals[1].as_i64(opname)?),
            )),
            "int_mul" => Ok(LLValue::Int(
                vals[0]
                    .as_i64(opname)?
                    .wrapping_mul(vals[1].as_i64(opname)?),
            )),
            "int_neg" => Ok(LLValue::Int(vals[0].as_i64(opname)?.wrapping_neg())),
            "int_is_true" => Ok(LLValue::Bool(vals[0].truth())),
            "int_eq" => Ok(LLValue::Bool(
                vals[0].as_i64(opname)? == vals[1].as_i64(opname)?,
            )),
            "int_ne" => Ok(LLValue::Bool(
                vals[0].as_i64(opname)? != vals[1].as_i64(opname)?,
            )),
            "int_lt" => Ok(LLValue::Bool(
                vals[0].as_i64(opname)? < vals[1].as_i64(opname)?,
            )),
            "int_le" => Ok(LLValue::Bool(
                vals[0].as_i64(opname)? <= vals[1].as_i64(opname)?,
            )),
            "int_gt" => Ok(LLValue::Bool(
                vals[0].as_i64(opname)? > vals[1].as_i64(opname)?,
            )),
            "int_ge" => Ok(LLValue::Bool(
                vals[0].as_i64(opname)? >= vals[1].as_i64(opname)?,
            )),
            "bool_not" => Ok(LLValue::Bool(!vals[0].as_bool(opname)?)),
            "cast_bool_to_int" => Ok(LLValue::Int(i64::from(vals[0].as_bool(opname)?))),
            // `op_uint_*` (opimpl.py, generated from the `r_uint` template):
            // the Signed carrier holds the machine word; reinterpret its
            // bits as `u64` for the unsigned operation, then store the
            // result word back as `i64`.
            "uint_floordiv" => {
                let divisor = vals[1].as_i64(opname)? as u64;
                if divisor == 0 {
                    return Err(TaskError {
                        message: "opimpl.py op_uint_floordiv: unsigned division by zero"
                            .to_string(),
                    });
                }
                Ok(LLValue::Int(
                    ((vals[0].as_i64(opname)? as u64) / divisor) as i64,
                ))
            }
            "uint_mod" => {
                let divisor = vals[1].as_i64(opname)? as u64;
                if divisor == 0 {
                    return Err(TaskError {
                        message: "opimpl.py op_uint_mod: unsigned modulo by zero".to_string(),
                    });
                }
                Ok(LLValue::Int(
                    ((vals[0].as_i64(opname)? as u64) % divisor) as i64,
                ))
            }
            // `op_uint_is_true` (opimpl.py, `is_true` template): the word is
            // true iff nonzero — sign-interpretation-independent.
            "uint_is_true" => Ok(LLValue::Bool(vals[0].truth())),
            // `op_cast_uint_to_int(b) = intmask(b)` (opimpl.py):
            // reinterpret the unsigned word's bits as Signed. The carrier
            // already stores the word as `i64`, so this is the identity.
            "cast_uint_to_int" => Ok(LLValue::Int(vals[0].as_i64(opname)?)),
            // `op_cast_primitive` between integer types reinterprets the
            // machine word; the Signed carrier already holds it, so the
            // Signed<->Unsigned cast is the identity on the i64 word.
            "cast_primitive" => Ok(LLValue::Int(vals[0].as_i64(opname)?)),
            // `op_cast_int_to_char(b) = chr(b)` (opimpl.py): the char
            // carrier is the one-char `Str`.
            "cast_int_to_char" => {
                let code = vals[0].as_i64(opname)?;
                let ch = u8::try_from(code).map_err(|_| TaskError {
                    message: format!("opimpl.py op_cast_int_to_char: {code} out of char range"),
                })? as char;
                Ok(LLValue::Str(ch.to_string()))
            }
            // Heap ops: `_struct`/`_array` are Arc-shared with interior
            // mutability, so writes through the cloned `_ptr` carrier persist
            // on the original allocation (pointer semantics).
            "malloc_varsize" => {
                let LLValue::Const(ConstValue::LowLevelType(t)) = &vals[0] else {
                    return Err(TaskError {
                        message: format!("{opname}: arg0 must be a LowLevelType constant"),
                    });
                };
                // `flavor='gc'` unless the flags dict spells `'raw'`
                // (rmodel gc_flavor_const: `Dict{"flavor": "gc"}`).
                let flavor = match &vals[1] {
                    LLValue::Const(ConstValue::Dict(flags))
                        if matches!(
                            flags.get(&ConstValue::byte_str("flavor")),
                            Some(ConstValue::ByteStr(f)) if f.as_slice() == b"raw"
                        ) =>
                    {
                        MallocFlavor::Raw
                    }
                    _ => MallocFlavor::Gc,
                };
                let n_signed = vals[2].as_i64(opname)?;
                let n = usize::try_from(n_signed).map_err(|_| TaskError {
                    message: format!("{opname}: negative varsize length {n_signed}"),
                })?;
                let ptr = malloc((**t).clone(), Some(n), flavor, false).map_err(|e| TaskError {
                    message: format!("{opname}: {e}"),
                })?;
                Ok(LLValue::Ptr(Box::new(ptr)))
            }
            "setfield" => {
                let LLValue::Ptr(p) = &vals[0] else {
                    return Err(TaskError {
                        message: format!("{opname}: arg0 must be a pointer"),
                    });
                };
                let LLValue::Str(field) = &vals[1] else {
                    return Err(TaskError {
                        message: format!("{opname}: arg1 must be a field name"),
                    });
                };
                let mut ptr = (**p).clone();
                ptr.setattr(field, llvalue_to_lowlevel(&vals[2], opname)?)
                    .map_err(|e| TaskError {
                        message: format!("{opname}: {e}"),
                    })?;
                Ok(LLValue::Void)
            }
            "getsubstruct" => {
                let LLValue::Ptr(p) = &vals[0] else {
                    return Err(TaskError {
                        message: format!("{opname}: arg0 must be a pointer"),
                    });
                };
                let LLValue::Str(field) = &vals[1] else {
                    return Err(TaskError {
                        message: format!("{opname}: arg1 must be a field name"),
                    });
                };
                // `op_getsubstruct` = `getattr(obj, field)` on the pointer,
                // yielding a pointer to the inlined sub-container. `_expose`
                // returns it as an interior pointer when the field is a Raw
                // container inlined into a Gc parent (`rpy_string.chars`);
                // materialise that into a plain container `_ptr` so the
                // array op handlers can read/write through it.
                let sub = match p.getattr(field).map_err(|e| TaskError {
                    message: format!("{opname}: {e}"),
                })? {
                    LowLevelValue::Ptr(pt) => *pt,
                    LowLevelValue::InteriorPtr(ip) => {
                        ip._as_container_ptr(p._solid).map_err(|e| TaskError {
                            message: format!("{opname}: {e}"),
                        })?
                    }
                    other => {
                        return Err(TaskError {
                            message: format!(
                                "{opname}: field {field:?} is not a container: {other:?}"
                            ),
                        });
                    }
                };
                Ok(LLValue::Ptr(Box::new(sub)))
            }
            "setarrayitem" => {
                let LLValue::Ptr(p) = &vals[0] else {
                    return Err(TaskError {
                        message: format!("{opname}: arg0 must be a pointer"),
                    });
                };
                let index = vals[1].as_i64(opname)? as usize;
                let mut ptr = (**p).clone();
                ptr.setitem(index, llvalue_to_lowlevel(&vals[2], opname)?)
                    .map_err(|e| TaskError {
                        message: format!("{opname}: {e}"),
                    })?;
                Ok(LLValue::Void)
            }
            "getfield" => self.op_getfield(vals),
            "getarrayitem" => self.op_getarrayitem(vals),
            "getarraysize" => self.op_getarraysize(vals),
            "getinteriorfield" => self.op_getinteriorfield(vals),
            "direct_call" => self.op_direct_call(vals),
            "int_xor" => Ok(LLValue::Int(
                vals[0].as_i64(opname)? ^ vals[1].as_i64(opname)?,
            )),
            "int_and" => Ok(LLValue::Int(
                vals[0].as_i64(opname)? & vals[1].as_i64(opname)?,
            )),
            "int_or" => Ok(LLValue::Int(
                vals[0].as_i64(opname)? | vals[1].as_i64(opname)?,
            )),
            "int_lshift" => Ok(LLValue::Int(signed_lshift(
                vals[0].as_i64(opname)?,
                vals[1].as_i64(opname)?,
            )?)),
            "int_rshift" => Ok(LLValue::Int(signed_rshift(
                vals[0].as_i64(opname)?,
                vals[1].as_i64(opname)?,
            )?)),
            "uint_eq" => Ok(LLValue::Bool(
                (vals[0].as_i64(opname)? as u64) == (vals[1].as_i64(opname)? as u64),
            )),
            "uint_ne" => Ok(LLValue::Bool(
                (vals[0].as_i64(opname)? as u64) != (vals[1].as_i64(opname)? as u64),
            )),
            "uint_and" => Ok(LLValue::Int(
                ((vals[0].as_i64(opname)? as u64) & (vals[1].as_i64(opname)? as u64)) as i64,
            )),
            "uint_or" => Ok(LLValue::Int(
                ((vals[0].as_i64(opname)? as u64) | (vals[1].as_i64(opname)? as u64)) as i64,
            )),
            "uint_xor" => Ok(LLValue::Int(
                ((vals[0].as_i64(opname)? as u64) ^ (vals[1].as_i64(opname)? as u64)) as i64,
            )),
            "uint_lshift" => Ok(LLValue::Int(unsigned_lshift(
                vals[0].as_i64(opname)? as u64,
                vals[1].as_i64(opname)?,
            )? as i64)),
            "uint_rshift" => Ok(LLValue::Int(unsigned_rshift(
                vals[0].as_i64(opname)? as u64,
                vals[1].as_i64(opname)?,
            )? as i64)),
            "llong_eq" | "ullong_eq" | "lllong_eq" | "ulllong_eq" => Ok(LLValue::Bool(
                vals[0].as_i64(opname)? == vals[1].as_i64(opname)?,
            )),
            "llong_and" | "ullong_and" => Ok(LLValue::Int(
                vals[0].as_i64(opname)? & vals[1].as_i64(opname)?,
            )),
            "llong_or" | "ullong_or" => Ok(LLValue::Int(
                vals[0].as_i64(opname)? | vals[1].as_i64(opname)?,
            )),
            "llong_xor" | "ullong_xor" => Ok(LLValue::Int(
                vals[0].as_i64(opname)? ^ vals[1].as_i64(opname)?,
            )),
            "llong_sub" => Ok(LLValue::Int(
                vals[0]
                    .as_i64(opname)?
                    .wrapping_sub(vals[1].as_i64(opname)?),
            )),
            "llong_add" => Ok(LLValue::Int(
                vals[0]
                    .as_i64(opname)?
                    .wrapping_add(vals[1].as_i64(opname)?),
            )),
            "llong_lshift" | "ullong_lshift" => Ok(LLValue::Int(signed_lshift(
                vals[0].as_i64(opname)?,
                vals[1].as_i64(opname)?,
            )?)),
            "llong_rshift" => Ok(LLValue::Int(signed_rshift(
                vals[0].as_i64(opname)?,
                vals[1].as_i64(opname)?,
            )?)),
            "ullong_rshift" => Ok(LLValue::Int(unsigned_rshift(
                vals[0].as_i64(opname)? as u64,
                vals[1].as_i64(opname)?,
            )? as i64)),
            "float_sub" => Ok(LLValue::Float(
                (vals[0].as_f64(opname)? - vals[1].as_f64(opname)?).to_bits(),
            )),
            "float_mul" => Ok(LLValue::Float(
                (vals[0].as_f64(opname)? * vals[1].as_f64(opname)?).to_bits(),
            )),
            "float_eq" => Ok(LLValue::Bool(
                vals[0].as_f64(opname)? == vals[1].as_f64(opname)?,
            )),
            "float_lt" => Ok(LLValue::Bool(
                vals[0].as_f64(opname)? < vals[1].as_f64(opname)?,
            )),
            "float_is_true" => Ok(LLValue::Bool(vals[0].as_f64(opname)? != 0.0)),
            "char_eq" | "unichar_eq" => {
                // `get_primitive_op_src` `op_char_eq` raises `TypeError` unless
                // both args are length-1 `str`; `op_unichar_eq` requires two
                // length-1 unicode values.
                match (&vals[0], &vals[1]) {
                    (LLValue::Str(a), LLValue::Str(b))
                        if a.chars().count() == 1 && b.chars().count() == 1 =>
                    {
                        Ok(LLValue::Bool(a == b))
                    }
                    _ => Err(TaskError {
                        message: format!("TypeError: {opname} expects two characters"),
                    }),
                }
            }
            "cast_char_to_int" | "cast_unichar_to_int" => {
                let LLValue::Str(s) = &vals[0] else {
                    return Err(TaskError {
                        message: format!("{opname}: expected Char, got {:?}", vals[0]),
                    });
                };
                let ch = s.chars().next().ok_or_else(|| TaskError {
                    message: format!("{opname}: empty char"),
                })?;
                Ok(LLValue::Int(ch as i64))
            }
            "cast_int_to_float" => Ok(LLValue::Float((vals[0].as_i64(opname)? as f64).to_bits())),
            "cast_float_to_int" => Ok(LLValue::Int(op_cast_float_to_int(vals[0].as_f64(opname)?)?)),
            "cast_longlong_to_float" => {
                Ok(LLValue::Float((vals[0].as_i64(opname)? as f64).to_bits()))
            }
            "cast_longlong_to_int"
            | "cast_lllong_to_int"
            | "cast_ulonglong_to_int"
            | "cast_ulllong_to_int"
            | "truncate_longlong_to_int" => Ok(LLValue::Int(vals[0].as_i64(opname)?)),
            "convert_float_bytes_to_longlong" => {
                Ok(LLValue::Int(vals[0].as_f64(opname)?.to_bits() as i64))
            }
            "convert_longlong_bytes_to_float" => Ok(LLValue::Float(
                f64::from_bits(vals[0].as_i64(opname)? as u64).to_bits(),
            )),
            "ptr_nonzero" => {
                checkptr_ll(&vals[0], opname)?;
                Ok(LLValue::Bool(vals[0].as_ptr(opname)?.nonzero()))
            }
            "ptr_iszero" => {
                checkptr_ll(&vals[0], opname)?;
                Ok(LLValue::Bool(!vals[0].as_ptr(opname)?.nonzero()))
            }
            "ptr_eq" => {
                checkptr_ll(&vals[0], opname)?;
                checkptr_ll(&vals[1], opname)?;
                Ok(LLValue::Bool(
                    vals[0].as_ptr(opname)? == vals[1].as_ptr(opname)?,
                ))
            }
            "ptr_ne" => {
                checkptr_ll(&vals[0], opname)?;
                checkptr_ll(&vals[1], opname)?;
                Ok(LLValue::Bool(
                    vals[0].as_ptr(opname)? != vals[1].as_ptr(opname)?,
                ))
            }
            "gc_identityhash" | "identityhash" => {
                checkptr_ll(&vals[0], opname)?;
                let p = vals[0].as_ptr(opname)?;
                if !p.nonzero() {
                    return Err(TaskError {
                        message: format!("{opname}: null pointer"),
                    });
                }
                Ok(LLValue::Int(identityhash(p)))
            }
            "cast_adr_to_int" => {
                let addr = vals[0].as_address(opname)?;
                let mode = match vals.get(1) {
                    Some(v) => Some(field_name(v, opname)?),
                    None => None,
                };
                let n =
                    llmemory::cast_adr_to_int(addr, mode.as_deref()).map_err(|e| TaskError {
                        message: format!("{opname}: {e}"),
                    })?;
                Ok(LLValue::Int(n))
            }
            "debug_assert" => {
                if !vals[0].truth() {
                    let msg = vals
                        .get(1)
                        .and_then(|v| match v {
                            LLValue::Str(s) => Some(s.as_str()),
                            _ => None,
                        })
                        .unwrap_or("debug_assert failed");
                    return Err(TaskError {
                        message: format!(
                            "llinterp.py LLFrame.op_debug_assert LLAssertFailure: {msg}"
                        ),
                    });
                }
                Ok(LLValue::Void)
            }
            "keepalive" => Ok(LLValue::Void),
            "jit_marker"
            | "jit_record_exact_class"
            | "jit_record_exact_value"
            | "gc_stack_bottom" => Ok(LLValue::Void),
            "hint" => vals.first().cloned().ok_or_else(|| TaskError {
                message: "llinterp.py LLFrame.op_hint expected a value".to_string(),
            }),
            "jit_conditional_call" => self.op_jit_conditional_call(vals),
            "jit_conditional_call_value" => self.op_jit_conditional_call_value(vals),
            _ => Err(TaskError {
                message: format!(
                    "llinterp.py:273 LLFrame.getoperationhandler — op_{opname} not yet ported"
                ),
            }),
        }
    }

    /// `LLFrame.op_direct_call` → `perform_call` (`llinterp.py`).
    fn op_direct_call(&self, vals: &[LLValue]) -> Result<LLValue, TaskError> {
        if vals.is_empty() {
            return Err(TaskError {
                message: "llinterp.py LLFrame.op_direct_call expected a funcptr".to_string(),
            });
        }
        let p = vals[0].as_ptr("direct_call")?;
        let ARGS = funcptr_ARGS(p)?;
        self.perform_call(&vals[0], &ARGS, &vals[1..])
    }

    /// `LLFrame.perform_call` (`llinterp.py`): graph → `newsubframe.eval`;
    /// otherwise `_callable`. `ARGS` is `lltype.typeOf(f).TO.ARGS`.
    fn perform_call(
        &self,
        f: &LLValue,
        ARGS: &[LowLevelType],
        args: &[LLValue],
    ) -> Result<LLValue, TaskError> {
        let p = f.as_ptr("direct_call")?;
        let func = func_from_ptr(p, "direct_call")?;
        if let Some(graph) = func.graph_ref() {
            validate_graph_funcptr_type(&graph, ARGS)?;
            let interp = LLInterpreter::current_interpreter().ok_or_else(|| TaskError {
                message: "llinterp.py LLFrame.perform_call: no current LLInterpreter".to_string(),
            })?;
            let frame = Rc::new(RefCell::new(LLFrame::new(graph, args.to_vec())));
            frame.borrow_mut().eval(&interp, frame.clone())
        } else if let Some(name) = &func._callable {
            invoke_callable(name, args)
        } else {
            Err(TaskError {
                message: format!(
                    "llinterp.py LLFrame.perform_call — don't know how to execute {}",
                    func._name
                ),
            })
        }
    }

    /// `LLFrame.op_getfield`: `checkptr`; assert the field on
    /// `typeOf(obj).TO` is not a `ContainerType`; then `getattr(obj, field)`.
    fn op_getfield(&self, vals: &[LLValue]) -> Result<LLValue, TaskError> {
        checkptr_ll(&vals[0], "getfield")?;
        let p = vals[0].as_ptr("getfield")?;
        let field = field_name(&vals[1], "getfield")?;
        let field_t = match &typeOf(p).TO {
            PtrTarget::Struct(s) => s.getattr_field_type(&field),
            PtrTarget::ForwardReference(fwd) => match fwd.resolved() {
                Some(LowLevelType::Struct(s)) => s.getattr_field_type(&field),
                _ => None,
            },
            _ => None,
        };
        if field_t.is_some_and(|ftyp| ftyp.is_container_type()) {
            return Err(TaskError {
                message: format!("AssertionError: getfield of container field {field}"),
            });
        }
        let value = p.getattr(&field).map_err(|e| TaskError {
            message: format!("getfield: {e}"),
        })?;
        lowlevel_to_llvalue(value, "getfield")
    }

    /// `op_getarrayitem` (`opimpl.py`): `array[index]`.
    fn op_getarrayitem(&self, vals: &[LLValue]) -> Result<LLValue, TaskError> {
        checkptr_ll(&vals[0], "getarrayitem")?;
        let p = vals[0].as_ptr("getarrayitem")?;
        let index = vals[1].as_i64("getarrayitem")?;
        if index < 0 {
            return Err(TaskError {
                message: format!("getarrayitem: negative index {index}"),
            });
        }
        let value = p.getitem(index as usize).map_err(|e| TaskError {
            message: format!("getarrayitem: {e}"),
        })?;
        lowlevel_to_llvalue(value, "getarrayitem")
    }

    /// `op_getarraysize` (`opimpl.py`): `len(array)`.
    fn op_getarraysize(&self, vals: &[LLValue]) -> Result<LLValue, TaskError> {
        checkptr_ll(&vals[0], "getarraysize")?;
        let p = vals[0].as_ptr("getarraysize")?;
        let n = p.len().map_err(|e| TaskError {
            message: format!("getarraysize: {e}"),
        })?;
        Ok(LLValue::Int(n as i64))
    }

    /// `op_getinteriorfield` (`llinterp.py`): walk field names / indices.
    fn op_getinteriorfield(&self, vals: &[LLValue]) -> Result<LLValue, TaskError> {
        checkptr_ll(&vals[0], "getinteriorfield")?;
        let mut current = llvalue_to_lowlevel(&vals[0], "getinteriorfield")?;
        for offset in &vals[1..] {
            current = match offset {
                LLValue::Int(i) => {
                    if *i < 0 {
                        return Err(TaskError {
                            message: format!("getinteriorfield: negative index {i}"),
                        });
                    }
                    match current {
                        LowLevelValue::Ptr(p) => p.getitem(*i as usize).map_err(|e| TaskError {
                            message: format!("getinteriorfield: {e}"),
                        })?,
                        LowLevelValue::InteriorPtr(ip) => {
                            ip.getitem(*i as usize).map_err(|e| TaskError {
                                message: format!("getinteriorfield: {e}"),
                            })?
                        }
                        other => {
                            return Err(TaskError {
                                message: format!(
                                    "getinteriorfield: not an array at index: {other:?}"
                                ),
                            });
                        }
                    }
                }
                other => {
                    let field = field_name(other, "getinteriorfield")?;
                    match current {
                        LowLevelValue::Ptr(p) => p.getattr(&field).map_err(|e| TaskError {
                            message: format!("getinteriorfield: {e}"),
                        })?,
                        LowLevelValue::InteriorPtr(ip) => {
                            ip.getattr(&field).map_err(|e| TaskError {
                                message: format!("getinteriorfield: {e}"),
                            })?
                        }
                        other => {
                            return Err(TaskError {
                                message: format!(
                                    "getinteriorfield: not a struct at field {field}: {other:?}"
                                ),
                            });
                        }
                    }
                }
            };
        }
        // `LLFrame.op_getinteriorfield`: `assert not isinstance(ob, _interior_ptr)`
        // after walking offsets, before materialising the value.
        if matches!(current, LowLevelValue::InteriorPtr(_)) {
            return Err(TaskError {
                message: "AssertionError: getinteriorfield landed on an interior pointer"
                    .to_string(),
            });
        }
        lowlevel_to_llvalue(current, "getinteriorfield")
    }

    /// `LLFrame.op_jit_conditional_call` (`llinterp.py`): the jitted
    /// arm of inlined `conditional_call`. llinterp takes the
    /// non-jitted arm.
    fn op_jit_conditional_call(&self, _vals: &[LLValue]) -> Result<LLValue, TaskError> {
        Err(TaskError {
            message: "NotImplementedError: should not be called while not jitted".to_string(),
        })
    }

    /// `LLFrame.op_jit_conditional_call_value` (`llinterp.py`): the
    /// jitted arm of inlined `conditional_call_elidable`. llinterp
    /// takes the non-jitted arm.
    fn op_jit_conditional_call_value(&self, _vals: &[LLValue]) -> Result<LLValue, TaskError> {
        Err(TaskError {
            message: "NotImplementedError: should not be called while not jitted".to_string(),
        })
    }
}

fn checkptr_ll(val: &LLValue, opname: &str) -> Result<(), TaskError> {
    match val {
        LLValue::Ptr(_) => Ok(()),
        other => Err(TaskError {
            message: format!("llinterp.checkptr: {opname} expected Ptr, got {other:?}"),
        }),
    }
}

fn exitcase_value(value: &Hlvalue) -> Option<ConstValue> {
    match value {
        Hlvalue::Constant(c) => Some(c.value.clone()),
        Hlvalue::Variable(_) => None,
    }
}

/// Upstream `llinterp.py:387` reads `block.exits[-1].exitcase ==
/// "default"` directly.  In the Rust port the exitcase carrier is
/// `Hlvalue`; the model encodes the default sentinel as
/// `Constant(Str("default"))` (see `flowspace::model::Block` rendering).
fn exitcase_is_default(exitcase: &Option<Hlvalue>) -> bool {
    matches!(
        exitcase,
        Some(Hlvalue::Constant(Constant { value, .. })) if value.string_eq("default")
    )
}

fn llvalue_matches_const(value: &LLValue, case: &ConstValue) -> bool {
    // `LLFrame.eval_block` picks the link whose `llexitcase` equals the
    // switch value. Unsigned constants convert to `LLValue::Int` via
    // `n as i64` (`const_to_llvalue` / `any_to_llvalue`); match that
    // bit pattern against `ConstValue::UInt` exit cases.
    if matches!(
        (value, case),
        (LLValue::Int(a), ConstValue::Int(b)) if a == b
    ) || matches!(
        (value, case),
        (LLValue::Int(a), ConstValue::UInt(b)) if *a == *b as i64
    ) || matches!(
        (value, case),
        (LLValue::Bool(a), ConstValue::Bool(b)) if a == b
    ) || matches!((value, case), (LLValue::Void, ConstValue::None))
    {
        return true;
    }
    // Item 2 string split: LLValue::Str canonicalises to a Rust
    // `String`. Both ByteStr (utf8-lossy) and UniStr compare against
    // it; the comparison itself stays Python 2-bytes-shaped.
    if let LLValue::Str(a) = value {
        match case {
            ConstValue::ByteStr(b) => return a.as_bytes() == b.as_slice(),
            ConstValue::UniStr(b) => return a == b,
            _ => {}
        }
    }
    false
}

/// Rust carrier for `wrap_graph()`'s returned callable.
#[derive(Clone)]
pub struct WrappedGraph {
    pub llinterpreter: Rc<LLInterpreter>,
    pub graph: GraphRef,
    pub self_arg: Vec<Rc<dyn Any>>,
}

impl WrappedGraph {
    pub fn call(&self, args: Vec<Rc<dyn Any>>) -> Result<Rc<dyn Any>, TaskError> {
        let mut graph_args = self.self_arg.clone();
        graph_args.extend(args);
        self.llinterpreter
            .eval_graph(self.graph.clone() as Rc<dyn Any>, graph_args, false)
    }
}

/// RPython `wrap_callable(llinterpreter, fn, obj, method_name)`.
pub fn wrap_callable(
    llinterpreter: Rc<LLInterpreter>,
    fn_graph: Option<GraphRef>,
    obj: Option<Rc<dyn Any>>,
    method_name: Option<&str>,
) -> Result<(String, WrappedGraph), TaskError> {
    if let Some(method_name) = method_name {
        return Err(TaskError {
            message: format!(
                "llinterp.py:1392 wrap_callable method lookup for {method_name:?} is not ported"
            ),
        });
    }
    let graph = fn_graph.ok_or_else(|| TaskError {
        message: "llinterp.py:1392 wrap_callable expected a staticmethod graph".to_string(),
    })?;
    let self_arg = obj.into_iter().collect();
    Ok(wrap_graph(llinterpreter, graph, self_arg))
}

/// RPython `wrap_graph(llinterpreter, graph, self_arg)`.
pub fn wrap_graph(
    llinterpreter: Rc<LLInterpreter>,
    graph: GraphRef,
    self_arg: Vec<Rc<dyn Any>>,
) -> (String, WrappedGraph) {
    let name = graph.borrow().name.clone();
    (
        name,
        WrappedGraph {
            llinterpreter,
            graph,
            self_arg,
        },
    )
}

/// RPython `enumerate_exceptions_top_down()`.
pub fn enumerate_exceptions_top_down() -> Result<Vec<String>, TyperError> {
    Err(TyperError::missing_rtype_operation(
        "llinterp.enumerate_exceptions_top_down requires host Python exception module reflection",
    ))
}

/// RPython `_address_of_local_var`.
#[derive(Clone)]
pub struct _address_of_local_var {
    pub frame: Rc<RefCell<LLFrame>>,
    pub v: Variable,
}

impl _address_of_local_var {
    pub const _TYPE: LowLevelType = LowLevelType::Address;

    pub fn new(frame: Rc<RefCell<LLFrame>>, v: Variable) -> Self {
        Self { frame, v }
    }

    pub fn _getaddress(&self) -> _address_of_local_var_accessor {
        _address_of_local_var_accessor::new(self.frame.clone(), self.v.clone())
    }

    pub fn address(&self) -> _address_of_local_var_accessor {
        self._getaddress()
    }
}

/// RPython `_address_of_local_var_accessor`.
#[derive(Clone)]
pub struct _address_of_local_var_accessor {
    pub frame: Rc<RefCell<LLFrame>>,
    pub v: Variable,
}

impl _address_of_local_var_accessor {
    pub fn new(frame: Rc<RefCell<LLFrame>>, v: Variable) -> Self {
        Self { frame, v }
    }

    /// RPython `__getitem__(0)`.
    pub fn __getitem__(&self, index: usize) -> Result<_address, TaskError> {
        if index != 0 {
            return Err(TaskError {
                message: "address of local vars only support [0] indexing".to_string(),
            });
        }
        match self
            .frame
            .borrow()
            .getval(&Hlvalue::Variable(self.v.clone()))?
        {
            LLValue::Ptr(p) => Ok(self.unwrap_possible_weakref(llmemory::cast_ptr_to_adr(&p))),
            other => Err(TaskError {
                message: format!(
                    "_address_of_local_var_accessor.__getitem__: expected local Ptr, got {other:?}"
                ),
            }),
        }
    }

    /// RPython `__setitem__(0, newvalue)`.
    pub fn __setitem__(&self, index: usize, newvalue: _address) -> Result<(), TaskError> {
        if index != 0 {
            return Err(TaskError {
                message: "address of local vars only support [0] indexing".to_string(),
            });
        }
        let ptr_type = match self.v.concretetype() {
            Some(LowLevelType::Ptr(ptr_type)) => ptr_type,
            other => {
                return Err(TaskError {
                    message: format!("llinterp.py:1456 expected Ptr concretetype, got {other:?}"),
                });
            }
        };
        let p = llmemory::cast_adr_to_ptr(&newvalue, &ptr_type).map_err(|message| TaskError {
            message: format!("llinterp.py:1456 cast_adr_to_ptr failed: {message}"),
        })?;
        self.frame.borrow_mut().setvar(
            &Hlvalue::Variable(self.v.clone()),
            LLValue::Ptr(Box::new(p)),
        )
    }

    pub fn unwrap_possible_weakref(&self, addr: _address) -> _address {
        addr
    }
}

/// RPython `_address_of_thread_local`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct _address_of_thread_local;

impl _address_of_thread_local {
    pub const _TYPE: LowLevelType = LowLevelType::Address;
    pub const is_fake_thread_local_addr: bool = true;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::flowspace::model::{Block, BlockRefExt, HOST_ENV, Link};
    use crate::translator::rtyper::lltypesystem::lltype::{
        _getconcretetype, Array, FuncType, LowLevelType, Ptr, functionptr,
        functionptr_for_graph_with_type, getfunctionptr, nullptr,
    };
    use crate::translator::rtyper::rclass::{Flavor, InstanceRepr, OBJECT};
    use crate::translator::rtyper::rmodel::{Repr, address_repr};
    use crate::translator::rtyper::rtyper::{LowLevelFunction, RPythonTyper};

    fn fixture_typer() -> Rc<RPythonTyper> {
        let ann = RPythonAnnotator::new(None, None, None, false);
        Rc::new(RPythonTyper::new(&ann))
    }

    #[test]
    fn llinterpreter_init_mirrors_upstream_slots() {
        // Upstream `:72-82`: `__init__(typer, tracing=True,
        // exc_data_ptr=None)` writes `bindings = {}`, `typer = typer`,
        // `exc_data_ptr = exc_data_ptr`, `frame_stack = []`, `tracer
        // = None or Tracer()`.
        let typer = fixture_typer();
        let interp = LLInterpreter::new(typer, true, None);
        assert!(interp.bindings.borrow().is_empty());
        assert!(interp.exc_data_ptr.is_none());
        assert!(interp.frame_stack.borrow().is_empty());
    }

    #[test]
    fn top_level_helper_parity_surface() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            MallocFlavor, Struct, malloc, typeOf,
        };

        assert_eq!(log.name, "llinterp");
        let err = PleaseOverwriteStoreException("overwrite me".to_string());
        assert_eq!(err.to_string(), "overwrite me");

        let struct_t = LowLevelType::Struct(Box::new(Struct::gc("pkg.C", vec![])));
        let ptr = malloc(struct_t, None, MallocFlavor::Gc, true).expect("malloc instance");
        checkptr(&LowLevelValue::Ptr(Box::new(ptr.clone()))).expect("checkptr accepts Ptr");
        checkadr(&LowLevelValue::Address(llmemory::cast_ptr_to_adr(&ptr)))
            .expect("checkadr accepts Address");
        assert!(checkptr(&LowLevelValue::Bool(true)).is_err());
        assert!(checkadr(&LowLevelValue::Bool(true)).is_err());

        let graph = Rc::new(RefCell::new(FunctionGraph::new(
            "addr_frame",
            Block::shared(vec![]),
        )));
        let frame = Rc::new(RefCell::new(LLFrame::new(graph, Vec::new())));
        let v = Variable::named("p");
        v.set_concretetype(Some(LowLevelType::Ptr(Box::new(typeOf(&ptr)))));
        frame
            .borrow_mut()
            .setvar(
                &Hlvalue::Variable(v.clone()),
                LLValue::Ptr(Box::new(ptr.clone())),
            )
            .expect("seed local ptr");
        let local_addr = _address_of_local_var::new(frame.clone(), v.clone());
        let accessor = local_addr.address();
        assert_eq!(
            accessor.__getitem__(0).expect("local address"),
            llmemory::cast_ptr_to_adr(&ptr)
        );
        assert!(accessor.__getitem__(1).is_err());
        accessor
            .__setitem__(0, llmemory::cast_ptr_to_adr(&ptr))
            .expect("write local address");

        assert_eq!(_address_of_local_var::_TYPE, LowLevelType::Address);
        assert_eq!(_address_of_thread_local::_TYPE, LowLevelType::Address);
        let missing = enumerate_exceptions_top_down().expect_err("host reflection deferred");
        assert!(missing.is_missing_rtype_operation());
    }

    #[test]
    fn eval_graph_rejects_non_graph_until_driver_passes_real_entry_graph() {
        let interp = Rc::new(LLInterpreter::new(fixture_typer(), false, None));
        let err = interp
            .eval_graph(Rc::new(()) as Rc<dyn Any>, Vec::new(), false)
            .expect_err("non-graph inputs still fail at the typed leaf");
        assert!(err.message.contains("llinterp.py:84"), "{}", err.message);
    }

    #[test]
    fn eval_graph_runs_straight_line_int_add_to_returnblock() {
        // Covers upstream `eval_graph` (`llinterp.py`),
        // `LLFrame.eval` (`:282-307`), `fillvars/getval/setvar`
        // (`:236-268`), single-exit `eval_block` (`:309-403`), and
        // primitive `eval_operation` dispatch (`:405-525`).
        let x = Variable::named("x");
        x.set_concretetype(Some(LowLevelType::Signed));
        let y = Variable::named("y");
        y.set_concretetype(Some(LowLevelType::Signed));
        let z = Variable::named("z");
        z.set_concretetype(Some(LowLevelType::Signed));

        let start = Block::shared(vec![x.clone().into(), y.clone().into()]);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![x.into(), y.into()],
            z.clone().into(),
        ));
        let retvar = Hlvalue::Variable(Variable::named("ret"));
        if let Hlvalue::Variable(v) = &retvar {
            v.set_concretetype(Some(LowLevelType::Signed));
        }
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "add",
            start.clone(),
            retvar,
        )));
        let returnblock = graph.borrow().returnblock.clone();
        start.closeblock(vec![
            Link::new(vec![z.into()], Some(returnblock), None).into_ref(),
        ]);

        let interp = Rc::new(LLInterpreter::new(fixture_typer(), false, None));
        let out = interp
            .eval_graph(
                graph as Rc<dyn Any>,
                vec![
                    Rc::new(40_i64) as Rc<dyn Any>,
                    Rc::new(2_i64) as Rc<dyn Any>,
                ],
                false,
            )
            .expect("straight-line int_add graph should run");
        assert_eq!(*out.downcast::<i64>().expect("Signed return"), 42);
        assert!(interp.frame_stack.borrow().is_empty());
        // Upstream `:99` `LLInterpreter.current_interpreter = self`:
        // every public eval_graph entry must have published the
        // thread-local pointer while running. After return the local
        // mirror is restored to the prior (empty) Weak.
        assert!(LLInterpreter::current_interpreter().is_none());
    }

    #[test]
    fn getoperationhandler_folds_uint_div_mod_and_casts() {
        // Covers the `op_uint_floordiv` / `op_uint_mod` / `op_uint_is_true`
        // / `op_cast_uint_to_int` folds (opimpl.py, `r_uint` templates +
        // :465-467), chained straight-line so one eval_graph run exercises
        // all four: 12345 //u 10 = 1234; 1234 %u 10 = 4; intmask(4) = 4;
        // is_true(4) = True; cast_bool_to_int(True) = 1. The result folds the
        // `cast_uint_to_int` value `s` back in (`int_add(s, c) = 4 + 1 = 5`)
        // so a broken `uint_floordiv`/`uint_mod` — which `uint_is_true` alone
        // would collapse to 1 for any nonzero result — changes the return.
        use crate::flowspace::model::Constant as FlowConstant;
        let ten = || Hlvalue::Constant(FlowConstant::new(ConstValue::Int(10)));

        let x = Variable::named("x");
        x.set_concretetype(Some(LowLevelType::Signed));
        let a = Variable::named("a");
        a.set_concretetype(Some(LowLevelType::Unsigned));
        let b = Variable::named("b");
        b.set_concretetype(Some(LowLevelType::Unsigned));
        let s = Variable::named("s");
        s.set_concretetype(Some(LowLevelType::Signed));
        let t = Variable::named("t");
        t.set_concretetype(Some(LowLevelType::Bool));
        let c = Variable::named("c");
        c.set_concretetype(Some(LowLevelType::Signed));
        let d = Variable::named("d");
        d.set_concretetype(Some(LowLevelType::Signed));

        let start = Block::shared(vec![x.clone().into()]);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "uint_floordiv",
            vec![x.into(), ten()],
            a.clone().into(),
        ));
        start.borrow_mut().operations.push(SpaceOperation::new(
            "uint_mod",
            vec![a.into(), ten()],
            b.clone().into(),
        ));
        start.borrow_mut().operations.push(SpaceOperation::new(
            "cast_uint_to_int",
            vec![b.into()],
            s.clone().into(),
        ));
        start.borrow_mut().operations.push(SpaceOperation::new(
            "uint_is_true",
            vec![s.clone().into()],
            t.clone().into(),
        ));
        start.borrow_mut().operations.push(SpaceOperation::new(
            "cast_bool_to_int",
            vec![t.into()],
            c.clone().into(),
        ));
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![s.into(), c.clone().into()],
            d.clone().into(),
        ));

        let retvar = Hlvalue::Variable(Variable::named("ret"));
        if let Hlvalue::Variable(v) = &retvar {
            v.set_concretetype(Some(LowLevelType::Signed));
        }
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "uint_chain",
            start.clone(),
            retvar,
        )));
        let returnblock = graph.borrow().returnblock.clone();
        start.closeblock(vec![
            Link::new(vec![d.into()], Some(returnblock), None).into_ref(),
        ]);

        let interp = Rc::new(LLInterpreter::new(fixture_typer(), false, None));
        let out = interp
            .eval_graph(
                graph as Rc<dyn Any>,
                vec![Rc::new(12345_i64) as Rc<dyn Any>],
                false,
            )
            .expect("uint fold chain should run");
        // 4 (cast_uint_to_int of 1234 %u 10) + 1 (cast_bool_to_int of is_true) = 5.
        assert_eq!(*out.downcast::<i64>().expect("Signed return"), 5);
    }

    #[test]
    fn getoperationhandler_mallocs_char_array_and_writes_through_pointer() {
        // Covers malloc_varsize (bare GcArray path), cast_int_to_char (char =
        // one-char Str), and setarrayitem writing through the Arc-shared
        // `_ptr` carrier: allocate a 3-char array, write 'a'/'b'/'c', return
        // the pointer, and confirm the writes persisted on the allocation.
        use crate::flowspace::model::Constant as FlowConstant;
        use crate::translator::rtyper::lltypesystem::lltype::Array;

        let char_array = LowLevelType::Array(Box::new(Array::gc(LowLevelType::Char)));
        let type_c = || {
            Hlvalue::Constant(FlowConstant::new(ConstValue::LowLevelType(Box::new(
                char_array.clone(),
            ))))
        };
        let flavor_c = || {
            Hlvalue::Constant(FlowConstant::new(ConstValue::Dict(std::sync::Arc::new(
                indexmap::IndexMap::new(),
            ))))
        };
        let int_c = |n: i64| Hlvalue::Constant(FlowConstant::new(ConstValue::Int(n)));

        let arr = Variable::named("arr");
        let start = Block::shared(vec![]);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "malloc_varsize",
            vec![type_c(), flavor_c(), int_c(3)],
            arr.clone().into(),
        ));
        for (idx, code) in [(0_i64, 97_i64), (1, 98), (2, 99)] {
            let ch = Variable::named("ch");
            ch.set_concretetype(Some(LowLevelType::Char));
            start.borrow_mut().operations.push(SpaceOperation::new(
                "cast_int_to_char",
                vec![int_c(code)],
                ch.clone().into(),
            ));
            let set = Variable::named("set");
            set.set_concretetype(Some(LowLevelType::Void));
            start.borrow_mut().operations.push(SpaceOperation::new(
                "setarrayitem",
                vec![arr.clone().into(), int_c(idx), ch.into()],
                set.into(),
            ));
        }

        let retvar = Hlvalue::Variable(Variable::named("ret"));
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "malloc_write",
            start.clone(),
            retvar,
        )));
        let returnblock = graph.borrow().returnblock.clone();
        start.closeblock(vec![
            Link::new(vec![arr.into()], Some(returnblock), None).into_ref(),
        ]);

        let interp = Rc::new(LLInterpreter::new(fixture_typer(), false, None));
        let out = interp
            .eval_graph(graph as Rc<dyn Any>, Vec::new(), false)
            .expect("malloc + write-through graph should run");
        let LowLevelValue::Ptr(ptr) = &*out.downcast::<LowLevelValue>().expect("Ptr return") else {
            panic!("expected an array pointer return");
        };
        for (idx, expected) in [(0_usize, 'a'), (1, 'b'), (2, 'c')] {
            match ptr.getitem(idx).expect("in-bounds read") {
                LowLevelValue::Char(c) => assert_eq!(c, expected, "char at {idx}"),
                other => panic!("expected Char at {idx}, got {other:?}"),
            }
        }
    }

    #[test]
    fn getoperationhandler_malloc_varsize_rejects_negative_length() {
        // A `malloc_varsize` whose length is negative must fail before any
        // allocation: the `usize::try_from(n_signed)` guard turns a `-1` length
        // into a `TaskError` ("negative varsize length") rather than allocating
        // a wrapped-huge array. Same graph shape as the write-through test,
        // stopped at the single failing malloc.
        use crate::flowspace::model::Constant as FlowConstant;
        use crate::translator::rtyper::lltypesystem::lltype::Array;

        let char_array = LowLevelType::Array(Box::new(Array::gc(LowLevelType::Char)));
        let type_c = Hlvalue::Constant(FlowConstant::new(ConstValue::LowLevelType(Box::new(
            char_array,
        ))));
        let flavor_c = Hlvalue::Constant(FlowConstant::new(ConstValue::Dict(std::sync::Arc::new(
            indexmap::IndexMap::new(),
        ))));
        let neg_len = Hlvalue::Constant(FlowConstant::new(ConstValue::Int(-1)));

        let arr = Variable::named("arr");
        let start = Block::shared(vec![]);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "malloc_varsize",
            vec![type_c, flavor_c, neg_len],
            arr.clone().into(),
        ));

        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "malloc_neg",
            start.clone(),
            arr.clone().into(),
        )));
        let returnblock = graph.borrow().returnblock.clone();
        start.closeblock(vec![
            Link::new(vec![arr.into()], Some(returnblock), None).into_ref(),
        ]);

        let interp = Rc::new(LLInterpreter::new(fixture_typer(), false, None));
        let err = interp
            .eval_graph(graph as Rc<dyn Any>, Vec::new(), false)
            .expect_err("negative varsize length must abort evaluation before allocating");
        let msg = format!("{err}");
        assert!(
            msg.contains("negative varsize length"),
            "expected a negative-length rejection, got {msg:?}"
        );
    }

    #[test]
    fn build_ll_int2dec_executes_to_decimal_strings() {
        // Run the synthesised `ll_int2dec` helper graph
        // (`lltypesystem/rstr.rs`) end-to-end through the interpreter and read
        // the decimal string back out of the returned STR pointer. This is the
        // executing correctness gate the hex builder's shape-only test lacks —
        // it catches an off-by-one in the reversed `total_len - j - 1` slot or a
        // swapped link argument that a BFS-of-opnames assertion cannot.
        use crate::translator::rtyper::lltypesystem::rstr::build_ll_int2dec_helper_graph;

        fn read_rpy_string(strptr: &_ptr) -> String {
            // Read `strptr.chars` the same way `getsubstruct` does: `getattr`
            // yields an interior pointer for the inlined array, materialised
            // into a plain container `_ptr` to index through.
            let chars = match strptr.getattr("chars").expect("chars field") {
                LowLevelValue::Ptr(p) => *p,
                LowLevelValue::InteriorPtr(ip) => {
                    ip._as_container_ptr(true).expect("materialise chars ptr")
                }
                other => panic!("chars is not a container: {other:?}"),
            };
            let mut s = String::new();
            let mut i = 0usize;
            while let Ok(LowLevelValue::Char(c)) = chars.getitem(i) {
                s.push(c);
                i += 1;
            }
            s
        }

        let interp = Rc::new(LLInterpreter::new(fixture_typer(), false, None));
        let run = |helper: &crate::flowspace::pygraph::PyGraph, input: i64| -> String {
            let out = interp
                .eval_graph(
                    helper.graph.clone() as Rc<dyn Any>,
                    vec![Rc::new(input) as Rc<dyn Any>],
                    false,
                )
                .expect("int2dec helper graph should run");
            let LowLevelValue::Ptr(p) = &*out.downcast::<LowLevelValue>().expect("Ptr return")
            else {
                panic!("expected a STR pointer return");
            };
            read_rpy_string(p)
        };

        // Unsigned specialisation: -1 arrives as the full u64 word.
        let unsigned = build_ll_int2dec_helper_graph("ll_int2dec", false).expect("build unsigned");
        for (input, expected) in [
            (0_i64, "0"),
            (7, "7"),
            (12345, "12345"),
            (-1, "18446744073709551615"),
        ] {
            assert_eq!(run(&unsigned, input), expected, "uint2dec({input})");
        }

        // Signed specialisation: the `val < 0` branch takes the magnitude.
        let signed =
            build_ll_int2dec_helper_graph("ll_int2dec_signed", true).expect("build signed");
        for (input, expected) in [
            (0_i64, "0"),
            (7, "7"),
            (-42, "-42"),
            (12345, "12345"),
            (-12345, "-12345"),
        ] {
            assert_eq!(run(&signed, input), expected, "int2dec({input})");
        }
    }

    #[test]
    fn eval_graph_publishes_current_interpreter_during_execution() {
        // Upstream `llinterp.py` `LLInterpreter.current_interpreter
        // = self` is mandatory side effect of every `eval_graph` call.
        // Without it, downstream opcode handlers (e.g. an `op_*` that
        // reads `LLInterpreter.current_interpreter` per upstream
        // `:405-525`) cannot recover the active interpreter.
        let x = Variable::named("x");
        x.set_concretetype(Some(LowLevelType::Signed));
        let start = Block::shared(vec![x.clone().into()]);
        let retvar = Hlvalue::Variable(Variable::named("ret"));
        if let Hlvalue::Variable(v) = &retvar {
            v.set_concretetype(Some(LowLevelType::Signed));
        }
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "passthrough",
            start.clone(),
            retvar,
        )));
        let returnblock = graph.borrow().returnblock.clone();
        start.closeblock(vec![
            Link::new(vec![x.into()], Some(returnblock), None).into_ref(),
        ]);

        let interp = Rc::new(LLInterpreter::new(fixture_typer(), false, None));
        // Before the call: thread-local has no current interpreter.
        assert!(LLInterpreter::current_interpreter().is_none());
        let _ = interp
            .eval_graph(
                graph as Rc<dyn Any>,
                vec![Rc::new(7_i64) as Rc<dyn Any>],
                false,
            )
            .expect("passthrough graph runs");
        // After the call: the thread-local has been restored to the
        // prior (empty) Weak. Upstream's class attribute survives the
        // call, but the local port saves/restores so re-entrant
        // evaluations behave correctly.
        assert!(LLInterpreter::current_interpreter().is_none());
    }

    #[test]
    fn llexception_carries_upstream_argument_layout() {
        // Upstream `:28-54`: `LLException(*args, error_value=...)`. The
        // local port preserves `args` and `error_value` slots so tests
        // tracking `e.error_value` (e.g. test_exceptiontransform per
        // `:30-32`) can land verbatim later.
        let exc = LLException::new(vec![Rc::new("etype".to_string()) as Rc<dyn Any>], None);
        assert_eq!(exc.args.len(), 1);
        assert!(exc.error_value.is_none());
    }

    #[test]
    fn llfatalerror_renders_args_separated_by_colons() {
        // Upstream `:56-58`: `__str__ = ': '.join([str(x) for x in
        // self.args])`.
        let err = LLFatalError(vec!["one".to_string(), "two".to_string()]);
        assert_eq!(err.to_string(), "one: two");
    }

    #[test]
    fn eval_block_switch_default_falls_through_when_no_case_matches() {
        // Upstream `llinterp.py:387-402`: the last exit's exitcase
        // `"default"` is consulted only when no non-default case
        // matches. Build a 3-exit switch: case 1, case 2, default.
        // The exitswitch value is 99 (no match) → must take default.
        use crate::flowspace::model::Constant as FlowConstant;
        let switch_var = Variable::named("s");
        switch_var.set_concretetype(Some(LowLevelType::Signed));
        let start = Block::shared(vec![switch_var.clone().into()]);
        start.borrow_mut().exitswitch = Some(switch_var.clone().into());

        let retvar = Hlvalue::Variable(Variable::named("ret"));
        if let Hlvalue::Variable(v) = &retvar {
            v.set_concretetype(Some(LowLevelType::Signed));
        }
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "switch_default",
            start.clone(),
            retvar,
        )));
        let returnblock = graph.borrow().returnblock.clone();

        // Cases produce distinct return constants so we can tell which arm fired.
        let case1 = Hlvalue::Constant(FlowConstant::new(ConstValue::Int(1)));
        let case2 = Hlvalue::Constant(FlowConstant::new(ConstValue::Int(2)));
        let default_case = Hlvalue::Constant(FlowConstant::new(ConstValue::byte_str("default")));

        let v100 = Hlvalue::Constant(FlowConstant::new(ConstValue::Int(100)));
        let v200 = Hlvalue::Constant(FlowConstant::new(ConstValue::Int(200)));
        let v999 = Hlvalue::Constant(FlowConstant::new(ConstValue::Int(999)));

        let mut link1 = Link::new(vec![v100], Some(returnblock.clone()), Some(case1.clone()));
        link1.llexitcase = Some(case1);
        let mut link2 = Link::new(vec![v200], Some(returnblock.clone()), Some(case2.clone()));
        link2.llexitcase = Some(case2);
        // Upstream `:390`: `assert defaultexit.llexitcase is None`.
        let link_default = Link::new(vec![v999], Some(returnblock), Some(default_case));
        start.closeblock(vec![
            link1.into_ref(),
            link2.into_ref(),
            link_default.into_ref(),
        ]);

        let interp = Rc::new(LLInterpreter::new(fixture_typer(), false, None));
        let out = interp
            .eval_graph(
                graph as Rc<dyn Any>,
                vec![Rc::new(99_i64) as Rc<dyn Any>],
                false,
            )
            .expect("switch with default arm runs");
        assert_eq!(*out.downcast::<i64>().expect("Signed return"), 999);

        // Unsigned switch: `llexitcase` is `ConstValue::UInt`; the
        // switch value is the same bit pattern (`LLValue::Int`).
        // Matching 0 takes the UInt(0) arm; 99 falls through to default.
        let uswitch = Variable::named("us");
        uswitch.set_concretetype(Some(LowLevelType::Unsigned));
        let ustart = Block::shared(vec![uswitch.clone().into()]);
        ustart.borrow_mut().exitswitch = Some(uswitch.clone().into());
        let uretvar = Hlvalue::Variable(Variable::named("uret"));
        if let Hlvalue::Variable(v) = &uretvar {
            v.set_concretetype(Some(LowLevelType::Signed));
        }
        let ugraph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "switch_uint",
            ustart.clone(),
            uretvar,
        )));
        let ureturnblock = ugraph.borrow().returnblock.clone();
        let ucase0 = Hlvalue::Constant(FlowConstant::new(ConstValue::UInt(0)));
        let ucase1 = Hlvalue::Constant(FlowConstant::new(ConstValue::UInt(1)));
        let udefault = Hlvalue::Constant(FlowConstant::new(ConstValue::byte_str("default")));
        let mut ulink0 = Link::new(
            vec![Hlvalue::Constant(FlowConstant::new(ConstValue::Int(10)))],
            Some(ureturnblock.clone()),
            Some(ucase0.clone()),
        );
        ulink0.llexitcase = Some(ucase0);
        let mut ulink1 = Link::new(
            vec![Hlvalue::Constant(FlowConstant::new(ConstValue::Int(20)))],
            Some(ureturnblock.clone()),
            Some(ucase1.clone()),
        );
        ulink1.llexitcase = Some(ucase1);
        let ulink_default = Link::new(
            vec![Hlvalue::Constant(FlowConstant::new(ConstValue::Int(999)))],
            Some(ureturnblock),
            Some(udefault),
        );
        ustart.closeblock(vec![
            ulink0.into_ref(),
            ulink1.into_ref(),
            ulink_default.into_ref(),
        ]);
        let uout0 = interp
            .eval_graph(
                ugraph.clone() as Rc<dyn Any>,
                vec![Rc::new(0_i64) as Rc<dyn Any>],
                false,
            )
            .expect("unsigned switch matches UInt(0)");
        assert_eq!(*uout0.downcast::<i64>().expect("Signed return"), 10);
        let uout_default = interp
            .eval_graph(
                ugraph as Rc<dyn Any>,
                vec![Rc::new(99_i64) as Rc<dyn Any>],
                false,
            )
            .expect("unsigned switch default arm runs");
        assert_eq!(*uout_default.downcast::<i64>().expect("Signed return"), 999);
    }

    #[test]
    fn eval_block_canraise_takes_normal_exit_when_no_exception() {
        // Upstream `llinterp.py:369-384`: a `canraise` block has its
        // normal flow on `block.exits[0]`, with exception handlers on
        // `block.exits[1:]`.  Without `LLException` propagation in the
        // local port, every `eval_operation` either succeeds or returns
        // a `TaskError` (= upstream `LLFatalError`); the no-exception
        // arm must fire.
        use crate::flowspace::model::{Constant as FlowConstant, LAST_EXCEPTION};
        let x = Variable::named("x");
        x.set_concretetype(Some(LowLevelType::Signed));
        let y = Variable::named("y");
        y.set_concretetype(Some(LowLevelType::Signed));

        let start = Block::shared(vec![x.clone().into(), y.clone().into()]);
        // A raising op must be the last operation per
        // `flowspace::model::Block::canraise()`.
        let z = Variable::named("z");
        z.set_concretetype(Some(LowLevelType::Signed));
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![x.clone().into(), y.clone().into()],
            z.clone().into(),
        ));
        start.borrow_mut().exitswitch = Some(Hlvalue::Constant(FlowConstant::new(
            ConstValue::Atom(LAST_EXCEPTION.clone()),
        )));
        assert!(start.borrow().canraise());

        let retvar = Hlvalue::Variable(Variable::named("ret"));
        if let Hlvalue::Variable(v) = &retvar {
            v.set_concretetype(Some(LowLevelType::Signed));
        }
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "canraise_no_exc",
            start.clone(),
            retvar,
        )));
        let returnblock = graph.borrow().returnblock.clone();

        // exits[0] = normal flow, exits[1] = exception handler.
        let normal_link = Link::new(vec![z.clone().into()], Some(returnblock.clone()), None);
        let raised_const = Hlvalue::Constant(FlowConstant::new(ConstValue::Int(-1)));
        let exc_link = Link::new(
            vec![raised_const],
            Some(returnblock),
            Some(Hlvalue::Constant(FlowConstant::new(ConstValue::byte_str(
                "BaseException",
            )))),
        );
        start.closeblock(vec![normal_link.into_ref(), exc_link.into_ref()]);

        let interp = Rc::new(LLInterpreter::new(fixture_typer(), false, None));
        let out = interp
            .eval_graph(
                graph as Rc<dyn Any>,
                vec![
                    Rc::new(40_i64) as Rc<dyn Any>,
                    Rc::new(2_i64) as Rc<dyn Any>,
                ],
                false,
            )
            .expect("canraise block with successful op takes exits[0]");
        assert_eq!(*out.downcast::<i64>().expect("Signed return"), 42);
    }

    #[test]
    fn exception_link_matches_a_superclass_handler() {
        // `exceptiondata.fn_exception_match` / `rclass.py ll_issubclass`:
        // a superclass handler catches a subclass.
        let frame = dummy_frame();
        let zdiv = HOST_ENV
            .lookup_exception_class("ZeroDivisionError")
            .expect("ZeroDivisionError");
        let exception = HOST_ENV
            .lookup_exception_class("Exception")
            .expect("Exception");
        let type_error = HOST_ENV
            .lookup_exception_class("TypeError")
            .expect("TypeError");

        let etype = LLValue::Const(ConstValue::HostObject(zdiv));
        let super_case = Hlvalue::Constant(Constant::new(ConstValue::HostObject(exception)));
        let unrelated = Hlvalue::Constant(Constant::new(ConstValue::HostObject(type_error)));

        assert!(frame.exception_link_matches(Some(&super_case), &etype));
        assert!(!frame.exception_link_matches(Some(&unrelated), &etype));
    }

    fn pending_overflow_name(frame: &LLFrame) -> Option<String> {
        match &frame.pending_exception {
            Some((LLValue::Const(c), _)) => c.host_class_name().map(str::to_owned),
            _ => None,
        }
    }

    #[test]
    fn int_add_ovf_returns_sum_and_raises_overflowerror() {
        // `LLFrame.op_int_add_ovf`: ovfcheck(x + y); OverflowError becomes
        // pending_exception via `make_llexception`.
        let mut frame = dummy_frame();
        let sum = frame
            .getoperationhandler("int_add_ovf", &[LLValue::Int(40), LLValue::Int(2)])
            .expect("int_add_ovf sum");
        assert_eq!(sum, LLValue::Int(42));
        assert!(frame.pending_exception.is_none());

        let overflowed = frame
            .getoperationhandler("int_add_ovf", &[LLValue::Int(i64::MAX), LLValue::Int(1)])
            .expect("int_add_ovf overflow returns default_error_value");
        assert_eq!(overflowed, LLValue::Int(0));
        assert_eq!(
            pending_overflow_name(&frame).as_deref(),
            Some("OverflowError")
        );

        frame.pending_exception = None;
        let nonneg = frame
            .getoperationhandler("int_add_nonneg_ovf", &[LLValue::Int(3), LLValue::Int(4)])
            .expect("int_add_nonneg_ovf sum");
        assert_eq!(nonneg, LLValue::Int(7));
        assert!(frame.pending_exception.is_none());

        let sub = frame
            .getoperationhandler("int_sub_ovf", &[LLValue::Int(i64::MIN), LLValue::Int(1)])
            .expect("int_sub_ovf overflow");
        assert_eq!(sub, LLValue::Int(0));
        assert_eq!(
            pending_overflow_name(&frame).as_deref(),
            Some("OverflowError")
        );

        frame.pending_exception = None;
        let mul = frame
            .getoperationhandler("int_mul_ovf", &[LLValue::Int(i64::MAX), LLValue::Int(2)])
            .expect("int_mul_ovf overflow");
        assert_eq!(mul, LLValue::Int(0));
        assert_eq!(
            pending_overflow_name(&frame).as_deref(),
            Some("OverflowError")
        );
    }

    fn malloc_resized_signed_list(list_lltype: &LowLevelType, items: &[i64]) -> _ptr {
        let LowLevelType::Ptr(ptr) = list_lltype else {
            panic!("ListRepr lltype is Ptr");
        };
        let PtrTarget::Struct(body) = &ptr.TO else {
            panic!("ListRepr Ptr target is Struct");
        };
        let items_t = body._flds.get("items").expect("items field").clone();
        let LowLevelType::Ptr(items_ptr) = &items_t else {
            panic!("items field is Ptr");
        };
        let PtrTarget::Array(arr_t) = &items_ptr.TO else {
            panic!("items TO is Array");
        };
        let mut header = malloc(
            LowLevelType::Struct(Box::new(body.clone())),
            None,
            MallocFlavor::Gc,
            true,
        )
        .expect("list header");
        let mut arr = malloc(
            LowLevelType::Array(Box::new(arr_t.clone())),
            Some(items.len()),
            MallocFlavor::Gc,
            true,
        )
        .expect("list items");
        for (i, v) in items.iter().enumerate() {
            arr.setitem(i, LowLevelValue::Signed(*v)).expect("setitem");
        }
        header
            .setattr("length", LowLevelValue::Signed(items.len() as i64))
            .expect("length");
        header
            .setattr("items", LowLevelValue::Ptr(Box::new(arr)))
            .expect("items field");
        header
    }

    fn list_signed_contents(list: &_ptr) -> Vec<i64> {
        let length = match list.getattr("length").expect("length") {
            LowLevelValue::Signed(n) => n as usize,
            other => panic!("length is Signed, got {other:?}"),
        };
        let items = match list.getattr("items").expect("items") {
            LowLevelValue::Ptr(p) => p,
            other => panic!("items is Ptr, got {other:?}"),
        };
        (0..length)
            .map(|i| match items.getitem(i).expect("getitem") {
                LowLevelValue::Signed(n) => n,
                other => panic!("item is Signed, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn ll_extend_runs_in_llinterp() {
        // Mint `ll_extend` the way `rlist` tests mint `rtype_method("append")`,
        // then `eval_funcptr` two small Signed lists.
        use std::sync::Arc;

        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::{SomeInteger, SomeList, SomeValue};
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rlist::ListRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let list_repr: Arc<ListRepr> =
            Arc::new(ListRepr::new(&rtyper, r_int).expect("ListRepr::new"));
        let list_lltype = list_repr.lowleveltype().clone();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let v_l1 = Variable::new();
        v_l1.set_concretetype(Some(list_lltype.clone()));
        let v_l2 = Variable::new();
        v_l2.set_concretetype(Some(list_lltype.clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "simple_call".to_string(),
                vec![Hlvalue::Variable(v_l1), Hlvalue::Variable(v_l2)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::List(SomeList::new(ListDef::new(
                None,
                SomeValue::Integer(SomeInteger::new(false, false)),
                true,
                true,
            ))),
            SomeValue::List(SomeList::new(ListDef::new(
                None,
                SomeValue::Integer(SomeInteger::new(false, false)),
                true,
                true,
            ))),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(list_repr.clone() as Arc<dyn Repr>),
            Some(list_repr.clone() as Arc<dyn Repr>),
        ]);

        list_repr
            .rtype_method("extend", &hop)
            .unwrap_or_else(|err| panic!("list extend: {err:?}"));
        let fptr = {
            let ops = llops.borrow();
            assert_eq!(ops.ops[0].opname, "direct_call");
            let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
                panic!("expected Constant funcptr as direct_call arg 0");
            };
            match &c.value {
                ConstValue::LLPtr(p) => (**p).clone(),
                other => panic!("expected LLPtr funcptr, got {other:?}"),
            }
        };

        let l1 = malloc_resized_signed_list(&list_lltype, &[1, 2]);
        let l2 = malloc_resized_signed_list(&list_lltype, &[3, 4]);
        eval_funcptr(
            &fptr,
            &[
                LowLevelValue::Ptr(Box::new(l1.clone())),
                LowLevelValue::Ptr(Box::new(l2)),
            ],
        )
        .expect("ll_extend eval_funcptr");
        assert_eq!(list_signed_contents(&l1), vec![1, 2, 3, 4]);
        let _keep = ann;
    }

    #[test]
    fn jit_intrinsic_funcptr_runs_its_false_body() {
        // `rlib/jit.py isconstant`: GRAPH is `return NonConstant(False)`;
        // `LLFrame.perform_call` runs `_ptr._obj.graph`.
        use crate::translator::rtyper::lltypesystem::rstr::jit_isconstant_intrinsic_funcptr;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let c = jit_isconstant_intrinsic_funcptr(&rtyper, LowLevelType::Signed)
            .expect("jit_isconstant_intrinsic_funcptr");
        let fptr = match &c.value {
            ConstValue::LLPtr(p) => (**p).clone(),
            other => panic!("expected LLPtr funcptr, got {other:?}"),
        };
        let out = eval_funcptr(&fptr, &[LowLevelValue::Signed(0)]).expect("eval_funcptr");
        assert_eq!(out, LowLevelValue::Bool(false));
        let _keep = ann;
    }

    fn add_graph() -> GraphRef {
        let x = Variable::named("x");
        x.set_concretetype(Some(LowLevelType::Signed));
        let y = Variable::named("y");
        y.set_concretetype(Some(LowLevelType::Signed));
        let z = Variable::named("z");
        z.set_concretetype(Some(LowLevelType::Signed));
        let start = Block::shared(vec![x.clone().into(), y.clone().into()]);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![x.into(), y.into()],
            z.clone().into(),
        ));
        let retvar = Hlvalue::Variable(Variable::named("ret"));
        if let Hlvalue::Variable(v) = &retvar {
            v.set_concretetype(Some(LowLevelType::Signed));
        }
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "add",
            start.clone(),
            retvar,
        )));
        let returnblock = graph.borrow().returnblock.clone();
        start.closeblock(vec![
            Link::new(vec![z.into()], Some(returnblock), None).into_ref(),
        ]);
        graph
    }

    fn graphless(name: &str, args: Vec<LowLevelType>, result: LowLevelType) -> _ptr {
        functionptr(
            FuncType { args, result },
            name,
            None,
            Some(name.to_string()),
        )
    }

    #[test]
    fn python_frexp_matches_ll_math_frexp_specials_and_normals() {
        // `ll_math.py ll_math_frexp`: non-finite / ±0 keep the input and
        // exponent 0; finite non-zero uses mantissa in [0.5, 1).
        assert_eq!(python_frexp(8.0), (0.5, 4));
        assert_eq!(python_frexp(f64::INFINITY), (f64::INFINITY, 0));
        assert_eq!(python_frexp(f64::NEG_INFINITY), (f64::NEG_INFINITY, 0));
        let (nan_m, nan_e) = python_frexp(f64::NAN);
        assert!(nan_m.is_nan());
        assert_eq!(nan_e, 0);
        let (m0, e0) = python_frexp(-0.0);
        assert_eq!(m0.to_bits(), (-0.0f64).to_bits());
        assert_eq!(e0, 0);
        let (mant, exp) = python_frexp(f64::from_bits(1));
        assert_eq!(mant, 0.5);
        assert_eq!(exp, -1073);
    }

    #[test]
    fn math_frexp_callable_returns_mantissa_exponent_tuple() {
        let fptr = graphless("math.frexp", vec![LowLevelType::Float], LowLevelType::Void);
        let LowLevelValue::Ptr(p) =
            eval_funcptr(&fptr, &[LowLevelValue::Float(8.0f64.to_bits())]).expect("math.frexp")
        else {
            panic!("math.frexp returns a tuple2 pointer");
        };
        assert_eq!(
            p.getattr("item0").expect("item0"),
            LowLevelValue::Float(0.5f64.to_bits())
        );
        assert_eq!(p.getattr("item1").expect("item1"), LowLevelValue::Signed(4));

        let LowLevelValue::Ptr(inf_p) =
            eval_funcptr(&fptr, &[LowLevelValue::Float(f64::INFINITY.to_bits())])
                .expect("math.frexp inf")
        else {
            panic!("math.frexp inf returns a tuple2 pointer");
        };
        assert_eq!(
            inf_p.getattr("item0").expect("item0"),
            LowLevelValue::Float(f64::INFINITY.to_bits())
        );
        assert_eq!(
            inf_p.getattr("item1").expect("item1"),
            LowLevelValue::Signed(0)
        );
    }

    #[test]
    fn c_frexp_callable_writes_exponent_through_intp() {
        let exp_p = malloc(
            LowLevelType::Array(Box::new(Array::new(LowLevelType::Signed))),
            Some(1),
            MallocFlavor::Raw,
            false,
        )
        .expect("INTP");
        let fptr = graphless(
            "frexp",
            vec![
                LowLevelType::Float,
                LowLevelType::Ptr(Box::new(Ptr {
                    TO: PtrTarget::Array(Array::new(LowLevelType::Signed)),
                })),
            ],
            LowLevelType::Float,
        );
        let mant = eval_funcptr(
            &fptr,
            &[
                LowLevelValue::Float(8.0f64.to_bits()),
                LowLevelValue::Ptr(Box::new(exp_p.clone())),
            ],
        )
        .expect("C frexp");
        assert_eq!(mant, LowLevelValue::Float(0.5f64.to_bits()));
        assert_eq!(
            exp_p.getitem(0).expect("exp_p[0]"),
            LowLevelValue::Signed(4)
        );
    }

    fn funcptr_for_hash_helper(llfn: LowLevelFunction) -> _ptr {
        let graph = llfn
            .graph
            .expect("hash helper carries a graph")
            .graph
            .clone();
        getfunctionptr(&graph, _getconcretetype).expect("hash helper funcptr")
    }

    #[test]
    fn ll_inst_hash_hashes_container_not_ptr_wrapper() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let repr = InstanceRepr::new_rootinstance(&rtyper, Flavor::Gc);
        let llfn = repr
            .get_ll_hash_function(&rtyper)
            .expect("ll_inst_hash helper")
            .expect("Some");
        let fptr = funcptr_for_hash_helper(llfn);
        let p = malloc(OBJECT.clone(), None, MallocFlavor::Gc, true).expect("instance");
        let p2 = _ptr::new_with_solid(p._TYPE.clone(), p._obj0_value(), p._solid);
        assert_ne!(p._hashable_identity(), p2._hashable_identity());
        let h1 = eval_funcptr(&fptr, &[LowLevelValue::Ptr(Box::new(p.clone()))])
            .expect("ll_inst_hash p");
        let h2 = eval_funcptr(&fptr, &[LowLevelValue::Ptr(Box::new(p2))]).expect("ll_inst_hash p2");
        assert_eq!(h1, h2);
        let q = malloc(OBJECT.clone(), None, MallocFlavor::Gc, true).expect("other instance");
        let h3 = eval_funcptr(&fptr, &[LowLevelValue::Ptr(Box::new(q))]).expect("ll_inst_hash q");
        assert_ne!(h1, h3);

        let null = nullptr(OBJECT.clone()).expect("null instance");
        assert_eq!(
            eval_funcptr(&fptr, &[LowLevelValue::Ptr(Box::new(null))]).expect("null hash"),
            LowLevelValue::Signed(0)
        );
    }

    #[test]
    fn ll_addrhash_uses_forced_cast_adr_to_int() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let llfn = address_repr()
            .get_ll_hash_function(&rtyper)
            .expect("ll_addrhash helper")
            .expect("Some");
        let fptr = funcptr_for_hash_helper(llfn);
        assert_eq!(
            eval_funcptr(&fptr, &[LowLevelValue::Address(_address::Null)]).expect("null addr hash"),
            LowLevelValue::Signed(0)
        );
        let live = malloc(
            LowLevelType::Struct(Box::new(Struct::new(
                "thing",
                vec![("x".into(), LowLevelType::Signed)],
            ))),
            None,
            MallocFlavor::Raw,
            false,
        )
        .expect("live");
        let live2 = _ptr::new_with_solid(live._TYPE.clone(), live._obj0_value(), live._solid);
        let h1 = eval_funcptr(
            &fptr,
            &[LowLevelValue::Address(llmemory::cast_ptr_to_adr(&live))],
        )
        .expect("addr hash");
        let h2 = eval_funcptr(
            &fptr,
            &[LowLevelValue::Address(llmemory::cast_ptr_to_adr(&live2))],
        )
        .expect("addr hash wrapper");
        assert_eq!(h1, h2);
    }

    #[test]
    fn ll_strhash_llinterp_takes_non_jitted_arm() {
        use crate::translator::rtyper::lltypesystem::rstr::{ll_strhash_value, llstr};
        use crate::translator::rtyper::rstr::string_repr;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let llfn = string_repr()
            .get_ll_hash_function(&rtyper)
            .expect("ll_strhash helper")
            .expect("Some");
        let fptr = funcptr_for_hash_helper(llfn);

        let bytes = b"hello";
        let s = llstr(bytes).expect("llstr hash=0");
        assert_eq!(
            s.getattr("hash").expect("hash field"),
            LowLevelValue::Signed(0)
        );
        let expected = ll_strhash_value(bytes);
        let h = eval_funcptr(&fptr, &[LowLevelValue::Ptr(Box::new(s.clone()))])
            .expect("ll_strhash miss");
        assert_eq!(h, LowLevelValue::Signed(expected));
        assert_eq!(
            s.getattr("hash").expect("hash field after miss"),
            LowLevelValue::Signed(expected)
        );

        let mut cached = llstr(bytes).expect("llstr cached");
        cached
            .setattr("hash", LowLevelValue::Signed(12345))
            .expect("set hash");
        let h_cached = eval_funcptr(&fptr, &[LowLevelValue::Ptr(Box::new(cached.clone()))])
            .expect("ll_strhash hit");
        assert_eq!(h_cached, LowLevelValue::Signed(12345));
        assert_eq!(
            cached.getattr("hash").expect("hash field after hit"),
            LowLevelValue::Signed(12345)
        );
    }

    #[test]
    fn jit_conditional_call_value_raises_while_not_jitted() {
        let x = Variable::named("x");
        x.set_concretetype(Some(LowLevelType::Signed));
        let f = Variable::named("f");
        f.set_concretetype(Some(LowLevelType::Signed));
        let result = Variable::named("result");
        result.set_concretetype(Some(LowLevelType::Signed));

        let start = Block::shared(vec![x.clone().into(), f.clone().into()]);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "jit_conditional_call_value",
            vec![x.into(), f.into()],
            result.clone().into(),
        ));
        let retvar = Hlvalue::Variable(Variable::named("ret"));
        if let Hlvalue::Variable(v) = &retvar {
            v.set_concretetype(Some(LowLevelType::Signed));
        }
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "bare_jit_conditional_call_value",
            start.clone(),
            retvar,
        )));
        let returnblock = graph.borrow().returnblock.clone();
        start.closeblock(vec![
            Link::new(vec![result.into()], Some(returnblock), None).into_ref(),
        ]);

        let interp = Rc::new(LLInterpreter::new(fixture_typer(), false, None));
        let err = interp
            .eval_graph(
                graph as Rc<dyn Any>,
                vec![Rc::new(0_i64) as Rc<dyn Any>, Rc::new(0_i64) as Rc<dyn Any>],
                false,
            )
            .expect_err("jit_conditional_call_value is refused while not jitted");
        assert!(
            err.message.contains("NotImplementedError"),
            "{}",
            err.message
        );
        assert!(
            err.message
                .contains("should not be called while not jitted"),
            "{}",
            err.message
        );
    }

    #[test]
    fn eval_funcptr_rejects_arity_and_type_mismatch_then_runs_matching_call() {
        let graph = add_graph();
        let arity_ft = FuncType {
            args: vec![],
            result: LowLevelType::Signed,
        };
        let arity_ptr = functionptr_for_graph_with_type(&graph, arity_ft);
        let arity_err = eval_funcptr(&arity_ptr, &[]).expect_err("arity mismatch");
        assert!(
            arity_err
                .message
                .contains("graph with 2 args called with wrong func ptr type"),
            "{}",
            arity_err.message
        );

        let type_ft = FuncType {
            args: vec![LowLevelType::Float, LowLevelType::Float],
            result: LowLevelType::Signed,
        };
        let type_ptr = functionptr_for_graph_with_type(&graph, type_ft);
        let type_err = eval_funcptr(
            &type_ptr,
            &[
                LowLevelValue::Float(0.0f64.to_bits()),
                LowLevelValue::Float(0.0f64.to_bits()),
            ],
        )
        .expect_err("type mismatch");
        assert!(
            type_err.message.contains("called with wrong func ptr type"),
            "{}",
            type_err.message
        );

        let ok_ptr = getfunctionptr(&graph, _getconcretetype).expect("matching funcptr");
        let out = eval_funcptr(
            &ok_ptr,
            &[LowLevelValue::Signed(40), LowLevelValue::Signed(2)],
        )
        .expect("matching call runs");
        assert_eq!(out, LowLevelValue::Signed(42));
    }

    #[test]
    fn eval_funcptr_keeps_unsigned_result_type() {
        let x = Variable::named("x");
        x.set_concretetype(Some(LowLevelType::Unsigned));
        let start = Block::shared(vec![x.clone().into()]);
        let retvar = Hlvalue::Variable(Variable::named("ret"));
        if let Hlvalue::Variable(v) = &retvar {
            v.set_concretetype(Some(LowLevelType::Unsigned));
        }
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "id_unsigned",
            start.clone(),
            retvar,
        )));
        let returnblock = graph.borrow().returnblock.clone();
        start.closeblock(vec![
            Link::new(vec![x.into()], Some(returnblock), None).into_ref(),
        ]);
        let ft = FuncType {
            args: vec![LowLevelType::Unsigned],
            result: LowLevelType::Unsigned,
        };
        let ptr = functionptr_for_graph_with_type(&graph, ft);
        let out = eval_funcptr(&ptr, &[LowLevelValue::Unsigned(7)]).expect("unsigned identity");
        assert_eq!(out, LowLevelValue::Unsigned(7));
    }

    fn dummy_frame() -> LLFrame {
        let graph = Rc::new(RefCell::new(FunctionGraph::new(
            "op_frame",
            Block::shared(vec![]),
        )));
        LLFrame::new(graph, Vec::new())
    }

    #[test]
    fn negative_shift_raises_value_error() {
        // Host Python shifts (`get_primitive_op_src` / `op_int_rshift` /
        // `op_uint_lshift` / `op_uint_rshift`): `y < 0` is ValueError.
        for err in [
            super::signed_lshift(1, -1).expect_err("signed_lshift"),
            super::signed_rshift(1, -1).expect_err("signed_rshift"),
            super::unsigned_lshift(1, -1).expect_err("unsigned_lshift"),
            super::unsigned_rshift(1, -1).expect_err("unsigned_rshift"),
        ] {
            assert!(err.message.contains("ValueError"), "{}", err.message);
        }
        assert_eq!(super::signed_rshift(-8, 70).unwrap(), -1);
        assert_eq!(super::signed_lshift(1, 64).unwrap(), 0);
    }

    #[test]
    fn cast_float_to_int_raises_on_nonfinite_and_wraps() {
        // `opimpl.py` `op_cast_float_to_int`.
        assert_eq!(
            super::op_cast_float_to_int(1e19).unwrap(),
            10000000000000000000i128 as i64
        );
        assert_eq!(super::op_cast_float_to_int(-2.9).unwrap(), -2);
        let inf = super::op_cast_float_to_int(f64::INFINITY).expect_err("inf");
        assert!(inf.message.contains("OverflowError"), "{}", inf.message);
        let nan = super::op_cast_float_to_int(f64::NAN).expect_err("nan");
        assert!(nan.message.contains("ValueError"), "{}", nan.message);
    }

    #[test]
    fn char_eq_rejects_non_char() {
        let mut frame = dummy_frame();
        let err = frame
            .getoperationhandler("char_eq", &[LLValue::Int(1), LLValue::Str("a".into())])
            .expect_err("non-char");
        assert!(err.message.contains("TypeError"), "{}", err.message);
        let err = frame
            .getoperationhandler(
                "unichar_eq",
                &[LLValue::Str("ab".into()), LLValue::Str("a".into())],
            )
            .expect_err("multi-char");
        assert!(err.message.contains("TypeError"), "{}", err.message);
        assert_eq!(
            frame
                .getoperationhandler(
                    "char_eq",
                    &[LLValue::Str("a".into()), LLValue::Str("a".into())]
                )
                .unwrap(),
            LLValue::Bool(true)
        );
        assert_eq!(
            frame
                .getoperationhandler(
                    "char_eq",
                    &[LLValue::Str("a".into()), LLValue::Str("b".into())]
                )
                .unwrap(),
            LLValue::Bool(false)
        );
    }

    fn nested_struct_ptr() -> _ptr {
        let inner = Struct::new("inner", vec![("x".into(), LowLevelType::Signed)]);
        let outer = Struct::gc(
            "outer",
            vec![
                ("sub".into(), LowLevelType::Struct(Box::new(inner))),
                ("n".into(), LowLevelType::Signed),
            ],
        );
        malloc(
            LowLevelType::Struct(Box::new(outer)),
            None,
            MallocFlavor::Gc,
            true,
        )
        .expect("nested struct")
    }

    #[test]
    fn getfield_of_container_field_raises() {
        // `LLFrame.op_getfield`: container field (inlined sub-struct) is
        // `AssertionError`, not a value.
        let mut frame = dummy_frame();
        let p = nested_struct_ptr();
        let err = frame
            .getoperationhandler(
                "getfield",
                &[LLValue::Ptr(Box::new(p)), LLValue::Str("sub".into())],
            )
            .expect_err("container field");
        assert!(err.message.contains("AssertionError"), "{}", err.message);
    }

    #[test]
    fn getinteriorfield_of_interior_pointer_raises() {
        // `LLFrame.op_getinteriorfield`: walking to an inlined Raw sub-struct
        // of a Gc parent lands on `_interior_ptr`.
        let mut frame = dummy_frame();
        let p = nested_struct_ptr();
        let err = frame
            .getoperationhandler(
                "getinteriorfield",
                &[LLValue::Ptr(Box::new(p)), LLValue::Str("sub".into())],
            )
            .expect_err("interior pointer");
        assert!(err.message.contains("AssertionError"), "{}", err.message);
    }
}
