//! Flow-space high-level operations.
//!
//! RPython upstream: `rpython/flowspace/operation.py` (764 LOC).
//!
//! Commit split:
//!
//! * **Commit 1 (this file's initial landing)** — data model skeleton.
//!   Every RPython `HLOperation` subclass surfaces as an `OpKind`
//!   enum variant; property tables (`arity`, `pure`, `can_overflow`,
//!   `dispatch`, `ovf_variant`) mirror the add_operator table line by
//!   line. `constfold()` returns `None` pending Commit 2. No wiring
//!   into `flowcontext.rs` yet — `record_pure_op` / `record_maybe_raise_op`
//!   still emit `SpaceOperation` directly via raw opname strings.
//! * **Commit 2** — real `constfold()` bodies (pyfunc mirror via match
//!   on `ConstValue`) + `BuiltinException::canraise` data populated
//!   from the trailing `_add_exceptions` / `_add_except_ovf` loop.
//! * **Commit 3** — the explicit subclasses with custom `eval()` logic
//!   (`NewDict`, `NewTuple`, `NewList`, `Pow`, `Iter`, `Next`,
//!   `GetAttr`, `SimpleCall`, `CallArgs`, `Contains`). Flowcontext
//!   migrates to OpKind-based emission.
//!
//! Rust adaptation (parity rule #1, minimum deviation):
//!
//! * Python `class HLOperation(SpaceOperation)` + `HLOperationMeta`
//!   metaclass dispatch collapses into `struct HLOperation { kind:
//!   OpKind, … }` plus a single enum. Each `OpKind` variant is the
//!   direct 1:1 mapping of an RPython `add_operator('name', …)` call
//!   or a top-level class declaration (`NewDict`, `Pow`, `SimpleCall`,
//!   …).
//! * RPython's global `op.*` namespace — populated by `HLOperationMeta.
//!   __init__` via `setattr(op, cls.opname, cls)` — is replaced by the
//!   `OpKind` variant itself; code that says `op.add` in RPython says
//!   `OpKind::Add` in Rust.
//! * `HLOperationMeta._registry` / `_transform` runtime dicts used by
//!   `SingleDispatchMixin` / `DoubleDispatchMixin` are not materialised
//!   here — they are consumed only by the annotator and rtyper, which
//!   land in Phases 4–6. `OpKind::dispatch()` preserves the `None` / 1
//!   / 2 classification so those phases can populate their own tables.

use std::collections::HashMap;
use std::sync::Arc;

use super::flowcontext::FlowingError;
use super::model::{ConstValue, Constant, Hlvalue, SpaceOperation, Variable};

/// RPython `NOT_REALLY_CONST` (operation.py). Maps a module
/// qualname to the set of attribute names that are *real* constants
/// on that module; any other attribute of a listed module is treated
/// as runtime-variable and [`constfold_getattr`] declines the fold.
///
/// Upstream uses `Constant(sys)` as the outer key and `Constant(name)`
/// as the inner set member; the Rust port flattens to string keys
/// because our `HostObject::Module` carries a unique qualname that
/// identifies the hosted module 1:1. Adding a module here is a parity
/// requirement — without it, `getattr(sys, 'path')` would fold to a
/// compile-time snapshot, diverging from upstream.
fn not_really_const_declines(module_qualname: &str, attr: &str) -> bool {
    static SYS_REAL_CONSTS: &[&str] = &[
        "maxint",
        "maxunicode",
        "api_version",
        "exit",
        "exc_info",
        "getrefcount",
        "getdefaultencoding",
    ];
    if module_qualname == "sys" {
        return !SYS_REAL_CONSTS.contains(&attr);
    }
    // Modules not in NOT_REALLY_CONST fold normally.
    false
}

/// RPython `rpython/flowspace/operation.py` — enumerates every
/// `HLOperation` subclass. Variant order matches the upstream
/// `add_operator` table then the explicit subclass block.
///
/// `opname()` returns the exact string identifier used on
/// `SpaceOperation.opname`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OpKind {
    // add_operator table (operation.py)
    // unary/binary operators registered via add_operator(); variant
    // names follow the upstream `opname` argument (UpperCamel'd), with
    // `_ovf` siblings as `*Ovf`.
    Is,
    Id,
    Type,
    IsSubtype,
    IsInstance,
    Repr,
    Str,
    Format,
    Len,
    Hash,
    SetAttr,
    DelAttr,
    GetItem,
    GetItemIdx,
    SetItem,
    DelItem,
    GetSlice,
    SetSlice,
    DelSlice,
    Trunc,
    Pos,
    Neg,
    NegOvf,
    Bool,
    Abs,
    AbsOvf,
    Hex,
    Oct,
    Bin,
    Ord,
    Invert,
    Add,
    AddOvf,
    Sub,
    SubOvf,
    Mul,
    MulOvf,
    TrueDiv,
    FloorDiv,
    FloorDivOvf,
    Div,
    DivOvf,
    Mod,
    ModOvf,
    DivMod,
    LShift,
    LShiftOvf,
    RShift,
    And,
    Or,
    Xor,
    Int,
    Index,
    Float,
    Long,
    InplaceAdd,
    InplaceSub,
    InplaceMul,
    InplaceTrueDiv,
    InplaceFloorDiv,
    InplaceDiv,
    InplaceMod,
    InplacePow,
    InplaceLShift,
    InplaceRShift,
    InplaceAnd,
    InplaceOr,
    InplaceXor,
    Lt,
    Le,
    Eq,
    Ne,
    Gt,
    Ge,
    Cmp,
    Coerce,
    Get,
    Set,
    Delete,
    UserDel,
    Buffer,
    Yield,
    NewSlice,
    Hint,

    // explicit subclasses (operation.py Contains and later HLOperation)
    // These carry custom `eval()` / `consider()` overrides in RPython.
    // Commit 3 lifts the custom logic into the Rust port.
    Contains,
    NewDict,
    NewTuple,
    NewList,
    /// `objectmodel.newlist_hint(sizehint)`'s ExtRegistry specialization.
    /// RPython does not expose this as an ordinary operation.py subclass;
    /// pyre gives the specialization a synthetic high-level opname so the
    /// Rust `try_vec_with_capacity` spelling can enter the same annotator and
    /// rtyper path without translating the Rust allocator body.
    NewListHint,
    /// `StringBuilder()` / `UnicodeBuilder()` construction.  A pyre
    /// adaptation of the RPython `rlib.rstring` ExtRegistryEntry ctor
    /// (`simple_call` on the builder class → `SomeStringBuilder`,
    /// `specialize_call` → `hop.r_result.rtyper_new`), modelled as a
    /// manual-dispatch construction op alongside `NewList`/`NewDict`.
    NewStringBuilder,
    /// Construction of a Rust `Vec<T>` with one-word items — the
    /// `ll_vec_newemptylist_*` / `ll_vec_newlist_hint_*` helpers seen at the
    /// high level, the way `newlist` / `newlist_hint` stand for
    /// `ll_newemptylist` / `ll_newlist_hint`. Argument 0 is a constant item
    /// lowleveltype (`lltype.py` `malloc` / `ann_malloc`: `Signed`,
    /// `Unsigned`, `Float`, or `GCREF`). An optional argument 1 is the
    /// capacity hint; arguments 1 and 2 are the fill count and value.
    NewRustVec,
    /// `ll_copy` (`rlist.py`) of a one-word slice into a raw `Vec`.
    /// Argument 0 is the constant item lowleveltype. Argument 1 is
    /// `self`. A `FixedSizeListRepr` self is
    /// `ll_vec_copy_from_fixed_list_{segment}`; any other self stays a
    /// `direct_call` of `to_vec`. `Signed` and `Unsigned` stay distinct
    /// (`lltype_to_annotation`). No oopspec: `ll_copy` is inlined by the JIT.
    LlVecCopyFromSlice,
    Pow,
    Iter,
    Next,
    GetAttr,
    SimpleCall,
    CallArgs,
}

/// RPython `operation.py` `canraise` entries reference the Python
/// exception classes `ValueError`, `OverflowError`, … directly. The
/// Rust port carries identities through this enum so the trailing
/// `_add_exceptions` table (operation.py) can populate
/// [`OpKind::canraise`] without pulling in the full `HOST_ENV`
/// exception hierarchy at this phase.
///
/// This list matches the set of exception names literally mentioned by
/// `operation.py`; `Exception` is the "any" fallback used by
/// `CallOp.canraise` and `op.getitem`.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BuiltinException {
    ValueError,
    UnicodeDecodeError,
    ZeroDivisionError,
    OverflowError,
    MemoryError,
    IndexError,
    KeyError,
    StopIteration,
    RuntimeError,
    Exception,
}

impl BuiltinException {
    /// Python class name — matches `HOST_ENV.lookup_builtin(...)` keys
    /// so `Bookkeeper.new_exception([BuiltinException::IndexError, ...])`
    /// can resolve each entry through the existing HostObject lookup.
    pub fn host_name(self) -> &'static str {
        match self {
            BuiltinException::ValueError => "ValueError",
            BuiltinException::UnicodeDecodeError => "UnicodeDecodeError",
            BuiltinException::ZeroDivisionError => "ZeroDivisionError",
            BuiltinException::OverflowError => "OverflowError",
            BuiltinException::MemoryError => "MemoryError",
            BuiltinException::IndexError => "IndexError",
            BuiltinException::KeyError => "KeyError",
            BuiltinException::StopIteration => "StopIteration",
            BuiltinException::RuntimeError => "RuntimeError",
            BuiltinException::Exception => "Exception",
        }
    }
}

/// RPython `SingleDispatchMixin.dispatch = 1` /
/// `DoubleDispatchMixin.dispatch = 2` / `HLOperation.dispatch = None`
/// (operation.py:70-72, 202-203, 258-259).
///
/// Consumed by the annotator to pick a specialisation; flowspace itself
/// only records the classification.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Dispatch {
    /// `HLOperation.dispatch = None` — special-cased op (NewDict,
    /// NewTuple, NewList, NewSlice, Pow, SimpleCall, CallArgs,
    /// Contains, Trunc, Format, Get, Set, Delete, UserDel, Buffer,
    /// Yield, Hint, InplacePow). Annotator dispatches manually.
    None,
    /// `SingleDispatchMixin.dispatch = 1` — dispatch on `args[0]`.
    Single,
    /// `DoubleDispatchMixin.dispatch = 2` — dispatch on
    /// `(args[0], args[1])`.
    Double,
}

impl OpKind {
    /// Inverse of [`Self::opname`]. Returns `None` when `name` is not
    /// one of RPython's canonical `SpaceOperation.opname` strings —
    /// `flowcontext.rs` also emits a few majit-local synthetic op
    /// names (`not_`, `newset`, `buildstr`, `ll_assert_not_none`,
    /// `list_to_tuple`, `set_typeparam_default`) that decline here.
    pub fn from_opname(name: &str) -> Option<OpKind> {
        Some(match name {
            "is_" => OpKind::Is,
            "id" => OpKind::Id,
            "type" => OpKind::Type,
            "issubtype" => OpKind::IsSubtype,
            "isinstance" => OpKind::IsInstance,
            "repr" => OpKind::Repr,
            "str" => OpKind::Str,
            "format" => OpKind::Format,
            "len" => OpKind::Len,
            "hash" => OpKind::Hash,
            "setattr" => OpKind::SetAttr,
            "delattr" => OpKind::DelAttr,
            "getitem" => OpKind::GetItem,
            "getitem_idx" => OpKind::GetItemIdx,
            "setitem" => OpKind::SetItem,
            "delitem" => OpKind::DelItem,
            "getslice" => OpKind::GetSlice,
            "setslice" => OpKind::SetSlice,
            "delslice" => OpKind::DelSlice,
            "trunc" => OpKind::Trunc,
            "pos" => OpKind::Pos,
            "neg" => OpKind::Neg,
            "neg_ovf" => OpKind::NegOvf,
            "bool" => OpKind::Bool,
            "abs" => OpKind::Abs,
            "abs_ovf" => OpKind::AbsOvf,
            "hex" => OpKind::Hex,
            "oct" => OpKind::Oct,
            "bin" => OpKind::Bin,
            "ord" => OpKind::Ord,
            "invert" => OpKind::Invert,
            "add" => OpKind::Add,
            "add_ovf" => OpKind::AddOvf,
            "sub" => OpKind::Sub,
            "sub_ovf" => OpKind::SubOvf,
            "mul" => OpKind::Mul,
            "mul_ovf" => OpKind::MulOvf,
            "truediv" => OpKind::TrueDiv,
            "floordiv" => OpKind::FloorDiv,
            "floordiv_ovf" => OpKind::FloorDivOvf,
            "div" => OpKind::Div,
            "div_ovf" => OpKind::DivOvf,
            "mod" => OpKind::Mod,
            "mod_ovf" => OpKind::ModOvf,
            "divmod" => OpKind::DivMod,
            "lshift" => OpKind::LShift,
            "lshift_ovf" => OpKind::LShiftOvf,
            "rshift" => OpKind::RShift,
            "and_" => OpKind::And,
            "or_" => OpKind::Or,
            "xor" => OpKind::Xor,
            "int" => OpKind::Int,
            "index" => OpKind::Index,
            "float" => OpKind::Float,
            "long" => OpKind::Long,
            "inplace_add" => OpKind::InplaceAdd,
            "inplace_sub" => OpKind::InplaceSub,
            "inplace_mul" => OpKind::InplaceMul,
            "inplace_truediv" => OpKind::InplaceTrueDiv,
            "inplace_floordiv" => OpKind::InplaceFloorDiv,
            "inplace_div" => OpKind::InplaceDiv,
            "inplace_mod" => OpKind::InplaceMod,
            "inplace_pow" => OpKind::InplacePow,
            "inplace_lshift" => OpKind::InplaceLShift,
            "inplace_rshift" => OpKind::InplaceRShift,
            "inplace_and" => OpKind::InplaceAnd,
            "inplace_or" => OpKind::InplaceOr,
            "inplace_xor" => OpKind::InplaceXor,
            "lt" => OpKind::Lt,
            "le" => OpKind::Le,
            "eq" => OpKind::Eq,
            "ne" => OpKind::Ne,
            "gt" => OpKind::Gt,
            "ge" => OpKind::Ge,
            "cmp" => OpKind::Cmp,
            "coerce" => OpKind::Coerce,
            "get" => OpKind::Get,
            "set" => OpKind::Set,
            "delete" => OpKind::Delete,
            "userdel" => OpKind::UserDel,
            "buffer" => OpKind::Buffer,
            "yield_" => OpKind::Yield,
            "newslice" => OpKind::NewSlice,
            "hint" => OpKind::Hint,
            "contains" => OpKind::Contains,
            "newdict" => OpKind::NewDict,
            "newtuple" => OpKind::NewTuple,
            "newlist" => OpKind::NewList,
            "newlist_hint" => OpKind::NewListHint,
            "newstringbuilder" => OpKind::NewStringBuilder,
            "newrustvec" => OpKind::NewRustVec,
            "ll_vec_copy_from_slice" => OpKind::LlVecCopyFromSlice,
            "pow" => OpKind::Pow,
            "iter" => OpKind::Iter,
            "next" => OpKind::Next,
            "getattr" => OpKind::GetAttr,
            "simple_call" => OpKind::SimpleCall,
            "call_args" => OpKind::CallArgs,
            _ => return None,
        })
    }

    /// RPython `cls.opname` (populated by `add_operator(name, …)`).
    ///
    /// Every variant's return value is the exact string used on
    /// `SpaceOperation.opname`, verified by the upstream test
    /// `rpython/flowspace/test/test_objspace.py`.
    pub fn opname(self) -> &'static str {
        match self {
            OpKind::Is => "is_",
            OpKind::Id => "id",
            OpKind::Type => "type",
            OpKind::IsSubtype => "issubtype",
            OpKind::IsInstance => "isinstance",
            OpKind::Repr => "repr",
            OpKind::Str => "str",
            OpKind::Format => "format",
            OpKind::Len => "len",
            OpKind::Hash => "hash",
            OpKind::SetAttr => "setattr",
            OpKind::DelAttr => "delattr",
            OpKind::GetItem => "getitem",
            OpKind::GetItemIdx => "getitem_idx",
            OpKind::SetItem => "setitem",
            OpKind::DelItem => "delitem",
            OpKind::GetSlice => "getslice",
            OpKind::SetSlice => "setslice",
            OpKind::DelSlice => "delslice",
            OpKind::Trunc => "trunc",
            OpKind::Pos => "pos",
            OpKind::Neg => "neg",
            OpKind::NegOvf => "neg_ovf",
            OpKind::Bool => "bool",
            OpKind::Abs => "abs",
            OpKind::AbsOvf => "abs_ovf",
            OpKind::Hex => "hex",
            OpKind::Oct => "oct",
            OpKind::Bin => "bin",
            OpKind::Ord => "ord",
            OpKind::Invert => "invert",
            OpKind::Add => "add",
            OpKind::AddOvf => "add_ovf",
            OpKind::Sub => "sub",
            OpKind::SubOvf => "sub_ovf",
            OpKind::Mul => "mul",
            OpKind::MulOvf => "mul_ovf",
            OpKind::TrueDiv => "truediv",
            OpKind::FloorDiv => "floordiv",
            OpKind::FloorDivOvf => "floordiv_ovf",
            OpKind::Div => "div",
            OpKind::DivOvf => "div_ovf",
            OpKind::Mod => "mod",
            OpKind::ModOvf => "mod_ovf",
            OpKind::DivMod => "divmod",
            OpKind::LShift => "lshift",
            OpKind::LShiftOvf => "lshift_ovf",
            OpKind::RShift => "rshift",
            OpKind::And => "and_",
            OpKind::Or => "or_",
            OpKind::Xor => "xor",
            OpKind::Int => "int",
            OpKind::Index => "index",
            OpKind::Float => "float",
            OpKind::Long => "long",
            OpKind::InplaceAdd => "inplace_add",
            OpKind::InplaceSub => "inplace_sub",
            OpKind::InplaceMul => "inplace_mul",
            OpKind::InplaceTrueDiv => "inplace_truediv",
            OpKind::InplaceFloorDiv => "inplace_floordiv",
            OpKind::InplaceDiv => "inplace_div",
            OpKind::InplaceMod => "inplace_mod",
            OpKind::InplacePow => "inplace_pow",
            OpKind::InplaceLShift => "inplace_lshift",
            OpKind::InplaceRShift => "inplace_rshift",
            OpKind::InplaceAnd => "inplace_and",
            OpKind::InplaceOr => "inplace_or",
            OpKind::InplaceXor => "inplace_xor",
            OpKind::Lt => "lt",
            OpKind::Le => "le",
            OpKind::Eq => "eq",
            OpKind::Ne => "ne",
            OpKind::Gt => "gt",
            OpKind::Ge => "ge",
            OpKind::Cmp => "cmp",
            OpKind::Coerce => "coerce",
            OpKind::Get => "get",
            OpKind::Set => "set",
            OpKind::Delete => "delete",
            OpKind::UserDel => "userdel",
            OpKind::Buffer => "buffer",
            OpKind::Yield => "yield_",
            OpKind::NewSlice => "newslice",
            OpKind::Hint => "hint",

            OpKind::Contains => "contains",
            OpKind::NewDict => "newdict",
            OpKind::NewTuple => "newtuple",
            OpKind::NewList => "newlist",
            OpKind::NewListHint => "newlist_hint",
            OpKind::NewStringBuilder => "newstringbuilder",
            OpKind::NewRustVec => "newrustvec",
            OpKind::LlVecCopyFromSlice => "ll_vec_copy_from_slice",
            OpKind::Pow => "pow",
            OpKind::Iter => "iter",
            OpKind::Next => "next",
            OpKind::GetAttr => "getattr",
            OpKind::SimpleCall => "simple_call",
            OpKind::CallArgs => "call_args",
        }
    }

    /// RPython `cls.arity` — the declared argument count. Matches the
    /// `arity=` argument of `add_operator()` / the explicit `arity`
    /// class attribute on each subclass.
    ///
    /// Returns `None` for variadic / manually dispatched operations
    /// where RPython leaves `arity` unset (`NewDict`, `NewTuple`,
    /// `NewList`, `SimpleCall`, `CallArgs`, `Hint`).
    pub fn arity(self) -> Option<usize> {
        match self {
            // unary
            OpKind::Id
            | OpKind::Type
            | OpKind::Repr
            | OpKind::Str
            | OpKind::Len
            | OpKind::Hash
            | OpKind::Trunc
            | OpKind::Pos
            | OpKind::Neg
            | OpKind::NegOvf
            | OpKind::Bool
            | OpKind::Abs
            | OpKind::AbsOvf
            | OpKind::Hex
            | OpKind::Oct
            | OpKind::Bin
            | OpKind::Ord
            | OpKind::Invert
            | OpKind::Int
            | OpKind::Index
            | OpKind::Float
            | OpKind::Long
            | OpKind::UserDel
            | OpKind::Buffer
            | OpKind::Yield
            | OpKind::Iter
            | OpKind::Next => Some(1),

            // binary
            OpKind::Is
            | OpKind::IsSubtype
            | OpKind::IsInstance
            | OpKind::Format
            | OpKind::DelAttr
            | OpKind::GetItem
            | OpKind::GetItemIdx
            | OpKind::DelItem
            | OpKind::Add
            | OpKind::AddOvf
            | OpKind::Sub
            | OpKind::SubOvf
            | OpKind::Mul
            | OpKind::MulOvf
            | OpKind::TrueDiv
            | OpKind::FloorDiv
            | OpKind::FloorDivOvf
            | OpKind::Div
            | OpKind::DivOvf
            | OpKind::Mod
            | OpKind::ModOvf
            | OpKind::DivMod
            | OpKind::LShift
            | OpKind::LShiftOvf
            | OpKind::RShift
            | OpKind::And
            | OpKind::Or
            | OpKind::Xor
            | OpKind::InplaceAdd
            | OpKind::InplaceSub
            | OpKind::InplaceMul
            | OpKind::InplaceTrueDiv
            | OpKind::InplaceFloorDiv
            | OpKind::InplaceDiv
            | OpKind::InplaceMod
            | OpKind::InplacePow
            | OpKind::InplaceLShift
            | OpKind::InplaceRShift
            | OpKind::InplaceAnd
            | OpKind::InplaceOr
            | OpKind::InplaceXor
            | OpKind::Lt
            | OpKind::Le
            | OpKind::Eq
            | OpKind::Ne
            | OpKind::Gt
            | OpKind::Ge
            | OpKind::Cmp
            | OpKind::Coerce
            | OpKind::Delete
            | OpKind::Contains
            | OpKind::GetAttr => Some(2),

            // ternary
            OpKind::SetAttr
            | OpKind::SetItem
            | OpKind::GetSlice
            | OpKind::DelSlice
            | OpKind::Get
            | OpKind::Set
            | OpKind::NewSlice
            | OpKind::Pow => Some(3),

            // quaternary
            OpKind::SetSlice => Some(4),

            // variadic / manual-dispatch
            OpKind::NewDict
            | OpKind::NewTuple
            | OpKind::NewList
            | OpKind::NewStringBuilder
            | OpKind::NewRustVec
            | OpKind::LlVecCopyFromSlice
            | OpKind::SimpleCall
            | OpKind::CallArgs
            | OpKind::Hint => None,

            // Pyre's synthetic spelling of the one-argument
            // `newlist_hint(sizehint)` ExtRegistry specialization.
            OpKind::NewListHint => Some(1),
        }
    }

    /// RPython `cls.pure` — whether the op is a `PureOperation`
    /// subclass and therefore eligible for `constfold()`.
    ///
    /// Side-effecting ops (setattr/setitem/setslice/call*/inplace*) are
    /// `False`; pure arithmetic / container construction is `True`.
    /// Matches the `pure=` argument of `add_operator` plus the explicit
    /// `PureOperation` base of subclasses like `NewTuple` and `Pow`.
    ///
    /// **`_ovf` siblings are NOT pure.** Upstream
    /// `add_operator(name, ovf=True)` registers BOTH the base op (with
    /// `base_cls = OverflowingOperation`, which extends
    /// `PureOperation`) AND a suffixed twin via
    /// `add_operator(name + '_ovf', arity, dispatch, pyfunc=ovf_func)`
    /// (`operation.py:337-338`). The recursive call carries no
    /// `pure=True` / `ovf=True`, so the suffixed twin's `base_cls`
    /// falls through to plain `HLOperation` (`operation.py`) —
    /// which has the default `constfold(self) -> None`
    /// (`operation.py`). Therefore `_ovf` ops never constfold:
    /// they remain as ops even when all operands are constants and a
    /// constant-time `ovf_func(*args)` would raise. Reading them as
    /// pure (and folding them) inserts a hard-error divergence at
    /// every constant-zero-divisor / overflow site that upstream
    /// would have left to the runtime.
    pub fn pure(self) -> bool {
        match self {
            OpKind::Is
            | OpKind::Type
            | OpKind::IsSubtype
            | OpKind::IsInstance
            | OpKind::Repr
            | OpKind::Str
            | OpKind::Len
            | OpKind::GetItem
            | OpKind::GetItemIdx
            | OpKind::GetSlice
            | OpKind::Pos
            | OpKind::Neg
            | OpKind::Bool
            | OpKind::Abs
            | OpKind::Hex
            | OpKind::Oct
            | OpKind::Bin
            | OpKind::Ord
            | OpKind::Invert
            | OpKind::Add
            | OpKind::Sub
            | OpKind::Mul
            | OpKind::TrueDiv
            | OpKind::FloorDiv
            | OpKind::Div
            | OpKind::Mod
            | OpKind::DivMod
            | OpKind::LShift
            | OpKind::RShift
            | OpKind::And
            | OpKind::Or
            | OpKind::Xor
            | OpKind::Int
            | OpKind::Index
            | OpKind::Float
            | OpKind::Long
            | OpKind::Lt
            | OpKind::Le
            | OpKind::Eq
            | OpKind::Ne
            | OpKind::Gt
            | OpKind::Ge
            | OpKind::Cmp
            | OpKind::Coerce
            | OpKind::Contains
            | OpKind::Get
            | OpKind::Buffer
            | OpKind::NewTuple
            | OpKind::Pow => true,

            // `_ovf` siblings: HLOperation default (no constfold).
            OpKind::NegOvf
            | OpKind::AbsOvf
            | OpKind::AddOvf
            | OpKind::SubOvf
            | OpKind::MulOvf
            | OpKind::FloorDivOvf
            | OpKind::DivOvf
            | OpKind::ModOvf
            | OpKind::LShiftOvf => false,

            // upstream `class GetAttr(SingleDispatchMixin, HLOperation)`
            // (operation.py) is NOT a PureOperation — only its
            // `constfold()` override is wired to fold constant
            // attribute lookups. The Rust port surfaces that via the
            // "custom constfold" eligibility gate in
            // `HLOperation::constfold`, not through `pure()`.
            OpKind::GetAttr
            | OpKind::Id
            | OpKind::Format
            | OpKind::Hash
            | OpKind::SetAttr
            | OpKind::DelAttr
            | OpKind::SetItem
            | OpKind::DelItem
            | OpKind::SetSlice
            | OpKind::DelSlice
            | OpKind::Trunc
            | OpKind::InplaceAdd
            | OpKind::InplaceSub
            | OpKind::InplaceMul
            | OpKind::InplaceTrueDiv
            | OpKind::InplaceFloorDiv
            | OpKind::InplaceDiv
            | OpKind::InplaceMod
            | OpKind::InplacePow
            | OpKind::InplaceLShift
            | OpKind::InplaceRShift
            | OpKind::InplaceAnd
            | OpKind::InplaceOr
            | OpKind::InplaceXor
            | OpKind::Set
            | OpKind::Delete
            | OpKind::UserDel
            | OpKind::Yield
            | OpKind::NewSlice
            | OpKind::Hint
            | OpKind::NewDict
            | OpKind::NewList
            | OpKind::NewListHint
            | OpKind::NewStringBuilder
            | OpKind::NewRustVec
            | OpKind::LlVecCopyFromSlice
            | OpKind::Iter
            | OpKind::Next
            | OpKind::SimpleCall
            | OpKind::CallArgs => false,
        }
    }

    /// RPython `cls.can_overflow` — `True` for `OverflowingOperation`
    /// (operation.py:194-195) variants. These are the BASE ops created
    /// via `add_operator(name, …, ovf=True)` (`operation.py,
    /// 475-477,479-481,483`): `neg`, `abs`, `add`, `sub`, `mul`, `div`,
    /// `floordiv`, `mod`, `lshift`. The `_ovf` siblings created by the
    /// recursive `add_operator(name + '_ovf', arity, dispatch,
    /// pyfunc=ovf_func)` (`operation.py:338`) drop both `pure` and
    /// `ovf` flags, so their base class is plain `HLOperation`
    /// (`operation.py:69 can_overflow = False`).
    pub fn can_overflow(self) -> bool {
        matches!(
            self,
            OpKind::Neg
                | OpKind::Abs
                | OpKind::Add
                | OpKind::Sub
                | OpKind::Mul
                | OpKind::FloorDiv
                | OpKind::Div
                | OpKind::Mod
                | OpKind::LShift
        )
    }

    /// RPython `cls.dispatch` classification used by the annotator to
    /// pick a specialisation.
    pub(crate) fn dispatch(self) -> Dispatch {
        match self {
            // SingleDispatchMixin: `add_operator(…, dispatch=1)`.
            OpKind::Id
            | OpKind::Type
            | OpKind::IsSubtype
            | OpKind::IsInstance
            | OpKind::Repr
            | OpKind::Str
            | OpKind::Len
            | OpKind::Hash
            | OpKind::SetAttr
            | OpKind::DelAttr
            | OpKind::GetSlice
            | OpKind::SetSlice
            | OpKind::DelSlice
            | OpKind::Pos
            | OpKind::Neg
            | OpKind::NegOvf
            | OpKind::Bool
            | OpKind::Abs
            | OpKind::AbsOvf
            | OpKind::Hex
            | OpKind::Oct
            | OpKind::Bin
            | OpKind::Ord
            | OpKind::Invert
            | OpKind::Int
            | OpKind::Float
            | OpKind::Long
            | OpKind::Hint
            | OpKind::Contains
            | OpKind::Iter
            | OpKind::Next
            | OpKind::GetAttr
            | OpKind::SimpleCall
            | OpKind::CallArgs => Dispatch::Single,

            // DoubleDispatchMixin: `add_operator(…, dispatch=2)`.
            OpKind::Is
            | OpKind::GetItem
            | OpKind::GetItemIdx
            | OpKind::SetItem
            | OpKind::DelItem
            | OpKind::Add
            | OpKind::AddOvf
            | OpKind::Sub
            | OpKind::SubOvf
            | OpKind::Mul
            | OpKind::MulOvf
            | OpKind::TrueDiv
            | OpKind::FloorDiv
            | OpKind::FloorDivOvf
            | OpKind::Div
            | OpKind::DivOvf
            | OpKind::Mod
            | OpKind::ModOvf
            | OpKind::LShift
            | OpKind::LShiftOvf
            | OpKind::RShift
            | OpKind::And
            | OpKind::Or
            | OpKind::Xor
            | OpKind::InplaceAdd
            | OpKind::InplaceSub
            | OpKind::InplaceMul
            | OpKind::InplaceTrueDiv
            | OpKind::InplaceFloorDiv
            | OpKind::InplaceDiv
            | OpKind::InplaceMod
            | OpKind::InplaceLShift
            | OpKind::InplaceRShift
            | OpKind::InplaceAnd
            | OpKind::InplaceOr
            | OpKind::InplaceXor
            | OpKind::Lt
            | OpKind::Le
            | OpKind::Eq
            | OpKind::Ne
            | OpKind::Gt
            | OpKind::Ge
            | OpKind::Cmp
            | OpKind::Coerce => Dispatch::Double,

            // manual dispatch / no dispatch=… argument on upstream.
            OpKind::Format
            | OpKind::Trunc
            | OpKind::Index
            | OpKind::InplacePow
            | OpKind::DivMod
            | OpKind::Get
            | OpKind::Set
            | OpKind::Delete
            | OpKind::UserDel
            | OpKind::Buffer
            | OpKind::Yield
            | OpKind::NewSlice
            | OpKind::NewDict
            | OpKind::NewTuple
            | OpKind::NewList
            | OpKind::NewListHint
            | OpKind::NewStringBuilder
            | OpKind::NewRustVec
            | OpKind::LlVecCopyFromSlice
            | OpKind::Pow => Dispatch::None,
        }
    }

    /// RPython `OverflowingOperation.ovf_variant` (operation.py).
    ///
    /// Returns the `_ovf` twin of a checked arithmetic op, or `None`
    /// when the op has no overflow variant.
    pub fn ovf_variant(self) -> Option<OpKind> {
        Some(match self {
            OpKind::Neg => OpKind::NegOvf,
            OpKind::Abs => OpKind::AbsOvf,
            OpKind::Add => OpKind::AddOvf,
            OpKind::Sub => OpKind::SubOvf,
            OpKind::Mul => OpKind::MulOvf,
            OpKind::FloorDiv => OpKind::FloorDivOvf,
            OpKind::Div => OpKind::DivOvf,
            OpKind::Mod => OpKind::ModOvf,
            OpKind::LShift => OpKind::LShiftOvf,
            _ => return None,
        })
    }

    /// RPython `cls.canraise` — populated by the trailing
    /// `_add_exceptions` / `_add_except_ovf` loop
    /// (`operation.py:728-764`).
    ///
    /// Trace order matches upstream, so each arm corresponds to a
    /// specific set of `lis.append(exc)` calls:
    ///
    /// * `op.getitem / getitem_idx / setitem / delitem` line 728-731.
    /// * `op.contains` line 732.
    /// * `_add_exceptions("div mod divmod truediv floordiv pow
    ///   inplace_div inplace_mod inplace_truediv inplace_floordiv
    ///   inplace_pow", ZeroDivisionError)` line 751-753.
    /// * `_add_exceptions("pow inplace_pow lshift inplace_lshift rshift
    ///   inplace_rshift", ValueError)` line 754-755.
    /// * `_add_exceptions("truediv divmod inplace_add inplace_sub
    ///   inplace_mul inplace_truediv inplace_floordiv inplace_div
    ///   inplace_mod inplace_pow inplace_lshift", OverflowError)`
    ///   line 756-759.
    /// * `_add_except_ovf("neg abs add sub mul floordiv div mod
    ///   lshift")` line 760-761 — copies the base canraise onto the
    ///   `_ovf` twin then appends `OverflowError`.
    /// * `_add_exceptions("pow", OverflowError)` line 762-763 — float
    ///   case.
    pub(crate) fn canraise(self) -> &'static [BuiltinException] {
        use BuiltinException::*;
        match self {
            // Explicit HLOperation subclasses with custom canraise.
            OpKind::GetAttr | OpKind::Iter => &[],
            OpKind::Next => &[StopIteration, RuntimeError],
            // `objectmodel.newlist_hint` specializes through
            // `hop.exception_is_here()`; allocation failure is the only
            // exception admitted by the fallible Rust source helper.
            OpKind::NewListHint => &[MemoryError],

            // operation.py:728-731.
            OpKind::GetItem | OpKind::GetItemIdx | OpKind::SetItem | OpKind::DelItem => {
                &[IndexError, KeyError, Exception]
            }
            // operation.py:732.
            OpKind::Contains => &[Exception],

            // `div`: ZeroDivisionError (751). No direct OverflowError.
            // `div_ovf`: `_add_except_ovf` duplicates the base list and
            // appends OverflowError (760).
            OpKind::Div => &[ZeroDivisionError],
            OpKind::DivOvf => &[ZeroDivisionError, OverflowError],
            OpKind::Mod => &[ZeroDivisionError],
            OpKind::ModOvf => &[ZeroDivisionError, OverflowError],
            OpKind::FloorDiv => &[ZeroDivisionError],
            OpKind::FloorDivOvf => &[ZeroDivisionError, OverflowError],
            // `truediv` / `divmod` — ZeroDivisionError (751) + OverflowError
            // (756). No `_ovf` variant in the table.
            OpKind::TrueDiv => &[ZeroDivisionError, OverflowError],
            OpKind::DivMod => &[ZeroDivisionError, OverflowError],

            // `pow` — ZeroDivisionError (751) + ValueError (754) +
            // OverflowError (762, float case).
            OpKind::Pow => &[ZeroDivisionError, ValueError, OverflowError],

            // `lshift`: ValueError (754); `lshift_ovf`: + OverflowError (760).
            OpKind::LShift => &[ValueError],
            OpKind::LShiftOvf => &[ValueError, OverflowError],
            // `rshift`: ValueError (754). No `_ovf`.
            OpKind::RShift => &[ValueError],

            // `_ovf` twins of the pure arithmetic family (760-761).
            // Their base variants have empty canraise.
            OpKind::NegOvf | OpKind::AbsOvf | OpKind::AddOvf | OpKind::SubOvf | OpKind::MulOvf => {
                &[OverflowError]
            }

            // inplace family (753, 754, 758-759). Each upstream line
            // appends one exception; the Rust port flattens the
            // cumulative set.
            OpKind::InplaceAdd => &[OverflowError],
            OpKind::InplaceSub => &[OverflowError],
            OpKind::InplaceMul => &[OverflowError],
            OpKind::InplaceDiv => &[ZeroDivisionError, OverflowError],
            OpKind::InplaceMod => &[ZeroDivisionError, OverflowError],
            OpKind::InplaceFloorDiv => &[ZeroDivisionError, OverflowError],
            OpKind::InplaceTrueDiv => &[ZeroDivisionError, OverflowError],
            OpKind::InplacePow => &[ZeroDivisionError, ValueError, OverflowError],
            OpKind::InplaceLShift => &[ValueError, OverflowError],
            OpKind::InplaceRShift => &[ValueError],

            // Default: upstream leaves `canraise = []`.
            _ => &[],
        }
    }
}

/// RPython `flowspace/operation.py` — `class HLOperation(SpaceOperation)`.
///
/// A high-level operation produced by flow objspace handlers. Each
/// `HLOperation` may be folded to a `Constant` via `constfold()` or
/// recorded as a `SpaceOperation` by `flowcontext.record()`.
#[derive(Clone, Debug)]
pub struct HLOperation {
    /// RPython type-erased identity; matches the Python subclass of
    /// `HLOperation` that would have been instantiated upstream.
    pub kind: OpKind,
    /// RPython `self.args = list(args)` (operation.py).
    pub args: Vec<Hlvalue>,
    /// RPython `self.result = Variable()` (operation.py:75).
    pub result: Variable,
    /// RPython `self.offset = -1` (operation.py:76) — rewritten by
    /// `flowcontext.record()` before the op becomes a `SpaceOperation`.
    pub offset: i64,
}

impl HLOperation {
    /// RPython `HLOperation.__init__(*args)` (operation.py:73-76).
    pub fn new(kind: OpKind, args: Vec<Hlvalue>) -> Self {
        HLOperation {
            kind,
            args,
            result: Variable::new(),
            offset: -1,
        }
    }

    /// RPython `HLOperation.replace(mapping)` (operation.py).
    ///
    /// The mapping is polymorphic (`Variable → Variable | Constant`)
    /// per `flowspace/model.py Variable.replace`. HLOperation.result
    /// is typed `Variable` in upstream too, and RPython's pre-rtyper
    /// stages never rename a Variable into a Constant — the Rust port
    /// keeps the field concrete and asserts that shape.
    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    pub fn replace(&self, mapping: &HashMap<Variable, Hlvalue>) -> HLOperation {
        let newargs: Vec<Hlvalue> = self.args.iter().map(|a| a.replace(mapping)).collect();
        let newresult = match self.result.replace(mapping) {
            Hlvalue::Variable(v) => v,
            Hlvalue::Constant(c) => panic!(
                "HLOperation.replace: Variable result renamed to Constant ({c:?}) \
                 — upstream HLOperation.result is always Variable"
            ),
        };
        HLOperation {
            kind: self.kind,
            args: newargs,
            result: newresult,
            offset: self.offset,
        }
    }

    /// RPython `HLOperation.constfold()` (operation.py, overridden
    /// on PureOperation at 120-132).
    ///
    /// Behaviour matches PureOperation upstream: require every arg to
    /// be foldable (`Constant.foldable()`), then apply the pyfunc
    /// equivalent. Non-pure ops and folds that would overflow (RPython
    /// `type(result) is long` branch at 141-142) return `Ok(None)` so
    /// the caller emits a `SpaceOperation` instead. Flow-time hard
    /// errors (currently the 3-arg `getattr` case and constant module
    /// attribute misses) surface as [`FlowingError`].
    ///
    /// Commit 2 scope: pure integer / float / bool / tuple / comparison
    /// arithmetic — the subset actually exercised by flowspace on
    /// realistic RPython inputs. Non-covered pure ops fall through to
    /// `None` (no fold attempted), which is a strict subset of upstream
    /// behaviour.
    pub fn constfold(&self) -> Result<Option<Hlvalue>, FlowingError> {
        // Pure ops go through PureOperation.constfold (operation.py).
        // A few non-pure ops (`GetAttr`, `Iter`, `Next`) carry their own
        // `constfold()` override upstream and must be allowed past the
        // gate. `SimpleCall` / `CallArgs` / `Pow` handle their own
        // special-casing inside pyfunc() above.
        let eligible =
            self.kind.pure() || matches!(self.kind, OpKind::GetAttr | OpKind::Iter | OpKind::Next);
        if !eligible {
            return Ok(None);
        }
        if self.kind == OpKind::GetAttr {
            return self.constfold_getattr();
        }
        for arg in &self.args {
            match arg {
                Hlvalue::Constant(c) if c.foldable() => {}
                _ => return Ok(None),
            }
        }
        let args: Vec<&ConstValue> = self
            .args
            .iter()
            .map(|a| match a {
                Hlvalue::Constant(c) => &c.value,
                _ => unreachable!("foldable() above excludes Variable args"),
            })
            .collect();
        // RPython `PureOperation.constfold` (operation.py) wraps
        // `pyfunc(*args)` in `try/except Exception` and re-raises the
        // captured exception as `FlowingError(...)` when all args are
        // foldable. Rust `pyfunc` returns `Option<ConstValue>`, conflating
        // "type combo not yet handled" (decline → `Ok(None)`) with
        // "operation deterministically raises" (e.g. division by zero,
        // mixed-type ordering comparison) which upstream surfaces as
        // FlowingError. Detect the deterministic-raise cases explicitly
        // before pyfunc fall-through so the exposed graph shape matches
        // upstream — record_pure_op consumers (build_flow / flowcontext)
        // see the FlowingError instead of silently emitting a runtime-
        // failing SpaceOperation.
        if let Some(err) = constfold_always_raises(self.kind, &args) {
            return Err(err);
        }
        let Some(result) = pyfunc(self.kind, &args) else {
            return Ok(None);
        };
        Ok(Some(Hlvalue::Constant(Constant::new(result))))
    }

    fn constfold_getattr(&self) -> Result<Option<Hlvalue>, FlowingError> {
        // upstream operation.py:624-646:
        //
        //     def constfold(self):
        //         if len(self.args) == 3:
        //             raise FlowingError("getattr() with three arguments not supported: %s" % (self,))
        //         w_obj, w_name = self.args
        //         # handling special things like sys
        //         if (w_obj in NOT_REALLY_CONST and
        //                 w_name not in NOT_REALLY_CONST[w_obj]):
        //             return
        //         if w_obj.foldable() and w_name.foldable():
        //             obj, name = w_obj.value, w_name.value
        //             try:
        //                 result = getattr(obj, name)
        //             except Exception as e:
        //                 raise FlowingError("getattr(%s, %s) always raises %s: %s" % ...)
        //             try:
        //                 return const(result)
        //             except WrapException:
        //                 pass
        if self.args.len() == 3 {
            return Err(FlowingError::new(format!(
                "getattr() with three arguments not supported: {self:?}"
            )));
        }
        let [w_obj_hl, w_name_hl] = self.args.as_slice() else {
            return Ok(None);
        };
        let (Hlvalue::Constant(w_obj), Hlvalue::Constant(w_name)) = (w_obj_hl, w_name_hl) else {
            return Ok(None);
        };
        let Some(name) = w_name.value.as_text() else {
            return Ok(None);
        };
        let name_str = name.to_string();

        // upstream operation.py — NOT_REALLY_CONST guard. In
        // the Rust port the table is keyed on module qualname; the
        // guard only fires for module objects.
        if let ConstValue::HostObject(h) = &w_obj.value
            && h.is_module()
            && not_really_const_declines(h.qualname(), &name_str)
        {
            return Ok(None);
        }

        // upstream operation.py — `if w_obj.foldable() and w_name.foldable()`.
        // Constant-str for `w_name` already satisfies foldable().
        if !w_obj.foldable() {
            return Ok(None);
        }

        match crate::flowspace::model::const_runtime_getattr(&w_obj.value, &name_str) {
            Ok(Some(value)) => Ok(Some(Hlvalue::Constant(Constant::new(value)))),
            // upstream WrapException path — flowspace couldn't wrap the
            // result, so fold declines silently. We report this as "no
            // surface for this attribute" by returning None.
            Ok(None) => Ok(None),
            Err(msg) => Err(FlowingError::new(msg)),
        }
    }

    /// RPython `OverflowingOperation.ovfchecked()` (operation.py)
    /// — returns the `_ovf` twin of this operation, carrying the same
    /// args / result / offset.
    pub fn ovfchecked(&self) -> Option<HLOperation> {
        let ovf_kind = self.kind.ovf_variant()?;
        Some(HLOperation {
            kind: ovf_kind,
            args: self.args.clone(),
            result: self.result.clone(),
            offset: self.offset,
        })
    }

    /// Lower this `HLOperation` into a plain `SpaceOperation` for
    /// `flowcontext.record()`. Mirrors RPython's implicit upcast via
    /// `HLOperation`'s `SpaceOperation` base class.
    pub fn into_space_operation(self) -> SpaceOperation {
        SpaceOperation::with_offset(
            self.kind.opname(),
            self.args,
            Hlvalue::Variable(self.result),
            self.offset,
        )
    }
}

/// RPython `cls.pyfunc(*args)` applied to constant args — returns
/// `Some(result)` when the pyfunc equivalent fires, or `None` when the
/// fold is skipped for a reason upstream also skips (overflow to long,
/// unhandled type combo, unknown op).
///
/// Commit 2 covers the pure arithmetic subset (int / float / bool / str /
/// tuple / comparisons / identity / type-conversion) that realistic
/// flowspace inputs actually fold. Commit 3 extends this with the
/// explicit subclasses that have custom `eval()` (Iter, Next, GetAttr,
/// SimpleCall, CallArgs, Pow).
pub(crate) fn pyfunc(kind: OpKind, args: &[&ConstValue]) -> Option<ConstValue> {
    // variadic / ternary ops that fall outside the fixed-arity
    // match below
    match (kind, args.len()) {
        // RPython `NewTuple` (operation.py). `PureOperation`
        // whose `pyfunc = lambda *args: args`.
        (OpKind::NewTuple, _) => {
            let items: Vec<ConstValue> = args.iter().map(|&v| v.clone()).collect();
            return Some(ConstValue::Tuple(items));
        }
        // RPython `Pow(PureOperation)` (operation.py) with
        // arity 3 — `pyfunc = pow`. We fold only the int/int/None
        // variant; float Pow / int-with-mod defer to runtime emit.
        (OpKind::Pow, 3) => {
            if let [
                ConstValue::Int(base),
                ConstValue::Int(exp),
                ConstValue::None,
            ] = args
            {
                if *exp < 0 {
                    return None;
                }
                let e: u32 = (*exp).try_into().ok()?;
                return base.checked_pow(e).map(ConstValue::Int);
            }
        }
        // RPython `GetAttr.constfold()` (operation.py).
        //
        // upstream shape:
        //   if len(self.args) == 3:
        //       raise FlowingError(...)
        //   w_obj, w_name = self.args
        //   if w_obj in NOT_REALLY_CONST and w_name not in NOT_REALLY_CONST[w_obj]:
        //       return                       # decline fold
        //   if w_obj.foldable() and w_name.foldable():
        //       try:
        //           result = getattr(obj, name)
        //       except Exception as e:
        //           raise FlowingError("%s always raises %s" % …)
        //       return const(result)
        //
        // Gaps documented per CLAUDE.md parity rule #1:
        //
        //  1. 3-arg `getattr(x, name, default)` support — upstream
        //     raises `FlowingError` at flow time, surfacing a hard
        //     error. The Rust port declines the fold here (falls
        //     through to SpaceOperation emission); `flowcontext.rs`
        //     does not yet surface a FlowingError at this site. The
        //     stricter check lands once `HLOperation::constfold`
        //     returns `Result<Option, FlowingError>` — scheduled
        //     alongside the Phase 5 annotator wiring.
        //  2. `NOT_REALLY_CONST` table (operation.py) blocks
        //     folds of `sys.path`, `sys.modules`, etc. while keeping
        //     `sys.maxint`, `sys.exc_info`, etc. foldable — PARITY
        //     with upstream's sys-attribute allowlist via the
        //     [`not_really_const_declines`] helper at the top of this
        //     file. Extra volatile modules gain entries by editing
        //     that helper.
        //  3. The generic `getattr(obj, name)` fall-through that
        //     upstream wraps in `try/except Exception` covers class
        //     methods / instance attributes that flowspace may see on
        //     real Python inputs; the Rust port only folds
        //     `HostObject::Module::module_get`. Extending to class
        //     attribute lookup requires a `HostObject::class_get`
        //     helper that resolves bound methods via
        //     `HOST_ENV.lookup_builtin`.
        (OpKind::GetAttr, 2) => {
            if let [ConstValue::HostObject(obj), name] = args
                && let Some(name) = name.as_text()
                && let Some(value) = obj.module_get(name)
            {
                return Some(ConstValue::HostObject(value));
            }
        }
        (OpKind::GetAttr, 3) => {
            // upstream raises FlowingError at flow time. Without an
            // error channel on constfold() we decline the fold and
            // let the runtime emit a 3-arg `getattr` SpaceOperation;
            // Phase 5's annotator will surface the error when it
            // consumes the op.
            return None;
        }
        _ => {}
    }

    match (kind, args) {
        // unary
        (OpKind::Bool, [v]) => v.truthy().map(ConstValue::Bool),
        (OpKind::Neg, [ConstValue::Int(n)]) => n.checked_neg().map(ConstValue::Int),
        (OpKind::NegOvf, [ConstValue::Int(n)]) => n.checked_neg().map(ConstValue::Int),
        // `base_int.__neg__` returns `self.__class__(-x)`; `unsigned_int.__new__`
        // then `val & MASK` (`rarithmetic.unsigned_int`).
        (OpKind::Neg, [ConstValue::UInt(n)]) => Some(ConstValue::UInt(n.wrapping_neg())),
        (OpKind::Neg, [ConstValue::Float(bits)]) => Some(ConstValue::float(-f64::from_bits(*bits))),
        (OpKind::Pos, [ConstValue::Int(n)]) => Some(ConstValue::Int(*n)),
        (OpKind::Pos, [ConstValue::UInt(n)]) => Some(ConstValue::UInt(*n)),
        (OpKind::Pos, [ConstValue::Float(bits)]) => Some(ConstValue::Float(*bits)),
        (OpKind::Abs, [ConstValue::Int(n)]) => n.checked_abs().map(ConstValue::Int),
        (OpKind::AbsOvf, [ConstValue::Int(n)]) => n.checked_abs().map(ConstValue::Int),
        (OpKind::Abs, [ConstValue::UInt(n)]) => Some(ConstValue::UInt(*n)),
        (OpKind::Abs, [ConstValue::Float(bits)]) => {
            Some(ConstValue::float(f64::from_bits(*bits).abs()))
        }
        (OpKind::Invert, [ConstValue::Int(n)]) => Some(ConstValue::Int(!*n)),
        (OpKind::Invert, [ConstValue::UInt(n)]) => Some(ConstValue::UInt(!*n)),
        (OpKind::Int, [ConstValue::Int(n)]) => Some(ConstValue::Int(*n)),
        (OpKind::Int, [ConstValue::Bool(b)]) => Some(ConstValue::Int(if *b { 1 } else { 0 })),
        // `int(r_uint(n))` is the value as int; above maxint it is a
        // long, which `_pure_result` declines.
        (OpKind::Int, [ConstValue::UInt(n)]) => i64::try_from(*n).ok().map(ConstValue::Int),
        (OpKind::Int, [ConstValue::Float(bits)]) => {
            let f = f64::from_bits(*bits);
            // RPython `int(float)` truncates toward zero and raises
            // OverflowError on infinities/NaN. Rust `as i64` saturates,
            // so guard the range explicitly and decline otherwise.
            if f.is_finite() && f >= (i64::MIN as f64) && f <= (i64::MAX as f64) {
                Some(ConstValue::Int(f as i64))
            } else {
                None
            }
        }
        (OpKind::Float, [ConstValue::Int(n)]) => Some(ConstValue::float(*n as f64)),
        (OpKind::Float, [ConstValue::UInt(n)]) => Some(ConstValue::float(*n as f64)),
        (OpKind::Float, [ConstValue::Float(bits)]) => Some(ConstValue::Float(*bits)),
        (OpKind::Float, [ConstValue::Bool(b)]) => {
            Some(ConstValue::float(if *b { 1.0 } else { 0.0 }))
        }
        (OpKind::Long, [ConstValue::Int(n)]) => Some(ConstValue::Int(*n)),
        // `Long` of `Int` folds to `Int`; `long(r_uint(n))` follows
        // that when `n` fits a signed word, else declines (a long).
        (OpKind::Long, [ConstValue::UInt(n)]) => i64::try_from(*n).ok().map(ConstValue::Int),
        (OpKind::Ord, [ConstValue::ByteStr(s)]) if s.len() == 1 => {
            Some(ConstValue::Int(s[0] as i64))
        }
        (OpKind::Ord, [ConstValue::UniStr(s)]) if s.chars().count() == 1 => {
            s.chars().next().map(|c| ConstValue::Int(c as i64))
        }
        (OpKind::Len, [ConstValue::ByteStr(s)]) => Some(ConstValue::Int(s.len() as i64)),
        (OpKind::Len, [ConstValue::UniStr(s)]) => Some(ConstValue::Int(s.chars().count() as i64)),
        (OpKind::Len, [ConstValue::Tuple(items)]) => Some(ConstValue::Int(items.len() as i64)),
        (OpKind::Len, [ConstValue::List(items)]) => Some(ConstValue::Int(items.len() as i64)),
        (OpKind::Len, [ConstValue::Dict(items)]) => Some(ConstValue::Int(items.len() as i64)),

        // binary identity
        // `operator.is_` compares Python identity. For the value set
        // the Rust port carries on `Constant`, structural equality of
        // `ConstValue` is a safe proxy: two `ConstValue::Int(3)` do
        // compare `==` in Python via small-int cache and we never
        // synthesise two distinct wrappers for the same primitive.
        (OpKind::Is, [a, b]) => Some(ConstValue::Bool(a == b)),

        // binary concat (str / tuple / list)
        // These come BEFORE the numeric arithmetic arms because the
        // generic `(OpKind::Add, [a, b])` dispatch via `fold_arith`
        // declines on non-numeric operand pairs, so concat-aware arms
        // need first chance at variant-specific (UniStr / ByteStr /
        // Tuple / List) operands.
        (OpKind::Add, [ConstValue::ByteStr(a), ConstValue::ByteStr(b)]) => {
            let mut out = a.clone();
            out.extend_from_slice(b);
            Some(ConstValue::ByteStr(out))
        }
        (OpKind::Add, [ConstValue::UniStr(a), ConstValue::UniStr(b)]) => {
            Some(ConstValue::uni_str(format!("{a}{b}")))
        }
        (OpKind::Add, [ConstValue::Tuple(a), ConstValue::Tuple(b)]) => {
            let mut out = a.clone();
            out.extend(b.iter().cloned());
            Some(ConstValue::Tuple(out))
        }
        (OpKind::Add, [ConstValue::List(a), ConstValue::List(b)]) => {
            let mut out = a.as_ref().clone();
            out.extend(b.iter().cloned());
            Some(ConstValue::List(Arc::new(out)))
        }

        // binary arithmetic (numeric, with Python 3 coercion)
        // `_ovf` siblings are NOT pure (`OpKind::pure()` rejects them),
        // so the `pure()` gate in `HLOperation::constfold` short-
        // circuits before reaching this dispatch — matching upstream
        // `add_operator(name + '_ovf', …)` falling through to plain
        // `HLOperation`. The arms below cover only the BASE pure ops.
        //
        // `coerce_arith` mirrors upstream `PureOperation.constfold`'s
        // direct call to `operator.<op>(*args)` — Python's numeric
        // tower coerces `bool ⊂ int ⊂ float` automatically, so
        // `True + 1`, `1 + 1.0`, `True * 2.0` all fold without
        // explicit per-pair arms.
        (OpKind::Add, [a, b]) => coerce_arith(a, b).and_then(|p| match p {
            ArithOps::Int(x, y) => x.checked_add(y).map(ConstValue::Int),
            ArithOps::UInt(x, y) => Some(ConstValue::UInt((x as u64).wrapping_add(y as u64))),
            ArithOps::Float(x, y) => Some(ConstValue::float(x + y)),
        }),
        (OpKind::Sub, [a, b]) => coerce_arith(a, b).and_then(|p| match p {
            ArithOps::Int(x, y) => x.checked_sub(y).map(ConstValue::Int),
            ArithOps::UInt(x, y) => Some(ConstValue::UInt((x as u64).wrapping_sub(y as u64))),
            ArithOps::Float(x, y) => Some(ConstValue::float(x - y)),
        }),
        (OpKind::Mul, [a, b]) => coerce_arith(a, b).and_then(|p| match p {
            ArithOps::Int(x, y) => x.checked_mul(y).map(ConstValue::Int),
            ArithOps::UInt(x, y) => Some(ConstValue::UInt((x as u64).wrapping_mul(y as u64))),
            ArithOps::Float(x, y) => Some(ConstValue::float(x * y)),
        }),
        // `floordiv` and `div` agree on int operands: both fold as
        // `operator.floordiv` (Python 3 floor-toward-`-inf`),
        // implemented by `ll_int_py_div` (rpython/rtyper/rint.py).
        // `int_py_floor_div` ports it line-by-line.
        // `ZeroDivisionError` is surfaced by the always-raises
        // whitelist before this fold runs.
        //
        // They diverge on floats: upstream `add_operator('floordiv',
        // …)` resolves to `operator.floordiv` (returns floor) and
        // `add_operator('div', …)` resolves to `operator.div` (Python 2
        // classic division, which on floats is true division). So
        // `floordiv(7.0, 2.0) == 3.0` but `div(7.0, 2.0) == 3.5`.
        // Folding the two together would silently produce a wrong
        // constant for `div` on floats.
        (OpKind::FloorDiv, [a, b]) => coerce_arith(a, b).and_then(|p| match p {
            ArithOps::Int(x, y) => int_py_floor_div(x, y).map(ConstValue::Int),
            // `base_int.__floordiv__` / `__div__` run Python `//` on the
            // widened longs then `_widen` back to `r_uint` (`MASK`).
            ArithOps::UInt(x, y) => python_floor_div_i128(x, y).map(|q| ConstValue::UInt(q as u64)),
            // Float zero divisor is caught by `constfold_always_raises`
            // (`FlowingError: float floor division by zero`); this arm
            // is only reached for `y != 0.0`. Delegates to
            // `float_py_floor_div` which mirrors upstream
            // `_divmod_w` (`pypy/objspace/std/floatobject.py`).
            ArithOps::Float(x, y) => Some(ConstValue::float(float_py_floor_div(x, y))),
        }),
        (OpKind::Div, [a, b]) => coerce_arith(a, b).and_then(|p| match p {
            ArithOps::Int(x, y) => int_py_floor_div(x, y).map(ConstValue::Int),
            ArithOps::UInt(x, y) => python_floor_div_i128(x, y).map(|q| ConstValue::UInt(q as u64)),
            // `add_operator('div', …)` resolves to Python 2
            // `operator.div`, which on floats is true division
            // (`div(7.0, 2.0) == 3.5`). Zero divisor surfaces via
            // `constfold_always_raises`.
            ArithOps::Float(x, y) => Some(ConstValue::float(x / y)),
        }),
        // `ll_int_py_mod` (rpython/rtyper/rint.py): Python's `%`
        // returns a remainder with the sign of the divisor (`3 % -2
        // == -1`, not `1`). Float mod delegates to `float_py_mod`
        // matching upstream `descr_mod`
        // (`pypy/objspace/std/floatobject.py`), which uses
        // `math_fmod` plus the sign-of-denominator correction and
        // `copysign(0.0, y)` signed-zero output.
        (OpKind::Mod, [a, b]) => coerce_arith(a, b).and_then(|p| match p {
            ArithOps::Int(x, y) => int_py_mod(x, y).map(ConstValue::Int),
            ArithOps::UInt(x, y) => python_mod_i128(x, y).map(|r| ConstValue::UInt(r as u64)),
            ArithOps::Float(x, y) => Some(ConstValue::float(float_py_mod(x, y))),
        }),
        // Python 3 `/` always returns float — even `1 / 2` is `0.5`,
        // not `0`. Integer operands fold via `truediv_int_operands`
        // (`_truediv` / `_bigint_true_divide` divide the integers
        // first and round once).
        (OpKind::TrueDiv, [a, b]) => coerce_arith(a, b).and_then(|p| match p {
            ArithOps::Int(x, y) => truediv_int_operands(x as i128, y as i128),
            ArithOps::UInt(x, y) => truediv_int_operands(x, y),
            ArithOps::Float(x, y) => {
                if y == 0.0 {
                    None
                } else {
                    Some(ConstValue::float(x / y))
                }
            }
        }),
        // Shifts: int-shape only (Python `1 << 1.0` raises TypeError).
        // Negative shift counts unreachable per
        // `constfold_always_raises`. Result type follows the LHS
        // (`base_int.__lshift__` / `__rlshift__`). `r_uint << n` is
        // `r_uint` after MASK, so a count in `64..=i64::MAX` folds to
        // `UInt(0)` (`type(result) is r_uint`, not `long`). A count
        // above `sys.maxsize` raises OverflowError in
        // `base_int.__lshift__` unless the left operand is zero
        // (`W_LongObject._lshift` returns zero before looking at the
        // count). Int `<< n` declines when the exact result does not
        // fit a signed word (`PureOperation._pure_result`:
        // `can_overflow and type(result) is long`), except `Int(0)`
        // (`_lshift`: `if not a: return 0`). Bool-lhs / UInt-rhs
        // uses `base_int.__rlshift__` / `__rrshift__` and rebuilds
        // as `n.__class__` → `bool`. RShift on `r_uint` is logical.
        (OpKind::LShift, [a, b]) => coerce_int_pair(a, b).and_then(|(x, y)| {
            if matches!(a, ConstValue::UInt(_)) {
                if x == 0 {
                    Some(ConstValue::UInt(0))
                } else if y > i64::MAX as i128 {
                    None
                } else if y >= 64 {
                    Some(ConstValue::UInt(0))
                } else {
                    Some(ConstValue::UInt((x as u64).wrapping_shl(y as u32)))
                }
            } else if matches!(a, ConstValue::Bool(_)) && matches!(b, ConstValue::UInt(_)) {
                // `base_int.__rlshift__`: `n.__class__(x << y)` is
                // `bool(result)`. `1 << n` is nonzero for every
                // count that fits `sys.maxsize`; `0 << n` is zero.
                if x == 0 {
                    Some(ConstValue::Bool(false))
                } else if y > i64::MAX as i128 {
                    None
                } else {
                    Some(ConstValue::Bool(true))
                }
            } else if matches!(a, ConstValue::Int(0)) {
                Some(ConstValue::Int(0))
            } else if y >= 64 {
                None
            } else {
                x.checked_shl(y as u32)
                    .and_then(|v| i64::try_from(v).ok())
                    .map(ConstValue::Int)
            }
        }),
        (OpKind::RShift, [a, b]) => coerce_int_pair(a, b).and_then(|(x, y)| {
            if matches!(a, ConstValue::UInt(_)) {
                Some(if y >= 64 {
                    ConstValue::UInt(0)
                } else {
                    ConstValue::UInt((x as u64) >> (y as u32))
                })
            } else if matches!(a, ConstValue::Bool(_)) && matches!(b, ConstValue::UInt(_)) {
                // `base_int.__rrshift__`: `n.__class__(x >> y)` is
                // `bool(result)`.
                Some(ConstValue::Bool(x != 0 && y == 0))
            } else {
                let x = i64::try_from(x).ok()?;
                Some(if y >= 64 {
                    ConstValue::Int(if x < 0 { -1 } else { 0 })
                } else {
                    ConstValue::Int(x >> (y as u32))
                })
            }
        }),
        // Bitwise on (Bool, Bool) keeps Bool: Python `True & True ==
        // True` (bool, not int). All other int-shaped combos
        // (Int / Int, Int / Bool, Bool / Int) widen to Int —
        // `True & 1 == 1` (int). Either operand `r_uint` widens to
        // `r_uint` (`compute_restype`).
        (OpKind::And, [ConstValue::Bool(a), ConstValue::Bool(b)]) => {
            Some(ConstValue::Bool(*a & *b))
        }
        (OpKind::Or, [ConstValue::Bool(a), ConstValue::Bool(b)]) => Some(ConstValue::Bool(*a | *b)),
        (OpKind::Xor, [ConstValue::Bool(a), ConstValue::Bool(b)]) => {
            Some(ConstValue::Bool(*a ^ *b))
        }
        (OpKind::And, [a, b]) => coerce_int_pair(a, b).map(|(x, y)| {
            if matches!(a, ConstValue::UInt(_)) || matches!(b, ConstValue::UInt(_)) {
                ConstValue::UInt((x & y) as u64)
            } else {
                ConstValue::Int((x & y) as i64)
            }
        }),
        (OpKind::Or, [a, b]) => coerce_int_pair(a, b).map(|(x, y)| {
            if matches!(a, ConstValue::UInt(_)) || matches!(b, ConstValue::UInt(_)) {
                ConstValue::UInt((x | y) as u64)
            } else {
                ConstValue::Int((x | y) as i64)
            }
        }),
        (OpKind::Xor, [a, b]) => coerce_int_pair(a, b).map(|(x, y)| {
            if matches!(a, ConstValue::UInt(_)) || matches!(b, ConstValue::UInt(_)) {
                ConstValue::UInt((x ^ y) as u64)
            } else {
                ConstValue::Int((x ^ y) as i64)
            }
        }),

        // comparisons (int / float / str / bool)
        (OpKind::Lt, [a, b]) => cmp_fold(a, b).map(|o| ConstValue::Bool(o.is_lt())),
        (OpKind::Le, [a, b]) => cmp_fold(a, b).map(|o| ConstValue::Bool(o.is_le())),
        (OpKind::Eq, [a, b]) => Some(ConstValue::Bool(python_eq_const(a, b))),
        (OpKind::Ne, [a, b]) => Some(ConstValue::Bool(!python_eq_const(a, b))),
        (OpKind::Gt, [a, b]) => cmp_fold(a, b).map(|o| ConstValue::Bool(o.is_gt())),
        (OpKind::Ge, [a, b]) => cmp_fold(a, b).map(|o| ConstValue::Bool(o.is_ge())),

        // Unhandled combination — decline the fold, let flowcontext
        // emit a SpaceOperation. This is the strict subset of upstream
        // behaviour that Commit 2 covers.
        _ => None,
    }
}

/// Ordering helper for the `Lt / Le / Gt / Ge` fold family. Returns
/// `None` for cross-type comparisons that RPython would raise on (e.g.
/// `int < str`) or for NaN float operands.
///
/// Mixed numeric (Bool ↔ Int ↔ Float) IS folded — Python 3 admits
/// `1 < 1.5`, `True < 2`, `1.0 > False` because `bool` is an `int`
/// subclass and `int`/`float` admit numeric coercion in
/// `operator.lt` etc. (`PureOperation.constfold` calls `pyfunc`
/// directly, so the dispatch follows Python's coercion rules).
pub(crate) fn cmp_fold(a: &ConstValue, b: &ConstValue) -> Option<std::cmp::Ordering> {
    use ConstValue as C;
    match (a, b) {
        (C::Int(x), C::Int(y)) => Some(x.cmp(y)),
        (C::Float(x), C::Float(y)) => f64::from_bits(*x).partial_cmp(&f64::from_bits(*y)),
        (C::ByteStr(x), C::ByteStr(y)) => Some(x.cmp(y)),
        (C::UniStr(x), C::UniStr(y)) => Some(x.cmp(y)),
        (C::Bool(x), C::Bool(y)) => Some(x.cmp(y)),
        // Bool ↔ Int (bool is int subclass).
        (C::Bool(x), C::Int(y)) => Some(i64::from(*x).cmp(y)),
        (C::Int(x), C::Bool(y)) => Some(x.cmp(&i64::from(*y))),
        // Int ↔ Float — exact comparison via `cmp_int_float_exact` (port
        // of `pypy/objspace/std/floatobject.py make_compare_func`
        // / `do_compare_bigint`). Naive `*x as f64` rounds to the
        // nearest representable f64 (mantissa is 53 bits), which would
        // misclassify `Int(2^53 + 1) == Float(2^53)` as equal.
        (C::Int(x), C::Float(b)) => cmp_int_float_exact(*x, f64::from_bits(*b)),
        (C::Float(b), C::Int(x)) => {
            cmp_int_float_exact(*x, f64::from_bits(*b)).map(std::cmp::Ordering::reverse)
        }
        // Bool ↔ Float (`Bool` is `0` or `1`, far below the 2^53
        // mantissa boundary; simple cast is exact).
        (C::Bool(x), C::Float(b)) => (i64::from(*x) as f64).partial_cmp(&f64::from_bits(*b)),
        (C::Float(b), C::Bool(x)) => f64::from_bits(*b).partial_cmp(&(i64::from(*x) as f64)),
        // `r_uint` compares numerically with int/bool (`base_int` is a
        // `long` subclass) and with float via the same bigint path.
        // `r_uint(2**64-1) != -1`.
        (C::UInt(_), _) | (_, C::UInt(_)) => match (numeric_i128(a), numeric_i128(b)) {
            (Some(x), Some(y)) => Some(x.cmp(&y)),
            (Some(x), None) => match b {
                C::Float(bits) => cmp_i128_float_exact(x, f64::from_bits(*bits)),
                _ => None,
            },
            (None, Some(y)) => match a {
                C::Float(bits) => {
                    cmp_i128_float_exact(y, f64::from_bits(*bits)).map(std::cmp::Ordering::reverse)
                }
                _ => None,
            },
            (None, None) => None,
        },
        _ => None,
    }
}

/// Exact ordering between an i64 and an f64. Mirrors
/// `pypy/objspace/std/floatobject.py _compare`'s W_IntObject
/// branch, which routes through `do_compare_bigint` whenever
/// `not int_between(-1, i2 >> 48, 1)` — i.e. when the int's magnitude
/// exceeds the f64 mantissa's exact range.
///
/// Algorithm:
/// 1. NaN: incomparable, return `None`.
/// 2. `|i| <= 2^53`: i is exactly representable as f64; simple cast
///    matches Python's coerced compare.
/// 3. Else: f could be ±inf, an integer-valued f64 outside i64 range,
///    or a finite non-integer. Compare against `floor(f)` exactly.
///    Mirrors upstream's `f1 = math.floor(f1); b1 =
///    rbigint.fromfloat(f1); return b1.lt/le/eq/gt/ge(b2)`.
fn cmp_int_float_exact(i: i64, f: f64) -> Option<std::cmp::Ordering> {
    cmp_i128_float_exact(i as i128, f)
}

fn cmp_i128_float_exact(i: i128, f: f64) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    if f.is_nan() {
        return None;
    }
    // Fast path: |i| within f64's exact-int range (mantissa is 53 bits).
    // `1 << 53` is safe to express as i128 (`< i128::MAX`).
    const MANTISSA_LIMIT: i128 = 1i128 << 53;
    if (-MANTISSA_LIMIT..=MANTISSA_LIMIT).contains(&i) {
        return (i as f64).partial_cmp(&f);
    }
    // |i| > 2^53. Casting `i` to f64 would round; instead reason about
    // f's integer floor and compare against the exact i128.
    if f.is_infinite() {
        return Some(if f > 0.0 {
            Ordering::Less
        } else {
            Ordering::Greater
        });
    }
    // f is finite. `floor(f)` is integer-valued; if it fits in i128,
    // compare directly. Else f's magnitude exceeds i128 representation
    // and the sign decides the outcome.
    let fl = f.floor();
    // i128::MIN as f64 is exact (-2^127). i128::MAX as f64 rounds; use
    // `< (i128::MAX as f64 + 1.0)` as the exclusive upper bound.
    let lo = i128::MIN as f64;
    let hi_excl = (i128::MAX as f64) + 1.0;
    if fl < lo {
        return Some(Ordering::Greater);
    }
    if fl >= hi_excl {
        return Some(Ordering::Less);
    }
    // fl is in i128 range — convert exactly.
    let fl_int = fl as i128;
    if fl == f {
        // f is an integer-valued f64 == fl_int.
        Some(i.cmp(&fl_int))
    } else {
        // f sits strictly between fl_int and fl_int + 1.
        if i <= fl_int {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Greater)
        }
    }
}

/// Promoted operand pair for binary arithmetic in `pyfunc`. Mirrors
/// Python 3's numeric tower (`bool ⊂ int ⊂ float`) plus `r_uint`
/// (`compute_restype`: int/bool yield `r_uint`; float wins). When both
/// operands are int-shaped (`Int` or `Bool`) the result widens to
/// `Int`; either `UInt` widens to `r_uint` (i128 then MASK); when at
/// least one is `Float` both widen to `f64`. Anything else
/// (`UniStr`, `ByteStr`, `HostObject`, …) yields `None`.
pub(crate) enum ArithOps {
    Int(i64, i64),
    UInt(i128, i128),
    Float(f64, f64),
}

/// Numeric coercion helper for [`pyfunc`]'s arithmetic arms. Replicates
/// the cross-type promotions that `operator.<op>(a, b)` would have
/// applied in upstream `PureOperation.constfold` (`operation.py
/// constfold` calling `pyfunc(*args)` directly — Python's numeric
/// tower handles `Bool + Int`, `Int + Float`, etc. without explicit
/// promotion code on the constfold side).
pub(crate) fn coerce_arith(a: &ConstValue, b: &ConstValue) -> Option<ArithOps> {
    use ConstValue as C;
    let int_view = |v: &ConstValue| match v {
        C::Int(n) => Some(*n),
        C::Bool(b) => Some(i64::from(*b)),
        _ => None,
    };
    let float_view = |v: &ConstValue| -> Option<f64> {
        match v {
            C::Int(n) => Some(*n as f64),
            C::UInt(n) => Some(*n as f64),
            C::Bool(b) => Some(i64::from(*b) as f64),
            C::Float(bits) => Some(f64::from_bits(*bits)),
            _ => None,
        }
    };
    match (a, b) {
        (C::Float(_), _) | (_, C::Float(_)) => {
            Some(ArithOps::Float(float_view(a)?, float_view(b)?))
        }
        (C::UInt(_), _) | (_, C::UInt(_)) => {
            Some(ArithOps::UInt(numeric_i128(a)?, numeric_i128(b)?))
        }
        _ => Some(ArithOps::Int(int_view(a)?, int_view(b)?)),
    }
}

/// Int-shaped coercion for shifts and (non-Bool/Bool) bitwise: Bool
/// widens to 0/1, Int/UInt pass through as i128, everything else
/// declines. Float operands are NOT admitted — Python `1 << 1.0`
/// raises TypeError, so the fold has nothing to produce.
pub(crate) fn coerce_int_pair(a: &ConstValue, b: &ConstValue) -> Option<(i128, i128)> {
    Some((numeric_i128(a)?, numeric_i128(b)?))
}

/// Mathematical integer view of `int` / `bool` / `r_uint`.
/// `r_uint(2**64-1)` is `2**64-1`, not `-1`.
fn numeric_i128(v: &ConstValue) -> Option<i128> {
    match v {
        ConstValue::Int(n) => Some(*n as i128),
        ConstValue::UInt(n) => Some(*n as i128),
        ConstValue::Bool(b) => Some(i128::from(*b)),
        _ => None,
    }
}

/// True division of integer operands. `_truediv` /
/// `_bigint_true_divide` divide the exact integers and round once.
/// Casting each operand to f64 first is only exact when both
/// `|operands| <= 2**53` (`DBL_MANT_DIG`); larger magnitudes use a
/// sticky-bit integer quotient so `u128 as f64` (nearest-even) is
/// the correctly rounded result. Zero divisor declines here;
/// `constfold_always_raises` already surfaces it as `FlowingError`.
fn truediv_int_operands(x: i128, y: i128) -> Option<ConstValue> {
    if y == 0 {
        return None;
    }
    const MANTISSA_LIMIT: i128 = 1 << 53;
    if (-MANTISSA_LIMIT..=MANTISSA_LIMIT).contains(&x)
        && (-MANTISSA_LIMIT..=MANTISSA_LIMIT).contains(&y)
    {
        return Some(ConstValue::float((x as f64) / (y as f64)));
    }
    let sign = (x < 0) != (y < 0);
    let a = x.unsigned_abs();
    let b = y.unsigned_abs();
    if a == 0 {
        return Some(ConstValue::float(if sign { -0.0 } else { 0.0 }));
    }
    let bitlen = |n: u128| 128u32 - n.leading_zeros();
    // `a << s` stays below 2**128: every operand here has magnitude
    // `< 2**64`, so `s <= 55 + 64 - 1`.
    let s = (55i32 + bitlen(b) as i32 - bitlen(a) as i32).max(0) as u32;
    let shifted = a << s;
    let mut q = shifted / b;
    let r = shifted % b;
    // Sticky bit below the 53-bit rounding position (`q` has at
    // least 55 bits).
    if r != 0 {
        q |= 1;
    }
    let mut result = (q as f64) * 2f64.powi(-(s as i32));
    if sign {
        result = -result;
    }
    Some(ConstValue::float(result))
}

/// Python floor division (`//`) over i128, then used with `as u64`
/// for `r_uint` MASK wrap (`unsigned_int.__new__`).
fn python_floor_div_i128(x: i128, y: i128) -> Option<i128> {
    if y == 0 {
        return None;
    }
    let q = x / y;
    let r = x % y;
    if r != 0 && (x < 0) != (y < 0) {
        q.checked_sub(1)
    } else {
        Some(q)
    }
}

/// Python `%` over i128 (sign of divisor), then MASK-wrapped for `r_uint`.
fn python_mod_i128(x: i128, y: i128) -> Option<i128> {
    if y == 0 {
        return None;
    }
    let r = x % y;
    if r != 0 && (r < 0) != (y < 0) {
        r.checked_add(y)
    } else {
        Some(r)
    }
}

/// Python-3 floor division over `i64`. Line-by-line port of
/// `rpython/rtyper/rint.py ll_int_py_div`:
///
/// ```text
/// r = trunc_div(x, y)        # C-style truncation toward 0
/// p = r * y
/// u = (p - x) if y < 0 else (x - p)
/// return r + (u >> (BITS - 1))
/// ```
///
/// `(u >> 63)` is `-1` when `u < 0` else `0`, applying the
/// floor-toward-`-inf` correction whenever C's truncated quotient
/// over-shot Python's floor (sign mismatch between dividend and
/// divisor with non-zero remainder).
///
/// Returns `None` for `y == 0` (the always-raises whitelist surfaces
/// `ZeroDivisionError`) and for `x == i64::MIN, y == -1` (Rust's
/// `checked_div` rejects the wrap; the runtime path handles bigint
/// promotion at upstream).
pub(crate) fn int_py_floor_div(x: i64, y: i64) -> Option<i64> {
    let r = x.checked_div(y)?;
    let p = r.checked_mul(y)?;
    let u = if y < 0 {
        p.checked_sub(x)?
    } else {
        x.checked_sub(p)?
    };
    r.checked_add(u >> (i64::BITS - 1))
}

/// Python-3 `%` over `i64`. Line-by-line port of
/// `rpython/rtyper/rint.py ll_int_py_mod`:
///
/// ```text
/// r = trunc_mod(x, y)        # C-style %, sign of dividend
/// u = -r if y < 0 else r
/// return r + (y & (u >> (BITS - 1)))
/// ```
///
/// The `(u >> 63)` mask is `-1` when `u < 0` else `0`; `y & -1 == y`
/// adds the divisor to flip the remainder's sign whenever Python's
/// `(sign of divisor)` rule disagrees with C's `(sign of dividend)`
/// rule.
///
/// `x % -1 == 0` for every `x` upstream — short-circuited before
/// `checked_rem` because `i64::MIN.checked_rem(-1)` returns `None`
/// (the *quotient* `2^63` overflows, even though the remainder `0`
/// itself fits). Without the short-circuit, the well-defined fold
/// `Mod(MIN, -1) == 0` would be silently declined.
///
/// Returns `None` for `y == 0` (caller has already classified
/// that as `ZeroDivisionError`).
pub(crate) fn int_py_mod(x: i64, y: i64) -> Option<i64> {
    if y == -1 {
        return Some(0);
    }
    let r = x.checked_rem(y)?;
    let u = if y < 0 { r.checked_neg()? } else { r };
    r.checked_add(y & (u >> (i64::BITS - 1)))
}

/// Python `float % float` mod result. Line-by-line port of
/// `pypy/objspace/std/floatobject.py W_FloatObject.descr_mod`.
/// Caller must guarantee `y != 0.0` (the zero-divisor branch surfaces
/// as `FlowingError` upstream and is gated by `constfold_always_raises`
/// before this fold runs).
///
/// ```text
/// mod = math_fmod(x, y)
/// if mod:
///     # ensure the remainder has the same sign as the denominator
///     if (y < 0.0) != (mod < 0.0):
///         mod += y
/// else:
///     # the remainder is zero, and in the presence of signed zeroes
///     # fmod returns different results across platforms; ensure
///     # it has the same sign as the denominator
///     mod = math.copysign(0.0, y)
/// ```
///
/// Rust's `%` over `f64` matches C `fmod` semantics (remainder with the
/// sign of the dividend, truncation toward zero), which is what
/// `math_fmod` calls into.
pub(crate) fn float_py_mod(x: f64, y: f64) -> f64 {
    let mut r = x % y;
    if r != 0.0 {
        if (y < 0.0) != (r < 0.0) {
            r += y;
        }
    } else {
        // Use `(0.0).copysign(y)` to give the zero result the sign of
        // the denominator, matching upstream's
        // `mod = math.copysign(0.0, y)`.
        r = (0.0_f64).copysign(y);
    }
    r
}

/// Python `float // float` floor-div result. Line-by-line port of the
/// `floordiv` half of `_divmod_w` at
/// `pypy/objspace/std/floatobject.py`, including the snap-to-
/// nearest-integer pass at `:850-857` that corrects the fp-precision
/// wobble in `(x - mod) / y` (mathematically integral, but the
/// approximation may land just below or above the true value).
/// Caller must guarantee `y != 0.0`.
///
/// ```text
/// mod = math_fmod(x, y)
/// div = (x - mod) / y
/// if mod:
///     if (y < 0.0) != (mod < 0.0):
///         mod += y
///         div -= 1.0
/// else:
///     mod *= mod  # hide "mod = +0" from optimizer
///     if y < 0.0:
///         mod = -mod
/// # snap quotient to nearest integral value
/// if div:
///     floordiv = math.floor(div)
///     if (div - floordiv > 0.5):
///         floordiv += 1.0
/// else:
///     # div is zero - get the same sign as the true quotient
///     div *= div  # hide "div = +0" from optimizers
///     floordiv = div * x / y  # zero w/ sign of vx/wx
/// ```
///
/// `mod` itself is not returned from this helper, but the
/// `div -= 1.0` adjustment in the sign-mismatch branch is observable
/// (it is the difference between e.g. `(-7.0) // 2.0 == -4.0` and the
/// naive `floor(-3.5) == -4.0` agreeing for ordinary ints, but not at
/// fp-precision boundaries).
fn float_py_floor_div(x: f64, y: f64) -> f64 {
    let r = x % y;
    let mut div = (x - r) / y;
    if r != 0.0 && (y < 0.0) != (r < 0.0) {
        div -= 1.0;
    }
    if div != 0.0 {
        let floordiv = div.floor();
        if div - floordiv > 0.5 {
            floordiv + 1.0
        } else {
            floordiv
        }
    } else {
        // div is zero: produce a zero with the sign of `x / y`.
        // `div * div` is `+0.0`, but the optimiser-hide trick from
        // upstream is unnecessary here — we want the sign-of-`x/y` zero
        // result regardless of input zero sign.
        let positive_zero = div * div;
        positive_zero * x / y
    }
}

/// Python-3 `==` semantics over `ConstValue`. Mirrors upstream
/// `PureOperation.constfold` calling `operator.eq(*args)` rather than
/// the host language's variant equality:
///
/// - **Numeric coercion** — Python's `bool` is a subclass of `int`,
///   and `int` / `float` admit cross-type equality via numeric
///   value (`True == 1`, `1 == 1.0`, `True == 1.0`). The naive Rust
///   `ConstValue::PartialEq` derives variant-strict equality and
///   would say those are unequal.
/// - **Distinct types stay false** — `"a" == b"a"`, `1 == "1"`,
///   `None == 0` all return `False` in Python 3 (`BytesWarning` is a
///   diagnostic, not an exception). For these the strict variant
///   compare matches semantics, so we fall through to `a == b`.
/// - **Containers** — `Tuple` / `List` of equal length compare
///   element-wise via the same `python_eq_const`, so
///   `(1, True) == (1, 1)` returns `True` (each numeric element
///   coerces). `Dict` deep-equality is left to variant compare for
///   now: keys hash by variant identity, so `{1: 'a'}` and
///   `{True: 'a'}` already key into different slots, and rebuilding
///   the lookup via Python-equality would require re-keying the
///   whole side-table — left as a follow-up.
pub(crate) fn python_eq_const(a: &ConstValue, b: &ConstValue) -> bool {
    use ConstValue as C;
    match (a, b) {
        (C::Bool(x), C::Bool(y)) => x == y,
        (C::Bool(x), C::Int(y)) | (C::Int(y), C::Bool(x)) => i64::from(*x) == *y,
        (C::Int(x), C::Int(y)) => x == y,
        (C::Bool(x), C::Float(bits)) | (C::Float(bits), C::Bool(x)) => {
            f64::from_bits(*bits) == (i64::from(*x) as f64)
        }
        // Int ↔ Float — route through `cmp_int_float_exact` so ints
        // beyond the 53-bit mantissa boundary compare correctly.
        // Naive `*x as f64` would round `Int(2^53 + 1)` to `2^53`,
        // misclassifying it as equal to `Float(2^53)`. Mirrors upstream
        // `pypy/objspace/std/floatobject.py do_compare_bigint`'s
        // eq branch: a non-integer-valued f64 can never equal an int.
        (C::Int(x), C::Float(bits)) | (C::Float(bits), C::Int(x)) => {
            cmp_int_float_exact(*x, f64::from_bits(*bits))
                .map(|o| o.is_eq())
                .unwrap_or(false)
        }
        // `r_uint` hashes/compares numerically with int/bool
        // (`r_uint(1) == 1 == True`) but `r_uint(2**64-1) != -1`.
        (C::UInt(_), _) | (_, C::UInt(_)) => match (numeric_i128(a), numeric_i128(b)) {
            (Some(x), Some(y)) => x == y,
            (Some(x), None) => match b {
                C::Float(bits) => cmp_i128_float_exact(x, f64::from_bits(*bits))
                    .map(|o| o.is_eq())
                    .unwrap_or(false),
                _ => a == b,
            },
            (None, Some(y)) => match a {
                C::Float(bits) => cmp_i128_float_exact(y, f64::from_bits(*bits))
                    .map(|o| o.is_eq())
                    .unwrap_or(false),
                _ => a == b,
            },
            (None, None) => a == b,
        },
        (C::Float(a), C::Float(b)) => f64::from_bits(*a) == f64::from_bits(*b),
        // Container deep equality — Python compares
        // `tuple` / `list` element-wise via `==` per element. So
        // `(True,) == (1,)` is True even though Bool/Int are
        // distinct variants. Cross-container types stay distinct
        // (`(1,) == [1]` is False in Python 3).
        (C::Tuple(xs), C::Tuple(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(ys.iter()).all(|(x, y)| python_eq_const(x, y))
        }
        (C::List(xs), C::List(ys)) => {
            xs.len() == ys.len() && xs.iter().zip(ys.iter()).all(|(x, y)| python_eq_const(x, y))
        }
        _ => a == b,
    }
}

/// Detect operation/argument combinations that RPython's
/// `PureOperation.constfold` (operation.py) catches as
/// `Exception` and re-raises as `FlowingError`. Returns
/// `Some(FlowingError)` when the combination ALWAYS raises in
/// upstream Python; `None` otherwise (let pyfunc decide whether to
/// fold or decline).
///
/// Covered cases (the audit-flagged subset that surfaces through
/// `record_pure_op` — see `HLOperation::constfold` callsite):
///
/// - **Integer division / modulo by zero** — upstream Python raises
///   `ZeroDivisionError`. Mirrors `Div(int, 0)`, `Mod(int, 0)`,
///   `FloorDiv(int, 0)` (the operator-pair entries from
///   `operation.py add_operator`).
/// - **TrueDiv with int / float zero divisor** — upstream Python
///   raises `ZeroDivisionError` for both `1 / 0` and `1.0 / 0.0`.
///   Mirrors `TrueDiv` registered as `add_operator('truediv', 2,
///   pure=True)`.
/// - **Mixed-type ordering comparison** — upstream Python 3 raises
///   `TypeError("<' not supported between instances of …")` for
///   `Lt`/`Le`/`Gt`/`Ge` between mismatched primitive types.
///   `Eq`/`Ne` are NOT covered: Python 3 returns `False` / `True`
///   for cross-type identity comparison, never raises.
///
/// Other "always raises" combinations (e.g. `len()` of a non-
/// iterable, `int()` of a non-numeric str) are intentionally NOT
/// covered here pending the broader pyfunc signature port — those
/// type-error cases are documented in their pyfunc match arms as
/// "decline" and continue to fall through to runtime emission until
/// pyfunc itself surfaces an error channel.
fn constfold_always_raises(kind: OpKind, args: &[&ConstValue]) -> Option<FlowingError> {
    let make_err = |reason: &str| {
        FlowingError::new(format!(
            "{}{:?} always raises {}",
            kind.opname(),
            args,
            reason,
        ))
    };
    match (kind, args) {
        // ZeroDivisionError on int /, //, %. upstream
        // `PureOperation.constfold` (operation.py) calls
        // `cls.pyfunc(*args)` and re-raises any `Exception` as
        // `FlowingError`. For the BASE ops, that's
        // `operator.{div,floordiv,mod}` which raises
        // `ZeroDivisionError` on a zero divisor.
        //
        // The `_ovf` siblings (DivOvf / FloorDivOvf / ModOvf) are
        // intentionally NOT covered here. Upstream creates them via
        // `add_operator(name + '_ovf', arity, dispatch, pyfunc=ovf_func)`
        // (`operation.py:337-338`) without `pure=True`, so their
        // base class is plain `HLOperation` whose default
        // `constfold(self) -> None` (`operation.py`) leaves
        // them as ops at compile time. Folding them here would
        // surface a hard-error divergence at every constant-zero-
        // divisor site that upstream leaves to the runtime.
        // `OpKind::pure()` enforces this by classifying `_ovf`
        // siblings as non-pure, so this whitelist arm is
        // unreachable for them; the explicit comment is here in
        // case a future caller bypasses the `pure()` gate.
        // Python `bool` is a subclass of `int` (`True == 1`,
        // `False == 0`), so all int-like LHS / int-like zero RHS
        // pairs raise `ZeroDivisionError` upstream regardless of
        // which side is `bool` and which is `int`: `1 // False`,
        // `True % False`, `False // False`, `1 % 0` all share the
        // same path. `is_int_like` admits both `Int(_)` and
        // `Bool(_)` on the LHS; `is_zero_int_like` catches `Int(0)`
        // and `Bool(false)` on the RHS.
        (OpKind::Div | OpKind::FloorDiv | OpKind::Mod, [lhs, rhs])
            if is_int_like(lhs) && is_zero_int_like(rhs) =>
        {
            Some(make_err("ZeroDivisionError: division by zero"))
        }
        // ZeroDivisionError on float div / floordiv / mod. Once any
        // operand is float, upstream `operator.{div,floordiv,mod}`
        // dispatches to the float method (`__truediv__` /
        // `__floordiv__` / `__mod__`) which raises
        // `ZeroDivisionError` with a float-specific message. Catches
        // `7.0 / 0.0`, `7.0 // 0.0`, `7.0 % 0.0`, mixed-type cases
        // (`Float / 0`, `7 // Float(0.0)`), and Bool zero (`True //
        // 0.0`). Without this arm `pyfunc` would decline at the
        // float zero check and downstream `record_pure_op` would
        // silently emit a runtime-failing SpaceOperation that
        // upstream's `try/except Exception` (`operation.py:125-131`)
        // catches at compile time.
        (OpKind::Div | OpKind::FloorDiv | OpKind::Mod, [lhs, rhs])
            if is_foldable_numeric(lhs)
                && is_zero_numeric(rhs)
                && (matches!(lhs, ConstValue::Float(_)) || matches!(rhs, ConstValue::Float(_))) =>
        {
            let reason = match kind {
                OpKind::Div => "ZeroDivisionError: float division by zero",
                OpKind::FloorDiv => "ZeroDivisionError: float floor division by zero",
                OpKind::Mod => "ZeroDivisionError: float modulo",
                _ => unreachable!(),
            };
            Some(make_err(reason))
        }
        // ZeroDivisionError on truediv. Upstream `PureOperation.const
        // fold` calls `operator.truediv(lhs, rhs)`; for numeric lhs
        // and zero rhs the dispatch raises `ZeroDivisionError`. For
        // non-numeric lhs (e.g. `"x" / 0`) Python raises `TypeError`
        // first ("unsupported operand type(s)") before the divisor is
        // even consulted — so we only classify zero-divisor truediv
        // as ZeroDivisionError when lhs is itself foldable-numeric
        // (Int / Bool / Float). For non-numeric lhs we decline here
        // and leave the type-error path to the broader pyfunc port.
        // `Bool(false)` is again caught by `is_zero_int_like` per the
        // `bool ⊂ int` rule.
        //
        // truediv with a float operand uses the float-specific message
        // (`float division by zero`); the all-integer case uses the
        // unified `division by zero`.
        (OpKind::TrueDiv, [lhs, rhs]) if is_foldable_numeric(lhs) && is_zero_numeric(rhs) => {
            let reason =
                if matches!(lhs, ConstValue::Float(_)) || matches!(rhs, ConstValue::Float(_)) {
                    "ZeroDivisionError: float division by zero"
                } else {
                    "ZeroDivisionError: division by zero"
                };
            Some(make_err(reason))
        }
        // TypeError on cross-type ordering comparisons. Eq/Ne deliberately
        // not included — upstream Python 3 returns Bool for those.
        (OpKind::Lt | OpKind::Le | OpKind::Gt | OpKind::Ge, [a, b])
            if cross_type_ordering(a, b) =>
        {
            Some(make_err(
                "TypeError: ordering comparison not supported between distinct primitive types",
            ))
        }
        // TypeError / ValueError on shifts. Upstream does not inspect the
        // RHS first; it calls `operator.lshift/rshift(lhs, rhs)` directly
        // from `PureOperation.constfold` (`operation.py`). That
        // means operand type dispatch wins before the "negative shift
        // count" check: `1 << -1` raises `ValueError`, but `1.0 << -1`
        // raises `TypeError` because float has no shift operation.
        //
        // Classify only carrier types whose Python builtins are known not
        // to implement shifts. `HostObject` is left alone because a user
        // object may define `__lshift__` / `__rshift__`.
        (OpKind::LShift | OpKind::RShift, [lhs, rhs]) if shift_type_error(lhs, rhs) => {
            Some(make_err("TypeError: unsupported operand type(s) for shift"))
        }
        // ValueError on negative shift count for int-shaped operands
        // (`bool` is an `int` subclass). Both base ops are pure;
        // `lshift_ovf` is HLOperation default (`pure=False` per
        // `add_operator(name + '_ovf', …)` at `:337-338`) so this
        // whitelist arm is unreachable for it via the `pure()` gate.
        (OpKind::LShift | OpKind::RShift, [lhs, ConstValue::Int(b)])
            if is_int_like(lhs) && *b < 0 =>
        {
            Some(make_err("ValueError: negative shift count"))
        }
        // OverflowError on an unsigned left shift whose count exceeds
        // `sys.maxsize`. `base_int.__lshift__` widens to Python long
        // first; `x << y` then raises, and `PureOperation.constfold`
        // turns that into `FlowingError`. Counts in `64..=i64::MAX`
        // still fold to `UInt(0)` in `pyfunc`. `W_LongObject._lshift`
        // returns zero when the left operand is zero before looking
        // at the count, so `r_uint(0) << r_uint(2**63)` is `r_uint(0)`.
        (OpKind::LShift, [ConstValue::UInt(x), ConstValue::UInt(n)])
            if *x != 0 && *n > i64::MAX as u64 =>
        {
            Some(make_err("OverflowError: shift count too large"))
        }
        _ => None,
    }
}

/// `true` for `ConstValue` variants that participate in upstream
/// numeric arithmetic (`int` / `bool` / `float`). Used by
/// [`constfold_always_raises`] to guard the truediv zero-divisor
/// classification: only numeric lhs reaches `ZeroDivisionError`
/// — non-numeric lhs raises `TypeError` first per upstream
/// `operator.truediv` dispatch.
pub(crate) fn is_foldable_numeric(v: &ConstValue) -> bool {
    matches!(
        v,
        ConstValue::Int(_) | ConstValue::UInt(_) | ConstValue::Bool(_) | ConstValue::Float(_)
    )
}

/// Helper for [`constfold_always_raises`]: returns `true` when `v` is
/// the integer zero in the Python `bool ⊂ int` sense — `Int(0)`,
/// `UInt(0)`, or `Bool(false)`. Mirrors upstream Python's
/// `0 == False == r_uint(0)` numeric equality at the divisor slot.
fn is_zero_int_like(v: &ConstValue) -> bool {
    matches!(
        v,
        ConstValue::Int(0) | ConstValue::UInt(0) | ConstValue::Bool(false)
    )
}

/// Helper for [`constfold_always_raises`]: returns `true` when `v` is
/// any integer-shaped operand under Python's `bool ⊂ int` rule —
/// `Int(_)` or `Bool(_)` or `r_uint`. Used at the LHS slot of
/// integer-only operations (Div / FloorDiv / Mod) so `True // False`
/// and `r_uint(1) // 0` participate in the same zero-divisor whitelist
/// as `1 // 0`.
fn is_int_like(v: &ConstValue) -> bool {
    matches!(
        v,
        ConstValue::Int(_) | ConstValue::UInt(_) | ConstValue::Bool(_)
    )
}

/// Helper for [`constfold_always_raises`]: returns `true` when `v` is
/// numerically zero across the int / bool / float rungs of Python's
/// numeric tower — `Int(0)`, `Bool(false)`, or `Float(0.0)`. Used to
/// catch zero divisors regardless of which numeric type carries the
/// zero (`7.0 // 0`, `7 % 0.0`, `True / 0.0` all share the same
/// `ZeroDivisionError` upstream).
fn is_zero_numeric(v: &ConstValue) -> bool {
    is_zero_int_like(v) || matches!(v, ConstValue::Float(bits) if f64::from_bits(*bits) == 0.0)
}

/// Helper for [`constfold_always_raises`]: returns `true` when
/// `operator.lshift/rshift(lhs, rhs)` would fail during operand type
/// dispatch before any shift-count check. Mirrors Python's shift
/// contract: only int-shaped operands (`int` / `bool`) are admitted by
/// the builtin numeric path.
fn shift_type_error(lhs: &ConstValue, rhs: &ConstValue) -> bool {
    fn known_non_shift_operand(v: &ConstValue) -> bool {
        matches!(
            v,
            ConstValue::Float(_)
                | ConstValue::ByteStr(_)
                | ConstValue::UniStr(_)
                | ConstValue::Tuple(_)
                | ConstValue::List(_)
                | ConstValue::Dict(_)
                | ConstValue::None
        )
    }
    (known_non_shift_operand(lhs) || known_non_shift_operand(rhs))
        && !(is_int_like(lhs) && is_int_like(rhs))
}

/// Helper for [`constfold_always_raises`]: returns `true` when `a`
/// and `b` are foldable primitives whose Python-3 cross-type
/// ordering would raise `TypeError` (e.g. `Int <-> UniStr`,
/// `ByteStr <-> Int`). Same-type, NaN-float, and Eq/Ne paths return
/// `false` (handled elsewhere or not raise).
fn cross_type_ordering(a: &ConstValue, b: &ConstValue) -> bool {
    use ConstValue as C;
    let kind = |v: &ConstValue| match v {
        C::Int(_) => Some(0),
        C::UInt(_) => Some(0), // r_uint is a long subclass — same kind
        C::Bool(_) => Some(0), // Python: bool is int — same kind
        C::Float(_) => Some(1),
        C::UniStr(_) => Some(2),
        C::ByteStr(_) => Some(3),
        _ => None,
    };
    match (kind(a), kind(b)) {
        (Some(ka), Some(kb)) => {
            // Int<->Float same-numeric kind (Python 3 admits int<->float
            // ordering); UniStr<->ByteStr distinct (Python 3 raises).
            (ka, kb) != (0, 1) && (ka, kb) != (1, 0) && ka != kb
        }
        _ => false,
    }
}

// Dispatcher plumbing (operation.py:66-300 + pairtype.py).  allow-line-citation
//
// Upstream `class SingleDispatchMixin` / `class DoubleDispatchMixin`
// store the registration table on the HLOperation subclass itself
// (`cls._registry`). The Rust port collapses all HLOperation subclasses
// into `OpKind`, so the per-class registries become a `HashMap<OpKind,
// ...>` keyed on the op identity.

use std::cell::RefCell;

use crate::tool::pairtype::DoubleDispatchRegistry;

/// RPython `specialized` closure value returned by
/// `get_specialization` (operation.py / 273-278).
///
/// Upstream this is a Python closure; it carries an optional
/// `can_only_throw` attribute that annotator `read_can_only_throw`
/// (model.py:837-841) consults. Rust packages the callable and the
/// attribute side-by-side.
pub(crate) struct Specialization {
    /// The actual annotation handler — `spec(annotator, *self.args)`
    /// upstream (operation.py:104).
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    pub apply: Box<
        dyn Fn(
            &crate::annotator::annrpython::RPythonAnnotator,
            &HLOperation,
        ) -> Option<crate::annotator::model::SomeValue>,
    >,
    /// RPython `specialized.can_only_throw = impl.can_only_throw`
    /// side-band (operation.py:234).
    pub can_only_throw: CanOnlyThrow,
}

fn annotator_error_from_panic(
    payload: &(dyn std::any::Any + Send),
) -> Option<crate::annotator::model::AnnotatorError> {
    fn parse_text(text: &str) -> Option<crate::annotator::model::AnnotatorError> {
        let suffix = text.strip_prefix("AnnotatorError")?;
        let msg = suffix.trim_start_matches([':', ' ']);
        Some(crate::annotator::model::AnnotatorError::new(msg))
    }

    payload
        .downcast_ref::<crate::annotator::model::AnnotatorError>()
        .cloned()
        .or_else(|| {
            payload
                .downcast_ref::<String>()
                .and_then(|text| parse_text(text))
        })
        .or_else(|| {
            payload
                .downcast_ref::<&'static str>()
                .and_then(|text| parse_text(text))
        })
}

pub(crate) fn apply_specialization(
    spec: &Specialization,
    annotator: &crate::annotator::annrpython::RPythonAnnotator,
    hlop: &HLOperation,
) -> Result<Option<crate::annotator::model::SomeValue>, crate::annotator::model::AnnotatorError> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (spec.apply)(annotator, hlop)
    })) {
        Ok(value) => Ok(value),
        Err(payload) => match annotator_error_from_panic(payload.as_ref()) {
            Some(err) => Err(err),
            None => std::panic::resume_unwind(payload),
        },
    }
}

/// Adapt a total annotation handler — one that always yields a concrete
/// `SomeValue` — into the `Optional[SomeObject]` shape `op.consider`
/// returns (operation.py:104). This is the common case: most handlers
/// always produce an annotation. Void handlers (`setattr` / `delattr`,
/// whose upstream methods `return None`) and explicit-block handlers
/// register a plain `Box::new` closure that returns `None` or
/// `Some(s_ImpossibleValue)` directly instead of going through `pure`.
#[expect(
    clippy::type_complexity,
    reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
)]
pub(crate) fn pure(
    f: impl Fn(
        &crate::annotator::annrpython::RPythonAnnotator,
        &HLOperation,
    ) -> crate::annotator::model::SomeValue
    + 'static,
) -> Box<
    dyn Fn(
        &crate::annotator::annrpython::RPythonAnnotator,
        &HLOperation,
    ) -> Option<crate::annotator::model::SomeValue>,
> {
    Box::new(move |ann, hl| Some(f(ann, hl)))
}

/// RPython `getattr(opimpl, 'can_only_throw', None)` polymorphism
/// (model.py:837-841).
///
/// Upstream the attribute is either absent, a list of exception
/// classes, or a callable that produces one. Rust models the three
/// branches explicitly.
pub enum CanOnlyThrow {
    /// Attribute absent — upstream `None`.
    Absent,
    /// `can_only_throw = [Exc, Exc, ...]` — upstream line 839
    /// `isinstance(can_only_throw, list)` branch.
    List(Vec<BuiltinException>),
    /// `can_only_throw = lambda *args: [...]` — upstream line 841
    /// `return can_only_throw(*args)` branch. Returns `None` to mirror
    /// `_dict_can_only_throw_*` helpers (binaryop.py _dict_can_only_throw_keyerror) that
    /// defer to `op.canraise` for r_dict's unrestricted throw set.
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    Callable(Box<dyn Fn(&[crate::annotator::model::SomeValue]) -> Option<Vec<BuiltinException>>>),
}

/// RPython `@op.<name>.register_transform(...)` handler signature.
///
/// Upstream transformers take `(annotator, *self.args)` and return
/// either `None` (no rewrite) or a Python list of `HLOperation`s that
/// replace the current op. The Rust port mirrors that shape.
///
/// Lives on a per-OpKind, per-SomeValue-kind registry (`_TRANSFORM_*`)
/// that `HLOperation::transform` looks up before returning.
pub(crate) type Transformation = Box<
    dyn Fn(
        &crate::annotator::annrpython::RPythonAnnotator,
        &[crate::flowspace::model::Hlvalue],
    ) -> Option<Vec<HLOperation>>,
>;

thread_local! {
    /// RPython `cls._registry` on `SingleDispatchMixin` (operation.py).
    ///
    /// Upstream: one dict per HLOperation subclass, keyed by the argument
    /// class (`Some_cls`). Lookup walks `Some_cls.__mro__`
    /// (operation.py). Rust collapses "per HLOperation subclass"
    /// into an outer `HashMap<OpKind, ...>` since OpKind replaces the
    /// class identity of the HLOperation subclass. The registry is
    /// `thread_local!` because its contents include non-Send `Rc`
    /// references and because RPython's annotator is single-threaded.
    pub static _REGISTRY_SINGLE: RefCell<
        std::collections::HashMap<
            OpKind,
            std::collections::HashMap<
                crate::annotator::model::SomeValueTag,
                Specialization,
            >,
        >,
    > = {
        let mut outer = std::collections::HashMap::new();
        crate::annotator::unaryop::init(&mut outer);
        RefCell::new(outer)
    };

    /// RPython `cls._registry = DoubleDispatchRegistry()` on
    /// `DoubleDispatchMixin` (operation.py). Per-OpKind pair registry
    /// using the [`DoubleDispatchRegistry`] ported from
    /// `rpython/tool/pairtype.py`. Initialized on first access by
    /// calling the module-import-time `init` helpers.
    pub static _REGISTRY_DOUBLE: RefCell<
        std::collections::HashMap<
            OpKind,
            DoubleDispatchRegistry<
                crate::annotator::model::SomeValueTag,
                crate::annotator::model::SomeValueTag,
                Specialization,
            >,
        >,
    > = {
        let mut outer = std::collections::HashMap::new();
        crate::annotator::binaryop::init(&mut outer);
        RefCell::new(outer)
    };

    /// RPython `cls._transform = {}` on `HLOperationMeta.__init__`
    /// (operation.py:60 for `dispatch == 1`). Keyed the same way as
    /// `_REGISTRY_SINGLE` — MRO lookup on `type(args_s[0])`.
    pub static _TRANSFORM_SINGLE: RefCell<
        std::collections::HashMap<
            OpKind,
            std::collections::HashMap<crate::annotator::model::SomeValueTag, Transformation>,
        >,
    > = {
        let mut outer = std::collections::HashMap::new();
        crate::annotator::unaryop::init_transform(&mut outer);
        RefCell::new(outer)
    };

    /// RPython `cls._transform = DoubleDispatchRegistry()` on
    /// `HLOperationMeta.__init__` (operation.py:63 for
    /// `dispatch == 2`). Same MRO cross-product lookup as
    /// `_REGISTRY_DOUBLE`.
    pub static _TRANSFORM_DOUBLE: RefCell<
        std::collections::HashMap<
            OpKind,
            DoubleDispatchRegistry<
                crate::annotator::model::SomeValueTag,
                crate::annotator::model::SomeValueTag,
                Transformation,
            >,
        >,
    > = {
        let mut outer = std::collections::HashMap::new();
        crate::annotator::binaryop::init_transform(&mut outer);
        RefCell::new(outer)
    };
}

impl HLOperation {
    /// RPython `HLOperation.consider(self, annotator)` (operation.py).
    ///
    /// ```python
    /// def consider(self, annotator):
    ///     args_s = [annotator.annotation(arg) for arg in self.args]
    ///     spec = type(self).get_specialization(*args_s)
    ///     return spec(annotator, *self.args)
    /// ```
    ///
    /// The Rust port splits on `self.kind.dispatch()` to pick the
    /// correct `get_specialization` path — upstream selects the same
    /// paths via MRO dispatch (`SingleDispatchMixin.get_specialization`
    /// / `DoubleDispatchMixin.get_specialization`).
    pub fn consider(
        &self,
        annotator: &crate::annotator::annrpython::RPythonAnnotator,
    ) -> Result<Option<crate::annotator::model::SomeValue>, crate::annotator::model::AnnotatorError>
    {
        use crate::annotator::model::{AnnotatorError, SomeValue};
        // upstream operation.py:101-104 —
        //     args_s = [annotator.annotation(arg) for arg in self.args]
        //     spec = type(self).get_specialization(*args_s)
        //     return spec(annotator, *self.args)
        //
        // TODO(consider-none-arg-propagation) — STRICT-PARITY DIVERGENCE.
        // Upstream `operation.py:101` collects `annotation(arg)` as a
        // Python list where `None` slots are kept intact;
        // `SingleDispatchMixin._dispatch` (`operation.py`)
        // walks `type(None).__mro__` so unbound args reach a
        // SomeObject-tagged spec.  `simple_call`
        // (`operation.py:663` `simple_call_SomeObject`) and
        // `unaryop.py immutablevalue` also tolerate None mid-
        // fixpoint.  Pyre's `Vec<SomeValue>` shape forces eager
        // unwrap before dispatch and raises "unbound argument"; this
        // closes off the Option<SomeValue> propagation epic at
        // tag/MRO time.  Converging needs a NoneType lattice tag and
        // a `Vec<Option<SomeValue>>` carrier through every spec arm
        // — this touches every binding registered in
        // `_REGISTRY_SINGLE` / `_REGISTRY_DOUBLE`).
        let mut args_s: Vec<SomeValue> = Vec::with_capacity(self.args.len());
        for a in &self.args {
            let s = annotator.annotation(a).ok_or_else(|| {
                AnnotatorError::new(format!(
                    "consider({:?}): unbound argument {:?}",
                    self.kind, a
                ))
            })?;
            args_s.push(s);
        }
        match self.kind.dispatch() {
            Dispatch::Single => {
                let tag = args_s
                    .first()
                    .ok_or_else(|| {
                        AnnotatorError::new(format!(
                            "consider: dispatch=1 op {:?} with 0 args",
                            self.kind
                        ))
                    })?
                    .tag();
                let result =
                    _REGISTRY_SINGLE.with(|cell| -> Result<Option<SomeValue>, AnnotatorError> {
                        let reg = cell.borrow();
                        let entries = reg.get(&self.kind).ok_or_else(|| {
                            AnnotatorError::new(format!(
                                "consider: no single-dispatch entries for {:?}",
                                self.kind
                            ))
                        })?;
                        // Upstream `SingleDispatchMixin._dispatch`
                        // (operation.py:212-219) walks
                        // `type(s_arg).__mro__`.
                        for c in tag.mro() {
                            if let Some(spec) = entries.get(c) {
                                return apply_specialization(spec, annotator, self);
                            }
                        }
                        Err(AnnotatorError::new(format!(
                            "consider: no unary spec for {:?}({:?})",
                            self.kind, tag
                        )))
                    })?;
                Ok(result)
            }
            Dispatch::Double => {
                let tag_l = args_s
                    .first()
                    .ok_or_else(|| {
                        AnnotatorError::new(format!(
                            "consider: dispatch=2 op {:?} with 0 args",
                            self.kind
                        ))
                    })?
                    .tag();
                let tag_r = args_s
                    .get(1)
                    .ok_or_else(|| {
                        AnnotatorError::new(format!(
                            "consider: dispatch=2 op {:?} with 1 arg",
                            self.kind
                        ))
                    })?
                    .tag();
                let result =
                    _REGISTRY_DOUBLE.with(|cell| -> Result<Option<SomeValue>, AnnotatorError> {
                        let reg = cell.borrow();
                        let entries = reg.get(&self.kind).ok_or_else(|| {
                            AnnotatorError::new(format!(
                                "consider: no double-dispatch entries for {:?}",
                                self.kind
                            ))
                        })?;
                        match entries.get((tag_l, tag_r), tag_l.mro(), tag_r.mro()) {
                            Some(spec) => apply_specialization(spec, annotator, self),
                            None => Err(AnnotatorError::new(format!(
                                "consider: no binary spec for {:?}({:?}, {:?})",
                                self.kind, tag_l, tag_r
                            ))),
                        }
                    })?;
                Ok(result)
            }
            Dispatch::None => {
                // operation.py — per-class `consider()`
                // overrides on explicit `Dispatch::None` subclasses.
                use crate::annotator::model::SomeTuple as AnSomeTuple;
                match self.kind {
                    // operation.py — NewDict.consider.
                    OpKind::NewDict => Ok(Some(SomeValue::Dict(annotator.bookkeeper.newdict()))),
                    // operation.py — NewTuple.consider.
                    OpKind::NewTuple => Ok(Some(SomeValue::Tuple(AnSomeTuple::new(args_s)))),
                    // operation.py — NewList.consider.
                    OpKind::NewList => {
                        let list = annotator.bookkeeper.newlist(&args_s, None)?;
                        Ok(Some(SomeValue::List(list)))
                    }
                    // objectmodel.py `Entry.compute_result_annotation`:
                    // create an empty list and mark it resized.  The size
                    // hint affects allocation only, not the item annotation.
                    OpKind::NewListHint => {
                        let list = annotator.bookkeeper.newlist(&[], None)?;
                        list.listdef.resize()?;
                        Ok(Some(SomeValue::List(list)))
                    }
                    // `StringBuilder()` construction — the ExtRegistryEntry
                    // `compute_result_annotation` yields `SomeStringBuilder`
                    // (rlib/rstring.py).  The optional `init_size` arg (an
                    // `SomeInteger`) does not refine the result.
                    OpKind::NewStringBuilder => Ok(Some(SomeValue::StringBuilder(
                        crate::annotator::model::SomeStringBuilder::new(),
                    ))),
                    // `lltype.py` `malloc` / `ann_malloc` takes a constant
                    // lowleveltype and annotates from it. `llannotation.py`
                    // `lltype_to_annotation` maps an integer item:
                    // `Unsigned` → `SomeInteger(unsigned=True)`, `Signed` →
                    // `SomeInteger()`. Float and reference items stay
                    // `SomeRustVec::for_kind`. The result concretetype is
                    // not an annotation input.
                    OpKind::NewRustVec | OpKind::LlVecCopyFromSlice => {
                        let bad_item = || {
                            AnnotatorError::new(format!(
                                "{}: argument 0 must be a constant item lowleveltype",
                                self.kind.opname()
                            ))
                        };
                        let item = match self.args.first() {
                            Some(Hlvalue::Constant(c)) => match &c.value {
                                crate::flowspace::model::ConstValue::LowLevelType(ll) => {
                                    ll.as_ref().clone()
                                }
                                _ => return Err(bad_item()),
                            },
                            _ => return Err(bad_item()),
                        };
                        let kind = crate::translator::rtyper::rrustvec::vec_item_kind_of(&item)
                            .ok_or_else(bad_item)?;
                        let s_item = if kind == majit_ir::rvec::VecItemKind::Int {
                            crate::translator::rtyper::llannotation::lltype_to_annotation(item)
                        } else {
                            *crate::annotator::model::SomeRustVec::for_kind(kind).s_item
                        };
                        Ok(Some(SomeValue::RustVec(
                            crate::annotator::model::SomeRustVec::new(s_item),
                        )))
                    }
                    // operation.py — NewSlice.consider raises
                    // AnnotatorError outright.
                    OpKind::NewSlice => Err(AnnotatorError::new(
                        "Cannot use extended slicing in rpython",
                    )),
                    // Unregistered HLOperation subclasses with
                    // `dispatch=None`. Upstream expects these to have
                    // been constfolded. Surface the gap as a proper
                    // error so keepgoing mode can continue.
                    _ => Err(AnnotatorError::new(format!(
                        "consider: no Dispatch::None override for {:?}",
                        self.kind
                    ))),
                }
            }
        }
    }

    /// RPython `HLOperation.transform(self, annotator)` (operation.py).
    ///
    /// ```python
    /// def transform(self, annotator):
    ///     args_s = [annotator.annotation(arg) for arg in self.args]
    ///     transformer = self.get_transformer(*args_s)
    ///     return transformer(annotator, *self.args)
    /// ```
    ///
    /// `get_transformer` does an MRO lookup (operation.py for
    /// single-dispatch, 295-300 for double-dispatch) and returns the
    /// default `lambda *args: None` when no registration matches.
    pub fn transform(
        &self,
        annotator: &crate::annotator::annrpython::RPythonAnnotator,
    ) -> Option<Vec<HLOperation>> {
        use crate::annotator::model::SomeValue;
        let args_s: Vec<SomeValue> = self
            .args
            .iter()
            .filter_map(|a| annotator.annotation(a))
            .collect();
        match self.kind.dispatch() {
            Dispatch::Single => {
                if args_s.is_empty() {
                    return None;
                }
                let tag = args_s[0].tag();
                _TRANSFORM_SINGLE.with(|cell| {
                    let reg = cell.borrow();
                    let entries = reg.get(&self.kind)?;
                    for c in tag.mro() {
                        if let Some(tx) = entries.get(c) {
                            return tx(annotator, &self.args);
                        }
                    }
                    None
                })
            }
            Dispatch::Double => {
                if args_s.len() < 2 {
                    return None;
                }
                let tag_l = args_s[0].tag();
                let tag_r = args_s[1].tag();
                _TRANSFORM_DOUBLE.with(|cell| {
                    let reg = cell.borrow();
                    let entries = reg.get(&self.kind)?;
                    let tx = entries.get((tag_l, tag_r), tag_l.mro(), tag_r.mro())?;
                    tx(annotator, &self.args)
                })
            }
            Dispatch::None => None,
        }
    }

    /// RPython `HLOperation.get_can_only_throw(self, annotator)`
    /// (operation.py, SingleDispatchMixin:221-224,
    /// DoubleDispatchMixin:283-286).
    pub(crate) fn get_can_only_throw(
        &self,
        annotator: &crate::annotator::annrpython::RPythonAnnotator,
    ) -> Option<Vec<BuiltinException>> {
        use crate::annotator::model::SomeValue;
        let args_s: Vec<SomeValue> = self
            .args
            .iter()
            .filter_map(|a| annotator.annotation(a))
            .collect();
        match self.kind.dispatch() {
            Dispatch::Single => {
                if args_s.is_empty() {
                    return None;
                }
                let tag = args_s[0].tag();
                _REGISTRY_SINGLE.with(|cell| {
                    let reg = cell.borrow();
                    let entries = reg.get(&self.kind)?;
                    for c in tag.mro() {
                        if let Some(spec) = entries.get(c) {
                            return crate::annotator::model::read_can_only_throw(
                                &spec.can_only_throw,
                                &args_s,
                            );
                        }
                    }
                    None
                })
            }
            Dispatch::Double => {
                if args_s.len() < 2 {
                    return None;
                }
                let tag_l = args_s[0].tag();
                let tag_r = args_s[1].tag();
                _REGISTRY_DOUBLE.with(|cell| {
                    let reg = cell.borrow();
                    let entries = reg.get(&self.kind)?;
                    entries
                        .get((tag_l, tag_r), tag_l.mro(), tag_r.mro())
                        .and_then(|spec| {
                            crate::annotator::model::read_can_only_throw(
                                &spec.can_only_throw,
                                &args_s,
                            )
                        })
                })
            }
            // Upstream `HLOperation.get_can_only_throw` default
            // (operation.py:106-107) returns None.
            Dispatch::None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::model::{ConstValue, Constant};
    use super::*;

    #[test]
    fn opname_matches_upstream_strings() {
        // A curated selection: one entry per upstream RPython string
        // that carries an underscore or a non-obvious mapping.
        assert_eq!(OpKind::Is.opname(), "is_");
        assert_eq!(OpKind::And.opname(), "and_");
        assert_eq!(OpKind::Or.opname(), "or_");
        assert_eq!(OpKind::Yield.opname(), "yield_");
        assert_eq!(OpKind::GetItemIdx.opname(), "getitem_idx");
        assert_eq!(OpKind::SimpleCall.opname(), "simple_call");
        assert_eq!(OpKind::CallArgs.opname(), "call_args");
        assert_eq!(OpKind::AddOvf.opname(), "add_ovf");
        assert_eq!(OpKind::IsSubtype.opname(), "issubtype");
        assert_eq!(OpKind::IsInstance.opname(), "isinstance");
    }

    #[test]
    fn ovf_variant_matches_upstream_table() {
        // upstream `add_operator(..., ovf=True)` recurses via
        // `add_operator(name+'_ovf', …)`; the pairs below trace that
        // recursion.
        assert_eq!(OpKind::Add.ovf_variant(), Some(OpKind::AddOvf));
        assert_eq!(OpKind::Sub.ovf_variant(), Some(OpKind::SubOvf));
        assert_eq!(OpKind::Mul.ovf_variant(), Some(OpKind::MulOvf));
        assert_eq!(OpKind::Neg.ovf_variant(), Some(OpKind::NegOvf));
        assert_eq!(OpKind::Abs.ovf_variant(), Some(OpKind::AbsOvf));
        assert_eq!(OpKind::LShift.ovf_variant(), Some(OpKind::LShiftOvf));
        // No overflow variant for Eq / Lt / And / ….
        assert_eq!(OpKind::Eq.ovf_variant(), None);
        assert_eq!(OpKind::And.ovf_variant(), None);
        // And the _ovf's themselves don't recurse further.
        assert_eq!(OpKind::AddOvf.ovf_variant(), None);
    }

    #[test]
    fn dispatch_classification_matches_upstream() {
        // `add_operator('add', 2, dispatch=2, …)` → Double.
        assert_eq!(OpKind::Add.dispatch(), Dispatch::Double);
        // `add_operator('len', 1, dispatch=1, …)` → Single.
        assert_eq!(OpKind::Len.dispatch(), Dispatch::Single);
        // `Pow` / `NewDict` / `NewTuple` carry no `dispatch=` →
        // `HLOperation.dispatch = None`.
        assert_eq!(OpKind::Pow.dispatch(), Dispatch::None);
        assert_eq!(OpKind::NewTuple.dispatch(), Dispatch::None);
    }

    #[test]
    fn arity_matches_upstream() {
        assert_eq!(OpKind::Add.arity(), Some(2));
        assert_eq!(OpKind::Len.arity(), Some(1));
        assert_eq!(OpKind::SetAttr.arity(), Some(3));
        assert_eq!(OpKind::SetSlice.arity(), Some(4));
        // variadic / manual-dispatch → None.
        assert_eq!(OpKind::NewTuple.arity(), None);
        assert_eq!(OpKind::SimpleCall.arity(), None);
    }

    #[test]
    fn newstringbuilder_opname_classification_mirrors_newlist() {
        // The builder ctor op round-trips through the opname map and is
        // classified like the other manual-dispatch construction ops.
        assert_eq!(OpKind::NewStringBuilder.opname(), "newstringbuilder");
        assert_eq!(
            OpKind::from_opname("newstringbuilder"),
            Some(OpKind::NewStringBuilder)
        );
        assert_eq!(OpKind::NewStringBuilder.dispatch(), Dispatch::None);
        assert_eq!(OpKind::NewStringBuilder.arity(), None);
        // Allocation with identity → not a PureOperation (never constfolds).
        assert!(!OpKind::NewStringBuilder.pure());
    }

    #[test]
    fn apply_specialization_reifies_typed_annotator_error_payload() {
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let hl = HLOperation::new(OpKind::Hash, vec![]);
        let spec = Specialization {
            apply: Box::new(|_, _| {
                std::panic::panic_any(crate::annotator::model::AnnotatorError::new(
                    "typed payload",
                ))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        };

        let err = apply_specialization(&spec, &ann, &hl)
            .expect_err("typed AnnotatorError panic must be reified");
        assert_eq!(err.msg.as_deref(), Some("typed payload"));
    }

    #[test]
    fn pure_matches_upstream() {
        assert!(OpKind::Add.pure());
        assert!(OpKind::NewTuple.pure());
        assert!(OpKind::Pow.pure());
        assert!(!OpKind::SetAttr.pure());
        assert!(!OpKind::SimpleCall.pure());
        assert!(!OpKind::InplaceAdd.pure());
        // `_ovf` siblings: NOT pure. upstream `add_operator(name +
        // '_ovf', arity, dispatch, pyfunc=ovf_func)` carries no
        // `pure=True` (`operation.py:337-338`), so the suffixed
        // class falls through to plain `HLOperation` whose default
        // `constfold(self) -> None` (`operation.py`) prevents
        // any constant-time overflow / zero-divisor folding.
        assert!(!OpKind::AddOvf.pure());
        assert!(!OpKind::SubOvf.pure());
        assert!(!OpKind::MulOvf.pure());
        assert!(!OpKind::NegOvf.pure());
        assert!(!OpKind::AbsOvf.pure());
        assert!(!OpKind::FloorDivOvf.pure());
        assert!(!OpKind::DivOvf.pure());
        assert!(!OpKind::ModOvf.pure());
        assert!(!OpKind::LShiftOvf.pure());
    }

    #[test]
    fn can_overflow_matches_upstream() {
        // `add_operator(name, ..., ovf=True)` (`operation.py`)
        // makes the BASE op an `OverflowingOperation` subclass with
        // `can_overflow = True`. The recursive
        // `add_operator(name + '_ovf', arity, dispatch,
        // pyfunc=ovf_func)` at `operation.py:338` drops both
        // `pure=` and `ovf=`, so the `_ovf` sibling's class is the
        // plain `HLOperation` whose default
        // `can_overflow = False` (`operation.py:69`).
        assert!(OpKind::Add.can_overflow());
        assert!(!OpKind::AddOvf.can_overflow());
        assert!(OpKind::Sub.can_overflow());
        assert!(!OpKind::SubOvf.can_overflow());
        assert!(OpKind::Mul.can_overflow());
        assert!(!OpKind::MulOvf.can_overflow());
        assert!(OpKind::Neg.can_overflow());
        assert!(!OpKind::NegOvf.can_overflow());
        assert!(OpKind::Abs.can_overflow());
        assert!(!OpKind::AbsOvf.can_overflow());
        assert!(OpKind::Div.can_overflow());
        assert!(!OpKind::DivOvf.can_overflow());
        assert!(OpKind::FloorDiv.can_overflow());
        assert!(!OpKind::FloorDivOvf.can_overflow());
        assert!(OpKind::Mod.can_overflow());
        assert!(!OpKind::ModOvf.can_overflow());
        assert!(OpKind::LShift.can_overflow());
        assert!(!OpKind::LShiftOvf.can_overflow());
        assert!(!OpKind::RShift.can_overflow());
        assert!(!OpKind::Eq.can_overflow());
    }

    fn c(v: ConstValue) -> Hlvalue {
        Hlvalue::Constant(Constant::new(v))
    }

    fn ci(n: i64) -> Hlvalue {
        c(ConstValue::Int(n))
    }

    fn cu(n: u64) -> Hlvalue {
        c(ConstValue::UInt(n))
    }

    fn cf(f: f64) -> Hlvalue {
        c(ConstValue::float(f))
    }

    fn cb(b: bool) -> Hlvalue {
        c(ConstValue::Bool(b))
    }

    fn cs(s: &str) -> Hlvalue {
        c(ConstValue::byte_str(s))
    }

    fn fold(kind: OpKind, args: Vec<Hlvalue>) -> Option<ConstValue> {
        let op = HLOperation::new(kind, args);
        op.constfold().ok().flatten().map(|v| match v {
            Hlvalue::Constant(c) => c.value,
            _ => unreachable!("constfold must not produce Variables"),
        })
    }

    #[test]
    fn canraise_table_matches_upstream() {
        use BuiltinException::*;
        // operation.py:728-731.
        assert_eq!(
            OpKind::GetItem.canraise(),
            &[IndexError, KeyError, Exception]
        );
        assert_eq!(
            OpKind::DelItem.canraise(),
            &[IndexError, KeyError, Exception]
        );
        // operation.py:732.
        assert_eq!(OpKind::Contains.canraise(), &[Exception]);
        assert_eq!(OpKind::GetAttr.canraise(), &[]);
        assert_eq!(OpKind::Iter.canraise(), &[]);
        assert_eq!(OpKind::Next.canraise(), &[StopIteration, RuntimeError]);
        // `add` / `sub` / `mul` — base variants have empty canraise.
        assert_eq!(OpKind::Add.canraise(), &[]);
        assert_eq!(OpKind::Sub.canraise(), &[]);
        assert_eq!(OpKind::Mul.canraise(), &[]);
        // `_ovf` twins pick up OverflowError (_add_except_ovf).
        assert_eq!(OpKind::AddOvf.canraise(), &[OverflowError]);
        assert_eq!(OpKind::NegOvf.canraise(), &[OverflowError]);
        // `div` picks up ZeroDivisionError only; `div_ovf` adds OverflowError.
        assert_eq!(OpKind::Div.canraise(), &[ZeroDivisionError]);
        assert_eq!(
            OpKind::DivOvf.canraise(),
            &[ZeroDivisionError, OverflowError]
        );
        // `lshift` picks up ValueError; `lshift_ovf` + OverflowError.
        assert_eq!(OpKind::LShift.canraise(), &[ValueError]);
        assert_eq!(OpKind::LShiftOvf.canraise(), &[ValueError, OverflowError]);
        // `pow` picks up all three (ZeroDiv from 751, ValueError from
        // 754, OverflowError from 762).
        assert_eq!(
            OpKind::Pow.canraise(),
            &[ZeroDivisionError, ValueError, OverflowError]
        );
        // `truediv` / `divmod` — ZeroDivisionError + OverflowError, no _ovf.
        assert_eq!(
            OpKind::TrueDiv.canraise(),
            &[ZeroDivisionError, OverflowError]
        );
        assert_eq!(
            OpKind::DivMod.canraise(),
            &[ZeroDivisionError, OverflowError]
        );
        // `inplace_add` — only OverflowError (line 756).
        assert_eq!(OpKind::InplaceAdd.canraise(), &[OverflowError]);
        // Comparisons / identity / len stay clean.
        assert_eq!(OpKind::Lt.canraise(), &[]);
        assert_eq!(OpKind::Is.canraise(), &[]);
        assert_eq!(OpKind::Len.canraise(), &[]);
    }

    #[test]
    fn constfold_declines_non_pure_ops() {
        // `setattr` is side-effecting → no fold.
        assert_eq!(fold(OpKind::SetAttr, vec![ci(1), cs("x"), ci(2)]), None);
        // `simple_call` is non-pure → no fold.
        assert_eq!(fold(OpKind::SimpleCall, vec![ci(1), ci(2)]), None);
    }

    #[test]
    fn constfold_declines_variable_args() {
        let v = Hlvalue::Variable(Variable::new());
        assert_eq!(fold(OpKind::Add, vec![v, ci(1)]), None);
    }

    #[test]
    fn getattr_constfold_raises_flowing_error_on_three_arg_form() {
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![
                ci(1),
                cs("real"),
                Hlvalue::Constant(Constant::new(ConstValue::None)),
            ],
        );
        let err = op
            .constfold()
            .expect_err("3-arg getattr must raise FlowingError");
        assert!(err.message.contains("three arguments not supported"));
    }

    #[test]
    fn getattr_constfold_folds_constant_module_member() {
        let module = super::super::model::HOST_ENV
            .import_module("rpython.rlib.rfile")
            .expect("bootstrap module must exist");
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::HostObject(module)), cs("create_file")],
        );
        let folded = op
            .constfold()
            .expect("module getattr should not error")
            .expect("module getattr should fold");
        let Hlvalue::Constant(constant) = folded else {
            panic!("expected Constant result");
        };
        assert!(matches!(constant.value, ConstValue::HostObject(_)));
    }

    #[test]
    fn getattr_constfold_declines_volatile_sys_attr() {
        // upstream operation.py:22-35 — `sys.path` / `sys.modules`
        // are runtime-variable (mutable state of the interpreter), so
        // the NOT_REALLY_CONST guard must decline the fold even if
        // the module exposes the attribute via `module_get`. Matches
        // the `if w_obj in NOT_REALLY_CONST and w_name not in …`
        // branch at operation.py:631-633.
        use crate::flowspace::model::HostObject;
        let sys = HostObject::new_module("sys");
        sys.module_set(
            "path".to_string(),
            HostObject::new_module("path_placeholder"),
        );
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::HostObject(sys)), cs("path")],
        );
        assert!(op.constfold().unwrap().is_none());
    }

    #[test]
    fn getattr_constfold_folds_allowlisted_sys_attr() {
        // upstream operation.py:25 — `sys.maxint` is explicitly
        // declared as a real constant. Folding must succeed when the
        // module exposes it.
        use crate::flowspace::model::HostObject;
        let sys = HostObject::new_module("sys");
        sys.module_set(
            "maxint".to_string(),
            HostObject::new_module("maxint_placeholder"),
        );
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::HostObject(sys)), cs("maxint")],
        );
        assert!(op.constfold().unwrap().is_some());
    }

    #[test]
    fn getattr_constfold_folds_class_attribute() {
        // upstream operation.py — `getattr(cls, 'method')`
        // folds when the class exposes the attribute via its __dict__.
        use crate::flowspace::model::HostObject;
        let cls = HostObject::new_class("Foo", vec![]);
        cls.class_set("CONST".to_string(), ConstValue::Int(42));
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::HostObject(cls)), cs("CONST")],
        );
        let folded = op
            .constfold()
            .expect("class getattr should not error")
            .expect("class getattr should fold");
        let Hlvalue::Constant(constant) = folded else {
            panic!("expected Constant result");
        };
        assert_eq!(constant.value, ConstValue::Int(42));
    }

    #[test]
    fn getattr_constfold_folds_instance_dict_attribute() {
        // upstream operation.py — `getattr(instance, 'attr')`
        // respects `inst.__dict__` before walking the class MRO. Rust
        // port mirrors this via HostObject::instance_get.
        use crate::flowspace::model::HostObject;
        let cls = HostObject::new_class("Foo", vec![]);
        cls.class_set("x".to_string(), ConstValue::Int(1)); // class-level
        let inst = HostObject::new_instance(cls, vec![]);
        inst.instance_set("x", ConstValue::Int(99)).unwrap(); // shadow
        inst.instance_set("y", ConstValue::Int(7)).unwrap(); // instance-only
        // Need to make the instance foldable for the test — it isn't by
        // default (foldable() returns false for user instances). The
        // exercise here is `const_runtime_getattr` itself, so call the
        // helper directly. Upstream's
        // `w_obj.foldable()` gate intentionally rejects user instances;
        // real folds hit this code path via `_freeze_` markers we
        // don't model. Verify the helper's correctness instead:
        let r = crate::flowspace::model::const_runtime_getattr(
            &ConstValue::HostObject(inst.clone()),
            "x",
        )
        .expect("instance x lookup");
        assert_eq!(r, Some(ConstValue::Int(99))); // shadowed
        let r2 = crate::flowspace::model::const_runtime_getattr(&ConstValue::HostObject(inst), "y")
            .expect("instance y lookup");
        assert_eq!(r2, Some(ConstValue::Int(7))); // instance-only
    }

    #[test]
    fn getattr_constfold_instance_falls_through_to_class_mro() {
        // Attribute missing on instance but present on class → use MRO.
        use crate::flowspace::model::HostObject;
        let parent = HostObject::new_class("Parent", vec![]);
        parent.class_set("z", ConstValue::Int(100));
        let child = HostObject::new_class("Child", vec![parent.clone()]);
        let inst = HostObject::new_instance(child, vec![]);
        let r = crate::flowspace::model::const_runtime_getattr(&ConstValue::HostObject(inst), "z")
            .expect("inherited attribute lookup");
        assert_eq!(r, Some(ConstValue::Int(100)));
    }

    #[test]
    fn getattr_constfold_raises_flowing_error_on_missing_class_attribute() {
        // upstream operation.py — `getattr(cls, 'missing')`
        // raises `AttributeError`, which flowspace escalates to
        // `FlowingError("… always raises AttributeError")`.
        use crate::flowspace::model::HostObject;
        let cls = HostObject::new_class("Foo", vec![]);
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::HostObject(cls)), cs("missing")],
        );
        let err = op
            .constfold()
            .expect_err("missing class attribute must escalate to FlowingError");
        assert!(err.message.contains("always raises AttributeError"));
    }

    #[test]
    fn getattr_constfold_user_function_missing_attr_raises() {
        // upstream operation.py:634-642 — user function constants are
        // foldable, so `getattr(function, "missing")` reaches the real
        // getattr() call and raises FlowingError on AttributeError.
        use crate::flowspace::model::{GraphFunc, HostObject};
        let globals = Constant::new(ConstValue::None);
        let gf = GraphFunc::new("user_fn".to_string(), globals);
        let func = HostObject::new_user_function(gf);
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::HostObject(func)), cs("whatever")],
        );
        let err = op
            .constfold()
            .expect_err("missing function attribute must escalate to FlowingError");
        assert!(err.message.contains("always raises AttributeError"));
    }

    #[test]
    fn getattr_constfold_user_function_dunder_name_returns_constant_str() {
        use crate::flowspace::model::{GraphFunc, HostObject};
        let globals = Constant::new(ConstValue::None);
        let gf = GraphFunc::new("pkg.demo.user_fn".to_string(), globals);
        let func = HostObject::new_user_function(gf);
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::HostObject(func)), cs("__name__")],
        );
        let folded = op.constfold().expect("function.__name__ should fold");
        let Some(Hlvalue::Constant(constant)) = folded else {
            panic!("expected Constant result");
        };
        assert_eq!(constant.value, ConstValue::byte_str("user_fn"));
    }

    #[test]
    fn getattr_constfold_int_dunder_class_returns_builtin_int_class() {
        use crate::flowspace::model::HOST_ENV;
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::Int(5)), cs("__class__")],
        );
        let folded = op.constfold().expect("int.__class__ should fold");
        let Some(Hlvalue::Constant(constant)) = folded else {
            panic!("expected Constant result");
        };
        let expected = HOST_ENV.lookup_builtin("int").expect("builtin int class");
        assert_eq!(constant.value, ConstValue::HostObject(expected));
    }

    #[test]
    fn getattr_constfold_int_dunder_class_name_returns_constant_str() {
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::Int(5)), cs("__class__")],
        );
        let folded = op.constfold().expect("int.__class__ should fold");
        let Some(Hlvalue::Constant(class_constant)) = folded else {
            panic!("expected Constant result");
        };
        let op2 = HLOperation::new(
            OpKind::GetAttr,
            vec![Hlvalue::Constant(class_constant), cs("__name__")],
        );
        let folded2 = op2.constfold().expect("int.__class__.__name__ should fold");
        let Some(Hlvalue::Constant(name_constant)) = folded2 else {
            panic!("expected Constant result");
        };
        assert_eq!(name_constant.value, ConstValue::byte_str("int"));
    }

    #[test]
    fn getattr_constfold_int_dunder_class_module_returns_constant_str() {
        let op = HLOperation::new(
            OpKind::GetAttr,
            vec![c(ConstValue::Int(5)), cs("__class__")],
        );
        let folded = op.constfold().expect("int.__class__ should fold");
        let Some(Hlvalue::Constant(class_constant)) = folded else {
            panic!("expected Constant result");
        };
        let op2 = HLOperation::new(
            OpKind::GetAttr,
            vec![Hlvalue::Constant(class_constant), cs("__module__")],
        );
        let folded2 = op2
            .constfold()
            .expect("int.__class__.__module__ should fold");
        let Some(Hlvalue::Constant(module_constant)) = folded2 else {
            panic!("expected Constant result");
        };
        assert_eq!(module_constant.value, ConstValue::byte_str("__builtin__"));
    }

    #[test]
    fn getattr_constfold_function_carrier_dunder_attrs() {
        let cases = [
            (
                "name",
                ConstValue::None,
                "__name__",
                ConstValue::byte_str("user_fn"),
            ),
            (
                "module",
                ConstValue::None,
                "__module__",
                ConstValue::byte_str("pkg.demo"),
            ),
            (
                "globals",
                ConstValue::Dict(std::sync::Arc::new(indexmap::IndexMap::new())),
                "__globals__",
                ConstValue::Dict(std::sync::Arc::new(indexmap::IndexMap::new())),
            ),
            (
                "defaults",
                ConstValue::Dict(std::sync::Arc::new(indexmap::IndexMap::new())),
                "__defaults__",
                ConstValue::None,
            ),
            (
                "closure",
                ConstValue::Dict(std::sync::Arc::new(indexmap::IndexMap::new())),
                "__closure__",
                ConstValue::None,
            ),
        ];
        for (name, globals, attr, want) in cases {
            let func = ConstValue::Function(Box::new(crate::flowspace::model::GraphFunc::new(
                "pkg.demo.user_fn",
                Constant::new(globals),
            )));
            let op = HLOperation::new(OpKind::GetAttr, vec![c(func), cs(attr)]);
            let folded_msg = format!("case {name}: function.{attr} should fold");
            let folded = op.constfold().expect(&folded_msg);
            let Some(Hlvalue::Constant(constant)) = folded else {
                panic!("case {name}: expected Constant result");
            };
            assert_eq!(constant.value, want, "case {name}");
        }
    }

    #[test]
    fn getattr_constfold_function_carrier_dunder_defaults_and_closure_return_tuple_payloads() {
        let globals = Constant::new(ConstValue::Dict(std::sync::Arc::new(
            indexmap::IndexMap::new(),
        )));
        let mut graph_func = crate::flowspace::model::GraphFunc::new("pkg.demo.user_fn", globals);
        graph_func.defaults = vec![Constant::new(ConstValue::Int(7))];
        graph_func.closure = vec![Constant::new(ConstValue::byte_str("cell"))];
        let func = ConstValue::Function(Box::new(graph_func));

        let op_defaults =
            HLOperation::new(OpKind::GetAttr, vec![c(func.clone()), cs("__defaults__")]);
        let folded_defaults = op_defaults
            .constfold()
            .expect("function.__defaults__ should fold");
        let Some(Hlvalue::Constant(defaults_constant)) = folded_defaults else {
            panic!("expected Constant result");
        };
        assert_eq!(
            defaults_constant.value,
            ConstValue::Tuple(vec![ConstValue::Int(7)])
        );

        let op_closure = HLOperation::new(OpKind::GetAttr, vec![c(func), cs("__closure__")]);
        let folded_closure = op_closure
            .constfold()
            .expect("function.__closure__ should fold");
        let Some(Hlvalue::Constant(closure_constant)) = folded_closure else {
            panic!("expected Constant result");
        };
        assert_eq!(
            closure_constant.value,
            ConstValue::Tuple(vec![ConstValue::byte_str("cell")])
        );
    }

    #[test]
    fn constfold_int_arithmetic() {
        assert_eq!(
            fold(OpKind::Add, vec![ci(3), ci(4)]),
            Some(ConstValue::Int(7))
        );
        assert_eq!(
            fold(OpKind::Sub, vec![ci(10), ci(4)]),
            Some(ConstValue::Int(6))
        );
        assert_eq!(
            fold(OpKind::Mul, vec![ci(6), ci(7)]),
            Some(ConstValue::Int(42))
        );
        assert_eq!(fold(OpKind::Neg, vec![ci(5)]), Some(ConstValue::Int(-5)));
        assert_eq!(fold(OpKind::Abs, vec![ci(-5)]), Some(ConstValue::Int(5)));
        assert_eq!(fold(OpKind::Invert, vec![ci(0)]), Some(ConstValue::Int(-1)));
        // `PureOperation.constfold` runs the real Python op on `r_uint`
        // (`base_int` / `unsigned_int` / `compute_restype`). Mixed
        // r_uint/int yields r_uint MASK-wrapped; `-r_uint(1)` wraps;
        // `>>` is logical; `r_uint << 64` is `r_uint(0)`.
        assert_eq!(
            fold(OpKind::Add, vec![cu(3), ci(4)]),
            Some(ConstValue::UInt(7))
        );
        assert_eq!(
            fold(OpKind::Add, vec![ci(3), cu(4)]),
            Some(ConstValue::UInt(7))
        );
        assert_eq!(
            fold(OpKind::Add, vec![cu(u64::MAX), ci(1)]),
            Some(ConstValue::UInt(0))
        );
        assert_eq!(
            fold(OpKind::Add, vec![cu(1), cb(true)]),
            Some(ConstValue::UInt(2))
        );
        assert_eq!(
            fold(OpKind::Add, vec![cu(1), cf(1.0)]),
            Some(ConstValue::float(2.0))
        );
        assert_eq!(
            fold(OpKind::Neg, vec![cu(1)]),
            Some(ConstValue::UInt(u64::MAX))
        );
        assert_eq!(fold(OpKind::Abs, vec![cu(1)]), Some(ConstValue::UInt(1)));
        assert_eq!(
            fold(OpKind::Invert, vec![cu(0)]),
            Some(ConstValue::UInt(u64::MAX))
        );
        assert_eq!(
            fold(OpKind::RShift, vec![cu(u64::MAX), ci(1)]),
            Some(ConstValue::UInt(u64::MAX >> 1))
        );
        assert_eq!(
            fold(OpKind::LShift, vec![cu(1), ci(64)]),
            Some(ConstValue::UInt(0))
        );
        assert_eq!(
            fold(OpKind::LShift, vec![cu(1), ci(i64::MAX)]),
            Some(ConstValue::UInt(0))
        );
        // `base_int.__lshift__`: count above `sys.maxsize` raises
        // OverflowError, which `PureOperation.constfold` turns into
        // FlowingError.
        let op = HLOperation::new(OpKind::LShift, vec![cu(1), cu(i64::MAX as u64 + 1)]);
        let err = op
            .constfold()
            .expect_err("unsigned lshift count > i64::MAX must raise OverflowError");
        assert!(
            err.message.contains("OverflowError") && err.message.contains("shift count too large"),
            "{}",
            err.message,
        );
        // `W_LongObject._lshift`: zero left operand returns zero
        // before looking at the count.
        assert_eq!(
            fold(OpKind::LShift, vec![cu(0), cu(i64::MAX as u64 + 1)]),
            Some(ConstValue::UInt(0)),
        );
        assert_eq!(
            fold(OpKind::FloorDiv, vec![cu(5), ci(-2)]),
            Some(ConstValue::UInt((-3i128) as u64))
        );
        assert_eq!(
            fold(OpKind::And, vec![cu(u64::MAX), ci(-1)]),
            Some(ConstValue::UInt(u64::MAX))
        );
    }

    #[test]
    fn constfold_int_overflow_declines() {
        // RPython `add_ovf` raises OverflowError; flow-space declines
        // the fold (`PureOperation._pure_result` skips long results).
        assert_eq!(
            fold(OpKind::AddOvf, vec![ci(i64::MAX), ci(1)]),
            None,
            "overflow must not fold"
        );
        assert_eq!(
            fold(OpKind::Add, vec![ci(i64::MAX), ci(1)]),
            None,
            "overflow on checked i64 must not fold"
        );
        assert_eq!(
            fold(OpKind::Neg, vec![ci(i64::MIN)]),
            None,
            "neg(i64::MIN) overflows"
        );
    }

    #[test]
    fn constfold_division_by_zero_raises_flowing_error() {
        // upstream `PureOperation.constfold` (operation.py)
        // wraps `pyfunc(*args)` in `try/except Exception` and re-raises
        // captured exceptions as `FlowingError`. Division by zero is
        // the canonical case — Python's int `/`, `//`, `%` raise
        // `ZeroDivisionError`, so a constant-folded `1 / 0` must
        // surface as FlowingError, not silent decline.
        // Only base ops raise at constfold time. upstream `_ovf`
        // siblings are NOT pure (`add_operator(name + '_ovf', …)`
        // carries no `pure=True`, `operation.py:337-338`), so their
        // `HLOperation.constfold` default returns `None`
        // (`operation.py`) — the op stays as an op for the
        // runtime to handle. Folding `_ovf` zero-divisor at compile
        // time would surface a hard-error divergence vs upstream.
        for kind in [OpKind::Div, OpKind::Mod, OpKind::FloorDiv] {
            let op = HLOperation::new(kind, vec![ci(1), ci(0)]);
            let err = op.constfold().expect_err("Int(0) divisor must raise");
            assert!(
                err.message.contains("ZeroDivisionError"),
                "{kind:?} Int(0): {}",
                err.message
            );
            // Python `bool` is a subclass of `int`, so every
            // (int-like LHS) / (zero-int-like RHS) pair raises
            // `ZeroDivisionError` regardless of which side is
            // `Bool` and which is `Int`. Pin all four combinations
            // (Int/Bool LHS × Int(0)/Bool(false) RHS).
            for (label, lhs) in [("Int(1)", ci(1)), ("Bool(true)", cb(true))] {
                for (rhs_label, rhs) in [("Int(0)", ci(0)), ("Bool(false)", cb(false))] {
                    let op = HLOperation::new(kind, vec![lhs.clone(), rhs.clone()]);
                    let err = op.constfold().expect_err(&format!(
                        "{kind:?} {label} / {rhs_label} must raise (bool ⊂ int)"
                    ));
                    assert!(
                        err.message.contains("ZeroDivisionError"),
                        "{kind:?} {label} / {rhs_label}: {}",
                        err.message,
                    );
                }
            }
            // `False // False` (LHS Bool zero, RHS Bool zero) — both
            // sides bool, divisor zero, must still raise.
            let op = HLOperation::new(kind, vec![cb(false), cb(false)]);
            let err = op
                .constfold()
                .expect_err(&format!("{kind:?} Bool(false) / Bool(false) must raise"));
            assert!(err.message.contains("ZeroDivisionError"));
            // `r_uint // 0` raises ZeroDivisionError (`unsigned_int` /
            // `base_int.__floordiv__`).
            let op = HLOperation::new(kind, vec![cu(1), cu(0)]);
            let err = op
                .constfold()
                .expect_err(&format!("{kind:?} UInt(1) / UInt(0) must raise"));
            assert!(err.message.contains("ZeroDivisionError"));
            let op = HLOperation::new(kind, vec![cu(1), ci(0)]);
            let err = op
                .constfold()
                .expect_err(&format!("{kind:?} UInt(1) / Int(0) must raise"));
            assert!(err.message.contains("ZeroDivisionError"));
        }
        for kind in [OpKind::DivOvf, OpKind::ModOvf, OpKind::FloorDivOvf] {
            let op = HLOperation::new(kind, vec![ci(1), ci(0)]);
            assert_eq!(
                op.constfold().expect("_ovf sibling must NOT raise at fold"),
                None,
                "{kind:?} is HLOperation default — must decline (op stays at runtime)",
            );
        }
        // TrueDiv with int or float zero divisor.
        let op = HLOperation::new(OpKind::TrueDiv, vec![ci(1), ci(0)]);
        let err = op
            .constfold()
            .expect_err("truediv int zero divisor must raise");
        assert!(err.message.contains("ZeroDivisionError"));
        // TrueDiv with `Bool(false)` divisor: `1 / False` is
        // `ZeroDivisionError` upstream (`bool ⊂ int`), so the fold
        // must raise here too.
        let op = HLOperation::new(OpKind::TrueDiv, vec![ci(1), cb(false)]);
        let err = op
            .constfold()
            .expect_err("truediv Bool(false) divisor must raise (bool ⊂ int)");
        assert!(err.message.contains("ZeroDivisionError"));
        let op = HLOperation::new(OpKind::TrueDiv, vec![cf(1.0), cf(0.0)]);
        let err = op
            .constfold()
            .expect_err("truediv float zero divisor must raise");
        assert!(err.message.contains("ZeroDivisionError"));
        // TrueDiv with non-numeric lhs: upstream
        // `operator.truediv("x", 0)` raises `TypeError` first (the
        // dispatch fails before the divisor is consulted), NOT
        // `ZeroDivisionError`. The always-raises whitelist therefore
        // declines this case rather than misclassifying it as a
        // zero-divisor error — fold falls through to the broader
        // pyfunc port, where the type-error path is documented as a
        // pre-existing gap.
        assert_eq!(
            fold(OpKind::TrueDiv, vec![cs("x"), ci(0)]),
            None,
            "non-numeric lhs / zero divisor must NOT classify as ZeroDivisionError",
        );
        assert_eq!(
            fold(OpKind::TrueDiv, vec![cs("x"), cf(0.0)]),
            None,
            "non-numeric lhs / float zero divisor must NOT classify as ZeroDivisionError",
        );
    }

    #[test]
    fn constfold_float_zero_divisor_raises_flowing_error() {
        // Once any operand is float, upstream `operator.{div,floordiv,
        // mod}` dispatches to the float method which raises
        // `ZeroDivisionError` with a float-specific message.
        // `PureOperation.constfold` (operation.py) catches the
        // exception and re-raises as `FlowingError`, so a constant-
        // folded `7.0 / 0.0`, `7.0 // 0.0`, `7.0 % 0.0` (and the
        // mixed-type variants) must surface as FlowingError, not
        // silent decline.
        let cases: &[(OpKind, &str)] = &[
            (OpKind::Div, "float division by zero"),
            (OpKind::FloorDiv, "float floor division by zero"),
            (OpKind::Mod, "float modulo"),
        ];
        for (kind, expected_msg) in cases {
            // Float / Float zero
            let op = HLOperation::new(*kind, vec![cf(7.0), cf(0.0)]);
            let err = op
                .constfold()
                .expect_err(&format!("{kind:?} Float(7.0) / Float(0.0) must raise"));
            assert!(
                err.message.contains("ZeroDivisionError") && err.message.contains(expected_msg),
                "{kind:?} Float/Float: {}",
                err.message,
            );
            // Mixed: Float lhs, Int(0) rhs → coerces to float division
            let op = HLOperation::new(*kind, vec![cf(7.0), ci(0)]);
            let err = op
                .constfold()
                .expect_err(&format!("{kind:?} Float(7.0) / Int(0) must raise"));
            assert!(
                err.message.contains("ZeroDivisionError") && err.message.contains(expected_msg),
                "{kind:?} Float/Int: {}",
                err.message,
            );
            // Mixed: Int lhs, Float(0.0) rhs
            let op = HLOperation::new(*kind, vec![ci(7), cf(0.0)]);
            let err = op
                .constfold()
                .expect_err(&format!("{kind:?} Int(7) / Float(0.0) must raise"));
            assert!(
                err.message.contains("ZeroDivisionError") && err.message.contains(expected_msg),
                "{kind:?} Int/Float: {}",
                err.message,
            );
            // Bool lhs, Float(0.0) rhs (`True // 0.0`)
            let op = HLOperation::new(*kind, vec![cb(true), cf(0.0)]);
            let err = op
                .constfold()
                .expect_err(&format!("{kind:?} Bool(true) / Float(0.0) must raise"));
            assert!(
                err.message.contains("ZeroDivisionError") && err.message.contains(expected_msg),
                "{kind:?} Bool/Float: {}",
                err.message,
            );
        }
        // Float lhs, integer-zero rhs for the existing arms still
        // produces the float-specific message — the new arm
        // supersedes the int-only arm via `is_int_like(lhs)` failing
        // for `Float(_)`.
        let op = HLOperation::new(OpKind::Div, vec![cf(-7.0), cb(false)]);
        let err = op
            .constfold()
            .expect_err("Div Float(-7.0) / Bool(false) must raise float ZDE");
        assert!(
            err.message.contains("float division by zero"),
            "{}",
            err.message
        );
    }

    #[test]
    fn constfold_int_float_compare_handles_mantissa_boundary() {
        // Port test for `pypy/objspace/std/floatobject.py make_compare_func
        // make_compare_func`'s bigint-aware Int↔Float compare. f64
        // mantissa is 53 bits; `Int(2^53 + 1)` rounds to `2^53` under
        // naive `as f64` cast, which would misclassify `Int(2^53 + 1)
        // == Float(2^53)` as True. Upstream routes through
        // `do_compare_bigint` whenever `not int_between(-1, i2 >> 48,
        // 1)` (i.e., the int's magnitude is large); we use the same
        // 53-bit mantissa boundary in `cmp_int_float_exact`.
        let big_int = 9007199254740993_i64; // 2^53 + 1
        let exact_2_to_53 = 9007199254740992.0_f64; // 2^53
        let i = ci(big_int);
        let f = cf(exact_2_to_53);

        // Eq: must be False (i is not exactly representable as f64).
        assert_eq!(
            fold(OpKind::Eq, vec![i.clone(), f.clone()]),
            Some(ConstValue::Bool(false)),
            "Int(2^53+1) == Float(2^53) must be False (bigint compare)",
        );
        assert_eq!(
            fold(OpKind::Eq, vec![f.clone(), i.clone()]),
            Some(ConstValue::Bool(false)),
            "Float(2^53) == Int(2^53+1) must be False (commute)",
        );
        // Ne: True.
        assert_eq!(
            fold(OpKind::Ne, vec![i.clone(), f.clone()]),
            Some(ConstValue::Bool(true)),
        );
        // Gt: i > f.
        assert_eq!(
            fold(OpKind::Gt, vec![i.clone(), f.clone()]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Lt, vec![f.clone(), i.clone()]),
            Some(ConstValue::Bool(true)),
        );
        // Ge / Le.
        assert_eq!(
            fold(OpKind::Ge, vec![i.clone(), f.clone()]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Le, vec![f.clone(), i.clone()]),
            Some(ConstValue::Bool(true)),
        );

        // Same-value check at the boundary: Int(2^53) == Float(2^53)
        // → True (the int IS exactly representable).
        let exact_int = 9007199254740992_i64;
        assert_eq!(
            fold(OpKind::Eq, vec![ci(exact_int), cf(exact_2_to_53)]),
            Some(ConstValue::Bool(true)),
            "Int(2^53) == Float(2^53) must be True",
        );

        // Negative side mirror: Int(-(2^53 + 1)) vs Float(-2^53).
        // -2^53 - 1 = -9007199254740993; rounds to -2^53 under cast.
        assert_eq!(
            fold(
                OpKind::Eq,
                vec![ci(-9007199254740993), cf(-9007199254740992.0)]
            ),
            Some(ConstValue::Bool(false)),
            "Int(-(2^53+1)) == Float(-2^53) must be False",
        );
        assert_eq!(
            fold(
                OpKind::Lt,
                vec![ci(-9007199254740993), cf(-9007199254740992.0)]
            ),
            Some(ConstValue::Bool(true)),
            "Int(-(2^53+1)) < Float(-2^53) must be True",
        );

        // Non-integer-valued float vs nearby int: never equal regardless
        // of magnitude.
        assert_eq!(
            fold(OpKind::Eq, vec![ci(big_int), cf(9007199254740992.5)]),
            Some(ConstValue::Bool(false)),
        );
        assert_eq!(
            fold(OpKind::Lt, vec![ci(big_int), cf(9007199254740992.5)]),
            Some(ConstValue::Bool(false)),
            "Int(2^53+1) < Float(2^53 + 0.5): 2^53+1 == 9007199254740993, " // for clarity
        );
    }

    #[test]
    fn constfold_truediv_float_zero_divisor_uses_float_specific_message() {
        // Upstream Python 3 message specificity:
        //   `1 / 0`     → "ZeroDivisionError: division by zero"
        //   `1.0 / 0`   → "ZeroDivisionError: float division by zero"
        //   `1 / 0.0`   → "ZeroDivisionError: float division by zero"
        //   `1.0 / 0.0` → "ZeroDivisionError: float division by zero"
        // `PureOperation.constfold` re-raises the captured exception
        // verbatim, so the FlowingError text must follow the same
        // discrimination — once any operand is float, the message
        // includes "float".
        let case = |kind, args, expected_substr: &str| {
            let op = HLOperation::new(kind, args);
            let err = op.constfold().expect_err("must raise");
            assert!(
                err.message.contains("ZeroDivisionError") && err.message.contains(expected_substr),
                "expected '{expected_substr}', got: {}",
                err.message,
            );
        };

        // Int / Int 0: generic message.
        case(OpKind::TrueDiv, vec![ci(1), ci(0)], "division by zero");
        let op = HLOperation::new(OpKind::TrueDiv, vec![ci(1), ci(0)]);
        let err = op.constfold().expect_err("must raise");
        assert!(
            !err.message.contains("float"),
            "Int/Int message must NOT mention float, got: {}",
            err.message,
        );

        // Float / Int 0: float-specific message (prior parity gap —
        // generic "division by zero" was emitted when lhs is Float).
        case(
            OpKind::TrueDiv,
            vec![cf(1.0), ci(0)],
            "float division by zero",
        );
        // Float / Bool(false): same.
        case(
            OpKind::TrueDiv,
            vec![cf(1.0), cb(false)],
            "float division by zero",
        );
        // Int / Float(0.0).
        case(
            OpKind::TrueDiv,
            vec![ci(1), cf(0.0)],
            "float division by zero",
        );
        // Float / Float(0.0).
        case(
            OpKind::TrueDiv,
            vec![cf(1.0), cf(0.0)],
            "float division by zero",
        );
    }

    #[test]
    fn constfold_floor_div_and_mod_python_semantics() {
        // upstream `operator.floordiv` / `operator.mod` over ints:
        // floor toward `-inf` (Python 3 semantics), NOT C truncation
        // toward 0 and NOT Euclidean (always non-negative remainder).
        // `rpython/rtyper/rint.py ll_int_py_div` and `:496
        // ll_int_py_mod` carry the correction over the C primitive;
        // our `int_py_floor_div` / `int_py_mod` helpers port those
        // line-by-line.
        //
        // Sign coverage matrix (4 quadrants × Div/FloorDiv/Mod):
        // Python results from the reference implementation —
        //   3 // 2 = 1,   3 % 2 = 1
        //   3 // -2 = -2, 3 % -2 = -1
        //   -3 // 2 = -2, -3 % 2 = 1
        //   -3 // -2 = 1, -3 % -2 = -1
        // (a) same sign: floor matches C-trunc, no correction.
        for kind in [OpKind::FloorDiv, OpKind::Div] {
            assert_eq!(
                fold(kind, vec![ci(3), ci(2)]),
                Some(ConstValue::Int(1)),
                "{kind:?}(3, 2) must be 1",
            );
            assert_eq!(
                fold(kind, vec![ci(-3), ci(-2)]),
                Some(ConstValue::Int(1)),
                "{kind:?}(-3, -2) must be 1",
            );
        }
        assert_eq!(
            fold(OpKind::Mod, vec![ci(3), ci(2)]),
            Some(ConstValue::Int(1))
        );
        assert_eq!(
            fold(OpKind::Mod, vec![ci(-3), ci(-2)]),
            Some(ConstValue::Int(-1)),
        );
        // (b) mixed sign — these are where C-trunc / Euclidean diverge
        // from Python floor. Pre-fix Euclidean would return wrong
        // values here (see commit message).
        for kind in [OpKind::FloorDiv, OpKind::Div] {
            assert_eq!(
                fold(kind, vec![ci(3), ci(-2)]),
                Some(ConstValue::Int(-2)),
                "{kind:?}(3, -2) must be -2 (Python floor)",
            );
            assert_eq!(
                fold(kind, vec![ci(-3), ci(2)]),
                Some(ConstValue::Int(-2)),
                "{kind:?}(-3, 2) must be -2 (Python floor)",
            );
        }
        assert_eq!(
            fold(OpKind::Mod, vec![ci(3), ci(-2)]),
            Some(ConstValue::Int(-1)),
            "Mod(3, -2) must be -1 (sign of divisor, not dividend)",
        );
        assert_eq!(
            fold(OpKind::Mod, vec![ci(-3), ci(2)]),
            Some(ConstValue::Int(1)),
        );
        // (c) i64::MIN / -1 declines (Rust checked_div rejects the
        // 2's-complement overflow; the runtime path handles bigint
        // promotion at upstream). `Mod(MIN, -1)` is special: the
        // *quotient* `2^63` overflows, but the remainder `0` itself
        // fits, so upstream `operator.mod(-2**63, -1) == 0` folds
        // unconditionally. Pyre's `int_py_mod` `y == -1` short-
        // circuit captures that.
        for kind in [OpKind::FloorDiv, OpKind::Div] {
            assert_eq!(
                fold(kind, vec![ci(i64::MIN), ci(-1)]),
                None,
                "{kind:?}(MIN, -1) must decline (quotient overflow)",
            );
        }
        assert_eq!(
            fold(OpKind::Mod, vec![ci(i64::MIN), ci(-1)]),
            Some(ConstValue::Int(0)),
            "Mod(MIN, -1) folds to 0 — quotient overflows but remainder is well-defined",
        );
        // `x % -1 == 0` for every other `x` too.
        assert_eq!(
            fold(OpKind::Mod, vec![ci(7), ci(-1)]),
            Some(ConstValue::Int(0)),
        );
        assert_eq!(
            fold(OpKind::Mod, vec![ci(-7), ci(-1)]),
            Some(ConstValue::Int(0)),
        );
    }

    #[test]
    fn constfold_shifts_python_semantics() {
        // Negative shift count: upstream `operator.lshift(_, -1)` /
        // `operator.rshift(_, -1)` raise `ValueError: negative
        // shift count`, captured by `PureOperation.constfold` and
        // re-raised as `FlowingError`. Walker treats this as a
        // hard error (Err), not silent decline.
        for kind in [OpKind::LShift, OpKind::RShift] {
            let op = HLOperation::new(kind, vec![ci(1), ci(-1)]);
            let err = op
                .constfold()
                .expect_err("negative shift count must raise ValueError");
            assert!(
                err.message.contains("ValueError") && err.message.contains("negative shift count"),
                "{kind:?}(_, -1): {}",
                err.message,
            );
        }
        // Upstream calls `operator.lshift/rshift(lhs, rhs)` directly.
        // Operand type dispatch happens before the negative-count check:
        // `1 << -1` is ValueError, but `1.0 << -1` is TypeError.
        for kind in [OpKind::LShift, OpKind::RShift] {
            for args in [
                vec![cf(1.0), ci(-1)],
                vec![ci(1), cf(1.0)],
                vec![cs("x"), ci(1)],
            ] {
                let op = HLOperation::new(kind, args);
                let err = op
                    .constfold()
                    .expect_err("non-int shift operands must raise TypeError");
                assert!(
                    err.message.contains("TypeError")
                        && !err.message.contains("negative shift count"),
                    "{kind:?}: {}",
                    err.message,
                );
            }
        }
        // RShift large positive count: Python arithmetic shift
        // saturates — `1 >> 64 == 0`, `-1 >> 64 == -1`. RShift is
        // NOT registered with `ovf=True` upstream
        // (`operation.py add_operator('rshift', 2,
        // dispatch=2, pure=True)`), so the `can_overflow` decline
        // arm doesn't apply; the constfold MUST produce the
        // saturated value.
        assert_eq!(
            fold(OpKind::RShift, vec![ci(1), ci(64)]),
            Some(ConstValue::Int(0)),
            "1 >> 64 == 0 (saturating non-negative)",
        );
        assert_eq!(
            fold(OpKind::RShift, vec![ci(-1), ci(64)]),
            Some(ConstValue::Int(-1)),
            "-1 >> 64 == -1 (sign-extending arithmetic shift)",
        );
        assert_eq!(
            fold(OpKind::RShift, vec![ci(100), ci(128)]),
            Some(ConstValue::Int(0)),
        );
        // LShift large positive count: result overflows long upstream,
        // so the `can_overflow and type(result) is long` arm declines
        // (`operation.py:140-142`). Walker leaves the const unbound.
        assert_eq!(
            fold(OpKind::LShift, vec![ci(1), ci(64)]),
            None,
            "1 << 64 declines (overflows i64 — would be bignum upstream)",
        );
        // `_lshift`: `if not a: return 0`.
        assert_eq!(
            fold(OpKind::LShift, vec![ci(0), ci(64)]),
            Some(ConstValue::Int(0)),
        );
        assert_eq!(
            fold(OpKind::LShift, vec![ci(0), ci(i64::MAX)]),
            Some(ConstValue::Int(0)),
        );
        // `1 << 63` is `2**63`, a Python long that does not fit i64
        // (`PureOperation._pure_result`: `can_overflow and
        // type(result) is long`). Same when the count is `r_uint`.
        assert_eq!(fold(OpKind::LShift, vec![ci(1), ci(63)]), None);
        assert_eq!(fold(OpKind::LShift, vec![ci(1), cu(63)]), None);
        assert_eq!(
            fold(OpKind::LShift, vec![ci(-1), ci(63)]),
            Some(ConstValue::Int(i64::MIN)),
        );
        assert_eq!(
            fold(OpKind::LShift, vec![ci(1), cu(4)]),
            Some(ConstValue::Int(16)),
        );
    }

    #[test]
    fn constfold_arith_bool_int_coercion() {
        // Python `bool ⊂ int`: `True + 1 == 2` (int), `1 + True == 2`,
        // `True + True == 2`. `coerce_arith` widens Bool to 0/1 inside
        // the Int arm so every numeric arm gets the same answer it
        // would from `operator.<op>(*args)` upstream.
        assert_eq!(
            fold(OpKind::Add, vec![ci(1), cb(true)]),
            Some(ConstValue::Int(2)),
            "Int + Bool widens via Bool → 1",
        );
        assert_eq!(
            fold(OpKind::Add, vec![cb(true), ci(1)]),
            Some(ConstValue::Int(2)),
            "Bool + Int widens via Bool → 1",
        );
        assert_eq!(
            fold(OpKind::Add, vec![cb(true), cb(true)]),
            Some(ConstValue::Int(2)),
            "True + True == 2 (int, not bool)",
        );
        // Sub / Mul follow the same coercion rule.
        assert_eq!(
            fold(OpKind::Sub, vec![ci(5), cb(true)]),
            Some(ConstValue::Int(4))
        );
        assert_eq!(
            fold(OpKind::Mul, vec![cb(true), ci(7)]),
            Some(ConstValue::Int(7))
        );
        // FloorDiv / Mod / TrueDiv: bool divisor (when truthy) widens
        // to 1; bool dividend widens to 0/1.
        assert_eq!(
            fold(OpKind::FloorDiv, vec![ci(7), cb(true)]),
            Some(ConstValue::Int(7)),
            "7 // True == 7 (True widens to 1)",
        );
        assert_eq!(
            fold(OpKind::Mod, vec![cb(true), ci(2)]),
            Some(ConstValue::Int(1)),
            "True % 2 == 1 % 2 == 1",
        );
        // Shifts: bool widens to int.
        assert_eq!(
            fold(OpKind::LShift, vec![ci(1), cb(true)]),
            Some(ConstValue::Int(2)),
            "1 << True == 1 << 1 == 2",
        );
        assert_eq!(
            fold(OpKind::RShift, vec![cb(true), ci(0)]),
            Some(ConstValue::Int(1)),
            "True >> 0 == 1 >> 0 == 1",
        );
        // `base_int.__rlshift__` / `__rrshift__` rebuild as
        // `n.__class__`, so a bool left operand and `r_uint` count
        // yield a bool.
        assert_eq!(
            fold(OpKind::LShift, vec![cb(true), cu(1)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::RShift, vec![cb(true), cu(1)]),
            Some(ConstValue::Bool(false)),
        );
        // Bitwise: (Bool, Bool) keeps Bool per Python; mixed widens.
        assert_eq!(
            fold(OpKind::And, vec![cb(true), cb(false)]),
            Some(ConstValue::Bool(false)),
            "Python: True & False == False (bool, not int)",
        );
        assert_eq!(
            fold(OpKind::Or, vec![cb(true), cb(false)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Xor, vec![cb(true), cb(true)]),
            Some(ConstValue::Bool(false)),
        );
        assert_eq!(
            fold(OpKind::And, vec![cb(true), ci(1)]),
            Some(ConstValue::Int(1)),
            "Python: True & 1 == 1 (int — mixed widens to int)",
        );
        assert_eq!(
            fold(OpKind::Or, vec![ci(0), cb(true)]),
            Some(ConstValue::Int(1)),
        );
    }

    #[test]
    fn constfold_arith_int_float_coercion() {
        // Python `int ⊂ float` for arithmetic: `1 + 1.0 == 2.0`,
        // `2 * 0.5 == 1.0`. `coerce_arith` promotes both operands to
        // f64 when either side is Float.
        assert_eq!(
            fold(OpKind::Add, vec![ci(1), cf(1.0)]),
            Some(ConstValue::float(2.0)),
            "Int + Float widens to Float",
        );
        assert_eq!(
            fold(OpKind::Add, vec![cf(2.5), ci(1)]),
            Some(ConstValue::float(3.5)),
        );
        assert_eq!(
            fold(OpKind::Mul, vec![cf(0.5), ci(4)]),
            Some(ConstValue::float(2.0)),
        );
        // Bool → Float coercion via the int rung.
        assert_eq!(
            fold(OpKind::Add, vec![cb(true), cf(1.5)]),
            Some(ConstValue::float(2.5)),
            "True + 1.5 == 2.5",
        );
        assert_eq!(
            fold(OpKind::Mul, vec![cf(2.0), cb(true)]),
            Some(ConstValue::float(2.0)),
        );
        // TrueDiv: Python 3 `1 / 2 == 0.5` (always Float).
        assert_eq!(
            fold(OpKind::TrueDiv, vec![ci(1), ci(2)]),
            Some(ConstValue::float(0.5)),
            "Python 3: int / int == float",
        );
        assert_eq!(
            fold(OpKind::TrueDiv, vec![cb(true), ci(4)]),
            Some(ConstValue::float(0.25)),
            "True / 4 == 1.0 / 4.0 == 0.25",
        );
        // `_truediv` / `_bigint_true_divide` divide the integers
        // exactly and round once, including above 2**53.
        assert_eq!(
            fold(OpKind::TrueDiv, vec![ci(1i64 << 54), ci(2)]),
            Some(ConstValue::float((1i64 << 53) as f64)),
        );
        assert_eq!(
            fold(OpKind::TrueDiv, vec![cu(9007199254740993), cu(3)]),
            Some(ConstValue::float(3002399751580331.0)),
        );
        assert_eq!(
            fold(OpKind::TrueDiv, vec![ci((1i64 << 53) + 1), ci(1)]),
            Some(ConstValue::float(9007199254740992.0)),
        );
        assert_eq!(
            fold(OpKind::TrueDiv, vec![ci(-(1i64 << 60) - 1), ci(3)]),
            Some(ConstValue::float(-384307168202282304.0)),
        );
        assert_eq!(
            fold(OpKind::TrueDiv, vec![ci(1i64 << 53), ci(1)]),
            Some(ConstValue::float((1i64 << 53) as f64)),
        );
        // FloorDiv / Mod over Float: Python `7.0 // 2.0 == 3.0`.
        assert_eq!(
            fold(OpKind::FloorDiv, vec![cf(7.0), cf(2.0)]),
            Some(ConstValue::float(3.0)),
        );
        assert_eq!(
            fold(OpKind::FloorDiv, vec![cf(-7.0), cf(2.0)]),
            Some(ConstValue::float(-4.0)),
            "Python: (-7.0) // 2.0 == -4.0 (floor-toward-negative-inf)",
        );
        assert_eq!(
            fold(OpKind::Mod, vec![cf(7.0), cf(2.0)]),
            Some(ConstValue::float(1.0)),
        );
        assert_eq!(
            fold(OpKind::Mod, vec![cf(-7.0), cf(2.0)]),
            Some(ConstValue::float(1.0)),
            "Python: (-7.0) % 2.0 == 1.0 (sign of divisor)",
        );
    }

    #[test]
    fn constfold_div_diverges_from_floordiv_on_floats() {
        // `add_operator('div', ...)` resolves to Python 2's
        // `operator.div`, which on floats is true division — so
        // `div(7.0, 2.0) == 3.5`. `add_operator('floordiv', ...)`
        // resolves to `operator.floordiv` and gives `3.0`. Folding
        // them together would silently produce a wrong constant for
        // `div` on floats. On ints both operators agree (floor
        // toward `-inf`), so the Int arm is shared.
        assert_eq!(
            fold(OpKind::Div, vec![cf(7.0), cf(2.0)]),
            Some(ConstValue::float(3.5)),
            "div(7.0, 2.0) == 3.5 (operator.div on floats == true division)",
        );
        assert_eq!(
            fold(OpKind::FloorDiv, vec![cf(7.0), cf(2.0)]),
            Some(ConstValue::float(3.0)),
            "floordiv(7.0, 2.0) == 3.0 (operator.floordiv)",
        );
        // Int/Int parity: both Div and FloorDiv fold via int_py_floor_div.
        assert_eq!(
            fold(OpKind::Div, vec![ci(-7), ci(2)]),
            Some(ConstValue::Int(-4)),
            "div(-7, 2) == -4 (Python 3 floor-toward-negative-inf)",
        );
        assert_eq!(
            fold(OpKind::FloorDiv, vec![ci(-7), ci(2)]),
            Some(ConstValue::Int(-4)),
        );
    }

    #[test]
    fn constfold_float_mod_signed_zero_matches_pypy() {
        // Line-by-line port test for `pypy/objspace/std/floatobject.py descr_mod
        // descr_mod`'s signed-zero handling. The naive `x - y * (x /
        // y).floor()` produces `+0.0` regardless of denominator sign;
        // upstream uses `mod = math.copysign(0.0, y)` so an exact
        // multiple of a *negative* divisor yields `-0.0`. Pin the
        // bit-level result so a future regression is caught.
        let bits = |v: f64| v.to_bits();
        let pos_zero = bits(0.0_f64);
        let neg_zero = bits(-0.0_f64);

        // 4.0 % 2.0: positive divisor → +0.0
        let r = match fold(OpKind::Mod, vec![cf(4.0), cf(2.0)]) {
            Some(ConstValue::Float(b)) => b,
            other => panic!("expected float, got {other:?}"),
        };
        assert_eq!(r, pos_zero, "4.0 % 2.0: positive divisor preserves +0.0",);

        // 4.0 % -2.0: negative divisor → -0.0 (signed-zero parity)
        let r = match fold(OpKind::Mod, vec![cf(4.0), cf(-2.0)]) {
            Some(ConstValue::Float(b)) => b,
            other => panic!("expected float, got {other:?}"),
        };
        assert_eq!(
            r, neg_zero,
            "4.0 % -2.0: copysign(0.0, -2.0) == -0.0 (PyPy parity)",
        );

        // -4.0 % -2.0: negative divisor → -0.0
        let r = match fold(OpKind::Mod, vec![cf(-4.0), cf(-2.0)]) {
            Some(ConstValue::Float(b)) => b,
            other => panic!("expected float, got {other:?}"),
        };
        assert_eq!(r, neg_zero, "-4.0 % -2.0: copysign(0.0, -2.0) == -0.0",);

        // -4.0 % 2.0: positive divisor → +0.0
        let r = match fold(OpKind::Mod, vec![cf(-4.0), cf(2.0)]) {
            Some(ConstValue::Float(b)) => b,
            other => panic!("expected float, got {other:?}"),
        };
        assert_eq!(r, pos_zero, "-4.0 % 2.0: copysign(0.0, 2.0) == +0.0",);
    }

    #[test]
    fn constfold_float_floordiv_matches_pypy_divmod_w() {
        // Spot-check `pypy/objspace/std/floatobject.py _divmod_w`
        // floordiv path. Sign-mismatch correction (`div -= 1.0`) and
        // snap-to-nearest pass land on the same ordinary integers as
        // the naive `(x / y).floor()` for these inputs, so this test
        // mainly locks the contract that the helper is wired up.
        assert_eq!(
            fold(OpKind::FloorDiv, vec![cf(7.0), cf(2.0)]),
            Some(ConstValue::float(3.0)),
        );
        assert_eq!(
            fold(OpKind::FloorDiv, vec![cf(-7.0), cf(2.0)]),
            Some(ConstValue::float(-4.0)),
            "Python: (-7.0) // 2.0 == -4.0 (sign-mismatch correction)",
        );
        assert_eq!(
            fold(OpKind::FloorDiv, vec![cf(7.0), cf(-2.0)]),
            Some(ConstValue::float(-4.0)),
            "Python: 7.0 // -2.0 == -4.0",
        );
        assert_eq!(
            fold(OpKind::FloorDiv, vec![cf(-7.0), cf(-2.0)]),
            Some(ConstValue::float(3.0)),
        );
        // Exact multiple — quotient is integer-valued.
        assert_eq!(
            fold(OpKind::FloorDiv, vec![cf(4.0), cf(2.0)]),
            Some(ConstValue::float(2.0)),
        );
        assert_eq!(
            fold(OpKind::FloorDiv, vec![cf(4.0), cf(-2.0)]),
            Some(ConstValue::float(-2.0)),
        );
        // Zero numerator — quotient is zero with the sign of x/y. Pin
        // bits so the snap-zero fallback (`div * div * x / y`) shape
        // is locked.
        let r = match fold(OpKind::FloorDiv, vec![cf(0.0), cf(2.0)]) {
            Some(ConstValue::Float(b)) => b,
            other => panic!("expected float, got {other:?}"),
        };
        assert_eq!(r, 0.0_f64.to_bits(), "0.0 // 2.0 == +0.0");
        let r = match fold(OpKind::FloorDiv, vec![cf(0.0), cf(-2.0)]) {
            Some(ConstValue::Float(b)) => b,
            other => panic!("expected float, got {other:?}"),
        };
        assert_eq!(r, (-0.0_f64).to_bits(), "0.0 // -2.0 == -0.0");
    }

    #[test]
    fn constfold_bitwise() {
        assert_eq!(
            fold(OpKind::And, vec![ci(0b1100), ci(0b1010)]),
            Some(ConstValue::Int(0b1000))
        );
        assert_eq!(
            fold(OpKind::Or, vec![ci(0b1100), ci(0b1010)]),
            Some(ConstValue::Int(0b1110))
        );
        assert_eq!(
            fold(OpKind::Xor, vec![ci(0b1100), ci(0b1010)]),
            Some(ConstValue::Int(0b0110))
        );
        assert_eq!(
            fold(OpKind::LShift, vec![ci(1), ci(4)]),
            Some(ConstValue::Int(16))
        );
        assert_eq!(
            fold(OpKind::RShift, vec![ci(16), ci(2)]),
            Some(ConstValue::Int(4))
        );
    }

    #[test]
    fn constfold_float_arithmetic() {
        assert_eq!(
            fold(OpKind::Add, vec![cf(1.5), cf(2.25)]),
            Some(ConstValue::float(3.75))
        );
        assert_eq!(
            fold(OpKind::Sub, vec![cf(10.0), cf(0.5)]),
            Some(ConstValue::float(9.5))
        );
        assert_eq!(
            fold(OpKind::Mul, vec![cf(2.0), cf(3.0)]),
            Some(ConstValue::float(6.0))
        );
        assert_eq!(
            fold(OpKind::TrueDiv, vec![cf(7.0), cf(2.0)]),
            Some(ConstValue::float(3.5))
        );
        // float zero divisor: upstream `operator.truediv(1.0, 0.0)`
        // raises `ZeroDivisionError`. `PureOperation.constfold` re-
        // raises that as `FlowingError`, so this must be observed as
        // `Err`, not silent decline. (The test helper `fold` collapses
        // `Err` to `None`, so we drive `constfold` directly here.)
        let op = HLOperation::new(OpKind::TrueDiv, vec![cf(1.0), cf(0.0)]);
        let err = op
            .constfold()
            .expect_err("float zero divisor must raise FlowingError");
        assert!(err.message.contains("ZeroDivisionError"), "{}", err.message);
    }

    #[test]
    fn constfold_comparisons() {
        assert_eq!(
            fold(OpKind::Lt, vec![ci(1), ci(2)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Gt, vec![ci(1), ci(2)]),
            Some(ConstValue::Bool(false))
        );
        assert_eq!(
            fold(OpKind::Eq, vec![ci(3), ci(3)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Ne, vec![ci(3), ci(4)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Le, vec![cs("a"), cs("b")]),
            Some(ConstValue::Bool(true))
        );
        // cross-type ordering comparison: upstream Python 3 raises
        // `TypeError("'<' not supported between instances of 'int' and
        // 'str'")`. `PureOperation.constfold` re-raises as
        // `FlowingError`. Eq/Ne are deliberately NOT raised — Python
        // 3 returns Bool for cross-type identity comparison.
        let op = HLOperation::new(OpKind::Lt, vec![ci(1), cs("a")]);
        let err = op
            .constfold()
            .expect_err("cross-type Lt must raise FlowingError");
        assert!(err.message.contains("TypeError"), "{}", err.message);
        // Eq/Ne tolerate cross-type — fold returns Bool(false)/Bool(true).
        assert_eq!(
            fold(OpKind::Eq, vec![ci(1), cs("a")]),
            Some(ConstValue::Bool(false))
        );
        assert_eq!(
            fold(OpKind::Ne, vec![ci(1), cs("a")]),
            Some(ConstValue::Bool(true))
        );
    }

    #[test]
    fn constfold_cross_type_numeric_ordering() {
        // upstream `operator.lt` / `operator.gt` / `operator.le` /
        // `operator.ge` over int/float/bool admit numeric coercion:
        // `1 < 1.5`, `True < 2`, `1.0 > False` are all foldable.
        // Pre-fix `cmp_fold` returned `None` for cross-type pairs,
        // forcing the runtime to evaluate trivially-foldable ordering
        // — wasted op + lost FlowingError surface for impossible-
        // ordering cases.
        // Int < Float.
        assert_eq!(
            fold(OpKind::Lt, vec![ci(1), cf(1.5)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Gt, vec![cf(2.0), ci(1)]),
            Some(ConstValue::Bool(true)),
        );
        // Bool < Int (bool subclass of int).
        assert_eq!(
            fold(OpKind::Lt, vec![cb(true), ci(2)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Le, vec![cb(false), ci(0)]),
            Some(ConstValue::Bool(true)),
        );
        // Bool < Float.
        assert_eq!(
            fold(OpKind::Lt, vec![cb(true), cf(1.5)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Ge, vec![cf(0.0), cb(false)]),
            Some(ConstValue::Bool(true)),
        );
        // Cross-NON-numeric (Int vs UniStr) still raises FlowingError
        // via `cross_type_ordering` whitelist (sanity).
        let op = HLOperation::new(OpKind::Lt, vec![ci(1), cs("a")]);
        let err = op.constfold().expect_err("Int < UniStr must raise");
        assert!(err.message.contains("TypeError"), "{}", err.message);
    }

    #[test]
    fn constfold_eq_ne_container_deep_equality() {
        // upstream tuple/list `==` compares element-wise via Python's
        // `==` per element, so `(True,) == (1,)` is True (Bool ↔ Int
        // numeric coercion). Pre-fix the variant-strict fallback
        // would return False, mismatching upstream constfold.
        let t_true = ConstValue::Tuple(vec![ConstValue::Bool(true)]);
        let t_one = ConstValue::Tuple(vec![ConstValue::Int(1)]);
        assert_eq!(
            fold(OpKind::Eq, vec![c(t_true.clone()), c(t_one.clone())]),
            Some(ConstValue::Bool(true)),
            "(True,) == (1,) must fold True per element-wise numeric coercion",
        );
        assert_eq!(
            fold(OpKind::Ne, vec![c(t_true), c(t_one)]),
            Some(ConstValue::Bool(false)),
        );
        // Length mismatch always False.
        let t_two = ConstValue::Tuple(vec![ConstValue::Int(1), ConstValue::Int(2)]);
        let t_one2 = ConstValue::Tuple(vec![ConstValue::Int(1)]);
        assert_eq!(
            fold(OpKind::Eq, vec![c(t_two), c(t_one2)]),
            Some(ConstValue::Bool(false)),
        );
        // List vs Tuple — distinct container types remain unequal
        // (Python 3: `[1] == (1,)` is False).
        let lst = ConstValue::List(Arc::new(vec![ConstValue::Int(1)]));
        let tup = ConstValue::Tuple(vec![ConstValue::Int(1)]);
        assert_eq!(
            fold(OpKind::Eq, vec![c(lst), c(tup)]),
            Some(ConstValue::Bool(false)),
        );
        // Nested element-wise — `((True,),)` vs `((1,),)`.
        let nested_a = ConstValue::Tuple(vec![ConstValue::Tuple(vec![ConstValue::Bool(true)])]);
        let nested_b = ConstValue::Tuple(vec![ConstValue::Tuple(vec![ConstValue::Int(1)])]);
        assert_eq!(
            fold(OpKind::Eq, vec![c(nested_a), c(nested_b)]),
            Some(ConstValue::Bool(true)),
        );
    }

    #[test]
    fn constfold_eq_ne_python_numeric_coercion() {
        // upstream `PureOperation.constfold` calls `operator.eq(*args)`
        // / `operator.ne(*args)` which dispatches Python's numeric
        // coercion: `True == 1`, `1 == 1.0`, `True == 1.0` all return
        // `True`. The naive Rust `ConstValue::PartialEq` derives
        // variant-strict equality and would say those are unequal —
        // diverging from constfold parity.
        // Bool ↔ Int.
        assert_eq!(
            fold(OpKind::Eq, vec![cb(true), ci(1)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Eq, vec![ci(1), cb(true)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Eq, vec![cb(false), ci(0)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Ne, vec![cb(true), ci(0)]),
            Some(ConstValue::Bool(true)),
        );
        // Int ↔ Float.
        assert_eq!(
            fold(OpKind::Eq, vec![ci(1), cf(1.0)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Eq, vec![cf(1.5), ci(1)]),
            Some(ConstValue::Bool(false)),
        );
        // Bool ↔ Float.
        assert_eq!(
            fold(OpKind::Eq, vec![cb(true), cf(1.0)]),
            Some(ConstValue::Bool(true)),
        );
        // Distinct types stay False (no numeric coercion).
        assert_eq!(
            fold(OpKind::Eq, vec![ci(1), cs("1")]),
            Some(ConstValue::Bool(false)),
        );
        // `r_uint(1) == 1 == True`; `r_uint(2**64-1) != -1`.
        assert_eq!(
            fold(OpKind::Eq, vec![cu(1), ci(1)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Eq, vec![cu(1), cb(true)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Eq, vec![cu(u64::MAX), ci(-1)]),
            Some(ConstValue::Bool(false)),
        );
        assert_eq!(
            fold(OpKind::Lt, vec![cu(1), ci(2)]),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            fold(OpKind::Gt, vec![cu(u64::MAX), ci(-1)]),
            Some(ConstValue::Bool(true)),
        );
    }

    #[test]
    fn constfold_bool_conversion() {
        assert_eq!(
            fold(OpKind::Bool, vec![ci(0)]),
            Some(ConstValue::Bool(false))
        );
        assert_eq!(
            fold(OpKind::Bool, vec![ci(5)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Bool, vec![cb(true)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Bool, vec![cs("")]),
            Some(ConstValue::Bool(false))
        );
        assert_eq!(
            fold(OpKind::Bool, vec![cs("x")]),
            Some(ConstValue::Bool(true))
        );
    }

    #[test]
    fn constfold_string_and_tuple_concat() {
        assert_eq!(
            fold(OpKind::Add, vec![cs("hi"), cs(" there")]),
            Some(ConstValue::byte_str("hi there"))
        );
        assert_eq!(
            fold(
                OpKind::Add,
                vec![
                    c(ConstValue::Tuple(vec![ConstValue::Int(1)])),
                    c(ConstValue::Tuple(vec![
                        ConstValue::Int(2),
                        ConstValue::Int(3)
                    ])),
                ],
            ),
            Some(ConstValue::Tuple(vec![
                ConstValue::Int(1),
                ConstValue::Int(2),
                ConstValue::Int(3),
            ]))
        );
    }

    #[test]
    fn constfold_identity() {
        assert_eq!(
            fold(OpKind::Is, vec![ci(3), ci(3)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Is, vec![ci(3), ci(4)]),
            Some(ConstValue::Bool(false))
        );
    }

    #[test]
    fn constfold_len() {
        assert_eq!(fold(OpKind::Len, vec![cs("abc")]), Some(ConstValue::Int(3)));
        assert_eq!(
            fold(
                OpKind::Len,
                vec![c(ConstValue::Tuple(vec![
                    ConstValue::Int(1),
                    ConstValue::Int(2)
                ]))],
            ),
            Some(ConstValue::Int(2))
        );
    }

    #[test]
    fn constfold_int_conversion() {
        assert_eq!(fold(OpKind::Int, vec![cf(3.75)]), Some(ConstValue::Int(3)));
        assert_eq!(fold(OpKind::Int, vec![cf(f64::INFINITY)]), None);
        assert_eq!(fold(OpKind::Int, vec![cf(f64::NAN)]), None);
        assert_eq!(fold(OpKind::Int, vec![cb(true)]), Some(ConstValue::Int(1)));
        assert_eq!(fold(OpKind::Int, vec![cu(7)]), Some(ConstValue::Int(7)));
        assert_eq!(fold(OpKind::Int, vec![cu(i64::MAX as u64 + 1)]), None);
        assert_eq!(
            fold(OpKind::Float, vec![cu(7)]),
            Some(ConstValue::float(7.0))
        );
        assert_eq!(fold(OpKind::Long, vec![cu(7)]), Some(ConstValue::Int(7)));
        assert_eq!(fold(OpKind::Long, vec![cu(i64::MAX as u64 + 1)]), None);
    }

    #[test]
    fn constfold_newtuple_is_variadic_pure() {
        // `NewTuple(PureOperation)` with `pyfunc = lambda *args: args`
        // (operation.py:542-548).
        assert_eq!(
            fold(OpKind::NewTuple, vec![ci(1), ci(2), cs("x")]),
            Some(ConstValue::Tuple(vec![
                ConstValue::Int(1),
                ConstValue::Int(2),
                ConstValue::byte_str("x"),
            ]))
        );
        // empty tuple.
        assert_eq!(
            fold(OpKind::NewTuple, vec![]),
            Some(ConstValue::Tuple(Vec::new()))
        );
    }

    #[test]
    fn constfold_pow_int_nomod() {
        // `Pow(PureOperation)` arity 3, `pyfunc = pow`; default third
        // arg is `Constant(None)` (operation.py:575).
        let none = c(ConstValue::None);
        assert_eq!(
            fold(OpKind::Pow, vec![ci(3), ci(4), none.clone()]),
            Some(ConstValue::Int(81))
        );
        // Negative exponent → not an int result → decline fold.
        assert_eq!(fold(OpKind::Pow, vec![ci(2), ci(-1), none.clone()]), None);
        // Overflow → i64::checked_pow returns None → decline fold.
        assert_eq!(fold(OpKind::Pow, vec![ci(2), ci(64), none]), None);
    }

    #[test]
    fn from_opname_round_trips_every_variant() {
        // Sample a few canonical strings; full round-trip is
        // brittle in a unit test, so we assert the ones most likely
        // to regress.
        for kind in [
            OpKind::Add,
            OpKind::AddOvf,
            OpKind::Is,
            OpKind::And,
            OpKind::Yield,
            OpKind::GetItemIdx,
            OpKind::SimpleCall,
            OpKind::CallArgs,
            OpKind::NewTuple,
            OpKind::Pow,
            OpKind::GetAttr,
        ] {
            let name = kind.opname();
            assert_eq!(
                OpKind::from_opname(name),
                Some(kind),
                "round-trip failed for {kind:?}"
            );
        }

        // Unknown majit-synthetic opnames decline.
        assert_eq!(OpKind::from_opname("not_"), None);
        assert_eq!(OpKind::from_opname("newset"), None);
        assert_eq!(OpKind::from_opname("ll_assert_not_none"), None);
    }

    #[test]
    fn ovfchecked_rewrites_kind() {
        let c1 = Hlvalue::Constant(Constant::new(ConstValue::Int(1)));
        let c2 = Hlvalue::Constant(Constant::new(ConstValue::Int(2)));
        let op = HLOperation::new(OpKind::Add, vec![c1.clone(), c2.clone()]);
        let ovf = op.ovfchecked().expect("add has ovf variant");
        assert_eq!(ovf.kind, OpKind::AddOvf);
        assert_eq!(ovf.args, vec![c1, c2]);

        // Ops without an ovf twin return None.
        let eq_op = HLOperation::new(OpKind::Eq, Vec::new());
        assert!(eq_op.ovfchecked().is_none());
    }

    #[test]
    fn into_space_operation_preserves_opname_and_args() {
        let c1 = Hlvalue::Constant(Constant::new(ConstValue::Int(1)));
        let c2 = Hlvalue::Constant(Constant::new(ConstValue::Int(2)));
        let op = HLOperation::new(OpKind::Add, vec![c1.clone(), c2.clone()]);
        let result_var = op.result.clone();
        let spaceop = op.into_space_operation();
        assert_eq!(spaceop.opname, "add");
        assert_eq!(spaceop.args, vec![c1, c2]);
        assert_eq!(spaceop.result, Hlvalue::Variable(result_var));
        assert_eq!(spaceop.offset, -1);
    }

    #[test]
    fn replace_remaps_args_and_result() {
        let src = Variable::new();
        let dst = Variable::new();
        let mut mapping: HashMap<Variable, Hlvalue> = HashMap::new();
        mapping.insert(src.clone(), Hlvalue::Variable(dst.clone()));

        let op = HLOperation {
            kind: OpKind::Add,
            args: vec![
                Hlvalue::Variable(src.clone()),
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
            ],
            result: src.clone(),
            offset: 42,
        };
        let replaced = op.replace(&mapping);
        assert_eq!(replaced.kind, OpKind::Add);
        assert_eq!(
            replaced.args,
            vec![
                Hlvalue::Variable(dst.clone()),
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
            ]
        );
        assert_eq!(replaced.result, dst);
        assert_eq!(replaced.offset, 42);
    }
}
