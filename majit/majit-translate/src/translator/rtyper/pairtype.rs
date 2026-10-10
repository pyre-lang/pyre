//! rtyper-side `class __extend__(pairtype(R_A, R_B))` double-dispatch glue.
//!
//! ## NOT a port of `rpython/tool/pairtype.py`
//!
//! Upstream's `tool/pairtype.py` (134 LOC — `pair()` / `pairtype()` /
//! `pairmro()` / `DoubleDispatchRegistry` / `extendabletype` metaclass)
//! is ported at [`crate::tool::pairtype`]. This file instead centralises
//! the `class __extend__(pairtype(R_A, R_B))` extension blocks that
//! upstream scatters across every `r*.py` file (rnone.py,
//! rbool.py, rfloat.py:75-135, rint.py, rptr.py, ...).  allow-line-citation
//! Python's metaclass machinery makes that distribution invisible —
//! each `class __extend__` block silently binds to the pair class
//! produced by `pairtype(R_A, R_B)`. Rust has no metaclass, so the
//! binding surface must be explicit; this module provides the
//! centralization point and enumerates the (R_A, R_B) arms that
//! upstream's metaclass resolves at import time.
//!
//! ## Pieces
//!
//! 1. [`ReprClassId`] — explicit enum tag that stands in for upstream's
//!    `type(repr)` identity. The [`Repr::repr_class_id`] default
//!    returns [`ReprClassId::Repr`] (the wildcard base used by
//!    `pairtype(Repr, X)` / `pairtype(X, Repr)` extension blocks —
//!    e.g. rnone.py or rmodel.py); every concrete `Repr`
//!    overrides to return its specific variant.
//!
//! 2. [`ReprClassId::mro`] + [`pair_mro`] — per-class MRO and its
//!    cross-product, mirroring upstream `pairtype.py pairmro`.
//!    MRO chains for `BoolRepr → IntegerRepr → FloatRepr → Repr` etc.
//!    match upstream Python class inheritance, so pairtype entries
//!    resolve through the same walk order.
//!
//! 3. `pair_convert_from_to` + `pair_rtype_*` central dispatchers —
//!    the per-(R_A, R_B) match arms that upstream metaclass lookup
//!    resolves implicitly. Each arm delegates to a concrete
//!    helper in the matching `r*.rs` module. Adding a new
//!    `class __extend__(pairtype(R_X, R_Y))` block upstream means
//!    adding an arm here + a helper in `r_x.rs` or `r_y.rs`.
//!
//! ## Why an enum instead of `TypeId`
//!
//! Using `std::any::TypeId` via an `Any` supertrait on `Repr` would
//! avoid the enum, but the enum surfaces the mapping explicitly —
//! every new `Repr` port must add an arm here and update the match
//! tables in concrete dispatchers. That explicitness catches missing
//! pairtype wiring at compile time, whereas TypeId-keyed HashMaps only
//! fail at runtime.

use crate::flowspace::model::{ConstValue, Hlvalue};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, inputconst_from_lltype};
use crate::translator::rtyper::rtyper::{ConvertedTo, GenopResult, HighLevelOp, LowLevelOpList};

/// Explicit tag used by [`Repr::repr_class_id`] to drive pairtype
/// lookup.
///
/// Variants mirror each concrete `class ...Repr(Repr)` in
/// `rpython/rtyper/`. The wildcard [`ReprClassId::Repr`] corresponds
/// to `pairtype(Repr, X)` / `pairtype(X, Repr)` blocks — upstream's
/// base-class catch-all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReprClassId {
    /// Catch-all matching upstream's `class __extend__(pairtype(Repr,
    /// X))` — any unregistered concrete Repr falls back to this tag.
    Repr,
    /// `rmodel.py VoidRepr`.
    VoidRepr,
    /// `rmodel.py SimplePointerRepr`.
    SimplePointerRepr,
    /// `rnone.py NoneRepr`.
    NoneRepr,
    /// `rbool.py BoolRepr(IntegerRepr)` — MRO includes
    /// [`ReprClassId::IntegerRepr`] + [`ReprClassId::FloatRepr`] so
    /// integer / float pairtype entries resolve transitively, matching
    /// upstream's inheritance chain.
    BoolRepr,
    /// `rint.py IntegerRepr(FloatRepr)` — MRO includes
    /// [`ReprClassId::FloatRepr`].
    IntegerRepr,
    /// `rfloat.py FloatRepr`.
    FloatRepr,
    /// `rfloat.py SingleFloatRepr`.
    SingleFloatRepr,
    /// `rfloat.py LongFloatRepr`.
    LongFloatRepr,
    /// `raddress.py AddressRepr`.
    AddressRepr,
    /// `raddress.py TypedAddressAccessRepr`.
    TypedAddressAccessRepr,
    /// `rptr.py PtrRepr`.
    PtrRepr,
    /// `rptr.py InteriorPtrRepr`.
    InteriorPtrRepr,
    /// `rptr.py LLADTMethRepr`.
    LLADTMethRepr,
    /// `rpbc.py FunctionRepr`.
    FunctionRepr,
    /// `rpbc.py FunctionsPBCRepr`.
    FunctionsPBCRepr,
    /// `rpbc.py SmallFunctionSetPBCRepr`.
    SmallFunctionSetPBCRepr,
    /// `rbuiltin.py BuiltinFunctionRepr`.
    BuiltinFunctionRepr,
    /// `rbuiltin.py BuiltinMethodRepr`.
    BuiltinMethodRepr,
    /// `rpbc.py SingleFrozenPBCRepr`.
    SingleFrozenPBCRepr,
    /// `rpbc.py MultipleUnrelatedFrozenPBCRepr`.
    MultipleUnrelatedFrozenPBCRepr,
    /// `rpbc.py MultipleFrozenPBCRepr`.
    MultipleFrozenPBCRepr,
    /// `rpbc.py MethodOfFrozenPBCRepr`.
    MethodOfFrozenPBCRepr,
    /// `rpbc.py MethodsPBCRepr`.
    MethodsPBCRepr,
    /// `controllerentry.py ControlledInstanceRepr`.
    ControlledInstanceRepr,
    /// `rpbc.py ClassesPBCRepr`.
    ClassesPBCRepr,
    /// `rclass.py InstanceRepr(Repr)`.
    InstanceRepr,
    /// `rtuple.py TupleRepr`.
    TupleRepr,
    /// `lltypesystem/rlist.py FixedSizeListRepr(
    /// AbstractFixedSizeListRepr, BaseListRepr)` — the non-resized list
    /// whose `LIST` lowers to a bare `Ptr(GcArray(ITEM))`. Slices
    /// (`&[T]`) annotate as a non-resized `SomeList`, so this is the
    /// repr the `len` lowering lands on.
    FixedSizeListRepr,
    /// `lltypesystem/rlist.py ListRepr(AbstractListRepr,
    /// BaseListRepr)` — the resized list whose `LIST` lowers to a
    /// `Ptr(GcStruct("list", ("length", Signed), ("items",
    /// Ptr(GcArray(ITEM)))))`. Minted when the `listdef` is `resized`
    /// (an `.append()` consumer marks it so).
    ListRepr,
    /// `rrustvec.rs RustVecRepr` — a Rust `Vec<T>` lowered to a pointer to
    /// its raw `{ptr, len, cap}` header.
    RustVecRepr,
    /// `rdict.py AbstractDictRepr`.
    AbstractDictRepr,
    /// `lltypesystem/rdict.py DictRepr(AbstractDictRepr)`.
    DictRepr,
    /// `lltypesystem/rordereddict.py OrderedDictRepr(AbstractDictRepr)`.
    OrderedDictRepr,
    /// `rdict.py AbstractDictIteratorRepr`.
    AbstractDictIteratorRepr,
    /// `lltypesystem/rdict.py DictIteratorRepr(AbstractDictIteratorRepr)`.
    DictIteratorRepr,
    /// `rstr.py:483 AbstractCharRepr` (`CharRepr` lltypesystem
    /// realisation, `lowleveltype = Char`).
    CharRepr,
    /// `rstr.py:758 AbstractUniCharRepr` (`UniCharRepr` lltypesystem
    /// realisation, `lowleveltype = UniChar`).
    UniCharRepr,
    /// `rstr.py:95 AbstractStringRepr` (`lltypesystem.rstr.StringRepr`
    /// realisation, `lowleveltype = Ptr(STR)`).
    StringRepr,
    /// `rstr.py:450 AbstractUnicodeRepr` (`lltypesystem.rstr.UnicodeRepr`
    /// realisation, `lowleveltype = Ptr(UNICODE)`).
    UnicodeRepr,
    /// `rbytearray.py AbstractByteArrayRepr(AbstractStringRepr)`
    /// (`lltypesystem.rbytearray.ByteArrayRepr`, `lowleveltype =
    /// Ptr(BYTEARRAY)`).
    ByteArrayRepr,
    /// Abstract base shared by `StringRepr` and `UnicodeRepr`.
    AbstractStringRepr,
    /// `rweakref.py WeakRefRepr(BaseWeakRefRepr)`.
    WeakRefRepr,
    /// `rweakref.py EmulatedWeakRefRepr(BaseWeakRefRepr)`.
    EmulatedWeakRefRepr,
    /// `lltypesystem/rgcref.py GCRefRepr`.
    GCRefRepr,
    /// `rrange.py:43 RangeRepr(AbstractRangeRepr)` — the immutable
    /// `range()`-result list repr (`GcStruct("range", start, stop)`).
    RangeRepr,
    /// `rlist.py AbstractListIteratorRepr(IteratorRepr)` — the
    /// iterator over a list/slice (`GcStruct("listiter", ("list", LIST),
    /// ("index", Signed))`).
    ListIteratorRepr,
    /// `rrange.py AbstractRangeIteratorRepr(IteratorRepr)` — the
    /// iterator over a `range()` list (`GcStruct("range", ("next",
    /// Signed), ("stop", Signed))`). Distinct from `RangeRepr`: an
    /// iterator repr is its own iterator, not the container.
    RangeIteratorRepr,
    /// `rrange.py EnumerateIteratorRepr(IteratorRepr)` — `enumerate()`
    /// over a list, sharing the base list iterator's lowleveltype.
    EnumerateIteratorRepr,
}

impl ReprClassId {
    /// Upstream MRO for this repr class. Used by [`pair_mro`] to build
    /// the double-dispatch resolution order.
    ///
    /// Each slice starts with `self` and ends with
    /// [`ReprClassId::Repr`] (the ultimate base). Intermediate entries
    /// mirror upstream's Python `__mro__`:
    ///
    /// | concrete | upstream chain |
    /// |---|---|
    /// | `BoolRepr` | `BoolRepr → IntegerRepr → FloatRepr → Repr` |
    /// | `IntegerRepr` | `IntegerRepr → FloatRepr → Repr` |
    /// | any other | `Self → Repr` |
    pub fn mro(self) -> &'static [ReprClassId] {
        use ReprClassId::*;
        match self {
            Repr => &[Repr],
            VoidRepr => &[VoidRepr, Repr],
            SimplePointerRepr => &[SimplePointerRepr, Repr],
            NoneRepr => &[NoneRepr, Repr],
            BoolRepr => &[BoolRepr, IntegerRepr, FloatRepr, Repr],
            IntegerRepr => &[IntegerRepr, FloatRepr, Repr],
            FloatRepr => &[FloatRepr, Repr],
            SingleFloatRepr => &[SingleFloatRepr, Repr],
            LongFloatRepr => &[LongFloatRepr, Repr],
            AddressRepr => &[AddressRepr, Repr],
            TypedAddressAccessRepr => &[TypedAddressAccessRepr, Repr],
            PtrRepr => &[PtrRepr, Repr],
            InteriorPtrRepr => &[InteriorPtrRepr, Repr],
            LLADTMethRepr => &[LLADTMethRepr, Repr],
            FunctionRepr => &[FunctionRepr, Repr],
            FunctionsPBCRepr => &[FunctionsPBCRepr, Repr],
            SmallFunctionSetPBCRepr => &[SmallFunctionSetPBCRepr, Repr],
            BuiltinFunctionRepr => &[BuiltinFunctionRepr, Repr],
            BuiltinMethodRepr => &[BuiltinMethodRepr, Repr],
            SingleFrozenPBCRepr => &[SingleFrozenPBCRepr, Repr],
            MultipleUnrelatedFrozenPBCRepr => &[MultipleUnrelatedFrozenPBCRepr, Repr],
            MultipleFrozenPBCRepr => &[MultipleFrozenPBCRepr, Repr],
            MethodOfFrozenPBCRepr => &[MethodOfFrozenPBCRepr, Repr],
            MethodsPBCRepr => &[MethodsPBCRepr, Repr],
            ControlledInstanceRepr => &[ControlledInstanceRepr, Repr],
            ClassesPBCRepr => &[ClassesPBCRepr, Repr],
            InstanceRepr => &[InstanceRepr, Repr],
            FixedSizeListRepr => &[FixedSizeListRepr, Repr],
            ListRepr => &[ListRepr, Repr],
            RustVecRepr => &[RustVecRepr, Repr],
            TupleRepr => &[TupleRepr, Repr],
            AbstractDictRepr => &[AbstractDictRepr, Repr],
            DictRepr => &[DictRepr, AbstractDictRepr, Repr],
            OrderedDictRepr => &[OrderedDictRepr, AbstractDictRepr, Repr],
            AbstractDictIteratorRepr => &[AbstractDictIteratorRepr, Repr],
            DictIteratorRepr => &[DictIteratorRepr, AbstractDictIteratorRepr, Repr],
            CharRepr => &[CharRepr, Repr],
            UniCharRepr => &[UniCharRepr, Repr],
            StringRepr => &[StringRepr, AbstractStringRepr, Repr],
            UnicodeRepr => &[UnicodeRepr, AbstractStringRepr, Repr],
            ByteArrayRepr => &[ByteArrayRepr, AbstractStringRepr, Repr],
            AbstractStringRepr => &[AbstractStringRepr, Repr],
            WeakRefRepr => &[WeakRefRepr, Repr],
            EmulatedWeakRefRepr => &[EmulatedWeakRefRepr, Repr],
            GCRefRepr => &[GCRefRepr, Repr],
            RangeRepr => &[RangeRepr, Repr],
            // `ListIteratorRepr → AbstractListIteratorRepr → IteratorRepr
            // → Repr`; the abstract bases carry no pairtype entries, so
            // the resolution chain collapses to `Self → Repr`.
            ListIteratorRepr => &[ListIteratorRepr, Repr],
            RangeIteratorRepr => &[RangeIteratorRepr, Repr],
            EnumerateIteratorRepr => &[EnumerateIteratorRepr, Repr],
        }
    }
}

/// RPython `pairmro(cls1, cls2)` (`pairtype.py`) — Repr-side wrapper.
///
/// Defers to [`crate::tool::pairtype::pairmro`], which already ports
/// upstream's cross-MRO iterator. This wrapper binds it to
/// [`ReprClassId::mro`] so the `rtyper` pairtype dispatchers can read
/// a pre-materialised `Vec<(ReprClassId, ReprClassId)>` without
/// threading lifetime-bound slices through match arms.
pub fn pair_mro(c1: ReprClassId, c2: ReprClassId) -> Vec<(ReprClassId, ReprClassId)> {
    crate::tool::pairtype::pairmro(c1.mro(), c2.mro()).collect()
}

/// RPython `pair(a, b).convert_from_to(v, llops)` (`pairtype.py`
/// + per-module `class __extend__(pairtype(R_A, R_B))` blocks).
///
/// Returns:
/// - `Ok(Some(v))` — conversion emitted one or more llops and produced
///   the converted value.
/// - `Ok(None)` — upstream `return NotImplemented` (pairtype.py walks
///   further up the MRO; exhausted MRO returns None).
/// - `Err(TyperError)` — pair handler raised.
///
/// The Rust adaptation bundles both "handler doesn't exist" and
/// "handler returned NotImplemented" into `Ok(None)`; the `convertvar`
/// caller surfaces the final TyperError.
pub fn pair_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    let c1 = r_from.repr_class_id();
    let c2 = r_to.repr_class_id();
    for (b1, b2) in pair_mro(c1, c2) {
        if let Some(result) = dispatch_convert_from_to(b1, b2, r_from, r_to, v, llops)? {
            return Ok(Some(result));
        }
    }
    Ok(None)
}

/// Single-step `(b1, b2)` dispatcher. Each arm corresponds to one
/// `class __extend__(pairtype(R_A, R_B)): def convert_from_to(...)`
/// block in upstream.
///
/// Returns `Ok(None)` both for "no registration at (b1, b2)" and for
/// "registration returned NotImplemented" — the caller keeps walking
/// pair_mro until a `Some` surfaces.
fn dispatch_convert_from_to(
    b1: ReprClassId,
    b2: ReprClassId,
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    use ReprClassId::*;
    match (b1, b2) {
        // rptr.py __extend__ — same pointer low-level type is identity.
        (PtrRepr, PtrRepr) => same_lowleveltype_convert_from_to(r_from, r_to, v),
        // rptr.py __extend__ and rptr.py __extend__ — matching pointer /
        // interior-pointer ADT-method low-level surfaces are identity
        // conversions.
        (PtrRepr, LLADTMethRepr) | (InteriorPtrRepr, LLADTMethRepr) => {
            same_lowleveltype_convert_from_to(r_from, r_to, v)
        }
        // rptr.py — InteriorPtrRepr -> InteriorPtrRepr is stricter
        // than lowleveltype equality: upstream checks `__dict__` equality.
        (InteriorPtrRepr, InteriorPtrRepr) => {
            same_interior_ptr_dict_convert_from_to(r_from, r_to, v)
        }
        // rpbc.py — pairtype(ClassesPBCRepr, ClassRepr). RootClassRepr
        // subclasses ClassRepr upstream; both pyre types report
        // ReprClassId::Repr, so the concrete type selects the handler.
        (ClassesPBCRepr, Repr)
            if r_to.type_id() == std::any::TypeId::of::<super::rclass::ClassRepr>()
                || r_to.type_id() == std::any::TypeId::of::<super::rclass::RootClassRepr>() =>
        {
            super::rpbc::pair_classes_pbc_class_convert_from_to(r_from, r_to, v, llops)
        }
        // rclass.py — pairtype(InstanceRepr,
        // InstanceRepr).convert_from_to: cast_pointer along the
        // subclass/superclass axis when one classdef is a base of the
        // other.
        (InstanceRepr, InstanceRepr) => {
            super::rclass::pair_instance_instance_convert_from_to(r_from, r_to, v, llops)
        }
        // rgcref.py — `pairtype(GCRefRepr, Repr).convert_from_to` for a
        // Gc-opaque pointer the annotator merged into the classdef-less
        // top instance.
        (PtrRepr, InstanceRepr) => {
            super::rclass::pair_opaque_ptr_instance_convert_from_to(r_from, r_to, v, llops)
        }
        // rbool.py __extend__ — bool participates in IntegerRepr's MRO but
        // carries explicit primitive casts for the common Bool edges.
        (BoolRepr, FloatRepr) => {
            super::rbool::pair_bool_float_convert_from_to(r_from, r_to, v, llops)
        }
        (FloatRepr, BoolRepr) => {
            super::rbool::pair_float_bool_convert_from_to(r_from, r_to, v, llops)
        }
        (BoolRepr, IntegerRepr) => {
            super::rbool::pair_bool_integer_convert_from_to(r_from, r_to, v, llops)
        }
        (IntegerRepr, BoolRepr) => {
            super::rbool::pair_integer_bool_convert_from_to(r_from, r_to, v, llops)
        }
        // rint.py convert_from_to — primitive numeric casts.
        (IntegerRepr, IntegerRepr) => {
            super::rint::pair_integer_integer_convert_from_to(r_from, r_to, v, llops)
        }
        (IntegerRepr, FloatRepr) => {
            super::rint::pair_integer_float_convert_from_to(r_from, r_to, v, llops)
        }
        (FloatRepr, IntegerRepr) => {
            super::rint::pair_float_integer_convert_from_to(r_from, r_to, v, llops)
        }
        // rbuiltin.py — pairtype(BuiltinMethodRepr,
        // BuiltinMethodRepr).convert_from_to: only converts between
        // same-methodname reprs, delegating the receiver lowering via
        // llops.convertvar(v, r_from.self_repr, r_to.self_repr).
        (BuiltinMethodRepr, BuiltinMethodRepr) => {
            super::rbuiltin::pair_builtin_method_convert_from_to(r_from, r_to, v, llops)
        }
        // rpbc.py — pairtype(FunctionRepr,
        // FunctionRepr).convert_from_to: upstream `return v`. Both
        // ends are Void-typed so the Variable can be passed through
        // unmodified — no need to emit any operation.
        (FunctionRepr, FunctionRepr) => Ok(Some(v.clone())),
        // rpbc.py — pairtype(FunctionsPBCRepr,
        // FunctionRepr).convert_from_to: upstream
        // `return inputconst(Void, None)`. FunctionRepr is Void-typed,
        // so the conversion collapses to a Void None constant
        // regardless of the FunctionsPBCRepr source variable.
        (FunctionsPBCRepr, FunctionRepr) => Ok(Some(Hlvalue::Constant(inputconst_from_lltype(
            &LowLevelType::Void,
            &ConstValue::None,
        )?))),
        // rpbc.py — pairtype(FunctionRepr,
        // FunctionsPBCRepr).convert_from_to: upstream
        // `return inputconst(r_fpbc2, r_fpbc1.s_pbc.const)`.
        //   FunctionRepr always carries a single-FunctionDesc PBC whose
        //   `s_pbc.const` is the host-side function pyobj. Re-input the
        //   constant against r_fpbc2's lowleveltype, which routes
        //   through `FunctionsPBCRepr.convert_const(pyobj)`.
        (FunctionRepr, FunctionsPBCRepr) => {
            super::rpbc::pair_function_repr_functions_pbc_convert_from_to(r_from, r_to, v, llops)
        }
        // rpbc.py — pairtype(MultipleFrozenPBCRepr,
        // FunctionRepr).convert_from_to: a frozen-PBC pointer feeding a
        // constant-function repr drops its runtime value and materialises
        // the target's own `s_pbc.const` at Void.
        //
        // The other five frozen-PBC pairs upstream registers
        // (`MultipleFrozenPBCRepr` × `MultipleUnrelatedFrozenPBCRepr` /
        // `MultipleFrozenPBCRepr`, `SingleFrozenPBCRepr` ×
        // `MultipleFrozenPBCRepr`, `MultipleFrozenPBCReprBase` ×
        // `SingleFrozenPBCRepr`, `FunctionRepr` ×
        // `MultipleFrozenPBCRepr`) are still unported. The
        // `MultipleFrozenPBCReprBase` one additionally needs that base in
        // `pair_mro`, which currently sends both `Multiple*` reprs
        // straight to `Repr`.
        (MultipleFrozenPBCRepr, FunctionRepr) => {
            super::rpbc::pair_multiple_frozen_pbc_function_repr_convert_from_to(r_to)
        }
        // rpbc.py — pairtype(FunctionsPBCRepr,
        // FunctionsPBCRepr).convert_from_to: upstream identity if
        // `r_fpbc1.lowleveltype == r_fpbc2.lowleveltype`, else
        // `NotImplemented`. The rtyper distinguishes different spec-func
        // Struct pointer types even within the same Repr subclass.
        (FunctionsPBCRepr, FunctionsPBCRepr) => same_lowleveltype_convert_from_to(r_from, r_to, v),
        // rpbc.py — pairtype(MethodsPBCRepr,
        // MethodsPBCRepr).convert_from_to: upstream identity if
        // `r_mpbc1.lowleveltype == r_mpbc2.lowleveltype`, else
        // `NotImplemented`. Two structurally identical method families
        // reach this with the same receiver Ptr lowleveltype (the
        // instance the bound method carries), so the pass-through applies.
        (MethodsPBCRepr, MethodsPBCRepr) => same_lowleveltype_convert_from_to(r_from, r_to, v),
        // rpbc.py — pairtype(SmallFunctionSetPBCRepr,
        // FunctionRepr).convert_from_to: `return inputconst(Void, None)`.
        //   FunctionRepr is Void-typed; the Char source variable is
        //   discarded.
        (SmallFunctionSetPBCRepr, FunctionRepr) => Ok(Some(Hlvalue::Constant(
            inputconst_from_lltype(&LowLevelType::Void, &ConstValue::None)?,
        ))),
        // rpbc.py — pairtype(SmallFunctionSetPBCRepr,
        // FunctionsPBCRepr).convert_from_to:
        //   `assert v.concretetype is Char;
        //    v_int = llops.genop('cast_char_to_int', [v], resulttype=Signed);
        //    return llops.genop('getarrayitem',
        //                        [r_set.c_pointer_table, v_int],
        //                        resulttype=r_ptr.lowleveltype)`.
        (SmallFunctionSetPBCRepr, FunctionsPBCRepr) => {
            super::rpbc::pair_small_function_set_functions_pbc_convert_from_to(
                r_from, r_to, v, llops,
            )
        }
        // rpbc.py — pairtype(FunctionRepr,
        // SmallFunctionSetPBCRepr).convert_from_to: pulls the unique
        // FunctionDesc out of `r_ptr.s_pbc.descriptions` and returns
        // `inputconst(Char, r_set.convert_desc(desc))`.
        (FunctionRepr, SmallFunctionSetPBCRepr) => {
            super::rpbc::pair_function_repr_small_function_set_convert_from_to(
                r_from, r_to, v, llops,
            )
        }
        // rpbc.py — pairtype(FunctionsPBCRepr,
        // SmallFunctionSetPBCRepr).convert_from_to:
        //   `ll_compress = compression_function(r_set);
        //    return llops.gendirectcall(ll_compress, v)`.
        (FunctionsPBCRepr, SmallFunctionSetPBCRepr) => {
            super::rpbc::pair_functions_pbc_small_function_set_convert_from_to(
                r_from, r_to, v, llops,
            )
        }
        // rpbc.py — pairtype(SmallFunctionSetPBCRepr,
        // SmallFunctionSetPBCRepr).convert_from_to:
        //   `c_table = conversion_table(r_from, r_to);
        //    if c_table:
        //        v_int = genop('cast_char_to_int', [v], Signed)
        //        return genop('getarrayitem', [c_table, v_int], Char)
        //    else:
        //        return v  # identity`.
        (SmallFunctionSetPBCRepr, SmallFunctionSetPBCRepr) => {
            super::rpbc::pair_small_function_set_small_function_set_convert_from_to(
                r_from, r_to, v, llops,
            )
        }
        // rtuple.py — `pairtype(TupleRepr, TupleRepr).convert_from_to`.
        // Same-arity tuple-to-tuple: identity if lltypes match, else
        // per-item getitem_internal + convertvar + newtuple.
        // Different-arity returns NotImplemented.
        (TupleRepr, TupleRepr) => {
            super::rtuple::pair_tuple_tuple_convert_from_to(r_from, r_to, v, llops)
        }
        // rgcref.py ll_hash_func — conversions to/from the generic GCREF
        // wrapper use `cast_opaque_ptr`, optionally converting through
        // the wrapped base repr first.
        (GCRefRepr, Repr) => {
            super::lltypesystem::rgcref::pair_gcref_repr_convert_from_to(r_from, r_to, v, llops)
        }
        (Repr, GCRefRepr) => {
            super::lltypesystem::rgcref::pair_repr_gcref_convert_from_to(r_from, r_to, v, llops)
        }
        // rstr.py:805-814 — Char→String and UniChar→Unicode conversion
        // routes through `ll_chr2str`, allocating a one-character
        // STR/UNICODE before downstream string operations reuse the
        // AbstractStringRepr machinery.
        (CharRepr, StringRepr) | (UniCharRepr, UnicodeRepr) => {
            super::rstr::pair_charlike_string_convert_from_to(r_from, r_to, v, llops)
        }
        // rstr.py:815-820 — String→Char conversion extracts
        // `ll_stritem_nonneg(s, 0)`.
        (StringRepr, CharRepr) => {
            super::rstr::pair_string_char_convert_from_to(r_from, r_to, v, llops)
        }
        // rmodel.py — `pairtype(Repr, VoidRepr).convert_from_to`
        // returns `inputconst(Void, None)`.
        (Repr, VoidRepr) => Ok(Some(Hlvalue::Constant(inputconst_from_lltype(
            &LowLevelType::Void,
            &ConstValue::None,
        )?))),
        // rnone.py — `pairtype(Repr, NoneRepr).convert_from_to`
        // returns `inputconst(Void, None)`.
        (Repr, NoneRepr) => super::rnone::pair_any_none_convert_from_to(r_from, r_to, v, llops),
        // rnone.py — `pairtype(NoneRepr, Repr).convert_from_to`
        // returns `inputconst(r_to, None)`.
        (NoneRepr, Repr) => super::rnone::pair_none_any_convert_from_to(r_from, r_to, v, llops),
        // raddress.py — `pairtype(PtrRepr, AddressRepr).convert_from_to`.
        (PtrRepr, AddressRepr) => {
            let result = llops.genop(
                "cast_ptr_to_adr",
                vec![v.clone()],
                GenopResult::LLType(LowLevelType::Address),
            );
            Ok(result.map(Hlvalue::Variable))
        }
        // A Rust `Vec` header is a raw pointer; `r_uint(vec)` is
        // `cast_ptr_to_int` of that address (`vec_header_i` / `vec_header_r`).
        (RustVecRepr, IntegerRepr) => {
            super::rrustvec::pair_rustvec_integer_convert_from_to(r_from, r_to, v, llops)
        }
        _ => Ok(None),
    }
}

fn same_lowleveltype_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
) -> Result<Option<Hlvalue>, TyperError> {
    if r_from.lowleveltype() == r_to.lowleveltype() {
        return Ok(Some(v.clone()));
    }
    Ok(None)
}

fn same_interior_ptr_dict_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
) -> Result<Option<Hlvalue>, TyperError> {
    let from_key = r_from.interior_ptr_repr_dict_key();
    if from_key.is_some() && from_key == r_to.interior_ptr_repr_dict_key() {
        return Ok(Some(v.clone()));
    }
    Ok(None)
}

type PairRTypeDispatch = Result<Option<Option<Hlvalue>>, TyperError>;

fn committed(result: RTypeResult) -> PairRTypeDispatch {
    result.map(Some)
}

fn pair_rtype_op(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp, opname: &str) -> RTypeResult {
    let c1 = r1.repr_class_id();
    let c2 = r2.repr_class_id();
    for (b1, b2) in pair_mro(c1, c2) {
        if let Some(result) = dispatch_rtype_op(b1, b2, r1, r2, hop, opname)? {
            return Ok(result);
        }
    }
    Err(TyperError::missing_rtype_operation(format!(
        "pair(rtype_{opname}) not implemented for ({:?}, {:?})",
        c1, c2
    )))
}

fn dispatch_rtype_op(
    b1: ReprClassId,
    b2: ReprClassId,
    r1: &dyn Repr,
    r2: &dyn Repr,
    hop: &HighLevelOp,
    opname: &str,
) -> PairRTypeDispatch {
    use ReprClassId::*;
    match (b1, b2, opname) {
        // rptr.py __extend__ / 301-329 — array and interior-array indexing.
        (PtrRepr, IntegerRepr, "getitem") | (InteriorPtrRepr, IntegerRepr, "getitem") => {
            committed(r1.rtype_getitem(hop))
        }
        // rlist.py — pair(AbstractBaseListRepr, IntegerRepr).rtype_getitem.
        // FixedSizeListRepr (Rust slice / fixed array) handles the nonneg +
        // checkidx=False branch (bare getarrayitem); the negative-index and
        // checkidx branches surface a TyperError until those helpers land.
        (FixedSizeListRepr, IntegerRepr, "getitem") => committed(r1.rtype_getitem(hop)),
        // rlist.py — the resized ListRepr shares the dispatch but
        // reads through the `items` array out of its `length`/`items`
        // header struct (getfield "items" → getarrayitem).
        (ListRepr, IntegerRepr, "getitem") => committed(r1.rtype_getitem(hop)),
        // `ll_vec_getitem_fast` on a Rust `Vec` header.
        (RustVecRepr, IntegerRepr, "getitem") => committed(r1.rtype_getitem(hop)),
        // rordereddict.py — `pairtype(OrderedDictRepr, rmodel.Repr).rtype_getitem`.
        // The second receiver is never read in the upstream body (only
        // `r_dict.key_repr` is), so this wildcards `_` on the key repr class,
        // same pattern as `(TupleRepr, _, "contains")` below.
        (OrderedDictRepr, _, "getitem") => committed(r1.rtype_getitem(hop)),
        // rordereddict.py — `pairtype(OrderedDictRepr, rmodel.Repr).rtype_delitem`.
        // Same wildcard-`_` dispatch rationale as the `"getitem"` arm above
        // (`r_key` is never read, only `r_dict.key_repr`).
        (OrderedDictRepr, _, "delitem") => committed(r1.rtype_delitem(hop)),
        // rrange.py — pair(AbstractRangeRepr, IntegerRepr).rtype_getitem.
        // RangeRepr handles the constant-step + nonneg + dum_nocheck branch
        // (start + index*step); the checkidx, negative-index, and
        // variable-step branches surface a TyperError until those land.
        (RangeRepr, IntegerRepr, "getitem") => committed(r1.rtype_getitem(hop)),
        // rtuple.py — `pairtype(TupleRepr, IntegerRepr).rtype_getitem`.
        (TupleRepr, IntegerRepr, "getitem") => {
            committed(super::rtuple::pair_tuple_int_rtype_getitem(r1, hop))
        }
        // rstr.py — `pair(AbstractStringRepr, IntegerRepr).rtype_getitem`.
        // Pyre handles the nonneg + checkidx=False branch; other
        // branches (negative-index ll_stritem, checkidx
        // ll_stritem_*_checked) surface a TyperError until those
        // helpers land.
        (StringRepr, IntegerRepr, "getitem") => {
            committed(super::rstr::pair_string_int_rtype_getitem(hop))
        }
        (UnicodeRepr, IntegerRepr, "getitem") => {
            committed(super::rstr::pair_unicode_int_rtype_getitem(hop))
        }
        // rstr.py — `rtype_getitem_idx` defers to
        // `rtype_getitem(hop, checkidx=True)`. The checked path raises
        // IndexError from inside ll_stritem_checked /
        // ll_stritem_nonneg_checked.
        (StringRepr, IntegerRepr, "getitem_idx") => {
            committed(super::rstr::pair_string_int_rtype_getitem_idx(hop))
        }
        (UnicodeRepr, IntegerRepr, "getitem_idx") => {
            committed(super::rstr::pair_unicode_int_rtype_getitem_idx(hop))
        }
        // rstr.py — `pairtype(AbstractCharRepr, IntegerRepr)`
        // and `pairtype(AbstractUniCharRepr, IntegerRepr)` rtype_getitem
        // share the same body: constant-0 pass-through, otherwise
        // Char→Str/UniChar→Unicode via `ll_chr2str` and reuse
        // AbstractStringRepr+IntegerRepr.
        (CharRepr, IntegerRepr, "getitem") => {
            committed(super::rstr::pair_char_int_rtype_getitem(hop))
        }
        (UniCharRepr, IntegerRepr, "getitem") => {
            committed(super::rstr::pair_unichar_int_rtype_getitem(hop))
        }
        (CharRepr, IntegerRepr, "getitem_idx") => {
            committed(super::rstr::pair_char_int_rtype_getitem_idx(hop))
        }
        (UniCharRepr, IntegerRepr, "getitem_idx") => {
            committed(super::rstr::pair_unichar_int_rtype_getitem_idx(hop))
        }
        // rstr.py — `pair(AbstractStringRepr, AbstractStringRepr).rtype_add`
        // (and `rtype_inplace_add`) lower to
        // `direct_call(ll_strconcat, v_s1, v_s2)` against the
        // helper graph synthesised by
        // `build_ll_strconcat_helper_graph` (malloc_varsize +
        // chars-array copy loop).
        (StringRepr, StringRepr, "add") | (StringRepr, StringRepr, "inplace_add") => {
            committed(super::rstr::pair_string_string_rtype_add(hop))
        }
        (UnicodeRepr, UnicodeRepr, "add") | (UnicodeRepr, UnicodeRepr, "inplace_add") => {
            committed(super::rstr::pair_unicode_unicode_rtype_add(hop))
        }
        (AbstractStringRepr, AbstractStringRepr, "add")
        | (AbstractStringRepr, AbstractStringRepr, "inplace_add") => {
            committed(super::rstr::pair_abstract_string_rtype_add(r1, r2, hop))
        }
        // `rtype_add` is defined at exactly one place, `pairtype(
        // AbstractStringRepr, AbstractStringRepr)` (rstr.py); no
        // char pairtype defines one. A char operand reaches that body
        // through `CharRepr(AbstractCharRepr, StringRepr)` and
        // `UniCharRepr(AbstractUniCharRepr, UnicodeRepr)`
        // (lltypesystem/rstr.py), which is also where `char_repr.repr`
        // comes from: `StringRepr.repr = string_repr` and
        // `UniCharRepr.repr = unicode_repr` (lltypesystem/rstr.py), so
        // `rtype_add` coerces both operands to the
        // string surface and calls the same `ll_strconcat`. The two
        // functions below already coerce that way, and
        // `pair_convert_from_to` already carries the Char→Str /
        // UniChar→Unicode `ll_chr2str` edge, so the mixed and char-only
        // pairs need dispatch entries and no lowering of their own.
        // Pyre's CharRepr pair_mro is [CharRepr, Repr] (explicit-arm
        // design, as for the comparisons below), so they are spelled
        // out rather than inherited.
        (StringRepr, CharRepr, "add" | "inplace_add")
        | (CharRepr, StringRepr, "add" | "inplace_add")
        | (CharRepr, CharRepr, "add" | "inplace_add") => {
            committed(super::rstr::pair_string_string_rtype_add(hop))
        }
        (UnicodeRepr, UniCharRepr, "add" | "inplace_add")
        | (UniCharRepr, UnicodeRepr, "add" | "inplace_add")
        | (UniCharRepr, UniCharRepr, "add" | "inplace_add") => {
            committed(super::rstr::pair_unicode_unicode_rtype_add(hop))
        }
        // rtuple.py — `pairtype(TupleRepr, TupleRepr).rtype_add`
        // (and its `rtype_inplace_add` alias) concatenates two tuples
        // by per-position getfield + newtuple_cached.
        (TupleRepr, TupleRepr, "add") | (TupleRepr, TupleRepr, "inplace_add") => {
            committed(super::rtuple::pair_tuple_tuple_rtype_add(r1, r2, hop))
        }
        // rtuple.py — `pairtype(TupleRepr, TupleRepr).rtype_eq`
        // dispatches to the per-shape `ll_eq` helper synthesised by
        // `gen_eq_function`.
        (TupleRepr, TupleRepr, "eq") => {
            committed(super::rtuple::pair_tuple_tuple_rtype_eq(r1, r2, hop))
        }
        // rtuple.py — `rtype_ne = rtype_eq + bool_not`.
        (TupleRepr, TupleRepr, "ne") => {
            committed(super::rtuple::pair_tuple_tuple_rtype_ne(r1, r2, hop))
        }
        // rtuple.py — `pairtype(TupleRepr, Repr).rtype_contains`
        // dispatches to a constant-tuple membership test using the
        // synthesised per-type `ll_equal` helper.
        (TupleRepr, _, "contains") => {
            committed(super::rtuple::pair_tuple_repr_rtype_contains(r1, r2, hop))
        }
        // rordereddict.py — `pairtype(OrderedDictRepr, rmodel.Repr).rtype_contains`.
        (OrderedDictRepr, _, "contains") => committed(
            super::lltypesystem::rordereddict::pair_ordereddict_repr_rtype_contains(r1, r2, hop),
        ),
        (PtrRepr, IntegerRepr, "setitem") | (InteriorPtrRepr, IntegerRepr, "setitem") => {
            committed(r1.rtype_setitem(hop))
        }
        // rlist.py — pair(AbstractBaseListRepr, IntegerRepr).rtype_setitem.
        // FixedSizeListRepr handles the nonneg + dum_nocheck branch (bare
        // setarrayitem); the checkidx (IndexError) and negative-index
        // branches surface a TyperError until those helpers land.
        (FixedSizeListRepr, IntegerRepr, "setitem") => committed(r1.rtype_setitem(hop)),
        // rlist.py — the resized ListRepr shares the dispatch but
        // reads through the `items` array out of its `length`/`items`
        // header struct (getfield "items" → setarrayitem).
        (ListRepr, IntegerRepr, "setitem") => committed(r1.rtype_setitem(hop)),
        // `ll_vec_setitem_fast` on a Rust `Vec` header.
        (RustVecRepr, IntegerRepr, "setitem") => committed(r1.rtype_setitem(hop)),
        // rordereddict.py — `pairtype(OrderedDictRepr, rmodel.Repr).rtype_setitem`.
        // Same wildcard-`_` dispatch rationale as the `"getitem"` arm above
        // (`r_key` is never read, only `r_dict.key_repr`).
        (OrderedDictRepr, _, "setitem") => committed(r1.rtype_setitem(hop)),

        // rptr.py __extend__ — pointer comparison accepts any repr on the
        // other side and coerces both args to the pointer repr.
        (PtrRepr, Repr, "eq") => committed(r1.rtype_eq(hop)),
        (PtrRepr, Repr, "ne") => committed(r1.rtype_ne(hop)),
        (Repr, PtrRepr, "eq") => committed(r2.rtype_eq(hop)),
        (Repr, PtrRepr, "ne") => committed(r2.rtype_ne(hop)),

        // rclass.py — `pairtype(InstanceRepr, InstanceRepr).rtype_eq
        // = rtype_is_`: both sides upcast to the common-base instance
        // repr, then pointer identity (`ptr_eq`).  rtype_ne (1072-1074)
        // negates it.
        (InstanceRepr, InstanceRepr, "eq") => {
            committed(super::rclass::pair_instance_instance_rtype_is_(r1, r2, hop))
        }
        (InstanceRepr, InstanceRepr, "ne") => {
            committed(super::rclass::pair_instance_instance_rtype_ne(r1, r2, hop))
        }

        // rint.py — IntegerRepr/IntegerRepr pair arithmetic and
        // comparisons. `truediv` is intentionally absent here: upstream
        // delegates it to FloatRepr through the MRO.
        (IntegerRepr, IntegerRepr, "add") => committed(super::rint::rtype_template(hop, "add")),
        (IntegerRepr, IntegerRepr, "add_ovf") => committed(super::rint::rtype_add_ovf(hop)),
        (IntegerRepr, IntegerRepr, "sub") => committed(super::rint::rtype_template(hop, "sub")),
        (IntegerRepr, IntegerRepr, "sub_ovf") => {
            committed(super::rint::rtype_template(hop, "sub_ovf"))
        }
        (IntegerRepr, IntegerRepr, "mul") => committed(super::rint::rtype_template(hop, "mul")),
        (IntegerRepr, IntegerRepr, "mul_ovf") => {
            committed(super::rint::rtype_template(hop, "mul_ovf"))
        }
        (IntegerRepr, IntegerRepr, "floordiv") | (IntegerRepr, IntegerRepr, "div") => committed(
            super::rint::rtype_call_helper(hop, "py_div".to_string(), &["ZeroDivisionError"]),
        ),
        (IntegerRepr, IntegerRepr, "floordiv_ovf") | (IntegerRepr, IntegerRepr, "div_ovf") => {
            committed(super::rint::rtype_call_helper(
                hop,
                "py_div_ovf".to_string(),
                &["ZeroDivisionError"],
            ))
        }
        (IntegerRepr, IntegerRepr, "mod") => committed(super::rint::rtype_call_helper(
            hop,
            "py_mod".to_string(),
            &["ZeroDivisionError"],
        )),
        (IntegerRepr, IntegerRepr, "mod_ovf") => committed(super::rint::rtype_call_helper(
            hop,
            "py_mod_ovf".to_string(),
            &["ZeroDivisionError"],
        )),
        (IntegerRepr, IntegerRepr, "xor") => committed(super::rint::rtype_template(hop, "xor")),
        (IntegerRepr, IntegerRepr, "and") => committed(super::rint::rtype_template(hop, "and")),
        (IntegerRepr, IntegerRepr, "or") => committed(super::rint::rtype_template(hop, "or")),
        (IntegerRepr, IntegerRepr, "lshift") => {
            committed(super::rint::rtype_template(hop, "lshift"))
        }
        (IntegerRepr, IntegerRepr, "lshift_ovf") => committed(super::rint::rtype_call_helper(
            hop,
            "lshift_ovf".to_string(),
            &[],
        )),
        (IntegerRepr, IntegerRepr, "rshift") => {
            committed(super::rint::rtype_template(hop, "rshift"))
        }
        (IntegerRepr, IntegerRepr, "eq") => {
            committed(super::rint::rtype_compare_template(hop, "eq"))
        }
        (IntegerRepr, IntegerRepr, "ne") => {
            committed(super::rint::rtype_compare_template(hop, "ne"))
        }
        (IntegerRepr, IntegerRepr, "lt") => {
            committed(super::rint::rtype_compare_template(hop, "lt"))
        }
        (IntegerRepr, IntegerRepr, "le") => {
            committed(super::rint::rtype_compare_template(hop, "le"))
        }
        (IntegerRepr, IntegerRepr, "gt") => {
            committed(super::rint::rtype_compare_template(hop, "gt"))
        }
        (IntegerRepr, IntegerRepr, "ge") => {
            committed(super::rint::rtype_compare_template(hop, "ge"))
        }
        (IntegerRepr, IntegerRepr, "is_") => {
            committed(super::rint::rtype_compare_template(hop, "eq"))
        }

        // rfloat.py — FloatRepr/FloatRepr pair arithmetic and
        // comparisons. IntegerRepr inherits FloatRepr upstream, so mixed
        // int/float arithmetic reaches these arms via `pair_mro`.
        (FloatRepr, FloatRepr, "add") => committed(super::rfloat::rtype_template(hop, "add")),
        (FloatRepr, FloatRepr, "sub") => committed(super::rfloat::rtype_template(hop, "sub")),
        (FloatRepr, FloatRepr, "mul") => committed(super::rfloat::rtype_template(hop, "mul")),
        (FloatRepr, FloatRepr, "truediv") | (FloatRepr, FloatRepr, "div") => {
            committed(super::rfloat::rtype_template(hop, "truediv"))
        }
        (FloatRepr, FloatRepr, "eq") => committed(super::rfloat::rtype_compare_template(hop, "eq")),
        (FloatRepr, FloatRepr, "ne") => committed(super::rfloat::rtype_compare_template(hop, "ne")),
        (FloatRepr, FloatRepr, "lt") => committed(super::rfloat::rtype_compare_template(hop, "lt")),
        (FloatRepr, FloatRepr, "le") => committed(super::rfloat::rtype_compare_template(hop, "le")),
        (FloatRepr, FloatRepr, "gt") => committed(super::rfloat::rtype_compare_template(hop, "gt")),
        (FloatRepr, FloatRepr, "ge") => committed(super::rfloat::rtype_compare_template(hop, "ge")),
        (FloatRepr, FloatRepr, "is_") => {
            committed(super::rfloat::rtype_compare_template(hop, "eq"))
        }

        // rstr.py — pairtype(AbstractCharRepr, AbstractCharRepr)
        // dispatches all six compare ops to `char_<func>` lloperations.
        (CharRepr, CharRepr, "eq") => {
            committed(super::rstr::pair_char_char_rtype_compare(hop, "eq"))
        }
        (CharRepr, CharRepr, "ne") => {
            committed(super::rstr::pair_char_char_rtype_compare(hop, "ne"))
        }
        (CharRepr, CharRepr, "lt") => {
            committed(super::rstr::pair_char_char_rtype_compare(hop, "lt"))
        }
        (CharRepr, CharRepr, "le") => {
            committed(super::rstr::pair_char_char_rtype_compare(hop, "le"))
        }
        (CharRepr, CharRepr, "gt") => {
            committed(super::rstr::pair_char_char_rtype_compare(hop, "gt"))
        }
        (CharRepr, CharRepr, "ge") => {
            committed(super::rstr::pair_char_char_rtype_compare(hop, "ge"))
        }

        // rstr.py — pairtype(AbstractUniCharRepr, AbstractUniCharRepr).
        // eq/ne dispatch to `unichar_<func>`; lt/le/gt/ge cast both
        // arms via `cast_unichar_to_int` and dispatch to `int_<func>`.
        (UniCharRepr, UniCharRepr, "eq") => committed(
            super::rstr::pair_unichar_unichar_rtype_compare_eqne(hop, "eq"),
        ),
        (UniCharRepr, UniCharRepr, "ne") => committed(
            super::rstr::pair_unichar_unichar_rtype_compare_eqne(hop, "ne"),
        ),
        (UniCharRepr, UniCharRepr, "lt") => committed(
            super::rstr::pair_unichar_unichar_rtype_compare_ord(hop, "lt"),
        ),
        (UniCharRepr, UniCharRepr, "le") => committed(
            super::rstr::pair_unichar_unichar_rtype_compare_ord(hop, "le"),
        ),
        (UniCharRepr, UniCharRepr, "gt") => committed(
            super::rstr::pair_unichar_unichar_rtype_compare_ord(hop, "gt"),
        ),
        (UniCharRepr, UniCharRepr, "ge") => committed(
            super::rstr::pair_unichar_unichar_rtype_compare_ord(hop, "ge"),
        ),

        // rstr.py — pairtype(AbstractStringRepr, AbstractStringRepr).
        // eq routes through ll_streq, ne wraps ll_streq with bool_not,
        // lt/le/gt/ge route through ll_strcmp + int_<func>(diff, 0).
        (StringRepr, StringRepr, "eq") => {
            committed(super::rstr::pair_string_string_rtype_compare(hop, "eq"))
        }
        (StringRepr, StringRepr, "ne") => {
            committed(super::rstr::pair_string_string_rtype_compare(hop, "ne"))
        }
        (StringRepr, StringRepr, "lt") => {
            committed(super::rstr::pair_string_string_rtype_compare(hop, "lt"))
        }
        (StringRepr, StringRepr, "le") => {
            committed(super::rstr::pair_string_string_rtype_compare(hop, "le"))
        }
        (StringRepr, StringRepr, "gt") => {
            committed(super::rstr::pair_string_string_rtype_compare(hop, "gt"))
        }
        (StringRepr, StringRepr, "ge") => {
            committed(super::rstr::pair_string_string_rtype_compare(hop, "ge"))
        }

        // rstr.py:661-692 via `CharRepr(AbstractCharRepr, StringRepr)` MRO —
        // a String compared with a Char coerces the Char operand to a 1-char
        // string (ll_chr2str) then reuses the AbstractStringRepr compare body.
        // Pyre's CharRepr pair_mro is [CharRepr, Repr] (explicit-arm design),
        // so the mixed pair needs its own arm; `str_arg_idx` names the string
        // operand both sides coerce to.
        (StringRepr, CharRepr, "eq" | "ne" | "lt" | "le" | "gt" | "ge") => {
            committed(super::rstr::pair_string_char_rtype_compare(hop, opname, 0))
        }
        (CharRepr, StringRepr, "eq" | "ne" | "lt" | "le" | "gt" | "ge") => {
            committed(super::rstr::pair_string_char_rtype_compare(hop, opname, 1))
        }

        // rstr.py inherited via AbstractUnicodeRepr(AbstractStringRepr) —
        // pairtype(AbstractUnicodeRepr, AbstractUnicodeRepr) routes through
        // the same body modulo the helper-graph identity (ll_unicode_eq /
        // ll_unicode_cmp).
        (UnicodeRepr, UnicodeRepr, "eq") => {
            committed(super::rstr::pair_unicode_unicode_rtype_compare(hop, "eq"))
        }
        (UnicodeRepr, UnicodeRepr, "ne") => {
            committed(super::rstr::pair_unicode_unicode_rtype_compare(hop, "ne"))
        }
        (UnicodeRepr, UnicodeRepr, "lt") => {
            committed(super::rstr::pair_unicode_unicode_rtype_compare(hop, "lt"))
        }
        (UnicodeRepr, UnicodeRepr, "le") => {
            committed(super::rstr::pair_unicode_unicode_rtype_compare(hop, "le"))
        }
        (UnicodeRepr, UnicodeRepr, "gt") => {
            committed(super::rstr::pair_unicode_unicode_rtype_compare(hop, "gt"))
        }
        (UnicodeRepr, UnicodeRepr, "ge") => {
            committed(super::rstr::pair_unicode_unicode_rtype_compare(hop, "ge"))
        }

        // raddress.py — `pairtype(TypedAddressAccessRepr, IntegerRepr).rtype_getitem`.
        (TypedAddressAccessRepr, IntegerRepr, "getitem") => {
            committed(pair_typed_address_access_int_rtype_getitem(r1, hop))
        }
        // raddress.py — `pairtype(TypedAddressAccessRepr, IntegerRepr).rtype_setitem`.
        (TypedAddressAccessRepr, IntegerRepr, "setitem") => {
            committed(pair_typed_address_access_int_rtype_setitem(r1, hop))
        }

        // raddress.py — `pairtype(AddressRepr, IntegerRepr).rtype_add/inplace_add`.
        (AddressRepr, IntegerRepr, "add") | (AddressRepr, IntegerRepr, "inplace_add") => {
            if *r2.lowleveltype() == LowLevelType::Signed {
                let v_args = hop.inputargs(vec![
                    ConvertedTo::LowLevelType(&LowLevelType::Address),
                    ConvertedTo::LowLevelType(&LowLevelType::Signed),
                ])?;
                Ok(Some(hop.genop(
                    "adr_add",
                    v_args,
                    GenopResult::LLType(LowLevelType::Address),
                )))
            } else {
                Ok(None)
            }
        }
        // raddress.py — `pairtype(AddressRepr, IntegerRepr).rtype_sub/inplace_sub`.
        (AddressRepr, IntegerRepr, "sub") | (AddressRepr, IntegerRepr, "inplace_sub") => {
            if *r2.lowleveltype() == LowLevelType::Signed {
                let v_args = hop.inputargs(vec![
                    ConvertedTo::LowLevelType(&LowLevelType::Address),
                    ConvertedTo::LowLevelType(&LowLevelType::Signed),
                ])?;
                Ok(Some(hop.genop(
                    "adr_sub",
                    v_args,
                    GenopResult::LLType(LowLevelType::Address),
                )))
            } else {
                Ok(None)
            }
        }
        // raddress.py — `pairtype(AddressRepr, AddressRepr).rtype_sub`.
        (AddressRepr, AddressRepr, "sub") => {
            let v_args = hop.inputargs(vec![
                ConvertedTo::LowLevelType(&LowLevelType::Address),
                ConvertedTo::LowLevelType(&LowLevelType::Address),
            ])?;
            Ok(Some(hop.genop(
                "adr_delta",
                v_args,
                GenopResult::LLType(LowLevelType::Signed),
            )))
        }
        // raddress.py — `pairtype(AddressRepr, AddressRepr)` comparisons.
        (AddressRepr, AddressRepr, "eq")
        | (AddressRepr, AddressRepr, "ne")
        | (AddressRepr, AddressRepr, "lt")
        | (AddressRepr, AddressRepr, "le")
        | (AddressRepr, AddressRepr, "gt")
        | (AddressRepr, AddressRepr, "ge") => {
            let v_args = hop.inputargs(vec![
                ConvertedTo::LowLevelType(&LowLevelType::Address),
                ConvertedTo::LowLevelType(&LowLevelType::Address),
            ])?;
            let op = format!("adr_{opname}");
            Ok(Some(hop.genop(
                &op,
                v_args,
                GenopResult::LLType(LowLevelType::Bool),
            )))
        }

        _ => Ok(None),
    }
}

/// raddress.py rtype_getitem
fn pair_typed_address_access_int_rtype_getitem(r_acc: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    use super::rmodel::TypedAddressAccessRepr;
    use crate::translator::rtyper::lltypesystem::llmemory::sizeof;

    let any_r: &dyn std::any::Any = r_acc;
    let r_acc = any_r
        .downcast_ref::<TypedAddressAccessRepr>()
        .ok_or_else(|| TyperError::message("expected TypedAddressAccessRepr"))?;
    let args_r = hop.args_r.borrow();
    let r0 = args_r[0]
        .as_ref()
        .ok_or_else(|| TyperError::message("missing repr for arg 0"))?;
    let v_args = hop.inputargs(vec![
        ConvertedTo::Repr(r0.as_ref()),
        ConvertedTo::LowLevelType(&LowLevelType::Signed),
    ])?;
    let v_addr = v_args[0].clone();
    let v_offs = v_args[1].clone();
    let size = sizeof(&r_acc.access_type, None).map_err(TyperError::message)?;
    let c_size = HighLevelOp::inputconst(&LowLevelType::Signed, &size)?;
    let v_offs_mult = hop
        .genop(
            "int_mul",
            vec![v_offs, Hlvalue::Constant(c_size)],
            GenopResult::LLType(LowLevelType::Signed),
        )
        .ok_or_else(|| TyperError::message("int_mul genop returned None"))?;
    Ok(hop.genop(
        "raw_load",
        vec![v_addr, v_offs_mult],
        GenopResult::LLType(r_acc.access_type.clone()),
    ))
}

/// raddress.py rtype_setitem
fn pair_typed_address_access_int_rtype_setitem(r_acc: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    use super::rmodel::TypedAddressAccessRepr;
    use crate::translator::rtyper::lltypesystem::llmemory::sizeof;

    let any_r: &dyn std::any::Any = r_acc;
    let r_acc = any_r
        .downcast_ref::<TypedAddressAccessRepr>()
        .ok_or_else(|| TyperError::message("expected TypedAddressAccessRepr"))?;
    let args_r = hop.args_r.borrow();
    let r0 = args_r[0]
        .as_ref()
        .ok_or_else(|| TyperError::message("missing repr for arg 0"))?;
    let v_args = hop.inputargs(vec![
        ConvertedTo::Repr(r0.as_ref()),
        ConvertedTo::LowLevelType(&LowLevelType::Signed),
        ConvertedTo::LowLevelType(&r_acc.access_type),
    ])?;
    let v_addr = v_args[0].clone();
    let v_offs = v_args[1].clone();
    let v_value = v_args[2].clone();
    let size = sizeof(&r_acc.access_type, None).map_err(TyperError::message)?;
    let c_size = HighLevelOp::inputconst(&LowLevelType::Signed, &size)?;
    let v_offs_mult = hop
        .genop(
            "int_mul",
            vec![v_offs, Hlvalue::Constant(c_size)],
            GenopResult::LLType(LowLevelType::Signed),
        )
        .ok_or_else(|| TyperError::message("int_mul genop returned None"))?;
    Ok(hop.genop(
        "raw_store",
        vec![v_addr, v_offs_mult, v_value],
        GenopResult::Void,
    ))
}

pub fn pair_rtype_getitem(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "getitem")
}

pub fn pair_rtype_setitem(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "setitem")
}

pub fn pair_rtype_delitem(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "delitem")
}

pub fn pair_rtype_add(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "add")
}

pub fn pair_rtype_add_ovf(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "add_ovf")
}

pub fn pair_rtype_sub(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "sub")
}

pub fn pair_rtype_sub_ovf(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "sub_ovf")
}

pub fn pair_rtype_mul(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "mul")
}

pub fn pair_rtype_mul_ovf(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "mul_ovf")
}

pub fn pair_rtype_truediv(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "truediv")
}

pub fn pair_rtype_floordiv(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "floordiv")
}

pub fn pair_rtype_floordiv_ovf(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "floordiv_ovf")
}

pub fn pair_rtype_div(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "div")
}

pub fn pair_rtype_div_ovf(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "div_ovf")
}

pub fn pair_rtype_mod(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "mod")
}

pub fn pair_rtype_mod_ovf(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "mod_ovf")
}

pub fn pair_rtype_xor(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "xor")
}

pub fn pair_rtype_and_(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "and")
}

pub fn pair_rtype_or_(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "or")
}

pub fn pair_rtype_lshift(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "lshift")
}

pub fn pair_rtype_lshift_ovf(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "lshift_ovf")
}

pub fn pair_rtype_rshift(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "rshift")
}

pub fn pair_rtype_cmp(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "cmp")
}

pub fn pair_rtype_coerce(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "coerce")
}

pub fn pair_rtype_contains(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "contains")
}

pub fn pair_rtype_eq(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "eq")
}

pub fn pair_rtype_ne(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "ne")
}

pub fn pair_rtype_lt(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "lt")
}

pub fn pair_rtype_le(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "le")
}

pub fn pair_rtype_gt(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "gt")
}

pub fn pair_rtype_ge(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    pair_rtype_op(r1, r2, hop, "ge")
}

/// RPython `pair(r1, r2).rtype_is_(hop)` (pairtype.py + rnone.py,
/// 61-64 + rfloat.py `rtype_is_ = rtype_eq`).
///
/// Equivalent of `translate_op_is` consulting the pair table. `pos`
/// defaults to 0 upstream for the first-arg side and 1 for the
/// second-arg side; the caller picks based on which pair matched.
pub fn pair_rtype_is_(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    let c1 = r1.repr_class_id();
    let c2 = r2.repr_class_id();
    for (b1, b2) in pair_mro(c1, c2) {
        if let Some(result) = dispatch_rtype_is_(b1, b2, r1, r2, hop)? {
            return Ok(Some(result));
        }
        if let Some(result) = dispatch_rtype_op(b1, b2, r1, r2, hop, "is_")? {
            return Ok(result);
        }
    }
    Err(TyperError::missing_rtype_operation(format!(
        "pair(rtype_is_) not implemented for ({:?}, {:?})",
        c1, c2
    )))
}

fn dispatch_rtype_is_(
    b1: ReprClassId,
    b2: ReprClassId,
    r1: &dyn Repr,
    r2: &dyn Repr,
    hop: &HighLevelOp,
) -> Result<Option<Hlvalue>, TyperError> {
    use ReprClassId::*;
    match (b1, b2) {
        // rnone.py — `pairtype(Repr, NoneRepr).rtype_is_` calls
        // `rtype_is_None(robj1, rnone2, hop, pos=0)`.
        (Repr, NoneRepr) => super::rnone::pair_any_none_rtype_is_(r1, r2, hop),
        // rnone.py — `pairtype(NoneRepr, Repr).rtype_is_` calls
        // `rtype_is_None(robj2, rnone1, hop, pos=1)`.
        (NoneRepr, Repr) => super::rnone::pair_none_any_rtype_is_(r1, r2, hop),
        // rpbc.py __extend__ — the one upstream block registers
        // `rtype_is_` on `pairtype(MU, MU)`, `pairtype(MU, Single)`,
        // and `pairtype(Single, MU)`. All three emit adr_eq after
        // converting both operands to the MU repr; a Single operand is
        // a constant PBC and converts via `inputconst -> convert_const`.
        (MultipleUnrelatedFrozenPBCRepr, MultipleUnrelatedFrozenPBCRepr)
        | (MultipleUnrelatedFrozenPBCRepr, SingleFrozenPBCRepr)
        | (SingleFrozenPBCRepr, MultipleUnrelatedFrozenPBCRepr) => {
            super::rpbc::pair_mu_mu_rtype_is_(r1, r2, hop).map(Some)
        }
        // rpbc.py — `pairtype(FunctionReprBase,
        // FunctionReprBase).rtype_is_`: union the two SomePBCs,
        // resolve the merged repr, convert both sides to it, then
        // emit `char_eq` (Char-typed Small set) or `ptr_eq` (Ptr-typed
        // Functions / Function set). Pyre's FunctionReprBase analogues
        // are `FunctionRepr`, `FunctionsPBCRepr`, and
        // `SmallFunctionSetPBCRepr` — all three pairwise.
        (FunctionRepr, FunctionRepr)
        | (FunctionRepr, FunctionsPBCRepr)
        | (FunctionRepr, SmallFunctionSetPBCRepr)
        | (FunctionsPBCRepr, FunctionRepr)
        | (FunctionsPBCRepr, FunctionsPBCRepr)
        | (FunctionsPBCRepr, SmallFunctionSetPBCRepr)
        | (SmallFunctionSetPBCRepr, FunctionRepr)
        | (SmallFunctionSetPBCRepr, FunctionsPBCRepr)
        | (SmallFunctionSetPBCRepr, SmallFunctionSetPBCRepr) => {
            super::rpbc::pair_function_repr_base_rtype_is_(r1, r2, hop).map(Some)
        }
        // rtuple.py — `pairtype(TupleRepr, TupleRepr).rtype_is_`
        // raises `TyperError("cannot compare tuples with 'is'")`. The
        // identity check is structurally meaningless on a tuple value,
        // and the rtyper rejects it eagerly so callers cannot route a
        // tuple through ptr_eq via the generic `(Repr, Repr)` arm below.
        (TupleRepr, TupleRepr) => super::rtuple::pair_tuple_tuple_rtype_is_(r1, r2, hop),
        // rclass.py — `pairtype(InstanceRepr, InstanceRepr)
        // .rtype_is_`: convert both sides to the common-base instance
        // repr (cast_pointer upcast), then pointer identity via the
        // generic `(Repr, Repr)` arm.
        (InstanceRepr, InstanceRepr) => {
            super::rclass::pair_instance_instance_rtype_is_(r1, r2, hop)
        }
        // rmodel.py rtype_is_ — generic identity comparison for pointer
        // low-level values, with Void adopting the opposite repr.
        (Repr, Repr) => pair_repr_repr_rtype_is_(r1, r2, hop).map(Some),
        _ => Ok(None),
    }
}

pub(crate) fn pair_repr_repr_rtype_is_(
    r1: &dyn Repr,
    r2: &dyn Repr,
    hop: &HighLevelOp,
) -> Result<Hlvalue, TyperError> {
    if let Some(value) = hop
        .s_result
        .borrow()
        .as_ref()
        .and_then(|s| s.const_().cloned())
    {
        return Ok(Hlvalue::Constant(inputconst_from_lltype(
            &LowLevelType::Bool,
            &value,
        )?));
    }

    let roriginal1 = r1;
    let roriginal2 = r2;
    let mut robj1 = r1;
    let mut robj2 = r2;
    if matches!(robj1.lowleveltype(), LowLevelType::Void) {
        robj1 = robj2;
    } else if matches!(robj2.lowleveltype(), LowLevelType::Void) {
        robj2 = robj1;
    }

    if !matches!(robj1.lowleveltype(), LowLevelType::Ptr(_))
        || !matches!(robj2.lowleveltype(), LowLevelType::Ptr(_))
    {
        return Err(TyperError::message(format!(
            "is of instances of the non-pointers: {}, {}",
            roriginal1.repr_string(),
            roriginal2.repr_string()
        )));
    }
    if robj1.lowleveltype() != robj2.lowleveltype() {
        return Err(TyperError::message(format!(
            "is of instances of different pointer types: {}, {}",
            roriginal1.repr_string(),
            roriginal2.repr_string()
        )));
    }

    let v_list = hop.inputargs(vec![ConvertedTo::Repr(robj1), ConvertedTo::Repr(robj2)])?;
    Ok(hop
        .genop("ptr_eq", v_list, GenopResult::LLType(LowLevelType::Bool))
        .expect("ptr_eq with Bool result returns a value"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::rtyper::rbool::bool_repr;
    use crate::translator::rtyper::rfloat::float_repr;
    use crate::translator::rtyper::rint::signed_repr;
    use crate::translator::rtyper::rnone::none_repr;
    use std::sync::Arc;

    #[test]
    fn repr_class_id_of_concrete_reprs_returns_their_variant() {
        // Exercises `Repr::repr_class_id` override on each landed Repr
        // — any new Repr port must add its arm here + the matching MRO
        // entry in [`ReprClassId::mro`].
        use ReprClassId::*;
        assert_eq!((*none_repr()).repr_class_id(), NoneRepr);
        assert_eq!((*bool_repr()).repr_class_id(), BoolRepr);
        assert_eq!((*signed_repr()).repr_class_id(), IntegerRepr);
        assert_eq!((*float_repr()).repr_class_id(), FloatRepr);
    }

    #[test]
    fn pair_mro_order_matches_upstream_python() {
        // pairtype.py — outer loop over cls2.__mro__, inner over
        // cls1.__mro__. For (BoolRepr, FloatRepr):
        //  mro1 = [BoolRepr, IntegerRepr, FloatRepr, Repr]
        //  mro2 = [FloatRepr, Repr]
        //  yielded pairs:
        //    (BoolRepr, FloatRepr), (IntegerRepr, FloatRepr),
        //    (FloatRepr, FloatRepr), (Repr, FloatRepr),
        //    (BoolRepr, Repr), (IntegerRepr, Repr), (FloatRepr, Repr),
        //    (Repr, Repr).
        use ReprClassId::*;
        let mro = pair_mro(BoolRepr, FloatRepr);
        assert_eq!(
            mro,
            vec![
                (BoolRepr, FloatRepr),
                (IntegerRepr, FloatRepr),
                (FloatRepr, FloatRepr),
                (Repr, FloatRepr),
                (BoolRepr, Repr),
                (IntegerRepr, Repr),
                (FloatRepr, Repr),
                (Repr, Repr),
            ]
        );
    }

    #[test]
    fn pair_mro_simple_cases_end_with_repr_repr() {
        // For pair(NoneRepr, IntegerRepr) the MRO terminates at
        // (Repr, Repr) after exhausting both chains.
        use ReprClassId::*;
        let mro = pair_mro(NoneRepr, IntegerRepr);
        assert_eq!(mro.first().copied(), Some((NoneRepr, IntegerRepr)));
        assert_eq!(mro.last().copied(), Some((Repr, Repr)));
    }

    #[test]
    fn convert_from_to_any_to_none_produces_void_constant() {
        // rnone.py — `pairtype(Repr, NoneRepr).convert_from_to`
        // returns `inputconst(Void, None)`.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);

        let r_from = signed_repr();
        let r_to = none_repr();
        let dummy_v = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(42),
            LowLevelType::Signed,
        ));
        let converted = pair_convert_from_to(r_from.as_ref(), r_to.as_ref(), &dummy_v, &mut llops)
            .expect("pair_convert_from_to should succeed")
            .expect("convert should not return NotImplemented");
        let Hlvalue::Constant(c) = converted else {
            panic!("expected Constant result");
        };
        assert_eq!(c.value, ConstValue::None);
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Void));
    }

    #[test]
    fn convert_from_to_same_repr_falls_through_to_caller_identity() {
        // Pairtype dispatch returns `Ok(None)` when no MRO entry
        // registers the combination — e.g. (FloatRepr, FloatRepr) has
        // no convert_from_to registration. The `convertvar` caller
        // short-circuits that to identity (same-repr) before even
        // calling the dispatcher, so this test directly exercises the
        // "no handler" NotImplemented path.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);

        let r = float_repr();
        let dummy_v = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Float(1.0_f64.to_bits()),
            LowLevelType::Float,
        ));
        let result = pair_convert_from_to(r.as_ref(), r.as_ref(), &dummy_v, &mut llops)
            .expect("pair_convert_from_to should not error");
        assert!(
            result.is_none(),
            "no (FloatRepr, FloatRepr) convert handler — MRO exhausts to None"
        );
    }

    #[test]
    fn convert_from_to_integer_integer_emits_rint_cast() {
        // rint.py convert_from_to — Signed -> Unsigned emits
        // `cast_int_to_uint` through pairtype(IntegerRepr, IntegerRepr).
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rint::unsigned_repr;
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);

        let r_from = signed_repr();
        let r_to = unsigned_repr();
        let dummy_v = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(1),
            LowLevelType::Signed,
        ));
        let converted = pair_convert_from_to(r_from.as_ref(), r_to.as_ref(), &dummy_v, &mut llops)
            .expect("pair_convert_from_to should succeed")
            .expect("integer conversion should commit");
        assert!(matches!(converted, Hlvalue::Variable(_)));
        assert_eq!(llops.ops.len(), 1);
        assert_eq!(llops.ops[0].opname, "cast_int_to_uint");
    }

    #[test]
    fn convert_from_to_ptr_lladtmeth_uses_same_lowleveltype_like_rptr() {
        // rptr.py — pairtype(PtrRepr, LLADTMethRepr) converts by
        // identity when the low-level pointer type matches.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::translator::rtyper::llannotation::SomeLLADTMeth;
        use crate::translator::rtyper::lltypesystem::lltype::{
            LowLevelPointerType, Ptr, PtrTarget, Struct,
        };
        use crate::translator::rtyper::rmodel::{LLADTMethRepr, PtrRepr};
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let ptr = Ptr {
            TO: PtrTarget::Struct(Struct::gc("S", vec![])),
        };
        let r_from = PtrRepr::new(ptr.clone());
        let adtmeth = SomeLLADTMeth::new(
            LowLevelPointerType::Ptr(ptr.clone()),
            ConstValue::Function(Box::new(crate::flowspace::model::GraphFunc::new(
                "meth",
                Constant::new(ConstValue::None),
            ))),
        );
        let r_to = LLADTMethRepr::new(&adtmeth);
        let dummy_v = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            r_from.lowleveltype().clone(),
        ));

        let converted = pair_convert_from_to(&r_from, &r_to, &dummy_v, &mut llops)
            .expect("pair_convert_from_to should succeed")
            .expect("matching PtrRepr -> LLADTMethRepr should commit");
        assert_eq!(converted, dummy_v);
    }

    #[test]
    fn convert_from_to_interiorptr_interiorptr_requires_dict_equality_like_rptr() {
        // rptr.py __extend__ — lowleveltype equality is not enough here;
        // upstream checks `r_from.__dict__ == r_to.__dict__`.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::translator::rtyper::lltypesystem::lltype::{
            InteriorOffset, InteriorPtr, LowLevelType, Struct,
        };
        use crate::translator::rtyper::rmodel::InteriorPtrRepr;
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let parent = LowLevelType::Struct(Box::new(Struct::gc(
            "S",
            vec![
                ("a".to_string(), LowLevelType::Signed),
                ("b".to_string(), LowLevelType::Signed),
            ],
        )));
        let target = LowLevelType::Signed;
        let ptr_a = InteriorPtr {
            PARENTTYPE: Box::new(parent.clone()),
            TO: Box::new(target.clone()),
            offsets: vec![InteriorOffset::Field("a".to_string())],
        };
        let ptr_b = InteriorPtr {
            PARENTTYPE: Box::new(parent),
            TO: Box::new(target),
            offsets: vec![InteriorOffset::Field("b".to_string())],
        };
        let r_from = InteriorPtrRepr::new(ptr_a.clone());
        let r_same = InteriorPtrRepr::new(ptr_a);
        let r_different = InteriorPtrRepr::new(ptr_b);
        assert_eq!(r_from.lowleveltype(), r_different.lowleveltype());
        let dummy_v = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            r_from.lowleveltype().clone(),
        ));

        let converted = pair_convert_from_to(&r_from, &r_same, &dummy_v, &mut llops)
            .expect("same dict conversion should not error")
            .expect("same dict conversion should commit");
        assert_eq!(converted, dummy_v);
        let rejected = pair_convert_from_to(&r_from, &r_different, &dummy_v, &mut llops)
            .expect("different dict conversion should not error");
        assert!(rejected.is_none());
    }

    #[test]
    fn pair_rtype_add_integer_integer_emits_int_add() {
        // rint.py rtype_add + rint.py _rtype_template.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{ConstValue, Constant, SpaceOperation, Variable};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "add".to_string(),
                vec![
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(2),
                        LowLevelType::Signed,
                    )),
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(3),
                        LowLevelType::Signed,
                    )),
                ],
                Hlvalue::Variable(Variable::new()),
            ),
            Vec::new(),
            llops.clone(),
        );
        let r_signed = signed_repr();
        let r_signed_dyn: Arc<dyn Repr> = r_signed.clone();
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Integer(SomeInteger::new(false, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_signed_dyn.clone()), Some(r_signed_dyn.clone())]);
        *hop.s_result.borrow_mut() = Some(SomeValue::Integer(SomeInteger::new(false, false)));
        *hop.r_result.borrow_mut() = Some(r_signed_dyn);

        let result = pair_rtype_add(r_signed.as_ref(), r_signed.as_ref(), &hop)
            .expect("int add should rtype")
            .expect("int add should return a value");
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "int_add");
    }

    #[test]
    fn pair_rtype_add_ovf_integer_integer_emits_int_add_ovf() {
        // rint.py rtype_add_ovf + rint.py — overflow add goes through
        // `_rtype_template`, marks the llop as the raising point, and
        // emits `int_add_ovf` when neither argument is proven non-negative.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{ConstValue, Constant, SpaceOperation, Variable};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "add_ovf".to_string(),
                vec![
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(2),
                        LowLevelType::Signed,
                    )),
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(-3),
                        LowLevelType::Signed,
                    )),
                ],
                Hlvalue::Variable(Variable::new()),
            ),
            Vec::new(),
            llops.clone(),
        );
        let r_signed = signed_repr();
        let r_signed_dyn: Arc<dyn Repr> = r_signed.clone();
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Integer(SomeInteger::new(false, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_signed_dyn.clone()), Some(r_signed_dyn.clone())]);
        *hop.s_result.borrow_mut() = Some(SomeValue::Integer(SomeInteger::new(false, false)));
        *hop.r_result.borrow_mut() = Some(r_signed_dyn);

        let result = pair_rtype_add_ovf(r_signed.as_ref(), r_signed.as_ref(), &hop)
            .expect("int add_ovf should rtype")
            .expect("int add_ovf should return a value");
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert!(ops._called_exception_is_here_or_cannot_occur);
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "int_add_ovf");
    }

    #[test]
    fn pair_rtype_add_ovf_nonneg_arg_emits_int_add_nonneg_ovf() {
        // rint.py:223-229 — for signed-int results, a proven non-negative
        // operand selects `add_nonneg_ovf`; if it is the first operand,
        // the copied hop swaps the first two args before template lowering.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{ConstValue, Constant, SpaceOperation, Variable};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "add_ovf".to_string(),
                vec![
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(4),
                        LowLevelType::Signed,
                    )),
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(-5),
                        LowLevelType::Signed,
                    )),
                ],
                Hlvalue::Variable(Variable::new()),
            ),
            Vec::new(),
            llops.clone(),
        );
        let r_signed = signed_repr();
        let r_signed_dyn: Arc<dyn Repr> = r_signed.clone();
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Integer(SomeInteger::new(true, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_signed_dyn.clone()), Some(r_signed_dyn.clone())]);
        *hop.s_result.borrow_mut() = Some(SomeValue::Integer(SomeInteger::new(false, false)));
        *hop.r_result.borrow_mut() = Some(r_signed_dyn);

        let result = pair_rtype_add_ovf(r_signed.as_ref(), r_signed.as_ref(), &hop)
            .expect("int add_ovf should rtype")
            .expect("int add_ovf should return a value");
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert!(ops._called_exception_is_here_or_cannot_occur);
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "int_add_nonneg_ovf");
    }

    #[test]
    fn pair_rtype_div_integer_integer_emits_ll_int_py_div_direct_call() {
        // rint.py rtype_floordiv + rint.py _rtype_call_helper — integer `div` is an alias
        // for floordiv and must lower through `_rtype_call_helper`, not
        // through FloatRepr's `float_truediv` MRO fallback.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{ConstValue, Constant, SpaceOperation, Variable};
        use crate::translator::rtyper::lltypesystem::lltype::{_ptr_obj, LowLevelType};
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "div".to_string(),
                vec![
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(7),
                        LowLevelType::Signed,
                    )),
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::Int(3),
                        LowLevelType::Signed,
                    )),
                ],
                Hlvalue::Variable(Variable::new()),
            ),
            Vec::new(),
            llops.clone(),
        );
        let r_signed = signed_repr();
        let r_signed_dyn: Arc<dyn Repr> = r_signed.clone();
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Integer(SomeInteger::new(false, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_signed_dyn.clone()), Some(r_signed_dyn.clone())]);
        *hop.s_result.borrow_mut() = Some(SomeValue::Integer(SomeInteger::new(false, false)));
        *hop.r_result.borrow_mut() = Some(r_signed_dyn);

        let result = pair_rtype_div(r_signed.as_ref(), r_signed.as_ref(), &hop)
            .expect("int div should rtype")
            .expect("int div should return a value");
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert!(ops._called_exception_is_here_or_cannot_occur);
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert_eq!(ops.ops[0].args.len(), 3);
        let Hlvalue::Constant(c_func) = &ops.ops[0].args[0] else {
            panic!("direct_call first arg must be the function constant");
        };
        let ConstValue::LLPtr(ptr) = &c_func.value else {
            panic!("direct_call first arg must be a low-level function pointer");
        };
        let _ptr_obj::Func(func) = ptr._obj().expect("function pointer must be concrete") else {
            panic!("direct_call first arg must point to a function");
        };
        assert_eq!(func._name, "ll_int_py_div");
    }

    #[test]
    fn pair_rtype_add_integer_float_reaches_float_pair_via_mro() {
        // IntegerRepr inherits FloatRepr upstream. For `(IntegerRepr,
        // FloatRepr)` there is no integer-pair `rtype_add`; pair_mro
        // reaches `(FloatRepr, FloatRepr)`, coercing the integer arg to
        // Float before emitting `float_add`.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeFloat, SomeInteger, SomeValue};
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let v_int = Variable::new();
        v_int.set_concretetype(Some(LowLevelType::Signed));
        let v_float = Variable::new();
        v_float.set_concretetype(Some(LowLevelType::Float));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "add".to_string(),
                vec![Hlvalue::Variable(v_int), Hlvalue::Variable(v_float)],
                Hlvalue::Variable(Variable::new()),
            ),
            Vec::new(),
            llops.clone(),
        );
        let r_signed = signed_repr();
        let r_float = float_repr();
        let r_signed_dyn: Arc<dyn Repr> = r_signed.clone();
        let r_float_dyn: Arc<dyn Repr> = r_float.clone();
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Integer(SomeInteger::new(false, false)),
            SomeValue::Float(SomeFloat::new()),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_signed_dyn), Some(r_float_dyn.clone())]);
        *hop.s_result.borrow_mut() = Some(SomeValue::Float(SomeFloat::new()));
        *hop.r_result.borrow_mut() = Some(r_float_dyn);

        let result = pair_rtype_add(r_signed.as_ref(), r_float.as_ref(), &hop)
            .expect("mixed add should rtype")
            .expect("mixed add should return a value");
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 2);
        assert_eq!(ops.ops[0].opname, "cast_int_to_float");
        assert_eq!(ops.ops[1].opname, "float_add");
    }

    #[test]
    fn pair_function_function_convert_from_to_is_identity_and_emits_no_ops() {
        // rpbc.py — pairtype(FunctionRepr,
        // FunctionRepr).convert_from_to is upstream `return v`. Both
        // reprs are Void-typed; the conversion should be a pass-through.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::annotator::description::{DescEntry, FunctionDesc};
        use crate::annotator::model::SomePBC;
        use crate::flowspace::argument::Signature;
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rpbc::FunctionRepr;
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::cell::RefCell as StdRefCell;
        use std::rc::Rc;

        fn f_entry(bk: &Rc<Bookkeeper>, name: &str) -> DescEntry {
            DescEntry::function(Rc::new(StdRefCell::new(FunctionDesc::new(
                bk.clone(),
                None,
                name,
                Signature::new(vec![], None, None),
                None,
                None,
            ))))
        }

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);

        let r_from: Arc<dyn Repr> = FunctionRepr::new(
            &rtyper,
            SomePBC::new(vec![f_entry(&ann.bookkeeper, "f")], false),
        )
        .unwrap();
        let r_to: Arc<dyn Repr> = FunctionRepr::new(
            &rtyper,
            SomePBC::new(vec![f_entry(&ann.bookkeeper, "g")], false),
        )
        .unwrap();

        let input_var = Variable::new();
        input_var.set_concretetype(Some(
            crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Void,
        ));
        let v_in = Hlvalue::Variable(input_var);

        let converted =
            pair_convert_from_to(r_from.as_ref(), r_to.as_ref(), &v_in, &mut llops).unwrap();
        assert_eq!(converted, Some(v_in));
        assert!(
            llops.ops.is_empty(),
            "identity conversion should not emit ops"
        );
    }

    #[test]
    fn pair_functions_pbc_to_function_returns_void_none_constant() {
        // rpbc.py — FunctionsPBCRepr → FunctionRepr is
        // `inputconst(Void, None)`, irrespective of the source variable.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::annotator::description::{DescEntry, FunctionDesc, GraphCacheKey};
        use crate::annotator::model::{SomeInteger, SomePBC, SomeValue};
        use crate::flowspace::argument::Signature;
        use crate::flowspace::model::{ConstValue, Variable};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rpbc::{FunctionRepr, FunctionsPBCRepr};
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::cell::RefCell as StdRefCell;
        use std::rc::Rc;

        fn f_entry(bk: &Rc<Bookkeeper>, name: &str) -> DescEntry {
            DescEntry::function(Rc::new(StdRefCell::new(FunctionDesc::new(
                bk.clone(),
                None,
                name,
                Signature::new(vec!["x".to_string()], None, None),
                None,
                None,
            ))))
        }

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);

        let fd_f = f_entry(&ann.bookkeeper, "f").as_function().unwrap();
        let fd_g = f_entry(&ann.bookkeeper, "g").as_function().unwrap();
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow().cache.borrow_mut().insert(
                GraphCacheKey::None,
                crate::translator::rtyper::rpbc::tests::make_pygraph(name),
            );
        }
        let args = crate::annotator::argument::ArgumentsForTranslation::new(
            vec![Some(SomeValue::Integer(SomeInteger::default()))],
            None,
            None,
        );
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &args,
            &SomeValue::Impossible,
            None,
        )
        .unwrap();

        let s_from = SomePBC::new(
            vec![DescEntry::function(fd_f), DescEntry::function(fd_g.clone())],
            false,
        );
        let r_from: Arc<dyn Repr> = FunctionsPBCRepr::new(&rtyper, s_from).unwrap();
        let r_to: Arc<dyn Repr> = FunctionRepr::new(
            &rtyper,
            SomePBC::new(vec![DescEntry::function(fd_g)], false),
        )
        .unwrap();

        let input_var = Variable::new();
        input_var.set_concretetype(Some(LowLevelType::Void));
        let v_in = Hlvalue::Variable(input_var);
        let converted =
            pair_convert_from_to(r_from.as_ref(), r_to.as_ref(), &v_in, &mut llops).unwrap();
        match converted {
            Some(Hlvalue::Constant(c)) => {
                assert!(matches!(c.value, ConstValue::None));
                assert_eq!(c.concretetype, Some(LowLevelType::Void));
            }
            other => panic!("expected Void None constant, got {other:?}"),
        }
    }

    #[test]
    fn pair_functions_pbc_identity_passes_through_when_lowleveltypes_match() {
        // rpbc.py — two FunctionsPBCRepr reprs with matching
        // lowleveltype collapse to `return v`. Re-using the same
        // FunctionsPBCRepr on both ends is the simplest way to trigger
        // the identity arm.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::annotator::description::{DescEntry, FunctionDesc, GraphCacheKey};
        use crate::annotator::model::{SomeInteger, SomePBC, SomeValue};
        use crate::flowspace::argument::Signature;
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rpbc::FunctionsPBCRepr;
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::cell::RefCell as StdRefCell;
        use std::rc::Rc;

        fn f_entry(bk: &Rc<Bookkeeper>, name: &str) -> DescEntry {
            DescEntry::function(Rc::new(StdRefCell::new(FunctionDesc::new(
                bk.clone(),
                None,
                name,
                Signature::new(vec!["x".to_string()], None, None),
                None,
                None,
            ))))
        }
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);

        let fd_f = f_entry(&ann.bookkeeper, "f").as_function().unwrap();
        let fd_g = f_entry(&ann.bookkeeper, "g").as_function().unwrap();
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow().cache.borrow_mut().insert(
                GraphCacheKey::None,
                crate::translator::rtyper::rpbc::tests::make_pygraph(name),
            );
        }
        let args = crate::annotator::argument::ArgumentsForTranslation::new(
            vec![Some(SomeValue::Integer(SomeInteger::default()))],
            None,
            None,
        );
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &args,
            &SomeValue::Impossible,
            None,
        )
        .unwrap();

        let s_pbc = SomePBC::new(
            vec![DescEntry::function(fd_f), DescEntry::function(fd_g)],
            false,
        );
        let r: Arc<dyn Repr> = FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap();

        let input_var = Variable::new();
        input_var.set_concretetype(Some(r.lowleveltype().clone()));
        let v_in = Hlvalue::Variable(input_var);
        let converted = pair_convert_from_to(r.as_ref(), r.as_ref(), &v_in, &mut llops).unwrap();
        assert_eq!(converted, Some(v_in));
        assert!(llops.ops.is_empty());
    }

    #[test]
    fn convert_from_to_instance_string_is_not_implemented() {
        // rstr.py registers convert_from_to only for Char↔String and
        // UniChar↔Unicode. rmodel.py pairtype(Repr, Repr).convert_from_to
        // returns NotImplemented. There is no (InstanceRepr, StringRepr)
        // pair: PyPy never produces one, because W_UnicodeObject.utf8_w
        // returns the `_utf8` str field (SomeString → StringRepr).
        //
        // A PyObject-shaped InstanceRepr has no `value`/`_utf8` field at
        // the rtyper's view (GcStruct { super, inst_ob_type, inst_w_class }),
        // so a conversion cannot emit getfield either. Inventing the pair
        // would compare a PyObject pointer against an rpy_string pointer.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::classdesc::ClassDef;
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::translator::rtyper::rclass::{Flavor, getinstancerepr};
        use crate::translator::rtyper::rstr::string_repr;
        use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let classdef = ClassDef::new_standalone("pyobject::PyObject", None);
        let r_from =
            getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("PyObject InstanceRepr");
        Repr::setup(r_from.as_ref()).expect("setup InstanceRepr");
        let r_to = string_repr();
        let dummy_v = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            r_from.lowleveltype().clone(),
        ));

        let converted = pair_convert_from_to(r_from.as_ref(), r_to.as_ref(), &dummy_v, &mut llops)
            .expect("pair_convert_from_to should not error");
        assert!(
            converted.is_none(),
            "no (InstanceRepr, StringRepr) convert_from_to — MRO exhausts to NotImplemented"
        );
        assert!(llops.ops.is_empty());

        let value_field = r_from.getfieldrepr("value");
        assert!(
            value_field.is_err(),
            "PyObject InstanceRepr has no `value` field to convert from: {value_field:?}"
        );
        let utf8_field = r_from.getfieldrepr("_utf8");
        assert!(
            utf8_field.is_err(),
            "PyObject InstanceRepr has no `_utf8` field to convert from: {utf8_field:?}"
        );
    }

    #[test]
    fn pair_rtype_eq_instance_string_is_missing_not_a_pointer_compare() {
        // translate_op_eq dispatches on hop.args_r before hop.inputargs
        // consults convert_from_to (rtyper.py). (InstanceRepr, StringRepr)
        // therefore cannot reach pairtype(StringRepr, StringRepr).rtype_eq
        // even if a conversion existed. The missing-pair error is the
        // honest wall; an rtype_eq arm that ptr_eq's the two pointers is
        // not a missing pairtype.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::classdesc::ClassDef;
        use crate::annotator::model::{SomeInstance, SomeString, SomeValue};
        use crate::flowspace::model::{ConstValue, Constant, SpaceOperation, Variable};
        use crate::translator::rtyper::rclass::{Flavor, getinstancerepr};
        use crate::translator::rtyper::rstr::string_repr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};
        use std::cell::RefCell;
        use std::rc::Rc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let classdef = ClassDef::new_standalone("pyobject::PyObject", None);
        let r_inst =
            getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("PyObject InstanceRepr");
        Repr::setup(r_inst.as_ref()).expect("setup InstanceRepr");
        let r_str = string_repr();
        let r_inst_dyn: Arc<dyn Repr> = r_inst.clone();
        let r_str_dyn: Arc<dyn Repr> = r_str.clone();

        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "eq".to_string(),
                vec![
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::None,
                        r_inst.lowleveltype().clone(),
                    )),
                    Hlvalue::Constant(Constant::with_concretetype(
                        ConstValue::ByteStr(b"NFC".to_vec()),
                        r_str.lowleveltype().clone(),
                    )),
                ],
                Hlvalue::Variable(Variable::new()),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
            SomeValue::String(SomeString::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_inst_dyn), Some(r_str_dyn.clone())]);
        *hop.s_result.borrow_mut() =
            Some(SomeValue::Bool(crate::annotator::model::SomeBool::new()));
        *hop.r_result.borrow_mut() =
            Some(crate::translator::rtyper::rbool::bool_repr() as Arc<dyn Repr>);

        let err = pair_rtype_eq(r_inst.as_ref(), r_str.as_ref(), &hop)
            .expect_err("InstanceRepr == StringRepr must not invent a pair");
        assert!(
            err.is_missing_rtype_operation(),
            "expected MissingRTypeOperation, got {err:?}"
        );
        let text = err.to_string();
        assert!(
            text.contains("pair(rtype_eq) not implemented for (InstanceRepr, StringRepr)"),
            "unexpected missing-op text: {text}"
        );
        assert!(
            llops.borrow().ops.is_empty(),
            "must not emit ptr_eq / ll_streq for (InstanceRepr, StringRepr)"
        );
    }
}
