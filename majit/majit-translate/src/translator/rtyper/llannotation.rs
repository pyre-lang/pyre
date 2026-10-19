//! `rpython/rtyper/llannotation.py` port.
//!
//! Carries the scalar / pointer branches that `SomePtr`,
//! `SomeInteriorPtr`, `SomeLLADTMeth`, `_ptrEntry.compute_annotation()`,
//! and low-level call-site annotation need. Address-family support
//! still lands in later line-by-line ports.
//!
//! Upstream, this module hosts:
//!
//! * `annotation_to_lltype` / `lltype_to_annotation` / `ll_to_annotation`
//!   (llannotation.py).
//! * `class SomeInteriorPtr(SomePtr)` (llannotation.py).
//! * `class SomeLLADTMeth(SomeObject)` (llannotation.py).
//! * `pairtype(SomePtr, SomeInteger)` / `pairtype(SomePtr, SomeObject)`
//!   binary dispatch (llannotation.py).
//!
//! In addition, upstream's `class SomePtr` (lltype.py) hosts
//! its own `getattr` / `setattr` / `len` / `bool` / `call` methods —
//! the Rust port re-homes those methods here next to `SomeInteriorPtr`
//! so every pointer-annotation surface sits in a single file. The
//! `SomePtr` struct itself mirrors its upstream home at
//! `lltypesystem/lltype.rs` and is re-exported from `annotator/model.rs`
//! so the `SomeValue::Ptr` variant keeps using the annotator-model path.

use std::collections::HashMap;

use super::lltypesystem::lltype;
use crate::annotator::argument::ArgumentsForTranslation;
use crate::annotator::bookkeeper;
use crate::annotator::model::{
    AnnotatorError, KnownType, SomeAddress, SomeBool, SomeBuiltinMethod, SomeChar, SomeFloat,
    SomeInteger, SomeLongFloat, SomeObject, SomeObjectTrait, SomePtr, SomeRustVec, SomeSingleFloat,
    SomeUnicodeCodePoint, SomeValue, SomeValueTag, s_bool, s_none,
};
use crate::flowspace::model::{ConstValue, Constant};
use crate::flowspace::operation::{CanOnlyThrow, HLOperation, OpKind, Specialization, pure};
use crate::tool::pairtype::DoubleDispatchRegistry;

// llannotation.py annotation_to_lltype — annotation ⇄ lltype helpers.

/// RPython `annotation_to_ll_map` (llannotation.py).
fn annotation_to_ll_map() -> Vec<(SomeValue, lltype::LowLevelType)> {
    vec![
        (
            SomeValue::SingleFloat(SomeSingleFloat::new()),
            lltype::LowLevelType::SingleFloat,
        ),
        (s_none(), lltype::LowLevelType::Void),
        (s_bool(), lltype::LowLevelType::Bool),
        (
            SomeValue::Float(SomeFloat::new()),
            lltype::LowLevelType::Float,
        ),
        (
            SomeValue::LongFloat(SomeLongFloat::new()),
            lltype::LowLevelType::LongFloat,
        ),
        (
            SomeValue::Char(SomeChar::new(false)),
            lltype::LowLevelType::Char,
        ),
        (
            SomeValue::UnicodeCodePoint(SomeUnicodeCodePoint::new(false)),
            lltype::LowLevelType::UniChar,
        ),
        (
            SomeValue::Address(SomeAddress::new()),
            lltype::LowLevelType::Address,
        ),
    ]
}

/// RPython `annotation_to_lltype(s_val, info=None)` (llannotation.py).
pub fn annotation_to_lltype(
    s_val: &SomeValue,
    info: Option<&str>,
) -> Result<lltype::LowLevelType, AnnotatorError> {
    if let SomeValue::InteriorPtr(s_ptr) = s_val {
        let p = &s_ptr.ll_ptrtype;
        if p.offsets
            .iter()
            .any(|offset| matches!(offset, lltype::InteriorOffset::Index(0)))
        {
            let zero_count = p
                .offsets
                .iter()
                .filter(|offset| matches!(offset, lltype::InteriorOffset::Index(0)))
                .count();
            assert_eq!(zero_count, 1);
            let parent_ptr = lltype::Ptr {
                TO: match &*p.PARENTTYPE {
                    lltype::LowLevelType::Struct(t) => lltype::PtrTarget::Struct((**t).clone()),
                    lltype::LowLevelType::Array(t) => lltype::PtrTarget::Array((**t).clone()),
                    lltype::LowLevelType::FixedSizeArray(t) => {
                        lltype::PtrTarget::FixedSizeArray((**t).clone())
                    }
                    lltype::LowLevelType::Opaque(t) => lltype::PtrTarget::Opaque((**t).clone()),
                    other => panic!("SomeInteriorPtr parent must be container type, got {other:?}"),
                },
            };
            return Ok(lltype::LowLevelType::Ptr(Box::new(lltype::Ptr {
                TO: lltype::PtrTarget::Struct(
                    parent_ptr._example()._interior_ptr_type_with_index(&p.TO),
                ),
            })));
        }
        return Ok(lltype::LowLevelType::Ptr(Box::new(lltype::Ptr {
            TO: match &*p.PARENTTYPE {
                lltype::LowLevelType::Struct(t) => lltype::PtrTarget::Struct((**t).clone()),
                lltype::LowLevelType::Array(t) => lltype::PtrTarget::Array((**t).clone()),
                lltype::LowLevelType::FixedSizeArray(t) => {
                    lltype::PtrTarget::FixedSizeArray((**t).clone())
                }
                lltype::LowLevelType::Opaque(t) => lltype::PtrTarget::Opaque((**t).clone()),
                other => panic!("SomeInteriorPtr parent must be container type, got {other:?}"),
            },
        })));
    }
    if let SomeValue::Ptr(s_ptr) = s_val {
        return Ok(lltype::LowLevelType::from(s_ptr.ll_ptrtype.clone()));
    }
    if let SomeValue::Integer(s_int) = s_val {
        return Ok(lltype::build_number(None, s_int.knowntype()));
    }

    for (witness, ty) in annotation_to_ll_map() {
        if witness.contains(s_val) {
            return Ok(ty);
        }
    }

    let prefix = info.map(|s| format!("{s}: ")).unwrap_or_default();
    Err(AnnotatorError::new(format!(
        "{prefix}should return a low-level type,\ngot instead {s_val:?}"
    )))
}

/// `FuncType.ARGS` of a function pointer target (`Ptr.TO` / resolved
/// `ForwardReference`).
fn fn_ptr_func_args(to: &lltype::PtrTarget) -> Option<Vec<lltype::LowLevelType>> {
    match to {
        lltype::PtrTarget::Func(ft) => Some(ft.args.clone()),
        lltype::PtrTarget::ForwardReference(fwd) => match fwd.resolved() {
            Some(lltype::LowLevelType::Func(t)) => Some(t.args.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// Convert one `SomePtr.call` argument to the example ll value
/// `_func.call` type-checks.
///
/// A `SomeInstance` whose classdef casts to `ARGS[i]`
/// (`cast_instance_to_base_ptr` / `instance_can_cast_to_fn_ptr_arg`)
/// is that pointer type's `_defl()`. `s_None` is Void in
/// `annotation_to_ll_map`; a Ptr slot takes the slot's null pointer
/// (`_func.call` accepts None for a pointer ARG). A classdef-less
/// instance at a GC Ptr slot is the same None mapping: `ConstNone`
/// materialises as `GcRef`, so Option None arrives as
/// `SomeInstance(classdef=None)` rather than `s_None`.
fn fn_ptr_arg_llvalue(
    s_arg: &SomeValue,
    arg_ty: Option<&lltype::LowLevelType>,
    info: &str,
) -> Result<lltype::LowLevelValue, AnnotatorError> {
    if let Some(arg_ty) = arg_ty {
        if let SomeValue::Instance(inst) = s_arg
            && crate::translator::rtyper::rclass::instance_can_cast_to_fn_ptr_arg(inst, arg_ty)
        {
            return Ok(arg_ty._defl());
        }
        match annotation_to_lltype(s_arg, Some(info)) {
            Ok(ty) => {
                if matches!(ty, lltype::LowLevelType::Void)
                    && matches!(arg_ty, lltype::LowLevelType::Ptr(_))
                {
                    return Ok(arg_ty._defl());
                }
                Ok(ty._defl())
            }
            Err(err) => {
                if let SomeValue::Instance(inst) = s_arg
                    && inst.classdef.is_none()
                    && matches!(arg_ty, lltype::LowLevelType::Ptr(p) if p._needsgc())
                {
                    return Ok(arg_ty._defl());
                }
                Err(err)
            }
        }
    } else {
        annotation_to_lltype(s_arg, Some(info)).map(|t| t._defl())
    }
}

/// RPython `lltype_to_annotation(T)` (llannotation.py).
///
/// Upstream falls back to `SomePtr(T)` / `SomeInteriorPtr(T)` for the
/// non-primitive cases — `SomePtr.__init__` asserts `isinstance(T,
/// Ptr)`, so bare `Struct`/`Array`/`Func`/`Opaque`/`ForwardReference`
/// would raise `AssertionError` at construction. The Rust port makes
/// that rejection explicit: callers must hand in `Ptr(T)` or
/// `InteriorPtr` rather than the bare container type.
pub fn lltype_to_annotation<T>(t: T) -> SomeValue
where
    T: Into<lltype::LowLevelType>,
{
    let ty = t.into();
    for (ann, mapped_ty) in annotation_to_ll_map() {
        if mapped_ty == ty {
            return ann;
        }
    }
    match ty {
        lltype::LowLevelType::Signed => {
            SomeValue::Integer(SomeInteger::new_with_knowntype(false, KnownType::Int))
        }
        lltype::LowLevelType::Unsigned => {
            SomeValue::Integer(SomeInteger::new_with_knowntype(false, KnownType::Ruint))
        }
        lltype::LowLevelType::SignedLongLong => {
            SomeValue::Integer(SomeInteger::new_with_knowntype(false, KnownType::LongLong))
        }
        lltype::LowLevelType::SignedLongLongLong => SomeValue::Integer(
            SomeInteger::new_with_knowntype(false, KnownType::LongLongLong),
        ),
        lltype::LowLevelType::UnsignedLongLong => {
            SomeValue::Integer(SomeInteger::new_with_knowntype(false, KnownType::ULongLong))
        }
        lltype::LowLevelType::UnsignedLongLongLong => SomeValue::Integer(
            SomeInteger::new_with_knowntype(false, KnownType::ULongLongLong),
        ),
        lltype::LowLevelType::InteriorPtr(t) => SomeValue::InteriorPtr(SomeInteriorPtr::new(*t)),
        lltype::LowLevelType::Ptr(t) => SomeValue::Ptr(SomePtr::new(*t)),
        other => {
            debug_assert!(other.is_container_type());
            panic!(
                "lltype_to_annotation: container type {:?} must be wrapped in Ptr(T) or \
                 InteriorPtr — upstream SomePtr.__init__ asserts isinstance(T, Ptr)",
                other
            )
        }
    }
}

/// RPython `ll_to_annotation(v)` (llannotation.py).
pub fn ll_to_annotation(v: lltype::LowLevelValue) -> SomeValue {
    if let lltype::LowLevelValue::InteriorPtr(ptr) = v {
        return SomeValue::InteriorPtr(SomeInteriorPtr::new(ptr._TYPE()));
    }
    lltype_to_annotation(lltype::typeOf_value(&v))
}

// lltype.py — SomePtr pointer-specific methods.

impl SomePtr {
    /// RPython `SomePtr.bool(self)` (lltype.py).
    pub fn bool(&self) -> SomeValue {
        let mut result = SomeBool::new();
        if self.is_constant()
            && let Some(c) = &self.base.const_box
            && let Some(truthy) = c.value.truthy()
        {
            result.base.const_box = Some(Constant::new(ConstValue::Bool(truthy)));
        }
        SomeValue::Bool(result)
    }

    /// RPython `SomePtr.len(self)` (lltype.py). Propagates the
    /// `TypeError` upstream raises from `_fixedlength` on non-array
    /// pointers as an `AnnotatorError`.
    pub fn len(&self) -> Result<SomeValue, AnnotatorError> {
        let length = self
            .ll_ptrtype
            ._example()
            ._fixedlength()
            .map_err(AnnotatorError::new)?;
        if let Some(length) = length {
            bookkeeper::immutablevalue(&ConstValue::Int(length))
        } else {
            Ok(SomeValue::Integer(SomeInteger::new(true, false)))
        }
    }

    /// RPython `SomePtr.getattr(self, s_attr)` (lltype.py).
    pub fn getattr(&self, s_attr: &SomeValue) -> Result<SomeValue, AnnotatorError> {
        if !s_attr.is_constant() {
            return Err(AnnotatorError::new(format!(
                "getattr on ptr {:?} with non-constant field-name",
                self.ll_ptrtype
            )));
        }
        // lltype.py `SomePtr.getattr` — upstream guards on
        // `s_attr.is_constant()` only and forwards `s_attr.const`
        // straight into `_lookup_adtmeth` / `getattr(example, ...)`
        // with no `isinstance(..., str)` gate. Use [`ConstValue::as_text`]
        // for the same `ByteStr`-or-`UniStr` acceptance as the
        // sibling `setattr` / `bool` / `len` arms below at lines 273 /
        // 340 / 388 — keeps the four ll field-name paths consistent.
        let Some(attr) = s_attr.const_().and_then(ConstValue::as_text) else {
            panic!(
                "SomePtr.getattr expects a constant string field-name, got {:?}",
                s_attr.const_()
            );
        };
        let example = self.ll_ptrtype._example();
        match example._lookup_adtmeth(attr) {
            Err(lltype::AttributeError) => match example.getattr(attr) {
                Ok(v) => Ok(ll_to_annotation(v)),
                // A raw-pointer `p.is_null()` whose receiver keeps its
                // `SomePtr` tag (e.g. the `*mut u8` opaque GCREF from
                // `try_gc_alloc_stable_raw`, `TO: Opaque(GCREF)`) reaches
                // this `SomePtr.getattr` rather than the `SomeObject` /
                // `SomeInstance` `is_null` arms.  `front::mir` already
                // routed it as `CallTarget::Method { name: "is_null",
                // receiver_root: "mut_ptr"/"const_ptr" }`, which
                // `jtransform` lowers to `ptr_iszero` regardless of the
                // receiver repr; answer the getattr with the same
                // `ptr_method_is_null` bound method so the result types as
                // `SomeBool`.  Last-resort after both `_lookup_adtmeth` and
                // the struct-field `getattr`, so a real `is_null` adt
                // member or field is never shadowed.
                Err(_) if attr == "is_null" => {
                    Ok(SomeValue::BuiltinMethod(SomeBuiltinMethod::new(
                        "ptr_method_is_null",
                        SomeValue::Ptr(self.clone()),
                        "is_null",
                    )))
                }
                // `Option<RawStruct>::as_ref` / `<*mut T>::as_ref`: the
                // receiver is already the nullable pointer word
                // (`raw_struct_ptr_annotation`). Identity, same as
                // `front::mir` `is_raw_ptr_as_ref`.
                Err(_) if attr == "as_ref" || attr == "as_mut" => Ok(SomeValue::BuiltinMethod(
                    SomeBuiltinMethod::new("ptr_method_as_ref", SomeValue::Ptr(self.clone()), attr),
                )),
                Err(e) => Err(AnnotatorError::new(e)),
            },
            Ok(lltype::LowLevelAdtMember::Method { ll_ptrtype, func }) => {
                Ok(SomeValue::LLADTMeth(SomeLLADTMeth::new(ll_ptrtype, func)))
            }
            Ok(lltype::LowLevelAdtMember::Value(v)) => bookkeeper::immutablevalue(&v),
        }
    }

    /// RPython `SomePtr.call(self, args)` (lltype.py).
    pub fn call(&self, args: &ArgumentsForTranslation) -> Result<SomeValue, AnnotatorError> {
        let (args_s_opt, kwds_s) = args
            .unpack()
            .map_err(|err| AnnotatorError::new(err.getmsg()))?;
        if !kwds_s.is_empty() {
            return Err(AnnotatorError::new(
                "keyword arguments to call to a low-level fn ptr",
            ));
        }
        let info = "argument to ll function pointer call";
        // `lltype.py SomePtr.call`: `[annotation_to_lltype(
        // s_arg, info)._defl() for s_arg in args_s]` — touches every
        // slot once via the list comprehension; `annotation_to_lltype`
        // raises `AttributeError` on `None.knowntype` at the first slot
        // that is unbound.  Mirror via per-touch [`arg_at`] so an
        // unannotated slot panics with the call boundary name + slot
        // index at the same iteration upstream would raise.
        // Instance arguments are converted against `FuncType.ARGS`
        // (`cast_instance_to_base_ptr`); `annotation_to_lltype` itself
        // has no instance arm.
        let func_args = fn_ptr_func_args(&self.ll_ptrtype.TO);
        let llargs = (0..args_s_opt.len())
            .map(|i| {
                let s_arg = crate::annotator::builtin::arg_at(&args_s_opt, i, "SomePtr.call");
                let arg_ty = func_args.as_ref().and_then(|args| args.get(i));
                fn_ptr_arg_llvalue(s_arg, arg_ty, info)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let v = self.ll_ptrtype._example().call(&llargs);
        Ok(ll_to_annotation(v))
    }

    /// RPython `SomePtr.setattr(self, s_attr, s_value)` (lltype.py).
    pub fn setattr(
        &self,
        s_attr: &SomeValue,
        s_value: &SomeValue,
    ) -> Result<SomeValue, AnnotatorError> {
        if !s_attr.is_constant() {
            return Err(AnnotatorError::new(format!(
                "setattr on ptr {:?} with non-constant field-name",
                self.ll_ptrtype
            )));
        }
        let Some(attr) = s_attr.const_().and_then(ConstValue::as_text) else {
            panic!(
                "SomePtr.setattr expects a constant string field-name, got {:?}",
                s_attr.const_()
            );
        };
        let mut example = self.ll_ptrtype._example();
        if example.getattr(attr).map_err(AnnotatorError::new)? != lltype::LowLevelValue::Void {
            let v_lltype = annotation_to_lltype(s_value, None)?;
            example
                .setattr(attr, v_lltype._defl())
                .map_err(AnnotatorError::new)?;
        }
        Ok(SomeValue::Impossible)
    }
}

/// Named scalar fields of `rust_vec_lltype`'s `Struct(raw) "RustVec"`
/// header (`ptr`, `len`, `cap`). Only `len` is a field getattr/setattr:
/// it is the word `Vec::set_len` writes and the word `len()` reads.
/// `ptr` is a container (`Ptr(Array(ITEM, hints={nolength}))`); `cap`
/// is `Signed` but the front never lowers a store of it.
fn rust_vec_header_field_type(attr: &str) -> Option<lltype::LowLevelType> {
    match attr {
        "len" => Some(lltype::LowLevelType::Signed),
        _ => None,
    }
}

/// Integer `Number` primitives (`lltype.py`). `Vec::set_len` takes
/// `usize` (`Unsigned` / `r_uint`) while the header `len` word is
/// `Signed` (`rlist.py` `l.length`). `RustVecRepr.rtype_setattr`
/// already `ConvertedTo` `Signed`; the annotator accepts the same
/// family so the store is not rejected before that conversion.
fn integer_number_lltype(ty: &lltype::LowLevelType) -> bool {
    matches!(
        ty,
        lltype::LowLevelType::Signed
            | lltype::LowLevelType::Unsigned
            | lltype::LowLevelType::SignedLongLong
            | lltype::LowLevelType::SignedLongLongLong
            | lltype::LowLevelType::UnsignedLongLong
            | lltype::LowLevelType::UnsignedLongLongLong
    )
}

impl SomeRustVec {
    /// `SomePtr.getattr` (lltype.py) for a pointer to the raw
    /// `{ptr, len, cap}` header. Methods (`append`, …) resolve through
    /// `find_method` first, the way `SomeObject.getattr` (unaryop.py)
    /// does, then the `len` field is `lltype_to_annotation(Signed)`.
    pub fn getattr(&self, s_attr: &SomeValue) -> Result<SomeValue, AnnotatorError> {
        if !s_attr.is_constant() {
            return Err(AnnotatorError::new(
                "getattr on RustVec with non-constant field-name",
            ));
        }
        let Some(attr) = s_attr.const_().and_then(ConstValue::as_text) else {
            panic!(
                "SomeRustVec.getattr expects a constant string field-name, got {:?}",
                s_attr.const_()
            );
        };
        if let Some(s_method) = SomeValue::RustVec(self.clone()).find_method(attr) {
            return Ok(s_method);
        }
        let Some(field_t) = rust_vec_header_field_type(attr) else {
            return Err(AnnotatorError::new(format!(
                "struct RustVec has no field {attr:?}"
            )));
        };
        Ok(lltype_to_annotation(field_t))
    }

    /// `SomePtr.setattr` (lltype.py): constant name, the field exists
    /// on the pointed-to header, and `s_value` fits its lowleveltype.
    /// Only `len` is settable (`Vec::set_len` → `FieldWrite` of `len`).
    pub fn setattr(
        &self,
        s_attr: &SomeValue,
        s_value: &SomeValue,
    ) -> Result<SomeValue, AnnotatorError> {
        if !s_attr.is_constant() {
            return Err(AnnotatorError::new(
                "setattr on RustVec with non-constant field-name",
            ));
        }
        let Some(attr) = s_attr.const_().and_then(ConstValue::as_text) else {
            panic!(
                "SomeRustVec.setattr expects a constant string field-name, got {:?}",
                s_attr.const_()
            );
        };
        let Some(field_t) = rust_vec_header_field_type(attr) else {
            return Err(AnnotatorError::new(format!(
                "struct RustVec has no field {attr:?}"
            )));
        };
        let v_lltype = annotation_to_lltype(s_value, None)?;
        if v_lltype != field_t
            && !(integer_number_lltype(&v_lltype) && integer_number_lltype(&field_t))
        {
            return Err(AnnotatorError::new(format!(
                "RustVec instance field {attr:?}:\nexpects {field_t:?}\n    got {v_lltype:?}"
            )));
        }
        Ok(SomeValue::Impossible)
    }
}

// llannotation.py — class SomeInteriorPtr(SomePtr)

/// RPython `class SomeInteriorPtr(SomePtr)` (llannotation.py).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeInteriorPtr {
    pub base: SomeObject,
    pub ll_ptrtype: lltype::InteriorPtr,
}

impl SomeInteriorPtr {
    pub fn new(ll_ptrtype: lltype::InteriorPtr) -> Self {
        SomeInteriorPtr {
            base: SomeObject::new(KnownType::LlPtr, true),
            ll_ptrtype,
        }
    }

    pub fn bool(&self) -> SomeValue {
        let mut result = SomeBool::new();
        if self.is_constant()
            && let Some(c) = &self.base.const_box
            && let Some(truthy) = c.value.truthy()
        {
            result.base.const_box = Some(Constant::new(ConstValue::Bool(truthy)));
        }
        SomeValue::Bool(result)
    }

    pub fn len(&self) -> Result<SomeValue, AnnotatorError> {
        let length = self
            .ll_ptrtype
            ._example()
            ._fixedlength()
            .map_err(AnnotatorError::new)?;
        if let Some(length) = length {
            bookkeeper::immutablevalue(&ConstValue::Int(length))
        } else {
            Ok(SomeValue::Integer(SomeInteger::new(true, false)))
        }
    }

    pub fn getattr(&self, s_attr: &SomeValue) -> Result<SomeValue, AnnotatorError> {
        if !s_attr.is_constant() {
            return Err(AnnotatorError::new(format!(
                "getattr on ptr {:?} with non-constant field-name",
                self.ll_ptrtype
            )));
        }
        let Some(attr) = s_attr.const_().and_then(ConstValue::as_text) else {
            panic!(
                "SomeInteriorPtr.getattr expects a constant string field-name, got {:?}",
                s_attr.const_()
            );
        };
        let example = self.ll_ptrtype._example();
        match example._lookup_adtmeth(attr) {
            Err(lltype::AttributeError) => {
                let v = example.getattr(attr).map_err(AnnotatorError::new)?;
                Ok(ll_to_annotation(v))
            }
            Ok(lltype::LowLevelAdtMember::Method { ll_ptrtype, func }) => {
                Ok(SomeValue::LLADTMeth(SomeLLADTMeth::new(ll_ptrtype, func)))
            }
            Ok(lltype::LowLevelAdtMember::Value(v)) => bookkeeper::immutablevalue(&v),
        }
    }

    pub fn call(&self, args: &ArgumentsForTranslation) -> Result<SomeValue, AnnotatorError> {
        let (args_s_opt, kwds_s) = args
            .unpack()
            .map_err(|err| AnnotatorError::new(err.getmsg()))?;
        if !kwds_s.is_empty() {
            return Err(AnnotatorError::new(
                "keyword arguments to call to a low-level fn ptr",
            ));
        }
        let info = "argument to ll function pointer call";
        // `lltype.py SomePtr.call`: list comprehension touches
        // every slot once via `annotation_to_lltype(s_arg, info)._defl()`;
        // `annotation_to_lltype` raises `AttributeError` on
        // `None.knowntype` at the first slot that is unbound.  Per-touch
        // [`arg_at`] mirror so an unannotated slot panics with the call
        // boundary name + slot index at the same iteration upstream
        // would raise (`unaryop.py simple_call_SomeBuiltin` bind-
        // then-body sequence).
        let func_args = match &*self.ll_ptrtype.TO {
            lltype::LowLevelType::Func(ft) => Some(ft.args.clone()),
            lltype::LowLevelType::Ptr(p) => fn_ptr_func_args(&p.TO),
            _ => None,
        };
        let llargs = (0..args_s_opt.len())
            .map(|i| {
                let s_arg = crate::annotator::builtin::arg_at(&args_s_opt, i, "SomeLLPtr.call");
                let arg_ty = func_args.as_ref().and_then(|args| args.get(i));
                fn_ptr_arg_llvalue(s_arg, arg_ty, info)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let v = self.ll_ptrtype._example().call(&llargs);
        Ok(ll_to_annotation(v))
    }

    pub fn setattr(
        &self,
        s_attr: &SomeValue,
        s_value: &SomeValue,
    ) -> Result<SomeValue, AnnotatorError> {
        if !s_attr.is_constant() {
            return Err(AnnotatorError::new(format!(
                "setattr on ptr {:?} with non-constant field-name",
                self.ll_ptrtype
            )));
        }
        let Some(attr) = s_attr.const_().and_then(ConstValue::as_text) else {
            panic!(
                "SomeInteriorPtr.setattr expects a constant string field-name, got {:?}",
                s_attr.const_()
            );
        };
        let mut example = self.ll_ptrtype._example();
        if example.getattr(attr).map_err(AnnotatorError::new)? != lltype::LowLevelValue::Void {
            let v_lltype = annotation_to_lltype(s_value, None)?;
            example
                .setattr(attr, v_lltype._defl())
                .map_err(AnnotatorError::new)?;
        }
        Ok(SomeValue::Impossible)
    }
}

impl SomeObjectTrait for SomeInteriorPtr {
    fn knowntype(&self) -> KnownType {
        KnownType::LlPtr
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

// llannotation.py — class SomeLLADTMeth(SomeObject)

/// RPython `class SomeLLADTMeth(SomeObject)` (llannotation.py).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeLLADTMeth {
    pub base: SomeObject,
    pub ll_ptrtype: lltype::LowLevelPointerType,
    pub func: ConstValue,
}

impl SomeLLADTMeth {
    pub fn new(ll_ptrtype: lltype::LowLevelPointerType, func: ConstValue) -> Self {
        SomeLLADTMeth {
            base: SomeObject::new(KnownType::Object, true),
            ll_ptrtype,
            func,
        }
    }

    /// RPython `SomeLLADTMeth.call(self, args)` (llannotation.py).
    pub fn call(&self, args: &ArgumentsForTranslation) -> Result<SomeValue, AnnotatorError> {
        let s_func = bookkeeper::immutablevalue(&self.func)?;
        // A low-level adt-method callee always yields a concrete annotation
        // (`ll_to_annotation(FUNCTYPE.RESULT)` / a function-result PBC), never
        // the void `None` that a builtin method analyser produces, so the
        // collapse is a guard rather than a behaviour change.
        Ok(s_func
            .call(&args.prepend(lltype_to_annotation(self.ll_ptrtype.clone())))?
            .unwrap_or(SomeValue::Impossible))
    }
}

impl SomeObjectTrait for SomeLLADTMeth {
    fn knowntype(&self) -> KnownType {
        KnownType::Object
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

// llannotation.py __extend__ — pairtype(SomePtr, SomeInteger/Object)

/// Register the `SomePtr`/`SomeInteger` and `SomePtr`/`SomeObject`
/// pair dispatch tables. Invoked from `annotator::binaryop::init` at
/// `_REGISTRY_DOUBLE` construction time; upstream runs as the `class
/// __extend__(pairtype(...))` module-import side effect.
pub(crate) fn init_pairtypes(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    init_ptr_integer_pairtype(reg);
    init_ptr_object_pairtype(reg);
    init_address_pairtypes(reg);
}

fn register(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
    op: OpKind,
    tag1: SomeValueTag,
    tag2: SomeValueTag,
    spec: Specialization,
) {
    reg.entry(op).or_default().set((tag1, tag2), spec);
}

fn init_ptr_integer_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    register(
        reg,
        OpKind::GetItem,
        SomeValueTag::Ptr,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(ptr_integer_getitem),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::SetItem,
        SomeValueTag::Ptr,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(ptr_integer_setitem),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
}

fn ptr_integer_getitem(
    ann: &crate::annotator::annrpython::RPythonAnnotator,
    hl: &HLOperation,
) -> Option<SomeValue> {
    // llannotation.py — `getitem`: on IndexError (e.g.
    // FixedSizeArray(0)) returns None (void → result bound to Impossible
    // without blocking); otherwise returns the item annotation.
    match ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible) {
        SomeValue::Ptr(p) => match p.ll_ptrtype._example().getitem(0) {
            Ok(v) => Some(ll_to_annotation(v)),
            Err(err) if err.contains("out of bounds") => None,
            Err(err) => panic!("{err}"),
        },
        SomeValue::InteriorPtr(p) => match p.ll_ptrtype._example().getitem(0) {
            Ok(v) => Some(ll_to_annotation(v)),
            Err(err) if err.contains("out of bounds") => None,
            Err(err) => panic!("{err}"),
        },
        _ => panic!("ptr_integer_getitem: arg 0 not ptr-like"),
    }
}

fn ptr_integer_setitem(
    ann: &crate::annotator::annrpython::RPythonAnnotator,
    hl: &HLOperation,
) -> Option<SomeValue> {
    let s_value = ann
        .annotation(&hl.args[2])
        .expect("ptr_integer_setitem: missing value arg");
    match ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible) {
        SomeValue::Ptr(p) => {
            let mut example = p.ll_ptrtype._example();
            match example.getitem(0) {
                Ok(lltype::LowLevelValue::Void) => {}
                Ok(_) => {
                    let v_lltype = annotation_to_lltype(&s_value, None)
                        .expect("ptr_integer_setitem: annotation_to_lltype failed");
                    example
                        .setitem(0, v_lltype._defl())
                        .unwrap_or_else(|err| panic!("{err}"));
                }
                Err(err) => panic!("{err}"),
            }
        }
        SomeValue::InteriorPtr(p) => {
            let mut example = p.ll_ptrtype._example();
            match example.getitem(0) {
                Ok(lltype::LowLevelValue::Void) => {}
                Ok(_) => {
                    let v_lltype = annotation_to_lltype(&s_value, None)
                        .expect("ptr_integer_setitem: annotation_to_lltype failed");
                    example
                        .setitem(0, v_lltype._defl())
                        .unwrap_or_else(|err| panic!("{err}"));
                }
                Err(err) => panic!("{err}"),
            }
        }
        _ => panic!("ptr_integer_setitem: arg 0 not ptr-like"),
    }
    // llannotation.py — setitem "just doing checking", returns None.
    None
}

fn init_ptr_object_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    register(
        reg,
        OpKind::GetItem,
        SomeValueTag::Ptr,
        SomeValueTag::Object,
        Specialization {
            apply: pure(|ann, hl| {
                let p = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
                let obj = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
                panic!(
                    "AnnotatorError: ptr {:?} getitem index not an int: {:?}",
                    match p {
                        SomeValue::Ptr(p) => lltype::LowLevelType::from(p.ll_ptrtype),
                        SomeValue::InteriorPtr(p) => lltype::LowLevelType::from(p.ll_ptrtype),
                        _ => panic!("ptr_object_getitem: arg 0 not ptr-like"),
                    },
                    obj
                );
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    register(
        reg,
        OpKind::SetItem,
        SomeValueTag::Ptr,
        SomeValueTag::Object,
        Specialization {
            apply: pure(|ann, hl| {
                let p = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
                let obj = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
                panic!(
                    "AnnotatorError: ptr {:?} setitem index not an int: {:?}",
                    match p {
                        SomeValue::Ptr(p) => lltype::LowLevelType::from(p.ll_ptrtype),
                        SomeValue::InteriorPtr(p) => lltype::LowLevelType::from(p.ll_ptrtype),
                        _ => panic!("ptr_object_setitem: arg 0 not ptr-like"),
                    },
                    obj
                );
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// llannotation.py __extend__ — address-family pair dispatch.

/// Register the `SomeTypedAddressAccess`/`SomeInteger` getitem/setitem,
/// `SomeAddress`/`SomeInteger` add/sub, and `SomeAddress`/`SomeAddress`
/// sub dispatch tables (llannotation.py).
fn init_address_pairtypes(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // llannotation.py `getitem.can_only_throw = []`.
    register(
        reg,
        OpKind::GetItem,
        SomeValueTag::TypedAddressAccess,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(typed_address_access_integer_getitem),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // llannotation.py `setitem.can_only_throw = []`.
    register(
        reg,
        OpKind::SetItem,
        SomeValueTag::TypedAddressAccess,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(typed_address_access_integer_setitem),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // llannotation.py `def add((s_addr, s_int)): return SomeAddress()`.
    register(
        reg,
        OpKind::Add,
        SomeValueTag::Address,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Address(SomeAddress::new())),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // llannotation.py `def sub((s_addr, s_int)): return SomeAddress()`.
    register(
        reg,
        OpKind::Sub,
        SomeValueTag::Address,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Address(SomeAddress::new())),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // llannotation.py `def sub((s_addr1, s_addr2))`.
    register(
        reg,
        OpKind::Sub,
        SomeValueTag::Address,
        SomeValueTag::Address,
        Specialization {
            apply: pure(address_address_sub),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // llannotation.py `def is_((s_addr1, s_addr2)): assert False,
    // "comparisons with is not supported by addresses"`. A more specific
    // (Address, Address) entry shadows the (Object, Object) `is__default`.
    register(
        reg,
        OpKind::Is,
        SomeValueTag::Address,
        SomeValueTag::Address,
        Specialization {
            apply: pure(|_ann, _hl| panic!("comparisons with is not supported by addresses")),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
}

/// llannotation.py `def getitem((s_taa, s_int)): return
/// lltype_to_annotation(s_taa.type)`.
fn typed_address_access_integer_getitem(
    ann: &crate::annotator::annrpython::RPythonAnnotator,
    hl: &HLOperation,
) -> SomeValue {
    match ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible) {
        SomeValue::TypedAddressAccess(taa) => lltype_to_annotation(taa.access_type),
        _ => panic!("typed_address_access_integer_getitem: arg 0 not TypedAddressAccess"),
    }
}

/// llannotation.py `def setitem((s_taa, s_int), s_value): assert
/// annotation_to_lltype(s_value) is s_taa.type`.
fn typed_address_access_integer_setitem(
    ann: &crate::annotator::annrpython::RPythonAnnotator,
    hl: &HLOperation,
) -> Option<SomeValue> {
    let s_value = ann
        .annotation(&hl.args[2])
        .expect("typed_address_access_integer_setitem: missing value arg");
    match ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible) {
        SomeValue::TypedAddressAccess(taa) => {
            let v_lltype = annotation_to_lltype(&s_value, None)
                .expect("typed_address_access_integer_setitem: annotation_to_lltype failed");
            assert_eq!(
                v_lltype, taa.access_type,
                "setitem value lltype must match the address access type",
            );
        }
        _ => panic!("typed_address_access_integer_setitem: arg 0 not TypedAddressAccess"),
    }
    // llannotation.py — setitem asserts the value type, returns None.
    None
}

/// llannotation.py sub — `addr - addr`. Both null-address constants
/// fold to the constant `0`; otherwise a plain `SomeInteger`.
fn address_address_sub(
    ann: &crate::annotator::annrpython::RPythonAnnotator,
    hl: &HLOperation,
) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    let s1 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
    if let (SomeValue::Address(a0), SomeValue::Address(a1)) = (&s0, &s1)
        && a0.is_null_address()
        && a1.is_null_address()
    {
        return bookkeeper::immutablevalue(&ConstValue::Int(0))
            .expect("immutablevalue(0) must succeed");
    }
    SomeValue::Integer(SomeInteger::new(false, false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::model::{KnownType, SomeBool};

    #[test]
    fn init_address_pairtypes_registers_getitem_setitem_add_sub() {
        use crate::flowspace::operation::OpKind;
        let mut reg: HashMap<
            OpKind,
            DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>,
        > = HashMap::new();
        init_address_pairtypes(&mut reg);

        let taa = SomeValueTag::TypedAddressAccess;
        let int = SomeValueTag::Integer;
        let addr = SomeValueTag::Address;

        assert!(
            reg[&OpKind::GetItem]
                .get((taa, int), taa.mro(), int.mro())
                .is_some()
        );
        assert!(
            reg[&OpKind::SetItem]
                .get((taa, int), taa.mro(), int.mro())
                .is_some()
        );
        assert!(
            reg[&OpKind::Add]
                .get((addr, int), addr.mro(), int.mro())
                .is_some()
        );
        assert!(
            reg[&OpKind::Sub]
                .get((addr, int), addr.mro(), int.mro())
                .is_some()
        );
        assert!(
            reg[&OpKind::Sub]
                .get((addr, addr), addr.mro(), addr.mro())
                .is_some()
        );
        // llannotation.py — (Address, Address) `is_` is registered
        // (and shadows the (Object, Object) `is__default`), so that
        // comparing two addresses with `is` reaches the asserting arm.
        assert!(
            reg[&OpKind::Is]
                .get((addr, addr), addr.mro(), addr.mro())
                .is_some()
        );
    }

    #[test]
    fn annotation_to_ll_map_exposes_upstream_witness_order() {
        let mapped: Vec<lltype::LowLevelType> = annotation_to_ll_map()
            .into_iter()
            .map(|(_ann, ty)| ty)
            .collect();
        assert_eq!(
            mapped,
            vec![
                lltype::LowLevelType::SingleFloat,
                lltype::LowLevelType::Void,
                lltype::LowLevelType::Bool,
                lltype::LowLevelType::Float,
                lltype::LowLevelType::LongFloat,
                lltype::LowLevelType::Char,
                lltype::LowLevelType::UniChar,
                lltype::LowLevelType::Address,
            ]
        );
    }

    #[test]
    fn annotation_to_lltype_maps_scalar_and_pointer_annotations() {
        let signed = annotation_to_lltype(
            &SomeValue::Integer(SomeInteger::new_with_knowntype(false, KnownType::Int)),
            None,
        )
        .unwrap();
        assert_eq!(signed, lltype::LowLevelType::Signed);

        let unsigned = annotation_to_lltype(
            &SomeValue::Integer(SomeInteger::new_with_knowntype(false, KnownType::Ruint)),
            None,
        )
        .unwrap();
        assert_eq!(unsigned, lltype::LowLevelType::Unsigned);

        let ptr = lltype::Ptr {
            TO: lltype::PtrTarget::Func(lltype::FuncType {
                args: vec![lltype::LowLevelType::Signed],
                result: lltype::LowLevelType::Void,
            }),
        };
        let s_ptr = SomeValue::Ptr(SomePtr::new(ptr.clone()));
        assert_eq!(
            annotation_to_lltype(&s_ptr, None).unwrap(),
            lltype::LowLevelType::from(ptr)
        );
        assert_eq!(
            annotation_to_lltype(&SomeValue::Bool(SomeBool::new()), None).unwrap(),
            lltype::LowLevelType::Bool
        );
        assert_eq!(
            annotation_to_lltype(&SomeValue::Address(SomeAddress::new()), None).unwrap(),
            lltype::LowLevelType::Address
        );

        let s_interior = SomeValue::InteriorPtr(SomeInteriorPtr::new(lltype::InteriorPtr {
            PARENTTYPE: Box::new(lltype::LowLevelType::Struct(Box::new(lltype::Struct::new(
                "S",
                vec![("x".into(), lltype::LowLevelType::Signed)],
            )))),
            TO: Box::new(lltype::LowLevelType::Signed),
            offsets: vec![lltype::InteriorOffset::Field("x".into())],
        }));
        assert!(matches!(
            annotation_to_lltype(&s_interior, None).unwrap(),
            lltype::LowLevelType::Ptr(_)
        ));
    }

    #[test]
    fn annotation_to_lltype_fn_ptr_arg_accepts_instance() {
        use crate::annotator::model::SomeInstance;
        let inst = SomeValue::Instance(SomeInstance::new(None, true, Default::default()));
        assert!(
            annotation_to_lltype(&inst, None).is_err(),
            "a high-level instance is not an lltype outside a fn-ptr call"
        );
        let err = annotation_to_lltype(&inst, Some("argument to ll function pointer call"))
            .expect_err("annotation_to_lltype has no instance arm");
        assert!(
            err.msg
                .unwrap_or_default()
                .contains("should return a low-level type"),
            "upstream annotation_to_lltype rejects a high-level instance"
        );
    }

    fn classdef_for_fn_ptr_call_test(
        name: &str,
    ) -> (
        std::rc::Rc<crate::annotator::annrpython::RPythonAnnotator>,
        std::rc::Rc<std::cell::RefCell<crate::annotator::classdesc::ClassDef>>,
    ) {
        use crate::annotator::classdesc::ClassDesc;
        use crate::annotator::description::DescEntry;
        use crate::flowspace::model::HostObject;
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let host = HostObject::new_class(name, vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let DescEntry::Class(class_rc) = &entry else {
            panic!("expected Class desc for {name}");
        };
        let classdef = ClassDesc::getuniqueclassdef(class_rc).unwrap();
        (ann, classdef)
    }

    #[test]
    fn someptr_call_accepts_matching_instance_as_fn_ptr_arg() {
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::model::SomeInstance;
        let (_ann, classdef) = classdef_for_fn_ptr_call_test("MatchingFnPtrInst");
        let inst =
            SomeValue::Instance(SomeInstance::new(Some(classdef), false, Default::default()));
        let s_ptr = SomePtr::new(lltype::Ptr {
            TO: lltype::PtrTarget::Func(lltype::FuncType {
                args: vec![crate::translator::rtyper::rclass::OBJECTPTR.clone()],
                result: lltype::LowLevelType::Void,
            }),
        });
        let args = ArgumentsForTranslation::new(vec![Some(inst)], None, None);
        s_ptr
            .call(&args)
            .expect("matching instance casts to OBJECTPTR at SomePtr.call");
    }

    #[test]
    fn someptr_call_rejects_mismatched_instance_as_fn_ptr_arg() {
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::model::SomeInstance;
        let (_ann, classdef) = classdef_for_fn_ptr_call_test("MismatchedFnPtrInst");
        let inst =
            SomeValue::Instance(SomeInstance::new(Some(classdef), false, Default::default()));
        let unrelated = lltype::LowLevelType::Ptr(Box::new(lltype::Ptr {
            TO: lltype::PtrTarget::Struct(lltype::Struct::gc("UnrelatedFnPtrArg", vec![])),
        }));
        let s_ptr = SomePtr::new(lltype::Ptr {
            TO: lltype::PtrTarget::Func(lltype::FuncType {
                args: vec![unrelated],
                result: lltype::LowLevelType::Void,
            }),
        });
        let args = ArgumentsForTranslation::new(vec![Some(inst)], None, None);
        let err = s_ptr
            .call(&args)
            .expect_err("mismatched instance is not a cast to ARGS[i]");
        assert!(
            err.msg
                .unwrap_or_default()
                .contains("should return a low-level type"),
            "mismatched instance keeps the upstream annotation_to_lltype message"
        );
    }

    #[test]
    fn someptr_call_maps_s_none_to_fn_ptr_ptr_arg() {
        use crate::annotator::argument::ArgumentsForTranslation;
        let s_ptr = SomePtr::new(lltype::Ptr {
            TO: lltype::PtrTarget::Func(lltype::FuncType {
                args: vec![crate::translator::rtyper::rclass::OBJECTPTR.clone()],
                result: lltype::LowLevelType::Void,
            }),
        });
        let args = ArgumentsForTranslation::new(vec![Some(s_none())], None, None);
        s_ptr
            .call(&args)
            .expect("s_None is the null pointer constant of a Ptr ARGS slot");
    }

    #[test]
    fn lltype_to_annotation_maps_scalars_and_ptrs() {
        assert!(matches!(
            lltype_to_annotation(lltype::LowLevelType::Void),
            SomeValue::None_(_)
        ));
        assert!(matches!(
            lltype_to_annotation(lltype::LowLevelType::Signed),
            SomeValue::Integer(_)
        ));
        let SomeValue::Integer(i) = lltype_to_annotation(lltype::LowLevelType::SignedLongLong)
        else {
            panic!("expected SomeInteger");
        };
        assert_eq!(i.knowntype(), KnownType::LongLong);
        let SomeValue::Integer(i) = lltype_to_annotation(lltype::LowLevelType::UnsignedLongLong)
        else {
            panic!("expected SomeInteger");
        };
        assert_eq!(i.knowntype(), KnownType::ULongLong);
        assert!(i.unsigned);
        assert!(matches!(
            lltype_to_annotation(lltype::LowLevelType::Bool),
            SomeValue::Bool(_)
        ));
        assert!(matches!(
            lltype_to_annotation(lltype::LowLevelType::UniChar),
            SomeValue::UnicodeCodePoint(_)
        ));
        assert!(matches!(
            lltype_to_annotation(lltype::LowLevelType::Address),
            SomeValue::Address(_)
        ));

        let ptr = lltype::Ptr {
            TO: lltype::PtrTarget::Func(lltype::FuncType {
                args: vec![],
                result: lltype::LowLevelType::Void,
            }),
        };
        let SomeValue::Ptr(s_ptr) = lltype_to_annotation(ptr.clone()) else {
            panic!("expected SomePtr");
        };
        assert_eq!(s_ptr.ll_ptrtype, ptr);

        let interior = lltype::InteriorPtr {
            PARENTTYPE: Box::new(lltype::LowLevelType::Struct(Box::new(lltype::Struct::new(
                "S",
                vec![("x".into(), lltype::LowLevelType::Signed)],
            )))),
            TO: Box::new(lltype::LowLevelType::Signed),
            offsets: vec![lltype::InteriorOffset::Field("x".into())],
        };
        let SomeValue::InteriorPtr(s_ptr) = lltype_to_annotation(
            lltype::LowLevelType::InteriorPtr(Box::new(interior.clone())),
        ) else {
            panic!("expected SomeInteriorPtr");
        };
        assert_eq!(s_ptr.ll_ptrtype, interior);
    }

    #[test]
    fn lltype_to_annotation_rejects_bare_compounds() {
        let cases = [
            (
                "struct",
                lltype::LowLevelType::Struct(Box::new(lltype::Struct::new(
                    "S",
                    vec![("x".into(), lltype::LowLevelType::Signed)],
                ))),
            ),
            (
                "func",
                lltype::LowLevelType::Func(Box::new(lltype::FuncType {
                    args: vec![],
                    result: lltype::LowLevelType::Void,
                })),
            ),
            (
                "array",
                lltype::LowLevelType::Array(Box::new(lltype::Array::new(
                    lltype::LowLevelType::Signed,
                ))),
            ),
        ];
        for (name, ty) in cases {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = lltype_to_annotation(ty);
            }));
            let Err(payload) = result else {
                panic!("case {name}: expected panic");
            };
            let msg = if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else {
                "assertion failed".to_string()
            };
            assert!(
                msg.contains("must be wrapped in Ptr(T)"),
                "case {name}: {msg}"
            );
        }
    }

    #[test]
    fn some_rust_vec_getattr_setattr_len_mirrors_someptr() {
        use crate::annotator::model::SomeInteger;
        use majit_ir::rvec::VecItemKind;

        let s_vec = SomeRustVec::for_kind(VecItemKind::Int);
        let s_len = {
            let mut s = crate::annotator::model::SomeString::new(false, false);
            s.inner.base.const_box = Some(Constant::new(ConstValue::byte_str("len")));
            SomeValue::String(s)
        };
        let s_cap = {
            let mut s = crate::annotator::model::SomeString::new(false, false);
            s.inner.base.const_box = Some(Constant::new(ConstValue::byte_str("cap")));
            SomeValue::String(s)
        };
        let s_n = SomeValue::Integer(SomeInteger::new_with_knowntype(true, KnownType::Ruint));
        assert!(matches!(
            s_vec.getattr(&s_len).unwrap(),
            SomeValue::Integer(_)
        ));
        assert!(s_vec.setattr(&s_len, &s_n).is_ok());
        let err = s_vec.setattr(&s_cap, &s_n).unwrap_err();
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("no field \"cap\""),
            "got {err:?}"
        );
        let SomeValue::BuiltinMethod(bound) = s_vec
            .getattr(&{
                let mut s = crate::annotator::model::SomeString::new(false, false);
                s.inner.base.const_box = Some(Constant::new(ConstValue::byte_str("append")));
                SomeValue::String(s)
            })
            .unwrap()
        else {
            panic!("append must resolve through find_method");
        };
        assert_eq!(bound.analyser_name, "rustvec_method_append");
    }

    #[test]
    fn ll_to_annotation_maps_low_level_values() {
        assert!(matches!(
            ll_to_annotation(lltype::LowLevelValue::Void),
            SomeValue::None_(_)
        ));
        assert!(matches!(
            ll_to_annotation(lltype::LowLevelValue::Signed(0)),
            SomeValue::Integer(_)
        ));
        assert!(matches!(
            ll_to_annotation(lltype::LowLevelValue::Bool(false)),
            SomeValue::Bool(_)
        ));
        assert!(matches!(
            ll_to_annotation(lltype::LowLevelValue::Char('\0')),
            SomeValue::Char(_)
        ));

        let interior = lltype::_interior_ptr {
            _T: lltype::LowLevelType::Signed,
            _parent: lltype::LowLevelValue::Struct(Box::new(
                lltype::Struct::new("S", vec![("x".into(), lltype::LowLevelType::Signed)])
                    ._container_example(),
            )),
            _offsets: vec![lltype::InteriorOffset::Field("x".into())],
        };
        assert!(matches!(
            ll_to_annotation(lltype::LowLevelValue::InteriorPtr(Box::new(interior))),
            SomeValue::InteriorPtr(_)
        ));
    }

    #[test]
    #[should_panic(expected = "must be wrapped in Ptr(T)")]
    fn ll_to_annotation_rejects_bare_struct_value() {
        let value = lltype::LowLevelValue::Struct(Box::new(
            lltype::Struct::new("S", vec![("x".into(), lltype::LowLevelType::Signed)])
                ._container_example(),
        ));
        let _ = ll_to_annotation(value);
    }
}
