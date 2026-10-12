//! A Rust `alloc::vec::Vec<T>` with one-word items — the raw-flavoured
//! counterpart of `lltypesystem/rlist.py`'s resizable `ListRepr`.
//!
//! A value is a pointer to the raw three-word header, so its lowleveltype is
//! `Ptr(Struct(raw) "RustVec" {ptr, len, cap})` in `majit_ir::rvec` word order
//! and its kind is `int`. The item buffer is a raw `Array(ITEM)` with no
//! length header, for every item kind: the interpreter treats `Vec` contents
//! as untraced and roots them where it allocates, and the lowering keeps that
//! contract rather than giving the buffer GC semantics the interpreter lacks.
//!
//! Every operation calls the `ll_vec_*` helper [`majit_ir::rvec`] names for
//! it and the item kind, the way `ListRepr` calls the `ll_*` helpers of
//! `lltypesystem/rlist.py`.

use std::sync::Arc;

use majit_ir::rvec::{
    VEC_CAP_WORD, VEC_LEN_WORD, VEC_PTR_WORD, VecItemKind, VecOp, vec_helper_path,
};

use crate::flowspace::model::{ConstValue, Hlvalue};
use crate::model::ConcreteType;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    Array, FuncType, GCREF, LowLevelType, Ptr, PtrTarget, Struct, functionptr,
};
use crate::translator::rtyper::rlist::ListIteratorRepr;
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState, inputconst_from_lltype};
use crate::translator::rtyper::rtyper::{ConvertedTo, GenopResult, HighLevelOp, LowLevelOpList};

/// Struct hint marking the `RustVec` header lltype.
pub const RUST_VEC_HINT: &str = "rust_vec";

/// `Ptr(Struct(raw) "RustVec" {...})` over `item`, with the header fields in
/// `VEC_{PTR,LEN,CAP}_WORD` order.
pub fn rust_vec_lltype(item: &LowLevelType) -> LowLevelType {
    let items = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(Array::with_hints(
            item.clone(),
            vec![("nolength".into(), ConstValue::Bool(true))],
        )),
    }));
    let mut fields = [
        (VEC_PTR_WORD, ("ptr".to_string(), items)),
        (VEC_LEN_WORD, ("len".to_string(), LowLevelType::Signed)),
        (VEC_CAP_WORD, ("cap".to_string(), LowLevelType::Signed)),
    ];
    fields.sort_by_key(|(word, _)| *word);
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(Struct::with_hints(
            "RustVec",
            fields.into_iter().map(|(_, field)| field).collect(),
            vec![(RUST_VEC_HINT.into(), ConstValue::Bool(true))],
        )),
    }))
}

/// The item lltype of a [`rust_vec_lltype`] header pointer.
pub fn rust_vec_item_lltype(ty: &LowLevelType) -> Option<&LowLevelType> {
    let LowLevelType::Ptr(ptr) = ty else {
        return None;
    };
    let PtrTarget::Struct(header) = &ptr.TO else {
        return None;
    };
    header._hints.get(RUST_VEC_HINT)?;
    let LowLevelType::Ptr(items) = header._flds.get("ptr")? else {
        return None;
    };
    let PtrTarget::Array(array) = &items.TO else {
        return None;
    };
    Some(&array.OF)
}

/// Register kind of an item lltype.
pub fn vec_item_kind_of(item: &LowLevelType) -> Option<VecItemKind> {
    match crate::model::getkind(item) {
        ConcreteType::Signed => Some(VecItemKind::Int),
        ConcreteType::GcRef => Some(VecItemKind::Ref),
        ConcreteType::Float => Some(VecItemKind::Float),
        ConcreteType::Void | ConcreteType::Unknown => None,
    }
}

/// The item kind of a [`rust_vec_lltype`] header pointer.
pub fn rust_vec_item_kind(ty: &LowLevelType) -> Option<VecItemKind> {
    rust_vec_item_lltype(ty).and_then(vec_item_kind_of)
}

#[derive(Debug)]
pub struct RustVecRepr {
    state: ReprState,
    lltype: LowLevelType,
    item_repr: Arc<dyn Repr>,
    kind: VecItemKind,
}

impl RustVecRepr {
    pub fn new(item_repr: Arc<dyn Repr>) -> Result<Self, TyperError> {
        let item = item_repr.lowleveltype().clone();
        let kind = vec_item_kind_of(&item).ok_or_else(|| {
            TyperError::message(format!(
                "RustVecRepr: item lltype {} is not one word",
                item.short_name()
            ))
        })?;
        Ok(RustVecRepr {
            state: ReprState::new(),
            lltype: rust_vec_lltype(&item),
            item_repr,
            kind,
        })
    }

    pub fn item_kind(&self) -> VecItemKind {
        self.kind
    }

    /// The helper this repr lowers `op` to.
    pub fn helper_path(&self, op: VecOp) -> &'static str {
        vec_helper_path(op, self.kind)
    }

    fn item_lltype(&self) -> &LowLevelType {
        self.item_repr.lowleveltype()
    }

    /// `direct_call(<helper for op>, *args_v)` with the given signature.
    fn call_helper(
        &self,
        hop: &HighLevelOp,
        op: VecOp,
        args_v: Vec<Hlvalue>,
        args: Vec<LowLevelType>,
        result: LowLevelType,
    ) -> RTypeResult {
        genop_helper_call(hop, self.helper_path(op), args_v, args, result)
    }
}

fn genop_helper_call(
    hop: &HighLevelOp,
    path: &str,
    args_v: Vec<Hlvalue>,
    args: Vec<LowLevelType>,
    result: LowLevelType,
) -> RTypeResult {
    let fptr = functionptr(
        FuncType {
            args,
            result: result.clone(),
        },
        path,
        None,
        Some(path.to_string()),
    );
    let fptr_type = LowLevelType::Ptr(Box::new(fptr._TYPE.clone()));
    let c_func = inputconst_from_lltype(&fptr_type, &ConstValue::LLPtr(Box::new(fptr)))?;
    let mut call_args = Vec::with_capacity(args_v.len() + 1);
    call_args.push(Hlvalue::Constant(c_func));
    call_args.extend(args_v);
    Ok(hop.genop("direct_call", call_args, GenopResult::LLType(result)))
}

impl Repr for RustVecRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "RustVecRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::RustVecRepr
    }

    /// Same iterator as a resized list: `ListIteratorRepr` over the header
    /// pointer, reading `len` / `ptr` instead of `length` / `items`.
    #[expect(
        clippy::arc_with_non_send_sync,
        reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
    )]
    fn make_iterator_repr(
        &self,
        variant: &[String],
        foldable: bool,
    ) -> Result<Arc<dyn Repr>, TyperError> {
        if !variant.is_empty() {
            return Err(TyperError::missing_rtype_operation(
                "RustVecRepr.make_iterator_repr: non-default variant (reversed) deferred",
            ));
        }
        Ok(Arc::new(ListIteratorRepr::new_with_header_fields(
            self.lltype.clone(),
            self.item_repr.clone(),
            self.item_repr.clone(),
            false,
            foldable,
            "len",
            "ptr",
        )?))
    }

    /// `ll_length`.
    fn rtype_len(&self, hop: &HighLevelOp) -> RTypeResult {
        let v = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        self.call_helper(
            hop,
            VecOp::Length,
            v,
            vec![self.lltype.clone()],
            LowLevelType::Signed,
        )
    }

    /// `ll_getitem_fast`: the index is in bounds.
    fn rtype_getitem(&self, hop: &HighLevelOp) -> RTypeResult {
        let v = hop.inputargs(vec![
            ConvertedTo::Repr(self),
            ConvertedTo::LowLevelType(&LowLevelType::Signed),
        ])?;
        hop.exception_cannot_occur()?;
        self.call_helper(
            hop,
            VecOp::GetItem,
            v,
            vec![self.lltype.clone(), LowLevelType::Signed],
            self.item_lltype().clone(),
        )
    }

    /// `ll_setitem_fast`: the index is in bounds.
    fn rtype_setitem(&self, hop: &HighLevelOp) -> RTypeResult {
        let v = hop.inputargs(vec![
            ConvertedTo::Repr(self),
            ConvertedTo::LowLevelType(&LowLevelType::Signed),
            ConvertedTo::Repr(self.item_repr.as_ref()),
        ])?;
        hop.exception_cannot_occur()?;
        self.call_helper(
            hop,
            VecOp::SetItem,
            v,
            vec![
                self.lltype.clone(),
                LowLevelType::Signed,
                self.item_lltype().clone(),
            ],
            LowLevelType::Void,
        )
    }

    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        match method_name {
            // `ll_append`.
            "append" => {
                let v = hop.inputargs(vec![
                    ConvertedTo::Repr(self),
                    ConvertedTo::Repr(self.item_repr.as_ref()),
                ])?;
                hop.exception_cannot_occur()?;
                self.call_helper(
                    hop,
                    VecOp::Append,
                    v,
                    vec![self.lltype.clone(), self.item_lltype().clone()],
                    LowLevelType::Void,
                )
            }
            // `ll_reverse`.
            "reverse" => {
                let v = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
                hop.exception_cannot_occur()?;
                self.call_helper(
                    hop,
                    VecOp::Reverse,
                    v,
                    vec![self.lltype.clone()],
                    LowLevelType::Void,
                )
            }
            // `ll_items`: the item pointer, a raw address word.
            "items" => {
                let v = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
                hop.exception_cannot_occur()?;
                self.call_helper(
                    hop,
                    VecOp::Items,
                    v,
                    vec![self.lltype.clone()],
                    LowLevelType::Unsigned,
                )
            }
            // `ll_extend` from the `(items, length)` pair of a slice.
            "extend_from_slice" => {
                let v = hop.inputargs(vec![
                    ConvertedTo::Repr(self),
                    ConvertedTo::LowLevelType(&LowLevelType::Unsigned),
                    ConvertedTo::LowLevelType(&LowLevelType::Unsigned),
                ])?;
                hop.exception_cannot_occur()?;
                self.call_helper(
                    hop,
                    VecOp::ExtendFromSlice,
                    v,
                    vec![
                        self.lltype.clone(),
                        LowLevelType::Unsigned,
                        LowLevelType::Unsigned,
                    ],
                    LowLevelType::Void,
                )
            }
            // `rlist.ll_extend(l1, l2)`: `l2` is the object GcArray word.
            // Only `_r`: a one-word GcArray source is an object slice.
            "extend" => {
                if self.kind != VecItemKind::Ref {
                    return Err(self.missing_rtype_operation("method_extend"));
                }
                let v = hop.inputargs(vec![
                    ConvertedTo::Repr(self),
                    ConvertedTo::LowLevelType(&*GCREF),
                ])?;
                hop.exception_cannot_occur()?;
                self.call_helper(
                    hop,
                    VecOp::Extend,
                    v,
                    vec![self.lltype.clone(), GCREF.clone()],
                    LowLevelType::Void,
                )
            }
            // `lltype.free(l, flavor='raw')` of the header and its buffer.
            "free" => {
                let v = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
                hop.exception_cannot_occur()?;
                self.call_helper(
                    hop,
                    VecOp::Free,
                    v,
                    vec![self.lltype.clone()],
                    LowLevelType::Void,
                )
            }
            _ => Err(self.missing_rtype_operation(&format!("method_{method_name}"))),
        }
    }
}

/// `r_uint(vec)` of a `Vec` header is the header address as an integer.
///
/// `rbuiltin.py rtype_cast_ptr_to_int` always yields `Signed`. An
/// `Unsigned` target then runs `rint.py IntegerRepr.convert_from_to`,
/// which emits `cast_int_to_uint`. A `Signed` target returns the
/// `cast_ptr_to_int` result directly.
pub fn pair_rustvec_integer_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    let _ = r_from;
    match r_to.lowleveltype() {
        LowLevelType::Unsigned | LowLevelType::Signed => {}
        _ => return Ok(None),
    }
    let v_signed = llops
        .genop(
            "cast_ptr_to_int",
            vec![v.clone()],
            GenopResult::LLType(LowLevelType::Signed),
        )
        .map(Hlvalue::Variable)
        .ok_or_else(|| {
            TyperError::message(
                "pair_rustvec_integer_convert_from_to: cast_ptr_to_int returned void",
            )
        })?;
    if r_to.lowleveltype() == &LowLevelType::Signed {
        return Ok(Some(v_signed));
    }
    super::rint::pair_integer_integer_convert_from_to(
        super::rint::signed_repr().as_ref(),
        r_to,
        &v_signed,
        llops,
    )
}

/// `newrustvec(kind)` / `newrustvec(kind, lengthhint)` — `ll_newemptylist` /
/// `ll_newlist_hint` for the result repr.
pub fn rtype_newrustvec(hop: &HighLevelOp) -> RTypeResult {
    let r_result = hop
        .r_result
        .borrow()
        .clone()
        .ok_or_else(|| TyperError::message("rtype_newrustvec: r_result missing"))?;
    if r_result.repr_class_id() != super::pairtype::ReprClassId::RustVecRepr {
        return Err(TyperError::message(
            "rtype_newrustvec: hop.r_result is not a RustVecRepr",
        ));
    }
    let lltype = r_result.lowleveltype().clone();
    let kind = rust_vec_item_kind(&lltype)
        .ok_or_else(|| TyperError::message("rtype_newrustvec: result is not a RustVec header"))?;
    hop.exception_cannot_occur()?;
    // `newrustvec(kind, count, item)` — `rlist.py ll_alloc_and_set`.
    if hop.nb_args() > 2 {
        let item_ll = rust_vec_item_lltype(&lltype)
            .cloned()
            .ok_or_else(|| TyperError::message("rtype_newrustvec: alloc_and_set item lltype"))?;
        let v_count = hop.inputarg(ConvertedTo::LowLevelType(&LowLevelType::Signed), 1)?;
        let v_item = hop.inputarg(ConvertedTo::LowLevelType(&item_ll), 2)?;
        return genop_helper_call(
            hop,
            vec_helper_path(VecOp::AllocAndSet, kind),
            vec![v_count, v_item],
            vec![LowLevelType::Signed, item_ll],
            lltype,
        );
    }
    if hop.nb_args() > 1 {
        let v_hint = hop.inputarg(ConvertedTo::LowLevelType(&LowLevelType::Signed), 1)?;
        genop_helper_call(
            hop,
            vec_helper_path(VecOp::NewHint, kind),
            vec![v_hint],
            vec![LowLevelType::Signed],
            lltype,
        )
    } else {
        genop_helper_call(
            hop,
            vec_helper_path(VecOp::NewEmpty, kind),
            vec![],
            vec![],
            lltype,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_lltype_is_a_raw_int_kind_pointer() {
        for (item, kind) in [
            (LowLevelType::Signed, VecItemKind::Int),
            (LowLevelType::Float, VecItemKind::Float),
        ] {
            let ty = rust_vec_lltype(&item);
            assert_eq!(crate::model::getkind(&ty), ConcreteType::Signed);
            assert_eq!(rust_vec_item_lltype(&ty), Some(&item));
            assert_eq!(rust_vec_item_kind(&ty), Some(kind));
            let LowLevelType::Ptr(ptr) = &ty else {
                unreachable!()
            };
            let PtrTarget::Struct(header) = &ptr.TO else {
                unreachable!()
            };
            let names: Vec<&str> = header._names.iter().map(String::as_str).collect();
            let mut expected = [
                (VEC_PTR_WORD, "ptr"),
                (VEC_LEN_WORD, "len"),
                (VEC_CAP_WORD, "cap"),
            ];
            expected.sort_by_key(|(word, _)| *word);
            assert_eq!(names, expected.map(|(_, n)| n));
        }
        assert_eq!(rust_vec_item_kind(&LowLevelType::Signed), None);
    }

    fn setup_rtyper() -> (
        std::rc::Rc<crate::annotator::annrpython::RPythonAnnotator>,
        std::rc::Rc<crate::translator::rtyper::rtyper::RPythonTyper>,
    ) {
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(crate::translator::rtyper::rtyper::RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        (ann, rtyper)
    }

    fn rust_vec_repr(
        rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
        kind: VecItemKind,
    ) -> Arc<dyn Repr> {
        use crate::annotator::model::{SomeRustVec, SomeValue};
        rtyper
            .getrepr(&SomeValue::RustVec(SomeRustVec::for_kind(kind)))
            .unwrap_or_else(|err| panic!("getrepr RustVec {kind:?}: {err:?}"))
    }

    fn assert_only_direct_call_to(
        llops: &std::rc::Rc<std::cell::RefCell<crate::translator::rtyper::rtyper::LowLevelOpList>>,
        path: &str,
    ) {
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1, "expected one direct_call");
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr, got {:?}", ops.ops[0].args[0]);
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains(&format!("\"{path}\"")),
            "expected funcptr {path} in {dbg}"
        );
    }

    /// Every item kind has a repr whose lowleveltype is the int-kind header
    /// and whose helper choice is the `majit_ir::rvec` table's.
    #[test]
    fn repr_helper_choice_is_the_table() {
        let (_ann, rtyper) = setup_rtyper();
        for kind in VecItemKind::ALL {
            let r = rust_vec_repr(&rtyper, kind);
            assert_eq!(
                r.repr_class_id(),
                super::super::pairtype::ReprClassId::RustVecRepr
            );
            assert_eq!(
                crate::model::getkind(r.lowleveltype()),
                ConcreteType::Signed
            );
            assert_eq!(rust_vec_item_kind(r.lowleveltype()), Some(kind));
            let item = rust_vec_item_lltype(r.lowleveltype()).expect("header item lltype");
            assert_eq!(vec_item_kind_of(item), Some(kind));
            let s_item = *crate::annotator::model::SomeRustVec::for_kind(kind).s_item;
            let r =
                RustVecRepr::new(rtyper.getrepr(&s_item).expect("item repr")).expect("RustVecRepr");
            assert_eq!(r.item_kind(), kind);
            for op in VecOp::ALL {
                assert_eq!(r.helper_path(op), vec_helper_path(op, kind));
            }
        }
    }

    /// `newrustvec(kind)` and `newrustvec(kind, hint)` go through the
    /// rtyper's `translate_operation` to `ll_vec_newemptylist_*` /
    /// `ll_vec_newlist_hint_*`.
    #[test]
    fn translate_operation_newrustvec_calls_the_ctor_helpers() {
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        let (_ann, rtyper) = setup_rtyper();
        for kind in VecItemKind::ALL {
            let r = rust_vec_repr(&rtyper, kind);
            for with_hint in [false, true] {
                let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
                    rtyper.clone(),
                    None,
                )));
                let v_result = Variable::new();
                v_result.set_concretetype(Some(r.lowleveltype().clone()));
                let kind_letter = Hlvalue::Constant(crate::flowspace::model::Constant::new(
                    ConstValue::byte_str(kind.kind_char().to_string()),
                ));
                let mut args = vec![kind_letter];
                if with_hint {
                    args.push(crate::translator::rtyper::rtyper::constant_with_lltype(
                        ConstValue::Int(8),
                        LowLevelType::Signed,
                    ));
                }
                let hop = HighLevelOp::new(
                    rtyper.clone(),
                    SpaceOperation::new(
                        "newrustvec".to_string(),
                        args,
                        Hlvalue::Variable(v_result),
                    ),
                    Vec::new(),
                    llops.clone(),
                );
                hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
                *hop.r_result.borrow_mut() = Some(r.clone());
                rtyper
                    .translate_operation(&hop)
                    .unwrap_or_else(|err| panic!("translate_operation newrustvec: {err:?}"));
                let op = if with_hint {
                    VecOp::NewHint
                } else {
                    VecOp::NewEmpty
                };
                assert_only_direct_call_to(&llops, vec_helper_path(op, kind));
            }
        }
    }

    /// `rbuiltin.py rtype_cast_ptr_to_int` always yields `Signed`.
    /// `rint.py IntegerRepr.convert_from_to` then emits `cast_int_to_uint`
    /// for an `Unsigned` target.
    #[test]
    fn pair_rustvec_integer_convert_from_to_signed_then_uint() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rint::{signed_repr, unsigned_repr};
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let (_ann, rtyper) = setup_rtyper();
        let r_from = rust_vec_repr(&rtyper, VecItemKind::Int);
        let v = Variable::new();
        v.set_concretetype(Some(r_from.lowleveltype().clone()));
        let v = Hlvalue::Variable(v);

        let mut signed_ops = LowLevelOpList::new(rtyper.clone(), None);
        let r_signed = signed_repr();
        let signed = pair_rustvec_integer_convert_from_to(
            r_from.as_ref(),
            r_signed.as_ref(),
            &v,
            &mut signed_ops,
        )
        .expect("Signed conversion rtypes")
        .expect("Signed conversion returns a value");
        assert_eq!(signed_ops.ops.len(), 1);
        assert_eq!(signed_ops.ops[0].opname, "cast_ptr_to_int");
        match &signed_ops.ops[0].result {
            Hlvalue::Variable(var) => {
                assert_eq!(var.concretetype(), Some(LowLevelType::Signed));
            }
            other => panic!("expected Variable result, got {other:?}"),
        }
        match &signed {
            Hlvalue::Variable(var) => {
                assert_eq!(var.concretetype(), Some(LowLevelType::Signed));
            }
            other => panic!("expected Variable, got {other:?}"),
        }

        let mut unsigned_ops = LowLevelOpList::new(rtyper, None);
        let r_unsigned = unsigned_repr();
        let unsigned = pair_rustvec_integer_convert_from_to(
            r_from.as_ref(),
            r_unsigned.as_ref(),
            &v,
            &mut unsigned_ops,
        )
        .expect("Unsigned conversion rtypes")
        .expect("Unsigned conversion returns a value");
        assert_eq!(unsigned_ops.ops.len(), 2);
        assert_eq!(unsigned_ops.ops[0].opname, "cast_ptr_to_int");
        assert_eq!(unsigned_ops.ops[1].opname, "cast_int_to_uint");
        match &unsigned_ops.ops[0].result {
            Hlvalue::Variable(var) => {
                assert_eq!(var.concretetype(), Some(LowLevelType::Signed));
            }
            other => panic!("expected Variable result, got {other:?}"),
        }
        match &unsigned_ops.ops[1].result {
            Hlvalue::Variable(var) => {
                assert_eq!(var.concretetype(), Some(LowLevelType::Unsigned));
            }
            other => panic!("expected Variable result, got {other:?}"),
        }
        match &unsigned {
            Hlvalue::Variable(var) => {
                assert_eq!(var.concretetype(), Some(LowLevelType::Unsigned));
            }
            other => panic!("expected Variable, got {other:?}"),
        }
    }
}
