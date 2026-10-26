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

use crate::annotator::model::SomeValue;
use crate::flowspace::model::{ConstValue, Hlvalue};
use crate::model::ConcreteType;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    Array, FuncType, GCREF, LowLevelType, Ptr, PtrTarget, Struct, functionptr,
};
use crate::translator::rtyper::rlist::{
    ListIteratorRepr, ListLayout, rtype_getslice_via_ll_listslice,
};
use crate::translator::rtyper::rmodel::{
    RTypeResult, Repr, ReprState, hop_arg_const_string, inputconst_from_lltype,
};
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

    /// Item repr `rlist.py ListRepr.item_repr` carries for `hop.inputargs`.
    pub fn item_repr(&self) -> &Arc<dyn Repr> {
        &self.item_repr
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

pub(crate) fn genop_helper_call(
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

    /// `PtrRepr.rtype_getattr` (rptr.py): a bound method forwards the
    /// receiver; `len` is `getfield` of the header word `rtype_setattr`
    /// writes.
    fn rtype_getattr(&self, hop: &HighLevelOp) -> RTypeResult {
        let attr = hop_arg_const_string(hop, 1)?;
        if matches!(&*hop.s_result.borrow(), Some(SomeValue::BuiltinMethod(_))) {
            return hop.inputarg(self, 0).map(Some);
        }
        if attr != "len" {
            return Err(self.missing_rtype_operation(&format!("getattr_{attr}")));
        }
        let void = LowLevelType::Void;
        let vlist = hop.inputargs(vec![ConvertedTo::from(self), ConvertedTo::from(&void)])?;
        Ok(hop.genop("getfield", vlist, GenopResult::LLType(LowLevelType::Signed)))
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

    /// RPython `AbstractBaseListRepr.rtype_getslice(r_lst, hop)` (`rlist.py`)
    /// via `rtype_getslice_via_ll_listslice`. The source is the raw
    /// `{ptr, len, cap}` header (`ListLayout::RustVec`); the result is a
    /// fresh `RustVecRepr` of the same item (`hop.r_result`).
    fn rtype_getslice(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_lst = hop.inputarg(ConvertedTo::Repr(self), 0)?;
        rtype_getslice_via_ll_listslice(
            hop,
            ListLayout::RustVec,
            self.lltype.clone(),
            self.item_lltype().clone(),
            v_lst,
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

    /// `rlist.py` `_ll_list_resize_really`: `l.length = newsize`, and
    /// `ll_newlist` after the hint allocation. `Vec::set_len` is that store
    /// on the raw header's `len` word; there is no `ll_vec_set_len` helper.
    fn rtype_setattr(&self, hop: &HighLevelOp) -> RTypeResult {
        let attr = hop_arg_const_string(hop, 1)?;
        if attr != "len" {
            return Err(self.missing_rtype_operation(&format!("setattr_{attr}")));
        }
        hop.exception_cannot_occur()?;
        let void = LowLevelType::Void;
        let signed = LowLevelType::Signed;
        let vlist = hop.inputargs(vec![
            ConvertedTo::from(self),
            ConvertedTo::from(&void),
            ConvertedTo::from(&signed),
        ])?;
        hop.genop("setfield", vlist, GenopResult::Void);
        Ok(None)
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
                    ConvertedTo::LowLevelType(&GCREF),
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

/// `newrustvec(ITEM)` / `newrustvec(ITEM, lengthhint)` — `ll_newemptylist` /
/// `ll_newlist_hint` for the result repr.
///
/// Argument 0 is the constant item lowleveltype (`lltype.py` `malloc` /
/// `ann_malloc`). The result lowleveltype stays `rust_vec_lltype(ITEM)` on
/// `hop.r_result`; this does not re-read the constant. `inputarg` starts at
/// index 1.
///
/// `newrustvec(ITEM, count, item)` follows `rtype_alloc_and_set`: the fill
/// value converts through the result repr's `item_repr`. `ListRepr._setup_repr`
/// and `FixedSizeListRepr._setup_repr` keep that field from `s_item` when it
/// is missing; [`RustVecRepr`] stores the same field from `SomeRustVec.s_item`
/// (`rtyper_makerepr`). An instance item then takes
/// `pair_instance_instance_convert_from_to` (upcast to the root `item_repr`)
/// instead of `getprimitiverepr` on the item pointer.
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
    // `newrustvec(ITEM, count, item)` — `rlist.py ll_alloc_and_set`.
    // `hop.inputargs(Signed, r_list.item_repr)`: a GC item has no primitive
    // repr (`rtyper.py getprimitiverepr`), so the fill must convert through
    // the vec's item repr the way `rtype_alloc_and_set` does.
    if hop.nb_args() > 2 {
        let item_ll = rust_vec_item_lltype(&lltype)
            .cloned()
            .ok_or_else(|| TyperError::message("rtype_newrustvec: alloc_and_set item lltype"))?;
        let any_r: &dyn std::any::Any = r_result.as_ref();
        let rust_vec = any_r.downcast_ref::<RustVecRepr>().ok_or_else(|| {
            TyperError::message("rtype_newrustvec: r_result is not a RustVecRepr")
        })?;
        let v_count = hop.inputarg(ConvertedTo::LowLevelType(&LowLevelType::Signed), 1)?;
        let v_item = hop.inputarg(ConvertedTo::Repr(rust_vec.item_repr().as_ref()), 2)?;
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
    use crate::annotator::model::SomeValue;
    use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

    #[test]
    fn header_lltype_is_a_raw_int_kind_pointer() {
        for (item, kind) in [
            (LowLevelType::Signed, VecItemKind::Int),
            (LowLevelType::Unsigned, VecItemKind::Int),
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

    /// `Vec<usize>` / `Vec<u64>` annotate as an unsigned integer item and
    /// lower to `Array(Unsigned)` (`rint.py` `unsigned_repr`). `Vec<isize>`
    /// / `Vec<i64>` stay `Array(Signed)`. Both register kinds are `Int`
    /// (`getkind` of `Unsigned` is the int kind). The existing
    /// `SomeInteger` union makes the result unsigned when the signed side
    /// is nonneg (`binaryop.py` `pairtype(SomeInteger, SomeInteger).union`).
    #[test]
    fn vec_usize_annotation_lowers_to_array_unsigned() {
        use crate::annotator::model::{SomeInteger, SomeRustVec, SomeValue};
        let (ann, rtyper) = setup_rtyper();
        let mut cases = vec![
            ("Vec<usize>", true, LowLevelType::Unsigned),
            ("*mut Vec<usize>", true, LowLevelType::Unsigned),
            ("Vec<isize>", false, LowLevelType::Signed),
            ("*mut Vec<isize>", false, LowLevelType::Signed),
        ];
        if crate::layout::target_word_size() == 8 {
            cases.push(("Vec<u64>", true, LowLevelType::Unsigned));
            cases.push(("Vec<i64>", false, LowLevelType::Signed));
        }
        for (spelling, unsigned, item_ll) in cases {
            let Some(SomeValue::RustVec(s_vec)) = ann.bookkeeper.project_rust_vec(spelling) else {
                panic!("{spelling} must project to SomeRustVec");
            };
            let SomeValue::Integer(s_int) = &*s_vec.s_item else {
                panic!(
                    "{spelling} item must be SomeInteger, got {:?}",
                    s_vec.s_item
                );
            };
            assert_eq!(s_int.unsigned, unsigned, "{spelling}");
            assert_eq!(s_vec.item_kind(), Some(VecItemKind::Int), "{spelling}");
            let repr = rtyper
                .getrepr(&SomeValue::RustVec(s_vec))
                .unwrap_or_else(|err| panic!("getrepr {spelling}: {err:?}"));
            let item = rust_vec_item_lltype(repr.lowleveltype()).expect("header item lltype");
            assert_eq!(item, &item_ll, "{spelling}");
            assert_eq!(vec_item_kind_of(item), Some(VecItemKind::Int), "{spelling}");
        }
        let unsigned = SomeValue::RustVec(SomeRustVec::new(SomeValue::Integer(SomeInteger::new(
            false, true,
        ))));
        let signed_nonneg = SomeValue::RustVec(SomeRustVec::new(SomeValue::Integer(
            SomeInteger::new(true, false),
        )));
        let SomeValue::RustVec(joined) =
            crate::annotator::model::union(&unsigned, &signed_nonneg).expect("RustVec union")
        else {
            panic!("RustVec ∪ RustVec must stay a RustVec");
        };
        let SomeValue::Integer(s_int) = &*joined.s_item else {
            panic!("joined item must be SomeInteger");
        };
        assert!(s_int.unsigned);
        assert!(s_int.nonneg);
    }

    /// `newrustvec` annotates from the constant item lowleveltype
    /// (`lltype.py` `malloc` / `ann_malloc`, `llannotation.py`
    /// `lltype_to_annotation`). A stamped result header is not an input:
    /// `Unsigned` stays unsigned even when the result concretetype is a
    /// signed header.
    #[test]
    fn newrustvec_item_lltype_annotates_integer_signedness() {
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::{Constant, Hlvalue, Variable};
        use crate::flowspace::operation::{HLOperation, OpKind};
        let (ann, _rtyper) = setup_rtyper();
        let item_const = |item: LowLevelType| {
            Hlvalue::Constant(Constant::new(ConstValue::LowLevelType(Box::new(item))))
        };
        let signed_op = HLOperation {
            kind: OpKind::NewRustVec,
            args: vec![item_const(LowLevelType::Signed)],
            result: Variable::new(),
            offset: -1,
        };
        let Some(SomeValue::RustVec(s_vec)) = signed_op.consider(&ann).expect("consider signed")
        else {
            panic!("newrustvec must annotate SomeRustVec");
        };
        let SomeValue::Integer(s_int) = &*s_vec.s_item else {
            panic!("item must be SomeInteger");
        };
        assert!(!s_int.unsigned);
        assert_eq!(s_vec.item_kind(), Some(VecItemKind::Int));

        let unsigned_result = Variable::new();
        unsigned_result.set_concretetype(Some(rust_vec_lltype(&LowLevelType::Signed)));
        let unsigned_op = HLOperation {
            kind: OpKind::NewRustVec,
            args: vec![item_const(LowLevelType::Unsigned)],
            result: unsigned_result,
            offset: -1,
        };
        let Some(SomeValue::RustVec(s_vec)) =
            unsigned_op.consider(&ann).expect("consider unsigned")
        else {
            panic!("newrustvec must annotate SomeRustVec");
        };
        let SomeValue::Integer(s_int) = &*s_vec.s_item else {
            panic!("item must be SomeInteger");
        };
        assert!(s_int.unsigned);
        assert_eq!(s_vec.item_kind(), Some(VecItemKind::Int));
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

    /// `newrustvec(ITEM)` and `newrustvec(ITEM, hint)` go through the
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
                let item = rust_vec_item_lltype(r.lowleveltype())
                    .expect("header item lltype")
                    .clone();
                let item_const = Hlvalue::Constant(crate::flowspace::model::Constant::new(
                    ConstValue::LowLevelType(Box::new(item)),
                ));
                let mut args = vec![item_const];
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

    /// `newrustvec(ITEM, count, item)` converts the fill value through the
    /// result `RustVecRepr.item_repr` (`rtype_alloc_and_set`). A classdef-less
    /// `SomeRustVec.s_item` is the root `InstanceRepr`; a concrete instance
    /// argument upcasts with `pair_instance_instance_convert_from_to`. `Int`
    /// and `Float` items stay on their own item repr.
    #[test]
    fn translate_operation_newrustvec_alloc_and_set_converts_through_item_repr() {
        use crate::annotator::classdesc::ClassDef;
        use crate::annotator::model::{SomeInstance, SomeRustVec, SomeValue};
        use crate::translator::rtyper::rclass::InstanceRepr;
        use std::collections::BTreeMap;

        let (_ann, rtyper) = setup_rtyper();
        let r_signed = rtyper
            .getprimitiverepr(&LowLevelType::Signed)
            .expect("Signed primitive repr");

        // Result `s_item` is a classdef-less instance. Its pointer has no
        // primitive repr, so `inputarg(LowLevelType(item_ll))` cannot rtype it.
        let s_item = SomeValue::Instance(SomeInstance::new(None, false, BTreeMap::new()));
        let r_vec = rtyper
            .getrepr(&SomeValue::RustVec(SomeRustVec::new(s_item)))
            .expect("classdef-less RustVec repr");
        let item_repr = {
            let any_r: &dyn std::any::Any = r_vec.as_ref();
            any_r
                .downcast_ref::<RustVecRepr>()
                .expect("RustVecRepr")
                .item_repr
                .clone()
        };
        let item_any: &dyn std::any::Any = item_repr.as_ref();
        let root = item_any
            .downcast_ref::<InstanceRepr>()
            .expect("classdef-less s_item is an InstanceRepr");
        assert!(root.classdef().is_none());
        let item_ll = item_repr.lowleveltype().clone();
        assert!(
            rtyper.getprimitiverepr(&item_ll).is_err(),
            "root instance pointer must not have a primitive repr"
        );

        let classdef = ClassDef::new_standalone("pyobject::PyObject", None);
        let r_src = rtyper
            .getrepr(&SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                true,
                BTreeMap::new(),
            )))
            .expect("PyObject instance repr");
        let llops = rtype_alloc_and_set(&rtyper, r_vec, r_signed.clone(), r_src);
        let ops = llops.borrow();
        assert!(
            ops.ops.iter().any(|op| op.opname == "cast_pointer"),
            "instance item must upcast to the root item repr, got {:?}",
            ops.ops.iter().map(|op| &op.opname).collect::<Vec<_>>()
        );
        let direct = ops
            .ops
            .iter()
            .find(|op| op.opname == "direct_call")
            .expect("alloc_and_set direct_call");
        let Hlvalue::Constant(func) = &direct.args[0] else {
            panic!("expected Constant funcptr, got {:?}", direct.args[0]);
        };
        let dbg = format!("{:?}", func.value);
        let path = vec_helper_path(VecOp::AllocAndSet, VecItemKind::Ref);
        assert!(
            dbg.contains(&format!("\"{path}\"")),
            "expected funcptr {path} in {dbg}"
        );
        // The fill argument is the upcast, whose concretetype is `item_repr`.
        match &direct.args[2] {
            Hlvalue::Variable(var) => {
                assert_eq!(var.concretetype().as_ref(), Some(&item_ll));
            }
            other => panic!("expected upcast Variable, got {other:?}"),
        }
        drop(ops);

        for kind in [VecItemKind::Int, VecItemKind::Float] {
            let r_vec = rust_vec_repr(&rtyper, kind);
            let r_item = {
                let any_r: &dyn std::any::Any = r_vec.as_ref();
                any_r
                    .downcast_ref::<RustVecRepr>()
                    .expect("RustVecRepr")
                    .item_repr
                    .clone()
            };
            let llops = rtype_alloc_and_set(&rtyper, r_vec, r_signed.clone(), r_item);
            assert_only_direct_call_to(&llops, vec_helper_path(VecOp::AllocAndSet, kind));
        }
    }

    /// Build `newrustvec(ITEM, count, item)` and rtype it. `r_item` is the
    /// source repr of the fill value; the result repr supplies `item_repr`.
    fn rtype_alloc_and_set(
        rtyper: &std::rc::Rc<crate::translator::rtyper::rtyper::RPythonTyper>,
        r_vec: Arc<dyn Repr>,
        r_count: Arc<dyn Repr>,
        r_item: Arc<dyn Repr>,
    ) -> std::rc::Rc<std::cell::RefCell<crate::translator::rtyper::rtyper::LowLevelOpList>> {
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(r_vec.lowleveltype().clone()));
        let item = rust_vec_item_lltype(r_vec.lowleveltype())
            .cloned()
            .expect("header item lltype");
        let item_const = Hlvalue::Constant(crate::flowspace::model::Constant::new(
            ConstValue::LowLevelType(Box::new(item)),
        ));
        let v_count = Variable::new();
        v_count.set_concretetype(Some(LowLevelType::Signed));
        let v_item = Variable::new();
        v_item.set_concretetype(Some(r_item.lowleveltype().clone()));
        let v_count_h = Hlvalue::Variable(v_count);
        let v_item_h = Hlvalue::Variable(v_item);
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "newrustvec".to_string(),
                vec![item_const.clone(), v_count_h.clone(), v_item_h.clone()],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v
            .borrow_mut()
            .extend([item_const, v_count_h, v_item_h]);
        hop.args_s.borrow_mut().extend([
            SomeValue::Impossible,
            SomeValue::Impossible,
            SomeValue::Impossible,
        ]);
        hop.args_r
            .borrow_mut()
            .extend([None, Some(r_count), Some(r_item)]);
        *hop.r_result.borrow_mut() = Some(r_vec);
        rtyper
            .translate_operation(&hop)
            .unwrap_or_else(|err| panic!("translate_operation newrustvec alloc_and_set: {err:?}"));
        llops
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

    fn rust_vec_hop(
        rtyper: &std::rc::Rc<crate::translator::rtyper::rtyper::RPythonTyper>,
        opname: &str,
        args: Vec<Hlvalue>,
    ) -> (
        HighLevelOp,
        std::rc::Rc<std::cell::RefCell<LowLevelOpList>>,
        Arc<dyn Repr>,
    ) {
        use crate::flowspace::model::{SpaceOperation, Variable};
        let r = rust_vec_repr(rtyper, VecItemKind::Int);
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                opname.to_string(),
                args.clone(),
                Hlvalue::Variable(Variable::new()),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(args);
        (hop, llops, r)
    }

    fn const_string_annotation(value: &str) -> SomeValue {
        use crate::annotator::model::SomeString;
        use crate::flowspace::model::Constant;
        let mut s_attr = SomeString::new(false, false);
        s_attr.inner.base.const_box = Some(Constant::new(ConstValue::byte_str(value)));
        SomeValue::String(s_attr)
    }

    /// `PtrRepr.rtype_setattr` (rptr.py): `setfield` of the header `len`.
    #[test]
    fn rtype_setattr_len_emits_setfield() {
        use crate::annotator::model::{SomeInteger, SomeRustVec};
        use crate::flowspace::model::{Constant, Variable};
        use crate::translator::rtyper::rint::signed_repr;

        let (_ann, rtyper) = setup_rtyper();
        let r_signed = signed_repr();
        let v_vec = Variable::new();
        let r_vec_ty = rust_vec_repr(&rtyper, VecItemKind::Int);
        v_vec.set_concretetype(Some(r_vec_ty.lowleveltype().clone()));
        let v_n = Variable::new();
        v_n.set_concretetype(Some(LowLevelType::Signed));
        let (hop, llops, r) = rust_vec_hop(
            &rtyper,
            "setattr",
            vec![
                Hlvalue::Variable(v_vec),
                Hlvalue::Constant(Constant::with_concretetype(
                    ConstValue::byte_str("len"),
                    LowLevelType::Void,
                )),
                Hlvalue::Variable(v_n),
            ],
        );
        hop.args_s.borrow_mut().extend([
            SomeValue::RustVec(SomeRustVec::for_kind(VecItemKind::Int)),
            const_string_annotation("len"),
            SomeValue::Integer(SomeInteger::new(true, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(r.clone()),
            Some(crate::translator::rtyper::rmodel::impossible_repr() as Arc<dyn Repr>),
            Some(r_signed as Arc<dyn Repr>),
        ]);
        assert!(r.rtype_setattr(&hop).unwrap().is_none());
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "setfield");
        assert_eq!(ops.ops[0].args.len(), 3);
    }

    /// `PtrRepr.rtype_getattr` (rptr.py): `getfield` of the header `len`.
    #[test]
    fn rtype_getattr_len_emits_getfield() {
        use crate::annotator::model::{SomeInteger, SomeRustVec};
        use crate::flowspace::model::{Constant, Variable};
        use crate::translator::rtyper::rint::signed_repr;

        let (_ann, rtyper) = setup_rtyper();
        let r_signed = signed_repr();
        let v_vec = Variable::new();
        let r_vec_ty = rust_vec_repr(&rtyper, VecItemKind::Int);
        v_vec.set_concretetype(Some(r_vec_ty.lowleveltype().clone()));
        let (hop, llops, r) = rust_vec_hop(
            &rtyper,
            "getattr",
            vec![
                Hlvalue::Variable(v_vec),
                Hlvalue::Constant(Constant::with_concretetype(
                    ConstValue::byte_str("len"),
                    LowLevelType::Void,
                )),
            ],
        );
        hop.args_s.borrow_mut().extend([
            SomeValue::RustVec(SomeRustVec::for_kind(VecItemKind::Int)),
            const_string_annotation("len"),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(r.clone()),
            Some(crate::translator::rtyper::rmodel::impossible_repr() as Arc<dyn Repr>),
        ]);
        *hop.s_result.borrow_mut() = Some(SomeValue::Integer(SomeInteger::new(false, false)));
        *hop.r_result.borrow_mut() = Some(r_signed);
        let result = r.rtype_getattr(&hop).unwrap().unwrap();
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "getfield");
        assert_eq!(ops.ops[0].args.len(), 2);
    }

    /// `v[start:stop]` on a `Vec<PyObjectRef>` (`VecItemKind::Ref`) with
    /// non-constant nonneg bounds decomposes to `"startstop"` and lowers
    /// to `gendirectcall(ll_listslice_startstop, …)` (`rlist.py`
    /// `AbstractBaseListRepr.rtype_getslice`).
    #[test]
    fn rtype_getslice_ref_startstop_emits_ll_listslice_startstop() {
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{SpaceOperation, Variable};
        use crate::translator::rtyper::rint::signed_repr;
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = setup_rtyper();
        let r_int: Arc<dyn Repr> = signed_repr();
        let r_vec = rust_vec_repr(&rtyper, VecItemKind::Ref);
        let vec_lltype = r_vec.lowleveltype().clone();

        let v_lst = Variable::new();
        v_lst.set_concretetype(Some(vec_lltype));
        let v_start = Variable::new();
        v_start.set_concretetype(Some(LowLevelType::Signed));
        let v_stop = Variable::new();
        v_stop.set_concretetype(Some(LowLevelType::Signed));
        let v_lst_h = Hlvalue::Variable(v_lst);
        let v_start_h = Hlvalue::Variable(v_start);
        let v_stop_h = Hlvalue::Variable(v_stop);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            "getslice".to_string(),
            vec![v_lst_h.clone(), v_start_h.clone(), v_stop_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = std::rc::Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_lst_h);
        hop.args_s.borrow_mut().push(SomeValue::RustVec(
            crate::annotator::model::SomeRustVec::for_kind(VecItemKind::Ref),
        ));
        hop.args_r.borrow_mut().push(Some(r_vec.clone()));
        hop.args_v.borrow_mut().push(v_start_h);
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(
                /* nonneg */ true, false,
            )));
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        hop.args_v.borrow_mut().push(v_stop_h);
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(
                /* nonneg */ true, false,
            )));
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        *hop.r_result.borrow_mut() = Some(r_vec.clone());

        let out = rtyper
            .translate_operation(&hop)
            .expect("translate_operation getslice must dispatch to RustVecRepr.rtype_getslice")
            .expect("getslice returns the fresh Vec Variable");
        let Hlvalue::Variable(_) = out else {
            panic!("getslice must return a Variable (the ll_listslice result)");
        };
        let ops = hop.llops.borrow();
        let direct_calls = ops
            .ops
            .iter()
            .filter(|op| op.opname == "direct_call")
            .count();
        assert_eq!(
            direct_calls,
            1,
            "expected a single ll_listslice_startstop direct_call, got {:?}",
            ops.ops.iter().map(|op| &op.opname).collect::<Vec<_>>()
        );
        let names: Vec<String> = ann
            .translator
            .graphs
            .borrow()
            .iter()
            .map(|g| g.borrow().name.clone())
            .collect();
        assert!(
            names.iter().any(|n| n == "ll_listslice_startstop"),
            "getslice must mint an ll_listslice_startstop helper graph, got {names:?}"
        );
    }

    /// `v[start:]` on a `Vec<PyObjectRef>` (`VecItemKind::Ref`) decomposes
    /// to `"startonly"` and lowers to
    /// `gendirectcall(ll_listslice_startonly, l, start)`.
    #[test]
    fn rtype_getslice_ref_startonly_emits_ll_listslice_startonly() {
        use crate::annotator::model::{SomeInteger, SomeValue, s_none};
        use crate::flowspace::model::{Constant, SpaceOperation, Variable};
        use crate::translator::rtyper::rint::signed_repr;
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = setup_rtyper();
        let r_int: Arc<dyn Repr> = signed_repr();
        let r_vec = rust_vec_repr(&rtyper, VecItemKind::Ref);
        let vec_lltype = r_vec.lowleveltype().clone();

        let v_lst = Variable::new();
        v_lst.set_concretetype(Some(vec_lltype));
        let v_start = Variable::new();
        v_start.set_concretetype(Some(LowLevelType::Signed));
        let v_lst_h = Hlvalue::Variable(v_lst);
        let v_start_h = Hlvalue::Variable(v_start);
        let c_stop = Hlvalue::Constant(Constant::new(ConstValue::None));
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            "getslice".to_string(),
            vec![v_lst_h.clone(), v_start_h.clone(), c_stop.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = std::rc::Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_lst_h);
        hop.args_s.borrow_mut().push(SomeValue::RustVec(
            crate::annotator::model::SomeRustVec::for_kind(VecItemKind::Ref),
        ));
        hop.args_r.borrow_mut().push(Some(r_vec.clone()));
        hop.args_v.borrow_mut().push(v_start_h);
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(
                /* nonneg */ true, false,
            )));
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        hop.args_v.borrow_mut().push(c_stop);
        hop.args_s.borrow_mut().push(s_none());
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        *hop.r_result.borrow_mut() = Some(r_vec.clone());

        rtyper
            .translate_operation(&hop)
            .expect("translate_operation getslice must dispatch to RustVecRepr.rtype_getslice")
            .expect("getslice returns the fresh Vec Variable");
        let ops = hop.llops.borrow();
        let direct_calls = ops
            .ops
            .iter()
            .filter(|op| op.opname == "direct_call")
            .count();
        assert_eq!(
            direct_calls,
            1,
            "expected a single ll_listslice_startonly direct_call, got {:?}",
            ops.ops.iter().map(|op| &op.opname).collect::<Vec<_>>()
        );
        let names: Vec<String> = ann
            .translator
            .graphs
            .borrow()
            .iter()
            .map(|g| g.borrow().name.clone())
            .collect();
        assert!(
            names.iter().any(|n| n == "ll_listslice_startonly"),
            "getslice [start:] must mint an ll_listslice_startonly helper graph, got {names:?}"
        );
    }
}
