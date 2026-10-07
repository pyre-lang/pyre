//! RPython `rpython/rtyper/rlist.py` + `lltypesystem/rlist.py` —
//! minimal `FixedSizeListRepr` slice covering the `len(list)` lowering.
//!
//! Rust slices (`&[T]`) and fixed arrays annotate as a non-resized
//! `SomeList` (`bookkeeper`), so `slice.len()` — routed to the rtyper's
//! `len` operation by `flowspace_adapter` — lands on this repr's
//! `rtype_len`.
//!
//! The resized `ListRepr` lands its data shape (`Ptr(GcStruct("list",
//! ("length", Signed), ("items", Ptr(GcArray(ITEM)))))`) and the
//! `rtype_len` lowering (`ll_length` reads the `length` field). The
//! `ADTIList` / `ADTIFixedList` adtmeth tables, `rtype_method_append` /
//! `getitem` / `setitem` / iterators, the `_ll_list_resize*` family and
//! the `pairtype(BaseListRepr, BaseListRepr)` operations are deferred to
//! follow-on slices. For `FixedSizeListRepr` only the data shape
//! `Ptr(GcArray(ITEM))` and the `ll_fixed_length` lowering land.

use std::rc::Rc;
use std::sync::Arc;

use crate::flowspace::model::{
    Block, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation, Variable,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::backendopt::constfold::WE_ARE_JITTED_TAG_ID;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    Array, FuncType, LowLevelType, Ptr, PtrTarget, Struct, functionptr,
};
use crate::translator::rtyper::lltypesystem::rstr::sub_helper_funcptr_constant;
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{
    ConvertedTo, GenopResult, HighLevelOp, LowLevelFunction, RPythonTyper, SliceKind,
    constant_with_lltype, exception_args, functionptr_const, helper_pygraph_from_graph,
    variable_with_lltype, void_field_const,
};

/// RPython `class FixedSizeListRepr(AbstractFixedSizeListRepr,
/// BaseListRepr)` (`lltypesystem/rlist.py`):
///
/// ```python
/// def _setup_repr(self):
///     if 'item_repr' not in self.__dict__:
///         self.external_item_repr, self.item_repr = externalvsinternal(
///             self.rtyper, self._item_repr_computer(), gcref=True)
///     if isinstance(self.LIST, GcForwardReference):
///         ITEMARRAY = self.get_itemarray_lowleveltype()
///         self.LIST.become(ITEMARRAY)
/// ```
///
/// `LIST` becomes a bare `GcArray(ITEM)` (no `length`/`items` wrapper —
/// that is the resized `ListRepr`), so the low-level type is
/// `Ptr(GcArray(ITEM))`.
#[derive(Debug)]
pub struct FixedSizeListRepr {
    state: ReprState,
    lltype: LowLevelType,
    /// `self.item_repr` (`lltypesystem/rlist.py:177`) — the internal
    /// (gcref-wrapped) element repr; its lowleveltype is the array
    /// element type and the `getitem` result type.
    item_repr: Arc<dyn Repr>,
    /// `self.external_item_repr` (`lltypesystem/rlist.py:177`) — the
    /// external element repr `recast` ([`list_recast`]) converts a getitem
    /// result back to. For primitive items this is a clone of the same
    /// cached `Arc` as `item_repr`, so the recast short-circuits to
    /// identity; for a gc `InstanceRepr` it is the concrete repr while
    /// `item_repr` is the generic `OBJECTPTR` root.
    external_item_repr: Arc<dyn Repr>,
}

impl FixedSizeListRepr {
    pub fn new(rtyper: &Rc<RPythonTyper>, item_repr: Arc<dyn Repr>) -> Result<Self, TyperError> {
        // `externalvsinternal(rtyper, item_repr, gcref=True)` —
        // gc `InstanceRepr` items become the generic `Ptr(OBJECT)`
        // gcref so the array element type is never a gc container
        // (which `Array::gc` rejects); non-instance reprs pass
        // through unchanged.
        let (external_item_repr, internal) =
            crate::translator::rtyper::rclass::externalvsinternal(rtyper, item_repr, true)?;
        let item_lltype = internal.lowleveltype().clone();
        let arr = Array::gc(item_lltype);
        let lltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(arr),
        }));
        Ok(FixedSizeListRepr {
            state: ReprState::new(),
            lltype,
            item_repr: internal,
            external_item_repr,
        })
    }

    pub(crate) fn item_lowleveltype(&self) -> LowLevelType {
        self.item_repr.lowleveltype().clone()
    }
}

/// RPython `AbstractBaseListRepr.recast(self, llops, v)` (`rlist.py`):
///
/// ```python
/// def recast(self, llops, v):
///     return llops.convertvar(v, self.item_repr, self.external_item_repr)
/// ```
///
/// Converts a `getitem` result from the internal element repr (the array
/// element type) back to the external repr the caller annotated. For the
/// primitive items a live subject builds, `item_repr` and
/// `external_item_repr` are clones of the same cached `Arc`, so `convertvar`
/// short-circuits to identity (its `ptr::eq` guard) and emits no op. A
/// gc-instance element list (`external != internal`) downcasts the generic
/// `OBJECTPTR` internal repr to the concrete external `InstanceRepr` via the
/// `cast_pointer` `pair(InstanceRepr, InstanceRepr)` arm.
fn list_recast(
    hop: &HighLevelOp,
    item_repr: &Arc<dyn Repr>,
    external_item_repr: &Arc<dyn Repr>,
    v: Hlvalue,
) -> Result<Hlvalue, TyperError> {
    hop.llops
        .borrow_mut()
        .convertvar(v, item_repr.as_ref(), external_item_repr.as_ref())
}

impl Repr for FixedSizeListRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "FixedSizeListRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::FixedSizeListRepr
    }

    /// RPython `AbstractBaseListRepr.rtype_len(self, hop)`
    /// (`rlist.py:124-130`):
    ///
    /// ```python
    /// def rtype_len(self, hop):
    ///     v_lst, = hop.inputargs(self)
    ///     if hop.args_s[0].listdef.listitem.resized:
    ///         ll_func = ll_len
    ///     else:
    ///         ll_func = ll_len_foldable
    ///     return hop.gendirectcall(ll_func, v_lst)
    /// ```
    ///
    /// `FixedSizeListRepr` is only minted for the non-resized case, so
    /// the lowering is the `ll_len_foldable` path: `l.ll_length()` →
    /// `ll_fixed_length` (`lltypesystem/rlist.py`) = `len(l)` =
    /// the `getarraysize` op on the `Ptr(GcArray)` receiver. The
    /// `len_foldable` oopspec is a tracing-time JIT hint; the lowered
    /// op is the bare `getarraysize`.
    fn rtype_len(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        let ptr_lltype = self.lltype.clone();
        let ptr_for_builder = ptr_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_fixed_length".to_string(),
            vec![ptr_lltype],
            LowLevelType::Signed,
            move |_rtyper, _args, _result| {
                build_ll_fixed_length_helper_graph("ll_fixed_length", ptr_for_builder.clone())
            },
        )?;
        hop.gendirectcall(&helper, vlist)
    }

    /// RPython `pair(AbstractBaseListRepr, IntegerRepr).rtype_getitem`
    /// (`rlist.py:247-267`):
    ///
    /// ```python
    /// def rtype_getitem((r_lst, r_int), hop, checkidx=False):
    ///     v_lst, v_index = hop.inputargs(r_lst, Signed)
    ///     ...
    ///     spec = dum_nocheck
    ///     hop.exception_cannot_occur()
    ///     ...
    ///     if hop.args_s[1].nonneg:
    ///         llfn = ll_getitem_nonneg
    ///     ...
    ///     v_res = hop.gendirectcall(llfn, ..., v_lst, v_index)
    ///     return r_lst.recast(hop.llops, v_res)
    /// ```
    ///
    /// `FixedSizeListRepr` is the non-resized list — a Rust slice
    /// (`&[T]`) or fixed array. The `(checkidx, nonneg)` selection is
    /// shared with the resized list through [`list_rtype_getitem`]: the
    /// nonneg + `dum_nocheck` fast path collapses through
    /// `ll_getitem_foldable_nonneg` → `ll_fixed_getitem_fast(l, index)` →
    /// `l[index]` (`lltypesystem/rlist.py`) to the bare
    /// `getarrayitem` on the `Ptr(GcArray)` receiver, while the
    /// negative-index (`ll_fixed_getitem`) and `checkidx`
    /// (IndexError-raising `ll_fixed_getitem_*_checked`) helpers fold / window
    /// the index before dispatching to that fast helper. Rust slice indexing
    /// only ever exercises the nonneg + `dum_nocheck` path.
    ///
    /// The upstream result `recast` (`rlist.py`
    /// `return r_lst.recast(hop.llops, v_res)`) is applied via
    /// [`list_recast`]: `convertvar(v_res, item_repr, external_item_repr)`.
    /// For the primitive items a live subject builds `external == internal`
    /// (clones of the same cached `Arc`), so the recast short-circuits to
    /// identity and emits no op; a GC-instance list downcasts the internal
    /// root repr to the concrete external repr.
    fn rtype_getitem(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_res = list_rtype_getitem(
            hop,
            self,
            ListLayout::Fixed,
            &self.lltype,
            self.item_repr.lowleveltype(),
        )?;
        list_recast(hop, &self.item_repr, &self.external_item_repr, v_res).map(Some)
    }

    /// RPython `pair(AbstractBaseListRepr, IntegerRepr).rtype_setitem`
    /// (`rlist.py:272-284`):
    ///
    /// ```python
    /// def rtype_setitem((r_lst, r_int), hop):
    ///     if hop.has_implicit_exception(IndexError):
    ///         spec = dum_checkidx
    ///     else:
    ///         spec = dum_nocheck
    ///     v_func = hop.inputconst(Void, spec)
    ///     v_lst, v_index, v_item = hop.inputargs(r_lst, Signed, r_lst.item_repr)
    ///     if hop.args_s[1].nonneg:
    ///         llfn = ll_setitem_nonneg
    ///     else:
    ///         llfn = ll_setitem
    ///     hop.exception_is_here()
    ///     return hop.gendirectcall(llfn, v_func, v_lst, v_index, v_item)
    /// ```
    ///
    /// Shares the `(checkidx, nonneg)` selection with the resized list via
    /// [`list_rtype_setitem`]. The `dum_nocheck` + nonneg fast path
    /// (`ll_setitem_nonneg(dum_nocheck, l, index, item)`, no IndexError
    /// branch — `index >= 0` is a debug `ll_assert`) collapses through
    /// `l.ll_setitem_fast` → `ll_fixed_setitem_fast(l, index, item)` →
    /// `l[index] = item` (`lltypesystem/rlist.py`) to the bare
    /// `setarrayitem` on the `Ptr(GcArray)` receiver; the negative-index
    /// (`ll_fixed_setitem`) and `checkidx` (IndexError-raising
    /// `ll_fixed_setitem_*_checked`) helpers fold / window the index first.
    /// The third inputarg converts to the internal `item_repr` (the
    /// gcref-wrapped element repr), so `rtype_setitem` does not `recast`.
    fn rtype_setitem(&self, hop: &HighLevelOp) -> RTypeResult {
        list_rtype_setitem(
            hop,
            self,
            ListLayout::Fixed,
            &self.lltype,
            self.item_repr.lowleveltype(),
            self.item_repr.as_ref(),
        )
    }

    /// RPython `AbstractBaseListRepr.rtype_method_reverse(self, hop)`
    /// (`rlist.py:138-143`):
    ///
    /// ```python
    /// def rtype_method_reverse(self, hop):
    ///     v_lst, = hop.inputargs(self)
    ///     hop.exception_cannot_occur()
    ///     hop.gendirectcall(ll_reverse, v_lst)
    /// ```
    ///
    /// `ll_reverse` (`rlist.py`) is an in-place swap loop over the
    /// `FixedSizeListRepr` receiver (the bare `Ptr(GcArray)`): it reads both
    /// endpoints, writes them crossed, and walks `i` up / `length_1_i` down
    /// toward the middle. The lowered body is the multi-block CFG built by
    /// [`build_ll_reverse_helper_graph`]. `reverse` returns `None` (void).
    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        match method_name {
            // `is_null` is the lltype `_ptr` nullity probe on the list
            // backing pointer — lower it as `ptr_iszero` (opimpl.py
            // `op_ptr_iszero`), matching the `ptr_method_is_null` bound
            // method the annotator seats on pointer-carrying receivers.
            "is_null" => {
                let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
                Ok(hop.genop("ptr_iszero", vlist, GenopResult::LLType(LowLevelType::Bool)))
            }
            "reverse" => {
                let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
                hop.exception_cannot_occur()?;
                let item_lltype = self.item_repr.lowleveltype().clone();
                let ptr_lltype = self.lltype.clone();
                let ptr_for_builder = ptr_lltype.clone();
                let item_for_builder = item_lltype.clone();
                let helper = hop.rtyper.lowlevel_helper_function_with_builder(
                    "ll_reverse".to_string(),
                    vec![ptr_lltype],
                    LowLevelType::Void,
                    move |_rtyper, _args, _result| {
                        build_ll_reverse_helper_graph(
                            "ll_reverse",
                            ptr_for_builder.clone(),
                            item_for_builder.clone(),
                        )
                    },
                )?;
                hop.gendirectcall(&helper, vlist)
            }
            _ => Err(TyperError::message(format!(
                "missing FixedSizeListRepr.rtype_method_{method_name}"
            ))),
        }
    }

    /// RPython `AbstractBaseListRepr.rtype_bltn_list(self, hop)`
    /// (`rlist.py:118-122`) — `list(slice)` copies the slice into a fresh
    /// resized list via `ll_copy(RESLIST, l)`. The source is the fixed-size
    /// receiver (bare `Ptr(GcArray)`).
    fn rtype_bltn_list(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        hop.exception_is_here()?;
        rtype_bltn_list_via_ll_copy(
            hop,
            ListLayout::Fixed,
            self.lltype.clone(),
            self.item_repr.lowleveltype().clone(),
            vlist,
        )
    }

    /// RPython `AbstractBaseListRepr.rtype_getslice(r_lst, hop)`
    /// (`rlist.py:409-414`) — the fixed-size receiver slices into a fresh
    /// resized list (`hop.r_result`), reading its length via `getarraysize`.
    fn rtype_getslice(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_lst = hop.inputarg(ConvertedTo::Repr(self), 0)?;
        rtype_getslice_via_ll_listslice(
            hop,
            ListLayout::Fixed,
            self.lltype.clone(),
            self.item_repr.lowleveltype().clone(),
            v_lst,
        )
    }

    /// RPython `BaseListRepr.make_iterator_repr(self, *variant)`
    /// (`lltypesystem/rlist.py`).
    #[expect(
        clippy::arc_with_non_send_sync,
        reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
    )]
    fn make_iterator_repr(
        &self,
        variant: &[String],
        foldable: bool,
    ) -> Result<Arc<dyn Repr>, TyperError> {
        list_make_iterator_repr(
            self.lltype.clone(),
            self.item_repr.clone(),
            self.external_item_repr.clone(),
            true,
            foldable,
            variant,
        )
    }
}

/// RPython `class AbstractListIteratorRepr(IteratorRepr)` (`rlist.py`).
///
/// Concrete list iterator layouts live in the lltypesystem-specific module.
/// The generic iterator lowering methods are still deferred.
#[derive(Debug, Default)]
pub struct AbstractListIteratorRepr;

fn rlist_runtime_deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!("rlist.{name} — list helper deferred"))
}

/// RPython `rtype_newlist(hop, v_sizehint=None)` (`rlist.py`) +
/// `newlist(llops, r_list, items_v, v_sizehint=None)` (`rlist.py`):
///
/// ```python
/// def rtype_newlist(hop, v_sizehint=None):
///     nb_args = hop.nb_args
///     r_list = hop.r_result
///     r_listitem = r_list.item_repr
///     items_v = [hop.inputarg(r_listitem, arg=i) for i in range(nb_args)]
///     return newlist(hop.llops, r_list, items_v, v_sizehint)
///
/// def newlist(llops, r_list, items_v, v_sizehint=None):
///     LIST = r_list.lowleveltype.TO
///     cno = inputconst(Signed, len(items_v))
///     v_result = llops.gendirectcall(LIST.ll_newlist, cno)
///     v_func = inputconst(Void, dum_nocheck)
///     for i, v_item in enumerate(items_v):
///         ci = inputconst(Signed, i)
///         llops.gendirectcall(ll_setitem_nonneg, v_func, v_result, ci, v_item)
///     return v_result
/// ```
///
/// Constructs a fresh list via the shared [`build_ll_newlist_helper_graph`],
/// then fills it positionally with `ll_setitem_fast` calls; the `dum_nocheck`
/// index is statically `0..n`, so the negative-index / bound-check wrappers
/// are skipped exactly as upstream's `ll_setitem_nonneg`-fast-path does. Each
/// element is coerced to `r_list.item_repr` (the internal gcref-wrapped
/// element repr) before being stored.
///
/// `LIST.ll_newlist` is a repr-generic typemethod upstream: the resized
/// `ListRepr` binds `ll_newlist` (`Ptr(GcStruct("list", length, items))`) and
/// the never-resized `FixedSizeListRepr` binds `ll_fixed_newlist`
/// (`Ptr(GcArray)`). This dispatches on the concrete `hop.r_result` repr and
/// forwards to [`rtype_newlist_layout`] with the matching [`ListLayout`].
pub fn rtype_newlist(hop: &HighLevelOp) -> RTypeResult {
    let r_result = hop
        .r_result
        .borrow()
        .clone()
        .ok_or_else(|| TyperError::message("rtype_newlist: r_result missing"))?;
    let any_r: &dyn std::any::Any = r_result.as_ref();
    // `hop.r_result` is a `list` display's repr: the resized `ListRepr`
    // (`Ptr(GcStruct("list", length, items))`) for a mutated `list(...)`, or
    // the `FixedSizeListRepr` (`Ptr(GcArray)`) for a never-resized `vec![...]`.
    // The `LIST.ll_newlist` typemethod is repr-generic upstream
    // (`ll_newlist` vs `ll_fixed_newlist`); dispatch on the concrete repr and
    // parameterise the shared body on [`ListLayout`].
    if let Some(r_list) = any_r.downcast_ref::<ListRepr>() {
        return rtype_newlist_layout(
            hop,
            r_list.lltype.clone(),
            r_list.item_repr.lowleveltype().clone(),
            r_list.item_repr.as_ref(),
            ListLayout::Resized,
        );
    }
    if let Some(r_list) = any_r.downcast_ref::<FixedSizeListRepr>() {
        return rtype_newlist_layout(
            hop,
            r_list.lltype.clone(),
            r_list.item_repr.lowleveltype().clone(),
            r_list.item_repr.as_ref(),
            ListLayout::Fixed,
        );
    }
    Err(TyperError::message(
        "rtype_newlist: hop.r_result is not a ListRepr or FixedSizeListRepr",
    ))
}

/// RPython `objectmodel.py`'s `newlist_hint` ExtRegistry specialization and
/// `lltypesystem/rlist.py:ll_newlist_hint`:
///
/// ```python
/// def specialize_call(self, hop):
///     v_hint = hop.inputarg(lltype.Signed, 0)
///     r_list = hop.r_result
///     hop.exception_is_here()
///     return hop.gendirectcall(r_list.LIST.ll_newlist_hint, v_hint)
///
/// def ll_newlist_hint(LIST, lengthhint):
///     ll_assert(lengthhint >= 0, "negative list length")
///     l = malloc(LIST)
///     l.length = 0
///     l.items = malloc(LIST.items.TO, lengthhint)
///     return l
/// ```
///
/// The annotator marks the result resized, so only [`ListRepr`] is valid.
/// Capacity and logical length intentionally differ: the backing array uses
/// `lengthhint`, while the list header starts at zero for subsequent append.
pub fn rtype_newlist_hint(hop: &HighLevelOp) -> RTypeResult {
    if hop.nb_args() != 1 {
        return Err(TyperError::message(format!(
            "rtype_newlist_hint: expected one size hint, got {}",
            hop.nb_args()
        )));
    }
    let r_result = hop
        .r_result
        .borrow()
        .clone()
        .ok_or_else(|| TyperError::message("rtype_newlist_hint: r_result missing"))?;
    let any_r: &dyn std::any::Any = r_result.as_ref();
    let r_list = any_r.downcast_ref::<ListRepr>().ok_or_else(|| {
        TyperError::message("rtype_newlist_hint: hop.r_result is not a resized ListRepr")
    })?;
    let v_hint = hop.inputarg(ConvertedTo::LowLevelType(&LowLevelType::Signed), 0)?;
    hop.exception_is_here()?;
    let ptr_lltype = r_list.lltype.clone();
    let item_lltype = r_list.item_repr.lowleveltype().clone();
    let helper = {
        let ptr = ptr_lltype.clone();
        let item = item_lltype.clone();
        hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_newlist_hint".to_string(),
            vec![LowLevelType::Signed],
            ptr_lltype,
            move |_rtyper, _args, _result| {
                build_ll_newlist_hint_helper_graph("ll_newlist_hint", ptr.clone(), item.clone())
            },
        )?
    };
    hop.gendirectcall(&helper, vec![v_hint])
}

/// Repr-generic body of [`rtype_newlist`], shared by the resized `ListRepr`
/// and the fixed-size `FixedSizeListRepr`. The `LIST.ll_newlist(n)` +
/// positional `ll_setitem_fast` shape is identical; only the receiver layout
/// (and hence the concrete helper graphs) differ.
///
/// The two layouts build DIFFERENT helper graphs (the resized newlist mallocs
/// a `GcStruct` header + items array and stores both fields; the fixed newlist
/// is the bare `malloc_varsize`), so they MUST use distinct helper-fn names —
/// `lowlevel_helper_function_with_builder` memoises on `(name, args, result)`
/// and would otherwise be at risk of serving a cached resized graph for a
/// fixed request. Distinct names (`ll_newlist` / `ll_fixed_newlist`,
/// `ll_setitem_fast` / `ll_fixed_setitem_fast`) also match the upstream
/// `ADTIList` / `ADTIFixedList` adtmeth entries.
fn rtype_newlist_layout(
    hop: &HighLevelOp,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    item_repr: &dyn Repr,
    layout: ListLayout,
) -> RTypeResult {
    let n = hop.nb_args();

    // upstream `items_v = [hop.inputarg(r_list.item_repr, i) for i in range(n)]`.
    let converted: Vec<ConvertedTo<'_>> = (0..n).map(|_| ConvertedTo::Repr(item_repr)).collect();
    let items_v = hop.inputargs(converted)?;

    let (newlist_name, setitem_name) = match layout {
        ListLayout::Fixed => ("ll_fixed_newlist", "ll_fixed_setitem_fast"),
        ListLayout::Resized => ("ll_newlist", "ll_setitem_fast"),
    };

    // upstream `v_result = llops.gendirectcall(LIST.ll_newlist, cno)`.
    let newlist_fn = {
        let ptr = ptr_lltype.clone();
        let item = item_lltype.clone();
        hop.rtyper.lowlevel_helper_function_with_builder(
            newlist_name.to_string(),
            vec![LowLevelType::Signed],
            ptr_lltype.clone(),
            move |_rtyper, _args, _result| {
                build_ll_newlist_helper_graph(newlist_name, layout, ptr.clone(), item.clone())
            },
        )?
    };
    let v_result = hop
        .gendirectcall(&newlist_fn, vec![signed_const(n as i64)])?
        .ok_or_else(|| TyperError::message("rtype_newlist: ll_newlist returned Void"))?;

    // upstream loop — `ll_setitem_nonneg(dum_nocheck, v_result, ci, v_item)`,
    // which bottoms out at `l.ll_setitem_fast(index, item)`. The index is the
    // static enumeration position, so the fast helper is called directly.
    let setitem_fn = {
        let ptr = ptr_lltype.clone();
        let item = item_lltype.clone();
        hop.rtyper.lowlevel_helper_function_with_builder(
            setitem_name.to_string(),
            vec![ptr_lltype, LowLevelType::Signed, item_lltype],
            LowLevelType::Void,
            move |_rtyper, _args, _result| match layout {
                ListLayout::Fixed => build_ll_fixed_setitem_fast_helper_graph(
                    setitem_name,
                    ptr.clone(),
                    item.clone(),
                ),
                ListLayout::Resized => {
                    build_ll_setitem_fast_helper_graph(setitem_name, ptr.clone(), item.clone())
                }
            },
        )?
    };
    for (i, v_item) in items_v.into_iter().enumerate() {
        hop.gendirectcall(
            &setitem_fn,
            vec![v_result.clone(), signed_const(i as i64), v_item],
        )?;
    }
    Ok(Some(v_result))
}

/// RPython `rtype_alloc_and_set(hop)` (`rlist.py`):
///
/// ```python
/// def rtype_alloc_and_set(hop):
///     r_list = hop.r_result
///     v_count, v_item = hop.inputargs(Signed, r_list.item_repr)
///     cLIST = hop.inputconst(Void, r_list.LIST)
///     return hop.gendirectcall(ll_alloc_and_set, cLIST, v_count, v_item)
/// ```
///
/// Mirrors [`rtype_newlist`]: dispatch on the concrete `hop.r_result` repr
/// (`ListRepr` → resized, `FixedSizeListRepr` → fixed) and forward to the
/// repr-generic [`rtype_alloc_and_set_layout`]. The upstream `cLIST` Void arg
/// selects `LIST.ll_newlist`; pyre folds it into the helper IDENTITY (a
/// distinct helper name per layout selects the newlist body), so it is not
/// passed at runtime.
pub fn rtype_alloc_and_set(hop: &HighLevelOp) -> RTypeResult {
    let r_result = hop
        .r_result
        .borrow()
        .clone()
        .ok_or_else(|| TyperError::message("rtype_alloc_and_set: r_result missing"))?;
    let any_r: &dyn std::any::Any = r_result.as_ref();
    if let Some(r_list) = any_r.downcast_ref::<ListRepr>() {
        return rtype_alloc_and_set_layout(
            hop,
            r_list.lltype.clone(),
            r_list.item_repr.lowleveltype().clone(),
            r_list.item_repr.as_ref(),
            ListLayout::Resized,
        );
    }
    if let Some(r_list) = any_r.downcast_ref::<FixedSizeListRepr>() {
        return rtype_alloc_and_set_layout(
            hop,
            r_list.lltype.clone(),
            r_list.item_repr.lowleveltype().clone(),
            r_list.item_repr.as_ref(),
            ListLayout::Fixed,
        );
    }
    Err(TyperError::message(
        "rtype_alloc_and_set: hop.r_result is not a ListRepr or FixedSizeListRepr",
    ))
}

/// Repr-generic body of [`rtype_alloc_and_set`]. `hop.inputargs(Signed,
/// r_list.item_repr)` yields `(v_count, v_item)`; then `ll_alloc_and_set` is
/// minted (with its full `{jit, nojit} → {clear, nonnull, arrayclear,
/// arrayfill}` sub-helper closure) and `direct_call`ed with `[v_count,
/// v_item]`.
///
/// Distinct helper names per layout (`ll_alloc_and_set` /
/// `ll_fixed_alloc_and_set`) keep the `(name, args, result)` helper cache from
/// serving a resized graph for a fixed request — the same discipline
/// [`rtype_newlist_layout`] documents.
fn rtype_alloc_and_set_layout(
    hop: &HighLevelOp,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    item_repr: &dyn Repr,
    layout: ListLayout,
) -> RTypeResult {
    // upstream `v_count, v_item = hop.inputargs(Signed, r_list.item_repr)`.
    let inputs = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::Signed),
        ConvertedTo::Repr(item_repr),
    ])?;
    let mut inputs = inputs.into_iter();
    let v_count = inputs
        .next()
        .ok_or_else(|| TyperError::message("rtype_alloc_and_set: missing count arg"))?;
    let v_item = inputs
        .next()
        .ok_or_else(|| TyperError::message("rtype_alloc_and_set: missing item arg"))?;

    let helper_name = alloc_and_set_helper_name(layout);
    let helper = {
        let ptr = ptr_lltype.clone();
        let item = item_lltype.clone();
        hop.rtyper.lowlevel_helper_function_with_builder(
            helper_name.to_string(),
            vec![LowLevelType::Signed, item_lltype.clone()],
            ptr_lltype.clone(),
            move |rtyper, _args, _result| {
                build_alloc_and_set_family(rtyper, layout, ptr.clone(), item.clone())
            },
        )?
    };
    hop.gendirectcall(&helper, vec![v_count, v_item])
}

/// Per-layout name for the dispatch helper (`ll_alloc_and_set`). The `(name,
/// args, result)` helper cache would otherwise serve a resized graph for a
/// fixed request, so the fixed layout gets its own identity.
fn alloc_and_set_helper_name(layout: ListLayout) -> &'static str {
    match layout {
        ListLayout::Fixed => "ll_fixed_alloc_and_set",
        ListLayout::Resized => "ll_alloc_and_set",
    }
}

/// Mint the `ll_alloc_and_set` family in dependency order and build the
/// top-level dispatch graph ([`build_ll_alloc_and_set_helper_graph`]).
///
/// The chain (`rlist.py` `ll_alloc_and_set`) is
/// `ll_alloc_and_set → {_ll_alloc_and_set_jit, _ll_alloc_and_set_nojit}`,
/// with the jit arm reaching `_ll_alloc_and_clear` or the
/// `look_inside_iff` wrapper around `_ll_alloc_and_set_nonnull`
/// (`rlib/jit.py` `look_inside_iff`). Both leaf arms reach
/// `LIST.ll_newlist` plus the T1 rgc helpers `rgc.ll_arrayclear` /
/// `rgc.ll_arrayfill`. Each sub-helper is minted through
/// [`RPythonTyper::lowlevel_helper_function_with_builder`], converted to a
/// callee `Constant` by [`functionptr_const`], and threaded into the parent
/// builder — the exact orchestration shape of `rtype_builder_new`.
///
/// The public nonnull graph stays the original four-block loop. The
/// wrapper, its `dont_look_inside` trampoline, and the `isconstant`
/// fallback are separate graphs; the jit arm calls the wrapper.
fn build_alloc_and_set_family(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let items_ptr = items_array_ptr_lltype(&item_lltype);
    let (newlist_name, clear_name, nonnull_name, jit_name, nojit_name) =
        alloc_and_set_sub_names(layout);

    // LIST.ll_newlist(count) — shared by every leaf arm.
    let newlist = {
        let ptr = ptr_lltype.clone();
        let item = item_lltype.clone();
        rtyper.lowlevel_helper_function_with_builder(
            newlist_name.to_string(),
            vec![LowLevelType::Signed],
            ptr_lltype.clone(),
            move |_r, _a, _res| {
                build_ll_newlist_helper_graph(newlist_name, layout, ptr.clone(), item.clone())
            },
        )?
    };

    // rgc.ll_arrayclear(items) / rgc.ll_arrayfill(items, item) — the T1
    // helpers.  Both take the ELEMENT lltype and build `Ptr(GcArray(ITEM))`
    // internally, so their `p` param IS the items-array ptr.
    let arrayclear = {
        let item = item_lltype.clone();
        rtyper.lowlevel_helper_function_with_builder(
            "ll_arrayclear".to_string(),
            vec![items_ptr.clone()],
            LowLevelType::Void,
            move |_r, _a, _res| {
                crate::translator::rtyper::lltypesystem::rbuilder::build_ll_arrayclear_helper_graph(
                    "ll_arrayclear",
                    item.clone(),
                )
            },
        )?
    };
    let arrayfill = {
        let item = item_lltype.clone();
        rtyper.lowlevel_helper_function_with_builder(
            "ll_arrayfill".to_string(),
            vec![items_ptr.clone(), item_lltype.clone()],
            LowLevelType::Void,
            move |_r, _a, _res| {
                crate::translator::rtyper::lltypesystem::rbuilder::build_ll_arrayfill_helper_graph(
                    "ll_arrayfill",
                    item.clone(),
                )
            },
        )?
    };

    // ll_setitem_fast(l, i, item) — the resized/fixed store used by the
    // nonnull loop.  It takes the LIST ptr directly (does the items getfield
    // internally for the resized layout).
    let setitem = {
        let ptr = ptr_lltype.clone();
        let item = item_lltype.clone();
        let setitem_name = layout.setitem_fast_name();
        rtyper.lowlevel_helper_function_with_builder(
            setitem_name.to_string(),
            vec![
                ptr_lltype.clone(),
                LowLevelType::Signed,
                item_lltype.clone(),
            ],
            LowLevelType::Void,
            move |_r, _a, _res| match layout {
                ListLayout::Fixed => build_ll_fixed_setitem_fast_helper_graph(
                    setitem_name,
                    ptr.clone(),
                    item.clone(),
                ),
                ListLayout::Resized => {
                    build_ll_setitem_fast_helper_graph(setitem_name, ptr.clone(), item.clone())
                }
            },
        )?
    };

    // @jit.oopspec("newlist_clear(count)") _ll_alloc_and_clear(LIST, count)
    // (rlist.py:522-528).  The oopspec is attached in lib.rs by CallPath —
    // the codewriter derives the path from this graph's NAME.
    let clear = {
        let ptr = ptr_lltype.clone();
        let newlist_fn = functionptr_const(rtyper, &newlist)?;
        let arrayclear_fn = functionptr_const(rtyper, &arrayclear)?;
        let item = item_lltype.clone();
        rtyper.lowlevel_helper_function_with_builder(
            clear_name.to_string(),
            vec![LowLevelType::Signed],
            ptr_lltype.clone(),
            move |_r, _a, _res| {
                build_ll_alloc_and_clear_helper_graph(
                    clear_name,
                    layout,
                    ptr.clone(),
                    item.clone(),
                    newlist_fn.clone(),
                    arrayclear_fn.clone(),
                )
            },
        )?
    };

    // _ll_alloc_and_set_nonnull(LIST, count, item) (rlist.py). The
    // graph under this name is the original loop. `look_inside_iff`
    // wraps it below; the jit arm calls the wrapper.
    let nonnull = {
        let ptr = ptr_lltype.clone();
        let newlist_fn = functionptr_const(rtyper, &newlist)?;
        let setitem_fn = functionptr_const(rtyper, &setitem)?;
        let item = item_lltype.clone();
        rtyper.lowlevel_helper_function_with_builder(
            nonnull_name.to_string(),
            vec![LowLevelType::Signed, item_lltype.clone()],
            ptr_lltype.clone(),
            move |_r, _a, _res| {
                build_ll_alloc_and_set_nonnull_helper_graph(
                    nonnull_name,
                    ptr.clone(),
                    item.clone(),
                    newlist_fn.clone(),
                    setitem_fn.clone(),
                )
            },
        )?
    };

    // `rlib/jit.py isconstant`. Both layouts share one graph: the
    // `(name, args, result)` cache hits on the second layout. The
    // helper body calls the marked `majit_rlib::jit::isconstant`
    // intrinsic so the codewriter rewrites the call to `int_isconstant`.
    let isconstant = rtyper.lowlevel_helper_function_with_builder(
        "_ll_alloc_and_set_isconstant".to_string(),
        vec![LowLevelType::Signed],
        LowLevelType::Bool,
        |_r, _a, _res| {
            build_ll_alloc_and_set_isconstant_helper_graph("_ll_alloc_and_set_isconstant")
        },
    )?;

    // `@dont_look_inside` trampoline: one call of the original loop.
    let trampoline_name = format!("{nonnull_name}_trampoline");
    let trampoline = {
        let ptr = ptr_lltype.clone();
        let item = item_lltype.clone();
        let orig_fn = functionptr_const(rtyper, &nonnull)?;
        let name = trampoline_name.clone();
        rtyper.lowlevel_helper_function_with_builder(
            trampoline_name,
            vec![LowLevelType::Signed, item_lltype.clone()],
            ptr_lltype.clone(),
            move |_r, _a, _res| {
                build_ll_alloc_and_set_trampoline_helper_graph(
                    &name,
                    ptr.clone(),
                    item.clone(),
                    orig_fn.clone(),
                )
            },
        )?
    };

    // `look_inside_iff` wrapper. The jit arm's nonnull callee is this
    // graph, not the loop (`rlib/jit.py` `look_inside_iff`).
    let wrapper_name = format!("{nonnull_name}_look_inside_iff");
    let wrapper = {
        let ptr = ptr_lltype.clone();
        let item = item_lltype.clone();
        let orig_fn = functionptr_const(rtyper, &nonnull)?;
        let trampoline_fn = functionptr_const(rtyper, &trampoline)?;
        let isconstant_fn = functionptr_const(rtyper, &isconstant)?;
        let name = wrapper_name.clone();
        rtyper.lowlevel_helper_function_with_builder(
            wrapper_name,
            vec![LowLevelType::Signed, item_lltype.clone()],
            ptr_lltype.clone(),
            move |_r, _a, _res| {
                build_ll_alloc_and_set_look_inside_iff_helper_graph(
                    &name,
                    ptr.clone(),
                    item.clone(),
                    orig_fn.clone(),
                    trampoline_fn.clone(),
                    isconstant_fn.clone(),
                )
            },
        )?
    };

    // _ll_alloc_and_set_jit(LIST, count, item) (rlist.py).
    let jit = {
        let ptr = ptr_lltype.clone();
        let clear_fn = functionptr_const(rtyper, &clear)?;
        let nonnull_fn = functionptr_const(rtyper, &wrapper)?;
        let item = item_lltype.clone();
        rtyper.lowlevel_helper_function_with_builder(
            jit_name.to_string(),
            vec![LowLevelType::Signed, item_lltype.clone()],
            ptr_lltype.clone(),
            move |_r, _a, _res| {
                build_ll_alloc_and_set_jit_helper_graph(
                    jit_name,
                    ptr.clone(),
                    item.clone(),
                    clear_fn.clone(),
                    nonnull_fn.clone(),
                )
            },
        )?
    };

    // _ll_alloc_and_set_nojit(LIST, count, item) (rlist.py).
    let nojit = {
        let ptr = ptr_lltype.clone();
        let newlist_fn = functionptr_const(rtyper, &newlist)?;
        let arrayclear_fn = functionptr_const(rtyper, &arrayclear)?;
        let arrayfill_fn = functionptr_const(rtyper, &arrayfill)?;
        let item = item_lltype.clone();
        rtyper.lowlevel_helper_function_with_builder(
            nojit_name.to_string(),
            vec![LowLevelType::Signed, item_lltype.clone()],
            ptr_lltype.clone(),
            move |_r, _a, _res| {
                build_ll_alloc_and_set_nojit_helper_graph(
                    nojit_name,
                    layout,
                    ptr.clone(),
                    item.clone(),
                    newlist_fn.clone(),
                    arrayclear_fn.clone(),
                    arrayfill_fn.clone(),
                )
            },
        )?
    };

    let jit_fn = functionptr_const(rtyper, &jit)?;
    let nojit_fn = functionptr_const(rtyper, &nojit)?;
    build_ll_alloc_and_set_helper_graph(
        alloc_and_set_helper_name(layout),
        ptr_lltype,
        item_lltype,
        jit_fn,
        nojit_fn,
    )
}

/// Per-layout names for the five sub-helpers minted alongside the dispatch
/// helper. The `newlist` name matches [`rtype_newlist_layout`]'s so the shared
/// `ll_newlist` graph is reused; the alloc-family sub-helpers get the upstream
/// spellings (Fixed prefixes its own so the helper cache stays disjoint).
///
/// Returns `(newlist, clear, nonnull, jit, nojit)`.
fn alloc_and_set_sub_names(
    layout: ListLayout,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    match layout {
        ListLayout::Fixed => (
            "ll_fixed_newlist",
            "_ll_fixed_alloc_and_clear",
            "_ll_fixed_alloc_and_set_nonnull",
            "_ll_fixed_alloc_and_set_jit",
            "_ll_fixed_alloc_and_set_nojit",
        ),
        ListLayout::Resized => (
            "ll_newlist",
            "_ll_alloc_and_clear",
            "_ll_alloc_and_set_nonnull",
            "_ll_alloc_and_set_jit",
            "_ll_alloc_and_set_nojit",
        ),
    }
}

pub fn _null_of_type() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("_null_of_type"))
}

pub fn ll_null_item() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_null_item"))
}

#[allow(non_snake_case)]
pub fn listItemType() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("listItemType"))
}

pub fn ll_arraycopy() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_arraycopy"))
}

pub fn ll_arraymove() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_arraymove"))
}

pub fn ll_copy() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_copy"))
}

pub fn ll_list_is_true() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_list_is_true"))
}

pub fn ll_list_is_true_foldable() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_list_is_true_foldable"))
}

pub fn ll_append() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_append"))
}

pub fn ll_prepend() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_prepend"))
}

pub fn ll_concat() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_concat"))
}

pub fn ll_insert_nonneg() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_insert_nonneg"))
}

pub fn ll_pop_nonneg() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_pop_nonneg"))
}

pub fn ll_pop_default() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_pop_default"))
}

pub fn ll_pop_zero() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_pop_zero"))
}

pub fn ll_pop() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_pop"))
}

pub fn ll_delitem_nonneg() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_delitem_nonneg"))
}

pub fn ll_delitem() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_delitem"))
}

pub fn ll_extend() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_extend"))
}

pub fn ll_extend_with_str() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_extend_with_str"))
}

pub fn ll_extend_with_str_slice_startonly() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_extend_with_str_slice_startonly"))
}

pub fn ll_extend_with_str_slice_startstop() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_extend_with_str_slice_startstop"))
}

pub fn ll_extend_with_str_slice_minusone() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_extend_with_str_slice_minusone"))
}

pub fn ll_extend_with_char_count() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_extend_with_char_count"))
}

pub fn ll_listdelslice_startonly() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_listdelslice_startonly"))
}

pub fn ll_listdelslice_startstop() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_listdelslice_startstop"))
}

pub fn ll_listsetslice() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_listsetslice"))
}

pub fn listeq_unroll_case() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("listeq_unroll_case"))
}

pub fn ll_listeq() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_listeq"))
}

pub fn ll_listcontains() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_listcontains"))
}

pub fn ll_listindex() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_listindex"))
}

pub fn ll_listremove() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_listremove"))
}

pub fn ll_inplace_mul() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_inplace_mul"))
}

pub fn ll_mul() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_mul"))
}

pub fn ll_mul_loop() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_mul_loop"))
}

/// Synthesise `LLHelpers`-style `ll_fixed_length`
/// (`lltypesystem/rlist.py`):
///
/// ```python
/// def ll_fixed_length(l):
///     return len(l)
/// ```
///
/// Single-block graph: `getarraysize(l) -> Signed`. Unlike
/// `ll_strlen` (which `getsubstruct`s the nested `chars` array first),
/// the `FixedSizeListRepr` receiver IS the `Ptr(GcArray)`, so the
/// length is one op.
pub(crate) fn build_ll_fixed_length_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("l", ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_len = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(arg)],
        Hlvalue::Variable(v_len.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_len)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string()],
        func,
    ))
}

/// RPython `class ListRepr(AbstractListRepr, BaseListRepr)`
/// (`lltypesystem/rlist.py`):
///
/// ```python
/// def _setup_repr(self):
///     if 'item_repr' not in self.__dict__:
///         self.external_item_repr, self.item_repr = externalvsinternal(
///             self.rtyper, self._item_repr_computer(), gcref=True)
///     if isinstance(self.LIST, GcForwardReference):
///         ITEM = self.item_repr.lowleveltype
///         ITEMARRAY = self.get_itemarray_lowleveltype()
///         self.LIST.become(GcStruct("list", ("length", Signed),
///                                           ("items", Ptr(ITEMARRAY)),
///                                   adtmeths = ADTIList({...}),
///                                   hints = {'list': True}))
/// ```
///
/// Unlike [`FixedSizeListRepr`] (whose `LIST` is the bare
/// `GcArray(ITEM)`), the resized list wraps the array in a `length` +
/// `items` header struct so `append`/resize can grow `items` while the
/// `length` counter tracks the live element count. The low-level type
/// is therefore `Ptr(GcStruct("list", length, items))`.
///
/// This slice lands the data shape and `rtype_len`; the `ADTIList`
/// adtmeth table and the `append`/`getitem`/`setitem`/resize ops are
/// follow-on slices.
#[derive(Debug)]
pub struct ListRepr {
    state: ReprState,
    lltype: LowLevelType,
    /// `self.item_repr` (`rlist.py:111`) — the internal (gcref-wrapped)
    /// element repr.
    item_repr: Arc<dyn Repr>,
    /// `self.external_item_repr` (`lltypesystem/rlist.py:110`) — the
    /// external element repr `recast` ([`list_recast`]) converts a getitem
    /// result back to (a clone of `item_repr` for primitives, the concrete
    /// `InstanceRepr` for a gc-instance element list).
    external_item_repr: Arc<dyn Repr>,
}

impl ListRepr {
    pub fn new(rtyper: &Rc<RPythonTyper>, item_repr: Arc<dyn Repr>) -> Result<Self, TyperError> {
        // `externalvsinternal(rtyper, item_repr, gcref=True)` — same
        // gcref normalisation as `FixedSizeListRepr`: gc `InstanceRepr`
        // items become the generic `Ptr(OBJECT)` gcref so the array
        // element type is never a gc container.
        let (external_item_repr, internal) =
            crate::translator::rtyper::rclass::externalvsinternal(rtyper, item_repr, true)?;
        let item_lltype = internal.lowleveltype().clone();
        // upstream `get_itemarray_lowleveltype()` — `GcArray(ITEM)` (the
        // `ADTIFixedList` adtmeths it carries are unused until the array
        // ops land, so the bare array suffices for this slice).
        let itemarray = Array::gc(item_lltype);
        let items_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(itemarray),
        }));
        // upstream `GcStruct("list", ("length", Signed), ("items",
        // Ptr(ITEMARRAY)), hints={'list': True})`.
        let list_struct = Struct::gc_with_hints(
            "list",
            vec![
                ("length".to_string(), LowLevelType::Signed),
                ("items".to_string(), items_ptr),
            ],
            vec![("list".to_string(), ConstValue::Bool(true))],
        );
        let lltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(list_struct),
        }));
        Ok(ListRepr {
            state: ReprState::new(),
            lltype,
            item_repr: internal,
            external_item_repr,
        })
    }

    pub(crate) fn item_lowleveltype(&self) -> LowLevelType {
        self.item_repr.lowleveltype().clone()
    }
}

impl Repr for ListRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "ListRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::ListRepr
    }

    /// RPython `AbstractBaseListRepr.rtype_len(self, hop)`
    /// (`rlist.py`): the resized list takes the `ll_len`
    /// (non-foldable) branch, which reads the struct `length` field —
    /// `ll_length(l)` = `l.length` (`lltypesystem/rlist.py` ADTIList).
    fn rtype_len(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        let ptr_lltype = self.lltype.clone();
        let ptr_for_builder = ptr_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_length".to_string(),
            vec![ptr_lltype],
            LowLevelType::Signed,
            move |_rtyper, _args, _result| {
                build_ll_length_helper_graph("ll_length", ptr_for_builder.clone())
            },
        )?;
        hop.gendirectcall(&helper, vlist)
    }

    /// RPython `pair(AbstractBaseListRepr, IntegerRepr).rtype_getitem`
    /// (`rlist.py:247-267`) for the resized list. Shares the dispatch with
    /// [`FixedSizeListRepr::rtype_getitem`] (nonneg + `dum_nocheck` fast
    /// path, `ll_getitem_nonneg` → `l.ll_getitem_fast(index)`), but the
    /// resized receiver is the `Ptr(GcStruct("list", length, items))`
    /// header, so `ll_getitem_fast` (`lltypesystem/rlist.py`) reads
    /// the `items` array out of the struct first
    /// (`l.ll_items()[index]` = `getfield(l, "items")` then
    /// `getarrayitem`). The negative-index (`ll_getitem`) and `checkidx`
    /// (IndexError-raising `ll_getitem_*_checked`) helpers fold / window the
    /// index before dispatching to that fast helper — all selected by the
    /// shared [`list_rtype_getitem`].
    ///
    /// The upstream result `recast` (`rlist.py`) is applied via
    /// [`list_recast`] — identical to `FixedSizeListRepr`: an identity
    /// short-circuit for the primitive items a live subject builds, a
    /// `cast_pointer` downcast for a gc-instance element list.
    fn rtype_getitem(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_res = list_rtype_getitem(
            hop,
            self,
            ListLayout::Resized,
            &self.lltype,
            self.item_repr.lowleveltype(),
        )?;
        list_recast(hop, &self.item_repr, &self.external_item_repr, v_res).map(Some)
    }

    /// RPython `pair(AbstractBaseListRepr, IntegerRepr).rtype_setitem`
    /// (`rlist.py:272-284`) for the resized list. Shares the dispatch with
    /// [`FixedSizeListRepr::rtype_setitem`] (nonneg + `dum_nocheck`,
    /// `ll_setitem_nonneg` → `l.ll_setitem_fast(index, item)`), but the
    /// resized receiver reads the `items` array out of the
    /// `Ptr(GcStruct("list", length, items))` header first
    /// (`lltypesystem/rlist.py` `l.ll_items()[index] = item` =
    /// `getfield(l, "items")` then `setarrayitem`). The negative-index
    /// (`ll_setitem`) and `checkidx` (IndexError-raising
    /// `ll_setitem_*_checked`) helpers fold / window the index before
    /// dispatching to that fast helper — all selected by the shared
    /// [`list_rtype_setitem`].
    fn rtype_setitem(&self, hop: &HighLevelOp) -> RTypeResult {
        list_rtype_setitem(
            hop,
            self,
            ListLayout::Resized,
            &self.lltype,
            self.item_repr.lowleveltype(),
            self.item_repr.as_ref(),
        )
    }

    /// RPython `AbstractBaseListRepr.rtype_method_reverse(self, hop)`
    /// (`rlist.py:138-143`) — defined on the common base, so it applies to
    /// the resized [`ListRepr`] as well as [`FixedSizeListRepr`]. The body
    /// shape is identical (`inputargs(self)`, `exception_cannot_occur()`,
    /// `direct_call(ll_reverse, v_lst)`); only the lowered `ll_reverse`
    /// differs — the resized receiver reads `length`/`items` out of the
    /// `Ptr(GcStruct("list", length, items))` header
    /// ([`build_ll_reverse_resized_helper_graph`]) rather than `getarraysize`
    /// / bare `getarrayitem` on a `Ptr(GcArray)`.
    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        match method_name {
            // `is_null` is the lltype `_ptr` nullity probe on the list
            // header pointer — lower it as `ptr_iszero` (opimpl.py
            // `op_ptr_iszero`), matching the `ptr_method_is_null` bound
            // method the annotator seats on pointer-carrying receivers.
            "is_null" => {
                let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
                Ok(hop.genop("ptr_iszero", vlist, GenopResult::LLType(LowLevelType::Bool)))
            }
            "reverse" => {
                let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
                hop.exception_cannot_occur()?;
                let item_lltype = self.item_repr.lowleveltype().clone();
                let ptr_lltype = self.lltype.clone();
                let ptr_for_builder = ptr_lltype.clone();
                let item_for_builder = item_lltype.clone();
                let helper = hop.rtyper.lowlevel_helper_function_with_builder(
                    "ll_reverse".to_string(),
                    vec![ptr_lltype],
                    LowLevelType::Void,
                    move |_rtyper, _args, _result| {
                        build_ll_reverse_resized_helper_graph(
                            "ll_reverse",
                            ptr_for_builder.clone(),
                            item_for_builder.clone(),
                        )
                    },
                )?;
                hop.gendirectcall(&helper, vlist)
            }
            // RPython `AbstractListRepr.rtype_method_append(self, hop)`
            // (`rlist.py:185`): `v_lst, v_value = hop.inputargs(self,
            // self.item_repr); hop.exception_cannot_occur();
            // hop.gendirectcall(ll_append, v_lst, v_value)`. Resized-only:
            // `ll_append` grows the `items` array via `_ll_list_resize_ge`.
            "append" => {
                let vargs = hop.inputargs(vec![
                    ConvertedTo::Repr(self),
                    ConvertedTo::Repr(self.item_repr.as_ref()),
                ])?;
                hop.exception_cannot_occur()?;
                let ptr_lltype = self.lltype.clone();
                let item_lltype = self.item_repr.lowleveltype().clone();
                let items_ptr = items_array_ptr_lltype(&item_lltype);

                // Mint the sub-helpers in dependency order so each is
                // annotated before the helper that direct_calls it builds
                // its graph (`sub_helper_funcptr_constant` needs the callee
                // annotated): ll_arraycopy <- _ll_list_resize_ge, and
                // ll_setitem_fast + _ll_list_resize_ge <- ll_append.
                let arraycopy = {
                    let item = item_lltype.clone();
                    hop.rtyper.lowlevel_helper_function_with_builder(
                        "ll_arraycopy".to_string(),
                        vec![items_ptr.clone(), items_ptr.clone(), LowLevelType::Signed],
                        LowLevelType::Void,
                        move |_rtyper, _args, _result| {
                            build_ll_arraycopy_helper_graph("ll_arraycopy", item.clone())
                        },
                    )?
                };
                let setitem_fast = {
                    let ptr = ptr_lltype.clone();
                    let item = item_lltype.clone();
                    hop.rtyper.lowlevel_helper_function_with_builder(
                        "ll_setitem_fast".to_string(),
                        vec![
                            ptr_lltype.clone(),
                            LowLevelType::Signed,
                            item_lltype.clone(),
                        ],
                        LowLevelType::Void,
                        move |_rtyper, _args, _result| {
                            build_ll_setitem_fast_helper_graph(
                                "ll_setitem_fast",
                                ptr.clone(),
                                item.clone(),
                            )
                        },
                    )?
                };
                let resize_ge = {
                    let ptr = ptr_lltype.clone();
                    let item = item_lltype.clone();
                    hop.rtyper.lowlevel_helper_function_with_builder(
                        "_ll_list_resize_ge".to_string(),
                        vec![ptr_lltype.clone(), LowLevelType::Signed],
                        LowLevelType::Void,
                        move |rtyper, _args, _result| {
                            build_ll_list_resize_ge_helper_graph(
                                rtyper,
                                "_ll_list_resize_ge",
                                ptr.clone(),
                                item.clone(),
                                &arraycopy,
                            )
                        },
                    )?
                };
                let append = {
                    let ptr = ptr_lltype.clone();
                    let item = item_lltype.clone();
                    hop.rtyper.lowlevel_helper_function_with_builder(
                        "ll_append".to_string(),
                        vec![ptr_lltype.clone(), item_lltype.clone()],
                        LowLevelType::Void,
                        move |rtyper, _args, _result| {
                            build_ll_append_helper_graph(
                                rtyper,
                                "ll_append",
                                ptr.clone(),
                                item.clone(),
                                &resize_ge,
                                &setitem_fast,
                            )
                        },
                    )?
                };
                hop.gendirectcall(&append, vargs)
            }
            "extend" => {
                // RPython `AbstractListRepr.rtype_method_extend` (rlist.py):
                //
                // ```python
                // def rtype_method_extend(self, hop):
                //     v_lst1, v_lst2 = hop.inputargs(*hop.args_r)
                //     hop.exception_cannot_occur()
                //     hop.gendirectcall(ll_extend, v_lst1, v_lst2)
                // ```
                //
                // The second list keeps its OWN repr (`*hop.args_r`, NOT
                // `self`): `extend_from_slice`'s argument is a slice
                // (`FixedSizeListRepr`, bare `Ptr(GcArray)`) while the
                // receiver is the resized `ListRepr`. `ll_extend`
                // (rlist.py:782) is layout-polymorphic via the ADT methods;
                // here the source layout is read off `l2_layout`.
                let l2_repr = hop
                    .args_r
                    .borrow()
                    .get(1)
                    .and_then(|o| o.clone())
                    .ok_or_else(|| {
                        TyperError::message("list.extend: missing source-list repr".to_string())
                    })?;
                let l2_layout = match l2_repr.repr_class_id() {
                    super::pairtype::ReprClassId::FixedSizeListRepr => ListLayout::Fixed,
                    super::pairtype::ReprClassId::ListRepr => ListLayout::Resized,
                    other => {
                        return Err(TyperError::message(format!(
                            "list.extend: unsupported source repr {other:?}"
                        )));
                    }
                };
                let l2_lltype = l2_repr.lowleveltype().clone();
                let vargs = hop.inputargs(vec![
                    ConvertedTo::Repr(self),
                    ConvertedTo::Repr(l2_repr.as_ref()),
                ])?;
                hop.exception_cannot_occur()?;
                let ptr_lltype = self.lltype.clone();
                let item_lltype = self.item_repr.lowleveltype().clone();
                let items_ptr = items_array_ptr_lltype(&item_lltype);

                // Mint sub-helpers in dependency order (callee annotated
                // before the helper that direct_calls it builds its graph):
                // start=0 ll_arraycopy <- _ll_list_resize_ge (its grow copy);
                // general ll_arraycopy + _ll_list_resize_ge <- ll_extend.
                let arraycopy_grow = {
                    let item = item_lltype.clone();
                    hop.rtyper.lowlevel_helper_function_with_builder(
                        "ll_arraycopy".to_string(),
                        vec![items_ptr.clone(), items_ptr.clone(), LowLevelType::Signed],
                        LowLevelType::Void,
                        move |_rtyper, _args, _result| {
                            build_ll_arraycopy_helper_graph("ll_arraycopy", item.clone())
                        },
                    )?
                };
                let resize_ge = {
                    let ptr = ptr_lltype.clone();
                    let item = item_lltype.clone();
                    hop.rtyper.lowlevel_helper_function_with_builder(
                        "_ll_list_resize_ge".to_string(),
                        vec![ptr_lltype.clone(), LowLevelType::Signed],
                        LowLevelType::Void,
                        move |rtyper, _args, _result| {
                            build_ll_list_resize_ge_helper_graph(
                                rtyper,
                                "_ll_list_resize_ge",
                                ptr.clone(),
                                item.clone(),
                                &arraycopy_grow,
                            )
                        },
                    )?
                };
                // The general `rgc.ll_arraycopy(src, dst, src_start,
                // dst_start, length)` (rgc.py) — extend copies the source
                // into `l1.items[len1 ..]`, so `dst_start = len1 != 0` and the
                // start=0 specialisation used by append's resize does not fit.
                let arraycopy_range = {
                    let item = item_lltype.clone();
                    hop.rtyper.lowlevel_helper_function_with_builder(
                        "ll_arraycopy".to_string(),
                        vec![
                            items_ptr.clone(),
                            items_ptr.clone(),
                            LowLevelType::Signed,
                            LowLevelType::Signed,
                            LowLevelType::Signed,
                        ],
                        LowLevelType::Void,
                        move |_rtyper, _args, _result| {
                            build_ll_arraycopy_general_helper_graph("ll_arraycopy", item.clone())
                        },
                    )?
                };
                let extend = {
                    let ptr = ptr_lltype.clone();
                    let item = item_lltype.clone();
                    let l2 = l2_lltype.clone();
                    hop.rtyper.lowlevel_helper_function_with_builder(
                        "ll_extend".to_string(),
                        vec![ptr_lltype.clone(), l2_lltype.clone()],
                        LowLevelType::Void,
                        move |rtyper, _args, _result| {
                            build_ll_extend_helper_graph(
                                rtyper,
                                "ll_extend",
                                ptr.clone(),
                                l2.clone(),
                                l2_layout,
                                item.clone(),
                                &resize_ge,
                                &arraycopy_range,
                            )
                        },
                    )?
                };
                hop.gendirectcall(&extend, vargs)
            }
            _ => Err(TyperError::message(format!(
                "missing ListRepr.rtype_method_{method_name}"
            ))),
        }
    }

    /// RPython `AbstractBaseListRepr.rtype_bltn_list(self, hop)`
    /// (`rlist.py:118-122`) — `list(l)` copies the resized receiver into a
    /// fresh resized list via `ll_copy(RESLIST, l)`.
    fn rtype_bltn_list(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        hop.exception_is_here()?;
        rtype_bltn_list_via_ll_copy(
            hop,
            ListLayout::Resized,
            self.lltype.clone(),
            self.item_repr.lowleveltype().clone(),
            vlist,
        )
    }

    /// RPython `AbstractBaseListRepr.rtype_getslice(r_lst, hop)`
    /// (`rlist.py:409-414`) — `l[start:stop]` allocates a fresh resized list
    /// and copies the slice via the per-kind `ll_listslice_%s` helper.
    fn rtype_getslice(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_lst = hop.inputarg(ConvertedTo::Repr(self), 0)?;
        rtype_getslice_via_ll_listslice(
            hop,
            ListLayout::Resized,
            self.lltype.clone(),
            self.item_repr.lowleveltype().clone(),
            v_lst,
        )
    }

    /// RPython `BaseListRepr.make_iterator_repr(self, *variant)`
    /// (`lltypesystem/rlist.py`). The resized receiver is flagged so
    /// `ll_listnext` / `ll_revlistnext` read `length` via the struct header.
    #[expect(
        clippy::arc_with_non_send_sync,
        reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
    )]
    fn make_iterator_repr(
        &self,
        variant: &[String],
        foldable: bool,
    ) -> Result<Arc<dyn Repr>, TyperError> {
        // A resized list is always `mutated`, so `foldable` is false here; it is
        // threaded for signature parity and gated again by `list_is_fixed`.
        list_make_iterator_repr(
            self.lltype.clone(),
            self.item_repr.clone(),
            self.external_item_repr.clone(),
            false,
            foldable,
            variant,
        )
    }
}

/// Synthesise `ll_length(l) -> Signed` (`lltypesystem/rlist.py` ADTIList
/// `ll_length`):
///
/// ```python
/// def ll_length(l):
///     return l.length
/// ```
///
/// Single-block graph: `getfield(l, "length") -> Signed`. The receiver
/// is the `Ptr(GcStruct("list", ...))`, so the length is the struct's
/// `length` header field (vs `FixedSizeListRepr`'s `getarraysize`).
pub(crate) fn build_ll_length_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("l", ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_len = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(arg), void_field_const("length")],
        Hlvalue::Variable(v_len.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_len)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string()],
        func,
    ))
}

/// Synthesise `ll_fixed_getitem_fast` (`lltypesystem/rlist.py`):
///
/// ```python
/// def ll_fixed_getitem_fast(l, index):
///     ll_assert(index < len(l), "fixed getitem out of bounds")
///     return l[index]
/// ```
///
/// The `ll_assert` is a debug-only bound check (no production op); the
/// `FixedSizeListRepr` receiver IS the `Ptr(GcArray)`, so the body is the
/// single `getarrayitem(l, index) -> ITEM` op (unlike `ll_stritem_nonneg`,
/// which `getsubstruct`s the nested `chars` array first).
pub(crate) fn build_ll_fixed_getitem_fast_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let l_arg = variable_with_lltype("l", ptr_lltype);
    let index_arg = variable_with_lltype("index", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l_arg.clone()),
        Hlvalue::Variable(index_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", item_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_item = variable_with_lltype("item", item_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarrayitem",
        vec![Hlvalue::Variable(l_arg), Hlvalue::Variable(index_arg)],
        Hlvalue::Variable(v_item.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_item)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string(), "index".to_string()],
        func,
    ))
}

/// Synthesise `ll_getitem_foldable_nonneg` (`rlist.py`):
///
/// ```python
/// def ll_getitem_foldable_nonneg(l, index):
///     ll_assert(index >= 0, "unexpectedly negative list getitem index")
///     return l.ll_getitem_fast(index)
/// ll_getitem_foldable_nonneg.oopspec = 'list.getitem_foldable(l, index)'
/// ```
///
/// Identical body to [`build_ll_fixed_getitem_fast_helper_graph`] (the
/// `ll_assert` is a debug-only bound check), except the element read is the
/// FOLDABLE `getarrayitem_pure`. `rtype_getitem` (rlist.py) selects
/// this helper instead of `ll_fixed_getitem_fast` when `not
/// listitem.mutated`, so the immutable element load can be folded / CSE'd;
/// the `oopspec = 'list.getitem_foldable'` is realised here by the distinct
/// `getarrayitem_pure` opname, which `lloperation` marks `canfold=True`.
pub(crate) fn build_ll_fixed_getitem_fast_foldable_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let l_arg = variable_with_lltype("l", ptr_lltype);
    let index_arg = variable_with_lltype("index", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l_arg.clone()),
        Hlvalue::Variable(index_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", item_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_item = variable_with_lltype("item", item_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarrayitem_pure",
        vec![Hlvalue::Variable(l_arg), Hlvalue::Variable(index_arg)],
        Hlvalue::Variable(v_item.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_item)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string(), "index".to_string()],
        func,
    ))
}

/// Synthesise `ll_fixed_setitem_fast` (`lltypesystem/rlist.py`):
///
/// ```python
/// def ll_fixed_setitem_fast(l, index, item):
///     ll_assert(index < len(l), "fixed setitem out of bounds")
///     l[index] = item
/// ```
///
/// The `ll_assert` is a debug-only bound check (no production op); the
/// `FixedSizeListRepr` receiver IS the `Ptr(GcArray)`, so the body is the
/// single `setarrayitem(l, index, item)` op. The function falls off the
/// end (`return None`), so the returnblock receives the `Void` `None`
/// constant.
pub(crate) fn build_ll_fixed_setitem_fast_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let l_arg = variable_with_lltype("l", ptr_lltype);
    let index_arg = variable_with_lltype("index", LowLevelType::Signed);
    let item_arg = variable_with_lltype("item", item_lltype);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l_arg.clone()),
        Hlvalue::Variable(index_arg.clone()),
        Hlvalue::Variable(item_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_void = variable_with_lltype("v", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setarrayitem",
        vec![
            Hlvalue::Variable(l_arg),
            Hlvalue::Variable(index_arg),
            Hlvalue::Variable(item_arg),
        ],
        Hlvalue::Variable(v_void),
    ));
    let none_const = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    ));
    startblock.closeblock(vec![
        Link::new(vec![none_const], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string(), "index".to_string(), "item".to_string()],
        func,
    ))
}

/// Synthesise the resized-list `ll_getitem_fast`
/// (`lltypesystem/rlist.py`):
///
/// ```python
/// def ll_getitem_fast(l, index):
///     ll_assert(index < l.ll_length(), "getitem out of bounds")
///     return l.ll_items()[index]
/// ```
///
/// The `ll_assert` is a debug-only bound check (no production op). Unlike
/// [`build_ll_fixed_getitem_fast_helper_graph`] (whose receiver IS the
/// `Ptr(GcArray)`), the resized receiver is the
/// `Ptr(GcStruct("list", length, items))` header, so the body reads the
/// `items` array out of the struct first: `getfield(l, "items")` →
/// `getarrayitem(items, index) -> ITEM`.
pub(crate) fn build_ll_getitem_fast_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let l_arg = variable_with_lltype("l", ptr_lltype);
    let index_arg = variable_with_lltype("index", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l_arg.clone()),
        Hlvalue::Variable(index_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", item_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let items_ptr_lltype = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(Array::gc(item_lltype.clone())),
    }));
    let v_items = variable_with_lltype("items", items_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(l_arg), void_field_const("items")],
        Hlvalue::Variable(v_items.clone()),
    ));
    let v_item = variable_with_lltype("item", item_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarrayitem",
        vec![Hlvalue::Variable(v_items), Hlvalue::Variable(index_arg)],
        Hlvalue::Variable(v_item.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_item)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string(), "index".to_string()],
        func,
    ))
}

/// Synthesise the resized-list `ll_setitem_fast`
/// (`lltypesystem/rlist.py`):
///
/// ```python
/// def ll_setitem_fast(l, index, item):
///     ll_assert(index < l.ll_length(), "setitem out of bounds")
///     l.ll_items()[index] = item
/// ```
///
/// The `ll_assert` is a debug-only bound check (no production op). The
/// resized receiver reads the `items` array out of the
/// `Ptr(GcStruct("list", length, items))` header first:
/// `getfield(l, "items")` → `setarrayitem(items, index, item)`. The
/// function falls off the end (`return None`), so the returnblock receives
/// the `Void` `None` constant.
pub(crate) fn build_ll_setitem_fast_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let l_arg = variable_with_lltype("l", ptr_lltype);
    let index_arg = variable_with_lltype("index", LowLevelType::Signed);
    let item_arg = variable_with_lltype("item", item_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l_arg.clone()),
        Hlvalue::Variable(index_arg.clone()),
        Hlvalue::Variable(item_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let items_ptr_lltype = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(Array::gc(item_lltype)),
    }));
    let v_items = variable_with_lltype("items", items_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(l_arg), void_field_const("items")],
        Hlvalue::Variable(v_items.clone()),
    ));
    let v_void = variable_with_lltype("v", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setarrayitem",
        vec![
            Hlvalue::Variable(v_items),
            Hlvalue::Variable(index_arg),
            Hlvalue::Variable(item_arg),
        ],
        Hlvalue::Variable(v_void),
    ));
    let none_const = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    ));
    startblock.closeblock(vec![
        Link::new(vec![none_const], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string(), "index".to_string(), "item".to_string()],
        func,
    ))
}

/// Synthesise `ll_reverse` (`rlist.py`):
///
/// ```python
/// def ll_reverse(l):
///     length = l.ll_length()
///     i = 0
///     length_1_i = length-1-i
///     while i < length_1_i:
///         tmp = l.ll_getitem_fast(i)
///         l.ll_setitem_fast(i, l.ll_getitem_fast(length_1_i))
///         l.ll_setitem_fast(length_1_i, tmp)
///         i += 1
///         length_1_i -= 1
/// ```
///
/// In-place swap loop over the `FixedSizeListRepr` receiver (the bare
/// `Ptr(GcArray)`): `ll_length` / `ll_getitem_fast` / `ll_setitem_fast` each
/// collapse to a single `getarraysize` / `getarrayitem` / `setarrayitem` op
/// (no `items` header indirection, unlike the resized list). Four-block CFG:
/// - **startblock**: `getarraysize(l) -> length`, `int_sub(length, 1) ->
///   length_1_i` (the `length - 1 - i` initial folds `- i` against the
///   loop-seeding constant `i = 0`); links to `block_loop_cond(l, 0,
///   length_1_i)`.
/// - **block_loop_cond**: `int_lt(i, length_1_i) -> cond`. `True` →
///   `block_loop_body`; `False` → returnblock (`None`). The strict `<` leaves
///   an odd-length list's middle element in place.
/// - **block_loop_body**: read BOTH endpoints before writing either (so the
///   swap captures pre-swap values) — `getarrayitem(l, i) -> tmp`,
///   `getarrayitem(l, length_1_i) -> v`, `setarrayitem(l, i, v)`,
///   `setarrayitem(l, length_1_i, tmp)` — then `int_add(i, 1)`,
///   `int_sub(length_1_i, 1)`, back to `block_loop_cond`.
pub(crate) fn build_ll_reverse_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let signed_const = |n: i64| {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(n),
            LowLevelType::Signed,
        ))
    };
    let bool_const = |b: bool| {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };

    let l_arg = variable_with_lltype("l", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(l_arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // The loop blocks carry (l, i, length_1_i) as their own fresh inputargs.
    let l_cond = variable_with_lltype("l", ptr_lltype.clone());
    let i_cond = variable_with_lltype("i", LowLevelType::Signed);
    let j_cond = variable_with_lltype("length_1_i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(l_cond.clone()),
        Hlvalue::Variable(i_cond.clone()),
        Hlvalue::Variable(j_cond.clone()),
    ]);

    let l_body = variable_with_lltype("l", ptr_lltype);
    let i_body = variable_with_lltype("i", LowLevelType::Signed);
    let j_body = variable_with_lltype("length_1_i", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(l_body.clone()),
        Hlvalue::Variable(i_body.clone()),
        Hlvalue::Variable(j_body.clone()),
    ]);

    // startblock: length = getarraysize(l); length_1_i = length - 1.
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(l_arg.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let length_1_i = variable_with_lltype("length_1_i", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![Hlvalue::Variable(length), signed_const(1)],
        Hlvalue::Variable(length_1_i.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_arg),
                signed_const(0),
                Hlvalue::Variable(length_1_i),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_loop_cond: int_lt(i, length_1_i). True -> body; False -> return None.
    let cond = variable_with_lltype("cond", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_cond.clone()),
                Hlvalue::Variable(j_cond.clone()),
            ],
            Hlvalue::Variable(cond.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
    let none_const = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    ));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_cond),
                Hlvalue::Variable(i_cond),
                Hlvalue::Variable(j_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![none_const],
            Some(graph.returnblock.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_loop_body: read both endpoints, write them crossed, step indices.
    let tmp = variable_with_lltype("tmp", item_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(l_body.clone()),
                Hlvalue::Variable(i_body.clone()),
            ],
            Hlvalue::Variable(tmp.clone()),
        ));
    let v = variable_with_lltype("v", item_lltype);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(l_body.clone()),
                Hlvalue::Variable(j_body.clone()),
            ],
            Hlvalue::Variable(v.clone()),
        ));
    let w_i = variable_with_lltype("v", LowLevelType::Void);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(l_body.clone()),
                Hlvalue::Variable(i_body.clone()),
                Hlvalue::Variable(v),
            ],
            Hlvalue::Variable(w_i),
        ));
    let w_j = variable_with_lltype("v", LowLevelType::Void);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(l_body.clone()),
                Hlvalue::Variable(j_body.clone()),
                Hlvalue::Variable(tmp),
            ],
            Hlvalue::Variable(w_j),
        ));
    let i_next = variable_with_lltype("i", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_body.clone()), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    let j_next = variable_with_lltype("length_1_i", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_sub",
            vec![Hlvalue::Variable(j_body.clone()), signed_const(1)],
            Hlvalue::Variable(j_next.clone()),
        ));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_body),
                Hlvalue::Variable(i_next),
                Hlvalue::Variable(j_next),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string()],
        func,
    ))
}

/// Synthesise the resized-list `ll_reverse` (`rlist.py`):
///
/// ```python
/// def ll_reverse(l):
///     length = l.ll_length()
///     i = 0
///     length_1_i = length-1-i
///     while i < length_1_i:
///         tmp = l.ll_getitem_fast(i)
///         l.ll_setitem_fast(i, l.ll_getitem_fast(length_1_i))
///         l.ll_setitem_fast(length_1_i, tmp)
///         i += 1
///         length_1_i -= 1
/// ```
///
/// Same four-block swap loop as [`build_ll_reverse_helper_graph`], but the
/// resized receiver is the `Ptr(GcStruct("list", length, items))` header, so
/// the `ll_length` / `ll_getitem_fast` / `ll_setitem_fast` adtmeths read the
/// header rather than a bare array: `length` is `getfield(l, "length")` (vs
/// `getarraysize`), and each element access reads the `items` array out of
/// the struct first — `getfield(l, "items")` then `getarrayitem` /
/// `setarrayitem` (matching [`build_ll_getitem_fast_helper_graph`] /
/// [`build_ll_setitem_fast_helper_graph`]). The repeated `items` reads fold
/// in the malloc/CSE passes, mirroring upstream's per-adtmeth `l.ll_items()`.
pub(crate) fn build_ll_reverse_resized_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let signed_const = |n: i64| {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(n),
            LowLevelType::Signed,
        ))
    };
    let bool_const = |b: bool| {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };
    let items_ptr_lltype = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(Array::gc(item_lltype.clone())),
    }));

    let l_arg = variable_with_lltype("l", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(l_arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // The loop blocks carry (l, i, length_1_i) as their own fresh inputargs.
    let l_cond = variable_with_lltype("l", ptr_lltype.clone());
    let i_cond = variable_with_lltype("i", LowLevelType::Signed);
    let j_cond = variable_with_lltype("length_1_i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(l_cond.clone()),
        Hlvalue::Variable(i_cond.clone()),
        Hlvalue::Variable(j_cond.clone()),
    ]);

    let l_body = variable_with_lltype("l", ptr_lltype);
    let i_body = variable_with_lltype("i", LowLevelType::Signed);
    let j_body = variable_with_lltype("length_1_i", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(l_body.clone()),
        Hlvalue::Variable(i_body.clone()),
        Hlvalue::Variable(j_body.clone()),
    ]);

    // startblock: length = getfield(l, "length"); length_1_i = length - 1.
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(l_arg.clone()), void_field_const("length")],
        Hlvalue::Variable(length.clone()),
    ));
    let length_1_i = variable_with_lltype("length_1_i", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![Hlvalue::Variable(length), signed_const(1)],
        Hlvalue::Variable(length_1_i.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_arg),
                signed_const(0),
                Hlvalue::Variable(length_1_i),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_loop_cond: int_lt(i, length_1_i). True -> body; False -> return None.
    let cond = variable_with_lltype("cond", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_cond.clone()),
                Hlvalue::Variable(j_cond.clone()),
            ],
            Hlvalue::Variable(cond.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
    let none_const = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    ));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_cond),
                Hlvalue::Variable(i_cond),
                Hlvalue::Variable(j_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![none_const],
            Some(graph.returnblock.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_loop_body: read BOTH endpoints (each via getfield "items" +
    // getarrayitem) before writing either, then step the indices.
    let items_tmp = variable_with_lltype("items", items_ptr_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(l_body.clone()), void_field_const("items")],
            Hlvalue::Variable(items_tmp.clone()),
        ));
    let tmp = variable_with_lltype("tmp", item_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(items_tmp),
                Hlvalue::Variable(i_body.clone()),
            ],
            Hlvalue::Variable(tmp.clone()),
        ));
    let items_v = variable_with_lltype("items", items_ptr_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(l_body.clone()), void_field_const("items")],
            Hlvalue::Variable(items_v.clone()),
        ));
    let v = variable_with_lltype("v", item_lltype);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(items_v),
                Hlvalue::Variable(j_body.clone()),
            ],
            Hlvalue::Variable(v.clone()),
        ));
    let items_wi = variable_with_lltype("items", items_ptr_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(l_body.clone()), void_field_const("items")],
            Hlvalue::Variable(items_wi.clone()),
        ));
    let w_i = variable_with_lltype("v", LowLevelType::Void);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(items_wi),
                Hlvalue::Variable(i_body.clone()),
                Hlvalue::Variable(v),
            ],
            Hlvalue::Variable(w_i),
        ));
    let items_wj = variable_with_lltype("items", items_ptr_lltype);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(l_body.clone()), void_field_const("items")],
            Hlvalue::Variable(items_wj.clone()),
        ));
    let w_j = variable_with_lltype("v", LowLevelType::Void);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(items_wj),
                Hlvalue::Variable(j_body.clone()),
                Hlvalue::Variable(tmp),
            ],
            Hlvalue::Variable(w_j),
        ));
    let i_next = variable_with_lltype("i", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_body.clone()), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    let j_next = variable_with_lltype("length_1_i", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_sub",
            vec![Hlvalue::Variable(j_body.clone()), signed_const(1)],
            Hlvalue::Variable(j_next.clone()),
        ));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_body),
                Hlvalue::Variable(i_next),
                Hlvalue::Variable(j_next),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string()],
        func,
    ))
}

/// Build the `Ptr(GcArray(ITEM))` low-level type of the resized list's
/// `items` field — must match [`ListRepr::new`]'s `Array::gc(item_lltype)`
/// exactly so the `malloc_varsize` result and the `items` field unify.
fn items_array_ptr_lltype(item_lltype: &LowLevelType) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(Array::gc(item_lltype.clone())),
    }))
}

fn none_void_const() -> Hlvalue {
    Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    ))
}

/// Synthesise `rgc.ll_arraycopy` (`rpython/rlib/rgc.py`), specialised
/// to `source_start == dest_start == 0` (the only shape
/// [`build_ll_list_resize_ge_helper_graph`] needs — a resize always copies
/// the live prefix to offset 0 of a fresh array):
///
/// ```python
/// def ll_arraycopy(source, dest, 0, 0, length):
///     i = 0
///     while i < length:
///         dest[i] = source[i]
///         i += 1
/// ```
///
/// Three-block loop over bare `Ptr(GcArray(ITEM))` operands (mirrors the
/// [`build_ll_reverse_helper_graph`] CFG shape). `length == 0` (the
/// first append into an empty list) makes `int_lt(0, 0)` false, so the
/// copy is a no-op — this is why the resize caller can drop upstream's
/// `if before_len:` guard against copying the prebuilt-empty array.
fn build_ll_arraycopy_helper_graph(
    name: &str,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let items_ptr = items_array_ptr_lltype(&item_lltype);

    let src = variable_with_lltype("source", items_ptr.clone());
    let dst = variable_with_lltype("dest", items_ptr.clone());
    let length = variable_with_lltype("length", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(src.clone()),
        Hlvalue::Variable(dst.clone()),
        Hlvalue::Variable(length.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Loop blocks carry (source, dest, length, i) as fresh inputargs.
    let src_c = variable_with_lltype("source", items_ptr.clone());
    let dst_c = variable_with_lltype("dest", items_ptr.clone());
    let len_c = variable_with_lltype("length", LowLevelType::Signed);
    let i_c = variable_with_lltype("i", LowLevelType::Signed);
    let block_cond = Block::shared(vec![
        Hlvalue::Variable(src_c.clone()),
        Hlvalue::Variable(dst_c.clone()),
        Hlvalue::Variable(len_c.clone()),
        Hlvalue::Variable(i_c.clone()),
    ]);

    let src_b = variable_with_lltype("source", items_ptr.clone());
    let dst_b = variable_with_lltype("dest", items_ptr.clone());
    let len_b = variable_with_lltype("length", LowLevelType::Signed);
    let i_b = variable_with_lltype("i", LowLevelType::Signed);
    let block_body = Block::shared(vec![
        Hlvalue::Variable(src_b.clone()),
        Hlvalue::Variable(dst_b.clone()),
        Hlvalue::Variable(len_b.clone()),
        Hlvalue::Variable(i_b.clone()),
    ]);

    // startblock: i = 0.
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src),
                Hlvalue::Variable(dst),
                Hlvalue::Variable(length),
                signed_const(0),
            ],
            Some(block_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_cond: int_lt(i, length). True -> body; False -> return None.
    let cond = variable_with_lltype("cond", LowLevelType::Bool);
    block_cond.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![
            Hlvalue::Variable(i_c.clone()),
            Hlvalue::Variable(len_c.clone()),
        ],
        Hlvalue::Variable(cond.clone()),
    ));
    block_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
    block_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_c),
                Hlvalue::Variable(dst_c),
                Hlvalue::Variable(len_c),
                Hlvalue::Variable(i_c),
            ],
            Some(block_body.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![none_void_const()],
            Some(graph.returnblock.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_body: dest[i] = source[i]; i += 1.
    let v = variable_with_lltype("v", item_lltype);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "getarrayitem",
        vec![
            Hlvalue::Variable(src_b.clone()),
            Hlvalue::Variable(i_b.clone()),
        ],
        Hlvalue::Variable(v.clone()),
    ));
    let store_void = variable_with_lltype("v", LowLevelType::Void);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "setarrayitem",
        vec![
            Hlvalue::Variable(dst_b.clone()),
            Hlvalue::Variable(i_b.clone()),
            Hlvalue::Variable(v),
        ],
        Hlvalue::Variable(store_void),
    ));
    let i_next = variable_with_lltype("i", LowLevelType::Signed);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![Hlvalue::Variable(i_b), signed_const(1)],
        Hlvalue::Variable(i_next.clone()),
    ));
    block_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_b),
                Hlvalue::Variable(dst_b),
                Hlvalue::Variable(len_b),
                Hlvalue::Variable(i_next),
            ],
            Some(block_cond),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "source".to_string(),
            "dest".to_string(),
            "length".to_string(),
        ],
        func,
    ))
}

/// Synthesise the general `rgc.ll_arraycopy(source, dest, source_start,
/// dest_start, length)` (`rpython/rlib/rgc.py`) as an element loop:
///
/// ```python
/// def ll_arraycopy(source, dest, source_start, dest_start, length):
///     i = 0
///     while i < length:
///         dest[dest_start + i] = source[source_start + i]
///         i += 1
/// ```
///
/// Unlike [`build_ll_arraycopy_helper_graph`] (specialised to
/// `source_start == dest_start == 0` for the append/resize grow copy), this
/// general form offsets both ends — `ll_extend` copies the source into
/// `l1.items[len1 ..]`, so `dest_start = len1`.
///
/// The element loop below IS upstream's `rpython/rlib/rgc.py`, the
/// arm reached when `gc_writebarrier_before_copy` answers False and the
/// bulk `raw_memcopy` is therefore not expressible.  Upstream emits no
/// barrier on that arm either — the call exists only to decide between the
/// two — so none is owed here.  What is missing is the other arm: no
/// `raw_memcopy` fast path is emitted, and the minted helper carries no
/// `list.ll_arraycopy` oopspec (`OS_ARRAYCOPY`).
pub(crate) fn build_ll_arraycopy_general_helper_graph(
    name: &str,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let items_ptr = items_array_ptr_lltype(&item_lltype);

    let src = variable_with_lltype("source", items_ptr.clone());
    let dst = variable_with_lltype("dest", items_ptr.clone());
    let src_start = variable_with_lltype("source_start", LowLevelType::Signed);
    let dst_start = variable_with_lltype("dest_start", LowLevelType::Signed);
    let length = variable_with_lltype("length", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(src.clone()),
        Hlvalue::Variable(dst.clone()),
        Hlvalue::Variable(src_start.clone()),
        Hlvalue::Variable(dst_start.clone()),
        Hlvalue::Variable(length.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Loop blocks carry (source, dest, source_start, dest_start, length, i).
    let src_c = variable_with_lltype("source", items_ptr.clone());
    let dst_c = variable_with_lltype("dest", items_ptr.clone());
    let sst_c = variable_with_lltype("source_start", LowLevelType::Signed);
    let dst_start_c = variable_with_lltype("dest_start", LowLevelType::Signed);
    let len_c = variable_with_lltype("length", LowLevelType::Signed);
    let i_c = variable_with_lltype("i", LowLevelType::Signed);
    let block_cond = Block::shared(vec![
        Hlvalue::Variable(src_c.clone()),
        Hlvalue::Variable(dst_c.clone()),
        Hlvalue::Variable(sst_c.clone()),
        Hlvalue::Variable(dst_start_c.clone()),
        Hlvalue::Variable(len_c.clone()),
        Hlvalue::Variable(i_c.clone()),
    ]);

    let src_b = variable_with_lltype("source", items_ptr.clone());
    let dst_b = variable_with_lltype("dest", items_ptr.clone());
    let sst_b = variable_with_lltype("source_start", LowLevelType::Signed);
    let dst_start_b = variable_with_lltype("dest_start", LowLevelType::Signed);
    let len_b = variable_with_lltype("length", LowLevelType::Signed);
    let i_b = variable_with_lltype("i", LowLevelType::Signed);
    let block_body = Block::shared(vec![
        Hlvalue::Variable(src_b.clone()),
        Hlvalue::Variable(dst_b.clone()),
        Hlvalue::Variable(sst_b.clone()),
        Hlvalue::Variable(dst_start_b.clone()),
        Hlvalue::Variable(len_b.clone()),
        Hlvalue::Variable(i_b.clone()),
    ]);

    // startblock: i = 0.
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src),
                Hlvalue::Variable(dst),
                Hlvalue::Variable(src_start),
                Hlvalue::Variable(dst_start),
                Hlvalue::Variable(length),
                signed_const(0),
            ],
            Some(block_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_cond: int_lt(i, length). True -> body; False -> return None.
    let cond = variable_with_lltype("cond", LowLevelType::Bool);
    block_cond.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![
            Hlvalue::Variable(i_c.clone()),
            Hlvalue::Variable(len_c.clone()),
        ],
        Hlvalue::Variable(cond.clone()),
    ));
    block_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
    block_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_c),
                Hlvalue::Variable(dst_c),
                Hlvalue::Variable(sst_c),
                Hlvalue::Variable(dst_start_c),
                Hlvalue::Variable(len_c),
                Hlvalue::Variable(i_c),
            ],
            Some(block_body.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![none_void_const()],
            Some(graph.returnblock.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_body: dest[dest_start + i] = source[source_start + i]; i += 1.
    let si = variable_with_lltype("si", LowLevelType::Signed);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![
            Hlvalue::Variable(sst_b.clone()),
            Hlvalue::Variable(i_b.clone()),
        ],
        Hlvalue::Variable(si.clone()),
    ));
    let di = variable_with_lltype("di", LowLevelType::Signed);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![
            Hlvalue::Variable(dst_start_b.clone()),
            Hlvalue::Variable(i_b.clone()),
        ],
        Hlvalue::Variable(di.clone()),
    ));
    let v = variable_with_lltype("v", item_lltype);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "getarrayitem",
        vec![Hlvalue::Variable(src_b.clone()), Hlvalue::Variable(si)],
        Hlvalue::Variable(v.clone()),
    ));
    let store_void = variable_with_lltype("v", LowLevelType::Void);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "setarrayitem",
        vec![
            Hlvalue::Variable(dst_b.clone()),
            Hlvalue::Variable(di),
            Hlvalue::Variable(v),
        ],
        Hlvalue::Variable(store_void),
    ));
    let i_next = variable_with_lltype("i", LowLevelType::Signed);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![Hlvalue::Variable(i_b), signed_const(1)],
        Hlvalue::Variable(i_next.clone()),
    ));
    block_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(src_b),
                Hlvalue::Variable(dst_b),
                Hlvalue::Variable(sst_b),
                Hlvalue::Variable(dst_start_b),
                Hlvalue::Variable(len_b),
                Hlvalue::Variable(i_next),
            ],
            Some(block_cond),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "source".to_string(),
            "dest".to_string(),
            "source_start".to_string(),
            "dest_start".to_string(),
            "length".to_string(),
        ],
        func,
    ))
}

/// Which way `rgc.ll_arraymove`'s element loop walks the overlap.
#[derive(Clone, Copy)]
enum ArraymoveDirection {
    /// Upstream's `delta < 0`: `i` runs `0..length`, low indices first.
    Ascending,
    /// Upstream's `delta > 0`: `i` runs `length - 1` down to `0`, high
    /// indices first.  `delta == 0` also lands here, where source and
    /// destination coincide and either order is correct.
    Descending,
}

/// Emit one arm of `ll_arraymove`'s element loop: a condition block testing
/// the induction variable and a body block copying
/// `array[source_start + i]` into `array[dest_start + i]` before stepping
/// it.  Returns the condition block, which is the arm's entry.
///
/// Both arms carry `(array, source_start, dest_start, length, i)` and differ
/// only in their loop bound and their step.
fn build_ll_arraymove_arm(
    returnblock: &BlockRef,
    items_ptr: &LowLevelType,
    item_lltype: &LowLevelType,
    direction: ArraymoveDirection,
) -> BlockRef {
    let array_c = variable_with_lltype("array", items_ptr.clone());
    let src_start_c = variable_with_lltype("source_start", LowLevelType::Signed);
    let dst_start_c = variable_with_lltype("dest_start", LowLevelType::Signed);
    let length_c = variable_with_lltype("length", LowLevelType::Signed);
    let i_c = variable_with_lltype("i", LowLevelType::Signed);
    let block_cond = Block::shared(vec![
        Hlvalue::Variable(array_c.clone()),
        Hlvalue::Variable(src_start_c.clone()),
        Hlvalue::Variable(dst_start_c.clone()),
        Hlvalue::Variable(length_c.clone()),
        Hlvalue::Variable(i_c.clone()),
    ]);

    let array_b = variable_with_lltype("array", items_ptr.clone());
    let src_start_b = variable_with_lltype("source_start", LowLevelType::Signed);
    let dst_start_b = variable_with_lltype("dest_start", LowLevelType::Signed);
    let length_b = variable_with_lltype("length", LowLevelType::Signed);
    let i_b = variable_with_lltype("i", LowLevelType::Signed);
    let block_body = Block::shared(vec![
        Hlvalue::Variable(array_b.clone()),
        Hlvalue::Variable(src_start_b.clone()),
        Hlvalue::Variable(dst_start_b.clone()),
        Hlvalue::Variable(length_b.clone()),
        Hlvalue::Variable(i_b.clone()),
    ]);

    let (bound_opname, bound_rhs) = match direction {
        ArraymoveDirection::Ascending => ("int_lt", Hlvalue::Variable(length_c.clone())),
        ArraymoveDirection::Descending => ("int_ge", signed_const(0)),
    };
    let cond = variable_with_lltype("cond", LowLevelType::Bool);
    block_cond.borrow_mut().operations.push(SpaceOperation::new(
        bound_opname,
        vec![Hlvalue::Variable(i_c.clone()), bound_rhs],
        Hlvalue::Variable(cond.clone()),
    ));
    block_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
    block_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(array_c),
                Hlvalue::Variable(src_start_c),
                Hlvalue::Variable(dst_start_c),
                Hlvalue::Variable(length_c),
                Hlvalue::Variable(i_c),
            ],
            Some(block_body.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![none_void_const()],
            Some(returnblock.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let source_index = variable_with_lltype("source_index", LowLevelType::Signed);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![
            Hlvalue::Variable(src_start_b.clone()),
            Hlvalue::Variable(i_b.clone()),
        ],
        Hlvalue::Variable(source_index.clone()),
    ));
    let dest_index = variable_with_lltype("dest_index", LowLevelType::Signed);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![
            Hlvalue::Variable(dst_start_b.clone()),
            Hlvalue::Variable(i_b.clone()),
        ],
        Hlvalue::Variable(dest_index.clone()),
    ));
    let item = variable_with_lltype("item", item_lltype.clone());
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "getarrayitem",
        vec![
            Hlvalue::Variable(array_b.clone()),
            Hlvalue::Variable(source_index),
        ],
        Hlvalue::Variable(item.clone()),
    ));
    let store_void = variable_with_lltype("store", LowLevelType::Void);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "setarrayitem",
        vec![
            Hlvalue::Variable(array_b.clone()),
            Hlvalue::Variable(dest_index),
            Hlvalue::Variable(item),
        ],
        Hlvalue::Variable(store_void),
    ));
    let step_opname = match direction {
        ArraymoveDirection::Ascending => "int_add",
        ArraymoveDirection::Descending => "int_sub",
    };
    let i_next = variable_with_lltype("i", LowLevelType::Signed);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        step_opname,
        vec![Hlvalue::Variable(i_b), signed_const(1)],
        Hlvalue::Variable(i_next.clone()),
    ));
    block_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(array_b),
                Hlvalue::Variable(src_start_b),
                Hlvalue::Variable(dst_start_b),
                Hlvalue::Variable(length_b),
                Hlvalue::Variable(i_next),
            ],
            Some(block_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    block_cond
}

/// Synthesise `rgc.ll_arraymove` (`rpython/rlib/rgc.py`, `ll_arraymove`)
/// with the upstream four-argument shape.
///
/// Upstream's manual-loop path picks its direction from the sign of `delta`
/// at runtime:
///
/// ```python
/// delta = dest_start - source_start
/// if delta < 0:
///     i = source_start
///     stop = source_start + length
///     while i < stop:
///         copy_item(array, array, i, i + delta)
///         i += 1
/// elif delta > 0:
///     i = source_start + length
///     while i > source_start:
///         i -= 1
///         copy_item(array, array, i, i + delta)
/// ```
///
/// The emitted loops rebase that induction variable to zero — `i` runs over
/// `0..length` and both ends are offset — instead of carrying `delta` and
/// `stop`.  The two are equal at every index; the rebase exists because the
/// caller reconstructed `(source_start, dest_start, length)` from MIR and
/// never materialised a `delta`.  Because the dispatch is a runtime test,
/// one helper serves both directions and the rtyper's
/// `(name, args, result)` helper cache cannot hand a right-move producer a
/// left-only loop.
///
/// Two parts of `ll_arraymove` are NOT emitted, and each is a named gap
/// rather than a simplification:
///
/// * The `length <= 1` head.  Upstream's own comment says it is there "to
///   ensure that we get a proper effectinfo.write_descrs_arrays", not for
///   speed.  The arms below reach that same effectinfo through their own
///   per-item `setarrayitem`, which upstream's memmove arm does not have.
///   The `list.ll_arraymove` oopspec this helper is marked with is what
///   `_handle_list_call` reads to emit OS_ARRAYMOVE; the same pairing
///   `ll_arraycopy` is still missing.
/// * The `must_split_gc_address_space()` dispatch and the
///   `raw_memmove_no_free` fast path it guards.  The body below IS
///   upstream's split-address-space branch, taken unconditionally.  That is
///   also the branch on which upstream does not emit
///   `gc_writebarrier_before_move`: the barrier sits on the `elif` that
///   guards the memmove, and here each per-item `setarrayitem` marks its
///   own card.  Emitting the barrier would only mean something alongside a
///   memmove arm for it to gate.
pub(crate) fn build_ll_arraymove_helper_graph(
    name: &str,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let items_ptr = items_array_ptr_lltype(&item_lltype);

    let array = variable_with_lltype("array", items_ptr.clone());
    let src_start = variable_with_lltype("source_start", LowLevelType::Signed);
    let dst_start = variable_with_lltype("dest_start", LowLevelType::Signed);
    let length = variable_with_lltype("length", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(array.clone()),
        Hlvalue::Variable(src_start.clone()),
        Hlvalue::Variable(dst_start.clone()),
        Hlvalue::Variable(length.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let ascending = build_ll_arraymove_arm(
        &graph.returnblock,
        &items_ptr,
        &item_lltype,
        ArraymoveDirection::Ascending,
    );
    let descending = build_ll_arraymove_arm(
        &graph.returnblock,
        &items_ptr,
        &item_lltype,
        ArraymoveDirection::Descending,
    );

    let delta = variable_with_lltype("delta", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![
            Hlvalue::Variable(dst_start.clone()),
            Hlvalue::Variable(src_start.clone()),
        ],
        Hlvalue::Variable(delta.clone()),
    ));
    let goes_up = variable_with_lltype("goes_up", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(delta), signed_const(0)],
        Hlvalue::Variable(goes_up.clone()),
    ));
    // The descending arm enters at `length - 1`.  Its start index is
    // computed here because exit links carry arguments, not operations.
    let last_index = variable_with_lltype("last_index", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![Hlvalue::Variable(length.clone()), signed_const(1)],
        Hlvalue::Variable(last_index.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(goes_up));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(array.clone()),
                Hlvalue::Variable(src_start.clone()),
                Hlvalue::Variable(dst_start.clone()),
                Hlvalue::Variable(length.clone()),
                signed_const(0),
            ],
            Some(ascending),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(array),
                Hlvalue::Variable(src_start),
                Hlvalue::Variable(dst_start),
                Hlvalue::Variable(length),
                Hlvalue::Variable(last_index),
            ],
            Some(descending),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec![
            "array".to_string(),
            "source_start".to_string(),
            "dest_start".to_string(),
            "length".to_string(),
        ],
        func,
    ))
}

/// Synthesise `_ll_list_resize_ge` fused with `_ll_list_resize_hint_really`
/// (`lltypesystem/rlist.py` + `:200-239`), specialised to the
/// grow-only `append` path (`overallocate=True`, `newsize > before_len > 0`
/// or `before_len == 0`):
///
/// ```python
/// def _ll_list_resize_ge(l, newsize):           # overallocate=True
///     if len(l.items) < newsize:
///         some = (3 if newsize < 9 else 6) + (newsize >> 3)
///         new_allocated = newsize + some
///         newitems = malloc(LIST.items.TO, new_allocated)
///         rgc.ll_arraycopy(l.items, newitems, 0, 0, l.length)
///         l.items = newitems
///     l.length = newsize
/// ```
///
/// Since `_ll_list_resize_ge` is only ever called with `newsize` larger
/// than the current length, `before_len < newsize` always holds, so
/// upstream's `p = min(before_len, newsize)` collapses to `before_len`
/// and the `if before_len:` guard collapses (a `before_len == 0` copy is a
/// no-op, see [`build_ll_arraycopy_helper_graph`]). The `newsize <= 0`
/// (empty-array reset) and `overallocate=False` branches — reached only by
/// `_ll_list_resize_le` / `_ll_list_resize` — are deferred.
///
/// CFG: start (capacity check) -> grow (overallocation `some` branch) ->
/// alloc (malloc + arraycopy + `l.items = newitems`) -> tail
/// (`l.length = newsize`); the no-grow start edge skips straight to tail.
fn build_ll_list_resize_ge_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    arraycopy: &LowLevelFunction,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};

    let items_ptr = items_array_ptr_lltype(&item_lltype);
    let arraycopy_const = sub_helper_funcptr_constant(rtyper, arraycopy)?;

    let l_arg = variable_with_lltype("l", ptr_lltype.clone());
    let newsize_arg = variable_with_lltype("newsize", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l_arg.clone()),
        Hlvalue::Variable(newsize_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // grow(l, newsize): chooses the overallocation `some` base.
    let l_grow = variable_with_lltype("l", ptr_lltype.clone());
    let newsize_grow = variable_with_lltype("newsize", LowLevelType::Signed);
    let block_grow = Block::shared(vec![
        Hlvalue::Variable(l_grow.clone()),
        Hlvalue::Variable(newsize_grow.clone()),
    ]);
    // alloc(l, newsize, some_base): malloc + copy + l.items = newitems.
    let l_alloc = variable_with_lltype("l", ptr_lltype.clone());
    let newsize_alloc = variable_with_lltype("newsize", LowLevelType::Signed);
    let some_base = variable_with_lltype("some_base", LowLevelType::Signed);
    let block_alloc = Block::shared(vec![
        Hlvalue::Variable(l_alloc.clone()),
        Hlvalue::Variable(newsize_alloc.clone()),
        Hlvalue::Variable(some_base.clone()),
    ]);
    // tail(l, newsize): l.length = newsize.
    let l_tail = variable_with_lltype("l", ptr_lltype.clone());
    let newsize_tail = variable_with_lltype("newsize", LowLevelType::Signed);
    let block_tail = Block::shared(vec![
        Hlvalue::Variable(l_tail.clone()),
        Hlvalue::Variable(newsize_tail.clone()),
    ]);

    // startblock: cond = len(l.items) < newsize.
    let items0 = variable_with_lltype("items", items_ptr.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(l_arg.clone()), void_field_const("items")],
        Hlvalue::Variable(items0.clone()),
    ));
    let allocated = variable_with_lltype("allocated", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(items0)],
        Hlvalue::Variable(allocated.clone()),
    ));
    let need_grow = variable_with_lltype("cond", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![
            Hlvalue::Variable(allocated),
            Hlvalue::Variable(newsize_arg.clone()),
        ],
        Hlvalue::Variable(need_grow.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(need_grow));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_arg.clone()),
                Hlvalue::Variable(newsize_arg.clone()),
            ],
            Some(block_grow.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(l_arg), Hlvalue::Variable(newsize_arg)],
            Some(block_tail.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // grow: some_base = 3 if newsize < 9 else 6.
    let small = variable_with_lltype("small", LowLevelType::Bool);
    block_grow.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(newsize_grow.clone()), signed_const(9)],
        Hlvalue::Variable(small.clone()),
    ));
    block_grow.borrow_mut().exitswitch = Some(Hlvalue::Variable(small));
    block_grow.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_grow.clone()),
                Hlvalue::Variable(newsize_grow.clone()),
                signed_const(3),
            ],
            Some(block_alloc.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(l_grow),
                Hlvalue::Variable(newsize_grow),
                signed_const(6),
            ],
            Some(block_alloc.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // alloc: new_allocated = newsize + some_base + (newsize >> 3);
    //      newitems = malloc(LIST.items.TO, new_allocated);
    //      ll_arraycopy(l.items, newitems, l.length); l.items = newitems.
    let shifted = variable_with_lltype("shifted", LowLevelType::Signed);
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_rshift",
            vec![Hlvalue::Variable(newsize_alloc.clone()), signed_const(3)],
            Hlvalue::Variable(shifted.clone()),
        ));
    let some = variable_with_lltype("some", LowLevelType::Signed);
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(some_base), Hlvalue::Variable(shifted)],
            Hlvalue::Variable(some.clone()),
        ));
    let new_allocated = variable_with_lltype("new_allocated", LowLevelType::Signed);
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Variable(newsize_alloc.clone()),
                Hlvalue::Variable(some),
            ],
            Hlvalue::Variable(new_allocated.clone()),
        ));
    let olditems = variable_with_lltype("items", items_ptr.clone());
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(l_alloc.clone()),
                void_field_const("items"),
            ],
            Hlvalue::Variable(olditems.clone()),
        ));
    let before_len = variable_with_lltype("before_len", LowLevelType::Signed);
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(l_alloc.clone()),
                void_field_const("length"),
            ],
            Hlvalue::Variable(before_len.clone()),
        ));
    let newitems = variable_with_lltype("newitems", items_ptr.clone());
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "malloc_varsize",
            vec![
                lowlevel_type_const(LowLevelType::Array(Box::new(Array::gc(
                    item_lltype.clone(),
                )))),
                gc_flavor_const()?,
                Hlvalue::Variable(new_allocated),
            ],
            Hlvalue::Variable(newitems.clone()),
        ));
    let copy_void = variable_with_lltype("v", LowLevelType::Void);
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(arraycopy_const),
                Hlvalue::Variable(olditems),
                Hlvalue::Variable(newitems.clone()),
                Hlvalue::Variable(before_len),
            ],
            Hlvalue::Variable(copy_void),
        ));
    let set_items_void = variable_with_lltype("v", LowLevelType::Void);
    block_alloc
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(l_alloc.clone()),
                void_field_const("items"),
                Hlvalue::Variable(newitems),
            ],
            Hlvalue::Variable(set_items_void),
        ));
    block_alloc.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(l_alloc), Hlvalue::Variable(newsize_alloc)],
            Some(block_tail.clone()),
            None,
        )
        .into_ref(),
    ]);

    // tail: l.length = newsize.
    let set_len_void = variable_with_lltype("v", LowLevelType::Void);
    block_tail.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(l_tail),
            void_field_const("length"),
            Hlvalue::Variable(newsize_tail),
        ],
        Hlvalue::Variable(set_len_void),
    ));
    block_tail.closeblock(vec![
        Link::new(
            vec![none_void_const()],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string(), "newsize".to_string()],
        func,
    ))
}

/// Synthesise `ll_append` (`lltypesystem/rlist.py:588`):
///
/// ```python
/// def ll_append(l, newitem):
///     length = l.length
///     _ll_list_resize_ge(l, length + 1)   # see build_ll_list_resize_ge
///     l.ll_setitem_fast(length, newitem)  # see build_ll_setitem_fast
/// ```
///
/// Single block: read `length`, `direct_call` the resize helper (which sets
/// `l.length = length + 1` and ensures capacity), then `direct_call` the
/// resized `ll_setitem_fast` to write `items[length] = newitem`.
fn build_ll_append_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    resize_ge: &LowLevelFunction,
    setitem_fast: &LowLevelFunction,
) -> Result<PyGraph, TyperError> {
    let resize_const = sub_helper_funcptr_constant(rtyper, resize_ge)?;
    let setitem_const = sub_helper_funcptr_constant(rtyper, setitem_fast)?;

    let l_arg = variable_with_lltype("l", ptr_lltype);
    let item_arg = variable_with_lltype("newitem", item_lltype);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l_arg.clone()),
        Hlvalue::Variable(item_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(l_arg.clone()), void_field_const("length")],
        Hlvalue::Variable(length.clone()),
    ));
    let newsize = variable_with_lltype("newsize", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![Hlvalue::Variable(length.clone()), signed_const(1)],
        Hlvalue::Variable(newsize.clone()),
    ));
    let resize_void = variable_with_lltype("v", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(resize_const),
            Hlvalue::Variable(l_arg.clone()),
            Hlvalue::Variable(newsize),
        ],
        Hlvalue::Variable(resize_void),
    ));
    let setitem_void = variable_with_lltype("v", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(setitem_const),
            Hlvalue::Variable(l_arg),
            Hlvalue::Variable(length),
            Hlvalue::Variable(item_arg),
        ],
        Hlvalue::Variable(setitem_void),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![none_void_const()],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string(), "newitem".to_string()],
        func,
    ))
}

/// Synthesise `ll_extend` (`rpython/rtyper/rlist.py`):
///
/// ```python
/// def ll_extend(l1, l2):
///     len1 = l1.ll_length()
///     len2 = l2.ll_length()
///     newlength = ovfcheck(len1 + len2)   # OverflowError -> MemoryError
///     l1._ll_resize_ge(newlength)
///     ll_arraycopy(l2, l1, 0, len1, len2)
/// ```
///
/// `l1` is the resized receiver (`Ptr(GcStruct("list", length, items))`);
/// `l2` is the source list whose layout is `l2_layout` — for
/// `extend_from_slice` it is a slice (`FixedSizeListRepr`, bare
/// `Ptr(GcArray)`), so `len2 = getarraysize(l2)` and the items array IS `l2`;
/// a resized source reads `length`/`items` out of its header. `len1` is read
/// BEFORE the resize (which overwrites `l1.length`), and `l1.items` AFTER
/// (the resize may reallocate it). The copy lands the source elements at
/// `l1.items[len1 ..]` via the general [`build_ll_arraycopy_general_helper_graph`].
/// `ovfcheck(len1 + len2)` is modelled: both addends are non-negative list
/// lengths, so the signed sum overflows iff `newlength < len1`, which branches
/// to a MemoryError raise. `rtype_method_extend` still declares
/// `exception_cannot_occur` — MemoryError is an implicit (always-possible)
/// exception, not a Python-level one the caller's flow graph handles.
#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
)]
fn build_ll_extend_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    ptr_lltype: LowLevelType,
    l2_lltype: LowLevelType,
    l2_layout: ListLayout,
    item_lltype: LowLevelType,
    resize_ge: &LowLevelFunction,
    arraycopy_general: &LowLevelFunction,
) -> Result<PyGraph, TyperError> {
    let resize_const = sub_helper_funcptr_constant(rtyper, resize_ge)?;
    let copy_const = sub_helper_funcptr_constant(rtyper, arraycopy_general)?;
    let items_ptr = items_array_ptr_lltype(&item_lltype);

    let l1_arg = variable_with_lltype("l1", ptr_lltype.clone());
    let l2_arg = variable_with_lltype("l2", l2_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l1_arg.clone()),
        Hlvalue::Variable(l2_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // len1 = l1.length (read BEFORE the resize overwrites it).
    let len1 = variable_with_lltype("len1", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![
            Hlvalue::Variable(l1_arg.clone()),
            void_field_const("length"),
        ],
        Hlvalue::Variable(len1.clone()),
    ));
    // len2 = l2 length (per layout).
    let len2 = variable_with_lltype("len2", LowLevelType::Signed);
    match l2_layout {
        ListLayout::Fixed => {
            startblock.borrow_mut().operations.push(SpaceOperation::new(
                "getarraysize",
                vec![Hlvalue::Variable(l2_arg.clone())],
                Hlvalue::Variable(len2.clone()),
            ));
        }
        ListLayout::Resized => {
            startblock.borrow_mut().operations.push(SpaceOperation::new(
                "getfield",
                vec![
                    Hlvalue::Variable(l2_arg.clone()),
                    void_field_const("length"),
                ],
                Hlvalue::Variable(len2.clone()),
            ));
        }
    }
    // newlength = len1 + len2.
    let newlength = variable_with_lltype("newlength", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![
            Hlvalue::Variable(len1.clone()),
            Hlvalue::Variable(len2.clone()),
        ],
        Hlvalue::Variable(newlength.clone()),
    ));
    // ovfcheck(len1 + len2): len1/len2 are list lengths (>= 0), so the signed
    // sum overflows iff it wraps below len1 — that path raises MemoryError.
    let overflow = variable_with_lltype("overflow", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![
            Hlvalue::Variable(newlength.clone()),
            Hlvalue::Variable(len1.clone()),
        ],
        Hlvalue::Variable(overflow.clone()),
    ));

    // continue block (overflow false): resize l1 + copy l2 into it.
    let l1_c = variable_with_lltype("l1", ptr_lltype.clone());
    let l2_c = variable_with_lltype("l2", l2_lltype.clone());
    let len1_c = variable_with_lltype("len1", LowLevelType::Signed);
    let len2_c = variable_with_lltype("len2", LowLevelType::Signed);
    let newlength_c = variable_with_lltype("newlength", LowLevelType::Signed);
    let block_continue = Block::shared(vec![
        Hlvalue::Variable(l1_c.clone()),
        Hlvalue::Variable(l2_c.clone()),
        Hlvalue::Variable(len1_c.clone()),
        Hlvalue::Variable(len2_c.clone()),
        Hlvalue::Variable(newlength_c.clone()),
    ]);

    // overflow true -> raise MemoryError; false -> block_continue.
    let exc_args = exception_args("MemoryError")?;
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(overflow));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(l1_arg),
                Hlvalue::Variable(l2_arg),
                Hlvalue::Variable(len1),
                Hlvalue::Variable(len2),
                Hlvalue::Variable(newlength),
            ],
            Some(block_continue.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // _ll_list_resize_ge(l1, newlength) — sets l1.length = newlength, grows items.
    let resize_void = variable_with_lltype("v", LowLevelType::Void);
    block_continue
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(resize_const),
                Hlvalue::Variable(l1_c.clone()),
                Hlvalue::Variable(newlength_c),
            ],
            Hlvalue::Variable(resize_void),
        ));
    // items1 = l1.items (read AFTER the resize: it may have reallocated).
    let items1 = variable_with_lltype("items1", items_ptr.clone());
    block_continue
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(l1_c), void_field_const("items")],
            Hlvalue::Variable(items1.clone()),
        ));
    // items2 = l2 items (per layout): the slice IS its own array.
    let items2_hlv = match l2_layout {
        ListLayout::Fixed => Hlvalue::Variable(l2_c),
        ListLayout::Resized => {
            let items2 = variable_with_lltype("items2", items_ptr.clone());
            block_continue
                .borrow_mut()
                .operations
                .push(SpaceOperation::new(
                    "getfield",
                    vec![Hlvalue::Variable(l2_c), void_field_const("items")],
                    Hlvalue::Variable(items2.clone()),
                ));
            Hlvalue::Variable(items2)
        }
    };
    // ll_arraycopy(items2, items1, 0, len1, len2).
    let copy_void = variable_with_lltype("v", LowLevelType::Void);
    block_continue
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(copy_const),
                items2_hlv,
                Hlvalue::Variable(items1),
                signed_const(0),
                Hlvalue::Variable(len1_c),
                Hlvalue::Variable(len2_c),
            ],
            Hlvalue::Variable(copy_void),
        ));
    block_continue.closeblock(vec![
        Link::new(
            vec![none_void_const()],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l1".to_string(), "l2".to_string()],
        func,
    ))
}

/// Synthesise `RESLIST.ll_newlist(length)`
/// (`rpython/rtyper/lltypesystem/rlist.py` `ll_newlist`): allocate a fresh
/// list holding `length` items. The resized layout mallocs the
/// `Ptr(GcStruct("list", length, items))` header plus a `malloc_varsize`
/// items `GcArray`, then stores both fields; the fixed layout is the bare
/// `Ptr(GcArray)`, so it is the `malloc_varsize` alone. `malloc_varsize`
/// zero-fills the items; [`build_ll_copy_helper_graph`] overwrites them.
pub(crate) fn build_ll_newlist_helper_graph(
    name: &str,
    layout: ListLayout,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    build_ll_newlist_helper_graph_inner(name, layout, ptr_lltype, item_lltype, false)
}

/// Synthesise the resized-list-only `ll_newlist_hint` helper.  Kept separate
/// from [`build_ll_newlist_helper_graph`] at the public surface because these
/// are distinct ADT methods upstream and therefore distinct memoized helper
/// graphs.
fn build_ll_newlist_hint_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    build_ll_newlist_helper_graph_inner(name, ListLayout::Resized, ptr_lltype, item_lltype, true)
}

fn build_ll_newlist_helper_graph_inner(
    name: &str,
    layout: ListLayout,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    length_is_hint: bool,
) -> Result<PyGraph, TyperError> {
    use crate::translator::rtyper::rmodel::{gc_flavor_const, lowlevel_type_const};
    let length_arg = variable_with_lltype("length", LowLevelType::Signed);
    let startblock = Block::shared(vec![Hlvalue::Variable(length_arg.clone())]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let array_type = LowLevelType::Array(Box::new(Array::gc(item_lltype.clone())));

    if length_is_hint {
        // lltypesystem/rlist.py `ll_newlist_hint` begins with
        // `ll_assert(lengthhint >= 0, "negative list length")`.
        let nonnegative = variable_with_lltype("nonnegative", LowLevelType::Bool);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "int_ge",
            vec![
                Hlvalue::Variable(length_arg.clone()),
                constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
            ],
            Hlvalue::Variable(nonnegative.clone()),
        ));
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "debug_assert",
            vec![
                Hlvalue::Variable(nonnegative),
                constant_with_lltype(
                    ConstValue::byte_str("negative list length"),
                    LowLevelType::Void,
                ),
            ],
            Hlvalue::Variable(variable_with_lltype("v", LowLevelType::Void)),
        ));
    }

    let new_lst = match layout {
        ListLayout::Fixed => {
            let l = variable_with_lltype("l", ptr_lltype.clone());
            startblock.borrow_mut().operations.push(SpaceOperation::new(
                "malloc_varsize",
                vec![
                    lowlevel_type_const(array_type),
                    gc_flavor_const()?,
                    Hlvalue::Variable(length_arg),
                ],
                Hlvalue::Variable(l.clone()),
            ));
            l
        }
        ListLayout::Resized => {
            let LowLevelType::Ptr(ptr) = &ptr_lltype else {
                return Err(TyperError::message(
                    "build_ll_newlist_helper_graph: resized list lltype is not Ptr",
                ));
            };
            let inner_struct = match &ptr.TO {
                PtrTarget::Struct(body) => body.clone(),
                other => {
                    return Err(TyperError::message(format!(
                        "build_ll_newlist_helper_graph: resized Ptr target must be Struct, got {other:?}"
                    )));
                }
            };
            let header = variable_with_lltype("l", ptr_lltype.clone());
            startblock.borrow_mut().operations.push(SpaceOperation::new(
                "malloc",
                vec![
                    lowlevel_type_const(LowLevelType::Struct(Box::new(inner_struct))),
                    gc_flavor_const()?,
                ],
                Hlvalue::Variable(header.clone()),
            ));
            let items = variable_with_lltype("items", items_array_ptr_lltype(&item_lltype));
            startblock.borrow_mut().operations.push(SpaceOperation::new(
                "malloc_varsize",
                vec![
                    lowlevel_type_const(array_type),
                    gc_flavor_const()?,
                    Hlvalue::Variable(length_arg.clone()),
                ],
                Hlvalue::Variable(items.clone()),
            ));
            startblock.borrow_mut().operations.push(SpaceOperation::new(
                "setfield",
                vec![
                    Hlvalue::Variable(header.clone()),
                    void_field_const("length"),
                    if length_is_hint {
                        signed_const(0)
                    } else {
                        Hlvalue::Variable(length_arg)
                    },
                ],
                Hlvalue::Variable(variable_with_lltype("v", LowLevelType::Void)),
            ));
            startblock.borrow_mut().operations.push(SpaceOperation::new(
                "setfield",
                vec![
                    Hlvalue::Variable(header.clone()),
                    void_field_const("items"),
                    Hlvalue::Variable(items),
                ],
                Hlvalue::Variable(variable_with_lltype("v", LowLevelType::Void)),
            ));
            header
        }
    };
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(new_lst)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["length".to_string()],
        func,
    ))
}

/// Emit the `_ll_zero_or_null(item)` op chain (`rlist.py`) onto
/// `block`, returning a Bool var that is `True` when `item` IS zero/null:
///
/// ```python
/// def _ll_zero_or_null(item):
///     T = typeOf(item)
///     if T is Char or T is UniChar:
///         check = ord(item)
///     elif isinstance(T, Number):
///         check = widen(item)
///     else:
///         check = item
///     return not check
/// ```
///
/// The `check`/`not check` shape is realised per element type: a Char/UniChar
/// is cast to int then `int_is_true` (matching `not ord(item)`); an integer
/// is tested with `<prefix>is_true` directly (`not widen(item)`); a Ptr with
/// `ptr_nonzero`.  `_ll_zero_or_null` on a Float/SingleFloat Primitive is
/// also `not item` (`float_is_true`), so `-0.0` would clear to `+0.0`; these
/// graphs lower Rust `vec![x; n]` / `[x; N]`, which store the fill's bit
/// pattern, so the Float arm compares bits (`convert_float_bytes_to_longlong`)
/// and only `+0.0` takes the cleared arm.  SingleFloat widens first with
/// `cast_singlefloat_to_float` (`bhimpl_cast_singlefloat_to_float`), which
/// keeps `-0.0`, then uses the same bit compare.  LongFloat has no Rust
/// source type and stays on `float_is_true`.  Every arm produces the
/// "nonzero" boolean, which is then negated with `bool_not` to yield the
/// "zero/null" result the caller branches on.
fn emit_zero_or_null(block: &BlockRef, item_lltype: &LowLevelType, item: Hlvalue) -> Variable {
    let (opname, operand) = match item_lltype {
        LowLevelType::Char => {
            let as_int = variable_with_lltype("check", LowLevelType::Signed);
            block.borrow_mut().operations.push(SpaceOperation::new(
                "cast_char_to_int",
                vec![item],
                Hlvalue::Variable(as_int.clone()),
            ));
            ("int_is_true", Hlvalue::Variable(as_int))
        }
        LowLevelType::UniChar => {
            let as_int = variable_with_lltype("check", LowLevelType::Signed);
            block.borrow_mut().operations.push(SpaceOperation::new(
                "cast_unichar_to_int",
                vec![item],
                Hlvalue::Variable(as_int.clone()),
            ));
            ("int_is_true", Hlvalue::Variable(as_int))
        }
        LowLevelType::Float => {
            // `_ll_zero_or_null` (`rlist.py`): Float is a Primitive, so
            // `not item`. IEEE `not -0.0` is true and would take the
            // cleared arm (`+0.0`); compare bits so only `+0.0` clears.
            let bits = variable_with_lltype("check", LowLevelType::SignedLongLong);
            block.borrow_mut().operations.push(SpaceOperation::new(
                "convert_float_bytes_to_longlong",
                vec![item],
                Hlvalue::Variable(bits.clone()),
            ));
            ("llong_is_true", Hlvalue::Variable(bits))
        }
        LowLevelType::SingleFloat => {
            // Same bit test as Float. `cast_singlefloat_to_float`
            // (`bhimpl_cast_singlefloat_to_float`) widens f32→f64 and
            // keeps `-0.0`; rtyper has no single-float bits llop.
            let as_float = variable_with_lltype("check", LowLevelType::Float);
            block.borrow_mut().operations.push(SpaceOperation::new(
                "cast_singlefloat_to_float",
                vec![item],
                Hlvalue::Variable(as_float.clone()),
            ));
            let bits = variable_with_lltype("check", LowLevelType::SignedLongLong);
            block.borrow_mut().operations.push(SpaceOperation::new(
                "convert_float_bytes_to_longlong",
                vec![Hlvalue::Variable(as_float)],
                Hlvalue::Variable(bits.clone()),
            ));
            ("llong_is_true", Hlvalue::Variable(bits))
        }
        LowLevelType::LongFloat => ("float_is_true", item),
        LowLevelType::Ptr(_) | LowLevelType::Address => ("ptr_nonzero", item),
        // Signed/Unsigned/LongLong/etc.: `not widen(item)` — the widened
        // integer is-true test.  `int_is_true` covers the machine-word ints
        // the list-repeat subjects reach here with.
        _ => ("int_is_true", item),
    };
    let nonzero = variable_with_lltype("nonzero", LowLevelType::Bool);
    block.borrow_mut().operations.push(SpaceOperation::new(
        opname,
        vec![operand],
        Hlvalue::Variable(nonzero.clone()),
    ));
    let is_zero = variable_with_lltype("is_zero_or_null", LowLevelType::Bool);
    block.borrow_mut().operations.push(SpaceOperation::new(
        "bool_not",
        vec![Hlvalue::Variable(nonzero)],
        Hlvalue::Variable(is_zero.clone()),
    ));
    is_zero
}

/// Read `l.ll_items()` for `layout` onto `block`, returning the items-array
/// ptr operand `rgc.ll_arrayclear` / `ll_arrayfill` receive.  A
/// `FixedSizeListRepr` value IS the array ptr (no read); the resized header
/// reaches its array through `getfield(l, "items")`.
fn emit_list_items_read(
    block: &BlockRef,
    layout: ListLayout,
    item_lltype: &LowLevelType,
    l: &Variable,
) -> Hlvalue {
    match layout {
        ListLayout::Fixed => Hlvalue::Variable(l.clone()),
        ListLayout::Resized => {
            let items = variable_with_lltype("items", items_array_ptr_lltype(item_lltype));
            block.borrow_mut().operations.push(SpaceOperation::new(
                "getfield",
                vec![Hlvalue::Variable(l.clone()), void_field_const("items")],
                Hlvalue::Variable(items.clone()),
            ));
            Hlvalue::Variable(items)
        }
    }
}

/// Synthesise `ll_alloc_and_set(LIST, count, item)` (`rlist.py`):
///
/// ```python
/// def ll_alloc_and_set(LIST, count, item):
///     count = int_force_ge_zero(count)
///     if jit.we_are_jitted():
///         return _ll_alloc_and_set_jit(LIST, count, item)
///     else:
///         return _ll_alloc_and_set_nojit(LIST, count, item)
/// ```
///
/// The startblock clamps `count` with `int_force_ge_zero`, then branches on
/// the identity-bearing `we_are_jitted()` `SpecTag` exitswitch (the append
/// helper's shape): true → `direct_call` the jit arm, false → the nojit arm.
/// `LIST` is folded into the callee identities, so only `(count, item)` cross
/// the call.
#[allow(clippy::too_many_arguments)]
fn build_ll_alloc_and_set_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    jit_fn: Constant,
    nojit_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let count_arg = variable_with_lltype("count", LowLevelType::Signed);
    let item_arg = variable_with_lltype("item", item_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(count_arg.clone()),
        Hlvalue::Variable(item_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // count = int_force_ge_zero(count).
    let count_clamped = variable_with_lltype("count", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_force_ge_zero",
        vec![Hlvalue::Variable(count_arg)],
        Hlvalue::Variable(count_clamped.clone()),
    ));

    // jit arm: return _ll_alloc_and_set_jit(count, item).
    let jit_count = variable_with_lltype("count", LowLevelType::Signed);
    let jit_item = variable_with_lltype("item", item_lltype.clone());
    let block_jit = Block::shared(vec![
        Hlvalue::Variable(jit_count.clone()),
        Hlvalue::Variable(jit_item.clone()),
    ]);
    let jit_res = variable_with_lltype("result", ptr_lltype.clone());
    block_jit.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(jit_fn),
            Hlvalue::Variable(jit_count),
            Hlvalue::Variable(jit_item),
        ],
        Hlvalue::Variable(jit_res.clone()),
    ));
    block_jit.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(jit_res)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    // nojit arm: return _ll_alloc_and_set_nojit(count, item).
    let nj_count = variable_with_lltype("count", LowLevelType::Signed);
    let nj_item = variable_with_lltype("item", item_lltype);
    let block_nojit = Block::shared(vec![
        Hlvalue::Variable(nj_count.clone()),
        Hlvalue::Variable(nj_item.clone()),
    ]);
    let nj_res = variable_with_lltype("result", ptr_lltype);
    block_nojit
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(nojit_fn),
                Hlvalue::Variable(nj_count),
                Hlvalue::Variable(nj_item),
            ],
            Hlvalue::Variable(nj_res.clone()),
        ));
    block_nojit.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(nj_res)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    startblock.borrow_mut().exitswitch = Some(Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::SpecTag(WE_ARE_JITTED_TAG_ID),
        LowLevelType::Bool,
    )));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(count_clamped.clone()),
                Hlvalue::Variable(item_arg.clone()),
            ],
            Some(block_jit),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(count_clamped),
                Hlvalue::Variable(item_arg),
            ],
            Some(block_nojit),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["count".to_string(), "item".to_string()],
        func,
    ))
}

/// Synthesise `_ll_alloc_and_set_jit(LIST, count, item)` (`rlist.py`):
///
/// ```python
/// def _ll_alloc_and_set_jit(LIST, count, item):
///     if _ll_zero_or_null(item):
///         return _ll_alloc_and_clear(LIST, count)
///     else:
///         return _ll_alloc_and_set_nonnull(LIST, count, item)
/// ```
///
/// Startblock runs the [`emit_zero_or_null`] op chain and branches on it:
/// true (zero/null) → `direct_call` the clear arm `(count)`; false → the
/// nonnull arm `(count, item)`.
fn build_ll_alloc_and_set_jit_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    clear_fn: Constant,
    nonnull_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let count_arg = variable_with_lltype("count", LowLevelType::Signed);
    let item_arg = variable_with_lltype("item", item_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(count_arg.clone()),
        Hlvalue::Variable(item_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // clear arm: return _ll_alloc_and_clear(count).
    let cl_count = variable_with_lltype("count", LowLevelType::Signed);
    let block_clear = Block::shared(vec![Hlvalue::Variable(cl_count.clone())]);
    let cl_res = variable_with_lltype("result", ptr_lltype.clone());
    block_clear
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![Hlvalue::Constant(clear_fn), Hlvalue::Variable(cl_count)],
            Hlvalue::Variable(cl_res.clone()),
        ));
    block_clear.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(cl_res)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    // nonnull arm: return _ll_alloc_and_set_nonnull(count, item).
    let nn_count = variable_with_lltype("count", LowLevelType::Signed);
    let nn_item = variable_with_lltype("item", item_lltype.clone());
    let block_nonnull = Block::shared(vec![
        Hlvalue::Variable(nn_count.clone()),
        Hlvalue::Variable(nn_item.clone()),
    ]);
    let nn_res = variable_with_lltype("result", ptr_lltype);
    block_nonnull
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(nonnull_fn),
                Hlvalue::Variable(nn_count),
                Hlvalue::Variable(nn_item),
            ],
            Hlvalue::Variable(nn_res.clone()),
        ));
    block_nonnull.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(nn_res)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let is_zero = emit_zero_or_null(
        &startblock,
        &item_lltype,
        Hlvalue::Variable(item_arg.clone()),
    );
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_zero));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(count_arg.clone())],
            Some(block_clear),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(count_arg), Hlvalue::Variable(item_arg)],
            Some(block_nonnull),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["count".to_string(), "item".to_string()],
        func,
    ))
}

/// Synthesise `_ll_alloc_and_set_nojit(LIST, count, item)`
/// (`rlist.py:494-508`):
///
/// ```python
/// def _ll_alloc_and_set_nojit(LIST, count, item):
///     l = LIST.ll_newlist(count)
///     if _ll_zero_or_null(item):
///         if not malloc_zero_filled:
///             if not isinstance(typeOf(item), Ptr):
///                 rgc.ll_arrayclear(l.ll_items())
///         return l
///     rgc.ll_arrayfill(l.ll_items(), item)
///     return l
/// ```
///
/// `l = ll_newlist(count)` runs in the startblock, which then branches on the
/// [`emit_zero_or_null`] chain.  The `not malloc_zero_filled` guard collapses
/// to always-do (a translated program has `malloc_zero_filled=False`, the
/// `mallocstr` precedent), and the `not isinstance(typeOf(item), Ptr)` guard
/// is a BUILD-TIME element-type check: a Ptr item's array was already zeroed
/// by `ll_newlist`'s `zero_gc_pointers_inside`, so the zero arm just returns
/// `l`; a non-Ptr item emits `rgc.ll_arrayclear(items)`.  The nonzero arm
/// always emits `rgc.ll_arrayfill(items, item)`.
#[allow(clippy::too_many_arguments)]
fn build_ll_alloc_and_set_nojit_helper_graph(
    name: &str,
    layout: ListLayout,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    newlist_fn: Constant,
    arrayclear_fn: Constant,
    arrayfill_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let count_arg = variable_with_lltype("count", LowLevelType::Signed);
    let item_arg = variable_with_lltype("item", item_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(count_arg.clone()),
        Hlvalue::Variable(item_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // l = LIST.ll_newlist(count).
    let l = variable_with_lltype("l", ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![Hlvalue::Constant(newlist_fn), Hlvalue::Variable(count_arg)],
        Hlvalue::Variable(l.clone()),
    ));

    // zero/null arm: (build-time) arrayclear unless the element is a Ptr; then
    // return l.  Threads `l` through the link and reads `items` in the arm.
    let cl_l = variable_with_lltype("l", ptr_lltype.clone());
    let cl_item = variable_with_lltype("item", item_lltype.clone());
    let block_zero = Block::shared(vec![
        Hlvalue::Variable(cl_l.clone()),
        Hlvalue::Variable(cl_item.clone()),
    ]);
    // upstream `not isinstance(typeOf(item), Ptr)` — only a `Ptr` element had
    // its slots pre-zeroed by `ll_newlist`'s `zero_gc_pointers_inside`, so only
    // a `Ptr` item skips `ll_arrayclear`. `Address` is not a `Ptr`, so it (and
    // every primitive) still clears.
    let item_is_ptr = matches!(item_lltype, LowLevelType::Ptr(_));
    if !item_is_ptr {
        let items = emit_list_items_read(&block_zero, layout, &item_lltype, &cl_l);
        let cl_void = variable_with_lltype("v", LowLevelType::Void);
        block_zero.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            vec![Hlvalue::Constant(arrayclear_fn), items],
            Hlvalue::Variable(cl_void),
        ));
    }
    block_zero.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(cl_l)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    // nonzero arm: rgc.ll_arrayfill(l.ll_items(), item); return l.
    let nz_l = variable_with_lltype("l", ptr_lltype.clone());
    let nz_item = variable_with_lltype("item", item_lltype.clone());
    let block_fill = Block::shared(vec![
        Hlvalue::Variable(nz_l.clone()),
        Hlvalue::Variable(nz_item.clone()),
    ]);
    let fill_items = emit_list_items_read(&block_fill, layout, &item_lltype, &nz_l);
    let fill_void = variable_with_lltype("v", LowLevelType::Void);
    block_fill.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(arrayfill_fn),
            fill_items,
            Hlvalue::Variable(nz_item),
        ],
        Hlvalue::Variable(fill_void),
    ));
    block_fill.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(nz_l)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let is_zero = emit_zero_or_null(
        &startblock,
        &item_lltype,
        Hlvalue::Variable(item_arg.clone()),
    );
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_zero));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l.clone()),
                Hlvalue::Variable(item_arg.clone()),
            ],
            Some(block_zero),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(l), Hlvalue::Variable(item_arg)],
            Some(block_fill),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["count".to_string(), "item".to_string()],
        func,
    ))
}

/// Synthesise `_ll_alloc_and_clear(LIST, count)` (`rlist.py`,
/// `@jit.oopspec("newlist_clear(count)")`):
///
/// ```python
/// def _ll_alloc_and_clear(LIST, count):
///     l = LIST.ll_newlist(count)
///     if malloc_zero_filled:
///         return l
///     rgc.ll_arrayclear(l.ll_items())
///     return l
/// ```
///
/// `malloc_zero_filled` collapses to `False` (translated program), so the
/// early return is dead and the body unconditionally emits `ll_newlist(count)`
/// → `rgc.ll_arrayclear(items)` → return `l`.  The `newlist_clear(count)`
/// oopspec is attached in lib.rs by CallPath (the codewriter derives the path
/// from this graph's NAME `_ll_alloc_and_clear`), so a JIT-path call lowers to
/// `new_array_clear` instead of residualizing.
fn build_ll_alloc_and_clear_helper_graph(
    name: &str,
    layout: ListLayout,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    newlist_fn: Constant,
    arrayclear_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let count_arg = variable_with_lltype("count", LowLevelType::Signed);
    let startblock = Block::shared(vec![Hlvalue::Variable(count_arg.clone())]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // l = LIST.ll_newlist(count).
    let l = variable_with_lltype("l", ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![Hlvalue::Constant(newlist_fn), Hlvalue::Variable(count_arg)],
        Hlvalue::Variable(l.clone()),
    ));
    // rgc.ll_arrayclear(l.ll_items()).
    let items = emit_list_items_read(&startblock, layout, &item_lltype, &l);
    let clear_void = variable_with_lltype("v", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![Hlvalue::Constant(arrayclear_fn), items],
        Hlvalue::Variable(clear_void),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(l)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["count".to_string()],
        func,
    ))
}

/// Synthesise `_ll_alloc_and_set_nonnull(LIST, count, item)`
/// (`rlist.py:530-537`):
///
/// ```python
/// def _ll_alloc_and_set_nonnull(LIST, count, item):
///     l = LIST.ll_newlist(count)
///     i = 0
///     while i < count:
///         l.ll_setitem_fast(i, item)
///         i += 1
///     return l
/// ```
///
/// This graph is the original loop body. `build_alloc_and_set_family`
/// mints the `look_inside_iff` wrapper (`rlib/jit.py` `look_inside_iff`:
/// `unroll_safe` on the original, a `dont_look_inside` trampoline, and
/// `if not we_are_jitted() or (isconstant(count) and count < 137)`) under
/// `{name}_look_inside_iff` and points the jit arm at that wrapper. The
/// loop itself stays four blocks: startblock `l = ll_newlist(count)` then
/// enters the header with `i = 0`; header `int_lt(i, count)` branches
/// true → body, false → return l; body `ll_setitem_fast(l, i, item)` then
/// `i += 1` back to header.
fn build_ll_alloc_and_set_nonnull_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    newlist_fn: Constant,
    setitem_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let count_arg = variable_with_lltype("count", LowLevelType::Signed);
    let item_arg = variable_with_lltype("item", item_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(count_arg.clone()),
        Hlvalue::Variable(item_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // Loop blocks carry (l, count, item, i) as fresh inputargs.
    let l_c = variable_with_lltype("l", ptr_lltype.clone());
    let count_c = variable_with_lltype("count", LowLevelType::Signed);
    let item_c = variable_with_lltype("item", item_lltype.clone());
    let i_c = variable_with_lltype("i", LowLevelType::Signed);
    let block_cond = Block::shared(vec![
        Hlvalue::Variable(l_c.clone()),
        Hlvalue::Variable(count_c.clone()),
        Hlvalue::Variable(item_c.clone()),
        Hlvalue::Variable(i_c.clone()),
    ]);

    let l_b = variable_with_lltype("l", ptr_lltype.clone());
    let count_b = variable_with_lltype("count", LowLevelType::Signed);
    let item_b = variable_with_lltype("item", item_lltype.clone());
    let i_b = variable_with_lltype("i", LowLevelType::Signed);
    let block_body = Block::shared(vec![
        Hlvalue::Variable(l_b.clone()),
        Hlvalue::Variable(count_b.clone()),
        Hlvalue::Variable(item_b.clone()),
        Hlvalue::Variable(i_b.clone()),
    ]);

    // startblock: l = ll_newlist(count); enter header with i = 0.
    let l = variable_with_lltype("l", ptr_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(newlist_fn),
            Hlvalue::Variable(count_arg.clone()),
        ],
        Hlvalue::Variable(l.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l),
                Hlvalue::Variable(count_arg),
                Hlvalue::Variable(item_arg),
                signed_const(0),
            ],
            Some(block_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // header: int_lt(i, count). True -> body; False -> return l.
    let cond = variable_with_lltype("cond", LowLevelType::Bool);
    block_cond.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![
            Hlvalue::Variable(i_c.clone()),
            Hlvalue::Variable(count_c.clone()),
        ],
        Hlvalue::Variable(cond.clone()),
    ));
    block_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
    block_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_c.clone()),
                Hlvalue::Variable(count_c),
                Hlvalue::Variable(item_c),
                Hlvalue::Variable(i_c),
            ],
            Some(block_body.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(l_c)],
            Some(graph.returnblock.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // body: ll_setitem_fast(l, i, item); i += 1.
    let set_void = variable_with_lltype("v", LowLevelType::Void);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(setitem_fn),
            Hlvalue::Variable(l_b.clone()),
            Hlvalue::Variable(i_b.clone()),
            Hlvalue::Variable(item_b.clone()),
        ],
        Hlvalue::Variable(set_void),
    ));
    let i_next = variable_with_lltype("i", LowLevelType::Signed);
    block_body.borrow_mut().operations.push(SpaceOperation::new(
        "int_add",
        vec![Hlvalue::Variable(i_b), signed_const(1)],
        Hlvalue::Variable(i_next.clone()),
    ));
    block_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_b),
                Hlvalue::Variable(count_b),
                Hlvalue::Variable(item_b),
                Hlvalue::Variable(i_next),
            ],
            Some(block_cond),
            None,
        )
        .into_ref(),
    ]);

    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    // `look_inside_iff` (`rlib/jit.py`): `unroll_safe(func)` then
    // `func._annspecialcase_ = "specialize:call_location"`.
    func._jit_unroll_safe_ = true;
    func.annspecialcase = Some("specialize:call_location".to_string());
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["count".to_string(), "item".to_string()],
        func,
    ))
}

/// Function pointer whose `_name` is the CallPath `lib.rs` marks
/// `jit.isconstant`. Spine B splits that name on `::` so
/// `rewrite_op_direct_call` emits `int_isconstant`.
fn isconstant_intrinsic_funcptr() -> Constant {
    let ft = FuncType {
        args: vec![LowLevelType::Signed],
        result: LowLevelType::Bool,
    };
    let ptr = functionptr(
        ft.clone(),
        "majit_rlib::jit::isconstant",
        None,
        Some("majit_rlib::jit::isconstant".to_string()),
    );
    Constant::with_concretetype(
        ConstValue::LLPtr(Box::new(ptr)),
        LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Func(ft),
        })),
    )
}

/// `rlib/jit.py isconstant`: the interpreter fallback is
/// `NonConstant(False)` inside the marked intrinsic; the traced path is
/// the oopspec `jit.isconstant`, rewritten to `int_isconstant`.
fn build_ll_alloc_and_set_isconstant_helper_graph(name: &str) -> Result<PyGraph, TyperError> {
    let value = variable_with_lltype("value", LowLevelType::Signed);
    let startblock = Block::shared(vec![Hlvalue::Variable(value.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let result = variable_with_lltype("result", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(isconstant_intrinsic_funcptr()),
            Hlvalue::Variable(value),
        ],
        Hlvalue::Variable(result.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);
    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    // `rlib/jit.py isconstant` is `@specialize.call_location()`.
    func.annspecialcase = Some("specialize:call_location".to_string());
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["value".to_string()],
        func,
    ))
}

/// `look_inside_iff`'s `@dont_look_inside` trampoline: one `direct_call`
/// of the original nonnull loop.
fn build_ll_alloc_and_set_trampoline_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    orig_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let count_arg = variable_with_lltype("count", LowLevelType::Signed);
    let item_arg = variable_with_lltype("item", item_lltype);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(count_arg.clone()),
        Hlvalue::Variable(item_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let res = variable_with_lltype("result", ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(orig_fn),
            Hlvalue::Variable(count_arg),
            Hlvalue::Variable(item_arg),
        ],
        Hlvalue::Variable(res.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(res)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);
    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    // `@dont_look_inside` (`rlib/jit.py` `look_inside_iff`); the trampoline
    // also receives `specialize:call_location`.
    func._jit_look_inside_ = Some(false);
    func.annspecialcase = Some("specialize:call_location".to_string());
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["count".to_string(), "item".to_string()],
        func,
    ))
}

/// `rlib/jit.py look_inside_iff` wrapper around the nonnull loop:
///
/// ```python
/// def f(count, item):
///     if not we_are_jitted() or (isconstant(count) and count < 137):
///         return orig(count, item)
///     else:
///         return trampoline(count, item)
/// ```
///
/// `we_are_jitted` is the `SpecTag` exitswitch (true link first, matching
/// [`build_ll_alloc_and_set_helper_graph`]). The predicate short-circuits:
/// `isconstant` false goes to the trampoline, true checks `count < 137`.
fn build_ll_alloc_and_set_look_inside_iff_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    orig_fn: Constant,
    trampoline_fn: Constant,
    isconstant_fn: Constant,
) -> Result<PyGraph, TyperError> {
    let count_arg = variable_with_lltype("count", LowLevelType::Signed);
    let item_arg = variable_with_lltype("item", item_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(count_arg.clone()),
        Hlvalue::Variable(item_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let orig_count = variable_with_lltype("count", LowLevelType::Signed);
    let orig_item = variable_with_lltype("item", item_lltype.clone());
    let block_orig = Block::shared(vec![
        Hlvalue::Variable(orig_count.clone()),
        Hlvalue::Variable(orig_item.clone()),
    ]);
    let orig_res = variable_with_lltype("result", ptr_lltype.clone());
    block_orig.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(orig_fn),
            Hlvalue::Variable(orig_count),
            Hlvalue::Variable(orig_item),
        ],
        Hlvalue::Variable(orig_res.clone()),
    ));
    block_orig.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(orig_res)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let tr_count = variable_with_lltype("count", LowLevelType::Signed);
    let tr_item = variable_with_lltype("item", item_lltype.clone());
    let block_tramp = Block::shared(vec![
        Hlvalue::Variable(tr_count.clone()),
        Hlvalue::Variable(tr_item.clone()),
    ]);
    let tr_res = variable_with_lltype("result", ptr_lltype);
    block_tramp
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(trampoline_fn),
                Hlvalue::Variable(tr_count),
                Hlvalue::Variable(tr_item),
            ],
            Hlvalue::Variable(tr_res.clone()),
        ));
    block_tramp.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(tr_res)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let lt_count = variable_with_lltype("count", LowLevelType::Signed);
    let lt_item = variable_with_lltype("item", item_lltype.clone());
    let block_lt = Block::shared(vec![
        Hlvalue::Variable(lt_count.clone()),
        Hlvalue::Variable(lt_item.clone()),
    ]);
    let below = variable_with_lltype("below", LowLevelType::Bool);
    block_lt.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(lt_count.clone()), signed_const(137)],
        Hlvalue::Variable(below.clone()),
    ));
    block_lt.borrow_mut().exitswitch = Some(Hlvalue::Variable(below));
    block_lt.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(lt_count.clone()),
                Hlvalue::Variable(lt_item.clone()),
            ],
            Some(block_orig.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(lt_count), Hlvalue::Variable(lt_item)],
            Some(block_tramp.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let pred_count = variable_with_lltype("count", LowLevelType::Signed);
    let pred_item = variable_with_lltype("item", item_lltype);
    let block_pred = Block::shared(vec![
        Hlvalue::Variable(pred_count.clone()),
        Hlvalue::Variable(pred_item.clone()),
    ]);
    let is_const = variable_with_lltype("is_constant", LowLevelType::Bool);
    block_pred.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(isconstant_fn),
            Hlvalue::Variable(pred_count.clone()),
        ],
        Hlvalue::Variable(is_const.clone()),
    ));
    block_pred.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_const));
    block_pred.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(pred_count.clone()),
                Hlvalue::Variable(pred_item.clone()),
            ],
            Some(block_lt),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(pred_count), Hlvalue::Variable(pred_item)],
            Some(block_tramp),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    startblock.borrow_mut().exitswitch = Some(Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::SpecTag(WE_ARE_JITTED_TAG_ID),
        LowLevelType::Bool,
    )));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(count_arg.clone()),
                Hlvalue::Variable(item_arg.clone()),
            ],
            Some(block_pred),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(count_arg), Hlvalue::Variable(item_arg)],
            Some(block_orig),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["count".to_string(), "item".to_string()],
        func,
    ))
}

/// Synthesise `ll_copy(RESLIST, l)` (`rpython/rtyper/rlist.py`):
///
/// ```python
/// def ll_copy(RESLIST, l):
///     length = l.ll_length()
///     new_lst = RESLIST.ll_newlist(length)
///     ll_arraycopy(l, new_lst, 0, 0, length)
///     return new_lst
/// ```
///
/// `l` is the source (`source_layout`); the `RESLIST` result is `result_layout`
/// (`list(x)` yields a resized list, but a non-resized result is handled too).
/// The length / items reads are layout-parameterised exactly like the
/// getitem / extend CFGs.
#[allow(clippy::too_many_arguments)]
fn build_ll_copy_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    source_layout: ListLayout,
    result_layout: ListLayout,
    source_ptr_lltype: LowLevelType,
    result_ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    newlist: &LowLevelFunction,
    arraycopy: &LowLevelFunction,
) -> Result<PyGraph, TyperError> {
    let newlist_const = sub_helper_funcptr_constant(rtyper, newlist)?;
    let copy_const = sub_helper_funcptr_constant(rtyper, arraycopy)?;
    let items_ptr = items_array_ptr_lltype(&item_lltype);

    let l_arg = variable_with_lltype("l", source_ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(l_arg.clone())]);
    let return_var = variable_with_lltype("result", result_ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // length = l.ll_length() (per source layout).
    let length = variable_with_lltype("length", LowLevelType::Signed);
    match source_layout {
        ListLayout::Fixed => {
            startblock.borrow_mut().operations.push(SpaceOperation::new(
                "getarraysize",
                vec![Hlvalue::Variable(l_arg.clone())],
                Hlvalue::Variable(length.clone()),
            ));
        }
        ListLayout::Resized => {
            startblock.borrow_mut().operations.push(SpaceOperation::new(
                "getfield",
                vec![Hlvalue::Variable(l_arg.clone()), void_field_const("length")],
                Hlvalue::Variable(length.clone()),
            ));
        }
    }
    // new_lst = RESLIST.ll_newlist(length).
    let new_lst = variable_with_lltype("new_lst", result_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(newlist_const),
            Hlvalue::Variable(length.clone()),
        ],
        Hlvalue::Variable(new_lst.clone()),
    ));
    // src/dst items (per layout): a FixedSizeListRepr value IS its items array;
    // the resized header reaches the array through `items`.
    let src_items = match source_layout {
        ListLayout::Fixed => Hlvalue::Variable(l_arg),
        ListLayout::Resized => {
            let items = variable_with_lltype("src_items", items_ptr.clone());
            startblock.borrow_mut().operations.push(SpaceOperation::new(
                "getfield",
                vec![Hlvalue::Variable(l_arg), void_field_const("items")],
                Hlvalue::Variable(items.clone()),
            ));
            Hlvalue::Variable(items)
        }
    };
    let dst_items = match result_layout {
        ListLayout::Fixed => Hlvalue::Variable(new_lst.clone()),
        ListLayout::Resized => {
            let items = variable_with_lltype("dst_items", items_ptr);
            startblock.borrow_mut().operations.push(SpaceOperation::new(
                "getfield",
                vec![
                    Hlvalue::Variable(new_lst.clone()),
                    void_field_const("items"),
                ],
                Hlvalue::Variable(items.clone()),
            ));
            Hlvalue::Variable(items)
        }
    };
    // ll_arraycopy(src_items, dst_items, length) — 0-to-0 full copy.
    let copy_void = variable_with_lltype("v", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(copy_const),
            src_items,
            dst_items,
            Hlvalue::Variable(length),
        ],
        Hlvalue::Variable(copy_void),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(new_lst)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string()],
        func,
    ))
}

/// Emit `new_lst = RESLIST.ll_newlist(newlength)` then
/// `ll_arraycopy(src_items, dst_items, src_start, 0, newlength)`, and close
/// `block` returning `new_lst`. Shared tail of the three `ll_listslice_*`
/// helper graphs — the only per-kind difference (`ll_listslice_%s`,
/// rlist.py:883-911) is how `newlength` and `src_start` are computed, which
/// the caller has already materialised into `block`.
#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
)]
fn emit_listslice_alloc_and_copy(
    block: &BlockRef,
    returnblock: &BlockRef,
    source_layout: ListLayout,
    result_ptr_lltype: LowLevelType,
    item_lltype: &LowLevelType,
    newlist_const: Constant,
    copy_const: Constant,
    l: &Variable,
    src_start: Hlvalue,
    newlength: &Variable,
) {
    let items_ptr = items_array_ptr_lltype(item_lltype);
    // The result repr (`hop.r_result`) is a fixed list when the slice is never
    // resized (`ListDef.offspring` keeps `resized=False`), so `ll_newlist`
    // returns a bare `Ptr(GcArray)` with no `items` field; a resized result is
    // a header reaching its array through `items`.  Read the result layout off
    // its pointer lltype and branch `dst_items` the same way `src_items`
    // branches on `source_layout` (`ll_arraycopy` uses `dest.ll_items()`,
    // rlist.py, which is `ll_fixed_items(l) = l` for a fixed list).
    let result_layout = list_layout_from_lltype(&result_ptr_lltype)
        .expect("getslice result lltype must be a list pointer");
    // l = RESLIST.ll_newlist(newlength).
    let new_lst = variable_with_lltype("new_lst", result_ptr_lltype);
    block.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(newlist_const),
            Hlvalue::Variable(newlength.clone()),
        ],
        Hlvalue::Variable(new_lst.clone()),
    ));
    // src items (per source layout): a fixed list IS its items array; a resized
    // header reaches the array through `items`.
    let src_items = match source_layout {
        ListLayout::Fixed => Hlvalue::Variable(l.clone()),
        ListLayout::Resized => {
            let items = variable_with_lltype("src_items", items_ptr.clone());
            block.borrow_mut().operations.push(SpaceOperation::new(
                "getfield",
                vec![Hlvalue::Variable(l.clone()), void_field_const("items")],
                Hlvalue::Variable(items.clone()),
            ));
            Hlvalue::Variable(items)
        }
    };
    // dst items (per result layout): mirror the source branch above.
    let dst_items = match result_layout {
        ListLayout::Fixed => Hlvalue::Variable(new_lst.clone()),
        ListLayout::Resized => {
            let items = variable_with_lltype("dst_items", items_ptr);
            block.borrow_mut().operations.push(SpaceOperation::new(
                "getfield",
                vec![
                    Hlvalue::Variable(new_lst.clone()),
                    void_field_const("items"),
                ],
                Hlvalue::Variable(items.clone()),
            ));
            Hlvalue::Variable(items)
        }
    };
    // ll_arraycopy(src_items, dst_items, src_start, 0, newlength).
    let copy_void = variable_with_lltype("v", LowLevelType::Void);
    block.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        vec![
            Hlvalue::Constant(copy_const),
            src_items,
            dst_items,
            src_start,
            signed_const(0),
            Hlvalue::Variable(newlength.clone()),
        ],
        Hlvalue::Variable(copy_void),
    ));
    block.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(new_lst)],
            Some(returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);
}

/// Synthesise `ll_listslice_startonly(RESLIST, l1, start)` (rlist.py):
///
/// ```python
/// def ll_listslice_startonly(RESLIST, l1, start):
///     len1 = l1.ll_length()
///     newlength = len1 - start
///     l = RESLIST.ll_newlist(newlength)
///     ll_arraycopy(l1, l, start, 0, newlength)
///     return l
/// ```
///
/// The `ll_assert(0 <= start <= len1)` bounds are debug-only (no production
/// op). Single-block graph: read len, `int_sub` for newlength, then the
/// shared alloc-and-copy tail with `src_start = start`.
#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
)]
fn build_ll_listslice_startonly_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    source_layout: ListLayout,
    source_ptr_lltype: LowLevelType,
    result_ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    newlist: &LowLevelFunction,
    arraycopy: &LowLevelFunction,
) -> Result<PyGraph, TyperError> {
    let newlist_const = sub_helper_funcptr_constant(rtyper, newlist)?;
    let copy_const = sub_helper_funcptr_constant(rtyper, arraycopy)?;

    let l_arg = variable_with_lltype("l1", source_ptr_lltype);
    let start_arg = variable_with_lltype("start", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l_arg.clone()),
        Hlvalue::Variable(start_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", result_ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // len1 = l1.ll_length(); newlength = len1 - start.
    let len1 = emit_list_length_read(&startblock, source_layout, &l_arg);
    let newlength = variable_with_lltype("newlength", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![
            Hlvalue::Variable(len1),
            Hlvalue::Variable(start_arg.clone()),
        ],
        Hlvalue::Variable(newlength.clone()),
    ));
    emit_listslice_alloc_and_copy(
        &startblock,
        &graph.returnblock,
        source_layout,
        result_ptr_lltype,
        &item_lltype,
        newlist_const,
        copy_const,
        &l_arg,
        Hlvalue::Variable(start_arg),
        &newlength,
    );

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l1".to_string(), "start".to_string()],
        func,
    ))
}

/// Synthesise `ll_listslice_minusone(RESLIST, l1)` (rlist.py):
///
/// ```python
/// def ll_listslice_minusone(RESLIST, l1):
///     newlength = l1.ll_length() - 1
///     l = RESLIST.ll_newlist(newlength)
///     ll_arraycopy(l1, l, 0, 0, newlength)
///     return l
/// ```
///
/// `ll_assert(newlength >= 0)` is debug-only. Single-block graph:
/// `int_sub(len, 1)` then the shared tail with `src_start = 0`.
#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
)]
fn build_ll_listslice_minusone_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    source_layout: ListLayout,
    source_ptr_lltype: LowLevelType,
    result_ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    newlist: &LowLevelFunction,
    arraycopy: &LowLevelFunction,
) -> Result<PyGraph, TyperError> {
    let newlist_const = sub_helper_funcptr_constant(rtyper, newlist)?;
    let copy_const = sub_helper_funcptr_constant(rtyper, arraycopy)?;

    let l_arg = variable_with_lltype("l1", source_ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(l_arg.clone())]);
    let return_var = variable_with_lltype("result", result_ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // newlength = l1.ll_length() - 1.
    let len1 = emit_list_length_read(&startblock, source_layout, &l_arg);
    let newlength = variable_with_lltype("newlength", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![Hlvalue::Variable(len1), signed_const(1)],
        Hlvalue::Variable(newlength.clone()),
    ));
    emit_listslice_alloc_and_copy(
        &startblock,
        &graph.returnblock,
        source_layout,
        result_ptr_lltype,
        &item_lltype,
        newlist_const,
        copy_const,
        &l_arg,
        signed_const(0),
        &newlength,
    );

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l1".to_string()],
        func,
    ))
}

/// Synthesise `ll_listslice_startstop(RESLIST, l1, start, stop)`
/// (rlist.py:893-904):
///
/// ```python
/// def ll_listslice_startstop(RESLIST, l1, start, stop):
///     length = l1.ll_length()
///     if stop > length:
///         stop = length
///     newlength = stop - start
///     l = RESLIST.ll_newlist(newlength)
///     ll_arraycopy(l1, l, start, 0, newlength)
///     return l
/// ```
///
/// The `stop > length` clamp is a two-way branch that merges `stop` back;
/// `ll_assert(0 <= start <= length and stop >= start)` is debug-only. Three
/// blocks: start (read len, `int_gt`) → clamp (pass `length` as the merged
/// stop) / merge (`int_sub` newlength, then the shared alloc-and-copy tail).
#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
)]
fn build_ll_listslice_startstop_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    source_layout: ListLayout,
    source_ptr_lltype: LowLevelType,
    result_ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    newlist: &LowLevelFunction,
    arraycopy: &LowLevelFunction,
) -> Result<PyGraph, TyperError> {
    let newlist_const = sub_helper_funcptr_constant(rtyper, newlist)?;
    let copy_const = sub_helper_funcptr_constant(rtyper, arraycopy)?;

    let l_arg = variable_with_lltype("l1", source_ptr_lltype.clone());
    let start_arg = variable_with_lltype("start", LowLevelType::Signed);
    let stop_arg = variable_with_lltype("stop", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l_arg.clone()),
        Hlvalue::Variable(start_arg.clone()),
        Hlvalue::Variable(stop_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", result_ptr_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // merge block carries (l1, start, stop) with stop already clamped to length.
    let l_m = variable_with_lltype("l1", source_ptr_lltype.clone());
    let start_m = variable_with_lltype("start", LowLevelType::Signed);
    let stop_m = variable_with_lltype("stop", LowLevelType::Signed);
    let block_merge = Block::shared(vec![
        Hlvalue::Variable(l_m.clone()),
        Hlvalue::Variable(start_m.clone()),
        Hlvalue::Variable(stop_m.clone()),
    ]);
    // clamp block carries (l1, start, length) — sets stop = length on the merge edge.
    let l_c = variable_with_lltype("l1", source_ptr_lltype);
    let start_c = variable_with_lltype("start", LowLevelType::Signed);
    let len_c = variable_with_lltype("length", LowLevelType::Signed);
    let block_clamp = Block::shared(vec![
        Hlvalue::Variable(l_c.clone()),
        Hlvalue::Variable(start_c.clone()),
        Hlvalue::Variable(len_c.clone()),
    ]);

    // startblock: length = ll_length; too_big = int_gt(stop, length); branch.
    let length = emit_list_length_read(&startblock, source_layout, &l_arg);
    let too_big = variable_with_lltype("too_big", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_gt",
        vec![
            Hlvalue::Variable(stop_arg.clone()),
            Hlvalue::Variable(length.clone()),
        ],
        Hlvalue::Variable(too_big.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(too_big));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_arg.clone()),
                Hlvalue::Variable(start_arg.clone()),
                Hlvalue::Variable(length),
            ],
            Some(block_clamp.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(l_arg),
                Hlvalue::Variable(start_arg),
                Hlvalue::Variable(stop_arg),
            ],
            Some(block_merge.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_clamp: stop := length. Merge edge passes `length` as stop.
    block_clamp.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_c),
                Hlvalue::Variable(start_c),
                Hlvalue::Variable(len_c),
            ],
            Some(block_merge.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_merge: newlength = stop - start; alloc + copy.
    let newlength = variable_with_lltype("newlength", LowLevelType::Signed);
    block_merge
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_sub",
            vec![
                Hlvalue::Variable(stop_m),
                Hlvalue::Variable(start_m.clone()),
            ],
            Hlvalue::Variable(newlength.clone()),
        ));
    emit_listslice_alloc_and_copy(
        &block_merge,
        &graph.returnblock,
        source_layout,
        result_ptr_lltype,
        &item_lltype,
        newlist_const,
        copy_const,
        &l_m,
        Hlvalue::Variable(start_m),
        &newlength,
    );

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l1".to_string(), "start".to_string(), "stop".to_string()],
        func,
    ))
}

/// Derive the [`ListLayout`] from a list repr's `Ptr` lltype: a resized list
/// is `Ptr(GcStruct("list", …))`, a fixed list is `Ptr(GcArray)`.
fn list_layout_from_lltype(ptr_lltype: &LowLevelType) -> Result<ListLayout, TyperError> {
    let LowLevelType::Ptr(ptr) = ptr_lltype else {
        return Err(TyperError::message(
            "list_layout_from_lltype: not a Ptr lltype",
        ));
    };
    match &ptr.TO {
        PtrTarget::Struct(_) => Ok(ListLayout::Resized),
        PtrTarget::Array(_) => Ok(ListLayout::Fixed),
        other => Err(TyperError::message(format!(
            "list_layout_from_lltype: unexpected Ptr target {other:?}"
        ))),
    }
}

/// Shared body of `FixedSizeListRepr` / `ListRepr` `rtype_bltn_list`
/// (`rpython/rtyper/rlist.py` `rtype_bltn_list`): the receiver has
/// already been threaded into `vlist` and `exception_is_here` declared. Mints
/// `ll_arraycopy` <- `ll_newlist` <- `ll_copy` in dependency order (the
/// result `RESLIST` is `hop.r_result`) and `gendirectcall`s `ll_copy`.
fn rtype_bltn_list_via_ll_copy(
    hop: &HighLevelOp,
    source_layout: ListLayout,
    source_ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    vlist: Vec<Hlvalue>,
) -> RTypeResult {
    let r_result = hop
        .r_result
        .borrow()
        .as_ref()
        .map(Arc::clone)
        .ok_or_else(|| TyperError::message("rtype_bltn_list: r_result not populated"))?;
    let result_ptr_lltype = r_result.lowleveltype().clone();
    let result_layout = list_layout_from_lltype(&result_ptr_lltype)?;
    let items_ptr = items_array_ptr_lltype(&item_lltype);

    let arraycopy = {
        let item = item_lltype.clone();
        hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_arraycopy".to_string(),
            vec![items_ptr.clone(), items_ptr.clone(), LowLevelType::Signed],
            LowLevelType::Void,
            move |_rtyper, _args, _result| {
                build_ll_arraycopy_helper_graph("ll_arraycopy", item.clone())
            },
        )?
    };
    let newlist = {
        let result_ptr = result_ptr_lltype.clone();
        let item = item_lltype.clone();
        hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_newlist".to_string(),
            vec![LowLevelType::Signed],
            result_ptr_lltype.clone(),
            move |_rtyper, _args, _result| {
                build_ll_newlist_helper_graph(
                    "ll_newlist",
                    result_layout,
                    result_ptr.clone(),
                    item.clone(),
                )
            },
        )?
    };
    let copy = {
        let source_ptr = source_ptr_lltype.clone();
        let result_ptr = result_ptr_lltype.clone();
        let item = item_lltype.clone();
        hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_copy".to_string(),
            vec![source_ptr_lltype],
            result_ptr_lltype,
            move |rtyper, _args, _result| {
                build_ll_copy_helper_graph(
                    rtyper,
                    "ll_copy",
                    source_layout,
                    result_layout,
                    source_ptr.clone(),
                    result_ptr.clone(),
                    item.clone(),
                    &newlist,
                    &arraycopy,
                )
            },
        )?
    };
    hop.gendirectcall(&copy, vlist)
}

/// RPython `AbstractBaseListRepr.rtype_getslice(r_lst, hop)` (rlist.py):
///
/// ```python
/// def rtype_getslice(r_lst, hop):
///     cRESLIST = hop.inputconst(Void, hop.r_result.LIST)
///     v_lst = hop.inputarg(r_lst, arg=0)
///     kind, vlist = hop.decompose_slice_args()
///     ll_listslice = globals()['ll_listslice_%s' % kind]
///     return hop.gendirectcall(ll_listslice, cRESLIST, v_lst, *vlist)
/// ```
///
/// `hop.r_result` is `listdef.offspring` (unaryop.py getslice), a fresh list
/// whose repr is `FixedSizeListRepr` when the slice is never resized and
/// `ListRepr` otherwise; the receiver's `source_layout` likewise varies. Mints
/// `ll_arraycopy` (general 5-arg), `ll_newlist`, and the per-`kind`
/// `ll_listslice_%s` helper in dependency order, then `gendirectcall`s it. The
/// `cRESLIST` Void const is implicit — the result repr is `hop.r_result`, which
/// the helper builders read directly (and off whose lltype they select the
/// result layout for the destination-items copy).
fn rtype_getslice_via_ll_listslice(
    hop: &HighLevelOp,
    source_layout: ListLayout,
    source_ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
    v_lst: Hlvalue,
) -> RTypeResult {
    let (kind, vlist) = hop.decompose_slice_args()?;

    let r_result = hop
        .r_result
        .borrow()
        .as_ref()
        .map(Arc::clone)
        .ok_or_else(|| TyperError::message("rtype_getslice: r_result not populated"))?;
    let result_ptr_lltype = r_result.lowleveltype().clone();
    let items_ptr = items_array_ptr_lltype(&item_lltype);

    // ll_arraycopy(src, dst, src_start, dst_start, length) — general 5-arg
    // (rgc.py): a slice copies from `src_start = start != 0`, so the
    // start=0 specialisation used by `ll_copy` does not fit.
    let arraycopy = {
        let item = item_lltype.clone();
        hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_arraycopy".to_string(),
            vec![
                items_ptr.clone(),
                items_ptr.clone(),
                LowLevelType::Signed,
                LowLevelType::Signed,
                LowLevelType::Signed,
            ],
            LowLevelType::Void,
            move |_rtyper, _args, _result| {
                build_ll_arraycopy_general_helper_graph("ll_arraycopy", item.clone())
            },
        )?
    };
    let newlist = {
        let result_ptr = result_ptr_lltype.clone();
        let item = item_lltype.clone();
        let result_layout = list_layout_from_lltype(&result_ptr_lltype)?;
        hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_newlist".to_string(),
            vec![LowLevelType::Signed],
            result_ptr_lltype.clone(),
            move |_rtyper, _args, _result| {
                build_ll_newlist_helper_graph(
                    "ll_newlist",
                    result_layout,
                    result_ptr.clone(),
                    item.clone(),
                )
            },
        )?
    };

    // The per-kind `ll_listslice_%s` helper (rlist.py).
    let (helper_name, helper_args): (&str, Vec<LowLevelType>) = match kind {
        SliceKind::MinusOne => ("ll_listslice_minusone", vec![source_ptr_lltype.clone()]),
        SliceKind::StartOnly => (
            "ll_listslice_startonly",
            vec![source_ptr_lltype.clone(), LowLevelType::Signed],
        ),
        SliceKind::StartStop => (
            "ll_listslice_startstop",
            vec![
                source_ptr_lltype.clone(),
                LowLevelType::Signed,
                LowLevelType::Signed,
            ],
        ),
    };
    let listslice = {
        let name = helper_name.to_string();
        let source_ptr = source_ptr_lltype.clone();
        let result_ptr = result_ptr_lltype.clone();
        let item = item_lltype.clone();
        hop.rtyper.lowlevel_helper_function_with_builder(
            helper_name.to_string(),
            helper_args,
            result_ptr_lltype.clone(),
            move |rtyper, _args, _result| match kind {
                SliceKind::MinusOne => build_ll_listslice_minusone_helper_graph(
                    rtyper,
                    &name,
                    source_layout,
                    source_ptr.clone(),
                    result_ptr.clone(),
                    item.clone(),
                    &newlist,
                    &arraycopy,
                ),
                SliceKind::StartOnly => build_ll_listslice_startonly_helper_graph(
                    rtyper,
                    &name,
                    source_layout,
                    source_ptr.clone(),
                    result_ptr.clone(),
                    item.clone(),
                    &newlist,
                    &arraycopy,
                ),
                SliceKind::StartStop => build_ll_listslice_startstop_helper_graph(
                    rtyper,
                    &name,
                    source_layout,
                    source_ptr.clone(),
                    result_ptr.clone(),
                    item.clone(),
                    &newlist,
                    &arraycopy,
                ),
            },
        )?
    };

    // gendirectcall(ll_listslice, v_lst, *vlist).
    let mut call_args = vec![v_lst];
    call_args.extend(vlist);
    hop.gendirectcall(&listslice, call_args)
}

/// `FixedSizeListRepr` (bare `Ptr(GcArray)`) vs the resized `ListRepr`
/// (`Ptr(GcStruct("list", length, items))`) differ only in how `ll_length` /
/// `ll_getitem_fast` / `ll_setitem_fast` reach the element array, so the
/// checked / negative-index `ll_getitem` / `ll_setitem` CFGs
/// (`rlist.py:688-748`) are written once and parameterised by this layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ListLayout {
    Fixed,
    Resized,
}

impl ListLayout {
    fn getitem_fast_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_getitem_fast",
            ListLayout::Resized => "ll_getitem_fast",
        }
    }
    /// rlist.py `ll_getitem_foldable_nonneg` — the foldable
    /// counterpart of `getitem_fast_name`, a DISTINCT function so the
    /// helper cache never serves a foldable graph to a mutated list (or
    /// vice-versa). Only the Fixed layout reaches it (Resized ⟹ mutated).
    fn getitem_fast_foldable_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_getitem_fast_foldable",
            ListLayout::Resized => "ll_getitem_fast_foldable",
        }
    }
    fn setitem_fast_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_setitem_fast",
            ListLayout::Resized => "ll_setitem_fast",
        }
    }
    fn getitem_neg_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_getitem",
            ListLayout::Resized => "ll_getitem",
        }
    }
    fn getitem_nonneg_checked_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_getitem_nonneg_checked",
            ListLayout::Resized => "ll_getitem_nonneg_checked",
        }
    }
    fn getitem_checked_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_getitem_checked",
            ListLayout::Resized => "ll_getitem_checked",
        }
    }
    fn setitem_neg_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_setitem",
            ListLayout::Resized => "ll_setitem",
        }
    }
    fn setitem_nonneg_checked_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_setitem_nonneg_checked",
            ListLayout::Resized => "ll_setitem_nonneg_checked",
        }
    }
    fn setitem_checked_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_setitem_checked",
            ListLayout::Resized => "ll_setitem_checked",
        }
    }
}

fn signed_const(n: i64) -> Hlvalue {
    Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::Int(n),
        LowLevelType::Signed,
    ))
}

fn bool_const(b: bool) -> Hlvalue {
    Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::Bool(b),
        LowLevelType::Bool,
    ))
}

/// RPython `pairtype(AbstractListRepr, AbstractListRepr).rtype_eq`
/// (`rlist.py`): `hop.gendirectcall(ll_listeq, v_lst1, v_lst2, r_lst1.get_eqfunc())`.
/// Primitive items (`get_ll_eq_function` is `None`) compare with `!=`.
pub fn pair_list_list_rtype_eq(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    let vlist = hop.inputargs(vec![ConvertedTo::Repr(r1), ConvertedTo::Repr(r2)])?;
    let (layout1, item1) = list_layout_and_item(r1)?;
    let (layout2, item2) = list_layout_and_item(r2)?;
    if item1 != item2 {
        return Err(TyperError::message(format!(
            "ll_listeq item types differ: {item1:?} vs {item2:?}"
        )));
    }
    let ptr1 = r1.lowleveltype().clone();
    let ptr2 = r2.lowleveltype().clone();
    let item = item1;
    let helper_name = format!(
        "ll_listeq_{}_{}",
        list_layout_tag(layout1),
        list_layout_tag(layout2)
    );
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        helper_name.clone(),
        vec![ptr1.clone(), ptr2.clone()],
        LowLevelType::Bool,
        move |_rtyper, _args, _result| {
            build_ll_listeq_helper_graph(
                &helper_name,
                ptr1.clone(),
                ptr2.clone(),
                item.clone(),
                layout1,
                layout2,
            )
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

/// RPython `rtype_ne`: `ll_listeq` then `bool_not`.
pub fn pair_list_list_rtype_ne(r1: &dyn Repr, r2: &dyn Repr, hop: &HighLevelOp) -> RTypeResult {
    let v_eq = pair_list_list_rtype_eq(r1, r2, hop)?
        .ok_or_else(|| TyperError::message("pair_list_list_rtype_ne: rtype_eq returned None"))?;
    Ok(hop.genop(
        "bool_not",
        vec![v_eq],
        GenopResult::LLType(LowLevelType::Bool),
    ))
}

fn list_layout_tag(layout: ListLayout) -> &'static str {
    match layout {
        ListLayout::Fixed => "fixed",
        ListLayout::Resized => "resized",
    }
}

fn list_layout_and_item(r: &dyn Repr) -> Result<(ListLayout, LowLevelType), TyperError> {
    use super::pairtype::ReprClassId;
    match r.repr_class_id() {
        ReprClassId::FixedSizeListRepr => {
            Ok((ListLayout::Fixed, array_item_of_ptr(r.lowleveltype())?))
        }
        ReprClassId::ListRepr => Ok((
            ListLayout::Resized,
            resized_list_item_of_ptr(r.lowleveltype())?,
        )),
        other => Err(TyperError::message(format!(
            "ll_listeq expected a list repr, got {other:?}"
        ))),
    }
}

fn array_item_of_ptr(ll: &LowLevelType) -> Result<LowLevelType, TyperError> {
    let LowLevelType::Ptr(ptr) = ll else {
        return Err(TyperError::message(format!(
            "fixed list lltype must be Ptr(GcArray), got {ll:?}"
        )));
    };
    let PtrTarget::Array(arr) = &ptr.TO else {
        return Err(TyperError::message(format!(
            "fixed list Ptr.TO must be Array, got {:?}",
            ptr.TO
        )));
    };
    Ok(arr.OF.clone())
}

fn resized_list_item_of_ptr(ll: &LowLevelType) -> Result<LowLevelType, TyperError> {
    let LowLevelType::Ptr(ptr) = ll else {
        return Err(TyperError::message(format!(
            "resized list lltype must be Ptr(GcStruct list), got {ll:?}"
        )));
    };
    let PtrTarget::Struct(st) = &ptr.TO else {
        return Err(TyperError::message(format!(
            "resized list Ptr.TO must be Struct, got {:?}",
            ptr.TO
        )));
    };
    let items = st
        ._flds
        .get("items")
        .ok_or_else(|| TyperError::message("resized list struct has no items field"))?;
    array_item_of_ptr(items)
}

fn primitive_item_eq_op(item: &LowLevelType) -> &'static str {
    match item {
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        LowLevelType::Float => "float_eq",
        LowLevelType::Ptr(_) => "ptr_eq",
        _ => "int_eq",
    }
}

/// `ll_listeq` (`rlist.py`) for primitive items (`eqfn is None`):
/// length mismatch → false, else `getitem_fast` per index until a
/// mismatch or the end.
fn build_ll_listeq_helper_graph(
    name: &str,
    ptr1: LowLevelType,
    ptr2: LowLevelType,
    item_lltype: LowLevelType,
    layout1: ListLayout,
    layout2: ListLayout,
) -> Result<PyGraph, TyperError> {
    let eq_op = primitive_item_eq_op(&item_lltype);
    let l1 = variable_with_lltype("l1", ptr1.clone());
    let l2 = variable_with_lltype("l2", ptr2.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l1.clone()),
        Hlvalue::Variable(l2.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let items1_ll = match layout1 {
        ListLayout::Fixed => ptr1.clone(),
        ListLayout::Resized => items_array_ptr_lltype(&item_lltype),
    };
    let items2_ll = match layout2 {
        ListLayout::Fixed => ptr2.clone(),
        ListLayout::Resized => items_array_ptr_lltype(&item_lltype),
    };

    let a1_cond = variable_with_lltype("a1", items1_ll.clone());
    let a2_cond = variable_with_lltype("a2", items2_ll.clone());
    let len_cond = variable_with_lltype("len1", LowLevelType::Signed);
    let j_cond = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(a1_cond.clone()),
        Hlvalue::Variable(a2_cond.clone()),
        Hlvalue::Variable(len_cond.clone()),
        Hlvalue::Variable(j_cond.clone()),
    ]);

    let a1_body = variable_with_lltype("a1", items1_ll);
    let a2_body = variable_with_lltype("a2", items2_ll);
    let len_body = variable_with_lltype("len1", LowLevelType::Signed);
    let j_body = variable_with_lltype("j", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(a1_body.clone()),
        Hlvalue::Variable(a2_body.clone()),
        Hlvalue::Variable(len_body.clone()),
        Hlvalue::Variable(j_body.clone()),
    ]);

    let len1 = emit_list_length_read(&startblock, layout1, &l1);
    let len2 = emit_list_length_read(&startblock, layout2, &l2);
    let lens_ne = variable_with_lltype("lens_ne", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ne",
        vec![Hlvalue::Variable(len1.clone()), Hlvalue::Variable(len2)],
        Hlvalue::Variable(lens_ne.clone()),
    ));
    let items1 = emit_list_items_read(&startblock, layout1, &item_lltype, &l1);
    let items2 = emit_list_items_read(&startblock, layout2, &item_lltype, &l2);
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(lens_ne));
    startblock.closeblock(vec![
        Link::new(
            vec![bool_const(false)],
            Some(graph.returnblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![items1, items2, Hlvalue::Variable(len1), signed_const(0)],
            Some(block_loop_cond.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let still = variable_with_lltype("still", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(j_cond.clone()),
                Hlvalue::Variable(len_cond.clone()),
            ],
            Hlvalue::Variable(still.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(still));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(a1_cond),
                Hlvalue::Variable(a2_cond),
                Hlvalue::Variable(len_cond),
                Hlvalue::Variable(j_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![bool_const(true)],
            Some(graph.returnblock.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let v1 = variable_with_lltype("v1", item_lltype.clone());
    let v2 = variable_with_lltype("v2", item_lltype);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(a1_body.clone()),
                Hlvalue::Variable(j_body.clone()),
            ],
            Hlvalue::Variable(v1.clone()),
        ));
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(a2_body.clone()),
                Hlvalue::Variable(j_body.clone()),
            ],
            Hlvalue::Variable(v2.clone()),
        ));
    let same = variable_with_lltype("same", LowLevelType::Bool);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            eq_op,
            vec![Hlvalue::Variable(v1), Hlvalue::Variable(v2)],
            Hlvalue::Variable(same.clone()),
        ));
    let next_j = variable_with_lltype("next_j", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(j_body), signed_const(1)],
            Hlvalue::Variable(next_j.clone()),
        ));
    block_loop_body.borrow_mut().exitswitch = Some(Hlvalue::Variable(same));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(a1_body),
                Hlvalue::Variable(a2_body),
                Hlvalue::Variable(len_body),
                Hlvalue::Variable(next_j),
            ],
            Some(block_loop_cond),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![bool_const(false)],
            Some(graph.returnblock.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    let mut func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    // `look_inside_iff` (`rlib/jit.py`): `unroll_safe(func)` then
    // `func._annspecialcase_ = "specialize:call_location"`.
    func._jit_unroll_safe_ = true;
    func.annspecialcase = Some("specialize:call_location".to_string());
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l1".to_string(), "l2".to_string()],
        func,
    ))
}

/// Push the `ll_length` read for `layout` onto `block` and return the Signed
/// length var: `getarraysize(l)` (Fixed, bare `Ptr(GcArray)`) or
/// `getfield(l, "length")` (Resized, struct header).
fn emit_list_length_read(block: &BlockRef, layout: ListLayout, l: &Variable) -> Variable {
    let length = variable_with_lltype("length", LowLevelType::Signed);
    let op = match layout {
        ListLayout::Fixed => SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(l.clone())],
            Hlvalue::Variable(length.clone()),
        ),
        ListLayout::Resized => SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(l.clone()), void_field_const("length")],
            Hlvalue::Variable(length.clone()),
        ),
    };
    block.borrow_mut().operations.push(op);
    length
}

/// Build (or retrieve cached) the layout's `ll_*_getitem_fast` sub-helper and
/// return a funcptr `Constant` to `direct_call` it (the `basegetitem` of
/// `rlist.py:266`).
fn list_getitem_fast_funcptr(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    foldable: bool,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<Constant, TyperError> {
    // rlist.py:264-266 — the `basegetitem` passed into every (nonneg×checkidx)
    // wrapper is the foldable `ll_getitem_foldable_nonneg` (rlist.py)
    // when the list item is not mutated.  Only Fixed has a foldable element
    // load; Resized is always mutated, so `foldable` is never set for it.
    let foldable_fixed = foldable && matches!(layout, ListLayout::Fixed);
    let name = if foldable_fixed {
        layout.getitem_fast_foldable_name().to_string()
    } else {
        layout.getitem_fast_name().to_string()
    };
    let name_owned = name.clone();
    let ptr_for_builder = ptr_lltype.clone();
    let item_for_builder = item_lltype.clone();
    let inner = rtyper.lowlevel_helper_function_with_builder(
        name,
        vec![ptr_lltype, LowLevelType::Signed],
        item_lltype,
        move |_rtyper, _args, _result| match layout {
            ListLayout::Fixed if foldable_fixed => {
                build_ll_fixed_getitem_fast_foldable_helper_graph(
                    &name_owned,
                    ptr_for_builder.clone(),
                    item_for_builder.clone(),
                )
            }
            ListLayout::Fixed => build_ll_fixed_getitem_fast_helper_graph(
                &name_owned,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
            ListLayout::Resized => build_ll_getitem_fast_helper_graph(
                &name_owned,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
        },
    )?;
    sub_helper_funcptr_constant(rtyper, &inner)
}

/// rlist.py `ll_getitem` with `func is dum_nocheck` — negative index,
/// no bound check: `if index < 0: index += l.ll_length(); return
/// basegetitem(l, index)`. 3-block CFG (start → block_neg_fix → block_dispatch)
/// forwarding the possibly-fixed index to a `direct_call` of the layout's
/// `ll_*_getitem_fast`.
fn build_ll_list_getitem_neg_helper_graph(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    name: &str,
    foldable: bool,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let c_fast = list_getitem_fast_funcptr(
        rtyper,
        layout,
        foldable,
        ptr_lltype.clone(),
        item_lltype.clone(),
    )?;

    let l = variable_with_lltype("l", ptr_lltype.clone());
    let i = variable_with_lltype("index", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l.clone()),
        Hlvalue::Variable(i.clone()),
    ]);
    let return_var = variable_with_lltype("result", item_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let l_fix = variable_with_lltype("l", ptr_lltype.clone());
    let i_fix = variable_with_lltype("index", LowLevelType::Signed);
    let block_neg_fix = Block::shared(vec![
        Hlvalue::Variable(l_fix.clone()),
        Hlvalue::Variable(i_fix.clone()),
    ]);

    let l_disp = variable_with_lltype("l", ptr_lltype);
    let i_disp = variable_with_lltype("index", LowLevelType::Signed);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(l_disp.clone()),
        Hlvalue::Variable(i_disp.clone()),
    ]);

    // start: is_neg = int_lt(index, 0); branch.
    let is_neg = variable_with_lltype("is_neg", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(i.clone()), signed_const(0)],
        Hlvalue::Variable(is_neg.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_neg));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(l.clone()), Hlvalue::Variable(i.clone())],
            Some(block_neg_fix.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(l), Hlvalue::Variable(i)],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_neg_fix: length = <len read>; i_fixed = int_add(index, length).
    let length = emit_list_length_read(&block_neg_fix, layout, &l_fix);
    let i_fixed = variable_with_lltype("index", LowLevelType::Signed);
    block_neg_fix
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_fix), Hlvalue::Variable(length)],
            Hlvalue::Variable(i_fixed.clone()),
        ));
    block_neg_fix.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(l_fix), Hlvalue::Variable(i_fixed)],
            Some(block_dispatch.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_dispatch: c = direct_call(fast, l, index); return c.
    let c = variable_with_lltype("c", item_lltype);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_fast),
                Hlvalue::Variable(l_disp),
                Hlvalue::Variable(i_disp),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    block_dispatch.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(c)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string(), "index".to_string()],
        func,
    ))
}

/// rlist.py `ll_getitem_nonneg` with `func is dum_checkidx` —
/// nonneg index, bound check: `if index >= l.ll_length(): raise IndexError;
/// return basegetitem(l, index)`. 2-block CFG plus graph.exceptblock.
fn build_ll_list_getitem_nonneg_checked_helper_graph(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    name: &str,
    foldable: bool,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let c_fast = list_getitem_fast_funcptr(
        rtyper,
        layout,
        foldable,
        ptr_lltype.clone(),
        item_lltype.clone(),
    )?;
    let exc_args = exception_args("IndexError")?;

    let l = variable_with_lltype("l", ptr_lltype.clone());
    let i = variable_with_lltype("index", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l.clone()),
        Hlvalue::Variable(i.clone()),
    ]);
    let return_var = variable_with_lltype("result", item_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let l_disp = variable_with_lltype("l", ptr_lltype);
    let i_disp = variable_with_lltype("index", LowLevelType::Signed);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(l_disp.clone()),
        Hlvalue::Variable(i_disp.clone()),
    ]);

    // start: length = <len read>; oob = int_ge(index, length); branch.
    let length = emit_list_length_read(&startblock, layout, &l);
    let oob = variable_with_lltype("oob", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ge",
        vec![Hlvalue::Variable(i.clone()), Hlvalue::Variable(length)],
        Hlvalue::Variable(oob.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(oob));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(l), Hlvalue::Variable(i)],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_dispatch: c = direct_call(fast, l, index); return c.
    let c = variable_with_lltype("c", item_lltype);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_fast),
                Hlvalue::Variable(l_disp),
                Hlvalue::Variable(i_disp),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    block_dispatch.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(c)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string(), "index".to_string()],
        func,
    ))
}

/// rlist.py `ll_getitem` with `func is dum_checkidx` — the negative
/// index is folded in (`index += length`) then the `0 <= index < length`
/// window is enforced, raising IndexError otherwise. The r_uint window test
/// is lowered to the signed-explicit `index >= length or index < 0` form
/// (matching the `ll_stritem_checked` lowering): 5-block CFG (start →
/// block_neg_fix → block_check_high → block_check_low → block_dispatch) plus
/// graph.exceptblock.
fn build_ll_list_getitem_checked_helper_graph(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    name: &str,
    foldable: bool,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let c_fast = list_getitem_fast_funcptr(
        rtyper,
        layout,
        foldable,
        ptr_lltype.clone(),
        item_lltype.clone(),
    )?;
    let exc_args = exception_args("IndexError")?;

    let l = variable_with_lltype("l", ptr_lltype.clone());
    let i = variable_with_lltype("index", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l.clone()),
        Hlvalue::Variable(i.clone()),
    ]);
    let return_var = variable_with_lltype("result", item_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let l_fix = variable_with_lltype("l", ptr_lltype.clone());
    let i_fix_u = variable_with_lltype("index", LowLevelType::Unsigned);
    let len_fix_u = variable_with_lltype("length", LowLevelType::Unsigned);
    let block_fixup = Block::shared(vec![
        Hlvalue::Variable(l_fix.clone()),
        Hlvalue::Variable(i_fix_u.clone()),
        Hlvalue::Variable(len_fix_u.clone()),
    ]);

    let l_disp = variable_with_lltype("l", ptr_lltype);
    let i_disp = variable_with_lltype("index", LowLevelType::Signed);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(l_disp.clone()),
        Hlvalue::Variable(i_disp.clone()),
    ]);

    // start: length = <len read>; index_u = cast_int_to_uint(index);
    //      length_u = cast_int_to_uint(length); oob = uint_ge(index_u, length_u);
    //      branch.  The common 0 <= index < length case falls straight through
    //      with no add (`ll_getitem`, rlist.py).
    let length = emit_list_length_read(&startblock, layout, &l);
    let i_u = variable_with_lltype("index", LowLevelType::Unsigned);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_int_to_uint",
        vec![Hlvalue::Variable(i.clone())],
        Hlvalue::Variable(i_u.clone()),
    ));
    let len_u = variable_with_lltype("length", LowLevelType::Unsigned);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_int_to_uint",
        vec![Hlvalue::Variable(length)],
        Hlvalue::Variable(len_u.clone()),
    ));
    let oob = variable_with_lltype("oob", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "uint_ge",
        vec![
            Hlvalue::Variable(i_u.clone()),
            Hlvalue::Variable(len_u.clone()),
        ],
        Hlvalue::Variable(oob.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(oob));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l.clone()),
                Hlvalue::Variable(i_u),
                Hlvalue::Variable(len_u),
            ],
            Some(block_fixup.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(l), Hlvalue::Variable(i)],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_fixup: index_u = uint_add(index_u, length_u);
    //      if uint_ge(index_u, length_u): raise IndexError;
    //      index = intmask(index_u); -> dispatch.
    let i_fixed_u = variable_with_lltype("index", LowLevelType::Unsigned);
    block_fixup
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "uint_add",
            vec![
                Hlvalue::Variable(i_fix_u),
                Hlvalue::Variable(len_fix_u.clone()),
            ],
            Hlvalue::Variable(i_fixed_u.clone()),
        ));
    let oob2 = variable_with_lltype("oob", LowLevelType::Bool);
    block_fixup
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "uint_ge",
            vec![
                Hlvalue::Variable(i_fixed_u.clone()),
                Hlvalue::Variable(len_fix_u),
            ],
            Hlvalue::Variable(oob2.clone()),
        ));
    let i_fixed = variable_with_lltype("index", LowLevelType::Signed);
    block_fixup
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "cast_uint_to_int",
            vec![Hlvalue::Variable(i_fixed_u)],
            Hlvalue::Variable(i_fixed.clone()),
        ));
    block_fixup.borrow_mut().exitswitch = Some(Hlvalue::Variable(oob2));
    block_fixup.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(l_fix), Hlvalue::Variable(i_fixed)],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_dispatch: c = direct_call(fast, l, index); return c.
    let c = variable_with_lltype("c", item_lltype);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_fast),
                Hlvalue::Variable(l_disp),
                Hlvalue::Variable(i_disp),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    block_dispatch.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(c)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string(), "index".to_string()],
        func,
    ))
}

/// Build (or retrieve cached) the list-getitem helper for the
/// `(checkidx, nonneg)` combination (`rlist.py:247-267`), selecting the
/// `dum_nocheck` fast path, the negative-index `ll_getitem`, the nonneg
/// `dum_checkidx` `ll_getitem_nonneg`, or the full checked `ll_getitem`.
fn list_getitem_helper(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    checkidx: bool,
    nonneg: bool,
    foldable: bool,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<LowLevelFunction, TyperError> {
    // rlist.py `basegetitem` selection by the listdef's `mutated` flag,
    // passed into ALL four (nonneg×checkidx) wrappers: the foldable
    // `ll_getitem_foldable_nonneg` (rlist.py) when not mutated, else
    // `ll_getitem_fast`.  Each foldable wrapper gets a distinct `_foldable`
    // cache name so `lowlevel_helper_function_with_builder` never serves a
    // foldable graph to a mutated list of the same item_lltype.  Only Fixed has
    // a foldable element load; Resized is always mutated (the `mutated | resized`
    // invariant in ListDef construction) so `foldable` is never set for it.
    let base = match (checkidx, nonneg) {
        (false, true) => layout.getitem_fast_name(),
        (false, false) => layout.getitem_neg_name(),
        (true, true) => layout.getitem_nonneg_checked_name(),
        (true, false) => layout.getitem_checked_name(),
    };
    let name = if foldable {
        format!("{base}_foldable")
    } else {
        base.to_string()
    };
    let name_owned = name.clone();
    let ptr_for_builder = ptr_lltype.clone();
    let item_for_builder = item_lltype.clone();
    rtyper.lowlevel_helper_function_with_builder(
        name,
        vec![ptr_lltype, LowLevelType::Signed],
        item_lltype,
        move |rtyper_inner, _args, _result| match (checkidx, nonneg) {
            (false, true) => match layout {
                // rlist.py `ll_getitem_foldable_nonneg` — the Fixed
                // fast load, but foldable (`oopspec = 'list.getitem_foldable'`).
                // The Resized layout is always mutated so `foldable` can only
                // fire on Fixed.
                ListLayout::Fixed if foldable => build_ll_fixed_getitem_fast_foldable_helper_graph(
                    &name_owned,
                    ptr_for_builder.clone(),
                    item_for_builder.clone(),
                ),
                ListLayout::Fixed => build_ll_fixed_getitem_fast_helper_graph(
                    &name_owned,
                    ptr_for_builder.clone(),
                    item_for_builder.clone(),
                ),
                ListLayout::Resized => build_ll_getitem_fast_helper_graph(
                    &name_owned,
                    ptr_for_builder.clone(),
                    item_for_builder.clone(),
                ),
            },
            (false, false) => build_ll_list_getitem_neg_helper_graph(
                rtyper_inner,
                layout,
                &name_owned,
                foldable,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
            (true, true) => build_ll_list_getitem_nonneg_checked_helper_graph(
                rtyper_inner,
                layout,
                &name_owned,
                foldable,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
            (true, false) => build_ll_list_getitem_checked_helper_graph(
                rtyper_inner,
                layout,
                &name_owned,
                foldable,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
        },
    )
}

/// Shared `pair(AbstractBaseListRepr, IntegerRepr).rtype_getitem`
/// (`rlist.py`) for both list layouts. The `getitem_idx` op collapses
/// onto `getitem` in the rtyper dispatch, so `hop.has_implicit_exception` is
/// the `dum_checkidx` selector — the same caught-IndexError signal
/// `rtype_setitem` uses (`rlist.py`). Returns the (internal-repr) element
/// value; the caller applies `recast`.
fn list_rtype_getitem(
    hop: &HighLevelOp,
    self_repr: &dyn Repr,
    layout: ListLayout,
    ptr_lltype: &LowLevelType,
    item_lltype: &LowLevelType,
) -> Result<Hlvalue, TyperError> {
    use crate::annotator::model::SomeValue;
    let checkidx = hop.has_implicit_exception("IndexError");
    // rlist.py:255-258 `basegetitem` selection by the listdef's mutated
    // flag: `ll_getitem_fast` (non-foldable) when `listdef.listitem.mutated`,
    // else `ll_getitem_foldable_nonneg` (foldable, `oopspec =
    // 'list.getitem_foldable(l, index)'`, rlist.py:721-724). `mutated` is the
    // gate, NOT the layout — a non-resized FixedSizeListRepr that is still
    // mutated (setitem without resize) is non-foldable; a Resized list carries
    // the `mutated | resized` construction invariant (listdef.py /
    // listdef.rs) so it is always mutated ⟹ never foldable.
    let s0 = hop
        .args_s
        .borrow()
        .first()
        .cloned()
        .ok_or_else(|| TyperError::message("list rtype_getitem: args_s[0] missing"))?;
    let foldable = match &s0 {
        SomeValue::List(lst) => !lst.listdef.listitem_rc().borrow().mutated,
        other => {
            return Err(TyperError::message(format!(
                "list rtype_getitem: args_s[0] must be SomeList, got {other:?}"
            )));
        }
    };
    let s1 = hop
        .args_s
        .borrow()
        .get(1)
        .cloned()
        .ok_or_else(|| TyperError::message("list rtype_getitem: args_s[1] missing"))?;
    let nonneg = match &s1 {
        SomeValue::Integer(i) => i.nonneg,
        other => {
            return Err(TyperError::message(format!(
                "list rtype_getitem: args_s[1] must be SomeInteger, got {other:?}"
            )));
        }
    };
    let args = hop.inputargs(vec![
        ConvertedTo::Repr(self_repr),
        ConvertedTo::LowLevelType(&LowLevelType::Signed),
    ])?;
    if checkidx {
        hop.exception_is_here()?;
    } else {
        hop.exception_cannot_occur()?;
    }
    let helper = list_getitem_helper(
        &hop.rtyper,
        layout,
        checkidx,
        nonneg,
        foldable,
        ptr_lltype.clone(),
        item_lltype.clone(),
    )?;
    hop.gendirectcall(&helper, args)?
        .ok_or_else(|| TyperError::message("list getitem helper unexpectedly returned Void"))
}

/// The `None` (Void) constant a void-returning helper's returnblock link
/// carries (`rlist.py` setitem helpers return nothing).
fn none_void() -> Hlvalue {
    Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    ))
}

/// Build (or retrieve cached) the layout's `ll_*_setitem_fast` sub-helper and
/// return a funcptr `Constant` to `direct_call` it (the `basesetitem` of
/// `rlist.py:283`).
fn list_setitem_fast_funcptr(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<Constant, TyperError> {
    let name = layout.setitem_fast_name().to_string();
    let name_owned = name.clone();
    let ptr_for_builder = ptr_lltype.clone();
    let item_for_builder = item_lltype.clone();
    let inner = rtyper.lowlevel_helper_function_with_builder(
        name,
        vec![ptr_lltype, LowLevelType::Signed, item_lltype],
        LowLevelType::Void,
        move |_rtyper, _args, _result| match layout {
            ListLayout::Fixed => build_ll_fixed_setitem_fast_helper_graph(
                &name_owned,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
            ListLayout::Resized => build_ll_setitem_fast_helper_graph(
                &name_owned,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
        },
    )?;
    sub_helper_funcptr_constant(rtyper, &inner)
}

/// rlist.py `ll_setitem` with `func is dum_nocheck` — negative index,
/// no bound check: `if index < 0: index += l.ll_length(); l.ll_setitem_fast(
/// index, item)`. 3-block CFG forwarding the possibly-fixed index + item to a
/// `direct_call` of the layout's `ll_*_setitem_fast` (Void).
fn build_ll_list_setitem_neg_helper_graph(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let c_fast =
        list_setitem_fast_funcptr(rtyper, layout, ptr_lltype.clone(), item_lltype.clone())?;

    let l = variable_with_lltype("l", ptr_lltype.clone());
    let i = variable_with_lltype("index", LowLevelType::Signed);
    let item = variable_with_lltype("item", item_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l.clone()),
        Hlvalue::Variable(i.clone()),
        Hlvalue::Variable(item.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let l_fix = variable_with_lltype("l", ptr_lltype.clone());
    let i_fix = variable_with_lltype("index", LowLevelType::Signed);
    let item_fix = variable_with_lltype("item", item_lltype.clone());
    let block_neg_fix = Block::shared(vec![
        Hlvalue::Variable(l_fix.clone()),
        Hlvalue::Variable(i_fix.clone()),
        Hlvalue::Variable(item_fix.clone()),
    ]);

    let l_disp = variable_with_lltype("l", ptr_lltype);
    let i_disp = variable_with_lltype("index", LowLevelType::Signed);
    let item_disp = variable_with_lltype("item", item_lltype);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(l_disp.clone()),
        Hlvalue::Variable(i_disp.clone()),
        Hlvalue::Variable(item_disp.clone()),
    ]);

    // start: is_neg = int_lt(index, 0); branch.
    let is_neg = variable_with_lltype("is_neg", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(i.clone()), signed_const(0)],
        Hlvalue::Variable(is_neg.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_neg));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l.clone()),
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(item.clone()),
            ],
            Some(block_neg_fix.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(l),
                Hlvalue::Variable(i),
                Hlvalue::Variable(item),
            ],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_neg_fix: length = <len read>; i_fixed = int_add(index, length).
    let length = emit_list_length_read(&block_neg_fix, layout, &l_fix);
    let i_fixed = variable_with_lltype("index", LowLevelType::Signed);
    block_neg_fix
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_fix), Hlvalue::Variable(length)],
            Hlvalue::Variable(i_fixed.clone()),
        ));
    block_neg_fix.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_fix),
                Hlvalue::Variable(i_fixed),
                Hlvalue::Variable(item_fix),
            ],
            Some(block_dispatch.clone()),
            None,
        )
        .into_ref(),
    ]);

    // block_dispatch: direct_call(fast, l, index, item); return None.
    let v_void = variable_with_lltype("v", LowLevelType::Void);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_fast),
                Hlvalue::Variable(l_disp),
                Hlvalue::Variable(i_disp),
                Hlvalue::Variable(item_disp),
            ],
            Hlvalue::Variable(v_void),
        ));
    block_dispatch.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string(), "index".to_string(), "item".to_string()],
        func,
    ))
}

/// rlist.py `ll_setitem_nonneg` with `func is dum_checkidx` — nonneg
/// index, bound check: `if index >= l.ll_length(): raise IndexError;
/// l.ll_setitem_fast(index, item)`. 2-block CFG plus graph.exceptblock.
fn build_ll_list_setitem_nonneg_checked_helper_graph(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let c_fast =
        list_setitem_fast_funcptr(rtyper, layout, ptr_lltype.clone(), item_lltype.clone())?;
    let exc_args = exception_args("IndexError")?;

    let l = variable_with_lltype("l", ptr_lltype.clone());
    let i = variable_with_lltype("index", LowLevelType::Signed);
    let item = variable_with_lltype("item", item_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l.clone()),
        Hlvalue::Variable(i.clone()),
        Hlvalue::Variable(item.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let l_disp = variable_with_lltype("l", ptr_lltype);
    let i_disp = variable_with_lltype("index", LowLevelType::Signed);
    let item_disp = variable_with_lltype("item", item_lltype);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(l_disp.clone()),
        Hlvalue::Variable(i_disp.clone()),
        Hlvalue::Variable(item_disp.clone()),
    ]);

    // start: length = <len read>; oob = int_ge(index, length); branch.
    let length = emit_list_length_read(&startblock, layout, &l);
    let oob = variable_with_lltype("oob", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ge",
        vec![Hlvalue::Variable(i.clone()), Hlvalue::Variable(length)],
        Hlvalue::Variable(oob.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(oob));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(l),
                Hlvalue::Variable(i),
                Hlvalue::Variable(item),
            ],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_dispatch: direct_call(fast, l, index, item); return None.
    let v_void = variable_with_lltype("v", LowLevelType::Void);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_fast),
                Hlvalue::Variable(l_disp),
                Hlvalue::Variable(i_disp),
                Hlvalue::Variable(item_disp),
            ],
            Hlvalue::Variable(v_void),
        ));
    block_dispatch.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string(), "index".to_string(), "item".to_string()],
        func,
    ))
}

/// rlist.py `ll_setitem` with `func is dum_checkidx` — fold the
/// negative index in (`index += length`) then enforce the `0 <= index <
/// length` window, raising IndexError otherwise (the r_uint window test
/// lowered to the signed-explicit `index >= length or index < 0` form, as in
/// [`build_ll_list_getitem_checked_helper_graph`]): 5-block CFG plus
/// graph.exceptblock.
fn build_ll_list_setitem_checked_helper_graph(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let c_fast =
        list_setitem_fast_funcptr(rtyper, layout, ptr_lltype.clone(), item_lltype.clone())?;
    let exc_args = exception_args("IndexError")?;

    let l = variable_with_lltype("l", ptr_lltype.clone());
    let i = variable_with_lltype("index", LowLevelType::Signed);
    let item = variable_with_lltype("item", item_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l.clone()),
        Hlvalue::Variable(i.clone()),
        Hlvalue::Variable(item.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let l_fix = variable_with_lltype("l", ptr_lltype.clone());
    let i_fix_u = variable_with_lltype("index", LowLevelType::Unsigned);
    let len_fix_u = variable_with_lltype("length", LowLevelType::Unsigned);
    let item_fix = variable_with_lltype("item", item_lltype.clone());
    let block_fixup = Block::shared(vec![
        Hlvalue::Variable(l_fix.clone()),
        Hlvalue::Variable(i_fix_u.clone()),
        Hlvalue::Variable(len_fix_u.clone()),
        Hlvalue::Variable(item_fix.clone()),
    ]);

    let l_disp = variable_with_lltype("l", ptr_lltype);
    let i_disp = variable_with_lltype("index", LowLevelType::Signed);
    let item_disp = variable_with_lltype("item", item_lltype);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(l_disp.clone()),
        Hlvalue::Variable(i_disp.clone()),
        Hlvalue::Variable(item_disp.clone()),
    ]);

    // start: length = <len read>; index_u = cast_int_to_uint(index);
    //      length_u = cast_int_to_uint(length); oob = uint_ge(index_u, length_u);
    //      branch.  The common 0 <= index < length case falls straight through
    //      with no add (`ll_setitem`, rlist.py).
    let length = emit_list_length_read(&startblock, layout, &l);
    let i_u = variable_with_lltype("index", LowLevelType::Unsigned);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_int_to_uint",
        vec![Hlvalue::Variable(i.clone())],
        Hlvalue::Variable(i_u.clone()),
    ));
    let len_u = variable_with_lltype("length", LowLevelType::Unsigned);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_int_to_uint",
        vec![Hlvalue::Variable(length)],
        Hlvalue::Variable(len_u.clone()),
    ));
    let oob = variable_with_lltype("oob", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "uint_ge",
        vec![
            Hlvalue::Variable(i_u.clone()),
            Hlvalue::Variable(len_u.clone()),
        ],
        Hlvalue::Variable(oob.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(oob));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l.clone()),
                Hlvalue::Variable(i_u),
                Hlvalue::Variable(len_u),
                Hlvalue::Variable(item.clone()),
            ],
            Some(block_fixup.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(l),
                Hlvalue::Variable(i),
                Hlvalue::Variable(item),
            ],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_fixup: index_u = uint_add(index_u, length_u);
    //      if uint_ge(index_u, length_u): raise IndexError;
    //      index = intmask(index_u); -> dispatch.
    let i_fixed_u = variable_with_lltype("index", LowLevelType::Unsigned);
    block_fixup
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "uint_add",
            vec![
                Hlvalue::Variable(i_fix_u),
                Hlvalue::Variable(len_fix_u.clone()),
            ],
            Hlvalue::Variable(i_fixed_u.clone()),
        ));
    let oob2 = variable_with_lltype("oob", LowLevelType::Bool);
    block_fixup
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "uint_ge",
            vec![
                Hlvalue::Variable(i_fixed_u.clone()),
                Hlvalue::Variable(len_fix_u),
            ],
            Hlvalue::Variable(oob2.clone()),
        ));
    let i_fixed = variable_with_lltype("index", LowLevelType::Signed);
    block_fixup
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "cast_uint_to_int",
            vec![Hlvalue::Variable(i_fixed_u)],
            Hlvalue::Variable(i_fixed.clone()),
        ));
    block_fixup.borrow_mut().exitswitch = Some(Hlvalue::Variable(oob2));
    block_fixup.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(l_fix),
                Hlvalue::Variable(i_fixed),
                Hlvalue::Variable(item_fix),
            ],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // block_dispatch: direct_call(fast, l, index, item); return None.
    let v_void = variable_with_lltype("v", LowLevelType::Void);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_fast),
                Hlvalue::Variable(l_disp),
                Hlvalue::Variable(i_disp),
                Hlvalue::Variable(item_disp),
            ],
            Hlvalue::Variable(v_void),
        ));
    block_dispatch.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string(), "index".to_string(), "item".to_string()],
        func,
    ))
}

/// Build (or retrieve cached) the list-setitem helper for the
/// `(checkidx, nonneg)` combination (`rlist.py:272-284`).
fn list_setitem_helper(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    checkidx: bool,
    nonneg: bool,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<LowLevelFunction, TyperError> {
    let name = match (checkidx, nonneg) {
        (false, true) => layout.setitem_fast_name(),
        (false, false) => layout.setitem_neg_name(),
        (true, true) => layout.setitem_nonneg_checked_name(),
        (true, false) => layout.setitem_checked_name(),
    }
    .to_string();
    let name_owned = name.clone();
    let ptr_for_builder = ptr_lltype.clone();
    let item_for_builder = item_lltype.clone();
    rtyper.lowlevel_helper_function_with_builder(
        name,
        vec![ptr_lltype, LowLevelType::Signed, item_lltype],
        LowLevelType::Void,
        move |rtyper_inner, _args, _result| match (checkidx, nonneg) {
            (false, true) => match layout {
                ListLayout::Fixed => build_ll_fixed_setitem_fast_helper_graph(
                    &name_owned,
                    ptr_for_builder.clone(),
                    item_for_builder.clone(),
                ),
                ListLayout::Resized => build_ll_setitem_fast_helper_graph(
                    &name_owned,
                    ptr_for_builder.clone(),
                    item_for_builder.clone(),
                ),
            },
            (false, false) => build_ll_list_setitem_neg_helper_graph(
                rtyper_inner,
                layout,
                &name_owned,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
            (true, true) => build_ll_list_setitem_nonneg_checked_helper_graph(
                rtyper_inner,
                layout,
                &name_owned,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
            (true, false) => build_ll_list_setitem_checked_helper_graph(
                rtyper_inner,
                layout,
                &name_owned,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
        },
    )
}

/// Shared `pair(AbstractBaseListRepr, IntegerRepr).rtype_setitem`
/// (`rlist.py:272-284`) for both list layouts. `hop.has_implicit_exception`
/// is the `dum_checkidx` selector; the item is the third inputarg, converted
/// to the internal `item_repr` (so `rtype_setitem` does not `recast`).
fn list_rtype_setitem(
    hop: &HighLevelOp,
    self_repr: &dyn Repr,
    layout: ListLayout,
    ptr_lltype: &LowLevelType,
    item_lltype: &LowLevelType,
    item_repr: &dyn Repr,
) -> RTypeResult {
    use crate::annotator::model::SomeValue;
    let checkidx = hop.has_implicit_exception("IndexError");
    let s1 = hop
        .args_s
        .borrow()
        .get(1)
        .cloned()
        .ok_or_else(|| TyperError::message("list rtype_setitem: args_s[1] missing"))?;
    let nonneg = match &s1 {
        SomeValue::Integer(i) => i.nonneg,
        other => {
            return Err(TyperError::message(format!(
                "list rtype_setitem: args_s[1] must be SomeInteger, got {other:?}"
            )));
        }
    };
    let args = hop.inputargs(vec![
        ConvertedTo::Repr(self_repr),
        ConvertedTo::LowLevelType(&LowLevelType::Signed),
        ConvertedTo::Repr(item_repr),
    ])?;
    hop.exception_is_here()?;
    let helper = list_setitem_helper(
        &hop.rtyper,
        layout,
        checkidx,
        nonneg,
        ptr_lltype.clone(),
        item_lltype.clone(),
    )?;
    hop.gendirectcall(&helper, args)
}

/// RPython `BaseListRepr.make_iterator_repr(self, *variant)`
/// (`lltypesystem/rlist.py`):
///
/// ```python
/// def make_iterator_repr(self, *variant):
///     if not variant:
///         return ListIteratorRepr(self)
///     elif variant == ("reversed",):
///         return ReversedListIteratorRepr(self)
///     else:
///         raise TyperError("unsupported %r iterator over a list" % (variant,))
/// ```
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
)]
fn list_make_iterator_repr(
    list_lltype: LowLevelType,
    item_repr: Arc<dyn Repr>,
    external_item_repr: Arc<dyn Repr>,
    list_is_fixed: bool,
    foldable: bool,
    variant: &[String],
) -> Result<Arc<dyn Repr>, TyperError> {
    match variant {
        [] => Ok(Arc::new(ListIteratorRepr::new(
            list_lltype,
            item_repr,
            external_item_repr,
            list_is_fixed,
            foldable,
        )?)),
        [name] if name == "reversed" => Ok(Arc::new(ReversedListIteratorRepr::new(
            list_lltype,
            item_repr,
            external_item_repr,
            list_is_fixed,
        )?)),
        _ => Err(TyperError::message(format!(
            "unsupported {variant:?} iterator over a list"
        ))),
    }
}

/// RPython `class ListIteratorRepr(AbstractListIteratorRepr)`
/// (`lltypesystem/rlist.py`):
///
/// ```python
/// class ListIteratorRepr(AbstractListIteratorRepr):
///     def __init__(self, r_list):
///         self.r_list = r_list
///         self.lowleveltype = Ptr(GcStruct('listiter',
///             ('list', r_list.lowleveltype),
///             ('index', Signed)))
///         self.ll_listiter = ll_listiter
///         self.ll_listnext = ll_listnext
///         self.ll_getnextindex = ll_getnextindex
/// ```
///
/// The Rust port stores the list's `lowleveltype` + `item_repr` + resized
/// flag rather than the full `r_list` repr: `make_iterator_repr` is a
/// `&self` method on the list repr and cannot reproduce the `Arc<dyn
/// Repr>` the upstream `r_list` field holds, so the iterator carries
/// exactly the data its `ll_listiter` / `ll_listnext` helpers consume —
/// the iter struct shape, the element repr, and the `getarraysize`-vs-
/// `length`-field length distinction.
#[derive(Debug)]
pub struct ListIteratorRepr {
    state: ReprState,
    /// `Ptr(GcStruct('listiter', ('list', LIST), ('index', Signed)))`.
    lltype: LowLevelType,
    /// `r_list.lowleveltype` — the `list` field type and the receiver
    /// `ll_listnext` reads through.
    list_lltype: LowLevelType,
    /// `r_list.item_repr` — the element repr `ll_listnext` returns.
    item_repr: Arc<dyn Repr>,
    /// `r_list.external_item_repr` (`lltypesystem/rlist.py`
    /// `self.external_item_repr = r_list.external_item_repr`) — the surface
    /// element repr `rtype_next` recasts the `ll_listnext` result back to
    /// (identity for primitive items; see [`list_recast`]).
    external_item_repr: Arc<dyn Repr>,
    /// The `ll_length` read-out SHAPE only: `true` for `FixedSizeListRepr`
    /// (length via `getarraysize` on the bare `Ptr(GcArray)`), `false` for
    /// the resized `ListRepr` (length via `getfield(l, "length")` on the
    /// header struct). Distinct from [`Self::foldable`]: the length read-out is
    /// the same `getarraysize`/`length` op whether or not the element load
    /// folds.
    list_is_fixed: bool,
    /// `not r_list.listitem.mutated` (`lltypesystem/rlist.py`): selects
    /// `ll_listnext_foldable` over `ll_listnext`, so an unmutated
    /// `FixedSizeListRepr`'s element load lowers to the PURE `getarrayitem_pure`
    /// the optimizer can fold / CSE across iterations. Only meaningful together
    /// with `list_is_fixed` — a `FixedSizeListRepr` can still be `mutated`
    /// (in-place setitem without resize) and a resized `ListRepr` is always
    /// `mutated`, so `rtype_next` gates the foldable read on
    /// `list_is_fixed && foldable`.
    foldable: bool,
    /// Header field `ll_listnext` reads for length (`"length"` on
    /// `ListRepr`, `"len"` on a `RustVec` header).
    length_field: &'static str,
    /// Header field `ll_listnext` reads for the item array (`"items"` on
    /// `ListRepr`, `"ptr"` on a `RustVec` header).
    items_field: &'static str,
}

impl ListIteratorRepr {
    pub fn new(
        list_lltype: LowLevelType,
        item_repr: Arc<dyn Repr>,
        external_item_repr: Arc<dyn Repr>,
        list_is_fixed: bool,
        foldable: bool,
    ) -> Result<Self, TyperError> {
        Self::new_with_header_fields(
            list_lltype,
            item_repr,
            external_item_repr,
            list_is_fixed,
            foldable,
            "length",
            "items",
        )
    }

    /// Same iterator layout as [`Self::new`], reading a raw header whose
    /// length/items fields are not the list struct's `"length"` / `"items"`.
    /// `RustVecRepr.make_iterator_repr` uses this for `{ptr, len, cap}`.
    pub fn new_with_header_fields(
        list_lltype: LowLevelType,
        item_repr: Arc<dyn Repr>,
        external_item_repr: Arc<dyn Repr>,
        list_is_fixed: bool,
        foldable: bool,
        length_field: &'static str,
        items_field: &'static str,
    ) -> Result<Self, TyperError> {
        // upstream `Ptr(GcStruct('listiter', ('list', r_list.lowleveltype),
        // ('index', Signed)))`.
        let listiter_struct = Struct::gc(
            "listiter",
            vec![
                ("list".to_string(), list_lltype.clone()),
                ("index".to_string(), LowLevelType::Signed),
            ],
        );
        let lltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(listiter_struct),
        }));
        Ok(ListIteratorRepr {
            state: ReprState::new(),
            lltype,
            list_lltype,
            item_repr,
            external_item_repr,
            list_is_fixed,
            foldable,
            length_field,
            items_field,
        })
    }
}

impl ListIteratorRepr {
    /// `self.r_list.external_item_repr` — the item repr `rtype_next`
    /// hands back, read by `EnumerateIteratorRepr.rtype_next` (`rrange.py`)
    /// as `r_item_src`.
    pub(crate) fn external_item_repr(&self) -> &Arc<dyn Repr> {
        &self.external_item_repr
    }

    /// `hop.gendirectcall(self.ll_getnextindex, v_iter)` — the call
    /// `EnumerateIteratorRepr.rtype_next` (`rrange.py`) makes through
    /// `self.ll_getnextindex = r_baseiter.ll_getnextindex`.
    pub(crate) fn gen_ll_getnextindex(
        &self,
        hop: &HighLevelOp,
        v_iter: Hlvalue,
    ) -> Result<Hlvalue, TyperError> {
        let iter_lltype = self.lltype.clone();
        let iter_for_builder = iter_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_getnextindex".to_string(),
            vec![iter_lltype],
            LowLevelType::Signed,
            move |_rtyper, _args, _result| {
                build_ll_getnextindex_helper_graph("ll_getnextindex", iter_for_builder.clone())
            },
        )?;
        hop.gendirectcall(&helper, vec![v_iter])?
            .ok_or_else(|| TyperError::message("ll_getnextindex returned Void"))
    }
}

/// Synthesise `ll_getnextindex` (`lltypesystem/rlist.py`):
///
/// ```python
/// def ll_getnextindex(iter):
///     return iter.index
/// ```
pub(crate) fn build_ll_getnextindex_helper_graph(
    name: &str,
    iter_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("iter", iter_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_index = variable_with_lltype("index", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(arg), void_field_const("index")],
        Hlvalue::Variable(v_index.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_index)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["iter".to_string()],
        func,
    ))
}

impl Repr for ListIteratorRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "ListIteratorRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::ListIteratorRepr
    }

    /// RPython `IteratorRepr.rtype_iter(self, hop)` (rmodel.py) —
    /// `iter(iter(x)) <==> iter(x)`: the iterator is its own iterator, so
    /// the op is the identity on the receiver.
    fn rtype_iter(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        Ok(Some(vlist[0].clone()))
    }

    /// RPython `AbstractListIteratorRepr.newiter(self, hop)`
    /// (`rlist.py:439-442`):
    ///
    /// ```python
    /// def newiter(self, hop):
    ///     v_lst, = hop.inputargs(self.r_list)
    ///     citerptr = hop.inputconst(Void, self.lowleveltype)
    ///     return hop.gendirectcall(self.ll_listiter, citerptr, v_lst)
    /// ```
    ///
    /// The Void `citerptr` type-tag is baked into the `ll_listiter`
    /// helper's `malloc` op (the helper is minted per-`ListIteratorRepr`),
    /// matching how `TupleRepr.newtuple` bakes the struct lltype into its
    /// malloc rather than threading a Void runtime arg.
    fn newiter(&self, hop: &HighLevelOp) -> RTypeResult {
        // upstream `v_lst, = hop.inputargs(self.r_list)`. The iter()
        // operand's repr (the list repr) is `hop.args_r[0]`; using it as
        // the conversion target keeps the `convertvar` identity
        // short-circuit. (A non-primitive `Ptr` list lltype has no
        // primitive repr, so `ConvertedTo::LowLevelType` cannot convert
        // it — the list repr the operand already carries is required.)
        let r_list = {
            let args_r = hop.args_r.borrow();
            args_r
                .first()
                .and_then(|o| o.clone())
                .ok_or_else(|| TyperError::message("ListIteratorRepr.newiter: arg0 repr missing"))?
        };
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(r_list.as_ref())])?;
        hop.exception_cannot_occur()?;
        let list_lltype = self.list_lltype.clone();
        let listiter_lltype = self.lltype.clone();
        let list_for_builder = list_lltype.clone();
        let iter_for_builder = listiter_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_listiter".to_string(),
            vec![list_lltype],
            listiter_lltype,
            move |_rtyper, _args, _result| {
                build_ll_listiter_helper_graph(
                    "ll_listiter",
                    list_for_builder.clone(),
                    iter_for_builder.clone(),
                )
            },
        )?;
        hop.gendirectcall(&helper, vlist)
    }

    /// RPython `AbstractListIteratorRepr.rtype_next(self, hop)`
    /// (`rlist.py:444-449`):
    ///
    /// ```python
    /// def rtype_next(self, hop):
    ///     v_iter, = hop.inputargs(self)
    ///     hop.has_implicit_exception(StopIteration)
    ///     hop.exception_is_here()
    ///     v_res = hop.gendirectcall(self.ll_listnext, v_iter)
    ///     return self.r_list.recast(hop.llops, v_res)
    /// ```
    ///
    /// `ll_listnext` (the `index >= ll_length()` bounds-check that raises
    /// `StopIteration`) lowers to [`build_ll_listnext_helper_graph`]. For an
    /// unmutated `FixedSizeListRepr` (`self.foldable && self.list_is_fixed`) the
    /// helper is `ll_listnext_foldable` (`lltypesystem/rlist.py,
    /// 484-491`), whose element read is the PURE `getarrayitem_pure`
    /// (`ll_getitem_foldable_nonneg`, the `list.getitem_foldable` oopspec) the
    /// optimizer can fold / CSE across iterations; every other list keeps the
    /// plain `getarrayitem`. The two helpers carry distinct names so the
    /// per-signature helper cache never serves a folded body for a mutated list
    /// of the same element type.
    /// The upstream result `recast` (`rlist.py` `self.r_list.recast`,
    /// `rlist.py`) converts the `ll_listnext` result back to
    /// `external_item_repr` via [`list_recast`] — identity for primitive
    /// items (no op emitted), pairtype dispatch for a GC-instance element
    /// list.
    fn rtype_next(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_iter = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        hop.has_implicit_exception("StopIteration");
        hop.exception_is_here()?;
        let item_lltype = self.item_repr.lowleveltype().clone();
        let iter_lltype = self.lltype.clone();
        let list_lltype = self.list_lltype.clone();
        let list_is_fixed = self.list_is_fixed;
        // lltypesystem/rlist.py — an unmutated FixedSizeListRepr reads
        // each element through the PURE `getarrayitem_pure`
        // (`ll_getitem_foldable_nonneg`); every other list takes the plain
        // getarrayitem. A resized list is always mutated, so `list_is_fixed`
        // also gates it.
        let foldable = self.foldable && list_is_fixed;
        let helper_name = if foldable {
            "ll_listnext_foldable"
        } else {
            "ll_listnext"
        };
        let iter_for_builder = iter_lltype.clone();
        let list_for_builder = list_lltype.clone();
        let item_for_builder = item_lltype.clone();
        let length_field = self.length_field;
        let items_field = self.items_field;
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            helper_name.to_string(),
            vec![iter_lltype],
            item_lltype,
            move |_rtyper, _args, _result| {
                build_ll_listnext_helper_graph(
                    helper_name,
                    iter_for_builder.clone(),
                    list_for_builder.clone(),
                    item_for_builder.clone(),
                    list_is_fixed,
                    foldable,
                    length_field,
                    items_field,
                )
            },
        )?;
        let v_res = hop
            .gendirectcall(&helper, v_iter)?
            .ok_or_else(|| TyperError::message("list rtype_next: ll_listnext returned Void"))?;
        Ok(Some(list_recast(
            hop,
            &self.item_repr,
            &self.external_item_repr,
            v_res,
        )?))
    }
}

/// RPython `class ReversedListIteratorRepr(AbstractListIteratorRepr)`
/// (`lltypesystem/rlist.py`):
///
/// ```python
/// class ReversedListIteratorRepr(AbstractListIteratorRepr):
///     def __init__(self, r_list):
///         self.r_list = r_list
///         self.lowleveltype = Ptr(GcStruct('revlistiter',
///             ('list', r_list.lowleveltype),
///             ('index', Signed),
///         ))
///         self.ll_listnext = ll_revlistnext
///         self.ll_listiter = ll_revlistiter
/// ```
#[derive(Debug)]
pub struct ReversedListIteratorRepr {
    state: ReprState,
    /// `Ptr(GcStruct('revlistiter', ('list', LIST), ('index', Signed)))`.
    lltype: LowLevelType,
    /// `r_list.lowleveltype` — the `list` field type.
    list_lltype: LowLevelType,
    item_repr: Arc<dyn Repr>,
    external_item_repr: Arc<dyn Repr>,
    list_is_fixed: bool,
    length_field: &'static str,
    items_field: &'static str,
}

impl ReversedListIteratorRepr {
    pub fn new(
        list_lltype: LowLevelType,
        item_repr: Arc<dyn Repr>,
        external_item_repr: Arc<dyn Repr>,
        list_is_fixed: bool,
    ) -> Result<Self, TyperError> {
        // upstream `Ptr(GcStruct('revlistiter', ('list', r_list.lowleveltype),
        // ('index', Signed)))`.
        let revlistiter_struct = Struct::gc(
            "revlistiter",
            vec![
                ("list".to_string(), list_lltype.clone()),
                ("index".to_string(), LowLevelType::Signed),
            ],
        );
        let lltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(revlistiter_struct),
        }));
        Ok(ReversedListIteratorRepr {
            state: ReprState::new(),
            lltype,
            list_lltype,
            item_repr,
            external_item_repr,
            list_is_fixed,
            length_field: "length",
            items_field: "items",
        })
    }
}

impl Repr for ReversedListIteratorRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "ReversedListIteratorRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::ReversedListIteratorRepr
    }

    /// RPython `IteratorRepr.rtype_iter(self, hop)` (rmodel.py) —
    /// `iter(iter(x)) <==> iter(x)`.
    fn rtype_iter(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        Ok(Some(vlist[0].clone()))
    }

    /// RPython `AbstractListIteratorRepr.newiter(self, hop)` (`rlist.py`)
    /// with `self.ll_listiter = ll_revlistiter`.
    fn newiter(&self, hop: &HighLevelOp) -> RTypeResult {
        let r_list = {
            let args_r = hop.args_r.borrow();
            args_r.first().and_then(|o| o.clone()).ok_or_else(|| {
                TyperError::message("ReversedListIteratorRepr.newiter: arg0 repr missing")
            })?
        };
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(r_list.as_ref())])?;
        hop.exception_cannot_occur()?;
        let list_lltype = self.list_lltype.clone();
        let listiter_lltype = self.lltype.clone();
        let list_for_builder = list_lltype.clone();
        let iter_for_builder = listiter_lltype.clone();
        let list_is_fixed = self.list_is_fixed;
        let length_field = self.length_field;
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_revlistiter".to_string(),
            vec![list_lltype],
            listiter_lltype,
            move |_rtyper, _args, _result| {
                build_ll_revlistiter_helper_graph(
                    "ll_revlistiter",
                    list_for_builder.clone(),
                    iter_for_builder.clone(),
                    list_is_fixed,
                    length_field,
                )
            },
        )?;
        hop.gendirectcall(&helper, vlist)
    }

    /// RPython `AbstractListIteratorRepr.rtype_next(self, hop)` (`rlist.py`)
    /// with `self.ll_listnext = ll_revlistnext`.
    fn rtype_next(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_iter = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        hop.has_implicit_exception("StopIteration");
        hop.exception_is_here()?;
        let item_lltype = self.item_repr.lowleveltype().clone();
        let iter_lltype = self.lltype.clone();
        let list_lltype = self.list_lltype.clone();
        let list_is_fixed = self.list_is_fixed;
        let iter_for_builder = iter_lltype.clone();
        let list_for_builder = list_lltype.clone();
        let item_for_builder = item_lltype.clone();
        let items_field = self.items_field;
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_revlistnext".to_string(),
            vec![iter_lltype],
            item_lltype,
            move |_rtyper, _args, _result| {
                build_ll_revlistnext_helper_graph(
                    "ll_revlistnext",
                    iter_for_builder.clone(),
                    list_for_builder.clone(),
                    item_for_builder.clone(),
                    list_is_fixed,
                    items_field,
                )
            },
        )?;
        let v_res = hop.gendirectcall(&helper, v_iter)?.ok_or_else(|| {
            TyperError::message("reversed list rtype_next: ll_revlistnext returned Void")
        })?;
        Ok(Some(list_recast(
            hop,
            &self.item_repr,
            &self.external_item_repr,
            v_res,
        )?))
    }
}

/// Synthesise `ll_listiter` (`lltypesystem/rlist.py`):
///
/// ```python
/// def ll_listiter(ITERPTR, lst):
///     iter = malloc(ITERPTR.TO)
///     iter.list = lst
///     iter.index = 0
///     return iter
/// ```
///
/// Single-block graph: `malloc(listiter struct)` → `setfield(iter,
/// "list", lst)` → `setfield(iter, "index", 0)` → return iter. The
/// `ITERPTR` type-tag is baked into the `malloc` op's Void operand (the
/// helper is minted with the iter lltype known), so the runtime signature
/// is `ll_listiter(lst)`.
pub(crate) fn build_ll_listiter_helper_graph(
    name: &str,
    list_lltype: LowLevelType,
    listiter_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let lst_arg = variable_with_lltype("lst", list_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(lst_arg.clone())]);
    let return_var = variable_with_lltype("result", listiter_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // upstream `malloc(ITERPTR.TO)` — the Void `c1` carries the inner
    // listiter Struct lltype, `cflags` the gc flavor, exactly as
    // `TupleRepr.newtuple` encodes its malloc.
    let LowLevelType::Ptr(ptr) = &listiter_lltype else {
        return Err(TyperError::message(
            "build_ll_listiter_helper_graph: listiter lltype is not Ptr",
        ));
    };
    let inner_struct = match &ptr.TO {
        PtrTarget::Struct(body) => body.clone(),
        other => {
            return Err(TyperError::message(format!(
                "build_ll_listiter_helper_graph: Ptr target must be Struct, got {other:?}"
            )));
        }
    };
    let c1 = Constant::with_concretetype(
        ConstValue::LowLevelType(Box::new(LowLevelType::Struct(Box::new(inner_struct)))),
        LowLevelType::Void,
    );
    let cflags = Constant::with_concretetype(ConstValue::byte_str("flavor=gc"), LowLevelType::Void);
    let v_iter = variable_with_lltype("iter", listiter_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc",
        vec![Hlvalue::Constant(c1), Hlvalue::Constant(cflags)],
        Hlvalue::Variable(v_iter.clone()),
    ));
    // iter.list = lst
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(v_iter.clone()),
            void_field_const("list"),
            Hlvalue::Variable(lst_arg),
        ],
        Hlvalue::Variable(variable_with_lltype("v0", LowLevelType::Void)),
    ));
    // iter.index = 0
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(v_iter.clone()),
            void_field_const("index"),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(0),
                LowLevelType::Signed,
            )),
        ],
        Hlvalue::Variable(variable_with_lltype("v1", LowLevelType::Void)),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_iter)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["lst".to_string()],
        func,
    ))
}

/// Synthesise `ll_listnext` (`lltypesystem/rlist.py`):
///
/// ```python
/// def ll_listnext(iter):
///     l = iter.list
///     index = iter.index
///     if index >= l.ll_length():
///         raise StopIteration
///     iter.index = index + 1
///     return l.ll_getitem_fast(index)
/// ```
///
/// Two-block CFG mirroring [`lowlevel_range_check_helper_graph`]'s
/// raise-to-`exceptblock` shape:
/// - **startblock**: `l = getfield(iter, "list")`,
///   `index = getfield(iter, "index")`, `len = ll_length(l)`
///   (`getarraysize(l)` for the fixed array receiver, `getfield(l,
///   "length")` for the resized header struct), `cond = int_lt(index,
///   len)`. `exitswitch = cond`: the `false` (out-of-bounds) exit links to
///   `graph.exceptblock` with `exception_args("StopIteration")`, the `true`
///   exit carries `(iter, l, index)` to the continue block.
/// - **continue**: `iter.index = int_add(index, 1)`, then
///   `res = ll_getitem_fast(l, index)` (`getarrayitem(l, index)` for the
///   fixed array; `getfield(l, "items")` then `getarrayitem(items, index)`
///   for the resized struct), return `res`.
pub(crate) fn build_ll_listnext_helper_graph(
    name: &str,
    iter_lltype: LowLevelType,
    list_lltype: LowLevelType,
    item_lltype: LowLevelType,
    list_is_fixed: bool,
    foldable: bool,
    length_field: &str,
    items_field: &str,
) -> Result<PyGraph, TyperError> {
    // The resized list keeps its element array in `items_field`; the
    // fixed list IS the bare `Ptr(GcArray)`.
    let items_lltype = if list_is_fixed {
        None
    } else {
        let extracted = match &list_lltype {
            LowLevelType::Ptr(p) => match &p.TO {
                PtrTarget::Struct(s) => s._flds.get(items_field).cloned(),
                _ => None,
            },
            _ => None,
        };
        Some(extracted.ok_or_else(|| {
            TyperError::message(format!(
                "build_ll_listnext_helper_graph: resized list lltype missing {items_field} field"
            ))
        })?)
    };

    let iter_arg = variable_with_lltype("iter", iter_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(iter_arg.clone())]);
    let return_var = variable_with_lltype("result", item_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // upstream `raise StopIteration` — the [etype, evalue] pair the
    // exceptblock link carries.
    let exc_args = exception_args("StopIteration")?;

    // startblock: l = iter.list; index = iter.index; len = ll_length(l);
    //             cond = index < len.
    let v_l = variable_with_lltype("l", list_lltype.clone());
    let v_index = variable_with_lltype("index", LowLevelType::Signed);
    let v_len = variable_with_lltype("len", LowLevelType::Signed);
    let v_cond = variable_with_lltype("cond", LowLevelType::Bool);
    {
        let mut b = startblock.borrow_mut();
        b.operations.push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(iter_arg.clone()),
                void_field_const("list"),
            ],
            Hlvalue::Variable(v_l.clone()),
        ));
        b.operations.push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(iter_arg.clone()),
                void_field_const("index"),
            ],
            Hlvalue::Variable(v_index.clone()),
        ));
        if list_is_fixed {
            b.operations.push(SpaceOperation::new(
                "getarraysize",
                vec![Hlvalue::Variable(v_l.clone())],
                Hlvalue::Variable(v_len.clone()),
            ));
        } else {
            b.operations.push(SpaceOperation::new(
                "getfield",
                vec![
                    Hlvalue::Variable(v_l.clone()),
                    void_field_const(length_field),
                ],
                Hlvalue::Variable(v_len.clone()),
            ));
        }
        b.operations.push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(v_index.clone()),
                Hlvalue::Variable(v_len.clone()),
            ],
            Hlvalue::Variable(v_cond.clone()),
        ));
        b.exitswitch = Some(Hlvalue::Variable(v_cond.clone()));
    }

    // continue block receives (iter, l, index).
    let c_iter = variable_with_lltype("iter", iter_lltype);
    let c_l = variable_with_lltype("l", list_lltype);
    let c_index = variable_with_lltype("index", LowLevelType::Signed);
    let cont = Block::shared(vec![
        Hlvalue::Variable(c_iter.clone()),
        Hlvalue::Variable(c_l.clone()),
        Hlvalue::Variable(c_index.clone()),
    ]);

    startblock.closeblock(vec![
        // index < len -> continue (carry iter, l, index).
        Link::new(
            vec![
                Hlvalue::Variable(iter_arg),
                Hlvalue::Variable(v_l),
                Hlvalue::Variable(v_index),
            ],
            Some(cont.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        // index >= len -> raise StopIteration.
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    // continue: iter.index = index + 1; res = ll_getitem_fast(l, index).
    let v_newindex = variable_with_lltype("newindex", LowLevelType::Signed);
    let v_res = variable_with_lltype("res", item_lltype.clone());
    {
        let mut b = cont.borrow_mut();
        b.operations.push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Variable(c_index.clone()),
                constant_with_lltype(ConstValue::Int(1), LowLevelType::Signed),
            ],
            Hlvalue::Variable(v_newindex.clone()),
        ));
        b.operations.push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(c_iter),
                void_field_const("index"),
                Hlvalue::Variable(v_newindex),
            ],
            Hlvalue::Variable(variable_with_lltype("v", LowLevelType::Void)),
        ));
        if let Some(items_lltype) = items_lltype {
            let v_items = variable_with_lltype("items", items_lltype);
            b.operations.push(SpaceOperation::new(
                "getfield",
                vec![
                    Hlvalue::Variable(c_l.clone()),
                    void_field_const(items_field),
                ],
                Hlvalue::Variable(v_items.clone()),
            ));
            b.operations.push(SpaceOperation::new(
                "getarrayitem",
                vec![Hlvalue::Variable(v_items), Hlvalue::Variable(c_index)],
                Hlvalue::Variable(v_res.clone()),
            ));
        } else {
            // `ll_listnext_foldable` (rlist.py) reads the element via
            // `ll_getitem_foldable_nonneg` = the PURE `getarrayitem_pure`; the
            // plain `ll_listnext` uses the non-pure `getarrayitem`. Only the
            // fixed list folds (a resized list is always mutated).
            let read_op = if foldable {
                "getarrayitem_pure"
            } else {
                "getarrayitem"
            };
            b.operations.push(SpaceOperation::new(
                read_op,
                vec![Hlvalue::Variable(c_l), Hlvalue::Variable(c_index)],
                Hlvalue::Variable(v_res.clone()),
            ));
        }
    }
    cont.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_res)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["iter".to_string()],
        func,
    ))
}

/// Synthesise `ll_revlistiter` (`lltypesystem/rlist.py`):
///
/// ```python
/// def ll_revlistiter(ITERPTR, lst):
///     iter = malloc(ITERPTR.TO)
///     iter.list = lst
///     iter.index = lst.ll_length() - 1
///     return iter
/// ```
pub(crate) fn build_ll_revlistiter_helper_graph(
    name: &str,
    list_lltype: LowLevelType,
    listiter_lltype: LowLevelType,
    list_is_fixed: bool,
    length_field: &str,
) -> Result<PyGraph, TyperError> {
    let lst_arg = variable_with_lltype("lst", list_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(lst_arg.clone())]);
    let return_var = variable_with_lltype("result", listiter_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let LowLevelType::Ptr(ptr) = &listiter_lltype else {
        return Err(TyperError::message(
            "build_ll_revlistiter_helper_graph: revlistiter lltype is not Ptr",
        ));
    };
    let inner_struct = match &ptr.TO {
        PtrTarget::Struct(body) => body.clone(),
        other => {
            return Err(TyperError::message(format!(
                "build_ll_revlistiter_helper_graph: Ptr target must be Struct, got {other:?}"
            )));
        }
    };
    let c1 = Constant::with_concretetype(
        ConstValue::LowLevelType(Box::new(LowLevelType::Struct(Box::new(inner_struct)))),
        LowLevelType::Void,
    );
    let cflags = Constant::with_concretetype(ConstValue::byte_str("flavor=gc"), LowLevelType::Void);
    let v_iter = variable_with_lltype("iter", listiter_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc",
        vec![Hlvalue::Constant(c1), Hlvalue::Constant(cflags)],
        Hlvalue::Variable(v_iter.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(v_iter.clone()),
            void_field_const("list"),
            Hlvalue::Variable(lst_arg.clone()),
        ],
        Hlvalue::Variable(variable_with_lltype("v0", LowLevelType::Void)),
    ));
    let v_len = variable_with_lltype("len", LowLevelType::Signed);
    if list_is_fixed {
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(lst_arg)],
            Hlvalue::Variable(v_len.clone()),
        ));
    } else {
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(lst_arg), void_field_const(length_field)],
            Hlvalue::Variable(v_len.clone()),
        ));
    }
    let v_index = variable_with_lltype("index", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![
            Hlvalue::Variable(v_len),
            constant_with_lltype(ConstValue::Int(1), LowLevelType::Signed),
        ],
        Hlvalue::Variable(v_index.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(v_iter.clone()),
            void_field_const("index"),
            Hlvalue::Variable(v_index),
        ],
        Hlvalue::Variable(variable_with_lltype("v1", LowLevelType::Void)),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_iter)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["lst".to_string()],
        func,
    ))
}

/// Synthesise `ll_revlistnext` (`lltypesystem/rlist.py`):
///
/// ```python
/// def ll_revlistnext(iter):
///     l = iter.list
///     index = iter.index
///     if index < 0:
///         raise StopIteration
///     iter.index -= 1
///     return l.ll_getitem_fast(index)
/// ```
pub(crate) fn build_ll_revlistnext_helper_graph(
    name: &str,
    iter_lltype: LowLevelType,
    list_lltype: LowLevelType,
    item_lltype: LowLevelType,
    list_is_fixed: bool,
    items_field: &str,
) -> Result<PyGraph, TyperError> {
    let items_lltype = if list_is_fixed {
        None
    } else {
        let extracted = match &list_lltype {
            LowLevelType::Ptr(p) => match &p.TO {
                PtrTarget::Struct(s) => s._flds.get(items_field).cloned(),
                _ => None,
            },
            _ => None,
        };
        Some(extracted.ok_or_else(|| {
            TyperError::message(format!(
                "build_ll_revlistnext_helper_graph: resized list lltype missing {items_field} field"
            ))
        })?)
    };

    let iter_arg = variable_with_lltype("iter", iter_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(iter_arg.clone())]);
    let return_var = variable_with_lltype("result", item_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let exc_args = exception_args("StopIteration")?;

    let v_l = variable_with_lltype("l", list_lltype.clone());
    let v_index = variable_with_lltype("index", LowLevelType::Signed);
    let v_cond = variable_with_lltype("cond", LowLevelType::Bool);
    {
        let mut b = startblock.borrow_mut();
        b.operations.push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(iter_arg.clone()),
                void_field_const("list"),
            ],
            Hlvalue::Variable(v_l.clone()),
        ));
        b.operations.push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(iter_arg.clone()),
                void_field_const("index"),
            ],
            Hlvalue::Variable(v_index.clone()),
        ));
        b.operations.push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(v_index.clone()),
                constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
            ],
            Hlvalue::Variable(v_cond.clone()),
        ));
        b.exitswitch = Some(Hlvalue::Variable(v_cond.clone()));
    }

    let c_iter = variable_with_lltype("iter", iter_lltype);
    let c_l = variable_with_lltype("l", list_lltype);
    let c_index = variable_with_lltype("index", LowLevelType::Signed);
    let cont = Block::shared(vec![
        Hlvalue::Variable(c_iter.clone()),
        Hlvalue::Variable(c_l.clone()),
        Hlvalue::Variable(c_index.clone()),
    ]);

    startblock.closeblock(vec![
        // index < 0 -> raise StopIteration.
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        // index >= 0 -> continue (carry iter, l, index).
        Link::new(
            vec![
                Hlvalue::Variable(iter_arg),
                Hlvalue::Variable(v_l),
                Hlvalue::Variable(v_index),
            ],
            Some(cont.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    let v_newindex = variable_with_lltype("newindex", LowLevelType::Signed);
    let v_res = variable_with_lltype("res", item_lltype.clone());
    {
        let mut b = cont.borrow_mut();
        b.operations.push(SpaceOperation::new(
            "int_sub",
            vec![
                Hlvalue::Variable(c_index.clone()),
                constant_with_lltype(ConstValue::Int(1), LowLevelType::Signed),
            ],
            Hlvalue::Variable(v_newindex.clone()),
        ));
        b.operations.push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(c_iter),
                void_field_const("index"),
                Hlvalue::Variable(v_newindex),
            ],
            Hlvalue::Variable(variable_with_lltype("v", LowLevelType::Void)),
        ));
        if let Some(items_lltype) = items_lltype {
            let v_items = variable_with_lltype("items", items_lltype);
            b.operations.push(SpaceOperation::new(
                "getfield",
                vec![
                    Hlvalue::Variable(c_l.clone()),
                    void_field_const(items_field),
                ],
                Hlvalue::Variable(v_items.clone()),
            ));
            b.operations.push(SpaceOperation::new(
                "getarrayitem",
                vec![Hlvalue::Variable(v_items), Hlvalue::Variable(c_index)],
                Hlvalue::Variable(v_res.clone()),
            ));
        } else {
            b.operations.push(SpaceOperation::new(
                "getarrayitem",
                vec![Hlvalue::Variable(c_l), Hlvalue::Variable(c_index)],
                Hlvalue::Variable(v_res.clone()),
            ));
        }
    }
    cont.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_res)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["iter".to_string()],
        func,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::listdef::ListDef;
    use crate::annotator::model::{SomeInteger, SomeList, SomeValue};
    use crate::flowspace::model::Variable;
    use crate::translator::rtyper::pairtype::ReprClassId;
    use crate::translator::rtyper::rint::{IntegerRepr, signed_repr, unsigned_repr};
    use crate::translator::rtyper::rmodel::rtyper_makerepr;
    use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

    fn fresh_rtyper() -> Rc<RPythonTyper> {
        let ann = RPythonAnnotator::new(None, None, None, false);
        Rc::new(RPythonTyper::new(&ann))
    }

    /// `initialize_exceptiondata` sets the typer's self-weak, required by
    /// `rtyper_makerepr`'s `SomeList` arm (`self_rc()`).
    fn fresh_rtyper_live() -> Rc<RPythonTyper> {
        let rtyper = fresh_rtyper();
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        rtyper
    }

    #[test]
    fn list_repr_lltype_is_ptr_gcstruct_length_items() {
        // `LIST = GcStruct("list", ("length", Signed), ("items",
        // Ptr(GcArray(ITEM))))`, lowleveltype `Ptr(LIST)`.
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr = ListRepr::new(&rtyper, r_int).expect("ListRepr::new");
        assert_eq!(repr.class_name(), "ListRepr");

        let LowLevelType::Ptr(ptr) = repr.lowleveltype() else {
            panic!("ListRepr lltype must be a Ptr");
        };
        let PtrTarget::Struct(body) = &ptr.TO else {
            panic!("ListRepr Ptr target must be a Struct");
        };
        assert_eq!(body._name, "list");
        assert_eq!(body._flds.get("length"), Some(&LowLevelType::Signed));
        let items = body._flds.get("items").expect("items field");
        let LowLevelType::Ptr(items_ptr) = items else {
            panic!("items field must be a Ptr");
        };
        assert!(
            matches!(items_ptr.TO, PtrTarget::Array(_)),
            "items must point to a GcArray"
        );
    }

    #[test]
    fn ll_newlist_hint_uses_hint_for_storage_but_zero_for_length() {
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr = ListRepr::new(&rtyper, r_int).expect("ListRepr::new");
        let graph = build_ll_newlist_hint_helper_graph(
            "ll_newlist_hint",
            repr.lltype.clone(),
            LowLevelType::Signed,
        )
        .expect("build ll_newlist_hint");
        let block = graph.graph.borrow().startblock.clone();
        let block = block.borrow();
        assert!(
            block
                .operations
                .iter()
                .any(|op| op.opname == "debug_assert"),
            "ll_newlist_hint must retain upstream's nonnegative assertion"
        );
        let malloc_items = block
            .operations
            .iter()
            .find(|op| op.opname == "malloc_varsize")
            .expect("hint helper allocates the backing array");
        assert!(
            matches!(&malloc_items.args[2], Hlvalue::Variable(v) if v.name().starts_with("length")),
            "backing allocation must use the size hint: {:?}",
            malloc_items.args
        );
        let set_length = block
            .operations
            .iter()
            .find(|op| {
                op.opname == "setfield"
                    && matches!(&op.args[1], Hlvalue::Constant(c) if matches!(&c.value, ConstValue::ByteStr(name) if name == b"length"))
            })
            .expect("hint helper initializes the logical length");
        assert!(
            matches!(&set_length.args[2], Hlvalue::Constant(c) if c.value == ConstValue::Int(0)),
            "logical length must start at zero: {:?}",
            set_length.args
        );
    }

    #[test]
    fn translate_operation_newlist_hint_accepts_rust_usize_and_calls_helper() {
        use crate::flowspace::model::SpaceOperation;
        use std::cell::RefCell as StdRef;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let r_item: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_list: Arc<dyn Repr> =
            Arc::new(ListRepr::new(&rtyper, r_item).expect("ListRepr::new"));
        let r_hint: Arc<dyn Repr> = unsigned_repr();
        let hint = Variable::new();
        hint.set_concretetype(Some(LowLevelType::Unsigned));
        let hint_h = Hlvalue::Variable(hint);
        let result = Variable::new();
        let spaceop = SpaceOperation::new(
            "newlist_hint",
            vec![hint_h.clone()],
            Hlvalue::Variable(result),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(hint_h);
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(true, true)));
        hop.args_r.borrow_mut().push(Some(r_hint));
        *hop.r_result.borrow_mut() = Some(r_list);

        rtyper
            .translate_operation(&hop)
            .expect("newlist_hint dispatch")
            .expect("newlist_hint result");
        let ops = hop.llops.borrow();
        assert!(
            ops.ops.iter().any(|op| op.opname == "cast_uint_to_int"),
            "Rust usize hint must enter upstream's Signed specialization"
        );
        assert!(
            ops.ops.iter().any(|op| op.opname == "direct_call"),
            "newlist_hint must call ll_newlist_hint"
        );
        assert!(ops._called_exception_is_here_or_cannot_occur);
    }

    /// `translate_operation("newlist")` routes to [`rtype_newlist`], which
    /// lowers an N-element list display to `ll_newlist(N)` followed by one
    /// `ll_setitem_fast` per element (RPython `rtype_newlist` / `newlist`,
    /// `rlist.py`). The dispatch is reached only here in two-phase rtyping;
    /// the `vec!` front-end emitter that produces `OpKind::NewList` is the
    /// other producer.
    #[test]
    fn translate_operation_newlist_emits_ll_newlist_and_setitems() {
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::SpaceOperation;
        use std::cell::RefCell as StdRef;

        // Minting the `ll_newlist` / `ll_setitem_fast` helpers derefs the
        // typer's annotator weak ref, so `ann` must outlive the call.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_list: Arc<dyn Repr> =
            Arc::new(ListRepr::new(&rtyper, r_int.clone()).expect("ListRepr::new"));

        let v_a = Variable::new();
        v_a.set_concretetype(Some(LowLevelType::Signed));
        let v_b = Variable::new();
        v_b.set_concretetype(Some(LowLevelType::Signed));
        let v_a_h = Hlvalue::Variable(v_a);
        let v_b_h = Hlvalue::Variable(v_b);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            "newlist".to_string(),
            vec![v_a_h.clone(), v_b_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        // Per-element args, each typed at the list's item repr (Signed).
        hop.args_v.borrow_mut().push(v_a_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        hop.args_v.borrow_mut().push(v_b_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        *hop.r_result.borrow_mut() = Some(r_list.clone());

        let out = rtyper
            .translate_operation(&hop)
            .expect("translate_operation newlist must dispatch to rtype_newlist")
            .expect("newlist returns the fresh list Variable");
        let Hlvalue::Variable(_) = out else {
            panic!("newlist must return a Variable (the ll_newlist result)");
        };
        let ops = hop.llops.borrow();
        let direct_calls = ops
            .ops
            .iter()
            .filter(|op| op.opname == "direct_call")
            .count();
        // `ll_newlist(len)` + one `ll_setitem_fast` per element (2) = 3.
        assert_eq!(
            direct_calls,
            3,
            "expected ll_newlist + 2×ll_setitem_fast direct_calls, got {:?}",
            ops.ops.iter().map(|op| &op.opname).collect::<Vec<_>>()
        );
    }

    /// The `vec![e0, …, eN]` never-resized display annotates as a
    /// `FixedSizeListRepr` (bare `Ptr(GcArray)`). `rtype_newlist` must take
    /// the Fixed arm — `ll_fixed_newlist(N)` + N `ll_fixed_setitem_fast` — and
    /// mint helper graphs under DISTINCT names from the resized variant, so
    /// the `(name, args, result)` helper cache never serves a resized-shaped
    /// graph for the fixed request.
    #[test]
    fn translate_operation_newlist_fixed_repr_emits_ll_fixed_helpers() {
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::SpaceOperation;
        use std::cell::RefCell as StdRef;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_list: Arc<dyn Repr> = Arc::new(
            FixedSizeListRepr::new(&rtyper, r_int.clone()).expect("FixedSizeListRepr::new"),
        );

        let v_a = Variable::new();
        v_a.set_concretetype(Some(LowLevelType::Signed));
        let v_b = Variable::new();
        v_b.set_concretetype(Some(LowLevelType::Signed));
        let v_a_h = Hlvalue::Variable(v_a);
        let v_b_h = Hlvalue::Variable(v_b);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            "newlist".to_string(),
            vec![v_a_h.clone(), v_b_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_a_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        hop.args_v.borrow_mut().push(v_b_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        *hop.r_result.borrow_mut() = Some(r_list.clone());

        rtyper
            .translate_operation(&hop)
            .expect("translate_operation newlist must dispatch to rtype_newlist")
            .expect("newlist returns the fresh list Variable");
        let ops = hop.llops.borrow();
        let direct_calls = ops
            .ops
            .iter()
            .filter(|op| op.opname == "direct_call")
            .count();
        // `ll_fixed_newlist(len)` + one `ll_fixed_setitem_fast` per element.
        assert_eq!(
            direct_calls,
            3,
            "expected ll_fixed_newlist + 2×ll_fixed_setitem_fast direct_calls, got {:?}",
            ops.ops.iter().map(|op| &op.opname).collect::<Vec<_>>()
        );

        // The Fixed arm must have minted its own `ll_fixed_*` helper graphs
        // (distinct cache keys), not reused the resized `ll_newlist` graph.
        let translator = ann.translator.clone();
        let names: Vec<String> = translator
            .graphs
            .borrow()
            .iter()
            .map(|g| g.borrow().name.clone())
            .collect();
        assert!(
            names.iter().any(|n| n == "ll_fixed_newlist"),
            "Fixed arm must mint an ll_fixed_newlist helper graph, got {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "ll_fixed_setitem_fast"),
            "Fixed arm must mint an ll_fixed_setitem_fast helper graph, got {names:?}"
        );
    }

    /// `translate_operation("alloc_and_set")` routes to
    /// [`rtype_alloc_and_set`] (rtyper.py), which lowers a
    /// `[item] * count` display to a single `gendirectcall(ll_alloc_and_set,
    /// count, item)` (rlist.py). Unlike `newlist`, the count is a
    /// runtime `Signed` operand (not a static element enumeration), so the
    /// dispatch emits exactly one `direct_call` and mints the
    /// `ll_alloc_and_set` dispatch helper graph.
    #[test]
    fn translate_operation_alloc_and_set_dispatches_to_ll_alloc_and_set() {
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::SpaceOperation;
        use std::cell::RefCell as StdRef;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_list: Arc<dyn Repr> =
            Arc::new(ListRepr::new(&rtyper, r_int.clone()).expect("ListRepr::new"));

        // Upstream `v_count, v_item = hop.inputargs(Signed, r_list.item_repr)`
        // — arg 0 is the runtime count, arg 1 is the fill item.
        let v_count = Variable::new();
        v_count.set_concretetype(Some(LowLevelType::Signed));
        let v_item = Variable::new();
        v_item.set_concretetype(Some(LowLevelType::Signed));
        let v_count_h = Hlvalue::Variable(v_count);
        let v_item_h = Hlvalue::Variable(v_item);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            "alloc_and_set".to_string(),
            vec![v_count_h.clone(), v_item_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_count_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r
            .borrow_mut()
            .push(Some(
                Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_"))) as Arc<dyn Repr>,
            ));
        hop.args_v.borrow_mut().push(v_item_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        *hop.r_result.borrow_mut() = Some(r_list.clone());

        let out = rtyper
            .translate_operation(&hop)
            .expect("translate_operation alloc_and_set must dispatch to rtype_alloc_and_set")
            .expect("alloc_and_set returns the fresh list Variable");
        let Hlvalue::Variable(_) = out else {
            panic!("alloc_and_set must return a Variable (the ll_alloc_and_set result)");
        };
        let ops = hop.llops.borrow();
        let direct_calls = ops
            .ops
            .iter()
            .filter(|op| op.opname == "direct_call")
            .count();
        // A single `gendirectcall(ll_alloc_and_set, count, item)`.
        assert_eq!(
            direct_calls,
            1,
            "expected one ll_alloc_and_set direct_call, got {:?}",
            ops.ops.iter().map(|op| &op.opname).collect::<Vec<_>>()
        );

        // The resized arm must mint the `ll_alloc_and_set` dispatch helper.
        let translator = ann.translator.clone();
        let names: Vec<String> = translator
            .graphs
            .borrow()
            .iter()
            .map(|g| g.borrow().name.clone())
            .collect();
        assert!(
            names.iter().any(|n| n == "ll_alloc_and_set"),
            "resized arm must mint an ll_alloc_and_set helper graph, got {names:?}"
        );
    }

    /// `l[start:stop]` on a resized `ListRepr` with non-constant nonneg int
    /// bounds decomposes to `"startstop"` (rtyper.py) and lowers to
    /// `gendirectcall(ll_listslice_startstop, l, start, stop)`, which mints the
    /// `ll_listslice_startstop` / `ll_newlist` / `ll_arraycopy` helper graphs
    /// (rlist.py:893-904).
    #[test]
    fn translate_operation_getslice_startstop_emits_ll_listslice_helpers() {
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::SpaceOperation;
        use std::cell::RefCell as StdRef;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_list: Arc<dyn Repr> =
            Arc::new(ListRepr::new(&rtyper, r_int.clone()).expect("ListRepr::new"));
        let list_lltype = r_list.lowleveltype().clone();

        let v_lst = Variable::new();
        v_lst.set_concretetype(Some(list_lltype));
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
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_lst_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_list.clone()));
        // Non-constant nonneg int bounds → the "startstop" arm.
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
        // The slice result is a fresh resized list (listdef.offspring).
        *hop.r_result.borrow_mut() = Some(r_list.clone());

        let out = rtyper
            .translate_operation(&hop)
            .expect("translate_operation getslice must dispatch to rtype_getslice")
            .expect("getslice returns the fresh list Variable");
        let Hlvalue::Variable(_) = out else {
            panic!("getslice must return a Variable (the ll_listslice result)");
        };
        // The lowering is a single `direct_call(ll_listslice_startstop, …)`.
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
        // The per-kind + support helper graphs were minted under their names.
        let translator = ann.translator.clone();
        let names: Vec<String> = translator
            .graphs
            .borrow()
            .iter()
            .map(|g| g.borrow().name.clone())
            .collect();
        for expected in ["ll_listslice_startstop", "ll_newlist", "ll_arraycopy"] {
            assert!(
                names.iter().any(|n| n == expected),
                "getslice must mint an {expected} helper graph, got {names:?}"
            );
        }
    }

    /// A runtime `start` that is NOT proved non-negative must be rejected by
    /// `decompose_slice_args` (rtyper.py) with "slice start must be
    /// proved non-negative", the guard that keeps an unsupported slice form off
    /// the `ll_listslice_*` path. Same hop as the startstop test but with a
    /// `nonneg == false` start annotation.
    #[test]
    fn translate_operation_getslice_non_nonneg_start_is_rejected() {
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::SpaceOperation;
        use std::cell::RefCell as StdRef;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_list: Arc<dyn Repr> =
            Arc::new(ListRepr::new(&rtyper, r_int.clone()).expect("ListRepr::new"));
        let list_lltype = r_list.lowleveltype().clone();

        let v_lst = Variable::new();
        v_lst.set_concretetype(Some(list_lltype));
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
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_lst_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_list.clone()));
        // A runtime start that is NOT proved non-negative.
        hop.args_v.borrow_mut().push(v_start_h);
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(
                /* nonneg */ false, false,
            )));
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        hop.args_v.borrow_mut().push(v_stop_h);
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::new(
                /* nonneg */ true, false,
            )));
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        *hop.r_result.borrow_mut() = Some(r_list.clone());

        let err = rtyper
            .translate_operation(&hop)
            .expect_err("a non-nonneg slice start must fail decompose_slice_args");
        let msg = format!("{err}");
        assert!(
            msg.contains("slice start must be proved non-negative"),
            "expected the non-negative start rejection, got {msg:?}"
        );
    }

    /// `l[start:]` — non-constant nonneg start, constant `None` stop —
    /// decomposes to `"startonly"` (rtyper.py) and lowers to
    /// `gendirectcall(ll_listslice_startonly, l, start)` (rlist.py).
    #[test]
    fn translate_operation_getslice_startonly_emits_startonly_helper() {
        use crate::annotator::model::{SomeInteger, SomeValue, s_none};
        use crate::flowspace::model::{Constant, SpaceOperation};
        use std::cell::RefCell as StdRef;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_list: Arc<dyn Repr> =
            Arc::new(ListRepr::new(&rtyper, r_int.clone()).expect("ListRepr::new"));
        let list_lltype = r_list.lowleveltype().clone();

        let v_lst = Variable::new();
        v_lst.set_concretetype(Some(list_lltype));
        let v_start = Variable::new();
        v_start.set_concretetype(Some(LowLevelType::Signed));
        let v_lst_h = Hlvalue::Variable(v_lst);
        let v_start_h = Hlvalue::Variable(v_start);
        // stop is a constant None (the [start:] shape).
        let c_stop = Hlvalue::Constant(Constant::new(ConstValue::None));
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            "getslice".to_string(),
            vec![v_lst_h.clone(), v_start_h.clone(), c_stop.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_lst_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_list.clone()));
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
        *hop.r_result.borrow_mut() = Some(r_list.clone());

        rtyper
            .translate_operation(&hop)
            .expect("translate_operation getslice must dispatch to rtype_getslice")
            .expect("getslice returns the fresh list Variable");
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
        let translator = ann.translator.clone();
        let names: Vec<String> = translator
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

    /// A `[start:]` slice whose result is never resized takes a
    /// `FixedSizeListRepr` result (`listdef.offspring` keeps `resized=False`),
    /// so `ll_newlist` returns a bare `Ptr(GcArray)` with no `items` field. The
    /// minted `ll_listslice_startonly` helper must therefore select `dst_items`
    /// as the result array ITSELF (`ll_fixed_items(l) = l`, rlist.py) and
    /// emit NO `getfield("items")` — a fixed source AND a fixed result yield a
    /// getfield-free copy body. A regression that hardcodes `getfield(new_lst,
    /// "items")` would target a nonexistent field and reintroduce a getfield.
    #[test]
    fn translate_operation_getslice_startonly_fixed_result_has_no_dst_items_getfield() {
        use crate::annotator::model::{SomeInteger, SomeValue, s_none};
        use crate::flowspace::model::{Constant, SpaceOperation};
        use std::cell::RefCell as StdRef;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        // Both the source and the result are FIXED lists: the receiver is a
        // never-resized `&[T]`-shaped list and the slice result is likewise
        // never resized, so both reach their array without an `items` field.
        let r_list: Arc<dyn Repr> = Arc::new(
            FixedSizeListRepr::new(&rtyper, r_int.clone()).expect("FixedSizeListRepr::new"),
        );
        let list_lltype = r_list.lowleveltype().clone();

        let v_lst = Variable::new();
        v_lst.set_concretetype(Some(list_lltype));
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
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_lst_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_list.clone()));
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
        // The slice result is a fresh FIXED list (never resized).
        *hop.r_result.borrow_mut() = Some(r_list.clone());

        rtyper
            .translate_operation(&hop)
            .expect("translate_operation getslice must dispatch to rtype_getslice")
            .expect("getslice returns the fresh list Variable");

        // Inspect the minted `ll_listslice_startonly` helper body: with a fixed
        // source and a fixed result, neither `src_items` nor `dst_items` reads
        // an `items` field, so the copy body carries zero `getfield` ops.
        let translator = ann.translator.clone();
        let graphs = translator.graphs.borrow();
        let helper = graphs
            .iter()
            .find(|g| g.borrow().name == "ll_listslice_startonly")
            .expect("getslice [start:] must mint an ll_listslice_startonly helper graph");
        // The listslice helper is a single-block graph; its whole copy body
        // lives in the startblock.
        let getfields: usize = helper
            .borrow()
            .startblock
            .borrow()
            .operations
            .iter()
            .filter(|op| op.opname == "getfield")
            .count();
        assert_eq!(
            getfields, 0,
            "a fixed-source, fixed-result getslice must not read an `items` field \
             (ll_newlist returns a bare array); found {getfields} getfield op(s)"
        );
    }

    /// `l[:-1]` decomposes to `"minusone"` (rtyper.py) — start const 0,
    /// stop const -1, no runtime bound args — and lowers to
    /// `gendirectcall(ll_listslice_minusone, l)` (rlist.py).
    #[test]
    fn translate_operation_getslice_minusone_emits_minusone_helper() {
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{Constant, SpaceOperation};
        use std::cell::RefCell as StdRef;

        // A constant `SomeInteger` carrying `value` — `is_constant()` reads
        // `const_box`, `const_()` reads its `ConstValue`.
        fn const_int(value: i64) -> SomeValue {
            let mut si = SomeInteger::new(value >= 0, false);
            si.base.const_box = Some(Constant::new(ConstValue::Int(value)));
            SomeValue::Integer(si)
        }

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_list: Arc<dyn Repr> =
            Arc::new(ListRepr::new(&rtyper, r_int.clone()).expect("ListRepr::new"));
        let list_lltype = r_list.lowleveltype().clone();

        let v_lst = Variable::new();
        v_lst.set_concretetype(Some(list_lltype));
        let v_lst_h = Hlvalue::Variable(v_lst);
        // start = const 0, stop = const -1 (the [:-1] shape).
        let c_start = Hlvalue::Constant(Constant::new(ConstValue::Int(0)));
        let c_stop = Hlvalue::Constant(Constant::new(ConstValue::Int(-1)));
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            "getslice".to_string(),
            vec![v_lst_h.clone(), c_start.clone(), c_stop.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_lst_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_list.clone()));
        // Constant bounds carried on the annotation (is_constant() reads const_box).
        hop.args_v.borrow_mut().push(c_start);
        hop.args_s.borrow_mut().push(const_int(0));
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        hop.args_v.borrow_mut().push(c_stop);
        hop.args_s.borrow_mut().push(const_int(-1));
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        *hop.r_result.borrow_mut() = Some(r_list.clone());

        rtyper
            .translate_operation(&hop)
            .expect("translate_operation getslice must dispatch to rtype_getslice")
            .expect("getslice returns the fresh list Variable");
        let ops = hop.llops.borrow();
        let direct_calls = ops
            .ops
            .iter()
            .filter(|op| op.opname == "direct_call")
            .count();
        assert_eq!(
            direct_calls,
            1,
            "expected a single ll_listslice_minusone direct_call, got {:?}",
            ops.ops.iter().map(|op| &op.opname).collect::<Vec<_>>()
        );
        let translator = ann.translator.clone();
        let names: Vec<String> = translator
            .graphs
            .borrow()
            .iter()
            .map(|g| g.borrow().name.clone())
            .collect();
        assert!(
            names.iter().any(|n| n == "ll_listslice_minusone"),
            "getslice [:-1] must mint an ll_listslice_minusone helper graph, got {names:?}"
        );
    }

    #[test]
    fn makerepr_somelist_routes_by_resized_flag() {
        let cases = [
            ("resized", true, "ListRepr", ReprClassId::ListRepr),
            (
                "nonresized",
                false,
                "FixedSizeListRepr",
                ReprClassId::FixedSizeListRepr,
            ),
        ];
        for (name, resized, class_name, class_id) in cases {
            let rtyper = fresh_rtyper_live();
            let ldef = ListDef::new(
                None,
                SomeValue::Integer(SomeInteger::new(false, false)),
                false,
                resized,
            );
            let sv = SomeValue::List(SomeList::new(ldef));
            let repr = rtyper_makerepr(&sv, &rtyper)
                .unwrap_or_else(|err| panic!("case {name}: rtyper_makerepr: {err:?}"));
            assert_eq!(repr.class_name(), class_name, "case {name}");
            assert_eq!(repr.repr_class_id(), class_id, "case {name}");
        }
    }

    /// rlist.py:247-267 nonneg + checkidx=False branch on an UNMUTATED
    /// `FixedSizeListRepr` — `basegetitem = ll_getitem_foldable_nonneg`
    /// (rlist.py:258), so `getitem` lowers to a `direct_call` of the foldable
    /// `ll_fixed_getitem_fast_foldable` (a single `getarrayitem_pure` on the
    /// `Ptr(GcArray)` receiver), preceded by `hop.exception_cannot_occur()`.
    #[test]
    fn fixed_size_list_getitem_nonneg_unmutated_emits_foldable_helper() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        // A single shared repr instance: `convertvar` short-circuits on
        // `std::ptr::eq` of the `&dyn Repr`, so `args_r[0]` and the `self`
        // routed through `rtype_getitem` must be the same object.
        let list_repr: Arc<FixedSizeListRepr> = Arc::new(
            FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
                .expect("FixedSizeListRepr::new"),
        );
        let list_lltype = list_repr.lowleveltype().clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_lltype));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_list), Hlvalue::Variable(v_idx)],
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
                /* mutated */ false,
                /* resized */ false,
            ))),
            SomeValue::Integer(SomeInteger::new(/* nonneg */ true, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(list_repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);

        let result = list_repr
            .rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("list getitem nonneg unmutated: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "checkidx=False path must call hop.exception_cannot_occur()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_fixed_getitem_fast_foldable"),
            "unmutated list must select the foldable helper, got {dbg}"
        );
    }

    /// rlist.py — a MUTATED `FixedSizeListRepr` (in-place setitem
    /// without resize) keeps `basegetitem = ll_getitem_fast`, so `getitem`
    /// lowers to the NON-foldable `ll_fixed_getitem_fast` (plain
    /// `getarrayitem`), proving the foldable selection is gated on
    /// `listitem.mutated`, not on the Fixed layout.
    #[test]
    fn fixed_size_list_getitem_nonneg_mutated_emits_nonfoldable_helper() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        let list_repr: Arc<FixedSizeListRepr> = Arc::new(
            FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
                .expect("FixedSizeListRepr::new"),
        );
        let list_lltype = list_repr.lowleveltype().clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_lltype));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_list), Hlvalue::Variable(v_idx)],
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
                /* mutated */ true,
                /* resized */ false,
            ))),
            SomeValue::Integer(SomeInteger::new(/* nonneg */ true, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(list_repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);

        let result = list_repr
            .rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("list getitem nonneg mutated: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_fixed_getitem_fast") && !dbg.contains("_foldable"),
            "mutated list must select the non-foldable helper, got {dbg}"
        );
    }

    /// The foldable helper body [`build_ll_fixed_getitem_fast_foldable_helper_graph`]
    /// emits the PURE `getarrayitem_pure` element load (rlist.py:721-724
    /// `oopspec = 'list.getitem_foldable'`), while the non-foldable
    /// [`build_ll_fixed_getitem_fast_helper_graph`] emits a plain
    /// `getarrayitem`.
    #[test]
    fn fixed_getitem_fast_foldable_body_emits_getarrayitem_pure() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let repr = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let ptr = repr.lowleveltype().clone();
        let item = LowLevelType::Signed;

        let foldable = build_ll_fixed_getitem_fast_foldable_helper_graph(
            "ll_fixed_getitem_fast_foldable",
            ptr.clone(),
            item.clone(),
        )
        .expect("build foldable helper");
        let nonfoldable =
            build_ll_fixed_getitem_fast_helper_graph("ll_fixed_getitem_fast", ptr, item)
                .expect("build non-foldable helper");

        // `block_op_sequences` also visits the (op-less) returnblock.
        assert_eq!(
            block_op_sequences(&foldable),
            vec![vec!["getarrayitem_pure".to_string()], vec![]],
            "foldable helper body must be the pure element load"
        );
        assert_eq!(
            block_op_sequences(&nonfoldable),
            vec![vec!["getarrayitem".to_string()], vec![]],
            "non-foldable helper body must be the plain element load"
        );
    }

    /// rlist.py:247-267 negative-index branch (`args_s[1].nonneg == false`,
    /// no caught IndexError → `dum_nocheck` `ll_getitem`) — `getitem` lowers
    /// to a `direct_call` of the neg-fix helper `ll_fixed_getitem` (not the
    /// `_fast` helper), preceded by `hop.exception_cannot_occur()`.
    #[test]
    fn fixed_size_list_getitem_negative_index_emits_direct_call_to_ll_fixed_getitem() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let list_repr: Arc<FixedSizeListRepr> = Arc::new(
            FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
                .expect("FixedSizeListRepr::new"),
        );
        let list_lltype = list_repr.lowleveltype().clone();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_lltype));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_list), Hlvalue::Variable(v_idx)],
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
                /* mutated */ false,
                /* resized */ false,
            ))),
            SomeValue::Integer(SomeInteger::new(/* nonneg */ false, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(list_repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);

        let result = list_repr
            .rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("list getitem neg: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "checkidx=False path must call hop.exception_cannot_occur()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_fixed_getitem") && !dbg.contains("ll_fixed_getitem_fast"),
            "expected the neg helper 'll_fixed_getitem' (not '_fast') in {dbg}"
        );
    }

    /// rlist.py:247-267 checkidx branch (`hop.has_implicit_exception(
    /// "IndexError")`, nonneg index → `ll_getitem_nonneg` with
    /// `dum_checkidx`) — `getitem` lowers to a `direct_call` of the
    /// bound-checking helper `ll_fixed_getitem_nonneg_checked`, preceded by
    /// `hop.exception_is_here()`.
    #[test]
    fn fixed_size_list_getitem_checkidx_emits_direct_call_to_nonneg_checked() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let list_repr: Arc<FixedSizeListRepr> = Arc::new(
            FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
                .expect("FixedSizeListRepr::new"),
        );
        let list_lltype = list_repr.lowleveltype().clone();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_lltype));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        // A caught IndexError exitcase on this getitem.
        let exitblock = std::rc::Rc::new(std::cell::RefCell::new(Block::new(vec![])));
        let cls_index = crate::flowspace::model::HOST_ENV
            .lookup_exception_class("IndexError")
            .expect("IndexError class");
        let link_index = std::rc::Rc::new(std::cell::RefCell::new(Link::new(
            vec![],
            Some(exitblock),
            Some(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                cls_index,
            )))),
        )));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_list), Hlvalue::Variable(v_idx)],
                Hlvalue::Variable(v_result),
            ),
            vec![link_index],
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::List(SomeList::new(ListDef::new(
                None,
                SomeValue::Integer(SomeInteger::new(false, false)),
                /* mutated */ false,
                /* resized */ false,
            ))),
            SomeValue::Integer(SomeInteger::new(/* nonneg */ true, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(list_repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);
        let result = list_repr
            .rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("list getitem checkidx: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "checkidx=True path must call hop.exception_is_here()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_fixed_getitem_nonneg_checked"),
            "expected 'll_fixed_getitem_nonneg_checked' in {dbg}"
        );
    }

    /// rlist.py nonneg + dum_nocheck branch — `setitem` on a
    /// `FixedSizeListRepr` lowers to a `direct_call` of
    /// `ll_fixed_setitem_fast` (a single `setarrayitem` on the
    /// `Ptr(GcArray)` receiver), preceded by `hop.exception_is_here()`.
    #[test]
    fn fixed_size_list_setitem_nonneg_emits_direct_call_to_ll_fixed_setitem_fast() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        let list_repr: Arc<FixedSizeListRepr> = Arc::new(
            FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
                .expect("FixedSizeListRepr::new"),
        );
        let list_lltype = list_repr.lowleveltype().clone();
        // The third inputarg converts to `item_repr`; `convertvar`
        // short-circuits on `std::ptr::eq`, so `args_r[2]` must be the
        // exact stored `item_repr` instance.
        let item_repr = list_repr.item_repr.clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_lltype));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_item = Variable::new();
        v_item.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "setitem".to_string(),
                vec![
                    Hlvalue::Variable(v_list),
                    Hlvalue::Variable(v_idx),
                    Hlvalue::Variable(v_item),
                ],
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
                /* mutated */ true,
                /* resized */ false,
            ))),
            SomeValue::Integer(SomeInteger::new(/* nonneg */ true, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(list_repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
            Some(item_repr),
        ]);

        let result = list_repr
            .rtype_setitem(&hop)
            .unwrap_or_else(|err| panic!("list setitem nonneg: {err:?}"));
        assert!(result.is_some());
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "rtype_setitem must call hop.exception_is_here()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_fixed_setitem_fast"),
            "expected 'll_fixed_setitem_fast' in {dbg}"
        );
    }

    /// rlist.py:272-284 negative-index branch (`args_s[1].nonneg == false`,
    /// no caught IndexError → `dum_nocheck` `ll_setitem`) — `setitem` lowers
    /// to a `direct_call` of the neg-fix helper `ll_fixed_setitem` (not the
    /// `_fast` helper), preceded by `hop.exception_is_here()`.
    #[test]
    fn fixed_size_list_setitem_negative_index_emits_direct_call_to_ll_fixed_setitem() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let list_repr: Arc<FixedSizeListRepr> = Arc::new(
            FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
                .expect("FixedSizeListRepr::new"),
        );
        let list_lltype = list_repr.lowleveltype().clone();
        let item_repr = list_repr.item_repr.clone();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_lltype));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_item = Variable::new();
        v_item.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "setitem".to_string(),
                vec![
                    Hlvalue::Variable(v_list),
                    Hlvalue::Variable(v_idx),
                    Hlvalue::Variable(v_item),
                ],
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
                /* mutated */ true,
                /* resized */ false,
            ))),
            SomeValue::Integer(SomeInteger::new(/* nonneg */ false, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(list_repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
            Some(item_repr),
        ]);

        let result = list_repr
            .rtype_setitem(&hop)
            .unwrap_or_else(|err| panic!("list setitem neg: {err:?}"));
        assert!(result.is_some());
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "rtype_setitem must call hop.exception_is_here()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_fixed_setitem") && !dbg.contains("ll_fixed_setitem_fast"),
            "expected the neg helper 'll_fixed_setitem' (not '_fast') in {dbg}"
        );
    }

    /// rlist.py:247-267 nonneg + checkidx=False branch — `getitem` on a
    /// resized `ListRepr` lowers to a `direct_call` of `ll_getitem_fast`
    /// (the resized helper, distinct from `FixedSizeListRepr`'s
    /// `ll_fixed_getitem_fast`), preceded by `hop.exception_cannot_occur()`.
    #[test]
    fn resized_list_getitem_nonneg_emits_direct_call_to_ll_getitem_fast() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        let list_repr: Arc<ListRepr> = Arc::new(
            ListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>).expect("ListRepr::new"),
        );
        let list_lltype = list_repr.lowleveltype().clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_lltype));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_list), Hlvalue::Variable(v_idx)],
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
                /* mutated */ false,
                /* resized */ true,
            ))),
            SomeValue::Integer(SomeInteger::new(/* nonneg */ true, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(list_repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);

        let result = list_repr
            .rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("resized list getitem nonneg: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "checkidx=False path must call hop.exception_cannot_occur()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_getitem_fast"),
            "expected 'll_getitem_fast' in {dbg}"
        );
    }

    /// rlist.py nonneg + dum_nocheck branch — `setitem` on a
    /// resized `ListRepr` lowers to a `direct_call` of `ll_setitem_fast`
    /// (the resized helper, distinct from `FixedSizeListRepr`'s
    /// `ll_fixed_setitem_fast`), preceded by `hop.exception_is_here()`.
    #[test]
    fn resized_list_setitem_nonneg_emits_direct_call_to_ll_setitem_fast() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        let list_repr: Arc<ListRepr> = Arc::new(
            ListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>).expect("ListRepr::new"),
        );
        let list_lltype = list_repr.lowleveltype().clone();
        let item_repr = list_repr.item_repr.clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_lltype));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_item = Variable::new();
        v_item.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "setitem".to_string(),
                vec![
                    Hlvalue::Variable(v_list),
                    Hlvalue::Variable(v_idx),
                    Hlvalue::Variable(v_item),
                ],
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
                /* mutated */ true,
                /* resized */ true,
            ))),
            SomeValue::Integer(SomeInteger::new(/* nonneg */ true, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(list_repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
            Some(item_repr),
        ]);

        let result = list_repr
            .rtype_setitem(&hop)
            .unwrap_or_else(|err| panic!("resized list setitem nonneg: {err:?}"));
        assert!(result.is_some());
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "rtype_setitem must call hop.exception_is_here()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_setitem_fast"),
            "expected 'll_setitem_fast' in {dbg}"
        );
    }

    /// The resized `ll_getitem_fast` helper body reads the `items` array
    /// out of the `length`/`items` header struct before indexing:
    /// `getfield(l, "items")` → `getarrayitem(items, index)` (vs
    /// `FixedSizeListRepr`'s single `getarrayitem` on the bare array).
    #[test]
    fn build_ll_getitem_fast_helper_reads_items_then_getarrayitem() {
        let rtyper = fresh_rtyper();
        let repr = ListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>).expect("ListRepr::new");
        let pygraph = build_ll_getitem_fast_helper_graph(
            "ll_getitem_fast",
            repr.lowleveltype().clone(),
            LowLevelType::Signed,
        )
        .expect("build_ll_getitem_fast_helper_graph");
        let graph = pygraph.graph.borrow();
        let ops: Vec<_> = graph
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(ops, vec!["getfield", "getarrayitem"]);
    }

    /// The resized `ll_setitem_fast` helper body reads the `items` array
    /// out of the `length`/`items` header struct before storing:
    /// `getfield(l, "items")` → `setarrayitem(items, index, item)`.
    #[test]
    fn build_ll_setitem_fast_helper_reads_items_then_setarrayitem() {
        let rtyper = fresh_rtyper();
        let repr = ListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>).expect("ListRepr::new");
        let pygraph = build_ll_setitem_fast_helper_graph(
            "ll_setitem_fast",
            repr.lowleveltype().clone(),
            LowLevelType::Signed,
        )
        .expect("build_ll_setitem_fast_helper_graph");
        let graph = pygraph.graph.borrow();
        let ops: Vec<_> = graph
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(ops, vec!["getfield", "setarrayitem"]);
    }

    /// `slice.reverse()` rtypes through `rtype_method("reverse")` to a
    /// `direct_call(ll_reverse, v_lst)` (`rlist.py`); `reverse`
    /// returns `None` (void), and the path calls
    /// `hop.exception_cannot_occur()`.
    #[test]
    fn fixed_size_list_reverse_emits_direct_call_to_ll_reverse() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let list_repr: Arc<FixedSizeListRepr> = Arc::new(
            FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
                .expect("FixedSizeListRepr::new"),
        );
        let list_lltype = list_repr.lowleveltype().clone();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_lltype));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "simple_call".to_string(),
                vec![Hlvalue::Variable(v_list)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .extend([SomeValue::List(SomeList::new(ListDef::new(
                None,
                SomeValue::Integer(SomeInteger::new(false, false)),
                /* mutated */ true,
                /* resized */ false,
            )))]);
        hop.args_r
            .borrow_mut()
            .extend([Some(list_repr.clone() as Arc<dyn Repr>)]);

        let result = list_repr
            .rtype_method("reverse", &hop)
            .unwrap_or_else(|err| panic!("list reverse: {err:?}"));
        assert!(result.is_some());
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "rtype_method_reverse must call hop.exception_cannot_occur()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(dbg.contains("ll_reverse"), "expected 'll_reverse' in {dbg}");
    }

    /// An unknown method name surfaces a `TyperError` so the subject stays
    /// on the legacy walker rather than miscompiling.
    #[test]
    fn fixed_size_list_unknown_method_is_deferred() {
        let rtyper = fresh_rtyper_live();
        let list_repr: Arc<FixedSizeListRepr> = Arc::new(
            FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
                .expect("FixedSizeListRepr::new"),
        );
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_repr.lowleveltype().clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "simple_call".to_string(),
                vec![Hlvalue::Variable(v_list)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops,
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_r
            .borrow_mut()
            .extend([Some(list_repr.clone() as Arc<dyn Repr>)]);
        assert!(list_repr.rtype_method("sort", &hop).is_err());
    }

    #[test]
    fn runtime_helper_surface_is_explicitly_deferred() {
        let _iter = AbstractListIteratorRepr;

        let err = ll_append().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_append"));

        let err = ll_listsetslice().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_listsetslice"));
    }

    /// The `ll_reverse` helper is a four-block swap loop: `startblock`
    /// (`getarraysize` + `int_sub`) → `block_loop_cond` (`int_lt`) →
    /// `block_loop_body` (two `getarrayitem` reads BEFORE two `setarrayitem`
    /// writes, then `int_add` / `int_sub`) → back to the cond block.
    #[test]
    fn build_ll_reverse_helper_has_swap_loop_blocks() {
        let rtyper = fresh_rtyper();
        let repr = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let pygraph = build_ll_reverse_helper_graph(
            "ll_reverse",
            repr.lowleveltype().clone(),
            LowLevelType::Signed,
        )
        .expect("build_ll_reverse_helper_graph");
        let graph = pygraph.graph.borrow();
        let block_op_seqs: Vec<Vec<String>> = graph
            .iterblocks()
            .iter()
            .map(|b| {
                b.borrow()
                    .operations
                    .iter()
                    .map(|op| op.opname.clone())
                    .collect()
            })
            .collect();
        // The startblock signature distinguishes reverse from get/setitem.
        assert!(
            block_op_seqs.contains(&vec!["getarraysize".to_string(), "int_sub".to_string()]),
            "startblock must be getarraysize + int_sub, got {block_op_seqs:?}"
        );
        // The loop-condition block.
        assert!(
            block_op_seqs.contains(&vec!["int_lt".to_string()]),
            "expected an int_lt condition block, got {block_op_seqs:?}"
        );
        // The swap body: both reads precede both writes, then the two steps.
        assert!(
            block_op_seqs.contains(&vec![
                "getarrayitem".to_string(),
                "getarrayitem".to_string(),
                "setarrayitem".to_string(),
                "setarrayitem".to_string(),
                "int_add".to_string(),
                "int_sub".to_string(),
            ]),
            "expected the read-both-before-write swap body, got {block_op_seqs:?}"
        );
    }

    /// The resized [`ListRepr`] inherits `reverse` from the common base
    /// (`rlist.py:138-143`); it rtypes through `rtype_method("reverse")` to a
    /// `direct_call(ll_reverse, v_lst)` just like `FixedSizeListRepr`, and the
    /// path calls `hop.exception_cannot_occur()`.
    #[test]
    fn resized_list_reverse_emits_direct_call_to_ll_reverse() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let list_repr: Arc<ListRepr> = Arc::new(
            ListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>).expect("ListRepr::new"),
        );
        let list_lltype = list_repr.lowleveltype().clone();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_lltype));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "simple_call".to_string(),
                vec![Hlvalue::Variable(v_list)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .extend([SomeValue::List(SomeList::new(ListDef::new(
                None,
                SomeValue::Integer(SomeInteger::new(false, false)),
                /* mutated */ true,
                /* resized */ true,
            )))]);
        hop.args_r
            .borrow_mut()
            .extend([Some(list_repr.clone() as Arc<dyn Repr>)]);

        let result = list_repr
            .rtype_method("reverse", &hop)
            .unwrap_or_else(|err| panic!("resized list reverse: {err:?}"));
        assert!(result.is_some());
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "rtype_method_reverse must call hop.exception_cannot_occur()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(dbg.contains("ll_reverse"), "expected 'll_reverse' in {dbg}");
    }

    /// The resized `ll_reverse` is the same four-block swap loop as the
    /// fixed-size one, but the startblock reads `length` via `getfield`
    /// (vs `getarraysize`) and every element access reads the `items` array
    /// out of the struct header first (`getfield` before each
    /// `getarrayitem` / `setarrayitem`).
    #[test]
    fn build_ll_reverse_resized_helper_has_swap_loop_blocks() {
        let rtyper = fresh_rtyper();
        let repr = ListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>).expect("ListRepr::new");
        let pygraph = build_ll_reverse_resized_helper_graph(
            "ll_reverse",
            repr.lowleveltype().clone(),
            LowLevelType::Signed,
        )
        .expect("build_ll_reverse_resized_helper_graph");
        let graph = pygraph.graph.borrow();
        let block_op_seqs: Vec<Vec<String>> = graph
            .iterblocks()
            .iter()
            .map(|b| {
                b.borrow()
                    .operations
                    .iter()
                    .map(|op| op.opname.clone())
                    .collect()
            })
            .collect();
        // The resized startblock reads `length` from the struct header.
        assert!(
            block_op_seqs.contains(&vec!["getfield".to_string(), "int_sub".to_string()]),
            "startblock must be getfield + int_sub, got {block_op_seqs:?}"
        );
        assert!(
            block_op_seqs.contains(&vec!["int_lt".to_string()]),
            "expected an int_lt condition block, got {block_op_seqs:?}"
        );
        // The swap body reads `items` (getfield) before each array op; both
        // reads precede both writes, then the two index steps.
        assert!(
            block_op_seqs.contains(&vec![
                "getfield".to_string(),
                "getarrayitem".to_string(),
                "getfield".to_string(),
                "getarrayitem".to_string(),
                "getfield".to_string(),
                "setarrayitem".to_string(),
                "getfield".to_string(),
                "setarrayitem".to_string(),
                "int_add".to_string(),
                "int_sub".to_string(),
            ]),
            "expected the items-indirected swap body, got {block_op_seqs:?}"
        );
    }

    /// Count the outgoing links across all reachable blocks of `pygraph`
    /// whose target is `graph.exceptblock` (the IndexError-raising arms).
    fn count_links_to_exceptblock(pygraph: &PyGraph) -> usize {
        let graph = pygraph.graph.borrow();
        graph
            .iterblocks()
            .iter()
            .flat_map(|b| b.borrow().exits.clone())
            .filter(|link| {
                link.borrow()
                    .target
                    .as_ref()
                    .is_some_and(|t| std::rc::Rc::ptr_eq(t, &graph.exceptblock))
            })
            .count()
    }

    fn block_op_sequences(pygraph: &PyGraph) -> Vec<Vec<String>> {
        pygraph
            .graph
            .borrow()
            .iterblocks()
            .iter()
            .map(|b| {
                b.borrow()
                    .operations
                    .iter()
                    .map(|op| op.opname.clone())
                    .collect()
            })
            .collect()
    }

    /// rlist.py checked `ll_getitem` (`dum_checkidx`, negative index
    /// folded in) lowers to the unsigned-window test `r_uint(index) >=
    /// r_uint(length)`: the start reads length (`getarraysize` on the
    /// `FixedSizeListRepr` receiver), casts index/length to unsigned and
    /// `uint_ge`-branches; the fixup block does `uint_add`, a second `uint_ge`
    /// raising IndexError, and `intmask` (`cast_uint_to_int`) feeding the
    /// `direct_call` dispatch.  Only the post-add bound check raises (one link
    /// to exceptblock); the common `0 <= index < length` case falls through.
    #[test]
    fn build_ll_list_getitem_checked_helper_fixed_has_window_checks() {
        // Keep `ann` alive for the duration: building the inner
        // `ll_*_getitem_fast` sub-helper annotates it through the typer's
        // weak annotator reference.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let repr = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let pygraph = build_ll_list_getitem_checked_helper_graph(
            &rtyper,
            ListLayout::Fixed,
            "ll_fixed_getitem_checked",
            /* foldable */ false,
            repr.lowleveltype().clone(),
            LowLevelType::Signed,
        )
        .expect("build_ll_list_getitem_checked_helper_graph");
        let seqs = block_op_sequences(&pygraph);
        assert!(
            seqs.contains(&vec![
                "getarraysize".to_string(),
                "cast_int_to_uint".to_string(),
                "cast_int_to_uint".to_string(),
                "uint_ge".to_string(),
            ]),
            "start must read length (getarraysize), cast to unsigned, then uint_ge, got {seqs:?}"
        );
        assert!(
            seqs.contains(&vec![
                "uint_add".to_string(),
                "uint_ge".to_string(),
                "cast_uint_to_int".to_string(),
            ]),
            "expected a uint_add/uint_ge/cast_uint_to_int fixup block, got {seqs:?}"
        );
        assert!(
            seqs.contains(&vec!["direct_call".to_string()]),
            "expected a direct_call dispatch block, got {seqs:?}"
        );
        assert_eq!(
            count_links_to_exceptblock(&pygraph),
            1,
            "only the post-add bound check raises IndexError"
        );
    }

    /// rlist.py:737-742 checked `ll_setitem` for the resized [`ListRepr`]: the
    /// length read is `getfield(l, "length")` (struct header) not
    /// `getarraysize`, the unsigned-window test then casts/`uint_ge`-branches
    /// and the fixup block (`uint_add`/`uint_ge`/`cast_uint_to_int`) threads the
    /// `item` operand through to a `direct_call` of `ll_setitem_fast`; only the
    /// post-add bound check raises IndexError.
    #[test]
    fn build_ll_list_setitem_checked_helper_resized_reads_length_via_getfield() {
        // Keep `ann` alive (see the getitem-checked test above).
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let repr = ListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>).expect("ListRepr::new");
        let pygraph = build_ll_list_setitem_checked_helper_graph(
            &rtyper,
            ListLayout::Resized,
            "ll_setitem_checked",
            repr.lowleveltype().clone(),
            LowLevelType::Signed,
        )
        .expect("build_ll_list_setitem_checked_helper_graph");
        let seqs = block_op_sequences(&pygraph);
        assert!(
            seqs.contains(&vec![
                "getfield".to_string(),
                "cast_int_to_uint".to_string(),
                "cast_int_to_uint".to_string(),
                "uint_ge".to_string(),
            ]),
            "resized start must read length (getfield), cast to unsigned, then uint_ge, got {seqs:?}"
        );
        assert!(
            seqs.contains(&vec![
                "uint_add".to_string(),
                "uint_ge".to_string(),
                "cast_uint_to_int".to_string(),
            ]),
            "expected a uint_add/uint_ge/cast_uint_to_int fixup block, got {seqs:?}"
        );
        assert!(
            seqs.contains(&vec!["direct_call".to_string()]),
            "expected a direct_call dispatch block, got {seqs:?}"
        );
        assert_eq!(
            count_links_to_exceptblock(&pygraph),
            1,
            "only the post-add bound check raises IndexError"
        );
    }

    /// `ListIteratorRepr`'s lowleveltype is `Ptr(GcStruct("listiter",
    /// ("list", LIST), ("index", Signed)))` (`lltypesystem/rlist.py`).
    #[test]
    fn list_iterator_repr_lltype_is_ptr_gcstruct_list_index() {
        let rtyper = fresh_rtyper();
        let r_list = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let r_iter = ListIteratorRepr::new(
            r_list.lowleveltype().clone(),
            signed_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            true,
            false,
        )
        .expect("ListIteratorRepr::new");
        assert_eq!(r_iter.class_name(), "ListIteratorRepr");
        assert_eq!(r_iter.repr_class_id(), ReprClassId::ListIteratorRepr);

        let LowLevelType::Ptr(ptr) = r_iter.lowleveltype() else {
            panic!("ListIteratorRepr lltype must be a Ptr");
        };
        let PtrTarget::Struct(body) = &ptr.TO else {
            panic!("ListIteratorRepr Ptr target must be a Struct");
        };
        assert_eq!(body._name, "listiter");
        assert_eq!(body._flds.get("index"), Some(&LowLevelType::Signed));
        // the `list` field carries the list repr's own lowleveltype.
        assert_eq!(body._flds.get("list"), Some(r_list.lowleveltype()));
    }

    /// `SomeIterator(SomeList)` routes through `SomeIterator.rtyper_makerepr`
    /// → `r_container.make_iterator_repr()` → `ListIteratorRepr`
    /// (`rmodel.py`).
    #[test]
    fn makerepr_somelist_iterator_routes_to_list_iterator_repr() {
        let rtyper = fresh_rtyper_live();
        let ldef = ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
            false,
        );
        let s_list = SomeValue::List(SomeList::new(ldef));
        let s_iter =
            SomeValue::Iterator(crate::annotator::model::SomeIterator::new(s_list, vec![]));
        let repr = rtyper_makerepr(&s_iter, &rtyper).expect("rtyper_makerepr list iterator");
        assert_eq!(repr.class_name(), "ListIteratorRepr");
        assert_eq!(repr.repr_class_id(), ReprClassId::ListIteratorRepr);
    }

    /// `ReversedListIteratorRepr`'s lowleveltype is
    /// `Ptr(GcStruct("revlistiter", ("list", LIST), ("index", Signed)))`
    /// (`lltypesystem/rlist.py`).
    #[test]
    fn reversed_list_iterator_repr_lltype_is_ptr_gcstruct_revlistiter() {
        let rtyper = fresh_rtyper();
        let r_list = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let r_iter = ReversedListIteratorRepr::new(
            r_list.lowleveltype().clone(),
            signed_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            true,
        )
        .expect("ReversedListIteratorRepr::new");
        assert_eq!(r_iter.class_name(), "ReversedListIteratorRepr");
        assert_eq!(
            r_iter.repr_class_id(),
            ReprClassId::ReversedListIteratorRepr
        );

        let LowLevelType::Ptr(ptr) = r_iter.lowleveltype() else {
            panic!("ReversedListIteratorRepr lltype must be a Ptr");
        };
        let PtrTarget::Struct(body) = &ptr.TO else {
            panic!("ReversedListIteratorRepr Ptr target must be a Struct");
        };
        assert_eq!(body._name, "revlistiter");
        assert_eq!(body._flds.get("index"), Some(&LowLevelType::Signed));
        assert_eq!(body._flds.get("list"), Some(r_list.lowleveltype()));
    }

    /// `SomeIterator(SomeList, variant=("reversed",))` routes through
    /// `SomeIterator.rtyper_makerepr` → `r_container.make_iterator_repr(*variant)`
    /// → `ReversedListIteratorRepr` (`rmodel.py`, `lltypesystem/rlist.py`
    /// `BaseListRepr.make_iterator_repr`).
    #[test]
    fn makerepr_somelist_reversed_iterator_routes_to_reversed_list_iterator_repr() {
        let rtyper = fresh_rtyper_live();
        let ldef = ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
            false,
        );
        let s_list = SomeValue::List(SomeList::new(ldef));
        let s_iter = SomeValue::Iterator(crate::annotator::model::SomeIterator::new(
            s_list,
            vec!["reversed".to_string()],
        ));
        let repr =
            rtyper_makerepr(&s_iter, &rtyper).expect("rtyper_makerepr reversed list iterator");
        assert_eq!(repr.class_name(), "ReversedListIteratorRepr");
        assert_eq!(repr.repr_class_id(), ReprClassId::ReversedListIteratorRepr);
    }

    /// `BaseListRepr.make_iterator_repr(("reversed",))` mints
    /// `ReversedListIteratorRepr`; any other nonempty variant is
    /// `TyperError` (`lltypesystem/rlist.py`).
    #[test]
    fn make_iterator_repr_dispatches_reversed_and_rejects_unknown_variants() {
        let rtyper = fresh_rtyper();
        let r_list = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let reversed = r_list
            .make_iterator_repr(&["reversed".to_string()], false)
            .expect("reversed variant");
        assert_eq!(reversed.class_name(), "ReversedListIteratorRepr");

        let r_resized =
            ListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>).expect("ListRepr::new");
        let reversed_resized = r_resized
            .make_iterator_repr(&["reversed".to_string()], false)
            .expect("resized reversed variant");
        assert_eq!(reversed_resized.class_name(), "ReversedListIteratorRepr");

        let err = r_list
            .make_iterator_repr(&["keys".to_string()], false)
            .expect_err("unknown list iterator variant");
        assert!(
            err.to_string()
                .contains("unsupported [\"keys\"] iterator over a list"),
            "unexpected error: {err}"
        );
    }

    /// `ll_revlistiter` body is `malloc(revlistiter)` → `setfield(list)` →
    /// `ll_length` → `int_sub(len, 1)` → `setfield(index)` (`lltypesystem/rlist.py`).
    #[test]
    fn build_ll_revlistiter_helper_emits_malloc_length_minus_one() {
        let rtyper = fresh_rtyper();
        let r_list = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let r_iter = ReversedListIteratorRepr::new(
            r_list.lowleveltype().clone(),
            signed_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            true,
        )
        .expect("ReversedListIteratorRepr::new");
        let pygraph = build_ll_revlistiter_helper_graph(
            "ll_revlistiter",
            r_list.lowleveltype().clone(),
            r_iter.lowleveltype().clone(),
            true,
            "length",
        )
        .expect("build_ll_revlistiter_helper_graph");
        let graph = pygraph.graph.borrow();
        let ops: Vec<_> = graph
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(
            ops,
            vec!["malloc", "setfield", "getarraysize", "int_sub", "setfield"]
        );
    }

    /// `ll_revlistnext` bounds-checks `index < 0` and raises StopIteration
    /// (`lltypesystem/rlist.py`).
    #[test]
    fn build_ll_revlistnext_helper_raises_on_negative_index() {
        let rtyper = fresh_rtyper_live();
        let r_list = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let r_iter = ReversedListIteratorRepr::new(
            r_list.lowleveltype().clone(),
            signed_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            true,
        )
        .expect("ReversedListIteratorRepr::new");
        let pygraph = build_ll_revlistnext_helper_graph(
            "ll_revlistnext",
            r_iter.lowleveltype().clone(),
            r_list.lowleveltype().clone(),
            LowLevelType::Signed,
            true,
            "items",
        )
        .expect("build_ll_revlistnext_helper_graph");
        let graph = pygraph.graph.borrow();
        let start_ops: Vec<_> = graph
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(start_ops, vec!["getfield", "getfield", "int_lt"]);
        let start = graph.startblock.borrow();
        assert_eq!(start.exits.len(), 2);
        drop(start);
        assert_eq!(
            count_links_to_exceptblock(&pygraph),
            1,
            "index < 0 must raise StopIteration"
        );
    }

    /// Production annotate → rtype path: a graph `next(reversed(lst))` does
    /// not hand-build `ReversedListIteratorRepr`; `SomeIterator.rtyper_makerepr`
    /// forwards the `"reversed"` variant and `rtype_builtin_reversed` calls
    /// `newiter`.
    #[test]
    fn annotate_and_rtype_reversed_list_iterates_via_revlist_helpers() {
        use std::cell::RefCell;

        use crate::flowspace::model::{
            BlockRefExt, ConstValue, Constant, HOST_ENV, Hlvalue, Link, SpaceOperation,
        };
        use crate::translator::translator::TranslationContext;

        let translator = TranslationContext::new();
        let v_lst = Variable::named("lst");
        let startblock = Block::shared(vec![Hlvalue::Variable(v_lst.clone())]);
        let reversed = HOST_ENV
            .lookup_builtin("reversed")
            .expect("HOST_ENV missing builtin reversed");
        let v_it = Variable::named("it");
        let v_x = Variable::named("x");
        let v_ret = Variable::named("ret");
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "simple_call",
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::HostObject(reversed))),
                Hlvalue::Variable(v_lst.clone()),
            ],
            Hlvalue::Variable(v_it.clone()),
        ));
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "next",
            vec![Hlvalue::Variable(v_it)],
            Hlvalue::Variable(v_x.clone()),
        ));
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "reversed_next",
            startblock.clone(),
            Hlvalue::Variable(v_ret),
        )));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v_x)],
                Some(graph.borrow().returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        translator.graphs.borrow_mut().push(Rc::clone(&graph));
        let ann = RPythonAnnotator::new(Some(translator), None, None, false);
        let ldef = ListDef::new(
            Some(ann.bookkeeper.clone()),
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
            false,
        );
        let s_list = SomeValue::List(SomeList::new(ldef));
        let annotated = ann
            .build_graph_types(&graph, &[Some(s_list)], true)
            .expect("annotate reversed(list)");
        assert!(
            matches!(annotated, Some(SomeValue::Integer(_))),
            "next(reversed(lst)) must annotate as the list element, got {annotated:?}"
        );

        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        rtyper
            .specialize(true)
            .expect("rtype reversed(list) through annotate → specialize");

        let callees = direct_call_callees(&graph.borrow());
        let names: Vec<&str> = callees.iter().map(|(_, name)| name.as_str()).collect();
        assert!(
            names.iter().any(|n| n.contains("ll_revlistiter")),
            "reversed(lst) must lower through ll_revlistiter, got {names:?}"
        );
        assert!(
            names.iter().any(|n| n.contains("ll_revlistnext")),
            "next(reversed(lst)) must lower through ll_revlistnext, got {names:?}"
        );
    }

    /// `ll_listiter` body is `malloc(listiter)` → `setfield(iter, "list",
    /// lst)` → `setfield(iter, "index", 0)` (`lltypesystem/rlist.py`).
    #[test]
    fn build_ll_listiter_helper_emits_malloc_then_two_setfields() {
        let rtyper = fresh_rtyper();
        let r_list = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let r_iter = ListIteratorRepr::new(
            r_list.lowleveltype().clone(),
            signed_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            true,
            false,
        )
        .expect("ListIteratorRepr::new");
        let pygraph = build_ll_listiter_helper_graph(
            "ll_listiter",
            r_list.lowleveltype().clone(),
            r_iter.lowleveltype().clone(),
        )
        .expect("build_ll_listiter_helper_graph");
        let graph = pygraph.graph.borrow();
        let ops: Vec<_> = graph
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(ops, vec!["malloc", "setfield", "setfield"]);
    }

    /// `iter(list)` rtypes through the default `Repr.rtype_iter`
    /// (`make_iterator_repr().newiter(hop)`) to a `direct_call(ll_listiter,
    /// v_lst)` (`rmodel.py` + `rlist.py:439-442`).  allow-line-citation
    #[test]
    fn fixed_size_list_iter_emits_direct_call_to_ll_listiter() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        let list_repr: Arc<FixedSizeListRepr> = Arc::new(
            FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
                .expect("FixedSizeListRepr::new"),
        );
        let list_lltype = list_repr.lowleveltype().clone();
        let iter_lltype = ListIteratorRepr::new(
            list_lltype.clone(),
            signed_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            true,
            false,
        )
        .expect("ListIteratorRepr::new")
        .lowleveltype()
        .clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_lltype));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(iter_lltype));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "iter".to_string(),
                vec![Hlvalue::Variable(v_list)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .push(SomeValue::List(SomeList::new(ListDef::new(
                None,
                SomeValue::Integer(SomeInteger::new(false, false)),
                false,
                false,
            ))));
        hop.args_r
            .borrow_mut()
            .push(Some(list_repr.clone() as Arc<dyn Repr>));

        let result = list_repr
            .rtype_iter(&hop)
            .unwrap_or_else(|err| panic!("list iter: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_listiter"),
            "expected 'll_listiter' in {dbg}"
        );
    }

    /// `ll_listnext` over a fixed list: startblock bounds-checks via
    /// `getfield`/`getfield`/`getarraysize`/`int_lt` and the continue block
    /// `int_add`/`setfield`/`getarrayitem` (`lltypesystem/rlist.py`).
    /// The out-of-bounds exit links to the graph's `exceptblock`.
    #[test]
    fn build_ll_listnext_helper_fixed_bounds_checks_and_getarrayitem() {
        let rtyper = fresh_rtyper_live();
        let r_list = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let r_iter = ListIteratorRepr::new(
            r_list.lowleveltype().clone(),
            signed_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            true,
            false,
        )
        .expect("ListIteratorRepr::new");
        let pygraph = build_ll_listnext_helper_graph(
            "ll_listnext",
            r_iter.lowleveltype().clone(),
            r_list.lowleveltype().clone(),
            LowLevelType::Signed,
            true,
            false,
            "length",
            "items",
        )
        .expect("build_ll_listnext_helper_graph");
        let graph = pygraph.graph.borrow();
        let start_ops: Vec<_> = graph
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(
            start_ops,
            vec!["getfield", "getfield", "getarraysize", "int_lt"]
        );
        // startblock branches on the bounds-check; one exit raises via the
        // graph's exceptblock.
        let start = graph.startblock.borrow();
        assert!(start.exitswitch.is_some(), "bounds-check exitswitch");
        let except_key = crate::flowspace::model::BlockKey::of(&graph.exceptblock);
        let raises = start.exits.iter().any(|lnk| {
            lnk.borrow()
                .target
                .as_ref()
                .is_some_and(|t| crate::flowspace::model::BlockKey::of(t) == except_key)
        });
        assert!(
            raises,
            "one startblock exit must link to exceptblock (raise StopIteration)"
        );
        // the non-raising exit's continue block reads the element.
        let cont_ops: Vec<Vec<String>> = graph
            .iterblocks()
            .iter()
            .map(|b| {
                b.borrow()
                    .operations
                    .iter()
                    .map(|op| op.opname.clone())
                    .collect()
            })
            .collect();
        assert!(
            cont_ops.iter().any(|seq| seq
                == &vec![
                    "int_add".to_string(),
                    "setfield".to_string(),
                    "getarrayitem".to_string()
                ]),
            "continue block must int_add/setfield/getarrayitem, got {cont_ops:?}"
        );
    }

    /// `ll_listnext_foldable` over an unmutated fixed list reads the element via
    /// the PURE `getarrayitem_pure` (`lltypesystem/rlist.py` →
    /// `ll_getitem_foldable_nonneg`), so the trace optimizer can fold / CSE the
    /// load across iterations.
    #[test]
    fn build_ll_listnext_foldable_helper_emits_getarrayitem_pure() {
        let rtyper = fresh_rtyper_live();
        let r_list = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let r_iter = ListIteratorRepr::new(
            r_list.lowleveltype().clone(),
            signed_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            true,
            true,
        )
        .expect("ListIteratorRepr::new");
        let pygraph = build_ll_listnext_helper_graph(
            "ll_listnext_foldable",
            r_iter.lowleveltype().clone(),
            r_list.lowleveltype().clone(),
            LowLevelType::Signed,
            true,
            true,
            "length",
            "items",
        )
        .expect("build_ll_listnext_helper_graph");
        let graph = pygraph.graph.borrow();
        let cont_ops: Vec<Vec<String>> = graph
            .iterblocks()
            .iter()
            .map(|b| {
                b.borrow()
                    .operations
                    .iter()
                    .map(|op| op.opname.clone())
                    .collect()
            })
            .collect();
        assert!(
            cont_ops.iter().any(|seq| seq
                == &vec![
                    "int_add".to_string(),
                    "setfield".to_string(),
                    "getarrayitem_pure".to_string()
                ]),
            "foldable continue block must emit getarrayitem_pure, got {cont_ops:?}"
        );
    }

    /// `ll_listnext` over a resized list reads `length` from the header and
    /// `items` array before `getarrayitem` (`lltypesystem/rlist.py` ADTIList).
    #[test]
    fn build_ll_listnext_helper_resized_reads_length_and_items() {
        let rtyper = fresh_rtyper_live();
        let r_list = ListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>).expect("ListRepr::new");
        let r_iter = ListIteratorRepr::new(
            r_list.lowleveltype().clone(),
            signed_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            false,
            false,
        )
        .expect("ListIteratorRepr::new");
        let pygraph = build_ll_listnext_helper_graph(
            "ll_listnext",
            r_iter.lowleveltype().clone(),
            r_list.lowleveltype().clone(),
            LowLevelType::Signed,
            false,
            false,
            "length",
            "items",
        )
        .expect("build_ll_listnext_helper_graph");
        let graph = pygraph.graph.borrow();
        let start_ops: Vec<_> = graph
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        // resized length via getfield "length" (not getarraysize).
        assert_eq!(
            start_ops,
            vec!["getfield", "getfield", "getfield", "int_lt"]
        );
        let cont_ops: Vec<Vec<String>> = graph
            .iterblocks()
            .iter()
            .map(|b| {
                b.borrow()
                    .operations
                    .iter()
                    .map(|op| op.opname.clone())
                    .collect()
            })
            .collect();
        assert!(
            cont_ops.iter().any(|seq| seq
                == &vec![
                    "int_add".to_string(),
                    "setfield".to_string(),
                    "getfield".to_string(),
                    "getarrayitem".to_string()
                ]),
            "resized continue must read items array before getarrayitem, got {cont_ops:?}"
        );
    }

    /// `next(iter)` rtypes through `ListIteratorRepr::rtype_next` to a
    /// `direct_call(ll_listnext, v_iter)`, recording the implicit
    /// `StopIteration` (`rlist.py`).
    #[test]
    fn list_iterator_next_emits_direct_call_to_ll_listnext() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        let list_repr = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let iter_repr: Arc<ListIteratorRepr> = Arc::new(
            ListIteratorRepr::new(
                list_repr.lowleveltype().clone(),
                signed_repr() as Arc<dyn Repr>,
                signed_repr() as Arc<dyn Repr>,
                true,
                false,
            )
            .expect("ListIteratorRepr::new"),
        );
        let iter_lltype = iter_repr.lowleveltype().clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_iter = Variable::new();
        v_iter.set_concretetype(Some(iter_lltype));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "next".to_string(),
                vec![Hlvalue::Variable(v_iter)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().push(SomeValue::Iterator(
            crate::annotator::model::SomeIterator::new(
                SomeValue::List(SomeList::new(ListDef::new(
                    None,
                    SomeValue::Integer(SomeInteger::new(false, false)),
                    false,
                    false,
                ))),
                vec![],
            ),
        ));
        hop.args_r
            .borrow_mut()
            .push(Some(iter_repr.clone() as Arc<dyn Repr>));

        let result = iter_repr
            .rtype_next(&hop)
            .unwrap_or_else(|err| panic!("list next: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "rtype_next must call hop.exception_is_here()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_listnext"),
            "expected 'll_listnext' in {dbg}"
        );
    }

    /// BFS the CFG from `start`, returning `(block_count, all_opnames)`.
    /// Mirrors the `walk_ops` helper the rbuilder tests use.
    fn walk_ops(start: &BlockRef) -> (usize, Vec<String>) {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![start.clone()];
        let mut all_ops: Vec<String> = Vec::new();
        let mut count = 0usize;
        while let Some(b) = stack.pop() {
            if !seen.insert(Rc::as_ptr(&b) as usize) {
                continue;
            }
            count += 1;
            let bb = b.borrow();
            for op in &bb.operations {
                all_ops.push(op.opname.clone());
            }
            for link in &bb.exits {
                if let Some(t) = link.borrow().target.clone() {
                    stack.push(t);
                }
            }
        }
        (count, all_ops)
    }

    fn direct_call_func_name(arg: &Hlvalue) -> Option<String> {
        let Hlvalue::Constant(constant) = arg else {
            return None;
        };
        let ConstValue::LLPtr(ptr) = &constant.value else {
            return None;
        };
        let crate::translator::rtyper::lltypesystem::lltype::PtrObj::Strong(obj) = &ptr._obj0
        else {
            return None;
        };
        let crate::translator::rtyper::lltypesystem::lltype::_ptr_obj::Func(func) = obj else {
            return None;
        };
        Some(func._name.clone())
    }

    fn direct_call_callees(graph: &FunctionGraph) -> Vec<(usize, String)> {
        let mut out = Vec::new();
        for block in graph.iterblocks() {
            for op in &block.borrow().operations {
                if op.opname != "direct_call" {
                    continue;
                }
                let Some(name) = op.args.first().and_then(direct_call_func_name) else {
                    continue;
                };
                out.push((op.args.len(), name));
            }
        }
        out
    }

    /// The jit arm calls the `look_inside_iff` wrapper, and the public
    /// nonnull name stays the original four-block loop.
    fn assert_look_inside_family(
        graphs: &[crate::flowspace::model::GraphRef],
        jit: &str,
        nonnull: &str,
        clear: &str,
    ) {
        let find = |name: &str| {
            graphs
                .iter()
                .find(|g| g.borrow().name == name)
                .unwrap_or_else(|| panic!("missing {name}"))
                .clone()
        };
        let wrapper = format!("{nonnull}_look_inside_iff");
        let trampoline = format!("{nonnull}_trampoline");
        for name in [
            wrapper.as_str(),
            trampoline.as_str(),
            "_ll_alloc_and_set_isconstant",
        ] {
            assert!(
                graphs.iter().any(|g| g.borrow().name == name),
                "family must mint `{name}`"
            );
        }
        let calls = direct_call_callees(&find(jit).borrow());
        assert!(
            calls.iter().any(|(n, name)| *n == 3 && name == &wrapper),
            "jit 3-arg callee is the wrapper, got {calls:?}"
        );
        assert!(
            calls.iter().any(|(n, name)| *n == 2 && name == clear),
            "jit 2-arg callee is the clear arm, got {calls:?}"
        );
        assert!(
            !calls.iter().any(|(_, name)| name == nonnull),
            "jit arm must not call the loop directly, got {calls:?}"
        );
        let tramp_func = find(&trampoline)
            .borrow()
            .func
            .as_ref()
            .cloned()
            .expect("trampoline GraphFunc");
        assert_eq!(tramp_func._jit_look_inside_, Some(false));
        assert_eq!(
            tramp_func.annspecialcase.as_deref(),
            Some("specialize:call_location")
        );
        let orig_func = find(nonnull)
            .borrow()
            .func
            .as_ref()
            .cloned()
            .expect("nonnull GraphFunc");
        assert!(
            orig_func._jit_unroll_safe_,
            "look_inside_iff marks the original unroll_safe"
        );
        assert_eq!(
            orig_func.annspecialcase.as_deref(),
            Some("specialize:call_location")
        );
        let isc = find("_ll_alloc_and_set_isconstant");
        let isc_g = isc.borrow();
        assert_eq!(
            isc_g.func.as_ref().unwrap().annspecialcase.as_deref(),
            Some("specialize:call_location")
        );
        let isc_calls = direct_call_callees(&isc_g);
        assert!(
            isc_calls
                .iter()
                .any(|(_, name)| name == "majit_rlib::jit::isconstant"),
            "isconstant helper calls the marked intrinsic, got {isc_calls:?}"
        );
        let wrap_func = find(&wrapper)
            .borrow()
            .func
            .as_ref()
            .cloned()
            .expect("wrapper GraphFunc");
        assert_eq!(
            wrap_func.annspecialcase, None,
            "look_inside_iff wrapper f does not get call_location"
        );
        let start = find(nonnull).borrow().startblock.clone();
        let (count, all_ops) = walk_ops(&start);
        assert_eq!(count, 4, "public nonnull name stays the 4-block loop");
        let n = |name: &str| all_ops.iter().filter(|op| op.as_str() == name).count();
        assert_eq!(n("direct_call"), 2);
        assert_eq!(n("int_lt"), 1);
        assert_eq!(n("int_add"), 1);
    }

    /// `ll_arraymove` must dispatch on the sign of `delta` and carry both
    /// element loops, so a right move — `ll_insert_nonneg`'s
    /// `ll_arraymove(l, index, index + 1, length - index)` — walks descending
    /// instead of smearing the first element across the overlap.
    ///
    /// The single name matters as much as the shape: the rtyper caches
    /// low-level helpers on `(name, args, result)`, and both directions
    /// present the identical key, so a direction-specialised graph would be
    /// served to whichever producer minted second.
    #[test]
    fn build_ll_arraymove_dispatches_on_delta_and_carries_both_directions() {
        let helper = build_ll_arraymove_helper_graph("ll_arraymove", LowLevelType::Signed)
            .expect("build_ll_arraymove_helper_graph");
        assert_eq!(helper.func.name, "ll_arraymove");
        let inner = helper.graph.borrow();

        // start: delta = dest_start - source_start; goes_up = delta < 0;
        // last_index = length - 1, then a two-way switch on `goes_up`.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["int_sub", "int_lt", "int_sub"]);
        assert_eq!(startblock.inputargs.len(), 4); // array, source_start, dest_start, length
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);

        let (count, all_ops) = walk_ops(&inner.startblock);
        // start, two cond blocks, two body blocks, returnblock.
        assert_eq!(count, 6);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        // One `int_lt` picks the direction, one bounds the ascending loop; the
        // descending loop bounds on `int_ge(i, 0)`.
        assert_eq!(n("int_lt"), 2);
        assert_eq!(n("int_ge"), 1);
        // `delta`, `last_index`, and the descending step.
        assert_eq!(n("int_sub"), 3);
        // Two index offsets per arm, plus the ascending step.
        assert_eq!(n("int_add"), 5);
        // One copy per arm.
        assert_eq!(n("getarrayitem"), 2);
        assert_eq!(n("setarrayitem"), 2);
        // The body is upstream's split-address-space arm, on which
        // `rgc.ll_arraymove` emits no move barrier — that call is the `elif`
        // guarding the `raw_memmove_no_free` fast path, which is not emitted.
        assert_eq!(n("gc_writebarrier_before_move"), 0);
        assert_eq!(n("raw_memmove_no_free"), 0);
    }

    fn dummy_funcptr_const() -> Constant {
        Constant::with_concretetype(ConstValue::None, LowLevelType::Void)
    }

    /// Signed `Ptr(GcStruct("list", …))` lltype for the resized layout tests.
    fn resized_list_lltype() -> LowLevelType {
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        ListRepr::new(&rtyper, r_int)
            .expect("ListRepr::new")
            .lowleveltype()
            .clone()
    }

    /// `rtype_alloc_and_set` on a resized `ListRepr` mints `ll_alloc_and_set`
    /// and emits a single top-level `direct_call` to it. The `(count, item)`
    /// pair is threaded; the `LIST` is folded into the helper identity.
    #[test]
    fn rtype_alloc_and_set_resized_mints_ll_alloc_and_set() {
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::SpaceOperation;
        use std::cell::RefCell as StdRef;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_list: Arc<dyn Repr> =
            Arc::new(ListRepr::new(&rtyper, r_int.clone()).expect("ListRepr::new"));

        let v_count = Variable::new();
        v_count.set_concretetype(Some(LowLevelType::Signed));
        let v_item = Variable::new();
        v_item.set_concretetype(Some(LowLevelType::Signed));
        let v_count_h = Hlvalue::Variable(v_count);
        let v_item_h = Hlvalue::Variable(v_item);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            "alloc_and_set".to_string(),
            vec![v_count_h.clone(), v_item_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_count_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        hop.args_v.borrow_mut().push(v_item_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        *hop.r_result.borrow_mut() = Some(r_list.clone());

        let out = rtype_alloc_and_set(&hop)
            .expect("rtype_alloc_and_set must lower")
            .expect("alloc_and_set returns the fresh list Variable");
        assert!(matches!(out, Hlvalue::Variable(_)));

        let ops = hop.llops.borrow();
        let calls: Vec<_> = ops
            .ops
            .iter()
            .filter(|op| op.opname == "direct_call")
            .collect();
        assert_eq!(
            calls.len(),
            1,
            "one top-level direct_call to ll_alloc_and_set"
        );

        // The minted family graphs must be registered under their upstream
        // names (the resized spellings), plus the `look_inside_iff` split.
        let graphs = ann.translator.graphs.borrow().clone();
        let names: Vec<String> = graphs.iter().map(|g| g.borrow().name.clone()).collect();
        for expected in [
            "ll_alloc_and_set",
            "_ll_alloc_and_set_jit",
            "_ll_alloc_and_set_nojit",
            "_ll_alloc_and_clear",
            "_ll_alloc_and_set_nonnull",
            "_ll_alloc_and_set_nonnull_look_inside_iff",
            "_ll_alloc_and_set_nonnull_trampoline",
            "_ll_alloc_and_set_isconstant",
            "ll_arrayclear",
            "ll_arrayfill",
            "ll_newlist",
        ] {
            assert!(
                names.iter().any(|n| n == expected),
                "family must mint `{expected}`, got {names:?}"
            );
        }
        assert_look_inside_family(
            &graphs,
            "_ll_alloc_and_set_jit",
            "_ll_alloc_and_set_nonnull",
            "_ll_alloc_and_clear",
        );
    }

    /// The fixed layout mints the `ll_fixed_*` dispatch/jit/nojit spellings so
    /// the `(name, args, result)` helper cache never serves a resized graph
    /// for a fixed request.
    #[test]
    fn rtype_alloc_and_set_fixed_mints_ll_fixed_alloc_and_set() {
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::SpaceOperation;
        use std::cell::RefCell as StdRef;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_list: Arc<dyn Repr> = Arc::new(
            FixedSizeListRepr::new(&rtyper, r_int.clone()).expect("FixedSizeListRepr::new"),
        );

        let v_count = Variable::new();
        v_count.set_concretetype(Some(LowLevelType::Signed));
        let v_item = Variable::new();
        v_item.set_concretetype(Some(LowLevelType::Signed));
        let v_count_h = Hlvalue::Variable(v_count);
        let v_item_h = Hlvalue::Variable(v_item);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            "alloc_and_set".to_string(),
            vec![v_count_h.clone(), v_item_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_count_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        hop.args_v.borrow_mut().push(v_item_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        *hop.r_result.borrow_mut() = Some(r_list.clone());

        rtype_alloc_and_set(&hop)
            .expect("rtype_alloc_and_set must lower")
            .expect("alloc_and_set returns the fresh list Variable");

        let graphs = ann.translator.graphs.borrow().clone();
        let names: Vec<String> = graphs.iter().map(|g| g.borrow().name.clone()).collect();
        assert!(
            names.iter().any(|n| n == "ll_fixed_alloc_and_set"),
            "fixed arm mints ll_fixed_alloc_and_set, got {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "ll_fixed_newlist"),
            "fixed arm mints ll_fixed_newlist, got {names:?}"
        );
        for expected in [
            "_ll_fixed_alloc_and_set_nonnull_look_inside_iff",
            "_ll_fixed_alloc_and_set_nonnull_trampoline",
            "_ll_alloc_and_set_isconstant",
        ] {
            assert!(
                names.iter().any(|n| n == expected),
                "fixed arm must mint `{expected}`, got {names:?}"
            );
        }
        assert!(
            !names.iter().any(|n| n == "ll_alloc_and_set"),
            "fixed arm must NOT mint the resized ll_alloc_and_set, got {names:?}"
        );
        assert_look_inside_family(
            &graphs,
            "_ll_fixed_alloc_and_set_jit",
            "_ll_fixed_alloc_and_set_nonnull",
            "_ll_fixed_alloc_and_clear",
        );
    }

    /// The two layouts must name their clear sub-helper DISTINCTLY, otherwise
    /// the `(name, args, ret)` helper cache and the `mark_oopspec` CallPath
    /// registration collide: a resized `newlist_clear` registration would
    /// overwrite (or serve) the fixed one.  Fixed prefixes `_ll_fixed_*`,
    /// resized stays unprefixed.
    #[test]
    fn alloc_and_set_sub_names_clear_helpers_are_per_layout_distinct() {
        let (_, fixed_clear, _, _, _) = alloc_and_set_sub_names(ListLayout::Fixed);
        let (_, resized_clear, _, _, _) = alloc_and_set_sub_names(ListLayout::Resized);
        assert_ne!(
            fixed_clear, resized_clear,
            "fixed and resized clear helpers must be distinct names"
        );
        assert_eq!(fixed_clear, "_ll_fixed_alloc_and_clear");
        assert_eq!(resized_clear, "_ll_alloc_and_clear");
    }

    /// `ll_alloc_and_set` clamps `count` with `int_force_ge_zero` and branches
    /// on the `we_are_jitted()` SpecTag: true → jit arm direct_call, false →
    /// nojit arm direct_call.
    #[test]
    fn build_ll_alloc_and_set_clamps_then_branches_on_we_are_jitted() {
        let helper = build_ll_alloc_and_set_helper_graph(
            "ll_alloc_and_set",
            resized_list_lltype(),
            LowLevelType::Signed,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_alloc_and_set_helper_graph");
        assert_eq!(helper.func.name, "ll_alloc_and_set");
        let inner = helper.graph.borrow();

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["int_force_ge_zero"]);
        assert_eq!(startblock.inputargs.len(), 2); // count, item
        assert_eq!(startblock.exits.len(), 2); // jit / nojit arms
        // exitswitch is the identity-bearing we_are_jitted SpecTag.
        let Some(Hlvalue::Constant(c)) = &startblock.exitswitch else {
            panic!("exitswitch must be the we_are_jitted SpecTag constant");
        };
        assert!(matches!(c.value, ConstValue::SpecTag(_)));
        drop(startblock);

        let (count, all_ops) = walk_ops(&inner.startblock);
        // start, jit, nojit, returnblock.
        assert_eq!(count, 4);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        assert_eq!(n("int_force_ge_zero"), 1);
        assert_eq!(n("direct_call"), 2); // jit + nojit arms
    }

    /// `_ll_alloc_and_set_jit` branches on `_ll_zero_or_null(item)`: true →
    /// clear arm `(count)`, false → nonnull arm `(count, item)`. The
    /// integer-item zero test is `int_is_true` negated by `bool_not`.
    #[test]
    fn build_ll_alloc_and_set_jit_branches_on_zero_or_null() {
        let helper = build_ll_alloc_and_set_jit_helper_graph(
            "_ll_alloc_and_set_jit",
            resized_list_lltype(),
            LowLevelType::Signed,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_alloc_and_set_jit_helper_graph");
        let inner = helper.graph.borrow();

        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        // integer item: int_is_true then bool_not.
        assert_eq!(start_ops, vec!["int_is_true", "bool_not"]);
        assert_eq!(startblock.exits.len(), 2);
        assert!(matches!(startblock.exitswitch, Some(Hlvalue::Variable(_))));
        drop(startblock);

        let (count, all_ops) = walk_ops(&inner.startblock);
        // start, clear arm, nonnull arm, returnblock.
        assert_eq!(count, 4);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        assert_eq!(n("direct_call"), 2); // clear + nonnull
        assert_eq!(n("bool_not"), 1);
    }

    /// `_ll_alloc_and_set_jit` with a Ptr item uses `ptr_nonzero` (negated by
    /// `bool_not`) for the zero/null test.
    #[test]
    fn build_ll_alloc_and_set_jit_ptr_item_uses_ptr_nonzero() {
        let ptr_list = resized_list_lltype();
        // item lltype = generic gc pointer.
        let item = items_array_ptr_lltype(&LowLevelType::Signed);
        let helper = build_ll_alloc_and_set_jit_helper_graph(
            "_ll_alloc_and_set_jit",
            ptr_list,
            item,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_alloc_and_set_jit_helper_graph");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["ptr_nonzero", "bool_not"]);
    }

    /// `_ll_alloc_and_set_nojit` (resized, non-Ptr item): `ll_newlist(count)`,
    /// branch on zero/null; the zero arm reads `items` (getfield) then
    /// `ll_arrayclear`, the nonzero arm reads `items` then `ll_arrayfill`.
    #[test]
    fn build_ll_alloc_and_set_nojit_resized_nonptr_clears_and_fills() {
        let helper = build_ll_alloc_and_set_nojit_helper_graph(
            "_ll_alloc_and_set_nojit",
            ListLayout::Resized,
            resized_list_lltype(),
            LowLevelType::Signed,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_alloc_and_set_nojit_helper_graph");
        let inner = helper.graph.borrow();

        let (count, all_ops) = walk_ops(&inner.startblock);
        // start, zero arm, nonzero arm, returnblock.
        assert_eq!(count, 4);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        // ll_newlist + arrayclear (zero arm) + arrayfill (nonzero arm).
        assert_eq!(n("direct_call"), 3);
        // Resized reads `items` on BOTH arms.
        assert_eq!(n("getfield"), 2);
        assert_eq!(n("bool_not"), 1);
    }

    /// A Ptr-element resized nojit clear arm skips `ll_arrayclear` entirely
    /// (the array was zeroed by `ll_newlist`), so only the nonzero arm reads
    /// `items` and calls `ll_arrayfill` — 2 direct_calls, 1 getfield.
    #[test]
    fn build_ll_alloc_and_set_nojit_ptr_item_skips_arrayclear() {
        let item = items_array_ptr_lltype(&LowLevelType::Signed);
        let helper = build_ll_alloc_and_set_nojit_helper_graph(
            "_ll_alloc_and_set_nojit",
            ListLayout::Resized,
            resized_list_lltype(),
            item,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_alloc_and_set_nojit_helper_graph");
        let inner = helper.graph.borrow();
        let (_count, all_ops) = walk_ops(&inner.startblock);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        // ll_newlist + arrayfill only (no arrayclear on the zero arm).
        assert_eq!(n("direct_call"), 2);
        // Only the nonzero (fill) arm reads `items`.
        assert_eq!(n("getfield"), 1);
    }

    /// Fixed-layout nojit: the receiver IS the array ptr, so NO `items`
    /// getfield is emitted on either arm.
    #[test]
    fn build_ll_alloc_and_set_nojit_fixed_passes_receiver_directly() {
        let rtyper = fresh_rtyper();
        let fixed_lltype = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new")
            .lowleveltype()
            .clone();
        let helper = build_ll_alloc_and_set_nojit_helper_graph(
            "_ll_fixed_alloc_and_set_nojit",
            ListLayout::Fixed,
            fixed_lltype,
            LowLevelType::Signed,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_alloc_and_set_nojit_helper_graph");
        let inner = helper.graph.borrow();
        let (_count, all_ops) = walk_ops(&inner.startblock);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        assert_eq!(n("getfield"), 0); // fixed value is already the array ptr
        assert_eq!(n("direct_call"), 3); // newlist + arrayclear + arrayfill
    }

    /// `_ll_alloc_and_clear` (resized): `ll_newlist(count)` → `getfield(items)`
    /// → `ll_arrayclear(items)` → return l — a single straight-line block
    /// (`malloc_zero_filled` collapses the early return away).
    #[test]
    fn build_ll_alloc_and_clear_newlist_then_arrayclear() {
        let helper = build_ll_alloc_and_clear_helper_graph(
            "_ll_alloc_and_clear",
            ListLayout::Resized,
            resized_list_lltype(),
            LowLevelType::Signed,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_alloc_and_clear_helper_graph");
        assert_eq!(helper.func.name, "_ll_alloc_and_clear");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|o| o.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["direct_call", "getfield", "direct_call"]);
        assert_eq!(startblock.inputargs.len(), 1); // count only (LIST folded)
        assert_eq!(startblock.exits.len(), 1); // straight-line return
    }

    /// `_ll_alloc_and_set_nonnull`: `ll_newlist(count)` then the runtime
    /// `while i < count: setitem_fast(l, i, item); i += 1` loop.
    #[test]
    fn build_ll_alloc_and_set_nonnull_newlist_then_setitem_loop() {
        let helper = build_ll_alloc_and_set_nonnull_helper_graph(
            "_ll_alloc_and_set_nonnull",
            resized_list_lltype(),
            LowLevelType::Signed,
            dummy_funcptr_const(),
            dummy_funcptr_const(),
        )
        .expect("build_ll_alloc_and_set_nonnull_helper_graph");
        let inner = helper.graph.borrow();

        let (count, all_ops) = walk_ops(&inner.startblock);
        // start, loop header, loop body, returnblock.
        assert_eq!(count, 4);
        let n = |name: &str| all_ops.iter().filter(|o| o.as_str() == name).count();
        assert_eq!(n("direct_call"), 2); // ll_newlist + ll_setitem_fast (loop body)
        assert_eq!(n("int_lt"), 1); // loop header
        assert_eq!(n("int_add"), 1); // i += 1
    }

    /// Char-element zero test uses `cast_char_to_int` then `int_is_true`;
    /// UniChar uses `cast_unichar_to_int`; Float compares bits;
    /// SingleFloat widens then compares bits.
    #[test]
    fn emit_zero_or_null_op_chain_per_element_type() {
        let probe = |lltype: LowLevelType| -> Vec<String> {
            let item = variable_with_lltype("item", lltype.clone());
            let block = Block::shared(vec![Hlvalue::Variable(item.clone())]);
            emit_zero_or_null(&block, &lltype, Hlvalue::Variable(item));
            block
                .borrow()
                .operations
                .iter()
                .map(|o| o.opname.clone())
                .collect()
        };
        assert_eq!(
            probe(LowLevelType::Char),
            vec!["cast_char_to_int", "int_is_true", "bool_not"]
        );
        assert_eq!(
            probe(LowLevelType::UniChar),
            vec!["cast_unichar_to_int", "int_is_true", "bool_not"]
        );
        assert_eq!(
            probe(LowLevelType::Float),
            vec![
                "convert_float_bytes_to_longlong",
                "llong_is_true",
                "bool_not"
            ]
        );
        assert_eq!(
            probe(LowLevelType::SingleFloat),
            vec![
                "cast_singlefloat_to_float",
                "convert_float_bytes_to_longlong",
                "llong_is_true",
                "bool_not"
            ]
        );
        assert_eq!(
            probe(LowLevelType::LongFloat),
            vec!["float_is_true", "bool_not"]
        );
        assert_eq!(probe(LowLevelType::Signed), vec!["int_is_true", "bool_not"]);
        assert_eq!(
            probe(items_array_ptr_lltype(&LowLevelType::Signed)),
            vec!["ptr_nonzero", "bool_not"]
        );
    }

    /// Fold the Float/SingleFloat `_ll_zero_or_null` chain: only `+0.0`
    /// has bits 0 and takes the cleared arm; `-0.0` fills.
    #[test]
    fn emit_zero_or_null_float_neg_zero_does_not_clear() {
        use crate::translator::rtyper::lltypesystem::opimpl::get_op_impl;

        let fold_ops = |ops: &[SpaceOperation], mut current: ConstValue| -> bool {
            for op in ops {
                let fold_op =
                    get_op_impl(&op.opname).unwrap_or_else(|| panic!("no fold for {}", op.opname));
                current = fold_op(std::slice::from_ref(&current))
                    .unwrap_or_else(|| panic!("fold {} failed", op.opname));
            }
            match current {
                ConstValue::Bool(is_zero) => is_zero,
                other => panic!("expected Bool, got {other:?}"),
            }
        };

        let fold_float = |item: f64| -> bool {
            let item_var = variable_with_lltype("item", LowLevelType::Float);
            let block = Block::shared(vec![Hlvalue::Variable(item_var.clone())]);
            emit_zero_or_null(&block, &LowLevelType::Float, Hlvalue::Variable(item_var));
            fold_ops(&block.borrow().operations, ConstValue::float(item))
        };
        assert!(fold_float(0.0), "+0.0 takes the cleared arm");
        assert!(!fold_float(-0.0), "-0.0 does not take the cleared arm");

        let fold_single = |item: f32| -> bool {
            let item_var = variable_with_lltype("item", LowLevelType::SingleFloat);
            let block = Block::shared(vec![Hlvalue::Variable(item_var.clone())]);
            emit_zero_or_null(
                &block,
                &LowLevelType::SingleFloat,
                Hlvalue::Variable(item_var),
            );
            let ops = &block.borrow().operations;
            assert_eq!(ops[0].opname, "cast_singlefloat_to_float");
            // `bhimpl_cast_singlefloat_to_float`: `f32 as f64` keeps `-0.0`.
            fold_ops(&ops[1..], ConstValue::float(item as f64))
        };
        assert!(fold_single(0.0), "SingleFloat +0.0 takes the cleared arm");
        assert!(
            !fold_single(-0.0),
            "SingleFloat -0.0 does not take the cleared arm"
        );
    }
}
