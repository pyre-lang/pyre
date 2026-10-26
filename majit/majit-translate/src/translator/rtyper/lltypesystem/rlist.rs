//! RPython `rpython/rtyper/lltypesystem/rlist.py` parity module.
//!
//! The list repr slice lives in [`crate::translator::rtyper::rlist`]
//! together with the abstract `rpython/rtyper/rlist.py` pieces.  This
//! module keeps the upstream lltypesystem import path available.

use crate::annotator::bookkeeper::Bookkeeper;
use crate::annotator::specialize::memotable_for_args;
use crate::flowspace::model::ConstValue;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{_ptr, LowLevelType, MallocFlavor, malloc};

pub use crate::translator::rtyper::lltypesystem::lltype;
pub use crate::translator::rtyper::rlist::{AbstractListIteratorRepr, FixedSizeListRepr, ListRepr};
pub use crate::translator::rtyper::rmodel::Repr;

pub const INITIAL_EMPTY_LIST_ALLOCATION: i64 = 0;

/// RPython `class BaseListRepr(AbstractBaseListRepr)`
/// (`lltypesystem/rlist.py`).
///
/// The concrete list repr fields live in [`ListRepr`] and
/// [`FixedSizeListRepr`]. This placeholder keeps the lltypesystem-specific
/// inheritance surface visible until the full BaseListRepr split lands.
#[derive(Debug, Default)]
pub struct BaseListRepr;

/// RPython `class ListIteratorRepr(AbstractListIteratorRepr)`
/// (`lltypesystem/rlist.py`).
#[derive(Debug, Default)]
pub struct ListIteratorRepr;

/// RPython `class ReversedListIteratorRepr(AbstractListIteratorRepr)`
/// (`lltypesystem/rlist.py`).
#[derive(Debug, Default)]
pub struct ReversedListIteratorRepr;

fn lltypesystem_rlist_deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!("lltypesystem.rlist.{name} — list helper deferred"))
}

#[allow(non_snake_case)]
pub fn LIST_OF() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("LIST_OF"))
}

pub fn _ll_list_resize_hint_really() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("_ll_list_resize_hint_really"))
}

pub fn _ll_list_resize_hint() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("_ll_list_resize_hint"))
}

pub fn _ll_list_resize_really() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("_ll_list_resize_really"))
}

pub fn _ll_list_resize() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("_ll_list_resize"))
}

pub fn _ll_list_resize_ge() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("_ll_list_resize_ge"))
}

pub fn _ll_list_resize_le() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("_ll_list_resize_le"))
}

pub fn ll_append_noresize() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_append_noresize"))
}

pub fn ll_both_none() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_both_none"))
}

pub fn ll_newlist() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_newlist"))
}

pub fn ll_newlist_hint() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_newlist_hint"))
}

/// Identity stand-in for `_ll_prebuilt_empty_array`'s missing FunctionDesc.
/// `memo()` keys `all_specializations` by `funcdesc as *const FunctionDesc as usize`.
static LL_PREBUILT_EMPTY_ARRAY_FUNCDESC_KEY: u8 = 0;

fn malloc_prebuilt_empty_array(args: &[ConstValue]) -> Result<ConstValue, String> {
    let ty = match args.first() {
        Some(ConstValue::LowLevelType(ty)) => (**ty).clone(),
        other => {
            return Err(format!(
                "_ll_prebuilt_empty_array: expected LISTITEM type, got {other:?}"
            ));
        }
    };
    malloc(ty, Some(0), MallocFlavor::Gc, true).map(|ptr| ConstValue::LLPtr(Box::new(ptr)))
}

/// `rlist.py _ll_prebuilt_empty_array`: `@specialize.memo()` so a
/// constant `LISTITEM` folds to one immortal `malloc(LISTITEM, 0)`.
///
/// `memo()` needs a FunctionDesc whose `pyobj` is a user-function
/// HostObject with `GraphFunc.host_call` so it can run `func(*args)`
/// and return a SpecializeResult (annotation / dispatch graph). This
/// helper is a builder-defined rtyper function: no pyobj, and the
/// rtyper needs the immortal `_ptr`. Same map, same key kind, keyed by
/// this helper's identity.
pub fn _ll_prebuilt_empty_array(
    bookkeeper: &Bookkeeper,
    listitem: &LowLevelType,
) -> Result<_ptr, TyperError> {
    let funcdesc_key = std::ptr::addr_of!(LL_PREBUILT_EMPTY_ARRAY_FUNCDESC_KEY) as usize;
    let args = vec![ConstValue::LowLevelType(Box::new(listitem.clone()))];
    let memotable = memotable_for_args(
        bookkeeper,
        funcdesc_key,
        "_ll_prebuilt_empty_array",
        args.clone(),
        malloc_prebuilt_empty_array,
    )
    .map_err(|e| {
        TyperError::message(
            e.msg
                .unwrap_or_else(|| "_ll_prebuilt_empty_array memo failed".into()),
        )
    })?;
    match memotable.borrow().table.get(&args) {
        Some(ConstValue::LLPtr(ptr)) => Ok((**ptr).clone()),
        other => Err(TyperError::message(format!(
            "_ll_prebuilt_empty_array: MemoTable missing malloc(LISTITEM, 0), got {other:?}"
        ))),
    }
}

/// `rlist.py _ll_new_empty_item_array`. `listitem` is `LIST.items.TO`.
pub fn _ll_new_empty_item_array(
    bookkeeper: &Bookkeeper,
    listitem: &LowLevelType,
) -> Result<_ptr, TyperError> {
    if INITIAL_EMPTY_LIST_ALLOCATION > 0 {
        malloc(
            listitem.clone(),
            Some(INITIAL_EMPTY_LIST_ALLOCATION as usize),
            MallocFlavor::Gc,
            false,
        )
        .map_err(TyperError::message)
    } else {
        _ll_prebuilt_empty_array(bookkeeper, listitem)
    }
}

pub fn ll_newemptylist() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_newemptylist"))
}

pub fn ll_length() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_length"))
}

pub fn ll_items() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_items"))
}

pub fn ll_getitem_fast() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_getitem_fast"))
}

pub fn ll_setitem_fast() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_setitem_fast"))
}

pub fn ll_fixed_newlist() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_fixed_newlist"))
}

pub fn ll_fixed_newemptylist() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_fixed_newemptylist"))
}

pub fn ll_fixed_length() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_fixed_length"))
}

pub fn ll_fixed_items() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_fixed_items"))
}

pub fn ll_fixed_getitem_fast() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_fixed_getitem_fast"))
}

pub fn ll_fixed_setitem_fast() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_fixed_setitem_fast"))
}

pub fn newlist() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("newlist"))
}

pub fn ll_set_maxlength() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_set_maxlength"))
}

pub fn ll_list2fixed() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_list2fixed"))
}

pub fn ll_list2fixed_exact() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_list2fixed_exact"))
}

pub fn ll_listiter() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_listiter"))
}

pub fn ll_listnext() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_listnext"))
}

pub fn ll_listnext_foldable() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_listnext_foldable"))
}

pub fn ll_getnextindex() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_getnextindex"))
}

pub fn ll_revlistiter() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_revlistiter"))
}

pub fn ll_revlistnext() -> Result<(), TyperError> {
    Err(lltypesystem_rlist_deferred("ll_revlistnext"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lltypesystem_rlist_exposes_deferred_helper_surface() {
        let _base = BaseListRepr;
        let _iter = ListIteratorRepr;
        let _rev_iter = ReversedListIteratorRepr;
        let _abstract_iter = AbstractListIteratorRepr;
        assert_eq!(INITIAL_EMPTY_LIST_ALLOCATION, 0);

        let err = ll_newlist().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_newlist"));

        let err = ll_revlistnext().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_revlistnext"));
    }

    fn empty_array_identity(ptr: &_ptr) -> usize {
        match &ptr._obj0 {
            lltype::PtrObj::Strong(lltype::_ptr_obj::Array(arr)) => arr.identity(),
            other => panic!("expected Array container, got {other:?}"),
        }
    }

    /// `_ll_prebuilt_empty_array` is `@specialize.memo()`: one immortal
    /// zero-length GcArray per `LISTITEM`.
    #[test]
    fn prebuilt_empty_array_is_memoized_per_item_lltype() {
        let signed_items = LowLevelType::Array(Box::new(lltype::Array::gc(LowLevelType::Signed)));
        let char_items = LowLevelType::Array(Box::new(lltype::Array::gc(LowLevelType::Char)));
        let bookkeeper = Bookkeeper::new();
        let a = _ll_prebuilt_empty_array(&bookkeeper, &signed_items).expect("prebuilt signed");
        let b =
            _ll_prebuilt_empty_array(&bookkeeper, &signed_items).expect("prebuilt signed again");
        let c = _ll_new_empty_item_array(&bookkeeper, &signed_items).expect("new empty signed");
        let d = _ll_prebuilt_empty_array(&bookkeeper, &char_items).expect("prebuilt char");
        assert_eq!(empty_array_identity(&a), empty_array_identity(&b));
        assert_eq!(empty_array_identity(&a), empty_array_identity(&c));
        assert_ne!(empty_array_identity(&a), empty_array_identity(&d));
        assert_eq!(bookkeeper.all_specializations.borrow().len(), 1);
    }

    /// Memo is per translation (`specialize.py MemoTable` on the
    /// bookkeeper): two rtypers allocate two arrays; one rtyper reuses.
    #[test]
    fn prebuilt_empty_array_memo_is_per_translation() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::rtyper::RPythonTyper;

        let signed_items = LowLevelType::Array(Box::new(lltype::Array::gc(LowLevelType::Signed)));
        let ann1 = RPythonAnnotator::new(None, None, None, false);
        let rtyper1 = RPythonTyper::new(&ann1);
        let ann2 = RPythonAnnotator::new(None, None, None, false);
        let rtyper2 = RPythonTyper::new(&ann2);
        let bk1 = rtyper1
            .annotator
            .upgrade()
            .expect("rtyper1 annotator")
            .bookkeeper
            .clone();
        let bk2 = rtyper2
            .annotator
            .upgrade()
            .expect("rtyper2 annotator")
            .bookkeeper
            .clone();
        let a = _ll_prebuilt_empty_array(&bk1, &signed_items).expect("rtyper1 first");
        let b = _ll_prebuilt_empty_array(&bk1, &signed_items).expect("rtyper1 reuse");
        let c = _ll_prebuilt_empty_array(&bk2, &signed_items).expect("rtyper2");
        assert_eq!(empty_array_identity(&a), empty_array_identity(&b));
        assert_ne!(empty_array_identity(&a), empty_array_identity(&c));
        assert_eq!(bk1.all_specializations.borrow().len(), 1);
        assert_eq!(bk2.all_specializations.borrow().len(), 1);
    }
}
