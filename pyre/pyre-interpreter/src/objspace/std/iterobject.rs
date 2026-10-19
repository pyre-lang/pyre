//! `pypy/objspace/std/iterobject.py` sequence iterator declarations.

use indexmap::IndexMap;
use pyre_object::gc_roots::RootedItems;
use pyre_object::typedef::{TypeDef, TypeDefValue};

type SeqIterMethod = (
    &'static str,
    crate::gateway::BuiltinCodeFn,
    u16,
    &'static str,
);

/// Pin each interp2app as it is born. `TypeDefValue::root` copies the pointer
/// into an untracked cell, so the house pin (`RootedItems`) must outlive the
/// later `TypeDef.from_rawdict` that publishes those cells.
unsafe fn insert_rooted_gateways(
    entries: &mut IndexMap<String, TypeDefValue>,
    items: &mut RootedItems,
    methods: &[SeqIterMethod],
) {
    let base = items.len();
    for &(name, func, arity, text_sig) in methods {
        let code = crate::gateway::builtin_code_new_with_arity(name, func, arity);
        let gateway = crate::gateway::interp2app(code);
        unsafe {
            (*(gateway as *mut crate::gateway::interp2app))._explicit_text_sig = Some(text_sig);
        }
        items.push(gateway);
    }
    for (offset, &(name, _, _, _)) in methods.iter().enumerate() {
        unsafe {
            entries.insert(name.into(), TypeDefValue::root(items.get(base + offset)));
        }
    }
}

/// Fresh W_AbstractSeqIterObject.typedef declarations. Each call allocates
/// new interp2app gateways, pinned in `items` until the caller publishes or
/// spacebinds them. Sequence iterator types that share this declaration list
/// must call this per type so each namespace owns its own bound methods.
pub(crate) unsafe fn rawdict(items: &mut RootedItems) -> IndexMap<String, TypeDefValue> {
    let mut entries = IndexMap::new();
    // [3.14-spec] PySeqIter_Type's public name/doc are "iterator"/None,
    // unlike W_AbstractSeqIterObject.typedef's "sequenceiterator"/iter doc.
    // Measured on the pinned free-threaded CPython v3.14.6 with a
    // __getitem__-only class (PySeqIter_Type in Objects/iterobject.c).
    entries.insert("__doc__".into(), TypeDefValue::None);
    insert_rooted_gateways(
        &mut entries,
        items,
        &[
            (
                "__iter__",
                crate::baseobjspace::iter_self_method as crate::gateway::BuiltinCodeFn,
                1,
                "($self, /)",
            ),
            (
                "__next__",
                crate::baseobjspace::iter_next_method,
                1,
                "($self, /)",
            ),
            (
                "__reduce__",
                crate::baseobjspace::seq_iter_reduce_method,
                1,
                "($self, /)",
            ),
            (
                "__length_hint__",
                crate::baseobjspace::seq_iter_length_hint_method,
                1,
                "($self, /)",
            ),
            (
                "__setstate__",
                crate::baseobjspace::seq_iter_setstate_method,
                2,
                "($self, object, /)",
            ),
        ],
    );
    entries
}

/// Identity of W_AbstractSeqIterObject.typedef, not a W_TypeObject cache.
pub(crate) fn typedef() -> *const TypeDef {
    static TYPEDEF: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TYPEDEF.get_or_init(|| unsafe {
        let mut items = RootedItems::new();
        TypeDef::from_rawdict(
            "iterator",
            vec![],
            rawdict(&mut items),
            &pyre_object::iterobject::SEQ_ITER_TYPE,
        ) as usize
    }) as *const TypeDef
}

/// W_ReverseSeqIterObject.typedef owns an independent declaration, unlike
/// the forward list/tuple implementations sharing W_AbstractSeqIterObject.
pub(crate) fn reverse_typedef() -> *const TypeDef {
    static TYPEDEF: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TYPEDEF.get_or_init(|| unsafe {
        let mut items = RootedItems::new();
        let mut rawdict = IndexMap::new();
        insert_rooted_gateways(
            &mut rawdict,
            &mut items,
            &[
                (
                    "__iter__",
                    crate::baseobjspace::iter_self_method as crate::gateway::BuiltinCodeFn,
                    1,
                    "($self, /)",
                ),
                (
                    "__next__",
                    crate::baseobjspace::iter_next_method,
                    1,
                    "($self, /)",
                ),
                (
                    "__reduce__",
                    crate::baseobjspace::list_reverse_iter_reduce_method,
                    1,
                    "($self, /)",
                ),
                (
                    "__setstate__",
                    crate::baseobjspace::list_reverse_iter_setstate_method,
                    2,
                    "($self, object, /)",
                ),
                (
                    "__length_hint__",
                    crate::baseobjspace::list_reverse_iter_length_hint_method,
                    1,
                    "($self, /)",
                ),
            ],
        );
        // [3.14-spec] PyListRevIter_Type.tp_name (Objects/listobject.c,
        // v3.14.6) is "list_reverseiterator", measured on python3.14t.
        // PyPy W_ReverseSeqIterObject.typedef calls it
        // "reversesequenceiterator". Preserve the existing public spelling;
        // the declaration and its cache identity follow the PyPy owner.
        TypeDef::from_rawdict(
            "list_reverseiterator",
            vec![],
            rawdict,
            &pyre_object::iterobject::LIST_REVERSE_ITER_TYPE,
        ) as usize
    }) as *const TypeDef
}
