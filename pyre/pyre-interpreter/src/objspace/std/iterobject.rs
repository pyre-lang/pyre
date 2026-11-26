//! `pypy/objspace/std/iterobject.py` sequence iterator payload and declarations.

use indexmap::IndexMap;
use pyre_object::gc_roots::RootedItems;
use pyre_object::typedef::{TypeDef, TypeDefValue};
use pyre_object::{PyObject, PyObjectRef, PyType};

/// `bytesobject.py W_BytesObject._iter_getitem_result` /
/// `bytearrayobject.py W_BytearrayObject._iter_getitem_result`:
/// `(seq, index) → item`, IndexError past the end.
pub type StringIterGetitemFn = fn(PyObjectRef, isize) -> crate::PyResult;

/// `iterobject.py W_StringIterObject`.  Own GC payload (`seq`, `index`,
/// `getitem_fn`); 3.15-visible as `bytes_iterator`.  `bytearray_iterator`
/// shares this layout the way `str_ascii_iterator` shares
/// `W_FastUnicodeIterObject`.
#[crate::pyre_class("bytes_iterator", static_name = "BYTES_ITER")]
pub struct W_StringIterObject {
    pub seq: PyObjectRef,
    /// `W_AbstractSeqIterObject.index`: an RPython `Signed` (machine word).
    pub index: isize,
    /// `iterobject.py W_StringIterObject.getitem_fn`.
    pub getitem_fn: StringIterGetitemFn,
}

// [3.15-spec] producer-specific iterator identities ↔
// `iterobject.py W_AbstractSeqIterObject.typedef` (`sequenceiterator`);
// evidence: lib-python/3/test/test_array.py MiscTest.test_disallow_instantiation;
// pyre/bench/synth/get_iter_str_bytes_array_subclass.py print("5");
// v3.15.0 `PyBytesIter_Type.tp_name` `"bytes_iterator"`,
// `PyByteArrayIter_Type.tp_name` `"bytearray_iterator"`,
// `_PyMemoryIter_Type.tp_name` `"memory_iterator"`,
// `arrayiter_spec.name` `"array.arrayiterator"`.
// `bytearray_iterator` shares `W_StringIterObject`'s payload and GC type id,
// and has its own `PyType` for its Python-visible name.
// Str uses [`W_FastUnicodeIterObject`].  Bytes/bytearray use
// [`W_StringIterObject`].
pub static BYTEARRAY_ITER_TYPE: PyType = pyre_object::new_pytype("bytearray_iterator");

/// `stringmethods.py StringMethods.descr_iter` → `W_StringIterObject`,
/// 3.15-visible as `bytes_iterator`.
pub fn w_bytes_iter_new(seq: PyObjectRef, getitem_fn: StringIterGetitemFn) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let seq = pyre_object::gc_roots::pin_root(seq);
    W_StringIterObject::allocate_stable(W_StringIterObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        seq,
        index: 0,
        getitem_fn,
    })
}

/// `stringmethods.py StringMethods.descr_iter` → `W_StringIterObject`,
/// 3.15-visible as `bytearray_iterator`.
pub fn w_bytearray_iter_new(seq: PyObjectRef, getitem_fn: StringIterGetitemFn) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let seq = pyre_object::gc_roots::pin_root(seq);
    let value = W_StringIterObject {
        ob: PyObject {
            ob_type: &BYTEARRAY_ITER_TYPE as *const PyType,
            w_class: pyre_object::get_instantiate(&BYTEARRAY_ITER_TYPE),
        },
        seq,
        index: 0,
        getitem_fn,
    };
    pyre_object::lltype::malloc_typed_stable(value) as PyObjectRef
}

/// `bytes_iterator` / `bytearray_iterator` — `iterobject.py
/// W_StringIterObject`.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_string_iter(obj: PyObjectRef) -> bool {
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return false;
    }
    if obj.is_null() {
        return false;
    }
    let tp = unsafe { (*obj).ob_type };
    std::ptr::eq(tp, &BYTES_ITER_TYPE) || std::ptr::eq(tp, &BYTEARRAY_ITER_TYPE)
}

/// `bytearray_iterator` — the one producer-specific identity
/// that carries exhaustion in the cursor: `__setstate__` turns a negative state
/// into the -1 exhausted sentinel instead of rewinding to the front, and no
/// later `__setstate__` revives it.  Its immutable siblings (`str`, `bytes`)
/// and `arrayiterator` clamp a negative state to zero.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_bytearray_iter(obj: PyObjectRef) -> bool {
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return false;
    }
    !obj.is_null() && std::ptr::eq((*obj).ob_type, &BYTEARRAY_ITER_TYPE)
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_string_iter_seq(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_StringIterObject)).seq
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_string_iter_index(obj: PyObjectRef) -> isize {
    (*(obj as *const W_StringIterObject)).index
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_string_iter_getitem_fn(obj: PyObjectRef) -> StringIterGetitemFn {
    (*(obj as *const W_StringIterObject)).getitem_fn
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_string_iter_set_seq(obj: PyObjectRef, seq: PyObjectRef) {
    (*(obj as *mut W_StringIterObject)).seq = seq;
    pyre_object::gc_hook::try_gc_write_barrier(obj as pyre_object::gc_hook::GCREF);
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_string_iter_set_index(obj: PyObjectRef, index: isize) {
    (*(obj as *mut W_StringIterObject)).index = index;
}

#[cfg(test)]
mod string_iter_tests {
    use super::*;

    fn stub_string_iter_getitem(_: PyObjectRef, _: isize) -> crate::PyResult {
        unimplemented!("constructor tests do not call getitem_fn")
    }

    #[test]
    fn string_iter_constructors_stamp_their_own_constant_type() {
        let seq = std::ptr::null_mut();
        let bytes = w_bytes_iter_new(seq, stub_string_iter_getitem);
        assert!(
            std::ptr::eq(unsafe { (*bytes).ob_type }, &BYTES_ITER_TYPE),
            "bytes_iterator constructor stamped a non-constant or wrong type"
        );
        let ba = w_bytearray_iter_new(seq, stub_string_iter_getitem);
        assert!(
            std::ptr::eq(unsafe { (*ba).ob_type }, &BYTEARRAY_ITER_TYPE),
            "bytearray_iterator constructor stamped a non-constant or wrong type"
        );
        for obj in [bytes, ba] {
            assert!(unsafe { is_string_iter(obj) });
            assert!(!unsafe { pyre_object::is_seq_iter(obj) });
            assert!(!unsafe { pyre_object::is_str_iter(obj) });
            assert_eq!(unsafe { w_string_iter_index(obj) }, 0);
        }
        assert!(unsafe { is_bytearray_iter(ba) });
        assert!(!unsafe { is_bytearray_iter(bytes) });
    }
}

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
/// `next_fn` is this type's own `__next__` gateway (one per exact iterator
/// type so a wrong-receiver call typechecks).
pub(crate) unsafe fn rawdict(
    items: &mut RootedItems,
    next_fn: crate::gateway::BuiltinCodeFn,
) -> IndexMap<String, TypeDefValue> {
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
            ("__next__", next_fn, 1, "($self, /)"),
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
            rawdict(&mut items, crate::baseobjspace::seq_iter_next_method),
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
                    crate::baseobjspace::reverse_seq_iter_next_method,
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
