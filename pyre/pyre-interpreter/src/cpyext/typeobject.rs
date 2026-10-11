//! C-defined types -- PyPy `cpyext/typeobject.py`.
//!
//! A `PyTypeObject` is the extension's own static storage, so it *is* the type
//! mirror: `PyType_Ready` links it to the interpreter type it builds rather
//! than allocating a block of its own.  A type pyre defines gets a synthesized
//! block of the same shape, so `Py_TYPE(x)->tp_name` reads something either
//! way.

use super::pyobject::{self, CPyObject, REFCNT_FROM_PYPY, REFCNT_IMMORTAL};
use pyre_object::{PY_NULL, PyObjectRef};
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};

/// `PyVarObject` — a `PyObject` with the variable-part length.
#[repr(C)]
pub struct CPyVarObject {
    pub ob_base: CPyObject,
    pub ob_size: isize,
}

/// `PyNumberMethods`.  Every slot is a raw pointer for the reason
/// [`CPyTypeObject`] gives.
#[repr(C)]
pub struct CPyNumberMethods {
    pub nb_add: *const c_void,
    pub nb_subtract: *const c_void,
    pub nb_multiply: *const c_void,
    pub nb_remainder: *const c_void,
    pub nb_divmod: *const c_void,
    pub nb_power: *const c_void,
    pub nb_negative: *const c_void,
    pub nb_positive: *const c_void,
    pub nb_absolute: *const c_void,
    pub nb_bool: *const c_void,
    pub nb_invert: *const c_void,
    pub nb_lshift: *const c_void,
    pub nb_rshift: *const c_void,
    pub nb_and: *const c_void,
    pub nb_xor: *const c_void,
    pub nb_or: *const c_void,
    pub nb_int: *const c_void,
    pub nb_reserved: *const c_void,
    pub nb_float: *const c_void,
    pub nb_inplace_add: *const c_void,
    pub nb_inplace_subtract: *const c_void,
    pub nb_inplace_multiply: *const c_void,
    pub nb_inplace_remainder: *const c_void,
    pub nb_inplace_power: *const c_void,
    pub nb_inplace_lshift: *const c_void,
    pub nb_inplace_rshift: *const c_void,
    pub nb_inplace_and: *const c_void,
    pub nb_inplace_xor: *const c_void,
    pub nb_inplace_or: *const c_void,
    pub nb_floor_divide: *const c_void,
    pub nb_true_divide: *const c_void,
    pub nb_inplace_floor_divide: *const c_void,
    pub nb_inplace_true_divide: *const c_void,
    pub nb_index: *const c_void,
    pub nb_matrix_multiply: *const c_void,
    pub nb_inplace_matrix_multiply: *const c_void,
}

/// `PySequenceMethods`.  The two `was_sq_*` holes are the pre-2.x slice slots,
/// which still occupy their offsets.
#[repr(C)]
pub struct CPySequenceMethods {
    pub sq_length: *const c_void,
    pub sq_concat: *const c_void,
    pub sq_repeat: *const c_void,
    pub sq_item: *const c_void,
    pub was_sq_slice: *const c_void,
    pub sq_ass_item: *const c_void,
    pub was_sq_ass_slice: *const c_void,
    pub sq_contains: *const c_void,
    pub sq_inplace_concat: *const c_void,
    pub sq_inplace_repeat: *const c_void,
}

/// `PyMappingMethods`.
#[repr(C)]
pub struct CPyMappingMethods {
    pub mp_length: *const c_void,
    pub mp_subscript: *const c_void,
    pub mp_ass_subscript: *const c_void,
}

/// `PyAsyncMethods`, declared for its offsets: nothing reads these yet.
#[repr(C)]
pub struct CPyAsyncMethods {
    pub am_await: *const c_void,
    pub am_aiter: *const c_void,
    pub am_anext: *const c_void,
    pub am_send: *const c_void,
}

/// `PyBufferProcs`.
#[repr(C)]
pub struct CPyBufferProcs {
    pub bf_getbuffer: *const c_void,
    pub bf_releasebuffer: *const c_void,
}

/// One row of a `PyMemberDef` table.
#[repr(C)]
pub struct CPyMemberDef {
    pub name: *const c_char,
    pub type_code: c_int,
    pub offset: isize,
    pub flags: MemberFlags,
    pub doc: *const c_char,
}

/// One row of a `PyGetSetDef` table.
#[repr(C)]
pub struct CPyGetSetDef {
    pub name: *const c_char,
    pub get: *const c_void,
    pub set: *const c_void,
    pub doc: *const c_char,
    pub closure: *mut c_void,
}

/// `PyDescrObject` — what every descriptor block opens with.
#[repr(C)]
pub struct CPyDescrObject {
    pub ob_base: CPyObject,
    pub d_type: *mut CPyTypeObject,
    pub d_name: *mut CPyObject,
    pub d_qualname: *mut CPyObject,
}

/// `PyMethodDescrObject`.
///
/// `PyMemberDescrObject` and `PyGetSetDescrObject` are this same block: the
/// one word past the common header is the row the descriptor was built from,
/// and only its declared type differs.
#[repr(C)]
pub struct CPyMethodDescrObject {
    pub d_common: CPyDescrObject,
    pub d_method: *mut super::methodobject::CPyMethodDef,
}

/// `struct wrapperbase` — what a slot wrapper carries beside the slot.
#[repr(C)]
pub struct CPyWrapperBase {
    pub name: *const c_char,
    pub offset: c_int,
    pub function: *mut c_void,
    pub wrapper: *mut c_void,
    pub doc: *const c_char,
    pub flags: c_int,
    pub name_strobj: *mut CPyObject,
}

/// `PyWrapperDescrObject`.
#[repr(C)]
pub struct CPyWrapperDescrObject {
    pub d_common: CPyDescrObject,
    pub d_base: *mut CPyWrapperBase,
    pub d_wrapped: *mut c_void,
}

impl CPyWrapperBase {
    /// The all-zero block `wrapperdescr_attach` hands out.
    const fn empty() -> Self {
        Self {
            name: std::ptr::null(),
            offset: 0,
            function: std::ptr::null_mut(),
            wrapper: std::ptr::null_mut(),
            doc: std::ptr::null(),
            flags: 0,
            name_strobj: std::ptr::null_mut(),
        }
    }
}

/// `PyTypeObject`, in the CPython 3.14 field order.
///
/// Every slot is a raw pointer here: the ones this slice reads are typed at
/// their use site, and the ones it does not read still have to occupy their
/// exact offsets, because the extension's static initializer writes them all.
#[repr(C)]
pub struct CPyTypeObject {
    pub ob_base: CPyVarObject,
    pub tp_name: *const c_char,
    pub tp_basicsize: isize,
    pub tp_itemsize: isize,
    pub tp_dealloc: *const c_void,
    pub tp_vectorcall_offset: isize,
    pub tp_getattr: *const c_void,
    pub tp_setattr: *const c_void,
    pub tp_as_async: *mut CPyAsyncMethods,
    pub tp_repr: *const c_void,
    pub tp_as_number: *mut CPyNumberMethods,
    pub tp_as_sequence: *mut CPySequenceMethods,
    pub tp_as_mapping: *mut CPyMappingMethods,
    pub tp_hash: *const c_void,
    pub tp_call: *const c_void,
    pub tp_str: *const c_void,
    pub tp_getattro: *const c_void,
    pub tp_setattro: *const c_void,
    pub tp_as_buffer: *mut CPyBufferProcs,
    pub tp_flags: TpFlags,
    pub tp_doc: *const c_char,
    pub tp_traverse: *const c_void,
    pub tp_clear: *const c_void,
    pub tp_richcompare: *const c_void,
    pub tp_weaklistoffset: isize,
    pub tp_iter: *const c_void,
    pub tp_iternext: *const c_void,
    pub tp_methods: *mut super::methodobject::CPyMethodDef,
    pub tp_members: *mut CPyMemberDef,
    pub tp_getset: *mut CPyGetSetDef,
    pub tp_base: *mut CPyTypeObject,
    pub tp_dict: *mut CPyObject,
    pub tp_descr_get: *const c_void,
    pub tp_descr_set: *const c_void,
    pub tp_dictoffset: isize,
    pub tp_init: *const c_void,
    pub tp_alloc: *const c_void,
    pub tp_new: *const c_void,
    pub tp_free: *const c_void,
    pub tp_is_gc: *const c_void,
    pub tp_bases: *mut CPyObject,
    pub tp_mro: *mut CPyObject,
    pub tp_cache: *mut CPyObject,
    pub tp_subclasses: *mut c_void,
    pub tp_weaklist: *mut CPyObject,
    pub tp_del: *const c_void,
    pub tp_version_tag: c_uint,
    pub tp_finalize: *const c_void,
    pub tp_vectorcall: *const c_void,
    pub tp_watched: std::ffi::c_uchar,
    pub tp_versions_used: u16,
}

/// C-visible `PyHeapTypeObject`, the twin of the struct in
/// `include/pyre3.14t/structmember.h`.
///
/// This is the shape every type mirror is allocated at, which is what
/// `tp_basicsize` answers for `type` and the metaclasses derived from it.  The
/// suites are the storage a type's own `tp_as_*` may name, and the references
/// past them are a heap type's `__name__`, `__slots__`, `__qualname__` and
/// `__module__`.
#[repr(C)]
pub struct CPyHeapTypeObject {
    pub ht_type: CPyTypeObject,
    pub as_async: CPyAsyncMethods,
    pub as_number: CPyNumberMethods,
    pub as_mapping: CPyMappingMethods,
    pub as_sequence: CPySequenceMethods,
    pub as_buffer: CPyBufferProcs,
    pub ht_name: *mut CPyObject,
    pub ht_slots: *mut CPyObject,
    pub ht_qualname: *mut CPyObject,
    pub ht_module: *mut CPyObject,
}

/// The `tp_flags` word, declared with the rest of the type object in
/// `pyre_object::typeobject` and compared against the header there.
pub use pyre_object::typeobject::TpFlags;

/// The fast-subclass flags, in the order `inherit_special` tests them
/// (`typeobject.py:492-509`): the first base that matches wins, so a type is
/// only ever marked as one of these.
const FAST_SUBCLASS_FLAGS: [(&pyre_object::pyobject::PyType, TpFlags); 8] = [
    (
        &pyre_object::interp_exceptions::EXCEPTION_TYPE,
        TpFlags::PY_TPFLAGS_BASE_EXC_SUBCLASS,
    ),
    (
        &pyre_object::pyobject::TYPE_TYPE,
        TpFlags::PY_TPFLAGS_TYPE_SUBCLASS,
    ),
    (
        &pyre_object::pyobject::INT_TYPE,
        TpFlags::PY_TPFLAGS_LONG_SUBCLASS,
    ),
    (
        &pyre_object::bytesobject::BYTES_TYPE,
        TpFlags::PY_TPFLAGS_BYTES_SUBCLASS,
    ),
    (
        &pyre_object::pyobject::STR_TYPE,
        TpFlags::PY_TPFLAGS_UNICODE_SUBCLASS,
    ),
    (
        &pyre_object::pyobject::TUPLE_TYPE,
        TpFlags::PY_TPFLAGS_TUPLE_SUBCLASS,
    ),
    (
        &pyre_object::pyobject::LIST_TYPE,
        TpFlags::PY_TPFLAGS_LIST_SUBCLASS,
    ),
    (
        &pyre_object::pyobject::DICT_TYPE,
        TpFlags::PY_TPFLAGS_DICT_SUBCLASS,
    ),
];

/// Mark `tp` with the one fast-subclass flag its base chain earns it.
fn set_fast_subclass_flags(tp: *mut CPyTypeObject, w_type: PyObjectRef) {
    for (builtin, flag) in FAST_SUBCLASS_FLAGS {
        let w_builtin = crate::typedef::gettypeobject(builtin);
        if w_builtin.is_null() {
            continue;
        }
        if crate::baseobjspace::issubclass(w_type, w_builtin).unwrap_or(false) {
            unsafe { (*tp).tp_flags |= flag };
            return;
        }
    }
}

const fn immortal_type() -> CPyTypeObject {
    CPyTypeObject {
        ob_base: CPyVarObject {
            ob_base: CPyObject {
                ob_refcnt: REFCNT_IMMORTAL,
                ob_pyre_link: pyre_object::PY_NULL,
                ob_pyre_pad: 0,
                ob_type: std::ptr::null_mut(),
            },
            ob_size: 0,
        },
        tp_name: std::ptr::null(),
        tp_basicsize: 0,
        tp_itemsize: 0,
        tp_dealloc: std::ptr::null(),
        tp_vectorcall_offset: 0,
        tp_getattr: std::ptr::null(),
        tp_setattr: std::ptr::null(),
        tp_as_async: std::ptr::null_mut(),
        tp_repr: std::ptr::null(),
        tp_as_number: std::ptr::null_mut(),
        tp_as_sequence: std::ptr::null_mut(),
        tp_as_mapping: std::ptr::null_mut(),
        tp_hash: std::ptr::null(),
        tp_call: std::ptr::null(),
        tp_str: std::ptr::null(),
        tp_getattro: std::ptr::null(),
        tp_setattro: std::ptr::null(),
        tp_as_buffer: std::ptr::null_mut(),
        tp_flags: TpFlags::PY_TPFLAGS_DEFAULT,
        tp_doc: std::ptr::null(),
        tp_traverse: std::ptr::null(),
        tp_clear: std::ptr::null(),
        tp_richcompare: std::ptr::null(),
        tp_weaklistoffset: 0,
        tp_iter: std::ptr::null(),
        tp_iternext: std::ptr::null(),
        tp_methods: std::ptr::null_mut(),
        tp_members: std::ptr::null_mut(),
        tp_getset: std::ptr::null_mut(),
        tp_base: std::ptr::null_mut(),
        tp_dict: std::ptr::null_mut(),
        tp_descr_get: std::ptr::null(),
        tp_descr_set: std::ptr::null(),
        tp_dictoffset: 0,
        tp_init: std::ptr::null(),
        tp_alloc: std::ptr::null(),
        tp_new: std::ptr::null(),
        tp_free: std::ptr::null(),
        tp_is_gc: std::ptr::null(),
        tp_bases: std::ptr::null_mut(),
        tp_mro: std::ptr::null_mut(),
        tp_cache: std::ptr::null_mut(),
        tp_subclasses: std::ptr::null_mut(),
        tp_weaklist: std::ptr::null_mut(),
        tp_del: std::ptr::null(),
        tp_version_tag: 0,
        tp_finalize: std::ptr::null(),
        tp_vectorcall: std::ptr::null(),
        tp_watched: 0,
        tp_versions_used: 0,
    }
}

/// The raw block `type_alloc` hands to `type_realize` upstream.
///
/// PyPy's `PyType_FromModuleAndSpec` returns that already-owned raw block
/// directly (`result_is_ll=True`).  Pyre's [`from_spec`] instead returns the
/// attached interpreter object through `object::result`, whose `make_ref`
/// supplies the one C reference at the API boundary.  Start the wrapper-owned
/// count at zero so that `attach_foreign_pyobj`'s rawrefcount share followed by
/// that `make_ref` lands on `REFCNT_FROM_PYPY + 1`, exactly as upstream does.
const fn allocated_type() -> CPyTypeObject {
    let mut tp = immortal_type();
    tp.ob_base.ob_base.ob_refcnt = 0;
    tp
}

/// Sentinel type for a `PyModuleDef`, which is C static storage rather than a
/// mirror of an interpreter object: its `ob_pyre_link` stays null and it is
/// never entered in the census.
pub static mut CPY_MODULE_DEF_TYPE: CPyTypeObject = immortal_type();

// ── the static type mirrors ─────────────────────────────────────────────

/// The `PyTypeObject` statics an extension names by address.
///
/// `api.py build_exported_objects` registers the same family: C spells
/// `&PyList_Type`, so each one has to be storage whose address is fixed at link
/// time and whose body the runtime fills in place, which is what a
/// [`PyAPI_DATA`] object is and what a mirror allocated on demand can never be.
/// A name whose type this build does not have stays unbound -- its `tp_name` is
/// then NULL, which is the state `PyType_Ready` would have left it in.
macro_rules! type_mirrors {
    ($($symbol:ident => $resolve:expr,)*) => {
        $(
            /// The mirror of the like-named builtin type.
            #[unsafe(no_mangle)]
            pub static mut $symbol: CPyTypeObject = immortal_type();
        )*

        /// Bind every static type mirror to the interpreter type it stands for.
        ///
        /// Called before the first `PyInit_*`, so a type reached through
        /// `make_ref` finds the static already entered and hands out its
        /// address rather than synthesizing a second block for the same type.
        pub fn init_type_mirrors() {
            let bound: &[(*mut CPyTypeObject, PyObjectRef)] = &[
                $( (&raw mut $symbol, $resolve), )*
            ];
            // `api.py attach_all` links every static before any of
            // them is filled: filling one resolves its base's mirror, and a
            // base that is about to be bound a few entries later would
            // otherwise have a second block synthesized for it.
            for &(mirror, w_type) in bound {
                let header = unsafe { &raw mut (*mirror).ob_base.ob_base };
                if w_type.is_null() || unsafe { !(*header).ob_pyre_link.is_null() } {
                    continue;
                }
                pyobject::attach_foreign(w_type, header);
            }
            for &(mirror, w_type) in bound {
                if w_type.is_null() || unsafe { !(*mirror).tp_name.is_null() } {
                    continue;
                }
                describe_interpreter_type(mirror, w_type);
            }
            // Deferred until every link above is in the table: a metatype is
            // itself one of these statics, and resolving it earlier would
            // synthesize a block for a type that is about to be bound here.
            for &(mirror, _) in bound {
                let header = unsafe { &raw mut (*mirror).ob_base.ob_base };
                let w_type = unsafe { (*header).ob_pyre_link };
                if w_type.is_null() || unsafe { !(*header).ob_type.is_null() } {
                    continue;
                }
                let of_type = pyobject::type_mirror(w_type);
                unsafe { pyobject::set_ob_type(header, of_type) };
            }
            // `api.py:1509-1513`, the drain of what `type_attach` deferred.
            for &(mirror, w_type) in bound {
                if w_type.is_null() {
                    continue;
                }
                finish_interpreter_type(mirror, w_type);
            }
        }

        fn ensure_type_mirrors_linked() {
            $( std::hint::black_box(&raw const $symbol); )*
        }
    };
}

type_mirrors! {
    // The object model.
    PyType_Type => crate::typedef::w_type(),
    PyBaseObject_Type => crate::typedef::w_object(),
    PySuper_Type => builtin_type(&pyre_object::descriptor::SUPER_TYPE),
    // The built-in data types.
    PyBool_Type => builtin_type(&pyre_object::pyobject::BOOL_TYPE),
    PyByteArray_Type => builtin_type(&pyre_object::bytearrayobject::BYTEARRAY_TYPE),
    PyBytes_Type => builtin_type(&pyre_object::bytesobject::BYTES_TYPE),
    PyComplex_Type => builtin_type(&pyre_object::pyobject::COMPLEX_TYPE),
    PyDict_Type => builtin_type(&pyre_object::pyobject::DICT_TYPE),
    PyEllipsis_Type => builtin_type(&pyre_object::pyobject::ELLIPSIS_TYPE),
    PyFloat_Type => builtin_type(&pyre_object::pyobject::FLOAT_TYPE),
    PyFrozenSet_Type => builtin_type(&pyre_object::setobject::FROZENSET_TYPE),
    PyList_Type => builtin_type(&pyre_object::pyobject::LIST_TYPE),
    PyLong_Type => builtin_type(&pyre_object::pyobject::INT_TYPE),
    PyMemoryView_Type => builtin_type(&pyre_object::memoryview::MEMORYVIEW_TYPE),
    PyModule_Type => builtin_type(&pyre_object::pyobject::MODULE_TYPE),
    PySet_Type => builtin_type(&pyre_object::setobject::SET_TYPE),
    PySlice_Type => builtin_type(&pyre_object::sliceobject::SLICE_TYPE),
    PyTuple_Type => builtin_type(&pyre_object::pyobject::TUPLE_TYPE),
    PyUnicode_Type => builtin_type(&pyre_object::pyobject::STR_TYPE),
    Py_GenericAliasType => builtin_type(&pyre_object::_pypy_generic_alias::GENERIC_ALIAS_TYPE),
    // The dict views.
    PyDictProxy_Type => builtin_type(&pyre_object::pyobject::MAPPING_PROXY_TYPE),
    PyDictItems_Type => builtin_type(&pyre_object::dictmultiobject::DICT_ITEMS_TYPE),
    PyDictKeys_Type => builtin_type(&pyre_object::dictmultiobject::DICT_KEYS_TYPE),
    PyDictValues_Type => builtin_type(&pyre_object::dictmultiobject::DICT_VALUES_TYPE),
    // Functions, methods and descriptors.  `PyCFunction_Type` names
    // `methodobject`'s own `builtin_function_or_method` -- the type a method
    // an extension defines carries, and so the one a type derived from it is
    // derived from.  The interpreter's `len` is the other
    // `builtin_function_or_method`, which no symbol here names, and
    // `PyCFunction_Check` answers no for it; `methodobject`'s
    // `pycfunction_type` says why that is the safe half of the gap.
    //
    // `PyMethodDescr_Type` and `PyClassMethodDescr_Type` are the same split
    // and the same half: a descriptor the interpreter built for one of its
    // own types carries no `PyMethodDef`, and reading one off the block is
    // what every caller that asks does next.
    PyCFunction_Type => super::methodobject::pycfunction_type(),
    PyCMethod_Type => super::methodobject::pycmethod_type(),
    PyClassMethodDescr_Type => classmethod_descriptor_type(),
    PyClassMethod_Type => builtin_type(&pyre_object::function::CLASSMETHOD_TYPE),
    PyFunction_Type => builtin_type(&crate::function::FUNCTION_TYPE),
    PyGetSetDescr_Type => builtin_type(&pyre_object::typedef::GETSET_DESCRIPTOR_TYPE),
    PyMemberDescr_Type => builtin_type(&pyre_object::typedef::MEMBER_TYPE),
    PyMethodDescr_Type => method_descriptor_type(),
    PyInstanceMethod_Type => builtin_type(&pyre_object::instancemethod::INSTANCEMETHOD_TYPE),
    PyMethod_Type => builtin_type(&pyre_object::function::METHOD_TYPE),
    PyProperty_Type => builtin_type(&pyre_object::descriptor::PROPERTY_TYPE),
    PyStaticMethod_Type => builtin_type(&pyre_object::function::STATICMETHOD_TYPE),
    PyWrapperDescr_Type => builtin_type(&crate::function::SLOT_WRAPPER_TYPE),
    // The built-ins that are types.
    PyEnum_Type => builtin_type(&pyre_object::functional::ENUMERATE_TYPE),
    PyFilter_Type => builtin_type(&pyre_object::functional::FILTER_TYPE),
    PyMap_Type => builtin_type(&pyre_object::functional::MAP_TYPE),
    PyRange_Type => builtin_type(&pyre_object::functional::RANGE_TYPE),
    PyReversed_Type => builtin_type(&pyre_object::functional::REVERSED_TYPE),
    PyZip_Type => builtin_type(&pyre_object::functional::ZIP_TYPE),
    // Frames, code and the objects a call leaves behind.
    PyAsyncGen_Type => builtin_type(&pyre_object::generator::ASYNC_GENERATOR_TYPE),
    PyCell_Type => builtin_type(&pyre_object::nestedscope::CELL_TYPE),
    PyCode_Type => builtin_type(&crate::pycode::CODE_TYPE),
    PyCoro_Type => builtin_type(&pyre_object::generator::COROUTINE_TYPE),
    PyFrame_Type => builtin_type(&crate::pyframe::FRAME_TYPE),
    PyGen_Type => builtin_type(&pyre_object::generator::GENERATOR_TYPE),
    PyTraceBack_Type => builtin_type(&crate::pytraceback::PYTRACEBACK_TYPE),
    _PyAsyncGenASend_Type => builtin_type(&pyre_object::generator::ASYNC_GEN_ASEND_TYPE),
    _PyWeakref_RefType => builtin_type(&pyre_object::weakref::WEAKREF_LAYOUT_TYPE),
}

/// The interpreter type object a layout static describes, or null when this
/// build has not built it yet.
fn builtin_type(layout: &'static pyre_object::pyobject::PyType) -> PyObjectRef {
    crate::typedef::gettypeobject(layout)
}

/// The name a synthesized mirror hands out as `tp_name`.
///
/// `tp_name` is a `const char *` an extension may keep for as long as it holds
/// the type, so the string has to outlive every read of the field and die with
/// the mirror rather than with this call.  Keyed by the mirror's address, which
/// is fixed for its life; [`forget_type_name`] is what releases it.
type NameTable = super::address_table::HeldMap<CString>;
use super::address_table::{AddressTable, hold};

static TYPE_NAMES: AddressTable<NameTable> =
    AddressTable::new(NameTable::with_hasher(std::hash::BuildHasherDefault::new()));

/// `typeobject.py type_dealloc` — release what a synthesized mirror's
/// own fields hold, and the string behind `tp_name`.
///
/// The table above is what says a dying block is a type mirror this layer
/// filled: an entry is made by [`describe_interpreter_type`] and by nothing
/// else, so a block whose `tp_base` and `tp_dict` would be some other type's
/// fields is not read at those offsets.  A type an extension readied is
/// deliberately immortal and never reaches here.
///
/// # Safety
/// `raw` must be a live block whose count has fallen to zero.
pub(super) unsafe fn forget_type_mirror(raw: *mut CPyObject) {
    if TYPE_NAMES.take(raw as usize).is_none() {
        return;
    }
    let mirror = raw as *mut CPyTypeObject;
    unsafe {
        pyobject::decref((*mirror).tp_dict);
        (*mirror).tp_dict = std::ptr::null_mut();
        // `type_dealloc`'s `decref(obj_pto.c_tp_bases)` and
        // `decref(obj_pto.c_tp_mro)`, which the fill above owes.
        pyobject::decref((*mirror).tp_bases);
        (*mirror).tp_bases = std::ptr::null_mut();
        pyobject::decref((*mirror).tp_mro);
        (*mirror).tp_mro = std::ptr::null_mut();
        // `type_dealloc` releases the base only for a heap type, which is the
        // only kind whose base can be one too.
        let base = (*mirror).tp_base;
        if is_heap_type(mirror) && !base.is_null() {
            pyobject::decref(&raw mut (*base).ob_base.ob_base);
            (*mirror).tp_base = std::ptr::null_mut();
        }
    }
}

/// `type_traverse`'s `visit(tp_dict)`, `visit(tp_mro)`, `visit(tp_bases)` and
/// `visit(tp_base)`, read as [`super::gc::c_edges`] reads a `tp_traverse`.
///
/// A type's MRO names the type itself, so the tuple minted for `tp_mro` refers
/// back through the mirror holding it: the tuple's count roots the tuple, the
/// tuple roots the type, and the type's mirror is what would give the count
/// back.  Reporting the reference is what lets a collection tell it from one an
/// extension holds, and it is the reason `type_traverse` reports these fields
/// at all.
///
/// Only the fields [`forget_type_mirror`] releases are reported, so an edge is
/// exactly a reference this mirror's death gives back.
pub(super) fn type_mirror_edges(edges: &mut Vec<(usize, Vec<usize>)>) {
    let names = TYPE_NAMES.lock();
    edges.reserve(names.len());
    for &block in names.keys() {
        let mirror = block.address() as *mut CPyTypeObject;
        let base = unsafe { (*mirror).tp_base };
        let base = match is_heap_type(mirror) && !base.is_null() {
            true => unsafe { &raw mut (*base).ob_base.ob_base },
            false => std::ptr::null_mut(),
        };
        let referents: Vec<usize> = unsafe {
            [
                (*mirror).tp_dict,
                (*mirror).tp_bases,
                (*mirror).tp_mro,
                base,
            ]
        }
        .into_iter()
        .filter(|raw| !raw.is_null())
        .map(|raw| raw as usize)
        .collect();
        if !referents.is_empty() {
            edges.push((block.address(), referents));
        }
    }
}

/// Fill a synthesized mirror for an interpreter type.
///
/// `tp_basicsize` stays 0 for all but one type: an instance of a pyre type is
/// exactly a `PyObject` mirror, and `make_ref` reads this field to size the
/// block.  The frame is the exception, and says so itself
/// (`super::frameobject::basicsize`).
///
/// The refcount is left as [`pyobject::attach`] set it: a synthesized mirror
/// carries the ordinary link share and is released with the type it stands for,
/// which is what keeps a class the extension merely observed collectable.
/// What `tp_basicsize` a mirror of `type`, or of a metaclass derived from it,
/// carries: an instance of one is a type, and every type mirror is a
/// [`CPyHeapTypeObject`].  0 for every other type, which asks for the plain
/// header.
fn heap_type_basicsize(w_type: PyObjectRef) -> isize {
    let class = match crate::typedef::gettypefor(&pyre_object::pyobject::TYPE_TYPE) {
        Some(class) => class.as_ptr(),
        None => return 0,
    };
    let derived = !w_type.is_null() && unsafe { crate::baseobjspace::issubtype_w(w_type, class) };
    match derived {
        true => size_of::<CPyHeapTypeObject>() as isize,
        false => 0,
    }
}

/// What `tp_itemsize` a synthesized mirror of `w_type` carries --
/// `typeobject.py type_attach`.
///
/// It is the number a caller sizing an allocation of its own multiplies by,
/// not a statement about where the items are: a block this runtime hands out
/// carries none of them, and no header here spells a struct that could reach
/// past the fields it declares.
fn mirror_itemsize(w_type: PyObjectRef) -> isize {
    if w_type.is_null() {
        return 0;
    }
    let derived = |builtin: &'static pyre_object::pyobject::PyType| {
        let class = crate::typedef::gettypefor(builtin);
        class.is_some_and(|class| unsafe {
            crate::baseobjspace::issubtype_w(w_type, class.as_ptr())
        })
    };
    if derived(&pyre_object::bytesobject::BYTES_TYPE) {
        return 1;
    }
    if derived(&pyre_object::pyobject::TUPLE_TYPE) {
        // `rffi.sizeof(PyObject)`, where `PyObject` is the pointer type: one
        // item of a tuple is one `PyObject *`, which is what a caller sizing
        // its own allocation multiplies by.
        return size_of::<*mut CPyObject>() as isize;
    }
    // `type` itself and no metaclass: the members follow the block a heap type
    // declared, and only a type declared in C has any.
    let is_type = crate::typedef::gettypefor(&pyre_object::pyobject::TYPE_TYPE)
        .is_some_and(|class| std::ptr::eq(w_type, class.as_ptr()));
    match is_type {
        true => size_of::<CPyMemberDef>() as isize,
        false => 0,
    }
}

/// Whether a mirror of an instance of `w_type` carries an `ob_size` --
/// `pyobject.py allocate`'s `if itemsize or issubtype_w(w_type, w_list)`.
///
/// A list has no item size of its own and is counted all the same, because
/// `Py_SIZE` reads the length off one.
fn counts_items(w_type: PyObjectRef) -> bool {
    if mirror_itemsize(w_type) != 0 {
        return true;
    }
    crate::typedef::gettypefor(&pyre_object::pyobject::LIST_TYPE)
        .is_some_and(|class| unsafe { crate::baseobjspace::issubtype_w(w_type, class.as_ptr()) })
}

/// Fill a fresh mirror's `ob_size` -- `pyobject.py create_ref`'s item count.
///
/// The count comes off the object's own layout rather than from `len()`:
/// this runs before the mirror is linked, and a `__len__` written in Python
/// would come back through here for the same object.
///
/// # Safety
/// `raw` must be a live mirror of `w_obj` whose block is `size` bytes.
pub(super) unsafe fn stamp_ob_size(raw: *mut CPyObject, w_obj: PyObjectRef, size: usize) {
    if size < size_of::<CPyVarObject>() {
        return;
    }
    let Some(w_type) = crate::typedef::r#type(w_obj) else {
        return;
    };
    if !counts_items(w_type.as_ptr()) {
        return;
    }
    let count = unsafe {
        if pyre_object::is_list(w_obj) {
            pyre_object::listobject::w_list_len(w_obj)
        } else if pyre_object::is_tuple(w_obj) {
            pyre_object::tupleobject::w_tuple_len(w_obj)
        } else if pyre_object::bytesobject::is_bytes(w_obj) {
            pyre_object::bytesobject::w_bytes_len(w_obj)
        } else {
            0
        }
    };
    unsafe { (*(raw as *mut CPyVarObject)).ob_size = count as isize };
}

/// What `tp_basicsize` a synthesized mirror of `w_type` carries.
///
/// The modules whose mirrors have fields of their own each answer for their
/// own types and 0 for everything else.  A type that only has a declared
/// layout (the typedef table `cpython_type_layout` publishes) uses that size;
/// only a type with neither falls back to the header.  A ready type never
/// carries a zero here -- an extension reads the field to size an allocation,
/// and Cython's `__Pyx_ImportType` refuses to import a class whose basicsize
/// is under the struct it declared.
fn mirror_basicsize(w_type: PyObjectRef) -> isize {
    let size = [
        heap_type_basicsize(w_type),
        super::frameobject::basicsize(w_type),
        super::sliceobject::basicsize(w_type),
        super::pyerrors::basicsize(w_type),
        super::cdatetime::basicsize(w_type),
        super::complexobject::basicsize(w_type),
        super::methodobject::basicsize(w_type),
        super::structobject::basicsize(w_type),
        descriptor_basicsize(w_type),
    ]
    .into_iter()
    .find(|&size| size != 0)
    .or_else(|| {
        crate::typedef::cpython_type_layout(w_type).map(|(basicsize, _)| basicsize as isize)
    })
    .unwrap_or(size_of::<CPyObject>() as isize);
    // `type_attach`, "Make sure Py_SIZE() can cast to PyVarObject": a block
    // whose length is read has to have room for the word that holds it.
    match counts_items(w_type) {
        true => size.max(size_of::<CPyVarObject>() as isize),
        false => size,
    }
}

pub(super) fn describe_interpreter_type(mirror: *mut CPyTypeObject, w_type: PyObjectRef) {
    let name = unsafe { pyre_object::typeobject::w_type_get_name(w_type) };
    let name = CString::new(name).unwrap_or_default();
    // The bytes are boxed, so moving the `CString` into the table below leaves
    // this pointer valid.
    let pointer = name.as_ptr();
    let heaptype = match unsafe { pyre_object::w_type_is_cpython_heaptype(w_type) } {
        true => TpFlags::PY_TPFLAGS_HEAPTYPE,
        false => TpFlags::empty(),
    };
    let static_builtin = match unsafe { pyre_object::w_type_is_cpython_static_builtin(w_type) } {
        true => TpFlags::_PY_TPFLAGS_STATIC_BUILTIN,
        false => TpFlags::empty(),
    };
    let immutabletype = match unsafe { pyre_object::w_type_is_cpython_immutabletype(w_type) } {
        true => TpFlags::PY_TPFLAGS_IMMUTABLETYPE,
        false => TpFlags::empty(),
    };
    // `type_attach` publishes `tp_name` before `tp_basicsize`. The size is
    // the typedescr keyed by `w_type`: a named block whose size fields are
    // still zero is not a C declaration (`declared_type_layout`).
    unsafe {
        (*mirror).tp_name = pointer;
    }
    let basicsize = mirror_basicsize(w_type);
    unsafe {
        (*mirror).tp_basicsize = basicsize;
        (*mirror).tp_itemsize = mirror_itemsize(w_type);
        (*mirror).tp_flags = TpFlags::PY_TPFLAGS_DEFAULT
            | TpFlags::PY_TPFLAGS_READY
            | TpFlags::PY_TPFLAGS_BASETYPE
            | heaptype
            | static_builtin
            | immutabletype;
        // `typeobject.py type_attach`.  A `tp_new` written for C ends
        // in `t->tp_alloc(t, 0)`, with `t` the type being built rather than
        // the one that declared the constructor, so a class derived in Python
        // from a C type is handed to a C allocator and has to carry the pair.
        (*mirror).tp_alloc = PyType_GenericAlloc as *const c_void;
        (*mirror).tp_free = PyObject_Free as *const c_void;
        // `type_attach`'s `pto.c_tp_dealloc`, which it fills for every type it
        // builds.  A deallocator written for a type derived from a builtin
        // ends in its base's, and a null slot there is a call through address
        // zero rather than a field merely left unset.
        (*mirror).tp_dealloc = object_dealloc as *const c_void;
    }
    // `finish_type_1`'s `c_tp_bases` and `finish_type_2`'s `c_tp_mro` -- the
    // two fields a reader walks a type's ancestry with from C.  Cython's
    // `__Pyx_MergeVtables` takes `PyTuple_GET_SIZE(tp_bases)` while laying out
    // a `cdef class`, so a null here is a size read off nothing rather than a
    // field merely left unset.
    //
    // The MRO is handed over as a fresh tuple, which allocates, so the type is
    // rooted across it and read back.
    let roots = pyre_object::gc_roots::push_roots();
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_type);
    // `inherit_special`'s "Setup fast subclass flags", which runs for every
    // type it builds and not only for one an extension declared: a `Py*_Check`
    // written for C is a flag test, so a mirror whose bit is clear answers no
    // for a type that is one.  It resolves the builtins, which allocates, so
    // it reads the type back off the shadow stack.
    set_fast_subclass_flags(mirror, pyre_object::gc_roots::shadow_stack_get(type_slot));
    // `object` names no base, and the empty tuple is what says so: a null is
    // a length read off nothing.
    let bases_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(
        match unsafe {
            pyre_object::typeobject::w_type_get_bases(pyre_object::gc_roots::shadow_stack_get(
                type_slot,
            ))
        } {
            bases if bases.is_null() => pyre_object::w_tuple_new(Vec::new()),
            bases => bases,
        },
    );
    let mro_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(unsafe {
        let mro = pyre_object::typeobject::w_type_get_mro(pyre_object::gc_roots::shadow_stack_get(
            type_slot,
        ));
        match mro.is_null() {
            true => pyre_object::w_tuple_new(Vec::new()),
            false => pyre_object::w_tuple_new((*mro).to_vec()),
        }
    });
    unsafe {
        (*mirror).tp_bases =
            pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(bases_slot));
        (*mirror).tp_mro = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(mro_slot));
    }
    let w_type = pyre_object::gc_roots::shadow_stack_get(type_slot);
    drop(roots);

    // `typeobject.py:801-802`: the base is resolved here, before the deferred
    // half reads its slots.
    let base = base_mirror(w_type);
    unsafe {
        (*mirror).tp_base = base;
        // `typeobject.py`, the same rule `inherit_special` states as
        // "if tp_basicsize is zero or too low, we copy it from the base": what
        // sizes an instance is the most derived struct in the chain, and only
        // the base knows how large that is.
        if !base.is_null() && (*base).tp_basicsize > (*mirror).tp_basicsize {
            (*mirror).tp_basicsize = (*base).tp_basicsize;
        }
        // `inherit_special`'s `COPYVAL(tp_itemsize)`.  A class derived in
        // Python from a var-sized type declared in C stays var-sized, and
        // whoever reads the element width off this mirror has only the base
        // to have learnt it from.
        if !base.is_null() && (*mirror).tp_itemsize == 0 {
            (*mirror).tp_itemsize = (*base).tp_itemsize;
        }
        // `COPYVAL(tp_dictoffset)` and `COPYVAL(tp_weaklistoffset)` beside
        // them.  The block an instance of this class is given is sized for
        // the base's struct, so a field the base declared is at the offset
        // the base declared it at, and a reader that walks up for it would
        // find the same number.
        if !base.is_null() && (*mirror).tp_dictoffset == 0 {
            (*mirror).tp_dictoffset = (*base).tp_dictoffset;
        }
        if !base.is_null() && (*mirror).tp_weaklistoffset == 0 {
            (*mirror).tp_weaklistoffset = (*base).tp_weaklistoffset;
        }
    }
    TYPE_NAMES.lock().insert(hold(mirror as usize), name);
}

/// The mirror of the base whose instance layout `w_type` extends —
/// `typeobject.py best_base`, which is not `__bases__[0]` when more
/// than one base is a type.
///
/// The reference this takes is the one `tp_base` holds; [`forget_type_links`]
/// is what releases it.
fn base_mirror(w_type: PyObjectRef) -> *mut CPyTypeObject {
    let w_base = unsafe { pyre_object::typeobject::w_type_get_best_base(w_type) };
    if w_base.is_null() {
        return std::ptr::null_mut();
    }
    pyobject::make_ref(w_base) as *mut CPyTypeObject
}

/// Allocate a zeroed suite on first use — `typeobject.py fill_slot`, which
/// mallocs the one a slot needs when the type carries none.  A type mirror is
/// immortal for the same reason its instances are, so the leak is the
/// intended lifetime.
macro_rules! table_of {
    ($tp:expr, $field:ident, $shape:ty) => {{
        unsafe {
            if (*$tp).$field.is_null() {
                (*$tp).$field = Box::leak(Box::new(std::mem::zeroed::<$shape>()));
            }
            (*$tp).$field
        }
    }};
}

// ── The slots a mirror carries for the methods this runtime answers ─────
//
// `typeobject.py update_all_slots_builtin` walks `slotdefs` and asks
// `slotdefs.py get_slot_tp_function` what C function each slot of a type the
// runtime defines should carry.  Every factory there closes over
// `w_type.lookup(<method>)` and calls it with the receiver; the trampolines
// below resolve the same name on the receiver's own type instead, so an
// override reaches a slot its base filled.

/// `w_type.lookup(method)`, on `w_owner` where the slot names one.
///
/// A null owner is the shared trampoline, which has no channel to be told
/// whose slot it is and so answers for the receiver's own type.  A bound
/// thunk names the type that owns the slot, which is what
/// `get_slot_tp_function`'s factories close over.
fn slot_method(
    w_owner: PyObjectRef,
    w_self: PyObjectRef,
    method: &str,
) -> Result<PyObjectRef, crate::PyError> {
    let w_type = match w_owner.is_null() {
        false => w_owner,
        true => match crate::typedef::r#type(w_self) {
            Some(w_type) => w_type.as_ptr(),
            None => {
                return Err(crate::PyError::type_error(format!(
                    "a receiver with no type reached {method}"
                )));
            }
        },
    };
    // The fill installs a slot only for a name the type has, so a miss here
    // is the type having lost it since.
    unsafe { crate::baseobjspace::lookup_in_type(w_type, method) }
        .ok_or_else(|| crate::PyError::type_error(format!("the type no longer defines {method}")))
}

/// `space.call_function(w_type.lookup(method), w_self, *arguments)`.
///
/// A type lookup, not an attribute one: an instance dict entry of the same
/// name is not what a slot answers with.
fn call_slot_method(
    w_owner: PyObjectRef,
    w_self: PyObjectRef,
    method: &str,
    arguments: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let mut live = Vec::with_capacity(arguments.len() + 2);
    live.push(w_self);
    live.extend_from_slice(arguments);
    live.push(w_owner);
    let base = roots.pin_roots(&live);
    // Last, so the indices above keep naming what they named.
    let owner_slot = base + arguments.len() + 1;
    let reload = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
    let function = slot_method(
        pyre_object::gc_roots::shadow_stack_get(owner_slot),
        reload(0),
        method,
    )?;
    let function_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(function);
    let mut call = Vec::with_capacity(arguments.len() + 1);
    for index in 0..=arguments.len() {
        call.push(reload(index));
    }
    crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(function_slot),
        &call,
    )
}

/// `slotdefs.py make_unary_slot` — the method called with the receiver alone.
fn unary_body(w_owner: PyObjectRef, raw: *mut CPyObject, method: &str) -> *mut CPyObject {
    let Some(w_self) = super::object::argument(raw) else {
        return std::ptr::null_mut();
    };
    super::object::result(call_slot_method(w_owner, w_self, method, &[]))
}

macro_rules! unary_slots {
    ($(($rust:ident, $method:literal),)*) => {
        $(
            unsafe extern "C" fn $rust(raw: *mut CPyObject) -> *mut CPyObject {
                unary_body(pyre_object::PY_NULL, raw, $method)
            }
        )*
    };
}

unary_slots! {
    (interp_tp_repr, "__repr__"),
    (interp_tp_str, "__str__"),
    (interp_nb_int, "__int__"),
    (interp_nb_float, "__float__"),
    (interp_nb_index, "__index__"),
    (interp_nb_negative, "__neg__"),
    (interp_nb_positive, "__pos__"),
    (interp_nb_absolute, "__abs__"),
    (interp_nb_invert, "__invert__"),
    (interp_am_await, "__await__"),
    (interp_am_aiter, "__aiter__"),
    (interp_am_anext, "__anext__"),
}

/// `slotdefs.py slot_tp_iter` — `iter(self)` for a type this runtime defines.
unsafe extern "C" fn interp_tp_iter(raw: *mut CPyObject) -> *mut CPyObject {
    interp_tp_iter_body(pyre_object::PY_NULL, raw)
}

fn interp_tp_iter_body(w_owner: PyObjectRef, raw: *mut CPyObject) -> *mut CPyObject {
    // A bound slot names the type that owns `__iter__`, so it resolves there;
    // the shared trampoline has no owner and takes the receiver's protocol.
    if !w_owner.is_null() {
        return unary_body(w_owner, raw, "__iter__");
    }
    let Some(w_self) = super::object::argument(raw) else {
        return std::ptr::null_mut();
    };
    super::object::result(crate::baseobjspace::iter(w_self))
}

/// `slotdefs.py slot_tp_iternext`.
///
/// Exhaustion is NULL with the indicator clear, which is what a caller that
/// read this slot off the type distinguishes from a failure.
unsafe extern "C" fn interp_tp_iternext(raw: *mut CPyObject) -> *mut CPyObject {
    interp_tp_iternext_body(pyre_object::PY_NULL, raw)
}

fn interp_tp_iternext_body(w_owner: PyObjectRef, raw: *mut CPyObject) -> *mut CPyObject {
    let Some(w_self) = super::object::argument(raw) else {
        return std::ptr::null_mut();
    };
    // Exhaustion has to stay NULL-with-the-indicator-clear on the bound path
    // too, so the owner's `__next__` is called through the same arm.
    let stepped = match w_owner.is_null() {
        true => crate::baseobjspace::next(w_self),
        false => call_slot_method(w_owner, w_self, "__next__", &[]),
    };
    match stepped {
        Ok(w_item) => pyobject::make_ref(w_item),
        Err(mut error) => {
            if !super::iterator::is_stop_iteration(&mut error) {
                super::pyerrors::set_pending_error(error);
            }
            std::ptr::null_mut()
        }
    }
}

/// `slotdefs.py make_binary_slot` — the method called with the receiver and
/// the operand the slot was handed.
fn binary_body(
    w_owner: PyObjectRef,
    raw: *mut CPyObject,
    operand: *mut CPyObject,
    method: &str,
) -> *mut CPyObject {
    let Some([w_self, w_operand]) = super::object::arguments([raw, operand]) else {
        return std::ptr::null_mut();
    };
    super::object::result(call_slot_method(w_owner, w_self, method, &[w_operand]))
}

macro_rules! binary_slots {
    ($(($rust:ident, $method:literal),)*) => {
        $(
            unsafe extern "C" fn $rust(
                raw: *mut CPyObject,
                operand: *mut CPyObject,
            ) -> *mut CPyObject {
                binary_body(pyre_object::PY_NULL, raw, operand, $method)
            }
        )*
    };
}

binary_slots! {
    (interp_nb_add, "__add__"),
    (interp_nb_subtract, "__sub__"),
    (interp_nb_multiply, "__mul__"),
    (interp_nb_remainder, "__mod__"),
    (interp_nb_divmod, "__divmod__"),
    (interp_nb_lshift, "__lshift__"),
    (interp_nb_rshift, "__rshift__"),
    (interp_nb_and, "__and__"),
    (interp_nb_xor, "__xor__"),
    (interp_nb_or, "__or__"),
    (interp_sq_concat, "__add__"),
    (interp_sq_inplace_concat, "__iadd__"),
    (interp_mp_subscript, "__getitem__"),
    (interp_tp_getattro, "__getattribute__"),
}

/// `slotdefs.py make_binary_slot_int` — the count the slot was handed,
/// passed on as the `int` the method takes.
fn ssize_arg_body(
    w_owner: PyObjectRef,
    raw: *mut CPyObject,
    index: isize,
    method: &str,
) -> *mut CPyObject {
    let Some(w_self) = super::object::argument(raw) else {
        return std::ptr::null_mut();
    };
    let roots = pyre_object::gc_roots::push_roots();
    let slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_self);
    let owner_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_owner);
    // Minting the index can collect, so both are read back.
    let w_index = pyre_object::w_int_new(index as i64);
    let w_self = pyre_object::gc_roots::shadow_stack_get(slot);
    let w_owner = pyre_object::gc_roots::shadow_stack_get(owner_slot);
    super::object::result(call_slot_method(w_owner, w_self, method, &[w_index]))
}

macro_rules! ssize_arg_slots {
    ($(($rust:ident, $method:literal),)*) => {
        $(
            unsafe extern "C" fn $rust(raw: *mut CPyObject, index: isize) -> *mut CPyObject {
                ssize_arg_body(pyre_object::PY_NULL, raw, index, $method)
            }
        )*
    };
}

ssize_arg_slots! {
    (interp_sq_item, "__getitem__"),
    (interp_sq_repeat, "__mul__"),
    (interp_sq_inplace_repeat, "__imul__"),
}

/// `slotdefs.py make_nb_power`.
///
/// The modulus is the third operand `pow()` takes; a caller with none of its
/// own passes `None`, and a NULL is read as that rather than refused.
unsafe extern "C" fn interp_nb_power(
    raw: *mut CPyObject,
    operand: *mut CPyObject,
    modulus: *mut CPyObject,
) -> *mut CPyObject {
    interp_nb_power_body(pyre_object::PY_NULL, raw, operand, modulus)
}

fn interp_nb_power_body(
    w_owner: PyObjectRef,
    raw: *mut CPyObject,
    operand: *mut CPyObject,
    modulus: *mut CPyObject,
) -> *mut CPyObject {
    unsafe { pyobject::realize(modulus) };
    let Some([w_self, w_operand]) = super::object::arguments([raw, operand]) else {
        return std::ptr::null_mut();
    };
    let w_modulus = match modulus.is_null() {
        true => pyre_object::w_none(),
        false => unsafe { pyobject::from_ref(modulus) },
    };
    super::object::result(call_slot_method(
        w_owner,
        w_self,
        "__pow__",
        &[w_operand, w_modulus],
    ))
}

/// `slotdefs.py make_sq_set_item` and `make_sq_ass_item` — one slot for the
/// assignment and the deletion both, told apart by the value being NULL.
fn interp_assign_item(
    w_owner: PyObjectRef,
    raw: *mut CPyObject,
    w_key: PyObjectRef,
    value: *mut CPyObject,
) -> c_int {
    let Some(w_self) = super::object::argument(raw) else {
        return -1;
    };
    let assigned = match value.is_null() {
        true => call_slot_method(w_owner, w_self, "__delitem__", &[w_key]),
        false => match super::object::argument(value) {
            Some(w_value) => call_slot_method(w_owner, w_self, "__setitem__", &[w_key, w_value]),
            None => return -1,
        },
    };
    match super::pyerrors::trap(assigned) {
        Some(_) => 0,
        None => -1,
    }
}

unsafe extern "C" fn interp_mp_ass_subscript(
    raw: *mut CPyObject,
    key: *mut CPyObject,
    value: *mut CPyObject,
) -> c_int {
    interp_mp_ass_subscript_body(pyre_object::PY_NULL, raw, key, value)
}

fn interp_mp_ass_subscript_body(
    w_owner: PyObjectRef,
    raw: *mut CPyObject,
    key: *mut CPyObject,
    value: *mut CPyObject,
) -> c_int {
    unsafe { pyobject::realize(key) };
    unsafe { pyobject::realize(value) };
    let Some(w_key) = super::object::argument(key) else {
        return -1;
    };
    interp_assign_item(w_owner, raw, w_key, value)
}

unsafe extern "C" fn interp_sq_ass_item(
    raw: *mut CPyObject,
    index: isize,
    value: *mut CPyObject,
) -> c_int {
    interp_sq_ass_item_body(pyre_object::PY_NULL, raw, index, value)
}

fn interp_sq_ass_item_body(
    w_owner: PyObjectRef,
    raw: *mut CPyObject,
    index: isize,
    value: *mut CPyObject,
) -> c_int {
    unsafe { pyobject::realize(value) };
    interp_assign_item(w_owner, raw, pyre_object::w_int_new(index as i64), value)
}

/// `slotdefs.py make_unary_slot_int` — the method's answer read as an
/// integer, with -1 the failure a caller checks the indicator for.
fn length_body(w_owner: PyObjectRef, raw: *mut CPyObject, method: &str) -> isize {
    let Some(w_self) = super::object::argument(raw) else {
        return -1;
    };
    let counted =
        call_slot_method(w_owner, w_self, method, &[]).and_then(crate::baseobjspace::int_w);
    match super::pyerrors::trap(counted) {
        Some(count) => count as isize,
        None => -1,
    }
}

macro_rules! length_slots {
    ($(($rust:ident, $method:literal),)*) => {
        $(
            unsafe extern "C" fn $rust(raw: *mut CPyObject) -> isize {
                length_body(pyre_object::PY_NULL, raw, $method)
            }
        )*
    };
}

length_slots! {
    (interp_tp_hash, "__hash__"),
    (interp_sq_length, "__len__"),
    (interp_mp_length, "__len__"),
}

/// Whether a number suite may be written for `w_type` — `typeobject.py
/// fill_slot`, which leaves `list` and `tuple` themselves without one, and
/// every `bytes` or `str` with it, so that a caller testing for the suite
/// reads them as the sequences they are.
fn fills_number_suite(w_type: PyObjectRef) -> bool {
    let exactly =
        |builtin| crate::typedef::gettypefor(builtin).is_some_and(|class| class.as_ptr() == w_type);
    let derived = |builtin| {
        crate::typedef::gettypefor(builtin).is_some_and(|class| unsafe {
            crate::baseobjspace::issubtype_w(w_type, class.as_ptr())
        })
    };
    !exactly(&pyre_object::pyobject::LIST_TYPE)
        && !exactly(&pyre_object::pyobject::TUPLE_TYPE)
        && !derived(&pyre_object::bytesobject::BYTES_TYPE)
        && !derived(&pyre_object::pyobject::STR_TYPE)
}

/// `space.call_args(callable, Arguments([w_self], *args, **kwds))` — the shape
/// the slots handed a call's own arguments share.
///
/// A NULL `args` is the empty tuple a caller with no positional arguments
/// would have passed; a NULL `kwds` is no keywords at all.
fn call_slot_arguments(
    callable: PyObjectRef,
    w_self: PyObjectRef,
    args: *mut CPyObject,
    kwds: *mut CPyObject,
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[callable, w_self, unsafe { pyobject::from_ref(kwds) }]);
    let reload = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
    // Minted last, so nothing pinned above it is a pre-move address.
    let starargs = match args.is_null() {
        true => pyre_object::tupleobject::w_tuple_new(Vec::new()),
        false => unsafe { pyobject::from_ref(args) },
    };
    let starargs_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(starargs);
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            return Err(crate::PyError::runtime_error(
                "a cpyext slot forwarded a call with no current frame",
            ));
        }
        crate::call::call_function_ex(
            unsafe { &mut *frame },
            reload(0),
            reload(1),
            pyre_object::gc_roots::shadow_stack_get(starargs_slot),
            reload(2),
        )
    })
}

/// `slotdefs.py make_tp_call` — `self(*args, **kwds)`.
unsafe extern "C" fn interp_tp_call(
    raw: *mut CPyObject,
    args: *mut CPyObject,
    kwds: *mut CPyObject,
) -> *mut CPyObject {
    interp_tp_call_body(pyre_object::PY_NULL, raw, args, kwds)
}

fn interp_tp_call_body(
    w_owner: PyObjectRef,
    raw: *mut CPyObject,
    args: *mut CPyObject,
    kwds: *mut CPyObject,
) -> *mut CPyObject {
    unsafe { pyobject::realize(args) };
    unsafe { pyobject::realize(kwds) };
    let Some(w_self) = super::object::argument(raw) else {
        return std::ptr::null_mut();
    };
    let called = slot_method(w_owner, w_self, "__call__")
        .and_then(|callable| call_slot_arguments(callable, w_self, args, kwds));
    super::object::result(called)
}

/// `slotdefs.py make_tp_init` — what `__init__` answers is discarded, as the
/// slot has no room for it.
unsafe extern "C" fn interp_tp_init(
    raw: *mut CPyObject,
    args: *mut CPyObject,
    kwds: *mut CPyObject,
) -> c_int {
    interp_tp_init_body(pyre_object::PY_NULL, raw, args, kwds)
}

fn interp_tp_init_body(
    w_owner: PyObjectRef,
    raw: *mut CPyObject,
    args: *mut CPyObject,
    kwds: *mut CPyObject,
) -> c_int {
    unsafe { pyobject::realize(args) };
    unsafe { pyobject::realize(kwds) };
    let Some(w_self) = super::object::argument(raw) else {
        return -1;
    };
    let started = slot_method(w_owner, w_self, "__init__")
        .and_then(|callable| call_slot_arguments(callable, w_self, args, kwds));
    match super::pyerrors::trap(started) {
        Some(_) => 0,
        None => -1,
    }
}

/// `slotdefs.py make_tp_new` — `cls.__new__(cls, *args, **kwds)`.
///
/// The receiver is the class being built, so the constructor is read off it
/// rather than off its type; the attribute read is what unwraps the static
/// method `__new__` is stored as.
unsafe extern "C" fn interp_tp_new(
    raw: *mut CPyTypeObject,
    args: *mut CPyObject,
    kwds: *mut CPyObject,
) -> *mut CPyObject {
    interp_tp_new_body(pyre_object::PY_NULL, raw, args, kwds)
}

fn interp_tp_new_body(
    w_owner: PyObjectRef,
    raw: *mut CPyTypeObject,
    args: *mut CPyObject,
    kwds: *mut CPyObject,
) -> *mut CPyObject {
    unsafe { pyobject::realize(args) };
    unsafe { pyobject::realize(kwds) };
    let Some(w_class) = super::object::argument(raw as *mut CPyObject) else {
        return std::ptr::null_mut();
    };
    // The constructor is read off the type that owns the slot where there is
    // one; the class being built stays the first argument, because that is
    // what `cls.__new__(cls, ...)` passes and what the allocator sizes for.
    let w_read = match w_owner.is_null() {
        true => w_class,
        false => w_owner,
    };
    let made = crate::baseobjspace::getattr_str(w_read, "__new__")
        .and_then(|callable| call_slot_arguments(callable, w_class, args, kwds));
    super::object::result(made)
}

/// `slotdefs.py make_tp_descr_get` — a NULL receiver is the class access,
/// which reaches `__get__` as `None`.
unsafe extern "C" fn interp_tp_descr_get(
    raw: *mut CPyObject,
    object: *mut CPyObject,
    of_type: *mut CPyObject,
) -> *mut CPyObject {
    interp_tp_descr_get_body(pyre_object::PY_NULL, raw, object, of_type)
}

fn interp_tp_descr_get_body(
    w_owner: PyObjectRef,
    raw: *mut CPyObject,
    object: *mut CPyObject,
    of_type: *mut CPyObject,
) -> *mut CPyObject {
    unsafe { pyobject::realize(object) };
    unsafe { pyobject::realize(of_type) };
    let Some(w_self) = super::object::argument(raw) else {
        return std::ptr::null_mut();
    };
    let or_none = |raw: *mut CPyObject| match raw.is_null() {
        true => pyre_object::w_none(),
        false => unsafe { pyobject::from_ref(raw) },
    };
    let got = call_slot_method(
        w_owner,
        w_self,
        "__get__",
        &[or_none(object), or_none(of_type)],
    );
    super::object::result(got)
}

/// `slotdefs.py make_tp_descr_set` — one slot for the assignment and the
/// deletion both, told apart by the value being NULL.
unsafe extern "C" fn interp_tp_descr_set(
    raw: *mut CPyObject,
    object: *mut CPyObject,
    value: *mut CPyObject,
) -> c_int {
    interp_tp_descr_set_body(pyre_object::PY_NULL, raw, object, value)
}

fn interp_tp_descr_set_body(
    w_owner: PyObjectRef,
    raw: *mut CPyObject,
    object: *mut CPyObject,
    value: *mut CPyObject,
) -> c_int {
    unsafe { pyobject::realize(object) };
    unsafe { pyobject::realize(value) };
    let (Some(w_self), Some(w_object)) = (
        super::object::argument(raw),
        super::object::argument(object),
    ) else {
        return -1;
    };
    let assigned = match value.is_null() {
        true => call_slot_method(w_owner, w_self, "__delete__", &[w_object]),
        false => match super::object::argument(value) {
            Some(w_value) => call_slot_method(w_owner, w_self, "__set__", &[w_object, w_value]),
            None => return -1,
        },
    };
    match super::pyerrors::trap(assigned) {
        Some(_) => 0,
        None => -1,
    }
}

// ── bound slots ────────────────────────────────────────────────────────
//
// `slotdefs.py get_slot_tp_function` is `@specialize.memo()` over a
// translation-time typedef, and `func_renamer` emits one C function per
// (typedef, slot).  Each closes over `w_type.lookup(attr)` -- the method of
// the type that OWNS the slot -- so a subclass reaching its base's slot runs
// the base's method.  A shared `extern "C"` function cannot do that: its only
// channel is its arguments, the owner is not among them, and re-resolving on
// the receiver sends `base->tp_slot(self)` from a subclass back to itself.
//
// The pool below is that channel.  Each thunk is one monomorphisation, so it
// carries an address of its own, and that address is what the mirror's slot
// holds; the index recovers the owner.  Where RPython enumerates the pairs at
// translation time this assigns them on demand, because a type of this
// runtime's is minted while it runs rather than declared in a closed table.
//
// The population is closed all the same.  Only a type this runtime defines
// reaches the fill -- a type an extension readied never does -- and an entry
// is keyed by the type that OWNS the method rather than by the type carrying
// the slot.  Both resolve the same method, because the lookup that found it
// landed on the owner, and a measured stdlib import needs 674 entries where
// per-type keying would need 1555.

const BOUND_ROW: usize = 16;

/// The type a bound thunk resolves its method on, and the name to resolve.
///
/// The owner is held as its mirror: that address is fixed for the mirror's
/// life, where the object it stands for is free to move.
#[derive(Clone, Copy)]
struct BoundSlot {
    owner: *mut CPyObject,
    method: &'static str,
}

/// What one pool holds: its entries, the thunks that name them, and the index
/// each `(owner, method)` was given.
struct BoundFamily {
    entries: &'static [std::sync::atomic::AtomicPtr<BoundSlot>],
    // Reached through a function rather than held as a table: a `static` may
    // not hold a `*const c_void`, which is not `Sync`, and a pointer may not
    // be cast to an address while the table is being built.  Each family's
    // own table is typed with its own signature, which is a function pointer
    // and so is `Sync`.
    thunk_at: fn(usize) -> *const c_void,
    taken: super::ForkMutex<BoundNames>,
}

/// Every address handed out of a pool.
///
/// Only an assigned thunk can appear in a slot, so this is what
/// [`is_bound_thunk`] asks rather than walking the tables.
static BOUND_MINTED: super::ForkMutex<BoundAddresses> = super::ForkMutex::new(
    BoundAddresses::with_hasher(std::hash::BuildHasherDefault::new()),
);

type BoundAddresses = super::address_table::AddressSet;

type BoundNames = std::collections::HashMap<
    (usize, &'static str),
    usize,
    std::hash::BuildHasherDefault<std::hash::DefaultHasher>,
>;

/// The entry behind a thunk's index, or `None` for one never assigned.
///
/// Read without the lock: an entry is published once and never rewritten, so
/// a reader finds either null or a block that outlives it.
fn bound_slot(family: &BoundFamily, index: usize) -> Option<BoundSlot> {
    let entry = family
        .entries
        .get(index)?
        .load(std::sync::atomic::Ordering::Acquire);
    match entry.is_null() {
        true => None,
        false => Some(unsafe { *entry }),
    }
}

/// The thunk standing for `(owner, method)`, minting one on first ask -- or
/// `None` once the pool is spent, which leaves the caller with the shared
/// trampoline it would have installed before this existed.
fn bind_slot(
    family: &'static BoundFamily,
    owner: *mut CPyObject,
    method: &'static str,
) -> Option<*const c_void> {
    let mut taken = family.taken.lock();
    if let Some(&index) = taken.get(&(owner as usize, method)) {
        return Some((family.thunk_at)(index));
    }
    let index = taken.len();
    if index >= family.entries.len() {
        return None;
    }
    // Published before the index is recorded, so a reader reaching a thunk
    // through the table always finds its entry filled.
    family.entries[index].store(
        Box::into_raw(Box::new(BoundSlot { owner, method })),
        std::sync::atomic::Ordering::Release,
    );
    taken.insert((owner as usize, method), index);
    let thunk = (family.thunk_at)(index);
    BOUND_MINTED.lock().insert(thunk as usize);
    Some(thunk)
}

/// One row of a pool's thunk table.
///
/// A row rather than a flat table because `macro_rules!` cannot nest two
/// repetitions its input does not nest: the inner list is written out here
/// and the outer one names the rows.
macro_rules! bound_row {
    ($thunk:ident, $hi:literal) => {
        bound_row!(@row $thunk, $hi, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    };
    (@row $thunk:ident, $hi:literal, [$($lo:literal),*]) => {
        &[$( $thunk::<{ $hi * BOUND_ROW + $lo }> ),*]
    };
}

/// Declare one pool: its entries, the thunk that indexes them, the table of
/// thunk addresses, and the family tying the three together.
///
/// The index reaches the shared dispatcher as an ARGUMENT, never through a
/// table read inside the thunk.  A body that only indexes a table the program
/// has not written yet folds to the same code for every index, and
/// identical-code-folding merges the whole pool into a single address -- which
/// would bind every slot in the process to one entry, and say nothing.
/// `every_thunk_has_an_address_of_its_own` is what holds this.
macro_rules! bound_pool {
    (
        $family:ident, $entries:ident, $thunks:ident, $at:ident, $thunk:ident,
        $dispatch:ident, $rows:literal,
        ($($arg:ident: $argty:ty),*) -> $ret:ty, ($($pass:ident),*),
        [$($hi:literal),* $(,)?]
    ) => {
        static $entries: [std::sync::atomic::AtomicPtr<BoundSlot>; $rows * BOUND_ROW] =
            [const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) };
                $rows * BOUND_ROW];

        unsafe extern "C" fn $thunk<const K: usize>($($arg: $argty),*) -> $ret {
            $dispatch(K, $($pass),*)
        }

        static $thunks: &[&[unsafe extern "C" fn($($argty),*) -> $ret]] =
            &[$( bound_row!($thunk, $hi) ),*];

        fn $at(index: usize) -> *const c_void {
            $thunks[index / BOUND_ROW][index % BOUND_ROW] as *const c_void
        }

        static $family: BoundFamily = BoundFamily {
            entries: &$entries,
            thunk_at: $at,
            taken: super::ForkMutex::new(BoundNames::with_hasher(
                std::hash::BuildHasherDefault::new(),
            )),
        };
    };
}

bound_pool!(
    UNARY_BOUND, UNARY_ENTRIES, UNARY_THUNKS, unary_thunk_at, bound_unary, dispatch_unary, 32,
    (raw: *mut CPyObject) -> *mut CPyObject, (raw),
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
);

bound_pool!(
    LENGTH_BOUND, LENGTH_ENTRIES, LENGTH_THUNKS, length_thunk_at, bound_length, dispatch_length, 8,
    (raw: *mut CPyObject) -> isize, (raw),
    [0, 1, 2, 3, 4, 5, 6, 7]
);

bound_pool!(
    BINARY_BOUND, BINARY_ENTRIES, BINARY_THUNKS, binary_thunk_at, bound_binary, dispatch_binary, 16,
    (raw: *mut CPyObject, operand: *mut CPyObject) -> *mut CPyObject, (raw, operand),
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
);

bound_pool!(
    SSIZE_BOUND, SSIZE_ENTRIES, SSIZE_THUNKS, ssize_thunk_at, bound_ssize, dispatch_ssize, 4,
    (raw: *mut CPyObject, index: isize) -> *mut CPyObject, (raw, index),
    [0, 1, 2, 3]
);

bound_pool!(
    TERNARY_BOUND, TERNARY_ENTRIES, TERNARY_THUNKS, ternary_thunk_at, bound_ternary, dispatch_ternary, 24,
    (raw: *mut CPyObject, second: *mut CPyObject, third: *mut CPyObject) -> *mut CPyObject, (raw, second, third),
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23]
);

bound_pool!(
    TERNINT_BOUND, TERNINT_ENTRIES, TERNINT_THUNKS, ternint_thunk_at, bound_ternint, dispatch_ternint, 8,
    (raw: *mut CPyObject, second: *mut CPyObject, third: *mut CPyObject) -> c_int, (raw, second, third),
    [0, 1, 2, 3, 4, 5, 6, 7]
);

bound_pool!(
    ASSIGN_BOUND, ASSIGN_ENTRIES, ASSIGN_THUNKS, assign_thunk_at, bound_assign, dispatch_assign, 4,
    (raw: *mut CPyObject, index: isize, value: *mut CPyObject) -> c_int, (raw, index, value),
    [0, 1, 2, 3]
);

/// Every pool, for the questions asked of all of them at once.
static BOUND_FAMILIES: &[&BoundFamily] = &[
    &UNARY_BOUND,
    &LENGTH_BOUND,
    &BINARY_BOUND,
    &SSIZE_BOUND,
    &TERNARY_BOUND,
    &TERNINT_BOUND,
    &ASSIGN_BOUND,
];

/// Whether `slot` is a thunk out of any pool -- the question
/// [`is_interpreter_slot`] asks of the shared trampolines.
fn is_bound_thunk(slot: *const c_void) -> bool {
    BOUND_MINTED.lock().contains(&(slot as usize))
}

/// The pool a shared trampoline belongs to, by its C signature.
///
/// Matched against the tables rather than told: a trampoline this does not
/// name simply keeps re-resolving on the receiver, which is what every one of
/// them did before the pools existed.
fn bound_family_of(function: *const c_void) -> Option<&'static BoundFamily> {
    let named = |candidate: *const c_void| std::ptr::eq(function, candidate);
    let in_table = |table: &[(&str, *const c_void)]| table.iter().any(|&(_, entry)| named(entry));
    if UNARY_SLOTS
        .iter()
        .any(|&(_, entry, _)| named(entry as *const c_void))
    {
        return Some(&UNARY_BOUND);
    }
    if LENGTH_SLOTS
        .iter()
        .any(|&(_, entry, _)| named(entry as *const c_void))
    {
        return Some(&LENGTH_BOUND);
    }
    if BINARY_SLOTS
        .iter()
        .any(|&(_, entry, _)| named(entry as *const c_void))
    {
        return Some(&BINARY_BOUND);
    }
    if SSIZE_ARG_SLOTS
        .iter()
        .any(|&(_, entry, _)| named(entry as *const c_void))
    {
        return Some(&SSIZE_BOUND);
    }
    let _ = in_table;
    if named(interp_nb_power as *const c_void)
        || named(interp_tp_call as *const c_void)
        || named(interp_tp_new as *const c_void)
        || named(interp_tp_descr_get as *const c_void)
    {
        return Some(&TERNARY_BOUND);
    }
    if named(interp_tp_init as *const c_void)
        || named(interp_tp_descr_set as *const c_void)
        || named(interp_mp_ass_subscript as *const c_void)
    {
        return Some(&TERNINT_BOUND);
    }
    if named(interp_sq_ass_item as *const c_void) {
        return Some(&ASSIGN_BOUND);
    }
    None
}

// ── the dispatchers ────────────────────────────────────────────────────
//
// One per C signature.  An index with no entry is a thunk reached before it
// was assigned, which cannot happen through a slot this layer wrote; the
// failure answer is the one the shared trampoline gives a bad receiver.

fn dispatch_unary(index: usize, raw: *mut CPyObject) -> *mut CPyObject {
    let Some(slot) = bound_slot(&UNARY_BOUND, index) else {
        return std::ptr::null_mut();
    };
    let w_owner = unsafe { pyobject::from_ref(slot.owner) };
    match slot.method {
        // Exhaustion is NULL with the indicator clear, which only this arm
        // answers with.
        "__next__" => interp_tp_iternext_body(w_owner, raw),
        method => unary_body(w_owner, raw, method),
    }
}

fn dispatch_length(index: usize, raw: *mut CPyObject) -> isize {
    let Some(slot) = bound_slot(&LENGTH_BOUND, index) else {
        return -1;
    };
    length_body(unsafe { pyobject::from_ref(slot.owner) }, raw, slot.method)
}

fn dispatch_binary(index: usize, raw: *mut CPyObject, operand: *mut CPyObject) -> *mut CPyObject {
    let Some(slot) = bound_slot(&BINARY_BOUND, index) else {
        return std::ptr::null_mut();
    };
    binary_body(
        unsafe { pyobject::from_ref(slot.owner) },
        raw,
        operand,
        slot.method,
    )
}

fn dispatch_ssize(index: usize, raw: *mut CPyObject, at: isize) -> *mut CPyObject {
    let Some(slot) = bound_slot(&SSIZE_BOUND, index) else {
        return std::ptr::null_mut();
    };
    ssize_arg_body(
        unsafe { pyobject::from_ref(slot.owner) },
        raw,
        at,
        slot.method,
    )
}

fn dispatch_ternary(
    index: usize,
    raw: *mut CPyObject,
    second: *mut CPyObject,
    third: *mut CPyObject,
) -> *mut CPyObject {
    let Some(slot) = bound_slot(&TERNARY_BOUND, index) else {
        return std::ptr::null_mut();
    };
    let w_owner = unsafe { pyobject::from_ref(slot.owner) };
    match slot.method {
        "__pow__" => interp_nb_power_body(w_owner, raw, second, third),
        "__call__" => interp_tp_call_body(w_owner, raw, second, third),
        "__new__" => interp_tp_new_body(w_owner, raw as *mut CPyTypeObject, second, third),
        _ => interp_tp_descr_get_body(w_owner, raw, second, third),
    }
}

fn dispatch_ternint(
    index: usize,
    raw: *mut CPyObject,
    second: *mut CPyObject,
    third: *mut CPyObject,
) -> c_int {
    let Some(slot) = bound_slot(&TERNINT_BOUND, index) else {
        return -1;
    };
    let w_owner = unsafe { pyobject::from_ref(slot.owner) };
    match slot.method {
        "__init__" => interp_tp_init_body(w_owner, raw, second, third),
        // One slot answers for the assignment and the deletion both, so the
        // name recorded here says which slot it is, not which of the two.
        "__set__" | "__delete__" => interp_tp_descr_set_body(w_owner, raw, second, third),
        _ => interp_mp_ass_subscript_body(w_owner, raw, second, third),
    }
}

fn dispatch_assign(index: usize, raw: *mut CPyObject, at: isize, value: *mut CPyObject) -> c_int {
    let Some(slot) = bound_slot(&ASSIGN_BOUND, index) else {
        return -1;
    };
    interp_sq_ass_item_body(unsafe { pyobject::from_ref(slot.owner) }, raw, at, value)
}

/// The two halves of reaching one slot: what it holds, and how it is written.
///
/// The read is what lets an inherited slot be taken from a base as it stands;
/// the write takes the type because a number suite is not written for every
/// one of them.
#[derive(Clone, Copy)]
struct SlotAccess {
    read: fn(*mut CPyTypeObject) -> *const c_void,
    write: fn(*mut CPyTypeObject, PyObjectRef, *const c_void),
}

macro_rules! scalar_entry {
    ($field:ident) => {
        SlotAccess {
            read: |tp| unsafe { (*tp).$field },
            write: |tp, _, slot| unsafe { (*tp).$field = slot },
        }
    };
}

/// The entry of a suite, which a type may carry none of.
macro_rules! suite_entry {
    ($suite:ident, $shape:ty, $field:ident, $write:expr) => {
        SlotAccess {
            read: |tp| unsafe {
                match (*tp).$suite.is_null() {
                    true => std::ptr::null(),
                    false => (*(*tp).$suite).$field,
                }
            },
            write: $write,
        }
    };
}

macro_rules! number_entry {
    ($field:ident) => {
        suite_entry!(
            tp_as_number,
            CPyNumberMethods,
            $field,
            |tp, w_type, slot| {
                if fills_number_suite(w_type) {
                    unsafe { (*table_of!(tp, tp_as_number, CPyNumberMethods)).$field = slot }
                }
            }
        )
    };
}

macro_rules! sequence_entry {
    ($field:ident) => {
        suite_entry!(
            tp_as_sequence,
            CPySequenceMethods,
            $field,
            |tp, _, slot| unsafe {
                (*table_of!(tp, tp_as_sequence, CPySequenceMethods)).$field = slot
            }
        )
    };
}

macro_rules! mapping_entry {
    ($field:ident) => {
        suite_entry!(
            tp_as_mapping,
            CPyMappingMethods,
            $field,
            |tp, _, slot| unsafe {
                (*table_of!(tp, tp_as_mapping, CPyMappingMethods)).$field = slot
            }
        )
    };
}

macro_rules! async_entry {
    ($field:ident) => {
        suite_entry!(tp_as_async, CPyAsyncMethods, $field, |tp, _, slot| unsafe {
            (*table_of!(tp, tp_as_async, CPyAsyncMethods)).$field = slot
        })
    };
}

macro_rules! buffer_entry {
    ($field:ident) => {
        suite_entry!(tp_as_buffer, CPyBufferProcs, $field, |tp, _, slot| unsafe {
            (*table_of!(tp, tp_as_buffer, CPyBufferProcs)).$field = slot
        })
    };
}

/// The `tp_` slots this runtime fills to reach a method it answers for
/// itself, and the method whose presence on the type earns each one.
const UNARY_SLOTS: [(
    &str,
    unsafe extern "C" fn(*mut CPyObject) -> *mut CPyObject,
    SlotAccess,
); 14] = [
    ("__repr__", interp_tp_repr, scalar_entry!(tp_repr)),
    ("__str__", interp_tp_str, scalar_entry!(tp_str)),
    ("__iter__", interp_tp_iter, scalar_entry!(tp_iter)),
    ("__next__", interp_tp_iternext, scalar_entry!(tp_iternext)),
    ("__int__", interp_nb_int, number_entry!(nb_int)),
    ("__float__", interp_nb_float, number_entry!(nb_float)),
    ("__index__", interp_nb_index, number_entry!(nb_index)),
    ("__neg__", interp_nb_negative, number_entry!(nb_negative)),
    ("__pos__", interp_nb_positive, number_entry!(nb_positive)),
    ("__abs__", interp_nb_absolute, number_entry!(nb_absolute)),
    ("__invert__", interp_nb_invert, number_entry!(nb_invert)),
    ("__await__", interp_am_await, async_entry!(am_await)),
    ("__aiter__", interp_am_aiter, async_entry!(am_aiter)),
    ("__anext__", interp_am_anext, async_entry!(am_anext)),
];

/// The slots whose C function answers a count rather than an object.
const LENGTH_SLOTS: [(
    &str,
    unsafe extern "C" fn(*mut CPyObject) -> isize,
    SlotAccess,
); 3] = [
    ("__hash__", interp_tp_hash, scalar_entry!(tp_hash)),
    ("__len__", interp_sq_length, sequence_entry!(sq_length)),
    ("__len__", interp_mp_length, mapping_entry!(mp_length)),
];

/// The slots taking a second object — `slotdefs.py make_binary_slot`.
const BINARY_SLOTS: [(
    &str,
    unsafe extern "C" fn(*mut CPyObject, *mut CPyObject) -> *mut CPyObject,
    SlotAccess,
); 14] = [
    ("__add__", interp_nb_add, number_entry!(nb_add)),
    ("__sub__", interp_nb_subtract, number_entry!(nb_subtract)),
    ("__mul__", interp_nb_multiply, number_entry!(nb_multiply)),
    ("__mod__", interp_nb_remainder, number_entry!(nb_remainder)),
    ("__divmod__", interp_nb_divmod, number_entry!(nb_divmod)),
    ("__lshift__", interp_nb_lshift, number_entry!(nb_lshift)),
    ("__rshift__", interp_nb_rshift, number_entry!(nb_rshift)),
    ("__and__", interp_nb_and, number_entry!(nb_and)),
    ("__xor__", interp_nb_xor, number_entry!(nb_xor)),
    ("__or__", interp_nb_or, number_entry!(nb_or)),
    ("__add__", interp_sq_concat, sequence_entry!(sq_concat)),
    (
        "__iadd__",
        interp_sq_inplace_concat,
        sequence_entry!(sq_inplace_concat),
    ),
    (
        "__getitem__",
        interp_mp_subscript,
        mapping_entry!(mp_subscript),
    ),
    (
        "__getattribute__",
        interp_tp_getattro,
        scalar_entry!(tp_getattro),
    ),
];

/// The slots taking a count — `slotdefs.py make_binary_slot_int`.
const SSIZE_ARG_SLOTS: [(
    &str,
    unsafe extern "C" fn(*mut CPyObject, isize) -> *mut CPyObject,
    SlotAccess,
); 3] = [
    ("__getitem__", interp_sq_item, sequence_entry!(sq_item)),
    ("__mul__", interp_sq_repeat, sequence_entry!(sq_repeat)),
    (
        "__imul__",
        interp_sq_inplace_repeat,
        sequence_entry!(sq_inplace_repeat),
    ),
];

/// Is `slot` one this runtime installed to reach a method of its own?
///
/// Such a slot must never be published back as that method: the call it makes
/// resolves the name again, and a published wrapper would answer with itself.
/// Only the slots `inherit_slots` copies can reach a type built from C, and
/// those are the scalar ones -- a suite is never handed down whole.
fn is_interpreter_slot(slot: *const c_void) -> bool {
    let named = |function| std::ptr::eq(slot, function);
    if is_bound_thunk(slot) {
        return true;
    }
    UNARY_SLOTS
        .iter()
        .any(|&(_, function, _)| named(function as *const c_void))
        || LENGTH_SLOTS
            .iter()
            .any(|&(_, function, _)| named(function as *const c_void))
        || BINARY_SLOTS
            .iter()
            .any(|&(_, function, _)| named(function as *const c_void))
        || named(interp_tp_call as *const c_void)
        || named(interp_tp_init as *const c_void)
        || named(interp_tp_new as *const c_void)
        || named(interp_tp_descr_get as *const c_void)
        || named(interp_tp_descr_set as *const c_void)
}

/// Install one slot — the body `typeobject.py update_all_slots` and
/// `update_all_slots_builtin` share.
///
/// The two differ in what earns a trampoline.  A type this runtime defines
/// answers for every method its MRO carries, so the lookup finding one is
/// enough.  A class written in Python earns one only for a method of its own;
/// for anything inherited it takes the base's slot as it stands, because a
/// trampoline installed for an inherited method resolves the name back to the
/// wrapper that reads this very slot, and the two would call each other until
/// the stack ran out.
fn install_slot(
    mirror: *mut CPyTypeObject,
    w_type: PyObjectRef,
    heaptype: bool,
    method: &'static str,
    access: SlotAccess,
    function: *const c_void,
) {
    let found = unsafe { crate::baseobjspace::lookup_where(w_type, method) };
    let owned = match heaptype {
        true => found.is_some_and(|(owner, _)| owner == w_type),
        false => found.is_some(),
    };
    let base = unsafe { (*mirror).tp_base };
    // `update_all_slots`'s one exception: `__call__` is not handed down, so a
    // heap type that does not define one carries no `tp_call` either.
    let inherits = method != "__call__";
    let inherited = match owned || base.is_null() || !inherits {
        true => std::ptr::null(),
        false => (access.read)(base),
    };
    let value = match inherited.is_null() {
        // Nothing to inherit, so the trampoline is the slot -- but only where
        // the type has the method at all.
        true if owned => bound_or_shared(heaptype, found, method, function),
        true => return,
        false => inherited,
    };
    (access.write)(mirror, w_type, value);
}

/// The slot to install for a method the type answers for itself: a thunk
/// bound to the type that owns the method, or the shared trampoline.
///
/// A class written in Python takes the shared one, and that is not a
/// shortfall.  Its method is resolved on the receiver by the language itself,
/// which is what `slot_tp_*` does for the same reason; `update_all_slots`
/// hands such a class the very same `slot_apifunc`.  Only a type this runtime
/// defines needs the owner recorded, because only its slot stands where a
/// concrete C function would stand for CPython.
fn bound_or_shared(
    heaptype: bool,
    found: Option<(PyObjectRef, PyObjectRef)>,
    method: &'static str,
    function: *const c_void,
) -> *const c_void {
    let Some((w_owner, _)) = found.filter(|_| !heaptype) else {
        return function;
    };
    let Some(family) = bound_family_of(function) else {
        return function;
    };
    // The mirror, because the pool outlives any one call and the object it
    // stands for is free to move.  A type's mirror is immortal, so the
    // reference this takes is never given back.
    let owner = pyobject::make_ref(w_owner);
    bind_slot(family, owner, method).unwrap_or(function)
}

/// `typeobject.py update_all_slots` and `update_all_slots_builtin` — fill the
/// slots `w_type` answers for itself.
fn fill_interpreter_slots(mirror: *mut CPyTypeObject, w_type: PyObjectRef) {
    let heaptype = unsafe { pyre_object::typeobject::w_type_is_heaptype(w_type) };
    let defines = |method| unsafe { crate::baseobjspace::lookup_in_type(w_type, method) }.is_some();
    macro_rules! fill {
        ($table:ident) => {
            for (method, function, access) in $table {
                install_slot(
                    mirror,
                    w_type,
                    heaptype,
                    method,
                    access,
                    function as *const c_void,
                );
            }
        };
    }
    fill!(UNARY_SLOTS);
    fill!(LENGTH_SLOTS);
    fill!(BINARY_SLOTS);
    fill!(SSIZE_ARG_SLOTS);
    install_slot(
        mirror,
        w_type,
        heaptype,
        "__pow__",
        number_entry!(nb_power),
        interp_nb_power as *const c_void,
    );
    for (method, function, access) in [
        (
            "__call__",
            interp_tp_call as *const c_void,
            scalar_entry!(tp_call),
        ),
        (
            "__init__",
            interp_tp_init as *const c_void,
            scalar_entry!(tp_init),
        ),
        (
            "__new__",
            interp_tp_new as *const c_void,
            scalar_entry!(tp_new),
        ),
        (
            "__get__",
            interp_tp_descr_get as *const c_void,
            scalar_entry!(tp_descr_get),
        ),
    ] {
        install_slot(mirror, w_type, heaptype, method, access, function);
    }
    // Either name earns the slot, because one answers for the assignment and
    // the deletion both and a type may define only one of them.
    if defines("__set__") || defines("__delete__") {
        install_slot(
            mirror,
            w_type,
            heaptype,
            match defines("__set__") {
                true => "__set__",
                false => "__delete__",
            },
            scalar_entry!(tp_descr_set),
            interp_tp_descr_set as *const c_void,
        );
    }
    // `slotdefs.py make_bf_getbuffer` earns the buffer slots from the type's
    // own declaration for a type this runtime defines, and the `__buffer__`
    // slotdef earns them from the method for a class written in Python --
    // which is the only one of the two a heap type can satisfy, so a heap
    // type that defines no `__buffer__` keeps whatever its base handed down.
    let exports_buffer = match heaptype {
        true => unsafe { crate::baseobjspace::lookup_where(w_type, "__buffer__") }
            .is_some_and(|(owner, _)| owner == w_type),
        false => super::buffer::declares_buffer(w_type),
    };
    if exports_buffer {
        for (function, access) in [
            (
                super::buffer::interp_bf_getbuffer as *const c_void,
                buffer_entry!(bf_getbuffer),
            ),
            (
                super::buffer::interp_bf_releasebuffer as *const c_void,
                buffer_entry!(bf_releasebuffer),
            ),
        ] {
            (access.write)(mirror, w_type, function);
        }
    }
    // Both names, because one slot answers for the assignment and the
    // deletion and there is no spelling for having only one of them.
    if defines("__setitem__") && defines("__delitem__") {
        install_slot(
            mirror,
            w_type,
            heaptype,
            "__setitem__",
            mapping_entry!(mp_ass_subscript),
            interp_mp_ass_subscript as *const c_void,
        );
        install_slot(
            mirror,
            w_type,
            heaptype,
            "__setitem__",
            sequence_entry!(sq_ass_item),
            interp_sq_ass_item as *const c_void,
        );
    }
}

/// `typeobject.py finish_type_2` — the half of the fill that reads
/// the base's own slots.
///
/// Deferred past the static bindings at startup for the reason
/// `api.py attach_all` defers it: a base bound a few entries later in
/// the same table has nothing in its slots yet.
///
/// Idempotent, so the deferred drain may reach a mirror the lazy path already
/// finished.
pub(super) fn finish_interpreter_type(mirror: *mut CPyTypeObject, w_type: PyObjectRef) {
    let base = unsafe { (*mirror).tp_base };
    inherit_mirror_slots(mirror, base);
    // `typeobject.py type_attach`.  A constructor is inherited from
    // the base for every type but one derived directly from `object` and not
    // written in Python, which is `object`'s own `tp_new` and nothing else.
    if unsafe { (*mirror).tp_new.is_null() } && !base.is_null() {
        let derived = !std::ptr::eq(base, &raw mut PyBaseObject_Type)
            || unsafe { (*mirror).tp_flags.contains(TpFlags::PY_TPFLAGS_HEAPTYPE) };
        if derived {
            unsafe { (*mirror).tp_new = (*base).tp_new };
        }
    }
    // `typeobject.py:1079-1085`, after the inheritance above so that a base's
    // own attribute hooks are not covered by the terminals.
    unsafe {
        if (*mirror).tp_setattro.is_null() {
            (*mirror).tp_setattro = super::object::PyObject_GenericSetAttr as *const c_void;
        }
        if (*mirror).tp_getattro.is_null() {
            (*mirror).tp_getattro = super::object::PyObject_GenericGetAttr as *const c_void;
        }
    }
    if unsafe { (*mirror).tp_dict.is_null() } {
        stamp_tp_dict(mirror, w_type);
    }
    // Last, as `type_attach` runs it: what the type answers for itself wins
    // over what the base handed down.
    fill_interpreter_slots(mirror, w_type);
}

/// Hand the vectorcall protocol down from `base` to `tp`.
///
/// The offset always travels, so that `PyVectorcall_Call` -- which reads it
/// without consulting the flag -- keeps working for a subclass; the flag
/// travels only to a subclass that leaves `tp_call` alone, and decides whether
/// the offset is read of its own accord.  A class derived in Python from a C
/// type that lends its instances a function keeps lending it: the instance is
/// sized by the base's `tp_basicsize`, which the caller copies for exactly that
/// reason, and filled by the base's `tp_new`.
///
/// Both have to run before `tp_call` is filled in.  A subclass that does
/// declare its own `tp_call` means it, and it is `tp_call` -- not the function
/// its base lends instances -- that has to answer for it.
fn inherit_vectorcall(tp: *mut CPyTypeObject, base: *mut CPyTypeObject) {
    unsafe {
        if (*tp).tp_vectorcall_offset == 0 {
            (*tp).tp_vectorcall_offset = (*base).tp_vectorcall_offset;
        }
        if (*tp).tp_call.is_null()
            && (*base)
                .tp_flags
                .contains(TpFlags::PY_TPFLAGS_HAVE_VECTORCALL)
        {
            (*tp).tp_flags |= TpFlags::PY_TPFLAGS_HAVE_VECTORCALL;
        }
    }
}

/// `typeobject.py inherit_slots` — what a mirror takes from its base.
///
/// Narrower than the [`inherit_slots`] `PyType_Ready` runs: the slots left out
/// here are the ones a type's own Python-level methods answer for, and copying
/// a base's would cover an override the subclass defines.
fn inherit_mirror_slots(tp: *mut CPyTypeObject, base: *mut CPyTypeObject) {
    if base.is_null() {
        return;
    }
    macro_rules! inherit {
        ($($field:ident),* $(,)?) => {
            $(
                unsafe {
                    if (*tp).$field.is_null() {
                        (*tp).$field = (*base).$field;
                    }
                }
            )*
        };
    }
    inherit!(
        tp_dealloc,
        tp_init,
        tp_alloc,
        tp_free,
        tp_setattro,
        tp_getattro,
    );
    // `typeobject.py inherit_slots`.  A mirror allocated as a heap
    // type already names suites of its own, so the pointer is taken from the
    // base only when there is none; otherwise what a subclass inherits is the
    // entries, one at a time.
    macro_rules! suite {
        ($field:ident, $($entry:ident),* $(,)?) => {
            unsafe {
                if (*tp).$field.is_null() {
                    (*tp).$field = (*base).$field;
                } else if !(*base).$field.is_null() {
                    $(
                        if (*(*tp).$field).$entry.is_null() {
                            (*(*tp).$field).$entry = (*(*base).$field).$entry;
                        }
                    )*
                }
            }
        };
    }
    suite!(tp_as_buffer, bf_getbuffer, bf_releasebuffer);
    suite!(
        tp_as_number,
        nb_add,
        nb_subtract,
        nb_multiply,
        nb_remainder,
        nb_divmod,
        nb_power,
        nb_negative,
        nb_positive,
        nb_absolute,
        nb_bool,
        nb_invert,
        nb_lshift,
        nb_rshift,
        nb_and,
        nb_xor,
        nb_or,
        nb_int,
        nb_float,
        nb_inplace_add,
        nb_inplace_subtract,
        nb_inplace_multiply,
        nb_inplace_remainder,
        nb_inplace_power,
        nb_inplace_lshift,
        nb_inplace_rshift,
        nb_inplace_and,
        nb_inplace_xor,
        nb_inplace_or,
        nb_floor_divide,
        nb_true_divide,
        nb_inplace_floor_divide,
        nb_inplace_true_divide,
        nb_index,
        nb_matrix_multiply,
        nb_inplace_matrix_multiply,
    );
    suite!(tp_as_async, am_await, am_aiter, am_anext);
    suite!(
        tp_as_sequence,
        sq_length,
        sq_concat,
        sq_repeat,
        sq_item,
        sq_ass_item,
        sq_contains,
        sq_inplace_concat,
        sq_inplace_repeat,
    );
    suite!(tp_as_mapping, mp_length, mp_subscript, mp_ass_subscript);
    // `fill_interpreter_slots` runs after this and installs the trampoline in
    // `tp_call`, so the vectorcall inheritance has to happen here.
    inherit_vectorcall(tp, base);
}

/// `true` when `tp` is a type whose storage is the mirror layer's to release —
/// the predicate `type_dealloc` and `_dealloc` both branch on
/// (`typeobject.py:716`, `object.py`).  allow-line-citation
pub(super) fn is_heap_type(tp: *mut CPyTypeObject) -> bool {
    !tp.is_null() && unsafe { (*tp).tp_flags.contains(TpFlags::PY_TPFLAGS_HEAPTYPE) }
}

/// `true` when a mirror is a `PyModuleDef` rather than a linked object.
pub(super) fn is_module_def(raw: *mut CPyObject) -> bool {
    !raw.is_null() && unsafe { std::ptr::eq((*raw).ob_type, &raw mut CPY_MODULE_DEF_TYPE) }
}

/// The interpreter type a `PyTypeObject` stands for, or null before
/// `PyType_Ready`.
pub(super) fn interpreter_type(tp: *mut CPyTypeObject) -> PyObjectRef {
    if tp.is_null() {
        return pyre_object::PY_NULL;
    }
    unsafe { pyobject::from_ref(&raw mut (*tp).ob_base.ob_base) }
}

/// `Py_tp_vectorcall` — the class object's own call, which answers `Type(...)`
/// instead of `__new__`/`__init__`.
///
/// Only a type `w_type_has_vectorcall` reports arrives here, so the mirror is
/// the extension's own static or the block `PyType_FromSpec` leaked: both
/// outlive the call, and neither is synthesized on demand.
///
/// `vectorcallfunc` takes the callable, the values, the count and the keyword
/// names in that order, which is `METH_FASTCALL | METH_KEYWORDS` — the count
/// goes over without `PY_VECTORCALL_ARGUMENTS_OFFSET`, the bit a caller sets
/// only to lend the callee the slot in front of the vector.
pub fn type_vectorcall(
    w_type: PyObjectRef,
    positional: &[PyObjectRef],
    keywords: &[(String, PyObjectRef)],
) -> Result<PyObjectRef, crate::PyError> {
    let tp = pyobject::as_pyobj(w_type) as *mut CPyTypeObject;
    if tp.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "type declares a vectorcall but has no mirror",
        ));
    }
    super::call_cfunction(
        unsafe { (*tp).tp_vectorcall },
        super::methodobject::MethFlags::METH_FASTCALL
            | super::methodobject::MethFlags::METH_KEYWORDS,
        w_type,
        positional,
        keywords,
    )
}

// ── slot lookup ─────────────────────────────────────────────────────────

/// The four numbers `type_members` publishes off a `PyTypeObject`, for a type
/// an extension declared in C, or `None` for one this runtime defines.
///
/// Read live rather than recorded when the type was readied: an extension
/// widens the fields after `PyType_Ready` returns, which is where Cython puts
/// the offset of a `cdef object __weakref__`, and the widened numbers are the
/// ones a reader is owed.
///
/// The nearest ancestor that declares them answers for a class derived in
/// Python, which is the walk `inherit_special` performs once per field as
/// `if (type->tp_basicsize == 0) type->tp_basicsize = base->tp_basicsize` and
/// `COPYVAL(tp_itemsize)` / `COPYVAL(tp_dictoffset)` /
/// `COPYVAL(tp_weaklistoffset)`.
pub(crate) fn declared_type_layout(
    w_type: PyObjectRef,
) -> Option<crate::typedef::DeclaredTypeLayout> {
    let mut w_current = w_type;
    while !w_current.is_null() {
        // The mirror as it stands, never a synthesized one: a type that has
        // not crossed into C declares nothing, and minting a block for it
        // here would answer with the very numbers being asked for.
        let tp = pyobject::as_pyobj(w_current) as *mut CPyTypeObject;
        // A mirror this layer filled describes a type pyre defines, so its
        // fields are what [`describe_interpreter_type`] synthesized rather
        // than a declaration. `TYPE_NAMES` holds the finished ones. A block
        // whose name is set while every size field is still zero is the
        // mirror `type_attach` has named but not yet sized -- not a
        // declaration either.
        let declared = !tp.is_null()
            && unsafe { !(*tp).tp_name.is_null() }
            && unsafe {
                (*tp).tp_basicsize != 0
                    || (*tp).tp_itemsize != 0
                    || (*tp).tp_dictoffset != 0
                    || (*tp).tp_weaklistoffset != 0
            }
            && !TYPE_NAMES.lock().contains_key(&(tp as usize));
        if declared {
            return Some(unsafe {
                crate::typedef::DeclaredTypeLayout {
                    basicsize: (*tp).tp_basicsize as i64,
                    itemsize: (*tp).tp_itemsize as i64,
                    dictoffset: (*tp).tp_dictoffset as i64,
                    weaklistoffset: (*tp).tp_weaklistoffset as i64,
                }
            });
        }
        w_current = unsafe { pyre_object::typeobject::w_type_get_best_base(w_current) };
    }
    None
}

/// The `PyTypeObject` of `w_type` and of each of its bases, nearest first.
///
/// A pyre type contributes a synthesized mirror whose slots are all null, so
/// walking the whole MRO costs nothing and lets a Python subclass of a
/// C-defined type reach its base's slots.
fn c_bases(w_type: PyObjectRef) -> Vec<*mut CPyTypeObject> {
    let mut out = Vec::new();
    let mro = unsafe { pyre_object::w_type_get_mro(w_type) };
    if mro.is_null() {
        let raw = pyobject::as_pyobj(w_type) as *mut CPyTypeObject;
        if !raw.is_null() {
            out.push(raw);
        }
        return out;
    }
    for &w_base in unsafe { (*mro).as_slice() } {
        let raw = pyobject::as_pyobj(w_base) as *mut CPyTypeObject;
        if !raw.is_null() {
            out.push(raw);
        }
    }
    out
}

/// A dying mirror's `tp_dealloc`, read off its own `ob_type`.
///
/// `cpyext/src/object.c:75-78 _Py_Dealloc` reads `obj->ob_type` directly and
/// performs no MRO walk: by the time a mirror is being deallocated its link is
/// already the deallocating marker, so there is no interpreter type to walk.
/// `PyType_Ready`'s `inherit_slots` is what put a base's `tp_dealloc` on the
/// subtype, so the single read finds it.
///
/// # Safety
/// `raw` must be a live mirror.
pub(super) unsafe fn tp_dealloc_of(raw: *mut CPyObject) -> Option<*const c_void> {
    let tp = unsafe { (*raw).ob_type };
    if tp.is_null() {
        return None;
    }
    let slot = unsafe { (*tp).tp_dealloc };
    if slot.is_null() { None } else { Some(slot) }
}

/// The nearest non-null `pick`ed slot along `w_type`'s MRO.
fn find_slot(w_type: PyObjectRef, pick: fn(*mut CPyTypeObject) -> *const c_void) -> *const c_void {
    for raw in c_bases(w_type) {
        let slot = pick(raw);
        if !slot.is_null() {
            return slot;
        }
    }
    std::ptr::null()
}

pub(super) fn slot_of(
    w_obj: PyObjectRef,
    pick: fn(*mut CPyTypeObject) -> *const c_void,
) -> *const c_void {
    match crate::typedef::r#type(w_obj) {
        Some(w_type) => find_slot(w_type.as_ptr(), pick),
        None => std::ptr::null(),
    }
}

/// The C block that carries a C-defined instance's fields.
///
/// `tp_alloc` built it, so it is already in the identity table; the fallback
/// covers an instance some other path produced and gives it the zero-filled
/// block its type declares.
pub(super) fn instance_block(w_self: PyObjectRef) -> *mut CPyObject {
    // Borrowed, not owned: the block's owner is the link, and the caller
    // reached this through `w_self`, which keeps the instance — and therefore
    // the block — alive for as long as it holds it.
    pyobject::borrow_mirror(w_self)
}

// ── the descriptors a method / member / getset table becomes ────────────

/// Reserved carrier keys, namespaced as `methodobject`'s are.
const DEF_KEY: &str = "__pyre_def__";
const NAME_KEY: &str = "__name__";
const QUALNAME_KEY: &str = "__qualname__";
const DOC_KEY: &str = "__doc__";
const OBJCLASS_KEY: &str = "__objclass__";

fn carrier_get(carrier: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    let dict = crate::baseobjspace::getdict_native(carrier);
    if dict.is_null() {
        return None;
    }
    unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(dict, key) }
}

fn carrier_set(carrier: PyObjectRef, key: &str, value: PyObjectRef) {
    let dict = crate::baseobjspace::getdict_native(carrier);
    if !dict.is_null() {
        unsafe { pyre_object::dictmultiobject::w_dict_setitem_str(dict, key, value) };
    }
}

fn carrier_def(carrier: PyObjectRef) -> usize {
    carrier_get(carrier, DEF_KEY)
        .filter(|&value| unsafe { pyre_object::is_int(value) })
        .map(|value| unsafe { pyre_object::w_int_get_value(value) } as usize)
        .unwrap_or(0)
}

fn text_or_none(pointer: *const c_char) -> PyObjectRef {
    if pointer.is_null() {
        return pyre_object::w_none();
    }
    pyre_object::w_str_new_managed(&unsafe { std::ffi::CStr::from_ptr(pointer) }.to_string_lossy())
}

/// Build one descriptor carrier: the definition address plus the names every
/// descriptor answers with.
fn new_carrier(
    carrier_type: PyObjectRef,
    definition: usize,
    name: *const c_char,
    doc: *const c_char,
    w_class: PyObjectRef,
) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let class_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_class);
    let carrier = pyre_object::w_instance_new(carrier_type);
    let carrier_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(carrier);
    let reload = |slot| pyre_object::gc_roots::shadow_stack_get(slot);

    let w_name = text_or_none(name);
    carrier_set(reload(carrier_slot), NAME_KEY, w_name);
    let w_name = carrier_get(reload(carrier_slot), NAME_KEY).unwrap_or_else(pyre_object::w_none);
    carrier_set(reload(carrier_slot), QUALNAME_KEY, w_name);
    let w_doc = text_or_none(doc);
    carrier_set(reload(carrier_slot), DOC_KEY, w_doc);
    carrier_set(reload(carrier_slot), OBJCLASS_KEY, reload(class_slot));
    let w_def = pyre_object::w_int_new(definition as i64);
    carrier_set(reload(carrier_slot), DEF_KEY, w_def);
    reload(carrier_slot)
}

/// `methodobject.py PyDescr_NewMethod` — the descriptor a `tp_methods` row
/// of `w_type` becomes, built on demand for a caller that has the row.
pub(super) fn new_method_descriptor(
    carrier_type: PyObjectRef,
    w_type: PyObjectRef,
    method: *mut super::methodobject::CPyMethodDef,
) -> PyObjectRef {
    new_carrier(
        carrier_type,
        method as usize,
        unsafe { (*method).ml_name },
        unsafe { (*method).ml_doc },
        w_type,
    )
}

fn descriptor_type(
    cell: &pyre_object::gc_roots::RootedOnceRef,
    name: &'static str,
    init: fn(PyObjectRef),
) -> PyObjectRef {
    cell.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type(name, |ns| {
            init(ns);
            super::methodobject::install_attribute_fence(ns);
        });
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp
    })
}

static METHOD_DESCRIPTOR_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();
static CLASSMETHOD_DESCRIPTOR_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();
static MEMBER_DESCRIPTOR_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();
static GETSET_DESCRIPTOR_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

/// `methodobject.py:W_PyCMethodObject` — a `tp_methods` row.
///
/// Unlike the module-level carrier it is a descriptor: the receiver is the
/// instance the attribute was read through, which `__get__` binds.
pub(super) fn method_descriptor_type() -> PyObjectRef {
    descriptor_type(&METHOD_DESCRIPTOR_TYPE, "method_descriptor", |ns| unsafe {
        let _root_scope = pyre_object::gc_roots::push_roots();
        let ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(ns);
        crate::__pyre_put_new!(
            ns_slot,
            "__get__",
            crate::make_builtin_function("__get__", method_descr_get)
        );
        crate::__pyre_put_new!(
            ns_slot,
            "__call__",
            crate::make_builtin_function("__call__", method_descr_call)
        );
        crate::__pyre_put_new!(
            ns_slot,
            "__repr__",
            crate::make_builtin_function_with_arity("__repr__", method_descr_repr, 1)
        );
    })
}

/// `methodobject.py W_PyCClassMethodObject` — a definition reached through
/// the class rather than through an instance.
///
/// What it binds is the whole of the difference from
/// [`method_descriptor_type`], so the two share the row they were built from.
pub(super) fn classmethod_descriptor_type() -> PyObjectRef {
    descriptor_type(
        &CLASSMETHOD_DESCRIPTOR_TYPE,
        "classmethod_descriptor",
        |ns| unsafe {
            let _root_scope = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ns);
            crate::__pyre_put_new!(
                ns_slot,
                "__get__",
                crate::make_builtin_function("__get__", descr_classmethod_get)
            );
            crate::__pyre_put_new!(
                ns_slot,
                "__call__",
                crate::make_builtin_function("__call__", classmethod_descr_call)
            );
            crate::__pyre_put_new!(
                ns_slot,
                "__repr__",
                // `PyClassMethodDescr_Type` names `method_repr` too: the two
                // descriptors report themselves the same way.
                crate::make_builtin_function_with_arity("__repr__", method_descr_repr, 1)
            );
        },
    )
}

fn member_descriptor_type() -> PyObjectRef {
    descriptor_type(&MEMBER_DESCRIPTOR_TYPE, "member_descriptor", |ns| unsafe {
        let _root_scope = pyre_object::gc_roots::push_roots();
        let ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(ns);
        crate::__pyre_put_new!(
            ns_slot,
            "__get__",
            crate::make_builtin_function("__get__", member_descr_get)
        );
        crate::__pyre_put_new!(
            ns_slot,
            "__set__",
            crate::make_builtin_function_with_arity("__set__", member_descr_set, 3)
        );
        crate::__pyre_put_new!(
            ns_slot,
            "__repr__",
            crate::make_builtin_function_with_arity("__repr__", member_descr_repr, 1)
        );
    })
}

fn getset_descriptor_type() -> PyObjectRef {
    descriptor_type(&GETSET_DESCRIPTOR_TYPE, "getset_descriptor", |ns| unsafe {
        let _root_scope = pyre_object::gc_roots::push_roots();
        let ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(ns);
        crate::__pyre_put_new!(
            ns_slot,
            "__get__",
            crate::make_builtin_function("__get__", getset_descr_get)
        );
        crate::__pyre_put_new!(
            ns_slot,
            "__set__",
            crate::make_builtin_function_with_arity("__set__", getset_descr_set, 3)
        );
        crate::__pyre_put_new!(
            ns_slot,
            "__repr__",
            crate::make_builtin_function_with_arity("__repr__", getset_descr_repr, 1)
        );
    })
}

fn descriptor_name(carrier: PyObjectRef) -> String {
    carrier_get(carrier, NAME_KEY)
        .filter(|&name| unsafe { pyre_object::unicodeobject::is_str(name) })
        .map(|name| unsafe { pyre_object::w_str_get_wtf8(name) }.to_string())
        .unwrap_or_else(|| "?".to_string())
}

/// The owner's `_PyType_Name` and the descriptor's own name, which is how a
/// message naming the unbound method spells it.
fn descriptor_qualname(carrier: PyObjectRef) -> String {
    let name = descriptor_name(carrier);
    let Some(owner) = carrier_objclass(carrier) else {
        return name;
    };
    let owner = unsafe { pyre_object::w_type_get_name(owner) };
    let owner = owner.rsplit('.').next().unwrap_or(owner);
    format!("{owner}.{name}")
}

/// The class to hand a `METH_METHOD` definition, which is the type the
/// descriptor was declared in.
///
/// A row without the flag takes none: the carrier a class is handed derives
/// from `PyCMethod_Type`, and `method_get` binds every other row as a plain
/// `PyCFunction`.
fn defining_class(
    carrier: PyObjectRef,
    method: *mut super::methodobject::CPyMethodDef,
) -> Option<PyObjectRef> {
    if !unsafe { (*method).ml_flags }.contains(super::methodobject::MethFlags::METH_METHOD) {
        return None;
    }
    carrier_objclass(carrier)
}

/// `descrobject.c` spells each kind differently: `method`, `member` and
/// `attribute` for the getset.
fn descr_repr(args: &[PyObjectRef], kind: &str) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    let owner = owner_name(carrier);
    Ok(pyre_object::w_str_new_managed(&format!(
        "<{kind} '{}' of '{owner}' objects>",
        descriptor_name(carrier)
    )))
}

/// The name of the type a descriptor was declared in, which is what its repr
/// and every refusal below name it by.
fn owner_name(carrier: PyObjectRef) -> String {
    carrier_objclass(carrier)
        .map(|owner| unsafe { pyre_object::typeobject::w_type_get_name(owner) }.to_string())
        .unwrap_or_else(|| "?".to_string())
}

fn method_descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    descr_repr(args, "method")
}

fn member_descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    descr_repr(args, "member")
}

fn getset_descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    descr_repr(args, "attribute")
}

/// The type a descriptor was declared in, when one was stamped on it.
///
/// `new_carrier` seeds the key with whatever class it was handed, and
/// `stamp_objclass` replaces it once the type is ready, so a carrier reached
/// before that answers `None`.
fn carrier_objclass(carrier: PyObjectRef) -> Option<PyObjectRef> {
    carrier_get(carrier, OBJCLASS_KEY)
        .filter(|&owner| !owner.is_null() && unsafe { pyre_object::is_type(owner) })
}

/// `descrobject.c descr_check` — a descriptor only applies to an instance of
/// the type it was declared in.
///
/// The definition it carries is an offset into that type's block, or a
/// function that casts the receiver to it, so a receiver of any other type
/// would be read and written at addresses that belong to something else.
fn descr_check(carrier: PyObjectRef, instance: PyObjectRef) -> Result<(), crate::PyError> {
    let Some(owner) = carrier_objclass(carrier) else {
        debug_assert!(
            false,
            "a cpyext descriptor reached a call without an __objclass__"
        );
        return Ok(());
    };
    if unsafe { crate::baseobjspace::issubtype_w((*instance).w_class, owner) } {
        return Ok(());
    }
    let received = unsafe { pyre_object::w_type_get_name((*instance).w_class) };
    Err(crate::PyError::type_error(format!(
        "descriptor '{}' for '{}' objects doesn't apply to a '{received}' object",
        descriptor_name(carrier),
        unsafe { pyre_object::w_type_get_name(owner) },
    )))
}

/// The receiver a `__get__(self, instance, owner)` call names, or `None` when
/// the attribute was read off the class.
fn bound_instance(args: &[PyObjectRef]) -> Option<PyObjectRef> {
    let instance = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    if instance.is_null() || unsafe { pyre_object::is_none(instance) } {
        return None;
    }
    Some(instance)
}

fn method_descr_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    let Some(instance) = bound_instance(args) else {
        return Ok(carrier);
    };
    descr_check(carrier, instance)?;
    let method = carrier_def(carrier) as *mut super::methodobject::CPyMethodDef;
    if method.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext method descriptor lost its definition",
        ));
    }
    // `method_get` binds with no module, so a bound method answers
    // `__module__` with `None`.
    super::methodobject::new_pycfunction_in_class(
        method,
        instance,
        pyre_object::w_none(),
        defining_class(carrier, method),
    )
}

/// The class a class-method descriptor binds, having checked that the
/// descriptor applies to it — `descrobject.c classmethod_get`'s three
/// refusals, in the order it makes them.
///
/// `named` is the `type` argument of `__get__`; where the call left it out,
/// the instance's own class is what the descriptor was reached through.
fn classmethod_owner(
    carrier: PyObjectRef,
    named: PyObjectRef,
    instance: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    let owner = match !named.is_null() && !unsafe { pyre_object::is_none(named) } {
        true => named,
        false => match instance {
            Some(instance) => unsafe { (*instance).w_class },
            None => {
                return Err(crate::PyError::type_error(format!(
                    "descriptor '{}' for type '{}' needs either an object or a type",
                    descriptor_name(carrier),
                    owner_name(carrier)
                )));
            }
        },
    };
    if !unsafe { pyre_object::is_type(owner) } {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{}' for type '{}' needs a type, not a '{}' as arg 2",
            descriptor_name(carrier),
            owner_name(carrier),
            crate::type_methods::arg_type_name(owner)
        )));
    }
    if let Some(declared) = carrier_objclass(carrier)
        && !unsafe { crate::baseobjspace::issubtype_w(owner, declared) }
    {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{}' requires a subtype of '{}' but received '{}'",
            descriptor_name(carrier),
            owner_name(carrier),
            unsafe { pyre_object::typeobject::w_type_get_name(owner) }
        )));
    }
    Ok(owner)
}

/// The row behind a class-method descriptor, or the error a carrier that lost
/// it answers with.
fn classmethod_def(
    carrier: PyObjectRef,
) -> Result<*mut super::methodobject::CPyMethodDef, crate::PyError> {
    let method = carrier_def(carrier) as *mut super::methodobject::CPyMethodDef;
    match method.is_null() {
        true => Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext class method descriptor lost its definition",
        )),
        false => Ok(method),
    }
}

fn descr_classmethod_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    let named = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
    let owner = classmethod_owner(carrier, named, bound_instance(args))?;
    let method = classmethod_def(carrier)?;
    super::methodobject::new_pycfunction_in_class(
        method,
        owner,
        pyre_object::w_none(),
        defining_class(carrier, method),
    )
}

/// `descr.__call__(cls, *args)`, the unbound spelling.
fn classmethod_descr_call(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(&args[1..]);
    let Some(&named) = positional.first() else {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{}' of '{}' object needs an argument",
            descriptor_name(carrier),
            owner_name(carrier)
        )));
    };
    let owner = classmethod_owner(carrier, named, None)?;
    let method = classmethod_def(carrier)?;
    super::methodobject::call_method_def_in_class(
        method,
        owner,
        defining_class(carrier, method),
        &positional[1..],
        kwargs,
    )
}

/// `descr.__call__(instance, *args)`, the unbound spelling.
fn method_descr_call(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(&args[1..]);
    let Some(&instance) = positional.first() else {
        return Err(crate::PyError::type_error(format!(
            "unbound method {}() needs an argument",
            descriptor_qualname(carrier)
        )));
    };
    descr_check(carrier, instance)?;
    let method = carrier_def(carrier) as *mut super::methodobject::CPyMethodDef;
    if method.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext method descriptor lost its definition",
        ));
    }
    super::methodobject::call_method_def_in_class(
        method,
        instance,
        defining_class(carrier, method),
        &positional[1..],
        kwargs,
    )
}

// ── `tp_members` ────────────────────────────────────────────────────────

/// The `structmember.h` type codes.
///
/// A code is not a flag -- a member names exactly one -- and C writes the
/// field, so these stay numbers rather than becoming an `enum` a stray value
/// could not inhabit.  Declaring each once mints the table
/// `every_member_type_code_is_the_number_the_header_gives_it` walks, so a code
/// added here is compared without anyone remembering to list it.
macro_rules! member_type_codes {
    ($($(#[$doc:meta])* $name:ident = $value:expr,)*) => {
        $($(#[$doc])* pub const $name: c_int = $value;)*

        #[cfg(test)]
        const ALL_MEMBER_TYPE_CODES: &[(&str, c_int)] = &[$((stringify!($name), $value),)*];
    };
}

member_type_codes! {
    T_SHORT = 0,
    T_INT = 1,
    T_LONG = 2,
    T_FLOAT = 3,
    T_DOUBLE = 4,
    T_STRING = 5,
    /// The string is the storage itself rather than a pointer to it.
    T_STRING_INPLACE = 13,
    T_OBJECT = 6,
    T_CHAR = 7,
    T_BYTE = 8,
    T_UBYTE = 9,
    T_USHORT = 10,
    T_UINT = 11,
    T_ULONG = 12,
    T_BOOL = 14,
    T_OBJECT_EX = 16,
    T_LONGLONG = 17,
    T_ULONGLONG = 18,
    T_PYSSIZET = 19,
    T_NONE = 20,
}
bitflags::bitflags! {
    /// The `object.h` flags a `PyMemberDef` row carries.
    ///
    /// Two places spell this table: the header an extension compiles against
    /// and this declaration.  `every_member_flag_is_the_bit_the_header_gives_it`
    /// compares the two, walking `Flags::FLAGS` rather than a list somebody
    /// has to remember to extend.  The names are the header's, uppercased;
    /// `structmember.h` spells the first one `READONLY` as well.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct MemberFlags: c_int {
        const PY_READONLY = 1;
        /// Reading the member emits `object.__getattr__`.
        const PY_AUDIT_READ = 2;
        /// `offset` counts from the extra data a negative `basicsize` asked
        /// for, not from the block.  [`from_spec`] resolves it.
        const PY_RELATIVE_OFFSET = 8;
    }
}

/// C writes a member's `flags` through its own `int` declaration, so the
/// mirror's field has to be that word and nothing wider or narrower.
const _: () = assert!(size_of::<MemberFlags>() == size_of::<c_int>());
const _: () = assert!(align_of::<MemberFlags>() == align_of::<c_int>());

/// A member still carrying [`MemberFlags::PY_RELATIVE_OFFSET`] has an `offset` its
/// reader cannot use: nothing but the type it was built for knows where the
/// extra data starts.  A table declared statically has no extra data at all,
/// so one that sets the flag is a declaration error.
fn reject_relative_offset(member: *mut CPyMemberDef, entry: &str) -> Result<(), crate::PyError> {
    if !unsafe { (*member).flags }.contains(MemberFlags::PY_RELATIVE_OFFSET) {
        return Ok(());
    }
    Err(crate::PyError::new(
        crate::PyErrorKind::SystemError,
        format!("{entry} used with Py_RELATIVE_OFFSET"),
    ))
}

fn member_address(w_self: PyObjectRef, member: *mut CPyMemberDef) -> *mut u8 {
    let block = instance_block(w_self);
    if block.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (block as *mut u8).offset((*member).offset) }
}

fn read_member(
    w_self: PyObjectRef,
    member: *mut CPyMemberDef,
) -> Result<PyObjectRef, crate::PyError> {
    reject_relative_offset(member, "PyMember_GetOne")?;
    let address = member_address(w_self, member);
    if address.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext member read on an object with no C storage",
        ));
    }
    let name = || {
        unsafe { std::ffi::CStr::from_ptr((*member).name) }
            .to_string_lossy()
            .into_owned()
    };
    let value = unsafe {
        match (*member).type_code {
            T_BOOL => pyre_object::boolobject::w_bool_from(*(address as *const i8) != 0),
            T_BYTE => pyre_object::w_int_new(*(address as *const i8) as i64),
            T_UBYTE => pyre_object::w_int_new(*(address as *const u8) as i64),
            T_SHORT => pyre_object::w_int_new(*(address as *const i16) as i64),
            T_USHORT => pyre_object::w_int_new(*(address as *const u16) as i64),
            T_INT => pyre_object::w_int_new(*(address as *const i32) as i64),
            T_UINT => pyre_object::w_int_new(*(address as *const u32) as i64),
            T_LONG | T_LONGLONG => pyre_object::w_int_new(*(address as *const i64)),
            T_ULONG | T_ULONGLONG => {
                let raw = *(address as *const u64);
                match i64::try_from(raw) {
                    Ok(value) => pyre_object::w_int_new(value),
                    Err(_) => {
                        return Err(crate::PyError::new(
                            crate::PyErrorKind::OverflowError,
                            "member value does not fit in a pyre int yet",
                        ));
                    }
                }
            }
            T_PYSSIZET => pyre_object::w_int_new(*(address as *const isize) as i64),
            T_FLOAT => pyre_object::w_float_new(*(address as *const f32) as f64),
            T_DOUBLE => pyre_object::w_float_new(*(address as *const f64)),
            T_CHAR => {
                pyre_object::w_str_new_managed(&(*(address as *const u8) as char).to_string())
            }
            T_STRING => {
                let text = *(address as *const *const c_char);
                text_or_none(text)
            }
            T_STRING_INPLACE => text_or_none(address as *const c_char),
            T_NONE => pyre_object::w_none(),
            T_OBJECT | T_OBJECT_EX => {
                let stored = *(address as *const *mut CPyObject);
                if stored.is_null() {
                    if (*member).type_code == T_OBJECT_EX {
                        return Err(crate::PyError::attribute_error(name()));
                    }
                    pyre_object::w_none()
                } else {
                    pyobject::from_ref(stored)
                }
            }
            _ => {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::SystemError,
                    "bad memberdescr type",
                ));
            }
        }
    };
    Ok(value)
}

fn write_member(
    w_self: PyObjectRef,
    member: *mut CPyMemberDef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    reject_relative_offset(member, "PyMember_SetOne")?;
    let name = || {
        unsafe { std::ffi::CStr::from_ptr((*member).name) }
            .to_string_lossy()
            .into_owned()
    };
    if unsafe { (*member).flags }.contains(MemberFlags::PY_READONLY) {
        return Err(crate::PyError::attribute_error("readonly attribute"));
    }
    let type_code = unsafe { (*member).type_code };
    // A string member is the storage, not a reference to it, so there is
    // nothing a write could give it -- and the refusal is a `TypeError`, which
    // is what tells it apart from the flag above.
    if matches!(type_code, T_STRING | T_STRING_INPLACE) {
        return Err(crate::PyError::type_error("readonly attribute"));
    }
    // Every branch below stores into the block, so the integer or float is
    // unwrapped first — the conversion can call back into Python and collect.
    let integer = matches!(
        type_code,
        T_BOOL
            | T_BYTE
            | T_UBYTE
            | T_SHORT
            | T_USHORT
            | T_INT
            | T_UINT
            | T_LONG
            | T_ULONG
            | T_LONGLONG
            | T_ULONGLONG
            | T_PYSSIZET
    );
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_self);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(value);
    let number = if integer {
        crate::baseobjspace::gateway_int_w(pyre_object::gc_roots::shadow_stack_get(value_slot))?
    } else {
        0
    };
    let real = if matches!(type_code, T_FLOAT | T_DOUBLE) {
        crate::baseobjspace::float_w(pyre_object::gc_roots::shadow_stack_get(value_slot))?
    } else {
        0.0
    };
    let w_self = pyre_object::gc_roots::shadow_stack_get(self_slot);
    let value = pyre_object::gc_roots::shadow_stack_get(value_slot);
    let address = member_address(w_self, member);
    if address.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext member write on an object with no C storage",
        ));
    }
    unsafe {
        match type_code {
            T_BOOL => *(address as *mut i8) = (number != 0) as i8,
            T_BYTE => *(address as *mut i8) = number as i8,
            T_UBYTE => *(address as *mut u8) = number as u8,
            T_SHORT => *(address as *mut i16) = number as i16,
            T_USHORT => *(address as *mut u16) = number as u16,
            T_INT => *(address as *mut i32) = number as i32,
            T_UINT => *(address as *mut u32) = number as u32,
            T_LONG | T_LONGLONG => *(address as *mut i64) = number,
            T_ULONG | T_ULONGLONG => *(address as *mut u64) = number as u64,
            T_PYSSIZET => *(address as *mut isize) = number as isize,
            T_FLOAT => *(address as *mut f32) = real as f32,
            T_DOUBLE => *(address as *mut f64) = real,
            T_CHAR => {
                let text = crate::baseobjspace::text0_wtf8_w(value)?;
                let bytes = text.as_bytes();
                if bytes.len() != 1 {
                    return Err(crate::PyError::type_error(
                        "a cpyext char member takes a string of length 1",
                    ));
                }
                *(address as *mut u8) = bytes[0];
            }
            T_OBJECT | T_OBJECT_EX => {
                let slot = address as *mut *mut CPyObject;
                let previous = *slot;
                *slot = pyobject::make_ref(value);
                pyobject::decref(previous);
            }
            _ => {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::SystemError,
                    "bad memberdescr type",
                ));
            }
        }
    }
    Ok(())
}

fn member_descr_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    let Some(instance) = bound_instance(args) else {
        return Ok(carrier);
    };
    descr_check(carrier, instance)?;
    let member = carrier_def(carrier) as *mut CPyMemberDef;
    if member.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext member descriptor lost its definition",
        ));
    }
    read_member(audit_member_read(instance, member)?, member)
}

/// `member_get`'s `object.__getattr__` event, which a `Py_AUDIT_READ` member
/// owes before its value is read.
///
/// The hooks are app code and so is the wrap of the member name, which makes
/// both collection points; the instance reaches them only as a copied pointer,
/// so it is rooted across them and read back out of its slot.
fn audit_member_read(
    instance: PyObjectRef,
    member: *mut CPyMemberDef,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { (*member).flags }.contains(MemberFlags::PY_AUDIT_READ)
        || !crate::module::sys::vm::audit_hooks_armed()
    {
        return Ok(instance);
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let slot = pyre_object::gc_roots::pin_roots(&[instance]);
    let w_name = text_or_none(unsafe { (*member).name });
    crate::module::sys::vm::audit(
        "object.__getattr__",
        &[pyre_object::gc_roots::shadow_stack_get(slot), w_name],
    )?;
    Ok(pyre_object::gc_roots::shadow_stack_get(slot))
}

fn member_descr_set(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    descr_check(carrier, args[1])?;
    let member = carrier_def(carrier) as *mut CPyMemberDef;
    if member.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext member descriptor lost its definition",
        ));
    }
    write_member(args[1], member, args[2])?;
    Ok(pyre_object::w_none())
}

// ── `tp_getset` ─────────────────────────────────────────────────────────

type Getter = unsafe extern "C" fn(*mut CPyObject, *mut c_void) -> *mut CPyObject;
type Setter = unsafe extern "C" fn(*mut CPyObject, *mut CPyObject, *mut c_void) -> c_int;

fn getset_descr_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    let Some(instance) = bound_instance(args) else {
        return Ok(carrier);
    };
    descr_check(carrier, instance)?;
    let getset = carrier_def(carrier) as *mut CPyGetSetDef;
    if getset.is_null() || unsafe { (*getset).get.is_null() } {
        return Err(crate::PyError::attribute_error(format!(
            "attribute '{}' of a cpyext object is not readable",
            descriptor_name(carrier)
        )));
    }
    let receiver = pyobject::make_ref(instance);
    let result = unsafe {
        let get: Getter = std::mem::transmute((*getset).get);
        get(receiver, (*getset).closure)
    };
    unsafe { pyobject::decref(receiver) };
    super::from_c_result(result)
}

fn getset_descr_set(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    descr_check(carrier, args[1])?;
    let getset = carrier_def(carrier) as *mut CPyGetSetDef;
    if getset.is_null() || unsafe { (*getset).set.is_null() } {
        return Err(crate::PyError::attribute_error(format!(
            "attribute '{}' of a cpyext object is not writable",
            descriptor_name(carrier)
        )));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(args[1]);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(args[2]);
    let receiver = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(self_slot));
    let value = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(value_slot));
    let result = unsafe {
        let set: Setter = std::mem::transmute((*getset).set);
        set(receiver, value, (*getset).closure)
    };
    unsafe {
        pyobject::decref(receiver);
        pyobject::decref(value);
    }
    if result != 0 {
        return Err(super::pyerrors::take_pending_error().unwrap_or_else(|| {
            crate::PyError::new(
                crate::PyErrorKind::SystemError,
                "a cpyext setter failed without setting an exception",
            )
        }));
    }
    Ok(pyre_object::w_none())
}

// ── the slot wrappers ───────────────────────────────────────────────────

/// The `(args, kwds)` pair a ternary slot takes, as owned references.
///
/// `call_cfunction` builds the same pair for `METH_VARARGS | METH_KEYWORDS`,
/// but a slot's result is not always a `PyObject *`, so the construction is
/// separated from the call here.
struct TernaryArgs {
    arguments: *mut CPyObject,
    keywords: *mut CPyObject,
}

impl Drop for TernaryArgs {
    fn drop(&mut self) {
        unsafe {
            pyobject::decref(self.arguments);
            pyobject::decref(self.keywords);
        }
    }
}

fn ternary_args(positional: &[PyObjectRef], keywords: &[(String, PyObjectRef)]) -> TernaryArgs {
    let roots = pyre_object::gc_roots::push_roots();
    let mut live = Vec::with_capacity(positional.len() + keywords.len());
    live.extend_from_slice(positional);
    for (_, value) in keywords {
        live.push(*value);
    }
    let base = roots.pin_roots(&live);
    let value_slot = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
    let items: Vec<PyObjectRef> = (0..positional.len()).map(value_slot).collect();
    let tuple = pyre_object::tupleobject::w_tuple_new(items);
    let tuple_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(tuple);
    let mut keywords_arg = std::ptr::null_mut();
    if !keywords.is_empty() {
        let dict = pyre_object::dictmultiobject::w_dict_new();
        let dict_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(dict);
        for (index, (name, _)) in keywords.iter().enumerate() {
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str(
                    pyre_object::gc_roots::shadow_stack_get(dict_slot),
                    name,
                    value_slot(positional.len() + index),
                )
            };
        }
        keywords_arg = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(dict_slot));
    }
    TernaryArgs {
        arguments: pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(tuple_slot)),
        keywords: keywords_arg,
    }
}

fn split_call(args: &[PyObjectRef]) -> (Vec<PyObjectRef>, Vec<(String, PyObjectRef)>) {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let keywords: Vec<(String, PyObjectRef)> = match kwargs {
        Some(dict) if crate::builtins::has_real_kwargs(kwargs) => unsafe {
            pyre_object::w_dict_str_entries(dict)
                .into_iter()
                .filter(|(key, _)| key != "__pyre_kw__")
                .collect()
        },
        _ => Vec::new(),
    };
    (positional.to_vec(), keywords)
}

/// `slot_tp_new` — `cls.__new__(cls, *args, **kwds)`.
fn slot_new(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args[0];
    let (positional, keywords) = split_call(&args[1..]);
    if slot.is_null() {
        return Err(crate::PyError::type_error(
            "cannot create instances of this cpyext type",
        ));
    }
    super::call_cfunction(
        slot,
        super::methodobject::MethFlags::METH_VARARGS
            | super::methodobject::MethFlags::METH_KEYWORDS,
        cls,
        &positional,
        &keywords,
    )
}

/// `slot_tp_init` — a slot whose result is an `int`, not an object.
fn slot_init(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = args[0];
    let (positional, keywords) = split_call(&args[1..]);
    if slot.is_null() {
        return Ok(pyre_object::w_none());
    }
    let built = ternary_args(&positional, &keywords);
    let receiver = pyobject::make_ref(w_self);
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyObject, *mut CPyObject) -> c_int =
            std::mem::transmute(slot);
        call(receiver, built.arguments, built.keywords)
    };
    unsafe { pyobject::decref(receiver) };
    if result != 0 {
        return Err(super::pyerrors::take_pending_error().unwrap_or_else(|| {
            crate::PyError::new(
                crate::PyErrorKind::SystemError,
                "a cpyext initializer failed without setting an exception",
            )
        }));
    }
    Ok(pyre_object::w_none())
}

fn slot_call(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = args[0];
    let (positional, keywords) = split_call(&args[1..]);
    if slot.is_null() {
        return Err(crate::PyError::type_error("cpyext object is not callable"));
    }
    // `_PyObject_VectorcallTstate`: the function the callable's type lends it
    // answers ahead of `tp_call`, and takes the values as they already lie.
    // The two routes end in the same function whenever `tp_call` is
    // `PyVectorcall_Call` -- but reaching it that way builds a tuple, and a
    // dict for the keywords, only for `_PyStack_UnpackDict` to take both apart
    // again on the other side.  Cython gives every compiled function and
    // `cdef` method this shape, so the round trip is most of what calling one
    // costs.
    //
    // `vectorcallfunc` takes the callable, the values, the count and the
    // keyword names, which is `METH_FASTCALL | METH_KEYWORDS`; the count goes
    // over without `PY_VECTORCALL_ARGUMENTS_OFFSET`, the bit a caller sets only
    // to lend the callee the slot in front of the vector.
    let vectorcall = unsafe { super::object::vectorcall_function(pyobject::as_pyobj(w_self)) };
    let (function, flags) = match vectorcall {
        Some(function) => (
            function as *const c_void,
            super::methodobject::MethFlags::METH_FASTCALL
                | super::methodobject::MethFlags::METH_KEYWORDS,
        ),
        None => (
            slot,
            super::methodobject::MethFlags::METH_VARARGS
                | super::methodobject::MethFlags::METH_KEYWORDS,
        ),
    };
    super::call_cfunction(function, flags, w_self, &positional, &keywords)
}

/// Run a `(self) -> PyObject *` slot.
fn unary_slot(
    slot: *const c_void,
    w_self: PyObjectRef,
    missing: &str,
) -> Result<PyObjectRef, crate::PyError> {
    if slot.is_null() {
        return Err(crate::PyError::type_error(format!(
            "cpyext object has no {missing}"
        )));
    }
    let receiver = pyobject::make_ref(w_self);
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject) -> *mut CPyObject =
            std::mem::transmute(slot);
        call(receiver)
    };
    unsafe { pyobject::decref(receiver) };
    super::from_c_result(result)
}

fn slot_repr(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    unary_slot(slot, args[0], "tp_repr")
}

fn slot_str(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    unary_slot(slot, args[0], "tp_str")
}

fn slot_iter(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    unary_slot(slot, args[0], "tp_iter")
}

/// `tp_iternext` reports exhaustion with NULL and no exception set, which is
/// the one place a NULL result is not an error.
fn slot_iternext(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = args[0];
    if slot.is_null() {
        return Err(crate::PyError::type_error(
            "cpyext object is not an iterator",
        ));
    }
    let receiver = pyobject::make_ref(w_self);
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject) -> *mut CPyObject =
            std::mem::transmute(slot);
        call(receiver)
    };
    unsafe { pyobject::decref(receiver) };
    if result.is_null() && !super::pyerrors::has_pending_error() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::StopIteration,
            String::new(),
        ));
    }
    super::from_c_result(result)
}

// ── the `tp_as_async` table ─────────────────────────────────────────────

/// One `PyAsyncMethods` slot, picked off a type that may declare no table.
macro_rules! async_slot {
    ($field:ident) => {
        (|tp: *mut CPyTypeObject| unsafe {
            let table = (*tp).tp_as_async;
            if table.is_null() {
                std::ptr::null()
            } else {
                (*table).$field
            }
        }) as fn(*mut CPyTypeObject) -> *const c_void
    };
}

fn slot_await(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    unary_slot(slot, args[0], "am_await")
}

fn slot_aiter(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    unary_slot(slot, args[0], "am_aiter")
}

/// `am_anext` ends an async iteration the way `tp_iternext` ends a synchronous
/// one, with `StopAsyncIteration` in place of `StopIteration`.
fn slot_anext(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = args[0];
    if slot.is_null() {
        return Err(crate::PyError::type_error(
            "cpyext object is not an async iterator",
        ));
    }
    let receiver = pyobject::make_ref(w_self);
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject) -> *mut CPyObject =
            std::mem::transmute(slot);
        call(receiver)
    };
    unsafe { pyobject::decref(receiver) };
    if result.is_null() && !super::pyerrors::has_pending_error() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::StopAsyncIteration,
            String::new(),
        ));
    }
    super::from_c_result(result)
}

fn slot_hash(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = args[0];
    if slot.is_null() {
        return Err(crate::PyError::type_error("cpyext object is not hashable"));
    }
    let receiver = pyobject::make_ref(w_self);
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject) -> isize = std::mem::transmute(slot);
        call(receiver)
    };
    unsafe { pyobject::decref(receiver) };
    if result == -1 && super::pyerrors::has_pending_error() {
        return Err(super::pyerrors::take_pending_error().expect("just observed"));
    }
    Ok(pyre_object::w_int_new(result as i64))
}

/// `Py_LT` .. `Py_GE`.
fn rich_compare(
    slot: *const c_void,
    args: &[PyObjectRef],
    operation: c_int,
) -> Result<PyObjectRef, crate::PyError> {
    let w_self = args[0];
    if slot.is_null() {
        return Ok(pyre_object::w_not_implemented());
    }
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_self);
    let other_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(args[1]);
    let receiver = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(self_slot));
    let other = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(other_slot));
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyObject, c_int) -> *mut CPyObject =
            std::mem::transmute(slot);
        call(receiver, other, operation)
    };
    unsafe {
        pyobject::decref(receiver);
        pyobject::decref(other);
    }
    super::from_c_result(result)
}

macro_rules! comparison_slots {
    ($($wrapper:ident => $dunder:literal, $operation:expr;)*) => {
        $(
            fn $wrapper(
                slot: *const c_void,
                args: &[PyObjectRef],
            ) -> Result<PyObjectRef, crate::PyError> {
                rich_compare(slot, args, $operation)
            }
        )*

        fn install_comparisons(ns: PyObjectRef, tp: *mut CPyTypeObject) {
            let slot = unsafe { (*tp).tp_richcompare };
            $(
                publish(ns, $dunder, 2, slot, $wrapper);
            )*
        }
    };
}

comparison_slots! {
    slot_lt => "__lt__", 0;
    slot_le => "__le__", 1;
    slot_eq => "__eq__", 2;
    slot_ne => "__ne__", 3;
    slot_gt => "__gt__", 4;
    slot_ge => "__ge__", 5;
}

// ── the protocol tables ─────────────────────────────────────────────────
//
// Each table slot becomes the dunder `slotdefs.py` names for it.  A wrapper
// re-derives the slot from the receiver's MRO, so a Python subclass of a
// C-defined type reaches the base's table through the namespace it inherits.

macro_rules! number_slot {
    ($field:ident) => {
        (|tp| unsafe {
            let table = (*tp).tp_as_number;
            if table.is_null() {
                std::ptr::null()
            } else {
                (*table).$field
            }
        }) as fn(*mut CPyTypeObject) -> *const c_void
    };
}

macro_rules! sequence_slot {
    ($field:ident) => {
        (|tp| unsafe {
            let table = (*tp).tp_as_sequence;
            if table.is_null() {
                std::ptr::null()
            } else {
                (*table).$field
            }
        }) as fn(*mut CPyTypeObject) -> *const c_void
    };
}

macro_rules! mapping_slot {
    ($field:ident) => {
        (|tp| unsafe {
            let table = (*tp).tp_as_mapping;
            if table.is_null() {
                std::ptr::null()
            } else {
                (*table).$field
            }
        }) as fn(*mut CPyTypeObject) -> *const c_void
    };
}

/// Run a `(PyObject *, PyObject *) -> PyObject *` slot.
fn call_binary(
    slot: *const c_void,
    first: PyObjectRef,
    second: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let first_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(first);
    let second_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(second);
    let left = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(first_slot));
    let right = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(second_slot));
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyObject) -> *mut CPyObject =
            std::mem::transmute(slot);
        call(left, right)
    };
    unsafe {
        pyobject::decref(left);
        pyobject::decref(right);
    }
    super::from_c_result(result)
}

/// `left op right` through `pick`, in the order the operands were written.
///
/// `nb_add` serves `__add__` and `__radd__` both, as `SLOT1BIN` does: the
/// reflected form hands the C function the pair the other way round, and a
/// table that does not handle it answers `NotImplemented` itself.
fn binary(
    slot: *const c_void,
    args: &[PyObjectRef],
    reflected: bool,
) -> Result<PyObjectRef, crate::PyError> {
    if slot.is_null() {
        return Ok(pyre_object::w_not_implemented());
    }
    let (first, second) = if reflected {
        (args[1], args[0])
    } else {
        (args[0], args[1])
    };
    call_binary(slot, first, second)
}

macro_rules! number_binaries {
    ($($field:ident, $direct_name:literal, $direct:ident, $reflected_name:literal, $reflected:ident;)*) => {
        $(
            fn $direct(
                slot: *const c_void,
                args: &[PyObjectRef],
            ) -> Result<PyObjectRef, crate::PyError> {
                binary(slot, args, false)
            }

            fn $reflected(
                slot: *const c_void,
                args: &[PyObjectRef],
            ) -> Result<PyObjectRef, crate::PyError> {
                binary(slot, args, true)
            }
        )*

        fn install_number_binaries(ns: PyObjectRef, tp: *mut CPyTypeObject) {
            $(
                let slot = (number_slot!($field))(tp);
                if !slot.is_null() {
                    publish(ns, $direct_name, 2, slot, $direct);
                    publish(ns, $reflected_name, 2, slot, $reflected);
                }
            )*
        }
    };
}

number_binaries! {
    nb_add, "__add__", nb_add_direct, "__radd__", nb_add_reflected;
    nb_subtract, "__sub__", nb_sub_direct, "__rsub__", nb_sub_reflected;
    nb_multiply, "__mul__", nb_mul_direct, "__rmul__", nb_mul_reflected;
    nb_remainder, "__mod__", nb_mod_direct, "__rmod__", nb_mod_reflected;
    nb_divmod, "__divmod__", nb_divmod_direct, "__rdivmod__", nb_divmod_reflected;
    nb_lshift, "__lshift__", nb_lshift_direct, "__rlshift__", nb_lshift_reflected;
    nb_rshift, "__rshift__", nb_rshift_direct, "__rrshift__", nb_rshift_reflected;
    nb_and, "__and__", nb_and_direct, "__rand__", nb_and_reflected;
    nb_xor, "__xor__", nb_xor_direct, "__rxor__", nb_xor_reflected;
    nb_or, "__or__", nb_or_direct, "__ror__", nb_or_reflected;
    nb_floor_divide, "__floordiv__", nb_floordiv_direct, "__rfloordiv__", nb_floordiv_reflected;
    nb_true_divide, "__truediv__", nb_truediv_direct, "__rtruediv__", nb_truediv_reflected;
    nb_matrix_multiply, "__matmul__", nb_matmul_direct, "__rmatmul__", nb_matmul_reflected;
}

macro_rules! number_inplace {
    ($($field:ident, $name:literal, $wrapper:ident;)*) => {
        $(
            fn $wrapper(
                slot: *const c_void,
                args: &[PyObjectRef],
            ) -> Result<PyObjectRef, crate::PyError> {
                binary(slot, args, false)
            }
        )*

        fn install_number_inplace(ns: PyObjectRef, tp: *mut CPyTypeObject) {
            $(
                let slot = (number_slot!($field))(tp);
                if !slot.is_null() {
                    publish(ns, $name, 2, slot, $wrapper);
                }
            )*
        }
    };
}

number_inplace! {
    nb_inplace_add, "__iadd__", nb_iadd;
    nb_inplace_subtract, "__isub__", nb_isub;
    nb_inplace_multiply, "__imul__", nb_imul;
    nb_inplace_remainder, "__imod__", nb_imod;
    nb_inplace_lshift, "__ilshift__", nb_ilshift;
    nb_inplace_rshift, "__irshift__", nb_irshift;
    nb_inplace_and, "__iand__", nb_iand;
    nb_inplace_xor, "__ixor__", nb_ixor;
    nb_inplace_or, "__ior__", nb_ior;
    nb_inplace_floor_divide, "__ifloordiv__", nb_ifloordiv;
    nb_inplace_true_divide, "__itruediv__", nb_itruediv;
    nb_inplace_matrix_multiply, "__imatmul__", nb_imatmul;
}

macro_rules! number_unaries {
    ($($field:ident, $name:literal, $wrapper:ident;)*) => {
        $(
            fn $wrapper(
                slot: *const c_void,
                args: &[PyObjectRef],
            ) -> Result<PyObjectRef, crate::PyError> {
                unary_slot(slot, args[0], stringify!($field))
            }
        )*

        fn install_number_unaries(ns: PyObjectRef, tp: *mut CPyTypeObject) {
            $(
                let slot = (number_slot!($field))(tp);
                if !slot.is_null() {
                    publish(ns, $name, 1, slot, $wrapper);
                }
            )*
        }
    };
}

number_unaries! {
    nb_negative, "__neg__", nb_neg;
    nb_positive, "__pos__", nb_pos;
    nb_absolute, "__abs__", nb_abs;
    nb_invert, "__invert__", nb_invert_wrapper;
    nb_int, "__int__", nb_int_wrapper;
    nb_float, "__float__", nb_float_wrapper;
    nb_index, "__index__", nb_index_wrapper;
}

/// `nb_bool` is an `inquiry`: a negative result is the error indicator.
fn nb_bool_wrapper(
    slot: *const c_void,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    if slot.is_null() {
        return Ok(pyre_object::boolobject::w_bool_from(true));
    }
    let receiver = pyobject::make_ref(args[0]);
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject) -> c_int = std::mem::transmute(slot);
        call(receiver)
    };
    unsafe { pyobject::decref(receiver) };
    if result < 0 {
        return Err(pending_or(
            "a cpyext truth test failed without setting an exception",
        ));
    }
    Ok(pyre_object::boolobject::w_bool_from(result != 0))
}

/// `nb_power` is a `ternaryfunc`, the third operand being the modulus `pow`
/// takes and every other route leaves as `None`.
fn power(
    slot: *const c_void,
    args: &[PyObjectRef],
    reflected: bool,
) -> Result<PyObjectRef, crate::PyError> {
    if slot.is_null() {
        return Ok(pyre_object::w_not_implemented());
    }
    let (first, second) = if reflected {
        (args[1], args[0])
    } else {
        (args[0], args[1])
    };
    let modulus = args.get(2).copied().unwrap_or_else(pyre_object::w_none);
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[first, second, modulus]);
    let owned =
        |index: usize| pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(base + index));
    let (left, right, third) = (owned(0), owned(1), owned(2));
    let result = unsafe {
        let call: unsafe extern "C" fn(
            *mut CPyObject,
            *mut CPyObject,
            *mut CPyObject,
        ) -> *mut CPyObject = std::mem::transmute(slot);
        call(left, right, third)
    };
    unsafe {
        pyobject::decref(left);
        pyobject::decref(right);
        pyobject::decref(third);
    }
    super::from_c_result(result)
}

fn nb_pow_direct(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    power(slot, args, false)
}

fn nb_pow_reflected(
    slot: *const c_void,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    power(slot, args, true)
}

fn nb_ipow(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if slot.is_null() {
        return Ok(pyre_object::w_not_implemented());
    }
    // The in-place slot has the same shape, so the direct path serves it.
    call_ternary(slot, args[0], args[1], args.get(2).copied())
}

fn call_ternary(
    slot: *const c_void,
    first: PyObjectRef,
    second: PyObjectRef,
    third: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[first, second, third.unwrap_or_else(pyre_object::w_none)]);
    let owned =
        |index: usize| pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(base + index));
    let (left, right, modulus) = (owned(0), owned(1), owned(2));
    let result = unsafe {
        let call: unsafe extern "C" fn(
            *mut CPyObject,
            *mut CPyObject,
            *mut CPyObject,
        ) -> *mut CPyObject = std::mem::transmute(slot);
        call(left, right, modulus)
    };
    unsafe {
        pyobject::decref(left);
        pyobject::decref(right);
        pyobject::decref(modulus);
    }
    super::from_c_result(result)
}

/// The pending C exception, or a `SystemError` naming the slot that returned
/// its failure indicator without setting one.
pub(super) fn pending_or(message: &str) -> crate::PyError {
    super::pyerrors::take_pending_error()
        .unwrap_or_else(|| crate::PyError::new(crate::PyErrorKind::SystemError, message))
}

// ── the sequence and mapping tables ─────────────────────────────────────

/// Run a `lenfunc`.
fn call_length(slot: *const c_void, w_self: PyObjectRef) -> Result<isize, crate::PyError> {
    let receiver = pyobject::make_ref(w_self);
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject) -> isize = std::mem::transmute(slot);
        call(receiver)
    };
    unsafe { pyobject::decref(receiver) };
    if result < 0 {
        return Err(pending_or(
            "a cpyext length slot failed without setting an exception",
        ));
    }
    Ok(result)
}

/// `mp_length` first, `sq_length` second -- the order `PyObject_Size` uses.
fn length_slot(w_self: PyObjectRef) -> *const c_void {
    let slot = slot_of(w_self, mapping_slot!(mp_length));
    if slot.is_null() {
        slot_of(w_self, sequence_slot!(sq_length))
    } else {
        slot
    }
}

fn slot_len(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if slot.is_null() {
        return Err(crate::PyError::type_error("cpyext object has no length"));
    }
    Ok(pyre_object::w_int_new(call_length(slot, args[0])? as i64))
}

/// The index `sq_item` and `sq_ass_item` take: `__index__` of the key, with a
/// negative value folded through the length as `wrap_sq_item` does.  The
/// length is the receiver's own, which is what the wrapper reads there too.
fn sequence_index(w_self: PyObjectRef, key: PyObjectRef) -> Result<isize, crate::PyError> {
    let mut index = crate::baseobjspace::getindex_w(key)? as isize;
    if index < 0 {
        let slot = length_slot(w_self);
        if !slot.is_null() {
            index += call_length(slot, w_self)?;
        }
    }
    Ok(index)
}

/// `mp_subscript` -- `wrap_binaryfunc`.
fn slot_subscript(
    slot: *const c_void,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    call_binary(slot, args[0], args[1])
}

/// `sq_item` -- `wrap_sq_item`, which reads the key as an index.
fn slot_item(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(args[0]);
    let key_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(args[1]);
    let index = sequence_index(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        pyre_object::gc_roots::shadow_stack_get(key_slot),
    )?;
    let receiver = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(self_slot));
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, isize) -> *mut CPyObject =
            std::mem::transmute(slot);
        call(receiver, index)
    };
    unsafe { pyobject::decref(receiver) };
    super::from_c_result(result)
}

/// Report what an assignment slot answered, which is a status and not a value.
fn assigned(failed: bool) -> Result<PyObjectRef, crate::PyError> {
    match failed {
        true => Err(pending_or(
            "a cpyext item assignment failed without setting an exception",
        )),
        false => Ok(pyre_object::w_none()),
    }
}

/// `mp_ass_subscript` -- `wrap_objobjargproc`, the deletion passing NULL.
fn assign_subscript(
    slot: *const c_void,
    args: &[PyObjectRef],
    value: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(args[0]);
    let key_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(args[1]);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(value.unwrap_or_else(pyre_object::w_none));
    let reload = |slot| pyre_object::gc_roots::shadow_stack_get(slot);

    let receiver = pyobject::make_ref(reload(self_slot));
    let index = pyobject::make_ref(reload(key_slot));
    let item = match value {
        Some(_) => pyobject::make_ref(reload(value_slot)),
        None => std::ptr::null_mut(),
    };
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyObject, *mut CPyObject) -> c_int =
            std::mem::transmute(slot);
        call(receiver, index, item)
    };
    unsafe {
        pyobject::decref(receiver);
        pyobject::decref(index);
        pyobject::decref(item);
    }
    assigned(result != 0)
}

/// `sq_ass_item` -- `wrap_sq_setitem` and `wrap_sq_delitem`.
fn assign_index(
    slot: *const c_void,
    args: &[PyObjectRef],
    value: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(args[0]);
    let key_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(args[1]);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(value.unwrap_or_else(pyre_object::w_none));
    let reload = |slot| pyre_object::gc_roots::shadow_stack_get(slot);

    let index = sequence_index(reload(self_slot), reload(key_slot))?;
    let receiver = pyobject::make_ref(reload(self_slot));
    let item = match value {
        Some(_) => pyobject::make_ref(reload(value_slot)),
        None => std::ptr::null_mut(),
    };
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, isize, *mut CPyObject) -> c_int =
            std::mem::transmute(slot);
        call(receiver, index, item)
    };
    unsafe {
        pyobject::decref(receiver);
        pyobject::decref(item);
    }
    assigned(result != 0)
}

fn slot_ass_subscript(
    slot: *const c_void,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    assign_subscript(slot, args, Some(args[2]))
}

fn slot_del_subscript(
    slot: *const c_void,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    assign_subscript(slot, args, None)
}

fn slot_ass_item(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assign_index(slot, args, Some(args[2]))
}

fn slot_del_item(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assign_index(slot, args, None)
}

fn slot_contains(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if slot.is_null() {
        return Err(crate::PyError::type_error(
            "cpyext object does not support membership tests",
        ));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(args[0]);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(args[1]);
    let receiver = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(self_slot));
    let value = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(value_slot));
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyObject) -> c_int =
            std::mem::transmute(slot);
        call(receiver, value)
    };
    unsafe {
        pyobject::decref(receiver);
        pyobject::decref(value);
    }
    if result < 0 {
        return Err(pending_or(
            "a cpyext membership test failed without setting an exception",
        ));
    }
    Ok(pyre_object::boolobject::w_bool_from(result != 0))
}

/// `sq_concat` and `sq_inplace_concat` are `binaryfunc`s like `nb_add`.
fn sq_concat_wrapper(
    slot: *const c_void,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    binary(slot, args, false)
}

/// `sq_repeat` takes the count as a `Py_ssize_t`, so `n * seq` and `seq * n`
/// reach it the same way.
fn repeat(
    slot: *const c_void,
    w_self: PyObjectRef,
    count: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if slot.is_null() {
        return Ok(pyre_object::w_not_implemented());
    }
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_self);
    let count_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(count);
    let times =
        crate::baseobjspace::getindex_w(pyre_object::gc_roots::shadow_stack_get(count_slot))?;
    let receiver = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(self_slot));
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, isize) -> *mut CPyObject =
            std::mem::transmute(slot);
        call(receiver, times as isize)
    };
    unsafe { pyobject::decref(receiver) };
    super::from_c_result(result)
}

/// `seq * n` and `n * seq` both repeat `seq`, so one wrapper serves both, and
/// the in-place slot has the same shape.
fn sq_repeat_wrapper(
    slot: *const c_void,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    repeat(slot, args[0], args[1])
}

/// Install everything the three tables contribute.
///
/// `__len__`, `__getitem__` and the item assignments are shared between the
/// mapping and sequence tables, so each is installed once and the wrapper
/// picks the slot; `__add__` and `__mul__` go to the number table when it has
/// them and to the sequence table otherwise, as `slotdefs.py` orders them.
fn install_protocols(ns: PyObjectRef, tp: *mut CPyTypeObject) {
    install_number_binaries(ns, tp);
    install_number_inplace(ns, tp);
    install_number_unaries(ns, tp);
    let of = |pick: fn(*mut CPyTypeObject) -> *const c_void| pick(tp);
    let bool_slot = of(number_slot!(nb_bool));
    if !bool_slot.is_null() {
        publish(ns, "__bool__", 1, bool_slot, nb_bool_wrapper);
    }
    let power_slot = of(number_slot!(nb_power));
    if !power_slot.is_null() {
        publish(
            ns,
            "__pow__",
            crate::gateway::HOPELESS,
            power_slot,
            nb_pow_direct,
        );
        publish(
            ns,
            "__rpow__",
            crate::gateway::HOPELESS,
            power_slot,
            nb_pow_reflected,
        );
    }
    let inplace_power = of(number_slot!(nb_inplace_power));
    if !inplace_power.is_null() {
        publish(
            ns,
            "__ipow__",
            crate::gateway::HOPELESS,
            inplace_power,
            nb_ipow,
        );
    }

    // `__len__`, `__getitem__` and the item assignments have a mapping form
    // and a sequence form of different shapes; the mapping one wins where the
    // type declares both, which is the order `slotdefs` lists them in.
    let mp_length = of(mapping_slot!(mp_length));
    let sq_length = of(sequence_slot!(sq_length));
    if !mp_length.is_null() || !sq_length.is_null() {
        let length = match mp_length.is_null() {
            true => sq_length,
            false => mp_length,
        };
        publish(ns, "__len__", 1, length, slot_len);
    }
    let subscript = of(mapping_slot!(mp_subscript));
    let item = of(sequence_slot!(sq_item));
    if !subscript.is_null() {
        publish(ns, "__getitem__", 2, subscript, slot_subscript);
    } else if !item.is_null() {
        publish(ns, "__getitem__", 2, item, slot_item);
    }
    let ass_subscript = of(mapping_slot!(mp_ass_subscript));
    let ass_item = of(sequence_slot!(sq_ass_item));
    if !ass_subscript.is_null() {
        publish(ns, "__setitem__", 3, ass_subscript, slot_ass_subscript);
        publish(ns, "__delitem__", 2, ass_subscript, slot_del_subscript);
    } else if !ass_item.is_null() {
        publish(ns, "__setitem__", 3, ass_item, slot_ass_item);
        publish(ns, "__delitem__", 2, ass_item, slot_del_item);
    }
    let contains = of(sequence_slot!(sq_contains));
    if !contains.is_null() {
        publish(ns, "__contains__", 2, contains, slot_contains);
    }
    // `__add__` and `__mul__` go to the number table when it has them and to
    // the sequence table otherwise, as `slotdefs.py` orders them.
    let concat = of(sequence_slot!(sq_concat));
    if !concat.is_null() && of(number_slot!(nb_add)).is_null() {
        publish(ns, "__add__", 2, concat, sq_concat_wrapper);
    }
    let inplace_concat = of(sequence_slot!(sq_inplace_concat));
    if !inplace_concat.is_null() && of(number_slot!(nb_inplace_add)).is_null() {
        publish(ns, "__iadd__", 2, inplace_concat, sq_concat_wrapper);
    }
    let sq_repeat = of(sequence_slot!(sq_repeat));
    if !sq_repeat.is_null() && of(number_slot!(nb_multiply)).is_null() {
        publish(ns, "__mul__", 2, sq_repeat, sq_repeat_wrapper);
        publish(ns, "__rmul__", 2, sq_repeat, sq_repeat_wrapper);
    }
    let inplace_repeat = of(sequence_slot!(sq_inplace_repeat));
    if !inplace_repeat.is_null() && of(number_slot!(nb_inplace_multiply)).is_null() {
        publish(ns, "__imul__", 2, inplace_repeat, sq_repeat_wrapper);
    }
}

// ── the block a descriptor is mirrored into ──────────────────────

/// A filled block, against the address of the [`CPyWrapperBase`] allocated for
/// it -- 0 for every family that has none.  An address rather than a pointer
/// because the table is shared across threads and a raw pointer is not.
type BlockSet = super::address_table::HeldMap<usize>;

/// The blocks [`descriptor_attach`] filled, and so the ones whose references
/// are this module's to release.
static DESCRIPTOR_BLOCKS: AddressTable<BlockSet> =
    AddressTable::new(BlockSet::with_hasher(std::hash::BuildHasherDefault::new()));

/// Whether `w_type` is one of this module's descriptor carriers.
///
/// A carrier that has not been built can have no instance to mirror, so the
/// cells are read rather than initialised: asking would build all four the
/// first time any type at all was described.
fn is_descriptor_carrier(w_type: PyObjectRef) -> bool {
    !w_type.is_null()
        && [
            &METHOD_DESCRIPTOR_TYPE,
            &CLASSMETHOD_DESCRIPTOR_TYPE,
            &MEMBER_DESCRIPTOR_TYPE,
            &GETSET_DESCRIPTOR_TYPE,
        ]
        .into_iter()
        .any(|cell| cell.get() == Some(w_type))
}

/// Whether `w_type` is `wrapper_descriptor`, whose blocks carry a slot and the
/// [`CPyWrapperBase`] beside it rather than a `PyMethodDef` row.
fn is_wrapper_carrier(w_type: PyObjectRef) -> bool {
    !w_type.is_null() && w_type == builtin_type(&crate::function::SLOT_WRAPPER_TYPE)
}

/// What `tp_basicsize` a descriptor carrier's mirror carries — the fields a
/// caller casting the block to one of the `PyDescrObject` shapes reads off it.
fn descriptor_basicsize(w_type: PyObjectRef) -> isize {
    if is_wrapper_carrier(w_type) {
        return size_of::<CPyWrapperDescrObject>() as isize;
    }
    match is_descriptor_carrier(w_type) {
        true => size_of::<CPyMethodDescrObject>() as isize,
        false => 0,
    }
}

/// Fill the common header every descriptor block opens with — `typeobject.py
/// init_descr`.
///
/// `d_qualname` is left as the allocation found it, which is where
/// `init_descr` leaves it.
fn init_descr(raw: *mut CPyObject, w_type: PyObjectRef, w_name: PyObjectRef) {
    // Minting either reference may collect and move the other, so both are
    // pinned and read back.
    let roots = pyre_object::gc_roots::push_roots();
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_type);
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_name);
    let reload = |slot| pyre_object::gc_roots::shadow_stack_get(slot);
    let d_type = pyobject::make_ref(reload(type_slot)) as *mut CPyTypeObject;
    let d_name = pyobject::make_ref(reload(name_slot));
    let descr = raw as *mut CPyDescrObject;
    unsafe {
        (*descr).d_type = d_type;
        (*descr).d_name = d_name;
    }
}

/// Fill a freshly allocated descriptor block — `typeobject.py
/// methoddescr_attach` and `wrapperdescr_attach`, which share `init_descr`
/// and differ in the one field past it.
pub(super) fn descriptor_attach(raw: *mut CPyObject, w_obj: PyObjectRef) {
    let w_class = unsafe { (*w_obj).w_class };
    let tp = unsafe { (*raw).ob_type };
    if tp.is_null() {
        return;
    }
    // The block was sized from the carrier's own mirror, so a block smaller
    // than the shape below is not that carrier's.
    let room = |wanted: usize| unsafe { (*tp).tp_basicsize } >= wanted as isize;

    let base = if is_wrapper_carrier(w_class) {
        if !room(size_of::<CPyWrapperDescrObject>()) {
            return;
        }
        let w_type = unsafe { crate::function::fget_func_objclass(w_obj) }.unwrap_or(PY_NULL);
        let w_name = unsafe { crate::function::fget_func_name(w_obj) };
        init_descr(raw, w_type, w_name);
        // Nothing here has anything to put in the block, but a source that
        // rewrites a slot's `__doc__` copies it before writing its own, so it
        // has to be there to be copied.
        let base = Box::into_raw(Box::new(CPyWrapperBase::empty()));
        let descr = raw as *mut CPyWrapperDescrObject;
        unsafe {
            (*descr).d_base = base;
            // The slot a wrapper the interpreter built for one of its own
            // types wraps is a body rather than an address an extension may
            // call, which is the same half of the split `pycfunction_type`
            // describes.
            (*descr).d_wrapped = std::ptr::null_mut();
        }
        base as usize
    } else {
        if !is_descriptor_carrier(w_class) || !room(size_of::<CPyMethodDescrObject>()) {
            return;
        }
        let definition = carrier_def(w_obj);
        init_descr(
            raw,
            carrier_objclass(w_obj).unwrap_or(PY_NULL),
            carrier_get(w_obj, NAME_KEY).unwrap_or(PY_NULL),
        );
        unsafe {
            (*(raw as *mut CPyMethodDescrObject)).d_method =
                definition as *mut super::methodobject::CPyMethodDef;
        }
        0
    };
    DESCRIPTOR_BLOCKS.lock().insert(hold(raw as usize), base);
}

/// Release what a descriptor block owns — `typeobject.py descr_dealloc` and
/// `wrapper_dealloc`, which frees the wrapper block on top of it.
pub(super) fn forget_descriptor_block(raw: *mut CPyObject) {
    let Some(base) = DESCRIPTOR_BLOCKS.take(raw as usize) else {
        return;
    };
    let descr = raw as *mut CPyDescrObject;
    unsafe {
        pyobject::decref((*descr).d_type as *mut CPyObject);
        pyobject::decref((*descr).d_name);
        (*descr).d_type = std::ptr::null_mut();
        (*descr).d_name = std::ptr::null_mut();
    }
    if base == 0 {
        return;
    }
    // The block is freed only while it is still the one that was allocated
    // here.  A source that rewrites a slot's `__doc__` leaves a block of its
    // own in the field, and that storage is the source's.
    let wrapper = raw as *mut CPyWrapperDescrObject;
    if unsafe { (*wrapper).d_base } as usize == base {
        unsafe { (*wrapper).d_base = std::ptr::null_mut() };
        drop(unsafe { Box::from_raw(base as *mut CPyWrapperBase) });
    }
}

// ── `PyType_Ready` ──────────────────────────────────────────────────────

fn store(mut ns: PyObjectRef, name: &str, mut value: PyObjectRef) {
    crate::__pyre_store!(ns, name, value);
}

/// The body a published wrapper routes to.
type WrapperFn = fn(*const c_void, &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>;

/// Publish `dunder` as a wrapper over `slot` -- `add_operators` handing
/// `PyDescr_NewWrapper` the function the type itself carries.
///
/// The capture is what makes the wrapper answer for the type it was published
/// on: reached through a subclass that overrides the method, it still runs
/// this slot rather than resolving the name again and arriving back where it
/// started.
fn publish(
    ns: PyObjectRef,
    dunder: &'static str,
    arity: u16,
    slot: *const c_void,
    call: WrapperFn,
) {
    let wrapper = Box::leak(Box::new(crate::gateway::WrapperCall { slot, call }));
    store(
        ns,
        dunder,
        crate::gateway::make_wrapper_over_slot(dunder, arity, wrapper),
    );
}

/// Copy every slot `tp` leaves null from its base -- CPython `inherit_slots`.
fn inherit_slots(tp: *mut CPyTypeObject, base: *mut CPyTypeObject) {
    if base.is_null() {
        return;
    }
    // Before the copy below fills `tp_call` in, because afterwards every
    // subclass reads as having declared one.
    inherit_vectorcall(tp, base);
    macro_rules! inherit {
        ($($field:ident),* $(,)?) => {
            $(
                unsafe {
                    if (*tp).$field.is_null() {
                        (*tp).$field = (*base).$field;
                    }
                }
            )*
        };
    }
    inherit!(
        tp_dealloc,
        tp_repr,
        tp_hash,
        tp_call,
        tp_str,
        tp_traverse,
        tp_clear,
        tp_richcompare,
        tp_iter,
        tp_iternext,
        tp_descr_get,
        tp_descr_set,
        tp_init,
        tp_alloc,
        tp_new,
        tp_free,
        tp_finalize,
    );
    // `inherit_slots` takes each legacy attribute hook together with its
    // object-key twin: a type that filled either one has declared an attribute
    // hook of its own, and copying the twin over it would cover that hook,
    // because `PyObject_GetAttr` and `PyObject_SetAttr` reach the legacy slot
    // only while the twin is null.
    unsafe {
        if (*tp).tp_getattr.is_null() && (*tp).tp_getattro.is_null() {
            (*tp).tp_getattr = (*base).tp_getattr;
            (*tp).tp_getattro = (*base).tp_getattro;
        }
        if (*tp).tp_setattr.is_null() && (*tp).tp_setattro.is_null() {
            (*tp).tp_setattr = (*base).tp_setattr;
            (*tp).tp_setattro = (*base).tp_setattro;
        }
        if (*tp).tp_basicsize == 0 {
            (*tp).tp_basicsize = (*base).tp_basicsize;
        }
        if (*tp).tp_itemsize == 0 {
            (*tp).tp_itemsize = (*base).tp_itemsize;
        }
        // The `COPYVAL(tp_dictoffset)` and `COPYVAL(tp_weaklistoffset)`
        // `inherit_special` performs beside the two above.
        if (*tp).tp_dictoffset == 0 {
            (*tp).tp_dictoffset = (*base).tp_dictoffset;
        }
        if (*tp).tp_weaklistoffset == 0 {
            (*tp).tp_weaklistoffset = (*base).tp_weaklistoffset;
        }
    }
}

/// Fill the namespace the interpreter type is built from.
fn install_namespace(ns: PyObjectRef, tp: *mut CPyTypeObject) {
    let roots = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(ns);
    let reload = || pyre_object::gc_roots::shadow_stack_get(ns_slot);

    if unsafe { !(*tp).tp_doc.is_null() } {
        let doc = text_or_none(unsafe { (*tp).tp_doc });
        store(reload(), DOC_KEY, doc);
    }

    let mut index = 0isize;
    while unsafe {
        !(*tp).tp_methods.is_null() && !(*(*tp).tp_methods.offset(index)).ml_name.is_null()
    } {
        let method = unsafe { (*tp).tp_methods.offset(index) };
        index += 1;
        let name = unsafe { std::ffi::CStr::from_ptr((*method).ml_name) }
            .to_string_lossy()
            .into_owned();
        // `typeobject.c:type_add_method` — the flags decide the descriptor,
        // each naming a different receiver.  A row carrying both is refused
        // before the type is built.
        let flags = unsafe { (*method).ml_flags };
        let carrier_type = if flags.contains(super::methodobject::MethFlags::METH_CLASS) {
            classmethod_descriptor_type()
        } else {
            method_descriptor_type()
        };
        let descriptor = new_carrier(
            carrier_type,
            method as usize,
            unsafe { (*method).ml_name },
            unsafe { (*method).ml_doc },
            pyre_object::PY_NULL,
        );
        let descriptor_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(descriptor);
        // A static row is a function bound to the type, wrapped so that
        // reading it through the class or an instance yields the function
        // itself rather than binding a receiver a second time.
        let descriptor = if flags.contains(super::methodobject::MethFlags::METH_STATIC) {
            let owner = reload();
            match super::methodobject::new_pycfunction(method, owner, owner) {
                Ok(function) => {
                    let function_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = roots.pin_root(function);
                    pyre_object::function::w_staticmethod_new(
                        pyre_object::gc_roots::shadow_stack_get(function_slot),
                    )
                }
                // The name is left unbound rather than bound to something
                // that would take its first argument as a receiver.
                Err(error) => {
                    super::pyerrors::set_pending_error(error);
                    continue;
                }
            }
        } else {
            pyre_object::gc_roots::shadow_stack_get(descriptor_slot)
        };
        let descriptor_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(descriptor);
        store(
            reload(),
            &name,
            pyre_object::gc_roots::shadow_stack_get(descriptor_slot),
        );
    }

    let mut index = 0isize;
    while unsafe {
        !(*tp).tp_members.is_null() && !(*(*tp).tp_members.offset(index)).name.is_null()
    } {
        let member = unsafe { (*tp).tp_members.offset(index) };
        index += 1;
        let name = unsafe { std::ffi::CStr::from_ptr((*member).name) }
            .to_string_lossy()
            .into_owned();
        let descriptor = new_carrier(
            member_descriptor_type(),
            member as usize,
            unsafe { (*member).name },
            unsafe { (*member).doc },
            pyre_object::PY_NULL,
        );
        let descriptor_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(descriptor);
        store(
            reload(),
            &name,
            pyre_object::gc_roots::shadow_stack_get(descriptor_slot),
        );
    }

    let mut index = 0isize;
    while unsafe { !(*tp).tp_getset.is_null() && !(*(*tp).tp_getset.offset(index)).name.is_null() }
    {
        let getset = unsafe { (*tp).tp_getset.offset(index) };
        index += 1;
        let name = unsafe { std::ffi::CStr::from_ptr((*getset).name) }
            .to_string_lossy()
            .into_owned();
        let descriptor = new_carrier(
            getset_descriptor_type(),
            getset as usize,
            unsafe { (*getset).name },
            unsafe { (*getset).doc },
            pyre_object::PY_NULL,
        );
        let descriptor_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(descriptor);
        store(
            reload(),
            &name,
            pyre_object::gc_roots::shadow_stack_get(descriptor_slot),
        );
    }

    let ns = reload();
    // A wrapper routes to the slot the receiver's type carries, and by here
    // `inherit_slots` has copied a base's.  With no slot to route to it can
    // only refuse, and storing it would cover what the base defines --
    // `type.__new__` for a metatype, which is the whole of what one is for.
    // `type_ready_set_new` and `add_operators` publish a wrapper for a slot
    // the type has and leave an inherited one to the base that declared it.
    // Every scalar slot a wrapper names, the arity its call shape declares,
    // and the body that runs it.  A slot filled with an interpreter
    // trampoline earns no wrapper: the trampoline answers by resolving the
    // very name this would publish.
    let scalars: [(&'static str, u16, *const c_void, WrapperFn); 13] = unsafe {
        [
            ("__new__", crate::gateway::HOPELESS, (*tp).tp_new, slot_new),
            (
                "__init__",
                crate::gateway::HOPELESS,
                (*tp).tp_init,
                slot_init,
            ),
            ("__repr__", 1, (*tp).tp_repr, slot_repr),
            ("__str__", 1, (*tp).tp_str, slot_str),
            ("__iter__", 1, (*tp).tp_iter, slot_iter),
            ("__next__", 1, (*tp).tp_iternext, slot_iternext),
            ("__hash__", 1, (*tp).tp_hash, slot_hash),
            (
                "__call__",
                crate::gateway::HOPELESS,
                (*tp).tp_call,
                slot_call,
            ),
            ("__getattribute__", 2, (*tp).tp_getattro, slot_getattro),
            ("__setattr__", 3, (*tp).tp_setattro, slot_setattro),
            ("__delattr__", 2, (*tp).tp_setattro, slot_delattro),
            ("__get__", 3, (*tp).tp_descr_get, slot_descr_get),
            ("__set__", 3, (*tp).tp_descr_set, slot_descr_set),
        ]
    };
    for (dunder, arity, slot, wrapper) in scalars {
        if !slot.is_null() && !is_interpreter_slot(slot) {
            publish(ns, dunder, arity, slot, wrapper);
        }
    }
    // `slotdefs` carries `TPSLOT(__setattr__, tp_setattr, NULL, NULL, "")` and
    // publishes no wrapper for it, because `PyObject_SetAttr` reaches the
    // legacy slot itself once the object-key twin is null.  Attribute access
    // here resolves the dunder instead, so the fallback is a wrapper.
    unsafe {
        if (*tp).tp_getattro.is_null() && !(*tp).tp_getattr.is_null() {
            publish(ns, "__getattribute__", 2, (*tp).tp_getattr, slot_getattr);
        }
        if (*tp).tp_setattro.is_null() && !(*tp).tp_setattr.is_null() {
            publish(ns, "__setattr__", 3, (*tp).tp_setattr, slot_setattr);
            publish(ns, "__delattr__", 2, (*tp).tp_setattr, slot_delattr);
        }
    }
    // The deletion shares its slot with the assignment above and goes in with
    // it, there being no spelling for a type that has only one of them.
    let descr_set = unsafe { (*tp).tp_descr_set };
    if !descr_set.is_null() && !is_interpreter_slot(descr_set) {
        publish(ns, "__delete__", 2, descr_set, slot_descr_delete);
    }
    if unsafe { !(*tp).tp_richcompare.is_null() } {
        install_comparisons(ns, tp);
    }
    let asynchronous: [(
        &'static str,
        fn(*mut CPyTypeObject) -> *const c_void,
        WrapperFn,
    ); 3] = [
        ("__await__", async_slot!(am_await), slot_await),
        ("__aiter__", async_slot!(am_aiter), slot_aiter),
        ("__anext__", async_slot!(am_anext), slot_anext),
    ];
    for (dunder, pick, wrapper) in asynchronous {
        if !pick(tp).is_null() {
            publish(ns, dunder, 1, pick(tp), wrapper);
        }
    }
    install_protocols(ns, tp);
}

// ── `tp_getattro`, `tp_setattro` and the descriptor slots ───────────────

fn slot_getattro(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if slot.is_null() {
        return Err(crate::PyError::type_error(
            "cpyext object has no attribute access",
        ));
    }
    call_binary(slot, args[0], args[1])
}

/// `tp_setattro` with a NULL value is the deletion, as it is for item
/// assignment.
fn set_attribute(
    slot: *const c_void,
    w_self: PyObjectRef,
    name: PyObjectRef,
    value: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    if slot.is_null() {
        return Err(crate::PyError::type_error(
            "cpyext object does not support attribute assignment",
        ));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[w_self, name, value.unwrap_or_else(pyre_object::w_none)]);
    let reload = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
    let receiver = pyobject::make_ref(reload(0));
    let key = pyobject::make_ref(reload(1));
    let item = match value {
        Some(_) => pyobject::make_ref(reload(2)),
        None => std::ptr::null_mut(),
    };
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyObject, *mut CPyObject) -> c_int =
            std::mem::transmute(slot);
        call(receiver, key, item)
    };
    unsafe {
        pyobject::decref(receiver);
        pyobject::decref(key);
        pyobject::decref(item);
    }
    if result != 0 {
        return Err(pending_or(
            "a cpyext attribute assignment failed without setting an exception",
        ));
    }
    Ok(pyre_object::w_none())
}

fn slot_setattro(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    set_attribute(slot, args[0], args[1], Some(args[2]))
}

fn slot_delattro(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    set_attribute(slot, args[0], args[1], None)
}

/// `tp_getattr` and `tp_setattr` take the name as a `char *`, so a name that is
/// not text or carries a NUL cannot be handed over -- the refusals
/// `PyObject_GetAttr` and `PyObject_SetAttr` raise before the call.
fn legacy_attribute_name(name: PyObjectRef) -> Result<std::ffi::CString, crate::PyError> {
    let text = crate::baseobjspace::text_w(name)?;
    std::ffi::CString::new(text)
        .map_err(|_| crate::PyError::value_error("attribute name must not contain null characters"))
}

fn slot_getattr(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let name = legacy_attribute_name(args[1])?;
    let roots = pyre_object::gc_roots::push_roots();
    let receiver_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(args[0]);
    let receiver = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(receiver_slot));
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, *mut std::ffi::c_char) -> *mut CPyObject =
            std::mem::transmute(slot);
        call(receiver, name.as_ptr().cast_mut())
    };
    unsafe { pyobject::decref(receiver) };
    super::from_c_result(result)
}

/// The legacy assignment, with a NULL value for the deletion as `tp_setattro`
/// has.
fn set_attribute_legacy(
    slot: *const c_void,
    w_self: PyObjectRef,
    name: PyObjectRef,
    value: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    let name = legacy_attribute_name(name)?;
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[w_self, value.unwrap_or_else(pyre_object::w_none)]);
    let reload = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
    let receiver = pyobject::make_ref(reload(0));
    let item = match value {
        Some(_) => pyobject::make_ref(reload(1)),
        None => std::ptr::null_mut(),
    };
    let result = unsafe {
        let call: unsafe extern "C" fn(
            *mut CPyObject,
            *mut std::ffi::c_char,
            *mut CPyObject,
        ) -> c_int = std::mem::transmute(slot);
        call(receiver, name.as_ptr().cast_mut(), item)
    };
    unsafe {
        pyobject::decref(receiver);
        pyobject::decref(item);
    }
    if result != 0 {
        return Err(pending_or(
            "a cpyext attribute assignment failed without setting an exception",
        ));
    }
    Ok(pyre_object::w_none())
}

fn slot_setattr(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    set_attribute_legacy(slot, args[0], args[1], Some(args[2]))
}

fn slot_delattr(slot: *const c_void, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    set_attribute_legacy(slot, args[0], args[1], None)
}

/// `tp_descr_get(self, obj, type)` — `obj` is NULL for a class access.
fn slot_descr_get(
    slot: *const c_void,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    if slot.is_null() {
        return Ok(args[0]);
    }
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[args[0], args[1], args[2]]);
    let reload = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
    let none = |value: PyObjectRef| unsafe { pyre_object::is_none(value) };
    let descriptor = pyobject::make_ref(reload(0));
    let instance = if none(reload(1)) {
        std::ptr::null_mut()
    } else {
        pyobject::make_ref(reload(1))
    };
    let owner = if none(reload(2)) {
        std::ptr::null_mut()
    } else {
        pyobject::make_ref(reload(2))
    };
    let result = unsafe {
        let call: unsafe extern "C" fn(
            *mut CPyObject,
            *mut CPyObject,
            *mut CPyObject,
        ) -> *mut CPyObject = std::mem::transmute(slot);
        call(descriptor, instance, owner)
    };
    unsafe {
        pyobject::decref(descriptor);
        pyobject::decref(instance);
        pyobject::decref(owner);
    }
    super::from_c_result(result)
}

fn descr_assign(
    slot: *const c_void,
    descriptor: PyObjectRef,
    instance: PyObjectRef,
    value: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    if slot.is_null() {
        return Err(crate::PyError::attribute_error(
            "cpyext descriptor does not support assignment",
        ));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[
        descriptor,
        instance,
        value.unwrap_or_else(pyre_object::w_none),
    ]);
    let reload = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
    let owner = pyobject::make_ref(reload(0));
    let target = pyobject::make_ref(reload(1));
    let item = match value {
        Some(_) => pyobject::make_ref(reload(2)),
        None => std::ptr::null_mut(),
    };
    let result = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyObject, *mut CPyObject) -> c_int =
            std::mem::transmute(slot);
        call(owner, target, item)
    };
    unsafe {
        pyobject::decref(owner);
        pyobject::decref(target);
        pyobject::decref(item);
    }
    if result != 0 {
        return Err(pending_or(
            "a cpyext descriptor assignment failed without setting an exception",
        ));
    }
    Ok(pyre_object::w_none())
}

fn slot_descr_set(
    slot: *const c_void,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    descr_assign(slot, args[0], args[1], Some(args[2]))
}

fn slot_descr_delete(
    slot: *const c_void,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    descr_assign(slot, args[0], args[1], None)
}

/// Give every descriptor the class it was defined on.
///
/// The carriers are built before the type exists, so `__objclass__` is stamped
/// once it does — the same shape `stamp_new_descr_self` uses for `__new__`.
/// `tp_dict` — the namespace an extension reads and writes through the field.
///
/// `__Pyx_setup_reduce` stores `__reduce__` here and calls `PyType_Modified`
/// afterwards, so the block has to be the type's own namespace: a copy would
/// take the write and nothing would answer with it.  What is stored is the
/// namespace's mirror, which stays put while the dict it links to moves.
///
/// The reference is never released, which is the whole of what a type mirror
/// is: [`ready`] makes it immortal, and the namespace outlives the type no
/// more than the type outlives itself.
fn stamp_tp_dict(tp: *mut CPyTypeObject, w_type: PyObjectRef) {
    // The block goes out as it stands, and a C reader has no `unwrap_cell` in
    // front of it, so a cell an earlier rebind parked would read as the
    // attribute.  `type_setdictvalue_wtf8` keeps a mirrored type's namespace
    // raw from here on; this is the same boundary for the stores that happened
    // before the mirror existed.
    //
    // De-celling boxes an `IntMutableCell` payload, so it can collect, and the
    // comment above says what that means here: the mirror stays put, the dict it
    // links to moves.  Root the type over the call and read both the type and
    // the block back from the shadow stack afterwards.
    let roots = pyre_object::gc_roots::push_roots();
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_type);
    unsafe {
        crate::objspace::std::classdict::uncell_type_namespace(
            pyre_object::gc_roots::shadow_stack_get(type_slot),
        )
    };
    let w_type = pyre_object::gc_roots::shadow_stack_get(type_slot);
    let w_dict = unsafe { pyre_object::w_type_get_dict_ptr(w_type) } as PyObjectRef;
    if w_dict.is_null() {
        return;
    }
    unsafe { (*tp).tp_dict = pyobject::make_ref(w_dict) };
}

fn stamp_objclass(w_type: PyObjectRef, tp: *mut CPyTypeObject) {
    let roots = pyre_object::gc_roots::push_roots();
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    let w_type = roots.pin_root(w_type);
    let mut names: Vec<String> = Vec::new();
    let mut collect = |mut next: Box<dyn FnMut(isize) -> *const c_char>| {
        let mut index = 0isize;
        loop {
            let name = next(index);
            if name.is_null() {
                break;
            }
            names.push(
                unsafe { std::ffi::CStr::from_ptr(name) }
                    .to_string_lossy()
                    .into_owned(),
            );
            index += 1;
        }
    };
    let methods = unsafe { (*tp).tp_methods };
    if !methods.is_null() {
        collect(Box::new(move |index| unsafe {
            (*methods.offset(index)).ml_name
        }));
    }
    let members = unsafe { (*tp).tp_members };
    if !members.is_null() {
        collect(Box::new(move |index| unsafe {
            (*members.offset(index)).name
        }));
    }
    let getset = unsafe { (*tp).tp_getset };
    if !getset.is_null() {
        collect(Box::new(move |index| unsafe {
            (*getset.offset(index)).name
        }));
    }
    for name in names {
        let w_type = pyre_object::gc_roots::shadow_stack_get(type_slot);
        if let Some(descriptor) = unsafe { crate::baseobjspace::lookup_in_type(w_type, &name) } {
            carrier_set(
                descriptor,
                OBJCLASS_KEY,
                pyre_object::gc_roots::shadow_stack_get(type_slot),
            );
        }
    }
}

fn ready(tp: *mut CPyTypeObject, w_metaclass: PyObjectRef) -> Result<(), crate::PyError> {
    if tp.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyType_Ready(): NULL type",
        ));
    }
    // Before `inherit_slots` copies a base's `tp_new` and the default below
    // fills the rest: what the metaclass check asks is whether the extension
    // *declared* one, which the field stops answering a few lines from here.
    record_declared_tp_new(tp);
    if unsafe { (*tp).tp_flags.contains(TpFlags::PY_TPFLAGS_READY) } {
        return Ok(());
    }
    if unsafe { (*tp).tp_flags.contains(TpFlags::PY_TPFLAGS_READYING) } {
        return Err(crate::PyError::runtime_error(
            "PyType_Ready(): circular type hierarchy",
        ));
    }
    unsafe { (*tp).tp_flags |= TpFlags::PY_TPFLAGS_READYING };
    // CPython 3.14 `PyType_Ready`: legacy non-heap extension statics are
    // immutable, but do not receive the private STATIC_BUILTIN bit reserved
    // for interpreter-core `_PyStaticType_InitBuiltin` owners.
    if !unsafe { (*tp).tp_flags.contains(TpFlags::PY_TPFLAGS_HEAPTYPE) } {
        unsafe { (*tp).tp_flags |= TpFlags::PY_TPFLAGS_IMMUTABLETYPE };
    }

    let base = unsafe { (*tp).tp_base };
    if !base.is_null() {
        // A base is readied for itself, so it derives its own metatype.
        ready(base, PY_NULL)?;
    }
    inherit_slots(tp, base);
    if unsafe { (*tp).tp_basicsize } == 0 {
        unsafe { (*tp).tp_basicsize = size_of::<CPyObject>() as isize };
    }
    if unsafe { (*tp).tp_alloc.is_null() } {
        unsafe { (*tp).tp_alloc = PyType_GenericAlloc as *const c_void };
    }
    if unsafe { (*tp).tp_free.is_null() } {
        unsafe { (*tp).tp_free = PyObject_Free as *const c_void };
    }
    // `type_ready_set_new`.  A static type that declared no constructor and
    // derives straight from `object` does not take `object`'s: the plain
    // instance it builds is not the storage the type's own methods are written
    // against, so the type refuses instantiation instead.  A heap type is the
    // runtime's own and does inherit, as does one derived from any other base.
    if !declares_tp_new(tp)
        && !unsafe { (*tp).tp_flags.contains(TpFlags::PY_TPFLAGS_HEAPTYPE) }
        && base_is_object(base)
    {
        unsafe { (*tp).tp_flags |= TpFlags::PY_TPFLAGS_DISALLOW_INSTANTIATION };
    }
    if unsafe {
        (*tp)
            .tp_flags
            .contains(TpFlags::PY_TPFLAGS_DISALLOW_INSTANTIATION)
    } {
        // `inherit_slots` copies a base's `tp_new` for every slot alike, and
        // this is the one slot a type carrying the flag must not have.
        unsafe { (*tp).tp_new = std::ptr::null() };
    } else if unsafe { (*tp).tp_new.is_null() }
        && !base_supplies_new(base)
        && !base_refuses_new(base)
    {
        // `PyType_GenericNew` builds an ordinary instance, which is not the
        // storage a base that has a constructor of its own would have built:
        // left unset, the slot routes to the base's `__new__` instead.
        unsafe { (*tp).tp_new = PyType_GenericNew as *const c_void };
    }

    let w_base = if base.is_null() {
        crate::typedef::w_object()
    } else {
        interpreter_type(base)
    };
    if w_base.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyType_Ready(): tp_base is not a ready type",
        ));
    }
    let qualified = if unsafe { (*tp).tp_name.is_null() } {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr((*tp).tp_name) }
            .to_string_lossy()
            .into_owned()
    };
    if qualified.is_empty() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyType_Ready(): the type has no tp_name",
        ));
    }
    // The qualified name is handed on whole: `make_builtin_type_with_base`
    // publishes the leading component as the type's `__module__`, which is
    // where `tp_name`'s prefix is meant to end up.

    let w_metatype = resolve_metatype(tp, w_metaclass, w_base)?;

    // `typeobject.c:type_add_method` refuses a row declaring both, because
    // each of the two names a different receiver.
    let mut index = 0isize;
    while unsafe {
        !(*tp).tp_methods.is_null() && !(*(*tp).tp_methods.offset(index)).ml_name.is_null()
    } {
        let method = unsafe { (*tp).tp_methods.offset(index) };
        index += 1;
        let flags = unsafe { (*method).ml_flags };
        if !flags.contains(super::methodobject::MethFlags::METH_CLASS)
            || !flags.contains(super::methodobject::MethFlags::METH_STATIC)
        {
            continue;
        }
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!(
                "method cannot be both class and static: {}",
                unsafe { std::ffi::CStr::from_ptr((*method).ml_name) }.to_string_lossy()
            ),
        ));
    }

    let roots = pyre_object::gc_roots::push_roots();
    let base_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_base);
    let metatype_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_metatype);
    // `finish_type_1`, which fills `c_tp_bases` before the interpreter type is
    // built.  An extension that declared none gets the single base it was
    // readied on; `single_base` has already collapsed a `Py_tp_bases` tuple to
    // that base, so this is where the tuple form comes back.
    if unsafe { (*tp).tp_bases.is_null() } {
        let bases_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(pyre_object::w_tuple_new(vec![
            pyre_object::gc_roots::shadow_stack_get(base_slot),
        ]));
        unsafe {
            (*tp).tp_bases =
                pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(bases_slot));
        }
    }
    let w_type = crate::typedef::make_builtin_type_with_metatype(
        &qualified,
        |ns| install_namespace(ns, tp),
        pyre_object::gc_roots::shadow_stack_get(base_slot),
        instance_layout(pyre_object::gc_roots::shadow_stack_get(base_slot)),
        pyre_object::gc_roots::shadow_stack_get(metatype_slot),
    );
    unsafe {
        pyre_object::w_type_set_cpython_type_flags(
            w_type,
            (*tp).tp_flags.contains(TpFlags::PY_TPFLAGS_HEAPTYPE),
            false,
            (*tp).tp_flags.contains(TpFlags::PY_TPFLAGS_IMMUTABLETYPE),
        );
    }
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_type);
    unsafe {
        pyre_object::typeobject::w_type_set_hasdict(
            pyre_object::gc_roots::shadow_stack_get(type_slot),
            (*tp).tp_dictoffset != 0,
        );
        // `typeobject.py create_all_slots` gives a C heap type's namespace the
        // default weakref support even when its `PyType_FromSpec` layout has
        // no `tp_weaklistoffset`.  A legacy static type gets it only from an
        // explicit (or inherited) offset; this keeps slotless statics such as
        // `_cffi_backend.FFI` non-weakrefable while retaining the heap-type
        // behaviour the PyPy cpyext oracle exposes.
        pyre_object::typeobject::w_type_set_weakrefable(
            pyre_object::gc_roots::shadow_stack_get(type_slot),
            (*tp).tp_weaklistoffset != 0 || (*tp).tp_flags.contains(TpFlags::PY_TPFLAGS_HEAPTYPE),
        );
        // `type_call`'s own test, which is the slot rather than the flag: a
        // subtype of a type that carries the flag inherits the empty slot
        // without the flag, and it is just as unable to build an instance.
        // With no constructor to route to, `Type()` is refused with `cannot
        // create ... instances` rather than answered with storage the type's
        // methods cannot read.
        if (*tp).tp_new.is_null() {
            pyre_object::w_type_set_disallow_instantiation(
                pyre_object::gc_roots::shadow_stack_get(type_slot),
            );
        }
        // `tp_vectorcall` is read off the class object itself, so it belongs
        // to this type alone: it is in no `COPYSLOT` list, so a subclass
        // readies with the null it declared and takes the ordinary route.
        if !(*tp).tp_vectorcall.is_null() {
            pyre_object::typeobject::w_type_set_has_vectorcall(
                pyre_object::gc_roots::shadow_stack_get(type_slot),
            );
        }
    };
    // `finish_type_2`'s `pto.c_tp_mro`, once there is a type to read one off.
    {
        let mro = unsafe {
            pyre_object::typeobject::w_type_get_mro(pyre_object::gc_roots::shadow_stack_get(
                type_slot,
            ))
        };
        if !mro.is_null() {
            let mro_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(pyre_object::w_tuple_new(unsafe { (*mro).to_vec() }));
            unsafe {
                (*tp).tp_mro =
                    pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(mro_slot));
            }
        }
    }
    stamp_tp_dict(tp, pyre_object::gc_roots::shadow_stack_get(type_slot));
    stamp_objclass(pyre_object::gc_roots::shadow_stack_get(type_slot), tp);

    // `type_realize` allocates a `W_PyCTypeObject`, then `track_reference`
    // dispatches to `W_PyCTypeObject._cpyext_attach_pyobj`: unlike ordinary
    // type mirrors, the extension's existing block is an O-link and its
    // address lives directly in the interpreter type's `_cpy_ref` field.
    unsafe {
        pyobject::attach_foreign_pyobj(
            pyre_object::gc_roots::shadow_stack_get(type_slot),
            &raw mut (*tp).ob_base.ob_base,
        );
        // Written rather than moved: whatever the static declared as its
        // metatype is discarded here, so there is no reference of this layer's
        // to release off it.
        let metatype = pyobject::type_mirror(pyre_object::gc_roots::shadow_stack_get(type_slot));
        (*tp).ob_base.ob_base.ob_type = metatype;
        pyobject::own_heap_type(metatype);
        (*tp).tp_flags =
            ((*tp).tp_flags & !TpFlags::PY_TPFLAGS_READYING) | TpFlags::PY_TPFLAGS_READY;
    }
    set_fast_subclass_flags(tp, pyre_object::gc_roots::shadow_stack_get(type_slot));
    Ok(())
}

/// The types whose `tp_new` the extension wrote itself.
///
/// [`ready`] fills an unset `tp_new` with `PyType_GenericNew` and
/// [`inherit_slots`] copies a base's before that, so by the time a metaclass is
/// checked the field answers "does this type have a `tp_new`", which is yes for
/// every readied type.  What the check asks is the narrower question this
/// records: did the extension declare one.
static DECLARED_TP_NEW: super::ForkMutex<super::address_table::AddressSet> = super::ForkMutex::new(
    super::address_table::AddressSet::with_hasher(std::hash::BuildHasherDefault::new()),
);

fn record_declared_tp_new(tp: *mut CPyTypeObject) {
    if unsafe { (*tp).tp_new.is_null() } {
        return;
    }
    DECLARED_TP_NEW.lock().insert(tp as usize);
}

fn declares_tp_new(tp: *mut CPyTypeObject) -> bool {
    !tp.is_null() && DECLARED_TP_NEW.lock().contains(&(tp as usize))
}

/// Whether `base` is `object` -- `type_ready_set_new`'s
/// `base == &PyBaseObject_Type`.  A type readied without a `tp_base` derives
/// from `object` too, that being the base `type_ready_set_base` fills in.
fn base_is_object(base: *mut CPyTypeObject) -> bool {
    if base.is_null() {
        return true;
    }
    let w_base = interpreter_type(base);
    !w_base.is_null() && std::ptr::eq(w_base, crate::typedef::w_object())
}

/// Whether the base has no constructor at all.
///
/// `inherit_slots` has already handed the subtype the empty slot, which is
/// what a base refusing instantiation owes it: an instance of the subtype is
/// one of the base too, and neither is buildable.  The stand-in below would
/// build one anyway.
fn base_refuses_new(base: *mut CPyTypeObject) -> bool {
    let w_base = interpreter_type(base);
    !w_base.is_null() && unsafe { pyre_object::w_type_disallows_instantiation(w_base) }
}

/// Whether the base has a constructor of its own, which `PyType_GenericNew`
/// would take away.
///
/// `inherit_slots` answers this for a C base by copying its `tp_new`.  It
/// cannot answer it for a class this runtime defines, whose mirror carries no
/// slots at all, so the namespaces are compared instead.  For `object` the two
/// constructors agree and nothing is lost; for `type` the difference is
/// between building a class and building an instance, and for an exception it
/// is between the storage the methods it inherits are written against and a
/// plain one.
fn base_supplies_new(base: *mut CPyTypeObject) -> bool {
    let w_base = interpreter_type(base);
    let w_object = crate::typedef::w_object();
    !w_base.is_null()
        && !w_object.is_null()
        && !std::ptr::eq(w_base, w_object)
        && unsafe { crate::baseobjspace::lookup_in_type(w_base, "__new__") }
            != unsafe { crate::baseobjspace::lookup_in_type(w_object, "__new__") }
}

/// The layout a type readied over `w_base` gives its instances: the base's
/// own, which is the storage the methods it inherits are written against.
///
/// Instances of a type derived from `type` are `W_TypeObject`s and instances
/// of one derived from an exception are `W_BaseException`s — the same choice
/// `_ctypes`' `make_ctypes_metatype` makes for its metaclasses.  The general
/// instance layout makes every inherited method refuse the receiver it is
/// handed, which is what a `descriptor ... requires a ... object` report from
/// a type built here means.
fn instance_layout(w_base: PyObjectRef) -> *const pyre_object::PyType {
    let instance = &pyre_object::pyobject::INSTANCE_TYPE as *const pyre_object::PyType;
    if w_base.is_null() {
        return instance;
    }
    let layout = unsafe { pyre_object::w_type_get_layout_ptr(w_base) };
    match layout.is_null() {
        true => instance,
        false => unsafe { (*(*layout).typedef).instance_type },
    }
}

/// The metatype a type being readied gets — `typeobject.c
/// _PyType_FromMetaclass_impl`'s metaclass resolution, plus the metatype a
/// static `PyTypeObject` declares for itself.
///
/// The named metaclass wins; failing that the static's own `ob_type`, which
/// `type_realize` reads back after `finish_type_1`; failing that the base's
/// metatype, so a type derived from one built through a metaclass keeps it.
fn resolve_metatype(
    tp: *mut CPyTypeObject,
    w_metaclass: PyObjectRef,
    w_base: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let declared = match w_metaclass.is_null() {
        false => w_metaclass,
        true => interpreter_type(unsafe { (*tp).ob_base.ob_base.ob_type }),
    };
    let w_metatype = match declared.is_null() {
        false => declared,
        // `finish_type_1` — a static that declares nothing is an instance of
        // whatever its base is an instance of.
        true => match crate::typedef::r#type(w_base) {
            Some(w_type) => w_type.as_ptr(),
            None => return Ok(PY_NULL),
        },
    };
    let w_type_type = crate::typedef::w_type();
    if w_type_type.is_null() || std::ptr::eq(w_metatype, w_type_type) {
        return Ok(PY_NULL);
    }
    if !unsafe { pyre_object::is_type(w_metatype) }
        || !unsafe { crate::baseobjspace::issubtype_w(w_metatype, w_type_type) }
    {
        return Err(crate::PyError::type_error(format!(
            "Metaclass '{}' is not a subclass of 'type'.",
            metatype_repr(w_metatype)
        )));
    }
    // Its instances have to be `W_TypeObject`s for the type below to be one.
    let layout = unsafe { pyre_object::w_type_get_layout_ptr(w_metatype) };
    let laid_out_as_a_type = !layout.is_null()
        && std::ptr::eq(
            unsafe { (*(*layout).typedef).instance_type },
            &pyre_object::pyobject::TYPE_TYPE as *const pyre_object::PyType,
        );
    if !laid_out_as_a_type {
        return Err(crate::PyError::type_error(format!(
            "Metaclass '{}' does not lay its instances out as types",
            unsafe { pyre_object::w_type_get_name(w_metatype) }
        )));
    }
    Ok(w_metatype)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_Ready(tp: *mut CPyTypeObject) -> c_int {
    match ready(tp, PY_NULL) {
        Ok(()) => 0,
        Err(error) => {
            if !tp.is_null() {
                unsafe { (*tp).tp_flags &= !TpFlags::PY_TPFLAGS_READYING };
            }
            super::pyerrors::set_pending_error(error);
            -1
        }
    }
}

// ── heap types: `PyType_FromSpec` ───────────────────────────────────────

/// One `(slot id, function)` pair of a [`CPyTypeSpec`].
#[repr(C)]
pub struct CPyTypeSlot {
    pub slot: c_int,
    pub pfunc: *mut c_void,
}

/// `PyType_Spec`.
#[repr(C)]
pub struct CPyTypeSpec {
    pub name: *const c_char,
    pub basicsize: c_int,
    pub itemsize: c_int,
    pub flags: c_uint,
    pub slots: *mut CPyTypeSlot,
}

/// The `typeslots.h` identifiers.
mod slot_id {
    use std::ffi::c_int;

    /// Declare each identifier once, so the constant and the name the test
    /// looks up in `Python.h` cannot drift apart.
    macro_rules! slot_ids {
        ($($name:ident = $value:expr,)*) => {
            $(pub const $name: c_int = $value;)*

            #[cfg(test)]
            pub(super) const ALL: &[(&str, c_int)] = &[$((stringify!($name), $value),)*];
        };
    }

    slot_ids! {
        BF_GETBUFFER = 1,
        BF_RELEASEBUFFER = 2,
        MP_ASS_SUBSCRIPT = 3,
        MP_LENGTH = 4,
        MP_SUBSCRIPT = 5,
        NB_ABSOLUTE = 6,
        NB_ADD = 7,
        NB_AND = 8,
        NB_BOOL = 9,
        NB_DIVMOD = 10,
        NB_FLOAT = 11,
        NB_FLOOR_DIVIDE = 12,
        NB_INDEX = 13,
        NB_INPLACE_ADD = 14,
        NB_INPLACE_AND = 15,
        NB_INPLACE_FLOOR_DIVIDE = 16,
        NB_INPLACE_LSHIFT = 17,
        NB_INPLACE_MULTIPLY = 18,
        NB_INPLACE_OR = 19,
        NB_INPLACE_POWER = 20,
        NB_INPLACE_REMAINDER = 21,
        NB_INPLACE_RSHIFT = 22,
        NB_INPLACE_SUBTRACT = 23,
        NB_INPLACE_TRUE_DIVIDE = 24,
        NB_INPLACE_XOR = 25,
        NB_INT = 26,
        NB_INVERT = 27,
        NB_LSHIFT = 28,
        NB_MULTIPLY = 29,
        NB_NEGATIVE = 30,
        NB_OR = 31,
        NB_POSITIVE = 32,
        NB_POWER = 33,
        NB_REMAINDER = 34,
        NB_RSHIFT = 35,
        NB_SUBTRACT = 36,
        NB_TRUE_DIVIDE = 37,
        NB_XOR = 38,
        SQ_ASS_ITEM = 39,
        SQ_CONCAT = 40,
        SQ_CONTAINS = 41,
        SQ_INPLACE_CONCAT = 42,
        SQ_INPLACE_REPEAT = 43,
        SQ_ITEM = 44,
        SQ_LENGTH = 45,
        SQ_REPEAT = 46,
        TP_ALLOC = 47,
        TP_BASE = 48,
        TP_BASES = 49,
        TP_CALL = 50,
        TP_CLEAR = 51,
        TP_DEALLOC = 52,
        TP_DEL = 53,
        TP_DESCR_GET = 54,
        TP_DESCR_SET = 55,
        TP_DOC = 56,
        TP_GETATTR = 57,
        TP_GETATTRO = 58,
        TP_HASH = 59,
        TP_INIT = 60,
        TP_IS_GC = 61,
        TP_ITER = 62,
        TP_ITERNEXT = 63,
        TP_METHODS = 64,
        TP_NEW = 65,
        TP_REPR = 66,
        TP_RICHCOMPARE = 67,
        TP_SETATTR = 68,
        TP_SETATTRO = 69,
        TP_STR = 70,
        TP_TRAVERSE = 71,
        // `tp_members` and `tp_getset` break the alphabetical run: they were added
        // after `tp_traverse` and keep the numbers they were given then.
        TP_MEMBERS = 72,
        TP_GETSET = 73,
        TP_FREE = 74,
        NB_MATRIX_MULTIPLY = 75,
        NB_INPLACE_MATRIX_MULTIPLY = 76,
        AM_AWAIT = 77,
        AM_AITER = 78,
        AM_ANEXT = 79,
        TP_FINALIZE = 80,
        AM_SEND = 81,
        TP_VECTORCALL = 82,
        TP_TOKEN = 83,
    }
}

fn apply_slot(tp: *mut CPyTypeObject, id: c_int, value: *mut c_void) -> Result<(), crate::PyError> {
    use slot_id::*;
    let function = value as *const c_void;
    macro_rules! number {
        ($field:ident) => {{
            let table = table_of!(tp, tp_as_number, CPyNumberMethods);
            unsafe { (*table).$field = function };
        }};
    }
    macro_rules! sequence {
        ($field:ident) => {{
            let table = table_of!(tp, tp_as_sequence, CPySequenceMethods);
            unsafe { (*table).$field = function };
        }};
    }
    macro_rules! mapping {
        ($field:ident) => {{
            let table = table_of!(tp, tp_as_mapping, CPyMappingMethods);
            unsafe { (*table).$field = function };
        }};
    }
    macro_rules! asynchronous {
        ($field:ident) => {{
            let table = table_of!(tp, tp_as_async, CPyAsyncMethods);
            unsafe { (*table).$field = function };
        }};
    }
    macro_rules! buffer {
        ($field:ident) => {{
            let table = table_of!(tp, tp_as_buffer, CPyBufferProcs);
            unsafe { (*table).$field = function };
        }};
    }
    macro_rules! own {
        ($field:ident, $shape:ty) => {
            unsafe { (*tp).$field = value as $shape }
        };
    }

    match id {
        BF_GETBUFFER => buffer!(bf_getbuffer),
        BF_RELEASEBUFFER => buffer!(bf_releasebuffer),
        MP_ASS_SUBSCRIPT => mapping!(mp_ass_subscript),
        MP_LENGTH => mapping!(mp_length),
        MP_SUBSCRIPT => mapping!(mp_subscript),
        NB_ABSOLUTE => number!(nb_absolute),
        NB_ADD => number!(nb_add),
        NB_AND => number!(nb_and),
        NB_BOOL => number!(nb_bool),
        NB_DIVMOD => number!(nb_divmod),
        NB_FLOAT => number!(nb_float),
        NB_FLOOR_DIVIDE => number!(nb_floor_divide),
        NB_INDEX => number!(nb_index),
        NB_INPLACE_ADD => number!(nb_inplace_add),
        NB_INPLACE_AND => number!(nb_inplace_and),
        NB_INPLACE_FLOOR_DIVIDE => number!(nb_inplace_floor_divide),
        NB_INPLACE_LSHIFT => number!(nb_inplace_lshift),
        NB_INPLACE_MULTIPLY => number!(nb_inplace_multiply),
        NB_INPLACE_OR => number!(nb_inplace_or),
        NB_INPLACE_POWER => number!(nb_inplace_power),
        NB_INPLACE_REMAINDER => number!(nb_inplace_remainder),
        NB_INPLACE_RSHIFT => number!(nb_inplace_rshift),
        NB_INPLACE_SUBTRACT => number!(nb_inplace_subtract),
        NB_INPLACE_TRUE_DIVIDE => number!(nb_inplace_true_divide),
        NB_INPLACE_XOR => number!(nb_inplace_xor),
        NB_INT => number!(nb_int),
        NB_INVERT => number!(nb_invert),
        NB_LSHIFT => number!(nb_lshift),
        NB_MULTIPLY => number!(nb_multiply),
        NB_NEGATIVE => number!(nb_negative),
        NB_OR => number!(nb_or),
        NB_POSITIVE => number!(nb_positive),
        NB_POWER => number!(nb_power),
        NB_REMAINDER => number!(nb_remainder),
        NB_RSHIFT => number!(nb_rshift),
        NB_SUBTRACT => number!(nb_subtract),
        NB_TRUE_DIVIDE => number!(nb_true_divide),
        NB_XOR => number!(nb_xor),
        NB_MATRIX_MULTIPLY => number!(nb_matrix_multiply),
        NB_INPLACE_MATRIX_MULTIPLY => number!(nb_inplace_matrix_multiply),
        SQ_ASS_ITEM => sequence!(sq_ass_item),
        SQ_CONCAT => sequence!(sq_concat),
        SQ_CONTAINS => sequence!(sq_contains),
        SQ_INPLACE_CONCAT => sequence!(sq_inplace_concat),
        SQ_INPLACE_REPEAT => sequence!(sq_inplace_repeat),
        SQ_ITEM => sequence!(sq_item),
        SQ_LENGTH => sequence!(sq_length),
        SQ_REPEAT => sequence!(sq_repeat),
        AM_AWAIT => asynchronous!(am_await),
        AM_AITER => asynchronous!(am_aiter),
        AM_ANEXT => asynchronous!(am_anext),
        AM_SEND => asynchronous!(am_send),
        TP_ALLOC => own!(tp_alloc, *const c_void),
        TP_BASE => own!(tp_base, *mut CPyTypeObject),
        TP_CALL => own!(tp_call, *const c_void),
        TP_CLEAR => own!(tp_clear, *const c_void),
        TP_DEALLOC => own!(tp_dealloc, *const c_void),
        TP_DEL => own!(tp_del, *const c_void),
        TP_DESCR_GET => own!(tp_descr_get, *const c_void),
        TP_DESCR_SET => own!(tp_descr_set, *const c_void),
        TP_DOC => own!(tp_doc, *const c_char),
        TP_GETATTR => own!(tp_getattr, *const c_void),
        TP_GETATTRO => own!(tp_getattro, *const c_void),
        TP_GETSET => own!(tp_getset, *mut CPyGetSetDef),
        TP_HASH => own!(tp_hash, *const c_void),
        TP_INIT => own!(tp_init, *const c_void),
        TP_IS_GC => own!(tp_is_gc, *const c_void),
        TP_ITER => own!(tp_iter, *const c_void),
        TP_ITERNEXT => own!(tp_iternext, *const c_void),
        TP_METHODS => own!(tp_methods, *mut super::methodobject::CPyMethodDef),
        TP_MEMBERS => own!(tp_members, *mut CPyMemberDef),
        TP_NEW => own!(tp_new, *const c_void),
        TP_REPR => own!(tp_repr, *const c_void),
        TP_RICHCOMPARE => own!(tp_richcompare, *const c_void),
        TP_SETATTR => own!(tp_setattr, *const c_void),
        TP_SETATTRO => own!(tp_setattro, *const c_void),
        TP_STR => own!(tp_str, *const c_void),
        TP_TRAVERSE => own!(tp_traverse, *const c_void),
        TP_FINALIZE => own!(tp_finalize, *const c_void),
        TP_FREE => own!(tp_free, *const c_void),
        // `typeslots.inc`'s last entry but one, `{-1, offsetof(PyTypeObject,
        // tp_vectorcall)}`: the field is written like any other, and it is
        // `PyType_Ready` that arms the type for it.
        TP_VECTORCALL => own!(tp_vectorcall, *const c_void),
        // `Py_tp_bases` is a tuple, which the caller may also pass through
        // `PyType_FromSpecWithBases`; both land in `single_base`.
        TP_BASES => {
            let bases = unsafe { pyobject::from_ref(value as *mut CPyObject) };
            unsafe { (*tp).tp_base = single_base(bases)? };
        }
        other => {
            return Err(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                format!("PyType_FromSpec() does not support slot {other}"),
            ));
        }
    }
    Ok(())
}

/// `_PyType_FromMetaclass_impl`'s scan of `tp_members` for the three offsets a
/// spec cannot name with a slot of its own.
///
/// `__vectorcalloffset__` is the one that has nowhere else to go:
/// [`super::object::PyVectorcall_Call`] reads the callable at that offset, and
/// a Cython function type declares it exactly this way.  The member is still
/// published beside the others, as `type_add_members` publishes all of them.
///
/// # Safety
/// `tp` must be a valid type block.
unsafe fn adopt_offset_members(tp: *mut CPyTypeObject) -> Result<(), crate::PyError> {
    let members = unsafe { (*tp).tp_members };
    if members.is_null() {
        return Ok(());
    }
    let mut index = 0isize;
    while !unsafe { (*members.offset(index)).name }.is_null() {
        let member = unsafe { members.offset(index) };
        index += 1;
        let destination = match unsafe { std::ffi::CStr::from_ptr((*member).name) }.to_bytes() {
            b"__vectorcalloffset__" => unsafe { &raw mut (*tp).tp_vectorcall_offset },
            b"__dictoffset__" => unsafe { &raw mut (*tp).tp_dictoffset },
            b"__weaklistoffset__" => unsafe { &raw mut (*tp).tp_weaklistoffset },
            _ => continue,
        };
        unsafe { *destination = special_offset(member)? };
    }
    Ok(())
}

/// `special_offset_from_member`.  Its relative arm is spent by the time this
/// runs: [`resolve_relative_members`] has already added the type data offset
/// and cleared the flag, which leaves the pair of flag sets it accepts spelled
/// the same way.
fn special_offset(member: *mut CPyMemberDef) -> Result<isize, crate::PyError> {
    let name = || {
        unsafe { std::ffi::CStr::from_ptr((*member).name) }
            .to_string_lossy()
            .into_owned()
    };
    let complaint = if unsafe { (*member).type_code } != T_PYSSIZET {
        format!("type of {} must be Py_T_PYSSIZET", name())
    } else if unsafe { (*member).flags } != MemberFlags::PY_READONLY {
        format!(
            "flags for {} must be Py_READONLY or (Py_READONLY | Py_RELATIVE_OFFSET)",
            name()
        )
    } else {
        return Ok(unsafe { (*member).offset });
    };
    Err(crate::PyError::new(
        crate::PyErrorKind::SystemError,
        complaint,
    ))
}

/// `_PyType_FromMetaclass_impl`'s rewrite of the `Py_RELATIVE_OFFSET` members
/// it copies into the type it is building: the offset is counted from the
/// extra data a negative `basicsize` asked for, and every reader below counts
/// from the block.
///
/// The rewrite lands on a copy because the table belongs to the extension: the
/// same static table can be named by a second spec, which would otherwise read
/// the offsets this call already resolved.
///
/// # Safety
/// `tp` must be a valid type block whose `tp_members`, when it is set, is a
/// name-terminated table.
unsafe fn resolve_relative_members(
    tp: *mut CPyTypeObject,
    declared: isize,
    type_data_offset: isize,
) -> Result<(), crate::PyError> {
    let members = unsafe { (*tp).tp_members };
    if members.is_null() {
        return Ok(());
    }
    let mut table: Vec<CPyMemberDef> = Vec::new();
    let mut index = 0isize;
    while !unsafe { (*members.offset(index)).name }.is_null() {
        let mut member = unsafe { std::ptr::read(members.offset(index)) };
        index += 1;
        if member.flags.contains(MemberFlags::PY_RELATIVE_OFFSET) {
            let complaint = if declared > 0 {
                "With Py_RELATIVE_OFFSET, basicsize must be negative."
            } else if member.offset < 0 || member.offset >= -declared {
                "Member offset out of range (0..-basicsize)"
            } else {
                member.flags &= !MemberFlags::PY_RELATIVE_OFFSET;
                member.offset += type_data_offset;
                ""
            };
            if !complaint.is_empty() {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::SystemError,
                    complaint,
                ));
            }
        }
        table.push(member);
    }
    // The table is name-terminated, and it outlives the call for the same
    // reason the type does.
    table.push(CPyMemberDef {
        name: std::ptr::null(),
        type_code: 0,
        offset: 0,
        flags: MemberFlags::empty(),
        doc: std::ptr::null(),
    });
    unsafe { (*tp).tp_members = Box::leak(table.into_boxed_slice()).as_mut_ptr() };
    Ok(())
}

/// The one base pyre can build a type on.
///
/// `make_builtin_type_with_base` takes a single base, so a spec naming more
/// than one is rejected rather than silently losing the rest.
fn single_base(bases: PyObjectRef) -> Result<*mut CPyTypeObject, crate::PyError> {
    if bases.is_null() {
        return Ok(std::ptr::null_mut());
    }
    let items: Vec<PyObjectRef> = if unsafe { pyre_object::is_tuple(bases) } {
        unsafe { pyre_object::tupleobject::w_tuple_items_copy_as_vec(bases) }
    } else {
        vec![bases]
    };
    match items.len() {
        0 => Ok(std::ptr::null_mut()),
        1 => Ok(pyobject::make_ref(items[0]) as *mut CPyTypeObject),
        _ => Err(crate::PyError::type_error(
            "PyType_FromSpec() with more than one base is not supported yet",
        )),
    }
}

/// A type's `ht_module` and `ht_token`, which pyre has no heap-type struct to
/// carry.
///
/// Both are keyed by the type's own address: a type built from a spec is
/// leaked deliberately, so the key is stable for the life of the process.  The
/// module is held as an owned mirror reference, and it is that reference — not
/// the table — that roots it.
type TypeSideTable = super::address_table::AddressMap<usize>;
static TYPE_MODULES: super::ForkMutex<TypeSideTable> = super::ForkMutex::new(
    TypeSideTable::with_hasher(std::hash::BuildHasherDefault::new()),
);
static TYPE_TOKENS: super::ForkMutex<TypeSideTable> = super::ForkMutex::new(
    TypeSideTable::with_hasher(std::hash::BuildHasherDefault::new()),
);

pub(super) unsafe fn after_fork_child() {
    unsafe {
        TYPE_MODULES.reinit_after_fork();
        TYPE_TOKENS.reinit_after_fork();
        TYPE_NAMES.reinit_after_fork();
        DESCRIPTOR_BLOCKS.reinit_after_fork();
        DECLARED_TP_NEW.reinit_after_fork();
        BOUND_MINTED.reinit_after_fork();
        for family in BOUND_FAMILIES {
            family.taken.reinit_after_fork();
        }
    }
}

/// The alignment a block of type data starts at, `_align_up`'s modulus: the
/// strictest a C compiler gives any object, so a field of any type declared in
/// the extra data is aligned wherever the base's fields end.
fn align_up(size: isize) -> isize {
    const ALIGNMENT: isize = super::pyobject::BLOCK_ALIGN as isize;
    (size + ALIGNMENT - 1) & !(ALIGNMENT - 1)
}

fn from_spec(
    spec: *mut CPyTypeSpec,
    bases: *mut CPyObject,
    module: *mut CPyObject,
    metaclass: *mut CPyTypeObject,
) -> Result<PyObjectRef, crate::PyError> {
    if spec.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyType_FromSpec(): NULL spec",
        ));
    }
    // The type block is raw storage, like `type_alloc` upstream. This port's
    // allocator record cannot yet reclaim the leaked Rust box, but its
    // refcount and rawrefcount ownership remain ordinary rather than immortal.
    let tp: *mut CPyTypeObject = Box::leak(Box::new(allocated_type()));
    unsafe {
        (*tp).tp_name = (*spec).name;
        (*tp).tp_itemsize = (*spec).itemsize as isize;
        (*tp).tp_flags = TpFlags::from_bits_retain((*spec).flags as std::ffi::c_ulong)
            | TpFlags::PY_TPFLAGS_HEAPTYPE;
        (*tp).tp_base = single_base(pyobject::from_ref(bases))?;
    }
    let mut token: *mut c_void = std::ptr::null_mut();
    let mut index = 0isize;
    loop {
        let entry = unsafe { (*spec).slots.offset(index) };
        if unsafe { (*spec).slots.is_null() } || unsafe { (*entry).slot } == 0 {
            break;
        }
        // `Py_tp_token` names no slot: it declares the type's identity, and
        // the null it may carry is `Py_TP_USE_SPEC`, which asks for the spec's
        // own address rather than for no token at all.
        if unsafe { (*entry).slot } == slot_id::TP_TOKEN {
            token = match unsafe { (*entry).pfunc } {
                pfunc if pfunc.is_null() => spec as *mut c_void,
                pfunc => pfunc,
            };
        } else {
            apply_slot(tp, unsafe { (*entry).slot }, unsafe { (*entry).pfunc })?;
        }
        index += 1;
    }

    // The base's own size is what a relative or inherited `basicsize` is
    // measured against, so it has to be final before either is resolved.
    let base = unsafe { (*tp).tp_base };
    if !base.is_null() {
        ready(base, PY_NULL)?;
    }
    // The bases are only final here: `single_base` resolved them before the
    // slot loop, whose `Py_tp_base`/`Py_tp_bases` arms may have replaced them.
    let w_metaclass = winning_metaclass(metaclass, base)?;
    let base_basicsize = if base.is_null() {
        size_of::<CPyObject>() as isize
    } else {
        unsafe { (*base).tp_basicsize }
    };
    let declared = unsafe { (*spec).basicsize } as isize;
    // Where the extra data starts, which is what a `Py_RELATIVE_OFFSET` member
    // counts from.  Only the extending spelling has any.
    let type_data_offset = match declared {
        negative if negative < 0 => align_up(base_basicsize),
        other => other,
    };
    unsafe {
        (*tp).tp_basicsize = match declared {
            // Inherit: an extension that declares no storage of its own gets
            // the block its base needs, not the bare header.
            0 => base_basicsize,
            // Extend: the magnitude is the extra data appended after the base,
            // which is what `PyType_GetTypeDataSize` reports back.
            negative if negative < 0 => type_data_offset + align_up(-negative),
            absolute => absolute,
        };
    }
    // Both of these read `tp_members` and the second reads what the first
    // resolved, and the sizes above are what they are resolved against.
    unsafe { resolve_relative_members(tp, declared, type_data_offset)? };
    // The three offsets a spec has no slot id for, which a type therefore
    // declares as members instead.
    unsafe { adopt_offset_members(tp)? };

    ready(tp, w_metaclass)?;
    if !token.is_null() {
        TYPE_TOKENS.lock().insert(tp as usize, token as usize);
    }
    if !module.is_null() {
        // An owned reference, which is what roots the module for as long as
        // the type: the type itself is immortal, so nothing ever releases it.
        unsafe { pyobject::incref(module) };
        TYPE_MODULES.lock().insert(tp as usize, module as usize);
    }
    let w_type = interpreter_type(tp);
    Ok(w_type)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_FromSpec(spec: *mut CPyTypeSpec) -> *mut CPyObject {
    super::object::result(from_spec(
        spec,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    ))
}

/// `PySlot` — one entry of the identifier-keyed array `PyType_FromSlots`
/// reads, which supersedes `PyType_Spec`'s fixed fields.  The value is a
/// union of a pointer, a function, and the integer widths; every arm is
/// eight bytes wide, so one of them stands for all.
#[repr(C)]
pub struct CPySlot {
    pub sl_id: u16,
    pub sl_flags: u16,
    pub sl_reserved: u32,
    pub sl_value: u64,
}

bitflags::bitflags! {
    /// The `object.h` flags a `PySlot` entry carries in `sl_flags`.
    ///
    /// `OPTIONAL` is the one the readers act on -- an entry they may skip
    /// rather than refuse.  The names are the header's, prefix and all; the
    /// test at the foot of this file compares the two, walking `Flags::FLAGS`.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub(super) struct SlotFlags: u16 {
        const PYSLOT_OPTIONAL = 0x01;
        const PYSLOT_STATIC = 0x02;
        const PYSLOT_INTPTR = 0x04;
    }
}

const _: () = assert!(size_of::<SlotFlags>() == size_of::<u16>());
const _: () = assert!(align_of::<SlotFlags>() == align_of::<u16>());

/// The identifiers `PyType_FromSlots` reads for itself; every other one is
/// left to the `Py_tp_slots` array, which is the `PyType_Spec` vocabulary.
mod from_slots_id {
    pub const TP_SLOTS: u16 = 93;
    pub const TP_NAME: u16 = 95;
    pub const TP_BASICSIZE: u16 = 96;
    pub const TP_EXTRA_BASICSIZE: u16 = 97;
    pub const TP_ITEMSIZE: u16 = 98;
    pub const TP_FLAGS: u16 = 99;
}

/// `PyType_FromSlots(slots)` — a heap type described by one array rather than
/// by a `PyType_Spec` beside one.
///
/// The array is read into the spec the rest of this layer already builds a
/// type from: the identifiers below carry what the spec's own fields carry,
/// and `Py_tp_slots` carries the array the spec would have pointed at.  A
/// zero identifier ends the array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_FromSlots(slots: *mut CPySlot) -> *mut CPyObject {
    if slots.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let mut spec = CPyTypeSpec {
        name: std::ptr::null(),
        basicsize: 0,
        itemsize: 0,
        flags: 0,
        slots: std::ptr::null_mut(),
    };
    let mut entry = slots;
    loop {
        let slot = unsafe { &*entry };
        if slot.sl_id == 0 {
            break;
        }
        let value = slot.sl_value;
        match slot.sl_id {
            from_slots_id::TP_NAME => spec.name = value as *const c_char,
            from_slots_id::TP_FLAGS => spec.flags = value as c_uint,
            from_slots_id::TP_BASICSIZE => spec.basicsize = value as isize as c_int,
            // A size relative to the base's, which the spec spells as the
            // negative of the same number.
            from_slots_id::TP_EXTRA_BASICSIZE => {
                spec.basicsize = -(value as isize as c_int);
            }
            from_slots_id::TP_ITEMSIZE => spec.itemsize = value as isize as c_int,
            from_slots_id::TP_SLOTS => spec.slots = value as *mut CPyTypeSlot,
            unknown => {
                if !SlotFlags::from_bits_retain(slot.sl_flags).contains(SlotFlags::PYSLOT_OPTIONAL)
                {
                    super::pyerrors::set_pending_error(crate::PyError::new(
                        crate::PyErrorKind::SystemError,
                        format!("PyType_FromSlots(): unrecognised slot {unknown}"),
                    ));
                    return std::ptr::null_mut();
                }
            }
        }
        entry = unsafe { entry.add(1) };
    }
    if spec.name.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyType_FromSlots(): the slot array has no Py_tp_name",
        ));
        return std::ptr::null_mut();
    }
    super::object::result(from_spec(
        &raw mut spec,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_FromSpecWithBases(
    spec: *mut CPyTypeSpec,
    bases: *mut CPyObject,
) -> *mut CPyObject {
    super::object::result(from_spec(
        spec,
        bases,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_FromModuleAndSpec(
    module: *mut CPyObject,
    spec: *mut CPyTypeSpec,
    bases: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([module, bases]);
    super::object::result(from_spec(spec, bases, module, std::ptr::null_mut()))
}

/// `PyType_GetSlot` — read one slot back off a ready type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetSlot(tp: *mut CPyTypeObject, id: c_int) -> *mut c_void {
    use slot_id::*;
    if tp.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    macro_rules! number {
        ($field:ident) => {
            (number_slot!($field))(tp)
        };
    }
    macro_rules! sequence {
        ($field:ident) => {
            (sequence_slot!($field))(tp)
        };
    }
    macro_rules! mapping {
        ($field:ident) => {
            (mapping_slot!($field))(tp)
        };
    }
    let value: *const c_void = match id {
        MP_ASS_SUBSCRIPT => mapping!(mp_ass_subscript),
        MP_LENGTH => mapping!(mp_length),
        MP_SUBSCRIPT => mapping!(mp_subscript),
        NB_ABSOLUTE => number!(nb_absolute),
        NB_ADD => number!(nb_add),
        NB_AND => number!(nb_and),
        NB_BOOL => number!(nb_bool),
        NB_DIVMOD => number!(nb_divmod),
        NB_FLOAT => number!(nb_float),
        NB_FLOOR_DIVIDE => number!(nb_floor_divide),
        NB_INDEX => number!(nb_index),
        NB_INT => number!(nb_int),
        NB_INVERT => number!(nb_invert),
        NB_LSHIFT => number!(nb_lshift),
        NB_MULTIPLY => number!(nb_multiply),
        NB_NEGATIVE => number!(nb_negative),
        NB_OR => number!(nb_or),
        NB_POSITIVE => number!(nb_positive),
        NB_POWER => number!(nb_power),
        NB_REMAINDER => number!(nb_remainder),
        NB_RSHIFT => number!(nb_rshift),
        NB_SUBTRACT => number!(nb_subtract),
        NB_TRUE_DIVIDE => number!(nb_true_divide),
        NB_XOR => number!(nb_xor),
        SQ_ASS_ITEM => sequence!(sq_ass_item),
        SQ_CONCAT => sequence!(sq_concat),
        SQ_CONTAINS => sequence!(sq_contains),
        SQ_ITEM => sequence!(sq_item),
        SQ_LENGTH => sequence!(sq_length),
        SQ_REPEAT => sequence!(sq_repeat),
        TP_ALLOC => unsafe { (*tp).tp_alloc },
        TP_CALL => unsafe { (*tp).tp_call },
        TP_DEALLOC => unsafe { (*tp).tp_dealloc },
        TP_DESCR_GET => unsafe { (*tp).tp_descr_get },
        TP_DESCR_SET => unsafe { (*tp).tp_descr_set },
        TP_DOC => unsafe { (*tp).tp_doc as *const c_void },
        TP_GETATTRO => unsafe { (*tp).tp_getattro },
        TP_GETSET => unsafe { (*tp).tp_getset as *const c_void },
        TP_HASH => unsafe { (*tp).tp_hash },
        TP_INIT => unsafe { (*tp).tp_init },
        TP_ITER => unsafe { (*tp).tp_iter },
        TP_ITERNEXT => unsafe { (*tp).tp_iternext },
        TP_METHODS => unsafe { (*tp).tp_methods as *const c_void },
        TP_MEMBERS => unsafe { (*tp).tp_members as *const c_void },
        TP_NEW => unsafe { (*tp).tp_new },
        TP_REPR => unsafe { (*tp).tp_repr },
        TP_RICHCOMPARE => unsafe { (*tp).tp_richcompare },
        TP_SETATTRO => unsafe { (*tp).tp_setattro },
        TP_STR => unsafe { (*tp).tp_str },
        TP_VECTORCALL => unsafe { (*tp).tp_vectorcall },
        _ => std::ptr::null(),
    };
    value as *mut c_void
}

/// `src/typeobject.c PyType_GetDict` — the namespace behind `tp_dict`, owned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetDict(tp: *mut CPyTypeObject) -> *mut CPyObject {
    if tp.is_null() {
        return std::ptr::null_mut();
    }
    let dict = unsafe { (*tp).tp_dict };
    if !dict.is_null() {
        unsafe { pyobject::incref(dict) };
    }
    dict
}

/// `_PyType_Name(type)` — the tail of `tp_name` after the last dot, which is
/// the name without the module qualifying it.
///
/// The answer points into `tp_name` itself, so it lives exactly as long as the
/// type object does and the caller frees nothing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyType_Name(tp: *mut CPyTypeObject) -> *const c_char {
    if tp.is_null() {
        return std::ptr::null();
    }
    let name = unsafe { (*tp).tp_name };
    if name.is_null() {
        return name;
    }
    let bytes = unsafe { CStr::from_ptr(name) }.to_bytes();
    match bytes.iter().rposition(|&byte| byte == b'.') {
        Some(dot) => unsafe { name.add(dot + 1) },
        None => name,
    }
}

/// `PyType_GetName` and `PyType_GetQualName` — `__name__`, which is the part
/// of `tp_name` after the last dot for a type whose name arrived qualified.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetName(tp: *mut CPyTypeObject) -> *mut CPyObject {
    let w_type = interpreter_type(tp);
    if w_type.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    pyobject::make_ref(unsafe { pyre_object::typeobject::w_type_get_name_obj(w_type) })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetQualName(tp: *mut CPyTypeObject) -> *mut CPyObject {
    unsafe { PyType_GetName(tp) }
}

/// `PyType_GenericAlloc` — build the interpreter object and its C block.
///
/// The block *is* the instance's storage, so it must live exactly as long as
/// the interpreter object.  Pyre's collector has no dead queue yet, so the
/// mirror is immortal instead: releasing it when the C side drops its last
/// reference would destroy fields the interpreter object still exposes.  That
/// makes an instance of a C-defined type unreclaimable, which is the same gap
/// the module header names for reference cycles through C.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GenericAlloc(
    tp: *mut CPyTypeObject,
    nitems: isize,
) -> *mut CPyObject {
    let w_type = interpreter_type(tp);
    if w_type.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyType_GenericAlloc(): the type is not ready",
        ));
        return std::ptr::null_mut();
    }
    let roots = pyre_object::gc_roots::push_roots();
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_type);
    let instance = pyre_object::w_instance_new(pyre_object::gc_roots::shadow_stack_get(type_slot));
    let instance_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(instance);
    let size = unsafe { (*tp).tp_basicsize + nitems.max(0) * (*tp).tp_itemsize } as usize;
    // `tp_alloc` returns a new reference, so the count is the link share plus
    // the one this call hands out.  Not immortal: the link is a rawrefcount
    // P-link, so the instance dies when neither side holds it and the
    // collector queues this block for `tp_dealloc`.
    let raw = pyobject::attach(
        pyre_object::gc_roots::shadow_stack_get(instance_slot),
        REFCNT_FROM_PYPY + 1,
        tp,
        size,
    );
    if unsafe { (*tp).tp_itemsize } != 0 {
        unsafe { (*(raw as *mut CPyVarObject)).ob_size = nitems };
    }
    // An instance of a `Py_TPFLAGS_HAVE_GC` type is collected from the moment
    // `tp_alloc` hands it back; only a type that allocates its own storage has
    // to call `PyObject_GC_Track` by hand.
    super::gc::track(raw);
    raw
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GenericNew(
    tp: *mut CPyTypeObject,
    _args: *mut CPyObject,
    _kwds: *mut CPyObject,
) -> *mut CPyObject {
    unsafe { PyType_GenericAlloc(tp, 0) }
}

/// `cpyext/src/object.c:102-105 PyObject_Free` — the deallocator half of
/// [`PyType_GenericAlloc`], and the `tp_free` every type inherits.
///
/// A `tp_dealloc` written for CPython ends in `Py_TYPE(self)->tp_free(self)`,
/// so this has to exist and has to be reachable through the slot before any
/// `tp_dealloc` is dispatched.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Free(object: *mut std::ffi::c_void) {
    if object.is_null() {
        return;
    }
    unsafe { pyobject::free_block(object as *mut CPyObject) };
}

/// `PyObject_Del` — the older spelling of [`PyObject_Free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Del(object: *mut std::ffi::c_void) {
    unsafe { PyObject_Free(object) };
}

/// `PyObject_GC_Del` -- what a collected type's `tp_free` is, and the same
/// deallocator: every block this layer hands out comes from
/// [`PyType_GenericAlloc`] whether or not the collector tracks it.
///
/// An entry point rather than a spelling of [`PyObject_Del`], because an
/// extension puts it in a `tp_free` slot -- cffi's `CData` types do -- and a
/// slot holds an address.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GC_Del(object: *mut std::ffi::c_void) {
    unsafe { PyObject_Free(object) };
}

/// `object.c _Py_object_dealloc` — the `tp_dealloc` a mirror carries, and the
/// end of the chain a deallocator written in C walks up.
///
/// A type derived from a builtin finishes its own `tp_dealloc` by calling its
/// base's; Cython spells that `__Pyx_PyType_GetSlot((&PyDict_Type), tp_dealloc,
/// destructor)(o)` for a `cdef class` derived from `dict`.  The base of such a
/// type is a mirror, so the slot has to hold a function on every type an
/// extension can name as one, not only on the types that declare a deallocator
/// of their own.
///
/// Nothing but the block is released here.  What a builtin's deallocator frees
/// is the storage its instances carry, and that storage is the interpreter
/// object this block is linked to rather than anything the block owns; the
/// reference the block holds in its type is released by `pyobject::dealloc`
/// once this returns, which is where upstream's trailing `Py_DECREF(pto)` went.
unsafe extern "C" fn object_dealloc(object: *mut CPyObject) {
    if object.is_null() {
        return;
    }
    let tp = unsafe { (*object).ob_type };
    let slot = match tp.is_null() {
        true => std::ptr::null(),
        false => unsafe { (*tp).tp_free },
    };
    // A block whose type reserves no `tp_free` still has to be handed back,
    // and `PyObject_Free` is what a mirror's slot holds in any case.
    let free: unsafe extern "C" fn(*mut std::ffi::c_void) = match slot.is_null() {
        true => PyObject_Free,
        false => unsafe { std::mem::transmute(slot) },
    };
    unsafe { free(object as *mut std::ffi::c_void) };
}

/// `PyObject_Init` — stamp a block's header and make it an object.
///
/// A block from [`PyType_GenericAlloc`] arrives linked, and only its `ob_type`
/// is (re)written.  A block from `PyObject_Malloc` arrives as raw bytes, which
/// is the other supported spelling of the same allocation, so the interpreter
/// object it stands for is created here and linked to it: the count it leaves
/// with is the link share plus the one reference the caller now owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Init(
    object: *mut CPyObject,
    tp: *mut CPyTypeObject,
) -> *mut CPyObject {
    if object.is_null() {
        return unsafe { super::pyerrors::PyErr_NoMemory() };
    }
    // `object.c _PyObject_Init`: a block reaching here is memory an allocator
    // handed out, and only one this layer handed out arrives with a header
    // that says anything -- so a type to release, and a link to stop at, are
    // read from the block only then.
    let stamped = pyobject::is_own_block(object as usize);
    let previous = if stamped {
        unsafe { (*object).ob_type }
    } else {
        std::ptr::null_mut()
    };
    unsafe { pyobject::exchange_ob_type(object, tp, previous) };
    if stamped && unsafe { !(*object).ob_pyre_link.is_null() } {
        return object;
    }
    let w_type = interpreter_type(tp);
    if w_type.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyObject_Init(): the type is not ready",
        ));
        return std::ptr::null_mut();
    }
    let roots = pyre_object::gc_roots::push_roots();
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_type);
    let instance = pyre_object::w_instance_new(pyre_object::gc_roots::shadow_stack_get(type_slot));
    pyobject::link_allocated(instance, object, REFCNT_FROM_PYPY + 1);
    object
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::is_type(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_IsSubtype(
    subtype: *mut CPyTypeObject,
    supertype: *mut CPyTypeObject,
) -> c_int {
    let (sub, sup) = (interpreter_type(subtype), interpreter_type(supertype));
    if sub.is_null() || sup.is_null() {
        return 0;
    }
    crate::baseobjspace::issubclass(sub, sup).unwrap_or(false) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetFlags(tp: *mut CPyTypeObject) -> std::ffi::c_ulong {
    if tp.is_null() {
        return 0;
    }
    unsafe { (*tp).tp_flags }.bits()
}

// ── what a type reports about its module, its token and its data ────────

/// `tp_name` as it reads in a message, without pretending it is always there.
fn type_name_of(tp: *mut CPyTypeObject) -> String {
    if tp.is_null() || unsafe { (*tp).tp_name.is_null() } {
        return "<unnamed>".to_string();
    }
    unsafe { CStr::from_ptr((*tp).tp_name) }
        .to_string_lossy()
        .into_owned()
}

/// The module recorded for `tp` alone — no MRO walk, as `ht_module` is a field
/// of one heap type and is never inherited.
fn own_module(tp: *mut CPyTypeObject) -> *mut CPyObject {
    if tp.is_null() {
        return std::ptr::null_mut();
    }
    TYPE_MODULES
        .lock()
        .get(&(tp as usize))
        .map(|&address| address as *mut CPyObject)
        .unwrap_or(std::ptr::null_mut())
}

/// Borrowed: the type's own reference is what keeps the module alive, and the
/// type outlives every caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetModule(tp: *mut CPyTypeObject) -> *mut CPyObject {
    if tp.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    if !unsafe { (*tp).tp_flags.contains(TpFlags::PY_TPFLAGS_HEAPTYPE) } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "PyType_GetModule: Type '{}' is not a heap type",
            type_name_of(tp)
        )));
        return std::ptr::null_mut();
    }
    let module = own_module(tp);
    if module.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "PyType_GetModule: Type '{}' has no associated module",
            type_name_of(tp)
        )));
    }
    module
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetModuleState(tp: *mut CPyTypeObject) -> *mut c_void {
    let module = unsafe { PyType_GetModule(tp) };
    if module.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { super::modsupport::PyModule_GetState(module) }
}

/// The nearest module along `tp`'s MRO built from `def`.
///
/// Borrowed for the same reason [`PyType_GetModule`] is.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetModuleByDef(
    tp: *mut CPyTypeObject,
    def: *mut super::modsupport::CPyModuleDef,
) -> *mut CPyObject {
    if tp.is_null() || def.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    for base in c_bases(interpreter_type(tp)) {
        let module = own_module(base);
        if !module.is_null() && unsafe { super::modsupport::PyModule_GetDef(module) } == def {
            return module;
        }
    }
    super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
        "PyType_GetModuleByDef: No superclass of '{}' has the given module",
        type_name_of(tp)
    )));
    std::ptr::null_mut()
}

/// `PyType_GetBaseByToken` — 1 with `result` filled, 0 for no match, -1 on a
/// caller error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetBaseByToken(
    tp: *mut CPyTypeObject,
    token: *mut c_void,
    result: *mut *mut CPyTypeObject,
) -> c_int {
    if !result.is_null() {
        unsafe { *result = std::ptr::null_mut() };
    }
    if token.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyType_GetBaseByToken called with token=NULL",
        ));
        return -1;
    }
    if tp.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    // A static type has no heap-type superclass, so the walk cannot find one.
    if !unsafe { (*tp).tp_flags.contains(TpFlags::PY_TPFLAGS_HEAPTYPE) } {
        return 0;
    }
    let wanted = token as usize;
    for base in c_bases(interpreter_type(tp)) {
        if TYPE_TOKENS.lock().get(&(base as usize)) != Some(&wanted) {
            continue;
        }
        if !result.is_null() {
            unsafe {
                pyobject::incref(&raw mut (*base).ob_base.ob_base);
                *result = base;
            }
        }
        return 1;
    }
    0
}

/// The extra storage a type declared beyond its base's, which is the
/// magnitude of a negative `PyType_Spec.basicsize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetTypeDataSize(tp: *mut CPyTypeObject) -> isize {
    if tp.is_null() {
        return 0;
    }
    let base = unsafe { (*tp).tp_base };
    let base_basicsize = if base.is_null() {
        size_of::<CPyObject>() as isize
    } else {
        unsafe { (*base).tp_basicsize }
    };
    let extra = unsafe { (*tp).tp_basicsize } - align_up(base_basicsize);
    extra.max(0)
}

/// The address of that storage inside one instance.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GetTypeData(
    object: *mut CPyObject,
    tp: *mut CPyTypeObject,
) -> *mut c_void {
    if object.is_null() || tp.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let base = unsafe { (*tp).tp_base };
    let base_basicsize = if base.is_null() {
        size_of::<CPyObject>() as isize
    } else {
        unsafe { (*base).tp_basicsize }
    };
    unsafe { (object as *mut u8).offset(align_up(base_basicsize)) as *mut c_void }
}

/// The variable part of an instance of a type that put its items at the end.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GetItemData(object: *mut CPyObject) -> *mut c_void {
    if object.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let tp = unsafe { (*object).ob_type };
    if tp.is_null() || !unsafe { (*tp).tp_flags.contains(TpFlags::PY_TPFLAGS_ITEMS_AT_END) } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "type '{}' does not have Py_TPFLAGS_ITEMS_AT_END",
            type_name_of(tp)
        )));
        return std::ptr::null_mut();
    }
    unsafe { (object as *mut u8).offset((*tp).tp_basicsize) as *mut c_void }
}

/// `typeobject.py:PyType_Modified` — drop what the interpreter cached about a
/// type whose C-side namespace has just been rewritten.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_Modified(tp: *mut CPyTypeObject) {
    let w_type = interpreter_type(tp);
    if w_type.is_null() {
        return;
    }
    // No key: the C caller says only that something changed.
    unsafe { crate::baseobjspace::mutated_absent(w_type) };
}

/// `PyType_ClearCache` — empty the method cache and report the version the
/// entries it dropped were stamped with.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_ClearCache() -> c_uint {
    crate::baseobjspace::clear_method_cache();
    // Upstream's tag is a per-interpreter counter; pyre mints one identity per
    // type instead, so the only honest whole-cache answer is that no single
    // version described it.
    0
}

/// `PyType_Freeze` — refuse further changes to a type's namespace.
///
/// Immutability is `flag_heaptype = false` here, the state every builtin type
/// is already in, and it is what `type.__setattr__` consults.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_Freeze(tp: *mut CPyTypeObject) -> c_int {
    let w_type = interpreter_type(tp);
    if w_type.is_null() || !unsafe { pyre_object::is_type(w_type) } {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let mro = unsafe { pyre_object::w_type_get_mro(w_type) };
    if mro.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::type_error(
            "unable to get the type MRO",
        ));
        return -1;
    }
    // The type itself is the head of its own MRO and is the one being frozen,
    // so only what it inherits from has to be immutable already.
    for &w_base in unsafe { (*mro).as_slice() }.iter().skip(1) {
        if !unsafe { pyre_object::w_type_is_cpython_immutabletype(w_base) } {
            super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
                "Creating immutable type {} from mutable base {}",
                type_name_of(tp),
                unsafe { pyre_object::typeobject::w_type_get_name(w_base) }
            )));
            return -1;
        }
    }
    unsafe {
        pyre_object::typeobject::w_type_set_heaptype(w_type, false);
        // Freezing changes CPython's immutable axis but not its HEAPTYPE owner.
        pyre_object::w_type_set_cpython_immutabletype(w_type, true);
        if !tp.is_null() {
            (*tp).tp_flags |= TpFlags::PY_TPFLAGS_IMMUTABLETYPE;
        }
        crate::baseobjspace::mutated_absent(w_type);
    }
    0
}

/// `__module__`, which is what a fully qualified name is built from.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetModuleName(tp: *mut CPyTypeObject) -> *mut CPyObject {
    let w_type = interpreter_type(tp);
    if w_type.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    super::object::result(crate::baseobjspace::getattr_str(w_type, "__module__"))
}

/// `module.qualname`, except for the two module names a name is never
/// qualified by.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_GetFullyQualifiedName(tp: *mut CPyTypeObject) -> *mut CPyObject {
    let w_type = interpreter_type(tp);
    if w_type.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let w_qualname = unsafe { pyre_object::typeobject::w_type_get_name_obj(w_type) };
    let qualified = unsafe { pyre_object::w_str_get_wtf8(w_qualname) }.to_string();
    // Naming the type object once more through its mirror, which never moves:
    // the name above may have been minted, and an allocation is the one thing
    // that can leave the pointer read before it stale.
    let module = crate::baseobjspace::getattr_str(interpreter_type(tp), "__module__")
        .ok()
        .filter(|&w_module| unsafe { pyre_object::is_str(w_module) })
        .map(|w_module| unsafe { pyre_object::w_str_get_wtf8(w_module) }.to_string());
    // A missing `__module__` is not the caller's problem: the bare name is
    // still the answer, so the lookup's own error goes no further.
    super::pyerrors::take_pending_error();
    let name = match module.as_deref() {
        Some("builtins") | Some("__main__") | None => qualified,
        Some(module) => format!("{module}.{qualified}"),
    };
    pyobject::make_ref(pyre_object::w_str_new_managed(&name))
}

/// The metaclass a type built from a spec is an instance of —
/// `typeobject.c _PyType_FromMetaclass_impl`'s metaclass resolution.
///
/// Nothing here reads the metaclass's `tp_basicsize`, `tp_itemsize` or
/// `tp_alloc`, and no check of them belongs here: the storage a type object
/// gets is a `W_TypeObject` this runtime allocates, not a block the metaclass
/// describes, so those fields have no part to play and requiring anything of
/// them would invent a rule the spec does not have.
fn winning_metaclass(
    metaclass: *mut CPyTypeObject,
    base: *mut CPyTypeObject,
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let metaclass_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(interpreter_type(metaclass));
    let w_base = interpreter_type(base);
    let bases = match w_base.is_null() {
        true => PY_NULL,
        false => {
            let _ = roots.pin_root(w_base);
            let w_base = pyre_object::gc_roots::shadow_stack_get(metaclass_slot + 1);
            pyre_object::w_tuple_new(vec![w_base])
        }
    };
    // A null metaclass is `type`, and the base's own metatype wins over it —
    // which is how a type derived from one built through a metaclass keeps it.
    let w_winner = crate::call::calculate_metaclass(
        pyre_object::gc_roots::shadow_stack_get(metaclass_slot),
        bases,
    )?;
    let w_type_type = crate::typedef::w_type();
    if w_winner.is_null() || w_type_type.is_null() || std::ptr::eq(w_winner, w_type_type) {
        return Ok(PY_NULL);
    }
    let roots = pyre_object::gc_roots::push_roots();
    let winner_slot = pyre_object::gc_roots::shadow_stack_len();
    let w_winner = roots.pin_root(w_winner);
    if !unsafe { pyre_object::is_type(w_winner) }
        || !unsafe { crate::baseobjspace::issubtype_w(w_winner, w_type_type) }
    {
        return Err(crate::PyError::type_error(format!(
            "Metaclass '{}' is not a subclass of 'type'.",
            metatype_repr(w_winner)
        )));
    }
    // The winner, not the argument: a base's metatype that beat it is the one
    // that would take the allocation over.  The type is built here and its
    // `__new__` never runs, so a metaclass that defines one is refused.
    let w_winner = pyre_object::gc_roots::shadow_stack_get(winner_slot);
    if declares_tp_new(pyobject::as_pyobj(w_winner) as *mut CPyTypeObject) {
        return Err(crate::PyError::type_error(
            "Metaclasses with custom tp_new are not supported.",
        ));
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(winner_slot))
}

/// A metatype as the refusals above name it — `%R` of a type, which is
/// `<class 'name'>`.
///
/// Built rather than evaluated: this runs with an error already being
/// assembled, and a `__repr__` that can itself raise has nowhere to report to.
fn metatype_repr(w_metatype: PyObjectRef) -> String {
    format!("<class '{}'>", unsafe {
        pyre_object::w_type_get_name(w_metatype)
    })
}

/// `PyType_FromMetaclass` — the general form of [`PyType_FromModuleAndSpec`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyType_FromMetaclass(
    metaclass: *mut CPyTypeObject,
    module: *mut CPyObject,
    spec: *mut CPyTypeSpec,
    bases: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([module, bases]);
    super::object::result(from_spec(spec, bases, module, metaclass))
}

/// `PyErr_NewExceptionWithDoc` — `type(name, bases, {'__doc__': doc})`.
///
/// Built through the interpreter's own `type` rather than [`PyType_Ready`]:
/// an exception class needs the exception layout, which only the normal class
/// statement path installs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_NewExceptionWithDoc(
    name: *const c_char,
    doc: *const c_char,
    base: *mut CPyObject,
    dict: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([base, dict]);
    super::object::result(new_exception(name, doc, base, dict))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_NewException(
    name: *const c_char,
    base: *mut CPyObject,
    dict: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([base, dict]);
    super::object::result(new_exception(name, std::ptr::null(), base, dict))
}

fn new_exception(
    name: *const c_char,
    doc: *const c_char,
    base: *mut CPyObject,
    dict: *mut CPyObject,
) -> Result<PyObjectRef, crate::PyError> {
    if name.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyErr_NewException(): NULL name",
        ));
    }
    let qualified = unsafe { std::ffi::CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned();
    let Some((module, short)) = qualified.rsplit_once('.') else {
        return Err(crate::PyError::system_error(
            "PyErr_NewException(): name must be module.class",
        ));
    };
    let w_base = unsafe { pyobject::from_ref(base) };
    let w_dict = unsafe { pyobject::from_ref(dict) };

    let roots = pyre_object::gc_roots::push_roots();
    let base_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(if w_base.is_null() {
        crate::builtins::lookup_exc_class("Exception").unwrap_or(pyre_object::PY_NULL)
    } else {
        w_base
    });
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(if w_dict.is_null() {
        pyre_object::dictmultiobject::w_dict_new()
    } else {
        w_dict
    });
    let reload = |slot| pyre_object::gc_roots::shadow_stack_get(slot);

    let namespace = reload(dict_slot);
    store(namespace, "__module__", pyre_object::w_str_new(module));
    if !doc.is_null() {
        let text = text_or_none(doc);
        store(reload(dict_slot), DOC_KEY, text);
    }
    let bases = pyre_object::tupleobject::w_tuple_new(vec![reload(base_slot)]);
    let bases_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(bases);
    let w_name = pyre_object::w_str_new(short);
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_name);
    let w_type = crate::typedef::gettypeobject(&pyre_object::TYPE_TYPE);
    crate::call::call_function_impl_result(
        w_type,
        &[reload(name_slot), reload(bases_slot), reload(dict_slot)],
    )
}

pub(super) fn ensure_linked() {
    std::hint::black_box(&raw const CPY_MODULE_DEF_TYPE);
    ensure_type_mirrors_linked();
    std::hint::black_box(PyObject_Free as *const ());
    std::hint::black_box(PyObject_Del as *const ());
    std::hint::black_box(PyObject_GC_Del as *const ());
    std::hint::black_box(PyType_Ready as *const ());
    std::hint::black_box(PyType_Check as *const ());
    std::hint::black_box(PyType_IsSubtype as *const ());
    std::hint::black_box(PyType_GetFlags as *const ());
    std::hint::black_box(_PyType_Name as *const ());
    std::hint::black_box(PyType_GetDict as *const ());
    std::hint::black_box(PyType_FromSpec as *const ());
    std::hint::black_box(PyType_FromSpecWithBases as *const ());
    std::hint::black_box(PyType_FromModuleAndSpec as *const ());
    std::hint::black_box(PyType_GetSlot as *const ());
    std::hint::black_box(PyType_GetName as *const ());
    std::hint::black_box(PyType_GetQualName as *const ());
    std::hint::black_box(PyType_GenericAlloc as *const ());
    std::hint::black_box(PyType_GenericNew as *const ());
    std::hint::black_box(PyObject_Init as *const ());
    std::hint::black_box(PyErr_NewException as *const ());
    std::hint::black_box(PyErr_NewExceptionWithDoc as *const ());
    std::hint::black_box(PyType_FromMetaclass as *const ());
    std::hint::black_box(PyType_GetModule as *const ());
    std::hint::black_box(PyType_GetModuleState as *const ());
    std::hint::black_box(PyType_GetModuleByDef as *const ());
    std::hint::black_box(PyType_GetModuleName as *const ());
    std::hint::black_box(PyType_GetFullyQualifiedName as *const ());
    std::hint::black_box(PyType_GetBaseByToken as *const ());
    std::hint::black_box(PyType_GetTypeDataSize as *const ());
    std::hint::black_box(PyObject_GetTypeData as *const ());
    std::hint::black_box(PyObject_GetItemData as *const ());
    std::hint::black_box(PyType_Modified as *const ());
    std::hint::black_box(PyType_ClearCache as *const ());
    std::hint::black_box(PyType_Freeze as *const ());
}

#[cfg(test)]
mod tests {
    /// `PyType_FromModuleAndSpec` is `result_is_ll=True` upstream: its
    /// `PyType_GenericAlloc` reference is the one returned to C.  Pyre routes
    /// the interpreter object through `object::result`, so the corresponding
    /// reference is added there and must not already be present in the raw
    /// wrapper block.
    #[test]
    fn spec_type_wrapper_starts_without_a_second_c_reference() {
        assert_eq!(super::allocated_type().ob_base.ob_base.ob_refcnt, 0);
    }

    /// `type_attach` publishes `tp_name` before `tp_basicsize`. A mirror named
    /// but not yet sized is not a C declaration: the size still comes from the
    /// typedescr keyed by `w_type`.
    #[test]
    fn named_unsized_mirror_keeps_typedescr_basicsize() {
        crate::typedef::init_typeobjects();
        let w_list = crate::typedef::gettypeobject(&pyre_object::LIST_TYPE);
        let name = std::ffi::CString::new("isolated-layout").unwrap();
        let mut named = super::immortal_type();
        named.tp_name = name.as_ptr();
        unsafe {
            let layout = pyre_object::typeobject::w_type_get_layout_ptr(w_list);
            let w_type = pyre_object::typeobject::w_type_new(
                "isolated-layout",
                pyre_object::PY_NULL,
                std::ptr::null_mut(),
            );
            pyre_object::typeobject::w_type_set_layout(w_type, layout);
            pyre_object::typeobject::w_type_set_cpy_ref(w_type, (&raw mut named).cast());
            let sized = super::mirror_basicsize(w_type);
            pyre_object::typeobject::w_type_set_cpy_ref(w_type, std::ptr::null_mut());
            assert_eq!(sized, 7 * std::mem::size_of::<usize>() as isize);
        }
    }

    /// Every thunk in every pool must have an address of its own.
    ///
    /// The address is the only channel telling a thunk which `(owner, method)`
    /// it stands for, so two that share one are two slots bound to a single
    /// entry -- a wrong answer with nothing to show for it.  Identical bodies
    /// are what identical-code-folding merges, and a body that reads its index
    /// out of a table the program has not written yet compiles to the same
    /// code for every index.  Passing the index to the dispatcher as an
    /// argument is what keeps the bodies apart, and this is what says so.
    #[test]
    fn every_thunk_has_an_address_of_its_own() {
        let mut seen = std::collections::HashSet::new();
        let mut total = 0;
        for family in super::BOUND_FAMILIES {
            for index in 0..family.entries.len() {
                total += 1;
                let thunk = (family.thunk_at)(index) as usize;
                assert!(
                    seen.insert(thunk),
                    "two thunks share the address {thunk:#x}; the pool cannot tell them apart"
                );
            }
        }
        assert_eq!(seen.len(), total);
        assert!(
            total >= 674,
            "the pools are smaller than a stdlib import needs: {total}"
        );
    }

    /// A pool hands out each `(owner, method)` once and answers with the same
    /// thunk the second time, so a mirror refilled reuses what it had.
    #[test]
    fn binding_the_same_pair_twice_answers_with_one_thunk() {
        let owner = 0x1000 as *mut super::CPyObject;
        let first = super::bind_slot(&super::UNARY_BOUND, owner, "__repr__");
        let again = super::bind_slot(&super::UNARY_BOUND, owner, "__repr__");
        let other = super::bind_slot(&super::UNARY_BOUND, owner, "__str__");
        assert!(first.is_some());
        assert_eq!(first, again);
        assert_ne!(first, other);
        assert!(super::is_bound_thunk(first.unwrap()));
        assert!(!super::is_bound_thunk(
            super::interp_tp_repr as *const std::ffi::c_void
        ));
    }

    /// `PyType_FromSpec` reads the numbers an extension compiled against
    /// `typeslots.h` wrote into its slot array, so the two tables are one ABI
    /// in two places.  This walks the header and rejects any identifier whose
    /// number the Rust side spells differently, or does not spell at all.
    #[test]
    fn every_slot_id_is_the_number_the_header_gives_it() {
        const HEADER: &str = include_str!("../../../../include/pyre3.14t/typeslots.h");

        let mut checked = 0;
        for line in HEADER.lines() {
            let Some(rest) = line.strip_prefix("#define Py_") else {
                continue;
            };
            let Some((name, value)) = rest.split_once(' ') else {
                continue;
            };
            // The identifiers are `Py_<table>_<slot>`; nothing else in the
            // header is named that way with a bare integer for a body.
            let Some((table, _)) = name.split_once('_') else {
                continue;
            };
            if !matches!(table, "tp" | "nb" | "sq" | "mp" | "am" | "bf") {
                continue;
            }
            let Ok(value) = value.trim().parse::<std::ffi::c_int>() else {
                continue;
            };

            let upper = name.to_ascii_uppercase();
            let found = super::slot_id::ALL.iter().find(|(n, _)| *n == upper);
            let Some((_, ours)) = found else {
                panic!("Python.h defines Py_{name} = {value}, slot_id has no {upper}");
            };
            assert_eq!(
                *ours, value,
                "Py_{name} is {value} in Python.h and {ours} in slot_id"
            );
            checked += 1;
        }

        assert_eq!(
            checked,
            super::slot_id::ALL.len(),
            "the header defines {checked} slot identifiers and slot_id has {}",
            super::slot_id::ALL.len()
        );
    }

    /// A member's `flags` word is one thing two places spell: an extension
    /// compiled against `include/pyre3.14t/object.h` builds it, and this file
    /// reads it.  Nothing tied the two together, so this walks the header and
    /// rejects any flag whose bit the Rust side spells differently.
    #[test]
    fn every_member_flag_is_the_bit_the_header_gives_it() {
        use bitflags::Flags as _;

        const HEADER: &str = include_str!("../../../../include/pyre3.14t/object.h");

        // `Py_READONLY`, `Py_AUDIT_READ`, `Py_RELATIVE_OFFSET` -- decimal, and
        // the only `Py_` macros that are.  The type codes beside them are
        // `Py_T_*`, which this leaves alone.
        let mut header: Vec<(&str, std::ffi::c_int)> = Vec::new();
        for line in HEADER.lines() {
            let Some(rest) = line.strip_prefix("#define Py_") else {
                continue;
            };
            let Some((name, body)) = rest.split_once(' ') else {
                continue;
            };
            if name.starts_with("T_") || name.starts_with("TPFLAGS_") {
                continue;
            }
            let Ok(value) = body.trim().parse::<std::ffi::c_int>() else {
                continue;
            };
            header.push((name, value));
        }

        // A parser that read nothing would agree with every flag below, so the
        // floor comes first.
        assert!(
            header.len() >= 3,
            "object.h declares more plain Py_ constants than the {} this read",
            header.len()
        );

        for flag in super::MemberFlags::FLAGS {
            let name = flag
                .name()
                .strip_prefix("PY_")
                .expect("every flag is declared under the header's own name");
            let ours = flag.value().bits();
            let (_, theirs) = header
                .iter()
                .find(|(known, _)| known.eq_ignore_ascii_case(name))
                .unwrap_or_else(|| panic!("object.h declares no Py_{name}"));
            assert_eq!(
                ours, *theirs,
                "Py_{name} is {theirs} in object.h and {ours} here"
            );
        }
    }

    /// A `PySlot`'s `sl_flags` is one thing two places spell: an extension
    /// compiled against `include/pyre3.14t/object.h` builds it, and the two
    /// slot-array readers test it.  This walks the header and rejects any flag
    /// whose bit the Rust side spells differently.
    #[test]
    fn every_slot_flag_is_the_bit_the_header_gives_it() {
        use bitflags::Flags as _;

        const HEADER: &str = include_str!("../../../../include/pyre3.14t/object.h");

        let mut header: Vec<(&str, u16)> = Vec::new();
        for line in HEADER.lines() {
            let Some(rest) = line.strip_prefix("#define PySlot_") else {
                continue;
            };
            let Some((name, body)) = rest.split_once(' ') else {
                continue;
            };
            let body = body.trim();
            let Some(digits) = body.strip_prefix("0x") else {
                continue;
            };
            let Ok(value) = u16::from_str_radix(digits, 16) else {
                continue;
            };
            header.push((name, value));
        }

        assert_eq!(
            header.len(),
            super::SlotFlags::FLAGS.len(),
            "object.h declares {} PySlot_ flags and this file declares {}",
            header.len(),
            super::SlotFlags::FLAGS.len()
        );

        for flag in super::SlotFlags::FLAGS {
            let name = flag
                .name()
                .strip_prefix("PYSLOT_")
                .expect("every flag is declared under the header's own name");
            let ours = flag.value().bits();
            let (_, theirs) = header
                .iter()
                .find(|(known, _)| known.eq_ignore_ascii_case(name))
                .unwrap_or_else(|| panic!("object.h declares no PySlot_{name}"));
            assert_eq!(
                ours, *theirs,
                "PySlot_{name} is {theirs} in object.h and {ours} here"
            );
        }
    }

    /// A member's `type` is the other half of the same ABI: `object.h` numbers
    /// the codes as `Py_T_*`, `structmember.h` gives each the bare spelling an
    /// extension still writes, and this file switches on the number that
    /// arrives.  This resolves the header pair and rejects any code the Rust
    /// side numbers differently, or does not spell at all.
    #[test]
    fn every_member_type_code_is_the_number_the_header_gives_it() {
        const OBJECT_H: &str = include_str!("../../../../include/pyre3.14t/object.h");
        const STRUCTMEMBER_H: &str = include_str!("../../../../include/pyre3.14t/structmember.h");

        // `Py_T_SHORT 0` and the two the interpreter keeps to itself,
        // `_Py_T_OBJECT` and `_Py_T_NONE`.
        let mut numbered: Vec<(&str, std::ffi::c_int)> = Vec::new();
        for line in OBJECT_H.lines() {
            let rest = line
                .strip_prefix("#define Py_T_")
                .or_else(|| line.strip_prefix("#define _Py_T_"));
            let Some(rest) = rest else { continue };
            let Some((name, body)) = rest.split_once(' ') else {
                continue;
            };
            let Ok(value) = body.trim().parse::<std::ffi::c_int>() else {
                continue;
            };
            numbered.push((name, value));
        }

        // Each bare spelling names one of those, so the number an extension
        // compiles reaches this file through two headers.
        let mut header: Vec<(&str, std::ffi::c_int)> = Vec::new();
        for line in STRUCTMEMBER_H.lines() {
            let Some(rest) = line.strip_prefix("#define T_") else {
                continue;
            };
            let Some((name, body)) = rest.split_once(' ') else {
                continue;
            };
            let body = body.trim();
            let alias = body
                .strip_prefix("Py_T_")
                .or_else(|| body.strip_prefix("_Py_T_"))
                .unwrap_or_else(|| panic!("T_{name} is defined as {body}, not a Py_T_ code"));
            let (_, value) = numbered
                .iter()
                .find(|(known, _)| *known == alias)
                .unwrap_or_else(|| panic!("T_{name} names Py_T_{alias}, which object.h lacks"));
            header.push((name, *value));
        }

        assert_eq!(
            header.len(),
            super::ALL_MEMBER_TYPE_CODES.len(),
            "the headers declare {} type codes and this file declares {}",
            header.len(),
            super::ALL_MEMBER_TYPE_CODES.len()
        );

        for (name, theirs) in &header {
            let found = super::ALL_MEMBER_TYPE_CODES
                .iter()
                .find(|(known, _)| known.strip_prefix("T_") == Some(name));
            let Some((_, ours)) = found else {
                panic!("structmember.h defines T_{name} = {theirs}, and this file has no T_{name}");
            };
            assert_eq!(
                ours, theirs,
                "T_{name} is {theirs} in the headers and {ours} here"
            );
        }
    }
}
