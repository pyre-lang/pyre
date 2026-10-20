/* `PyObject`, the type object, and the protocols a type answers.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_OBJECT_H
#define PYRE_OBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* `ob_pyre_link` is the interpreter object this mirror stands for.  It is
   opaque to C: the interpreter moves its objects, and only the runtime may
   read or write the slot. */
struct _object {
    Py_ssize_t ob_refcnt;
    Py_ssize_t ob_pyre_link;
    Py_ssize_t ob_pyre_pad;
    PyTypeObject *ob_type;
};

#define PyObject_HEAD PyObject ob_base;
#define PyObject_HEAD_INIT(type) { 0, 0, 0, type },
#define Py_REFCNT(ob) (((PyObject *)(ob))->ob_refcnt)

/* `Py_TYPE` is exported as well as spelled here, because an extension that
   declares the prototype itself rather than including this header still has to
   find a symbol.  The declaration comes before the macro that replaces it, the
   way `lock.h` declares `PyMutex_Lock` ahead of its own inline fast path. */
PyAPI_FUNC(PyTypeObject *) Py_TYPE(PyObject *);
#define Py_TYPE(ob) (((PyObject *)(ob))->ob_type)

/* Exported as well as spelled here, for the reason `Py_TYPE` is. */
PyAPI_FUNC(int) Py_IS_TYPE(PyObject *, PyTypeObject *);
#define Py_IS_TYPE(ob, type) (Py_TYPE(ob) == (type))
/* bpo-39573: writing `ob_type` and `ob_size` goes through these. */
#define Py_SET_TYPE(ob, type) ((void)(Py_TYPE(ob) = (type)))

/* The reference-counting macros are in `refcount.h`, which `Python.h` includes
   after the declarations they expand to. */

/* The singletons.  Each is a mirror the runtime binds to its interpreter
   object before the first `PyInit_*` runs, so pointer comparison against them
   is the identity test C code expects it to be. */
PyAPI_DATA(PyObject) _Py_NoneStruct;
PyAPI_DATA(PyObject) _Py_TrueStruct;
PyAPI_DATA(PyObject) _Py_FalseStruct;
PyAPI_DATA(PyObject) _Py_NotImplementedStruct;
PyAPI_DATA(PyObject) _Py_EllipsisObject;
#define Py_None (&_Py_NoneStruct)
#define Py_True (&_Py_TrueStruct)
#define Py_False (&_Py_FalseStruct)
#define Py_NotImplemented (&_Py_NotImplementedStruct)
#define Py_Ellipsis (&_Py_EllipsisObject)
#define Py_RETURN_NONE do { Py_INCREF(Py_None); return Py_None; } while (0)
#define Py_RETURN_TRUE do { Py_INCREF(Py_True); return Py_True; } while (0)
#define Py_RETURN_FALSE do { Py_INCREF(Py_False); return Py_False; } while (0)
#define Py_RETURN_NOTIMPLEMENTED \
    do { Py_INCREF(Py_NotImplemented); return Py_NotImplemented; } while (0)
#define Py_Is(x, y) ((x) == (y))
#define Py_IsNone(x) Py_Is((x), Py_None)
#define Py_IsTrue(x) Py_Is((x), Py_True)
#define Py_IsFalse(x) Py_Is((x), Py_False)
/* Types. */
struct PyVarObject {
    PyObject ob_base;
    Py_ssize_t ob_size;
};

#define PyObject_VAR_HEAD PyVarObject ob_base;
#define PyVarObject_HEAD_INIT(type, size) { PyObject_HEAD_INIT(type) size },
#define Py_SIZE(ob) (((PyVarObject *)(ob))->ob_size)
#define Py_SET_SIZE(ob, size) ((void)(Py_SIZE(ob) = (size)))

typedef void (*destructor)(PyObject *);
typedef PyObject *(*unaryfunc)(PyObject *);
typedef PyObject *(*binaryfunc)(PyObject *, PyObject *);
typedef PyObject *(*ternaryfunc)(PyObject *, PyObject *, PyObject *);
typedef PyObject *(*reprfunc)(PyObject *);
typedef Py_hash_t (*hashfunc)(PyObject *);
typedef PyObject *(*getattrfunc)(PyObject *, char *);
typedef int (*setattrfunc)(PyObject *, char *, PyObject *);
typedef PyObject *(*getattrofunc)(PyObject *, PyObject *);
typedef int (*setattrofunc)(PyObject *, PyObject *, PyObject *);
typedef PyObject *(*richcmpfunc)(PyObject *, PyObject *, int);
typedef PyObject *(*getiterfunc)(PyObject *);
typedef PyObject *(*iternextfunc)(PyObject *);
typedef PyObject *(*descrgetfunc)(PyObject *, PyObject *, PyObject *);
typedef int (*descrsetfunc)(PyObject *, PyObject *, PyObject *);
typedef int (*initproc)(PyObject *, PyObject *, PyObject *);
typedef PyObject *(*newfunc)(PyTypeObject *, PyObject *, PyObject *);
typedef PyObject *(*allocfunc)(PyTypeObject *, Py_ssize_t);
typedef PyObject *(*getter)(PyObject *, void *);
typedef int (*setter)(PyObject *, PyObject *, void *);
typedef PyObject *(*vectorcallfunc)(PyObject *, PyObject *const *, size_t, PyObject *);

typedef Py_ssize_t (*lenfunc)(PyObject *);
typedef PyObject *(*ssizeargfunc)(PyObject *, Py_ssize_t);
/* The two-index slot no type carries any more.  It is named because an
   extension written against the older layout still spells the type, and
   because a `PyType_Slot` array may hold one. */
typedef PyObject *(*ssizessizeargfunc)(PyObject *, Py_ssize_t, Py_ssize_t);
typedef int (*ssizeobjargproc)(PyObject *, Py_ssize_t, PyObject *);
typedef int (*ssizessizeobjargproc)(PyObject *, Py_ssize_t, Py_ssize_t, PyObject *);
typedef int (*objobjproc)(PyObject *, PyObject *);
typedef int (*objobjargproc)(PyObject *, PyObject *, PyObject *);

typedef struct {
    binaryfunc nb_add;
    binaryfunc nb_subtract;
    binaryfunc nb_multiply;
    binaryfunc nb_remainder;
    binaryfunc nb_divmod;
    ternaryfunc nb_power;
    unaryfunc nb_negative;
    unaryfunc nb_positive;
    unaryfunc nb_absolute;
    inquiry nb_bool;
    unaryfunc nb_invert;
    binaryfunc nb_lshift;
    binaryfunc nb_rshift;
    binaryfunc nb_and;
    binaryfunc nb_xor;
    binaryfunc nb_or;
    unaryfunc nb_int;
    void *nb_reserved;
    unaryfunc nb_float;
    binaryfunc nb_inplace_add;
    binaryfunc nb_inplace_subtract;
    binaryfunc nb_inplace_multiply;
    binaryfunc nb_inplace_remainder;
    ternaryfunc nb_inplace_power;
    binaryfunc nb_inplace_lshift;
    binaryfunc nb_inplace_rshift;
    binaryfunc nb_inplace_and;
    binaryfunc nb_inplace_xor;
    binaryfunc nb_inplace_or;
    binaryfunc nb_floor_divide;
    binaryfunc nb_true_divide;
    binaryfunc nb_inplace_floor_divide;
    binaryfunc nb_inplace_true_divide;
    unaryfunc nb_index;
    binaryfunc nb_matrix_multiply;
    binaryfunc nb_inplace_matrix_multiply;
} PyNumberMethods;

typedef struct {
    lenfunc sq_length;
    binaryfunc sq_concat;
    ssizeargfunc sq_repeat;
    ssizeargfunc sq_item;
    void *was_sq_slice;
    ssizeobjargproc sq_ass_item;
    void *was_sq_ass_slice;
    objobjproc sq_contains;
    binaryfunc sq_inplace_concat;
    ssizeargfunc sq_inplace_repeat;
} PySequenceMethods;

typedef struct {
    lenfunc mp_length;
    binaryfunc mp_subscript;
    objobjargproc mp_ass_subscript;
} PyMappingMethods;

typedef enum {
    PYGEN_RETURN = 0,
    PYGEN_ERROR = -1,
    PYGEN_NEXT = 1
} PySendResult;

typedef PySendResult (*sendfunc)(PyObject *, PyObject *, PyObject **);

typedef struct {
    unaryfunc am_await;
    unaryfunc am_aiter;
    unaryfunc am_anext;
    sendfunc am_send;
} PyAsyncMethods;

struct Py_buffer {
    void *buf;
    PyObject *obj;
    Py_ssize_t len;
    Py_ssize_t itemsize;
    int readonly;
    int ndim;
    char *format;
    Py_ssize_t *shape;
    Py_ssize_t *strides;
    Py_ssize_t *suboffsets;
    void *internal;
};

#define PyBUF_MAX_NDIM 64

#define PyBUF_SIMPLE 0
#define PyBUF_WRITABLE 0x0001
#define PyBUF_WRITEABLE PyBUF_WRITABLE
#define PyBUF_FORMAT 0x0004
#define PyBUF_ND 0x0008
#define PyBUF_STRIDES (0x0010 | PyBUF_ND)
#define PyBUF_C_CONTIGUOUS (0x0020 | PyBUF_STRIDES)
#define PyBUF_F_CONTIGUOUS (0x0040 | PyBUF_STRIDES)
#define PyBUF_ANY_CONTIGUOUS (0x0080 | PyBUF_STRIDES)
#define PyBUF_INDIRECT (0x0100 | PyBUF_STRIDES)

#define PyBUF_CONTIG (PyBUF_ND | PyBUF_WRITABLE)
#define PyBUF_CONTIG_RO (PyBUF_ND)
#define PyBUF_STRIDED (PyBUF_STRIDES | PyBUF_WRITABLE)
#define PyBUF_STRIDED_RO (PyBUF_STRIDES)
#define PyBUF_RECORDS (PyBUF_STRIDES | PyBUF_WRITABLE | PyBUF_FORMAT)
#define PyBUF_RECORDS_RO (PyBUF_STRIDES | PyBUF_FORMAT)
#define PyBUF_FULL (PyBUF_INDIRECT | PyBUF_WRITABLE | PyBUF_FORMAT)
#define PyBUF_FULL_RO (PyBUF_INDIRECT | PyBUF_FORMAT)

#define PyBUF_READ 0x100
#define PyBUF_WRITE 0x200

typedef int (*getbufferproc)(PyObject *, Py_buffer *, int);
typedef void (*releasebufferproc)(PyObject *, Py_buffer *);

typedef struct {
    getbufferproc bf_getbuffer;
    releasebufferproc bf_releasebuffer;
} PyBufferProcs;

struct PyMemberDef {
    const char *name;
    int type;
    Py_ssize_t offset;
    int flags;
    const char *doc;
};

#define Py_T_SHORT 0
#define Py_T_INT 1
#define Py_T_LONG 2
#define Py_T_FLOAT 3
#define Py_T_DOUBLE 4
#define Py_T_STRING 5
#define _Py_T_OBJECT 6
#define Py_T_OBJECT_EX 16
#define Py_T_CHAR 7
#define Py_T_BYTE 8
#define Py_T_UBYTE 9
#define Py_T_USHORT 10
#define Py_T_UINT 11
#define Py_T_ULONG 12
#define Py_T_STRING_INPLACE 13
#define Py_T_BOOL 14
#define Py_T_LONGLONG 17
#define Py_T_ULONGLONG 18
#define Py_T_PYSSIZET 19
#define _Py_T_NONE 20
#define Py_READONLY 1
#define Py_AUDIT_READ 2
#define Py_RELATIVE_OFFSET 8
/* Type flags.  `Py_TPFLAGS_DEFAULT` is 0 here as it is upstream, where it
   reduces to `Py_TPFLAGS_HAVE_STACKLESS_EXTENSION | 0` and that half is 0 off
   Stackless.

   Only the eight `*_SUBCLASS` flags and the three pyre sets itself -- READY,
   READYING and HEAPTYPE -- are ever filled in by pyre. The rest are declared
   so that an extension naming one compiles, and carry whatever the extension
   put there; a flag describing an object layout pyre does not have (the
   MANAGED/PREHEADER/INLINE_VALUES/ITEMS_AT_END group) is never acted on. */
#define Py_TPFLAGS_DEFAULT 0UL
#define Py_TPFLAGS_MANAGED_WEAKREF (1UL << 3)
#define Py_TPFLAGS_MANAGED_DICT (1UL << 4)
#define Py_TPFLAGS_PREHEADER (Py_TPFLAGS_MANAGED_WEAKREF | Py_TPFLAGS_MANAGED_DICT)
#define Py_TPFLAGS_SEQUENCE (1UL << 5)
#define Py_TPFLAGS_MAPPING (1UL << 6)
#define Py_TPFLAGS_DISALLOW_INSTANTIATION (1UL << 7)
#define Py_TPFLAGS_IMMUTABLETYPE (1UL << 8)
#define Py_TPFLAGS_HEAPTYPE (1UL << 9)
#define Py_TPFLAGS_BASETYPE (1UL << 10)
#define Py_TPFLAGS_HAVE_VECTORCALL (1UL << 11)
#define Py_TPFLAGS_READY (1UL << 12)
#define Py_TPFLAGS_READYING (1UL << 13)
#define Py_TPFLAGS_HAVE_GC (1UL << 14)
#define Py_TPFLAGS_HAVE_STACKLESS_EXTENSION 0UL
#define Py_TPFLAGS_METHOD_DESCRIPTOR (1UL << 17)
#define Py_TPFLAGS_HAVE_VERSION_TAG (1UL << 18)
#define Py_TPFLAGS_VALID_VERSION_TAG (1UL << 19)
#define Py_TPFLAGS_IS_ABSTRACT (1UL << 20)
#define Py_TPFLAGS_INLINE_VALUES (1UL << 2)
#define Py_TPFLAGS_ITEMS_AT_END (1UL << 23)
#define Py_TPFLAGS_LONG_SUBCLASS (1UL << 24)
#define Py_TPFLAGS_LIST_SUBCLASS (1UL << 25)
#define Py_TPFLAGS_TUPLE_SUBCLASS (1UL << 26)
#define Py_TPFLAGS_BYTES_SUBCLASS (1UL << 27)
#define Py_TPFLAGS_UNICODE_SUBCLASS (1UL << 28)
#define Py_TPFLAGS_DICT_SUBCLASS (1UL << 29)
#define Py_TPFLAGS_BASE_EXC_SUBCLASS (1UL << 30)
#define Py_TPFLAGS_TYPE_SUBCLASS (1UL << 31)
#define Py_TPFLAGS_HAVE_FINALIZE (1UL << 0)

#define PyType_HasFeature(type, feature) ((PyType_GetFlags(type) & (feature)) != 0)
#define PyType_FastSubclass(type, flag) PyType_HasFeature(type, flag)

/* `tp_richcompare` operations. */
#define Py_LT 0
#define Py_LE 1
#define Py_EQ 2
#define Py_NE 3
#define Py_GT 4
#define Py_GE 5
/* No string quotes etc. `PyFile_WriteObject` writes `str` when this is set. */
#define Py_PRINT_RAW 1

/* Heap types.  Only single inheritance is supported: a spec naming more than
   one base is rejected rather than silently losing the rest. */
struct PyType_Slot {
    int slot;
    void *pfunc;
};

struct PyType_Spec {
    const char *name;
    int basicsize;
    int itemsize;
    unsigned int flags;
    PyType_Slot *slots;
};

/* `PySlot` — one entry of the identifier-keyed array `PyType_FromSlots` reads,
   which carries what a `PyType_Spec`'s fixed fields carry and points at a
   `PyType_Slot` array for the rest.  A zero `sl_id` ends the array. */
struct PySlot {
    uint16_t sl_id;
    uint16_t sl_flags;
    uint32_t sl_reserved;
    union {
        void *sl_ptr;
        void (*sl_func)(void);
        Py_ssize_t sl_size;
        int64_t sl_int64;
        uint64_t sl_uint64;
    };
};

/* `PySlot.sl_flags`. */
#define PySlot_OPTIONAL 0x01
#define PySlot_STATIC 0x02
#define PySlot_INTPTR 0x04

/* `PyType_GetModuleByDef` needs `PyModuleDef`, so it is declared with it. */

#define PyObject_TypeCheck(ob, type) \
    (Py_TYPE(ob) == (type) || PyType_IsSubtype(Py_TYPE(ob), (type)))

/* The built-in types, as storage rather than as a pointer: C names them by
   address (`&PyList_Type`, `tp_base = &PyDict_Type`, `PyObject_TypeCheck(x,
   &PyType_Type)`), so each is a block the runtime binds to its interpreter
   type before the first `PyInit_*` runs.  A name this build has no type for
   keeps a NULL `tp_name`. */
PyAPI_DATA(PyTypeObject) PyType_Type;
PyAPI_DATA(PyTypeObject) PyBaseObject_Type;
PyAPI_DATA(PyTypeObject) PySuper_Type;
/* The data types. */
PyAPI_DATA(PyTypeObject) PyBool_Type;
PyAPI_DATA(PyTypeObject) PyByteArray_Type;
PyAPI_DATA(PyTypeObject) PyBytes_Type;
PyAPI_DATA(PyTypeObject) PyComplex_Type;
PyAPI_DATA(PyTypeObject) PyDict_Type;
PyAPI_DATA(PyTypeObject) PyEllipsis_Type;
PyAPI_DATA(PyTypeObject) PyFloat_Type;
PyAPI_DATA(PyTypeObject) PyFrozenSet_Type;
PyAPI_DATA(PyTypeObject) PyList_Type;
PyAPI_DATA(PyTypeObject) PyLong_Type;
PyAPI_DATA(PyTypeObject) PyMemoryView_Type;
PyAPI_DATA(PyTypeObject) PyModule_Type;
PyAPI_DATA(PyTypeObject) PySet_Type;
PyAPI_DATA(PyTypeObject) PySlice_Type;
PyAPI_DATA(PyTypeObject) PyTuple_Type;
PyAPI_DATA(PyTypeObject) PyUnicode_Type;
PyAPI_DATA(PyTypeObject) Py_GenericAliasType;
/* The dict views. */
PyAPI_DATA(PyTypeObject) PyDictProxy_Type;
PyAPI_DATA(PyTypeObject) PyDictItems_Type;
PyAPI_DATA(PyTypeObject) PyDictKeys_Type;
PyAPI_DATA(PyTypeObject) PyDictValues_Type;
/* Functions, methods and descriptors. */
PyAPI_DATA(PyTypeObject) PyClassMethodDescr_Type;
PyAPI_DATA(PyTypeObject) PyClassMethod_Type;
PyAPI_DATA(PyTypeObject) PyFunction_Type;
PyAPI_DATA(PyTypeObject) PyGetSetDescr_Type;
PyAPI_DATA(PyTypeObject) PyMemberDescr_Type;
PyAPI_DATA(PyTypeObject) PyMethodDescr_Type;
PyAPI_DATA(PyTypeObject) PyMethod_Type;
PyAPI_DATA(PyTypeObject) PyProperty_Type;
PyAPI_DATA(PyTypeObject) PyStaticMethod_Type;
PyAPI_DATA(PyTypeObject) PyWrapperDescr_Type;
/* The built-ins that are types. */
PyAPI_DATA(PyTypeObject) PyEnum_Type;
PyAPI_DATA(PyTypeObject) PyFilter_Type;
PyAPI_DATA(PyTypeObject) PyMap_Type;
PyAPI_DATA(PyTypeObject) PyRange_Type;
PyAPI_DATA(PyTypeObject) PyReversed_Type;
PyAPI_DATA(PyTypeObject) PyZip_Type;
/* Frames, code and what a call leaves behind. */
PyAPI_DATA(PyTypeObject) PyAsyncGen_Type;
PyAPI_DATA(PyTypeObject) PyCell_Type;
PyAPI_DATA(PyTypeObject) PyCode_Type;
PyAPI_DATA(PyTypeObject) PyCoro_Type;
PyAPI_DATA(PyTypeObject) PyFrame_Type;
PyAPI_DATA(PyTypeObject) PyGen_Type;
PyAPI_DATA(PyTypeObject) PyTraceBack_Type;
PyAPI_DATA(PyTypeObject) _PyAsyncGenASend_Type;
PyAPI_DATA(PyTypeObject) _PyWeakref_RefType;
/* Objects. */

/* The `object.__getattribute__` / `__setattr__` / `__dict__` terminals, which
   are what `tp_getattro`, `tp_setattro` and the `__dict__` getset default to.
   A NULL `value` is the deletion spelling for both setters. */

/* Every object pyre holds is reachable by its collector and nothing runs
   `tp_finalize`, so these two are constants. */

/* The buffer protocol. */

/* The objects `Py_GetConstant` answers with, named by the identifier it
   takes. */
#define Py_CONSTANT_NONE 0
#define Py_CONSTANT_FALSE 1
#define Py_CONSTANT_TRUE 2
#define Py_CONSTANT_ELLIPSIS 3
#define Py_CONSTANT_NOT_IMPLEMENTED 4
#define Py_CONSTANT_ZERO 5
#define Py_CONSTANT_ONE 6
#define Py_CONSTANT_EMPTY_STR 7
#define Py_CONSTANT_EMPTY_BYTES 8
#define Py_CONSTANT_EMPTY_TUPLE 9

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_OBJECT_H */
