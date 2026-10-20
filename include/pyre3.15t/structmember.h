/* `PyMemberDef` and its type codes.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_STRUCTMEMBER_H
#define PYRE_STRUCTMEMBER_H

#ifdef __cplusplus
extern "C" {
#endif
/* The `structmember.h` spellings, which are what most sources still use. */
#define T_SHORT Py_T_SHORT
#define T_INT Py_T_INT
#define T_LONG Py_T_LONG
#define T_FLOAT Py_T_FLOAT
#define T_DOUBLE Py_T_DOUBLE
#define T_STRING Py_T_STRING
#define T_OBJECT _Py_T_OBJECT
#define T_OBJECT_EX Py_T_OBJECT_EX
#define T_CHAR Py_T_CHAR
#define T_BYTE Py_T_BYTE
#define T_UBYTE Py_T_UBYTE
#define T_USHORT Py_T_USHORT
#define T_UINT Py_T_UINT
#define T_ULONG Py_T_ULONG
#define T_STRING_INPLACE Py_T_STRING_INPLACE
#define T_BOOL Py_T_BOOL
#define T_LONGLONG Py_T_LONGLONG
#define T_ULONGLONG Py_T_ULONGLONG
#define T_PYSSIZET Py_T_PYSSIZET
#define T_NONE _Py_T_NONE
#define READONLY Py_READONLY

struct PyGetSetDef {
    const char *name;
    getter get;
    setter set;
    const char *doc;
    void *closure;
};

struct _typeobject {
    PyObject_VAR_HEAD
    const char *tp_name;
    Py_ssize_t tp_basicsize;
    Py_ssize_t tp_itemsize;
    destructor tp_dealloc;
    Py_ssize_t tp_vectorcall_offset;
    getattrfunc tp_getattr;
    setattrfunc tp_setattr;
    PyAsyncMethods *tp_as_async;
    reprfunc tp_repr;
    PyNumberMethods *tp_as_number;
    PySequenceMethods *tp_as_sequence;
    PyMappingMethods *tp_as_mapping;
    hashfunc tp_hash;
    ternaryfunc tp_call;
    reprfunc tp_str;
    getattrofunc tp_getattro;
    setattrofunc tp_setattro;
    PyBufferProcs *tp_as_buffer;
    unsigned long tp_flags;
    const char *tp_doc;
    traverseproc tp_traverse;
    inquiry tp_clear;
    richcmpfunc tp_richcompare;
    Py_ssize_t tp_weaklistoffset;
    getiterfunc tp_iter;
    iternextfunc tp_iternext;
    PyMethodDef *tp_methods;
    PyMemberDef *tp_members;
    PyGetSetDef *tp_getset;
    PyTypeObject *tp_base;
    PyObject *tp_dict;
    descrgetfunc tp_descr_get;
    descrsetfunc tp_descr_set;
    Py_ssize_t tp_dictoffset;
    initproc tp_init;
    allocfunc tp_alloc;
    newfunc tp_new;
    freefunc tp_free;
    inquiry tp_is_gc;
    PyObject *tp_bases;
    PyObject *tp_mro;
    PyObject *tp_cache;
    void *tp_subclasses;
    PyObject *tp_weaklist;
    destructor tp_del;
    unsigned int tp_version_tag;
    destructor tp_finalize;
    vectorcallfunc tp_vectorcall;
    unsigned char tp_watched;
    uint16_t tp_versions_used;
};

/* The block every type is mirrored into.  The five method suites sit inside
   it so that a type's own `tp_as_*` can name them without a second
   allocation, and the four references past them are what a heap type answers
   `__name__`, `__slots__`, `__qualname__` and `__module__` with. */
typedef struct _heaptypeobject {
    PyTypeObject ht_type;
    PyAsyncMethods as_async;
    PyNumberMethods as_number;
    PyMappingMethods as_mapping;
    PySequenceMethods as_sequence;
    PyBufferProcs as_buffer;
    PyObject *ht_name, *ht_slots, *ht_qualname;
    PyObject *ht_module;
} PyHeapTypeObject;

/* The members a heap type declared, which follow its block rather than
   sitting in it. */
#define PyHeapType_GET_MEMBERS(etype) \
    ((PyMemberDef *)(((char *)etype) + Py_TYPE(etype)->tp_basicsize))

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_STRUCTMEMBER_H */
