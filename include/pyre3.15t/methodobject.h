/* `PyMethodDef` and the calling conventions it selects.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_METHODOBJECT_H
#define PYRE_METHODOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
struct PyMethodDef {
    const char *ml_name;
    PyCFunction ml_meth;
    int ml_flags;
    const char *ml_doc;
};

/* A `METH_NOARGS` or `METH_O` function is written with the parameters its own
   convention passes, which need not be the two `PyCFunction` names.  Calling
   one through a mismatched pointer type is undefined, so a table entry says
   here that the cast is deliberate (bpo-33012). */
#define _PyCFunction_CAST(func) _Py_FUNC_CAST(PyCFunction, func)

#define METH_VARARGS 0x0001
#define METH_KEYWORDS 0x0002
#define METH_NOARGS 0x0004
#define METH_O 0x0008
#define METH_CLASS 0x0010
#define METH_STATIC 0x0020
#define METH_COEXIST 0x0040
#define METH_FASTCALL 0x0080
/* A method that is handed the class defining it beside its receiver, and so
   carries a `PyCMethodObject` rather than a `PyCFunctionObject`. */
#define METH_METHOD 0x0200

/* The two layouts a C function carries.
 *
 * A carrier's block holds these fields, so a source spelling them as field
 * reads gets what the entry points below answer -- cffi's `_cpyextfunc_get`
 * reads `m_ml` and `m_module` that way, and the module read is an identity
 * test against the reference it handed `PyCFunction_NewEx`.  The macros stay
 * calls because a type derived from `PyCFunction_Type` embeds one of these in
 * its own instance layout and fills it itself. */
typedef struct {
    PyObject_HEAD
    PyMethodDef *m_ml;
    PyObject *m_self;
    PyObject *m_module;
    PyObject *m_weakreflist;
    vectorcallfunc vectorcall;
} PyCFunctionObject;

typedef struct {
    PyCFunctionObject func;
    PyTypeObject *mm_class;
} PyCMethodObject;

/* The type an extension's own `PyMethodDef` becomes, and so the base a type
   derived from it names.  An interpreter builtin such as `len` is a different
   `builtin_function_or_method` that no symbol here names, and
   `PyCFunction_Check` answers no for it: it carries no `PyMethodDef`, so
   `PyCFunction_GetFunction` would have nothing to hand back. */
PyAPI_DATA(PyTypeObject) PyCFunction_Type;

/* The `METH_METHOD` carrier, which holds the defining class beside everything
   `PyCFunction_Type` holds and so derives from it. */
PyAPI_DATA(PyTypeObject) PyCMethod_Type;

#define PyCFunction_Check(op) PyObject_TypeCheck((op), &PyCFunction_Type)
#define PyCFunction_CheckExact(op) Py_IS_TYPE((op), &PyCFunction_Type)
#define PyCMethod_CheckExact(op) Py_IS_TYPE((op), &PyCMethod_Type)
#define PyCMethod_Check(op) PyObject_TypeCheck((op), &PyCMethod_Type)

#define PyCFunction_GET_FUNCTION(func) PyCFunction_GetFunction((PyObject *)(func))
#define PyCFunction_GET_SELF(func) PyCFunction_GetSelf((PyObject *)(func))
#define PyCFunction_GET_FLAGS(func) PyCFunction_GetFlags((PyObject *)(func))
/* `PyCMethod_GetClass` is the name the entry point carries; the reference
   header reaches the same answer by reading `mm_class` behind a `METH_METHOD`
   test, which a mirror has no field to do. */
#define PyCFunction_GET_CLASS(func) PyCMethod_GetClass((PyObject *)(func))

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_METHODOBJECT_H */
