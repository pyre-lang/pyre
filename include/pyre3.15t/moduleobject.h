/* `PyModuleDef` and the module constructors.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_MODULEOBJECT_H
#define PYRE_MODULEOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
typedef struct PyModuleDef_Base {
    PyObject ob_base;
    PyObject *(*m_init)(void);
    Py_ssize_t m_index;
    PyObject *m_copy;
} PyModuleDef_Base;

struct PyModuleDef_Slot {
    int slot;
    void *value;
};

#define Py_mod_create 1
#define Py_mod_exec 2
#define Py_mod_multiple_interpreters 3
#define Py_mod_gil 4

/* The highest slot number defined above, so that anything past it is a slot
   no reader claims.  `PyModule_FromDefAndSpec` rejects one, and a def that
   wants to be rejected names `_Py_mod_LAST_SLOT + 1`. */
#define _Py_mod_LAST_SLOT 4

/* for Py_mod_multiple_interpreters: */
#define Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED ((void *)0)
#define Py_MOD_MULTIPLE_INTERPRETERS_SUPPORTED ((void *)1)
#define Py_MOD_PER_INTERPRETER_GIL_SUPPORTED ((void *)2)

/* for Py_mod_gil: */
#define Py_MOD_GIL_USED ((void *)0)
#define Py_MOD_GIL_NOT_USED ((void *)1)

struct PyModuleDef {
    PyModuleDef_Base m_base;
    const char *m_name;
    const char *m_doc;
    Py_ssize_t m_size;
    PyMethodDef *m_methods;
    PyModuleDef_Slot *m_slots;
    traverseproc m_traverse;
    inquiry m_clear;
    freefunc m_free;
};

#define PyModuleDef_HEAD_INIT \
    { { 1, 0, 0, NULL }, NULL, 0, NULL }

#define PyModule_Create(module) PyModule_Create2((module), PYTHON_API_VERSION)

#define PyModule_FromDefAndSpec(module, spec) \
    PyModule_FromDefAndSpec2((module), (spec), PYTHON_API_VERSION)

#define PyModule_AddIntMacro(module, macro) \
    PyModule_AddIntConstant((module), #macro, (macro))
#define PyModule_AddStringMacro(module, macro) \
    PyModule_AddStringConstant((module), #macro, (macro))
/* `PyModule_AddType` readies the type and binds it under the part of
   `tp_name` after the last dot. */
static inline int PyModule_AddType(PyObject *module, PyTypeObject *type)
{
    const char *name;
    const char *dot;
    if (PyType_Ready(type) < 0) {
        return -1;
    }
    name = type->tp_name;
    dot = strrchr(name, '.');
    if (dot != NULL) {
        name = dot + 1;
    }
    Py_INCREF(type);
    if (PyModule_AddObject(module, name, (PyObject *)type) < 0) {
        Py_DECREF(type);
        return -1;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_MODULEOBJECT_H */
