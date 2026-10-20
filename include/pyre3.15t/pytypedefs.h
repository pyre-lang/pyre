/* Forward names, so a declaration never waits on a definition.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_PYTYPEDEFS_H
#define PYRE_PYTYPEDEFS_H

#ifdef __cplusplus
extern "C" {
#endif
typedef struct _object PyObject;
typedef struct _typeobject PyTypeObject;
typedef PyObject *(*PyCFunction)(PyObject *, PyObject *);
typedef PyObject *(*PyCFunctionWithKeywords)(PyObject *, PyObject *, PyObject *);
typedef PyObject *(*_PyCFunctionFast)(PyObject *, PyObject *const *, Py_ssize_t);
typedef PyObject *(*_PyCFunctionFastWithKeywords)(PyObject *, PyObject *const *,
                                                  Py_ssize_t, PyObject *);
typedef PyObject *(*PyCMethod)(PyObject *, PyTypeObject *, PyObject *const *,
                               Py_ssize_t, PyObject *);
typedef int (*visitproc)(PyObject *, void *);
typedef int (*traverseproc)(PyObject *, visitproc, void *);
typedef int (*inquiry)(PyObject *);
typedef void (*freefunc)(void *);
typedef int (*converter)(PyObject *, void *);

/* Named here because a declaration only ever takes a pointer to one, and
   the header that defines each is included after `pyre_decl.h`. */
typedef struct PyVarObject PyVarObject;
typedef struct PyModuleDef PyModuleDef;
typedef struct PyModuleDef_Slot PyModuleDef_Slot;
typedef struct PyMethodDef PyMethodDef;
typedef struct PyMemberDef PyMemberDef;
typedef struct PyGetSetDef PyGetSetDef;
typedef struct PyType_Slot PyType_Slot;
typedef struct PyType_Spec PyType_Spec;
typedef struct PySlot PySlot;
typedef struct Py_buffer Py_buffer;
/* Neither an `int` nor a `str` is a distinct object here, so the reference
   header's `PyLongObject` and `PyUnicodeObject` are the ordinary mirror under
   their upstream names. */
typedef PyObject PyLongObject;
typedef PyObject PyUnicodeObject;
typedef void (*PyCapsule_Destructor)(PyObject *);
/* The buffer a `str` is written into piece by piece.  Opaque: only this
   runtime ever looks inside one. */
typedef struct PyUnicodeWriter PyUnicodeWriter;
/* A code object.  Opaque: nothing outside this runtime reads a field of one,
   and an extension only ever holds a pointer. */
typedef struct PyCodeObject PyCodeObject;
/* A frame.  Named here rather than in `frameobject.h` because the generated
   declarations come first; the fields are declared there. */
typedef struct _frame PyFrameObject;

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_PYTYPEDEFS_H */
