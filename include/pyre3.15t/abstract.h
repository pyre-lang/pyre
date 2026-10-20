/* The protocols that work on any object, and the call interface.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_ABSTRACT_H
#define PYRE_ABSTRACT_H

#ifdef __cplusplus
extern "C" {
#endif
/* The vectorcall protocol.  `args[0 .. nargs)` are positional and the rest are
   the values for the names in `kwnames`, in order.  The high bit of `nargsf`
   says the caller left a spare slot before args[0]; pyre only reads the array,
   so the bit just has to be stripped from the count. */
#define PY_VECTORCALL_ARGUMENTS_OFFSET ((size_t)1 << (8 * sizeof(size_t) - 1))

static inline Py_ssize_t PyVectorcall_NARGS(size_t n)
{
    return (Py_ssize_t)(n & ~PY_VECTORCALL_ARGUMENTS_OFFSET);
}

static inline PyObject *PyObject_CallMethodNoArgs(PyObject *self, PyObject *name)
{
    PyObject *args[1];
    args[0] = self;
    return PyObject_VectorcallMethod(name, args, 1, NULL);
}

static inline PyObject *PyObject_CallMethodOneArg(PyObject *self, PyObject *name,
                                                  PyObject *arg)
{
    PyObject *args[2];
    args[0] = self;
    args[1] = arg;
    return PyObject_VectorcallMethod(name, args, 2, NULL);
}

/* The number protocol. */

/* The sequence protocol. */

/* Functions rather than the macros the reference header spells: a mirror has
   no item array of its own, so the length and the items come from the
   interpreter object behind it. */

/* The mapping protocol. */

/* The iterator protocol. */

PyAPI_FUNC(PyObject *) PyObject_CallFunctionObjArgs(PyObject *, ...);
PyAPI_FUNC(PyObject *) PyObject_CallMethodObjArgs(PyObject *, PyObject *, ...);
PyAPI_FUNC(PyObject *) PyObject_CallFunction(PyObject *, const char *, ...);
PyAPI_FUNC(PyObject *) PyObject_CallMethod(PyObject *, const char *, const char *, ...);

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_ABSTRACT_H */
