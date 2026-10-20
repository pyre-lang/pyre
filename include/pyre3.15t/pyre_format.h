/* `PyUnicode_FromFormat` and the two entry points built on the same `%`
 * engine.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. `PyErr_Format` is here rather than beside the rest of the
 * exception interface because it is this engine with the result handed to
 * `PyErr_SetObject`, which is the shape `_PyErr_FormatV` gives it too.
 *
 * The bodies are C, compiled into the interpreter and exported from it:
 * `pyre/pyre-interpreter/src/cpyext/src/unicodeobject.c` and `.../pyerrors.c`,
 * beside the peers `pypy/module/cpyext/src/` holds.
 */
#ifndef PYRE_PYRE_FORMAT_H
#define PYRE_PYRE_FORMAT_H

#ifdef __cplusplus
extern "C" {
#endif

PyAPI_FUNC(PyObject *) PyUnicode_FromFormat(const char *, ...);
PyAPI_FUNC(PyObject *) PyUnicode_FromFormatV(const char *, va_list);
PyAPI_FUNC(PyObject *) PyErr_Format(PyObject *, const char *, ...);
PyAPI_FUNC(void) PyErr_FormatUnraisable(const char *, ...);

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_PYRE_FORMAT_H */
