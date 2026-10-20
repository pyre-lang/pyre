/* The variadic warning entry points.

   A variadic function cannot be exported from Rust, so each one is written
   here as `PyUnicode_FromFormatV` over the exported core it shares with the
   non-variadic spellings: `_PyPyre_WarnUnicode` for the ones that let the
   machinery find the location, `_PyPyre_WarnExplicitMessage` for the one that
   states it. */

#ifndef PYRE_WARNINGS_H
#define PYRE_WARNINGS_H

#ifdef __cplusplus
extern "C" {
#endif

static inline int PyErr_WarnFormat(PyObject *category, Py_ssize_t stack_level,
                                   const char *format, ...)
{
    va_list va;
    PyObject *message;
    int result;

    va_start(va, format);
    message = PyUnicode_FromFormatV(format, va);
    va_end(va);
    if (message == NULL) {
        return -1;
    }
    result = _PyPyre_WarnUnicode(NULL, category, message, stack_level);
    Py_DECREF(message);
    return result;
}

/* `ResourceWarning` with the object the resource belongs to, which reaches
   `sys.unraisablehook` as the `source` when the warning is turned into an
   error late in shutdown. */
static inline int PyErr_ResourceWarning(PyObject *source, Py_ssize_t stack_level,
                                        const char *format, ...)
{
    va_list va;
    PyObject *message;
    int result;

    va_start(va, format);
    message = PyUnicode_FromFormatV(format, va);
    va_end(va);
    if (message == NULL) {
        return -1;
    }
    result = _PyPyre_WarnUnicode(source, PyExc_ResourceWarning, message, stack_level);
    Py_DECREF(message);
    return result;
}

static inline int PyErr_WarnExplicitFormat(PyObject *category,
                                           const char *filename, int lineno,
                                           const char *module, PyObject *registry,
                                           const char *format, ...)
{
    va_list va;
    PyObject *message;
    int result;

    va_start(va, format);
    message = PyUnicode_FromFormatV(format, va);
    va_end(va);
    if (message == NULL) {
        return -1;
    }
    result = _PyPyre_WarnExplicitMessage(category, message, filename, lineno, module,
                                         registry);
    Py_DECREF(message);
    return result;
}

/* Kept for the extensions still calling it; `PyErr_WarnEx` is the spelling
   with a stack level. */
#define PyErr_Warn(category, message) PyErr_WarnEx((category), (message), 1)

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_WARNINGS_H */
