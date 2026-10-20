/* `str`.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_UNICODEOBJECT_H
#define PYRE_UNICODEOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* str. */

/* A `str` here is a mirror rather than a compact object with a readable
   `length` field, so the fast spelling is the call. */
#define PyUnicode_GET_LENGTH(op) PyUnicode_GetLength((PyObject *)(op))

/* The canonical representation, PEP 393.
 *
 * A mirror carries no code points either, so `PyUnicode_KIND` and
 * `PyUnicode_DATA` are calls as well; the block they answer with belongs to the
 * mirror and lives exactly as long as it does. What reads or writes that block
 * -- `PyUnicode_READ`, `PyUnicode_WRITE` and the typed `*_DATA` casts -- is
 * arithmetic on the caller's own pointer, so it is the reference text. */

#define PyUnicode_1BYTE_KIND 1
#define PyUnicode_2BYTE_KIND 2
#define PyUnicode_4BYTE_KIND 4

#define PyUnicode_KIND(op) PyUnicode_KIND((PyObject *)(op))
#define PyUnicode_DATA(op) PyUnicode_DATA((PyObject *)(op))
#define PyUnicode_IS_ASCII(op) PyUnicode_IS_ASCII((PyObject *)(op))
#define PyUnicode_MAX_CHAR_VALUE(op) PyUnicode_MAX_CHAR_VALUE((PyObject *)(op))

#define PyUnicode_1BYTE_DATA(op) ((Py_UCS1 *)PyUnicode_DATA(op))
#define PyUnicode_2BYTE_DATA(op) ((Py_UCS2 *)PyUnicode_DATA(op))
#define PyUnicode_4BYTE_DATA(op) ((Py_UCS4 *)PyUnicode_DATA(op))

static inline void _PyPyre_UnicodeWrite(int kind, void *data,
                                        Py_ssize_t index, Py_UCS4 value)
{
    if (kind == PyUnicode_1BYTE_KIND) {
        ((Py_UCS1 *)data)[index] = (Py_UCS1)value;
    } else if (kind == PyUnicode_2BYTE_KIND) {
        ((Py_UCS2 *)data)[index] = (Py_UCS2)value;
    } else {
        ((Py_UCS4 *)data)[index] = value;
    }
}
#define PyUnicode_WRITE(kind, data, index, value) \
    _PyPyre_UnicodeWrite((int)(kind), (void *)(data), (index), (Py_UCS4)(value))

static inline Py_UCS4 _PyPyre_UnicodeRead(int kind, const void *data,
                                          Py_ssize_t index)
{
    if (kind == PyUnicode_1BYTE_KIND) {
        return ((const Py_UCS1 *)data)[index];
    }
    if (kind == PyUnicode_2BYTE_KIND) {
        return ((const Py_UCS2 *)data)[index];
    }
    return ((const Py_UCS4 *)data)[index];
}
#define PyUnicode_READ(kind, data, index) \
    _PyPyre_UnicodeRead((int)(kind), (const void *)(data), (index))

#define PyUnicode_READ_CHAR(op, index) \
    PyUnicode_READ(PyUnicode_KIND(op), PyUnicode_DATA(op), (index))

/* Every string is ready, so the backward-compatible check is the constant its
   reference declaration is. */
#define PyUnicode_READY(op) ((void)(op), 0)

/* `PyUnicodeWriter_Format` is written here rather than exported: it is
   variadic, and the format engine it needs already sits in `pyre_format.h`.
   The formatted string is written into the writer and then given up. */
static inline int PyUnicodeWriter_Format(PyUnicodeWriter *writer,
                                         const char *format, ...)
{
    va_list vargs;
    va_start(vargs, format);
    PyObject *formatted = PyUnicode_FromFormatV(format, vargs);
    va_end(vargs);
    if (formatted == NULL) {
        return -1;
    }
    int written = PyUnicodeWriter_WriteStr(writer, formatted);
    Py_DECREF(formatted);
    return written;
}

/* The encoding the filesystem paths are decoded with.  It is always `utf-8`
   here, and it is a data symbol rather than a call because an extension reads
   it as one -- cffi compares it against a literal. */
PyAPI_DATA(const char *) Py_FileSystemDefaultEncoding;

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_UNICODEOBJECT_H */
