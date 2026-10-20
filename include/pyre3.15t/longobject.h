/* `int` and `bool`.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_LONGOBJECT_H
#define PYRE_LONGOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* int / bool. */

/* The fixed-width conversions report failure through the `int` return and
   write the value out, so a value that happens to be -1 is not an error. */

/* Copying an int in and out of a C variable's bytes.  `Py_ASNATIVEBYTES_*`
   select the byte order and how the sign is read; -1 is the default, which is
   native order, a signed destination and no `__index__` call. */
#define Py_ASNATIVEBYTES_DEFAULTS -1
#define Py_ASNATIVEBYTES_BIG_ENDIAN 0
#define Py_ASNATIVEBYTES_LITTLE_ENDIAN 1
#define Py_ASNATIVEBYTES_NATIVE_ENDIAN 3
#define Py_ASNATIVEBYTES_UNSIGNED_BUFFER 4
#define Py_ASNATIVEBYTES_REJECT_NEGATIVE 8
#define Py_ASNATIVEBYTES_ALLOW_INDEX 16

/* An `int` here is an ordinary mirror with no digit array of its own, so the
   name exists only so these two declarations read as they do upstream. */

/* The unchecked spelling, which a caller uses where it already knows the
   object is an `int` small enough to fit.  There is no field to read, so it
   is the checked call. */
#define PyLong_AS_LONG(op) PyLong_AsLong((PyObject *)(op))

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_LONGOBJECT_H */
