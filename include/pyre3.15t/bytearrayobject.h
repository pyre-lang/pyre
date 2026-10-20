/* `bytearray`.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_BYTEARRAYOBJECT_H
#define PYRE_BYTEARRAYOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* bytearray. */

/* The storage a `bytearray` mirror hands out is a cached copy rather than a
   field, so both unchecked spellings are the checked call. */
#define PyByteArray_AS_STRING(op) PyByteArray_AsString((PyObject *)(op))
#define PyByteArray_GET_SIZE(op) PyByteArray_Size((PyObject *)(op))

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_BYTEARRAYOBJECT_H */
