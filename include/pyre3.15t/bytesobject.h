/* `bytes`.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_BYTESOBJECT_H
#define PYRE_BYTESOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* bytes. */

/* The storage a `bytes` mirror hands out is a cached copy, not a field, so
   the size answer is the same call the checked spelling makes. */
#define PyBytes_GET_SIZE(op) PyBytes_Size((PyObject *)(op))

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_BYTESOBJECT_H */
