/* `set` and `frozenset`.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_SETOBJECT_H
#define PYRE_SETOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* The set protocol (`cpyext/setobject.py`).  The `Check` spellings are
   functions here, as pyre's other type checks are. */

typedef struct {
    PyObject_HEAD
    PyObject *_tmplist; /* a private place to put values during _PySet_Next */
} PySetObject;

#define PySet_GET_SIZE(ob) PySet_Size((PyObject *)(ob))

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_SETOBJECT_H */
