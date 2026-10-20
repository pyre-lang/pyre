/* `dict`.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_DICTOBJECT_H
#define PYRE_DICTOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* dict.  `PyDict_Next` keeps the keys it walks on this side rather than
   in the mirror, so the field is only what makes the block a struct. */
typedef struct {
    PyObject_HEAD
    PyObject *_tmpkeys; /* a private place to put keys during PyDict_Next */
} PyDictObject;

/* A call rather than the field read upstream spells, so it can fail: a
   non-dict argument answers -1 with a `TypeError` set.  `cpyext`'s own
   `dictobject.h` states `PyObject_Length` for the same reason. */
#define PyDict_GET_SIZE(ob) PyDict_Size((PyObject *)(ob))

#define PyDoc_STRVAR(name, str) static const char name[] = str
#define PyDoc_STR(str) str

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_DICTOBJECT_H */
