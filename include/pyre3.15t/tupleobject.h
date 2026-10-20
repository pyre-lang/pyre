/* `tuple`.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_TUPLEOBJECT_H
#define PYRE_TUPLEOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* tuple. */

/* A `tuple` mirror hands out an array of its items, which is the `ob_item`
   field `tupleobject.py` gives its own mirror: an extension reads a slot
   without a call, and takes the address of one -- cffi writes
   `(CTypeDescrObject **)&PyTuple_GET_ITEM(fargs, 0)`.  Dereferencing what
   `_PyTuple_ITEMS` answers with is that lvalue.

   The struct is spelled out because an extension names the type, and its
   `ob_item` is declared the way the reference header declares it: one element
   standing for however many `ob_size` says.  Nothing here reads it through the
   struct -- the array lives beside the block rather than at the end of it, so
   `_PyTuple_ITEMS` is what finds it. */
typedef struct {
    PyObject_VAR_HEAD
    PyObject *ob_item[1];
} PyTupleObject;

#define PyTuple_GET_SIZE(ob) Py_SIZE(ob)
#define PyTuple_GET_ITEM(ob, i) (_PyTuple_ITEMS((PyObject *)(ob))[i])
#define PyTuple_SET_ITEM(ob, i, v) _PyTuple_SET_ITEM((PyObject *)(ob), (i), (v))

/* Variadic, so it is built here out of the non-variadic exports. */
static inline PyObject *PyTuple_Pack(Py_ssize_t n, ...)
{
    PyObject *result = PyTuple_New(n);
    va_list vargs;
    Py_ssize_t i;

    if (result == NULL) {
        return NULL;
    }
    va_start(vargs, n);
    for (i = 0; i < n; i++) {
        PyObject *item = va_arg(vargs, PyObject *);
        Py_XINCREF(item);
        if (PyTuple_SetItem(result, i, item) < 0) {
            va_end(vargs);
            Py_DECREF(result);
            return NULL;
        }
    }
    va_end(vargs);
    return result;
}

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_TUPLEOBJECT_H */
