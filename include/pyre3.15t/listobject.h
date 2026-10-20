/* `list`.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_LISTOBJECT_H
#define PYRE_LISTOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* list. */

/* Each is exported as well as spelled here, for the reason `Py_TYPE` is: a
   caller that declares the prototype itself rather than including this header
   still has to find a symbol.  The declaration comes before the macro that
   replaces it. */
PyAPI_FUNC(Py_ssize_t) PyList_GET_SIZE(PyObject *);
#define PyList_GET_SIZE(ob) PyList_Size((PyObject *)(ob))
PyAPI_FUNC(PyObject *) PyList_GET_ITEM(PyObject *, Py_ssize_t);
#define PyList_GET_ITEM(ob, i) PyList_GetItem((PyObject *)(ob), (i))
PyAPI_FUNC(void) PyList_SET_ITEM(PyObject *, Py_ssize_t, PyObject *);
#define PyList_SET_ITEM(ob, i, v) ((void)PyList_SetItem((PyObject *)(ob), (i), (v)))

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_LISTOBJECT_H */
