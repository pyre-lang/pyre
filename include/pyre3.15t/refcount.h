/* Reference counting.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 *
 * Included after `pyre_decl.h` rather than beside `object.h`, because the
 * inline functions below call the `Py_IncRef` that file declares.
 */
#ifndef PYRE_REFCOUNT_H
#define PYRE_REFCOUNT_H

#ifdef __cplusplus
extern "C" {
#endif

#define Py_INCREF(ob) Py_IncRef((PyObject *)(ob))
#define Py_DECREF(ob) Py_DecRef((PyObject *)(ob))
#define Py_XINCREF(ob) do { if ((ob) != NULL) Py_INCREF(ob); } while (0)
#define Py_XDECREF(ob) do { if ((ob) != NULL) Py_DECREF(ob); } while (0)
#define Py_CLEAR(ob) do { PyObject *_tmp = (PyObject *)(ob); (ob) = NULL; Py_XDECREF(_tmp); } while (0)
/* The count `Py_REFCNT` reports is the whole field, including the share the
   interpreter link contributes, so a value written back here has to have come
   from `Py_REFCNT` rather than be a number counted up from zero. */
#define Py_SET_REFCNT(ob, refcnt) _Py_SetRefcnt((PyObject *)(ob), (refcnt))

#define Py_SETREF(ob, value) do { PyObject *_old = (PyObject *)(ob); (ob) = (value); Py_XDECREF(_old); } while (0)

/* A reference taken and handed back in one expression, so that a borrowed
   object can be returned without naming it twice. */
static inline PyObject *_Py_NewRef(PyObject *ob)
{
    Py_INCREF(ob);
    return ob;
}

static inline PyObject *_Py_XNewRef(PyObject *ob)
{
    Py_XINCREF(ob);
    return ob;
}

#define Py_NewRef(ob) _Py_NewRef((PyObject *)(ob))
#define Py_XNewRef(ob) _Py_XNewRef((PyObject *)(ob))

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_REFCOUNT_H */
