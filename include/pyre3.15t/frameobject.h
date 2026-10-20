/* Frames.
 *
 * One of the headers `Python.h` includes -- `PyFrame_Check` is reachable from
 * `Python.h` alone in both reference header sets -- and an extension may also
 * include it by name, which is the spelling most of them use.
 */
#ifndef PYRE_FRAMEOBJECT_H
#define PYRE_FRAMEOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* The storage a frame an extension built for itself carries.
 *
 * A frame this runtime is executing is not one of these: it lives in the
 * interpreter, and a mirror of it is filled from that side.  What the fields
 * are declared for is the other direction -- `PyFrame_New` hands back a frame
 * whose line number the caller then writes, which is a field assignment in
 * every extension that reports a traceback of its own.
 *
 * `pyre-interpreter/src/cpyext/frameobject.rs` holds this layout from the
 * other side; each offset is pinned in both places. */
struct _frame {
    PyObject_HEAD
    PyCodeObject *f_code;
    PyObject *f_globals;
    PyObject *f_locals;
    int f_lineno;
    struct _frame *f_back;
};

#define PyFrame_Check(op) Py_IS_TYPE((op), &PyFrame_Type)

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_FRAMEOBJECT_H */
