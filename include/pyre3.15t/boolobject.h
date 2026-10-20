/* `bool`.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_BOOLOBJECT_H
#define PYRE_BOOLOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* bool. */

/* `bool` has no subclass to tell apart, so the exact test is the only one.
   `Py_True` and `Py_False` are named beside `Py_None` in `object.h`. */
#define PyBool_Check(op) (Py_TYPE(op) == &PyBool_Type)

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_BOOLOBJECT_H */
