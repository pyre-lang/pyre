/* Bound methods.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_FUNCOBJECT_H
#define PYRE_FUNCOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* Exact, because `method` is not a base a type may derive from and because
   `PyMethod_Function` / `PyMethod_Self` answer for this type alone: a
   predicate wider than the accessors' own guard would admit objects they then
   refuse with a `SystemError` the caller never looks for. */
#define PyMethod_Check(op) Py_IS_TYPE((op), &PyMethod_Type)

/* The reference header reads the two members straight out of the struct.  A
   mirror has no members to read, so each is the call that answers with the
   same borrowed reference. */
#define PyMethod_GET_FUNCTION(obj) PyMethod_Function((PyObject *)(obj))
#define PyMethod_GET_SELF(obj) PyMethod_Self((PyObject *)(obj))

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_FUNCOBJECT_H */
