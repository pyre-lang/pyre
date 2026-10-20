/* Instance method objects.
 *
 * One of the headers `Python.h` includes. The exported entry points are
 * declared together in `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_CLASSOBJECT_H
#define PYRE_CLASSOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif

PyAPI_DATA(PyTypeObject) PyInstanceMethod_Type;

/* `instancemethod` is not a base class, and `PyInstanceMethod_Function`
   answers for this type alone. */
#define PyInstanceMethod_Check(op) Py_IS_TYPE((op), &PyInstanceMethod_Type)

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_CLASSOBJECT_H */
