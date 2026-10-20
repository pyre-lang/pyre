/* Tracebacks.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_TRACEBACK_H
#define PYRE_TRACEBACK_H

#ifdef __cplusplus
extern "C" {
#endif
#define PyTraceBack_Check(v) Py_IS_TYPE((v), &PyTraceBack_Type)

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_TRACEBACK_H */
