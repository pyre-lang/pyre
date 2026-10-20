/* What `Py_CompileString` and its callers name a compilation unit's shape by.
 *
 * An extension includes this by name rather than through `Python.h`, which is
 * where the reference header set puts it too.
 */
#ifndef PYRE_COMPILE_H
#define PYRE_COMPILE_H

#ifdef __cplusplus
extern "C" {
#endif
#define Py_single_input 256
#define Py_file_input 257
#define Py_eval_input 258
#define Py_func_type_input 345

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_COMPILE_H */
