/* The raw allocator.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_PYMEM_H
#define PYRE_PYMEM_H

#ifdef __cplusplus
extern "C" {
#endif
/* The raw allocators.  The `PyMem_*` and `PyMem_Raw*` halves are the same
   functions, as they are upstream (`cpyext/src/pymem.c:57`).  Memory from
   these never holds an interpreter object. */

#define PyMem_New(type, n) \
    ((type *)((size_t)(n) > (size_t)PY_SSIZE_T_MAX / sizeof(type) \
              ? NULL : PyMem_Malloc((n) * sizeof(type))))
#define PyMem_Resize(p, type, n) \
    ((p) = ((size_t)(n) > (size_t)PY_SSIZE_T_MAX / sizeof(type) \
            ? NULL : (type *)PyMem_Realloc((p), (n) * sizeof(type))))
#define PyMem_Del PyMem_Free
#define PyMem_DEL PyMem_Free
#define PyMem_MALLOC PyMem_Malloc
#define PyMem_REALLOC PyMem_Realloc
#define PyMem_FREE PyMem_Free

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_PYMEM_H */
