/* The compiler-facing spellings: linkage, and the integer types.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_PYPORT_H
#define PYRE_PYPORT_H

#ifdef __cplusplus
extern "C" {
#endif
#if defined(_WIN32)
#  define PyAPI_FUNC(RTYPE) __declspec(dllimport) RTYPE
#  define PyAPI_DATA(RTYPE) extern __declspec(dllimport) RTYPE
#  define PyMODINIT_FUNC __declspec(dllexport) PyObject *
#elif defined(__cplusplus)
#  define PyAPI_FUNC(RTYPE) __attribute__((visibility("default"))) RTYPE
#  define PyAPI_DATA(RTYPE) extern __attribute__((visibility("default"))) RTYPE
#  define PyMODINIT_FUNC extern "C" __attribute__((visibility("default"))) PyObject *
#else
#  define PyAPI_FUNC(RTYPE) __attribute__((visibility("default"))) RTYPE
#  define PyAPI_DATA(RTYPE) extern __attribute__((visibility("default"))) RTYPE
#  define PyMODINIT_FUNC __attribute__((visibility("default"))) PyObject *
#endif
/* C++ has no implicit conversion from `const char *[]` to `char **`, so the
 * keyword list a C++ extension can build is only accepted when the parameter
 * carries this qualifier.  C keeps it unqualified, where an extension may pass
 * a non-const array. */
#ifndef PY_CXX_CONST
#  ifdef __cplusplus
#    define PY_CXX_CONST const
#  else
#    define PY_CXX_CONST
#  endif
#endif

/* A cast written so the one spelling compiles as C and as C++, where the
   implicit conversions C allows are errors. */
#define _Py_CAST(type, expr) ((type)(expr))

/* A function pointer cast to `T`, through `void (*)(void)` so that no warning
   is emitted for the change of signature.  It silences the compiler only: the
   function is still called through whichever type the caller uses, and calling
   it through the wrong one is undefined at run time. */
#define _Py_FUNC_CAST(T, func) _Py_CAST(T, _Py_CAST(void (*)(void), (func)))

typedef uintptr_t Py_uintptr_t;
typedef intptr_t Py_intptr_t;
typedef Py_intptr_t Py_ssize_t;
#define PY_SSIZE_T_MAX ((Py_ssize_t)(((size_t)-1) >> 1))
#define PY_SSIZE_T_MIN (-PY_SSIZE_T_MAX - 1)
typedef Py_ssize_t Py_hash_t;
typedef size_t Py_uhash_t;
/* The widest integer the C compiler offers, under the name an extension
   spells it by, and the limits `<limits.h>` gives it. */
#define PY_LONG_LONG long long
#define PY_LLONG_MIN LLONG_MIN
#define PY_LLONG_MAX LLONG_MAX
#define PY_ULLONG_MAX ULLONG_MAX

/* The fixed-width integer names an extension may spell a field or a cast
   with, alongside the `<stdint.h>` names they stand for. */
#define PY_UINT32_T uint32_t
#define PY_UINT64_T uint64_t
#define PY_INT32_T int32_t
#define PY_INT64_T int64_t

typedef uint32_t Py_UCS4;
typedef uint16_t Py_UCS2;
typedef uint8_t Py_UCS1;

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_PYPORT_H */
