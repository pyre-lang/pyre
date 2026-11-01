/* Macros with no object of their own.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_PYMACRO_H
#define PYRE_PYMACRO_H

#ifdef __cplusplus
extern "C" {
#endif
/* Minimum value between x and y */
#define Py_MIN(x, y) (((x) > (y)) ? (y) : (x))

/* Maximum value between x and y */
#define Py_MAX(x, y) (((x) > (y)) ? (x) : (y))

/* Names an argument a function does not read, so the compiler stops warning
   about it: `int func(int a, int Py_UNUSED(b))`. */
#if defined(__GNUC__) || defined(__clang__)
#  define Py_UNUSED(name) _unused_##name __attribute__((unused))
#else
#  define Py_UNUSED(name) _unused_##name
#endif

// _Py_ANONYMOUS: modifier for declaring an anonymous union.
// Usage: _Py_ANONYMOUS union { ... };
// Standards/compiler support:
// - C++ allows anonymous unions, but not structs
// - C11 and above allows anonymous unions and structs
// - MSVC has warning(disable: 4201) "nonstandard extension used : nameless
//   struct/union". This is specific enough that we disable it for all of
//   Python.h.
// - GCC & clang needs __extension__ before C11
// To allow unsupported platforms which need other spellings, we use a
// predefined value of _Py_ANONYMOUS if it exists.
#ifndef _Py_ANONYMOUS
#   if (defined(__GNUC__) || defined(__clang__)) \
          && !(defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
#       define _Py_ANONYMOUS __extension__
#   else
#       define _Py_ANONYMOUS
#   endif
#endif

#define _Py_PACK_VERSION(X, Y) _Py_PACK_FULL_VERSION(X, Y, 0, 0, 0)

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_PYMACRO_H */
