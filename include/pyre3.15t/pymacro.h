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

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_PYMACRO_H */
