/* What the build answered about the platform and about itself.
 *
 * An extension includes only `Python.h`, which includes this first. It is
 * also included on its own: `cffi`'s generated source reads it before
 * `Python.h` to learn whether the build is a debug one.
 */
#ifndef Py_PYCONFIG_H
#define Py_PYCONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

/* The compiler is a C99 one or better, which is what every entry point below
   is declared against. */
#define HAVE_PROTOTYPES 1
#define HAVE_STDARG_PROTOTYPES 1
#define STDC_HEADERS 1
#define HAVE_LONG_LONG 1
#define HAVE_WCHAR_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_SYS_STAT_H 1

/* The length modifier a `%`-format naming a `Py_ssize_t` or a `long long`
   carries.  `pyre_format.h` is what reads them. */
#define PY_FORMAT_SIZE_T "z"
#define PY_FORMAT_LONG_LONG "ll"

/* A docstring an extension declares is kept and handed out. */
#define WITH_DOC_STRINGS

/* `Py_UNICODE` is `wchar_t`, which the entry points taking one are declared
   against. */
#define HAVE_USABLE_WCHAR_T 1
#ifndef _WIN32
#  define SIZEOF_WCHAR_T 4
#else
#  define SIZEOF_WCHAR_T 2
#endif

/* Whether a `va_list` can be copied by assignment.  Where it cannot, an
   extension that reads its arguments twice has to `va_copy` -- and this is the
   name that question is asked under. */
#ifndef _WIN32
#  define VA_LIST_IS_ARRAY
#endif

/* This used to say whether the build had threads.  It always does, and the
   name is kept because extensions still test it -- cffi refuses to compile
   without it. */
#ifndef WITH_THREAD
#  define WITH_THREAD
#endif

#ifdef __cplusplus
}
#endif

#endif /* !Py_PYCONFIG_H */
