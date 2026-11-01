/* The version this ABI is.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_PATCHLEVEL_H
#define PYRE_PATCHLEVEL_H

#ifdef __cplusplus
extern "C" {
#endif
/* Values for PY_RELEASE_LEVEL */
#define PY_RELEASE_LEVEL_ALPHA 0xA
#define PY_RELEASE_LEVEL_BETA 0xB
#define PY_RELEASE_LEVEL_GAMMA 0xC /* For release candidates */
#define PY_RELEASE_LEVEL_FINAL 0xF /* Serial should be 0 here */

/* Version parsed out into numeric values */
#define PY_MAJOR_VERSION 3
#define PY_MINOR_VERSION 15
#define PY_MICRO_VERSION 0
#define PY_RELEASE_LEVEL PY_RELEASE_LEVEL_FINAL
#define PY_RELEASE_SERIAL 0

/* Version as a string */
#define PY_VERSION "3.15.0"

/* The same four digits packed as `PY_VERSION_HEX` packs them, read at run
   time: the macros above are what an extension was compiled against, and this
   is what it is running against. */
PyAPI_DATA(const unsigned long) Py_Version;

/* Pyre version as a string. Keep in sync with the `pyrex` package version,
   which is what `sys.pyre_version_info` reports; `cpyext_patchlevel.rs`
   compares the two. */
#define PYRE_VERSION "0.0.2"
#define PYRE_VERSION_NUM 0x00000200

/* What an extension asks when it wants to know whether the object layouts
   behind these headers are CPython's.  They are not: pyre's cpyext mirrors
   `pypy/module/cpyext`, so a `PyObject *` names a block that stands for an
   interpreter object rather than the object itself, and nothing may read a
   field of one that is not declared here.

   `PYPY_VERSION` is the name that question is already spelled under -- Cython
   branches on it, and so does much of the ecosystem -- and the value tracks
   PyPy's own so that a version gate written for PyPy lands on the same side
   for pyre.  Code that wants to know which of the two it has reads
   `PYRE_VERSION` above.

   A module cffi generated is the exception, and `_CFFI_` is how it says so:
   `_cffi_include.h` defines it on its first line, before it includes this
   header.  Such a module has two forms -- one built out of `_cffi_exports` and
   `PyInit_`, one exporting `_cffi_pypyinit_` and handing its type context to a
   built-in backend -- and it picks between them on `PYPY_VERSION`.  The backend
   here is cffi's own `_cffi_backend` extension, which is the first form's, so
   that is the form a cffi module compiles to. */
#ifndef _CFFI_
#define PYPY_VERSION "8.0.0-alpha0"
#define PYPY_VERSION_NUM 0x08000000
#endif

/* A reference cpyext holds is a regular one: it keeps the interpreter object
   alive for as long as the block does. */
#define PYPY_CPYEXT_GC 1
#define PyPy_Borrow(a, b) ((void)0)

#define _Py_PACK_FULL_VERSION(X, Y, Z, LEVEL, SERIAL) ( \
    (((X) & 0xff) << 24) |                              \
    (((Y) & 0xff) << 16) |                              \
    (((Z) & 0xff) << 8) |                               \
    (((LEVEL) & 0xf) << 4) |                            \
    (((SERIAL) & 0xf) << 0))

/* Version as a single 4-byte hex number, e.g. 0x030F00F0 == 3.15.0 final.
   Use this for numeric comparisons, e.g. #if PY_VERSION_HEX >= ...
   The release-level nibble is 0xF for a final release, so a value ending in
   0x00 would put every `#if PY_VERSION_HEX >= 0x030F00F0` extension on its
   pre-release branch. */
#define PY_VERSION_HEX _Py_PACK_FULL_VERSION( \
    PY_MAJOR_VERSION,                         \
    PY_MINOR_VERSION,                         \
    PY_MICRO_VERSION,                         \
    PY_RELEASE_LEVEL,                         \
    PY_RELEASE_SERIAL)

#define PYTHON_API_VERSION 1013
#define PYTHON_ABI_VERSION 3

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_PATCHLEVEL_H */
