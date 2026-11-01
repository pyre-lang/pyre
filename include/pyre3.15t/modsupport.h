/* The variadic entry points: argument parsing and value building.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The non-variadic entry points these are built out of are
 * declared together in `pyre_decl.h`, which is generated.
 *
 * `PyArg_ParseTuple` and `Py_BuildValue` are C functions with C bodies, and
 * pyre compiles them into the interpreter and exports them from it, which is
 * where `pypy/module/cpyext/src/getargs.c` and `.../modsupport.c` put the
 * same ones. The bodies live beside their peers in
 * `pyre/pyre-interpreter/src/cpyext/src/`; only the declarations are here.
 */
#ifndef PYRE_MODSUPPORT_H
#define PYRE_MODSUPPORT_H

#ifdef __cplusplus
extern "C" {
#endif

PyAPI_FUNC(int) PyArg_Parse(PyObject *, const char *, ...);
PyAPI_FUNC(int) PyArg_ParseTuple(PyObject *, const char *, ...);
PyAPI_FUNC(int) PyArg_ParseTupleAndKeywords(PyObject *, PyObject *, const char *,
                                            PY_CXX_CONST char *const *, ...);
PyAPI_FUNC(int) PyArg_UnpackTuple(PyObject *, const char *, Py_ssize_t, Py_ssize_t, ...);
PyAPI_FUNC(int) _PyArg_CheckPositional(const char *, Py_ssize_t, Py_ssize_t, Py_ssize_t);

/* A flag that runs one piece of initialization once, whichever thread arrives
   first; `std::call_once` is the C++11 spelling of it. */
typedef struct {
    uint8_t v;
} _PyOnceFlag;

/* The keyword table Argument Clinic emits beside a function, together with
   what the first call through it works out: which parameters are
   positional-only, and the tuple of interned names the rest are looked up by.
   Clinic names the fields it fills, so a parser it wrote carries `keywords`,
   `fname` and sometimes a prebuilt `kwtuple`, and leaves the rest zeroed. */
typedef struct _PyArg_Parser {
    const char *format;
    const char * const *keywords;
    const char *fname;
    const char *custom_msg;
    _PyOnceFlag once;       /* atomic one-time initialization flag */
    int is_kwtuple_owned;   /* does this parser own the kwtuple object? */
    int pos;                /* number of positional-only arguments */
    int min;                /* minimal number of arguments */
    int max;                /* maximal number of positional arguments */
    PyObject *kwtuple;      /* tuple of keyword parameter names */
    struct _PyArg_Parser *next;
} _PyArg_Parser;

PyAPI_FUNC(int) _PyArg_ParseTupleAndKeywordsFast(PyObject *, PyObject *,
                                                 struct _PyArg_Parser *, ...);
PyAPI_FUNC(PyObject *) Py_BuildValue(const char *, ...);
PyAPI_FUNC(PyObject *) Py_VaBuildValue(const char *, va_list);

/* ABI info & checking (new in 3.15) */
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= _Py_PACK_VERSION(3, 15)
typedef struct PyABIInfo {
    uint8_t abiinfo_major_version;
    uint8_t abiinfo_minor_version;
    uint16_t flags;
    uint32_t build_version;
    uint32_t abi_version;
} PyABIInfo;
#define PyABIInfo_STABLE        0x0001
#define PyABIInfo_GIL           0x0002
#define PyABIInfo_FREETHREADED  0x0004
#define PyABIInfo_INTERNAL      0x0008

#define PyABIInfo_FREETHREADING_AGNOSTIC (PyABIInfo_GIL|PyABIInfo_FREETHREADED)

PyAPI_FUNC(int) PyABIInfo_Check(PyABIInfo *info, const char *module_name);

// Define the defaults
#ifdef Py_LIMITED_API
    #define _PyABIInfo_DEFAULT_FLAG_STABLE PyABIInfo_STABLE
    #if Py_LIMITED_API == 3
        #define PyABIInfo_DEFAULT_ABI_VERSION _Py_PACK_VERSION(3, 2)
    #else
        #define PyABIInfo_DEFAULT_ABI_VERSION Py_LIMITED_API
    #endif
#else
    #define _PyABIInfo_DEFAULT_FLAG_STABLE 0
    #define PyABIInfo_DEFAULT_ABI_VERSION PY_VERSION_HEX
#endif
#if defined(Py_LIMITED_API) && defined(_Py_OPAQUE_PYOBJECT)
    #define _PyABIInfo_DEFAULT_FLAG_FT PyABIInfo_FREETHREADING_AGNOSTIC
#elif defined(Py_GIL_DISABLED)
    #define _PyABIInfo_DEFAULT_FLAG_FT PyABIInfo_FREETHREADED
#else
    #define _PyABIInfo_DEFAULT_FLAG_FT PyABIInfo_GIL
#endif
#if defined(Py_BUILD_CORE)
    #define _PyABIInfo_DEFAULT_FLAG_INTERNAL PyABIInfo_INTERNAL
#else
    #define _PyABIInfo_DEFAULT_FLAG_INTERNAL 0
#endif

#define PyABIInfo_DEFAULT_FLAGS (                           \
        _PyABIInfo_DEFAULT_FLAG_STABLE                      \
        | _PyABIInfo_DEFAULT_FLAG_FT                        \
        | _PyABIInfo_DEFAULT_FLAG_INTERNAL                  \
    )                                                       \
    /////////////////////////////////////////////////////////

#define _PyABIInfo_DEFAULT {                                \
        1, 0,                                               \
        PyABIInfo_DEFAULT_FLAGS,                            \
        PY_VERSION_HEX,                                     \
        PyABIInfo_DEFAULT_ABI_VERSION }                     \
    /////////////////////////////////////////////////////////

#define PyABIInfo_VAR(NAME) \
    static PyABIInfo NAME = _PyABIInfo_DEFAULT

#undef _PyABIInfo_DEFAULT_STABLE
#undef _PyABIInfo_DEFAULT_FT
#undef _PyABIInfo_DEFAULT_INTERNAL

#ifdef Py_BUILD_CORE
// For internal use in stdlib. Needs C99 compound literals.
#define _Py_ABI_SLOT {Py_mod_abi, (void*) &(PyABIInfo) _PyABIInfo_DEFAULT}
#endif

#endif /* ABI info (new in 3.15) */

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_MODSUPPORT_H */
