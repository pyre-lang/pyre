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

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_MODSUPPORT_H */
