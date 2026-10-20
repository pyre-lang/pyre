/* The argument-parsing entry points Argument Clinic writes calls to.
 *
 * CPython keeps these out of the installed headers, so an extension reaching
 * them is one built inside the interpreter's own tree.  pyre ships them under
 * `internal/` on the same terms: not ABI an ordinary extension may depend on.
 *
 * Each declaration is paired with the fast-path macro its callers expand, so
 * the common shapes -- no keywords at all, a call already in range -- never
 * reach the function.
 */
#ifndef PYRE_PYCORE_MODSUPPORT_H
#define PYRE_PYCORE_MODSUPPORT_H

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Declared with the other variadic entry points in `modsupport.h`, which
   `Python.h` includes; the macro is what this header adds. */
#define _Py_ANY_VARARGS(n) ((n) == PY_SSIZE_T_MAX)
#define _PyArg_CheckPositional(funcname, nargs, min, max) \
    ((!_Py_ANY_VARARGS(max) && (min) <= (nargs) && (nargs) <= (max)) \
     || _PyArg_CheckPositional((funcname), (nargs), (min), (max)))

/* Bind a vectorcall's arguments to `parser`'s parameters, answering the array
   the caller reads them out of: `args` itself when nothing had to be moved,
   and otherwise `buf`, filled slot by slot with a missing optional left NULL.
   Answers NULL with an exception set. */
PyAPI_FUNC(PyObject * const *) _PyArg_UnpackKeywords(
    PyObject *const *args,
    Py_ssize_t nargs,
    PyObject *kwargs,
    PyObject *kwnames,
    struct _PyArg_Parser *parser,
    int minpos,
    int maxpos,
    int minkw,
    int varpos,
    PyObject **buf);
#define _PyArg_UnpackKeywords(args, nargs, kwargs, kwnames, parser, minpos, maxpos, minkw, varpos, buf) \
    (((minkw) == 0 && (kwargs) == NULL && (kwnames) == NULL && \
      (minpos) <= (nargs) && ((varpos) || (nargs) <= (maxpos)) && (args) != NULL) ? \
      (args) : \
     _PyArg_UnpackKeywords((args), (nargs), (kwargs), (kwnames), (parser), \
                           (minpos), (maxpos), (minkw), (varpos), (buf)))

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_PYCORE_MODSUPPORT_H */
