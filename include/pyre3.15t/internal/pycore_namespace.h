/* `_PyNamespace_New`, which builds a `types.SimpleNamespace` from a mapping.
 *
 * CPython keeps this out of the installed headers, so an extension reaching
 * it is one built inside the interpreter's own tree.  pyre ships it under
 * `internal/` for the same reason and on the same terms: it is not ABI an
 * ordinary extension may depend on.
 */
#ifndef PYRE_PYCORE_NAMESPACE_H
#define PYRE_PYCORE_NAMESPACE_H

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#ifdef __cplusplus
extern "C" {
#endif

PyAPI_FUNC(PyObject *) _PyNamespace_New(PyObject *kwds);

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_PYCORE_NAMESPACE_H */
