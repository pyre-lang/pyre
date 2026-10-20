/* The audit hooks, which `sys.addaudithook` installs and any code -- C or
 * Python -- raises events against.
 *
 * `Include/audit.h` is where CPython declares the pair.  Only the tuple form
 * is an export here; the variadic one is the header's, for the reason
 * `modsupport.h` gives.
 */
#ifndef PYRE_AUDIT_H
#define PYRE_AUDIT_H

#ifdef __cplusplus
extern "C" {
#endif

/* `sys_audit_tstate` — the arguments are whatever the format builds, wrapped
   in a one-element tuple when the format builds a single value rather than a
   tuple.  An empty or NULL format raises the event with no arguments. */
static inline int PySys_Audit(const char *event, const char *argFormat, ...)
{
    PyObject *args;
    if (argFormat != NULL && argFormat[0] != '\0') {
        va_list va;
        va_start(va, argFormat);
        PyObject *built = Py_VaBuildValue(argFormat, va);
        va_end(va);
        if (built == NULL) {
            return -1;
        }
        if (PyTuple_Check(built)) {
            args = built;
        } else {
            args = PyTuple_Pack(1, built);
            Py_DECREF(built);
            if (args == NULL) {
                return -1;
            }
        }
    } else {
        args = PyTuple_New(0);
        if (args == NULL) {
            return -1;
        }
    }
    int answer = PySys_AuditTuple(event, args);
    Py_DECREF(args);
    return answer;
}

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_AUDIT_H */
