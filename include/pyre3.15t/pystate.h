/* Thread state and the GIL.
 *
 * A thread holds the GIL for as long as it runs pyre code, so an extension
 * needs the same two spellings it uses on CPython: one to give the GIL up
 * around a blocking call, one to take it before calling in from a thread of
 * its own.
 *
 * An extension receives a `PyThreadState` from `PyEval_SaveThread` and hands it
 * back to `PyEval_RestoreThread`. `interp` is the one field it reads directly,
 * which is how it asks which interpreter the thread runs in without a call;
 * everything else the struct stands for is reached through an entry point.
 */
#ifndef PYRE_PYSTATE_H
#define PYRE_PYSTATE_H

#ifdef __cplusplus
extern "C" {
#endif

/* The interpreter a thread runs in.  Opaque: an extension compares one handle
   against another and passes it on, and reaches everything behind it through
   an entry point. */
typedef struct _is PyInterpreterState;

/* The twin of `CPyThreadState` in `cpyext/pystate.rs`, which pins each offset
   below with a static assertion.

   `interp` and `dict` are the fields an extension reads directly; the rest of
   what the state stands for is reached through an entry point.  `_status` is
   published because an extension writes to it -- cffi clears `bound_gilstate`
   on a state it is about to delete -- and it has to land in storage of its own
   rather than past the end of the block.  Nothing here reads it. */
typedef struct _ts {
    PyInterpreterState *interp;
    PyObject *dict;
    uint64_t id;
    struct {
        unsigned int initialized:1;
        unsigned int bound:1;
        unsigned int unbound:1;
        unsigned int bound_gilstate:1;
        unsigned int active:1;
        unsigned int finalizing:1;
        unsigned int cleared:1;
        unsigned int finalized:1;
        unsigned int :24;
    } _status;
} PyThreadState;

/* The spelling an extension written before 3.9 uses for the same call. */
#define PyThreadState_GET() PyThreadState_Get()

typedef enum { PyGILState_LOCKED, PyGILState_UNLOCKED } PyGILState_STATE;

/* Declared here rather than in the generated `pyre_decl.h` because they are
   not implemented in the `cpyext` layer: a build without it exports these two
   as well, for cffi's embedding header. */
PyAPI_FUNC(PyGILState_STATE) PyGILState_Ensure(void);
PyAPI_FUNC(void) PyGILState_Release(PyGILState_STATE);

/* The block an extension wraps a blocking call in. `_save` is the extension's
   own local, so the two halves have to appear in one scope, which is what the
   braces enforce. */
#define Py_BEGIN_ALLOW_THREADS                                                 \
    {                                                                          \
        PyThreadState *_save;                                                  \
        _save = PyEval_SaveThread();
#define Py_BLOCK_THREADS PyEval_RestoreThread(_save);
#define Py_UNBLOCK_THREADS _save = PyEval_SaveThread();
#define Py_END_ALLOW_THREADS                                                   \
    PyEval_RestoreThread(_save);                                               \
    }

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_PYSTATE_H */
