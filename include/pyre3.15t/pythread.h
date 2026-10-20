/* The lock an extension allocates for itself, and the identity of the thread
   holding it. */

#ifndef Py_PYTHREAD_H
#define Py_PYTHREAD_H
#ifdef __cplusplus
extern "C" {
#endif

typedef void *PyThread_type_lock;

/* Failure is 0 and success 1, so a caller may test either the status or the
   plain `PyThread_acquire_lock` int. */
typedef enum PyLockStatus {
    PY_LOCK_FAILURE = 0,
    PY_LOCK_ACQUIRED = 1,
    PY_LOCK_INTR
} PyLockStatus;

#define WAIT_LOCK       1
#define NOWAIT_LOCK     0

#define PY_TIMEOUT_T long long

PyAPI_FUNC(unsigned long) PyThread_get_thread_ident(void);
PyAPI_FUNC(PyThread_type_lock) PyThread_allocate_lock(void);
PyAPI_FUNC(void) PyThread_free_lock(PyThread_type_lock);
PyAPI_FUNC(int) PyThread_acquire_lock(PyThread_type_lock, int);
/* 0 asks without waiting, a negative count waits until the lock is taken, and
   a positive one waits that many microseconds.  `intr_flag` has nothing to do
   here: the wait is not interruptible, so PY_LOCK_INTR is never answered. */
PyAPI_FUNC(PyLockStatus) PyThread_acquire_lock_timed(PyThread_type_lock,
                                                     PY_TIMEOUT_T microseconds,
                                                     int intr_flag);
PyAPI_FUNC(void) PyThread_release_lock(PyThread_type_lock);

#ifdef __cplusplus
}
#endif
#endif /* !Py_PYTHREAD_H */
