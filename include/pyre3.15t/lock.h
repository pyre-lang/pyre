/* PyMutex -- a mutex that occupies one byte, which the caller embeds and
   zero-initializes.

   The two low bits are the state; only the lowest is ever written here.  The
   fast paths below are inline so that an uncontended take is a compare-exchange
   and nothing else, exactly as the reference header spells it. */

#ifndef Py_LOCK_H
#define Py_LOCK_H
#ifdef __cplusplus
extern "C" {
#endif

#define _Py_UNLOCKED    0
#define _Py_LOCKED      1

typedef struct PyMutex {
    uint8_t _bits;  /* (private) */
} PyMutex;

PyAPI_FUNC(void) PyMutex_Lock(PyMutex *m);
PyAPI_FUNC(void) PyMutex_Unlock(PyMutex *m);
PyAPI_FUNC(int) PyMutex_IsLocked(PyMutex *m);

/* Takes the mutex, parking the calling thread until it is free.  A parked
   thread gives the GIL up, so whoever holds the mutex can run. */
static inline void
_PyMutex_Lock(PyMutex *m)
{
    uint8_t expected = _Py_UNLOCKED;
    if (!__atomic_compare_exchange_n(&m->_bits, &expected, _Py_LOCKED, 0,
                                     __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
        PyMutex_Lock(m);
    }
}
#define PyMutex_Lock _PyMutex_Lock

static inline void
_PyMutex_Unlock(PyMutex *m)
{
    uint8_t expected = _Py_LOCKED;
    if (!__atomic_compare_exchange_n(&m->_bits, &expected, _Py_UNLOCKED, 0,
                                     __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
        PyMutex_Unlock(m);
    }
}
#define PyMutex_Unlock _PyMutex_Unlock

static inline int
_PyMutex_IsLocked(PyMutex *m)
{
    return __atomic_load_n(&m->_bits, __ATOMIC_ACQUIRE) & _Py_LOCKED;
}
#define PyMutex_IsLocked _PyMutex_IsLocked

/* The two words a critical section occupies, which the caller embeds.  Nothing
   the macros below expand to names one -- they open a plain block -- but an
   extension that calls `PyCriticalSection_Begin` directly allocates the struct
   itself, so its size has to be the reference one. */
typedef struct PyCriticalSection {
    uintptr_t _cs_prev;   /* (private) */
    PyMutex *_cs_mutex;   /* (private) */
} PyCriticalSection;

/* A critical section serializes the operations that name the same object.
   Here every Python thread runs under one global lock, which already gives
   that ordering, so entering one costs nothing and the macros are the braces
   that scope the block an extension wrote between them.  `op` is not
   evaluated: an extension that writes `Py_BEGIN_CRITICAL_SECTION(f(x))` gets
   no call, the same as when the reference header serializes on the lock. */
#define Py_BEGIN_CRITICAL_SECTION(op)               {
#define Py_BEGIN_CRITICAL_SECTION_MUTEX(mutex)      {
#define Py_END_CRITICAL_SECTION()                   }
#define Py_BEGIN_CRITICAL_SECTION2(a, b)            {
#define Py_BEGIN_CRITICAL_SECTION2_MUTEX(m1, m2)    {
#define Py_END_CRITICAL_SECTION2()                  }

#ifdef __cplusplus
}
#endif
#endif /* !Py_LOCK_H */
