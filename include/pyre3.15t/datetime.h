/* The `datetime` C API.
 *
 * `Python.h` includes this, as `cpyext/include/Python.h` does: the entry
 * points `pyre_decl.h` declares name `PyDateTime_CAPI`, so the struct has to
 * be in scope by then.  An extension that includes it again on its own -- the
 * spelling every extension uses -- gets the guard.
 *
 * `datetime` is a Python module here (`cpyext/cdatetime.py` states the same
 * for PyPy), so the accessors CPython spells as macros over an instance struct
 * cannot read one: each is an entry point declared in `pyre_decl.h` that reads
 * the attribute it names, and this header leaves those names alone so the
 * declaration is what a call resolves to.
 *
 * `pyre/pyre-interpreter/src/cpyext/cdatetime.rs` carries the twin.
 */
#ifndef DATETIME_H
#define DATETIME_H

#ifdef __cplusplus
extern "C" {
#endif

/* The two fields an extension does read through a block. `hastzinfo` is 0 for
   a naive object, and `tzinfo` is then NULL. */
typedef struct {
    PyObject_HEAD
    unsigned char hastzinfo;
    PyObject *tzinfo;
} PyDateTime_Time;

typedef struct {
    PyObject_HEAD
    unsigned char hastzinfo;
    PyObject *tzinfo;
} PyDateTime_DateTime;

typedef struct {
    PyObject_HEAD
    int days;
    int seconds;
    int microseconds;
} PyDateTime_Delta;

typedef struct {
    PyObject_HEAD
} PyDateTime_Date;

typedef struct {
    PyObject_HEAD
} PyDateTime_TZInfo;

/* The table `PyDateTime_IMPORT` binds. */
typedef struct {
    /* type objects */
    PyTypeObject *DateType;
    PyTypeObject *DateTimeType;
    PyTypeObject *TimeType;
    PyTypeObject *DeltaType;
    PyTypeObject *TZInfoType;

    /* singletons */
    PyObject *TimeZone_UTC;

    /* constructors */
    PyObject *(*Date_FromDate)(int, int, int, PyTypeObject *);
    PyObject *(*DateTime_FromDateAndTime)(int, int, int, int, int, int, int,
                                          PyObject *, PyTypeObject *);
    PyObject *(*Time_FromTime)(int, int, int, int, PyObject *, PyTypeObject *);
    PyObject *(*Delta_FromDelta)(int, int, int, int, PyTypeObject *);
    PyObject *(*TimeZone_FromTimeZone)(PyObject *, PyObject *);

    /* constructors for the DB API */
    PyObject *(*DateTime_FromTimestamp)(PyObject *, PyObject *, PyObject *);
    PyObject *(*Date_FromTimestamp)(PyObject *, PyObject *);

    /* PEP 495 constructors */
    PyObject *(*DateTime_FromDateAndTimeAndFold)(int, int, int, int, int, int,
                                                 int, PyObject *, int,
                                                 PyTypeObject *);
    PyObject *(*Time_FromTimeAndFold)(int, int, int, int, PyObject *, int,
                                      PyTypeObject *);
} PyDateTime_CAPI;

/* Upstream's `PyDateTimeAPI` is a per-translation-unit `static` an extension
   fills from a capsule.  Here it is one object the runtime owns, which is what
   `cpyext/include/datetime.h` does. */
PyAPI_DATA(PyDateTime_CAPI *) PyDateTimeAPI;

#define PyDateTime_IMPORT (PyDateTimeAPI = _PyDateTime_Import())

/* Macros for accessing constructors in a simplified fashion. */
#define PyDate_FromDate(year, month, day) \
    PyDateTimeAPI->Date_FromDate(year, month, day, PyDateTimeAPI->DateType)

#define PyDateTime_FromDateAndTime(year, month, day, hour, min, sec, usec) \
    PyDateTimeAPI->DateTime_FromDateAndTime(year, month, day, hour, \
        min, sec, usec, Py_None, PyDateTimeAPI->DateTimeType)

#define PyDateTime_FromDateAndTimeAndFold(year, month, day, hour, min, sec, usec, fold) \
    PyDateTimeAPI->DateTime_FromDateAndTimeAndFold(year, month, day, hour, \
        min, sec, usec, Py_None, fold, PyDateTimeAPI->DateTimeType)

#define PyTime_FromTime(hour, minute, second, usecond) \
    PyDateTimeAPI->Time_FromTime(hour, minute, second, usecond, \
        Py_None, PyDateTimeAPI->TimeType)

#define PyTime_FromTimeAndFold(hour, minute, second, usecond, fold) \
    PyDateTimeAPI->Time_FromTimeAndFold(hour, minute, second, usecond, \
        Py_None, fold, PyDateTimeAPI->TimeType)

#define PyDelta_FromDSU(days, seconds, useconds) \
    PyDateTimeAPI->Delta_FromDelta(days, seconds, useconds, 1, \
        PyDateTimeAPI->DeltaType)

#define PyTimeZone_FromOffset(offset) \
    PyDateTimeAPI->TimeZone_FromTimeZone(offset, NULL)

#define PyTimeZone_FromOffsetAndName(offset, name) \
    PyDateTimeAPI->TimeZone_FromTimeZone(offset, name)

/* Access to the UTC singleton. */
#define PyDateTime_TimeZone_UTC PyDateTimeAPI->TimeZone_UTC

/* PEP 495 named this one and CPython implemented the other, so both spellings
   have to reach the same entry point. */
#define PyDateTime_DATE_GET_FOLD PyDateTime_GET_FOLD

#ifdef __cplusplus
}
#endif
#endif /* !DATETIME_H */
