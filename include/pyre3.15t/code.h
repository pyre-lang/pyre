/* The flag bits a code object's `co_flags` is made of.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_CODE_H
#define PYRE_CODE_H

#ifdef __cplusplus
extern "C" {
#endif
/* Each bit is the one `inspect` publishes under the same name, so a value an
   extension builds here and a value it reads from Python mean the same thing.
   A code object itself stays opaque -- `PyCodeObject` is named in
   `pytypedefs.h` and has no fields an extension may read. */
#define CO_OPTIMIZED            0x0001
#define CO_NEWLOCALS            0x0002
#define CO_VARARGS              0x0004
#define CO_VARKEYWORDS          0x0008
#define CO_NESTED               0x0010
#define CO_GENERATOR            0x0020
#define CO_COROUTINE            0x0080
#define CO_ITERABLE_COROUTINE   0x0100
#define CO_ASYNC_GENERATOR      0x0200
#define CO_HAS_DOCSTRING        0x4000000
#define CO_METHOD               0x8000000

/* The `__future__` bits and the monitoring opt-out share `co_flags` with the
   set above but are not published by `inspect`: a compiler flag names them,
   and an extension composing one for a compile call needs the value.  They
   start at 0x20000 so that the `PyCF_` compiler flags can have the range
   below them. */
#define CO_FUTURE_DIVISION          0x20000
#define CO_FUTURE_ABSOLUTE_IMPORT   0x40000
#define CO_FUTURE_WITH_STATEMENT    0x80000
#define CO_FUTURE_PRINT_FUNCTION    0x100000
#define CO_FUTURE_UNICODE_LITERALS  0x200000
#define CO_FUTURE_BARRY_AS_BDFL     0x400000
#define CO_FUTURE_GENERATOR_STOP    0x800000
#define CO_FUTURE_ANNOTATIONS       0x1000000
#define CO_NO_MONITORING_EVENTS     0x2000000

#define PyCode_Check(op) Py_IS_TYPE((op), &PyCode_Type)

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_CODE_H */
