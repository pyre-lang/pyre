/* The pair a complex number is carried across the boundary as.
 *
 * Included before `pyre_decl.h` rather than beside the other per-type headers,
 * because entry points there take and answer with one of these by value and so
 * need the definition, not a forward name.
 */
#ifndef PYRE_COMPLEXOBJECT_H
#define PYRE_COMPLEXOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double real;
    double imag;
} Py_complex;

/* The block a `complex` is mirrored into.  The pair is a snapshot filled when
   the block is made; `complex` is immutable, so it cannot come apart from the
   object it was read off. */
typedef struct {
    PyObject_HEAD
    Py_complex cval;
} PyComplexObject;

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_COMPLEXOBJECT_H */
