/* `slice`.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_SLICEOBJECT_H
#define PYRE_SLICEOBJECT_H

#ifdef __cplusplus
extern "C" {
#endif
/* slice. */

/* A `slice` mirror carries its three bounds as fields, which is how an
   extension reads them without a call: `sliceobject.py slice_attach` fills
   them when the mirror is minted and its deallocator releases them.  They are
   the object the slice was built with, so a bound left out is `None` rather
   than NULL. */
typedef struct {
    PyObject_HEAD
    PyObject *start;
    PyObject *stop;
    PyObject *step;
} PySliceObject;

/* The spelling every extension compiles against.  The exported function is the
   same composition, so a caller reaching either gets the same answer. */
#define PySlice_GetIndicesEx(slice, length, start, stop, step, slicelen) (      \
    PySlice_Unpack((slice), (start), (stop), (step)) < 0 ?                      \
    ((*(slicelen) = 0), -1) :                                                   \
    ((*(slicelen) = PySlice_AdjustIndices((length), (start), (stop), *(step))), \
     0))

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_SLICEOBJECT_H */
