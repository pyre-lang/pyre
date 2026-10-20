/* Imports.
 *
 * One of the headers `Python.h` includes; an extension includes only
 * `Python.h`. The exported entry points are declared together in
 * `pyre_decl.h`, which is generated.
 */
#ifndef PYRE_IMPORT_H
#define PYRE_IMPORT_H

#ifdef __cplusplus
extern "C" {
#endif
/* Imports.  The borrowed-reference `PyImport_AddModule` is absent: pyre has no
   container to hang the borrow on, so only the strong-reference form exists.
   `PyImport_GetModuleDict` is present because `sys.modules` outlives every
   caller, which is what makes its borrow sound. */

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_IMPORT_H */
