/* `long double` accessors.
 *
 * Rust has no `long double`, and its width and representation differ per
 * target (80-bit x87 padded to 16 bytes on x86-64 System V, plain `double`
 * on aarch64 Darwin, 128-bit quad elsewhere).  The C compiler is the only
 * thing that knows, so every operation on that representation is behind one
 * of these shims.
 *
 * `misc.py` reaches for C the same way: `pypy__is_nonnull_longdouble` is an
 * `ExternalCompilationInfo` snippet, and `longdouble2str` calls `sprintf`
 * with `%LE`.
 */

#include "src/precommondefs.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

struct pyre_cffi_align_long_double {
    char x;
    long double y;
};

RPY_EXTERN size_t pyre_cffi_sizeof_long_double(void)
{
    return sizeof(long double);
}

RPY_EXTERN size_t pyre_cffi_alignof_long_double(void)
{
    return offsetof(struct pyre_cffi_align_long_double, y);
}

RPY_EXTERN double pyre_cffi_read_long_double(const char *p)
{
    long double value;
    memcpy(&value, p, sizeof(long double));
    return (double)value;
}

RPY_EXTERN void pyre_cffi_write_long_double(char *p, double v)
{
    long double value = (long double)v;
    memcpy(p, &value, sizeof(long double));
}

/* `misc.py:_is_nonnull_longdouble` — the comparison has to happen in the
 * `long double` domain, so it cannot be done on the copied-out `double`. */
RPY_EXTERN int pyre_cffi_nonnull_long_double(const char *p)
{
    long double value;
    memcpy(&value, p, sizeof(long double));
    return value != 0.0L;
}

/* `misc.py:longdouble2str`. */
RPY_EXTERN void pyre_cffi_str_long_double(const char *p, char *out, size_t n)
{
    long double value;
    memcpy(&value, p, sizeof(long double));
    snprintf(out, n, "%LE", value);
}

/* The `int_fast*_t` layouts.
 *
 * C picks these for speed rather than width, so they differ per platform:
 * glibc makes `int_fast16_t` a `long`, Darwin makes it a `short`, and MSVC
 * makes it an `int`.  `rfficache.py` learns the same numbers the same way,
 * by asking the C compiler, which is why they cannot be spelled in Rust.
 */

#define PYRE_CFFI_FAST_INT(bits)                                              \
    struct pyre_cffi_align_fast##bits {                                       \
        char x;                                                               \
        int_fast##bits##_t y;                                                 \
    };                                                                        \
    RPY_EXTERN size_t pyre_cffi_sizeof_fast##bits(void)                       \
    {                                                                         \
        return sizeof(int_fast##bits##_t);                                    \
    }                                                                         \
    RPY_EXTERN size_t pyre_cffi_alignof_fast##bits(void)                      \
    {                                                                         \
        return offsetof(struct pyre_cffi_align_fast##bits, y);                \
    }

PYRE_CFFI_FAST_INT(8)
PYRE_CFFI_FAST_INT(16)
PYRE_CFFI_FAST_INT(32)
PYRE_CFFI_FAST_INT(64)

/* `wchar_t`, which is two bytes on Windows and four on the platforms whose
   `wchar_t` is an `int`.  `rffi.py` lists it among the names `rfficache`
   measures with the C compiler rather than spelling out. */

struct pyre_cffi_align_wchar {
    char x;
    wchar_t y;
};

RPY_EXTERN size_t pyre_cffi_sizeof_wchar(void)
{
    return sizeof(wchar_t);
}

RPY_EXTERN size_t pyre_cffi_alignof_wchar(void)
{
    return offsetof(struct pyre_cffi_align_wchar, y);
}
