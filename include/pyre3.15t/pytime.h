/* The `PyTime_t` clock API.
 *
 * A `PyTime_t` is a count of nanoseconds, and the entry points that read a
 * clock are declared in `pyre_decl.h`.  This header carries the type itself,
 * which those declarations name, so it is included ahead of them.
 */
#ifndef PYRE_PYTIME_H
#define PYRE_PYTIME_H

#ifdef __cplusplus
extern "C" {
#endif

typedef int64_t PyTime_t;
#define PyTime_MIN INT64_MIN
#define PyTime_MAX INT64_MAX

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_PYTIME_H */
