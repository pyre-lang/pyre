# Residual-caught handler loop (bare then nested under ValueError) plus the
# alternating-nullity interleaved loop in one process.  Each loop alone is
# fine; together a compiled residual_handler_loop whose PUSH_EXC_INFO prev
# nullity guard resumed at the handler-landing `-live-`, whose live set
# omitted the caught exception (`pyjitpl.py` `get_list_of_active_boxes`).
# A nested run's GuardIsnull failure then resumed `raise/r` (`blackhole.py`
# `bhimpl_raise` / `resume.py` `_prepare_next_section`) with a non-exception.
# `pyopcode.py PUSH_EXC_INFO` keeps `w_exc` live across `if prev_operr is
# not None`; the later `-live-` after binding that slot is the guard's.
#
# No `max-pypy-ratio`: this is a crash/oracle fixture, and its pypy arm runs
# well under a tenth of a second.
#
# Expected output:
#   bare (True, None)
#   nested (True, 'type')
#   still type
#   interleaved True ('bare', True, True)
import sys

N = 4000


def residual_handler_loop(n):
    last_inside = None
    last_after_name = None
    for i in range(n):
        try:
            {}[i]
        except KeyError:
            last_inside = sys.exc_info()[0] is KeyError
        after = sys.exc_info()[0]
        last_after_name = None if after is None else type(after).__name__
    return last_inside, last_after_name


def interleaved(n):
    interleaved_ok = True
    last = None
    for i in range(n):
        if i % 2 == 0:
            try:
                raise ValueError("o")
            except ValueError:
                try:
                    {}[i]
                except KeyError:
                    inn = sys.exc_info()[0] is KeyError
                after = sys.exc_info()[0] is ValueError
                last = ("outer", inn, after)
                if not (inn and after):
                    interleaved_ok = False
        else:
            try:
                {}[i]
            except KeyError:
                inn = sys.exc_info()[0] is KeyError
            after = sys.exc_info()[0] is None
            last = ("bare", inn, after)
            if not (inn and after):
                interleaved_ok = False
    return interleaved_ok, last


def main():
    bare = residual_handler_loop(N)
    try:
        raise ValueError("outer")
    except ValueError:
        nested = residual_handler_loop(N)
        still = type(sys.exc_info()[0]).__name__
    ok, last = interleaved(N)
    print("bare", bare)
    print("nested", nested)
    print("still", still)
    print("interleaved", ok, last)


main()
