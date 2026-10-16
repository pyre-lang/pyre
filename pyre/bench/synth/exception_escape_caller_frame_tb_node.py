# pyre-check: max-pypy-ratio=22
# Local dynasm check.py reads ~11x; 22 is twice that. 116 was the N=40000
# blanket from the first synth sweep.
# An exception escaping a compiled frame that holds a try block it does NOT
# match must keep the CALLER's traceback node:
#
#   correct:            ('<module>', 'd_nonmatching_except', 'mid', 'leaf')
#   caller node lost:   ('d_nonmatching_except', 'mid', 'leaf')
#
# The non-matching clause falls through to the except-cleanup RERAISE, so the
# blackhole classified the exiting frame as a bare reraise and cleared
# `attach_tb` on the escaping error.  That flag suppresses one frame's
# traceback record, and the compiled frame never runs `handle_exception`
# itself - so the first frame to read it was the interpreter caller, which
# then dropped its own node.  The compiled frame's own record decision is
# already made where the record is skipped, so the flag must not leave it.
#
# `d_no_try` is the control: with no try block in the compiled frame the
# escape takes the guard-exception exit, which never touched `attach_tb`, and
# the caller node survived even before the fix.  `d_finally` and
# `d_while_nonmatching` cover the other two cleanup shapes that reach the same
# RERAISE, and `d_no_mid` pins that neither an inlined intermediate frame nor
# a compiled exception-edge bridge is needed to trigger it.
N = 1600000


def names(e):
    out = []
    tb = e.__traceback__
    while tb is not None:
        out.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    return tuple(out)


def leaf(i, n):
    if i == n - 1:
        raise KeyError("escape")
    return i


def mid(i, n):
    return leaf(i, n)


def d_no_try(n):
    acc = 0
    for i in range(n):
        acc += mid(i, n)
    return acc


def d_nonmatching_except(n):
    acc = 0
    for i in range(n):
        try:
            acc += mid(i, n)
        except ValueError:
            acc += 1
    return acc


def d_finally(n):
    acc = 0
    for i in range(n):
        try:
            acc += mid(i, n)
        finally:
            acc += 0
    return acc


def d_no_mid(n):
    acc = 0
    for i in range(n):
        try:
            acc += leaf(i, n)
        except ValueError:
            acc += 1
    return acc


def d_while_nonmatching(n):
    acc = 0
    i = 0
    while i < n:
        try:
            acc += mid(i, n)
        except ValueError:
            acc += 1
        i += 1
    return acc


for driver in (
    d_no_try,
    d_nonmatching_except,
    d_finally,
    d_no_mid,
    d_while_nonmatching,
):
    try:
        driver(N)
    except KeyError as e:
        print(driver.__name__, names(e))
