# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=drive,entry-bridge:locs
# A traceback node must expose the catching frame's live locals while its
# exception handler is still running, including the exception target `e`.
#
# This used to be parked with a wasm exemption.  Once `drive` and `locs` ran as
# compiled portal callees, wasm intermittently read the outermost traceback
# frame as if the implicit handler-exit `del e` had already run.  The resulting
# set contained both the live shape and a stale shape without `e`:
#
#     ("drive", ("e", "k", "seen"))
#     ("drive", ("k", "seen"))
#
# The same fixture also covers the GC-root walk that motivated it originally:
# reading `f_locals` through the traceback can leave a virtual ref in the
# frame's value area while a nursery allocation from compiled code triggers a
# collection.  `walk_frame_value_slot` must forward that virtual ref before
# the raw exception-root walk interprets the slot as a PyObject.
#
# PyPy's `MIFrame` and blackhole resume machinery keep one red frame per
# inlined call.  Keeping `drive`, `locs`, and `mid` attached to their own live
# frames is therefore load-bearing here: the traceback frame's locals cannot
# be recovered from one portal-wide anchor.  The compile declarations pin the
# `drive` loop, which inlines `mid` and the first `locs` iteration, and the
# `locs` entry bridge it calls at the first back-edge, so an interpreted
# fallback cannot make this test pass vacuously.

N = 90000


def mid(i):
    raise ValueError("boom")


def locs(tb):
    out = []
    idx = 0
    while tb is not None:
        frame = tb.tb_frame
        if idx == 0:
            out.append((frame.f_code.co_name, tuple(sorted(frame.f_locals))))
        else:
            out.append(frame.f_code.co_name)
        tb = tb.tb_next
        idx += 1
    return tuple(out)


def drive():
    seen = set()
    k = 0
    while k < N:
        try:
            mid(k)
        except ValueError as e:
            seen.add(locs(e.__traceback__))
            e.__traceback__ = None
        k += 1
    return seen


actual = drive()
expected = {(('drive', ('e', 'k', 'seen')), 'mid')}
assert actual == expected, (actual, expected)
print("PASS")
