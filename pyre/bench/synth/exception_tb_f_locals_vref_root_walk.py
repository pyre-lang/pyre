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
# The PyPy oracle (`PYPYLOG=jit-log-opt` on pypy3 3.11) compiles `loop:locs`
# and `loop:drive` and nothing for `mid`; pyre now also compiles nothing for
# `mid` (it is inlined into the traced callers).  `locs` is an entry bridge
# instead of a loop because CPython 3.14 compiles `while tb is not None:`
# with the `JUMP_BACKWARD` landing on the condition test, so when the
# back-edge counter fires the traced iteration can take the exit and the
# trace ends in FINISH (`compile_done_with_this_frame`); PyPy's compiler
# lands the back edge in the loop body.  The declarations pin the shapes
# so an interpreted fallback cannot pass vacuously.

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
