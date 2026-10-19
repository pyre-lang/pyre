# CPython-suite gap: traceback tests cannot create, clear, and walk pyre JIT
# frames after return or exception unwind.
# parity-tests reason: this guards JIT frame ownership and the executing
# caller's operand stack while a traceback chain is cleared.
#
# jit_loop_exit_frame_clear:
# a frame returned after a compiled loop exit must be finished and clearable after
# resuming normal execution. cover frame lifecycle across a JIT loop exit.

"""Completed JIT frames clear; the executing frame refuses without damage."""

import sys


def returns_frame():
    return sys._getframe()


for _ in range(3000):
    frame = returns_frame()
    assert frame.f_code is returns_frame.__code__
    frame.clear()


def raises():
    raise ValueError("expected")


for _ in range(3000):
    try:
        raises()
    except ValueError as exc:
        traceback = exc.__traceback__
        while traceback.tb_next is not None:
            traceback = traceback.tb_next
        assert traceback.tb_frame.f_code is raises.__code__
        traceback.tb_frame.clear()
    else:
        raise AssertionError("raises() returned")


def clear_chain(traceback):
    while traceback is not None:
        try:
            traceback.tb_frame.clear()
        except RuntimeError:
            pass
        traceback = traceback.tb_next


for _ in range(4000):
    try:
        raises()
    except ValueError:
        clear_chain(sys.exc_info()[2])

def jit_loop_exit_frame_clear():
    import sys


    def frame_after_loop(n):
        frame = sys._getframe()
        total = 0
        for i in range(n):
            total += i
        return frame


    for _ in range(30):
        frame = frame_after_loop(500)
        assert frame.f_code is frame_after_loop.__code__
        assert frame.f_locals["total"] == 124750
        frame.clear()
jit_loop_exit_frame_clear()
print("OK")
