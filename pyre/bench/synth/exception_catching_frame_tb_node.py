# pyre-check: max-pypy-ratio=14
# Tightened 53 -> 42 when `seeded_callee_resume` stopped requiring the
# callee's own exception table: execution-only time here fell 1.53x against a
# same-day build of the parent commit, so the previous headroom is kept and
# then some -- the ceiling moves by less than the measured gain.
# Before that change the runners read this fixture between 3.0x and 26.2x
# and the ceiling was twice the slowest of those.
# The frame holding the `try` contributes its own traceback node, including
# when its `except` is reached from inside its own compiled trace.
#
# When an inlined callee raises, the caller-side handler scan routes the raise
# straight to the caller's `except` and the walk keeps going, so the handler
# ends up inside the trace. Compiled, the trace then catches the exception
# itself: the frame never surfaces an error to the interpreter, so unless the
# trace records the node it is the recording pass alone that ever applies it
# and the chain comes out missing its outermost frame.
#
# The module-level loop compiles with `shape` inlined into it, and with
# `raise_mid` and `raise_leaf` inlined in turn, so the whole raise-and-catch
# chain sits in one trace and the handler runs as compiled code. pypy compiles
# the same arrangement: a 688-op `<module>` loop whose merge points read
# `shape, raise_mid, raise_leaf, shape, frame_names`. The dict lookup in the
# loop stays because it is what the recorded counters were measured against.
#
# wasm reads the same three loops but serves one bridge fewer and fails guards
# 4.5x as often as the native backends on the identical trace. Nothing in the
# bridge-decline census is nonzero on either side and the guest prints no
# `mc_diag`, so that reading is recorded rather than explained.
N = 240000


def frame_names(traceback):
    names = []
    while traceback is not None:
        names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    return tuple(names)


def raise_leaf(i):
    raise ValueError(i)


def raise_mid(i):
    raise_leaf(i)


def shape(i):
    try:
        raise_mid(i)
    except ValueError as e:
        return frame_names(e.__traceback__)


seen = {}
shapes = set()
for i in range(N):
    shapes.add(shape(i))
    seen.get(0, 0)
print(sorted(shapes))
