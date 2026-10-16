# pyre-check: max-pypy-ratio=30
# Readings 7.8x-18.8x (Linux 7.8x-18.3x, macOS 11.9x-16.0x; Windows 33.4x
# once); the floor, a fourth of the ceiling, sits under the fastest. pypy exec
# is ~0.05s, so the reading moves with it.
# A list item stored just before an `append` that grows the list.
#
# `lst[k] = v` stays a lazy SETARRAYITEM_GC in the optimizer, and the
# `append` reaches `_ll_list_resize_hint_really` through a COND_CALL whose
# EffectInfo names the items ARRAY (rgc.py `ll_arraycopy` `copy_item`), so
# heap.py `force_from_effectinfo` writes the item before the copy reads it.
# Without that, the grown list carries the old item and `lst[k]` reads it
# back. The list escapes into `B.l`, so it is not virtual.
N = 2000000


class Box(object):
    pass


class P(object):
    def __init__(self, v):
        self.v = v


B = Box()


def ints(n):
    s = 0
    for i in range(n):
        lst = [0, 0, 0, 0]
        B.l = lst
        lst[0] = i
        lst.append(i)
        s += lst[0]
    return s


def floats(n):
    s = 0.0
    for i in range(n):
        lst = [0.0, 0.0, 0.0, 0.0]
        B.l = lst
        x = i * 1.0
        lst[2] = x
        lst.append(x)
        s += lst[2]
    return s


def objs(n):
    s = 0
    for i in range(n):
        lst = [None, None, None, None]
        B.l = lst
        lst[0] = P(i)
        lst.append(None)
        s += lst[0].v
    return s


expected = N * (N - 1) // 2
assert ints(N) == expected
assert floats(N) == float(expected)
assert objs(N) == expected
print("ok")
