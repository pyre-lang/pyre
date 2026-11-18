# pyre-check: gate=1
n = 10000
a = b'a' * n
b = b'b' * n
haystack = a + a + b + a + a
needle = a + b + b + a
result = True
for constructor in (bytes, bytearray):
    value = constructor(haystack)
    sub = constructor(needle)
    result = (
        result
        and value.find(sub) == -1
        and value.rfind(sub) == -1
        and value.count(sub) == 0
        and (value + sub).find(sub) == len(value)
        and (value + sub).count(sub) == 1
    )

assert result


class Needle:
    def __init__(self, hay, index=0):
        self.hay = hay
        self.index = index

    def __buffer__(self, flags):
        self.hay[self.index] = ord("z")
        return memoryview(b"z")


class GrowNeedle:
    def __init__(self, hay):
        self.hay = hay

    def __buffer__(self, flags):
        self.hay.extend(b"zzzz")
        return memoryview(b"z")


h = bytearray(b"abc")
assert h.find(Needle(h)) == 0
assert h == bytearray(b"zbc")
h = bytearray(b"abc")
assert h.rfind(Needle(h)) == 0
assert h == bytearray(b"zbc")
h = bytearray(b"abc")
assert h.index(Needle(h)) == 0
assert h == bytearray(b"zbc")
h = bytearray(b"abc")
assert h.rindex(Needle(h)) == 0
assert h == bytearray(b"zbc")
h = bytearray(b"abc")
assert h.count(Needle(h)) == 1
assert h == bytearray(b"zbc")
h = bytearray(b"abc")
assert (Needle(h) in h) is True
assert h == bytearray(b"zbc")
h = bytearray(b"abc")
assert h.startswith(Needle(h)) is True
assert h == bytearray(b"zbc")
h = bytearray(b"abc")
assert h.startswith((Needle(h),)) is True
assert h == bytearray(b"zbc")
h = bytearray(b"abc")
assert h.endswith(Needle(h, -1)) is True
assert h == bytearray(b"abz")
h = bytearray(b"abc")
assert h.endswith((Needle(h, -1),)) is True
assert h == bytearray(b"abz")
h = bytearray(b"abc")
assert h.removeprefix(Needle(h)) == bytearray(b"bc")
assert h == bytearray(b"zbc")
h = bytearray(b"abc")
assert h.removesuffix(Needle(h, -1)) == bytearray(b"ab")
assert h == bytearray(b"abz")


def grow_refused(op):
    h = bytearray(b"abc")
    try:
        op(h)
        refused = False
    except BufferError:
        refused = True
    assert refused and h == bytearray(b"abc")


grow_refused(lambda h: h.find(GrowNeedle(h)))
grow_refused(lambda h: h.rfind(GrowNeedle(h)))
grow_refused(lambda h: h.index(GrowNeedle(h)))
grow_refused(lambda h: h.rindex(GrowNeedle(h)))
grow_refused(lambda h: h.count(GrowNeedle(h)))
grow_refused(lambda h: GrowNeedle(h) in h)
grow_refused(lambda h: h.startswith(GrowNeedle(h)))
grow_refused(lambda h: h.endswith(GrowNeedle(h)))
