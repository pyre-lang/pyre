# pyre-check: max-pypy-ratio=20
# str subscripts resolve a code point index, not a byte offset, for every
# payload width, and a slice keeps that indexing under negative and skipping
# steps. bytes/bytearray iterate their live buffer as ordinals, so a bytearray
# resized mid-loop is observed by the cursor rather than frozen at iter() time.
# A non-ASCII payload resolves its subscript through a code point index table
# built once per string, so `groups` walks strings long enough to span several
# of its entries and folds every index, both signs, plus the stride and group
# boundaries and slices crossing them, into one number per payload kind. A
# wrong byte offset anywhere changes that number.
# The warm loops run the ASCII and wide subscripts and the bytes cursor hot
# first so the traced paths are the ones under test; `warm_wide` scatters its
# reads over a long payload, so a regression to resolving each subscript by
# walking the buffer costs orders of magnitude rather than staying flat.
# Output verified against CPython/PyPy.
ASCII = "abcdefghij"
WIDE = "\xe9一\U0001f600ÿあ"
MIXED = "a一b\U0001f600c"
UNITS = ("\xe9\xe8\xea\xeb", "一二三四", "\U0001f600\U0001f601\U0001f602\U0001f603",
         "a\xe9一\U0001f600")


def warm_index(n):
    acc = 0
    for i in range(n):
        acc += ord(ASCII[i % 10])
    return acc


def warm_wide(n):
    # Scattered reads over a payload far longer than one index-table entry
    # covers, so resolving a subscript by walking the buffer costs thousands of
    # steps apiece instead of a lookup.
    s = UNITS[3] * 2048
    m = len(s)
    acc = 0
    for i in range(n):
        acc += ord(s[i * 7919 % m])
    return acc


def warm_bytes(n):
    payload = bytes(range(64))
    acc = 0
    for _ in range(n):
        for byte in payload:
            acc += byte
    return acc


def fold(acc, s):
    for ch in s:
        acc = (acc * 31 + ord(ch)) & 0xFFFFFFFF
    return acc


def groups(unit):
    acc = 0
    for repeat in (16, 17, 64, 100):
        s = unit * repeat
        n = len(s)
        for i in range(n):
            acc = (acc * 31 + ord(s[i])) & 0xFFFFFFFF
        for i in range(-n, 0):
            acc = (acc * 31 + ord(s[i])) & 0xFFFFFFFF
        # The 4-code-point delta stride and the 64-entry group edges, then
        # slices that cross them in both directions.
        for part in (s[0], s[63 % n], s[64 % n], s[n - 1],
                     s[60:70], s[70:60:-1], s[::-3], s[1::7]):
            acc = fold(acc, part)
        # Iteration and reversal must agree with indexing.
        assert [c for c in s] == [s[i] for i in range(n)]
        assert list(reversed(s)) == [s[i] for i in range(n - 1, -1, -1)]
    return acc


def show(label, fn):
    try:
        print(label, "->", ascii(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__, ascii(str(e)))


def main():
    print("warm_index", warm_index(666667))
    print("warm_wide", warm_wide(666667))
    print("warm_bytes", warm_bytes(400))

    for name, s in (("ascii", ASCII), ("wide", WIDE), ("mixed", MIXED)):
        print(name, "len", len(s))
        for i in range(-len(s), len(s)):
            show(f"{name}[{i}]", lambda s=s, i=i: s[i])
        show(f"{name}[oob]", lambda s=s: s[len(s)])
        show(f"{name}[-oob]", lambda s=s: s[-len(s) - 1])
        for sl in ((None, None, 2), (1, 4, None), (None, None, -1), (4, 0, -2), (-2, None, None)):
            show(f"{name}[{sl}]", lambda s=s, sl=sl: s[slice(*sl)])
        show(f"{name}.iter", lambda s=s: list(iter(s)))
        show(f"{name}.for", lambda s=s: [c for c in s])
        show(f"{name}.rev", lambda s=s: list(reversed(s)))

    for i, unit in enumerate(UNITS):
        show(f"groups[{i}]", lambda u=unit: groups(u))

    show("bytes.iter", lambda: list(iter(b"abc")))
    show("bytes.for", lambda: [x for x in b"abc"])
    show("bytearray.iter", lambda: list(iter(bytearray(b"abc"))))
    show("bytes.empty", lambda: list(iter(b"")))

    def grow():
        buf = bytearray(b"abc")
        seen = []
        for x in buf:
            seen.append(x)
            if len(seen) == 1:
                buf.append(100)
        return seen

    def shrink():
        buf = bytearray(b"abcdef")
        seen = []
        for x in buf:
            seen.append(x)
            if len(seen) == 1:
                del buf[2:]
        return seen

    show("bytearray.grow", grow)
    show("bytearray.shrink", shrink)


main()
