# pyre-check: trace-shape=warm_find:absent=CallMayForceR
# The bound-method fold, 267 firings across the corpus and undeclared. The str
# method calls below drive it 22 times, more than any other fixture.
# Every str surface that converts between a code point index and a byte offset:
# find/rfind/index/rindex and count take code point bounds and report a code
# point result, startswith/endswith slice by them, and the width methods read
# the string's own length. Each is folded over payloads whose code points are
# 1-4 bytes wide and over strings long enough to span several index-table
# groups, so an off-by-one in either direction changes the number.
# `warm_find` searches a long wide payload repeatedly: resolving those bounds
# by walking, or by materialising the whole offset table per call, costs orders
# of magnitude rather than staying flat. Output verified against CPython/PyPy.
SAMPLES = (
    ("empty", ""),
    ("ascii", "abcabcabc"),
    ("latin1", "\xe9\xe8\xe9\xe8\xe9"),
    ("wide", "一二一二一"),
    ("astral", "\U0001f600\U0001f601\U0001f600\U0001f601"),
    ("mixed", "a一b\U0001f600c一a"),
    ("long_wide", "一二三四" * 40),
    ("long_mixed", ("a\xe9一\U0001f600" * 40) + "zz"),
)
NEEDLES = ("", "a", "一", "\U0001f600", "ab", "一二", "zz")
BOUNDS = (None, 0, 1, 3, -1, -3, -100, 100, 63, 64, 65, 160)


def warm_find(reps):
    s = "一二三四" * 2048 + "zq"
    acc = 0
    for _ in range(reps):
        acc += s.find("zq") + s.rfind("一") + s.count("二", 7, -7)
    return acc


def fold(acc, value):
    for ch in str(value):
        acc = (acc * 31 + ord(ch)) & 0xFFFFFFFF
    return acc


def searches(s):
    acc = 0
    for sub in NEEDLES:
        for a in BOUNDS:
            for b in BOUNDS:
                acc = fold(acc, s.find(sub, a, b))
                acc = fold(acc, s.rfind(sub, a, b))
                # An empty needle is excluded from count: PyPy answers the
                # byte length plus one on a non-ASCII payload where 3.14
                # answers the code point length plus one, so the corpus
                # cannot compare it. Every other needle agrees.
                if sub:
                    acc = fold(acc, s.count(sub, a, b))
                acc = fold(acc, s.startswith(sub, a, b))
                acc = fold(acc, s.endswith(sub, a, b))
                for meth in (s.index, s.rindex):
                    # Only the raised type, not its text: the "substring not
                    # found" message is worded differently across runtimes.
                    try:
                        acc = fold(acc, meth(sub, a, b))
                    except ValueError:
                        acc = fold(acc, "ValueError")
    # index/rindex must agree with find/rfind wherever the latter hit.
    for sub in NEEDLES:
        assert (s.index(sub) if s.find(sub) >= 0 else -1) == s.find(sub)
        assert (s.rindex(sub) if s.rfind(sub) >= 0 else -1) == s.rfind(sub)
    return acc


def widths(s):
    acc = 0
    for width in (0, 1, 5, 12, 200):
        acc = fold(acc, s.center(width, "一"))
        acc = fold(acc, s.ljust(width, "\U0001f600"))
        acc = fold(acc, s.rjust(width, "\xe9"))
        acc = fold(acc, s.zfill(width))
        # A padded result is exactly `width` code points once it is wider.
        assert len(s.center(width, "一")) == max(width, len(s))
    return acc


def main():
    print("warm_find", warm_find(2000))
    for name, s in SAMPLES:
        print(name, "search", searches(s), "width", widths(s))
    for x, y in (("", ""), ("ab", "一二"), ("一二", "ab"), ("abc", "ab")):
        try:
            print("maketrans", ascii(x), ascii(y), sorted(str.maketrans(x, y).items()))
        except ValueError as e:
            print("maketrans", ascii(x), ascii(y), "!!", ascii(str(e)))


main()
