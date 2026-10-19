# CPython-suite gap: find/rfind/count stay residual in a compiled loop.
# parity-tests reason: pins unicodeobject.py _unwrap_and_search as an
# elidable call across a loop long enough to compile, including a wide
# payload and a negative end bound.
#
# str_find_bounds_virtual_int:
# test_email reaches this only through `email._header_value_parser`, and crashed there; no
# suite test drives `str.find` / `count` with int bounds in a loop hot enough to compile.
# the bounds reach the elidable search as machine ints. A boxed bound read back as an
# object crashes once the JIT makes it virtual. Virtual int bounds on str.find / rfind /
# count must reach the search as machine ints. A pointer cast of the boxed bound crashes

"""Compiled str.find / rfind / count on exact strs."""

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


def search_loop(n):
    s = "一二三四" * 64 + "zq"
    acc = 0
    i = 0
    while i < n:
        acc += s.find("zq") + s.rfind("一") + s.count("二", 7, -7)
        i = i + 1
    return acc


N = 30000
got = search_loop(N)
s = "一二三四" * 64 + "zq"
expect = N * (s.find("zq") + s.rfind("一") + s.count("二", 7, -7))
assert got == expect, (got, expect)

# Empty needle and a miss stay correct after compile.
t = "abcabc"
assert t.find("") == 0
assert t.rfind("z") == -1
assert t.count("a", 1, 5) == 1

def str_find_bounds_virtual_int():
    N = 20000


    def parse(s):
        s = ";" + str(s)
        plist = []
        start = 0
        while s.find(";", start) == start:
            start += 1
            assert s.rfind(";", start - 1, start) == start - 1
            end = s.find(";", start)
            ind, diff = start, 0
            while end > 0:
                diff += s.count('"', ind, end) - s.count('\\"', ind, end)
                if diff % 2 == 0:
                    break
                end, ind = ind, s.find(";", end + 1)
            if end < 0:
                end = len(s)
            i = s.find("=", start, end)
            if i == -1:
                f = s[start:end]
            else:
                f = s[start:i].rstrip().lower() + "=" + s[i + 1 : end].lstrip()
            plist.append(f.strip())
            start = end
        return plist


    def main():
        for s in ["", "foo=bar", " FOO = bar    "]:
            src = f"{s};" * (N - 1) + s
            res = parse(src)
            assert (len(res), len(set(res))) == (N, 1), (s, len(res), len(set(res)))


    main()
str_find_bounds_virtual_int()
print("OK")
