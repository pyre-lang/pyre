# pyre-check: requires-modules=unicodedata
# pyre-check: pypy-diverges: pypy3 implements 3.11, which has none of these 3.15 unicodedata additions
# CPython-suite gap: test_unicodedata covers isxidstart/iter_graphemes separately;
# this fixture pins the combined 3.15 module-level surface.
# parity-tests reason: 3.15 adds unicodedata.isxidstart/isxidcontinue,
# grapheme_cluster_break/indic_conjunct_break/extended_pictographic/iter_graphemes
# (with GraphemeBreakIterator and Segment), module-level only.

import unicodedata as u

s = "e\u0301x\U0001F468\u200D\U0001F469"
it = u.iter_graphemes(s, 0)
print("it-type", type(it).__name__, type(it).__module__)
print("iter-is", iter(it) is it)
seg = next(it)
print("seg", repr(seg), str(seg), seg.start, seg.end)
print("seg-type", type(seg).__name__, type(seg).__module__)
print("dir-tail", dir(seg)[-4:])
print("remaining", list(map(str, it)))

print("neg", list(map(str, u.iter_graphemes("abc", -2))))
print("neg-end", list(map(str, u.iter_graphemes("abc", 0, -1))))
print("oversize-start", list(map(str, u.iter_graphemes("abc", 5))))
print("oversize-end", list(map(str, u.iter_graphemes("abc", 0, 99))))
print("empty", list(map(str, u.iter_graphemes(""))))
print("surrogate", list(map(str, u.iter_graphemes("a\ud800b"))))

try:
    type(seg)()
except TypeError as e:
    print("seg-new", str(e))
try:
    type(u.iter_graphemes("a"))()
except TypeError as e:
    print("gbi-new", str(e))
print("hasattr-Segment", hasattr(u, "Segment"))
print("hasattr-GBI", hasattr(u, "GraphemeBreakIterator"))

chars = ["a", "_", "1", "$", "é", "·", "\ud800"]
for c in chars:
    print("xid", repr(c), u.isxidstart(c), u.isxidcontinue(c))

for c in ["\r", "\n", "a", "\u0301", "\u200d", "\U0001F1E6", "ᄀ"]:
    print("gcb", repr(c), u.grapheme_cluster_break(c))

for c in ["क", "्", "a"]:
    print("incb", repr(c), u.indic_conjunct_break(c))

for c in ["\U0001F600", "a"]:
    print("ext", repr(c), u.extended_pictographic(c))

for name in (
    "isxidstart",
    "isxidcontinue",
    "grapheme_cluster_break",
    "indic_conjunct_break",
    "extended_pictographic",
):
    fn = getattr(u, name)
    try:
        fn("ab")
    except TypeError as e:
        print(name + "-len2", str(e))
    try:
        fn(5)
    except TypeError as e:
        print(name + "-int", str(e))

try:
    u.iter_graphemes(5)
except TypeError as e:
    print("iter-int", str(e))

missing = [
    n
    for n in (
        "isxidstart",
        "isxidcontinue",
        "grapheme_cluster_break",
        "indic_conjunct_break",
        "extended_pictographic",
        "iter_graphemes",
    )
    if hasattr(u.ucd_3_2_0, n)
]
print("ucd32", missing)

print("OK")
