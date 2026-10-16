# pyre-check: spec-folds=builtin_len_descent
# pyre-check: trace-shape=hot_len:absent=CallMayForceR
# Hot-loop str/unicode subscript and length over every string kind: ASCII /
# latin1 (1-byte code units), BMP (2-byte), and non-BMP astral (4-byte). The
# subscripts emit residual STRGETITEM/UNICODEGETITEM and the lengths STRLEN/
# UNICODELEN, exercising the backend's descr-driven item/length reads at each
# item size (item_size 1/2/4, with the STR null-terminator base adjustment).
# A rolling ordinal checksum makes a wrong offset or width a visibly wrong
# number rather than a silent pass. Deterministic; output asserted cpython==pypy.
#
# `hot_len` also runs over `bytes` and `bytearray`. The `bytes` arm reads
# `W_BytesObject.len` — the precomputed count standing in for the `strlen`
# `bytesobject.py` takes off `_value` — and the `bytearray` arm reads
# `W_BytearrayObject.length`, which is what `bytearrayobject.py`'s `_len`
# reaches through `self._data` as `rlist.py`'s `("length", Signed)`. Without
# those paths the call stays an opaque forcing residual; `spec-folds` catches
# a regression in the generated builtin descent directly.
#
# `hot_mutating_len` is a correctness leg, not a speed one. `bytearray`'s
# length is MUTABLE, so the compiled loop re-reads the field instead of
# hoisting it, and the field only stays right because every length-changing
# mutator republishes it through `w_bytearray_sync_alloc`. Appending and
# deleting a prefix inside the loop walks both the push path and the one that
# moves `logical_offset`; a length that went stale anywhere shows up as a wrong
# checksum here rather than as silently wrong compiled code.


def hot_checksum(n, s):
    length = len(s)
    acc = 0
    for i in range(n):
        acc = (acc * 131 + ord(s[i % length]) + length) & 0xFFFFFFFFFFFF
    return acc


def hot_len(n, s):
    acc = 0
    for i in range(n):
        acc += len(s)
    return acc


def hot_mutating_len(n, ba):
    acc = 0
    for i in range(n):
        ba.append(i & 0xFF)
        acc = (acc + len(ba)) & 0xFFFFFFFFFFFF
        if len(ba) > 64:
            del ba[0:32]
    return acc


def main():
    try:
        import pypyjit

        pypyjit.set_param("threshold=20,function_threshold=20")
    except ImportError:
        pass

    ascii_s = "The quick brown fox jumps over the lazy dog 0123456789!?"
    latin1_s = "café déjà vu naïve résumé — ¡Hola! ½¾ ©®µ"
    bmp_s = "αβγδεζ ελληνικά Ω ДЖЕМ кириллица 日本語 テスト"
    astral_s = "𝕒𝕓𝕔𝕕 𝟙𝟚𝟛𝟜 😀🎉🚀 𐀀𐀁 mixed astral"

    ascii_b = b"The quick brown fox jumps over the lazy dog 0123456789!?"
    short_b = b"abc"

    n = 120000
    print("ascii", hot_checksum(n, ascii_s))
    print("latin1", hot_checksum(n, latin1_s))
    print("bmp", hot_checksum(n, bmp_s))
    print("astral", hot_checksum(n, astral_s))
    print("len", hot_len(n, ascii_s), hot_len(n, bmp_s), hot_len(n, astral_s))
    # The descent census verifies the immutable bytes and mutable bytearray
    # length bodies directly; each leg only needs to stay hot enough to compile.
    bn = 300000
    print("blen", hot_len(bn, ascii_b), hot_len(bn, short_b))
    ban = 300000
    print("balen", hot_len(ban, bytearray(ascii_b)), hot_len(ban, bytearray(short_b)))
    print("bamut", hot_mutating_len(120000, bytearray(short_b)))


main()
