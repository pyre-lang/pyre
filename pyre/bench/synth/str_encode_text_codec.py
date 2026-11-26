# pyre-check: max-pypy-ratio=27
# pyre-check: requires-modules=binascii,zlib
# str.encode() rejects non-text (binary) codecs with LookupError instead of
# silently producing bytes: the binary codecs' CodecInfo._is_text_encoding
# (False, stored on a tuple subclass instance) is honored. Output verified
# against CPython/PyPy.
N = 20000


def kind(codec):
    try:
        "hello".encode(codec)
    except LookupError:
        return "L"
    except Exception:
        return "E"
    return "ok"


def main():
    n = 0
    for _ in range(N):
        r = "".join(kind(c) for c in ("hex", "rot13", "base64", "zlib"))
        if r == "LLLL":
            n += 1
    print(n)


main()
