# pyre-check: max-pypy-ratio=5.6
# ubuntu dynasm 4.9x at pypy exec 0.10s (wall 0.50s). 5.6 is 15% over that
# reading. The rename aborts tracing (abrt_force_qmut), as PyPy's does, so
# both run interpreted.
N = 80000

# Exercises the writable type.__name__ setter under the JIT: a successful
# rename, the non-heap / non-str / null-char rejections, and an astral
# (4-byte non-surrogate) name.  The lone-surrogate rejection is covered by
# synth/type_name_surrogate_reject (the shared _check_surrogate path); it is
# left out here because raising UnicodeEncodeError out of a JIT-compiled
# function frame trips an unrelated resume bug.
class C:
    pass


def main():
    total = 0
    i = 0
    astral = 'Z' + chr(0x1f600)
    while i < N:
        acc = 0
        # heap type rename works
        C.__name__ = 'R1'
        if C.__name__ == 'R1':
            acc = acc + 1
        # non-heap type rejected (message text differs across runtimes;
        # only the exception type is runtime-agnostic)
        try:
            int.__name__ = 'x'
        except TypeError:
            acc = acc + 1
        except Exception:
            pass
        # non-str value rejected
        try:
            C.__name__ = 123
        except TypeError:
            acc = acc + 1
        except Exception:
            pass
        # embedded null rejected
        try:
            C.__name__ = 'a\x00b'
        except ValueError:
            acc = acc + 1
        except Exception:
            pass
        # astral (4-byte non-surrogate) name accepted
        C.__name__ = astral
        if C.__name__ == astral:
            acc = acc + 1
        total = total + acc
        i = i + 1

    # rename is visible through an instance and leaves __qualname__ untouched
    q0 = C.__qualname__
    C.__name__ = 'Final'
    if C().__class__.__name__ == 'Final' and C.__qualname__ == q0:
        total = total + 1
    print(total)


main()
