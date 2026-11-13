# pyre-check: max-pypy-ratio=3.2
# macOS dynasm reads 0.9x; the floor, a fourth of the ceiling, sits under it.
# Owner is just over INLINE_EAGER_MAX_BYTES, so the merge waits on
# 40*owner_bytes bridge entries. N runs past that trip.
N = 45454546


def main():
    # OSError / FileNotFoundError have a non-trivial constructor: it sets
    # errno / strerror / filename and removes filename from args. The
    # traced inline constructor only reproduces kind/w_class/args_w, so
    # these kinds must fall back to the full runtime constructor — this
    # bench locks in that the special fields survive under the JIT.
    errno_sum = 0
    fname_ok = 0
    arglen = 0
    i = 0
    while i < N:
        try:
            if (i & 1) == 0:
                raise OSError(2, "no such file", "/tmp/a")
            else:
                raise FileNotFoundError(2, "missing", "/tmp/b")
        except OSError as e:
            errno_sum = errno_sum + e.errno
            if e.filename is not None:
                fname_ok = fname_ok + 1
            arglen = arglen + len(e.args)
        i = i + 1
    print(errno_sum, fname_ok, arglen)


main()
