# `optional_none_arg_inline` with the default changed from `None` to an int, so
# the local the `is`-against-None branch tests is the one the callee's register
# banks hold unboxed.  The callee inlines all the same: a mid-body guard encodes
# its frame from the liveness at the carried jitcode pc, so the operand goes
# out in the bank it lives in.  `_read_from_buffer(self, size=-1)` is the
# standard-library shape.
#
# The gate is `loops_compiled` reading 1, as `pypy3` reports -- the callee
# folds into the caller's trace.  A count of 2 means it was declined and kept a
# trace of its own.  The printed total covers both arms of the test.
N = 200000


def read_n(buf, size=-1):
    if size is None or size < 0:
        return buf
    return size


def main():
    total = 0
    for i in range(N):
        total += read_n(i)
    total += read_n(1, 7)
    print(total)


main()
