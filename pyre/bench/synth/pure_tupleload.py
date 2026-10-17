# pyre-check: max-pypy-ratio=2.9
# #171/#11 Approach C, SUBSCRIPT slice: canonical-tuple `t[i]` emits a PURE
# getarrayitem in the JIT walker (OptPure CSEs / const-folds the element load).
# Nested tracing compiles the getitem helpers, so dynasm sits at pypy parity
# (~1.0x). A ceiling of 4 arms a 1x floor (`perf_gate_floor`) that fails that
# knife-edge. 2.9 derives no floor (under 3) and still covers cranelift ~1.0x /
# wasm ~1.7x. With the arity-2 reader off, the loop pays the opaque residual
# (about 198x when the retired `subscr_specialised_pair` fold was the only
# reader).
#
# Case A exercises the canonical array-backed `W_TupleObject` (arity > 2) on the
# hot path — the pure element load is the point of this slice.
#
# Case B reads a 2-element literal tuple in a loop. A 2-int literal tuple may be
# stored as SPECIALISED_TUPLE_II (inline value0/value1, NO wrappeditems block);
# the trace-time `ob_type == &TUPLE_TYPE` gate declines it to the non-pure /
# residual path. It MUST NOT SIGSEGV and MUST stay correct.
#
# Case C builds a fresh arity-2 pair per iteration and reads both halves. This
# is the shape `subscr_tuple_descent` serves by descending `w_tuple_getitem`;
# the hand-written `subscr_specialised_pair` reader it replaced measured 46.3x
# here when suppressed, the largest single-fold effect in the corpus, so this
# loop is what keeps the ceiling below honest.


def main():
    # Case A: canonical array-backed tuple (arity 5 > 2).
    # Iteration counts are sized so pypy's own execution clears Windows
    # `FLOOR_GATE_MIN_BASELINE_S` (~0.16s).  Below that the same binary
    # read 1.6x-9.5x in the `?` band and crossed the ceiling.
    t = (10, 20, 30, 40, 50)
    s = 0
    for _ in range(4300000):
        s += t[0] + t[1] + t[2] + t[3] + t[4]
    print(s)

    # Case B: 2-element tuple (specialised-tuple path must be safe + correct).
    t2 = (1, 2)
    s2 = 0
    for _ in range(4300000):
        s2 += t2[0] + t2[1]
    print(s2)


main()


def hot_specialised_pair(n):
    """Case C: a fresh 2-tuple per iteration, both halves read."""
    s = 0
    i = 0
    while i < n:
        p = (i, i + 1)
        s += p[0] + p[1]
        i += 1
    return s


print(hot_specialised_pair(430000000))
