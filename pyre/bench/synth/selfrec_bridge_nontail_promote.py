# pyre-check: max-pypy-ratio=3.6
# The measured reading stays under 4, so the ceiling is 4.
# Coverage for the self-recursive root-bridge inline when the recursion is
# non-tail and carries a Ref local.
#
# `walk` is self-recursive, takes exact-integer arguments and holds a
# `BinaryOp` residual. A guard-failure bridge that reaches its CALL inlines
# through `_opimpl_recursive_call` / `perform_call`.
# `bridge_recursion_overflow` already covers that admission, but only in its
# easiest form: tail recursion whose live set is two machine integers. Two
# ingredients of the "a Ref reached an int operation" failure it is meant to
# guard against were therefore unexercised.
#
# `acc * 2` crosses the machine-int boundary partway down the recursion, so the
# accumulator promotes to a long — a Ref — at a level that moves with the
# caller's seed, and the overflow guard fires inside the recursive frame. `tag`
# keeps a second Ref live across the recursive CALL beside it, and the non-tail
# `inner + len(tag)` leaves a paused caller chain, so the guard's resume stream
# is multi-frame and mixes Ref with Int rather than being one frame of
# integers.
#
# Byte-parity against CPython/PyPy is the gate: Python integers are unbounded,
# so the promotion must not be observable in the result.
_TAGS = ("a", "bb", "ccc", "dddd")


def walk(n, acc):
    if n == 0:
        return acc
    tag = _TAGS[n & 3]
    nxt = acc * 2 + len(tag)
    inner = walk(n - 1, nxt)
    return inner + len(tag)


out = []
for i in range(322581):
    out.append(walk(24, (1 << 50) + i) % 1000000007)
print(out[0], out[-1], sum(out) % 1000000007)
