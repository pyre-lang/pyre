# pyre-check: spec-folds=binary_op_descent,compare_op_descent
# pyre-check: trace-shape=entry-bridge:rec_plain:absent=CallMayForceR,absent=CallMayForceI,entry-bridge:rec_memo:present=GuardNonnullClass
# These three carry most of the corpus's fold traffic, and nothing
# declared any of them, so switching one off was a silent change. This fixture
# fires all three repeatedly, the widest margin of any fixture doing so.
# `compare_op_int` used to fire 61 times here, on the int compares whose
# descent declined at a helper call inside the compare helper's body; that
# call now resumes at the enclosing helper, the descent takes those sites, and
# the hand fold has been removed.
# Memoized vs plain recursion with post-warm-up branch divergence. The
# memo-dict store (memo[n] = r) once died with a TypeError after warm-up
# (an empty-string type name from a clobbered class read on the dict-store
# path during a deopt-resumed recursive frame). Deterministic; divergence
# (deeper args, branch flip) starts after the loop is compiled.
MOD = 1000003

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

memo = {}


def rec_memo(n):
    if n <= 1:
        return n
    if n in memo:
        return memo[n]
    if n % 2 == 0:
        r = (rec_memo(n // 2) * 3 + 7) % MOD
    else:
        r = (rec_memo(n - 1) + n * 5) % MOD
    memo[n] = r
    return r


def rec_plain(n):
    if n <= 1:
        return n
    if n % 2 == 0:
        return (rec_plain(n // 2) * 3 + 7) % MOD
    return (rec_plain(n - 1) + n * 5) % MOD


def main():
    acc = 0
    for i in range(1, 5001):
        n = (i * 37) % 211 + 2
        if i > 3500:
            n = n * 31 + 1
            acc = (acc + rec_plain(n) * 2 + rec_memo(n)) % MOD
        else:
            acc = (acc + rec_memo(n) + rec_plain(n)) % MOD
        if i % 1000 == 0:
            print("checksum1", i, acc)
    print("final1", acc, len(memo))


main()
