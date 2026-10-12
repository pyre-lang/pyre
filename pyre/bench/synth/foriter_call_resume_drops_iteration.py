# pyre-check: max-pypy-ratio=20
# A `for` body that calls two DISTINCT recursive functions and then runs one
# more statement, with the branch it takes flipping after warm-up. Once a callee
# owns a compiled loop the walk aborts the enclosing trace
# (`LoopBearingCalleeInlineUnsupported`) and resumes the caller forward at the
# CALL. That resume rebuilds the caller's operand stack from the overrides
# collected at the CALL, and the `null_or_self` slot a `PUSH_NULL` leaves under
# it has no box in the walk and no distinguishable shadow value.  When that
# slot goes missing the resume declines, and the legacy rollback+replay
# refuses in-flight FOR_ITER delivery, DROPPING that iteration. The loss is silent: one
# iteration's accumulation never happens and the answer is simply too small.
#
# All three ingredients are required — with one callee, without the branch
# divergence, or without the trailing statement the shape does not abort here.
MOD = 1000003


def ra(n):
    if n <= 1:
        return n
    if n % 2 == 0:
        return (ra(n // 2) + 1) % MOD
    return (ra(n - 1) + 2) % MOD


def rb(n):
    if n <= 1:
        return n
    if n % 2 == 0:
        return (rb(n // 2) + 3) % MOD
    return (rb(n - 1) + 4) % MOD


def main():
    acc = 0
    seen = 0
    for i in range(1, 9001):
        n = (i * 37) % 211 + 2
        if i > 7000:
            n = n * 31 + 1
            acc = (acc + ra(n) * 2 + rb(n)) % MOD
        else:
            acc = (acc + rb(n) + ra(n)) % MOD
        seen += 1
    print(acc, seen)


main()
