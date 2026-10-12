# CPython-suite gap: no suite test counts a later inline abort after a declined list.append fold.
# parity-tests reason: a list-append rollback must drop its abort-coordinate claim
# with the discarded image; a leftover claim makes a later effectful inline abort
# fall back to CALL replay and run the effect twice.

"""A later inline abort after a declined list.append fold does not re-run the effect."""

try:
    import pypyjit
except ImportError:
    pypyjit = None

if pypyjit is not None:
    pypyjit.set_param("threshold=1,function_threshold=1")

N = 4000
RARE = 997
calls = [0]


def bump(fail):
    calls[0] += 1
    if fail:
        raise ValueError("x")
    return 1


raised = 0
xs = []
for i in range(N):
    # Integer-strategy appends first, then a user object that forces
    # `switch_to_object_strategy`. That switch is a nested helper sub-walk;
    # a decline there rolls the append attempt back and must also release
    # the abort-coordinate claim so `bump` can latch its own abort.
    if i < 80:
        xs.append(i)
    else:
        xs.append(object())
    try:
        bump(i % RARE == RARE - 1)
    except ValueError:
        raised += 1

assert calls[0] == N, (calls[0], N)
assert raised == N // RARE, (raised, N // RARE)
assert len(xs) == N, len(xs)

print("OK")
