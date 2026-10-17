# pyre-check: max-pypy-ratio=8
# Readings 2.2x-4.0x local (dynasm 2.2x-2.3x, cranelift 3.6x-4.0x); wasm reads
# ~2.2x dynasm on CI. The floor, a fourth of the ceiling, sits under the fastest.
# Deep self-recursion whose caller frame is relocated by a minor collection
# while the recursive callee runs.  The bytecode CALL fast path drops the
# arguments, runs the callee, then pushes its result onto the caller's value
# stack.  The callee allocates enough to trigger a minor collection that moves
# the caller frame, so the raw pointer captured before the call goes stale; the
# result push and its valuestackdepth bump must land on the forwarded live
# frame.  When they hit the abandoned copy the live frame keeps a stack depth
# one slot short, the following BINARY_OP reads the range iterator instead of
# the recursion result ("unsupported operand type(s) for +: 'range_iterator'
# and 'int'"), and the dropped exception segfaults.  cat(5) sums to 5! per
# call; N outer iterations warm the JIT and sustain the allocation pressure
# that forces the relocation.  Sized so pypy's own execution clears the
# measurement floor: below it the ratio gate divides by the floor and reads
# startup rather than this loop.  Branchy recursion is still the architectural
# JIT gap, and what this fixture is really for is the correctness/crash guard.
# Expected: N * 120
#
# Once `for k in range(n)` folds on a live bound, `cat`'s inner loop gets a
# procedure token and the recursive call records CALL_ASSEMBLER.  While the
# real loop is still compiling, PyPy installs a compiled temporary callback;
# attaching the real procedure redirects that token and grows its frame info
# if necessary.  This fixture therefore also guards that bootstrap contract:
# a bodyless pending-token shortcut makes wasm reject the loop repeatedly and
# eventually consumes the key's abort budget.
N = 40000


def cat(n):
    if n <= 1:
        return 1
    r = 0
    for k in range(n):
        r += cat(n - 1)
    return r


total = 0
for i in range(N):
    total += cat(5)
print(total)
