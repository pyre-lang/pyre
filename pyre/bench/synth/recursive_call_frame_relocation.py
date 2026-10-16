# pyre-check: max-pypy-ratio=96
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
# call; 1000 outer iterations warm the JIT and sustain the allocation pressure
# that forces the relocation.  The ratio ceiling above is deliberately loose --
# branchy recursion is the architectural JIT gap -- and what this fixture is
# really for is the correctness/crash guard.
#
# Once `for k in range(n)` folds on a live bound, `cat`'s inner loop gets a
# procedure token and the recursive call records CALL_ASSEMBLER.  While the
# real loop is still compiling, PyPy installs a compiled temporary callback;
# attaching the real procedure redirects that token and grows its frame info
# if necessary.  This fixture therefore also guards that bootstrap contract:
# a bodyless pending-token shortcut makes wasm reject the loop repeatedly and
# eventually consumes the key's abort budget.
def cat(n):
    if n <= 1:
        return 1
    r = 0
    for k in range(n):
        r += cat(n - 1)
    return r


total = 0
for i in range(4000):
    total += cat(5)
print(total)
