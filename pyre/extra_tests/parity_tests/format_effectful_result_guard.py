# CPython-suite gap: no suite test counts an effectful __format__ whose
# answer later becomes a non-str in a loop hot enough to compile.
# parity-tests reason: an inlined `__format__` that mutates must still
# raise TypeError when a later iteration returns a non-str, and must not
# re-run the dunder on that deopt.

"""An effectful inlined __format__ still checks that the result is a str."""

try:
    import pypyjit
except ImportError:
    pypyjit = None

if pypyjit is not None:
    pypyjit.set_param("threshold=1,function_threshold=1")

N = 30000
SWITCH = N // 2


class C:
    calls = 0

    def __format__(self, spec):
        C.calls += 1
        return self.out


obj = C()
obj.out = "s"
raised = 0
for i in range(N):
    if i == SWITCH:
        obj.out = 42
    try:
        f"{obj:s}"
    except TypeError:
        raised += 1

assert C.calls == N, (C.calls, N)
assert raised == N - SWITCH, (raised, N - SWITCH)


class D:
    calls = 0

    def __format__(self, spec):
        D.calls += 1
        return 42


traced = D()
traced_raised = 0
for i in range(N):
    try:
        f"{traced:s}"
    except TypeError:
        traced_raised += 1

assert D.calls == N, (D.calls, N)
assert traced_raised == N, (traced_raised, N)

print("OK")

