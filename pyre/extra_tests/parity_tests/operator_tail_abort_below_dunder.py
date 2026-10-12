# CPython-suite gap: no suite test counts an abort below an inlined operator dunder.
# parity-tests reason: an abort below an effectful `__init__` / `__len__` /
# `__bool__` / `__format__` must convert the continuation level in place;
# CALL replay runs the dunder twice. A post-return Len/Truth guard that
# later fails, and an invalid result seen while tracing, must continue in
# the operator tail rather than re-enter the dunder.

"""An abort below an effectful operator dunder does not re-run the effect."""

try:
    import pypyjit
except ImportError:
    pypyjit = None

if pypyjit is not None:
    pypyjit.set_param("threshold=1,function_threshold=1")

N = 4000
RARE = 997
SWITCH = N // 2


class Ctor:
    calls = 0

    def __init__(self, fail):
        Ctor.calls += 1
        if fail:
            raise ValueError("x")


class Len:
    calls = 0

    def __init__(self, fail):
        self.fail = fail

    def __len__(self):
        Len.calls += 1
        if self.fail:
            raise ValueError("x")
        return 1


class Truth:
    calls = 0

    def __init__(self, fail):
        self.fail = fail

    def __bool__(self):
        Truth.calls += 1
        if self.fail:
            raise ValueError("x")
        return True


class Format:
    calls = 0

    def __init__(self, fail):
        self.fail = fail

    def __format__(self, spec):
        Format.calls += 1
        if self.fail:
            raise ValueError("x")
        return "s"


def run(kind):
    kind.calls = 0
    raised = 0
    for i in range(N):
        fail = i % RARE == RARE - 1
        try:
            if kind is Ctor:
                Ctor(fail)
            elif kind is Len:
                len(Len(fail))
            elif kind is Truth:
                bool(Truth(fail))
            else:
                format(Format(fail), "s")
        except ValueError:
            raised += 1
    assert kind.calls == N, (kind.__name__, kind.calls, N)
    assert raised == N // RARE, (kind.__name__, raised, N // RARE)


run(Ctor)
run(Len)
run(Truth)
run(Format)


class LenNeg:
    calls = 0

    def __init__(self):
        self.n = 1

    def __len__(self):
        LenNeg.calls += 1
        return self.n


class LenNonInt:
    calls = 0

    def __init__(self):
        self.n = 1

    def __len__(self):
        LenNonInt.calls += 1
        return self.n


class TruthNonBool:
    calls = 0

    def __init__(self):
        self.out = True

    def __bool__(self):
        TruthNonBool.calls += 1
        return self.out


def run_compiled_invalid(kind, invalidate, expect_exc):
    kind.calls = 0
    obj = kind()
    raised = 0
    for i in range(N):
        if i == SWITCH:
            invalidate(obj)
        try:
            if kind is TruthNonBool:
                if obj:
                    pass
            else:
                len(obj)
        except expect_exc:
            raised += 1
    assert kind.calls == N, (kind.__name__, "compiled-invalid", kind.calls, N)
    assert raised == N - SWITCH, (kind.__name__, "compiled-invalid", raised, N - SWITCH)


run_compiled_invalid(LenNeg, lambda obj: setattr(obj, "n", -1), ValueError)
run_compiled_invalid(LenNonInt, lambda obj: setattr(obj, "n", object()), TypeError)
run_compiled_invalid(TruthNonBool, lambda obj: setattr(obj, "out", 1), TypeError)


class LenTracedNeg:
    calls = 0

    def __len__(self):
        LenTracedNeg.calls += 1
        return -1


class LenTracedNonInt:
    calls = 0

    def __len__(self):
        LenTracedNonInt.calls += 1
        return object()


class TruthTracedNonBool:
    calls = 0

    def __bool__(self):
        TruthTracedNonBool.calls += 1
        return 1


def run_traced_invalid(kind, consume, expect_exc):
    kind.calls = 0
    obj = kind()
    raised = 0
    for i in range(N):
        try:
            consume(obj)
        except expect_exc:
            raised += 1
    assert kind.calls == N, (kind.__name__, "traced-invalid", kind.calls, N)
    assert raised == N, (kind.__name__, "traced-invalid", raised, N)


def _if_obj(obj):
    if obj:
        return 1
    return 0


run_traced_invalid(LenTracedNeg, len, ValueError)
run_traced_invalid(LenTracedNonInt, len, TypeError)
run_traced_invalid(TruthTracedNonBool, _if_obj, TypeError)

print("OK")
