# CPython-suite gap: no suite test counts an abort below an inlined operator dunder.
# parity-tests reason: an abort below an effectful `__init__` / `__len__` /
# `__bool__` / `__format__` must convert the continuation level in place;
# CALL replay runs the dunder twice. A post-return Len/Truth guard that
# later fails, and an invalid result seen while tracing, must continue in
# the operator tail rather than re-enter the dunder.
#
# operator_tail_fback_after_dunder:
# no suite test inspects f_back from __index__ after an inlined __len__ whose result later
# fails the operator-tail guard. a blackholed `__len__` that returns a non-int must run
# the Len tail's `space.index` with the dunder's caller as `f_back`, not the finished
# dunder frame. A mid-trace abort below the dunder must convert the continuation in place
# the same way `operator_tail_abort_below_dunder.py` does.
#
# operator_tail_nested_inline_caller:
# no suite test inlines a Python caller around an effectful operator dunder. a post-return
# operator-tail guard or abort under an already-inlined caller must keep the whole frame
# chain, and a collection inside the dunder must not leave the caller image stale.

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

def operator_tail_fback_after_dunder():
    """f_back after a blackholed operator dunder is the dunder's caller."""

    import sys

    N = 4000
    RARE = 997
    SWITCH = N // 2

    names = []

    class Idx:
        def __index__(self):
            fr = sys._getframe(0)
            assert fr.f_back is not None
            names.append(sys._getframe(1).f_code.co_name)
            return 1

    class Obj:
        calls = 0

        def __init__(self):
            self.n = 1
            self.fail = False

        def __len__(self):
            Obj.calls += 1
            if self.fail:
                raise ValueError("x")
            return self.n

    def driver():
        obj = Obj()
        raised = 0
        for i in range(N):
            obj.fail = i % RARE == RARE - 1
            if i == SWITCH:
                obj.n = Idx()
            try:
                len(obj)
            except ValueError:
                raised += 1
        return raised

    raised = driver()
    print(sorted(set(names)))
    print(Obj.calls, raised, len(names))
operator_tail_fback_after_dunder()

def operator_tail_nested_inline_caller():
    """Nested inlined caller around effectful operator dunders.

    A hot loop calls a Python function ``f`` so ``f`` is inlined. ``f`` then
    runs ``format``, ``len`` and ``bool`` on objects whose dunders are Python
    and have effects. After warmup the dunder result changes so the
    post-return guard fails and resumes in the operator tail still under
    ``f``; an invalid result raises in that tail and is caught outside ``f``.
    Each dunder allocates, so a minor collection can run inside the body.
    Effects run once per call.
    """

    N = 4000
    SWITCH = N // 2

    def churn():
        return [[i, i + 1] for i in range(64)]

    class Fmt:
        calls = 0

        def __init__(self):
            self.out = "s"

        def __format__(self, spec):
            Fmt.calls += 1
            churn()
            return self.out

    class LenObj:
        calls = 0

        def __init__(self):
            self.n = 1

        def __len__(self):
            LenObj.calls += 1
            churn()
            return self.n

    class TruthObj:
        calls = 0

        def __init__(self):
            self.out = True

        def __bool__(self):
            TruthObj.calls += 1
            churn()
            return self.out

    def f(fmt, ln, tr, spec):
        s = format(fmt, spec)
        n = len(ln)
        b = bool(tr)
        return s, n, b

    def run_phase(kind):
        Fmt.calls = 0
        LenObj.calls = 0
        TruthObj.calls = 0
        fmt = Fmt()
        ln = LenObj()
        tr = TruthObj()
        raised = 0
        last = None
        for i in range(N):
            if i == SWITCH:
                if kind == "fmt":
                    fmt.out = 42
                elif kind == "len":
                    ln.n = -1
                else:
                    tr.out = 1
            try:
                last = f(fmt, ln, tr, "s")
            except TypeError:
                raised += 1
            except ValueError:
                raised += 1
        return (
            Fmt.calls,
            LenObj.calls,
            TruthObj.calls,
            raised,
            last,
        )

    fmt_calls, fmt_len_calls, fmt_truth_calls, fmt_raised, fmt_last = run_phase("fmt")
    len_fmt_calls, len_calls, len_truth_calls, len_raised, len_last = run_phase("len")
    truth_fmt_calls, truth_len_calls, truth_calls, truth_raised, truth_last = run_phase(
        "truth"
    )

    print(fmt_calls, fmt_len_calls, fmt_truth_calls, fmt_raised, fmt_last)
    print(len_fmt_calls, len_calls, len_truth_calls, len_raised, len_last)
    print(truth_fmt_calls, truth_len_calls, truth_calls, truth_raised, truth_last)
operator_tail_nested_inline_caller()
print("OK")
