# pyre-check: max-pypy-ratio=37
# Local dynasm check.py reads 10.3-18.3x; 37 is twice the slowest.
# Exception-resume bridge (GuardException) with a CONSTANT pre-call operand.
# Each `try` divides/indexes with a CONSTANT numerator/addend that lives on the
# operand stack at the raising bytecode. The loop warms up exception-free
# (i < 2000, so the no-exception path compiles), then every later iteration
# raises. The compiled exception-resume bridge re-enters the interpreter at the
# raising bytecode, which must rematerialise that constant operand; if it were
# lost (NULL), the resume would run op(NULL, ...) and raise a spurious
# TypeError (bumping `other`) instead of the real ZeroDivisionError / IndexError.
# Output must stay byte-exact across backends and the oracle.
N = 1000000
DATA = [3, 1, 4, 1, 5]


def work_floatdiv(n):
    zdiv = 0
    other = 0
    i = 0
    while i < n:
        d = 1 if i < 2000 else i % 2  # 0 on even i after warm-up
        try:
            q = 3.5 / d          # constant float numerator on pre-call stack
            r = 10.0 / d         # second constant, deeper stack
        except ZeroDivisionError:
            zdiv += 1
        except TypeError:
            other += 1
        i += 1
    return zdiv, other


def work_intfloordiv(n):
    zdiv = 0
    other = 0
    i = 0
    while i < n:
        d = 1 if i < 2000 else i % 2
        try:
            q = 100 // d         # constant int numerator
        except ZeroDivisionError:
            zdiv += 1
        except TypeError:
            other += 1
        i += 1
    return zdiv, other


def work_constindex(n):
    idxe = 0
    other = 0
    i = 0
    while i < n:
        j = 0 if i < 2000 else (i % 9)   # > 4 -> IndexError after warm-up
        try:
            v = DATA[j] + 7.25           # constant 7.25 on pre-call stack
        except IndexError:
            idxe += 1
        except TypeError:
            other += 1
        i += 1
    return idxe, other


print(work_floatdiv(N))
print(work_intfloordiv(N))
print(work_constindex(N))
