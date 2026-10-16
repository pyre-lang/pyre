# pyre-check: max-pypy-ratio=17
# Implicit `__context__` chaining: raising inside an `except` block links the
# new exception's `__context__` to the one being handled (the raise records
# `SETFIELD w_context = ec.sys_exc_value`). The outer handler reads
# `e.__context__` and checks its type, so a wrong/absent context chain (or a
# context slot that aliased the wrong exception) changes the output.
#   inner raise ValueError -> caught -> raise KeyError (chained)
#   outer catches KeyError, e.__context__ must be the ValueError instance
# Per-iteration contribution is 1 when the chain is correct; a mismatch jumps
# by 1000, so any regression is loud. Deterministic.
N = 720000


def run(n):
    acc = 0
    i = 0
    while i < n:
        try:
            try:
                raise ValueError("a")
            except ValueError:
                raise KeyError("b")
        except KeyError as e:
            ctx = e.__context__
            if type(ctx) is ValueError and ctx.args == ("a",):
                acc += 1
            else:
                acc += 1000
        i += 1
    return acc


def main():
    print(run(N))


main()
