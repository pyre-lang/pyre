# pyre-check: max-pypy-ratio=17
# A named `except X as m:` handler that raises a SECOND exception of a
# different class (conditionally, alongside a bare reraise) while the function
# also has a sibling `except` clause, and a hot `return` path. The func-entry
# JIT trace forms on the return path; the cold raise deopts through the
# blackhole, which must propagate the handler-raised exception out of the
# function (a sibling clause does NOT catch a handler-raised exception) so the
# caller routes it to the correct `except`. Output verified against CPython/PyPy.
N = 2700000


def f(i):
    try:
        if i % 9 == 0:
            raise ValueError(i)
        return 1
    except ValueError as m:
        if m.args[0] % 2 == 0:
            raise
        raise KeyError
    except KeyError:
        return 100


def run(n):
    ve = ke = other = 0
    for i in range(n):
        try:
            other += f(i)
        except ValueError:
            ve += 1
        except KeyError:
            ke += 1
    return ve, ke, other


print(run(N))
