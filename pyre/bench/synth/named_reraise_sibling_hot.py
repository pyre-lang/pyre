# pyre-check: max-pypy-ratio=25
# A named `except E as m:` handler whose body bare-re-raises compiles its
# implicit cleanup to `DELETE_FAST m; RERAISE`, immediately followed by a
# sibling `except` clause's type check. On the re-raise path the exception must
# route to the outer re-raise landing and the sibling clause must NOT run; the
# binding `m` is cleared. A `del` of a bound local inside the try body plus a
# sibling clause exercises the same walker-native DELETE_FAST bound
# continuation. Output is verified against CPython/PyPy.
N = 810000


def reraise_sibling(i):
    try:
        if i % 9 == 0:
            raise ValueError
        return 1
    except ValueError as m:
        raise
    except KeyError:
        return 100


def dual_reraise(i):
    try:
        if i % 9 == 0:
            raise ValueError
        if i % 7 == 1:
            raise TypeError
        return 1
    except ValueError as v:
        raise
    except TypeError:
        raise


def del_in_try(i):
    x = i
    try:
        if i % 15 == 0:
            raise ValueError
        del x
        return 1
    except ValueError as m:
        raise
    except IndexError:
        return 50


def run():
    total = 0
    for i in range(N):
        try:
            total += reraise_sibling(i)
        except ValueError:
            total += 2
        try:
            total += dual_reraise(i)
        except ValueError:
            total += 3
        except TypeError:
            total += 5
        try:
            total += del_in_try(i)
        except ValueError:
            total += 7
    return total


print(run())
