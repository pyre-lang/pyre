# CALL_KW whose keywords fail to bind. `_match_signature` returns
# `MatchOutcome` as a value (`argument.py`); `parse_into_scope` raises
# TypeError and never enters the callee. A hot loop of those failures
# must still raise, including when recording saw a successful bind and a
# later iteration of the same call site does not.
N = 2000


def f(a):
    return a


def g(a, b):
    return a + b


def h(a):
    return a


def accept(a, zz=0):
    return a + zz


def reject(a):
    return a


def main():
    unexpected = []
    missing = []
    duplicate = []
    mixed = []
    mixed_sum = 0
    for i in range(N):
        try:
            f(a=1, zz=2)
        except TypeError as e:
            unexpected.append(str(e))
    for i in range(N):
        try:
            g(b=1)
        except TypeError as e:
            missing.append(str(e))
    for i in range(N):
        try:
            h(1, a=2)
        except TypeError as e:
            duplicate.append(str(e))
    # Same CALL_KW spelling; `accept` binds `zz` and `reject` does not.
    # Recording is on the common `accept` path; every 7th call is `reject`.
    fns = (accept, accept, accept, accept, accept, accept, reject)
    for i in range(N):
        try:
            mixed_sum += fns[i % 7](a=i, zz=1)
        except TypeError as e:
            mixed.append(str(e))
    print(len(unexpected), len(set(unexpected)))
    print(len(missing), len(set(missing)))
    print(len(duplicate), len(set(duplicate)))
    print(len(mixed), mixed_sum, len(set(mixed)))
    for msg in sorted(set(unexpected + missing + duplicate + mixed)):
        print(msg)


main()
