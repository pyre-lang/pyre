# CALL_FUNCTION_EX in a hot FOR_ITER body. The starred call is the same
# MayForce boundary as CALL and CALL_KW for the whole-frame safety gate.
N = 3600000


def add(a, b):
    return a + b


def main():
    total = 0
    for i in range(N):
        args = (i, 1)
        total += add(*args)
    print(total)


main()
# Expected: 6480001800000
