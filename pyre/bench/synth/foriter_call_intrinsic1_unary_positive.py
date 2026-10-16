# CALL_INTRINSIC_1/UnaryPositive in a hot FOR_ITER body. This is one of the two
# variants with a codewriter residual; the other variants remain gate declines.
N = 7200000


def main():
    total = 0
    for i in range(N):
        total += +i
    print(total)


main()
# Expected: 25919996400000
