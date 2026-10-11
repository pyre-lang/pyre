# pyre-check: gate=1
# CALL_FUNCTION_EX of a builtin __new__ over a (cls, int) star tuple built
# in the same trace. The builtin must receive the tuple's own elements: a
# compiled loop once handed int.__new__ a null element, reported as
# "TypeError: int() argument ... not 'object'" from an IntFlag class body
# whose members are auto().
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

from enum import IntFlag, auto


def new_from_star(args):
    return int.__new__(*args)


def make_flag():
    class Color(IntFlag):
        RED = auto()
        GREEN = auto()
        BLUE = auto()

    return Color


def main():
    total = 0
    for i in range(40):
        total += new_from_star((int, i))
    assert total == sum(range(40)), total

    for _ in range(40):
        color = make_flag()
        assert int(color.RED) == 1
        assert int(color.GREEN) == 2
        assert int(color.BLUE) == 4
    print("PASS")


if __name__ == "__main__":
    main()
