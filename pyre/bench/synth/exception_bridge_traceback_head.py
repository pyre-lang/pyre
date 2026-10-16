# Two exception classes in one hot try force an exception-edge bridge for the
# non-recorded class. Reading the traceback head in each handler pins that the
# compiled catching frame contributes its node before handler entry.
#
# Expected output: [('T', 'outer'), ('V', 'outer')]
N = 240000


def leaf(i):
    if i % 3 == 1:
        raise ValueError(i)
    if i % 3 == 2:
        raise TypeError(i)
    return i


def mid(i):
    return leaf(i)


def outer():
    shapes = set()
    acc = 0
    for i in range(N):
        try:
            acc += mid(i)
        except ValueError as e:
            shapes.add(("V", e.__traceback__.tb_frame.f_code.co_name))
        except TypeError as e:
            shapes.add(("T", e.__traceback__.tb_frame.f_code.co_name))
    return sorted(shapes)


print(outer())
