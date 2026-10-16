# Two-argument BUILD_SLICE must pass the Python None singleton as its implicit
# step.  Negative bounds make a malformed null step observable during slice
# normalization, including on guard-failure blackhole resume.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 400000


def delete_neg3_open():
    items = []
    checksum = 0
    for i in range(N):
        if len(items) > 11:
            del items[-3:]
        else:
            items.append(i % 31)
        checksum += len(items)
    return checksum * 100 + len(items)


def delete_neg1_open():
    items = []
    checksum = 0
    for i in range(N):
        if len(items) > 11:
            del items[-1:]
        else:
            items.append(i % 29)
        checksum += len(items)
    return checksum * 100 + len(items)


def delete_open_neg2():
    items = []
    checksum = 0
    for i in range(N):
        if len(items) > 11:
            del items[:-2]
        else:
            items.append(i % 23)
        checksum += len(items)
    return checksum * 100 + len(items)


def delete_neg2_neg1():
    items = []
    checksum = 0
    for i in range(N):
        if len(items) > 11:
            del items[-2:-1]
        else:
            items.append(i % 19)
        checksum += len(items)
    return checksum * 100 + len(items)


print(
    delete_neg3_open()
    + delete_neg1_open()
    + delete_open_neg2()
    + delete_neg2_neg1()
)
