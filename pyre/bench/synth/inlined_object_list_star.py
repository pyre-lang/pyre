# An inlined BUILD_LIST must retain the concrete items-array allocation as
# well as the wrapper. LIST_EXTEND otherwise leaves a length-only append
# behind when a helper descent declines, inserting a raw NULL into *args.


def make_list(prefix, pair):
    return [prefix, *pair]


def invoke(callback, prefix, pair):
    return callback(prefix, *pair)


def callback(first, left, right):
    assert first is prefix
    return left + right


prefix = object()
pair = (12, 13)
total = 0
for i in range(4000):
    actual = make_list(prefix, pair)
    assert len(actual) == 3, (i, actual)
    assert actual[0] is prefix
    assert actual[1] == 12, (i, actual)
    assert actual[2] == 13, (i, actual)
    total += invoke(callback, prefix, pair)
print(total)
