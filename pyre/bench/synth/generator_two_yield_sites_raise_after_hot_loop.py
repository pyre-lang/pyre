# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=consume
# A generator with two yield sites in an if/else that raises after a hot
# `for` loop has consumed it must still propagate the exception.
# `should_not_inline` sends the resume through `generatorentry_driver`;
# `_invoke_execute_frame` must raise rather than treating the finished
# frame as a clean RETURN (`genentry_raise_returned` / StopIteration).


def gen(data, flag):
    i = 0
    while True:
        if i == len(data):
            raise ValueError("exhausted")
        code = data[i]
        i += 1
        if flag:
            yield code, i
        else:
            yield code
        if code == 0:
            break


def consume(data):
    n = 0
    for x in gen(data, False):
        n += 1
    return n


def run():
    for _ in range(3000):
        consume([1, 1, 0])
    try:
        n = consume([1])
    except ValueError as e:
        assert str(e) == "exhausted", e
        print("PASS generator two yield sites raise after hot loop")
        return
    raise SystemExit(f"BUG: no error {n}")


run()
