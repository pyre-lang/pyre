# pyre-check: max-pypy-ratio=60
MODULUS = 1_000_003


def maybe_raise(step):
    if step >= 3:
        raise StopIteration(step * 7)
    return step + 1


def run():
    total = 0
    for _ in range(348_496):
        value = 0
        step = 0
        while step < 10:
            try:
                value = maybe_raise(step)
            except StopIteration as exc:
                value = (value + (exc.value or 0)) % MODULUS
                break
            total = (total + (value or 0)) % MODULUS
            step += 1
        total = (total + (value or 0)) % MODULUS
    return total


print(run())
