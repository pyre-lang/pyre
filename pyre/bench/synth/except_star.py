# No `max-pypy-ratio`: wall-clock is startup-dominated next to pypy. The
# jitstats baselines gate it (CALL_ASSEMBLER compiles the except* loop;
# pypy jit-summary reports 16 loops / 23 bridges).
def m(label, value):
    print(label, "->", repr(value))


def leaf_names(exc):
    if isinstance(exc, BaseExceptionGroup):
        result = []
        for child in exc.exceptions:
            result.extend(leaf_names(child))
        return result
    return [type(exc).__name__]


def exercise():
    caught = []
    try:
        raise ExceptionGroup("warmup", [ValueError(1), TypeError(2)])
    except* ValueError as group:
        caught.extend(type(exc).__name__ for exc in group.exceptions)
    except* TypeError as group:
        caught.extend(type(exc).__name__ for exc in group.exceptions)
    return caught


for _ in range(12904):
    exercise()

m("matches", exercise())

try:
    raise ValueError(1)
except* ValueError as group:
    m("naked", (type(group).__name__, [type(exc).__name__ for exc in group.exceptions]))

try:
    try:
        raise ExceptionGroup("g", [ValueError(1), TypeError(2)])
    except* ValueError:
        pass
except* TypeError as group:
    m("rest", [type(exc).__name__ for exc in group.exceptions])

try:
    try:
        raise ExceptionGroup("g", [ValueError(1), TypeError(2), KeyError(3)])
    except* ValueError:
        raise
    except* TypeError:
        pass
except ExceptionGroup as escaped:
    m("reraised", sorted(leaf_names(escaped)))

try:
    try:
        raise ExceptionGroup("g", [ValueError(1)])
    except* ExceptionGroup:
        pass
except TypeError as exc:
    m("group target", str(exc))
