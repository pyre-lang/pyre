# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot_instance,hot_returns_int,hot_instance_in_loop,hot_returns_int_in_loop,entry-bridge:__init__,loop:__init__
# A loop-bearing `__init__` cut into CALL_ASSEMBLER that forces the caller
# while that residual is still recording must resume in descr_call's tail:
# `C()` is the instance, and a non-None `__init__` return raises TypeError.
# `W_TypeObject.descr_call` stays on the framestack during `get_and_call_args`,
# checks the result, and returns `w_newobject`. The four recording-time force
# shapes (instance / return 1, after the loop / inside it) are combined here.
import sys

try:
    import pypyjit

    pypyjit.set_param("threshold=100,function_threshold=100")
except ImportError:
    pass

WARM = 400
N = 4000
force = [0]


def force_caller():
    frame = sys._getframe(1)
    while frame is not None and not frame.f_code.co_name.startswith("hot"):
        frame = frame.f_back
    if frame is not None:
        frame.f_locals


class InstanceAfter:
    def __init__(self):
        i = 0
        while i < 8:
            i += 1
        if force[0]:
            force_caller()


class ReturnsIntAfter:
    def __init__(self):
        i = 0
        while i < 8:
            i += 1
        if force[0]:
            force_caller()
            return 1


class InstanceInLoop:
    def __init__(self):
        i = 0
        while i < 8:
            i += 1
            if force[0] and i == 4:
                force_caller()


class ReturnsIntInLoop:
    def __init__(self):
        i = 0
        while i < 8:
            i += 1
            if force[0] and i == 4:
                force_caller()
        if force[0]:
            return 1


def hot_instance(n):
    none = 0
    ok = 0
    other = 0
    for _ in range(n):
        obj = InstanceAfter()
        if obj is None:
            none += 1
        elif type(obj) is InstanceAfter:
            ok += 1
        else:
            other += 1
    return ok, none, other


def hot_returns_int(n):
    raised = 0
    no_raise = 0
    for _ in range(n):
        try:
            ReturnsIntAfter()
            no_raise += 1
        except TypeError:
            raised += 1
    return raised, no_raise


def hot_instance_in_loop(n):
    none = 0
    ok = 0
    other = 0
    for _ in range(n):
        obj = InstanceInLoop()
        if obj is None:
            none += 1
        elif type(obj) is InstanceInLoop:
            ok += 1
        else:
            other += 1
    return ok, none, other


def hot_returns_int_in_loop(n):
    raised = 0
    no_raise = 0
    for _ in range(n):
        try:
            ReturnsIntInLoop()
            no_raise += 1
        except TypeError:
            raised += 1
    return raised, no_raise


def warmup_instance(cls):
    for _ in range(WARM):
        if cls() is None:
            return False
    return True


def warmup_raises(cls):
    for _ in range(WARM):
        obj = cls()
        if obj is None or type(obj) is not cls:
            return False
    return True


def main():
    force[0] = 0
    if not warmup_instance(InstanceAfter):
        print("FAIL warmup InstanceAfter() was None")
        return 1
    if not warmup_raises(ReturnsIntAfter):
        print("FAIL warmup ReturnsIntAfter() was not an instance")
        return 1
    if not warmup_instance(InstanceInLoop):
        print("FAIL warmup InstanceInLoop() was None")
        return 1
    if not warmup_raises(ReturnsIntInLoop):
        print("FAIL warmup ReturnsIntInLoop() was not an instance")
        return 1

    force[0] = 1
    ok, none, other = hot_instance(N)
    if none or other or ok != N:
        print(
            "FAIL instance after loop: ok=%d none=%d other=%d (owed %d instances)"
            % (ok, none, other, N)
        )
        return 1
    raised, no_raise = hot_returns_int(N)
    if no_raise or raised != N:
        print(
            "FAIL return-1 after loop: raised=%d no_raise=%d (owed %d TypeErrors)"
            % (raised, no_raise, N)
        )
        return 1
    ok, none, other = hot_instance_in_loop(N)
    if none or other or ok != N:
        print(
            "FAIL instance in loop: ok=%d none=%d other=%d (owed %d instances)"
            % (ok, none, other, N)
        )
        return 1
    raised, no_raise = hot_returns_int_in_loop(N)
    if no_raise or raised != N:
        print(
            "FAIL return-1 in loop: raised=%d no_raise=%d (owed %d TypeErrors)"
            % (raised, no_raise, N)
        )
        return 1
    print("PASS recording-time constructor escape resumed in descr_call tail")
    return 0


sys.exit(main())
