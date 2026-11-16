# pyre-check: jitstats-band=guard_failures=64
# The `<module>` GUARD_VALUE on `FrameDebugData.w_locals` has one counter per
# failing value (`compile.py` `AbstractResumeGuardDescr.make_a_counter_per_value`
# / `must_compile` hash the failing Ref), and every repeat execs a fresh
# namespace. Its failures are the cold count of one namespace address plus 200
# for the next: 0+200, 8+200 or 64+200 with the GC schedule (412, 420 and 476
# in total at the same 8 loops / 2 bridges; cranelift read 509 on CI).
# `decay=0` does not change the reading.
# A class body and an `exec(code, ns)` body are the two frames that receive
# their namespace through `setdictscope`, and both route a hot loop through the
# JIT portal.
#
# What this file gates in a release build is the namespace read: `exec` accepts
# any namespace that passes `PyDict_Check`, so the dict subclass and the
# OrderedDict below are both legal, and neither carries the raw `W_DictObject`
# layout that the tracer read namespace lengths and callee globals through.
# Reading them that way faulted, and this workload segfaults without the
# guards.  A fresh namespace per repeat also has the recorded trace looked up
# again against a length that cannot be read.
#
# These frames also read a Ref static virtualizable field for LOAD_NAME /
# STORE_NAME, which is what makes them the shapes that reach
# `check_synchronized_virtualizable`; a function-frame loop reads no Ref vable
# field and never looks.  Mirroring a static vable field into the shadow
# without flushing it back left the live frame behind and tripped that check on
# `last_instr` -- but the check is `debug_assertions`-gated and the values stay
# correct without it, so that half is not observable here.  It needs a debug
# build, which this suite does not run; the gate for it is
# `class_body_and_exec_loops_keep_the_vable_shadow_synchronized` in
# pyre-jit's `gc_stress` test binary.
#
# The lowered threshold lets each 200-iteration body compile in one entry.
# Every repeat still uses fresh namespace objects, so compiled code rechecks
# the class namespace, dict subclass and OrderedDict owners.
import collections

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 12000
REPEAT = 3


def class_body_while():
    class C:
        i = 0
        total = 0
        while i < N:
            total += i
            i += 1

    return C.total


def class_body_for():
    class C:
        total = 0
        for i in range(N):
            total += i

    return C.total


CODE = compile(
    "total = 0\ni = 0\nwhile i < N:\n    total += i\n    i += 1\n",
    "<exec-ns>",
    "exec",
)


class NS(dict):
    pass


def exec_into(ns):
    ns["N"] = N
    exec(CODE, ns)
    return ns["total"]


def main():
    checksum = 0
    for _ in range(REPEAT):
        checksum += class_body_while()
        checksum += class_body_for()
        checksum += exec_into(NS())
        checksum += exec_into(collections.OrderedDict())
    print(checksum)


main()
