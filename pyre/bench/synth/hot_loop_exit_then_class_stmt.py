# A hot module-level `while` loop whose exit deopts into the blackhole, followed
# by observable statements and then a `class` statement.
#
# The blackhole resumes the module frame past the loop guard and executes the
# statements between the loop and the `class` concretely; the `class` statement
# then hits an op it cannot perform and aborts.  When the abort path restored
# the pre-blackhole frame snapshot (locals / valuestackdepth / last_instr) and
# handed control back to the interpreter, the interpreter re-ran the region and
# every effect already performed by the blackhole was applied a second time:
# `print` fired twice, `append` appended twice, `n` counted twice.
#
# Everything here runs at the default thresholds — no `pypyjit.set_param`.
N = 1080000

log = []
n = 0

i = 0
while i < N:
    i = i + 1

print("after loop")
log.append("a")
n = n + 1


class C:
    pass


print("log =", log, "n =", n, "i =", i)


# The same shape one level down: the loop, the effects and the `class` all sit
# inside a function frame, so the blackhole resumes a non-module frame.
def inner():
    entries = []
    count = 0
    j = 0
    while j < N:
        j = j + 1
    entries.append("b")
    count += 1

    class D:
        pass

    return entries, count, D.__name__


print(inner())
