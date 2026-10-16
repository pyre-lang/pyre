# pyre-check: max-pypy-ratio=21
# A module body whose JitCode lowering stops early must not let a later loop
# header borrow the truncation's resume marker. `class` compiles to
# LOAD_BUILD_CLASS, which has no JitCode lowering, so the body ends there with
# an `abort_permanent` block; every py_pc after it is unlowered, and the
# marker table forward-carries the loop header onto that block. Walking from
# there aborts at the marker and back-translates it to the unported
# instruction's own py_pc — before the loop — rewinding the frame so the whole
# span between them runs twice.
#
# The counters below sit in that span. `steps` is the loop's own work, so the
# body stays hot enough to reach the trace threshold on every backend; `runs`
# and the list length report the span's execution count, which is 1.

runs = 0
seen = []


class Marker:
    pass


runs = runs + 1
seen.append("once")

i = 0
steps = 0
while i < 2400000:
    steps = steps + i
    i = i + 1

print(runs, len(seen), seen, steps)
