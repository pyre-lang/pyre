# `locals()` inside a hot `for` loop must report the frame's live values.
#
# `locals()` reaches its frame through `gettopframe_nohidden`, which walks the
# `f_backref` chain without forcing anything, and `fast2locals` then reads
# `locals_cells_stack_w` directly. Once the loop compiles, an unforced
# virtualizable still holds whatever the frame last wrote out, so the mapping
# comes back with correct KEYS (they come from the code object) and stale
# VALUES -- a key-only assertion does not see it.
#
# The shape matters: only a `FOR_ITER` loop reproduces. The same body written
# with `while` reports live values whether or not the frame is forced.


def probe():
    seen = []
    for i in range(1000000):
        a = i * 2
        b = a + 1
        if i == 5000 or i == 999999:
            d = locals()
            seen.append((d.get("i"), d.get("a"), d.get("b")))
    return seen


def probe_vars():
    out = None
    for i in range(1000000):
        a = i * 2
        if i == 999999:
            out = vars().get("a")
    return out


print(probe())
print(probe_vars())
