# pyre-check: max-pypy-ratio=17
# A compiled module loop may be reused by exec() with a fresh globals/locals
# dict.  Guard-failure resume must carry this frame's debugdata and w_globals,
# not the recording frame's mappings, or DELETE_NAME targets the stale locals.
src = "acc = 1\nfor r in range(576000):\n    pass\ndel acc\n"
code = compile(src, "<exec-fresh-globals>", "exec")

checksum = 0
for _ in range(6):
    exec(code, {"range": range})
    checksum += 1

print(checksum)
