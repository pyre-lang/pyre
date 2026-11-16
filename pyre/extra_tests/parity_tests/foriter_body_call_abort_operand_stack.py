# CPython-suite gap: iteration tests cannot exercise a JIT inline-walk abort stack.
# parity-tests reason: this is a pyre trace-abort operand-stack regression.

# An aborted CALL sub-walk rebuilds the FOR_ITER caller's stack from
# `registers_r` / `virtualizable_boxes` and the encoded residual. Method-form
# CALLs have a variable-width leading Int list; reading them at the plain-call
# offset once restored the iterator as the subscript index. `re._compile_info` reaches that exact hot BRANCH/`p[0]`
# shape only with a large alternation, and compilation itself exposes a wrong
# operand by raising.

import re

BRANCHES = 2000

pattern = "|".join("%d" % x for x in range(BRANCHES))
compiled = re.compile(pattern)

# Alternation is first-match, so "1999" is matched by the earlier "1" branch.
assert compiled.match("1999").group(0) == "1"
assert compiled.match("nope") is None

print("OK")
