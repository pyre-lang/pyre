# CPython-suite gap: test_unpack_ex does not warm this starred tuple
# expansion past the JIT threshold inside an inlined Python call.
# parity-tests reason: a helper guard failure after native list extension
# must not repeat the mutation when execution resumes.
# Regression: LIST_EXTEND inside an inlined Python call must execute once
# when a helper guard cannot be recorded and tracing switches to blackhole.
# email.header_source_parse uses the same (value, *continuation_lines) shape.


def expand(items):
    return ("head", *items)


for iteration in range(3000):
    result = expand(["tail"])
    assert result == ("head", "tail"), (iteration, result)

print("OK")
