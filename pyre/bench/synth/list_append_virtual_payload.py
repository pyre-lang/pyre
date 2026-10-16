# A list strategy consumes the payload of a freshly boxed comprehension value.
# Keep both integer and float payloads live across hot range iterations: an
# append specialization must not retain a trace-entry value or lose a payload
# when the comprehension crosses from tracing into compiled execution.

expected = list(range(100))
checksum = 0
for outer in range(4000):
    ints = [item for item in range(100)]
    assert ints == expected
    floats = [float(item) - 0.5 for item in range(100)]
    assert floats[0] == -0.5
    assert floats[-1] == 98.5
    checksum += ints[-1] + int(floats[-1])

print(checksum)
