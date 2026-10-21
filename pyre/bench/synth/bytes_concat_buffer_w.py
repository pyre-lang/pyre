# Bytes/bytearray/memoryview concat and bytes(memoryview). Hot path is
# space.buffer_w / _try_buffer_w (W_BytesObject.buffer_w for exact bytes).
# No max-pypy-ratio: counter oracle (loops, residual calls).
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 30000


def main():
    s = 0
    i = 0
    while i < N:
        a = b"ab" + b"cd"
        b = b"ab" + bytearray(b"cd")
        c = b"ab" + memoryview(b"cd")
        d = bytes(memoryview(b"xy"))
        s = s + a[0] + b[0] + c[0] + d[0] + len(a) + len(b) + len(c) + len(d)
        i = i + 1
    print(s)


main()
