# list.extend(iter(callable, sentinel)) drains a buffered line reader.
# `_do_extend_from_iterable` appends every `space.next` result; a tracing
# walk that runs `next` and then aborts must keep that item
# (`warmspot.py` `handle_jitexception`). Each seed prints True when the
# extend matches a manual readline drain of the same stream.


class NeedMoreData:
    pass


class Buf:
    def __init__(self):
        self.partial = []
        self.lines = []

    def push(self, data):
        self.partial.append(data)
        joined = "".join(self.partial)
        parts = joined.splitlines(True)
        if parts and not parts[-1].endswith(("\n", "\r")):
            self.partial = [parts.pop()]
        else:
            self.partial = []
        self.lines.extend(reversed(parts))

    def readline(self):
        if not self.lines:
            return NeedMoreData
        return self.lines.pop()


def run(seed):
    x = seed
    chars = "abcd \t\r\n"
    out = []
    for _ in range(10000):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(chars[x % 8])
    s = "".join(out) + "\n"
    ref = Buf()
    target = []
    for i in range(0, len(s), 5):
        ref.push(s[i : i + 5])
        while True:
            line = ref.readline()
            if line is NeedMoreData:
                break
            target.append(line)
    bsf = Buf()
    lines = []
    for i in range(0, len(s), 5):
        bsf.push(s[i : i + 5])
        lines.extend(iter(bsf.readline, NeedMoreData))
    return (len(lines), len(target)) if lines != target else True


print([run(seed) for seed in range(8)])
