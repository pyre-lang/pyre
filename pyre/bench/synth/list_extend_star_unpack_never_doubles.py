# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=entry-bridge:star,entry-bridge:hsp,parse_headers
# Star-unpack `(value, *rest)` lowers to LIST_EXTEND, which residual-executes
# `extend_from_list` inside a transparent helper and then aborts. The helper
# is its own MIFrame and must resume past that residual so the WALK_ABORT
# blackhole does not re-run LIST_EXTEND and duplicate the donor.

try:
    import pypyjit

    pypyjit.set_param("threshold=3,function_threshold=3")
except ImportError:
    pass


def star(value, rest):
    return (value, *rest)


def hsp(sourcelines):
    name, value = sourcelines[0].split(":", 1)
    value = "".join((value, *sourcelines[1:])).lstrip(" \t\r\n")
    return (name, value.rstrip("\r\n"))


def parse_headers(lines):
    lastheader = ""
    lastvalue = []
    out = []
    for line in lines:
        if line[0] in " \t":
            if not lastheader:
                continue
            lastvalue.append(line)
            continue
        if lastheader:
            out.append(hsp(lastvalue))
            lastheader, lastvalue = "", []
        i = line.find(":")
        lastheader = line[:i]
        lastvalue = [line]
    if lastheader:
        out.append(hsp(lastvalue))
    return out


H = [
    "Content-Type: text/plain\n",
    "Content-Disposition: inline;\n",
    '\tfilename="msg.txt"\n',
    "Content-Transfer-Encoding: 7bit\n",
]
EXPECTED = 'inline;\n\tfilename="msg.txt"'

bad = []
for i in range(100):
    t = star("inline;\n", ['\tfilename="msg.txt"\n'])
    if t != ("inline;\n", '\tfilename="msg.txt"\n') or len(t) != 2:
        bad.append((i, t, len(t)))

for i in range(100):
    out = parse_headers(H)
    disp = [v for k, v in out if k == "Content-Disposition"]
    if disp != [EXPECTED]:
        bad.append((i, disp))

assert not bad, bad
print("PASS")
