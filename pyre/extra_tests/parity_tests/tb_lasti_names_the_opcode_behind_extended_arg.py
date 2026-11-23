# pyre-check: pypy-diverges: pypy3's dispatch_bytecode leaves last_instr at the EXTENDED_ARG prefix, so tb_lasti names EXTENDED_ARG there
# CPython-suite gap: no frame or traceback test reads `tb_lasti` for an
# instruction behind an EXTENDED_ARG prefix.
# parity-tests reason: the dispatch jitcode stores the post-prefix opcode pc
# into `last_instr` on a prefixed case; the traceback must name the real
# opcode, not the prefix, both interpreted and from compiled code.
import dis


def _make():
    attrs = ", ".join("o.n%d" % i for i in range(300))
    src = (
        "def f(o):\n"
        "    if o is None:\n"
        "        (" + attrs + ")\n"
        "    return o.zz_missing\n"
    )
    ns = {}
    exec(src, ns)
    return ns["f"]


f = _make()
ops = {ins.offset: ins.opname for ins in dis.get_instructions(f)}


class O:
    pass


def lasti_opname():
    try:
        f(O())
    except AttributeError as e:
        return ops.get(e.__traceback__.tb_next.tb_lasti)
    return None


def main(n):
    seen = {}
    for _ in range(n):
        name = lasti_opname()
        seen[name] = seen.get(name, 0) + 1
    return sorted(seen.items())


assert main(1) == [("LOAD_ATTR", 1)], main(1)
assert main(3000) == [("LOAD_ATTR", 3000)], main(3000)
print("OK")
