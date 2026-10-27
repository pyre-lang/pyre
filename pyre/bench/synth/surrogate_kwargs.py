# pyre-check: max-pypy-ratio=125
# A lone-surrogate keyword name survives the f(**dict) call path: it lands
# in **kwargs unchanged and an unmatched one is named in the TypeError.
# Arguments keeps keyword names as byte-ish str (argument.py keywords: [str]),
# so unpacking a `**{'\udc80': v}` dict must not drop or mangle the key.

N = 200000

S1 = '\udc81'            # lone surrogate keyword name
S2 = '\udc84\udc85'      # multi-surrogate keyword name


def collect(**kw):
    return kw


def needs_a(a):
    return a


def main():
    # Surrogate keyword names passed through ** reach **kwargs unchanged,
    # interleaved with a plain name.
    kw = collect(**{'plain': 1, S1: 2, S2: 3})
    acc = 0
    acc = acc + len(kw)
    acc = acc + kw['plain'] + kw[S1] + kw[S2]
    acc = acc + (1 if S1 in kw else 0)
    acc = acc + (1 if S2 in kw else 0)
    # Ordinal round-trip of the surrogate keys read back from **kwargs.
    acc = acc + sum(ord(c) for c in S1)
    acc = acc + sum(ord(c) for c in S2)

    # An unmatched surrogate keyword raises TypeError naming the callable.
    try:
        needs_a(**{S1: 9})
    except TypeError:
        acc = acc + 100000

    # Hot loop: the JIT-compiled f(**dict) path threads the surrogate key
    # into **kwargs every iteration instead of dropping it.
    i = 0
    while i < N:
        k = collect(**{S1: 1})
        if S1 in k:
            acc = acc + 1
        i = i + 1
    print(acc)


main()
