# A `**kwargs` callee the inline takes: the keywords no parameter names are
# collected into a fresh kwargs-strategy dict (`_match_signature`), built by
# residual stores the trace records, and the callee body reads it.  The loop
# covers an empty mapping, keywords next to positionals, a positional-only
# name that has to land in the mapping, a keyword overriding a default, a
# callee that also takes `*args`, and a mapping the callee mutates; the
# printed tuples pin every binding, and the final TypeError pins that a
# duplicate binding still raises.


def f(**kw):
    return kw


def g(a, **kw):
    return a, kw


def h(a, /, **kw):
    return a, kw


def k(a, b=5, **kw):
    return a, b, kw


def m(*args, **kw):
    return args, kw


def main(n):
    acc = []
    total = 0
    for i in range(n):
        r1 = f(a=i, b=2)
        r2 = f()
        r3 = g(i, x=1)
        r4 = g(a=i, y=3)
        r5 = h(i, a=7)
        r6 = k(i, c=4)
        r7 = k(i, b=9, z=1)
        r8 = m(i, q=1)
        r9 = g(i)
        d = f(z=i)
        d['w'] = 1
        total += r1['a'] + len(r1) + len(d)
        if i == n - 1 or i == 3:
            acc.append((r1, r2, r3, r4, r5, r6, r7, r8, r9, d, list(r1)))
    try:
        g(1, a=2)
    except TypeError as e:
        acc.append(str(e))
    print(total)
    for x in acc:
        print(x)


main(9000)
