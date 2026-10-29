# pyre-check: max-pypy-ratio=109
# Merged synth parity smoke suite: independent feature-level hot loops, each
# kept verbatim from its former standalone file with module-level names prefixed
# by the source name. Bug-repro / resume / kept-stack tests are NOT merged (they
# stay isolated so a miscompile is not diluted). check.py runs every *.py here.

# ── recursion ──
# The trace-limit exit converts the complete recursive MIFrame chain through
# blackhole.py convert_and_run_from_pyjitpl instead of CALL-entry replay.
# One multi-frame adoption preserves the existing loop, bridge and guard counts.
def recursion__fib(n):
    if n < 2:
        return n
    return recursion__fib(n - 1) + recursion__fib(n - 2)

def recursion__main():
    i = 0
    acc = 0
    while i < 8:
        acc = acc + recursion__fib(18)
        i = i + 1
    print(acc)
recursion__main()

# ── closures ──
closures__N = 100000

def closures__make_adder(k):

    def inner(x):
        return x + k
    return inner

def closures__main():
    add5 = closures__make_adder(5)
    add9 = closures__make_adder(9)
    i = 0
    acc = 0
    while i < closures__N:
        acc = acc + add5(i)
        acc = acc - add9(i // 2)
        i = i + 1
    print(acc)
closures__main()

# ── function_calls ──
function_calls__N = 120000

def function_calls__add3(a, b, c):
    return a + b + c

def function_calls__mix(a, b):
    if a & 1:
        return function_calls__add3(a, b, 7)
    return function_calls__add3(b, a, -3)

def function_calls__main():
    i = 0
    acc = 0
    while i < function_calls__N:
        acc = acc + function_calls__mix(i, acc & 255)
        i = i + 1
    print(acc)
function_calls__main()

# ── default_keyword_args ──
default_keyword_args__N = 100000

def default_keyword_args__f(a, b=3, c=5):
    return a + b * 2 - c

def default_keyword_args__main():
    i = 0
    acc = 0
    while i < default_keyword_args__N:
        acc = acc + default_keyword_args__f(i)
        acc = acc + default_keyword_args__f(i, c=7)
        acc = acc + default_keyword_args__f(i, b=11, c=13)
        i = i + 1
    print(acc)
default_keyword_args__main()

# ── generator_iteration ──
generator_iteration__N = 30000

def generator_iteration__gen(n):
    i = 0
    while i < n:
        yield (i * 2 + 1)
        i = i + 1

def generator_iteration__main():
    i = 0
    acc = 0
    while i < generator_iteration__N:
        for x in generator_iteration__gen(6):
            acc = acc + x + (i & 3)
        i = i + 1
    print(acc)
generator_iteration__main()

# ── load_deref ──
load_deref__N = 100000

def load_deref__make_adder():
    base = 7

    def run(n):
        acc = 0
        i = 0
        while i < n:
            acc = acc + base
            i = i + 1
        return acc
    return run
print(load_deref__make_adder()(load_deref__N))
