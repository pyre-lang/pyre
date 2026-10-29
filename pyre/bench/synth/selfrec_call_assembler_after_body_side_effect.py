# A side effect that runs before a self-recursive call must apply exactly
# once when that call folds to CALL_ASSEMBLER.  Four bodies: list.append
# before the recursive return, a dict write with two parameters, a global
# counter in a fib-shaped recursion, and the append shape executed with a
# plain dict as globals.  A double-apply from folding after an eager
# residual would show as a larger event count than the interpreter.
append_events = []


def append_rec(n):
    if n <= 0:
        return 0
    append_events.append(n)
    return append_rec(n - 1) + 1


append_total = 0
for i in range(3000):
    append_total += append_rec(12)
print(append_total, len(append_events), sum(append_events))

dict_seen = {}


def dict_walk(n, acc):
    dict_seen[n] = dict_seen.get(n, 0) + 1
    if n == 0:
        return acc
    return dict_walk(n - 1, acc + n)


dict_total = 0
for i in range(2000):
    dict_total += dict_walk(15, i)
print(dict_total, sorted(dict_seen.items())[:3], sum(dict_seen.values()))

count = 0


def count_rec(n):
    global count
    count += 1
    if n < 2:
        return n
    return count_rec(n - 1) + count_rec(n - 2)


count_total = 0
for i in range(23):
    count_total += count_rec(i)
print(count_total, count)

SRC = """
plain_events = []
def plain_rec(n):
    if n <= 0:
        return 0
    plain_events.append(n)
    return plain_rec(n - 1) + 1
plain_total = 0
for i in range(3000):
    plain_total += plain_rec(12)
"""
g = {}
exec(compile(SRC, "<s>", "exec"), g)
print(g["plain_total"], len(g["plain_events"]), sum(g["plain_events"]))
