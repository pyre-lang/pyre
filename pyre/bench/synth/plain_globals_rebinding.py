# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=loop:hot,loop:builtin_hot
# pyopcode.py PyFrame._load_global: each live frame supplies its globals
# and builtins. Reusing a code object must preserve that choice through
# compiled loops, rebinding, deletion and builtin shadowing.
import types

SRC = '''
def hot(n):
    total = 0
    for i in range(n):
        total += K
    return total

def builtin_hot(n):
    total = 0
    for i in range(n):
        total += len(())
    return total
'''

code = compile(SRC, '<globals-descent>', 'exec')
for g, k in ((types.ModuleType('m').__dict__, 1), ({}, 13), ({}, 71)):
    g['K'] = k
    exec(code, g)
    hot = g['hot']
    assert hot(10000) == 10000 * k
    g['K'] = k + 3
    assert hot(10000) == 10000 * (k + 3)
    del g['K']
    try:
        hot(1)
    except NameError as exc:
        assert exc.name == 'K'
    else:
        raise AssertionError('deleted global remained visible')
    g['K'] = k + 9
    assert hot(10000) == 10000 * (k + 9)
    builtin_hot = g['builtin_hot']
    assert builtin_hot(10000) == 0
    g['len'] = lambda value: 7
    assert builtin_hot(10000) == 70000
    del g['len']
    assert builtin_hot(10000) == 0
print('PASS')
