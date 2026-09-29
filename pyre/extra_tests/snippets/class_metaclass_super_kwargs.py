# pyre-check: gate=1
# PyPy typeobject.py W_TypeObject.descr_call receives one Arguments
# object and forwards it unchanged to __new__ and __init__.  The
# default metaclass descriptor reached through super().__call__ must
# therefore keep `config` as a keyword instead of exposing pyre's
# builtin keyword-marker dict as a positional argument.
class Meta(type):
    def __call__(cls, *args, **kwargs):
        return super().__call__(*args, **kwargs)
class Config:
    rootpath = 'ok'
class Node(metaclass=Meta):
    def __init__(self, config):
        self.rootpath = config.rootpath
node = Node(config=Config())
result = node.rootpath

assert result == 'ok'

assert result == 'ok'

class M(type):
    def __call__(cls, *args):
        return ('meta', args)
class C(metaclass=M):
    pass
assert C(7) == ('meta', (7,))
