# cpyext-fixture: cpyext_smoke
# cpyext-env: PYRE_CPYEXT_INIT_MARKER=initialized
# cpyext-env: PYRE_CPYEXT_UNLOAD_MARKER=unloaded
# cpyext-expect: cpyext-smoke-ok

# End-to-end check for single-phase extension loading.
#
# The fixture is compiled against pyre's own 3.15 ABI header, discovered by
# the normal import path, loaded through `PyInit_*`, and created by the C call
# back into the executable's exported `PyModule_Create2`.

import _imp
import _sysconfig
import sys
expected_suffix = ('.pyre315-darwin.so' if __import__('sys').platform == 'darwin' else '.pyre315-x86_64-linux-gnu.so' if __import__('platform').machine() == 'x86_64' else '.pyre315-aarch64-linux-gnu.so' if __import__('platform').machine() == 'aarch64' else '.pyre315-linux-gnu.so')
suffixes = _imp.extension_suffixes()
assert suffixes[0] == expected_suffix
assert '.abi3.so' in suffixes
assert '.so' not in suffixes
config = _sysconfig.config_vars()
assert config['EXT_SUFFIX'] == expected_suffix
assert expected_suffix == '.' + config['SOABI'] + '.so'
old_dlopenflags = sys.getdlopenflags()
sys.setdlopenflags(0)
try:
    import cpyext_smoke
finally:
    sys.setdlopenflags(old_dlopenflags)
again = __import__('cpyext_smoke')
assert cpyext_smoke is again
assert cpyext_smoke.__name__ == 'cpyext_smoke'
assert cpyext_smoke.__doc__ == 'pyre cpyext smoke module'
assert cpyext_smoke.__file__.endswith(_imp.extension_suffixes()[0])
first = cpyext_smoke
first.__doc__ = 'mutated'
first.runtime_only = 1
del sys.modules['cpyext_smoke']
del cpyext_smoke
reloaded = __import__('cpyext_smoke')
assert reloaded is not first
assert reloaded.__name__ == 'cpyext_smoke'
assert reloaded.__doc__ == 'pyre cpyext smoke module'
assert not hasattr(reloaded, 'runtime_only')
try:
    with open(__import__('os').environ['PYRE_CPYEXT_UNLOAD_MARKER'], 'rb'):
        pass
except FileNotFoundError:
    pass
else:
    raise AssertionError('extension unloaded during cache restore')
print('cpyext-smoke-ok')
