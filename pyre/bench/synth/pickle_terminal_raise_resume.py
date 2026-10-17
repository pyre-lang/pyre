# pyre-check: spec-folds=load_attr,builtin_getattr
# pyre-check: max-wasm-ratio=4.1
# Function.call_args now enters every application-level callee through the
# recursive portal, matching PyPy's PyCode.funcrun -> PyFrame.run chain.  The
# oracle reaches 446 loops / 4 bridges in this pure-Python pickle workload;
# pyre's wasm run reaches 72 / 0, so suppressing those entry traces to recover
# the old timing would move away from the oracle.  wasm materializes the traces
# as separate modules, while this fixture also retains 450 host residual calls.
# The allowance was 6.3x, fitted to darwin-arm64 readings of 5.3x and 5.4x.
# Four readings on that host now give 3.2x, 3.3x, 3.3x and 3.5x, so 4.1x is
# the highest of them plus WASM_RATIO_FIT_HEADROOM (15%).  ubuntu-24.04 reads
# 2.4x and 2.7x and would gate this fixture at WASM_MAX_DYNASM_RATIO without
# an allowance at all; the allowance is for the darwin reading no CI run
# measures.
# The attribute fold, 690 firings across the corpus and undeclared. Pure-Python
# pickle drives it 122 times here, more than any other fixture.  It also reaches
# `builtin_getattr`, which a corpus census found no other fixture firing: once
# here, out of 77 consultations.
# Pure-Python pickle terminates its unpickler loop by raising the private
# `_Stop` exception from `load_stop`.  The JIT's after-residual guard snapshot
# must resume from the emitted post-call `-live-` anchor even though the
# semantic Python fallthrough is past the end of `load_stop`; replaying from
# the call pops the result a second time.
# The list getitem path records the list-strategy guard
# (`W_ListObject.descr_getitem` -> strategy `getitem`); its failures on
# other-strategy lists are part of guard_failures.
#
# Import pickle (and its stdlib graph) before dropping the thresholds.
# At `function_threshold=1`, nested `warmstate.py maybe_compile_and_run`
# compiles every function entry the import walks. That walk is `_find_spec`
# over `sys.meta_path`, and the first finder is not the same on every binary:
# `_pypy_abi3_tags.install` inserts `_Abi3TagsFinder` at `meta_path[0]` iff
# `'.abi3.so'` is in `_imp.extension_suffixes()`. The product dynasm binary
# lists that suffix (cpyext); the core cranelift binary lists none, and
# `install` returns on win32. PYRE_FBW_INLINE_DIAG then resolves
# `finder.find_spec` to `_Abi3TagsFinder.find_spec` versus
# `BuiltinImporter.find_spec`, so the compiled importlib set (including
# wasm's extra `get_data`) is a function of that finder. The subject loops
# below still compile pickle's own `save`/`load_*` entries at threshold 1.
import pickle

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass


checksum = 0
for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    # Keep every protocol and terminal `_Stop` path hot without turning this
    # resume regression into a separate polymorphic-trace stress test.
    for value in range(8):
        original = bytes([value & 0xFF]) * (1 + value // 256)
        payload = pickle._dumps(original, protocol)
        restored = pickle._loads(payload)
        if restored != original:
            raise AssertionError((protocol, value, restored, original))
        checksum += len(restored) + restored[0]

print("checksum =", checksum)
