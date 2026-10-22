# Regression guard: a hot STORE_GLOBAL / DELETE_GLOBAL whose namespace is an
# ordinary dict rather than a module dict.
#
# `exec(src, {})` and `types.FunctionType(code, {})` both hand the frame a plain
# `dict` for globals, so every module-dict specialization the walker owns has to
# recognise that shape and decline. Reading the module-dict strategy off such a
# frame lands on a different field, and the quasi-immutable version watcher it
# then reaches is not one — the write that follows targets read-only memory.
#
# The loop must stay hot enough to trace, and it must actually write the
# namespace on every iteration so the store specializations are reached.
#
# Import types before dropping the thresholds. At `function_threshold=1`,
# nested `warmstate.py maybe_compile_and_run` compiles every function entry
# the import walks. That walk is `_find_spec` over `sys.meta_path`, and the
# first finder is not the same on every binary: `_pypy_abi3_tags.install`
# inserts `_Abi3TagsFinder` at `meta_path[0]` iff `'.abi3.so'` is in
# `_imp.extension_suffixes()`. The product dynasm binary lists that suffix
# (cpyext); the core cranelift binary lists none, and `install` returns on
# win32. PYRE_FBW_INLINE_DIAG then resolves `finder.find_spec` to
# `_Abi3TagsFinder.find_spec` (Dirty, has a for-loop, residualized) versus
# `BuiltinImporter.find_spec` (inlined), so the compiled importlib set is a
# function of that finder, not of a clock. Counting the import therefore
# cannot hold one `.jitstats` across hosts or backends.
import types

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass

N = 774194

SRC = """
def bump(n):
    global counter, scratch
    counter = 0
    for i in range(n):
        counter = counter + i
        scratch = i
        del scratch
    return counter
"""

# exec into a bare dict: the frame's globals is a plain W_DictObject.
exec_ns = {}
exec(SRC, exec_ns)
print("exec", exec_ns["bump"](N))
print("exec-counter", exec_ns["counter"])
print("exec-scratch", "scratch" in exec_ns)

# Same code object rebound onto another plain dict through FunctionType.
alias_ns = {"__builtins__": __builtins__}
rebound = types.FunctionType(exec_ns["bump"].__code__, alias_ns)
print("rebound", rebound(N))
print("rebound-counter", alias_ns["counter"])

# A module-scope loop over the real module dict, so both namespace shapes run in
# one process and the plain-dict decline cannot be mistaken for "never traced".
total = 0
for i in range(N):
    total = total + i
print("module", total)
