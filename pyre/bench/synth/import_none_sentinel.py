# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# A None entry in sys.modules blocks the name: the import raises
# ModuleNotFoundError('import of X halted; None in sys.modules') with .name set,
# rather than reloading the module or searching for it. The sentinel is checked
# before any search, so it applies to a name that was never importable too, and
# it reaches the `from X import Y` form through the same lookup. Output verified
# against CPython/PyPy.
import sys


def blocked(name, statement):
    sys.modules[name] = None
    try:
        exec(statement)
    except ImportError as e:
        print(type(e).__name__, "|", e, "| name=", e.name)
    else:
        print(statement, "did not raise")
    finally:
        del sys.modules[name]


def main():
    for _ in range(1200):
        blocked("errno", "import errno")
        blocked("pyre_never_a_real_module", "import pyre_never_a_real_module")
        blocked("pyre_never_a_real_pkg", "from pyre_never_a_real_pkg import thing")


main()
