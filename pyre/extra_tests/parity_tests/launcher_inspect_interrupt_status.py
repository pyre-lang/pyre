# pyre-check: platforms=linux,darwin
# pyre-check: pypy-diverges: `app_main.py` returns the prompt's status and forgets the program's interrupt, so the `-i` rows answer 0 instead of SIGINT
# CPython-suite gap: command-line tests pin SIGINT without `-i`, but never the
# status of a prompt opened after an interrupt.
# parity-tests reason: `_PyErr_PrintEx` records `KeyboardInterrupt` and
# `Py_RunMain` consults it after the prompt and finalization.
# gh-152132 changed the `exit(n)` precedence in 3.14.7, so those expectations
# follow the micro version the runtime reports. `subprocess` spells SIGINT as
# `-signal.SIGINT`; Windows is excluded by the platform header.
#
# launcher_inspect_prompt_over_a_blocked_sys:
# import tests cover the `None` sentinel and command-line tests cover `-i`, but never
# combine them. the launcher reads `sys.ps1`/`ps2` from `PyInterpreterState` through
# `_PySys_GetOptionalAttr`; a program's blocked `sys.modules['sys']` entry must not
# prevent the requested prompt. `app_main.py` imports the blocked `sys` name, prints an
# OperationError and exits 1 without opening the prompt
import signal
import subprocess
import sys

INTERRUPTED = -signal.SIGINT
EXIT_ANSWERS_THE_INTERRUPT = sys.version_info[:3] < (3, 14, 7)


def run(args, feed=""):
    return subprocess.run(
        [sys.executable, *args], input=feed, capture_output=True, text=True
    ).returncode


# The control: without `-i` every runtime already dies of the interrupt, so the
# rows below are about the prompt and nothing else.
assert run(["-c", "raise KeyboardInterrupt"]) == INTERRUPTED

# A prompt that simply ended does not answer the interrupt, so the run still
# reports itself as interrupted.
assert run(["-i", "-c", "raise KeyboardInterrupt"]) == INTERRUPTED

# An interrupt raised at the prompt itself is recorded the same way.
assert run(["-i", "-c", "pass"], "raise KeyboardInterrupt\n") == INTERRUPTED

# An explicit status is the answer whenever there is no record to outrank it,
# which is what makes the two rows below about the record and not about `exit`.
assert run(["-i", "-c", "pass"], "exit(0)\n") == 0
assert run(["-i", "-c", "pass"], "exit(3)\n") == 3

# With a record standing, which of the two answers the shell is the part that
# moved.  A successful `exit(0)` is the case that cannot be explained by "the
# largest status wins" either way.
answer_to_exit_0 = 0 if EXIT_ANSWERS_THE_INTERRUPT else INTERRUPTED
answer_to_exit_3 = 3 if EXIT_ANSWERS_THE_INTERRUPT else INTERRUPTED
assert run(["-i", "-c", "raise KeyboardInterrupt"], "exit(0)\n") == answer_to_exit_0
assert run(["-i", "-c", "raise KeyboardInterrupt"], "exit(3)\n") == answer_to_exit_3

# A different exception is not an interrupt: the prompt's own clean end stands.
assert run(["-i", "-c", "raise ValueError('boom')"]) == 0

def launcher_inspect_prompt_over_a_blocked_sys():
    import subprocess
    import sys

    PROGRAM = "import sys; sys.modules['sys'] = None"
    MARKER = "prompt-ran"


    def run(args, feed=""):
        proc = subprocess.run(
            [sys.executable, *args], input=feed, capture_output=True, text=True
        )
        return proc.returncode, proc.stdout


    # The control: the assignment itself is ordinary, and a run without `-i` ends
    # cleanly, so the rows below are about the prompt and nothing else.
    assert run(["-c", PROGRAM]) == (0, "")

    # The prompt opens over it and runs what is typed there.
    status, out = run(["-i", "-c", PROGRAM], f"print({MARKER!r})\n")
    assert status == 0, f"the prompt did not end cleanly: {status}"
    assert MARKER in out, f"the prompt did not run the statement it was given: {out!r}"

    # And it is the prompt's own status that ends the run, not the program's.
    status, _ = run(["-i", "-c", PROGRAM], "raise SystemExit(3)\n")
    assert status == 3, f"the prompt's exit status did not reach the shell: {status}"
launcher_inspect_prompt_over_a_blocked_sys()
print("OK")
