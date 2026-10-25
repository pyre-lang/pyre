# pyre-check: gate=1
# pyre-check: platforms=win32
# CPython-suite gap: test_sys and test_io read `sys.stdout.encoding` only
# through `PYTHONIOENCODING`, and the vendored suites never compare the
# unnamed answer against the host's own code page.
# parity-tests reason: `initstdio` took PyPy's `_WIN32 and not encoding`
# shortcut on every host, so a redirected stream opened as utf-8 where
# `config_get_locale_encoding` answers the ANSI code page -- a script whose
# output is piped into a Windows tool wrote bytes that tool cannot read.

# pyre-check: pypy-diverges: pypy3 always runs with `sys.flags.utf8_mode == 1`,
# so it ignores `-X utf8=0` and answers utf-8 in the code-page case.

"""Outside UTF-8 mode a redirected standard stream opens with the locale's
encoding; 3.15 turns UTF-8 mode on by default (PEP 686)."""

import locale
import subprocess
import sys

PROBE = (
    "import sys;"
    " print(sys.stdout.encoding, sys.stderr.encoding, sys.stdin.encoding,"
    " sys.stdout.errors, sys.stderr.errors, sys.flags.utf8_mode)"
)


def probe(*options, **env_overrides):
    """The child's stream configuration, with every stream a pipe."""
    import os

    env = dict(os.environ)
    # Both are read at startup; an inherited one would decide the answer
    # before the case under test does.  `PYTHONUTF8` settles utf8_mode ahead
    # of the locale, which is the very rule the first case reads.
    env.pop("PYTHONIOENCODING", None)
    env.pop("PYTHONUTF8", None)
    env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, *options, "-c", PROBE],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    return result.stdout.decode("ascii").split()


# The code page the host names, spelled the way `_Py_GetLocaleEncoding` spells
# it.  `locale.getencoding()` is the same call, so the two must agree.
code_page = locale.getencoding()

# UTF-8 mode is the default, and it fixes every stream at utf-8 whatever the
# code page is.
stdout, stderr, stdin, out_errors, _, utf8_mode = probe()
assert utf8_mode == "1", utf8_mode
assert (stdout, stderr, stdin) == ("utf-8", "utf-8", "utf-8"), (stdout, stderr, stdin)
assert out_errors == "surrogateescape", out_errors

stdout, stderr, stdin, out_errors, err_errors, utf8_mode = probe("-X", "utf8=0")
assert utf8_mode == "0", utf8_mode
assert stdout == code_page, (stdout, code_page)
assert stderr == code_page, (stderr, code_page)
assert stdin == code_page, (stdin, code_page)
# Windows always asks for surrogateescape; stderr replaces it so that printing
# a traceback cannot fail on an unencodable character.
assert out_errors == "surrogateescape", out_errors
assert err_errors == "backslashreplace", err_errors

# PYTHONUTF8=0 turns the mode off the same way.
stdout, _, _, _, _, utf8_mode = probe(PYTHONUTF8="0")
assert utf8_mode == "0", utf8_mode
assert stdout == code_page, (stdout, code_page)

# A named encoding wins over both, and naming one settles the error handler
# at strict.
stdout, stderr, stdin, out_errors, err_errors, _ = probe(
    PYTHONIOENCODING="iso8859-1"
)
assert (stdout, stderr, stdin) == ("iso8859-1", "iso8859-1", "iso8859-1"), stdout
assert out_errors == "strict", out_errors
assert err_errors == "backslashreplace", err_errors

# The encoding is what the bytes are actually written in, not only what the
# stream reports.  U+00E9 has a one-byte spelling in latin-1 and a two-byte
# one in utf-8.
written = subprocess.run(
    [sys.executable, "-c", 'import sys; sys.stdout.write("\\u00e9")'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**__import__("os").environ, "PYTHONIOENCODING": "iso8859-1"},
    check=False,
)
assert written.returncode == 0, written.stderr
assert written.stdout == b"\xe9", written.stdout

print("OK")
