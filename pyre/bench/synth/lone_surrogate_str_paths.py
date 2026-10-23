# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code. The jitstats baselines gate it.
# Lone surrogates must survive str paths or raise UnicodeEncodeError.
# They must not abort the process. Messages are printed with ascii() so a
# surrogate in the text stays on stdout. PyPy's AttributeError text writes
# the surrogate as the six characters \ud800; folding that back to the
# code point makes the line match the reference. Other exception messages
# already match, so the fold stays on AttributeError alone.

import ast
import codecs
import os
import sys

S = chr(0xD800)


def show(name, fn):
    try:
        value = fn()
        text = "OK " + ascii(value)
    except Exception as exc:
        message = str(exc)
        if type(exc) is AttributeError:
            message = message.replace("\\ud800", "\ud800")
        text = "EXC " + type(exc).__name__ + " " + ascii(message)
    print(name, text)


def error_handler(err):
    return ("Z", err.end)


codecs.register_error("lone_surrogate_probe", error_handler)


class Box:
    pass


box = Box()

# ast
show("ast-dump", lambda: ast.dump(ast.Constant(value=S)))
show("ast-id", lambda: ast.Name(id=S).id)

# getattr / setattr names
show("getattr", lambda: getattr(sys, S))
show("setattr", lambda: (setattr(box, S, 7), getattr(box, S))[1])
show("hasattr", lambda: hasattr(box, S))

# format
show("fmt", lambda: "{}".format(S))
show("pct", lambda: "%s" % S)

# dict keys
show("dict", lambda: {S: 1}[S])
show("kw", lambda: (lambda **k: k[S])(**{S: 1}))
show("kwbad", lambda: (lambda a: a)(**{S: 1}))

class Named:
    pass


Named.__module__ = S
show("cls", lambda: repr(Named))
box_named = Named()
show("obj", lambda: repr(type(box_named)))

# str methods
show("add", lambda: S + "a")
show("mul", lambda: S * 2)
show("find", lambda: "a".find(S))
show("replace", lambda: "ab".replace(S, "x"))
show("split", lambda: S.split())
show("startswith", lambda: "a".startswith(S))
show("upper", lambda: S.upper())
show("count", lambda: ("a" + S).count(S))

# encode / decode and an error handler
show("enc", lambda: S.encode("utf-8"))
show("encsp", lambda: S.encode("utf-8", "surrogatepass"))
show("dec", lambda: b"\xed\xa0\x80".decode("utf-8"))
show("decsp", lambda: b"\xed\xa0\x80".decode("utf-8", "surrogatepass"))
show("handler", lambda: "\xff".encode("ascii", "lone_surrogate_probe"))

# os path functions. On POSIX a str path is encoded with the filesystem
# encoding, so a lone surrogate has to fail the encode or be escaped. Windows
# hands the name to the wide-char API, where the lookup itself fails and the
# message is the host's; that is not what this fixture checks.
show("path", lambda: os.path.split(os.path.join(S, "a")))
if os.name == "posix":
    for fn in (os.listdir, open):
        try:
            fn(S)
        except UnicodeEncodeError as exc:
            assert exc.object == S
            assert (exc.start, exc.end, exc.reason) == (0, 1, "surrogates not allowed")
        else:
            raise AssertionError("a lone high surrogate reached the filesystem")
# wasm uses POSIX paths even on a Windows host. Keep the output portable
# while asserting the POSIX encoding failure above on every POSIX runtime.
print("filesystem path checks done")

print("done")
