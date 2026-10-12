# pyre-check: gate=1
# A successful argparse parse then a failing one goes through parser.error
# → print_usage → gettext.find → Mapping.get.  Mapping.get's `except KeyError`
# is CHECK_EXC_MATCH.  An anchored fold guard emitted in an inlined callee
# against that compare handed it a null operand.
#
# HelpFormatter._set_color is stubbed so a jit-core binary (no `_tokenize`)
# can import argparse.  Theme attributes are real empty strings.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

import argparse
import io
import sys


class Theme:
    heading = ""
    reset = ""
    prog_extra = ""
    prog = ""
    usage = ""
    summary_action = ""
    summary_long_option = ""
    summary_short_option = ""
    summary_label = ""
    action = ""
    long_option = ""
    short_option = ""
    label = ""


def _set_color(self, color):
    self._theme = Theme()
    self._decolor = lambda x: x


argparse.HelpFormatter._set_color = _set_color


class ArgumentParserError(Exception):
    pass


def wrap(fn, *args, **kwargs):
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        try:
            return fn(*args, **kwargs)
        except SystemExit as e:
            raise ArgumentParserError(
                "SystemExit", sys.stdout.getvalue(), sys.stderr.getvalue(), e.code
            ) from None
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class P(argparse.ArgumentParser):
    def parse_args(self, *a, **k):
        return wrap(super().parse_args, *a, **k)

    def error(self, *a, **k):
        return wrap(super().error, *a, **k)


def make_parser():
    parser = P(prog="PROG", description="main description")
    parser.add_argument("--foo", action="store_true", help="foo help")
    parser.add_argument("bar", type=float, help="bar help")
    sub = parser.add_subparsers(help="command help", required=False)
    p1 = sub.add_parser("1", description="1 description")
    p1.add_argument("-w", type=int, help="w help")
    p1.add_argument("x", choices=["a", "b", "c"], help="x help")
    p2 = sub.add_parser("2", description="2 description")
    p2.add_argument("-y", choices=["1", "2", "3"], help="y help")
    p2.add_argument("z", type=complex, nargs="*", help="z help")
    return parser


FAILS = ["", "a", "a a", "0.5 a", "0.5 1", "0.5 1 -y", "0.5 2 -w"]
OK = "0.5 1 b -w 7"
N = 200


parser = make_parser()
for i in range(N):
    ns = parser.parse_args(OK.split())
    assert ns.bar == 0.5 and ns.x == "b" and ns.w == 7, ns
    for args_str in FAILS:
        try:
            parser.parse_args(args_str.split())
        except ArgumentParserError:
            pass
        else:
            raise AssertionError("expected failure: %r" % args_str)
