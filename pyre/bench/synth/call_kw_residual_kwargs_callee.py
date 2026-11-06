# A CALL_KW whose callee takes `*rest` and `**kw`, reading `kw["step"]`.
# The plain-user keyword call walks the interpreter's binder
# (`bind_plain_user_kw_scope` → `match_signature_indexed`), which stores
# the surplus name into the `**kw` dict, and that when that binder
# declines the call stays a residual CallMayForce. The result depends on
# both the positional and the keyword arg so the call cannot be
# constant-folded away, and the exact aggregate makes a single dropped
# iteration observable.
#
# When the binder declines, the call stays a residual. That residual is
# what the NULL-Ref-arg refusal sees, and its arg index 1 is the
# `null_or_self` receiver slot, holding the PY_NULL sentinel (GcRef(0)) of
# a plain no-receiver call. The refusal carries an `is_call_kw` exemption
# for it because that null is a checked sentinel, not an unresolved ref.
# Without the exemption the refusal declines the recording iteration's
# call to a symbolic op and drops that iteration's effect, leaving the
# sum exactly one term short.
#
# call_kw_hot_loop.py is the other half: the same call spelling with a
# callee that has no `**kw` dict.
N = 200000


def g(x, *rest, **kw):
    return x + kw["step"]


def main():
    total = 0
    for i in range(N):
        total += g(i, step=2)
    print(total)


main()
