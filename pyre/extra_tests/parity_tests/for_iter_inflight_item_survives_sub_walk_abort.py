# CPython-suite gap: the loss is a jit walk-abort path, invisible to an interpreter-level suite.
# parity-tests reason: pins a consumed FOR_ITER item against an abort inside a nested inline callee, and against a root-trace too-long abort at an EXTENDED_ARG-prefixed FOR_ITER.

"""A FOR_ITER item consumed before an aborted sub-walk still reaches the body.

`step` is a bound method that mutates before it calls, so the walk admits it as
an inline callee and the append commits; `rec` is self-recursive, so descending
into it is refused and the walk aborts with the item already taken off the
iterator. Redelivering it would replay the append, so delivery is refused
whenever a body effect has committed -- which leaves the abort itself owing the
item a carrier. Without one the item is neither delivered nor rolled back, and
the loop body runs for it while its value never lands in `res`: the failure
signature is `len(res) < len(w.seen)`.

The accumulator is a statement loop, so no LIST_APPEND is involved and what is
pinned is the item rather than the opcode that accumulates it. The trip counts
are the ones the loss was observed at -- the walk has to warm up before the
abort is reached, and a shorter sweep never reaches it.
"""


def ck(seq):
    h = 7
    for v in seq:
        h = (h * 1000003 + v) & 0xFFFFFFFFFF
    return h


def rec(x):
    if x <= 0:
        # `id` is opaque, so the descent stops here rather than folding away.
        return 1 if id(x) else 0
    return rec(x - 1) + 1


class W:
    def __init__(self):
        self.seen = []

    def step(self, x):
        # Committed before the call below, and no rollback undoes it.
        self.seen.append(x)
        return x * 1000003 + rec(x % 4)


hh = 7
bad = []
for trial in range(1000):
    n = trial % 20
    w = W()
    res = []
    for i in range(n):
        res.append(w.step(i))
    if len(res) != n or len(w.seen) != n:
        bad.append((trial, n, len(res), len(w.seen)))
    # Consuming both lists every trial is part of the shape the loss needs.
    hh = (hh * 1000003 + ck(res) + ck(w.seen) + len(w.seen)) & 0xFFFFFFFFFF

assert not bad, f"dropped items, (trial, n, len(res), len(seen)): {bad}"
assert hh == 950649862511, hh


# A list FOR_ITER whose consume the walk descended, in a root loop trace that
# runs out of length before the iteration's body completes.  The trace limit
# puts that abort inside `sum`'s loop, whose FOR_ITER carries an EXTENDED_ARG:
# the frame is parked on the prefix, one unit before the FOR_ITER.  The
# delivery has to recognise that header.  The root walk continues forward
# from the latched image rather than restoring the iterator cursor; without
# that latch the resume would skip the consumed item and the sum would come
# out short (19898 for 19900).
try:
    import pypyjit

    pypyjit.set_param("trace_limit=70,threshold=1,function_threshold=1")
except ImportError:
    pass

values = list(range(200))
short = []
for trial in range(30):
    s = sum(values)
    if s != 19900:
        short.append((trial, s))
assert not short, f"sum over list(range(200)) came out wrong, (trial, sum): {short}"

# Nested `readline` with its own FOR_ITER, called from user `__next__`,
# interleaved with a short FOR_ITER. `fbw_foriter_inflight_capture` runs
# after `__next__` returns, so the first `instance_next` consume has an
# empty stash and used to inline that nested body; `popleft` then
# `collect_callee_active_boxes` aborted the enclosing loop and the
# header/body blank reappeared in the payload.
from email.feedparser import FeedParser

SAMPLE = """\
Return-Path: <gkj@gregorykjohnson.com>
X-Original-To: gkj+person@localhost
Delivered-To: gkj+person@localhost
Received: from localhost (localhost [127.0.0.1])
        by andy.gregorykjohnson.com (Postfix) with ESMTP id 356ED9DD17
        for <gkj+person@localhost>; Wed, 13 Jul 2005 17:23:16 -0400 (EDT)
Delivered-To: gkj@sundance.gregorykjohnson.com
Received: from localhost [127.0.0.1]
        by localhost with POP3 (fetchmail-6.2.5)
        for gkj+person@localhost (single-drop); Wed, 13 Jul 2005 17:23:16 -0400 (EDT)
Received: from andy.gregorykjohnson.com (andy.gregorykjohnson.com [64.32.235.228])
        by sundance.gregorykjohnson.com (Postfix) with ESMTP id 5B056316746
        for <gkj@gregorykjohnson.com>; Wed, 13 Jul 2005 17:23:11 -0400 (EDT)
Received: by andy.gregorykjohnson.com (Postfix, from userid 1000)
        id 490CD9DD17; Wed, 13 Jul 2005 17:23:11 -0400 (EDT)
Date: Wed, 13 Jul 2005 17:23:11 -0400
From: "Gregory K. Johnson" <gkj@gregorykjohnson.com>
To: gkj@gregorykjohnson.com
Subject: Sample message
Message-ID: <20050713212311.GC4701@andy.gregorykjohnson.com>
Mime-Version: 1.0
Content-Type: multipart/mixed; boundary="NMuMz9nt05w80d4+"
Content-Disposition: inline
User-Agent: Mutt/1.5.9i


--NMuMz9nt05w80d4+
Content-Type: text/plain; charset=us-ascii
Content-Disposition: inline

This is a sample message.

--
Gregory K. Johnson

--NMuMz9nt05w80d4+
Content-Type: application/octet-stream
Content-Disposition: attachment; filename="text.gz"
Content-Transfer-Encoding: base64

H4sICM2D1UIAA3RleHQAC8nILFYAokSFktSKEoW0zJxUPa7wzJIMhZLyfIWczLzUYj0uAHTs
3FYlAAAA

--NMuMz9nt05w80d4+--
"""
GZIP_B64 = """H4sICM2D1UIAA3RleHQAC8nILFYAokSFktSKEoW0zJxUPa7wzJIMhZLyfIWczLzUYj0uAHTs
3FYlAAAA
"""


def only_iter(x):
    n = 0
    for a in x:
        n += 1
    return n


pairs = [("From", "a@b.c"), ("Content-Type", "text/plain")] * 20
for i in range(80):
    for t in pairs:
        only_iter(t)
    fp = FeedParser()
    fp.feed(SAMPLE)
    msg = fp.close()
    p1 = msg.get_payload()[1].get_payload()
    assert p1 == GZIP_B64, (i, p1[:80] if isinstance(p1, str) else p1)

print("OK")
