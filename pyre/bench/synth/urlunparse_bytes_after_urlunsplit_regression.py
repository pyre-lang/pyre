# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=entry-bridge:_coerce_args
# pyre-check: requires-modules=_tokenize,math,unicodedata
# After compiling `_coerce_args` on 5-field `SplitResult` (urlunsplit via RFC
# urljoin traffic), a 6-field `ParseResultBytes.geturl` must still unpack
# through `urlunparse` (`scheme, netloc, url, params, query, fragment,
# coerce_result = _coerce_args(*components)`).
#
# A branch-guard resume that restored the opcode-start `-live-` of an
# inlined `_decode_args` FOR_ITER re-ran `for_iter_next` and dropped the
# sixth field: `_coerce_args(*six)` yielded five strings plus the coerce
# function (expected 7, got 6). Resume at the jump's own `-live-` restores
# the already-consumed item and re-executes only `goto_if_not`.

import sys
from pathlib import Path

_lib = Path(__file__).resolve().parents[3] / "lib-python" / "3"
if _lib.is_dir():
    sys.path.insert(0, str(_lib))

from test.test_urlparse import UrlParseTestCase
from urllib.parse import urlparse

t = UrlParseTestCase()
t.test_RFC1808()
t.test_RFC2368()
t.test_RFC2396()
t.test_RFC2732()
t.test_RFC2732_invalid()
t.test_RFC3986()

sip = b"sip:alice@atlanta.com;maddr=239.255.255.1;ttl=15"
got = urlparse(sip).geturl()
if got != sip:
    print(f"FAIL geturl {got!r} expected {sip!r}")
    raise SystemExit(1)
print("PASS")
