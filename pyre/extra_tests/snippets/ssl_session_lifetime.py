# pyre-check: requires-modules=_ssl
# PyPy _stdssl.SSLSession.__new__ retains ssl.ctx as self._ctx. A session
# must keep its Python context alive after the original connection is gone.
import gc
import os
import ssl
import weakref

CERT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "lib-python", "3", "test", "certdata", "keycert3.pem",
)


def session_and_context_ref(version):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    context.minimum_version = context.maximum_version = version
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(CERT)
    incoming, outgoing, peer_incoming, peer_outgoing = (
        ssl.MemoryBIO() for _ in range(4)
    )
    client = context.wrap_bio(incoming, outgoing, server_hostname="localhost")
    server = server_context.wrap_bio(peer_incoming, peer_outgoing, server_side=True)
    done = [False, False]
    for _ in range(32):
        for index, connection in enumerate((client, server)):
            try:
                connection.do_handshake()
                done[index] = True
            except (ssl.SSLWantReadError, ssl.SSLWantWriteError):
                pass
        for output, target in ((outgoing, peer_incoming), (peer_outgoing, incoming)):
            data = output.read()
            if data:
                target.write(data)
        if all(done):
            return client.session, weakref.ref(context)
    raise AssertionError("handshake stalled")


for version in (ssl.TLSVersion.TLSv1_2, ssl.TLSVersion.TLSv1_3):
    session, context_ref = session_and_context_ref(version)
    assert session is not None
    gc.collect()
    gc.collect()
    assert context_ref() is not None
    assert session.time > 0 and session.timeout > 0
    del session
    gc.collect()
    gc.collect()
    assert context_ref() is None

print("OK")
