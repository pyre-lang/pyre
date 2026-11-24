# pyre-check: no-cpython
# pyre-check: skip-backends=wasm
# pyre-check: requires-modules=math
# The dump's destination is a file. The wasm guest filesystem is read-only
# and tempfile finds no usable temporary directory
# (`FileNotFoundError: No usable temporary directory found`), so the dump
# has nowhere to be written. Nothing here is about the backend.
# `tempfile` imports `random`, which imports `math`.
import gc
import os
import sys
import tempfile


# inspector.py:HeapDumper writes signed native words, with the root/non-root
# marker between the two graph phases. The translation-time type tables use
# one text line per member and expose the header-id mapping separately.
typeids_z = gc.get_typeids_z()
typeids_list = gc.get_typeids_list()
assert isinstance(typeids_z, bytes)
assert len(typeids_z) > 2
assert isinstance(typeids_list, list) and len(typeids_list) > 1
assert isinstance(typeids_list[0], int)
assert isinstance(typeids_list[-1], int)

word_size = 8 if sys.maxsize > 2**32 else 4
marker = b"\x00" * (word_size * 3) + (-1).to_bytes(
    word_size, sys.byteorder, signed=True
)


def check_dump(data):
    assert len(data) % word_size == 0
    assert marker in data


# A pid-derived `/tmp` name is predictable and `open(..., "wb")` follows
# symlinks, so another local process could aim the truncation at a file of its
# choosing. A private directory removes both.
with tempfile.TemporaryDirectory(prefix="pyre-gc-dump-") as dump_dir:
    # app_referents.py:44-49 — the descriptor arm. `os.open` is descriptor-
    # backed on every target, so this shape reaches the dump writer everywhere.
    fd_path = os.path.join(dump_dir, "byfd.bin")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
    fd = os.open(fd_path, flags, 0o600)
    try:
        assert gc.dump_rpy_heap(fd) is None
    finally:
        os.close(fd)
    with open(fd_path, "rb") as dump_file:
        check_dump(dump_file.read())

    # The remaining two arms both take their descriptor from the builtin
    # `open`: `app_referents.py:23` for the filename arm and `:48` for the file
    # object.
    dump_path = os.path.join(dump_dir, "heap.bin")
    with open(dump_path, "wb") as dump_file:
        assert gc.dump_rpy_heap(dump_file) is None
    with open(dump_path, "rb") as dump_file:
        check_dump(dump_file.read())

    # app_referents.py:22-42 — the filename arm opens and truncates the dump
    # itself, then materializes typeids.txt/typeids.lst beside it when they are
    # absent. Neither sidecar is written for a file object or a descriptor, so
    # this is the only call shape that reaches them.
    named_dir = os.path.join(dump_dir, "named")
    os.mkdir(named_dir)
    named_path = os.path.join(named_dir, "heap.bin")
    assert gc.dump_rpy_heap(named_path) is None
    with open(named_path, "rb") as dump_file:
        check_dump(dump_file.read())
    typeids_txt = os.path.join(named_dir, "typeids.txt")
    typeids_lst = os.path.join(named_dir, "typeids.lst")
    with open(typeids_txt, "rb") as sidecar:
        assert len(sidecar.read()) > 2
    with open(typeids_lst, "r") as sidecar:
        lines = sidecar.read().splitlines()
    assert len(lines) == len(typeids_list)
    assert [int(line) for line in lines] == typeids_list

    # The second call finds both sidecars in place and leaves them untouched.
    with open(typeids_txt, "ab") as sidecar:
        sidecar.write(b"sentinel\n")
    assert gc.dump_rpy_heap(named_path) is None
    with open(typeids_txt, "rb") as sidecar:
        assert sidecar.read().endswith(b"sentinel\n")

print("OK")
