# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=entry-bridge:_rlistdir
# The inner `_rlistdir` next is a residual StopIteration from
# `ll_generatorentry_portal_runner_shim` (several_yields portal). `join`'s
# ForIter then fails a non-exception guard inside the compiled `_rlistdir` entry bridge.
# pyopcode.py FOR_ITER `e.match(StopIteration)` already consumed that publish;
# blackhole.py `_prepare_resume_from_failure` must not re-raise it, or
# generator.py `_leak_stopiteration` reports
# RuntimeError: generator raised StopIteration.
ROUNDS = 400


def _get_sep(path):
    if isinstance(path, bytes):
        return b"/"
    return "/"


def join(a, *p):
    sep = _get_sep(a)
    path = a
    try:
        for b in p:
            if b.startswith(sep) or not path:
                path = b
            elif path.endswith(sep):
                path += b
            else:
                path += sep + b
    except (TypeError, AttributeError):
        raise
    return path


def _join(dirname, basename):
    if not dirname or not basename:
        return dirname or basename
    return join(dirname, basename)


def _ishidden(path):
    return path[0] in (".", b"."[0])


def _isrecursive(pattern):
    return pattern == "**"


class _Entry:
    def __init__(self, name, isdir):
        self.name = name
        self._isdir = isdir

    def is_dir(self, follow_symlinks=True):
        return self._isdir


class _Dir:
    def __init__(self, entries):
        self._entries = entries
        self._i = 0
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed or self._i >= len(self._entries):
            self.close()
            raise StopIteration
        entry = self._entries[self._i]
        self._i += 1
        return entry

    def close(self):
        self._closed = True


TREE = {
    "": ("EF", "ZZZ", "a", ".aa", "aaa", "aab", "sym1"),
    "a": ("D", "bcd"),
    "a/bcd": ("EF", "efg"),
    "a/bcd/efg": ("ha",),
    ".aa": ("G",),
    "aaa": ("zzzF",),
    "aab": ("F",),
}
DIRS = {"", "a", "a/bcd", "a/bcd/efg", ".aa", "aaa", "aab"}


def _norm(path):
    if not path or path == ".":
        return ""
    return path


def _scandir(path):
    key = _norm(path)
    names = TREE.get(key, ())
    entries = []
    for name in names:
        child = name if not key else key + "/" + name
        entries.append(_Entry(name, child in DIRS))
    entries.sort(key=lambda e: (1 if e._isdir else 0, e.name))
    return _Dir(entries)


def _isdir(path):
    return _norm(path) in DIRS


def _iterdir(dirname, dir_fd, dironly):
    try:
        if dirname:
            arg = dirname
        else:
            arg = "."
        try:
            with _scandir(arg) as it:
                for entry in it:
                    try:
                        if not dironly or entry.is_dir():
                            yield entry.name
                    except OSError:
                        pass
        finally:
            pass
    except OSError:
        return


def _listdir(dirname, dir_fd, dironly):
    it = _iterdir(dirname, dir_fd, dironly)
    try:
        return list(it)
    finally:
        it.close()


def _rlistdir(dirname, dir_fd, dironly, include_hidden=False):
    names = _listdir(dirname, dir_fd, dironly)
    for x in names:
        if include_hidden or not _ishidden(x):
            yield x
            path = _join(dirname, x) if dirname else x
            for y in _rlistdir(
                path, dir_fd, dironly, include_hidden=include_hidden
            ):
                yield _join(x, y)


def _glob2(dirname, pattern, dir_fd, dironly, include_hidden=False):
    if not dirname or _isdir(dirname):
        yield pattern[:0]
    yield from _rlistdir(
        dirname, dir_fd, dironly, include_hidden=include_hidden
    )


def _iglob(pathname):
    if _isrecursive(pathname):
        yield from _glob2("", pathname, None, False)
        return
    yield pathname


def glob(pathname):
    it = _iglob(pathname)
    if _isrecursive(pathname[:2]):
        try:
            s = next(it)
            if s:
                yield s
        except StopIteration:
            return
    yield from it


def main():
    first = None
    r = 0
    while r < ROUNDS:
        got = list(glob("**"))
        if first is None:
            first = got
        elif got != first:
            raise AssertionError((r, got))
        r += 1
    print("PASS")


main()
