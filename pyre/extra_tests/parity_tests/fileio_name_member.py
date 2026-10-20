# CPython-suite gap: test_fileio.testRepr covers del name, not assignment.
# test_linecache keeps closed FileIO objects across minors and later traces
# w_name; the suite does not assert that slot.
# test_linecache MultiThreadingTest opens text files from 20 threads; a crash is its only signal.
# parity-tests reason: interp_fileio.py interp_member_w w_name drives repr.
# interp_fileio.py descr_init space.setattr(self, "name", w_name) stores a
# young str on an old rawmalloced FileIO; the original w_name is a livevar,
# not a getattr of opened after a collecting call.
# Also covers posix/pyexpat/codecs arguments held across a Python callback.
# parity-env: PYPY_GC_NURSERY=4096
# parity-env: MAJIT_GC_NURSERY_POISON=1

"""FileIO.name is the interp_member_w w_name field.

`del f.name` and `f.name = value` update the typed field, so `repr(f)`
follows the live name or switches to the fd form.
"""

import gc
import io
import os
import sys
import tempfile


path = tempfile.mktemp()
try:
    with open(path, "wb") as fh:
        fh.write(b"x")
    f = io.FileIO(path, "rb")
    print("open-repr", "name=" in repr(f), f.name == path)
    f.name = "renamed"
    print("set-repr", "name='renamed'" in repr(f) or 'name="renamed"' in repr(f), f.name)
    del f.name
    print("del-repr", "fd=" in repr(f), "name=" not in repr(f))
    try:
        f.name
    except AttributeError:
        print("del-get", "AttributeError")
    else:
        print("del-get", "ok")
    f.close()
    print("closed-repr", "[closed]" in repr(f))
finally:
    try:
        os.unlink(path)
    except OSError:
        pass


def closed_fileio_name_stays_a_live_str_across_minors():
    N = 40
    CHURN = 80
    tmpdir = tempfile.mkdtemp()
    files = []
    paths = []
    try:
        for i in range(N):
            path = os.path.join(tmpdir, "f%d.txt" % i)
            with open(path, "w") as fh:
                fh.write("x")
            path = sys.intern(path)
            f = io.FileIO(path, "rb")
            f.close()
            files.append(f)
            paths.append(path)
        junk = []
        for i in range(CHURN):
            junk.append([object() for _ in range(64)])
            if i % 8 == 0:
                gc.collect()
        for f, path in zip(files, paths):
            name = f.name
            if name != path:
                raise AssertionError((name, path))
    finally:
        for path in paths:
            try:
                os.unlink(path)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def threaded_text_open_keeps_fileio_name_live():
    import threading
    NT = 12
    ITERS = 40
    tmpdir = tempfile.mkdtemp()
    names = []
    try:
        for i in range(6):
            path = os.path.join(tmpdir, "t%d.py" % i)
            with open(path, "w") as fh:
                fh.write("import time\nimport system\n")
            names.append(path)
        barrier = threading.Barrier(NT)
        errors = []
        kept = []

        def worker():
            barrier.wait()
            try:
                for _ in range(ITERS):
                    for path in names:
                        with open(path, encoding="utf-8") as fp:
                            if fp.read() != "import time\nimport system\n":
                                raise AssertionError(path)
                            kept.append((fp.buffer.raw, path))
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(NT)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        gc.collect()
        for raw, path in kept:
            if raw.name != path:
                raise AssertionError((raw.name, path))
    finally:
        for path in names:
            try:
                os.unlink(path)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def _churn_small_strings():
    junk = []
    for i in range(3000):
        junk.append(str(i))
    return junk[-1]


class _AllocPath:
    def __init__(self, path):
        self._path = path

    def __fspath__(self):
        _churn_small_strings()
        return self._path


class _AllocBool:
    def __bool__(self):
        _churn_small_strings()
        return False

    def __index__(self):
        _churn_small_strings()
        return 0


def os_open_path_survives_fspath_collection():
    fd = os.open(_AllocPath(os.devnull), os.O_RDONLY, 0o777)
    os.close(fd)
    print("open-fspath ok")


def os_symlink_args_survive_a_collection():
    if not hasattr(os, "symlink"):
        print("symlink-fspath ok")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        dst = os.path.join(tmpdir, "dst")
        try:
            os.symlink(_AllocPath("src"), _AllocPath(dst), _AllocBool())
        except OSError:
            print("symlink-fspath ok")
            return
        if os.readlink(dst) != "src":
            raise AssertionError((os.readlink(dst), "src"))
    print("symlink-fspath ok")


def posix_spawn_keywords_survive_a_collection():
    if not hasattr(os, "posix_spawn"):
        print("posix_spawn-kw", 0)
        return
    if os.path.exists("/bin/true"):
        exe = "/bin/true"
        argv = [exe]
    else:
        exe = sys.executable
        argv = [exe, "-c", "pass"]
    pid = os.posix_spawn(
        _AllocPath(exe),
        argv,
        os.environ,
        setpgroup=None,
        resetids=False,
        setsid=False,
        file_actions=[],
    )
    _wpid, status = os.waitpid(pid, 0)
    if hasattr(os, "waitstatus_to_exitcode"):
        code = os.waitstatus_to_exitcode(status)
    else:
        code = status >> 8
    print("posix_spawn-kw", code)


def expat_parsefile_read_method_survives_a_collection():
    import xml.parsers.expat

    texts = []

    class F:
        def __init__(self):
            self.n = 0

        def read(self, n):
            _churn_small_strings()
            self.n += 1
            if self.n == 1:
                return b"<a>"
            if self.n == 2:
                return b"x" * 50
            if self.n == 3:
                return b"</a>"
            return b""

    parser = xml.parsers.expat.ParserCreate()
    parser.CharacterDataHandler = lambda data: texts.append(data)
    parser.ParseFile(F())
    print("parsefile-read", sum(len(t) for t in texts))


def codec_search_function_survives_a_collection():
    import codecs

    def search(name):
        _churn_small_strings()
        if name == "pyregcutf":
            return codecs.lookup("utf-8")
        return None

    codecs.register(search)
    try:
        first = codecs.lookup("pyregcutf").name
        second = codecs.lookup("pyregcutf").name
        encoded = "x".encode("pyregcutf")
        print("codec-search", first, second, encoded)
    finally:
        codecs.unregister(search)


closed_fileio_name_stays_a_live_str_across_minors()
threaded_text_open_keeps_fileio_name_live()
os_open_path_survives_fspath_collection()
os_symlink_args_survive_a_collection()
posix_spawn_keywords_survive_a_collection()
expat_parsefile_read_method_survives_a_collection()
codec_search_function_survives_a_collection()
print("OK")
