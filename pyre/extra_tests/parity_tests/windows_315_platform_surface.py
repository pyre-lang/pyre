# pyre-check: platforms=win32
# pyre-check: pypy-diverges: pypy3 implements 3.11, which has no winreg.DeleteTree, no _winapi event-log calls, no _sysconfig, and compares two HKEYs by identity.
# CPython-suite gap: test_winreg, test_winapi, test_time, test_os and test_sysconfig cover these, but CI runs no CPython suite on Windows.
# parity-tests reason: 3.15 adds DeleteTree, handle-value HKEY comparison, the event-log calls and _sysconfig.get_platform,
# normalizes a listed directory before the wildcard, and lets spawnv argv errors other than TypeError through.
# LC_CTYPE is the user locale from startup, and tzname is GetTimeZoneInformation's wide names.
# chmod on a link does not follow it unless asked to.
# Windows select must monitor sockets beyond PyPy's 64-entry SDK fd_set.
import locale
import os
import select
import socket
import stat
import sysconfig
import tempfile
import time
import winreg

import _winapi


def hkeys_compare_by_handle_value():
    key1 = winreg.OpenKey(winreg.HKEY_CURRENT_USER, None)
    key2 = winreg.OpenKey(winreg.HKEY_CURRENT_USER, None)
    key3 = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, None)
    try:
        assert key1 is not key2
        assert key1 == key2 and not (key1 != key2)
        assert key1 != key3 and not (key1 == key3)
        assert key1.__eq__(1) is NotImplemented
        assert key1.__lt__(key2) is NotImplemented
    finally:
        for key in (key1, key2, key3):
            key.Close()
    assert key1 == key3, "closed keys all hold handle 0"


def delete_tree_removes_every_descendant():
    name = r"Software\pyre-parity-delete-tree"
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, name) as root:
        with winreg.CreateKey(root, "sub") as sub:
            winreg.SetValueEx(sub, "value", 0, winreg.REG_SZ, "x")
            winreg.CreateKey(sub, "subsub").Close()
    winreg.DeleteTree(winreg.HKEY_CURRENT_USER, name)
    try:
        winreg.OpenKey(winreg.HKEY_CURRENT_USER, name).Close()
    except OSError:
        pass
    else:
        raise AssertionError("DeleteTree left the key")
    try:
        winreg.DeleteTree()
    except TypeError as e:
        assert str(e) == "DeleteTree expected at least 1 argument, got 0", str(e)
    else:
        raise AssertionError("DeleteTree() accepted no arguments")


def event_source_round_trip():
    handle = _winapi.RegisterEventSource(None, "PyreParityEventSource")
    try:
        assert handle != _winapi.INVALID_HANDLE_VALUE
        for bad in (42, None):
            try:
                _winapi.ReportEvent(handle, _winapi.EVENTLOG_SUCCESS, 1, 1002, bad)
            except TypeError:
                pass
            else:
                raise AssertionError(("ReportEvent accepted", bad))
    finally:
        _winapi.DeregisterEventSource(handle)
    assert (_winapi.EVENTLOG_SUCCESS, _winapi.EVENTLOG_ERROR_TYPE,
            _winapi.EVENTLOG_WARNING_TYPE, _winapi.EVENTLOG_INFORMATION_TYPE,
            _winapi.EVENTLOG_AUDIT_SUCCESS, _winapi.EVENTLOG_AUDIT_FAILURE) == (0, 1, 2, 4, 8, 16)


def startup_sets_the_user_ctype_locale():
    assert locale.setlocale(locale.LC_CTYPE) != "C"
    lt = time.localtime()
    assert lt.tm_zone == time.tzname[lt.tm_isdst], (lt.tm_zone, time.tzname)


def sysconfig_names_the_windows_platform():
    import _sysconfig
    assert _sysconfig.get_platform() in ("win-amd64", "win-arm64", "win32"), _sysconfig.get_platform()
    assert sysconfig.get_platform() == _sysconfig.get_platform()


def chmod_on_a_link_changes_the_link():
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "target")
        link = os.path.join(d, "link")
        open(target, "w").close()
        try:
            os.symlink(target, link)
        except OSError:
            return  # no symlink privilege
        target_mode = os.stat(target).st_mode
        link_mode = os.lstat(link).st_mode
        try:
            os.chmod(link, link_mode & ~stat.S_IWRITE)
            assert os.stat(target).st_mode == target_mode
            assert os.lstat(link).st_mode == link_mode & ~0o222
            os.chmod(link, link_mode | stat.S_IWRITE)
            os.chmod(link, target_mode & ~stat.S_IWRITE, follow_symlinks=True)
            assert os.stat(target).st_mode == target_mode & ~0o222
        finally:
            os.chmod(target, target_mode)
            os.chmod(link, link_mode)


def a_trailing_space_names_the_directory():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "file.txt"), "w").close()
        assert os.listdir(d + " ") == ["file.txt"]
        with os.scandir(d + " ") as entries:
            assert [(e.name, e.path) for e in entries] == [
                ("file.txt", os.path.join(d + " ", "file.txt"))]


def spawnv_lets_a_value_error_through():
    import sys
    try:
        os.spawnv(os.P_NOWAIT, sys.executable, [sys.executable, "embedded\0null"])
    except ValueError:
        pass
    else:
        raise AssertionError("spawnv accepted an embedded null")
    try:
        os.spawnv(os.P_NOWAIT, sys.executable, [sys.executable, 123])
    except TypeError as e:
        assert str(e) == "spawnv() arg 2 must contain only strings", str(e)
    else:
        raise AssertionError("spawnv accepted an int")


def select_monitors_the_last_socket_past_the_sdk_default():
    sockets = []
    try:
        for _ in range(100):
            sockets.extend(socket.socketpair())
        sockets[-1].send(b'x')
        ready, _, _ = select.select(sockets[::2], [], [], 5)
        assert ready == [sockets[-2]], ready
        ready, _, _ = select.select([sockets[-2]] * 512, [], [], 0)
        assert len(ready) == 512, len(ready)
        try:
            select.select([sockets[-2]] * 513, [], [], 0)
        except ValueError as exc:
            assert str(exc) == 'too many file descriptors in select()', str(exc)
        else:
            raise AssertionError('select accepted 513 entries')
    finally:
        for sock in sockets:
            sock.close()


hkeys_compare_by_handle_value()
a_trailing_space_names_the_directory()
spawnv_lets_a_value_error_through()
delete_tree_removes_every_descendant()
event_source_round_trip()
startup_sets_the_user_ctype_locale()
sysconfig_names_the_windows_platform()
chmod_on_a_link_changes_the_link()
select_monitors_the_last_socket_past_the_sdk_default()
print("OK")
