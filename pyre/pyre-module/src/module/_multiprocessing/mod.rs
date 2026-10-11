//! _multiprocessing module — PyPy: `pypy/module/_multiprocessing/`.
//!
//! Exposes `SemLock(kind, value, maxvalue, name, unlink)` and
//! `sem_unlink(name)`, plus the three socket calls `connection.py` reaches for
//! on Windows.  POSIX calls `sem_open` and the other `external()` symbols.
//! Windows stays on `CreateSemaphoreW`. Either path needs `host_env`, so
//! other platforms get an empty module and `import _multiprocessing`
//! still succeeds.
//!
//! `W_SemLock`'s fields (`interp_semaphore.py`) live in the instance
//! dict rather than a typed payload: `handle`, `kind`, `maxvalue` and `name`
//! are the values behind the `GetSetProperty`s of
//! `interp_semaphore.py`, and `count`/`last_tid` are the recursion
//! bookkeeping `_ismine` reads.  A dict-backed field is also readable as a
//! plain attribute, which is wider than the typedef; the alternative — a
//! handle-keyed side table — has no upstream counterpart.
//!
//! POSIX semaphores are `interp_semaphore.py external()` (`sem_open` and the
//! rest). Windows stays on `CreateSemaphore` / `WaitForSingleObject`.

#[cfg(all(any(unix, windows), feature = "host_env"))]
use pyre_object::*;

#[cfg(all(windows, feature = "host_env"))]
use rustpython_host_env::multiprocessing as host_mp;

/// `interp_semaphore.py external()` — POSIX `sem_*`.
#[cfg(all(unix, feature = "host_env"))]
mod ll {
    use majit_rlib::rffi::{INT, RFFI_SAVE_ERRNO, UINT};

    // Darwin links nothing; every other POSIX target links `rt` (`libraries`).
    #[cfg(target_vendor = "apple")]
    majit_rlib::rffi::external_compilation_info! {
        const ECI = {
            includes: ["sys/time.h", "limits.h", "semaphore.h"],
        };
    }
    #[cfg(not(target_vendor = "apple"))]
    majit_rlib::rffi::external_compilation_info! {
        const ECI = {
            includes: ["sys/time.h", "limits.h", "semaphore.h"],
            libraries: ["rt"],
        };
    }

    // libc private build-script cfgs (`gnu_time_bits64`, `musl_redir_time64`)
    // select 32-bit redirects (`__sem_timedwait64`, `__gettimeofday64`).
    // Native targets are 64-bit, so those `link_name`s are not copied.

    // `sem_open` is `sem_t *sem_open(const char *, int, ...)`. On Apple arm64
    // the anonymous arguments are passed on the stack; a fixed 4-argument
    // prototype returns EINVAL. `natural_arity = 2` is that variadic prototype.
    // `interp_semaphore.external` (`_sem_open`) has no `macro` and no
    // `natural_arity`.
    majit_rlib::rffi::llexternal!(
        pub(super) _sem_open = "sem_open",
        [*const libc::c_char, INT, INT, UINT],
        *mut libc::sem_t,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        natural_arity = 2
    );
    majit_rlib::rffi::llexternal!(
        pub(super) _sem_close_no_errno = "sem_close",
        [*mut libc::sem_t],
        INT,
        compilation_info = ECI,
        releasegil = false
    );
    majit_rlib::rffi::llexternal!(
        pub(super) _sem_close = "sem_close",
        [*mut libc::sem_t],
        INT,
        compilation_info = ECI,
        releasegil = false,
        save_err = RFFI_SAVE_ERRNO
    );
    majit_rlib::rffi::llexternal!(
        pub(super) _sem_unlink = "sem_unlink",
        [*const libc::c_char],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    majit_rlib::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "sem_wait$UNIX2003"
        )]
        pub(super) _sem_wait = "sem_wait",
        [*mut libc::sem_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    majit_rlib::rffi::llexternal!(
        pub(super) _sem_trywait = "sem_trywait",
        [*mut libc::sem_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    majit_rlib::rffi::llexternal!(
        pub(super) _sem_post = "sem_post",
        [*mut libc::sem_t],
        INT,
        compilation_info = ECI,
        releasegil = false,
        save_err = RFFI_SAVE_ERRNO
    );
    // Darwin has no `sem_getvalue` (`HAVE_BROKEN_SEM_GETVALUE`).
    #[cfg(not(target_vendor = "apple"))]
    majit_rlib::rffi::llexternal!(
        pub(super) _sem_getvalue = "sem_getvalue",
        [*mut libc::sem_t, *mut INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    // Darwin has no `sem_timedwait`. The substitute is `_sem_timedwait_save`.
    #[cfg(not(target_vendor = "apple"))]
    majit_rlib::rffi::llexternal!(
        pub(super) _sem_timedwait = "sem_timedwait",
        [*mut libc::sem_t, *const libc::timespec],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    // `gettimeofday`'s second argument is `*mut timezone` on glibc, uClibc,
    // Android, FreeBSD, DragonFly, OpenBSD, Redox, Hurd and L4Re, and
    // `*mut c_void` elsewhere. NetBSD's symbol is `__gettimeofday50`.
    #[cfg(any(
        all(target_os = "linux", not(target_env = "musl")),
        target_os = "android",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "openbsd",
        target_os = "redox",
        target_os = "hurd",
        target_os = "l4re",
    ))]
    majit_rlib::rffi::llexternal!(
        pub(super) _gettimeofday = "gettimeofday",
        [*mut libc::timeval, *mut libc::timezone],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    #[cfg(not(any(
        all(target_os = "linux", not(target_env = "musl")),
        target_os = "android",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "openbsd",
        target_os = "redox",
        target_os = "hurd",
        target_os = "l4re",
    )))]
    majit_rlib::rffi::llexternal!(
        #[cfg_attr(target_os = "netbsd", link_name = "__gettimeofday50")]
        pub(super) _gettimeofday = "gettimeofday",
        [*mut libc::timeval, *mut libc::c_void],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );

    /// `interp_semaphore.py _sem_timedwait_save`. `sem_trywait`, then a
    /// `select` sleep whose delay grows by 1000µs and caps at 20000µs.
    /// The deadline's nanoseconds are compared with `gettimeofday`'s
    /// microseconds, and the difference mixes those units; both are upstream.
    #[cfg(target_vendor = "apple")]
    pub(super) unsafe fn _sem_timedwait_save(
        sem: *mut libc::sem_t,
        deadline: libc::timespec,
    ) -> INT {
        let mut delay: i64 = 0;
        loop {
            if unsafe { _sem_trywait(sem) } == 0 {
                return 0;
            }
            if majit_rlib::rposix::get_saved_errno() != libc::EAGAIN {
                return -1;
            }
            let mut now = libc::timeval {
                tv_sec: 0,
                tv_usec: 0,
            };
            if unsafe { _gettimeofday(&mut now, core::ptr::null_mut()) } < 0 {
                return -1;
            }
            let c_tv_sec = deadline.tv_sec as i64;
            let c_tv_nsec = deadline.tv_nsec as i64;
            let now_sec = now.tv_sec as i64;
            let now_usec = now.tv_usec as i64;
            if c_tv_sec < now_sec || (c_tv_sec == now_sec && c_tv_nsec <= now_usec) {
                majit_rlib::rposix::set_saved_errno(libc::ETIMEDOUT);
                return -1;
            }
            let difference = (c_tv_sec - now_sec) * 1_000_000 + (c_tv_nsec - now_usec);
            if delay > 20_000 {
                delay = 20_000;
            }
            if delay > difference {
                delay = difference;
            }
            delay += 1000;
            let mut tv = libc::timeval {
                tv_sec: (delay / 1_000_000) as _,
                tv_usec: (delay % 1_000_000) as _,
            };
            // `select` is already a `save_err` external. Do not release again.
            if unsafe {
                majit_rlib::_rsocket_rffi::select(
                    0,
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    core::ptr::null_mut(),
                    &mut tv,
                )
            } < 0
            {
                return -1;
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn sem_open_trywait_post_close_unlink() {
            let name =
                std::ffi::CString::new(format!("/pyre-rffi-{}", std::process::id())).unwrap();
            struct Cleanup(std::ffi::CString, *mut libc::sem_t);
            impl Drop for Cleanup {
                fn drop(&mut self) {
                    unsafe {
                        if !self.1.is_null() && self.1 != libc::SEM_FAILED {
                            _sem_close_no_errno(self.1);
                        }
                        _sem_unlink(self.0.as_ptr());
                    }
                }
            }
            unsafe { _sem_unlink(name.as_ptr()) };
            let sem = unsafe { _sem_open(name.as_ptr(), libc::O_CREAT | libc::O_EXCL, 0o600, 0) };
            let mut cleanup = Cleanup(name, sem);
            assert_ne!(
                sem,
                libc::SEM_FAILED,
                "sem_open errno {}",
                majit_rlib::rposix::get_saved_errno()
            );
            assert_eq!(unsafe { _sem_trywait(sem) }, -1);
            assert_eq!(majit_rlib::rposix::get_saved_errno(), libc::EAGAIN);
            assert_eq!(unsafe { _sem_post(sem) }, 0);
            assert_eq!(unsafe { _sem_trywait(sem) }, 0);
            assert_eq!(unsafe { _sem_close(sem) }, 0);
            cleanup.1 = core::ptr::null_mut();
        }
    }
}

/// The platform's semaphore, as the instance stores it: an integer `handle`
/// that `_rebuild` takes back.  Both spellings are raw pointers, so the
/// round trip through `usize` is the same on either.
#[cfg(all(unix, feature = "host_env"))]
type SemRaw = *mut libc::sem_t;
#[cfg(all(windows, feature = "host_env"))]
type SemRaw = host_mp::RawHandle;

/// The Win32 code the last call left behind, as an `OSError` carrying it in
/// `.winerror` (`PyErr_SetExcFromWindowsErr`).
#[cfg(all(windows, feature = "host_env"))]
fn last_windows_error() -> pyre_interpreter::PyError {
    windows_error(std::io::Error::last_os_error().raw_os_error().unwrap_or(0))
}

#[cfg(all(windows, feature = "host_env"))]
fn windows_error(winerror: i32) -> pyre_interpreter::PyError {
    pyre_interpreter::PyError::os_error_win32_syscall2(winerror, PY_NULL, PY_NULL)
}

/// `interp_semaphore.py RECURSIVE_MUTEX, SEMAPHORE = range(2)`.
#[cfg(all(any(unix, windows), feature = "host_env"))]
const RECURSIVE_MUTEX: i64 = 0;
#[cfg(all(any(unix, windows), feature = "host_env"))]
const SEMAPHORE: i64 = 1;

#[cfg(all(any(unix, windows), feature = "host_env"))]
fn semlock_get_handle(obj: PyObjectRef) -> SemRaw {
    let d = pyre_interpreter::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return core::ptr::null_mut();
    }
    if let Some(v) = unsafe { w_dict_getitem_str(d, "_handle") }
        && unsafe { is_int(v) }
    {
        return unsafe { w_int_get_value(v) } as usize as SemRaw;
    }
    core::ptr::null_mut()
}

/// Read one of the integer fields of `interp_semaphore.py W_SemLock`.  A missing
/// or non-int entry reads as 0, which only happens on an instance whose dict a
/// caller has torn up.
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn semlock_get_i64(obj: PyObjectRef, key: &str) -> i64 {
    let d = pyre_interpreter::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return 0;
    }
    match unsafe { w_dict_getitem_str(d, key) } {
        Some(v) if unsafe { is_int(v) } => unsafe { w_int_get_value(v) },
        _ => 0,
    }
}

/// Write one of those fields.  Boxing the value and materialising the dict can
/// both collect, so every operand is published and re-read from the shadow
/// stack at the store, exactly as `semlock_instance` does.
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn semlock_set_i64(obj: PyObjectRef, key: &str, value: i64) {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::pin_roots(&[obj]);
    let boxed_slot = obj_slot + 1;
    let _ = pyre_object::gc_roots::pin_root(w_int_new(value));
    let dict = pyre_interpreter::baseobjspace::getdict_native(
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
    );
    if dict.is_null() {
        return;
    }
    let dict_slot = boxed_slot + 1;
    let _ = pyre_object::gc_roots::pin_root(dict);
    unsafe {
        w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            key,
            pyre_object::gc_roots::shadow_stack_get(boxed_slot),
        )
    };
}

/// `interp_semaphore.py W_SemLock._ismine`.
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn semlock_ismine(mut obj: PyObjectRef) -> bool {
    pyre_object::with_roots!(obj => semlock_get_i64(obj, "count")) > 0
        && pyre_interpreter::module::thread::current_ident() == semlock_get_i64(obj, "last_tid")
}

#[cfg(all(unix, feature = "host_env"))]
fn sem_oserror(ctx: &str) -> pyre_interpreter::PyError {
    pyre_interpreter::PyError::os_error_with_errno(majit_rlib::rposix::get_saved_errno(), ctx)
}

#[cfg(all(unix, feature = "host_env"))]
fn sem_c_name(name: &str) -> Result<std::ffi::CString, pyre_interpreter::PyError> {
    std::ffi::CString::new(name)
        .map_err(|_| pyre_interpreter::PyError::value_error("embedded null character"))
}

/// `sem_post`. `releasegil=False`, so a successful release does not drop the GIL.
#[cfg(all(unix, feature = "host_env"))]
fn semlock_post(handle: SemRaw) -> Result<(), pyre_interpreter::PyError> {
    if unsafe { ll::_sem_post(handle) } < 0 {
        Err(sem_oserror("sem_post"))
    } else {
        Ok(())
    }
}

/// `sem_getvalue`. Not built on darwin (`HAVE_BROKEN_SEM_GETVALUE`); the
/// `sem_trywait` fallbacks run instead. A negative waiter count clamps to 0.
#[cfg(all(unix, feature = "host_env", not(target_vendor = "apple")))]
fn semlock_getvalue(handle: SemRaw) -> Result<i64, pyre_interpreter::PyError> {
    let mut sval: libc::c_int = 0;
    if unsafe { ll::_sem_getvalue(handle, &mut sval) } < 0 {
        return Err(sem_oserror("sem_getvalue"));
    }
    let val = i64::from(sval);
    Ok(if val < 0 { 0 } else { val })
}

/// `semlock_iszero`. Darwin (`HAVE_BROKEN_SEM_GETVALUE`) probes with
/// `sem_trywait` and posts back. EINTR is an error here, not a retry.
#[cfg(all(unix, feature = "host_env"))]
fn semlock_iszero(handle: SemRaw) -> Result<bool, pyre_interpreter::PyError> {
    #[cfg(target_vendor = "apple")]
    {
        if unsafe { ll::_sem_trywait(handle) } == 0 {
            semlock_post(handle)?;
            return Ok(false);
        }
        let errno = majit_rlib::rposix::get_saved_errno();
        if errno == libc::EAGAIN {
            Ok(true)
        } else {
            Err(pyre_interpreter::PyError::os_error_with_errno(
                errno,
                "sem_trywait",
            ))
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        Ok(semlock_getvalue(handle)? == 0)
    }
}

/// The value the semaphore currently holds, for `_get_value`.
#[cfg(all(unix, feature = "host_env"))]
fn semlock_value(handle: SemRaw) -> Result<i64, pyre_interpreter::PyError> {
    // `semlock_getvalue`: `HAVE_BROKEN_SEM_GETVALUE` raises.
    #[cfg(target_vendor = "apple")]
    {
        let _ = handle;
        Err(pyre_interpreter::PyError::not_implemented(
            "sem_getvalue is not implemented on this system",
        ))
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        semlock_getvalue(handle)
    }
}

/// `semaphore.c semlock_getvalue`'s Windows arm: take the count down by one
/// and give it straight back, which is the only way to read it.  A wait that
/// times out is a semaphore holding nothing.
#[cfg(all(windows, feature = "host_env"))]
fn semlock_value(handle: SemRaw) -> Result<i64, pyre_interpreter::PyError> {
    host_mp::get_semaphore_value(handle)
        .map(i64::from)
        .map_err(|()| last_windows_error())
}

/// `semaphore.c semlock_iszero`'s Windows arm.
#[cfg(all(windows, feature = "host_env"))]
fn semlock_iszero(handle: SemRaw) -> Result<bool, pyre_interpreter::PyError> {
    let status = host_mp::wait_for_single_object(handle, 0);
    if status == host_mp::wait_object_0() {
        host_mp::release_semaphore(handle).map_err(|code| windows_error(code as i32))?;
        return Ok(false);
    }
    if status == host_mp::wait_timeout() {
        return Ok(true);
    }
    Err(last_windows_error())
}

/// `semaphore.c semlock_acquire`'s Windows arm, with the recursion
/// bookkeeping left to the caller as on the POSIX side.
///
/// The wait runs in slices rather than as one call: it is the runtime's own
/// wait, which a Python signal does not interrupt, and the event handshake
/// `semaphore.c` interrupts it with needs the signal module to own a Win32
/// event.  Between slices a pending signal is delivered, which is how every
/// other blocking call in the interpreter answers a Ctrl-C.
#[cfg(all(windows, feature = "host_env"))]
fn semlock_acquire(
    handle: SemRaw,
    block: bool,
    timeout: Option<f64>,
) -> Result<bool, pyre_interpreter::PyError> {
    const SLICE_MS: u32 = 100;
    // `None` waits forever; `Some(0)` is the non-blocking poll.
    let mut remaining = match (block, timeout) {
        (false, _) => Some(0),
        (true, None) => None,
        (true, Some(seconds)) => {
            // `interp_semaphore.py:268-275` — a negative timeout is a poll,
            // and one at half of `INFINITE` (about 25 days) is refused rather
            // than saturated, so no wait silently becomes a different one.
            let msecs = (seconds * 1000.0).max(0.0);
            if msecs >= 0.5 * f64::from(u32::MAX) {
                return Err(pyre_interpreter::PyError::overflow_error(
                    "timeout is too large",
                ));
            }
            Some((msecs + 0.5) as u32)
        }
    };
    loop {
        let slice = remaining.map_or(SLICE_MS, |left| left.min(SLICE_MS));
        let status = {
            let _blocked = pyre_interpreter::module::thread::before_external_block();
            host_mp::wait_for_single_object(handle, slice)
        };
        // `interp_semaphore.py:311-315` — the wait has taken the count, so it
        // is reported before anything that can raise.  A signal pending at
        // this moment is delivered at the next checkpoint like any other;
        // raising here would consume the semaphore without handing it over.
        if status == host_mp::wait_object_0() {
            return Ok(true);
        }
        if status != host_mp::wait_timeout() {
            return Err(last_windows_error());
        }
        pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
        if let Some(left) = &mut remaining {
            *left -= slice;
            if *left == 0 {
                return Ok(false);
            }
        }
    }
}

/// `semaphore.c semlock_release`'s Windows arm.  Neither the kind nor the
/// maximum is read: `ReleaseSemaphore` enforces the maximum itself, and the
/// refusal it reports is the one the POSIX arm makes out of `sem_getvalue`.
#[cfg(all(windows, feature = "host_env"))]
fn semlock_release(
    handle: SemRaw,
    kind: i64,
    maxvalue: i64,
) -> Result<(), pyre_interpreter::PyError> {
    let _ = (kind, maxvalue);
    host_mp::release_semaphore(handle).map_err(|code| {
        if code == rustpython_host_env::errno::errors::ERROR_TOO_MANY_POSTS {
            pyre_interpreter::PyError::value_error("semaphore or lock released too many times")
        } else {
            windows_error(code as i32)
        }
    })
}

/// `create_semaphore`: `sem_open(name, O_CREAT|O_EXCL, 0600, value)`.
/// The name is passed through; a missing leading `/` is not added. `unlink`
/// then calls `sem_unlink` and the instance reports no name. A failed unlink
/// closes the semaphore with `_sem_close_no_errno` because the caller never
/// receives the handle. There is no finalizer (`delete_semaphore` is upstream
/// and this object still has no typed payload).
#[cfg(all(unix, feature = "host_env"))]
fn semlock_create(
    name: &str,
    value: i64,
    maxvalue: i64,
    unlink: bool,
) -> Result<(SemRaw, Option<String>), pyre_interpreter::PyError> {
    let _ = maxvalue;
    let c_name = sem_c_name(name)?;
    let kept_name = if unlink { None } else { Some(name.to_owned()) };
    let handle = unsafe {
        ll::_sem_open(
            c_name.as_ptr(),
            libc::O_CREAT | libc::O_EXCL,
            0o600,
            value as libc::c_uint,
        )
    };
    if handle == libc::SEM_FAILED {
        return Err(sem_oserror("sem_open failed"));
    }
    if unlink && unsafe { ll::_sem_unlink(c_name.as_ptr()) } < 0 {
        unsafe { ll::_sem_close_no_errno(handle) };
        return Err(sem_oserror("sem_unlink failed"));
    }
    Ok((handle, kept_name))
}

/// A Windows semaphore is anonymous — `CreateSemaphoreW` takes the two counts
/// and nothing else, and it is the handle that travels to another process —
/// so the name is only what the instance reports back.  A `value` above
/// `maxvalue` is the call's own `ERROR_INVALID_PARAMETER`.
#[cfg(all(windows, feature = "host_env"))]
fn semlock_create(
    name: &str,
    value: i64,
    maxvalue: i64,
    unlink: bool,
) -> Result<(SemRaw, Option<String>), pyre_interpreter::PyError> {
    let (Ok(value), Ok(maxvalue)) = (i32::try_from(value), i32::try_from(maxvalue)) else {
        return Err(pyre_interpreter::PyError::overflow_error(
            "SemLock() value out of range",
        ));
    };
    let handle = host_mp::SemHandle::create(value, maxvalue)
        .map_err(|error| windows_error(error.raw_os_error().unwrap_or(0)))?;
    let raw = handle.as_raw();
    // As on the POSIX arm: the drop closes the handle, and the Python object
    // owns it from here.
    core::mem::forget(handle);
    Ok((raw, (!unlink).then(|| name.to_owned())))
}

/// The semaphore `_rebuild` reattaches to.
#[cfg(all(unix, feature = "host_env"))]
fn semlock_rebuild_raw(
    w_handle: PyObjectRef,
    name: Option<&str>,
) -> Result<SemRaw, pyre_interpreter::PyError> {
    match name {
        // `W_SemLock.rebuild`: a name reopens with `sem_open(name, 0, 0600, 0)`
        // and ignores `w_handle`.
        Some(name) => {
            let c_name = sem_c_name(name)?;
            let handle = unsafe { ll::_sem_open(c_name.as_ptr(), 0, 0o600, 0) };
            if handle == libc::SEM_FAILED {
                return Err(sem_oserror("sem_open failed"));
            }
            Ok(handle)
        }
        // `handle_w`: the integer stored on the instance.
        None => Ok(pyre_interpreter::baseobjspace::int_w(w_handle)? as usize as SemRaw),
    }
}

/// There is no name to reopen through on Windows, so the handle is what
/// `_rebuild` reattaches to whether or not a name came with it.
#[cfg(all(windows, feature = "host_env"))]
fn semlock_rebuild_raw(
    w_handle: PyObjectRef,
    name: Option<&str>,
) -> Result<SemRaw, pyre_interpreter::PyError> {
    let _ = name;
    Ok(pyre_interpreter::baseobjspace::int_w(w_handle)? as usize as SemRaw)
}

/// `semlock_acquire` builds the deadline the way `semlock_acquire` does:
/// `int(timeout)` truncates toward 0, `int(1e9 * (timeout - sec) + 0.5)`,
/// then `gettimeofday`, then carry nanoseconds. A non-finite or overflowing
/// timeout is `OverflowError`. A negative timeout is not clamped to zero.
#[cfg(all(unix, feature = "host_env"))]
fn sem_deadline(timeout: f64) -> Result<libc::timespec, pyre_interpreter::PyError> {
    if !timeout.is_finite() || timeout > i64::MAX as f64 || timeout < i64::MIN as f64 {
        return Err(pyre_interpreter::PyError::overflow_error(
            "timeout is too large",
        ));
    }
    let sec = timeout as i64;
    let nsec_f = 1e9 * (timeout - sec as f64) + 0.5;
    if !nsec_f.is_finite() || nsec_f > i64::MAX as f64 || nsec_f < i64::MIN as f64 {
        return Err(pyre_interpreter::PyError::overflow_error(
            "timeout is too large",
        ));
    }
    let nsec = nsec_f as i64;
    let mut now = libc::timeval {
        tv_sec: 0,
        tv_usec: 0,
    };
    if unsafe { ll::_gettimeofday(&mut now, core::ptr::null_mut()) } < 0 {
        return Err(sem_oserror("gettimeofday failed"));
    }
    let dl_nsec = (now.tv_usec as i64)
        .checked_mul(1000)
        .and_then(|usec| usec.checked_add(nsec))
        .ok_or_else(|| pyre_interpreter::PyError::overflow_error("timeout is too large"))?;
    // `c_tv_nsec / 1000000000` and `%` in `semlock_acquire` are floor division.
    let carry = if dl_nsec % 1_000_000_000 < 0 {
        dl_nsec / 1_000_000_000 - 1
    } else {
        dl_nsec / 1_000_000_000
    };
    let dl_sec = (now.tv_sec as i64)
        .checked_add(sec)
        .and_then(|sum| sum.checked_add(carry))
        .ok_or_else(|| pyre_interpreter::PyError::overflow_error("timeout is too large"))?;
    let dl_nsec = if dl_nsec % 1_000_000_000 < 0 {
        dl_nsec % 1_000_000_000 + 1_000_000_000
    } else {
        dl_nsec % 1_000_000_000
    };
    Ok(libc::timespec {
        tv_sec: dl_sec as _,
        tv_nsec: dl_nsec as _,
    })
}

/// `semlock_acquire` — the platform wait alone. `last_tid` / `count` stay
/// with the caller, which updates them on the success return. EINTR runs
/// pending signals and retries the whole call (the Darwin poll delay is local
/// to `_sem_timedwait_save`, so it starts again). EAGAIN and ETIMEDOUT return
/// false. The externals release the GIL themselves.
#[cfg(all(unix, feature = "host_env"))]
fn semlock_acquire(
    handle: SemRaw,
    block: bool,
    timeout: Option<f64>,
) -> Result<bool, pyre_interpreter::PyError> {
    let deadline = if block {
        match timeout {
            Some(timeout) => Some(sem_deadline(timeout)?),
            None => None,
        }
    } else {
        None
    };
    let op = if !block {
        "sem_trywait"
    } else if deadline.is_none() {
        "sem_wait"
    } else {
        "sem_timedwait"
    };
    loop {
        let rc = if !block {
            unsafe { ll::_sem_trywait(handle) }
        } else if let Some(deadline) = deadline {
            #[cfg(target_vendor = "apple")]
            {
                unsafe { ll::_sem_timedwait_save(handle, deadline) }
            }
            #[cfg(not(target_vendor = "apple"))]
            {
                unsafe { ll::_sem_timedwait(handle, &deadline) }
            }
        } else {
            unsafe { ll::_sem_wait(handle) }
        };
        if rc == 0 {
            pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
            return Ok(true);
        }
        let errno = majit_rlib::rposix::get_saved_errno();
        if errno == libc::EINTR {
            pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
            continue;
        }
        if errno == libc::EAGAIN || errno == libc::ETIMEDOUT {
            return Ok(false);
        }
        return Err(pyre_interpreter::PyError::os_error_with_errno(errno, op));
    }
}

/// `interp_semaphore.py semlock_release`.
#[cfg(all(unix, feature = "host_env"))]
fn semlock_release(
    handle: SemRaw,
    kind: i64,
    maxvalue: i64,
) -> Result<(), pyre_interpreter::PyError> {
    if kind == RECURSIVE_MUTEX {
        return semlock_post(handle);
    }
    #[cfg(target_vendor = "apple")]
    {
        // `HAVE_BROKEN_SEM_GETVALUE`: only the maxvalue == 1 case can be
        // checked properly.
        if maxvalue == 1 {
            if unsafe { ll::_sem_trywait(handle) } == 0 {
                // it was not locked, so undo the wait and raise
                semlock_post(handle)?;
                return Err(pyre_interpreter::PyError::value_error(
                    "semaphore or lock released too many times",
                ));
            }
            let errno = majit_rlib::rposix::get_saved_errno();
            if errno != libc::EAGAIN {
                return Err(pyre_interpreter::PyError::os_error_with_errno(
                    errno,
                    "sem_trywait",
                ));
            }
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        // This check is not an absolute guarantee that the semaphore does not
        // rise above maxvalue.
        if semlock_getvalue(handle)? >= maxvalue {
            return Err(pyre_interpreter::PyError::value_error(
                "semaphore or lock released too many times",
            ));
        }
    }
    semlock_post(handle)
}

/// `interp_semaphore.py W_SemLock.acquire` — shared by `acquire` and
/// `__enter__`, which the class methods cannot reach through each other.
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn w_semlock_acquire(
    self_obj: PyObjectRef,
    block: bool,
    timeout: Option<f64>,
) -> Result<bool, pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::pin_roots(&[self_obj]);
    // Every field helper can collect — `semlock_get_i64` materialises the
    // instance dict, `semlock_set_i64` boxes the value — so the receiver is
    // read back from its slot at each use rather than kept in a local. Only
    // `pin_roots` normalises a forwarded pointer on the way in; the reads do
    // not, so a stale binding would have them consult a moved object's dict.
    let me = || pyre_object::gc_roots::shadow_stack_get(self_slot);
    // check whether we already own the lock
    if semlock_get_i64(me(), "kind") == RECURSIVE_MUTEX && semlock_ismine(me()) {
        // `semlock_get_i64` collects: read the count before fetching the receiver.
        let count = semlock_get_i64(me(), "count");
        semlock_set_i64(me(), "count", count + 1);
        return Ok(true);
    }
    let handle = semlock_get_handle(me());
    if handle.is_null() {
        return Err(pyre_interpreter::PyError::value_error(
            "SemLock handle is null",
        ));
    }
    let got = semlock_acquire(handle, block, timeout)?;
    if got {
        // `interp_semaphore.py:512-516` — these steps need to be as close as
        // possible to acquiring the semlock for `_ismine` to support multiple
        // threads.  The wait can run signal handlers, so the receiver comes
        // back off the shadow stack, and again after the `last_tid` store
        // boxes its value.
        semlock_set_i64(
            me(),
            "last_tid",
            pyre_interpreter::module::thread::current_ident(),
        );
        let count = semlock_get_i64(me(), "count");
        semlock_set_i64(me(), "count", count + 1);
    }
    Ok(got)
}

/// `interp_semaphore.py W_SemLock.release` — shared by `release` and
/// `__exit__`.
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn w_semlock_release(self_obj: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::pin_roots(&[self_obj]);
    // As in `w_semlock_acquire`: every field helper can collect, so the
    // receiver is read back from its slot at each use.
    let me = || pyre_object::gc_roots::shadow_stack_get(self_slot);
    let kind = semlock_get_i64(me(), "kind");
    if kind == RECURSIVE_MUTEX {
        if !semlock_ismine(me()) {
            return Err(pyre_interpreter::PyError::new(
                pyre_interpreter::error::PyErrorKind::AssertionError,
                "attempt to release recursive lock not owned by thread",
            ));
        }
        let count = semlock_get_i64(me(), "count");
        if count > 1 {
            semlock_set_i64(me(), "count", count - 1);
            return Ok(());
        }
    }
    let handle = semlock_get_handle(me());
    if handle.is_null() {
        return Err(pyre_interpreter::PyError::value_error(
            "SemLock handle is null",
        ));
    }
    semlock_release(handle, kind, semlock_get_i64(me(), "maxvalue"))?;
    // `semlock_get_i64` collects: read the count before fetching the receiver.
    let count = semlock_get_i64(me(), "count");
    semlock_set_i64(me(), "count", count - 1);
    Ok(())
}

#[cfg(all(any(unix, windows), feature = "host_env"))]
fn semlock_instance(
    w_subtype: PyObjectRef,
    raw: SemRaw,
    kind: i64,
    maxvalue: i64,
    kept_name: Option<String>,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let obj = w_instance_new(w_subtype);
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    // `getdict_native` materialises the instance dict, so it can collect and
    // move `obj`; read the receiver back from its slot the way every store
    // below does, rather than handing over the pre-pin copy.
    let dict = pyre_interpreter::baseobjspace::getdict_native(
        pyre_object::gc_roots::shadow_stack_get(root_base),
    );
    if dict.is_null() {
        return Err(pyre_interpreter::PyError::runtime_error(
            "SemLock instance has no storage",
        ));
    }
    let _ = pyre_object::gc_roots::pin_root(dict);
    macro_rules! store {
        ($name:literal, $value:expr) => {{
            let value = $value;
            unsafe {
                w_dict_setitem_str(
                    pyre_object::gc_roots::shadow_stack_get(root_base + 1),
                    $name,
                    value,
                )
            };
        }};
    }
    store!("_handle", w_int_new(raw as usize as i64));
    store!("handle", w_int_new(raw as usize as i64));
    store!("kind", w_int_new(kind));
    store!("maxvalue", w_int_new(maxvalue));
    store!(
        "name",
        kept_name.map_or_else(w_none, |name| w_str_new_managed(&name))
    );
    // interp_semaphore.py `self.count = 0`, `self.last_tid = -1`.
    store!("count", w_int_new(0));
    store!("last_tid", w_int_new(-1));
    Ok(pyre_object::gc_roots::shadow_stack_get(root_base))
}

/// `_multiprocessing.SemLock.__new__` declares
/// `kind: int, value: int, maxvalue: int, name: str, unlink: int`, all five
/// positional-or-keyword, so each binds by name and each converter reports
/// its own argument.
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn semlock_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let Some((&(mut w_subtype), rest)) = args.split_first() else {
        return Err(pyre_interpreter::PyError::type_error(
            "_multiprocessing.SemLock.__new__(): not enough arguments",
        ));
    };
    let scope = pyre_interpreter::builtins::bind_builtin_kwargs(
        rest,
        &["kind", "value", "maxvalue", "name", "unlink"],
        &[true; 5],
        "SemLock",
    )?;
    let kind =
        pyre_object::with_roots!(w_subtype => pyre_interpreter::builtins::space_index_w(scope[0]))?;
    let value =
        pyre_object::with_roots!(w_subtype => pyre_interpreter::builtins::space_index_w(scope[1]))?;
    let maxvalue =
        pyre_object::with_roots!(w_subtype => pyre_interpreter::builtins::space_index_w(scope[2]))?;
    if !unsafe { is_str(scope[3]) } {
        let type_name = pyre_interpreter::type_methods::clinic_arg_type_name(scope[3]);
        return Err(pyre_interpreter::PyError::type_error(format!(
            "SemLock() argument 'name' must be str, not {type_name}"
        )));
    }
    let name = pyre_object::with_roots!(w_subtype => pyre_interpreter::baseobjspace::str_utf8_w(scope[3]))?.to_string();
    // `unwrap_spec(unlink=int)` (interp_semaphore.py:572) — the flag is
    // converted the same way `kind`, `value` and `maxvalue` beside it are,
    // so a type whose `__bool__` and `__index__` disagree does not decide it.
    let unlink = pyre_object::with_roots!(w_subtype => pyre_interpreter::builtins::space_index_w(scope[4]))?
        != 0;
    // interp_semaphore.py:574-575.
    if kind != RECURSIVE_MUTEX && kind != SEMAPHORE {
        return Err(pyre_interpreter::PyError::value_error("unrecognized kind"));
    }
    // `sem_open` releases the GIL. `w_subtype` is a heap class and can move.
    let (raw, kept_name) = pyre_object::with_roots!(w_subtype => {
        semlock_create(&name, value, maxvalue, unlink)
    })?;
    semlock_instance(w_subtype, raw, kind, maxvalue, kept_name)
}

/// `interp_semaphore.py W_SemLock.rebuild`, registered as a
/// classmethod (`:606`), so `args[0]` is the bound class.
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn semlock_rebuild(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if args.len() != 5 {
        return Err(pyre_interpreter::PyError::type_error(
            "_rebuild() takes exactly 4 arguments",
        ));
    }
    // `int_w` and `str_utf8_w` collect, so the arguments still read after
    // them travel in rooted locals rather than through the native `args`.
    let mut w_cls = args[0];
    let mut w_handle = args[1];
    let mut w_maxvalue = args[3];
    let mut w_name = args[4];
    let kind = pyre_object::with_roots!(w_cls, w_handle, w_maxvalue, w_name =>
        pyre_interpreter::baseobjspace::int_w(args[2])
    )?;
    let maxvalue = pyre_object::with_roots!(w_cls, w_handle, w_name =>
        pyre_interpreter::baseobjspace::int_w(w_maxvalue)
    )?;
    // `unwrap_spec(name='text_or_none')` — an unlinked semaphore carries no
    // name and travels as its raw handle instead.
    let name = if unsafe { is_none(w_name) } {
        None
    } else if unsafe { is_str(w_name) } {
        Some(
            pyre_object::with_roots!(w_cls, w_handle =>
                pyre_interpreter::baseobjspace::str_utf8_w(w_name)
            )?
            .to_string(),
        )
    } else {
        return Err(pyre_interpreter::PyError::type_error(
            "_rebuild() argument 'name' must be str or None",
        ));
    };
    let raw = pyre_object::with_roots!(w_cls => semlock_rebuild_raw(w_handle, name.as_deref()))?;
    semlock_instance(w_cls, raw, kind, maxvalue, name)
}

#[cfg(all(any(unix, windows), feature = "host_env"))]
pyre_interpreter::py_class! {
    "SemLock",
    methods: {
        fn acquire(
            mut self_obj: PyObjectRef,
            blocking: Option<i64>,
            timeout: Option<PyObjectRef>,
        ) -> Result<bool, pyre_interpreter::PyError> {
            let block = blocking.map(|v| v != 0).unwrap_or(true);
            let timeout = match timeout {
                Some(value) if unsafe { !is_none(value) } => {
                    Some(pyre_object::with_roots!(self_obj => pyre_interpreter::baseobjspace::float_w(value))?)
                }
                _ => None,
            };
            w_semlock_acquire(self_obj, block, timeout)
        }
        fn release(self_obj: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
            w_semlock_release(self_obj)
        }
        // interp_semaphore.py W_SemLock.get_count
        fn _count(self_obj: PyObjectRef) -> i64 {
            semlock_get_i64(self_obj, "count")
        }
        // interp_semaphore.py W_SemLock.is_mine
        fn _is_mine(self_obj: PyObjectRef) -> bool {
            semlock_ismine(self_obj)
        }
        // interp_semaphore.py W_SemLock.after_fork
        fn _after_fork(self_obj: PyObjectRef) {
            semlock_set_i64(self_obj, "count", 0);
        }
        // interp_semaphore.py W_SemLock.is_zero
        fn _is_zero(self_obj: PyObjectRef) -> Result<bool, pyre_interpreter::PyError> {
            let handle = semlock_get_handle(self_obj);
            if handle.is_null() {
                return Err(pyre_interpreter::PyError::value_error("SemLock handle is null"));
            }
            semlock_iszero(handle)
        }
        // interp_semaphore.py W_SemLock.get_value
        fn _get_value(self_obj: PyObjectRef) -> Result<i64, pyre_interpreter::PyError> {
            let handle = semlock_get_handle(self_obj);
            if handle.is_null() {
                return Err(pyre_interpreter::PyError::value_error("SemLock handle is null"));
            }
            semlock_value(handle)
        }
        // interp_semaphore.py W_SemLock.enter
        fn __enter__(self_obj: PyObjectRef) -> Result<bool, pyre_interpreter::PyError> {
            w_semlock_acquire(self_obj, true, None)
        }
        // interp_semaphore.py W_SemLock.exit
        fn __exit__(
            self_obj: PyObjectRef,
            exc_type: Option<PyObjectRef>,
            exc_value: Option<PyObjectRef>,
            traceback: Option<PyObjectRef>,
        ) -> Result<(), pyre_interpreter::PyError> {
            let _ = (exc_type, exc_value, traceback);
            w_semlock_release(self_obj)
        }
    }
}

#[cfg(all(any(unix, windows), feature = "host_env"))]
#[pyre_interpreter::pyre_function]
fn sem_unlink(name: &str) -> Result<(), pyre_interpreter::PyError> {
    #[cfg(unix)]
    {
        let c_name = sem_c_name(name)?;
        if unsafe { ll::_sem_unlink(c_name.as_ptr()) } < 0 {
            return Err(sem_oserror("sem_unlink failed"));
        }
        Ok(())
    }
    // A Windows semaphore has no name in the filesystem sense, so there is
    // nothing to remove and `SEM_UNLINK` is the constant success the call
    // reads (`semaphore.c`).
    #[cfg(windows)]
    {
        let _ = name;
        Ok(())
    }
}

/// The three socket calls `multiprocessing/connection.py` binds as default
/// arguments on Windows, where a `Connection` is a socket rather than a
/// descriptor.  A failure is reported by `WSAGetLastError`, so it carries a
/// Win32 code rather than an errno.
///
/// All three release the interpreter around the call, as
/// `_multiprocessing_recv_impl` and its two neighbours do.  A `Connection` is
/// what a manager server and its proxies talk over, so a `recv` that waits for
/// the peer's next request otherwise holds the interpreter for as long as the
/// peer takes to send one — which, when the peer is itself waiting on this
/// process, is forever.
#[cfg(all(windows, feature = "host_env"))]
#[pyre_interpreter::pyre_function]
fn closesocket(handle: i64) -> Result<(), pyre_interpreter::PyError> {
    let result = {
        let _blocked = pyre_interpreter::module::thread::before_external_block();
        host_mp::close_socket(handle as usize as host_mp::RawSocket)
    };
    result.map_err(|error| windows_error(error.raw_os_error().unwrap_or(0)))
}

#[cfg(all(windows, feature = "host_env"))]
#[pyre_interpreter::pyre_function]
fn recv(handle: i64, size: i64) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let size = usize::try_from(size)
        .map_err(|_| pyre_interpreter::PyError::value_error("negative buffer size"))?;
    let result = {
        let _blocked = pyre_interpreter::module::thread::before_external_block();
        host_mp::recv_socket(handle as usize as host_mp::RawSocket, size)
    };
    let data = result.map_err(|error| windows_error(error.raw_os_error().unwrap_or(0)))?;
    Ok(pyre_object::w_bytes_from_bytes(&data))
}

#[cfg(all(windows, feature = "host_env"))]
#[pyre_interpreter::pyre_function]
fn send(handle: i64, buf: &[u8]) -> Result<i64, pyre_interpreter::PyError> {
    // Copied out before the interpreter is released: the borrow reaches into
    // the argument object, and a collection running in another thread can move
    // it.  `Py_buffer` holds the original still for the same span.
    let buf = buf.to_vec();
    let result = {
        let _blocked = pyre_interpreter::module::thread::before_external_block();
        host_mp::send_socket(handle as usize as host_mp::RawSocket, &buf)
    };
    result
        .map(i64::from)
        .map_err(|error| windows_error(error.raw_os_error().unwrap_or(0)))
}

pyre_interpreter::py_module! {
    "_multiprocessing",
    extra_init: |ns| {
        #[cfg(all(any(unix, windows), feature = "host_env"))]
        {
            let mut semlock_type = pyre_object::with_roots!(ns => type_object());
            pyre_interpreter::__pyre_store!(ns, "SemLock", semlock_type);
            // interp_semaphore.py W_SemLock.typedef publishes this
            // constant on the class (the module also exports its own copy).
            // `SEM_VALUE_MAX` is what the platform will count to: the
            // POSIX limit, or `LONG_MAX` where `CreateSemaphoreW` takes the
            // maximum as its own argument.
            #[cfg(unix)]
            let value_max = {
                let n = unsafe { libc::sysconf(libc::_SC_SEM_VALUE_MAX) };
                if n < 0 || n > i32::MAX as libc::c_long {
                    i64::from(i32::MAX)
                } else {
                    n as i64
                }
            };
            #[cfg(windows)]
            let value_max = i64::from(i32::MAX);
            let _roots = pyre_object::gc_roots::push_roots();
            // Both words already exist. Sequential `pin_root` would query
            // after the first write and leave the other word invisible
            // (`RootScope::pin_roots`).
            let ns_slot = pyre_object::gc_roots::pin_roots(&[ns, semlock_type]);
            let semlock_slot = ns_slot + 1;
            let vmax_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_int_new(value_max));
            let dict_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(unsafe {
                pyre_object::w_type_get_dict_ptr(pyre_object::gc_roots::shadow_stack_get(
                    semlock_slot,
                )) as PyObjectRef
            });
            unsafe {
                pyre_object::w_dict_setitem_str_no_proxy(
                    pyre_object::gc_roots::shadow_stack_get(dict_slot),
                    "SEM_VALUE_MAX",
                    pyre_object::gc_roots::shadow_stack_get(vmax_slot),
                );
                pyre_interpreter::__pyre_put_new!(
                    dict_slot,
                    "__new__",
                    // interp_semaphore.py:572-573 declares the parameters
                    // through `@unwrap_spec(kind=int, value=int, maxvalue=int,
                    // name='text', unlink=int)`, so `interp2app` binds them by
                    // name as well as by position.  Registering the same names
                    // here routes a keyword call through the gateway binder,
                    // which hands the body the slots in positional order.
                    // `subtype` stays positional-only: it is the class the
                    // descriptor was reached through, not a parameter.
                    pyre_interpreter::make_builtin_function_with_signature(
                        "__new__",
                        semlock_descr_new,
                        pyre_interpreter::gateway::Signature::new(
                            vec!["subtype", "kind", "value", "maxvalue", "name", "unlink"],
                            None,
                            None,
                            0,
                            1,
                        ),
                    )
                );
                // interp_semaphore.py `as_classmethod=True` — `_rebuild`
                // allocates on the class it is called through.
                let rebuild_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(pyre_interpreter::make_builtin_function(
                    "_rebuild",
                    semlock_rebuild,
                ));
                pyre_interpreter::__pyre_put_new!(
                    dict_slot,
                    "_rebuild",
                    pyre_object::function::w_classmethod_new(
                        pyre_object::gc_roots::shadow_stack_get(rebuild_slot)
                    )
                );
                // PyPy `W_SemLock.typedef` owns `descr_new` in its rawdict,
                // so `TypeDef.acceptable_as_base_class` is true.  This manual
                // type installs the same descriptor after construction;
                // reflect that rawdict result on its own TypeDef now.
                pyre_object::w_type_set_acceptable_as_base_class(
                    pyre_object::gc_roots::shadow_stack_get(semlock_slot),
                    true,
                );
            }
            ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
            pyre_interpreter::__pyre_store!(ns, "sem_unlink", pyre_interpreter::make_builtin_function_with_arity("sem_unlink", sem_unlink, 1));
        }
        #[cfg(all(windows, feature = "host_env"))]
        {
            pyre_interpreter::__pyre_store!(ns, "closesocket", pyre_interpreter::make_builtin_function_with_arity("closesocket", closesocket, 1));
            pyre_interpreter::__pyre_store!(ns, "recv", pyre_interpreter::make_builtin_function_with_arity("recv", recv, 2));
            pyre_interpreter::__pyre_store!(ns, "send", pyre_interpreter::make_builtin_function_with_arity("send", send, 2));
            // `flags` reports the build-time semaphore capabilities the
            // POSIX build is configured with; this one has none to report.
            pyre_interpreter::__pyre_store!(ns, "flags", pyre_object::w_dict_new());
        }
    }
}
