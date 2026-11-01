//! select.kqueue — PyPy: pypy/module/select/interp_kqueue.py W_Kqueue.
//!
//! Kept separate from `interp_kevent` because each `#[pyre_class]`
//! emits its own module-scoped `type_object()`.

#![allow(dead_code)]

#[cfg(all(target_os = "macos", feature = "host_env"))]
use super::interp_kevent::W_Kevent;
#[cfg(all(target_os = "macos", feature = "host_env"))]
use pyre_object::PyObjectRef;

/// `interp_kqueue.py` `eci`.
#[cfg(all(target_os = "macos", feature = "host_env"))]
mod ll {
    use majit_rlib::rffi::{INT, RFFI_SAVE_ERRNO};

    majit_rlib::rffi::external_compilation_info! {
        const ECI = {
            includes: ["sys/types.h", "sys/event.h", "sys/time.h"],
        };
    }

    // This module is macOS-only; NetBSD `__kevent50` is not.
    majit_rlib::rffi::llexternal!(
        pub(super) syscall_kqueue = "kqueue",
        [],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    majit_rlib::rffi::llexternal!(
        pub(super) syscall_kevent = "kevent",
        [
            INT,
            *const libc::kevent,
            INT,
            *mut libc::kevent,
            INT,
            *const libc::timespec
        ],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );

    /// `interp_kqueue.py fill_timespec`: `int(time_float)` and
    /// `int(1e9 * (time_float - sec))`.
    pub(super) fn fill_timespec(secs: f64) -> Result<libc::timespec, pyre_interpreter::PyError> {
        if !secs.is_finite() || secs > i64::MAX as f64 {
            return Err(pyre_interpreter::PyError::overflow_error(
                "timeout is too large",
            ));
        }
        let sec = secs as i64;
        let nsec = ((secs - sec as f64) * 1e9) as i64;
        Ok(libc::timespec {
            tv_sec: sec as _,
            tv_nsec: nsec as _,
        })
    }
}

/// `select.kqueue` object — PyPy: `interp_kqueue.py class W_Kqueue`.
///
/// Wraps a kqueue file descriptor (`-1` once closed).  `control()`
/// marshals a changelist of `kevent`s into the syscall and returns the
/// triggered events as fresh `kevent` instances.
#[cfg(all(target_os = "macos", feature = "host_env"))]
// CPython 3.14 Modules/selectmodule.c:select_exec creates
// kqueue_queue_Type_spec as a mutable module heap type.
#[pyre_interpreter::pyre_class("select.kqueue", cpython_mutable)]
pub struct W_Kqueue {
    kqfd: i32,
}

#[cfg(all(target_os = "macos", feature = "host_env"))]
impl Default for W_Kqueue {
    fn default() -> Self {
        W_Kqueue {
            ob: Default::default(),
            kqfd: -1,
        }
    }
}

#[cfg(all(target_os = "macos", feature = "host_env"))]
fn kqueue_oserror(errno: i32, what: &str) -> pyre_interpreter::PyError {
    let e = std::io::Error::from_raw_os_error(errno);
    pyre_interpreter::PyError::os_error_with_errno(errno, format!("{what}: {e}"))
}

/// `kqueue_queue_internal_close`: mark `kqfd = -1` first, then close.
/// Returns the saved errno, or 0. A finalizer swallows this; the
/// Python `close` method reports it.
#[cfg(all(target_os = "macos", feature = "host_env"))]
fn kqueue_queue_internal_close(self_: &mut W_Kqueue) -> i32 {
    if self_.kqfd < 0 {
        return 0;
    }
    let kqfd = self_.kqfd;
    self_.kqfd = -1;
    let rc = unsafe { majit_rlib::_rsocket_rffi::socketclose(kqfd) };
    if rc < 0 {
        majit_rlib::rposix::get_saved_errno()
    } else {
        0
    }
}

#[cfg(all(target_os = "macos", feature = "host_env"))]
#[pyre_interpreter::pyre_methods(doc = "kqueue() -> kqueue object")]
impl W_Kqueue {
    /// `interp_kqueue.py descr__new__` — opens a fresh kqueue fd,
    /// clearing its inheritable flag. A failed `set_inheritable` does not
    /// close the fd.
    #[staticmethod]
    fn __new__(_cls: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        let kqfd = unsafe { ll::syscall_kqueue() };
        if kqfd < 0 {
            return Err(kqueue_oserror(
                majit_rlib::rposix::get_saved_errno(),
                "kqueue",
            ));
        }
        if let Err(errno) = majit_rlib::rposix::set_inheritable(kqfd, false) {
            return Err(kqueue_oserror(errno, "kqueue"));
        }
        Ok(W_Kqueue::allocate(W_Kqueue {
            kqfd,
            ..Default::default()
        }))
    }

    /// `interp_kqueue.py descr_fromfd` — wraps an existing fd.
    #[classmethod]
    fn fromfd(_cls: PyObjectRef, fd: i64) -> PyObjectRef {
        W_Kqueue::allocate(W_Kqueue {
            kqfd: fd as i32,
            ..Default::default()
        })
    }

    #[getter]
    fn closed(&self) -> bool {
        self.kqfd < 0
    }

    fn fileno(&self) -> Result<i64, pyre_interpreter::PyError> {
        if self.kqfd < 0 {
            return Err(pyre_interpreter::PyError::value_error(
                "I/O operation on closed kqueue fd",
            ));
        }
        Ok(self.kqfd as i64)
    }

    /// [3.15-spec] `select_kqueue_close_impl` reports OSError from
    /// `kqueue_queue_internal_close` ↔ PyPy `W_Kqueue.close` uses
    /// `socketclose_no_errno` (silent) — `os.close` then `kqueue.close`
    /// raises `OSError(EBADF)`; evidence: select_kqueue_close_impl
    fn close(&mut self) -> Result<(), pyre_interpreter::PyError> {
        let err = kqueue_queue_internal_close(self);
        if err != 0 {
            return Err(pyre_interpreter::PyError::os_error_with_errno(err, ""));
        }
        Ok(())
    }

    /// `interp_kqueue.py descr_control` — apply `changelist` (a list
    /// of kevents or None) and collect up to `max_events` triggered
    /// events.  `timeout` is in seconds (float) or None to block.
    fn control(
        &mut self,
        w_changelist: PyObjectRef,
        max_events: i64,
        #[default(pyre_object::w_none())] mut w_timeout: PyObjectRef,
    ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        if self.kqfd < 0 {
            return Err(pyre_interpreter::PyError::value_error(
                "I/O operation on closed kqueue fd",
            ));
        }
        if max_events < 0 {
            return Err(pyre_interpreter::PyError::value_error(format!(
                "Length of eventlist must be 0 or positive, got {max_events}"
            )));
        }

        let changelist_is_none = unsafe { pyre_object::is_none(w_changelist) };
        let mut changelist: Vec<libc::kevent> = Vec::new();
        if !changelist_is_none {
            // `descr_control` — `space.listview` accepts any iterable.
            let items = pyre_object::with_roots!(w_timeout => pyre_interpreter::baseobjspace::unpackiterable(w_changelist, -1))?;
            for item in items {
                let ev = W_Kevent::from_obj(item).ok_or_else(|| {
                    pyre_interpreter::PyError::type_error(
                        "arg 1 must be a sequence of kevent objects",
                    )
                })?;
                changelist.push(libc::kevent {
                    ident: ev.ident as libc::uintptr_t,
                    filter: ev.filter,
                    flags: ev.flags,
                    fflags: ev.fflags,
                    data: ev.data as libc::intptr_t,
                    udata: ev.udata as *mut core::ffi::c_void,
                });
            }
        }

        let timeout_is_none = unsafe { pyre_object::is_none(w_timeout) };
        // `(timespec, deadline)`. `None` is a null timeout pointer. The
        // timespec stays in this local so each pass can point at it, and an
        // EINTR retry overwrites the same slot before the next call.
        let mut timed: Option<(libc::timespec, std::time::Instant)> = if timeout_is_none {
            None
        } else {
            // `descr_control` — `space.float_w` honours `__float__`.
            let secs = pyre_interpreter::baseobjspace::float_w(w_timeout)?;
            if secs < 0.0 {
                return Err(pyre_interpreter::PyError::value_error(format!(
                    "Timeout must be None or >= 0, got {secs}"
                )));
            }
            let ts = ll::fill_timespec(secs)?;
            let deadline = std::time::Duration::try_from_secs_f64(secs)
                .ok()
                .and_then(|d| std::time::Instant::now().checked_add(d))
                .ok_or_else(|| pyre_interpreter::PyError::overflow_error("timeout is too large"))?;
            Some((ts, deadline))
        };

        let mut eventlist = vec![
            libc::kevent {
                ident: 0,
                filter: 0,
                flags: 0,
                fflags: 0,
                data: 0,
                udata: core::ptr::null_mut(),
            };
            max_events as usize
        ];
        // `None` passes a null changelist. A present list passes its pointer
        // even when the list is empty (`nchanges == 0`).
        let pchangelist: *const libc::kevent = if changelist_is_none {
            core::ptr::null()
        } else {
            changelist.as_ptr()
        };
        let nchanges = changelist.len() as i32;

        let nfds = loop {
            let ptimeout: *const libc::timespec = match timed.as_ref() {
                Some((ts, _)) => ts,
                None => core::ptr::null(),
            };
            let n = unsafe {
                ll::syscall_kevent(
                    self.kqfd,
                    pchangelist,
                    nchanges,
                    eventlist.as_mut_ptr(),
                    max_events as i32,
                    ptimeout,
                )
            };
            if n >= 0 {
                break n;
            }
            let errno = majit_rlib::rposix::get_saved_errno();
            if errno == libc::EINTR {
                pyre_interpreter::module::signal::interp_signal::checksignals_now()?;
                if let Some((ts, dl)) = timed.as_mut() {
                    let now = std::time::Instant::now();
                    let remaining = if now >= *dl {
                        0.0
                    } else {
                        (*dl - now).as_secs_f64()
                    };
                    *ts = ll::fill_timespec(remaining)?;
                }
                continue;
            }
            return Err(kqueue_oserror(errno, "kevent"));
        };

        // Each `allocate` can move the previous kevent. Pin as they are
        // minted; `w_list_new` reads the live slots back.
        let mut result = pyre_object::gc_roots::RootedItems::new();
        for evt in eventlist.iter().take(nfds as usize) {
            result.push(W_Kevent::allocate(W_Kevent {
                ident: evt.ident as u64,
                filter: evt.filter,
                flags: evt.flags,
                fflags: evt.fflags,
                data: evt.data as i64,
                udata: evt.udata as u64,
                ..Default::default()
            }));
        }
        Ok(pyre_object::w_list_new(result.take()))
    }
}

#[cfg(all(test, target_os = "macos", feature = "host_env"))]
mod tests {
    #[test]
    fn kqueue_opens_and_closes() {
        let fd = unsafe { super::ll::syscall_kqueue() };
        assert!(
            fd >= 0,
            "kqueue errno {}",
            majit_rlib::rposix::get_saved_errno()
        );
        majit_rlib::rposix::set_inheritable(fd, false).expect("cloexec");
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        assert!(flags >= 0, "F_GETFD");
        assert_ne!(flags & libc::FD_CLOEXEC, 0);
        unsafe { majit_rlib::_rsocket_rffi::socketclose_no_errno(fd) };
    }
}
