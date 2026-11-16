//! `rpython/rlib/rpoll.py` — `poll` and `select` over `_rsocket_rffi`.
//!
//! The Win32 `_poll` (`WSAWaitForMultipleEvents`) stays unported: upstream
//! renames it off `poll` and marks it broken.

#![cfg(unix)]

use std::collections::HashMap;

use crate::_rsocket_rffi as _c;

/// `rpoll.eventnames` entries that `_rsocket_rffi.constants` defines here.
pub const POLLIN: i16 = libc::POLLIN as i16;
pub const POLLPRI: i16 = libc::POLLPRI as i16;
pub const POLLOUT: i16 = libc::POLLOUT as i16;
pub const POLLERR: i16 = libc::POLLERR as i16;
pub const POLLHUP: i16 = libc::POLLHUP as i16;
pub const POLLNVAL: i16 = libc::POLLNVAL as i16;
pub const POLLRDNORM: i16 = libc::POLLRDNORM as i16;
pub const POLLRDBAND: i16 = libc::POLLRDBAND as i16;
pub const POLLWRNORM: i16 = libc::POLLWRNORM as i16;
pub const POLLWRBAND: i16 = libc::POLLWRBAND as i16;
pub const FD_SETSIZE: usize = _c::FD_SETSIZE;

/// `rpoll.PollError`.
#[derive(Debug)]
pub struct PollError {
    pub errno: i32,
}

/// `rpoll.SelectError`.
#[derive(Debug)]
pub struct SelectError {
    pub errno: i32,
}

/// `rpoll.poll`. `timeout` is milliseconds; `-1` blocks.
pub fn poll(fddict: &HashMap<i32, i16>, timeout: i32) -> Result<Vec<(i32, i32)>, PollError> {
    let numfd = fddict.len();
    let mut pollfds: Vec<libc::pollfd> = Vec::with_capacity(numfd);
    for (&fd, &events) in fddict {
        pollfds.push(libc::pollfd {
            fd,
            events: events as _,
            revents: 0,
        });
    }
    let ret = unsafe { _c::poll(pollfds.as_mut_ptr(), numfd as libc::nfds_t, timeout) };
    if ret < 0 {
        return Err(PollError {
            errno: _c::geterrno(),
        });
    }
    let mut retval = Vec::new();
    for pollfd in &pollfds {
        let revents = pollfd.revents as i32;
        if revents != 0 {
            retval.push((pollfd.fd, revents));
        }
    }
    Ok(retval)
}

/// `lltype.malloc(_c.fd_set.TO, flavor='raw')` / `lltype.free` (`rpoll.select`).
struct FdSet(_c::fd_set);

impl FdSet {
    fn malloc() -> Result<Self, crate::rffi::MemoryError> {
        Ok(Self(crate::rffi::malloc_opaque::<_c::fd_set>()?))
    }

    fn as_ptr(&self) -> _c::fd_set {
        self.0
    }
}

impl Drop for FdSet {
    fn drop(&mut self) {
        unsafe { crate::rffi::free_opaque(self.0) };
    }
}

fn prepare_set(fds: &[i32], nfds: &mut i32) -> Result<Option<FdSet>, crate::rffi::MemoryError> {
    if fds.is_empty() {
        return Ok(None);
    }
    let set = FdSet::malloc()?;
    unsafe { _c::FD_ZERO(set.as_ptr()) };
    for &fd in fds {
        unsafe { _c::FD_SET(fd, set.as_ptr()) };
        if fd > *nfds {
            *nfds = fd;
        }
    }
    Ok(Some(set))
}

fn set_ptr(set: &Option<FdSet>) -> _c::fd_set {
    match set {
        Some(set) => set.as_ptr(),
        None => core::ptr::null_mut(),
    }
}

fn collect_ready(fds: &[i32], set: &Option<FdSet>) -> Vec<i32> {
    let Some(set) = set else {
        return Vec::new();
    };
    fds.iter()
        .copied()
        .filter(|&fd| unsafe { _c::FD_ISSET(fd, set.as_ptr()) } != 0)
        .collect()
}

/// `rpoll.select`. `timeout` is seconds; negative blocks. `handle_eintr`
/// retries a blocking call and turns an interrupted timed call into a timeout.
pub fn select(
    inl: &[i32],
    outl: &[i32],
    excl: &[i32],
    timeout: f64,
    handle_eintr: bool,
) -> Result<Result<(Vec<i32>, Vec<i32>, Vec<i32>), SelectError>, crate::rffi::MemoryError> {
    let mut nfds = 0;
    let ll_inl = prepare_set(inl, &mut nfds)?;
    let ll_outl = prepare_set(outl, &mut nfds)?;
    let ll_excl = prepare_set(excl, &mut nfds)?;

    let res = if timeout < 0.0 {
        let mut res;
        loop {
            res = unsafe {
                _c::select(
                    nfds + 1,
                    set_ptr(&ll_inl),
                    set_ptr(&ll_outl),
                    set_ptr(&ll_excl),
                    core::ptr::null_mut(),
                )
            };
            if !handle_eintr || res >= 0 || _c::geterrno() != libc::EINTR {
                break;
            }
        }
        res
    } else {
        let sec = timeout as i64;
        let usec = ((timeout - sec as f64) * 1_000_000.0) as i64;
        let mut ll_timeval = libc::timeval {
            tv_sec: sec as _,
            tv_usec: usec as _,
        };
        let res = unsafe {
            _c::select(
                nfds + 1,
                set_ptr(&ll_inl),
                set_ptr(&ll_outl),
                set_ptr(&ll_excl),
                &raw mut ll_timeval,
            )
        };
        if handle_eintr && res < 0 && _c::geterrno() == libc::EINTR {
            0
        } else {
            res
        }
    };

    if res == -1 {
        return Ok(Err(SelectError {
            errno: _c::geterrno(),
        }));
    }
    if res == 0 {
        return Ok(Ok((Vec::new(), Vec::new(), Vec::new())));
    }
    Ok(Ok((
        collect_ready(inl, &ll_inl),
        collect_ready(outl, &ll_outl),
        collect_ready(excl, &ll_excl),
    )))
}

#[cfg(all(test, feature = "host_env", not(feature = "sandbox")))]
mod tests {
    use super::*;

    /// Pipe tests share the process fd table. A closed descriptor is reused
    /// by a sibling test's `pipe` unless these tests take turns.
    fn fd_table() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|err| err.into_inner())
    }

    struct Close([i32; 2]);
    impl Drop for Close {
        fn drop(&mut self) {
            unsafe {
                libc::close(self.0[0]);
                libc::close(self.0[1]);
            }
        }
    }

    fn pipe() -> Close {
        let mut fds = [0; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        Close(fds)
    }

    #[test]
    fn poll_sees_bytes_written_to_a_pipe() {
        let _table = fd_table();
        let fds = pipe();
        let mut fddict = HashMap::new();
        fddict.insert(fds.0[0], POLLIN);
        let idle = poll(&fddict, 0).expect("poll");
        assert!(idle.is_empty());
        assert_eq!(unsafe { libc::write(fds.0[1], b"x".as_ptr().cast(), 1) }, 1);
        let ready = poll(&fddict, 1_000).expect("poll");
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].0, fds.0[0]);
        assert_ne!(ready[0].1 & i32::from(POLLIN), 0);
    }

    #[test]
    fn select_reports_the_pipe_ends() {
        let _table = fd_table();
        let fds = pipe();
        let (readable, writable, exceptional) = select(&[fds.0[0]], &[fds.0[1]], &[], 0.0, false)
            .expect("select malloc")
            .expect("select");
        assert!(readable.is_empty());
        assert_eq!(writable, vec![fds.0[1]]);
        assert!(exceptional.is_empty());
        assert_eq!(unsafe { libc::write(fds.0[1], b"x".as_ptr().cast(), 1) }, 1);
        let (readable, _, _) = select(&[fds.0[0]], &[], &[], 1.0, false)
            .expect("select malloc")
            .expect("select");
        assert_eq!(readable, vec![fds.0[0]]);
    }

    #[test]
    fn select_on_a_closed_fd_fails() {
        let _table = fd_table();
        // `pipe` reuses the lowest free descriptor, so a sibling test can
        // occupy this number between `close` and `select`. Repeat until the
        // number is still closed.
        for _ in 0..32 {
            let mut raw = [0; 2];
            assert_eq!(unsafe { libc::pipe(raw.as_mut_ptr()) }, 0);
            unsafe { libc::close(raw[0]) };
            let err = select(&[raw[0]], &[], &[], 0.0, false).expect("select malloc");
            unsafe { libc::close(raw[1]) };
            if let Err(err) = err {
                assert_eq!(err.errno, libc::EBADF);
                return;
            }
        }
        panic!("closed fd stayed readable");
    }
}
