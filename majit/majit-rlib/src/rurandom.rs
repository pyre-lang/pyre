//! `rpython/rlib/rurandom.py` — `urandom`.
//!
//! Linux tries `syscall(SYS_getrandom, …, GRND_NONBLOCK)` first
//! (`rurandom._getrandom`). Every Unix host then reads `/dev/urandom`
//! through `rposix.c_open` / `c_read` / `c_close`. Windows `BCryptGenRandom`
//! is unported.

#![cfg(unix)]

#[cfg(any(target_os = "linux", target_os = "android"))]
use crate::rffi::{CCHARP, INT, LONG, SIGNED};
use crate::rposix::{c_close, c_open, c_read, get_saved_errno};
#[cfg(any(target_os = "linux", target_os = "android"))]
use majit_jitcode::rffi::RFFI_SAVE_ERRNO;

#[cfg(any(target_os = "linux", target_os = "android"))]
crate::rffi::external_compilation_info! {
    const GETRANDOM_ECI = {
        includes: ["sys/syscall.h", "linux/random.h"],
    };
}

// `rurandom.syscall`. `save_err=RFFI_SAVE_ERRNO`.
#[cfg(any(target_os = "linux", target_os = "android"))]
crate::rffi::llexternal!(
    syscall = "syscall",
    [SIGNED, CCHARP, LONG, INT],
    SIGNED,
    compilation_info = GETRANDOM_ECI,
    save_err = RFFI_SAVE_ERRNO
);

/// `rurandom.SYS_getrandom`.
#[cfg(any(target_os = "linux", target_os = "android"))]
const SYS_GETRANDOM: SIGNED = libc::SYS_getrandom as SIGNED;

/// `rurandom.GRND_NONBLOCK` (`linux/random.h`; `0x0001` when the header
/// does not define it).
#[cfg(any(target_os = "linux", target_os = "android"))]
const GRND_NONBLOCK: INT = {
    #[cfg(target_os = "linux")]
    {
        libc::GRND_NONBLOCK as INT
    }
    #[cfg(not(target_os = "linux"))]
    {
        0x0001
    }
};

/// `rurandom.getrandom_works`. Process-global, like `rposix.ioctl_works`.
#[cfg(any(target_os = "linux", target_os = "android"))]
static GETRANDOM_WORKS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// `rurandom._getrandom`. Remaining `n` after a successful drain is 0; a
/// kernel that lacks the syscall returns the original `n` so `urandom`
/// falls through to `/dev/urandom`.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn _getrandom(
    mut n: usize,
    result: &mut Vec<u8>,
    signal_checker: &mut Option<&mut dyn FnMut() -> bool>,
) -> Result<usize, i32> {
    use std::sync::atomic::Ordering;
    if !GETRANDOM_WORKS.load(Ordering::Relaxed) {
        return Ok(n);
    }
    while n > 0 {
        let mut buf = vec![0u8; n];
        let got = unsafe {
            syscall(
                SYS_GETRANDOM,
                buf.as_mut_ptr() as CCHARP,
                n as LONG,
                GRND_NONBLOCK,
            )
        };
        if got >= 0 {
            let got = got as usize;
            result.extend_from_slice(&buf[..got]);
            n -= got;
            continue;
        }
        let err = get_saved_errno();
        if err == libc::ENOSYS || err == libc::EPERM || err == libc::EAGAIN {
            GETRANDOM_WORKS.store(false, Ordering::Relaxed);
            return Ok(n);
        }
        if err == libc::EINTR {
            if let Some(checker) = signal_checker.as_mut() {
                // `rurandom._getrandom`: product `interp_posix.urandom`
                // `_signal_checker`. True continues; false stops.
                if !checker() {
                    return Err(err);
                }
            }
            continue;
        }
        return Err(err);
    }
    Ok(n)
}

/// `rurandom.urandom`. `signal_checker` is invoked on `EINTR` from
/// `syscall` (`rurandom._getrandom`). True continues the read; false
/// stops and returns that `EINTR`.
pub fn urandom(
    n: usize,
    mut signal_checker: Option<&mut dyn FnMut() -> bool>,
) -> Result<Vec<u8>, i32> {
    let mut result = Vec::new();
    let mut n = n;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        n = _getrandom(n, &mut result, &mut signal_checker)?;
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        let _ = &mut signal_checker;
    }
    if n == 0 {
        return Ok(result);
    }

    let path = c"/dev/urandom";
    let fd = unsafe { c_open(path.as_ptr(), libc::O_RDONLY, 0o777 as _) };
    if fd < 0 {
        return Err(get_saved_errno());
    }
    let close_fd = |fd: crate::rffi::INT| unsafe {
        let _ = c_close(fd);
    };
    while n > 0 {
        let mut buf = vec![0u8; n];
        let got = unsafe { c_read(fd, buf.as_mut_ptr().cast(), n) };
        if got < 0 {
            let err = get_saved_errno();
            if err != libc::EINTR {
                close_fd(fd);
                return Err(err);
            }
            continue;
        }
        let got = got as usize;
        result.extend_from_slice(&buf[..got]);
        n -= got;
    }
    close_fd(fd);
    Ok(result)
}

/// Count-only `ll_os.ll_os_urandom` shape. `sandlib` has this name as a pyre
/// extension. The product host `interp_posix.urandom` still calls [`urandom`]
/// with a signal checker, which the trampoline cannot marshal.
#[majit_macros::replace_os_function("urandom")]
pub fn urandom_os(count: i64) -> Result<Vec<u8>, i32> {
    if count < 0 {
        return Err(libc::EINVAL);
    }
    urandom(count as usize, None)
}

#[cfg(all(test, feature = "host_env", not(feature = "sandbox")))]
mod tests {
    use super::{urandom, urandom_os};

    #[test]
    fn urandom_16_bytes() {
        let buf = urandom(16, None).expect("rurandom.urandom");
        assert_eq!(buf.len(), 16);
        assert_eq!(urandom_os(16).unwrap().len(), 16);
        assert_eq!(urandom_os(-1), Err(libc::EINVAL));
    }
}
