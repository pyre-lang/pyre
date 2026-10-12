//! `rpython/rlib/rsignal.py` — Unix libc signal llexternals.
//!
//! The `pypysig_*` helpers live in `rpython/translator/c/src/signals.c` and
//! are not declared here. `rsignal.external` sets `sandboxsafe=True`.

#![cfg(unix)]

use crate::rffi::{CCHARP, INT};

/// `rsignal.timeval`.
#[allow(non_camel_case_types)]
pub type timeval = libc::timeval;
/// `rsignal.itimerval`.
#[allow(non_camel_case_types)]
pub type itimerval = libc::itimerval;
/// `rsignal.itimervalP`.
#[allow(non_camel_case_types)]
pub type itimervalP = *mut itimerval;
/// `rsignal.c_sigset_t`.
#[allow(non_camel_case_types)]
pub type c_sigset_t = *mut libc::sigset_t;

crate::rffi::external_compilation_info! {
    const SIGNAL_ECI = {
        includes: ["stdlib.h", "signal.h", "sys/time.h"],
    };
}

// `rsignal.c_raise`.
crate::rffi::llexternal!(
    pub c_raise = "raise",
    [INT],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true
);

// `rsignal.c_alarm`.
crate::rffi::llexternal!(
    pub c_alarm = "alarm",
    [INT],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true
);

// `rsignal.c_pause`.
crate::rffi::llexternal!(
    pub c_pause = "pause",
    [],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true,
    releasegil = true
);

// `rsignal.c_setitimer`.
crate::rffi::llexternal!(
    pub c_setitimer = "setitimer",
    [INT, itimervalP, itimervalP],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true,
    save_err = crate::rffi::RFFI_SAVE_ERRNO
);

// `rsignal.c_getitimer`.
crate::rffi::llexternal!(
    pub c_getitimer = "getitimer",
    [INT, itimervalP],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true
);

// `rsignal.c_strsignal`.
crate::rffi::llexternal!(
    pub c_strsignal = "strsignal",
    [INT],
    CCHARP,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true
);

// `rsignal.c_sigemptyset`.
crate::rffi::llexternal!(
    pub c_sigemptyset = "sigemptyset",
    [c_sigset_t],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true
);

// `rsignal.c_sigfillset`.
crate::rffi::llexternal!(
    pub c_sigfillset = "sigfillset",
    [c_sigset_t],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true
);

// `rsignal.c_sigaddset`.
crate::rffi::llexternal!(
    pub c_sigaddset = "sigaddset",
    [c_sigset_t, INT],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true
);

// `rsignal.c_sigismember`.
crate::rffi::llexternal!(
    pub c_sigismember = "sigismember",
    [c_sigset_t, INT],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true
);

// `rsignal.c_sigwait`. Returns the error number; `save_err` still runs.
crate::rffi::llexternal!(
    pub c_sigwait = "sigwait",
    [c_sigset_t, *mut INT],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true,
    releasegil = true,
    save_err = crate::rffi::RFFI_SAVE_ERRNO
);

// `rsignal.c_siginterrupt` is `pypysig_siginterrupt` (`signals.c`).
// `siginterrupt(2)` is deprecated on glibc 2.21+; the helper uses
// `sigaction` + `SA_RESTART` instead. `rsignal.external` is
// `sandboxsafe=True` and this call also saves errno.
unsafe fn pypysig_siginterrupt(sig: INT, flag: INT) -> INT {
    let mut act = unsafe { std::mem::zeroed::<libc::sigaction>() };
    if unsafe { libc::sigaction(sig, std::ptr::null(), &mut act) } < 0 {
        return -1;
    }
    if flag != 0 {
        act.sa_flags &= !libc::SA_RESTART;
    } else {
        act.sa_flags |= libc::SA_RESTART;
    }
    unsafe { libc::sigaction(sig, &act, std::ptr::null_mut()) }
}

crate::rffi::llexternal!(
    pub c_siginterrupt = "pypysig_siginterrupt",
    [INT, INT],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true,
    save_err = crate::rffi::RFFI_SAVE_ERRNO,
    macro = pypysig_siginterrupt
);

// `rsignal.c_sigpending`.
crate::rffi::llexternal!(
    pub c_sigpending = "sigpending",
    [c_sigset_t],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true,
    save_err = crate::rffi::RFFI_SAVE_ERRNO
);

// `rsignal.c_pthread_sigmask`.
crate::rffi::llexternal!(
    pub c_pthread_sigmask = "pthread_sigmask",
    [INT, c_sigset_t, c_sigset_t],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true,
    save_err = crate::rffi::RFFI_SAVE_ERRNO
);

/// `rsignal.strsignal`.
pub fn strsignal(signum: INT) -> Option<String> {
    let res = unsafe { c_strsignal(signum) };
    if res.is_null() {
        return None;
    }
    let bytes = unsafe { crate::rffi::charp2str(res) };
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

#[cfg(test)]
mod tests {
    #[test]
    fn c_alarm() {
        let ret = unsafe { super::c_alarm(0) };
        assert!(ret >= 0);
    }

    #[test]
    fn c_strsignal() {
        let p = unsafe { super::c_strsignal(libc::SIGINT) };
        assert!(!p.is_null());
    }

    #[test]
    fn c_sigemptyset() {
        let mut set = unsafe { std::mem::zeroed::<libc::sigset_t>() };
        assert_eq!(unsafe { super::c_sigemptyset(&mut set) }, 0);
        assert_eq!(unsafe { super::c_sigaddset(&mut set, libc::SIGINT) }, 0);
        assert_eq!(unsafe { super::c_sigismember(&mut set, libc::SIGINT) }, 1);
    }
}
