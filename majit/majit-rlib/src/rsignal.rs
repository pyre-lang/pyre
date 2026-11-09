//! `rpython/rlib/rsignal.py` — Unix libc signal llexternals.
//!
//! The `pypysig_*` C helpers are implemented in `rpython/translator/c/src/signals.c`;
//! `pypysig_getaddr_occurred` is declared here as `rsignal.py` does; the libc
//! llexternals are unix-only. `rsignal.external` sets `sandboxsafe=True`.

use std::sync::atomic::AtomicIsize;

#[cfg(unix)]
use crate::rffi::{CCHARP, INT};

/// `rsignal.timeval`.
#[cfg(unix)]
#[allow(non_camel_case_types)]
pub type timeval = libc::timeval;
/// `rsignal.itimerval`.
#[cfg(unix)]
#[allow(non_camel_case_types)]
pub type itimerval = libc::itimerval;
/// `rsignal.itimervalP`.
#[cfg(unix)]
#[allow(non_camel_case_types)]
pub type itimervalP = *mut itimerval;
/// `rsignal.c_sigset_t`.
#[cfg(unix)]
#[allow(non_camel_case_types)]
pub type c_sigset_t = *mut libc::sigset_t;

/// `rsignal.SIG_DFL`.
#[cfg(unix)]
pub const SIG_DFL: libc::sighandler_t = libc::SIG_DFL;
/// `rsignal.SIG_IGN`.
#[cfg(unix)]
pub const SIG_IGN: libc::sighandler_t = libc::SIG_IGN;

#[cfg(unix)]
crate::rffi::external_compilation_info! {
    const SIGNAL_ECI = {
        includes: ["stdlib.h", "signal.h", "sys/time.h"],
        separate_module_sources: [
            r#"RPY_EXTERN int pypysig_siginterrupt(int sig, int flag)
{
#ifdef SA_RESTART
    struct sigaction act;
    if (sigaction(sig, NULL, &act) < 0)
        return -1;
    if (flag)
        act.sa_flags &= ~SA_RESTART;
    else
        act.sa_flags |= SA_RESTART;
    return sigaction(sig, &act, NULL);
#else
    return siginterrupt(sig, flag);
#endif
}"#,
        ],
    };
}

// `rsignal.c_raise`.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_raise = "raise",
    [INT],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true
);

// `rsignal.c_alarm`.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_alarm = "alarm",
    [INT],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true
);

// `rsignal.c_pause`.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_pause = "pause",
    [],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true,
    releasegil = true
);

// `rsignal.c_setitimer`.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_setitimer = "setitimer",
    [INT, itimervalP, itimervalP],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true,
    save_err = crate::rffi::RFFI_SAVE_ERRNO
);

// `rsignal.c_getitimer`.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_getitimer = "getitimer",
    [INT, itimervalP],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true
);

// `rsignal.c_strsignal`.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_strsignal = "strsignal",
    [INT],
    CCHARP,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true
);

// `rsignal.c_sigemptyset`.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_sigemptyset = "sigemptyset",
    [c_sigset_t],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true
);

// `rsignal.c_sigfillset`.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_sigfillset = "sigfillset",
    [c_sigset_t],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true
);

// `rsignal.c_sigaddset`.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_sigaddset = "sigaddset",
    [c_sigset_t, INT],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true
);

// `rsignal.c_sigismember`.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_sigismember = "sigismember",
    [c_sigset_t, INT],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true
);

// `rsignal.c_sigwait`. Returns the error number; `save_err` still runs.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_sigwait = "sigwait",
    [c_sigset_t, *mut INT],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true,
    releasegil = true,
    save_err = crate::rffi::RFFI_SAVE_ERRNO
);

// `signals.c pypysig_siginterrupt`, compiled from `SIGNAL_ECI`.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_siginterrupt = "pypysig_siginterrupt",
    [INT, INT],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true,
    save_err = crate::rffi::RFFI_SAVE_ERRNO
);

// `rsignal.c_sigpending`.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_sigpending = "sigpending",
    [c_sigset_t],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true,
    save_err = crate::rffi::RFFI_SAVE_ERRNO
);

// `rsignal.c_pthread_sigmask`.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_pthread_sigmask = "pthread_sigmask",
    [INT, c_sigset_t, c_sigset_t],
    INT,
    compilation_info = SIGNAL_ECI,
    sandboxsafe = true,
    save_err = crate::rffi::RFFI_SAVE_ERRNO
);

/// `rsignal.strsignal`.
#[cfg(unix)]
pub fn strsignal(signum: INT) -> Result<Option<Vec<u8>>, crate::rffi::MemoryError> {
    let res = unsafe { c_strsignal(signum) };
    if res.is_null() {
        return Ok(None);
    }
    let bytes = unsafe { crate::rffi::charp2str(res)? };
    Ok(Some(bytes))
}

#[cfg(unix)]
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

/// `rsignal.py STRUCT_ONEFIELD` (`pypysig_long_struct_inner`): the one-field view of
/// `signals.c pypysig_counter`. `c_value` is the action ticker.
#[repr(C)]
pub struct PypysigLongStructInner {
    pub c_value: AtomicIsize,
}

/// `signals.c pypysig_counter`.
static PYPYSIG_COUNTER: PypysigLongStructInner = PypysigLongStructInner {
    c_value: AtomicIsize::new(0),
};

/// `rsignal.py pypysig_getaddr_occurred` (`_nowrapper=True, elidable_function=True`).
#[majit_macros::elidable]
pub fn pypysig_getaddr_occurred() -> *mut PypysigLongStructInner {
    &PYPYSIG_COUNTER as *const PypysigLongStructInner as *mut PypysigLongStructInner
}

/// One-word residual-call ABI for [`pypysig_getaddr_occurred`].
pub extern "C" fn pypysig_getaddr_occurred_jit_abi() -> i64 {
    majit_ir::icf_identity!(concat!(
        module_path!(),
        "::",
        "pypysig_getaddr_occurred_jit_abi"
    ));
    pypysig_getaddr_occurred() as i64
}
