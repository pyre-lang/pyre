//! `rpython/rlib/rthread.py` — the thread-local fields of
//! `pypy_threadlocal_s`.
//!
//! Compiled code receives the address of this thread's container as its
//! `threadlocal_addr` argument (`llmodel.py` `threadlocalref_addr`) and reads a
//! field at `threadlocal_addr + tlfield.getoffset()`. The fields are words;
//! an `rffi.INT` field keeps its value in the low half, as
//! `llerrno.py long2int` reads it on a little-endian host.
//!
//! The words below [`FIRST_TLFIELD_OFS`] are the `ThreadLocalReference`
//! slots `THREADLOCALREF_GET` reads.

use std::cell::UnsafeCell;

/// Bytes in one container word.
pub const WORD: usize = 8;

/// Words in the container.
const THREADLOCAL_WORDS: usize = 32;

/// First word the fixed fields own; the words below it are
/// `ThreadLocalReference` slots.
pub const FIRST_TLFIELD_OFS: usize = 24 * WORD;

/// `tlfield_p_errno` — the address of this thread's C `errno`.
pub const TLFIELD_P_ERRNO_OFS: usize = 24 * WORD;
/// `tlfield_rpy_errno`.
pub const TLFIELD_RPY_ERRNO_OFS: usize = 25 * WORD;
/// `tlfield_alt_errno`.
pub const TLFIELD_ALT_ERRNO_OFS: usize = 26 * WORD;
/// `tlfield_rpy_lasterror`.
pub const TLFIELD_RPY_LASTERROR_OFS: usize = 27 * WORD;
/// `tlfield_alt_lasterror`.
pub const TLFIELD_ALT_LASTERROR_OFS: usize = 28 * WORD;
/// The `errno` a target without a C library points `p_errno` at.
#[cfg(target_arch = "wasm32")]
const NO_LIBC_ERRNO_OFS: usize = 29 * WORD;
/// `tlfield_thread_ident` — filled from `pthread_self` / `GetCurrentThreadId`
/// in `_RPy_ThreadLocals_Init`.
pub const TLFIELD_THREAD_IDENT_OFS: usize = 30 * WORD;

#[repr(C, align(16))]
struct ThreadLocals(UnsafeCell<[i64; THREADLOCAL_WORDS]>);

thread_local! {
    static THREADLOCALS: ThreadLocals =
        const { ThreadLocals(UnsafeCell::new([0; THREADLOCAL_WORDS])) };
}

/// The address of this thread's C `errno` (`llerrno.py _fetch_addr_errno`).
fn fetch_addr_errno(base: *mut i64) -> *mut i32 {
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
    {
        let _ = base;
        unsafe { libc::__error() }
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let _ = base;
        unsafe { libc::__errno_location() }
    }
    #[cfg(target_os = "windows")]
    {
        unsafe extern "C" {
            fn _errno() -> *mut i32;
        }
        let _ = base;
        unsafe { _errno() }
    }
    #[cfg(target_arch = "wasm32")]
    {
        unsafe { base.byte_add(NO_LIBC_ERRNO_OFS).cast::<i32>() }
    }
}

fn fetch_thread_ident() -> i64 {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    {
        unsafe { libc::pthread_self() as u64 as i64 }
    }
    #[cfg(windows)]
    {
        unsafe extern "system" {
            fn GetCurrentThreadId() -> u32;
        }
        unsafe { GetCurrentThreadId() as i64 }
    }
    #[cfg(not(any(all(unix, not(target_arch = "wasm32")), windows)))]
    {
        1
    }
}

/// `llop.threadlocalref_addr` — this thread's container.
#[inline]
pub fn threadlocalref_addr() -> *mut i64 {
    THREADLOCALS.with(|t| {
        let base = t.0.get().cast::<i64>();
        // `rthread.py` fills `p_errno` when the thread starts; the container
        // here is created on first use, so fill it then.
        let p_errno = unsafe { base.byte_add(TLFIELD_P_ERRNO_OFS) };
        if unsafe { *p_errno } == 0 {
            unsafe { *p_errno = fetch_addr_errno(base) as i64 };
        }
        let ident = unsafe { base.byte_add(TLFIELD_THREAD_IDENT_OFS) };
        if unsafe { *ident } == 0 {
            unsafe { *ident = fetch_thread_ident() };
        }
        base
    })
}

/// `llop.threadlocalref_get` (`rthread.py ThreadLocalField.getraw`).
///
/// `ofs` and `loop_invariant` must be call-site constants, matching
/// `CDefinedIntSymbolic.loop_invariant`. The flag is not read at
/// runtime; the translator projects it onto `ExtraEffect`.
#[inline(never)]
pub fn threadlocalref_get_int(ofs: usize, loop_invariant: bool) -> i64 {
    let _ = loop_invariant;
    threadlocalref_get(ofs)
}

/// `llop.threadlocalref_store` (`rthread.py ThreadLocalField.setraw`).
#[inline(never)]
pub fn threadlocalref_store_int(ofs: usize, value: i64) {
    assert!(ofs < THREADLOCAL_WORDS * WORD && ofs.is_multiple_of(WORD));
    unsafe { *threadlocalref_addr().byte_add(ofs) = value };
}

/// `rthread.get_ident` — `tlfield_thread_ident.getraw()`.
#[inline(always)]
pub fn get_ident() -> i64 {
    threadlocalref_get_int(TLFIELD_THREAD_IDENT_OFS, true)
}

// `thread_pthread.c RPyThread_kill`. `rthread.llexternal` defaults to
// `sandboxsafe=True`.
#[cfg(unix)]
crate::rffi::external_compilation_info! {
    const THREAD_KILL_ECI = {
        includes: ["pthread.h", "signal.h"],
        separate_module_sources: [
            "RPY_EXTERN int RPyThread_kill(Signed id, int signal) { return pthread_kill((pthread_t)id, signal); }",
        ],
    };
}

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_pthread_kill = "RPyThread_kill",
    [crate::rffi::SIGNED, crate::rffi::INT],
    crate::rffi::INT,
    compilation_info = THREAD_KILL_ECI,
    sandboxsafe = true,
    save_err = crate::rffi::RFFI_SAVE_ERRNO
);

/// `ThreadLocalField.getraw()` for an `rffi.INT` field.
/// Errno save fields are not loop-invariant (`tlfield_rpy_errno`).
#[inline(always)]
pub fn tlfield_getraw_int(ofs: usize) -> i32 {
    threadlocalref_get_int(ofs, false) as i32
}

/// `ThreadLocalField.setraw()` for an `rffi.INT` field.
#[inline(always)]
pub fn tlfield_setraw_int(ofs: usize, value: i32) {
    threadlocalref_store_int(ofs, value as i64);
}

/// `tlfield_p_errno.getraw()`.
#[inline(always)]
pub fn tlfield_p_errno() -> *mut i32 {
    threadlocalref_get_int(TLFIELD_P_ERRNO_OFS, true) as *mut i32
}

/// The word at byte offset `ofs`, as `llop.threadlocalref_get` reads it.
pub fn threadlocalref_get(ofs: usize) -> i64 {
    assert!(ofs < THREADLOCAL_WORDS * WORD && ofs.is_multiple_of(WORD));
    unsafe { *threadlocalref_addr().byte_add(ofs) }
}

/// Write the `ThreadLocalReference` slot at byte offset `ofs`.
pub fn threadlocalref_set(ofs: usize, value: i64) {
    assert!(ofs < FIRST_TLFIELD_OFS && ofs.is_multiple_of(WORD));
    unsafe { *threadlocalref_addr().byte_add(ofs) = value };
}

/// `os_lock.py` / `thread.h` `RPY_LOCK_FAILURE`, `RPY_LOCK_ACQUIRED`, `RPY_LOCK_INTR`.
pub const RPY_LOCK_FAILURE: i64 = 0;
pub const RPY_LOCK_ACQUIRED: i64 = 1;
pub const RPY_LOCK_INTR: i64 = 2;

/// `thread_pthread.c struct RPyOpaque_ThreadLock`.
#[cfg(target_os = "linux")]
pub struct ThreadLock {
    sem: Box<UnsafeCell<libc::sem_t>>,
}

/// `thread_pthread.c struct RPyOpaque_ThreadLock`, mutex+cond build.
#[cfg(not(target_os = "linux"))]
struct CondLock {
    locked: std::sync::Mutex<bool>,
    lock_released: std::sync::Condvar,
}

/// `thread_pthread.c struct RPyOpaque_ThreadLock`.
#[cfg(not(target_os = "linux"))]
pub struct ThreadLock {
    /// The native lock lives behind a pointer, like `rthread.py`'s
    /// `c_thread_allocate_lock` result.
    state: Box<std::cell::UnsafeCell<CondLock>>,
}

// The cell is written only by `reinit`, in the forked child, where no other
// thread exists.
unsafe impl Send for ThreadLock {}
unsafe impl Sync for ThreadLock {}

#[cfg(target_os = "linux")]
fn rpythread_fix_status(status: i32) -> i32 {
    if status == -1 {
        std::io::Error::last_os_error()
            .raw_os_error()
            .unwrap_or(status)
    } else {
        status
    }
}

/// `RPY_MICROSECONDS_TO_TIMESPEC`: `gettimeofday` plus `microseconds`,
/// then `tv_nsec` below 1e9.
#[cfg(target_os = "linux")]
fn rpy_microseconds_to_timespec(microseconds: i64) -> libc::timespec {
    let mut tv = libc::timeval {
        tv_sec: 0,
        tv_usec: 0,
    };
    unsafe {
        libc::gettimeofday(&mut tv, std::ptr::null_mut());
    }
    let mut usec = tv.tv_usec as i64 + microseconds % 1_000_000;
    let mut sec = tv.tv_sec as i64 + microseconds / 1_000_000;
    sec += usec / 1_000_000;
    usec %= 1_000_000;
    libc::timespec {
        tv_sec: sec as libc::time_t,
        tv_nsec: (usec * 1000) as _,
    }
}

#[cfg(target_os = "linux")]
impl Default for ThreadLock {
    fn default() -> Self {
        Self::allocate()
    }
}

#[cfg(target_os = "linux")]
impl Drop for ThreadLock {
    /// `RPyOpaqueDealloc_ThreadLock`.
    fn drop(&mut self) {
        unsafe {
            libc::sem_destroy(self.sem.get());
        }
    }
}

// `rthread.py` reaches every primitive below as an `llexternal`, so none is traced.
#[cfg(target_os = "linux")]
impl ThreadLock {
    /// `rthread.py allocate_lock` / `RPyThreadLockInit`.
    #[majit_macros::dont_look_inside]
    pub fn allocate() -> Self {
        let sem = Box::new(UnsafeCell::new(unsafe { std::mem::zeroed() }));
        if unsafe { libc::sem_init(sem.get(), 0, 1) } != 0 {
            panic!("sem_init failed");
        }
        Self { sem }
    }

    /// `RPyThreadAcquireLockTimed`.
    #[majit_macros::dont_look_inside]
    pub fn acquire_timed(&self, microseconds: i64, intr_flag: bool) -> i64 {
        let thelock = self.sem.get();
        let ts = (microseconds > 0).then(|| rpy_microseconds_to_timespec(microseconds));
        let status = loop {
            let raw = if microseconds > 0 {
                unsafe { libc::sem_timedwait(thelock, ts.as_ref().unwrap()) }
            } else if microseconds == 0 {
                unsafe { libc::sem_trywait(thelock) }
            } else {
                unsafe { libc::sem_wait(thelock) }
            };
            let status = rpythread_fix_status(raw);
            if intr_flag || status != libc::EINTR {
                break status;
            }
        };
        if status == 0 {
            RPY_LOCK_ACQUIRED
        } else if intr_flag && status == libc::EINTR {
            RPY_LOCK_INTR
        } else {
            RPY_LOCK_FAILURE
        }
    }

    /// `RPyThreadReleaseLock`.
    #[majit_macros::dont_look_inside]
    pub fn release(&self) -> i64 {
        let thelock = self.sem.get();
        let mut current_value = 0;
        unsafe {
            libc::sem_getvalue(thelock, &mut current_value);
        }
        if current_value > 0 {
            return -1;
        }
        unsafe {
            libc::sem_post(thelock);
        }
        0
    }

    #[majit_macros::dont_look_inside]
    pub fn is_locked(&self) -> bool {
        let mut current_value = 0;
        unsafe {
            libc::sem_getvalue(self.sem.get(), &mut current_value);
        }
        current_value == 0
    }

    /// Fork-child `RPyThreadLockInit` on the same storage.
    #[majit_macros::dont_look_inside]
    pub fn reinit(&self) {
        let sem = self.sem.get();
        unsafe {
            std::ptr::write(sem, std::mem::zeroed());
            if libc::sem_init(sem, 0, 1) != 0 {
                panic!("sem_init failed");
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
impl Default for ThreadLock {
    fn default() -> Self {
        Self::allocate()
    }
}

#[cfg(not(target_os = "linux"))]
impl ThreadLock {
    /// `rthread.py allocate_lock` / `RPyThreadLockInit`.
    #[majit_macros::dont_look_inside]
    pub fn allocate() -> Self {
        Self {
            state: Box::new(std::cell::UnsafeCell::new(CondLock {
                locked: std::sync::Mutex::new(false),
                lock_released: std::sync::Condvar::new(),
            })),
        }
    }

    /// `RPyThreadAcquireLockTimed`.
    #[majit_macros::dont_look_inside]
    pub fn acquire_timed(&self, microseconds: i64, intr_flag: bool) -> i64 {
        // The cell is written only by `reinit`, in the forked child.
        let state = unsafe { &*self.state.get() };
        let mut locked = state
            .locked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut success;
        if !*locked {
            success = RPY_LOCK_ACQUIRED;
        } else if microseconds == 0 {
            success = RPY_LOCK_FAILURE;
        } else {
            let deadline = (microseconds > 0).then(|| {
                std::time::Instant::now() + std::time::Duration::from_micros(microseconds as u64)
            });
            success = RPY_LOCK_FAILURE;
            while success == RPY_LOCK_FAILURE {
                if let Some(deadline) = deadline {
                    let now = std::time::Instant::now();
                    if now >= deadline {
                        break;
                    }
                    let (guard, timeout) = state
                        .lock_released
                        .wait_timeout(locked, deadline - now)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    locked = guard;
                    if timeout.timed_out() {
                        break;
                    }
                } else {
                    locked = state
                        .lock_released
                        .wait(locked)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                if intr_flag && *locked {
                    success = RPY_LOCK_INTR;
                    break;
                } else if !*locked {
                    success = RPY_LOCK_ACQUIRED;
                } else {
                    success = RPY_LOCK_FAILURE;
                }
            }
        }
        if success == RPY_LOCK_ACQUIRED {
            *locked = true;
        }
        success
    }

    /// `RPyThreadReleaseLock`.
    #[majit_macros::dont_look_inside]
    pub fn release(&self) -> i64 {
        let state = unsafe { &*self.state.get() };
        let mut locked = state
            .locked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let result = if !*locked { -1 } else { 0 };
        *locked = false;
        state.lock_released.notify_one();
        result
    }

    #[majit_macros::dont_look_inside]
    pub fn is_locked(&self) -> bool {
        let state = unsafe { &*self.state.get() };
        *state
            .locked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Fork-child `RPyThreadLockInit` without dropping the inherited primitives:
    /// the inherited mutex may be held by a thread that vanished at fork.
    #[majit_macros::dont_look_inside]
    pub fn reinit(&self) {
        unsafe {
            self.state.get().write(CondLock {
                locked: std::sync::Mutex::new(false),
                lock_released: std::sync::Condvar::new(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p_errno_is_filled_on_first_use() {
        assert!(!tlfield_p_errno().is_null());
    }

    #[test]
    fn int_fields_are_per_thread() {
        tlfield_setraw_int(TLFIELD_ALT_ERRNO_OFS, 7);
        let other = std::thread::spawn(|| tlfield_getraw_int(TLFIELD_ALT_ERRNO_OFS))
            .join()
            .unwrap();
        assert_eq!(other, 0);
        assert_eq!(tlfield_getraw_int(TLFIELD_ALT_ERRNO_OFS), 7);
    }

    #[test]
    fn reference_slots_stay_below_the_fixed_fields() {
        threadlocalref_set(16, 0x5678);
        assert_eq!(threadlocalref_get(16), 0x5678);
        assert_eq!(unsafe { *threadlocalref_addr().byte_add(16) }, 0x5678);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn get_ident_differs_across_threads() {
        let here = get_ident();
        let other = std::thread::spawn(get_ident).join().unwrap();
        assert_ne!(here, 0);
        assert_ne!(other, 0);
        assert_ne!(here, other);
    }

    #[cfg(unix)]
    unsafe extern "C" fn nop_sigusr1(_: libc::c_int) {}

    #[cfg(unix)]
    #[test]
    fn acquire_timed_reports_a_signal_as_intr() {
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = nop_sigusr1 as *const () as libc::sighandler_t;
            sa.sa_flags = 0;
            libc::sigemptyset(&mut sa.sa_mask);
            libc::sigaction(libc::SIGUSR1, &sa, std::ptr::null_mut());
        }
        let lock = ThreadLock::default();
        assert_eq!(lock.acquire_timed(0, true), RPY_LOCK_ACQUIRED);
        let target = unsafe { libc::pthread_self() as usize };
        let killer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            unsafe {
                libc::pthread_kill(target as libc::pthread_t, libc::SIGUSR1);
            }
        });
        let start = std::time::Instant::now();
        let result = lock.acquire_timed(5_000_000, true);
        let elapsed = start.elapsed();
        assert_eq!(result, RPY_LOCK_INTR);
        assert!(elapsed < std::time::Duration::from_secs(3));
        killer.join().unwrap();
        assert_eq!(lock.release(), 0);
        assert_eq!(lock.release(), -1);
    }
}
