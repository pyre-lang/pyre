//! faulthandler implementation — PyPy: pypy/module/faulthandler/handler.py
//!
//! `Handler` owns the files the installed handlers write to; the handlers
//! themselves and the traceback walk are `cintf` (`cintf.py` +
//! `faulthandler.c`) and `dumper` (`dumper.py`), on both hosts.

/// `handler.py` `Handler`: one record per space.  The signal callback is a
/// bare `extern "C" fn` and cannot capture, so everything it reads lives in
/// `cintf`'s `fatal_error`; this keeps `Handler._cleanup_`'s
/// `fatal_error_w_file` / `dump_traceback_later_w_file` / `user_w_files`.
#[cfg(all(any(unix, windows), feature = "host_env"))]
struct Handler {
    /// `self.fatal_error_w_file`. Walked as a GC root.
    fatal_error_w_file: std::sync::atomic::AtomicPtr<pyre_object::PyObject>,
    /// `self.dump_traceback_later_w_file`. Walked as a GC root.
    dump_traceback_later_w_file: std::sync::atomic::AtomicPtr<pyre_object::PyObject>,
    state_lock: parking_lot::Mutex<()>,
    /// `self.user_w_files`. Addresses rather than `PyObjectRef` so the
    /// table is `Sync`.
    #[cfg(unix)]
    user_w_files: parking_lot::Mutex<Vec<(libc::c_int, usize)>>,
    /// The vectored exception handler `enable` installed; 0 is none.
    #[cfg(windows)]
    exc_handler: std::sync::atomic::AtomicUsize,
}

#[cfg(all(any(unix, windows), feature = "host_env"))]
static HANDLER: Handler = Handler {
    fatal_error_w_file: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
    dump_traceback_later_w_file: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
    state_lock: parking_lot::const_mutex(()),
    #[cfg(unix)]
    user_w_files: parking_lot::const_mutex(Vec::new()),
    #[cfg(windows)]
    exc_handler: std::sync::atomic::AtomicUsize::new(0),
};

/// Take ownership of the object owning the fatal-error descriptor; a null
/// drops it (`enable` with a plain fd, and `disable`).
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn set_fatal_error_file(w_file: pyre_object::PyObjectRef) {
    HANDLER
        .fatal_error_w_file
        .store(w_file, std::sync::atomic::Ordering::Relaxed);
}

/// `self.dump_traceback_later_w_file = w_file`; a null drops it.
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn set_dump_traceback_later_file(w_file: pyre_object::PyObjectRef) {
    HANDLER
        .dump_traceback_later_w_file
        .store(w_file, std::sync::atomic::Ordering::Relaxed);
}

/// `enable` / `disable` / `register` / `unregister` /
/// `dump_traceback_later` / `cancel_dump_traceback_later` each install (or
/// remove) a handler and then hand the matching file to the owner slot in a
/// second statement.  `Handler` runs those two statements under the GIL, so
/// no other thread can interleave between them; pyre has no GIL, and an
/// interleaving leaves the installed descriptor owned by the *other* call's
/// file — the file this call resolved is then unrooted, collected, and its
/// finalizer closes the descriptor the handler still writes through.
/// Serialize the pairs.
///
/// Held across the host install or removal and the owner bookkeeping that
/// follows it.  The file is resolved — which runs Python — before the guard is
/// taken, and the fatal-signal handler reads `fatal_error.fd` atomically
/// without acquiring anything.  A failed install still builds its exception
/// inside the guard, so take it through `lock_faulthandler_state`.
///
/// Only a contended acquisition blocks, and a thread parked in the futex can no
/// longer poll the eval breaker, so it has to leave the collector's RUNNING
/// census for that wait — otherwise a holder that allocates (`register` builds
/// an `OSError` when the install fails, and `os_error_syscall2` allocates the
/// exception and its args) requests a stop-the-world the waiter can never
/// acknowledge, and both threads hang.  Same try-then-block split as
/// `w_list_lock`.
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn lock_faulthandler_state() -> parking_lot::MutexGuard<'static, ()> {
    if let Some(guard) = HANDLER.state_lock.try_lock() {
        return guard;
    }
    let blocked = pyre_interpreter::module::thread::before_external_block();
    let guard = HANDLER.state_lock.lock();
    drop(blocked);
    guard
}

/// `handler.py:123-125` — take ownership of a registered signal's file; a null
/// (a plain fd, which `get_fileno_and_file` answers `None` for) drops any
/// previous owner for that signal.
#[cfg(all(unix, feature = "host_env"))]
fn set_user_signal_file(signum: libc::c_int, w_file: pyre_object::PyObjectRef) {
    let mut files = HANDLER.user_w_files.lock();
    files.retain(|&(s, _)| s != signum);
    if !w_file.is_null() {
        files.push((signum, w_file as usize));
    }
}

/// `handler.py` `self.user_w_files.pop(signum, None)`.
#[cfg(all(unix, feature = "host_env"))]
fn clear_user_signal_file(signum: libc::c_int) {
    HANDLER.user_w_files.lock().retain(|&(s, _)| s != signum);
}

/// Root walker for the file-owner slots, registered alongside the other
/// process-global interpreter roots.  Forwards each slot in place so a moving
/// collection relocates the held file rather than leaving a stale address the
/// installed handler would keep writing through.
///
/// Runs from the collector inside the stop-the-world window, so no load/forward/
/// store here can be torn by a concurrent owner update, and it must NOT take
/// `Handler.state_lock`: the thread that requested the collection may be
/// holding that guard (`register` allocates its `OSError` inside it), and the
/// collector would then wait on a lock only a quiesced mutator can release.
#[cfg(all(any(unix, windows), feature = "host_env"))]
pub fn walk_faulthandler_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let mut forward = |addr: usize| -> usize {
        let mut slot: pyre_object::PyObjectRef = addr as pyre_object::PyObjectRef;
        // SAFETY: `PyObjectRef` and `GcRef` are layout-compatible.
        visitor(unsafe {
            &mut *(&mut slot as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef)
        });
        slot as usize
    };
    for owner in [
        &HANDLER.fatal_error_w_file,
        &HANDLER.dump_traceback_later_w_file,
    ] {
        let file = owner.load(std::sync::atomic::Ordering::Relaxed);
        if !file.is_null() {
            owner.store(
                forward(file as usize) as pyre_object::PyObjectRef,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    }
    #[cfg(unix)]
    {
        for entry in HANDLER.user_w_files.lock().iter_mut() {
            entry.1 = forward(entry.1);
        }
    }
}

/// No fatal-signal handlers to own a file without a native host environment.
#[cfg(not(all(any(unix, windows), feature = "host_env")))]
pub fn walk_faulthandler_roots(_visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {}

/// `handler.py finish`, called from `moduledef.py Module.shutdown`:
/// `Handler.finish` tears the handlers and the watchdog down, then
/// `_cleanup_` drops the files they wrote to.
#[cfg(all(any(unix, windows), feature = "host_env"))]
pub fn finish() {
    let _state = lock_faulthandler_state();
    super::cintf::pypy_faulthandler_teardown();
    #[cfg(windows)]
    rustpython_host_env::faulthandler::remove_vectored_exception_handler(
        HANDLER
            .exc_handler
            .swap(0, std::sync::atomic::Ordering::Relaxed),
    );
    // `Handler._cleanup_`.
    set_fatal_error_file(pyre_object::PY_NULL);
    set_dump_traceback_later_file(pyre_object::PY_NULL);
    #[cfg(unix)]
    HANDLER.user_w_files.lock().clear();
}

#[cfg(not(all(any(unix, windows), feature = "host_env")))]
pub fn finish() {}

/// [3.15-spec] `Modules/faulthandler.c faulthandler_exc_handler`.  PyPy has
/// no vectored exception handler: a fatal fault on Windows arrives as a
/// structured exception, not as a signal — a null read never reaches the
/// `SIGSEGV` handler — so `enable()` there prints nothing and the process
/// dies at 0xc0000005 in silence. Evidence:
/// `lib-python/3/test/test_faulthandler.py` `test_raise_exception` expects
/// "Windows fatal exception: access violation" followed by a traceback.
///
/// Writes through `cintf`'s `fatal_error` descriptor and dumps with the same
/// `faulthandler_dump_traceback`, then answers `EXCEPTION_CONTINUE_SEARCH`:
/// this only reports, and the handler that decides the process's fate runs
/// after it.
#[cfg(all(windows, feature = "host_env"))]
unsafe extern "system" fn faulthandler_exc_handler(
    exc_info: *mut rustpython_host_env::faulthandler::ExceptionPointers,
) -> i32 {
    use rustpython_host_env::faulthandler::EXCEPTION_CONTINUE_SEARCH;
    let fd = super::cintf::fatal_error_fd();
    let code = unsafe { rustpython_host_env::faulthandler::exception_code(exc_info) };
    // `faulthandler_ignore_exception`: a non-error status, and the C++ / CLR
    // throws a running program uses for control flow, are none of this
    // handler's business — and every Rust panic under MSVC is one of them.
    if rustpython_host_env::faulthandler::ignore_exception(code) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    super::cintf::pypy_faulthandler_write(fd, b"Windows fatal exception: ");
    match rustpython_host_env::faulthandler::exception_description(code) {
        Some(name) => super::cintf::pypy_faulthandler_write(fd, name.as_bytes()),
        None => {
            // An unnamed one is `code 0x` plus eight hexadecimal digits.
            let mut text = *b"code 0x00000000";
            for (index, shift) in (0..8u32).rev().enumerate() {
                text[7 + index] = b"0123456789abcdef"[((code >> (shift * 4)) & 0xf) as usize];
            }
            super::cintf::pypy_faulthandler_write(fd, &text);
        }
    }
    super::cintf::pypy_faulthandler_write(fd, b"\n\n");
    // The access violation is also delivered to the program as SIGSEGV, and
    // reporting it twice helps nobody.
    if rustpython_host_env::faulthandler::is_access_violation(code) {
        super::cintf::faulthandler_disable_fatal_handler(libc::SIGSEGV);
    }
    super::cintf::faulthandler_dump_fatal_traceback(fd);
    super::cintf::faulthandler_dump_c_stack(fd);
    EXCEPTION_CONTINUE_SEARCH
}

/// The exception statuses `faulthandler` publishes on Windows, spelled the way
/// `PyModule_AddIntConstant` hands the NTSTATUS values over: as signed 32-bit
/// ints, so `_EXCEPTION_ACCESS_VIOLATION` reads -0x3ffffffb rather than
/// 0xc0000005.
#[cfg(all(windows, feature = "host_env"))]
const WINDOWS_EXCEPTIONS: [(&str, i32); 5] = [
    (
        "_EXCEPTION_ACCESS_VIOLATION",
        rustpython_host_env::faulthandler::EXCEPTION_ACCESS_VIOLATION as i32,
    ),
    (
        "_EXCEPTION_INT_DIVIDE_BY_ZERO",
        rustpython_host_env::faulthandler::EXCEPTION_INT_DIVIDE_BY_ZERO as i32,
    ),
    (
        "_EXCEPTION_NONCONTINUABLE",
        rustpython_host_env::faulthandler::EXCEPTION_NONCONTINUABLE as i32,
    ),
    (
        "_EXCEPTION_NONCONTINUABLE_EXCEPTION",
        rustpython_host_env::faulthandler::EXCEPTION_NONCONTINUABLE_EXCEPTION as i32,
    ),
    (
        "_EXCEPTION_STACK_OVERFLOW",
        rustpython_host_env::faulthandler::EXCEPTION_STACK_OVERFLOW as i32,
    ),
];

#[cfg(all(windows, not(feature = "host_env")))]
const WINDOWS_EXCEPTIONS: [(&str, i32); 5] = [
    ("_EXCEPTION_ACCESS_VIOLATION", 0xc000_0005u32 as i32),
    ("_EXCEPTION_INT_DIVIDE_BY_ZERO", 0xc000_0094u32 as i32),
    ("_EXCEPTION_NONCONTINUABLE", 0x1),
    ("_EXCEPTION_NONCONTINUABLE_EXCEPTION", 0xc000_0025u32 as i32),
    ("_EXCEPTION_STACK_OVERFLOW", 0xc000_00fdu32 as i32),
];

/// `handler.py Handler.check_err`.
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn check_err(p_err: *const libc::c_char) -> Result<(), pyre_interpreter::PyError> {
    if p_err.is_null() {
        return Ok(());
    }
    let msg = unsafe { std::ffi::CStr::from_ptr(p_err) }.to_string_lossy();
    Err(pyre_interpreter::PyError::runtime_error(format!(
        "faulthandler: {msg}"
    )))
}

/// `handler.py Handler.setup`.
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn setup() -> Result<(), pyre_interpreter::PyError> {
    check_err(super::cintf::pypy_faulthandler_setup(
        super::dumper::_dump_callback,
    ))
}

/// `handler.py Handler.check_signum`.
#[cfg(all(unix, feature = "host_env"))]
fn check_signum(signum: libc::c_int) -> Result<(), pyre_interpreter::PyError> {
    let err = super::cintf::pypy_faulthandler_check_signum(signum as libc::c_long);
    if err < 0 {
        if err == -1 {
            Err(pyre_interpreter::PyError::runtime_error(format!(
                "signal {signum} cannot be registered, use enable() instead"
            )))
        } else {
            Err(pyre_interpreter::PyError::value_error(
                "signal number out of range",
            ))
        }
    } else {
        Ok(())
    }
}

/// An `@unwrap_spec(x=int)` argument; a slot the binder left null takes
/// `default`.
fn int_arg(
    args: &[pyre_object::PyObjectRef],
    index: usize,
    default: i64,
) -> Result<i64, pyre_interpreter::PyError> {
    args.get(index)
        .copied()
        .filter(|a| !a.is_null())
        .map(pyre_interpreter::baseobjspace::gateway_int_w)
        .transpose()
        .map(|value| value.unwrap_or(default))
}

/// [3.15-spec] The kw-only `max_threads=100` of `enable`, `dump_traceback`,
/// `dump_traceback_later` and `register`, which `handler.py` does not take.
/// Evidence: `lib-python/3/test/test_faulthandler.py`
/// `test_dump_traceback_max_threads` calls
/// `dump_traceback(file=sys.stderr, max_threads=CAP)` and
/// `test_register_max_threads` calls `register(..., max_threads=CAP)`;
/// `dump_traceback_later` has no test, and its `max_threads: Py_ssize_t = 100`
/// is the clinic signature of `faulthandler_dump_traceback_later_impl` in
/// `Modules/faulthandler.c` read at v3.15.0rc2.
fn with_max_threads_default(function: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let fn_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(function);
    pyre_interpreter::gateway::function_set_kwonly_default_pair(
        fn_slot,
        "max_threads",
        pyre_object::w_int_new(100),
    );
    roots.get(fn_slot)
}

/// `handler.py Handler.get_fileno_and_file` — resolve a
/// file-or-fd-or-None argument to `(fileno, file)`.  None resolves the CURRENT
/// `sys.stderr` rather than a hard-coded fd 2, so a redirected stderr is
/// honoured; an int is used directly and names no file; anything else is asked
/// for its `fileno()` and then flushed, with an ordinary flush error ignored.
///
/// The returned file is the object the descriptor belongs to, and the caller
/// parks it for as long as the handlers are installed — a null means there is
/// nothing to own.
fn faulthandler_get_fileno_and_file(
    w_file: pyre_object::PyObjectRef,
) -> Result<(i32, pyre_object::PyObjectRef), pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let resolved = if w_file.is_null() || unsafe { pyre_object::is_none(w_file) } {
        let sys = pyre_interpreter::importing::get_interpreter_sys_module()
            .ok_or_else(|| pyre_interpreter::PyError::runtime_error("sys.stderr is None"))?;
        let w_stderr = pyre_interpreter::baseobjspace::getattr_str(sys, "stderr")?;
        if w_stderr.is_null() || unsafe { pyre_object::is_none(w_stderr) } {
            return Err(pyre_interpreter::PyError::runtime_error(
                "sys.stderr is None",
            ));
        }
        w_stderr
    } else if unsafe { pyre_object::is_int(w_file) } {
        let fd = unsafe { pyre_object::w_int_get_value(w_file) } as i32;
        if fd < 0 {
            return Err(pyre_interpreter::PyError::value_error(
                "file is not a valid file descriptor",
            ));
        }
        return Ok((fd, pyre_object::PY_NULL));
    } else {
        w_file
    };
    // `fileno` and `flush` both run Python; keep the file addressable across
    // them so the caller receives the relocated pointer, not a stale one.
    // `pin_root` normalizes a forwarding stub into the slot, not into the
    // caller's copy, so read the pinned value back before using it.
    let _ = pyre_object::gc_roots::pin_root(resolved);
    let file_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let resolved = pyre_object::gc_roots::shadow_stack_get(file_slot);
    let method = pyre_interpreter::baseobjspace::getattr_str(resolved, "fileno")?;
    let res = pyre_interpreter::call::call_function_impl_result(method, &[])?;
    if !unsafe { pyre_object::is_int(res) } {
        return Err(pyre_interpreter::PyError::type_error(
            "fileno() returned non-integer",
        ));
    }
    let fd = unsafe { pyre_object::w_int_get_value(res) } as i32;
    if fd < 0 {
        // `handler.py:42` hands the value straight to
        // `pypy_faulthandler_enable`, and PyPy accepts a negative descriptor
        // (measured: `enable()` succeeds); 3.14 rejects it, with a message
        // naming `fileno()` to distinguish it from the direct-int arm above.
        return Err(pyre_interpreter::PyError::runtime_error(
            "file.fileno() is not a valid file descriptor",
        ));
    }
    // `handler.py:44-48`:
    //
    //     try:
    //         space.call_method(w_file, 'flush')
    //     except OperationError as e:
    //         if e.async(space):
    //             raise
    //         pass   # ignore flush() error
    //
    // 3.14 clears the flush error unconditionally instead, so a `SystemExit`
    // raised out of `flush` does NOT abort `enable` there (measured on 3.14.5
    // and on PyPy: they disagree, and 3.14 is the behaviour target).  Ignore
    // every flush failure, including an asynchronous one.
    let resolved = pyre_object::gc_roots::shadow_stack_get(file_slot);
    let _ = pyre_interpreter::baseobjspace::getattr_str(resolved, "flush")
        .and_then(|flush| pyre_interpreter::call::call_function_impl_result(flush, &[]));
    Ok((fd, pyre_object::gc_roots::shadow_stack_get(file_slot)))
}

pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    crate::__pyre_put_new!(
        ns_slot,
        "enable",
        // `handler.py enable(space, w_file=None, all_threads=0)`; clinic also
        // takes `c_stack=True` and `*, max_threads=100`.
        with_max_threads_default(pyre_interpreter::gateway::function_set_null_positional_defaults(
            pyre_interpreter::make_builtin_function_with_signature(
                "enable",
                |args| {
                    // `handler.py Handler.enable`. Resolving the argument runs a
                    // user `fileno()` and `flush()`; keep those side effects
                    // behind the support gate, so a build that can only answer
                    // NotImplementedError does not run them first.
                    #[cfg(all(any(unix, windows), feature = "host_env"))]
                    {
                        let all_threads = int_arg(args, 1, 1)?;
                        // [3.15-spec] `c_stack: bool = True` is not
                        // upstream's; see `cintf::FatalError::c_stack`.
                        let c_stack = match args.get(2).copied().filter(|a| !a.is_null()) {
                            Some(w_c_stack) => pyre_interpreter::baseobjspace::is_true(w_c_stack)?,
                            None => true,
                        };
                        let max_threads = int_arg(args, 3, 100)?;
                        let (fd, w_file) = faulthandler_get_fileno_and_file(
                            args.first().copied().unwrap_or(pyre_object::PY_NULL),
                        )?;
                        // Nothing else names the resolved file between here and the
                        // owner store below, so keep it rooted across the install.
                        let _roots = pyre_object::gc_roots::push_roots();
                        let file_slot = (!w_file.is_null()).then(|| {
                            let _ = pyre_object::gc_roots::pin_root(w_file);
                            pyre_object::gc_roots::shadow_stack_len() - 1
                        });
                        let _state = lock_faulthandler_state();
                        setup()?;
                        // `self.fatal_error_w_file = w_file` comes before the
                        // install's `check_err`.
                        set_fatal_error_file(file_slot.map_or(
                            pyre_object::PY_NULL,
                            pyre_object::gc_roots::shadow_stack_get,
                        ));
                        check_err(super::cintf::pypy_faulthandler_enable(
                            fd,
                            all_threads as libc::c_int,
                            c_stack as libc::c_int,
                            max_threads as libc::c_int,
                        ))?;
                        #[cfg(windows)]
                        if HANDLER
                            .exc_handler
                            .load(std::sync::atomic::Ordering::Relaxed)
                            == 0
                        {
                            HANDLER.exc_handler.store(
                                rustpython_host_env::faulthandler::add_vectored_exception_handler(
                                    Some(faulthandler_exc_handler),
                                ),
                                std::sync::atomic::Ordering::Relaxed,
                            );
                        }
                        Ok(pyre_object::w_none())
                    }
                    #[cfg(not(all(any(unix, windows), feature = "host_env")))]
                    {
                        let _ = args;
                        Err(pyre_interpreter::PyError::not_implemented(
                            "faulthandler.enable requires host_env feature",
                        ))
                    }
                },
                pyre_interpreter::Signature::new(
                    vec!["file", "all_threads", "c_stack", "max_threads"],
                    None,
                    None,
                    1,
                    0,
                ),
            ),
            3,
        ))
    );
    crate::__pyre_put_new!(
        ns_slot,
        "disable",
        pyre_interpreter::make_builtin_function_with_arity(
            "disable",
            |_| {
                // `handler.py Handler.disable`.
                #[cfg(all(any(unix, windows), feature = "host_env"))]
                {
                    let _state = lock_faulthandler_state();
                    super::cintf::pypy_faulthandler_disable();
                    #[cfg(windows)]
                    rustpython_host_env::faulthandler::remove_vectored_exception_handler(
                        HANDLER
                            .exc_handler
                            .swap(0, std::sync::atomic::Ordering::Relaxed),
                    );
                    set_fatal_error_file(pyre_object::PY_NULL);
                }
                Ok(pyre_object::w_none())
            },
            0,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "is_enabled",
        pyre_interpreter::make_builtin_function_with_arity(
            "is_enabled",
            |_| {
                #[cfg(all(any(unix, windows), feature = "host_env"))]
                {
                    Ok(pyre_object::w_bool_from(
                        super::cintf::pypy_faulthandler_is_enabled(),
                    ))
                }
                #[cfg(not(all(any(unix, windows), feature = "host_env")))]
                Ok(pyre_object::w_bool_from(false))
            },
            0,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "dump_traceback",
        // `handler.py dump_traceback(space, w_file=None, all_threads=0)`, plus
        // the kw-only `max_threads`.
        with_max_threads_default(pyre_interpreter::gateway::function_set_null_positional_defaults(
            pyre_interpreter::make_builtin_function_with_signature(
                "dump_traceback",
                |args| {
                    // `handler.py Handler.dump_traceback`.
                    #[cfg(all(any(unix, windows), feature = "host_env"))]
                    {
                        let all_threads = int_arg(args, 1, 1)? as libc::c_int;
                        let max_threads = int_arg(args, 2, 100)? as libc::c_int;
                        let (fd, w_file) = faulthandler_get_fileno_and_file(
                            args.first().copied().unwrap_or(pyre_object::PY_NULL),
                        )?;
                        // `keepalive_until_here(w_file)`.
                        let _roots = pyre_object::gc_roots::push_roots();
                        if !w_file.is_null() {
                            let _ = pyre_object::gc_roots::pin_root(w_file);
                        }
                        setup()?;
                        if all_threads != 0 {
                            // Without a GIL the other threads keep pushing and
                            // freeing frames while their chains are walked;
                            // `faulthandler_dump_traceback_py_impl` stops the
                            // world around the walk for the same reason
                            // (gh-128400).
                            majit_gc::gc_sync::request_stw(|_| {
                                super::cintf::pypy_faulthandler_dump_traceback(
                                    fd,
                                    all_threads,
                                    max_threads,
                                    std::ptr::null_mut(),
                                )
                            });
                        } else {
                            super::cintf::pypy_faulthandler_dump_traceback(
                                fd,
                                all_threads,
                                max_threads,
                                std::ptr::null_mut(),
                            );
                        }
                        Ok(pyre_object::w_none())
                    }
                    #[cfg(not(all(any(unix, windows), feature = "host_env")))]
                    {
                        let _ = args;
                        pyre_interpreter::host_seam::emit_stderr(
                            b"<faulthandler: pyre has no Python-level traceback yet>\n",
                        );
                        Ok(pyre_object::w_none())
                    }
                },
                pyre_interpreter::Signature::new(
                    vec!["file", "all_threads", "max_threads"],
                    None,
                    None,
                    1,
                    0,
                ),
            ),
            2,
        ))
    );
    crate::__pyre_put_new!(
        ns_slot,
        "dump_c_stack",
        // [3.15-spec] `faulthandler.dump_c_stack(file=sys.stderr)` has no
        // PyPy counterpart; see `cintf::pypy_faulthandler_dump_stack`.
        pyre_interpreter::gateway::function_set_null_positional_defaults(
            pyre_interpreter::make_builtin_function_with_signature(
                "dump_c_stack",
                |args| {
                    #[cfg(all(any(unix, windows), feature = "host_env"))]
                    {
                        let (fd, w_file) = faulthandler_get_fileno_and_file(
                            args.first().copied().unwrap_or(pyre_object::PY_NULL),
                        )?;
                        let _roots = pyre_object::gc_roots::push_roots();
                        if !w_file.is_null() {
                            let _ = pyre_object::gc_roots::pin_root(w_file);
                        }
                        super::cintf::pypy_faulthandler_dump_stack(fd);
                        Ok(pyre_object::w_none())
                    }
                    #[cfg(not(all(any(unix, windows), feature = "host_env")))]
                    {
                        let _ = args;
                        Err(pyre_interpreter::PyError::not_implemented(
                            "faulthandler.dump_c_stack requires host_env feature",
                        ))
                    }
                },
                pyre_interpreter::Signature::new(vec!["file"], None, None, 0, 0),
            ),
            1,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "dump_traceback_later",
        // `handler.py` `@unwrap_spec(timeout=float, repeat=int, exit=int)
        // dump_traceback_later(space, timeout, repeat=0, w_file=None, exit=0)`,
        // plus the kw-only `max_threads`.
        with_max_threads_default(pyre_interpreter::gateway::function_set_null_positional_defaults(
            pyre_interpreter::make_builtin_function_with_signature(
                "dump_traceback_later",
                |args| {
                    let Some(&w_timeout) = args.first().filter(|a| !a.is_null()) else {
                        return Err(pyre_interpreter::PyError::type_error(
                            "dump_traceback_later() missing required argument 'timeout' (pos 1)",
                        ));
                    };
                    // `handler.py Handler.dump_traceback_later`.
                    let timeout = pyre_interpreter::baseobjspace::float_w(w_timeout)? * 1e6;
                    let repeat = int_arg(args, 1, 0)?;
                    let exit = int_arg(args, 3, 0)?;
                    let max_threads = int_arg(args, 4, 100)?;
                    // `ovfcheck_float_to_longlong`: in range or OverflowError,
                    // NaN included; the conversion truncates.
                    if !(-9.223372036854776e18..9.223372036854776e18).contains(&timeout) {
                        return Err(pyre_interpreter::PyError::overflow_error(
                            "timeout value is too large",
                        ));
                    }
                    let microseconds = timeout as i64;
                    if microseconds <= 0 {
                        return Err(pyre_interpreter::PyError::value_error(
                            "timeout must be greater than 0",
                        ));
                    }
                    #[cfg(all(any(unix, windows), feature = "host_env"))]
                    {
                        let (fd, w_file) = faulthandler_get_fileno_and_file(
                            args.get(2).copied().unwrap_or(pyre_object::PY_NULL),
                        )?;
                        let _roots = pyre_object::gc_roots::push_roots();
                        let file_slot = (!w_file.is_null()).then(|| {
                            let _ = pyre_object::gc_roots::pin_root(w_file);
                            pyre_object::gc_roots::shadow_stack_len() - 1
                        });
                        let _state = lock_faulthandler_state();
                        setup()?;
                        check_err(super::cintf::pypy_faulthandler_dump_traceback_later(
                            microseconds,
                            repeat as libc::c_int,
                            fd,
                            exit as libc::c_int,
                            max_threads as libc::c_int,
                        ))?;
                        set_dump_traceback_later_file(file_slot.map_or(
                            pyre_object::PY_NULL,
                            pyre_object::gc_roots::shadow_stack_get,
                        ));
                        Ok(pyre_object::w_none())
                    }
                    #[cfg(not(all(any(unix, windows), feature = "host_env")))]
                    {
                        let _ = (microseconds, repeat, exit, max_threads);
                        Err(pyre_interpreter::PyError::not_implemented(
                            "faulthandler.dump_traceback_later requires host_env feature",
                        ))
                    }
                },
                pyre_interpreter::Signature::new(
                    vec!["timeout", "repeat", "file", "exit", "max_threads"],
                    None,
                    None,
                    1,
                    0,
                ),
            ),
            3,
        ))
    );
    crate::__pyre_put_new!(
        ns_slot,
        "cancel_dump_traceback_later",
        pyre_interpreter::make_builtin_function_with_arity(
            "cancel_dump_traceback_later",
            |_| {
                // `handler.py Handler.cancel_dump_traceback_later`.
                #[cfg(all(any(unix, windows), feature = "host_env"))]
                {
                    setup()?;
                    let _state = lock_faulthandler_state();
                    super::cintf::pypy_faulthandler_cancel_dump_traceback_later();
                    set_dump_traceback_later_file(pyre_object::PY_NULL);
                }
                Ok(pyre_object::w_none())
            },
            0,
        )
    );
    // `handler.py register(signum, file=None, all_threads=True, chain=False)`.
    // Unix installs `faulthandler_user`, which dumps to `user->fd` with
    // `user->all_threads` and honours `chain`.
    //
    // The pair exists only off Windows, which has no user signal to hand them
    // (`moduledef.py` guards the two `extra_interpdef` calls on the platform).
    // A name that is there but answers every call with an error is worse than
    // no name: `hasattr(faulthandler, "register")` is how its callers ask.
    #[cfg(unix)]
    crate::__pyre_put_new!(
        ns_slot,
        "register",
        // `handler.py register(space, signum, w_file=None, all_threads=1, chain=0)`
        with_max_threads_default(pyre_interpreter::gateway::function_set_null_positional_defaults(
            pyre_interpreter::make_builtin_function_with_signature(
                "register",
                |args| {
                    let w_signum = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                    if w_signum.is_null() {
                        return Err(pyre_interpreter::PyError::type_error(
                            "register() missing signal",
                        ));
                    }
                    let signum = (unsafe { pyre_object::w_int_get_value(w_signum) }) as libc::c_int;
                    // `handler.py register` `@unwrap_spec(all_threads=int, chain=int)`
                    // with defaults `all_threads=1, chain=0`: the arguments are
                    // coerced as integers (`gateway_int_w`, raising on a non-int),
                    // not by truthiness; `register` then tests `if all_threads:`.  An
                    // omitted keyword leaves a null slot from the signature binding, so
                    // treat null as the default.
                    let all_threads = args
                        .get(2)
                        .copied()
                        .filter(|a| !a.is_null())
                        .map(pyre_interpreter::baseobjspace::gateway_int_w)
                        .transpose()?
                        .unwrap_or(1)
                        != 0;
                    let chain = args
                        .get(3)
                        .copied()
                        .filter(|a| !a.is_null())
                        .map(pyre_interpreter::baseobjspace::gateway_int_w)
                        .transpose()?
                        .unwrap_or(0)
                        != 0;
                    let max_threads = int_arg(args, 4, 100)? as libc::c_int;
                    #[cfg(all(unix, feature = "host_env"))]
                    {
                        // Resolving the file runs a user `fileno()` and `flush()`.
                        // `register(signum, file, all_threads, chain)` parses and
                        // coerces the two integer arguments before touching the
                        // file, and a build that can only answer NotImplementedError
                        // must not run those side effects at all — so this sits
                        // after the coercions above and inside the support gate.
                        check_signum(signum)?;
                        let (fd, w_file) = faulthandler_get_fileno_and_file(
                            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
                        )?;
                        // Nothing else names the resolved file between here and the
                        // owner store below, so keep it rooted across the install.
                        let _roots = pyre_object::gc_roots::push_roots();
                        let file_slot = (!w_file.is_null()).then(|| {
                            let _ = pyre_object::gc_roots::pin_root(w_file);
                            pyre_object::gc_roots::shadow_stack_len() - 1
                        });
                        let _state = lock_faulthandler_state();
                        setup()?;
                        check_err(super::cintf::pypy_faulthandler_register(
                            signum,
                            fd,
                            all_threads as libc::c_int,
                            chain as libc::c_int,
                            max_threads,
                        ))?;
                        // `handler.py` `self.user_w_files[signum] = w_file`,
                        // after `check_err` — a register that failed owns nothing.
                        set_user_signal_file(
                            signum,
                            file_slot.map_or(
                                pyre_object::PY_NULL,
                                pyre_object::gc_roots::shadow_stack_get,
                            ),
                        );
                        Ok(pyre_object::w_none())
                    }
                    #[cfg(not(all(unix, feature = "host_env")))]
                    {
                        let _ = (signum, all_threads, chain, max_threads);
                        Err(pyre_interpreter::PyError::not_implemented(
                            "faulthandler.register requires host_env feature",
                        ))
                    }
                },
                pyre_interpreter::Signature::new(
                    vec!["signum", "file", "all_threads", "chain", "max_threads"],
                    None,
                    None,
                    1,
                    0,
                ),
            ),
            3,
        ))
    );
    #[cfg(unix)]
    crate::__pyre_put_new!(
        ns_slot,
        "unregister",
        pyre_interpreter::make_builtin_function_with_arity(
            "unregister",
            |args| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    if args.is_empty() {
                        return Err(pyre_interpreter::PyError::type_error(
                            "unregister() missing signal",
                        ));
                    }
                    let signum = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                    check_signum(signum)?;
                    let _state = lock_faulthandler_state();
                    let changed = super::cintf::pypy_faulthandler_unregister(signum) == 1;
                    // `handler.py` `self.user_w_files.pop(signum, None)`,
                    // run whether or not the signal was registered.
                    clear_user_signal_file(signum);
                    Ok(pyre_object::w_bool_from(changed))
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let _ = args;
                    Ok(pyre_object::w_bool_from(false))
                }
            },
            1,
        )
    );

    // `handler.py:225-245` test-only crash helpers from
    // `moduledef.py:14-22`.  Each unconditionally takes down the
    // process — only ever called from test_faulthandler.py in a
    // subprocess.  Pyre cannot construct an OperationError here
    // because the abort/segfault leaves no caller to catch it.
    crate::__pyre_put_new!(
        ns_slot,
        "_read_null",
        pyre_interpreter::make_builtin_function("_read_null", |args| {
            if args.len() > 1 {
                return Err(pyre_interpreter::PyError::type_error(format!(
                    "_read_null() takes at most 1 argument ({} given)",
                    args.len()
                )));
            }
            // `@unwrap_spec(release_gil=int) read_null(space, release_gil=0)`.
            let release_gil = int_arg(args, 0, 0)? != 0;
            #[cfg(all(any(unix, windows), feature = "host_env"))]
            {
                // `pypy_faulthandler_read_null_releasegil`.
                let _blocked =
                    release_gil.then(pyre_interpreter::module::thread::before_external_block);
                super::cintf::pypy_faulthandler_read_null();
            }
            // Without `cintf` the fault is still the point: `read_null`
            // calls its C helper unconditionally.
            #[cfg(not(all(any(unix, windows), feature = "host_env")))]
            {
                let _ = release_gil;
                let p: *const u8 = std::ptr::null();
                let _ = unsafe { p.read_volatile() };
            }
            Ok(pyre_object::w_none())
        })
    );
    crate::__pyre_put_new!(
        ns_slot,
        "_sigsegv",
        pyre_interpreter::make_builtin_function("_sigsegv", |args| {
            if args.len() > 1 {
                return Err(pyre_interpreter::PyError::type_error(format!(
                    "_sigsegv() takes at most 1 argument ({} given)",
                    args.len()
                )));
            }
            // `@unwrap_spec(release_gil=int) sigsegv(space, release_gil=0)`.
            let release_gil = int_arg(args, 0, 0)? != 0;
            #[cfg(all(any(unix, windows), feature = "host_env"))]
            {
                // `pypy_faulthandler_sigsegv_releasegil`.
                let _blocked =
                    release_gil.then(pyre_interpreter::module::thread::before_external_block);
                super::cintf::pypy_faulthandler_sigsegv();
            }
            #[cfg(not(all(any(unix, windows), feature = "host_env")))]
            {
                let _ = release_gil;
                #[cfg(any(unix, windows))]
                unsafe {
                    libc::raise(libc::SIGSEGV);
                }
            }
            Ok(pyre_object::w_none())
        })
    );
    crate::__pyre_put_new!(
        ns_slot,
        "_sigfpe",
        pyre_interpreter::make_builtin_function_with_arity(
            "_sigfpe",
            |_| {
                #[cfg(all(any(unix, windows), feature = "host_env"))]
                super::cintf::pypy_faulthandler_sigfpe();
                #[cfg(all(any(unix, windows), not(feature = "host_env")))]
                unsafe {
                    libc::raise(libc::SIGFPE);
                }
                Ok(pyre_object::w_none())
            },
            0,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "_sigabrt",
        pyre_interpreter::make_builtin_function_with_arity(
            "_sigabrt",
            |_| {
                #[cfg(all(any(unix, windows), feature = "host_env"))]
                super::cintf::pypy_faulthandler_sigabrt();
                #[cfg(all(any(unix, windows), not(feature = "host_env")))]
                unsafe {
                    libc::abort()
                }
                #[cfg(not(all(any(unix, windows), not(feature = "host_env"))))]
                Ok(pyre_object::w_none())
            },
            0,
        )
    );
    // `faulthandler.c:1416-1436` publishes the statuses and `_raise_exception`
    // only where structured exceptions exist.
    #[cfg(windows)]
    {
        for (name, code) in WINDOWS_EXCEPTIONS {
            crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(i64::from(code)));
        }
        crate::__pyre_put_new!(
            ns_slot,
            "_raise_exception",
            pyre_interpreter::make_builtin_function("_raise_exception", |args| {
                // `_raise_exception(code, flags=0)`.
                let Some(&w_code) = args.first() else {
                    return Err(pyre_interpreter::PyError::type_error(
                        "_raise_exception() missing required argument 'code'",
                    ));
                };
                let code = pyre_interpreter::baseobjspace::int_w(w_code)? as u32;
                let flags = match args.get(1) {
                    Some(&w_flags) => pyre_interpreter::baseobjspace::int_w(w_flags)? as u32,
                    None => 0,
                };
                #[cfg(feature = "host_env")]
                {
                    // CPython's test helper raises the C++ and CLR status
                    // values only to verify that the vectored handler ignores
                    // them.  Letting either unwind through Rust's
                    // `std::sys::thread::thread_start` is converted into
                    // `panic_cannot_unwind`, so finish with the same silent
                    // process status the unhandled SEH would have produced.
                    if code == 0xE06D_7363 || code == 0xE043_4352 {
                        rustpython_host_env::winapi::terminate_process(
                            rustpython_host_env::winapi::get_current_process(),
                            code,
                        );
                    }
                    rustpython_host_env::faulthandler::raise_exception(code, flags);
                }
                Ok(pyre_object::w_none())
            })
        );
    }
    crate::__pyre_put_new!(
        ns_slot,
        "_stack_overflow",
        // `@unwrap_spec(levels=int) stack_overflow(space, levels=100000000)`.
        pyre_interpreter::make_builtin_function("_stack_overflow", |args| {
            if args.len() > 1 {
                return Err(pyre_interpreter::PyError::type_error(format!(
                    "_stack_overflow() takes at most 1 argument ({} given)",
                    args.len()
                )));
            }
            let levels = int_arg(args, 0, 100_000_000)? as f64;
            #[cfg(all(any(unix, windows), feature = "host_env"))]
            {
                Ok(pyre_object::w_float_new(
                    super::cintf::pypy_faulthandler_stackoverflow(levels),
                ))
            }
            #[cfg(not(all(any(unix, windows), feature = "host_env")))]
            {
                let _ = levels;
                Ok(pyre_object::w_none())
            }
        })
    );
    #[cfg(unix)]
    if let Some(mut function) = unsafe {
        pyre_object::w_dict_getitem_str(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            "register",
        )
    } {
        // `register(signum, file=None, all_threads=1, chain=0)`: trailing
        // omitted slots take `Function.defs_w` (`interp2app._getdefaults`).
        function = pyre_interpreter::gateway::function_set_null_positional_defaults(function, 3);
        pyre_interpreter::__pyre_put_new!(ns_slot, "register", function);
    }
    Ok(())
}
