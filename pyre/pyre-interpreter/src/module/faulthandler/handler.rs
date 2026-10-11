//! faulthandler implementation — PyPy: pypy/module/faulthandler/handler.py
//!
//! Verbatim move of the inline block previously in importing.rs.  `init_faulthandler`
//! was renamed to `register_module`; the host_env signal handlers and the
//! `faulthandler_extract_fd` helper stay private.

// faulthandler module — PyPy: pypy/module/faulthandler/.
//
// CPython's faulthandler dumps the Python traceback on fatal signals.
// Pyre has no Python-level traceback machinery yet, so our handler
// writes a short "Fatal Python error: <name>" line to the descriptor
// `enable` was given and then restores the disposition the signal had
// before `enable` + reraises it so the process dies the normal way.
// ──────────────────────────────────────────────────────────────────────

/// `handler.py` `Handler`: one record per space.  The signal callback is a
/// bare `extern "C" fn` and cannot capture, so the fields that callback
/// reads stay atomic; the rest matches `Handler._cleanup_` /
/// `fatal_error_w_file` / `user_w_files`.
#[cfg(all(any(unix, windows), feature = "host_env"))]
struct Handler {
    enabled: std::sync::atomic::AtomicBool,
    /// The descriptor `enable` resolved from its `file` argument. Defaults
    /// to 2, which is what `get_fileno_and_file` answers for None.
    fd: std::sync::atomic::AtomicI32,
    /// `self.fatal_error_w_file`. Walked as a GC root.
    fatal_error_w_file: std::sync::atomic::AtomicPtr<pyre_object::PyObject>,
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
    enabled: std::sync::atomic::AtomicBool::new(false),
    fd: std::sync::atomic::AtomicI32::new(2),
    fatal_error_w_file: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
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

/// `enable` / `disable` / `register` / `unregister` each install (or remove) a
/// handler and then hand the matching file to the owner slot in a second
/// statement.  `Handler` runs those two statements under the GIL, so no other
/// thread can interleave between them; pyre has no GIL, and an interleaving
/// leaves the installed descriptor owned by the *other* call's file — the file
/// this call resolved is then unrooted, collected, and its finalizer closes the
/// descriptor the handler still writes through.  Serialize the pairs.
///
/// Held across the host install or removal and the owner bookkeeping that
/// follows it.  The file is resolved — which runs Python — before the guard is
/// taken, and the fatal-signal handler reads `Handler.fd` atomically
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

/// Root walker for the two file-owner tables, registered alongside the other
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
    let fatal = HANDLER
        .fatal_error_w_file
        .load(std::sync::atomic::Ordering::Relaxed);
    if !fatal.is_null() {
        HANDLER.fatal_error_w_file.store(
            forward(fatal as usize) as pyre_object::PyObjectRef,
            std::sync::atomic::Ordering::Relaxed,
        );
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

/// Write one unsigned decimal without allocating.  The structured-exception
/// callback runs at an arbitrary fault point, so it cannot enter formatting
/// machinery or the managed heap.
#[cfg(all(windows, feature = "host_env"))]
fn faulthandler_write_decimal(fd: i32, mut value: usize) {
    let mut digits = [0u8; 20];
    let mut start = digits.len();
    loop {
        start -= 1;
        digits[start] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    rustpython_host_env::faulthandler::write_fd(fd, &digits[start..]);
}

/// Write the current thread's application frames from the same live
/// `ExecutionContext.topframeref` chain that PyPy's `dumper._dump_callback`
/// walks.  Every referenced filename/name is immutable storage owned by the
/// code object, and all scratch space is on the native stack: the exception
/// callback must remain allocation-free.
#[cfg(all(windows, feature = "host_env"))]
unsafe fn faulthandler_dump_current_traceback(fd: i32) {
    rustpython_host_env::faulthandler::write_fd(fd, b"Current thread 0x");
    let thread_id = rustpython_host_env::faulthandler::current_thread_id() as usize;
    let mut hex = [0u8; 2 * std::mem::size_of::<usize>()];
    let mut start = hex.len();
    let mut value = thread_id;
    loop {
        start -= 1;
        hex[start] = b"0123456789abcdef"[value & 0xf];
        value >>= 4;
        if value == 0 {
            break;
        }
    }
    rustpython_host_env::faulthandler::write_fd(fd, &hex[start..]);
    rustpython_host_env::faulthandler::write_fd(fd, b" (most recent call first):\n");

    let ec = pyre_interpreter::call::getexecutioncontext();
    let mut frame = if ec.is_null() {
        std::ptr::null_mut()
    } else {
        unsafe { (*ec).topframeref }
    };
    let mut depth = 0usize;
    while !frame.is_null() && depth < 100 {
        let pycode = unsafe { (*frame).pycode as *const pyre_interpreter::pycode::PyCode };
        if pycode.is_null() || unsafe { (*pycode).code_ptr.is_null() } {
            break;
        }
        let code = unsafe { &*((*pycode).code_ptr as *const pyre_interpreter::CodeObject) };
        let filename = if unsafe { (*pycode).filename_bytes.is_null() } {
            code.source_path.as_bytes()
        } else {
            unsafe { &*(*pycode).filename_bytes }.as_slice()
        };
        rustpython_host_env::faulthandler::write_fd(fd, b"  File \"");
        rustpython_host_env::faulthandler::write_fd(fd, filename);
        rustpython_host_env::faulthandler::write_fd(fd, b"\", line ");
        let lineno = unsafe { (*frame).get_lineno_at((*frame).last_instr) }.max(0) as usize;
        faulthandler_write_decimal(fd, lineno);
        rustpython_host_env::faulthandler::write_fd(fd, b" in ");
        let cached_name = unsafe { (*pycode).w_name };
        let name_bytes = if cached_name.is_null() {
            code.obj_name.as_bytes()
        } else {
            unsafe { pyre_object::w_str_get_wtf8(cached_name) }.as_bytes()
        };
        rustpython_host_env::faulthandler::write_fd(fd, name_bytes);
        rustpython_host_env::faulthandler::write_fd(fd, b"\n");
        frame = unsafe { (*frame).f_backref };
        depth += 1;
    }

    // CPython's `_Py_DumpStack` prints this section after the Python stack.
    // Pyre has no async-safe native symbolizer yet; its accepted failure form
    // is preferable to calling an allocator or loader lock from an SEH hook.
    rustpython_host_env::faulthandler::write_fd(
        fd,
        b"Current thread's C stack trace (most recent call first):\n<C stack unavailable>\n",
    );
}

/// `faulthandler.c:280-330 faulthandler_exc_handler`.  A fatal fault on Windows
/// arrives as a structured exception, not as a signal — a null read never
/// reaches the `SIGSEGV` handler — so without this, `enable()` there prints
/// nothing and the process dies at 0xc0000005 in silence.
///
/// Answers `EXCEPTION_CONTINUE_SEARCH` like the original: this only reports,
/// and the handler that decides the process's fate runs after it.
#[cfg(all(windows, feature = "host_env"))]
unsafe extern "system" fn faulthandler_exc_handler(
    exc_info: *mut rustpython_host_env::faulthandler::ExceptionPointers,
) -> i32 {
    use rustpython_host_env::faulthandler::EXCEPTION_CONTINUE_SEARCH;
    let code = unsafe { rustpython_host_env::faulthandler::exception_code(exc_info) };
    // `faulthandler.c:283-286`: a non-error status, and the C++ / CLR throws a
    // running program uses for control flow, are none of this handler's
    // business — and every Rust panic under MSVC is one of them.
    if rustpython_host_env::faulthandler::ignore_exception(code) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // Assembled on the stack for the same reason as the signal handler's.
    const PREFIX: &[u8] = b"Windows fatal exception: ";
    let mut msg = [0u8; 64];
    msg[..PREFIX.len()].copy_from_slice(PREFIX);
    let mut end = PREFIX.len();
    match rustpython_host_env::faulthandler::exception_description(code) {
        Some(name) => {
            let name = &name.as_bytes()[..name.len().min(msg.len() - end - 2)];
            msg[end..end + name.len()].copy_from_slice(name);
            end += name.len();
        }
        None => {
            // `faulthandler.c:322-324` spells an unnamed one `code 0x` plus
            // eight hexadecimal digits.
            const UNNAMED: &[u8] = b"code 0x";
            msg[end..end + UNNAMED.len()].copy_from_slice(UNNAMED);
            end += UNNAMED.len();
            for digit in (0..8).rev() {
                msg[end] = b"0123456789abcdef"[((code >> (digit * 4)) & 0xf) as usize];
                end += 1;
            }
        }
    }
    msg[end] = b'\n';
    msg[end + 1] = b'\n';
    let fd = HANDLER.fd.load(std::sync::atomic::Ordering::Relaxed);
    rustpython_host_env::faulthandler::write_fd(fd, &msg[..end + 2]);
    unsafe { faulthandler_dump_current_traceback(fd) };
    // `faulthandler.c:326-334`: the access violation is also delivered to the
    // program as SIGSEGV, and reporting it twice helps nobody.
    if rustpython_host_env::faulthandler::is_access_violation(code) {
        rustpython_host_env::faulthandler::disable_fatal_signal(libc::SIGSEGV);
    }
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

/// `faulthandler.c:1043 faulthandler_suppress_crash_report` — the crash helpers
/// below take the process down on purpose, so the OS must not stop to offer a
/// crash dialog first and hang a test run waiting on it.
fn suppress_crash_report() {
    #[cfg(all(windows, feature = "host_env"))]
    rustpython_host_env::faulthandler::suppress_crash_report();
}

#[cfg(all(any(unix, windows), feature = "host_env"))]
extern "C" fn faulthandler_signal_handler(signum: libc::c_int) {
    // Stay async-signal-safe: write with raw libc::write, and reraise so the
    // process dies through the disposition the program actually had.
    //
    // `faulthandler.c:512-520` restores `handler->previous` and clears
    // `handler->enabled` before writing anything, then `faulthandler.c:536-537`
    // reraises so that handler runs.  Replacing it with SIG_DFL instead would
    // silently drop a disposition the program installed for SIGABRT/SIGFPE/…
    // before calling `faulthandler.enable()`.
    //
    // SIGSEGV keeps the default disposition.  `faulthandler.c:529-535` already
    // refuses to hand SIGSEGV back to its previous handler, and here the
    // previous one is the Rust runtime's stack-overflow probe, which returns
    // without terminating when the address is not its guard page — which is
    // every software `raise`, `_sigsegv` below included.
    if signum != libc::SIGSEGV {
        rustpython_host_env::faulthandler::disable_fatal_signal(signum);
    }
    let name =
        rustpython_host_env::faulthandler::fatal_signal_name(signum).unwrap_or("unknown signal");
    // Assembled on the stack: a `format!` here would call into the allocator,
    // and the fault this is handling may well have come from inside it.
    const PREFIX: &[u8] = b"Fatal Python error: ";
    let mut msg = [0u8; 64];
    msg[..PREFIX.len()].copy_from_slice(PREFIX);
    let name = &name.as_bytes()[..name.len().min(msg.len() - PREFIX.len() - 1)];
    msg[PREFIX.len()..PREFIX.len() + name.len()].copy_from_slice(name);
    msg[PREFIX.len() + name.len()] = b'\n';
    let fd = HANDLER.fd.load(std::sync::atomic::Ordering::Relaxed);
    rustpython_host_env::faulthandler::write_fd(fd, &msg[..PREFIX.len() + name.len() + 1]);
    if signum == libc::SIGSEGV {
        rustpython_host_env::faulthandler::signal_default_and_raise(signum);
    } else {
        rustpython_host_env::faulthandler::raise_signal(signum);
    }
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
        pyre_interpreter::make_builtin_function_with_signature(
            "enable",
            |args| {
                // `handler.py enable` — file=None, all_threads=True.
                // Resolving the argument runs a user `fileno()` and `flush()`;
                // keep those side effects behind the support gate, so a build
                // that can only answer NotImplementedError does not run them
                // first.
                #[cfg(all(any(unix, windows), feature = "host_env"))]
                {
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
                    // `pypy_faulthandler_enable(fileno, all_threads)` takes the
                    // descriptor as an argument, so the handler never observes
                    // a descriptor the install did not commit to. Split across
                    // two statements here, the new fd has to be visible before
                    // the handlers go in — a fatal signal in between would
                    // otherwise dump to the old one — and has to be rolled back
                    // when the install fails, or a failed re-enable would
                    // redirect the handlers already installed.
                    let _state = lock_faulthandler_state();
                    let previous_fd = HANDLER.fd.swap(fd, std::sync::atomic::Ordering::Relaxed);
                    #[cfg(unix)]
                    let flags = libc::SA_NODEFER | libc::SA_ONSTACK;
                    #[cfg(windows)]
                    let flags = 0;
                    let ok = rustpython_host_env::faulthandler::enable_fatal_handlers(
                        faulthandler_signal_handler,
                        flags,
                    );
                    if ok {
                        // `faulthandler.c:1049-1052` installs the exception
                        // handler alongside the signal handlers.  A second
                        // `enable()` must not stack another one, and the
                        // installed handle is what `disable` needs.
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
                        HANDLER
                            .enabled
                            .store(true, std::sync::atomic::Ordering::Relaxed);
                        // `handler.py` `self.fatal_error_w_file = w_file`.
                        set_fatal_error_file(file_slot.map_or(
                            pyre_object::PY_NULL,
                            pyre_object::gc_roots::shadow_stack_get,
                        ));
                        return Ok(pyre_object::w_none());
                    }
                    HANDLER
                        .fd
                        .store(previous_fd, std::sync::atomic::Ordering::Relaxed);
                    Err(pyre_interpreter::PyError::runtime_error(
                        "faulthandler.enable: sigaction failed",
                    ))
                }
                #[cfg(not(all(any(unix, windows), feature = "host_env")))]
                {
                    let _ = args;
                    Err(pyre_interpreter::PyError::not_implemented(
                        "faulthandler.enable requires host_env feature",
                    ))
                }
            },
            // `enable(file=sys.stderr, all_threads=True, c_stack=True)` —
            // `c_stack` (3.14) selects C-stack dumping; accept and ignore it.
            pyre_interpreter::Signature::new(
                vec!["file", "all_threads", "c_stack"],
                None,
                None,
                0,
                0,
            ),
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "disable",
        pyre_interpreter::make_builtin_function_with_arity(
            "disable",
            |_| {
                #[cfg(all(any(unix, windows), feature = "host_env"))]
                {
                    let _state = lock_faulthandler_state();
                    rustpython_host_env::faulthandler::disable_fatal_handlers();
                    #[cfg(windows)]
                    rustpython_host_env::faulthandler::remove_vectored_exception_handler(
                        HANDLER
                            .exc_handler
                            .swap(0, std::sync::atomic::Ordering::Relaxed),
                    );
                    HANDLER
                        .enabled
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                    // `handler.py:150` `self.fatal_error_w_file = None`.
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
                        HANDLER.enabled.load(std::sync::atomic::Ordering::Relaxed),
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
        pyre_interpreter::make_builtin_function("dump_traceback", |_| {
            // No Python-level traceback machinery — emit a placeholder
            // so callers that want a forensic dump at least see *something*
            // instead of silent success.  Through the stderr seam, so it
            // reaches an embedder that has no fd 2 and every host gets it, not
            // just the ones with a `libc::write`.
            pyre_interpreter::host_seam::emit_stderr(
                b"<faulthandler: pyre has no Python-level traceback yet>\n",
            );
            Ok(pyre_object::w_none())
        })
    );
    crate::__pyre_put_new!(
        ns_slot,
        "dump_traceback_later",
        pyre_interpreter::make_builtin_function("dump_traceback_later", |_| {
            Ok(pyre_object::w_none())
        })
    );
    crate::__pyre_put_new!(
        ns_slot,
        "cancel_dump_traceback_later",
        pyre_interpreter::make_builtin_function_with_arity(
            "cancel_dump_traceback_later",
            |_| Ok(pyre_object::w_none()),
            0,
        )
    );
    // register/unregister user signals: host_env supports the full API,
    // but it needs the user-signal handler to be a fixed extern "C" fn.
    // Provide a "registered → no-op" pattern: install the handler when
    // registering, restore on unregister.  The handler writes a short
    // "user signal NN delivered" message to fd 2 (no traceback).
    // `handler.py register(signum, file=None, all_threads=True, chain=False)`.
    //
    // The pair exists only off Windows, which has no user signal to hand them
    // (`moduledef.py` guards the two `extra_interpdef` calls on the platform).
    // A name that is there but answers every call with an error is worse than
    // no name: `hasattr(faulthandler, "register")` is how its callers ask.
    #[cfg(unix)]
    crate::__pyre_put_new!(
        ns_slot,
        "register",
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
                // handler.py:174 `@unwrap_spec(all_threads=int, chain=int)`
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
                #[cfg(all(unix, feature = "host_env"))]
                {
                    // Resolving the file runs a user `fileno()` and `flush()`.
                    // `register(signum, file, all_threads, chain)` parses and
                    // coerces the two integer arguments before touching the
                    // file, and a build that can only answer NotImplementedError
                    // must not run those side effects at all — so this sits
                    // after the coercions above and inside the support gate.
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
                    rustpython_host_env::faulthandler::register_user_signal(
                        signum,
                        fd,
                        all_threads,
                        chain,
                        faulthandler_user_handler,
                    )
                    .map_err(|e| {
                        pyre_interpreter::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(0),
                            format!("register: {e}"),
                        )
                    })?;
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
                    let _ = (signum, all_threads, chain);
                    Err(pyre_interpreter::PyError::not_implemented(
                        "faulthandler.register requires host_env feature",
                    ))
                }
            },
            pyre_interpreter::Signature::new(
                vec!["signum", "file", "all_threads", "chain"],
                None,
                None,
                0,
                0,
            ),
        )
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
                    let _state = lock_faulthandler_state();
                    let changed = rustpython_host_env::faulthandler::unregister_user_signal(signum);
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
            if let Some(&release_gil) = args.first() {
                let _ = pyre_interpreter::baseobjspace::int_w(release_gil)?;
            }
            suppress_crash_report();
            // `handler.py read_null` — null-pointer deref.
            let p: *const u8 = std::ptr::null();
            let _ = unsafe { p.read_volatile() };
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
            if let Some(&release_gil) = args.first() {
                let _ = pyre_interpreter::baseobjspace::int_w(release_gil)?;
            }
            suppress_crash_report();
            // `raise` is in the Windows CRT too, and `enable` installs the
            // fatal handlers there through `signal()`, so the software raise
            // reaches them on either host.
            #[cfg(any(unix, windows))]
            unsafe {
                libc::raise(libc::SIGSEGV);
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
                suppress_crash_report();
                // `handler.py sigfpe` raises the signal, which is what an
                // integer division by zero traps as on unix.  Windows delivers
                // that fault as a structured exception instead, and Rust checks
                // the divisor rather than letting the CPU trap, so name the
                // status the hardware would have raised.
                #[cfg(all(windows, feature = "host_env"))]
                rustpython_host_env::faulthandler::raise_exception(
                    rustpython_host_env::faulthandler::EXCEPTION_INT_DIVIDE_BY_ZERO,
                    0,
                );
                #[cfg(unix)]
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
                suppress_crash_report();
                #[cfg(any(unix, windows))]
                unsafe {
                    libc::abort();
                }
                #[cfg(not(any(unix, windows)))]
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
        pyre_interpreter::make_builtin_function_with_arity(
            "_stack_overflow",
            |_| {
                suppress_crash_report();
                // `handler.py stack_overflow` — infinite recursion.
                fn blow() {
                    let _buf = [0u8; 4096];
                    blow();
                    std::hint::black_box(_buf);
                }
                blow();
                #[allow(unreachable_code)]
                Ok(pyre_object::w_none())
            },
            0,
        )
    );
    Ok(())
}

#[cfg(all(unix, feature = "host_env"))]
extern "C" fn faulthandler_user_handler(signum: libc::c_int) {
    let msg = format!("User signal {signum} delivered (faulthandler)\n");
    rustpython_host_env::faulthandler::write_fd(2, msg.as_bytes());
}
