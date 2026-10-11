//! select implementation — PyPy: pypy/module/select/interp_select.py
//!
//! Verbatim move of the inline block previously in importing.rs.

#[cfg(all(any(unix, windows), feature = "host_env"))]
use pyre_object::PyObjectRef;

/// `select.poll` object — PyPy: `interp_select.py class Poll`.
///
/// Holds the registered `{fd: events}` map and a re-entrancy guard.
/// Instances are created only through the module-level `select.poll()`
/// factory (`interp_select.py`); the type has no public constructor.
#[cfg(all(unix, feature = "host_env"))]
// CPython 3.14 Modules/selectmodule.c:select_exec uses
// PyType_FromModuleAndSpec; poll_Type_spec is a mutable heap type.
#[pyre_interpreter::pyre_class("select.poll", cpython_mutable)]
#[derive(Default)]
pub struct Poll {
    fddict: std::collections::HashMap<i32, i16>,
    running: bool,
}

/// `interp_select.py defaultevents = POLLIN | POLLOUT | POLLPRI`.
#[cfg(all(unix, feature = "host_env"))]
fn default_poll_events() -> i16 {
    majit_rlib::rpoll::POLLIN | majit_rlib::rpoll::POLLOUT | majit_rlib::rpoll::POLLPRI
}

/// Resolve a Python fd argument (int or object with `fileno()`) to a
/// raw descriptor — `space.c_filedescriptor_w`.
#[cfg(all(any(unix, windows), feature = "host_env"))]
pub(crate) fn filedescriptor_w(w_fd: PyObjectRef) -> Result<i32, pyre_interpreter::PyError> {
    unsafe {
        // A real int (or int subclass / bignum) is taken directly; otherwise
        // `fileno()` is called.  An object with only `__int__` is rejected.
        let w_int = if pyre_object::is_int_or_long(w_fd) {
            w_fd
        } else {
            let fileno =
                pyre_interpreter::baseobjspace::getattr_str(w_fd, "fileno").map_err(|_| {
                    pyre_interpreter::PyError::type_error(
                        "argument must be an int, or have a fileno() method.",
                    )
                })?;
            let res = pyre_interpreter::call::call_function_impl_result(fileno, &[])?;
            if !pyre_object::is_int_or_long(res) {
                return Err(pyre_interpreter::PyError::type_error(
                    "fileno() returned a non-integer",
                ));
            }
            res
        };
        // `c_int_w` — OverflowError if it does not fit a 32-bit int.
        let fd = pyre_interpreter::baseobjspace::c_int_w(w_int)?;
        if fd < 0 {
            return Err(pyre_interpreter::PyError::value_error(format!(
                "file descriptor cannot be a negative integer ({fd})"
            )));
        }
        Ok(fd)
    }
}

#[cfg(all(unix, feature = "host_env"))]
#[pyre_interpreter::pyre_methods(
    doc = "Returns a polling object.\n\nSee the poll() documentation.",
    unhashable
)]
impl Poll {
    /// `interp_select.py descr_new` — the type is not directly
    /// instantiable; `select.poll()` is the module-level factory.
    #[staticmethod]
    fn __new__(_cls: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        Err(pyre_interpreter::PyError::type_error(
            "cannot create 'select.poll' instances",
        ))
    }

    /// `interp_select.py Poll.register` — `events` defaults to
    /// `POLLIN | POLLOUT | POLLPRI`.
    fn register(
        &mut self,
        mut w_fd: PyObjectRef,
        #[default(pyre_object::w_none())] w_events: PyObjectRef,
    ) -> Result<(), pyre_interpreter::PyError> {
        // @unwrap_spec(events="c_ushort"): reject negative / >0xffff.  The
        // gateway converts it before the body resolves the descriptor.
        let events = if unsafe { pyre_object::is_none(w_events) } {
            default_poll_events()
        } else {
            pyre_object::with_roots!(w_fd => pyre_interpreter::baseobjspace::c_ushort_w(w_events))?
                as i16
        };
        let fd = filedescriptor_w(w_fd)?;
        self.fddict.insert(fd, events);
        Ok(())
    }

    /// `interp_select.py Poll.modify` — raises `OSError(ENOENT)` for
    /// a descriptor that was never registered.
    fn modify(
        &mut self,
        mut w_fd: PyObjectRef,
        w_events: PyObjectRef,
    ) -> Result<(), pyre_interpreter::PyError> {
        // @unwrap_spec(events="c_ushort"): reject negative / >0xffff.  The
        // gateway converts it before the body resolves the descriptor.
        let events = pyre_object::with_roots!(w_fd => pyre_interpreter::baseobjspace::c_ushort_w(w_events))?
            as i16;
        let fd = filedescriptor_w(w_fd)?;
        let known = self.fddict.contains_key(&fd);
        if known {
            self.fddict.insert(fd, events);
            Ok(())
        } else {
            Err(pyre_interpreter::PyError::os_error_with_errno(
                libc::ENOENT,
                "poll.modify",
            ))
        }
    }

    /// `interp_select.py Poll.unregister` — raises `KeyError(fd)` for
    /// an unknown descriptor.
    fn unregister(&mut self, w_fd: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
        let fd = filedescriptor_w(w_fd)?;
        if self.fddict.remove(&fd).is_none() {
            return Err(pyre_interpreter::PyError::key_error_with_key(
                pyre_object::w_int_new(fd as i64),
            ));
        }
        Ok(())
    }

    /// `interp_select.py Poll.poll` — `timeout` is in milliseconds;
    /// `None` or a negative value blocks indefinitely.  Returns a list
    /// of `(fd, revents)` for the descriptors with pending events.
    fn poll(
        &mut self,
        #[default(pyre_object::w_none())] w_timeout: PyObjectRef,
    ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
        // `None` / negative → block indefinitely (timeout = -1).  Otherwise
        // `c_int_w(space.int(w_timeout))`: truncate a float to int, then
        // range-check to a 32-bit C int (millisecond count).
        let timeout: i32 = if unsafe { pyre_object::is_none(w_timeout) } {
            -1
        } else if unsafe { pyre_object::is_int(w_timeout) } {
            let t = unsafe { pyre_object::w_int_get_value(w_timeout) };
            if t < 0 {
                -1
            } else if t > i32::MAX as i64 {
                return Err(pyre_interpreter::PyError::overflow_error(
                    "expected a 32-bit integer",
                ));
            } else {
                t as i32
            }
        } else if unsafe { pyre_object::is_float(w_timeout) } {
            let t = unsafe { pyre_object::w_float_get_value(w_timeout) };
            if t < 0.0 {
                -1
            } else {
                let trunc = t.trunc();
                if trunc > i32::MAX as f64 {
                    return Err(pyre_interpreter::PyError::overflow_error(
                        "expected a 32-bit integer",
                    ));
                }
                trunc as i32
            }
        } else {
            return Err(pyre_interpreter::PyError::type_error(
                "timeout must be an integer or None",
            ));
        };

        if self.running {
            return Err(pyre_interpreter::PyError::runtime_error(
                "concurrent poll() invocation",
            ));
        }

        // EINTR retry with a recomputed timeout (`Poll.poll` rounds the
        // remaining time up to the next millisecond). A negative timeout
        // stays blocking: `end_time` is only consulted after EINTR when the
        // caller asked for a finite wait.
        let deadline = (timeout >= 0)
            .then(|| std::time::Instant::now() + std::time::Duration::from_millis(timeout as u64));
        let mut cur_timeout = timeout;
        self.running = true;
        let ready = loop {
            // Snapshot first. `rpoll.poll` releases the GIL inside
            // `_rsocket_rffi.poll`, and this `&mut self` must not stay
            // borrowed across that call.
            let snapshot = self.fddict.clone();
            match majit_rlib::rpoll::poll(&snapshot, cur_timeout) {
                Ok(ready) => break ready,
                Err(err) if err.errno == libc::EINTR => {
                    // Deliver a pending signal, then retry. Reset `running`
                    // first so a raised handler does not leave the poll object
                    // wedged (`Poll.poll`'s `finally: self.running = False`).
                    if let Err(err) =
                        pyre_interpreter::module::signal::interp_signal::checksignals_now()
                    {
                        self.running = false;
                        return Err(err);
                    }
                    if let Some(dl) = deadline {
                        let now = std::time::Instant::now();
                        cur_timeout = if now >= dl {
                            0
                        } else {
                            ((dl - now).as_secs_f64() * 1000.0 + 0.999) as i32
                        };
                    }
                    continue;
                }
                Err(err) => {
                    self.running = false;
                    let e = std::io::Error::from_raw_os_error(err.errno);
                    return Err(pyre_interpreter::PyError::os_error_with_errno(
                        err.errno,
                        format!("poll: {e}"),
                    ));
                }
            }
        };
        self.running = false;

        let mut retval = pyre_object::gc_roots::RootedItems::new();
        for (fd, revents) in ready {
            let entry = {
                let mut fields = pyre_object::gc_roots::RootedItems::new();
                fields.push(pyre_object::w_int_new(fd as i64));
                fields.push(pyre_object::w_int_new(revents as i64));
                pyre_object::w_tuple_new(fields.take())
            };
            retval.push(entry);
        }
        Ok(pyre_object::w_list_new(retval.take()))
    }
}

/// Convert a descriptor resolved from one of the three sequences into what
/// the platform's fd_set holds, rejecting one the set cannot represent —
/// `seq2set`.  `index` is how many entries that sequence already contributed.
///
/// A POSIX fd_set is a bitmap indexed by the descriptor, so `FD_SET` on an fd
/// at or above FD_SETSIZE writes outside it (`_PyIsSelectable_fd`).
#[cfg(all(unix, feature = "host_env"))]
fn selectable_fd(fd: i32, _index: usize) -> Result<i32, pyre_interpreter::PyError> {
    // `_build_fd_set`: `MAX_FD_SIZE is not None and fd >= MAX_FD_SIZE`.
    if let Some(max) = majit_rlib::_rsocket_rffi::MAX_FD_SIZE {
        if fd >= max {
            return Err(pyre_interpreter::PyError::value_error(
                "file descriptor out of range in select()",
            ));
        }
    }
    Ok(fd)
}

/// `selectable_fd` for WinSock, whose fd_set is a count plus an array of
/// SOCKETs instead of a bitmap: the handle value itself is unconstrained, but
/// only `FD_SETSIZE` of them fit and `FD_SET` past that silently drops the
/// socket. `_rsocket_rffi.FD_SETSIZE` is the SDK default 64.
/// `_build_fd_set` does not apply this count (`MAX_FD_SIZE` is `None`); the
/// rejection and its message are the ones already raised here.
#[cfg(all(windows, feature = "host_env"))]
fn selectable_fd(fd: i32, index: usize) -> Result<i32, pyre_interpreter::PyError> {
    if index >= majit_rlib::_rsocket_rffi::FD_SETSIZE {
        return Err(pyre_interpreter::PyError::value_error(
            "too many file descriptors in select()",
        ));
    }
    // `select()` takes SOCKET handles here, never CRT file descriptors; a
    // handle is 32-bit significant, which is what `fileno()` hands back.  A
    // descriptor that is not a socket fails the call itself with WSAENOTSOCK.
    // `FD_SET` takes `rffi.INT`; the Winsock body widens it to `SOCKET`.
    Ok(fd)
}

#[cfg(all(unix, feature = "host_env"))]
type OsFdSet = libc::fd_set;
#[cfg(all(windows, feature = "host_env"))]
type OsFdSet = majit_rlib::_rsocket_rffi::fd_set;

#[cfg(all(unix, feature = "host_env"))]
type OsTimeval = libc::timeval;
#[cfg(all(windows, feature = "host_env"))]
type OsTimeval = majit_rlib::_rsocket_rffi::timeval;

/// `select` allocates an `fd_set` only for a non-empty list (`ll_inl` stays
/// a null pointer when `iwtd_w` is empty).
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn prepare_fd_set(fds: &[(usize, i32)]) -> Option<Box<OsFdSet>> {
    if fds.is_empty() {
        return None;
    }
    let mut set = Box::new(unsafe { core::mem::zeroed::<OsFdSet>() });
    unsafe {
        majit_rlib::_rsocket_rffi::FD_ZERO(set.as_mut() as *mut OsFdSet);
        for &(_, fd) in fds {
            majit_rlib::_rsocket_rffi::FD_SET(fd, set.as_mut() as *mut OsFdSet);
        }
    }
    Some(set)
}

#[cfg(all(any(unix, windows), feature = "host_env"))]
fn fd_set_ptr(set: &mut Option<Box<OsFdSet>>) -> *mut OsFdSet {
    match set {
        Some(set) => set.as_mut() as *mut OsFdSet,
        None => std::ptr::null_mut(),
    }
}

/// `_call_select` writes `int(timeout)` and `int((timeout - sec) * 1000000)`.
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn fill_timeval(timeout: f64, tv: &mut OsTimeval) {
    let sec = timeout as i64;
    let usec = ((timeout - sec as f64) * 1_000_000.0) as i64;
    tv.tv_sec = sec as _;
    tv.tv_usec = usec as _;
}

/// Dispose of a failed `select()`.  `Ok` means the call was interrupted and
/// the caller must retry it with the remaining timeout.
///
/// `interp_select.py` — an EINTR return delivers the pending signal first,
/// so a handler that raises (KeyboardInterrupt) wins over the retry.
#[cfg(all(unix, feature = "host_env"))]
fn select_failure(e: std::io::Error) -> Result<(), pyre_interpreter::PyError> {
    if e.raw_os_error() == Some(libc::EINTR) {
        return pyre_interpreter::module::signal::interp_signal::checksignals_now();
    }
    Err(pyre_interpreter::PyError::os_error_with_errno(
        e.raw_os_error().unwrap_or(0),
        format!("select: {e}"),
    ))
}

/// `select_failure` for WinSock, which reports through `WSAGetLastError`: the
/// code is a Win32 error kept in `.winerror`, not an errno
/// (`PyErr_SetExcFromWindowsErr`).  A blocking WinSock call is not interrupted
/// by a signal, so there is nothing to retry.
#[cfg(all(windows, feature = "host_env"))]
fn select_failure(e: std::io::Error) -> Result<(), pyre_interpreter::PyError> {
    Err(pyre_interpreter::PyError::os_error_win32_syscall2(
        e.raw_os_error().unwrap_or(0),
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
    ))
}

/// _select module — PyPy: pypy/module/select/.
///
/// Implements `select.select(rlist, wlist, xlist, timeout=None)` via
/// `_rsocket_rffi.select` / `FD_*` — on Windows over WinSock's `select`,
/// which accepts sockets only — and the `select.poll()` polling object,
/// which POSIX alone has.  epoll is not implemented yet.
pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let mut ns = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_store!(
        ns,
        "select", // Module functions are non-descriptors. `selectors.SelectSelector`
        // stores this object directly as its `_select` class attribute; a
        // descriptor-shaped builtin would bind the selector instance and
        // shift the three fd-set arguments.
        pyre_interpreter::make_module_builtin_function("select", |args| {
            #[cfg(all(any(unix, windows), feature = "host_env"))]
            {
                if args.len() < 3 {
                    return Err(pyre_interpreter::PyError::type_error(
                        "select() takes at least 3 arguments",
                    ));
                }

                // `interp_select.py:226` — `space.unpackiterable` accepts any
                // iterable (list, tuple, generator, …); each item is an int
                // fd or an object exposing fileno().
                //
                // `list_w` is a GC list upstream; the items come back here as
                // native copies, and every later `fileno()`, `__float__` and
                // the blocking `select` itself collect. Each item is pinned on
                // the caller's root bracket and named by its slot.
                fn collect_fds(
                    seq: pyre_object::PyObjectRef,
                ) -> Result<Vec<(usize, i32)>, pyre_interpreter::PyError> {
                    let items = pyre_interpreter::baseobjspace::unpackiterable(seq, -1)?;
                    let base = pyre_object::gc_roots::pin_roots(&items);
                    let mut out = Vec::with_capacity(items.len());
                    for slot in base..base + items.len() {
                        // `interp_select.py _build_fd_set` — each item is
                        // resolved through `space.c_filedescriptor_w`, then
                        // checked against what this platform's fd_set holds.
                        let fd = filedescriptor_w(pyre_object::gc_roots::shadow_stack_get(slot))?;
                        out.push((slot, selectable_fd(fd, out.len())?));
                    }
                    Ok(out)
                }

                // `args` is a native slice the gateway copied out of its own
                // slots: it keeps the arguments alive but cannot rewrite this
                // copy, and each `collect_fds` runs Python twice — the
                // iteration protocol and `fileno()`.  The three fd sequences
                // are usually lists, whose header moves, so reading the second
                // and third out of the slice after the first was collected
                // hands `unpackiterable` a pre-move address.  Pin them here and
                // read each back at its own call.  All four are published
                // before the first forwarding query.
                let arg_roots = pyre_object::gc_roots::push_roots();
                let args_base = arg_roots.pin_roots(&[
                    args[0],
                    args[1],
                    args[2],
                    args.get(3).copied().unwrap_or(pyre_object::PY_NULL),
                ]);
                let rfds = collect_fds(arg_roots.get(args_base))?;
                let wfds = collect_fds(arg_roots.get(args_base + 1))?;
                let xfds = collect_fds(arg_roots.get(args_base + 2))?;

                // The first `select()` argument: POSIX scans descriptors
                // `0..nfds`, so it must exceed the highest one.
                #[cfg(unix)]
                let nfds: i32 = {
                    let mut highest: i32 = -1;
                    for fds in [&rfds, &wfds, &xfds] {
                        for &(_, fd) in fds {
                            if fd > highest {
                                highest = fd;
                            }
                        }
                    }
                    highest + 1
                };
                // WinSock ignores it: each fd_set carries its own count.
                #[cfg(windows)]
                let nfds: i32 = 0;

                // `interp_select.py:230-235` — `None` blocks forever, else
                // `space.float_w` (applies `__float__`); a negative count is
                // a ValueError.
                let w_timeout = arg_roots.get(args_base + 3);
                let timeout_secs: Option<f64> = match (!w_timeout.is_null()).then_some(w_timeout) {
                    None => None,
                    Some(t) if unsafe { pyre_object::is_none(t) } => None,
                    Some(t) => {
                        let secs = pyre_interpreter::baseobjspace::float_w(t)?;
                        if secs < 0.0 {
                            return Err(pyre_interpreter::PyError::value_error(
                                "timeout must be non-negative",
                            ));
                        }
                        Some(secs)
                    }
                };

                // `_call_select` builds each fd_set once, then retries on
                // EINTR without rebuilding. POSIX leaves the sets unmodified
                // when `select` fails. The first pass writes the caller's
                // timeout. After EINTR only a positive timeout is replaced
                // with the time left until the deadline taken here; a zero
                // timeout stays zero. A non-finite timeout, or one `Duration`
                // cannot represent, is a ValueError: `float_w` accepts it,
                // and it is not a finite timeval.
                let mut timeout_left = timeout_secs;
                let end_time = if let Some(s) = timeout_secs.filter(|s| *s > 0.0) {
                    Some(
                        std::time::Duration::try_from_secs_f64(s)
                            .ok()
                            .and_then(|d| std::time::Instant::now().checked_add(d))
                            .ok_or_else(|| {
                                pyre_interpreter::PyError::value_error("timeout is too large")
                            })?,
                    )
                } else if timeout_secs.is_some_and(|s| !s.is_finite()) {
                    return Err(pyre_interpreter::PyError::value_error(
                        "timeout is too large",
                    ));
                } else {
                    None
                };
                let mut rset = prepare_fd_set(&rfds);
                let mut wset = prepare_fd_set(&wfds);
                let mut xset = prepare_fd_set(&xfds);
                let mut tv = OsTimeval {
                    tv_sec: 0,
                    tv_usec: 0,
                };
                let res = loop {
                    let tv_ptr = match timeout_left {
                        None => std::ptr::null_mut(),
                        Some(t) => {
                            fill_timeval(t, &mut tv);
                            &raw mut tv
                        }
                    };
                    // `_rsocket_rffi.select` releases the GIL and stashes errno.
                    let res = unsafe {
                        majit_rlib::_rsocket_rffi::select(
                            nfds,
                            fd_set_ptr(&mut rset),
                            fd_set_ptr(&mut wset),
                            fd_set_ptr(&mut xset),
                            tv_ptr,
                        )
                    };
                    if res >= 0 {
                        break res;
                    }
                    let errno = majit_rlib::_rsocket_rffi::geterrno();
                    // A retryable failure returns `Ok`, having delivered any
                    // pending signal. `_call_select` then updates `timeout`
                    // only when it was positive.
                    select_failure(std::io::Error::from_raw_os_error(errno))?;
                    if let (Some(t), Some(dl)) = (timeout_left, end_time) {
                        if t > 0.0 {
                            let now = std::time::Instant::now();
                            let remaining = if now >= dl {
                                0.0
                            } else {
                                (dl - now).as_secs_f64()
                            };
                            timeout_left = Some(remaining);
                        }
                    }
                };

                fn build_ready(
                    set: &mut Option<Box<OsFdSet>>,
                    inputs: &[(usize, i32)],
                ) -> pyre_object::PyObjectRef {
                    // `_unbuild_fd_set` runs only when `res > 0`. A timeout
                    // (`res == 0`) returns three empty lists.
                    let Some(set) = set.as_mut() else {
                        return pyre_object::w_list_new(Vec::new());
                    };
                    let items: Vec<_> = inputs
                        .iter()
                        .filter(|&&(_, fd)| unsafe {
                            majit_rlib::_rsocket_rffi::FD_ISSET(fd, set.as_mut() as *mut OsFdSet)
                                != 0
                        })
                        .map(|&(slot, _)| pyre_object::gc_roots::shadow_stack_get(slot))
                        .collect();
                    pyre_object::w_list_new(items)
                }

                // Each list is freshly minted and a list header moves, so the
                // allocation the next `build_ready` performs can relocate the
                // previous one and leave its pre-move address in the local.
                // Pin each at its mint and read all three back where the tuple
                // is built. `_unbuild_fd_set` runs only when `res > 0`.
                let roots = pyre_object::gc_roots::push_roots();
                let ready_base = roots.base();
                if res > 0 {
                    let _ = roots.pin_root(build_ready(&mut rset, &rfds));
                    let _ = roots.pin_root(build_ready(&mut wset, &wfds));
                    let _ = roots.pin_root(build_ready(&mut xset, &xfds));
                } else {
                    let _ = roots.pin_root(pyre_object::w_list_new(Vec::new()));
                    let _ = roots.pin_root(pyre_object::w_list_new(Vec::new()));
                    let _ = roots.pin_root(pyre_object::w_list_new(Vec::new()));
                }
                Ok(pyre_object::w_tuple_new(vec![
                    roots.get(ready_base),
                    roots.get(ready_base + 1),
                    roots.get(ready_base + 2),
                ]))
            }
            #[cfg(not(all(any(unix, windows), feature = "host_env")))]
            {
                let _ = args;
                Err(pyre_interpreter::PyError::not_implemented(
                    "select.select requires host_env feature on a Unix or Windows platform",
                ))
            }
        })
    );

    // `interp_select.py poll()` — factory returning a fresh polling
    // object.  The type has no public constructor, matching
    // `interp_select.py descr_new` which raises TypeError.
    #[cfg(all(unix, feature = "host_env"))]
    {
        // Force the `select.poll` type to register so instances carry a
        // valid `ob_type`.  `interp_select.py
        // Poll.typedef.acceptable_as_base_class = False`.
        let _ = type_object();
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(type_object(), false) };
        pyre_interpreter::__pyre_store!(
            ns,
            "poll", // A module-level function, not a descriptor: `selectors.py` keeps
            // it as a class attribute (`_selector_cls = select.poll`) and
            // calling it through the instance must not bind a receiver.
            pyre_interpreter::make_module_builtin_function_with_arity(
                "poll",
                |_args| Ok(Poll::allocate(Poll::default())),
                0,
            )
        );
        // `interp_select.py` exposes the rpoll event names as module
        // constants (`rpoll.eventnames`).
        macro_rules! ev {
            ($name:literal, $val:expr) => {
                pyre_interpreter::__pyre_store!(ns, $name, pyre_object::w_int_new($val as i64));
            };
        }
        ev!("POLLIN", majit_rlib::rpoll::POLLIN);
        ev!("POLLPRI", majit_rlib::rpoll::POLLPRI);
        ev!("POLLOUT", majit_rlib::rpoll::POLLOUT);
        ev!("POLLERR", majit_rlib::rpoll::POLLERR);
        ev!("POLLHUP", majit_rlib::rpoll::POLLHUP);
        ev!("POLLNVAL", majit_rlib::rpoll::POLLNVAL);
        ev!("POLLRDNORM", majit_rlib::rpoll::POLLRDNORM);
        ev!("POLLRDBAND", majit_rlib::rpoll::POLLRDBAND);
        ev!("POLLWRNORM", majit_rlib::rpoll::POLLWRNORM);
        ev!("POLLWRBAND", majit_rlib::rpoll::POLLWRBAND);
        ev!("FD_SETSIZE", majit_rlib::rpoll::FD_SETSIZE);
    }

    // `interp_kqueue.py` — kqueue() / kevent objects plus the KQ_* event
    // filter and flag constants (BSD/macOS only).
    #[cfg(all(target_os = "macos", feature = "host_env"))]
    {
        pyre_interpreter::__pyre_store!(ns, "kqueue", super::interp_kqueue::type_object());
        pyre_interpreter::__pyre_store!(ns, "kevent", super::interp_kevent::type_object());
        // `interp_kqueue.py W_Kqueue.typedef.acceptable_as_base_class
        // = False` / `:406 W_Kevent.typedef.acceptable_as_base_class =
        // False`.
        unsafe {
            pyre_object::w_type_set_acceptable_as_base_class(
                super::interp_kqueue::type_object(),
                false,
            );
            pyre_object::w_type_set_acceptable_as_base_class(
                super::interp_kevent::type_object(),
                false,
            );
        }
        macro_rules! kq {
            ($name:literal, $val:expr) => {
                pyre_interpreter::__pyre_store!(ns, $name, pyre_object::w_int_new($val as i64));
            };
        }
        // `interp_kqueue.py symbol_map` — KQ_FILTER_* / KQ_EV_*.
        kq!("KQ_FILTER_READ", libc::EVFILT_READ);
        kq!("KQ_FILTER_WRITE", libc::EVFILT_WRITE);
        kq!("KQ_FILTER_AIO", libc::EVFILT_AIO);
        kq!("KQ_FILTER_VNODE", libc::EVFILT_VNODE);
        kq!("KQ_FILTER_PROC", libc::EVFILT_PROC);
        kq!("KQ_FILTER_SIGNAL", libc::EVFILT_SIGNAL);
        kq!("KQ_FILTER_TIMER", libc::EVFILT_TIMER);
        kq!("KQ_EV_ADD", libc::EV_ADD);
        kq!("KQ_EV_DELETE", libc::EV_DELETE);
        kq!("KQ_EV_ENABLE", libc::EV_ENABLE);
        kq!("KQ_EV_DISABLE", libc::EV_DISABLE);
        kq!("KQ_EV_ONESHOT", libc::EV_ONESHOT);
        kq!("KQ_EV_CLEAR", libc::EV_CLEAR);
        kq!("KQ_EV_EOF", libc::EV_EOF);
        kq!("KQ_EV_ERROR", libc::EV_ERROR);
        // `symbol_map` stops here. It comments out `KQ_EV_SYSFLAGS` and
        // `KQ_EV_FLAG1`, and it never names the `NOTE_*` family. Those
        // extras stay: dropping them changes `dir(select)` on darwin.
        kq!("KQ_EV_SYSFLAGS", libc::EV_SYSFLAGS);
        kq!("KQ_EV_FLAG1", libc::EV_FLAG1);
        // READ / WRITE filter flag.
        kq!("KQ_NOTE_LOWAT", libc::NOTE_LOWAT);
        // VNODE filter flags.
        kq!("KQ_NOTE_DELETE", libc::NOTE_DELETE);
        kq!("KQ_NOTE_WRITE", libc::NOTE_WRITE);
        kq!("KQ_NOTE_EXTEND", libc::NOTE_EXTEND);
        kq!("KQ_NOTE_ATTRIB", libc::NOTE_ATTRIB);
        kq!("KQ_NOTE_LINK", libc::NOTE_LINK);
        kq!("KQ_NOTE_RENAME", libc::NOTE_RENAME);
        kq!("KQ_NOTE_REVOKE", libc::NOTE_REVOKE);
        // PROC filter flags. `NOTE_PCTRLMASK` is `0xfff00000`; publishing
        // it as a signed int makes it negative.
        kq!("KQ_NOTE_EXIT", libc::NOTE_EXIT);
        kq!("KQ_NOTE_FORK", libc::NOTE_FORK);
        kq!("KQ_NOTE_EXEC", libc::NOTE_EXEC);
        kq!("KQ_NOTE_PCTRLMASK", libc::NOTE_PCTRLMASK as i32);
        kq!("KQ_NOTE_PDATAMASK", libc::NOTE_PDATAMASK);
        kq!("KQ_NOTE_TRACK", libc::NOTE_TRACK);
        kq!("KQ_NOTE_CHILD", libc::NOTE_CHILD);
        kq!("KQ_NOTE_TRACKERR", libc::NOTE_TRACKERR);
    }

    // `interp_select.py:35 W_Error = OSError` — expose the real type so
    // `except select.error` catches what selectors raise.
    let mut w_os_error = pyre_interpreter::builtins::lookup_exc_class("OSError")
        .expect("OSError must be installed before select init");
    pyre_interpreter::__pyre_store!(ns, "error", w_os_error);
    #[cfg(unix)]
    {
        pyre_interpreter::__pyre_store!(
            ns,
            "PIPE_BUF",
            pyre_object::w_int_new(libc::PIPE_BUF as i64)
        );
    }
    Ok(())
}
