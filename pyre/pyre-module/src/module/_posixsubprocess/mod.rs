//! _posixsubprocess module — PyPy: `pypy/module/_posixsubprocess/`.
//!
//! Backs `subprocess` on POSIX through `fork_exec`.  The whole surface is
//! gated on `cfg(all(unix, feature = "host_env"))`; non-Unix /
//! `host_env = off` builds expose an empty module so `import
//! _posixsubprocess` still succeeds (matching PyPy's mixedmodule
//! behaviour when the conditional `interpleveldefs` entry is absent).

use pyre_object::*;

#[cfg(all(unix, feature = "host_env"))]
mod imp {
    use super::*;
    use core::{convert::Infallible, ffi::CStr, marker::PhantomData};
    use pyre_interpreter::PyError;
    use std::ffi::CString;
    use std::os::fd::{AsFd, AsRawFd, BorrowedFd};

    /// Null-terminated `*const c_char` array, kept alive by the borrowed
    /// `CString`s it points into.  `argv`/`envp` for `exec*`.
    #[derive(Default)]
    struct CharPtrVec<'a> {
        vec: Vec<*const libc::c_char>,
        marker: PhantomData<Vec<&'a CStr>>,
    }

    impl<'a, T: AsRef<CStr>> FromIterator<&'a T> for CharPtrVec<'a> {
        fn from_iter<I: IntoIterator<Item = &'a T>>(iter: I) -> Self {
            let vec = iter
                .into_iter()
                .map(|x| x.as_ref().as_ptr())
                .chain(core::iter::once(core::ptr::null()))
                .collect();
            Self {
                vec,
                marker: PhantomData,
            }
        }
    }

    impl CharPtrVec<'_> {
        fn as_ptr(&self) -> *const *const libc::c_char {
            self.vec.as_ptr()
        }
    }

    fn io_err(e: std::io::Error) -> PyError {
        PyError::os_error_with_errno(e.raw_os_error().unwrap_or(0), e.to_string())
    }

    fn last_err() -> std::io::Error {
        std::io::Error::last_os_error()
    }

    fn set_inheritable(fd: BorrowedFd<'_>, inheritable: bool) -> std::io::Result<()> {
        let current = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFD) };
        if current < 0 {
            return Err(last_err());
        }
        let new = if inheritable {
            current & !libc::FD_CLOEXEC
        } else {
            current | libc::FD_CLOEXEC
        };
        if new != current && unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, new) } < 0 {
            return Err(last_err());
        }
        Ok(())
    }

    fn close_raw(fd: i32) -> std::io::Result<()> {
        if unsafe { libc::close(fd) } < 0 {
            Err(last_err())
        } else {
            Ok(())
        }
    }

    fn dup_raw(fd: i32) -> std::io::Result<i32> {
        let n = unsafe { libc::dup(fd) };
        if n < 0 { Err(last_err()) } else { Ok(n) }
    }

    fn dup2_raw(fd: i32, newfd: i32) -> std::io::Result<()> {
        if unsafe { libc::dup2(fd, newfd) } < 0 {
            Err(last_err())
        } else {
            Ok(())
        }
    }

    fn dup_into_stdio(fd: i32, io_fd: i32) -> std::io::Result<()> {
        if fd < 0 {
            return Ok(());
        }
        if fd == io_fd {
            set_inheritable(unsafe { BorrowedFd::borrow_raw(fd) }, true)
        } else {
            dup2_raw(fd, io_fd)
        }
    }

    fn setup_child_fds(
        fds_to_keep: &[BorrowedFd<'_>],
        errpipe_write: BorrowedFd<'_>,
        p2cread: i32,
        p2cwrite: i32,
        c2pread: i32,
        c2pwrite: i32,
        errread: i32,
        errwrite: i32,
        errpipe_read: i32,
    ) -> std::io::Result<()> {
        for &fd in fds_to_keep {
            if fd.as_raw_fd() != errpipe_write.as_raw_fd() {
                set_inheritable(fd, true)?;
            }
        }
        for fd in [p2cwrite, c2pread, errread] {
            if fd >= 0 {
                close_raw(fd)?;
            }
        }
        close_raw(errpipe_read)?;
        let c2pwrite = if c2pwrite == 0 {
            let dup = dup_raw(c2pwrite)?;
            set_inheritable(unsafe { BorrowedFd::borrow_raw(dup) }, true)?;
            dup
        } else {
            c2pwrite
        };
        let mut errwrite = errwrite;
        while errwrite == 0 || errwrite == 1 {
            let dup = dup_raw(errwrite)?;
            set_inheritable(unsafe { BorrowedFd::borrow_raw(dup) }, true)?;
            errwrite = dup;
        }
        dup_into_stdio(p2cread, 0)?;
        dup_into_stdio(c2pwrite, 1)?;
        dup_into_stdio(errwrite, 2)?;
        Ok(())
    }

    fn should_keep(above: i32, keep: &[BorrowedFd<'_>], fd: i32) -> bool {
        fd > above
            && keep
                .binary_search_by_key(&fd, BorrowedFd::as_raw_fd)
                .is_err()
    }

    fn close_dir_fds(above: i32, keep: &[BorrowedFd<'_>]) -> std::io::Result<()> {
        #[cfg(any(
            target_os = "dragonfly",
            target_os = "freebsd",
            target_os = "netbsd",
            target_os = "openbsd",
            target_vendor = "apple",
            target_os = "linux",
            target_os = "android",
        ))]
        {
            #[cfg(any(
                target_os = "dragonfly",
                target_os = "freebsd",
                target_os = "netbsd",
                target_os = "openbsd",
                target_vendor = "apple",
            ))]
            let fd_dir_name = c"/dev/fd";
            #[cfg(any(target_os = "linux", target_os = "android"))]
            let fd_dir_name = c"/proc/self/fd";
            let dir = unsafe { libc::opendir(fd_dir_name.as_ptr()) };
            if dir.is_null() {
                return Err(last_err());
            }
            let dirfd = unsafe { libc::dirfd(dir) };
            loop {
                majit_rlib::rposix::_set_errno(0);
                let entry = unsafe { libc::readdir(dir) };
                if entry.is_null() {
                    break;
                }
                let name = unsafe { std::ffi::CStr::from_ptr((*entry).d_name.as_ptr()) };
                let Some(fd) = name.to_bytes().iter().try_fold(0i32, |n, &c| {
                    let digit = (c as char).to_digit(10)?;
                    n.checked_mul(10)?.checked_add(digit as i32)
                }) else {
                    continue;
                };
                if fd != dirfd && should_keep(above, keep, fd) {
                    let _ = close_raw(fd);
                }
            }
            let _ = unsafe { libc::closedir(dir) };
            Ok(())
        }
        #[cfg(not(any(
            target_os = "dragonfly",
            target_os = "freebsd",
            target_os = "netbsd",
            target_os = "openbsd",
            target_vendor = "apple",
            target_os = "linux",
            target_os = "android",
        )))]
        {
            let _ = (above, keep);
            Err(std::io::Error::from_raw_os_error(libc::ENOSYS))
        }
    }

    fn close_fds_brute_force(above: i32, keep: &[BorrowedFd<'_>]) {
        let max_fd = unsafe { libc::sysconf(libc::_SC_OPEN_MAX) };
        let max_fd = if max_fd > 0 { max_fd as i32 } else { 256 };
        let mut prev = above;
        for fd in keep
            .iter()
            .map(BorrowedFd::as_raw_fd)
            .chain(core::iter::once(max_fd))
        {
            for candidate in prev + 1..fd {
                let _ = unsafe { libc::close(candidate) };
            }
            prev = fd;
        }
    }

    fn close_fds(above: i32, keep: &[BorrowedFd<'_>]) {
        if close_dir_fds(above, keep).is_ok() {
            return;
        }
        close_fds_brute_force(above, keep);
    }

    fn restore_signals() {
        unsafe {
            libc::signal(libc::SIGPIPE, libc::SIG_DFL);
            libc::signal(libc::SIGXFSZ, libc::SIG_DFL);
        }
    }

    fn exec_replace(
        exec_list: &[CString],
        argv: *const *const libc::c_char,
        envp: Option<*const *const libc::c_char>,
    ) -> i32 {
        let mut first_err = None;
        for exec in exec_list {
            if let Some(envp) = envp {
                unsafe { libc::execve(exec.as_ptr(), argv, envp) };
            } else {
                unsafe { libc::execv(exec.as_ptr(), argv) };
            }
            let e = last_err().raw_os_error().unwrap_or(0);
            if e != libc::ENOENT && e != libc::ENOTDIR && first_err.is_none() {
                first_err = Some(e);
            }
        }
        first_err.unwrap_or_else(|| last_err().raw_os_error().unwrap_or(0))
    }

    fn is_none_obj(o: PyObjectRef) -> bool {
        unsafe { is_none(o) }
    }

    fn fd_arg(o: PyObjectRef) -> i32 {
        (unsafe { w_int_get_value(o) }) as i32
    }

    fn seq_len(o: PyObjectRef, what: &str) -> Result<usize, PyError> {
        unsafe {
            if is_list(o) {
                Ok(w_list_len(o))
            } else if is_tuple(o) {
                Ok(w_tuple_len(o))
            } else {
                Err(PyError::type_error(format!(
                    "fork_exec(): {what} must be a list or tuple"
                )))
            }
        }
    }

    fn seq_getitem(o: PyObjectRef, index: usize) -> Option<PyObjectRef> {
        unsafe {
            if is_list(o) {
                w_list_getitem(o, index as i64)
            } else if is_tuple(o) {
                w_tuple_getitem(o, index as i64)
            } else {
                None
            }
        }
    }

    fn seq_items(o: PyObjectRef, what: &str) -> Result<Vec<PyObjectRef>, PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let seq_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(o);
        let n = seq_len(pyre_object::gc_roots::shadow_stack_get(seq_slot), what)?;
        let item_base = pyre_object::gc_roots::shadow_stack_len();
        let mut count = 0;
        for i in 0..n {
            let item = match seq_getitem(pyre_object::gc_roots::shadow_stack_get(seq_slot), i) {
                Some(item) => item,
                None => continue,
            };
            let _ = pyre_object::gc_roots::pin_root(item);
            count += 1;
        }
        Ok((0..count)
            .map(|i| pyre_object::gc_roots::shadow_stack_get(item_base + i))
            .collect())
    }

    fn obj_to_cstring(o: PyObjectRef, what: &str) -> Result<CString, PyError> {
        let bytes = unsafe {
            if is_str(o) {
                // The exec boundary carries OS bytes: a program name or argv
                // entry that came back from the filesystem holds surrogate
                // escapes, which fold back to the original bytes here rather
                // than having no `&str` spelling at all.
                pyre_interpreter::gateway::fsencode(o)?
            } else if is_bytes(o) {
                w_bytes_data(o).to_vec()
            } else {
                return Err(PyError::type_error(format!(
                    "fork_exec(): {what} must be str or bytes"
                )));
            }
        };
        CString::new(bytes)
            .map_err(|_| PyError::value_error(format!("fork_exec(): embedded null in {what}")))
    }

    fn collect_cstrings(o: PyObjectRef, what: &str) -> Result<Vec<CString>, PyError> {
        // `obj_to_cstring` / `fsencode` allocate, so the sequence and the
        // current item are re-read from the shadow stack rather than held
        // as a `Vec` of raw pointers across that call.
        let _roots = pyre_object::gc_roots::push_roots();
        let seq_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(o);
        let n = seq_len(pyre_object::gc_roots::shadow_stack_get(seq_slot), what)?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let item = match seq_getitem(pyre_object::gc_roots::shadow_stack_get(seq_slot), i) {
                Some(item) => item,
                None => continue,
            };
            let item_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(item);
            out.push(obj_to_cstring(
                pyre_object::gc_roots::shadow_stack_get(item_slot),
                what,
            )?);
        }
        Ok(out)
    }

    /// `interp_subprocess.py:185-187`:
    ///
    /// ```python
    /// argv = [space.fsencode_w(space.next(w_iter))
    ///         for i in range(space.len_w(w_process_args))]
    /// ```
    ///
    /// Process arguments, unlike the already-fsencoded executable/env arrays,
    /// accept `os.PathLike` entries.
    fn collect_fsencoded_cstrings(o: PyObjectRef, what: &str) -> Result<Vec<CString>, PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let seq_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(o);
        let n = seq_len(pyre_object::gc_roots::shadow_stack_get(seq_slot), what)?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let item = match seq_getitem(pyre_object::gc_roots::shadow_stack_get(seq_slot), i) {
                Some(item) => item,
                None => continue,
            };
            let item_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(item);
            let bytes = pyre_interpreter::gateway::fsencode_bytes_w(
                pyre_object::gc_roots::shadow_stack_get(item_slot),
            )?;
            out.push(CString::new(bytes).map_err(|_| {
                PyError::value_error(format!("fork_exec(): embedded null in {what}"))
            })?);
        }
        Ok(out)
    }

    fn opt_fsencoded_cstring(o: PyObjectRef, what: &str) -> Result<Option<CString>, PyError> {
        if is_none_obj(o) {
            return Ok(None);
        }
        let bytes = pyre_interpreter::gateway::fsencode_bytes_w(o)?;
        CString::new(bytes)
            .map(Some)
            .map_err(|_| PyError::value_error(format!("fork_exec(): embedded null in {what}")))
    }

    fn collect_fds(o: PyObjectRef) -> Result<Vec<BorrowedFd<'static>>, PyError> {
        Ok(seq_items(o, "fds_to_keep")?
            .into_iter()
            .map(|x| unsafe { BorrowedFd::borrow_raw((w_int_get_value(x)) as i32) })
            .collect())
    }

    /// `_Py_Gid_Converter`/`_Py_Uid_Converter`: accept `id >= -1`, with
    /// `-1` mapping to `u32::MAX` (an unset sentinel the `set*id_if_needed`
    /// helpers skip).
    fn try_from_id(o: PyObjectRef, name: &str) -> Result<u32, PyError> {
        use core::cmp::Ordering;
        let i = unsafe { w_int_get_value(o) };
        match i.cmp(&-1) {
            Ordering::Greater => u32::try_from(i)
                .map_err(|_| PyError::overflow_error(format!("{name} is larger than maximum"))),
            Ordering::Less => Err(PyError::overflow_error(format!(
                "{name} is less than minimum"
            ))),
            Ordering::Equal => Ok(-1i32 as u32),
        }
    }

    fn opt_id(o: PyObjectRef, name: &str) -> Result<Option<u32>, PyError> {
        if is_none_obj(o) {
            Ok(None)
        } else {
            Ok(Some(try_from_id(o, name)?))
        }
    }

    /// `interp_subprocess.fork_exec`:
    ///
    /// ```python
    /// groups_w = space.unpackiterable(w_groups_list)
    /// for i, w_group in enumerate(groups_w):
    ///     gid_val = space.int_w(w_group)
    /// ```
    ///
    /// The sequence stays rooted and each item is fetched right before its
    /// `int_w`, which can run `__index__` and collect.
    fn collect_gids(o: PyObjectRef) -> Result<Vec<u32>, PyError> {
        let _roots = pyre_object::gc_roots::push_roots();
        let seq_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(o);
        let n = seq_len(pyre_object::gc_roots::shadow_stack_get(seq_slot), "gids")?;
        // `interp_subprocess.fork_exec` rejects the sequence before allocating
        // its raw gid_t array.  POSIX permits sysconf to be indeterminate;
        // PyPy's configure-time fallback for that case is 64.
        let configured_max = unsafe { libc::sysconf(libc::_SC_NGROUPS_MAX) };
        let max_groups = if configured_max < 0 {
            64
        } else {
            configured_max as usize
        };
        if n > max_groups {
            return Err(PyError::value_error("too many groups"));
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let Some(x) = seq_getitem(pyre_object::gc_roots::shadow_stack_get(seq_slot), i) else {
                continue;
            };
            // PyPy `fork_exec` converts supplementary groups separately
            // from the uid/gid fields: `-1` is not an unset sentinel here.
            // CPython `_Py_Gid_Converter` likewise exposes negative and
            // over-gid_t entries as ValueError for `extra_groups`.
            let value = match pyre_interpreter::baseobjspace::int_w(x) {
                Ok(value) => value,
                Err(error) if error.kind == pyre_interpreter::PyErrorKind::OverflowError => {
                    return Err(PyError::value_error("group id is greater than maximum"));
                }
                Err(error) => return Err(error),
            };
            if value < 0 {
                return Err(PyError::value_error("group id is negative"));
            }
            out.push(
                u32::try_from(value)
                    .map_err(|_| PyError::value_error("group id is greater than maximum"))?,
            );
        }
        Ok(out)
    }

    /// Decoded `fork_exec` arguments, allocated before `fork()` so the
    /// child does no further allocation before `exec`.
    struct Decoded<'a> {
        exec_list: &'a [CString],
        argv: *const *const libc::c_char,
        envp: Option<*const *const libc::c_char>,
        fds_to_keep: &'a [BorrowedFd<'static>],
        extra_groups: Option<&'a [u32]>,
        cwd: Option<&'a CString>,
        preexec_fn: Option<PyObjectRef>,
        close_fds: bool,
        restore_signals: bool,
        call_setsid: bool,
        pgid_to_set: libc::pid_t,
        gid: Option<u32>,
        uid: Option<u32>,
        child_umask: i32,
        p2cread: i32,
        p2cwrite: i32,
        c2pread: i32,
        c2pwrite: i32,
        errread: i32,
        errwrite: i32,
        errpipe_read: i32,
        errpipe_write: i32,
    }

    enum ExecErrorContext {
        NoExec,
        ChDir,
        PreExec,
        Exec,
    }

    impl ExecErrorContext {
        const fn as_msg(&self) -> &'static str {
            match self {
                Self::NoExec => "noexec",
                Self::ChDir => "noexec:chdir",
                Self::PreExec => "Exception occurred in preexec_fn.",
                Self::Exec => "",
            }
        }
    }

    fn exec_inner(d: &Decoded<'_>, ctx: &mut ExecErrorContext) -> std::io::Result<Infallible> {
        let errpipe_write = unsafe { BorrowedFd::borrow_raw(d.errpipe_write) };
        setup_child_fds(
            d.fds_to_keep,
            errpipe_write.as_fd(),
            d.p2cread,
            d.p2cwrite,
            d.c2pread,
            d.c2pwrite,
            d.errread,
            d.errwrite,
            d.errpipe_read,
        )?;

        if let Some(cwd) = d.cwd {
            if unsafe { libc::chdir(cwd.as_ptr()) } < 0 {
                *ctx = ExecErrorContext::ChDir;
                return Err(last_err());
            }
        }

        if d.child_umask >= 0 {
            unsafe { libc::umask(d.child_umask as libc::mode_t) };
        }

        if d.restore_signals {
            restore_signals();
        }

        if d.call_setsid && unsafe { libc::setsid() } < 0 {
            return Err(last_err());
        }
        if d.pgid_to_set > -1 && unsafe { libc::setpgid(0, d.pgid_to_set) } < 0 {
            return Err(last_err());
        }
        #[cfg(not(any(target_os = "ios", target_os = "redox")))]
        if let Some(groups) = d.extra_groups {
            let ret = unsafe {
                libc::setgroups(groups.len() as _, groups.as_ptr().cast::<libc::gid_t>())
            };
            if ret < 0 {
                return Err(last_err());
            }
        }
        if let Some(gid) = d.gid.filter(|&x| x != u32::MAX)
            && unsafe { libc::setregid(gid as libc::gid_t, gid as libc::gid_t) } < 0
        {
            return Err(last_err());
        }
        if let Some(uid) = d.uid.filter(|&x| x != u32::MAX)
            && unsafe { libc::setreuid(uid as libc::uid_t, uid as libc::uid_t) } < 0
        {
            return Err(last_err());
        }

        // Call preexec_fn after all process setup but before closing FDs.
        if let Some(preexec_fn) = d.preexec_fn {
            let r = pyre_interpreter::baseobjspace::call_function(preexec_fn, &[]);
            if r.is_null() {
                // Cannot safely stringify the exception after fork.
                let _ = pyre_interpreter::call::take_call_error();
                *ctx = ExecErrorContext::PreExec;
                return Err(std::io::Error::from_raw_os_error(0));
            }
        }

        *ctx = ExecErrorContext::Exec;

        if d.close_fds {
            close_fds(2, d.fds_to_keep);
        }

        let err = exec_replace(d.exec_list, d.argv, d.envp);
        Err(std::io::Error::from_raw_os_error(err))
    }

    fn exec(d: &Decoded<'_>) -> ! {
        let mut ctx = ExecErrorContext::NoExec;
        match exec_inner(d, &mut ctx) {
            Ok(infallible) => match infallible {},
            Err(e) => {
                let errpipe = unsafe { BorrowedFd::borrow_raw(d.errpipe_write) };
                let msg = if matches!(ctx, ExecErrorContext::PreExec) {
                    // preexec_fn failures use SubprocessError format (errno=0).
                    format!("SubprocessError:0:{}", ctx.as_msg())
                } else {
                    // errno is written in hex.
                    let errno = e.raw_os_error().unwrap_or(0);
                    format!("OSError:{errno:x}:{}", ctx.as_msg())
                };
                let _ = unsafe { libc::write(errpipe.as_raw_fd(), msg.as_ptr().cast(), msg.len()) };
                unsafe { libc::_exit(255) }
            }
        }
    }

    pub fn fork_exec(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        let (pos, _kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
        if pos.len() != 22 {
            return Err(PyError::type_error(format!(
                "fork_exec() takes exactly 22 arguments ({} given)",
                pos.len()
            )));
        }

        // [3.14-spec] PyPy `interp_subprocess.fork_exec` permits preexec_fn
        // during shutdown, but CPython 3.14 `_posixsubprocess.fork_exec`
        // rejects this observable unsafe call.  Keep PyPy's fork/exec shape
        // and add only the finalization gate required by that public contract.
        if !is_none_obj(pos[21]) && pyre_interpreter::module::thread::is_finalizing() {
            return Err(pyre_interpreter::builtins::finalization_error(Some(
                "preexec_fn not supported at interpreter shutdown",
            )));
        }

        // `pos` is the gateway's native copy; the conversions below run
        // Python and collect, so every argument is read from its slot.
        let _roots = pyre_object::gc_roots::push_roots();
        let pos_base = _roots.pin_roots(pos);
        // Decode everything (and pre-allocate the argv/envp arrays) before
        // fork(): the child must not allocate before exec.
        let args_list = collect_fsencoded_cstrings(_roots.get(pos_base), "args")?;
        let exec_list = collect_cstrings(_roots.get(pos_base + 1), "executable_list")?;
        let close_fds = pyre_interpreter::baseobjspace::is_true(_roots.get(pos_base + 2))?;
        let fds_to_keep = collect_fds(_roots.get(pos_base + 3))?;
        let cwd = opt_fsencoded_cstring(_roots.get(pos_base + 4), "cwd")?;
        let env_list = if is_none_obj(_roots.get(pos_base + 5)) {
            None
        } else {
            Some(collect_cstrings(_roots.get(pos_base + 5), "env_list")?)
        };
        let p2cread = fd_arg(_roots.get(pos_base + 6));
        let p2cwrite = fd_arg(_roots.get(pos_base + 7));
        let c2pread = fd_arg(_roots.get(pos_base + 8));
        let c2pwrite = fd_arg(_roots.get(pos_base + 9));
        let errread = fd_arg(_roots.get(pos_base + 10));
        let errwrite = fd_arg(_roots.get(pos_base + 11));
        let errpipe_read = fd_arg(_roots.get(pos_base + 12));
        let errpipe_write = fd_arg(_roots.get(pos_base + 13));
        let restore_signals = pyre_interpreter::baseobjspace::is_true(_roots.get(pos_base + 14))?;
        let call_setsid = pyre_interpreter::baseobjspace::is_true(_roots.get(pos_base + 15))?;
        let pgid_to_set = (unsafe { w_int_get_value(_roots.get(pos_base + 16)) }) as libc::pid_t;
        let gid = opt_id(_roots.get(pos_base + 17), "gid")?;
        let extra_groups = if is_none_obj(_roots.get(pos_base + 18)) {
            None
        } else {
            Some(collect_gids(_roots.get(pos_base + 18))?)
        };
        let uid = opt_id(_roots.get(pos_base + 19), "uid")?;
        let child_umask = (unsafe { w_int_get_value(_roots.get(pos_base + 20)) }) as i32;
        let preexec_fn = if is_none_obj(_roots.get(pos_base + 21)) {
            None
        } else {
            Some(_roots.get(pos_base + 21))
        };

        let argv = args_list.iter().collect::<CharPtrVec<'_>>();
        let envp = env_list
            .as_ref()
            .map(|e| e.iter().collect::<CharPtrVec<'_>>());

        let decoded = Decoded {
            exec_list: &exec_list,
            argv: argv.as_ptr(),
            envp: envp.as_ref().map(CharPtrVec::as_ptr),
            fds_to_keep: &fds_to_keep,
            extra_groups: extra_groups.as_deref(),
            cwd: cwd.as_ref(),
            preexec_fn,
            close_fds,
            restore_signals,
            call_setsid,
            pgid_to_set,
            gid,
            uid,
            child_umask,
            p2cread,
            p2cwrite,
            c2pread,
            c2pwrite,
            errread,
            errwrite,
            errpipe_read,
            errpipe_write,
        };

        // `rposix.c_fork` is `_nowrapper`, so the live errno is the failure.
        let pid = unsafe { majit_rlib::rposix::c_fork() };
        match pid {
            0 => exec(&decoded),
            pid if pid > 0 => Ok(w_int_new(pid as i64)),
            _ => Err(io_err(last_err())),
        }
    }
}

pyre_interpreter::py_module! {
    "_posixsubprocess",
    extra_init: |ns| {
        #[cfg(all(unix, feature = "host_env"))]
        pyre_interpreter::__pyre_store!(ns, "fork_exec", pyre_interpreter::make_builtin_function("fork_exec", imp::fork_exec));
        #[cfg(not(all(unix, feature = "host_env")))]
        let _ = ns;
    }
}
