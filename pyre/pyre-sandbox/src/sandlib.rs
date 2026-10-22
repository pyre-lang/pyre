//! Controller-side policy + message loop — port of
//! `rpython/translator/sandbox/sandlib.py`.
//!
//! `SandboxPolicy` merges the two upstream mixins that the only concrete
//! controller (`PyPySandboxedProc`) inherits from: `VirtualizedSandboxedProc`
//! (a virtual filesystem + virtual env over `vfs.py`) and
//! `SimpleIOSandboxedProc` (stdin/stdout/stderr pass-through + real time). The
//! `do_ll_os__*` / `do_ll_time__*` handlers are dispatched by the function-name
//! string the sandboxed client marshals.

use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::vfs::{self, FsNode, GID, ReadSeek, UID};
use majit_rlib::rmarshal::{
    IntFlavor, Loader, MarshalValue, ReadNeedMore, StatResult, dump_int, dump_longlong_result,
    dump_statresult, dump_value, load_string, load_value,
};
use majit_rlib::rsandbox::{SandboxError, SandboxResult, code_for_error};

// sandlib.py `virtual_fd_range = range(3, 50)`.
const FD_RANGE_START: i32 = 3;
const FD_RANGE_END: i32 = 50;

// don't read more than 256KB from a virtual file at once (sandlib.py:506).
const MAX_READ: usize = 256 * 1024;

/// The static shape of a successful reply, mirroring `sandlib.write_message`'s
/// `resulttype` parameter (`sandlib.py`).
pub enum Reply {
    /// Marshalled with the normal `_marshal` codec.
    Value(MarshalValue),
    /// `RESULTTYPE_STATRESULT` — hand-packed `os.stat_result`.
    Stat(StatResult),
    /// `RESULTTYPE_LONGLONG` — forced 64-bit int (`ll_os_lseek`).
    LongLong(i64),
}

/// The controller's console streams (the SimpleIO side). For `pyre interact`
/// these are the real process stdin/stdout/stderr; tests substitute buffers.
pub struct Console<'a> {
    pub input: &'a mut dyn Read,
    pub output: &'a mut dyn Write,
    pub error: &'a mut dyn Write,
    pub input_isatty: bool,
}

struct OpenFd {
    handle: Box<dyn ReadSeek>,
    node: FsNode,
}

/// Shared timeout state between the controller's watchdog and the request
/// handlers, restoring sandlib.py's idle/poll behaviour across pyre's
/// policy/controller split. Defaults to "not idle, not cancelled" so a policy
/// run without a watchdog (no `--timeout`) behaves unchanged.
#[derive(Clone, Default)]
pub struct TimeoutControl {
    /// True while the controller is blocked on interactive console input. The
    /// watchdog keeps the activity clock fresh while it is set, so time spent
    /// waiting at the prompt is not charged against `--timeout` (sandlib.py
    /// enter_idle/leave_idle).
    pub idle: Arc<AtomicBool>,
    /// Set when the watchdog SIGKILLs the child; a long sleep being serviced
    /// polls this between chunks and returns early instead of parking the
    /// controller for the full requested duration (sandlib.py
    /// do_ll_time__ll_time_sleep's self.poll()).
    pub cancelled: Arc<AtomicBool>,
}

/// The virtualized sandbox policy. Owns the virtual filesystem, the virtual
/// environment, and the open-fd table.
pub struct SandboxPolicy {
    pub virtual_root: FsNode,
    pub virtual_cwd: String,
    pub virtual_env: Vec<(Vec<u8>, Vec<u8>)>,
    pub virtual_console_isatty: bool,
    open_fds: HashMap<i32, OpenFd>,
    /// Opt-in `tcp://host:port` mediation (`VirtualizedSocketProc`, sandlib.py).
    /// Off by default so the standard policy stays network-closed; the controller
    /// flips it on with `set_allow_net` when `--allow-net` is passed.
    allow_net: bool,
    /// Fds returned by a `tcp://` open, sharing the `open_fds` fd space
    /// (`VirtualizedSocketProc.sockets`, sandlib.py). read/write route these
    /// to the connected stream instead of a virtual file.
    sockets: HashMap<i32, TcpStream>,
    /// Append-mode log of the guest's stdin (`inputlogfile`, sandlib.py:294),
    /// enabled by `--log FILE`. Each fd-0 read appends the bytes handed to the
    /// child. `None` unless a log file was opened.
    input_log: Option<std::fs::File>,
    clock_start: Option<Instant>,
    timeout_control: TimeoutControl,
}

impl SandboxPolicy {
    pub fn new(
        virtual_root: FsNode,
        virtual_cwd: impl Into<String>,
        virtual_env: Vec<(Vec<u8>, Vec<u8>)>,
        virtual_console_isatty: bool,
    ) -> Self {
        SandboxPolicy {
            virtual_root,
            virtual_cwd: virtual_cwd.into(),
            virtual_env,
            virtual_console_isatty,
            open_fds: HashMap::new(),
            allow_net: false,
            sockets: HashMap::new(),
            input_log: None,
            clock_start: None,
            timeout_control: TimeoutControl::default(),
        }
    }

    /// Share the controller's timeout state so `do_read`/`do_sleep` can honour
    /// the idle and cancellation signals. Called once before the request loop.
    pub fn set_timeout_control(&mut self, control: TimeoutControl) {
        self.timeout_control = control;
    }

    /// Enable `tcp://host:port` mediation (`VirtualizedSocketProc`). Off by
    /// default; the controller calls this when the operator opts in with
    /// `--allow-net`.
    pub fn set_allow_net(&mut self, allow: bool) {
        self.allow_net = allow;
    }

    /// Log the guest's stdin to `file` (`setlogfile`, sandlib.py). The
    /// controller opens the file in append mode before the request loop.
    pub fn set_input_log(&mut self, file: std::fs::File) {
        self.input_log = Some(file);
    }

    // ── path resolution (sandlib.py translate_path) ─────────────────────────────────

    // sandlib.py `translate_path`.
    fn translate_path(&self, vpath: &str) -> SandboxResult<(FsNode, String)> {
        let joined = posix_join(&self.virtual_cwd, vpath);
        let norm = posix_normpath(&joined);
        let components: Vec<&str> = norm.split('/').collect();
        let (last, dirs) = components
            .split_last()
            .expect("split always yields >= 1 component");
        let mut dirnode = self.virtual_root.clone();
        for component in dirs {
            if !component.is_empty() {
                dirnode = dirnode.join(component).map_err(vfs_err)?;
                if !vfs::is_dir(dirnode.kind()) {
                    return Err(SandboxError::Os(libc::ENOTDIR));
                }
            }
        }
        Ok((dirnode, (*last).to_owned()))
    }

    // sandlib.py `get_node`.
    fn get_node(&self, vpath: &str) -> SandboxResult<FsNode> {
        let (dirnode, name) = self.translate_path(vpath)?;
        if name.is_empty() {
            Ok(dirnode)
        } else {
            dirnode.join(&name).map_err(vfs_err)
        }
    }

    // sandlib.py `allocate_fd`. Files and `tcp://` sockets share one fd
    // space, so both tables are consulted for a free slot.
    fn allocate_fd(&mut self, handle: Box<dyn ReadSeek>, node: FsNode) -> SandboxResult<i32> {
        let fd = self.next_free_fd()?;
        self.open_fds.insert(fd, OpenFd { handle, node });
        Ok(fd)
    }

    // Socket variant of `allocate_fd` (`VirtualizedSocketProc`, sandlib.py).
    fn allocate_socket_fd(&mut self, stream: TcpStream) -> SandboxResult<i32> {
        let fd = self.next_free_fd()?;
        self.sockets.insert(fd, stream);
        Ok(fd)
    }

    fn next_free_fd(&self) -> SandboxResult<i32> {
        for fd in FD_RANGE_START..FD_RANGE_END {
            if !self.open_fds.contains_key(&fd) && !self.sockets.contains_key(&fd) {
                return Ok(fd);
            }
        }
        Err(SandboxError::Os(libc::EMFILE))
    }

    // ── dispatch (sandlib.py handle_message) ────────────────────────────────────────

    /// Dispatch a marshalled request to the matching `do_*` handler. Rejects any
    /// fnname containing `"__"` first (`sandlib.py:277`), so only the curated
    /// `ll_os.*` / `ll_time.*` names below are reachable.
    pub fn handle_message(
        &mut self,
        fnname: &str,
        args: &MarshalValue,
        console: &mut Console,
    ) -> SandboxResult<Reply> {
        if fnname.contains("__") {
            return Err(SandboxError::Value);
        }
        let args = match args {
            MarshalValue::Tuple(items) => items.as_slice(),
            _ => return Err(SandboxError::Value),
        };
        match fnname {
            "ll_os.ll_os_open" => self.do_open(args),
            "ll_os.ll_os_close" => self.do_close(args),
            "ll_os.ll_os_read" => self.do_read(args, console),
            "ll_os.ll_os_write" => self.do_write(args, console),
            "ll_os.ll_os_stat" | "ll_os.ll_os_lstat" => self.do_stat(args),
            "ll_os.ll_os_fstat" => self.do_fstat(args),
            "ll_os.ll_os_lseek" => self.do_lseek(args),
            "ll_os.ll_os_access" => self.do_access(args),
            "ll_os.ll_os_isatty" => self.do_isatty(args),
            "ll_os.ll_os_getcwd" => self.do_getcwd(),
            "ll_os.ll_os_strerror" => self.do_strerror(args),
            "ll_os.ll_os_listdir" => self.do_listdir(args),
            "ll_os.ll_os_getenv" => self.do_getenv(args),
            "ll_os.ll_os_envitems" => self.do_envitems(),
            "ll_os.ll_os_unlink" | "ll_os.ll_os_mkdir" => Err(SandboxError::Os(libc::EPERM)),
            "ll_os.ll_os_urandom" => self.do_urandom(args),
            "ll_os.ll_os_getuid" | "ll_os.ll_os_geteuid" => {
                Ok(Reply::Value(MarshalValue::Int(UID as i64)))
            }
            "ll_os.ll_os_getgid" | "ll_os.ll_os_getegid" => {
                Ok(Reply::Value(MarshalValue::Int(GID as i64)))
            }
            "ll_time.ll_time_time" => self.do_time(),
            "ll_time.ll_time_clock" => self.do_clock(),
            "ll_time.ll_time_sleep" => self.do_sleep(args),
            _ => Err(SandboxError::Runtime),
        }
    }

    // ── VirtualizedSandboxedProc handlers ────────────────────────────────────

    // sandlib.py `do_ll_os__ll_os_open`, with the `VirtualizedSocketProc`
    // `tcp://` override (sandlib.py) folded in when `allow_net` is set.
    fn do_open(&mut self, args: &[MarshalValue]) -> SandboxResult<Reply> {
        let vpath = arg_path(args, 0)?;
        // sandlib.py:555: sockets are checked before the read-only flag gate,
        // since a connected stream is inherently read-write.
        if self.allow_net
            && let Some(target) = vpath.strip_prefix("tcp://")
        {
            return self.do_open_socket(target);
        }
        let flags = arg_int(args, 1)? as i32;
        let node = self.get_node(&vpath)?;
        if flags & libc::O_ACCMODE != libc::O_RDONLY {
            return Err(SandboxError::Os(libc::EPERM)); // "write access denied"
        }
        let handle = node.open().map_err(vfs_err)?;
        let fd = self.allocate_fd(handle, node)?;
        Ok(Reply::Value(MarshalValue::Int(fd as i64)))
    }

    // sandlib.py: `host, port = name[6:].split(":")`, connect a real
    // AF_INET/SOCK_STREAM socket on the trusted side, and hand the child an fd.
    fn do_open_socket(&mut self, target: &str) -> SandboxResult<Reply> {
        let (host, port) = target
            .split_once(':')
            .ok_or(SandboxError::Os(libc::EINVAL))?;
        let port: u16 = port.parse().map_err(|_| SandboxError::Os(libc::EINVAL))?;
        let stream = TcpStream::connect((host, port))
            .map_err(|e| SandboxError::Os(e.raw_os_error().unwrap_or(libc::ECONNREFUSED)))?;
        let fd = self.allocate_socket_fd(stream)?;
        Ok(Reply::Value(MarshalValue::Int(fd as i64)))
    }

    // sandlib.py `do_ll_os__ll_os_close`.
    fn do_close(&mut self, args: &[MarshalValue]) -> SandboxResult<Reply> {
        let fd = arg_int(args, 0)? as i32;
        // A `tcp://` fd closes by dropping the stream (sandlib.py:495-496).
        if self.open_fds.remove(&fd).is_none() && self.sockets.remove(&fd).is_none() {
            return Err(SandboxError::Os(libc::EBADF));
        }
        Ok(Reply::Value(MarshalValue::None))
    }

    // sandlib.py `do_ll_os__ll_os_read` (+ SimpleIO fallback for fd 0).
    fn do_read(&mut self, args: &[MarshalValue], console: &mut Console) -> SandboxResult<Reply> {
        let fd = arg_int(args, 0)? as i32;
        let size = arg_int(args, 1)?;
        // sandlib.py do_ll_os__ll_os_read: a `tcp://` fd recv's one chunk from the stream.
        if let Some(stream) = self.sockets.get_mut(&fd) {
            if size < 0 {
                return Err(SandboxError::Os(libc::EINVAL));
            }
            // Cap like the virtual-file branch so the child cannot force an
            // unbounded controller-side buffer allocation.
            let want = (size as usize).min(MAX_READ);
            let mut buf = vec![0u8; want];
            let got = stream
                .read(&mut buf)
                .map_err(|_| SandboxError::Os(libc::EIO))?;
            buf.truncate(got);
            return Ok(Reply::Value(MarshalValue::Str(buf)));
        }
        if let Some(entry) = self.open_fds.get_mut(&fd) {
            if size < 0 {
                return Err(SandboxError::Os(libc::EINVAL));
            }
            let want = (size as usize).min(MAX_READ);
            let data =
                read_upto(&mut entry.handle, want).map_err(|_| SandboxError::Os(libc::EIO))?;
            Ok(Reply::Value(MarshalValue::Str(data)))
        } else if fd == 0 {
            // SimpleIOSandboxedProc.do_ll_os__ll_os_read (sandlib.py).
            if size < 0 {
                return Err(SandboxError::Os(libc::EINVAL));
            }
            // Cap the request like the virtual-file branch above so an
            // untrusted child cannot pin a 4 GiB read buffer on the controller.
            let want = (size as usize).min(MAX_READ);
            let data = if self.virtual_console_isatty || console.input_isatty {
                // Waiting at the interactive prompt is idle time: flag it so the
                // watchdog keeps the activity clock fresh and does not charge the
                // wait against --timeout (sandlib.py enter_idle/leave_idle).
                self.timeout_control.idle.store(true, Ordering::Relaxed);
                let r = read_line(console.input, want);
                self.timeout_control.idle.store(false, Ordering::Relaxed);
                r
            } else {
                read_upto(console.input, want)
            }
            .map_err(|_| SandboxError::Io)?;
            // sandlib.py:355-356: mirror the bytes handed to the child into the
            // input log when `--log` opened one.
            if let Some(log) = self.input_log.as_mut() {
                let _ = log.write_all(&data);
            }
            Ok(Reply::Value(MarshalValue::Str(data)))
        } else {
            // sandlib.py:358 raises an errno-less OSError("trying to read from
            // fd ..."); write_exception serializes its None errno as EPERM.
            Err(SandboxError::Os(0))
        }
    }

    // sandlib.py `do_ll_os__ll_os_write` (SimpleIO: fd 1/2 only).
    fn do_write(&mut self, args: &[MarshalValue], console: &mut Console) -> SandboxResult<Reply> {
        let fd = arg_int(args, 0)? as i32;
        let data = arg_bytes(args, 1)?;
        // sandlib.py do_ll_os__ll_os_write: a `tcp://` fd send's the payload down the stream.
        if let Some(stream) = self.sockets.get_mut(&fd) {
            let sent = stream.write(&data).map_err(|_| SandboxError::Io)?;
            return Ok(Reply::Value(MarshalValue::Int(sent as i64)));
        }
        let sink: &mut dyn Write = match fd {
            1 => console.output,
            2 => console.error,
            // sandlib.py:367 raises an errno-less OSError("trying to write to
            // fd ..."); write_exception serializes its None errno as EPERM.
            _ => return Err(SandboxError::Os(0)),
        };
        sink.write_all(&data).map_err(|_| SandboxError::Io)?;
        sink.flush().map_err(|_| SandboxError::Io)?;
        Ok(Reply::Value(MarshalValue::Int(data.len() as i64)))
    }

    // Mediated host entropy. pyre's os.urandom / _random seed use getrandom
    // rather than reading /dev/urandom through ll_os, so the trusted controller
    // serves the bytes here instead of the untrusted child touching host entropy.
    fn do_urandom(&mut self, args: &[MarshalValue]) -> SandboxResult<Reply> {
        let size = arg_int(args, 0)?;
        if size < 0 {
            return Err(SandboxError::Os(libc::EINVAL));
        }
        // Bound the reply like reads so the child cannot demand an unbounded
        // allocation; os.urandom needs its exact length, so refuse an over-large
        // request rather than truncating it.
        if size as usize > MAX_READ {
            return Err(SandboxError::Os(libc::EINVAL));
        }
        let mut buf = vec![0u8; size as usize];
        std::fs::File::open("/dev/urandom")
            .and_then(|mut f| f.read_exact(&mut buf))
            .map_err(|_| SandboxError::Io)?;
        Ok(Reply::Value(MarshalValue::Str(buf)))
    }

    // sandlib.py `do_ll_os__ll_os_stat` (and lstat alias).
    fn do_stat(&mut self, args: &[MarshalValue]) -> SandboxResult<Reply> {
        let vpath = arg_path(args, 0)?;
        let node = self.get_node(&vpath)?;
        Ok(Reply::Stat(node.stat().map_err(vfs_err)?))
    }

    // sandlib.py `do_ll_os__ll_os_fstat`.
    fn do_fstat(&mut self, args: &[MarshalValue]) -> SandboxResult<Reply> {
        let fd = arg_int(args, 0)? as i32;
        let entry = self
            .open_fds
            .get(&fd)
            .ok_or(SandboxError::Os(libc::EBADF))?;
        Ok(Reply::Stat(entry.node.stat().map_err(vfs_err)?))
    }

    // sandlib.py `do_ll_os__ll_os_lseek`.
    fn do_lseek(&mut self, args: &[MarshalValue]) -> SandboxResult<Reply> {
        let fd = arg_int(args, 0)? as i32;
        let pos = arg_int(args, 1)?;
        let how = arg_int(args, 2)? as i32;
        let entry = self
            .open_fds
            .get_mut(&fd)
            .ok_or(SandboxError::Os(libc::EBADF))?;
        let whence = match how {
            // A negative absolute offset is invalid; without this guard `pos as
            // u64` wraps to a huge position the cursor would accept, so Python
            // sees success instead of OSError(EINVAL).
            libc::SEEK_SET if pos < 0 => return Err(SandboxError::Os(libc::EINVAL)),
            libc::SEEK_SET => SeekFrom::Start(pos as u64),
            libc::SEEK_CUR => SeekFrom::Current(pos),
            libc::SEEK_END => SeekFrom::End(pos),
            _ => return Err(SandboxError::Os(libc::EINVAL)),
        };
        let newpos = entry
            .handle
            .seek(whence)
            .map_err(|_| SandboxError::Os(libc::EINVAL))?;
        Ok(Reply::LongLong(newpos as i64))
    }

    // sandlib.py `do_ll_os__ll_os_access`.
    fn do_access(&mut self, args: &[MarshalValue]) -> SandboxResult<Reply> {
        let vpath = arg_path(args, 0)?;
        let mode = arg_int(args, 1)? as u32;
        match self.get_node(&vpath) {
            Ok(node) => Ok(Reply::Value(MarshalValue::Bool(
                node.access(mode).map_err(vfs_err)?,
            ))),
            Err(SandboxError::Os(e)) if e == libc::ENOENT => {
                Ok(Reply::Value(MarshalValue::Bool(false)))
            }
            Err(e) => Err(e),
        }
    }

    // sandlib.py `do_ll_os__ll_os_isatty`.
    fn do_isatty(&mut self, args: &[MarshalValue]) -> SandboxResult<Reply> {
        let fd = arg_int(args, 0)?;
        let isatty = self.virtual_console_isatty && (fd == 0 || fd == 1 || fd == 2);
        Ok(Reply::Value(MarshalValue::Bool(isatty)))
    }

    // sandlib.py `do_ll_os__ll_os_getcwd`.
    fn do_getcwd(&mut self) -> SandboxResult<Reply> {
        Ok(Reply::Value(MarshalValue::Str(
            self.virtual_cwd.as_bytes().to_vec(),
        )))
    }

    // sandlib.py `do_ll_os__ll_os_strerror`.
    fn do_strerror(&mut self, args: &[MarshalValue]) -> SandboxResult<Reply> {
        let errnum = arg_int(args, 0)? as i32;
        Ok(Reply::Value(MarshalValue::Str(strerror(errnum))))
    }

    // sandlib.py `do_ll_os__ll_os_listdir`.
    fn do_listdir(&mut self, args: &[MarshalValue]) -> SandboxResult<Reply> {
        let vpath = arg_path(args, 0)?;
        let node = self.get_node(&vpath)?;
        let names = node.keys().map_err(vfs_err)?;
        Ok(Reply::Value(MarshalValue::List(
            names
                .into_iter()
                .map(|n| MarshalValue::Str(n.into_bytes()))
                .collect(),
        )))
    }

    // sandlib.py `do_ll_os__ll_os_getenv`.
    fn do_getenv(&mut self, args: &[MarshalValue]) -> SandboxResult<Reply> {
        let name = arg_bytes(args, 0)?;
        let value = self
            .virtual_env
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| MarshalValue::Str(v.clone()))
            .unwrap_or(MarshalValue::None);
        Ok(Reply::Value(value))
    }

    // sandlib.py `do_ll_os__ll_os_envitems`.
    fn do_envitems(&mut self) -> SandboxResult<Reply> {
        let items = self
            .virtual_env
            .iter()
            .map(|(k, v)| {
                MarshalValue::Tuple(vec![
                    MarshalValue::Str(k.clone()),
                    MarshalValue::Str(v.clone()),
                ])
            })
            .collect();
        Ok(Reply::Value(MarshalValue::List(items)))
    }

    // ── SimpleIOSandboxedProc time handlers ──────────────────────────────────

    // sandlib.py `do_ll_time__ll_time_time`.
    fn do_time(&mut self) -> SandboxResult<Reply> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        Ok(Reply::Value(MarshalValue::Float(now)))
    }

    // sandlib.py `do_ll_time__ll_time_clock`.
    fn do_clock(&mut self) -> SandboxResult<Reply> {
        let start = *self.clock_start.get_or_insert_with(Instant::now);
        Ok(Reply::Value(MarshalValue::Float(
            start.elapsed().as_secs_f64(),
        )))
    }

    // sandlib.py `do_ll_time__ll_time_sleep`.
    fn do_sleep(&mut self, args: &[MarshalValue]) -> SandboxResult<Reply> {
        let mut seconds = arg_float(args, 0)?;
        // Reject a non-finite request (NaN/±inf) rather than looping forever or
        // feeding `Duration::from_secs_f64` a value it would panic on.
        if !seconds.is_finite() {
            return Err(SandboxError::Os(libc::EINVAL));
        }
        // Sleep in 5-second chunks and poll the cancellation flag between them
        // (sandlib.py's self.poll()), so a child SIGKILLed mid-sleep does
        // not keep the controller parked for the full requested duration.
        while seconds > 5.0 {
            std::thread::sleep(std::time::Duration::from_secs(5));
            seconds -= 5.0;
            if self.timeout_control.cancelled.load(Ordering::Relaxed) {
                return Ok(Reply::Value(MarshalValue::None));
            }
        }
        if seconds > 0.0 {
            std::thread::sleep(std::time::Duration::from_secs_f64(seconds));
        }
        Ok(Reply::Value(MarshalValue::None))
    }

    // ── the request/reply loop (sandlib.py handle_until_return) ──────────────────────────

    /// Read marshalled `(fnname, args)` requests from `child_stdout`, dispatch
    /// them, and marshal replies to `child_stdin`, until the child closes its
    /// stdout (clean EOF between messages).
    pub fn handle_until_return<RIN, WOUT>(
        &mut self,
        child_stdout: RIN,
        child_stdin: &mut WOUT,
        console: &mut Console,
    ) -> io::Result<()>
    where
        RIN: Read,
        WOUT: Write,
    {
        self.handle_until_return_ticked(child_stdout, child_stdin, console, &mut || {})
    }

    /// Like [`handle_until_return`](Self::handle_until_return), but `on_message`
    /// is invoked once before the loop and after every serviced message. The
    /// controller uses it to reset the per-message timeout watchdog — the
    /// `signal.alarm` reset RPython wraps around each `read_message`
    /// (`sandlib.py:_signal_alarm`).
    pub fn handle_until_return_ticked<RIN, WOUT>(
        &mut self,
        child_stdout: RIN,
        child_stdin: &mut WOUT,
        console: &mut Console,
        on_message: &mut dyn FnMut(),
    ) -> io::Result<()>
    where
        RIN: Read,
        WOUT: Write,
    {
        let mut loader = Loader::new(Vec::new(), ReadNeedMore::new(child_stdout));
        on_message();
        loop {
            // A clean EOF *at a message boundary* is normal termination; a read
            // error mid-stream (the child died, or sent a truncated frame) is a
            // real failure and must surface — otherwise a crashed child is
            // reported as a successful run.
            match loader.at_message_boundary_eof() {
                Ok(true) => break,
                Ok(false) => {}
                Err(e) => return Err(protocol_io_error(e)),
            }
            let fnname = load_string(&mut loader).map_err(protocol_io_error)?;
            let args = load_value(&mut loader).map_err(protocol_io_error)?;
            let fnname = String::from_utf8_lossy(&fnname).into_owned();
            let mut out = Vec::new();
            match self.handle_message(&fnname, &args, console) {
                Ok(reply) => {
                    dump_int(&mut out, 0, IntFlavor::Marshal); // success code
                    encode_reply(&mut out, &reply);
                }
                Err(exc) => encode_exception(&mut out, &exc),
            }
            child_stdin.write_all(&out)?;
            child_stdin.flush()?;
            // Reclaim the bytes of the message just serviced so the loader's
            // buffer does not grow without bound over a long request stream.
            loader.drain_consumed();
            on_message();
        }
        Ok(())
    }
}

/// Map a marshal-decode failure on the request stream to an `io::Error` so the
/// controller loop propagates a crashed/rogue child instead of treating the
/// malformed input as a clean exit. A truncated frame surfaces as `UnexpectedEof`;
/// anything else as `InvalidData`.
fn protocol_io_error(e: SandboxError) -> io::Error {
    match e {
        SandboxError::Protocol(msg) if msg.contains("unexpected EOF") => {
            io::Error::new(io::ErrorKind::UnexpectedEof, msg)
        }
        other => io::Error::new(io::ErrorKind::InvalidData, other.to_string()),
    }
}

// sandlib.py reply framing + sandlib.py write_message resulttype encoding.
fn encode_reply(out: &mut Vec<u8>, reply: &Reply) {
    match reply {
        Reply::Value(v) => dump_value(out, v, IntFlavor::Marshal),
        Reply::Stat(st) => dump_statresult(out, st),
        Reply::LongLong(v) => dump_longlong_result(out, *v),
    }
}

// sandlib.py `write_exception`.
fn encode_exception(out: &mut Vec<u8>, exc: &SandboxError) {
    dump_int(out, code_for_error(exc), IntFlavor::Marshal);
    if let SandboxError::Os(errno) = exc {
        let errno = if *errno == 0 { libc::EPERM } else { *errno };
        dump_int(out, errno as i64, IntFlavor::Marshal);
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn vfs_err(e: vfs::VfsError) -> SandboxError {
    SandboxError::Os(e.errno)
}

fn arg_int(args: &[MarshalValue], i: usize) -> SandboxResult<i64> {
    match args.get(i) {
        Some(MarshalValue::Int(v)) => Ok(*v),
        Some(MarshalValue::Bool(b)) => Ok(*b as i64),
        _ => Err(SandboxError::Value),
    }
}

fn arg_float(args: &[MarshalValue], i: usize) -> SandboxResult<f64> {
    match args.get(i) {
        Some(MarshalValue::Float(v)) => Ok(*v),
        Some(MarshalValue::Int(v)) => Ok(*v as f64),
        _ => Err(SandboxError::Value),
    }
}

fn arg_bytes(args: &[MarshalValue], i: usize) -> SandboxResult<Vec<u8>> {
    match args.get(i) {
        Some(MarshalValue::Str(s)) => Ok(s.clone()),
        _ => Err(SandboxError::Value),
    }
}

fn arg_path(args: &[MarshalValue], i: usize) -> SandboxResult<String> {
    // The VFS is keyed by `String` (vfs.rs), so a path is decoded lossily here
    // rather than carried as raw bytes through posixpath as PyPy's sandlib.py
    // does. A non-UTF-8 child path gets U+FFFD substituted, which cannot
    // synthesize `/` or `..` and so only fails to resolve (ENOENT) — strictly
    // more restrictive than a host lookup, never an escape.
    let bytes = arg_bytes(args, i)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

// Read up to `n` bytes, looping over short reads until `n` or EOF.
fn read_upto(reader: &mut dyn Read, n: usize) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut chunk = [0u8; 8192];
    while out.len() < n {
        let want = (n - out.len()).min(chunk.len());
        let got = reader.read(&mut chunk[..want])?;
        if got == 0 {
            break;
        }
        out.extend_from_slice(&chunk[..got]);
    }
    Ok(out)
}

// Read at most `n` bytes, stopping early after a newline (tty line behaviour,
// sandlib.py:344-352).
fn read_line(reader: &mut dyn Read, n: usize) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut byte = [0u8; 1];
    while out.len() < n {
        let got = reader.read(&mut byte)?;
        if got == 0 {
            break;
        }
        out.push(byte[0]);
        if byte[0] == b'\n' {
            break;
        }
    }
    Ok(out)
}

fn strerror(errno: i32) -> Vec<u8> {
    unsafe {
        let ptr = libc::strerror(errno);
        if ptr.is_null() {
            return format!("Unknown error {errno}").into_bytes();
        }
        std::ffi::CStr::from_ptr(ptr).to_bytes().to_vec()
    }
}

// posixpath.join(a, b) — `b` absolute wins (posixpath.py).
fn posix_join(a: &str, b: &str) -> String {
    if b.starts_with('/') {
        b.to_owned()
    } else if a.is_empty() || a.ends_with('/') {
        format!("{a}{b}")
    } else {
        format!("{a}/{b}")
    }
}

// posixpath.normpath — collapse '.', '..', and redundant separators.
fn posix_normpath(path: &str) -> String {
    if path.is_empty() {
        return ".".to_owned();
    }
    let absolute = path.starts_with('/');
    // posixpath preserves exactly two leading slashes, but the sandbox never
    // relies on that quirk; a single leading slash is sufficient here.
    let mut out: Vec<&str> = Vec::new();
    for comp in path.split('/') {
        match comp {
            "" | "." => {}
            ".." => {
                if let Some(&last) = out.last()
                    && last != ".."
                {
                    out.pop();
                    continue;
                }
                if !absolute {
                    out.push("..");
                }
            }
            other => out.push(other),
        }
    }
    let joined = out.join("/");
    match (absolute, joined.is_empty()) {
        (true, _) => format!("/{joined}"),
        (false, true) => ".".to_owned(),
        (false, false) => joined,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfs::{Dir, File, RealFile};
    use indexmap::IndexMap;
    use std::rc::Rc;

    fn dir(entries: Vec<(&str, FsNode)>) -> FsNode {
        let mut map: IndexMap<String, FsNode> = IndexMap::new();
        for (k, v) in entries {
            map.insert(k.to_owned(), v);
        }
        Rc::new(Dir::new(map))
    }

    // VFS used by the sandlib integration tests, mirroring
    // test_sandlib.py SandboxedProcWithFiles.build_virtual_root.
    fn files_root() -> FsNode {
        dir(vec![
            ("hi.txt", Rc::new(File::new("Hello, world!\n"))),
            ("this.bin", Rc::new(RealFile::new(file!(), 0))),
        ])
    }

    fn policy() -> SandboxPolicy {
        SandboxPolicy::new(files_root(), "/", vec![], false)
    }

    fn no_console() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        (Vec::new(), Vec::new(), Vec::new())
    }

    fn tup(items: Vec<MarshalValue>) -> MarshalValue {
        MarshalValue::Tuple(items)
    }

    fn call(p: &mut SandboxPolicy, fnname: &str, args: Vec<MarshalValue>) -> SandboxResult<Reply> {
        let (i, mut o, mut e) = no_console();
        let mut console = Console {
            input: &mut i.as_slice(),
            output: &mut o,
            error: &mut e,
            input_isatty: false,
        };
        p.handle_message(fnname, &tup(args), &mut console)
    }

    fn expect_int(r: SandboxResult<Reply>) -> i64 {
        match r {
            Ok(Reply::Value(MarshalValue::Int(v))) => v,
            other => panic!("expected int reply, got {:?}", reply_dbg(&other)),
        }
    }

    fn reply_dbg(r: &SandboxResult<Reply>) -> String {
        match r {
            Ok(Reply::Value(v)) => format!("Value({v:?})"),
            Ok(Reply::Stat(_)) => "Stat".into(),
            Ok(Reply::LongLong(v)) => format!("LongLong({v})"),
            Err(e) => format!("Err({e})"),
        }
    }

    #[test]
    fn unsafe_fnname_rejected() {
        // a fnname with '__' must be refused before dispatch (sandlib.py:277).
        let mut p = policy();
        assert!(matches!(
            call(&mut p, "ll_os.ll_os__secret", vec![]),
            Err(SandboxError::Value)
        ));
    }

    #[test]
    fn unhandled_fnname_is_runtimeerror() {
        // dup/dup2/ftruncate are test-harness helpers, not part of the
        // production controller; an unknown fnname maps to RuntimeError
        // ("no handler", sandlib.py:284).
        let mut p = policy();
        for name in [
            "ll_os.ll_os_dup",
            "ll_os.ll_os_dup2",
            "ll_os.ll_os_ftruncate",
        ] {
            assert!(matches!(
                call(&mut p, name, vec![MarshalValue::Int(3)]),
                Err(SandboxError::Runtime)
            ));
        }
    }

    #[test]
    fn getuid_family_is_1000() {
        // test_sandlib.py test_getuid
        let mut p = policy();
        for name in [
            "ll_os.ll_os_getuid",
            "ll_os.ll_os_geteuid",
            "ll_os.ll_os_getgid",
            "ll_os.ll_os_getegid",
        ] {
            assert_eq!(expect_int(call(&mut p, name, vec![])), 1000);
        }
    }

    #[test]
    fn open_read_close_virtual_file() {
        let mut p = policy();
        let fd = expect_int(call(
            &mut p,
            "ll_os.ll_os_open",
            vec![
                MarshalValue::Str(b"/hi.txt".to_vec()),
                MarshalValue::Int(libc::O_RDONLY as i64),
                MarshalValue::Int(0o777),
            ],
        ));
        assert_eq!(fd, 3); // first fd in range 3..50
        let data = match call(
            &mut p,
            "ll_os.ll_os_read",
            vec![MarshalValue::Int(fd), MarshalValue::Int(100)],
        ) {
            Ok(Reply::Value(MarshalValue::Str(d))) => d,
            other => panic!("read: {}", reply_dbg(&other)),
        };
        assert_eq!(data, b"Hello, world!\n");
        assert!(matches!(
            call(&mut p, "ll_os.ll_os_close", vec![MarshalValue::Int(fd)]),
            Ok(Reply::Value(MarshalValue::None))
        ));
        // closing again -> EBADF
        assert!(matches!(
            call(&mut p, "ll_os.ll_os_close", vec![MarshalValue::Int(fd)]),
            Err(SandboxError::Os(_))
        ));
    }

    #[test]
    fn open_write_mode_denied() {
        // an obvious attack: opening for write must fail with EPERM.
        let mut p = policy();
        let r = call(
            &mut p,
            "ll_os.ll_os_open",
            vec![
                MarshalValue::Str(b"/hi.txt".to_vec()),
                MarshalValue::Int((libc::O_WRONLY | libc::O_CREAT) as i64),
                MarshalValue::Int(0o666),
            ],
        );
        assert!(matches!(r, Err(SandboxError::Os(e)) if e == libc::EPERM));
    }

    #[test]
    fn unlink_and_mkdir_denied() {
        let mut p = policy();
        for name in ["ll_os.ll_os_unlink", "ll_os.ll_os_mkdir"] {
            assert!(matches!(
                call(&mut p, name, vec![MarshalValue::Str(b"/x".to_vec())]),
                Err(SandboxError::Os(e)) if e == libc::EPERM
            ));
        }
    }

    #[test]
    fn fstat_matches_stat() {
        // test_sandlib.py test_fstat
        let mut p = policy();
        let stat_st = match call(
            &mut p,
            "ll_os.ll_os_stat",
            vec![MarshalValue::Str(b"/hi.txt".to_vec())],
        ) {
            Ok(Reply::Stat(s)) => s,
            other => panic!("stat: {}", reply_dbg(&other)),
        };
        let fd = expect_int(call(
            &mut p,
            "ll_os.ll_os_open",
            vec![
                MarshalValue::Str(b"/hi.txt".to_vec()),
                MarshalValue::Int(libc::O_RDONLY as i64),
                MarshalValue::Int(0o777),
            ],
        ));
        let fstat_st = match call(&mut p, "ll_os.ll_os_fstat", vec![MarshalValue::Int(fd)]) {
            Ok(Reply::Stat(s)) => s,
            other => panic!("fstat: {}", reply_dbg(&other)),
        };
        assert_eq!(stat_st, fstat_st);
        assert_eq!(stat_st.st_size, 14); // "Hello, world!\n"
    }

    #[test]
    fn lseek_offsets() {
        // test_sandlib.py test_lseek
        let mut p = policy();
        let fd = expect_int(call(
            &mut p,
            "ll_os.ll_os_open",
            vec![
                MarshalValue::Str(b"/hi.txt".to_vec()),
                MarshalValue::Int(libc::O_RDONLY as i64),
                MarshalValue::Int(0o777),
            ],
        ));
        let lseek = |p: &mut SandboxPolicy, pos: i64, how: i32| -> i64 {
            match call(
                p,
                "ll_os.ll_os_lseek",
                vec![
                    MarshalValue::Int(fd),
                    MarshalValue::Int(pos),
                    MarshalValue::Int(how as i64),
                ],
            ) {
                Ok(Reply::LongLong(v)) => v,
                other => panic!("lseek: {}", reply_dbg(&other)),
            }
        };
        assert_eq!(lseek(&mut p, 0, libc::SEEK_END), 14);
        assert_eq!(lseek(&mut p, 0, libc::SEEK_SET), 0);
        assert_eq!(lseek(&mut p, 7, libc::SEEK_CUR), 7);
    }

    #[test]
    fn too_many_opens_emfile() {
        // test_sandlib.py test_too_many_opens — fd range is 3..50 (47 slots).
        let mut p = policy();
        for _ in 0..(FD_RANGE_END - FD_RANGE_START) {
            expect_int(call(
                &mut p,
                "ll_os.ll_os_open",
                vec![
                    MarshalValue::Str(b"/hi.txt".to_vec()),
                    MarshalValue::Int(libc::O_RDONLY as i64),
                    MarshalValue::Int(0o777),
                ],
            ));
        }
        let r = call(
            &mut p,
            "ll_os.ll_os_open",
            vec![
                MarshalValue::Str(b"/hi.txt".to_vec()),
                MarshalValue::Int(libc::O_RDONLY as i64),
                MarshalValue::Int(0o777),
            ],
        );
        assert!(matches!(r, Err(SandboxError::Os(e)) if e == libc::EMFILE));
    }

    #[test]
    fn stdout_stderr_write() {
        let mut p = policy();
        let mut i: &[u8] = b"";
        let mut o: Vec<u8> = Vec::new();
        let mut e: Vec<u8> = Vec::new();
        {
            let mut console = Console {
                input: &mut i,
                output: &mut o,
                error: &mut e,
                input_isatty: false,
            };
            let n = p
                .handle_message(
                    "ll_os.ll_os_write",
                    &tup(vec![
                        MarshalValue::Int(1),
                        MarshalValue::Str(b"hi\n".to_vec()),
                    ]),
                    &mut console,
                )
                .unwrap();
            assert!(matches!(n, Reply::Value(MarshalValue::Int(3))));
            p.handle_message(
                "ll_os.ll_os_write",
                &tup(vec![
                    MarshalValue::Int(2),
                    MarshalValue::Str(b"err".to_vec()),
                ]),
                &mut console,
            )
            .unwrap();
        }
        assert_eq!(o, b"hi\n");
        assert_eq!(e, b"err");
    }

    #[test]
    fn access_missing_is_false() {
        let mut p = policy();
        let r = call(
            &mut p,
            "ll_os.ll_os_access",
            vec![MarshalValue::Str(b"/nope".to_vec()), MarshalValue::Int(4)],
        );
        assert!(matches!(r, Ok(Reply::Value(MarshalValue::Bool(false)))));
    }

    #[test]
    fn handle_until_return_drives_a_request_stream() {
        // Build a request stream (client -> controller) for getcwd, feed it
        // through the loop, and decode the reply.
        let mut p = SandboxPolicy::new(files_root(), "/tmp", vec![], false);
        let mut request = Vec::new();
        majit_rlib::rmarshal::dump_string(&mut request, b"ll_os.ll_os_getcwd");
        majit_rlib::rmarshal::dump_tuple(&mut request, &[], IntFlavor::Rmarshal);

        let mut replies: Vec<u8> = Vec::new();
        let (i, mut o, mut e) = no_console();
        {
            let mut console = Console {
                input: &mut i.as_slice(),
                output: &mut o,
                error: &mut e,
                input_isatty: false,
            };
            p.handle_until_return(request.as_slice(), &mut replies, &mut console)
                .unwrap();
        }
        // reply = success int 0, then the cwd string.
        let mut ld = Loader::from_bytes(replies);
        assert_eq!(majit_rlib::rmarshal::load_int(&mut ld).unwrap(), 0);
        assert_eq!(majit_rlib::rmarshal::load_string(&mut ld).unwrap(), b"/tmp");
        ld.check_finished().unwrap();
    }

    #[test]
    fn tcp_open_denied_when_net_disabled() {
        // Default policy is network-closed: `tcp://` is not special-cased and
        // falls through to virtual-file resolution, which has no such node.
        let mut p = policy();
        let r = call(
            &mut p,
            "ll_os.ll_os_open",
            vec![
                MarshalValue::Str(b"tcp://127.0.0.1:9".to_vec()),
                MarshalValue::Int(libc::O_RDONLY as i64),
                MarshalValue::Int(0o777),
            ],
        );
        assert!(
            matches!(r, Err(SandboxError::Os(_))),
            "tcp:// must not open without --allow-net: {}",
            reply_dbg(&r)
        );
    }

    #[test]
    fn tcp_open_read_write_roundtrip() {
        // VirtualizedSocketProc (sandlib.py): with allow_net on, opening
        // `tcp://host:port` connects a real socket and routes read/write to it.
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().expect("accept");
            let mut buf = [0u8; 4];
            sock.read_exact(&mut buf).expect("server read");
            assert_eq!(&buf, b"ping");
            sock.write_all(b"pong\n").expect("server write");
        });

        let mut p = policy();
        p.set_allow_net(true);
        let fd = expect_int(call(
            &mut p,
            "ll_os.ll_os_open",
            vec![
                MarshalValue::Str(format!("tcp://127.0.0.1:{port}").into_bytes()),
                MarshalValue::Int(libc::O_RDONLY as i64),
                MarshalValue::Int(0o777),
            ],
        ));
        assert!(fd >= 3, "socket fd in virtual range: {fd}");

        let sent = expect_int(call(
            &mut p,
            "ll_os.ll_os_write",
            vec![MarshalValue::Int(fd), MarshalValue::Str(b"ping".to_vec())],
        ));
        assert_eq!(sent, 4);

        let data = match call(
            &mut p,
            "ll_os.ll_os_read",
            vec![MarshalValue::Int(fd), MarshalValue::Int(100)],
        ) {
            Ok(Reply::Value(MarshalValue::Str(d))) => d,
            other => panic!("read: {}", reply_dbg(&other)),
        };
        assert_eq!(data, b"pong\n");

        assert!(matches!(
            call(&mut p, "ll_os.ll_os_close", vec![MarshalValue::Int(fd)]),
            Ok(Reply::Value(MarshalValue::None))
        ));
        // closing again -> EBADF (socket fd is gone from both tables)
        assert!(matches!(
            call(&mut p, "ll_os.ll_os_close", vec![MarshalValue::Int(fd)]),
            Err(SandboxError::Os(e)) if e == libc::EBADF
        ));
        server.join().expect("server thread");
    }

    #[test]
    fn input_log_records_guest_stdin() {
        // setlogfile/inputlogfile (sandlib.py, 355-356): an fd-0 read
        // appends the bytes handed to the child into the log file.
        let mut path = std::env::temp_dir();
        path.push(format!("pyre_sandbox_inputlog_{}.log", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .expect("open log");

        let mut p = policy();
        p.set_input_log(file);

        let input = b"hello\nworld".to_vec();
        let (mut o, mut e) = (Vec::new(), Vec::new());
        {
            let mut console = Console {
                input: &mut input.as_slice(),
                output: &mut o,
                error: &mut e,
                input_isatty: false,
            };
            let r = p.handle_message(
                "ll_os.ll_os_read",
                &tup(vec![MarshalValue::Int(0), MarshalValue::Int(100)]),
                &mut console,
            );
            assert!(matches!(r, Ok(Reply::Value(MarshalValue::Str(_)))));
        }

        let mut logged = Vec::new();
        std::fs::File::open(&path)
            .unwrap()
            .read_to_end(&mut logged)
            .unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(logged, input);
    }
}
