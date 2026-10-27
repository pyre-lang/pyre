//! Port of `rpython/translator/goal/unixcheckpoint.py`.
//!
//! The upstream module exposes:
//! * `restart_process()` (`unixcheckpoint.py`) — replace the current
//!   process with `os.execv(sys.executable, [sys.executable] + sys.argv)`.
//! * `restartable_point_fork(auto=None, extra_msg=None)` (`:7-66`) — prompt
//!   for `run` / `cont` / `quit` / `pdb` / `restart-it-all`, then fork.  The
//!   parent waits for the child and loops; the child returns to the caller.
//! * `restartable_point_nofork(auto=None)` (`:69-73`) — Windows/VMware helper.
//!
//! Public functions use the real process boundary.  Unit tests exercise the
//! line-by-line control flow through a small runtime trait so the test process
//! is not actually forked.

#[cfg(unix)]
use std::ffi::CString;
use std::io::{self, BufRead, Write};

use crate::translator::tool::taskengine::TaskError;

/// Port of upstream `restart_process()` (`unixcheckpoint.py`).
///
/// On Unix this calls `execv`, so a successful call never returns.  On
/// non-Unix targets it surfaces a `TaskError` because Rust has no direct
/// equivalent of replacing the current process image in this port yet.
pub fn restart_process() -> Result<(), TaskError> {
    RealRuntime.restart_process()
}

/// Platform-selected port of upstream module global `restartable_point`
/// (`unixcheckpoint.py:75-77`).
pub fn restartable_point(auto: Option<&str>) -> Result<(), TaskError> {
    #[cfg(windows)]
    {
        restartable_point_nofork(auto)
    }
    #[cfg(not(windows))]
    {
        restartable_point_fork(auto, None)
    }
}

/// Port of upstream `restartable_point_fork(auto=None, extra_msg=None)`
/// (`unixcheckpoint.py`).
pub fn restartable_point_fork(
    auto: Option<&str>,
    extra_msg: Option<&str>,
) -> Result<(), TaskError> {
    // Upstream `unixcheckpoint.py:13-16`: when `auto` is set, the
    // prompt is bypassed entirely and `raw_input()` is never called.
    // The driver always passes `auto='run'`
    // (`driver.py:347-371` / `:621-622`), so callers reach the inner
    // `break` on the very first iteration and proceed directly to
    // `os.fork`.  No tty inspection happens in upstream; the Rust port
    // mirrors that by reading the terminal only when actually needed.
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    restartable_point_fork_with(auto, extra_msg, &mut input, &mut output, &RealRuntime)
}

/// Port of upstream `restartable_point_nofork(auto=None)` (`:69-73`).
pub fn restartable_point_nofork(auto: Option<&str>) -> Result<(), TaskError> {
    // Upstream `unixcheckpoint.py`: forwards into `restartable_point_fork`
    // with `auto=None` and a fixed extra message.  No tty check is
    // present; mirror that behaviour exactly.
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    restartable_point_fork_with(
        None,
        Some(
            "+++ this system does not support fork +++\n\
             if you have a virtual machine, you can save a snapshot now",
        ),
        &mut input,
        &mut output,
        &NoForkRuntime {
            _ignored_auto: auto,
        },
    )
}

#[cfg(unix)]
type Pid = libc::pid_t;
#[cfg(not(unix))]
type Pid = i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForkResult {
    #[cfg_attr(
        not(unix),
        allow(dead_code, reason = "fork parent result is only constructed on Unix")
    )]
    Parent(Pid),
    #[cfg_attr(
        not(unix),
        allow(dead_code, reason = "fork child result is only constructed on Unix")
    )]
    Child,
    NotSupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WaitStatus {
    #[cfg_attr(
        not(unix),
        allow(dead_code, reason = "wait exit status is only constructed on Unix")
    )]
    Exited(i32),
    #[cfg_attr(
        not(unix),
        allow(dead_code, reason = "wait signal status is only constructed on Unix")
    )]
    Signaled(i32),
    Other(i32),
}

trait CheckpointRuntime {
    fn restart_process(&self) -> Result<(), TaskError>;
    fn fork(&self) -> Result<ForkResult, TaskError>;
    fn waitpid(&self, pid: Pid) -> Result<WaitStatus, WaitError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WaitError {
    #[cfg_attr(
        not(unix),
        allow(dead_code, reason = "EINTR wait retry is only constructed on Unix")
    )]
    Interrupted,
    #[cfg_attr(
        not(unix),
        allow(dead_code, reason = "wait errno is only constructed on Unix")
    )]
    Other(i32),
}

struct RealRuntime;

impl CheckpointRuntime for RealRuntime {
    fn restart_process(&self) -> Result<(), TaskError> {
        restart_process_real()
    }

    fn fork(&self) -> Result<ForkResult, TaskError> {
        #[cfg(unix)]
        {
            // Upstream `unixcheckpoint.py`: `pid = os.fork()`.
            //
            // SAFETY: this intentionally mirrors Python's process boundary.
            // The child returns immediately to the caller and must avoid
            // touching shared Rust synchronization primitives created by
            // other threads before the fork.  That matches the upstream
            // checkpoint contract: callers opt into a fork of the translator
            // process at this exact point.
            let pid = unsafe { libc::fork() };
            if pid < 0 {
                return Err(last_os_task_error("unixcheckpoint.py:38 os.fork"));
            }
            if pid == 0 {
                Ok(ForkResult::Child)
            } else {
                Ok(ForkResult::Parent(pid))
            }
        }
        #[cfg(not(unix))]
        {
            Ok(ForkResult::NotSupported)
        }
    }

    fn waitpid(&self, pid: Pid) -> Result<WaitStatus, WaitError> {
        #[cfg(unix)]
        {
            let mut status = 0;
            // Upstream `unixcheckpoint.py`: `os.waitpid(pid, 0)`.
            let waited = unsafe { libc::waitpid(pid, &mut status, 0) };
            if waited < 0 {
                let code = io::Error::last_os_error().raw_os_error().unwrap_or(-1);
                if code == libc::EINTR {
                    return Err(WaitError::Interrupted);
                }
                return Err(WaitError::Other(code));
            }
            Ok(wait_status_from_raw(status))
        }
        #[cfg(not(unix))]
        {
            let _ = pid;
            Ok(WaitStatus::Other(0))
        }
    }
}

struct NoForkRuntime<'a> {
    _ignored_auto: Option<&'a str>,
}

impl CheckpointRuntime for NoForkRuntime<'_> {
    fn restart_process(&self) -> Result<(), TaskError> {
        restart_process()
    }

    fn fork(&self) -> Result<ForkResult, TaskError> {
        Ok(ForkResult::NotSupported)
    }

    fn waitpid(&self, _pid: Pid) -> Result<WaitStatus, WaitError> {
        Ok(WaitStatus::Other(0))
    }
}

fn restartable_point_fork_with<R: BufRead, W: Write, P: CheckpointRuntime>(
    mut auto: Option<&str>,
    extra_msg: Option<&str>,
    input: &mut R,
    output: &mut W,
    runtime: &P,
) -> Result<(), TaskError> {
    loop {
        loop {
            // Upstream `unixcheckpoint.py:10-11`.
            if let Some(msg) = extra_msg {
                writeln_task(output, msg)?;
            }
            // Upstream `:12`.
            writeln_task(
                output,
                "---> Checkpoint: cont / restart-it-all / quit / pdb ?",
            )?;

            let line = if let Some(tok) = auto.take() {
                // Upstream `:13-16`.
                writeln_task(output, &format!("auto-{tok}"))?;
                tok.trim().to_ascii_lowercase()
            } else {
                // Upstream `:18-22`: KeyboardInterrupt/EOFError are ignored.
                let mut line = String::new();
                match input.read_line(&mut line) {
                    Ok(0) => {
                        writeln_task(output, "(EOFError ignored)")?;
                        continue;
                    }
                    Ok(_) => line.trim().to_ascii_lowercase(),
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {
                        writeln_task(output, "(KeyboardInterrupt ignored)")?;
                        continue;
                    }
                    Err(e) => {
                        return Err(TaskError {
                            message: format!("unixcheckpoint.py:18 raw_input failed: {e}"),
                        });
                    }
                }
            };

            // Upstream `:23-35`.
            match line.as_str() {
                "run" | "cont" => break,
                "quit" => {
                    return Err(TaskError {
                        message: "unixcheckpoint.py:25 SystemExit".to_string(),
                    });
                }
                "pdb" => {
                    // TODO: upstream `unixcheckpoint.py:27`
                    // calls `pdb.set_trace()` to drop into Python's interactive
                    // debugger.  Rust has no in-process equivalent of `pdb`, so
                    // the prompt accepts the keyword and writes the same
                    // `(NotImplementedError ignored)` message upstream prints
                    // when `pdb.set_trace` itself raises and is suppressed by
                    // the surrounding `:18-22 except (KeyboardInterrupt,
                    // EOFError)` clause. The loop continues, matching upstream.
                    writeln_task(output, "(NotImplementedError ignored)")?;
                    continue;
                }
                "restart-it-all" => runtime.restart_process()?,
                _ => continue,
            }
        }

        match runtime.fork()? {
            // Upstream `:39-41`: no `os.fork` attribute means return.
            ForkResult::NotSupported => return Ok(()),
            ForkResult::Parent(pid) => {
                // Upstream `:43-50`: parent waits, ignoring Ctrl-C.
                let status = loop {
                    match runtime.waitpid(pid) {
                        Ok(status) => break status,
                        Err(WaitError::Interrupted) => continue,
                        Err(WaitError::Other(code)) => {
                            return Err(TaskError {
                                message: format!(
                                    "unixcheckpoint.py:45 os.waitpid({pid}, 0) failed: errno {code}"
                                ),
                            });
                        }
                    }
                };
                // Upstream `:51-65`: report child status and restart outer loop.
                writeln_task(output, "")?;
                writeln_task(output, &"_".repeat(78))?;
                match status {
                    WaitStatus::Exited(code) => {
                        writeln_task(output, &format!("Child {pid} exited (exit code {code})"))?;
                    }
                    WaitStatus::Signaled(signal) => {
                        writeln_task(
                            output,
                            &format!("Child {pid} exited (caught signal {signal})"),
                        )?;
                    }
                    WaitStatus::Other(raw) => {
                        writeln_task(
                            output,
                            &format!("Child {pid} exited abnormally (status 0x{raw:x})"),
                        )?;
                    }
                }
                continue;
            }
            ForkResult::Child => {
                // Upstream `:68-69`: in child, print line and return to caller.
                writeln_task(output, &"_".repeat(78))?;
                return Ok(());
            }
        }
    }
}

fn writeln_task<W: Write>(output: &mut W, msg: &str) -> Result<(), TaskError> {
    writeln!(output, "{msg}").map_err(|e| TaskError {
        message: format!("unixcheckpoint.py:print failed: {e}"),
    })
}

#[cfg(unix)]
fn restart_process_real() -> Result<(), TaskError> {
    use std::os::unix::ffi::OsStrExt;

    let executable = std::env::current_exe().map_err(|e| TaskError {
        message: format!("unixcheckpoint.py:4 sys.executable unavailable: {e}"),
    })?;

    // Upstream `unixcheckpoint.py:5`: `os.execv(sys.executable,
    // [sys.executable] + sys.argv)`.  In CPython `sys.argv[0]` is the
    // script path, distinct from `sys.executable`.  Rust's
    // `std::env::args_os()` already starts with the binary path
    // (analogue of `argv[0]`), so the line-by-line equivalent is
    // `[current_exe] + args[1:]`, not `[current_exe] + args[..]`
    // (which duplicates the binary path).
    let mut argv = Vec::new();
    argv.push(executable.as_os_str().as_bytes().to_vec());
    for arg in std::env::args_os().skip(1) {
        argv.push(arg.as_os_str().as_bytes().to_vec());
    }

    let c_executable = CString::new(executable.as_os_str().as_bytes()).map_err(|_| TaskError {
        message: "unixcheckpoint.py:4 sys.executable contains NUL byte".to_string(),
    })?;
    let c_argv = argv
        .iter()
        .map(|arg| {
            CString::new(arg.as_slice()).map_err(|_| TaskError {
                message: "unixcheckpoint.py:4 sys.argv contains NUL byte".to_string(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut argv_ptrs = c_argv.iter().map(|arg| arg.as_ptr()).collect::<Vec<_>>();
    argv_ptrs.push(std::ptr::null());

    // Upstream `unixcheckpoint.py:5`: `os.execv(...)`.
    let rc = unsafe { libc::execv(c_executable.as_ptr(), argv_ptrs.as_ptr()) };
    debug_assert_eq!(rc, -1);
    Err(last_os_task_error("unixcheckpoint.py:5 os.execv"))
}

#[cfg(not(unix))]
fn restart_process_real() -> Result<(), TaskError> {
    Err(TaskError {
        message: "unixcheckpoint.py:3 restart_process requires os.execv".to_string(),
    })
}

#[cfg_attr(
    not(unix),
    expect(
        dead_code,
        reason = "OS-error formatting is only used by Unix syscalls"
    )
)]
fn last_os_task_error(context: &str) -> TaskError {
    TaskError {
        message: format!("{context} failed: {}", io::Error::last_os_error()),
    }
}

#[cfg(unix)]
fn wait_status_from_raw(status: i32) -> WaitStatus {
    if libc::WIFEXITED(status) {
        WaitStatus::Exited(libc::WEXITSTATUS(status))
    } else if libc::WIFSIGNALED(status) {
        WaitStatus::Signaled(libc::WTERMSIG(status))
    } else {
        WaitStatus::Other(status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::io::Cursor;

    struct FakeRuntime {
        forks: RefCell<Vec<ForkResult>>,
        waits: RefCell<Vec<Result<WaitStatus, WaitError>>>,
        restart_count: Cell<usize>,
    }

    impl FakeRuntime {
        fn new(forks: Vec<ForkResult>, waits: Vec<Result<WaitStatus, WaitError>>) -> Self {
            Self {
                forks: RefCell::new(forks),
                waits: RefCell::new(waits),
                restart_count: Cell::new(0),
            }
        }
    }

    impl CheckpointRuntime for FakeRuntime {
        fn restart_process(&self) -> Result<(), TaskError> {
            self.restart_count.set(self.restart_count.get() + 1);
            Ok(())
        }

        fn fork(&self) -> Result<ForkResult, TaskError> {
            Ok(self.forks.borrow_mut().remove(0))
        }

        fn waitpid(&self, _pid: Pid) -> Result<WaitStatus, WaitError> {
            self.waits.borrow_mut().remove(0)
        }
    }

    #[test]
    fn auto_run_child_returns_to_caller() {
        let runtime = FakeRuntime::new(vec![ForkResult::Child], vec![]);
        let mut input = Cursor::new(Vec::<u8>::new());
        let mut output = Vec::new();

        restartable_point_fork_with(Some("run"), None, &mut input, &mut output, &runtime)
            .expect("child path returns");

        let output = String::from_utf8(output).expect("utf8");
        assert!(output.contains("auto-run"));
        assert!(output.contains(&"_".repeat(78)));
    }

    #[test]
    fn parent_waits_reports_and_reprompts() {
        let runtime = FakeRuntime::new(
            vec![ForkResult::Parent(123), ForkResult::Child],
            vec![Err(WaitError::Interrupted), Ok(WaitStatus::Exited(5))],
        );
        let mut input = Cursor::new(b"cont\n".to_vec());
        let mut output = Vec::new();

        restartable_point_fork_with(Some("run"), None, &mut input, &mut output, &runtime)
            .expect("second child path returns");

        let output = String::from_utf8(output).expect("utf8");
        assert!(output.contains("Child 123 exited (exit code 5)"));
        assert_eq!(output.matches("---> Checkpoint").count(), 2);
    }

    #[test]
    fn restart_it_all_calls_restart_process_and_keeps_prompting() {
        let runtime = FakeRuntime::new(vec![ForkResult::Child], vec![]);
        let mut input = Cursor::new(b"run\n".to_vec());
        let mut output = Vec::new();

        restartable_point_fork_with(
            Some("restart-it-all"),
            None,
            &mut input,
            &mut output,
            &runtime,
        )
        .expect("child path returns");

        assert_eq!(runtime.restart_count.get(), 1);
    }

    #[test]
    fn quit_surfaces_system_exit() {
        let runtime = FakeRuntime::new(vec![], vec![]);
        let mut input = Cursor::new(Vec::<u8>::new());
        let mut output = Vec::new();

        let err =
            restartable_point_fork_with(Some("quit"), None, &mut input, &mut output, &runtime)
                .expect_err("quit raises SystemExit upstream");

        assert!(err.message.contains("unixcheckpoint.py:25"));
    }

    #[test]
    fn nofork_returns_after_run_or_cont() {
        let runtime = FakeRuntime::new(vec![ForkResult::NotSupported], vec![]);
        let mut input = Cursor::new(Vec::<u8>::new());
        let mut output = Vec::new();

        restartable_point_fork_with(Some("run"), None, &mut input, &mut output, &runtime)
            .expect("nofork path returns");
    }
}
