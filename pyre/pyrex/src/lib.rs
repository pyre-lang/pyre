//! pyre — A Rust meta-tracing JIT Python interpreter.

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::OnceLock;

use lexopt::Arg::*;
use lexopt::ValueExt;

use pyre_interpreter::app_main::{is_import_error, raised_is_instance_of, seed_main_loader};
use pyre_interpreter::call::{register_build_class, set_last_exec_ctx};
use pyre_interpreter::importing;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::shutdown::finalize_runtime;
use pyre_interpreter::{CompileOpts, Mode, PyDisplay, PyErrorKind, PyExecutionContext};
use pyre_jit::eval::eval_with_jit;

pub mod memory_ceiling;
pub use memory_ceiling::ProcessAllocator;
mod repl;
mod repl_readline;

enum RunMode {
    Script(String),
    Command(String),
    Module(String),
    /// Program read from stdin: an explicit `-` argument, or no script
    /// argument at all (app_main.py `options["run_stdin"]`). `argv0` is the
    /// literal `sys.argv[0]` the launcher records — `"-"` when the dash was
    /// given, `""` when it was not. Whether this drives the prompt or executes
    /// stdin as one unit is decided at dispatch by `stdin_is_interactive`.
    Stdin {
        argv0: String,
    },
    /// `pyre interact <executable> [args…]` — run the trusted sandbox
    /// controller (`pypy/sandbox/pypy_interact.py`) over an untrusted child.
    Interact {
        exe: String,
        args: Vec<String>,
        tmpdir: Option<String>,
        lib_root: Option<String>,
        timeout: Option<f64>,
        allow_net: bool,
        log_file: Option<String>,
        verbose: bool,
    },
}

// The option block and the environment fold over it live in the interpreter,
// beside the `SYS_*` statics they end up in, because the wasm32 embedding
// resolves the same block without a command line of its own.
use pyre_interpreter::launch_env::LaunchFlags;

fn usage(binary_name: &str) -> String {
    format!(
        "\
usage: {binary_name} [option] ... [-c cmd | file | -] [arg] ...
Options:
-c cmd : program passed in as string (terminates option list)
-m mod : run library module as a script (terminates option list)
-d     : debug output from parser; also PYTHONDEBUG=x
-h     : print this help message and exit (also --help)
-i     : inspect interactively after running script
-E     : ignore PYTHON* environment variables
-I     : isolate Python from the user's environment
-B     : don't write .pyc files on import (also PYTHONDONTWRITEBYTECODE=x)
-O     : remove assert and __debug__-dependent statements (also PYTHONOPTIMIZE=x)
-OO    : do -O changes and also discard docstrings
-q     : don't print version on interactive startup
-s     : don't add user site directory to sys.path
-S     : don't imply 'import site' on initialization
-P     : don't prepend a potentially unsafe path to sys.path (also PYTHONSAFEPATH)
-v     : verbose (trace import statements); also PYTHONVERBOSE=x
-V     : print the Python version number and exit (also --version)
-X opt : set implementation-specific option
file   : program read from script file
-      : program read from stdin (default; interactive mode if a tty)
arg ...: arguments passed to program in sys.argv[1:]

Subcommands:
interact [--tmp DIR] [--lib DIR] [--timeout SECS] [--heapsize N] [--log FILE] [--allow-net] [--verbose] <exe> [arg ...]
       : run the trusted sandbox controller over an untrusted sandbox binary
"
    )
}

/// Drain the parser's remaining raw arguments to become `sys.argv[1:]`.
/// `-c`, `-m`, and a script path each terminate option parsing, so anything
/// after them belongs to the program rather than the launcher.
fn drain_args(parser: &mut lexopt::Parser) -> Result<Vec<std::ffi::OsString>, lexopt::Error> {
    Ok(parser.raw_args()?.collect())
}

/// Remove the common space/tab prefix from the `-c` source's nonblank lines and
/// empty its blank ones.
///
/// A line is what sits between two `\n`, so a `\r` is content like any other
/// character: `"  \r"` holds something, narrows the margin to its two spaces
/// and keeps its carriage return, where `"  "` alone is blank. That is a
/// narrower notion of blank than `textwrap.dedent`'s, which empties every line
/// `str.isspace()` answers for.
///
/// A carriage return does reach this point, and is meant to: the dedent runs on
/// the argument as it was given, and the tokenizer's own `\r\n` / lone-`\r`
/// rewrite (`pytokenizer.py universal_newline`) happens after it. Counting `"  \r"` as a
/// line that holds something is what the reference does as well -- against a
/// two-space margin a `"    \r"` line comes back as `"  "`, where a `"    "`
/// line comes back empty.
fn dedent_command(source: &str) -> std::borrow::Cow<'_, str> {
    fn split_newline(line: &str) -> (&str, &str) {
        match line.strip_suffix('\n') {
            Some(content) => (content, "\n"),
            None => (line, ""),
        }
    }
    fn is_blank(content: &str) -> bool {
        content.bytes().all(|byte| matches!(byte, b' ' | b'\t'))
    }

    // The margin is the common leading run of spaces and tabs over the lines
    // that hold something. A blank line never narrows it, and a run stops at
    // the first character that is neither, so an indent built from any other
    // character is not a margin at all.
    let mut margin: Option<&str> = None;
    for line in source.split_inclusive('\n') {
        let (content, _) = split_newline(line);
        if is_blank(content) {
            continue;
        }
        let indent_len = content
            .find(|c| !matches!(c, ' ' | '\t'))
            .unwrap_or(content.len());
        let indent = &content[..indent_len];
        margin = Some(match margin {
            None => indent,
            Some(current) => {
                let common_len = current
                    .bytes()
                    .zip(indent.bytes())
                    .take_while(|(left, right)| left == right)
                    .count();
                &current[..common_len]
            }
        });
    }

    // No margin, no rewriting: a source with a line at column zero keeps its
    // blank lines as they were written, rather than having them emptied on
    // their own.
    let margin = margin.unwrap_or("");
    if margin.is_empty() {
        return std::borrow::Cow::Borrowed(source);
    }
    let mut dedented = String::with_capacity(source.len());
    for line in source.split_inclusive('\n') {
        let (content, newline) = split_newline(line);
        if !is_blank(content) {
            // Every nonblank line contributed to `margin`, so it is a prefix.
            dedented.push_str(&content[margin.len()..]);
        }
        dedented.push_str(newline);
    }
    std::borrow::Cow::Owned(dedented)
}

/// Emit the `preconfig_init_utf8_mode` fatal error for an invalid PYTHONUTF8 /
/// `-X utf8` value and exit; the value is validated during pre-init config.
fn fatal_utf8_config_error(detail: &str) -> ! {
    eprintln!("Fatal Python error: preconfig_init_utf8_mode: {detail}");
    eprintln!("Python runtime state: preinitializing");
    eprintln!();
    std::process::exit(1);
}

/// Fold the process environment over the parsed options. The launcher owns a
/// real environment, so `launch_env` reads `std::env` directly here; an invalid
/// PYTHONUTF8 or cpu count is fatal during config, as it is for `-X utf8`.
fn finalize_flags(flags: LaunchFlags) -> LaunchFlags {
    match pyre_interpreter::launch_env::finalize(flags) {
        Ok(flags) => flags,
        Err(e) => {
            eprint!("{}", e.banner());
            std::process::exit(1);
        }
    }
}

fn parse_args(
    binary_name: &str,
    argv: Vec<std::ffi::OsString>,
) -> Result<(RunMode, LaunchFlags, Vec<std::ffi::OsString>), lexopt::Error> {
    let mut parser = lexopt::Parser::from_iter(argv);
    let mut flags = LaunchFlags::default();

    while let Some(arg) = parser.next()? {
        match arg {
            Short('c') => {
                let cmd = parser.value()?.string()?;
                let rest = drain_args(&mut parser)?;
                return Ok((RunMode::Command(cmd), finalize_flags(flags), rest));
            }
            Short('m') => {
                let module = parser.value()?.string()?;
                let rest = drain_args(&mut parser)?;
                return Ok((RunMode::Module(module), finalize_flags(flags), rest));
            }
            Short('h') | Long("help") => {
                print!("{}", usage(binary_name));
                std::process::exit(0);
            }
            Short('i') => {
                flags.inspect = true;
                flags.interactive = true;
            }
            // app_main.py `cmdline_options['E']` / `X_option`.
            Short('E') => flags.ignore_environment = true,
            // app_main.py `isolated_option`: -I implies -E, -s and -P.
            Short('I') => {
                flags.isolated = true;
                flags.ignore_environment = true;
                flags.no_user_site = true;
                flags.safe_path = true;
            }
            Short('X') => {
                // The value reaches `sys._xoptions` in the host's own
                // spelling, so it is kept as an `OsString`; only the options
                // pyre acts on are matched, and every one of those is ASCII,
                // so a value with no UTF-8 form simply cannot be one of them.
                let option: std::ffi::OsString = parser.value()?;
                // `sys._xoptions` splits each option at the first `=`, so an
                // option `run_command_line` tests with `in` answers to its key
                // alone while one that reads the value does not.  The split is
                // on the encoded bytes because `pycache_prefix` carries a
                // *directory*, which is not required to have a UTF-8 form; `=`
                // is ASCII, and `OsStr` documents its encoded form as
                // splittable at an ASCII byte, so each half is a whole `OsStr`.
                let bytes = option.as_encoded_bytes();
                let (key, value) = match bytes.iter().position(|b| *b == b'=') {
                    Some(at) => (&bytes[..at], Some(&bytes[at + 1..])),
                    None => (bytes, None),
                };
                // `'faulthandler' in sys._xoptions`, so every spelling that
                // names the key installs the handlers -- `=0` included, which
                // is not a way to ask for it to stay off.
                if key == b"faulthandler" {
                    flags.faulthandler = true;
                }
                match option.to_str() {
                    Some("dev") => flags.dev_mode = true,
                    Some("warn_default_encoding") => flags.warn_default_encoding = true,
                    Some("no_debug_ranges") => flags.no_debug_ranges = true,
                    Some("utf8") | Some("utf8=1") => flags.utf8_mode = Some(1),
                    Some("utf8=0") => flags.utf8_mode = Some(0),
                    Some(value) if value.starts_with("utf8=") => {
                        fatal_utf8_config_error("invalid -X utf8 option value")
                    }
                    // An empty value is recorded rather than dropped: a
                    // named option and an unnamed one are what the three-state
                    // fold tells apart, and only the second lets
                    // PYTHONPYCACHEPREFIX supply a prefix.
                    _ => {
                        if key == b"pycache_prefix" {
                            let path = value.unwrap_or(&bytes[..0]);
                            flags.pycache_prefix = Some(
                                unsafe { std::ffi::OsStr::from_encoded_bytes_unchecked(path) }
                                    .to_os_string(),
                            );
                        }
                    }
                }
                flags.xoptions.push(option);
            }
            // Accepted for CPython/PyPy command-line compatibility.  Warning
            // filter installation is handled by the warnings module; keeping
            // the option/value in launcher parsing lets stdlib subprocesses
            // reach their command or module.
            Short('W') => {
                flags.warnoptions.push(parser.value()?);
            }
            // `-O` / `-OO` raise the optimization level; each occurrence counts
            // (app_main.py `optimize`). PYTHONOPTIMIZE folds in during finalize.
            Short('O') => flags.optimize = flags.optimize.saturating_add(1),
            // PyPy app_main.py `simple_option`: repeated `-b` increments the
            // bytes-warning level (`-bb` therefore records 2).
            Short('b') => flags.bytes_warning = flags.bytes_warning.saturating_add(1),
            // The other two `simple_option` counters. `-v` reaches the import
            // machinery through `sys.flags.verbose`, which `_verbose_message`
            // reads; `-d` names parser debug output there is none of, and is
            // carried so `sys.flags.debug` reports it.
            Short('v') => flags.verbose = flags.verbose.saturating_add(1),
            Short('d') => flags.debug = flags.debug.saturating_add(1),
            // `-B` disables writing bytecode caches (PYTHONDONTWRITEBYTECODE).
            Short('B') => flags.dont_write_bytecode = true,
            // PyPy app_main.py `unbuffered`: writing streams use raw FileIO
            // underneath TextIOWrapper and set write_through.
            Short('u') => flags.unbuffered = true,
            Short('q') => flags.quiet = true,
            Short('s') => flags.no_user_site = true,
            Short('S') => flags.no_site = true,
            Short('P') => flags.safe_path = true,
            Short('V') | Long("version") => {
                println!("{binary_name} 0.0.1");
                std::process::exit(0);
            }
            Value(script) => {
                let script = script.string()?;
                if script == "interact" {
                    let mode = parse_interact(&mut parser)?;
                    return Ok((mode, finalize_flags(flags), vec![]));
                }
                if script == "-" {
                    // A leading `-` selects stdin and stays `sys.argv[0]`; the
                    // words after it are the program's arguments, exactly as
                    // for a script path.
                    let rest = drain_args(&mut parser)?;
                    return Ok((
                        RunMode::Stdin { argv0: script },
                        finalize_flags(flags),
                        rest,
                    ));
                }
                let rest = drain_args(&mut parser)?;
                return Ok((RunMode::Script(script), finalize_flags(flags), rest));
            }
            _ => return Err(arg.unexpected()),
        }
    }
    // No script argument at all reads stdin too, with an empty `sys.argv[0]`.
    Ok((
        RunMode::Stdin {
            argv0: String::new(),
        },
        finalize_flags(flags),
        vec![],
    ))
}

/// Parse a `--heapsize` value (`pypy_interact.py main`): a byte count with an
/// optional `k`/`m`/`g` suffix. The suffixes are the controller's spelling and
/// stop here — what it hands the sandboxed program is the byte count written
/// out, so [`take_heapsize_option`] on the other side parses a plain integer.
fn parse_heapsize(value: &str) -> Result<u64, lexopt::Error> {
    let value = value.trim().to_ascii_lowercase();
    let (digits, mult) = match value.strip_suffix('k') {
        Some(d) => (d, 1024u64),
        None => match value.strip_suffix('m') {
            Some(d) => (d, 1024 * 1024),
            None => match value.strip_suffix('g') {
                Some(d) => (d, 1024 * 1024 * 1024),
                None => (value.as_str(), 1),
            },
        },
    };
    let n: u64 = digits.parse().map_err(|_| {
        lexopt::Error::Custom(format!("interact: invalid --heapsize value: {value}").into())
    })?;
    let bytes = n.checked_mul(mult).ok_or_else(|| {
        lexopt::Error::Custom(format!("interact: --heapsize overflow: {value}").into())
    })?;
    if bytes == 0 {
        return Err(lexopt::Error::Custom(
            "interact: --heapsize must be positive".into(),
        ));
    }
    // `bytes > sys.maxint` is the upstream OverflowError. `sys.maxsize` is
    // `i64::MAX` here on every target, and the child parses the count back out
    // of its own command line, so a value it could not name is refused now.
    if bytes > i64::MAX as u64 {
        return Err(lexopt::Error::Custom(
            format!("interact: --heapsize maximum is {}", i64::MAX).into(),
        ));
    }
    Ok(bytes)
}

/// Base-10 `LLHelpers.ll_int`, the translation of
/// `targetpypystandalone.py entry_point`'s `int(argv[2])`. A `ValueError`
/// there is `None` here; out of i64 range is also `None`.
fn parse_python_int_i64(raw: &str) -> Option<i64> {
    let chars = raw.as_bytes();
    let strlen = chars.len();
    let mut i = 0;
    // only space is allowed as white space
    while i < strlen && chars[i] == b' ' {
        i += 1;
    }
    if !(i < strlen) {
        return None;
    }
    let mut sign = 1i64;
    if chars[i] == b'-' {
        sign = -1;
        i += 1;
    } else if chars[i] == b'+' {
        i += 1;
    }
    while i < strlen && chars[i] == b' ' {
        i += 1;
    }
    let base = 10u64;
    let mut val = 0u64;
    let oldpos = i;
    while i < strlen {
        let c = chars[i];
        let digit = if b'a' <= c && c <= b'z' {
            (c - b'a') as u64 + 10
        } else if b'A' <= c && c <= b'Z' {
            (c - b'A') as u64 + 10
        } else if b'0' <= c && c <= b'9' {
            (c - b'0') as u64
        } else {
            break;
        };
        if digit >= base {
            break;
        }
        val = val.checked_mul(base)?.checked_add(digit)?;
        i += 1;
    }
    if i == oldpos {
        return None;
    }
    while i < strlen && chars[i] == b' ' {
        i += 1;
    }
    if i != strlen {
        return None;
    }
    // `return sign * val`. Overflow raises; keep the existing miss.
    if sign < 0 && val == (i64::MAX as u64) + 1 {
        Some(i64::MIN)
    } else {
        let magnitude = i64::try_from(val).ok()?;
        Some(if sign < 0 { -magnitude } else { magnitude })
    }
}

/// `targetpypystandalone.py entry_point`: `--heapsize N`, in that position and
/// no other, is an undocumented interp-level option that exists to support
/// sandboxing — `pypy_interact.py` is what puts it there. Taking it out here
/// leaves the argv `parse_args` reads, and the one `sys.orig_argv` records,
/// equal to what `app_main.py` would have received: `entry_point` rewrites
/// `argv` before app-level parsing ever sees it.
fn take_heapsize_option(
    binary_name: &str,
    mut argv: Vec<std::ffi::OsString>,
) -> (Vec<std::ffi::OsString>, Option<i64>) {
    if argv.len() <= 2 || argv[1] != "--heapsize" {
        return (argv, None);
    }
    // `int(argv[2])` (`targetpypystandalone.py entry_point`) is
    // `LLHelpers.ll_int` at base 10. A value that does not match or does
    // not fit i64 is refused rather than taken.
    let size = match argv[2].to_str().and_then(parse_python_int_i64) {
        Some(size) => size,
        None => {
            eprintln!("{binary_name}: --heapsize: not a byte count");
            std::process::exit(2);
        }
    };
    argv.drain(1..3);
    (argv, Some(size))
}

/// Parse the `interact` subcommand: `pyre interact [--tmp DIR] [--lib DIR]
/// [--timeout SECS] [--heapsize N] [--log FILE] [--allow-net] [--verbose]
/// <executable> [program args…]`. The first positional is the untrusted sandbox
/// binary; everything after it is passed through as that program's arguments.
fn parse_interact(parser: &mut lexopt::Parser) -> Result<RunMode, lexopt::Error> {
    let mut tmpdir = None;
    let mut lib_root = None;
    let mut timeout = None;
    let mut allow_net = false;
    let mut log_file = None;
    let mut verbose = false;
    let mut heapsize = None;
    while let Some(arg) = parser.next()? {
        match arg {
            Long("tmp") => tmpdir = Some(parser.value()?.string()?),
            Long("lib") => lib_root = Some(parser.value()?.string()?),
            Long("timeout") => {
                let secs: f64 = parser.value()?.parse()?;
                // `Duration::from_secs_f64` panics on NaN/inf/negative; reject
                // them here as a usage error instead.
                if !secs.is_finite() || secs < 0.0 {
                    return Err(lexopt::Error::Custom(
                        format!(
                            "interact: --timeout must be a non-negative finite number (got {secs})"
                        )
                        .into(),
                    ));
                }
                timeout = Some(secs);
            }
            // `pypy_interact.py main`: the controller does not limit its own
            // heap. It resolves the suffix and carries the byte count into the
            // sandboxed program's arguments, where `take_heapsize_option` acts
            // on it.
            Long("heapsize") => {
                heapsize = Some(parse_heapsize(&parser.value()?.string()?)?);
            }
            // setlogfile (sandlib.py): append the guest's stdin to FILE.
            Long("log") => log_file = Some(parser.value()?.string()?),
            // VirtualizedSocketProc (sandlib.py): opt in to `tcp://host:port`
            // os.open mediation. Off by default so the sandbox stays
            // network-closed.
            Long("allow-net") => allow_net = true,
            Long("verbose") => verbose = true,
            Value(exe) => {
                let exe = exe.string()?;
                // The controller's arguments are its own, not a Python
                // program's: they are rendered into the child's command line
                // as text, so one with no UTF-8 form is reported here rather
                // than carried the way `sys.argv` carries it.
                let program_args = drain_args(parser)?
                    .into_iter()
                    .map(|arg| arg.into_string().map_err(lexopt::Error::NonUnicodeValue))
                    .collect::<Result<Vec<String>, _>>()?;
                // `extraoptions + arguments[1:]`: the controller's own options
                // lead the child's command line, ahead of the arguments meant
                // for the program itself.
                let mut args = match heapsize {
                    Some(bytes) => vec!["--heapsize".to_string(), bytes.to_string()],
                    None => Vec::new(),
                };
                args.extend(program_args);
                return Ok(RunMode::Interact {
                    exe,
                    args,
                    tmpdir,
                    lib_root,
                    timeout,
                    allow_net,
                    log_file,
                    verbose,
                });
            }
            _ => return Err(arg.unexpected()),
        }
    }
    Err(lexopt::Error::Custom(
        "interact: missing <executable> argument".into(),
    ))
}

/// The working directory to seed `sys.path` with. Under sandbox it is read from
/// the controller (the virtual `/tmp`) through `rposix.getcwd`, so an untrusted
/// child never learns the trusted parent's real working directory; off sandbox
/// it is the process cwd.
fn sys_path_cwd() -> std::path::PathBuf {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let got = {
            #[cfg(feature = "sandbox")]
            {
                match majit_rlib::rsandbox::catch_os_error(|| majit_rlib::rposix::getcwd()) {
                    Ok(Ok(Ok(bytes))) => Ok(bytes),
                    Ok(Ok(Err(_))) => Err(()),
                    Ok(Err(_)) => panic!("MemoryError while reading the current directory"),
                    Err(_) => Err(()),
                }
            }
            #[cfg(not(feature = "sandbox"))]
            {
                // `rposix.getcwd` lets `MemoryError` escape; only an OS error falls back.
                match majit_rlib::rposix::getcwd() {
                    Ok(Ok(bytes)) => Ok(bytes),
                    Ok(Err(_)) => Err(()),
                    Err(_) => panic!("MemoryError while reading the current directory"),
                }
            }
        };
        match got {
            Ok(bytes) => std::path::PathBuf::from(std::ffi::OsStr::from_bytes(&bytes)),
            Err(()) => Path::new(".").to_path_buf(),
        }
    }
    #[cfg(not(unix))]
    {
        std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf())
    }
}

/// `stdin_is_interactive` — a stdin run drives the prompt when fd 0 is a
/// terminal or `-i` was given, and executes stdin as a script otherwise.
///
/// The argument is `config->interactive`, which only `-i` sets.  PYTHONINSPECT
/// does not reach it, so a run that owes the flag to the environment still
/// needs the terminal.
fn stdin_is_interactive(interactive: bool) -> bool {
    if interactive {
        return true;
    }
    #[cfg(unix)]
    {
        // `rposix.isatty`; under sandbox the trampoline reports the virtual
        // console rather than the trusted parent's terminal.
        majit_rlib::rposix::isatty(0)
    }
    #[cfg(not(unix))]
    {
        std::io::IsTerminal::is_terminal(&std::io::stdin())
    }
}

/// `pymain_repl`'s three tests for the prompt that opens over a program that
/// has just finished.
///
/// PYTHONINSPECT is read here rather than taken from the option block, because
/// upstream checks it "at the end, to give programs the opportunity to set it
/// from Python" -- and `os.environ` writes reach `putenv`, so a program that
/// sets the name during its own run is asking for this prompt.  `-i` is the
/// only spelling that opens one over a stdin that is not a terminal.
///
/// `run_code` is `config_run_code`: the prompt that follows `-c`, `-m` or a
/// script.  Stdin read as a program is not one of them, and neither the option
/// nor the variable gives it a prompt afterwards.
fn prompt_requested(run_code: bool) -> bool {
    if !run_code {
        return false;
    }
    // The startup guard already refused every prompt this build can be asked
    // for, and it could still say why.  Reaching one from here would be past
    // that refusal, so the answer is no.
    #[cfg(feature = "sandbox")]
    {
        return false;
    }
    #[cfg(not(feature = "sandbox"))]
    {
        if importing::interactive_flag() {
            return true;
        }
        let inspect = importing::inspect_flag()
            || (!importing::ignore_environment_flag()
                && std::env::var_os("PYTHONINSPECT").is_some_and(|value| !value.is_empty()));
        inspect && stdin_is_interactive(false)
    }
}

/// `sys.stdin.read()` — the whole of stdin as the program source.
fn read_stdin_source() -> std::io::Result<String> {
    // Under sandbox the read goes through `rposix.read`, so a sandboxed child
    // observes virtual stdin rather than the trusted parent's stdin.
    #[cfg(feature = "sandbox")]
    let data = {
        let mut data = Vec::new();
        loop {
            match majit_rlib::rposix::read(0, 65536) {
                Ok(Ok(llobj)) => {
                    let chunk = unsafe {
                        majit_rlib::lltypesystem::rstr::str_as_bytes(
                            llobj.0 as *const majit_rlib::lltypesystem::rstr::STR,
                        )
                    };
                    if chunk.is_empty() {
                        break;
                    }
                    data.extend_from_slice(chunk);
                }
                Ok(Err(errno)) => return Err(std::io::Error::from_raw_os_error(errno)),
                Err(e) => majit_rlib::rsandbox::reraise(e),
            }
        }
        data
    };
    #[cfg(not(feature = "sandbox"))]
    let data = {
        let mut data = Vec::new();
        std::io::Read::read_to_end(&mut std::io::stdin(), &mut data)?;
        data
    };
    String::from_utf8(data)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "stdin not utf-8"))
}

/// Catch a non-OS [`SandboxError`] from the stdin trampoline the way
/// `app_main.py run_toplevel` reports an uncaught exception from `run_it`.
/// `OSError` stays in [`read_stdin_source`]'s `Result`; any other payload is
/// resumed so it is not swallowed here.
#[cfg(feature = "sandbox")]
fn read_stdin_source_catching_sandbox() -> std::io::Result<String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(read_stdin_source)) {
        Ok(result) => result,
        Err(payload) => match payload.downcast::<majit_rlib::rsandbox::SandboxError>() {
            Ok(err) => {
                let execution_context = setup_exec_context();
                let ec_ptr = Rc::as_ptr(&execution_context);
                let (canonical, _main_module) =
                    pyre_interpreter::app_main::prepare_main_module(&execution_context);
                let pyerr = pyre_interpreter::error::get_converted_unexpected_exception(
                    pyre_object::PY_NULL,
                    &*err,
                );
                handle_main_error(pyerr, canonical, ec_ptr);
            }
            Err(payload) => std::panic::resume_unwind(payload),
        },
    }
}

// Native stack the interpreter runs on. Both figures sit well above the byte
// budget `sys.setrecursionlimit` can buy
// (`stack_check::DEFAULT_RUNTIME_THREAD_STACK_SIZE`): `stack_check` gates
// Python call levels, and the native recursion it does not gate — nested
// container `repr`, the parser, GC tracing — draws on the rest. The two entry
// shapes below want it in different currencies.

/// Soft `RLIMIT_STACK` the interpreter asks for when it runs on the thread
/// that started the process, whose stack the kernel grows on demand rather
/// than reserving up front.  64 MiB is as far as that goes: linux keeps the
/// mmap region a minimum of 128 MiB below the stack however high the limit is
/// set, so a larger request would only push the stack into it.
#[cfg(all(target_os = "linux", not(feature = "sandbox")))]
const INTERPRETER_STACK_SIZE: usize = 64 * 1024 * 1024;

/// Stack reserved outright for a spawned interpreter thread.
#[cfg(not(feature = "sandbox"))]
const INTERPRETER_THREAD_STACK_SIZE: usize = 256 * 1024 * 1024;

/// Size of the current thread's mapped stack, or 0 if it cannot be read.
///
/// `setrlimit(RLIMIT_STACK)` succeeding does not mean the stack can grow:
/// a container, systemd `LimitSTACK`, or a VM can leave the mapping at
/// the process default (~8 MiB). Announcing a 20 MiB byte budget against
/// that mapping puts the RecursionError guard past the guard page.
#[cfg(target_os = "linux")]
fn mapped_thread_stack_size() -> usize {
    unsafe {
        let mut attr: libc::pthread_attr_t = std::mem::zeroed();
        if libc::pthread_getattr_np(libc::pthread_self(), &mut attr) != 0 {
            return 0;
        }
        let mut addr = std::ptr::null_mut();
        let mut size = 0usize;
        let rc = libc::pthread_attr_getstack(&attr, &mut addr, &mut size);
        libc::pthread_attr_destroy(&mut attr);
        if rc == 0 { size } else { 0 }
    }
}

pub fn main_entry(binary_name: &'static str) {
    configure_root_only_jit_stats();
    // The sandboxed child runs single-threaded like pypy-c-sandbox: it neither
    // manages signals (the controller does) nor spawns a worker thread, so it
    // issues no sigprocmask/clone syscalls. Run `real_main` directly.
    //
    // Skipping the dedicated interpreter thread does not lose stack-overflow
    // protection: the recursion limit is a native byte budget (~3.75 MiB at the
    // default 5000), not a frame counter, and `stack_check` raises RecursionError
    // before the budget is exceeded — well under the main thread's default ~8 MiB
    // stack. A child that raises the limit far past that, or shrinks the stack
    // via ulimit, gets a SIGSEGV reaped by the trusted controller, not an escape.
    #[cfg(feature = "sandbox")]
    {
        real_main(binary_name);
        post_run_diagnostics();
    }

    // `pypy_main_function` runs the interpreter on the thread that started the
    // process, and a tracer attached to that process follows that thread alone
    // unless it is asked to follow clones — so an interpreter on a thread of
    // its own makes the program's whole syscall stream invisible to `strace`.
    // Staying there needs a stack far past the ~8 MiB a process starts with.
    // Linux grows that stack on demand against the soft `RLIMIT_STACK`, so
    // raising the limit here is enough. Darwin cannot: `setrlimit` moves the
    // limit without moving the guard the runtime installed from the size the
    // process started with, and the stack still faults at 8 MiB. The linker
    // sizes the origin thread instead (`-Wl,-stack_size` in build.rs); if
    // that grant is large enough the interpreter stays on this thread, and a
    // host that links or loads without it falls back to a spawned thread.
    #[cfg(all(target_os = "linux", not(feature = "sandbox")))]
    {
        // Announce the stack this thread is budgeted against, and capture the
        // base here at the outermost interpreter entry — the same first
        // statement `_thread`'s worker runs (`module/thread/mod.rs`).
        //
        // The announced figure is the shared one, not the stack actually
        // granted: the budget it sizes lives in a process-global word every
        // thread's inline probe reads, so one figure has to describe every
        // Python-running thread. Announcing it takes a stack at least that
        // large — against a smaller one the guard would sit past the guard
        // page and the interpreter would fault where it should raise
        // RecursionError — so a host that grants less hands the question back
        // to `getrlimit`.
        let granted =
            pyre_interpreter::stack_check::raise_main_thread_stack_limit(INTERPRETER_STACK_SIZE);
        let mapped = mapped_thread_stack_size();
        let usable = if mapped == 0 {
            granted
        } else {
            granted.min(mapped)
        };
        // Capture the inherited mask before either path so `real_main`'s
        // unblock restores signals the caller had blocked, instead of
        // opening SIGINT/SIGALRM unconditionally.
        pyre_interpreter::module::signal::signalstate::block_async_signals_on_origin_thread();
        if usable >= pyre_interpreter::stack_check::DEFAULT_RUNTIME_THREAD_STACK_SIZE {
            pyre_interpreter::stack_check::configure_current_thread_stack_size(
                pyre_interpreter::stack_check::DEFAULT_RUNTIME_THREAD_STACK_SIZE,
            );
            real_main(binary_name);
            post_run_diagnostics();
        } else {
            // The soft limit went up but the mapping did not. Same situation
            // as darwin: reserve a thread whose stack is actually that large,
            // and route process-directed signals onto that thread.
            std::thread::Builder::new()
                .stack_size(INTERPRETER_THREAD_STACK_SIZE)
                .spawn(move || {
                    pyre_interpreter::stack_check::configure_current_thread_stack_size(
                        pyre_interpreter::stack_check::DEFAULT_RUNTIME_THREAD_STACK_SIZE,
                    );
                    real_main(binary_name);
                    post_run_diagnostics();
                })
                .expect("spawn interpreter thread")
                .join()
                .unwrap();
        }
    }

    #[cfg(not(any(target_os = "linux", feature = "sandbox")))]
    {
        // Capture the inherited mask before either path so `real_main`'s
        // unblock restores signals the caller had blocked, instead of
        // opening SIGINT/SIGALRM unconditionally.
        pyre_interpreter::module::signal::signalstate::block_async_signals_on_origin_thread();
        let run_on_origin_thread = {
            #[cfg(target_os = "macos")]
            {
                // `getrlimit(RLIMIT_STACK)` does not follow `-Wl,-stack_size`.
                let granted = unsafe { libc::pthread_get_stacksize_np(libc::pthread_self()) };
                granted >= pyre_interpreter::stack_check::DEFAULT_RUNTIME_THREAD_STACK_SIZE
            }
            #[cfg(not(target_os = "macos"))]
            {
                false
            }
        };
        let run = move || {
            // Same first statement `_thread`'s worker runs
            // (`module/thread/mod.rs`): announce the stack this thread is
            // budgeted against, and capture the base here at the outermost
            // interpreter entry. Left out, `effective_stack_length` falls
            // back to `getrlimit(RLIMIT_STACK)`. On darwin that still reports
            // the process default (~8 MiB) even when the linker granted a
            // larger origin-thread stack; on a spawned thread it describes
            // the process's original thread and not this one. Either way the
            // byte guard — not the recursion limit — would decide how deep
            // Python can recurse.
            //
            // The announced figure is the shared one, not
            // INTERPRETER_THREAD_STACK_SIZE: the budget it sizes is stored
            // in a process-global word every thread's inline probe reads.
            pyre_interpreter::stack_check::configure_current_thread_stack_size(
                pyre_interpreter::stack_check::DEFAULT_RUNTIME_THREAD_STACK_SIZE,
            );
            real_main(binary_name);
            post_run_diagnostics();
        };
        if run_on_origin_thread {
            run();
        } else {
            std::thread::Builder::new()
                .stack_size(INTERPRETER_THREAD_STACK_SIZE)
                .spawn(run)
                .expect("spawn interpreter thread")
                .join()
                .unwrap();
        }
    }
}

/// Keep test-runner diagnostics on the pyre process the runner launched.
///
/// `test_asyncio` deliberately starts `sys.executable` children and asserts
/// their stdout/stderr protocol byte-for-byte. The CPython-suite runner still
/// needs `MAJIT_STATS` from its direct child to detect a silently recovered JIT
/// compile panic, but passing that variable through to grandchildren changes
/// the Python program's observable subprocess output. The runner opts into
/// this process-tree policy with `PYRE_MAJIT_STATS_ROOT_ONLY`; the inherited
/// marker distinguishes its direct child from later pyre descendants.
static PRINT_ROOT_ONLY_JIT_STATS: OnceLock<bool> = OnceLock::new();

fn configure_root_only_jit_stats() {
    let root_only = std::env::var_os("PYRE_MAJIT_STATS_ROOT_ONLY").is_some();
    let descendant = std::env::var_os("PYRE_MAJIT_STATS_ANCESTOR").is_some();
    let print_here = !root_only || !descendant;
    PRINT_ROOT_ONLY_JIT_STATS
        .set(print_here)
        .expect("JIT stats process policy configured twice");
    if root_only && !descendant {
        // main_entry runs before pyre starts any thread, so no other thread
        // can concurrently read or mutate the process environment.
        unsafe { std::env::set_var("PYRE_MAJIT_STATS_ANCESTOR", "1") };
    }
    if !print_here {
        // `majit_trace::Logger` prints its own `=== JIT Statistics ===` block
        // from `Drop`, and it reads the environment rather than this policy.
        // Clearing what turns it on is how the policy reaches it: a descendant
        // then writes nothing of pyre's own to stderr, which is what
        // `test_large_pool` asserts of the script it runs through
        // `assert_python_ok`.
        unsafe {
            std::env::remove_var("MAJIT_STATS");
            std::env::remove_var("MAJIT_LOG");
        }
    }
}

/// Diagnostics that have to read the descr universe *after* it has finished
/// growing, so they run once `real_main` returns rather than inside it. Each is
/// gated on its own environment variable internally.
///
/// The counts these name are already on the `[jit-stats]` line, which
/// `maybe_print_jit_stats` emits from `real_main`'s exit paths and so survives
/// a `process::exit` this hook does not reach. These print *which* members,
/// never whether the gate holds.
fn post_run_diagnostics() {
    if std::env::var_os("PYRE_FIELD_IDENTITY_CENSUS").is_some() {
        pyre_jit::field_descr_identity_census_now();
    }
    pyre_jit::descr_spelling_gate_recheck_now();
}

fn real_main(binary_name: &str) {
    // Ahead of everything else: the hash secret is fixed by the first digest
    // taken in the process, so `PYTHONHASHSEED` has to be resolved before any
    // Python object exists. `-E` does not reach it — the seed is settled before
    // command-line flags are, so an ignored environment still supplies it.
    if let Err(detail) = pyre_interpreter::builtins::init_hash_secret_from_env() {
        eprintln!("{detail}");
        std::process::exit(1);
    }
    // Receive process-directed async signals on this thread (see
    // `main_entry`) so blocking syscalls here are interrupted by Ctrl-C /
    // alarms.  Hosts that stay on the original thread already receive them
    // here; `unblock` is then a no-op.  The sandboxed child does not touch
    // the signal mask at all.
    #[cfg(not(feature = "sandbox"))]
    pyre_interpreter::module::signal::signalstate::unblock_async_signals_on_interp_thread();
    // Suppress panic messages for the optimizer's silent control-flow panics
    // (InvalidLoop, SpeculativeError) — these are caught by catch_unwind in
    // the JIT compile paths but the default panic hook still prints to stderr,
    // making a graceful trace-abandon look like a crash.
    // RPython: both are silent exceptions, not errors
    // (`unroll.py:119-123 except SpeculativeError: raise InvalidLoop`). The
    // optimizer raises SpeculativeError to decline a speculative heap fold;
    // compile_loop / compile_bridge catch it and abandon the trace (correct
    // blackhole fallback), so its message is a false crash signal. A genuinely
    // uncaught one still aborts with a nonzero exit, so suppressing only the
    // message does not hide a real failure.
    // `rsandbox.reraise_error` raises an ordinary exception; the trampoline
    // unwinds with `SandboxError`, and `catch_sandbox_stub` turns it into
    // the Python error, so its panic message is not a crash either.
    #[cfg(unix)]
    fn is_sandbox_error(payload: &(dyn std::any::Any + Send)) -> bool {
        payload
            .downcast_ref::<majit_rlib::rsandbox::SandboxError>()
            .is_some()
    }
    #[cfg(not(unix))]
    fn is_sandbox_error(_payload: &(dyn std::any::Any + Send)) -> bool {
        false
    }
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let payload = info.payload();
        let is_silent_control_flow = payload
            .downcast_ref::<majit_metainterp::optimize::InvalidLoop>()
            .is_some()
            || payload
                .downcast_ref::<majit_metainterp::optimize::SpeculativeError>()
                .is_some()
            || payload
                .downcast_ref::<pyre_interpreter::runtime_ops::SandboxStub>()
                .is_some()
            || is_sandbox_error(payload);
        if !is_silent_control_flow {
            // The suite runner keeps a signal death's last stderr line, and
            // the default hook's trailer is that line. Put the location on
            // its own line first so a CI log can name the panic.
            let loc = info
                .location()
                .map(|location| location.to_string())
                .unwrap_or_else(|| "<unknown>".to_string());
            eprintln!("panicked at {loc}: {info}");
            default_hook(info);
        }
    }));
    // pypy/interpreter/app_main.py `entry_point`: preserve the executable and
    // every launcher argument reaching app level before command-line parsing
    // rewrites them into the run mode and `sys.argv`. Only the interp-level
    // option below is missing from that list, exactly as `sys.orig_argv[:] =
    // [executable] + argv` records the list `entry_point` handed on.
    let (argv, heapsize) = take_heapsize_option(binary_name, std::env::args_os().collect());
    importing::set_sys_orig_argv(argv.clone());
    let (mode, mut flags, args) = match parse_args(binary_name, argv) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{binary_name}: {e}");
            std::process::exit(2);
        }
    };
    // A `._pth` beside the executable names every `sys.path` entry itself, so
    // it also isolates the interpreter from the environment and skips `site`.
    // Applied after the option parse, which is where the variables it
    // overrides have already been folded in.
    importing::apply_pth_config(&mut flags);
    let LaunchFlags {
        interactive,
        quiet,
        no_site,
        ..
    } = flags;

    // The `interact` controller does not run the embedded interpreter; it only
    // spawns and relays for an untrusted child, so it skips the interpreter
    // bootstrap (recursion budget, JIT hooks) entirely.
    let is_interact = matches!(mode, RunMode::Interact { .. });

    // The interactive REPL drives raw stdin/stdout directly, which under sandbox
    // are the controller's marshalling pipes — running it would corrupt the
    // protocol and bypass fd-0 mediation. Reject the prompt / `-i` in sandbox
    // builds; a piped stdin program reads through the seam and is fine.
    #[cfg(feature = "sandbox")]
    if !is_interact
        && (flags.inspect || (matches!(mode, RunMode::Stdin { .. }) && stdin_is_interactive(false)))
    {
        eprintln!("{binary_name}: interactive mode is unavailable in the sandbox");
        std::process::exit(2);
    }

    // The module hooks must be installed before the collector is built:
    // `init_jit_hooks` / `build_gc` numbers their GC classes.
    #[cfg(feature = "pyre-module")]
    pyre_module::register();

    if !is_interact {
        // Eagerly install pyre-jit's hooks into pyre-interpreter so that
        // sys.settrace / set_jit_param routing is live from the very first
        // user statement, not only after the first JIT-traced bytecode.
        pyre_jit::eval::init_jit_hooks();
        // `entry_point` sets the limit ahead of `space.startup()`; the earliest
        // equivalent here is the line above, which is what builds the
        // collector. Nothing has allocated yet either way, so the thresholds
        // `set_max_heap_size` pulls down are still their startup values.
        // `PYRE_MAX_MEMORY` is not applied here: a non-zero `max_heap_size`
        // lowers the major-collection threshold (`set_max_heap_size`) even
        // when the heap is far below it. The process charge is the ceiling.
        if let Some(size) = heapsize {
            let _entry = majit_gc::gc_sync::enter_external_callback();
            majit_gc::gc_set_max_heap_size(size);
        }
    }

    // Record `-S` before the first `import sys` so `sys.flags.no_site`
    // reflects whether site initialization was skipped.
    importing::set_no_site(no_site);
    importing::set_runtime_flags(&flags);

    // OS-level hardening (default-on in any Linux `sandbox` build): lock the
    // sandboxed child to a host-neutral syscall allowlist so any un-mediated
    // syscall — a missed reroute, the linked host_env crate, or one reached by a
    // memory-safety exploit — is denied by the kernel (the analog of RPython's
    // os_level_sandboxing). The compile-out seam is the primary mechanism; this
    // kernel allowlist is the defense-in-depth backstop, always installed here so
    // it cannot be forgotten. The `interact` controller keeps full syscall access
    // (it spawns and relays for the child), so it is excluded. Set
    // PYRE_SANDBOX_NO_SECCOMP to bypass when debugging the allowlist.
    #[cfg(all(target_os = "linux", feature = "sandbox"))]
    if !is_interact && std::env::var_os("PYRE_SANDBOX_NO_SECCOMP").is_none() {
        // The GC is built lazily on the first allocation, and its Linux
        // construction reads `/proc/meminfo` (`get_total_memory`) — a direct
        // `openat` the filter below traps. Force the per-thread JIT driver, and
        // with it `MiniMarkGC::new`, to run now, while file opens are still
        // allowed; the post-lockdown allocations reuse the constructed GC and
        // issue no `openat`. Same warm-up-before-filter shape as the `gmtime_r`
        // tz-cache prime in `install_runtime_filter`.
        pyre_jit::eval::driver_pair();
        if let Err(e) = pyre_sandbox::seccomp::install_runtime_filter() {
            eprintln!("{binary_name}: failed to install sandbox seccomp filter: {e}");
            std::process::exit(1);
        }
    }

    match mode {
        RunMode::Command(cmd) => {
            // `_PyPathConfig_ComputeSysPath0` yields "" for a `-c` argv[0]. The
            // cwd is still the anchor the shadowing check compares module
            // origins against.
            let cwd = sys_path_cwd();
            importing::init_sys_path(&cwd, std::ffi::OsStr::new(""));
            let mut argv = vec![std::ffi::OsString::from("-c")];
            argv.extend(args);
            importing::set_sys_argv(&argv);
            let cmd = dedent_command(&cmd);
            let session = run_source(cmd.as_ref(), Mode::Exec, "<string>", no_site, true);
            if session.is_some() {
                repl::run_repl(true, no_site, session);
            }
        }
        RunMode::Module(module) => {
            // `-m`: sys.path[0] is the cwd, and argv[0] is `-m` while the
            // module's packages import (`parse_command_line` returns
            // `['-m'] + args`); `_run_module_as_main` then sets it to the
            // module's resolved origin.
            let cwd = sys_path_cwd();
            importing::init_sys_path(&cwd, cwd.as_os_str());
            let mut argv = vec![std::ffi::OsString::from("-m")];
            argv.extend(args);
            importing::set_sys_argv(&argv);
            let session = run_module(&module, no_site, true);
            if session.is_some() {
                repl::run_repl(true, no_site, session);
            }
        }
        RunMode::Script(path) => {
            let script_dir = script_startup_dir(&path);
            importing::init_sys_path(&script_dir, script_dir.as_os_str());
            // sys.argv[0] is the script path; remaining values go to argv[1:].
            let mut argv = vec![std::ffi::OsString::from(&path)];
            argv.extend(args);
            importing::set_sys_argv(&argv);
            let compile_path = script_compile_path(&path);
            let session = run_script_path(&compile_path, no_site, binary_name, true);
            if session.is_some() {
                repl::run_repl(true, no_site, session);
            }
        }
        RunMode::Stdin { argv0 } => {
            // The prompt and piped stdin both take "" for `sys.path[0]`
            // (`_PyPathConfig_ComputeSysPath0` on an argv[0] of "" / "-"); the
            // cwd stays the shadowing-check anchor.
            let cwd = sys_path_cwd();
            importing::init_sys_path(&cwd, std::ffi::OsStr::new(""));
            // `sys.argv` is `['']` with no script argument and `['-', …]` for
            // an explicit dash.
            let mut argv = vec![std::ffi::OsString::from(argv0)];
            argv.extend(args);
            importing::set_sys_argv(&argv);
            if stdin_is_interactive(interactive) {
                // A tty (or `-i`) prints the banner unless `-q` and drops into
                // the prompt.
                repl::run_repl(quiet, no_site, None);
            } else {
                // Otherwise stdin is read to EOF and executed as a single
                // `exec` unit named `<stdin>`.
                let source = match {
                    #[cfg(feature = "sandbox")]
                    {
                        read_stdin_source_catching_sandbox()
                    }
                    #[cfg(not(feature = "sandbox"))]
                    {
                        read_stdin_source()
                    }
                } {
                    Ok(source) => source,
                    Err(e) => {
                        eprintln!("{binary_name}: cannot read stdin: {e}");
                        std::process::exit(1);
                    }
                };
                run_source(&source, Mode::Exec, "<stdin>", no_site, false);
            }
        }
        RunMode::Interact {
            exe,
            args,
            tmpdir,
            lib_root,
            timeout,
            allow_net,
            log_file,
            verbose,
        } => {
            #[cfg(unix)]
            run_interact(
                binary_name,
                exe,
                args,
                tmpdir,
                lib_root,
                timeout,
                allow_net,
                log_file,
                verbose,
            );
            #[cfg(not(unix))]
            {
                let _ = (
                    exe, args, tmpdir, lib_root, timeout, allow_net, log_file, verbose,
                );
                eprintln!(
                    "{binary_name}: 'interact' (sandbox controller) is only supported on Unix"
                );
                std::process::exit(1);
            }
        }
    }
}

/// `pyre interact` — drive the trusted sandbox controller over the untrusted
/// `exe`, then exit with the child's return code. Port of
/// `pypy/sandbox/pypy_interact.py`'s `main`. Unix-only: the controller relies
/// on a fork/fd-based child, so non-unix hosts reject `interact` at dispatch.
#[cfg(unix)]
// Keep the command-line options explicit at the sandbox-controller boundary;
// grouping them would only duplicate the CLI parser's argument structure.
#[allow(clippy::too_many_arguments)]
fn run_interact(
    binary_name: &str,
    exe: String,
    args: Vec<String>,
    tmpdir: Option<String>,
    lib_root: Option<String>,
    timeout: Option<f64>,
    allow_net: bool,
    log_file: Option<String>,
    _verbose: bool,
) {
    use pyre_sandbox::controller::PyPySandboxedProc;

    let tmpdir = tmpdir.map(std::path::PathBuf::from);
    let lib_root = lib_root.map(std::path::PathBuf::from);
    let timeout = timeout.map(std::time::Duration::from_secs_f64);
    let log_file = log_file.map(std::path::PathBuf::from);

    let mut proc =
        match PyPySandboxedProc::new(&exe, &args, tmpdir, lib_root, timeout, allow_net, log_file) {
            Ok(proc) => proc,
            Err(e) => {
                eprintln!("{binary_name}: cannot start sandbox '{exe}': {e}");
                std::process::exit(1);
            }
        };
    match proc.interact() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("{binary_name}: sandbox controller error: {e}");
            std::process::exit(1);
        }
    }
}

/// Print a one-line JIT statistics summary to stderr when `MAJIT_STATS` is
/// set. `internal_compile_panics > 0` means an internal JIT bug silently
/// disabled compilation for some traces (graceful degradation in release);
/// `check.py` asserts it stays 0. Must be called before any `process::exit`
/// since exits skip destructors.
fn maybe_print_jit_stats() {
    if !PRINT_ROOT_ONLY_JIT_STATS.get().copied().unwrap_or(true) {
        return;
    }
    // `WarmRunnerDesc.add_finish`: `if profiler.initialized: profiler.finish()`.
    // `Profiler.print_stats` opens `debug_start("jit-summary")` and emits
    // the body only while that section's ready bit is set, so the
    // `MAJIT_LOG` filter applies inside `finish`. The driver is borrowed
    // only when this thread already built it; constructing one here would
    // run `init_gc_subsystem` on a run that never entered the JIT.
    if let Some(pair) = pyre_jit::eval::existing_driver_pair() {
        let profiler = &pair.0.meta_interp().staticdata.profiler;
        if profiler
            .initialized
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            profiler.finish();
        }
    }
    maybe_print_mc_diag();
    maybe_print_gc_diag();
    maybe_print_spec_census();
    maybe_print_cell_census();
    maybe_print_fbw_depth_census();
    if std::env::var_os("MAJIT_STATS").is_none() {
        return;
    }
    // Gate tallies for the trace-entry and guard→bridge paths. These were
    // reachable only through the wasm runner's `pyre_jit_mc_diag` export, so a
    // native run had no way to see which gate declined a trace.
    //
    // `all_descrs` rides along because it is the one table the optimizer grows
    // per trace while `descr.py:47` asserts it stays under 2**15 — upstream
    // fills it once at translation time, so only pyre can run into that bound.
    //
    // Printed BEFORE the structural summary, which is where it used to have to
    // be: `check.py` `_jit_stats_snapshot` once kept only the LAST
    // `[jit-stats]` line, so a diag line emitted after the counters displaced
    // them and the loops_aborted / internal_compile_panics floor silently
    // stopped firing. That snapshot now merges every line, so the order is no
    // longer load-bearing — but the reading stays grouped diag-then-summary.
    // The wasm runner orders its diag lines the same way.
    let all_descrs_len = pyre_jit::eval::driver_pair()
        .0
        .meta_interp()
        .staticdata
        .all_descrs()
        .lock()
        .len();
    eprintln!(
        "[jit-stats] mc_diag {} all_descrs={all_descrs_len}",
        majit_metainterp::mc_diag_summary()
    );
    // The portal's own decisions, joined against the labels declared beside
    // the counters (`pyre_jit::eval::PORTAL_DIAG_LABELS`). Read from the same
    // reader the wasm export calls, so the two backends' lines diff field by
    // field. This is the only portal instrument that works on wasm at all: the
    // rest are `std::env::var` reads, and the guest has no environment.
    {
        let portal: Vec<String> = pyre_jit::eval::PORTAL_DIAG_LABELS
            .iter()
            .enumerate()
            .map(|(i, lbl)| format!("{lbl}={}", pyre_jit::eval::portal_diag(i)))
            .collect();
        eprintln!("[jit-stats] portal_diag {}", portal.join(" "));
    }
    // How many of the rehydrated descr pool a run actually names, under
    // `PYRE_DESCR_DEMAND`. Off by default; `(0, pool)` means the probe is off.
    let (descr_demanded, descr_pool) = pyre_jit::descr_demand_summary();
    if descr_demanded != 0 {
        eprintln!("[jit-stats] descr_demand touched={descr_demanded} pool={descr_pool}");
    }
    // Every full-body-walk tally, joined against the names declared beside the
    // counters (`pyre_jit_trace::trace::fbw_diag::LABELS`). ALL of them, in
    // slot order: this reader and the wasm runner used to print two disjoint
    // subsets, so five slots were bumped on native and printed only by the wasm
    // host, and five more the other way round — a counter nobody can read on
    // the backend that bumps it cannot answer the question it was added for.
    // The runner prints the same keys in the same order, so the two lines diff
    // field by field.
    //
    // What the individual slots mean is documented at the declaration; the
    // gated ones are why this line may not lose a key silently:
    //
    // * `fbw_rolled_back_with_effects` — walks that ended uncommitted after a
    //   residual had already run an effect no journal undoes (it wrote live
    //   heap or entered a Python frame), so the replay the caller falls back to
    //   applies that effect twice. A nonzero value names a walk-abort variant
    //   that reaches `run_perfn_walk`'s epilogue without an adopt leg.
    // * `fbw_store_journal_rollback_failed` — the hole subtraction would
    //   otherwise open: a journaled store the rollback could not restore is
    //   subtracted out of the count above, so it needs one of its own.
    // * the two adoption tallies are the same population read from the other
    //   end: the walks that DID hand the interpreter a resumable image. Nothing
    //   else counts them, so a change that silently sends a shape back to the
    //   legacy replay moves no gated counter without these.
    let mut fbw = String::from("[jit-stats] fbw_diag");
    for (slot, key) in pyre_jit::FBW_DIAG_LABELS.iter().enumerate() {
        fbw.push_str(&format!(" {key}={}", pyre_jit::fbw_diag_counter(slot)));
    }
    eprintln!("{fbw}");
    let stats = pyre_jit::eval::driver_pair().0.get_stats();
    eprintln!(
        "[jit-stats] loops_compiled={} bridges_compiled={} retraces_compiled={} loops_aborted={} \
         guard_failures={} back_edge_polls={} internal_compile_panics={}",
        stats.loops_compiled,
        stats.bridges_compiled,
        stats.retraces_compiled,
        stats.loops_aborted,
        stats.guard_failures,
        stats.back_edge_polls,
        stats.internal_compile_panics,
    );
    // How those `guard_failures` are distributed: a handful of guards eating
    // most of them means the deopt keeps returning to the runtime for a guard
    // that should already have a bridge, while a wide flat spread means many
    // guards each below `trace_eagerness`. Off unless `MAJIT_GUARD_CENSUS`.
    if std::env::var_os("MAJIT_GUARD_CENSUS").is_some() {
        eprintln!("[jit-stats] {}", majit_metainterp::guard_census_summary(12));
    }
    // What the minor collections those deopts drive actually reclaim. Off
    // unless `MAJIT_GC_DRAIN_CENSUS`.
    if majit_gc::drain_census_enabled() {
        eprintln!("[jit-stats] {}", majit_gc::drain_census_summary());
    }
    // The descr-universe invariants. `descr_set_stale_absent` re-asks the
    // `AbsentContainer` question against the universe as it stands now, so it
    // has to be read here, after the run published everything it is going to.
    eprintln!("[jit-stats] {}", pyre_jit::descr_set_jit_stats());
    // Whether those slots were filled CORRECTLY, which the line above
    // cannot say. See `field_position_jit_stats`.
    eprintln!("[jit-stats] {}", pyre_jit::field_position_jit_stats());
    // Which mints those `field_pos_unresolved` are. Off unless
    // `MAJIT_FIELD_POS_UNRESOLVED`, and it takes `all_descrs_len` because the
    // table needs a denominator that is not one of its own numbers: a run that
    // never loaded the descr universe records nothing, and so does a tree with
    // nothing wrong.
    for line in pyre_jit::field_position_unresolved_report(all_descrs_len) {
        eprintln!("[jit-stats] {line}");
    }
}

/// Names for the `MC_DIAG` tallies, in index order — the array
/// `majit_metainterp` declares beside the counters themselves, so a slot cannot
/// be added there and go unnamed here. The wasm runner prints the same names for
/// the `pyre_jit_mc_diag` export, making a native run and a wasm run of the same
/// program diffable field by field.
use majit_metainterp::MC_DIAG_LABELS as MC_DIAG_FIELDS;

/// Print the GC's deterministic pressure counters.
///
/// The perf question this answers is the one a loaded machine's clock cannot:
/// "did that change actually allocate less?".  A minor collection happens once
/// per nursery-full, so `minor` is a direct proxy for cumulative allocation and
/// is identical across runs of the same program — unlike wall or even user CPU
/// time, which on this tree can swing by more than any single change is worth.
///
/// Same prefix discipline as [`maybe_print_mc_diag`]: not `[jit-stats]`, and
/// behind its own env gate, so the committed per-bench baselines never see it.
fn maybe_print_gc_diag() {
    if std::env::var_os("PYRE_GC_DIAG").is_none() {
        return;
    }
    let (minor, major, heap_bytes, nursery_bytes) = pyre_jit::gc_diag_counters();
    eprintln!(
        "[jit-gc-diag] minor={minor} major={major} heap_bytes={heap_bytes} \
         nursery_bytes={nursery_bytes}"
    );
}

/// Deliberately not under the `[jit-stats]` prefix, following
/// [`maybe_print_mc_diag`] and [`maybe_print_gc_diag`]: `check.py`'s
/// `_jit_stats_snapshot` merges every such line, so the specialization census
/// uses its own prefix.
fn maybe_print_spec_census() {
    if std::env::var_os("PYRE_FBW_SPEC_CENSUS").is_none() {
        return;
    }
    eprint!("{}", pyre_jit::spec_census_summary());
}

fn maybe_print_fbw_depth_census() {
    if std::env::var_os("PYRE_FBW_DEPTH_CENSUS").is_none() {
        return;
    }
    eprint!("{}", pyre_jit::fbw_depth_census_summary());
}

/// Print the guard-failure → bridge-trace gate tallies. Until this existed the
/// counters were bumped on every backend but only readable through the wasm
/// guest export, so the question "did this guard failure ever reach a bridge,
/// or was it short-circuited by the decline blacklist" could not be asked of a
/// native run at all.
///
/// Deliberately NOT under the `[jit-stats]` prefix: `check.py`'s
/// `_jit_stats_snapshot` merges every such line into one key/value set, so a
/// key printed under that prefix joins the counters instead of sitting beside
/// them. Its own env gate keeps it off the default `check.py` run, which sets
/// `MAJIT_STATS` for every bench.
fn maybe_print_mc_diag() {
    if std::env::var_os("PYRE_MC_DIAG").is_none() {
        return;
    }
    let mut line = String::from("[jit-mc-diag]");
    for (i, name) in MC_DIAG_FIELDS.iter().enumerate() {
        line.push_str(&format!(" {name}={}", majit_metainterp::mc_diag(i)));
    }
    eprintln!("{line}");
}

/// Print the JIT cell table's occupancy at exit.
///
/// `WarmEnterState.cells` is an `IndexMap` keyed by the whole green-key hash,
/// and nothing production-reachable removes an entry from it: `install_new_cell`
/// prunes only within one bucket, which needs a hash collision, and `gc_cells` —
/// the whole-table sweep — has `test_gc_cells` as its only caller. So the
/// table's size is a function of the workload rather than of any bound, and the
/// number nobody had was how fast it grows on a real program.
///
/// Read it as a ladder: the level alone is a workload fact, the slope across
/// runs of increasing size is the rate. `loops_compiled` on the `[jit-stats]`
/// line is the arming witness — a run that compiled nothing exercised no cell
/// mint, and its `cells` says nothing about growth.
///
/// `pinned_refs` counts the GC owner-root slots held by stored Ref greens.
/// Pyre registers `majit_ir::set_ref_resolver` during JIT bootstrap, matching
/// the ordinary traced fields `JitCell.__init__` stores upstream; a nonzero
/// value therefore reports the live code/string objects the cell table owns.
///
/// Same prefix discipline as [`maybe_print_gc_diag`]: not `[jit-stats]`, and
/// behind its own env gate, so the committed per-bench baselines never see it.
/// Keep it off that line: `cells` depends on workload and host state, while
/// `check.py` gates jit-stats rows on exact equality. A counter
/// earns a gated row by being a function of the program alone.
fn maybe_print_cell_census() {
    if std::env::var_os("PYRE_CELL_CENSUS").is_none() {
        return;
    }
    let stats = pyre_jit::eval::driver_pair()
        .0
        .meta_interp()
        .warm_state_ref_for_driver(pyre_jit::PyreJitState::PYPYJIT_JD_INDEX)
        .expect("pypyjit warmstate")
        .get_stats();
    // Unconditional once the gate is set, including on a zero: a missing line
    // then means the probe is not in the binary, which is a different fact from
    // an empty table and must not print the same bytes.
    eprintln!(
        "[jit-cell-census] cells={} compiled={} tracing={} invalidated={} \
         dont_trace_here={} pinned_refs={}",
        stats.num_cells,
        stats.num_compiled,
        stats.num_tracing,
        stats.num_invalidated,
        stats.num_disable_noninlinable_function,
        stats.num_pinned_refs,
    );
}

/// Shared top-level launcher bootstrap for `run_source` and `run_module`:
/// register `__build_class__`, create the process `ExecutionContext`, seed
/// the `LAST_EXEC_CTX` TLS slot so
/// `space.getexecutioncontext()` (sys.settrace/getframe) resolves from the
/// first user statement, and install SIGINT handling (app_main.py:926).
fn setup_exec_context() -> Rc<PyExecutionContext> {
    // Register __build_class__ callback (PyPy: setup_builtin_modules)
    register_build_class();
    let execution_context = Rc::new(PyExecutionContext::default());
    set_last_exec_ctx(Rc::as_ptr(&execution_context));
    unsafe {
        let ec_ptr = Rc::as_ptr(&execution_context) as *mut PyExecutionContext;
        (*ec_ptr).install_user_del_action();
        pyre_interpreter::module::signal::interp_signal::install_signal_handling(&mut *ec_ptr);
    }
    execution_context
}

/// Resolve a pending `SystemExit`'s status, then finalize and exit with it.
///
/// The order matters: `app_main.py handle_sys_exit` runs at
/// application level, i.e. *before* `targetpypystandalone.py:88
/// finally: space.finish()`. Resolving `e.code` is an ordinary attribute
/// lookup that walks the exception's type and that type's dict, and
/// `finalize_runtime` collects — pinning the three raw `PyObjectRef` fields
/// of the Rust `PyError` (which, unlike PyPy's GC-visible `OperationError`,
/// the collector cannot see) does not keep that type's dict alive. Reading
/// the code after the collection spun forever inside the type dict's
/// `IndexMap` probe for a user-defined `SystemExit` subclass.
fn finalize_system_exit(
    error: pyre_interpreter::PyError,
    canonical: pyre_object::PyObjectRef,
    ec_ptr: *const PyExecutionContext,
) -> ! {
    let code = pyre_interpreter::system_exit_code(&error);
    finalize_runtime(canonical, ec_ptr);
    maybe_print_jit_stats();
    std::process::exit(code);
}

/// Die by SIGINT for an uncaught KeyboardInterrupt after shutdown has run
/// (`app_main.py:1133-1153`).
/// Whether a top-level `KeyboardInterrupt` has been reported and not answered
/// by an explicit exit status yet.
///
/// `_PyErr_PrintEx` records one, and `Py_RunMain` consults it after the prompt
/// and after `Py_FinalizeEx`, overwriting any `exitcode` the prompt produced.
/// An interrupt raised by the program under `-i` still ends the process by
/// SIGINT once the prompt is done -- and so does one raised at the prompt
/// itself, even when the prompt then calls `exit(n)`.
static UNHANDLED_KEYBOARD_INTERRUPT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub(crate) fn note_unhandled_keyboard_interrupt(error: &pyre_interpreter::PyError) {
    if is_keyboard_interrupt(error) {
        UNHANDLED_KEYBOARD_INTERRUPT.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

pub(crate) fn had_unhandled_keyboard_interrupt() -> bool {
    UNHANDLED_KEYBOARD_INTERRUPT.load(std::sync::atomic::Ordering::Relaxed)
}

fn terminate_by_sigint() -> ! {
    // A Win32 process has no SIGINT to die of: restoring `SIG_DFL` and calling
    // `raise(SIGINT)` runs the CRT's default action, which returns and ends the
    // process with status 3 rather than the interrupt a shell reads. The status
    // that reads as one is `STATUS_CONTROL_C_EXIT`, so it is exited with
    // directly. Both callers have already finalized and flushed.
    #[cfg(windows)]
    {
        const STATUS_CONTROL_C_EXIT: u32 = 0xC000_013A;
        std::process::exit(STATUS_CONTROL_C_EXIT as i32);
    }
    #[cfg(not(windows))]
    {
        unsafe {
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            if libc::kill(libc::getpid(), libc::SIGINT) != 0 {
                std::process::exit(1);
            }
        }
        std::process::abort();
    }
}

/// [`pyre_interpreter::app_main::import_site`], ending the process with status
/// 1 when it reports that startup cannot continue.
pub(crate) fn import_site(
    no_site: bool,
    w_main_globals: pyre_object::PyObjectRef,
    ec_ptr: *const PyExecutionContext,
) {
    if !pyre_interpreter::app_main::import_site(no_site, w_main_globals, ec_ptr) {
        std::process::exit(1);
    }
}

fn is_keyboard_interrupt(error: &pyre_interpreter::PyError) -> bool {
    raised_is_instance_of(error, "KeyboardInterrupt")
}

/// Run a library module as `__main__` via `runpy._run_module_as_main`,
/// the `-m` entry point. `vm.run_module` analog.
fn run_module(module: &str, no_site: bool, run_code: bool) -> Option<MainSession> {
    let execution_context = setup_exec_context();
    let ec_ptr = Rc::as_ptr(&execution_context);

    // `_run_module_as_main` reads `sys.modules["__main__"].__dict__` and runs
    // the module's code in it, so a `__main__` module backed by a fresh dict
    // must exist before runpy is imported. Use the canonical W_DictObject so
    // `__main__.__dict__` and `globals()` share one identity (module.py getdict
    // Module.getdict()).
    let w_globals = execution_context.fresh_module_globals();
    let _root = pyre_object::gc_roots::push_roots();
    let w_globals = pyre_object::gc_roots::pin_root(w_globals);
    unsafe {
        pyre_object::w_dict_setitem_str(w_globals, "__name__", pyre_object::w_str_new("__main__"))
    };
    let canonical = w_globals;
    // PyPy `module.py Module` is an ordinary GC object. The managed twin
    // matches that, so detaching `sys.modules` can make the module/dict
    // cycle unreachable. The allocation can collect: the dict is already
    // on `_root`, and the result is pinned before `set_sys_module`.
    let main_module = pyre_object::module::w_module_new_aliasing_dict_managed(
        pyre_object::w_str_new("__main__"),
        canonical,
    );
    let main_module = pyre_object::gc_roots::pin_root(main_module);
    importing::set_sys_module("__main__", main_module);

    let result = (|| -> Result<(), pyre_interpreter::PyError> {
        importing::init_importlib_bootstrap(canonical, ec_ptr)?;
        import_site(no_site, canonical, ec_ptr);
        runpy_run_module_as_main(canonical, ec_ptr, module, true)?;
        Ok(())
    })();
    // Drop the launcher pin on `canonical` before `finalize_modules`
    // collects. `sys.modules` still owns `__main__` until
    // `release_sys_modules_for_shutdown`; `finalize_runtime` re-pins
    // `canonical` only until that detach.
    drop(_root);

    let inspect = prompt_requested(run_code);
    if let Err(e) = result {
        // `_run_module_as_main` ends the same run as a script does, so its
        // failure takes the same exit, and `-i` opens the prompt over it the
        // same way -- `SystemExit` included.
        report_or_end_on_main_error(e, inspect, canonical, ec_ptr);
    }
    finish_or_inspect(inspect, execution_context, canonical, main_module)
}

/// `os.path.abspath` for the run filename, or `None` for the `-c "<string>"`
/// command path, which has no script.
///
/// A relative path resolves against `sys_path_cwd()` — the seam-provided
/// virtual cwd under sandbox — before normalizing, so `std::path::absolute`
/// never consults (and leaks) the trusted process cwd. Off sandbox this is
/// identical to `absolute(filename)`.
fn absolute_script_path(filename: &str) -> Option<String> {
    if filename == "<string>" {
        return None;
    }
    if filename == "<stdin>" {
        // A stdin run records the literal `<stdin>`: there is no path to
        // absolutize.
        return Some(filename.to_string());
    }
    let path = Path::new(filename);
    let to_absolutize = if path.is_absolute() {
        path.to_path_buf()
    } else {
        sys_path_cwd().join(path)
    };
    Some(match std::path::absolute(&to_absolutize) {
        Ok(p) => p.to_string_lossy().into_owned(),
        Err(_) => filename.to_string(),
    })
}

fn script_startup_dir(path: &str) -> PathBuf {
    // Initialize sys.path with the script's directory. Under sandbox,
    // absolutizing would consult the trusted process cwd instead of the
    // controller VFS, so use the virtual path as given.
    #[cfg(feature = "sandbox")]
    {
        Path::new(path)
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf()
    }
    #[cfg(not(feature = "sandbox"))]
    {
        // `parent()` of a bare `x.py` is the empty path;
        // `_PyPathConfig_ComputeSysPath0` absolutizes the script's
        // directory, so resolve the empty case against the cwd rather
        // than leaving a relative `.` on `sys.path`.
        let parent = Path::new(path).parent().unwrap_or(Path::new("."));
        let parent = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        // PyPy `resolvedirof` (`initpath.py`) uses `rpath.rabspath`,
        // whose Windows implementation is `GetFullPathNameW`.  It makes the
        // spelling absolute without resolving junctions or symlinks.  Rust's
        // `canonicalize` is a realpath operation and incorrectly changed a
        // runner-temp junction into its target in `sys.path[0]`.
        std::path::absolute(parent).unwrap_or_else(|_| sys_path_cwd())
    }
}

fn script_compile_path(path: &str) -> String {
    // A script is compiled with the same absolute path exposed as
    // `__main__.__file__`; warnings use `code.co_filename`, so the two must
    // not diverge when argv[0] was relative.
    #[cfg(not(feature = "sandbox"))]
    {
        std::path::absolute(Path::new(path))
            .unwrap_or_else(|_| Path::new(path).to_path_buf())
            .to_string_lossy()
            .into_owned()
    }
    #[cfg(feature = "sandbox")]
    {
        path.to_string()
    }
}

fn path_hook_accepts(
    filename: &str,
    canonical: pyre_object::PyObjectRef,
    ec_ptr: *const PyExecutionContext,
) -> Result<bool, pyre_interpreter::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let sys = importing::importhook(
        rustpython_wtf8::Wtf8::new("sys"),
        canonical,
        pyre_object::PY_NULL,
        0,
        ec_ptr,
    )?;
    let path_hooks = pyre_interpreter::baseobjspace::getattr_str(sys, "path_hooks")?;
    // `for hook in sys.path_hooks` is a plain iteration, so a `sitecustomize`
    // that replaces the list with a tuple or any other iterable still gets its
    // hooks called — one step at a time.  Draining the iterable up front
    // instead would surface an error the loop never reaches: a generator that
    // yields an accepting hook and raises on the step after it breaks out at
    // `app_main.py:1056` before that step is ever taken.
    let iterator = pyre_interpreter::baseobjspace::iter(path_hooks)?;

    let _roots = push_roots();
    // A hook call runs arbitrary Python and can drive a collection, which moves
    // the iterator that has to survive it. A Rust local is not walked, so
    // publish it in a shadow-stack slot and read it back for the next step.
    // Pin the already-live iterator first; the path string is minted after
    // that publish so the mint cannot move an unpublished iterator.
    let iterator_slot = shadow_stack_len();
    let _ = pin_root(iterator);
    let filename_slot = shadow_stack_len();
    let _ = pin_root(pyre_object::w_str_new_managed(filename));
    loop {
        let hook = match pyre_interpreter::baseobjspace::next(shadow_stack_get(iterator_slot)) {
            Ok(hook) => hook,
            // `else: importer = None` — the loop ran to exhaustion, so no hook
            // claimed the path.
            Err(error) if error.kind == PyErrorKind::StopIteration => return Ok(false),
            Err(error) => return Err(error),
        };
        match pyre_interpreter::call::call_function_impl_result(
            hook,
            &[shadow_stack_get(filename_slot)],
        ) {
            // `importer = hook(filename); break` — the first hook that does not
            // raise `ImportError` ends the walk, and `importer is None` then
            // means no importer claimed the path.
            Ok(importer) => return Ok(!unsafe { pyre_object::is_none(importer) }),
            Err(error) if is_import_error(&error) => continue,
            Err(error) => return Err(error),
        }
    }
}

fn runpy_run_module_as_main(
    canonical: pyre_object::PyObjectRef,
    ec_ptr: *const PyExecutionContext,
    module: &str,
    alter_argv: bool,
) -> Result<(), pyre_interpreter::PyError> {
    let runpy = importing::importhook(
        rustpython_wtf8::Wtf8::new("runpy"),
        canonical,
        pyre_object::PY_NULL,
        0,
        ec_ptr,
    )?;
    let func = pyre_interpreter::getattr(runpy, pyre_object::w_str_new("_run_module_as_main"))?;
    let _roots = pyre_object::gc_roots::push_roots();
    let func_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(func);
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(module));
    let args = if alter_argv {
        vec![pyre_object::gc_roots::shadow_stack_get(module_slot)]
    } else {
        vec![
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            pyre_object::w_bool_from(false),
        ]
    };
    let res =
        pyre_interpreter::call_function(pyre_object::gc_roots::shadow_stack_get(func_slot), &args);
    if res.is_null() {
        return Err(
            pyre_interpreter::call::take_call_error().unwrap_or_else(|| {
                pyre_interpreter::PyError::new(
                    pyre_interpreter::PyErrorKind::RuntimeError,
                    "runpy._run_module_as_main returned NULL without an exception",
                )
            }),
        );
    }
    Ok(())
}

fn handle_main_error(
    e: pyre_interpreter::PyError,
    canonical: pyre_object::PyObjectRef,
    ec_ptr: *const PyExecutionContext,
) -> ! {
    if e.kind == PyErrorKind::SystemExit {
        finalize_system_exit(e, canonical, ec_ptr);
    }
    let is_keyboard_interrupt = is_keyboard_interrupt(&e);
    // targetpypystandalone.py:88 `finally: space.finish()` — finalize
    // on every exit path, not only on SystemExit.  Print first: the raw
    // `PyObjectRef` fields of `e` are not GC-visible and
    // `finalize_runtime` collects.
    //
    // `PyErr_Print` is what ends a top-level run upstream, and it reports
    // through `sys.excepthook`; the direct print is the arm it takes when there
    // is no hook to reach.
    let mut e = e;
    if !pyre_interpreter::error::print_exception_via_excepthook(&mut e) {
        pyre_interpreter::eprint_exception(&e, true);
    }
    finalize_runtime(canonical, ec_ptr);
    maybe_print_jit_stats();
    if is_keyboard_interrupt {
        terminate_by_sigint();
    }
    std::process::exit(1);
}

fn finish_main(canonical: pyre_object::PyObjectRef, ec_ptr: *const PyExecutionContext) {
    finalize_runtime(canonical, ec_ptr);
    maybe_print_jit_stats();
}

/// End a startup failure the way `Py_RunMain` does.
///
/// `pymain_run_python` reports the failure and returns a status; `Py_FinalizeEx`
/// then runs regardless, so the `atexit` callbacks, module teardown and stream
/// flush a live interpreter owes are delivered before the status is.  Every
/// caller here is past `import_site`, so `site` and any `sitecustomize` have
/// already had their chance to register one.
fn end_after_startup_failure(
    canonical: pyre_object::PyObjectRef,
    ec_ptr: *const PyExecutionContext,
    status: i32,
) -> ! {
    finish_main(canonical, ec_ptr);
    std::process::exit(status);
}

/// The interpreter a program leaves running for `-i` to continue in.
///
/// `pymain_run_python` runs the program, then the prompt, and reaches
/// `Py_FinalizeEx` only after both, so the two share one interpreter and one
/// `__main__`.  That sharing is the whole of `-i`: the names the program bound
/// are the ones the prompt inspects.  A program that failed is inspected the
/// same way -- an uncaught exception, a source that would not compile, even a
/// `SystemExit` -- because the point of asking for the prompt is to look at
/// what went wrong.
pub(crate) struct MainSession {
    pub(crate) execution_context: Rc<PyExecutionContext>,
    pub(crate) canonical: pyre_object::PyObjectRef,
    pub(crate) main_module: pyre_object::PyObjectRef,
}

/// Finalize here, or hand the live interpreter to the prompt to finalize once
/// it is done with it.
fn finish_or_inspect(
    inspect: bool,
    execution_context: Rc<PyExecutionContext>,
    canonical: pyre_object::PyObjectRef,
    main_module: pyre_object::PyObjectRef,
) -> Option<MainSession> {
    if !inspect {
        finish_main(canonical, Rc::as_ptr(&execution_context));
        return None;
    }
    Some(MainSession {
        execution_context,
        canonical,
        main_module,
    })
}

/// Report a top-level failure the prompt is about to open over.
///
/// `handle_main_error` is the arm that ends the process, and it finalizes on
/// the way out; this one only prints, because the interpreter that failed is
/// the one the prompt continues in.  `SystemExit` takes this path too: `-i`
/// turns the request to exit into a report, which is why the run ends with the
/// prompt's status rather than the program's.
fn report_main_error_for_inspect(mut e: pyre_interpreter::PyError) {
    note_unhandled_keyboard_interrupt(&e);
    if !pyre_interpreter::error::print_exception_via_excepthook(&mut e) {
        pyre_interpreter::eprint_exception(&e, true);
    }
}

/// Report a top-level failure, and end the process unless `-i` asked for the
/// prompt to open over it.
///
/// `handle_main_error` finalizes on its way out, so it is the arm that must not
/// be taken while the prompt still has the interpreter ahead of it.
fn report_or_end_on_main_error(
    e: pyre_interpreter::PyError,
    inspect: bool,
    canonical: pyre_object::PyObjectRef,
    ec_ptr: *const PyExecutionContext,
) {
    if !inspect {
        handle_main_error(e, canonical, ec_ptr);
    }
    report_main_error_for_inspect(e);
}

enum MainProgram<'a> {
    Source(Box<dyn FnOnce() -> String + 'a>, Mode),
    Bytecode(Box<dyn FnOnce() -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> + 'a>),
}

fn eval_program_in_main(
    program: MainProgram<'_>,
    filename: &str,
    main_file: Option<&str>,
    execution_context: Rc<PyExecutionContext>,
    canonical: pyre_object::PyObjectRef,
    main_module: pyre_object::PyObjectRef,
    no_site: bool,
    run_code: bool,
) -> Option<MainSession> {
    let ec_ptr = Rc::as_ptr(&execution_context);
    let sourceless = matches!(program, MainProgram::Bytecode(_));
    // The frame below takes the context by value; the prompt needs it too.
    let session_context = Rc::clone(&execution_context);

    pyre_interpreter::app_main::seed_main_module_attrs(canonical, main_module, main_file);
    let script_file = if filename == "<stdin>" {
        None
    } else {
        main_file
    };
    seed_main_loader(canonical, script_file, sourceless, ec_ptr);
    import_site(no_site, canonical, ec_ptr);

    // `main.c pymain_header`, which runs once startup is complete -- after
    // `site` and the warnings bootstrap, so it follows the import trace `-v`
    // produced for all of that rather than heading it, which is what makes the
    // banner label that trace.  Its three tests are the three here: `-q`
    // suppresses outright, a program that is not a tty needs `verbose` to get a
    // banner at all, and `site_import` decides the second line.  `<stdin>`
    // names the branch -- a tty takes the prompt, which prints its own.
    if filename == "<stdin>" && !importing::quiet_flag() && importing::verbose_flag() != 0 {
        // `fprintf(stderr, ...)`, the process descriptor rather than the
        // `sys.stderr` a `sitecustomize` may have replaced by now.
        eprintln!("{}", repl::BANNER);
        // The `site_import` half: the second line names the four builtins
        // `site` installs, so `-S` leaves it nothing to offer.
        if !no_site {
            eprintln!("{}", repl::BANNER_COPYRIGHT);
        }
    }

    // Source is read and compiled only once startup is complete, which is the order
    // `pymain_run_python` runs in: `site` is imported by interpreter
    // initialisation, ahead of every path that reaches user source.  A program
    // that will not open, or will not compile, still gets it -- the report
    // below goes through `sys.excepthook`, which `site` is entitled to have
    // replaced, and the failure still finalizes on its way out.
    let mut source = None;
    let frame = match program {
        MainProgram::Bytecode(load_bytecode) => {
            let code = match load_bytecode() {
                Ok(code) => code,
                Err(error) => {
                    let inspect = prompt_requested(run_code);
                    report_or_end_on_main_error(error, inspect, canonical, ec_ptr);
                    return finish_or_inspect(inspect, session_context, canonical, main_module);
                }
            };
            PyFrame::new_with_context_and_globals_obj(code, execution_context, canonical)
        }
        MainProgram::Source(read_source, mode) => {
            let text = read_source();
            let code = match pyre_interpreter::syntax_warnings::compile_with_codegen_warnings(
                &text,
                mode,
                rustpython_wtf8::Wtf8::new(filename),
                CompileOpts {
                    optimize: importing::optimize_flag(),
                    debug_ranges: importing::code_debug_ranges_flag(),
                    int_max_str_digits: pyre_interpreter::module::sys::state::int_max_str_digits()
                        .max(0) as usize,
                    ..Default::default()
                },
            ) {
                Ok(code) => code,
                Err(pyre_interpreter::syntax_warnings::SourceCompileError::Compile(e)) => {
                    let mut err = pyre_interpreter::compile_err_to_syntax_error(e, &text, mode);
                    if !pyre_interpreter::error::print_exception_via_excepthook(&mut err) {
                        pyre_interpreter::eprint_syntax_error(&err);
                    }
                    if prompt_requested(run_code) {
                        return finish_or_inspect(true, session_context, canonical, main_module);
                    }
                    end_after_startup_failure(canonical, ec_ptr, 1);
                }
                Err(pyre_interpreter::syntax_warnings::SourceCompileError::Warning(error)) => {
                    let inspect = prompt_requested(run_code);
                    report_or_end_on_main_error(error, inspect, canonical, ec_ptr);
                    return finish_or_inspect(inspect, session_context, canonical, main_module);
                }
            };
            source = Some(text);
            PyFrame::new_with_context_and_globals(code, execution_context, canonical)
        }
    };

    // The frame handle's `owner_root` and the extra pin below keep `w_globals`
    // reachable for the run. Both must die before `finalize_modules` collects,
    // or a `__main__` global's `__del__` never sees the detached `sys.modules`.
    // PyPy `executioncontext.py leave` has already popped the frame.
    let failure = match frame {
        Ok(mut frame) => {
            // The frame is a GC object, and the only root that reaches one is the
            // `CURRENT_FRAME` / `f_backref` chain the frame walker follows — which it
            // joins when `eval_with_jit` below installs it.  What runs between here
            // and that call is Python -- the `linecache` registration below -- and can
            // therefore drive a collection that would reclaim the frame and hand its
            // storage to the next frame allocation.  RPython keeps a local holding a GC
            // object alive through the translated shadow stack; publish it explicitly
            // for the same span.
            let _main_frame_root = pyre_object::gc_roots::push_roots();
            let _ = pyre_object::gc_roots::pin_root(
                &*frame as *const PyFrame as pyre_object::PyObjectRef,
            );

            // `<string>` names no file, so `linecache._interactive_cache` is the only
            // place a traceback frame or `inspect.getsource` can reach the command's
            // text; publish the compiled tree there before it runs. A script keeps its
            // own path and a stdin run keeps `<stdin>`, which is the same distinction
            // `__file__` above already makes. Registration is cosmetic, so a failure
            // must not stop the command.
            if filename == "<string>"
                && let Some(source) = source.as_deref()
            {
                let _ = repl::register_interactive_code(
                    canonical,
                    ec_ptr,
                    frame.fget_f_code(),
                    source,
                    filename,
                );
            }

            match eval_with_jit(&mut frame, None) {
                Ok(result) => {
                    if !result.is_null() && !unsafe { pyre_object::is_none(result) } {
                        println!("{}", PyDisplay(result));
                    }
                    None
                }
                Err(e) => Some(e),
            }
        }
        Err(e) => {
            pyre_interpreter::eprint_exception(&e, true);
            if prompt_requested(run_code) {
                return finish_or_inspect(true, session_context, canonical, main_module);
            }
            end_after_startup_failure(canonical, ec_ptr, 1);
        }
    };
    // Asked once the program is done, because the program is one of the things
    // that answers it -- `os.environ["PYTHONINSPECT"] = "1"` is a request for
    // the prompt below.
    let inspect = prompt_requested(run_code);
    if let Some(e) = failure {
        report_or_end_on_main_error(e, inspect, canonical, ec_ptr);
    }
    finish_or_inspect(inspect, session_context, canonical, main_module)
}

/// Report a script that could not be opened and end the run.
///
/// `pymain_run_file_obj` writes `"%S: can't open file %R: [Errno %d] %s"` --
/// the path's `repr`, then the C `errno` and `strerror` for the failure rather
/// than the message the error itself displays, which on Windows names the
/// Win32 failure in the language the OS is localised to.  It answers
/// `EXIT_STATUS_ERROR`, which is 2 -- the status a shell reads as "the program
/// was never run", distinct from the 1 a program that ran and failed exits
/// with.
fn end_on_unopenable_script(
    binary_name: &str,
    path: &str,
    error: &std::io::Error,
    canonical: pyre_object::PyObjectRef,
    ec_ptr: *const PyExecutionContext,
) -> ! {
    let path = pyre_interpreter::display::format_wtf8_repr(rustpython_wtf8::Wtf8::new(path));
    let (errno, strerror) = pyre_interpreter::PyError::io_errno_strerror(error);
    eprintln!("{binary_name}: can't open file {path}: [Errno {errno}] {strerror}");
    end_after_startup_failure(canonical, ec_ptr, 2);
}

fn read_script_source(
    path: &str,
    binary_name: &str,
    canonical: pyre_object::PyObjectRef,
    ec_ptr: *const PyExecutionContext,
) -> String {
    // The script is read through the import machinery's source provider, so
    // under sandbox the controller VFS mediates it over the same channel
    // module imports use.  The bytes then go through the tokenizer's BOM /
    // PEP 263 decoding in every build.
    match importing::read_source_bytes(Path::new(path)) {
        Ok(bytes) => match pyre_interpreter::decode_file_source_bytes(
            &bytes,
            rustpython_wtf8::Wtf8::new(path),
            false,
        ) {
            Ok(source) => source,
            Err(mut error) => {
                // A decoded source file can fail before a frame exists (an
                // embedded NUL or a bad encoding declaration), but its
                // SyntaxError still carries filename/line/text. Route it
                // through the same startup hook and syntax fallback as the
                // compile failure below so that location is not collapsed
                // into the generic one-line exception rendering.
                if !pyre_interpreter::error::print_exception_via_excepthook(&mut error) {
                    pyre_interpreter::eprint_syntax_error(&error);
                }
                end_after_startup_failure(canonical, ec_ptr, 1);
            }
        },
        Err(e) => end_on_unopenable_script(binary_name, path, &e, canonical, ec_ptr),
    }
}

/// `app_main.py:run_command_line` only folds the filename case on Windows
/// before recognizing `.pyc`; POSIX keeps its case-sensitive path semantics.
fn script_path_is_pyc(path: &str) -> bool {
    #[cfg(windows)]
    {
        path.to_ascii_lowercase().ends_with(".pyc")
    }
    #[cfg(not(windows))]
    {
        path.ends_with(".pyc")
    }
}

fn run_script_path(
    filename: &str,
    no_site: bool,
    binary_name: &str,
    run_code: bool,
) -> Option<MainSession> {
    let execution_context = setup_exec_context();
    let ec_ptr = Rc::as_ptr(&execution_context);
    let (canonical, main_module) =
        pyre_interpreter::app_main::prepare_main_module(&execution_context);

    let result = (|| -> Result<bool, pyre_interpreter::PyError> {
        // Import `sys` up front so its creation flushes the native search-path
        // seed into `sys.path` before `site` and user code read it.
        let _ = importing::importhook(
            rustpython_wtf8::Wtf8::new("sys"),
            canonical,
            pyre_object::PY_NULL,
            0,
            ec_ptr,
        );
        // pylifecycle.c init_importlib before site: install the importlib
        // bootstrap so `sys.meta_path` / `sys.path_hooks` are populated before
        // the run target is chosen.
        if let Err(e) = importing::init_importlib_bootstrap(canonical, ec_ptr) {
            eprintln!("pyre: importlib bootstrap failed: {}", e.message_text());
        }
        path_hook_accepts(filename, canonical, ec_ptr)
    })();

    match result {
        Ok(true) => {
            importing::restage_sys_path_0(std::ffi::OsStr::new(filename));
            // `app_main.py:869-870` binds `__main__.__loader__` before
            // `app_main.py:875-882` imports `site`, so a `sitecustomize` that
            // reads the main module finds it already there.  This target has no
            // source file of its own to bind, and runpy rebinds the attribute to
            // the package's own loader when it runs `__main__` below.
            seed_main_loader(canonical, None, false, ec_ptr);
            import_site(no_site, canonical, ec_ptr);
            let outcome = runpy_run_module_as_main(canonical, ec_ptr, "__main__", false);
            // Asked after the entry point has run, the way the source-file and
            // `-m` paths ask it: `pymain_run_python` reads `PYTHONINSPECT` at
            // the end "to give programs the opportunity to set it from Python",
            // and a directory or zipapp `__main__` is such a program.
            let inspect = prompt_requested(run_code);
            if let Err(e) = outcome {
                report_or_end_on_main_error(e, inspect, canonical, ec_ptr);
            }
            finish_or_inspect(inspect, execution_context, canonical, main_module)
        }
        Ok(false) => {
            let main_file = absolute_script_path(filename);
            let filename = main_file.as_deref().unwrap_or(filename);
            let program = if script_path_is_pyc(filename) {
                // Deferred for the reason the source read below is, and for one
                // more that is this arm's alone: the unmarshalled code object is
                // a GC object living in a Rust local, and `import_site` runs
                // Python between here and the frame that roots it.  A collection
                // in that window reclaims the code, and the module the script
                // then builds carries references into freed storage.
                MainProgram::Bytecode(Box::new(move || {
                    let bytes = match importing::read_source_bytes(Path::new(filename)) {
                        Ok(bytes) => bytes,
                        Err(e) => {
                            end_on_unopenable_script(binary_name, filename, &e, canonical, ec_ptr)
                        }
                    };
                    importing::load_pyc_script(&bytes)
                }))
            } else {
                MainProgram::Source(
                    // The resolved path rather than the argv spelling: the read is
                    // deferred until `site` has been imported, so a `sitecustomize`
                    // that changes the working directory would otherwise resolve a
                    // relative argv against the new one -- opening a different
                    // same-named file than the one `__file__` names, or none.  The
                    // file a run opens and the file it reports are the same file.
                    Box::new(|| read_script_source(filename, binary_name, canonical, ec_ptr)),
                    Mode::Exec,
                )
            };
            eval_program_in_main(
                program,
                filename,
                main_file.as_deref(),
                execution_context,
                canonical,
                main_module,
                no_site,
                run_code,
            )
        }
        Err(e) => {
            let inspect = prompt_requested(run_code);
            report_or_end_on_main_error(e, inspect, canonical, ec_ptr);
            finish_or_inspect(inspect, execution_context, canonical, main_module)
        }
    }
}

fn run_source(
    source: &str,
    mode: Mode,
    filename: &str,
    no_site: bool,
    run_code: bool,
) -> Option<MainSession> {
    // `config_run_filename_abspath` absolutizes the run filename before the
    // module is compiled, so `co_filename` — and with it every `File "…"` line
    // a traceback prints — carries the absolute path, not the literal argv
    // spelling.  `sys.argv[0]` keeps the spelling and is set elsewhere.
    let main_file = absolute_script_path(filename);
    let filename = main_file.as_deref().unwrap_or(filename);
    let execution_context = setup_exec_context();
    let ec_ptr = Rc::as_ptr(&execution_context);
    let (canonical, main_module) =
        pyre_interpreter::app_main::prepare_main_module(&execution_context);

    // `moduledef.py` `startup_at_translation_time_only` reads `sys` with
    // `space.getbuiltinmodule('sys')`. The module is already registered by
    // `install_builtin_modules`; `importhook` walks the pre-bootstrap finder
    // for that same object and flushes `sys.path` no earlier.
    let _ = importing::getbuiltinmodule("sys", false, true, ec_ptr);

    // pylifecycle.c init_importlib before site: install the importlib
    // bootstrap so `builtins.__import__` routes imports through
    // `sys.meta_path` / `sys.path_hooks` from the first user statement.
    // A failure (no reachable stdlib) is non-fatal — the native importer
    // keeps serving imports, the minimal-importer role.
    if let Err(e) = importing::init_importlib_bootstrap(canonical, ec_ptr) {
        eprintln!("pyre: importlib bootstrap failed: {}", e.message_text());
    }

    eval_program_in_main(
        MainProgram::Source(Box::new(|| source.to_string()), mode),
        filename,
        main_file.as_deref(),
        execution_context,
        canonical,
        main_module,
        no_site,
        run_code,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        RunMode, dedent_command, parse_heapsize, parse_interact, parse_python_int_i64,
        set_last_exec_ctx, setup_exec_context, take_heapsize_option,
    };

    #[test]
    fn command_dedent_removes_shared_space_prefix() {
        let source = "\n    import sys\n    print(\"ok\")\n";
        assert_eq!(
            dedent_command(source).as_ref(),
            "\nimport sys\nprint(\"ok\")\n"
        );
    }

    #[test]
    fn command_dedent_leaves_extra_indentation() {
        let source = "  print(\"first\")\n    print(\"second\")\n";
        assert_eq!(
            dedent_command(source).as_ref(),
            "print(\"first\")\n  print(\"second\")\n"
        );
    }

    #[test]
    fn command_dedent_treats_tabs_as_distinct_prefix_bytes() {
        let source = "\tprint(\"first\")\n\tprint(\"second\")\n";
        assert_eq!(
            dedent_command(source).as_ref(),
            "print(\"first\")\nprint(\"second\")\n"
        );
    }

    #[test]
    fn command_dedent_handles_one_line_without_leading_newline() {
        assert_eq!(dedent_command("    print(\"B\")").as_ref(), "print(\"B\")");
    }

    #[test]
    fn command_dedent_empties_whitespace_only_line() {
        let cases = [
            ("shallower", "  "),
            ("deeper", "        "),
            ("equal", "    "),
        ];
        for (name, indent) in cases {
            let source = format!("    print(\"first\")\n{indent}\n    print(\"second\")\n");
            assert_eq!(
                dedent_command(&source).as_ref(),
                "print(\"first\")\n\nprint(\"second\")\n",
                "case {name}",
            );
        }
    }

    #[test]
    fn command_dedent_is_raw_text_and_stops_at_column_zero() {
        let source = "    text = \"\"\"\n  inside\ninside\n\"\"\"\n";
        assert!(matches!(
            dedent_command(source),
            std::borrow::Cow::Borrowed(value) if value == source
        ));
    }

    #[test]
    fn command_dedent_leaves_a_blank_line_alone_when_there_is_no_margin() {
        let source = "x = \"\"\"a\n  \nb\"\"\"\n";
        assert!(matches!(
            dedent_command(source),
            std::borrow::Cow::Borrowed(value) if value == source
        ));
    }

    #[test]
    fn command_dedent_counts_a_carriage_return_as_content() {
        // Python tokenization normalizes the retained carriage return.
        let source = "    a\r\n  \r\n    b\r\n";
        assert_eq!(dedent_command(source).as_ref(), "  a\r\n\r\n  b\r\n");
    }

    #[test]
    fn command_dedent_counts_whitespace_that_is_not_space_or_tab_as_content() {
        // A form feed on its own line is content at column zero, which leaves
        // no margin for anything to lose.
        for content in ["\u{c}", "\u{b}", "\u{85}"] {
            let source = format!("    a\n{content}\n    b\n");
            assert!(
                matches!(dedent_command(&source), std::borrow::Cow::Borrowed(value) if value == source),
                "{content:?}"
            );
        }
    }

    #[test]
    fn heapsize_suffixes_and_validation() {
        // pypy_interact.py:88-102: k/m/g multipliers, decimal fallback.
        assert_eq!(parse_heapsize("512").unwrap(), 512);
        assert_eq!(parse_heapsize("64k").unwrap(), 64 * 1024);
        assert_eq!(parse_heapsize("128M").unwrap(), 128 * 1024 * 1024);
        assert_eq!(parse_heapsize("2g").unwrap(), 2 * 1024 * 1024 * 1024);
        // non-positive and malformed values are usage errors.
        assert!(parse_heapsize("0").is_err());
        assert!(parse_heapsize("abc").is_err());
        assert!(parse_heapsize("").is_err());
        // `bytes > sys.maxint` is refused rather than handed on.
        assert_eq!(
            parse_heapsize("9223372036854775807").unwrap(),
            i64::MAX as u64
        );
        assert!(parse_heapsize("9223372036854775808").is_err());
        // 2**33 gibibytes is 2**63 — one past the maximum, and reached through
        // a multiplier rather than by naming the count.
        assert!(parse_heapsize("8589934592g").is_err());
    }

    #[test]
    fn interact_carries_heapsize_into_the_child_argv() {
        // `extraoptions[:0] = ['--heapsize', str(bytes)]` then
        // `extraoptions + arguments[1:]`: the suffix is resolved by the
        // controller, and the child is handed the byte count in front of its
        // own arguments.
        let mode = parse_interact(&mut lexopt::Parser::from_args([
            "--heapsize",
            "64k",
            "./pyre-sandbox",
            "prog.py",
            "--flag",
        ]))
        .unwrap();
        let RunMode::Interact { exe, args, .. } = mode else {
            panic!("expected an interact mode");
        };
        assert_eq!(exe, "./pyre-sandbox");
        assert_eq!(args, ["--heapsize", "65536", "prog.py", "--flag"]);

        // Without the option the child's arguments are its own alone.
        let mode = parse_interact(&mut lexopt::Parser::from_args([
            "./pyre-sandbox",
            "prog.py",
        ]))
        .unwrap();
        let RunMode::Interact { args, .. } = mode else {
            panic!("expected an interact mode");
        };
        assert_eq!(args, ["prog.py"]);
    }

    #[test]
    fn the_leading_heapsize_option_is_taken_out_of_argv() {
        // `if len(argv) > 2 and argv[1] == '--heapsize': ... argv = argv[:1] +
        // argv[3:]`.
        let argv = |args: &[&str]| -> Vec<std::ffi::OsString> {
            args.iter().map(std::ffi::OsString::from).collect()
        };
        let (rest, size) =
            take_heapsize_option("pyre", argv(&["pyre", "--heapsize", "65536", "prog.py"]));
        assert_eq!(size, Some(65536));
        assert_eq!(rest, argv(&["pyre", "prog.py"]));

        // The position is the whole test: anywhere else it is an ordinary
        // argument, and `parse_args` reports it as one.
        let (rest, size) =
            take_heapsize_option("pyre", argv(&["pyre", "prog.py", "--heapsize", "65536"]));
        assert_eq!(size, None);
        assert_eq!(rest, argv(&["pyre", "prog.py", "--heapsize", "65536"]));

        // `len(argv) > 2` — a trailing `--heapsize` has no value to read.
        let (rest, size) = take_heapsize_option("pyre", argv(&["pyre", "--heapsize"]));
        assert_eq!(size, None);
        assert_eq!(rest, argv(&["pyre", "--heapsize"]));

        // `int(argv[2])`: a negative count is taken; only a positive value
        // lowers the collector thresholds.
        let (rest, size) =
            take_heapsize_option("pyre", argv(&["pyre", "--heapsize", "-1", "x.py"]));
        assert_eq!(size, Some(-1));
        assert_eq!(rest, argv(&["pyre", "x.py"]));

        // `LLHelpers.ll_int` accepts a space between the sign and the digits.
        let (rest, size) =
            take_heapsize_option("pyre", argv(&["pyre", "--heapsize", "- 1", "x.py"]));
        assert_eq!(size, Some(-1));
        assert_eq!(rest, argv(&["pyre", "x.py"]));

        let (rest, size) =
            take_heapsize_option("pyre", argv(&["pyre", "--heapsize", " 5 ", "x.py"]));
        assert_eq!(size, Some(5));
        assert_eq!(rest, argv(&["pyre", "x.py"]));

        let (rest, size) =
            take_heapsize_option("pyre", argv(&["pyre", "--heapsize", "+7", "x.py"]));
        assert_eq!(size, Some(7));
        assert_eq!(rest, argv(&["pyre", "x.py"]));

        let (rest, size) =
            take_heapsize_option("pyre", argv(&["pyre", "--heapsize", "16777216", "x.py"]));
        assert_eq!(size, Some(16777216));
        assert_eq!(rest, argv(&["pyre", "x.py"]));

        let (rest, size) = take_heapsize_option("pyre", argv(&["pyre", "x.py"]));
        assert_eq!(size, None);
        assert_eq!(rest, argv(&["pyre", "x.py"]));
    }

    #[test]
    fn heapsize_ll_int_grammar() {
        // `LLHelpers.ll_int` at base 10: no `_`, and only `' '` is whitespace.
        assert_eq!(parse_python_int_i64("1_000"), None);
        assert_eq!(parse_python_int_i64("\t5"), None);
        assert_eq!(parse_python_int_i64("5\n"), None);
        assert_eq!(parse_python_int_i64(" 5 "), Some(5));
        assert_eq!(parse_python_int_i64("- 1"), Some(-1));
        assert_eq!(parse_python_int_i64("+7"), Some(7));
    }

    #[test]
    fn repeated_runtime_setup_reuses_space_signal_action() {
        // pypy/module/signal/moduledef.py:64-66 owns this action on `space`,
        // not on an ExecutionContext.  Embeddings may create more than one
        // top-level EC in a process; each must cache the same action pointer
        // without adding another process-lifetime Box/actionflag entry.
        let first = setup_exec_context();
        let first_action = first
            .check_signal_action
            .expect("runtime setup installs signal action");
        let second = setup_exec_context();
        let second_action = second
            .check_signal_action
            .expect("repeated runtime setup installs signal action");
        assert!(std::ptr::addr_eq(first_action, second_action));
        set_last_exec_ctx(std::ptr::null());
    }
}
