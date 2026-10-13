//! Compile the `_ctypes` SEH fence and, when the host can link it, the
//! `_minimal_curses` wrappers.
//!
//! `__try`/`__except` is the only thing that reaches a structured exception
//! and no Rust compiler emits one, so the fence a foreign call is made
//! inside (`src/module/_ctypes/seh.rs`) is a C translation unit of its own.
//!
//! `fficurses.py guess_eci` compiles a probe that references `setupterm`
//! and keeps the first `ExternalCompilationInfo` that links. The same
//! probe emits the uppercase integer constants `moduledef.py` copies out
//! of `_curses`. `term.h` is included only in that translation unit.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Uppercase `int` names `pypy/module/_minimal_curses` exposes. A header
/// that does not define one contributes nothing; the value is never
/// written here.
const CURSES_NAMES: &[&str] = &[
    "ALL_MOUSE_EVENTS",
    "A_ALTCHARSET",
    "A_ATTRIBUTES",
    "A_BLINK",
    "A_BOLD",
    "A_CHARTEXT",
    "A_COLOR",
    "A_DIM",
    "A_HORIZONTAL",
    "A_INVIS",
    "A_LEFT",
    "A_LOW",
    "A_NORMAL",
    "A_PROTECT",
    "A_REVERSE",
    "A_RIGHT",
    "A_STANDOUT",
    "A_TOP",
    "A_UNDERLINE",
    "A_VERTICAL",
    "BUTTON1_CLICKED",
    "BUTTON1_DOUBLE_CLICKED",
    "BUTTON1_PRESSED",
    "BUTTON1_RELEASED",
    "BUTTON1_TRIPLE_CLICKED",
    "BUTTON2_CLICKED",
    "BUTTON2_DOUBLE_CLICKED",
    "BUTTON2_PRESSED",
    "BUTTON2_RELEASED",
    "BUTTON2_TRIPLE_CLICKED",
    "BUTTON3_CLICKED",
    "BUTTON3_DOUBLE_CLICKED",
    "BUTTON3_PRESSED",
    "BUTTON3_RELEASED",
    "BUTTON3_TRIPLE_CLICKED",
    "BUTTON4_CLICKED",
    "BUTTON4_DOUBLE_CLICKED",
    "BUTTON4_PRESSED",
    "BUTTON4_RELEASED",
    "BUTTON4_TRIPLE_CLICKED",
    "BUTTON_ALT",
    "BUTTON_CTRL",
    "BUTTON_SHIFT",
    "COLOR_BLACK",
    "COLOR_BLUE",
    "COLOR_CYAN",
    "COLOR_GREEN",
    "COLOR_MAGENTA",
    "COLOR_RED",
    "COLOR_WHITE",
    "COLOR_YELLOW",
    "ERR",
    "KEY_A1",
    "KEY_A3",
    "KEY_B2",
    "KEY_BACKSPACE",
    "KEY_BEG",
    "KEY_BREAK",
    "KEY_BTAB",
    "KEY_C1",
    "KEY_C3",
    "KEY_CANCEL",
    "KEY_CATAB",
    "KEY_CLEAR",
    "KEY_CLOSE",
    "KEY_COMMAND",
    "KEY_COPY",
    "KEY_CREATE",
    "KEY_CTAB",
    "KEY_DC",
    "KEY_DL",
    "KEY_DOWN",
    "KEY_EIC",
    "KEY_END",
    "KEY_ENTER",
    "KEY_EOL",
    "KEY_EOS",
    "KEY_EXIT",
    "KEY_F0",
    "KEY_F1",
    "KEY_F10",
    "KEY_F11",
    "KEY_F12",
    "KEY_F13",
    "KEY_F14",
    "KEY_F15",
    "KEY_F16",
    "KEY_F17",
    "KEY_F18",
    "KEY_F19",
    "KEY_F2",
    "KEY_F20",
    "KEY_F21",
    "KEY_F22",
    "KEY_F23",
    "KEY_F24",
    "KEY_F25",
    "KEY_F26",
    "KEY_F27",
    "KEY_F28",
    "KEY_F29",
    "KEY_F3",
    "KEY_F30",
    "KEY_F31",
    "KEY_F32",
    "KEY_F33",
    "KEY_F34",
    "KEY_F35",
    "KEY_F36",
    "KEY_F37",
    "KEY_F38",
    "KEY_F39",
    "KEY_F4",
    "KEY_F40",
    "KEY_F41",
    "KEY_F42",
    "KEY_F43",
    "KEY_F44",
    "KEY_F45",
    "KEY_F46",
    "KEY_F47",
    "KEY_F48",
    "KEY_F49",
    "KEY_F5",
    "KEY_F50",
    "KEY_F51",
    "KEY_F52",
    "KEY_F53",
    "KEY_F54",
    "KEY_F55",
    "KEY_F56",
    "KEY_F57",
    "KEY_F58",
    "KEY_F59",
    "KEY_F6",
    "KEY_F60",
    "KEY_F61",
    "KEY_F62",
    "KEY_F63",
    "KEY_F7",
    "KEY_F8",
    "KEY_F9",
    "KEY_FIND",
    "KEY_HELP",
    "KEY_HOME",
    "KEY_IC",
    "KEY_IL",
    "KEY_LEFT",
    "KEY_LL",
    "KEY_MARK",
    "KEY_MAX",
    "KEY_MESSAGE",
    "KEY_MIN",
    "KEY_MOUSE",
    "KEY_MOVE",
    "KEY_NEXT",
    "KEY_NPAGE",
    "KEY_OPEN",
    "KEY_OPTIONS",
    "KEY_PPAGE",
    "KEY_PREVIOUS",
    "KEY_PRINT",
    "KEY_REDO",
    "KEY_REFERENCE",
    "KEY_REFRESH",
    "KEY_REPLACE",
    "KEY_RESET",
    "KEY_RESIZE",
    "KEY_RESTART",
    "KEY_RESUME",
    "KEY_RIGHT",
    "KEY_SAVE",
    "KEY_SBEG",
    "KEY_SCANCEL",
    "KEY_SCOMMAND",
    "KEY_SCOPY",
    "KEY_SCREATE",
    "KEY_SDC",
    "KEY_SDL",
    "KEY_SELECT",
    "KEY_SEND",
    "KEY_SEOL",
    "KEY_SEXIT",
    "KEY_SF",
    "KEY_SFIND",
    "KEY_SHELP",
    "KEY_SHOME",
    "KEY_SIC",
    "KEY_SLEFT",
    "KEY_SMESSAGE",
    "KEY_SMOVE",
    "KEY_SNEXT",
    "KEY_SOPTIONS",
    "KEY_SPREVIOUS",
    "KEY_SPRINT",
    "KEY_SR",
    "KEY_SREDO",
    "KEY_SREPLACE",
    "KEY_SRESET",
    "KEY_SRIGHT",
    "KEY_SRSUME",
    "KEY_SSAVE",
    "KEY_SSUSPEND",
    "KEY_STAB",
    "KEY_SUNDO",
    "KEY_SUSPEND",
    "KEY_UNDO",
    "KEY_UP",
    "OK",
    "REPORT_MOUSE_POSITION",
];

/// One `ExternalCompilationInfo` from `try_eci`.
struct Candidate {
    compile_args: Vec<String>,
    library_dirs: Vec<String>,
    libraries: Vec<String>,
    link_extra: Vec<String>,
    /// `includes=['ncurses/curses.h', 'ncurses/term.h']`.
    ncurses_prefix: bool,
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Named by pyre-interpreter/build.rs; this crate only branches on it.
    println!("cargo:rustc-check-cfg=cfg(pyre_ffi_type_longdouble)");
    println!("cargo:rustc-check-cfg=cfg(pyre_minimal_curses)");
    println!("cargo:rerun-if-changed=src/module/_minimal_curses/fficurses.c");
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.ends_with("-pc-windows-msvc") {
        println!("cargo:rerun-if-changed=src/module/_ctypes/seh.c");
        cc::Build::new()
            .file("src/module/_ctypes/seh.c")
            .compile("pyre_ctypes_seh");
    }
    if curses_probe_enabled() {
        configure_minimal_curses();
    }
}

/// `fficurses.py guess_eci` runs on the host that will execute the
/// module. A cross compile (wasm included) leaves the module out, the
/// same as a failed probe.
fn curses_probe_enabled() -> bool {
    if std::env::var_os("CARGO_CFG_UNIX").is_none() {
        return false;
    }
    if std::env::var_os("CARGO_FEATURE_HOST_ENV").is_none() {
        return false;
    }
    if std::env::var_os("CARGO_FEATURE_SANDBOX").is_some() {
        return false;
    }
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("wasm32") {
        return false;
    }
    let host = std::env::var("HOST").unwrap_or_default();
    host == target
}

fn out_dir() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"))
}

fn record_failure(candidate: &Candidate, detail: &[u8]) {
    let mut buf = format!(
        "libs={:?} dirs={:?} prefix={} compile={:?} link_extra={:?}\n",
        candidate.libraries,
        candidate.library_dirs,
        candidate.ncurses_prefix,
        candidate.compile_args,
        candidate.link_extra
    )
    .into_bytes();
    buf.extend_from_slice(detail);
    let _ = fs::write(out_dir().join("curses_probe_last_stderr.txt"), buf);
}

/// `try_tools` then `try_cflags` × `try_ldflags`. The first candidate
/// whose probe runs and whose wrapper compiles is `eci`.
fn configure_minimal_curses() {
    if let Some(candidate) = config_tool("ncursesw6-config")
        && accept(&candidate)
    {
        return;
    }
    if let Some(candidate) = config_tool("ncurses5-config")
        && accept(&candidate)
    {
        return;
    }
    // `try_tools` asks pkg-config for ncursesw twice.
    if let Some(candidate) = pkg_config_ncursesw()
        && accept(&candidate)
    {
        return;
    }
    if let Some(candidate) = pkg_config_ncursesw()
        && accept(&candidate)
    {
        return;
    }
    for candidate in cflags_ldflags_grid() {
        if accept(&candidate) {
            return;
        }
    }
}

fn config_tool(name: &str) -> Option<Candidate> {
    let cflags = command_stdout(name, &["--cflags"])?;
    let libs = command_stdout(name, &["--libs"])?;
    candidate_from_flags(&cflags, &libs, false)
}

fn pkg_config_ncursesw() -> Option<Candidate> {
    let exists = Command::new("pkg-config")
        .args(["ncursesw", "--exists"])
        .status()
        .ok()?;
    if !exists.success() {
        return None;
    }
    let cflags = command_stdout("pkg-config", &["ncursesw", "--cflags"])?;
    let libs = command_stdout("pkg-config", &["ncursesw", "--libs"])?;
    candidate_from_flags(&cflags, &libs, false)
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// `from_compiler_flags` / `from_linker_flags`. A flag in the wrong
/// group drops the candidate, as those parsers raise `ValueError`.
fn candidate_from_flags(cflags: &str, libs: &str, ncurses_prefix: bool) -> Option<Candidate> {
    let mut compile_args = Vec::new();
    for arg in cflags.split_whitespace() {
        if arg.starts_with("-I") {
            compile_args.push(arg.to_string());
        } else if let Some(macro_def) = arg.strip_prefix("-D") {
            let macro_name = macro_def.split('=').next().unwrap_or(macro_def);
            // `from_compiler_flags` skips `_XOPEN_SOURCE`.
            if macro_name == "_XOPEN_SOURCE" {
                continue;
            }
            compile_args.push(arg.to_string());
        } else if arg.starts_with("-L") || arg.starts_with("-l") {
            return None;
        } else {
            compile_args.push(arg.to_string());
        }
    }
    let mut library_dirs = Vec::new();
    let mut libraries = Vec::new();
    let mut link_extra = Vec::new();
    for arg in libs.split_whitespace() {
        if let Some(dir) = arg.strip_prefix("-L") {
            library_dirs.push(dir.to_string());
        } else if let Some(lib) = arg.strip_prefix("-l") {
            libraries.push(lib.to_string());
        } else if arg.starts_with("-I") || arg.starts_with("-D") {
            return None;
        } else {
            link_extra.push(arg.to_string());
        }
    }
    Some(Candidate {
        compile_args,
        library_dirs,
        libraries,
        link_extra,
        ncurses_prefix,
    })
}

fn cflags_ldflags_grid() -> Vec<Candidate> {
    let cflags: [(Vec<String>, bool); 4] = [
        (Vec::new(), false),
        (vec!["-I/usr/include/ncurses".to_string()], false),
        (vec!["-I/usr/include/ncursesw".to_string()], false),
        (Vec::new(), true),
    ];
    let ldflags: [(Vec<String>, Vec<String>); 6] = [
        (vec!["curses".to_string(), "tinfo".to_string()], Vec::new()),
        (vec!["curses".to_string()], Vec::new()),
        (vec!["ncurses".to_string(), "tinfo".to_string()], Vec::new()),
        (vec!["ncurses".to_string()], Vec::new()),
        (vec!["ncurses".to_string()], vec!["/usr/lib64".to_string()]),
        (vec!["ncursesw".to_string()], vec!["/usr/lib64".to_string()]),
    ];
    let mut out = Vec::new();
    for (compile_args, ncurses_prefix) in cflags {
        for (libraries, library_dirs) in &ldflags {
            out.push(Candidate {
                compile_args: compile_args.clone(),
                library_dirs: library_dirs.clone(),
                libraries: libraries.clone(),
                link_extra: Vec::new(),
                ncurses_prefix,
            });
        }
    }
    out
}

fn accept(candidate: &Candidate) -> bool {
    let stdout = match run_probe(candidate) {
        Ok(stdout) => stdout,
        Err(detail) => {
            record_failure(candidate, &detail);
            return false;
        }
    };
    let Some(body) = constants_body(&stdout) else {
        record_failure(candidate, stdout.as_bytes());
        return false;
    };
    let path = out_dir().join("curses_ints.c");
    if fs::write(&path, body).is_err() {
        return false;
    }
    if !compile_wrapper(candidate) {
        record_failure(candidate, b"fficurses.c failed to compile\n");
        return false;
    }
    for dir in &candidate.library_dirs {
        println!("cargo:rustc-link-search=native={dir}");
    }
    for lib in &candidate.libraries {
        println!("cargo:rustc-link-lib={lib}");
    }
    // `_link_args_from_eci` appends `link_extra` to the final link line.
    // `rustc-link-arg` reaches this package's tests. Numbered metadata is
    // what a dependent binary passes to its own link.
    for (index, extra) in candidate.link_extra.iter().enumerate() {
        println!("cargo:rustc-link-arg={extra}");
        // `cargo::metadata=KEY=VALUE` is `DEP_PYRE_MINIMAL_CURSES_KEY`.
        println!("cargo::metadata=LINK_ARG_{index}={extra}");
    }
    println!("cargo:rustc-cfg=pyre_minimal_curses");
    let _ = fs::remove_file(out_dir().join("curses_probe_last_stderr.txt"));
    true
}

fn run_probe(candidate: &Candidate) -> Result<String, Vec<u8>> {
    let dir = out_dir();
    let src = dir.join("curses_const_probe.c");
    let bin = dir.join("curses_const_probe");
    fs::write(&src, probe_source()).map_err(|err| err.to_string().into_bytes())?;
    let mut cmd = cc::Build::new().get_compiler().to_command();
    cmd.arg("-o").arg(&bin);
    for arg in &candidate.compile_args {
        cmd.arg(arg);
    }
    if candidate.ncurses_prefix {
        cmd.arg("-DPYRE_CURSES_INCLUDE_NCURSES_PREFIX");
    }
    cmd.arg(&src);
    for library_dir in &candidate.library_dirs {
        cmd.arg(format!("-L{library_dir}"));
    }
    for extra in &candidate.link_extra {
        cmd.arg(extra);
    }
    for lib in &candidate.libraries {
        cmd.arg(format!("-l{lib}"));
    }
    let compiled = cmd
        .output()
        .map_err(|err| format!("spawn compiler: {err}").into_bytes())?;
    if !compiled.status.success() {
        let mut detail = compiled.stderr;
        detail.extend_from_slice(&compiled.stdout);
        return Err(detail);
    }
    let ran = Command::new(&bin)
        .output()
        .map_err(|err| format!("spawn probe: {err}").into_bytes())?;
    if !ran.status.success() {
        return Err(ran.stderr);
    }
    Ok(String::from_utf8_lossy(&ran.stdout).into_owned())
}

fn probe_source() -> String {
    let mut src = String::from(
        "#include <stdio.h>\n\
         #if defined(PYRE_CURSES_INCLUDE_NCURSES_PREFIX)\n\
         #include <ncurses/curses.h>\n\
         #include <ncurses/term.h>\n\
         #else\n\
         #include <curses.h>\n\
         #include <term.h>\n\
         #endif\n\
         int main(void) {\n\
             void *volatile kept = 0;\n\
             kept = (void *)setupterm;\n\
             kept = (void *)tigetstr;\n\
             kept = (void *)tparm;\n\
             if (kept == 0) return 1;\n",
    );
    for name in CURSES_NAMES {
        // `_curses._setup` skips `keyname(KEY_MIN..KEY_MAX)` and `A_INVIS`
        // when `_m_NetBSD` (`__NetBSD__` in `_curses_build.py`). `KEY_MIN`
        // and `KEY_MAX` are in the unconditional copy list.
        if suppressed_on_netbsd(name) {
            src.push_str("#ifndef __NetBSD__\n");
        }
        // `curses.h` spells function keys as `#define KEY_F(n) (KEY_F0+(n))`.
        // `KEY_F1`..`KEY_F63` are not object-like macros, so `#ifdef KEY_F1`
        // is false wherever the header only provides `KEY_F`.
        if let Some(n) = key_f_index(name) {
            src.push_str("#if defined(");
            src.push_str(name);
            src.push_str(")\n");
            push_const_printf(&mut src, name, name);
            src.push_str("#elif defined(KEY_F)\n");
            push_const_printf(&mut src, name, &format!("KEY_F({n})"));
            src.push_str("#endif\n");
        } else {
            src.push_str("#ifdef ");
            src.push_str(name);
            src.push('\n');
            push_const_printf(&mut src, name, name);
            src.push_str("#endif\n");
        }
        if suppressed_on_netbsd(name) {
            src.push_str("#endif\n");
        }
    }
    src.push_str("    return 0;\n}\n");
    src
}

/// `KEY_*` from `keyname()`, plus `A_INVIS`. `KEY_MIN` and `KEY_MAX` stay.
fn suppressed_on_netbsd(name: &str) -> bool {
    if name == "A_INVIS" {
        return true;
    }
    name.starts_with("KEY_") && name != "KEY_MIN" && name != "KEY_MAX"
}

/// `KEY_F1`..`KEY_F63`. `KEY_F0` stays an object-like macro.
fn key_f_index(name: &str) -> Option<&str> {
    let rest = name.strip_prefix("KEY_F")?;
    if rest.is_empty() || rest.starts_with('0') || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: u32 = rest.parse().ok()?;
    if (1..=63).contains(&n) {
        Some(rest)
    } else {
        None
    }
}

fn push_const_printf(src: &mut String, name: &str, expr: &str) {
    src.push_str("    printf(\"(\\\"");
    src.push_str(name);
    src.push_str("\\\", %lld),\\n\", (long long)(");
    src.push_str(expr);
    src.push_str("));\n");
}

/// Keep a line only when it is `("NAME", <integer>),` and `NAME` is one
/// of [`CURSES_NAMES`]. Anything else rejects the candidate.
fn constants_body(stdout: &str) -> Option<String> {
    let mut elements = String::new();
    let mut seen = std::collections::BTreeSet::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (name, value) = parse_const_line(line)?;
        if !CURSES_NAMES.contains(&name.as_str()) || !seen.insert(name.clone()) {
            return None;
        }
        elements.push_str(&format!("    {{\"{name}\", {value}}},\n"));
    }
    if elements.is_empty() {
        None
    } else {
        Some(format!(
            "struct rpy_curses_int_entry {{\n    const char *name;\n    long long value;\n}};\n\
             static const struct rpy_curses_int_entry RPY_CURSES_INTS[] = {{\n{elements}}};\n\
             int rpy_curses_int_count(void) {{\n\
                 return (int)(sizeof RPY_CURSES_INTS / sizeof RPY_CURSES_INTS[0]);\n\
             }}\n\
             const char *rpy_curses_int_name(int index) {{\n\
                 return RPY_CURSES_INTS[index].name;\n\
             }}\n\
             long long rpy_curses_int_value(int index) {{\n\
                 return RPY_CURSES_INTS[index].value;\n\
             }}\n"
        ))
    }
}

fn parse_const_line(line: &str) -> Option<(String, String)> {
    let rest = line.strip_prefix("(\"")?.strip_suffix("),")?;
    let (name, value) = rest.split_once("\", ")?;
    if name.is_empty()
        || !name
            .chars()
            .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
    {
        return None;
    }
    let digits = value.strip_prefix('-').unwrap_or(value);
    if digits.is_empty() || !digits.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    if digits.len() > 1 && digits.starts_with('0') {
        return None;
    }
    Some((name.to_string(), value.to_string()))
}

fn compile_wrapper(candidate: &Candidate) -> bool {
    let mut build = cc::Build::new();
    build.file("src/module/_minimal_curses/fficurses.c");
    build.file(out_dir().join("curses_ints.c"));
    if candidate.ncurses_prefix {
        build.define("PYRE_CURSES_INCLUDE_NCURSES_PREFIX", None);
    }
    for arg in &candidate.compile_args {
        if let Some(dir) = arg.strip_prefix("-I") {
            build.include(dir);
        } else if let Some(def) = arg.strip_prefix("-D") {
            if let Some((key, value)) = def.split_once('=') {
                build.define(key, value);
            } else {
                build.define(def, None);
            }
        } else {
            build.flag(arg);
        }
    }
    build.try_compile("pyre_minimal_curses").is_ok()
}
