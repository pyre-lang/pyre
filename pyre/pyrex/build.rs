use std::path::Path;

fn main() {
    forward_minimal_curses_link_args();
    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    // INTERPRETER_THREAD_STACK_SIZE in lib.rs is 256 MiB (0x10000000). The
    // linker, not setrlimit, is what sizes the origin thread's stack on
    // darwin; keep the two figures the same so the interpreter's usable
    // stack is unchanged when it stays on that thread.
    if target.as_str() == "macos" {
        println!("cargo::rustc-link-arg-bins=-Wl,-stack_size,0x10000000");
    }
    let mut symbols = vec![
        "PyThreadState_SetAsyncExc".to_string(),
        "PyGILState_Ensure".to_string(),
        "PyGILState_Release".to_string(),
    ];
    // The C-API entry points a loaded extension resolves against. They exist
    // only in a `cpyext` build, and the same predicate gates the loader itself
    // (see pyre-interpreter's `cpyext` feature).
    if std::env::var_os("CARGO_FEATURE_CPYEXT").is_some()
        && std::env::var_os("CARGO_FEATURE_SANDBOX").is_none()
        && matches!(target.as_str(), "macos" | "linux")
    {
        symbols.extend(cpyext_symbols());
    }
    for symbol in symbols {
        match target.as_str() {
            "macos" => println!(
                "cargo::rustc-link-arg-bins=-Wl,-exported_symbol,_{}",
                symbol
            ),
            "linux" => println!(
                "cargo::rustc-link-arg-bins=-Wl,--export-dynamic-symbol={}",
                symbol
            ),
            _ => {}
        }
    }
    embed_windows_manifest(&target);
}

/// `ExternalCompilationInfo.link_extra`, carried as
/// `DEP_PYRE_MINIMAL_CURSES_LINK_ARG_<n>`.
fn forward_minimal_curses_link_args() {
    let mut args: Vec<(u32, String)> = Vec::new();
    for (key, value) in std::env::vars() {
        let Some(suffix) = key.strip_prefix("DEP_PYRE_MINIMAL_CURSES_LINK_ARG_") else {
            continue;
        };
        let Ok(index) = suffix.parse::<u32>() else {
            continue;
        };
        args.push((index, value));
    }
    args.sort_by_key(|(index, _)| *index);
    for (_, arg) in args {
        println!("cargo::rustc-link-arg-bins={arg}");
        println!("cargo::rustc-link-arg-tests={arg}");
        println!("cargo::rustc-link-arg-examples={arg}");
        println!("cargo::rustc-link-arg-benches={arg}");
    }
}

/// Embed `python.manifest` in the Windows executables, the way
/// `rpython/translator/platform/windows.py` links `pypy/goal/python.manifest`
/// with `/MANIFEST:EMBED /MANIFESTINPUT:`.
///
/// Its `longPathAware` setting is the one that carries weight here: without a
/// manifest the Win32 layer applies the MAX_PATH limit before the wide call
/// behind it runs, so `open()` on a 270-character name reports ENOENT while the
/// same name spelled `\\?\...` opens.
fn embed_windows_manifest(target: &str) {
    if target != "windows" || std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        return;
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("python.manifest");
    println!("cargo::rerun-if-changed={}", manifest.display());
    println!("cargo::rustc-link-arg-bins=/MANIFEST:EMBED");
    println!(
        "cargo::rustc-link-arg-bins=/MANIFESTINPUT:{}",
        manifest.display()
    );
}

/// Every `#[unsafe(no_mangle)]` item the cpyext layer defines.
///
/// Read out of the sources rather than repeated here: the list is the layer's
/// entire public ABI, and a hand-kept copy would silently stop exporting a new
/// entry point — which fails at `dlopen` time in a loaded extension, not at
/// build time.
fn cpyext_symbols() -> Vec<String> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../pyre-interpreter/src/cpyext")
        .canonicalize()
        .expect("the cpyext sources are part of the workspace");
    println!("cargo::rerun-if-changed={}", directory.display());
    let mut symbols = Vec::new();
    let mut sources: Vec<_> = std::fs::read_dir(&directory)
        .expect("read the cpyext source directory")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|suffix| suffix == "rs"))
        .collect();
    sources.sort();
    for source in sources {
        println!("cargo::rerun-if-changed={}", source.display());
        let text = std::fs::read_to_string(&source).expect("read a cpyext source file");
        let mut exported = false;
        for line in text.lines() {
            let line = line.trim();
            if line == "#[unsafe(no_mangle)]" {
                exported = true;
                continue;
            }
            if !exported || line.starts_with("///") || line.starts_with("//") {
                continue;
            }
            exported = false;
            if let Some(name) = declared_symbol(line) {
                symbols.push(name);
            }
        }
    }
    // A macro-generated static carries the attribute inside the expansion, so
    // the scan above cannot see it; each such family is named by the table its
    // macro is invoked with.
    symbols.extend(macro_generated_symbols(&directory));
    // The C-bodied entry points carry no attribute at all: this side declares
    // them rather than defining them, and the declarations are the one list.
    symbols.extend(declared_c_symbols(&directory));
    symbols.sort();
    symbols.dedup();
    symbols
}

/// The symbol a `pub ... fn NAME` / `pub static [mut] NAME` item declares.
fn declared_symbol(line: &str) -> Option<String> {
    let rest = line.strip_prefix("pub ")?;
    let rest = if let Some(rest) = rest.strip_prefix("static mut ") {
        rest
    } else if let Some(rest) = rest.strip_prefix("static ") {
        rest
    } else {
        let mut rest = rest;
        for keyword in ["unsafe ", "extern \"C\" ", "extern \"C-unwind\" "] {
            rest = rest.strip_prefix(keyword).unwrap_or(rest);
        }
        rest.strip_prefix("fn ")?
    };
    let name: String = rest
        .chars()
        .take_while(|character| character.is_alphanumeric() || *character == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// The entry points `cpyext/csources.rs` declares, whose bodies are the C
/// translation units `pyre-interpreter/build.rs` compiles.
fn declared_c_symbols(directory: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(directory.join("csources.rs"))
        .expect("read the cpyext C-source declarations");
    let Some(block) = text.split("unsafe extern \"C\" {").nth(1) else {
        return Vec::new();
    };
    let Some(block) = block.split("\n}").next() else {
        return Vec::new();
    };
    block
        .lines()
        .filter_map(|line| line.trim().strip_prefix("fn "))
        .filter_map(|rest| rest.split('(').next())
        .map(str::to_string)
        .collect()
}

/// The globals a table-driven macro declares: the `PyExc_*` pointers
/// (`pyerrors.rs`) and the `Py*_Type` blocks (`typeobject.rs`).
fn macro_generated_symbols(directory: &Path) -> Vec<String> {
    const TABLES: [(&str, &str); 2] = [
        ("pyerrors.rs", "exception_mirrors! {"),
        ("typeobject.rs", "type_mirrors! {"),
    ];
    let mut symbols = Vec::new();
    for (source, opening) in TABLES {
        let text = std::fs::read_to_string(directory.join(source))
            .expect("read a table-carrying cpyext source file");
        let Some(table) = text.split(opening).nth(1) else {
            continue;
        };
        let Some(table) = table.split("\n}").next() else {
            continue;
        };
        symbols.extend(
            table
                .lines()
                .filter_map(|line| line.trim().split(" =>").next())
                .filter(|name| {
                    !name.is_empty()
                        && name
                            .chars()
                            .all(|character| character.is_alphanumeric() || character == '_')
                })
                .map(str::to_string),
        );
    }
    symbols
}
