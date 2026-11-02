//! Port of `rpython/config/translationoption.py`.
//!
//! Upstream is a 408-LOC schema declaration file whose bulk
//! (lines 44-282) is a deeply nested `OptionDescription(children=[...])`
//! literal that enumerates every translation-time option
//! (`translation.backend`, `translation.gc`, `translation.jit`,
//! `translation.backendopt.inline_threshold`, …). The port mirrors the
//! structure line-by-line using the builder API ported in
//! [`super::config`]:
//!
//! - [`translation_optiondescription`] builds the top-level
//!   `translation` tree (upstream name-preserving).
//! - [`get_combined_translation_config`] mints a fresh `Config` around
//!   the tree (upstream `:284-314`).
//! - [`set_opt_level`] applies one of the `OPT_TABLE` level strings
//!   (`'0'`, `'1'`, `'size'`, `'mem'`, `'2'`, `'3'`, `'jit'`) to a
//!   freshly minted `Config` (upstream `:342-382`).
//! - [`_GLOBAL_TRANSLATIONCONFIG`] + [`get_translation_config`] mirror
//!   upstream's in-translation global (`:399-407`).
//!
//! **Rust adaptations** (each is the minimal line-for-line port):
//!
//! 1. **`IS_64_BITS`** is derived from `usize` width at compile time
//!    (upstream compares `sys.maxint`). Same observable value on every
//!    supported target.
//! 2. **`SUPPORT__THREAD`** is a `cfg!(target_os = "linux")` +
//!    aarch64-exclusion predicate. Upstream at `:23-31` hand-maintains
//!    the same Linux-only-and-no-aarch64 list of C compilers that
//!    accept `__thread`.
//! 3. **`CACHE_DIR`**: `realpath(join(MAINDIR, '_cache', MACHINE))`
//!    with `MAINDIR` = the `majit/` directory (parent of this crate's
//!    `CARGO_MANIFEST_DIR`). Analog of upstream `MAINDIR` /
//!    `CACHE_DIR` (`translationoption.py`): `MAINDIR =
//!    dirname(dirname(__file__))` is the `rpython/` package dir.
//!    `cache_dir()` is `os.path.realpath` of that join: canonicalize the
//!    longest existing ancestor, then append remaining components.
//! 4. **`ArbitraryOption` payload tests** (`instrumentctl`) avoid
//!    upstream's Python-dict storage — the Rust port's
//!    `OptionValue::Arbitrary(Rc<dyn Any>)` preserves the dynamic-type
//!    shape but requires an explicit type-check on read. No observable
//!    divergence for translation-option semantics.
//! 5. **`_GLOBAL_TRANSLATIONCONFIG`** is `RefCell<Option<Rc<Config>>>`.
//!    Upstream uses a module-global `None` / dict pair. The RefCell
//!    provides Python's mutate-through-module-reference semantics.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::config::{
    ArbitraryOption, BoolOption, Child, ChoiceOption, Config, ConfigError, DependencyEdge,
    FloatOption, IntOption, Option, OptionDescription, OptionValue, Owner, StrOption,
};
use crate::support::detect_number_of_processors;

// Upstream module-level constants. Only those consumed by the schema
// are ported; platform-specific constants follow upstream semantics.

/// Upstream `:11 DEFL_INLINE_THRESHOLD = 32.4` — "just enough to
/// inline `add__Int_Int()` and just small enough to prevent inlining
/// of some rlist functions."
pub const DEFL_INLINE_THRESHOLD: f64 = 32.4;

/// Upstream `:14 DEFL_PROF_BASED_INLINE_THRESHOLD = 32.4`.
pub const DEFL_PROF_BASED_INLINE_THRESHOLD: f64 = 32.4;

/// Upstream `:15 DEFL_LOW_INLINE_THRESHOLD = DEFL_INLINE_THRESHOLD / 2.0`.
pub const DEFL_LOW_INLINE_THRESHOLD: f64 = DEFL_INLINE_THRESHOLD / 2.0;

/// Upstream `:17 DEFL_GC = "incminimark"`.
pub const DEFL_GC: &str = "incminimark";

/// Upstream `:19 DEFL_ROOTFINDER_WITHJIT = "shadowstack"`.
pub const DEFL_ROOTFINDER_WITHJIT: &str = "shadowstack";

/// Upstream `:21 IS_64_BITS = sys.maxint > 2147483647`. The Rust port
/// checks `usize::BITS` at compile time.
pub const IS_64_BITS: bool = usize::BITS == 64;

/// Upstream `:23-31 SUPPORT__THREAD`. `True` on Linux x86_64 (and
/// other non-aarch64 Linux arches). False on macOS, Windows, and
/// aarch64 Linux per upstream's hand-maintained list.
pub const SUPPORT__THREAD: bool = cfg!(target_os = "linux") && !cfg!(target_arch = "aarch64");

/// `_stdlib_platform.machine()` (`translationoption.MACHINE`).
///
/// `platform.machine()` is `uname -m` on posix (`arm64` on Darwin
/// aarch64, `aarch64` on Linux aarch64, `x86_64` on posix x86_64) and
/// `PROCESSOR_ARCHITECTURE` on Windows (`AMD64` / `ARM64` / `x86`).
/// majit-config has no `libc` dependency, so the spelling is the
/// `(target_os, target_arch)` table for those values.
pub const fn machine() -> &'static str {
    if cfg!(all(windows, target_arch = "x86_64")) {
        "AMD64"
    } else if cfg!(all(windows, target_arch = "aarch64")) {
        "ARM64"
    } else if cfg!(all(windows, target_arch = "x86")) {
        "x86"
    } else if cfg!(all(windows, target_arch = "arm")) {
        "ARM"
    } else if cfg!(all(target_vendor = "apple", target_arch = "aarch64")) {
        "arm64"
    } else {
        std::env::consts::ARCH
    }
}

/// `MACHINE` (`translationoption.py`): `_stdlib_platform.machine()`.
pub const MACHINE: &str = machine();

/// `MAINDIR` (`translationoption.py`): `os.path.dirname(os.path.dirname(__file__))`,
/// the `rpython/` package dir. Analog is the `majit/` directory, parent of
/// this crate's `CARGO_MANIFEST_DIR`.
pub fn maindir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("majit-config crate lives under majit/")
        .to_path_buf()
}

/// `os.path.realpath`: canonicalize the longest existing ancestor, then
/// append remaining components (`translationoption.CACHE_DIR`).
pub fn realpath(path: &Path) -> PathBuf {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    let mut ancestor = path.clone();
    let mut missing = Vec::new();
    loop {
        if let Ok(canonical) = ancestor.canonicalize() {
            let mut resolved = canonical;
            for component in missing.iter().rev() {
                resolved.push(component);
            }
            return resolved;
        }
        match ancestor.file_name() {
            Some(name) => {
                missing.push(name.to_os_string());
                match ancestor.parent() {
                    Some(parent) if parent.as_os_str() != ancestor.as_os_str() => {
                        ancestor = parent.to_path_buf();
                    }
                    _ => return path,
                }
            }
            None => return path,
        }
    }
}

/// `CACHE_DIR` (`translationoption.py`).
///
/// `os.path.realpath(os.path.join(MAINDIR, '_cache', MACHINE))`.
pub fn cache_dir() -> PathBuf {
    realpath(&maindir().join("_cache").join(MACHINE))
}

/// Upstream `:39-42 PLATFORMS = ['host', 'arm']`. Kept as a module
/// constant so `ChoiceOption("platform", ...)` can reuse it.
pub const PLATFORMS: &[&str] = &["host", "arm"];

// Helpers that shape the upstream Python literal into the Rust builder
// API. Each upstream element is ported verbatim; the helpers exist
// only to tame the repetition.

/// Short-hand: `(path, value)` → [`DependencyEdge`]. Mirrors upstream's
/// tuple literal syntax `("translation.foo", value)` in
/// `requires=[...]` / `suggests=[...]` lists.
fn edge(path: &str, value: OptionValue) -> DependencyEdge {
    DependencyEdge::new(path, value)
}

fn str_edges<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a str)>) -> Vec<DependencyEdge> {
    pairs
        .into_iter()
        .map(|(p, v)| edge(p, OptionValue::Choice(v.to_string())))
        .collect()
}

fn opt<O: Option + 'static>(o: O) -> Child {
    Child::Option(Rc::new(o))
}

fn desc(d: OptionDescription) -> Child {
    Child::Description(Rc::new(d))
}

// Upstream `:44-282 translation_optiondescription = OptionDescription(
//     "translation", "Translation Options", [...])`

/// Port of upstream `translation_optiondescription` at
/// `translationoption.py`. Every leaf option / nested
/// description mirrors upstream's entry line-for-line; ordering is
/// preserved so reviewers can diff the two files.
fn translation_optiondescription() -> OptionDescription {
    // Upstream `:46-48`: continuation — requires lltype type_system.
    let continuation = BoolOption::new("continuation", "enable single-shot continuations")
        .with_default(false)
        .with_cmdline(Some("--continuation".to_string()))
        .with_requires(str_edges([("translation.type_system", "lltype")]));

    // Upstream `:49-50`: type_system — single-choice ["lltype"].
    let type_system = ChoiceOption::new(
        "type_system",
        "Type system to use when RTyping",
        vec!["lltype".to_string()],
    )
    .with_default("lltype")
    .with_cmdline(None);

    // Upstream `:51-56`: backend — single-choice ["c"].
    let backend = {
        let mut reqs: HashMap<String, Vec<DependencyEdge>> = HashMap::new();
        reqs.insert(
            "c".to_string(),
            str_edges([("translation.type_system", "lltype")]),
        );
        ChoiceOption::new(
            "backend",
            "Backend to use for code generation",
            vec!["c".to_string()],
        )
        .with_default("c")
        .with_requires(reqs)
        .with_cmdline(Some("-b --backend".to_string()))
    };

    // Upstream `:58-59`: shared.
    let shared = BoolOption::new("shared", "Build as a shared library")
        .with_default(false)
        .with_cmdline(Some("--shared".to_string()));

    // Upstream `:61-62`: log.
    let log = BoolOption::new(
        "log",
        "Include debug prints in the translation (PYPYLOG=...)",
    )
    .with_default(true)
    .with_cmdline(Some("--log".to_string()));

    // Upstream `:65-82`: gc — the big ChoiceOption.
    let gc = {
        let mut reqs: HashMap<String, Vec<DependencyEdge>> = HashMap::new();
        reqs.insert(
            "ref".to_string(),
            vec![
                edge("translation.rweakref", OptionValue::Bool(false)),
                edge(
                    "translation.gctransformer",
                    OptionValue::Choice("ref".to_string()),
                ),
            ],
        );
        reqs.insert(
            "none".to_string(),
            vec![
                edge("translation.rweakref", OptionValue::Bool(false)),
                edge(
                    "translation.gctransformer",
                    OptionValue::Choice("none".to_string()),
                ),
            ],
        );
        for strategy in &[
            "semispace",
            "statistics",
            "generation",
            "hybrid",
            "minimark",
            "incminimark",
        ] {
            reqs.insert(
                strategy.to_string(),
                str_edges([("translation.gctransformer", "framework")]),
            );
        }
        reqs.insert(
            "boehm".to_string(),
            vec![
                edge("translation.continuation", OptionValue::Bool(false)),
                edge(
                    "translation.gctransformer",
                    OptionValue::Choice("boehm".to_string()),
                ),
            ],
        );
        ChoiceOption::new(
            "gc",
            "Garbage Collection Strategy",
            [
                "boehm",
                "ref",
                "semispace",
                "statistics",
                "generation",
                "hybrid",
                "minimark",
                "incminimark",
                "none",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        )
        .with_default("ref")
        .with_requires(reqs)
        .with_cmdline(Some("--gc".to_string()))
    };

    // Upstream `:83-93`: gctransformer — internal.
    let gctransformer = {
        let mut reqs: HashMap<String, Vec<DependencyEdge>> = HashMap::new();
        let make = || -> Vec<DependencyEdge> {
            vec![
                edge(
                    "translation.gcrootfinder",
                    OptionValue::Choice("n/a".to_string()),
                ),
                edge("translation.gcremovetypeptr", OptionValue::Bool(false)),
            ]
        };
        reqs.insert("boehm".to_string(), make());
        reqs.insert("ref".to_string(), make());
        reqs.insert("none".to_string(), make());
        ChoiceOption::new(
            "gctransformer",
            "GC transformer that is used - internal",
            ["boehm", "ref", "framework", "none"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        )
        .with_default("ref")
        .with_cmdline(None)
        .with_requires(reqs)
    };

    // Upstream `:94-95`: gcremovetypeptr.
    let gcremovetypeptr =
        BoolOption::new("gcremovetypeptr", "Remove the typeptr from every object")
            .with_default(IS_64_BITS)
            .with_cmdline(Some("--gcremovetypeptr".to_string()));

    // Upstream `:96-103`: gcrootfinder.
    let gcrootfinder = {
        let mut reqs: HashMap<String, Vec<DependencyEdge>> = HashMap::new();
        reqs.insert(
            "shadowstack".to_string(),
            str_edges([("translation.gctransformer", "framework")]),
        );
        ChoiceOption::new(
            "gcrootfinder",
            "Strategy for finding GC Roots (framework GCs only)",
            vec!["n/a".to_string(), "shadowstack".to_string()],
        )
        .with_default("shadowstack")
        .with_cmdline(Some("--gcrootfinder".to_string()))
        .with_requires(reqs)
    };

    // Upstream `:106-107`: thread.
    let thread = BoolOption::new("thread", "enable use of threading primitives")
        .with_default(false)
        .with_cmdline(Some("--thread".to_string()));

    // Upstream `:108-113`: sandbox.
    let sandbox = BoolOption::new("sandbox", "Produce a fully-sandboxed executable")
        .with_default(false)
        .with_cmdline(Some("--sandbox".to_string()))
        .with_suggests(vec![
            edge(
                "translation.gc",
                OptionValue::Choice("generation".to_string()),
            ),
            edge(
                "translation.gcrootfinder",
                OptionValue::Choice("shadowstack".to_string()),
            ),
            edge("translation.thread", OptionValue::Bool(false)),
        ]);

    // Upstream `:114-115`: rweakref.
    let rweakref = BoolOption::new("rweakref", "The backend supports RPython-level weakrefs")
        .with_default(true);

    // Upstream `:118-122`: jit.
    let jit = BoolOption::new("jit", "generate a JIT")
        .with_default(false)
        .with_suggests(vec![
            edge("translation.gc", OptionValue::Choice(DEFL_GC.to_string())),
            edge(
                "translation.gcrootfinder",
                OptionValue::Choice(DEFL_ROOTFINDER_WITHJIT.to_string()),
            ),
            edge(
                "translation.list_comprehension_operations",
                OptionValue::Bool(true),
            ),
        ]);

    // Upstream `:123-125`: jit_backend.
    let jit_backend = ChoiceOption::new(
        "jit_backend",
        "choose the backend for the JIT",
        ["auto", "x86", "x86-without-sse2", "arm"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    )
    .with_default("auto")
    .with_cmdline(Some("--jit-backend".to_string()));

    // Upstream `:126-128`: jit_profiler.
    let jit_profiler = ChoiceOption::new(
        "jit_profiler",
        "integrate profiler support into the JIT",
        ["off", "oprofile"].iter().map(|s| s.to_string()).collect(),
    )
    .with_default("off");

    // Upstream `:129-131`: check_str_without_nul.
    let check_str_without_nul = BoolOption::new(
        "check_str_without_nul",
        "Forbid NUL chars in strings in some external function calls",
    )
    .with_default(false)
    .with_cmdline(None);

    // Upstream `:134-135`: verbose.
    let verbose = BoolOption::new("verbose", "Print extra information")
        .with_default(false)
        .with_cmdline(Some("--verbose".to_string()));

    // Upstream `:136`: cc.
    let cc = StrOption::new("cc", "Specify compiler to use for compiling generated C")
        .with_cmdline(Some("--cc".to_string()));

    // Upstream `:137-138`: profopt.
    let profopt = BoolOption::new(
        "profopt",
        "Enable profile guided optimization. Defaults to enabling this for PyPy. \
         For other training workloads, please specify them in profoptargs",
    )
    .with_default(false)
    .with_cmdline(Some("--profopt".to_string()));

    // Upstream `:139`: profoptargs.
    let profoptargs = StrOption::new(
        "profoptargs",
        "Absolute path to the profile guided optimization training script + the \
         necessary arguments of the script",
    )
    .with_cmdline(Some("--profoptargs".to_string()));

    // Upstream `:140-141`: instrument.
    let instrument = BoolOption::new("instrument", "internal: turn instrumentation on")
        .with_default(false)
        .with_cmdline(None);

    // Upstream `:142-143`: countmallocs.
    let countmallocs = BoolOption::new("countmallocs", "Count mallocs and frees")
        .with_default(false)
        .with_cmdline(None);

    // Upstream `:144-145`: countfieldaccess.
    let countfieldaccess =
        BoolOption::new("countfieldaccess", "Count field access for C structs").with_default(false);

    // Upstream `:146-150`: fork_before.
    let fork_before = ChoiceOption::new(
        "fork_before",
        "(UNIX) Create restartable checkpoint before step",
        [
            "annotate",
            "rtype",
            "backendopt",
            "database",
            "source",
            "pyjitpl",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
    )
    .with_cmdline(Some("--fork-before".to_string()));

    // Upstream `:151-154`: dont_write_c_files.
    let dont_write_c_files = BoolOption::new(
        "dont_write_c_files",
        "Make the C backend write everyting to /dev/null. Useful for benchmarking, \
         so you don't actually involve the disk",
    )
    .with_default(false)
    .with_cmdline(Some("--dont-write-c-files".to_string()));

    // Upstream `:155-156`: instrumentctl — ArbitraryOption(default=None).
    let instrumentctl =
        ArbitraryOption::new("instrumentctl", "internal").with_default(OptionValue::None);

    // Upstream `:157`: output.
    let output =
        StrOption::new("output", "Output file name").with_cmdline(Some("--output".to_string()));

    // Upstream `:158-160`: secondaryentrypoints.
    let secondaryentrypoints = StrOption::new(
        "secondaryentrypoints",
        "Comma separated list of keys choosing secondary entrypoints",
    )
    .with_cmdline(Some("--entrypoints".to_string()))
    .with_default("main");

    // Upstream `:162-164`: dump_static_data_info.
    let dump_static_data_info = BoolOption::new("dump_static_data_info", "Dump static data info")
        .with_cmdline(Some("--dump_static_data_info".to_string()))
        .with_default(false)
        .with_requires(str_edges([("translation.backend", "c")]));

    // Upstream `:167-170`: no__thread. Upstream identifier has a double
    // underscore — preserved for grep parity.
    #[allow(non_snake_case)]
    let no__thread = BoolOption::new("no__thread", "don't use __thread for implementing TLS")
        .with_default(!SUPPORT__THREAD)
        .with_cmdline(Some("--no__thread".to_string()))
        .with_negation(false);

    // Upstream `:171-173`: make_jobs.
    let make_jobs = IntOption::new(
        "make_jobs",
        "Specify -j argument to make for compilation (C backend only)",
    )
    .with_cmdline(Some("--make-jobs".to_string()))
    .with_default(detect_number_of_processors());

    // Upstream `:176-181`: list_comprehension_operations.
    let list_comprehension_operations = BoolOption::new(
        "list_comprehension_operations",
        "When true, look for and special-case the sequence of operations that \
         results from a list comprehension and attempt to pre-allocate the list",
    )
    .with_default(false)
    .with_cmdline(Some("--listcompr".to_string()));

    // Upstream `:182-184`: withsmallfuncsets.
    let withsmallfuncsets = IntOption::new(
        "withsmallfuncsets",
        "Represent groups of less funtions than this as indices into an array",
    )
    .with_default(0);

    // Upstream `:185-188`: taggedpointers.
    let taggedpointers = BoolOption::new(
        "taggedpointers",
        "When true, enable the use of tagged pointers. If false, use normal boxing",
    )
    .with_default(false);

    // Upstream `:189-192`: keepgoing.
    let keepgoing = BoolOption::new(
        "keepgoing",
        "Continue annotating when errors are encountered, and report them all at \
         the end of the annotation phase",
    )
    .with_default(false)
    .with_cmdline(Some("--keepgoing".to_string()));

    // Upstream `:193-195`: lldebug.
    let lldebug = BoolOption::new("lldebug", "If true, makes an lldebug build")
        .with_default(false)
        .with_cmdline(Some("--lldebug".to_string()));

    // Upstream `:196-198`: lldebug0.
    let lldebug0 = BoolOption::new("lldebug0", "If true, makes an lldebug0 build")
        .with_default(false)
        .with_cmdline(Some("--lldebug0".to_string()));

    // Upstream `:199-201`: lto.
    let lto = BoolOption::new("lto", "enable link time optimization")
        .with_default(false)
        .with_cmdline(Some("--lto".to_string()))
        .with_requires(str_edges([("translation.gcrootfinder", "shadowstack")]));

    // Upstream `:202`: icon.
    let icon = StrOption::new(
        "icon",
        "Path to the (Windows) icon to use for the executable",
    );
    // Upstream `:203-204`: manifest.
    let manifest = StrOption::new(
        "manifest",
        "Path to the (Windows) manifest to embed in the executable",
    );
    // Upstream `:205-206`: libname.
    let libname = StrOption::new(
        "libname",
        "Windows: name and possibly location of the lib file to create",
    );

    // Upstream `:208-262`: backendopt — nested description.
    let backendopt_children: Vec<Child> = vec![
        // :210-211: inline.
        opt(BoolOption::new("inline", "Do basic inlining and malloc removal").with_default(true)),
        // :212-213: inline_threshold.
        opt(FloatOption::new(
            "inline_threshold",
            "Threshold when to inline functions",
        )
        .with_default(DEFL_INLINE_THRESHOLD)
        .with_cmdline(Some("--inline-threshold".to_string()))),
        // :214-217: inline_heuristic.
        opt(
            StrOption::new("inline_heuristic", "Dotted name of an heuristic function for inlining")
                .with_default("rpython.translator.backendopt.inline.inlining_heuristic")
                .with_cmdline(Some("--inline-heuristic".to_string())),
        ),
        // :219-220: print_statistics.
        opt(
            BoolOption::new("print_statistics", "Print statistics while optimizing")
                .with_default(false),
        ),
        // :221-222: merge_if_blocks.
        opt(BoolOption::new("merge_if_blocks", "Merge if ... elif chains")
            .with_cmdline(Some("--if-block-merge".to_string()))
            .with_default(true)),
        // :223: mallocs.
        opt(BoolOption::new("mallocs", "Remove mallocs").with_default(true)),
        // :224-225: constfold.
        opt(BoolOption::new("constfold", "Constant propagation").with_default(true)),
        // :227-230: profile_based_inline.
        opt(StrOption::new(
            "profile_based_inline",
            "Use call count profiling to drive inlining, specify arguments",
        )),
        // :231-235: profile_based_inline_threshold.
        opt(FloatOption::new(
            "profile_based_inline_threshold",
            "Threshold when to inline functions for profile based inlining",
        )
        .with_default(DEFL_PROF_BASED_INLINE_THRESHOLD)),
        // :236-240: profile_based_inline_heuristic.
        opt(StrOption::new(
            "profile_based_inline_heuristic",
            "Dotted name of an heuristic function for profile based inlining",
        )
        .with_default("rpython.translator.backendopt.inline.inlining_heuristic")),
        // :242-245: remove_asserts.
        opt(BoolOption::new(
            "remove_asserts",
            "Remove operations that look like 'raise AssertionError', which lets the C optimizer remove the asserts",
        )
        .with_default(false)),
        // :246-249: really_remove_asserts.
        opt(BoolOption::new(
            "really_remove_asserts",
            "Really remove operations that look like 'raise AssertionError', without relying on the C compiler",
        )
        .with_default(false)),
        // :251: storesink.
        opt(BoolOption::new("storesink", "Perform store sinking").with_default(true)),
        // :252-254: replace_we_are_jitted.
        opt(
            BoolOption::new("replace_we_are_jitted", "Replace we_are_jitted() calls by False")
                .with_default(false)
                .with_cmdline(None),
        ),
        // :255-261: none (no-backendopts aggregate flag).
        opt(BoolOption::new("none", "Do not run any backend optimizations").with_requires(
            vec![
                edge("translation.backendopt.inline", OptionValue::Bool(false)),
                edge(
                    "translation.backendopt.inline_threshold",
                    OptionValue::Int(0),
                ),
                edge(
                    "translation.backendopt.merge_if_blocks",
                    OptionValue::Bool(false),
                ),
                edge("translation.backendopt.mallocs", OptionValue::Bool(false)),
                edge(
                    "translation.backendopt.constfold",
                    OptionValue::Bool(false),
                ),
            ],
        )),
    ];
    let backendopt = OptionDescription::new(
        "backendopt",
        "Backend Optimization Options",
        backendopt_children,
    );

    // Upstream `:264-268`: platform.
    let platform = {
        let mut platforms_values: Vec<String> = vec!["host".to_string()];
        platforms_values.extend(PLATFORMS.iter().map(|s| s.to_string()));
        let mut sugs: HashMap<String, Vec<DependencyEdge>> = HashMap::new();
        sugs.insert(
            "arm".to_string(),
            vec![
                edge(
                    "translation.gcrootfinder",
                    OptionValue::Choice("shadowstack".to_string()),
                ),
                edge(
                    "translation.jit_backend",
                    OptionValue::Choice("arm".to_string()),
                ),
            ],
        );
        ChoiceOption::new("platform", "target platform", platforms_values)
            .with_default("host")
            .with_cmdline(Some("--platform".to_string()))
            .with_suggests(sugs)
    };

    // Upstream `:270-271`: split_gc_address_space.
    let split_gc_address_space = BoolOption::new(
        "split_gc_address_space",
        "Ensure full separation of GC and non-GC pointers",
    )
    .with_default(false);

    // Upstream `:272-278`: reverse_debugger.
    let reverse_debugger = BoolOption::new(
        "reverse_debugger",
        "Give an executable that writes a log file for reverse debugging",
    )
    .with_default(false)
    .with_cmdline(Some("--revdb".to_string()))
    .with_requires(vec![
        edge(
            "translation.split_gc_address_space",
            OptionValue::Bool(true),
        ),
        edge("translation.jit", OptionValue::Bool(false)),
        edge("translation.gc", OptionValue::Choice("boehm".to_string())),
        edge("translation.continuation", OptionValue::Bool(false)),
    ]);

    // Upstream `:279-281`: rpython_translate.
    let rpython_translate = BoolOption::new(
        "rpython_translate",
        "Set to true by rpython/bin/rpython and translate.py",
    )
    .with_default(false);

    // Upstream ordering preserved — see `:44-282`.
    let children: Vec<Child> = vec![
        opt(continuation),
        opt(type_system),
        opt(backend),
        opt(shared),
        opt(log),
        opt(gc),
        opt(gctransformer),
        opt(gcremovetypeptr),
        opt(gcrootfinder),
        opt(thread),
        opt(sandbox),
        opt(rweakref),
        opt(jit),
        opt(jit_backend),
        opt(jit_profiler),
        opt(check_str_without_nul),
        opt(verbose),
        opt(cc),
        opt(profopt),
        opt(profoptargs),
        opt(instrument),
        opt(countmallocs),
        opt(countfieldaccess),
        opt(fork_before),
        opt(dont_write_c_files),
        opt(instrumentctl),
        opt(output),
        opt(secondaryentrypoints),
        opt(dump_static_data_info),
        opt(no__thread),
        opt(make_jobs),
        opt(list_comprehension_operations),
        opt(withsmallfuncsets),
        opt(taggedpointers),
        opt(keepgoing),
        opt(lldebug),
        opt(lldebug0),
        opt(lto),
        opt(icon),
        opt(manifest),
        opt(libname),
        desc(backendopt),
        opt(platform),
        opt(split_gc_address_space),
        opt(reverse_debugger),
        opt(rpython_translate),
    ];

    OptionDescription::new("translation", "Translation Options", children)
}

// Upstream `:284-314 get_combined_translation_config`.

/// Upstream `get_combined_translation_config(other_optdescr=None,
/// existing_config=None, overrides=None, translating=False)` at
/// `:284-314`.
///
/// The `other_optdescr` / `existing_config` parameters are the hook
/// for PyPy-side callers to compose their own option tree above the
/// RPython translation tree. All four arguments are `Option`al,
/// mirroring upstream's defaults of `None` / `None` / `None` / `False`.
pub fn get_combined_translation_config(
    other_optdescr: std::option::Option<OptionDescription>,
    existing_config: std::option::Option<&Rc<Config>>,
    overrides: std::option::Option<HashMap<String, OptionValue>>,
    translating: bool,
) -> Result<Rc<Config>, ConfigError> {
    let overrides = overrides.unwrap_or_default();

    // Upstream `:290-292`: the `translating` indicator option itself.
    let translating_opt = BoolOption::new(
        "translating",
        "indicates whether we are translating currently",
    )
    .with_default(false)
    .with_cmdline(None);

    // Upstream `:293-298`.
    let (mut children, newname): (Vec<Child>, String) = match other_optdescr {
        Some(d) => {
            let name = d._name.clone();
            (vec![desc(d)], name)
        }
        None => (Vec::new(), String::new()),
    };

    // Upstream `:299-303`.
    if let Some(existing) = existing_config {
        // Upstream copies the children of `existing_config._cfgimpl_descr`
        // excluding the one that collides with `other_optdescr._name`.
        for child in existing._cfgimpl_descr._children.iter() {
            if child.name() == newname {
                continue;
            }
            children.push(child.clone());
        }
    } else {
        children.push(opt(translating_opt));
        children.push(desc(translation_optiondescription()));
    }

    let descr = Rc::new(OptionDescription::new("pypy", "all options", children));
    let config = Config::new(descr, overrides)?;

    // Upstream `:306-307`.
    if translating {
        config.set_value("translating", OptionValue::Bool(true))?;
    }

    // Upstream `:308-313`.
    if let Some(existing) = existing_config {
        for child in existing._cfgimpl_descr._children.iter() {
            let name = child.name();
            if name == newname {
                continue;
            }
            // Upstream `:312`: `value = getattr(existing_config, child._name)`.
            let value = existing.get(name)?;
            // Upstream `:313`:
            // `config._cfgimpl_values[child._name] = value`.
            //
            // Routed through [`Config::_cfgimpl_set_raw`] which
            // mirrors PyPy's bare dict write line-by-line: no owner
            // change, no validation, no requires/suggests cascade.
            // `setoption(_, _, Owner::User)` would diverge by promoting
            // the slot's owner to `User`, which downstream owner-
            // aware readers (`config.py:553` / `:572`) would then see
            // as a user-set value rather than an inherited default.
            config._cfgimpl_set_raw(name, value)?;
        }
    }

    Ok(config)
}

// Upstream `:316-340 OPT_LEVELS / OPT_TABLE / OPT_TABLE_DOC`.

/// Upstream `:318 OPT_LEVELS`.
pub const OPT_LEVELS: &[&str] = &["0", "1", "size", "mem", "2", "3", "jit"];

/// Upstream `:319 DEFAULT_OPT_LEVEL = '2'`.
pub const DEFAULT_OPT_LEVEL: &str = "2";

/// Upstream `:321-329 OPT_TABLE_DOC`. Returned by function rather than
/// top-level `HashMap` so the order is deterministic and the table is
/// lazily materialised.
#[cfg(test)]
fn opt_table_doc() -> Vec<(&'static str, &'static str)> {
    vec![
        ("0", "No optimization.  Uses the Boehm GC."),
        (
            "1",
            "Enable a default set of optimizations.  Uses the Boehm GC.",
        ),
        (
            "size",
            "Optimize for the size of the executable.  Uses the Boehm GC.",
        ),
        (
            "mem",
            "Optimize for run-time memory usage and use a memory-saving GC.",
        ),
        (
            "2",
            "Enable most optimizations and use a high-performance GC.",
        ),
        (
            "3",
            "Enable all optimizations and use a high-performance GC.",
        ),
        ("jit", "Enable the JIT."),
    ]
}

/// Upstream `:331-340 OPT_TABLE`. Each entry's right-hand side is a
/// whitespace-separated string of `gc` + backendopt words.
fn opt_table_entry(level: &str) -> std::option::Option<&'static str> {
    match level {
        "0" => Some("boehm nobackendopt"),
        "1" => Some("boehm lowinline"),
        "size" => Some("boehm lowinline remove_asserts"),
        "mem" => Some("incminimark lowinline remove_asserts removetypeptr"),
        "2" => Some("incminimark extraopts"),
        "3" => Some("incminimark extraopts remove_asserts"),
        "jit" => Some("incminimark extraopts jit"),
        _ => None,
    }
}

// Upstream `:342-382 set_opt_level`.

/// Upstream `set_opt_level(config, level)` at `:342-382`. Applies
/// optimisation suggestions on the `translation` subgroup of the
/// supplied `config`, keyed on one of [`OPT_LEVELS`].
pub fn set_opt_level(config: &Rc<Config>, level: &str) -> Result<(), ConfigError> {
    // Upstream `:346-349`.
    let opts = opt_table_entry(level)
        .ok_or_else(|| ConfigError::Generic(format!("no such optimization level: {:?}", level)))?;

    // Upstream `:350-351`.
    let mut words: Vec<&str> = opts.split_ascii_whitespace().collect();
    let gc_word = words.remove(0);

    // Upstream `:353-356`:
    //     if config.translation._cfgimpl_value_owners['gc'] != 'suggested':
    //         config.translation.suggest(gc=gc)
    let translation = match config.get("translation")? {
        crate::config::ConfigValue::SubConfig(c) => c,
        _ => {
            return Err(ConfigError::Generic(
                "set_opt_level: `translation` subgroup not found".to_string(),
            ));
        }
    };
    let gc_owner = *translation
        ._cfgimpl_value_owners()
        .get("gc")
        .unwrap_or(&Owner::Default);
    if gc_owner != Owner::Suggested {
        let mut kwargs = HashMap::new();
        kwargs.insert("gc".to_string(), OptionValue::Choice(gc_word.to_string()));
        translation.suggest(kwargs)?;
    }

    // Upstream `:358-374`: the backendopt words.
    for word in &words {
        match *word {
            "nobackendopt" => {
                let backendopt = match translation.get("backendopt")? {
                    crate::config::ConfigValue::SubConfig(c) => c,
                    _ => {
                        return Err(ConfigError::Generic(
                            "set_opt_level: `translation.backendopt` subgroup not found"
                                .to_string(),
                        ));
                    }
                };
                let mut kwargs = HashMap::new();
                kwargs.insert("none".to_string(), OptionValue::Bool(true));
                backendopt.suggest(kwargs)?;
            }
            "lowinline" => {
                let backendopt = match translation.get("backendopt")? {
                    crate::config::ConfigValue::SubConfig(c) => c,
                    _ => {
                        return Err(ConfigError::Generic(
                            "set_opt_level: `translation.backendopt` subgroup not found"
                                .to_string(),
                        ));
                    }
                };
                let mut kwargs = HashMap::new();
                kwargs.insert(
                    "inline_threshold".to_string(),
                    OptionValue::Float(DEFL_LOW_INLINE_THRESHOLD),
                );
                backendopt.suggest(kwargs)?;
            }
            "remove_asserts" => {
                let backendopt = match translation.get("backendopt")? {
                    crate::config::ConfigValue::SubConfig(c) => c,
                    _ => {
                        return Err(ConfigError::Generic(
                            "set_opt_level: `translation.backendopt` subgroup not found"
                                .to_string(),
                        ));
                    }
                };
                let mut kwargs = HashMap::new();
                kwargs.insert("remove_asserts".to_string(), OptionValue::Bool(true));
                backendopt.suggest(kwargs)?;
            }
            "extraopts" => {
                let mut kwargs = HashMap::new();
                kwargs.insert("withsmallfuncsets".to_string(), OptionValue::Int(5));
                translation.suggest(kwargs)?;
            }
            "jit" => {
                let mut kwargs = HashMap::new();
                kwargs.insert("jit".to_string(), OptionValue::Bool(true));
                translation.suggest(kwargs)?;
            }
            "removetypeptr" => {
                let mut kwargs = HashMap::new();
                kwargs.insert("gcremovetypeptr".to_string(), OptionValue::Bool(true));
                translation.suggest(kwargs)?;
            }
            other => {
                // Upstream `:374`: `raise ValueError(word)`.
                return Err(ConfigError::Generic(other.to_string()));
            }
        }
    }

    // Upstream `:377-378`:
    //     config.translation.suggest(list_comprehension_operations=True)
    let mut kwargs = HashMap::new();
    kwargs.insert(
        "list_comprehension_operations".to_string(),
        OptionValue::Bool(true),
    );
    translation.suggest(kwargs)?;

    // Upstream `:381-382`:
    //     config.translation.gc = config.translation.gc
    // This forces the gc choice through its setoption path, which
    // propagates `_requires` into effect. Without this, the suggested
    // gc is stored but its dependent options (gctransformer,
    // rweakref) may not have been updated yet.
    let current_gc = match translation.get("gc")? {
        crate::config::ConfigValue::Value(v) => v,
        _ => {
            return Err(ConfigError::Generic(
                "set_opt_level: `translation.gc` is not a leaf option".to_string(),
            ));
        }
    };
    translation.set_value("gc", current_gc)?;

    Ok(())
}

// Upstream `:399-407 _GLOBAL_TRANSLATIONCONFIG` / `get_translation_config`.

thread_local! {
    /// Upstream `:401 _GLOBAL_TRANSLATIONCONFIG = None`. The Rust port
    /// stores it in a `thread_local! RefCell<Option<Rc<Config>>>` so
    /// that tests can seed and clear per-thread without cross-test
    /// interference.
    pub static _GLOBAL_TRANSLATIONCONFIG: RefCell<std::option::Option<Rc<Config>>> =
        const { RefCell::new(None) };
}

/// Upstream `get_translation_config()` at `:404-407`.
pub fn get_translation_config() -> std::option::Option<Rc<Config>> {
    _GLOBAL_TRANSLATIONCONFIG.with(|slot| slot.borrow().clone())
}

/// Upstream patches the module global at translation-time via direct
/// attribute write. Mirror with a `set` helper that the driver can
/// call. Private-ish (no upstream symbol) but necessary for the
/// `thread_local!` storage pattern.
fn _set_translation_config(config: std::option::Option<Rc<Config>>) {
    _GLOBAL_TRANSLATIONCONFIG.with(|slot| {
        *slot.borrow_mut() = config;
    });
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_combined() -> Rc<Config> {
        get_combined_translation_config(None, None, None, false)
            .expect("combined translation config must build")
    }

    #[test]
    fn cache_dir_ends_with_cache_machine_under_maindir() {
        let expected = maindir().join("_cache").join(MACHINE);
        let dir = cache_dir();
        assert_eq!(dir, realpath(&expected));
        assert!(
            dir.ends_with(std::path::Path::new("_cache").join(MACHINE)),
            "cache_dir() must end with _cache/{MACHINE} under maindir(), got {}",
            dir.display()
        );
        let main = maindir();
        let main_canon = main.canonicalize().unwrap_or_else(|_| main.clone());
        assert!(
            dir.starts_with(&main) || dir.starts_with(&main_canon),
            "cache_dir() must sit under maindir() {}, got {}",
            main.display(),
            dir.display()
        );
    }

    #[test]
    fn machine_uses_platform_machine_spelling() {
        assert_eq!(MACHINE, machine());
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert_eq!(MACHINE, "arm64");
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        assert_eq!(MACHINE, "aarch64");
        #[cfg(all(unix, not(target_os = "macos"), target_arch = "x86_64"))]
        assert_eq!(MACHINE, "x86_64");
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        assert_eq!(MACHINE, "x86_64");
        #[cfg(all(windows, target_arch = "x86_64"))]
        assert_eq!(MACHINE, "AMD64");
        #[cfg(all(windows, target_arch = "aarch64"))]
        assert_eq!(MACHINE, "ARM64");
        #[cfg(all(windows, target_arch = "x86"))]
        assert_eq!(MACHINE, "x86");
    }

    #[cfg(unix)]
    #[test]
    fn realpath_resolves_symlinked_parent_with_missing_leaf() {
        let root = std::env::temp_dir().join(format!(
            "majit-realpath-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let real_parent = root.join("real-parent");
        std::fs::create_dir(&real_parent).unwrap();
        let link_parent = root.join("link-parent");
        std::os::unix::fs::symlink(&real_parent, &link_parent).unwrap();
        let missing_leaf = link_parent.join("missing-leaf");
        assert!(!missing_leaf.exists());
        let got = realpath(&missing_leaf);
        let expected = real_parent.canonicalize().unwrap().join("missing-leaf");
        assert_eq!(got, expected);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn translation_tree_has_expected_leaf_count() {
        // Upstream `:44-282`: 45 immediate children + `backendopt`
        // nested group (1 description) = 46 top-level children on the
        // `translation` subgroup. Pin the count so any accidental add
        // or drop of an option surfaces here.
        let desc = translation_optiondescription();
        let top_children = &desc._children;
        assert_eq!(
            top_children.len(),
            46,
            "translation subgroup child count (upstream `translationoption.py:44-282`)"
        );
    }

    #[test]
    fn combined_config_seeds_translating_false_by_default() {
        let config = fresh_combined();
        match config.get("translating").expect("translating leaf") {
            crate::config::ConfigValue::Value(OptionValue::Bool(v)) => {
                assert!(!v, "upstream `:290-292` default=False");
            }
            other => panic!("translating should be a Bool leaf, got {:?}", other),
        }
    }

    #[test]
    fn combined_config_with_translating_true_sets_leaf() {
        let config = get_combined_translation_config(None, None, None, true).expect("combined");
        match config.get("translating").expect("translating leaf") {
            crate::config::ConfigValue::Value(OptionValue::Bool(v)) => {
                assert!(v, "upstream `:306-307` sets translating=True");
            }
            other => panic!("translating should be a Bool leaf, got {:?}", other),
        }
    }

    #[test]
    fn translation_verbose_defaults_to_false() {
        let config = fresh_combined();
        match config.get("translation.verbose").expect("verbose") {
            crate::config::ConfigValue::Value(OptionValue::Bool(v)) => {
                assert!(!v, "upstream `:134-135` default=False");
            }
            other => panic!("verbose should be Bool leaf, got {:?}", other),
        }
    }

    #[test]
    fn translation_gc_defaults_to_ref_choice() {
        let config = fresh_combined();
        match config.get("translation.gc").expect("gc") {
            crate::config::ConfigValue::Value(OptionValue::Choice(s)) => {
                assert_eq!(s, "ref", "upstream `:65-82` default=ref");
            }
            other => panic!("gc should be Choice leaf, got {:?}", other),
        }
    }

    #[test]
    fn translation_backend_defaults_to_c() {
        let config = fresh_combined();
        match config.get("translation.backend").expect("backend") {
            crate::config::ConfigValue::Value(OptionValue::Choice(s)) => {
                assert_eq!(s, "c", "upstream `:51-56` default=c");
            }
            other => panic!("backend should be Choice leaf, got {:?}", other),
        }
    }

    #[test]
    fn backendopt_inline_threshold_defaults_to_defl() {
        let config = fresh_combined();
        match config
            .get("translation.backendopt.inline_threshold")
            .expect("inline_threshold")
        {
            crate::config::ConfigValue::Value(OptionValue::Float(v)) => {
                assert_eq!(v, DEFL_INLINE_THRESHOLD);
            }
            other => panic!("inline_threshold should be Float leaf, got {:?}", other),
        }
    }

    #[test]
    fn backendopt_nested_description_is_present() {
        let config = fresh_combined();
        match config
            .get("translation.backendopt")
            .expect("backendopt subgroup")
        {
            crate::config::ConfigValue::SubConfig(sub) => {
                // `none` option inside the nested description.
                match sub.get("none").expect("backendopt.none") {
                    crate::config::ConfigValue::Value(OptionValue::None) => {
                        // upstream `:255-261` has no default → None sentinel.
                    }
                    other => panic!("backendopt.none default should be None, got {:?}", other),
                }
            }
            _ => panic!("translation.backendopt should be a SubConfig"),
        }
    }

    #[test]
    fn set_opt_level_zero_suggests_boehm_and_nobackendopt() {
        let config = fresh_combined();
        set_opt_level(&config, "0").expect("opt level 0");
        // gc should now be "boehm" (suggested).
        match config.get("translation.gc").expect("gc") {
            crate::config::ConfigValue::Value(OptionValue::Choice(s)) => {
                assert_eq!(s, "boehm", "OPT_TABLE['0'] suggests boehm");
            }
            _ => panic!("unexpected gc shape"),
        }
        // backendopt.none should now be true.
        match config.get("translation.backendopt.none").expect("none") {
            crate::config::ConfigValue::Value(OptionValue::Bool(v)) => {
                assert!(v, "nobackendopt word must set backendopt.none=True");
            }
            _ => panic!("unexpected backendopt.none shape"),
        }
    }

    #[test]
    fn set_opt_level_rejects_unknown_level() {
        let config = fresh_combined();
        let result = set_opt_level(&config, "nope");
        assert!(
            matches!(result, Err(ConfigError::Generic(_))),
            "unknown level must be Generic ConfigError, got {:?}",
            result
        );
    }

    #[test]
    fn translation_config_global_starts_unset() {
        // Per-thread by virtue of thread_local!, so this is deterministic.
        _set_translation_config(None);
        assert!(get_translation_config().is_none());
    }

    #[test]
    fn translation_config_global_roundtrip() {
        let config = fresh_combined();
        _set_translation_config(Some(Rc::clone(&config)));
        let got = get_translation_config().expect("set");
        assert!(Rc::ptr_eq(&got, &config));
        _set_translation_config(None);
    }

    #[test]
    fn opt_table_doc_enumerates_all_opt_levels() {
        // Every OPT_LEVELS entry must have a doc entry.
        let docs = opt_table_doc();
        for level in OPT_LEVELS {
            assert!(
                docs.iter().any(|(k, _)| k == level),
                "opt_table_doc missing entry for {}",
                level
            );
        }
    }
}
