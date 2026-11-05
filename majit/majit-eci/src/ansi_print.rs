//! Port of `rpython/tool/ansi_print.py` — minimal `AnsiLogger`.
//!
//! Upstream's logger is "color-when-isatty, plain-otherwise"; the
//! color path is supplied by `py.io.ansi_print` (`:6 from py.io
//! import ansi_print`) and an interactive Mandelbrot dot driver
//! (`:7 from rpython.tool.ansi_mandelbrot import Driver`). Both are
//! presentation-only and orthogonal to log content.
//!
//! In the non-TTY branch upstream collapses to `ansi_print(text, ())`
//! at `:27`, where an empty `colors` tuple means "no ANSI escapes —
//! just `sys.stderr.write(text + '\n')`". Pyre tests run with stderr
//! captured and never observe color, so this minimal port mirrors
//! exactly that branch (plain stderr emission, no Mandelbrot dots).
//!
//! Re-introduce the `py.io.ansi_print` color path and the
//! `ansi_mandelbrot.Driver` once those upstream modules are ported.
//! Until then `dot(...)` is a no-op (`:69-77` upstream's `if not
//! isatty(): return` branch).

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};

/// RPython `class AnsiLogger(object)` (`ansi_print.py`).
///
/// Output shape `[name:subname] text` matches `_make_method` at
/// `ansi_print.py` for the non-TTY branch. The Rust port keeps
/// `name` as a `&'static str` because every call site so far names a
/// channel via a string literal (`"backendopt"`, `"c"`, …).
///
/// `output_disabled` is an `AtomicBool` rather than `bool` so that
/// upstream's runtime monkeypatch shape (`log.output_disabled = True`
/// at `rpython/rtyper/llinterp.py`,
/// `rpython/rtyper/test/test_llinterp.py:20-22`,
/// `rpython/memory/test/gc_test_base.py:38-41`,
/// `rpython/tool/test/test_ansi_print.py:95`) is reachable when
/// `AnsiLogger` lives behind `pub static LOG: AnsiLogger`. The atomic
/// is the smallest Rust-language adaptation for the Python
/// instance-attribute mutation; semantically `Ordering::Relaxed` is
/// sufficient because the flag races against logging output, not
/// against any other state.
pub struct AnsiLogger {
    /// RPython `self.name` (`:37`).
    pub name: &'static str,
    /// RPython `output_disabled = False` class attribute (`:34`).
    output_disabled: AtomicBool,
}

impl AnsiLogger {
    /// RPython `AnsiLogger.__init__(self, name)` (`:36-37`).
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            output_disabled: AtomicBool::new(false),
        }
    }

    /// Setter for `output_disabled`, mirroring upstream's
    /// `log.output_disabled = value` instance-attribute write.
    pub fn set_output_disabled(&self, value: bool) {
        self.output_disabled.store(value, Ordering::Relaxed);
    }

    fn is_disabled(&self) -> bool {
        self.output_disabled.load(Ordering::Relaxed)
    }

    /// Common emission path matching `_make_method`'s body at
    /// `ansi_print.py`. `subname_with_colon` is `""` for
    /// upstream entries whose `_make_method` first argument is the
    /// empty string (`__call__`, `event`, `red`, `bold`, `timeout`)
    /// and `":<methodname>"` for the rest.
    fn emit(&self, subname_with_colon: &str, text: &str) {
        if self.is_disabled() {
            return;
        }
        let _ = writeln!(
            std::io::stderr(),
            "[{}{}] {}",
            self.name,
            subname_with_colon,
            text
        );
    }

    /// RPython `event = _make_method('', (1,))` (`:44`).
    pub fn event(&self, text: &str) {
        self.emit("", text);
    }

    /// RPython `red = _make_method('', (31,))` (`:41`).
    pub fn red(&self, text: &str) {
        self.emit("", text);
    }

    /// RPython `bold = _make_method('', (1,))` (`:42`).
    pub fn bold(&self, text: &str) {
        self.emit("", text);
    }

    /// RPython `timeout = _make_method('', (1, 31))` (`:55`).
    pub fn timeout(&self, text: &str) {
        self.emit("", text);
    }

    /// RPython `WARNING = _make_method(':WARNING', (31,))` (`:43`).
    #[allow(non_snake_case)]
    pub fn WARNING(&self, text: &str) {
        self.emit(":WARNING", text);
    }

    /// RPython `ERROR = _make_method(':ERROR', (1, 31))` (`:45`).
    #[allow(non_snake_case)]
    pub fn ERROR(&self, text: &str) {
        self.emit(":ERROR", text);
    }

    /// RPython `Error = _make_method(':Error', (1, 31))` (`:46`).
    #[allow(non_snake_case)]
    pub fn Error(&self, text: &str) {
        self.emit(":Error", text);
    }

    /// RPython `info = _make_method(':info', (35,))` (`:47`).
    pub fn info(&self, text: &str) {
        self.emit(":info", text);
    }

    /// RPython `stub = _make_method(':stub', (34,))` (`:48`).
    pub fn stub(&self, text: &str) {
        self.emit(":stub", text);
    }

    /// RPython `call = _make_method(':call', (34,))` (`:51`).
    pub fn call(&self, text: &str) {
        self.emit(":call", text);
    }

    /// RPython `result = _make_method(':result', (34,))` (`:52`).
    pub fn result(&self, text: &str) {
        self.emit(":result", text);
    }

    /// RPython `exception = _make_method(':exception', (34,))` (`:53`).
    pub fn exception(&self, text: &str) {
        self.emit(":exception", text);
    }

    /// RPython `vpath = _make_method(':vpath', (35,))` (`:54`).
    pub fn vpath(&self, text: &str) {
        self.emit(":vpath", text);
    }

    /// RPython `__call__ = _make_method('', ())` (`:58`).
    ///
    /// Direct-call entry; outputs `[name] text` with no subname colon.
    /// Rust has no `Fn` impl on `&AnsiLogger` (would require unstable
    /// `Fn` trait), so upstream's `log(text)` syntax surfaces here as
    /// `LOG.plain(text)`. The output shape and side-effect set are
    /// identical to `event` / `red` / `bold` / `timeout` (all four
    /// also use `_make_method('', _)`); the parallel methods exist to
    /// preserve the upstream API surface so 1:1 callsite ports do not
    /// have to choose between aliases.
    pub fn plain(&self, text: &str) {
        self.emit("", text);
    }

    /// RPython `__getattr__(self, name)` (`:60-66`).
    ///
    /// Upstream's Python dispatches `log.<subname>(text)` through
    /// `__getattr__` to a freshly-installed method bound to
    /// `':<subname>'`. Rust has no dynamic method synthesis, so the
    /// subname is passed explicitly. Output shape `[name:subname]
    /// text` matches `_make_method(':<subname>', ())(self, text)` at
    /// `:17-30`.
    ///
    /// `assert!` mirrors upstream's `name[0].isalpha()` guard at
    /// `:62` end-to-end: upstream raises `AttributeError(name)` which
    /// is process-fatal unless the caller has a try/except. Rust's
    /// closest analogue is `panic!`; `debug_assert!` would silently
    /// drop the check in release, which would be a deviation because
    /// the strict-name invariant disappears at runtime. Use the
    /// always-on form instead.
    pub fn method(&self, subname: &str, text: &str) {
        assert!(
            subname
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic()),
            "AnsiLogger.method({subname:?}) — upstream `__getattr__` at \
             ansi_print.py:62 raises AttributeError when name[0] is not \
             alphabetic"
        );
        let with_colon = format!(":{subname}");
        self.emit(&with_colon, text);
    }

    /// RPython `dot(self)` (`:68-76`).
    ///
    /// Upstream emits a Mandelbrot character to stderr when stderr is
    /// a TTY; the non-TTY branch (`if not isatty(): return`) is a
    /// no-op. Pyre's port stays in the no-op branch until
    /// `rpython/tool/ansi_mandelbrot.py` lands.
    pub fn dot(&self) {}

    /// RPython `debug(self, info)` (`:78-79`).
    ///
    /// Upstream docstring: "For messages that are dropped. Can be
    /// monkeypatched in tests." The body is a literal pass.
    pub fn debug(&self, _info: &str) {}
}
