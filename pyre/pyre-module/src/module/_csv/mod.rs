//! `_csv` — interp-level CSV reader/writer accelerator for the `csv`
//! stdlib module.
//!
//! Port of `pypy/module/_csv/` (`interp_csv.py` W_Dialect + `_build_dialect`,
//! `interp_reader.py` W_Reader state machine, `interp_writer.py` W_Writer).
//! The dialect-format validation messages and the `QUOTE_STRINGS` /
//! `QUOTE_NOTNULL` quoting styles target CPython 3.14, which PyPy (3.11-era)
//! predates.
//!
//! `csv.py` does `from _csv import (Error, writer, reader, register_dialect,
//! unregister_dialect, get_dialect, list_dialects, field_size_limit,
//! QUOTE_MINIMAL, QUOTE_ALL, QUOTE_NONNUMERIC, QUOTE_NONE, QUOTE_STRINGS,
//! QUOTE_NOTNULL)` and `from _csv import Dialect`, so every one of those names
//! is exported here.

use pyre_object::PyObjectRef;
use pyre_object::gc_roots;

use pyre_interpreter::PyError;

// `interp_csv.py` quoting styles, extended with the 3.14 additions.
const QUOTE_MINIMAL: i64 = 0;
const QUOTE_ALL: i64 = 1;
const QUOTE_NONNUMERIC: i64 = 2;
const QUOTE_NONE: i64 = 3;
const QUOTE_STRINGS: i64 = 4;
const QUOTE_NOTNULL: i64 = 5;

// `interp_reader.py` parser states.
const START_RECORD: u8 = 0;
const START_FIELD: u8 = 1;
const ESCAPED_CHAR: u8 = 2;
const IN_FIELD: u8 = 3;
const IN_QUOTED_FIELD: u8 = 4;
const ESCAPE_IN_QUOTED_FIELD: u8 = 5;
const QUOTE_IN_QUOTED_FIELD: u8 = 6;
const EAT_CRNL: u8 = 7;
const AFTER_ESCAPED_CRNL: u8 = 8;

// `interp_reader.py FieldLimit.limit` — process-global max parsed field
// size; a plain `i64` so it needs no GC root.
static FIELD_LIMIT: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(128 * 1024);

/// Resolved dialect in the parser/serializer's internal form: code points
/// for the single-character options (with `None` standing in for the
/// `NOT_SET` sentinel), and plain values for the rest.
struct DialectConfig {
    delimiter: u32,
    doublequote: bool,
    escapechar: Option<u32>,
    lineterminator: String,
    quotechar: Option<u32>,
    quoting: i64,
    skipinitialspace: bool,
    strict: bool,
}

/// Build a `PyError` whose raised object is an instance of `_csv.Error`
/// (registered by the `exceptions:` block), with `msg` as the single
/// argument — `interp_csv.py W_Reader.error` / `W_Writer.error`.
fn csv_error(msg: impl Into<rustpython_wtf8::Wtf8Buf>) -> PyError {
    let msg = msg.into();
    // `interp_writer.py W_Writer.error`: `OperationError(w_error, space.newtext(msg))`.
    let Some(cls) = pyre_interpreter::builtins::lookup_exc_class("_csv.Error") else {
        return PyError::runtime_error(msg);
    };
    let _roots = pyre_object::gc_roots::push_roots();
    let cls_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(cls);
    let w_value = pyre_object::w_str_from_wtf8_managed(msg);
    PyError::from_type_and_value(pyre_object::gc_roots::shadow_stack_get(cls_slot), w_value)
}

// ── dialect format parsing (`interp_csv.py` `_get_*` + `_build_dialect`) ──

fn codepoint_kind(can_be_none: bool) -> &'static str {
    if can_be_none {
        "a unicode character or None"
    } else {
        "a unicode character"
    }
}

/// `_set_char` / `_set_char_or_none` — resolve a single-character option.
/// `default` applies when the slot is absent (`PY_NULL`); a Python `None`
/// maps to `NOT_SET` (`None`) when `can_be_none`, else raises.
fn get_codepoint(
    w_src: PyObjectRef,
    default: Option<u32>,
    name: &str,
    can_be_none: bool,
) -> Result<Option<u32>, PyError> {
    if w_src.is_null() {
        return Ok(default);
    }
    if unsafe { pyre_object::is_none(w_src) } {
        if can_be_none {
            return Ok(None);
        }
        return Err(PyError::type_error(
            pyre_interpreter::display::wtf8_format!(
                format!("\"{name}\" must be {}, not ", codepoint_kind(can_be_none)),
                unsafe { pyre_interpreter::baseobjspace::getfulltypename(w_src) },
            ),
        ));
    }
    if !unsafe { pyre_object::is_str(w_src) } {
        return Err(PyError::type_error(
            pyre_interpreter::display::wtf8_format!(
                format!("\"{name}\" must be {}, not ", codepoint_kind(can_be_none)),
                unsafe { pyre_interpreter::baseobjspace::getfulltypename(w_src) },
            ),
        ));
    }
    let s = unsafe { pyre_object::w_str_get_wtf8(w_src) };
    let mut cps = s.code_points();
    if let Some(cp) = cps.next()
        && cps.next().is_none()
    {
        return Ok(Some(cp.to_u32()));
    }
    Err(PyError::type_error(format!(
        "\"{name}\" must be {}, not a string of length {}",
        codepoint_kind(can_be_none),
        s.code_points().count(),
    )))
}

/// `_get_bool` — `None`/absent → default, else truthiness.
fn get_bool(w_src: PyObjectRef, default: bool) -> Result<bool, PyError> {
    if w_src.is_null() {
        return Ok(default);
    }
    pyre_interpreter::baseobjspace::is_true(w_src)
}

/// `_get_int` — absent → default; a non-int (including Python `None`)
/// raises `TypeError`.
fn get_int(w_src: PyObjectRef, default: i64, name: &str) -> Result<i64, PyError> {
    if w_src.is_null() {
        return Ok(default);
    }
    if !unsafe { pyre_object::is_int(w_src) } {
        return Err(PyError::type_error(format!(
            "\"{name}\" must be an integer"
        )));
    }
    Ok(unsafe { pyre_object::w_int_get_value(w_src) })
}

/// `_get_str` — absent → default; a non-str raises `TypeError`.
fn get_str(w_src: PyObjectRef, default: &str, name: &str) -> Result<String, PyError> {
    if w_src.is_null() {
        return Ok(default.to_string());
    }
    if !unsafe { pyre_object::is_str(w_src) } {
        return Err(PyError::type_error(
            pyre_interpreter::display::wtf8_format!(
                format!("\"{name}\" must be a string, not "),
                unsafe { pyre_interpreter::baseobjspace::getfulltypename(w_src) },
            ),
        ));
    }
    Ok(pyre_interpreter::baseobjspace::str_utf8_w(w_src)?.to_string())
}

/// `dialect_check_char` / `dialect_check_chars` — the cross-field
/// constraints `dialect_init` applies after each option is parsed:
/// `delimiter` / `quotechar` / `escapechar` may not be `\r` / `\n`, may not
/// be a space when `skipinitialspace` (except `delimiter`), must be pairwise
/// distinct, and may not appear in `lineterminator`. Each violation is a
/// `ValueError`.
fn validate_dialect(cfg: &DialectConfig) -> Result<(), PyError> {
    let check_char = |name: &str, c: u32, allow_space: bool| -> Result<(), PyError> {
        if c == '\r' as u32
            || c == '\n' as u32
            || (c == ' ' as u32 && cfg.skipinitialspace && !allow_space)
        {
            return Err(PyError::value_error(format!("bad {name} value")));
        }
        Ok(())
    };
    check_char("delimiter", cfg.delimiter, true)?;
    if let Some(e) = cfg.escapechar {
        check_char("escapechar", e, false)?;
    }
    if let Some(q) = cfg.quotechar {
        check_char("quotechar", q, false)?;
    }
    let pairs = [
        (
            "delimiter",
            "escapechar",
            Some(cfg.delimiter),
            cfg.escapechar,
        ),
        ("delimiter", "quotechar", Some(cfg.delimiter), cfg.quotechar),
        ("escapechar", "quotechar", cfg.escapechar, cfg.quotechar),
    ];
    for (n1, n2, a, b) in pairs {
        if let (Some(x), Some(y)) = (a, b)
            && x == y
        {
            return Err(PyError::value_error(format!("bad {n1} or {n2} value")));
        }
    }
    for c in cfg.lineterminator.chars() {
        let cp = c as u32;
        if cp == cfg.delimiter || Some(cp) == cfg.quotechar || Some(cp) == cfg.escapechar {
            return Err(PyError::value_error(
                "bad dialect value: a special character is also in the lineterminator".to_string(),
            ));
        }
    }
    Ok(())
}

fn valid_quoting(q: i64) -> bool {
    (QUOTE_MINIMAL..=QUOTE_NOTNULL).contains(&q)
}

/// `_fetch` — `space.findattr`; a missing attribute (AttributeError) is the
/// "not provided" marker (`PY_NULL`), other errors propagate.
fn fetch(obj: PyObjectRef, name: &str) -> Result<PyObjectRef, PyError> {
    match pyre_interpreter::baseobjspace::getattr_str(obj, name) {
        Ok(v) => Ok(v),
        Err(e) if e.kind == pyre_interpreter::PyErrorKind::AttributeError => {
            Ok(pyre_object::PY_NULL)
        }
        Err(e) => Err(e),
    }
}

fn is_csv_dialect(obj: PyObjectRef) -> Result<bool, PyError> {
    pyre_interpreter::baseobjspace::isinstance(obj, type_object())
}

enum BuildOutcome {
    Existing(PyObjectRef),
    Config(DialectConfig),
}

/// `interp_csv.py _build_dialect`. Each `w_*` is `PY_NULL` when the option
/// was not supplied; a string `w_dialect` is resolved through the registry,
/// and an unmodified `W_Dialect` short-circuits.
#[allow(clippy::too_many_arguments)]
fn build_dialect_config(
    w_dialect: PyObjectRef,
    mut w_delimiter: PyObjectRef,
    mut w_doublequote: PyObjectRef,
    mut w_escapechar: PyObjectRef,
    mut w_lineterminator: PyObjectRef,
    mut w_quotechar: PyObjectRef,
    mut w_quoting: PyObjectRef,
    mut w_skipinitialspace: PyObjectRef,
    mut w_strict: PyObjectRef,
) -> Result<BuildOutcome, PyError> {
    if !w_dialect.is_null() {
        let mut w_dialect = w_dialect;
        if unsafe { pyre_object::is_str(w_dialect) } {
            w_dialect = pyre_object::with_roots!(w_delimiter, w_doublequote, w_escapechar, w_lineterminator, w_quotechar, w_quoting, w_skipinitialspace, w_strict => lookup_registered_dialect(w_dialect))?;
        }
        if pyre_object::with_roots!(w_delimiter, w_dialect, w_doublequote, w_escapechar, w_lineterminator, w_quotechar, w_quoting, w_skipinitialspace, w_strict => is_csv_dialect(w_dialect))?
            && w_delimiter.is_null()
            && w_doublequote.is_null()
            && w_escapechar.is_null()
            && w_lineterminator.is_null()
            && w_quotechar.is_null()
            && w_quoting.is_null()
            && w_skipinitialspace.is_null()
            && w_strict.is_null()
        {
            return Ok(BuildOutcome::Existing(w_dialect));
        }
        if w_delimiter.is_null() {
            w_delimiter = pyre_object::with_roots!(w_dialect, w_doublequote, w_escapechar, w_lineterminator, w_quotechar, w_quoting, w_skipinitialspace, w_strict => fetch(w_dialect, "delimiter"))?;
        }
        if w_doublequote.is_null() {
            w_doublequote = pyre_object::with_roots!(w_delimiter, w_dialect, w_escapechar, w_lineterminator, w_quotechar, w_quoting, w_skipinitialspace, w_strict => fetch(w_dialect, "doublequote"))?;
        }
        if w_escapechar.is_null() {
            w_escapechar = pyre_object::with_roots!(w_delimiter, w_dialect, w_doublequote, w_lineterminator, w_quotechar, w_quoting, w_skipinitialspace, w_strict => fetch(w_dialect, "escapechar"))?;
        }
        if w_lineterminator.is_null() {
            w_lineterminator = pyre_object::with_roots!(w_delimiter, w_dialect, w_doublequote, w_escapechar, w_quotechar, w_quoting, w_skipinitialspace, w_strict => fetch(w_dialect, "lineterminator"))?;
        }
        if w_quotechar.is_null() {
            w_quotechar = pyre_object::with_roots!(w_delimiter, w_dialect, w_doublequote, w_escapechar, w_lineterminator, w_quoting, w_skipinitialspace, w_strict => fetch(w_dialect, "quotechar"))?;
        }
        if w_quoting.is_null() {
            w_quoting = pyre_object::with_roots!(w_delimiter, w_dialect, w_doublequote, w_escapechar, w_lineterminator, w_quotechar, w_skipinitialspace, w_strict => fetch(w_dialect, "quoting"))?;
        }
        if w_skipinitialspace.is_null() {
            w_skipinitialspace = pyre_object::with_roots!(w_delimiter, w_dialect, w_doublequote, w_escapechar, w_lineterminator, w_quotechar, w_quoting, w_strict => fetch(w_dialect, "skipinitialspace"))?;
        }
        if w_strict.is_null() {
            w_strict = pyre_object::with_roots!(w_delimiter, w_doublequote, w_escapechar, w_lineterminator, w_quotechar, w_quoting, w_skipinitialspace => fetch(w_dialect, "strict"))?;
        }
    }

    let delimiter = get_codepoint(w_delimiter, Some(',' as u32), "delimiter", false)?;
    let doublequote = pyre_object::with_roots!(w_escapechar, w_lineterminator, w_quotechar, w_quoting, w_skipinitialspace, w_strict => get_bool(w_doublequote, true))?;
    let escapechar = get_codepoint(w_escapechar, None, "escapechar", true)?;
    let lineterminator = pyre_object::with_roots!(w_quotechar, w_quoting, w_skipinitialspace, w_strict => get_str(w_lineterminator, "\r\n", "lineterminator"))?;
    let mut quoting = get_int(w_quoting, QUOTE_MINIMAL, "quoting")?;
    if !valid_quoting(quoting) {
        return Err(PyError::type_error("bad \"quoting\" value"));
    }
    // `quotechar=None` with no explicit `quoting` forces `QUOTE_NONE`.
    if !w_quotechar.is_null() && unsafe { pyre_object::is_none(w_quotechar) } && w_quoting.is_null()
    {
        quoting = QUOTE_NONE;
    }
    let quotechar = get_codepoint(w_quotechar, Some('"' as u32), "quotechar", true)?;
    let skipinitialspace =
        pyre_object::with_roots!(w_strict => get_bool(w_skipinitialspace, false))?;
    let strict = get_bool(w_strict, false)?;

    let delimiter = delimiter
        .ok_or_else(|| PyError::type_error("\"delimiter\" must be a 1-character string"))?;
    if quoting != QUOTE_NONE && quotechar.is_none() {
        return Err(PyError::type_error(
            "quotechar must be set if quoting enabled",
        ));
    }

    let cfg = DialectConfig {
        delimiter,
        doublequote,
        escapechar,
        lineterminator,
        quotechar,
        quoting,
        skipinitialspace,
        strict,
    };
    validate_dialect(&cfg)?;
    Ok(BuildOutcome::Config(cfg))
}

fn char_obj(cp: u32) -> PyObjectRef {
    let s: String = char::from_u32(cp)
        .map(|c| c.to_string())
        .unwrap_or_default();
    pyre_object::w_str_new_managed(&s)
}

/// `interp_csv.py` `NOT_SET`. A code point is never negative.
const NOT_SET: i32 = -1;

fn codepoint_field(cp: Option<u32>) -> i32 {
    match cp {
        Some(c) => c as i32,
        None => NOT_SET,
    }
}

fn codepoint_obj(cp: i32) -> PyObjectRef {
    if cp == NOT_SET {
        pyre_object::w_none()
    } else {
        char_obj(cp as u32)
    }
}

/// `interp_csv.py` `_build_dialect` materialises a `W_Dialect`. `cls` is the
/// requested subtype (`W_Dialect___new__`); reader/writer construction passes
/// the exact `_csv.Dialect` type.
fn config_to_dialect(cfg: &DialectConfig, cls: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let w_line = pyre_object::w_str_new_managed(&cfg.lineterminator);
    let _roots = gc_roots::push_roots();
    let _ = gc_roots::pin_root(cls);
    let _ = gc_roots::pin_root(w_line);
    let cls_slot = gc_roots::shadow_stack_len() - 2;
    let line_slot = cls_slot + 1;
    let obj = W_Dialect::allocate_instance(
        W_Dialect {
            delimiter: cfg.delimiter as i32,
            doublequote: cfg.doublequote,
            escapechar: codepoint_field(cfg.escapechar),
            quotechar: codepoint_field(cfg.quotechar),
            quoting: cfg.quoting,
            skipinitialspace: cfg.skipinitialspace,
            strict: cfg.strict,
            ..W_Dialect::default()
        },
        gc_roots::shadow_stack_get(cls_slot),
    );
    let _ = gc_roots::pin_root(obj);
    let obj_slot = line_slot + 1;
    let obj = gc_roots::shadow_stack_get(obj_slot);
    pyre_object::gc_hook::try_gc_write_barrier(obj as pyre_object::gc_hook::GCREF);
    let dialect = W_Dialect::from_obj(obj).expect("a fresh _csv.Dialect has the Dialect layout");
    dialect.lineterminator = gc_roots::shadow_stack_get(line_slot);
    Ok(gc_roots::shadow_stack_get(obj_slot))
}

/// Reader/writer hot path: the fields `W_Dialect` stores directly.
fn derive_config(d: PyObjectRef) -> Result<DialectConfig, PyError> {
    let dialect = W_Dialect::from_obj(d)
        .ok_or_else(|| PyError::type_error("_csv.Dialect instance expected"))?;
    let delimiter = dialect.delimiter as u32;
    let doublequote = dialect.doublequote;
    let escapechar = (dialect.escapechar != NOT_SET).then_some(dialect.escapechar as u32);
    let quotechar = (dialect.quotechar != NOT_SET).then_some(dialect.quotechar as u32);
    let quoting = dialect.quoting;
    let skipinitialspace = dialect.skipinitialspace;
    let strict = dialect.strict;
    let w_line = dialect.lineterminator;
    let lineterminator = if unsafe { pyre_object::is_str(w_line) } {
        pyre_interpreter::baseobjspace::str_utf8_w(w_line)?.to_string()
    } else {
        "\r\n".to_string()
    };
    Ok(DialectConfig {
        delimiter,
        doublequote,
        escapechar,
        lineterminator,
        quotechar,
        quoting,
        skipinitialspace,
        strict,
    })
}

// ── registry (`app_csv.py`) ──

/// `_csvstate.dialects` — the mapping `register_dialect` fills and
/// `get_dialect_from_registry` reads.
///
/// Upstream reaches it as module state, off the defining class, and PyPy as a
/// module global of `app_csv.py`; neither asks `sys.modules` where the module
/// is.  Naming the module instead resolves through the running `sys.modules`,
/// which a program is entitled to block -- and then `csv.list_dialects()`
/// fails on a module it never asked for.
///
/// The reference is the atomic word itself, not a field inside an allocation
/// some other atomic names.  A re-import replaces the mapping while another
/// thread is inside `list_dialects`, and a plain field written and read across
/// threads is a data race however the enclosing allocation is reached -- the
/// sibling `_ast` slot escapes that only because it is published once and never
/// replaced, which a per-import registry cannot be.  Same shape as
/// `faulthandler`'s owner slot.
static CSV_DIALECTS: std::sync::atomic::AtomicPtr<pyre_object::PyObject> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Forward the registry.  The module namespace holds the same mapping, but
/// only for as long as the module itself is reachable, and a program that
/// takes `_csv` out of `sys.modules` leaves this slot its only owner.
///
/// Runs from the collector inside the stop-the-world window, so no publish can
/// interleave between this load and the store that forwards it.
pub fn walk_csv_state_gc(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    let mut dialects = CSV_DIALECTS.load(std::sync::atomic::Ordering::Acquire);
    if dialects.is_null() {
        return;
    }
    visitor(&mut dialects);
    CSV_DIALECTS.store(dialects, std::sync::atomic::Ordering::Release);
}

/// Publish the mapping `init` just built.  A re-import mints a fresh one, the
/// way a fresh module state would.
fn publish_csv_dialects(dialects: PyObjectRef) {
    CSV_DIALECTS.store(dialects, std::sync::atomic::Ordering::Release);
}

fn csv_dialects() -> Result<PyObjectRef, PyError> {
    let dialects = CSV_DIALECTS.load(std::sync::atomic::Ordering::Acquire);
    if dialects.is_null() {
        return Err(PyError::runtime_error("_csv module not initialized"));
    }
    Ok(dialects)
}

fn lookup_registered_dialect(name: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let dialects = csv_dialects()?;
    match unsafe { pyre_object::dictmultiobject::w_dict_lookup(dialects, name) } {
        Some(d) => Ok(d),
        None => Err(csv_error("unknown dialect".to_string())),
    }
}

// ── `_csv.Dialect` type ──

/// `interp_csv.py` `W_Dialect`. A user subclass is `typedef.py`
/// `_getusercls`, allocated by `allocate_instance`.
#[pyre_interpreter::pyre_class("_csv.Dialect", user_layout)]
#[derive(Default)]
pub struct W_Dialect {
    pub delimiter: i32,
    pub doublequote: bool,
    pub escapechar: i32,
    pub lineterminator: PyObjectRef,
    pub quotechar: i32,
    pub quoting: i64,
    pub skipinitialspace: bool,
    pub strict: bool,
}

#[pyre_interpreter::pyre_methods(
    doc = "CSV dialect\n\nThe Dialect type records CSV parsing and generation options.\n"
)]
impl W_Dialect {
    /// `W_Dialect___new__`.
    #[staticmethod]
    fn __new__(
        cls: PyObjectRef,
        #[default(pyre_object::PY_NULL)] dialect: PyObjectRef,
        #[default(pyre_object::PY_NULL)] delimiter: PyObjectRef,
        #[default(pyre_object::PY_NULL)] doublequote: PyObjectRef,
        #[default(pyre_object::PY_NULL)] escapechar: PyObjectRef,
        #[default(pyre_object::PY_NULL)] lineterminator: PyObjectRef,
        #[default(pyre_object::PY_NULL)] quotechar: PyObjectRef,
        #[default(pyre_object::PY_NULL)] quoting: PyObjectRef,
        #[default(pyre_object::PY_NULL)] skipinitialspace: PyObjectRef,
        #[default(pyre_object::PY_NULL)] strict: PyObjectRef,
    ) -> Result<PyObjectRef, PyError> {
        // `check_user_subclass` can collect. The slot is the live word;
        // this pin's argument is not read again.
        let _roots = pyre_object::gc_roots::push_roots();
        let cls_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(cls);
        pyre_interpreter::typedef::check_user_subclass(
            type_object(),
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        )?;
        let outcome = build_dialect_config(
            dialect,
            delimiter,
            doublequote,
            escapechar,
            lineterminator,
            quotechar,
            quoting,
            skipinitialspace,
            strict,
        )?;
        match outcome {
            BuildOutcome::Existing(d)
                if std::ptr::eq(
                    pyre_object::gc_roots::shadow_stack_get(cls_slot),
                    type_object(),
                ) =>
            {
                Ok(d)
            }
            BuildOutcome::Existing(d) => {
                let cfg = derive_config(d)?;
                config_to_dialect(&cfg, pyre_object::gc_roots::shadow_stack_get(cls_slot))
            }
            BuildOutcome::Config(cfg) => {
                config_to_dialect(&cfg, pyre_object::gc_roots::shadow_stack_get(cls_slot))
            }
        }
    }

    /// `__new__` already consumed the format options. The same arguments are
    /// still presented to `__init__`.
    fn __init__(&mut self, _args: &[PyObjectRef]) -> Result<(), PyError> {
        Ok(())
    }

    /// `W_Dialect.reduce_ex_w`.
    fn __reduce_ex__(&self, _protocol: PyObjectRef) -> Result<PyObjectRef, PyError> {
        Err(PyError::type_error("can't pickle _csv.Dialect objects"))
    }

    /// Same refusal as `reduce_ex_w` when the caller asks for `__reduce__`.
    fn __reduce__(&self) -> Result<PyObjectRef, PyError> {
        Err(PyError::type_error("can't pickle _csv.Dialect objects"))
    }

    #[getter]
    fn delimiter(&self) -> PyObjectRef {
        codepoint_obj(self.delimiter)
    }

    #[getter]
    fn doublequote(&self) -> bool {
        self.doublequote
    }

    #[getter]
    fn escapechar(&self) -> PyObjectRef {
        codepoint_obj(self.escapechar)
    }

    #[getter]
    fn lineterminator(&self) -> PyObjectRef {
        self.lineterminator
    }

    #[getter]
    fn quotechar(&self) -> PyObjectRef {
        codepoint_obj(self.quotechar)
    }

    #[getter]
    fn quoting(&self) -> i64 {
        self.quoting
    }

    #[getter]
    fn skipinitialspace(&self) -> bool {
        self.skipinitialspace
    }

    #[getter]
    fn strict(&self) -> bool {
        self.strict
    }
}

/// The GC types this module owns. Appended after every earlier module so
/// established module tids stay put. The builtin is registered before its
/// `_getusercls` child.
pub(crate) fn gc_types(types: &mut Vec<pyre_interpreter::importing::ModuleGcType>) {
    use pyre_interpreter::importing::{ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    let pyre_class = ModuleGcLayout::PyreClass {
        memory_pressure_offset: None,
    };
    types.push(ModuleGcType {
        descriptor: <W_Dialect as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: pyre_class,
        destructor: None,
    });
    types.push(ModuleGcType {
        descriptor: &W_DIALECT_USER_PYRE_CLASS_DESCRIPTOR,
        layout: pyre_class,
        destructor: None,
    });
}

// ── `_csv.reader` ──

fn add_char(
    field: &mut String,
    field_len: &mut usize,
    c: char,
    limit: i64,
    line_num: i64,
) -> Result<(), PyError> {
    if *field_len as i64 >= limit {
        return Err(csv_error(format!(
            "line {line_num}: field larger than field limit"
        )));
    }
    field.push(c);
    *field_len += 1;
    Ok(())
}

/// `parse_save_field` — record the finished field's text, whether it was
/// unquoted, and its code-point length; the final value (str / float / None)
/// is computed once the record is complete (`parse_save_field`'s quoting
/// conversions).
fn save_field(
    fields: &mut Vec<(String, bool, usize)>,
    field: &mut String,
    field_len: &mut usize,
    unquoted: &mut bool,
) {
    let len = *field_len;
    fields.push((std::mem::take(field), *unquoted, len));
    *unquoted = true;
    *field_len = 0;
}

fn to_float(w_str: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let float_type = pyre_interpreter::typedef::gettypefor(&pyre_object::FLOAT_TYPE)
        .ok_or_else(|| PyError::runtime_error("float type unavailable"))?;
    pyre_interpreter::call::call_function_impl_result(float_type.as_ptr(), &[w_str])
}

/// `W_Reader.next_w` — parse the next CSV record from the underlying line
/// iterator. Re-entering the reader from its own line iterator (gh-145105) is
/// rejected with a `_csv.Error`.
fn reader_next_impl(mut self_obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let reading = pyre_object::with_roots!(self_obj => pyre_interpreter::baseobjspace::getattr_str(self_obj, "_reading")
        .ok()
        .map(|v| pyre_interpreter::baseobjspace::is_true(v).unwrap_or(false))
        .unwrap_or(false));
    if reading {
        return Err(csv_error("reader is already iterating".to_string()));
    }
    let _roots = gc_roots::push_roots();
    let self_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(self_obj);
    pyre_interpreter::baseobjspace::setattr_str(
        gc_roots::shadow_stack_get(self_slot),
        "_reading",
        pyre_object::w_bool_from(true),
    )?;
    let result = reader_next_inner(gc_roots::shadow_stack_get(self_slot));
    // FINALLY reset of the re-entrancy guard. The inner `result` must be
    // returned/propagated unchanged, so a failure of this reset write is
    // deliberately ignored rather than masking the inner exception.
    if let Err(_e) = pyre_interpreter::baseobjspace::setattr_str(
        gc_roots::shadow_stack_get(self_slot),
        "_reading",
        pyre_object::w_bool_from(false),
    ) {}
    result
}

fn reader_next_inner(mut self_obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let dialect_obj = pyre_object::with_roots!(self_obj => pyre_interpreter::baseobjspace::getattr_str(self_obj, "dialect"))?;
    let cfg = pyre_object::with_roots!(self_obj => derive_config(dialect_obj))?;
    let limit = FIELD_LIMIT.load(std::sync::atomic::Ordering::Relaxed);
    let mut line_num = {
        let v = pyre_object::with_roots!(self_obj => pyre_interpreter::baseobjspace::getattr_str(self_obj, "line_num"))?;
        if unsafe { pyre_object::is_int(v) } {
            unsafe { pyre_object::w_int_get_value(v) }
        } else {
            0
        }
    };

    let _roots = gc_roots::push_roots();
    // `getattr_str` collects: pin self before looking up the iterator.
    let self_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(self_obj);
    let iter_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(pyre_interpreter::baseobjspace::getattr_str(
        gc_roots::shadow_stack_get(self_slot),
        "_iterator",
    )?);

    let mut fields: Vec<(String, bool, usize)> = Vec::new();
    let mut field = String::new();
    let mut field_len: usize = 0;
    let mut field_unquoted = true;
    let mut state = START_RECORD;

    'lines: loop {
        let w_iter = gc_roots::shadow_stack_get(iter_slot);
        let line = match pyre_interpreter::baseobjspace::next(w_iter) {
            Ok(l) => l,
            Err(e) => {
                let (stop, e) = e.matches_stop_iteration_keep();
                if stop {
                    if state != START_RECORD
                        && state != EAT_CRNL
                        && (field_len > 0 || state == IN_QUOTED_FIELD)
                    {
                        if cfg.strict {
                            return Err(csv_error(format!(
                                "line {line_num}: unexpected end of data"
                            )));
                        }
                        save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                        break 'lines;
                    }
                    return Err(PyError::stop_iteration());
                } else {
                    return Err(e);
                }
            }
        };
        line_num += 1;
        if unsafe { pyre_object::bytesobject::is_bytes(line) } {
            return Err(csv_error(format!(
                "line {line_num}: iterator should return strings, not bytes (the file should be opened in text mode)"
            )));
        }
        if !unsafe { pyre_object::is_str(line) } {
            return Err(csv_error(pyre_interpreter::display::wtf8_format!(
                format!("line {line_num}: iterator should return strings, not "),
                unsafe { pyre_interpreter::baseobjspace::getfulltypename(line) },
                " (the file should be opened in text mode)",
            )));
        }
        // Field text is a `String`, so a lone surrogate has no encoding here.
        // `str_utf8_w` reports UnicodeEncodeError ("surrogates not allowed").
        let s = pyre_interpreter::baseobjspace::str_utf8_w(line)?;
        for c in s.chars() {
            let cp = c as u32;
            let is_nl = cp == 10 || cp == 13;

            if state == START_RECORD {
                if is_nl {
                    state = EAT_CRNL;
                    continue;
                }
                state = START_FIELD;
            }

            if state == START_FIELD {
                if is_nl {
                    save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                    state = EAT_CRNL;
                } else if Some(cp) == cfg.quotechar && cfg.quoting != QUOTE_NONE {
                    field_unquoted = false;
                    state = IN_QUOTED_FIELD;
                } else if Some(cp) == cfg.escapechar {
                    state = ESCAPED_CHAR;
                } else if cp == 32 && cfg.skipinitialspace {
                    // ignore leading space
                } else if cp == cfg.delimiter {
                    save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                } else {
                    add_char(&mut field, &mut field_len, c, limit, line_num)?;
                    state = IN_FIELD;
                }
            } else if state == ESCAPED_CHAR {
                add_char(&mut field, &mut field_len, c, limit, line_num)?;
                state = if is_nl { AFTER_ESCAPED_CRNL } else { IN_FIELD };
            } else if state == IN_FIELD || state == AFTER_ESCAPED_CRNL {
                if is_nl {
                    save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                    state = EAT_CRNL;
                } else if Some(cp) == cfg.escapechar {
                    state = ESCAPED_CHAR;
                } else if cp == cfg.delimiter {
                    save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                    state = START_FIELD;
                } else {
                    add_char(&mut field, &mut field_len, c, limit, line_num)?;
                }
            } else if state == IN_QUOTED_FIELD {
                if Some(cp) == cfg.escapechar {
                    state = ESCAPE_IN_QUOTED_FIELD;
                } else if Some(cp) == cfg.quotechar && cfg.quoting != QUOTE_NONE {
                    state = if cfg.doublequote {
                        QUOTE_IN_QUOTED_FIELD
                    } else {
                        IN_FIELD
                    };
                } else {
                    add_char(&mut field, &mut field_len, c, limit, line_num)?;
                }
            } else if state == ESCAPE_IN_QUOTED_FIELD {
                add_char(&mut field, &mut field_len, c, limit, line_num)?;
                state = IN_QUOTED_FIELD;
            } else if state == QUOTE_IN_QUOTED_FIELD {
                if cfg.quoting != QUOTE_NONE && Some(cp) == cfg.quotechar {
                    add_char(&mut field, &mut field_len, c, limit, line_num)?;
                    state = IN_QUOTED_FIELD;
                } else if cp == cfg.delimiter {
                    save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                    state = START_FIELD;
                } else if is_nl {
                    save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                    state = EAT_CRNL;
                } else if !cfg.strict {
                    add_char(&mut field, &mut field_len, c, limit, line_num)?;
                    state = IN_FIELD;
                } else {
                    let dc = char::from_u32(cfg.delimiter).unwrap_or('?');
                    let qc = cfg.quotechar.and_then(char::from_u32).unwrap_or('?');
                    return Err(csv_error(format!(
                        "line {line_num}: '{dc}' expected after '{qc}'"
                    )));
                }
            } else if state == EAT_CRNL && !is_nl {
                return Err(csv_error(format!(
                    "line {line_num}: new-line character seen in unquoted field - do you need to open the file with newline=''?"
                )));
            }
        }

        match state {
            s if s == IN_FIELD || s == QUOTE_IN_QUOTED_FIELD => {
                save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                break 'lines;
            }
            s if s == ESCAPED_CHAR => {
                add_char(&mut field, &mut field_len, '\n', limit, line_num)?;
                state = IN_FIELD;
            }
            s if s == IN_QUOTED_FIELD => {}
            s if s == ESCAPE_IN_QUOTED_FIELD => {
                add_char(&mut field, &mut field_len, '\n', limit, line_num)?;
                state = IN_QUOTED_FIELD;
            }
            s if s == START_FIELD => {
                save_field(&mut fields, &mut field, &mut field_len, &mut field_unquoted);
                break 'lines;
            }
            s if s == AFTER_ESCAPED_CRNL => {}
            _ => break 'lines,
        }
    }

    let self_obj = gc_roots::shadow_stack_get(self_slot);
    pyre_interpreter::baseobjspace::setattr_str(
        self_obj,
        "line_num",
        pyre_object::w_int_new(line_num),
    )?;

    let result = pyre_object::listobject::w_list_new(Vec::new());
    let result_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(result);
    for (s, unquoted, len) in fields {
        // `parse_save_field` quoting conversions: an empty unquoted field is
        // `None` under QUOTE_NOTNULL / QUOTE_STRINGS; a non-empty unquoted
        // field is coerced to `float` under QUOTE_NONNUMERIC / QUOTE_STRINGS.
        let w = if unquoted
            && len == 0
            && (cfg.quoting == QUOTE_NOTNULL || cfg.quoting == QUOTE_STRINGS)
        {
            pyre_object::w_none()
        } else {
            let ws = pyre_object::w_str_new_managed(&s);
            if unquoted
                && len != 0
                && (cfg.quoting == QUOTE_NONNUMERIC || cfg.quoting == QUOTE_STRINGS)
            {
                to_float(ws)?
            } else {
                ws
            }
        };
        let result = gc_roots::shadow_stack_get(result_slot);
        unsafe { pyre_object::listobject::w_list_append(result, w) };
    }
    Ok(gc_roots::shadow_stack_get(result_slot))
}

mod reader_class {
    use super::*;

    pyre_interpreter::py_class! {
        "_csv.reader",
        methods: {
            fn __iter__(self_obj: PyObjectRef) -> PyObjectRef {
                self_obj
            }
            fn __next__(self_obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
                reader_next_impl(self_obj)
            }
        }
    }
}

// ── `_csv.writer` ──

fn special_chars(cfg: &DialectConfig) -> Vec<u32> {
    let mut s = vec![cfg.delimiter, 13, 10];
    for c in cfg.lineterminator.chars() {
        s.push(c as u32);
    }
    if let Some(e) = cfg.escapechar {
        s.push(e);
    }
    if let Some(q) = cfg.quotechar {
        s.push(q);
    }
    s
}

/// `W_Writer.writerow` — serialize one record.
fn writer_writerow_impl(
    mut self_obj: PyObjectRef,
    mut w_fields: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let dialect_obj = pyre_object::with_roots!(self_obj, w_fields => pyre_interpreter::baseobjspace::getattr_str(self_obj, "dialect"))?;
    let cfg = pyre_object::with_roots!(self_obj, w_fields => derive_config(dialect_obj))?;
    let mut w_filewrite = pyre_object::with_roots!(w_fields => pyre_interpreter::baseobjspace::getattr_str(self_obj, "_write"))?;

    let row = match pyre_object::with_roots!(w_fields, w_filewrite => pyre_interpreter::builtins::collect_iterable(w_fields))
    {
        Ok(r) => r,
        Err(e) if e.kind == pyre_interpreter::PyErrorKind::TypeError => {
            let r =
                unsafe { pyre_interpreter::display::py_repr_wtf8(w_fields) }.unwrap_or_default();
            return Err(csv_error(pyre_interpreter::wtf8_format!(
                "iterable expected, not ",
                r
            )));
        }
        Err(e) => return Err(e),
    };

    let special = special_chars(&cfg);
    let quote_char = cfg.quotechar.and_then(char::from_u32).unwrap_or('"');
    let delim_char = char::from_u32(cfg.delimiter).unwrap_or(',');
    let n = row.len();
    let mut rec = rustpython_wtf8::Wtf8Buf::new();

    // Rendering a field runs its `__str__` / `__repr__` and `float_w` its
    // `__float__`, so every turn is a collection point.  The collected row and
    // the bound `_write` are native locals no root walker updates, so publish
    // them and read each field back after the render that precedes its
    // remaining type queries.
    let _roots = pyre_object::gc_roots::push_roots();
    let write_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_filewrite);
    let row_base = pyre_object::gc_roots::shadow_stack_len();
    for &item in &row {
        let _ = pyre_object::gc_roots::pin_root(item);
    }

    for i in 0..n {
        let w_field = pyre_object::gc_roots::shadow_stack_get(row_base + i);
        let field = if unsafe { pyre_object::is_none(w_field) } {
            rustpython_wtf8::Wtf8Buf::new()
        } else if unsafe { pyre_object::is_float(w_field) } {
            unsafe { pyre_interpreter::display::py_repr_wtf8(w_field) }?
        } else {
            unsafe { pyre_interpreter::display::py_str_wtf8(w_field) }?
        };
        let w_field = pyre_object::gc_roots::shadow_stack_get(row_base + i);

        let mut quoted = match cfg.quoting {
            QUOTE_NONNUMERIC => pyre_interpreter::baseobjspace::float_w(w_field).is_err(),
            QUOTE_ALL => true,
            QUOTE_MINIMAL => {
                let mut q = false;
                for c in field.code_points() {
                    let cp = c.to_u32();
                    if !special.contains(&cp) {
                        continue;
                    }
                    if Some(cp) == cfg.escapechar {
                        continue;
                    }
                    if Some(cp) != cfg.quotechar || cfg.doublequote {
                        q = true;
                        break;
                    }
                }
                q
            }
            QUOTE_STRINGS => unsafe { pyre_object::is_str(w_field) },
            QUOTE_NOTNULL => !unsafe { pyre_object::is_none(w_field) },
            _ => false,
        };

        // An empty field can only be represented by quoting it. The quoting
        // styles that never quote a field that is not already quoted
        // (QUOTE_NONE and — for a non-quotable value — QUOTE_STRINGS /
        // QUOTE_NOTNULL) raise instead of silently dropping it.
        let cannot_force_quote = cfg.quoting == QUOTE_NONE
            || cfg.quoting == QUOTE_STRINGS
            || cfg.quoting == QUOTE_NOTNULL;
        if field.is_empty() {
            if cfg.delimiter == ' ' as u32 && cfg.skipinitialspace && !quoted {
                if cannot_force_quote {
                    return Err(csv_error(
                        "empty field must be quoted if delimiter is a space and skipinitialspace is true"
                            .to_string(),
                    ));
                }
                quoted = true;
            }
            if n == 1 && !quoted {
                if cannot_force_quote {
                    return Err(csv_error(
                        "single empty field record must be quoted".to_string(),
                    ));
                }
                quoted = true;
            }
        }

        if i > 0 {
            rec.push_char(delim_char);
        }
        if quoted {
            rec.push_char(quote_char);
        }

        for c in field.code_points() {
            let cp = c.to_u32();
            if special.contains(&cp) {
                let want_escape = if cfg.quoting == QUOTE_NONE {
                    true
                } else {
                    let mut we = false;
                    if Some(cp) == cfg.quotechar {
                        if cfg.doublequote {
                            rec.push_char(quote_char);
                        } else {
                            we = true;
                        }
                    }
                    if Some(cp) == cfg.escapechar {
                        we = true;
                    }
                    we
                };
                if want_escape {
                    match cfg.escapechar.and_then(char::from_u32) {
                        Some(e) => rec.push_char(e),
                        None => {
                            return Err(csv_error(
                                "need to escape, but no escapechar set".to_string(),
                            ));
                        }
                    }
                }
            }
            rec.push(c);
        }

        if quoted {
            rec.push_char(quote_char);
        }
    }

    rec.push_str(&cfg.lineterminator);
    let rec_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_from_wtf8_managed(rec));
    pyre_interpreter::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(write_slot),
        &[pyre_object::gc_roots::shadow_stack_get(rec_slot)],
    )
}

/// `W_Writer.writerows` — serialize a sequence of records.
fn writer_writerows_impl(
    mut self_obj: PyObjectRef,
    w_seqseq: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let it = pyre_object::with_roots!(self_obj => pyre_interpreter::baseobjspace::iter(w_seqseq))?;
    let _roots = gc_roots::push_roots();
    let it_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(it);
    let self_slot = gc_roots::shadow_stack_len();
    let _ = gc_roots::pin_root(self_obj);
    loop {
        let it = gc_roots::shadow_stack_get(it_slot);
        let row = match pyre_interpreter::baseobjspace::next(it) {
            Ok(r) => r,
            Err(e) => {
                let (stop, e) = e.matches_stop_iteration_keep();
                if stop {
                    break;
                }
                return Err(e);
            }
        };
        writer_writerow_impl(gc_roots::shadow_stack_get(self_slot), row)?;
    }
    Ok(pyre_object::w_none())
}

mod writer_class {
    use super::*;

    pyre_interpreter::py_class! {
        "_csv.writer",
        methods: {
            fn writerow(self_obj: PyObjectRef, row: PyObjectRef) -> Result<PyObjectRef, PyError> {
                writer_writerow_impl(self_obj, row)
            }
            fn writerows(self_obj: PyObjectRef, rows: PyObjectRef) -> Result<PyObjectRef, PyError> {
                writer_writerows_impl(self_obj, rows)
            }
        }
    }
}

/// Resolve the dialect object for a reader/writer constructor.
#[allow(clippy::too_many_arguments)]
fn resolve_dialect(
    w_dialect: PyObjectRef,
    w_delimiter: PyObjectRef,
    w_doublequote: PyObjectRef,
    w_escapechar: PyObjectRef,
    w_lineterminator: PyObjectRef,
    w_quotechar: PyObjectRef,
    w_quoting: PyObjectRef,
    w_skipinitialspace: PyObjectRef,
    w_strict: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let outcome = build_dialect_config(
        w_dialect,
        w_delimiter,
        w_doublequote,
        w_escapechar,
        w_lineterminator,
        w_quotechar,
        w_quoting,
        w_skipinitialspace,
        w_strict,
    )?;
    match outcome {
        BuildOutcome::Existing(d) => Ok(d),
        BuildOutcome::Config(cfg) => config_to_dialect(&cfg, type_object()),
    }
}

/// `app_csv.list_dialects` — the registered dialect names; takes no args.
fn list_dialects_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if !args.is_empty() {
        return Err(PyError::type_error(format!(
            "list_dialects() takes no arguments ({} given)",
            args.len()
        )));
    }
    let dialects = csv_dialects()?;
    let items = unsafe { pyre_object::dictmultiobject::w_dict_items(dialects) };
    Ok(pyre_object::listobject::w_list_new(
        items.into_iter().map(|(k, _)| k).collect(),
    ))
}

/// `csv_field_size_limit` — return the current limit, and set it when an
/// integer argument is supplied.
fn field_size_limit_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if args.len() > 1 {
        return Err(PyError::type_error(format!(
            "field_size_limit() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    let old = FIELD_LIMIT.load(std::sync::atomic::Ordering::Relaxed);
    if let Some(&v) = args.first() {
        if !unsafe { pyre_object::is_int(v) } {
            return Err(PyError::type_error("limit must be an integer"));
        }
        FIELD_LIMIT.store(
            unsafe { pyre_object::w_int_get_value(v) },
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    Ok(pyre_object::w_int_new(old))
}

pyre_interpreter::py_module! {
    "_csv",
    interpleveldefs: {
        "Dialect" => type_object(),
        "__version__" => pyre_object::w_str_new("1.0"),
    },
    int_constants: {
        "QUOTE_MINIMAL" => QUOTE_MINIMAL,
        "QUOTE_ALL" => QUOTE_ALL,
        "QUOTE_NONNUMERIC" => QUOTE_NONNUMERIC,
        "QUOTE_NONE" => QUOTE_NONE,
        "QUOTE_STRINGS" => QUOTE_STRINGS,
        "QUOTE_NOTNULL" => QUOTE_NOTNULL,
    },
    inline_functions: {
        // `csv_reader` — build the reader over an iterable of lines.
        fn reader(
            iterable: PyObjectRef,
            #[default(pyre_object::PY_NULL)] mut dialect: PyObjectRef,
            #[default(pyre_object::PY_NULL)] mut delimiter: PyObjectRef,
            #[default(pyre_object::PY_NULL)] mut doublequote: PyObjectRef,
            #[default(pyre_object::PY_NULL)] mut escapechar: PyObjectRef,
            #[default(pyre_object::PY_NULL)] mut lineterminator: PyObjectRef,
            #[default(pyre_object::PY_NULL)] mut quotechar: PyObjectRef,
            #[default(pyre_object::PY_NULL)] mut quoting: PyObjectRef,
            #[default(pyre_object::PY_NULL)] mut skipinitialspace: PyObjectRef,
            #[default(pyre_object::PY_NULL)] mut strict: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            let mut w_iter = pyre_object::with_roots!(delimiter, dialect, doublequote, escapechar, lineterminator, quotechar, quoting, skipinitialspace, strict => pyre_interpreter::baseobjspace::iter(iterable))?;
            let dialect_obj = pyre_object::with_roots!(w_iter => resolve_dialect(
                dialect, delimiter, doublequote, escapechar, lineterminator,
                quotechar, quoting, skipinitialspace, strict,
            ))?;
            let r = pyre_object::w_instance_new(reader_class::type_object());
            let _roots = gc_roots::push_roots();
            let slot = gc_roots::shadow_stack_len();
            let _ = gc_roots::pin_root(r);
            let _ = gc_roots::pin_root(dialect_obj);
            let _ = gc_roots::pin_root(w_iter);
            pyre_interpreter::baseobjspace::setattr_str(gc_roots::shadow_stack_get(slot), "dialect", gc_roots::shadow_stack_get(slot + 1))?;
            pyre_interpreter::baseobjspace::setattr_str(gc_roots::shadow_stack_get(slot), "_iterator", gc_roots::shadow_stack_get(slot + 2))?;
            pyre_interpreter::baseobjspace::setattr_str(gc_roots::shadow_stack_get(slot), "line_num", pyre_object::w_int_new(0))?;
            pyre_interpreter::baseobjspace::setattr_str(gc_roots::shadow_stack_get(slot), "_reading", pyre_object::w_bool_from(false))?;
            Ok(gc_roots::shadow_stack_get(slot))
        }

        // `csv_writer` — build the writer over a file-like object's `write`.
        fn writer(
            mut fileobj: PyObjectRef,
            #[default(pyre_object::PY_NULL)] dialect: PyObjectRef,
            #[default(pyre_object::PY_NULL)] delimiter: PyObjectRef,
            #[default(pyre_object::PY_NULL)] doublequote: PyObjectRef,
            #[default(pyre_object::PY_NULL)] escapechar: PyObjectRef,
            #[default(pyre_object::PY_NULL)] lineterminator: PyObjectRef,
            #[default(pyre_object::PY_NULL)] quotechar: PyObjectRef,
            #[default(pyre_object::PY_NULL)] quoting: PyObjectRef,
            #[default(pyre_object::PY_NULL)] skipinitialspace: PyObjectRef,
            #[default(pyre_object::PY_NULL)] strict: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            let mut dialect_obj = pyre_object::with_roots!(fileobj => resolve_dialect(
                dialect, delimiter, doublequote, escapechar, lineterminator,
                quotechar, quoting, skipinitialspace, strict,
            ))?;
            // A missing `write` attribute is a TypeError ("argument 1 must
            // have a write method"); a `write` whose access itself raises
            // (e.g. a property) propagates that error unchanged.
            let w_write = match pyre_object::with_roots!(dialect_obj => pyre_interpreter::baseobjspace::getattr_str(fileobj, "write")) {
                Ok(w) => w,
                Err(e) if e.kind == pyre_interpreter::PyErrorKind::AttributeError => {
                    return Err(PyError::type_error("argument 1 must have a write method"));
                }
                Err(e) => return Err(e),
            };
            let w = pyre_object::w_instance_new(writer_class::type_object());
            let _roots = gc_roots::push_roots();
            let slot = gc_roots::shadow_stack_len();
            let _ = gc_roots::pin_root(w);
            let _ = gc_roots::pin_root(dialect_obj);
            let _ = gc_roots::pin_root(w_write);
            pyre_interpreter::baseobjspace::setattr_str(gc_roots::shadow_stack_get(slot), "dialect", gc_roots::shadow_stack_get(slot + 1))?;
            pyre_interpreter::baseobjspace::setattr_str(gc_roots::shadow_stack_get(slot), "_write", gc_roots::shadow_stack_get(slot + 2))?;
            Ok(gc_roots::shadow_stack_get(slot))
        }

        // `app_csv.register_dialect` — validate + register under `name`.
        fn register_dialect(
            mut name: PyObjectRef,
            #[default(pyre_object::PY_NULL)] dialect: PyObjectRef,
            #[default(pyre_object::PY_NULL)] delimiter: PyObjectRef,
            #[default(pyre_object::PY_NULL)] doublequote: PyObjectRef,
            #[default(pyre_object::PY_NULL)] escapechar: PyObjectRef,
            #[default(pyre_object::PY_NULL)] lineterminator: PyObjectRef,
            #[default(pyre_object::PY_NULL)] quotechar: PyObjectRef,
            #[default(pyre_object::PY_NULL)] quoting: PyObjectRef,
            #[default(pyre_object::PY_NULL)] skipinitialspace: PyObjectRef,
            #[default(pyre_object::PY_NULL)] strict: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            if !unsafe { pyre_object::is_str(name) } {
                return Err(PyError::type_error("dialect name must be a string"));
            }
            let dialect_obj = pyre_object::with_roots!(name => resolve_dialect(
                dialect, delimiter, doublequote, escapechar, lineterminator,
                quotechar, quoting, skipinitialspace, strict,
            ))?;
            let dialects = csv_dialects()?;
            unsafe { pyre_object::dictmultiobject::w_dict_store(dialects, name, dialect_obj) };
            Ok(pyre_object::w_none())
        }

        // `app_csv.unregister_dialect`.
        fn unregister_dialect(name: PyObjectRef) -> Result<PyObjectRef, PyError> {
            let dialects = csv_dialects()?;
            if unsafe { pyre_object::dictmultiobject::w_dict_delitem(dialects, name) } {
                Ok(pyre_object::w_none())
            } else {
                Err(csv_error("unknown dialect".to_string()))
            }
        }

        // `app_csv.get_dialect`.
        fn get_dialect(name: PyObjectRef) -> Result<PyObjectRef, PyError> {
            lookup_registered_dialect(name)
        }
    },
    functions: {
        // `list_dialects` / `field_size_limit` are varargs so the
        // "takes no arguments" / "at most 1 argument" guards can be enforced
        // (the flat builtin ABI does not reject extra positionals on its own).
        "list_dialects" / * = list_dialects_fn,
        "field_size_limit" / * = field_size_limit_fn,
    },
    extra_init: |ns| {
        // `_csv.Error` — built here rather than in the `exceptions:` arm
        // because `_csvstate_init` builds it from a type spec that names its
        // own `basicsize`, and a spec that does gets no managed weakref.  It
        // is therefore the one module exception class that is not
        // weak-referenceable, together with `ssl.SSLError`.
        let mut ns = ns;
        let mut w_error = pyre_object::with_roots!(ns => pyre_interpreter::builtins::make_exc_type(
            "_csv.Error",
            pyre_interpreter::builtins::exc_exception_new,
            pyre_interpreter::builtins::lookup_exc_class("Exception")
                .expect("Exception must be installed before _csv init"),
        ));
        pyre_interpreter::__pyre_store!(ns, "Error", w_error);
        // `app_csv._dialects = {}` — the registry mapping.  It is stored in
        // the module namespace under the name PyPy gives it and published to
        // the state the accelerator reads, which is what keeps it reachable
        // once the module is not.
        let mut dialects = pyre_object::with_roots!(ns => pyre_object::w_dict_new());
        pyre_interpreter::__pyre_store!(ns, "_dialects", dialects);
        publish_csv_dialects(dialects);
    },
}
