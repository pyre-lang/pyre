//! `pypy/module/unicodedata/interp_ucd.py`: the `UCD` type and the
//! character-database queries.
//!
//! Real implementation backed by the runtime-independent
//! `rustpython-unicode` crate.  The module-level functions read the latest
//! bundled database; `unicodedata.ucd_3_2_0` reads the Unicode 3.2.0 view
//! (used by `stringprep`).  `normalize` / `is_normalized` are
//! version-independent, matching the crate, so the 3.2.0 instance shares those
//! callables with the module. `lookup` consults the instance's UCD: the
//! modern view resolves name aliases and named sequences; `ucd_3_2_0` does not.
//!
//! Signatures and error types/messages follow CPython 3.14.
//!
//! Data-version caveat: the pinned crate bundles a newer Unicode release
//! (`unidata_version` reports 17.0.0) than CPython 3.14 (16.0.0), so results
//! differ on code points that release newly assigns (they read as unassigned
//! under 16.0.0), and on two assigned-character cases the crate handles
//! differently: `decomposition` of Hangul LVT syllables yields the two-part
//! `<LV> <T>` form rather than the fully expanded jamo, and `numeric` of
//! vulgar-fraction code points carries f32-rounded values.  These are
//! upstream-crate properties, to reconcile when the pin is revisited.

use pyre_object::*;
use rustpython_unicode::{self as ucd_core, NormalizeForm};
use rustpython_wtf8::{CodePoint, Wtf8, Wtf8Buf};

use pyre_interpreter::PyError;

type PyResult = Result<PyObjectRef, PyError>;

/// `interp_ucd.py UCD._unicodedb` — the view is a field of the UCD
/// instance, not a constructor the translator would residualize.
/// `legacy` is `W_UCD.legacy`; module callables pass `false` (latest tables).
const UCD_MODERN: ucd_core::Ucd = ucd_core::Ucd::new(true);
const UCD_LEGACY: ucd_core::Ucd = ucd_core::Ucd::new(false);

/// Residual choke: `Ucd::new` is a foreign constructor. The two views
/// are prebuilt consts (`interp_ucd.py` `_unicodedb` field).
#[majit_macros::dont_look_inside]
fn ucd(legacy: bool) -> &'static ucd_core::Ucd {
    if legacy { &UCD_LEGACY } else { &UCD_MODERN }
}

/// Extract the single-code-point argument of a character function.
///
/// `argno` is `Some(1)` for functions that also accept an optional default
/// (the argument clinic numbers those "argument 1"); `None` for single-argument
/// functions.  A non-string argument and a string not exactly one code point
/// long both raise `TypeError`, matching the clinic's converter messages.
fn extract_char(func: &str, argno: Option<u32>, obj: PyObjectRef) -> Result<CodePoint, PyError> {
    let argword = match argno {
        Some(n) => format!("argument {n}"),
        None => "argument".to_string(),
    };
    if !unsafe { is_str(obj) } {
        let ty = pyre_interpreter::baseobjspace::object_functionstr_type_name(obj);
        return Err(PyError::type_error(format!(
            "{func}() {argword} must be a unicode character, not {ty}"
        )));
    }
    let s = unsafe { w_str_get_wtf8(obj) };
    let mut cps = s.code_points();
    if let Some(cp) = cps.next()
        && cps.next().is_none()
    {
        return Ok(cp);
    }
    let n = s.code_points().count();
    Err(PyError::type_error(format!(
        "{func}(): {argword} must be a unicode character, not a string of length {n}"
    )))
}

/// Single required character argument (`category`, `bidirectional`, …).
fn one_char(func: &str, args: &[PyObjectRef]) -> Result<CodePoint, PyError> {
    let (args, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
    if pyre_interpreter::builtins::has_real_kwargs(kwargs) {
        return Err(PyError::type_error(format!(
            "unicodedata.{func}() takes no keyword arguments"
        )));
    }
    if args.len() != 1 {
        return Err(PyError::type_error(format!(
            "unicodedata.{func}() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    extract_char(func, None, args[0])
}

/// Character argument plus an optional default (`digit`, `decimal`, `numeric`,
/// `name`).
fn char_and_default(
    func: &str,
    args: &[PyObjectRef],
) -> Result<(CodePoint, Option<PyObjectRef>), PyError> {
    // UCD instance methods still arrive with a packed marker; the
    // module-level Signature path has already bound and filled omitted
    // slots with PY_NULL.
    let (args, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
    if pyre_interpreter::builtins::has_real_kwargs(kwargs) {
        return Err(PyError::type_error(format!(
            "unicodedata.{func}() takes no keyword arguments"
        )));
    }
    let given = args.iter().filter(|a| !a.is_null()).count();
    if given == 0 {
        return Err(PyError::type_error(format!(
            "{func} expected at least 1 argument, got 0"
        )));
    }
    if given > 2 {
        return Err(PyError::type_error(format!(
            "{func} expected at most 2 arguments, got {given}"
        )));
    }
    let cp = extract_char(func, Some(1), args[0])?;
    Ok((cp, args.get(1).copied().filter(|value| !value.is_null())))
}

// Version-sensitive queries take a `Ucd` built from `W_UCD.legacy`
// (`interp_ucd.py` `self._unicodedb`).

fn category_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    Ok(w_str_new_managed(db.category(one_char("category", args)?)))
}
pub(super) fn category(args: &[PyObjectRef]) -> PyResult {
    category_impl(ucd(false), args)
}

fn bidirectional_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    Ok(w_str_new_managed(
        db.bidirectional(one_char("bidirectional", args)?),
    ))
}
pub(super) fn bidirectional(args: &[PyObjectRef]) -> PyResult {
    bidirectional_impl(ucd(false), args)
}

fn east_asian_width_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    Ok(w_str_new_managed(
        db.east_asian_width(one_char("east_asian_width", args)?),
    ))
}
pub(super) fn east_asian_width(args: &[PyObjectRef]) -> PyResult {
    east_asian_width_impl(ucd(false), args)
}

fn combining_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    Ok(w_int_new(db.combining(one_char("combining", args)?) as i64))
}
pub(super) fn combining(args: &[PyObjectRef]) -> PyResult {
    combining_impl(ucd(false), args)
}

fn mirrored_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    Ok(w_int_new(db.mirrored(one_char("mirrored", args)?) as i64))
}
pub(super) fn mirrored(args: &[PyObjectRef]) -> PyResult {
    mirrored_impl(ucd(false), args)
}

fn decomposition_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    Ok(w_str_new_managed(
        &db.decomposition(one_char("decomposition", args)?),
    ))
}
pub(super) fn decomposition(args: &[PyObjectRef]) -> PyResult {
    decomposition_impl(ucd(false), args)
}

fn digit_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    let (cp, default) = char_and_default("digit", args)?;
    match db.digit(cp) {
        Some(v) => Ok(w_int_new(v as i64)),
        None => default.ok_or_else(|| PyError::value_error("not a digit")),
    }
}
pub(super) fn digit(args: &[PyObjectRef]) -> PyResult {
    digit_impl(ucd(false), args)
}

fn decimal_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    let (cp, default) = char_and_default("decimal", args)?;
    match db.decimal(cp) {
        Some(v) => Ok(w_int_new(v as i64)),
        None => default.ok_or_else(|| PyError::value_error("not a decimal")),
    }
}
pub(super) fn decimal(args: &[PyObjectRef]) -> PyResult {
    decimal_impl(ucd(false), args)
}

fn numeric_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    let (cp, default) = char_and_default("numeric", args)?;
    match db.numeric(cp) {
        Some(v) => Ok(w_float_new(v)),
        None => default.ok_or_else(|| PyError::value_error("not a numeric character")),
    }
}
pub(super) fn numeric(args: &[PyObjectRef]) -> PyResult {
    numeric_impl(ucd(false), args)
}

/// Short general-category name of an unassigned code point.
const UNASSIGNED_CATEGORY: &str = "Cn";

fn name_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    let (cp, default) = char_and_default("name", args)?;
    // A view has no name for a code point its own release leaves unassigned,
    // so the version's general category gates the name table: `ucd_3_2_0.name`
    // rejects characters assigned after Unicode 3.2.0.
    if db.category(cp) != UNASSIGNED_CATEGORY
        && let Some(name) = cp.to_char().and_then(ucd_core::character_name)
    {
        return Ok(w_str_new_managed(&name));
    }
    default.ok_or_else(|| PyError::value_error("no such name"))
}
pub(super) fn name(args: &[PyObjectRef]) -> PyResult {
    name_impl(ucd(false), args)
}

// Version-independent queries: `normalize` / `is_normalized` do not depend
// on the database version, so the 3.2.0 instance binds the same callables.
// `lookup` is versioned: aliases and named sequences exist only on the
// modern UCD.

fn lookup_impl(db: &ucd_core::Ucd, args: &[PyObjectRef]) -> PyResult {
    if args.len() != 1 {
        return Err(PyError::type_error(format!(
            "lookup expected 1 argument, got {}",
            args.len()
        )));
    }
    let obj = args[0];
    if !unsafe { is_str(obj) } {
        let ty = pyre_interpreter::baseobjspace::object_functionstr_type_name(obj);
        return Err(PyError::type_error(format!(
            "lookup() argument must be str, not {ty}"
        )));
    }
    let name = unsafe { w_str_get_wtf8(obj) };
    if let Ok(name) = name.as_str()
        && let Some(found) = db.lookup(name)
    {
        let text = match found {
            ucd_core::LookupResult::Character(ch) => {
                let mut buf = String::with_capacity(ch.len_utf8());
                buf.push(ch);
                buf
            }
            ucd_core::LookupResult::Sequence(seq) => seq.to_string(),
        };
        return Ok(w_str_new_managed(&text));
    }
    let mut msg = Wtf8Buf::from_string("undefined character name '".to_string());
    msg.push_wtf8(name);
    msg.push_str("'");
    Err(PyError::key_error(msg))
}

pub(super) fn lookup(args: &[PyObjectRef]) -> PyResult {
    lookup_impl(ucd(false), args)
}

/// Parse the normalization-form argument (`NFC`/`NFKC`/`NFD`/`NFKD`).
fn normalize_form(func: &str, obj: PyObjectRef) -> Result<NormalizeForm, PyError> {
    if !unsafe { is_str(obj) } {
        let ty = pyre_interpreter::baseobjspace::object_functionstr_type_name(obj);
        return Err(PyError::type_error(format!(
            "{func}() argument 1 must be str, not {ty}"
        )));
    }
    let s = unsafe { w_str_get_wtf8(obj) };
    // `interp_ucd.py UCD.normalize` selects the four forms with an explicit
    // source-order string branch.  Keep that graph shape instead of routing
    // through Rust's generic `FromStr::from_str` + `Result::map_err`, which
    // has no RPython counterpart and hides the useful constant comparisons
    // behind two foreign generic calls.
    if s == "NFC" {
        Ok(NormalizeForm::Nfc)
    } else if s == "NFD" {
        Ok(NormalizeForm::Nfd)
    } else if s == "NFKC" {
        Ok(NormalizeForm::Nfkc)
    } else if s == "NFKD" {
        Ok(NormalizeForm::Nfkd)
    } else {
        Err(PyError::value_error("invalid normalization form"))
    }
}

/// Borrow the string argument to be normalized (`unistr`).
fn normalize_text(func: &str, obj: PyObjectRef) -> Result<&'static Wtf8, PyError> {
    if !unsafe { is_str(obj) } {
        let ty = pyre_interpreter::baseobjspace::object_functionstr_type_name(obj);
        return Err(PyError::type_error(format!(
            "{func}() argument 2 must be str, not {ty}"
        )));
    }
    Ok(unsafe { w_str_get_wtf8(obj) })
}

pub(super) fn normalize(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 2 {
        return Err(PyError::type_error(format!(
            "normalize expected 2 arguments, got {}",
            args.len()
        )));
    }
    let form = normalize_form("normalize", args[0])?;
    let text = normalize_text("normalize", args[1])?;
    // interp_ucd.py returns `space.newutf8(content, strlen)` for
    // ASCII.  PyPy's W_UnicodeObject.is_w treats wrappers sharing that UTF-8
    // buffer as identical; returning the exact base str preserves that O(1)
    // identity without copying Pyre's owned Wtf8Buf.  A subclass must still
    // be converted to a base str, just like `space.newutf8`.
    if text.as_bytes().is_ascii()
        && unsafe { pyre_object::is_exact_type(args[1], &pyre_object::STR_TYPE) }
    {
        return Ok(args[1]);
    }
    Ok(w_str_from_wtf8_managed(ucd_core::normalize(form, text)))
}

pub(super) fn is_normalized(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 2 {
        return Err(PyError::type_error(format!(
            "is_normalized expected 2 arguments, got {}",
            args.len()
        )));
    }
    let form = normalize_form("is_normalized", args[0])?;
    let text = normalize_text("is_normalized", args[1])?;
    Ok(w_bool_from(ucd_core::is_normalized(form, text)))
}

/// PyPy `interp_ucd.py:UCD` — a Unicode database view whose methods are
/// descriptors on the type, rather than entries in an instance dictionary.
// CPython 3.14 Modules/unicodedata.c:PyInit_unicodedata uses PyType_FromSpec;
// ucd_type_spec carries IMMUTABLETYPE.
#[pyre_interpreter::pyre_class("unicodedata.UCD", cpython_heaptype)]
pub struct W_UCD {
    legacy: bool,
}

#[inline]
fn ucd_method_args(args: &[PyObjectRef]) -> &[PyObjectRef] {
    args.get(1..).unwrap_or(&[])
}

#[pyre_interpreter::pyre_methods(doc = "Unicode character database.")]
impl W_UCD {
    fn category(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        category_impl(ucd(self.legacy), ucd_method_args(args))
    }
    fn bidirectional(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        bidirectional_impl(ucd(self.legacy), ucd_method_args(args))
    }
    fn east_asian_width(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        east_asian_width_impl(ucd(self.legacy), ucd_method_args(args))
    }
    fn combining(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        combining_impl(ucd(self.legacy), ucd_method_args(args))
    }
    fn mirrored(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        mirrored_impl(ucd(self.legacy), ucd_method_args(args))
    }
    fn decomposition(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        decomposition_impl(ucd(self.legacy), ucd_method_args(args))
    }
    fn digit(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        digit_impl(ucd(self.legacy), ucd_method_args(args))
    }
    fn decimal(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        decimal_impl(ucd(self.legacy), ucd_method_args(args))
    }
    fn numeric(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        numeric_impl(ucd(self.legacy), ucd_method_args(args))
    }
    fn name(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        name_impl(ucd(self.legacy), ucd_method_args(args))
    }
    fn lookup(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        lookup_impl(ucd(self.legacy), ucd_method_args(args))
    }
    fn normalize(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        normalize(ucd_method_args(args))
    }
    fn is_normalized(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        is_normalized(ucd_method_args(args))
    }

    #[getter]
    fn unidata_version(&self) -> PyObjectRef {
        if self.legacy {
            w_str_new("3.2.0")
        } else {
            w_str_new(&ucd_core::unicode_version())
        }
    }
}

/// `interp_ucd.py ucd_3_2_0`: the prebuilt `UCD` instance pinned to the
/// Unicode 3.2.0 database.  The caller roots it before storing it.
pub(super) fn ucd_3_2_0() -> PyObjectRef {
    W_UCD::allocate_stable(W_UCD {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        legacy: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ucd_legacy_instance_uses_typed_descriptors() {
        pyre_interpreter::test_hooks::install_hash_hook();
        pyre_interpreter::typedef::init_typeobjects();
        let _ = type_object();
        let obj = W_UCD::allocate_stable(W_UCD {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            legacy: true,
        });
        let category = W_UCD::from_obj(obj)
            .expect("typed UCD instance")
            .category(&[obj, w_str_new("A")])
            .expect("category");
        assert_eq!(unsafe { w_str_get_wtf8(category) }.to_string(), "Lu");
        assert_eq!(
            unsafe { w_str_get_wtf8(W_UCD::from_obj(obj).unwrap().unidata_version()) }.to_string(),
            "3.2.0"
        );
    }

    #[test]
    fn ucd_legacy_name_rejects_later_assignments() {
        pyre_interpreter::test_hooks::install_hash_hook();
        pyre_interpreter::typedef::init_typeobjects();
        let _ = type_object();
        let obj = W_UCD::allocate_stable(W_UCD {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            legacy: true,
        });
        // U+0221 was assigned in Unicode 4.0, so the 3.2.0 view has no name
        // for it while the latest view does.
        let ucd = W_UCD::from_obj(obj).expect("typed UCD instance");
        assert!(ucd.name(&[obj, w_str_new("\u{221}")]).is_err());
        let modern = name(&[w_str_new("\u{221}")]).expect("name");
        assert_eq!(
            unsafe { w_str_get_wtf8(modern) }.to_string(),
            "LATIN SMALL LETTER D WITH CURL"
        );
    }
}
