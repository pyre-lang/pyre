//! W_BaseException — Python exception instance.
//!
//! Each exception carries a `kind` tag (mapping to PyErrorKind) and a
//! message string. `ob_type` is the instance layout vtable, not the Python
//! class: `allocate_instance` (`objspace.py`) stamps the realbase interp
//! class when `w_class` is that realbase's own type, and the realbase's
//! `_getusercls` layout (`typedef.py`) otherwise. `_new_exception` classes
//! (`W_ValueError`, `W_KeyError`, `W_Exception`, ...) are never the
//! allocated layout. `w_class` is the Python class. `EXCEPTION_TYPE` is the
//! BaseException root; `is_exception` is an `ll_isinstance` against it via
//! the assigned `subclassrange_{min,max}`.

use crate::pyobject::*;
use rustpython_wtf8::Wtf8;

/// Per-kind class vtable. Instance allocation does not use this pointer
/// unless `w_class` is exactly that kind's realbase (`exc_instance_pytype`).
/// `W_BaseException.typedef` is hasdict and not weakrefable, so the vtable
/// publishes neither a mapdict offset nor a `_lifeline_` field.
const fn exc_pytype(name: &'static str) -> PyType {
    crate::pyobject::new_pytype(name)
}

pub static EXCEPTION_TYPE: PyType = exc_pytype("BaseException");
/// PyPy `interp_group.W_BaseExceptionGroup` is a concrete interpreter class
/// over `W_BaseException`, so its TypeDef owns a child instance Layout.
///
/// Pyre flattens the group's fields into [`W_BaseException`], and group
/// instances therefore keep the BaseException/Exception `ob_type` selected by
/// their leaf policy.  This static is consequently a TypeDef/Layout identity,
/// not a separately allocated object vtable, and is deliberately absent from
/// `pyobject::all_foreign_pytypes`.
pub static EXC_BASE_EXCEPTION_GROUP_LAYOUT_TYPE: PyType =
    crate::pyobject::new_pytype("BaseExceptionGroup");
pub static EXC_EXCEPTION_TYPE: PyType = exc_pytype("Exception");
pub static EXC_ARITHMETIC_ERROR_TYPE: PyType = exc_pytype("ArithmeticError");
pub static EXC_OVERFLOW_ERROR_TYPE: PyType = exc_pytype("OverflowError");
pub static EXC_ZERO_DIVISION_ERROR_TYPE: PyType = exc_pytype("ZeroDivisionError");
pub static EXC_TYPE_ERROR_TYPE: PyType = exc_pytype("TypeError");
pub static EXC_VALUE_ERROR_TYPE: PyType = exc_pytype("ValueError");
pub static EXC_NAME_ERROR_TYPE: PyType = exc_pytype("NameError");
pub static EXC_UNBOUND_LOCAL_ERROR_TYPE: PyType = exc_pytype("UnboundLocalError");
pub static EXC_INDEX_ERROR_TYPE: PyType = exc_pytype("IndexError");
pub static EXC_KEY_ERROR_TYPE: PyType = exc_pytype("KeyError");
pub static EXC_ATTRIBUTE_ERROR_TYPE: PyType = exc_pytype("AttributeError");
pub static EXC_RUNTIME_ERROR_TYPE: PyType = exc_pytype("RuntimeError");
pub static EXC_STOP_ITERATION_TYPE: PyType = exc_pytype("StopIteration");
pub static EXC_STOP_ASYNC_ITERATION_TYPE: PyType = exc_pytype("StopAsyncIteration");
pub static EXC_IMPORT_ERROR_TYPE: PyType = exc_pytype("ImportError");
pub static EXC_MODULE_NOT_FOUND_ERROR_TYPE: PyType = exc_pytype("ModuleNotFoundError");
pub static EXC_NOT_IMPLEMENTED_ERROR_TYPE: PyType = exc_pytype("NotImplementedError");
pub static EXC_ASSERTION_ERROR_TYPE: PyType = exc_pytype("AssertionError");
pub static EXC_REFERENCE_ERROR_TYPE: PyType = exc_pytype("ReferenceError");
pub static EXC_GENERATOR_EXIT_TYPE: PyType = exc_pytype("GeneratorExit");
pub static EXC_RECURSION_ERROR_TYPE: PyType = exc_pytype("RecursionError");
pub static EXC_OS_ERROR_TYPE: PyType = exc_pytype("OSError");
pub static EXC_FILE_NOT_FOUND_ERROR_TYPE: PyType = exc_pytype("FileNotFoundError");
pub static EXC_UNICODE_DECODE_ERROR_TYPE: PyType = exc_pytype("UnicodeDecodeError");
pub static EXC_UNICODE_ENCODE_ERROR_TYPE: PyType = exc_pytype("UnicodeEncodeError");
/// PyPy `pypy/module/exceptions/interp_exceptions.py W_UnicodeTranslateError
/// W_UnicodeTranslateError = _new_exception('UnicodeTranslateError',
/// W_UnicodeError, ...)` — subclass of UnicodeError.  A dedicated PyType
/// + ExcKind for isinstance / `ob_type` discrimination, with the 4-arg
/// `(object, start, end, reason)` init signature and the class's own
/// `__str__` formatting.  See the `ExcKind::UnicodeTranslateError` doc
/// for the field flattening its payload slots follow.
pub static EXC_UNICODE_TRANSLATE_ERROR_TYPE: PyType = exc_pytype("UnicodeTranslateError");
pub static EXC_SYSTEM_EXIT_TYPE: PyType = exc_pytype("SystemExit");
pub static EXC_MEMORY_ERROR_TYPE: PyType = exc_pytype("MemoryError");
pub static EXC_SYSTEM_ERROR_TYPE: PyType = exc_pytype("SystemError");
/// PyPy `W_EOFError`, a direct `Exception` subclass used by stream readers
/// such as pickle and marshal.
pub static EXC_EOF_ERROR_TYPE: PyType = exc_pytype("EOFError");
/// `BufferError` — raised when an operation cannot proceed because a
/// buffer is exported (e.g. resizing a bytearray that backs a live
/// memoryview).  Direct subclass of Exception.
pub static EXC_BUFFER_ERROR_TYPE: PyType = exc_pytype("BufferError");
/// PyPy `pypy/module/exceptions/interp_exceptions.py:474
/// W_LookupError = _new_exception('LookupError', W_Exception, ...)`
/// — intermediate parent for IndexError and KeyError.
pub static EXC_LOOKUP_ERROR_TYPE: PyType = exc_pytype("LookupError");
/// PyPy `pypy/module/exceptions/interp_exceptions.py:418
/// W_UnicodeError = _new_exception('UnicodeError', W_ValueError, ...)`
/// — intermediate parent for UnicodeDecodeError and UnicodeEncodeError.
pub static EXC_UNICODE_ERROR_TYPE: PyType = exc_pytype("UnicodeError");
/// `pypy/module/exceptions/interp_exceptions.py W_SyntaxError` — subclass
/// of Exception raised by `compile`/`exec`/`eval`/`ast.parse`.
pub static EXC_SYNTAX_ERROR_TYPE: PyType = exc_pytype("SyntaxError");

/// Per-`ExcKind` class-identity vtable. Instance allocation uses
/// [`exc_instance_pytype`]. `allocate_exception`'s exact-group arm is the
/// one caller that still stamps this pointer onto a slim kind.
#[inline]
pub fn exc_kind_to_pytype(kind: ExcKind) -> &'static PyType {
    match kind {
        ExcKind::BaseException => &EXCEPTION_TYPE,
        ExcKind::Exception => &EXC_EXCEPTION_TYPE,
        ExcKind::ArithmeticError => &EXC_ARITHMETIC_ERROR_TYPE,
        ExcKind::OverflowError => &EXC_OVERFLOW_ERROR_TYPE,
        ExcKind::ZeroDivisionError => &EXC_ZERO_DIVISION_ERROR_TYPE,
        ExcKind::TypeError => &EXC_TYPE_ERROR_TYPE,
        ExcKind::ValueError => &EXC_VALUE_ERROR_TYPE,
        ExcKind::NameError => &EXC_NAME_ERROR_TYPE,
        ExcKind::UnboundLocalError => &EXC_UNBOUND_LOCAL_ERROR_TYPE,
        ExcKind::IndexError => &EXC_INDEX_ERROR_TYPE,
        ExcKind::KeyError => &EXC_KEY_ERROR_TYPE,
        ExcKind::AttributeError => &EXC_ATTRIBUTE_ERROR_TYPE,
        ExcKind::RuntimeError => &EXC_RUNTIME_ERROR_TYPE,
        ExcKind::StopIteration => &EXC_STOP_ITERATION_TYPE,
        ExcKind::StopAsyncIteration => &EXC_STOP_ASYNC_ITERATION_TYPE,
        ExcKind::ImportError => &EXC_IMPORT_ERROR_TYPE,
        ExcKind::ModuleNotFoundError => &EXC_MODULE_NOT_FOUND_ERROR_TYPE,
        ExcKind::NotImplementedError => &EXC_NOT_IMPLEMENTED_ERROR_TYPE,
        ExcKind::AssertionError => &EXC_ASSERTION_ERROR_TYPE,
        ExcKind::ReferenceError => &EXC_REFERENCE_ERROR_TYPE,
        ExcKind::GeneratorExit => &EXC_GENERATOR_EXIT_TYPE,
        ExcKind::RecursionError => &EXC_RECURSION_ERROR_TYPE,
        ExcKind::OSError => &EXC_OS_ERROR_TYPE,
        ExcKind::FileNotFoundError => &EXC_FILE_NOT_FOUND_ERROR_TYPE,
        ExcKind::UnicodeDecodeError => &EXC_UNICODE_DECODE_ERROR_TYPE,
        ExcKind::UnicodeEncodeError => &EXC_UNICODE_ENCODE_ERROR_TYPE,
        ExcKind::SystemExit => &EXC_SYSTEM_EXIT_TYPE,
        ExcKind::MemoryError => &EXC_MEMORY_ERROR_TYPE,
        ExcKind::SystemError => &EXC_SYSTEM_ERROR_TYPE,
        ExcKind::EOFError => &EXC_EOF_ERROR_TYPE,
        ExcKind::BufferError => &EXC_BUFFER_ERROR_TYPE,
        ExcKind::LookupError => &EXC_LOOKUP_ERROR_TYPE,
        ExcKind::UnicodeError => &EXC_UNICODE_ERROR_TYPE,
        ExcKind::UnicodeTranslateError => &EXC_UNICODE_TRANSLATE_ERROR_TYPE,
        ExcKind::SyntaxError => &EXC_SYNTAX_ERROR_TYPE,
    }
}

/// Numeric tags for exception kinds — must stay in sync with PyErrorKind.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExcKind {
    BaseException = 0,
    Exception = 1,
    TypeError = 2,
    ValueError = 3,
    ZeroDivisionError = 4,
    NameError = 5,
    IndexError = 6,
    KeyError = 7,
    AttributeError = 8,
    RuntimeError = 9,
    // The jd1 drain-match fusion bakes this discriminant as a literal
    // `ConstInt(10)` kind test (majit-translate front/result_exc.rs); keep the
    // two in sync — renumbering silently miscompiles the fused break test.
    StopIteration = 10,
    OverflowError = 11,
    ArithmeticError = 12,
    ImportError = 13,
    NotImplementedError = 14,
    AssertionError = 15,
    /// Raised by `_weakref` when a proxy is dereferenced after the
    /// referent has been collected — pypy/module/_weakref/interp__weakref.py
    /// `oefmt(space.w_ReferenceError, "weakly referenced object no longer exists")`.
    /// `interp__weakref::force` raises the hyphenated `proxy_check_ref`
    /// wording instead.
    ReferenceError = 16,
    GeneratorExit = 17,
    RecursionError = 18,
    /// Base class for all operating-system errors
    /// (formerly IOError / WindowsError / EnvironmentError in Python 2).
    /// pypy/module/exceptions/interp_exceptions.py W_OSError.
    OSError = 19,
    /// Subclass of OSError raised when a file or directory is not found.
    FileNotFoundError = 20,
    /// Subclass of ValueError raised by codecs on invalid input.
    UnicodeDecodeError = 21,
    /// Subclass of ValueError raised by codecs on invalid input.
    UnicodeEncodeError = 22,
    /// Raised by sys.exit(). Subclass of BaseException, not Exception.
    SystemExit = 23,
    /// rpython/jit/metainterp/compile.py `memory_error = MemoryError()`
    /// — module-level singleton instance the JIT raises through
    /// `PropagateExceptionDescr.handle_fail` when a malloc helper
    /// returns NULL.  Subclass of Exception per
    /// pypy/module/exceptions/interp_exceptions.py.
    MemoryError = 24,
    /// `pypy/module/exceptions/interp_exceptions.py W_SystemError` —
    /// raised by interpreter-internal invariants (e.g.
    /// `chain_exceptions` rejecting non-BaseException context).
    SystemError = 25,
    /// `pypy/module/exceptions/interp_exceptions.py:474
    /// W_LookupError = _new_exception('LookupError', W_Exception, ...)`
    /// — intermediate parent for IndexError and KeyError.
    LookupError = 26,
    /// `pypy/module/exceptions/interp_exceptions.py:418
    /// W_UnicodeError = _new_exception('UnicodeError', W_ValueError, ...)`
    /// — intermediate parent for UnicodeDecodeError and
    /// UnicodeEncodeError.
    UnicodeError = 27,
    /// `pypy/module/exceptions/interp_exceptions.py W_UnicodeTranslateError
    /// W_UnicodeTranslateError = _new_exception('UnicodeTranslateError',
    /// W_UnicodeError, ...)`.  A dedicated kind so `ob_type` and
    /// `isinstance` discriminate it correctly, with the 4-arg
    /// `(object, start, end, reason)` `__init__` and the class's own
    /// `__str__`.
    ///
    /// Pyre takes the "union of all per-class fields" route: a single
    /// GC type id for `W_BaseException`, with every per-subclass slot
    /// flattened onto it.  W_UnicodeDecodeError / W_UnicodeEncodeError /
    /// W_UnicodeTranslateError carry `w_object`/`w_start`/`w_end`/
    /// `w_reason`/`w_encoding`; W_OSError carries `w_errno`/`w_strerror`/
    /// `w_filename`/`w_filename2`; W_StopIteration carries `w_value`;
    /// W_ImportError carries `w_exc_name`/`w_import_path`/`w_import_msg`;
    /// W_AttributeError carries `w_exc_name`/`w_attr_obj`.  The
    /// alternative — per-subclass `W_<Kind>Object` structs, one GC type
    /// id per kind with isolated layouts — would be more PyPy-orthodox
    /// but is not implemented.
    UnicodeTranslateError = 28,
    /// Subclass of ImportError raised when a module cannot be found.
    /// Reads ImportError's flattened `w_exc_name` / `w_import_path` /
    /// `w_import_msg` slots.
    ModuleNotFoundError = 29,
    /// `pypy/module/exceptions/interp_exceptions.py W_SyntaxError` —
    /// raised by `compile` / `exec` / `eval` / `ast.parse` on malformed
    /// source.  A dedicated kind so `ob_type` and
    /// `isinstance(e, SyntaxError)` discriminate it, with the
    /// `(msg, (filename, lineno, offset, text))` `__init__` writing the
    /// flattened `w_syntax_msg` / `w_syntax_filename` /
    /// `w_syntax_lineno` / `w_syntax_offset` / `w_syntax_text` slots.
    SyntaxError = 30,
    /// Raised when a buffer-related operation cannot proceed — e.g.
    /// resizing a `bytearray` whose storage backs a live `memoryview`
    /// (`PyByteArray_Resize`: "Existing exports of data: object cannot
    /// be re-sized").  Direct subclass of Exception.
    BufferError = 31,
    /// Subclass of NameError raised when a fast local is read while unbound.
    UnboundLocalError = 32,
    /// Signals exhaustion of an asynchronous iterator.  Appended so the
    /// existing discriminants embedded by the JIT remain stable.
    StopAsyncIteration = 33,
    /// Direct Exception subclass raised when a stream ends before an object.
    /// Appended so existing JIT-baked discriminants remain stable.
    EOFError = 34,
}

impl ExcKind {
    /// The largest valid discriminant.  `PyError::kind_from_exc` matches
    /// every variant with no wildcard arm, so the compiler is free to lower
    /// it to a bounds-check-free jump table over exactly `0..=` this value;
    /// a byte outside the range reaching that match is an indirect branch to
    /// an address computed from garbage.  Anything reading the tag out of a
    /// value whose provenance is not proven must go through
    /// `w_exception_kind_checked`, which range-checks against this.
    pub const MAX_DISCRIMINANT: u8 = ExcKind::EOFError as u8;
}

/// Layout: `[ob_header | kind: ExcKind | args_w | w_cause | w_context |
/// w_traceback | suppress_context | w_dict]`.
///
/// Matches `interp_exceptions.py W_BaseException`. `_new_exception` classes
/// that add no instance fields allocate [`W_BaseExceptionUser`] instead.
/// Subclasses that declare extra slots live in [`W_ExceptionExtended`] or
/// [`W_ExceptionExtendedUser`].
///
/// `args_w` mirrors `W_BaseException.descr_init`:
///
/// ```python
/// def descr_init(self, space, args_w):
///     self.args_w = args_w
/// ```
///
/// The stored list is fixed-size. `Arguments.__init__` calls
/// `make_sure_not_resized` on `arguments_w`, `descr_new` assigns that
/// list, and `descr_setargs` assigns `space.fixedview`. `FixedSizeListRepr`
/// lowers to the item `GcArray` (`ll_fixed_newlist`), so the slot points
/// at an [`crate::object_array::ItemsBlock`], not the resizable LIST
/// header. `w_exception_get_args` is `descr_getargs`
/// (`return space.newtuple(self.args_w)`): a new tuple header each read.
/// Length 2 uses `makespecialisedtuple`. Every other length stores this
/// array as `W_TupleObject.wrappeditems`.
///
/// `PY_NULL` means "not yet set" — the `args` getattr arm surfaces an
/// empty tuple in that case, matching the path where the constructor
/// is bypassed (e.g. internal `w_exception_new` callers in
/// `gateway.rs`).
#[repr(C)]
pub struct W_BaseException {
    pub ob_header: PyObject,
    pub kind: ExcKind,
    pub args_w: PyObjectRef,
    /// `interp_exceptions.py W_BaseException.w_cause = None` —
    /// `raise X from Y` cause set by `descr_setcause`.
    /// `PY_NULL` mirrors PyPy's "internal None" (raises AttributeError
    /// on read in CPython; PyPy returns `space.w_None`).
    pub w_cause: PyObjectRef,
    /// `interp_exceptions.py W_BaseException.w_context = None` —
    /// chained exception context set by `descr_setcontext`.
    pub w_context: PyObjectRef,
    /// `interp_exceptions.py W_BaseException.w_traceback = None` —
    /// traceback object stamped by `descr_settraceback`
    /// and the `raise` machinery via `OperationError.normalize_exception`.
    pub w_traceback: PyObjectRef,
    /// `interp_exceptions.py W_BaseException.suppress_context =
    /// False` — `raise X from Y` flips this to True via
    /// `descr_setcause`.
    pub suppress_context: bool,
    /// `interp_exceptions.py W_BaseException.w_dict = None` — the
    /// per-instance attribute dict, lazily allocated by `getdict`
    /// and replaced wholesale by `setdict`.
    /// Extra attributes (`e.note = ...`, PEP 678 `__notes__`) live
    /// here. `W_BaseException.typedef` is hasdict, so `_getusercls` does
    /// not mix `MapdictDictSupport`: ordinary attributes stay in this slot.
    pub w_dict: PyObjectRef,
}

/// Extra-field subclasses of `W_BaseException`.
///
/// PyPy gives each of `W_OSError`, `W_ImportError`, `W_SyntaxError`,
/// `W_Unicode*Error`, `W_StopIteration`, `W_NameError`,
/// `W_AttributeError`, `W_SystemExit`, and `W_BaseExceptionGroup` its
/// own interp-level class and SizeDescr.  Until those are split, they
/// share this prefix-compatible extended layout so a fieldless class
/// stays on the slim layout ([`W_BaseException`] or
/// [`W_BaseExceptionUser`]) instead of carrying every unused subclass slot.
#[repr(C)]
pub struct W_ExceptionExtended {
    pub base: W_BaseException,
    /// `interp_exceptions.py W_UnicodeTranslateError.w_object` /
    /// `W_UnicodeDecodeError.w_object` /
    /// `W_UnicodeEncodeError.w_object`.  The offending string /
    /// bytes object passed to `__init__`.  Populated by
    /// `descr_init`; `PY_NULL` for non-Unicode-error kinds and for
    /// Unicode errors constructed without going through the public
    /// `descr_init` path (matches PyPy's class-default `w_object = None`
    /// — `descr_str` checks `if self.object is None: return ""`).
    pub w_object: PyObjectRef,
    /// `interp_exceptions.py W_UnicodeTranslateError.w_start`, and
    /// `W_UnicodeDecodeError.w_start` / `W_UnicodeEncodeError.w_start`.
    pub w_start: PyObjectRef,
    /// `interp_exceptions.py W_UnicodeTranslateError.w_end`, and
    /// `W_UnicodeDecodeError.w_end` / `W_UnicodeEncodeError.w_end`.
    pub w_end: PyObjectRef,
    /// `interp_exceptions.py W_UnicodeTranslateError.w_reason`, and
    /// `W_UnicodeDecodeError.w_reason` / `W_UnicodeEncodeError.w_reason`.
    pub w_reason: PyObjectRef,
    /// `interp_exceptions.py W_UnicodeDecodeError.w_encoding` /
    /// `W_UnicodeEncodeError.w_encoding`.  `W_UnicodeTranslateError`
    /// has no `w_encoding` field per PyPy — left `PY_NULL` for Translate.
    pub w_encoding: PyObjectRef,
    /// `interp_exceptions.py W_OSError.w_errno` — writable
    /// `readwrite_attrproperty_w('w_errno', W_OSError)` slot.
    /// The slot is the value. `PY_NULL` is the class default `None`.
    /// `W_OSError.descr_new` / `_init_error` fills it.
    pub w_errno: PyObjectRef,
    /// `interp_exceptions.py W_OSError.w_winerror` — the Windows error
    /// code, exposed as the writable `winerror` attribute only on the platform
    /// that has one — `W_OSError.descr_new` gates it on `rwin32.WIN32`.
    /// PyPy declares the slot everywhere and reads it only under that gate;
    /// keeping it unconditional here leaves one exception layout for every
    /// target instead of a Windows-only field ordering.
    pub w_winerror: PyObjectRef,
    /// `interp_exceptions.py W_OSError.w_strerror` /
    /// `readwrite_attrproperty_w('w_strerror', W_OSError)`.
    /// The slot is the value. `PY_NULL` is the class default `None`.
    /// `W_OSError.descr_new` / `_init_error` fills it.
    pub w_strerror: PyObjectRef,
    /// `interp_exceptions.py W_OSError.w_filename` /
    /// `readwrite_attrproperty_w('w_filename', W_OSError)`.
    /// The slot is the value. `PY_NULL` is the class default `None`.
    /// `W_OSError.descr_new` / `_init_error` fills it.
    pub w_filename: PyObjectRef,
    /// `interp_exceptions.py W_OSError.w_filename2` /
    /// `readwrite_attrproperty_w('w_filename2', W_OSError)`.
    /// The slot is the value. `PY_NULL` is the class default `None`.
    /// `W_OSError.descr_new` / `_init_error` fills it.
    pub w_filename2: PyObjectRef,
    /// `interp_exceptions.py W_OSError.written = -1` — the independent
    /// integer slot exposed by the `characters_written` GetSetProperty.
    /// A numeric third argument on an exact BlockingIOError stamps it; later
    /// descriptor writes and deletes mutate this slot without changing args.
    pub written: i64,
    /// `interp_exceptions.py W_SystemExit.w_code` /
    /// `readwrite_attrproperty_w('w_code', W_SystemExit)`.
    /// The slot is the value. `PY_NULL` is the class default `None`.
    /// `W_SystemExit.descr_init` fills it.
    pub w_code: PyObjectRef,
    /// `interp_exceptions.py W_StopIteration.w_value` — initialized to
    /// None, replaced with the first argument by `descr_init`, and exposed as
    /// the writable `value` attribute.
    pub w_value: PyObjectRef,
    /// Shared `w_name` slot for the exception kinds that expose a
    /// `name` attribute: `W_ImportError.w_name`
    /// (`interp_exceptions.py`
    /// `readwrite_attrproperty_w('w_name', W_ImportError)`),
    /// `W_NameError.w_name` and `W_AttributeError.w_name` (Python
    /// 3.10+).  An exception is exactly one kind, so a single slot
    /// serves all three.  `PY_NULL` is the class default `None`; set
    /// from the `name=` keyword by `descr_init` and writable via
    /// `e.name = ...`.
    pub w_exc_name: PyObjectRef,
    /// `W_AttributeError.w_obj` — the object whose attribute lookup
    /// failed, set from the `obj=` keyword (Python 3.10+), default
    /// `None`.
    pub w_attr_obj: PyObjectRef,
    /// `interp_exceptions.py W_ImportError.w_path` /
    /// `readwrite_attrproperty_w('w_path', W_ImportError)`, set
    /// from the `path=` keyword.
    pub w_import_path: PyObjectRef,
    /// `W_ImportError.w_name_from` — set from the `name_from=` keyword,
    /// exposed as the `name_from` attribute (default `None`).
    pub w_import_name_from: PyObjectRef,
    /// `interp_exceptions.py W_ImportError.w_msg` /
    /// `readwrite_attrproperty_w('w_msg', W_ImportError)`.  Set to the
    /// single positional argument by `descr_init`; read back by the `msg`
    /// attrproperty; class default `None`.
    pub w_import_msg: PyObjectRef,
    /// `interp_exceptions.py W_SyntaxError` per-instance fields.
    /// PyPy keeps these on `W_SyntaxError`; pyre's shared exception GC layout
    /// flattens subclass payloads onto `W_BaseException`, as it does for the
    /// Unicode and OSError families above.
    pub w_syntax_msg: PyObjectRef,
    pub w_syntax_filename: PyObjectRef,
    pub w_syntax_lineno: PyObjectRef,
    pub w_syntax_offset: PyObjectRef,
    pub w_syntax_text: PyObjectRef,
    pub w_syntax_end_lineno: PyObjectRef,
    pub w_syntax_end_offset: PyObjectRef,
    pub w_syntax_print_file_and_line: PyObjectRef,
    /// CPython 3.14's private `SyntaxError._metadata` member, a
    /// `(line, offset, source)` triple.  This is the 3.14-specific extension
    /// to PyPy's `W_SyntaxError` field set: the pinned stdlib reads it in
    /// `traceback.py` `StackSummary._extract_from_extended_frame_gen`
    /// (`self._exc_metadata = getattr(exc_value, "_metadata", None)`) and
    /// unpacks it in `TracebackException.format_exception_only`.
    pub w_syntax_metadata: PyObjectRef,
    /// `interp_group.py W_BaseExceptionGroup.descr_new` `exc.w_message`,
    /// exposed as the read-only `message` attrproperty
    /// (`interp_attrproperty_w('w_message', W_BaseExceptionGroup)`).
    pub w_group_message: PyObjectRef,
    /// `interp_group.py` `exc.w_exceptions`, exposed as the read-only
    /// `exceptions` attrproperty
    /// (`interp_attrproperty_w('w_exceptions', W_BaseExceptionGroup)`).
    /// An exact tuple is stored as itself (`W_TupleObject.descr_new`).
    /// Every other sequence is copied into a new tuple. Replacing `args`
    /// later does not rewrite this slot.
    pub w_group_exceptions: PyObjectRef,
    /// The `repr` of the sequence `descr_new` received, rendered before it was
    /// flattened into `w_group_exceptions`.  `BaseExceptionGroup.__repr__`
    /// reproduces the constructor-time spelling, which a later mutation of
    /// `args` must not change; `PY_NULL` selects the derive-from-args path.
    pub w_group_exceptions_repr: PyObjectRef,
}

/// `typedef.py` `_getusercls(W_BaseException)`: the base payload plus
/// `MapdictStorageMixin`. The typedef is hasdict, so the map is not the
/// instance dict; it carries `__slots__` and, because the typedef is not
/// weakrefable, the `MapdictWeakrefSupport` `"weakref"` SPECIAL slot.
#[repr(C)]
pub struct W_BaseExceptionUser {
    pub base: W_BaseException,
    pub map: usize,
    pub storage: *mut crate::object_array::ItemsBlock,
}

const _: () = {
    assert!(
        std::mem::offset_of!(W_BaseExceptionUser, storage)
            == std::mem::offset_of!(W_BaseExceptionUser, map) + std::mem::size_of::<usize>()
    );
};

/// `typedef.py` `_getusercls` of an extra-field realbase (`W_OSError`,
/// `W_ImportError`, `W_StopIteration`, ...). Same mapdict tail as
/// [`W_BaseExceptionUser`], prefixed by [`W_ExceptionExtended`].
#[repr(C)]
pub struct W_ExceptionExtendedUser {
    pub base: W_ExceptionExtended,
    pub map: usize,
    pub storage: *mut crate::object_array::ItemsBlock,
}

const _: () = {
    assert!(
        std::mem::offset_of!(W_ExceptionExtendedUser, storage)
            == std::mem::offset_of!(W_ExceptionExtendedUser, map) + std::mem::size_of::<usize>()
    );
};

/// Slim `_getusercls` typeptr (`typedef.py` `_getusercls(W_BaseException)`).
pub static BASE_EXCEPTION_USER_TYPE: PyType = new_user_pytype(
    "BaseException",
    &EXCEPTION_TYPE,
    std::mem::offset_of!(W_BaseExceptionUser, map),
);

/// Extra-field `_getusercls` typeptr. One vtable covers every extended
/// realbase's user layout; `w_class` names the Python class.
pub static EXCEPTION_EXTENDED_USER_TYPE: PyType = new_user_pytype(
    "BaseException",
    &EXCEPTION_TYPE,
    std::mem::offset_of!(W_ExceptionExtendedUser, map),
);

pub const EXC_KIND_OFFSET: usize = std::mem::offset_of!(W_BaseException, kind);
pub const EXC_ARGS_W_OFFSET: usize = std::mem::offset_of!(W_BaseException, args_w);
pub const EXC_W_CAUSE_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_cause);
pub const EXC_W_CONTEXT_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_context);
pub const EXC_W_TRACEBACK_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_traceback);
pub const EXC_SUPPRESS_CONTEXT_OFFSET: usize =
    std::mem::offset_of!(W_BaseException, suppress_context);
pub const EXC_W_DICT_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_dict);
pub const EXC_W_OBJECT_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtended, w_object);
pub const EXC_W_START_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtended, w_start);
pub const EXC_W_END_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtended, w_end);
pub const EXC_W_REASON_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtended, w_reason);
pub const EXC_W_ENCODING_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtended, w_encoding);
pub const EXC_W_ERRNO_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtended, w_errno);
pub const EXC_W_WINERROR_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtended, w_winerror);
pub const EXC_W_STRERROR_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtended, w_strerror);
pub const EXC_W_FILENAME_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtended, w_filename);
pub const EXC_W_FILENAME2_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtended, w_filename2);
pub const EXC_WRITTEN_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtended, written);
pub const EXC_W_CODE_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtended, w_code);
pub const EXC_W_VALUE_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtended, w_value);
pub const EXC_W_NAME_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtended, w_exc_name);
pub const EXC_W_ATTR_OBJ_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtended, w_attr_obj);
pub const EXC_W_IMPORT_PATH_OFFSET: usize =
    std::mem::offset_of!(W_ExceptionExtended, w_import_path);
pub const EXC_W_IMPORT_NAME_FROM_OFFSET: usize =
    std::mem::offset_of!(W_ExceptionExtended, w_import_name_from);
pub const EXC_W_IMPORT_MSG_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtended, w_import_msg);
pub const EXC_W_SYNTAX_MSG_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtended, w_syntax_msg);
pub const EXC_W_SYNTAX_FILENAME_OFFSET: usize =
    std::mem::offset_of!(W_ExceptionExtended, w_syntax_filename);
pub const EXC_W_SYNTAX_LINENO_OFFSET: usize =
    std::mem::offset_of!(W_ExceptionExtended, w_syntax_lineno);
pub const EXC_W_SYNTAX_OFFSET_OFFSET: usize =
    std::mem::offset_of!(W_ExceptionExtended, w_syntax_offset);
pub const EXC_W_SYNTAX_TEXT_OFFSET: usize =
    std::mem::offset_of!(W_ExceptionExtended, w_syntax_text);
pub const EXC_W_SYNTAX_END_LINENO_OFFSET: usize =
    std::mem::offset_of!(W_ExceptionExtended, w_syntax_end_lineno);
pub const EXC_W_SYNTAX_END_OFFSET_OFFSET: usize =
    std::mem::offset_of!(W_ExceptionExtended, w_syntax_end_offset);
pub const EXC_W_SYNTAX_PRINT_FILE_AND_LINE_OFFSET: usize =
    std::mem::offset_of!(W_ExceptionExtended, w_syntax_print_file_and_line);
pub const EXC_W_SYNTAX_METADATA_OFFSET: usize =
    std::mem::offset_of!(W_ExceptionExtended, w_syntax_metadata);
pub const EXC_W_GROUP_MESSAGE_OFFSET: usize =
    std::mem::offset_of!(W_ExceptionExtended, w_group_message);
pub const EXC_W_GROUP_EXCEPTIONS_OFFSET: usize =
    std::mem::offset_of!(W_ExceptionExtended, w_group_exceptions);
pub const EXC_W_GROUP_EXCEPTIONS_REPR_OFFSET: usize =
    std::mem::offset_of!(W_ExceptionExtended, w_group_exceptions_repr);
pub const EXC_USER_MAP_OFFSET: usize = std::mem::offset_of!(W_BaseExceptionUser, map);
pub const EXC_USER_STORAGE_OFFSET: usize = std::mem::offset_of!(W_BaseExceptionUser, storage);
pub const EXC_EXTENDED_USER_MAP_OFFSET: usize = std::mem::offset_of!(W_ExceptionExtendedUser, map);
pub const EXC_EXTENDED_USER_STORAGE_OFFSET: usize =
    std::mem::offset_of!(W_ExceptionExtendedUser, storage);

/// The pointer slots a traced construction emit must reproduce itself.
///
/// Values are read through their exported byte offsets so this census cannot
/// drift from the instance layout.  Slim [`W_BaseException`] kinds only
/// expose the base slots; extra-field kinds also census the extended
/// payload.
pub unsafe fn w_exception_traced_construction_slots(obj: PyObjectRef) -> Vec<(usize, PyObjectRef)> {
    const BASE_OFFSETS: [usize; 3] = [
        EXC_W_CAUSE_OFFSET,
        EXC_W_TRACEBACK_OFFSET,
        EXC_W_DICT_OFFSET,
    ];
    const EXTENDED_OFFSETS: [usize; 32] = [
        EXC_W_CAUSE_OFFSET,
        EXC_W_TRACEBACK_OFFSET,
        EXC_W_OBJECT_OFFSET,
        EXC_W_START_OFFSET,
        EXC_W_END_OFFSET,
        EXC_W_REASON_OFFSET,
        EXC_W_ENCODING_OFFSET,
        EXC_W_ERRNO_OFFSET,
        EXC_W_WINERROR_OFFSET,
        EXC_W_STRERROR_OFFSET,
        EXC_W_FILENAME_OFFSET,
        EXC_W_FILENAME2_OFFSET,
        EXC_W_CODE_OFFSET,
        EXC_W_VALUE_OFFSET,
        EXC_W_NAME_OFFSET,
        EXC_W_ATTR_OBJ_OFFSET,
        EXC_W_IMPORT_PATH_OFFSET,
        EXC_W_IMPORT_NAME_FROM_OFFSET,
        EXC_W_IMPORT_MSG_OFFSET,
        EXC_W_SYNTAX_MSG_OFFSET,
        EXC_W_SYNTAX_FILENAME_OFFSET,
        EXC_W_SYNTAX_LINENO_OFFSET,
        EXC_W_SYNTAX_OFFSET_OFFSET,
        EXC_W_SYNTAX_TEXT_OFFSET,
        EXC_W_SYNTAX_END_LINENO_OFFSET,
        EXC_W_SYNTAX_END_OFFSET_OFFSET,
        EXC_W_SYNTAX_PRINT_FILE_AND_LINE_OFFSET,
        EXC_W_SYNTAX_METADATA_OFFSET,
        EXC_W_GROUP_MESSAGE_OFFSET,
        EXC_W_GROUP_EXCEPTIONS_OFFSET,
        EXC_W_GROUP_EXCEPTIONS_REPR_OFFSET,
        EXC_W_DICT_OFFSET,
    ];
    let kind = unsafe { (*(obj as *const W_BaseException)).kind };
    let offsets: &[usize] = if exc_kind_uses_extended_layout(kind) {
        &EXTENDED_OFFSETS
    } else {
        &BASE_OFFSETS
    };
    let base = obj.cast::<u8>();
    offsets
        .iter()
        .map(|&offset| {
            let value = unsafe { base.add(offset).cast::<PyObjectRef>().read() };
            (offset, value)
        })
        .collect()
}

/// GC pointer slots on the slim [`W_BaseException`] layout
/// (`interp_exceptions.py W_BaseException` class defaults).
pub const W_BASE_EXCEPTION_GC_PTR_OFFSETS: [usize; 5] = [
    EXC_ARGS_W_OFFSET,
    EXC_W_CAUSE_OFFSET,
    EXC_W_CONTEXT_OFFSET,
    EXC_W_TRACEBACK_OFFSET,
    EXC_W_DICT_OFFSET,
];

/// Slim user layout: the base pointers plus mapdict `storage`.
/// `map` is a `usize`, not a GC edge.
pub const W_BASE_EXCEPTION_USER_GC_PTR_OFFSETS: [usize; 6] = [
    EXC_ARGS_W_OFFSET,
    EXC_W_CAUSE_OFFSET,
    EXC_W_CONTEXT_OFFSET,
    EXC_W_TRACEBACK_OFFSET,
    EXC_W_DICT_OFFSET,
    EXC_USER_STORAGE_OFFSET,
];

/// GC pointer slots on [`W_ExceptionExtended`] — the slim base plus every
/// extra-field subclass slot PyPy keeps on a dedicated interp class.
pub const W_EXCEPTION_EXTENDED_GC_PTR_OFFSETS: [usize; 34] = [
    EXC_ARGS_W_OFFSET,
    EXC_W_CAUSE_OFFSET,
    EXC_W_CONTEXT_OFFSET,
    EXC_W_TRACEBACK_OFFSET,
    EXC_W_OBJECT_OFFSET,
    EXC_W_START_OFFSET,
    EXC_W_END_OFFSET,
    EXC_W_REASON_OFFSET,
    EXC_W_ENCODING_OFFSET,
    EXC_W_ERRNO_OFFSET,
    EXC_W_WINERROR_OFFSET,
    EXC_W_STRERROR_OFFSET,
    EXC_W_FILENAME_OFFSET,
    EXC_W_FILENAME2_OFFSET,
    EXC_W_CODE_OFFSET,
    EXC_W_VALUE_OFFSET,
    EXC_W_NAME_OFFSET,
    EXC_W_ATTR_OBJ_OFFSET,
    EXC_W_IMPORT_PATH_OFFSET,
    EXC_W_IMPORT_NAME_FROM_OFFSET,
    EXC_W_IMPORT_MSG_OFFSET,
    EXC_W_SYNTAX_MSG_OFFSET,
    EXC_W_SYNTAX_FILENAME_OFFSET,
    EXC_W_SYNTAX_LINENO_OFFSET,
    EXC_W_SYNTAX_OFFSET_OFFSET,
    EXC_W_SYNTAX_TEXT_OFFSET,
    EXC_W_SYNTAX_END_LINENO_OFFSET,
    EXC_W_SYNTAX_END_OFFSET_OFFSET,
    EXC_W_SYNTAX_PRINT_FILE_AND_LINE_OFFSET,
    EXC_W_SYNTAX_METADATA_OFFSET,
    EXC_W_GROUP_MESSAGE_OFFSET,
    EXC_W_GROUP_EXCEPTIONS_OFFSET,
    EXC_W_GROUP_EXCEPTIONS_REPR_OFFSET,
    EXC_W_DICT_OFFSET,
];

/// Extended user layout: the extended pointers plus mapdict `storage`.
pub const W_EXCEPTION_EXTENDED_USER_GC_PTR_OFFSETS: [usize; 35] = [
    EXC_ARGS_W_OFFSET,
    EXC_W_CAUSE_OFFSET,
    EXC_W_CONTEXT_OFFSET,
    EXC_W_TRACEBACK_OFFSET,
    EXC_W_OBJECT_OFFSET,
    EXC_W_START_OFFSET,
    EXC_W_END_OFFSET,
    EXC_W_REASON_OFFSET,
    EXC_W_ENCODING_OFFSET,
    EXC_W_ERRNO_OFFSET,
    EXC_W_WINERROR_OFFSET,
    EXC_W_STRERROR_OFFSET,
    EXC_W_FILENAME_OFFSET,
    EXC_W_FILENAME2_OFFSET,
    EXC_W_CODE_OFFSET,
    EXC_W_VALUE_OFFSET,
    EXC_W_NAME_OFFSET,
    EXC_W_ATTR_OBJ_OFFSET,
    EXC_W_IMPORT_PATH_OFFSET,
    EXC_W_IMPORT_NAME_FROM_OFFSET,
    EXC_W_IMPORT_MSG_OFFSET,
    EXC_W_SYNTAX_MSG_OFFSET,
    EXC_W_SYNTAX_FILENAME_OFFSET,
    EXC_W_SYNTAX_LINENO_OFFSET,
    EXC_W_SYNTAX_OFFSET_OFFSET,
    EXC_W_SYNTAX_TEXT_OFFSET,
    EXC_W_SYNTAX_END_LINENO_OFFSET,
    EXC_W_SYNTAX_END_OFFSET_OFFSET,
    EXC_W_SYNTAX_PRINT_FILE_AND_LINE_OFFSET,
    EXC_W_SYNTAX_METADATA_OFFSET,
    EXC_W_GROUP_MESSAGE_OFFSET,
    EXC_W_GROUP_EXCEPTIONS_OFFSET,
    EXC_W_GROUP_EXCEPTIONS_REPR_OFFSET,
    EXC_W_DICT_OFFSET,
    EXC_EXTENDED_USER_STORAGE_OFFSET,
];

/// GC type id assigned to slim `W_BaseException` at JitDriver init time.
pub const W_BASE_EXCEPTION_GC_TYPE_ID: u32 = 31;

/// Closed tid of [`W_ExceptionExtended`]. Registered in `build_gc` immediately
/// after the `_io.FileIO` user layout and before [`W_BaseExceptionUser`], so
/// the extended user layout can parent on it. Constant before GC init:
/// `try_gc_alloc_collecting_rooted` with no hook is `NoRoute` and the
/// allocator falls through to `malloc_typed`.
pub const W_EXCEPTION_EXTENDED_GC_TYPE_ID: u32 = 214;

/// `W_BaseExceptionUser` (`typedef.py` `_getusercls`). Parents on
/// [`W_BASE_EXCEPTION_GC_TYPE_ID`].
pub const W_BASE_EXCEPTION_USER_GC_TYPE_ID: u32 = 215;

/// `W_ExceptionExtendedUser`. Parents on [`W_EXCEPTION_EXTENDED_GC_TYPE_ID`].
pub const W_EXCEPTION_EXTENDED_USER_GC_TYPE_ID: u32 = 216;

/// Publish the GC tid for the extra-field exception layout. `build_gc`
/// calls this with the constant tid as a registration-order check.
pub fn set_exception_extended_gc_type_id(tid: u32) {
    assert_eq!(
        tid, W_EXCEPTION_EXTENDED_GC_TYPE_ID,
        "W_ExceptionExtended must register at its closed tid"
    );
}

/// Tid of [`W_ExceptionExtended`].
pub fn exception_extended_gc_type_id() -> u32 {
    W_EXCEPTION_EXTENDED_GC_TYPE_ID
}

/// rlist.py `LIST = GcStruct("list", ("length", Signed), ("items", Ptr(ITEMARRAY)))`.
///
/// Not what `W_BaseException.args_w` stores. That field is a
/// `FixedSizeListRepr` (`GcArray`). This header stays registered so the
/// type ids published after it do not move.
#[repr(C)]
pub struct RList {
    pub length: i64,
    pub items: *mut crate::object_array::ItemsBlock,
}

pub const RLIST_SIZE: usize = std::mem::size_of::<RList>();
pub const RLIST_LENGTH_OFFSET: usize = std::mem::offset_of!(RList, length);
pub const RLIST_ITEMS_OFFSET: usize = std::mem::offset_of!(RList, items);

static RLIST_GC_TYPE_ID_CELL: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn set_rlist_gc_type_id(tid: u32) {
    debug_assert_ne!(tid, 0, "0 is the unpublished sentinel");
    RLIST_GC_TYPE_ID_CELL.store(tid, std::sync::atomic::Ordering::Release);
}

#[majit_macros::dont_look_inside]
pub fn rlist_gc_type_id() -> u32 {
    RLIST_GC_TYPE_ID_CELL.load(std::sync::atomic::Ordering::Acquire)
}

impl crate::lltype::GcType for RList {
    fn type_id() -> u32 {
        rlist_gc_type_id()
    }
    const SIZE: usize = RLIST_SIZE;
}

/// Record an old→young edge when a `W_BaseException` slot
/// (`W_BASE_EXCEPTION_GC_PTR_OFFSETS`) is overwritten after allocation.
/// No-op while the exception still lives in the nursery; once it is
/// promoted, the minor collector relies on the remembered set to find
/// the young pointers reachable only through it. The slot writers below
/// call this for the same reason `function.rs` barriers its setters.
#[inline]
fn exception_write_barrier(obj: PyObjectRef) {
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

/// Fixed payload size (`framework.py` `malloc` / `init_gc_object`) of the slim base layout.
pub const W_BASE_EXCEPTION_SIZE: usize = std::mem::size_of::<W_BaseException>();

/// Payload size of extra-field exception subclasses.
pub const W_EXCEPTION_EXTENDED_SIZE: usize = std::mem::size_of::<W_ExceptionExtended>();

impl crate::lltype::GcType for W_BaseException {
    fn type_id() -> u32 {
        W_BASE_EXCEPTION_GC_TYPE_ID
    }
    const SIZE: usize = W_BASE_EXCEPTION_SIZE;
}

impl crate::lltype::GcType for W_ExceptionExtended {
    fn type_id() -> u32 {
        W_EXCEPTION_EXTENDED_GC_TYPE_ID
    }
    const SIZE: usize = W_EXCEPTION_EXTENDED_SIZE;
}

/// Payload size of [`W_BaseExceptionUser`].
pub const W_BASE_EXCEPTION_USER_SIZE: usize = std::mem::size_of::<W_BaseExceptionUser>();

/// Payload size of [`W_ExceptionExtendedUser`].
pub const W_EXCEPTION_EXTENDED_USER_SIZE: usize = std::mem::size_of::<W_ExceptionExtendedUser>();

impl crate::lltype::GcType for W_BaseExceptionUser {
    fn type_id() -> u32 {
        W_BASE_EXCEPTION_USER_GC_TYPE_ID
    }
    const SIZE: usize = W_BASE_EXCEPTION_USER_SIZE;
}

impl crate::lltype::GcType for W_ExceptionExtendedUser {
    fn type_id() -> u32 {
        W_EXCEPTION_EXTENDED_USER_GC_TYPE_ID
    }
    const SIZE: usize = W_EXCEPTION_EXTENDED_USER_SIZE;
}

/// True when `kind` is a PyPy class that declares extra instance fields
/// (or inherits such a class): `W_OSError`, `W_ImportError`,
/// `W_SyntaxError`, `W_Unicode*Error`, `W_StopIteration`, `W_NameError`,
/// `W_AttributeError`, `W_SystemExit`, and their `_new_exception`
/// children.  `_new_exception` classes that add no fields stay on the
/// slim [`W_BaseException`] layout.
#[inline]
pub fn exc_kind_uses_extended_layout(kind: ExcKind) -> bool {
    matches!(
        kind,
        ExcKind::ImportError
            | ExcKind::ModuleNotFoundError
            | ExcKind::OSError
            | ExcKind::FileNotFoundError
            | ExcKind::StopIteration
            | ExcKind::NameError
            | ExcKind::UnboundLocalError
            | ExcKind::AttributeError
            | ExcKind::SyntaxError
            | ExcKind::SystemExit
            | ExcKind::UnicodeDecodeError
            | ExcKind::UnicodeEncodeError
            | ExcKind::UnicodeTranslateError
    )
}

/// The realbase whose interp class owns `kind`'s payload.
///
/// `_new_exception` children share that realbase: `ModuleNotFoundError` is
/// `W_ImportError`, `FileNotFoundError` is `W_OSError`, `UnboundLocalError`
/// is `W_NameError`. Fieldless classes share `W_BaseException`.
#[inline]
pub fn exc_realbase_pytype(kind: ExcKind) -> &'static PyType {
    match kind {
        ExcKind::ImportError | ExcKind::ModuleNotFoundError => &EXC_IMPORT_ERROR_TYPE,
        ExcKind::StopIteration => &EXC_STOP_ITERATION_TYPE,
        ExcKind::OSError | ExcKind::FileNotFoundError => &EXC_OS_ERROR_TYPE,
        ExcKind::NameError | ExcKind::UnboundLocalError => &EXC_NAME_ERROR_TYPE,
        ExcKind::SyntaxError => &EXC_SYNTAX_ERROR_TYPE,
        ExcKind::SystemExit => &EXC_SYSTEM_EXIT_TYPE,
        ExcKind::UnicodeDecodeError => &EXC_UNICODE_DECODE_ERROR_TYPE,
        ExcKind::UnicodeEncodeError => &EXC_UNICODE_ENCODE_ERROR_TYPE,
        ExcKind::UnicodeTranslateError => &EXC_UNICODE_TRANSLATE_ERROR_TYPE,
        ExcKind::AttributeError => &EXC_ATTRIBUTE_ERROR_TYPE,
        _ => &EXCEPTION_TYPE,
    }
}

/// Whether the canonical class of `kind` allocates the `_getusercls` layout.
///
/// Realbases (`W_BaseException`, `W_OSError`, `W_StopIteration`, ...) allocate
/// their own interp class. Every other `ExcKind` is a `_new_exception` class
/// and allocates the realbase's user layout. App-level subclasses are not
/// canonical; [`exc_instance_pytype`] decides those from `w_class`.
#[inline]
pub fn exc_kind_canonical_is_user_layout(kind: ExcKind) -> bool {
    !matches!(
        kind,
        ExcKind::BaseException
            | ExcKind::ImportError
            | ExcKind::StopIteration
            | ExcKind::OSError
            | ExcKind::NameError
            | ExcKind::SyntaxError
            | ExcKind::SystemExit
            | ExcKind::UnicodeDecodeError
            | ExcKind::UnicodeEncodeError
            | ExcKind::UnicodeTranslateError
            | ExcKind::AttributeError
    )
}

/// `ExcKind` of the realbase class `allocate_instance` compares against.
///
/// `_new_exception` children share that class: `FileNotFoundError` and
/// `ModuleNotFoundError` are not their own realbase, and every fieldless
/// class shares `W_BaseException`.
#[inline]
fn exc_realbase_kind(kind: ExcKind) -> ExcKind {
    match kind {
        ExcKind::ImportError | ExcKind::ModuleNotFoundError => ExcKind::ImportError,
        ExcKind::StopIteration => ExcKind::StopIteration,
        ExcKind::OSError | ExcKind::FileNotFoundError => ExcKind::OSError,
        ExcKind::NameError | ExcKind::UnboundLocalError => ExcKind::NameError,
        ExcKind::SyntaxError => ExcKind::SyntaxError,
        ExcKind::SystemExit => ExcKind::SystemExit,
        ExcKind::UnicodeDecodeError => ExcKind::UnicodeDecodeError,
        ExcKind::UnicodeEncodeError => ExcKind::UnicodeEncodeError,
        ExcKind::UnicodeTranslateError => ExcKind::UnicodeTranslateError,
        ExcKind::AttributeError => ExcKind::AttributeError,
        _ => ExcKind::BaseException,
    }
}

/// `allocate_instance`: exact realbase class → base layout, anything else →
/// `_getusercls`.
///
/// The realbase is the class `register_exc_class_for_kind` stored for
/// [`exc_realbase_kind`]. Extended realbase vtables never receive
/// `set_instantiate` (only `EXCEPTION_TYPE` does, and that slot is the
/// pre-init `"exception"` stub), so the instantiate word is not the Python
/// class. Before registration, the comparison falls back to
/// `get_instantiate` of the realbase vtable. A null fallback is the
/// exact-base case: an allocation before either slot is filled does not
/// stamp a mapdict typeptr.
#[inline]
pub fn exc_instance_is_user_layout(kind: ExcKind, w_class: PyObjectRef) -> bool {
    let registered = lookup_exc_class_for_kind(exc_realbase_kind(kind));
    let realbase = if !registered.is_null() {
        registered
    } else {
        get_instantiate(exc_realbase_pytype(kind))
    };
    !realbase.is_null() && !std::ptr::eq(w_class, realbase)
}

/// Instance `ob_type` for `(kind, w_class)`.
///
/// Exact realbase → the vtable that realbase uses (`EXCEPTION_TYPE` for the
/// slim base, `exc_realbase_pytype` for an extended realbase). Otherwise the
/// matching user PyType.
#[inline]
pub fn exc_instance_pytype(kind: ExcKind, w_class: PyObjectRef) -> &'static PyType {
    if exc_instance_is_user_layout(kind, w_class) {
        if exc_kind_uses_extended_layout(kind) {
            &EXCEPTION_EXTENDED_USER_TYPE
        } else {
            &BASE_EXCEPTION_USER_TYPE
        }
    } else if exc_kind_uses_extended_layout(kind) {
        exc_realbase_pytype(kind)
    } else {
        &EXCEPTION_TYPE
    }
}

/// Whether `ob_type` is a `_getusercls` exception vtable.
#[inline]
pub fn exc_typeptr_is_user_layout(ob_type: *const PyType) -> bool {
    std::ptr::eq(ob_type, &BASE_EXCEPTION_USER_TYPE)
        || std::ptr::eq(ob_type, &EXCEPTION_EXTENDED_USER_TYPE)
}

/// Whether `obj` was allocated by `_getusercls` (`exc_instance_pytype`).
///
/// # Safety
/// `obj` must be a live object. A null object answers false.
#[inline]
pub unsafe fn exc_obj_is_user_layout(obj: PyObjectRef) -> bool {
    !obj.is_null() && exc_typeptr_is_user_layout(unsafe { (*obj).ob_type })
}

/// GC pointer slots of an unmanaged exception, selected by `ob_type`.
///
/// User layouts include mapdict `storage`. Exact group instances keep a slim
/// `kind` on the extended struct and stay on the kind table.
///
/// # Safety
/// `obj` must be a live exception (`is_exception`).
pub unsafe fn exception_unmanaged_gc_ptr_offsets(obj: PyObjectRef) -> &'static [usize] {
    let tp = unsafe { (*obj).ob_type };
    if std::ptr::eq(tp, &BASE_EXCEPTION_USER_TYPE) {
        &W_BASE_EXCEPTION_USER_GC_PTR_OFFSETS
    } else if std::ptr::eq(tp, &EXCEPTION_EXTENDED_USER_TYPE) {
        &W_EXCEPTION_EXTENDED_USER_GC_PTR_OFFSETS
    } else {
        let kind = unsafe { w_exception_get_kind(obj) };
        if exc_kind_uses_extended_layout(kind) {
            &W_EXCEPTION_EXTENDED_GC_PTR_OFFSETS
        } else {
            &W_BASE_EXCEPTION_GC_PTR_OFFSETS
        }
    }
}

/// Allocate a new exception object on the heap.
///
/// `ob_header.w_class` is populated from the per-`ExcKind` class
/// registry (`register_exc_class_for_kind`) when the interpreter has
/// finished installing builtin exception types; otherwise it falls
/// back to the generic `EXCEPTION_TYPE` instantiate slot. Callers
/// that rely on `space.type(w_exc)` returning the specific class
/// (e.g. `cmp_exc_match` at `pyopcode.py`) get the registered
/// class once init has run; pre-init callers see the generic
/// placeholder, matching the legacy "internal `w_exception_new`"
/// path.
pub fn w_exception_new(kind: ExcKind, message: &str) -> PyObjectRef {
    let exc = w_exception_new_empty(kind);
    // `oefmt(space.w_ValueError, "...")` parity — an internal raise with
    // a message stores it as the single constructor arg
    // (`args_w = [space.newtext(msg)]`); `descr_str` then derives the
    // string lazily.  Empty message → no args (the `args_w` stays
    // `PY_NULL` so `args` reads as `()`), matching the prebuilt
    // singletons (`MemoryError`, `StopIteration`).
    if message.is_empty() {
        return exc;
    }
    // Root the fresh exception across the args array. `w_exception_args_new`
    // can collect, so read `exc` back out of the slot afterwards.
    let _roots = crate::gc_roots::push_roots();
    let exc_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(exc);
    let arg_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(crate::unicodeobject::w_str_new_managed(message));
    unsafe {
        let arg = crate::gc_roots::shadow_stack_get(arg_slot);
        let args_list = w_exception_args_new(vec![arg]);
        w_exception_set_args(crate::gc_roots::shadow_stack_get(exc_slot), args_list);
        // `W_SystemExit.descr_init`: one argument is `w_code`.
        if kind == ExcKind::SystemExit {
            w_exception_set_code(
                crate::gc_roots::shadow_stack_get(exc_slot),
                crate::gc_roots::shadow_stack_get(arg_slot),
            );
        }
    }
    crate::gc_roots::shadow_stack_get(exc_slot)
}

/// Like `w_exception_new` but stores an arbitrary WTF-8 message,
/// preserving lone surrogates that a `&str` message cannot carry.
pub fn w_exception_new_wtf8(kind: ExcKind, message: &Wtf8) -> PyObjectRef {
    w_exception_new_wtf8_for_class(kind, message, PY_NULL)
}

/// [`w_exception_new_wtf8`] for a caller that already resolved `cls`.
///
/// `W_OSError.descr_new` retargets `w_subtype` through `ERRNO_MAP` before
/// `allocate_instance`, so the user-vs-base choice sees that class.
pub fn w_exception_new_wtf8_for_class(
    kind: ExcKind,
    message: &Wtf8,
    cls: PyObjectRef,
) -> PyObjectRef {
    let exc = w_exception_new_empty_for_class(kind, cls);
    if message.is_empty() {
        return exc;
    }
    // See `w_exception_new`: pin `exc` across the allocating arg build and read
    // it back out of the slot rather than carrying the raw local forward.
    let _roots = crate::gc_roots::push_roots();
    let exc_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(exc);
    let arg_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(crate::unicodeobject::w_str_from_wtf8_managed(
        message.to_wtf8_buf(),
    ));
    unsafe {
        let arg = crate::gc_roots::shadow_stack_get(arg_slot);
        let args_list = w_exception_args_new(vec![arg]);
        w_exception_set_args(crate::gc_roots::shadow_stack_get(exc_slot), args_list);
        // `W_SystemExit.descr_init`: one argument is `w_code`.
        if kind == ExcKind::SystemExit {
            w_exception_set_code(
                crate::gc_roots::shadow_stack_get(exc_slot),
                crate::gc_roots::shadow_stack_get(arg_slot),
            );
        }
    }
    crate::gc_roots::shadow_stack_get(exc_slot)
}

/// Allocate a `W_BaseException` of `kind` with no constructor args
/// (`args_w = PY_NULL`).  The public Python `__new__` path
/// (`exc_constructor`) and the message helpers above attach `args_w`
/// afterwards via `w_exception_set_args`.
pub fn w_exception_new_empty(kind: ExcKind) -> PyObjectRef {
    w_exception_new_empty_impl(kind, false)
}

/// Allocate the extra-field layout even when `kind` is a slim class.
///
/// Prefer [`w_exception_new_empty_extended_for_class`] when the Python class
/// is already known: a group instance is [`W_ExceptionExtended`] or
/// [`W_ExceptionExtendedUser`] while its `kind` stays `Exception` /
/// `BaseException`. This entry uses the canonical class of `kind`.
pub fn w_exception_new_empty_extended(kind: ExcKind) -> PyObjectRef {
    w_exception_new_empty_extended_for_class(kind, PY_NULL, false)
}

/// Immortal variant for the prebuilt singletons (`memory_error_singleton` /
/// `standard_exc_instance`): they are cached in `OnceLock<usize>` (GC-invisible)
/// and baked into JIT constant pools as immediate pointers, so they must never
/// be swept — keep them `malloc_typed`-immortal (stable, never reclaimed).
pub fn w_exception_new_empty_immortal(kind: ExcKind) -> PyObjectRef {
    w_exception_new_empty_impl(kind, true)
}

/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`), the
/// `w_dict_new` / `w_dict_view_iterator_new_direction` twin: the body picks
/// the exception's type word at runtime (`exc_instance_pytype`), and a
/// cluster whose type word is not a constant address does not lower — the word
/// rides on the allocation or not at all, since a `setfield_gc` whose descr
/// `is_typeptr()` is removed downstream. The primary allocation is hand-rolled
/// through `try_gc_alloc_collecting_rooted` besides, which is not one of the
/// `lltype::malloc*` spellings `fuse_boxing_alloc` recognises. Residualise the
/// whole constructor — the JIT models it by signature as a plain
/// `PyObjectRef` GCREF and emits a residual call.
/// `framework.py malloc_fixedsize` for a non-immortal exception. Nursery,
/// same as `ll_fixed_newlist`: a full nursery takes `collect_and_reserve`
/// (a minor collection) instead of spilling the instance straight into the
/// old generation. A born-old instance plus a nursery `args_w` array is a
/// permanent old→young edge: if the setter misses the remembered set, the
/// next minor recycles the array and a later read walks a dead block.
///
/// `value` is built first, including the typeptr
/// `get_unique_interplevel_subclass` selected (`*_USER_TYPE` for a user
/// layout). At birth the only live GC child is `w_class`
/// (`w_exception_base_defaults` / the extended-layout builder leave every
/// other pointer `PY_NULL`). That word is the rooted slot
/// `try_gc_alloc_collecting_rooted` forwards across the minor; the shadow
/// stack does not stand in for it. The collector rewrites the slot, so the
/// word is stored back into `value` before `ptr::write`.
///
/// The nursery shell is uninitialized until that write. `pin_root` publishes
/// it (`normalize_published_slot`), and a foreign collector can trace it
/// from then on, so the header is written before the pin. The barrier is a
/// `gc_op` and can move the shell; the returned word is the one the slot
/// carries afterwards.
fn alloc_exception_nursery<T: crate::lltype::GcType>(mut value: T) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let mut rooted_class = exception_header_w_class(&value);
    let mut needs_write_barrier = true;
    let tid = T::type_id();
    let raw = if tid != 0 {
        crate::gc_hook::GcAllocOutcome::from_hook(unsafe {
            crate::gc_hook::try_gc_alloc_collecting_rooted(
                tid,
                T::SIZE,
                (&mut rooted_class as *mut PyObjectRef).cast(),
                &mut needs_write_barrier,
            )
        })
        .allocated_or_abort(T::SIZE)
        .unwrap_or(std::ptr::null_mut())
    } else {
        std::ptr::null_mut()
    };
    set_exception_header_w_class(&mut value, rooted_class);
    if !raw.is_null() {
        unsafe {
            std::ptr::write(raw as *mut T, value);
        }
        let slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(raw as PyObjectRef);
        let raw = crate::gc_roots::shadow_stack_get(slot) as *mut u8;
        // A nursery header needs no creation barrier. The collecting
        // allocator can still spill old (pinned nursery gap); only that
        // placement remembers the `w_class` edge.
        if needs_write_barrier {
            crate::gc_hook::try_gc_write_barrier(raw as crate::gc_hook::GCREF);
        }
        return crate::gc_roots::shadow_stack_get(slot);
    }
    crate::lltype::malloc_typed(value) as PyObjectRef
}

/// `ob_header.w_class` of a `W_BaseException` or a `W_ExceptionExtended`.
/// Both layouts start with that header (`base` is the extended struct's
/// first field), so the offset is the `PyObject.w_class` offset.
fn exception_header_w_class<T>(value: &T) -> PyObjectRef {
    let base = std::ptr::from_ref(value).cast::<u8>();
    unsafe {
        *base
            .add(std::mem::offset_of!(PyObject, w_class))
            .cast::<PyObjectRef>()
    }
}

fn set_exception_header_w_class<T>(value: &mut T, w_class: PyObjectRef) {
    let base = std::ptr::from_mut(value).cast::<u8>();
    unsafe {
        *base
            .add(std::mem::offset_of!(PyObject, w_class))
            .cast::<PyObjectRef>() = w_class;
    }
}

fn canonical_exc_class(kind: ExcKind) -> PyObjectRef {
    let w_class = lookup_exc_class_for_kind(kind);
    if w_class != PY_NULL {
        w_class
    } else {
        get_instantiate(&EXCEPTION_TYPE)
    }
}

fn w_exception_base_defaults(
    kind: ExcKind,
    ob_type: *const PyType,
    w_class: PyObjectRef,
) -> W_BaseException {
    W_BaseException {
        ob_header: PyObject { ob_type, w_class },
        kind,
        args_w: PY_NULL,
        w_cause: PY_NULL,
        w_context: PY_NULL,
        w_traceback: PY_NULL,
        suppress_context: false,
        w_dict: PY_NULL,
    }
}

fn extended_payload(base: W_BaseException) -> W_ExceptionExtended {
    W_ExceptionExtended {
        base,
        w_object: PY_NULL,
        w_start: PY_NULL,
        w_end: PY_NULL,
        w_reason: PY_NULL,
        w_encoding: PY_NULL,
        w_errno: PY_NULL,
        w_winerror: PY_NULL,
        w_strerror: PY_NULL,
        w_filename: PY_NULL,
        w_filename2: PY_NULL,
        written: -1,
        w_code: PY_NULL,
        w_value: PY_NULL,
        w_exc_name: PY_NULL,
        w_attr_obj: PY_NULL,
        w_import_path: PY_NULL,
        w_import_name_from: PY_NULL,
        w_import_msg: PY_NULL,
        w_syntax_msg: PY_NULL,
        w_syntax_filename: PY_NULL,
        w_syntax_lineno: PY_NULL,
        w_syntax_offset: PY_NULL,
        w_syntax_text: PY_NULL,
        w_syntax_end_lineno: PY_NULL,
        w_syntax_end_offset: PY_NULL,
        w_syntax_print_file_and_line: PY_NULL,
        w_syntax_metadata: PY_NULL,
        w_group_message: PY_NULL,
        w_group_exceptions: PY_NULL,
        w_group_exceptions_repr: PY_NULL,
    }
}

fn alloc_typed<T: crate::lltype::GcType>(value: T, immortal: bool) -> PyObjectRef {
    if immortal {
        crate::lltype::malloc_typed(value) as PyObjectRef
    } else {
        alloc_exception_nursery(value)
    }
}

/// `allocate_instance` enqueues a user finalizer after `user_setup`.
/// Pin first: the hook can collect.
fn register_user_finalizer(obj: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    crate::gc_hook::maybe_register_finalizer(crate::gc_roots::shadow_stack_get(slot));
    crate::gc_roots::shadow_stack_get(slot)
}

/// `objspace.py allocate_instance` for an exception class.
///
/// `force_extended` selects [`W_ExceptionExtended`] even for a slim `kind`
/// (exception groups). `force_user` overrides [`exc_instance_is_user_layout`]:
/// `Some(false)` on a slim kind stamps `exc_kind_to_pytype` (the vtable
/// `W_BaseExceptionGroup.descr_new` uses for an exact group class);
/// `Some(true)` stamps the extended user PyType. `None` is
/// [`exc_instance_pytype`].
#[majit_macros::dont_look_inside]
fn allocate_exception(
    kind: ExcKind,
    w_class: PyObjectRef,
    immortal: bool,
    force_extended: bool,
    force_user: Option<bool>,
) -> PyObjectRef {
    let extended = force_extended || exc_kind_uses_extended_layout(kind);
    let user = force_user.unwrap_or_else(|| exc_instance_is_user_layout(kind, w_class));
    let ob_type: *const PyType = match force_user {
        Some(false) if !exc_kind_uses_extended_layout(kind) => exc_kind_to_pytype(kind),
        _ if extended && user => &EXCEPTION_EXTENDED_USER_TYPE,
        _ if user => &BASE_EXCEPTION_USER_TYPE,
        _ if extended && exc_kind_uses_extended_layout(kind) => exc_realbase_pytype(kind),
        _ if extended => exc_kind_to_pytype(kind),
        _ => &EXCEPTION_TYPE,
    };
    let base = w_exception_base_defaults(kind, ob_type, w_class);
    let obj = if !extended && !user {
        alloc_typed(base, immortal)
    } else if !extended {
        alloc_typed(
            W_BaseExceptionUser {
                base,
                map: 0,
                storage: std::ptr::null_mut(),
            },
            immortal,
        )
    } else if !user {
        alloc_typed(extended_payload(base), immortal)
    } else {
        alloc_typed(
            W_ExceptionExtendedUser {
                base: extended_payload(base),
                map: 0,
                storage: std::ptr::null_mut(),
            },
            immortal,
        )
    };
    if user && !immortal {
        register_user_finalizer(obj)
    } else {
        obj
    }
}

#[majit_macros::dont_look_inside]
fn w_exception_new_empty_impl(kind: ExcKind, immortal: bool) -> PyObjectRef {
    allocate_exception(kind, canonical_exc_class(kind), immortal, false, None)
}

/// Per-`ExcKind` class-pointer registry. Populated by
/// `pyre-interpreter::builtins::register_exc_class` during
/// `install_default_builtins`; consumed by `w_exception_new` so each
/// builtin-raised exception's `ob_header.w_class` points at the
/// specific class object (rather than the generic `EXCEPTION_TYPE`).
/// PyPy's equivalent is the `space.w_TypeError` / `space.w_ValueError`
/// / ... attributes on `ObjSpace`.
///
/// The builtin `W_TypeObject` identities and this registry are process-global.
/// A class installed by one execution-context thread must therefore be the
/// same class used to stamp and match exceptions on every other thread.
/// Registration is first-writer-wins so rebuilding a builtins dictionary
/// cannot replace a canonical class. The pointer is stored as `usize` because
/// `PyObjectRef` itself is neither `Send` nor `Sync`; builtin type objects are
/// immortal and process-global.
/// One slot per `ExcKind` variant.  Indexed by `kind as u8 as usize`,
/// so `EXC_KIND_COUNT - 1` is the largest valid index.  Public so
/// downstream crates (e.g. pyre-jit's GC init) can size per-kind
/// arrays against the same authoritative bound.  Anchored on the
/// highest-numbered variant so adding new ExcKinds at the end of the
/// enum extends the bound automatically.
pub const EXC_KIND_COUNT: usize = (ExcKind::EOFError as u8 as usize) + 1;

static EXC_CLASS_BY_KIND: [std::sync::atomic::AtomicUsize; EXC_KIND_COUNT] =
    [const { std::sync::atomic::AtomicUsize::new(0) }; EXC_KIND_COUNT];

/// Register `cls` for `kind` if the process-global slot is empty and return
/// the canonical class selected by the first writer.
pub fn register_exc_class_for_kind(kind: ExcKind, cls: PyObjectRef) -> PyObjectRef {
    let slot = &EXC_CLASS_BY_KIND[kind as u8 as usize];
    match slot.compare_exchange(
        0,
        cls as usize,
        std::sync::atomic::Ordering::AcqRel,
        std::sync::atomic::Ordering::Acquire,
    ) {
        Ok(_) => cls,
        Err(canonical) => canonical as PyObjectRef,
    }
}

/// Reads the process-global `EXC_CLASS_BY_KIND`, a root the tracer cannot
/// type, so the JIT residualises the read.  It is elidable: a slot is written
/// once, by `register_exc_class_for_kind`'s first writer, and an instance of
/// a kind exists only after its class was registered (the internal raise paths
/// build instances of the init-time kinds only), so a read made for a live
/// instance never observes the slot's empty state.  The residual call resolves
/// its address by qualified path in `jit_trace_fnaddrs`.
#[majit_macros::elidable]
pub fn lookup_exc_class_for_kind(kind: ExcKind) -> PyObjectRef {
    EXC_CLASS_BY_KIND[kind as u8 as usize].load(std::sync::atomic::Ordering::Acquire) as PyObjectRef
}

/// True when `cls` is one of the canonical process-global builtin exception
/// classes registered via `register_exc_class_for_kind` — i.e. its
/// constructor is the Rust `descr_init` (no Python `__init__`).
pub fn is_canonical_exc_class(cls: PyObjectRef) -> bool {
    !cls.is_null()
        && EXC_CLASS_BY_KIND
            .iter()
            .any(|slot| slot.load(std::sync::atomic::Ordering::Acquire) == cls as usize)
}

/// Canonical `ExcKind` for `cls` when `cls` is a registered builtin
/// exception class.  Heap subclasses are not registered and answer `None`.
///
/// The atomic slot read stays in [`lookup_exc_class_for_kind`] (`@jit.elidable`
/// on that helper, same as the process-global class table). This body is a
/// signed trip over the contiguous discriminants so the prepass lifts the
/// graph (`rpython/annotator/builtin.py` `builtin_range`).
pub fn kind_of_canonical_exc_class(cls: PyObjectRef) -> Option<ExcKind> {
    if cls.is_null() {
        return None;
    }
    for raw in 0..=ExcKind::MAX_DISCRIMINANT {
        let kind = unsafe { std::mem::transmute::<u8, ExcKind>(raw) };
        if lookup_exc_class_for_kind(kind) == cls {
            return Some(kind);
        }
    }
    None
}

/// The canonical kind whose instance layout `cls` extends.
///
/// `typeobject.py find_best_base` picks the most derived base layout, so
/// `class VS(ValueError, StopIteration)` carries StopIteration's fields even
/// though ValueError supplies `__new__`.  Allocating by `kind` alone hands
/// `StopIteration.__init__` a slim object to write `w_value` past.
pub fn exception_layout_kind_for_class(kind: ExcKind, cls: PyObjectRef) -> ExcKind {
    if cls.is_null() {
        return kind;
    }
    let own = lookup_exc_class_for_kind(kind);
    if own.is_null() {
        return kind;
    }
    if unsafe { crate::typeobject::w_type_get_layout_ptr(cls) }
        == unsafe { crate::typeobject::w_type_get_layout_ptr(own) }
    {
        return kind;
    }
    let mut cur = cls;
    while !cur.is_null() {
        if let Some(found) = kind_of_canonical_exc_class(cur) {
            return found;
        }
        cur = unsafe { crate::typeobject::w_type_get_best_base(cur) };
    }
    kind
}

/// Allocate for `cls(...)`, where `cls` may be a heap subclass whose best
/// base owns a wider layout than the class that supplied `__new__`.
///
/// `cls` is the Python class `allocate_instance` compares against the
/// realbase. Null `cls` falls back to the canonical class of the layout kind.
#[majit_macros::dont_look_inside]
pub fn w_exception_new_empty_for_class(kind: ExcKind, cls: PyObjectRef) -> PyObjectRef {
    let layout_kind = exception_layout_kind_for_class(kind, cls);
    let w_class = if cls.is_null() {
        canonical_exc_class(layout_kind)
    } else {
        cls
    };
    allocate_exception(layout_kind, w_class, false, false, None)
}

/// Group allocation: `kind` stays slim (`Exception` / `BaseException`) while
/// the bytes are [`W_ExceptionExtended`] or [`W_ExceptionExtendedUser`].
///
/// `user_layout` is `w_subtype is not W_BaseExceptionGroup` after
/// `W_BaseExceptionGroup.descr_new` promotes an all-Exception payload to
/// `ExceptionGroup`. Exact `BaseExceptionGroup` passes `false` and keeps
/// `exc_kind_to_pytype(kind)` as its vtable.
#[majit_macros::dont_look_inside]
pub fn w_exception_new_empty_extended_for_class(
    kind: ExcKind,
    cls: PyObjectRef,
    user_layout: bool,
) -> PyObjectRef {
    let w_class = if cls.is_null() {
        canonical_exc_class(kind)
    } else {
        cls
    };
    allocate_exception(kind, w_class, false, true, Some(user_layout))
}

/// `interp_exceptions.py W_BaseException.descr_getargs` parity —
///
/// ```python
/// def descr_getargs(self, space):
///     return space.newtuple(self.args_w)
/// ```
///
/// Returns a new tuple for the internal list, or an empty tuple when
/// the exception was constructed without `descr_init` (`args_w` stays
/// `PY_NULL`). Each call allocates a new tuple header, so
/// `e.args is e.args` is false.
///
/// `newtuple` / `wraptuple`: length 2 is `makespecialisedtuple`. Every
/// other length is `W_TupleObject(args_w)`, which stores that array.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_args(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let stored = (*(obj as *const W_BaseException)).args_w;
        if stored.is_null() {
            return crate::tupleobject::w_tuple_new(Vec::new());
        }
        // `wraptuple`: length 2 copies through `wraptuple2` and leaves
        // `args_w` on the exception. Every other length adopts the array.
        crate::tupleobject::wraptuple(stored as *mut crate::object_array::ItemsBlock)
    }
}

/// Build the `args_w` storage for an exception.
///
/// `FixedSizeListRepr`: one `GcArray` of `W_Root` (`ll_fixed_newlist`).
pub fn w_exception_args_new(items: Vec<PyObjectRef>) -> PyObjectRef {
    rlist_new(items)
}

/// `ll_fixed_newlist` — `malloc` the item array, including length 0.
///
/// `Arguments.__init__` fixes `arguments_w` with `make_sure_not_resized`,
/// `descr_new` stores that list on `exc.args_w`, and `descr_setargs`
/// stores `space.fixedview`. The resizable LIST header is a different
/// repr and is not allocated here.
#[majit_macros::dont_look_inside]
pub fn rlist_new(items: Vec<PyObjectRef>) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let items_base = crate::gc_roots::shadow_stack_len();
    for &item in &items {
        let _ = crate::gc_roots::pin_root(item);
    }
    let n = items.len();
    // Exact-size `malloc(ITEMARRAY, length)`, including 0.
    // `alloc_list_items_block_gc` would clamp empty to `cap.max(1)`.
    // Fill from the pinned slots, not a Vec snapshot — the block malloc
    // can collect.
    unsafe { crate::object_array::alloc_tuple_items_block_gc(items_base, n) as PyObjectRef }
}

/// `ll_fixed_length` — `len` of the item array.
#[inline]
pub unsafe fn rlist_len(list: PyObjectRef) -> usize {
    if list.is_null() {
        return 0;
    }
    unsafe { (*(list as *const crate::object_array::ItemsBlock)).capacity }
}

/// `ll_fixed_getitem_fast` for a known-in-bounds index.
#[inline]
pub unsafe fn rlist_getitem(list: PyObjectRef, index: usize) -> PyObjectRef {
    debug_assert!(index < unsafe { rlist_len(list) });
    let base = unsafe {
        crate::object_array::items_block_items_base(list as *mut crate::object_array::ItemsBlock)
    };
    unsafe { *base.add(index) }
}

/// Raw `args_w` storage for JIT field mirrors.  Unlike
/// [`w_exception_get_args`], this does not allocate the public tuple view.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_args_storage(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).args_w }
}

/// `interp_exceptions.py W_BaseException.descr_init` /
/// `:156-157 descr_setargs` parity —
///
/// ```python
/// def descr_init(self, space, args_w):
///     self.args_w = args_w
///
/// def descr_setargs(self, space, w_newargs):
///     self.args_w = space.fixedview(w_newargs)
/// ```
///
/// Stores the fixed-size array `space.fixedview` produced.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_args(obj: PyObjectRef, args_list: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).args_w = args_list;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py descr_getcause` parity —
///
/// ```python
/// def descr_getcause(self, space):
///     return self.w_cause
/// ```
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_cause(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_cause }
}

/// `interp_exceptions.py descr_setcause` parity — writes the
/// `w_cause` slot.  Type validation (None or BaseException subclass
/// instance) is enforced at the call site (`baseobjspace::setattr_str`).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_cause(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_cause = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py descr_getcontext` parity —
///
/// ```python
/// def descr_getcontext(self, space):
///     return self.w_context
/// ```
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_context(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_context }
}

/// `interp_exceptions.py descr_setcontext` parity — writes
/// the `w_context` slot.  Type validation lives in
/// `baseobjspace::setattr_str`.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_context(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_context = value;
        exception_write_barrier(obj);
    }
}

/// The raw `self.w_traceback` slot read.  `descr_gettraceback`
/// (`interp_exceptions.py`) and `OperationError.get_traceback`
/// (`error.py`) are this read plus `tb.frame.mark_as_escaped()`;
/// that mark lives in `pytraceback::mark_traceback_escaped`, since the
/// frame type is not visible from this crate.  Callers mirroring either
/// getter pair the two; callers mirroring a direct `_application_traceback`
/// read (printing, chain trimming) use this alone.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_traceback(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_traceback }
}

/// `interp_exceptions.py descr_settraceback` parity — writes
/// the `w_traceback` slot.  Type validation lives in
/// `baseobjspace::setattr_str`.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_traceback(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_traceback = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py getdict` parity —
///
/// ```python
/// def getdict(self, space):
///     if self.w_dict is None:
///         self.w_dict = space.newdict(instance=True)
///     return self.w_dict
/// ```
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_getdict(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let exc = obj as *mut W_BaseException;
        if !(*exc).w_dict.is_null() {
            return (*exc).w_dict;
        }
        // The exception is nursery-allocated.  The instance dict is a
        // collecting allocation, so the receiver is pinned and the store
        // goes through the forwarded address.
        let _roots = crate::gc_roots::push_roots();
        let obj_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(obj);
        let w_dict = crate::dictmultiobject::w_dict_new_instance();
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let exc = obj as *mut W_BaseException;
        (*exc).w_dict = w_dict;
        exception_write_barrier(obj);
        (*exc).w_dict
    }
}

/// `descr_reduce` reads `self.w_dict` WITHOUT allocating it — returns the
/// raw slot (`PY_NULL` when unset), so a `__reduce__` over an attribute-less
/// exception does not leave behind a fresh empty instance dict.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_peek_dict(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_dict }
}

/// `interp_exceptions.py setdict` parity — writes the `w_dict`
/// slot.  The non-dict `TypeError` check lives in the caller
/// (`baseobjspace::setdict`).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_setdict(obj: PyObjectRef, w_dict: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_dict = w_dict;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py descr_getsuppresscontext` parity —
///
/// ```python
/// def descr_getsuppresscontext(self, space):
///     return space.newbool(self.suppress_context)
/// ```
///
/// Returns the raw bool; the caller wraps with `w_bool_from`.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_suppress_context(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_BaseException)).suppress_context }
}

/// `interp_exceptions.py descr_setsuppresscontext` parity —
/// writes the `suppress_context` slot after the caller has resolved
/// `space.bool_w(w_value)` into a Rust bool.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_suppress_context(obj: PyObjectRef, value: bool) {
    unsafe {
        (*(obj as *mut W_BaseException)).suppress_context = value;
    }
}

// ─── Unicode*Error per-class field accessors ────────────────────────
//
// `interp_exceptions.py W_UnicodeTranslateError.typedef`
// (and `:1080-1084 W_UnicodeDecodeError.typedef` /
// `:1200-1204 W_UnicodeEncodeError.typedef`) wire each field via
// `readwrite_attrproperty_w('w_object', ...)` etc.  Pyre's
// `baseobjspace::getattr_str` and `setattr` arms dispatch on the
// attribute name + ExcKind and route here.
//
// All five accessors return `space.w_None` (resolved by the caller)
// when the slot is `PY_NULL`, matching PyPy's class-default
// `w_object = None` etc. — `descr_str` checks `if self.object is
// None:` and short-circuits to `""`.

/// `interp_exceptions.py readwrite_attrproperty_w('w_object', ...)`
/// — `e.object` reader.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_object(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_object }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_object', ...)`
/// — `e.object = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_object(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_object = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_start', ...)`
/// — `e.start` reader.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_start(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_start }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_start', ...)`
/// — `e.start = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_start(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_start = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_end', ...)`
/// — `e.end` reader.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_end(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_end }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_end', ...)`
/// — `e.end = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_end(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_end = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_reason', ...)`
/// — `e.reason` reader.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_reason(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_reason }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_reason', ...)`
/// — `e.reason = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_reason(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_reason = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py` `readwrite_attrproperty_w('w_encoding',
/// W_UnicodeDecodeError)` / `readwrite_attrproperty_w('w_encoding',
/// W_UnicodeEncodeError)` — `e.encoding` reader (Decode / Encode only;
/// Translate has no encoding field but the slot is still backed by
/// `PY_NULL`).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_encoding(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_encoding }
}

/// `interp_exceptions.py` `readwrite_attrproperty_w('w_encoding',
/// W_UnicodeDecodeError)` / `readwrite_attrproperty_w('w_encoding',
/// W_UnicodeEncodeError)` — `e.encoding = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_encoding(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_encoding = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_errno', ...)`
/// — `e.errno` reader.  `PY_NULL` means the slot was never written
/// (the `errno` getattr arm then derives the value from `args_w`).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_errno(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_errno }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_errno', ...)`
/// — `e.errno = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_errno(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_errno = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_winerror', ...)`
/// — `e.winerror` reader.  `PY_NULL` means no Windows error code was
/// supplied, which is every instance off Windows and the ones built from
/// an errno on it.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_winerror(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_winerror }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_winerror', ...)`
/// — `e.winerror = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_winerror(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_winerror = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_strerror', ...)`
/// — `e.strerror` reader.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_strerror(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_strerror }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_strerror', ...)`
/// — `e.strerror = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_strerror(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_strerror = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_filename', ...)`
/// — `e.filename` reader.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_filename(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_filename }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_filename', ...)`
/// — `e.filename = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_filename(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_filename = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_filename2', ...)`
/// — `e.filename2` reader.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_filename2(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_filename2 }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_filename2', ...)`
/// — `e.filename2 = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_filename2(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_filename2 = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py W_OSError.descr_{get,set,del}_written`.
/// `-1` is the unset sentinel; the descriptor converts values before storing.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_written(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_ExceptionExtended)).written }
}

/// Store the `W_OSError.written` integer slot.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_written(obj: PyObjectRef, value: i64) {
    unsafe { (*(obj as *mut W_ExceptionExtended)).written = value };
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_code', ...)`
/// — `e.code` reader.  `PY_NULL` is the class default `None`.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_code(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_code }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_code', ...)`
/// — `e.code = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_code(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_code = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_value', ...)` —
/// `StopIteration.value` reader.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_value(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_value }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_value', ...)` —
/// `StopIteration.value = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_value(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_value = value;
        exception_write_barrier(obj);
    }
}

/// Shared `e.name` reader for ImportError / NameError / AttributeError
/// (`readwrite_attrproperty_w('w_name', ...)`).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_name(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_exc_name }
}

/// Shared `e.name = ...` writer for ImportError / NameError /
/// AttributeError.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_name(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_exc_name = value;
        exception_write_barrier(obj);
    }
}

/// `e.obj` reader (W_AttributeError).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_attr_obj(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_attr_obj }
}

/// `e.obj = ...` writer (W_AttributeError).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_attr_obj(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_attr_obj = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_path', ...)`
/// — `e.path` reader (W_ImportError).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_import_path(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_import_path }
}

/// `e.path = ...` writer (W_ImportError).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_import_path(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_import_path = value;
        exception_write_barrier(obj);
    }
}

/// `e.name_from` reader (W_ImportError).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_import_name_from(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_import_name_from }
}

/// `e.name_from = ...` writer (W_ImportError).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_import_name_from(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_import_name_from = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py readwrite_attrproperty_w('w_msg', ...)`
/// — `e.msg` reader (W_ImportError).  `PY_NULL` means the slot was never
/// written (the `msg` getattr arm then derives the value from `args_w`).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_import_msg(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_import_msg }
}

/// `e.msg = ...` writer (W_ImportError).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_import_msg(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_import_msg = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py W_SyntaxError.w_filename` reader.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_get_syntax_filename(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_syntax_filename }
}

/// `interp_exceptions.py W_SyntaxError.w_filename` writer.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_set_syntax_filename(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_syntax_filename = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py W_SyntaxError.w_lineno` reader.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_get_syntax_lineno(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_syntax_lineno }
}

/// `interp_exceptions.py W_SyntaxError.w_lineno` writer.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_set_syntax_lineno(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_syntax_lineno = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py W_SyntaxError.w_offset` reader.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_get_syntax_offset(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_syntax_offset }
}

/// `interp_exceptions.py W_SyntaxError.w_offset` writer.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_set_syntax_offset(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_syntax_offset = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py W_SyntaxError.w_text` reader.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_get_syntax_text(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_syntax_text }
}

/// `interp_exceptions.py W_SyntaxError.w_text` writer.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_set_syntax_text(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_syntax_text = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py W_SyntaxError.w_msg` reader.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_get_syntax_msg(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_syntax_msg }
}

/// `interp_exceptions.py W_SyntaxError.w_msg` writer.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_set_syntax_msg(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_syntax_msg = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py W_SyntaxError.w_print_file_and_line` reader.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_get_syntax_print_file_and_line(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_syntax_print_file_and_line }
}

/// `interp_exceptions.py W_SyntaxError.w_print_file_and_line` writer.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_set_syntax_print_file_and_line(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_syntax_print_file_and_line = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py W_SyntaxError.w_end_lineno` reader.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_get_syntax_end_lineno(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_syntax_end_lineno }
}

/// `interp_exceptions.py W_SyntaxError.w_end_lineno` writer.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_set_syntax_end_lineno(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_syntax_end_lineno = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py W_SyntaxError.w_end_offset` reader.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_get_syntax_end_offset(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_syntax_end_offset }
}

/// `interp_exceptions.py W_SyntaxError.w_end_offset` writer.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_set_syntax_end_offset(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_syntax_end_offset = value;
        exception_write_barrier(obj);
    }
}

/// CPython 3.14 `SyntaxError._metadata` reader.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_get_syntax_metadata(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_syntax_metadata }
}

/// CPython 3.14 `SyntaxError._metadata` writer.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_set_syntax_metadata(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_syntax_metadata = value;
        exception_write_barrier(obj);
    }
}

/// `interp_group.py` `exc.w_message` reader.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_get_group_message(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_group_message }
}

/// `interp_group.py` `exc.w_message` writer.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_set_group_message(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_group_message = value;
        exception_write_barrier(obj);
    }
}

/// `interp_group.py` `exc.w_exceptions` reader.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_get_group_exceptions(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_group_exceptions }
}

/// `interp_group.py` `exc.w_exceptions` writer.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_set_group_exceptions(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_group_exceptions = value;
        exception_write_barrier(obj);
    }
}

/// Constructor-time `repr` of the sequence `descr_new` received.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_get_group_exceptions_repr(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionExtended)).w_group_exceptions_repr }
}

/// Constructor-time `repr` of the sequence `descr_new` received.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_exception_set_group_exceptions_repr(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionExtended)).w_group_exceptions_repr = value;
        exception_write_barrier(obj);
    }
}

/// `compile.py` `memory_error = MemoryError()` parity — module-level
/// singleton instance the JIT raises through
/// `PropagateExceptionDescr.handle_fail` when a malloc helper returns
/// NULL.  RPython allocates the singleton at translation time; pyre
/// allocates lazily on first OOM (most workloads never trigger it).
///
/// Stored as `usize` because `PyObjectRef` is `*mut PyObject`, which is
/// neither `Send` nor `Sync` — `OnceLock<usize>` is the standard escape
/// hatch.  The `W_BaseException` lives forever: it is cached in the
/// GC-invisible `OnceLock` and baked into JIT constant pools, so it must
/// stay immortal (`w_exception_new_empty_immortal`), never GC-swept.
pub fn memory_error_singleton() -> PyObjectRef {
    *MEMORY_ERROR_SINGLETON
        .get_or_init(|| w_exception_new_empty_immortal(ExcKind::MemoryError) as usize)
        as PyObjectRef
}

static MEMORY_ERROR_SINGLETON: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

static STANDARD_EXC_INSTANCES: [std::sync::OnceLock<usize>; EXC_KIND_COUNT] =
    [const { std::sync::OnceLock::new() }; EXC_KIND_COUNT];

/// Visit every immortal exception singleton that has actually been created.
///
/// The singletons themselves are `malloc_typed` and outlive every collection,
/// but the `args_w` / `w_traceback` / `w_context` a raise attaches to them are
/// ordinary GC-managed objects. Because the holder is not managed, neither the
/// write barrier nor major seeding reaches those children, so they survive only
/// while some raw carrier happens to be parked on the singleton. Enumerating
/// them here lets a root walker forward the children per object instead.
///
/// Only initialized slots are reported — reading through `get()` never forces
/// an allocation, which must not happen from inside a collection.
pub fn for_each_immortal_exception_singleton(mut visit: impl FnMut(PyObjectRef)) {
    if let Some(&raw) = MEMORY_ERROR_SINGLETON.get() {
        visit(raw as PyObjectRef);
    }
    for slot in STANDARD_EXC_INSTANCES.iter() {
        if let Some(&raw) = slot.get() {
            visit(raw as PyObjectRef);
        }
    }
    if let Ok(table) = PREBUILT_EXC_WITH_MESSAGE.lock() {
        for (_, _, raw) in table.iter() {
            visit(*raw as PyObjectRef);
        }
    }
}

/// `rpython/rtyper/exceptiondata.py get_standard_ll_exc_instance`
/// parity — return the reusable prebuilt instance for `kind`.  RPython's
/// `r_inst.get_reusable_prebuilt_instance()` materialises a single
/// instance per classdef at rtyper construction time and reuses it for
/// every `flatten.py self.emitline("raise", c)` call site (the
/// `_ovf` direct raise path).
///
/// Pyre allocates per `ExcKind` lazily on first access; the resulting
/// pointer is valid for the lifetime of the process and stable across
/// calls so a JIT'd constant pool can carry it as an immediate pointer.
/// Same `OnceLock<usize>` escape hatch as `memory_error_singleton`
/// because `PyObjectRef` is neither `Send` nor `Sync`.
pub fn standard_exc_instance(kind: ExcKind) -> PyObjectRef {
    let slot = &STANDARD_EXC_INSTANCES[kind as u8 as usize];
    *slot.get_or_init(|| w_exception_new_empty_immortal(kind) as usize) as PyObjectRef
}

static PREBUILT_EXC_WITH_MESSAGE: std::sync::Mutex<Vec<(u8, String, usize)>> =
    std::sync::Mutex::new(Vec::new());

/// Immortal exception instance with one text argument.
///
/// Empty `message` is [`standard_exc_instance`]. A repeated `(kind, message)`
/// returns the same object. The instance is visited by
/// [`for_each_immortal_exception_singleton`] so its argument list stays
/// reachable across collections.
pub fn prebuilt_exception_with_message(kind: ExcKind, message: &str) -> PyObjectRef {
    if message.is_empty() {
        return standard_exc_instance(kind);
    }
    if let Some((_, _, raw)) = PREBUILT_EXC_WITH_MESSAGE
        .lock()
        .unwrap()
        .iter()
        .find(|(k, text, _)| *k == kind as u8 && text == message)
    {
        return *raw as PyObjectRef;
    }
    let exc = w_exception_new_empty_immortal(kind);
    let text = crate::unicodeobject::box_str_constant(
        rustpython_wtf8::Wtf8::from_bytes(message.as_bytes())
            .expect("prebuilt exception message is UTF-8"),
    );
    let args = w_exception_args_new(vec![text]);
    unsafe { w_exception_set_args(exc, args) };
    PREBUILT_EXC_WITH_MESSAGE
        .lock()
        .unwrap()
        .push((kind as u8, message.to_string(), exc as usize));
    exc
}

/// Check if an object is an exception instance.
///
/// Uses `ll_isinstance` against the `BaseException` root
/// (`EXCEPTION_TYPE`); every per-kind exception `PyType` is registered
/// as a descendant via `all_foreign_pytypes`, so the
/// `subclassrange_{min,max}` check (`rclass.py ll_issubclass`) matches
/// every subclass without pointer-identity coupling.
///
/// Subclass ranges are stamped by interpreter/JIT startup before Python runs,
/// matching RPython's rtyper-time initialization.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_exception(obj: PyObjectRef) -> bool {
    // rclass.py ll_isinstance reads the once-published vtable directly.
    unsafe { ll_isinstance(obj, &EXCEPTION_TYPE) }
}

/// Get the exception kind tag.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_kind(obj: PyObjectRef) -> ExcKind {
    unsafe { (*(obj as *const W_BaseException)).kind }
}

/// The raw tag byte, read without interpreting it as an `ExcKind`.
///
/// # Safety
/// `obj` must be non-null and point to at least
/// `offset_of!(W_BaseException, kind) + 1` readable bytes.
#[inline]
pub unsafe fn w_exception_kind_byte(obj: PyObjectRef) -> u8 {
    unsafe {
        std::ptr::addr_of!((*(obj as *const W_BaseException)).kind)
            .cast::<u8>()
            .read()
    }
}

/// `w_exception_get_kind` for a value whose provenance is not proven — a raw
/// resume word handed back by the blackhole, say.
///
/// `blackhole.py _exit_frame_with_exception` casts its value to
/// GCREF and every later classification runs through a genuine class lookup
/// (`bh_classof` / `space.exception_match`, i.e. the `rclass.py ll_issubclass`
/// subclass ranges).  Pyre reads a `#[repr(u8)]` tag out of the object
/// instead, which turns a bad value into an out-of-range index into a
/// bounds-check-free jump table (`ExcKind::MAX_DISCRIMINANT`).  This restores
/// the class check in front of the tag read: `is_exception` is the
/// `ll_isinstance` port.
///
/// The class check runs *before* the tag range check. `kind` is a tail field
/// past `ob_header`, so reading it out of a value that is not an exception can
/// read past the object — `W_NoneObject` is a bare header and ends exactly
/// where `kind` starts. `ob_type` sits at offset 0, so the class check needs
/// only the header to be readable, and once `is_exception` holds the object is
/// a `W_BaseException` and the tag load is in bounds. The range check stays
/// after it because the class check does not prove the byte is a live
/// discriminant, and that byte is what indexes the jump table.
///
/// The screens stop at the shape of `ob_type`: they do not prove it addresses
/// a live type. `is_exception` reads the subclass range through it, so an
/// aligned non-null word that is not a type header still faults. Proving it
/// would need a heap-membership test, and the only one available (`is_tracked`)
/// runs as a `gc_op`, which asserts GIL ownership and takes the reentry guard —
/// neither is something a classification on the blackhole resume path may take
/// on. The caller's contract carries that obligation instead.
///
/// # Safety
/// `obj` must be null or point to at least `size_of::<PyObject>()` readable
/// bytes, and its `ob_type` must be null or address a readable type header.
#[inline]
pub unsafe fn w_exception_kind_checked(obj: PyObjectRef) -> Option<ExcKind> {
    // Every screen below precedes the dereference it protects, so that a value
    // which is not an object at all is rejected rather than faulting:
    // alignment before the `ob_type` load, `ob_type`'s own shape before
    // `ll_issubclass` reads the subclass range through it, and the class
    // before the tail-field tag load.
    if obj.is_null() || !(obj as usize).is_multiple_of(align_of::<W_BaseException>()) {
        return None;
    }
    let ob_type = unsafe { (*obj).ob_type };
    if ob_type.is_null()
        || !(ob_type as usize).is_multiple_of(align_of::<crate::pyobject::PyType>())
    {
        return None;
    }
    if !unsafe { is_exception(obj) } {
        return None;
    }
    if unsafe { w_exception_kind_byte(obj) } > ExcKind::MAX_DISCRIMINANT {
        return None;
    }
    Some(unsafe { w_exception_get_kind(obj) })
}

/// Reads the caught exception's `kind` discriminant as an integer, the
/// residual-callable twin of `w_exception_get_kind`.  The tracer cannot
/// model the raw pointer read, so the JIT residualises the call rather than
/// tracing into it (`@dont_look_inside`); the residual resolves its address
/// by qualified path in `jit_trace_fnaddrs`.  A non-inline standalone graph
/// (unlike the `#[inline]` accessor) is what the census residualises.
#[majit_macros::dont_look_inside]
#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
)]
pub fn exc_kind_discriminant(evalue: PyObjectRef) -> i64 {
    // Safety: `evalue` is a valid `W_BaseException` (the caught exception).
    unsafe { w_exception_get_kind(evalue) as i64 }
}

/// Get the Python type name string for an ExcKind.
pub fn exc_kind_name(kind: ExcKind) -> &'static str {
    match kind {
        ExcKind::BaseException => "BaseException",
        ExcKind::Exception => "Exception",
        ExcKind::TypeError => "TypeError",
        ExcKind::ValueError => "ValueError",
        ExcKind::ZeroDivisionError => "ZeroDivisionError",
        ExcKind::NameError => "NameError",
        ExcKind::UnboundLocalError => "UnboundLocalError",
        ExcKind::IndexError => "IndexError",
        ExcKind::KeyError => "KeyError",
        ExcKind::AttributeError => "AttributeError",
        ExcKind::RuntimeError => "RuntimeError",
        ExcKind::StopIteration => "StopIteration",
        ExcKind::StopAsyncIteration => "StopAsyncIteration",
        ExcKind::OverflowError => "OverflowError",
        ExcKind::ArithmeticError => "ArithmeticError",
        ExcKind::ImportError => "ImportError",
        ExcKind::ModuleNotFoundError => "ModuleNotFoundError",
        ExcKind::NotImplementedError => "NotImplementedError",
        ExcKind::AssertionError => "AssertionError",
        ExcKind::ReferenceError => "ReferenceError",
        ExcKind::GeneratorExit => "GeneratorExit",
        ExcKind::RecursionError => "RecursionError",
        ExcKind::OSError => "OSError",
        ExcKind::FileNotFoundError => "FileNotFoundError",
        ExcKind::UnicodeDecodeError => "UnicodeDecodeError",
        ExcKind::UnicodeEncodeError => "UnicodeEncodeError",
        ExcKind::SystemExit => "SystemExit",
        ExcKind::MemoryError => "MemoryError",
        ExcKind::SystemError => "SystemError",
        ExcKind::EOFError => "EOFError",
        ExcKind::LookupError => "LookupError",
        ExcKind::UnicodeError => "UnicodeError",
        ExcKind::UnicodeTranslateError => "UnicodeTranslateError",
        ExcKind::SyntaxError => "SyntaxError",
        ExcKind::BufferError => "BufferError",
    }
}

/// Check if `exc_kind` matches `type_name`, considering Python's
/// exception hierarchy (e.g. ZeroDivisionError is-a ArithmeticError
/// is-a Exception is-a BaseException).
pub fn exc_kind_matches(kind: ExcKind, type_name: &str) -> bool {
    if type_name == "BaseException" {
        return true;
    }
    if type_name == "Exception" {
        return !matches!(
            kind,
            ExcKind::BaseException | ExcKind::GeneratorExit | ExcKind::SystemExit
        );
    }
    if type_name == "ArithmeticError" {
        return matches!(
            kind,
            ExcKind::ArithmeticError | ExcKind::ZeroDivisionError | ExcKind::OverflowError
        );
    }
    if type_name == "RuntimeError" {
        return matches!(kind, ExcKind::RuntimeError | ExcKind::RecursionError);
    }
    if type_name == "NameError" {
        return matches!(kind, ExcKind::NameError | ExcKind::UnboundLocalError);
    }
    // ImportError hierarchy — ModuleNotFoundError is-a ImportError.
    if type_name == "ImportError" {
        return matches!(kind, ExcKind::ImportError | ExcKind::ModuleNotFoundError);
    }
    // OSError hierarchy — FileNotFoundError is-a OSError is-a Exception.
    // IOError / EnvironmentError are aliases for OSError in Python 3.
    if type_name == "OSError" || type_name == "IOError" || type_name == "EnvironmentError" {
        return matches!(kind, ExcKind::OSError | ExcKind::FileNotFoundError);
    }
    // Unicode errors are subclasses of UnicodeError which is a
    // subclass of ValueError, so "ValueError" matches everything in
    // the UnicodeError subtree too.
    if type_name == "ValueError" {
        return matches!(
            kind,
            ExcKind::ValueError
                | ExcKind::UnicodeError
                | ExcKind::UnicodeDecodeError
                | ExcKind::UnicodeEncodeError
                | ExcKind::UnicodeTranslateError
        );
    }
    if type_name == "UnicodeError" {
        return matches!(
            kind,
            ExcKind::UnicodeError
                | ExcKind::UnicodeDecodeError
                | ExcKind::UnicodeEncodeError
                | ExcKind::UnicodeTranslateError
        );
    }
    // LookupError is the intermediate parent of IndexError and KeyError
    // (`interp_exceptions.py` `W_LookupError`).
    if type_name == "LookupError" {
        return matches!(
            kind,
            ExcKind::LookupError | ExcKind::IndexError | ExcKind::KeyError
        );
    }
    exc_kind_name(kind) == type_name
}

/// Convert a Python exception type name to an ExcKind.
pub fn exc_kind_from_name(name: &str) -> Option<ExcKind> {
    match name {
        "BaseException" => Some(ExcKind::BaseException),
        "Exception" => Some(ExcKind::Exception),
        "TypeError" => Some(ExcKind::TypeError),
        "ValueError" => Some(ExcKind::ValueError),
        "ZeroDivisionError" => Some(ExcKind::ZeroDivisionError),
        "NameError" => Some(ExcKind::NameError),
        "UnboundLocalError" => Some(ExcKind::UnboundLocalError),
        "IndexError" => Some(ExcKind::IndexError),
        "KeyError" => Some(ExcKind::KeyError),
        "AttributeError" => Some(ExcKind::AttributeError),
        "RuntimeError" => Some(ExcKind::RuntimeError),
        "StopIteration" => Some(ExcKind::StopIteration),
        "StopAsyncIteration" => Some(ExcKind::StopAsyncIteration),
        "OverflowError" => Some(ExcKind::OverflowError),
        "ArithmeticError" => Some(ExcKind::ArithmeticError),
        "ImportError" => Some(ExcKind::ImportError),
        "ModuleNotFoundError" => Some(ExcKind::ModuleNotFoundError),
        "NotImplementedError" => Some(ExcKind::NotImplementedError),
        "AssertionError" => Some(ExcKind::AssertionError),
        "ReferenceError" => Some(ExcKind::ReferenceError),
        "GeneratorExit" => Some(ExcKind::GeneratorExit),
        // `rpython/rlib/rstackovf.py StackOverflow` is a
        // `RuntimeError` subclass that RPython's rtyper synthesizes
        // catch/convert code for; `rpython/annotator/exception.py:3`
        // lists `_StackOverflow` in the standard set so
        // `get_standard_ll_exc_instance_by_class` has a prebuilt
        // instance for it.  Pyre doesn't have an LL-side StackOverflow
        // class — the stack-check slowpath raises a Python-level
        // `RecursionError` directly (`eval.rs stack_check_slow
        // path → pos_exception()`) — so we alias the RPython name to
        // pyre's `RecursionError` ExcKind: every consumer that looks
        // up the standard pointer receives the singleton instance
        // whose `kind` is the user-visible class, matching what user
        // code would catch.
        "RecursionError" | "_StackOverflow" | "StackOverflow" => Some(ExcKind::RecursionError),
        "OSError" | "IOError" | "EnvironmentError" => Some(ExcKind::OSError),
        "FileNotFoundError" => Some(ExcKind::FileNotFoundError),
        "UnicodeDecodeError" => Some(ExcKind::UnicodeDecodeError),
        "UnicodeEncodeError" => Some(ExcKind::UnicodeEncodeError),
        "SystemExit" => Some(ExcKind::SystemExit),
        "MemoryError" => Some(ExcKind::MemoryError),
        "SystemError" => Some(ExcKind::SystemError),
        "EOFError" => Some(ExcKind::EOFError),
        "LookupError" => Some(ExcKind::LookupError),
        "UnicodeError" => Some(ExcKind::UnicodeError),
        "UnicodeTranslateError" => Some(ExcKind::UnicodeTranslateError),
        "SyntaxError" => Some(ExcKind::SyntaxError),
        "BufferError" => Some(ExcKind::BufferError),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The standalone pyre-object test binary does not run interpreter startup.
    fn seed_subclass_ranges() {
        crate::pyobject::ensure_object_subclass_ranges_initialized();
    }

    #[test]
    fn test_exception_create_and_read() {
        seed_subclass_ranges();
        let obj = w_exception_new(ExcKind::ValueError, "bad value");
        unsafe {
            assert!(is_exception(obj));
            assert_eq!(w_exception_get_kind(obj), ExcKind::ValueError);
            // The message is stored as the single constructor arg.
            let args = w_exception_get_args(obj);
            assert_eq!(crate::tupleobject::w_tuple_len(args), 1);
            let arg0 = crate::tupleobject::w_tuple_getitem(args, 0).unwrap();
            assert_eq!(
                crate::unicodeobject::w_str_get_wtf8(arg0),
                Wtf8::new("bad value")
            );
        }
    }

    /// A byte that is not a live discriminant must not reach `kind_from_exc`:
    /// that match has no wildcard arm, so an out-of-range byte indexes a
    /// bounds-check-free jump table.
    ///
    /// The subject is a stack copy of a real exception, so it clears the
    /// alignment and class screens and the tag range is the gate under test.
    /// A `[u8; N]` buffer would not: its alignment is 1 and its header is
    /// null, so either of the earlier screens could be the one that rejects.
    #[test]
    fn test_kind_checked_rejects_out_of_range_tag() {
        seed_subclass_ranges();
        let live = w_exception_new(ExcKind::ValueError, "bad value");
        let mut copy: W_BaseException = unsafe { std::ptr::read(live as *const W_BaseException) };
        let obj = std::ptr::addr_of_mut!(copy) as PyObjectRef;
        assert_eq!(
            unsafe { w_exception_kind_checked(obj) },
            Some(ExcKind::ValueError),
            "the copy must pass every screen before the tag is corrupted"
        );
        let tag = std::ptr::addr_of_mut!(copy.kind).cast::<u8>();
        for raw in [ExcKind::MAX_DISCRIMINANT + 1, 128, 160, u8::MAX] {
            unsafe { tag.write(raw) };
            assert_eq!(
                unsafe { w_exception_kind_checked(obj) },
                None,
                "tag byte {raw} must be rejected"
            );
        }
    }

    /// The class screen rejects a word whose header is not an exception type
    /// without ever loading the tail field the tag lives in.
    #[test]
    fn test_kind_checked_rejects_non_exception_header() {
        seed_subclass_ranges();
        let buf = [0usize; std::mem::size_of::<W_BaseException>() / 8];
        assert_eq!(
            unsafe { w_exception_kind_checked(buf.as_ptr() as PyObjectRef) },
            None
        );
        assert_eq!(
            unsafe { w_exception_kind_checked(crate::noneobject::w_none()) },
            None
        );
    }

    #[test]
    fn test_kind_checked_accepts_real_exception() {
        seed_subclass_ranges();
        for kind in [ExcKind::ValueError, ExcKind::EOFError] {
            let obj = w_exception_new(kind, "bad value");
            assert_eq!(unsafe { w_exception_kind_checked(obj) }, Some(kind));
        }
        assert_eq!(
            unsafe { w_exception_kind_checked(std::ptr::null_mut()) },
            None
        );
    }

    #[test]
    fn test_exc_kind_matches_hierarchy() {
        assert!(exc_kind_matches(
            ExcKind::ZeroDivisionError,
            "ZeroDivisionError"
        ));
        assert!(exc_kind_matches(
            ExcKind::ZeroDivisionError,
            "ArithmeticError"
        ));
        assert!(exc_kind_matches(ExcKind::ZeroDivisionError, "Exception"));
        assert!(exc_kind_matches(
            ExcKind::ZeroDivisionError,
            "BaseException"
        ));
        assert!(!exc_kind_matches(ExcKind::ZeroDivisionError, "ValueError"));
        assert!(exc_kind_matches(
            ExcKind::UnboundLocalError,
            "UnboundLocalError"
        ));
        assert!(exc_kind_matches(ExcKind::UnboundLocalError, "NameError"));
        assert!(exc_kind_matches(ExcKind::UnboundLocalError, "Exception"));
        assert!(exc_kind_matches(
            ExcKind::UnboundLocalError,
            "BaseException"
        ));
    }

    #[test]
    fn test_exc_kind_from_name_roundtrip() {
        // Every variant of ExcKind must round-trip through
        // exc_kind_name → exc_kind_from_name so the per-kind class
        // registry (`register_exc_class_for_kind`) plumbed by
        // pyre-interpreter::builtins::register_exc_class can install a
        // class pointer for every `w_exception_new(kind, ...)` callsite.
        // A gap here would leave that kind's `ob_header.w_class` at the
        // generic `EXCEPTION_TYPE` stub, breaking the "the object's
        // class is the exception type" invariant on the w_class read
        // path.
        for kind in [
            ExcKind::BaseException,
            ExcKind::Exception,
            ExcKind::TypeError,
            ExcKind::ValueError,
            ExcKind::ZeroDivisionError,
            ExcKind::NameError,
            ExcKind::UnboundLocalError,
            ExcKind::IndexError,
            ExcKind::KeyError,
            ExcKind::AttributeError,
            ExcKind::RuntimeError,
            ExcKind::StopIteration,
            ExcKind::StopAsyncIteration,
            ExcKind::OverflowError,
            ExcKind::ArithmeticError,
            ExcKind::ImportError,
            ExcKind::NotImplementedError,
            ExcKind::AssertionError,
            ExcKind::ReferenceError,
            ExcKind::GeneratorExit,
            ExcKind::RecursionError,
            ExcKind::OSError,
            ExcKind::FileNotFoundError,
            ExcKind::UnicodeDecodeError,
            ExcKind::UnicodeEncodeError,
            ExcKind::SystemExit,
            ExcKind::MemoryError,
            ExcKind::SystemError,
            ExcKind::EOFError,
            ExcKind::LookupError,
            ExcKind::UnicodeError,
            ExcKind::UnicodeTranslateError,
            ExcKind::BufferError,
        ] {
            let name = exc_kind_name(kind);
            assert_eq!(
                exc_kind_from_name(name),
                Some(kind),
                "exc_kind_from_name({name:?}) round-trip failed for {kind:?}",
            );
        }
    }

    #[test]
    fn memory_error_singleton_is_idempotent_and_typed() {
        seed_subclass_ranges();
        let a = memory_error_singleton();
        let b = memory_error_singleton();
        assert_eq!(a as usize, b as usize, "singleton must be stable");
        unsafe {
            assert!(is_exception(a));
            assert_eq!(w_exception_get_kind(a), ExcKind::MemoryError);
            // Empty message → no constructor args (`args == ()`).
            assert_eq!(crate::tupleobject::w_tuple_len(w_exception_get_args(a)), 0);
        }
    }

    #[test]
    fn standard_exc_instance_is_idempotent_and_per_kind_distinct() {
        seed_subclass_ranges();
        // RPython `get_standard_ll_exc_instance` returns the same
        // prebuilt instance pointer across repeated lookups (it's the
        // `_reusable_prebuilt_instance` slot on the InstanceRepr).
        // Pyre matches by caching per-`ExcKind`; the test pins both
        // the idempotence (same kind → same pointer) and the per-kind
        // distinctness (different kinds → different pointers, so the
        // JIT cannot accidentally merge `raise OverflowError` and
        // `raise ZeroDivisionError` into the same singleton).
        let overflow_a = standard_exc_instance(ExcKind::OverflowError);
        let overflow_b = standard_exc_instance(ExcKind::OverflowError);
        assert_eq!(
            overflow_a as usize, overflow_b as usize,
            "per-kind singleton must be stable across calls"
        );
        let zerodiv = standard_exc_instance(ExcKind::ZeroDivisionError);
        assert_ne!(
            overflow_a as usize, zerodiv as usize,
            "distinct ExcKinds must yield distinct singleton pointers"
        );
        unsafe {
            assert!(is_exception(overflow_a));
            assert_eq!(w_exception_get_kind(overflow_a), ExcKind::OverflowError);
            assert_eq!(w_exception_get_kind(zerodiv), ExcKind::ZeroDivisionError);
        }
    }

    #[test]
    fn immortal_singleton_enumeration_reports_created_and_forces_none() {
        seed_subclass_ranges();
        // The root walker forwards each reported singleton's reference slots,
        // so the enumeration must cover every singleton that exists — and must
        // never itself create one, since it runs from inside a collection.
        let mem = memory_error_singleton();
        let key = standard_exc_instance(ExcKind::KeyError);

        let mut seen = Vec::new();
        for_each_immortal_exception_singleton(|exc| seen.push(exc as usize));
        assert!(
            seen.contains(&(mem as usize)),
            "MemoryError singleton must be reported once created"
        );
        assert!(
            seen.contains(&(key as usize)),
            "per-kind singleton must be reported once created"
        );
        assert!(
            seen.len() <= EXC_KIND_COUNT + 1,
            "enumeration is bounded by the per-kind slots plus MemoryError"
        );

        // Enumerating must not initialize a slot: a second pass still reports
        // every singleton the first one did, stays within the bound, and hands
        // back real exception objects.  The slots are process-global and the
        // test binary runs its cases concurrently, so a sibling case creating
        // another kind in between makes the second set a superset — comparing
        // the two for equality would fail on that alone.
        let mut again = Vec::new();
        for_each_immortal_exception_singleton(|exc| again.push(exc as usize));
        for raw in &seen {
            assert!(again.contains(raw), "enumeration must not drop a singleton");
        }
        assert!(
            again.len() <= EXC_KIND_COUNT + 1,
            "enumeration is bounded by the per-kind slots plus MemoryError"
        );
        for &raw in &again {
            assert!(unsafe { is_exception(raw as PyObjectRef) });
        }
    }

    #[test]
    fn w_exception_gc_type_id_matches_descr() {
        assert_eq!(W_BASE_EXCEPTION_GC_TYPE_ID, 31);
        assert_eq!(
            <W_BaseException as crate::lltype::GcType>::type_id(),
            W_BASE_EXCEPTION_GC_TYPE_ID
        );
        assert_eq!(
            <W_BaseException as crate::lltype::GcType>::SIZE,
            W_BASE_EXCEPTION_SIZE
        );
        assert_eq!(
            <W_ExceptionExtended as crate::lltype::GcType>::SIZE,
            W_EXCEPTION_EXTENDED_SIZE
        );
        assert_eq!(W_EXCEPTION_EXTENDED_GC_TYPE_ID, 214);
        assert_eq!(W_BASE_EXCEPTION_USER_GC_TYPE_ID, 215);
        assert_eq!(W_EXCEPTION_EXTENDED_USER_GC_TYPE_ID, 216);
        assert_eq!(
            <W_ExceptionExtended as crate::lltype::GcType>::type_id(),
            W_EXCEPTION_EXTENDED_GC_TYPE_ID
        );
        assert_eq!(
            <W_BaseExceptionUser as crate::lltype::GcType>::type_id(),
            W_BASE_EXCEPTION_USER_GC_TYPE_ID
        );
        assert_eq!(
            <W_ExceptionExtendedUser as crate::lltype::GcType>::type_id(),
            W_EXCEPTION_EXTENDED_USER_GC_TYPE_ID
        );
        assert_eq!(
            <W_BaseExceptionUser as crate::lltype::GcType>::SIZE,
            W_BASE_EXCEPTION_USER_SIZE
        );
        assert_eq!(
            <W_ExceptionExtendedUser as crate::lltype::GcType>::SIZE,
            W_EXCEPTION_EXTENDED_USER_SIZE
        );
        assert_eq!(EXCEPTION_TYPE.weakref_offset, 0);
        assert_eq!(EXCEPTION_TYPE.mapdict_offset, 0);
        assert_eq!(BASE_EXCEPTION_USER_TYPE.weakref_offset, 0);
        assert_eq!(EXCEPTION_EXTENDED_USER_TYPE.weakref_offset, 0);
        assert_ne!(BASE_EXCEPTION_USER_TYPE.mapdict_offset, 0);
        assert_ne!(EXCEPTION_EXTENDED_USER_TYPE.mapdict_offset, 0);
        assert!(
            W_BASE_EXCEPTION_SIZE <= 80,
            "slim W_BaseException must stay near PyPy SizeDescr 72, got {}",
            W_BASE_EXCEPTION_SIZE
        );
        assert!(
            W_EXCEPTION_EXTENDED_SIZE > W_BASE_EXCEPTION_SIZE,
            "extended layout must be larger than the slim base"
        );
        assert_eq!(RLIST_SIZE, 16);
        assert_eq!(RLIST_ITEMS_OFFSET, 8);
    }

    #[test]
    fn rlist_new_roundtrip() {
        let a = crate::intobject::w_int_new(7);
        let b = crate::intobject::w_int_new(8);
        let list = rlist_new(vec![a, b]);
        assert_eq!(unsafe { rlist_len(list) }, 2);
        assert_eq!(
            unsafe { crate::intobject::w_int_get_value(rlist_getitem(list, 0)) },
            7
        );
        assert_eq!(
            unsafe { crate::intobject::w_int_get_value(rlist_getitem(list, 1)) },
            8
        );
        let empty = rlist_new(Vec::new());
        assert!(
            !empty.is_null(),
            "ll_fixed_newlist mallocs a 0-length array"
        );
        assert_eq!(unsafe { rlist_len(empty) }, 0);
    }

    #[test]
    fn get_args_adopts_fixed_list_except_pair() {
        let exc = w_exception_new_empty(ExcKind::ValueError);
        let stored = w_exception_args_new(vec![
            crate::intobject::w_int_new(1),
            crate::intobject::w_int_new(2),
            crate::intobject::w_int_new(3),
        ]);
        unsafe { w_exception_set_args(exc, stored) };
        let first = unsafe { w_exception_get_args(exc) };
        let second = unsafe { w_exception_get_args(exc) };
        assert!(!std::ptr::eq(first, second));
        assert_eq!(unsafe { crate::tupleobject::w_tuple_len(first) }, 3);
        let adopted = unsafe {
            (*(first as *const crate::tupleobject::W_TupleObject)).wrappeditems as PyObjectRef
        };
        assert!(std::ptr::eq(adopted, stored));
        let adopted_again = unsafe {
            (*(second as *const crate::tupleobject::W_TupleObject)).wrappeditems as PyObjectRef
        };
        assert!(std::ptr::eq(adopted_again, stored));

        let pair = w_exception_args_new(vec![
            crate::intobject::w_int_new(4),
            crate::intobject::w_int_new(5),
        ]);
        unsafe { w_exception_set_args(exc, pair) };
        let specialised = unsafe { w_exception_get_args(exc) };
        assert_eq!(unsafe { crate::tupleobject::w_tuple_len(specialised) }, 2);
        assert_eq!(
            unsafe {
                crate::intobject::w_int_get_value(
                    crate::tupleobject::w_tuple_getitem(specialised, 0).unwrap(),
                )
            },
            4
        );
        assert_eq!(
            unsafe {
                crate::intobject::w_int_get_value(
                    crate::tupleobject::w_tuple_getitem(specialised, 1).unwrap(),
                )
            },
            5
        );
    }

    /// Install `kind`'s realbase instantiate slot and class registry.
    ///
    /// The pyre-object test binary never runs `init_typeobjects`. A null
    /// instantiate reads as the exact realbase, so `ValueError()` would stay
    /// on `EXCEPTION_TYPE` until a distinct class is registered. Only this
    /// test registers these kinds. If the slot already holds a class, keep
    /// that class: overwriting a foreign instantiate would retarget every
    /// later exact construction in this process.
    fn install_realbase(kind: ExcKind, tp: &'static PyType) -> PyObjectRef {
        let shell = crate::typeobject::w_type_alloc_builtin();
        let installed = tp.instantiate.compare_exchange(
            std::ptr::null_mut(),
            shell,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        );
        let instantiate = match installed {
            Ok(_) => shell,
            Err(existing) => existing,
        };
        let registered = register_exc_class_for_kind(kind, instantiate);
        if installed.is_ok() && !std::ptr::eq(registered, shell) {
            set_instantiate(tp, registered);
        }
        registered
    }

    #[test]
    fn allocate_instance_stamps_user_layout_unless_class_is_realbase() {
        seed_subclass_ranges();
        let base_cls = install_realbase(ExcKind::BaseException, &EXCEPTION_TYPE);
        let value_shell = crate::typeobject::w_type_alloc_builtin();
        let value_cls = register_exc_class_for_kind(ExcKind::ValueError, value_shell);
        assert!(
            !std::ptr::eq(value_cls, base_cls),
            "ValueError must not be the BaseException realbase"
        );
        let _os_cls = install_realbase(ExcKind::OSError, &EXC_OS_ERROR_TYPE);
        let perm = crate::typeobject::w_type_alloc_builtin();

        let base = w_exception_new_empty(ExcKind::BaseException);
        let value = w_exception_new_empty(ExcKind::ValueError);
        let os = w_exception_new_empty(ExcKind::OSError);
        let perm_exc = w_exception_new_empty_for_class(ExcKind::OSError, perm);
        unsafe {
            assert!(std::ptr::eq((*base).ob_type, &EXCEPTION_TYPE));
            assert!(std::ptr::eq((*value).ob_type, &BASE_EXCEPTION_USER_TYPE));
            assert!(std::ptr::eq((*os).ob_type, &EXC_OS_ERROR_TYPE));
            assert!(std::ptr::eq(
                (*perm_exc).ob_type,
                &EXCEPTION_EXTENDED_USER_TYPE
            ));
            assert!(is_exception(base));
            assert!(is_exception(value));
            assert!(is_exception(os));
            assert!(is_exception(perm_exc));
            assert_eq!(w_exception_get_kind(base), ExcKind::BaseException);
            assert_eq!(w_exception_get_kind(value), ExcKind::ValueError);
            assert_eq!(w_exception_get_kind(os), ExcKind::OSError);
            assert_eq!(w_exception_get_kind(perm_exc), ExcKind::OSError);
        }
    }
}
