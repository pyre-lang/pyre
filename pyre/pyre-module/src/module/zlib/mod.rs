//! zlib module — PyPy: `pypy/module/zlib/`.
//!
//! CRC-32 / Adler-32 checksums plus the DEFLATE compress/decompress surface.
//! The DEFLATE machinery is shared with RustPython through
//! `rustpython_common::compression::zlib`, which Charon never enters because it
//! is a git dependency; this module is the W_Root object glue.
//!
//! `Compress` / `Decompress` / `_ZlibDecompressor` own their native stream
//! directly, matching `interp_zlib.py`'s `self.stream`.  Each stream also owns
//! its PyPy-style per-object lock; there is no process-global side table.

use pyre_object::*;
use rustpython_common::compression::zlib as backend;

use parking_lot::Mutex;

/// PyPy `interp_zlib.py Compress`: the stream and lock belong to the
/// wrapper object. A user subclass is `typedef.py` `_getusercls`
/// (`MapdictStorageMixin`), allocated by `objspace.py` `allocate_instance`.
// CPython 3.14 Modules/zlibmodule.c:zlib_exec creates this heap spec without
// IMMUTABLETYPE, so its type namespace remains mutable.
#[pyre_interpreter::pyre_class("zlib.Compress", cpython_mutable, user_layout)]
#[derive(Default)]
pub struct W_Compress {
    backend: *mut Mutex<backend::Compressor>,
}

/// PyPy `interp_zlib.py Decompress` object-owned stream state. A user
/// subclass is `typedef.py` `_getusercls`, same as `Compress`.
// Same `zlib_exec` mutable heap owner as Compress.
#[pyre_interpreter::pyre_class("zlib.Decompress", cpython_mutable, user_layout)]
#[derive(Default)]
pub struct W_Decompress {
    backend: *mut Mutex<backend::Decompressor>,
}

/// PyPy `interp_zlib.py default_buffer_size ZlibDecompressor` object-owned buffered stream.
/// Not an acceptable base: the payload is the stream alone.
// CPython 3.14's ZlibDecompressorType spec adds IMMUTABLETYPE, unlike the two
// public stream types above.
#[pyre_interpreter::pyre_class("zlib._ZlibDecompressor", cpython_heaptype)]
#[derive(Default)]
pub struct W_ZlibDecompressor {
    backend: *mut Mutex<backend::ZlibDecompressor>,
}

fn compressor_this(obj: PyObjectRef) -> Result<&'static mut W_Compress, pyre_interpreter::PyError> {
    W_Compress::from_obj(obj)
        .ok_or_else(|| pyre_interpreter::PyError::type_error("expected Compress object"))
}

fn decompressor_this(
    obj: PyObjectRef,
) -> Result<&'static mut W_Decompress, pyre_interpreter::PyError> {
    W_Decompress::from_obj(obj)
        .ok_or_else(|| pyre_interpreter::PyError::type_error("expected Decompress object"))
}

fn zdecompressor_this(
    obj: PyObjectRef,
) -> Result<&'static mut W_ZlibDecompressor, pyre_interpreter::PyError> {
    W_ZlibDecompressor::from_obj(obj)
        .ok_or_else(|| pyre_interpreter::PyError::type_error("expected _ZlibDecompressor object"))
}

/// Sweep-time counterparts of the three RPython `_finalize_` methods.  The
/// Box contains the per-object mutex and native stream; dropping it neither
/// allocates nor calls back into Python.
pub unsafe fn w_compress_dealloc(obj: PyObjectRef) {
    if let Some(this) = W_Compress::from_obj(obj)
        && !this.backend.is_null()
    {
        unsafe { drop(Box::from_raw(this.backend)) };
        this.backend = std::ptr::null_mut();
    }
}

pub unsafe fn w_decompress_dealloc(obj: PyObjectRef) {
    if let Some(this) = W_Decompress::from_obj(obj)
        && !this.backend.is_null()
    {
        unsafe { drop(Box::from_raw(this.backend)) };
        this.backend = std::ptr::null_mut();
    }
}

pub unsafe fn w_zdecompress_dealloc(obj: PyObjectRef) {
    if let Some(this) = W_ZlibDecompressor::from_obj(obj)
        && !this.backend.is_null()
    {
        unsafe { drop(Box::from_raw(this.backend)) };
        this.backend = std::ptr::null_mut();
    }
}

// ── errors ──────────────────────────────────────────────────────────────

fn zlib_error(msg: impl Into<String>) -> pyre_interpreter::PyError {
    let msg = msg.into();
    // `interp_zlib.py zlib_error`: `OperationError(w_error, space.newtext(msg))`.
    let Some(cls) = pyre_interpreter::builtins::lookup_exc_class("zlib.error") else {
        return pyre_interpreter::PyError::value_error(msg);
    };
    let _roots = pyre_object::gc_roots::push_roots();
    let cls_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(cls);
    let w_value = w_str_new_managed(&msg);
    pyre_interpreter::PyError::from_type_and_value(
        pyre_object::gc_roots::shadow_stack_get(cls_slot),
        w_value,
    )
}

fn init_error(error: backend::InitError) -> pyre_interpreter::PyError {
    match error {
        backend::InitError::InvalidOption => {
            pyre_interpreter::PyError::value_error("Invalid initialization option")
        }
        backend::InitError::Zlib(message) => zlib_error(message),
    }
}

/// `interp_zlib.py` `compress` / `decompress` wrap the `deflateInit` /
/// `inflateInit` call in `except ValueError` and re-raise it as `zlib.error`
/// carrying their own message; the object constructors let the `ValueError`
/// through instead.
fn oneshot_init_error(error: backend::InitError, bad_option: &str) -> pyre_interpreter::PyError {
    let message = match error {
        backend::InitError::InvalidOption => bad_option.to_owned(),
        backend::InitError::Zlib(message) => message,
    };
    zlib_error(message)
}

fn eof_error(msg: &str) -> pyre_interpreter::PyError {
    pyre_interpreter::PyError::new(pyre_interpreter::PyErrorKind::EOFError, msg)
}

// ── argument helpers ────────────────────────────────────────────────────

/// `interp_zlib.py` spells every data argument `'bufferstr'`, so an
/// `array.array` or a C-contiguous `memoryview` is as acceptable as `bytes`.
fn as_bytes(obj: PyObjectRef) -> Result<Vec<u8>, pyre_interpreter::PyError> {
    pyre_interpreter::baseobjspace::charbuf_w(obj)
}

/// Coerce an argument slot to an integer, `None`/absent (`PY_NULL`) → default.
///
/// The Signature-bound call path fills every declared slot: a supplied
/// keyword / positional carries the value, an omitted optional carries
/// `PY_NULL`.
fn int_or_default(o: PyObjectRef, default: i64) -> Result<i64, pyre_interpreter::PyError> {
    if o.is_null() || unsafe { is_none(o) } {
        Ok(default)
    } else {
        pyre_interpreter::baseobjspace::int_w(o)
    }
}

/// Coerce an optional `zdict` slot to bytes, `None`/absent (`PY_NULL`) → None.
fn zdict_or_none(o: PyObjectRef) -> Result<Option<Vec<u8>>, pyre_interpreter::PyError> {
    if o.is_null() || unsafe { is_none(o) } {
        Ok(None)
    } else {
        Ok(Some(as_bytes(o)?))
    }
}

/// `rzlib` passes window bits through a C `int`.  Preserve that 32-bit
/// wrapping boundary; narrowing to `i8` would incorrectly turn e.g. 271 into
/// the valid value 15.
fn to_wbits(v: i64) -> i32 {
    v as i32
}

// ── checksums ───────────────────────────────────────────────────────────

fn crc32_compute(buf: &[u8], start: u32) -> u32 {
    let mut crc = !start;
    for &b in buf {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn adler32_compute(buf: &[u8], start: u32) -> u32 {
    let mut a = start & 0xffff;
    let mut b = (start >> 16) & 0xffff;
    for &x in buf {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

// ── Compress (compressobj) ──────────────────────────────────────────────

static COMPRESS_RUNTIME_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();
static DECOMPRESS_RUNTIME_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();
static ZDECOMPRESS_RUNTIME_TYPE: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

/// [3.14-spec] CPython's zlib specs publish `__module__ = "zlib"`; PyPy's
/// unqualified `TypeDef('Compress')` / `TypeDef('Decompress')` do not.  Keep
/// those PyPy type names and add only the observable namespace entry.
fn publish_cpython_module(ns: PyObjectRef) {
    let roots = pyre_object::gc_roots::push_roots();
    let _ = roots.pin_root(ns);
    let slot = roots.base();
    let module = pyre_object::w_str_new("zlib");
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            roots.get(slot),
            "__module__",
            module,
        );
    }
}

fn compress_type() -> PyObjectRef {
    COMPRESS_RUNTIME_TYPE.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type_with_layout(
            "Compress",
            init_compress_type,
            pyre_interpreter::typedef::w_object(),
            <W_Compress as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        // CPython 3.14 Modules/zlibmodule.c:zlib_exec creates Comptype from a
        // mutable heap spec.
        pyre_interpreter::typedef::mark_cpython_heap_type(tp, false);
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_Compress as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        tp
    })
}

fn init_compress_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    publish_cpython_module(pyre_object::gc_roots::shadow_stack_get(ns_slot));
    let new_sig = {
        let mut b = pyre_interpreter::SignatureBuilder::default();
        b.append("cls");
        b.marker_posonly();
        b.append("level");
        b.append("method");
        b.append("wbits");
        b.append("memLevel");
        b.append("strategy");
        b.append("zdict");
        b.signature()
    };
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__new__",
        pyre_interpreter::typedef::make_new_descr_maybe_sig(compress_new, Some(new_sig))
    );
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "compress",
            pyre_interpreter::make_builtin_function_with_arity(
                "compress",
                |args| {
                    if args.len() < 2 {
                        return Err(pyre_interpreter::PyError::type_error(
                            "compress() missing data",
                        ));
                    }
                    let mut w_self = args[0];
                    let w_data = args[1];
                    let data = pyre_object::with_roots!(w_self => as_bytes(w_data))?;
                    let this = compressor_this(w_self)?;
                    if this.backend.is_null() {
                        return Err(zlib_error("Error -2: inconsistent stream state"));
                    }
                    let mut c = { &*this.backend }.lock();
                    let out = c.compress(&data).map_err(zlib_error)?;
                    Ok(bytesobject::w_bytes_from_bytes(&out))
                },
                2,
            )
        )
    };
    for name in ["copy", "__copy__"] {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            name,
            pyre_interpreter::make_builtin_function_with_arity(name, compress_copy, 1)
        );
    }
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__deepcopy__",
        pyre_interpreter::make_builtin_function_with_arity("__deepcopy__", compress_copy, 2,)
    );
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "flush",
            pyre_interpreter::make_builtin_function("flush", |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error(
                        "flush() missing self",
                    ));
                }
                // interp_zlib.py `@unwrap_spec(mode="c_int")` — the
                // converter reports a value outside the C `int` range rather
                // than truncating it into a different flush mode.
                let mut w_self = args[0];
                let mode = match args.get(1).copied() {
                    Some(o) if !{ is_none(o) } => {
                        pyre_object::with_roots!(w_self => pyre_interpreter::baseobjspace::c_int_w(o))?
                    }
                    _ => backend::Z_FINISH,
                };
                let this = compressor_this(w_self)?;
                if this.backend.is_null() {
                    return Err(zlib_error("Error -2: inconsistent stream state"));
                }
                let mut c = { &*this.backend }.lock();
                let out = c.flush(mode).map_err(zlib_error)?;
                Ok(bytesobject::w_bytes_from_bytes(&out))
            })
        )
    };
}

fn compress_new(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // An empty argument list's class is a fresh type object.  Otherwise slot
    // 0 is the class the caller passed, and every later index is read back
    // from the same pin: each `int_or_default` can allocate.
    let n_args = args.len();
    let _roots = pyre_object::gc_roots::push_roots();
    let base = if n_args == 0 {
        pyre_object::gc_roots::pin_roots(&[compress_type()])
    } else {
        pyre_object::gc_roots::pin_roots(args)
    };
    let arg = |index: usize| {
        if n_args == 0 || index >= n_args {
            PY_NULL
        } else {
            pyre_object::gc_roots::shadow_stack_get(base + index)
        }
    };
    let level = int_or_default(arg(1), -1)? as i32;
    let method = int_or_default(arg(2), 8)? as i32;
    let wbits = to_wbits(int_or_default(arg(3), backend::MAX_WBITS as i64)?);
    let mem_level = int_or_default(arg(4), 8)? as i32;
    let strategy = int_or_default(arg(5), 0)? as i32;
    let zdict = zdict_or_none(arg(6))?;
    make_compress_for(
        pyre_object::gc_roots::shadow_stack_get(base),
        level,
        method,
        wbits,
        mem_level,
        strategy,
        zdict,
    )
}

fn make_compress(
    level: i32,
    method: i32,
    wbits: i32,
    mem_level: i32,
    strategy: i32,
    zdict: Option<Vec<u8>>,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    make_compress_for(
        compress_type(),
        level,
        method,
        wbits,
        mem_level,
        strategy,
        zdict,
    )
}

fn make_compress_for(
    cls: PyObjectRef,
    level: i32,
    method: i32,
    wbits: i32,
    mem_level: i32,
    strategy: i32,
    zdict: Option<Vec<u8>>,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if !(-1..=9).contains(&level)
        || method != 8
        || !((-15..=-9).contains(&wbits) || (8..=15).contains(&wbits) || (25..=31).contains(&wbits))
        || !(1..=9).contains(&mem_level)
        || !(0..=4).contains(&strategy)
    {
        // interp_zlib.py:244-247: rzlib.deflateInit reports invalid
        // initialization options as ValueError, distinct from operational
        // zlib.error failures.
        return Err(pyre_interpreter::PyError::value_error(
            "Invalid initialization option",
        ));
    }
    let c = backend::Compressor::new(level, method, wbits, mem_level, strategy, zdict.as_deref())
        .map_err(init_error)?;
    allocate_compress(cls, c)
}

/// `interp_zlib.py Compress.copy` and `descr_deepcopy`: copy the live
/// deflate stream while holding the source object's lock, then construct the
/// base `Compress` type (PyPy deliberately does not preserve a subclass).
fn compress_copy(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let obj = args.first().copied().unwrap_or(PY_NULL);
    let copied = {
        let this = compressor_this(obj)?;
        if this.backend.is_null() {
            return Err(pyre_interpreter::PyError::value_error(
                "Compressor was already flushed",
            ));
        }
        let mut compressor = unsafe { &*this.backend }.lock();
        if compressor.is_finished() {
            return Err(pyre_interpreter::PyError::value_error(
                "Compressor was already flushed",
            ));
        }
        compressor.copy().map_err(zlib_error)?
    };
    allocate_compress(compress_type(), copied)
}

fn allocate_compress(
    cls: PyObjectRef,
    backend: backend::Compressor,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let backend = Box::into_raw(Box::new(Mutex::new(backend)));
    let _roots = gc_roots::push_roots();
    let _ = gc_roots::pin_root(cls);
    let cls_slot = gc_roots::shadow_stack_len() - 1;
    Ok(W_Compress::allocate_instance(
        W_Compress {
            ob: PyObject::default(),
            backend,
        },
        gc_roots::shadow_stack_get(cls_slot),
    ))
}

// ── Decompress (decompressobj) ──────────────────────────────────────────

fn decompress_type() -> PyObjectRef {
    DECOMPRESS_RUNTIME_TYPE.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type_with_layout(
            "Decompress",
            init_decompress_type,
            pyre_interpreter::typedef::w_object(),
            <W_Decompress as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        // `zlib_exec` creates Decomptype from a mutable heap spec.
        pyre_interpreter::typedef::mark_cpython_heap_type(tp, false);
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_Decompress as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        tp
    })
}

fn decompress_getset(
    ns: PyObjectRef,
    name: &'static str,
    f: pyre_interpreter::gateway::BuiltinCodeFn,
) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        name,
        pyre_interpreter::typedef::make_getset_descriptor_named(
            pyre_interpreter::make_builtin_function_with_arity(name, f, 2),
            name,
        )
    );
}

// interp_zlib.py `decompress(self, data, __posonly__=None,
// max_length=0)`, exposed as `($self, data, /, max_length=0)`.  The signature
// binder is essential here: pip's wheel reader passes max_length by keyword,
// and an unbound native callable otherwise sees the kwargs transport object as
// a positional integer.
fn decompress_decompress(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let data_obj = args.get(1).copied().unwrap_or(PY_NULL);
    if data_obj.is_null() {
        return Err(pyre_interpreter::PyError::type_error(
            "decompress() missing data",
        ));
    }
    let mut w_self = args[0];
    let mut w_max_length = args.get(2).copied().unwrap_or(PY_NULL);
    let data = pyre_object::with_roots!(w_self, w_max_length => as_bytes(data_obj))?;
    // An omitted `max_length` is unlimited; a supplied value — including
    // `None` — goes through `int_w`.  Zero also means unlimited here, and a
    // negative value is rejected.
    let max_length = if w_max_length.is_null() {
        None
    } else {
        let v = pyre_object::with_roots!(w_self => pyre_interpreter::baseobjspace::int_w(w_max_length))?;
        if v < 0 {
            return Err(pyre_interpreter::PyError::value_error(
                "max_length must be non-negative",
            ));
        }
        (v != 0).then_some(v as usize)
    };
    let this = decompressor_this(w_self)?;
    if this.backend.is_null() {
        return Err(zlib_error("Error -2: inconsistent stream state"));
    }
    let mut d = unsafe { &*this.backend }.lock();
    let out = d.decompress(&data, max_length).map_err(zlib_error)?;
    Ok(bytesobject::w_bytes_from_bytes(&out))
}

fn init_decompress_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    publish_cpython_module(pyre_object::gc_roots::shadow_stack_get(ns_slot));
    let new_sig = {
        let mut b = pyre_interpreter::SignatureBuilder::default();
        b.append("cls");
        b.marker_posonly();
        b.append("wbits");
        b.append("zdict");
        b.signature()
    };
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__new__",
        pyre_interpreter::typedef::make_new_descr_maybe_sig(decompress_new, Some(new_sig))
    );
    let decompress_sig = {
        let mut b = pyre_interpreter::SignatureBuilder::default();
        b.append("self");
        b.append("data");
        b.marker_posonly();
        b.append("max_length");
        b.signature()
    };
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "decompress",
        pyre_interpreter::make_builtin_function_maybe_sig(
            "decompress",
            decompress_decompress,
            Some(decompress_sig),
        )
    );
    for name in ["copy", "__copy__"] {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            name,
            pyre_interpreter::make_builtin_function_with_arity(name, decompress_copy, 1)
        );
    }
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__deepcopy__",
        pyre_interpreter::make_builtin_function_with_arity("__deepcopy__", decompress_copy, 2,)
    );
    unsafe {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            "flush",
            pyre_interpreter::make_builtin_function("flush", |args| {
                if args.is_empty() {
                    return Err(pyre_interpreter::PyError::type_error(
                        "flush() missing self",
                    ));
                }
                let mut w_self = args[0];
                let length = match args.get(1).copied() {
                    Some(o) if !{ is_none(o) } => {
                        let v = pyre_object::with_roots!(w_self => pyre_interpreter::baseobjspace::int_w(o))?;
                        if v <= 0 {
                            return Err(pyre_interpreter::PyError::value_error(
                                "length must be greater than zero",
                            ));
                        }
                        v as usize
                    }
                    _ => backend::DEF_BUF_SIZE,
                };
                let this = decompressor_this(w_self)?;
                if this.backend.is_null() {
                    return Err(zlib_error("Error -2: inconsistent stream state"));
                }
                let mut d = { &*this.backend }.lock();
                let out = d.flush(length).map_err(zlib_error)?;
                Ok(bytesobject::w_bytes_from_bytes(&out))
            })
        )
    };
    decompress_getset(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "unused_data",
        |args| {
            let this = decompressor_this(args.get(1).copied().unwrap_or(PY_NULL))?;
            let data = if this.backend.is_null() {
                Vec::new()
            } else {
                unsafe { &*this.backend }.lock().unused_data().to_vec()
            };
            Ok(bytesobject::w_bytes_from_bytes(&data))
        },
    );
    decompress_getset(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "unconsumed_tail",
        |args| {
            let this = decompressor_this(args.get(1).copied().unwrap_or(PY_NULL))?;
            let data = if this.backend.is_null() {
                Vec::new()
            } else {
                unsafe { &*this.backend }.lock().unconsumed_tail().to_vec()
            };
            Ok(bytesobject::w_bytes_from_bytes(&data))
        },
    );
    decompress_getset(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "eof",
        |args| {
            let this = decompressor_this(args.get(1).copied().unwrap_or(PY_NULL))?;
            Ok(w_bool_from(
                !this.backend.is_null() && unsafe { &*this.backend }.lock().eof(),
            ))
        },
    );
}

fn decompress_new(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // Same shape as `compress_new`: the class is the fresh type only when no
    // arguments were passed, and is argument 0 otherwise.
    let n_args = args.len();
    let _roots = pyre_object::gc_roots::push_roots();
    let base = if n_args == 0 {
        pyre_object::gc_roots::pin_roots(&[decompress_type()])
    } else {
        pyre_object::gc_roots::pin_roots(args)
    };
    let arg = |index: usize| {
        if n_args == 0 || index >= n_args {
            PY_NULL
        } else {
            pyre_object::gc_roots::shadow_stack_get(base + index)
        }
    };
    let wbits = to_wbits(int_or_default(arg(1), backend::MAX_WBITS as i64)?);
    let zdict = zdict_or_none(arg(2))?;
    make_decompress_for(pyre_object::gc_roots::shadow_stack_get(base), wbits, zdict)
}

fn make_decompress(
    wbits: i32,
    zdict: Option<Vec<u8>>,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    make_decompress_for(decompress_type(), wbits, zdict)
}

fn make_decompress_for(
    cls: PyObjectRef,
    wbits: i32,
    zdict: Option<Vec<u8>>,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if !(wbits == 0
        || (-15..=-8).contains(&wbits)
        || (8..=15).contains(&wbits)
        || (24..=31).contains(&wbits)
        || (40..=47).contains(&wbits))
    {
        // interp_zlib.py:411-414 mirrors the deflate constructor's ValueError
        // conversion for inflateInit2.
        return Err(pyre_interpreter::PyError::value_error(
            "Invalid initialization option",
        ));
    }
    let d = backend::Decompressor::new(wbits, zdict).map_err(init_error)?;
    allocate_decompress(cls, d)
}

/// `interp_zlib.py Decompress.copy`: clone both inflate state and its
/// visible tails/dictionary under the per-object lock, returning the base type.
fn decompress_copy(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let obj = args.first().copied().unwrap_or(PY_NULL);
    let copied = {
        let this = decompressor_this(obj)?;
        if this.backend.is_null() {
            return Err(pyre_interpreter::PyError::value_error(
                "Decompressor was already flushed",
            ));
        }
        let decompressor = unsafe { &*this.backend }.lock();
        if decompressor.is_finished() {
            return Err(pyre_interpreter::PyError::value_error(
                "Decompressor was already flushed",
            ));
        }
        decompressor.copy().map_err(zlib_error)?
    };
    allocate_decompress(decompress_type(), copied)
}

fn allocate_decompress(
    cls: PyObjectRef,
    backend: backend::Decompressor,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let backend = Box::into_raw(Box::new(Mutex::new(backend)));
    let _roots = gc_roots::push_roots();
    let _ = gc_roots::pin_root(cls);
    let cls_slot = gc_roots::shadow_stack_len() - 1;
    Ok(W_Decompress::allocate_instance(
        W_Decompress {
            ob: PyObject::default(),
            backend,
        },
        gc_roots::shadow_stack_get(cls_slot),
    ))
}

// ── _ZlibDecompressor (buffered; used by gzip reading) ──────────────────

fn zdecompress_type() -> PyObjectRef {
    ZDECOMPRESS_RUNTIME_TYPE.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type_with_layout(
            "_ZlibDecompressor",
            init_zdecompress_type,
            pyre_interpreter::typedef::w_object(),
            <W_ZlibDecompressor as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        // `zlib_exec` creates ZlibDecompressorType from an immutable heap
        // spec, unlike Compress and Decompress.
        pyre_interpreter::typedef::mark_cpython_heap_type(tp, true);
        // [3.14-spec] CPython's `ZlibDecompressorType` omits BASETYPE.  PyPy
        // has no separate `_ZlibDecompressor` owner; keep this existing
        // compatibility payload isolated and suppress its public base
        // capability without changing object's TypeDef.
        unsafe { pyre_object::w_type_suppress_cpython_basetype(tp) };
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_ZlibDecompressor as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        tp
    })
}

fn zdecompress_getset(
    ns: PyObjectRef,
    name: &'static str,
    f: pyre_interpreter::gateway::BuiltinCodeFn,
) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        name,
        pyre_interpreter::typedef::make_getset_descriptor_named(
            pyre_interpreter::make_builtin_function_with_arity(name, f, 2),
            name,
        )
    );
}

// _ZlibDecompressor(wbits=MAX_WBITS, zdict=b'') — the DecompressReader factory
// gzip calls with wbits=-MAX_WBITS.  `cls` positional-only, `wbits`/`zdict`
// positional-or-keyword; the Signature-bound call path fills omitted optionals
// with PY_NULL.
fn zdecompress_new(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    // args[0] is the type; args[1..] are the constructor arguments.
    let mut w_cls = args.first().copied().unwrap_or(PY_NULL);
    let w_wbits = args.get(1).copied().unwrap_or(PY_NULL);
    let mut w_zdict = args.get(2).copied().unwrap_or(PY_NULL);
    let wbits = to_wbits(pyre_object::with_roots!(w_cls, w_zdict => int_or_default(
        w_wbits,
        backend::MAX_WBITS as i64,
    ))?);
    let zdict = pyre_object::with_roots!(w_cls => zdict_or_none(w_zdict))?;
    let d = backend::ZlibDecompressor::new(wbits, zdict).map_err(init_error)?;
    let cls = if w_cls.is_null() {
        zdecompress_type()
    } else {
        w_cls
    };
    allocate_zdecompress(cls, d)
}

fn allocate_zdecompress(
    cls: PyObjectRef,
    backend: backend::ZlibDecompressor,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let backend = Box::into_raw(Box::new(Mutex::new(backend)));
    let _roots = gc_roots::push_roots();
    let _ = gc_roots::pin_root(cls);
    let cls_slot = gc_roots::shadow_stack_len() - 1;
    let obj = W_ZlibDecompressor::allocate_stable(W_ZlibDecompressor {
        ob: PyObject::default(),
        backend,
    });
    Ok(pyre_interpreter::typedef::tag_subclass_instance(
        obj,
        gc_roots::shadow_stack_get(cls_slot),
    ))
}

// `_ZlibDecompressor.decompress(self, /, data, max_length=-1)` — `self`
// positional-only, `data` positional-or-keyword.
fn zdecompress_decompress(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let data_obj = args.get(1).copied().unwrap_or(PY_NULL);
    if data_obj.is_null() {
        return Err(pyre_interpreter::PyError::type_error(
            "decompress() missing data",
        ));
    }
    let mut w_self = args[0];
    let mut w_max_length = args.get(2).copied().unwrap_or(PY_NULL);
    let data = pyre_object::with_roots!(w_self, w_max_length => as_bytes(data_obj))?;
    // `max_length=-1` (the default) means unlimited; an omitted slot behaves
    // the same.  Only PY_NULL selects that default: a supplied value —
    // including `None` — goes through `int_w`, which raises for `None` and on
    // ssize_t overflow.  A negative value is unlimited, zero caps the output
    // at zero bytes.
    let max_length = if w_max_length.is_null() {
        None
    } else {
        let v = pyre_object::with_roots!(w_self => pyre_interpreter::baseobjspace::int_w(w_max_length))?;
        (v >= 0).then_some(v as usize)
    };
    let this = zdecompressor_this(w_self)?;
    if this.backend.is_null() {
        return Err(zlib_error("Error -2: inconsistent stream state"));
    }
    let mut d = unsafe { &*this.backend }.lock();
    match d.decompress(&data, max_length) {
        Ok(out) => Ok(bytesobject::w_bytes_from_bytes(&out)),
        Err(backend::DecompressError::Zlib(m)) => Err(zlib_error(m)),
        Err(backend::DecompressError::Eof) => Err(eof_error("End of stream already reached")),
    }
}

fn init_zdecompress_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    publish_cpython_module(pyre_object::gc_roots::shadow_stack_get(ns_slot));
    let new_sig = {
        let mut b = pyre_interpreter::SignatureBuilder::default();
        b.append("cls");
        b.marker_posonly();
        b.append("wbits");
        b.append("zdict");
        b.signature()
    };
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "__new__",
        pyre_interpreter::typedef::make_new_descr_maybe_sig(zdecompress_new, Some(new_sig))
    );
    for name in ["__reduce__", "__reduce_ex__"] {
        pyre_interpreter::__pyre_put_new!(
            ns_slot,
            name,
            pyre_interpreter::make_builtin_function(name, |_| {
                Err(pyre_interpreter::PyError::type_error(
                    "cannot pickle 'zlib._ZlibDecompressor' object",
                ))
            })
        );
    }
    let decompress_sig = {
        let mut b = pyre_interpreter::SignatureBuilder::default();
        b.append("self");
        b.marker_posonly();
        b.append("data");
        b.append("max_length");
        b.signature()
    };
    pyre_interpreter::__pyre_put_new!(
        ns_slot,
        "decompress",
        pyre_interpreter::make_builtin_function_maybe_sig(
            "decompress",
            zdecompress_decompress,
            Some(decompress_sig),
        )
    );
    zdecompress_getset(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "unused_data",
        |args| {
            let this = zdecompressor_this(args.get(1).copied().unwrap_or(PY_NULL))?;
            let data = if this.backend.is_null() {
                Vec::new()
            } else {
                unsafe { &*this.backend }.lock().unused_data().to_vec()
            };
            Ok(bytesobject::w_bytes_from_bytes(&data))
        },
    );
    zdecompress_getset(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "eof",
        |args| {
            let this = zdecompressor_this(args.get(1).copied().unwrap_or(PY_NULL))?;
            Ok(w_bool_from(
                !this.backend.is_null() && unsafe { &*this.backend }.lock().eof(),
            ))
        },
    );
    zdecompress_getset(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "needs_input",
        |args| {
            let this = zdecompressor_this(args.get(1).copied().unwrap_or(PY_NULL))?;
            Ok(w_bool_from(
                this.backend.is_null() || unsafe { &*this.backend }.lock().needs_input(),
            ))
        },
    );
}

pyre_interpreter::py_module! {
    "zlib",
    interpleveldefs: {
        "ZLIB_VERSION" => w_str_new("1.3.1"),
        "ZLIB_RUNTIME_VERSION" => w_str_new("1.3.1"),
        "_ZlibDecompressor" => zdecompress_type(),
    },
    int_constants: {
        "DEFLATED" => backend::Z_DEFLATED,
        "MAX_WBITS" => backend::MAX_WBITS,
        "DEF_MEM_LEVEL" => backend::DEF_MEM_LEVEL,
        "DEF_BUF_SIZE" => backend::DEF_BUF_SIZE as i64,
        "Z_DEFAULT_COMPRESSION" => backend::Z_DEFAULT_COMPRESSION,
        "Z_NO_COMPRESSION" => backend::Z_NO_COMPRESSION,
        "Z_BEST_SPEED" => backend::Z_BEST_SPEED,
        "Z_BEST_COMPRESSION" => backend::Z_BEST_COMPRESSION,
        "Z_DEFAULT_STRATEGY" => backend::Z_DEFAULT_STRATEGY,
        "Z_FILTERED" => backend::Z_FILTERED,
        "Z_HUFFMAN_ONLY" => backend::Z_HUFFMAN_ONLY,
        "Z_RLE" => backend::Z_RLE,
        "Z_FIXED" => backend::Z_FIXED,
        "Z_NO_FLUSH" => backend::Z_NO_FLUSH,
        "Z_PARTIAL_FLUSH" => backend::Z_PARTIAL_FLUSH,
        "Z_SYNC_FLUSH" => backend::Z_SYNC_FLUSH,
        "Z_FULL_FLUSH" => backend::Z_FULL_FLUSH,
        "Z_FINISH" => backend::Z_FINISH,
        "Z_BLOCK" => backend::Z_BLOCK,
        "Z_TREES" => backend::Z_TREES,
    },
    exceptions: {
        "error" => pyre_interpreter::builtins::lookup_exc_class("Exception").expect("Exception installed"),
    },
    inline_functions: {
        // interp_zlib.py `compress(data, __posonly__=None, level, wbits)` —
        // `data` positional-only, `level`/`wbits` positional-or-keyword.
        fn compress(
            data: PyBufferStr,
            #[posonly]
            #[default(w_none())]
            level: PyObjectRef,
            #[default(w_none())]
            mut wbits: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let level = pyre_object::with_roots!(wbits => int_or_default(level, -1))? as i32;
            let wbits = to_wbits(int_or_default(wbits, backend::MAX_WBITS as i64)?);
            let out = backend::compress(&data, level, wbits)
                .map_err(|e| oneshot_init_error(e, "Bad compression level"))?;
            Ok(bytesobject::w_bytes_from_bytes(&out))
        }
        // interp_zlib.py `decompress(string, __posonly__=None, wbits, bufsize)`.
        fn decompress(
            data: PyBufferStr,
            #[posonly]
            #[default(w_none())]
            wbits: PyObjectRef,
            #[default(w_none())]
            mut bufsize: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let wbits = to_wbits(pyre_object::with_roots!(bufsize => int_or_default(wbits, backend::MAX_WBITS as i64))?);
            let bufsize = int_or_default(bufsize, backend::DEF_BUF_SIZE as i64)?;
            if bufsize < 0 {
                return Err(pyre_interpreter::PyError::value_error("bufsize must be non-negative"));
            }
            let out = backend::decompress(&data, wbits, bufsize as usize)
                .map_err(|e| oneshot_init_error(e, "Bad window buffer size"))?;
            Ok(bytesobject::w_bytes_from_bytes(&out))
        }
        // interp_zlib.py `Compress___new__(level, method, wbits, memLevel,
        // strategy, w_zdict)` — all six positional-or-keyword and all native
        // initialization options are passed through to deflateInit2.
        fn compressobj(
            #[default(w_none())] level: PyObjectRef,
            #[default(w_none())] mut method: PyObjectRef,
            #[default(w_none())] mut wbits: PyObjectRef,
            #[default(w_none())] mut memLevel: PyObjectRef,
            #[default(w_none())] mut strategy: PyObjectRef,
            #[default(w_none())] mut zdict: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let level = pyre_object::with_roots!(memLevel, method, strategy, wbits, zdict => int_or_default(level, -1))? as i32;
            let method = pyre_object::with_roots!(memLevel, strategy, wbits, zdict => int_or_default(method, 8))? as i32;
            let wbits = to_wbits(pyre_object::with_roots!(memLevel, strategy, zdict => int_or_default(wbits, backend::MAX_WBITS as i64))?);
            let mem_level = pyre_object::with_roots!(strategy, zdict => int_or_default(memLevel, 8))? as i32;
            let strategy = pyre_object::with_roots!(zdict => int_or_default(strategy, 0))? as i32;
            make_compress(
                level,
                method,
                wbits,
                mem_level,
                strategy,
                zdict_or_none(zdict)?,
            )
        }
        // interp_zlib.py `Decompress___new__(wbits, w_zdict)`.
        fn decompressobj(
            #[default(w_none())] wbits: PyObjectRef,
            #[default(w_none())] mut zdict: PyObjectRef,
        ) -> Result<PyObjectRef, pyre_interpreter::PyError> {
            let wbits = to_wbits(pyre_object::with_roots!(zdict => int_or_default(wbits, backend::MAX_WBITS as i64))?);
            make_decompress(wbits, zdict_or_none(zdict)?)
        }
    },
    functions: {
        "crc32" / * = |args| {
            let w_data = args.first().copied().unwrap_or(w_none());
            let mut w_start = args.get(1).copied().unwrap_or(PY_NULL);
            let data = pyre_object::with_roots!(w_start => as_bytes(w_data))?;
            let start = if w_start.is_null() { 0 } else { unsafe { w_int_get_value(w_start) as u32 } };
            Ok(w_int_new(crc32_compute(&data, start) as i64))
        },
        "adler32" / * = |args| {
            let w_data = args.first().copied().unwrap_or(w_none());
            let mut w_start = args.get(1).copied().unwrap_or(PY_NULL);
            let data = pyre_object::with_roots!(w_start => as_bytes(w_data))?;
            let start = if w_start.is_null() { 1 } else { unsafe { w_int_get_value(w_start) as u32 } };
            Ok(w_int_new(adler32_compute(&data, start) as i64))
        },
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_type_owner_flags_and_module_match_cpython_314() {
        pyre_interpreter::typedef::init_typeobjects();
        const IMMUTABLETYPE: i64 =
            pyre_object::typeobject::TpFlags::PY_TPFLAGS_IMMUTABLETYPE.as_int();
        const HEAPTYPE: i64 = pyre_object::typeobject::TpFlags::PY_TPFLAGS_HEAPTYPE.as_int();
        const MASK: i64 = IMMUTABLETYPE | HEAPTYPE;
        for (name, ty, expected) in [
            ("Compress", compress_type(), HEAPTYPE),
            ("Decompress", decompress_type(), HEAPTYPE),
            (
                "_ZlibDecompressor",
                zdecompress_type(),
                HEAPTYPE | IMMUTABLETYPE,
            ),
        ] {
            assert_eq!(
                unsafe { pyre_object::w_type_get_flags(ty) } & MASK,
                expected
            );
            assert_eq!(
                unsafe {
                    pyre_object::w_str_get_wtf8(
                        pyre_interpreter::baseobjspace::getattr_str(ty, "__module__").unwrap(),
                    )
                },
                "zlib",
                "{name}.__module__"
            );
            assert!(!unsafe { pyre_object::w_type_is_heaptype(ty) });
        }
    }
}

/// The GC types this module owns, in `build_gc` registration order.
pub(crate) fn gc_types(types: &mut Vec<pyre_interpreter::importing::ModuleGcType>) {
    use pyre_interpreter::importing::{ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    // `interp_zlib.py` stores each rzlib stream and its lock directly on the
    // corresponding W_Root owner. `Compress` and `Decompress` are acceptable
    // bases: `typedef.py` `_getusercls` puts `MapdictStorageMixin` on the user
    // layout, and the sweep destructor releases the native stream. The
    // buffered decompressor is not a base and has no Python reference.
    let pyre_class = ModuleGcLayout::PyreClass {
        memory_pressure_offset: None,
    };
    types.push(ModuleGcType {
        descriptor: <W_Compress as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: pyre_class,
        destructor: Some(gc_destructor!(w_compress_dealloc)),
    });
    types.push(ModuleGcType {
        descriptor: &W_COMPRESS_USER_PYRE_CLASS_DESCRIPTOR,
        layout: pyre_class,
        destructor: Some(gc_destructor!(w_compress_dealloc)),
    });
    types.push(ModuleGcType {
        descriptor: <W_Decompress as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: pyre_class,
        destructor: Some(gc_destructor!(w_decompress_dealloc)),
    });
    types.push(ModuleGcType {
        descriptor: &W_DECOMPRESS_USER_PYRE_CLASS_DESCRIPTOR,
        layout: pyre_class,
        destructor: Some(gc_destructor!(w_decompress_dealloc)),
    });
    types.push(ModuleGcType {
        descriptor: <W_ZlibDecompressor as PyreClassPyTypeOf>::DESCRIPTOR,
        layout: pyre_class,
        destructor: Some(gc_destructor!(w_zdecompress_dealloc)),
    });
}
