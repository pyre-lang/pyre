//! `marshal` — PyPy `pypy/module/marshal/` object glue over the shared
//! RustPython compiler-core marshal format.
//!
//! PyPy's `interp_marshal.py` owns the Python object dispatch while
//! `marshal_impl.py` defines the wire typecodes.  Pyre follows that split:
//! this module maps `W_Root` objects to the shared wire implementation, and
//! compiler-core serializes/deserializes the authoritative `CodeObject`.

use num_complex::Complex64;
use pyre_object::*;
use rustpython_compiler_core::bytecode::{BasicBag, CodeObject, ConstantBag, ConstantData};
use rustpython_compiler_core::marshal::{self as wire, DumpableValue, Write};
use rustpython_wtf8::Wtf8;

use crate::{PyError, PyResult};

const MAX_DEPTH: usize = wire::MAX_MARSHAL_STACK_DEPTH;

fn marshal_error(error: wire::MarshalError) -> PyError {
    match error {
        // `r_string`'s buffer path, where fewer bytes are left than the run
        // being read is long.
        wire::MarshalError::DataTooShort => eof_error("marshal data too short"),
        // `r_byte` running out, which is a field's byte rather than the byte an
        // object starts with.
        wire::MarshalError::Eof => eof_error("EOF read where not expected"),
        // `r_object` replaces `r_byte`'s message when it is the type byte that
        // never came.
        wire::MarshalError::EofObject => eof_error("EOF read where object expected"),
        _ => PyError::value_error("bad marshal data"),
    }
}

fn eof_error(message: &str) -> PyError {
    PyError::new(crate::PyErrorKind::EOFError, message)
}

fn call_method(obj: PyObjectRef, name: &str, args: &[PyObjectRef]) -> PyResult {
    let result = crate::baseobjspace::call_method(obj, name, args);
    if result.is_null() {
        Err(crate::call::take_call_error()
            .unwrap_or_else(|| PyError::runtime_error("method call failed")))
    } else {
        Ok(result)
    }
}

enum MarshalBytes {
    Borrowed(&'static [u8]),
    Owned(Vec<u8>),
}

impl MarshalBytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Owned(bytes) => bytes,
        }
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

fn bytes_like(obj: PyObjectRef, function: &str) -> Result<MarshalBytes, PyError> {
    if unsafe { bytesobject::is_bytes_like(obj) } {
        return Ok(MarshalBytes::Borrowed(unsafe {
            bytesobject::bytes_like_data(obj)
        }));
    }
    if unsafe { memoryview::is_w_memoryview(obj) } {
        unsafe { crate::builtins::memoryview_check_released(obj) }?;
        crate::typedef::require_contiguous_buffer(obj)?;
        let view = unsafe { memoryview::w_memoryview_view(obj) };
        return Ok(match unsafe { view.as_contiguous_bytes() } {
            Some(bytes) => MarshalBytes::Borrowed(bytes),
            None => MarshalBytes::Owned(unsafe { view.gather() }),
        });
    }
    // Any readable buffer is accepted (`interp_marshal` unwraps via
    // `space.readbuf_w`): `SourcelessFileLoader.get_code` hands `loads` a
    // sliced memoryview of the pyc payload.
    if let Some(src) = crate::typedef::buffer_as_bytes_like(obj)? {
        return Ok(MarshalBytes::Owned(
            unsafe { bytesobject::bytes_like_data(src) }.to_vec(),
        ));
    }
    Err(PyError::type_error(format!(
        "{function}() argument must be a bytes-like object"
    )))
}

/// PyPy `marshal_impl.py marshal` buffer fallback.
///
/// Heap types bypass all builtin marshallers, then any object satisfying a
/// simple contiguous buffer request is written as `TYPE_STRING`.  Keep the
/// acquired bytes in a Rust `Vec`: unlike materialising a Python `bytes`, this
/// cannot move objects while `write_object` holds raw traversal locals.
fn marshal_buffer_bytes(obj: PyObjectRef) -> Result<Option<Vec<u8>>, PyError> {
    if let Some(target) = crate::module::__pypy__::interp_buffer::forwarded_exporter(obj) {
        return marshal_buffer_bytes(target?);
    }
    unsafe {
        if bytesobject::is_bytes_like(obj) {
            return Ok(Some(bytesobject::bytes_like_data(obj).to_vec()));
        }
        if interp_array::is_array(obj) {
            return Ok(Some(interp_array::w_array_bytes(obj).to_vec()));
        }
        if memoryview::is_w_memoryview(obj) {
            crate::builtins::memoryview_check_released(obj)?;
            crate::typedef::require_contiguous_buffer(obj)?;
            return Ok(Some(crate::builtins::memoryview_gather_bytes(obj)));
        }
    }
    Ok(None)
}

/// Transient equivalent of PyPy's `Marshaller.all_refs` dict.  One entry is
/// reserved for every non-singleton object written, not only for the shared
/// ones, so the table reaches thousands of entries on a module-sized code
/// tree.  `entries` holds a shadow-stack slot per object, which a collection
/// forwards; `by_identity` reproduces the O(1) lookup of the RPython dict.
struct WriterRefs {
    entries: Vec<WriterRefEntry>,
    /// `gc_identity_hash` survives a move while a raw address does not, so it
    /// keys the buckets; the entry inside a bucket is confirmed against the
    /// forwarded shadow-stack address.
    by_identity: std::collections::HashMap<usize, Vec<u32>>,
}

#[derive(Clone, Copy)]
struct WriterRefEntry {
    slot: usize,
    incomplete: bool,
}

impl WriterRefs {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            by_identity: std::collections::HashMap::new(),
        }
    }

    fn find(&self, obj: PyObjectRef) -> Option<(u32, bool)> {
        let obj =
            pyre_object::gc_hook::try_gc_current_object_address(obj as pyre_object::gc_hook::GCREF)
                as PyObjectRef;
        let bucket = self
            .by_identity
            .get(&pyre_object::gc_hook::gc_identity_hash(obj as usize))?;
        bucket
            .iter()
            .copied()
            .find(|&index| {
                std::ptr::eq(
                    pyre_object::gc_roots::shadow_stack_get(self.entries[index as usize].slot),
                    obj,
                )
            })
            .map(|index| (index, self.entries[index as usize].incomplete))
    }

    fn reserve(&mut self, obj: PyObjectRef, incomplete: bool) -> Result<u32, PyError> {
        if self.entries.len() >= i32::MAX as usize {
            return Err(PyError::value_error("too many objects to marshal"));
        }
        let index = self.entries.len() as u32;
        let identity = pyre_object::gc_hook::gc_identity_hash(obj as usize);
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        self.entries.push(WriterRefEntry { slot, incomplete });
        self.by_identity.entry(identity).or_default().push(index);
        Ok(index)
    }

    fn complete(&mut self, index: u32) {
        let entry = &mut self.entries[index as usize];
        debug_assert!(entry.incomplete);
        entry.incomplete = false;
    }
}

fn is_singleton(obj: PyObjectRef) -> bool {
    (unsafe { is_none(obj) || is_bool(obj) || is_ellipsis(obj) })
        || crate::builtins::lookup_exc_class("StopIteration") == Some(obj)
}

fn write_len(out: &mut Vec<u8>, len: usize) -> Result<(), PyError> {
    out.write_u32(
        len.try_into()
            .map_err(|_| PyError::value_error("object too large to marshal"))?,
    );
    Ok(())
}

fn write_bytes(out: &mut Vec<u8>, value: &[u8]) -> Result<(), PyError> {
    out.write_u8(b's');
    write_len(out, value.len())?;
    out.write_slice(value);
    Ok(())
}

fn write_marshal_str(out: &mut Vec<u8>, value: &str, version: i32) -> Result<(), PyError> {
    write_marshal_wtf8(out, value.as_bytes(), version)
}

fn write_marshal_wtf8(out: &mut Vec<u8>, bytes: &[u8], version: i32) -> Result<(), PyError> {
    if version >= 4 && bytes.len() < 256 && bytes.is_ascii() {
        out.write_u8(b'z');
        out.write_u8(bytes.len() as u8);
    } else {
        out.write_u8(b'u');
        write_len(out, bytes.len())?;
    }
    out.write_slice(bytes);
    Ok(())
}

fn write_name_tuple<N: AsRef<str>>(
    out: &mut Vec<u8>,
    names: &[N],
    version: i32,
) -> Result<(), PyError> {
    out.write_u8(b'(');
    write_len(out, names.len())?;
    for name in names {
        write_marshal_str(out, name.as_ref(), version)?;
    }
    Ok(())
}

/// CPython 3.14 `w_code` / PyPy `marshal_pycode`, in compiler-core's 3.14
/// field order.  The one deliberate boundary from `wire::serialize_code` is
/// co_consts: PyPy writes `x.co_consts_w`, so recurse through the wrapped
/// objects owned by this exact PyCode instead of the compiler backing enum.
unsafe fn write_code(
    out: &mut Vec<u8>,
    w_code: PyObjectRef,
    refs: &mut Option<WriterRefs>,
    version: i32,
    depth: usize,
) -> Result<(), PyError> {
    let code_root = Rooted::new(w_code);
    let code =
        unsafe { &*(crate::pycode::w_code_get_ptr(code_root.get()) as *const crate::CodeObject) };
    out.write_u32(code.arg_count);
    out.write_u32(code.posonlyarg_count);
    out.write_u32(code.kwonlyarg_count);
    out.write_u32(code.max_stackdepth);
    out.write_u32(code.flags.bits());

    write_bytes(out, &unsafe { crate::pycode::code_bytes(code_root.get()) })?;

    out.write_u8(b'(');
    write_len(out, code.constants.len())?;
    // Realizing one constant may collect and move the wrapper. Keep a shadow
    // slot and reload it just like PyPy keeps `x` live throughout w_code.
    let constant_count = code.constants.len();
    for index in 0..constant_count {
        let constant = crate::pycode::w_code_const(code_root.get(), index);
        if constant.is_null() {
            return Err(PyError::value_error("unmarshallable object"));
        }
        write_object(out, constant, refs, version, depth - 1)?;
    }

    // A wrapped constant may have allocated and moved `w_code`; recover the
    // backing payload from the forwarded wrapper before touching later fields.
    let code =
        unsafe { &*(crate::pycode::w_code_get_ptr(code_root.get()) as *const crate::CodeObject) };
    write_name_tuple(out, &code.names, version)?;

    let cell_only_names: Vec<&str> = code
        .cellvars
        .iter()
        .filter(|cell| !code.varnames.iter().any(|local| local == *cell))
        .map(|cell| cell.as_ref())
        .collect();
    out.write_u8(b'(');
    write_len(
        out,
        code.varnames.len() + cell_only_names.len() + code.freevars.len(),
    )?;
    for name in &code.varnames {
        write_marshal_str(out, name.as_ref(), version)?;
    }
    for name in cell_only_names {
        write_marshal_str(out, name, version)?;
    }
    for name in &code.freevars {
        write_marshal_str(out, name.as_ref(), version)?;
    }

    write_bytes(out, &code.localspluskinds)?;
    write_marshal_str(out, code.source_path.as_ref(), version)?;
    // `marshal_pycode` writes `co_name` / `co_qualname` as the str objects
    // (`w_code_name_obj`), not the compiler `String` mirror.
    let w_name = Rooted::new(unsafe { crate::pycode::w_code_name_obj(code_root.get()) });
    let w_qualname = Rooted::new(unsafe { crate::pycode::w_code_qualname_obj(code_root.get()) });
    write_marshal_wtf8(
        out,
        unsafe { pyre_object::w_str_get_wtf8(w_name.get()) }.as_bytes(),
        version,
    )?;
    write_marshal_wtf8(
        out,
        unsafe { pyre_object::w_str_get_wtf8(w_qualname.get()) }.as_bytes(),
        version,
    )?;
    out.write_u32(
        unsafe { (*(code_root.get() as *const crate::pycode::PyCode)).co_firstlineno_raw } as u32,
    );
    write_bytes(out, &code.linetable)?;
    write_bytes(out, &code.exceptiontable)?;
    Ok(())
}

/// `w_float_str`: the decimal spelling versions 0 and 1 carry a float in.
/// `PyOS_double_to_string(v, 'g', 17, 0, NULL)` rendered through
/// `w_short_pstring`, i.e. a one-byte length and that many bytes.  The
/// infinities and NaN name themselves, and a negative NaN loses its sign.
fn w_float_str(out: &mut Vec<u8>, value: f64) {
    let text = rustpython_common::format::FormatSpec::parse(".17g")
        .and_then(|spec| spec.format_float(value))
        .expect("'.17g' is a valid float format spec");
    out.write_u8(text.len() as u8);
    out.write_slice(text.as_bytes());
}

fn write_object(
    out: &mut Vec<u8>,
    obj: PyObjectRef,
    refs: &mut Option<WriterRefs>,
    version: i32,
    depth: usize,
) -> Result<(), PyError> {
    if depth == 0 {
        return Err(PyError::value_error("object too deeply nested to marshal"));
    }
    // Recursive writes can realize code constants and run the collector. Keep
    // every current object rooted even for marshal versions without a ref
    // table, then reload containers before each child fetch.
    let obj_root = Rooted::new(obj);
    // PyPy `marshal_impl._marshal_unicode` asks
    // `ObjSpace.get_interned_str` before `write_ref`.  An AsciiListStrategy
    // materializes a fresh wrapper on each getitem, so keying the reference
    // table by that transient wrapper loses sharing for an interned value.
    // Canonicalize only values already present in the intern table; ordinary
    // strings must not become interned merely because they are marshalled.
    let canonical_str_root = if version >= 3
        && unsafe { is_exact_type(obj_root.get(), &pyre_object::STR_TYPE) }
    {
        unicodeobject::get_interned_wtf8(unsafe { w_str_get_wtf8(obj_root.get()) }).map(Rooted::new)
    } else {
        None
    };
    let obj_root = canonical_str_root.unwrap_or(obj_root);
    let obj = obj_root.get();

    if !is_singleton(obj)
        && let Some(table) = refs.as_ref()
        && let Some((index, incomplete)) = table.find(obj)
    {
        // CPython 3.14 `w_ref`: code and slice entries retain the high-bit
        // incomplete marker until `w_complete`, because their immutable
        // representation cannot be reconstructed from a recursive reference.
        if incomplete {
            return Err(PyError::value_error(format!(
                "cannot marshal recursion {} objects",
                crate::error::type_name_of(obj)
            )));
        }
        out.write_u8(b'r');
        out.write_u32(index);
        return Ok(());
    }

    let type_pos = out.len();
    let use_ref = refs.is_some() && !is_singleton(obj);
    // CPython 3.14 `w_ref` marks only code and slice objects incomplete and
    // clears that marker in `w_complete` after their contents are written.
    let requires_completion = unsafe { crate::pycode::is_code(obj) || is_slice(obj) };
    let ref_index = if use_ref {
        Some(refs.as_mut().unwrap().reserve(obj, requires_completion)?)
    } else {
        None
    };

    // PyPy `marshal`: instances of user heap types skip the builtin
    // marshaller table completely.  They may still reach the buffer fallback
    // below (notably bytes/bytearray subclasses).
    // `reserve` pins `obj`. The type test reads the word that pin published.
    let is_heap_type = crate::typedef::r#type(obj_root.get())
        .is_some_and(|w_type| unsafe { typeobject::w_type_is_heaptype(w_type.as_ptr()) });

    unsafe {
        let obj = obj_root.get();
        if !is_heap_type && is_none(obj) {
            out.write_u8(b'N');
        } else if !is_heap_type && crate::builtins::lookup_exc_class("StopIteration") == Some(obj) {
            out.write_u8(b'S');
        } else if !is_heap_type && is_bool(obj) {
            out.write_u8(if w_bool_get_value(obj) { b'T' } else { b'F' });
        } else if !is_heap_type && is_ellipsis(obj) {
            out.write_u8(b'.');
        } else if !is_heap_type && is_int_or_long(obj) {
            let value = crate::builtins::obj_to_bigint(obj);
            let value = crate::rbigint_to_compiler_bigint(&value);
            wire::serialize_value::<_, ConstantData>(out, DumpableValue::Integer(&value))
                .unwrap_or_else(|never| match never {});
        } else if !is_heap_type && is_float(obj) {
            let value = w_float_get_value(obj);
            if version > 1 {
                out.write_u8(b'g');
                out.write_u64(value.to_bits());
            } else {
                out.write_u8(b'f');
                w_float_str(out, value);
            }
        } else if !is_heap_type && is_complex(obj) {
            let real = w_complex_get_real(obj);
            let imag = w_complex_get_imag(obj);
            if version > 1 {
                out.write_u8(b'y');
                out.write_u64(real.to_bits());
                out.write_u64(imag.to_bits());
            } else {
                out.write_u8(b'x');
                w_float_str(out, real);
                w_float_str(out, imag);
            }
        } else if !is_heap_type && is_str(obj) {
            let value = w_str_get_wtf8(obj).as_bytes();
            let interned = unicodeobject::is_interned_exact_str(obj);
            let ascii = value.iter().all(u8::is_ascii);
            if version >= 4 && ascii && value.len() < 256 {
                out.write_u8(if interned { b'Z' } else { b'z' });
                out.write_u8(value.len() as u8);
            } else {
                out.write_u8(if version >= 4 && ascii {
                    if interned { b'A' } else { b'a' }
                } else if version >= 3 && interned {
                    b't'
                } else {
                    b'u'
                });
                out.write_u32(
                    value
                        .len()
                        .try_into()
                        .map_err(|_| PyError::value_error("object too large to marshal"))?,
                );
            }
            out.write_slice(value);
        } else if let Some(value) = marshal_buffer_bytes(obj)? {
            out.write_u8(b's');
            out.write_u32(
                value
                    .len()
                    .try_into()
                    .map_err(|_| PyError::value_error("object too large to marshal"))?,
            );
            out.write_slice(&value);
        } else if !is_heap_type && is_tuple(obj) {
            let len = w_tuple_len(obj);
            out.write_u8(if version >= 4 && len < 256 {
                b')'
            } else {
                b'('
            });
            if version >= 4 && len < 256 {
                out.write_u8(len as u8);
            } else {
                out.write_u32(
                    len.try_into()
                        .map_err(|_| PyError::value_error("object too large to marshal"))?,
                );
            }
            for index in 0..len {
                let item = w_tuple_getitem(obj_root.get(), index as i64)
                    .ok_or_else(|| PyError::value_error("unmarshallable object"))?;
                write_object(out, item, refs, version, depth - 1)?;
            }
        } else if !is_heap_type && is_list(obj) {
            // PyPy marshal_list snapshots `w_list.getitems()[:]` before
            // writing. Besides insulating traversal from mutation, this
            // preserves AbstractUnwrappedStrategy.getitems_copy's wrapper
            // reuse for consecutive identical unboxed storage entries.
            // Rooted uses the marshal call's outer root bracket. A nested
            // RootScope here would discard WriterRefs' longer-lived slots
            // when it dropped after recursive writes.
            let snapshot: Vec<_> = w_list_items_copy_as_vec(obj_root.get())
                .into_iter()
                .map(Rooted::new)
                .collect();
            let len = snapshot.len();
            out.write_u8(b'[');
            out.write_u32(
                len.try_into()
                    .map_err(|_| PyError::value_error("object too large to marshal"))?,
            );
            for item in snapshot {
                write_object(out, item.get(), refs, version, depth - 1)?;
            }
        } else if !is_heap_type && is_dict(obj) {
            out.write_u8(b'{');
            let len = dictmultiobject::w_dict_len(obj);
            // A slot walk, not a positional one: a deleted entry leaves a
            // tombstone `_ll_dictnext` steps over.
            let mut cursor = 0;
            for _ in 0..len {
                let (slot, key, value) = dictmultiobject::w_dict_next_item(obj_root.get(), cursor)
                    .ok_or_else(|| PyError::value_error("unmarshallable object"))?;
                cursor = slot + 1;
                let key_root = Rooted::new(key);
                let value_root = Rooted::new(value);
                write_object(out, key_root.get(), refs, version, depth - 1)?;
                write_object(out, value_root.get(), refs, version, depth - 1)?;
            }
            out.write_u8(b'0');
        } else if !is_heap_type && setobject::is_set_or_frozenset(obj) {
            out.write_u8(if setobject::is_frozenset(obj) {
                b'>'
            } else {
                b'<'
            });
            let len = setobject::w_set_len(obj);
            write_len(out, len)?;
            // A slot walk, as the dict arm above: `set_remove_slot` tombstones
            // the slot it empties, so `len` counts the elements but does not
            // bound their slot numbers.
            let mut cursor = 0;
            for _ in 0..len {
                let slot = setobject::w_set_next_slot(obj_root.get(), cursor)
                    .ok_or_else(|| PyError::value_error("unmarshallable object"))?;
                let item = setobject::w_set_key_at(obj_root.get(), slot)
                    .ok_or_else(|| PyError::value_error("unmarshallable object"))?
                    .obj;
                cursor = slot + 1;
                let item_root = Rooted::new(item);
                write_object(out, item_root.get(), refs, version, depth - 1)?;
            }
        } else if !is_heap_type && crate::pycode::is_code(obj) {
            let ptr = crate::pycode::w_code_get_ptr(obj_root.get()) as *const crate::CodeObject;
            if ptr.is_null() {
                return Err(PyError::value_error("unmarshallable object"));
            }
            out.write_u8(b'c');
            write_code(out, obj_root.get(), refs, version, depth)?;
        } else if !is_heap_type && is_slice(obj) && version >= 5 {
            out.write_u8(b':');
            let start = Rooted::new(sliceobject::w_slice_get_start(obj_root.get()));
            let stop = Rooted::new(sliceobject::w_slice_get_stop(obj_root.get()));
            let step = Rooted::new(sliceobject::w_slice_get_step(obj_root.get()));
            write_object(out, start.get(), refs, version, depth - 1)?;
            write_object(out, stop.get(), refs, version, depth - 1)?;
            write_object(out, step.get(), refs, version, depth - 1)?;
        } else {
            return Err(PyError::value_error("unmarshallable object"));
        }
    }

    if use_ref {
        out[type_pos] |= wire::FLAG_REF;
    }
    if requires_completion && let (Some(table), Some(index)) = (refs.as_mut(), ref_index) {
        table.complete(index);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Rooted(usize);

impl Rooted {
    fn new(obj: PyObjectRef) -> Self {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        Self(slot)
    }

    fn get(self) -> PyObjectRef {
        pyre_object::gc_roots::shadow_stack_get(self.0)
    }
}

/// Drain a decode iterator into live object pointers.
///
/// The wire reader yields children lazily (`process_results`): each `next`
/// may allocate and collect. `Rooted` is a shadow-stack index, so collecting
/// the slots first keeps every child pinned; the pointers are read only after
/// the last child exists. Copying `Rooted::get()` while the iterator still
/// runs leaves earlier addresses in an unrooted `Vec` that a later child
/// allocation can evacuate.
fn rooted_ptrs(elements: impl Iterator<Item = Rooted>) -> Vec<PyObjectRef> {
    let slots: Vec<Rooted> = elements.collect();
    slots.iter().copied().map(Rooted::get).collect()
}

/// PyPy `interp_marshal.py FileReader`, with CPython 3.14's
/// `readinto()` validation when that method is available.
///
/// `wire::Read` requires the returned bytes to borrow from the reader, so each
/// exact read is copied into this reusable Rust buffer.  Python exceptions are
/// parked separately and restored by `load_impl` after the wire reader stops.
struct FileReader {
    file: Rooted,
    scratch: Vec<u8>,
    has_readinto: bool,
    errors: ErrorSink,
}

impl FileReader {
    /// Probe `readinto` before the caller parks `pending_error`. The lookup
    /// collects, and an `Option<PyError>` that already exists is live across it.
    fn probe(file: PyObjectRef) -> Result<(Rooted, bool), PyError> {
        let file = Rooted::new(file);
        let has_readinto = crate::baseobjspace::findattr_result(file.get(), "readinto")?.is_some();
        Ok((file, has_readinto))
    }

    fn from_probed(file: Rooted, has_readinto: bool, errors: ErrorSink) -> Self {
        Self {
            file,
            scratch: Vec::new(),
            has_readinto,
            errors,
        }
    }

    fn python_error<T>(&mut self, error: PyError) -> Result<T, wire::MarshalError> {
        self.errors.remember(error);
        Err(wire::MarshalError::BadType)
    }

    fn read_with_read(&mut self, n: usize) -> Result<(), wire::MarshalError> {
        let roots = pyre_object::gc_roots::push_roots();
        // Build and root the argument before reloading the file. Integer
        // construction may collect, so neither raw pointer crosses it.
        let count_slot = roots.base();
        let _ = roots.pin_root(w_int_new(n as i64));
        let result = match call_method(self.file.get(), "read", &[roots.get(count_slot)]) {
            Ok(result) => result,
            Err(error) => return self.python_error(error),
        };
        let result_slot = count_slot + 1;
        let _ = roots.pin_root(result);
        let bytes = match bytes_like(roots.get(result_slot), "load") {
            Ok(bytes) => bytes,
            Err(error) => return self.python_error(error),
        };
        if bytes.len() < n {
            return Err(wire::MarshalError::Eof);
        }
        if bytes.len() > n {
            return self.python_error(PyError::value_error(format!(
                "read() returned too much data: {n} bytes requested, {} returned",
                bytes.len()
            )));
        }
        self.scratch.extend_from_slice(bytes.as_slice());
        Ok(())
    }

    fn read_with_readinto(&mut self, n: usize) -> Result<(), wire::MarshalError> {
        let roots = pyre_object::gc_roots::push_roots();
        let buffer_slot = roots.base();
        let _ = roots.pin_root(bytearrayobject::w_bytearray_new(n));
        let count = match call_method(self.file.get(), "readinto", &[roots.get(buffer_slot)]) {
            Ok(count) => count,
            // AttributeError from inside `readinto` is an application error,
            // not evidence that the method was absent: absence was cached by
            // `new` before any reads began.
            Err(error) => return self.python_error(error),
        };
        // The file protocol reads a byte count through the index protocol, so
        // an object carrying only `__int__` is a TypeError rather than a
        // silently accepted length.
        let count = match crate::baseobjspace::getindex_w(count) {
            Ok(count) => count,
            Err(error) => return self.python_error(error),
        };
        if count < 0 || count as usize > n {
            return self.python_error(PyError::value_error("readinto() returned invalid length"));
        }
        if count as usize != n {
            return Err(wire::MarshalError::Eof);
        }
        self.scratch.extend_from_slice(unsafe {
            bytearrayobject::w_bytearray_data(roots.get(buffer_slot))
        });
        Ok(())
    }
}

/// `read_wtf8`'s validation, done with `rutf8::wtf8_from_bytes`.
///
/// The trait's own body uses `Wtf8::from_bytes`, which accepts three-byte
/// sequences that encode no code point, so a `u`/`a`/`z` payload naming one
/// reaches `w_str_from_wtf8` with a code point count its buffer disagrees with.
/// `r_object` decodes `TYPE_UNICODE` with `surrogatepass`, so surrogates are
/// admitted and a rejection is that call's `UnicodeDecodeError`, reported at
/// the position the validator stopped at, which the sink carries past the
/// wire reader's own error type.
fn strict_wtf8(bytes: &[u8], errors: ErrorSink) -> Result<&Wtf8, wire::MarshalError> {
    pyre_object::rutf8::wtf8_from_bytes(bytes, true).map_err(|error| {
        errors.remember(crate::typedef::utf8_decode_error_from(
            bytes,
            error.pos as usize,
        ));
        wire::MarshalError::InvalidUtf8
    })
}

/// A marshal byte stream, wrapped so `read_wtf8` is the strict one.
struct BytesReader<'a> {
    data: &'a [u8],
    errors: ErrorSink,
}

impl wire::Read for BytesReader<'_> {
    fn read_slice(&mut self, n: u32) -> Result<&[u8], wire::MarshalError> {
        self.data.read_slice(n)
    }

    fn read_wtf8(&mut self, len: u32) -> Result<&Wtf8, wire::MarshalError> {
        let errors = self.errors;
        strict_wtf8(self.read_slice(len)?, errors)
    }
}

impl wire::Read for FileReader {
    fn read_wtf8(&mut self, len: u32) -> Result<&Wtf8, wire::MarshalError> {
        let errors = self.errors;
        strict_wtf8(self.read_slice(len)?, errors)
    }

    fn read_slice(&mut self, n: u32) -> Result<&[u8], wire::MarshalError> {
        // A hostile length prefix must not allocate `n` bytes before the file
        // proves they exist. Fill the reusable result incrementally with a
        // bounded Python buffer (or bounded `read` request) per iteration.
        const CHUNK: usize = 1024 * 1024;
        self.scratch.clear();
        let mut remaining = n as usize;
        while remaining != 0 {
            let chunk = remaining.min(CHUNK);
            if self.has_readinto {
                self.read_with_readinto(chunk)?;
            } else {
                self.read_with_read(chunk)?;
            }
            remaining -= chunk;
        }
        Ok(&self.scratch)
    }
}

/// Parks a Python error until `load_impl` restores it after the wire reader
/// stops. The handle is a movable GC object, so the parked word lives in a
/// shadow-stack slot owned by that outer `RootScope` — an inner `read` /
/// `readinto` bracket would truncate a pin taken at park time.
#[derive(Clone, Copy)]
struct ErrorSink {
    pending: *mut Option<PyError>,
    roots: *const pyre_object::gc_roots::RootScope,
    slot: usize,
}

impl ErrorSink {
    fn park(roots: &pyre_object::gc_roots::RootScope, pending: &mut Option<PyError>) -> Self {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(pyre_object::PY_NULL);
        Self {
            pending,
            roots,
            slot,
        }
    }

    fn remember(self, mut error: PyError) {
        unsafe {
            if (*self.pending).is_none() {
                let roots = &*self.roots;
                roots.set(self.slot, error.as_raw() as pyre_object::PyObjectRef);
                error.reload(roots, self.slot);
                *self.pending = Some(error);
            }
        }
    }

    fn restore(self, wire_error: wire::MarshalError) -> PyError {
        unsafe {
            match (*self.pending).take() {
                Some(mut error) => {
                    error.reload(&*self.roots, self.slot);
                    error
                }
                None => marshal_error(wire_error),
            }
        }
    }
}

#[derive(Clone, Copy)]
struct PyreMarshalBag {
    errors: ErrorSink,
    /// Shadow-stack slots of every `str_from_value` read while decoding one
    /// code object (`co_names`, localsplus names, filename, `co_name`,
    /// `co_qualname`). Nested code objects pop their own run first.
    names: *mut Vec<usize>,
}

impl PyreMarshalBag {
    fn new(errors: ErrorSink, names: &mut Vec<usize>) -> Self {
        Self { errors, names }
    }

    fn remember_python_error(&self, error: PyError) -> wire::MarshalError {
        // The pointer names the stack-owned slot of the synchronous loads/load
        // call.  `MarshalBag: Copy` copies only this borrow; no state escapes
        // the deserialize call and no TLS side channel is involved.
        self.errors.remember(error);
        wire::MarshalError::BadType
    }

    /// Room for a container the decoder publishes before it reads what goes
    /// in it.  RustPython's runtime bag asks for this room explicitly because
    /// the length comes from untrusted marshal input; mirror that shape so an
    /// impossible reservation becomes MemoryError rather than an allocator
    /// abort.
    fn placeholder_elements(&self, len: usize) -> Result<Vec<PyObjectRef>, wire::MarshalError> {
        let mut elements = Vec::new();
        elements
            .try_reserve_exact(len)
            .map_err(|_| self.remember_python_error(PyError::memory_error("")))?;
        elements.resize(len, PY_NULL);
        Ok(elements)
    }

    fn make_runtime_code(
        &self,
        code: CodeObject<ConstantData>,
        constants: Vec<Rooted>,
        raw_code_bytes: Option<Vec<u8>>,
    ) -> Result<Rooted, wire::MarshalError> {
        // Hand the decoded object over rather than copying it: `code` is owned
        // here and dropped immediately after, so `box_code_constant`'s borrow
        // form duplicated the whole recursive constants graph and then threw
        // the original away — once per code object in every unmarshalled
        // module.
        // `str_from_value` ran once per co_names entry, once per localsplus
        // name, then filename / co_name / co_qualname.
        let name_count = code.names.len() + code.localspluskinds.len() + 3;
        // `str_from_value` pushed `co_names`, then localsplus names, then
        // filename / co_name / co_qualname. The first run is the interned
        // objects `pycode.py` `PyCode.__init__` would build again.
        let slots = unsafe { &*self.names };
        let interned_name_slots: &[usize] = if slots.len() >= name_count {
            let start = slots.len() - name_count;
            &slots[start..start + code.names.len()]
        } else {
            &[]
        };
        let code = Rooted::new(crate::pycode::box_code_object_with_interned_name_slots(
            code,
            interned_name_slots,
        ));
        // `box_code_object` allocates, so read each constant out of its
        // shadow-stack slot only now. PyPy gives the complete decoded wrapped
        // list to `PyCode.__init__`; replace the compiler-boundary eager values
        // with those exact marshal objects.
        let constants: Vec<_> = constants.into_iter().map(Rooted::get).collect();
        unsafe { crate::pycode::w_code_fill_wrapped_consts(code.get(), &constants) };
        unsafe { crate::pycode::set_co_code_bytes(code.get(), raw_code_bytes) };
        // `unmarshal_pycode` passes the decoded str objects to `PyCode`.
        // A lone surrogate has no `String` spelling; `install_code_name_objects`
        // is the same store `code_replace` uses.
        let slots = unsafe { &mut *self.names };
        if slots.len() >= name_count {
            let name_obj = pyre_object::gc_roots::shadow_stack_get(slots[slots.len() - 2]);
            let qual_obj = pyre_object::gc_roots::shadow_stack_get(slots[slots.len() - 1]);
            slots.truncate(slots.len() - name_count);
            let name = unsafe { (!pyre_object::w_str_is_utf8(name_obj)).then_some(name_obj) };
            let qualname = unsafe { (!pyre_object::w_str_is_utf8(qual_obj)).then_some(qual_obj) };
            return Ok(Rooted::new(crate::pycode::install_code_name_objects(
                code.get(),
                name,
                qualname,
            )));
        }
        Ok(code)
    }
}

impl wire::MarshalBag for PyreMarshalBag {
    type Value = Rooted;
    type ConstantBag = BasicBag;

    fn make_bool(&self, value: bool) -> Rooted {
        Rooted::new(w_bool_from(value))
    }

    fn make_none(&self) -> Rooted {
        Rooted::new(w_none())
    }

    fn make_ellipsis(&self) -> Rooted {
        Rooted::new(special::w_ellipsis())
    }

    fn make_float(&self, value: f64) -> Rooted {
        Rooted::new(w_float_new(value))
    }

    fn make_complex(&self, value: Complex64) -> Rooted {
        Rooted::new(w_complex_new(value.re, value.im))
    }

    fn make_str(&self, value: &Wtf8) -> Rooted {
        // `objspace.py` `newutf8` copies these marshal bytes once into the
        // STR payload.  An owned `Wtf8Buf` here would be a second buffer.
        Rooted::new(w_str_from_wtf8_managed_borrowed(value))
    }

    fn make_interned_str(&self, value: &Wtf8) -> Rooted {
        // The reader holds only the characters, so ask the intern table with
        // them.  Building a `str` first and handing it to `intern_exact_str`
        // would allocate a `malloc_typed`-immortal object per occurrence and
        // abandon it whenever the value is already interned, retaining it for
        // the process lifetime; interned names repeat heavily across a
        // module's code objects, so nearly every occurrence is such a hit.
        Rooted::new(unicodeobject::intern_wtf8_value(value))
    }

    fn make_bytes(&self, value: &[u8]) -> Rooted {
        Rooted::new(bytesobject::w_bytes_from_bytes(value))
    }

    fn make_int(&self, value: malachite_bigint::BigInt) -> Rooted {
        let value = crate::compiler_bigint_to_rbigint(&value);
        let obj = if longobject::jit_bigint_to_i64_fits(&value) != 0 {
            w_int_new(longobject::jit_bigint_to_i64_value(&value))
        } else {
            longobject::w_long_new(value)
        };
        Rooted::new(obj)
    }

    fn make_tuple(&self, elements: impl Iterator<Item = Rooted>) -> Rooted {
        Rooted::new(w_tuple_new(rooted_ptrs(elements)))
    }

    fn make_tuple_placeholder(&self, len: usize) -> Result<Option<Rooted>, wire::MarshalError> {
        let elements = self.placeholder_elements(len)?;
        Ok(Some(Rooted::new(tupleobject::w_tuple_new_array_backed(
            elements,
        ))))
    }

    fn set_tuple_item(
        &self,
        tuple: &Rooted,
        index: usize,
        value: Rooted,
    ) -> Result<(), wire::MarshalError> {
        unsafe { tupleobject::w_tuple_setitem_initializing(tuple.get(), index, value.get()) }
            .then_some(())
            .ok_or(wire::MarshalError::BadType)
    }

    fn make_code(&self, code: CodeObject<ConstantData>) -> Result<Rooted, wire::MarshalError> {
        // Owned here and used nowhere else — hand it over rather than copying
        // the recursive constants graph, as in `make_runtime_code`.
        Ok(Rooted::new(crate::pycode::box_code_object(code)))
    }

    fn make_stop_iter(&self) -> Result<Rooted, wire::MarshalError> {
        crate::builtins::lookup_exc_class("StopIteration")
            .map(Rooted::new)
            .ok_or(wire::MarshalError::BadType)
    }

    fn make_list(
        &self,
        elements: impl Iterator<Item = Rooted>,
    ) -> Result<Rooted, wire::MarshalError> {
        Ok(Rooted::new(w_list_new(rooted_ptrs(elements))))
    }

    fn make_list_placeholder(&self, len: usize) -> Result<Option<Rooted>, wire::MarshalError> {
        let elements = self.placeholder_elements(len)?;
        Ok(Some(Rooted::new(listobject::w_list_new_object(elements))))
    }

    fn set_list_item(
        &self,
        list: &Rooted,
        index: usize,
        value: Rooted,
    ) -> Result<(), wire::MarshalError> {
        unsafe { listobject::w_list_setitem(list.get(), index as i64, value.get()) }
            .then_some(())
            .ok_or(wire::MarshalError::BadType)
    }

    fn make_set(
        &self,
        elements: impl Iterator<Item = Rooted>,
    ) -> Result<Rooted, wire::MarshalError> {
        let set = Rooted::new(setobject::w_set_new());
        for item in elements {
            let hash = crate::baseobjspace::hash_w_strict(item.get())
                .map_err(|error| self.remember_python_error(error))?;
            unsafe { setobject::w_set_add_hashed_checked(set.get(), item.get(), hash) }.map_err(
                |error| {
                    self.remember_python_error(crate::baseobjspace::map_set_update_error(error))
                },
            )?;
        }
        Ok(set)
    }

    fn make_set_placeholder(&self) -> Option<Rooted> {
        Some(Rooted::new(setobject::w_set_new()))
    }

    fn insert_set_item(&self, set: &Rooted, item: Rooted) -> Result<(), wire::MarshalError> {
        let hash = crate::baseobjspace::hash_w_strict(item.get())
            .map_err(|error| self.remember_python_error(error))?;
        unsafe { setobject::w_set_add_hashed_checked(set.get(), item.get(), hash) }.map_err(
            |error| self.remember_python_error(crate::baseobjspace::map_set_update_error(error)),
        )
    }

    fn make_frozenset(
        &self,
        elements: impl Iterator<Item = Rooted>,
    ) -> Result<Rooted, wire::MarshalError> {
        let set = Rooted::new(setobject::w_frozenset_new());
        for item in elements {
            let hash = crate::baseobjspace::hash_w_strict(item.get())
                .map_err(|error| self.remember_python_error(error))?;
            unsafe { setobject::w_set_add_hashed_checked(set.get(), item.get(), hash) }.map_err(
                |error| {
                    self.remember_python_error(crate::baseobjspace::map_set_update_error(error))
                },
            )?;
        }
        Ok(set)
    }

    fn make_dict(
        &self,
        elements: impl Iterator<Item = (Rooted, Rooted)>,
    ) -> Result<Rooted, wire::MarshalError> {
        let dict = Rooted::new(w_dict_new());
        for (key, value) in elements {
            unsafe { w_dict_store_checked(dict.get(), key.get(), value.get()) }.map_err(|_| {
                self.remember_python_error(crate::baseobjspace::take_pending_dict_key_error(
                    key.get(),
                ))
            })?;
        }
        Ok(dict)
    }

    fn make_dict_placeholder(&self) -> Option<Rooted> {
        Some(Rooted::new(w_dict_new()))
    }

    fn insert_dict_item(
        &self,
        dict: &Rooted,
        key: Rooted,
        value: Rooted,
    ) -> Result<(), wire::MarshalError> {
        unsafe { w_dict_store_checked(dict.get(), key.get(), value.get()) }.map_err(|_| {
            self.remember_python_error(crate::baseobjspace::take_pending_dict_key_error(key.get()))
        })
    }

    fn make_slice(
        &self,
        start: Rooted,
        stop: Rooted,
        step: Rooted,
    ) -> Result<Rooted, wire::MarshalError> {
        Ok(Rooted::new(w_slice_new(
            start.get(),
            stop.get(),
            step.get(),
        )))
    }

    fn constant_bag(self) -> BasicBag {
        BasicBag
    }

    fn constant_ref_from_value(&self, value: &Rooted) -> Option<ConstantData> {
        unsafe { crate::pycode::obj_to_constant_data(value.get()).ok() }
    }

    /// A `co_consts` entry the compiler enum cannot describe — a list, a dict,
    /// a set — is legal in a code object and marshals fine; it takes a shape
    /// placeholder here and reaches its slot through
    /// `make_code_with_constants`, which is what `read_code_consts` does for
    /// `code.replace(co_consts=...)`.
    fn code_constant_from_value(&self, value: &Rooted) -> Result<ConstantData, wire::MarshalError> {
        // `unmarshal_pycode` hands its decoded constants straight to
        // `PyCode.__init__`, whose `co_consts_w` is the only constants table
        // upstream keeps. Serializing the child back into pyre's second,
        // compiler-level table copies its whole body into a second permanent
        // owner; `None` is the arm `is_wrapped_constant` always stores, so the
        // already-decoded wrapper stays this slot's authority instead.
        if unsafe { crate::pycode::is_code(value.get()) } {
            return Ok(ConstantData::None);
        }
        Ok(unsafe { crate::pycode::obj_to_constant_data(value.get()) }
            .unwrap_or(ConstantData::None))
    }

    fn make_code_with_constants(
        &self,
        code: CodeObject<ConstantData>,
        constants: Vec<Rooted>,
    ) -> Result<Rooted, wire::MarshalError> {
        self.make_runtime_code(code, constants, None)
    }

    fn code_units_from_bytes(
        &self,
        code_bytes: &[u8],
    ) -> Result<crate::bytecode::CodeUnits, wire::MarshalError> {
        crate::pycode::decode_code_units(code_bytes)
            .map(|(instructions, _)| instructions)
            .map_err(|()| wire::MarshalError::InvalidBytecode)
    }

    fn make_code_with_constants_and_bytes(
        &self,
        code: CodeObject<ConstantData>,
        constants: Vec<Rooted>,
        code_bytes: Vec<u8>,
    ) -> Result<Rooted, wire::MarshalError> {
        // `code_units_from_bytes` already substituted the `Reserved`
        // placeholder for every opcode byte `CodeUnit` cannot spell, so the
        // decoded stream itself says whether the exact public bytes still have
        // to be carried; decoding them a second time would only rediscover it.
        let raw_code_bytes = code
            .instructions
            .iter()
            .any(|unit| matches!(unit.op, crate::bytecode::Instruction::Reserved))
            .then_some(code_bytes);
        self.make_runtime_code(code, constants, raw_code_bytes)
    }

    // `deserialize_code_value_inner` reads a code object's fields as bag
    // values, so the three shapes `write_code` writes them in — bytes for
    // co_code/co_localspluskinds/co_linetable/co_exceptiontable, tuples for
    // co_consts/co_names/co_localsplusnames, and str for the three names —
    // have to be readable back out of the wrapped objects.  Without them the
    // reader has no way to reach the bytes and every code object it decodes
    // fails as `BadType`.
    fn bytes_from_value(&self, value: &Rooted) -> Option<Vec<u8>> {
        let obj = value.get();
        unsafe { bytesobject::is_bytes(obj) }
            .then(|| unsafe { bytesobject::w_bytes_data(obj) }.to_vec())
    }

    fn str_from_value(&self, value: &Rooted) -> Option<String> {
        let obj = value.get();
        if !unsafe { unicodeobject::is_str(obj) } {
            return None;
        }
        // A code object's names are overwhelmingly ascii, and an ascii payload
        // is one byte per code point, so the cached counts behind this
        // accessor decide it carries no surrogate without reading the bytes.
        // `to_string_lossy` has no such shortcut and rescans every name.
        unsafe { (*self.names).push(value.0) };
        Some(match unsafe { unicodeobject::w_str_get_value_opt(obj) } {
            Some(text) => text.to_owned(),
            None => unsafe { unicodeobject::w_str_get_wtf8(obj) }
                .to_string_lossy()
                .into_owned(),
        })
    }

    fn tuple_elements_from_value(&self, value: &Rooted) -> Option<Vec<Rooted>> {
        let tuple = Rooted::new(value.get());
        if !unsafe { is_tuple(tuple.get()) } {
            return None;
        }
        // Each element is rooted as it is read: nothing here allocates, but
        // the caller holds these across the reads for the remaining fields.
        (0..unsafe { w_tuple_len(tuple.get()) })
            .map(|index| unsafe { w_tuple_getitem(tuple.get(), index as i64) }.map(Rooted::new))
            .collect()
    }
}

/// Resolve the optional `version` slot: an omitted slot uses the current
/// format version; an explicit value goes through `int_w`, so a `None`
/// raises `TypeError` like any other non-integer.
fn resolve_version(version: Option<PyObjectRef>) -> Result<i32, PyError> {
    match version {
        Some(value) => Ok(crate::baseobjspace::int_w(value)? as i32),
        None => Ok(wire::FORMAT_VERSION as i32),
    }
}

/// Resolve the keyword-only `allow_code` slot by truth-testing; an omitted
/// slot defaults to true.
fn resolve_allow_code(allow_code: Option<PyObjectRef>) -> Result<bool, PyError> {
    match allow_code {
        Some(value) => crate::baseobjspace::is_true(value),
        None => Ok(true),
    }
}

/// Serialize one object to a marshal byte stream at `version`.  With
/// `allow_code` false, a nested code object is rejected before writing
/// (marshal.check_no_code).
fn marshal_to_bytes(
    value: PyObjectRef,
    version: i32,
    allow_code: bool,
) -> Result<Vec<u8>, PyError> {
    if !allow_code {
        reject_code(value)?;
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let value = pyre_object::gc_roots::pin_root(value);
    let mut out = Vec::new();
    let mut refs = (version >= 3).then(WriterRefs::new);
    write_object(&mut out, value, &mut refs, version, MAX_DEPTH)?;
    Ok(out)
}

/// Deserialize one object from a marshal byte stream.  With `allow_code`
/// false, a decoded code object is rejected (marshal.check_no_code).
fn unmarshal_bytes(data: &[u8], allow_code: bool) -> PyResult {
    let roots = pyre_object::gc_roots::push_roots();
    let mut pending_error = None;
    let mut name_slots = Vec::with_capacity(64);
    let errors = ErrorSink::park(&roots, &mut pending_error);
    let bag = PyreMarshalBag::new(errors, &mut name_slots);
    let mut reader = BytesReader {
        data,
        errors: bag.errors,
    };
    let result = match wire::deserialize_value(&mut reader, bag) {
        Ok(result) => result,
        Err(error) => return Err(errors.restore(error)),
    };
    let result = result.get();
    if !allow_code {
        reject_code(result)?;
    }
    Ok(result)
}

/// RustPython `marshal.check_no_code`, matching CPython's recursive
/// `allow_code=False` check.  The Vec is a transient identity set (PyPy's
/// traversal does not persist it); it also prevents container cycles from
/// recursing forever without introducing a side table.
fn contains_code(obj: PyObjectRef, seen: &mut Vec<PyObjectRef>) -> bool {
    if unsafe { crate::pycode::is_code(obj) } {
        return true;
    }
    if seen.iter().any(|&seen_obj| std::ptr::eq(seen_obj, obj)) {
        return false;
    }
    let is_container = unsafe {
        is_tuple(obj) || is_list(obj) || is_dict(obj) || setobject::is_set_or_frozenset(obj)
    };
    if !is_container {
        return false;
    }
    seen.push(obj);
    unsafe {
        if is_tuple(obj) {
            (0..w_tuple_len(obj)).any(|index| {
                w_tuple_getitem(obj, index as i64).is_some_and(|item| contains_code(item, seen))
            })
        } else if is_list(obj) {
            (0..w_list_len(obj)).any(|index| {
                w_list_getitem(obj, index as i64).is_some_and(|item| contains_code(item, seen))
            })
        } else if is_dict(obj) {
            dictmultiobject::w_dict_items(obj)
                .into_iter()
                .any(|(key, value)| contains_code(key, seen) || contains_code(value, seen))
        } else {
            setobject::w_set_items(obj)
                .into_iter()
                .any(|item| contains_code(item, seen))
        }
    }
}

fn reject_code(value: PyObjectRef) -> Result<(), PyError> {
    if contains_code(value, &mut Vec::new()) {
        Err(PyError::value_error(
            "unmarshalling code objects is disallowed",
        ))
    } else {
        Ok(())
    }
}

/// `PyMarshal_ReadObjectFromString` — deserialize one object out of a raw byte
/// buffer.  `_imp.get_frozen_object` unmarshals caller-supplied frozen data
/// through this and rewrites any failure into its own diagnostic.
pub(crate) fn loads_bytes(data: &[u8]) -> PyResult {
    unmarshal_bytes(data, true)
}

/// `PyMarshal_WriteObjectToString` — serialize one object into a raw byte
/// buffer at the current format version.  `_imp.find_frozen(withdata=True)`
/// hands these bytes back as the frozen data, so they are the same stream
/// `loads_bytes` reads.
pub(crate) fn dumps_bytes(value: PyObjectRef) -> Result<Vec<u8>, PyError> {
    marshal_to_bytes(value, wire::FORMAT_VERSION as i32, true)
}

crate::py_module! {
    "marshal",
    int_constants: {
        "version" => wire::FORMAT_VERSION as i64,
    },
    inline_functions: {
        // interp_marshal.py `dumps(w_data, version=Py_MARSHAL_VERSION)` —
        // `value` / `version` positional-only, `allow_code` keyword-only
        // (the 3.13 accelerator signature `dumps(value, version, /, *,
        // allow_code=True)`).  `version` stays `Option` so an explicit
        // `None` reaches `int_w` and raises rather than defaulting.
        fn dumps(
            value: PyObjectRef,
            version: Option<PyObjectRef>,
            #[posonly]
            #[kwonly]
            allow_code: Option<PyObjectRef>,
        ) -> Result<PyObjectRef, crate::PyError> {
            let _roots = pyre_object::gc_roots::push_roots();
            let has_version = version.is_some();
            let has_allow = allow_code.is_some();
            let mut live = [pyre_object::PY_NULL; 3];
            live[0] = value;
            let mut n = 1;
            if let Some(v) = version {
                live[n] = v;
                n += 1;
            }
            if let Some(v) = allow_code {
                live[n] = v;
                n += 1;
            }
            let base = pyre_object::gc_roots::pin_roots(&live[..n]);
            let version_slot = has_version.then_some(base + 1);
            let allow_slot = has_allow.then_some(base + 1 + usize::from(has_version));
            let version =
                resolve_version(version_slot.map(pyre_object::gc_roots::shadow_stack_get))?;
            let allow_code =
                resolve_allow_code(allow_slot.map(pyre_object::gc_roots::shadow_stack_get))?;
            let out = marshal_to_bytes(
                pyre_object::gc_roots::shadow_stack_get(base),
                version,
                allow_code,
            )?;
            Ok(bytesobject::w_bytes_from_bytes(&out))
        }
        // interp_marshal.py `loads(w_str)` — `bytes` positional-only,
        // `allow_code` keyword-only (`loads(bytes, /, *, allow_code=True)`).
        fn loads(
            data: PyObjectRef,
            #[posonly]
            #[kwonly]
            allow_code: Option<PyObjectRef>,
        ) -> Result<PyObjectRef, crate::PyError> {
            let _roots = pyre_object::gc_roots::push_roots();
            let has_allow = allow_code.is_some();
            let mut live = [pyre_object::PY_NULL; 2];
            live[0] = data;
            let mut n = 1;
            if let Some(v) = allow_code {
                live[n] = v;
                n += 1;
            }
            let base = pyre_object::gc_roots::pin_roots(&live[..n]);
            let allow_slot = has_allow.then_some(base + 1);
            let allow_code =
                resolve_allow_code(allow_slot.map(pyre_object::gc_roots::shadow_stack_get))?;
            let data = bytes_like(pyre_object::gc_roots::shadow_stack_get(base), "loads")?;
            unmarshal_bytes(data.as_slice(), allow_code)
        }
        // interp_marshal.py `dump(w_data, w_f, version=Py_MARSHAL_VERSION)`
        // — writes the stream `dumps` would return to `f.write`
        // (`dump(value, file, version, /, *, allow_code=True)`).
        fn dump(
            value: PyObjectRef,
            file: PyObjectRef,
            version: Option<PyObjectRef>,
            #[posonly]
            #[kwonly]
            allow_code: Option<PyObjectRef>,
        ) -> Result<PyObjectRef, crate::PyError> {
            let _roots = pyre_object::gc_roots::push_roots();
            let has_version = version.is_some();
            let has_allow = allow_code.is_some();
            let mut live = [pyre_object::PY_NULL; 4];
            live[0] = value;
            live[1] = file;
            let mut n = 2;
            if let Some(v) = version {
                live[n] = v;
                n += 1;
            }
            if let Some(v) = allow_code {
                live[n] = v;
                n += 1;
            }
            let base = pyre_object::gc_roots::pin_roots(&live[..n]);
            let version_slot = has_version.then_some(base + 2);
            let allow_slot = has_allow.then_some(base + 2 + usize::from(has_version));
            let version =
                resolve_version(version_slot.map(pyre_object::gc_roots::shadow_stack_get))?;
            let allow_code =
                resolve_allow_code(allow_slot.map(pyre_object::gc_roots::shadow_stack_get))?;
            let out = marshal_to_bytes(
                pyre_object::gc_roots::shadow_stack_get(base),
                version,
                allow_code,
            )?;
            let bytes = bytesobject::w_bytes_from_bytes(&out);
            call_method(
                pyre_object::gc_roots::shadow_stack_get(base + 1),
                "write",
                &[bytes],
            )?;
            Ok(w_none())
        }
        // interp_marshal.py `load(w_f)` reads one value from `f` and
        // rewinds `f` past exactly the bytes consumed
        // (`load(file, /, *, allow_code=True)`).
        fn load(
            file: PyObjectRef,
            #[posonly]
            #[kwonly]
            allow_code: Option<PyObjectRef>,
        ) -> Result<PyObjectRef, crate::PyError> {
            let roots = pyre_object::gc_roots::push_roots();
            let has_allow = allow_code.is_some();
            let mut live = [pyre_object::PY_NULL; 2];
            live[0] = file;
            let mut n = 1;
            if let Some(v) = allow_code {
                live[n] = v;
                n += 1;
            }
            let base = pyre_object::gc_roots::pin_roots(&live[..n]);
            let allow_slot = has_allow.then_some(base + 1);
            let allow_code =
                resolve_allow_code(allow_slot.map(pyre_object::gc_roots::shadow_stack_get))?;
            let (rooted_file, has_readinto) =
                FileReader::probe(pyre_object::gc_roots::shadow_stack_get(base))?;
            let mut pending_error = None;
            let mut name_slots = Vec::with_capacity(64);
            let errors = ErrorSink::park(&roots, &mut pending_error);
            let bag = PyreMarshalBag::new(errors, &mut name_slots);
            let mut reader = FileReader::from_probed(rooted_file, has_readinto, bag.errors);
            let result = match wire::deserialize_value(&mut reader, bag) {
                Ok(result) => result,
                Err(error) => return Err(errors.restore(error)),
            };
            let result = result.get();
            if !allow_code {
                reject_code(result)?;
            }
            Ok(result)
        }
    },
    extra_init: |ns| {
        // `loads(bytes, /, *, allow_code=True)` and the dump/dumps/load
        // siblings. `_match_signature` requires `Function.w_kw_defs` for
        // that declared default (`interp2app._getdefaults`).
        let ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(ns);
        for name in ["dumps", "loads", "dump", "load"] {
            if let Some(mut function) = unsafe {
                pyre_object::w_dict_getitem_str(
                    pyre_object::gc_roots::shadow_stack_get(ns_slot),
                    name,
                )
            } {
                function = crate::gateway::function_set_kwonly_default_pairs(
                    function,
                    &[("allow_code", pyre_object::w_bool_from(true))],
                );
                crate::__pyre_put_new!(ns_slot, name, function);
            }
        }
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_str_reads_the_marshal_buffer_without_an_owned_copy() {
        let roots = pyre_object::gc_roots::push_roots();
        let mut pending_error = None;
        let mut name_slots = Vec::new();
        let bag = PyreMarshalBag::new(ErrorSink::park(&roots, &mut pending_error), &mut name_slots);
        let text = rustpython_wtf8::Wtf8Buf::from("stat_result");
        let rooted = wire::MarshalBag::make_str(&bag, &text);
        unsafe {
            assert_eq!(pyre_object::w_str_len(rooted.get()), 11);
            assert_eq!(pyre_object::w_str_get_wtf8(rooted.get()), "stat_result");
        }
        let wide = rustpython_wtf8::Wtf8Buf::from("é");
        let rooted = wire::MarshalBag::make_str(&bag, &wide);
        unsafe {
            assert_eq!(pyre_object::w_str_len(rooted.get()), 1);
            assert_eq!(pyre_object::w_str_get_wtf8(rooted.get()), "é");
        }
    }

    #[test]
    fn exact_bytes_are_borrowed_for_unmarshal() {
        let input = bytesobject::w_bytes_from_bytes(b"payload");
        let bytes = bytes_like(input, "loads").expect("bytes input");

        assert!(matches!(bytes, MarshalBytes::Borrowed(b"payload")));
    }

    #[test]
    fn decoded_code_constant_uses_the_wrapped_pycode_as_authority() {
        let w_code = crate::pycode::w_code_new(std::ptr::null());
        let rooted = Rooted::new(w_code);
        let roots = pyre_object::gc_roots::push_roots();
        let mut pending_error = None;
        let mut name_slots = Vec::new();
        let bag = PyreMarshalBag::new(ErrorSink::park(&roots, &mut pending_error), &mut name_slots);

        let placeholder = wire::MarshalBag::code_constant_from_value(&bag, &rooted)
            .expect("PyCode constant must be accepted");

        assert!(matches!(placeholder, ConstantData::None));
    }

    /// A later Python `read` can collect while `ErrorSink` still holds the
    /// first parked handle. The load_impl slot is the root; restore reloads
    /// from it (`shadowstack.py expand_pop_roots`).
    #[test]
    fn parked_python_error_survives_a_collection_before_restore() {
        let roots = pyre_object::gc_roots::push_roots();
        let mut pending_error = None;
        let sink = ErrorSink::park(&roots, &mut pending_error);
        let original = PyError::value_error("parked marshal error");
        let handle = original.as_raw();
        sink.remember(original);

        let forwarded =
            pyre_object::lltype::malloc_typed_managed(unsafe { std::ptr::read(handle) });
        pyre_object::gc_roots::walk_shadow_stack(|slot| {
            if std::ptr::eq(*slot, handle as pyre_object::PyObjectRef) {
                *slot = forwarded as pyre_object::PyObjectRef;
            }
        });

        let restored = sink.restore(wire::MarshalError::BadType);
        assert_eq!(restored.as_raw(), forwarded);
        assert_eq!(restored.kind, crate::PyErrorKind::ValueError);
    }
}
