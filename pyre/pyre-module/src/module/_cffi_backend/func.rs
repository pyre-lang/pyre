//! The module-level functions — PyPy:
//! `pypy/module/_cffi_backend/func.py`.

use pyre_interpreter::PyError;
use pyre_object::PyObjectRef;

use super::cdataobj::{self, W_CData};
use super::ctypeobj;
use super::newtype;

/// `func.py OffsetInBytes`.
#[pyre_interpreter::pyre_class("_cffi_backend._OffsetInBytes")]
#[derive(Default)]
pub struct OffsetInBytes {
    /// `OffsetInBytes.bytes`.
    pub w_bytes: PyObjectRef,
    /// `OffsetInBytes.offset`.
    pub offset: i64,
}

static OFFSET_IN_BYTES_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

fn offset_in_bytes_type() -> PyObjectRef {
    OFFSET_IN_BYTES_TYPE_OBJ.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type_with_layout(
            "_cffi_backend._OffsetInBytes",
            |_| {},
            pyre_interpreter::typedef::w_object(),
            <OffsetInBytes as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<OffsetInBytes as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        unsafe {
            pyre_object::w_type_set_disallow_instantiation(tp);
            pyre_object::w_type_set_acceptable_as_base_class(tp, false);
        }
        tp
    })
}

/// The positional-or-keyword binding a `gateway.py` `unwrap_spec` wrapper does
/// for a module-level entry point that carries a default, and so cannot
/// declare a fixed arity at registration.  `argnames` is the whole parameter
/// list and `required` how many of them have no default; a parameter that was
/// not supplied comes back as `PY_NULL`, which [`optional`] reads as absent.
/// The gateway `Signature` has already bound keywords into positional order.
pub(super) fn bind_entry_point(
    args: &[PyObjectRef],
    name: &'static str,
    argnames: &[&'static str],
    required: usize,
) -> Result<Vec<PyObjectRef>, PyError> {
    let mut bound = Vec::with_capacity(argnames.len());
    for i in 0..argnames.len() {
        let value = args
            .get(i)
            .copied()
            .filter(|value| !value.is_null())
            .unwrap_or(pyre_object::PY_NULL);
        bound.push(value);
    }
    let missing: Vec<&str> = argnames[..required]
        .iter()
        .enumerate()
        .filter(|(index, _)| bound[*index].is_null())
        .map(|(_, argname)| *argname)
        .collect();
    if !missing.is_empty() {
        let plural = if missing.len() == 1 { "" } else { "s" };
        return Err(PyError::type_error(format!(
            "{name}() missing {} required positional argument{plural}: {}",
            missing.len(),
            join_argnames(&missing)
        )));
    }
    Ok(bound)
}

/// `argument.py ArgErrCount`'s name list: `'a'`, `'a' and 'b'`, `'a', 'b', and
/// 'c'`.
fn join_argnames(names: &[&str]) -> String {
    let quoted: Vec<String> = names.iter().map(|name| format!("'{name}'")).collect();
    match quoted.as_slice() {
        [only] => only.clone(),
        [first, second] => format!("{first} and {second}"),
        [head @ .., last] => format!("{}, and {last}", head.join(", ")),
        [] => String::new(),
    }
}

/// One optional slot of [`bind_entry_point`]'s result.
fn optional(bound: &[PyObjectRef], index: usize) -> Option<PyObjectRef> {
    let value = bound[index];
    (!value.is_null()).then_some(value)
}

/// `func.py newp`.
pub fn newp(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_entry_point(args, "newp", &["ctype", "init"], 1)?;
    let w_init = optional(&a, 1).unwrap_or_else(pyre_object::w_none);
    ctypeobj::newp(a[0], w_init)
}

/// `func.py cast`.
pub fn cast(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    ctypeobj::cast(args[0], args[1])
}

/// `func.py callback`.
pub fn callback(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_entry_point(
        args,
        "callback",
        &["ctype", "callable", "error", "onerror"],
        2,
    )?;
    let w_error = optional(&a, 2).unwrap_or_else(pyre_object::w_none);
    let w_onerror = optional(&a, 3).unwrap_or_else(pyre_object::w_none);
    super::ccallback::make_callback(a[0], a[1], w_error, w_onerror)
}

/// `func.py typeof`.
pub fn typeof_(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(cdataobj::cdata_arg(args[0])?.ctype)
}

/// `func.py sizeof` — a cdata reports what it owns, a ctype its own size.
pub fn sizeof(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let mut w_obj = args[0];
    let (size, name) = if W_CData::from_obj(w_obj).is_some() {
        let size = pyre_object::with_roots!(w_obj => cdataobj::cdata_sizeof(w_obj))?;
        let ct = ctypeobj::ctype_at(cdataobj::cdata_arg(w_obj)?.ctype)
            .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
        (size, ct.name())
    } else if let Some(ct) = ctypeobj::ctype_at(w_obj) {
        (ct.size, ct.name())
    } else {
        return Err(PyError::type_error("expected a 'cdata' or 'ctype' object"));
    };
    if size < 0 {
        return Err(PyError::value_error(format!(
            "ctype '{name}' is of unknown size"
        )));
    }
    Ok(pyre_object::w_int_new(size))
}

/// `func.py alignof`.
pub fn alignof(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(pyre_object::w_int_new(
        ctypeobj::ctype_arg(args[0])?.alignof()?,
    ))
}

/// `func.py getcname`.
pub fn getcname(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let ct = ctypeobj::ctype_arg(args[0])?;
    let replace_with = pyre_interpreter::baseobjspace::text_w(args[1])?;
    let (name, _) = ct.insert_name(replace_with, 0);
    Ok(pyre_object::w_str_new_managed(&name))
}

/// `func.py string`.
pub fn string(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_entry_point(args, "string", &["cdata", "maxlen"], 1)?;
    let maxlen = match optional(&a, 1) {
        Some(w_maxlen) => pyre_interpreter::baseobjspace::int_w(w_maxlen)?,
        None => -1,
    };
    ctypeobj::string(a[0], maxlen)
}

/// `func.py unpack`.
pub fn unpack(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let mut w_cdata = args[0];
    let length =
        pyre_object::with_roots!(w_cdata => pyre_interpreter::baseobjspace::int_w(args[1]))?;
    cdataobj::unpack(w_cdata, length)
}

/// `func.py typeoffsetof`.
pub fn typeoffsetof(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_entry_point(
        args,
        "typeoffsetof",
        &["ctype", "field_or_index", "following"],
        2,
    )?;
    let following = match optional(&a, 2) {
        Some(w_following) => pyre_interpreter::baseobjspace::int_w(w_following)? != 0,
        None => false,
    };
    let (w_ctype, offset) = direct_typeoffsetof(a[0], a[1], following)?;
    let roots = pyre_object::gc_roots::push_roots();
    let field_slot = roots.base();
    let _ = roots.pin_root(w_ctype);
    // The two-item `w_tuple_new` pins its first item, which is a collection
    // point while an unrooted second item is still only a Rust local.
    let _ = roots.pin_root(pyre_object::w_int_new(offset));
    Ok(pyre_object::w_tuple_new(vec![
        roots.get(field_slot),
        roots.get(field_slot + 1),
    ]))
}

/// `W_CType.direct_typeoffsetof`, shared by the module-level function and
/// `W_FFIObject._more_{addressof,offsetof}`.
pub fn direct_typeoffsetof(
    w_ctype: PyObjectRef,
    w_field_or_index: PyObjectRef,
    following: bool,
) -> Result<(PyObjectRef, i64), PyError> {
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    if unsafe { pyre_object::unicodeobject::is_str(w_field_or_index) } {
        // `W_CTypePointer.typeoffsetof_field` reads the item's field, but
        // only for the first element of the array the pointer may be.
        let owner = if ct.is_struct_or_union() {
            ct
        } else if ct.kind == ctypeobj::KIND_POINTER && !following {
            ctypeobj::ctype_arg(ct.ctitem)?
        } else {
            return Err(PyError::type_error(
                "with a field name argument, expected a struct or union ctype",
            ));
        };
        if !owner.is_struct_or_union() {
            return Err(PyError::type_error(
                "with a field name argument, expected a struct or union ctype",
            ));
        }
        let fieldname = pyre_interpreter::baseobjspace::text_w(w_field_or_index)?;
        return super::ctypestruct::typeoffsetof_field(owner, fieldname);
    }
    let Ok(index) = pyre_interpreter::baseobjspace::int_w(w_field_or_index) else {
        return Err(PyError::type_error("field name or array index expected"));
    };
    // `W_CTypePointer.typeoffsetof_index`, which an array reaches through
    // its own pointer type.
    let ct = match ct.kind {
        ctypeobj::KIND_POINTER => ct,
        ctypeobj::KIND_ARRAY => ctypeobj::ctype_arg(ct.ctptr)?,
        _ => {
            return Err(PyError::type_error(
                "with an integer argument, expected an array or pointer ctype",
            ));
        }
    };
    let ctitem = super::ctypeptr::item_of(ct)?;
    if ctitem.size < 0 {
        return Err(PyError::type_error("pointer to opaque"));
    }
    let offset = index
        .checked_mul(ctitem.size)
        .ok_or_else(|| PyError::overflow_error("array offset would overflow a ssize_t"))?;
    Ok((ctitem.as_object(), offset))
}

/// `func.py rawaddressof`.
pub fn rawaddressof(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_entry_point(args, "rawaddressof", &["ctype", "cdata", "offset"], 3)?;
    let ct = ctypeobj::ctype_arg(a[0])?;
    if ct.kind != ctypeobj::KIND_POINTER {
        return Err(PyError::type_error("expected a pointer ctype"));
    }
    let cdata = cdataobj::cdata_arg(a[1])?;
    let source_ct = ctypeobj::ctype_at(cdata.ctype)
        .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
    if !source_ct.is_ptr_or_array() && !source_ct.is_struct_or_union() {
        return Err(PyError::type_error(
            "expected a cdata struct/union/array/pointer object",
        ));
    }
    let offset = pyre_interpreter::baseobjspace::int_w(a[2])?;
    let ptr = cdata.ptr.wrapping_add_signed(offset as isize);
    Ok(cdataobj::new_cdata(ptr, a[0]))
}

/// `func.py RawBytes` — a `str2charp` copy of a movable bytes STR, freed
/// when the weak-key entry is dropped (`RawBytes.__del__`).
struct RawBytesEntry {
    /// `WEAKDICTENTRY.key` — `GcWeakrefBox` to the bytes object.
    key: usize,
    /// `RawBytes.ptr`.
    ptr: usize,
}

/// `func.py RawBytesCache` — `space.fromcache(RawBytesCache)`.
/// `self.wdict` is `RWeakKeyDictionary(W_Root, RawBytes)`.
///
/// The map is keyed by `gc_identity_hash` of the live bytes object
/// (`ll_get` / `compute_identity_hash`). That hash is stable across a
/// moving GC; a HashMap keyed by the box's target address is not (the
/// target can move). Identity is confirmed with `w_gc_weakref_box_deref`.
struct RawBytesCache {
    wdict: parking_lot::Mutex<std::collections::HashMap<usize, Vec<RawBytesEntry>>>,
}

/// Process-wide `space.fromcache(RawBytesCache)` singleton (`_pickle`
/// `PICKLE_STATE` AtomicPtr / `OnceLock` shape). `HashMap::new` is not
/// `const` here, so the map is built on first use. Weak-box keys are
/// forwarded by [`walk_raw_bytes_cache_gc`] from `walk_prebuilt_slots`.
static RAW_BYTES_CACHE: std::sync::OnceLock<RawBytesCache> = std::sync::OnceLock::new();

fn raw_bytes_cache() -> &'static RawBytesCache {
    RAW_BYTES_CACHE.get_or_init(|| RawBytesCache {
        wdict: parking_lot::Mutex::new(std::collections::HashMap::new()),
    })
}

impl RawBytesCache {
    fn walk_roots(&self, forward: &mut dyn FnMut(&mut PyObjectRef)) {
        let mut wdict = self.wdict.lock();
        for bucket in wdict.values_mut() {
            for entry in bucket.iter_mut() {
                if entry.key != 0 {
                    let mut key = entry.key as PyObjectRef;
                    forward(&mut key);
                    entry.key = key as usize;
                }
            }
        }
    }

    fn prune(wdict: &mut std::collections::HashMap<usize, Vec<RawBytesEntry>>) {
        wdict.retain(|_, bucket| {
            bucket.retain(|entry| {
                let target = unsafe {
                    pyre_object::weakref::w_gc_weakref_box_deref(entry.key as PyObjectRef)
                };
                if target.is_null() {
                    if entry.ptr != 0 {
                        unsafe {
                            majit_rlib::rffi::free_charp(entry.ptr as *mut u8, false);
                        }
                    }
                    false
                } else {
                    true
                }
            });
            !bucket.is_empty()
        });
    }

    fn lookup(
        wdict: &std::collections::HashMap<usize, Vec<RawBytesEntry>>,
        w_x: PyObjectRef,
    ) -> Option<*mut u8> {
        let hash = pyre_object::gc_hook::gc_identity_hash(w_x as usize);
        let bucket = wdict.get(&hash)?;
        for entry in bucket {
            let target =
                unsafe { pyre_object::weakref::w_gc_weakref_box_deref(entry.key as PyObjectRef) };
            if std::ptr::eq(target, w_x) {
                return Some(entry.ptr as *mut u8);
            }
        }
        None
    }

    fn insert(
        wdict: &mut std::collections::HashMap<usize, Vec<RawBytesEntry>>,
        w_x: PyObjectRef,
        key: PyObjectRef,
        ptr: *mut u8,
    ) {
        let hash = pyre_object::gc_hook::gc_identity_hash(w_x as usize);
        let bucket = wdict.entry(hash).or_default();
        // `ll_set_nonnull` replaces a matching live key.
        for entry in bucket.iter_mut() {
            let target =
                unsafe { pyre_object::weakref::w_gc_weakref_box_deref(entry.key as PyObjectRef) };
            if std::ptr::eq(target, w_x) {
                if entry.ptr != 0 && entry.ptr != ptr as usize {
                    unsafe {
                        majit_rlib::rffi::free_charp(entry.ptr as *mut u8, false);
                    }
                }
                entry.key = key as usize;
                entry.ptr = ptr as usize;
                return;
            }
        }
        bucket.push(RawBytesEntry {
            key: key as usize,
            ptr: ptr as usize,
        });
    }
}

/// Forward `RawBytesCache.wdict` weak-box keys (`WEAKDICTENTRY.key`).
pub(crate) fn walk_raw_bytes_cache_gc(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    if let Some(cache) = RAW_BYTES_CACHE.get() {
        cache.walk_roots(visitor);
    }
}

/// `func.py get_raw_address_of_string` — CFFI `ffi.from_buffer` on a bytes
/// object. Movable STRs are copied into a cached raw block; immovable STRs
/// hand out the interior pointer (`rgc.can_move`). Two calls with the same
/// live bytes object return the same pointer.
#[majit_macros::dont_look_inside]
unsafe fn get_raw_address_of_string(w_x: PyObjectRef) -> Result<*mut u8, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let x_slot = _roots.pin_roots(&[w_x]);
    let w_x = _roots.get(x_slot);
    let block = unsafe { pyre_object::bytesobject::w_bytes_block(w_x) };
    // MiniMark: `rgc.must_split_gc_address_space()` is False.
    if !majit_gc::can_move(majit_ir::GcRef(block as usize)) {
        let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_x) };
        let ptr = data.as_ptr() as *mut u8;
        unsafe {
            *ptr.add(data.len()) = 0;
        }
        return Ok(ptr);
    }
    // Allocate the weakref box and the `str2charp` copy before the lock
    // (`ll_set_nonnull` creates the weakref before lookup). Collection is
    // forbidden while `wdict` is held.
    let key = pyre_object::weakref::w_gc_weakref_box_new(_roots.get(x_slot));
    if key.is_null() {
        return Err(PyError::memory_error(""));
    }
    let key_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = _roots.pin_root(key);
    let data = unsafe { pyre_object::bytesobject::w_bytes_data(_roots.get(x_slot)) };
    let ptr = majit_rlib::rffi::str2charp(data, false).map_err(|_| PyError::memory_error(""))?;
    let mut drop_copy = false;
    let result = {
        let mut wdict = raw_bytes_cache().wdict.lock();
        RawBytesCache::prune(&mut wdict);
        if let Some(existing) = RawBytesCache::lookup(&wdict, _roots.get(x_slot)) {
            drop_copy = true;
            existing
        } else {
            RawBytesCache::insert(
                &mut wdict,
                _roots.get(x_slot),
                pyre_object::gc_roots::shadow_stack_get(key_slot),
                ptr,
            );
            ptr
        }
    };
    if drop_copy {
        unsafe {
            majit_rlib::rffi::free_charp(ptr, false);
        }
    }
    Ok(result)
}

/// `func.py from_buffer`.
pub fn from_buffer(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_entry_point(args, "from_buffer", &["ctype", "x", "require_writable"], 2)?;
    let w_ctype = a[0];
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    if !ct.is_ptr_or_array() {
        return Err(PyError::type_error(format!(
            "expected a pointer or array ctype, got '{}'",
            ct.name()
        )));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let object_slot = roots.pin_roots(&[a[1], w_ctype]);
    let ctype_slot = object_slot + 1;
    if unsafe { pyre_object::unicodeobject::is_str(roots.get(object_slot)) } {
        return Err(PyError::type_error(
            "from_buffer() cannot return the address of a unicode object",
        ));
    }
    let require_writable = match optional(&a, 2) {
        Some(w) => pyre_interpreter::baseobjspace::int_w(w)? != 0,
        None => false,
    };
    // `func.py from_buffer`: `_fetch_as_write_buffer` / `_fetch_as_read_buffer`
    // first, then `isinstance_w(w_x, space.w_bytes)` picks
    // `get_raw_address_of_string` vs `buf.get_raw_address()`. Length is
    // `buf.getlength()` of the acquired buffer.
    let (fetched_ptr, buffersize, fetched_owner) = if require_writable {
        let (data, owner, _) =
            unsafe { pyre_interpreter::builtins::fileio_writebuf(roots.get(object_slot)) }?;
        (data.as_mut_ptr(), data.len() as i64, owner)
    } else {
        let buf = unsafe { pyre_interpreter::builtins::acquire_readbuf(roots.get(object_slot)) }?;
        (
            buf.as_ptr() as *mut u8,
            buf.len() as i64,
            roots.get(object_slot),
        )
    };
    let bytes_type =
        pyre_interpreter::typedef::gettypeobject(&pyre_object::bytesobject::BYTES_TYPE);
    let (ptr, buffersize, owner) = if unsafe {
        pyre_interpreter::baseobjspace::isinstance_w(roots.get(object_slot), bytes_type)
    } {
        let ptr = unsafe { get_raw_address_of_string(roots.get(object_slot)) }?;
        (ptr, buffersize, roots.get(object_slot))
    } else if require_writable {
        (fetched_ptr, buffersize, fetched_owner)
    } else {
        let (ptr, _len) = unsafe {
            pyre_interpreter::builtins::acquire_readbuf_raw_address(roots.get(object_slot))
        }?;
        (ptr, buffersize, roots.get(object_slot))
    };
    let owner_slot = object_slot + 2;
    let _ = roots.pin_root(owner);
    let held = unsafe { pyre_interpreter::builtins::buffer_export_incref(roots.get(owner_slot)) };

    let arraylength = if ct.kind != ctypeobj::KIND_ARRAY {
        buffersize
    } else if ct.length >= 0 {
        if buffersize < ct.size {
            if held {
                unsafe { pyre_interpreter::builtins::buffer_export_decref(roots.get(owner_slot)) };
            }
            return Err(PyError::value_error(format!(
                "buffer is too small ({buffersize} bytes) for '{}' ({} bytes)",
                ct.name(),
                ct.size
            )));
        }
        ct.length
    } else {
        let itemsize = super::ctypeptr::item_of(ct)?.size;
        if itemsize == 1 {
            buffersize
        } else if itemsize > 0 {
            buffersize / itemsize
        } else {
            if held {
                unsafe { pyre_interpreter::builtins::buffer_export_decref(roots.get(owner_slot)) };
            }
            return Err(PyError::new(
                pyre_interpreter::PyErrorKind::ZeroDivisionError,
                format!(
                    "from_buffer('{}', ..): the actual length of the array cannot be computed",
                    ct.name()
                ),
            ));
        }
    };
    Ok(cdataobj::new_cdata_from_buffer(
        ptr as usize,
        arraylength,
        roots.get(ctype_slot),
        roots.get(object_slot),
        roots.get(owner_slot),
        held,
    ))
}

/// `func.py gcp`.
pub fn gcp(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_entry_point(args, "gcp", &["cdata", "destructor", "size"], 2)?;
    let roots = pyre_object::gc_roots::push_roots();
    let cdata_slot = roots.base();
    let _ = roots.pin_root(a[0]);
    let destructor_slot = cdata_slot + 1;
    let _ = roots.pin_root(a[1]);
    let size = match optional(&a, 2) {
        Some(w_size) => pyre_interpreter::baseobjspace::int_w(w_size)?,
        None => 0,
    };
    cdataobj::with_gc(roots.get(cdata_slot), roots.get(destructor_slot), size)
}

/// `func.py offset_in_bytes`.
pub fn offset_in_bytes(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_entry_point(args, "_offset_in_bytes", &["bytes", "offset"], 2)?;
    if !unsafe { pyre_object::bytesobject::is_bytes(a[0]) } {
        return Err(PyError::type_error(format!(
            "must be bytes, not {}",
            pyre_interpreter::type_methods::arg_type_name(a[0])
        )));
    }
    let _ = offset_in_bytes_type();
    let roots = pyre_object::gc_roots::push_roots();
    let bytes_slot = roots.base();
    let _ = roots.pin_root(a[0]);
    let offset = pyre_interpreter::baseobjspace::int_w(a[1])?;
    let obj = OffsetInBytes::allocate_stable(OffsetInBytes {
        offset,
        ..Default::default()
    });
    OffsetInBytes::from_obj(obj)
        .expect("allocate_stable hands back this layout")
        .w_bytes = roots.get(bytes_slot);
    // The wrapper is born old-gen and the bytes object may be young.
    pyre_object::gc_hook::try_gc_write_barrier_managed(obj as pyre_object::gc_hook::GCREF);
    Ok(obj)
}

/// `func.py memmove`.
pub fn memmove(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // `simple_buffer_bytes` copies a non-cdata source into a host `Vec`
    // (`charbuf_w`). The size argument's `__int__` runs arbitrary Python, so
    // dest/src/n stay pinned across it.
    let roots = pyre_object::gc_roots::push_roots();
    let args_base = roots.pin_roots(&[args[0], args[1], args[2]]);
    let src_slot = args_base + 1;
    let n = pyre_interpreter::baseobjspace::int_w(roots.get(args_base + 2))?;
    if n < 0 {
        return Err(PyError::value_error("negative size"));
    }
    // `@unwrap_spec(n=int)` binds a machine word, so a size that does not fit
    // one is refused rather than truncated into the checks and the copy.
    let n = usize::try_from(n)
        .map_err(|_| PyError::value_error("size does not fit in a machine word"))?;
    let mut dest_buffer = None;
    let w_dest = roots.get(args_base);
    let dest = if let Some(cdata) = W_CData::from_obj(w_dest) {
        unsafe_escaping_ptr_for_ptr_or_array(cdata)?
    } else {
        dest_buffer = Some(unsafe { pyre_interpreter::builtins::WritableBuffer::acquire(w_dest) }?);
        let slice = unsafe { dest_buffer.as_mut().expect("just filled").as_mut_slice() };
        // `_fetch_as_write_buffer`'s non-raw arm writes with `setitem`, which
        // refuses an index past the end, so a request longer than the buffer
        // is an error rather than a write past it. The cdata arm stays
        // unchecked: a bare pointer carries no length.
        if n > slice.len() {
            return Err(PyError::value_error(
                "destination buffer is smaller than the requested size",
            ));
        }
        slice.as_mut_ptr()
    };
    let mut source_buffer = None;
    let src = if let Some(cdata) = W_CData::from_obj(roots.get(src_slot)) {
        unsafe_escaping_ptr_for_ptr_or_array(cdata)?.cast_const()
    } else {
        let Some(buffer) =
            pyre_interpreter::baseobjspace::simple_buffer_bytes(roots.get(src_slot))?
        else {
            return Err(PyError::type_error("expected a readable buffer"));
        };
        // `getslice(0, 1, n)` bounds the same read upstream.
        if n > buffer.as_bytes().len() {
            buffer.release();
            return Err(PyError::value_error(
                "source buffer is smaller than the requested size",
            ));
        }
        let ptr = buffer.as_bytes().as_ptr();
        source_buffer = Some(buffer);
        ptr
    };
    unsafe { std::ptr::copy(src, dest, n) };
    if let Some(buffer) = source_buffer {
        buffer.release();
    }
    drop(dest_buffer);
    Ok(pyre_object::w_none())
}

/// `func.py unsafe_escaping_ptr_for_ptr_or_array`.
fn unsafe_escaping_ptr_for_ptr_or_array(cdata: &W_CData) -> Result<*mut u8, PyError> {
    let ct = ctypeobj::ctype_at(cdata.ctype)
        .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
    if !ct.has(ctypeobj::CTypeFlags::NONFUNC_POINTER_OR_ARRAY) {
        return Err(PyError::type_error(format!(
            "expected a pointer or array ctype, got '{}'",
            ct.name()
        )));
    }
    Ok(cdata.ptr as *mut u8)
}

/// `func.py release` — `W_CData.enter_exit(exit_now=True)`.
pub fn release(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    cdataobj::enter_exit(args[0], true)?;
    Ok(pyre_object::w_none())
}

/// `func.py _get_types` — the two types cffi's Python half checks against.
pub fn get_types(_args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let cdata_slot = roots.base();
    let _ = roots.pin_root(cdataobj::cdata_type());
    let ctype_slot = cdata_slot + 1;
    let _ = roots.pin_root(ctypeobj::ctype_type());
    Ok(pyre_object::w_tuple_new(vec![
        roots.get(cdata_slot),
        roots.get(ctype_slot),
    ]))
}

// ── the `newtype.py` entry points ───────────────────────────────────────

/// `newtype.py new_primitive_type`.
pub fn new_primitive_type(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let name = pyre_interpreter::baseobjspace::text_w(args[0])?;
    newtype::new_primitive_type(name)
}

/// `newtype.py new_void_type`.
pub fn new_void_type(_args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(newtype::new_void_type())
}

/// `newtype.py new_pointer_type`.
pub fn new_pointer_type(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    newtype::new_pointer_type(args[0])
}

/// `newtype.py new_array_type`.
pub fn new_array_type(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let mut w_ctptr = args[0];
    let w_length = args[1];
    let length = pyre_object::with_roots!(w_ctptr => newtype::array_length_arg(w_length))?;
    newtype::new_array_type(w_ctptr, length)
}

/// `newtype.py new_struct_type`.
pub fn new_struct_type(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(newtype::new_struct_type(
        pyre_interpreter::baseobjspace::text_w(args[0])?,
    ))
}

/// `newtype.py new_union_type`.
pub fn new_union_type(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(newtype::new_union_type(
        pyre_interpreter::baseobjspace::text_w(args[0])?,
    ))
}

/// `newtype.py complete_struct_or_union` — the third argument is ignored, as
/// its name says, and the last four carry defaults.
pub fn complete_struct_or_union(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_entry_point(
        args,
        "complete_struct_or_union",
        &[
            "ctype",
            "fields",
            "ignored",
            "totalsize",
            "totalalignment",
            "sflags",
            "pack",
        ],
        2,
    )?;
    let int_arg = |i: usize, default: i64| -> Result<i64, PyError> {
        match optional(&a, i) {
            Some(w) => pyre_interpreter::baseobjspace::int_w(w),
            None => Ok(default),
        }
    };
    let mut w_ctype = a[0];
    let mut w_fields = a[1];
    let totalsize = pyre_object::with_roots!(w_ctype, w_fields => int_arg(3, -1))?;
    let totalalignment = pyre_object::with_roots!(w_ctype, w_fields => int_arg(4, -1))?;
    let sflags = pyre_object::with_roots!(w_ctype, w_fields => int_arg(5, 0))?;
    let pack = pyre_object::with_roots!(w_ctype, w_fields => int_arg(6, 0))?;
    newtype::complete_struct_or_union(w_ctype, w_fields, totalsize, totalalignment, sflags, pack)?;
    Ok(pyre_object::w_none())
}

/// `newtype.py new_enum_type`.
pub fn new_enum_type(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_name = args[0];
    let mut w_enumerators = args[1];
    let mut w_enumvalues = args[2];
    let mut w_basectype = args[3];
    let name = pyre_object::with_roots!(w_enumerators, w_enumvalues, w_basectype =>
        pyre_interpreter::baseobjspace::text_w(w_name)
    )?;
    newtype::new_enum_type(name, w_enumerators, w_enumvalues, w_basectype)
}

/// `newtype.py new_function_type`.
pub fn new_function_type(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_entry_point(
        args,
        "new_function_type",
        &["fargs", "fresult", "ellipsis", "abi"],
        2,
    )?;
    let ellipsis = match optional(&a, 2) {
        Some(w) => pyre_interpreter::baseobjspace::int_w(w)? != 0,
        None => false,
    };
    let abi = match optional(&a, 3) {
        Some(w) => pyre_interpreter::baseobjspace::int_w(w)?,
        None => super::interp_cffi_backend::default_abi() as i64,
    };
    newtype::new_function_type(a[0], a[1], ellipsis, abi)
}

/// `libraryobj.py load_library`.
pub fn load_library(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let a = bind_entry_point(args, "load_library", &["filename", "flags"], 1)?;
    let flags = match optional(&a, 1) {
        Some(w) => pyre_interpreter::baseobjspace::int_w(w)?,
        None => 0,
    };
    super::libraryobj::load_library(a[0], flags)
}
