//! `_cffi_backend._CDataBase` — PyPy:
//! `pypy/module/_cffi_backend/cdataobj.py`.
//!
//! `W_CData.typedef.acceptable_as_base_class = False` and none of the
//! subclasses (`W_CDataMem`, `W_CDataNewStd`, `W_CDataSliced`, …) declares a
//! typedef of its own, so the family is one Python type here as well; the
//! subclass is carried as [`W_CData::flavor`].

use pyre_interpreter::PyError;
use pyre_object::PyObjectRef;
use rustpython_wtf8::Wtf8Buf;

use super::ctypeobj::{self, W_CType};
use super::misc;

// ── the subclass discriminant ───────────────────────────────────────────

/// `cdataobj.py W_CData` itself: a view that owns nothing.
pub const FLAVOR_STATIC: i64 = 0;
/// `W_CDataMem` — the buffer `cast()` allocates for a primitive value.
pub const FLAVOR_MEM: i64 = 1;
/// `W_CDataNewStd` — what `newp()` returns through the default allocator.
pub const FLAVOR_NEW_STD: i64 = 2;
/// `W_CDataSliced` — a slice of an array or pointer, with its own length.
pub const FLAVOR_SLICED: i64 = 3;
/// `W_CDataPtrToStructOrUnion` — what `newp()` returns for a `struct *`: it
/// owns nothing itself and co-owns the `W_CDataNewStd` holding the struct.
pub const FLAVOR_PTR_TO_STRUCT: i64 = 4;
/// `W_CDataHandle` — a non-moving cdata whose address hides a Python object.
pub const FLAVOR_HANDLE: i64 = 5;
/// `W_CDataFromBuffer` — a borrowed raw buffer with a retained export.
pub const FLAVOR_FROM_BUFFER: i64 = 6;
/// `W_CDataGCP` — a cdata view with an application-level destructor.
pub const FLAVOR_GCP: i64 = 7;
/// `W_CDataNewNonStd` — memory returned by an application allocator.
pub const FLAVOR_NEW_NONSTD: i64 = 8;
/// `W_CDataCallback` — a libffi closure calling an application-level object.
pub const FLAVOR_CALLBACK: i64 = 9;

/// `cdataobj.py W_CData` and the RPython subclasses sharing its typedef.
#[pyre_interpreter::pyre_class("_cffi_backend._CDataBase", weakrefable)]
// `W_CData._immutable_fields_` names `_ptr` and `ctype`.  Declaring `ctype`
// is what lets a call through a constant cdata fold its function type, and
// with it the `cif_descr` behind it; declaring `ptr` folds the address the
// call goes to.  The only write to `ptr` after construction is the one
// `w_cdata_dealloc` makes on a GC-dead object, which no live reference can
// read back.
#[majit_macros::jit_immutable_fields("ctype", "ptr")]
#[derive(Default)]
pub struct W_CData {
    /// `W_CData.ctype`.
    pub ctype: PyObjectRef,
    /// `W_CData._ptr` — `rffi.CCHARP`.  A raw-flavour pointer is
    /// integer-kind, so the address belongs in the integer register bank
    /// rather than among the references a collector traces and rewrites.
    pub ptr: usize,
    /// Which RPython subclass this is; one of the `FLAVOR_*`.
    pub flavor: i64,
    /// `W_CDataNewOwning.allocated_length` for an owning cdata,
    /// `W_CDataSliced.length` for a slice, and the raw callback side-block
    /// address for `W_CDataCallback`.  -1 when none applies.
    pub length: i64,
    /// `W_CDataNewStd.datasize`, which becomes -1 once the memory is freed.
    /// A `W_CDataFromBuffer` keeps no size here and uses the slot for whether
    /// it still holds the export it must decref, which `release` clears.
    pub datasize: i64,
    /// Whatever the cdata must keep alive: `W_CDataPtrToStructOrUnion`'s
    /// `structobj`, `W_CDataFromBuffer`'s `w_keepalive`, `W_CDataGCP`'s
    /// `w_original_cdata`, or `W_CDataCallback.w_callable`.
    pub w_keepalive: PyObjectRef,
    /// `W_CDataGCP.w_destructor`, the object a `W_CDataFromBuffer` took its
    /// export from, or `W_CDataCallback.w_onerror`.
    pub w_destructor: PyObjectRef,
    /// The translated `special_memory_pressure` field installed for classes
    /// passed as the object argument to `rgc.add_memory_pressure`.
    pub special_memory_pressure: i64,
}

impl W_CData {
    // Spelled as a `match` rather than `Option::ok_or_else`: the closure the
    // combinator takes is a callee of its own, and a traced `W_CData.call`
    // would stop at it.
    fn ctype_ref(&self) -> Result<&'static mut W_CType, PyError> {
        match ctypeobj::ctype_at(self.ctype) {
            Some(ct) => Ok(ct),
            None => Err(PyError::system_error("cdata without a ctype")),
        }
    }

    /// `W_CData._sizeof` and the overrides of it.
    fn sizeof(&self) -> Result<i64, PyError> {
        let ct = self.ctype_ref()?;
        match self.flavor {
            FLAVOR_NEW_STD | FLAVOR_NEW_NONSTD if self.length >= 0 => {
                if ct.kind == ctypeobj::KIND_ARRAY {
                    let item = ctypeobj::ctype_at(ct.ctitem)
                        .ok_or_else(|| PyError::system_error("array without an item type"))?;
                    Ok(self.length * item.size)
                } else {
                    // A var-sized struct records its total size directly.
                    Ok(self.length)
                }
            }
            FLAVOR_SLICED | FLAVOR_FROM_BUFFER if ct.kind == ctypeobj::KIND_ARRAY => {
                let item = ctypeobj::ctype_at(ct.ctitem)
                    .ok_or_else(|| PyError::system_error("slice without an item type"))?;
                Ok(self.length * item.size)
            }
            _ => Ok(ct.size),
        }
    }

    /// `W_CData.get_array_length`.
    pub fn array_length(&self) -> Result<i64, PyError> {
        match self.flavor {
            FLAVOR_NEW_STD | FLAVOR_NEW_NONSTD | FLAVOR_SLICED | FLAVOR_FROM_BUFFER => {
                Ok(self.length)
            }
            _ => {
                let ct = self.ctype_ref()?;
                Ok(ct.length)
            }
        }
    }

    /// `W_CData._repr_extra` and the overrides of it.
    pub fn repr_extra(&self) -> Result<Wtf8Buf, PyError> {
        let ct = self.ctype_ref()?;
        match self.flavor {
            FLAVOR_NEW_STD | FLAVOR_NEW_NONSTD => {
                // `W_CData._repr_extra_owning`: a pointer reports the size of
                // what it points at, not the size of the pointer.
                let bytes = if ct.kind == ctypeobj::KIND_POINTER {
                    ctypeobj::ctype_at(ct.ctitem)
                        .ok_or_else(|| PyError::system_error("pointer without an item type"))?
                        .size
                } else {
                    self.sizeof()?
                };
                Ok(Wtf8Buf::from(format!("owning {bytes} bytes")))
            }
            FLAVOR_SLICED => Ok(Wtf8Buf::from(format!("sliced length {}", self.length))),
            // `W_CDataPtrToStructOrUnion._repr_extra`.
            FLAVOR_PTR_TO_STRUCT => match W_CData::from_obj(self.w_keepalive) {
                Some(structobj) => structobj.repr_extra(),
                None => Ok(Wtf8Buf::from("NULL")),
            },
            // `W_CDataHandle._repr_extra`. The keepalive repr is `text_w`:
            // a lone surrogate stays in the spelling.
            FLAVOR_HANDLE => {
                let roots = pyre_object::gc_roots::push_roots();
                let keepalive_slot = roots.base();
                let _ = roots.pin_root(self.w_keepalive);
                let w_repr =
                    pyre_interpreter::builtins::builtin_repr(&[roots.get(keepalive_slot)])?;
                let mut msg = Wtf8Buf::from("handle to ");
                msg.push_wtf8(unsafe { pyre_object::w_str_get_wtf8(w_repr) });
                Ok(msg)
            }
            // `W_CDataFromBuffer._repr_extra`.
            FLAVOR_FROM_BUFFER => {
                if self.w_keepalive.is_null() {
                    return Ok(Wtf8Buf::from("buffer RELEASED"));
                }
                let type_name = pyre_interpreter::type_methods::arg_type_name(self.w_keepalive);
                if ct.kind == ctypeobj::KIND_ARRAY {
                    Ok(Wtf8Buf::from(format!(
                        "buffer len {} from '{}' object",
                        self.length, type_name
                    )))
                } else {
                    Ok(Wtf8Buf::from(format!("buffer from '{}' object", type_name)))
                }
            }
            // `W_ExternPython._repr_extra`.
            FLAVOR_CALLBACK => {
                let roots = pyre_object::gc_roots::push_roots();
                let callable_slot = roots.base();
                let _ = roots.pin_root(self.w_keepalive);
                let w_repr = pyre_interpreter::builtins::builtin_repr(&[roots.get(callable_slot)])?;
                let mut msg = Wtf8Buf::from("calling ");
                msg.push_wtf8(unsafe { pyre_object::w_str_get_wtf8(w_repr) });
                Ok(msg)
            }
            _ => Ok(Wtf8Buf::from(unsafe {
                ct.extra_repr(self.ptr as *const u8)
            }?)),
        }
    }
}

/// The receiver of a traced `W_CData.call` whose ctype is a function pointer.
///
/// `W_CTypeFunc.call` promotes `self` once `cif_descr` is set, so the
/// generated descent follows that arm. The body-wide scan still joins the
/// other ctype kinds and sees `allocate_stable` there.
pub fn is_function_pointer_cdata(obj: PyObjectRef) -> bool {
    let Some(cdata) = W_CData::from_obj(obj) else {
        return false;
    };
    ctypeobj::ctype_at(cdata.ctype).is_some_and(|ct| ct.kind == ctypeobj::KIND_FUNC)
}

/// `@unwrap_spec(w_cdata=cdataobj.W_CData)`.
pub fn cdata_arg(w_cdata: PyObjectRef) -> Result<&'static mut W_CData, PyError> {
    match W_CData::from_obj(w_cdata) {
        Some(cdata) => Ok(cdata),
        None => Err(unsafe { PyError::from_exc_object(expected_cdata_object(w_cdata)) }),
    }
}

/// `@unwrap_spec(w_cdata=cdataobj.W_CData)` miss — `oefmt`.
#[majit_macros::dont_look_inside]
pub(crate) fn expected_cdata_object(w_got: PyObjectRef) -> PyObjectRef {
    PyError::type_error(format!(
        "expected a cdata object, got '{}'",
        pyre_interpreter::type_methods::arg_type_name(w_got)
    ))
    .to_exc_object()
}

// ── construction ────────────────────────────────────────────────────────

/// `cdataobj.py W_CData(space, ptr, ctype)` — a view that owns no memory.
pub fn new_cdata(ptr: usize, w_ctype: PyObjectRef) -> PyObjectRef {
    new_cdata_full(ptr, w_ctype, FLAVOR_STATIC, -1, -1, pyre_object::PY_NULL)
}

/// `cdataobj.py W_CDataSliced(space, ptr, ctype, length)`.
pub fn new_cdata_sliced(ptr: usize, w_ctype: PyObjectRef, length: i64) -> PyObjectRef {
    new_cdata_full(
        ptr,
        w_ctype,
        FLAVOR_SLICED,
        length,
        -1,
        pyre_object::PY_NULL,
    )
}

/// `cdataobj.py W_CDataMem(space, ctype)` — `ctype.size` bytes of raw memory
/// this cdata owns and frees.
pub fn new_cdata_mem(w_ctype: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let size = ctypeobj::ctype_arg(w_ctype)?.size.max(0);
    let ptr = raw_alloc(size, false)?;
    Ok(new_cdata_full(
        ptr,
        w_ctype,
        FLAVOR_MEM,
        -1,
        size,
        pyre_object::PY_NULL,
    ))
}

/// `W_CDataPtrToStructOrUnion.__init__` — a pointer that co-owns the cdata
/// really holding the struct.
pub fn new_cdata_ptr_to_struct(
    ptr: usize,
    w_ctype: PyObjectRef,
    w_structobj: PyObjectRef,
) -> PyObjectRef {
    new_cdata_full(ptr, w_ctype, FLAVOR_PTR_TO_STRUCT, -1, -1, w_structobj)
}

/// `W_CDataHandle.__init__`, together with the `instantiate` /
/// `hide_nonmovable_gcref` pair `newp_handle` performs ahead of it:
/// `allocate_stable` supplies the non-moving address `hide_object` exposes,
/// and the pointer is that address, so it is filled in after the allocation
/// the way the late `W_CData.__init__` call does upstream.
pub fn new_cdata_handle(w_ctype: PyObjectRef, w_keepalive: PyObjectRef) -> PyObjectRef {
    let obj = new_cdata_full(0, w_ctype, FLAVOR_HANDLE, -1, -1, w_keepalive);
    W_CData::from_obj(obj)
        .expect("new_cdata_full returns a cdata")
        .ptr = super::hide_reveal::hide_object(obj) as usize;
    obj
}

/// `W_CDataCallback.__init__`'s cdata half.  The callback flavor reuses
/// `w_keepalive` for `w_callable`, `w_destructor` for `w_onerror`, and
/// `length` for the raw side-block address.
pub fn new_cdata_callback(
    ptr: usize,
    w_ctype: PyObjectRef,
    w_callable: PyObjectRef,
    w_onerror: PyObjectRef,
    raw_side_block: i64,
) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let onerror_slot = roots.base();
    let _ = roots.pin_root(w_onerror);
    let obj = new_cdata_full(
        ptr,
        w_ctype,
        FLAVOR_CALLBACK,
        raw_side_block,
        -1,
        w_callable,
    );
    W_CData::from_obj(obj)
        .expect("new_cdata_full returns a cdata")
        .w_destructor = roots.get(onerror_slot);
    pyre_object::gc_hook::try_gc_write_barrier_managed(obj as pyre_object::gc_hook::GCREF);
    obj
}

/// `W_CDataFromBuffer.__init__`.
pub fn new_cdata_from_buffer(
    ptr: usize,
    length: i64,
    w_ctype: PyObjectRef,
    w_keepalive: PyObjectRef,
    w_export_owner: PyObjectRef,
    export_held: bool,
) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let owner_slot = roots.base();
    let _ = roots.pin_root(w_export_owner);
    let obj = new_cdata_full(
        ptr,
        w_ctype,
        FLAVOR_FROM_BUFFER,
        length,
        i64::from(export_held),
        w_keepalive,
    );
    W_CData::from_obj(obj)
        .expect("new_cdata_full returns a cdata")
        .w_destructor = roots.get(owner_slot);
    pyre_object::gc_hook::try_gc_write_barrier_managed(obj as pyre_object::gc_hook::GCREF);
    if export_held {
        pyre_interpreter::executioncontext::register_finalizer(obj);
    }
    obj
}

/// `W_CDataGCP.__init__`.
pub fn new_cdata_gcp(
    ptr: usize,
    w_ctype: PyObjectRef,
    w_original: PyObjectRef,
    w_destructor: PyObjectRef,
) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let destructor_slot = roots.base();
    let _ = roots.pin_root(w_destructor);
    let obj = new_cdata_full(ptr, w_ctype, FLAVOR_GCP, -1, -1, w_original);
    W_CData::from_obj(obj)
        .expect("new_cdata_full returns a cdata")
        .w_destructor = roots.get(destructor_slot);
    pyre_object::gc_hook::try_gc_write_barrier_managed(obj as pyre_object::gc_hook::GCREF);
    pyre_interpreter::executioncontext::register_finalizer(obj);
    obj
}

/// `W_CDataNewNonStd.__init__`.
pub fn new_cdata_nonstd(
    ptr: usize,
    w_ctype: PyObjectRef,
    length: i64,
    w_raw_cdata: PyObjectRef,
    w_free: PyObjectRef,
) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let free_slot = roots.base();
    let _ = roots.pin_root(w_free);
    let obj = new_cdata_full(ptr, w_ctype, FLAVOR_NEW_NONSTD, length, -1, w_raw_cdata);
    W_CData::from_obj(obj)
        .expect("new_cdata_full returns a cdata")
        .w_destructor = roots.get(free_slot);
    pyre_object::gc_hook::try_gc_write_barrier_managed(obj as pyre_object::gc_hook::GCREF);
    if !roots.get(free_slot).is_null() {
        pyre_interpreter::executioncontext::register_finalizer(obj);
    }
    obj
}

/// `W_CTypeStructOrUnion.copy_and_convert_to_object` — a `W_CDataNewStd`
/// holding a copy of what `source` points at.
///
/// # Safety
/// `source` must be readable for `size` bytes.
pub unsafe fn new_cdata_copy(
    w_ctype: PyObjectRef,
    source: *const u8,
    size: i64,
) -> Result<PyObjectRef, PyError> {
    let ptr = raw_alloc(size, false)?;
    misc::raw_memcopy(source as usize, ptr, size.max(0) as usize);
    Ok(new_cdata_full(
        ptr,
        w_ctype,
        FLAVOR_NEW_STD,
        -1,
        size,
        pyre_object::PY_NULL,
    ))
}

/// `W_CData.get_structobj` and the two overrides of it — the owning cdata a
/// variable-length array reads its bound from.
pub fn structobj_of(w_cdata: PyObjectRef) -> Option<&'static mut W_CData> {
    let cdata = W_CData::from_obj(w_cdata)?;
    match cdata.flavor {
        FLAVOR_NEW_STD | FLAVOR_NEW_NONSTD => Some(cdata),
        FLAVOR_PTR_TO_STRUCT => W_CData::from_obj(cdata.w_keepalive)
            .filter(|s| matches!(s.flavor, FLAVOR_NEW_STD | FLAVOR_NEW_NONSTD)),
        _ => None,
    }
}

/// `allocator.py default_allocator.allocate` — zeroed memory owned by the
/// cdata it hands back.
pub fn new_cdata_owning(
    w_ctype: PyObjectRef,
    datasize: i64,
    length: i64,
) -> Result<PyObjectRef, PyError> {
    let ptr = raw_alloc(datasize, true)?;
    let obj = new_cdata_full(
        ptr,
        w_ctype,
        FLAVOR_NEW_STD,
        length,
        datasize,
        pyre_object::PY_NULL,
    );
    add_memory_pressure(obj, datasize);
    Ok(obj)
}

fn new_cdata_full(
    ptr: usize,
    w_ctype: PyObjectRef,
    flavor: i64,
    length: i64,
    datasize: i64,
    w_keepalive: PyObjectRef,
) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let ctype_slot = roots.base();
    let _ = roots.pin_root(w_ctype);
    let keepalive_slot = ctype_slot + 1;
    let _ = roots.pin_root(w_keepalive);
    // The struct literal is built before `allocate_stable` runs, and that
    // allocation is itself a collection point, so the movable keepalive is
    // stored from its slot afterwards rather than from the literal.
    let obj = W_CData::allocate_stable(W_CData {
        ctype: roots.get(ctype_slot),
        ptr,
        flavor,
        length,
        datasize,
        ..Default::default()
    });
    W_CData::from_obj(obj)
        .expect("allocate_stable hands back this layout")
        .w_keepalive = roots.get(keepalive_slot);
    // The cdata is born old-gen; the keepalive it just took may be young, so
    // the barrier has to run again after this write.
    pyre_object::gc_hook::try_gc_write_barrier_managed(obj as pyre_object::gc_hook::GCREF);
    obj
}

/// `W_CDataNewStd.__init__` for an allocator-selected raw block.
pub fn new_cdata_full_for_allocator(
    ptr: usize,
    w_ctype: PyObjectRef,
    length: i64,
    datasize: i64,
) -> PyObjectRef {
    let obj = new_cdata_full(
        ptr,
        w_ctype,
        FLAVOR_NEW_STD,
        length,
        datasize,
        pyre_object::PY_NULL,
    );
    add_memory_pressure(obj, datasize);
    obj
}

/// `rgc.add_memory_pressure(size, cdata)`.
pub fn add_memory_pressure(w_cdata: PyObjectRef, size: i64) {
    if size != 0 {
        majit_gc::add_memory_pressure(size as isize, majit_ir::GcRef(w_cdata as usize));
    }
}

/// `lltype.malloc(rffi.CCHARP.TO, size, flavor='raw')`.  A zero-size request
/// still returns a distinct address, so `newp` on an empty array does not
/// hand back NULL.
/// `lltype.malloc(rffi.CCHARP.TO, size, flavor='raw')` — the raw block a
/// traced caller allocates; the JIT sees it as the `raw_malloc_varsize_char`
/// residual (`support.py ll_raw_malloc_varsize_char`).  Zero on exhaustion;
/// the caller raises the MemoryError.
///
/// The whole raw-memory family below spells its addresses `usize` rather
/// than `*mut u8`.  `getkind(Ptr(TO))` answers `int` whenever `TO._gckind`
/// is `raw` (`history.py`), which is what puts a raw block address in the
/// integer register bank and lets `raw_ptradd` be an `int_add` over it.
/// `*mut u8` cannot carry that: it is this tree's erased spelling of a
/// *managed* object — the parameter type of every GC hook, the return type
/// of every GC allocator, and the pointee of every root slot — so it banks
/// as a reference the collector traces and rewrites.  A raw block is
/// neither traced nor moved, and the two must not share a spelling.
#[inline(never)]
#[majit_macros::oopspec("raw_malloc_varsize_char(size)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_malloc_varsize_char(size: usize) -> usize {
    unsafe { libc::malloc(size.max(1)) as usize }
}

/// `lltype.malloc(..., flavor='raw', zero=True)` — the
/// `_ll_1_raw_malloc_varsize_zero` residual. Same
/// `OS_RAW_MALLOC_VARSIZE_CHAR` as the non-zero Char helper
/// (`jtransform.py _rewrite_raw_malloc` when `TYPE.OF == Char`).
#[inline(never)]
#[majit_macros::oopspec("raw_malloc_varsize_zero(size)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_malloc_varsize_zero(size: usize) -> usize {
    unsafe { libc::calloc(size.max(1), 1) as usize }
}

/// `lltype.free(ptr, flavor='raw')` — the `raw_free` residual
/// (`support.py ll_raw_free`).
#[inline(never)]
#[majit_macros::oopspec("raw_free(ptr)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_free(ptr: usize) {
    unsafe { libc::free(ptr as *mut libc::c_void) }
}

/// `rffi.ptradd` — advance a raw byte address without a memory access
/// (`direct_ptradd` residual shape).
#[majit_macros::oopspec("raw_ptradd(ptr, offset)")]
#[majit_macros::elidable_cannot_raise]
pub fn raw_ptradd(ptr: usize, offset: usize) -> usize {
    ptr.wrapping_add(offset)
}

/// One pointer-sized load out of an exchange-buffer argument slot
/// (`rffi.cast(rffi.CCHARPP, data)[0]`).
#[majit_macros::oopspec("raw_read_ptr(data)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_read_ptr(data: usize) -> usize {
    unsafe { (data as *const usize).read_unaligned() }
}

/// One pointer-sized store into an exchange-buffer argument slot
/// (`rffi.cast(rffi.CCHARPP, data)[0] = value`).
#[majit_macros::oopspec("raw_write_ptr(data, value)")]
#[majit_macros::dont_look_inside_cannot_raise]
pub fn raw_write_ptr(data: usize, value: usize) {
    unsafe { (data as *mut usize).write_unaligned(value) }
}

pub fn raw_alloc(size: i64, zero: bool) -> Result<usize, PyError> {
    if size < 0 {
        return Err(PyError::value_error("negative allocation size"));
    }
    let bytes = size.max(1) as usize;
    // `support.py _ll_1_raw_malloc_varsize[_zero]`: the C leaf stays
    // inside the residual. Inlining `raw_alloc` onto `libc::calloc`
    // made the descent scan decline on that un-lowered helper.
    let ptr = if zero {
        raw_malloc_varsize_zero(bytes)
    } else {
        raw_malloc_varsize_char(bytes)
    };
    if ptr == 0 {
        return Err(PyError::new(
            pyre_interpreter::PyErrorKind::MemoryError,
            "out of memory",
        ));
    }
    Ok(ptr)
}

/// `lltype.free(self._ptr, flavor='raw')` in the light finalizers of
/// `W_CDataMem` and `W_CDataNewStd`, and `Closure.__del__` for a callback.
///
/// # Safety
/// `obj` must be a GC-dead `W_CData`.
pub unsafe fn w_cdata_dealloc(obj: PyObjectRef) {
    let cdata = unsafe { &mut *(obj as *mut W_CData) };
    if cdata.flavor == FLAVOR_CALLBACK {
        // Clear the slot before releasing it: the release is a `Box::from_raw`,
        // so a second sweep of the same object must find nothing left to free.
        let side_block = std::mem::replace(&mut cdata.length, 0);
        unsafe { super::ccallback::free_callback_side_block(side_block, cdata.ptr as *mut u8) };
    } else if matches!(cdata.flavor, FLAVOR_MEM | FLAVOR_NEW_STD) && cdata.datasize >= 0 {
        unsafe { libc::free(cdata.ptr as *mut libc::c_void) };
    }
    cdata.ptr = 0;
    cdata.datasize = -1;
}

/// `W_CDataFromBuffer._finalize_`, `W_CDataGCP._finalize_`, and
/// `W_CDataNewNonStd._finalize_`.
///
/// This runs from the interpreter finalizer queue, where calling Python is
/// permitted.  It is separate from [`w_cdata_dealloc`], the allocation-free
/// sweep destructor.
pub fn finalize(mut w_cdata: PyObjectRef) -> Result<(), PyError> {
    let flavor = pyre_object::with_roots!(w_cdata => cdata_arg(w_cdata))?.flavor;
    match flavor {
        FLAVOR_FROM_BUFFER => release_buffer_export(w_cdata),
        FLAVOR_GCP | FLAVOR_NEW_NONSTD => invoke_destructor(w_cdata),
        _ => Ok(()),
    }
}

/// `Some(calls_python)` when `obj` is a buffer/GCP/nonstd cdata this path
/// handles. `None` leaves the object to the rest of the finalizer queue.
pub fn ec_finalizer_kind(obj: PyObjectRef) -> Option<bool> {
    let cdata = W_CData::from_obj(obj)?;
    if !matches!(
        cdata.flavor,
        FLAVOR_FROM_BUFFER | FLAVOR_GCP | FLAVOR_NEW_NONSTD
    ) {
        return None;
    }
    Some(matches!(cdata.flavor, FLAVOR_GCP | FLAVOR_NEW_NONSTD))
}

/// Run [`finalize`] and write an unraisable if it raises.
pub fn run_ec_finalize(mut obj: PyObjectRef) {
    if let Err(mut error) = pyre_object::with_roots!(obj => finalize(obj)) {
        error.write_unraisable(
            pyre_object::w_none(),
            rustpython_wtf8::Wtf8::new("Exception ignored in cffi destructor"),
            obj,
        );
    }
}

/// `W_CDataGCP.invoke_finalizer` and `W_CDataNewNonStd._do_exit`.
fn invoke_destructor(w_cdata: PyObjectRef) -> Result<(), PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let cdata_slot = roots.base();
    let _ = roots.pin_root(w_cdata);
    let destructor_slot = cdata_slot + 1;
    let _ = roots.pin_root(cdata_arg(roots.get(cdata_slot))?.w_destructor);
    if roots.get(destructor_slot).is_null() {
        return Ok(());
    }
    // Clear first: a recursive release or a resurrected cdata must not call
    // the destructor twice.
    cdata_arg(roots.get(cdata_slot))?.w_destructor = pyre_object::PY_NULL;
    let original = cdata_arg(roots.get(cdata_slot))?.w_keepalive;
    let original_slot = destructor_slot + 1;
    let _ = roots.pin_root(original);
    pyre_object::gc_hook::try_gc_write_barrier_managed(
        roots.get(cdata_slot) as pyre_object::gc_hook::GCREF
    );
    pyre_interpreter::call::call_function_impl_result(
        roots.get(destructor_slot),
        &[roots.get(original_slot)],
    )?;
    Ok(())
}

/// `W_CDataFromBuffer.enter_exit`'s export release.
fn release_buffer_export(mut w_cdata: PyObjectRef) -> Result<(), PyError> {
    let cdata = pyre_object::with_roots!(w_cdata => cdata_arg(w_cdata))?;
    if cdata.datasize != 0 && !cdata.w_destructor.is_null() {
        unsafe { pyre_interpreter::builtins::buffer_export_decref(cdata.w_destructor) };
        cdata.datasize = 0;
    }
    cdata.w_keepalive = pyre_object::PY_NULL;
    cdata.w_destructor = pyre_object::PY_NULL;
    pyre_object::gc_hook::try_gc_write_barrier_managed(w_cdata as pyre_object::gc_hook::GCREF);
    Ok(())
}

// ── the Python type ─────────────────────────────────────────────────────

static CDATA_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

/// `_cffi_backend._CDataBase`.
pub fn cdata_type() -> PyObjectRef {
    CDATA_TYPE_OBJ.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type_with_layout(
            "_cffi_backend._CDataBase",
            init_cdata_type,
            pyre_interpreter::typedef::w_object(),
            <W_CData as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_CData as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        tp
    })
}

const CDATA_DOC: &str = "The internal base type for CData objects.  Use FFI.CData to access it.  Always check with isinstance(): subtypes are sometimes returned on CPython, for performance reasons.";

fn init_cdata_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    let store =
        |name: &str, value: PyObjectRef| pyre_interpreter::__pyre_put_new!(ns_slot, name, value);
    store("__doc__", pyre_object::w_str_new(CDATA_DOC));
    // Both are typedef entries in PyPy, so they answer on an instance too.
    store("__module__", pyre_object::w_str_new("_cffi_backend"));
    store("__name__", pyre_object::w_str_new("<cdata>"));
    store(
        "__weakref__",
        pyre_interpreter::typedef::make_weakref_descr(pyre_object::gc_roots::shadow_stack_get(
            ns_slot,
        )),
    );
    for (name, f, arity) in [
        (
            "__repr__",
            cdata_repr as pyre_interpreter::gateway::BuiltinCodeFn,
            1u16,
        ),
        ("__bool__", cdata_bool, 1),
        ("__int__", cdata_int, 1),
        ("__float__", cdata_float, 1),
        ("__complex__", cdata_complex, 1),
        ("__len__", cdata_len, 1),
        ("__hash__", cdata_hash, 1),
        ("__dir__", cdata_dir, 1),
        ("__enter__", cdata_enter, 1),
        ("__iter__", cdata_iter, 1),
        ("__lt__", cdata_lt, 2),
        ("__le__", cdata_le, 2),
        ("__eq__", cdata_eq, 2),
        ("__ne__", cdata_ne, 2),
        ("__gt__", cdata_gt, 2),
        ("__ge__", cdata_ge, 2),
        ("__getitem__", cdata_getitem, 2),
        ("__add__", cdata_add, 2),
        ("__radd__", cdata_add, 2),
        ("__sub__", cdata_sub, 2),
        ("__getattr__", cdata_getattr, 2),
        ("__setitem__", cdata_setitem, 3),
        ("__setattr__", cdata_setattr, 3),
    ] {
        store(
            name,
            pyre_interpreter::make_builtin_function_with_arity(name, f, arity),
        );
    }
    // `__call__` and `__exit__` take a variable number of arguments.
    store(
        "__call__",
        pyre_interpreter::make_builtin_function("__call__", __majit_wrap_cdata_call),
    );
    store(
        "__exit__",
        pyre_interpreter::make_builtin_function("__exit__", cdata_exit),
    );
}

/// `W_CData.repr`.
fn cdata_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    let extra = cdata.repr_extra()?;
    // A struct this cdata does not own is written "struct foo &", so that it
    // does not read like the owning "<cdata 'struct foo' 0x...>".
    let owning = matches!(cdata.flavor, FLAVOR_NEW_STD | FLAVOR_NEW_NONSTD);
    let extra1 = if !owning && ct.is_struct_or_union() {
        " &"
    } else {
        ""
    };
    Ok(pyre_object::w_str_from_wtf8_managed(
        pyre_interpreter::display::wtf8_format!("<cdata '", ct.name(), extra1, "' ", extra, ">"),
    ))
}

/// `W_CData.bool`.
fn cdata_bool(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    Ok(pyre_object::boolobject::w_bool_from(unsafe {
        ctypeobj::nonzero(ct, cdata.ptr as *const u8)?
    }))
}

/// `W_CData.int`.
fn cdata_int(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    unsafe { ctypeobj::cast_to_int(ct, cdata.ptr as *const u8) }
}

/// `W_CData.float`.
fn cdata_float(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    unsafe { ctypeobj::float(ct, cdata.ptr as *const u8) }
}

/// `W_CData.complex`.
fn cdata_complex(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    unsafe { ctypeobj::complex(ct, cdata.ptr as *const u8) }
}

/// `W_CData.len`.
fn cdata_len(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    if ct.kind != ctypeobj::KIND_ARRAY {
        return Err(PyError::type_error(format!(
            "cdata of type '{}' has no len()",
            ct.name()
        )));
    }
    Ok(pyre_object::w_int_new(cdata.array_length()?))
}

/// `W_CData.hash`.
fn cdata_hash(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    if ct.is_primitive() {
        let w_ob = unsafe { ctypeobj::convert_to_object(ct, cdata.ptr)? };
        if W_CData::from_obj(w_ob).is_none() {
            return Ok(pyre_object::w_int_new(
                pyre_interpreter::baseobjspace::hash_w_strict(w_ob)?,
            ));
        }
    }
    // Pointers are hashed by address, folded so the always-zero alignment
    // bits do not end up in every key's low bits.
    let h = cdata.ptr as i64;
    Ok(pyre_object::w_int_new(h ^ (h >> 4)))
}

/// `W_CData.dir` — the fields of what the cdata points at.
fn cdata_dir(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let mut ct = cdata.ctype_ref()?;
    if ct.kind == ctypeobj::KIND_POINTER {
        ct = ctypeobj::ctype_at(ct.ctitem)
            .ok_or_else(|| PyError::system_error("pointer without an item type"))?;
    }
    // `W_CType.cdata_dir` is empty for every ctype a struct is not.
    let names = if ct.is_struct_or_union() {
        super::ctypestruct::cdata_dir(ct)?
    } else {
        Vec::new()
    };
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for name in &names {
        let _ = roots.pin_root(pyre_object::w_str_new(name));
    }
    let items = (0..names.len()).map(|i| roots.get(base + i)).collect();
    Ok(pyre_object::w_list_new(items))
}

/// `W_CData.descr_enter`.
fn cdata_enter(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let mut w_self = args[0];
    pyre_object::with_roots!(w_self => enter_exit(w_self, false))?;
    Ok(w_self)
}

/// `W_CData.descr_exit`.
fn cdata_exit(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    enter_exit(args[0], true)?;
    Ok(pyre_object::w_none())
}

/// `W_CData.enter_exit` and the overrides of it.
pub fn enter_exit(mut w_cdata: PyObjectRef, exit_now: bool) -> Result<(), PyError> {
    let cdata = pyre_object::with_roots!(w_cdata => cdata_arg(w_cdata))?;
    // `W_CDataPtrToStructOrUnion.enter_exit` reaches the struct it co-owns
    // through `_do_exit`, not through that object's own `enter_exit`: its
    // ctype is the struct itself, which the owning check below refuses.
    if cdata.flavor == FLAVOR_PTR_TO_STRUCT {
        if exit_now && !cdata.w_keepalive.is_null() {
            return do_exit(cdata.w_keepalive);
        }
        return Ok(());
    }
    if cdata.flavor == FLAVOR_FROM_BUFFER {
        if exit_now {
            pyre_object::with_roots!(w_cdata => release_buffer_export(w_cdata))?;
            pyre_interpreter::executioncontext::may_ignore_finalizer(w_cdata);
        }
        return Ok(());
    }
    // `W_CDataGCP.enter_exit`, which carries no owning check of its own.
    if cdata.flavor == FLAVOR_GCP {
        if exit_now {
            pyre_object::with_roots!(w_cdata => invoke_destructor(w_cdata))?;
            pyre_interpreter::executioncontext::may_ignore_finalizer(w_cdata);
        }
        return Ok(());
    }
    // `W_CDataNewOwning.enter_exit`.
    let ct = cdata.ctype_ref()?;
    if !matches!(cdata.flavor, FLAVOR_NEW_STD | FLAVOR_NEW_NONSTD) || !ct.is_ptr_or_array() {
        return Err(PyError::value_error(
            "only 'cdata' object from ffi.new(), ffi.gc(), ffi.from_buffer() or ffi.new_allocator()() can be used with the 'with' keyword or ffi.release()",
        ));
    }
    if exit_now {
        do_exit(w_cdata)?;
    }
    Ok(())
}

/// `W_CDataNewStd._do_exit` and `W_CDataNewNonStd._do_exit`.  Any other
/// flavor is not a `W_CDataNewOwning`, and owns nothing to release.
fn do_exit(mut w_cdata: PyObjectRef) -> Result<(), PyError> {
    let cdata = pyre_object::with_roots!(w_cdata => cdata_arg(w_cdata))?;
    match cdata.flavor {
        FLAVOR_NEW_STD => {
            if cdata.datasize >= 0 {
                add_memory_pressure(w_cdata, -cdata.datasize);
                cdata.datasize = -1;
                pyre_interpreter::executioncontext::may_ignore_finalizer(w_cdata);
                // The freed address stays in `ptr`: reading a released
                // cdata is the caller's error, and `datasize` is what makes
                // a second release a no-op.
                unsafe { libc::free(cdata.ptr as *mut libc::c_void) };
            }
        }
        FLAVOR_NEW_NONSTD => {
            if !cdata.w_destructor.is_null() {
                add_memory_pressure(w_cdata, -cdata.sizeof()?);
            }
            pyre_object::with_roots!(w_cdata => invoke_destructor(w_cdata))?;
            pyre_interpreter::executioncontext::may_ignore_finalizer(w_cdata);
        }
        _ => {}
    }
    Ok(())
}

/// `W_CData.with_gc`.
pub fn with_gc(
    mut w_cdata: PyObjectRef,
    mut w_destructor: PyObjectRef,
    size: i64,
) -> Result<PyObjectRef, PyError> {
    let cdata = pyre_object::with_roots!(w_cdata, w_destructor => cdata_arg(w_cdata))?;
    if unsafe { pyre_object::pyobject::is_none(w_destructor) } {
        if cdata.flavor != FLAVOR_GCP {
            return Err(PyError::type_error(
                "Can remove destructor only on a object previously returned by ffi.gc()",
            ));
        }
        cdata.w_destructor = pyre_object::PY_NULL;
        pyre_object::gc_hook::try_gc_write_barrier_managed(w_cdata as pyre_object::gc_hook::GCREF);
        pyre_interpreter::executioncontext::may_ignore_finalizer(w_cdata);
        add_memory_pressure(w_cdata, size);
        return Ok(pyre_object::w_none());
    }
    let result = new_cdata_gcp(cdata.ptr, cdata.ctype, w_cdata, w_destructor);
    add_memory_pressure(result, size);
    Ok(result)
}

/// `W_CData.get_maximum_buffer_size` and its owning overrides.
pub fn maximum_buffer_size(w_cdata: PyObjectRef) -> Result<i64, PyError> {
    let cdata = cdata_arg(w_cdata)?;
    match cdata.flavor {
        FLAVOR_NEW_STD => Ok(cdata.datasize),
        FLAVOR_PTR_TO_STRUCT => match W_CData::from_obj(cdata.w_keepalive) {
            Some(structobj) => structobj.sizeof(),
            None => Ok(-1),
        },
        _ => Ok(-1),
    }
}

/// `W_CData.iter`.
fn cdata_iter(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let mut w_self = args[0];
    let cdata = pyre_object::with_roots!(w_self => cdata_arg(w_self))?;
    let ct = cdata.ctype_ref()?;
    if ct.kind != ctypeobj::KIND_ARRAY {
        return Err(PyError::type_error(format!(
            "cdata '{}' does not support iteration",
            ct.name()
        )));
    }
    super::ctypearray::new_cdata_iter(w_self)
}

/// `W_CData.call`.
///
/// Named `__majit_wrap_*` and published below: `BuiltinCode.func` is a PBC
/// whose family `builtin_wrapper_indirect_graphs` builds out of exactly the
/// wrapper paths that carry a registered graph, and a traced cdata call
/// reaches this body through `bytecode_for_address` on the function's
/// address.  Registered under any other name it has no member in that family,
/// and every `cdata(...)` stays a `bh_call_fn` residual with the argument
/// conversions and the foreign call opaque behind it.
pub fn __majit_wrap_cdata_call(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // A gateway body reads its argument count before any argument; the
    // walker locates the argument array's item descriptor by that order.
    if args.is_empty() {
        return Err(PyError::type_error("__call__ needs a cdata receiver"));
    }
    // `W_CData.call`: `with self as ptr: return self.ctype.call(ptr, args_w)`.
    // The gateway reads `len` first so the walker can name the array
    // descriptor; the pins then keep every argument across `cdata_arg`.
    let nargs = args.len();
    let roots = pyre_object::gc_roots::push_roots();
    let args_base = roots.pin_roots(args);
    let cdata = cdata_arg(roots.get(args_base))?;
    let ct = cdata.ctype_ref()?;
    if ct.kind == ctypeobj::KIND_FUNC {
        let mut rest = vec![pyre_object::PY_NULL; nargs - 1];
        pyre_object::gc_roots::shadow_stack_copy_range(args_base + 1, &mut rest);
        return super::ctypefunc::call(ct, cdata.ptr, &rest);
    }
    Err(unsafe { PyError::from_exc_object(cdata_not_callable(ct)) })
}

/// `W_CType.call` — `oefmt("cdata '%s' is not callable", self.name)`,
/// shaped as [`ctypeobj::cannot_return_cdata`].
#[majit_macros::dont_look_inside]
pub(crate) fn cdata_not_callable(ct: &W_CType) -> PyObjectRef {
    PyError::type_error(format!("cdata '{}' is not callable", ct.name())).to_exc_object()
}

pyre_interpreter::builtin_wrapper_descriptor!(
    __majit_builtin_wrapper_target_cdata_call,
    __majit_wrap_cdata_call
);

/// `W_CData.getattr`.
fn cdata_getattr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let mut w_self = args[0];
    let mut w_name = args[1];
    let cdata = pyre_object::with_roots!(w_self, w_name => cdata_arg(w_self))?;
    let ct = cdata.ctype_ref()?;
    let name = pyre_object::with_roots!(w_self => pyre_interpreter::baseobjspace::text_w(w_name))?;
    let field = pyre_object::with_roots!(w_self => ctypeobj::getcfield(ct, name, "read"))?;
    unsafe { super::ctypestruct::read(field, cdata.ptr as *mut u8, w_self) }
}

/// `W_CData.setattr`.
fn cdata_setattr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // The value outlives a conversion that runs arbitrary Python, so it is
    // read back out of its slot.
    let roots = pyre_object::gc_roots::push_roots();
    let args_base = roots.pin_roots(&[args[0], args[1], args[2]]);
    let value_slot = args_base + 2;
    let cdata = cdata_arg(roots.get(args_base))?;
    let ct = cdata.ctype_ref()?;
    let field = ctypeobj::getcfield(
        ct,
        pyre_interpreter::baseobjspace::text_w(roots.get(args_base + 1))?,
        "write",
    )?;
    unsafe { super::ctypestruct::write(field, cdata.ptr as *mut u8, roots.get(value_slot))? };
    Ok(pyre_object::w_none())
}

// ── comparison ──────────────────────────────────────────────────────────

/// `W_CData._compare_mode`'s three answers.
enum CompareMode {
    /// Both operands are pointer-like: compare the addresses.
    Addresses(usize, usize),
    /// Exactly one operand is pointer-like: the comparison is undefined.
    Incomparable,
    /// Neither is: compare what they hold.
    Objects(PyObjectRef, PyObjectRef),
}

fn compare_mode(w_self: PyObjectRef, mut w_other: PyObjectRef) -> Result<CompareMode, PyError> {
    let cdata = pyre_object::with_roots!(w_other => cdata_arg(w_self))?;
    let self_is_ptr = !cdata.ctype_ref()?.is_primitive();
    let other = W_CData::from_obj(w_other);
    let other_is_ptr = other
        .as_ref()
        .and_then(|o| ctypeobj::ctype_at(o.ctype))
        .is_some_and(|ct| !ct.is_primitive());
    if self_is_ptr && other_is_ptr {
        let other = other.expect("other_is_ptr implies a cdata");
        return Ok(CompareMode::Addresses(cdata.ptr, other.ptr));
    }
    if self_is_ptr || other_is_ptr {
        return Ok(CompareMode::Incomparable);
    }
    // Boxing this side's value allocates, so the other operand — which the
    // non-cdata arm hands straight back — has to survive it.
    let roots = pyre_object::gc_roots::push_roots();
    let other_slot = roots.base();
    let _ = roots.pin_root(w_other);
    let ob1_slot = other_slot + 1;
    let _ = roots.pin_root(unsafe { ctypeobj::convert_to_object(cdata.ctype_ref()?, cdata.ptr)? });
    let w_ob2 = match other {
        Some(other) => unsafe { ctypeobj::convert_to_object(other.ctype_ref()?, other.ptr)? },
        None => roots.get(other_slot),
    };
    Ok(CompareMode::Objects(roots.get(ob1_slot), w_ob2))
}

fn compare(
    args: &[PyObjectRef],
    op: pyre_interpreter::bytecode::ComparisonOperator,
    on_addresses: fn(usize, usize) -> bool,
) -> Result<PyObjectRef, PyError> {
    match compare_mode(args[0], args[1])? {
        CompareMode::Addresses(a, b) => {
            Ok(pyre_object::boolobject::w_bool_from(on_addresses(a, b)))
        }
        CompareMode::Incomparable => Ok(pyre_object::special::w_not_implemented()),
        CompareMode::Objects(a, b) => pyre_interpreter::opcode_ops::compare_value(a, b, op),
    }
}

macro_rules! comparison {
    ($name:ident, $op:ident, $addr:expr) => {
        fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
            compare(
                args,
                pyre_interpreter::bytecode::ComparisonOperator::$op,
                $addr,
            )
        }
    };
}

comparison!(cdata_lt, Less, |a, b| a < b);
comparison!(cdata_le, LessOrEqual, |a, b| a <= b);
comparison!(cdata_eq, Equal, |a, b| a == b);
comparison!(cdata_ne, NotEqual, |a, b| a != b);
comparison!(cdata_gt, Greater, |a, b| a > b);
comparison!(cdata_ge, GreaterOrEqual, |a, b| a >= b);

// ── indexing ────────────────────────────────────────────────────────────

/// `W_CData.getitem`.
fn cdata_getitem(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let mut w_self = args[0];
    let w_index = args[1];
    if unsafe { pyre_object::sliceobject::is_slice(w_index) } {
        return do_getslice(w_self, w_index);
    }
    let i =
        pyre_object::with_roots!(w_self => pyre_interpreter::baseobjspace::getindex_w(w_index))?;
    let cdata = cdata_arg(w_self)?;
    let ct = check_subscript_index(cdata, i)?;
    // `W_CDataPtrToStructOrUnion._do_getitem` — `p[0]` is the struct itself.
    if cdata.flavor == FLAVOR_PTR_TO_STRUCT {
        return Ok(cdata.w_keepalive);
    }
    let item = ctypeobj::ctype_at(ct.ctitem)
        .ok_or_else(|| PyError::system_error("indexed ctype without an item type"))?;
    unsafe {
        ctypeobj::convert_to_object(
            item,
            cdata.ptr.wrapping_add_signed((i * item.size) as isize),
        )
    }
}

/// `W_CData.setitem`.
fn cdata_setitem(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let w_index = args[1];
    let w_value = args[2];
    if unsafe { pyre_object::sliceobject::is_slice(w_index) } {
        do_setslice(w_self, w_index, w_value)?;
        return Ok(pyre_object::w_none());
    }
    // The index's `__index__` is arbitrary Python, so the value being stored
    // has to be read back out of its slot afterwards.
    let roots = pyre_object::gc_roots::push_roots();
    let value_slot = roots.pin_roots(&[w_value, w_self]);
    let i = pyre_interpreter::baseobjspace::getindex_w(w_index)?;
    let cdata = cdata_arg(roots.get(value_slot + 1))?;
    let ct = check_subscript_index(cdata, i)?;
    let item = ctypeobj::ctype_at(ct.ctitem)
        .ok_or_else(|| PyError::system_error("indexed ctype without an item type"))?;
    unsafe {
        ctypeobj::convert_from_object(
            item,
            cdata.ptr.wrapping_add_signed((i * item.size) as isize),
            roots.get(value_slot),
        )?;
    }
    Ok(pyre_object::w_none())
}

/// `W_CType._check_subscript_index` and the overrides of it, returning the
/// ctype whose `ctitem` names the element.
fn check_subscript_index(cdata: &W_CData, i: i64) -> Result<&'static mut W_CType, PyError> {
    let ct = cdata.ctype_ref()?;
    match ct.kind {
        // `W_CTypePointer._check_subscript_index`.
        ctypeobj::KIND_POINTER => {
            if matches!(
                cdata.flavor,
                FLAVOR_NEW_STD | FLAVOR_NEW_NONSTD | FLAVOR_PTR_TO_STRUCT
            ) {
                if i != 0 {
                    return Err(PyError::index_error(format!(
                        "cdata '{}' can only be indexed by 0",
                        ct.name()
                    )));
                }
            } else if cdata.ptr == 0 {
                return Err(PyError::runtime_error(format!(
                    "cannot dereference null pointer from cdata '{}'",
                    ct.name()
                )));
            }
            Ok(ct)
        }
        // `W_CTypeArray._check_subscript_index`.
        ctypeobj::KIND_ARRAY => {
            if i < 0 {
                return Err(PyError::index_error("negative index"));
            }
            let length = cdata.array_length()?;
            if i >= length {
                return Err(PyError::index_error(format!(
                    "index too large for cdata '{}' (expected {i} < {length})",
                    ct.name()
                )));
            }
            Ok(ct)
        }
        _ => Err(PyError::type_error(format!(
            "cdata of type '{}' cannot be indexed",
            ct.name()
        ))),
    }
}

/// `W_CData._do_getslicearg` — the pointer ctype the slice reads through,
/// the start index, and the length.
fn getslicearg(
    w_self: PyObjectRef,
    w_slice: PyObjectRef,
) -> Result<(PyObjectRef, i64, i64), PyError> {
    // Each bound's `__index__` is arbitrary Python, so all three components
    // are pinned before the first conversion runs.
    let roots = pyre_object::gc_roots::push_roots();
    let start_slot = unsafe {
        roots.pin_roots(&[
            pyre_object::sliceobject::w_slice_get_start(w_slice),
            pyre_object::sliceobject::w_slice_get_stop(w_slice),
            pyre_object::sliceobject::w_slice_get_step(w_slice),
            w_self,
        ])
    };
    let (stop_slot, step_slot, self_slot) = (start_slot + 1, start_slot + 2, start_slot + 3);
    if unsafe { pyre_object::pyobject::is_none(roots.get(start_slot)) } {
        return Err(PyError::index_error("slice start must be specified"));
    }
    let start = pyre_interpreter::baseobjspace::int_w(roots.get(start_slot))?;
    if unsafe { pyre_object::pyobject::is_none(roots.get(stop_slot)) } {
        return Err(PyError::index_error("slice stop must be specified"));
    }
    let stop = pyre_interpreter::baseobjspace::int_w(roots.get(stop_slot))?;
    if !unsafe { pyre_object::pyobject::is_none(roots.get(step_slot)) } {
        return Err(PyError::index_error("slice with step not supported"));
    }
    if start > stop {
        return Err(PyError::index_error("slice start > stop"));
    }
    let cdata = cdata_arg(roots.get(self_slot))?;
    let ct = cdata.ctype_ref()?;
    // `W_CType._check_slice_index` and the overrides of it.
    let w_ctptr = match ct.kind {
        ctypeobj::KIND_POINTER => cdata.ctype,
        ctypeobj::KIND_ARRAY => {
            if start < 0 {
                return Err(PyError::index_error("negative index"));
            }
            let length = cdata.array_length()?;
            if stop > length {
                return Err(PyError::index_error(format!(
                    "index too large (expected {stop} <= {length})"
                )));
            }
            ct.ctptr
        }
        _ => {
            return Err(PyError::type_error(format!(
                "cdata of type '{}' cannot be indexed",
                ct.name()
            )));
        }
    };
    Ok((w_ctptr, start, stop - start))
}

/// `W_CData._do_getslice`.
fn do_getslice(mut w_self: PyObjectRef, w_slice: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let (w_ctptr, start, length) =
        pyre_object::with_roots!(w_self => getslicearg(w_self, w_slice))?;
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = roots.base();
    let _ = roots.pin_root(w_self);
    let w_ctarray = super::newtype::cached_unbounded_array_type(w_ctptr)?;
    let array_slot = self_slot + 1;
    let _ = roots.pin_root(w_ctarray);
    let item_size = ctypeobj::ctype_at(ctypeobj::ctype_arg(roots.get(array_slot))?.ctitem)
        .ok_or_else(|| PyError::system_error("array without an item type"))?
        .size;
    let cdata = cdata_arg(roots.get(self_slot))?;
    let ptr = cdata.ptr.wrapping_add_signed((start * item_size) as isize);
    Ok(new_cdata_sliced(ptr, roots.get(array_slot), length))
}

/// `W_CData._do_setslice`.
fn do_setslice(
    w_self: PyObjectRef,
    w_slice: PyObjectRef,
    w_value: PyObjectRef,
) -> Result<(), PyError> {
    // The slice bounds run `__index__`, so the assigned value is pinned first
    // and read back out of its slot for every arm below.
    let value_roots = pyre_object::gc_roots::push_roots();
    let value_slot = value_roots.base();
    let _ = value_roots.pin_root(w_value);
    let (w_ctptr, start, length) = getslicearg(w_self, w_slice)?;
    let w_value = value_roots.get(value_slot);
    let ctptr = ctypeobj::ctype_arg(w_ctptr)?;
    let item = ctypeobj::ctype_at(ctptr.ctitem)
        .ok_or_else(|| PyError::system_error("pointer without an item type"))?;
    let item_size = item.size;
    let cdata = cdata_arg(w_self)?;
    let target = cdata.ptr.wrapping_add_signed((start * item_size) as isize);

    // The fast path: copying from an array of exactly the item type.
    if let Some(source) = W_CData::from_obj(w_value)
        && let Some(source_ct) = ctypeobj::ctype_at(source.ctype)
        && source_ct.kind == ctypeobj::KIND_ARRAY
        && std::ptr::eq(
            ctypeobj::ctype_at(source_ct.ctitem).map_or(std::ptr::null(), |c| c as *const W_CType),
            item as *const W_CType,
        )
        && source.array_length()? == length
    {
        unsafe {
            std::ptr::copy_nonoverlapping(
                source.ptr as *const u8,
                target as *mut u8,
                (item_size * length) as usize,
            );
        }
        return Ok(());
    }

    // `<char[]>[0:N] = b"somestring"`, and the same for a bytearray.
    if item.is_primitive() && item_size == 1 {
        let source = unsafe {
            if pyre_object::bytesobject::is_bytes(w_value) {
                Some(("string", pyre_object::bytesobject::w_bytes_data(w_value)))
            } else if pyre_object::bytearrayobject::is_bytearray(w_value) {
                Some((
                    "bytearray",
                    &*pyre_object::bytearrayobject::w_bytearray_data(w_value),
                ))
            } else {
                None
            }
        };
        if let Some((kind, value)) = source {
            if value.len() as i64 != length {
                return Err(PyError::value_error(format!(
                    "need a {kind} of length {length}, got {}",
                    value.len()
                )));
            }
            unsafe {
                std::ptr::copy_nonoverlapping(value.as_ptr(), target as *mut u8, value.len())
            };
            return Ok(());
        }
    }

    // `W_CData._do_setslice_iterate`.  Each value is written as it arrives,
    // so a sequence of the wrong length leaves the ones that came first
    // stored before the error names the mismatch.
    let roots = pyre_object::gc_roots::push_roots();
    let iter_slot = roots.base();
    let _ = roots.pin_root(pyre_interpreter::baseobjspace::iter(w_value)?);
    let item_slot = iter_slot + 1;
    let _ = roots.pin_root(pyre_object::PY_NULL);
    for i in 0..length {
        match pyre_interpreter::baseobjspace::next(roots.get(iter_slot)) {
            Ok(w_item) => roots.set(item_slot, w_item),
            Err(err) => {
                let (stop, err) = err.matches_stop_iteration_keep();
                if stop {
                    return Err(PyError::value_error(format!(
                        "need {length} values to unpack, got {i}"
                    )));
                } else {
                    return Err(err);
                }
            }
        }
        let element = target.wrapping_add_signed((i * item_size) as isize);
        unsafe { ctypeobj::convert_from_object(item, element as usize, roots.get(item_slot))? };
    }
    match pyre_interpreter::baseobjspace::next(roots.get(iter_slot)) {
        Ok(_) => Err(PyError::value_error(format!(
            "got more than {length} values to unpack"
        ))),
        Err(err) => {
            let (stop, err) = err.matches_stop_iteration_keep();
            if stop { Ok(()) } else { Err(err) }
        }
    }
}

// ── arithmetic ──────────────────────────────────────────────────────────

/// `W_CData.add`, which is also `__radd__`.
fn cdata_add(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    add_or_sub(args[0], args[1], 1)
}

/// `W_CData.sub`.
fn cdata_sub(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let w_other = args[1];
    let Some(other) = W_CData::from_obj(w_other) else {
        return add_or_sub(w_self, w_other, -1);
    };
    let cdata = cdata_arg(w_self)?;
    let self_ct = cdata.ctype_ref()?;
    let other_ct = ctypeobj::ctype_at(other.ctype)
        .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
    // An array is compared through the pointer type it decays to.
    let other_ct = if other_ct.kind == ctypeobj::KIND_ARRAY {
        ctypeobj::ctype_at(other_ct.ctptr)
            .ok_or_else(|| PyError::system_error("array without a pointer type"))?
    } else {
        other_ct
    };
    let item_size = ctypeobj::ctype_at(other_ct.ctitem).map_or(-1, |it| it.size);
    if !std::ptr::eq(self_ct as *const W_CType, other_ct as *const W_CType)
        || other_ct.kind != ctypeobj::KIND_POINTER
        || (item_size <= 0 && !other_ct.has(ctypeobj::CTypeFlags::VOID_PTR))
    {
        return Err(PyError::type_error(format!(
            "cannot subtract cdata '{}' and cdata '{}'",
            self_ct.name(),
            other_ct.name()
        )));
    }
    let mut diff = cdata.ptr as i64 - other.ptr as i64;
    if item_size > 1 {
        if diff % item_size != 0 {
            return Err(PyError::value_error(
                "pointer subtraction: the distance between the two pointers is not a multiple of the item size",
            ));
        }
        diff /= item_size;
    }
    Ok(pyre_object::w_int_new(diff))
}

/// `W_CData._add_or_sub`.
fn add_or_sub(
    mut w_self: PyObjectRef,
    w_other: PyObjectRef,
    sign: i64,
) -> Result<PyObjectRef, PyError> {
    let i = sign
        * pyre_object::with_roots!(w_self => pyre_interpreter::baseobjspace::getindex_w(w_other))?;
    let cdata = cdata_arg(w_self)?;
    ctypeobj::add(cdata.ctype, cdata.ptr as *mut u8, i)
}

// ── module-level entry points that read a cdata ─────────────────────────

/// `func.py sizeof` on a cdata.
pub fn cdata_sizeof(w_cdata: PyObjectRef) -> Result<i64, PyError> {
    cdata_arg(w_cdata)?.sizeof()
}

/// `W_CData.unpack`.
pub fn unpack(mut w_cdata: PyObjectRef, length: i64) -> Result<PyObjectRef, PyError> {
    let cdata = pyre_object::with_roots!(w_cdata => cdata_arg(w_cdata))?;
    let ct = cdata.ctype_ref()?;
    if !ct.has(ctypeobj::CTypeFlags::NONFUNC_POINTER_OR_ARRAY) {
        return Err(PyError::type_error(format!(
            "expected a pointer or array, got '{}'",
            ct.name()
        )));
    }
    if length < 0 {
        return Err(PyError::value_error("'length' cannot be negative"));
    }
    if cdata.ptr == 0 {
        let w_repr = pyre_interpreter::builtins::builtin_repr(&[w_cdata])?;
        return Err(PyError::runtime_error(
            pyre_interpreter::display::wtf8_format!("cannot use unpack() on ", unsafe {
                pyre_object::w_str_get_wtf8(w_repr)
            }),
        ));
    }
    let item = ctypeobj::ctype_at(ct.ctitem)
        .ok_or_else(|| PyError::system_error("pointer without an item type"))?;
    super::ctypeptr::unpack_ptr(ct, item, cdata.ptr as *mut u8, length)
}
