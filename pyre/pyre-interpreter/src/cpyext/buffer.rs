//! The buffer protocol -- PyPy `cpyext/buffer.py` and `cpyext/memoryobject.py`.
//!
//! Two directions meet here.  A C-defined type's `bf_getbuffer` becomes a
//! `memoryview` over the exported memory ([`buffer_view`]), the exporter's
//! `bf_releasebuffer` running when that view is released ([`release_view`]).
//! In the other direction `PyObject_GetBuffer` hands C a `Py_buffer`; for an
//! interpreter object that is a **snapshot**, because pyre's collector moves
//! object storage and the address a `Py_buffer` carries has to stay put for as
//! long as C holds it.  A snapshot cannot receive writes, so a `PyBUF_WRITABLE`
//! request for an interpreter object is refused rather than silently dropped.

use super::object::{argument, result};
use super::pyobject::{self, CPyObject};
use super::typeobject::{CPyTypeObject, pending_or, slot_of};
use pyre_object::PyObjectRef;
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::hash::BuildHasherDefault;

/// `Py_buffer`.
#[repr(C)]
pub struct CPyBuffer {
    pub buf: *mut c_void,
    pub obj: *mut CPyObject,
    pub len: isize,
    pub itemsize: isize,
    pub readonly: c_int,
    pub ndim: c_int,
    pub format: *mut c_char,
    pub shape: *mut isize,
    pub strides: *mut isize,
    pub suboffsets: *mut isize,
    pub internal: *mut c_void,
}

bitflags::bitflags! {
    /// The `object.h` request flags a caller hands `PyObject_GetBuffer`.
    ///
    /// Two places spell this table: the header an extension compiles against
    /// and this declaration.  `every_buffer_flag_is_the_bit_the_header_gives_it`
    /// compares the two, walking `Flags::FLAGS` rather than a list somebody
    /// has to remember to extend.
    ///
    /// The names are the header's, uppercased.  Four of them ask for a whole
    /// shape rather than a single property and carry `PYBUF_ND` or
    /// `PYBUF_STRIDES` inside them, which is why the tests below read
    /// `contains` rather than a single bit.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct BufFlags: c_int {
        const PYBUF_SIMPLE = 0;
        const PYBUF_WRITABLE = 0x0001;
        const PYBUF_FORMAT = 0x0004;
        const PYBUF_ND = 0x0008;
        const PYBUF_STRIDES = 0x0010 | Self::PYBUF_ND.bits();
        const PYBUF_C_CONTIGUOUS = 0x0020 | Self::PYBUF_STRIDES.bits();
        const PYBUF_F_CONTIGUOUS = 0x0040 | Self::PYBUF_STRIDES.bits();
        const PYBUF_ANY_CONTIGUOUS = 0x0080 | Self::PYBUF_STRIDES.bits();
        const PYBUF_INDIRECT = 0x0100 | Self::PYBUF_STRIDES.bits();
    }
}

/// C hands `PyObject_GetBuffer` its own `int`, so the word this file reads has
/// to be that word and nothing wider or narrower.
const _: () = assert!(size_of::<BufFlags>() == size_of::<c_int>());
const _: () = assert!(align_of::<BufFlags>() == align_of::<c_int>());

/// `PyBUF_MAX_NDIM` — the dimension count a `Py_buffer` may declare.
const PYBUF_MAX_NDIM: i64 = 64;
/// `PyBUF_READ` and `PyBUF_WRITE` name which way a `PyMemoryView_FromMemory`
/// buffer runs.  They are their own argument, not bits in the request word,
/// which is why they share `PyBUF_INDIRECT`'s value without colliding.
const PYBUF_READ: c_int = 0x100;
const PYBUF_WRITE: c_int = 0x200;

/// Unsigned bytes — the format `PyBuffer_FillInfo` reports and the fallback for
/// an exporter that leaves `format` NULL.
static DEFAULT_FORMAT: &[u8] = b"B\0";

// ── the tables a bare `PyBufferProcs` lookup goes through ───────────────

fn getbuffer_of(tp: *mut CPyTypeObject) -> *const c_void {
    unsafe {
        let table = (*tp).tp_as_buffer;
        if table.is_null() {
            std::ptr::null()
        } else {
            (*table).bf_getbuffer
        }
    }
}

fn releasebuffer_of(tp: *mut CPyTypeObject) -> *const c_void {
    unsafe {
        let table = (*tp).tp_as_buffer;
        if table.is_null() {
            std::ptr::null()
        } else {
            (*table).bf_releasebuffer
        }
    }
}

// ── the exports this layer owns ─────────────────────────────────────────

/// Keyed by an address this layer handed out, so the hasher is never fed a key
/// C chose; the default one is spelled out because it is the const-
/// constructible form a `static` needs.
type BufferTable = super::address_table::AddressMap<usize>;
type BufferSet = super::address_table::AddressSet;

/// The `Py_buffer` behind each live `memoryview` this layer built, keyed by the
/// `BufferView` that memoryview carries.
///
/// A view has to hand the exporter back the exact structure it filled in --
/// `internal` is the exporter's own state and nothing else can reconstruct it,
/// and two live views over one address each have their own.  The key is the
/// `BufferView` allocation rather than the memoryview because the collector
/// moves the object and not that block, and rather than the exported address
/// because several exports can name the same one.
static EXPORTS: super::ForkMutex<BufferTable> =
    super::ForkMutex::new(BufferTable::with_hasher(BuildHasherDefault::new()));

/// The snapshots [`PyObject_GetBuffer`] took of interpreter objects, keyed by
/// the address handed to C, with the byte length `PyBuffer_Release` frees.
static SNAPSHOTS: super::ForkMutex<BufferTable> =
    super::ForkMutex::new(BufferTable::with_hasher(BuildHasherDefault::new()));

/// Each table's lock word can be held by a thread the child does not have, and
/// the payload is what the child keeps: `fork` copies the address space, so
/// every exported address and every snapshot is still mapped and still owed a
/// release.
pub(super) unsafe fn after_fork_child() {
    unsafe {
        EXPORTS.reinit_after_fork();
        SNAPSHOTS.reinit_after_fork();
        GEOMETRIES.reinit_after_fork();
    }
}

fn record_export(carrier: usize, view: *mut CPyBuffer) {
    EXPORTS.lock().insert(carrier, view as usize);
}

fn take_export(carrier: usize) -> Option<*mut CPyBuffer> {
    EXPORTS
        .lock()
        .remove(&carrier)
        .map(|view| view as *mut CPyBuffer)
}

// ── reading the geometry a `Py_buffer` carries ──────────────────────────

fn format_of(format: *const c_char) -> String {
    if format.is_null() {
        return "B".to_string();
    }
    unsafe { CStr::from_ptr(format) }
        .to_string_lossy()
        .into_owned()
}

/// One dimension vector as the exporter wrote it, or nothing when it
/// answered a request that does not ask for one.
fn dims(pointer: *const isize, ndim: i64) -> Vec<i64> {
    if ndim <= 0 || pointer.is_null() {
        return Vec::new();
    }
    (0..ndim as usize)
        .map(|index| unsafe { *pointer.add(index) } as i64)
        .collect()
}

/// `init_shape_strides` — the dimension vectors a filled `Py_buffer`
/// describes, with the ones a request that does not ask for them left NULL
/// reconstructed.
///
/// A zero-dimensional view has no dimensions, and fabricating one leaves
/// every caller reading an axis its own index vector does not have.  One
/// dimension derives `[len / itemsize]` and `[itemsize]`.  Beyond that a NULL
/// `strides` means the C-contiguous layout for `shape`, while a NULL `shape`
/// describes nothing that can be reconstructed: the vectors come back short
/// and the caller reports it rather than reading an axis that is not there.
///
/// # Safety
/// `view` must point at a filled `Py_buffer`.
unsafe fn shape_and_strides(
    view: *const CPyBuffer,
    ndim: i64,
    itemsize: i64,
    length: i64,
) -> (Vec<i64>, Vec<i64>) {
    let (shape_ptr, strides_ptr) = unsafe { ((*view).shape, (*view).strides) };
    if ndim <= 0 {
        return (Vec::new(), Vec::new());
    }
    if ndim == 1 {
        let elements = if itemsize > 0 { length / itemsize } else { 0 };
        let shape = if shape_ptr.is_null() {
            vec![elements]
        } else {
            dims(shape_ptr, 1)
        };
        let strides = if strides_ptr.is_null() {
            vec![itemsize]
        } else {
            dims(strides_ptr, 1)
        };
        return (shape, strides);
    }
    let shape = dims(shape_ptr, ndim);
    let strides = if strides_ptr.is_null() {
        crate::builtins::strides_from_shape(&shape, itemsize)
    } else {
        dims(strides_ptr, ndim)
    };
    (shape, strides)
}

/// `_IsCContiguous`.
fn c_contiguous(shape: &[i64], strides: &[i64], itemsize: i64, length: i64) -> bool {
    // `len == 0` holds exactly when some extent is zero; such a view is
    // contiguous in both orders whatever its strides say.
    if length == 0 {
        return true;
    }
    let mut expected = itemsize;
    for (&extent, &stride) in shape.iter().zip(strides).rev() {
        if extent > 1 && stride != expected {
            return false;
        }
        expected *= extent;
    }
    true
}

/// `_IsFortranContiguous`.
fn fortran_contiguous(shape: &[i64], strides: &[i64], itemsize: i64, length: i64) -> bool {
    if length == 0 {
        return true;
    }
    let mut expected = itemsize;
    for (&extent, &stride) in shape.iter().zip(strides) {
        if extent > 1 && stride != expected {
            return false;
        }
        expected *= extent;
    }
    true
}

// ── `bf_getbuffer` as a `memoryview` ────────────────────────────────────

/// The `bf_getbuffer` of a **C** exporter, or null.
///
/// A slot filled with [`interp_bf_getbuffer`] is not one: that trampoline
/// answers by acquiring a `memoryview`, which is where this lookup is made
/// from, so following it would call straight back into the caller.
fn c_getbuffer_of(w_obj: PyObjectRef) -> *const c_void {
    let slot = slot_of(w_obj, getbuffer_of);
    match std::ptr::eq(slot, interp_bf_getbuffer as *const c_void) {
        true => std::ptr::null(),
        false => slot,
    }
}

/// Does `w_obj`'s type export a buffer through a C `bf_getbuffer`?
pub fn exports_buffer(w_obj: PyObjectRef) -> bool {
    !c_getbuffer_of(w_obj).is_null()
}

/// The view a C exporter's `bf_getbuffer` produces, or `None` when `w_obj`'s
/// type declares none.
pub fn buffer_view(
    w_obj: PyObjectRef,
    flags: c_int,
) -> Option<Result<PyObjectRef, crate::PyError>> {
    let slot = c_getbuffer_of(w_obj);
    if slot.is_null() {
        return None;
    }
    Some(acquire(w_obj, slot, BufFlags::from_bits_retain(flags)))
}

fn acquire(
    w_obj: PyObjectRef,
    slot: *const c_void,
    flags: BufFlags,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::buffer::Buffer;
    use pyre_object::bufferview::BufferView;

    let roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_obj);
    let view = Box::into_raw(Box::new(unsafe { std::mem::zeroed::<CPyBuffer>() }));
    let receiver = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(obj_slot));
    let status = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyBuffer, c_int) -> c_int =
            std::mem::transmute(slot);
        call(receiver, view, flags.bits())
    };
    unsafe { pyobject::decref(receiver) };
    if status < 0 {
        drop(unsafe { Box::from_raw(view) });
        return Err(pending_or(
            "a cpyext buffer export failed without setting an exception",
        ));
    }

    // `Py_buffer.obj` is the exporter that owns this particular view, and a
    // `bf_getbuffer` is allowed to redirect it away from the object the caller
    // requested.  PyPy's `CPyBuffer.getndim`/`memory_attach` retain the filled
    // buffer itself, so its `obj` remains authoritative.  Root that owner
    // before allocating the memoryview header; using `w_obj` here made both
    // `memoryview.obj` and `bf_releasebuffer` name the redirector instead.
    let owner_slot = unsafe {
        let raw = (*view).obj;
        if raw.is_null() {
            None
        } else {
            let slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(pyobject::from_ref(raw));
            Some(slot)
        }
    };

    let (address, length, itemsize, readonly, ndim) = unsafe {
        (
            (*view).buf as usize,
            (*view).len as i64,
            if (*view).itemsize > 0 {
                (*view).itemsize as i64
            } else {
                1
            },
            (*view).readonly != 0,
            (*view).ndim as i64,
        )
    };
    let format = format_of(unsafe { (*view).format });
    let (shape, strides) = unsafe { shape_and_strides(view, ndim, itemsize, length) };
    let refuse = |kind: crate::PyErrorKind, message: &str| {
        release_c_view(view);
        Err(crate::PyError::new(kind, message.to_string()))
    };
    let suboffsets = dims(unsafe { (*view).suboffsets }, ndim);
    if ndim > PYBUF_MAX_NDIM {
        return refuse(
            crate::PyErrorKind::ValueError,
            "memoryview: number of dimensions must not exceed 64",
        );
    }
    if shape.len() as i64 != ndim || strides.len() as i64 != ndim {
        return refuse(
            crate::PyErrorKind::BufferError,
            "cpyext: a multi-dimensional buffer export must carry a shape",
        );
    }

    let base = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(pyre_object::w_str_new_managed(&format));
    let reload = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
    let mv = pyre_object::memoryview::w_memoryview_alloc_header(false, true);
    let r_obj = owner_slot
        .map(pyre_object::gc_roots::shadow_stack_get)
        .unwrap_or_else(pyre_object::w_none);
    let built = carrier_for(
        Geometry {
            address,
            length,
            itemsize,
            readonly,
            ndim,
            shape,
            strides,
            suboffsets,
        },
        r_obj,
        reload(0),
    );
    let carrier = pyre_object::memoryview::bufferview_alloc(built);
    unsafe { pyre_object::memoryview::w_memoryview_set_view(mv, carrier) };
    record_export(carrier as usize, view);
    unsafe { crate::builtins::memoryview_register_finalizer(mv) };
    Ok(mv)
}

/// The geometry words a filled `Py_buffer` carries, read once.
struct Geometry {
    address: usize,
    length: i64,
    itemsize: i64,
    readonly: bool,
    ndim: i64,
    shape: Vec<i64>,
    strides: Vec<i64>,
    suboffsets: Vec<i64>,
}

/// The view a filled `Py_buffer`'s geometry describes.
///
/// A one-dimensional export stepping by `itemsize` over exactly its own
/// length is the shape a `RawBufferView` covers, and is exactly the set this
/// layer accepted before it carried geometry of its own.  Everything else
/// rides on its own vectors as a `CPyBuffer`.
fn carrier_for(
    geometry: Geometry,
    w_obj: PyObjectRef,
    w_fmt: PyObjectRef,
) -> pyre_object::bufferview::BufferView {
    use pyre_object::buffer::Buffer;
    use pyre_object::bufferview::BufferView;
    let Geometry {
        address,
        length,
        itemsize,
        readonly,
        ndim,
        shape,
        strides,
        suboffsets,
    } = geometry;
    if ndim == 1
        && suboffsets.is_empty()
        && strides.first() == Some(&itemsize)
        && shape.first().copied().unwrap_or(0) * itemsize == length
    {
        return BufferView::Raw {
            backing: Buffer::External {
                w_obj,
                address,
                size: length.max(0) as usize,
                readonly,
            },
            w_obj,
            w_fmt,
            itemsize,
            length,
        };
    }
    // A zero-dimensional export addresses one element, but `len` is the
    // extent its exporter published and the storage behind it is contiguous,
    // so the window is `len` bytes from `buf`.
    //
    // An indirect export's storage reaches only as far as the first dimension
    // whose suboffset dereferences: past that the addresses come from the
    // exporter's own pointers and no window describes them.  The elements up
    // to and including that dimension are those pointers, so the window is
    // measured in pointer widths.
    let indirect = suboffsets.iter().position(|&suboffset| suboffset >= 0);
    let (low, high) = match (ndim, indirect) {
        (0, _) => (0, length),
        (_, Some(dim)) => pyre_object::bufferview::addressable_window(
            &shape[..=dim],
            &strides[..=dim],
            std::mem::size_of::<*const u8>() as i64,
        ),
        (_, None) => pyre_object::bufferview::addressable_window(&shape, &strides, itemsize),
    };
    BufferView::CBuffer {
        backing: Buffer::External {
            w_obj,
            address: (address as isize + low as isize) as usize,
            size: (high - low).max(0) as usize,
            readonly,
        },
        w_obj,
        w_fmt,
        itemsize,
        length,
        ndim,
        offset: -low,
        shape,
        strides,
        suboffsets,
    }
}

/// Run `bf_releasebuffer` for a structure this layer allocated, drop the
/// reference it holds, and free it.
fn release_c_view(view: *mut CPyBuffer) {
    let raw = unsafe { (*view).obj };
    if !raw.is_null() {
        let w_obj = unsafe { pyobject::from_ref(raw) };
        let release = slot_of(w_obj, releasebuffer_of);
        if !release.is_null() {
            unsafe {
                let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyBuffer) =
                    std::mem::transmute(release);
                call(raw, view);
            }
        }
        unsafe { pyobject::decref(raw) };
    }
    drop(unsafe { Box::from_raw(view) });
}

/// End the export a released `memoryview` held, if a C exporter produced it.
///
/// # Safety
/// `mv` must be a live memoryview and `backing` the object its view names.
pub unsafe fn release_view(mv: PyObjectRef, backing: PyObjectRef) -> bool {
    use pyre_object::buffer::Buffer;
    let carrier = unsafe { pyre_object::memoryview::w_memoryview_view(mv) };
    let Buffer::External { .. } = *carrier.backing() else {
        return false;
    };
    if let Some(view) = take_export(carrier as *const _ as usize) {
        release_c_view(view);
        return true;
    }
    // Preserve the pre-existing derived-view case: its cloned carrier has no
    // table row of its own, but the C backing still identifies the native
    // release path.  The table lookup comes first so a legacy export whose
    // filled `Py_buffer.obj` is NULL is nevertheless freed.
    !slot_of(backing, getbuffer_of).is_null()
}

// ── the buffer slots of a synthesized type mirror ───────────────────────

/// Does `w_type` declare that its instances export a buffer?
///
/// The counterpart of `TypeDef.__buffer`, which each type declares beside
/// itself and which reaches exactly one reader, `make_bf_getbuffer`, as a
/// plain yes/no -- the `'read'` / `'read-write'` vocabulary it is written in
/// is never consulted.  A type that declares nothing takes its bases'
/// declaration, so the answer follows the hierarchy rather than exact
/// identity and a class derived from `bytearray` exports what `bytearray`
/// does.
pub(super) fn declares_buffer(w_type: PyObjectRef) -> bool {
    // TypeDef.__buffer, including a base that declared it.
    unsafe { pyre_object::w_type_declares_buffer(w_type) }
}

/// What one export by [`interp_bf_getbuffer`] owns beyond the words it wrote
/// into the caller's `Py_buffer`.
///
/// The geometry vectors are here because a `Py_buffer` only points at its
/// format, shape and strides; upstream keeps them in fields of its own
/// declaration of the structure, which is not open to a runtime whose
/// extensions were compiled against the canonical one.
///
/// `view` is an owning reference to the `memoryview` the acquisition
/// produced.  It is what holds the export open: until it is released the
/// exporter refuses to be resized, which is what keeps the address handed to
/// C pointing at the bytes it was taken from.
struct Export {
    view: *mut CPyObject,
    /// Whether the input object whose owning reference is already in
    /// `Py_buffer.obj` is a memoryview with one export count to release.  The
    /// derived `view` above keeps the underlying bytes alive.  CPython 3.14
    /// `memory_getbuf` also forbids releasing this source memoryview until
    /// `PyBuffer_Release`; PyPy `W_MemoryView.buffer_w` deliberately omits
    /// that count, so this is the caller-visible 3.14 lifetime rule over
    /// PyPy's derived-view ownership shape.
    source_is_memoryview: bool,
    format: Vec<u8>,
    shape: Vec<isize>,
    strides: Vec<isize>,
    suboffsets: Vec<isize>,
}

/// The live [`Export`] blocks, so a release frees one this layer wrote and
/// leaves a C exporter's own `internal` word alone.
static GEOMETRIES: super::ForkMutex<BufferSet> =
    super::ForkMutex::new(BufferSet::with_hasher(BuildHasherDefault::new()));

/// What a request asks of the layout that this export does not answer, if
/// anything -- `memory_getbuf`'s contiguity checks, in its order.
///
/// `contiguous` is whether the export is contiguous in C and in Fortran
/// order, and `indirect` whether any of its dimensions carries a suboffset.
fn contiguity_refusal(
    flags: BufFlags,
    contiguous: (bool, bool),
    indirect: bool,
) -> Option<&'static str> {
    let requires = |mask: BufFlags| flags.contains(mask);
    let (c_order, fortran) = contiguous;
    if requires(BufFlags::PYBUF_C_CONTIGUOUS) && !c_order {
        return Some("memoryview: underlying buffer is not C-contiguous");
    }
    if requires(BufFlags::PYBUF_F_CONTIGUOUS) && !fortran {
        return Some("memoryview: underlying buffer is not Fortran contiguous");
    }
    if requires(BufFlags::PYBUF_ANY_CONTIGUOUS) && !(c_order || fortran) {
        return Some("memoryview: underlying buffer is not contiguous");
    }
    // A request that does not ask to follow pointers cannot read a view whose
    // elements are behind them.
    if !requires(BufFlags::PYBUF_INDIRECT) && indirect {
        return Some("memoryview: underlying buffer requires suboffsets");
    }
    // A request with no strides in it reads the address as one contiguous
    // run, which a strided export is not.
    if !requires(BufFlags::PYBUF_STRIDES) && !c_order {
        return Some("memoryview: underlying buffer is not C-contiguous");
    }
    // A request with no shape in it reads the bytes as unsigned chars, which
    // leaves nothing for a format to describe.
    if !requires(BufFlags::PYBUF_ND) && requires(BufFlags::PYBUF_FORMAT) {
        return Some("memoryview: cannot cast to unsigned bytes if the format flag is present");
    }
    None
}

/// `slotdefs.py slot_from_buffer_w` -- the `bf_getbuffer` of a type this
/// runtime defines, and of a class written in Python that defines
/// `__buffer__`.
///
/// Both reach the same acquisition, because the `memoryview` constructor is
/// already the one place that knows how each exporter answers.
pub(super) unsafe extern "C" fn interp_bf_getbuffer(
    object: *mut CPyObject,
    view: *mut CPyBuffer,
    flags: c_int,
) -> c_int {
    let request = BufFlags::from_bits_retain(flags);
    // A request with no structure to fill asks for nothing.
    if view.is_null() {
        return 0;
    }
    let Some(w_obj) = argument(object) else {
        return -1;
    };
    let roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let w_obj = roots.pin_root(w_obj);
    let acquired = crate::builtins::w_memoryview_new_with_flags(w_obj, flags);
    let Some(w_view) = super::pyerrors::trap(acquired) else {
        return -1;
    };
    let view_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_view);
    let reload = |slot| pyre_object::gc_roots::shadow_stack_get(slot);
    let contiguous = unsafe { crate::builtins::memoryview_contiguity(reload(view_slot)) };
    let indirect = unsafe {
        !pyre_object::memoryview::w_memoryview_native_suboffsets(reload(view_slot)).is_empty()
    };
    if let Some(message) = contiguity_refusal(request, contiguous, indirect) {
        release_acquired(reload(view_slot));
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::BufferError,
            message,
        ));
        return -1;
    }
    // Both references first: minting one can move what the other names.
    let holder = pyobject::make_ref(reload(view_slot));
    let exporter = pyobject::make_ref(reload(obj_slot));
    let source_is_memoryview = unsafe {
        if pyre_object::memoryview::is_w_memoryview(reload(obj_slot)) {
            pyre_object::memoryview::w_memoryview_exports_incref(reload(obj_slot));
            true
        } else {
            false
        }
    };
    let carrier = unsafe { pyre_object::memoryview::w_memoryview_view(reload(view_slot)) };
    let mut format = unsafe { carrier.format_str() }.as_bytes().to_vec();
    format.push(0);
    let widen = |dims: Vec<i64>| dims.into_iter().map(|extent| extent as isize).collect();
    let mut export = Box::new(Export {
        view: holder,
        source_is_memoryview,
        format,
        shape: widen(unsafe { carrier.native_shape() }),
        strides: widen(unsafe { carrier.native_strides() }),
        suboffsets: widen(unsafe { carrier.native_suboffsets() }),
    });
    // The exporter's own storage, which the collector does not move: the
    // address stays the one C was handed for as long as the export is open.
    let first = unsafe { carrier.backing().as_bytes() }.as_ptr() as usize;
    // `init_shape_strides` answers NULL for a zero-dimensional view, and an
    // empty `Vec` hands out a dangling address rather than one.
    let axes = carrier.ndim() != 0;
    unsafe {
        (*view).buf = (first + carrier.offset() as usize) as *mut c_void;
        (*view).obj = exporter;
        (*view).len = carrier.length() as isize;
        (*view).itemsize = carrier.itemsize() as isize;
        (*view).readonly = carrier.readonly() as c_int;
        // `memory_getbuf`: a request with no shape in it reads the whole
        // export as one run of unsigned chars, whatever its own rank.
        (*view).ndim = match request.contains(BufFlags::PYBUF_ND) {
            true => carrier.ndim() as c_int,
            false => 1,
        };
        (*view).format = match request.contains(BufFlags::PYBUF_FORMAT) {
            true => export.format.as_mut_ptr() as *mut c_char,
            false => std::ptr::null_mut(),
        };
        (*view).shape = match axes && request.contains(BufFlags::PYBUF_ND) {
            true => export.shape.as_mut_ptr(),
            false => std::ptr::null_mut(),
        };
        (*view).strides = match axes && request.contains(BufFlags::PYBUF_STRIDES) {
            true => export.strides.as_mut_ptr(),
            false => std::ptr::null_mut(),
        };
        (*view).suboffsets =
            match !export.suboffsets.is_empty() && request.contains(BufFlags::PYBUF_INDIRECT) {
                true => export.suboffsets.as_mut_ptr(),
                false => std::ptr::null_mut(),
            };
        (*view).internal = Box::into_raw(export) as *mut c_void;
        GEOMETRIES.lock().insert((*view).internal as usize);
    }
    0
}

/// `slotdefs.py make_bf_releasebuffer` -- end what [`interp_bf_getbuffer`]
/// began.
///
/// Upstream fills this slot only for a type that defines
/// `__release_buffer__`, because its acquisition holds nothing to give back.
/// This one always does: releasing the `memoryview` is what drops the
/// exporter's count, and it runs the `__release_buffer__` of a class written
/// in Python on the way.
pub(super) unsafe extern "C" fn interp_bf_releasebuffer(
    object: *mut CPyObject,
    view: *mut CPyBuffer,
) {
    if view.is_null() {
        return;
    }
    let internal = unsafe { (*view).internal } as usize;
    if internal == 0 || !GEOMETRIES.lock().remove(&internal) {
        return;
    }
    unsafe { (*view).internal = std::ptr::null_mut() };
    let export = unsafe { Box::from_raw(internal as *mut Export) };
    release_acquired(unsafe { pyobject::from_ref(export.view) });
    unsafe { pyobject::decref(export.view) };
    if export.source_is_memoryview && !object.is_null() {
        // `PyBuffer_Release` clears/decrefs `view.obj` only after this slot
        // returns, exactly as CPython `memory_releasebuf` relies on.
        let source = unsafe { pyobject::from_ref(object) };
        unsafe { pyre_object::memoryview::w_memoryview_exports_decref(source) };
    }
}

/// Release a `memoryview` this layer acquired.  A release failure belongs to
/// no operation the caller can be told about -- the slot answers nothing --
/// so it is reported the way an unraisable is.
fn release_acquired(w_view: PyObjectRef) {
    if let Err(mut error) = crate::builtins::memoryview_release(&[w_view]) {
        error.write_unraisable_with_traceback(
            pyre_object::w_none(),
            rustpython_wtf8::Wtf8::new("in __release_buffer__"),
            w_view,
            true,
            rustpython_wtf8::Wtf8::new(""),
        );
    }
}

// ── `PyObject_GetBuffer` and its snapshot ───────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_CheckBuffer(object: *mut CPyObject) -> c_int {
    let w_obj = unsafe { pyobject::from_ref(object) };
    if w_obj.is_null() {
        return 0;
    }
    if !slot_of(w_obj, getbuffer_of).is_null() {
        return 1;
    }
    unsafe { crate::baseobjspace::lookup(w_obj, "__buffer__") }.is_some() as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GetBuffer(
    object: *mut CPyObject,
    view: *mut CPyBuffer,
    flags: c_int,
) -> c_int {
    if view.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let Some(w_obj) = argument(object) else {
        return -1;
    };
    let slot = slot_of(w_obj, getbuffer_of);
    if !slot.is_null() {
        return unsafe {
            let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyBuffer, c_int) -> c_int =
                std::mem::transmute(slot);
            call(object, view, flags)
        };
    }
    if BufFlags::from_bits_retain(flags).contains(BufFlags::PYBUF_WRITABLE) {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::BufferError,
            "an interpreter object exports a read-only snapshot to C",
        ));
        return -1;
    }
    let acquired = crate::baseobjspace::full_ro_buffer_bytes(w_obj);
    let bytes = match super::pyerrors::trap(acquired) {
        Some(Some(acquired)) => {
            let bytes = acquired.as_bytes().to_vec();
            acquired.release();
            bytes
        }
        Some(None) => {
            super::pyerrors::set_pending_error(crate::PyError::type_error(
                "a bytes-like object is required",
            ));
            return -1;
        }
        None => return -1,
    };
    let length = bytes.len();
    let block = Box::into_raw(bytes.into_boxed_slice()) as *mut u8;
    SNAPSHOTS.lock().insert(block as usize, length);
    unsafe {
        fill_info(
            view,
            object,
            block as *mut c_void,
            length as isize,
            1,
            BufFlags::from_bits_retain(flags),
        )
    }
}

/// `PyBuffer_FillInfo`'s body, shared with the snapshot path.
///
/// # Safety
/// `view` must point at a writable `Py_buffer`.
unsafe fn fill_info(
    view: *mut CPyBuffer,
    object: *mut CPyObject,
    buf: *mut c_void,
    length: isize,
    readonly: c_int,
    flags: BufFlags,
) -> c_int {
    if flags.contains(BufFlags::PYBUF_WRITABLE) && readonly != 0 {
        super::pyerrors::set_pending_error(crate::PyError::value_error("Object is not writable"));
        return -1;
    }
    unsafe {
        (*view).buf = buf;
        (*view).obj = pyobject::make_ref(pyobject::from_ref(object));
        (*view).len = length;
        (*view).itemsize = 1;
        (*view).readonly = readonly;
        (*view).ndim = 1;
        (*view).format = if flags.contains(BufFlags::PYBUF_FORMAT) {
            DEFAULT_FORMAT.as_ptr() as *mut c_char
        } else {
            std::ptr::null_mut()
        };
        // The one-dimensional geometry is the structure's own `len` and
        // `itemsize`, so no separate storage has to outlive the call.
        (*view).shape = if flags.contains(BufFlags::PYBUF_ND) {
            &raw mut (*view).len
        } else {
            std::ptr::null_mut()
        };
        (*view).strides = if flags.contains(BufFlags::PYBUF_STRIDES) {
            &raw mut (*view).itemsize
        } else {
            std::ptr::null_mut()
        };
        (*view).suboffsets = std::ptr::null_mut();
        (*view).internal = std::ptr::null_mut();
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBuffer_FillInfo(
    view: *mut CPyBuffer,
    object: *mut CPyObject,
    buf: *mut c_void,
    length: isize,
    readonly: c_int,
    flags: c_int,
) -> c_int {
    if view.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    unsafe {
        fill_info(
            view,
            object,
            buf,
            length,
            readonly,
            BufFlags::from_bits_retain(flags),
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBuffer_Release(view: *mut CPyBuffer) {
    if view.is_null() {
        return;
    }
    let raw = unsafe { (*view).obj };
    if raw.is_null() {
        return;
    }
    let block = unsafe { (*view).buf } as usize;
    let snapshot = SNAPSHOTS.lock().remove(&block);
    match snapshot {
        Some(length) => drop(unsafe {
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(block as *mut u8, length))
        }),
        None => {
            let w_obj = unsafe { pyobject::from_ref(raw) };
            let release = slot_of(w_obj, releasebuffer_of);
            if !release.is_null() {
                unsafe {
                    let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyBuffer) =
                        std::mem::transmute(release);
                    call(raw, view);
                }
            }
        }
    }
    unsafe {
        (*view).obj = std::ptr::null_mut();
        (*view).buf = std::ptr::null_mut();
        pyobject::decref(raw);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBuffer_IsContiguous(view: *const CPyBuffer, order: c_char) -> c_int {
    if view.is_null() {
        return 0;
    }
    if unsafe { !(*view).suboffsets.is_null() } {
        return 0;
    }
    let ndim = unsafe { (*view).ndim } as i64;
    let itemsize = unsafe { (*view).itemsize } as i64;
    let length = unsafe { (*view).len } as i64;
    let (shape, strides) = unsafe { shape_and_strides(view, ndim, itemsize, length) };
    contiguous_in(order, &shape, &strides, itemsize, length) as c_int
}

/// Whether the geometry is contiguous in `order`, which is `'C'`, `'F'`ortran
/// or `'A'`ny of the two.  An order that is none of those is not contiguous:
/// the entry points that reject one do so before asking.
fn contiguous_in(
    order: c_char,
    shape: &[i64],
    strides: &[i64],
    itemsize: i64,
    length: i64,
) -> bool {
    match order as u8 {
        b'C' => c_contiguous(shape, strides, itemsize, length),
        b'F' => fortran_contiguous(shape, strides, itemsize, length),
        b'A' => {
            c_contiguous(shape, strides, itemsize, length)
                || fortran_contiguous(shape, strides, itemsize, length)
        }
        _ => false,
    }
}

/// The number of bytes one item of `format` occupies -- `_struct.calcsize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBuffer_SizeFromFormat(format: *const c_char) -> isize {
    if format.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let format = format_of(format);
    let sized = super::import_::import_module("_struct").and_then(|module| {
        let roots = pyre_object::gc_roots::push_roots();
        let base = roots.base();
        let _ = roots.pin_root(module);
        let _ = roots.pin_root(pyre_object::w_str_new_managed(&format));
        let reload = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
        let calcsize = crate::baseobjspace::getattr_str(reload(0), "calcsize")?;
        let size = crate::call::call_function_impl_result(calcsize, &[reload(1)])?;
        crate::baseobjspace::int_w(size)
    });
    match super::pyerrors::trap(sized) {
        Some(size) => size as isize,
        None => -1,
    }
}

// ── the strided copies ──────────────────────────────────────────────────

/// Step `indices` one element in C order (`order == b'C'`) or Fortran order,
/// answering `false` once the walk is over.
fn advance(indices: &mut [i64], shape: &[i64], order: u8) -> bool {
    let axes: Vec<usize> = if order == b'F' {
        (0..indices.len()).collect()
    } else {
        (0..indices.len()).rev().collect()
    };
    for axis in axes {
        indices[axis] += 1;
        if indices[axis] < shape[axis] {
            return true;
        }
        indices[axis] = 0;
    }
    false
}

/// The `(shape, strides, suboffsets, itemsize, total)` a filled `Py_buffer`
/// describes.  An empty `suboffsets` is NULL.
///
/// # Safety
/// `view` must point at a filled `Py_buffer`.
unsafe fn geometry(view: *const CPyBuffer) -> (Vec<i64>, Vec<i64>, Vec<i64>, i64, i64) {
    let ndim = unsafe { (*view).ndim } as i64;
    let itemsize = unsafe { (*view).itemsize } as i64;
    let length = unsafe { (*view).len } as i64;
    let (shape, strides) = unsafe { shape_and_strides(view, ndim, itemsize, length) };
    let suboffsets = dims(unsafe { (*view).suboffsets }, ndim);
    (shape, strides, suboffsets, itemsize, length)
}

/// `PyBuffer_GetPointer`'s per-axis chain: advance by each dimension's stride
/// and follow the pointer a non-negative suboffset names.
///
/// # Safety
/// `base` must address live storage the given geometry describes, and
/// `indices` must name one in-bounds index per dimension.
unsafe fn element_at(
    base: *mut u8,
    strides: &[i64],
    suboffsets: &[i64],
    indices: &[i64],
) -> *mut u8 {
    let mut pointer = base;
    for (axis, &index) in indices.iter().enumerate() {
        let stride = strides.get(axis).copied().unwrap_or(0);
        pointer = unsafe { pointer.offset((stride * index) as isize) };
        match suboffsets.get(axis).copied() {
            Some(suboffset) if suboffset >= 0 => {
                pointer = unsafe { (*(pointer as *mut *mut u8)).offset(suboffset as isize) };
            }
            _ => {}
        }
    }
    pointer
}

/// The address of the element `indices` names, over the geometry the view
/// describes: a request without `PyBUF_STRIDES` leaves `strides` NULL and the
/// element steps by the C-contiguous strides `init_shape_strides`
/// reconstructs, not by nothing.
///
/// # Safety
/// `view` must point at a filled `Py_buffer`, and `indices` must already be
/// bounds-checked against its shape.
unsafe fn element_of(view: *const CPyBuffer, indices: &[i64]) -> *mut u8 {
    let (_, strides, suboffsets, _, _) = unsafe { geometry(view) };
    unsafe { element_at((*view).buf as *mut u8, &strides, &suboffsets, indices) }
}

/// Walk every element of `view` in `order`, handing each element's address and
/// its position in the contiguous block to `visit`.
///
/// # Safety
/// `view` must point at a filled `Py_buffer`.
unsafe fn walk(view: *const CPyBuffer, order: u8, mut visit: impl FnMut(*mut u8, usize, usize)) {
    let (shape, strides, suboffsets, itemsize, _) = unsafe { geometry(view) };
    if shape.iter().any(|&extent| extent <= 0) {
        return;
    }
    let base = unsafe { (*view).buf } as *mut u8;
    let mut indices = vec![0i64; shape.len()];
    let mut position = 0usize;
    loop {
        let element = unsafe { element_at(base, &strides, &suboffsets, &indices) };
        visit(element, position, itemsize as usize);
        position += itemsize as usize;
        if !advance(&mut indices, &shape, order) {
            break;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBuffer_GetPointer(
    view: *const CPyBuffer,
    indices: *const isize,
) -> *mut c_void {
    let ndim = unsafe { (*view).ndim } as usize;
    let named: Vec<i64> = (0..ndim)
        .map(|axis| unsafe { *indices.add(axis) } as i64)
        .collect();
    unsafe { element_of(view, &named) as *mut c_void }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBuffer_ToContiguous(
    buf: *mut c_void,
    view: *const CPyBuffer,
    length: isize,
    order: c_char,
) -> c_int {
    if buf.is_null() || view.is_null() || unsafe { (*view).len } != length {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let target = buf as *mut u8;
    // A view already contiguous in the requested order copies as one run.
    // This is what 'A' asks for over a Fortran-contiguous source: its own
    // physical bytes, not the C-order walk below.
    if unsafe { PyBuffer_IsContiguous(view, order) } != 0 {
        unsafe { std::ptr::copy_nonoverlapping((*view).buf as *const u8, target, length as usize) };
        return 0;
    }
    unsafe {
        walk(view, order as u8, |element, position, itemsize| {
            std::ptr::copy_nonoverlapping(element, target.add(position), itemsize)
        })
    };
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBuffer_FromContiguous(
    view: *const CPyBuffer,
    buf: *const c_void,
    length: isize,
    order: c_char,
) -> c_int {
    if buf.is_null() || view.is_null() || unsafe { (*view).len } != length {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    if unsafe { (*view).readonly } != 0 {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::BufferError,
            "Object is not writable.",
        ));
        return -1;
    }
    let source = buf as *const u8;
    // As in `PyBuffer_ToContiguous`, an already-contiguous destination takes
    // the whole run at once.
    if unsafe { PyBuffer_IsContiguous(view, order) } != 0 {
        unsafe { std::ptr::copy_nonoverlapping(source, (*view).buf as *mut u8, length as usize) };
        return 0;
    }
    unsafe {
        walk(view, order as u8, |element, position, itemsize| {
            std::ptr::copy_nonoverlapping(source.add(position), element, itemsize)
        })
    };
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_CopyData(
    destination: *mut CPyObject,
    source: *mut CPyObject,
) -> c_int {
    super::object::realize_all([destination, source]);
    let mut target = unsafe { std::mem::zeroed::<CPyBuffer>() };
    let mut origin = unsafe { std::mem::zeroed::<CPyBuffer>() };
    if unsafe {
        PyObject_GetBuffer(
            destination,
            &raw mut target,
            (BufFlags::PYBUF_WRITABLE | BufFlags::PYBUF_ND).bits(),
        )
    } < 0
    {
        return -1;
    }
    if unsafe { PyObject_GetBuffer(source, &raw mut origin, BufFlags::PYBUF_ND.bits()) } < 0 {
        unsafe { PyBuffer_Release(&raw mut target) };
        return -1;
    }
    let status = if target.len < origin.len {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::BufferError,
            "destination is too small to receive data from source",
        ));
        -1
    } else {
        let bytes = vec![0u8; origin.len as usize];
        let block = bytes.into_boxed_slice();
        let block = Box::into_raw(block) as *mut u8;
        unsafe {
            walk(&raw const origin, b'C', |element, position, itemsize| {
                std::ptr::copy_nonoverlapping(element, block.add(position), itemsize)
            });
            walk(&raw const target, b'C', |element, position, itemsize| {
                std::ptr::copy_nonoverlapping(block.add(position), element, itemsize)
            });
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                block,
                origin.len as usize,
            )));
        }
        0
    };
    unsafe {
        PyBuffer_Release(&raw mut origin);
        PyBuffer_Release(&raw mut target);
    }
    status
}

// ── the legacy character-buffer entry points ────────────────────────────

/// These spellings hand out an address and have no release call at all, so the
/// export is closed here as `buffer.py:179-195` closes it -- an exporter that
/// counts its exports, or that allocated `internal`, otherwise never gets the
/// balancing call, and its memory outlives the export either way.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_AsCharBuffer(
    object: *mut CPyObject,
    buffer: *mut *const c_char,
    size: *mut isize,
) -> c_int {
    let Some(w_obj) = argument(object) else {
        return -1;
    };
    if !slot_of(w_obj, getbuffer_of).is_null() {
        let mut view = unsafe { std::mem::zeroed::<CPyBuffer>() };
        if unsafe { PyObject_GetBuffer(object, &raw mut view, 0) } < 0 {
            return -1;
        }
        unsafe {
            *buffer = view.buf as *const c_char;
            *size = view.len;
            PyBuffer_Release(&raw mut view);
        }
        return 0;
    }
    // An interpreter object has no exporter to hand an address back to, so the
    // bytes C reads are this layer's own copy.  With no release call to free it
    // by, the copy is the mirror's cached one, which dies with the mirror -- a
    // per-call snapshot would have no owner at all.
    let acquired = crate::baseobjspace::full_ro_buffer_bytes(w_obj);
    let bytes = match super::pyerrors::trap(acquired) {
        Some(Some(acquired)) => {
            let bytes = acquired.as_bytes().to_vec();
            acquired.release();
            bytes
        }
        Some(None) => {
            super::pyerrors::set_pending_error(crate::PyError::type_error(
                "a bytes-like object is required",
            ));
            return -1;
        }
        None => return -1,
    };
    let (pointer, length) = unsafe { pyobject::cached_bytes(object, || bytes) };
    unsafe {
        *buffer = pointer;
        *size = length as isize;
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_AsReadBuffer(
    object: *mut CPyObject,
    buffer: *mut *const c_void,
    size: *mut isize,
) -> c_int {
    let mut text: *const c_char = std::ptr::null();
    let status = unsafe { PyObject_AsCharBuffer(object, &raw mut text, size) };
    if status == 0 {
        unsafe { *buffer = text as *const c_void };
    }
    status
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_AsWriteBuffer(
    object: *mut CPyObject,
    buffer: *mut *mut c_void,
    size: *mut isize,
) -> c_int {
    let mut view = unsafe { std::mem::zeroed::<CPyBuffer>() };
    if unsafe { PyObject_GetBuffer(object, &raw mut view, BufFlags::PYBUF_WRITABLE.bits()) } < 0 {
        return -1;
    }
    unsafe {
        *buffer = view.buf;
        *size = view.len;
        PyBuffer_Release(&raw mut view);
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_CheckReadBuffer(object: *mut CPyObject) -> c_int {
    unsafe { PyObject_CheckBuffer(object) }
}

// ── memoryview ──────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMemoryView_Check(object: *mut CPyObject) -> c_int {
    let w_obj = unsafe { pyobject::from_ref(object) };
    if w_obj.is_null() {
        return 0;
    }
    unsafe { pyre_object::memoryview::is_w_memoryview(w_obj) as c_int }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMemoryView_FromObject(object: *mut CPyObject) -> *mut CPyObject {
    let Some(w_obj) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::builtins::w_memoryview_new(w_obj))
}

/// A view over memory nothing in the interpreter owns.
///
/// The window keeps no reference: the caller guarantees the memory outlives
/// every view of it, which is what this entry point's contract already says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMemoryView_FromMemory(
    memory: *mut c_char,
    size: isize,
    flags: c_int,
) -> *mut CPyObject {
    if memory.is_null() || size < 0 {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    result(Ok(unowned_view(
        memory as usize,
        size as i64,
        1,
        "B",
        flags != PYBUF_WRITE,
    )))
}

/// `PyMemoryView_FromBuffer` — the caller hands its filled `Py_buffer` over and
/// must not release it itself.
///
/// The handed-over structure's `obj` is a borrowed reference that must not be
/// decremented, so the view carries no exporter and records no export: it
/// keeps only the geometry and the address.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMemoryView_FromBuffer(view: *const CPyBuffer) -> *mut CPyObject {
    if view.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    if unsafe { (*view).buf.is_null() } {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "PyMemoryView_FromBuffer(): info->buf must not be NULL",
        ));
        return std::ptr::null_mut();
    }
    let (shape, strides, suboffsets, itemsize, length) = unsafe { geometry(view) };
    let ndim = unsafe { (*view).ndim } as i64;
    if ndim > PYBUF_MAX_NDIM {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "memoryview: number of dimensions must not exceed 64",
        ));
        return std::ptr::null_mut();
    }
    if shape.len() as i64 != ndim || strides.len() as i64 != ndim {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::BufferError,
            "cpyext: a multi-dimensional buffer export must carry a shape",
        ));
        return std::ptr::null_mut();
    }
    let format = format_of(unsafe { (*view).format });
    result(Ok(unowned_view_nd(
        Geometry {
            address: unsafe { (*view).buf } as usize,
            length,
            itemsize,
            readonly: unsafe { (*view).readonly } != 0,
            ndim,
            shape,
            strides,
            suboffsets,
        },
        &format,
    )))
}

/// A memoryview over `object` laid out in `order` --
/// `memoryobject.py PyMemoryView_GetContiguous`.
///
/// `buffertype` is `PyBUF_READ` or `PyBUF_WRITE` and says what the caller will
/// do with the memory, not how the view is built: a writable request over a
/// read-only exporter is refused rather than copied.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMemoryView_GetContiguous(
    object: *mut CPyObject,
    buffertype: c_int,
    order: c_char,
) -> *mut CPyObject {
    if buffertype != PYBUF_READ && buffertype != PYBUF_WRITE {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "buffertype must be PyBUF_READ or PyBUF_WRITE",
        ));
        return std::ptr::null_mut();
    }
    if !matches!(order as u8, b'C' | b'F' | b'A') {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "order must be in ('C', 'F', 'A')",
        ));
        return std::ptr::null_mut();
    }
    let Some(w_obj) = argument(object) else {
        return std::ptr::null_mut();
    };
    let Some(mv) = super::pyerrors::trap(crate::builtins::w_memoryview_new(w_obj)) else {
        return std::ptr::null_mut();
    };
    let roots = pyre_object::gc_roots::push_roots();
    let slot = pyre_object::gc_roots::shadow_stack_len();
    let mv = roots.pin_root(mv);
    let view = unsafe { pyre_object::memoryview::w_memoryview_view(mv) };
    let refuse = |kind, message: &str| {
        super::pyerrors::set_pending_error(crate::PyError::new(kind, message.to_string()));
        std::ptr::null_mut()
    };
    if buffertype == PYBUF_WRITE && view.readonly() {
        return refuse(
            crate::PyErrorKind::BufferError,
            "underlying buffer is not writable",
        );
    }
    let shape = unsafe { view.native_shape() };
    let strides = unsafe { view.native_strides() };
    let indirect = !unsafe { view.native_suboffsets() }.is_empty();
    let laid_out = !indirect
        && contiguous_in(order, &shape, &strides, view.itemsize(), unsafe {
            view.length()
        });
    if laid_out {
        return pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(slot));
    }
    if buffertype == PYBUF_WRITE {
        return refuse(
            crate::PyErrorKind::BufferError,
            "writable contiguous buffer requested for a non-contiguous object.",
        );
    }
    // `memory_from_contiguous_copy` — the bytes laid out in `order`, kept in a
    // `bytes` object so the copy is collected with the view, carrying the
    // source's format, itemsize and shape with the steps that layout implies.
    let itemsize = view.itemsize();
    let fortran = order as u8 == b'F';
    let format = unsafe { view.format_str() }.to_owned();
    let block = unsafe { view.gather_order(fortran) };
    let copy_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(pyre_object::bytesobject::w_bytes_from_bytes(&block));
    let fmt_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(pyre_object::w_str_new_managed(&format));
    let copy = pyre_object::gc_roots::shadow_stack_get(copy_slot);
    let w_fmt = pyre_object::gc_roots::shadow_stack_get(fmt_slot);
    let out = pyre_object::memoryview::w_memoryview_alloc_header(false, false);
    let backing = pyre_object::buffer::Buffer::String { w_obj: copy };
    let length = block.len() as i64;
    let built = if shape.len() <= 1 {
        pyre_object::bufferview::BufferView::Raw {
            backing,
            w_obj: copy,
            w_fmt,
            itemsize,
            length,
        }
    } else {
        pyre_object::bufferview::BufferView::CBuffer {
            backing,
            w_obj: copy,
            w_fmt,
            itemsize,
            length,
            ndim: shape.len() as i64,
            offset: 0,
            strides: pyre_object::bufferview::contiguous_strides(&shape, itemsize, fortran),
            shape,
            suboffsets: Vec::new(),
        }
    };
    unsafe {
        pyre_object::memoryview::w_memoryview_set_view(
            out,
            pyre_object::memoryview::bufferview_alloc(built),
        )
    };
    pyobject::make_ref(out)
}

fn unowned_view(
    address: usize,
    length: i64,
    itemsize: i64,
    format: &str,
    readonly: bool,
) -> PyObjectRef {
    use pyre_object::buffer::Buffer;
    use pyre_object::bufferview::BufferView;
    let roots = pyre_object::gc_roots::push_roots();
    let fmt_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(pyre_object::w_str_new_managed(format));
    let mv = pyre_object::memoryview::w_memoryview_alloc_header(false, false);
    let w_obj = pyre_object::w_none();
    let built = BufferView::Raw {
        backing: Buffer::External {
            w_obj,
            address,
            size: length as usize,
            readonly,
        },
        w_obj,
        w_fmt: pyre_object::gc_roots::shadow_stack_get(fmt_slot),
        itemsize,
        length,
    };
    unsafe {
        pyre_object::memoryview::w_memoryview_set_view(
            mv,
            pyre_object::memoryview::bufferview_alloc(built),
        )
    };
    mv
}

/// The [`unowned_view`] sibling for a view that carries geometry of its own.
fn unowned_view_nd(geometry: Geometry, format: &str) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let fmt_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(pyre_object::w_str_new_managed(format));
    let mv = pyre_object::memoryview::w_memoryview_alloc_header(false, false);
    let w_obj = pyre_object::w_none();
    let built = carrier_for(
        geometry,
        w_obj,
        pyre_object::gc_roots::shadow_stack_get(fmt_slot),
    );
    unsafe {
        pyre_object::memoryview::w_memoryview_set_view(
            mv,
            pyre_object::memoryview::bufferview_alloc(built),
        )
    };
    mv
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyObject_CheckBuffer as *const ());
    std::hint::black_box(PyObject_GetBuffer as *const ());
    std::hint::black_box(PyBuffer_Release as *const ());
    std::hint::black_box(PyBuffer_FillInfo as *const ());
    std::hint::black_box(PyBuffer_IsContiguous as *const ());
    std::hint::black_box(PyBuffer_SizeFromFormat as *const ());
    std::hint::black_box(PyBuffer_GetPointer as *const ());
    std::hint::black_box(PyBuffer_ToContiguous as *const ());
    std::hint::black_box(PyBuffer_FromContiguous as *const ());
    std::hint::black_box(PyObject_CopyData as *const ());
    std::hint::black_box(PyObject_AsCharBuffer as *const ());
    std::hint::black_box(PyObject_AsReadBuffer as *const ());
    std::hint::black_box(PyObject_AsWriteBuffer as *const ());
    std::hint::black_box(PyObject_CheckReadBuffer as *const ());
    std::hint::black_box(PyMemoryView_Check as *const ());
    std::hint::black_box(PyMemoryView_FromObject as *const ());
    std::hint::black_box(PyMemoryView_FromMemory as *const ());
    std::hint::black_box(PyMemoryView_FromBuffer as *const ());
    std::hint::black_box(PyMemoryView_GetContiguous as *const ());
}

#[cfg(test)]
mod tests {
    /// A request word is one thing two places spell: an extension compiled
    /// against `include/pyre3.15t/object.h` builds it, and this file reads it.
    /// Nothing tied the two together -- upstream has no such pair, because
    /// `api.py` reads each value out of the real header with
    /// `rffi_platform.ConstantInteger` -- so this walks the header and rejects
    /// any flag whose bits the Rust side spells differently.
    ///
    /// Half the header's `PyBUF_` macros name other macros rather than a
    /// number, so the walk resolves them: it keeps what it has read and
    /// evaluates an `A | B` body against that.
    #[test]
    fn every_buffer_flag_is_the_bit_the_header_gives_it() {
        use bitflags::Flags as _;

        const HEADER: &str = include_str!("../../../../include/pyre3.15t/object.h");

        let mut header: Vec<(&str, std::ffi::c_int)> = Vec::new();
        let lookup = |known: &Vec<(&str, std::ffi::c_int)>, name: &str| {
            known
                .iter()
                .find(|(spelt, _)| *spelt == name)
                .map(|(_, value)| *value)
        };
        for line in HEADER.lines() {
            let Some(rest) = line.strip_prefix("#define PyBUF_") else {
                continue;
            };
            let Some((name, body)) = rest.split_once(' ') else {
                continue;
            };
            let body = body.trim().trim_start_matches('(').trim_end_matches(')');
            let mut value = Some(0);
            for term in body.split('|') {
                let term = term.trim();
                let read = if let Some(digits) = term.strip_prefix("0x") {
                    std::ffi::c_int::from_str_radix(digits, 16).ok()
                } else if let Some(named) = term.strip_prefix("PyBUF_") {
                    lookup(&header, named)
                } else {
                    term.parse::<std::ffi::c_int>().ok()
                };
                value = value.zip(read).map(|(a, b)| a | b);
            }
            if let Some(value) = value {
                header.push((name, value));
            }
        }

        // A parser that read nothing would agree with every flag below, so the
        // floor comes first.  The header declares 21 `PyBUF_` macros, all of
        // which resolve.
        assert_eq!(
            header.len(),
            21,
            "object.h declares {} PyBUF_ macros this could resolve, not 21",
            header.len()
        );

        for flag in super::BufFlags::FLAGS {
            let name = flag
                .name()
                .strip_prefix("PYBUF_")
                .expect("every flag is declared under the header's own name");
            let ours = flag.value().bits();
            let theirs = lookup(&header, name)
                .unwrap_or_else(|| panic!("object.h declares no PyBUF_{name}"));
            assert_eq!(
                ours, theirs,
                "PyBUF_{name} is {theirs:#x} in object.h and {ours:#x} here"
            );
        }

        // The three that are not request bits: a dimension bound and the two
        // directions `PyMemoryView_FromMemory` takes.
        for (name, ours) in [
            ("MAX_NDIM", super::PYBUF_MAX_NDIM as std::ffi::c_int),
            ("READ", super::PYBUF_READ),
            ("WRITE", super::PYBUF_WRITE),
        ] {
            let theirs = lookup(&header, name)
                .unwrap_or_else(|| panic!("object.h declares no PyBUF_{name}"));
            assert_eq!(
                ours, theirs,
                "PyBUF_{name} is {theirs} in object.h and {ours} here"
            );
        }
    }
}
