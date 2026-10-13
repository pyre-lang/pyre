//! View layer for the buffer protocol — the pyre analogue of
//! `pypy/interpreter/buffer.py`'s `BufferView`.  A `BufferView` carries the
//! geometry (offset / shape / strides / format / itemsize) over a byte-level
//! [`Buffer`] and gathers the live logical bytes in C order, honouring a
//! strided or N-D layout, without detaching a copy of the backing.
//!
//! `memoryview`'s `W_MemoryView` holds one of these off the GC heap; the GC
//! reaches the refs inside (the backing exporter, the `.obj` exporter, and
//! any stored format / shape / strides objects) through `W_MemoryView`'s
//! custom trace.  The specialised variants (`Simple` / `Raw` / `Slice`)
//! derive their geometry natively and store only exporter refs plus scalars;
//! callers that need a Python `str` / `tuple` wrap a fresh one at the `descr`
//! boundary.

use crate::buffer::Buffer;
use crate::pyobject::PyObjectRef;

/// `_copy_base` — push one `isz`-wide element at byte offset `base`, dropping
/// it when the address falls outside the backing (a reversed / strided slice
/// past the end), so the gather never panics.
fn copy_base(full: &[u8], base: i64, isz: usize, out: &mut Vec<u8>) {
    if isz > 0 && base >= 0 && base as usize + isz <= full.len() {
        let b = base as usize;
        out.extend_from_slice(&full[b..b + isz]);
    }
}

/// `_copy_rec` — recursive C-order copy of dimension `idim`.  The innermost
/// dimension walks `shape[ndim-1]` elements by `strides[ndim-1]`; an outer
/// dimension recurses `shape[idim]` times, advancing `off` by `strides[idim]`.
///
/// A stride of zero addresses the same element at every index of that
/// dimension, so it is copied `shape[idim]` times rather than skipped:
/// `_copy_base` stops early only because `range(off, off, 0)` is not a walk.
#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
)]
fn copy_rec(
    full: &[u8],
    shape: &[i64],
    strides: &[i64],
    ndim: i64,
    idim: i64,
    mut off: i64,
    isz: usize,
    out: &mut Vec<u8>,
) {
    let dimshape = shape.get(idim as usize).copied().unwrap_or(0);
    let dimstride = strides.get(idim as usize).copied().unwrap_or(0);
    if idim == ndim - 1 {
        for _ in 0..dimshape {
            copy_base(full, off, isz, out);
            off += dimstride;
        }
    } else {
        for _ in 0..dimshape {
            copy_rec(full, shape, strides, ndim, idim + 1, off, isz, out);
            off += dimstride;
        }
    }
}

/// `ADJUST_PTR` — follow the pointer stored at `ptr` when dimension `dim`
/// carries a non-negative suboffset, and leave it alone otherwise.  An empty
/// slice is identity, exactly as the macro is for a NULL `suboffsets`.
///
/// # Safety
/// When dimension `dim` has a non-negative suboffset, `ptr` must address a
/// live pointer that the exporter published.
unsafe fn adjust_ptr(ptr: *const u8, suboffsets: &[i64], dim: usize) -> *const u8 {
    match suboffsets.get(dim) {
        Some(&suboffset) if suboffset >= 0 => unsafe {
            (*ptr.cast::<*const u8>()).offset(suboffset as isize)
        },
        _ => ptr,
    }
}

/// `init_strides_from_shape` / `init_fortran_strides_from_shape` — the byte
/// steps of a block laid out contiguously in the requested order.
pub fn contiguous_strides(shape: &[i64], itemsize: i64, fortran: bool) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];
    let mut step = itemsize;
    if fortran {
        for (dim, &extent) in shape.iter().enumerate() {
            strides[dim] = step;
            step *= extent;
        }
    } else {
        for (dim, &extent) in shape.iter().enumerate().rev() {
            strides[dim] = step;
            step *= extent;
        }
    }
    strides
}

/// `copy_rec` for a view whose dimensions carry suboffsets.  The cursor is a
/// raw pointer because a suboffset dereferences it, so no byte index into the
/// backing describes the element; the address is chained one dimension at a
/// time and the stride advances the *un*adjusted cursor.
///
/// The source is always walked outermost dimension first, because a pointer
/// chain has to be followed in the order it was built.  A Fortran-order
/// result therefore comes from the *destination* steps `dest_strides`
/// carries, which is how `buffer_to_contiguous` produces one.
///
/// # Safety
/// The geometry must be the one its exporter published, every pointer a
/// suboffset reaches must be live, and `out` must be `product(shape) * isz`
/// bytes long.
#[expect(
    clippy::too_many_arguments,
    reason = "The source geometry, the destination geometry and both cursors are each a separate argument; grouping them would hide which side an argument belongs to"
)]
unsafe fn copy_rec_indirect(
    shape: &[i64],
    strides: &[i64],
    suboffsets: &[i64],
    dest_strides: &[i64],
    idim: usize,
    ptr: *const u8,
    dest: i64,
    isz: usize,
    out: &mut [u8],
) {
    let dimshape = shape.get(idim).copied().unwrap_or(0);
    let dimstride = strides.get(idim).copied().unwrap_or(0);
    let deststride = dest_strides.get(idim).copied().unwrap_or(0);
    let mut cursor = ptr;
    let mut at = dest;
    for _ in 0..dimshape {
        let element = unsafe { adjust_ptr(cursor, suboffsets, idim) };
        if idim + 1 == shape.len() {
            let to = at.max(0) as usize;
            if to + isz <= out.len() {
                out[to..to + isz]
                    .copy_from_slice(unsafe { std::slice::from_raw_parts(element, isz) });
            }
        } else {
            unsafe {
                copy_rec_indirect(
                    shape,
                    strides,
                    suboffsets,
                    dest_strides,
                    idim + 1,
                    element,
                    at,
                    isz,
                    out,
                )
            };
        }
        cursor = unsafe { cursor.offset(dimstride as isize) };
        at += deststride;
    }
}

/// `buffer_to_contiguous(..., 'F')` — recursive Fortran-order copy.  The
/// first dimension is innermost, so dimension 0 changes fastest in the
/// destination byte string (`memoryobject.c:init_fortran_strides_from_shape`).
fn copy_rec_fortran(
    full: &[u8],
    shape: &[i64],
    strides: &[i64],
    idim: i64,
    mut off: i64,
    isz: usize,
    out: &mut Vec<u8>,
) {
    let dimshape = shape.get(idim as usize).copied().unwrap_or(0);
    let dimstride = strides.get(idim as usize).copied().unwrap_or(0);
    if idim == 0 {
        for _ in 0..dimshape {
            copy_base(full, off, isz, out);
            off += dimstride;
        }
    } else {
        for _ in 0..dimshape {
            copy_rec_fortran(full, shape, strides, idim - 1, off, isz, out);
            off += dimstride;
        }
    }
}

/// Read a `tuple[int]` (shape or strides) into a native vector.
///
/// # Safety
/// `t` must point to a valid tuple of ints.
unsafe fn read_dims(t: PyObjectRef) -> Vec<i64> {
    unsafe {
        let n = crate::tupleobject::w_tuple_len(t);
        let mut dims: Vec<i64> = Vec::with_capacity(n);
        for i in 0..n {
            let d = crate::tupleobject::w_tuple_getitem(t, i as i64)
                .map(|w| crate::intobject::w_int_get_value(w))
                .unwrap_or(0);
            dims.push(d);
        }
        dims
    }
}

/// The `[low, high)` byte window a strided geometry can address, relative to
/// element zero: `low` is the sum of the negative extents a reversed stride
/// reaches below element zero, `high` the sum of the positive ones plus one
/// element.  An empty extent addresses nothing, so the window is `(0, 0)`.
///
/// This has no upstream twin — `get_offset` and `ADJUST_PTR` both index raw
/// pointers, which need no window.  It exists because [`Buffer::as_bytes`]
/// hands out a *bounded* slice and [`copy_base`] drops any element outside
/// it, so a foreign export whose strides reach below its `buf` must have its
/// storage lowered to `low` and the shift carried in the view's offset.
pub fn addressable_window(shape: &[i64], strides: &[i64], itemsize: i64) -> (i64, i64) {
    if shape.contains(&0) {
        return (0, 0);
    }
    let mut low = 0;
    let mut high = itemsize;
    for (&extent, &stride) in shape.iter().zip(strides) {
        let span = stride * (extent - 1);
        if span < 0 {
            low += span;
        } else {
            high += span;
        }
    }
    (low, high)
}

/// A view of a [`Buffer`]'s bytes with offset / shape / stride geometry and a
/// buffer-protocol format.
///
/// One variant per class of PyPy's view hierarchy — `SimpleView` /
/// `RawBufferView` (plain 1-D), `BufferSlice` (strided slice),
/// `BufferView1D` / `BufferViewND` (cast), `ReadonlyWrapper` — each carrying
/// only the state its class stores and deriving the rest.
///
/// Views are immutable once built (`_immutable_ = True`), so `Clone` shares
/// semantics: PyPy hands the same view object to a wrapper / copy, pyre
/// clones it into the new owner's box.
#[derive(Clone)]
pub enum BufferView {
    /// `SimpleView` (`pypy/interpreter/buffer.py`) — a plain contiguous
    /// 1-D byte view (`bytes` / `bytearray`).  Format `'B'`, itemsize 1,
    /// ndim 1, offset 0, shape `[length]`, strides `[1]` are all derived;
    /// `readonly` comes from the backing, so nothing but the exporter refs is
    /// stored.
    Simple {
        backing: Buffer,
        w_obj: PyObjectRef,
        length: i64,
    },
    /// `RawBufferView` (`buffer.py`) — a typed contiguous 1-D view
    /// (`array.array`).  Format / itemsize are explicit; ndim 1, offset 0,
    /// shape `[length / itemsize]` (`[0]` when empty), strides `[itemsize]`
    /// derive; `readonly` comes from the backing.
    Raw {
        backing: Buffer,
        w_obj: PyObjectRef,
        w_fmt: PyObjectRef,
        itemsize: i64,
        length: i64,
    },
    /// `BufferSlice` (`buffer.py`) — a strided dimension-0 window over a
    /// parent view, produced by a step≠1 slice (a step==1 slice of a
    /// `Simple` / `Raw` view re-specialises over a [`Buffer::Sub`] window
    /// instead).  `start` / `step` are in parent dimension-0 element units
    /// and `length` is the element count (`shape[0]`); shape / strides /
    /// format / itemsize derive from the parent with dimension 0 replaced,
    /// so no geometry objects are stored.  A slice of a slice composes into
    /// the parent's coordinates, keeping the wrapper depth at 1.
    Slice {
        parent: Box<BufferView>,
        w_obj: PyObjectRef,
        start: i64,
        step: i64,
        length: i64,
    },
    /// `BufferView1D` (`memoryobject.py`) — a cast to a new 1-D element
    /// format over the parent's bytes (`_cast_to_1D`).  Format / itemsize are
    /// explicit; ndim 1, shape `[parent_length / itemsize]`, strides
    /// `[itemsize]` derive; byte access delegates to the parent
    /// (`IndirectView`).
    View1D {
        parent: Box<BufferView>,
        w_obj: PyObjectRef,
        w_fmt: PyObjectRef,
        itemsize: i64,
    },
    /// `BufferViewND` (`memoryobject.py`) — a cast to an N-dimensional
    /// shape over a 1-D parent (`_cast_to_ND`).  `shape` / `strides` ride as
    /// their tuple objects; format / itemsize come from the parent.
    ViewND {
        parent: Box<BufferView>,
        w_obj: PyObjectRef,
        ndim: i64,
        w_shape: PyObjectRef,
        w_strides: PyObjectRef,
    },
    /// `CPyBuffer` (`pypy/module/cpyext/buffer.py`) — a foreign `Py_buffer`'s
    /// own geometry over the storage its exporter handed over.  A root view:
    /// it holds the storage rather than a parent, exactly as `CPyBuffer`
    /// holds `self.ptr` / `self.size` instead of subclassing `IndirectView`.
    /// The backing is that export's block, or the in-heap copy a contiguous
    /// materialisation made of it.
    ///
    /// `shape` / `strides` ride as native vectors, as `CPyBuffer` stores
    /// them as lists, and both have `ndim` entries once `init_shape_strides`
    /// has reconstructed the ones the exporter left NULL.  `length` is
    /// `Py_buffer.len` verbatim (`CPyBuffer.size`), which for a 0-dimensional
    /// or PIL-style export is not `product(shape) * itemsize`.  `offset` is
    /// the byte position of element zero within `backing`, non-zero when a
    /// negative stride forced the storage down to the lowest addressable
    /// byte (see [`addressable_window`]).  An empty `suboffsets` is NULL.
    CBuffer {
        backing: Buffer,
        w_obj: PyObjectRef,
        w_fmt: PyObjectRef,
        itemsize: i64,
        length: i64,
        ndim: i64,
        offset: i64,
        shape: Vec<i64>,
        strides: Vec<i64>,
        suboffsets: Vec<i64>,
    },
    /// `ReadonlyWrapper` (`buffer.py`) — `toreadonly`'s wrapper: every
    /// read delegates to the wrapped view, `readonly` is forced true.
    Readonly {
        view: Box<BufferView>,
        w_obj: PyObjectRef,
    },
}

impl BufferView {
    /// Clone this `Py_buffer` geometry while replacing only its immediate
    /// owner.  CPython `slot_bf_getbuffer` copies the returned memoryview's
    /// `Py_buffer` and then replaces `buffer->obj` with `_buffer_wrapper`;
    /// its underlying storage and nested geometry remain unchanged.
    pub fn clone_with_obj(&self, new_obj: PyObjectRef) -> BufferView {
        match self {
            BufferView::Simple {
                backing, length, ..
            } => BufferView::Simple {
                backing: backing.clone(),
                w_obj: new_obj,
                length: *length,
            },
            BufferView::Raw {
                backing,
                w_fmt,
                itemsize,
                length,
                ..
            } => BufferView::Raw {
                backing: backing.clone(),
                w_obj: new_obj,
                w_fmt: *w_fmt,
                itemsize: *itemsize,
                length: *length,
            },
            BufferView::Slice {
                parent,
                start,
                step,
                length,
                ..
            } => BufferView::Slice {
                parent: parent.clone(),
                w_obj: new_obj,
                start: *start,
                step: *step,
                length: *length,
            },
            BufferView::View1D {
                parent,
                w_fmt,
                itemsize,
                ..
            } => BufferView::View1D {
                parent: parent.clone(),
                w_obj: new_obj,
                w_fmt: *w_fmt,
                itemsize: *itemsize,
            },
            BufferView::ViewND {
                parent,
                ndim,
                w_shape,
                w_strides,
                ..
            } => BufferView::ViewND {
                parent: parent.clone(),
                w_obj: new_obj,
                ndim: *ndim,
                w_shape: *w_shape,
                w_strides: *w_strides,
            },
            BufferView::CBuffer {
                backing,
                w_fmt,
                itemsize,
                length,
                ndim,
                offset,
                shape,
                strides,
                suboffsets,
                ..
            } => BufferView::CBuffer {
                backing: backing.clone(),
                w_obj: new_obj,
                w_fmt: *w_fmt,
                itemsize: *itemsize,
                length: *length,
                ndim: *ndim,
                offset: *offset,
                shape: shape.clone(),
                strides: strides.clone(),
                suboffsets: suboffsets.clone(),
            },
            BufferView::Readonly { view, .. } => BufferView::Readonly {
                view: view.clone(),
                w_obj: new_obj,
            },
        }
    }

    /// The backing byte storage (the root exporter's [`Buffer`]).
    #[inline]
    pub fn backing(&self) -> &Buffer {
        match self {
            BufferView::Simple { backing, .. }
            | BufferView::Raw { backing, .. }
            | BufferView::CBuffer { backing, .. } => backing,
            BufferView::Slice { parent, .. }
            | BufferView::View1D { parent, .. }
            | BufferView::ViewND { parent, .. } => parent.backing(),
            BufferView::Readonly { view, .. } => view.backing(),
        }
    }
    /// The exporter reported by `memoryview.obj`.
    #[inline]
    pub fn w_obj(&self) -> PyObjectRef {
        match self {
            BufferView::Simple { w_obj, .. }
            | BufferView::Raw { w_obj, .. }
            | BufferView::Slice { w_obj, .. }
            | BufferView::View1D { w_obj, .. }
            | BufferView::ViewND { w_obj, .. }
            | BufferView::CBuffer { w_obj, .. }
            | BufferView::Readonly { w_obj, .. } => *w_obj,
        }
    }
    /// The element format string (`getformat`), read natively — the callers
    /// that need a Python `str` wrap a fresh one at the `descr` boundary.  A
    /// `Simple` view derives `'B'`; a `Raw` view reads its explicit format.
    ///
    /// # Safety
    /// The view's stored format object must be a live `str`.
    #[inline]
    pub unsafe fn format_str(&self) -> &'static str {
        unsafe {
            match self {
                BufferView::Simple { .. } => "B",
                BufferView::Raw { w_fmt, .. }
                | BufferView::View1D { w_fmt, .. }
                | BufferView::CBuffer { w_fmt, .. } => {
                    // A lone surrogate is not a format code. Empty is already
                    // rejected by the format-length check.
                    crate::w_str_get_value_opt(*w_fmt).unwrap_or("")
                }
                BufferView::Slice { parent, .. } | BufferView::ViewND { parent, .. } => {
                    parent.format_str()
                }
                BufferView::Readonly { view, .. } => view.format_str(),
            }
        }
    }
    /// The per-dimension element counts (`getshape`) as native extents.  A
    /// `Simple` view is `[length]`; a `Raw` view is `[length / itemsize]`
    /// (`[0]` when empty, `buffer.py:254`).
    ///
    /// # Safety
    /// The view's stored shape object must be a live tuple of ints.
    #[inline]
    pub unsafe fn native_shape(&self) -> Vec<i64> {
        unsafe {
            match self {
                BufferView::Simple { length, .. } => vec![*length],
                BufferView::Raw {
                    itemsize, length, ..
                } => {
                    if *length == 0 {
                        vec![0]
                    } else {
                        vec![*length / *itemsize]
                    }
                }
                // Dimension 0 takes the slice's element count; later
                // dimensions ride along (`shape[0] = length`, buffer.py).
                BufferView::Slice { parent, length, .. } => {
                    let mut shape = parent.native_shape();
                    if let Some(s0) = shape.first_mut() {
                        *s0 = *length;
                    }
                    shape
                }
                // `[getlength() // itemsize]` (memoryobject.py).
                BufferView::View1D {
                    parent, itemsize, ..
                } => vec![parent.length() / *itemsize],
                BufferView::ViewND { w_shape, .. } => read_dims(*w_shape),
                BufferView::CBuffer { shape, .. } => shape.clone(),
                BufferView::Readonly { view, .. } => view.native_shape(),
            }
        }
    }
    /// The per-dimension byte steps (`getstrides`) as native extents.  A
    /// `Simple` view is `[1]`; a `Raw` view is `[itemsize]`.
    ///
    /// # Safety
    /// The view's stored strides object must be a live tuple of ints.
    #[inline]
    pub unsafe fn native_strides(&self) -> Vec<i64> {
        unsafe {
            match self {
                BufferView::Simple { .. } => vec![1],
                BufferView::Raw { itemsize, .. } | BufferView::View1D { itemsize, .. } => {
                    vec![*itemsize]
                }
                // Dimension 0 steps by the parent's stride times the slice
                // step (`strides[0] *= step`, buffer.py).
                BufferView::Slice { parent, step, .. } => {
                    let mut strides = parent.native_strides();
                    if !strides.is_empty() {
                        let stride0 = strides[0] * *step;
                        strides[0] = stride0;
                    }
                    strides
                }
                BufferView::ViewND { w_strides, .. } => read_dims(*w_strides),
                BufferView::CBuffer { strides, .. } => strides.clone(),
                BufferView::Readonly { view, .. } => view.native_strides(),
            }
        }
    }
    /// The per-dimension suboffsets (`getsuboffsets`).  Empty is NULL: only a
    /// foreign export carries any, and every derived view reports its
    /// parent's.
    ///
    /// # Safety
    /// The view's stored geometry objects must be live.
    #[inline]
    pub unsafe fn native_suboffsets(&self) -> Vec<i64> {
        unsafe {
            match self {
                BufferView::Simple { .. } | BufferView::Raw { .. } => Vec::new(),
                BufferView::CBuffer { suboffsets, .. } => suboffsets.clone(),
                BufferView::Slice { parent, .. }
                | BufferView::View1D { parent, .. }
                | BufferView::ViewND { parent, .. } => parent.native_suboffsets(),
                BufferView::Readonly { view, .. } => view.native_suboffsets(),
            }
        }
    }
    /// `strides[0]` — the signed byte step between consecutive elements of a
    /// 1-D view, falling back to `itemsize` when the strides are empty.
    ///
    /// # Safety
    /// The view's stored strides object must be a live tuple of ints.
    #[inline]
    pub unsafe fn stride0(&self) -> i64 {
        unsafe {
            match self {
                BufferView::Simple { .. } => 1,
                BufferView::Raw { itemsize, .. } | BufferView::View1D { itemsize, .. } => *itemsize,
                BufferView::Slice { parent, step, .. } => parent.stride0() * *step,
                BufferView::ViewND {
                    parent, w_strides, ..
                } => crate::tupleobject::w_tuple_getitem(*w_strides, 0)
                    .map(|s| crate::intobject::w_int_get_value(s))
                    .unwrap_or_else(|| parent.itemsize()),
                BufferView::CBuffer {
                    strides, itemsize, ..
                } => strides.first().copied().unwrap_or(*itemsize),
                BufferView::Readonly { view, .. } => view.stride0(),
            }
        }
    }
    #[inline]
    pub fn itemsize(&self) -> i64 {
        match self {
            BufferView::Raw { itemsize, .. }
            | BufferView::View1D { itemsize, .. }
            | BufferView::CBuffer { itemsize, .. } => *itemsize,
            BufferView::Simple { .. } => 1,
            BufferView::Slice { parent, .. } | BufferView::ViewND { parent, .. } => {
                parent.itemsize()
            }
            BufferView::Readonly { view, .. } => view.itemsize(),
        }
    }
    #[inline]
    pub fn ndim(&self) -> i64 {
        match self {
            BufferView::Simple { .. } | BufferView::Raw { .. } | BufferView::View1D { .. } => 1,
            BufferView::Slice { parent, .. } => parent.ndim(),
            BufferView::ViewND { ndim, .. } => *ndim,
            BufferView::CBuffer { ndim, .. } => *ndim,
            BufferView::Readonly { view, .. } => view.ndim(),
        }
    }
    /// Byte offset of the view's first element within the backing storage.
    /// A `Slice` starts `start` parent elements past the parent's own offset
    /// (`getbytes` adds `self.start * self.parent.getstrides()[0]`,
    /// buffer.py:340).
    ///
    /// # Safety
    /// A `Slice` parent's stored strides object must be live.
    #[inline]
    pub unsafe fn offset(&self) -> i64 {
        unsafe {
            match self {
                BufferView::Simple { .. } | BufferView::Raw { .. } => 0,
                BufferView::Slice { parent, start, .. } => {
                    parent.offset() + *start * parent.stride0()
                }
                // IndirectView delegates byte access to the parent unshifted.
                BufferView::View1D { parent, .. } | BufferView::ViewND { parent, .. } => {
                    parent.offset()
                }
                BufferView::CBuffer { offset, .. } => *offset,
                BufferView::Readonly { view, .. } => view.offset(),
            }
        }
    }
    /// `getlength` — the view's byte count.  A `Slice` spans `shape[0]`
    /// elements (`buffer.py:336`); a `ViewND` spans `product(shape)` elements
    /// (`memoryobject.py:922`).
    ///
    /// # Safety
    /// A `ViewND`'s stored shape object must be a live tuple of ints.
    #[inline]
    pub unsafe fn length(&self) -> i64 {
        unsafe {
            match self {
                BufferView::Simple { length, .. }
                | BufferView::Raw { length, .. }
                | BufferView::CBuffer { length, .. } => *length,
                // `init_len` — `product(shape) * itemsize`, so a slice of an
                // N-dimensional parent keeps the trailing extents.  `length`
                // is `shape[0]` alone (`BufferSlice.getlength`), which
                // under-reports by their product.
                BufferView::Slice { parent, .. } => {
                    self.native_shape().iter().product::<i64>() * parent.itemsize()
                }
                BufferView::View1D { parent, .. } => parent.length(),
                BufferView::ViewND {
                    parent, w_shape, ..
                } => {
                    let shape = read_dims(*w_shape);
                    // CPython 3.14 exposes a resized zero-field ctypes
                    // Structure as a 0-D, itemsize-zero buffer whose nbytes
                    // is nevertheless the full resized backing.  Preserve
                    // that parent's byte extent so `view.cast("B")` can
                    // flatten it; ordinary BufferViewND follows PyPy's
                    // product(shape) * itemsize calculation below.
                    if shape.is_empty() && parent.itemsize() == 0 {
                        parent.length()
                    } else {
                        shape.iter().product::<i64>() * parent.itemsize()
                    }
                }
                BufferView::Readonly { view, .. } => view.length(),
            }
        }
    }
    #[inline]
    pub fn readonly(&self) -> bool {
        match self {
            BufferView::Simple { backing, .. }
            | BufferView::Raw { backing, .. }
            | BufferView::CBuffer { backing, .. } => backing.readonly(),
            BufferView::Slice { parent, .. }
            | BufferView::View1D { parent, .. }
            | BufferView::ViewND { parent, .. } => parent.readonly(),
            BufferView::Readonly { .. } => true,
        }
    }

    /// `new_slice(start, step, slicelength)` — a live dimension-0 sub-view
    /// sharing the parent's storage (`buffer.py:149,261,310,403`).  `start` /
    /// `slicelength` are in dimension-0 element units.
    ///
    /// A step==1 slice of a `Simple` / `Raw` view re-specialises over a
    /// [`Buffer::Sub`] window of the same storage, so the result stays a
    /// plain contiguous view; every other case wraps the parent in a
    /// [`Slice`](BufferView::Slice).  Slicing a `Slice` composes into its
    /// parent's coordinates — `start` maps through `parent_index`
    /// (`self.start + self.step * idx`, buffer.py) so a re-slice of a
    /// strided slice lands on the elements the composed stride selects.
    ///
    /// # Safety
    /// The view's stored geometry objects must be live.
    pub unsafe fn new_slice(&self, start: i64, step: i64, slicelength: i64) -> BufferView {
        let wrap = |view: &BufferView| BufferView::Slice {
            parent: Box::new(view.clone()),
            w_obj: view.w_obj(),
            start,
            step,
            length: slicelength,
        };
        match self {
            BufferView::Simple { backing, w_obj, .. } if step == 1 => BufferView::Simple {
                backing: Buffer::sub(backing.clone(), start as usize, slicelength),
                w_obj: *w_obj,
                length: slicelength,
            },
            BufferView::Raw {
                backing,
                w_obj,
                w_fmt,
                itemsize,
                ..
            } if step == 1 => {
                let n = *itemsize;
                BufferView::Raw {
                    backing: Buffer::sub(backing.clone(), (start * n) as usize, slicelength * n),
                    w_obj: *w_obj,
                    w_fmt: *w_fmt,
                    itemsize: n,
                    length: slicelength * n,
                }
            }
            BufferView::Slice {
                parent,
                w_obj,
                start: pstart,
                step: pstep,
                ..
            } => BufferView::Slice {
                parent: parent.clone(),
                w_obj: *w_obj,
                start: pstart + pstep * start,
                step: pstep * step,
                length: slicelength,
            },
            // A foreign export wraps like any other non-specialised parent:
            // `Slice::offset` is `parent.offset() + start * parent.stride0()`
            // and `Slice::native_strides` rescales dimension 0 only, which is
            // `adjust_buf` plus `strides[dim] *= step` at `dim == 0`.  The arm
            // is spelled out rather than left to the fallback so the
            // suboffset case has a place to land.
            BufferView::CBuffer { .. } => wrap(self),
            // A slice of a read-only wrapper stays wrapped (buffer.py:462).
            BufferView::Readonly { w_obj, .. } => BufferView::Readonly {
                view: Box::new(wrap(self)),
                w_obj: *w_obj,
            },
            other => wrap(other),
        }
    }

    /// `ptr_from_index` / `ptr_from_tuple` — the address of the element at
    /// `indices`, one wrapped index per dimension.  The address is *chained*
    /// rather than summed: each dimension's stride advances the cursor and
    /// `ADJUST_PTR` may then follow a pointer stored there.
    ///
    /// # Safety
    /// `indices` must already be bounds-checked against the shape, and the
    /// backing export must be live.
    pub unsafe fn element_ptr(&self, indices: &[i64]) -> *const u8 {
        unsafe {
            let base = self.backing().as_bytes().as_ptr();
            self.walk_to(base, indices)
        }
    }

    /// [`element_ptr`](Self::element_ptr) over writable storage, or `None`
    /// when the backing refuses one.
    ///
    /// # Safety
    /// As [`element_ptr`](Self::element_ptr).
    pub unsafe fn element_ptr_mut(&self, indices: &[i64]) -> Option<*mut u8> {
        unsafe {
            let base = self.backing().as_bytes_mut()?.as_mut_ptr();
            Some(self.walk_to(base as *const u8, indices) as *mut u8)
        }
    }

    /// # Safety
    /// As [`element_ptr`](Self::element_ptr).
    unsafe fn walk_to(&self, base: *const u8, indices: &[i64]) -> *const u8 {
        unsafe {
            let strides = self.native_strides();
            let suboffsets = self.native_suboffsets();
            let mut ptr = base.offset(self.offset() as isize);
            for (dim, &index) in indices.iter().enumerate() {
                let stride = strides.get(dim).copied().unwrap_or(0);
                ptr = ptr.offset((stride * index) as isize);
                ptr = adjust_ptr(ptr, &suboffsets, dim);
            }
            ptr
        }
    }

    /// The LIVE logical bytes of the view (`buffer.py as_str`), read from the
    /// backing object's own storage — no detached copy — so the view observes
    /// later mutation of a bytearray / array source.  Honours offset / shape /
    /// strides so a strided slice (`m[::2]`, `m[::-1]`) or an N-D view gathers
    /// the right elements in C order.
    ///
    /// # Safety
    /// The backing [`Buffer`]'s `w_obj` must point to a live object of its
    /// tagged kind.
    pub unsafe fn gather(&self) -> Vec<u8> {
        unsafe { self.gather_order(false) }
    }

    /// Borrow the live byte run when this view already has the `BUF_SIMPLE`
    /// shape PyPy's `StringUnmarshaller` consumes directly.
    ///
    /// A step-one slice over `Simple`/`Raw` is normalized into a
    /// [`Buffer::Sub`] by [`new_slice`](Self::new_slice), so the common pyc
    /// payload view remains in these two arms and [`Buffer::as_bytes`] already
    /// applies its offset and length. More general geometry deliberately
    /// returns `None`; callers then use [`gather`](Self::gather), preserving
    /// the existing strided/N-D semantics without pretending it is one run.
    ///
    /// # Safety
    /// The backing export must remain live and unresized while the returned
    /// slice is used, exactly as for [`Buffer::as_bytes`].
    pub unsafe fn as_contiguous_bytes(&self) -> Option<&'static [u8]> {
        unsafe {
            match self {
                BufferView::Simple {
                    backing, length, ..
                }
                | BufferView::Raw {
                    backing, length, ..
                } => {
                    let bytes = backing.as_bytes();
                    let length = ((*length).max(0) as usize).min(bytes.len());
                    Some(&bytes[..length])
                }
                BufferView::Readonly { view, .. } => view.as_contiguous_bytes(),
                BufferView::Slice { .. }
                | BufferView::View1D { .. }
                | BufferView::ViewND { .. }
                | BufferView::CBuffer { .. } => None,
            }
        }
    }

    /// The `PyBuffer_ToContiguous` copy for C order (`fortran == false`) or
    /// Fortran order (`fortran == true`).  CPython constructs destination
    /// strides for the requested order and copies matching logical indices;
    /// iterating the source coordinates in that destination order is the same
    /// operation without materialising a second `Py_buffer`.
    ///
    /// # Safety
    /// The backing [`Buffer`]'s `w_obj` must point to a live object of its
    /// tagged kind.
    pub unsafe fn gather_order(&self, fortran: bool) -> Vec<u8> {
        unsafe {
            let itemsize = self.itemsize();
            let ndim = self.ndim();
            let offset = self.offset();
            let full = self.backing().as_bytes();
            let isz = itemsize.max(0) as usize;
            if ndim == 0 {
                let mut out = Vec::with_capacity(isz);
                copy_base(full, offset, isz, &mut out);
                return out;
            }
            let shape = self.native_shape();
            let strides = self.native_strides();
            let count = if itemsize > 0 {
                self.length() / itemsize
            } else {
                0
            };
            let mut out = Vec::with_capacity(count.max(0) as usize * isz);
            let suboffsets = self.native_suboffsets();
            if !suboffsets.is_empty() {
                // No byte index into the backing describes an element behind a
                // suboffset, so the walk carries an address instead.  The
                // bounds clamp `copy_base` applies goes with the index: the
                // exporter's pointers are the only description of the target.
                let mut block = vec![0u8; count.max(0) as usize * isz];
                copy_rec_indirect(
                    &shape,
                    &strides,
                    &suboffsets,
                    &contiguous_strides(&shape, itemsize, fortran),
                    0,
                    full.as_ptr().offset(offset as isize),
                    0,
                    isz,
                    &mut block,
                );
                return block;
            }
            if fortran {
                copy_rec_fortran(full, &shape, &strides, ndim - 1, offset, isz, &mut out);
            } else {
                copy_rec(full, &shape, &strides, ndim, 0, offset, isz, &mut out);
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Geometry-only tests: `Simple` / `Raw` derive everything from scalars,
    // so a fake exporter address is never dereferenced.
    fn fake(addr: usize) -> PyObjectRef {
        addr as PyObjectRef
    }

    fn simple(len: i64) -> BufferView {
        BufferView::Simple {
            backing: Buffer::String {
                w_obj: fake(0x1000),
            },
            w_obj: fake(0x1000),
            length: len,
        }
    }

    #[test]
    fn simple_step1_slice_respecializes_over_sub_window() {
        // SimpleView.new_slice step==1 (buffer.py): the result is another
        // SimpleView over a SubBuffer window, not a BufferSlice wrapper.
        let s = unsafe { simple(10).new_slice(2, 1, 5) };
        match &s {
            BufferView::Simple {
                backing: Buffer::Sub { offset, size, .. },
                length,
                ..
            } => {
                assert_eq!((*offset, *size, *length), (2, 5, 5));
            }
            _ => panic!("expected Simple over Sub"),
        }
        unsafe {
            assert_eq!(s.native_shape(), vec![5]);
            assert_eq!(s.native_strides(), vec![1]);
            assert_eq!(s.offset(), 0);
        }
    }

    #[test]
    fn simple_step1_slice_borrows_its_exact_sub_window() {
        let exporter = crate::bytesobject::w_bytes_from_bytes(b"0123456789");
        let view = BufferView::Simple {
            backing: Buffer::String { w_obj: exporter },
            w_obj: exporter,
            length: 10,
        };
        let sliced = unsafe { view.new_slice(2, 1, 5) };

        assert_eq!(unsafe { sliced.as_contiguous_bytes() }, Some(&b"23456"[..]));
    }

    #[test]
    fn raw_step1_slice_scales_the_window_by_itemsize() {
        // RawBufferView.new_slice step==1 (buffer.py:262-265): the SubBuffer
        // window is `start * itemsize .. + slicelength * itemsize`.
        let raw = BufferView::Raw {
            backing: Buffer::Array {
                w_obj: fake(0x2000),
            },
            w_obj: fake(0x2000),
            w_fmt: fake(0x2004),
            itemsize: 4,
            length: 40,
        };
        let s = unsafe { raw.new_slice(1, 1, 3) };
        match &s {
            BufferView::Raw {
                backing: Buffer::Sub { offset, size, .. },
                itemsize,
                length,
                ..
            } => {
                assert_eq!((*offset, *size, *itemsize, *length), (4, 12, 4, 12));
            }
            _ => panic!("expected Raw over Sub"),
        }
        unsafe {
            assert_eq!(s.native_shape(), vec![3]);
            assert_eq!(s.native_strides(), vec![4]);
        }
    }

    #[test]
    fn strided_step_wraps_in_a_slice_with_derived_geometry() {
        let s = unsafe { simple(10).new_slice(0, 2, 5) };
        assert!(matches!(s, BufferView::Slice { .. }));
        assert!(unsafe { s.as_contiguous_bytes() }.is_none());
        unsafe {
            assert_eq!(s.native_shape(), vec![5]);
            assert_eq!(s.native_strides(), vec![2]);
            assert_eq!(s.offset(), 0);
        }
        assert_eq!(s.itemsize(), 1);
        assert_eq!(s.ndim(), 1);
        assert_eq!(unsafe { s.length() }, 5);
    }

    #[test]
    fn slice_of_slice_composes_through_the_parent_step() {
        // Re-slicing a strided slice maps `start` through `parent_index`
        // (buffer.py:386) and multiplies the steps, collapsing to one
        // wrapper over the original parent.
        let s2 = unsafe { simple(10).new_slice(0, 2, 5).new_slice(1, 1, 2) };
        match &s2 {
            BufferView::Slice {
                parent,
                start,
                step,
                length,
                ..
            } => {
                assert_eq!((*start, *step, *length), (2, 2, 2));
                assert!(matches!(**parent, BufferView::Simple { .. }));
            }
            _ => panic!("expected collapsed Slice"),
        }
        unsafe {
            assert_eq!(s2.offset(), 2);
            assert_eq!(s2.native_strides(), vec![2]);
        }
    }

    fn cbuffer(shape: Vec<i64>, strides: Vec<i64>, itemsize: i64, offset: i64) -> BufferView {
        let length = shape.iter().product::<i64>() * itemsize;
        BufferView::CBuffer {
            backing: Buffer::External {
                w_obj: fake(0x3000),
                address: 0x40000,
                size: length.max(0) as usize,
                readonly: false,
            },
            w_obj: fake(0x3000),
            w_fmt: fake(0x3004),
            itemsize,
            length,
            ndim: shape.len() as i64,
            offset,
            shape,
            strides,
            suboffsets: vec![],
        }
    }

    #[test]
    fn cbuffer_reports_its_stored_geometry() {
        // A foreign export answers from its own vectors rather than deriving
        // any of them from a parent, and `length` is `Py_buffer.len` verbatim.
        let view = cbuffer(vec![2, 3], vec![12, 4], 4, 0);
        unsafe {
            assert_eq!(view.native_shape(), vec![2, 3]);
            assert_eq!(view.native_strides(), vec![12, 4]);
            assert_eq!(view.stride0(), 12);
            assert_eq!(view.offset(), 0);
            assert_eq!(view.length(), 24);
            assert!(view.as_contiguous_bytes().is_none());
        }
        assert_eq!(view.itemsize(), 4);
        assert_eq!(view.ndim(), 2);
        assert!(!view.readonly());
    }

    #[test]
    fn cbuffer_negative_stride_window_lowers_the_base() {
        // A reversed dimension addresses bytes below element zero, so the
        // window starts there and the view's offset carries the shift.
        assert_eq!(addressable_window(&[4], &[-4], 4), (-12, 4));
        assert_eq!(addressable_window(&[2, 3], &[12, 4], 4), (0, 24));
        assert_eq!(addressable_window(&[2, 3], &[12, -4], 4), (-8, 16));
        // An empty extent addresses nothing whatever the strides say.
        assert_eq!(addressable_window(&[0, 3], &[12, 4], 4), (0, 0));
        // A zero-dimensional export still addresses its one element.
        assert_eq!(addressable_window(&[], &[], 8), (0, 8));
    }

    /// A `CBuffer` over `block`, whose address the geometry-only helper above
    /// never dereferences but these two tests do.
    fn cbuffer_over(
        block: *const u8,
        size: usize,
        shape: Vec<i64>,
        strides: Vec<i64>,
        suboffsets: Vec<i64>,
        itemsize: i64,
    ) -> BufferView {
        BufferView::CBuffer {
            backing: Buffer::External {
                w_obj: fake(0x3000),
                address: block as usize,
                size,
                readonly: true,
            },
            w_obj: fake(0x3000),
            w_fmt: fake(0x3004),
            itemsize,
            length: shape.iter().product::<i64>() * itemsize,
            ndim: shape.len() as i64,
            offset: 0,
            shape,
            strides,
            suboffsets,
        }
    }

    #[test]
    fn indirect_gather_follows_the_row_pointers() {
        // `ADJUST_PTR`: dimension 0 holds a table of row pointers and its
        // suboffset is zero, so the walk dereferences the table entry and
        // continues into the row; dimension 1's -1 leaves the cursor alone.
        let row0: [u8; 3] = [1, 2, 3];
        let row1: [u8; 3] = [4, 5, 6];
        let table: [*const u8; 2] = [row0.as_ptr(), row1.as_ptr()];
        let view = cbuffer_over(
            table.as_ptr().cast::<u8>(),
            std::mem::size_of_val(&table),
            vec![2, 3],
            vec![std::mem::size_of::<*const u8>() as i64, 1],
            vec![0, -1],
            1,
        );
        assert_eq!(unsafe { view.gather() }, vec![1, 2, 3, 4, 5, 6]);
        // Fortran order reads dimension 0 fastest through the same table.
        assert_eq!(unsafe { view.gather_order(true) }, vec![1, 4, 2, 5, 3, 6]);
        // One element, addressed by chaining rather than summing.
        assert_eq!(unsafe { *view.element_ptr(&[1, 2]) }, 6);
    }

    #[test]
    fn direct_gather_is_unchanged_when_suboffsets_are_empty() {
        // The pointer walk is entered only for a view that has suboffsets;
        // an empty vector keeps the byte-index walk, strides and all.
        let block: [u8; 6] = [1, 2, 3, 4, 5, 6];
        let view = cbuffer_over(
            block.as_ptr(),
            block.len(),
            vec![2, 3],
            vec![3, 1],
            vec![],
            1,
        );
        assert_eq!(unsafe { view.gather() }, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(unsafe { view.gather_order(true) }, vec![1, 4, 2, 5, 3, 6]);
        assert_eq!(unsafe { *view.element_ptr(&[1, 2]) }, 6);
        // A reversed dimension 0 walks backwards from element zero, which
        // `acquire` makes addressable by lowering the storage to the window's
        // low end and carrying the shift in the view's offset.
        let (low, high) = addressable_window(&[2, 3], &[-3, 1], 1);
        assert_eq!((low, high), (-3, 3));
        let reversed = BufferView::CBuffer {
            backing: Buffer::External {
                w_obj: fake(0x3000),
                address: unsafe { block.as_ptr().offset(3).offset(low as isize) } as usize,
                size: (high - low) as usize,
                readonly: true,
            },
            w_obj: fake(0x3000),
            w_fmt: fake(0x3004),
            itemsize: 1,
            length: 6,
            ndim: 2,
            offset: -low,
            shape: vec![2, 3],
            strides: vec![-3, 1],
            suboffsets: vec![],
        };
        assert_eq!(unsafe { reversed.gather() }, vec![4, 5, 6, 1, 2, 3]);
    }

    #[test]
    fn cbuffer_backing_stays_the_external_export() {
        // `release_view` ends a foreign export only for a view whose backing
        // is the `External` block the export handed over; a variant that
        // stopped answering with one would silently never be released.
        let view = cbuffer(vec![2, 3], vec![12, 4], 4, 0);
        assert!(matches!(view.backing(), Buffer::External { .. }));
    }

    #[test]
    fn slice_of_a_cbuffer_shifts_the_base_and_rescales_dim0() {
        // The wrapper composes: dimension 0 takes the slice's extent and its
        // stride is multiplied by the step, later dimensions ride along.
        let s = unsafe { cbuffer(vec![4, 3], vec![12, 4], 4, 0).new_slice(1, 2, 2) };
        assert!(matches!(s, BufferView::Slice { .. }));
        unsafe {
            assert_eq!(s.native_shape(), vec![2, 3]);
            assert_eq!(s.native_strides(), vec![24, 4]);
            assert_eq!(s.offset(), 12);
            assert_eq!(s.length(), 24);
        }
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.itemsize(), 4);
    }

    #[test]
    fn slice_of_an_nd_view_reports_product_shape_nbytes() {
        // `init_len` is `product(shape) * itemsize`, so a dimension-0 slice
        // of an N-dimensional view keeps the trailing extents; reporting
        // `shape[0]` alone under-reports by their product.
        let exporter = crate::bytesobject::w_bytes_from_bytes(&[0u8; 10]);
        let flat = BufferView::Simple {
            backing: Buffer::String { w_obj: exporter },
            w_obj: exporter,
            length: 10,
        };
        let dims = |extents: [i64; 2]| {
            crate::tupleobject::w_tuple_new(
                extents
                    .iter()
                    .map(|&extent| crate::intobject::w_int_new(extent))
                    .collect(),
            )
        };
        let nd = BufferView::ViewND {
            parent: Box::new(flat),
            w_obj: exporter,
            ndim: 2,
            w_shape: dims([2, 5]),
            w_strides: dims([5, 1]),
        };
        let s = unsafe { nd.new_slice(0, 2, 1) };
        unsafe {
            assert_eq!(s.native_shape(), vec![1, 5]);
            assert_eq!(s.native_strides(), vec![10, 1]);
            assert_eq!(s.length(), 5);
        }
    }

    #[test]
    fn reversed_slice_starts_at_the_high_end() {
        let s = unsafe { simple(10).new_slice(9, -1, 10) };
        unsafe {
            assert_eq!(s.offset(), 9);
            assert_eq!(s.native_strides(), vec![-1]);
            assert_eq!(s.native_shape(), vec![10]);
        }
        assert!(s.readonly()); // String backing stays readonly through the wrapper
    }
}
