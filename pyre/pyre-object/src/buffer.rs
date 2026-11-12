//! Byte-storage layer for the buffer protocol — the pyre analogue of
//! `rpython/rlib/buffer.py`'s `Buffer` hierarchy.
//!
//! Each variant tags the concrete exporter, so a byte read dispatches to that
//! exporter's own storage accessor and a `bytes`/`bytearray`/`array` *subclass*
//! is read through its own fields.  The concrete kind is decided once, at
//! construction time, by the objspace-level `buffer_w` — which lives in the
//! interpreter crate because picking the variant needs `isinstance_w`, a
//! dependency pyre-object must not take.  The `memoryview` `BufferView` /
//! `W_MemoryView` layers sit on top of this.

use crate::pyobject::PyObjectRef;

/// `Buffer.get_raw_address` failure (`rpython/rlib/buffer.py`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawAddressError {
    /// The buffer cannot give a raw address (`ValueError`).
    NoAddress,
    /// `str2charp` / `get_raw_address_of_string` OOM; MemoryError is pending.
    Memory,
}

/// Flat byte storage behind a buffer-protocol exporter.
///
/// `Clone` shares the storage, not the bytes: a variant holds only the
/// exporter ref (plus window scalars for `Sub`), so a clone is another
/// window onto the same live storage — how PyPy shares one immutable
/// `Buffer` object between views.
#[derive(Clone)]
pub enum Buffer {
    /// `bytes` — read-only (`StringBuffer`).
    String { w_obj: PyObjectRef },
    /// `bytearray` — mutable (`ByteBuffer`).
    Byte { w_obj: PyObjectRef },
    /// `array.array` — its raw element bytes.
    Array { w_obj: PyObjectRef },
    /// Raw foreign memory used by `ctypes.memoryview_at`.
    External {
        w_obj: PyObjectRef,
        address: usize,
        size: usize,
        readonly: bool,
    },
    /// A `[offset, offset+size)` window over another `Buffer` (`SubBuffer`,
    /// `rpython/rlib/buffer.py`).  Sub-buffers never nest — see [`sub`].
    ///
    /// `size` is signed: a negative value (canonically `-1`) means "up to the
    /// end of the parent" (`buffer.py:398`), so it cannot be a `usize`.
    ///
    /// [`sub`]: Buffer::sub
    Sub {
        parent: Box<Buffer>,
        offset: usize,
        size: i64,
    },
}

/// Drop the export an `External` window holds on its owner.  The owners that
/// keep a count (`mmap`) live above this crate, so the interpreter installs
/// the callback at start-up; until then, and for owners that keep no count
/// (`ctypes.memoryview_at`, whose `w_obj` is `None`), releasing is a no-op.
static EXTERNAL_RELEASE: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
static EXTERNAL_ACQUIRE: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
static EXTERNAL_VIEW: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
static EXTERNAL_LAYOUT: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// `W_MMap.readbuf_w` / `writebuf_w`: address, length, readonly.
/// `None` is "not this exporter"; `Err` is a closed mapping.
pub type ExternalBufferViewFn =
    fn(PyObjectRef) -> Option<Result<(usize, usize, bool), &'static str>>;

/// # Safety
/// `f` must stay valid for the process lifetime and accept any `w_obj` that
/// was stored in an `External` buffer.
pub unsafe fn set_external_release_hook(f: unsafe fn(PyObjectRef)) {
    EXTERNAL_RELEASE.store(f as usize, core::sync::atomic::Ordering::Release);
}

/// Install the counted-external exporter (`mmap`) beside the release hook.
///
/// # Safety
/// Each function must stay valid for the process lifetime.
pub unsafe fn set_external_buffer_exporter(
    view: ExternalBufferViewFn,
    acquire: fn(PyObjectRef),
    release: unsafe fn(PyObjectRef),
    has_layout: fn(PyObjectRef) -> bool,
) {
    EXTERNAL_VIEW.store(view as usize, core::sync::atomic::Ordering::Release);
    EXTERNAL_ACQUIRE.store(acquire as usize, core::sync::atomic::Ordering::Release);
    EXTERNAL_RELEASE.store(release as usize, core::sync::atomic::Ordering::Release);
    EXTERNAL_LAYOUT.store(has_layout as usize, core::sync::atomic::Ordering::Release);
}

/// The live mapping window, or `None` when `obj` is not the counted
/// external exporter.
pub fn external_buffer_view(
    obj: PyObjectRef,
) -> Option<Result<(usize, usize, bool), &'static str>> {
    let f = EXTERNAL_VIEW.load(core::sync::atomic::Ordering::Acquire);
    if f == 0 {
        return None;
    }
    unsafe { core::mem::transmute::<usize, ExternalBufferViewFn>(f)(obj) }
}

/// # Safety
/// `obj` must be the exporter paired with a successful view.
pub unsafe fn external_buffer_acquire(obj: PyObjectRef) {
    let f = EXTERNAL_ACQUIRE.load(core::sync::atomic::Ordering::Acquire);
    if f != 0 && !obj.is_null() {
        unsafe { core::mem::transmute::<usize, fn(PyObjectRef)>(f)(obj) }
    }
}

/// Whether `obj` has the counted-external exporter layout, including a
/// Python subclass.
pub fn has_external_buffer_layout(obj: PyObjectRef) -> bool {
    let f = EXTERNAL_LAYOUT.load(core::sync::atomic::Ordering::Acquire);
    if f == 0 {
        return false;
    }
    unsafe { core::mem::transmute::<usize, fn(PyObjectRef) -> bool>(f)(obj) }
}

/// # Safety
/// `w_obj` must be the exporter paired with a successful acquisition.
pub unsafe fn external_buffer_release(w_obj: PyObjectRef) {
    unsafe { external_release(w_obj) };
}

unsafe fn external_release(w_obj: PyObjectRef) {
    let f = EXTERNAL_RELEASE.load(core::sync::atomic::Ordering::Acquire);
    if f != 0 && !w_obj.is_null() {
        unsafe { core::mem::transmute::<usize, unsafe fn(PyObjectRef)>(f)(w_obj) }
    }
}

impl Buffer {
    /// Release the single exporter lock owned by a root memoryview.
    /// Derived/sub views do not call this; their root view owns the count.
    /// # Safety
    /// The caller must uphold every validity, runtime-type, aliasing, and lifetime
    /// invariant required by the object and pointer arguments for the entire call.
    pub unsafe fn release_export(&self) {
        unsafe {
            match self {
                Buffer::Byte { w_obj } => {
                    crate::bytearrayobject::w_bytearray_exports_decref(*w_obj)
                }
                Buffer::Array { w_obj } => crate::interp_array::w_array_exports_decref(*w_obj),
                Buffer::String { .. } => {}
                Buffer::External { w_obj, .. } => external_release(*w_obj),
                Buffer::Sub { parent, .. } => parent.release_export(),
            }
        }
    }

    /// `BufferView.needs_release` (`pypy/interpreter/buffer.py`): whether
    /// [`Self::release_export`] has a side effect on the exporter.
    /// `BytearrayBuffer.needs_release` answers true; the array and external
    /// exporters here count their exports the same way, and `bytes` counts
    /// nothing.  A `Sub` answers for its parent.
    pub fn needs_release(&self) -> bool {
        match self {
            Buffer::Byte { .. } | Buffer::Array { .. } | Buffer::External { .. } => true,
            Buffer::String { .. } => false,
            Buffer::Sub { parent, .. } => parent.needs_release(),
        }
    }

    /// `SubBuffer(parent, offset, size)` (`rpython/rlib/buffer.py`).  A
    /// `Sub` over a `Sub` is collapsed to a single window over the inner
    /// buffer (`buffer.py:397` — "don't nest them"): the offsets sum and the
    /// outer window clamps to the inner one, so the wrapper depth never
    /// exceeds 1.  A negative `size` means "up to the end" (`buffer.py:398`);
    /// when the inner window is itself unbounded (`inner_size < 0`) the outer
    /// `size` is carried through unchanged, otherwise it is clamped to the
    /// bytes remaining in the inner window (`buffer.py:399-403`).
    pub fn sub(parent: Buffer, offset: usize, size: i64) -> Buffer {
        match parent {
            Buffer::Sub {
                parent: inner,
                offset: inner_off,
                size: inner_size,
            } => {
                let size = if inner_size < 0 {
                    size
                } else {
                    let at_most = (inner_size - offset as i64).max(0);
                    if size < 0 || size > at_most {
                        at_most
                    } else {
                        size
                    }
                };
                Buffer::Sub {
                    parent: inner,
                    offset: inner_off + offset,
                    size,
                }
            }
            other => Buffer::Sub {
                parent: Box::new(other),
                offset,
                size,
            },
        }
    }

    /// Whether the exporter's storage is read-only (`Buffer.readonly`,
    /// `rpython/rlib/buffer.py`).  `bytes` is immutable; `bytearray` /
    /// `array` are writable; a `Sub` inherits its parent's mutability.
    #[inline]
    pub fn readonly(&self) -> bool {
        match self {
            Buffer::String { .. } => true,
            Buffer::Byte { .. } | Buffer::Array { .. } => false,
            Buffer::External { readonly, .. } => *readonly,
            Buffer::Sub { parent, .. } => parent.readonly(),
        }
    }

    /// The root exporter object whose storage this buffer reads/writes; a
    /// `Sub` reports its parent's exporter (`SubBuffer` has no `.obj` of its
    /// own).
    #[inline]
    pub fn w_obj(&self) -> PyObjectRef {
        match self {
            Buffer::String { w_obj }
            | Buffer::Byte { w_obj }
            | Buffer::Array { w_obj }
            | Buffer::External { w_obj, .. } => *w_obj,
            Buffer::Sub { parent, .. } => parent.w_obj(),
        }
    }

    /// The byte storage this buffer exposes (`getlength` is its `.len()`).  A
    /// `Sub` is the `[offset, offset+size)` window of its parent, clamped to
    /// the parent's live length (`SubBuffer.getlength`).
    ///
    /// A `String` slice is valid until the next collection that moves the STR.
    ///
    /// # Safety
    /// The variant's `w_obj` must point to a live object of the tagged kind.
    #[inline]
    pub unsafe fn as_bytes(&self) -> &'static [u8] {
        unsafe {
            match self {
                Buffer::String { w_obj } => crate::bytesobject::w_bytes_data(*w_obj),
                Buffer::Byte { w_obj } => crate::bytearrayobject::w_bytearray_data(*w_obj),
                Buffer::Array { w_obj } => crate::interp_array::w_array_bytes(*w_obj),
                Buffer::External {
                    w_obj,
                    address,
                    size,
                    ..
                } => {
                    // A counted exporter (`mmap`) is read through its live
                    // mapping on every access, the way `MMapBuffer` reaches
                    // `W_MMap.mmap`: a view derived from a released memoryview
                    // holds no export of its own, so `close` or `resize` may
                    // have unmapped or moved the window it was built over.  A
                    // closed mapping reads as empty.  Any other owner keeps
                    // the window it was built with.
                    let (address, size) = match external_buffer_view(*w_obj) {
                        Some(Ok((address, size, _))) => (address, size),
                        Some(Err(_)) => (core::ptr::NonNull::<u8>::dangling().as_ptr() as usize, 0),
                        None => (*address, *size),
                    };
                    core::slice::from_raw_parts(address as *const u8, size)
                }
                Buffer::Sub {
                    parent,
                    offset,
                    size,
                } => {
                    // `SubBuffer.getlength` (buffer.py:411): the window is the
                    // requested `size` clamped to the bytes remaining after
                    // `offset`, with a negative `size` meaning "to the end".
                    let full = parent.as_bytes();
                    let off = (*offset).min(full.len());
                    let at_most = full.len() - off;
                    let length = if *size >= 0 && (*size as usize) <= at_most {
                        *size as usize
                    } else {
                        at_most
                    };
                    &full[off..off + length]
                }
            }
        }
    }

    /// `Buffer.get_raw_address` (`rpython/rlib/buffer.py`).
    ///
    /// A `String` buffer uses `rffi.get_raw_address_of_string`. [`RawAddressError::Memory`]
    /// means MemoryError is pending. [`RawAddressError::NoAddress`] is the
    /// default `ValueError` arm. The exporter must be rooted: moving out of
    /// the nursery can collect. A `Sub` is `parent.get_raw_address() + offset`
    /// (`SubBuffer.get_raw_address`).
    ///
    /// # Safety
    /// The variant's `w_obj` must point to a live object of the tagged kind.
    pub unsafe fn get_raw_address(&self) -> Result<*mut u8, RawAddressError> {
        unsafe {
            match self {
                Buffer::String { w_obj } => {
                    let ptr = crate::bytesobject::w_bytes_raw_address(*w_obj);
                    if ptr.is_null() {
                        Err(RawAddressError::Memory)
                    } else {
                        Ok(ptr)
                    }
                }
                Buffer::Byte { w_obj } => {
                    Ok(crate::bytearrayobject::w_bytearray_data(*w_obj).as_ptr() as *mut u8)
                }
                Buffer::Array { w_obj } => {
                    Ok(crate::interp_array::w_array_bytes(*w_obj).as_ptr() as *mut u8)
                }
                Buffer::External {
                    w_obj,
                    address,
                    size,
                    ..
                } => {
                    let (address, _) = match external_buffer_view(*w_obj) {
                        Some(Ok((address, size, _))) => (address, size),
                        Some(Err(_)) => (core::ptr::NonNull::<u8>::dangling().as_ptr() as usize, 0),
                        None => (*address, *size),
                    };
                    Ok(address as *mut u8)
                }
                Buffer::Sub { parent, offset, .. } => {
                    let base = parent.get_raw_address()?;
                    Ok(base.wrapping_add(*offset))
                }
            }
        }
    }

    /// Mutable byte storage of a writable buffer, or `None` when the
    /// exporter is read-only (`bytes`).  The variant tag already encodes the
    /// concrete exporter kind (including a subclass, tagged at construction),
    /// so dispatch needs no isinstance.  A `Sub` yields its window of the
    /// parent's storage with the same clamp as [`as_bytes`](Buffer::as_bytes).
    ///
    /// # Safety
    /// The variant's `w_obj` must point to a live object of the tagged kind.
    #[inline]
    pub unsafe fn as_bytes_mut(&self) -> Option<&'static mut [u8]> {
        unsafe {
            match self {
                Buffer::String { .. } => None,
                Buffer::Byte { w_obj } => {
                    Some(crate::bytearrayobject::w_bytearray_data_mut(*w_obj))
                }
                Buffer::Array { w_obj } => {
                    Some(crate::interp_array::w_array_vec_mut(*w_obj).as_mut_slice())
                }
                Buffer::External { readonly, .. } => {
                    if *readonly {
                        None
                    } else {
                        let window = self.as_bytes();
                        Some(core::slice::from_raw_parts_mut(
                            window.as_ptr() as *mut u8,
                            window.len(),
                        ))
                    }
                }
                Buffer::Sub {
                    parent,
                    offset,
                    size,
                } => {
                    let full = parent.as_bytes_mut()?;
                    let off = (*offset).min(full.len());
                    let at_most = full.len() - off;
                    let length = if *size >= 0 && (*size as usize) <= at_most {
                        *size as usize
                    } else {
                        at_most
                    };
                    Some(&mut full[off..off + length])
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `w_obj` / `sub` never dereference the exporter, so a fake address is a
    // valid stand-in for the geometry-only tests below.
    fn fake(addr: usize) -> PyObjectRef {
        addr as PyObjectRef
    }

    #[test]
    fn sub_wraps_a_leaf_buffer() {
        match Buffer::sub(
            Buffer::String {
                w_obj: fake(0x1000),
            },
            2,
            5,
        ) {
            Buffer::Sub {
                parent,
                offset,
                size,
            } => {
                assert_eq!((offset, size), (2, 5));
                assert!(matches!(*parent, Buffer::String { .. }));
            }
            _ => panic!("expected Sub"),
        }
    }

    #[test]
    fn sub_over_sub_collapses_to_depth_one() {
        // `SubBuffer.__init__` (buffer.py:397): the offsets sum and the parent
        // is the inner buffer, so the wrapper never nests.
        let leaf = Buffer::Byte {
            w_obj: fake(0x2000),
        };
        let nested = Buffer::sub(Buffer::sub(leaf, 2, 5), 1, 3);
        match nested {
            Buffer::Sub {
                parent,
                offset,
                size,
            } => {
                assert_eq!((offset, size), (3, 3)); // 2+1, min(3, 5-1)
                assert!(matches!(*parent, Buffer::Byte { .. }));
                assert_eq!(parent.w_obj(), fake(0x2000));
            }
            _ => panic!("expected collapsed Sub"),
        }
    }

    #[test]
    fn sub_over_sub_clamps_size_to_inner_window() {
        let nested = Buffer::sub(
            Buffer::sub(
                Buffer::Array {
                    w_obj: fake(0x3000),
                },
                4,
                6,
            ),
            2,
            100,
        );
        match nested {
            Buffer::Sub { offset, size, .. } => assert_eq!((offset, size), (6, 4)), // 4+2, 6-2
            _ => panic!("expected Sub"),
        }
    }

    #[test]
    fn sub_to_end_sentinel_carries_through_unbounded_inner() {
        // A negative size means "up to the end" (buffer.py:398); over an
        // unbounded inner window the outer sentinel is carried through.
        let leaf = Buffer::String {
            w_obj: fake(0x5000),
        };
        match Buffer::sub(Buffer::sub(leaf, 2, -1), 3, -1) {
            Buffer::Sub { offset, size, .. } => assert_eq!((offset, size), (5, -1)), // 2+3, -1
            _ => panic!("expected Sub"),
        }
    }

    #[test]
    fn sub_to_end_resolves_over_bounded_inner() {
        // Outer "to the end" (-1) over a bounded inner window resolves to the
        // inner remainder (buffer.py:399-403).
        let leaf = Buffer::Byte {
            w_obj: fake(0x6000),
        };
        match Buffer::sub(Buffer::sub(leaf, 1, 8), 2, -1) {
            Buffer::Sub { offset, size, .. } => assert_eq!((offset, size), (3, 6)), // 1+2, 8-2
            _ => panic!("expected Sub"),
        }
    }

    #[test]
    fn w_obj_recurses_through_sub_to_the_root_exporter() {
        let s = Buffer::sub(
            Buffer::Array {
                w_obj: fake(0x4000),
            },
            1,
            2,
        );
        assert_eq!(s.w_obj(), fake(0x4000));
    }
}
