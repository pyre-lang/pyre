//! GC carrier for `pypy.module.cpyext.buffer.CPyBuffer`.
//!
//! This is an interpreter-level GC structure, not a `W_Root`: the C buffer
//! pointer is taken exactly once by the cpyext finalizer queue, while `w_obj`
//! keeps the exporter alive until that release runs.

use std::sync::atomic::{AtomicU32, Ordering};

use crate::lltype::{GcType, TypeIdCell};
use crate::pyobject::{PY_NULL, PyObjectRef};

pub static CPYBUFFER_CARRIER_GC_TYPE_ID_CELL: TypeIdCell = TypeIdCell::auto();

static CPYBUFFER_CARRIER_GC_TYPE_ID: AtomicU32 = AtomicU32::new(0);

pub fn set_cpybuffer_carrier_gc_type_id(id: u32) {
    CPYBUFFER_CARRIER_GC_TYPE_ID.store(id, Ordering::Relaxed);
}

fn cpybuffer_carrier_gc_type_id() -> u32 {
    CPYBUFFER_CARRIER_GC_TYPE_ID.load(Ordering::Relaxed)
}

/// `CPyBuffer._immutable_ = True` (`pypy.module.cpyext.buffer.CPyBuffer`).
#[repr(C)]
#[majit_macros::jit_immutable_fields("buffer", "w_obj")]
pub struct CPyBufferCarrier {
    pub buffer: *mut u8,
    pub w_obj: PyObjectRef,
}

impl GcType for CPyBufferCarrier {
    fn type_id() -> u32 {
        cpybuffer_carrier_gc_type_id()
    }

    const SIZE: usize = std::mem::size_of::<Self>();
}

/// Allocate before attaching the managed exporter reference. The caller pins
/// the result, reloads the exporter after this collection point, and stores it
/// with [`set_w_obj`].
pub fn alloc(buffer: *mut u8) -> *mut CPyBufferCarrier {
    let mut allocation_root: crate::gc_hook::GCREF = std::ptr::null_mut();
    let mut needs_write_barrier = true;
    let raw = unsafe {
        crate::gc_hook::try_gc_alloc_collecting_rooted(
            cpybuffer_carrier_gc_type_id(),
            std::mem::size_of::<CPyBufferCarrier>(),
            &mut allocation_root,
            &mut needs_write_barrier,
        )
    };
    let payload = CPyBufferCarrier {
        buffer,
        w_obj: PY_NULL,
    };
    match crate::gc_hook::GcAllocOutcome::from_hook(raw).allocated_or_raise() {
        Some(raw) => unsafe {
            std::ptr::write(raw as *mut CPyBufferCarrier, payload);
            if needs_write_barrier {
                crate::gc_hook::try_gc_write_barrier_managed(raw);
            }
            raw as *mut CPyBufferCarrier
        },
        None if crate::gc_hook::gc_alloc_failed_pending() => std::ptr::null_mut(),
        None => crate::lltype::malloc_typed(payload),
    }
}

/// Store the exporter after both it and `carrier` have been rooted.
///
/// # Safety
/// `carrier` must point to a live [`CPyBufferCarrier`].
pub unsafe fn set_w_obj(carrier: *mut CPyBufferCarrier, w_obj: PyObjectRef) {
    unsafe {
        (*carrier).w_obj = w_obj;
    }
    crate::gc_hook::try_gc_write_barrier_managed(carrier.cast());
}

/// Take the exact C buffer once; later calls return null.
///
/// # Safety
/// `carrier` must point to a live [`CPyBufferCarrier`].
pub unsafe fn take_buffer(carrier: *mut CPyBufferCarrier) -> *mut u8 {
    unsafe {
        let buffer = (*carrier).buffer;
        (*carrier).buffer = std::ptr::null_mut();
        buffer
    }
}

/// The Python exporter kept alive by `carrier`.
///
/// # Safety
/// `carrier` must point to a live [`CPyBufferCarrier`].
pub unsafe fn w_obj(carrier: *mut CPyBufferCarrier) -> PyObjectRef {
    unsafe { (*carrier).w_obj }
}

/// Drop the managed exporter edge after `CPyBuffer.releasebuffer` returns.
///
/// # Safety
/// `carrier` must point to a live [`CPyBufferCarrier`].
pub unsafe fn clear_w_obj(carrier: *mut CPyBufferCarrier) {
    unsafe {
        (*carrier).w_obj = PY_NULL;
    }
}

/// Whether `obj` is the internal carrier rather than a Python object. The GC
/// type id is the discriminator; no payload word is read as an object header.
pub fn is_carrier(obj: PyObjectRef) -> bool {
    let carrier_tid = cpybuffer_carrier_gc_type_id();
    carrier_tid != 0
        && majit_gc::get_actual_typeid(majit_ir::GcRef(obj as usize)) == Some(carrier_tid)
}
