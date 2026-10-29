//! Pointer hiding — PyPy: `pypy/module/_cffi_backend/hide_reveal.py`.
//!
//! Pyre's stable GC allocations are the non-moving address-space case of
//! `HideRevealCast`: the hidden pointer is the object address, and reveal
//! first asks the collector whether the address is still managed before
//! checking the payload type and flavor.

use pyre_object::PyObjectRef;
use pyre_object::gc_hook::GCREF;

use super::cdataobj::{self, W_CData};

/// `HideRevealCast.hide_object` for a `W_CDataHandle`.
pub fn hide_object(w_handle: PyObjectRef) -> *mut u8 {
    // hide_reveal.py HideRevealCast.hide_object: hide_nonmovable_gcref + rffi.cast(PTR, raw)
    w_handle.cast::<u8>()
}

/// `HideRevealCast.reveal_object` for a `W_CDataHandle`.
pub fn reveal_object(ptr: *mut u8) -> Option<PyObjectRef> {
    // hide_reveal.py HideRevealCast.reveal_object / rgc.reveal_gcref
    let gcref = ptr as GCREF;
    if ptr.is_null() || !pyre_object::gc_hook::try_gc_owns_object(gcref) {
        return None;
    }
    let obj = gcref.cast::<pyre_object::PyObject>();
    W_CData::from_obj(obj)
        .filter(|cdata| cdata.flavor == cdataobj::FLAVOR_HANDLE)
        .map(|_| obj)
}

/// `hide_reveal1.hide_object` for a `W_CDataCallback`.
pub fn hide_callback(w_callback: PyObjectRef) -> *mut u8 {
    // hide_reveal.py HideRevealCast.hide_object: hide_nonmovable_gcref + rffi.cast(PTR, raw)
    w_callback.cast::<u8>()
}

/// `hide_reveal1.reveal_object` for a `W_CDataCallback`.
pub fn reveal_callback(ptr: *mut u8) -> Option<PyObjectRef> {
    // hide_reveal.py HideRevealCast.reveal_object / rgc.reveal_gcref
    let gcref = ptr as GCREF;
    if ptr.is_null() || !pyre_object::gc_hook::try_gc_owns_object(gcref) {
        return None;
    }
    let obj = gcref.cast::<pyre_object::PyObject>();
    W_CData::from_obj(obj)
        .filter(|cdata| cdata.flavor == cdataobj::FLAVOR_CALLBACK)
        .map(|_| obj)
}
