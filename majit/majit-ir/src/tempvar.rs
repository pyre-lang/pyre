//! Backend register-allocation scratch box.
//!
//! `regalloc.py TempVar(AbstractResOpOrInputArg)` exists to own
//! `_forwarded`. The optimizer never sees one.

use crate::forwarding::{Forwarded, ForwardingHost};
use crate::resoperation::ForwardedSlot;
use std::rc::Rc;

/// `regalloc.py TempVar`.
pub struct TempVar {
    pub id: u32,
    pub forwarded: ForwardedSlot,
}

/// Shared `TempVar` identity.
pub type TempVarRc = Rc<TempVar>;

impl TempVar {
    pub fn new(id: u32) -> Self {
        TempVar {
            id,
            forwarded: ForwardedSlot::new(Forwarded::None),
        }
    }

    pub fn new_rc(id: u32) -> TempVarRc {
        Rc::new(Self::new(id))
    }
}

impl ForwardingHost for TempVar {
    fn get_forwarded(&self) -> Forwarded {
        self.forwarded.borrow()
    }
    fn store_forwarded(&self, value: Forwarded) {
        self.forwarded.set(value);
    }
    fn packed_forwarded_word(&self) -> u64 {
        self.forwarded.packed_word()
    }
}

/// Packed `_forwarded` word of a live `TempVar`, without cloning the slot.
///
/// SAFETY: `tv` must point at a `TempVar` kept alive by a strong count.
#[inline]
pub(crate) unsafe fn packed_forwarded_of_tempvar(tv: *const TempVar) -> u64 {
    unsafe { (*tv).forwarded.packed_word() }
}

impl std::fmt::Debug for TempVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TempVar").field("id", &self.id).finish()
    }
}
