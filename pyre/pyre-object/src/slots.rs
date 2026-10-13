//! Shared `__slots__` storage helpers for native subclasses (float, complex).
//!
//! Each host object stores its `__slots__` values in a list held by a
//! `w_slots: PyObjectRef` field. Reads and deletes take that storage value
//! directly, the way `mapdict.py`'s `MapdictStorageMixin._mapdict_read_storage`
//! does (`return self.storage[storageindex]`) — NOT the way
//! `BaseUserClassMapdict.getslotvalue` does, which reaches the value through
//! the promoted map with `attrkind = SLOTS_STARTING_FROM + slotindex`.
//! Growing writes use a compile-time expansion so each concrete layout
//! reloads its own field after a GC move; no runtime function-pointer
//! dispatch exists in the translated graph.
//!
//! PRE-EXISTING-ADAPTATION: the storage OWNER differs from upstream. pyre
//! indexes a `w_slots` list by slot index and [`slot_del`] writes `PY_NULL`
//! into it; `mapdict.py`'s `delslotvalue` deletes the attribute from the map
//! and installs a new storage+map pair. Nothing here is the mapdict
//! attribute machinery, so read the citations above as naming the shape of
//! one read, not as a claim of mapdict parity.

use crate::pyobject::*;

/// Read slot `index`, or `None` when unset (null list or null entry).
///
/// # Safety
/// `slots` must be null or a valid slot-storage list.
pub unsafe fn slot_get(slots: PyObjectRef, index: usize) -> Option<PyObjectRef> {
    if slots.is_null() {
        return None;
    }
    unsafe { crate::listobject::w_list_getitem(slots, index as i64) }
        .filter(|value| !value.is_null())
}

/// Expand mapdict-style slot storage into a concrete host layout.
///
/// Upstream copies the storage accessors INTO each concrete class at
/// translation time — `mapdict.py` spells it
/// `objectmodel.import_from_mixin(MapdictStorageMixin)` plus
/// `_share_methods(BaseUserClassMapdict, Object)` — so every class ends up
/// with its own direct `self.storage[...]` access and no dispatch. Pyre has
/// several native host layouts carrying the field, and a Rust
/// function-pointer accessor would add an indirect call upstream does not
/// have. This macro is that same translation-time method copy: after
/// expansion Charon sees only direct `$field` reads/writes on `$ty`,
/// including the required reload after each allocating list grow.
#[macro_export]
macro_rules! slot_set_direct {
    ($obj:expr, $index:expr, $value:expr, $ty:ty, $field:ident) => {{
        let obj = $obj;
        let index = $index;
        let value = $value;
        let slots = unsafe { (*(obj as *const $ty)).$field };
        if !slots.is_null() && unsafe { $crate::listobject::w_list_len(slots) } > index {
            unsafe { $crate::listobject::w_list_setitem(slots, index as i64, value) };
        } else {
            let _roots = $crate::gc_roots::push_roots();
            let root_base = $crate::gc_roots::shadow_stack_len();
            let _ = $crate::gc_roots::pin_root(obj);
            let _ = $crate::gc_roots::pin_root(value);
            if slots.is_null() {
                let new_slots = $crate::listobject::w_list_new(vec![$crate::PY_NULL; index + 1]);
                let obj = $crate::gc_roots::shadow_stack_get(root_base);
                unsafe { (*(obj as *mut $ty)).$field = new_slots };
                $crate::gc_hook::try_gc_write_barrier(obj as $crate::gc_hook::GCREF);
            } else {
                let mut slots = slots;
                while unsafe { $crate::listobject::w_list_len(slots) } <= index {
                    unsafe { $crate::listobject::w_list_append(slots, $crate::PY_NULL) };
                    let obj = $crate::gc_roots::shadow_stack_get(root_base);
                    slots = unsafe { (*(obj as *const $ty)).$field };
                }
            }
            let obj = $crate::gc_roots::shadow_stack_get(root_base);
            let slots = unsafe { (*(obj as *const $ty)).$field };
            let value = $crate::gc_roots::shadow_stack_get(root_base + 1);
            unsafe { $crate::listobject::w_list_setitem(slots, index as i64, value) };
        }
    }};
}

/// Clear slot `index`, returning whether it held a value beforehand.
///
/// # Safety
/// `slots` must be null or a valid slot-storage list.
pub unsafe fn slot_del(slots: PyObjectRef, index: usize) -> bool {
    if slots.is_null() || unsafe { crate::listobject::w_list_len(slots) } <= index {
        return false;
    }
    let present = unsafe { crate::listobject::w_list_getitem(slots, index as i64) }
        .is_some_and(|value| !value.is_null());
    if present {
        unsafe { crate::listobject::w_list_setitem(slots, index as i64, PY_NULL) };
    }
    present
}
