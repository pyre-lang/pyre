//! Storage for `pyjitpl.py MIFrame.registers_{i,r,f}`.
//!
//! A frame owns the list; execution and `MetaInterp.replace_box` share its
//! identity, not a captured mutable slice. Slot access returns values so a
//! caller cannot retain an exclusive slot borrow across a descendant call.
//! Construction follows `MIFrame.copy_constants`/frame setup: the full
//! register-and-constant list is seeded before publication. Its size stays
//! fixed during execution; Cell permits shared access to the actual slots.

use majit_ir::OpRef;
use std::cell::Cell;
use std::rc::Rc;

/// Value-only access used while the remaining typed banks are migrated.
pub trait RegisterValues {
    fn len(&self) -> usize;
    fn get_box(&self, index: usize) -> Option<OpRef>;
}

impl RegisterValues for [OpRef] {
    fn len(&self) -> usize {
        <[OpRef]>::len(self)
    }
    fn get_box(&self, index: usize) -> Option<OpRef> {
        self.get(index).copied()
    }
}

impl RegisterValues for Vec<OpRef> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn get_box(&self, index: usize) -> Option<OpRef> {
        self.as_slice().get(index).copied()
    }
}

impl<const N: usize> RegisterValues for [OpRef; N] {
    fn len(&self) -> usize {
        N
    }
    fn get_box(&self, index: usize) -> Option<OpRef> {
        self.as_slice().get(index).copied()
    }
}

#[derive(Clone, Default, Debug)]
pub struct RegisterBank(Option<Rc<RegisterSlots>>, usize);

#[derive(Debug)]
struct RegisterSlots(Box<[Cell<OpRef>]>);

impl std::ops::Deref for RegisterSlots {
    type Target = [Cell<OpRef>];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// The translated stack root for MIFrame's register list. Publication owns
/// the same list as execution, not a pointer into a borrowed PyreSym.
pub(crate) struct RegisterBankRoot {
    // Retire the area before releasing its owner (field drop order).
    _area: Option<majit_gc::shadow_stack::MutatorExtraAreaGuard>,
    _owner: RegisterBank,
}

unsafe fn walk_register_slots(data: *const (), visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let slots = unsafe { &*(data as *const RegisterSlots) };
    for slot in slots.iter() {
        let mut value = slot.get();
        // history.py `ConstPtr.value` is inline on the Box; `RefFrontendOp._resref`
        // is a field of that same Box in `MIFrame.registers_r`.
        value.walk_const_ptr_refs_mut(visitor);
        majit_metainterp::forward_live_frontend_resref(value, visitor);
        slot.set(value);
    }
}

impl RegisterValues for RegisterBank {
    fn len(&self) -> usize {
        self.len()
    }
    fn get_box(&self, index: usize) -> Option<OpRef> {
        self.get(index)
    }
}

impl RegisterBank {
    pub fn new(values: impl IntoIterator<Item = OpRef>) -> Self {
        let slots: Vec<_> = values.into_iter().map(Cell::new).collect();
        let num_regs = slots.len();
        Self::from_slots(slots, num_regs)
    }

    /// MIFrame.setup/copy_constants: only the prefix contains mutable
    /// registers; the suffix holds the JitCode's constants.
    pub fn with_constants(values: impl IntoIterator<Item = OpRef>, num_regs: usize) -> Self {
        Self::from_slots(values.into_iter().map(Cell::new).collect(), num_regs)
    }

    fn from_slots(slots: Vec<Cell<OpRef>>, num_regs: usize) -> Self {
        assert!(num_regs <= slots.len());
        // MIFrame.setup leaves registers_f=None when there are no registers
        // or constants. Do not allocate an empty shared bank for every call.
        if slots.is_empty() {
            Self::default()
        } else {
            Self(
                Some(Rc::new(RegisterSlots(slots.into_boxed_slice()))),
                num_regs,
            )
        }
    }

    pub fn len(&self) -> usize {
        self.0.as_ref().map_or(0, |slots| slots.len())
    }

    pub(crate) fn root(&self) -> RegisterBankRoot {
        let area = self.0.as_ref().map(|slots| unsafe {
            // The Rc allocation is stable and the returned guard retains it.
            // The collector accesses only shared Cell slots, never a frame's
            // exclusive borrow or Rc's reference count. Registry walks run
            // synchronously on the owner or under foreign-mutator STW.
            majit_gc::shadow_stack::MutatorExtraAreaGuard::new(
                walk_register_slots,
                Rc::as_ptr(slots).cast(),
                "miframe_registers",
            )
        });
        RegisterBankRoot {
            _area: area,
            _owner: self.clone(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<OpRef> {
        self.0.as_ref()?.get(index).map(Cell::get)
    }

    pub fn set(&self, index: usize, value: OpRef) {
        self.0.as_ref().expect("write to an empty register bank")[index].set(value);
    }

    pub fn to_vec(&self) -> Vec<OpRef> {
        self.0
            .iter()
            .flat_map(|slots| slots.iter().map(Cell::get))
            .collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = OpRef> + '_ {
        self.0.iter().flat_map(|slots| slots.iter().map(Cell::get))
    }

    pub fn replace_active_box(&self, old: OpRef, new: OpRef) {
        // pyjitpl.py MIFrame.replace_active_box_in_frame stops at num_regs,
        // not num_regs_and_consts: replacement must not rewrite constants.
        for slot in self.0.iter().flat_map(|slots| slots.iter()).take(self.1) {
            if slot.get() == old {
                slot.set(new);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_replacement_updates_live_aliases_but_preserves_constants() {
        // MIFrame.setup leaves absent register lists unallocated.
        let empty = RegisterBank::new([]);
        assert!(empty.0.is_none());
        // pyjitpl.py MIFrame.replace_active_box_in_frame: all three banks
        // replace every live alias, stopping before the constant suffix.
        for (old, new) in [
            (
                OpRef::const_ptr(majit_ir::GcRef(0x1000)),
                OpRef::input_arg_ref(0),
            ),
            (OpRef::const_int(7), OpRef::input_arg_int(0)),
            (OpRef::const_float(2.5), OpRef::input_arg_float(0)),
        ] {
            let owner = RegisterBank::with_constants([old, old, old], 2);
            let paused = owner.clone();
            owner.set(1, new);
            paused.replace_active_box(old, new);
            assert_eq!(owner.to_vec(), [new, new, old]);
            drop(owner);
            paused.replace_active_box(new, old);
            assert_eq!(paused.to_vec(), [old, old, old]);
            let constants_only = RegisterBank::with_constants([old], 0);
            constants_only.replace_active_box(old, new);
            assert_eq!(constants_only.to_vec(), [old]);
        }
    }
}
