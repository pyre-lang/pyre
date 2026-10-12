//! Growable storage for the remaining MIFrame Ref-register mirrors.
//!
//! Like MIFrame.registers_r, the list has a stable owner even when its backing
//! allocation grows. Only value access escapes: no slot borrow can survive a
//! collection. This preserves the existing semantic-mirror layout while its
//! consumers converge on the codewriter's register colors.

use majit_ir::OpRef;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Default, Debug)]
pub struct RegisterList(Rc<RefCell<Option<Vec<OpRef>>>>);

impl From<Vec<OpRef>> for RegisterList {
    fn from(values: Vec<OpRef>) -> Self {
        Self(Rc::new(RefCell::new(Some(values))))
    }
}

impl RegisterList {
    pub fn len(&self) -> usize {
        self.0.borrow().as_ref().map_or(0, Vec::len)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn is_some(&self) -> bool {
        self.0.borrow().is_some()
    }
    pub fn is_none(&self) -> bool {
        !self.is_some()
    }
    pub fn as_ref(&self) -> Option<&Self> {
        self.is_some().then_some(self)
    }
    pub fn as_mut(&self) -> Option<&Self> {
        self.as_ref()
    }
    pub fn get_or_insert_with(&self, make: impl FnOnce() -> Vec<OpRef>) -> &Self {
        if self.is_none() {
            self.replace(make());
        }
        self
    }
    pub fn get(&self, index: usize) -> Option<OpRef> {
        self.0.borrow().as_ref()?.get(index).copied()
    }
    pub fn set(&self, index: usize, value: OpRef) {
        self.0.borrow_mut().as_mut().expect("absent register list")[index] = value;
    }
    pub fn resize(&self, len: usize, value: OpRef) {
        self.0
            .borrow_mut()
            .get_or_insert_with(Vec::new)
            .resize(len, value);
    }
    pub fn truncate(&self, len: usize) {
        if let Some(values) = self.0.borrow_mut().as_mut() {
            values.truncate(len);
        }
    }
    pub fn swap(&self, a: usize, b: usize) {
        self.0
            .borrow_mut()
            .as_mut()
            .expect("absent register list")
            .swap(a, b);
    }
    pub fn extend(&self, values: impl IntoIterator<Item = OpRef>) {
        // Evaluate the producer before borrowing the list: producers may call
        // back into the tracer (or allocate), just as RPython list extension
        // cannot keep an exclusive Rust borrow over such a call.
        for value in values {
            self.0.borrow_mut().get_or_insert_with(Vec::new).push(value);
        }
    }
    pub fn replace(&self, values: Vec<OpRef>) {
        *self.0.borrow_mut() = Some(values);
    }
    pub fn iter(&self) -> impl Iterator<Item = OpRef> + '_ {
        (0..self.len()).map(|i| self.get(i).expect("register list shrank during iteration"))
    }
    pub fn to_vec(&self) -> Vec<OpRef> {
        self.0.borrow().clone().unwrap_or_default()
    }
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn root(&self) -> RegisterListRoot {
        self.root_with(majit_metainterp::FrontendResrefOwner::NONE)
    }

    pub(crate) fn root_with(
        &self,
        owner: majit_metainterp::FrontendResrefOwner,
    ) -> RegisterListRoot {
        let payload = super::FrontendBoxArea::new(Rc::as_ptr(&self.0), owner);
        let area = unsafe {
            majit_gc::shadow_stack::MutatorExtraAreaGuard::new(
                walk_register_list,
                std::ptr::from_ref(payload.as_ref()).cast(),
                "miframe_ref_list",
            )
        };
        RegisterListRoot {
            _area: area,
            _payload: payload,
            _owner: self.clone(),
        }
    }
}

pub(crate) struct RegisterListRoot {
    _area: majit_gc::shadow_stack::MutatorExtraAreaGuard,
    _payload: Box<super::FrontendBoxArea>,
    _owner: RegisterList,
}

unsafe fn walk_register_list(data: *const (), visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    // Registered storage is Rc-owned, not part of a borrowed PyreSym. All
    // mutator methods release the list borrow before any collecting call.
    let area = unsafe { &*(data as *const super::FrontendBoxArea) };
    let list = unsafe { &*(area.storage as *const RefCell<Option<Vec<OpRef>>>) };
    if let Some(values) = list.borrow_mut().as_mut() {
        for value in values {
            super::walk_frontend_box(value, area.owner, visitor);
        }
    }
}

impl super::RegisterValues for RegisterList {
    fn len(&self) -> usize {
        self.len()
    }
    fn get_box(&self, index: usize) -> Option<OpRef> {
        self.get(index)
    }
}
