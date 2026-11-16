//! `regalloc.py Lifetime(AbstractInfo)` — per-box liveness stored in `_forwarded`.
//!
//! Register slots are `RegisterManager.all_regs` indices. Frame slots are
//! frame positions (the same unit as `hint_frame_pos`). Packed as a thin
//! `Rc` under `FW_INFO_OTHER` with bit-3 subtag 1.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::rc::Rc;

use smallvec::SmallVec;

/// `regalloc.py UNDEF_POS`
pub const UNDEF_POS: i32 = -42;

/// `regalloc.py Lifetime` — liveness information for a single variable.
/// Interior mutability so a shared `Rc<Lifetime>` (`Lifetime.share_with`)
/// can be updated without a side-table lookup.
///
/// `#[repr(align(16))]` keeps the `Rc` payload 16-byte aligned so bit 3 of
/// the packed `_forwarded` word is free for the OTHER subtag.
#[repr(align(16))]
pub struct Lifetime {
    /// `Lifetime.definition_pos` — operations-slice index where the variable is defined
    pub definition_pos: Cell<i32>,
    /// `Lifetime.last_usage` — operations-slice index of last use (including failargs/jump)
    pub last_usage: Cell<i32>,
    /// `Lifetime.real_usages` — *real* usages (as op argument, excluding
    /// jump/failargs). Four inline slots cover the common SSA use-count
    /// so the first `push` does not heap-grow 16 B per live box.
    pub real_usages: RefCell<Option<SmallVec<[i32; 4]>>>,
    /// `Lifetime.fixed_positions` — `(opindex, all_regs index)` pairs
    pub fixed_positions: RefCell<Option<Vec<(i32, i32)>>>,
    /// `Lifetime.share_with` — another Lifetime that wants to share a register
    pub share_with: RefCell<Option<Rc<Lifetime>>>,
    /// `Lifetime._definition_pos_shared`
    pub _definition_pos_shared: Cell<i32>,
    /// `Lifetime.current_register_index` into RegisterManager.all_regs (-1 = not in reg)
    pub current_register_index: Cell<i32>,
    /// `Lifetime.current_frame_loc` — frame position, or `None` when not in frame
    pub current_frame_loc: Cell<Option<i32>>,
    /// `Lifetime.hint_frame_pos` — hinted frame location (at the end of the trace)
    pub hint_frame_pos: Cell<i32>,
}

impl Lifetime {
    pub fn new(definition_pos: i32, last_usage: i32) -> Self {
        Lifetime {
            definition_pos: Cell::new(definition_pos),
            last_usage: Cell::new(last_usage),
            real_usages: RefCell::new(None),
            fixed_positions: RefCell::new(None),
            share_with: RefCell::new(None),
            _definition_pos_shared: Cell::new(UNDEF_POS),
            current_register_index: Cell::new(-1),
            current_frame_loc: Cell::new(None),
            hint_frame_pos: Cell::new(-1),
        }
    }

    /// `Lifetime.last_usage_including_sharing` — follows `share_with` as a
    /// Lifetime object, with no map lookup.
    pub fn last_usage_including_sharing(&self) -> i32 {
        let mut result = self.last_usage.get();
        let mut share = self.share_with.borrow().clone();
        while let Some(lt) = share {
            result = lt.last_usage.get();
            share = lt.share_with.borrow().clone();
        }
        result
    }

    /// `Lifetime.is_last_real_use_before`
    pub fn is_last_real_use_before(&self, position: i32) -> bool {
        match &*self.real_usages.borrow() {
            None => true,
            Some(usages) => *usages.last().unwrap() <= position,
        }
    }

    /// `Lifetime.next_real_usage` — binary search for next real usage after position.
    pub fn next_real_usage(&self, position: i32) -> i32 {
        debug_assert!(position >= self.definition_pos.get());
        let usages = self.real_usages.borrow();
        let l = usages.as_ref().unwrap();
        if position >= *l.last().unwrap() {
            return -1;
        }
        let mut low: usize = 0;
        let mut high: usize = l.len();
        while low < high {
            let mid = low + (high - low) / 2;
            if position < l[mid] {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        l[low]
    }

    /// `Lifetime.definition_pos_shared`
    pub fn definition_pos_shared(&self) -> i32 {
        let shared = self._definition_pos_shared.get();
        if shared != UNDEF_POS {
            shared
        } else {
            self.definition_pos.get()
        }
    }

    /// `Lifetime.fixed_register` — `reg` is an `all_regs` index.
    pub fn fixed_register(&self, position: i32, reg: i32) -> i32 {
        debug_assert!(self.definition_pos.get() <= position && position <= self.last_usage.get());
        let mut positions = self.fixed_positions.borrow_mut();
        let res;
        if let Some(pos) = positions.as_ref() {
            debug_assert!(position > pos.last().unwrap().0);
            res = pos.last().unwrap().0;
        } else {
            *positions = Some(Vec::new());
            res = self.definition_pos_shared();
        }
        positions.as_mut().unwrap().push((position, reg));
        res
    }

    /// `Lifetime.find_fixed_register` — follows `share_with` as a Lifetime
    /// object, with no map lookup. Returns an `all_regs` index.
    pub fn find_fixed_register(&self, opindex: i32) -> Option<i32> {
        if let Some(ref positions) = *self.fixed_positions.borrow() {
            for &(index, reg) in positions {
                if opindex <= index {
                    return Some(reg);
                }
            }
        }
        if let Some(share_lt) = self.share_with.borrow().clone() {
            return share_lt.find_fixed_register(opindex);
        }
        None
    }

    /// `Lifetime.check_invariants`
    pub fn check_invariants(&self) {
        debug_assert!(self.definition_pos.get() <= self.last_usage.get());
        if let Some(ref usages) = *self.real_usages.borrow() {
            let mut sorted = usages.clone();
            sorted.sort();
            debug_assert_eq!(usages, &sorted);
            debug_assert!(self.last_usage.get() >= *usages.iter().max().unwrap());
            debug_assert!(self.definition_pos.get() < *usages.iter().min().unwrap());
        }
    }
}

impl fmt::Debug for Lifetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{:?}({})",
            self.definition_pos.get(),
            *self.real_usages.borrow(),
            self.last_usage.get()
        )?;
        if let Some(ref fp) = *self.fixed_positions.borrow() {
            for &(index, reg) in fp {
                write!(f, " @{} in {}", index, reg)?;
            }
        }
        if self.current_register_index.get() >= 0 {
            write!(f, " reg_idx:{}", self.current_register_index.get())?;
        }
        if let Some(pos) = self.current_frame_loc.get() {
            write!(f, " frame:{}", pos)?;
        }
        if self.hint_frame_pos.get() >= 0 {
            write!(f, " hint:{}", self.hint_frame_pos.get())?;
        }
        Ok(())
    }
}
