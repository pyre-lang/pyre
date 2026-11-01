/// llsupport/regalloc.py: Register allocation.
///
/// Classes:
///   Lifetime           — regalloc.py
///   FixedRegisterPositions — regalloc.py
///   LifetimeManager    — regalloc.py
///   FrameManager       — regalloc.py
///   RegisterManager    — regalloc.py
///   RegAlloc           — regalloc.py BaseRegalloc
///
/// Functions:
///   compute_vars_longevity — regalloc.py
///   valid_addressing_size  — regalloc.py
///   get_scale              — regalloc.py
///
/// The per-arch allocators live next to their assemblers:
/// `x86/regalloc.rs` holds x86/regalloc.py `RegAlloc`'s `consider_*` family
/// and `aarch64/regalloc.rs` holds aarch64/regalloc.py `Regalloc`'s
/// `prepare_op_*` family, each with its own opcode dispatch.
use indexmap::IndexMap;
use majit_backend::BackendError;
use majit_ir::IndexMapExt;
use rustc_hash::FxBuildHasher;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::arch::*;
use crate::gcmap::{allocate_gcmap, gcmap_set_bit};
use crate::jump::RegallocMoves;
use crate::regloc::*;
#[cfg(debug_assertions)]
use majit_ir::forwarding::{Forwarded, ForwardingHost};
use majit_ir::operand::Operand;
use majit_ir::{InputArgRc, Op, OpCode, OpRc, OpRef, OpTypeIndex, TempVar, Type};
use smallvec::SmallVec;

// ── Constants ──────────────────────────────────────────────────────

/// regalloc.py:909
pub const UNDEF_POS: i32 = -42;

/// llsupport/regalloc.py `SAVE_DEFAULT_REGS` / `SAVE_GCREF_REGS` /
/// `SAVE_ALL_REGS`.
pub const SAVE_DEFAULT_REGS: u8 = 0;
pub const SAVE_GCREF_REGS: u8 = 2;
pub const SAVE_ALL_REGS: u8 = 1;

// ── Lifetime ───────────────────────────────────────────────────────

/// regalloc.py `Lifetime` — liveness information for a single variable.
/// Interior mutability so a shared `Rc<Lifetime>` (`Lifetime.share_with`)
/// can be updated without a side-table lookup.
pub struct Lifetime {
    /// `Lifetime.definition_pos` — operations-slice index where the variable is defined
    pub definition_pos: Cell<i32>,
    /// `Lifetime.last_usage` — operations-slice index of last use (including failargs/jump)
    pub last_usage: Cell<i32>,
    /// `Lifetime.real_usages` — *real* usages (as op argument, excluding
    /// jump/failargs). Four inline slots cover the common SSA use-count
    /// so the first `push` does not heap-grow 16 B per live box.
    pub real_usages: RefCell<Option<SmallVec<[i32; 4]>>>,
    /// `Lifetime.fixed_positions` — positions requiring specific registers
    pub fixed_positions: RefCell<Option<Vec<(i32, RegLoc)>>>,
    /// `Lifetime.share_with` — another Lifetime that wants to share a register
    pub share_with: RefCell<Option<Rc<Lifetime>>>,
    /// `Lifetime._definition_pos_shared`
    _definition_pos_shared: Cell<i32>,
    /// `Lifetime.current_register_index` into RegisterManager.all_regs (-1 = not in reg)
    pub current_register_index: Cell<i32>,
    /// `Lifetime.current_frame_loc` — frame location where the box currently lives
    pub current_frame_loc: Cell<Option<FrameLoc>>,
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

    /// `Lifetime.fixed_register`
    pub fn fixed_register(&self, position: i32, reg: RegLoc) -> i32 {
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
    /// object, with no map lookup.
    pub fn find_fixed_register(&self, opindex: i32) -> Option<RegLoc> {
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

impl std::fmt::Debug for Lifetime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{:?}({})",
            self.definition_pos.get(),
            *self.real_usages.borrow(),
            self.last_usage.get()
        )?;
        if let Some(ref fp) = *self.fixed_positions.borrow() {
            for &(index, reg) in fp {
                write!(f, " @{} in {:?}", index, reg)?;
            }
        }
        if self.current_register_index.get() >= 0 {
            write!(f, " reg_idx:{}", self.current_register_index.get())?;
        }
        if self.current_frame_loc.get().is_some() {
            write!(f, " frame:{:?}", self.current_frame_loc.get())?;
        }
        if self.hint_frame_pos.get() >= 0 {
            write!(f, " hint:{}", self.hint_frame_pos.get())?;
        }
        Ok(())
    }
}

// ── FixedRegisterPositions ─────────────────────────────────────────

/// regalloc.py:1025
pub struct FixedRegisterPositions {
    pub register: RegLoc,
    pub index_lifetimes: Vec<(i32, i32)>, // (opindex, definition_pos)
}

impl FixedRegisterPositions {
    pub fn new(register: RegLoc) -> Self {
        FixedRegisterPositions {
            register,
            index_lifetimes: Vec::new(),
        }
    }

    /// regalloc.py:1031
    pub fn fixed_register(&mut self, opindex: i32, definition_pos: i32) {
        if let Some(last) = self.index_lifetimes.last() {
            debug_assert!(opindex > last.0);
        }
        self.index_lifetimes.push((opindex, definition_pos));
    }

    /// regalloc.py:1036
    pub fn free_until_pos(&self, opindex: i32) -> i32 {
        for &(index, definition_pos) in &self.index_lifetimes {
            if opindex <= index {
                if definition_pos >= opindex {
                    return definition_pos;
                } else {
                    return index;
                }
            }
        }
        i32::MAX
    }
}

// ── LifetimeManager ────────────────────────────────────────────────

/// `regalloc.py get_lifetime(box)` — `box.get_forwarded()` as a `Lifetime`.
#[inline]
pub fn get_lifetime(box_: &Operand) -> Option<Rc<Lifetime>> {
    let info = box_.get_lifetime_info()?;
    Rc::downcast(info).ok()
}

/// Walk every box that may host a Lifetime and visit those with a current
/// frame location. Assemblers use this after `_walk_operations` to sync
/// leftover `current_frame_loc` into `opref_to_slot`.
pub fn visit_boxes_with_frame_loc(
    inputargs: &[InputArgRc],
    operations: &[OpRc],
    fm: &FrameManager,
    mut visit: impl FnMut(OpRef, FrameLoc),
) {
    let mut visit_box = |box_: &Operand| {
        if let Some(lt) = get_lifetime(box_)
            && let Some(floc) = lt.current_frame_loc.get()
        {
            visit(box_.to_opref(), floc);
        }
    };
    for iarg in inputargs {
        visit_box(&Operand::from_bound_inputarg(iarg));
    }
    for op_rc in operations {
        visit_box(&Operand::from_bound_op(op_rc));
    }
    for slot in fm.boxes_in_frame.iter().flatten() {
        visit_box(slot);
    }
}

/// regalloc.py `LifetimeManager` — owns `fixed_register_use` only.
/// Per-box Lifetime lives on the box (`LifetimeManager.__setitem__` →
/// `box.set_forwarded(lt)`; `__getitem__` → `get_lifetime`).
pub struct LifetimeManager {
    /// `LifetimeManager.fixed_register_use` — register → FixedRegisterPositions
    pub fixed_register_use: IndexMap<RegLoc, FixedRegisterPositions, FxBuildHasher>,
}

impl Default for LifetimeManager {
    fn default() -> Self {
        Self::new()
    }
}

impl LifetimeManager {
    pub fn new() -> Self {
        LifetimeManager {
            fixed_register_use: IndexMap::with_hasher(FxBuildHasher),
        }
    }

    pub fn contains(&self, v: &Operand) -> bool {
        get_lifetime(v).is_some()
    }

    pub fn get(&self, v: &Operand) -> Option<Rc<Lifetime>> {
        get_lifetime(v)
    }

    pub fn set(&mut self, v: &Operand, lifetime: Lifetime) -> Rc<Lifetime> {
        debug_assert!(
            get_lifetime(v).is_none() || v.is_temp_var(),
            "LifetimeManager.__setitem__: box already has a Lifetime"
        );
        let rc = Rc::new(lifetime);
        v.set_lifetime_info(rc.clone());
        rc
    }

    /// `LifetimeManager.fixed_register`
    pub fn fixed_register(&mut self, opindex: i32, register: RegLoc, var: Option<&Operand>) {
        let definition_pos = match var {
            None => opindex,
            Some(v) => {
                let lt = get_lifetime(v).unwrap();
                lt.fixed_register(opindex, register)
            }
        };
        self.fixed_register_use
            .entry_or_insert_with(register, || FixedRegisterPositions::new(register))
            .fixed_register(opindex, definition_pos);
    }

    /// `LifetimeManager.try_use_same_register`
    pub fn try_use_same_register(&mut self, v0: &Operand, v1: &Operand) {
        let lt0 = get_lifetime(v0).unwrap();
        let lt1 = get_lifetime(v1).unwrap();
        let def0 = lt0.definition_pos.get();
        let last0 = lt0.last_usage.get();
        let def1 = lt1.definition_pos.get();
        debug_assert!(def0 < def1);
        if last0 != def1 {
            return; // not supported for now
        }
        let def0_shared = lt0.definition_pos_shared();
        *lt0.share_with.borrow_mut() = Some(Rc::clone(&lt1));
        lt1._definition_pos_shared.set(def0_shared);
    }

    /// regalloc.py:1091
    pub fn longest_free_reg(&self, position: i32, free_regs: &[RegLoc]) -> (Option<RegLoc>, i32) {
        let mut max_free_pos = position;
        let mut best_reg = None;
        // reverse for compatibility with old code
        for i in (0..free_regs.len()).rev() {
            let reg = free_regs[i];
            match self.fixed_register_use.get(&reg) {
                None => return (Some(reg), i32::MAX),
                Some(fixed_reg_pos) => {
                    let free_until_pos = fixed_reg_pos.free_until_pos(position);
                    if free_until_pos > max_free_pos {
                        best_reg = Some(reg);
                        max_free_pos = free_until_pos;
                    }
                }
            }
        }
        (best_reg, max_free_pos)
    }

    /// regalloc.py:1111
    pub fn free_reg_whole_lifetime(
        &self,
        position: i32,
        v: &Operand,
        free_regs: &[RegLoc],
    ) -> Option<RegLoc> {
        let longevity_var = get_lifetime(v).unwrap();
        let last_usage = longevity_var.last_usage_including_sharing();
        let mut min_fixed_use_after = i32::MAX;
        let mut best_reg = None;
        let mut unfixed_reg = None;
        for &reg in free_regs {
            match self.fixed_register_use.get(&reg) {
                None => {
                    unfixed_reg = Some(reg);
                    continue;
                }
                Some(fixed_reg_pos) => {
                    let use_after = fixed_reg_pos.free_until_pos(position);
                    if use_after <= last_usage {
                        continue; // can't fit
                    }
                    if use_after < min_fixed_use_after {
                        best_reg = Some(reg);
                        min_fixed_use_after = use_after;
                    }
                }
            }
        }
        if best_reg.is_some() {
            return best_reg;
        }
        unfixed_reg
    }

    /// regalloc.py:1138
    pub fn try_pick_free_reg(
        &self,
        position: i32,
        v: &Operand,
        free_regs: &[RegLoc],
    ) -> Option<RegLoc> {
        if free_regs.is_empty() {
            return None;
        }
        let longevity_var = get_lifetime(v).unwrap();
        // check whether there is a fixed register and whether it's free
        let fixed_reg = longevity_var.find_fixed_register(position);
        if let Some(reg) = fixed_reg
            && free_regs.contains(&reg)
        {
            return Some(reg);
        }
        // try to find a register that's free for the whole lifetime of v
        let loc = self.free_reg_whole_lifetime(position, v, free_regs);
        if loc.is_some() {
            return loc;
        }
        // can't fit v completely, pick the register free the longest
        let (loc, _free_until) = self.longest_free_reg(position, free_regs);
        loc
    }
}

// ── compute_vars_longevity ─────────────────────────────────────────

/// regalloc.py compute_vars_longevity — backward liveness analysis.
pub fn compute_vars_longevity(inputargs: &[InputArgRc], operations: &[OpRc]) -> LifetimeManager {
    // x86/regalloc.py `_prepare` asserts `arg.get_forwarded() is None` for
    // inputargs; `LifetimeManager.__setitem__` asserts the slot is empty.
    #[cfg(debug_assertions)]
    {
        for iarg in inputargs {
            debug_assert!(
                matches!(iarg.as_ref().get_forwarded(), Forwarded::None),
                "x86/regalloc.py _prepare: inputarg.get_forwarded() is None"
            );
        }
        for op_rc in operations {
            debug_assert!(
                matches!(op_rc.as_ref().get_forwarded(), Forwarded::None),
                "LifetimeManager.__setitem__: op forwarded is None"
            );
        }
    }
    let mut longevity = LifetimeManager::new();

    // regalloc.py:1179 iterate operations in REVERSE
    for i in (0..operations.len()).rev() {
        let op_rc = &operations[i];
        let op = op_rc.as_ref();
        let opnum = op.opcode;
        let result = Operand::from_bound_op(op_rc);
        let i = i as i32;

        match longevity.get(&result) {
            Some(lt) => lt.definition_pos.set(i),
            None => {
                // `compute_vars_longevity`: if op.type != 'v' and has_no_side_effect:
                // result not used, operation has no side-effect → dead code
                if opnum.result_type() != Type::Void && opnum.has_no_side_effect() {
                    continue;
                }
                longevity.set(&result, Lifetime::new(i, i));
            }
        }

        // `compute_vars_longevity` process arguments
        for arg in op.args_slice() {
            if arg.is_constant() {
                continue;
            }
            let lifetime = match longevity.get(arg) {
                Some(lt) => lt,
                None => longevity.set(arg, Lifetime::new(UNDEF_POS, i)),
            };
            // real_usages: exclude JUMP and LABEL
            if opnum != OpCode::Jump && opnum != OpCode::Label {
                let mut usages = lifetime.real_usages.borrow_mut();
                if usages.is_none() {
                    *usages = Some(SmallVec::new());
                }
                usages.as_mut().unwrap().push(i);
            }
        }

        // regalloc.py:1202-1208 guard failargs
        if opnum.is_guard()
            && let Some(fail_args) = op.guard_fail_args()
        {
            for arg in fail_args.iter() {
                if arg.is_none() {
                    continue; // hole
                }
                debug_assert!(!arg.is_constant());
                if longevity.get(arg).is_none() {
                    longevity.set(arg, Lifetime::new(UNDEF_POS, i));
                }
            }
        }
    }

    // regalloc.py:1210-1213 input arguments
    for iarg in inputargs {
        let box_ = Operand::from_bound_inputarg(iarg);
        if longevity.get(&box_).is_none() {
            longevity.set(&box_, Lifetime::new(-1, -1));
        }
    }

    // `compute_vars_longevity` reverse real_usages and check invariants
    let reverse_usages = |box_: &Operand| {
        if let Some(lifetime) = get_lifetime(box_) {
            if let Some(ref mut usages) = *lifetime.real_usages.borrow_mut() {
                usages.reverse();
            }
            #[cfg(debug_assertions)]
            lifetime.check_invariants();
        }
    };
    for iarg in inputargs {
        reverse_usages(&Operand::from_bound_inputarg(iarg));
    }
    for op_rc in operations {
        reverse_usages(&Operand::from_bound_op(op_rc));
    }

    longevity
}

// ── FrameManager ───────────────────────────────────────────────────

/// regalloc.py FrameManager — frame slot allocation.
/// Combined with x86/regalloc.py X86FrameManager.
pub struct FrameManager {
    /// regalloc.py:69
    pub current_frame_depth: usize,
    /// regalloc.py:70
    pub boxes_in_frame: Vec<Option<Operand>>,
    /// x86/regalloc.py:134 base_ofs (byte offset of frame field 0 from ebp)
    pub base_ofs: i32,
}

impl FrameManager {
    /// regalloc.py:68 + x86/regalloc.py __init__  allow-line-citation
    pub fn new(base_ofs: i32) -> Self {
        FrameManager {
            current_frame_depth: 0,
            boxes_in_frame: Vec::new(),
            base_ofs,
        }
    }

    /// regalloc.py:81
    pub fn get_frame_depth(&self) -> usize {
        self.current_frame_depth
    }

    /// regalloc.py:84
    fn _increase_frame_depth(&mut self, incby: usize) {
        self.current_frame_depth += incby;
        for _ in 0..incby {
            self.boxes_in_frame.push(None);
        }
    }

    /// regalloc.py:89
    pub fn get(&self, v: &Operand, longevity: &LifetimeManager) -> Option<FrameLoc> {
        longevity.get(v).and_then(|lt| lt.current_frame_loc.get())
    }

    /// regalloc.py:95
    pub fn loc(
        &mut self,
        v: &Operand,
        tp: Type,
        must_exist: bool,
        longevity: &mut LifetimeManager,
    ) -> FrameLoc {
        if let Some(loc) = self.get(v, longevity) {
            return loc;
        }
        if must_exist {
            panic!("FrameManager.loc: box {:?} not found", v);
        }
        self.get_new_loc(v, tp, longevity)
    }

    /// regalloc.py:105
    pub fn get_new_loc(
        &mut self,
        v: &Operand,
        tp: Type,
        longevity: &mut LifetimeManager,
    ) -> FrameLoc {
        let size = Self::frame_size(tp);
        let hint = self.get_frame_pos_hint(v, longevity);
        let newloc = self._find_frame_location(size, tp, hint);
        let newloc = match newloc {
            Some(loc) => loc,
            None => {
                // regalloc.py:114-126
                let mut index = self.get_frame_depth();
                if size == 2 && index & 1 == 1 {
                    if index > 0 && self.boxes_in_frame[index - 1].is_none() {
                        index -= 1;
                    } else {
                        index += 1;
                    }
                    let loc = Self::frame_pos(index, tp, self.base_ofs);
                    let needed = index + size;
                    if needed > self.current_frame_depth {
                        self._increase_frame_depth(needed - self.current_frame_depth);
                    }
                    loc
                } else {
                    let loc = Self::frame_pos(index, tp, self.base_ofs);
                    self._increase_frame_depth(size);
                    loc
                }
            }
        };
        self.bind(v, newloc, tp, longevity);
        newloc
    }

    /// regalloc.py:142
    fn _find_frame_location(&self, size: usize, tp: Type, hint: i32) -> Option<FrameLoc> {
        debug_assert!(size == 1 || size == 2);
        if size == 1 {
            // regalloc.py:145 — check 0 <= hint < depth
            if 0 <= hint
                && (hint as usize) < self.current_frame_depth
                && self.boxes_in_frame[hint as usize].is_none()
            {
                return Some(Self::frame_pos(hint as usize, tp, self.base_ofs));
            }
            for i in 0..self.current_frame_depth {
                if self.boxes_in_frame[i].is_none() {
                    return Some(Self::frame_pos(i, tp, self.base_ofs));
                }
            }
            None
        } else {
            let limit = (self.current_frame_depth >> 1) << 1;
            let mut i = 0;
            while i < limit {
                if self.boxes_in_frame[i].is_none() && self.boxes_in_frame[i + 1].is_none() {
                    return Some(Self::frame_pos(i, tp, self.base_ofs));
                }
                i += 2;
            }
            None
        }
    }

    /// regalloc.py:164
    pub fn bind(&mut self, v: &Operand, loc: FrameLoc, tp: Type, longevity: &mut LifetimeManager) {
        let pos = loc.get_position();
        let size = Self::frame_size(tp);
        if pos + size > self.current_frame_depth {
            self._increase_frame_depth(pos + size - self.current_frame_depth);
        }
        debug_assert!(self.boxes_in_frame[pos].is_none());
        for index in pos..pos + size {
            self.boxes_in_frame[index] = Some(v.clone());
        }
        let lifetime = longevity.get(v).expect("bind: variable not in longevity");
        lifetime.current_frame_loc.set(Some(loc));
    }

    /// regalloc.py:181
    pub fn finish_binding(&self) {
        #[cfg(debug_assertions)]
        self._check_invariants();
    }

    /// regalloc.py:185
    pub fn mark_as_free(&mut self, v: &Operand, tp: Type, longevity: &mut LifetimeManager) {
        let loc = match self.get(v, longevity) {
            Some(loc) => loc,
            None => return, // not in frame
        };
        let lifetime = longevity.get(v).unwrap();
        debug_assert!(lifetime.current_frame_loc.get().is_some());
        lifetime.current_frame_loc.set(None);
        let pos = loc.get_position();
        let size = Self::frame_size(tp);
        debug_assert!(self.boxes_in_frame[pos].as_ref() == Some(v));
        for index in pos..pos + size {
            self.boxes_in_frame[index] = None;
        }
    }

    /// regalloc.py:201
    pub fn add_frame_pos_hint(&self, v: &Operand, loc: FrameLoc, longevity: &mut LifetimeManager) {
        if let Some(lifetime) = longevity.get(v) {
            lifetime.hint_frame_pos.set(loc.get_position() as i32);
        }
    }

    /// `FrameManager.get_frame_pos_hint`
    pub fn get_frame_pos_hint(&self, v: &Operand, longevity: &LifetimeManager) -> i32 {
        longevity
            .get(v)
            .map(|lt| lt.hint_frame_pos.get())
            .unwrap_or(-1)
    }

    /// regalloc.py:212
    fn _check_invariants(&self) {
        debug_assert_eq!(self.boxes_in_frame.len(), self.current_frame_depth);
    }

    // ── x86/regalloc.py X86FrameManager methods ──

    /// x86/regalloc.py frame_pos(i, box_type)
    pub fn frame_pos(position: usize, tp: Type, base_ofs: i32) -> FrameLoc {
        let ebp_offset = get_ebp_ofs(base_ofs, position);
        FrameLoc::new(position, ebp_offset, tp == Type::Float)
    }

    /// x86/regalloc.py frame_size(box_type)
    pub fn frame_size(tp: Type) -> usize {
        // x86_64: always 1 word (IS_X86_32 && FLOAT → 2, but we're 64-bit only)
        let _ = tp;
        1
    }

    /// x86/regalloc.py get_loc_index(loc)
    pub fn get_loc_index(loc: &FrameLoc) -> usize {
        loc.get_position()
    }
}

/// x86/regalloc.py get_ebp_ofs(base_ofs, position)
///
/// RPython: rbp points past the frame header, slots grow downward:
///   -(position+1)*WORD + base_ofs
///
/// aarch64/locations.py get_fp_offset / arm/locations.py get_fp_offset
/// get_fp_offset(base_ofs, position) = base_ofs + WORD * (position + JITFRAME_FIXED_SIZE)
///
/// Here `JITFRAME_FIXED_SIZE` is the arch-specific managed-register area
/// reserved at the start of `jf_frame`, not the object-header byte size.
/// llsupport/regalloc.py `not_implemented`.  `compile_loop` catches the
/// panic and declines the trace, as `NotImplementedError` does upstream.
pub fn not_implemented(msg: &str) -> ! {
    panic!("[llsupport/regalloc] {msg}")
}

pub fn get_ebp_ofs(base_ofs: i32, position: usize) -> i32 {
    base_ofs + WORD as i32 * (position as i32 + JITFRAME_FIXED_SIZE as i32)
}

// arch-specific register configuration is delegated to the per-arch
// regalloc.rs (mirrors RPython's `rpython/jit/backend/x86/regalloc.py`
// vs `rpython/jit/backend/aarch64/regalloc.py` split).  The shared
// file here corresponds to `rpython/jit/backend/llsupport/regalloc.py`
// and only consumes the configuration via these re-exports.
#[cfg(target_arch = "aarch64")]
use crate::aarch64::regalloc as arch_regalloc;
#[cfg(target_arch = "x86_64")]
use crate::x86::regalloc as arch_regalloc;

use arch_regalloc::{
    MALLOC_NURSERY_CLOBBER, all_core_regs, all_float_regs, call_result_fpr, call_result_gpr,
    core_reg_index, frame_reg, save_around_call_core_regs,
};

// ── RegisterManager ────────────────────────────────────────────────

/// regalloc.py RegisterManager — register allocation logic.
///
/// RPython stores `longevity` and `frame_manager` as attributes;
/// in Rust we pass them explicitly to methods to avoid borrow conflicts.
pub struct RegisterManager {
    // ── Class-level configuration (set at construction) ──
    /// regalloc.py:362
    pub all_regs: Vec<RegLoc>,
    /// regalloc.py:363
    pub no_lower_byte_regs: Vec<RegLoc>,
    /// regalloc.py:364
    pub save_around_call_regs: Vec<RegLoc>,
    /// regalloc.py:365
    pub frame_reg: RegLoc,
    /// Type filter: INT/REF for GPR, FLOAT for XMM
    pub box_types: Option<Vec<Type>>,
    /// x86/regalloc.py call_result_location — hardcoded return register.
    pub call_result_reg: RegLoc,

    // ── Instance state ──
    /// regalloc.py:369
    pub free_regs: Vec<RegLoc>,
    /// `RegisterManager.reg_bindings_list` — register index → box (None = free)
    pub reg_bindings_list: Vec<Option<Operand>>,
    /// regalloc.py:373
    pub temp_boxes: Vec<Operand>,
    /// regalloc.py:375
    pub box_currently_in_frame_reg: Option<Operand>,
    /// regalloc.py:376
    pub position: i32,
    /// Spill moves generated by _sync_var_to_stack.
    /// RPython calls self.assembler.regalloc_mov() directly; the register
    /// manager has no assembler, so moves queue here until
    /// `RegAlloc::flush_moves` hands them over before the next op.
    pub spill_moves: Vec<(Loc, Loc)>,
    /// `X86XMMRegisterManager` / `VFPRegisterManager`
    /// `assembler.datablockwrapper`. `None` on the GPR manager, and on the
    /// XMM / VFP manager outside `RegAlloc::walk_operations_with_datablockwrapper`,
    /// which installs the borrowed wrapper for the walk and detaches it
    /// before returning.
    datablockwrapper: Option<DataBlockWrapperPtr>,
    /// First `MachineDataBlockWrapper.malloc_aligned` failure.
    /// `open_malloc` raises `MemoryError`; `walk_operations` returns it as
    /// `BackendError::CompilationFailed` so `consider_*` stays infallible.
    pool_error: Option<std::io::Error>,
}

/// `assembler.datablockwrapper` as the XMM / VFP manager reaches it during a
/// walk. The wasm32 guest has no data block, so the pointer never exists.
#[cfg(not(target_arch = "wasm32"))]
type DataBlockWrapperPtr = std::ptr::NonNull<majit_backend::MachineDataBlockWrapper>;
#[cfg(target_arch = "wasm32")]
type DataBlockWrapperPtr = std::convert::Infallible;

/// Resolve a constant `OpRef` to its `i64` bit pattern.
///
/// Every caller is an operand slot that RPython reads with `.getint()` —
/// a GC_LOAD offset (aarch64/regalloc.py:535), a GC_LOAD_INDEXED scale
/// (:566), an itemsize (:568). Upstream can spell those as bare `getint()`
/// calls because the rewrite pass guarantees the slot holds a `ConstInt`
/// before the backend ever sees the op (rewrite.py:199-205 builds every one
/// of them out of `ConstInt(...)`).
///
/// Inline-Const variants (history.py/268/314 `ConstXxx.value`) carry the
/// value directly. The legacy pool-indexed variants must be present in the
/// optimizer's `constants` snapshot. Either way the backend must never
/// silently substitute `0`, which would miscompile the constant as zero, so
/// this panics at the parity hole instead — matching the Cranelift backend's
/// `missing_legacy_constant`.
///
/// The two failures it separates are different bugs and want different fixes:
/// a *constant-namespace* raw that is missing means the producer never seeded
/// the pool, while a *body* OpRef arriving at all means the slot was filled
/// with a runtime value — an op that should have been lowered upstream of the
/// backend and was not. `#[track_caller]` reports which slot, because the
/// `consider_*` body is the only thing that identifies the op.
#[track_caller]
fn const_bits_or_panic(v: &Operand, constants: &indexmap::IndexMap<u32, i64>, where_: &str) -> i64 {
    let v = v.to_opref();
    if let Some(bits) = v.inline_const_bits() {
        return bits;
    }
    let caller = std::panic::Location::caller();
    let raw = v.raw();
    constants.get(&raw).copied().unwrap_or_else(|| {
        let diagnosis = if OpRef::raw_is_constant(raw) {
            "legacy pool-indexed const absent from the constants snapshot — the \
             optimizer producer must seed it (or mint an inline Const) instead of \
             the backend baking 0; RPython ConstInt.value (history.py:227) is \
             always present"
        } else {
            "a BODY OpRef, not a constant at all — this operand slot is one \
             RPython reads with `.getint()`, so a runtime value reaching it means \
             the op was never lowered. Upstream guarantees the slot by rewriting \
             before assembly (aarch64/regalloc.py:188 -> gc.py:109-112), e.g. \
             RAW_LOAD_I becomes GC_LOAD_INDEXED_I with ConstInt scale/offset/size \
             (rewrite.py:228-232)"
        };
        panic!("dynasm {where_} at {caller}: {v:?} (raw={raw}) is {diagnosis}.")
    })
}

/// `X86XMMRegisterManager.convert_to_imm` / `VFPRegisterManager.convert_to_imm`:
/// `malloc_aligned(8, 8)`, write `getfloatstorage()`, return `ConstFloatLoc(adr)`.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn park_float_const(
    wrapper: &mut majit_backend::MachineDataBlockWrapper,
    bits: u64,
) -> Result<ConstFloatLoc, std::io::Error> {
    let adr = wrapper.malloc_aligned(8, 8)?;
    // `compile_loop` / `compile_bridge` hold `AssemblerWriting::enter`
    // across the walk, so this store is inside the write window.
    unsafe { (adr as *mut u64).write(bits) };
    Ok(ConstFloatLoc { value: adr })
}

impl RegisterManager {
    /// regalloc.py:368
    pub fn new(
        all_regs: Vec<RegLoc>,
        no_lower_byte_regs: Vec<RegLoc>,
        save_around_call_regs: Vec<RegLoc>,
        frame_reg: RegLoc,
        box_types: Option<Vec<Type>>,
        call_result_reg: RegLoc,
    ) -> Self {
        let n = all_regs.len();
        let mut free_regs = all_regs.clone();
        free_regs.reverse(); // regalloc.py:370
        RegisterManager {
            all_regs,
            no_lower_byte_regs,
            save_around_call_regs,
            frame_reg,
            box_types,
            call_result_reg,
            free_regs,
            reg_bindings_list: vec![None; n],
            temp_boxes: Vec::new(),
            box_currently_in_frame_reg: None,
            position: -1,
            spill_moves: Vec::new(),
            datablockwrapper: None,
            pool_error: None,
        }
    }

    // ── RegBindingsDict equivalent ──

    fn _register_index(&self, reg: RegLoc) -> usize {
        self.all_regs
            .iter()
            .position(|r| *r == reg)
            .expect("register not in all_regs")
    }

    /// regalloc.py / RegBindingsDict.get
    pub fn reg_bindings_get(&self, v: &Operand, longevity: &LifetimeManager) -> Option<RegLoc> {
        if let Some(lifetime) = longevity.get(v) {
            let index = lifetime.current_register_index.get();
            if index >= 0 {
                return Some(self.all_regs[index as usize]);
            }
        }
        None
    }

    /// RegBindingsDict.__contains__
    pub fn reg_bindings_contains(&self, v: &Operand, longevity: &LifetimeManager) -> bool {
        self.reg_bindings_get(v, longevity).is_some()
    }

    /// RegBindingsDict.__setitem__
    ///
    /// Overwrites the reverse slot (`reg_bindings_list[new]`) but leaves any
    /// previous occupant's `current_register_index` alone. `loc()` reads that
    /// index, so two liveboxes recovered from the same physical register
    /// (duplicate `rd_locs` entries) stay findable. Clearing the old occupant
    /// here made `RegisterManager.loc` panic on the first box.
    pub fn reg_bindings_set(&mut self, v: &Operand, reg: RegLoc, longevity: &mut LifetimeManager) {
        let new_index = self._register_index(reg);
        // Clear this variable's previous reverse slot first.
        let old_index = longevity
            .get(v)
            .map(|lt| lt.current_register_index.get())
            .unwrap_or(-1);
        if old_index >= 0 {
            self.reg_bindings_list[old_index as usize] = None;
        }
        let lifetime = longevity
            .get(v)
            .expect("reg_bindings_set: not in longevity");
        lifetime.current_register_index.set(new_index as i32);
        self.reg_bindings_list[new_index] = Some(v.clone());
    }

    /// `RegBindingsDict.__delitem__`
    pub fn reg_bindings_del(&mut self, v: &Operand, longevity: &mut LifetimeManager) {
        let lifetime = longevity
            .get(v)
            .expect("reg_bindings_del: not in longevity");
        let index = lifetime.current_register_index.get();
        assert!(
            index >= 0,
            "reg_bindings_del: {:?} not in register (index={})",
            v,
            index
        );
        self.reg_bindings_list[index as usize] = None;
        lifetime.current_register_index.set(-1);
    }

    /// `RegBindingsDict.pop`
    pub fn reg_bindings_pop(&mut self, v: &Operand, longevity: &mut LifetimeManager) -> RegLoc {
        let lifetime = longevity
            .get(v)
            .expect("reg_bindings_pop: not in longevity");
        let index = lifetime.current_register_index.get();
        assert!(
            index >= 0,
            "reg_bindings_pop: {:?} not in register (index={})",
            v,
            index
        );
        let reg = self.all_regs[index as usize];
        self.reg_bindings_list[index as usize] = None;
        lifetime.current_register_index.set(-1);
        reg
    }

    /// Iterate (OpRef, RegLoc) bindings — RegBindingsIterItems equivalent.
    pub fn reg_bindings_iter(&self) -> impl Iterator<Item = (Operand, RegLoc)> + '_ {
        self.reg_bindings_list
            .iter()
            .enumerate()
            .filter_map(|(i, opt)| opt.as_ref().map(|v| (v.clone(), self.all_regs[i])))
    }

    /// Collect current register bindings as Vec.
    pub fn reg_bindings_items(&self) -> Vec<(Operand, RegLoc)> {
        self.reg_bindings_iter().collect()
    }

    /// Count of active bindings.
    pub fn reg_bindings_len(&self) -> usize {
        self.reg_bindings_list
            .iter()
            .filter(|x| x.is_some())
            .count()
    }

    // ── Core methods ──

    /// regalloc.py:380
    pub fn is_still_alive(&self, v: &Operand, longevity: &LifetimeManager) -> bool {
        longevity
            .get(v)
            .map(|lt| lt.last_usage.get() >= self.position)
            .unwrap_or(false)
    }

    /// `RegisterManager.stays_alive`
    pub fn stays_alive(&self, v: &Operand, longevity: &LifetimeManager) -> bool {
        longevity
            .get(v)
            .map(|lt| lt.last_usage.get() > self.position)
            .unwrap_or(false)
    }

    /// regalloc.py:390
    pub fn next_instruction(&mut self, incr: i32) {
        self.position += incr;
    }

    /// regalloc.py:409
    pub fn possibly_free_var(
        &mut self,
        v: &Operand,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
        v_type: Type,
    ) {
        if v.is_constant() {
            return;
        }
        let should_free =
            !longevity.contains(v) || longevity.get(v).unwrap().last_usage.get() <= self.position;
        if should_free {
            if let Some(reg) = self.reg_bindings_get(v, longevity) {
                self.free_regs.push(reg);
                self.reg_bindings_del(v, longevity);
            }
            if self.box_currently_in_frame_reg.as_ref() == Some(v) {
                self.box_currently_in_frame_reg = None;
            }
            fm.mark_as_free(v, v_type, longevity);
        }
    }

    /// regalloc.py:427
    pub fn possibly_free_vars(
        &mut self,
        vars: &[Operand],
        types: &[Type],
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
    ) {
        for (v, tp) in vars.iter().zip(types.iter()) {
            self.possibly_free_var(v, longevity, fm, *tp);
        }
    }

    /// regalloc.py:437
    pub fn free_temp_vars(&mut self, longevity: &mut LifetimeManager, fm: &mut FrameManager) {
        let temps: Vec<Operand> = std::mem::take(&mut self.temp_boxes);
        for v in temps {
            // Temp vars are always INT type
            self.possibly_free_var(&v, longevity, fm, Type::Int);
        }
    }

    /// regalloc.py:460
    pub fn try_allocate_reg(
        &mut self,
        v: &Operand,
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
        longevity: &mut LifetimeManager,
    ) -> Option<RegLoc> {
        debug_assert!(!v.is_constant());
        if v.is_temp_var() {
            longevity.set(v, Lifetime::new(self.position, self.position));
        }

        if let Some(selected) = selected_reg {
            // regalloc.py:474-486
            let res = self.reg_bindings_get(v, longevity);
            if let Some(existing) = res {
                if existing == selected {
                    return Some(existing);
                } else {
                    self.reg_bindings_del(v, longevity);
                    self.free_regs.push(existing);
                }
            }
            if self.free_regs.contains(&selected) {
                self.free_regs.retain(|r| *r != selected);
                self.reg_bindings_set(v, selected, longevity);
                return Some(selected);
            }
            return None;
        }

        if need_lower_byte {
            // regalloc.py:487-501
            let loc = self.reg_bindings_get(v, longevity);
            if let Some(reg) = loc
                && !self.no_lower_byte_regs.contains(&reg)
            {
                return Some(reg);
            }
            let free_regs: Vec<RegLoc> = self
                .free_regs
                .iter()
                .filter(|r| !self.no_lower_byte_regs.contains(r))
                .copied()
                .collect();
            let newloc = longevity.try_pick_free_reg(self.position, v, &free_regs);
            newloc?;
            let newloc = newloc.unwrap();
            self.free_regs.retain(|r| *r != newloc);
            if let Some(old) = loc {
                self.free_regs.push(old);
            }
            self.reg_bindings_set(v, newloc, longevity);
            return Some(newloc);
        }

        // regalloc.py:502-512 — default path
        let res = self.reg_bindings_get(v, longevity);
        if res.is_some() {
            return res;
        }
        let loc = longevity.try_pick_free_reg(self.position, v, &self.free_regs);
        match loc {
            None => None,
            Some(reg) => {
                self.reg_bindings_set(v, reg, longevity);
                self.free_regs.retain(|r| *r != reg);
                Some(reg)
            }
        }
    }

    /// regalloc.py _spill_var
    pub fn _spill_var(
        &mut self,
        forbidden_vars: &[Operand],
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
    ) -> RegLoc {
        let v_to_spill = self._pick_variable_to_spill(
            forbidden_vars,
            selected_reg,
            need_lower_byte,
            None,
            longevity,
        );
        let loc = self.reg_bindings_get(&v_to_spill, longevity).unwrap();
        let tp = self._type_of(&v_to_spill);
        self._sync_var_to_stack(&v_to_spill, tp, longevity, fm);
        self.reg_bindings_del(&v_to_spill, longevity);
        loc
    }

    /// regalloc.py _pick_variable_to_spill
    pub fn _pick_variable_to_spill(
        &self,
        forbidden_vars: &[Operand],
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
        regs: Option<&[Operand]>,
        longevity: &LifetimeManager,
    ) -> Operand {
        let binding_keys: Vec<Operand>;
        let regs = match regs {
            Some(r) => r,
            None => {
                binding_keys = self
                    .reg_bindings_list
                    .iter()
                    .filter_map(|x| x.clone())
                    .collect();
                &binding_keys
            }
        };

        let position = self.position;
        let mut cur_max_use_distance: i32 = -1;
        let mut candidate: Option<Operand> = None;
        let mut cur_max_age_failargs: i32 = -1;
        let mut candidate_from_failargs: Option<Operand> = None;

        for next in regs {
            let reg = self.reg_bindings_get(next, longevity).unwrap();
            if forbidden_vars.contains(next) {
                continue;
            }
            // regalloc.py:544
            if self.temp_boxes.contains(next) {
                continue;
            }
            if let Some(sel) = selected_reg {
                if reg == sel {
                    return next.clone();
                }
                continue;
            }
            if need_lower_byte && self.no_lower_byte_regs.contains(&reg) {
                continue;
            }
            let lifetime = longevity.get(next).unwrap();
            if lifetime.is_last_real_use_before(position) {
                // regalloc.py:554-561
                let max_age = lifetime.last_usage.get();
                if cur_max_age_failargs < max_age {
                    cur_max_age_failargs = max_age;
                    candidate_from_failargs = Some(next.clone());
                }
            } else {
                // regalloc.py:563-566
                let use_distance = lifetime.next_real_usage(position) - position;
                if cur_max_use_distance < use_distance {
                    cur_max_use_distance = use_distance;
                    candidate = Some(next.clone());
                }
            }
        }
        if let Some(c) = candidate_from_failargs {
            return c;
        }
        if let Some(c) = candidate {
            return c;
        }
        panic!("NoVariableToSpill");
    }

    /// regalloc.py:573
    pub fn force_allocate_reg(
        &mut self,
        v: &Operand,
        forbidden_vars: &[Operand],
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
    ) -> RegLoc {
        if v.is_temp_var() {
            longevity.set(v, Lifetime::new(self.position, self.position));
        }
        let loc = self.try_allocate_reg(v, selected_reg, need_lower_byte, longevity);
        if let Some(reg) = loc {
            return reg;
        }
        let spilled_loc =
            self._spill_var(forbidden_vars, selected_reg, need_lower_byte, longevity, fm);
        let prev_loc = self.reg_bindings_get(v, longevity);
        if let Some(prev) = prev_loc {
            self.free_regs.push(prev);
        }
        self.reg_bindings_set(v, spilled_loc, longevity);
        spilled_loc
    }

    /// regalloc.py:597
    pub fn force_allocate_frame_reg(&mut self, v: &Operand) {
        debug_assert!(self.box_currently_in_frame_reg.is_none());
        self.box_currently_in_frame_reg = Some(v.clone());
    }

    /// regalloc.py:602
    pub fn force_spill_var(
        &mut self,
        var: &Operand,
        tp: Type,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
    ) {
        self._sync_var_to_stack(var, tp, longevity, fm);
        if let Some(reg) = self.reg_bindings_get(var, longevity) {
            self.reg_bindings_del(var, longevity);
            self.free_regs.push(reg);
        }
    }

    /// `box.type == FLOAT` query for a constant `OpRef`.
    ///
    /// Typed variants (`OpRef::ConstFloat`) carry the type tag intrinsically,
    /// matching `ConstFloat.type = FLOAT` (history.py).
    #[inline]
    fn is_float_constant(&self, v: &Operand) -> bool {
        v.type_() == Type::Float
    }

    /// regalloc.py:611
    pub fn loc(
        &mut self,
        v: &Operand,
        tp: Type,
        must_exist: bool,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
        constants: &indexmap::IndexMap<u32, i64>,
    ) -> Loc {
        if v.is_constant() {
            // `RegisterManager.loc`: a Const is always `convert_to_imm`, so
            // the GPR manager's movable-`ConstPtr` check runs here too, and
            // the XMM / VFP manager with `datablockwrapper` set yields a
            // `ConstFloatLoc`.
            return self.convert_to_imm(v, constants);
        }
        if let Some(reg) = self.reg_bindings_get(v, longevity) {
            return Loc::Reg(reg);
        }
        if self.box_currently_in_frame_reg.as_ref() == Some(v) {
            return Loc::Reg(self.frame_reg);
        }
        // regalloc.py:622 → frame_manager.loc(box, must_exist)
        if let Some(frame_loc) = fm.get(v, longevity) {
            return Loc::Frame(frame_loc);
        }
        // Fallback: check constants map. The optimizer may leave constants
        // as plain OpRefs (without CONST_BIT) when they originate from
        // short preamble or constant folding.
        let opref = v.to_opref();
        if let Some(val) = opref
            .inline_const_bits()
            .or_else(|| constants.get(&opref.raw()).copied())
        {
            if tp == Type::Float || self.is_float_constant(v) {
                return self.float_const_loc(val as u64);
            }
            return Loc::immed(val);
        }
        if must_exist {
            panic!("RegisterManager.loc: box {:?} not found", v);
        }
        Loc::Frame(fm.loc(v, tp, false, longevity))
    }

    /// regalloc.py:625 return_constant
    pub fn return_constant(
        &mut self,
        v: &Operand,
        forbidden_vars: &[Operand],
        selected_reg: Option<RegLoc>,
        constants: &indexmap::IndexMap<u32, i64>,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
        _pending_moves: &mut Vec<(Loc, Loc)>,
    ) -> Loc {
        debug_assert!(v.is_constant());
        let immloc = self.convert_to_imm(v, constants);
        if let Some(selected) = selected_reg {
            if self.free_regs.contains(&selected) {
                self.spill_moves.push((immloc, Loc::Reg(selected)));
                return Loc::Reg(selected);
            }
            let spilled = self._spill_var(forbidden_vars, Some(selected), false, longevity, fm);
            self.free_regs.push(spilled);
            self.spill_moves.push((immloc, Loc::Reg(spilled)));
            return Loc::Reg(spilled);
        }
        immloc
    }

    /// regalloc.py:642
    pub fn make_sure_var_in_reg(
        &mut self,
        v: &Operand,
        tp: Type,
        forbidden_vars: &[Operand],
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
        constants: &indexmap::IndexMap<u32, i64>,
        pending_moves: &mut Vec<(Loc, Loc)>,
    ) -> Loc {
        if v.is_constant() {
            return self.return_constant(
                v,
                forbidden_vars,
                selected_reg,
                constants,
                longevity,
                fm,
                pending_moves,
            );
        }
        let prev_loc = self.loc(v, tp, true, longevity, fm, constants);
        if matches!(prev_loc, Loc::Reg(r) if r == self.frame_reg) && selected_reg.is_none() {
            return prev_loc;
        }
        let reg = self.force_allocate_reg(
            v,
            forbidden_vars,
            selected_reg,
            need_lower_byte,
            longevity,
            fm,
        );
        let new_loc = Loc::Reg(reg);
        if !loc_eq(&prev_loc, &new_loc) {
            self.spill_moves.push((prev_loc, new_loc));
        }
        new_loc
    }

    /// regalloc.py:661
    fn _reallocate_from_to(
        &mut self,
        from_v: &Operand,
        to_v: &Operand,
        longevity: &mut LifetimeManager,
    ) -> RegLoc {
        let reg = self.reg_bindings_pop(from_v, longevity);
        self.reg_bindings_set(to_v, reg, longevity);
        reg
    }

    /// regalloc.py:667
    pub fn force_result_in_reg(
        &mut self,
        result_v: &Operand,
        v: &Operand,
        tp: Type,
        forbidden_vars: &[Operand],
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
        constants: &indexmap::IndexMap<u32, i64>,
        _pending_moves: &mut Vec<(Loc, Loc)>,
    ) -> Loc {
        if v.is_constant() {
            let result_loc =
                self.force_allocate_reg(result_v, forbidden_vars, None, false, longevity, fm);
            let imm = self.convert_to_imm(v, constants);
            self.spill_moves.push((imm, Loc::Reg(result_loc)));
            return Loc::Reg(result_loc);
        }
        let v_keeps_living = longevity.get(v).unwrap().last_usage.get() > self.position;
        // regalloc.py:685 — two cases where we allocate a new register
        let not_in_reg = !self.reg_bindings_contains(v, longevity);
        if not_in_reg || (v_keeps_living && !self.free_regs.is_empty()) {
            let v_loc = self.loc(v, tp, false, longevity, fm, constants);
            let result_loc =
                self.force_allocate_reg(result_v, forbidden_vars, None, false, longevity, fm);
            let new_loc = Loc::Reg(result_loc);
            self.spill_moves.push((v_loc, new_loc));
            return new_loc;
        }
        if v_keeps_living {
            // regalloc.py:692 — no free registers, spill v
            self._sync_var_to_stack(v, tp, longevity, fm);
        }
        let reg = self._reallocate_from_to(v, result_v, longevity);
        Loc::Reg(reg)
    }

    /// regalloc.py:696
    pub fn _sync_var_to_stack(
        &mut self,
        v: &Operand,
        tp: Type,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
    ) {
        if fm.get(v, longevity).is_none() {
            // regalloc.py:696-704: allocate frame slot and emit move.
            let reg = self.reg_bindings_get(v, longevity);
            let to = fm.loc(v, tp, false, longevity);
            if let Some(r) = reg {
                // RPython: self.assembler.regalloc_mov(reg, to)
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[regalloc] _sync_var_to_stack: v={:?} reg={:?} → frame pos={}",
                        v, r, to.position
                    );
                }
                self.spill_moves.push((Loc::Reg(r), Loc::Frame(to)));
            }
        }
        // otherwise it's clean (already on stack)
    }

    /// regalloc.py _bc_spill
    fn _bc_spill(
        &mut self,
        v: &Operand,
        tp: Type,
        new_free_regs: &mut Vec<RegLoc>,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
    ) {
        self._sync_var_to_stack(v, tp, longevity, fm);
        let reg = self.reg_bindings_pop(v, longevity);
        new_free_regs.push(reg);
    }

    /// regalloc.py — RPython reads `v.type` directly during the
    /// reg-binding walk; pyre routes the lookup through `OpTypeIndex`
    /// since `OpRef(u32)` has no intrinsic type.
    pub fn before_call(
        &mut self,
        force_store: &[Operand],
        save_all_regs: u8,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
        pending_moves: &mut Vec<(Loc, Loc)>,
    ) {
        self.spill_or_move_registers_before_call(
            &self.save_around_call_regs.clone(),
            force_store,
            save_all_regs,
            longevity,
            fm,
            pending_moves,
        );
    }

    /// regalloc.py:714
    pub fn spill_or_move_registers_before_call(
        &mut self,
        save_sublist: &[RegLoc],
        force_store: &[Operand],
        save_all_regs: u8,
        longevity: &mut LifetimeManager,
        fm: &mut FrameManager,
        _pending_moves: &mut Vec<(Loc, Loc)>,
    ) {
        let mut new_free_regs: Vec<RegLoc> = Vec::new();
        let mut move_or_spill: Vec<Operand> = Vec::new();

        // regalloc.py iterate current bindings
        let items: Vec<(Operand, RegLoc)> = self.reg_bindings_items();
        for (v, reg) in items {
            let max_age = longevity.get(&v).unwrap().last_usage.get();
            // regalloc.py reads `v.type` directly; box.type is
            // intrinsic on every real Box (history.py:220). A type-less
            // binding is only a regalloc-scratch TempVar (`_check_type`
            // exemption, regalloc.py:405), resolved to Type::Int by
            // `box_type_or_temp`.
            let v_type = v.type_();
            if !force_store.contains(&v) && max_age <= self.position {
                // variable dies
                self.reg_bindings_del(&v, longevity);
                new_free_regs.push(reg);
                continue;
            }

            if save_all_regs == SAVE_ALL_REGS {
                // regalloc.py:782-784
                self._bc_spill(&v, v_type, &mut new_free_regs, longevity, fm);
            } else if save_all_regs == SAVE_GCREF_REGS && v_type == Type::Ref {
                // regalloc.py:786-788 — spill GC ptrs only
                self._bc_spill(&v, v_type, &mut new_free_regs, longevity, fm);
            } else if !save_sublist.contains(&reg) {
                continue; // callee-saved: fine where it is
            } else {
                move_or_spill.push(v);
            }
        }

        if !move_or_spill.is_empty() {
            // regalloc.py:799
            let free_regs: Vec<RegLoc> = self
                .free_regs
                .iter()
                .filter(|r| !self.save_around_call_regs.contains(r))
                .copied()
                .collect();
            // regalloc.py:802-805
            while move_or_spill.len() > free_regs.len() {
                let v =
                    self._pick_variable_to_spill(&[], None, false, Some(&move_or_spill), longevity);
                // RPython llsupport/regalloc.py routes through _bc_spill(v)
                // on the manager that owns `v`; the spill type follows that
                // manager (xrm -> Float, rm -> Int/Ref), not a hardcoded Int.
                self._bc_spill(&v, self._type_of(&v), &mut new_free_regs, longevity, fm);
                move_or_spill.retain(|x| *x != v);
            }
            // regalloc.py:807-821
            for v in &move_or_spill {
                let new_reg = loop {
                    let r = self.free_regs.pop().expect("no free register for move");
                    if self.save_around_call_regs.contains(&r) {
                        new_free_regs.push(r);
                        continue;
                    }
                    break r;
                };
                let old_reg = self.reg_bindings_get(v, longevity).unwrap();
                self.spill_moves
                    .push((Loc::Reg(old_reg), Loc::Reg(new_reg)));
                self.reg_bindings_set(v, new_reg, longevity);
                new_free_regs.push(old_reg);
            }
        }

        // regalloc.py:826-827 re-add in reverse
        while let Some(r) = new_free_regs.pop() {
            self.free_regs.push(r);
        }
    }

    /// regalloc.py:829
    pub fn after_call(&mut self, v: &Operand, longevity: &mut LifetimeManager) -> RegLoc {
        let r = self.call_result_location();
        self.reg_bindings_set(v, r, longevity);
        self.free_regs.retain(|fr| *fr != r);
        r
    }

    // ── x86-specific methods ──

    /// `X86RegisterManager.convert_to_imm` / `X86XMMRegisterManager.convert_to_imm`
    /// / `VFPRegisterManager.convert_to_imm`.
    ///
    /// The XMM / VFP manager (`datablockwrapper` installed) does
    /// `malloc_aligned(8, 8)`, writes `getfloatstorage()`, and returns
    /// `ConstFloatLoc`. A `MemoryError` from `open_malloc` is kept in
    /// `pool_error` and surfaced by `walk_operations`. The GPR manager
    /// returns `imm`.
    pub fn convert_to_imm(&mut self, v: &Operand, constants: &indexmap::IndexMap<u32, i64>) -> Loc {
        debug_assert!(v.is_constant());
        let val = const_bits_or_panic(v, constants, "convert_to_imm");
        if self.datablockwrapper.is_some() {
            return self.float_const_loc(val as u64);
        }
        // x86/regalloc.py:58-61: a non-null `ConstPtr` whose object can
        // still move must never be baked as an immediate — `remove_constptr`
        // (rewrite.py:1100) routes those through the gc_table. Null and
        // non-moving (prebuilt / old-gen / no active GC) ConstPtrs are baked
        // directly. `rgc.can_move` parity via `majit_gc::can_move`; the
        // `we_are_translated()` clause is subsumed because `can_move` is
        // false when no moving GC is active.
        if let Some(majit_ir::Value::Ref(g)) = v.const_value()
            && !g.is_null()
            && majit_gc::can_move(g)
        {
            panic!("convert_to_imm: ConstPtr needs special care");
        }
        if self.is_float_constant(v) {
            Loc::immed_float(val)
        } else {
            Loc::immed(val)
        }
    }

    /// `X86XMMRegisterManager.convert_to_imm` / `VFPRegisterManager.convert_to_imm`
    /// when this manager owns the data block; otherwise the bit pattern stays
    /// an `ImmedFloat`.
    pub(crate) fn float_const_loc(&mut self, bits: u64) -> Loc {
        let Some(wrapper) = self.datablockwrapper else {
            return Loc::immed_float(bits as i64);
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Installed only for the duration of the walk that borrows it.
            let wrapper = unsafe { &mut *wrapper.as_ptr() };
            match park_float_const(wrapper, bits) {
                Ok(loc) => Loc::ConstFloat(loc),
                Err(err) => {
                    if self.pool_error.is_none() {
                        self.pool_error = Some(err);
                    }
                    // Not emitted: `walk_operations` returns before the assembler.
                    Loc::ConstFloat(ConstFloatLoc { value: 0 })
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        match wrapper {}
    }

    /// x86/regalloc.py call_result_location → eax (GPR) or xmm0 (XMM).
    pub fn call_result_location(&self) -> RegLoc {
        self.call_result_reg
    }

    /// Infer type for an OpRef (INT for GPR manager, FLOAT for XMM).
    fn _type_of(&self, _v: &Operand) -> Type {
        match &self.box_types {
            Some(types) if types.contains(&Type::Float) => Type::Float,
            _ => Type::Int,
        }
    }
}

// ── RegAlloc ───────────────────────────────────────────────────────

/// x86/regalloc.py `self.assembler` / aarch64/regalloc.py `self.assembler`:
/// the walk drives the assembler directly. `perform*` hand it each op's
/// locations while the register managers still hold the state of that
/// point, so an emitter reads allocator facts (`xrm.reg_bindings` in
/// `_write_barrier_fastpath`, `rm.reg_bindings` in `cond_call`, the frame
/// depth of a guard's gcmap) off `regalloc` instead of extra operands.
/// `regalloc_mov` comes from [`RegallocMoves`], which `remap_frame_layout`
/// drives the same way.
pub(crate) trait RegAllocAssembler: RegallocMoves {
    /// assembler.py `regalloc_perform`. `gcmap` is the map a collecting op
    /// captured with `get_gcmap` before its call (`malloc_cond`, `cond_call`).
    fn regalloc_perform(
        &mut self,
        regalloc: &RegAlloc<'_>,
        op_index: usize,
        arglocs: &[Loc],
        result_loc: Option<Loc>,
        gcmap: Option<usize>,
    );
    /// assembler.py `regalloc_perform_discard`.
    fn regalloc_perform_discard(
        &mut self,
        regalloc: &RegAlloc<'_>,
        op_index: usize,
        arglocs: &[Loc],
    );
    /// assembler.py `regalloc_perform_guard`.
    fn regalloc_perform_guard(
        &mut self,
        regalloc: &RegAlloc<'_>,
        op_index: usize,
        faillocs: &[Option<Loc>],
        arglocs: &[Loc],
        result_loc: Option<Loc>,
        frame_depth: usize,
    );
    /// x86/regalloc.py `consider_zero_array`'s `self.assembler.mc.LEA(loc,
    /// self._scaled_addr(index_loc, itemsize_loc, base_loc, ofs_loc))`,
    /// emitted from the allocator before its later loads and spills.
    #[cfg(target_arch = "x86_64")]
    fn regalloc_lea_scaled_addr(
        &mut self,
        dst: Loc,
        index_loc: &Loc,
        itemsize: i64,
        base_loc: &Loc,
        ofs: i64,
    );
}

/// regalloc.py `BaseRegalloc` — the state x86/regalloc.py `RegAlloc` and
/// aarch64/regalloc.py `Regalloc` share (`rm`/`xrm` are `vfprm` on aarch64)
/// and the bookkeeping both define identically: `loc`, `possibly_free_var`,
/// `force_allocate_reg`, `perform*`, `locs_for_fail`, `get_gcmap`, the walk.
/// The opcode dispatch and every `consider_*` / `prepare_op_*` method are in
/// the arch module.
///
/// Borrows the trace's `inputargs` and `operations` for its lifetime so
/// `OpRef → Type` can resolve via `op.type_` / `inputarg.tp` directly
/// (RPython `box.type` parity); no `value_types: HashMap` side-table.
pub struct RegAlloc<'a> {
    /// Lifetime information for all variables.
    pub longevity: LifetimeManager,
    /// GPR register manager — x86/regalloc.py X86RegisterManager
    pub rm: RegisterManager,
    /// XMM register manager — x86/regalloc.py X86XMMRegisterManager
    pub xrm: RegisterManager,
    /// Frame manager — x86/regalloc.py X86FrameManager
    pub fm: FrameManager,
    /// Constants map (OpRef const_index → i64 value).
    pub constants: indexmap::IndexMap<u32, i64>,
    /// Pending register moves to be emitted by the assembler.
    /// Each entry is (source_loc, dest_loc).
    pub pending_moves: Vec<(Loc, Loc)>,
    /// Trace inputargs — borrowed for `opref_type` lookups.
    pub(crate) inputargs: &'a [InputArgRc],
    /// Trace operations — borrowed for `opref_type` lookups (reads
    /// `op.type_` directly, RPython `box.type` parity).
    pub(crate) operations: &'a [OpRc],
    /// `arg.index` raw -> idx in inputargs. Mirrors
    /// `OpTypeIndex::inputarg_pos`.
    pub(crate) inputarg_pos: majit_ir::PosIndex,
    /// `op.pos.raw()` -> idx in operations, skipping Void/None ops.
    /// Mirrors `OpTypeIndex::op_pos`.
    pub(crate) op_pos: majit_ir::PosIndex,
    /// x86/regalloc.py:1305 self.jump_target_descr — TargetToken descriptor
    /// of the closing JUMP. PyPy keys it by Python object identity; we use
    /// the underlying `Arc<dyn Descr>` allocation address.
    pub(crate) jump_target_descr: Option<usize>,
    /// x86/regalloc.py:1407 self.final_jump_op — paired (descr identity,
    /// jump arglist) so consider_label can recognize a self-loop and emit
    /// frame position hints via add_frame_pos_hint.
    pub(crate) final_jump_args: Option<(usize, Vec<Operand>)>,
    /// x86/regalloc.py:1270 self.final_jump_op_position — position of the
    /// closing JUMP, used by `_compute_hint_locations_from_descr` to pin
    /// reg hints via `longevity.fixed_register(position, reg, box)`.
    pub(crate) final_jump_op_position: i32,
    /// Per-trace counter feeding `OpRef::fresh_temp_var()` — RPython;
    /// `TempVar()` / `TempInt()` allocate a fresh Python object per
    /// call (`x86/regalloc.py:470,514,521,605`,
    /// `aarch64/regalloc.py`); pyre's flat-OpRef encoding needs a
    /// counter to feed the unique raw payload reservation.
    pub(crate) temp_var_counter: u32,
    /// `locs_for_fail`'s list, kept across guards so a guard does not
    /// allocate a fresh `Vec`.
    pub(crate) faillocs: Vec<Option<Loc>>,
}

pub(crate) fn is_math_sqrt_call(op: &Op) -> bool {
    op.with_call_descr(|cd| cd.get_extra_info().oopspecindex)
        == Some(majit_ir::OopSpecIndex::MathSqrt)
}

impl<'a> RegAlloc<'a> {
    /// x86/regalloc.py:170
    pub fn new(
        constants: indexmap::IndexMap<u32, i64>,
        inputargs: &'a [InputArgRc],
        operations: &'a [OpRc],
    ) -> Self {
        // We create with empty longevity/managers; they get initialized in _prepare.
        let longevity = LifetimeManager::new();
        let rm = Self::make_gpr_manager();
        let xrm = Self::make_xmm_manager();
        let fm = FrameManager::new(0);
        let inputarg_pos = OpTypeIndex::<OpRc, InputArgRc>::build_inputarg_pos(inputargs);
        let op_pos = OpTypeIndex::<OpRc, InputArgRc>::build_op_pos(operations);
        RegAlloc {
            longevity,
            rm,
            xrm,
            fm,
            constants,
            pending_moves: Vec::new(),
            inputargs,
            operations,
            inputarg_pos,
            op_pos,
            jump_target_descr: None,
            final_jump_args: None,
            final_jump_op_position: -1,
            temp_var_counter: 0,
            faillocs: Vec::new(),
        }
    }

    /// `regalloc.py TempVar()`. Each call returns an `Operand` holding a
    /// fresh `TempVarRc`. Callers that still take `OpRef` use `.to_opref()`.
    /// `RegisterManager::try_allocate_reg` / `force_allocate_reg` insert the
    /// Lifetime.
    pub fn fresh_temp_var(&mut self) -> Operand {
        let counter = self.temp_var_counter;
        self.temp_var_counter += 1;
        let OpRef::TempVar(id) = OpRef::fresh_temp_var(counter) else {
            unreachable!("fresh_temp_var always returns OpRef::TempVar");
        };
        Operand::temp_var(TempVar::new_rc(id))
    }

    /// rpython/jit/metainterp/history.py is_constant `box.type` parity.
    /// Single source of truth: `op.type_` for ops, `inputarg.tp` for
    /// inputargs, and the variant tag for constant OpRefs.  During the
    /// regalloc walk, flat-OpRef collisions are resolved at the current
    /// operation index so a future op result cannot shadow the live
    /// inputarg Box.
    #[inline]
    pub fn opref_type(&self, opref: OpRef) -> Option<Type> {
        let at = (self.rm.position >= 0).then_some(self.rm.position as usize);
        self.opref_type_at(opref, at)
    }

    #[inline]
    pub(crate) fn opref_type_at(&self, opref: OpRef, at_op_index: Option<usize>) -> Option<Type> {
        // history.py:182 / resoperation.py AbstractValue: `box.type` lives on the  allow-line-citation
        // Box object itself; pyre's typed `OpRef` variants carry the
        // matching type tag (ConstInt/InputArgInt/IntOp …) so the
        // variant tag IS Box class identity. Delegate to the shared
        // `OpTypeIndex` so the variant-tag fast path and side-table
        // fallbacks stay in lockstep with `compile.rs` and the rest of
        // the metainterp.
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        match at_op_index {
            Some(at) => type_index.opref_type_at(opref, at),
            None => type_index.opref_type(opref),
        }
    }

    /// x86/regalloc.py X86_64_RegisterManager configuration.
    pub(crate) fn make_gpr_manager() -> RegisterManager {
        let all_regs = all_core_regs();
        // x86/regalloc.py:71
        let no_lower_byte_regs = vec![];
        let save_around_call_regs = save_around_call_core_regs();
        let frame_reg = frame_reg();
        RegisterManager::new(
            all_regs,
            no_lower_byte_regs,
            save_around_call_regs,
            frame_reg,
            Some(vec![Type::Int, Type::Ref]),
            call_result_gpr(),
        )
    }

    /// x86/regalloc.py X86_64_XMMRegisterManager configuration.
    pub(crate) fn make_xmm_manager() -> RegisterManager {
        let all_regs = all_float_regs();
        let save_around_call_regs = all_regs.clone();
        let frame_reg = frame_reg();
        RegisterManager::new(
            all_regs,
            vec![],
            save_around_call_regs,
            frame_reg,
            Some(vec![Type::Float]),
            call_result_fpr(),
        )
    }

    /// x86/regalloc.py prepare_loop
    pub fn prepare_loop(&mut self) {
        self._prepare();
        let inputargs = self.inputargs;
        self._set_initial_bindings(inputargs);
        // Free input variables that are dead at position 0
        for iarg in inputargs {
            let tp = iarg.tp.get();
            let box_ = Operand::from_bound_inputarg(iarg);
            if tp == Type::Float {
                self.xrm
                    .possibly_free_var(&box_, &mut self.longevity, &mut self.fm, tp);
            } else {
                self.rm
                    .possibly_free_var(&box_, &mut self.longevity, &mut self.fm, tp);
            }
        }
        self.compute_hint_frame_locations();
    }

    /// x86/regalloc.py prepare_bridge
    pub fn prepare_bridge(&mut self, arglocs: &[Loc]) {
        self._prepare();
        let inputargs = self.inputargs;
        self._update_bindings(arglocs, inputargs);
        self.compute_hint_frame_locations();
    }

    /// regalloc.py _set_initial_bindings — place all inputargs in frame slots.
    pub(crate) fn _set_initial_bindings(&mut self, inputargs: &[InputArgRc]) {
        for iarg in inputargs {
            let box_ = Operand::from_bound_inputarg(iarg);
            let _loc = self
                .fm
                .get_new_loc(&box_, iarg.tp.get(), &mut self.longevity);
        }
    }

    /// x86/regalloc.py _update_bindings — bind bridge inputargs to their locations.
    pub(crate) fn _update_bindings(&mut self, locs: &[Loc], inputargs: &[InputArgRc]) {
        let mut used: IndexMap<RegLoc, (), FxBuildHasher> = IndexMap::with_hasher(FxBuildHasher);

        // x86/regalloc.py:295-312
        for (iarg, loc) in inputargs.iter().zip(locs.iter()) {
            let v = Operand::from_bound_inputarg(iarg);
            let tp = iarg.tp.get();
            match loc {
                Loc::Reg(reg) => {
                    if tp == Type::Float {
                        self.xrm.reg_bindings_set(&v, *reg, &mut self.longevity);
                        used.insert(*reg, ());
                    } else if *reg == self.rm.frame_reg {
                        // x86/regalloc.py:305-307
                        debug_assert!(self.rm.box_currently_in_frame_reg.is_none());
                        self.rm.box_currently_in_frame_reg = Some(v);
                    } else {
                        self.rm.reg_bindings_set(&v, *reg, &mut self.longevity);
                        used.insert(*reg, ());
                    }
                }
                Loc::Frame(floc) => {
                    self.fm.bind(&v, *floc, tp, &mut self.longevity);
                }
                _ => {
                    self.fm.bind(
                        &v,
                        FrameManager::frame_pos(0, tp, self.fm.base_ofs),
                        tp,
                        &mut self.longevity,
                    );
                }
            }
        }

        // x86/regalloc.py:313-320 rebuild free_regs from scratch
        self.rm.free_regs = Vec::new();
        for &reg in &self.rm.all_regs {
            if !used.contains_key(&reg) {
                self.rm.free_regs.push(reg);
            }
        }
        self.xrm.free_regs = Vec::new();
        for &reg in &self.xrm.all_regs {
            if !used.contains_key(&reg) {
                self.xrm.free_regs.push(reg);
            }
        }

        // x86/regalloc.py:321
        for iarg in inputargs {
            let box_ = Operand::from_bound_inputarg(iarg);
            self.possibly_free_var(&box_, iarg.tp.get());
        }
        // x86/regalloc.py:322
        self.fm.finish_binding();
    }

    // ── Location dispatch (type-based routing to rm or xrm) ──

    /// x86/regalloc.py loc(v)
    pub fn loc(&mut self, v: &Operand, tp: Type) -> Loc {
        self.loc_with_mode(v, tp, false)
    }

    pub fn loc_must_exist(&mut self, v: &Operand, tp: Type) -> Loc {
        self.loc_with_mode(v, tp, true)
    }

    pub(crate) fn loc_with_mode(&mut self, v: &Operand, tp: Type, must_exist: bool) -> Loc {
        if tp == Type::Float {
            self.xrm.loc(
                v,
                tp,
                must_exist,
                &mut self.longevity,
                &mut self.fm,
                &self.constants,
            )
        } else {
            self.rm.loc(
                v,
                tp,
                must_exist,
                &mut self.longevity,
                &mut self.fm,
                &self.constants,
            )
        }
    }

    /// x86/regalloc.py possibly_free_var(v)
    pub fn possibly_free_var(&mut self, v: &Operand, tp: Type) {
        if tp == Type::Float {
            self.xrm
                .possibly_free_var(v, &mut self.longevity, &mut self.fm, tp);
        } else {
            self.rm
                .possibly_free_var(v, &mut self.longevity, &mut self.fm, tp);
        }
    }

    /// x86/regalloc.py `RegAlloc.possibly_free_vars_for_op`.
    /// `perform_guard` already freed `getfailargs()`.
    pub fn possibly_free_vars_for_op(&mut self, op: &Op, dst: &Operand) {
        self._possibly_free_vars(op.args_slice().iter());
        let tp = op.opcode.result_type();
        if tp != Type::Void && !dst.is_none() {
            self.possibly_free_var(dst, tp);
        }
    }

    /// x86/regalloc.py force_allocate_reg
    pub fn force_allocate_reg(
        &mut self,
        v: &Operand,
        tp: Type,
        forbidden_vars: &[Operand],
        selected_reg: Option<RegLoc>,
        need_lower_byte: bool,
    ) -> RegLoc {
        if tp == Type::Float {
            self.xrm.force_allocate_reg(
                v,
                forbidden_vars,
                selected_reg,
                need_lower_byte,
                &mut self.longevity,
                &mut self.fm,
            )
        } else {
            self.rm.force_allocate_reg(
                v,
                forbidden_vars,
                selected_reg,
                need_lower_byte,
                &mut self.longevity,
                &mut self.fm,
            )
        }
    }

    /// x86/regalloc.py force_spill_var
    pub fn force_spill_var(&mut self, v: &Operand, tp: Type) {
        if tp == Type::Float {
            self.xrm
                .force_spill_var(v, tp, &mut self.longevity, &mut self.fm);
        } else {
            self.rm
                .force_spill_var(v, tp, &mut self.longevity, &mut self.fm);
        }
    }

    // ── Perform / emit ──

    /// Hand the queued moves to the assembler before the op that needs them.
    pub(crate) fn flush_moves(&mut self, asm: &mut dyn RegAllocAssembler) {
        // Operand moves and spills share one queue per register manager and
        // run in the order they were queued. Integer and float registers do
        // not alias, so one manager's queue flushes before the other's.
        // `pending_moves` is the out-of-line immediate materialisation from
        // `gc_offset_loc`, queued after those operand moves.
        let moves = self.rm.spill_moves.drain(..);
        let moves = moves.chain(self.xrm.spill_moves.drain(..));
        for (src, dst) in moves.chain(self.pending_moves.drain(..)) {
            if majit_ir::debug::have_debug_prints() {
                majit_ir::debug::log_one("jit-backend", &format!("move: {src:?} → {dst:?}"));
            }
            asm.regalloc_mov(&src, &dst);
        }
    }

    /// x86/regalloc.py perform
    pub(crate) fn perform(
        &mut self,
        op_index: usize,
        arglocs: impl AsRef<[Loc]>,
        result_loc: Option<Loc>,
        asm: &mut dyn RegAllocAssembler,
    ) {
        self.flush_moves(asm);
        asm.regalloc_perform(self, op_index, arglocs.as_ref(), result_loc, None);
    }

    pub(crate) fn perform_with_gcmap_ptr(
        &mut self,
        op_index: usize,
        arglocs: impl AsRef<[Loc]>,
        result_loc: Option<Loc>,
        gcmap: usize,
        asm: &mut dyn RegAllocAssembler,
    ) {
        self.flush_moves(asm);
        asm.regalloc_perform(self, op_index, arglocs.as_ref(), result_loc, Some(gcmap));
    }

    /// Variant of `perform` that captures the current regalloc state as a
    /// GC bitmap, for collecting-call ops such as CallMallocNursery. The
    /// emit path stores the returned pointer into `jf_gcmap` before the
    /// slow-path call so the collector can trace live Refs pinned to the
    /// jitframe.
    pub(crate) fn perform_with_gcmap(
        &mut self,
        op_index: usize,
        arglocs: impl AsRef<[Loc]>,
        result_loc: Option<Loc>,
        asm: &mut dyn RegAllocAssembler,
    ) {
        // aarch64/regalloc.py prepare_op_call_malloc_nursery:
        //   `gcmap = self.get_gcmap([r.x0, r.x1])`
        // The malloc-nursery clobber regs are the result register plus a
        // scratch temp — both are reserved for the slow-path convention
        // (x0 = total_size in, allocated pointer out) and cannot carry a
        // Ref spill through the collecting call. Excluding them from the
        // bitmap prevents the collector from reading garbage out of
        // jitframe slots 0/1 during `do_collect_nursery`.
        let gcmap = self.get_gcmap(&MALLOC_NURSERY_CLOBBER, false) as usize;
        self.perform_with_gcmap_ptr(op_index, arglocs, result_loc, gcmap, asm);
    }

    /// x86/regalloc.py perform_guard
    pub(crate) fn perform_guard(
        &mut self,
        op: &Op,
        op_index: usize,
        arglocs: impl AsRef<[Loc]>,
        result_loc: Option<Loc>,
        asm: &mut dyn RegAllocAssembler,
    ) {
        self.flush_moves(asm);
        let faillocs = self.locs_for_fail(op);
        asm.regalloc_perform_guard(
            self,
            op_index,
            &faillocs,
            arglocs.as_ref(),
            result_loc,
            self.fm.get_frame_depth(),
        );
        self.faillocs = faillocs;
        // self.possibly_free_vars(guard_op.getfailargs())
        if let Some(fail_args) = op.guard_fail_args() {
            self._possibly_free_vars(fail_args.iter());
        }
    }

    /// aarch64/regalloc.py get_gcmap.
    pub fn get_gcmap(&self, forbidden_regs: &[RegLoc], noregs: bool) -> *mut usize {
        let frame_depth = self.fm.get_frame_depth();
        let gcmap = allocate_gcmap(frame_depth, JITFRAME_FIXED_SIZE);
        for (v, reg) in self.rm.reg_bindings_iter() {
            if forbidden_regs.contains(&reg) {
                continue;
            }
            if v.type_() == Type::Ref && self.rm.is_still_alive(&v, &self.longevity) {
                assert!(!noregs);
                let val = core_reg_index(reg).expect("gcmap: reg not in managed core regs");
                gcmap_set_bit(gcmap, val);
            }
        }
        // regalloc.py BindingsIterItems; regalloc.py bindings_iteritems.
        let mut index = 0;
        while index < self.fm.current_frame_depth {
            let Some(v) = self.fm.boxes_in_frame[index].as_ref() else {
                index += 1;
                continue;
            };
            let loc = self
                .longevity
                .get(v)
                .and_then(|lifetime| lifetime.current_frame_loc.get())
                .expect("gcmap frame binding without current_frame_loc");
            debug_assert_eq!(FrameManager::get_loc_index(&loc), index);
            let size = FrameManager::frame_size(self.tp(v));
            if v.type_() == Type::Ref && self.rm.is_still_alive(v, &self.longevity) {
                gcmap_set_bit(gcmap, loc.get_position() + JITFRAME_FIXED_SIZE);
            }
            index += size;
        }
        gcmap
    }

    /// x86/regalloc.py perform_discard
    pub(crate) fn perform_discard(
        &mut self,
        op_index: usize,
        arglocs: impl AsRef<[Loc]>,
        asm: &mut dyn RegAllocAssembler,
    ) {
        self.flush_moves(asm);
        asm.regalloc_perform_discard(self, op_index, arglocs.as_ref());
    }

    /// `GC_STORE`'s `[value, base, ofs, imm(size)]`.
    pub(crate) fn perform_discard_gc_store(
        &mut self,
        op_index: usize,
        value: Loc,
        base: Loc,
        ofs: Loc,
        size: i64,
        asm: &mut dyn RegAllocAssembler,
    ) {
        self.perform_discard(op_index, [value, base, ofs, Loc::immed(size)], asm);
    }

    /// `GC_LOAD`'s `[base, ofs, res, imm(nsize)]`.
    pub(crate) fn perform_gc_load(
        &mut self,
        op_index: usize,
        base: Loc,
        ofs: Loc,
        res: Loc,
        nsize: i64,
        asm: &mut dyn RegAllocAssembler,
    ) {
        self.perform(
            op_index,
            [base, ofs, res, Loc::immed(nsize)],
            Some(res),
            asm,
        );
    }

    /// x86/regalloc.py locs_for_fail. Iterates `guard_op.getfailargs()`
    /// directly; an absent list skipped `store_final_boxes_in_guard` /
    /// `ResumeDataVirtualAdder.finish`.
    pub fn locs_for_fail(&mut self, guard_op: &Op) -> Vec<Option<Loc>> {
        let fail_args = guard_op.guard_fail_args().unwrap_or_else(|| {
            panic!(
                "{:?} reached the backend without fail args",
                guard_op.opcode
            )
        });
        let mut faillocs = std::mem::take(&mut self.faillocs);
        faillocs.clear();
        for arg in fail_args.iter() {
            if arg.is_none() {
                faillocs.push(None);
                continue;
            }
            let tp = self.tp(arg);
            // x86/regalloc.py `RegAlloc.locs_for_fail` calls `self.loc(arg)`:
            // a box reused from ResumeDataLoopMemo may be live only in this
            // guard and therefore need its frame binding allocated here.
            let loc = self.loc(arg, tp);
            faillocs.push(Some(loc));
        }
        faillocs
    }

    // ── Type resolution ──

    /// `box.type` on the bound operand (`history.py` ConstInt / ConstFloat /
    /// ConstPtr pin `type` at construction). A regalloc-scratch `TempVar`
    /// answers `Type::Int` (`regalloc.py` `_check_type` TempVar exemption).
    pub(crate) fn tp(&self, v: &Operand) -> Type {
        v.type_()
    }

    /// Extract the integer value of a constant OpRef (like RPython's getint()).
    ///
    /// The constants map is keyed by `OpRef.0` (raw value, WITH the
    /// CONST_BIT tag preserved) — see `convert_to_imm` at regalloc.rs
    /// line 1475 and the fail-args handling at line 2079. This helper
    /// must use the same key convention.
    #[track_caller]
    pub(crate) fn const_value(&self, v: &Operand) -> i64 {
        const_bits_or_panic(v, &self.constants, "const_value")
    }

    // ── walk_operations + consider_* ──

    /// `walk_operations` with `X86XMMRegisterManager` / `VFPRegisterManager`
    /// reaching `self.assembler.datablockwrapper`. `prepare_*` rebuilds
    /// `xrm`, so the wrapper is installed here, borrowed for exactly this
    /// walk, and detached before returning or unwinding: `float_const_loc`
    /// never reads it once the borrow ends.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn walk_operations_with_datablockwrapper(
        &mut self,
        datablockwrapper: &mut majit_backend::MachineDataBlockWrapper,
        asm: &mut dyn RegAllocAssembler,
    ) -> Result<(), BackendError> {
        self.xrm.datablockwrapper = Some(std::ptr::NonNull::from(datablockwrapper));
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.walk_operations(asm)));
        self.xrm.datablockwrapper = None;
        result.unwrap_or_else(|payload| std::panic::resume_unwind(payload))
    }

    /// x86/regalloc.py walk_operations — main dispatch loop.
    ///
    /// `MachineDataBlockWrapper.malloc_aligned` → `_allocate_next_block` →
    /// `open_malloc` raises `MemoryError`. That becomes
    /// `BackendError::CompilationFailed` for `assemble_loop` / `assemble_bridge`.
    pub(crate) fn walk_operations(
        &mut self,
        asm: &mut dyn RegAllocAssembler,
    ) -> Result<(), BackendError> {
        let operations: &'a [OpRc] = self.operations;
        let inputargs: &'a [InputArgRc] = self.inputargs;

        for (i, op) in operations.iter().enumerate() {
            self.rm.position = i as i32;
            self.xrm.position = i as i32;

            // x86/regalloc.py:383-386 skip dead ops
            let result = Operand::from_bound_op(op);
            if op.opcode.has_no_side_effect() && !self.longevity.contains(&result) {
                self.possibly_free_vars_for_op(op, &result);
                continue;
            }

            // aarch64/assembler.py `_walk_operations`: an
            // `INT_*_OVF` op publishes its overflow answer in the condition
            // flags, and the guard that reads them is `operations[i + 1]`.
            // Anything emitted in between overwrites NZCV, so the guard would
            // branch on the wrong comparison. Upstream asserts the adjacency
            // rather than tolerating it; so do we — `compile_loop` catches the
            // panic and declines the trace, which beats a silent wrong answer.
            assert!(
                !op.opcode.is_ovf()
                    || matches!(
                        operations.get(i + 1).map(|n| n.opcode),
                        Some(OpCode::GuardOverflow) | Some(OpCode::GuardNoOverflow)
                    ),
                "{:?} at {i} is not followed by GUARD_[NO_]OVERFLOW (got {:?})",
                op.opcode,
                operations.get(i + 1).map(|n| n.opcode),
            );

            // x86/regalloc.py walk_operations: oplist[op.getopnum()](self, op)
            // then possibly_free_vars_for_op(op).
            self._dispatch_op(op, i, asm);
            self.possibly_free_vars_for_op(op, &Operand::from_bound_op(op));
            if let Some(err) = self
                .xrm
                .pool_error
                .take()
                .or_else(|| self.rm.pool_error.take())
            {
                return Err(BackendError::CompilationFailed(format!(
                    "MachineDataBlockWrapper.malloc_aligned: {err}"
                )));
            }
        }

        // x86/regalloc.py:400-401 free inputargs
        for iarg in inputargs {
            let box_ = Operand::from_bound_inputarg(iarg);
            self.possibly_free_var(&box_, iarg.tp.get());
        }

        Ok(())
    }

    /// Free operands in order, skipping constants and None.
    pub(crate) fn _possibly_free_vars<'b>(&mut self, uses: impl Iterator<Item = &'b Operand>) {
        for arg in uses {
            if !arg.is_constant() && !arg.is_none() {
                let tp = self.tp(arg);
                self.possibly_free_var(arg, tp);
            }
        }
    }

    /// llsupport/regalloc.py `next_op_can_accept_cc` parity.
    ///
    /// Returns `true` when `operations[i]`'s result is consumed solely
    /// by `operations[i + 1]`, and that op is `GuardTrue/False` or
    /// `CondCallN`. We also accept `GuardNonnull/Isnull` against an
    /// integer compare since our emit treats those equivalently. The
    /// CompOp emitter then leaves the result in the condition flags
    /// instead of materialising it via `setcc`+`movzx`, and the next
    /// op's emit consumes those flags directly:
    ///
    /// - `Guard{True,False,Nonnull,Isnull}` via `implement_guard`.
    /// - `CondCallN` via `guard_success_cc` (see
    ///   `genop_discard_cond_call`, mirrors `x86/assembler.py
    ///   cond_call`).
    pub(crate) fn next_op_can_accept_cc<T: AsRef<Op>>(
        &self,
        ops: &[T],
        i: usize,
        result: &Operand,
    ) -> bool {
        if i + 1 >= ops.len() {
            return false;
        }
        let next_op = ops[i + 1].as_ref();
        let opnum = next_op.opcode;
        if !matches!(
            opnum,
            OpCode::GuardTrue
                | OpCode::GuardFalse
                | OpCode::GuardNonnull
                | OpCode::GuardIsnull
                | OpCode::CondCallN
        ) {
            return false;
        }
        // history.py `Const.is_constant()` — Const operands are not
        // op-result identities; comparison via raw position is invalid.
        if next_op.num_args() == 0
            || next_op.arg(0).is_constant()
            || result.is_constant()
            || !next_op.arg(0).is_same_box(result)
        {
            return false;
        }
        if let Some(lt) = self.longevity.get(result)
            && lt.last_usage.get() > (i as i32) + 1
        {
            return false;
        }
        // history.py `Const.same_constant` — variant-aware equality
        // (`OpRef`'s derived `PartialEq`) handles inline-Const fail args /
        // cond_call tail args without calling `.raw()`, which panics on
        // inline-Const variants. Identity comparison against a non-Const
        // `result` always yields `false` for any Const operand, which is
        // the correct semantics: a constant fail-arg can never alias a
        // body op result.
        if opnum != OpCode::CondCallN {
            if let Some(fail_args) = next_op.guard_fail_args()
                && fail_args.iter().any(|a| a.is_same_box(result))
            {
                return false;
            }
        } else if next_op
            .args_slice()
            .iter()
            .skip(1)
            .any(|a| a.is_same_box(result))
        {
            return false;
        }
        true
    }

    /// x86/regalloc.py `force_allocate_reg_or_cc` parity.
    ///
    /// If the next op consumes the condition flags directly, return
    /// the `frame_reg` (rbp) sentinel — the assembler reads this back
    /// in the CompOp emit and threads `guard_success_cc` through to the
    /// following guard. Otherwise force-allocate a general-purpose
    /// register (with `need_lower_byte` so `SETcc r8b` is encodable).
    ///
    /// Both backends recognise the sentinel: `x86/assembler.rs flush_cc`
    /// and `aarch64/assembler.rs flush_cc` publish `guard_success_cc` and
    /// emit nothing when `result_loc` is the frame register, so neither
    /// clobbers rbp / x29.
    pub(crate) fn force_allocate_reg_or_cc<T: AsRef<Op>>(
        &mut self,
        result: &Operand,
        ops: &[T],
        i: usize,
    ) -> Loc {
        if self.next_op_can_accept_cc(ops, i, result) {
            self.rm.force_allocate_frame_reg(result);
            return Loc::Reg(arch_regalloc::frame_reg());
        }
        Loc::Reg(self.force_allocate_reg(result, Type::Int, &[], None, true))
    }

    /// `RegisterManager.return_constant` with `selected_reg` queues
    /// `convert_to_imm`'s `ConstFloatLoc` into that xmm/vfp register, but
    /// leaves the register in `free_regs`. A `TempVar` owns it afterwards
    /// (`VFPRegisterManager.get_scratch_reg` / `temp_boxes`) so the next
    /// ConstFloat in this instruction cannot reuse it. Binding first would
    /// make `return_constant` spill the scratch: it only queues the move
    /// while the register is still free.
    pub(crate) fn const_float_in_reg(&mut self, v: &Operand, forbidden: &[Operand]) -> Loc {
        debug_assert!(v.is_constant() && self.tp(v) == Type::Float);
        let tmp = self.fresh_temp_var();
        let reg = if let Some(selected) = self.xrm.free_regs.last().copied() {
            let loaded = self.xrm.return_constant(
                v,
                forbidden,
                Some(selected),
                &self.constants,
                &mut self.longevity,
                &mut self.fm,
                &mut self.pending_moves,
            );
            let Loc::Reg(reg) = loaded else {
                panic!("return_constant(selected_reg) must yield a register, got {loaded:?}");
            };
            self.force_allocate_reg(&tmp, Type::Float, forbidden, Some(reg), false)
        } else {
            let reg = self.force_allocate_reg(&tmp, Type::Float, forbidden, None, false);
            let imm = self.xrm.convert_to_imm(v, &self.constants);
            self.xrm.spill_moves.push((imm, Loc::Reg(reg)));
            reg
        };
        self.xrm.temp_boxes.push(tmp);
        Loc::Reg(reg)
    }

    /// `emit_call_from_arglocs` silently falls back to all-Int argument types
    /// when the descr's arity disagrees with the call, which would place a
    /// float argument in a GPR. Every cond-call-value producer builds the arg
    /// list and `arg_types` from the same source, so check it here rather than
    /// letting the emitter guess — `_consider_call` carries the same check
    /// for plain calls. `num_args` counts the leading value and function
    /// operands, which the descr does not describe.
    pub(crate) fn check_cond_call_value_descr_arity(op: &Op, num_args: usize) {
        if !op.opcode.is_cond_call_value() {
            return;
        }
        let descr_arc = op.getdescr().expect("cond_call_value without CallDescr");
        let calldescr = descr_arc
            .as_call_descr()
            .expect("cond_call_value without CallDescr");
        assert_eq!(calldescr.arg_types().len(), num_args - 2);
    }

    /// Advance position in both register managers.
    pub fn next_instruction(&mut self) {
        self.rm.next_instruction(1);
        self.xrm.next_instruction(1);
    }

    /// Get final frame depth.
    pub fn get_final_frame_depth(&self) -> usize {
        self.fm.get_frame_depth()
    }

    /// Drain pending moves.
    pub fn take_pending_moves(&mut self) -> Vec<(Loc, Loc)> {
        std::mem::take(&mut self.pending_moves)
    }
}

// ── Utility functions ──────────────────────────────────────────────

/// regalloc.py:1236
pub fn valid_addressing_size(size: usize) -> bool {
    size == 1 || size == 2 || size == 4 || size == 8
}

/// regalloc.py:1239
pub fn get_scale(size: usize) -> u8 {
    assert!(valid_addressing_size(size), "get_scale({size})");
    if size < 4 {
        (size - 1) as u8 // 1, 2 => 0, 1
    } else {
        ((size >> 2) + 1) as u8 // 4, 8 => 2, 3
    }
}

/// rx86.py fits_in_32bits
pub fn fits_in_32bits(value: i64) -> bool {
    value == value as i32 as i64
}

/// Compare two Loc values for equality.
fn loc_eq(a: &Loc, b: &Loc) -> bool {
    match (a, b) {
        (Loc::Reg(ra), Loc::Reg(rb)) => ra == rb,
        (Loc::Frame(fa), Loc::Frame(fb)) => fa.position == fb.position,
        (Loc::Immed(ia) | Loc::ImmedFloat(ia), Loc::Immed(ib) | Loc::ImmedFloat(ib)) => {
            ia.value == ib.value
        }
        (Loc::ConstFloat(a), Loc::ConstFloat(b)) => a.value == b.value,
        _ => false,
    }
}

// ── Tests ──────────────────────────────────────────────────────────

/// Test-only: rebind every arg/failarg box to this trace's own
/// `InputArgRc`/`OpRc` producers (`runner_test.py` builds args from the
/// producer box object).
#[cfg(test)]
pub(crate) fn rebind_trace_arg_boxes(inputargs: &[InputArgRc], operations: &[OpRc]) {
    let mut map = IndexMap::<OpRef, Operand>::new();
    for ia in inputargs {
        map.insert(ia.opref(), Operand::from_bound_inputarg(ia));
    }
    for op in operations {
        if op.opcode.result_type() != Type::Void {
            map.insert(op.pos().get(), Operand::from_bound_op(op));
        }
    }
    for op in operations {
        for slot in op.args_slice_mut() {
            if slot.is_constant() || slot.is_none() {
                continue;
            }
            if let Some(mapped) = map.get(&slot.to_opref()) {
                *slot = mapped.clone();
            }
        }
        op.map_failargs_in_place(|slot| {
            if slot.is_constant() || slot.is_none() {
                return;
            }
            if let Some(mapped) = map.get(&slot.to_opref()) {
                *slot = mapped.clone();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use majit_ir::operand::Operand;
    use majit_ir::{InputArg, Op, OpCode, OpRc, OpRef, Type};

    use majit_ir::forwarding::bound_operand_from_opref as rb;

    fn ia_pair(tp: Type, index: u32) -> (InputArgRc, Operand) {
        let ia = InputArg::from_type_rc(tp, index);
        let box_ = Operand::from_bound_inputarg(&ia);
        (ia, box_)
    }

    fn rcs(inputargs: &[InputArgRc], ops: Vec<Op>) -> Vec<OpRc> {
        let ops: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
        rebind_trace_arg_boxes(inputargs, &ops);
        ops
    }

    fn make_op(opcode: OpCode, pos: u32, args: &[OpRef]) -> Op {
        let bx: Vec<Operand> = args.iter().map(|a| rb(*a)).collect();
        let op = Op::new(opcode, &bx);
        op.pos().set(OpRef::int_op(pos));
        op
    }

    fn make_guard(opcode: OpCode, pos: u32, args: &[OpRef], fail_args: &[OpRef]) -> Op {
        let op = make_op(opcode, pos, args);
        op.setfailargs(fail_args.iter().map(|a| rb(*a)).collect());
        op
    }

    /// llsupport/test/test_regalloc.py `MockAsm`: records what the walk
    /// hands the assembler, in order.
    #[derive(Debug)]
    enum Emitted {
        Move {
            src: Loc,
            dst: Loc,
        },
        Perform {
            op_index: usize,
            arglocs: Vec<Loc>,
            result_loc: Option<Loc>,
        },
        PerformDiscard {
            op_index: usize,
            arglocs: Vec<Loc>,
        },
        PerformGuard {
            op_index: usize,
            arglocs: Vec<Loc>,
            faillocs: Vec<Option<Loc>>,
        },
        #[cfg(target_arch = "x86_64")]
        Lea {
            dst: Loc,
            index_loc: Loc,
        },
    }

    #[derive(Default)]
    struct MockAsm {
        emitted: Vec<Emitted>,
    }

    impl RegallocMoves for MockAsm {
        fn regalloc_mov(&mut self, src: &Loc, dst: &Loc) {
            self.emitted.push(Emitted::Move {
                src: *src,
                dst: *dst,
            });
        }
        fn regalloc_push(&mut self, loc: &Loc) {
            unreachable!("the walk does not push {loc:?}")
        }
        fn regalloc_pop(&mut self, loc: &Loc) {
            unreachable!("the walk does not pop {loc:?}")
        }
        fn _move(&mut self, src: &Loc, dst: &Loc, _tmpreg: &Loc) {
            self.regalloc_mov(src, dst);
        }
    }

    impl RegAllocAssembler for MockAsm {
        fn regalloc_perform(
            &mut self,
            _regalloc: &RegAlloc<'_>,
            op_index: usize,
            arglocs: &[Loc],
            result_loc: Option<Loc>,
            _gcmap: Option<usize>,
        ) {
            self.emitted.push(Emitted::Perform {
                op_index,
                arglocs: arglocs.to_vec(),
                result_loc,
            });
        }
        fn regalloc_perform_discard(
            &mut self,
            _regalloc: &RegAlloc<'_>,
            op_index: usize,
            arglocs: &[Loc],
        ) {
            self.emitted.push(Emitted::PerformDiscard {
                op_index,
                arglocs: arglocs.to_vec(),
            });
        }
        fn regalloc_perform_guard(
            &mut self,
            _regalloc: &RegAlloc<'_>,
            op_index: usize,
            faillocs: &[Option<Loc>],
            arglocs: &[Loc],
            _result_loc: Option<Loc>,
            _frame_depth: usize,
        ) {
            self.emitted.push(Emitted::PerformGuard {
                op_index,
                arglocs: arglocs.to_vec(),
                faillocs: faillocs.to_vec(),
            });
        }
        #[cfg(target_arch = "x86_64")]
        fn regalloc_lea_scaled_addr(
            &mut self,
            dst: Loc,
            index_loc: &Loc,
            _itemsize: i64,
            _base_loc: &Loc,
            _ofs: i64,
        ) {
            self.emitted.push(Emitted::Lea {
                dst,
                index_loc: *index_loc,
            });
        }
    }

    fn walk(ra: &mut RegAlloc<'_>) -> Vec<Emitted> {
        let mut asm = MockAsm::default();
        ra.walk_operations(&mut asm).expect("walk_operations");
        asm.emitted
    }

    fn has_move_from(emitted: &[Emitted], expected: &Loc) -> bool {
        emitted.iter().any(|e| {
            if let Emitted::Move { src, .. } = e {
                loc_eq(src, expected)
            } else {
                false
            }
        })
    }

    fn guard_faillocs(emitted: &[Emitted], index: usize) -> Option<&[Option<Loc>]> {
        emitted.iter().find_map(|e| match e {
            Emitted::PerformGuard {
                op_index, faillocs, ..
            } if *op_index == index => Some(&faillocs[..]),
            _ => None,
        })
    }

    fn perform_arglocs(emitted: &[Emitted], index: usize) -> Option<&[Loc]> {
        emitted.iter().find_map(|e| match e {
            Emitted::Perform {
                op_index, arglocs, ..
            } if *op_index == index => Some(&arglocs[..]),
            _ => None,
        })
    }

    #[test]
    fn test_compute_vars_longevity_simple() {
        // i0 = inputarg
        // i1 = int_add(i0, i0)  [pos 0]
        // guard_true(i1)         [pos 1, fail_args=[i0]]
        // jump(i1)               [pos 2]
        let inputargs = vec![InputArg::from_type_rc(Type::Int, 100)];
        let i0 = Operand::from_bound_inputarg(&inputargs[0]);
        let add = Op::new(OpCode::IntAdd, &[i0.clone(), i0.clone()]);
        add.pos().set(OpRef::int_op(0));
        let add_rc = OpRc::new(add);
        let i1 = Operand::from_bound_op(&add_rc);
        let guard = Op::new(OpCode::GuardTrue, &[i1.clone()]);
        guard.pos().set(OpRef::int_op(1));
        guard.setfailargs(vec![i0.clone()].into());
        let jump = Op::new(OpCode::Jump, &[i1.clone()]);
        jump.pos().set(OpRef::int_op(2));
        let ops = vec![add_rc, OpRc::new(guard), OpRc::new(jump)];

        let longevity = compute_vars_longevity(&inputargs, &ops);

        // i0: defined at -1 (input), last used at 1 (guard failarg), real usages: [0, 0]
        // (used twice as arg to int_add at position 0)
        let lt_i0 = longevity.get(&i0).unwrap();
        assert_eq!(lt_i0.last_usage.get(), 1);
        assert_eq!(
            lt_i0.real_usages.borrow().as_ref().unwrap().as_slice(),
            &[0, 0]
        );

        // i1: defined at 0, last used at 2 (jump), real usages: [1]
        let lt_i1 = longevity.get(&i1).unwrap();
        assert_eq!(lt_i1.definition_pos.get(), 0);
        assert_eq!(lt_i1.last_usage.get(), 2);
        assert_eq!(
            lt_i1.real_usages.borrow().as_ref().unwrap().as_slice(),
            &[1]
        );
    }

    #[test]
    fn ref_inputarg_longevity_follows_failargs() {
        let inputargs = vec![
            InputArg::from_type_rc(Type::Int, 0),
            InputArg::from_type_rc(Type::Ref, 1),
        ];
        let i0 = Operand::from_bound_inputarg(&inputargs[0]);
        let vm = Operand::from_bound_inputarg(&inputargs[1]);
        let add_op = Op::new(OpCode::IntAdd, &[i0.clone(), i0.clone()]);
        add_op.pos().set(OpRef::int_op(0));
        let add_rc = OpRc::new(add_op);
        let add = Operand::from_bound_op(&add_rc);
        let guard = Op::new(OpCode::GuardTrue, &[add.clone()]);
        guard.pos().set(OpRef::int_op(1));
        guard.setfailargs(vec![vm.clone()].into());
        let jump = Op::new(OpCode::Jump, &[add.clone()]);
        jump.pos().set(OpRef::int_op(2));
        let ops = vec![add_rc, OpRc::new(guard), OpRc::new(jump)];
        let longevity = compute_vars_longevity(&inputargs, &ops);
        assert_eq!(
            longevity.get(&vm).unwrap().last_usage.get(),
            1,
            "compute_vars_longevity: last_usage is the last failarg/arg use"
        );
    }

    #[test]
    fn shared_register_keeps_both_boxes_findable() {
        let regs = all_core_regs();
        let mut rm = RegisterManager::new(
            regs.clone(),
            vec![],
            regs.clone(),
            frame_reg(),
            None,
            call_result_gpr(),
        );
        let (_ia0, first) = ia_pair(Type::Ref, 1218);
        let (_ia1, second) = ia_pair(Type::Ref, 1220);
        let mut longevity = LifetimeManager::new();
        longevity.set(&first, Lifetime::new(-1, 1));
        longevity.set(&second, Lifetime::new(-1, 1));
        let shared = regs[regs.len() - 1];
        rm.reg_bindings_set(&first, shared, &mut longevity);
        rm.reg_bindings_set(&second, shared, &mut longevity);
        assert_eq!(rm.reg_bindings_get(&first, &longevity), Some(shared));
        assert_eq!(rm.reg_bindings_get(&second, &longevity), Some(shared));
    }

    #[test]
    fn test_lifetime_next_real_usage() {
        let lt = Lifetime::new(0, 10);
        *lt.real_usages.borrow_mut() = Some(SmallVec::from_slice(&[2, 5, 8]));

        assert_eq!(lt.next_real_usage(0), 2);
        assert_eq!(lt.next_real_usage(1), 2);
        assert_eq!(lt.next_real_usage(2), 5);
        assert_eq!(lt.next_real_usage(4), 5);
        assert_eq!(lt.next_real_usage(5), 8);
        assert_eq!(lt.next_real_usage(8), -1);
    }

    #[test]
    fn test_frame_manager_allocate() {
        let mut longevity = LifetimeManager::new();
        let (_ia0, v0) = ia_pair(Type::Int, 0);
        let (_ia1, v1) = ia_pair(Type::Int, 1);
        let (_ia2, v2) = ia_pair(Type::Int, 2);
        longevity.set(&v0, Lifetime::new(0, 10));
        longevity.set(&v1, Lifetime::new(1, 10));
        longevity.set(&v2, Lifetime::new(2, 10));

        let mut fm = FrameManager::new(0);
        let loc0 = fm.get_new_loc(&v0, Type::Int, &mut longevity);
        let loc1 = fm.get_new_loc(&v1, Type::Int, &mut longevity);
        let loc2 = fm.get_new_loc(&v2, Type::Int, &mut longevity);

        assert_eq!(loc0.position, 0);
        assert_eq!(loc1.position, 1);
        assert_eq!(loc2.position, 2);
        assert_eq!(fm.get_frame_depth(), 3);
    }

    #[test]
    fn test_register_manager_allocate() {
        let mut longevity = LifetimeManager::new();
        let (_ia0, v0) = ia_pair(Type::Int, 0);
        let (_ia1, v1) = ia_pair(Type::Int, 1);
        longevity.set(&v0, Lifetime::new(0, 5));
        longevity.set(&v1, Lifetime::new(1, 5));

        let mut rm = RegisterManager::new(
            vec![EAX, ECX, EDX],
            vec![],
            vec![EAX, ECX, EDX],
            EBP,
            Some(vec![Type::Int, Type::Ref]),
            EAX,
        );
        rm.position = 0;

        let reg0 = rm.try_allocate_reg(&v0, None, false, &mut longevity);
        assert!(reg0.is_some());
        let reg0 = reg0.unwrap();

        let reg1 = rm.try_allocate_reg(&v1, None, false, &mut longevity);
        assert!(reg1.is_some());
        let reg1 = reg1.unwrap();

        assert_ne!(reg0, reg1);
        assert_eq!(rm.reg_bindings_get(&v0, &longevity), Some(reg0));
        assert_eq!(rm.reg_bindings_get(&v1, &longevity), Some(reg1));
    }

    #[test]
    fn test_register_spill() {
        let mut longevity = LifetimeManager::new();
        let lt0 = Lifetime::new(0, 10);
        *lt0.real_usages.borrow_mut() = Some(SmallVec::from_slice(&[5]));
        let lt1 = Lifetime::new(1, 10);
        *lt1.real_usages.borrow_mut() = Some(SmallVec::from_slice(&[6]));
        let lt2 = Lifetime::new(2, 10);
        *lt2.real_usages.borrow_mut() = Some(SmallVec::from_slice(&[7]));
        let (_ia0, v0) = ia_pair(Type::Int, 0);
        let (_ia1, v1) = ia_pair(Type::Int, 1);
        let (_ia2, v2) = ia_pair(Type::Int, 2);
        longevity.set(&v0, lt0);
        longevity.set(&v1, lt1);
        longevity.set(&v2, lt2);

        // Only 2 registers available
        let mut rm = RegisterManager::new(vec![EAX, ECX], vec![], vec![EAX, ECX], EBP, None, EAX);
        rm.position = 3;
        let mut fm = FrameManager::new(0);

        let reg0 = rm.force_allocate_reg(&v0, &[], None, false, &mut longevity, &mut fm);
        let reg1 = rm.force_allocate_reg(&v1, &[], None, false, &mut longevity, &mut fm);
        assert!(reg0 != reg1);

        // Third allocation forces a spill
        let reg2 = rm.force_allocate_reg(&v2, &[], None, false, &mut longevity, &mut fm);
        assert!(reg2 == reg0 || reg2 == reg1);
        // The spilled variable should now be in the frame
        assert!(fm.get_frame_depth() >= 1);
    }

    #[test]
    fn test_fixed_nursery_malloc_keeps_unrelated_live_register_bound() {
        // aarch64/regalloc.py prepare_op_call_malloc_nursery and x86/regalloc.py consider_call_malloc_nursery move
        // values away from the two malloc-cond registers only.  In
        // particular, a live Ref in any other managed register stays there;
        // the collecting slow path saves and restores it through the
        // jitframe.  SAVE_ALL_REGS at this call site used to turn every such
        // binding into an eager spill in the allocation fast path.
        let live_ref = OpRef::input_arg_ref(0);
        let malloc_result = OpRef::ref_op(0);
        let size = OpRef::const_int(16);
        let inputargs = vec![InputArg::from_type_rc(Type::Ref, live_ref.raw())];

        let malloc = Op::new(OpCode::CallMallocNursery, &[rb(size)]);
        malloc.pos().set(malloc_result);
        let finish = Op::new(OpCode::Finish, &[rb(live_ref), rb(malloc_result)]);
        finish.pos().set(OpRef::int_op(1));
        finish.setfailargs(vec![].into());
        finish.set_fail_arg_types(vec![Type::Ref, Type::Ref]);
        let ops = vec![malloc, finish];

        let ops = rcs(&inputargs, ops);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let unrelated = all_core_regs()
            .into_iter()
            .find(|reg| !MALLOC_NURSERY_CLOBBER.contains(reg))
            .expect("architecture must expose a managed register outside malloc clobbers");
        let live_box = Operand::from_bound_inputarg(&inputargs[0]);
        ra.rm.force_allocate_reg(
            &live_box,
            &[],
            Some(unrelated),
            false,
            &mut ra.longevity,
            &mut ra.fm,
        );
        ra.pending_moves.clear();
        ra.rm.spill_moves.clear();

        let mut asm = MockAsm::default();
        ra._dispatch_op(&ops[0], 0, &mut asm);
        let output = asm.emitted;

        assert_eq!(
            ra.rm.reg_bindings_get(&live_box, &ra.longevity),
            Some(unrelated),
            "fixed nursery allocation must preserve unrelated live bindings"
        );
        assert!(
            !has_move_from(&output, &Loc::Reg(unrelated)),
            "fixed nursery allocation must not spill an unrelated live register"
        );
    }

    #[test]
    fn loc_size() {
        assert!(std::mem::size_of::<Loc>() <= 16);
    }

    /// x86/regalloc.py `consider_zero_array`: the destination `LEA` is
    /// emitted right after `dstaddr_loc` is allocated, before the length is
    /// loaded and before `before_call`, so a later load or spill cannot
    /// overwrite a register the address still reads.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_zero_array_lea_precedes_length_load() {
        let base = OpRef::input_arg_ref(0);
        let start = OpRef::input_arg_int(1);
        let length = OpRef::input_arg_int(2);
        let inputargs = vec![
            InputArg::from_type_rc(Type::Ref, base.raw()),
            InputArg::from_type_rc(Type::Int, start.raw()),
            InputArg::from_type_rc(Type::Int, length.raw()),
        ];
        let zero = Op::new(
            OpCode::ZeroArray,
            &[
                rb(base),
                rb(start),
                rb(length),
                rb(OpRef::const_int(8)),
                rb(OpRef::const_int(8)),
            ],
        );
        zero.pos().set(OpRef::void_op(3));
        zero.setdescr(std::sync::Arc::new(majit_ir::descr::SimpleArrayDescr::new(
            0,
            16,
            8,
            1,
            Type::Int,
        )));
        let finish = Op::new(OpCode::Finish, &[]);
        finish.pos().set(OpRef::void_op(4));
        finish.setfailargs(vec![].into());
        finish.set_fail_arg_types(vec![]);

        let ops = rcs(&inputargs, vec![zero, finish]);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let emitted = walk(&mut ra);
        let leas: Vec<usize> = emitted
            .iter()
            .enumerate()
            .filter_map(|(k, e)| matches!(e, Emitted::Lea { .. }).then_some(k))
            .collect();
        let discard = emitted
            .iter()
            .position(|e| matches!(e, Emitted::PerformDiscard { op_index: 0, .. }))
            .expect("ZERO_ARRAY is a discard op");
        assert_eq!(
            leas.len(),
            2,
            "destination and byte-count LEAs: {emitted:#?}"
        );
        assert!(leas[1] < discard);
        let Emitted::PerformDiscard { arglocs, .. } = &emitted[discard] else {
            unreachable!()
        };
        let (
            Emitted::Lea { dst: dstaddr, .. },
            Emitted::Lea {
                dst: bytes,
                index_loc,
                ..
            },
        ) = (&emitted[leas[0]], &emitted[leas[1]])
        else {
            unreachable!()
        };
        assert!(
            arglocs.len() == 2 && loc_eq(&arglocs[0], dstaddr) && loc_eq(&arglocs[1], bytes),
            "memset takes the LEA results: {arglocs:?}"
        );
        assert!(
            !loc_eq(index_loc, dstaddr),
            "the length must not sit in dstaddr_loc"
        );
        // The length is loaded after the destination LEA, not before it.
        let load = emitted
            .iter()
            .position(|e| matches!(e, Emitted::Move { dst, .. } if loc_eq(dst, index_loc)))
            .unwrap_or_else(|| panic!("no move loads the length: {emitted:#?}"));
        assert!(leas[0] < load && load < leas[1], "{emitted:#?}");
    }

    #[test]
    fn test_perform_two_locs() {
        // x86/regalloc.py perform(op, [loc, argloc], res) — a 2-element list.
        let i0 = OpRef::input_arg_int(0);
        let i1 = OpRef::input_arg_int(1);
        let i2 = OpRef::int_op(2);
        let i3 = OpRef::int_op(3);
        let inputargs = vec![
            InputArg::from_type_rc(Type::Int, i0.raw()),
            InputArg::from_type_rc(Type::Int, i1.raw()),
        ];
        let add1 = Op::new(OpCode::IntAdd, &[rb(i0), rb(i1)]);
        add1.pos().set(i2);
        let add2 = Op::new(OpCode::IntAdd, &[rb(i2), rb(i1)]);
        add2.pos().set(i3);
        let finish = Op::new(OpCode::Finish, &[rb(i3)]);
        finish.pos().set(OpRef::void_op(4));
        finish.setfailargs(vec![].into());
        finish.set_fail_arg_types(vec![]);

        let ops = rcs(&inputargs, vec![add1, add2, finish]);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let emitted = walk(&mut ra);
        for index in [0, 1] {
            let arglocs = perform_arglocs(&emitted, index).expect("IntAdd perform");
            assert_eq!(arglocs.len(), 2, "IntAdd is a 2-loc perform: {emitted:?}");
        }
    }

    #[test]
    fn test_j2_deopt_only_failarg_kept_in_register_at_guard() {
        // A deopt-only fail arg (dead on the fast path after the guard) is
        // captured from its register at the guard, not eagerly spilled to a
        // frame slot: the failure path saves every register before rebuilding
        // the frame, so a register faillocs is fully recoverable, and the
        // fast path pays no spill store.
        //
        // i0 is the typed inputarg slot 0 (Int) that the regalloc
        // registers from `inputargs.opref()`. i1/i2 are op result
        // positions; their variant tag is unconstrained, so plain
        // `from_raw` is fine.
        let i0 = OpRef::input_arg_int(0);
        let i1 = OpRef::int_op(1);
        let i2 = OpRef::int_op(2);
        let c1 = OpRef::const_int(1);

        let inputargs = vec![InputArg::from_type_rc(Type::Int, i0.raw())];

        let add = Op::new(OpCode::IntAdd, &[rb(i0), rb(c1)]);
        add.pos().set(i1);
        let is_true = Op::new(OpCode::IntIsTrue, &[rb(i1)]);
        is_true.pos().set(i2);
        let guard = Op::new(OpCode::GuardTrue, &[rb(i2)]);
        guard.pos().set(OpRef::int_op(3));
        guard.setfailargs(vec![rb(i1)].into());
        let finish = Op::new(OpCode::Finish, &[]);
        finish.pos().set(OpRef::int_op(4));
        finish.setfailargs(vec![].into());
        finish.set_fail_arg_types(vec![]);

        let constants: indexmap::IndexMap<u32, i64> = indexmap::IndexMap::new();

        let ops = vec![add, is_true, guard, finish];
        let ops = rcs(&inputargs, ops);
        let mut ra = RegAlloc::new(constants, &inputargs, &ops);
        ra.prepare_loop();
        let emitted = walk(&mut ra);
        let Some(faillocs) = guard_faillocs(&emitted, 2) else {
            panic!("guard op was not lowered through regalloc_perform_guard");
        };
        assert!(
            matches!(faillocs, [Some(Loc::Reg(_))]),
            "deopt-only failarg should be captured from a register: {:?}",
            faillocs
        );
    }

    /// Build `[IntMulOvf -> i2, <optional interposed IntGt>, GuardNoOverflow,
    /// Finish(i2)]` and walk it. The `Finish` use keeps the OVF result live so
    /// the dead-op skip does not swallow the OVF op before the adjacency check.
    fn walk_ovf_trace(interpose_comparison: bool) {
        let i0 = OpRef::input_arg_int(0);
        let i1 = OpRef::input_arg_int(1);
        let i2 = OpRef::int_op(2);
        let i3 = OpRef::int_op(3);

        let inputargs = vec![
            InputArg::from_type_rc(Type::Int, i0.raw()),
            InputArg::from_type_rc(Type::Int, i1.raw()),
        ];

        let mul_ovf = Op::new(OpCode::IntMulOvf, &[rb(i0), rb(i1)]);
        mul_ovf.pos().set(i2);
        let guard = Op::new(OpCode::GuardNoOverflow, &[]);
        guard.pos().set(OpRef::int_op(4));
        guard.setfailargs(vec![rb(i2)].into());
        let finish = Op::new(OpCode::Finish, &[rb(i2)]);
        finish.pos().set(OpRef::int_op(5));
        finish.setfailargs(vec![].into());
        finish.set_fail_arg_types(vec![]);

        let mut ops = vec![mul_ovf];
        if interpose_comparison {
            let gt = Op::new(OpCode::IntGt, &[rb(i0), rb(i1)]);
            gt.pos().set(i3);
            ops.push(gt);
        }
        ops.push(guard);
        ops.push(finish);

        let ops = rcs(&inputargs, ops);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        walk(&mut ra);
    }

    /// aarch64/assembler.py:1191-1195: the OVF op's answer lives in NZCV, so
    /// its guard must be `operations[i + 1]`. Adjacent is accepted.
    #[test]
    fn test_ovf_op_adjacent_to_its_guard_walks() {
        walk_ovf_trace(false);
    }

    /// ...and an op emitted between them — which overwrites NZCV — is rejected
    /// rather than silently compiled into a guard on the wrong comparison.
    #[test]
    #[should_panic(expected = "is not followed by GUARD_[NO_]OVERFLOW")]
    fn test_ovf_op_separated_from_its_guard_is_rejected() {
        walk_ovf_trace(true);
    }

    /// `prepare_op_call_malloc_nursery_varsize` reads `lengthloc` only after
    /// `spill_or_move_registers_before_call(SAVE_ALL_REGS)` has popped every
    /// register binding, so a length that outlives the allocation reaches the
    /// assembler in its frame slot, not in a register. Upstream says the same
    /// out loud — x86/assembler.py:2617-2621 loads a non-`RegLoc` `lengthloc`
    /// with `MOV(edx, lengthloc)`, and aarch64/assembler.py:745-751 asserts
    /// `lengthloc.is_stack()` before `regalloc_mov`.
    ///
    /// Both backends carry an arm for this shape. Neither may substitute an
    /// immediate 0, which sizes the block for no items while the
    /// `gen_initialize_len` store that follows stamps the real length into it.
    #[test]
    fn call_malloc_nursery_varsize_hands_over_a_frame_resident_length() {
        let (ia, length) = ia_pair(Type::Int, 0);
        let inputargs = vec![ia];

        // arglocs = [lengthloc, imm(itemsize), imm(kind)] is built from
        // args [kind, itemsize, length].
        let malloc = Op::new(
            OpCode::CallMallocNurseryVarsize,
            &[
                rb(OpRef::const_int(0)),
                rb(OpRef::const_int(8)),
                length.clone(),
            ],
        );
        malloc.pos().set(OpRef::ref_op(1));
        // The length outlives the allocation, which is the case upstream calls
        // typical: "it's typically also present in the next operation that will
        // copy it inside the new array".
        let finish = Op::new(OpCode::Finish, &[length]);
        finish.pos().set(OpRef::void_op(2));
        let ops = vec![malloc, finish];

        let ops = rcs(&inputargs, ops);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let emitted = walk(&mut ra);
        let lengthloc = perform_arglocs(&emitted, 0)
            .and_then(|arglocs| arglocs.first().copied())
            .expect("the varsize allocation reached the assembler");
        assert!(
            matches!(lengthloc, Loc::Frame(_) | Loc::Ebp(_)),
            "the length is spilled before it is read, so the assembler needs a \
             memory arm for it; got {lengthloc:?}"
        );
    }

    /// x86/regalloc.py `perform_guard` frees `getfailargs()` before
    /// `walk_operations` reaches `possibly_free_vars_for_op(op)`, so a
    /// failarg's register goes back to `free_regs` ahead of the guard
    /// argument's, and the argument's register is picked first afterwards.
    #[test]
    fn guard_frees_failargs_before_args() {
        let i0 = OpRef::input_arg_int(0);
        let i1 = OpRef::input_arg_int(1);
        let i2 = OpRef::int_op(0);
        let i3 = OpRef::int_op(1);
        let inputargs = vec![InputArg::new_int_rc(0), InputArg::new_int_rc(1)];
        let finish = Op::new(OpCode::Finish, &[]);
        finish.pos().set(OpRef::void_op(3));
        let ops = rcs(
            &inputargs,
            vec![
                make_op(OpCode::IntAdd, 0, &[i0, OpRef::const_int(1)]),
                make_op(OpCode::IntAdd, 1, &[i1, OpRef::const_int(1)]),
                make_guard(OpCode::GuardTrue, 2, &[i2], &[i3]),
                finish,
            ],
        );
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let output = walk(&mut ra);
        let (arg_loc, fail_loc) = output
            .iter()
            .find_map(|e| match e {
                Emitted::PerformGuard {
                    arglocs, faillocs, ..
                } => Some((arglocs[0], faillocs[0])),
                _ => None,
            })
            .expect("GUARD_TRUE must reach regalloc_perform_guard");
        let (Loc::Reg(arg_reg), Some(Loc::Reg(fail_reg))) = (arg_loc, fail_loc) else {
            panic!("both guard operands must be in registers: {arg_loc:?} {fail_loc:?}");
        };
        let pos = |r: RegLoc| ra.rm.free_regs.iter().position(|&f| f == r).unwrap();
        assert!(
            pos(fail_reg) < pos(arg_reg),
            "failarg {fail_reg:?} must be freed before arg {arg_reg:?}: {:?}",
            ra.rm.free_regs
        );
    }

    /// x86/regalloc.py `consider_label`: a LABEL argument located in ebp
    /// (a bridge input bound by `_update_bindings`) is stored to its frame
    /// slot, `box_currently_in_frame_reg` is reset, and the label's
    /// location is that slot.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn label_moves_ebp_argument_to_its_frame_slot() {
        let i0 = OpRef::input_arg_int(0);
        let i1 = OpRef::int_op(1);
        let inputargs = vec![InputArg::new_int_rc(0)];
        let label = Op::new(OpCode::Label, &[rb(i0)]);
        label.pos().set(OpRef::void_op(0));
        let ops = rcs(
            &inputargs,
            vec![
                label,
                make_op(OpCode::IntAdd, 1, &[i0, OpRef::const_int(1)]),
                make_op(OpCode::Finish, 2, &[i1]),
            ],
        );
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_bridge(&[Loc::Reg(EBP)]);
        assert_eq!(
            ra.rm.box_currently_in_frame_reg,
            Some(Operand::from_bound_inputarg(&inputargs[0]))
        );
        let output = walk(&mut ra);
        let label_pos = output
            .iter()
            .position(|e| matches!(e, Emitted::Perform { op_index: 0, .. }))
            .expect("LABEL perform");
        let Emitted::Perform { arglocs, .. } = &output[label_pos] else {
            unreachable!()
        };
        let label_loc = arglocs[0];
        let Loc::Frame(slot) = label_loc else {
            panic!("LABEL argument must be its frame slot, got {label_loc:?}");
        };
        assert!(
            output[..label_pos].iter().any(|op| matches!(
                op,
                Emitted::Move { src: Loc::Reg(r), dst: Loc::Frame(f) }
                    if *r == EBP && f.position == slot.position
            )),
            "missing MOV [frame slot], ebp: {output:?}"
        );
        assert_eq!(ra.rm.box_currently_in_frame_reg, None);
    }

    /// llsupport/regalloc.py `locs_for_call_assembler`: the virtualizable
    /// argument is synced to the stack and passed as `self.fm.loc(arg1)`,
    /// even while it is still in a register.
    #[test]
    fn call_assembler_vable_loc_is_its_frame_slot() {
        let p0 = OpRef::input_arg_ref(0);
        let p1 = OpRef::input_arg_ref(1);
        let p2 = OpRef::ref_op(0);
        let inputargs = vec![InputArg::new_ref_rc(0), InputArg::new_ref_rc(1)];
        let same = Op::new(OpCode::SameAsR, &[rb(p1)]);
        same.pos().set(p2);
        let call = Op::new(OpCode::CallAssemblerN, &[rb(p0), rb(p2)]);
        call.pos().set(OpRef::void_op(1));
        let finish = Op::new(OpCode::Finish, &[rb(p2)]);
        finish.pos().set(OpRef::void_op(2));
        let ops = rcs(&inputargs, vec![same, call, finish]);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let output = walk(&mut ra);
        let arglocs = perform_arglocs(&output, 1).expect("CALL_ASSEMBLER perform");
        assert!(
            matches!(arglocs[1], Loc::Frame(_)),
            "vable must be passed as its frame slot: {arglocs:?}"
        );
    }

    /// `consider_check_memory_error` is `perform_discard(op, [x])` with
    /// `x` in a register.
    #[test]
    fn check_memory_error_is_a_discard() {
        let p0 = OpRef::input_arg_ref(0);
        let inputargs = vec![InputArg::new_ref_rc(0)];
        let check = Op::new(OpCode::CheckMemoryError, &[rb(p0)]);
        check.pos().set(OpRef::void_op(0));
        let finish = Op::new(OpCode::Finish, &[rb(p0)]);
        finish.pos().set(OpRef::void_op(1));
        let ops = rcs(&inputargs, vec![check, finish]);
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let output = walk(&mut ra);
        let arglocs = output
            .iter()
            .find_map(|e| match e {
                Emitted::PerformDiscard {
                    op_index: 0,
                    arglocs,
                } => Some(arglocs.clone()),
                _ => None,
            })
            .expect("CHECK_MEMORY_ERROR perform_discard");
        assert!(
            matches!(arglocs[..], [Loc::Reg(r)] if !r.is_xmm),
            "one GPR argloc: {arglocs:?}"
        );
    }

    #[test]
    fn test_get_scale() {
        assert_eq!(get_scale(1), 0);
        assert_eq!(get_scale(2), 1);
        assert_eq!(get_scale(4), 2);
        assert_eq!(get_scale(8), 3);
    }

    #[test]
    fn test_guard_not_forced_2_emits_fail_locs_without_frame_depth_argloc() {
        // aarch64/regalloc.py prepare_op_guard_not_forced_2 spills the
        // failargs and returns only the guard fail locations. The guard does
        // NOT carry a frame-depth argloc: check_frame_depth (assembler.py:927)
        // is a bridge-prologue stack check whose expected size is patched
        // after assembly (emit_check_frame_depth + patch_stack_checks), not a
        // value threaded through the guard's arglocs.
        let force = Op::new(OpCode::ForceToken, &[]);
        force.pos().set(OpRef::ref_op(0));
        let force_rc = OpRc::new(force);
        let token = Operand::from_bound_op(&force_rc);
        let guard = Op::new(OpCode::GuardNotForced2, &[]);
        guard.pos().set(OpRef::int_op(1));
        guard.setfailargs(vec![token].into());
        let inputargs = vec![];
        let ops = vec![force_rc, OpRc::new(guard)];
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let output = walk(&mut ra);
        // FORCE_TOKEN produces a Ref in a register, with no pre-existing
        // inputarg frame home. SAVE_ALL_REGS must emit a real spill for it.
        let Some(Emitted::Perform {
            result_loc: Some(Loc::Reg(register)),
            ..
        }) = output.first()
        else {
            panic!("FORCE_TOKEN must produce a register-resident Ref: {output:?}")
        };
        assert!(
            has_move_from(&output, &Loc::Reg(*register)),
            "missing force spill"
        );

        let Some(Emitted::PerformGuard {
            arglocs, faillocs, ..
        }) = output
            .iter()
            .find(|e| matches!(e, Emitted::PerformGuard { .. }))
        else {
            panic!("expected regalloc_perform_guard: {output:?}")
        };
        assert_eq!(faillocs.len(), 1);
        assert!(
            matches!(faillocs[0], Some(Loc::Frame(_))),
            "GuardNotForced2 must spill every failarg into the jitframe, got {faillocs:?}"
        );
        assert!(
            !arglocs
                .iter()
                .any(|l| matches!(l, Loc::Immed(_) | Loc::ImmedFloat(_))),
            "GuardNotForced2 must not carry a frame-depth immediate argloc, got {arglocs:?}"
        );
    }

    /// `X86XMMRegisterManager.convert_to_imm` parks the constant with
    /// `MachineDataBlockWrapper.malloc_aligned` and hands that address to the
    /// walk. `open_malloc` raises `MemoryError`; `walk_operations` returns it
    /// as `BackendError::CompilationFailed`.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn xmm_convert_to_imm_address_and_malloc_failure() {
        fn float_add_trace() -> (Vec<majit_ir::InputArgRc>, Vec<OpRc>) {
            let inputargs = vec![InputArg::new_float_rc(0)];
            let add = Op::new(
                OpCode::FloatAdd,
                &[rb(OpRef::input_arg_float(0)), rb(OpRef::const_float(1.5))],
            );
            add.pos().set(OpRef::float_op(1));
            let finish = Op::new(OpCode::Finish, &[rb(OpRef::float_op(1))]);
            finish.pos().set(OpRef::void_op(2));
            finish.setfailargs(vec![].into());
            finish.set_fail_arg_types(vec![Type::Float]);
            let ops = rcs(&inputargs, vec![add, finish]);
            (inputargs, ops)
        }

        fn same_as_float_trace() -> (Vec<majit_ir::InputArgRc>, Vec<OpRc>) {
            let same = Op::new(OpCode::SameAsF, &[rb(OpRef::const_float(1.5))]);
            same.pos().set(OpRef::float_op(1));
            let finish = Op::new(OpCode::Finish, &[rb(OpRef::float_op(1))]);
            finish.pos().set(OpRef::void_op(2));
            finish.setfailargs(vec![].into());
            finish.set_fail_arg_types(vec![Type::Float]);
            (vec![], rcs(&[], vec![same, finish]))
        }

        fn emitted_locs(emitted: &[Emitted]) -> Vec<Loc> {
            let mut locs = Vec::new();
            for e in emitted {
                match e {
                    Emitted::Move { src, dst } => locs.extend([*src, *dst]),
                    Emitted::Perform {
                        arglocs,
                        result_loc,
                        ..
                    } => {
                        locs.extend(arglocs.iter().copied());
                        locs.extend(*result_loc);
                    }
                    _ => {}
                }
            }
            locs
        }

        fn const_float_addrs(emitted: &[Emitted]) -> Vec<usize> {
            emitted_locs(emitted)
                .into_iter()
                .filter_map(|loc| match loc {
                    Loc::ConstFloat(loc) => Some(loc.value),
                    _ => None,
                })
                .collect()
        }

        fn contains_immed_float(emitted: &[Emitted]) -> bool {
            emitted_locs(emitted)
                .iter()
                .any(|loc| matches!(loc, Loc::ImmedFloat(_)))
        }

        let bits = 1.5f64.to_bits();
        let (inputargs, ops) = float_add_trace();
        let stats = std::sync::Arc::new(majit_backend::AsmMemoryManagerStats::default());
        let manager = majit_backend::AsmMemoryManager::new_isolated(std::sync::Arc::clone(&stats));
        let mut wrapper = majit_backend::MachineDataBlockWrapper::new(manager);
        let _writing = majit_backend::AssemblerWriting::enter();
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let mut asm = MockAsm::default();
        ra.walk_operations_with_datablockwrapper(&mut wrapper, &mut asm)
            .expect("walk_operations");
        let ra_ops = asm.emitted;
        let addrs = const_float_addrs(&ra_ops);
        assert!(
            !addrs.is_empty(),
            "float constant must be a ConstFloatLoc from convert_to_imm, got {ra_ops:?}"
        );
        for addr in &addrs {
            assert_ne!(*addr, 0);
            assert_eq!(*addr % 8, 0);
            let stored = unsafe { (*addr as *const u64).read() };
            assert_eq!(
                stored, bits,
                "pool slot {addr:#x} must hold the convert_to_imm bits"
            );
        }

        // `_consider_same_as` of a float constant is `self.loc` →
        // `X86XMMRegisterManager.convert_to_imm`, so the assembler mov
        // source is `ConstFloatLoc` rather than `ImmedFloat`.
        let (inputargs, ops) = same_as_float_trace();
        let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
        ra.prepare_loop();
        let mut asm = MockAsm::default();
        ra.walk_operations_with_datablockwrapper(&mut wrapper, &mut asm)
            .expect("walk_operations");
        let ra_ops = asm.emitted;
        let addrs = const_float_addrs(&ra_ops);
        assert!(
            !addrs.is_empty(),
            "SameAsF float constant must be a ConstFloatLoc, got {ra_ops:?}"
        );
        assert!(
            !contains_immed_float(&ra_ops),
            "SameAsF must not hand the assembler an ImmedFloat move, got {ra_ops:?}"
        );
        for addr in &addrs {
            assert_ne!(*addr, 0);
            assert_eq!(*addr % 8, 0);
            let stored = unsafe { (*addr as *const u64).read() };
            assert_eq!(
                stored, bits,
                "SameAsF pool slot {addr:#x} must hold the bits"
            );
        }
        drop(_writing);
        drop(wrapper);
        drop(stats);

        let (inputargs, ops) = float_add_trace();
        let stats = std::sync::Arc::new(majit_backend::AsmMemoryManagerStats::default());
        let manager = majit_backend::AsmMemoryManager::new_isolated(stats);
        let mut wrapper = majit_backend::MachineDataBlockWrapper::new(manager);
        let err = majit_backend::with_assembler_map_lock(|| {
            struct ClearCeiling;
            impl Drop for ClearCeiling {
                fn drop(&mut self) {
                    majit_gc::arm_process_memory_ceiling(0);
                }
            }
            let _clear = ClearCeiling;
            majit_gc::arm_process_memory_ceiling(1);
            let _writing = majit_backend::AssemblerWriting::enter();
            let mut ra = RegAlloc::new(indexmap::IndexMap::new(), &inputargs, &ops);
            ra.prepare_loop();
            ra.walk_operations_with_datablockwrapper(&mut wrapper, &mut MockAsm::default())
        });
        match err {
            Err(majit_backend::BackendError::CompilationFailed(msg)) => {
                assert!(
                    msg.contains("malloc_aligned"),
                    "expected malloc_aligned failure, got {msg}"
                );
            }
            other => panic!("malloc_aligned must return Err, got {other:?}"),
        }
    }
}
