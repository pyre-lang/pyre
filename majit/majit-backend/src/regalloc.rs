//! llsupport/regalloc.py: shared lifetime and frame occupancy.
//!
//! Classes:
//!   Lifetime           — regalloc.py
//!   FixedRegisterPositions — regalloc.py
//!   LifetimeManager    — regalloc.py
//!   FrameManager       — regalloc.py occupancy; arch `frame_pos` /
//!                        `frame_size` / `get_loc_index` via [`FrameLayout`]
//!
//! Functions:
//!   compute_vars_longevity — regalloc.py
use std::cell::{Cell, RefCell};
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;
use std::rc::Rc;

use indexmap::IndexMap;
use majit_ir::IndexMapExt;
use rustc_hash::FxBuildHasher;

#[cfg(debug_assertions)]
use majit_ir::forwarding::{Forwarded, ForwardingHost};
use majit_ir::operand::Operand;
use majit_ir::{InputArgRc, OpCode, OpRc, Type};
use smallvec::SmallVec;

// ── Constants ──────────────────────────────────────────────────────

/// regalloc.py UNDEF_POS
pub const UNDEF_POS: i32 = -42;

// ── Lifetime ───────────────────────────────────────────────────────

/// regalloc.py `Lifetime` — liveness information for a single variable.
/// Interior mutability so a shared `Rc<Lifetime>` (`Lifetime.share_with`)
/// can be updated without a side-table lookup.
pub struct Lifetime<R = (), F = ()> {
    /// `Lifetime.definition_pos` — operations-slice index where the variable is defined
    pub definition_pos: Cell<i32>,
    /// `Lifetime.last_usage` — operations-slice index of last use (including failargs/jump)
    pub last_usage: Cell<i32>,
    /// `Lifetime.real_usages` — *real* usages (as op argument, excluding
    /// jump/failargs). Four inline slots cover the common SSA use-count
    /// so the first `push` does not heap-grow 16 B per live box.
    pub real_usages: RefCell<Option<SmallVec<[i32; 4]>>>,
    /// `Lifetime.fixed_positions` — positions requiring specific registers
    pub fixed_positions: RefCell<Option<Vec<(i32, R)>>>,
    /// `Lifetime.share_with` — another Lifetime that wants to share a register
    pub share_with: RefCell<Option<Rc<Lifetime<R, F>>>>,
    /// `Lifetime._definition_pos_shared`
    _definition_pos_shared: Cell<i32>,
    /// `Lifetime.current_register_index` into RegisterManager.all_regs (-1 = not in reg)
    pub current_register_index: Cell<i32>,
    /// `Lifetime.current_frame_loc` — frame location where the box currently lives
    pub current_frame_loc: Cell<Option<F>>,
    /// `Lifetime.hint_frame_pos` — hinted frame location (at the end of the trace)
    pub hint_frame_pos: Cell<i32>,
}

impl<R, F> Lifetime<R, F> {
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
    pub fn fixed_register(&self, position: i32, reg: R) -> i32 {
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
    pub fn find_fixed_register(&self, opindex: i32) -> Option<R>
    where
        R: Copy,
    {
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

impl<R: fmt::Debug + Copy, F: fmt::Debug + Copy> fmt::Debug for Lifetime<R, F> {
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

/// regalloc.py FixedRegisterPositions
pub struct FixedRegisterPositions<R = ()> {
    pub register: R,
    pub index_lifetimes: Vec<(i32, i32)>, // (opindex, definition_pos)
}

impl<R> FixedRegisterPositions<R> {
    pub fn new(register: R) -> Self {
        FixedRegisterPositions {
            register,
            index_lifetimes: Vec::new(),
        }
    }

    /// regalloc.py FixedRegisterPositions.fixed_register
    pub fn fixed_register(&mut self, opindex: i32, definition_pos: i32) {
        if let Some(last) = self.index_lifetimes.last() {
            debug_assert!(opindex > last.0);
        }
        self.index_lifetimes.push((opindex, definition_pos));
    }

    /// regalloc.py FixedRegisterPositions.free_until_pos
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
pub fn get_lifetime<R: 'static, F: 'static>(box_: &Operand) -> Option<Rc<Lifetime<R, F>>> {
    let info = box_.get_lifetime_info()?;
    Rc::downcast(info).ok()
}

/// regalloc.py `LifetimeManager` — owns `fixed_register_use` only.
/// Per-box Lifetime lives on the box (`LifetimeManager.__setitem__` →
/// `box.set_forwarded(lt)`; `__getitem__` → `get_lifetime`).
///
/// `F` is carried so `get`/`contains`/`set` downcast the same
/// `Lifetime<R, F>` the boxes hold. `fixed_register_use` is keyed by `R` only.
pub struct LifetimeManager<R = (), F = ()> {
    /// `LifetimeManager.fixed_register_use` — register → FixedRegisterPositions
    pub fixed_register_use: IndexMap<R, FixedRegisterPositions<R>, FxBuildHasher>,
    _frame: PhantomData<F>,
}

impl<R, F> Default for LifetimeManager<R, F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R, F> LifetimeManager<R, F> {
    pub fn new() -> Self {
        LifetimeManager {
            fixed_register_use: IndexMap::with_hasher(FxBuildHasher),
            _frame: PhantomData,
        }
    }
}

impl<R: Copy + Eq + Hash + 'static, F: Copy + 'static> LifetimeManager<R, F> {
    pub fn contains(&self, v: &Operand) -> bool {
        get_lifetime::<R, F>(v).is_some()
    }

    pub fn get(&self, v: &Operand) -> Option<Rc<Lifetime<R, F>>> {
        get_lifetime(v)
    }

    pub fn set(&mut self, v: &Operand, lifetime: Lifetime<R, F>) -> Rc<Lifetime<R, F>> {
        debug_assert!(
            get_lifetime::<R, F>(v).is_none() || v.is_temp_var(),
            "LifetimeManager.__setitem__: box already has a Lifetime"
        );
        let rc = Rc::new(lifetime);
        v.set_lifetime_info(rc.clone());
        rc
    }

    /// `LifetimeManager.fixed_register`
    pub fn fixed_register(&mut self, opindex: i32, register: R, var: Option<&Operand>) {
        let definition_pos = match var {
            None => opindex,
            Some(v) => {
                let lt = get_lifetime::<R, F>(v).unwrap();
                lt.fixed_register(opindex, register)
            }
        };
        self.fixed_register_use
            .entry_or_insert_with(register, || FixedRegisterPositions::new(register))
            .fixed_register(opindex, definition_pos);
    }

    /// `LifetimeManager.try_use_same_register`
    pub fn try_use_same_register(&mut self, v0: &Operand, v1: &Operand) {
        let lt0 = get_lifetime::<R, F>(v0).unwrap();
        let lt1 = get_lifetime::<R, F>(v1).unwrap();
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

    /// regalloc.py LifetimeManager.longest_free_reg
    pub fn longest_free_reg(&self, position: i32, free_regs: &[R]) -> (Option<R>, i32) {
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

    /// regalloc.py LifetimeManager.free_reg_whole_lifetime
    pub fn free_reg_whole_lifetime(
        &self,
        position: i32,
        v: &Operand,
        free_regs: &[R],
    ) -> Option<R> {
        let longevity_var = get_lifetime::<R, F>(v).unwrap();
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

    /// regalloc.py LifetimeManager.try_pick_free_reg
    pub fn try_pick_free_reg(&self, position: i32, v: &Operand, free_regs: &[R]) -> Option<R> {
        if free_regs.is_empty() {
            return None;
        }
        let longevity_var = get_lifetime::<R, F>(v).unwrap();
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
pub fn compute_vars_longevity<R: Copy + Eq + Hash + 'static, F: Copy + 'static>(
    inputargs: &[InputArgRc],
    operations: &[OpRc],
) -> LifetimeManager<R, F> {
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

    // regalloc.py compute_vars_longevity: iterate operations in REVERSE
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

        // regalloc.py compute_vars_longevity guard failargs
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

    // regalloc.py compute_vars_longevity input arguments
    for iarg in inputargs {
        let box_ = Operand::from_bound_inputarg(iarg);
        if longevity.get(&box_).is_none() {
            longevity.set(&box_, Lifetime::new(-1, -1));
        }
    }

    // `compute_vars_longevity` reverse real_usages and check invariants
    let reverse_usages = |box_: &Operand| {
        if let Some(lifetime) = get_lifetime::<R, F>(box_) {
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

// ── FrameLayout / FrameManager ─────────────────────────────────────

/// Arch `FrameManager` subclass hooks: `frame_pos` / `frame_size` /
/// `get_loc_index` (`X86FrameManager` / `ARMFrameManager`).
pub trait FrameLayout {
    type Loc: Copy;
    fn frame_pos(position: usize, tp: Type, base_ofs: i32) -> Self::Loc;
    fn frame_size(tp: Type) -> usize;
    fn get_loc_index(loc: &Self::Loc) -> usize;
}

/// regalloc.py FrameManager — frame slot allocation.
/// Arch `frame_pos` / `frame_size` / `get_loc_index` via [`FrameLayout`].
pub struct FrameManager<L: FrameLayout> {
    /// regalloc.py FrameManager.current_frame_depth
    pub current_frame_depth: usize,
    /// regalloc.py FrameManager.boxes_in_frame
    pub boxes_in_frame: Vec<Option<Operand>>,
    /// x86/regalloc.py X86FrameManager.base_ofs (byte offset of frame field 0 from ebp)
    pub base_ofs: i32,
    _layout: PhantomData<L>,
}

impl<L: FrameLayout> FrameManager<L> {
    /// `FrameManager.__init__` plus the arch subclass `__init__(base_ofs)`.
    pub fn new(base_ofs: i32) -> Self {
        FrameManager {
            current_frame_depth: 0,
            boxes_in_frame: Vec::new(),
            base_ofs,
            _layout: PhantomData,
        }
    }

    /// regalloc.py FrameManager.get_frame_depth
    pub fn get_frame_depth(&self) -> usize {
        self.current_frame_depth
    }

    /// regalloc.py FrameManager._increase_frame_depth
    fn _increase_frame_depth(&mut self, incby: usize) {
        self.current_frame_depth += incby;
        for _ in 0..incby {
            self.boxes_in_frame.push(None);
        }
    }

    /// regalloc.py FrameManager.get
    pub fn get<R: Copy + Eq + Hash + 'static>(
        &self,
        v: &Operand,
        longevity: &LifetimeManager<R, L::Loc>,
    ) -> Option<L::Loc>
    where
        L::Loc: Copy + 'static,
    {
        longevity.get(v).and_then(|lt| lt.current_frame_loc.get())
    }

    /// regalloc.py FrameManager.loc
    pub fn loc<R: Copy + Eq + Hash + 'static>(
        &mut self,
        v: &Operand,
        tp: Type,
        must_exist: bool,
        longevity: &mut LifetimeManager<R, L::Loc>,
    ) -> L::Loc
    where
        L::Loc: Copy + 'static,
    {
        if let Some(loc) = self.get(v, longevity) {
            return loc;
        }
        if must_exist {
            panic!("FrameManager.loc: box {:?} not found", v);
        }
        self.get_new_loc(v, tp, longevity)
    }

    /// regalloc.py FrameManager.get_new_loc
    pub fn get_new_loc<R: Copy + Eq + Hash + 'static>(
        &mut self,
        v: &Operand,
        tp: Type,
        longevity: &mut LifetimeManager<R, L::Loc>,
    ) -> L::Loc
    where
        L::Loc: Copy + 'static,
    {
        let size = L::frame_size(tp);
        let hint = self.get_frame_pos_hint(v, longevity);
        let newloc = self._find_frame_location(size, tp, hint);
        let newloc = match newloc {
            Some(loc) => loc,
            None => {
                // regalloc.py FrameManager.get_new_loc odd size-2 index
                let mut index = self.get_frame_depth();
                if size == 2 && index & 1 == 1 {
                    if index > 0 && self.boxes_in_frame[index - 1].is_none() {
                        index -= 1;
                    } else {
                        index += 1;
                    }
                    let loc = L::frame_pos(index, tp, self.base_ofs);
                    let needed = index + size;
                    if needed > self.current_frame_depth {
                        self._increase_frame_depth(needed - self.current_frame_depth);
                    }
                    loc
                } else {
                    let loc = L::frame_pos(index, tp, self.base_ofs);
                    self._increase_frame_depth(size);
                    loc
                }
            }
        };
        self.bind(v, newloc, tp, longevity);
        newloc
    }

    /// regalloc.py FrameManager._find_frame_location
    fn _find_frame_location(&self, size: usize, tp: Type, hint: i32) -> Option<L::Loc> {
        debug_assert!(size == 1 || size == 2);
        if size == 1 {
            // regalloc.py FrameManager._find_frame_location — check 0 <= hint < depth
            if 0 <= hint
                && (hint as usize) < self.current_frame_depth
                && self.boxes_in_frame[hint as usize].is_none()
            {
                return Some(L::frame_pos(hint as usize, tp, self.base_ofs));
            }
            for i in 0..self.current_frame_depth {
                if self.boxes_in_frame[i].is_none() {
                    return Some(L::frame_pos(i, tp, self.base_ofs));
                }
            }
            None
        } else {
            let limit = (self.current_frame_depth >> 1) << 1;
            let mut i = 0;
            while i < limit {
                if self.boxes_in_frame[i].is_none() && self.boxes_in_frame[i + 1].is_none() {
                    return Some(L::frame_pos(i, tp, self.base_ofs));
                }
                i += 2;
            }
            None
        }
    }

    /// regalloc.py FrameManager.bind
    pub fn bind<R: Copy + Eq + Hash + 'static>(
        &mut self,
        v: &Operand,
        loc: L::Loc,
        tp: Type,
        longevity: &mut LifetimeManager<R, L::Loc>,
    ) where
        L::Loc: Copy + 'static,
    {
        let pos = L::get_loc_index(&loc);
        let size = L::frame_size(tp);
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

    /// regalloc.py FrameManager.finish_binding
    pub fn finish_binding(&self) {
        #[cfg(debug_assertions)]
        self._check_invariants();
    }

    /// regalloc.py FrameManager.mark_as_free
    pub fn mark_as_free<R: Copy + Eq + Hash + 'static>(
        &mut self,
        v: &Operand,
        tp: Type,
        longevity: &mut LifetimeManager<R, L::Loc>,
    ) where
        L::Loc: Copy + 'static,
    {
        let loc = match self.get(v, longevity) {
            Some(loc) => loc,
            None => return, // not in frame
        };
        let lifetime = longevity.get(v).unwrap();
        debug_assert!(lifetime.current_frame_loc.get().is_some());
        lifetime.current_frame_loc.set(None);
        let pos = L::get_loc_index(&loc);
        let size = L::frame_size(tp);
        debug_assert!(self.boxes_in_frame[pos].as_ref() == Some(v));
        for index in pos..pos + size {
            self.boxes_in_frame[index] = None;
        }
    }

    /// regalloc.py FrameManager.add_frame_pos_hint
    pub fn add_frame_pos_hint<R: Copy + Eq + Hash + 'static>(
        &self,
        v: &Operand,
        loc: L::Loc,
        longevity: &mut LifetimeManager<R, L::Loc>,
    ) where
        L::Loc: Copy + 'static,
    {
        if let Some(lifetime) = longevity.get(v) {
            lifetime.hint_frame_pos.set(L::get_loc_index(&loc) as i32);
        }
    }

    /// `FrameManager.get_frame_pos_hint`
    pub fn get_frame_pos_hint<R: Copy + Eq + Hash + 'static>(
        &self,
        v: &Operand,
        longevity: &LifetimeManager<R, L::Loc>,
    ) -> i32
    where
        L::Loc: Copy + 'static,
    {
        longevity
            .get(v)
            .map(|lt| lt.hint_frame_pos.get())
            .unwrap_or(-1)
    }

    /// regalloc.py FrameManager._check_invariants
    fn _check_invariants(&self) {
        debug_assert_eq!(self.boxes_in_frame.len(), self.current_frame_depth);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::operand::Operand;
    use majit_ir::{InputArg, InputArgRc, Op, OpCode, OpRc, OpRef, Type};

    fn ia_pair(tp: Type, index: u32) -> (InputArgRc, Operand) {
        let ia = InputArg::from_type_rc(tp, index);
        let box_ = Operand::from_bound_inputarg(&ia);
        (ia, box_)
    }

    #[derive(Clone, Copy, Debug)]
    struct TestFrameLoc {
        position: usize,
    }

    struct TestFrameLayout;

    impl FrameLayout for TestFrameLayout {
        type Loc = TestFrameLoc;
        fn frame_pos(position: usize, tp: Type, _base_ofs: i32) -> Self::Loc {
            let _ = tp;
            TestFrameLoc { position }
        }
        fn frame_size(_tp: Type) -> usize {
            1
        }
        fn get_loc_index(loc: &Self::Loc) -> usize {
            loc.position
        }
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

        let longevity: LifetimeManager = compute_vars_longevity(&inputargs, &ops);

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
        let longevity: LifetimeManager = compute_vars_longevity(&inputargs, &ops);
        assert_eq!(
            longevity.get(&vm).unwrap().last_usage.get(),
            1,
            "compute_vars_longevity: last_usage is the last failarg/arg use"
        );
    }

    #[test]
    fn test_lifetime_next_real_usage() {
        let lt: Lifetime = Lifetime::new(0, 10);
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
        let mut longevity = LifetimeManager::<(), TestFrameLoc>::new();
        let (_ia0, v0) = ia_pair(Type::Int, 0);
        let (_ia1, v1) = ia_pair(Type::Int, 1);
        let (_ia2, v2) = ia_pair(Type::Int, 2);
        longevity.set(&v0, Lifetime::new(0, 10));
        longevity.set(&v1, Lifetime::new(1, 10));
        longevity.set(&v2, Lifetime::new(2, 10));

        let mut fm = FrameManager::<TestFrameLayout>::new(0);
        let loc0 = fm.get_new_loc(&v0, Type::Int, &mut longevity);
        let loc1 = fm.get_new_loc(&v1, Type::Int, &mut longevity);
        let loc2 = fm.get_new_loc(&v2, Type::Int, &mut longevity);

        assert_eq!(loc0.position, 0);
        assert_eq!(loc1.position, 1);
        assert_eq!(loc2.position, 2);
        assert_eq!(fm.get_frame_depth(), 3);
    }

    #[test]
    fn jump_args_count_in_last_usage_not_real_usages() {
        let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
        let i0 = Operand::from_bound_inputarg(&inputargs[0]);
        let jump = Op::new(OpCode::Jump, &[i0.clone()]);
        jump.pos().set(OpRef::int_op(0));
        let ops = vec![OpRc::new(jump)];
        let longevity: LifetimeManager = compute_vars_longevity(&inputargs, &ops);
        let lt = longevity.get(&i0).unwrap();
        assert_eq!(lt.last_usage.get(), 0);
        assert!(lt.real_usages.borrow().is_none());
    }

    #[test]
    fn unused_inputarg_last_usage_is_negative() {
        let inputargs = vec![
            InputArg::from_type_rc(Type::Int, 0),
            InputArg::from_type_rc(Type::Int, 1),
        ];
        let i0 = Operand::from_bound_inputarg(&inputargs[0]);
        let i1 = Operand::from_bound_inputarg(&inputargs[1]);
        let finish = Op::new(OpCode::Finish, &[i0.clone()]);
        finish.pos().set(OpRef::int_op(0));
        let ops = vec![OpRc::new(finish)];
        let longevity: LifetimeManager = compute_vars_longevity(&inputargs, &ops);
        let lt = longevity.get(&i1).unwrap();
        assert_eq!(lt.last_usage.get(), -1);
        assert_eq!(lt.definition_pos.get(), -1);
    }
}
