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

pub use majit_ir::lifetime::{Lifetime, UNDEF_POS};

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
pub fn get_lifetime(box_: &Operand) -> Option<Rc<Lifetime>> {
    box_.get_lifetime_info()
}

/// regalloc.py `LifetimeManager` — owns `fixed_register_use` only.
/// Per-box Lifetime lives on the box (`LifetimeManager.__setitem__` →
/// `box.set_forwarded(lt)`; `__getitem__` → `get_lifetime`).
///
/// `fixed_register_use` is keyed by the concrete register `R`.
pub struct LifetimeManager<R = ()> {
    /// `LifetimeManager.fixed_register_use` — register → FixedRegisterPositions
    pub fixed_register_use: IndexMap<R, FixedRegisterPositions<R>, FxBuildHasher>,
}

impl<R> Default for LifetimeManager<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R> LifetimeManager<R> {
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
}

impl<R: Copy + Eq + Hash> LifetimeManager<R> {
    fn all_regs_index(all_regs: &[R], register: R) -> i32 {
        all_regs
            .iter()
            .position(|&r| r == register)
            .expect("fixed_register: register not in all_regs") as i32
    }

    /// `LifetimeManager.fixed_register`
    ///
    /// `all_regs` is the `RegisterManager.all_regs` that `register` belongs
    /// to; the box's Lifetime stores that list's index.
    pub fn fixed_register(
        &mut self,
        opindex: i32,
        register: R,
        var: Option<&Operand>,
        all_regs: &[R],
    ) {
        let definition_pos = match var {
            None => opindex,
            Some(v) => {
                let lt = get_lifetime(v).unwrap();
                lt.fixed_register(opindex, Self::all_regs_index(all_regs, register))
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

    /// regalloc.py LifetimeManager.try_pick_free_reg
    ///
    /// `all_regs` is the same `RegisterManager.all_regs` whose indices
    /// `Lifetime.find_fixed_register` stores.
    pub fn try_pick_free_reg(
        &self,
        position: i32,
        v: &Operand,
        free_regs: &[R],
        all_regs: &[R],
    ) -> Option<R> {
        if free_regs.is_empty() {
            return None;
        }
        let longevity_var = get_lifetime(v).unwrap();
        // check whether there is a fixed register and whether it's free
        if let Some(idx) = longevity_var.find_fixed_register(position) {
            // `regalloc.py Lifetime.find_fixed_register` stores the register's
            // index in `all_regs`; the non-generic `Lifetime` in `majit-ir`
            // cannot name the backend register type.
            let reg = all_regs[idx as usize];
            if free_regs.contains(&reg) {
                return Some(reg);
            }
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
pub fn compute_vars_longevity<R>(
    inputargs: &[InputArgRc],
    operations: &[OpRc],
) -> LifetimeManager<R> {
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

/// `BindingsIterItems` — lazy walk of occupied frame slots.
///
/// `Lifetime.current_frame_loc` is a frame position; the iterator rebuilds
/// `L::Loc` via [`FrameLayout::frame_pos`].
pub struct BindingsIterItems<'a, L: FrameLayout, R> {
    fm: &'a FrameManager<L>,
    longevity: &'a LifetimeManager<R>,
    index: usize,
}

impl<'a, L: FrameLayout, R> Iterator for BindingsIterItems<'a, L, R> {
    type Item = (Operand, L::Loc);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let index = self.index;
            if index >= self.fm.current_frame_depth {
                return None;
            }
            let Some(box_) = self.fm.boxes_in_frame[index].clone() else {
                self.index = index + 1;
                continue;
            };
            self.index = index + L::frame_size(box_.type_());
            let lifetime = self.longevity.get(&box_).expect("lifetime is not None");
            let pos = lifetime.current_frame_loc.get().expect("loc is not None");
            debug_assert_eq!(pos as usize, index);
            let loc = L::frame_pos(pos as usize, box_.type_(), self.fm.base_ofs);
            return Some((box_, loc));
        }
    }
}

impl<L: FrameLayout> FrameManager<L> {
    /// `FrameManager.__init__` plus the arch subclass `__init__(base_ofs)`.
    pub fn new(base_ofs: i32) -> Self {
        Self::with_start_free_depth(base_ofs, 0)
    }

    /// `FrameManager.__init__(start_free_depth)` plus the arch `base_ofs`.
    /// `_find_frame_location` still searches from index 0.
    pub fn with_start_free_depth(base_ofs: i32, start_free_depth: usize) -> Self {
        FrameManager {
            current_frame_depth: start_free_depth,
            boxes_in_frame: vec![None; start_free_depth],
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

    /// Rebuild `L::Loc` from a stored frame position.
    fn loc_from_pos(&self, v: &Operand, pos: i32) -> L::Loc {
        L::frame_pos(pos as usize, v.type_(), self.base_ofs)
    }

    /// regalloc.py FrameManager.get
    pub fn get<R>(&self, v: &Operand, longevity: &LifetimeManager<R>) -> Option<L::Loc> {
        longevity
            .get(v)
            .and_then(|lt| lt.current_frame_loc.get())
            .map(|pos| self.loc_from_pos(v, pos))
    }

    /// regalloc.py FrameManager.loc
    pub fn loc<R>(
        &mut self,
        v: &Operand,
        tp: Type,
        must_exist: bool,
        longevity: &mut LifetimeManager<R>,
    ) -> L::Loc {
        if let Some(loc) = self.get(v, longevity) {
            return loc;
        }
        if must_exist {
            panic!("FrameManager.loc: box {:?} not found", v);
        }
        self.get_new_loc(v, tp, longevity)
    }

    /// regalloc.py FrameManager.get_new_loc
    pub fn get_new_loc<R>(
        &mut self,
        v: &Operand,
        tp: Type,
        longevity: &mut LifetimeManager<R>,
    ) -> L::Loc {
        let size = L::frame_size(tp);
        let hint = self.get_frame_pos_hint(v, longevity);
        let newloc = self._find_frame_location(size, tp, hint);
        let newloc = match newloc {
            Some(loc) => loc,
            None => {
                // FrameManager.get_new_loc odd size-2 index
                let mut index = self.get_frame_depth();
                let loc = if size == 2 && index & 1 == 1 {
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
                };
                #[cfg(debug_assertions)]
                {
                    let testindex = L::get_loc_index(&loc);
                    debug_assert_eq!(testindex, index);
                    for i in testindex..testindex + size {
                        debug_assert!(self.boxes_in_frame[i].is_none());
                    }
                }
                loc
            }
        };
        debug_assert!(self.boxes_in_frame[L::get_loc_index(&newloc)].is_none());
        self.bind(v, newloc, tp, longevity);
        #[cfg(debug_assertions)]
        self._check_invariants(longevity);
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
    pub fn bind<R>(
        &mut self,
        v: &Operand,
        loc: L::Loc,
        tp: Type,
        longevity: &mut LifetimeManager<R>,
    ) {
        let pos = L::get_loc_index(&loc);
        let size = L::frame_size(tp);
        if pos + size > self.current_frame_depth {
            self._increase_frame_depth(pos + size - self.current_frame_depth);
        }
        debug_assert!(self.boxes_in_frame[pos].is_none());
        debug_assert!(
            !self.boxes_in_frame.iter().any(|b| b.as_ref() == Some(v)),
            "FrameManager.bind: box already in frame"
        );
        for index in pos..pos + size {
            self.boxes_in_frame[index] = Some(v.clone());
        }
        let lifetime = longevity.get(v).expect("bind: variable not in longevity");
        lifetime.current_frame_loc.set(Some(pos as i32));
    }

    /// `FrameManager.bindings_iteritems`
    pub fn bindings_iteritems<'a, R>(
        &'a self,
        longevity: &'a LifetimeManager<R>,
    ) -> BindingsIterItems<'a, L, R> {
        BindingsIterItems {
            fm: self,
            longevity,
            index: 0,
        }
    }

    /// `FrameManager.finish_binding`
    pub fn finish_binding<R>(&self, longevity: &LifetimeManager<R>) {
        #[cfg(debug_assertions)]
        self._check_invariants(longevity);
        #[cfg(not(debug_assertions))]
        let _ = longevity;
    }

    /// regalloc.py FrameManager.mark_as_free
    pub fn mark_as_free<R>(&mut self, v: &Operand, tp: Type, longevity: &mut LifetimeManager<R>) {
        let loc = match self.get(v, longevity) {
            Some(loc) => loc,
            None => return, // not in frame
        };
        let lifetime = longevity.get(v).unwrap();
        debug_assert_eq!(
            lifetime.current_frame_loc.get(),
            Some(L::get_loc_index(&loc) as i32),
            "FrameManager.mark_as_free: current_frame_loc is loc"
        );
        lifetime.current_frame_loc.set(None);
        let pos = L::get_loc_index(&loc);
        let size = L::frame_size(tp);
        debug_assert!(self.boxes_in_frame[pos].as_ref() == Some(v));
        for index in pos..pos + size {
            self.boxes_in_frame[index] = None;
        }
        #[cfg(debug_assertions)]
        self._check_invariants(longevity);
    }

    /// regalloc.py FrameManager.add_frame_pos_hint
    pub fn add_frame_pos_hint<R>(
        &self,
        v: &Operand,
        loc: L::Loc,
        longevity: &mut LifetimeManager<R>,
    ) {
        let lifetime = longevity.get(v);
        debug_assert!(
            lifetime.is_some(),
            "add_frame_pos_hint: box has no Lifetime"
        );
        if let Some(lifetime) = lifetime {
            lifetime.hint_frame_pos.set(L::get_loc_index(&loc) as i32);
        }
    }

    /// `FrameManager.get_frame_pos_hint`
    pub fn get_frame_pos_hint<R>(&self, v: &Operand, longevity: &LifetimeManager<R>) -> i32 {
        longevity
            .get(v)
            .map(|lt| lt.hint_frame_pos.get())
            .unwrap_or(-1)
    }

    /// `FrameManager._check_invariants`
    #[cfg(debug_assertions)]
    fn _check_invariants<R>(&self, longevity: &LifetimeManager<R>) {
        debug_assert_eq!(self.boxes_in_frame.len(), self.current_frame_depth);
        let mut all = vec![false; self.get_frame_depth()];
        for (b, loc) in self.bindings_iteritems(longevity) {
            let size = L::frame_size(b.type_());
            let pos = L::get_loc_index(&loc);
            for i in pos..pos + size {
                debug_assert!(!all[i]);
                all[i] = true;
                debug_assert!(self.boxes_in_frame[i].as_ref() == Some(&b));
            }
        }
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
        let mut longevity = LifetimeManager::<()>::new();
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

    #[test]
    fn with_start_free_depth_preallocates_free_slots_and_allocates_from_zero() {
        let mut longevity = LifetimeManager::<()>::new();
        let (_ia, v) = ia_pair(Type::Int, 0);
        longevity.set(&v, Lifetime::new(0, 10));

        let mut fm = FrameManager::<TestFrameLayout>::with_start_free_depth(0, 3);
        assert_eq!(fm.get_frame_depth(), 3);
        assert_eq!(fm.boxes_in_frame.len(), 3);
        assert!(fm.boxes_in_frame.iter().all(|slot| slot.is_none()));

        let loc = fm.get_new_loc(&v, Type::Int, &mut longevity);
        assert_eq!(loc.position, 0);
        assert_eq!(TestFrameLayout::get_loc_index(&loc), 0);
    }

    #[test]
    fn bindings_iteritems_yields_each_bound_box_once_with_its_index() {
        let mut longevity = LifetimeManager::<()>::new();
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

        let items: Vec<_> = fm.bindings_iteritems(&longevity).collect();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].0, v0);
        assert_eq!(TestFrameLayout::get_loc_index(&items[0].1), loc0.position);
        assert_eq!(items[1].0, v1);
        assert_eq!(TestFrameLayout::get_loc_index(&items[1].1), loc1.position);
        assert_eq!(items[2].0, v2);
        assert_eq!(TestFrameLayout::get_loc_index(&items[2].1), loc2.position);
        fm.finish_binding(&longevity);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "FrameManager.bind: box already in frame")]
    fn bind_same_box_twice_panics_in_debug() {
        let mut longevity = LifetimeManager::<()>::new();
        let (_ia0, v0) = ia_pair(Type::Int, 0);
        longevity.set(&v0, Lifetime::new(0, 10));

        let mut fm = FrameManager::<TestFrameLayout>::new(0);
        let _loc0 = fm.get_new_loc(&v0, Type::Int, &mut longevity);
        let loc1 = TestFrameLayout::frame_pos(1, Type::Int, 0);
        fm.bind(&v0, loc1, Type::Int, &mut longevity);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "add_frame_pos_hint: box has no Lifetime")]
    fn add_frame_pos_hint_without_lifetime_panics_in_debug() {
        let mut longevity = LifetimeManager::<()>::new();
        let (_ia0, v0) = ia_pair(Type::Int, 0);
        let fm = FrameManager::<TestFrameLayout>::new(0);
        let loc = TestFrameLayout::frame_pos(0, Type::Int, 0);
        fm.add_frame_pos_hint(&v0, loc, &mut longevity);
    }
}
